r"""Distribution of the summed score given the latent trait (DESIGN.md §18.1).

Given :math:`\theta`, item scores :math:`X_j\in\{0,\dots,K_j-1\}` are independent with category
probabilities :math:`P_{jk}(\theta)`. The summed score :math:`S=\sum_jX_j` then follows the
Poisson-binomial distribution for dichotomous items and its generalization to sums of
independent categorical scores for polytomous items. The Lord–Wingersky (1984) recursion is
the direct convolution algorithm for it.

Everything here is a pure JAX kernel: ``jit``, ``vmap`` and autodiff work through every method.
"""

from __future__ import annotations

from typing import Literal, get_args

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from flexpyirt.errors import SpecificationError

__all__ = ["SummedScoreMethod", "summed_score_distribution"]

SummedScoreMethod = Literal["recursion", "fft", "normal", "refined_normal", "poisson"]
"""Algorithms of :func:`summed_score_distribution`."""

_APPROXIMATIONS = ("normal", "refined_normal", "poisson")


def _shift(values: Array, k: int, fill: float) -> Array:
    """Shift ``values`` by ``k`` places along the last axis, filling the front with ``fill``."""
    if k == 0:
        return values
    pad = jnp.full((*values.shape[:-1], k), fill, dtype=values.dtype)
    return jnp.concatenate([pad, values[..., :-k]], axis=-1)


def _recursion(probs: Array) -> Array:
    r"""Run the Lord–Wingersky recursion on a fixed-length vector.

    .. math::
        L_j(s)=\sum_{k=0}^{K-1}L_{j-1}(s-k)\,P_{jk},\qquad L_0(0)=1

    (Lord & Wingersky, 1984; Thissen et al., 1995). ``probs`` is ``(..., J, K)``; the result is
    ``(..., J (K - 1) + 1)``.
    """
    n_items, n_categories = probs.shape[-2], probs.shape[-1]
    length = n_items * (n_categories - 1) + 1
    start = jnp.zeros((*probs.shape[:-2], length), dtype=probs.dtype).at[..., 0].set(1.0)

    def add_item(lik: Array, p_item: Array) -> tuple[Array, None]:
        new = jnp.zeros_like(lik)
        for k in range(n_categories):
            new = new + _shift(lik, k, 0.0) * p_item[..., k : k + 1]
        return new, None

    out, _ = jax.lax.scan(add_item, start, jnp.moveaxis(probs, -2, 0))
    return out


def _log_recursion(probs: Array) -> Array:
    r"""Run the Lord–Wingersky recursion in log space.

    .. math::
        \log L_j(s)=\operatorname{logsumexp}_k\big[\log L_{j-1}(s-k)+\log P_{jk}\big]

    Impossible scores are :math:`-\infty`. The log-sum-exp is written with an explicit maximum
    and double ``where`` guards so that gradients stay finite where all terms are
    :math:`-\infty`.
    """
    n_items, n_categories = probs.shape[-2], probs.shape[-1]
    length = n_items * (n_categories - 1) + 1
    positive = probs > 0.0
    log_p = jnp.where(positive, jnp.log(jnp.where(positive, probs, 1.0)), -jnp.inf)
    start = jnp.full((*probs.shape[:-2], length), -jnp.inf, dtype=probs.dtype).at[..., 0].set(0.0)

    def add_item(log_lik: Array, log_p_item: Array) -> tuple[Array, None]:
        terms = jnp.stack(
            [_shift(log_lik, k, -jnp.inf) + log_p_item[..., k : k + 1] for k in range(n_categories)]
        )
        top = jnp.max(terms, axis=0)
        finite = jnp.isfinite(top)
        safe_top = jnp.where(finite, top, 0.0)
        safe_terms = jnp.where(finite[None], terms, 0.0)
        total = jnp.sum(jnp.exp(safe_terms - safe_top[None]), axis=0)
        return jnp.where(finite, safe_top + jnp.log(total), -jnp.inf), None

    out, _ = jax.lax.scan(add_item, start, jnp.moveaxis(log_p, -2, 0))
    return out


def _fft(probs: Array) -> Array:
    r"""Invert the characteristic function by a discrete Fourier transform (DFT-CF; Hong, 2013).

    .. math::
        \varphi(\omega_l)=\prod_j\sum_kP_{jk}e^{i\omega_lk},\quad
        \omega_l=\frac{2\pi l}{S_{\max}+1},\qquad
        P(S=s)=\frac1{S_{\max}+1}\sum_l\varphi(\omega_l)e^{-i\omega_ls}

    The error is absolute (about ``1e-16``); negative round-off is clipped at 0.
    """
    n_items, n_categories = probs.shape[-2], probs.shape[-1]
    length = n_items * (n_categories - 1) + 1
    omega = 2.0 * jnp.pi * jnp.arange(length, dtype=probs.dtype) / length
    # (L, K) e^{i omega_l k}; contract with (..., J, K) to the item characteristic functions.
    phase = jnp.exp(1j * omega[:, None] * jnp.arange(n_categories, dtype=probs.dtype)[None, :])
    item_cf = jnp.einsum("...jk,lk->...jl", probs.astype(phase.dtype), phase)
    cf = jnp.prod(item_cf, axis=-2)
    pmf = jnp.real(jnp.fft.fft(cf, axis=-1)) / length
    return jnp.maximum(pmf, 0.0)


def _moments(p: Array) -> tuple[Array, Array, Array]:
    r"""Return the mean, standard deviation and skewness of a Poisson-binomial variable.

    .. math::
        \mu=\sum_jp_j,\quad \sigma^2=\sum_jp_j(1-p_j),\quad
        \gamma=\sigma^{-3}\sum_jp_j(1-p_j)(1-2p_j)
    """
    mean = jnp.sum(p, axis=-1, keepdims=True)
    var = jnp.sum(p * (1.0 - p), axis=-1, keepdims=True)
    sd = jnp.sqrt(var)
    skew = jnp.sum(p * (1.0 - p) * (1.0 - 2.0 * p), axis=-1, keepdims=True) / (var * sd)
    return mean, sd, skew


def _normal(probs: Array, *, refined: bool) -> Array:
    r"""Return the normal or refined normal approximation for dichotomous items.

    .. math::
        P(S\le s)\approx G\Big(\frac{s+0.5-\mu}{\sigma}\Big),\qquad
        G(x)=\Phi(x)+\frac{\gamma(1-x^2)\phi(x)}{6}

    clipped to :math:`[0,1]` (Volkova, 1996; Hong, 2013); the normal approximation drops the
    :math:`\gamma` term. The probabilities are differences of consecutive values.
    """
    p = probs[..., 1]
    mean, sd, skew = _moments(p)
    scores = jnp.arange(p.shape[-1] + 1, dtype=probs.dtype)
    x = (scores + 0.5 - mean) / sd
    cdf = jax.scipy.stats.norm.cdf(x)
    if refined:
        cdf = cdf + skew * (1.0 - x**2) * jax.scipy.stats.norm.pdf(x) / 6.0
        cdf = jnp.clip(cdf, 0.0, 1.0)
    return jnp.diff(cdf, axis=-1, prepend=jnp.zeros_like(cdf[..., :1]))


def _poisson(probs: Array) -> Array:
    r"""Return the Poisson approximation for dichotomous items (Le Cam, 1960).

    .. math:: P(S=s)\approx e^{-\mu}\mu^s/s!,\qquad \mu=\sum_jp_j
    """
    p = probs[..., 1]
    mean = jnp.sum(p, axis=-1, keepdims=True)
    scores = jnp.arange(p.shape[-1] + 1, dtype=probs.dtype)
    return jax.scipy.stats.poisson.pmf(scores, mean)


def summed_score_distribution(
    probs: ArrayLike,
    *,
    method: SummedScoreMethod = "recursion",
    log: bool = False,
) -> Array:
    r"""Return the distribution of the summed score of independent item scores.

    Item :math:`j` is scored :math:`0,\dots,K_j-1` with probabilities :math:`P_{jk}` (for
    example the category probabilities of an IRT model at one :math:`\theta`). The summed score
    :math:`S=\sum_jX_j` follows the Poisson-binomial distribution when every item is
    dichotomous, and its generalization to sums of independent categorical scores otherwise.
    The default method is the Lord–Wingersky recursion

    .. math::
        L_j(s)=\sum_{k=0}^{K_j-1}L_{j-1}(s-k)\,P_{jk},\qquad L_0(0)=1

    (Lord & Wingersky, 1984; Thissen, Pommerich, Billeaud & Williams, 1995).

    The function is a pure JAX kernel: it can be used inside ``jit`` and ``vmap`` and
    differentiated with respect to ``probs``. Values are not checked.

    Parameters
    ----------
    probs : array-like, shape (..., J, K)
        Category probabilities of ``J`` items. Each row sums to 1; items with fewer than ``K``
        categories are padded with zeros. Leading axes are batch axes (quadrature nodes,
        persons, ...).
    method : {"recursion", "fft", "normal", "refined_normal", "poisson"}, default "recursion"
        ``"recursion"`` is exact, with a small relative error in every probability, tails
        included. ``"fft"`` (the discrete Fourier transform of the characteristic function;
        Hong, 2013) is exact up to an absolute error of about ``1e-16``, so probabilities below
        about ``1e-15`` are not reliable. ``"normal"`` (with continuity correction),
        ``"refined_normal"`` (Volkova, 1996) and ``"poisson"`` (Le Cam, 1960) are
        approximations for dichotomous items (``K = 2``) only; the two normal approximations
        need not sum to exactly 1.
    log : bool, default False
        Return log probabilities. With ``method="recursion"`` the recursion itself runs in log
        space, which keeps tail probabilities that would underflow to 0; for the other methods
        the logarithm of the result is taken.

    Returns
    -------
    Array, shape (..., J * (K - 1) + 1)
        Probabilities (or log probabilities) of the summed scores ``0 .. J * (K - 1)``. Scores
        above the largest possible score of the items have probability 0.

    Raises
    ------
    SpecificationError
        If ``probs`` has fewer than two axes, no item or fewer than two categories, ``method``
        is unknown, or an approximation is requested with ``K != 2``.
    """
    if method not in get_args(SummedScoreMethod):
        raise SpecificationError(
            f"Unknown method={method!r}; use one of {list(get_args(SummedScoreMethod))}."
        )
    p = jnp.asarray(probs)
    if not jnp.issubdtype(p.dtype, jnp.floating):
        p = p.astype(jnp.result_type(float))
    if p.ndim < 2 or p.shape[-2] < 1 or p.shape[-1] < 2:
        raise SpecificationError(
            "probs must have shape (..., J, K) with J >= 1 items and K >= 2 categories, got "
            f"{p.shape}."
        )
    if method in _APPROXIMATIONS and p.shape[-1] != 2:
        raise SpecificationError(
            f"method={method!r} is available for dichotomous items only (K = 2), got "
            f"K = {p.shape[-1]}; use 'recursion' or 'fft'."
        )
    if method == "recursion":
        return _log_recursion(p) if log else _recursion(p)
    if method == "fft":
        pmf = _fft(p)
    elif method == "poisson":
        pmf = _poisson(p)
    else:
        pmf = _normal(p, refined=method == "refined_normal")
    if not log:
        return pmf
    positive = pmf > 0.0
    return jnp.where(positive, jnp.log(jnp.where(positive, pmf, 1.0)), -jnp.inf)
