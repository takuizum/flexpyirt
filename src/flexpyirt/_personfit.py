r"""Person fit and residual-based item fit (Core layer, DESIGN.md §20).

Pure JAX kernels on :class:`~flexpyirt._core.CompiledProblem` pytrees, evaluated per unique
response pattern at that pattern's ability estimate :math:`\hat\theta_p`: the standardized
log-likelihood :math:`l_z` (Drasgow, Levine & Williams, 1985), its correction for an estimated
ability :math:`l_z^*` (Snijders, 2001; Sinharay, 2016), and the per-item residual moments
(:math:`r_{pj}^2`, :math:`W_{pj}`, :math:`C_{pj}`) from which the person and item infit / outfit
mean squares and their Wilson–Hilferty :math:`z` values are formed in NumPy
(Wright & Masters, 1982). Derivatives in :math:`\theta` come from JAX autodiff. Unidimensional
models only.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from flexpyirt._core import CompiledProblem
from flexpyirt._scoring import _expand_items, _expand_latent, _pattern_info
from flexpyirt.items import ItemKernel

__all__ = [
    "CORRECTION_MAP",
    "CORRECTION_ML",
    "CORRECTION_NONE",
    "CORRECTION_WLE",
    "Q_CAP",
    "W_FLOOR",
    "mean_square",
    "pattern_statistics",
]

CORRECTION_NONE = 0
"""No estimating-equation term (EAP: :math:`l_z^*` is not defined and is returned as NaN)."""
CORRECTION_ML = 1
r""":math:`r_0(\theta)=0` (maximum likelihood)."""
CORRECTION_MAP = 2
r""":math:`r_0(\theta)=-(\theta-\mu_g)/\sigma_g^2` (normal prior of the respondent's group)."""
CORRECTION_WLE = 3
r""":math:`r_0(\theta)=I'(\theta)/(2I(\theta))` (Warm's weight, as in ``wle_scores``)."""

W_FLOOR = float(np.sqrt(1e-5))
r"""Floor on the residual variance :math:`W_{pj}`; mirt's ``personfit()`` sets
:math:`W_{pj}=\sqrt{10^{-5}}` whenever :math:`W_{pj}^2<10^{-5}` (a guard against near-zero
variances at extreme :math:`\hat\theta`)."""
Q_CAP = 1.4142
r"""Cap on the standard deviation :math:`q` of a mean square (Wright & Masters, 1982; mirt)."""


def _item_moments(t: Array, p: Array, k: ItemKernel) -> tuple[Array, ...]:
    """Category probabilities and their log-derivatives of one item at the scalar ``t``."""

    def lp_fn(theta: Array) -> Array:
        return k.log_prob(theta[None], p)[0]

    lp = lp_fn(t)
    dlp = jax.jacfwd(lp_fn)(t)
    return jnp.exp(lp), lp, dlp


def pattern_statistics(
    prob: CompiledProblem, eta: Array, theta: Array, correction: int
) -> dict[str, Array]:
    r"""Person-fit quantities of every unique pattern at its ability estimate.

    For pattern :math:`p` with answered items :math:`j\in A_p`, category probabilities
    :math:`P_{jk}=P_{jk}(\hat\theta_p)` and observed category :math:`y_{pj}`:

    .. math::
        l_0 = \sum_{j\in A_p}\log P_{j y_{pj}},\quad
        \mu = \sum_{j\in A_p}\sum_k P_{jk}\log P_{jk},\quad
        \sigma^2 = \sum_{j\in A_p}\Big[\sum_k P_{jk}(\log P_{jk})^2
            - \Big(\sum_k P_{jk}\log P_{jk}\Big)^2\Big],\quad
        l_z = \frac{l_0-\mu}{\sigma}

    (Drasgow, Levine & Williams, 1985). With the item score :math:`b_j=\partial\log P_{j
    y_{pj}}/\partial\theta` and the estimating equation :math:`\sum_j b_j + r_0(\hat\theta)=0`
    of the ability estimator,

    .. math::
        c = \frac{\sum_j\sum_k P'_{jk}\log P_{jk}}{\sum_j I_j},\qquad
        \tau^2 = \sum_j\big[\sigma_j^2 - 2c\sum_k P'_{jk}\log P_{jk} + c^2 I_j\big],\qquad
        l_z^* = \frac{l_0-\mu + c\,r_0(\hat\theta)}{\tau}

    (Snijders, 2001; the polytomous form of Sinharay, 2016), where
    :math:`I_j=\sum_k P'^2_{jk}/P_{jk}` is the item information and :math:`\sigma_j^2` the item's
    term of :math:`\sigma^2`. For the residual statistics, with category scores
    :math:`0,\dots,K_j-1`:

    .. math::
        E_{pj}=\sum_k kP_{jk},\quad r_{pj}=y_{pj}-E_{pj},\quad
        W_{pj}=\sum_k (k-E_{pj})^2P_{jk},\quad C_{pj}=\sum_k (k-E_{pj})^4P_{jk}.

    Parameters
    ----------
    prob : CompiledProblem
        Unidimensional problem whose ``y`` holds the (padded) unique patterns.
    eta : Array
        Free-parameter vector.
    theta : (P,) Array
        Ability estimate of each pattern; non-finite values give NaN statistics.
    correction : int
        One of the ``CORRECTION_*`` constants (the estimator's :math:`r_0`).

    Returns
    -------
    dict of Array
        ``lz``, ``lz_star`` and ``n_items`` of shape ``(P,)``; ``resid2``, ``w`` and ``c`` of
        shape ``(P, J)`` with NaN for unanswered items and :math:`W_{pj}` already floored at
        :data:`W_FLOOR`.
    """
    xi_list = _expand_items(prob, eta)
    latent = _expand_latent(prob, eta)
    n_items = prob.y.shape[1]

    def one(y_p: Array, g_p: Array, t_raw: Array) -> dict[str, Array]:
        finite = jnp.isfinite(t_raw)
        t = jnp.where(finite, t_raw, 0.0)
        ll = jnp.zeros((), t.dtype)
        mu = jnp.zeros((), t.dtype)
        var = jnp.zeros((), t.dtype)
        cov = jnp.zeros((), t.dtype)
        info = jnp.zeros((), t.dtype)
        resid2 = jnp.full((n_items,), jnp.nan, t.dtype)
        w_all = jnp.full((n_items,), jnp.nan, t.dtype)
        c_all = jnp.full((n_items,), jnp.nan, t.dtype)
        for fam, xi in zip(prob.families, xi_list, strict=True):
            y_f = y_p[fam.columns]
            obs = y_f >= 0

            def moments(p: Array, k: ItemKernel = fam.kernel) -> tuple[Array, ...]:
                return _item_moments(t, p, k)

            prob_k, lp, dlp = jax.vmap(moments)(xi[g_p])  # (J_f, K) each
            pos = prob_k > 0.0
            safe_lp = jnp.where(pos, lp, 0.0)
            plogp = jnp.where(pos, prob_k * lp, 0.0)
            mu_j = plogp.sum(axis=1)
            var_j = jnp.where(pos, prob_k * safe_lp**2, 0.0).sum(axis=1) - mu_j**2
            dprob = prob_k * dlp  # P'_jk
            cov_j = jnp.where(pos, dprob * safe_lp, 0.0).sum(axis=1)
            info_j = jnp.where(pos, prob_k * dlp**2, 0.0).sum(axis=1)
            y_safe = jnp.maximum(y_f, 0)
            picked = lp[jnp.arange(y_f.shape[0]), y_safe]
            ll = ll + jnp.sum(jnp.where(obs, picked, 0.0))
            mu = mu + jnp.sum(jnp.where(obs, mu_j, 0.0))
            var = var + jnp.sum(jnp.where(obs, var_j, 0.0))
            cov = cov + jnp.sum(jnp.where(obs, cov_j, 0.0))
            info = info + jnp.sum(jnp.where(obs, info_j, 0.0))

            score = jnp.arange(prob_k.shape[1], dtype=t.dtype)
            expected = prob_k @ score
            dev = score[None, :] - expected[:, None]
            w_j = (dev**2 * prob_k).sum(axis=1)
            w_j = jnp.where(w_j**2 < 1e-5, W_FLOOR, w_j)
            c_j = (dev**4 * prob_k).sum(axis=1)
            r2_j = (y_f - expected) ** 2
            resid2 = resid2.at[fam.columns].set(jnp.where(obs, r2_j, jnp.nan))
            w_all = w_all.at[fam.columns].set(jnp.where(obs, w_j, jnp.nan))
            c_all = c_all.at[fam.columns].set(jnp.where(obs, c_j, jnp.nan))

        n_obs = jnp.sum(y_p >= 0)
        ok = finite & (n_obs > 0)
        lz = jnp.where(
            ok & (var > 0.0), (ll - mu) / jnp.sqrt(jnp.where(var > 0.0, var, 1.0)), jnp.nan
        )

        if correction == CORRECTION_ML:
            r0 = jnp.zeros((), t.dtype)
        elif correction == CORRECTION_MAP:
            mean_g, var_g = latent[g_p, 0], latent[g_p, 1]
            r0 = -(t - mean_g) / var_g
        elif correction == CORRECTION_WLE:

            def test_info(s: Array) -> Array:
                return _pattern_info(s, y_p, g_p, prob.families, xi_list)

            r0 = jax.grad(test_info)(t) / (2.0 * test_info(t))
        else:
            r0 = jnp.full((), jnp.nan, t.dtype)
        c = cov / jnp.where(info > 0.0, info, 1.0)
        tau2 = var - 2.0 * c * cov + c**2 * info
        lz_star = jnp.where(
            ok & (info > 0.0) & (tau2 > 0.0),
            (ll - mu + c * r0) / jnp.sqrt(jnp.where(tau2 > 0.0, tau2, 1.0)),
            jnp.nan,
        )
        nan_row = jnp.full((n_items,), jnp.nan, t.dtype)
        return {
            "lz": lz,
            "lz_star": lz_star,
            "n_items": n_obs,
            "resid2": jnp.where(finite, resid2, nan_row),
            "w": jnp.where(finite, w_all, nan_row),
            "c": jnp.where(finite, c_all, nan_row),
        }

    return jax.vmap(one)(prob.y, prob.group, theta)


def mean_square(
    resid2: np.ndarray, w: np.ndarray, c: np.ndarray, *, axis: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Infit / outfit mean squares and their :math:`z` values along ``axis`` (NaN = skipped).

    Over the :math:`n` non-missing entries (items of a person for ``axis=1``, persons of an item
    for ``axis=0``):

    .. math::
        \mathrm{outfit} = \frac1n\sum\frac{r^2}{W},\quad
        q_{\mathrm{out}}^2 = \frac1{n^2}\sum\frac{C}{W^2}-\frac1n,\qquad
        \mathrm{infit} = \frac{\sum r^2}{\sum W},\quad
        q_{\mathrm{in}}^2 = \frac{\sum (C-W^2)}{(\sum W)^2},

    each :math:`q` capped at 1.4142, and the Wilson–Hilferty cube-root transformation
    :math:`z = (\mathrm{MS}^{1/3}-1)\,(3/q) + q/3` (Wright & Masters, 1982, ch. 5).

    Returns
    -------
    infit, z_infit, outfit, z_outfit : ndarray
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        n = np.sum(~np.isnan(resid2), axis=axis).astype(float)
        n = np.where(n > 0, n, np.nan)
        outfit = np.nansum(resid2 / w, axis=axis) / n
        q_out = np.sqrt(np.abs(np.nansum(c / w**2, axis=axis) / n**2 - 1.0 / n))
        q_out = np.minimum(q_out, Q_CAP)
        z_outfit = (np.cbrt(outfit) - 1.0) * (3.0 / q_out) + q_out / 3.0
        sum_w = np.nansum(w, axis=axis)
        sum_w = np.where(sum_w > 0, sum_w, np.nan)
        infit = np.nansum(resid2, axis=axis) / sum_w
        q_in = np.minimum(np.sqrt(np.nansum(c - w**2, axis=axis)) / sum_w, Q_CAP)
        z_infit = (np.cbrt(infit) - 1.0) * (3.0 / q_in) + q_in / 3.0
    return infit, z_infit, outfit, z_outfit
