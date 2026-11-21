"""Latent (population) distributions (Spec layer) and their quadrature weights (Core layer)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from flexpyirt.errors import SpecificationError

__all__ = [
    "EmpiricalHistogram",
    "LatentSpec",
    "Normal",
    "eh_moments",
    "eh_project_floor",
    "eh_regrid_cdf",
    "eh_standardize_params",
    "normal_log_weights",
]


@dataclass(frozen=True)
class Normal:
    r"""Normal latent distribution :math:`\theta\sim N(\mu,\sigma^2)` on a quadrature grid.

    Parameters
    ----------
    mean, var : float
        Value of the mean and variance; the starting value when the parameter is free.
    free_mean, free_var : bool
        Whether the mean / variance is estimated.

    Examples
    --------
    >>> Normal.standard()          # N(0, 1), nothing estimated
    >>> Normal(free_var=True)      # N(0, sigma^2), variance estimated (Rasch default)
    >>> Normal.free()              # mean and variance estimated
    """

    mean: float = 0.0
    var: float = 1.0
    free_mean: bool = False
    free_var: bool = False

    def __post_init__(self) -> None:
        """Validate values."""
        if not math.isfinite(self.mean):
            raise SpecificationError(f"Normal mean must be finite, got {self.mean}.")
        if not (math.isfinite(self.var) and self.var > 0):
            raise SpecificationError(
                f"Normal variance must be positive and finite, got {self.var}."
            )

    @classmethod
    def standard(cls) -> Normal:
        """Return the fixed standard normal :math:`N(0,1)`."""
        return cls()

    @classmethod
    def free(cls, *, mean: float = 0.0, var: float = 1.0) -> Normal:
        """Return a normal with both mean and variance estimated (starting at ``mean``, ``var``)."""
        return cls(mean=mean, var=var, free_mean=True, free_var=True)


def normal_log_weights(nodes: Array, mean: Array, var: Array) -> Array:
    r"""Log quadrature weights of a normal density normalized on the grid.

    .. math::
        \log w_q = \log\phi(\theta_q;\mu,\sigma^2) - \log\sum_{q'}\phi(\theta_{q'};\mu,\sigma^2)

    The normalizing constant of :math:`\phi` cancels, so only the kernel
    :math:`-(\theta_q-\mu)^2/(2\sigma^2)` is evaluated (DESIGN.md §4.4).

    Parameters
    ----------
    nodes : Array, shape (Q,)
    mean, var : Array, scalar

    Returns
    -------
    Array, shape (Q,)
    """
    kernel = -0.5 * (nodes - mean) ** 2 / var
    return jax.nn.log_softmax(kernel)


@dataclass(frozen=True, kw_only=True)
class EmpiricalHistogram:
    r"""Empirical histogram (EH) nonparametric latent distribution (Woods, 2007).

    Represents each group's latent distribution by discrete weights
    :math:`w_{g1},\dots,w_{gQ}` on the fixed quadrature grid :math:`\theta_1,\dots,\theta_Q`
    satisfying :math:`w_{gq}\ge\tau>0` and :math:`\sum_q w_{gq}=1` (DESIGN.md §4.4, §15.29).
    During EM, each group's EH M-step solves
    :math:`\max_{w\in\Delta^{Q-1},\,w_q\ge\tau}\sum_q \bar n_{gq}\log w_q` via exact KKT
    water-filling (reducing identically to :math:`\bar n_{gq}/N_g` when no node hits the floor).
    Unless the scale is pinned by fixed item parameters (FIPC), the reference group is
    standardized at every iteration by an affine scale change
    :math:`\theta^*=(\theta-m)/s`, transforming item parameters back and re-gridding every
    group's weights via mass-conserving piecewise-linear CDF interpolation.

    Parameters
    ----------
    min_weight : float, default 1e-10
        Strict positive floor :math:`\tau` on every quadrature node's weight so empty nodes keep
        finite log-weights. Must satisfy :math:`0 < \tau < 1/Q` for the quadrature grid used.
    """

    min_weight: float = 1e-10

    def __post_init__(self) -> None:
        """Validate ``min_weight``."""
        if isinstance(self.min_weight, bool) or not isinstance(self.min_weight, (int, float)):
            raise SpecificationError(
                f"EmpiricalHistogram min_weight must be a float, got {self.min_weight!r}."
            )
        val = float(self.min_weight)
        if not (math.isfinite(val) and 0.0 < val < 1.0):
            raise SpecificationError(
                "EmpiricalHistogram min_weight must be finite and in (0, 1), "
                f"got {self.min_weight}."
            )


LatentSpec = Normal | EmpiricalHistogram
"""Union of supported latent distribution specifications."""

_EH_STD_REFINE_STEPS = 12
"""Number of damped refinement steps solving for exact post-regrid reference mean 0 / var 1."""


def eh_project_floor(counts: Array, min_weight: Array) -> Array:
    r"""Exact KKT maximizer of :math:`\sum_q \bar n_q\log w_q` on the floored simplex.

    Solves

    .. math::
        \max_{w\in\Delta^{Q-1},\;w_q\ge\tau}\sum_{q=1}^Q \bar n_q \log w_q,

    whose unique KKT solution is :math:`w_q=\max(\tau,\,c^*\bar n_q)` with :math:`c^*>0` chosen
    so :math:`\sum_q w_q=1` (DESIGN.md §4.4, §15.29). Reduces identically to
    :math:`\bar n_q / \sum_{q'}\bar n_{q'}` whenever no node hits the floor :math:`\tau`.
    """
    n_nodes = counts.shape[0]
    total = jnp.sum(counts)
    r = jnp.where(total > 0.0, counts / jnp.where(total > 0.0, total, 1.0), 1.0 / n_nodes)
    order = jnp.argsort(r)
    r_sorted = r[order]
    suf = jnp.cumsum(r_sorted[::-1])[::-1]
    ks = jnp.arange(n_nodes, dtype=r.dtype)
    rem = 1.0 - ks * min_weight
    c_k = jnp.where(suf > 0.0, rem / jnp.where(suf > 0.0, suf, 1.0), 0.0)
    above = (c_k * r_sorted) > min_weight
    has_above = jnp.any(above)
    k_opt = jnp.argmax(above)
    c_opt = c_k[k_opt]
    w = jnp.where(has_above, jnp.maximum(min_weight, c_opt * r), jnp.full_like(r, 1.0 / n_nodes))
    return w / jnp.sum(w)


def eh_regrid_cdf(nodes: Array, weights: Array, mean: Array, sd: Array, min_weight: Array) -> Array:
    r"""Re-map weights from :math:`\theta^*=(\theta-m)/s` back onto ``nodes`` by CDF interpolation.

    Treats ``weights`` as a piecewise-constant histogram on the quadrature bin edges
    :math:`e_0=\theta_1-\Delta_1/2,\;e_q=(\theta_q+\theta_{q+1})/2,\;e_Q=\theta_Q+\Delta_{Q-1}/2`,
    evaluates the transformed piecewise-linear CDF :math:`F^*(x)=F(m+sx)` at ``edges``, absorbs
    any mass outside :math:`[e_0, e_Q]` into the boundary bins, and projects onto the floored
    simplex via :func:`eh_project_floor` (DESIGN.md §4.4, §15.29). Exact identity at
    :math:`(m, s) = (0, 1)`.
    """
    d_left = nodes[1] - nodes[0]
    d_right = nodes[-1] - nodes[-2]
    mids = 0.5 * (nodes[:-1] + nodes[1:])
    edges = jnp.concatenate([nodes[:1] - 0.5 * d_left, mids, nodes[-1:] + 0.5 * d_right])
    edges_std = (edges - mean) / sd
    cdf = jnp.concatenate([jnp.zeros((1,), dtype=weights.dtype), jnp.cumsum(weights)])
    cdf = cdf.at[-1].set(1.0)
    cdf_new = jnp.interp(edges, edges_std, cdf, left=0.0, right=1.0)
    w_new = jnp.diff(cdf_new)
    w_new = w_new.at[0].add(cdf_new[0])
    w_new = w_new.at[-1].add(1.0 - cdf_new[-1])
    return eh_project_floor(jnp.maximum(w_new, 0.0), min_weight)


def eh_standardize_params(
    nodes: Array,
    weights: Array,
    min_weight: Array,
    *,
    standardize_mean: bool,
    standardize_var: bool,
) -> tuple[Array, Array]:
    r"""Compute affine standardization :math:`(m, s)` so re-gridded weights have mean 0 / var 1.

    Starts from the raw moments of ``weights`` on ``nodes`` and runs ``_EH_STD_REFINE_STEPS``
    damped fixed-point updates so that :func:`eh_regrid_cdf` produces reference weights with
    :math:`\sum_q w_q\theta_q = 0` (when ``standardize_mean``) and
    :math:`\sum_q w_q(\theta_q - \bar\theta)^2 = 1` (when ``standardize_var``) after every EM
    iteration (DESIGN.md §15.29).
    """
    dtype = nodes.dtype
    zero = jnp.zeros((), dtype=dtype)
    one = jnp.ones((), dtype=dtype)
    if not standardize_mean and not standardize_var:
        return zero, one

    raw_mean = jnp.sum(weights * nodes)
    raw_var = jnp.sum(weights * (nodes - raw_mean) ** 2)
    m0 = raw_mean if standardize_mean else zero
    s0 = jnp.sqrt(jnp.maximum(raw_var, 1e-12)) if standardize_var else one

    def body(_: int, state: tuple[Array, Array]) -> tuple[Array, Array]:
        m, s = state
        wn = eh_regrid_cdf(nodes, weights, m, s, min_weight)
        mk = jnp.sum(wn * nodes)
        vk = jnp.sum(wn * (nodes - mk) ** 2)
        m_next = m + 0.7 * s * mk if standardize_mean else m
        s_next = s * (jnp.maximum(vk, 1e-12) ** 0.35) if standardize_var else s
        return m_next, s_next

    m_fin, s_fin = jax.lax.fori_loop(0, _EH_STD_REFINE_STEPS, body, (m0, s0))
    return m_fin, s_fin


def eh_moments(nodes: np.ndarray, weights: np.ndarray) -> tuple[float, float, float, float, float]:
    r"""Return ``(mean, var, sd, skewness, kurtosis)`` of discrete EH weights on ``nodes``.

    .. math::
        \mu = \sum_q w_q\theta_q,\quad
        \sigma^2 = \sum_q w_q(\theta_q-\mu)^2,\quad
        \gamma_1 = \frac{\sum_q w_q(\theta_q-\mu)^3}{\sigma^3},\quad
        \gamma_2 = \frac{\sum_q w_q(\theta_q-\mu)^4}{\sigma^4} - 3

    where :math:`\gamma_2` is Fisher's excess kurtosis (0 for a normal distribution).
    """
    th = np.asarray(nodes, dtype=float)
    w = np.asarray(weights, dtype=float)
    mean = float(np.sum(w * th))
    centered = th - mean
    var = float(np.sum(w * centered**2))
    if var <= 0.0:
        return mean, max(var, 0.0), 0.0, math.nan, math.nan
    sd = math.sqrt(var)
    skewness = float(np.sum(w * centered**3)) / (sd**3)
    kurtosis = float(np.sum(w * centered**4)) / (var**2) - 3.0
    return mean, var, sd, skewness, kurtosis
