r"""Polychoric / tetrachoric correlations: NumPy / SciPy kernels (DESIGN.md §16.18).

The bivariate normal distribution function uses Owen's (1956) :math:`T` function
(:func:`scipy.special.owens_t`), the two-step maximum likelihood estimator of the correlation
follows Olsson (1979). Every function works on arrays of pairs at once; names, validation and
warnings live in :mod:`flexpyctt.correlation`.

flexpyirt has a JAX implementation of the same estimator (``flexpyirt._polychoric``, Genz's 2004
quadrature and automatic differentiation). The two are independent and are tested to agree.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.special import ndtr, owens_t

__all__ = [
    "PairFit",
    "bivariate_normal_cdf",
    "bivariate_normal_density",
    "cell_probabilities",
    "fit_pairs",
    "pair_log_likelihood",
]

_TWO_PI = 2.0 * np.pi
# Probabilities below this are treated as this value in the log-likelihood (and as constant).
_TINY = 1e-300
_MAX_HALVINGS = 40


def bivariate_normal_cdf(
    h: np.ndarray | float, k: np.ndarray | float, r: np.ndarray | float
) -> np.ndarray:
    r"""Evaluate the standard bivariate normal distribution function :math:`\Phi_2(h,k;r)`.

    .. math::
        \Phi_2(h,k;r)=P(X\le h, Y\le k)=\tfrac12\Phi(h)+\tfrac12\Phi(k)-T(h,a_h)-T(k,a_k)-\beta,

    .. math::
        a_h=\frac{k-rh}{h\sqrt{1-r^2}},\quad a_k=\frac{h-rk}{k\sqrt{1-r^2}},\quad
        \beta=\begin{cases}0 & hk>0\ \text{or}\ (hk=0,\ h+k\ge 0)\\ \tfrac12 & \text{otherwise}
        \end{cases}

    with Owen's :math:`T(h,a)=\frac{1}{2\pi}\int_0^a\frac{e^{-h^2(1+x^2)/2}}{1+x^2}\,dx` (Owen,
    1956). For :math:`h=0` the limit :math:`a_h=\pm\infty` (sign of :math:`k`) is used, and
    :math:`\Phi_2(0,0;r)=\tfrac14+\arcsin r/(2\pi)`. Infinite limits are exact. Requires
    :math:`|r|<1`. The arguments broadcast.
    """
    h, k, r = np.broadcast_arrays(
        np.asarray(h, dtype=np.float64),
        np.asarray(k, dtype=np.float64),
        np.asarray(r, dtype=np.float64),
    )
    out = np.zeros(h.shape, dtype=np.float64)  # 0 where a limit is -inf
    h_inf, k_inf = h == np.inf, k == np.inf
    lower = (h == -np.inf) | (k == -np.inf)
    only_h = h_inf & ~lower
    out[only_h] = ndtr(k[only_h])
    only_k = k_inf & ~h_inf & ~lower
    out[only_k] = ndtr(h[only_k])
    finite = np.isfinite(h) & np.isfinite(k)
    hf, kf, rf = h[finite], k[finite], r[finite]
    scale = np.sqrt(1.0 - rf * rf)
    with np.errstate(divide="ignore", invalid="ignore"):
        a_h = np.where(
            hf == 0.0, np.where(kf >= 0.0, np.inf, -np.inf), (kf - rf * hf) / (hf * scale)
        )
        a_k = np.where(
            kf == 0.0, np.where(hf >= 0.0, np.inf, -np.inf), (hf - rf * kf) / (kf * scale)
        )
    product = hf * kf
    beta = np.where((product > 0.0) | ((product == 0.0) & (hf + kf >= 0.0)), 0.0, 0.5)
    core = 0.5 * ndtr(hf) + 0.5 * ndtr(kf) - owens_t(hf, a_h) - owens_t(kf, a_k) - beta
    origin = (hf == 0.0) & (kf == 0.0)
    core = np.where(origin, 0.25 + np.arcsin(rf) / _TWO_PI, core)
    out[finite] = np.clip(core, 0.0, 1.0)
    return out


def bivariate_normal_density(
    h: np.ndarray | float, k: np.ndarray | float, r: np.ndarray | float
) -> np.ndarray:
    r"""Evaluate the standard bivariate normal density.

    .. math::
        \phi_2(h,k;r)=\frac{1}{2\pi\sqrt{1-r^2}}
        \exp\Big(-\frac{h^2-2rhk+k^2}{2(1-r^2)}\Big),

    which is 0 when ``h`` or ``k`` is infinite. By Plackett (1954),
    :math:`\partial\Phi_2/\partial r=\phi_2`.
    """
    return _density_and_slope(h, k, r)[0]


def _density_and_slope(
    h: np.ndarray | float, k: np.ndarray | float, r: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return :math:`\phi_2` and :math:`\partial\phi_2/\partial r` (0 at infinite limits).

    .. math::
        \frac{\partial\phi_2}{\partial r}=\phi_2\Big[\frac{r}{s}
        +\frac{hk(1+r^2)-r(h^2+k^2)}{s^2}\Big],\qquad s=1-r^2 .
    """
    h, k, r = np.broadcast_arrays(
        np.asarray(h, dtype=np.float64),
        np.asarray(k, dtype=np.float64),
        np.asarray(r, dtype=np.float64),
    )
    density = np.zeros(h.shape, dtype=np.float64)
    slope = np.zeros(h.shape, dtype=np.float64)
    finite = np.isfinite(h) & np.isfinite(k)
    hf, kf, rf = h[finite], k[finite], r[finite]
    s = 1.0 - rf * rf
    sum_sq = hf * hf + kf * kf
    product = hf * kf
    dens = np.exp(-(sum_sq - 2.0 * rf * product) / (2.0 * s)) / (_TWO_PI * np.sqrt(s))
    density[finite] = dens
    slope[finite] = dens * (rf / s + (product * (1.0 + rf * rf) - rf * sum_sq) / (s * s))
    return density, slope


def _rectangles(grid: np.ndarray) -> np.ndarray:
    """Second differences of a ``(..., K + 1, K + 1)`` grid: the ``(..., K, K)`` cell values."""
    out: np.ndarray = (
        grid[..., 1:, 1:] - grid[..., :-1, 1:] - grid[..., 1:, :-1] + grid[..., :-1, :-1]
    )
    return out


def cell_probabilities(
    bounds_row: np.ndarray, bounds_col: np.ndarray, rho: np.ndarray
) -> np.ndarray:
    r"""Cell probabilities of the discretized bivariate normal (Olsson, 1979, eq. 2).

    .. math::
        \pi_{ab}=\Phi_2(\tau_a,\tau'_b)-\Phi_2(\tau_{a-1},\tau'_b)
        -\Phi_2(\tau_a,\tau'_{b-1})+\Phi_2(\tau_{a-1},\tau'_{b-1}),

    with ``bounds_row`` :math:`=(-\infty,\tau_1,\dots,\tau_{K-1},\infty,\dots)` of shape
    ``(P, K_max + 1)`` for ``P`` pairs (padding repeats :math:`+\infty`, so padded cells have
    probability 0) and ``rho`` of shape ``(P,)``. Returns ``(P, K_max, K_max)``.
    """
    rho = np.asarray(rho, dtype=np.float64)
    cdf = bivariate_normal_cdf(bounds_row[:, :, None], bounds_col[:, None, :], rho[:, None, None])
    return _rectangles(cdf)


def pair_log_likelihood(
    z: np.ndarray, bounds_row: np.ndarray, bounds_col: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    r"""Two-step log-likelihood of each pair in :math:`z=\operatorname{atanh}\rho`.

    .. math::
        \ell(\rho)=\sum_{a,b} n_{ab}\log\pi_{ab}(\rho)

    (Olsson, 1979, eq. 3, thresholds fixed at their marginal estimates). Cells with
    :math:`n_{ab}=0` contribute 0. All arguments have a leading pair axis.
    """
    probs = cell_probabilities(bounds_row, bounds_col, np.tanh(np.asarray(z, dtype=np.float64)))
    observed = counts > 0.0
    safe = np.where(observed, np.maximum(probs, _TINY), 1.0)
    out: np.ndarray = np.sum(np.where(observed, counts * np.log(safe), 0.0), axis=(1, 2))
    return out


def _derivatives(
    z: np.ndarray, bounds_row: np.ndarray, bounds_col: np.ndarray, counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Log-likelihood, gradient and second derivative in :math:`z` of each pair.

    With :math:`\pi'_{ab}=\Delta\phi_2` and :math:`\pi''_{ab}=\Delta\,\partial_\rho\phi_2`
    (rectangle differences over the cell's thresholds),

    .. math::
        \ell_\rho=\sum n_{ab}\frac{\pi'_{ab}}{\pi_{ab}},\quad
        \ell_{\rho\rho}=\sum n_{ab}\Big[\frac{\pi''_{ab}}{\pi_{ab}}
        -\Big(\frac{\pi'_{ab}}{\pi_{ab}}\Big)^2\Big],\quad
        \ell_z=s\,\ell_\rho,\quad \ell_{zz}=s^2\ell_{\rho\rho}-2\rho s\,\ell_\rho,

    where :math:`s=1-\rho^2=d\rho/dz`.
    """
    rho = np.tanh(z)
    row, col, r = bounds_row[:, :, None], bounds_col[:, None, :], rho[:, None, None]
    probs = _rectangles(bivariate_normal_cdf(row, col, r))
    density, slope = _density_and_slope(row, col, r)
    first, second = _rectangles(density), _rectangles(slope)
    used = (counts > 0.0) & (probs > _TINY)
    safe = np.where(used, probs, 1.0)
    observed = counts > 0.0
    log_probs = np.log(np.where(observed, np.maximum(probs, _TINY), 1.0))
    loglik = np.sum(np.where(observed, counts * log_probs, 0.0), axis=(1, 2))
    ratio = np.where(used, first / safe, 0.0)
    grad_rho = np.sum(counts * ratio, axis=(1, 2))
    hess_rho = np.sum(counts * (np.where(used, second / safe, 0.0) - ratio * ratio), axis=(1, 2))
    s = 1.0 - rho * rho
    return loglik, s * grad_rho, s * s * hess_rho - 2.0 * rho * s * grad_rho


class PairFit(NamedTuple):
    """Result of :func:`fit_pairs` for each pair (leading axis)."""

    rho: np.ndarray
    log_likelihood: np.ndarray
    gradient: np.ndarray
    n_iter: np.ndarray
    converged: np.ndarray
    at_boundary: np.ndarray


def fit_pairs(
    z0: np.ndarray,
    bounds_row: np.ndarray,
    bounds_col: np.ndarray,
    counts: np.ndarray,
    *,
    z_max: float,
    tol: float,
    max_iter: int,
) -> PairFit:
    r"""Maximize :func:`pair_log_likelihood` over ``z`` in ``[-z_max, z_max]`` for every pair.

    Newton's method with the analytic derivatives of :func:`_derivatives`, run on all pairs that
    have not converged at once. A step is the Newton step when the log-likelihood is concave at
    the current point and half a unit along the gradient otherwise, limited to 2 in absolute
    value and halved (at most 40 times) while it lowers the log-likelihood. A pair stops when the
    projected gradient is at most ``tol`` times its number of observations or the step is below
    :math:`10^{-14}`. Afterwards a pair whose log-likelihood at the bound is at least that at the
    stationary point is moved to the bound and flagged ``at_boundary`` (the log-likelihood then
    increases monotonically towards :math:`|\rho|=1`, e.g. a 2 x 2 table with an empty cell).
    """
    z = np.clip(np.asarray(z0, dtype=np.float64), -z_max, z_max).copy()
    n_pairs = z.size
    n_total = counts.sum(axis=(1, 2))
    loglik, grad, hess = _derivatives(z, bounds_row, bounds_col, counts)
    n_iter = np.zeros(n_pairs, dtype=np.int64)
    done = np.zeros(n_pairs, dtype=bool)
    for _ in range(max_iter):
        active = np.flatnonzero(~done)
        if active.size == 0:
            break
        rows, cols, tables = bounds_row[active], bounds_col[active], counts[active]
        z_old, ll_old, g, hz = z[active], loglik[active], grad[active], hess[active]
        concave = hz < 0.0
        step = np.where(concave, -g / np.where(concave, hz, -1.0), np.sign(g) * 0.5)
        step = np.clip(step, -2.0, 2.0)
        z_try = np.clip(z_old + step, -z_max, z_max)
        ll_try = pair_log_likelihood(z_try, rows, cols, tables)
        for _halving in range(_MAX_HALVINGS):
            worse = np.flatnonzero(ll_try < ll_old - 1e-12 * np.abs(ll_old))
            if worse.size == 0:
                break
            step[worse] /= 2.0
            z_try[worse] = np.clip(z_old[worse] + step[worse], -z_max, z_max)
            ll_try[worse] = pair_log_likelihood(
                z_try[worse], rows[worse], cols[worse], tables[worse]
            )
        ll_new, g_new, h_new = _derivatives(z_try, rows, cols, tables)
        pushes_out = ((z_try >= z_max) & (g_new > 0.0)) | ((z_try <= -z_max) & (g_new < 0.0))
        projected = np.where(pushes_out, 0.0, g_new)
        z[active], loglik[active], grad[active], hess[active] = z_try, ll_new, g_new, h_new
        n_iter[active] += 1
        done[active] = (np.abs(projected) <= tol * n_total[active]) | (
            np.abs(z_try - z_old) <= 1e-14
        )
    z_bound = np.where(z >= 0.0, z_max, -z_max)
    ll_bound, g_bound, _ = _derivatives(z_bound, bounds_row, bounds_col, counts)
    at_bound = ll_bound >= loglik
    z = np.where(at_bound, z_bound, z)
    return PairFit(
        rho=np.tanh(z),
        log_likelihood=np.where(at_bound, ll_bound, loglik),
        gradient=np.where(at_bound, g_bound, grad),
        n_iter=n_iter,
        converged=done | at_bound,
        at_boundary=at_bound,
    )
