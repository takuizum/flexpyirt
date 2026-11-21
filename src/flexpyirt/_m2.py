"""Limited-information goodness of fit: the M2 / M2* statistic (Result layer, DESIGN.md §16.3).

Model-implied item-score moments and their Jacobian come from the fitted Core problem on its
quadrature grid (the Jacobian by JAX forward-mode autodiff); the asymptotic covariance, SVD
orthogonal complement, and fit indices run in NumPy / SciPy outside ``jit``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy import optimize, stats

from flexpyirt import _core
from flexpyirt._core import CompiledProblem
from flexpyirt.errors import FlexpyirtWarning, SpecificationError

__all__ = [
    "M2Result",
    "ci_column_names",
    "conditional_moments",
    "m2_statistic",
    "model_margins",
    "null_m2",
    "rmsea",
    "rmsea_ci",
]


@dataclass(frozen=True)
class M2Result:
    """Values reported by :meth:`FitResult.m2` before DataFrame formatting."""

    m2: float
    df: int
    p: float
    rmsea: float
    rmsea_lower: float
    rmsea_upper: float
    srmsr: float
    tli: float
    cfi: float


def ci_column_names(ci: float) -> tuple[str, str]:
    """Column names for the RMSEA confidence interval (e.g. ``rmsea_5``, ``rmsea_95`` for 0.90)."""
    lo_pct = round(50.0 * (1.0 - ci), 10)
    hi_pct = round(50.0 * (1.0 + ci), 10)
    return f"rmsea_{lo_pct:g}", f"rmsea_{hi_pct:g}"


def _pairs(n_items: int) -> tuple[np.ndarray, np.ndarray]:
    """Return pairs ``(i, j)`` with ``i < j`` in mirt's column-major ``lower.tri`` order."""
    ii, jj = [], []
    for i in range(n_items):
        for j in range(i + 1, n_items):
            ii.append(i)
            jj.append(j)
    return np.asarray(ii, dtype=int), np.asarray(jj, dtype=int)


def conditional_moments(prob: CompiledProblem, eta: Array) -> tuple[Array, Array]:
    r"""Return ``(J, Q)`` conditional first and second moments of item scores.

    .. math::
        m_{1,jq} = \sum_{k=0}^{K_j-1} k\,P_{jk}(\theta_q),\qquad
        m_{2,jq} = \sum_{k=0}^{K_j-1} k^2 P_{jk}(\theta_q).
    """
    n_items = int(prob.y.shape[1])
    n_nodes = int(prob.nodes.shape[0])
    m1 = jnp.zeros((n_items, n_nodes), dtype=prob.nodes.dtype)
    m2 = jnp.zeros_like(m1)
    for fam, logp in zip(prob.families, _core._item_log_probs(prob, eta), strict=True):
        p = jnp.exp(logp[0])  # (J_f, Q, K_f)
        k = jnp.arange(fam.kernel.n_cat, dtype=p.dtype)
        m1 = m1.at[fam.columns, :].set(jnp.sum(p * k, axis=-1))
        m2 = m2.at[fam.columns, :].set(jnp.sum(p * k**2, axis=-1))
    return m1, m2


def model_margins(prob: CompiledProblem, eta: Array) -> Array:
    r"""Return ``(s,)`` model-implied means and cross-products of item scores.

    Maydeu-Olivares & Joe (2005, 2006); mirt's ``M2*``: :math:`E[y_j]` for every item, then
    :math:`E[y_iy_j]` for :math:`i < j` in column-major order.
    """
    m1, _ = conditional_moments(prob, eta)
    w = jnp.exp(_core._latent_log_weights(prob, eta)[0])  # (Q,)
    ii, jj = _pairs(int(m1.shape[0]))
    e1 = m1 @ w
    e2 = (m1[ii, :] * m1[jj, :]) @ w
    return jnp.concatenate([e1, e2])


def _xi(m1: np.ndarray, m2: np.ndarray, w: np.ndarray, e: np.ndarray) -> np.ndarray:
    r"""Asymptotic covariance :math:`\Xi_2` of sample margins under local independence.

    For margins :math:`a, b` (single items or pairs of distinct items; Maydeu-Olivares & Joe,
    2005, 2006), :math:`\Xi_{ab} = E[y_a y_b] - E[y_a] E[y_b]`, where the conditional moment of
    the product uses :math:`m_2` for any item appearing in both :math:`a` and :math:`b`.
    """
    n_items = m1.shape[0]
    ii, jj = _pairs(n_items)
    n_pairs = ii.size
    p_idx = np.arange(n_pairs)
    e1, e2 = e[:n_items], e[n_items:]

    # Block (1, 1): E[y_a y_b] - E[y_a] E[y_b].
    xi11 = (m1 * w) @ m1.T
    xi11[np.diag_indices(n_items)] = m2 @ w
    xi11 -= np.outer(e1, e1)

    # Block (1, 2): E[y_a y_i y_j] - E[y_a] E[y_i y_j].
    p1 = m1[ii] * m1[jj]  # (P, Q)
    xi12 = (m1 * w) @ p1.T
    xi12[ii, p_idx] = (m2[ii] * m1[jj]) @ w
    xi12[jj, p_idx] = (m1[ii] * m2[jj]) @ w
    xi12 -= np.outer(e1, e2)

    # Block (2, 2): E[y_i y_j y_k y_l] - E[y_i y_j] E[y_k y_l].
    xi22 = (p1 * w) @ p1.T
    # Pairs sharing exactly one item x: overwrite the (J-1) x (J-1) submatrix of pairs containing x.
    for x in range(n_items):
        mask_i = ii == x
        mask_j = jj == x
        p_x = np.flatnonzero(mask_i | mask_j)
        other_x = np.where(mask_i[p_x], jj[p_x], ii[p_x])
        m1_other = m1[other_x]
        xi22[np.ix_(p_x, p_x)] = ((m2[x] * w) * m1_other) @ m1_other.T
    # Diagonal (pairs sharing both items): E[y_i^2 y_j^2].
    xi22[p_idx, p_idx] = (m2[ii] * m2[jj]) @ w
    xi22 -= np.outer(e2, e2)

    xi = np.block([[xi11, xi12], [xi12.T, xi22]])
    return np.asarray(0.5 * (xi + xi.T), dtype=float)


def _quadratic_form(resid: np.ndarray, delta: np.ndarray, xi: np.ndarray) -> float:
    r"""Evaluate :math:`e^\top \Delta_c (\Delta_c^\top \Xi_2 \Delta_c)^{-1} \Delta_c^\top e`.

    Uses an orthonormal basis :math:`\Delta_c` for the orthogonal complement of
    :math:`\operatorname{col}(\Delta)` (numerical rank by SVD). Returns ``NaN`` when
    :math:`\Delta_c` is empty.
    """
    n_margins, n_cols = delta.shape
    if n_cols == 0:
        comp = np.eye(n_margins, dtype=float)
    else:
        u, sv, _ = np.linalg.svd(delta, full_matrices=True)
        tol = float(sv.max(initial=0.0)) * max(n_margins, n_cols) * float(np.finfo(float).eps)
        rank = int(np.sum(sv > tol))
        if rank >= n_margins:
            return float("nan")
        comp = u[:, rank:]
    proj = comp.T @ resid
    mid = comp.T @ xi @ comp
    mid = 0.5 * (mid + mid.T)
    try:
        sol = np.linalg.solve(mid, proj)
    except np.linalg.LinAlgError as exc:
        raise SpecificationError(
            "Could not invert the orthogonal complement covariance matrix for M2."
        ) from exc
    return float(abs(proj @ sol))


def rmsea(x2: float, df: int, n: int) -> float:
    r"""Compute :math:`\sqrt{\max(0,\,X^2/df - 1)/(N - 1)}` (mirt's ``rmsea``).

    Returns ``NaN`` when :math:`df \le 0`, :math:`N \le 1`, or ``x2`` is non-finite.
    """
    if df <= 0 or n <= 1 or not np.isfinite(x2):
        return float("nan")
    return float(np.sqrt(max(0.0, x2 / df - 1.0) / (n - 1)))


def rmsea_ci(x2: float, df: int, n: int, level: float = 0.90) -> tuple[float, float]:
    r"""Compute RMSEA confidence interval by inverting the noncentral :math:`\chi^2` CDF.

    MacCallum, Browne & Sugawara (1996): :math:`\sqrt{\lambda / (N\,df)}` matching mirt's
    ``RMSEA.CI``. A bound whose root is not bracketed on :math:`[0, X^2]` (lower) or
    :math:`[0, \max(N, 5 X^2)]` (upper) is 0.
    """
    if df <= 0 or n <= 0 or not np.isfinite(x2):
        return float("nan"), float("nan")
    alpha = (1.0 - level) / 2.0

    def _root(target: float, upper: float) -> float:
        def f(lam: float) -> float:
            cdf = float(stats.ncx2.cdf(x2, df, lam) if lam > 0.0 else stats.chi2.cdf(x2, df))
            return cdf - target

        if upper <= 0.0 or f(0.0) * f(upper) > 0.0:
            return 0.0
        return float(optimize.brentq(f, 0.0, upper, xtol=1e-12, rtol=1e-14))

    lam_l = _root(1.0 - alpha, x2)
    lam_u = _root(alpha, max(float(n), 5.0 * x2))
    return float(np.sqrt(lam_l / (n * df))), float(np.sqrt(lam_u / (n * df)))


def null_m2(y: np.ndarray) -> tuple[float, int]:
    r""":math:`M_2` of the dichotomous independence (null) model, for TLI and CFI.

    With every slope fixed at 0 and :math:`J` free item intercepts (whose MLE reproduces the
    sample proportions :math:`\bar y_j`), the orthogonal complement of :math:`\Delta_{\text{null}}`
    maps residuals to sample covariances and diagonalizes :math:`\Xi_2`, simplifying exactly to

    .. math::
        M_{2,\text{null}} = N \sum_{i < j} r_{ij}^2, \qquad
        df_{\text{null}} = \binom{J}{2},

    where :math:`r_{ij}` is the sample Pearson correlation between items :math:`i` and :math:`j`.
    """
    n_obs, n_items = y.shape
    df_null = n_items * (n_items - 1) // 2
    if df_null <= 0:
        return float("nan"), df_null
    yf = y.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(yf, rowvar=False)
    ii, jj = _pairs(n_items)
    r_pairs = corr[ii, jj]
    if not np.all(np.isfinite(r_pairs)):
        return float("nan"), df_null
    return float(n_obs * np.sum(r_pairs**2)), df_null


def m2_statistic(
    prob: CompiledProblem,
    eta: Array,
    y: np.ndarray,
    *,
    n_params: int,
    n_cat: Sequence[int],
    level: float = 0.90,
    _latent_in_delta: bool = True,
) -> M2Result:
    r"""Compute the :math:`M_2` / :math:`M_2^*` statistic (Maydeu-Olivares & Joe, 2005, 2006).

    .. math::
        M_2 = N\,(\hat p_2 - \pi_2)^\top C_2\,(\hat p_2 - \pi_2),\qquad
        C_2 = \Delta_c (\Delta_c^\top \Xi_2 \Delta_c)^{-1} \Delta_c^\top,\qquad
        df = s - q,

    where :math:`\Delta_c` is an orthonormal basis for the orthogonal complement of
    :math:`\Delta_2 = \partial\pi_2 / \partial\eta^\top` (shared parameters once), :math:`\Xi_2`
    is the asymptotic covariance of :math:`\sqrt{N}(\hat p_2 - \pi_2)` under local independence,
    :math:`s = J + \binom{J}{2}`, and :math:`q = \text{n\_params}`.

    By default ``_latent_in_delta=True`` includes free latent parameters in :math:`\Delta_2`
    (Maydeu-Olivares & Joe, 2005, 2006); setting ``_latent_in_delta=False`` drops them from
    :math:`\Delta_2` while keeping them in ``df`` to match R mirt 1.46.1's convention.
    """
    margins = np.asarray(model_margins(prob, eta), dtype=float)
    jac = np.asarray(jax.jacfwd(lambda e: model_margins(prob, e))(eta), dtype=float)
    if not _latent_in_delta:
        lat_fids = {int(i) for i in np.asarray(prob.latent.free_index).ravel() if int(i) >= 0}
        if lat_fids:
            keep = [c for c in range(jac.shape[1]) if c not in lat_fids]
            jac = jac[:, keep]

    m1, m2 = (np.asarray(a, dtype=float) for a in conditional_moments(prob, eta))
    w = np.exp(np.asarray(_core._latent_log_weights(prob, eta)[0], dtype=float))
    n_obs, n_items = y.shape
    s = int(margins.shape[0])
    ii, jj = _pairs(n_items)

    yf = y.astype(float)
    obs = np.concatenate([yf.mean(axis=0), (yf[:, ii] * yf[:, jj]).mean(axis=0)])
    resid = np.sqrt(float(n_obs)) * (obs - margins)
    xi = _xi(m1, m2, w, margins)
    stat = _quadratic_form(resid, jac, xi)
    df = int(s - n_params)

    # SRMSR: root mean squared residual of item-score correlations.
    mean_e = margins[:n_items]
    cov_o = np.cov(yf, rowvar=False, bias=True)
    cov_e = np.zeros((n_items, n_items), dtype=float)
    cov_e[ii, jj] = margins[n_items:] - mean_e[ii] * mean_e[jj]
    cov_e = cov_e + cov_e.T
    cov_e[np.diag_indices(n_items)] = m2 @ w - mean_e**2
    with np.errstate(divide="ignore", invalid="ignore"):
        r_o = cov_o / np.sqrt(np.outer(np.diag(cov_o), np.diag(cov_o)))
        r_e = cov_e / np.sqrt(np.outer(np.diag(cov_e), np.diag(cov_e)))
    srmsr = float(np.sqrt(np.mean((r_o[ii, jj] - r_e[ii, jj]) ** 2)))

    if df > 0 and np.isfinite(stat):
        pval = float(stats.chi2.sf(stat, df))
        rmsea_val = rmsea(stat, df, n_obs)
        lo, hi = rmsea_ci(stat, df, n_obs, level)
    else:
        pval = float("nan")
        rmsea_val = float("nan")
        lo, hi = float("nan"), float("nan")

    tli_val = float("nan")
    cfi_val = float("nan")
    if df > 0 and np.isfinite(stat) and all(k == 2 for k in n_cat):
        x2_null, df_null = null_m2(y)
        if np.isfinite(x2_null) and df_null > 0:
            if x2_null > stat:
                denom_tli = x2_null / df_null - 1.0
                if denom_tli != 0.0:
                    tli_val = float((x2_null / df_null - stat / df) / denom_tli)
                denom_cfi = x2_null - df_null
                if denom_cfi != 0.0:
                    cfi_val = float(np.clip(1.0 - (stat - df) / denom_cfi, 0.0, 1.0))
            else:
                warnings.warn(
                    "Null model M2 is smaller than or equal to the fitted model M2; "
                    "TLI and CFI are NaN.",
                    FlexpyirtWarning,
                    stacklevel=3,
                )

    return M2Result(stat, df, pval, rmsea_val, lo, hi, srmsr, tli_val, cfi_val)
