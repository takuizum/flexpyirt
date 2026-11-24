r"""Exploratory factor analysis of a correlation matrix and the Hull method (DESIGN.md §16.14).

Unrotated ULS (minres) and ML factor solutions by minimizing over the uniquenesses
:math:`\Psi` with L-BFGS-B and analytic gradients (Jöreskog, 1967; Harman & Jones, 1966), the
EFAtools fit indices (CAF, :math:`\chi^2`, CFI, RMSEA) and the Hull selection rule
(Lorenzo-Seva, Timmerman & Kiers, 2011). Pure NumPy / SciPy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2 as chi2_dist

__all__ = ["FactorSolution", "factor_df", "fit_factors", "fit_indices", "hull_select", "kmo"]

UNIQUENESS_FLOOR = 0.005


@dataclass(frozen=True)
class FactorSolution:
    """Unrotated solution: ``loadings`` ``(p, k)``, ``uniquenesses`` ``(p,)`` and diagnostics."""

    loadings: np.ndarray
    uniquenesses: np.ndarray
    objective: float
    converged: bool
    n_iter: int


def _top_eigen(mat: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(mat)
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


def _uls_objective(psi: np.ndarray, corr: np.ndarray, k: int) -> tuple[float, np.ndarray]:
    r"""ULS objective and gradient in :math:`\Psi`.

    .. math::
        F(\Psi)=\tfrac12\lVert R-\Psi-\Lambda\Lambda^\top\rVert_F^2=\tfrac12\sum_{m>k}\lambda_m^2,
        \qquad \frac{\partial F}{\partial\psi_i}=(\Lambda\Lambda^\top+\Psi-R)_{ii},

    with :math:`\lambda_m, v_m` the eigenpairs of :math:`R-\Psi` and
    :math:`\Lambda=V_k\operatorname{diag}(\max(\lambda_{1..k},0))^{1/2}` (the gradient is psych's
    ``FAgr.minres``).
    """
    values, vectors = _top_eigen(corr - np.diag(psi), k)
    loadings = vectors[:, :k] * np.sqrt(np.maximum(values[:k], 0.0))
    resid = corr - np.diag(psi) - loadings @ loadings.T
    return 0.5 * float(np.sum(resid * resid)), -np.diag(resid).copy()


def _ml_loadings(psi: np.ndarray, corr: np.ndarray, k: int) -> np.ndarray:
    scale = 1.0 / np.sqrt(psi)
    values, vectors = _top_eigen(corr * scale[:, None] * scale[None, :], k)
    out: np.ndarray = (
        np.sqrt(psi)[:, None] * vectors[:, :k] * np.sqrt(np.maximum(values[:k] - 1.0, 0.0))
    )
    return out


def _ml_objective(psi: np.ndarray, corr: np.ndarray, k: int) -> tuple[float, np.ndarray]:
    r"""ML concentrated objective and gradient in :math:`\Psi` (Jöreskog, 1967).

    .. math::
        F(\Psi)=-\sum_{m>k}(\log\theta_m-\theta_m)-k+p\ \ (\text{psych's sign}),\qquad
        \frac{\partial F}{\partial\psi_i}=\frac{(\Lambda\Lambda^\top+\Psi-R)_{ii}}{\psi_i^2},

    with :math:`\theta_m` the eigenvalues of :math:`\Psi^{-1/2}R\Psi^{-1/2}` (psych's ``FAfn``
    and ``FAgr``).
    """
    scale = 1.0 / np.sqrt(psi)
    values, _ = _top_eigen(corr * scale[:, None] * scale[None, :], k)
    tail = values[k:]
    value = -(float(np.sum(np.log(tail) - tail)) - k + corr.shape[0])
    loadings = _ml_loadings(psi, corr, k)
    grad = np.diag(loadings @ loadings.T + np.diag(psi) - corr) / psi**2
    return value, grad


def smc(corr: np.ndarray) -> np.ndarray:
    """Squared multiple correlations, clipped to ``[0, 1]`` (EFAtools' ``.smc_start``)."""
    return np.clip(1.0 - 1.0 / np.diag(np.linalg.inv(corr)), 0.0, 1.0)


def fit_factors(corr: np.ndarray, k: int, method: str) -> FactorSolution:
    r"""Fit ``k`` factors by ULS or ML, starting at :math:`\psi_i=1-\mathrm{SMC}_i`.

    Uniquenesses are bounded to :math:`[0.005,\max(1,\mathrm{SMC})]` as in psych and EFAtools.
    Columns are signed so that their sums are non-negative.
    """
    start_smc = smc(corr)
    start = 1.0 - start_smc
    upper = max(1.0, float(np.max(start_smc)))
    fun = _uls_objective if method == "uls" else _ml_objective
    res = minimize(
        fun,
        np.clip(start, UNIQUENESS_FLOOR, upper),
        args=(corr, k),
        jac=True,
        method="L-BFGS-B",
        bounds=[(UNIQUENESS_FLOOR, upper)] * corr.shape[0],
        options={"ftol": 1e-15, "gtol": 1e-11, "maxiter": 10000, "maxcor": 30},
    )
    psi = np.asarray(res.x, dtype=np.float64)
    if method == "uls":
        values, vectors = _top_eigen(corr - np.diag(psi), k)
        loadings = vectors[:, :k] * np.sqrt(np.maximum(values[:k], 0.0))
    else:
        loadings = _ml_loadings(psi, corr, k)
    signs = np.where(loadings.sum(axis=0) < 0, -1.0, 1.0)
    return FactorSolution(
        loadings=loadings * signs,
        uniquenesses=psi,
        objective=float(res.fun),
        converged=bool(res.success),
        n_iter=int(res.nit),
    )


def kmo(corr: np.ndarray) -> float:
    r"""Kaiser–Meyer–Olkin measure :math:`\sum_{i\ne j}r_{ij}^2/(\sum_{i\ne j}r_{ij}^2+q_{ij}^2)`.

    :math:`q_{ij}` are the partial correlations from :math:`R^{-1}` (EFAtools' ``.compute_kmo``).
    """
    inv = np.linalg.inv(corr)
    d = 1.0 / np.sqrt(np.diag(inv))
    partial = inv * d[:, None] * d[None, :]
    off = ~np.eye(corr.shape[0], dtype=bool)
    r2 = float(np.sum(corr[off] ** 2))
    q2 = float(np.sum(partial[off] ** 2))
    return r2 / (r2 + q2)


def factor_df(p: int, k: int) -> int:
    """Degrees of freedom :math:`((p-k)^2-(p+k))/2` of a ``k``-factor model."""
    return ((p - k) ** 2 - (p + k)) // 2


def max_identified_factors(p: int) -> int:
    """Largest ``k`` with positive degrees of freedom (EFAtools' ``.det_max_factors``)."""
    return max(0, int(np.floor((2 * p + 1 - np.sqrt(8 * p + 9)) / 2)))


def fit_indices(loadings: np.ndarray, corr: np.ndarray, n_obs: int) -> dict[str, float]:
    r"""EFAtools' fit indices of a factor solution (its ``.gof``).

    With :math:`\hat\Sigma=\Lambda\Lambda^\top` with unit diagonal:
    :math:`\mathrm{CAF}=1-\mathrm{KMO}(R-\Lambda\Lambda^\top\text{ with unit diagonal})`;
    :math:`F=\operatorname{tr}(\hat\Sigma^{-1}R)+\log|\hat\Sigma|-\log|R|-p` (at least 0),
    the reported :math:`\chi^2=F\,(N-1-(2p+5)/6-2k/3)` (Bartlett's correction) with its
    :math:`p`-value; CFI and RMSEA use the uncorrected :math:`\chi^2_u=F\,(N-1)` and
    :math:`\chi^2_0=-\log|R|\,(N-1)` with :math:`df_0=p(p-1)/2`:
    :math:`\mathrm{CFI}=1-\max(\chi^2_u-df,0)/\max(\chi^2_u-df,\chi^2_0-df_0)` (1 when both are 0),
    :math:`\mathrm{RMSEA}=\min\{\sqrt{\max(0,\chi^2_u-df)/(df(N-1))},1\}`.
    """
    p, k = loadings.shape
    implied = loadings @ loadings.T
    resid = corr - implied
    np.fill_diagonal(resid, 1.0)
    caf = 1.0 - kmo(resid)
    sigma = implied.copy()
    np.fill_diagonal(sigma, 1.0)
    log_det_r = float(np.linalg.slogdet(corr)[1])
    sign_s, log_det_s = np.linalg.slogdet(sigma)
    if sign_s <= 0:
        discrepancy = float("nan")
    else:
        discrepancy = max(
            0.0, float(np.trace(np.linalg.solve(sigma, corr))) + float(log_det_s) - log_det_r - p
        )
    df = factor_df(p, k)
    chi2_u = discrepancy * (n_obs - 1)
    chi2 = discrepancy * (n_obs - 1 - (2 * p + 5) / 6 - 2 * k / 3)
    chi2_null = -log_det_r * (n_obs - 1)
    df_null = p * (p - 1) // 2
    d_model = max(chi2_u - df, 0.0)
    d_null = max(chi2_null - df_null, 0.0)
    cfi = 1.0 if (df == 0 or max(d_model, d_null) == 0) else 1.0 - d_model / max(d_model, d_null)
    rmsea = 0.0 if df == 0 else min(np.sqrt(max(0.0, chi2_u - df) / (df * (n_obs - 1))), 1.0)
    p_value = float(chi2_dist.sf(chi2, df)) if df > 0 else float("nan")
    return {
        "chi2": chi2,
        "df": float(df),
        "p": p_value,
        "caf": caf,
        "cfi": float(cfi),
        "rmsea": float(rmsea),
    }


def null_row(corr: np.ndarray, n_obs: int) -> dict[str, float]:
    """Return the zero-factor row of the Hull table (EFAtools' ``efa_hull``)."""
    p = corr.shape[0]
    df0 = p * (p - 1) // 2
    chi2_null = -float(np.linalg.slogdet(corr)[1]) * (n_obs - 1)
    rmsea0 = np.sqrt(max(0.0, chi2_null - df0) / (df0 * (n_obs - 1)))
    return {"caf": 1.0 - kmo(corr), "cfi": 0.0, "rmsea": 1.0 - float(rmsea0), "df": float(df0)}


def hull_select(fit: np.ndarray, df: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    r"""Hull selection (Lorenzo-Seva, Timmerman & Kiers, 2011), as EFAtools' ``.hull_calc``.

    Rows are :math:`k=0..J`; ``fit`` is increasing-is-better, ``df`` decreasing. Solutions whose
    fit does not increase are dropped, then interior points on or below the line through their
    neighbours (the upper convex boundary in (df, fit)) are removed repeatedly. For the remaining
    interior points
    :math:`st_i=\frac{(f_i-f_{i-1})/(df_i-df_{i-1})}{(f_{i+1}-f_i)/(df_{i+1}-df_i)}`, and the
    selected :math:`k` maximizes :math:`st`. With fewer than three points on the hull the
    :math:`k` with the largest fit is returned and :math:`st` is NaN.

    Returns
    -------
    tuple
        Selected ``k``, ``st`` per row (NaN off the hull's interior), and whether each row is on
        the hull.
    """
    keep = np.flatnonzero(np.isfinite(fit))
    ks = keep.copy()
    f = fit[keep]
    d = df[keep]
    diff = np.diff(f)
    while np.any(diff < 0):
        mask = np.concatenate([[True], diff > 0])
        ks, f, d = ks[mask], f[mask], d[mask]
        if ks.size == 1:
            break
        diff = np.diff(f)
    i = 1
    while i <= ks.size - 2:
        predicted = f[i - 1] + (f[i + 1] - f[i - 1]) / (d[i + 1] - d[i - 1]) * (d[i] - d[i - 1])
        if f[i] <= predicted:
            ks, f, d = np.delete(ks, i), np.delete(f, i), np.delete(d, i)
            i = 0
        i += 1
    st = np.full(fit.shape[0], np.nan)
    on_hull = np.isin(np.arange(fit.shape[0]), ks)
    if ks.size < 3:
        return int(ks[int(np.argmax(f))]), st, on_hull
    inner = np.empty(ks.size - 2)
    for i in range(1, ks.size - 1):
        inner[i - 1] = ((f[i] - f[i - 1]) / (d[i] - d[i - 1])) / (
            (f[i + 1] - f[i]) / (d[i + 1] - d[i])
        )
        st[ks[i]] = inner[i - 1]
    return int(ks[1 + int(np.argmax(inner))]), st, on_hull
