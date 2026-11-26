"""Kernels for observed-score DIF (DESIGN.md §16.20).

Every function here works on plain NumPy arrays for one studied item: ``x`` (the item scores),
``member`` (0 = reference, 1..G-1 = focal groups) and ``score`` (the matching variable, one value
per respondent; every distinct value is a stratum). The public layer (:mod:`flexpyctt.dif`)
builds the matching scores, runs purification and assembles the tables.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit
from scipy.stats import chi2 as chi2_dist

__all__ = [
    "adjust_p",
    "generalized_mh_item",
    "liu_agresti_item",
    "logistic_fit",
    "mantel_haenszel_item",
    "mantel_item",
]


def adjust_p(p: np.ndarray, method: str) -> np.ndarray:
    """Holm (1979) or Benjamini–Hochberg (1995) adjusted p-values; NaN entries are left out.

    Identical to R's ``p.adjust(method = "holm" / "BH")`` on the finite entries.
    """
    p = np.asarray(p, dtype=np.float64)
    if method == "none":
        return p.copy()
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    m = ok.size
    if m == 0:
        return out
    order = ok[np.argsort(p[ok], kind="stable")]
    ranks = np.arange(1, m + 1)
    if method == "holm":
        adj = np.maximum.accumulate((m - ranks + 1) * p[order])
    else:  # "BH"
        adj = np.minimum.accumulate((p[order] * m / ranks)[::-1])[::-1]
    out[order] = np.minimum(adj, 1.0)
    return out


def _strata(score: np.ndarray) -> tuple[np.ndarray, int]:
    _, inverse = np.unique(score, return_inverse=True)
    inverse = np.asarray(inverse, dtype=np.int64).ravel()
    return inverse, int(inverse.max()) + 1 if inverse.size else 0


def mantel_haenszel_item(
    x: np.ndarray, member: np.ndarray, score: np.ndarray, *, continuity_correction: bool
) -> dict[str, float]:
    r"""Mantel–Haenszel statistic, common odds ratio and the variance of its log for one item.

    With, in stratum :math:`k`, :math:`A_k, B_k` the correct / incorrect reference counts,
    :math:`C_k, D_k` the focal ones, :math:`n_{Rk}, n_{Fk}` the group sizes, :math:`m_{1k},
    m_{0k}` the numbers correct / incorrect and :math:`T_k` the stratum size (strata with
    :math:`T_k>1` only; Mantel & Haenszel, 1959; Holland & Thayer, 1988):

    .. math::
        \chi^2_{MH}=\frac{\big(\lvert\sum_kA_k-\sum_kE(A_k)\rvert-c\big)^2}{\sum_k\operatorname{Var}(A_k)},
        \quad E(A_k)=\frac{n_{Rk}m_{1k}}{T_k},\quad
        \operatorname{Var}(A_k)=\frac{n_{Rk}n_{Fk}m_{1k}m_{0k}}{T_k^2(T_k-1)},

    :math:`c=1/2` with the continuity correction (always, as Holland & Thayer, 1988, and difR;
    ``stats::mantelhaen.test`` drops it when the difference is below 1/2), else 0;
    :math:`\hat\alpha_{MH}=\sum_kA_kD_k/T_k\big/\sum_kB_kC_k/T_k` and the Phillips & Holland
    (1987) variance

    .. math::
        \widehat{\operatorname{Var}}(\ln\hat\alpha_{MH})=\frac{\sum_k(A_kD_k+\hat\alpha_{MH}B_kC_k)
        (A_k+D_k+\hat\alpha_{MH}(B_k+C_k))/T_k^2}{2\left(\sum_kA_kD_k/T_k\right)^2}.
    """
    inverse, n_strata = _strata(score)
    ref = member == 0
    correct = x == 1

    def count(mask: np.ndarray) -> np.ndarray:
        return np.bincount(inverse, weights=mask.astype(np.float64), minlength=n_strata)

    a = count(ref & correct)
    b = count(ref & ~correct)
    c = count(~ref & correct)
    d = count(~ref & ~correct)
    total = a + b + c + d
    keep = total > 1
    a, b, c, d, total = a[keep], b[keep], c[keep], d[keep], total[keep]
    n_ref = a + b
    n_foc = c + d
    m1 = a + c
    m0 = b + d
    expected = n_ref * m1 / total
    variance = n_ref * n_foc * m1 * m0 / (total**2 * (total - 1.0))
    diff = float(np.sum(a - expected))
    var_sum = float(np.sum(variance))
    yates = 0.5 if continuity_correction else 0.0
    stat = (abs(diff) - yates) ** 2 / var_sum if var_sum > 0 else np.nan
    ad = float(np.sum(a * d / total))
    bc = float(np.sum(b * c / total))
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = ad / bc if bc > 0 else (np.inf if ad > 0 else np.nan)
        if np.isfinite(alpha) and alpha > 0:
            var_log = float(
                np.sum((a * d + alpha * b * c) * (a + d + alpha * (b + c)) / total**2)
                / (2.0 * ad**2)
            )
        else:
            var_log = np.nan
    return {"chi2": float(stat), "odds_ratio": float(alpha), "var_log_odds_ratio": var_log}


def mantel_item(x: np.ndarray, member: np.ndarray, score: np.ndarray) -> float:
    r"""Mantel's (1963) statistic for an ordered item and two groups.

    With :math:`F_k=\sum_{i\in F,k}x_i` the sum of the focal scores in stratum :math:`k`,

    .. math::
        \chi^2_{M}=\frac{\big(\sum_kF_k-\sum_kE(F_k)\big)^2}{\sum_k\operatorname{Var}(F_k)},\quad
        E(F_k)=n_{Fk}\bar x_k,\quad
        \operatorname{Var}(F_k)=\frac{n_{Fk}n_{Rk}}{T_k^2(T_k-1)}
        \Big(T_k\sum_{i\in k}x_i^2-\big(\sum_{i\in k}x_i\big)^2\Big),

    over strata with both groups present (other strata add nothing); :math:`df=1`. For 0 / 1
    items it equals :math:`\chi^2_{MH}` without continuity correction.
    """
    inverse, n_strata = _strata(score)
    foc = (member == 1).astype(np.float64)
    n_foc = np.bincount(inverse, weights=foc, minlength=n_strata)
    total = np.bincount(inverse, minlength=n_strata).astype(np.float64)
    n_ref = total - n_foc
    sum_x = np.bincount(inverse, weights=x, minlength=n_strata)
    sum_x2 = np.bincount(inverse, weights=x * x, minlength=n_strata)
    sum_foc = np.bincount(inverse, weights=x * foc, minlength=n_strata)
    keep = (n_foc > 0) & (n_ref > 0) & (total > 1)
    t = total[keep]
    diff = float(np.sum(sum_foc[keep] - n_foc[keep] * sum_x[keep] / t))
    var = float(
        np.sum(
            n_foc[keep] * n_ref[keep] / (t**2 * (t - 1.0)) * (t * sum_x2[keep] - sum_x[keep] ** 2)
        )
    )
    return diff**2 / var if var > 0 else np.nan


def liu_agresti_item(x: np.ndarray, member: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    r"""Liu–Agresti (1996) cumulative common odds ratio and the standard error of its log.

    With the categories :math:`y_1<\dots<y_L` of the item and, in stratum :math:`k` and for
    :math:`j<L`, :math:`a_{jk}` / :math:`c_{jk}` the reference / focal counts with
    :math:`x\le y_j`, :math:`b_{jk}=n_{Rk}-a_{jk}`, :math:`d_{jk}=n_{Fk}-c_{jk}` (Liu & Agresti's
    eq. 2 with the focal group as their row 1, so :math:`X^*_{1jk}=c_{jk}`, :math:`X^*_{2jk}=
    a_{jk}`, :math:`n_{1k}=n_{Fk}`, :math:`n_{2k}=n_{Rk}`):

    .. math::
        \hat\psi=\frac{\sum_k\sum_jb_{jk}c_{jk}/T_k}{\sum_k\sum_ja_{jk}d_{jk}/T_k},

    oriented so that it reduces to :math:`\hat\alpha_{MH}` for 0 / 1 items (values above 1
    favour the reference group; difR's ``liu_agresti_ccor`` reports :math:`1/\hat\psi`). The
    variance of :math:`\ln\hat\psi` is Liu & Agresti's (1996) eq. 3, which generalizes Robins,
    Breslow & Greenland (1986) and is consistent both with a fixed number of large strata and
    with many sparse strata:

    .. math::
        \widehat{\operatorname{Var}}(\ln\hat\psi)=
        \frac{\sum_k\hat\xi_k}{\big(\sum_k\sum_jb_{jk}c_{jk}/T_k\big)^2},\quad
        \hat\xi_k=\sum_j\hat\phi_{jjk}+2\sum_{j<s}\hat\phi_{jsk},

    .. math::
        \hat\phi_{jsk}=\frac{n_{Fk}n_{Rk}}{T_k^2}\Big\{
        \frac{\hat\psi\,d_{sk}a_{jk}}{n_{Fk}}\Big[1+(\hat\psi-1)\frac{a_{sk}}{n_{Rk}}\Big]
        +\frac{c_{jk}b_{sk}}{n_{Rk}}\Big[\hat\psi-(\hat\psi-1)\frac{c_{sk}}{n_{Fk}}\Big]\Big\}
        \quad(j\le s).

    (Liu & Agresti's denominator :math:`\hat\theta^2(\sum S)^2` equals the one above.) For
    0 / 1 items it is a different dually consistent estimator from the Robins–Breslow–Greenland
    (= Phillips–Holland) variance of :func:`mantel_haenszel_item`; the two agree closely except in
    very small samples. Strata without both groups add nothing and are dropped.
    """
    cats = np.unique(x)
    n_cat = cats.size
    if n_cat < 2:
        return np.nan, np.nan
    inverse, n_strata = _strata(score)
    cat_index = np.searchsorted(cats, x)
    foc = member == 1
    ref_counts = np.zeros((n_strata, n_cat))
    foc_counts = np.zeros((n_strata, n_cat))
    np.add.at(ref_counts, (inverse[~foc], cat_index[~foc]), 1.0)
    np.add.at(foc_counts, (inverse[foc], cat_index[foc]), 1.0)
    n_ref = ref_counts.sum(axis=1)
    n_foc = foc_counts.sum(axis=1)
    keep = (n_ref > 0) & (n_foc > 0)
    ref_counts, foc_counts = ref_counts[keep], foc_counts[keep]
    n_ref, n_foc = n_ref[keep][:, None], n_foc[keep][:, None]
    total = n_ref + n_foc
    a = np.cumsum(ref_counts, axis=1)[:, :-1]
    c = np.cumsum(foc_counts, axis=1)[:, :-1]
    b = n_ref - a
    d = n_foc - c
    s_sum = float(np.sum(b * c / total))
    r_sum = float(np.sum(a * d / total))
    if s_sum <= 0 or r_sum <= 0:
        return (s_sum / r_sum if r_sum > 0 else np.nan), np.nan
    psi = s_sum / r_sum
    # phi[k, j, s]: the j index enters through a_j, c_j and the s index through a_s .. d_s.
    first = (psi * a / n_foc)[:, :, None] * (d * (1.0 + (psi - 1.0) * a / n_ref))[:, None, :]
    second = (c / n_ref)[:, :, None] * (b * (psi - (psi - 1.0) * c / n_foc))[:, None, :]
    phi = (n_foc * n_ref / total**2)[:, :, None] * (first + second)
    # xi_k = sum_j phi_jj + 2 sum_{j<s} phi_js: weight 1 on the diagonal, 2 above it.
    weight = np.triu(np.full((n_cat - 1, n_cat - 1), 2.0), k=1) + np.eye(n_cat - 1)
    var = float(np.sum(phi * weight))
    if not var > 0:
        return psi, np.nan
    return psi, float(np.sqrt(var) / s_sum)


def generalized_mh_item(
    x: np.ndarray, member: np.ndarray, score: np.ndarray, n_groups: int
) -> tuple[float, int]:
    r"""Generalized Mantel–Haenszel statistic for a groups x categories table in each stratum.

    With :math:`\mathbf n_k` the counts of the focal groups (rows :math:`2..G`) and the
    categories :math:`1..C-1` in stratum :math:`k` (strata with :math:`T_k>1`), row and column
    proportions :math:`\mathbf p_k, \mathbf q_k` (Landis, Heyman & Koch, 1978; Somes, 1986):

    .. math::
        \mathbf m_k=T_k\,(\mathbf p_k\otimes\mathbf q_k),\quad
        \mathbf V_k=\frac{T_k^2}{T_k-1}\big(D_{\mathbf p_k}-\mathbf p_k\mathbf p_k^\top\big)
        \otimes\big(D_{\mathbf q_k}-\mathbf q_k\mathbf q_k^\top\big),

    :math:`Q=\big(\sum_k\mathbf n_k-\mathbf m_k\big)^\top\big(\sum_k\mathbf V_k\big)^{-1}
    \big(\sum_k\mathbf n_k-\mathbf m_k\big)` with :math:`df=(G-1)(C-1)` (the rank of
    :math:`\sum_k\mathbf V_k`; a generalized inverse is used when it is singular). Returns
    ``(Q, df)``.
    """
    cats = np.unique(x)
    n_cat = cats.size
    if n_cat < 2:
        return np.nan, 0
    inverse, n_strata = _strata(score)
    cat_index = np.searchsorted(cats, x)
    counts = np.zeros((n_strata, n_groups, n_cat))
    np.add.at(counts, (inverse, member, cat_index), 1.0)
    total = counts.sum(axis=(1, 2))
    counts = counts[total > 1]
    total = total[total > 1]
    dim = (n_groups - 1) * (n_cat - 1)
    observed = np.zeros(dim)
    expected = np.zeros(dim)
    cov = np.zeros((dim, dim))
    for table, t in zip(counts, total, strict=True):
        p = table.sum(axis=1) / t
        q = table.sum(axis=0) / t
        observed += table[1:, :-1].ravel()
        expected += t * np.outer(p[1:], q[:-1]).ravel()
        row = (np.diag(p) - np.outer(p, p))[1:, 1:]
        col = (np.diag(q) - np.outer(q, q))[:-1, :-1]
        cov += t**2 / (t - 1.0) * np.kron(row, col)
    diff = observed - expected
    rank = int(np.linalg.matrix_rank(cov, hermitian=True))
    if rank == 0:
        return np.nan, 0
    if rank == dim:
        stat = float(diff @ np.linalg.solve(cov, diff))
    else:
        stat = float(diff @ np.linalg.pinv(cov, hermitian=True) @ diff)
    return stat, rank


def logistic_fit(
    y: np.ndarray, design: np.ndarray, *, max_iter: int = 100, tol: float = 1e-12
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    r"""Maximum-likelihood logistic regression by Newton–Raphson (= IRLS for the canonical link).

    :math:`\beta\leftarrow\beta+(X^\top WX)^{-1}X^\top(y-p)`, :math:`W=\operatorname{diag}
    (p(1-p))`, from :math:`\beta=0` until the step is below ``tol`` (max-norm, relative to
    :math:`1+\lvert\beta\rvert`). Returns the coefficients, their covariance
    :math:`(X^\top WX)^{-1}`, the deviance :math:`-2\sum[y\eta-\ln(1+e^\eta)]` and whether the
    iteration converged (it does not under complete or quasi-complete separation, where the
    information matrix becomes singular; a pseudo-inverse is then returned).
    """
    beta = np.zeros(design.shape[1])
    converged = False
    for _ in range(max_iter):
        eta = design @ beta
        prob = expit(eta)
        weight = prob * (1.0 - prob)
        hessian = design.T @ (design * weight[:, None])
        try:
            step = np.linalg.solve(hessian, design.T @ (y - prob))
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)):
            break
        beta = beta + step
        if np.max(np.abs(step) / (1.0 + np.abs(beta))) < tol:
            converged = True
            break
    eta = design @ beta
    prob = expit(eta)
    weight = prob * (1.0 - prob)
    information = design.T @ (design * weight[:, None])
    try:
        cov = np.linalg.inv(information)
    except np.linalg.LinAlgError:
        converged = False
        cov = np.linalg.pinv(information, hermitian=True)
    deviance = float(-2.0 * np.sum(y * eta - np.logaddexp(0.0, eta)))
    return beta, cov, deviance, converged


def chi2_sf(stat: np.ndarray, df: np.ndarray | int) -> np.ndarray:
    """Upper tail of the chi-square distribution (NaN where the statistic is NaN)."""
    return np.asarray(chi2_dist.sf(stat, df), dtype=np.float64)
