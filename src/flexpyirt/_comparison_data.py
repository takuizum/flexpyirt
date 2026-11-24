r"""Comparison data (Ruscio & Roche, 2012): population generation and sampling (DESIGN.md §16.15).

Pure NumPy: principal axis factoring, Ruscio & Kaczetow's (2008) iterative generation of a
population with ``k`` common factors that reproduces the observed correlations and marginal
distributions (EFAtools' ``.simulate_cfm_empirical``), and the eigenvalue RMSE of bootstrap
samples. Random inputs are arguments, so the generator is deterministic given them.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu

__all__ = ["eigenvalue_rmse", "generate_population", "paf_loadings", "wilcoxon_less"]


def paf_loadings(
    corr: np.ndarray, k: int, *, criterion: float = 1e-3, max_iter: int = 50
) -> np.ndarray:
    r"""Principal axis factoring from unit communalities (EFAtools' ``.paf_iter``).

    Starting from the matrix as given (unit diagonal), repeat: :math:`\Lambda=V_k|D_k|^{1/2}`
    from the leading eigenpairs (absolute eigenvalues, as SPSS), :math:`h^2=\operatorname{rowsum}
    (\Lambda^2)`, put :math:`h^2` on the diagonal, until the largest change of :math:`h^2` is at
    most ``criterion`` or ``max_iter`` passes. The loadings of the last pass are returned, each
    column signed to a non-negative sum.
    """
    mat = corr.copy()
    h2 = np.ones(mat.shape[0])
    loadings = np.zeros((mat.shape[0], k))
    for _ in range(max_iter):
        values, vectors = np.linalg.eigh(mat)
        top = values[::-1][:k]
        vecs = vectors[:, ::-1][:, :k]
        loadings = vecs * np.sqrt(np.abs(top))
        new_h2 = np.sum(loadings**2, axis=1)
        delta = float(np.max(np.abs(h2 - new_h2)))
        np.fill_diagonal(mat, new_h2)
        h2 = new_h2
        if delta <= criterion:
            break
    signs = np.where(loadings.sum(axis=0) < 0, -1.0, 1.0)
    return loadings * signs


def _build(
    loadings: np.ndarray, shared: np.ndarray, unique: np.ndarray, dists: np.ndarray
) -> np.ndarray:
    """Common-factor scores mapped onto the empirical marginals, column by column."""
    lam = np.clip(loadings, -1.0, 1.0)
    if lam[0, 0] < 0:
        lam = -lam
    h2 = np.sum(lam**2, axis=1)
    unique_load = np.sqrt(np.where(h2 < 1.0, 1.0 - h2, 0.0))
    sim: np.ndarray = shared @ lam.T + unique * unique_load
    for i in range(sim.shape[1]):
        sim = sim[np.argsort(sim[:, i], kind="stable")]
        sim[:, i] = dists[:, i]
    return sim


def _corr(x: np.ndarray) -> np.ndarray:
    out: np.ndarray = np.corrcoef(x, rowvar=False)
    return out


def generate_population(
    corr: np.ndarray,
    dists: np.ndarray,
    shared: np.ndarray,
    unique: np.ndarray,
    *,
    max_trials: int = 5,
    initial_multiplier: float = 1.0,
    max_iter: int = 50,
) -> tuple[np.ndarray, float]:
    r"""Generate a ``k``-factor population (Ruscio & Kaczetow, 2008).

    ``dists`` holds each variable's ``N`` resampled observed values sorted ascending, ``shared``
    ``(N, k)`` and ``unique`` ``(N, p)`` standard normal draws. An intermediate correlation matrix
    :math:`R^*` (initially the observed :math:`R`) is factored by :func:`paf_loadings`; the scores
    :math:`F\Lambda^\top+E\,\mathrm{diag}(\sqrt{1-h^2})` are mapped onto the marginals by rank, and
    :math:`R^*` is moved by the residual :math:`R-R_{\text{rep}}` while the RMSE of the residual
    correlations improves; after a failure the step halves from the best :math:`R^*`, and the
    search stops after ``max_trials`` failures in a row.

    Returns
    -------
    tuple of numpy.ndarray and float
        The ``(N, p)`` population and the RMSE of its correlations.
    """
    p = corr.shape[0]
    lower = np.tril_indices(p, k=-1)
    k = shared.shape[1]
    best_rmse = 1.0
    r_inter = corr.copy()
    r_best = r_inter
    res_best = np.zeros_like(corr)
    no_improvement = 0
    while no_improvement < max_trials:
        sim = _build(paf_loadings(r_inter, k, max_iter=max_iter), shared, unique, dists)
        residual = corr - _corr(sim)
        rmse = float(np.sqrt(np.sum(residual[lower] ** 2) / (0.5 * (p * p - p))))
        if rmse < best_rmse:
            best_rmse, r_best, res_best = rmse, r_inter, residual
            r_inter = r_inter + initial_multiplier * residual
            no_improvement = 0
        else:
            no_improvement += 1
            r_inter = r_best + initial_multiplier * 0.5**no_improvement * res_best
    sim = _build(paf_loadings(r_best, k, max_iter=max_iter), shared, unique, dists)
    return sim, best_rmse


def eigenvalue_rmse(samples: np.ndarray, observed: np.ndarray) -> np.ndarray:
    r"""RMSE :math:`\sqrt{\sum_j(\lambda_j^{(b)}-\lambda_j)^2/p}` of each sample's eigenvalues.

    ``samples`` is ``(B, n, p)``; eigenvalues of each sample's Pearson correlation matrix.
    """
    x = samples - samples.mean(axis=1, keepdims=True)
    cov = np.einsum("bnp,bnq->bpq", x, x)
    sd = np.sqrt(np.einsum("bpp->bp", cov))
    corr = cov / sd[:, :, None] / sd[:, None, :]
    values = np.linalg.eigvalsh(corr)[:, ::-1]
    out: np.ndarray = np.sqrt(np.mean((values - observed) ** 2, axis=1))
    return out


def wilcoxon_less(x: np.ndarray, y: np.ndarray) -> float:
    r"""One-sided rank-sum :math:`p` for "``x`` tends to be smaller than ``y``".

    Normal approximation with continuity and tie corrections, as R's ``wilcox.test(x, y,
    "less")`` for samples of 50 or more.
    """
    return float(
        mannwhitneyu(x, y, alternative="less", method="asymptotic", use_continuity=True).pvalue
    )
