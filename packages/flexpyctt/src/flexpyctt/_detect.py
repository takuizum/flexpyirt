r"""DETECT: conditional covariances and indices (DESIGN.md §16.13).

Pure NumPy helpers for :func:`flexpyctt.detect` and :func:`flexpyctt.exploratory_detect`.
Conditional covariances follow Zhang & Stout (1999): the average of the covariance conditional
on the total score and on the rest score, as sirt's ``ccov.np(use_sum_score = TRUE,
bias_corr = TRUE)``.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

__all__ = ["conditional_covariances", "detect_indices", "ward_clusters"]

INDEX_NAMES: tuple[str, ...] = ("DETECT", "ASSI", "RATIO", "MADCOV100", "MCOV100")


def _grouped_covariance(group: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    r"""Return :math:`\sum_s (n_s/n)\,\widehat{\mathrm{Cov}}_{ML}(x,y\mid g=s)`."""
    _, idx = np.unique(group, return_inverse=True)
    n_s = np.bincount(idx).astype(np.float64)
    mx = np.bincount(idx, weights=x) / n_s
    my = np.bincount(idx, weights=y) / n_s
    mxy = np.bincount(idx, weights=x * y) / n_s
    return float(np.sum(n_s * (mxy - mx * my)) / x.size)


def conditional_covariances(x: np.ndarray) -> np.ndarray:
    r"""Estimate the conditional covariance of every item pair (Zhang & Stout, 1999).

    With total score :math:`S=\sum_j X_j` and rest score :math:`S_{-ij}=S-X_i-X_j`,

    .. math::
        \widehat{\mathrm{ccov}}_{ij}=\tfrac12\Big[\sum_s\frac{n_s}{n}
        \widehat{\mathrm{Cov}}(X_i,X_j\mid S=s)+\sum_s\frac{n'_s}{n}
        \widehat{\mathrm{Cov}}(X_i,X_j\mid S_{-ij}=s)\Big],

    with maximum-likelihood (divide-by-:math:`n_s`) covariances within score groups. The
    total-score estimate is biased negatively and the rest-score one positively; their average
    reduces the bias.

    Parameters
    ----------
    x : numpy.ndarray
        ``(N, J)`` complete item scores.

    Returns
    -------
    numpy.ndarray
        ``(J, J)`` symmetric matrix of conditional covariances, zero diagonal.
    """
    n_items = x.shape[1]
    total = x.sum(axis=1)
    out = np.zeros((n_items, n_items))
    for i in range(n_items):
        for j in range(i + 1, n_items):
            xi, xj = x[:, i], x[:, j]
            given_total = _grouped_covariance(total, xi, xj)
            given_rest = _grouped_covariance(total - xi - xj, xi, xj)
            out[i, j] = out[j, i] = 0.5 * (given_total + given_rest)
    return out


def detect_indices(
    ccov: np.ndarray, clusters: np.ndarray, n_pairs: np.ndarray | None = None
) -> dict[str, tuple[float, float]]:
    r"""DETECT, ASSI, RATIO, MADCOV100 and MCOV100, unweighted and weighted.

    With :math:`\delta_{ij}=1` for items in the same cluster and :math:`-1` otherwise, over the
    :math:`\binom J2` pairs (Zhang & Stout, 1999; Zhang, 2007):
    :math:`\mathrm{DETECT}=100\,\overline{\delta\,\mathrm{ccov}}`,
    :math:`\mathrm{ASSI}=\overline{\delta\operatorname{sign}\mathrm{ccov}}`,
    :math:`\mathrm{RATIO}=\sum\delta\,\mathrm{ccov}/\sum|\mathrm{ccov}|`,
    :math:`\mathrm{MADCOV100}=100\,\overline{|\mathrm{ccov}|}`,
    :math:`\mathrm{MCOV100}=100\,\overline{\mathrm{ccov}}`. The weighted versions weight pairs by
    :math:`\sqrt{n_{ij}}` (sirt's ``detect.index``); with complete data they coincide.

    Returns
    -------
    dict of str to tuple of float
        ``{name: (unweighted, weighted)}``.
    """
    upper = np.triu_indices(ccov.shape[0], k=1)
    c = ccov[upper]
    same = clusters[upper[0]] == clusters[upper[1]]
    delta = np.where(same, 1.0, -1.0)
    w = np.ones_like(c) if n_pairs is None else np.sqrt(n_pairs[upper].astype(np.float64))

    def both(values: np.ndarray) -> tuple[float, float]:
        return float(np.mean(values)), float(np.sum(values * w) / np.sum(w))

    abs_c = np.abs(c)
    ratio = (
        float(np.sum(c * delta) / np.sum(abs_c)),
        float(np.sum(c * delta * w) / np.sum(abs_c * w)),
    )
    detect = both(100.0 * c * delta)
    return {
        "DETECT": detect,
        "ASSI": both(np.sign(c) * delta),
        "RATIO": ratio,
        "MADCOV100": both(100.0 * abs_c),
        "MCOV100": both(100.0 * c),
    }


def ward_clusters(ccov: np.ndarray, n_clusters: list[int]) -> dict[int, np.ndarray]:
    r"""Ward clustering of items on :math:`d_{ij}=\max(\mathrm{ccov})-\mathrm{ccov}_{ij}`.

    This is sirt's ``expl.detect``: R's ``hclust(as.dist(d), method = "ward.D")``, which applies
    the Lance–Williams Ward update to :math:`d` itself. SciPy's ``"ward"`` applies it to squared
    distances, so it is given :math:`\sqrt d`. Clusters are numbered by first appearance in item
    order, as R's ``cutree``.

    Returns
    -------
    dict of int to numpy.ndarray
        For each requested number of clusters, the cluster labels ``1..k`` of the items.
    """
    dist = np.max(ccov) - ccov
    upper = np.triu_indices(ccov.shape[0], k=1)
    tree = linkage(np.sqrt(np.maximum(dist[upper], 0.0)), method="ward")
    out: dict[int, np.ndarray] = {}
    for k in n_clusters:
        raw = fcluster(tree, t=k, criterion="maxclust")
        order: dict[int, int] = {}
        for label in raw:
            order.setdefault(int(label), len(order) + 1)
        out[k] = np.array([order[int(label)] for label in raw], dtype=np.int64)
    return out
