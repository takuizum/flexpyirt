r"""Local-dependence statistics of a fitted model (DESIGN.md §16.11).

Pure NumPy helpers used by :meth:`flexpyirt.FitResult.local_dependence`: Yen's (1984) Q3 from
expected-score residuals, Chen & Thissen's (1997) LD :math:`X^2` / :math:`G^2` from model-implied
bivariate tables, and the Goodman–Kruskal :math:`\gamma` that signs them (as mirt's
``residuals(type = "LD")``).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import chi2

__all__ = [
    "goodman_kruskal_gamma",
    "ld_pairs",
    "ld_pairs_from_tables",
    "q3_matrix",
    "reduced_pair_probabilities",
]


def goodman_kruskal_gamma(table: np.ndarray) -> float:
    r"""Goodman–Kruskal :math:`\gamma=(C-D)/(C+D)` of a two-way table of counts.

    :math:`C=\sum_{a,b} n_{ab}\sum_{a'>a,b'>b}n_{a'b'}` (concordant pairs) and
    :math:`D=\sum_{a,b} n_{ab}\sum_{a'>a,b'<b}n_{a'b'}` (discordant pairs). NaN when
    :math:`C+D=0`.
    """
    t = np.asarray(table, dtype=np.float64)
    # below_right[a, b] = sum_{a' > a, b' > b} t[a', b'];  below_left: b' < b.
    rev = t[::-1, ::-1].cumsum(axis=0).cumsum(axis=1)[::-1, ::-1]
    below_right = np.zeros_like(t)
    below_right[:-1, :-1] = rev[1:, 1:]
    rev_left = t[::-1, :].cumsum(axis=0)[::-1, :].cumsum(axis=1)
    below_left = np.zeros_like(t)
    below_left[:-1, 1:] = rev_left[1:, :-1]
    concordant = float(np.sum(t * below_right))
    discordant = float(np.sum(t * below_left))
    total = concordant + discordant
    return (concordant - discordant) / total if total > 0 else float("nan")


def q3_matrix(residuals: np.ndarray) -> np.ndarray:
    r"""Yen's (1984) Q3: pairwise-complete Pearson correlations of residuals.

    ``residuals`` is ``(N, J)`` with :math:`d_{ij}=x_{ij}-E(X_j\mid\hat\theta_i)`, NaN where
    missing. The diagonal is 1.
    """
    mat = (
        pd.DataFrame(residuals)
        .corr(method="pearson", min_periods=2)
        .to_numpy(dtype=np.float64, copy=True)
    )
    np.fill_diagonal(mat, 1.0)
    return mat


def ld_pairs_from_tables(
    y: np.ndarray, pair_probabilities: Callable[[int, int], np.ndarray]
) -> dict[str, np.ndarray]:
    r"""Chen & Thissen's (1997) LD statistics for every pair :math:`i<j`.

    With :math:`n_{ij}` respondents answering both items and the model-implied joint
    probabilities :math:`\pi_{ab}` of the pair, :math:`E_{ab}=n_{ij}\pi_{ab}` and

    .. math::
        X^2=\sum_{a,b}\frac{(O_{ab}-E_{ab})^2}{E_{ab}},\qquad
        G^2=2\sum_{O_{ab}>0}O_{ab}\log\frac{O_{ab}}{E_{ab}},\qquad
        V=\operatorname{sign}(\gamma_O-\gamma_E)\sqrt{\frac{X^2}{n_{ij}\min(K_i-1,K_j-1)}}

    with :math:`df=(K_i-1)(K_j-1)` and the sign from the difference of Goodman–Kruskal
    :math:`\gamma` of the observed and expected tables (positive when it is 0; :math:`V` is NaN when
    a :math:`\gamma` is undefined), as in mirt's ``residuals(type = "LD")``.

    Parameters
    ----------
    y : numpy.ndarray
        ``(N, J)`` responses ``0..K_j-1``, ``-1`` for missing.
    pair_probabilities : callable
        ``(i, j) -> (K_i, K_j)`` joint category probabilities of items ``i < j`` under the model.

    Returns
    -------
    dict of str to numpy.ndarray
        ``i``, ``j``, ``n``, ``x2``, ``g2``, ``df``, ``p_x2``, ``p_g2``, ``cramer_v``, one entry
        per pair.
    """
    n_items = y.shape[1]
    out: dict[str, list[float]] = {
        k: [] for k in ("i", "j", "n", "x2", "g2", "df", "p_x2", "p_g2", "cramer_v")
    }
    for i in range(n_items):
        for j in range(i + 1, n_items):
            both = (y[:, i] >= 0) & (y[:, j] >= 0)
            n = float(both.sum())
            expected = n * pair_probabilities(i, j)
            ki, kj = expected.shape
            observed = np.zeros((ki, kj))
            np.add.at(observed, (y[both, i], y[both, j]), 1.0)
            x2 = float(np.sum((observed - expected) ** 2 / expected))
            pos = observed > 0
            g2 = float(2.0 * np.sum(observed[pos] * np.log(observed[pos] / expected[pos])))
            df = (ki - 1) * (kj - 1)
            diff = goodman_kruskal_gamma(observed) - goodman_kruskal_gamma(expected)
            sign = float("nan") if not np.isfinite(diff) else (1.0 if diff == 0 else np.sign(diff))
            out["i"].append(i)
            out["j"].append(j)
            out["n"].append(n)
            out["x2"].append(x2)
            out["g2"].append(g2)
            out["df"].append(df)
            out["p_x2"].append(float(chi2.sf(x2, df)))
            out["p_g2"].append(float(chi2.sf(g2, df)))
            out["cramer_v"].append(sign * float(np.sqrt(x2 / (n * min(ki - 1, kj - 1)))))
    return {k: np.asarray(v) for k, v in out.items()}


def ld_pairs(y: np.ndarray, traces: list[np.ndarray], weights: np.ndarray) -> dict[str, np.ndarray]:
    r"""LD statistics of :func:`ld_pairs_from_tables` for items on one shared grid.

    The joint probabilities are
    :math:`\pi_{ab}=\sum_q P_{ia}(\theta_q)P_{jb}(\theta_q)w_q`.

    Parameters
    ----------
    y : numpy.ndarray
        ``(N, J)`` responses ``0..K_j-1``, ``-1`` for missing.
    traces : list of numpy.ndarray
        Per item, ``(Q, K_j)`` category probabilities on the quadrature nodes (the product grid for
        several factors).
    weights : numpy.ndarray
        ``(Q,)`` latent weights summing to 1.

    Returns
    -------
    dict of str to numpy.ndarray
        As :func:`ld_pairs_from_tables`.
    """

    def on_shared_grid(i: int, j: int) -> np.ndarray:
        table: np.ndarray = (traces[i] * weights[:, None]).T @ traces[j]
        return table

    return ld_pairs_from_tables(y, on_shared_grid)


def reduced_pair_probabilities(
    traces: list[np.ndarray],
    item_block: np.ndarray,
    primary_weights: np.ndarray,
    block_weights: np.ndarray,
) -> Callable[[int, int], np.ndarray]:
    r"""Joint probabilities of item pairs in a bifactor or two-tier model (DESIGN.md §17.3).

    With primary nodes :math:`\boldsymbol\theta_g` (weights :math:`w^{(0)}_g`), specific nodes
    :math:`\theta_t` (weights :math:`w^{(s)}_t`),

    .. math::
        \bar P_{ja}(\boldsymbol\theta_g)=\sum_t w^{(s(j))}_tP_{ja}(\boldsymbol\theta_g,\theta_t),

    .. math::
        \pi_{ab}=\sum_g w^{(0)}_g\sum_t w^{(s)}_tP_{ia}(\boldsymbol\theta_g,\theta_t)
        P_{jb}(\boldsymbol\theta_g,\theta_t)\quad(s(i)=s(j)=s),\qquad
        \pi_{ab}=\sum_g w^{(0)}_g\bar P_{ia}(\boldsymbol\theta_g)\bar P_{jb}(\boldsymbol\theta_g)
        \quad(\text{otherwise}),

    which equals the sum over the full product grid because the specific factors are independent
    of each other and of the primary factors (Gibbons & Hedeker, 1992; Cai, 2010).

    Parameters
    ----------
    traces : list of numpy.ndarray
        Per item, ``(Q_p, n, K_j)`` category probabilities on the item's block grid.
    item_block : numpy.ndarray
        ``(J,)`` block of each item: ``0..S-1`` its specific factor, ``S`` for none.
    primary_weights : numpy.ndarray
        ``(Q_p,)`` weights of the primary nodes, summing to 1.
    block_weights : numpy.ndarray
        ``(S + 1, n)`` weights of each block's specific nodes (uniform for block ``S``), each
        row summing to 1.

    Returns
    -------
    callable
        ``(i, j) -> (K_i, K_j)`` joint probabilities.
    """
    given_primary = [
        np.einsum("gtk,t->gk", trace, block_weights[int(block)])
        for trace, block in zip(traces, item_block, strict=True)
    ]

    def on_block_grids(i: int, j: int) -> np.ndarray:
        table: np.ndarray
        if item_block[i] == item_block[j]:
            joint = np.outer(primary_weights, block_weights[int(item_block[i])])
            table = np.einsum("gt,gta,gtb->ab", joint, traces[i], traces[j])
        else:
            table = (given_primary[i] * primary_weights[:, None]).T @ given_primary[j]
        return table

    return on_block_grids
