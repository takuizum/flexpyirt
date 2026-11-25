r"""Local-dependence statistics of a fitted model (DESIGN.md §16.11).

Pure NumPy helpers used by :meth:`flexpyirt.FitResult.local_dependence`: Yen's (1984) Q3 from
expected-score residuals, Chen & Thissen's (1997) LD :math:`X^2` / :math:`G^2` from model-implied
bivariate tables, and the Goodman–Kruskal :math:`\gamma` that signs them (as mirt's
``residuals(type = "LD")``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2

__all__ = ["goodman_kruskal_gamma", "ld_pairs", "q3_matrix"]


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


def ld_pairs(y: np.ndarray, traces: list[np.ndarray], weights: np.ndarray) -> dict[str, np.ndarray]:
    r"""Chen & Thissen's (1997) LD statistics for every pair :math:`i<j`.

    With :math:`n_{ij}` respondents answering both items, the expected table is
    :math:`E_{ab}=n_{ij}\sum_q P_{ia}(\theta_q)P_{jb}(\theta_q)w_q` and

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
    traces : list of numpy.ndarray
        Per item, ``(Q, K_j)`` category probabilities on the quadrature nodes (the product grid for
        several factors).
    weights : numpy.ndarray
        ``(Q,)`` latent weights summing to 1.

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
            ki, kj = traces[i].shape[1], traces[j].shape[1]
            observed = np.zeros((ki, kj))
            np.add.at(observed, (y[both, i], y[both, j]), 1.0)
            n = float(both.sum())
            expected = n * (traces[i] * weights[:, None]).T @ traces[j]
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
