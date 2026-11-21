"""Item fit: Orlando & Thissen's S-X2 (Result layer, DESIGN.md §15.40).

Post-fit bookkeeping on NumPy arrays (summed-score tables and cell collapsing are data dependent,
so they run outside ``jit``); the item category probabilities on the quadrature grid come from
the JAX item kernels.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "collapse_cells",
    "lord_wingersky",
    "observed_tables",
    "s_x2",
    "s_x2_expected_tables",
]


def lord_wingersky(traces: list[np.ndarray]) -> np.ndarray:
    r"""Summed-score likelihoods :math:`L(s\mid\theta_q)` by the Lord–Wingersky recursion.

    .. math::
        L_j(s\mid\theta) = \sum_{k=0}^{K_j-1} L_{j-1}(s-k\mid\theta)\,P_{jk}(\theta),
        \qquad L_0(0\mid\theta)=1

    (Lord & Wingersky, 1984), with item
    :math:`j` scored :math:`0,\dots,K_j-1`.

    Parameters
    ----------
    traces : list of (Q, K_j) arrays
        Category probabilities of each item on the quadrature nodes.

    Returns
    -------
    ndarray, shape (S + 1, Q)
        :math:`L(s\mid\theta_q)` for summed scores :math:`s = 0,\dots,S`, :math:`S=\sum_j(K_j-1)`.
    """
    n_nodes = traces[0].shape[0] if traces else 1
    lik = np.ones((1, n_nodes))
    for tr in traces:
        k_j = tr.shape[1]
        new = np.zeros((lik.shape[0] + k_j - 1, n_nodes))
        for k in range(k_j):
            new[k : k + lik.shape[0]] += lik * tr[:, k][None, :]
        lik = new
    return lik


def observed_tables(y: np.ndarray, n_cat: list[int]) -> list[np.ndarray]:
    """Observed counts ``O[s, k]`` of category ``k`` at summed score ``s`` for every item.

    Rows are summed scores ``1 .. S - 1`` (the minimum and maximum scores are dropped, as in
    mirt's ``makeObstables``); ``y`` holds complete responses coded ``0 .. K_j - 1``.
    """
    total = int(sum(k - 1 for k in n_cat))
    scores = y.sum(axis=1)
    out = []
    for j, k_j in enumerate(n_cat):
        tab = np.zeros((total + 1, k_j))
        np.add.at(tab, (scores, y[:, j]), 1.0)
        out.append(tab[1:-1])
    return out


def s_x2_expected_tables(traces: list[np.ndarray], weights: np.ndarray) -> list[np.ndarray]:
    r"""Model-implied proportions :math:`E[s,k]` of category :math:`k` at summed score :math:`s`.

    .. math::
        E_j[s,k] = \frac{\sum_q P_{jk}(\theta_q)\,L_{-j}(s-k\mid\theta_q)\,w_q}
                        {\sum_q L(s\mid\theta_q)\,w_q}

    (Orlando & Thissen, 2000, 2003), where :math:`L_{-j}` leaves item :math:`j` out and
    :math:`w_q` are the latent quadrature weights. Impossible cells (:math:`s-k` outside the
    range of :math:`L_{-j}`) are NaN. Rows are summed scores ``1 .. S - 1``.
    """
    lik_all = lord_wingersky(traces)
    denom = lik_all @ weights  # (S + 1,)
    out = []
    for j, tr in enumerate(traces):
        rest = lord_wingersky([t for i, t in enumerate(traces) if i != j])
        e = np.full((lik_all.shape[0], tr.shape[1]), np.nan)
        for k in range(tr.shape[1]):
            e[k : k + rest.shape[0], k] = (rest @ (tr[:, k] * weights)) / denom[
                k : k + rest.shape[0]
            ]
        out.append(e[1:-1])
    return out


def _merge_rows(o: np.ndarray, e: np.ndarray, src: int, dst: int) -> tuple[np.ndarray, np.ndarray]:
    o = o.copy()
    e = e.copy()
    o[dst] += o[src]
    e[dst] += e[src]
    return np.delete(o, src, axis=0), np.delete(e, src, axis=0)


def collapse_cells(
    o_in: np.ndarray, e_in: np.ndarray, mincell: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse sparse cells of one item's S-X2 table exactly as mirt's ``collapseCells``.

    A line-by-line port of ``mirt:::collapseCells`` (mirt 1.46–1.48) so that S-X2 and its
    degrees of freedom match mirt: (1) merge the leading/trailing rows that contain impossible
    (NaN) cells into their inner neighbours; (2) drop rows with no observations; (3) merge rows
    with a single observation into the next row; (4) for polytomous items, merge the smallest
    expected categories within rows that had a cell below ``mincell``; (5) merge sparse top
    categories and categories that are nonzero in fewer than 5% of rows; (6) merge rows with a
    nonzero expected cell below ``mincell`` into the neighbour with fewer observations. Cells
    whose expected count is 0 are returned as NaN (excluded from the statistic and its df).
    """
    o = np.array(o_in, dtype=float)
    e = np.array(e_in, dtype=float)

    # (1) rows with impossible cells: the first half are merged up, the second half down.
    drop = np.flatnonzero(np.isnan(e).any(axis=1))
    e = np.nan_to_num(e, nan=0.0)
    if drop.size:
        # R: up <- drop[1]:drop[n/2]; low <- drop[n/2 + 1]:drop[n] (fractional indices truncate)
        n_half = drop.size // 2
        up = np.arange(drop[0], drop[n_half - 1] + 1) if n_half >= 1 else np.array([], dtype=int)
        low = np.arange(drop[n_half], drop[-1] + 1)
        if up.size:
            tgt = up.max() + 1
            e[tgt] = e[np.r_[up, tgt]].sum(axis=0)
            o[tgt] = o[np.r_[up, tgt]].sum(axis=0)
        tgt = low.min() - 1
        e[tgt] = e[np.r_[low, tgt]].sum(axis=0)
        o[tgt] = o[np.r_[low, tgt]].sum(axis=0)
        keep = np.setdiff1d(np.arange(e.shape[0]), np.r_[up, low])
        e, o = e[keep], o[keep]
    if o.shape[0] == 0:
        return o, e

    # (2) rows without observations.
    nonempty = o.sum(axis=1) != 0
    o, e = o[nonempty], e[nonempty]
    low_cells = e < mincell  # computed here and reused below, as in mirt

    # (3) rows with a single observation merge into the next row.
    single = []
    for j in range(o.shape[0] - 1):
        if o[j].sum() == 1:
            single.append(j)
            o[j + 1] += o[j]
            e[j + 1] += e[j]
    if single:
        o = np.delete(o, single, axis=0)
        e = np.delete(e, single, axis=0)
    if o.shape[0] >= 2 and o[-1].sum() == 1:
        o, e = _merge_rows(o, e, o.shape[0] - 1, o.shape[0] - 2)

    n_col = e.shape[1]
    # (4) polytomous: merge the smallest expected categories within flagged rows.
    if n_col > 2:
        for j in range(e.shape[0]):
            if not low_cells[j].any():
                continue
            tmp = list(e[j])
            tmp2 = list(o[j])
            while len(tmp) > 2:
                m = min(tmp)
                whc = max(i for i, v in enumerate(tmp) if v == m)
                if whc == 0:
                    pick = 1
                elif whc == len(tmp) - 1:
                    pick = len(tmp) - 2
                else:
                    pick = whc - 1 if min(tmp[whc - 1], tmp[whc + 1]) == tmp[whc - 1] else whc + 1
                tmp[pick] += tmp[whc]
                tmp2[pick] += tmp2[whc]
                del tmp[whc]
                del tmp2[whc]
                if all(v >= mincell for v in tmp):
                    break
            e[j] = tmp + [np.nan] * (n_col - len(tmp))
            o[j] = tmp2 + [0.0] * (n_col - len(tmp2))
    e = np.nan_to_num(e, nan=0.0)

    # (5a) sparse top categories.
    if e.shape[1] > 2:
        while True:
            pick = e.sum(axis=0) < mincell * math.ceil(e.shape[0] * 0.1)
            if not pick[-1] or e.shape[1] == 2:
                break
            o[:, -2] += o[:, -1]
            e[:, -2] += e[:, -1]
            o, e = o[:, :-1], e[:, :-1]
    # (5b) categories nonzero in fewer than 5% of rows merge into the previous category.
    dropcol = np.zeros(e.shape[1], dtype=bool)
    for j in range(e.shape[1] - 1, 0, -1):
        if np.sum(e[:, j] > 0) / e.shape[0] < 0.05:
            dropcol[j] = True
            e[:, j - 1] += e[:, j]
            o[:, j - 1] += o[:, j]
    e, o = e[:, ~dropcol], o[:, ~dropcol]

    # (6) rows with a nonzero expected cell below mincell.
    while e.shape[0] > 1:
        flagged = ((e < mincell) & (e != 0)).any(axis=1)
        if not flagged.any():
            break
        whc = int(np.argmax(flagged))
        last = e.shape[0] - 1
        if whc == 0:
            o, e = _merge_rows(o, e, 0, 1)
        elif whc == last:
            o, e = _merge_rows(o, e, last, last - 1)
        else:
            ss_up, ss_down = o[whc - 1].sum(), o[whc + 1].sum()
            pick = whc - 1 if min(ss_up, ss_down) == ss_up else whc + 1
            o, e = _merge_rows(o, e, whc, pick)
    e = np.where(e == 0, np.nan, e)
    return o, e


def s_x2(
    o: np.ndarray, e: np.ndarray, n_free_item: int, n_free_latent: int, n_obs: int
) -> tuple[float, int, float, float]:
    r"""S-X2, its degrees of freedom, RMSEA and p-value for one collapsed table.

    .. math::
        S\text{-}X^2_j = \sum_{s,k}\frac{(O_{sk}-E_{sk})^2}{E_{sk}},\qquad
        df_j = \#\{E_{sk}\ \text{kept}\} - \#\text{rows} - p_j - p_{\text{latent}}

    with :math:`p_j` free parameters of item :math:`j` and :math:`p_{\text{latent}}` free latent
    parameters (mirt's convention). :math:`\mathrm{RMSEA}=\sqrt{(X^2/df-1)/(N-1)}` (0 when
    :math:`X^2<df`); the p-value is the upper :math:`\chi^2_{df}` tail. ``df <= 0`` gives
    ``df = 0`` and NaN statistics, as in mirt.
    """
    from scipy.stats import chi2

    kept = ~np.isnan(e)
    stat = float(np.sum(((o - e) ** 2 / np.where(kept, e, 1.0))[kept]))
    df = int(kept.sum() - e.shape[0] - n_free_item - n_free_latent)
    if df <= 0:
        return math.nan, 0, math.nan, math.nan
    rmsea = math.sqrt((stat / df - 1.0) / (n_obs - 1)) if stat / df - 1.0 > 0 else 0.0
    return stat, df, rmsea, float(chi2.sf(stat, df))
