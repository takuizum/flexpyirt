r"""DIMTEST: Stout's test of essential unidimensionality (DESIGN.md §16.16).

Pure NumPy. The statistic is Nandakumar & Stout's (1993) :math:`T_L`; its bias is removed with
Stout, Froelich & Gao's (2001) bootstrap from kernel-smoothed unidimensional item response
functions (Ramsay, 1991). Random draws are arguments, so everything is deterministic given them.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtri

__all__ = ["kernel_irfs", "simulate_from_irfs", "t_l_statistic"]


def t_l_statistic(at: np.ndarray, pt_score: np.ndarray, min_group: int) -> tuple[float, int]:
    r"""Nandakumar & Stout's (1993) :math:`T_L` for binary assessment-test items.

    Respondents are grouped by the partitioning-test score; groups with fewer than
    ``min_group`` respondents are dropped. In group :math:`k` (size :math:`J_k`), with
    :math:`Y` the AT score, :math:`\hat p_{ik}` the AT item proportions,

    .. math::
        \hat\sigma_k^2=\frac{1}{J_k}\sum_j(Y_j-\bar Y_k)^2,\quad
        \hat\sigma_{U,k}^2=\sum_i\hat p_{ik}(1-\hat p_{ik}),\quad
        S_k^2=\frac{\hat\mu_{4,k}-\hat\sigma_k^4+\hat\delta_{4,k}}{J_k},

    :math:`\hat\mu_{4,k}=\frac1{J_k}\sum_j(Y_j-\bar Y_k)^4`,
    :math:`\hat\delta_{4,k}=\sum_i\hat p_{ik}(1-\hat p_{ik})(1-2\hat p_{ik})^2`, and

    .. math::
        T_L=\frac{1}{\sqrt K}\sum_{k}\frac{\hat\sigma_k^2-\hat\sigma_{U,k}^2}{S_k}

    over the :math:`K` retained groups. :math:`\hat\sigma_k^2-\hat\sigma_{U,k}^2` is twice the sum
    of the AT items' covariances given the PT score, positive when the AT measures something the
    PT does not. Groups with :math:`S_k=0` are dropped too.

    Returns
    -------
    tuple of float and int
        :math:`T_L` (NaN with no usable group) and :math:`K`.
    """
    y = at.sum(axis=1)
    total = 0.0
    k_used = 0
    for score in np.unique(pt_score):
        rows = pt_score == score
        n_k = int(rows.sum())
        if n_k < min_group:
            continue
        yk = y[rows]
        dev = yk - yk.mean()
        var = float(np.mean(dev**2))
        mu4 = float(np.mean(dev**4))
        p = at[rows].mean(axis=0)
        var_u = float(np.sum(p * (1 - p)))
        delta4 = float(np.sum(p * (1 - p) * (1 - 2 * p) ** 2))
        s2 = (mu4 - var**2 + delta4) / n_k
        if s2 <= 0:
            continue
        total += (var - var_u) / np.sqrt(s2)
        k_used += 1
    if k_used == 0:
        return float("nan"), 0
    return total / np.sqrt(k_used), k_used


def kernel_irfs(
    x: np.ndarray, tie_noise: np.ndarray, grid: np.ndarray, bandwidth: float
) -> np.ndarray:
    r"""Kernel-smoothed item response functions (Ramsay, 1991).

    Ability estimates are normal scores of the total score ranks,
    :math:`\hat\theta_j=\Phi^{-1}(r_j/(N+1))`, with ties broken by ``tie_noise`` (uniform draws);
    each item's IRF on ``grid`` is the Nadaraya–Watson regression with a Gaussian kernel of
    width ``bandwidth``.

    Returns
    -------
    numpy.ndarray
        ``(len(grid), J)`` response probabilities.
    """
    n = x.shape[0]
    order = np.lexsort((tie_noise, x.sum(axis=1)))
    ranks = np.empty(n)
    ranks[order] = np.arange(1, n + 1)
    theta = ndtri(ranks / (n + 1))
    w = np.exp(-0.5 * ((grid[:, None] - theta[None, :]) / bandwidth) ** 2)
    out: np.ndarray = (w @ x) / w.sum(axis=1, keepdims=True)
    return out


def simulate_from_irfs(
    irfs: np.ndarray, grid: np.ndarray, theta: np.ndarray, uniform: np.ndarray
) -> np.ndarray:
    r"""Draw binary responses with :math:`P(X_i=1\mid\theta)` interpolated from ``irfs``."""
    probs = np.column_stack([np.interp(theta, grid, irfs[:, i]) for i in range(irfs.shape[1])])
    out: np.ndarray = (uniform < probs).astype(np.float64)
    return out
