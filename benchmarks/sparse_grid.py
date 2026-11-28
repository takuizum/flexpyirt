r"""Smolyak sparse-grid quadrature for the standard multivariate normal (study code, not shipped).

Pure NumPy. Not part of the ``flexpyirt`` package and not wired into ``Quadrature`` or ``fit()``:
the design study in DESIGN.md §12 (open question 6) rejected Smolyak grids for BA-EM because of
their signed weights. Kept so that the study (``benchmarks/sparse_grid_study.py``) can be rerun and
the question reopened from tested code. Nodes and weights match R ``SparseGrid``'s ``"GQN"`` rule
(``tests/test_sparse_grid.py``).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable

import numpy as np

__all__ = ["gauss_hermite_rule", "smolyak_normal", "total_degree_exact"]

Rule = tuple[np.ndarray, np.ndarray]


def gauss_hermite_rule(level: int) -> Rule:
    r"""Return the ``level``-point Gauss–Hermite rule for :math:`N(0, 1)`.

    Nodes :math:`\sqrt2 x_i` and weights :math:`w_i/\sqrt\pi` from the rule for :math:`e^{-x^2}`
    (Golub & Welsch, 1969), exact for polynomials of degree :math:`2\,\text{level}-1`. This is the
    non-nested "GQN" sequence of Heiss & Winschel (2008).
    """
    if level < 1:
        raise ValueError(f"level must be >= 1, got {level}.")
    x, w = np.polynomial.hermite.hermgauss(level)
    nodes = math.sqrt(2.0) * x
    nodes[np.abs(nodes) < 1e-14] = 0.0
    return nodes, w / math.sqrt(math.pi)


def smolyak_normal(
    n_factors: int,
    level: int,
    *,
    rule: Callable[[int], Rule] = gauss_hermite_rule,
    decimals: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return Smolyak nodes ``(Q, D)`` and signed weights ``(Q,)`` for :math:`N(0, I_D)`.

    Combination technique (Smolyak, 1963; Wasilkowski & Woźniakowski, 1995; Heiss & Winschel,
    2008) with accuracy level :math:`L` (``level``) in :math:`D` dimensions:

    .. math::
        A_{L,D} = \sum_{L \le |\mathbf i| \le L + D - 1}
        (-1)^{L + D - 1 - |\mathbf i|}\binom{D - 1}{L + D - 1 - |\mathbf i|}
        \, U^{i_1}\otimes\cdots\otimes U^{i_D}, \qquad i_m \ge 1,

    where :math:`U^i` is the 1-D rule of level :math:`i`. With Gauss–Hermite rules
    (:math:`U^i` exact to degree :math:`2i-1`) the result is exact for every polynomial of total
    degree :math:`\le 2L-1`. Identical nodes from different tensor terms are merged (rounded to
    ``decimals``) and their weights summed. The weights sum to 1 but some are negative for
    :math:`L \ge 2`, :math:`D \ge 2`.
    """
    if n_factors < 1:
        raise ValueError(f"n_factors must be >= 1, got {n_factors}.")
    if level < 1:
        raise ValueError(f"level must be >= 1, got {level}.")
    rules = {i: rule(i) for i in range(1, level + 1)}
    acc: dict[tuple[float, ...], float] = {}
    q = level + n_factors - 1
    for total in range(max(n_factors, level), q + 1):
        coef = (-1) ** (q - total) * math.comb(n_factors - 1, q - total)
        for index in _compositions(total, n_factors):
            if max(index) > level:
                continue
            axes = [rules[i] for i in index]
            for combo in itertools.product(*(range(len(a[0])) for a in axes)):
                point = tuple(
                    round(float(axes[m][0][c]), decimals) + 0.0 for m, c in enumerate(combo)
                )
                weight = coef * math.prod(float(axes[m][1][c]) for m, c in enumerate(combo))
                acc[point] = acc.get(point, 0.0) + weight
    points = np.array(sorted(acc), dtype=np.float64).reshape(-1, n_factors)
    weights = np.array([acc[tuple(p)] for p in points.tolist()], dtype=np.float64)
    keep = np.abs(weights) > 1e-15
    return points[keep], weights[keep]


def total_degree_exact(level: int) -> int:
    """Return the total degree :math:`2L-1` integrated exactly by Gauss–Hermite Smolyak."""
    return 2 * level - 1


def _compositions(total: int, parts: int) -> list[tuple[int, ...]]:
    """Return all tuples of ``parts`` positive integers summing to ``total``."""
    out = []
    for cuts in itertools.combinations(range(1, total), parts - 1):
        bounds = (0, *cuts, total)
        out.append(tuple(bounds[k + 1] - bounds[k] for k in range(parts)))
    return out
