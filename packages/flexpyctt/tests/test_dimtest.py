"""D6: DIMTEST (DESIGN.md §16.16, VALIDATION.md V30).

There is no reference implementation outside DIMPACK, so DIMTEST is accepted on a simulation
study (``validation/simulation/dimtest_study.py``); these tests check the pieces against direct
computations and run a small version of the study.
"""

from __future__ import annotations

from typing import Any

import jax
import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtri

from flexpyirt import dimtest
from flexpyirt._dimtest import kernel_irfs, simulate_from_irfs, t_l_statistic
from flexpyirt.errors import SpecificationError


def _simulate(seed: int, n: int, n_items: int, rho: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = rng.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], size=n)
    a = rng.uniform(0.8, 1.8, n_items)
    d = rng.normal(0.0, 0.8, n_items)
    dim = np.repeat([0, 1], n_items // 2)
    return (rng.uniform(size=(n, n_items)) < 1 / (1 + np.exp(-(a * theta[:, dim] + d)))).astype(
        float
    )


def test_t_l_statistic_matches_direct_formula() -> None:
    x = _simulate(0, 800, 10, 0.6)
    at, pt = x[:, :3], x[:, 3:].sum(axis=1)
    got, k = t_l_statistic(at, pt, min_group=20)
    total, used = 0.0, 0
    for s in np.unique(pt):
        g = at[pt == s]
        if g.shape[0] < 20:
            continue
        y = g.sum(axis=1)
        var = y.var()
        mu4 = np.mean((y - y.mean()) ** 4)
        p = g.mean(axis=0)
        s2 = (mu4 - var**2 + np.sum(p * (1 - p) * (1 - 2 * p) ** 2)) / g.shape[0]
        total += (var - np.sum(p * (1 - p))) / np.sqrt(s2)
        used += 1
    assert k == used
    assert got == pytest.approx(total / np.sqrt(used), rel=1e-12)


def test_t_l_statistic_without_groups_is_nan() -> None:
    x = _simulate(1, 30, 6, 1.0)
    value, k = t_l_statistic(x[:, :2], x[:, 2:].sum(axis=1), min_group=100)
    assert k == 0 and np.isnan(value)


def test_kernel_irfs_are_smoothed_proportions() -> None:
    x = _simulate(2, 3000, 8, 1.0)
    noise = np.random.default_rng(0).uniform(size=3000)
    grid = np.linspace(-4, 4, 161)
    irfs = kernel_irfs(x, noise, grid, 1.1 * 3000 ** (-0.2))
    assert irfs.shape == (161, 8)
    assert ((irfs >= -1e-12) & (irfs <= 1 + 1e-12)).all()
    order = np.lexsort((noise, x.sum(axis=1)))
    ranks = np.empty(3000)
    ranks[order] = np.arange(1, 3001)
    theta = ndtri(ranks / 3001)
    w = np.exp(-0.5 * ((0.3 - theta) / (1.1 * 3000 ** (-0.2))) ** 2)
    direct = w @ x / w.sum()
    np.testing.assert_allclose(
        np.array([np.interp(0.3, grid, irfs[:, i]) for i in range(8)]), direct, atol=2e-3
    )


def test_simulate_from_irfs_uses_the_interpolated_probabilities() -> None:
    grid = np.array([-1.0, 1.0])
    irfs = np.array([[0.2, 0.6], [0.4, 0.8]])
    theta = np.zeros(100000)
    uniform = np.random.default_rng(1).uniform(size=(100000, 2))
    sim = simulate_from_irfs(irfs, grid, theta, uniform)
    np.testing.assert_allclose(sim.mean(axis=0), [0.3, 0.7], atol=5e-3)


def test_dimtest_detects_two_dimensions_and_keeps_one() -> None:
    two = dimtest(_simulate(3, 2000, 20, 0.5), key=jax.random.key(0))
    assert two.p_value < 1e-6
    assert set(two.assessment_items) <= {f"item{i}" for i in range(1, 11)} or set(
        two.assessment_items
    ) <= {f"item{i}" for i in range(11, 21)}
    one = dimtest(_simulate(4, 2000, 20, 1.0), key=jax.random.key(0))
    assert one.p_value > 0.01
    assert one.training_rows.size == 2000 // 3
    assert np.intersect1d(one.training_rows, one.test_rows).size == 0
    assert one.t_g.size == 100
    assert one.statistic == pytest.approx(
        (one.t_l - one.t_g.mean()) / np.sqrt(1 + 1 / one.t_g.size)
    )


def test_dimtest_with_given_assessment_items_uses_all_rows() -> None:
    data = pd.DataFrame(_simulate(5, 1500, 12, 0.5), columns=[f"q{i}" for i in range(12)])
    res = dimtest(data, key=jax.random.key(1), assessment_items=["q0", "q1", "q2", "q3"])
    assert res.assessment_items == ["q0", "q1", "q2", "q3"]
    assert res.training_rows.size == 0 and res.test_rows.size == 1500
    assert res.p_value < 1e-3


def test_dimtest_is_reproducible() -> None:
    data = _simulate(6, 900, 12, 1.0)
    first = dimtest(data, key=jax.random.key(2), n_bootstrap=20)
    second = dimtest(data, key=jax.random.key(2), n_bootstrap=20)
    assert first.statistic == second.statistic
    np.testing.assert_array_equal(first.t_g, second.t_g)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_bootstrap": 1}, "n_bootstrap"),
        ({"min_group": 1}, "min_group"),
        ({"training_fraction": 1.0}, "training_fraction"),
        ({"max_assessment_size": 1}, "max_assessment_size"),
        ({"assessment_items": ["item1"]}, "at least 2"),
        ({"assessment_items": ["zz", "item1"]}, "Unknown"),
        ({"min_group": 10**6, "assessment_items": ["item1", "item2"]}, "min_group"),
    ],
)
def test_invalid_options(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(SpecificationError, match=match):
        dimtest(_simulate(7, 300, 8, 1.0), key=jax.random.key(0), **kwargs)


def test_invalid_data() -> None:
    x = _simulate(8, 300, 8, 1.0)
    poly = x.copy()
    poly[0, 0] = 2.0
    with pytest.raises(SpecificationError, match="binary"):
        dimtest(poly, key=jax.random.key(0))
    with pytest.raises(SpecificationError, match="at least 4"):
        dimtest(x[:, :3], key=jax.random.key(0))


def test_small_simulation_study_type_i_error_and_power() -> None:
    """A small version of validation/simulation/dimtest_study.py (VALIDATION.md V30)."""
    null = [dimtest(_simulate(100 + r, 1000, 20, 1.0), key=jax.random.key(r)) for r in range(20)]
    t = np.array([res.statistic for res in null])
    assert np.mean([res.p_value < 0.05 for res in null]) <= 0.15
    assert -0.8 < t.mean() < 0.5
    alternative = [
        dimtest(_simulate(200 + r, 1000, 20, 0.7), key=jax.random.key(r)) for r in range(10)
    ]
    assert sum(res.p_value < 0.05 for res in alternative) >= 8
