"""Unit tests for local dependence in bifactor and two-tier models (DESIGN.md §17.3).

The expected pair tables on the reduced grids are checked against brute-force sums over the full
``(P + S)``-dimensional product grid, and Q3 against its definition; the agreement with mirt's
``residuals()`` on ``bfactor()`` fits is in ``tests/test_mirt_m7_local_dependence_bifactor.py``
(VALIDATION.md V38).
"""

from __future__ import annotations

import itertools
import warnings
from functools import cache

import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal

from flexpyirt import BAEM, GRM, FitResult, Model, Quadrature, TwoPL, fit
from flexpyirt._local_dependence import ld_pairs, q3_matrix, reduced_pair_probabilities
from flexpyirt.errors import SpecificationError

ITEMS = [f"i{k}" for k in range(10)]
N_POINTS = 7
BOUNDS = (-4.0, 4.0)
# i4 cross-loads on both primary factors; i3, i4, i8 and i9 have no specific factor.
PRIMARY = {"p1": ITEMS[:5], "p2": ITEMS[4:]}
SPECIFIC = {"s1": ITEMS[:3], "s2": ITEMS[5:8]}
BLOCK = np.array([0, 0, 0, 2, 2, 1, 1, 1, 2, 2])
# Bifactor: i9 loads on the general factor only.
BIFACTOR = {"s1": ITEMS[:3], "s2": ITEMS[3:6], "s3": ITEMS[6:9]}


def _sim(n: int = 500, seed: int = 5) -> pd.DataFrame:
    """2PL items and a 3-category item (i6); 5% missing at random."""
    rng = np.random.default_rng(seed)
    cov = np.eye(4)
    cov[0, 1] = cov[1, 0] = 0.4
    theta = rng.multivariate_normal(np.zeros(4), cov, size=n)
    a = np.zeros((10, 4))
    a[:5, 0] = rng.uniform(1.0, 1.7, 5)
    a[4:, 1] = rng.uniform(1.0, 1.7, 6)
    a[:3, 2] = rng.uniform(0.6, 1.1, 3)
    a[5:8, 3] = rng.uniform(0.6, 1.1, 3)
    z = theta @ a.T + rng.normal(0.0, 0.6, 10)
    y = (rng.random((n, 10)) < 1.0 / (1.0 + np.exp(-z))).astype(float)
    y[:, 6] = np.digitize(z[:, 6] + rng.logistic(size=n), [-0.8, 0.8])
    y[rng.random(y.shape) < 0.05] = np.nan
    return pd.DataFrame(y, columns=ITEMS)


def _base() -> Model:
    kernels = {name: (GRM(3) if name == "i6" else TwoPL()) for name in ITEMS}
    return Model.from_items(kernels)


@cache
def _fitted(kind: str) -> tuple[FitResult, pd.DataFrame]:
    """A few EM iterations: the statistics are defined at any parameter values."""
    df = _sim()
    if kind == "two_tier":
        spec = _base().with_two_tier(primary=PRIMARY, specific=SPECIFIC)
    else:
        spec = _base().with_bifactor(specific=BIFACTOR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fit(
            spec,
            df,
            quadrature=Quadrature.equispaced(N_POINTS, bounds=BOUNDS),
            se=None,
            method=BAEM(max_iter=5),
        )
    return res, df


def _brute_force(res: FitResult, df: pd.DataFrame) -> dict[str, np.ndarray]:
    """LD statistics from the full product grid with multivariate-normal weights."""
    n_factors = 4
    nodes_1d = np.linspace(*BOUNDS, N_POINTS)
    full = np.array(list(itertools.product(nodes_1d, repeat=n_factors)))
    cov = np.eye(n_factors)
    latent = res.latent.iloc[0]
    if "cor_12" in latent:
        cov[0, 1] = cov[1, 0] = float(latent["cor_12"])
    weights = multivariate_normal(np.zeros(n_factors), cov).pdf(full)
    weights /= weights.sum()
    y = np.where(df.isna(), -1, df.fillna(0)).astype(np.int64)
    prob = res._compiled_for(np.zeros((1, y.shape[1])))
    return ld_pairs(y, res._traces(prob, full), weights)


@pytest.mark.parametrize("kind", ["bifactor", "two_tier"])
def test_expected_tables_equal_the_full_product_grid(kind: str) -> None:
    res, df = _fitted(kind)
    ld = res.local_dependence(df).pairs
    brute = _brute_force(res, df)
    assert len(ld) == 45
    np.testing.assert_allclose(ld["x2"], brute["x2"], rtol=1e-10)
    np.testing.assert_allclose(ld["g2"], brute["g2"], rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(ld["cramer_v"], brute["cramer_v"], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(ld["n"], brute["n"])
    np.testing.assert_array_equal(ld["df"], brute["df"])


def test_q3_is_the_correlation_of_residuals_at_the_vector_eap() -> None:
    res, df = _fitted("two_tier")
    ld = res.local_dependence(df)
    theta = res.score(df).filter(regex="^theta").to_numpy()
    assert theta.shape == (len(df), 4)
    y = df.to_numpy()
    prob = res._compiled_for(np.zeros((1, y.shape[1])))
    expected = np.column_stack(
        [trace @ np.arange(trace.shape[1]) for trace in res._traces(prob, theta)]
    )
    q3 = q3_matrix(y - expected)
    np.testing.assert_allclose(ld.q3.to_numpy(), q3, rtol=0, atol=1e-10)
    upper = np.triu_indices(10, k=1)
    assert ld.q3_mean == pytest.approx(float(q3[upper].mean()))
    np.testing.assert_allclose(ld.pairs["q3_star"], q3[upper] - ld.q3_mean)


def test_reduced_pair_probabilities_cases() -> None:
    """Same specific factor, different specific factors and no specific factor."""
    rng = np.random.default_rng(0)
    n_primary, n_points = 5, 4
    primary = rng.dirichlet(np.ones(n_primary))
    blocks = np.vstack([rng.dirichlet(np.ones(n_points), size=2), np.full(n_points, 0.25)])
    item_block = np.array([0, 0, 1, 2, 2])
    traces = []
    for block in item_block:
        p = rng.uniform(0.1, 0.9, (n_primary, n_points))
        if block == 2:  # no specific factor: constant over the specific node
            p = np.repeat(p[:, :1], n_points, axis=1)
        traces.append(np.stack([1 - p, p], axis=2))
    table = reduced_pair_probabilities(traces, item_block, primary, blocks)
    for i, j in itertools.combinations(range(5), 2):
        want = np.zeros((2, 2))
        # Product grid over the primary node and both specific nodes.
        for g, t0, t1 in itertools.product(range(n_primary), range(n_points), range(n_points)):
            node = {0: t0, 1: t1, 2: 0}
            w = primary[g] * blocks[0, t0] * blocks[1, t1]
            want += w * np.outer(
                traces[i][g, node[int(item_block[i])]], traces[j][g, node[int(item_block[j])]]
            )
        got = table(i, j)
        np.testing.assert_allclose(got, want, rtol=1e-12)
        assert got.sum() == pytest.approx(1.0)


def test_an_unmodelled_testlet_is_flagged() -> None:
    """Dependence beyond the specific factors: a testlet left out of ``specific``."""
    rng = np.random.default_rng(11)
    n = 1500
    names = [f"q{k}" for k in range(12)]
    theta = rng.normal(size=(n, 4))
    z = 1.2 * theta[:, [0]] + rng.normal(0.0, 0.5, 12)
    z[:, :9] += 1.3 * theta[:, np.repeat([1, 2, 3], 3)]
    df = pd.DataFrame((rng.random((n, 12)) < 1 / (1 + np.exp(-z))).astype(float), columns=names)
    # The third testlet (q6-q8) is not declared; q9-q11 load on the general factor only.
    spec = Model.uniform(TwoPL(), items=names).with_bifactor(
        specific={"s1": names[:3], "s2": names[3:6]}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fit(spec, df, quadrature=Quadrature.equispaced(15, bounds=(-6, 6)), se=None)
    pairs = res.local_dependence(df).pairs
    inside = pairs["item_i"].isin(names[6:9]) & pairs["item_j"].isin(names[6:9])
    assert inside.sum() == 3
    assert (pairs.loc[inside, "cramer_v"] > 0).all()
    assert (pairs.loc[inside, "p_x2"] < 1e-2).all()
    assert pairs.loc[inside, "x2"].min() > pairs.loc[~inside, "x2"].max()


@pytest.mark.parametrize("kind", ["bifactor", "two_tier"])
def test_jsi_is_not_defined_for_reduced_models(kind: str) -> None:
    res, df = _fitted(kind)
    with pytest.raises(SpecificationError, match="unidimensional"):
        res.jsi(df)
