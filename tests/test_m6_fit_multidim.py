"""Unit tests: M2 / M2* and S-X2 for confirmatory multidimensional models (DESIGN.md §16.19).

The product-grid statistics are checked against brute-force enumeration of every response
pattern; agreement with R mirt is in ``tests/test_mirt_m6_fit_multidim.py`` (V39).
"""

from __future__ import annotations

import dataclasses
import itertools
import warnings
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from flexpyirt import (
    BAEM,
    GRM,
    FitResult,
    Model,
    Normal,
    Quadrature,
    TwoPL,
    _core,
    _itemfit,
    _m2,
    fit,
)
from flexpyirt._compile import LATENT_ITEM
from flexpyirt.errors import FlexpyirtWarning, SpecificationError

ITEMS = [f"i{j}" for j in range(6)]
N_CATEGORIES = [2, 2, 3, 2, 2, 3]
FACTORS = {"F1": ITEMS[:4], "F2": ITEMS[3:]}  # i3 loads on both factors


@cache
def _data(n: int = 700, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    theta = rng.multivariate_normal(np.zeros(2), cov, size=n)
    load = np.array([[1.2, 0], [0.9, 0], [1.4, 0], [0.8, 0.7], [0, 1.1], [0, 1.3]])
    z = theta @ load.T + rng.logistic(size=(n, 6))
    y = (z > np.linspace(-0.5, 0.5, 6)).astype(float)
    for j in (2, 5):
        y[:, j] = np.digitize(z[:, j], [-0.9, 0.7])
    return pd.DataFrame(y, columns=ITEMS)


def _spec() -> Model:
    models = {name: (GRM(3) if k == 3 else TwoPL()) for name, k in zip(ITEMS, N_CATEGORIES)}
    return Model.from_items(models).with_factors(FACTORS)


@cache
def _fitted() -> FitResult:
    """A few EM cycles only: the checks below hold at any parameter value."""
    quad = Quadrature.equispaced(9, bounds=(-4.0, 4.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FlexpyirtWarning)
        return fit(_spec(), _data(), method=BAEM(tol=1e-2, max_iter=5), quadrature=quad, se=None)


def _grid(res: FitResult) -> tuple[_core.CompiledProblem, list[np.ndarray], np.ndarray]:
    """Compiled problem, per-item ``(Q, K_j)`` probabilities and ``(Q,)`` weights of the fit."""
    prob = res._compiled_for(np.zeros((1, len(ITEMS)), dtype=np.int64))
    weights = np.exp(np.asarray(_core._latent_log_weights(prob, jnp.asarray(res._eta))[0]))
    return prob, res._traces(prob), weights


def _pattern_probabilities(
    traces: list[np.ndarray], weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Every response pattern and its marginal probability on the grid."""
    pats = np.array(list(itertools.product(*[range(t.shape[1]) for t in traces])))
    cond = np.ones((pats.shape[0], weights.size))
    for j, tr in enumerate(traces):
        cond *= tr[:, pats[:, j]].T
    return pats, cond @ weights


def test_expected_tables_on_a_product_grid_match_enumeration() -> None:
    rng = np.random.default_rng(0)
    n_cat = [2, 3, 2, 4]
    nodes = np.array(list(itertools.product(np.linspace(-2, 2, 4), np.linspace(-2, 2, 3))))
    weights = rng.dirichlet(np.ones(nodes.shape[0]))  # not a product of marginal weights
    traces = [rng.dirichlet(np.ones(k), size=nodes.shape[0]) for k in n_cat]
    pats, p_pat = _pattern_probabilities(traces, weights)
    assert p_pat.sum() == pytest.approx(1.0)
    score = pats.sum(axis=1)
    expected = _itemfit.s_x2_expected_tables(traces, weights)
    total = sum(k - 1 for k in n_cat)
    for j, k_j in enumerate(n_cat):
        for s in range(1, total):
            at_s = score == s
            for k in range(k_j):
                brute = p_pat[at_s & (pats[:, j] == k)].sum() / p_pat[at_s].sum()
                got = expected[j][s - 1, k]
                if np.isnan(got):  # impossible cell
                    assert brute == 0.0
                else:
                    assert got == pytest.approx(brute, rel=1e-12, abs=1e-15)


def test_margins_and_xi_match_the_pattern_distribution() -> None:
    """pi_2 and Xi_2 on the product grid are the moments of the multinomial over patterns."""
    res = _fitted()
    prob, traces, weights = _grid(res)
    assert weights.size == 81 and prob.nodes.shape == (81, 2)
    pats, p_pat = _pattern_probabilities(traces, weights)
    ii, jj = _m2._pairs(len(ITEMS))
    stats = np.concatenate([pats, pats[:, ii] * pats[:, jj]], axis=1).astype(float)
    mean = p_pat @ stats
    cov = (stats * p_pat[:, None]).T @ stats - np.outer(mean, mean)

    eta = jnp.asarray(res._eta)
    margins = np.asarray(_m2.model_margins(prob, eta))
    np.testing.assert_allclose(margins, mean, rtol=1e-12)
    m1, m2 = (np.asarray(a) for a in _m2.conditional_moments(prob, eta))
    np.testing.assert_allclose(_m2._xi(m1, m2, weights, margins), cov, rtol=1e-10, atol=1e-13)


def test_delta_has_a_column_for_the_free_correlation() -> None:
    res = _fitted()
    prob, _, _ = _grid(res)
    eta = jnp.asarray(res._eta)
    jac = np.asarray(jax.jacfwd(lambda e: _m2.model_margins(prob, e))(eta))
    assert jac.shape == (6 + 15, res.n_params)
    table = res.parameters
    cor = table[(table["item"] == LATENT_ITEM) & table["free"]]
    assert list(cor["param"]) == ["cor_12"]
    col = int(cor["free_id"].iloc[0])
    step = 1e-6
    shift = np.zeros(res.n_params)
    shift[col] = step
    numeric = (
        np.asarray(_m2.model_margins(prob, eta + shift))
        - np.asarray(_m2.model_margins(prob, eta - shift))
    ) / (2 * step)
    np.testing.assert_allclose(jac[:, col], numeric, rtol=1e-6, atol=1e-9)
    # The correlation moves cross-products of items on different factors; the mean of an item
    # that loads on one factor does not depend on it (up to the truncation of the grid), while
    # the cross-loading item's does (the variance of its linear predictor changes).
    np.testing.assert_allclose(jac[[0, 1, 2, 4, 5], col], 0.0, atol=1e-5)
    assert abs(jac[3, col]) > 1e-4
    ii, jj = _m2._pairs(6)
    across = (ii < 3) & (jj > 3)
    assert np.all(np.abs(jac[6:, col][across]) > 1e-4)


def test_m2_and_item_fit_of_a_two_factor_model() -> None:
    res, df = _fitted(), _data()
    out = res.m2(df)
    row = out.iloc[0]
    # 7 slopes (i3 loads on both factors) + 4 + 2 * 2 intercepts + 1 correlation.
    assert res.n_params == 16
    assert int(row["df"]) == 6 + 15 - 16
    assert np.isfinite(row["m2"]) and 0.0 <= row["p"] <= 1.0
    assert np.isfinite(row["srmsr"])
    assert np.isnan(row["tli"]) and np.isnan(row["cfi"])  # polytomous items

    fit_table = res.item_fit(df)
    assert list(fit_table["item"]) == ITEMS
    assert (fit_table["df"] > 0).all()
    assert np.isfinite(fit_table["s_x2"]).all()

    # Recompute one item from the pieces: df is net of the item's and the latent free parameters.
    _, traces, weights = _grid(res)
    y = df.to_numpy().astype(int)
    observed = _itemfit.observed_tables(y, N_CATEGORIES)
    expected = _itemfit.s_x2_expected_tables(traces, weights)
    n_s = observed[0].sum(axis=1)
    for j, n_free_item in ((0, 2), (3, 3), (5, 3)):
        o, e = _itemfit.collapse_cells(observed[j], expected[j] * n_s[:, None], 1.0)
        stat, dof, _, _ = _itemfit.s_x2(o, e, n_free_item, 1, y.shape[0])
        assert fit_table["s_x2"].iloc[j] == stat
        assert fit_table["df"].iloc[j] == dof
        assert dof == int((~np.isnan(e)).sum() - e.shape[0] - n_free_item - 1)


def test_unsupported_multidimensional_structures_are_rejected() -> None:
    """The guards read the model only, so the fitted result is given each structure in turn."""
    res, df = _fitted(), _data()
    base = Model.uniform(TwoPL(), items=ITEMS)
    bifactor = base.with_bifactor(specific={"s1": ITEMS[:3], "s2": ITEMS[3:]})
    two_tier = base.with_two_tier(
        primary={"p1": ITEMS[:3], "p2": ITEMS[3:]}, specific={"s": ITEMS[1:5]}
    )
    multigroup = (
        _spec()
        .with_groups(ref=Normal.standard(), focal=Normal.standard())
        .equal_across_groups(items=ITEMS)
    )
    for spec in (bifactor, two_tier):
        reduced = dataclasses.replace(res, _model=spec)
        with pytest.raises(SpecificationError, match="bifactor or two-tier"):
            reduced.m2(df)
        with pytest.raises(SpecificationError, match="bifactor, two-tier or multiple-group"):
            reduced.item_fit(df)
    grouped = dataclasses.replace(res, _model=multigroup)
    with pytest.raises(SpecificationError, match="single-group"):
        grouped.m2(df)
    with pytest.raises(SpecificationError, match="multiple-group multidimensional"):
        grouped.item_fit(df, group=np.tile(["ref", "focal"], len(df) // 2))
