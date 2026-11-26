"""Unit tests for the weighted and cluster-robust sandwich (DESIGN.md §17.6).

The comparison with R (mirt and an independent numDeriv computation) is in
``tests/test_mirt_v57_weighted_cluster_se.py`` (VALIDATION.md V57).
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from flexpyirt import (
    BAEM,
    FitResult,
    Model,
    Normal,
    NormalPrior,
    Quadrature,
    TwoPL,
    _core,
    _robust_se,
    fit,
)
from flexpyirt._compile import compile_problem, meat_patterns, prepare
from flexpyirt.errors import FlexpyirtWarning, SpecificationError

TIGHT = BAEM(tol=1e-10, max_iter=20000)


def _simulate(seed: int, n_clusters: int, per_cluster: int, n_items: int) -> pd.DataFrame:
    """2PL responses with a cluster effect in the trait (schools)."""
    rng = np.random.default_rng(seed)
    cluster = np.repeat(np.arange(n_clusters), per_cluster)
    theta = rng.normal(0.0, 0.6, n_clusters)[cluster] + rng.normal(0.0, 0.8, cluster.size)
    slope = rng.uniform(0.8, 2.0, n_items)
    intercept = rng.normal(0.0, 0.8, n_items)
    prob = 1.0 / (1.0 + np.exp(-(theta[:, None] * slope + intercept)))
    data = (rng.random((cluster.size, n_items)) < prob).astype(float)
    data[rng.random(data.shape) < 0.04] = np.nan
    return pd.DataFrame(data, columns=[f"q{j + 1:02d}" for j in range(n_items)])


@cache
def _school_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    data = _simulate(1, 25, 16, 6)
    rng = np.random.default_rng(2)
    school = np.repeat([f"s{c:02d}" for c in range(25)], 16)
    weights = rng.uniform(0.3, 3.0, len(data)) * np.repeat(rng.uniform(0.5, 2.0, 25), 16)
    return data, school, weights


def _spec(data: pd.DataFrame) -> Model:
    return Model.uniform(TwoPL(), items=list(data.columns))


@cache
def _fits(kind: str) -> FitResult:
    data, school, weights = _school_data()
    spec = _spec(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FlexpyirtWarning)
        if kind == "oakes":
            return fit(spec, data, weights=weights, method=TIGHT)
        if kind == "weighted":
            return fit(spec, data, weights=weights, method=TIGHT, se="sandwich")
        if kind == "cluster":
            return fit(spec, data, weights=weights, cluster=school, method=TIGHT, se="sandwich")
        if kind == "singleton":
            singletons = np.arange(len(data))
            return fit(spec, data, weights=weights, cluster=singletons, method=TIGHT, se="sandwich")
        if kind == "unweighted_cluster":
            return fit(spec, data, cluster=school, method=TIGHT, se="sandwich")
    raise AssertionError(kind)


# ---- kernel ------------------------------------------------------------------------------------


def _row_scores(
    spec: Model,
    data: pd.DataFrame,
    group: np.ndarray | None,
    eta: jax.Array,
    quadrature: Quadrature,
) -> np.ndarray:
    """``(N, n_free)`` person scores by a dense forward-mode Jacobian of one row per pattern."""
    prep = prepare(spec, data, group=group, item_names=None, table=None)
    comp = compile_problem(spec, prep.y, prep.group, prep.table, quadrature)
    rows = dataclasses.replace(
        comp.problem,
        y=jnp.asarray(prep.y, dtype=jnp.int32),
        group=jnp.asarray(prep.group, dtype=jnp.int32),
        counts=jnp.ones(prep.y.shape[0]),
        blocks=None,
    )

    def logliks(e: jax.Array) -> jax.Array:
        return jax.scipy.special.logsumexp(_core._log_joint(rows, e), axis=1)

    return np.asarray(jax.jacfwd(logliks)(eta))


@pytest.mark.slow
@pytest.mark.parametrize("budget", [None, 5_000])
def test_meat_kernels_match_dense_person_scores(
    budget: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Squared-weight and cluster cross-products equal sums over dense per-person scores.

    Two groups with clusters that span them; a small chunk budget forces several chunks.
    """
    data = _simulate(3, 12, 20, 5)
    group = np.where(np.arange(len(data)) % 3 == 0, "focal", "ref")
    spec = (
        _spec(data)
        .with_groups(ref=Normal.standard(), focal=Normal.free())
        .equal_across_groups(items=["q01", "q02"])
    )
    rng = np.random.default_rng(4)
    weights = rng.uniform(0.2, 2.5, len(data))
    cluster = np.repeat(np.arange(12), 20)
    prep = prepare(
        spec, data, group=group, item_names=None, table=None, weights=weights, cluster=cluster
    )
    comp = compile_problem(
        spec, prep.y, prep.group, prep.table, Quadrature.equispaced(21), weights=prep.weights
    )
    eta = jnp.asarray(comp.eta0) + 0.05
    scores = _row_scores(spec, data, group, eta, Quadrature.equispaced(21))
    if budget is not None:
        monkeypatch.setattr(_robust_se, "_CHUNK_ELEMENTS", budget)

    def problem_of(rows: object) -> _core.CompiledProblem:
        return dataclasses.replace(
            comp.problem,
            y=jnp.asarray(rows.y, dtype=jnp.int32),  # type: ignore[attr-defined]
            group=jnp.asarray(rows.group, dtype=jnp.int32),  # type: ignore[attr-defined]
            counts=jnp.asarray(rows.counts),  # type: ignore[attr-defined]
            blocks=None,
        )

    squared = meat_patterns(prep.y, prep.group, prep.weights)
    if budget is not None:
        assert _robust_se._chunk_size(problem_of(squared)) < squared.y.shape[0]
    got = np.asarray(
        jax.jit(lambda p, e: _robust_se.score_cross_product(p, e))(problem_of(squared), eta)
    )
    expected = (weights[:, None] ** 2 * scores).T @ scores
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-10 * np.abs(expected).max())

    clustered = meat_patterns(prep.y, prep.group, prep.weights, prep.cluster, prep.n_clusters)
    assert clustered.n_clusters == 12 and clustered.cluster is not None
    got_cl = np.asarray(
        jax.jit(_robust_se.score_cluster_cross_product, static_argnames="n_clusters")(
            problem_of(clustered), eta, jnp.asarray(clustered.cluster), n_clusters=12
        )
    )
    totals = np.zeros((12, eta.shape[0]))
    np.add.at(totals, cluster, weights[:, None] * scores)
    expected_cl = totals.T @ totals
    np.testing.assert_allclose(got_cl, expected_cl, rtol=0, atol=1e-10 * np.abs(expected_cl).max())


def test_meat_patterns_compress_by_cluster_and_square_weights() -> None:
    y = np.array([[0, 1], [0, 1], [1, 1], [0, 1]], dtype=np.int32)
    group = np.zeros(4, dtype=np.int32)
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    plain = meat_patterns(y, group, weights)
    assert plain.cluster is None
    assert sorted(plain.counts[plain.counts > 0].tolist()) == [9.0, 21.0]
    clustered = meat_patterns(y, group, weights, np.array([0, 1, 1, 0], dtype=np.int32), 2)
    assert clustered.cluster is not None
    real = clustered.counts > 0
    got = sorted(
        zip(clustered.cluster[real].tolist(), clustered.counts[real].tolist(), strict=True)
    )
    assert got == [(0, 5.0), (1, 2.0), (1, 3.0)]


# ---- fit() -------------------------------------------------------------------------------------


@pytest.mark.slow
def test_singleton_clusters_are_the_weighted_sandwich_times_n_over_n_minus_1() -> None:
    weighted, singleton = _fits("weighted"), _fits("singleton")
    assert weighted.vcov is not None and singleton.vcov is not None
    n = weighted.n_obs
    np.testing.assert_array_equal(singleton.parameters["estimate"], weighted.parameters["estimate"])
    np.testing.assert_allclose(singleton.vcov, n / (n - 1) * weighted.vcov, rtol=1e-10)
    assert singleton._n_clusters == n and weighted._n_clusters is None


def test_cluster_sandwich_is_built_from_the_oakes_bread() -> None:
    """``V = V_oakes B_cl V_oakes``; clustering changes ``B`` only, not the estimates or ``H``."""
    oakes, weighted, cluster = _fits("oakes"), _fits("weighted"), _fits("cluster")
    assert oakes.vcov is not None and weighted.vcov is not None and cluster.vcov is not None
    np.testing.assert_array_equal(cluster.parameters["estimate"], oakes.parameters["estimate"])
    meat = np.linalg.inv(oakes.vcov) @ cluster.vcov @ np.linalg.inv(oakes.vcov)
    np.testing.assert_allclose(meat, meat.T, atol=1e-8 * np.abs(meat).max())
    assert np.all(np.linalg.eigvalsh(meat) > 0)
    # The trait has a school effect: clustering inflates the standard errors.
    assert np.mean(np.diag(cluster.vcov) / np.diag(weighted.vcov)) > 1.1
    table = cluster.parameters
    free = table["free"].to_numpy()
    np.testing.assert_allclose(
        table.loc[free, "se"], np.sqrt(np.diag(cluster.vcov))[table.loc[free, "free_id"]]
    )


def test_unit_weights_give_the_unweighted_sandwich() -> None:
    data, school, _ = _school_data()
    spec = _spec(data)
    plain = fit(spec, data, method=TIGHT, se="sandwich")
    unit = fit(spec, data, method=TIGHT, se="sandwich", weights=np.ones(len(data)))
    assert plain.vcov is not None and unit.vcov is not None
    np.testing.assert_allclose(unit.vcov, plain.vcov, rtol=1e-10)
    unweighted_cluster = _fits("unweighted_cluster")
    unit_cluster = fit(
        spec, data, method=TIGHT, se="sandwich", weights=np.ones(len(data)), cluster=school
    )
    assert unweighted_cluster.vcov is not None and unit_cluster.vcov is not None
    np.testing.assert_allclose(unit_cluster.vcov, unweighted_cluster.vcov, rtol=1e-10)


def test_weighted_sandwich_does_not_depend_on_the_scale_of_the_weights() -> None:
    data, school, weights = _school_data()
    spec = _spec(data)
    scaled = fit(spec, data, weights=7.5 * weights, method=TIGHT, se="sandwich")
    scaled_cluster = fit(
        spec, data, weights=7.5 * weights, cluster=school, method=TIGHT, se="sandwich"
    )
    for got, ref in ((scaled, _fits("weighted")), (scaled_cluster, _fits("cluster"))):
        assert got.vcov is not None and ref.vcov is not None
        np.testing.assert_allclose(got.vcov, ref.vcov, rtol=1e-6)


def test_zero_weight_rows_are_dropped_with_their_clusters() -> None:
    """A cluster whose rows all have weight 0 does not count in ``G``."""
    data, school, weights = _school_data()
    spec = _spec(data)
    extra = pd.concat([data, data.iloc[:10]], ignore_index=True)
    extra_school = np.concatenate([school, np.repeat("empty", 10)])
    extra_weights = np.concatenate([weights, np.zeros(10)])
    res = fit(spec, extra, weights=extra_weights, cluster=extra_school, method=TIGHT, se="sandwich")
    ref = _fits("cluster")
    assert res._n_clusters == ref._n_clusters == 25
    assert res.vcov is not None and ref.vcov is not None
    np.testing.assert_allclose(res.vcov, ref.vcov, rtol=1e-10)


def test_cluster_labels_can_be_any_hashable_and_relabelling_changes_nothing() -> None:
    data, school, weights = _school_data()
    spec = _spec(data)
    codes = pd.Series(pd.factorize(school)[0] * 7 + 3)
    res = fit(spec, data, weights=weights, cluster=codes, method=TIGHT, se="sandwich")
    ref = _fits("cluster")
    assert res.vcov is not None and ref.vcov is not None
    np.testing.assert_allclose(res.vcov, ref.vcov, rtol=1e-12)


@pytest.mark.slow
def test_map_adds_the_prior_curvature_to_the_cluster_meat() -> None:
    data, school, weights = _school_data()
    spec = _spec(data).prior(param="a", dist=NormalPrior(1.0, 0.5))
    with pytest.warns(FlexpyirtWarning):
        oakes = fit(spec, data, weights=weights, method=TIGHT)
    cluster = fit(spec, data, weights=weights, cluster=school, method=TIGHT, se="sandwich")
    assert oakes.vcov is not None and cluster.vcov is not None
    prep = prepare(
        spec, data, group=None, item_names=None, table=None, weights=weights, cluster=school
    )
    comp = compile_problem(
        spec, prep.y, prep.group, prep.table, Quadrature.default(1), weights=prep.weights
    )
    eta = jnp.asarray(cluster._eta)
    rows = meat_patterns(prep.y, prep.group, prep.weights, prep.cluster, prep.n_clusters)
    problem = dataclasses.replace(
        comp.problem,
        y=jnp.asarray(rows.y, dtype=jnp.int32),
        group=jnp.asarray(rows.group, dtype=jnp.int32),
        counts=jnp.asarray(rows.counts),
        blocks=None,
    )
    assert rows.cluster is not None
    scores = np.asarray(
        _robust_se.score_cluster_cross_product(problem, eta, jnp.asarray(rows.cluster), 25)
    )
    curvature = -np.asarray(jax.hessian(lambda e: _core.log_prior(comp.problem, e))(eta))
    assert np.abs(curvature).max() > 0
    expected = oakes.vcov @ (25 / 24 * scores + curvature) @ oakes.vcov
    np.testing.assert_allclose(cluster.vcov, expected, rtol=1e-8, atol=1e-14)


# ---- options, warnings, errors -----------------------------------------------------------------


@pytest.mark.slow
def test_sandwich_with_weights_does_not_warn_but_the_others_do() -> None:
    data, school, weights = _school_data()
    spec = _spec(data)
    method = BAEM(tol=1e-4)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FlexpyirtWarning)
        fit(spec, data, weights=weights, method=method, se="sandwich")
        fit(spec, data, weights=weights, cluster=school, method=method, se="sandwich")
    for se in ("oakes", "hessian", "crossprod"):
        with pytest.warns(FlexpyirtWarning, match="use se='sandwich'"):
            fit(spec, data, weights=weights, method=method, se=se)  # type: ignore[arg-type]


@pytest.mark.slow
def test_crossprod_with_weights_is_minus_the_weighted_cross_product() -> None:
    """``se="crossprod"`` inverts ``sum_p c_p s_p s_p^T`` with ``c_p`` the weight sums (mirt)."""
    data, _, weights = _school_data()
    spec = _spec(data)
    with pytest.warns(FlexpyirtWarning):
        res = fit(spec, data, weights=weights, method=TIGHT, se="crossprod")
    scores = _row_scores(spec, data, None, jnp.asarray(res._eta), Quadrature.default(1))
    assert res.vcov is not None
    np.testing.assert_allclose(
        np.linalg.inv(res.vcov), (weights[:, None] * scores).T @ scores, rtol=1e-8
    )


@pytest.mark.parametrize("se", ["oakes", "hessian", "crossprod", None])
def test_cluster_needs_the_sandwich(se: str | None) -> None:
    data, school, _ = _school_data()
    with pytest.raises(SpecificationError, match="cluster= is only used by se='sandwich'"):
        fit(_spec(data), data, cluster=school, se=se)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        (lambda n: np.zeros(n), "at least two clusters"),
        (lambda n: np.arange(n - 1), "one label per row"),
        (lambda n: np.where(np.arange(n) == 3, np.nan, np.arange(n) % 4), "missing labels"),
        (lambda n: [None, *range(n - 1)], "missing labels"),
        (lambda n: "school", "not a column name"),
    ],
)
def test_invalid_clusters_raise(labels: object, message: str) -> None:
    data, _, _ = _school_data()
    cluster = labels(len(data))  # type: ignore[operator]
    with pytest.raises(SpecificationError, match=message):
        fit(_spec(data), data, cluster=cluster, se="sandwich")


def test_summary_and_json_keep_the_number_of_clusters() -> None:
    res = _fits("cluster")
    assert "sandwich (cluster-robust, 25 clusters)" in res.summary()
    assert "cluster-robust, 25 clusters" in res._repr_html_()
    payload = json.loads(res.to_json())
    assert payload["schema_version"] == 9 and payload["n_clusters"] == 25
    back = FitResult.from_json(json.dumps(payload))
    assert back._n_clusters == 25 and back.se_method == "sandwich"
    assert back.vcov is not None and res.vcov is not None
    np.testing.assert_array_equal(back.vcov, res.vcov)
    assert "n_clusters" not in json.loads(_fits("weighted").to_json())
    old = dict(payload, schema_version=8)
    with pytest.raises(SpecificationError, match="Unexpected keys"):
        FitResult.from_json(json.dumps(old))
    wrong = dict(payload, se_method="oakes")
    with pytest.raises(SpecificationError, match="n_clusters is only valid"):
        FitResult.from_json(json.dumps(wrong))
    for bad in (1, 2.5, "3"):
        with pytest.raises(SpecificationError):
            FitResult.from_json(json.dumps(dict(payload, n_clusters=bad)))
