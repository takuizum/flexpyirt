"""Unit tests for sampling weights, ``fit(..., weights=...)`` (DESIGN.md §17.4)."""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from typing import Any

import jax
import numpy as np
import pandas as pd
import pytest

from flexpyirt import (
    BAEM,
    GRM,
    EmpiricalHistogram,
    FitResult,
    Model,
    Normal,
    NormalPrior,
    Quadrature,
    TwoPL,
    fit,
    model_parallel_analysis,
)
from flexpyirt._compile import compress_patterns, prepare
from flexpyirt._json import SCHEMA_VERSION
from flexpyirt.errors import FlexpyirtWarning, SpecificationError

TIGHT = BAEM(tol=1e-10, max_iter=20000)


def _sim(n: int = 400, j: int = 5, seed: int = 11, n_categories: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=(n, 1))
    a = rng.uniform(0.8, 1.6, size=j)
    if n_categories == 2:
        d = rng.normal(0.0, 0.8, size=j)
        y = (rng.random((n, j)) < 1 / (1 + np.exp(-(theta * a + d)))).astype(float)
    else:
        cuts = np.sort(rng.normal(0.0, 1.0, size=(j, n_categories - 1)), axis=1)[:, ::-1]
        z = theta[:, :, None] * a[None, :, None] + cuts[None]
        y = (rng.random((n, j))[:, :, None] < 1 / (1 + np.exp(-z))).sum(axis=2).astype(float)
    return pd.DataFrame(y, columns=[f"i{k}" for k in range(j)])


def _fit(spec: Model, data: pd.DataFrame, **kwargs: Any) -> FitResult:
    """``fit`` with the weighted-SE warning silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FlexpyirtWarning)
        return fit(spec, data, **{"method": TIGHT, **kwargs})


def _free(res: FitResult, col: str) -> np.ndarray:
    return np.asarray(res.parameters.loc[res.parameters["free"], col], dtype=float)


def _assert_same_fit(res: FitResult, other: FitResult, *, tol: float = 1e-6) -> None:
    np.testing.assert_allclose(res.loglik, other.loglik, rtol=1e-10)
    np.testing.assert_allclose(_free(res, "estimate"), _free(other, "estimate"), atol=tol)
    np.testing.assert_allclose(_free(res, "se"), _free(other, "se"), rtol=1e-5)
    np.testing.assert_allclose(res.aic, other.aic, rtol=1e-10)
    np.testing.assert_allclose(res.bic, other.bic, rtol=1e-10)


# ---- validation ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (np.ones(7), "one weight per row"),
        (np.ones((8, 1)), "one weight per row"),
        ([1.0, 2.0, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0], "finite"),
        ([1.0, 2.0, np.inf, 1.0, 1.0, 1.0, 1.0, 1.0], "finite"),
        ([1.0, -0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "non-negative"),
        (np.zeros(8), "positive total"),
        (["a"] * 8, "numeric"),
    ],
)
def test_invalid_weights_raise(bad: Any, message: str) -> None:
    data = _sim(8, 3)
    spec = Model.uniform(TwoPL(), items=list(data.columns))
    with pytest.raises(SpecificationError, match=message):
        fit(spec, data, weights=bad)


def test_group_with_only_zero_weights_raises() -> None:
    data = _sim(40, 3)
    group = np.repeat(["a", "b"], 20)
    spec = (
        Model.uniform(TwoPL(), items=list(data.columns))
        .with_groups(a=Normal.standard(), b=Normal.free())
        .equal_across_groups(items=list(data.columns))
    )
    weights = np.where(group == "a", 1.0, 0.0)
    with pytest.raises(SpecificationError, match=r"weights are all zero.*'b'"):
        fit(spec, data, group=group, weights=weights)


def test_category_seen_only_in_zero_weight_rows_is_not_estimable() -> None:
    data = _sim(60, 3, n_categories=3)
    top = data["i0"] == 2
    assert top.any()
    spec = Model.uniform(GRM(n_categories=3), items=list(data.columns))
    with pytest.raises(SpecificationError, match="never observed"):
        fit(spec, data, weights=np.where(top, 0.0, 1.0))


# ---- compile layer ---------------------------------------------------------------------------


def test_compress_patterns_sums_weights() -> None:
    y = np.array([[0, 1], [1, 1], [0, 1], [1, 1], [0, 1]], dtype=np.int32)
    group = np.array([0, 0, 0, 1, 1], dtype=np.int32)
    weights = np.array([0.5, 2.0, 1.25, 3.0, 0.75])
    patterns, pat_groups, counts, inverse = compress_patterns(y, group, weights)
    assert patterns.shape[0] == 4
    for p in range(patterns.shape[0]):
        np.testing.assert_allclose(counts[p], weights[inverse == p].sum())
    np.testing.assert_array_equal(patterns[inverse], y)
    np.testing.assert_array_equal(pat_groups[inverse], group)
    _, _, unweighted, _ = compress_patterns(y, group)
    np.testing.assert_array_equal(unweighted, np.bincount(inverse).astype(float))


def test_prepare_drops_zero_weight_rows() -> None:
    data = _sim(30, 3)
    spec = Model.uniform(TwoPL(), items=list(data.columns))
    weights = np.linspace(0.5, 2.0, 30)
    weights[[3, 17]] = 0.0
    prep = prepare(spec, data, group=None, item_names=None, table=None, weights=weights)
    keep = weights > 0
    assert prep.y.shape[0] == 28
    assert prep.weights is not None
    np.testing.assert_array_equal(prep.weights, weights[keep])
    np.testing.assert_array_equal(prep.y, data.to_numpy(dtype=np.int32)[keep])
    unweighted = prepare(spec, data, group=None, item_names=None, table=None)
    assert unweighted.weights is None


# ---- equivalences ----------------------------------------------------------------------------


def test_integer_weights_equal_row_replication_with_missing() -> None:
    data = _sim(200, 4)
    rng = np.random.default_rng(5)
    data = data.mask(rng.random(data.shape) < 0.1)
    weights = rng.integers(0, 4, size=len(data))
    spec = Model.uniform(TwoPL(), items=list(data.columns))
    weighted = _fit(spec, data, weights=weights)
    replicated = fit(
        spec, data.loc[data.index.repeat(weights)].reset_index(drop=True), method=TIGHT
    )
    _assert_same_fit(weighted, replicated)
    assert weighted.n_obs == int((weights > 0).sum())
    assert replicated.n_obs == int(weights.sum())
    assert weighted._sum_weights == float(weights.sum())
    assert replicated._sum_weights is None


def test_unit_weights_and_zero_weight_rows() -> None:
    data = _sim(300, 5)
    spec = Model.uniform(TwoPL(), items=list(data.columns))
    plain = fit(spec, data, method=TIGHT)
    unit = _fit(spec, data, weights=np.ones(len(data)))
    np.testing.assert_array_equal(_free(unit, "estimate"), _free(plain, "estimate"))
    np.testing.assert_array_equal(_free(unit, "se"), _free(plain, "se"))
    assert unit.loglik == plain.loglik
    assert unit.bic == plain.bic

    # Rows of weight 0 are the same as absent rows, exactly.
    weights = np.random.default_rng(2).uniform(0.2, 3.0, size=len(data))
    weights[:40] = 0.0
    with_zeros = _fit(spec, data, weights=weights)
    dropped = _fit(spec, data.iloc[40:], weights=weights[40:])
    np.testing.assert_array_equal(_free(with_zeros, "estimate"), _free(dropped, "estimate"))
    assert with_zeros.loglik == dropped.loglik
    assert with_zeros.n_obs == len(data) - 40


def test_rescaling_weights_scales_loglik_and_information() -> None:
    data = _sim(300, 5)
    spec = Model.uniform(TwoPL(), items=list(data.columns))
    weights = np.random.default_rng(3).lognormal(0.0, 0.5, size=len(data))
    base = _fit(spec, data, weights=pd.Series(weights))
    scaled = _fit(spec, data, weights=(4.0 * weights).tolist())
    np.testing.assert_allclose(_free(scaled, "estimate"), _free(base, "estimate"), atol=1e-7)
    np.testing.assert_allclose(scaled.loglik, 4.0 * base.loglik, rtol=1e-10)
    np.testing.assert_allclose(_free(scaled, "se"), _free(base, "se") / 2.0, rtol=1e-6)
    np.testing.assert_allclose(
        scaled.bic, -2.0 * scaled.loglik + scaled.n_params * np.log(4.0 * weights.sum())
    )
    # The weighted estimate is not the unweighted one.
    plain = fit(spec, data, method=TIGHT, se=None)
    assert np.max(np.abs(_free(base, "estimate") - _free(plain, "estimate"))) > 1e-3


@pytest.mark.slow
@pytest.mark.parametrize("squarem", [False, True])
def test_integer_weights_two_group_grm(squarem: bool) -> None:
    data = _sim(600, 5, n_categories=3)
    group = np.repeat(["ref", "focal"], 300)
    weights = np.random.default_rng(8).integers(1, 4, size=len(data))
    spec = (
        Model.uniform(GRM(n_categories=3), items=list(data.columns))
        .with_groups(ref=Normal.standard(), focal=Normal.free())
        .equal_across_groups(items=list(data.columns)[:3])
    )
    method = BAEM(tol=1e-10, max_iter=20000, squarem=squarem)
    weighted = _fit(spec, data, group=group, weights=weights, method=method, se="hessian")
    idx = data.index.repeat(weights)
    replicated = fit(
        spec,
        data.loc[idx].reset_index(drop=True),
        group=group[idx],
        method=method,
        se="hessian",
    )
    _assert_same_fit(weighted, replicated)


@pytest.mark.slow
def test_integer_weights_empirical_histogram_fipc_and_map() -> None:
    data = _sim(500, 6)
    items = list(data.columns)
    weights = np.random.default_rng(9).integers(1, 4, size=len(data))
    repeated = data.loc[data.index.repeat(weights)].reset_index(drop=True)
    quad = Quadrature.equispaced(41, bounds=(-5, 5))

    # FIPC: three items fixed, the latent distribution an empirical histogram.
    eh = Model.uniform(TwoPL(), items=items).with_latent(EmpiricalHistogram())
    for item in items[:3]:
        eh = eh.fix(item=item, a=1.1, d=0.1)
    method = BAEM(tol=1e-9, max_iter=20000, squarem=True)
    weighted = _fit(eh, data, weights=weights, quadrature=quad, method=method, se=None)
    replicated = fit(eh, repeated, quadrature=quad, method=method, se=None)
    np.testing.assert_allclose(weighted.loglik, replicated.loglik, rtol=1e-9)
    np.testing.assert_allclose(
        _free(weighted, "estimate"), _free(replicated, "estimate"), atol=1e-5
    )
    w_eh, r_eh = weighted.latent_weights, replicated.latent_weights
    assert w_eh is not None and r_eh is not None
    np.testing.assert_allclose(
        w_eh["weight"].to_numpy(dtype=float), r_eh["weight"].to_numpy(dtype=float), atol=1e-5
    )

    # MAP: the prior is not weighted, so integer weights still equal replication.
    spec = Model.uniform(TwoPL(), items=items).prior(param="a", dist=NormalPrior(1.0, 0.5))
    weighted_map = _fit(spec, data, weights=weights)
    replicated_map = fit(spec, repeated, method=TIGHT)
    np.testing.assert_allclose(weighted_map.log_posterior, replicated_map.log_posterior, rtol=1e-10)
    np.testing.assert_allclose(
        _free(weighted_map, "estimate"), _free(replicated_map, "estimate"), atol=1e-6
    )
    np.testing.assert_allclose(_free(weighted_map, "se"), _free(replicated_map, "se"), rtol=1e-5)


# ---- standard-error warning, result layer, JSON ------------------------------------------------


def test_se_warning_only_with_weights_and_se() -> None:
    data = _sim(200, 4)
    spec = Model.uniform(TwoPL(), items=list(data.columns))
    weights = np.random.default_rng(4).uniform(0.5, 2.0, size=len(data))
    method = BAEM(tol=1e-4)
    for se in ("oakes", "hessian"):
        with pytest.warns(FlexpyirtWarning, match="sampling weights"):
            fit(spec, data, weights=weights, method=method, se=se)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FlexpyirtWarning)
        res = fit(spec, data, weights=weights, method=method, se=None)
        fit(spec, data, method=method)
    assert res.vcov is None


@pytest.fixture(scope="module")
def weighted_result() -> tuple[FitResult, pd.DataFrame, np.ndarray]:
    data = _sim(300, 5)
    weights = np.random.default_rng(6).uniform(0.5, 2.0, size=len(data))
    spec = Model.uniform(TwoPL(), items=list(data.columns))
    return _fit(spec, data, weights=weights, method=BAEM(tol=1e-6)), data, weights


def test_diagnostics_reject_weighted_fits(
    weighted_result: tuple[FitResult, pd.DataFrame, np.ndarray],
) -> None:
    res, data, _ = weighted_result
    calls: dict[str, Callable[[], object]] = {
        "item_fit": lambda: res.item_fit(data),
        "m2": lambda: res.m2(data),
        "local_dependence": lambda: res.local_dependence(data),
        "jsi": lambda: res.jsi(data),
        "dif": lambda: res.dif(data, group=np.repeat(["a", "b"], 150), items=["i0"]),
        "model_parallel_analysis": lambda: model_parallel_analysis(
            res, data, key=jax.random.key(0)
        ),
    }
    for name, call in calls.items():
        with pytest.raises(SpecificationError, match=rf"{name}\(\).*sampling weights"):
            call()


def test_scoring_and_summary_of_weighted_fit(
    weighted_result: tuple[FitResult, pd.DataFrame, np.ndarray],
) -> None:
    res, data, weights = weighted_result
    scores = res.score(data)
    assert len(scores) == len(data)
    assert np.isfinite(scores.to_numpy(dtype=float)).all()
    label = f"300 observations (weighted, sum of weights {weights.sum():.6g})"
    assert label in res.summary()
    assert label in res._repr_html_()
    np.testing.assert_allclose(
        res.bic, -2.0 * res.loglik + res.n_params * np.log(weights.sum()), rtol=1e-12
    )
    np.testing.assert_allclose(res.aic, -2.0 * res.loglik + 2.0 * res.n_params, rtol=1e-12)


def test_json_round_trip_keeps_sum_of_weights(
    weighted_result: tuple[FitResult, pd.DataFrame, np.ndarray],
) -> None:
    res, data, weights = weighted_result
    payload = json.loads(res.to_json())
    assert payload["schema_version"] == SCHEMA_VERSION == 7
    assert payload["sum_weights"] == float(weights.sum())
    loaded = FitResult.from_json(json.dumps(payload))
    assert loaded._sum_weights == res._sum_weights
    assert loaded.bic == res.bic
    with pytest.raises(SpecificationError, match="sampling weights"):
        loaded.item_fit(data)

    for bad in (0.0, -1.0, "NaN", "Infinity", None, True):
        broken = {**payload, "sum_weights": bad}
        with pytest.raises(SpecificationError, match="sum_weights"):
            FitResult.from_json(json.dumps(broken))

    # Version 6 has no sum_weights: such a file loads without it and rejects the key.
    with pytest.raises(SpecificationError, match=r"Unexpected keys.*sum_weights"):
        FitResult.from_json(json.dumps({**payload, "schema_version": 6}))
    old = {k: v for k, v in payload.items() if k != "sum_weights"} | {"schema_version": 6}
    assert FitResult.from_json(json.dumps(old))._sum_weights is None

    unweighted = json.loads(fit(res._model, data, method=BAEM(tol=1e-4), se=None).to_json())
    assert "sum_weights" not in unweighted
