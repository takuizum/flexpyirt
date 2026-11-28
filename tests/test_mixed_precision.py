"""The optional ``float32`` warmup, ``BAEM(warmup_dtype="float32")`` (DESIGN.md §28).

The warmup only moves the starting values: the estimates, log-likelihood and standard errors
come from a ``float64`` run on the original problem that stops at the user's ``tol``. So a fit
with the warmup must agree with a pure ``float64`` fit to about ``tol``, and the default must not
run any ``float32`` code. The mirt comparisons with the warmup on are in ``test_mirt_m3.py`` and
``test_mirt_m4_mirt.py`` (V58).
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from functools import cache
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from flexpyirt import (
    BAEM,
    GRM,
    EmpiricalHistogram,
    FitResult,
    LogNormalPrior,
    Model,
    Normal,
    TwoPL,
    _core,
    fit,
)
from flexpyirt._json import SCHEMA_VERSION
from flexpyirt.errors import FlexpyirtWarning, SpecificationError

TOL = 1e-7
"""EM tolerance of the comparisons: both fits stop within a few ``tol`` of the same maximum."""
TOL_EH = 1e-4
"""EM tolerance with ``EmpiricalHistogram``, whose EM converges too slowly to reach ``TOL``."""
TOL_ETA = 1e-5
TOL_ETA_EH = 1e-3
TOL_LOGLIK = 1e-8
TOL_SE_REL = 1e-4


def _simulate(n: int, n_items: int, seed: int, n_categories: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=(n, 1))
    a = rng.uniform(0.8, 2.0, size=n_items)
    if n_categories == 2:
        d = rng.normal(size=n_items)
        y = (rng.random((n, n_items)) < 1.0 / (1.0 + np.exp(-(a * theta + d)))).astype(float)
    else:
        cuts = np.sort(rng.normal(size=(n_items, n_categories - 1)), axis=1)
        latent = a * theta + rng.logistic(size=(n, n_items))
        y = (latent[:, :, None] > cuts[None] * a[None, :, None]).sum(axis=2).astype(float)
    y[rng.random(y.shape) < 0.03] = np.nan
    return pd.DataFrame(y, columns=[f"q{j + 1:02d}" for j in range(n_items)])


def _specs() -> dict[str, tuple[Model, pd.DataFrame, Any]]:
    df = _simulate(1500, 12, seed=1)
    poly = _simulate(1200, 8, seed=2, n_categories=4)
    groups = np.repeat(["ref", "focal"], [800, 700])
    two_group = (
        Model.uniform(TwoPL(), items=df.columns)
        .with_groups(ref=Normal.standard(), focal=Normal.free())
        .equal_across_groups(items=list(df.columns[:6]))
    )
    return {
        "2pl": (Model.uniform(TwoPL(), items=df.columns), df, None),
        "grm": (Model.uniform(GRM(n_categories=4), items=poly.columns), poly, None),
        "two_group": (two_group, df, groups),
        "eh": (
            Model.uniform(TwoPL(), items=df.columns).with_latent(EmpiricalHistogram()),
            df,
            None,
        ),
        "map": (
            Model.uniform(TwoPL(), items=df.columns).prior(
                param="a", dist=LogNormalPrior(0.0, 0.5), items=list(df.columns)
            ),
            df,
            None,
        ),
    }


CASES = ("2pl", "grm", "two_group", "eh", "map")


def _tol(case: str) -> float:
    return TOL_EH if case == "eh" else TOL


@cache
def _fit(case: str, warmup: str | None, squarem: bool) -> FitResult:
    spec, data, group = _specs()[case]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FlexpyirtWarning)
        return fit(
            spec,
            data,
            group=group,
            method=BAEM(tol=_tol(case), squarem=squarem, warmup_dtype=warmup),  # type: ignore[arg-type]
            se="oakes",
        )


# ---- options ----------------------------------------------------------------------------------


def test_default_has_no_warmup() -> None:
    assert BAEM().warmup_dtype is None


@pytest.mark.parametrize("value", ["float16", "float64", "Float32", "", 32])
def test_invalid_warmup_dtype_raises(value: object) -> None:
    with pytest.raises(SpecificationError, match="warmup_dtype"):
        BAEM(warmup_dtype=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(("tol", "expected"), [(1e-4, 1e-3), (1e-6, 1e-3), (1e-2, 0.1)])
def test_warmup_tol(tol: float, expected: float) -> None:
    assert BAEM(tol=tol, warmup_dtype="float32").warmup_tol == pytest.approx(expected)


def test_cast_floats_keeps_integer_leaves() -> None:
    spec, data, group = _specs()["two_group"]
    from flexpyirt._compile import compile_problem, prepare
    from flexpyirt.quadrature import Quadrature

    prep = prepare(spec, data, group=group, item_names=None, table=None, weights=None, cluster=None)
    problem = compile_problem(
        spec, prep.y, prep.group, prep.table, Quadrature.default(1), weights=None
    ).problem
    cast = _core.cast_floats(problem, jnp.float32)
    for before, after in zip(
        jax.tree_util.tree_leaves(problem), jax.tree_util.tree_leaves(cast), strict=True
    ):
        before_dtype = jnp.asarray(before).dtype
        if jnp.issubdtype(before_dtype, jnp.floating):
            assert jnp.asarray(after).dtype == jnp.float32
        else:
            assert jnp.asarray(after).dtype == before_dtype
            np.testing.assert_array_equal(np.asarray(after), np.asarray(before))
    assert str(jax.tree_util.tree_structure(cast)) == str(jax.tree_util.tree_structure(problem))


# ---- agreement with pure float64 --------------------------------------------------------------


@pytest.mark.parametrize("squarem", [False, True])
@pytest.mark.parametrize("case", CASES)
def test_warmup_matches_float64(case: str, squarem: bool) -> None:
    """Estimates, log-likelihood and Oakes SEs agree with a pure float64 fit to about ``tol``."""
    ref, res = _fit(case, None, squarem), _fit(case, "float32", squarem)
    assert ref.converged and res.converged
    assert ref.n_warmup_iter == 0 and res.n_warmup_iter > 0
    if case == "eh" and squarem:
        # The EH likelihood is so flat along the histogram tails that EM and SQUAREM stop far
        # apart (0.19 here) at the same tol; any change of start moves SQUAREM's stopping point
        # (0.018 for the warmup). Require no more than the EM-vs-SQUAREM gap.
        other = _fit(case, None, False)
        gap = float(np.max(np.abs(other._eta - ref._eta)))
        assert float(np.max(np.abs(res._eta - ref._eta))) <= gap
        assert abs(res.loglik - ref.loglik) <= abs(other.loglik - ref.loglik)
        return
    atol = TOL_ETA_EH if case == "eh" else TOL_ETA
    np.testing.assert_allclose(res._eta, ref._eta, rtol=0, atol=atol)
    assert res.loglik == pytest.approx(ref.loglik, rel=TOL_LOGLIK)
    assert res.log_posterior == pytest.approx(ref.log_posterior, rel=TOL_LOGLIK)
    assert ref.vcov is not None and res.vcov is not None
    np.testing.assert_allclose(
        np.sqrt(np.diag(res.vcov)), np.sqrt(np.diag(ref.vcov)), rtol=TOL_SE_REL
    )


@pytest.mark.parametrize("case", CASES)
def test_float64_stage_meets_tol(case: str) -> None:
    """The reported iterations are the float64 ones, which stop at the user's ``tol``."""
    res = _fit(case, "float32", False)
    assert res._eta.dtype == np.float64
    assert res.history["max_change"].iloc[-1] < _tol(case)
    assert len(res.history) == res.n_iter
    # The float32 stage brings the start close, so the float64 stage is short.
    assert res.n_iter < _fit(case, None, False).n_iter
    lp = res.history["log_posterior"].to_numpy()
    assert np.all(np.diff(lp) >= -1e-9 * np.abs(lp[1:]))


def test_non_finite_warmup_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """A warmup with NaN results is discarded; the fit then equals pure float64 exactly."""
    spec, data, _ = _specs()["2pl"]

    cast_floats = _core.cast_floats

    def nan_problem(prob: _core.CompiledProblem, dtype: Any) -> _core.CompiledProblem:
        cast = cast_floats(prob, dtype)
        return dataclasses.replace(cast, counts=jnp.full_like(cast.counts, jnp.nan))

    monkeypatch.setattr(_core, "cast_floats", nan_problem)
    method = BAEM(tol=1e-4, warmup_dtype="float32", max_iter=50)
    with pytest.warns(FlexpyirtWarning, match="non-finite"):
        res = fit(spec, data, method=method, se=None)
    ref = fit(spec, data, method=BAEM(tol=1e-4), se=None)
    np.testing.assert_array_equal(res._eta, ref._eta)
    assert res.loglik == ref.loglik
    assert res.n_iter == ref.n_iter


def test_summary_mentions_warmup() -> None:
    res = _fit("2pl", "float32", False)
    assert f"after {res.n_warmup_iter} float32 warmup iterations" in res._convergence_label()
    assert "warmup" not in _fit("2pl", None, False)._convergence_label()


# ---- JSON -------------------------------------------------------------------------------------


def test_json_round_trip() -> None:
    res = _fit("2pl", "float32", False)
    raw = json.loads(res.to_json())
    assert raw["schema_version"] == SCHEMA_VERSION
    assert raw["method"]["warmup_dtype"] == "float32"
    assert raw["n_warmup_iter"] == res.n_warmup_iter
    loaded = FitResult.from_json(json.dumps(raw))
    assert loaded._method == res._method
    assert loaded.n_warmup_iter == res.n_warmup_iter

    plain = json.loads(_fit("2pl", None, False).to_json())
    assert plain["method"]["warmup_dtype"] is None and "n_warmup_iter" not in plain
    assert FitResult.from_json(json.dumps(plain)).n_warmup_iter == 0

    # n_warmup_iter needs a method with a warmup.
    bad = {**raw, "method": {**raw["method"], "warmup_dtype": None}}
    with pytest.raises(SpecificationError, match="n_warmup_iter"):
        FitResult.from_json(json.dumps(bad))
    # Version 9 has neither key.
    with pytest.raises(SpecificationError, match="warmup_dtype"):
        FitResult.from_json(json.dumps({**plain, "schema_version": 9}))
    old = {**plain, "schema_version": 9}
    old["method"] = {k: v for k, v in plain["method"].items() if k != "warmup_dtype"}
    assert FitResult.from_json(json.dumps(old))._method.warmup_dtype is None
    with pytest.raises(SpecificationError, match="n_warmup_iter"):
        FitResult.from_json(json.dumps({**old, "n_warmup_iter": 3}))
