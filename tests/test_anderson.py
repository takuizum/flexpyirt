"""Anderson-accelerated EM, ``BAEM(anderson=True)`` (DESIGN.md §15.49, #155).

Checks: settings and JSON, monotone log posterior on random problems, the same fixed point as
plain EM with far fewer EM map evaluations, and agreement with the R mirt references of the 3PL /
4PL (V11) and EH (V6) cases at those modules' tolerances.
"""

from __future__ import annotations

import json
import warnings
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from flexpyirt import (
    BAEM,
    GPCM,
    GRM,
    FitResult,
    Model,
    Quadrature,
    Rasch,
    ThreePL,
    TwoPL,
    fit,
)
from flexpyirt._json import SCHEMA_VERSION
from flexpyirt.errors import ConvergenceWarning, HeywoodWarning, SpecificationError
from flexpyirt.items import ItemModel
from tests import test_mirt_m1, test_mirt_m2, test_mirt_m3, test_mirt_m3_gpcm, test_mirt_v6_eh
from tests._reference import load as load_m1
from tests.test_m3_squarem import _simulate_2pl, _simulate_gpcm, _simulate_grm


def test_baem_anderson_option_default_and_validation() -> None:
    assert BAEM().anderson is False
    assert BAEM(anderson=True).anderson is True
    with pytest.raises(SpecificationError, match="anderson"):
        BAEM(anderson=1)  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match="at most one"):
        BAEM(anderson=True, squarem=True)


@cache
def _small_fit(anderson: bool) -> FitResult:
    df = _simulate_2pl(np.linspace(0.8, 1.6, 6), np.linspace(-1.0, 1.0, 6), 300, seed=3)
    return fit(
        Model.uniform(TwoPL(), items=df.columns),
        df,
        method=BAEM(tol=1e-6, anderson=anderson),
        se=None,
    )


def test_json_round_trip_and_old_schema(tmp_path: Path) -> None:
    res = _small_fit(True)
    path = tmp_path / "fit.json"
    res.to_json(path)
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == SCHEMA_VERSION == 10
    assert payload["method"]["anderson"] is True
    assert FitResult.from_json(path)._method == res._method

    old = dict(payload, schema_version=9)
    old["method"] = {k: v for k, v in payload["method"].items() if k != "anderson"}
    old_path = tmp_path / "old.json"
    old_path.write_text(json.dumps(old))
    assert FitResult.from_json(old_path)._method.anderson is False


def test_history_counters_and_converged_estimate() -> None:
    plain, aa = _small_fit(False), _small_fit(True)
    assert aa.converged
    assert aa.n_em_evaluations == int(aa.history["em_evaluations"].iloc[-1])
    assert aa.n_safeguards == int(aa.history["safeguard"].sum())
    assert aa.n_em_evaluations >= aa.n_iter
    assert aa.n_em_evaluations < plain.n_em_evaluations
    np.testing.assert_allclose(aa._eta, plain._eta, rtol=0, atol=1e-4)


def test_zero_free_parameters_falls_back_to_plain_em() -> None:
    case = load_m1("v1_lsat7_2pl")
    spec = Model.uniform(TwoPL(), items=case.items)
    table = spec.parameter_table(case.data)
    table["free"] = False
    plain = fit(spec, case.data, table=table, se=None)
    aa = fit(spec, case.data, table=table, method=BAEM(anderson=True), se=None)
    assert aa.converged and aa.n_iter == plain.n_iter
    assert aa.loglik == pytest.approx(plain.loglik, rel=1e-15)


@pytest.mark.slow
@settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    seed=st.integers(0, 2**31 - 1),
    kind=st.sampled_from(["2pl", "rasch", "grm", "3pl", "gpcm"]),
)
def test_anderson_is_monotone(seed: int, kind: str) -> None:
    rng = np.random.default_rng(seed)
    if kind in ("2pl", "rasch", "3pl"):
        a = np.ones(6) if kind == "rasch" else rng.lognormal(0.1, 0.35, 6)
        d = rng.normal(0.0, 1.0, 6)
        df = _simulate_2pl(a, d, 400, seed, sd=1.3 if kind == "rasch" else 1.0)
        if (df.nunique() < 2).any():
            return
        item_models: dict[str, ItemModel] = {"2pl": TwoPL(), "rasch": Rasch(), "3pl": ThreePL()}
        spec = Model.uniform(item_models[kind], items=df.columns)
    elif kind == "grm":
        a = rng.lognormal(0.1, 0.3, 5)
        gaps = rng.uniform(0.5, 1.2, (5, 2))
        d1 = rng.normal(0.8, 0.3, 5)
        d = np.column_stack([d1, d1 - gaps[:, 0], d1 - gaps[:, 0] - gaps[:, 1]])
        df = _simulate_grm(a, d, 500, seed)
        if any(df[c].nunique() < 4 for c in df.columns):
            return
        spec = Model.uniform(GRM(n_categories=4), items=df.columns)
    else:
        a = rng.lognormal(0.1, 0.3, 5)
        d = rng.normal(0.0, 0.8, (5, 3))
        df = _simulate_gpcm(a, d, 500, seed)
        if any(df[c].nunique() < 4 for c in df.columns):
            return
        spec = Model.uniform(GPCM(n_categories=4), items=df.columns)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", HeywoodWarning)
        warnings.simplefilter("ignore", ConvergenceWarning)
        res = fit(spec, df, method=BAEM(tol=1e-7, max_iter=200, anderson=True), se=None)

    ll = res.history["loglik"].to_numpy()
    assert np.all(np.diff(ll) >= -1e-9 * np.abs(ll[1:]))
    assert res.loglik >= ll[-1] - 1e-9 * abs(ll[-1])
    assert res.n_em_evaluations == int(res.history["em_evaluations"].iloc[-1])
    free = res.parameters[res.parameters["free"]]
    assert (free["estimate"] > free["lower"]).all()
    assert (free["estimate"] < free["upper"]).all()


@pytest.mark.slow
@pytest.mark.parametrize(
    "case_id",
    ["v2_lsat7_rasch", "v2_sim_rasch", "v4_twogroup_2pl", "v8_science_gpcm"],
)
def test_anderson_reaches_plain_em_fixed_point_with_fewer_evaluations(case_id: str) -> None:
    """Same fixed point as plain EM at ``tol = 1e-9`` with less than half the EM evaluations."""
    if case_id.startswith("v2_"):
        c1 = load_m1(case_id)
        spec, data, group, quad = test_mirt_m1._spec(c1), c1.data, None, Quadrature.equispaced()
    elif case_id.startswith("v4_"):
        c2 = test_mirt_m2.load(case_id)
        spec, data, group, quad = c2.spec, c2.data, c2.group, c2.quadrature
    else:
        c3 = test_mirt_m3_gpcm.load(case_id)
        spec, data, group, quad = c3.spec, c3.data, None, c3.quadrature
    runs = {
        name: fit(
            spec,
            data,
            group=group,
            method=BAEM(tol=1e-9, max_iter=20000, anderson=name == "anderson"),
            quadrature=quad,
            se=None,
        )
        for name in ("plain", "anderson")
    }
    plain, aa = runs["plain"], runs["anderson"]
    assert plain.converged and aa.converged
    assert aa.n_em_evaluations < 0.5 * plain.n_em_evaluations
    np.testing.assert_allclose(aa._eta, plain._eta, rtol=0, atol=1e-6)
    assert aa.loglik == pytest.approx(plain.loglik, rel=1e-10)


@pytest.mark.slow
@pytest.mark.parametrize("name", test_mirt_m3.CASES)
def test_anderson_matches_mirt_3pl_4pl(name: str) -> None:
    """V11 at the tolerances of ``tests/test_mirt_m3.py`` (boundary-free 3PL / 4PL)."""
    case = test_mirt_m3.load(name)
    with warnings.catch_warnings():
        warnings.simplefilter("error", HeywoodWarning)
        res = fit(
            case.spec,
            case.data,
            method=BAEM(tol=1e-9, max_iter=20000, anderson=True),
            quadrature=case.quadrature,
            se="oakes",
        )
    assert res.converged
    ref_ll = float(case.raw["loglik"])
    assert res.loglik == pytest.approx(ref_ll, rel=test_mirt_m3.TOL_LOGLIK_REL)
    ref = case.params()
    ours = res.items.set_index("item").loc[ref.index]
    for p in ("a", "d", "g", "u"):
        np.testing.assert_allclose(
            ours[p], ref[p].astype(float), rtol=0, atol=test_mirt_m3.TOL_PARAM
        )
    for col in ("a_se", "d_se", "g_se", "u_se"):
        expected = ref[col].astype(float).to_numpy()
        got = ours[col].to_numpy()
        free = np.isfinite(expected)
        np.testing.assert_allclose(got[free], expected[free], rtol=test_mirt_m3.TOL_SE_REL)


@pytest.mark.slow
@pytest.mark.parametrize("name", test_mirt_v6_eh.CASES)
def test_anderson_matches_mirt_eh(name: str) -> None:
    """V6 (EH, FIPC) at the tolerances of ``tests/test_mirt_v6_eh.py``; monotone history."""
    case = test_mirt_v6_eh.load(name)
    res = fit(
        case.spec,
        case.data,
        group=case.group,
        method=BAEM(tol=1e-6, max_iter=20000, anderson=True),
        quadrature=test_mirt_v6_eh.QUAD,
        se=None,
    )
    assert res.converged
    ll = res.history["loglik"].to_numpy()
    assert np.all(np.diff(ll) >= -1e-10 * np.abs(ll[1:]))
    ref = float(case.raw["loglik"])
    assert res.loglik >= ref - 1e-7 * abs(ref)
    assert res.loglik - ref <= 1e-5 * abs(ref)
    for g in case.groups:
        ref_items = case.params(g)
        ours = test_mirt_v6_eh._items(res, g).loc[ref_items.index]
        np.testing.assert_allclose(ours["a"], ref_items["a"], rtol=0, atol=5e-4)
        np.testing.assert_allclose(ours["d"], ref_items["d"], rtol=0, atol=5e-4)
