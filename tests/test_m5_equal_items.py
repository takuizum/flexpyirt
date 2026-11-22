"""Unit tests for equality constraints across items (DESIGN.md §16.1)."""

from __future__ import annotations

import json
from typing import Any

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
    Quadrature,
    Rasch,
    TwoPL,
    _core,
    fit,
)
from flexpyirt._compile import compile_problem, prepare
from flexpyirt.errors import IdentificationError, SpecificationError

TIGHT = BAEM(tol=1e-8, max_iter=5000)


def _val(frame: pd.DataFrame, item: str, col: str) -> float:
    value: Any = frame.at[item, col]
    return float(value)


def _sim(n: int = 1500, j: int = 6, seed: int = 3, n_categories: int = 2) -> pd.DataFrame:
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


def test_spec_validation() -> None:
    spec = Model.uniform(TwoPL(), items=["x", "y", "z"])
    with pytest.raises(SpecificationError, match="at least two items"):
        spec.equal_across_items(items=["x"], params=["a"])
    with pytest.raises(SpecificationError, match="no parameters"):
        spec.equal_across_items(items=["x", "y"], params=["g"])
    with pytest.raises(SpecificationError, match="not a str"):
        spec.equal_across_items(items=["x", "y"], params="a")
    with pytest.raises(SpecificationError, match="at least one parameter"):
        spec.equal_across_items(items=["x", "y"], params=[])
    with pytest.raises(SpecificationError, match="Unknown items"):
        spec.equal_across_items(items=["x", "w"], params=["a"])
    eq = spec.equal_across_items(items=["x", "y"], params=["a"])
    with pytest.raises(SpecificationError, match="before"):
        eq.with_factors({"F1": ["x", "y"], "F2": ["z"]})


def test_table_shares_ids_and_averages_starts() -> None:
    df = _sim()
    base = Model.uniform(TwoPL(), items=df.columns)
    t0 = base.parameter_table(df)
    t1 = base.equal_across_items(items=["i0", "i1", "i2"], params=["a"]).parameter_table(df)
    slopes = t1[(t1["param"] == "a") & t1["item"].isin(["i0", "i1", "i2"])]
    assert slopes["free_id"].nunique() == 1
    expected = t0.loc[slopes.index, "value"].mean()
    np.testing.assert_allclose(slopes["value"], expected)
    assert t1["free_id"].max() == t0["free_id"].max() - 2
    # free ids stay contiguous and in row order
    ids = t1.loc[t1["free"], "free_id"].to_numpy()
    assert ids[0] == 0 and np.all(np.diff(np.unique(ids)) == 1)


def test_union_with_across_group_equalities() -> None:
    df = _sim(n=1200)
    grp = np.array(["r", "f"] * 600)
    spec = (
        Model.uniform(TwoPL(), items=df.columns)
        .with_groups(r=Normal.standard(), f=Normal.free())
        .equal_across_groups(items=["i0", "i1", "i2"])
        .equal_across_items(items=["i0", "i3"], params=["a"])
    )
    t = spec.parameter_table(df, group=grp)
    a_rows = t[(t["param"] == "a")].set_index(["group", "item"])["free_id"]
    # i0 is equal across groups and to i3 within each group, so all four rows share one id.
    assert a_rows[("r", "i0")] == a_rows[("f", "i0")] == a_rows[("r", "i3")] == a_rows[("f", "i3")]
    assert a_rows[("r", "i4")] != a_rows[("f", "i4")]


def test_coupled_blocks_are_compiled_only_when_needed() -> None:
    df = _sim()
    quad = Quadrature.equispaced()
    plain = Model.uniform(TwoPL(), items=df.columns)
    p0 = prepare(plain, df, group=None, item_names=None, table=None)
    c0 = compile_problem(plain, p0.y, p0.group, p0.table, quad)
    assert c0.problem.coupled == ()
    spec = plain.equal_across_items(items=["i0", "i1"], params=["a"]).equal_across_items(
        items=["i4", "i5"], params=["d"]
    )
    p1 = prepare(spec, df, group=None, item_names=None, table=None)
    c1 = compile_problem(spec, p1.y, p1.group, p1.table, quad)
    assert len(c1.problem.coupled) == 2
    sizes = sorted(int(b.slot_free_index.shape[0]) for b in c1.problem.coupled)
    assert sizes == [3, 3]  # (a, d0, d1) and (a4, a5, d)


def test_common_slope_matches_shared_free_id_loglik_and_is_monotone() -> None:
    df = _sim()
    spec = Model.uniform(TwoPL(), items=df.columns).equal_across_items(items=None, params=["a"])
    res = fit(spec, df, method=TIGHT, se="oakes")
    assert res.converged
    assert res.items["a"].nunique() == 1
    assert res.n_params == df.shape[1] + 1
    ll = res.history["log_posterior"].to_numpy()
    assert np.all(np.diff(ll) >= -1e-9 * np.abs(ll[1:]))
    # Nested in the unconstrained 2PL, never above it.
    free = fit(Model.uniform(TwoPL(), items=df.columns), df, method=TIGHT, se=None)
    assert res.loglik <= free.loglik + 1e-8
    # SQUAREM reaches the same fixed point.
    sq = fit(spec, df, method=BAEM(tol=1e-8, max_iter=5000, squarem=True), se=None)
    np.testing.assert_allclose(sq.items["a"], res.items["a"], atol=1e-6)
    assert sq.loglik == pytest.approx(res.loglik, abs=1e-7)


def test_shared_prior_is_counted_once() -> None:
    df = _sim()
    prior = LogNormalPrior(0.0, 0.5)
    spec = (
        Model.uniform(TwoPL(), items=df.columns)
        .prior(param="a", dist=prior)
        .equal_across_items(items=None, params=["a"])
    )
    res = fit(spec, df, method=TIGHT, se="oakes")
    a = float(res.items["a"].iloc[0])
    expected = res.loglik + float(prior.log_pdf(jnp.asarray(a)))
    assert res.log_posterior == pytest.approx(expected, rel=1e-12)
    hess = fit(spec, df, method=TIGHT, se="hessian")
    assert res.vcov is not None and hess.vcov is not None
    np.testing.assert_allclose(res.vcov, hess.vcov, rtol=1e-6, atol=1e-12)


def test_mixed_families_oakes_equals_hessian() -> None:
    df2 = _sim(n=1500, j=4, seed=5)
    df3 = _sim(n=1500, j=3, seed=5, n_categories=3).rename(columns=lambda c: f"g{c}")
    df = pd.concat([df2, df3], axis=1)
    spec = Model.from_items(
        {**{c: TwoPL() for c in df2.columns}, **{c: GRM(n_categories=3) for c in df3.columns}}
    ).equal_across_items(items=["i0", "gi0", "gi1"], params=["a"])
    oakes = fit(spec, df, method=TIGHT, se="oakes")
    hess = fit(spec, df, method=TIGHT, se="hessian")
    assert oakes.converged
    assert oakes.vcov is not None and hess.vcov is not None
    np.testing.assert_allclose(oakes.vcov, hess.vcov, rtol=1e-6, atol=1e-12)
    items = oakes.items.set_index("item")
    slopes = [_val(items, i, "a") for i in ("i0", "gi0", "gi1")]
    assert slopes[0] == slopes[1] == slopes[2]
    assert _val(items, "i0", "a_se") == _val(items, "gi1", "a_se")


def test_rasch_rows_are_fixed_so_equal_slopes_are_a_no_op() -> None:
    df = _sim()
    base = Model.uniform(Rasch(), items=df.columns)
    eq = base.equal_across_items(items=None, params=["a"])
    t0, t1 = base.parameter_table(df), eq.parameter_table(df)
    pd.testing.assert_frame_equal(t0, t1)


def test_common_slope_with_free_variance_is_unidentified() -> None:
    df = _sim()
    spec = (
        Model.uniform(TwoPL(), items=df.columns)
        .with_latent(Normal(free_var=True))
        .equal_across_items(items=None, params=["a"])
    )
    with pytest.raises(IdentificationError):
        fit(spec, df)


def test_latent_and_item_cannot_share_free_id() -> None:
    df = _sim()
    spec = Model.uniform(TwoPL(), items=df.columns).with_latent(Normal(free_mean=False))
    table = spec.parameter_table(df)
    table.loc[(table["item"] == "i0") & (table["param"] == "a"), "free_id"] = 0
    var_row = (table["item"] == "__latent__") & (table["param"] == "var")
    table.loc[var_row, ["free", "free_id"]] = [True, 0]
    with pytest.raises(SpecificationError, match="latent-distribution"):
        fit(spec, df, table=table)


def test_eh_with_equal_items_runs_and_is_monotone() -> None:
    df = _sim(n=1500)
    spec = (
        Model.uniform(TwoPL(), items=df.columns)
        .with_latent(EmpiricalHistogram())
        .equal_across_items(items=["i0", "i1", "i2"], params=["a"])
    )
    res = fit(spec, df, method=BAEM(tol=1e-5, max_iter=3000), se="oakes")
    ll = res.history["loglik"].to_numpy()
    assert np.all(np.diff(ll) >= -1e-9 * np.abs(ll[1:]))
    items = res.items.set_index("item")
    assert _val(items, "i0", "a") == _val(items, "i2", "a")


def test_multidimensional_equal_loadings() -> None:
    rng = np.random.default_rng(11)
    n = 1500
    theta = rng.multivariate_normal([0, 0], [[1, 0.4], [0.4, 1]], size=n)
    load = np.array([[1.2, 0], [1.2, 0], [1.0, 0], [0, 1.1], [0, 1.1], [0, 0.9]])
    d = rng.normal(0, 0.7, size=6)
    y = (rng.random((n, 6)) < 1 / (1 + np.exp(-(theta @ load.T + d)))).astype(float)
    df = pd.DataFrame(y, columns=[f"i{k}" for k in range(6)])
    spec = (
        Model.uniform(TwoPL(), items=df.columns)
        .with_factors({"F1": ["i0", "i1", "i2"], "F2": ["i3", "i4", "i5"]})
        .equal_across_items(items=["i0", "i1"], params=["a1"])
        .equal_across_items(items=["i3", "i4"], params=["a2"])
    )
    oakes = fit(spec, df, method=TIGHT, se="oakes")
    hess = fit(spec, df, method=TIGHT, se="hessian")
    items = oakes.items.set_index("item")
    assert _val(items, "i0", "a1") == _val(items, "i1", "a1")
    assert _val(items, "i3", "a2") == _val(items, "i4", "a2")
    assert oakes.vcov is not None and hess.vcov is not None
    np.testing.assert_allclose(oakes.vcov, hess.vcov, rtol=1e-6, atol=1e-12)


def test_json_round_trip() -> None:
    df = _sim()
    spec = Model.uniform(TwoPL(), items=df.columns).equal_across_items(
        items=["i0", "i1"], params=["a", "d"]
    )
    res = fit(spec, df, method=BAEM(tol=1e-4), se="oakes")
    payload = res.to_json()
    assert json.loads(payload)["model"]["equal_items"] == [
        {"items": ["i0", "i1"], "params": ["a", "d"]}
    ]
    loaded = FitResult.from_json(payload)
    assert loaded._model == spec
    pd.testing.assert_frame_equal(loaded.items, res.items, check_exact=True)
    bad = json.loads(payload)
    bad["model"]["equal_items"][0]["items"] = ["i0"]
    with pytest.raises(SpecificationError, match="at least two items"):
        FitResult.from_json(json.dumps(bad))
    bad["model"]["equal_items"][0]["items"] = ["i0", "nope"]
    with pytest.raises(SpecificationError, match="unknown items"):
        FitResult.from_json(json.dumps(bad))
    v2 = json.loads(payload)
    v2["schema_version"] = 2
    with pytest.raises(SpecificationError, match="equal_items"):
        FitResult.from_json(json.dumps(v2))


def test_core_m_step_does_not_decrease_q() -> None:
    """One M-step on a coupled problem never decreases Q (generalized EM, §16.1)."""
    df = _sim()
    spec = Model.uniform(TwoPL(), items=df.columns).equal_across_items(
        items=["i0", "i1", "i2", "i3"], params=["a"]
    )
    prep = prepare(spec, df, group=None, item_names=None, table=None)
    comp = compile_problem(spec, prep.y, prep.group, prep.table, Quadrature.equispaced())
    prob, eta = comp.problem, jnp.asarray(comp.eta0)
    counts, _ = _core.e_step(prob, eta)
    new = _core.m_step(prob, eta, counts, 3)
    q0 = _core.expected_complete_loglik(prob, eta, counts)
    q1 = _core.expected_complete_loglik(prob, new, counts)
    assert float(q1) >= float(q0)
    assert float(jnp.max(jnp.abs(new - eta))) > 0
