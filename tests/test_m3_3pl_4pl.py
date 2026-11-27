"""Unit and property tests for 3PL and 4PL models, kernels, bounds, and HeywoodWarning."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from flexpyirt import BAEM, FourPL, Model, Normal, Rasch, ThreePL, TwoPL, fit
from flexpyirt.errors import HeywoodWarning, IdentificationError, SpecificationError
from flexpyirt.items import DichotomousKernel, FourPLKernel, ItemModel


def _sim_4pl(
    a: np.ndarray,
    d: np.ndarray,
    g: np.ndarray,
    u: np.ndarray,
    n: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, 1.0, size=(n, 1))
    p = g + (u - g) / (1.0 + np.exp(-(theta * a + d)))
    y = (rng.random(p.shape) < p).astype(float)
    return pd.DataFrame(y, columns=[f"q{j:02d}" for j in range(1, len(a) + 1)])


# ---- FourPLKernel --------------------------------------------------------------------------


@settings(deadline=None)
@given(
    a=st.floats(-4.0, 4.0),
    d=st.floats(-6.0, 6.0),
    g=st.floats(0.0, 0.45),
    u=st.floats(0.55, 1.0),
)
def test_fourpl_probabilities_sum_to_one(a: float, d: float, g: float, u: float) -> None:
    theta = jnp.linspace(-6.0, 6.0, 25)
    lp = FourPLKernel().log_prob(theta, jnp.array([a, d, g, u]))
    p = np.exp(np.asarray(lp))
    np.testing.assert_allclose(p.sum(axis=1), 1.0, rtol=1e-12)
    expected_p1 = g + (u - g) / (1.0 + np.exp(-(a * np.asarray(theta) + d)))
    np.testing.assert_allclose(p[:, 1], expected_p1, rtol=1e-11, atol=1e-14)


@pytest.mark.slow
@settings(deadline=None)
@given(
    a=st.floats(-4.0, 4.0),
    d=st.floats(-6.0, 6.0),
    g=st.one_of(st.just(0.0), st.floats(1e-6, 0.4)),
    u=st.one_of(st.just(1.0), st.floats(0.6, 1.0 - 1e-6)),
)
def test_fourpl_finite_grad_and_hessian_at_boundaries(
    a: float, d: float, g: float, u: float
) -> None:
    """Gradients and Hessians for all 4 params are finite even when g=0 and/or u=1."""
    theta = jnp.linspace(-6.0, 6.0, 13)
    params = jnp.array([a, d, g, u])
    kernel = FourPLKernel()

    def obj(p: jax.Array) -> jax.Array:
        return kernel.log_prob(theta, p).sum()

    grad = np.asarray(jax.grad(obj)(params))
    hess = np.asarray(jax.hessian(obj)(params))
    assert np.isfinite(grad).all(), f"non-finite grad at {(a, d, g, u)}: {grad}"
    assert np.isfinite(hess).all(), f"non-finite hess at {(a, d, g, u)}: {hess}"


def test_fourpl_invalid_asymptotes_give_nan() -> None:
    theta = jnp.array([0.0])
    kernel = FourPLKernel()
    bad_cases = (
        [1.0, 0.0, 0.5, 0.5],
        [1.0, 0.0, 0.7, 0.4],
        [1.0, 0.0, -0.1, 1.0],
        [1.0, 0.0, 0.1, 1.1],
    )
    for bad in bad_cases:
        lp = kernel.log_prob(theta, jnp.array(bad))
        assert np.isnan(np.asarray(lp)).any()


def test_threepl_fixed_g0_equals_2pl_kernel() -> None:
    theta = jnp.linspace(-6.0, 6.0, 61)
    for a, d in [(1.2, -0.5), (0.5, 2.0), (2.5, -3.0)]:
        lp_2pl = DichotomousKernel().log_prob(theta, jnp.array([a, d]))
        lp_4pl = FourPLKernel().log_prob(theta, jnp.array([a, d, 0.0, 1.0]))
        np.testing.assert_array_equal(np.asarray(lp_4pl), np.asarray(lp_2pl))


@pytest.mark.slow
def test_threepl_fixed_g0_equals_2pl_fit() -> None:
    a = np.array([1.1, 1.4, 0.9, 1.3, 1.0, 1.2])
    d = np.array([-0.6, 0.3, -0.2, 0.5, 0.0, -0.4])
    df = _sim_4pl(a, d, np.zeros_like(a), np.ones_like(a), n=800, seed=101)
    method = BAEM(tol=1e-9, max_iter=5000)

    res_2pl = fit(Model.uniform(TwoPL(), items=df.columns), df, method=method, se="oakes")
    spec_3pl = Model.uniform(ThreePL(), items=df.columns)
    for col in df.columns:
        spec_3pl = spec_3pl.fix(item=col, g=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", HeywoodWarning)
        res_3pl = fit(spec_3pl, df, method=method, se="oakes")

    assert res_3pl.converged
    assert res_3pl.loglik == pytest.approx(res_2pl.loglik, rel=1e-12)
    np.testing.assert_allclose(res_3pl.items["a"], res_2pl.items["a"], atol=1e-8)
    np.testing.assert_allclose(res_3pl.items["d"], res_2pl.items["d"], atol=1e-8)
    np.testing.assert_allclose(res_3pl.items["a_se"], res_2pl.items["a_se"], rtol=1e-7)
    np.testing.assert_allclose(res_3pl.items["d_se"], res_2pl.items["d_se"], rtol=1e-7)
    assert (res_3pl.items["g"] == 0.0).all()
    assert (res_3pl.items["u"] == 1.0).all()
    assert res_3pl.items["g_se"].isna().all()
    assert res_3pl.items["u_se"].isna().all()
    np.testing.assert_allclose(res_3pl.score(df)["theta"], res_2pl.score(df)["theta"], atol=1e-8)


# ---- HeywoodWarning ------------------------------------------------------------------------


def test_heywood_warning_on_boundary_guessing() -> None:
    """An easy item with zero true guessing drives g -> 0 and triggers HeywoodWarning."""
    a = np.array([1.5, 1.4, 1.3, 1.2, 1.1, 1.6, 1.4, 1.2])
    d = np.array([2.2, -0.5, -0.8, -0.3, -1.0, -0.4, -0.7, -0.2])
    g = np.array([0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
    df = _sim_4pl(a, d, g, np.ones_like(a), n=1500, seed=202)
    spec = Model.uniform(ThreePL(), items=df.columns)
    with pytest.warns(HeywoodWarning, match="q01"):
        res = fit(spec, df, method=BAEM(tol=1e-6, max_iter=3000), se=None)
    assert float(res.items.set_index("item")["g"].loc["q01"]) < 1e-4


def test_heywood_warning_on_steep_slope() -> None:
    """A near-deterministic item yields |a| > 20 and triggers HeywoodWarning."""
    rng = np.random.default_rng(303)
    n = 1000
    theta = rng.normal(0.0, 1.0, size=(n, 1))
    a = np.array([1.2, 1.4, 1.1, 1.3, 1.5, 1.0])
    d = np.array([-0.3, 0.2, -0.5, 0.4, -0.1, 0.1])
    p = 1.0 / (1.0 + np.exp(-(theta * a + d)))
    y = (rng.random(p.shape) < p).astype(float)
    # Add an item that is an exact threshold on theta (quasi-complete separation).
    y_sep = (theta[:, 0] > 0.0).astype(float)
    # Give a second copy of y_sep so latent theta is pinned closely to the sign of theta.
    df = pd.DataFrame(
        np.column_stack([y, y_sep, y_sep, y_sep, y_sep]),
        columns=[f"q{j:02d}" for j in range(1, 11)],
    )
    spec = Model.uniform(TwoPL(), items=df.columns)
    with pytest.warns(HeywoodWarning, match="q07"):
        res = fit(spec, df, method=BAEM(tol=1e-4, max_iter=500), se=None)
    assert float(res.items.set_index("item")["a"].loc["q07"]) > 20.0


# ---- Mixed models, FIPC, identification, validation ----------------------------------------


@pytest.mark.slow
def test_mixed_twopl_threepl_fourpl_model() -> None:
    a = np.array([1.5, 1.6, 1.8, 1.7, 2.2, 2.1, 1.8, 1.6])
    d = np.array([0.2, -0.3, -0.6, -0.7, 0.0, -0.1, -0.5, 0.4])
    g = np.array([0.0, 0.0, 0.2, 0.2, 0.18, 0.18, 0.2, 0.0])
    u = np.array([1.0, 1.0, 1.0, 1.0, 0.82, 0.82, 1.0, 1.0])
    df = _sim_4pl(a, d, g, u, n=4000, seed=404)
    item_models: dict[str, ItemModel] = {
        "q01": TwoPL(),
        "q02": TwoPL(),
        "q03": ThreePL(),
        "q04": ThreePL(),
        "q05": FourPL(),
        "q06": FourPL(),
        "q07": ThreePL(),
        "q08": TwoPL(),
    }
    spec = Model.from_items(item_models)
    with warnings.catch_warnings():
        warnings.simplefilter("error", HeywoodWarning)
        res = fit(spec, df, method=BAEM(tol=1e-6, max_iter=5000), se="oakes")
    assert res.converged
    items = res.items.set_index("item")
    assert list(items["model"]) == ["2PL", "2PL", "3PL", "3PL", "4PL", "4PL", "3PL", "2PL"]
    assert items.loc["q01", ["g", "u"]].isna().all()
    assert float(items["u"].loc["q03"]) == 1.0 and np.isnan(float(items["u_se"].loc["q03"]))
    assert np.isfinite(items.loc["q05", ["a", "d", "g", "u"]].to_numpy(dtype=float)).all()
    scores = res.score(df.iloc[:10])
    assert len(scores) == 10 and scores["theta"].notna().all()


def test_asymptotes_do_not_identify_latent_scale_or_location() -> None:
    a = np.array([1.3, 1.4, 1.2, 1.5, 1.1])
    d = np.array([-0.5, 0.2, -0.3, 0.4, 0.0])
    g = np.full(5, 0.2)
    df = _sim_4pl(a, d, g, np.ones(5), n=400, seed=505)
    # Fixing g (and u, fixed by default) gives neither a fixed slope nor a fixed intercept.
    spec_var = (
        Model.uniform(ThreePL(), items=df.columns)
        .fix(item="q01", g=0.2)
        .with_latent(Normal(free_var=True))
    )
    with pytest.raises(IdentificationError, match="variance"):
        fit(spec_var, df, se=None)

    spec_mean = (
        Model.uniform(FourPL(), items=df.columns)
        .fix(item="q01", g=0.2, u=0.9)
        .with_latent(Normal(free_mean=True))
    )
    with pytest.raises(IdentificationError, match="mean"):
        fit(spec_mean, df, se=None)


def test_fix_from_3pl_and_4pl() -> None:
    bank_3pl = pd.DataFrame({"item": ["q01"], "a": [1.5], "b": [0.4], "g": [0.2]})
    spec3 = Model.uniform(ThreePL(), items=["q01", "q02"]).fix_from(
        bank_3pl, parameterization="irt"
    )
    fixed3 = {(i, p): v for i, p, v in spec3.fixed}
    assert fixed3 == {("q01", "a"): 1.5, ("q01", "d"): pytest.approx(-0.6), ("q01", "g"): 0.2}

    bank_4pl = pd.DataFrame({"item": ["q01"], "a": [1.5], "b": [0.4], "g": [0.15], "u": [0.9]})
    spec4 = Model.uniform(FourPL(), items=["q01", "q02"]).fix_from(bank_4pl, parameterization="irt")
    fixed4 = {(i, p): v for i, p, v in spec4.fixed}
    assert fixed4 == {
        ("q01", "a"): 1.5,
        ("q01", "d"): pytest.approx(-0.6),
        ("q01", "g"): 0.15,
        ("q01", "u"): 0.9,
    }

    # Missing u for a 4PL item raises SpecificationError.
    with pytest.raises(SpecificationError, match="no column 'u'"):
        Model.uniform(FourPL(), items=["q01", "q02"]).fix_from(bank_3pl, parameterization="irt")


def test_fix_from_rejects_several_rows_per_item() -> None:
    # Issue #145: the last row won silently, e.g. with the items table of a multiple-group fit.
    bank = pd.DataFrame(
        {
            "group": ["g1", "g1", "g2", "g2"],
            "item": ["q01", "q02", "q01", "q02"],
            "a": [1.0, 1.2, 1.5, 0.7],
            "d": [0.0, 0.3, 0.1, -0.2],
        }
    )
    spec = Model.uniform(TwoPL(), items=["q01", "q02", "q03"])
    with pytest.raises(SpecificationError, match=r"several rows for \['q01', 'q02'\]"):
        spec.fix_from(bank)
    # Duplicates of items that are not fixed do not matter.
    with pytest.raises(SpecificationError, match=r"several rows for \['q02'\]"):
        spec.fix_from(bank.iloc[1:])
    one = spec.fix_from(bank.iloc[[1, 0, 3]], items=["q01"])
    assert {(i, p): v for i, p, v in one.fixed} == {("q01", "a"): 1.0, ("q01", "d"): 0.0}
    per_group = spec.fix_from(bank[bank["group"] == "g2"])
    assert dict(((i, p), v) for i, p, v in per_group.fixed)[("q02", "a")] == 0.7


def test_fix_from_mixed_bank_nan_and_missing_slope() -> None:
    # Mixed Rasch/3PL/4PL bank: NaN cells for omitted default-fixed parameters (Rasch a, 3PL u)
    # are treated as not given.
    spec = Model.from_items({"q01": ThreePL(), "q02": FourPL(), "q03": Rasch()})
    bank = pd.DataFrame(
        [
            {"item": "q01", "a": 1.5, "d": -0.5, "g": 0.2},
            {"item": "q02", "a": 1.4, "d": -0.3, "g": 0.15, "u": 0.9},
            {"item": "q03", "d": 0.4},
        ]
    )
    fixed = {(i, p): v for i, p, v in spec.fix_from(bank).fixed}
    assert fixed == {
        ("q01", "a"): 1.5,
        ("q01", "d"): -0.5,
        ("q01", "g"): 0.2,
        ("q02", "a"): 1.4,
        ("q02", "d"): -0.3,
        ("q02", "g"): 0.15,
        ("q02", "u"): 0.9,
        ("q03", "d"): 0.4,
    }

    # NaN in a required parameter (4PL u, or any non-Rasch slope a) raises SpecificationError.
    bank_nan_u = pd.DataFrame(
        [
            {"item": "q01", "a": 1.5, "d": -0.5, "g": 0.2, "u": np.nan},
            {"item": "q02", "a": 1.4, "d": -0.3, "g": 0.15, "u": np.nan},
        ]
    )
    with pytest.raises(SpecificationError, match=r"'q02'.*'u'"):
        Model.from_items({"q01": ThreePL(), "q02": FourPL()}).fix_from(bank_nan_u)

    bank_no_a = pd.DataFrame({"item": ["q01"], "d": [-0.5], "g": [0.2], "u": [0.9]})
    for item_model in (TwoPL(), ThreePL(), FourPL()):
        with pytest.raises(SpecificationError, match=r"'q01'.*'a'"):
            Model.uniform(item_model, items=["q01", "q02"]).fix_from(bank_no_a)

    bank_nan_a = pd.DataFrame(
        [
            {"item": "q01", "d": -0.5, "g": 0.2},
            {"item": "q02", "a": 1.2, "d": 0.3, "g": 0.1},
        ]
    )
    with pytest.raises(SpecificationError, match=r"'q01'.*'a'"):
        Model.uniform(ThreePL(), items=["q01", "q02"]).fix_from(bank_nan_a)

    # Rasch with omitted 'a' in IRT parameterization still converts b -> -b using slope 1.
    bank_rasch_irt = pd.DataFrame({"item": ["q01"], "b": [0.4]})
    fixed_rasch = {
        (i, p): v
        for i, p, v in Model.uniform(Rasch(), items=["q01", "q02"])
        .fix_from(bank_rasch_irt, parameterization="irt")
        .fixed
    }
    assert fixed_rasch == {("q01", "d"): pytest.approx(-0.4)}


def test_asymptote_table_validation() -> None:
    a = np.array([1.2, 1.3, 1.1, 1.4])
    d = np.array([-0.2, 0.1, -0.4, 0.3])
    df = _sim_4pl(a, d, np.full(4, 0.15), np.full(4, 0.9), n=300, seed=606)
    spec = Model.uniform(FourPL(), items=df.columns)
    with pytest.raises(SpecificationError, match="outside"):
        fit(spec.fix(item="q01", g=-0.1), df, se=None)
    with pytest.raises(SpecificationError, match="outside"):
        fit(spec.fix(item="q01", u=1.1), df, se=None)
    with pytest.raises(SpecificationError, match="Lower asymptote"):
        fit(spec.fix(item="q01", g=0.8, u=0.5), df, se=None)


def test_edited_table_bounds_validated_against_model_domain() -> None:
    a = np.array([1.2, 1.3, 1.1, 1.4])
    d = np.array([-0.2, 0.1, -0.4, 0.3])
    df = _sim_4pl(a, d, np.full(4, 0.15), np.full(4, 0.9), n=300, seed=606)
    spec = Model.uniform(FourPL(), items=df.columns)
    base = spec.parameter_table(df)

    # Widening g lower below 0 (even with value updated inside the widened range) is rejected.
    t_g_lo = base.copy()
    row_g = (t_g_lo["item"] == "q01") & (t_g_lo["param"] == "g")
    t_g_lo.loc[row_g, ["lower", "value"]] = [-0.5, -0.05]
    with pytest.raises(SpecificationError, match=r"q01.*g"):
        fit(spec, df, table=t_g_lo, se=None)

    # Widening g upper above 1 is rejected.
    t_g_hi = base.copy()
    t_g_hi.loc[row_g, "upper"] = 1.2
    with pytest.raises(SpecificationError, match=r"q01.*g"):
        fit(spec, df, table=t_g_hi, se=None)

    # Widening u lower below 0 or upper above 1 is rejected.
    row_u = (base["item"] == "q02") & (base["param"] == "u")
    t_u_lo = base.copy()
    t_u_lo.loc[row_u, "lower"] = -0.1
    with pytest.raises(SpecificationError, match=r"q02.*u"):
        fit(spec, df, table=t_u_lo, se=None)

    t_u_hi = base.copy()
    t_u_hi.loc[row_u, "upper"] = 1.1
    with pytest.raises(SpecificationError, match=r"q02.*u"):
        fit(spec, df, table=t_u_hi, se=None)

    # Widening latent variance lower below 0 is rejected.
    t_var = base.copy()
    row_var = (t_var["item"] == "__latent__") & (t_var["param"] == "var")
    t_var.loc[row_var, "lower"] = -0.5
    with pytest.raises(SpecificationError, match=r"__latent__.*var"):
        fit(spec, df, table=t_var, se=None)

    # Tightening bounds inside the model domain is allowed.
    t_tight = base.copy()
    t_tight.loc[row_g, ["lower", "upper", "value"]] = [0.05, 0.5, 0.2]
    t_tight.loc[row_u, ["lower", "upper", "value"]] = [0.5, 0.99, 0.9]
    t_tight.loc[row_var, ["lower", "upper"]] = [0.1, 10.0]
    from flexpyirt._compile import prepare

    prepared = prepare(spec, df, group=None, item_names=None, table=t_tight)
    assert float(prepared.table.loc[row_g, "lower"].iloc[0]) == 0.05
    assert float(prepared.table.loc[row_u, "upper"].iloc[0]) == 0.99
    assert float(prepared.table.loc[row_var, "lower"].iloc[0]) == 0.1


@pytest.mark.slow
def test_fixed_asymptote_adjusts_free_asymptote_start_value() -> None:
    a = np.array([1.4, 1.5, 1.3, 1.6, 1.4, 1.5])
    d = np.array([0.0, -0.3, -0.5, -0.2, -0.4, -0.1])
    df = _sim_4pl(a, d, np.full(6, 0.2), np.full(6, 0.9), n=500, seed=707)

    # Fixing u below the default g start (~0.2) adjusts g's start strictly below u.
    spec_3pl_u = Model.uniform(ThreePL(), items=df.columns).fix(item="q01", u=0.15)
    t_3pl = spec_3pl_u.parameter_table(df).set_index(["item", "param"])
    g0 = float(t_3pl["value"].loc[("q01", "g")])
    assert 0.0 < g0 < 0.15
    assert bool(t_3pl["free"].loc[("q01", "g")])
    assert float(t_3pl["value"].loc[("q01", "u")]) == 0.15
    assert not bool(t_3pl["free"].loc[("q01", "u")])

    # Same for FourPL when u is fixed low.
    spec_4pl_u = Model.uniform(FourPL(), items=df.columns).fix(item="q01", u=0.15)
    t_4pl_u = spec_4pl_u.parameter_table(df).set_index(["item", "param"])
    assert 0.0 < float(t_4pl_u["value"].loc[("q01", "g")]) < 0.15

    # Fixing g above the default u start (~0.95) adjusts u's start strictly above g.
    spec_4pl_g = Model.uniform(FourPL(), items=df.columns).fix(item="q01", g=0.96)
    t_4pl_g = spec_4pl_g.parameter_table(df).set_index(["item", "param"])
    u0 = float(t_4pl_g["value"].loc[("q01", "u")])
    assert 0.96 < u0 < 1.0
    assert bool(t_4pl_g["free"].loc[("q01", "u")])
    assert float(t_4pl_g["value"].loc[("q01", "g")]) == 0.96
    assert not bool(t_4pl_g["free"].loc[("q01", "g")])

    # Fitting with adjusted starting values proceeds without SpecificationError.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_u = fit(spec_3pl_u, df, method=BAEM(tol=1e-4, max_iter=10), se=None)
        res_g = fit(spec_4pl_g, df, method=BAEM(tol=1e-4, max_iter=10), se=None)
    assert np.isfinite(res_u.loglik)
    assert float(res_u.items.set_index("item")["g"].loc["q01"]) < 0.15
    assert np.isfinite(res_g.loglik)
    assert float(res_g.items.set_index("item")["u"].loc["q01"]) > 0.96

    # Boundary-infeasible single-asymptote fixes are still rejected.
    with pytest.raises(SpecificationError, match="Lower asymptote"):
        Model.uniform(ThreePL(), items=df.columns).fix(item="q01", u=0.0).parameter_table(df)
    with pytest.raises(SpecificationError, match="Lower asymptote"):
        Model.uniform(FourPL(), items=df.columns).fix(item="q01", g=1.0).parameter_table(df)
    with pytest.raises(SpecificationError, match=r"outside.*'u'"):
        Model.uniform(ThreePL(), items=df.columns).fix(item="q01", u=-0.1).parameter_table(df)
