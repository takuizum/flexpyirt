"""Unit and property tests for NominalKernel and Nominal item model (DESIGN.md §16.6)."""

from __future__ import annotations

import warnings
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from flexpyirt import (
    BAEM,
    GPCM,
    EmpiricalHistogram,
    FitResult,
    LogNormal,
    Model,
    Nominal,
    Normal,
    NormalPrior,
    Quadrature,
    TwoPL,
    fit,
)
from flexpyirt.errors import HeywoodWarning, IdentificationError, SpecificationError
from flexpyirt.fit import _warn_heywood
from flexpyirt.items import DichotomousKernel, GPCMKernel, ItemModel, NominalKernel


def _sim_nominal(
    a: np.ndarray,
    ak: np.ndarray,
    d: np.ndarray,
    n: int,
    seed: int,
    *,
    mu: float = 0.0,
    var: float = 1.0,
) -> pd.DataFrame:
    """Simulate responses from the nominal model with full ak (J, K) and d (J, K)."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(mu, np.sqrt(var), size=n)
    n_items = a.shape[0]
    y = np.empty((n, n_items), dtype=float)
    for j in range(n_items):
        logits = np.outer(theta * a[j], ak[j]) + d[j][None, :]
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        cdf = np.cumsum(probs, axis=1)
        u = rng.random(n)[:, None]
        y[:, j] = (u >= cdf).sum(axis=1)
    return pd.DataFrame(y, columns=[f"q{j:02d}" for j in range(1, n_items + 1)])


# ---- NominalKernel -------------------------------------------------------------------------


@settings(deadline=None)
@given(
    a=st.floats(-3.0, 3.0),
    ak1=st.floats(-3.0, 5.0),
    ak2=st.floats(-3.0, 5.0),
    d1=st.floats(-4.0, 4.0),
    d2=st.floats(-4.0, 4.0),
    d3=st.floats(-4.0, 4.0),
)
def test_nominal_probabilities_sum_to_one(
    a: float, ak1: float, ak2: float, d1: float, d2: float, d3: float
) -> None:
    """Probabilities sum to 1, match the nominal formula, and have finite derivatives."""
    theta = jnp.linspace(-6.0, 6.0, 25)
    params = jnp.array([a, ak1, ak2, d1, d2, d3])
    kernel = NominalKernel(4)
    lp = kernel.log_prob(theta, params)
    p = np.exp(np.asarray(lp))
    np.testing.assert_allclose(p.sum(axis=1), 1.0, rtol=1e-12)

    th = np.asarray(theta)
    raw = np.column_stack(
        [np.zeros_like(th), a * ak1 * th + d1, a * ak2 * th + d2, a * 3.0 * th + d3]
    )
    raw -= raw.max(axis=1, keepdims=True)
    expected = np.exp(raw) / np.exp(raw).sum(axis=1, keepdims=True)
    np.testing.assert_allclose(p, expected, rtol=1e-12, atol=1e-15)

    def obj(x: jax.Array) -> jax.Array:
        return kernel.log_prob(theta, x).sum()

    grad = np.asarray(jax.grad(obj)(params))
    hess = np.asarray(jax.hessian(obj)(params))
    assert np.isfinite(grad).all()
    assert np.isfinite(hess).all()


def test_nominal_multidimensional_kernel_and_finite_diff_gradients() -> None:
    """1D, 2D, and 3D NominalKernel gradients match central finite differences."""
    rng = np.random.default_rng(17)
    for n_factors in (1, 2, 3):
        kernel = NominalKernel(4).with_factors(n_factors)
        assert kernel.n_factors == n_factors
        assert kernel.irt_names == ()
        assert kernel.irt_dep_indices == ()
        assert kernel.irt_thresholds(jnp.zeros(len(kernel.param_names))).shape == (0,)
        theta = jnp.asarray(rng.normal(0.0, 1.0, size=(15, n_factors)))
        weights = jnp.asarray(rng.uniform(0.5, 2.0, size=(15, 4)))
        params = jnp.asarray(rng.normal(0.0, 0.8, size=(len(kernel.param_names),)))

        def q_fn(
            p: jax.Array,
            k: NominalKernel = kernel,
            th: jax.Array = theta,
            w: jax.Array = weights,
        ) -> jax.Array:
            return jnp.sum(w * k.log_prob(th, p))

        grad_ad = np.asarray(jax.grad(q_fn)(params))
        eps = 1e-6
        grad_fd = np.empty_like(grad_ad)
        for i in range(len(params)):
            e = jnp.zeros_like(params).at[i].set(eps)
            grad_fd[i] = float((q_fn(params + e) - q_fn(params - e)) / (2.0 * eps))
        np.testing.assert_allclose(grad_ad, grad_fd, rtol=1e-6, atol=1e-7)


def test_nominal_n_cat_and_n_factors_validation() -> None:
    for bad in (1, 0, -3):
        with pytest.raises(SpecificationError, match="at least 2"):
            NominalKernel(bad)
        with pytest.raises(SpecificationError, match="at least 2"):
            Nominal(bad)
    for bad_f in (0, 4):
        with pytest.raises(SpecificationError, match="1 to 3 factors"):
            NominalKernel(3, n_factors=bad_f)


def test_nominal_reduces_to_gpcm_and_twopl() -> None:
    """Fixing ak_k = k reduces Nominal(K) to GPCM(K) in kernel, estimates, SEs, and EAP."""
    theta = jnp.linspace(-6.0, 6.0, 61)
    for a, d1, d2, d3 in [(1.2, 0.8, 0.2, -0.7), (0.6, -0.5, 1.1, 0.4)]:
        lp_gpcm = GPCMKernel(4).log_prob(theta, jnp.array([a, d1, d2, d3]))
        lp_nom = NominalKernel(4).log_prob(theta, jnp.array([a, 1.0, 2.0, d1, d2, d3]))
        np.testing.assert_allclose(np.asarray(lp_nom), np.asarray(lp_gpcm), rtol=1e-13, atol=1e-15)

    for a, d in [(1.3, -0.4), (-0.7, 0.9)]:
        lp_2pl = DichotomousKernel().log_prob(theta, jnp.array([a, d]))
        lp_nom2 = NominalKernel(2).log_prob(theta, jnp.array([a, d]))
        np.testing.assert_allclose(np.asarray(lp_nom2), np.asarray(lp_2pl), rtol=1e-13, atol=1e-15)

    a_arr = np.array([1.1, 1.3, 0.9, 1.2, 1.0])
    ak_arr = np.tile(np.array([0.0, 1.0, 2.0, 3.0]), (5, 1))
    d_arr = np.array(
        [
            [0.0, 0.8, 0.4, -0.5],
            [0.0, 0.3, -0.2, -0.9],
            [0.0, 1.0, 0.7, 0.1],
            [0.0, -0.2, -0.1, -0.8],
            [0.0, 0.5, 0.6, -0.3],
        ]
    )
    df = _sim_nominal(a_arr, ak_arr, d_arr, n=700, seed=55)
    method = BAEM(tol=1e-9, max_iter=5000)

    spec_gpcm = Model.uniform(GPCM(4), items=df.columns)
    spec_nom_fixed = Model.uniform(Nominal(4), items=df.columns)
    for itm in df.columns:
        spec_nom_fixed = spec_nom_fixed.fix(item=itm, ak1=1.0, ak2=2.0)

    res_gpcm = fit(spec_gpcm, df, method=method, se="oakes")
    res_nom = fit(spec_nom_fixed, df, method=method, se="oakes")

    assert res_nom.converged
    assert res_nom.n_params == res_gpcm.n_params
    assert res_nom.loglik == pytest.approx(res_gpcm.loglik, rel=1e-12)
    for col in ("a", "d1", "d2", "d3", "a_se", "d1_se", "d2_se", "d3_se"):
        np.testing.assert_allclose(res_nom.items[col], res_gpcm.items[col], rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(
        res_nom.score(df)["theta"], res_gpcm.score(df)["theta"], rtol=0, atol=1e-8
    )

    # S-X2 item_fit also matches GPCM when ak_k = k are fixed.
    fit_gpcm = res_gpcm.item_fit(df)
    fit_nom = res_nom.item_fit(df)
    np.testing.assert_array_equal(fit_nom["df"], fit_gpcm["df"])
    np.testing.assert_allclose(fit_nom["s_x2"], fit_gpcm["s_x2"], rtol=1e-7, atol=1e-8)


def test_nominal_two_categories_equals_twopl_fit() -> None:
    """Nominal(2) has no free ak parameters and matches TwoPL."""
    rng = np.random.default_rng(88)
    n = 500
    theta = rng.normal(0.0, 1.0, size=n)
    a = np.array([1.2, 0.9, 1.4, 1.0])
    d = np.array([-0.3, 0.4, -0.1, 0.2])
    p = 1.0 / (1.0 + np.exp(-(np.outer(theta, a) + d)))
    df = pd.DataFrame((rng.random(p.shape) < p).astype(float), columns=["q1", "q2", "q3", "q4"])
    method = BAEM(tol=1e-9, max_iter=3000)

    res_2pl = fit(Model.uniform(TwoPL(), items=df.columns), df, method=method, se="oakes")
    res_nom2 = fit(Model.uniform(Nominal(2), items=df.columns), df, method=method, se="oakes")
    assert res_nom2.converged
    assert res_nom2.loglik == pytest.approx(res_2pl.loglik, rel=1e-12)
    np.testing.assert_allclose(res_nom2.items["a"], res_2pl.items["a"], atol=1e-8)
    np.testing.assert_allclose(res_nom2.items["d1"], res_2pl.items["d"], atol=1e-8)


def test_nominal_starting_values() -> None:
    rng = np.random.default_rng(19)
    item = rng.integers(0, 4, size=200).astype(float)
    rest = rng.integers(0, 12, size=200).astype(float)
    sv_nom = Nominal(4).start_values(item, rest)
    sv_gpcm = GPCM(4).start_values(item, rest)
    assert sv_nom.shape == (6,)
    assert sv_nom[0] == pytest.approx(sv_gpcm[0])
    np.testing.assert_allclose(sv_nom[1:3], [1.0, 2.0])
    np.testing.assert_allclose(sv_nom[3:], sv_gpcm[1:])


# ---- Identification, estimability, and fix_from --------------------------------------------


def test_nominal_identification_and_unobserved_category() -> None:
    a_arr = np.ones(4)
    ak_arr = np.tile(np.array([0.0, 1.0, 2.0, 3.0]), (4, 1))
    d_arr = np.tile(np.array([0.0, 0.5, 0.2, -0.4]), (4, 1))
    df = _sim_nominal(a_arr, ak_arr, d_arr, n=400, seed=301)

    with pytest.raises(IdentificationError, match="variance"):
        fit(Model.uniform(Nominal(4), items=df.columns).with_latent(Normal(free_var=True)), df)
    with pytest.raises(IdentificationError, match="mean"):
        fit(
            Model.uniform(Nominal(4), items=df.columns)
            .fix(item="q01", a=1.0)
            .with_latent(Normal.free()),
            df,
        )
    # Fixing ak1, ak2 alone reduces Nominal to GPCM and must NOT identify a free variance or mean.
    with pytest.raises(IdentificationError, match="variance"):
        fit(
            Model.uniform(Nominal(4), items=df.columns)
            .fix(item="q01", ak1=1.0, ak2=2.0)
            .with_latent(Normal(free_var=True)),
            df,
        )
    with pytest.raises(IdentificationError, match="mean"):
        fit(
            Model.uniform(Nominal(4), items=df.columns)
            .fix(item="q01", a=1.0, ak1=1.0, ak2=2.0)
            .with_latent(Normal.free()),
            df,
        )

    # Unobserved category rejected by _check_estimable.
    df_bad = df.copy()
    df_bad.loc[df_bad["q01"] == 2, "q01"] = 1
    with pytest.raises(SpecificationError, match=r"categories \[2\]"):
        fit(Model.uniform(Nominal(4), items=df.columns), df_bad, se=None)


def test_nominal_degenerate_item_warns() -> None:
    """Near-zero slope (mirt's NR saddle) or diverging ak / d raise a HeywoodWarning."""
    a_arr = np.ones(4)
    ak_arr = np.tile(np.array([0.0, 1.0, 2.0, 3.0]), (4, 1))
    d_arr = np.tile(np.array([0.0, 0.5, 0.2, -0.4]), (4, 1))
    df = _sim_nominal(a_arr, ak_arr, d_arr, n=400, seed=302)
    spec = Model.uniform(Nominal(4), items=df.columns)
    table = spec.parameter_table(df)
    good = table.assign(estimate=table["value"].astype(float))
    is_a = good["param"] == "a"

    with warnings.catch_warnings():
        warnings.simplefilter("error", HeywoodWarning)
        _warn_heywood(good, spec)

    saddle = good.copy()
    saddle.loc[(saddle["item"] == "q01") & is_a, "estimate"] = -3.6e-31
    with pytest.warns(HeywoodWarning, match=r"Degenerate nominal items \['q01'\]"):
        _warn_heywood(saddle, spec)

    diverging = good.copy()
    diverging.loc[(diverging["item"] == "q02") & (diverging["param"] == "d3"), "estimate"] = -80.0
    with pytest.warns(HeywoodWarning, match=r"Degenerate nominal items \['q02'\]"):
        _warn_heywood(diverging, spec)

    # A user-fixed zero slope is a deliberate choice, not a degenerate estimate.
    fixed = saddle.copy()
    fixed.loc[(fixed["item"] == "q01") & is_a, "free"] = False
    with warnings.catch_warnings():
        warnings.simplefilter("error", HeywoodWarning)
        _warn_heywood(fixed, spec)


def test_nominal_fix_from() -> None:
    bank = pd.DataFrame(
        [
            {"item": "q01", "a": 1.2, "ak1": 0.8, "ak2": 2.1, "d1": 0.4, "d2": -0.2, "d3": -0.5},
            {"item": "q02", "a": 0.9, "ak1": 1.3, "ak2": 1.9, "d1": 0.1, "d2": 0.3, "d3": -0.4},
        ]
    )
    spec = Model.uniform(Nominal(4), items=["q01", "q02", "q03"]).fix_from(bank)
    fixed = {(i, p): v for i, p, v in spec.fixed}
    assert fixed[("q01", "a")] == 1.2
    assert fixed[("q01", "ak1")] == 0.8
    assert fixed[("q01", "ak2")] == 2.1
    assert fixed[("q01", "d3")] == -0.5

    with pytest.raises(SpecificationError, match="not supported for Nominal"):
        Model.uniform(Nominal(4), items=["q01", "q02"]).fix_from(bank, parameterization="irt")


# ---- Downstream machinery: JSON, item_fit, scoring, MG/DIF, priors, equal_items, 2D --------


def test_nominal_json_item_fit_and_scoring() -> None:
    """Fit a nominal model with an unordered item, test JSON round-trip, item_fit, and scoring."""
    a_arr = np.array([1.1, 1.0, 1.2, 0.9, 1.1])
    ak_arr = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 0.8, 2.1, 3.0],
            [0.0, 1.2, 1.9, 3.0],
            [0.0, 2.2, 0.7, 3.0],  # swapped categories 1 and 2
            [0.0, -0.5, 1.8, 3.0],  # category 1 has lower slope than category 0
        ]
    )
    d_arr = np.array(
        [
            [0.0, 0.4, 0.2, -0.3],
            [0.0, 0.2, 0.5, -0.1],
            [0.0, -0.1, 0.3, -0.4],
            [0.0, -0.2, 0.1, -0.2],
            [0.0, 0.1, -0.2, -0.5],
        ]
    )
    df = _sim_nominal(a_arr, ak_arr, d_arr, n=1000, seed=412)
    spec = Model.uniform(Nominal(4), items=df.columns)
    with warnings.catch_warnings():
        warnings.simplefilter("error", HeywoodWarning)
        res = fit(spec, df, method=BAEM(tol=1e-7, max_iter=5000, squarem=True), se="oakes")

    assert res.converged
    assert "Nominal(4)" in res.summary()
    assert "Nominal(4)" in res._repr_html_()
    assert "b1" not in res.items.columns

    # JSON round-trip
    loaded = FitResult.from_json(res.to_json())
    assert loaded.summary() == res.summary()
    pd.testing.assert_frame_equal(loaded.items, res.items)
    assert loaded.vcov is not None and res.vcov is not None
    np.testing.assert_allclose(loaded.vcov, res.vcov)

    # S-X2 item_fit
    s_x2 = res.item_fit(df)
    assert list(s_x2["item"]) == list(df.columns)
    assert (s_x2["df"] > 0).all()
    assert np.isfinite(s_x2["s_x2"]).all()
    assert ((s_x2["p"] >= 0.0) & (s_x2["p"] <= 1.0)).all()

    # Scoring: EAP, MAP, ML, WLE, PV
    # Check extreme pattern handling when q05 has ak1 < 0:
    # Lowest-theta pattern is (0, 0, 0, 0, 1); all-0 pattern (0, 0, 0, 0, 0) has an interior ML!
    est_q05_ak1 = float(res.items.set_index("item")["ak1"].loc["q05"])
    assert est_q05_ak1 < 0.0
    pats = pd.DataFrame(
        [
            [0.0, 0.0, 0.0, 0.0, 1.0],  # true all-lowest-slope pattern -> -inf
            [3.0, 3.0, 3.0, 3.0, 3.0],  # all-highest-slope pattern -> +inf
            [0.0, 0.0, 0.0, 0.0, 0.0],  # category 0 on q05 is NOT lowest slope -> finite ML
            [1.0, 2.0, 1.0, 2.0, 2.0],  # interior pattern
        ],
        columns=df.columns,
    )
    ml = res.score(pats, method="ML")
    ml_theta = ml["theta"].to_numpy(dtype=float)
    ml_se = ml["se"].to_numpy(dtype=float)
    assert ml_theta[0] == -np.inf and np.isnan(ml_se[0])
    assert ml_theta[1] == np.inf and np.isnan(ml_se[1])
    assert np.isfinite(ml_theta[2]) and np.isfinite(ml_se[2])
    assert np.isfinite(ml_theta[3]) and np.isfinite(ml_se[3])

    methods: tuple[Literal["EAP", "MAP", "WLE"], ...] = ("EAP", "MAP", "WLE")
    for m in methods:
        sc = res.score(pats, method=m)
        assert np.isfinite(sc["theta"]).all() and np.isfinite(sc["se"]).all()
    pvs = res.score(pats, method="PV", key=jax.random.key(3), n_draws=4)
    assert pvs.shape == (4, 4) and np.isfinite(pvs.to_numpy()).all()


def test_nominal_priors_equal_items_multigroup_and_2d() -> None:
    """Nominal works with priors, equal_across_items, multiple groups + DIF, EH, and 2D MIRT."""
    a_arr = np.array([1.1, 1.0, 1.2, 0.9])
    ak_arr = np.tile(np.array([0.0, 1.0, 2.0]), (4, 1))
    d_arr = np.array([[0.0, 0.3, -0.2], [0.0, 0.1, -0.4], [0.0, -0.2, -0.1], [0.0, 0.4, 0.1]])
    df1 = _sim_nominal(a_arr, ak_arr, d_arr, n=500, seed=501)

    # 1. Priors + equal_across_items
    spec_eq = (
        Model.uniform(Nominal(3), items=df1.columns)
        .equal_across_items(items=["q01", "q02"], params=["a", "ak1"])
        .prior(param="a", dist=LogNormal(0.0, 0.5))
        .prior(param="ak1", dist=NormalPrior(1.0, 1.0))
    )
    res_eq = fit(spec_eq, df1, method=BAEM(tol=1e-6), se="oakes")
    assert res_eq.converged
    items_eq = res_eq.items.set_index("item")
    assert float(items_eq["a"].loc["q01"]) == pytest.approx(float(items_eq["a"].loc["q02"]))
    assert float(items_eq["ak1"].loc["q01"]) == pytest.approx(float(items_eq["ak1"].loc["q02"]))

    # 2. Multiple groups + DIF
    df2 = _sim_nominal(a_arr, ak_arr, d_arr, n=500, seed=502, mu=0.3, var=1.1)
    df_mg = pd.concat([df1, df2], ignore_index=True)
    grp = np.repeat(["ref", "focal"], 500)
    spec_mg = (
        Model.uniform(Nominal(3), items=df1.columns)
        .with_groups(ref=Normal.standard(), focal=Normal.free())
        .equal_across_groups(items=["q01", "q02"])
    )
    res_mg = fit(
        spec_mg,
        df_mg,
        group=grp,
        method=BAEM(tol=1e-5, squarem=True),
        quadrature=Quadrature.equispaced(41, lim=(-6, 6)),
        se="oakes",
    )
    assert res_mg.converged
    dif_df = res_mg.dif(df_mg, group=grp, items=["q03", "q04"], params=["ak1", "d1", "d2"])
    assert list(dif_df["df"]) == [3, 3]
    assert np.isfinite(dif_df["x2"]).all()

    # 3. EmpiricalHistogram
    res_eh = fit(
        Model.uniform(Nominal(3), items=df1.columns).with_latent(EmpiricalHistogram()),
        df1,
        method=BAEM(tol=1e-4, squarem=True),
        quadrature=Quadrature.equispaced(21, lim=(-5, 5)),
        se="oakes",
    )
    assert res_eh.converged and res_eh.vcov is not None and np.isfinite(res_eh.vcov).all()

    # 4. 2D confirmatory Nominal + 2PL mixed model + JSON round-trip
    rng = np.random.default_rng(503)
    p_2pl = 1.0 / (1.0 + np.exp(-rng.normal(0.0, 1.0, size=(500, 2))))
    df_2pl = pd.DataFrame((rng.random((500, 2)) < p_2pl).astype(float), columns=["b01", "b02"])
    df_2d = pd.concat([df1, df_2pl], axis=1)
    models_2d: dict[str, ItemModel] = {
        "q01": Nominal(3),
        "q02": Nominal(3),
        "q03": Nominal(3),
        "q04": Nominal(3),
        "b01": TwoPL(),
        "b02": TwoPL(),
    }
    spec_2d = Model.from_items(models_2d).with_factors(
        {"F1": ["q01", "q02", "b01"], "F2": ["q03", "q04", "b02"]}
    )
    res_2d = fit(
        spec_2d,
        df_2d,
        method=BAEM(tol=1e-5, squarem=True),
        quadrature=Quadrature.equispaced(15, lim=(-5, 5)),
        se="oakes",
    )
    assert res_2d.converged
    loaded_2d = FitResult.from_json(res_2d.to_json())
    pd.testing.assert_frame_equal(loaded_2d.items, res_2d.items)
    sc_2d = res_2d.score(df_2d.iloc[:5], method="MAP")
    assert list(sc_2d.columns) == ["theta_F1", "theta_F2", "se_F1", "se_F2"]
