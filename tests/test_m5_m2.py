"""Unit tests for M2 / M2* limited-information goodness of fit (DESIGN.md §16.3)."""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from flexpyirt import (
    BAEM,
    GPCM,
    GRM,
    EmpiricalHistogram,
    Model,
    Normal,
    NormalPrior,
    Quadrature,
    ThreePL,
    TwoPL,
    _core,
    _m2,
    fit,
)
from flexpyirt._compile import coerce_groups, coerce_responses, compile_problem
from flexpyirt.errors import FlexpyirtWarning, HeywoodWarning, SpecificationError


def _sim_2pl(n_obs: int = 400, n_items: int = 6, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.lognormal(0.1, 0.25, n_items)
    d = rng.normal(0.0, 0.8, n_items)
    theta = rng.normal(0.0, 1.0, n_obs)
    p = 1.0 / (1.0 + np.exp(-(np.outer(theta, a) + d)))
    return pd.DataFrame(
        (rng.random(p.shape) < p).astype(int),
        columns=[f"q{j + 1:02d}" for j in range(n_items)],
    )


def _sim_grm(n_obs: int = 400, n_items: int = 6, n_cat: int = 3, seed: int = 43) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.lognormal(0.1, 0.25, n_items)
    d = np.sort(rng.normal(0.0, 1.0, (n_items, n_cat - 1)), axis=1)[:, ::-1]
    theta = rng.normal(0.0, 1.0, n_obs)
    u = rng.random((n_obs, n_items))
    y = np.zeros((n_obs, n_items), dtype=int)
    for j in range(n_items):
        p_ge = 1.0 / (1.0 + np.exp(-(theta[:, None] * a[j] + d[j][None, :])))
        y[:, j] = (u[:, j : j + 1] < p_ge).sum(axis=1)
    return pd.DataFrame(y, columns=[f"g{j + 1:02d}" for j in range(n_items)])


def _brute_xi(m1: np.ndarray, m2: np.ndarray, w: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Reference set-union implementation of Xi_2 for cross-checking the vectorized _xi."""
    n_items, n_nodes = m1.shape
    ii, jj = _m2._pairs(n_items)
    items = [(j,) for j in range(n_items)] + list(zip(ii.tolist(), jj.tolist(), strict=True))
    sets = [set(a) for a in items]
    s = len(sets)
    out = np.empty((s, s), dtype=float)
    for b, sb in enumerate(sets):
        for a, sa in enumerate(sets):
            prod = np.ones(n_nodes, dtype=float)
            for x in sa | sb:
                prod = prod * (m2[x] if (x in sa and x in sb) else m1[x])
            out[a, b] = float(prod @ w)
    return out - np.outer(e, e)


def test_m2_rejects_unsupported_models_and_inputs() -> None:
    df = _sim_2pl(n_obs=300, n_items=6, seed=1)
    items = list(df.columns)

    # Multiple-group model.
    grp = np.repeat(["ref", "focal"], len(df) // 2)
    spec_mg = (
        Model.uniform(TwoPL(), items=items)
        .with_groups(ref=Normal.standard(), focal=Normal.free())
        .equal_across_groups(items=items[:3])
    )
    res_mg = fit(spec_mg, df, group=grp, method=BAEM(tol=1e-4, squarem=True), se=None)
    with pytest.raises(SpecificationError, match="single-group unidimensional"):
        res_mg.m2(df)

    # EmpiricalHistogram latent distribution.
    spec_eh = Model.uniform(TwoPL(), items=items).with_latent(EmpiricalHistogram())
    res_eh = fit(spec_eh, df, method=BAEM(tol=1e-3, max_iter=50, squarem=True), se=None)
    with pytest.raises(SpecificationError, match="single-group unidimensional"):
        res_eh.m2(df)

    # 2-factor multidimensional model.
    spec_2d = Model.uniform(TwoPL(), items=items).with_factors({"F1": items[:3], "F2": items[3:]})
    res_2d = fit(
        spec_2d,
        df,
        method=BAEM(tol=1e-4, squarem=True),
        quadrature=Quadrature.equispaced(11, lim=(-4, 4)),
        se=None,
    )
    with pytest.raises(SpecificationError, match="single-group unidimensional"):
        res_2d.m2(df)

    # MAP fit (with priors).
    spec_map = Model.uniform(TwoPL(), items=items).prior(param="d", dist=NormalPrior(0.0, 2.0))
    res_map = fit(spec_map, df, method=BAEM(tol=1e-4, squarem=True), se=None)
    with pytest.raises(SpecificationError, match="MLE fit without priors"):
        res_map.m2(df)

    # Standard 2PL fit: missing data, invalid ci, too few respondents.
    res = fit(Model.uniform(TwoPL(), items=items), df, method=BAEM(tol=1e-4, squarem=True), se=None)
    df_miss = df.astype(float)
    df_miss.iloc[0, 0] = np.nan
    with pytest.raises(SpecificationError, match="complete data"):
        res.m2(df_miss)

    for bad_ci in (0.0, 1.0, -0.5, 1.5, True, float("nan")):
        with pytest.raises(SpecificationError, match="ci must be"):
            res.m2(df, ci=bad_ci)

    with pytest.raises(SpecificationError, match="at least 2 respondents"):
        res.m2(df.iloc[:1])


@pytest.mark.parametrize("n_items,expected_df", [(3, 0), (2, -1)])
def test_m2_nonpositive_df_returns_nan_indices(n_items: int, expected_df: int) -> None:
    df = _sim_2pl(n_obs=400, n_items=n_items, seed=2)
    res = fit(
        Model.uniform(TwoPL(), items=list(df.columns)),
        df,
        method=BAEM(tol=1e-6, squarem=True),
        se=None,
    )
    out = res.m2(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert int(row["df"]) == expected_df
    assert np.isnan(row["m2"])
    assert np.isnan(row["p"])
    assert np.isnan(row["rmsea"])
    assert np.isnan(row["rmsea_5"])
    assert np.isnan(row["rmsea_95"])
    assert np.isnan(row["tli"])
    assert np.isnan(row["cfi"])
    assert np.isfinite(row["srmsr"]) and row["srmsr"] >= 0.0


def test_m2_invariant_to_item_order() -> None:
    df2 = _sim_2pl(n_obs=500, n_items=6, seed=3)
    perm = ["q05", "q02", "q06", "q01", "q04", "q03"]
    r1 = fit(
        Model.uniform(TwoPL(), items=list(df2.columns)),
        df2,
        method=BAEM(tol=1e-9, squarem=True),
        se=None,
    ).m2(df2)
    r2 = fit(
        Model.uniform(TwoPL(), items=perm),
        df2[perm],
        method=BAEM(tol=1e-9, squarem=True),
        se=None,
    ).m2(df2[perm])
    assert int(r1["df"].iloc[0]) == int(r2["df"].iloc[0])
    for col in ("m2", "p", "rmsea", "rmsea_5", "rmsea_95", "srmsr", "tli", "cfi"):
        np.testing.assert_allclose(r1[col].iloc[0], r2[col].iloc[0], rtol=1e-7, atol=1e-9)

    # Also check polytomous (GRM) invariance.
    dfg = _sim_grm(n_obs=500, n_items=6, n_cat=3, seed=4)
    gperm = ["g04", "g01", "g06", "g02", "g05", "g03"]
    rg1 = fit(
        Model.uniform(GRM(n_cat=3), items=list(dfg.columns)),
        dfg,
        method=BAEM(tol=1e-9, squarem=True),
        se=None,
    ).m2(dfg)
    rg2 = fit(
        Model.uniform(GRM(n_cat=3), items=gperm),
        dfg[gperm],
        method=BAEM(tol=1e-9, squarem=True),
        se=None,
    ).m2(dfg[gperm])
    assert int(rg1["df"].iloc[0]) == int(rg2["df"].iloc[0])
    for col in ("m2", "p", "rmsea", "rmsea_5", "rmsea_95", "srmsr"):
        np.testing.assert_allclose(rg1[col].iloc[0], rg2[col].iloc[0], rtol=1e-7, atol=1e-9)
    assert np.isnan(rg1["tli"].iloc[0]) and np.isnan(rg2["tli"].iloc[0])
    assert np.isnan(rg1["cfi"].iloc[0]) and np.isnan(rg2["cfi"].iloc[0])


def test_rmsea_and_rmsea_ci_against_hand_values() -> None:
    # Exact hand values for rmsea.
    assert _m2.rmsea(5.0, df=10, n=500) == 0.0
    assert _m2.rmsea(10.0, df=10, n=500) == 0.0
    np.testing.assert_allclose(_m2.rmsea(30.0, df=10, n=501), np.sqrt((3.0 - 1.0) / 500.0))
    assert np.isnan(_m2.rmsea(10.0, df=0, n=500))
    assert np.isnan(_m2.rmsea(10.0, df=-3, n=500))
    assert np.isnan(_m2.rmsea(float("nan"), df=5, n=500))

    # Noncentral chi-square inversion check for rmsea_ci.
    x2, df, n, level = 25.0, 10, 400, 0.90
    alpha = (1.0 - level) / 2.0
    lo, hi = _m2.rmsea_ci(x2, df, n, level=level)
    assert 0.0 < lo < _m2.rmsea(x2, df, n) < hi
    np.testing.assert_allclose(stats.ncx2.cdf(x2, df, n * df * lo**2), 1.0 - alpha, atol=1e-10)
    np.testing.assert_allclose(stats.ncx2.cdf(x2, df, n * df * hi**2), alpha, atol=1e-10)

    # Small X2: lower bound unbracketed -> 0, upper bound positive and exact.
    lo_s, hi_s = _m2.rmsea_ci(8.0, df=10, n=400, level=0.90)
    assert lo_s == 0.0
    assert hi_s > 0.0
    np.testing.assert_allclose(stats.ncx2.cdf(8.0, 10, 400 * 10 * hi_s**2), 0.05, atol=1e-10)

    # Very small X2 where P(chi2_df <= X2) < alpha: both bounds 0.
    lo_z, hi_z = _m2.rmsea_ci(1.0, df=10, n=400, level=0.90)
    assert lo_z == 0.0 and hi_z == 0.0

    # Non-positive df gives (NaN, NaN).
    lo_n, hi_n = _m2.rmsea_ci(10.0, df=0, n=400)
    assert np.isnan(lo_n) and np.isnan(hi_n)

    # CI column names and width ordering.
    assert _m2.ci_column_names(0.90) == ("rmsea_5", "rmsea_95")
    assert _m2.ci_column_names(0.95) == ("rmsea_2.5", "rmsea_97.5")
    lo95, hi95 = _m2.rmsea_ci(x2, df, n, level=0.95)
    assert lo95 < lo < hi < hi95


def test_xi_is_symmetric_psd_and_matches_brute_force() -> None:
    df_bin = _sim_2pl(n_obs=300, n_items=7, seed=5)
    df_poly = _sim_grm(n_obs=300, n_items=6, n_cat=4, seed=6)
    configs = [
        (Model.uniform(TwoPL(), items=list(df_bin.columns)), df_bin),
        (Model.uniform(ThreePL(), items=list(df_bin.columns)), df_bin),
        (Model.uniform(GRM(n_cat=4), items=list(df_poly.columns)), df_poly),
        (Model.uniform(GPCM(n_cat=4), items=list(df_poly.columns)), df_poly),
    ]
    for spec, df in configs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", HeywoodWarning)
            res = fit(spec, df, method=BAEM(tol=1e-3, max_iter=100, squarem=True), se=None)
        y = coerce_responses(df, res._model)
        codes, _ = coerce_groups(None, res._model, y.shape[0], require_all=False)
        compiled = compile_problem(res._model, y, codes, res.parameters, res._quadrature)
        eta = jnp.asarray(res._eta)
        m1_j, m2_j = _m2.conditional_moments(compiled.problem, eta)
        m1, m2 = np.asarray(m1_j, dtype=float), np.asarray(m2_j, dtype=float)
        logw = _core._latent_log_weights(compiled.problem, eta)[0]
        w = np.exp(np.asarray(logw, dtype=float))
        e = np.asarray(_m2.model_margins(compiled.problem, eta), dtype=float)

        xi = _m2._xi(m1, m2, w, e)
        brute = _brute_xi(m1, m2, w, e)
        np.testing.assert_allclose(xi, brute, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(xi, xi.T, rtol=0.0, atol=1e-15)
        eigvals = np.linalg.eigvalsh(xi)
        assert float(eigvals.min()) >= -1e-13


def test_m2_custom_ci_array_input_and_equality_constraints() -> None:
    df = _sim_2pl(n_obs=400, n_items=6, seed=7)
    items = list(df.columns)
    spec = Model.uniform(TwoPL(), items=items).equal_across_items(items=items, params=["a"])
    res = fit(spec, df, method=BAEM(tol=1e-7, squarem=True), se=None)

    out90 = res.m2(df, ci=0.90)
    out95 = res.m2(df.to_numpy(), item_names=items, ci=0.95)
    assert list(out95.columns) == [
        "m2",
        "df",
        "p",
        "rmsea",
        "rmsea_2.5",
        "rmsea_97.5",
        "srmsr",
        "tli",
        "cfi",
    ]
    # J=6 -> s = 6 + 15 = 21; common slope + 6 intercepts -> n_params = 7 -> df = 14.
    assert int(out90["df"].iloc[0]) == 14
    assert int(out95["df"].iloc[0]) == 14
    for col in ("m2", "p", "rmsea", "srmsr", "tli", "cfi"):
        np.testing.assert_allclose(out90[col].iloc[0], out95[col].iloc[0], rtol=1e-14)
    assert out95["rmsea_2.5"].iloc[0] <= out90["rmsea_5"].iloc[0]
    assert out95["rmsea_97.5"].iloc[0] >= out90["rmsea_95"].iloc[0]


def test_m2_all_fixed_and_null_warning() -> None:
    """All-fixed model (n_params=0) works; a badly misspecified model triggers the null warning."""
    df = _sim_2pl(n_obs=400, n_items=5, seed=8)
    items = list(df.columns)
    spec = Model.uniform(TwoPL(), items=items)
    table = spec.parameter_table(df)
    table["free"] = False
    # Strongly misspecified intercepts/slopes so model M2 exceeds null M2.
    table.loc[table["param"] == "a", "value"] = -2.5
    table.loc[table["param"] == "d", "value"] = 2.0
    res = fit(spec, df, table=table, se=None)
    assert res.n_params == 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = res.m2(df)
    assert any(issubclass(w.category, FlexpyirtWarning) for w in caught)
    row = out.iloc[0]
    assert int(row["df"]) == 15  # 5 + 10 - 0
    assert np.isfinite(row["m2"]) and row["m2"] > 0.0
    assert np.isnan(row["tli"])
    assert np.isnan(row["cfi"])
