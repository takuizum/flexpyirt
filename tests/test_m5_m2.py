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
    PCM,
    EmpiricalHistogram,
    FourPL,
    Model,
    Normal,
    NormalPrior,
    Rasch,
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


def _sim_grm(
    n_obs: int = 400, n_items: int = 6, n_categories: int = 3, seed: int = 43
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.lognormal(0.1, 0.25, n_items)
    d = np.sort(rng.normal(0.0, 1.0, (n_items, n_categories - 1)), axis=1)[:, ::-1]
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


@pytest.mark.slow
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
    with pytest.raises(SpecificationError, match="single-group models"):
        res_mg.m2(df)

    # EmpiricalHistogram latent distribution.
    spec_eh = Model.uniform(TwoPL(), items=items).with_latent(EmpiricalHistogram())
    res_eh = fit(spec_eh, df, method=BAEM(tol=1e-3, max_iter=50, squarem=True), se=None)
    with pytest.raises(SpecificationError, match="single-group models"):
        res_eh.m2(df)

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

    # 1-item model.
    res_1item = fit(
        Model.uniform(Rasch(), items=["q01"]).with_latent(Normal.standard()),
        df[["q01"]],
        method=BAEM(tol=1e-4, squarem=True),
        se=None,
    )
    with pytest.raises(SpecificationError, match="at least 2 items"):
        res_1item.m2(df[["q01"]])


@pytest.mark.slow
@pytest.mark.parametrize("n_items,expected_df", [(3, 0), (2, -1)])
def test_m2_nonpositive_df_returns_nan_indices(n_items: int, expected_df: int) -> None:
    df = _sim_2pl(n_obs=400, n_items=n_items, seed=2)
    res = fit(
        Model.uniform(TwoPL(), items=list(df.columns)),
        df,
        method=BAEM(tol=1e-6, squarem=True),
        se=None,
    )
    with pytest.warns(FlexpyirtWarning, match="non-positive degrees of freedom"):
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


@pytest.mark.slow
def test_m2_nonpositive_df_with_low_rank_and_rank_deficient_positive_df() -> None:
    """Both df <= 0 with rank < s and df > 0 with rank < n_cols warn and return NaN M2."""
    # Case 1: df <= 0 with rank(Delta_2) < s (3 GPCM(4) items with slopes fixed at 0:
    # s = 6, q = 9, df = -3, rank = 3 < 6).
    df3 = _sim_grm(n_obs=400, n_items=3, n_categories=4, seed=11)
    spec_low = Model.uniform(GPCM(n_categories=4), items=list(df3.columns))
    for item in df3.columns:
        spec_low = spec_low.fix(item=item, a=0.0)
    res_low = fit(spec_low, df3, method=BAEM(tol=1e-6, squarem=True), se=None)
    with pytest.warns(FlexpyirtWarning, match="not locally identified"):
        out_low = res_low.m2(df3)
    row_low = out_low.iloc[0]
    assert int(row_low["df"]) == -3
    for col in ("m2", "p", "rmsea", "rmsea_5", "rmsea_95", "tli", "cfi"):
        assert np.isnan(row_low[col])
    assert np.isfinite(row_low["srmsr"])

    # Case 2: df > 0 with rank-deficient Delta_2 (4 Rasch + 1 GPCM(7), Normal(0,1) fixed:
    # s = 15, q = 4 + 7 = 11, df = 4, but item 5's 7 columns lie in 5 rows -> rank <= 9 < 11).
    rng = np.random.default_rng(12)
    n_obs = 600
    theta = rng.normal(0.0, 1.0, n_obs)
    d_r = rng.normal(0.0, 0.5, 4)
    y_r = (rng.random((n_obs, 4)) < 1.0 / (1.0 + np.exp(-(theta[:, None] + d_r)))).astype(int)
    y_p = rng.integers(0, 7, size=(n_obs, 1))
    df5 = pd.DataFrame(np.hstack([y_r, y_p]), columns=["r1", "r2", "r3", "r4", "p5"])
    spec_rd = Model.from_items(
        {"r1": Rasch(), "r2": Rasch(), "r3": Rasch(), "r4": Rasch(), "p5": GPCM(n_categories=7)}
    ).with_latent(Normal.standard())
    res_rd = fit(spec_rd, df5, method=BAEM(tol=1e-4, max_iter=200, squarem=True), se=None)
    assert res_rd.n_params == 11
    with pytest.warns(FlexpyirtWarning, match="not locally identified"):
        out_rd = res_rd.m2(df5)
    row_rd = out_rd.iloc[0]
    assert int(row_rd["df"]) == 4
    for col in ("m2", "p", "rmsea", "rmsea_5", "rmsea_95", "tli", "cfi"):
        assert np.isnan(row_rd[col])
    assert np.isfinite(row_rd["srmsr"])


@pytest.mark.slow
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
    dfg = _sim_grm(n_obs=500, n_items=6, n_categories=3, seed=4)
    gperm = ["g04", "g01", "g06", "g02", "g05", "g03"]
    rg1 = fit(
        Model.uniform(GRM(n_categories=3), items=list(dfg.columns)),
        dfg,
        method=BAEM(tol=1e-9, squarem=True),
        se=None,
    ).m2(dfg)
    rg2 = fit(
        Model.uniform(GRM(n_categories=3), items=gperm),
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

    # Non-positive df or n <= 1 gives (NaN, NaN).
    lo_n, hi_n = _m2.rmsea_ci(10.0, df=0, n=400)
    assert np.isnan(lo_n) and np.isnan(hi_n)
    lo_n1, hi_n1 = _m2.rmsea_ci(10.0, df=5, n=1)
    assert np.isnan(lo_n1) and np.isnan(hi_n1)

    # Small N or high confidence level: upper bracket expands instead of collapsing to 0.
    lo_hi, hi_hi = _m2.rmsea_ci(8.0, df=2, n=50, level=0.9999)
    assert lo_hi == 0.0 < _m2.rmsea(8.0, df=2, n=50) < hi_hi
    np.testing.assert_allclose(stats.ncx2.cdf(8.0, 2, 50 * 2 * hi_hi**2), 0.00005, atol=1e-10)
    lo_hi2, hi_hi2 = _m2.rmsea_ci(25.0, df=2, n=50, level=0.9999)
    assert 0.0 < lo_hi2 < _m2.rmsea(25.0, df=2, n=50) < hi_hi2
    np.testing.assert_allclose(stats.ncx2.cdf(25.0, 2, 50 * 2 * lo_hi2**2), 0.99995, atol=1e-10)
    np.testing.assert_allclose(stats.ncx2.cdf(25.0, 2, 50 * 2 * hi_hi2**2), 0.00005, atol=1e-10)
    lo_sm, hi_sm = _m2.rmsea_ci(1.0, df=1, n=3, level=0.90)
    assert lo_sm == 0.0 and hi_sm > 0.0
    np.testing.assert_allclose(stats.ncx2.cdf(1.0, 1, 3 * 1 * hi_sm**2), 0.05, atol=1e-10)

    # CI column names and width ordering.
    assert _m2.ci_column_names(0.90) == ("rmsea_5", "rmsea_95")
    assert _m2.ci_column_names(0.95) == ("rmsea_2.5", "rmsea_97.5")
    lo95, hi95 = _m2.rmsea_ci(x2, df, n, level=0.95)
    assert lo95 < lo < hi < hi95


@pytest.mark.slow
def test_xi_is_symmetric_psd_and_matches_brute_force() -> None:
    df_bin = _sim_2pl(n_obs=300, n_items=7, seed=5)
    df_poly = _sim_grm(n_obs=300, n_items=6, n_categories=4, seed=6)
    configs = [
        (Model.uniform(TwoPL(), items=list(df_bin.columns)), df_bin),
        (Model.uniform(ThreePL(), items=list(df_bin.columns)), df_bin),
        (Model.uniform(GRM(n_categories=4), items=list(df_poly.columns)), df_poly),
        (Model.uniform(GPCM(n_categories=4), items=list(df_poly.columns)), df_poly),
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
def test_null_m2_matches_explicit_independence_model() -> None:
    """Closed-form null_m2 equals m2_statistic on an explicit a=0 dichotomous independence model."""
    df = _sim_2pl(n_obs=500, n_items=6, seed=9)
    items = list(df.columns)
    spec = Model.uniform(TwoPL(), items=items)
    table = spec.parameter_table(df)
    table.loc[table["param"] == "a", ["value", "free"]] = [0.0, False]
    p_bar = df.mean(axis=0)
    for item in items:
        p = float(p_bar[item])
        table.loc[(table["item"] == item) & (table["param"] == "d"), "value"] = np.log(
            p / (1.0 - p)
        )
    res_null = fit(spec, df, table=table, method=BAEM(tol=1e-11, squarem=True), se=None)
    y = coerce_responses(df, res_null._model)
    codes, _ = coerce_groups(None, res_null._model, y.shape[0], require_all=False)
    compiled = compile_problem(res_null._model, y, codes, res_null.parameters, res_null._quadrature)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FlexpyirtWarning)
        explicit = _m2.m2_statistic(
            compiled.problem,
            jnp.asarray(res_null._eta),
            y,
            n_params=res_null.n_params,
            n_categories=[2] * len(items),
        )
    closed_stat, closed_df = _m2.null_m2(y)
    assert closed_df == explicit.df
    np.testing.assert_allclose(closed_stat, explicit.m2, rtol=1e-12)


@pytest.mark.slow
def test_m2_fourpl_and_pcm() -> None:
    """FourPL (dichotomous) and PCM (polytomous Rasch with free latent variance) work cleanly."""
    # FourPL with J=10: s = 55, q = 40 -> df = 15.
    rng = np.random.default_rng(13)
    n_obs, n_items = 800, 10
    theta = rng.normal(0.0, 1.0, n_obs)
    a = rng.lognormal(0.3, 0.2, n_items)
    d = rng.normal(0.0, 0.5, n_items)
    g = rng.uniform(0.10, 0.20, n_items)
    u = rng.uniform(0.82, 0.92, n_items)
    prob = g + (u - g) / (1.0 + np.exp(-(np.outer(theta, a) + d)))
    df4 = pd.DataFrame(
        (rng.random(prob.shape) < prob).astype(int),
        columns=[f"f{j + 1:02d}" for j in range(n_items)],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", HeywoodWarning)
        res4 = fit(
            Model.uniform(FourPL(), items=list(df4.columns)),
            df4,
            method=BAEM(tol=1e-4, max_iter=300, squarem=True),
            se=None,
        )
    out4 = res4.m2(df4).iloc[0]
    assert int(out4["df"]) == 15
    for col in ("m2", "p", "rmsea", "rmsea_5", "rmsea_95", "srmsr", "tli", "cfi"):
        assert np.isfinite(out4[col])

    # PCM(n_categories=3) with J=6: s = 21, q = 6*2 + 1 = 13 -> df = 8; latent variance in
    # Delta differs from _latent_in_delta=False, and tli/cfi are NaN (polytomous).
    df_pcm = _sim_grm(n_obs=500, n_items=6, n_categories=3, seed=14)
    res_pcm = fit(
        Model.uniform(PCM(n_categories=3), items=list(df_pcm.columns)),
        df_pcm,
        method=BAEM(tol=1e-7, squarem=True),
        se=None,
    )
    out_pcm = res_pcm.m2(df_pcm).iloc[0]
    assert int(out_pcm["df"]) == 8
    for col in ("m2", "p", "rmsea", "rmsea_5", "rmsea_95", "srmsr"):
        assert np.isfinite(out_pcm[col])
    assert np.isnan(out_pcm["tli"]) and np.isnan(out_pcm["cfi"])

    y = coerce_responses(df_pcm, res_pcm._model)
    codes, _ = coerce_groups(None, res_pcm._model, y.shape[0], require_all=False)
    compiled = compile_problem(res_pcm._model, y, codes, res_pcm.parameters, res_pcm._quadrature)
    nolat = _m2.m2_statistic(
        compiled.problem,
        jnp.asarray(res_pcm._eta),
        y,
        n_params=res_pcm.n_params,
        n_categories=[3] * 6,
        _latent_in_delta=False,
    )
    assert abs(float(out_pcm["m2"]) - nolat.m2) > 1e-6
