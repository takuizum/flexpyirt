"""Unit tests for item-parameter priors and MAP estimation (DESIGN §4.7, §4.8, §5.1)."""

from __future__ import annotations

import warnings
from fractions import Fraction
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from flexpyirt import (
    BAEM,
    GRM,
    BetaPrior,
    FourPL,
    LogNormalPrior,
    Model,
    Normal,
    NormalPrior,
    Quadrature,
    ThreePL,
    TwoPL,
    _core,
    fit,
)
from flexpyirt._compile import compile_problem, prepare
from flexpyirt.errors import FlexpyirtWarning, HeywoodWarning, SpecificationError
from flexpyirt.priors import (
    PRIOR_BETA,
    PRIOR_LOGNORMAL,
    PRIOR_NONE,
    PRIOR_NORMAL,
    eval_log_prior,
)
from tests._reference import load as load_m1


def _simulate_2pl(a: np.ndarray, d: np.ndarray, n: int, seed: int, sd: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, sd, size=(n, 1))
    p = 1.0 / (1.0 + np.exp(-(theta * a + d)))
    y = (rng.random(p.shape) < p).astype(int)
    return pd.DataFrame(y, columns=[f"q{j + 1:02d}" for j in range(len(a))])


def _simulate_3pl(a: np.ndarray, d: np.ndarray, g: np.ndarray, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, 1.0, size=(n, 1))
    p = g + (1.0 - g) / (1.0 + np.exp(-(theta * a + d)))
    y = (rng.random(p.shape) < p).astype(int)
    return pd.DataFrame(y, columns=[f"q{j + 1:02d}" for j in range(len(a))])


# ---- prior distributions: log_pdf against scipy.stats & validation --------------------------


def test_normal_prior_log_pdf_matches_scipy() -> None:
    xs = np.array([-3.5, -1.0, 0.0, 0.75, 4.2])
    pri = NormalPrior(mean=0.3, sd=1.7)
    got = np.asarray(pri.log_pdf(xs))
    expected = stats.norm.logpdf(xs, loc=0.3, scale=1.7)
    np.testing.assert_allclose(got, expected, rtol=1e-14, atol=1e-14)
    assert float(pri.log_pdf(0.3)) == pytest.approx(float(stats.norm.logpdf(0.3, 0.3, 1.7)))


def test_lognormal_prior_log_pdf_matches_scipy() -> None:
    xs = np.array([0.05, 0.5, 1.0, 1.8, 5.0])
    pri = LogNormalPrior(mean_log=0.2, sd_log=0.35)
    got = np.asarray(pri.log_pdf(xs))
    # scipy.stats.lognorm(s=sd_log, scale=exp(mean_log)) matches R dlnorm(x, mean_log, sd_log).
    expected = stats.lognorm.logpdf(xs, s=0.35, scale=np.exp(0.2))
    np.testing.assert_allclose(got, expected, rtol=1e-14, atol=1e-14)
    assert float(pri.log_pdf(0.0)) == -np.inf
    assert float(pri.log_pdf(-1.0)) == -np.inf


def test_beta_prior_log_pdf_matches_scipy() -> None:
    xs = np.array([0.01, 0.15, 0.25, 0.5, 0.92])
    pri = BetaPrior(a=5.0, b=17.0)
    got = np.asarray(pri.log_pdf(xs))
    expected = stats.beta.logpdf(xs, a=5.0, b=17.0)
    np.testing.assert_allclose(got, expected, rtol=1e-13, atol=1e-13)
    assert float(pri.log_pdf(0.0)) == -np.inf
    assert float(pri.log_pdf(1.0)) == -np.inf
    assert float(pri.log_pdf(-0.1)) == -np.inf
    assert float(pri.log_pdf(1.2)) == -np.inf


def test_prior_parameter_validation() -> None:
    bad_pos: tuple[Any, ...] = (
        0.0,
        -1.0,
        np.int64(0),
        np.int64(-3),
        np.float32(np.nan),
        np.float32(np.inf),
        np.nan,
        np.inf,
        -np.inf,
        True,
        np.bool_(True),
        1 + 0j,
        np.complex128(1.0),
        "1.0",
        np.array(1.0),
        10**400,
    )
    for bad in bad_pos:
        with pytest.raises(SpecificationError):
            NormalPrior(0.0, bad)
        with pytest.raises(SpecificationError):
            LogNormalPrior(0.0, bad)
        with pytest.raises(SpecificationError):
            BetaPrior(bad, 2.0)
        with pytest.raises(SpecificationError):
            BetaPrior(2.0, bad)
    bad_loc: tuple[Any, ...] = (
        np.nan,
        np.inf,
        -np.inf,
        np.float32(np.nan),
        False,
        np.bool_(False),
        1 + 0j,
        np.complex128(0.0),
        "0.0",
        np.array(0.0),
        10**400,
    )
    for bad in bad_loc:
        with pytest.raises(SpecificationError):
            NormalPrior(bad, 1.0)
        with pytest.raises(SpecificationError):
            LogNormalPrior(bad, 1.0)


def test_prior_accepts_real_numeric_scalars_and_stores_python_floats() -> None:
    """Python and NumPy real scalars (non-bool) are accepted and normalized to Python float."""
    beta_i64 = BetaPrior(np.int64(5), np.int64(17))
    assert beta_i64 == BetaPrior(5.0, 17.0)
    assert type(beta_i64.a) is float and type(beta_i64.b) is float

    norm_np = NormalPrior(np.int32(-2), np.float32(1.5))
    assert norm_np == NormalPrior(-2.0, 1.5)
    assert type(norm_np.mean) is float and type(norm_np.sd) is float

    ln_np = LogNormalPrior(np.float16(0.5), np.uint8(2))
    assert ln_np == LogNormalPrior(0.5, 2.0)
    assert type(ln_np.mean_log) is float and type(ln_np.sd_log) is float

    beta_frac = BetaPrior(Fraction(5, 2), 3)
    assert beta_frac == BetaPrior(2.5, 3.0)
    assert type(beta_frac.a) is float and type(beta_frac.b) is float


def test_eval_log_prior_branch_free_gradients_and_hessians() -> None:
    """Unselected branches and padding slots never leak NaN/inf into autodiff."""
    # Include negative, zero, and >1 values on NONE / NORMAL slots to stress log guards.
    x = jnp.array([-2.0, 0.0, 1.5, -1.3, 1.4, 0.22])
    kind = jnp.array(
        [PRIOR_NONE, PRIOR_NONE, PRIOR_NONE, PRIOR_NORMAL, PRIOR_LOGNORMAL, PRIOR_BETA],
        dtype=jnp.int32,
    )
    p1 = jnp.array([0.0, 0.0, 0.0, 0.0, 0.2, 5.0])
    p2 = jnp.array([1.0, 1.0, 1.0, 2.0, 0.3, 17.0])

    vals = eval_log_prior(x, kind, p1, p2)
    np.testing.assert_allclose(vals[:3], 0.0, atol=0.0)
    assert float(vals[3]) == pytest.approx(float(NormalPrior(0.0, 2.0).log_pdf(-1.3)), rel=1e-14)
    assert float(vals[4]) == pytest.approx(float(LogNormalPrior(0.2, 0.3).log_pdf(1.4)), rel=1e-14)
    assert float(vals[5]) == pytest.approx(float(BetaPrior(5.0, 17.0).log_pdf(0.22)), rel=1e-14)

    def total(v: jax.Array) -> jax.Array:
        return jnp.sum(eval_log_prior(v, kind, p1, p2))

    grad = jax.grad(total)(x)
    hess = jax.hessian(total)(x)
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert bool(jnp.all(jnp.isfinite(hess)))
    np.testing.assert_array_equal(np.asarray(grad[:3]), 0.0)
    np.testing.assert_array_equal(np.asarray(hess[:3, :3]), 0.0)

    # Analytic 1st and 2nd derivatives at the three active entries:
    expected_grad = np.array(
        [
            -(-1.3 - 0.0) / 4.0,
            -(1.0 + (np.log(1.4) - 0.2) / 0.09) / 1.4,
            4.0 / 0.22 - 16.0 / 0.78,
        ]
    )
    expected_hess_diag = np.array(
        [
            -1.0 / 4.0,
            1.0 / 1.4**2 - 1.0 / (1.4**2 * 0.09) + (np.log(1.4) - 0.2) / (1.4**2 * 0.09),
            -4.0 / 0.22**2 - 16.0 / 0.78**2,
        ]
    )
    np.testing.assert_allclose(np.asarray(grad[3:]), expected_grad, rtol=1e-12)
    np.testing.assert_allclose(np.diag(np.asarray(hess))[3:], expected_hess_diag, rtol=1e-12)


# ---- Model.prior() API and ParameterTable validation ----------------------------------------


def test_model_prior_validation_and_table_column() -> None:
    df = load_m1("v1_lsat7_2pl").data
    spec = Model.uniform(TwoPL(), items=df.columns)

    with pytest.raises(SpecificationError, match="Unsupported prior"):
        spec.prior(param="a", dist=Normal.standard())  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match="latent-distribution"):
        spec.prior(param="mean", dist=NormalPrior(0.0, 1.0))
    with pytest.raises(SpecificationError, match="latent-distribution"):
        spec.prior(param="var", dist=LogNormalPrior(0.0, 0.5))
    with pytest.raises(SpecificationError, match="no parameter"):
        spec.prior(param="g", dist=BetaPrior(5.0, 17.0))
    with pytest.raises(SpecificationError, match="selected no items"):
        spec.prior(param="a", dist=LogNormalPrior(0.0, 0.5), items=[])
    with pytest.raises(SpecificationError, match="Unknown items"):
        spec.prior(param="a", dist=LogNormalPrior(0.0, 0.5), items=["missing"])

    # Callable selector + override on one item
    spec_p = (
        spec.prior(param="a", dist=LogNormalPrior(0.0, 0.5))
        .prior(param="a", dist=NormalPrior(1.0, 0.5), items=["Item.1"])
        .prior(param="d", dist=NormalPrior(0.0, 2.0), items=lambda n: n.endswith(("1", "2")))
    )
    table = spec_p.parameter_table(df)
    assert "prior" in table.columns
    assert table["prior"].dtype == object
    idx = table.set_index(["item", "param"])["prior"]
    assert idx[("Item.1", "a")] == NormalPrior(1.0, 0.5)
    assert idx[("Item.2", "a")] == LogNormalPrior(0.0, 0.5)
    assert idx[("Item.1", "d")] == NormalPrior(0.0, 2.0)
    assert idx[("Item.3", "d")] is None
    assert idx[("__latent__", "mean")] is None
    assert idx[("__latent__", "var")] is None


# ---- Fixed parameter ignores its prior ------------------------------------------------------


def test_prior_on_fixed_parameter_has_no_effect() -> None:
    df = load_m1("v1_lsat7_2pl").data
    base = Model.uniform(TwoPL(), items=df.columns).fix(item="Item.1", a=1.0, d=0.0)
    with_fixed_prior = base.prior(param="a", dist=LogNormalPrior(0.2, 0.3), items=["Item.1"]).prior(
        param="d", dist=NormalPrior(5.0, 0.1), items=["Item.1"]
    )

    r_base = fit(base, df, method=BAEM(tol=1e-9), se="oakes")
    r_pri = fit(with_fixed_prior, df, method=BAEM(tol=1e-9), se="oakes")

    assert r_pri.loglik == pytest.approx(r_base.loglik, rel=1e-14)
    assert r_pri.log_posterior == pytest.approx(r_pri.loglik, rel=1e-15)
    assert r_pri.log_posterior == pytest.approx(r_base.log_posterior, rel=1e-14)
    np.testing.assert_allclose(r_pri.items["a"], r_base.items["a"], atol=1e-12)
    np.testing.assert_allclose(r_pri.items["d"], r_base.items["d"], atol=1e-12)
    assert r_pri.vcov is not None and r_base.vcov is not None
    np.testing.assert_allclose(r_pri.vcov, r_base.vcov, rtol=1e-12)

    # No active priors -> AIC and BIC must NOT warn.
    with warnings.catch_warnings():
        warnings.simplefilter("error", FlexpyirtWarning)
        _ = r_pri.aic
        _ = r_pri.bic

    # Even an out-of-support fixed value ignores its prior.
    neg_fixed = (
        Model.uniform(TwoPL(), items=df.columns)
        .fix(item="Item.1", a=-0.5)
        .prior(param="a", dist=LogNormalPrior(0.0, 0.5), items=["Item.1"])
    )
    r_neg = fit(neg_fixed, df, se=None)
    assert r_neg.converged
    assert r_neg.log_posterior == pytest.approx(r_neg.loglik, rel=1e-15)


# ---- Shared (free_id) parameter counted once ------------------------------------------------


def test_shared_free_id_prior_counted_once_across_groups() -> None:
    a = np.array([1.1, 1.4, 0.9, 1.3, 1.2, 1.0])
    d = np.array([-0.8, -0.3, 0.2, 0.6, -0.4, 0.5])
    df_ref = _simulate_2pl(a, d, 500, seed=10)
    df_foc = _simulate_2pl(a, d, 500, seed=11, sd=1.1)
    df = pd.concat([df_ref, df_foc], ignore_index=True)
    group = np.repeat(["ref", "focal"], 500)
    anchors = ["q01", "q02", "q03"]

    spec = (
        Model.uniform(TwoPL(), items=df.columns)
        .with_groups(ref=Normal.standard(), focal=Normal.free())
        .equal_across_groups(items=anchors)
        .prior(param="a", dist=LogNormalPrior(0.2, 0.3))
        .prior(param="d", dist=NormalPrior(0.0, 2.0))
    )
    res_oakes = fit(spec, df, group=group, method=BAEM(tol=1e-9, max_iter=5000), se="oakes")
    res_hess = fit(spec, df, group=group, method=BAEM(tol=1e-9, max_iter=5000), se="hessian")
    assert res_oakes.converged

    # Sum log_pdf over unique free_ids only (each anchor counted once, non-anchors once per group).
    unique_free = res_oakes.parameters[res_oakes.parameters["free"]].drop_duplicates("free_id")
    expected_lp = sum(
        float(row["prior"].log_pdf(row["estimate"]))
        for _, row in unique_free.iterrows()
        if row["prior"] is not None
    )
    assert res_oakes.log_posterior - res_oakes.loglik == pytest.approx(expected_lp, rel=1e-12)

    # Confirm the anchor prior was NOT double-counted.
    n_anchors = len(anchors)
    n_non_anchors = len(a) - n_anchors
    n_priors_active = sum(p is not None for p in unique_free["prior"])
    assert n_priors_active == 2 * (n_anchors + 2 * n_non_anchors)

    # Oakes == Hessian on the multigroup MAP model too.
    assert res_oakes.vcov is not None and res_hess.vcov is not None
    np.testing.assert_allclose(res_oakes.vcov, res_hess.vcov, rtol=1e-6, atol=1e-12)

    # Conflicting priors on rows sharing a free_id must raise SpecificationError.
    bad_table = spec.parameter_table(df, group=group)
    row_foc_q01_a = (
        (bad_table["group"] == "focal") & (bad_table["item"] == "q01") & (bad_table["param"] == "a")
    )
    bad_table.loc[row_foc_q01_a, "prior"] = LogNormalPrior(0.5, 0.5)  # type: ignore[call-overload]
    with pytest.raises(SpecificationError, match="conflicting priors"):
        fit(spec, df, group=group, table=bad_table, se=None)


# ---- Edited ParameterTable priors -----------------------------------------------------------


def test_edited_table_priors_match_spec_and_validate() -> None:
    df = load_m1("v1_lsat7_2pl").data
    spec_plain = Model.uniform(TwoPL(), items=df.columns)
    spec_pri = spec_plain.prior(param="a", dist=LogNormalPrior(0.1, 0.4)).prior(
        param="d", dist=NormalPrior(0.0, 2.0)
    )

    table = spec_plain.parameter_table(df)
    items_mask = table["item"] != "__latent__"
    table.loc[items_mask & (table["param"] == "a"), "prior"] = LogNormalPrior(0.1, 0.4)  # type: ignore[call-overload]
    table.loc[items_mask & (table["param"] == "d"), "prior"] = NormalPrior(0.0, 2.0)  # type: ignore[call-overload]

    r_spec = fit(spec_pri, df, method=BAEM(tol=1e-9), se="oakes")
    r_tbl = fit(spec_plain, df, table=table, method=BAEM(tol=1e-9), se="oakes")
    assert r_tbl.loglik == pytest.approx(r_spec.loglik, rel=1e-14)
    assert r_tbl.log_posterior == pytest.approx(r_spec.log_posterior, rel=1e-14)
    np.testing.assert_allclose(r_tbl.items["a"], r_spec.items["a"], atol=1e-12)
    np.testing.assert_allclose(r_tbl.items["d"], r_spec.items["d"], atol=1e-12)
    assert r_tbl.vcov is not None and r_spec.vcov is not None
    np.testing.assert_allclose(r_tbl.vcov, r_spec.vcov, rtol=1e-12)

    # Clearing priors in the table reverts to MLE.
    cleared = spec_pri.parameter_table(df)
    cleared["prior"] = None
    r_mle = fit(spec_plain, df, method=BAEM(tol=1e-9), se=None)
    r_clr = fit(spec_pri, df, table=cleared, method=BAEM(tol=1e-9), se=None)
    assert r_clr.loglik == pytest.approx(r_mle.loglik, rel=1e-13)
    assert r_clr.log_posterior == pytest.approx(r_clr.loglik, rel=1e-15)

    # Invalid prior edits must raise SpecificationError.
    t_lat = spec_plain.parameter_table(df)
    t_lat.loc[t_lat["item"] == "__latent__", "prior"] = NormalPrior(0.0, 1.0)  # type: ignore[call-overload]
    with pytest.raises(SpecificationError, match="latent-distribution"):
        fit(spec_plain, df, table=t_lat, se=None)

    t_bad = spec_plain.parameter_table(df)
    t_bad.loc[0, "prior"] = "norm"
    with pytest.raises(SpecificationError, match="Invalid prior"):
        fit(spec_plain, df, table=t_bad, se=None)

    t_oob = spec_plain.parameter_table(df)
    row_a = (t_oob["item"] == "Item.1") & (t_oob["param"] == "a")
    t_oob.loc[row_a, "value"] = -0.5
    t_oob.loc[row_a, "prior"] = LogNormalPrior(0.0, 0.5)  # type: ignore[call-overload]
    with pytest.raises(SpecificationError, match="outside the support"):
        fit(spec_plain, df, table=t_oob, se=None)

    t_beta_oob = spec_plain.parameter_table(df)
    row_d = (t_beta_oob["item"] == "Item.1") & (t_beta_oob["param"] == "d")
    t_beta_oob.loc[row_d, "value"] = 1.5
    t_beta_oob.loc[row_d, "prior"] = BetaPrior(5.0, 17.0)  # type: ignore[call-overload]
    with pytest.raises(SpecificationError, match="outside the support"):
        fit(spec_plain, df, table=t_beta_oob, se=None)


# ---- AIC / BIC warnings, repr, posterior information, SQUAREM monotonicity -----------------


def test_aic_bic_warn_when_priors_present_and_silent_without() -> None:
    df = load_m1("v1_lsat7_2pl").data
    r_mle = fit(Model.uniform(TwoPL(), items=df.columns), df, se=None)
    assert r_mle.log_posterior == pytest.approx(r_mle.loglik, rel=1e-15)
    assert "log_posterior" not in repr(r_mle)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FlexpyirtWarning)
        _ = r_mle.aic
        _ = r_mle.bic

    spec_map = Model.uniform(TwoPL(), items=df.columns).prior(
        param="a", dist=LogNormalPrior(0.0, 0.5)
    )
    r_map = fit(spec_map, df, se=None)
    assert r_map.log_posterior != pytest.approx(r_map.loglik)
    assert "log_posterior=" in repr(r_map)
    with pytest.warns(FlexpyirtWarning, match="AIC"):
        aic = r_map.aic
    with pytest.warns(FlexpyirtWarning, match="BIC"):
        bic = r_map.bic
    assert aic == pytest.approx(-2.0 * r_map.loglik + 2.0 * r_map.n_params)
    assert bic == pytest.approx(-2.0 * r_map.loglik + r_map.n_params * np.log(r_map.n_obs))


def test_posterior_information_adds_prior_hessian_exactly() -> None:
    """I_MAP(eta) - I_lik(eta) == -diag(d^2/dx^2 log_pdf(x)) for both Oakes and Hessian."""
    a = np.array([1.2, 1.5, 1.0, 1.3, 1.6])
    d = np.array([-0.5, 0.2, -0.8, 0.4, 0.0])
    g = np.array([0.18, 0.22, 0.15, 0.20, 0.25])
    df = _simulate_3pl(a, d, g, n=800, seed=99)

    spec_lik = Model.uniform(ThreePL(), items=df.columns)
    spec_map = (
        spec_lik.prior(param="a", dist=LogNormalPrior(0.2, 0.3))
        .prior(param="d", dist=NormalPrior(0.0, 2.0))
        .prior(param="g", dist=BetaPrior(5.0, 17.0))
    )
    prep_lik = prepare(spec_lik, df, group=None, item_names=None, table=None)
    prep_map = prepare(spec_map, df, group=None, item_names=None, table=None)
    comp_lik = compile_problem(
        spec_lik, prep_lik.y, prep_lik.group, prep_lik.table, Quadrature.equispaced()
    )
    comp_map = compile_problem(
        spec_map, prep_map.y, prep_map.group, prep_map.table, Quadrature.equispaced()
    )
    eta = jnp.asarray(comp_map.eta0)

    info_lik = np.asarray(_core.information_oakes(comp_lik.problem, eta))
    info_map_oakes = np.asarray(_core.information_oakes(comp_map.problem, eta))
    info_map_hess = np.asarray(_core.information_hessian(comp_map.problem, eta))
    prior_hess = np.asarray(jax.hessian(lambda e: _core.log_prior(comp_map.problem, e))(eta))

    np.testing.assert_allclose(info_map_oakes, info_map_hess, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(info_map_oakes - info_lik, -prior_hess, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("squarem", [False, True])
def test_map_3pl_monotone_and_prevents_heywood(squarem: bool) -> None:
    """Beta(5, 17) prior stabilizes 3PL guessing estimates and stays monotone under EM/SQUAREM."""
    a = np.array([1.2, 1.5, 0.9, 1.4, 1.1, 1.3])
    d = np.array([-0.5, 0.3, -1.0, 0.7, -0.2, 0.5])
    g = np.array([0.2, 0.15, 0.25, 0.18, 0.22, 0.19])
    df = _simulate_3pl(a, d, g, n=600, seed=2026)

    spec = (
        Model.uniform(ThreePL(), items=df.columns)
        .prior(param="a", dist=LogNormalPrior(0.2, 0.3))
        .prior(param="d", dist=NormalPrior(0.0, 2.0))
        .prior(param="g", dist=BetaPrior(5.0, 17.0))
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", HeywoodWarning)
        res = fit(spec, df, method=BAEM(tol=1e-8, max_iter=5000, squarem=squarem), se="oakes")

    assert res.converged
    lp_hist = res.history["log_posterior"].to_numpy()
    assert np.all(np.diff(lp_hist) >= -1e-9 * np.abs(lp_hist[1:]))
    assert res.log_posterior >= lp_hist[-1] - 1e-9 * abs(lp_hist[-1])
    assert (res.items["a"] > 0).all()
    assert ((res.items["g"] > 0.01) & (res.items["g"] < 0.99)).all()


def _simulate_grm(a: np.ndarray, d: np.ndarray, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, 1.0, size=(n, 1))
    cols: dict[str, np.ndarray] = {}
    for j in range(len(a)):
        cum = 1.0 / (1.0 + np.exp(-(theta * a[j] + d[j])))
        u = rng.random((n, 1))
        cols[f"q{j + 1:02d}"] = (u < cum).sum(axis=1).astype(int)
    return pd.DataFrame(cols)


def test_default_starts_adjusted_into_prior_support() -> None:
    """Default starts move strictly into (model bounds ∩ prior support) for 2PL, 3PL/4PL, GRM."""
    # 1. BetaPrior(2, 2) on 2PL slopes (heuristic a0 > 1 on LSAT7 Item.2).
    df_lsat = load_m1("v1_lsat7_2pl").data
    spec_beta_a = Model.uniform(TwoPL(), items=df_lsat.columns).prior(
        param="a", dist=BetaPrior(2.0, 2.0)
    )
    tbl_beta_a = spec_beta_a.parameter_table(df_lsat)
    a_starts = tbl_beta_a.loc[tbl_beta_a["param"] == "a", "value"].to_numpy()
    assert ((a_starts > 0.0) & (a_starts < 1.0)).all()
    res_beta_a = fit(spec_beta_a, df_lsat, method=BAEM(tol=1e-7), se=None)
    assert res_beta_a.converged
    assert ((res_beta_a.items["a"] > 0.0) & (res_beta_a.items["a"] < 1.0)).all()

    # 2. LogNormalPrior(0, 0.5) on 2PL intercepts when items have proportion correct <= 0.5
    #    (d0 <= 0).
    df_hard = _simulate_2pl(
        np.array([1.1, 1.3, 1.0, 1.2, 1.4]),
        np.array([-1.2, -0.5, 0.0, 0.4, -0.8]),
        n=400,
        seed=42,
    )
    spec_ln_d = Model.uniform(TwoPL(), items=df_hard.columns).prior(
        param="d", dist=LogNormalPrior(0.0, 0.5)
    )
    tbl_ln_d = spec_ln_d.parameter_table(df_hard)
    d_starts = tbl_ln_d.loc[tbl_ln_d["param"] == "d", "value"].to_numpy()
    assert (d_starts > 0.0).all()
    res_ln_d = fit(spec_ln_d, df_hard, method=BAEM(tol=1e-7), se=None)
    assert res_ln_d.converged
    assert (res_ln_d.items["d"] > 0.0).all()

    # Extreme LogNormalPrior mean_log (exp overflow / underflow) still yields a finite > 0 start.
    for extreme_meanlog in (1000.0, -1000.0):
        t_ext = (
            Model.uniform(TwoPL(), items=df_hard.columns)
            .prior(param="d", dist=LogNormalPrior(extreme_meanlog, 1.0))
            .parameter_table(df_hard)
        )
        d_ext = t_ext.loc[t_ext["param"] == "d", "value"].to_numpy()
        assert (np.isfinite(d_ext) & (d_ext > 0.0)).all()

    # 3. 3PL and 4PL keep 0 < g < u < 1 with priors and fixed asymptotes.
    df_3pl = _simulate_3pl(
        np.array([1.2, 1.4, 1.1, 1.3, 1.5]),
        np.array([-0.4, 0.2, -0.6, 0.3, -0.1]),
        np.full(5, 0.2),
        n=400,
        seed=77,
    )
    spec_3pl = (
        Model.uniform(ThreePL(), items=df_3pl.columns)
        .fix(item="q01", u=0.15)
        .prior(param="g", dist=BetaPrior(2.0, 2.0))
    )
    t_3pl = spec_3pl.parameter_table(df_3pl).set_index(["item", "param"])
    assert 0.0 < float(t_3pl["value"].loc[("q01", "g")]) < 0.15

    spec_4pl = (
        Model.uniform(FourPL(), items=df_3pl.columns)
        .fix(item="q01", g=0.96)
        .prior(param="g", dist=BetaPrior(2.0, 2.0))
        .prior(param="u", dist=BetaPrior(2.0, 2.0))
    )
    t_4pl = spec_4pl.parameter_table(df_3pl).set_index(["item", "param"])
    for item in df_3pl.columns:
        g0 = float(t_4pl["value"].loc[(item, "g")])
        u0 = float(t_4pl["value"].loc[(item, "u")])
        assert 0.0 < g0 < u0 < 1.0

    # 4. GRM keeps d1 > d2 > ... > d(K-1) strictly inside prior support
    #    (LogNormalPrior, BetaPrior, fixed).
    df_grm = _simulate_grm(
        np.array([1.1, 1.3, 1.0, 1.2]),
        np.array([[1.0, -0.2, -1.2], [0.8, 0.0, -1.0], [1.2, 0.1, -0.9], [0.6, -0.3, -1.4]]),
        n=400,
        seed=88,
    )
    spec_grm_ln = (
        Model.uniform(GRM(n_categories=4), items=df_grm.columns)
        .prior(param="d1", dist=LogNormalPrior(0.0, 0.5))
        .prior(param="d2", dist=LogNormalPrior(0.0, 0.5))
        .prior(param="d3", dist=LogNormalPrior(0.0, 0.5))
    )
    t_grm_ln = spec_grm_ln.parameter_table(df_grm).set_index(["item", "param"])
    for item in df_grm.columns:
        ds = [float(t_grm_ln["value"].loc[(item, f"d{k}")]) for k in (1, 2, 3)]
        assert ds[0] > ds[1] > ds[2] > 0.0
    res_grm_ln = fit(spec_grm_ln, df_grm, method=BAEM(tol=1e-6), se=None)
    assert res_grm_ln.converged
    assert np.isfinite(res_grm_ln.log_posterior)
    assert (res_grm_ln.items["d1"] > res_grm_ln.items["d2"]).all()
    assert (res_grm_ln.items["d2"] > res_grm_ln.items["d3"]).all()
    assert (res_grm_ln.items["d3"] > 0.0).all()

    spec_grm_beta = (
        Model.uniform(GRM(n_categories=4), items=df_grm.columns)
        .fix(item="q01", d1=0.8)
        .prior(param="d1", dist=BetaPrior(2.0, 2.0))
        .prior(param="d2", dist=BetaPrior(2.0, 2.0))
        .prior(param="d3", dist=BetaPrior(2.0, 2.0))
    )
    t_grm_beta = spec_grm_beta.parameter_table(df_grm).set_index(["item", "param"])
    for item in df_grm.columns:
        ds = [float(t_grm_beta["value"].loc[(item, f"d{k}")]) for k in (1, 2, 3)]
        assert 1.0 > ds[0] > ds[1] > ds[2] > 0.0
    res_grm_beta = fit(spec_grm_beta, df_grm, method=BAEM(tol=1e-6), se=None)
    assert res_grm_beta.converged
    assert np.isfinite(res_grm_beta.log_posterior)
    assert ((res_grm_beta.items["d1"] < 1.0) & (res_grm_beta.items["d3"] > 0.0)).all()
    assert (res_grm_beta.items["d1"] > res_grm_beta.items["d2"]).all()
    assert (res_grm_beta.items["d2"] > res_grm_beta.items["d3"]).all()


def test_heywood_warning_on_prior_support_boundary() -> None:
    """Finite bounds induced by an attached prior's support also trigger HeywoodWarning."""
    df = _simulate_2pl(
        np.array([2.0, 1.3, 1.1, 1.4, 1.2]),
        np.array([-1.8, 0.2, -0.3, 0.4, 0.1]),
        n=600,
        seed=314,
    )

    # Upper support boundary: BetaPrior(1, 1) on a (true a = 2.0) pushes q01.a -> 1.
    spec_upper = Model.uniform(TwoPL(), items=df.columns).prior(
        param="a", dist=BetaPrior(1.0, 1.0), items=["q01"]
    )
    with pytest.warns(HeywoodWarning, match=r"q01\.a"):
        res_upper = fit(spec_upper, df, method=BAEM(tol=1e-7), se=None)
    assert res_upper.converged
    assert float(res_upper.items.set_index("item")["a"].loc["q01"]) > 1.0 - 1e-4

    # Lower support boundary: BetaPrior(1, 1) on d (true d = -1.8) pushes q01.d -> 0.
    spec_lower = Model.uniform(TwoPL(), items=df.columns).prior(
        param="d", dist=BetaPrior(1.0, 1.0), items=["q01"]
    )
    with pytest.warns(HeywoodWarning, match=r"q01\.d"):
        res_lower = fit(spec_lower, df, method=BAEM(tol=1e-7), se=None)
    assert res_lower.converged
    assert float(res_lower.items.set_index("item")["d"].loc["q01"]) < 1e-4

    # Interior solution under BetaPrior(1, 1) does NOT warn.
    spec_interior = Model.uniform(TwoPL(), items=df.columns).prior(
        param="d", dist=BetaPrior(1.0, 1.0), items=["q04"]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", HeywoodWarning)
        res_interior = fit(spec_interior, df, method=BAEM(tol=1e-7), se=None)
    assert res_interior.converged
    d4 = float(res_interior.items.set_index("item")["d"].loc["q04"])
    assert 0.1 < d4 < 0.9
