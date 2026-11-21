"""Unit and property tests for confirmatory 2-3 dimensional MIRT (M4, DESIGN.md §4.9, §15)."""

from __future__ import annotations

import warnings
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from flexpyirt import (
    BAEM,
    GPCM,
    GRM,
    PCM,
    FourPL,
    LogNormal,
    Model,
    Normal,
    NormalPrior,
    Quadrature,
    Rasch,
    ThreePL,
    TwoPL,
    _core,
    fit,
)
from flexpyirt._compile import compile_problem, prepare
from flexpyirt.errors import (
    ConvergenceWarning,
    HeywoodWarning,
    IdentificationError,
    SpecificationError,
)
from flexpyirt.items import DichotomousKernel, FourPLKernel, GPCMKernel, GradedKernel
from flexpyirt.latent import correlation_matrix_is_pd, latent_param_names, mvn_log_weights
from tests._reference import load as load_m1


def _cell(frame: pd.DataFrame, row: Any, col: str) -> float:
    return float(frame[col].loc[row])


def _sim_mirt_2pl(
    a: np.ndarray,
    d: np.ndarray,
    cov: np.ndarray,
    n: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    theta = rng.multivariate_normal(np.zeros(cov.shape[0]), cov, size=n)
    p = 1.0 / (1.0 + np.exp(-(theta @ a.T + d[None, :])))
    y = (rng.random(p.shape) < p).astype(float)
    return pd.DataFrame(y, columns=[f"q{j:02d}" for j in range(1, a.shape[0] + 1)])


def _sim_mirt_grm(
    a: np.ndarray,
    d: np.ndarray,
    cov: np.ndarray,
    n: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    theta = rng.multivariate_normal(np.zeros(cov.shape[0]), cov, size=n)
    lin = theta @ a.T  # (n, J)
    cum = 1.0 / (1.0 + np.exp(-(lin[:, :, None] + d[None, :, :])))  # (n, J, K-1)
    u = rng.random((n, a.shape[0], 1))
    y = (u < cum).sum(axis=-1).astype(float)
    return pd.DataFrame(y, columns=[f"g{j:02d}" for j in range(1, a.shape[0] + 1)])


def _reference_oakes(prob: _core.CompiledProblem, eta: jax.Array) -> jax.Array:
    def q(e: jax.Array, e_prime: jax.Array) -> jax.Array:
        counts, _ = _core.e_step(prob, e_prime)
        return _core.expected_complete_loglik(prob, e, counts) + _core.log_prior(prob, e)

    h_own, h_cross = jax.jacfwd(jax.grad(q, argnums=0), argnums=(0, 1))(eta, eta)
    info: jax.Array = -(h_own + h_cross)
    return info


# --------------------------------------------------------------------------------------------
# Quadrature product grids
# --------------------------------------------------------------------------------------------


def test_quadrature_defaults_and_product_grid_shapes() -> None:
    assert Quadrature.default(1) == Quadrature.equispaced(61, lim=(-6.0, 6.0))
    assert Quadrature.default(2) == Quadrature.equispaced(31, lim=(-6.0, 6.0))
    assert Quadrature.default(3) == Quadrature.equispaced(15, lim=(-6.0, 6.0))
    with pytest.raises(SpecificationError, match="1 to 3"):
        Quadrature.default(4)
    with pytest.raises(SpecificationError, match="1 to 3"):
        Quadrature.default(0)

    q = Quadrature.equispaced(5, lim=(-2.0, 2.0))
    assert q.nodes().shape == (5,)
    g1 = q.grid(1)
    g2 = q.grid(2)
    g3 = q.grid(3)
    assert g1.shape == (5, 1)
    assert g2.shape == (25, 2)
    assert g3.shape == (125, 3)
    np.testing.assert_allclose(g1[:, 0], q.nodes())
    assert g2.min() == -2.0 and g2.max() == 2.0
    assert g3.min() == -2.0 and g3.max() == 2.0
    with pytest.raises(SpecificationError, match="1 to 3"):
        q.grid(4)


# --------------------------------------------------------------------------------------------
# Multidimensional kernels
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n_factors", [2, 3])
def test_multidimensional_kernels_sum_to_one_and_differentiate(n_factors: int) -> None:
    rng = np.random.default_rng(100 + n_factors)
    theta = jnp.asarray(rng.normal(size=(17, n_factors)), dtype=jnp.float64)
    slopes = np.linspace(0.5, 1.5, n_factors)

    # Dichotomous (2PL / Rasch)
    k_2pl = DichotomousKernel(n_factors=n_factors)
    assert k_2pl.param_names == (*[f"a{m}" for m in range(1, n_factors + 1)], "d")
    assert k_2pl.irt_names == ()
    p_2pl = jnp.asarray([*slopes, -0.3])
    lp_2pl = k_2pl.log_prob(theta, p_2pl)
    assert lp_2pl.shape == (17, 2)
    np.testing.assert_allclose(jnp.exp(lp_2pl).sum(axis=1), 1.0, rtol=1e-12)
    z_ref = np.asarray(theta) @ slopes - 0.3
    np.testing.assert_allclose(jnp.exp(lp_2pl[:, 1]), 1.0 / (1.0 + np.exp(-z_ref)), rtol=1e-12)
    assert np.isfinite(jax.grad(lambda p: k_2pl.log_prob(theta, p).sum())(p_2pl)).all()

    # FourPL (3PL / 4PL)
    k_4pl = FourPLKernel(n_factors=n_factors)
    assert k_4pl.param_names == (*[f"a{m}" for m in range(1, n_factors + 1)], "d", "g", "u")
    p_4pl = jnp.asarray([*slopes, 0.2, 0.15, 0.92])
    lp_4pl = k_4pl.log_prob(theta, p_4pl)
    assert lp_4pl.shape == (17, 2)
    np.testing.assert_allclose(jnp.exp(lp_4pl).sum(axis=1), 1.0, rtol=1e-12)
    # Fixed g=0, u=1 matches 2PL identically
    lp_4pl_2pl = k_4pl.log_prob(theta, jnp.asarray([*slopes, -0.3, 0.0, 1.0]))
    np.testing.assert_allclose(lp_4pl_2pl, lp_2pl, rtol=1e-13)
    assert np.isfinite(jax.grad(lambda p: k_4pl.log_prob(theta, p).sum())(p_4pl)).all()

    # Graded (GRM)
    k_grm = GradedKernel(n_cat=4, n_factors=n_factors)
    assert k_grm.param_names == (*[f"a{m}" for m in range(1, n_factors + 1)], "d1", "d2", "d3")
    p_grm = jnp.asarray([*slopes, 1.2, 0.1, -1.1])
    lp_grm = k_grm.log_prob(theta, p_grm)
    assert lp_grm.shape == (17, 4)
    np.testing.assert_allclose(jnp.exp(lp_grm).sum(axis=1), 1.0, rtol=1e-12)
    assert np.isfinite(jax.grad(lambda p: k_grm.log_prob(theta, p).sum())(p_grm)).all()
    assert np.isnan(k_grm.log_prob(theta, jnp.asarray([*slopes, -1.0, 0.5, -2.0]))).any()

    # GPCM / PCM
    k_gpcm = GPCMKernel(n_cat=4, n_factors=n_factors)
    assert k_gpcm.param_names == (*[f"a{m}" for m in range(1, n_factors + 1)], "d1", "d2", "d3")
    p_gpcm = jnp.asarray([*slopes, 0.8, -0.2, -0.9])
    lp_gpcm = k_gpcm.log_prob(theta, p_gpcm)
    assert lp_gpcm.shape == (17, 4)
    np.testing.assert_allclose(jnp.exp(lp_gpcm).sum(axis=1), 1.0, rtol=1e-12)
    assert np.isfinite(jax.grad(lambda p: k_gpcm.log_prob(theta, p).sum())(p_gpcm)).all()


# --------------------------------------------------------------------------------------------
# MVN quadrature weights
# --------------------------------------------------------------------------------------------


def test_mvn_log_weights_match_scipy_2d_and_3d() -> None:
    q2 = Quadrature.equispaced(11, lim=(-4.0, 4.0)).grid(2)
    params2 = jnp.array([0.3, -0.2, 1.4, 0.8, 0.55])
    cov2 = np.array([[1.4, 0.55 * np.sqrt(1.4 * 0.8)], [0.55 * np.sqrt(1.4 * 0.8), 0.8]])
    ref2 = np.asarray(multivariate_normal.logpdf(q2, mean=[0.3, -0.2], cov=cov2), dtype=float)
    ref2 -= float(logsumexp(ref2))
    got2 = np.asarray(mvn_log_weights(jnp.asarray(q2), params2))
    np.testing.assert_allclose(got2, ref2, rtol=1e-12, atol=1e-13)
    assert float(np.exp(got2).sum()) == pytest.approx(1.0, rel=1e-13)
    assert np.isfinite(jax.jacfwd(lambda p: mvn_log_weights(jnp.asarray(q2), p))(params2)).all()

    q3 = Quadrature.equispaced(7, lim=(-3.0, 3.0)).grid(3)
    params3 = jnp.array([0.1, -0.2, 0.3, 1.2, 0.9, 1.5, 0.4, -0.2, 0.5])
    sds3 = np.sqrt([1.2, 0.9, 1.5])
    r3 = np.array([[1.0, 0.4, -0.2], [0.4, 1.0, 0.5], [-0.2, 0.5, 1.0]])
    cov3 = np.outer(sds3, sds3) * r3
    ref3 = np.asarray(multivariate_normal.logpdf(q3, mean=[0.1, -0.2, 0.3], cov=cov3), dtype=float)
    ref3 -= float(logsumexp(ref3))
    got3 = np.asarray(mvn_log_weights(jnp.asarray(q3), params3))
    np.testing.assert_allclose(got3, ref3, rtol=1e-12, atol=1e-13)
    assert float(np.exp(got3).sum()) == pytest.approx(1.0, rel=1e-13)
    assert np.isfinite(jax.jacfwd(lambda p: mvn_log_weights(jnp.asarray(q3), p))(params3)).all()


def test_mvn_log_weights_rejects_non_pd_inputs() -> None:
    q2 = jnp.asarray(Quadrature.equispaced(5, lim=(-2.0, 2.0)).grid(2))
    assert np.isnan(mvn_log_weights(q2, jnp.array([0.0, 0.0, 1.0, 1.0, 1.0]))).all()
    assert np.isnan(mvn_log_weights(q2, jnp.array([0.0, 0.0, 1.0, 1.0, -1.2]))).all()
    assert np.isnan(mvn_log_weights(q2, jnp.array([0.0, 0.0, -0.1, 1.0, 0.2]))).all()

    q3 = jnp.asarray(Quadrature.equispaced(5, lim=(-2.0, 2.0)).grid(3))
    # r12=0.8, r13=0.8, r23=-0.5 has det < 0
    bad3 = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.8, 0.8, -0.5])
    assert not correlation_matrix_is_pd([0.8, 0.8, -0.5], 3)
    assert np.isnan(mvn_log_weights(q3, bad3)).all()
    assert latent_param_names(1) == ("mean", "var")
    assert latent_param_names(2) == ("mean_1", "mean_2", "var_1", "var_2", "cor_12")
    assert latent_param_names(3) == (
        "mean_1",
        "mean_2",
        "mean_3",
        "var_1",
        "var_2",
        "var_3",
        "cor_12",
        "cor_13",
        "cor_23",
    )


# --------------------------------------------------------------------------------------------
# D = 1 preservation
# --------------------------------------------------------------------------------------------


def test_unidimensional_model_stays_unchanged() -> None:
    c = load_m1("v1_lsat7_2pl")
    spec = Model.uniform(TwoPL(), items=c.items)
    assert not spec.is_multidimensional
    assert spec.n_factors == 1
    assert spec.factor_names == ()
    res = fit(spec, c.data, method=BAEM(tol=1e-9))
    assert res.converged
    assert list(res.latent.columns) == ["group", "mean", "var", "mean_se", "var_se"]
    assert "a" in res.items.columns and "b" in res.items.columns and "b_se" in res.items.columns
    sc = res.score(c.data.iloc[:5])
    assert list(sc.columns) == ["theta", "se"]


# --------------------------------------------------------------------------------------------
# 2D and 3D estimation, monotonicity, Oakes == Hessian, scoring, and results
# --------------------------------------------------------------------------------------------


def test_2d_2pl_monotone_oakes_equals_hessian_and_eap() -> None:
    a = np.array(
        [
            [1.3, 0.0],
            [1.1, 0.0],
            [1.4, 0.0],
            [1.0, 0.0],
            [0.8, 0.9],  # cross-loading
            [0.0, 1.2],
            [0.0, 1.5],
            [0.0, 1.0],
            [0.0, 1.3],
        ]
    )
    d = np.linspace(-0.8, 0.8, a.shape[0])
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    df = _sim_mirt_2pl(a, d, cov, n=800, seed=101)
    items = list(df.columns)
    spec = Model.uniform(TwoPL(), items=items).with_factors(
        {"math": items[:5], "verbal": items[4:]}
    )
    quad = Quadrature.equispaced(15, lim=(-5.0, 5.0))

    res_em = fit(spec, df, method=BAEM(tol=1e-9), quadrature=quad, se="oakes")
    res_hess = fit(spec, df, method=BAEM(tol=1e-9), quadrature=quad, se="hessian")
    res_sq = fit(spec, df, method=BAEM(tol=1e-9, squarem=True), quadrature=quad, se="oakes")
    assert res_em.converged and res_hess.converged and res_sq.converged
    assert res_em.vcov is not None and res_hess.vcov is not None and res_sq.vcov is not None

    # Monotonicity for both plain EM and SQUAREM
    for res in (res_em, res_sq):
        ll = res.history["loglik"].to_numpy()
        assert np.all(np.diff(ll) >= -1e-9 * np.abs(ll[1:]))
        assert res.loglik >= ll[-1] - 1e-9 * abs(ll[-1])

    # Agreement between Oakes and Hessian at the same iterate, and between EM and SQUAREM
    assert res_em.loglik == pytest.approx(res_sq.loglik, rel=1e-12)
    np.testing.assert_allclose(res_em.vcov, res_hess.vcov, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(res_em.vcov, res_sq.vcov, rtol=1e-5, atol=1e-8)

    prep = prepare(spec, df, group=None, item_names=None, table=None)
    comp = compile_problem(spec, prep.y, prep.group, prep.table, quad)
    eta = jnp.asarray(res_em._eta)
    info_oakes = np.asarray(jax.jit(_core.information_oakes)(comp.problem, eta))
    info_hess = np.asarray(jax.jit(_core.information_hessian)(comp.problem, eta))
    info_ref = np.asarray(jax.jit(_reference_oakes)(comp.problem, eta))
    scale = max(float(np.max(np.abs(info_ref))), 1.0)
    np.testing.assert_allclose(info_oakes, info_ref, rtol=1e-10, atol=1e-12 * scale)
    np.testing.assert_allclose(info_oakes, info_hess, rtol=1e-10, atol=1e-12 * scale)

    # E-step count conservation
    counts, ll_e = _core.e_step(comp.problem, eta)
    assert float(counts.n.sum()) == pytest.approx(len(df))
    assert float(ll_e) == pytest.approx(res_em.loglik, rel=1e-12)

    # Results schema
    assert list(res_em.items.columns) == [
        "item",
        "model",
        "a1",
        "a2",
        "d",
        "a1_se",
        "a2_se",
        "d_se",
    ]
    assert list(res_em.latent.columns) == [
        "group",
        "mean_1",
        "mean_2",
        "var_1",
        "var_2",
        "cor_12",
        "mean_1_se",
        "mean_2_se",
        "var_1_se",
        "var_2_se",
        "cor_12_se",
    ]
    # Unloaded slopes are 0 with NaN SE; cross-loading item q05 has both free
    it = res_em.items.set_index("item")
    assert _cell(it, "q01", "a2") == 0.0 and np.isnan(_cell(it, "q01", "a2_se"))
    assert _cell(it, "q09", "a1") == 0.0 and np.isnan(_cell(it, "q09", "a1_se"))
    assert np.isfinite(_cell(it, "q05", "a1_se")) and np.isfinite(_cell(it, "q05", "a2_se"))
    # Correlation is estimated positively near 0.5
    cor_hat = _cell(res_em.latent, 0, "cor_12")
    assert 0.3 < cor_hat < 0.7
    assert np.isfinite(_cell(res_em.latent, 0, "cor_12_se"))

    # EAP scoring returns factor-named columns, preserves index, handles missing data
    sub = df.iloc[:6].copy()
    sub.index = pd.Index([f"p{i}" for i in range(6)])
    sub.iloc[0, 0] = np.nan
    sub.iloc[1, :] = np.nan  # all-missing returns prior mean 0 and grid SD ~1
    scores = res_em.score(sub, method="EAP")
    assert list(scores.columns) == ["theta_math", "theta_verbal", "se_math", "se_verbal"]
    assert list(scores.index) == list(sub.index)
    assert np.isfinite(scores.to_numpy()).all()
    assert abs(_cell(scores, "p1", "theta_math")) < 1e-12
    assert abs(_cell(scores, "p1", "theta_verbal")) < 1e-12
    assert 0.99 < _cell(scores, "p1", "se_math") < 1.01

    # ML and WLE scoring raise SpecificationError for D > 1 (MAP is supported, §15.39)
    for m in ("ML", "WLE"):
        with pytest.raises(SpecificationError, match="unidimensional"):
            res_em.score(sub, method=m)
    with pytest.raises(SpecificationError, match="unidimensional"):
        res_em.score(sub, method="PV", key=jax.random.key(0))

    # Summary & HTML rendering
    txt = res_em.summary()
    assert "Factors:" in txt and "math, verbal" in txt and "15^2 = 225" in txt
    html_out = res_em._repr_html_()
    assert "Factors" in html_out and "math, verbal" in html_out and "cor_12" in html_out


def test_2d_orthogonal_rasch_grm_mixed_priors_and_fix() -> None:
    # Orthogonal 2D Rasch: variances free, correlation fixed at 0
    a_r = np.array([[1.0, 0.0]] * 4 + [[0.0, 1.0]] * 4)
    d_r = np.linspace(-0.7, 0.7, 8)
    cov_r = np.array([[1.4, 0.0], [0.0, 0.8]])
    df_r = _sim_mirt_2pl(a_r, d_r, cov_r, n=600, seed=102)
    items_r = list(df_r.columns)
    spec_r = Model.uniform(Rasch(), items=items_r).with_factors(
        {"F1": items_r[:4], "F2": items_r[4:]}, correlated=False
    )
    res_r = fit(
        spec_r,
        df_r,
        method=BAEM(tol=1e-7, squarem=True),
        quadrature=Quadrature.equispaced(15, lim=(-6.0, 6.0)),
        se="oakes",
    )
    assert res_r.converged
    assert _cell(res_r.latent, 0, "cor_12") == 0.0
    assert np.isnan(_cell(res_r.latent, 0, "cor_12_se"))
    assert np.isfinite(_cell(res_r.latent, 0, "var_1_se"))
    assert np.isfinite(_cell(res_r.latent, 0, "var_2_se"))
    assert _cell(res_r.latent, 0, "var_1") > _cell(res_r.latent, 0, "var_2")

    # 2D GRM + priors + fix_from (slope-intercept)
    a_g = np.array([[1.2, 0.0], [1.0, 0.0], [1.3, 0.0], [0.0, 1.1], [0.0, 1.4], [0.0, 1.0]])
    d_g = np.tile(np.array([1.0, 0.0, -1.0]), (6, 1))
    df_g = _sim_mirt_grm(a_g, d_g, np.array([[1.0, 0.4], [0.4, 1.0]]), n=600, seed=103)
    items_g = list(df_g.columns)
    spec_g = (
        Model.uniform(GRM(4), items=items_g)
        .with_factors({"F1": items_g[:3], "F2": items_g[3:]})
        .prior(param="a1", dist=LogNormal(0.1, 0.4))
        .prior(param="d1", dist=NormalPrior(0.0, 2.0))
        .fix(item="g01", a1=1.2)
    )
    res_g = fit(
        spec_g,
        df_g,
        method=BAEM(tol=1e-6, squarem=True),
        quadrature=Quadrature.equispaced(11, lim=(-5.0, 5.0)),
        se="oakes",
    )
    assert res_g.converged
    assert res_g.log_posterior < res_g.loglik  # active priors contribute negative log density
    it_g = res_g.items.set_index("item")
    assert _cell(it_g, "g01", "a1") == 1.2
    assert np.isnan(_cell(it_g, "g01", "a1_se"))

    # fix_from with slope-intercept works; parameterization='irt' raises SpecificationError
    bank = pd.DataFrame({"item": ["g01"], "a1": [1.2], "d1": [1.0], "d2": [0.0], "d3": [-1.0]})
    spec_fix = Model.uniform(GRM(4), items=items_g).with_factors(
        {"F1": items_g[:3], "F2": items_g[3:]}
    )
    spec_fipc = spec_fix.fix_from(bank)
    tbl = spec_fipc.parameter_table(df_g)
    g01 = tbl[(tbl["item"] == "g01")].set_index("param")
    assert not bool(g01["free"].loc["a1"]) and _cell(g01, "a1", "value") == 1.2
    assert not bool(g01["free"].loc["a2"]) and _cell(g01, "a2", "value") == 0.0
    with pytest.raises(SpecificationError, match="unidimensional"):
        spec_fix.fix_from(bank, parameterization="irt")


def test_2d_mixed_item_families() -> None:
    rng = np.random.default_rng(104)
    n = 500
    df = pd.DataFrame(
        {
            "m_2pl": rng.integers(0, 2, n).astype(float),
            "m_rasch": rng.integers(0, 2, n).astype(float),
            "m_3pl": rng.integers(0, 2, n).astype(float),
            "m_4pl": rng.integers(0, 2, n).astype(float),
            "m_grm": rng.integers(0, 3, n).astype(float),
            "m_gpcm": rng.integers(0, 3, n).astype(float),
            "m_pcm": rng.integers(0, 3, n).astype(float),
        }
    )
    spec = Model.from_items(
        {
            "m_2pl": TwoPL(),
            "m_rasch": Rasch(),
            "m_3pl": ThreePL(),
            "m_4pl": FourPL(),
            "m_grm": GRM(3),
            "m_gpcm": GPCM(3),
            "m_pcm": PCM(3),
        }
    ).with_factors(
        {"F1": ["m_2pl", "m_rasch", "m_3pl", "m_grm"], "F2": ["m_4pl", "m_gpcm", "m_pcm"]}
    )
    quad = Quadrature.equispaced(9, lim=(-4.0, 4.0))
    prep = prepare(spec, df, group=None, item_names=None, table=None)
    comp = compile_problem(spec, prep.y, prep.group, prep.table, quad)
    eta = jnp.asarray(comp.eta0)
    info_o = np.asarray(jax.jit(_core.information_oakes)(comp.problem, eta))
    info_h = np.asarray(jax.jit(_core.information_hessian)(comp.problem, eta))
    scale = max(float(np.max(np.abs(info_h))), 1.0)
    np.testing.assert_allclose(info_o, info_h, rtol=1e-9, atol=1e-11 * scale)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", HeywoodWarning)
        warnings.simplefilter("ignore", ConvergenceWarning)
        res = fit(spec, df, method=BAEM(max_iter=5), quadrature=quad, se=None)
    ll = res.history["loglik"].to_numpy()
    assert np.all(np.diff(ll) >= -1e-9 * np.abs(ll[1:]))


def test_3d_2pl_oakes_equals_hessian_and_monotone() -> None:
    a = np.array(
        [
            [1.3, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [1.4, 0.0, 0.0],
            [0.0, 1.2, 0.0],
            [0.0, 1.4, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.1],
            [0.0, 0.0, 1.5],
            [0.0, 0.0, 1.2],
        ]
    )
    d = np.linspace(-0.7, 0.7, 9)
    cov = np.array([[1.0, 0.4, 0.2], [0.4, 1.0, 0.5], [0.2, 0.5, 1.0]])
    df = _sim_mirt_2pl(a, d, cov, n=600, seed=105)
    items = list(df.columns)
    spec = Model.uniform(TwoPL(), items=items).with_factors(
        {"F1": items[:3], "F2": items[3:6], "F3": items[6:]}
    )
    quad = Quadrature.equispaced(7, lim=(-4.0, 4.0))
    res = fit(spec, df, method=BAEM(tol=1e-7, squarem=True), quadrature=quad, se="oakes")
    assert res.converged
    ll = res.history["loglik"].to_numpy()
    assert np.all(np.diff(ll) >= -1e-9 * np.abs(ll[1:]))
    assert res.loglik >= ll[-1] - 1e-9 * abs(ll[-1])

    prep = prepare(spec, df, group=None, item_names=None, table=None)
    comp = compile_problem(spec, prep.y, prep.group, prep.table, quad)
    eta = jnp.asarray(res._eta)
    info_o = np.asarray(jax.jit(_core.information_oakes)(comp.problem, eta))
    info_h = np.asarray(jax.jit(_core.information_hessian)(comp.problem, eta))
    scale = max(float(np.max(np.abs(info_h))), 1.0)
    np.testing.assert_allclose(info_o, info_h, rtol=1e-9, atol=1e-11 * scale)
    assert list(res.latent.columns) == [
        "group",
        "mean_1",
        "mean_2",
        "mean_3",
        "var_1",
        "var_2",
        "var_3",
        "cor_12",
        "cor_13",
        "cor_23",
        "mean_1_se",
        "mean_2_se",
        "mean_3_se",
        "var_1_se",
        "var_2_se",
        "var_3_se",
        "cor_12_se",
        "cor_13_se",
        "cor_23_se",
    ]
    sc = res.score(df.iloc[:4])
    assert list(sc.columns) == ["theta_F1", "theta_F2", "theta_F3", "se_F1", "se_F2", "se_F3"]


# --------------------------------------------------------------------------------------------
# SpecificationError and IdentificationError checks
# --------------------------------------------------------------------------------------------


def test_mirt_specification_errors() -> None:
    items = [f"q{i}" for i in range(1, 7)]
    base = Model.uniform(TwoPL(), items=items)

    with pytest.raises(SpecificationError, match="2 to 3"):
        base.with_factors({"F1": items})
    with pytest.raises(SpecificationError, match="2 to 3"):
        base.with_factors({"F1": items[:2], "F2": items[2:3], "F3": items[3:4], "F4": items[4:]})
    with pytest.raises(SpecificationError, match="non-empty"):
        base.with_factors({"": items[:3], "F2": items[3:]})
    with pytest.raises(SpecificationError, match="single string"):
        base.with_factors({"F1": "q1", "F2": items[1:]})
    with pytest.raises(SpecificationError, match="at least one item"):
        base.with_factors({"F1": [], "F2": items})
    with pytest.raises(SpecificationError, match="duplicated items"):
        base.with_factors({"F1": ["q1", "q1", "q2"], "F2": items[2:]})
    with pytest.raises(SpecificationError, match="not assigned"):
        base.with_factors({"F1": items[:2], "F2": items[3:]})
    with pytest.raises(SpecificationError, match="Unknown items"):
        base.with_factors({"F1": ["q1", "nope"], "F2": items[1:]})

    m2 = base.with_factors({"F1": items[:3], "F2": items[3:]})
    with pytest.raises(SpecificationError, match="Multiple-group"):
        m2.with_groups(ref=Normal.standard(), focal=Normal.free())
    with pytest.raises(SpecificationError, match="Multiple-group"):
        base.with_groups(ref=Normal.standard(), focal=Normal.free()).with_factors(
            {"F1": items[:3], "F2": items[3:]}
        )
    with pytest.raises(SpecificationError, match="unidimensional"):
        m2.with_latent(Normal.standard())
    with pytest.raises(SpecificationError, match="unidimensional"):
        base.with_latent(Normal.standard()).with_factors({"F1": items[:3], "F2": items[3:]})
    with pytest.raises(SpecificationError, match="with_factors\\(\\) before fix"):
        base.fix(item="q1", a=1.0).with_factors({"F1": items[:3], "F2": items[3:]})
    with pytest.raises(SpecificationError, match="with_factors\\(\\) before prior"):
        base.prior(param="a", dist=LogNormal(0.0, 0.5)).with_factors(
            {"F1": items[:3], "F2": items[3:]}
        )
    with pytest.raises(SpecificationError, match="latent-distribution"):
        m2.prior(param="cor_12", dist=NormalPrior(0.0, 0.5))


def test_mirt_table_validation_and_identification_errors() -> None:
    rng = np.random.default_rng(106)
    items = [f"q{i}" for i in range(1, 7)]
    df = pd.DataFrame(rng.integers(0, 2, (120, 6)).astype(float), columns=items)
    base = Model.uniform(TwoPL(), items=items)

    # Unrestricted exploratory loading pattern (all items load on both F1 and F2) -> unidentified
    spec_exp_cor = base.with_factors({"F1": items, "F2": items}, correlated=True)
    with pytest.raises(IdentificationError, match="rotational"):
        fit(spec_exp_cor, df)
    spec_exp_orth = base.with_factors({"F1": items, "F2": items}, correlated=False)
    with pytest.raises(IdentificationError, match="identical free loading"):
        fit(spec_exp_orth, df)

    spec = base.with_factors({"F1": items[:3], "F2": items[3:]})
    tbl = spec.parameter_table(df)

    # Freeing var_1 without fixed nonzero a1 -> IdentificationError
    t_var = tbl.copy()
    t_var.loc[(t_var["item"] == "__latent__") & (t_var["param"] == "var_1"), "free"] = True
    with pytest.raises(IdentificationError, match="var_1"):
        fit(spec, df, table=t_var)

    # Freeing mean_1 without fixed intercept -> IdentificationError
    t_mean = tbl.copy()
    t_mean.loc[(t_mean["item"] == "__latent__") & (t_mean["param"] == "mean_1"), "free"] = True
    with pytest.raises(IdentificationError, match="mean_1"):
        fit(spec, df, table=t_mean)

    # Non-PD correlation (cor_12 = 1.0 or -1.0) -> SpecificationError
    t_cor = tbl.copy()
    t_cor.loc[(t_cor["item"] == "__latent__") & (t_cor["param"] == "cor_12"), "value"] = 1.0
    with pytest.raises(SpecificationError, match="strictly inside"):
        fit(spec, df, table=t_cor)

    # All starting slopes on F2 set to 0 -> SpecificationError
    t_zero = tbl.copy()
    t_zero.loc[t_zero["param"] == "a2", "value"] = 0.0
    with pytest.raises(SpecificationError, match="starting slopes equal to 0"):
        fit(spec, df, table=t_zero)

    # 3D non-PD correlation triple -> SpecificationError
    items3 = [f"q{i}" for i in range(1, 7)]
    spec3 = base.with_factors({"F1": items3[:2], "F2": items3[2:4], "F3": items3[4:]})
    t3 = spec3.parameter_table(df)
    t3.loc[t3["param"] == "cor_12", "value"] = 0.8
    t3.loc[t3["param"] == "cor_13", "value"] = 0.8
    t3.loc[t3["param"] == "cor_23", "value"] = -0.5
    with pytest.raises(SpecificationError, match="positive definite"):
        fit(spec3, df, table=t3)


# ---- identification regressions (review findings) ----------------------------------------


def _binary_df(n_items: int, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=(600, 1))
    y = (rng.random((600, n_items)) < 1.0 / (1.0 + np.exp(-(theta - 0.2)))).astype(float)
    return pd.DataFrame(y, columns=[f"q{j}" for j in range(1, n_items + 1)])


def test_all_loadings_fixed_at_zero_is_identification_error() -> None:
    df = _binary_df(4)
    spec = (
        Model.uniform(TwoPL(), items=df.columns)
        .with_factors({"F1": ["q1", "q2"], "F2": ["q3", "q4"]})
        .fix(item="q1", a1=0.0)
        .fix(item="q2", a1=0.0)
    )
    with pytest.raises(IdentificationError, match="no active loadings"):
        fit(spec, df)


def test_proportional_fixed_loadings_are_identification_error() -> None:
    df = _binary_df(6)
    items = list(df.columns)
    spec = Model.uniform(Rasch(), items=items).with_factors({"F1": items, "F2": items})
    with pytest.raises(IdentificationError, match="proportional fixed loadings"):
        fit(spec, df)


def test_to_json_refuses_multidimensional_results() -> None:
    """The JSON schema does not store the factor structure yet (DESIGN §15.33)."""
    rng = np.random.default_rng(5)
    theta = rng.normal(size=(500, 2))
    a = np.array([[1.2, 0.0], [1.0, 0.0], [0.8, 0.0], [0.0, 1.1], [0.0, 0.9], [0.0, 1.3]])
    d = rng.normal(size=6)
    y = (rng.random((500, 6)) < 1.0 / (1.0 + np.exp(-(theta @ a.T + d)))).astype(float)
    df = pd.DataFrame(y, columns=[f"q{j}" for j in range(6)])
    spec = Model.uniform(TwoPL(), items=df.columns).with_factors(
        {"F1": ["q0", "q1", "q2"], "F2": ["q3", "q4", "q5"]}
    )
    res = fit(spec, df, se=None)
    with pytest.raises(SpecificationError, match="multidimensional"):
        res.to_json()
