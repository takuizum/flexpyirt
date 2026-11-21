"""Unit tests for EmpiricalHistogram latent distributions (DESIGN.md §4.4, §4.5, §15.29–§15.33)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from flexpyirt import (
    BAEM,
    GPCM,
    GRM,
    PCM,
    Beta,
    EmpiricalHistogram,
    FitResult,
    FourPL,
    LogNormal,
    Model,
    Normal,
    Quadrature,
    Rasch,
    ThreePL,
    TwoPL,
    _core,
    fit,
)
from flexpyirt._compile import compile_problem, eh_standardization_flags, prepare
from flexpyirt.errors import (
    FlexpyirtWarning,
    HeywoodWarning,
    IdentificationError,
    SpecificationError,
)
from flexpyirt.items import DichotomousKernel, FourPLKernel, GPCMKernel, GradedKernel, ItemKernel
from flexpyirt.latent import eh_moments, eh_project_floor, eh_regrid_cdf


def _simulate_2pl(
    a: np.ndarray, d: np.ndarray, theta: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    p = 1.0 / (1.0 + np.exp(-(np.outer(theta, a) + d)))
    out: np.ndarray = (rng.random(p.shape) < p).astype(float)
    return out


def _skewed_2pl_data(
    seed: int = 31415, n: int = 1200, j: int = 10
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.9, 1.8, j)
    d = np.linspace(-1.5, 1.5, j)
    raw = rng.gamma(shape=4.0, scale=1.0, size=n)
    theta = (raw - raw.mean()) / raw.std()
    df = pd.DataFrame(
        _simulate_2pl(a, d, theta, rng), columns=[f"q{k:02d}" for k in range(1, j + 1)]
    )
    return df, a, d


# --------------------------------------------------------------------------------------------
# Specification, validation, and identification
# --------------------------------------------------------------------------------------------


def test_eh_spec_validation() -> None:
    eh = EmpiricalHistogram()
    assert eh.min_weight == 1e-10
    assert EmpiricalHistogram(min_weight=1e-6).min_weight == 1e-6

    for bad in (0.0, -1e-5, 1.0, 1.5, float("nan"), float("inf"), True, "1e-6"):
        with pytest.raises(SpecificationError):
            EmpiricalHistogram(min_weight=bad)  # type: ignore[arg-type]

    # Keyword-only option
    with pytest.raises(TypeError):
        EmpiricalHistogram(1e-8)  # type: ignore[call-arg]


def test_eh_min_weight_exceeds_grid_raises() -> None:
    df, _, _ = _skewed_2pl_data(n=300, j=5)
    spec = Model.uniform(TwoPL(), items=df.columns).with_latent(EmpiricalHistogram(min_weight=0.1))
    with pytest.raises(SpecificationError, match="min_weight"):
        fit(spec, df, quadrature=Quadrature.equispaced(11, lim=(-4, 4)), se=None)


def test_mixing_eh_and_normal_across_groups_raises() -> None:
    items = ["q1", "q2", "q3"]
    base = Model.uniform(TwoPL(), items=items)
    with pytest.raises(SpecificationError, match="Cannot mix EmpiricalHistogram and Normal"):
        base.with_groups(ref=EmpiricalHistogram(), focal=Normal.free())
    with pytest.raises(SpecificationError, match="Cannot mix EmpiricalHistogram and Normal"):
        base.with_groups(ref=Normal.standard(), focal=EmpiricalHistogram())
    with pytest.raises(SpecificationError, match="Cannot mix EmpiricalHistogram and Normal"):
        Model(
            items=base.items,
            groups=(("ref", EmpiricalHistogram()), ("focal", Normal.standard())),
        )


def test_eh_multigroup_identification_and_anchor_warning() -> None:
    df, _, _ = _skewed_2pl_data(n=400, j=6)
    grp = np.repeat(["ref", "focal"], 200)
    base = Model.uniform(TwoPL(), items=df.columns).with_groups(
        ref=EmpiricalHistogram(), focal=EmpiricalHistogram()
    )
    # No anchors -> IdentificationError
    with pytest.raises(IdentificationError):
        fit(base, df, group=grp, se=None)
    # Only slope anchors (no intercept anchors) -> IdentificationError
    with pytest.raises(IdentificationError):
        fit(base.equal_across_groups(items=["q01", "q02"], params=["a"]), df, group=grp, se=None)
    # Only intercept anchors (no slope anchors) -> IdentificationError
    with pytest.raises(IdentificationError):
        fit(base.equal_across_groups(items=["q01", "q02"], params=["d"]), df, group=grp, se=None)
    # Single anchor -> FlexpyirtWarning
    with pytest.warns(FlexpyirtWarning, match="single anchor"):
        prepare(
            base.equal_across_groups(items=["q01"]),
            df,
            group=grp,
            item_names=None,
            table=None,
        )


# --------------------------------------------------------------------------------------------
# KKT floor projection, moments, and affine invariance
# --------------------------------------------------------------------------------------------


def test_eh_project_floor_exact_kkt() -> None:
    # No active nodes -> exact normalization
    counts = jnp.array([10.0, 20.0, 30.0, 40.0])
    w = np.asarray(eh_project_floor(counts, jnp.asarray(0.01)))
    np.testing.assert_allclose(w, [0.1, 0.2, 0.3, 0.4], atol=1e-15)

    # Active nodes (including cascading active node!) -> exact KKT solution
    counts2 = jnp.array([0.0, 5.0, 10.1, 42.4, 42.5])
    tau = 0.1
    w2 = np.asarray(eh_project_floor(counts2, jnp.asarray(tau)))
    assert np.isclose(w2.sum(), 1.0, atol=1e-14)
    assert np.all(w2 >= tau - 1e-15)
    # Verify objective sum n_q log w_q is >= any feasible perturbation
    obj_opt = float(np.sum(np.asarray(counts2) * np.log(w2)))
    rng = np.random.default_rng(7)
    for _ in range(50):
        cand = rng.dirichlet(np.ones(5))
        cand = tau + (1.0 - 5 * tau) * cand
        assert obj_opt >= float(np.sum(np.asarray(counts2) * np.log(cand))) - 1e-12


def test_eh_moments_analytic() -> None:
    # Known asymmetric discrete distribution
    nodes = np.array([-2.0, -1.0, 0.0, 1.0, 3.0])
    weights = np.array([0.1, 0.25, 0.35, 0.2, 0.1])
    mean, var, sd, skew, kurt = eh_moments(nodes, weights)
    exp_mean = float(np.sum(weights * nodes))
    c = nodes - exp_mean
    exp_var = float(np.sum(weights * c**2))
    exp_sd = float(np.sqrt(exp_var))
    exp_skew = float(np.sum(weights * c**3) / exp_sd**3)
    exp_kurt = float(np.sum(weights * c**4) / exp_var**2 - 3.0)
    assert mean == pytest.approx(exp_mean, abs=1e-14)
    assert var == pytest.approx(exp_var, abs=1e-14)
    assert sd == pytest.approx(exp_sd, abs=1e-14)
    assert skew == pytest.approx(exp_skew, abs=1e-14)
    assert kurt == pytest.approx(exp_kurt, abs=1e-14)

    # Normal density on fine grid has mean ~ 0, var ~ 1, skew ~ 0, excess kurtosis ~ 0
    g_nodes = np.linspace(-6.0, 6.0, 121)
    g_w = np.exp(-0.5 * g_nodes**2)
    g_w /= g_w.sum()
    m_n, v_n, sd_n, sk_n, ku_n = eh_moments(g_nodes, g_w)
    assert abs(m_n) < 1e-14
    assert v_n == pytest.approx(1.0, abs=1e-6)
    assert sd_n == pytest.approx(1.0, abs=1e-6)
    assert abs(sk_n) < 1e-14
    assert abs(ku_n) < 1e-5


def test_item_kernel_affine_transform_exact_invariance() -> None:
    """Every kernel's affine_transform satisfies P(y | theta; xi) == P(y | (theta-m)/s; xi')."""
    theta = jnp.linspace(-4.0, 4.0, 41)
    m = jnp.asarray(0.65)
    s = jnp.asarray(1.35)
    theta_star = (theta - m) / s

    cases: list[tuple[ItemKernel, jax.Array]] = [
        (DichotomousKernel(), jnp.array([1.4, -0.5])),
        (FourPLKernel(), jnp.array([1.6, 0.3, 0.18, 0.91])),
        (GradedKernel(4), jnp.array([1.2, 1.5, 0.1, -1.3])),
        (GPCMKernel(4), jnp.array([1.3, 0.8, -0.2, -1.4])),
    ]
    for kernel, params in cases:
        transformed = kernel.affine_transform(params, m, s)
        lp_orig = np.asarray(kernel.log_prob(theta, params))
        lp_star = np.asarray(kernel.log_prob(theta_star, transformed))
        np.testing.assert_allclose(lp_star, lp_orig, rtol=0.0, atol=1e-13)


def test_loglik_invariance_under_eh_standardization() -> None:
    """Marginal log-likelihood is invariant under affine standardization up to O(Delta^2) error."""
    df, _, _ = _skewed_2pl_data(seed=77, n=600, j=8)
    spec = Model.uniform(TwoPL(), items=df.columns).with_latent(EmpiricalHistogram())
    prep = prepare(spec, df, group=None, item_names=None, table=None)

    # Exact identity when already standardized (mean=0, var=1)
    quad61 = Quadrature.equispaced(61, lim=(-6, 6))
    comp61 = compile_problem(spec, prep.y, prep.group, prep.table, quad61)
    nodes61 = comp61.problem.nodes
    w_std = eh_regrid_cdf(
        nodes61,
        comp61.problem.latent.eh_weights[0],
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        jnp.asarray(1e-10),
    )
    np.testing.assert_allclose(
        np.asarray(w_std), np.asarray(comp61.problem.latent.eh_weights[0]), atol=1e-15
    )

    # Shifted/scaled weights: interpolation error shrinks as grid gets finer (O(Delta^2))
    errors = []
    for n_pts in (31, 61, 121):
        quad = Quadrature.equispaced(n_pts, lim=(-6, 6))
        nodes = quad.nodes()
        w_shift = np.exp(-0.5 * ((nodes - 0.5) / 1.25) ** 2)
        w_shift /= w_shift.sum()
        comp = compile_problem(
            spec, prep.y, prep.group, prep.table, quad, eh_weights=w_shift[None, :]
        )
        eta = jnp.asarray(comp.eta0)
        ll_before = float(_core.marginal_loglik(comp.problem, eta))
        eta_std, w_after = _core._eh_standardize(comp.problem, eta, comp.problem.latent.eh_weights)
        ll_after = float(
            _core.marginal_loglik(_core._with_eh_weights(comp.problem, w_after), eta_std)
        )
        errors.append(abs(ll_after - ll_before) / abs(ll_before))
    assert errors[1] < 2e-4  # < 2e-4 relative on default 61-point grid
    assert errors[2] < errors[1] < errors[0]  # shrinks O(Delta^2) (~4x per halving)


# --------------------------------------------------------------------------------------------
# Per-iteration standardization, FIPC, Rasch/PCM, monotonicity, SQUAREM
# --------------------------------------------------------------------------------------------


def test_reference_mean_zero_var_one_after_every_iteration() -> None:
    """Reference group EH satisfies mean=0, var=1 after EVERY EM iteration."""
    df, _, _ = _skewed_2pl_data(seed=101, n=800, j=8)
    spec = Model.uniform(TwoPL(), items=df.columns).with_latent(EmpiricalHistogram())
    quad = Quadrature.equispaced(61, lim=(-6, 6))
    prep = prepare(spec, df, group=None, item_names=None, table=None)
    comp = compile_problem(spec, prep.y, prep.group, prep.table, quad)
    assert comp.problem.latent.standardize_mean is True
    assert comp.problem.latent.standardize_var is True

    eta = jnp.asarray(comp.eta0)
    w = comp.problem.latent.eh_weights
    nodes = np.asarray(comp.problem.nodes)
    prev_ll = -np.inf
    for _ in range(15):
        prob_t = _core._with_eh_weights(comp.problem, w)
        counts, ll = _core.e_step(prob_t, eta)
        assert float(ll) >= prev_ll - 1e-9 * abs(float(ll))
        prev_ll = float(ll)
        eta, w = _core._m_step_eh(prob_t, eta, counts, n_steps=3)
        w_np = np.asarray(w[0])
        assert np.all(w_np >= 1e-10 - 1e-15)
        assert np.isclose(w_np.sum(), 1.0, atol=1e-12)
        mean, var, _, _, _ = eh_moments(nodes, w_np)
        assert abs(mean) < 1e-7
        assert abs(var - 1.0) < 1e-7


def test_fipc_does_not_standardize_and_recovers_shifted_moments() -> None:
    """Under FIPC, reference EH is NOT standardized and recovers non-zero mean & non-unit var."""
    rng = np.random.default_rng(202)
    j, n = 10, 2000
    a = rng.uniform(1.0, 1.8, j)
    d = np.linspace(-1.5, 1.5, j)
    raw = rng.gamma(shape=4.0, scale=1.0, size=n)
    theta = 0.7 + 1.3 * (raw - raw.mean()) / raw.std()
    items = [f"q{k:02d}" for k in range(1, j + 1)]
    df = pd.DataFrame(_simulate_2pl(a, d, theta, rng), columns=items)
    bank = pd.DataFrame({"item": items[:5], "a": a[:5], "d": d[:5]})
    spec = Model.uniform(TwoPL(), items=items).fix_from(bank).with_latent(EmpiricalHistogram())

    prep = prepare(spec, df, group=None, item_names=None, table=None)
    assert eh_standardization_flags(prep.table, spec) == (False, False)

    quad = Quadrature.equispaced(61, lim=(-6, 6))
    res = fit(spec, df, method=BAEM(tol=1e-4, max_iter=2000), quadrature=quad)
    assert res.converged
    # n_params: 5 free items * 2 + (Q - 1) = 10 + 60 = 70
    assert res.n_params == 10 + (quad.n_points - 1)
    lat = res.latent.iloc[0]
    assert abs(lat["mean"] - 0.7) < 0.15
    assert abs(lat["var"] - 1.69) < 0.40
    assert lat["skewness"] > 0.3  # skewed gamma recovered


def test_rasch_and_pcm_with_eh_standardize_mean_only() -> None:
    """With fixed slopes (Rasch/PCM), EH standardizes mean=0 only and leaves variance free."""
    rng = np.random.default_rng(303)
    j, n = 10, 1500
    d = np.linspace(-1.5, 1.5, j)
    theta = rng.normal(0.0, 1.4, n)  # true variance = 1.96 != 1
    items = [f"r{k:02d}" for k in range(1, j + 1)]
    df = pd.DataFrame(_simulate_2pl(np.ones(j), d, theta, rng), columns=items)
    spec = Model.uniform(Rasch(), items=items).with_latent(EmpiricalHistogram())

    prep = prepare(spec, df, group=None, item_names=None, table=None)
    assert eh_standardization_flags(prep.table, spec) == (True, False)

    quad = Quadrature.equispaced(41, lim=(-6, 6))
    res = fit(spec, df, method=BAEM(tol=1e-4, max_iter=2000), quadrature=quad)
    assert res.converged
    # n_params: J free intercepts + (Q - 2)
    assert res.n_params == j + (quad.n_points - 2)
    lat = res.latent.iloc[0]
    assert abs(lat["mean"]) < 1e-7
    assert abs(lat["var"] - 1.96) < 0.35  # free variance recovered, not forced to 1


def test_eh_monotone_and_squarem_matches_plain_em() -> None:
    """Both plain EM and logit-scale SQUAREM are monotone and converge to the same fixed point."""
    df, _, _ = _skewed_2pl_data(seed=404, n=1200, j=10)
    quad = Quadrature.equispaced(9, lim=(-3.5, 3.5))
    spec = Model.uniform(TwoPL(), items=df.columns).with_latent(EmpiricalHistogram(min_weight=1e-6))

    res_em = fit(spec, df, method=BAEM(tol=1e-7, max_iter=5000, squarem=False), quadrature=quad)
    res_sq = fit(spec, df, method=BAEM(tol=1e-7, max_iter=5000, squarem=True), quadrature=quad)
    assert res_em.converged
    assert res_sq.converged

    for res in (res_em, res_sq):
        ll = res.history["loglik"].to_numpy()
        assert np.all(np.diff(ll) >= -1e-8 * np.abs(ll[1:]))
        assert res.loglik >= ll[-1] - 1e-8 * abs(ll[-1])
        assert res.latent_weights is not None
        w = res.latent_weights["weight"].to_numpy()
        assert np.all(np.isfinite(np.log(w)))
        assert np.all(w >= 1e-6 - 1e-14)
        assert np.isclose(w.sum(), 1.0, atol=1e-12)

    assert abs(res_sq.loglik - res_em.loglik) <= 1e-6 * abs(res_em.loglik)
    np.testing.assert_allclose(res_sq.items["a"], res_em.items["a"], atol=1e-4)
    np.testing.assert_allclose(res_sq.items["d"], res_em.items["d"], atol=1e-4)
    assert res_sq.latent_weights is not None and res_em.latent_weights is not None
    np.testing.assert_allclose(
        res_sq.latent_weights["weight"], res_em.latent_weights["weight"], atol=1e-4
    )


def test_eh_n_params_aic_bic_and_oakes_equals_hessian() -> None:
    """Verify n_params, AIC/BIC, and exact Oakes == Hessian conditional SEs on 1G and 2G EH."""
    df, _, _ = _skewed_2pl_data(seed=505, n=1000, j=8)
    quad = Quadrature.equispaced(31, lim=(-5, 5))
    spec1 = Model.uniform(TwoPL(), items=df.columns).with_latent(EmpiricalHistogram())
    res1_o = fit(spec1, df, method=BAEM(tol=1e-4), quadrature=quad, se="oakes")
    res1_h = fit(spec1, df, method=BAEM(tol=1e-4), quadrature=quad, se="hessian")
    assert res1_o.n_params == 16 + (31 - 3)
    assert res1_o.aic == pytest.approx(-2.0 * res1_o.loglik + 2.0 * res1_o.n_params)
    assert res1_o.bic == pytest.approx(
        -2.0 * res1_o.loglik + res1_o.n_params * np.log(res1_o.n_obs)
    )
    assert res1_o.vcov is not None and res1_h.vcov is not None
    assert res1_o.vcov.shape == (16, 16)
    np.testing.assert_allclose(res1_o.vcov, res1_h.vcov, rtol=1e-8, atol=1e-12)
    assert res1_o.items["b_se"].notna().all()

    # Two-group EH with 4 anchors
    grp = np.repeat(["ref", "focal"], 500)
    spec2 = (
        Model.uniform(TwoPL(), items=df.columns)
        .with_groups(ref=EmpiricalHistogram(), focal=EmpiricalHistogram())
        .equal_across_groups(items=["q01", "q02", "q03", "q04"])
    )
    res2_o = fit(spec2, df, group=grp, method=BAEM(tol=1e-4), quadrature=quad, se="oakes")
    res2_h = fit(spec2, df, group=grp, method=BAEM(tol=1e-4), quadrature=quad, se="hessian")
    # Free item params: 4 anchors * 2 + 4 non-anchors * 2 groups * 2 = 24
    # EH params: (Q - 3) + (Q - 1) = 28 + 30 = 58
    assert res2_o.n_params == 24 + (31 - 3) + (31 - 1)
    assert res2_o.vcov is not None and res2_h.vcov is not None
    assert res2_o.vcov.shape == (24, 24)
    np.testing.assert_allclose(res2_o.vcov, res2_h.vcov, rtol=1e-8, atol=1e-12)
    ref_lat = res2_o.latent.set_index("group").loc["ref"]
    assert abs(float(ref_lat["mean"])) < 1e-7
    assert abs(float(ref_lat["var"]) - 1.0) < 1e-7


def test_eh_works_with_all_families_missing_and_priors() -> None:
    """EH works with mixed families (2PL, Rasch, 3PL, 4PL, GRM, GPCM, PCM), missing, and priors."""
    rng = np.random.default_rng(606)
    n = 800
    theta = rng.normal(0.0, 1.0, n)
    p_bin = 1.0 / (1.0 + np.exp(-np.outer(theta, [1.2, 1.0, 1.4, 1.5])))
    p_bin[:, 2] = 0.2 + 0.8 * p_bin[:, 2]
    p_bin[:, 3] = 0.15 + 0.7 * p_bin[:, 3]
    y_bin = (rng.random(p_bin.shape) < p_bin).astype(float)
    cuts = np.array([-0.9, 0.2, 1.1])
    y_poly = np.searchsorted(
        cuts, np.outer(theta, [1.2, 1.1, 1.0]) + rng.logistic(0, 1, (n, 3))
    ).astype(float)
    df = pd.DataFrame(
        np.hstack([y_bin, y_poly]),
        columns=["i_2pl", "i_rasch", "i_3pl", "i_4pl", "i_grm", "i_gpcm", "i_pcm"],
    )
    df[rng.random(df.shape) < 0.08] = np.nan

    spec = (
        Model.from_items(
            {
                "i_2pl": TwoPL(),
                "i_rasch": Rasch(),
                "i_3pl": ThreePL(),
                "i_4pl": FourPL(),
                "i_grm": GRM(4),
                "i_gpcm": GPCM(4),
                "i_pcm": PCM(4),
            }
        )
        .with_latent(EmpiricalHistogram())
        .prior(
            param="a",
            dist=LogNormal(0.2, 0.4),
            items=["i_2pl", "i_3pl", "i_4pl", "i_grm", "i_gpcm"],
        )
        .prior(param="g", dist=Beta(5.0, 17.0), items=["i_3pl", "i_4pl"])
        .prior(param="u", dist=Beta(17.0, 3.0), items=["i_4pl"])
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", HeywoodWarning)
        res = fit(
            spec,
            df,
            method=BAEM(tol=1e-4, max_iter=1500),
            quadrature=Quadrature.equispaced(25, lim=(-5, 5)),
            se="oakes",
        )
    assert res.converged
    assert res.log_posterior != res.loglik
    lp = res.history["log_posterior"].to_numpy()
    assert np.all(np.diff(lp) >= -1e-7 * np.abs(lp[1:]))
    assert res.vcov is not None and np.all(np.isfinite(res.vcov))


# --------------------------------------------------------------------------------------------
# Scoring, summary, HTML, and JSON round-trip
# --------------------------------------------------------------------------------------------


def test_eh_scoring_methods() -> None:
    """EAP, ML, WLE, PV work with EH (PV matches EAP); MAP raises SpecificationError."""
    df, _, _ = _skewed_2pl_data(seed=707, n=1000, j=8)
    quad = Quadrature.equispaced(41, lim=(-5, 5))
    spec = Model.uniform(TwoPL(), items=df.columns).with_latent(EmpiricalHistogram())
    res = fit(spec, df, method=BAEM(tol=1e-4), quadrature=quad, se=None)

    sub = df.iloc[:15]
    eap_s = res.score(sub, method="EAP")
    ml_s = res.score(sub, method="ML")
    wle_s = res.score(sub, method="WLE")
    assert list(eap_s.columns) == ["theta", "se"]
    assert np.all(np.isfinite(eap_s["theta"]))
    assert np.all(eap_s["se"] > 0)
    assert np.all(np.isfinite(wle_s["theta"]))
    assert ml_s.shape == (15, 2)

    # MAP raises SpecificationError on EH
    with pytest.raises(SpecificationError, match="MAP"):
        res.score(sub, method="MAP")

    # PV mean and SD match EAP and posterior SD for a single pattern
    key = jax.random.key(123)
    pv = res.score(sub.iloc[:3], method="PV", key=key, n_draws=4000)
    assert pv.shape == (3, 4000)
    pv_means = pv.mean(axis=1).to_numpy()
    pv_sds = pv.std(axis=1).to_numpy()
    np.testing.assert_allclose(pv_means, eap_s["theta"].iloc[:3].to_numpy(), atol=0.04)
    np.testing.assert_allclose(pv_sds, eap_s["se"].iloc[:3].to_numpy(), atol=0.04)


def test_eh_summary_html_and_json_roundtrip(tmp_path: Path) -> None:
    """summary(), _repr_html_(), and strict JSON round-trip preserve EH results identically."""
    df, _, _ = _skewed_2pl_data(seed=808, n=800, j=6)
    grp = np.repeat(["ref", "focal"], 400)
    quad = Quadrature.equispaced(21, lim=(-4, 4))
    spec = (
        Model.uniform(TwoPL(), items=df.columns)
        .with_groups(
            ref=EmpiricalHistogram(min_weight=1e-8), focal=EmpiricalHistogram(min_weight=1e-8)
        )
        .equal_across_groups(items=["q01", "q02", "q03"])
    )
    res = fit(spec, df, group=grp, method=BAEM(tol=1e-4), quadrature=quad, se="oakes")

    # Normal model has latent_weights is None
    res_norm = fit(Model.uniform(TwoPL(), items=df.columns), df, method=BAEM(tol=1e-4), se=None)
    assert res_norm.latent_weights is None

    # Summary & HTML include moments and EH weights
    txt = res.summary()
    assert "EmpiricalHistogram" in txt
    assert "skewness" in txt and "kurtosis" in txt
    assert "Empirical histogram weights" in txt
    html_out = res._repr_html_()
    assert "EmpiricalHistogram" in html_out
    assert "Empirical histogram weights" in html_out
    assert "skewness" in html_out

    # JSON round-trip (file and string)
    out_file = tmp_path / "eh_fit.json"
    json_str = res.to_json(out_file)
    parsed = json.loads(json_str)
    assert parsed["model"]["groups"][0]["latent"] == {
        "type": "EmpiricalHistogram",
        "min_weight": 1e-8,
    }
    assert "latent_weights" in parsed
    assert len(parsed["latent_weights"]) == 2
    assert len(parsed["latent_weights"][0]) == 21

    for loaded in (FitResult.from_json(out_file), FitResult.from_json(json_str)):
        assert loaded.loglik == res.loglik
        assert loaded.n_params == res.n_params
        assert loaded.aic == res.aic
        assert loaded.bic == res.bic
        assert loaded._model == res._model
        pd.testing.assert_frame_equal(loaded.latent, res.latent, check_exact=True)
        assert loaded.latent_weights is not None and res.latent_weights is not None
        pd.testing.assert_frame_equal(loaded.latent_weights, res.latent_weights, check_exact=True)
        pd.testing.assert_frame_equal(loaded.items, res.items, check_exact=True)
        assert loaded.summary() == res.summary()
        assert loaded._repr_html_() == res._repr_html_()
        sub, sub_g = df.iloc[:10], grp[:10]
        for meth in ("EAP", "ML", "WLE"):
            np.testing.assert_array_equal(
                loaded.score(sub, method=meth, group=sub_g).to_numpy(),
                res.score(sub, method=meth, group=sub_g).to_numpy(),
            )
        key = jax.random.key(9)
        np.testing.assert_array_equal(
            loaded.score(sub, method="PV", group=sub_g, key=key).to_numpy(),
            res.score(sub, method="PV", group=sub_g, key=key).to_numpy(),
        )

    # Malformed EH JSON errors
    bad_missing = dict(parsed)
    del bad_missing["latent_weights"]
    with pytest.raises(SpecificationError, match="latent_weights"):
        FitResult.from_json(json.dumps(bad_missing))

    bad_shape = dict(parsed, latent_weights=parsed["latent_weights"][:1])
    with pytest.raises(SpecificationError, match="group rows"):
        FitResult.from_json(json.dumps(bad_shape))

    bad_sum = dict(parsed, latent_weights=[[0.5] * 21, [0.5] * 21])
    with pytest.raises(SpecificationError, match="sum to 1"):
        FitResult.from_json(json.dumps(bad_sum))

    bad_floor = json.loads(json_str)
    bad_floor["model"]["groups"][0]["latent"]["min_weight"] = 0.01
    with pytest.raises(SpecificationError, match="below min_weight"):
        FitResult.from_json(json.dumps(bad_floor))

    bad_mw_grid = json.loads(json_str)
    bad_mw_grid["model"]["groups"][0]["latent"]["min_weight"] = 0.05  # 0.05 * 21 >= 1
    with pytest.raises(SpecificationError, match=r"min_weight \* n_points < 1"):
        FitResult.from_json(json.dumps(bad_mw_grid))

    # Normal JSON with unexpected latent_weights rejected
    norm_parsed = json.loads(res_norm.to_json())
    assert "latent_weights" not in norm_parsed
    norm_parsed["latent_weights"] = [[1.0 / 61] * 61]
    with pytest.raises(SpecificationError, match="Unexpected key 'latent_weights'"):
        FitResult.from_json(json.dumps(norm_parsed))
