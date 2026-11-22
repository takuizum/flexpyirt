"""Unit tests for Davidian-curve latent distributions (DESIGN.md §4.4, §16.4)."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from flexpyirt import (
    BAEM,
    DavidianCurve,
    FitResult,
    Model,
    Normal,
    NormalPrior,
    Quadrature,
    Rasch,
    TwoPL,
    fit,
)
from flexpyirt._compile import _davidian_phi_starts
from flexpyirt._core import information_hessian, information_oakes
from flexpyirt._json import SCHEMA_VERSION
from flexpyirt.errors import IdentificationError, SpecificationError
from flexpyirt.latent import (
    MAX_DAVIDIAN_DEGREE,
    davidian_hermite_coeffs,
    davidian_log_weights,
    davidian_moments,
    davidian_param_names,
    normal_log_weights,
)


def _skewed_2pl_data(
    *, seed: int = 20261121, n_obs: int = 800, n_items: int = 10
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    theta = (rng.chisquare(4.0, size=n_obs) - 4.0) / np.sqrt(8.0)
    a = rng.uniform(0.9, 1.8, size=n_items)
    d = np.linspace(-1.6, 1.6, n_items)
    probs = 1.0 / (1.0 + np.exp(-(theta[:, None] * a[None, :] + d[None, :])))
    y = (rng.random((n_obs, n_items)) < probs).astype(int)
    cols = [f"q{j + 1:02d}" for j in range(n_items)]
    return pd.DataFrame(y, columns=cols), a, d


@cache
def _fitted_dc3() -> tuple[pd.DataFrame, FitResult]:
    df, _, _ = _skewed_2pl_data()
    spec = Model.uniform(TwoPL(), items=df.columns).with_latent(DavidianCurve(3))
    res = fit(spec, df, method=BAEM(tol=1e-7, max_iter=1000, squarem=True), se="oakes")
    return df, res


# --------------------------------------------------------------------------------------------
# Spec & parameter-name validation
# --------------------------------------------------------------------------------------------


def test_davidian_spec_validation() -> None:
    d = DavidianCurve(3)
    assert (d.degree, d.mean, d.var, d.free_mean, d.free_var) == (3, 0.0, 1.0, False, False)
    df = DavidianCurve.free(4, mean=0.2, var=1.5)
    assert (df.degree, df.mean, df.var, df.free_mean, df.free_var) == (4, 0.2, 1.5, True, True)

    for bad_deg in (0, -2, MAX_DAVIDIAN_DEGREE + 1, True, False, 2.5, "3"):
        with pytest.raises(SpecificationError, match="degree"):
            DavidianCurve(bad_deg)  # type: ignore[arg-type]
        with pytest.raises(SpecificationError, match="degree"):
            davidian_param_names(bad_deg)  # type: ignore[arg-type]

    for bad_mean in (float("nan"), float("inf"), float("-inf"), True, "0"):
        with pytest.raises(SpecificationError, match="mean"):
            DavidianCurve(2, mean=bad_mean)  # type: ignore[arg-type]

    for bad_var in (0.0, -1.0, float("nan"), float("inf"), False, "1"):
        with pytest.raises(SpecificationError, match="var"):
            DavidianCurve(2, var=bad_var)  # type: ignore[arg-type]

    with pytest.raises(SpecificationError, match="free_mean"):
        DavidianCurve(2, free_mean=1)  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match="free_var"):
        DavidianCurve(2, free_var=0)  # type: ignore[arg-type]

    assert davidian_param_names(3) == ("mean", "var", "phi_1", "phi_2", "phi_3")


def test_model_unsupported_combinations() -> None:
    items = ["q1", "q2", "q3", "q4"]
    base = Model.uniform(TwoPL(), items=items)

    with pytest.raises(SpecificationError, match="single-group"):
        base.with_groups(ref=DavidianCurve(3), focal=DavidianCurve(3))
    with pytest.raises(SpecificationError, match="single-group"):
        base.with_groups(ref=Normal.standard(), focal=DavidianCurve(3))
    with pytest.raises(SpecificationError, match="single-group"):
        Model(items=base.items, groups=(("ref", DavidianCurve(2)), ("focal", DavidianCurve(2))))

    with pytest.raises(SpecificationError, match="unidimensional"):
        base.with_factors({"F1": ["q1", "q2"], "F2": ["q3", "q4"]}).with_latent(DavidianCurve(2))
    with pytest.raises(SpecificationError, match="with_latent"):
        base.with_latent(DavidianCurve(2)).with_factors({"F1": ["q1", "q2"], "F2": ["q3", "q4"]})

    with pytest.raises(SpecificationError, match="latent"):
        base.with_latent(DavidianCurve(2)).prior(param="phi_1", dist=NormalPrior(0.0, 1.0))

    df, _, _ = _skewed_2pl_data(n_obs=200, n_items=4)
    spec = Model.uniform(TwoPL(), items=df.columns).with_latent(DavidianCurve(2))
    tbl = spec.parameter_table(df)
    tbl["prior"] = pd.Series(
        [
            NormalPrior(0.0, 1.0) if p == "phi_1" else cur
            for p, cur in zip(tbl["param"], tbl["prior"], strict=True)
        ],
        dtype=object,
    )
    with pytest.raises(SpecificationError, match="Priors on latent"):
        fit(spec, df, table=tbl, se=None)

    with pytest.raises(SpecificationError, match="equispaced"):
        fit(spec, df, quadrature=Quadrature.gauss_hermite(21), se=None)


# --------------------------------------------------------------------------------------------
# Mathematical properties of Hermite spherical parameterization & weights
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("degree", [1, 2, 3, 4, 6, 10])
def test_hermite_coeffs_sphere_and_zero_origin(degree: int) -> None:
    rng = np.random.default_rng(100 + degree)
    phi = jnp.asarray(rng.uniform(-3.0, 3.0, size=degree), dtype=jnp.float64)
    coeffs = davidian_hermite_coeffs(phi)
    assert coeffs.shape == (degree + 1,)
    assert float(jnp.sum(coeffs * coeffs)) == pytest.approx(1.0, abs=1e-14)

    phi0 = jnp.zeros((degree,), dtype=jnp.float64)
    c0 = davidian_hermite_coeffs(phi0)
    expected = np.zeros(degree + 1)
    expected[0] = 1.0
    np.testing.assert_allclose(np.asarray(c0), expected, atol=1e-15)
    jac0 = np.asarray(jax.jacfwd(davidian_hermite_coeffs)(phi0))
    np.testing.assert_allclose(jac0[1:, :], np.eye(degree), atol=1e-15)


@pytest.mark.parametrize("degree", [1, 2, 3, 5])
def test_moments_match_numerical_integration(degree: int) -> None:
    rng = np.random.default_rng(200 + degree)
    phi = jnp.asarray(rng.uniform(-1.2, 1.2, size=degree), dtype=jnp.float64)
    coeffs = davidian_hermite_coeffs(phi)
    mu_exact, var_exact = davidian_moments(coeffs)

    z = np.linspace(-10.0, 10.0, 4001)
    # Normalized probabilists' Hermite basis
    h = np.zeros((z.size, degree + 1))
    h[:, 0] = 1.0
    if degree >= 1:
        h[:, 1] = z
    for j in range(2, degree + 1):
        h[:, j] = (z * h[:, j - 1] - np.sqrt(j - 1) * h[:, j - 2]) / np.sqrt(j)
    poly = h @ np.asarray(coeffs)
    dens = (poly**2) * np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    area = float(np.trapezoid(dens, z))
    mu_num = float(np.trapezoid(z * dens, z))
    var_num = float(np.trapezoid(((z - mu_num) ** 2) * dens, z))
    assert area == pytest.approx(1.0, abs=1e-11)
    assert float(mu_exact) == pytest.approx(mu_num, abs=1e-10)
    assert float(var_exact) == pytest.approx(var_num, abs=1e-10)


@pytest.mark.parametrize("degree", [1, 2, 3, 4, 8])
def test_weights_sum_to_one_zero_equals_normal_and_moments_standardized(degree: int) -> None:
    nodes = jnp.linspace(-6.0, 6.0, 61, dtype=jnp.float64)
    # All-zero angles reproduce Normal(mean, var) identically.
    for mean, var in ((0.0, 1.0), (0.7, 1.8)):
        params0 = jnp.asarray([mean, var, *([0.0] * degree)], dtype=jnp.float64)
        lw_dc = davidian_log_weights(nodes, params0)
        lw_nm = normal_log_weights(nodes, jnp.asarray(mean), jnp.asarray(var))
        np.testing.assert_allclose(np.asarray(lw_dc), np.asarray(lw_nm), rtol=0, atol=1e-14)
        assert float(jnp.sum(jnp.exp(lw_dc))) == pytest.approx(1.0, abs=1e-14)

    # Arbitrary angles: weights sum to 1, and on a wide grid discrete moments match (mean, var).
    rng = np.random.default_rng(300 + degree)
    phi = rng.uniform(-1.0, 1.0, size=degree)
    wide = jnp.linspace(-10.0, 10.0, 201, dtype=jnp.float64)
    mean, var = 0.35, 1.4
    params = jnp.asarray([mean, var, *phi], dtype=jnp.float64)
    w = np.exp(np.asarray(davidian_log_weights(wide, params)))
    assert float(np.sum(w)) == pytest.approx(1.0, abs=1e-14)
    w_mean = float(np.sum(w * np.asarray(wide)))
    w_var = float(np.sum(w * (np.asarray(wide) - w_mean) ** 2))
    assert w_mean == pytest.approx(mean, abs=1e-8)
    assert w_var == pytest.approx(var, abs=1e-7)

    # Non-positive variance returns NaN.
    bad = jnp.asarray([0.0, -0.5, *phi], dtype=jnp.float64)
    assert bool(jnp.all(jnp.isnan(davidian_log_weights(nodes, bad))))


def test_degree_nesting() -> None:
    nodes = jnp.linspace(-6.0, 6.0, 61, dtype=jnp.float64)
    p2 = jnp.asarray([0.2, 1.3, -0.4, 0.35], dtype=jnp.float64)
    p3 = jnp.asarray([0.2, 1.3, -0.4, 0.35, 0.0], dtype=jnp.float64)
    np.testing.assert_allclose(
        np.asarray(davidian_log_weights(nodes, p2)),
        np.asarray(davidian_log_weights(nodes, p3)),
        rtol=0,
        atol=1e-14,
    )


def test_finite_gradients_and_polynomial_root_guard() -> None:
    nodes = jnp.linspace(-6.0, 6.0, 61, dtype=jnp.float64)
    n_bar = jnp.exp(-0.5 * (nodes - 0.3) ** 2)
    n_bar = n_bar / jnp.sum(n_bar) * 500.0
    params = jnp.asarray([0.1, 1.2, -0.3, 0.25, 0.15], dtype=jnp.float64)

    def obj(p: jax.Array) -> jax.Array:
        lw = davidian_log_weights(nodes, p)
        return jnp.sum(jnp.where(n_bar > 0.0, n_bar * lw, 0.0))

    grad_ad = np.asarray(jax.grad(obj)(params))
    eps = 1e-6
    grad_fd = np.zeros_like(grad_ad)
    p_np = np.asarray(params)
    for i in range(p_np.size):
        step = np.zeros_like(p_np)
        step[i] = eps
        grad_fd[i] = (
            float(obj(jnp.asarray(p_np + step))) - float(obj(jnp.asarray(p_np - step)))
        ) / (2.0 * eps)
    np.testing.assert_allclose(grad_ad, grad_fd, rtol=1e-6, atol=1e-7)
    assert np.all(np.isfinite(np.asarray(jax.hessian(obj)(params))))

    # Exact polynomial root at theta=0: k=1, phi=pi/2 -> c=(0, 1) -> P_1(z) = z.
    p_root = jnp.asarray([0.0, 1.0, 0.5 * np.pi], dtype=jnp.float64)
    w_root = jnp.exp(davidian_log_weights(nodes, p_root))
    n_root = w_root * 200.0  # zero count at node 30 (theta=0)

    def obj_root(p: jax.Array) -> jax.Array:
        lw = davidian_log_weights(nodes, p)
        return jnp.sum(jnp.where(n_root > 0.0, n_root * lw, 0.0))

    assert np.all(np.isfinite(np.asarray(jax.grad(obj_root)(p_root))))
    assert np.all(np.isfinite(np.asarray(jax.hessian(obj_root)(p_root))))


# --------------------------------------------------------------------------------------------
# Identification checks & model variants
# --------------------------------------------------------------------------------------------


def test_identification_errors_and_rasch_fipc() -> None:
    df, a_true, d_true = _skewed_2pl_data(n_obs=500, n_items=8)
    with pytest.raises(IdentificationError, match="variance"):
        fit(
            Model.uniform(TwoPL(), items=df.columns).with_latent(DavidianCurve(2, free_var=True)),
            df,
            se=None,
        )
    with pytest.raises(IdentificationError, match="mean"):
        fit(
            Model.uniform(TwoPL(), items=df.columns).with_latent(DavidianCurve(2, free_mean=True)),
            df,
            se=None,
        )

    # Rasch with free variance and Davidian curve is identified.
    res_rasch = fit(
        Model.uniform(Rasch(), items=df.columns).with_latent(DavidianCurve(2, free_var=True)),
        df,
        method=BAEM(tol=1e-5, squarem=True),
        se="oakes",
    )
    assert res_rasch.converged
    assert np.isnan(float(res_rasch.latent["mean_se"].iloc[0]))
    assert float(res_rasch.latent["var_se"].iloc[0]) > 0.0
    assert float(res_rasch.latent["phi_1_se"].iloc[0]) > 0.0

    # FIPC with free mean, var, and Davidian curve is identified.
    bank = pd.DataFrame({"item": list(df.columns[:4]), "a": a_true[:4], "d": d_true[:4]})
    res_fipc = fit(
        Model.uniform(TwoPL(), items=df.columns).fix_from(bank).with_latent(DavidianCurve.free(2)),
        df,
        method=BAEM(tol=1e-5, squarem=True),
        se="oakes",
    )
    assert res_fipc.converged
    assert float(res_fipc.latent["mean_se"].iloc[0]) > 0.0
    assert float(res_fipc.latent["var_se"].iloc[0]) > 0.0


# --------------------------------------------------------------------------------------------
# Estimation, Oakes == Hessian, Normal equivalence, scoring, reporting, JSON
# --------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_plain_em_vs_squarem_monotonicity_and_oakes_equals_hessian() -> None:
    df, res_sq = _fitted_dc3()
    spec = res_sq._model
    res_em = fit(spec, df, method=BAEM(tol=1e-7, max_iter=1000, squarem=False), se="hessian")

    assert res_sq.converged and res_em.converged
    for res in (res_em, res_sq):
        ll = res.history["loglik"].to_numpy()
        assert np.all(np.diff(ll) >= -1e-10 * np.abs(ll[1:]))

    assert res_sq.loglik == pytest.approx(res_em.loglik, rel=1e-10)
    np.testing.assert_allclose(
        res_sq.items["a"].to_numpy(), res_em.items["a"].to_numpy(), rtol=0, atol=1e-5
    )
    np.testing.assert_allclose(
        res_sq.items["d"].to_numpy(), res_em.items["d"].to_numpy(), rtol=0, atol=1e-5
    )
    for col in ("phi_1", "phi_2", "phi_3"):
        assert float(res_sq.latent[col].iloc[0]) == pytest.approx(
            float(res_em.latent[col].iloc[0]), abs=1e-5
        )

    assert res_sq.vcov is not None and res_em.vcov is not None
    # Also check Oakes == Hessian at the exact same eta.
    from flexpyirt._compile import compile_problem, prepare

    prep = prepare(spec, df, group=None, item_names=None, table=res_sq.parameters)
    comp = compile_problem(spec, prep.y, prep.group, prep.table, res_sq._quadrature)
    eta = jnp.asarray(res_sq._eta)
    oakes = np.asarray(information_oakes(comp.problem, eta))
    hess = np.asarray(information_hessian(comp.problem, eta))
    np.testing.assert_allclose(oakes, hess, rtol=1e-9, atol=1e-9 * np.abs(hess).max())


def test_fixed_zero_angles_match_normal_fit() -> None:
    df, _, _ = _skewed_2pl_data(n_obs=400, n_items=6)
    spec_nm = Model.uniform(TwoPL(), items=df.columns)
    spec_dc = Model.uniform(TwoPL(), items=df.columns).with_latent(DavidianCurve(3))
    tbl = spec_dc.parameter_table(df)
    is_phi = tbl["param"].str.startswith("phi_")
    tbl.loc[is_phi, "value"] = 0.0
    tbl.loc[is_phi, "free"] = False

    res_nm = fit(spec_nm, df, method=BAEM(tol=1e-8, squarem=True), se="oakes")
    res_dc = fit(spec_dc, df, table=tbl, method=BAEM(tol=1e-8, squarem=True), se="oakes")
    assert res_dc.loglik == pytest.approx(res_nm.loglik, rel=1e-12)
    np.testing.assert_allclose(
        res_dc.items["a"].to_numpy(), res_nm.items["a"].to_numpy(), rtol=0, atol=1e-7
    )
    assert res_dc.vcov is not None and res_nm.vcov is not None
    np.testing.assert_allclose(res_dc.vcov, res_nm.vcov, rtol=1e-7, atol=1e-9)


def test_scoring_reporting_item_fit_and_json_roundtrip(tmp_path: Path) -> None:
    df, res = _fitted_dc3()
    sub = df.iloc[:30]

    # Result attributes, summary, and HTML.
    assert res.latent_weights is None
    assert list(res.latent.columns) == [
        "group",
        "mean",
        "var",
        "phi_1",
        "phi_2",
        "phi_3",
        "mean_se",
        "var_se",
        "phi_1_se",
        "phi_2_se",
        "phi_3_se",
    ]
    assert res.n_params == 2 * df.shape[1] + 3
    txt = res.summary()
    assert "DavidianCurve, degree=3" in txt
    assert "phi_1" in txt
    html_out = res._repr_html_()
    assert "DavidianCurve, degree=3" in html_out
    assert "phi_1" in html_out

    # EAP uses the fitted Davidian prior (differs from N(0,1) prior with identical item params).
    eap_dc = res.score(sub, method="EAP")
    norm_spec = Model.uniform(TwoPL(), items=df.columns)
    norm_tbl = norm_spec.parameter_table(df)
    for _, row in res.items.iterrows():
        for p in ("a", "d"):
            sel = (norm_tbl["item"] == row["item"]) & (norm_tbl["param"] == p)
            norm_tbl.loc[sel, "value"] = float(row[p])
            norm_tbl.loc[sel, "free"] = False
    res_norm_fixed = fit(norm_spec, df, table=norm_tbl, se=None)
    eap_nm = res_norm_fixed.score(sub, method="EAP")
    assert float(np.max(np.abs(eap_dc["theta"].to_numpy() - eap_nm["theta"].to_numpy()))) > 0.01

    # ML and WLE are likelihood-only and match the Normal model with identical item params.
    meths: tuple[Literal["ML", "WLE"], ...] = ("ML", "WLE")
    for meth in meths:
        sc_dc = res.score(sub, method=meth)
        sc_nm = res_norm_fixed.score(sub, method=meth)
        np.testing.assert_allclose(sc_dc.to_numpy(), sc_nm.to_numpy(), rtol=1e-10, atol=1e-10)

    # PV uses the fitted Davidian density: deterministic given key, sample mean/SD match EAP.
    pv1 = res.score(sub.iloc[:5], method="PV", key=jax.random.key(0), n_draws=4000)
    pv2 = res.score(sub.iloc[:5], method="PV", key=jax.random.key(0), n_draws=4000)
    np.testing.assert_array_equal(pv1.to_numpy(), pv2.to_numpy())
    pv_means = pv1.to_numpy().mean(axis=1)
    pv_sds = pv1.to_numpy().std(axis=1, ddof=1)
    np.testing.assert_allclose(pv_means, eap_dc["theta"].to_numpy()[:5], atol=0.05)
    np.testing.assert_allclose(pv_sds, eap_dc["se"].to_numpy()[:5], atol=0.05)

    # MAP and item_fit raise SpecificationError.
    with pytest.raises(SpecificationError, match="MAP"):
        res.score(sub, method="MAP")
    with pytest.raises(SpecificationError, match="Normal"):
        res.item_fit(df)

    # JSON round trip preserves everything bit-for-bit (and keeps schema_version == 3).
    out_path = tmp_path / "dc3.json"
    raw_str = res.to_json(out_path)
    payload = json.loads(raw_str)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["model"]["latent"]["type"] == "DavidianCurve"
    assert "latent_weights" not in payload

    loaded = FitResult.from_json(out_path)
    assert loaded.loglik == res.loglik
    assert loaded.summary() == res.summary()
    assert loaded._repr_html_() == res._repr_html_()
    pd.testing.assert_frame_equal(loaded.latent, res.latent)
    pd.testing.assert_frame_equal(loaded.items, res.items)
    assert loaded.vcov is not None and res.vcov is not None
    np.testing.assert_array_equal(loaded.vcov, res.vcov)
    np.testing.assert_array_equal(loaded.score(sub, method="EAP").to_numpy(), eap_dc.to_numpy())
    np.testing.assert_array_equal(
        loaded.score(sub.iloc[:5], method="PV", key=jax.random.key(1), n_draws=5).to_numpy(),
        res.score(sub.iloc[:5], method="PV", key=jax.random.key(1), n_draws=5).to_numpy(),
    )


# --------------------------------------------------------------------------------------------
# Edge cases: start signs, degree 1 and 10 fits, polynomial root at a node, malformed JSON
# --------------------------------------------------------------------------------------------


def _third_standardized_moment(params: list[float]) -> float:
    z = jnp.linspace(-12.0, 12.0, 2401, dtype=jnp.float64)
    w = np.exp(np.asarray(davidian_log_weights(z, jnp.asarray(params, dtype=jnp.float64))))
    zn = np.asarray(z)
    m = float(np.sum(w * zn))
    v = float(np.sum(w * (zn - m) ** 2))
    return float(np.sum(w * (zn - m) ** 3)) / float(v**1.5)


@pytest.mark.parametrize("degree", [1, 2, 3, 10])
def test_start_angles_match_total_score_skewness(degree: int) -> None:
    df, _, _ = _skewed_2pl_data(n_obs=600, n_items=8)
    y = df.to_numpy()
    starts = _davidian_phi_starts(degree, y)
    assert starts == _davidian_phi_starts(degree, y)  # deterministic
    # Positively skewed data -> positive start skewness; reversed scores -> negative.
    assert _third_standardized_moment([0.0, 1.0, *starts]) > 0.0
    flipped = _davidian_phi_starts(degree, 1 - y)
    assert _third_standardized_moment([0.0, 1.0, *flipped]) < 0.0
    # The start is off the stationary point phi = 0.
    assert any(abs(s) > 0.0 for s in starts)


@pytest.mark.parametrize("degree", [1, 10])
def test_fit_degree_extremes(degree: int) -> None:
    df, _, _ = _skewed_2pl_data()
    spec = Model.uniform(TwoPL(), items=df.columns).with_latent(DavidianCurve(degree))
    res = fit(spec, df, method=BAEM(tol=1e-6, max_iter=3000, squarem=True), se="oakes")
    assert res.converged
    ll = res.history["loglik"].to_numpy()
    assert np.all(np.diff(ll) >= -1e-9 * np.abs(ll[1:]))
    assert res.vcov is not None
    assert np.all(np.isfinite(res.vcov))
    assert np.all(np.linalg.eigvalsh(res.vcov) > 0.0)
    se = res.latent[[f"phi_{j}_se" for j in range(1, degree + 1)]].to_numpy()
    assert np.all(np.isfinite(se)) and np.all(se > 0.0)
    assert res.n_params == 2 * df.shape[1] + degree


@pytest.mark.parametrize("free_phi", [False, True])
def test_fit_with_polynomial_root_on_a_quadrature_node(free_phi: bool) -> None:
    # degree=1, phi_1 = pi/2 -> P_1(z) = z, whose root z = 0 is the grid node theta = 0, so
    # log w = -inf at a node while the data put positive likelihood there.
    df, _, _ = _skewed_2pl_data()
    spec = Model.uniform(TwoPL(), items=df.columns).with_latent(DavidianCurve(1))
    tbl = spec.parameter_table(df)
    is_phi = tbl["param"] == "phi_1"
    tbl.loc[is_phi, "value"] = 0.5 * np.pi
    tbl.loc[is_phi, "free"] = free_phi
    res = fit(spec, df, table=tbl, method=BAEM(tol=1e-6, squarem=True), se="oakes")
    assert res.converged
    assert np.isfinite(res.loglik)
    ll = res.history["loglik"].to_numpy()
    assert np.all(np.isfinite(ll))
    assert np.all(np.diff(ll) >= -1e-9 * np.abs(ll[1:]))
    assert res.vcov is not None and np.all(np.isfinite(res.vcov))
    if not free_phi:
        assert float(res.latent["phi_1"].iloc[0]) == 0.5 * np.pi
        sc = res.score(df.iloc[:10], method="EAP")
        assert np.all(np.isfinite(sc.to_numpy()))
    else:
        assert float(res.latent["phi_1_se"].iloc[0]) > 0.0


def test_malformed_davidian_json() -> None:
    _, res = _fitted_dc3()
    payload = json.loads(res.to_json())

    def load_with(match: str, *, drop: str | None = None, **update: Any) -> None:
        bad = json.loads(json.dumps(payload))
        latent: dict[str, Any] = bad["model"]["latent"]
        if drop is not None:
            latent.pop(drop)
        latent.update(update)
        with pytest.raises(SpecificationError, match=match):
            FitResult.from_json(json.dumps(bad))

    for key in ("degree", "mean", "var", "free_mean", "free_var"):
        load_with(key, drop=key)
    load_with("extra", extra=1)
    for bad_degree in (0, MAX_DAVIDIAN_DEGREE + 1, True, 2.5, "3"):
        load_with("degree", degree=bad_degree)
    load_with("var", var=0.0)
    load_with("free_mean", free_mean="yes")
    # phi rows must match the declared degree (3 phi rows in the payload).
    load_with("phi_3|parameter", degree=2)
    load_with("phi_4|parameter", degree=4)
