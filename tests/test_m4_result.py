"""Unit tests for M4 Result layer polish: delta-method IRT SEs, summary(), _repr_html_()."""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from flexpyirt import (
    BAEM,
    GPCM,
    GRM,
    PCM,
    FitResult,
    FourPL,
    LogNormalPrior,
    Model,
    Normal,
    NormalPrior,
    Quadrature,
    Rasch,
    ThreePL,
    TwoPL,
    fit,
)
from flexpyirt.errors import ConvergenceWarning, FlexpyirtWarning, SpecificationError
from tests._reference import load


def _cell(frame: pd.DataFrame, row: Any, col: str) -> float:
    return float(frame[col].loc[row])


def _free_id_map(res: FitResult) -> dict[tuple[str, str, str], int]:
    p = res.parameters
    return {
        (str(g), str(i), str(k)): int(fid)
        for g, i, k, fid in zip(p["group"], p["item"], p["param"], p["free_id"], strict=True)
    }


def _analytic_ratio_se(
    a: float,
    d: float,
    vcov: np.ndarray,
    fid_a: int,
    fid_d: int,
) -> float:
    """Analytic delta-method SE of b = -d / a with fixed parameters contributing 0 variance."""
    va = vcov[fid_a, fid_a] if fid_a >= 0 else 0.0
    vd = vcov[fid_d, fid_d] if fid_d >= 0 else 0.0
    vad = vcov[fid_a, fid_d] if (fid_a >= 0 and fid_d >= 0) else 0.0
    ga = d / (a * a)
    gd = -1.0 / a
    return float(np.sqrt(max(ga * ga * va + gd * gd * vd + 2.0 * ga * gd * vad, 0.0)))


def _analytic_gpcm_step_se(
    a: float,
    d_prev: float,
    d_curr: float,
    vcov: np.ndarray,
    fid_a: int,
    fid_prev: int,
    fid_curr: int,
) -> float:
    """Analytic delta-method SE of b_k = -(d_k - d_{k-1}) / a."""
    fids = [fid_a, fid_prev, fid_curr]
    grad = np.array([(d_curr - d_prev) / (a * a), 1.0 / a, -1.0 / a], dtype=float)
    var = 0.0
    for i, fi in enumerate(fids):
        if fi < 0:
            continue
        for j, fj in enumerate(fids):
            if fj < 0:
                continue
            var += grad[i] * grad[j] * vcov[fi, fj]
    return float(np.sqrt(max(var, 0.0)))


# --------------------------------------------------------------------------------------------
# Delta-method SEs for display IRT thresholds
# --------------------------------------------------------------------------------------------


def test_delta_se_2pl_matches_analytic_and_includes_covariance() -> None:
    case = load("v1_lsat7_2pl")
    res = fit(Model.uniform(TwoPL(), items=case.items), case.data, method=BAEM(tol=1e-8))
    assert res.vcov is not None
    fids = _free_id_map(res)
    items = res.items.set_index("item")
    assert "b_se" in items.columns

    cov_matters = False
    for name in case.items:
        a = _cell(items, name, "a")
        d = _cell(items, name, "d")
        fa, fd = fids[("all", name, "a")], fids[("all", name, "d")]
        expected = _analytic_ratio_se(a, d, res.vcov, fa, fd)
        assert _cell(items, name, "b_se") == pytest.approx(expected, rel=1e-12)
        # Verify the a-d covariance term is genuinely non-negligible and included.
        no_cov = float(
            np.sqrt((d / (a * a)) ** 2 * res.vcov[fa, fa] + (1.0 / a) ** 2 * res.vcov[fd, fd])
        )
        if abs(expected - no_cov) / expected > 1e-2:
            cov_matters = True
    assert cov_matters


def test_delta_se_rasch_and_fixed_slope() -> None:
    case = load("v2_lsat7_rasch")
    res = fit(Model.uniform(Rasch(), items=case.items), case.data, method=BAEM(tol=1e-8))
    items = res.items
    np.testing.assert_allclose(items["b"], -items["d"], atol=1e-14)
    np.testing.assert_allclose(items["b_se"], items["d_se"], rtol=1e-12)

    # Fixed non-unit slope a = 2.0 -> b_se = d_se / 2.0.
    spec = Model.uniform(TwoPL(), items=case.items).fix(item=case.items[0], a=2.0)
    res_fix = fit(spec, case.data, method=BAEM(tol=1e-8))
    it_fix = res_fix.items.set_index("item")
    assert np.isnan(_cell(it_fix, case.items[0], "a_se"))
    assert _cell(it_fix, case.items[0], "b_se") == pytest.approx(
        _cell(it_fix, case.items[0], "d_se") / 2.0, rel=1e-12
    )


def test_delta_se_grm_matches_analytic() -> None:
    rng = np.random.default_rng(4001)
    theta = rng.normal(0.0, 1.0, 600)
    cuts = np.array([-1.1, 0.1, 1.2])
    y = np.searchsorted(cuts, np.outer(theta, [1.2, 0.9, 1.5, 1.1]) + rng.logistic(0, 1, (600, 4)))
    df = pd.DataFrame(y.astype(float), columns=["s1", "s2", "s3", "s4"])
    res = fit(Model.uniform(GRM(4), items=df.columns), df, method=BAEM(tol=1e-8))
    assert res.vcov is not None
    fids = _free_id_map(res)
    items = res.items.set_index("item")
    for k in (1, 2, 3):
        assert f"b{k}_se" in items.columns
    for name in df.columns:
        a = _cell(items, name, "a")
        fa = fids[("all", name, "a")]
        for k in (1, 2, 3):
            dk = _cell(items, name, f"d{k}")
            fd = fids[("all", name, f"d{k}")]
            expected = _analytic_ratio_se(a, dk, res.vcov, fa, fd)
            assert _cell(items, name, f"b{k}_se") == pytest.approx(expected, rel=1e-12)


def test_delta_se_gpcm_and_pcm_include_step_covariances() -> None:
    rng = np.random.default_rng(4002)
    theta = rng.normal(0.0, 1.0, 700)
    cuts = np.array([-1.0, 0.2, 1.3])
    y = np.searchsorted(cuts, np.outer(theta, [1.1, 0.8, 1.4, 1.0]) + rng.logistic(0, 1, (700, 4)))
    df = pd.DataFrame(y.astype(float), columns=["p1", "p2", "p3", "p4"])

    # GPCM: covariance between a and d_k AND between d_{k-1} and d_k.
    res_gpcm = fit(Model.uniform(GPCM(4), items=df.columns), df, method=BAEM(tol=1e-8))
    assert res_gpcm.vcov is not None
    fids_g = _free_id_map(res_gpcm)
    items_g = res_gpcm.items.set_index("item")
    d_cov_matters = False
    for name in df.columns:
        a = _cell(items_g, name, "a")
        fa = fids_g[("all", name, "a")]
        ds = [0.0, *(_cell(items_g, name, f"d{k}") for k in (1, 2, 3))]
        f_ds = [-1, *(fids_g[("all", name, f"d{k}")] for k in (1, 2, 3))]
        for k in (1, 2, 3):
            expected = _analytic_gpcm_step_se(
                a, ds[k - 1], ds[k], res_gpcm.vcov, fa, f_ds[k - 1], f_ds[k]
            )
            assert _cell(items_g, name, f"b{k}_se") == pytest.approx(expected, rel=1e-12)
            if k >= 2:
                # Zero out only Cov(d_{k-1}, d_k) to confirm it matters.
                v_nocov = res_gpcm.vcov.copy()
                v_nocov[f_ds[k - 1], f_ds[k]] = 0.0
                v_nocov[f_ds[k], f_ds[k - 1]] = 0.0
                alt = _analytic_gpcm_step_se(a, ds[k - 1], ds[k], v_nocov, fa, f_ds[k - 1], f_ds[k])
                if abs(expected - alt) / expected > 1e-2:
                    d_cov_matters = True
    assert d_cov_matters

    # PCM: a = 1 fixed -> b1_se == d1_se; b2_se = sqrt(Var(d1)+Var(d2)-2Cov(d1,d2)).
    res_pcm = fit(Model.uniform(PCM(4), items=df.columns), df, method=BAEM(tol=1e-8))
    assert res_pcm.vcov is not None
    fids_p = _free_id_map(res_pcm)
    items_p = res_pcm.items.set_index("item")
    np.testing.assert_allclose(items_p["b1_se"], items_p["d1_se"], rtol=1e-12)
    for name in df.columns:
        f1 = fids_p[("all", name, "d1")]
        f2 = fids_p[("all", name, "d2")]
        f3 = fids_p[("all", name, "d3")]
        expected_b2 = float(
            np.sqrt(res_pcm.vcov[f1, f1] + res_pcm.vcov[f2, f2] - 2.0 * res_pcm.vcov[f1, f2])
        )
        expected_b3 = float(
            np.sqrt(res_pcm.vcov[f2, f2] + res_pcm.vcov[f3, f3] - 2.0 * res_pcm.vcov[f2, f3])
        )
        assert _cell(items_p, name, "b2_se") == pytest.approx(expected_b2, rel=1e-12)
        assert _cell(items_p, name, "b3_se") == pytest.approx(expected_b3, rel=1e-12)


def test_delta_se_gpcm2_equals_2pl_and_pcm2_equals_rasch() -> None:
    case = load("v1_lsat7_2pl")
    r_2pl = fit(Model.uniform(TwoPL(), items=case.items), case.data, method=BAEM(tol=1e-9))
    r_gpcm = fit(Model.uniform(GPCM(2), items=case.items), case.data, method=BAEM(tol=1e-9))
    np.testing.assert_allclose(r_gpcm.items["b1_se"], r_2pl.items["b_se"], rtol=1e-7)

    r_rasch = fit(Model.uniform(Rasch(), items=case.items), case.data, method=BAEM(tol=1e-9))
    r_pcm = fit(Model.uniform(PCM(2), items=case.items), case.data, method=BAEM(tol=1e-9))
    np.testing.assert_allclose(r_pcm.items["b1_se"], r_rasch.items["b_se"], rtol=1e-7)


def test_delta_se_3pl_4pl_and_fixed_a_d_free_g() -> None:
    rng = np.random.default_rng(4003)
    theta = rng.normal(0.0, 1.0, 1500)
    a = np.array([1.7, 1.8, 1.6, 1.9, 1.7, 1.8, 1.6, 1.7])
    d = np.array([-0.8, -0.5, -1.0, -0.6, -0.7, -0.9, -0.4, -0.8])
    g = np.full(len(a), 0.18)
    u = np.full(len(a), 0.88)
    p = g + (u - g) / (1.0 + np.exp(-(np.outer(theta, a) + d)))
    cols = [f"q{j}" for j in range(1, len(a) + 1)]
    df = pd.DataFrame((rng.random(p.shape) < p).astype(float), columns=cols)

    # 3PL with MAP priors to keep interior estimates clean (no HeywoodWarning).
    spec3 = (
        Model.uniform(ThreePL(), items=cols)
        .prior(param="a", dist=LogNormalPrior(0.4, 0.3))
        .prior(param="g", dist=NormalPrior(0.2, 0.1))
    )
    res3 = fit(spec3, df, method=BAEM(tol=1e-8))
    assert res3.vcov is not None
    fids3 = _free_id_map(res3)
    items3 = res3.items.set_index("item")
    for name in cols:
        expected = _analytic_ratio_se(
            _cell(items3, name, "a"),
            _cell(items3, name, "d"),
            res3.vcov,
            fids3[("all", name, "a")],
            fids3[("all", name, "d")],
        )
        assert _cell(items3, name, "b_se") == pytest.approx(expected, rel=1e-12)

    # Fixing a and d while leaving g free: b does not depend on g, so b_se must be NaN.
    spec_fix_ad = spec3.fix(item="q1", a=1.7, d=-0.8)
    res_fix_ad = fit(spec_fix_ad, df, method=BAEM(tol=1e-8))
    it_fix_ad = res_fix_ad.items.set_index("item")
    assert np.isfinite(_cell(it_fix_ad, "q1", "g_se"))
    assert np.isnan(_cell(it_fix_ad, "q1", "a_se"))
    assert np.isnan(_cell(it_fix_ad, "q1", "d_se"))
    assert np.isnan(_cell(it_fix_ad, "q1", "b_se"))

    # 4PL (interior via priors).
    spec4 = (
        Model.uniform(FourPL(), items=cols)
        .prior(param="a", dist=LogNormalPrior(0.4, 0.3))
        .prior(param="g", dist=NormalPrior(0.18, 0.08))
        .prior(param="u", dist=NormalPrior(0.88, 0.08))
    )
    res4 = fit(spec4, df, method=BAEM(tol=1e-8))
    assert res4.vcov is not None
    fids4 = _free_id_map(res4)
    items4 = res4.items.set_index("item")
    for name in cols:
        expected = _analytic_ratio_se(
            _cell(items4, name, "a"),
            _cell(items4, name, "d"),
            res4.vcov,
            fids4[("all", name, "a")],
            fids4[("all", name, "d")],
        )
        assert _cell(items4, name, "b_se") == pytest.approx(expected, rel=1e-12)


def test_delta_se_fixed_items_fixed_d_and_vcov_none_or_nan() -> None:
    case = load("v1_lsat7_2pl")
    # FIPC: first 2 items fixed, latent free.
    spec = (
        Model.uniform(TwoPL(), items=case.items)
        .fix(item="Item.1", a=1.0, d=1.8)
        .fix(item="Item.2", a=1.1, d=0.8)
        .with_latent(Normal.free())
    )
    res = fit(
        spec,
        case.data,
        method=BAEM(tol=1e-8),
        quadrature=Quadrature.equispaced(81, bounds=(-8, 8)),
    )
    items = res.items.set_index("item")
    assert np.isnan(_cell(items, "Item.1", "b_se"))
    assert np.isnan(_cell(items, "Item.2", "b_se"))
    assert np.all(np.isfinite(items["b_se"].loc[["Item.3", "Item.4", "Item.5"]]))

    # Fixed d != 0, free a -> b_se = |d| / a^2 * a_se.
    spec_fix_d = Model.uniform(TwoPL(), items=case.items).fix(item="Item.1", d=1.5)
    res_fix_d = fit(spec_fix_d, case.data, method=BAEM(tol=1e-8))
    it_d = res_fix_d.items.set_index("item")
    assert _cell(it_d, "Item.1", "b_se") == pytest.approx(
        abs(1.5) / (_cell(it_d, "Item.1", "a") ** 2) * _cell(it_d, "Item.1", "a_se"), rel=1e-12
    )

    # se=None -> vcov is None -> b_se column present and all NaN.
    res_none = fit(Model.uniform(TwoPL(), items=case.items), case.data, se=None)
    assert "b_se" in res_none.items.columns
    assert res_none.items["b_se"].isna().all()

    # Non-PD vcov (all NaN) -> b_se all NaN without error.
    assert res.vcov is not None
    res_nan = replace(res, vcov=np.full_like(res.vcov, np.nan))
    assert res_nan.items["b_se"].isna().all()


def test_delta_se_multigroup_shared_and_partial_equality() -> None:
    rng = np.random.default_rng(4004)
    n_g = 600
    a_true = np.array([1.2, 1.5, 1.0, 1.4, 1.1, 1.3])
    d_ref = np.array([-0.6, 0.2, 0.8, -0.2, 0.4, -0.5])
    d_foc = d_ref.copy()
    d_foc[4:] += np.array([0.4, -0.3])
    th_ref = rng.normal(0.0, 1.0, n_g)
    th_foc = rng.normal(0.4, 1.1, n_g)

    def sim(th: np.ndarray, d: np.ndarray) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-(np.outer(th, a_true) + d)))
        return np.asarray(rng.random(p.shape) < p, dtype=float)

    cols = [f"q{j}" for j in range(1, 7)]
    df = pd.DataFrame(np.vstack([sim(th_ref, d_ref), sim(th_foc, d_foc)]), columns=cols)
    grp = np.repeat(["ref", "focal"], n_g)

    # Full anchors q1..q3; partial equality (a shared, d group-specific) on q4; free on q5..q6.
    spec = (
        Model.uniform(TwoPL(), items=cols)
        .with_groups(ref=Normal.standard(), focal=Normal.free())
        .equal_across_groups(items=["q1", "q2", "q3"])
        .equal_across_groups(items=["q4"], params=["a"])
    )
    res = fit(
        spec,
        df,
        group=grp,
        method=BAEM(tol=1e-8),
        quadrature=Quadrature.equispaced(81, bounds=(-8, 8)),
    )
    assert res.vcov is not None
    fids = _free_id_map(res)
    items = res.items.set_index(["group", "item"])

    # Full anchors have identical b and b_se across groups.
    for anc in ("q1", "q2", "q3"):
        assert _cell(items, ("ref", anc), "b") == _cell(items, ("focal", anc), "b")
        assert _cell(items, ("ref", anc), "b_se") == _cell(items, ("focal", anc), "b_se")

    # Partial anchor q4 shares a across groups, has distinct d per group; verify exact vcov match.
    assert fids[("ref", "q4", "a")] == fids[("focal", "q4", "a")]
    assert fids[("ref", "q4", "d")] != fids[("focal", "q4", "d")]
    for g_name in ("ref", "focal"):
        for item in cols:
            expected = _analytic_ratio_se(
                _cell(items, (g_name, item), "a"),
                _cell(items, (g_name, item), "d"),
                res.vcov,
                fids[(g_name, item, "a")],
                fids[(g_name, item, "d")],
            )
            assert _cell(items, (g_name, item), "b_se") == pytest.approx(expected, rel=1e-12)


def test_delta_se_mixed_models_and_oakes_equals_hessian() -> None:
    rng = np.random.default_rng(4005)
    n = 600
    theta = rng.normal(0.0, 1.0, n)
    p_bin = 1.0 / (1.0 + np.exp(-(np.outer(theta, [1.2, 1.0]) + np.array([0.3, -0.4]))))
    y_bin = (rng.random(p_bin.shape) < p_bin).astype(float)
    y_grm = np.searchsorted([-1.0, 0.1, 1.1], theta + rng.logistic(0, 1, n)).astype(float)
    y_gpcm = np.searchsorted([-0.7, 0.8], theta + rng.logistic(0, 1, n)).astype(float)
    y_pcm = np.searchsorted([-0.5, 0.6], theta + rng.logistic(0, 1, n)).astype(float)
    df = pd.DataFrame(
        {"b1_2pl": y_bin[:, 0], "b2_rasch": y_bin[:, 1], "g1": y_grm, "p1": y_gpcm, "p2": y_pcm}
    )
    spec = Model.from_items(
        {"b1_2pl": TwoPL(), "b2_rasch": Rasch(), "g1": GRM(4), "p1": GPCM(3), "p2": PCM(3)}
    )
    res_o = fit(spec, df, method=BAEM(tol=1e-8), se="oakes")
    res_h = fit(spec, df, method=BAEM(tol=1e-8), se="hessian")
    items = res_o.items.set_index("item")
    for col in ("b", "b1", "b2", "b3", "b_se", "b1_se", "b2_se", "b3_se"):
        assert col in items.columns

    # Dichotomous items have b_se finite and b1_se..b3_se NaN.
    for name in ("b1_2pl", "b2_rasch"):
        assert np.isfinite(_cell(items, name, "b_se"))
        assert items.loc[name, ["b1_se", "b2_se", "b3_se"]].isna().all()
    assert _cell(items, "b2_rasch", "b_se") == pytest.approx(
        _cell(items, "b2_rasch", "d_se"), rel=1e-12
    )
    # GRM(4) has b1_se..b3_se finite, b_se NaN.
    assert np.isnan(_cell(items, "g1", "b_se"))
    assert items.loc["g1", ["b1_se", "b2_se", "b3_se"]].notna().all()
    # GPCM(3) and PCM(3) have b1_se..b2_se finite, b_se and b3_se NaN.
    for name in ("p1", "p2"):
        assert np.isnan(_cell(items, name, "b_se"))
        assert items.loc[name, ["b1_se", "b2_se"]].notna().all()
        assert np.isnan(_cell(items, name, "b3_se"))
    assert _cell(items, "p2", "b1_se") == pytest.approx(_cell(items, "p2", "d1_se"), rel=1e-12)

    # Oakes and Hessian delta-method SEs agree closely.
    items_h = res_h.items.set_index("item")
    for col in ("b_se", "b1_se", "b2_se", "b3_se"):
        mask = items[col].notna()
        np.testing.assert_allclose(items.loc[mask, col], items_h.loc[mask, col], rtol=1e-6)


# --------------------------------------------------------------------------------------------
# FitResult.summary() and FitResult._repr_html_()
# --------------------------------------------------------------------------------------------


def test_summary_and_repr_html_single_group() -> None:
    case = load("v1_lsat7_2pl")
    res = fit(Model.uniform(TwoPL(), items=case.items), case.data)

    # __repr__ stays short and single-line.
    rep = repr(res)
    assert "\n" not in rep
    assert "FitResult(" in rep and "loglik=" in rep

    text = res.summary()
    html = res._repr_html_()

    for out in (text, html):
        assert "2PL" in out
        assert "MMLE" in out
        assert "1000" in out
        assert "5 items" in out
        assert "10 free parameters" in out
        assert "Converged" in out
        assert str(res.n_iter) in out
        assert f"{res.n_em_evaluations} EM evaluations" in out
        assert "oakes" in out
        assert "Log-likelihood" in out
        assert "Log-posterior" in out
        assert "AIC" in out
        assert "BIC" in out
        assert f"{res.loglik:.4f}" in out
        assert f"{res.aic:.4f}" in out
        assert f"{res.bic:.4f}" in out
        assert "Latent distribution" in out
        assert "Item parameters" in out
        assert "b_se" in out
        for name in case.items:
            assert name in out
        # No priors warning when no priors are active.
        assert "Priors are active" not in out

    # HTML structure: plain HTML/CSS, no external assets.
    assert "<table" in html and "</table>" in html
    assert "<style>" in html
    for forbidden in ("http://", "https://", "<script", "<link"):
        assert forbidden not in html.lower()


def test_summary_and_repr_html_map_priors_warning_and_no_extra_warning() -> None:
    case = load("v1_lsat7_2pl")
    spec = (
        Model.uniform(TwoPL(), items=case.items)
        .prior(param="a", dist=LogNormalPrior(0.0, 0.5))
        .prior(param="d", dist=NormalPrior(0.0, 2.0))
    )
    res = fit(spec, case.data)
    assert res.log_posterior != res.loglik

    # Rendering summary() and _repr_html_() should show the warning inline without emitting
    # duplicate FlexpyirtWarning runtime warnings.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FlexpyirtWarning)
        text = res.summary()
        html = res._repr_html_()
    assert len(caught) == 0

    for out in (text, html):
        assert "MAP" in out
        assert f"{res.loglik:.4f}" in out
        assert f"{res.log_posterior:.4f}" in out
        assert "Priors are active" in out
        assert "MAP estimate" in out


def test_summary_and_repr_html_multigroup_and_mixed() -> None:
    rng = np.random.default_rng(4006)
    n_g = 300
    th_r = rng.normal(0.0, 1.0, n_g)
    th_f = rng.normal(0.5, 1.1, n_g)
    cols = ["i1", "i2", "i3", "g1"]

    def sim(th: np.ndarray) -> np.ndarray:
        bin_p = 1.0 / (1.0 + np.exp(-(np.outer(th, [1.2, 1.4, 1.1]) + np.array([-0.3, 0.2, 0.5]))))
        y_bin = (rng.random(bin_p.shape) < bin_p).astype(float)
        y_grm = np.searchsorted([-0.9, 0.1, 1.1], th + rng.logistic(0, 1, len(th))).astype(float)
        return np.column_stack([y_bin, y_grm])

    df = pd.DataFrame(np.vstack([sim(th_r), sim(th_f)]), columns=cols)
    grp = np.repeat(["ref", "focal"], n_g)
    spec = (
        Model.from_items({"i1": TwoPL(), "i2": TwoPL(), "i3": TwoPL(), "g1": GRM(4)})
        .with_groups(ref=Normal.standard(), focal=Normal.free())
        .equal_across_groups(items=["i1", "i2"])
    )
    res = fit(spec, df, group=grp, quadrature=Quadrature.equispaced(81, bounds=(-8, 8)))
    for out in (res.summary(), res._repr_html_()):
        assert "2PL (3)" in out
        assert "GRM(4) (1)" in out
        assert "2 (ref, focal)" in out
        assert "ref" in out and "focal" in out
        assert "b1_se" in out


def test_summary_and_repr_html_squarem_nonconverged_se_none_and_digits() -> None:
    case = load("v1_lsat7_2pl")
    spec = Model.uniform(TwoPL(), items=case.items)

    # SQUAREM: reports EM evaluations distinct from iterations.
    res_sq = fit(spec, case.data, method=BAEM(tol=1e-6, squarem=True), se=None)
    assert res_sq.n_em_evaluations > res_sq.n_iter
    for out in (res_sq.summary(), res_sq._repr_html_()):
        assert f"{res_sq.n_em_evaluations} EM evaluations" in out
        assert "None" in out
        assert "NaN" in out

    # Non-converged fit.
    with pytest.warns(ConvergenceWarning):
        res_nc = fit(spec, case.data, method=BAEM(max_iter=1), se=None)
    assert not res_nc.converged
    for out in (res_nc.summary(), res_nc._repr_html_()):
        assert "NOT converged" in out

    # Custom digits and validation.
    s2 = res_sq.summary(digits=2)
    assert f"{res_sq.loglik:.2f}" in s2
    with pytest.raises(SpecificationError):
        res_sq.summary(digits=-1)
    with pytest.raises(SpecificationError):
        res_sq.summary(digits=True)


def test_repr_html_escapes_special_characters() -> None:
    case = load("v1_lsat7_2pl")
    tricky = ['q<1>&"a"', "q2", "q3", "q4", "q5"]
    df = case.data.copy()
    df.columns = tricky
    # Alternate rows (case.data is grouped by response pattern) so both groups see all patterns.
    grp = np.tile(["g<ref>&1", "g<foc>&2"], len(df) // 2)
    spec = (
        Model.uniform(TwoPL(), items=tricky)
        .with_groups(**{"g<ref>&1": Normal.standard(), "g<foc>&2": Normal.free()})
        .equal_across_groups(items=tricky[:3])
    )
    res = fit(spec, df, group=grp, se=None, quadrature=Quadrature.equispaced(81, bounds=(-8, 8)))
    html_out = res._repr_html_()
    assert 'q<1>&"a"' not in html_out
    assert "q&lt;1&gt;&amp;&quot;a&quot;" in html_out
    assert "g<ref>&1" not in html_out
    assert "g&lt;ref&gt;&amp;1" in html_out
