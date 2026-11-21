"""Tests for FitResult JSON serialization and deserialization (to_json / from_json)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import warnings
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import jax
import numpy as np
import pandas as pd
import pytest

import flexpyirt
from flexpyirt import (
    BAEM,
    GPCM,
    GRM,
    PCM,
    Beta,
    FitResult,
    FourPL,
    LogNormal,
    Model,
    Normal,
    NormalPrior,
    Quadrature,
    Rasch,
    ThreePL,
    TwoPL,
    fit,
)
from flexpyirt._json import FORMAT_TAG, SCHEMA_VERSION
from flexpyirt.errors import ConvergenceWarning, FlexpyirtWarning, SpecificationError
from tests._reference import load


def _reject_constant(token: str) -> Any:
    raise ValueError(f"Non-standard JSON constant {token!r}")


def _assert_strict_json(text: str) -> dict[str, Any]:
    parsed = json.loads(text, parse_constant=_reject_constant)
    assert isinstance(parsed, dict)
    assert parsed["format"] == FORMAT_TAG
    assert parsed["schema_version"] == SCHEMA_VERSION
    assert parsed["flexpyirt_version"] == flexpyirt.__version__
    return parsed


def _assert_fit_result_identical(
    orig: FitResult,
    loaded: FitResult,
    score_data: pd.DataFrame,
    *,
    group: Any = None,
) -> None:
    """Assert ``loaded`` reproduces every attribute, summary, and scoring method of ``orig``."""
    # Scalar and metadata attributes
    assert loaded.loglik == orig.loglik
    assert loaded.log_posterior == orig.log_posterior
    assert loaded.converged is orig.converged
    assert loaded.n_iter == orig.n_iter
    assert loaded.n_em_evaluations == orig.n_em_evaluations
    assert loaded.n_safeguards == orig.n_safeguards
    assert loaded.n_params == orig.n_params
    assert loaded.n_obs == orig.n_obs
    assert loaded.se_method == orig.se_method
    assert loaded._model == orig._model
    assert loaded._quadrature == orig._quadrature
    assert loaded._method == orig._method
    np.testing.assert_array_equal(loaded._eta, orig._eta)

    # AIC / BIC (with warning behavior under MAP)
    if orig._has_priors:
        with pytest.warns(FlexpyirtWarning):
            orig_aic = orig.aic
        with pytest.warns(FlexpyirtWarning):
            load_aic = loaded.aic
        with pytest.warns(FlexpyirtWarning):
            orig_bic = orig.bic
        with pytest.warns(FlexpyirtWarning):
            load_bic = loaded.bic
    else:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", FlexpyirtWarning)
            orig_aic, load_aic = orig.aic, loaded.aic
            orig_bic, load_bic = orig.bic, loaded.bic
        assert len(caught) == 0
    assert load_aic == orig_aic
    assert load_bic == orig_bic

    # Covariance matrix
    if orig.vcov is None:
        assert loaded.vcov is None
    else:
        assert loaded.vcov is not None
        assert loaded.vcov.dtype == np.float64
        np.testing.assert_array_equal(loaded.vcov, orig.vcov)

    # Tables (exact values, dtypes, column order, and index)
    pd.testing.assert_frame_equal(loaded.parameters, orig.parameters, check_exact=True)
    pd.testing.assert_frame_equal(loaded.items, orig.items, check_exact=True)
    pd.testing.assert_frame_equal(loaded.latent, orig.latent, check_exact=True)
    pd.testing.assert_frame_equal(loaded.history, orig.history, check_exact=True)
    if orig.latent_weights is None:
        assert loaded.latent_weights is None
    else:
        assert loaded.latent_weights is not None
        pd.testing.assert_frame_equal(loaded.latent_weights, orig.latent_weights, check_exact=True)

    # Text / HTML representations
    assert repr(loaded) == repr(orig)
    assert loaded.summary() == orig.summary()
    assert loaded.summary(digits=2) == orig.summary(digits=2)
    assert loaded._repr_html_() == orig._repr_html_()

    # All supported scoring methods (MAP is Normal-only)
    det_methods: tuple[Literal["EAP", "MAP", "ML", "WLE"], ...] = (
        ("EAP", "ML", "WLE") if orig._model.is_eh else ("EAP", "MAP", "ML", "WLE")
    )
    for method in det_methods:
        s_orig = orig.score(score_data, method=method, group=group)
        s_load = loaded.score(score_data, method=method, group=group)
        assert list(s_load.columns) == ["theta", "se"]
        pd.testing.assert_index_equal(s_load.index, s_orig.index)
        np.testing.assert_array_equal(s_load["theta"].to_numpy(), s_orig["theta"].to_numpy())
        np.testing.assert_array_equal(s_load["se"].to_numpy(), s_orig["se"].to_numpy())

    key = jax.random.key(42)
    pv_orig = orig.score(score_data, method="PV", group=group, key=key, n_draws=4)
    pv_load = loaded.score(score_data, method="PV", group=group, key=key, n_draws=4)
    assert list(pv_load.columns) == list(pv_orig.columns)
    pd.testing.assert_index_equal(pv_load.index, pv_orig.index)
    np.testing.assert_array_equal(pv_load.to_numpy(), pv_orig.to_numpy())


def _roundtrip(
    res: FitResult,
    tmp_path: Path,
    name: str,
    *,
    indent: int | None = 2,
) -> tuple[FitResult, FitResult, str]:
    """Round-trip ``res`` through both a file path and an in-memory JSON string."""
    out_path = tmp_path / f"{name}.json"
    text = res.to_json(out_path, indent=indent)
    assert out_path.read_text(encoding="utf-8") == text
    _assert_strict_json(text)
    from_file = FitResult.from_json(out_path)
    from_str = FitResult.from_json(text)
    return from_file, from_str, text


# --------------------------------------------------------------------------------------------
# Representative fit round-trips
# --------------------------------------------------------------------------------------------


def test_roundtrip_2pl_oakes_and_hessian(tmp_path: Path) -> None:
    case = load("v1_lsat7_2pl")
    spec = Model.uniform(TwoPL(), items=case.items)
    res = fit(spec, case.data, method=BAEM(tol=1e-6), se="oakes")
    from_file, from_str, _ = _roundtrip(res, tmp_path, "2pl_oakes")
    score_sub = case.data.iloc[:25]
    _assert_fit_result_identical(res, from_file, score_sub)
    _assert_fit_result_identical(res, from_str, score_sub)

    # Also test loading via str path and compact indent=None with se="hessian"
    res_h = fit(spec, case.data, method=BAEM(tol=1e-6), se="hessian")
    compact_path = tmp_path / "2pl_hessian.json"
    compact_text = res_h.to_json(str(compact_path), indent=None)
    assert "\n" not in compact_text
    _assert_strict_json(compact_text)
    loaded_h = FitResult.from_json(str(compact_path))
    _assert_fit_result_identical(res_h, loaded_h, score_sub)


def test_roundtrip_rasch_free_variance(tmp_path: Path) -> None:
    case = load("v2_lsat7_rasch")
    spec = Model.uniform(Rasch(), items=case.items)
    res = fit(spec, case.data, method=BAEM(tol=1e-6))
    assert res._model.latent is None  # default free variance rule preserved
    from_file, from_str, _ = _roundtrip(res, tmp_path, "rasch")
    score_sub = case.data.iloc[:25]
    _assert_fit_result_identical(res, from_file, score_sub)
    _assert_fit_result_identical(res, from_str, score_sub)


def test_roundtrip_grm_gpcm_pcm_and_mixed(tmp_path: Path) -> None:
    rng = np.random.default_rng(5001)
    n = 600
    theta = rng.normal(0.0, 1.0, n)
    cuts4 = np.array([-1.1, 0.1, 1.2])
    y4 = np.searchsorted(cuts4, np.outer(theta, [1.2, 0.9, 1.5, 1.1]) + rng.logistic(0, 1, (n, 4)))
    df4 = pd.DataFrame(y4.astype(float), columns=["s1", "s2", "s3", "s4"])
    score_sub = df4.iloc[:25]

    for label, item_model in (("grm", GRM(4)), ("gpcm", GPCM(4)), ("pcm", PCM(4))):
        res = fit(Model.uniform(item_model, items=df4.columns), df4, method=BAEM(tol=1e-6))
        from_file, from_str, _ = _roundtrip(res, tmp_path, label)
        _assert_fit_result_identical(res, from_file, score_sub)
        _assert_fit_result_identical(res, from_str, score_sub)

    # Mixed item models in one Model (2PL, Rasch, GRM(4), GPCM(3), PCM(3))
    p_bin = 1.0 / (1.0 + np.exp(-(np.outer(theta, [1.2, 1.0]) + np.array([0.3, -0.4]))))
    y_bin = (rng.random(p_bin.shape) < p_bin).astype(float)
    y_grm = np.searchsorted([-1.0, 0.1, 1.1], theta + rng.logistic(0, 1, n)).astype(float)
    y_gpcm = np.searchsorted([-0.7, 0.8], theta + rng.logistic(0, 1, n)).astype(float)
    y_pcm = np.searchsorted([-0.5, 0.6], theta + rng.logistic(0, 1, n)).astype(float)
    df_mix = pd.DataFrame(
        {
            "i_2pl": y_bin[:, 0],
            "i_rasch": y_bin[:, 1],
            "i_grm": y_grm,
            "i_gpcm": y_gpcm,
            "i_pcm": y_pcm,
        }
    )
    spec_mix = Model.from_items(
        {"i_2pl": TwoPL(), "i_rasch": Rasch(), "i_grm": GRM(4), "i_gpcm": GPCM(3), "i_pcm": PCM(3)}
    )
    res_mix = fit(spec_mix, df_mix, method=BAEM(tol=1e-6))
    from_file, _, _ = _roundtrip(res_mix, tmp_path, "mixed")
    _assert_fit_result_identical(res_mix, from_file, df_mix.iloc[:25])


def test_roundtrip_3pl_4pl_and_priors_map(tmp_path: Path) -> None:
    rng = np.random.default_rng(5002)
    theta = rng.normal(0.0, 1.0, 1500)
    a = np.array([1.7, 1.8, 1.6, 1.9, 1.7, 1.8, 1.6, 1.7])
    d = np.array([-0.8, -0.5, -1.0, -0.6, -0.7, -0.9, -0.4, -0.8])
    g = np.full(len(a), 0.18)
    u = np.full(len(a), 0.88)
    p = g + (u - g) / (1.0 + np.exp(-(np.outer(theta, a) + d)))
    cols = [f"q{j}" for j in range(1, len(a) + 1)]
    df = pd.DataFrame((rng.random(p.shape) < p).astype(float), columns=cols)
    score_sub = df.iloc[:25]

    # 3PL with all three prior families (LogNormal, NormalPrior, Beta)
    spec3 = (
        Model.uniform(ThreePL(), items=cols)
        .prior(param="a", dist=LogNormal(0.4, 0.3))
        .prior(param="d", dist=NormalPrior(-0.7, 1.5))
        .prior(param="g", dist=Beta(5.0, 20.0))
    )
    res3 = fit(spec3, df, method=BAEM(tol=1e-6))
    assert res3.log_posterior != res3.loglik
    from_file, from_str, _ = _roundtrip(res3, tmp_path, "3pl_map")
    _assert_fit_result_identical(res3, from_file, score_sub)
    _assert_fit_result_identical(res3, from_str, score_sub)

    # 4PL with interior priors on a, g, u
    spec4 = (
        Model.uniform(FourPL(), items=cols)
        .prior(param="a", dist=LogNormal(0.4, 0.3))
        .prior(param="g", dist=Beta(5.0, 20.0))
        .prior(param="u", dist=Beta(22.0, 3.0))
    )
    res4 = fit(spec4, df, method=BAEM(tol=1e-6))
    from_file4, from_str4, _ = _roundtrip(res4, tmp_path, "4pl_map")
    _assert_fit_result_identical(res4, from_file4, score_sub)
    _assert_fit_result_identical(res4, from_str4, score_sub)


def test_roundtrip_two_group_with_anchors(tmp_path: Path) -> None:
    rng = np.random.default_rng(5003)
    n_g = 500
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
        method=BAEM(tol=1e-6),
        quadrature=Quadrature.equispaced(81, lim=(-8, 8)),
    )
    from_file, from_str, _ = _roundtrip(res, tmp_path, "two_group")
    idx = [*range(15), *(n_g + i for i in range(15))]
    _assert_fit_result_identical(res, from_file, df.iloc[idx], group=grp[idx])
    _assert_fit_result_identical(res, from_str, df.iloc[idx], group=grp[idx])


def test_roundtrip_fipc(tmp_path: Path) -> None:
    case = load("v1_lsat7_2pl")
    bank = pd.DataFrame({"item": ["Item.1", "Item.2"], "a": [1.0, 1.1], "d": [1.8, 0.8]})
    spec = Model.uniform(TwoPL(), items=case.items).fix_from(bank).with_latent(Normal.free())
    res = fit(
        spec,
        case.data,
        method=BAEM(tol=1e-6),
        quadrature=Quadrature.equispaced(81, lim=(-8, 8)),
    )
    from_file, from_str, _ = _roundtrip(res, tmp_path, "fipc")
    _assert_fit_result_identical(res, from_file, case.data.iloc[:25])
    _assert_fit_result_identical(res, from_str, case.data.iloc[:25])


def test_roundtrip_missing_data_and_all_missing_scoring(tmp_path: Path) -> None:
    case = load("v1_lsat7_2pl")
    rng = np.random.default_rng(5004)
    df_miss = case.data.copy().astype(float)
    mask = rng.random(df_miss.shape) < 0.12
    # Keep at least one observed item per row during fitting
    mask[:, 0] &= ~mask.all(axis=1)
    df_miss[mask] = np.nan

    res = fit(Model.uniform(TwoPL(), items=case.items), df_miss, method=BAEM(tol=1e-6))
    from_file, from_str, _ = _roundtrip(res, tmp_path, "missing")

    # Include partial missing, extreme patterns (all-0 / all-1), and an all-missing row for scoring
    score_df = df_miss.iloc[:20].copy()
    score_df.iloc[0] = np.nan
    score_df.iloc[1] = 0.0
    score_df.iloc[2] = 1.0
    _assert_fit_result_identical(res, from_file, score_df)
    _assert_fit_result_identical(res, from_str, score_df)


def test_roundtrip_squarem_and_se_none(tmp_path: Path) -> None:
    case = load("v1_lsat7_2pl")
    spec = Model.uniform(TwoPL(), items=case.items)
    method = BAEM(tol=1e-7, max_iter=500, m_step_newton_steps=4, squarem=True)
    res_sq = fit(spec, case.data, method=method, se="oakes")
    assert res_sq.n_em_evaluations > res_sq.n_iter
    from_file, from_str, _ = _roundtrip(res_sq, tmp_path, "squarem")
    _assert_fit_result_identical(res_sq, from_file, case.data.iloc[:25])
    _assert_fit_result_identical(res_sq, from_str, case.data.iloc[:25])

    res_none = fit(spec, case.data, method=BAEM(tol=1e-6), se=None)
    assert res_none.vcov is None
    assert res_none.se_method is None
    from_file_n, from_str_n, _ = _roundtrip(res_none, tmp_path, "se_none")
    _assert_fit_result_identical(res_none, from_file_n, case.data.iloc[:25])
    _assert_fit_result_identical(res_none, from_str_n, case.data.iloc[:25])


# --------------------------------------------------------------------------------------------
# Edge cases: edited table, non-converged, non-PD vcov, all-fixed, special names, -0.0
# --------------------------------------------------------------------------------------------


def test_roundtrip_edited_parameter_table(tmp_path: Path) -> None:
    case = load("v1_lsat7_2pl")
    spec = Model.uniform(TwoPL(), items=case.items)
    table = spec.parameter_table(case.data)
    row_a = (table["item"] == "Item.2") & (table["param"] == "a")
    table.loc[row_a, "value"] = 1.25
    table.loc[row_a, "free"] = False
    row_d = (table["item"] == "Item.3") & (table["param"] == "d")
    table.loc[row_d, "lower"] = -5.0
    table.loc[row_d, "upper"] = 5.0
    table["prior"] = pd.Series(
        [NormalPrior(0.0, 2.0) if m else p for m, p in zip(row_d, table["prior"], strict=True)],
        dtype=object,
    )

    res = fit(spec, case.data, table=table, method=BAEM(tol=1e-6))
    from_file, _, _ = _roundtrip(res, tmp_path, "edited_table")
    _assert_fit_result_identical(res, from_file, case.data.iloc[:25])


def test_roundtrip_nonconverged_nonpd_vcov_and_all_fixed(tmp_path: Path) -> None:
    case = load("v1_lsat7_2pl")
    spec = Model.uniform(TwoPL(), items=case.items)

    # Non-converged fit
    with pytest.warns(ConvergenceWarning):
        res_nc = fit(spec, case.data, method=BAEM(max_iter=1), se="oakes")
    assert not res_nc.converged
    from_nc, _, _ = _roundtrip(res_nc, tmp_path, "nonconverged")
    _assert_fit_result_identical(res_nc, from_nc, case.data.iloc[:15])

    # Non-PD vcov (all NaN matrix) -> strict JSON still holds, shape and NaNs preserved
    assert res_nc.vcov is not None
    res_nan_vcov = replace(
        res_nc,
        vcov=np.full_like(res_nc.vcov, np.nan),
        parameters=res_nc.parameters.assign(se=np.nan),
    )
    from_nan, _, text_nan = _roundtrip(res_nan_vcov, tmp_path, "nan_vcov")
    assert '"NaN"' in text_nan
    assert from_nan.vcov is not None
    assert from_nan.vcov.shape == res_nc.vcov.shape
    assert np.isnan(from_nan.vcov).all()
    _assert_fit_result_identical(res_nan_vcov, from_nan, case.data.iloc[:15])

    # All parameters fixed (n_free == 0)
    spec_all_fixed = spec
    for item in case.items:
        spec_all_fixed = spec_all_fixed.fix(item=item, a=1.0, d=0.0)
    res_zero = fit(spec_all_fixed, case.data, se="oakes")
    assert res_zero.n_params == 0
    assert res_zero.vcov is None
    assert res_zero.se_method == "oakes"
    from_zero, _, _ = _roundtrip(res_zero, tmp_path, "all_fixed")
    _assert_fit_result_identical(res_zero, from_zero, case.data.iloc[:15])


def test_roundtrip_special_item_and_group_names(tmp_path: Path) -> None:
    case = load("v1_lsat7_2pl")
    # Include string names that coincide with special float tokens, HTML chars, and Unicode.
    tricky = ["NaN", "Infinity", "-Infinity", 'q<1>&"a"', "項目5"]
    df = case.data.copy()
    df.columns = tricky
    grp = np.tile(["g<ref>&1", "NaN"], len(df) // 2)
    spec = (
        Model.uniform(TwoPL(), items=tricky)
        .with_groups(**{"g<ref>&1": Normal.standard(), "NaN": Normal.free()})
        .equal_across_groups(items=tricky[:3])
    )
    res = fit(spec, df, group=grp, se=None, quadrature=Quadrature.equispaced(81, lim=(-8, 8)))
    from_file, from_str, _ = _roundtrip(res, tmp_path, "special_names")
    _assert_fit_result_identical(res, from_file, df.iloc[:20], group=grp[:20])
    _assert_fit_result_identical(res, from_str, df.iloc[:20], group=grp[:20])


# --------------------------------------------------------------------------------------------
# Precision check on from_json (check_x64)
# --------------------------------------------------------------------------------------------


def test_from_json_raises_precision_error_without_x64(tmp_path: Path) -> None:
    case = load("v1_lsat7_2pl")
    res = fit(Model.uniform(TwoPL(), items=case.items), case.data, se=None)
    path = tmp_path / "fit.json"
    res.to_json(path)

    code = textwrap.dedent(
        f"""
        import jax
        from flexpyirt import FitResult
        from flexpyirt.errors import PrecisionError

        assert not jax.config.jax_enable_x64
        try:
            FitResult.from_json({str(path)!r})
        except PrecisionError as exc:
            assert "jax_enable_x64" in str(exc)
        else:
            raise SystemExit("PrecisionError not raised by FitResult.from_json")
        """
    )
    env = {**os.environ, "JAX_ENABLE_X64": "0", "JAX_PLATFORMS": "cpu"}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


# --------------------------------------------------------------------------------------------
# Malformed files, strict JSON rejections, and invalid inputs
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_payload() -> dict[str, Any]:
    case = load("v1_lsat7_2pl")
    spec = Model.uniform(TwoPL(), items=case.items).prior(param="a", dist=LogNormal(0.0, 0.5))
    res = fit(spec, case.data, method=BAEM(tol=1e-5), se="oakes")
    return _assert_strict_json(res.to_json())


def _clone(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = json.loads(json.dumps(payload, allow_nan=False))
    return out


def test_malformed_format_and_schema_version(
    sample_payload: dict[str, Any], tmp_path: Path
) -> None:
    # Wrong format tag (both via string and via file)
    bad_fmt = _clone(sample_payload)
    bad_fmt["format"] = "other.Result"
    bad_path = tmp_path / "bad_fmt.json"
    bad_path.write_text(json.dumps(bad_fmt), encoding="utf-8")
    with pytest.raises(SpecificationError, match="Unsupported format"):
        FitResult.from_json(bad_path)

    # Missing format tag
    no_fmt = _clone(sample_payload)
    del no_fmt["format"]
    with pytest.raises(SpecificationError, match="Unsupported format"):
        FitResult.from_json(json.dumps(no_fmt))

    # Unknown schema_version
    for bad_ver in (0, 4, 99, "1", True):
        bad = _clone(sample_payload)
        bad["schema_version"] = bad_ver
        with pytest.raises(SpecificationError, match="schema_version"):
            FitResult.from_json(json.dumps(bad))

    # Missing schema_version
    no_ver = _clone(sample_payload)
    del no_ver["schema_version"]
    with pytest.raises(SpecificationError, match="schema_version"):
        FitResult.from_json(json.dumps(no_ver))


def test_malformed_registry_types(sample_payload: dict[str, Any]) -> None:
    # Unknown item model
    bad_item = _clone(sample_payload)
    bad_item["model"]["items"][0]["model"] = {"type": "5PL"}
    with pytest.raises(SpecificationError, match="Unknown item model type '5PL'"):
        FitResult.from_json(json.dumps(bad_item))

    # Invalid polytomous n_cat
    bad_ncat = _clone(sample_payload)
    bad_ncat["model"]["items"][0]["model"] = {"type": "GRM", "n_cat": 1}
    with pytest.raises(SpecificationError, match="at least 2 categories"):
        FitResult.from_json(json.dumps(bad_ncat))

    bad_ncat_bool = _clone(sample_payload)
    bad_ncat_bool["model"]["items"][0]["model"] = {"type": "GRM", "n_cat": True}
    with pytest.raises(SpecificationError, match="Expected an integer"):
        FitResult.from_json(json.dumps(bad_ncat_bool))

    # Extra key on simple item model
    extra_item_key = _clone(sample_payload)
    extra_item_key["model"]["items"][0]["model"] = {"type": "2PL", "n_cat": 2}
    with pytest.raises(SpecificationError, match="Unexpected keys"):
        FitResult.from_json(json.dumps(extra_item_key))

    # Unknown latent distribution
    bad_lat = _clone(sample_payload)
    bad_lat["model"]["latent"] = {"type": "DavidianCurve"}
    with pytest.raises(SpecificationError, match="Unknown latent distribution type"):
        FitResult.from_json(json.dumps(bad_lat))

    # Unknown prior
    bad_pri = _clone(sample_payload)
    bad_pri["model"]["priors"][0]["dist"] = {"type": "Cauchy", "loc": 0.0, "scale": 1.0}
    with pytest.raises(SpecificationError, match="Unknown prior type 'Cauchy'"):
        FitResult.from_json(json.dumps(bad_pri))

    # Unknown quadrature
    bad_quad = _clone(sample_payload)
    bad_quad["quadrature"] = {"type": "quasi_monte_carlo", "n_points": 41}
    with pytest.raises(SpecificationError, match="Unknown quadrature type"):
        FitResult.from_json(json.dumps(bad_quad))

    # Unknown method
    bad_meth = _clone(sample_payload)
    bad_meth["method"] = {"type": "MHRM"}
    with pytest.raises(SpecificationError, match="Unknown estimation method type"):
        FitResult.from_json(json.dumps(bad_meth))


def test_malformed_json_and_nonstandard_constants(
    sample_payload: dict[str, Any], tmp_path: Path
) -> None:
    # Literal NaN / Infinity in JSON text must be rejected
    raw_nan = json.dumps(sample_payload).replace('"-Infinity"', "-Infinity", 1)
    with pytest.raises(SpecificationError, match="Non-standard JSON constant"):
        FitResult.from_json(raw_nan)

    raw_nan_file = tmp_path / "bare_nan.json"
    raw_nan_file.write_text('{"format": "flexpyirt.FitResult", "loglik": NaN}', encoding="utf-8")
    with pytest.raises(SpecificationError, match="Non-standard JSON constant"):
        FitResult.from_json(raw_nan_file)

    # Syntax errors and non-object top-level JSON
    for bad_text in ("{not json", "not json", "[]", "123", '"hello"'):
        with pytest.raises(SpecificationError):
            FitResult.from_json(bad_text)

    # Non-existent file paths raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        FitResult.from_json(tmp_path / "does_not_exist.json")
    with pytest.raises(FileNotFoundError):
        FitResult.from_json(str(tmp_path / "does_not_exist.json"))

    # Invalid indent in to_json
    case = load("v1_lsat7_2pl")
    res = fit(Model.uniform(TwoPL(), items=case.items), case.data, se=None)
    for bad_indent in (-1, True, "2"):
        with pytest.raises(SpecificationError, match="indent"):
            res.to_json(indent=bad_indent)  # type: ignore[arg-type]


def test_malformed_internal_consistency(sample_payload: dict[str, Any]) -> None:
    # Missing parameter row
    missing_row = _clone(sample_payload)
    missing_row["parameters"].pop()
    with pytest.raises(SpecificationError, match="Missing parameter rows"):
        FitResult.from_json(json.dumps(missing_row))

    # Non-contiguous free_id
    bad_fid = _clone(sample_payload)
    bad_fid["parameters"][0]["free_id"] = 999
    with pytest.raises(SpecificationError, match="contiguous"):
        FitResult.from_json(json.dumps(bad_fid))

    # Mismatched vcov shape
    bad_vcov = _clone(sample_payload)
    bad_vcov["vcov"] = bad_vcov["vcov"][:-1]
    with pytest.raises(SpecificationError, match="vcov must have shape"):
        FitResult.from_json(json.dumps(bad_vcov))

    # Mismatched history length
    bad_hist = _clone(sample_payload)
    bad_hist["history"] = bad_hist["history"][:-1]
    with pytest.raises(SpecificationError, match="history length"):
        FitResult.from_json(json.dumps(bad_hist))

    # Fixed row with estimate != value
    bad_fix = _clone(sample_payload)
    latent_row = next(r for r in bad_fix["parameters"] if not r["free"])
    latent_row["estimate"] = float(latent_row["value"]) + 1.0
    with pytest.raises(SpecificationError, match="estimate"):
        FitResult.from_json(json.dumps(bad_fix))

    # Unknown se_method
    bad_se = _clone(sample_payload)
    bad_se["se_method"] = "bootstrap"
    with pytest.raises(SpecificationError, match="Unknown se_method"):
        FitResult.from_json(json.dumps(bad_se))

    # Reserved sentinel item name "__latent__"
    bad_latent_item = _clone(sample_payload)
    bad_latent_item["model"]["items"][0]["name"] = "__latent__"
    for p in bad_latent_item["model"]["priors"]:
        if p["item"] == sample_payload["model"]["items"][0]["name"]:
            p["item"] = "__latent__"
    for row in bad_latent_item["parameters"]:
        if row["item"] == sample_payload["model"]["items"][0]["name"]:
            row["item"] = "__latent__"
    with pytest.raises(SpecificationError, match="reserved"):
        FitResult.from_json(json.dumps(bad_latent_item))


def test_malformed_parameter_invariants_and_se(sample_payload: dict[str, Any]) -> None:
    # Non-finite value or estimate (including NaN on both value and estimate of a fixed row)
    for token in ("NaN", "Infinity", "-Infinity"):
        bad_val = _clone(sample_payload)
        bad_val["parameters"][0]["value"] = token
        with pytest.raises(SpecificationError, match="must be finite"):
            FitResult.from_json(json.dumps(bad_val))

        bad_est = _clone(sample_payload)
        bad_est["parameters"][0]["estimate"] = token
        with pytest.raises(SpecificationError, match="must be finite"):
            FitResult.from_json(json.dumps(bad_est))

        bad_fixed_nan = _clone(sample_payload)
        lat_mean = next(
            r
            for r in bad_fixed_nan["parameters"]
            if r["item"] == "__latent__" and r["param"] == "mean"
        )
        lat_mean["value"] = token
        lat_mean["estimate"] = token
        with pytest.raises(SpecificationError, match="must be finite"):
            FitResult.from_json(json.dumps(bad_fixed_nan))

    # Non-positive latent variance (0.0 and -1.0)
    for bad_var in (0.0, -1.0):
        bad_lat = _clone(sample_payload)
        lat_var = next(
            r for r in bad_lat["parameters"] if r["item"] == "__latent__" and r["param"] == "var"
        )
        lat_var["value"] = bad_var
        lat_var["estimate"] = bad_var
        if bad_var < 0.0:
            lat_var["lower"] = -2.0
        with pytest.raises(SpecificationError, match=r"positive|exceed model domain"):
            FitResult.from_json(json.dumps(bad_lat))

    # Bounds exceeding model domain (__latent__.var lower < 0)
    bad_dom = _clone(sample_payload)
    lat_var = next(
        r for r in bad_dom["parameters"] if r["item"] == "__latent__" and r["param"] == "var"
    )
    lat_var["lower"] = -0.5
    with pytest.raises(SpecificationError, match="exceed model domain"):
        FitResult.from_json(json.dumps(bad_dom))

    # Free estimate outside [lower, upper] or prior.support
    bad_bounds = _clone(sample_payload)
    bad_bounds["parameters"][1]["lower"] = -1.0
    bad_bounds["parameters"][1]["upper"] = 1.0
    bad_bounds["parameters"][1]["value"] = 0.0
    bad_bounds["parameters"][1]["estimate"] = 1.0
    with pytest.raises(SpecificationError, match="not strictly inside"):
        FitResult.from_json(json.dumps(bad_bounds))

    bad_prior_est = _clone(sample_payload)
    # parameters[0] is Item.1.a with LogNormal prior -> support (0, inf)
    bad_prior_est["parameters"][0]["estimate"] = -0.5
    with pytest.raises(SpecificationError, match="outside the support"):
        FitResult.from_json(json.dumps(bad_prior_est))

    # Invalid se: finite on fixed row; negative or infinite on free row
    bad_fix_se = _clone(sample_payload)
    lat_row = next(r for r in bad_fix_se["parameters"] if not r["free"])
    lat_row["se"] = 0.1
    with pytest.raises(SpecificationError, match="se == NaN"):
        FitResult.from_json(json.dumps(bad_fix_se))

    for bad_se_val in (-1.0, "-Infinity", "Infinity"):
        bad_free_se = _clone(sample_payload)
        bad_free_se["parameters"][0]["se"] = bad_se_val
        with pytest.raises(SpecificationError, match="finite non-negative or NaN se"):
            FitResult.from_json(json.dumps(bad_free_se))


def test_malformed_asymptotes_grm_order_and_shared_bounds() -> None:
    # 3PL asymptote invariant 0 <= g < u <= 1 on value and estimate
    rng = np.random.default_rng(5010)
    theta = rng.normal(0.0, 1.0, 400)
    a = np.full(5, 1.5)
    d = np.array([-0.5, -0.2, 0.0, 0.2, 0.5])
    p = 0.15 + 0.85 / (1.0 + np.exp(-(np.outer(theta, a) + d)))
    cols = [f"q{j}" for j in range(1, 6)]
    df3 = pd.DataFrame((rng.random(p.shape) < p).astype(float), columns=cols)
    res3 = fit(
        Model.uniform(ThreePL(), items=cols)
        .prior(param="a", dist=LogNormal(0.4, 0.3))
        .prior(param="g", dist=Beta(5.0, 20.0)),
        df3,
        method=BAEM(tol=1e-4),
        se=None,
    )
    payload3 = _assert_strict_json(res3.to_json())

    bad_g_est = _clone(payload3)
    g_row = next(r for r in bad_g_est["parameters"] if r["item"] == "q1" and r["param"] == "g")
    u_row = next(r for r in bad_g_est["parameters"] if r["item"] == "q1" and r["param"] == "u")
    u_row["value"] = 0.5
    u_row["estimate"] = 0.5
    g_row["value"] = 0.2
    g_row["estimate"] = 0.6
    with pytest.raises(SpecificationError, match="Lower asymptote"):
        FitResult.from_json(json.dumps(bad_g_est))

    bad_g_dom = _clone(payload3)
    g_row_dom = next(r for r in bad_g_dom["parameters"] if r["item"] == "q1" and r["param"] == "g")
    g_row_dom["lower"] = -0.1
    with pytest.raises(SpecificationError, match="exceed model domain"):
        FitResult.from_json(json.dumps(bad_g_dom))

    # GRM intercept ordering d1 > d2 > d3 on estimate and value
    cuts4 = np.array([-1.0, 0.0, 1.0])
    y4 = np.searchsorted(
        cuts4, np.outer(theta, [1.2, 1.0, 1.3, 1.1]) + rng.logistic(0, 1, (400, 4))
    )
    df_grm = pd.DataFrame(y4.astype(float), columns=["g1", "g2", "g3", "g4"])
    spec_grm = Model.uniform(GRM(4), items=df_grm.columns)
    res_grm = fit(spec_grm, df_grm, method=BAEM(tol=1e-4), se=None)
    payload_grm = _assert_strict_json(res_grm.to_json())

    bad_grm_est = _clone(payload_grm)
    d1 = next(r for r in bad_grm_est["parameters"] if r["item"] == "g1" and r["param"] == "d1")
    d2 = next(r for r in bad_grm_est["parameters"] if r["item"] == "g1" and r["param"] == "d2")
    d1["estimate"] = d2["estimate"]
    with pytest.raises(SpecificationError, match="GRM intercepts"):
        FitResult.from_json(json.dumps(bad_grm_est))

    bad_grm_val = _clone(payload_grm)
    d1_v = next(r for r in bad_grm_val["parameters"] if r["item"] == "g1" and r["param"] == "d1")
    d2_v = next(r for r in bad_grm_val["parameters"] if r["item"] == "g1" and r["param"] == "d2")
    d1_v["value"], d2_v["value"] = d2_v["value"], d1_v["value"]
    with pytest.raises(SpecificationError, match="GRM intercepts"):
        FitResult.from_json(json.dumps(bad_grm_val))

    # validate_table in fit(..., table=...) also rejects disordered GRM starts
    tbl_grm = spec_grm.parameter_table(df_grm)
    m1 = (tbl_grm["item"] == "g1") & (tbl_grm["param"] == "d1")
    m2 = (tbl_grm["item"] == "g1") & (tbl_grm["param"] == "d2")
    tbl_grm.loc[m1, "value"] = -2.0
    tbl_grm.loc[m2, "value"] = 2.0
    with pytest.raises(SpecificationError, match="GRM intercepts"):
        fit(spec_grm, df_grm, table=tbl_grm, se=None)

    # Two-group shared free_id: conflicting bounds and conflicting se
    n_g = 200
    grp = np.repeat(["ref", "focal"], n_g)
    spec_2g = (
        Model.uniform(TwoPL(), items=cols)
        .with_groups(ref=Normal.standard(), focal=Normal.free())
        .equal_across_groups(items=["q1", "q2", "q3"])
    )
    res_2g = fit(
        spec_2g,
        df3,
        group=grp,
        method=BAEM(tol=1e-4),
        se="oakes",
        quadrature=Quadrature.equispaced(41, lim=(-6, 6)),
    )
    payload_2g = _assert_strict_json(res_2g.to_json())

    bad_shared_bounds = _clone(payload_2g)
    q1a_foc = next(
        r
        for r in bad_shared_bounds["parameters"]
        if r["group"] == "focal" and r["item"] == "q1" and r["param"] == "a"
    )
    q1a_foc["lower"] = -10.0
    with pytest.raises(SpecificationError, match="conflicting bounds"):
        FitResult.from_json(json.dumps(bad_shared_bounds))

    bad_shared_se = _clone(payload_2g)
    q1a_foc_se = next(
        r
        for r in bad_shared_se["parameters"]
        if r["group"] == "focal" and r["item"] == "q1" and r["param"] == "a"
    )
    q1a_foc_se["se"] = float(q1a_foc_se["se"]) + 0.5
    with pytest.raises(SpecificationError, match="conflicting standard errors"):
        FitResult.from_json(json.dumps(bad_shared_se))

    # Reversed parameter rows in JSON restore canonical (group, item, param) order
    rev_payload = _clone(payload_2g)
    rev_payload["parameters"] = list(reversed(rev_payload["parameters"]))
    loaded_rev = FitResult.from_json(json.dumps(rev_payload))
    assert list(loaded_rev.latent["group"]) == ["ref", "focal"]
    assert list(loaded_rev.items.columns) == list(res_2g.items.columns)
    idx = [*range(10), *(n_g + i for i in range(10))]
    _assert_fit_result_identical(res_2g, loaded_rev, df3.iloc[idx], group=grp[idx])


def test_malformed_overflow_and_unbounded_ints(sample_payload: dict[str, Any]) -> None:
    # Overflowing decimal exponents (1e400 / -1e400) must raise SpecificationError, not become inf
    base_text = json.dumps(sample_payload)
    ll_token = json.dumps(sample_payload["loglik"])
    for raw_num in ("1e400", "-1e400"):
        bad_exp = base_text.replace(f'"loglik": {ll_token}', f'"loglik": {raw_num}', 1)
        with pytest.raises(SpecificationError, match="overflows float64"):
            FitResult.from_json(bad_exp)

    # Huge integers (10**400, 10**5000) in float or int fields must raise SpecificationError
    for huge_digits in ("1" + "0" * 400, "1" + "0" * 5000):
        bad_ll_int = base_text.replace(f'"loglik": {ll_token}', f'"loglik": {huge_digits}', 1)
        with pytest.raises(SpecificationError, match="64-bit signed range"):
            FitResult.from_json(bad_ll_int)

        bad_nobs = base_text.replace(
            f'"n_obs": {sample_payload["n_obs"]}', f'"n_obs": {huge_digits}', 1
        )
        with pytest.raises(SpecificationError, match="64-bit signed range"):
            FitResult.from_json(bad_nobs)

    # Huge integer in history iter / em_evaluations (int64 / int32 overflow)
    bad_hist_iter = _clone(sample_payload)
    bad_hist_iter["history"][0]["iter"] = 10**100
    with pytest.raises(SpecificationError, match="64-bit signed range"):
        FitResult.from_json(json.dumps(bad_hist_iter))

    bad_hist_eval = _clone(sample_payload)
    bad_hist_eval["history"][0]["em_evaluations"] = 10**12
    with pytest.raises(SpecificationError, match="em_evaluations must be <="):
        FitResult.from_json(json.dumps(bad_hist_eval))

    # Huge polytomous n_cat must fail immediately without OOM/hang
    for model_type in ("GRM", "GPCM", "PCM"):
        for bad_n_cat in (10**9, 1025):
            bad_poly = _clone(sample_payload)
            bad_poly["model"]["items"][0]["model"] = {"type": model_type, "n_cat": bad_n_cat}
            with pytest.raises(SpecificationError, match="n_cat must be <="):
                FitResult.from_json(json.dumps(bad_poly))

    # Huge quadrature n_points must fail at load time (not MemoryError in score())
    for bad_pts in (10**18, 100_001):
        bad_quad = _clone(sample_payload)
        bad_quad["quadrature"]["n_points"] = bad_pts
        with pytest.raises(SpecificationError, match="n_points must be <="):
            FitResult.from_json(json.dumps(bad_quad))

    # Documented string-encoded non-finite floats remain accepted on non-finite slots
    ok_nonfinite = _clone(sample_payload)
    ok_nonfinite["loglik"] = "-Infinity"
    ok_nonfinite["log_posterior"] = "-Infinity"
    loaded_inf = FitResult.from_json(json.dumps(ok_nonfinite))
    assert loaded_inf.loglik == -np.inf
    assert loaded_inf.log_posterior == -np.inf


def test_malformed_cross_field_consistency(sample_payload: dict[str, Any]) -> None:
    # se_method=None with non-null vcov (reviewer scenario: NaN se with finite delta b_se)
    bad_none_vcov = _clone(sample_payload)
    bad_none_vcov["se_method"] = None
    for row in bad_none_vcov["parameters"]:
        row["se"] = "NaN"
    with pytest.raises(SpecificationError, match="vcov must be null when se_method is null"):
        FitResult.from_json(json.dumps(bad_none_vcov))

    # se_method="oakes" with vcov=None when n_free > 0
    bad_missing_vcov = _clone(sample_payload)
    bad_missing_vcov["vcov"] = None
    with pytest.raises(SpecificationError, match="vcov cannot be null"):
        FitResult.from_json(json.dumps(bad_missing_vcov))

    # se_method=None, vcov=None, but finite se left in parameters
    bad_se_when_null = _clone(sample_payload)
    bad_se_when_null["se_method"] = None
    bad_se_when_null["vcov"] = None
    with pytest.raises(SpecificationError, match="standard errors must be NaN"):
        FitResult.from_json(json.dumps(bad_se_when_null))

    # n_free == 0 with non-null (empty []) vcov
    all_fixed = _clone(sample_payload)
    for row in all_fixed["parameters"]:
        row["free"] = False
        row["free_id"] = -1
        row["estimate"] = row["value"]
        row["se"] = "NaN"
    all_fixed["vcov"] = []
    with pytest.raises(SpecificationError, match="no free parameters"):
        FitResult.from_json(json.dumps(all_fixed))

    # History iter sequence must be 1..n_iter
    bad_iter_seq = _clone(sample_payload)
    bad_iter_seq["history"][0]["iter"] = 2
    with pytest.raises(SpecificationError, match=r"history\[0\]\.iter must be 1"):
        FitResult.from_json(json.dumps(bad_iter_seq))

    # sum(history.safeguard) must match n_safeguards
    bad_safe_sum = _clone(sample_payload)
    bad_safe_sum["n_safeguards"] = int(bad_safe_sum["n_safeguards"]) + 1
    with pytest.raises(SpecificationError, match="n_safeguards"):
        FitResult.from_json(json.dumps(bad_safe_sum))

    # history.em_evaluations must be non-decreasing
    assert len(sample_payload["history"]) >= 2
    bad_eval_dec = _clone(sample_payload)
    bad_eval_dec["history"][1]["em_evaluations"] = 0
    with pytest.raises(SpecificationError, match="non-decreasing"):
        FitResult.from_json(json.dumps(bad_eval_dec))

    # Final history.em_evaluations must match n_em_evaluations
    bad_eval_end = _clone(sample_payload)
    bad_eval_end["n_em_evaluations"] = int(bad_eval_end["n_em_evaluations"]) + 5
    with pytest.raises(SpecificationError, match="n_em_evaluations"):
        FitResult.from_json(json.dumps(bad_eval_end))

    # n_iter == 0 requires n_em_evaluations == 0
    bad_zero_iter = _clone(sample_payload)
    bad_zero_iter["n_iter"] = 0
    bad_zero_iter["history"] = []
    bad_zero_iter["n_safeguards"] = 0
    bad_zero_iter["n_em_evaluations"] = 1
    with pytest.raises(SpecificationError, match="n_em_evaluations must be 0"):
        FitResult.from_json(json.dumps(bad_zero_iter))


def test_read_source_non_utf8_and_brace_prefixed_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Binary / non-UTF-8 file raises SpecificationError (never UnicodeDecodeError)
    bin_path = tmp_path / "not_utf8.json"
    bin_path.write_bytes(b"\x80\x04\x95\x05\x00\x00\x00\xff\xfe")
    with pytest.raises(SpecificationError, match="not valid UTF-8"):
        FitResult.from_json(bin_path)
    with pytest.raises(SpecificationError, match="not valid UTF-8"):
        FitResult.from_json(str(bin_path))

    # Relative filename starting with '{' round-trips through to_json / from_json
    case = load("v1_lsat7_2pl")
    res = fit(Model.uniform(TwoPL(), items=case.items), case.data, se=None)
    monkeypatch.chdir(tmp_path)
    res.to_json("{fit}.json")
    loaded = FitResult.from_json("{fit}.json")
    _assert_fit_result_identical(res, loaded, case.data.iloc[:15])

    # Non-existent relative '{...}.json' still raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        FitResult.from_json("{missing}.json")


# --------------------------------------------------------------------------------------------
# Schema version 2: confirmatory multidimensional models (DESIGN.md §15.33)
# --------------------------------------------------------------------------------------------


def _mirt_data(n_factors: int, seed: int) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    rng = np.random.default_rng(seed)
    n, per = 600, 4
    j = n_factors * per
    theta = rng.multivariate_normal(np.zeros(n_factors), 0.4 + 0.6 * np.eye(n_factors), size=n)
    a = np.zeros((j, n_factors))
    for m in range(n_factors):
        a[m * per : (m + 1) * per, m] = rng.uniform(0.8, 1.6, per)
    a[per, 0] = 0.6  # one cross-loading
    d = rng.normal(size=j)
    y = (rng.random((n, j)) < 1.0 / (1.0 + np.exp(-(theta @ a.T + d)))).astype(float)
    items = [f"q{k:02d}" for k in range(j)]
    factors = {f"F{m + 1}": items[m * per : (m + 1) * per] for m in range(n_factors)}
    factors["F1"] = [*factors["F1"], items[per]]
    return pd.DataFrame(y, columns=items), factors


@pytest.mark.parametrize(("n_factors", "correlated"), [(2, True), (3, False)])
def test_multidimensional_round_trip(n_factors: int, correlated: bool, tmp_path: Path) -> None:
    df, factors = _mirt_data(n_factors, seed=40 + n_factors)
    spec = Model.uniform(TwoPL(), items=df.columns).with_factors(factors, correlated=correlated)
    spec = spec.fix(item="q00", a1=1.0)
    res = fit(spec, df, method=BAEM(tol=1e-4), se="oakes")
    path = tmp_path / "mirt.json"
    res.to_json(path)
    parsed = _assert_strict_json(path.read_text(encoding="utf-8"))
    assert parsed["model"]["factors"] == [
        {"name": name, "items": list(items)} for name, items in spec.factors or ()
    ]
    assert parsed["model"]["correlated_factors"] is correlated
    loaded = FitResult.from_json(path)
    assert loaded._model == res._model
    pd.testing.assert_frame_equal(loaded.parameters, res.parameters, check_exact=True)
    pd.testing.assert_frame_equal(loaded.items, res.items, check_exact=True)
    pd.testing.assert_frame_equal(loaded.latent, res.latent, check_exact=True)
    assert res.vcov is not None and loaded.vcov is not None
    np.testing.assert_array_equal(loaded.vcov, res.vcov)
    assert loaded.summary() == res.summary()
    assert loaded._repr_html_() == res._repr_html_()
    assert (loaded.aic, loaded.bic, loaded.n_params) == (res.aic, res.bic, res.n_params)
    pd.testing.assert_frame_equal(loaded.score(df), res.score(df), check_exact=True)


def test_schema_version_1_payload_still_loads(sample_payload: dict[str, Any]) -> None:
    """Files written before factors were stored (schema 1) load as unidimensional models."""
    old = _clone(sample_payload)
    old["schema_version"] = 1
    del old["model"]["factors"]
    del old["model"]["correlated_factors"]
    del old["model"]["equal_items"]
    loaded = FitResult.from_json(json.dumps(old))
    assert not loaded._model.is_multidimensional
    expected = FitResult.from_json(json.dumps(sample_payload))
    pd.testing.assert_frame_equal(loaded.items, expected.items, check_exact=True)
    # schema 1 must not carry the new keys, and schema 2 must have them
    with_keys = _clone(sample_payload)
    with_keys["schema_version"] = 1
    del with_keys["model"]["equal_items"]
    with pytest.raises(SpecificationError, match="factors"):
        FitResult.from_json(json.dumps(with_keys))
    missing = _clone(sample_payload)
    del missing["model"]["factors"]
    with pytest.raises(SpecificationError, match="factors"):
        FitResult.from_json(json.dumps(missing))


@pytest.fixture(scope="module")
def mirt_payload() -> dict[str, Any]:
    df, factors = _mirt_data(2, seed=7)
    spec = Model.uniform(TwoPL(), items=df.columns).with_factors(factors)
    return _assert_strict_json(fit(spec, df, method=BAEM(tol=1e-3), se=None).to_json())


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda m: m["factors"][0].update(items=["nope"]), "unknown items"),
        (lambda m: m["factors"][1].update(name=m["factors"][0]["name"]), "Duplicated factor"),
        (lambda m: m["factors"].pop(), "2 to 3 factors"),
        (lambda m: m["factors"][1]["items"].pop(), "not assigned to any factor"),
        (lambda m: m.update(factors=m["factors"] * 2), "Duplicated factor"),
        (lambda m: m.update(correlated_factors="yes"), "correlated_factors"),
        (
            lambda m: m.update(
                latent={
                    "type": "Normal",
                    "mean": 0.0,
                    "var": 1.0,
                    "free_mean": False,
                    "free_var": False,
                }
            ),
            "factors",
        ),
        (
            lambda m: m["fixed"].append({"item": "q00", "param": "a", "value": 1.0}),
            "unknown parameter",
        ),
        (lambda m: m["factors"][0].update(extra=1), "Unexpected"),
    ],
)
def test_malformed_factor_structure(
    mirt_payload: dict[str, Any], mutate: Callable[[dict[str, Any]], Any], match: str
) -> None:
    bad = _clone(mirt_payload)
    mutate(bad["model"])
    with pytest.raises(SpecificationError, match=match):
        FitResult.from_json(json.dumps(bad))
