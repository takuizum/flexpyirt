"""V21: multiple-group confirmatory 2D MIRT vs R mirt ``multipleGroup()`` (DESIGN.md §15.43).

Reference values come from ``validation/r/m4_mirt_multigroup_reference.R``: two groups, anchors on
both factors, focal means and variances free (``invariance = c(anchors, "free_means",
"free_var")``), correlations free in both groups, 81^2 nodes on [-6, 6].

mirt 1.48 stops short of the maximum along one factor's ridge (that factor's focal mean/variance
traded against its anchor slopes): at mirt's estimates the score of the shared log-likelihood is
~0.03 for that factor's focal variance and mean, while flexpyirt's solution is stationary and has
a higher log-likelihood (VALIDATION.md V21). Those ridge parameters are therefore checked by
likelihood and stationarity; every other parameter agrees with mirt to 1e-4.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from flexpyirt import BAEM, FitResult, Model, Normal, Quadrature, TwoPL, _core, fit
from flexpyirt._compile import compile_problem, prepare
from flexpyirt.errors import IdentificationError
from tests._reference import check_reference_meta

REFERENCE = Path(__file__).parent / "reference" / "v21_mirt_multigroup.json"
CASES = ["mg2d_simple", "mg2d_cross"]
RIDGE_FACTOR = {"mg2d_simple": 1, "mg2d_cross": 2}
"""Factor whose focal mean/variance mirt does not fully optimize (see module docstring)."""
TOL_LOGLIK_REL = 1e-6
TOL_PARAM = 1e-4
TOL_SE_REL = 1e-3
QUAD = Quadrature.equispaced(81, lim=(-6, 6))


@cache
def _case(name: str) -> dict[str, Any]:
    raw = json.loads(REFERENCE.read_text())["cases"][name]
    check_reference_meta(raw["meta"], require_grid_check=True)
    return dict(raw)


def _spec(raw: dict[str, Any]) -> Model:
    return (
        Model.uniform(TwoPL(), items=raw["items"])
        .with_factors(raw["factors"])
        .with_groups(ref=Normal.standard(), focal=Normal.free())
        .equal_across_groups(items=raw["anchors"])
    )


def _data(raw: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray]:
    pats = pd.DataFrame(np.asarray(raw["patterns"], dtype=float), columns=raw["items"])
    reps = np.asarray(raw["counts"], dtype=int)
    df = pats.loc[pats.index.repeat(reps)].reset_index(drop=True)
    return df, np.repeat(np.asarray(raw["pattern_groups"]), reps)


@cache
def _fit(name: str, squarem: bool = False) -> FitResult:
    raw = _case(name)
    df, grp = _data(raw)
    method = BAEM(tol=1e-9, max_iter=20000, squarem=squarem)
    return fit(_spec(raw), df, group=grp, method=method, quadrature=QUAD, se="oakes")


@cache
def _loglik_at_mirt(name: str) -> float:
    raw = _case(name)
    df, grp = _data(raw)
    spec = _spec(raw)
    table = spec.parameter_table(df, group=grp)
    for i, row in table.iterrows():
        est = raw["estimates"][row["group"]]
        src = est["latent"] if row["item"] == "__latent__" else est["params"][row["item"]]
        table.at[i, "value"] = src[row["param"]]
    table["free"] = False
    return fit(spec, df, group=grp, table=table, quadrature=QUAD, se=None).loglik


def _ridge(name: str) -> tuple[set[str], set[str], int]:
    """mirt's unconverged block: the ridge factor's focal mean/variance and its items."""
    raw, m = _case(name), RIDGE_FACTOR[name]
    factor = list(raw["factors"])[m - 1]
    return {f"mean_{m}", f"var_{m}"}, set(raw["factors"][factor]), m


def _on_ridge(name: str, item: str, param: str) -> bool:
    _, ridge_items, m = _ridge(name)
    return item in ridge_items and param in (f"a{m}", "d")


@pytest.mark.parametrize("squarem", [False, True])
@pytest.mark.parametrize("name", CASES)
def test_loglik_and_estimates(name: str, squarem: bool) -> None:
    raw, res = _case(name), _fit(name, squarem)
    assert res.converged
    assert abs(res.loglik - raw["loglik"]) <= TOL_LOGLIK_REL * abs(raw["loglik"])
    assert res.loglik > _loglik_at_mirt(name)
    ridge_latent, _, _ = _ridge(name)
    items = res.items.set_index(["group", "item"])
    latent = res.latent.set_index("group")
    for g in raw["groups"]:
        ref = pd.DataFrame(raw["estimates"][g]["params"]).T.astype(float)
        for p in ("a1", "a2", "d"):
            keep = [i for i in ref.index if not _on_ridge(name, i, p)]
            np.testing.assert_allclose(
                items.loc[g].loc[keep, p], ref.loc[keep, p], rtol=0, atol=TOL_PARAM
            )
        for p in ("mean_1", "mean_2", "var_1", "var_2", "cor_12"):
            if g == "focal" and p in ridge_latent:
                continue
            assert latent.loc[g, p] == pytest.approx(
                raw["estimates"][g]["latent"][p], abs=TOL_PARAM
            )


@pytest.mark.parametrize("name", CASES)
def test_standard_errors(name: str) -> None:
    raw, res = _case(name), _fit(name)
    items = res.items.set_index(["group", "item"])
    latent = res.latent.set_index("group")
    ridge_latent, _, _ = _ridge(name)
    for g in raw["groups"]:
        ref = pd.DataFrame(raw["estimates"][g]["params"]).T
        for p in ("a1_se", "a2_se", "d_se"):
            keep = [i for i in ref.index if not _on_ridge(name, i, p.removesuffix("_se"))]
            expected = ref.loc[keep, p].astype(float).to_numpy()
            got = items.loc[g].loc[keep, p].to_numpy()
            free = np.isfinite(expected)
            np.testing.assert_allclose(got[free], expected[free], rtol=TOL_SE_REL)
            assert np.isnan(got[~free]).all()
        for p in ("mean_1", "mean_2", "var_1", "var_2", "cor_12"):
            if g == "focal" and (p in ridge_latent or p == "cor_12"):
                continue  # focal cor_12: compared on mirt's covariance scale below
            expected = raw["estimates"][g]["latent"][f"{p}_se"]
            if expected is None:
                assert np.isnan(latent.loc[g, f"{p}_se"])
            else:
                assert latent.loc[g, f"{p}_se"] == pytest.approx(expected, rel=TOL_SE_REL)


@pytest.mark.parametrize("name", CASES)
def test_focal_covariance_se(name: str) -> None:
    """mirt estimates the focal covariance COV_21; compare its SE with flexpyirt's delta-method SE.

    flexpyirt estimates (var_1, var_2, cor_12); cov_12 = cor_12 * sqrt(var_1 * var_2), and its SE
    follows from the Oakes covariance of those three parameters (the variances' own SEs agree
    with mirt directly).
    """
    raw, res = _case(name), _fit(name)
    assert res.vcov is not None
    t = res.parameters
    fid = t[t["free"]].drop_duplicates("free_id").set_index(["group", "item", "param"])["free_id"]
    ids = [int(fid[("focal", "__latent__", p)]) for p in ("var_1", "var_2", "cor_12")]
    v1, v2, c = (float(res._eta[i]) for i in ids)
    grad = np.array([0.5 * c * np.sqrt(v2 / v1), 0.5 * c * np.sqrt(v1 / v2), np.sqrt(v1 * v2)])
    se_cov = float(np.sqrt(grad @ res.vcov[np.ix_(ids, ids)] @ grad))
    assert se_cov == pytest.approx(raw["estimates"]["focal"]["latent"]["cov_12_se"], rel=TOL_SE_REL)


@pytest.mark.parametrize("name", CASES)
def test_mirt_stops_short_on_one_ridge(name: str) -> None:
    """Score of the shared log-likelihood: ~0 at flexpyirt's, nonzero only on mirt's ridge."""
    raw = _case(name)
    df, grp = _data(raw)
    spec, res = _spec(raw), _fit(name)
    prep = prepare(spec, df, group=grp, item_names=None, table=None)
    table = prep.table.copy()
    for i, row in table.iterrows():
        est = raw["estimates"][row["group"]]
        src = est["latent"] if row["item"] == "__latent__" else est["params"][row["item"]]
        table.at[i, "value"] = src[row["param"]]

    def score(tab: pd.DataFrame, eta: np.ndarray) -> np.ndarray:
        comp = compile_problem(spec, prep.y, prep.group, tab, QUAD)
        return np.asarray(
            jax.grad(lambda e: _core.marginal_loglik(comp.problem, e))(jnp.asarray(eta))
        )

    comp_m = compile_problem(spec, prep.y, prep.group, table, QUAD)
    g_ours = score(res.parameters, res._eta)
    g_mirt = score(table, comp_m.eta0)
    assert np.abs(g_ours).max() < 1e-5
    names = table[table["free"]].drop_duplicates("free_id").sort_values("free_id")
    ridge_latent, _, _ = _ridge(name)
    on_ridge = (
        (names["group"] == "focal")
        & (names["item"] == "__latent__")
        & names["param"].isin(ridge_latent)
    ).to_numpy()
    assert np.abs(g_mirt[on_ridge]).min() > 1e-2
    assert np.abs(g_mirt[~on_ridge]).max() < 1e-3


def test_monotone_and_oakes_equals_hessian() -> None:
    raw = _case("mg2d_simple")
    res = _fit("mg2d_simple")
    ll = res.history["loglik"].to_numpy()
    assert np.all(np.diff(ll) >= -1e-10 * np.abs(ll[1:]))
    df, grp = _data(raw)
    hess = fit(_spec(raw), df, group=grp, method=BAEM(tol=1e-9), quadrature=QUAD, se="hessian")
    assert res.vcov is not None and hess.vcov is not None
    np.testing.assert_allclose(res.vcov, hess.vcov, rtol=1e-8, atol=1e-12)


def test_unlinked_focal_factor_is_identification_error() -> None:
    raw = _case("mg2d_simple")
    df, grp = _data(raw)
    spec = (
        Model.uniform(TwoPL(), items=raw["items"])
        .with_factors(raw["factors"])
        .with_groups(ref=Normal.standard(), focal=Normal.free())
        .equal_across_groups(items=["q01", "q02", "q03"])  # no anchors on F2
    )
    with pytest.raises(IdentificationError, match="F2"):
        fit(spec, df, group=grp, quadrature=QUAD)


def test_scoring_and_json_round_trip() -> None:
    raw = _case("mg2d_cross")
    df, grp = _data(raw)
    res = _fit("mg2d_cross")
    sub, sub_grp = df.iloc[:40], grp[:40]
    for method in ("EAP", "MAP", "WLE"):
        scores = res.score(sub, method=method, group=sub_grp)
        assert np.isfinite(scores.to_numpy()).all()
    ml = res.score(sub, method="ML", group=sub_grp).to_numpy()
    assert not np.isnan(ml[:, :2]).any()
    pv = res.score(sub, method="PV", group=sub_grp, key=jax.random.key(0), n_draws=3)
    assert pv.shape == (40, 6) and np.isfinite(pv.to_numpy()).all()
    loaded = FitResult.from_json(res.to_json())
    pd.testing.assert_frame_equal(loaded.latent, res.latent, check_exact=True)
    pd.testing.assert_frame_equal(
        loaded.score(sub, group=sub_grp), res.score(sub, group=sub_grp), check_exact=True
    )
