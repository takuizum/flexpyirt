"""Numerical agreement with R mirt for M1 (VALIDATION.md V1, V2, V10).

Tolerances are those of VALIDATION.md. Rasch latent variances are compared with the note of
VALIDATION.md: mirt's closed-form latent update converges to a point with a slightly lower
discretized log-likelihood than the exact M-step used here (DESIGN.md §4.4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flexpyirt import BAEM, FitResult, Model, Rasch, TwoPL, fit
from tests._reference import Case, load

CASES = ["v1_lsat7_2pl", "v1_sim_2pl", "v2_lsat7_rasch", "v2_sim_rasch"]
METHOD = BAEM(tol=1e-9, max_iter=20000)

TOL_LOGLIK_REL = 1e-6
TOL_PARAM = 1e-4
TOL_SE_REL = 1e-3
TOL_EAP = 1e-5


def _spec(case: Case) -> Model:
    return Model.uniform(TwoPL() if case.itemtype == "2PL" else Rasch(), items=case.items)


_FITS: dict[tuple[str, str], FitResult] = {}


def _fit(name: str, se: str = "oakes") -> tuple[Case, FitResult]:
    case = load(name)
    if (name, se) not in _FITS:
        _FITS[(name, se)] = fit(_spec(case), case.data, method=METHOD, se=se)  # type: ignore[arg-type]
    return case, _FITS[(name, se)]


@pytest.mark.parametrize("name", CASES)
def test_loglik(name: str) -> None:
    case, res = _fit(name)
    assert res.converged
    ref = float(case.raw["loglik"])
    assert abs(res.loglik - ref) <= TOL_LOGLIK_REL * abs(ref)
    # Never worse than mirt's point on the same (discretized) likelihood.
    assert res.loglik >= ref - 1e-8


@pytest.mark.parametrize("name", CASES)
def test_item_parameters(name: str) -> None:
    case, res = _fit(name)
    items = res.items.set_index("item").loc[case.items]
    ref = case.params()
    np.testing.assert_allclose(items["a"], ref["a"], rtol=0, atol=TOL_PARAM)
    np.testing.assert_allclose(items["d"], ref["d"], rtol=0, atol=TOL_PARAM)


@pytest.mark.parametrize("name", CASES)
def test_latent_parameters(name: str) -> None:
    case, res = _fit(name)
    lat = res.latent.iloc[0]
    ref = case.latent()
    assert lat["mean"] == pytest.approx(ref["mean"], abs=TOL_PARAM)
    assert lat["var"] == pytest.approx(ref["var"], abs=TOL_PARAM)


@pytest.mark.parametrize("name", CASES)
def test_oakes_se_matches_mirt_oakes(name: str) -> None:
    """V10: Oakes SEs against mirt's SE.type = "Oakes"."""
    case, res = _fit(name, "oakes")
    _check_se(case, res, key="params")


@pytest.mark.parametrize("name", CASES)
def test_hessian_se_matches_mirt_richardson(name: str) -> None:
    """V10: direct-Hessian SEs against mirt's numerical Hessian (SE.type = "Richardson")."""
    case, res = _fit(name, "hessian")
    _check_se(case, res, key="se_richardson")


def _check_se(case: Case, res: FitResult, *, key: str) -> None:
    items = res.items.set_index("item").loc[case.items]
    ref = case.params(key)
    np.testing.assert_allclose(items["d_se"], ref["d_se"], rtol=TOL_SE_REL)
    if case.itemtype == "2PL":
        np.testing.assert_allclose(items["a_se"], ref["a_se"], rtol=TOL_SE_REL)
    else:
        assert items["a_se"].isna().all()
        var_se = case.latent("latent" if key == "params" else key)["var_se"]
        assert res.latent.iloc[0]["var_se"] == pytest.approx(var_se, rel=TOL_SE_REL)


@pytest.mark.parametrize("name", CASES)
def test_oakes_equals_hessian(name: str) -> None:
    """Oakes' identity is exact, so both information matrices agree to rounding."""
    _, oakes = _fit(name, "oakes")
    _, hess = _fit(name, "hessian")
    assert oakes.vcov is not None and hess.vcov is not None
    np.testing.assert_allclose(oakes.vcov, hess.vcov, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("name", CASES)
def test_eap_with_mirt_parameters(name: str) -> None:
    """EAP and posterior SD given mirt's estimates (all parameters fixed at mirt's values)."""
    case = load(name)
    spec = _spec(case)
    table = spec.parameter_table(case.data)
    ref = case.params()
    for param in ("a", "d"):
        values = ref[param].to_numpy(dtype=float)
        for item, value in zip(case.items, values, strict=True):
            row = (table["item"] == item) & (table["param"] == param)
            table.loc[row, "value"] = value
            table.loc[row, "free"] = False
    lat = case.latent()
    for param in ("mean", "var"):
        row = (table["item"] == "__latent__") & (table["param"] == param)
        table.loc[row, "value"] = lat[param]
        table.loc[row, "free"] = False
    res = fit(spec, case.data, table=table, se=None)
    assert res.n_params == 0
    scores = res.score(case.patterns)
    np.testing.assert_allclose(scores["theta"], case.raw["eap"]["theta"], rtol=0, atol=TOL_EAP)
    np.testing.assert_allclose(scores["se"], case.raw["eap"]["se"], rtol=0, atol=TOL_EAP)


@pytest.mark.parametrize("name", CASES)
def test_em_monotone(name: str) -> None:
    _, res = _fit(name)
    ll = res.history["loglik"].to_numpy()
    assert np.all(np.diff(ll) >= -1e-9 * np.abs(ll[1:]))


def test_rasch_defaults_free_variance() -> None:
    case, res = _fit("v2_lsat7_rasch")
    p = res.parameters.set_index(["item", "param"])
    assert not p.loc[("__latent__", "mean"), "free"]
    assert p.loc[("__latent__", "var"), "free"]
    assert (res.items["a"] == 1.0).all()
    assert res.n_params == len(case.items) + 1


def test_result_frames() -> None:
    case, res = _fit("v1_lsat7_2pl")
    assert list(res.items["item"]) == case.items
    assert np.allclose(res.items["b"], -res.items["d"] / res.items["a"])
    assert res.aic == pytest.approx(-2 * res.loglik + 2 * 10)
    assert res.bic == pytest.approx(-2 * res.loglik + 10 * np.log(1000))
    scores = res.score(case.data)
    assert isinstance(scores, pd.DataFrame) and len(scores) == len(case.data)
