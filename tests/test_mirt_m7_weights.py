"""Numerical agreement with R mirt for sampling weights (VALIDATION.md V34).

Reference files come from validation/r/m7_weights_reference.R (mirt's ``survey.weights``, Oakes
SEs): V34a a 2PL with non-integer weights that do not sum to N and five zero weights, V34b the
Science graded model with weights normalized to N, V34c two groups with anchors, a free focal
latent distribution and 5% missing responses, V34d FIPC with a free latent distribution. Each
file stores the raw rows with one weight per row.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from flexpyirt import BAEM, GRM, FitResult, Model, Normal, Quadrature, TwoPL, fit
from flexpyirt.errors import FlexpyirtWarning
from tests._reference import check_reference_meta

# mirt reference comparison: nightly and release-PR CI only (CONTRIBUTING.md, "Tests").
pytestmark = pytest.mark.slow

REFERENCE_DIR = Path(__file__).parent / "reference"
SQUAREM = (False, True)
TOL_LOGLIK_REL = 1e-6
TOL_PARAM = 1e-4
TOL_SE_REL = 1e-3
TOL_IC = 1e-4
CASES = ["v34_weights_2pl", "v34_weights_grm", "v34_weights_twogroup", "v34_weights_fipc"]
WIDE = Quadrature.equispaced(81, bounds=(-8, 8))


@dataclass(frozen=True)
class Case:
    name: str
    raw: dict[str, Any]
    spec: Model
    quadrature: Quadrature

    @property
    def groups(self) -> list[str]:
        return list(self.raw["groups"])

    @property
    def data(self) -> pd.DataFrame:
        rows = np.asarray(self.raw["data"], dtype=float)  # null (missing) becomes NaN
        return pd.DataFrame(rows, columns=self.raw["items"])

    @property
    def group(self) -> np.ndarray | None:
        return np.asarray(self.raw["group"]) if self.spec.is_multigroup else None

    @property
    def weights(self) -> np.ndarray:
        return np.asarray(self.raw["weights"], dtype=float)


@cache
def load(name: str) -> Case:
    raw = dict(json.loads((REFERENCE_DIR / f"{name}.json").read_text()))
    check_reference_meta(raw["meta"], require_grid_check=True)
    items = list(raw["items"])
    if name == "v34_weights_grm":
        return Case(
            name, raw, Model.uniform(GRM(n_categories=4), items=items), Quadrature.equispaced()
        )
    if name == "v34_weights_twogroup":
        spec = (
            Model.uniform(TwoPL(), items=items)
            .with_groups(ref=Normal.standard(), focal=Normal.free())
            .equal_across_groups(items=raw["anchors"])
        )
        return Case(name, raw, spec, WIDE)
    if name == "v34_weights_fipc":
        spec = (
            Model.uniform(TwoPL(), items=items)
            .fix_from(pd.DataFrame(raw["bank"]))
            .with_latent(Normal.free())
        )
        return Case(name, raw, spec, WIDE)
    return Case(name, raw, Model.uniform(TwoPL(), items=items), Quadrature.equispaced())


@cache
def _fit(name: str, se: str | None, squarem: bool = False) -> FitResult:
    case = load(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FlexpyirtWarning)  # Oakes / Hessian SEs under weights
        return fit(
            case.spec,
            case.data,
            group=case.group,
            weights=case.weights,
            method=BAEM(tol=1e-9, max_iter=50000, squarem=squarem),
            quadrature=case.quadrature,
            se=se,  # type: ignore[arg-type]
        )


def _pairs(case: Case, res: FitResult, column: str) -> list[tuple[str, float, float]]:
    """``(label, flexpyirt value, mirt value)`` for every free parameter."""
    suffix = "" if column == "estimate" else "_se"
    out = []
    params = res.parameters[res.parameters["free"]]
    for g, item, param, value in zip(
        params["group"], params["item"], params["param"], params[column], strict=True
    ):
        ref = case.raw["estimates"][g]
        block = ref["latent"] if item == "__latent__" else ref["params"][item]
        out.append((f"{g}.{item}.{param}", float(value), float(block[param + suffix])))
    return out


def test_reference_files_hold_weighted_fits() -> None:
    for name in CASES:
        case = load(name)
        weights = case.weights
        assert len(weights) == len(case.data) == case.raw["n_rows"]
        assert np.all(weights >= 0) and not np.allclose(weights, np.round(weights))
        np.testing.assert_allclose(weights.sum(), case.raw["sum_weights"], rtol=1e-12)
        # The weights change the fit (mirt did not ignore them).
        assert abs(case.raw["loglik"] - case.raw["loglik_unweighted"]) > 1e-3
    assert int((load("v34_weights_2pl").weights == 0).sum()) == 5
    assert abs(load("v34_weights_2pl").raw["sum_weights"] - 1000) > 100
    assert load("v34_weights_twogroup").data.isna().to_numpy().any()


@pytest.mark.parametrize("squarem", SQUAREM)
@pytest.mark.parametrize("name", CASES)
def test_estimates_and_loglik(name: str, squarem: bool) -> None:
    case, res = load(name), _fit(name, "oakes", squarem)
    assert res.converged
    assert res.n_params == case.raw["meta"]["n_free"]
    assert abs(res.loglik - case.raw["loglik"]) <= TOL_LOGLIK_REL * abs(case.raw["loglik"])
    for label, got, want in _pairs(case, res, "estimate"):
        assert abs(got - want) <= TOL_PARAM, label


@pytest.mark.parametrize("name", CASES)
def test_information_criteria_use_sum_of_weights(name: str) -> None:
    case, res = load(name), _fit(name, "oakes")
    assert res.n_obs == int((case.weights > 0).sum())
    assert res._sum_weights == pytest.approx(case.raw["sum_weights"], rel=1e-12)
    assert abs(res.aic - case.raw["aic"]) <= TOL_IC
    assert abs(res.bic - case.raw["bic"]) <= TOL_IC


@pytest.mark.parametrize("se", ["oakes", "hessian"])
@pytest.mark.parametrize("name", CASES)
def test_standard_errors_match_mirt_oakes(name: str, se: str) -> None:
    case, res = load(name), _fit(name, se)
    for label, got, want in _pairs(case, res, "se"):
        assert abs(got - want) <= TOL_SE_REL * want, label


@pytest.mark.parametrize("name", CASES)
def test_weighted_differs_from_unweighted(name: str) -> None:
    """The reference estimates are not reproduced without the weights."""
    case = load(name)
    plain = fit(
        case.spec,
        case.data,
        group=case.group,
        method=BAEM(tol=1e-9, max_iter=50000),
        quadrature=case.quadrature,
        se=None,
    )
    assert abs(plain.loglik - case.raw["loglik_unweighted"]) <= TOL_LOGLIK_REL * abs(plain.loglik)
    worst = max(abs(got - want) for _, got, want in _pairs(case, plain, "estimate"))
    assert worst > 10 * TOL_PARAM


def test_measured_differences(capsys: pytest.CaptureFixture[str]) -> None:
    """Print the measured differences recorded in VALIDATION.md (run with ``-s``)."""
    for name in CASES:
        case = load(name)
        for squarem in SQUAREM:
            res = _fit(name, "oakes", squarem)
            est = max(abs(g - w) for _, g, w in _pairs(case, res, "estimate"))
            se = max(abs(g - w) / w for _, g, w in _pairs(case, res, "se"))
            hess = max(abs(g - w) / w for _, g, w in _pairs(case, _fit(name, "hessian"), "se"))
            loglik = abs(res.loglik - case.raw["loglik"]) / abs(case.raw["loglik"])
            aic, bic = abs(res.aic - case.raw["aic"]), abs(res.bic - case.raw["bic"])
            with capsys.disabled():
                print(
                    f"\n{name} squarem={squarem}: loglik rel {loglik:.1e}, "
                    f"params {est:.1e}, SE oakes rel {se:.1e}, SE hessian rel {hess:.1e}, "
                    f"AIC {aic:.1e}, BIC {bic:.1e}"
                )
