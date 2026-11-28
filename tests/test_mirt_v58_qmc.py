"""Agreement with R mirt for a 4-factor confirmatory 2PL fitted with QMC (VALIDATION.md V58, #164).

Reference ``tests/reference/v58_qmc.json`` comes from ``validation/r/v58_qmc_reference.R``:
mirt's ordinary EM on a 13^4 = 28,561-node product grid on [-4.8, 4.8], ``TOL = 1e-6``,
``SE.type = "Oakes"``. flexpyirt fits the same data with ``Quadrature.qmc(10_000)`` (DESIGN.md
§28), so both sides carry quadrature error; the tolerances come from that, not from the 1e-4 of
the product-grid cases. mirt's own ``method = "QMCEM"`` is not the reference: on this case it
drifts to near-1 correlations (stored under ``mirt_qmcem`` and checked below as a record).
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from flexpyirt import BAEM, FitResult, Model, Quadrature, TwoPL, fit
from tests._reference import check_reference_meta

# mirt reference comparison: nightly and release-PR CI only (CONTRIBUTING.md, "Tests").
pytestmark = pytest.mark.slow

REFERENCE_FILE = Path(__file__).parent / "reference" / "v58_qmc.json"
N_POINTS = 10_000
TOL_PARAM = 1.5e-2  # measured 5.3e-3 (quadrature error of both fits)
TOL_COR = 5e-3  # measured 2.0e-3
TOL_LOGLIK = 1.0  # measured 0.48
TOL_SE_REL = 2e-2  # measured 4.6e-3 (items), 5.5e-3 (correlations)


@cache
def _reference() -> dict[str, Any]:
    raw: dict[str, Any] = json.loads(REFERENCE_FILE.read_text())
    check_reference_meta(raw["meta"], require_grid_check=True)
    case: dict[str, Any] = raw["cases"]["qmc4d_2pl"]
    check_reference_meta(case["meta"], require_grid_check=True)
    return case


@cache
def _fit() -> FitResult:
    case = _reference()
    data = pd.DataFrame(np.asarray(case["patterns"], dtype=float), columns=case["items"])
    data = data.loc[data.index.repeat(np.asarray(case["counts"], dtype=int))]
    spec = Model.uniform(TwoPL(), items=case["items"]).with_factors(case["factors"])
    return fit(spec, data, method=BAEM(tol=1e-6), quadrature=Quadrature.qmc(N_POINTS))


def _free_item_params(case: dict[str, Any]) -> list[tuple[str, str]]:
    ref = case["estimates"]["params"]
    return [
        (item, name)
        for item in case["items"]
        for name in ref[item]
        if not name.endswith("_se") and ref[item].get(f"{name}_se") is not None
    ]


def test_items_correlations_and_loglik_match_mirt_em() -> None:
    case = _reference()
    res = _fit()
    assert res.converged
    params = res.parameters.set_index(["item", "param"])
    ref = case["estimates"]["params"]
    for item, name in _free_item_params(case):
        got = float(params.at[(item, name), "estimate"])  # type: ignore[arg-type]
        assert got == pytest.approx(ref[item][name], abs=TOL_PARAM), (item, name)
    for name, value in case["estimates"]["latent"].items():
        if name.endswith("_se"):
            continue
        assert float(res.latent[name].iloc[0]) == pytest.approx(value, abs=TOL_COR), name
    assert res.loglik == pytest.approx(case["loglik"], abs=TOL_LOGLIK)


def test_oakes_standard_errors_match_mirt_em() -> None:
    case = _reference()
    res = _fit()
    params = res.parameters.set_index(["item", "param"])
    ref = case["estimates"]["params"]
    for item, name in _free_item_params(case):
        got = float(params.at[(item, name), "se"])  # type: ignore[arg-type]
        assert got == pytest.approx(ref[item][f"{name}_se"], rel=TOL_SE_REL), (item, name)
    for name, value in case["estimates"]["latent"].items():
        if name.endswith("_se"):
            got = float(res.latent[name].iloc[0])
            assert got == pytest.approx(value, rel=TOL_SE_REL), name


def test_mirt_qmcem_record_is_the_documented_failure() -> None:
    # A record, not a requirement on flexpyirt: mirt's QMCEM drifted on this case
    # (VALIDATION.md V58). If a new mirt fixes it, regenerate and revisit V58.
    qmcem = _reference()["mirt_qmcem"]
    assert min(qmcem["latent"].values()) > 0.85
