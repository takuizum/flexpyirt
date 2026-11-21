"""Numerical agreement of IRT scale linking with base-R references on mirt fits (VALIDATION V20).

Reference values come from ``validation/r/m5_linking_reference.R`` ->
``tests/reference/v20_linking.json``. Because R ``mirt`` does not implement scale linking, each
form is calibrated with mirt (``optimizer="NR"``, ``TOL=1e-10``, 61 nodes on [-6, 6], N(0, 1),
Oakes SEs), and the Haebara (1980) and Stocking–Lord (1983) criteria (non-symmetric and
Kim & Lee, 2006 symmetric variants) plus Ogasawara (2001) delta-method SEs are computed in base R
directly from the formulas using ``mirt::probtrace()`` and ``mirt::expected.item()``.

Tests separate linking accuracy (linking directly on mirt's item estimates) from end-to-end
calibration + linking accuracy (fitting both forms in flexpyirt and linking).
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from flexpyirt import BAEM, GPCM, GRM, FitResult, Model, Quadrature, ThreePL, TwoPL, fit, link
from flexpyirt.items import ItemModel
from tests._reference import check_reference_meta

REFERENCE_DIR = Path(__file__).parent / "reference"
REFERENCE_FILE = "v20_linking.json"

CASES = ["v20a_2pl", "v20b_grm", "v20c_gpcm", "v20d_3pl"]
VARIANTS = ["stocking_lord", "stocking_lord_sym", "haebara", "haebara_sym"]

TOL_AB_DIRECT = 1e-7
TOL_CRIT_REL_DIRECT = 1e-8
TOL_AB_FULL = 1e-6
TOL_CRIT_REL_FULL = 1e-5
TOL_SE_REL = 1e-3


@cache
def _bundle() -> dict[str, Any]:
    raw = dict(json.loads((REFERENCE_DIR / REFERENCE_FILE).read_text()))
    check_reference_meta(raw["meta"], require_grid_check=True)
    return raw


@cache
def _case(name: str) -> dict[str, Any]:
    raw = dict(_bundle()["cases"][name])
    check_reference_meta(raw["meta"], require_grid_check=True)
    return raw


def _item_model(itemtype: str, n_cat: int) -> ItemModel:
    if itemtype == "2PL":
        return TwoPL()
    if itemtype == "3PL":
        return ThreePL()
    if itemtype == "graded":
        return GRM(n_cat=n_cat)
    if itemtype == "gpcm":
        return GPCM(n_cat=n_cat)
    raise ValueError(f"Unsupported reference itemtype {itemtype!r}")


def _form_data(form: dict[str, Any]) -> pd.DataFrame:
    pats = pd.DataFrame(np.asarray(form["patterns"], dtype=int), columns=form["items"])
    counts = np.asarray(form["counts"], dtype=int)
    return pats.loc[pats.index.repeat(counts)].reset_index(drop=True)


@cache
def _fixed_at_mirt(name: str, side: str) -> FitResult:
    """Create a FitResult with all parameters fixed at mirt's estimates (zero calibration diff)."""
    case = _case(name)
    form = case[side]
    meta = case["meta"]
    spec = Model.uniform(_item_model(meta["itemtype"], int(meta["n_cat"])), items=form["items"])
    data = _form_data(form)
    quad = Quadrature.equispaced(int(meta["quadpts"]), lim=tuple(meta["theta_lim"]))
    table = spec.parameter_table(data)
    vals = []
    for item, param in zip(table["item"], table["param"], strict=True):
        if item == "__latent__":
            vals.append(0.0 if param == "mean" else 1.0)
        else:
            vals.append(float(form["params"][item][param]))
    table["value"] = np.asarray(vals, dtype=float)
    table["free"] = False
    return fit(spec, data, table=table, quadrature=quad, se=None)


@cache
def _fit_form(name: str, side: str) -> FitResult:
    """Calibrate one form from scratch in flexpyirt with tight tolerance and Oakes SEs."""
    case = _case(name)
    form = case[side]
    meta = case["meta"]
    spec = Model.uniform(_item_model(meta["itemtype"], int(meta["n_cat"])), items=form["items"])
    data = _form_data(form)
    quad = Quadrature.equispaced(int(meta["quadpts"]), lim=tuple(meta["theta_lim"]))
    return fit(
        spec,
        data,
        method=BAEM(tol=1e-9, max_iter=20000, squarem=True),
        quadrature=quad,
        se="oakes",
    )


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("side", ("target", "source"))
def test_form_calibration_matches_mirt(case_name: str, side: str) -> None:
    case = _case(case_name)
    form = case[side]
    res = _fit_form(case_name, side)
    assert res.converged
    ref_ll = float(form["loglik"])
    assert res.loglik == pytest.approx(ref_ll, rel=1e-6)
    items_df = res.items.set_index("item")
    for item, pvals in form["params"].items():
        for col, ref_val in pvals.items():
            if col.endswith("_se") or ref_val is None:
                continue
            assert items_df.loc[item, col] == pytest.approx(float(ref_val), abs=1e-4)


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("variant", VARIANTS)
def test_linking_on_mirt_estimates_matches_r_reference(case_name: str, variant: str) -> None:
    """Given identical mirt item estimates, A, B, and criterion match the R reference tightly."""
    case = _case(case_name)
    ref = case["linking"][variant]
    src = _fixed_at_mirt(case_name, "source")
    tgt = _fixed_at_mirt(case_name, "target")
    linked = link(
        src,
        tgt,
        method=ref["method"],
        symmetric=bool(ref["symmetric"]),
    )
    assert linked.converged
    assert list(linked.common_items) == list(case["common_items"])
    assert pytest.approx(float(ref["A"]), abs=TOL_AB_DIRECT) == linked.A
    assert pytest.approx(float(ref["B"]), abs=TOL_AB_DIRECT) == linked.B
    assert linked.criterion == pytest.approx(
        float(ref["criterion"]), rel=TOL_CRIT_REL_DIRECT, abs=1e-12
    )


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("variant", VARIANTS)
def test_end_to_end_fit_and_link_matches_r_reference(case_name: str, variant: str) -> None:
    """Fit both forms in flexpyirt, link, and compare A, B, criterion, and Ogasawara SEs."""
    case = _case(case_name)
    ref = case["linking"][variant]
    src = _fit_form(case_name, "source")
    tgt = _fit_form(case_name, "target")
    linked = link(
        src,
        tgt,
        method=ref["method"],
        symmetric=bool(ref["symmetric"]),
    )
    assert linked.converged
    assert pytest.approx(float(ref["A"]), abs=TOL_AB_FULL) == linked.A
    assert pytest.approx(float(ref["B"]), abs=TOL_AB_FULL) == linked.B
    assert linked.criterion == pytest.approx(
        float(ref["criterion"]), rel=TOL_CRIT_REL_FULL, abs=1e-10
    )
    assert linked.vcov is not None
    assert linked.vcov.shape == (2, 2)
    np.testing.assert_allclose(linked.vcov, linked.vcov.T, atol=1e-15)
    assert np.all(np.linalg.eigvalsh(linked.vcov) > 0.0)
    assert linked.A_se == pytest.approx(float(ref["A_se"]), rel=TOL_SE_REL)
    assert linked.B_se == pytest.approx(float(ref["B_se"]), rel=TOL_SE_REL)
    np.testing.assert_allclose(linked.vcov, np.asarray(ref["vcov"], dtype=float), rtol=TOL_SE_REL)
