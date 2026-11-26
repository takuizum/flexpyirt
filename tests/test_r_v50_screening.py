"""Common-item screening against a base-R recomputation on mirt calibrations (VALIDATION V50).

Reference values come from ``validation/r/v50_screening_reference.R`` ->
``tests/reference/v50_screening.json``. No R package implements these statistics with
characteristic-curve linking, so the R script recomputes the rules of DESIGN.md §22 in base R:
each form is calibrated with mirt (``optimizer="NR"``, ``TOL=1e-10``, 61 nodes on [-6, 6], latent
N(0, 1)), the linking is the V20 base-R Stocking–Lord / Haebara, and the locations, robust z
(``median``, ``quantile(type = 7)``), displacement, expected-score RMSD (``expected.item()``) and
the purification loop follow the formulas. ``equateIRT::dif.test`` (Lord's Wald test after
Stocking–Lord linking, with purification) is recorded as an independent check of which items
drifted.

The comparison runs :func:`flexpyirt.screen_common_items` on fits whose parameters are all fixed
at mirt's estimates, so only the screening and linking are compared.
"""

from __future__ import annotations

import json
import warnings
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from flexpyirt import GPCM, GRM, FitResult, Model, Quadrature, TwoPL, fit, screen_common_items
from flexpyirt.errors import ScreeningWarning
from flexpyirt.items import ItemModel
from tests._reference import check_reference_meta

# mirt / R reference comparison: nightly and release-PR CI only (CONTRIBUTING.md, "Tests").
pytestmark = pytest.mark.slow

REFERENCE_FILE = Path(__file__).parent / "reference" / "v50_screening.json"

# The R linking is polished to max|grad| ~1e-12 and flexpyirt's Newton stops at 1e-10; observed
# max |dA|, |dB| = 2.3e-11 over all steps of all variants.
TOL_AB = 1e-9
TOL_CRIT_REL = 1e-7
# Per-item quantities inherit the (A, B) difference (observed max 5.0e-12); the robust z divides by
# 0.74 IQR (observed max 3.4e-10).
TOL_ITEM = 1e-9
TOL_Z = 1e-8

VARIANTS = {
    "v50a_2pl": [
        "robust_z",
        "robust_z_all",
        "displacement",
        "rmsd",
        "displacement_haebara_sym",
        "robust_z_max_iter_1",
        "rmsd_min_items_11",
    ],
    "v50b_mixed": ["robust_z", "displacement", "rmsd", "rmsd_haebara_all"],
}
CASE_VARIANTS = [(case, v) for case, vs in VARIANTS.items() for v in vs]
ITEM_COLUMNS = [
    "a_source",
    "b_source",
    "a_target",
    "b_target",
    "log_ratio_a",
    "difference_b",
    "displacement",
    "rmsd",
]


@cache
def _bundle() -> dict[str, Any]:
    raw = dict(json.loads(REFERENCE_FILE.read_text()))
    check_reference_meta(raw["meta"], require_grid_check=True)
    return raw


@cache
def _case(name: str) -> dict[str, Any]:
    raw = dict(_bundle()["cases"][name])
    check_reference_meta(raw["meta"], require_grid_check=True)
    return raw


def _as_list(value: Any) -> list[str]:
    """R's ``auto_unbox`` writes a length-1 character vector as a scalar."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _item_model(itemtype: str, n_categories: int) -> ItemModel:
    if itemtype == "2PL":
        return TwoPL()
    if itemtype == "graded":
        return GRM(n_categories=n_categories)
    if itemtype == "gpcm":
        return GPCM(n_categories=n_categories)
    raise ValueError(f"Unsupported reference itemtype {itemtype!r}")


@cache
def _fixed_at_mirt(name: str, side: str) -> FitResult:
    """A FitResult with every parameter fixed at mirt's estimates."""
    case = _case(name)
    form = case[side]
    spec = Model.from_items(
        {
            item: _item_model(str(form["itemtypes"][item]), int(form["n_cats"][item]))
            for item in form["items"]
        }
    )
    pats = pd.DataFrame(np.asarray(form["patterns"], dtype=int), columns=form["items"])
    data = pats.loc[pats.index.repeat(np.asarray(form["counts"], dtype=int))].reset_index(drop=True)
    quad = Quadrature.equispaced(
        int(case["meta"]["quadpts"]), bounds=tuple(case["meta"]["theta_lim"])
    )
    table = spec.parameter_table(data)
    table["value"] = [
        float(form["latent"][str(param)])
        if item == "__latent__"
        else float(form["params"][item][param])
        for item, param in zip(table["item"], table["param"], strict=True)
    ]
    table["free"] = False
    return fit(spec, data, table=table, quadrature=quad, se=None)


@cache
def _screen(case_name: str, variant: str) -> tuple[Any, list[warnings.WarningMessage]]:
    ref = _case(case_name)["screening"][variant]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = screen_common_items(
            _fixed_at_mirt(case_name, "source"),
            _fixed_at_mirt(case_name, "target"),
            items=_case(case_name)["common_items"],
            method=ref["method"],
            symmetric=bool(ref["symmetric"]),
            statistic=ref["statistic"],
            threshold=float(ref["threshold"]),
            removal=ref["removal"],
            min_items=int(ref["min_items"]),
            max_iter=int(ref["max_iter"]),
        )
    return res, list(caught)


@pytest.mark.parametrize(("case_name", "variant"), CASE_VARIANTS)
def test_removal_order_and_constants_per_step_match_r(case_name: str, variant: str) -> None:
    ref = _case(case_name)["screening"][variant]
    res, _ = _screen(case_name, variant)
    assert list(res.removed_items) == _as_list(ref["removed_items"])
    assert len(res.history) == len(ref["history"])
    for row, ref_step in zip(res.history.itertuples(), ref["history"], strict=True):
        assert row.iteration == ref_step["iteration"]
        assert row.n_items == ref_step["n_items"]
        assert list(row.removed) == _as_list(ref_step["removed"])
        assert row.converged
        np.testing.assert_allclose([row.A, row.B], [ref_step["A"], ref_step["B"]], atol=TOL_AB)
        np.testing.assert_allclose(row.criterion, ref_step["criterion"], rtol=TOL_CRIT_REL)
    assert pytest.approx(ref["history"][-1]["A"], abs=TOL_AB) == res.link.A
    assert pytest.approx(ref["history"][-1]["B"], abs=TOL_AB) == res.link.B


@pytest.mark.parametrize(("case_name", "variant"), CASE_VARIANTS)
def test_final_table_matches_r(case_name: str, variant: str) -> None:
    ref_table = pd.DataFrame(_case(case_name)["screening"][variant]["table"])
    res, _ = _screen(case_name, variant)
    table = res.table
    assert list(table["item"]) == list(ref_table["item"])
    for col in ITEM_COLUMNS:
        np.testing.assert_allclose(
            table[col].to_numpy(dtype=float), ref_table[col].to_numpy(dtype=float), atol=TOL_ITEM
        )
    for col in ["robust_z_a", "robust_z_b", "statistic"]:
        np.testing.assert_allclose(
            table[col].to_numpy(dtype=float), ref_table[col].to_numpy(dtype=float), atol=TOL_Z
        )
    assert list(table["flagged"]) == list(ref_table["flagged"].astype(bool))
    assert list(table["kept"]) == list(ref_table["kept"].astype(bool))
    ref_removed = [None if pd.isna(v) else int(v) for v in ref_table["iteration_removed"]]
    got_removed = [None if pd.isna(v) else int(v) for v in table["iteration_removed"]]
    assert got_removed == ref_removed


@pytest.mark.parametrize(("case_name", "variant"), CASE_VARIANTS)
def test_screening_warning_iff_kept_items_remain_flagged(case_name: str, variant: str) -> None:
    res, caught = _screen(case_name, variant)
    expected = bool((res.table["flagged"] & res.table["kept"]).any())
    got = any(issubclass(w.category, ScreeningWarning) for w in caught)
    assert got == expected
    assert expected == variant.endswith(("max_iter_1", "min_items_11"))


@pytest.mark.parametrize("case_name", list(VARIANTS))
def test_planted_drift_is_removed_by_robust_z(case_name: str) -> None:
    """Sanity check against the simulation: every planted drift is removed."""
    drifted = set(_as_list(_case(case_name)["drifted_items"]))
    res, _ = _screen(case_name, "robust_z")
    assert drifted <= set(res.removed_items)


def test_equateirt_wald_test_flags_the_planted_drift() -> None:
    """Independent cross-check: equateIRT::dif.test flags the items robust z removes first."""
    case = _case("v50a_2pl")
    flagged = set(_as_list(case["cross_check"]["flagged"]))
    drifted = set(_as_list(case["drifted_items"]))
    assert drifted <= flagged
    res, _ = _screen("v50a_2pl", "robust_z")
    assert set(res.removed_items[: len(drifted)]) == drifted
