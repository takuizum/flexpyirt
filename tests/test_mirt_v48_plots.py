"""V48: the numbers behind the plots vs R mirt and TAM.

Reference values come from ``validation/r/v48_plots_reference.R`` (DESIGN.md §20,
VALIDATION.md V48): mirt's ``itemfit(fit_stats = "X2", return.tables = TRUE)`` tables per
:math:`\\theta` bin (cell collapsing off), ``probtrace()``, Thurstonian thresholds found with
``uniroot()`` on ``probtrace()``, ``coef(IRTpars = TRUE)`` and TAM's ``tam.threshold()``. The
estimates of mirt (TAM) are put into a ``FitResult`` with every parameter fixed, so only the
derived numbers are compared.
"""

from __future__ import annotations

import json
import re
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest

from flexpyirt import (
    GPCM,
    GRM,
    FitResult,
    Model,
    Nominal,
    Normal,
    Quadrature,
    ThreePL,
    TwoPL,
    empirical_item_curve,
    fit,
    item_probabilities,
    wright_map_table,
)
from flexpyirt.items import ItemModel
from tests._reference import REFERENCE_DIR, check_reference_meta

pytestmark = pytest.mark.slow

SINGLE = ["twopl", "twopl_missing", "grm", "mixed"]
ALL = [*SINGLE, "twogroup"]

TOL_EAP = 1e-9
"""Our EAP scores vs mirt's fscores(method = "EAP") on the same 61-point grid."""
TOL_TABLE = 1e-9
"""Observed / expected proportions of the itemfit() tables."""
TOL_BIN_THETA = 5.000001e-5
"""Bin means vs mirt's labels, which are rounded to 4 decimals."""
TOL_PROBTRACE = 1e-12
TOL_THRESHOLD = 1e-9
"""Thurstonian thresholds vs uniroot(tol = 1e-13) and coef(IRTpars = TRUE)."""
TOL_TAM = 1e-4
"""tam.threshold() bisects until the interval is below 1e-4."""


@cache
def _bundle() -> dict[str, Any]:
    raw = json.loads((REFERENCE_DIR / "v48_plots.json").read_text())
    check_reference_meta(raw["meta"])
    return dict(raw)


def _ref(name: str) -> dict[str, Any]:
    raw = _bundle()
    if name in ("twogroup", "gpcm_tam"):
        return dict(raw[name])
    return dict(raw["cases"][name])


def _item_model(itemtype: str, n_categories: int) -> ItemModel:
    if itemtype == "2PL":
        return TwoPL()
    if itemtype == "3PL":
        return ThreePL()
    if itemtype == "graded":
        return GRM(n_categories=n_categories)
    if itemtype == "gpcm":
        return GPCM(n_categories=n_categories)
    return Nominal(n_categories=n_categories)


@cache
def _data(name: str) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Every row of the data, in mirt's order, and its group labels."""
    ref = _ref(name)
    pats = np.asarray(
        [[np.nan if v is None else v for v in row] for row in ref["patterns"]], dtype=float
    )
    rows = np.asarray(ref["row_pattern"], dtype=int)
    data = pd.DataFrame(pats[rows], columns=ref["items"])
    groups = np.asarray(ref["pattern_groups"])[rows] if name == "twogroup" else None
    return data, groups


def _fixed_fit(
    items: dict[str, ItemModel],
    params: dict[str | None, dict[str, Any]],
    data: pd.DataFrame,
    groups: np.ndarray | None,
) -> FitResult:
    """A ``FitResult`` with every parameter fixed at ``params`` (keyed by group or None)."""
    meta = _bundle()["meta"]
    spec = Model.from_items(items)
    if groups is not None:
        spec = spec.with_groups(
            **{
                g: Normal(mean=block["latent"]["mean"], var=block["latent"]["var"])
                for g, block in params.items()
                if g is not None
            }
        )
    table = spec.parameter_table(data, group=groups)
    values = []
    for group, item, param in zip(table["group"], table["item"], table["param"], strict=True):
        source = params[group if groups is not None else None]
        if item == "__latent__":
            values.append(float(source["latent"][param]))
        else:
            values.append(float(source["params"][item][param]))
    table["value"] = np.asarray(values)
    table["free"] = False
    quad = Quadrature.equispaced(int(meta["quadpts"]), bounds=tuple(meta["theta_lim"]))
    return fit(spec, data, group=groups, table=table, quadrature=quad, se=None)


@cache
def _at_mirt(name: str) -> FitResult:
    ref = _ref(name)
    items = {
        item: _item_model(t, k)
        for item, t, k in zip(ref["items"], ref["itemtype"], ref["n_categories"], strict=True)
    }
    data, groups = _data(name)
    params: dict[str | None, dict[str, Any]]
    if name == "twogroup":
        params = {g: ref["groups"][g] for g in ref["group_names"]}
    else:
        params = {None: ref}
    return _fixed_fit(items, params, data, groups)


def _blocks(name: str) -> list[tuple[str | None, dict[str, Any]]]:
    ref = _ref(name)
    if name == "twogroup":
        return [(g, ref["groups"][g]) for g in ref["group_names"]]
    return [(None, ref)]


@pytest.mark.parametrize("name", ALL)
def test_eap_scores_match_mirt(name: str) -> None:
    data, groups = _data(name)
    eap = _at_mirt(name).score(data, method="EAP", group=groups)["theta"]
    np.testing.assert_allclose(eap, _ref(name)["eap"], rtol=0, atol=TOL_EAP)


def _mirt_tables(name: str, item_index: int, group: str | None) -> dict[str, Any]:
    tables = _ref(name)["empirical"][item_index]
    return dict(tables if group is None else tables[group])


@pytest.mark.parametrize("name", ALL)
def test_empirical_item_curve_matches_mirt_itemfit_tables(name: str) -> None:
    ref = _ref(name)
    data, groups = _data(name)
    n_bins = int(_bundle()["meta"]["n_bins"])
    out = empirical_item_curve(_at_mirt(name), data, n_bins=n_bins, group=groups)
    expected_columns = ["item", "bin", "theta", "n", "category", "observed", "expected"]
    if name == "twogroup":
        expected_columns = ["group", *expected_columns]
    assert list(out.columns) == expected_columns
    for group, _ in _blocks(name):
        got_g = out if group is None else out[out["group"] == group]
        for j, item in enumerate(ref["items"]):
            got = got_g[got_g["item"] == item]
            tab = _mirt_tables(name, j, group)
            k_j = int(ref["n_categories"][j])
            assert len(tab["label"]) == n_bins
            assert got["bin"].nunique() == n_bins
            theta = got.groupby("bin")["theta"].first().to_numpy()
            n = got.groupby("bin")["n"].first().to_numpy()
            labels = np.asarray([float(re.sub("theta = ", "", s)) for s in tab["label"]])
            np.testing.assert_allclose(theta, labels, rtol=0, atol=TOL_BIN_THETA)
            observed = np.asarray(tab["observed"], dtype=float).reshape(n_bins, k_j)
            expected = np.asarray(tab["expected"], dtype=float).reshape(n_bins, k_j)
            np.testing.assert_array_equal(n, observed.sum(axis=1))
            np.testing.assert_allclose(
                got["observed"].to_numpy().reshape(n_bins, k_j) * n[:, None],
                observed,
                rtol=0,
                atol=TOL_TABLE,
            )
            np.testing.assert_allclose(
                got["expected"].to_numpy().reshape(n_bins, k_j),
                expected / n[:, None],
                rtol=0,
                atol=TOL_TABLE,
            )


@pytest.mark.parametrize("name", ALL)
def test_item_probabilities_match_mirt_probtrace(name: str) -> None:
    ref = _ref(name)
    res = _at_mirt(name)
    for group, block in _blocks(name):
        theta = np.asarray(block["theta"], dtype=float)
        out = item_probabilities(res, theta, group=group)
        assert list(out.columns) == ["item", "category", "theta", "probability"]
        for j, item in enumerate(ref["items"]):
            got = out[out["item"] == item]
            k_j = int(ref["n_categories"][j])
            wide = got["probability"].to_numpy().reshape(k_j, theta.size).T
            np.testing.assert_allclose(
                wide, np.asarray(block["probtrace"][j]), rtol=0, atol=TOL_PROBTRACE
            )


@pytest.mark.parametrize("name", ALL)
def test_wright_map_locations_match_uniroot_and_irt_b(name: str) -> None:
    ref = _ref(name)
    data, groups = _data(name)
    table = wright_map_table(_at_mirt(name), data, group=groups)
    np.testing.assert_allclose(table.persons["theta"], ref["eap"], rtol=0, atol=TOL_EAP)
    for group, block in _blocks(name):
        items = table.items if group is None else table.items[table.items["group"] == group]
        for j, item in enumerate(ref["items"]):
            got = items[items["item"] == item]
            want = np.atleast_1d(np.asarray(block["thresholds"][j], dtype=float))
            assert list(got["category"]) == list(range(1, want.size + 1))
            np.testing.assert_allclose(got["location"], want, rtol=0, atol=TOL_THRESHOLD)
            if item in block["irt_b"]:
                b = np.atleast_1d(np.asarray(block["irt_b"][item], dtype=float))
                np.testing.assert_allclose(got["location"], b, rtol=0, atol=TOL_THRESHOLD)
    if name == "mixed":
        # 3PL: P = 1/2 is below b, where P = (1 + g) / 2.
        items_3pl = _at_mirt(name).items
        b_3pl = float(items_3pl.loc[items_3pl["item"] == "q04", "b"].iloc[0])
        assert float(items[items["item"] == "q04"]["location"].iloc[0]) < b_3pl


def test_wright_map_locations_match_tam_threshold() -> None:
    ref = _ref("gpcm_tam")
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.integers(0, 4, size=(50, len(ref["items"]))), columns=ref["items"])
    items: dict[str, ItemModel] = {str(item): GPCM(n_categories=4) for item in ref["items"]}
    res = _fixed_fit(items, {None: ref}, data, None)
    table = wright_map_table(res, data)
    got = table.items["location"].to_numpy().reshape(len(ref["items"]), 3)
    np.testing.assert_allclose(got, np.asarray(ref["tam_threshold"]), rtol=0, atol=TOL_TAM)
