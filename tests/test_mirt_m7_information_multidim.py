"""V43: information matrix, directional information and expected scores of multidimensional models.

Reference values come from ``validation/r/m7_information_multidim_reference.R`` (DESIGN.md
§18.3, VALIDATION.md V43): mirt's ``iteminfo(multidim_matrix = TRUE)``,
``iteminfo(degrees = )``, ``testinfo(degrees = )``, ``expected.item()`` and ``expected.test()``
at chosen parameter values of a two-factor model with every item family and cross-loadings, a
three-factor model, a bifactor model and a two-tier model. The same values are put into a
``FitResult`` with every parameter fixed.
"""

from __future__ import annotations

import json
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest

from flexpyirt import (
    GPCM,
    GRM,
    FitResult,
    FourPL,
    Model,
    Nominal,
    ThreePL,
    TwoPL,
    expected_score,
    fit,
    information_matrix,
    item_information,
    test_information,
)
from flexpyirt.items import ItemModel
from tests._reference import REFERENCE_DIR, check_reference_meta

pytestmark = pytest.mark.slow

CASES = ["cfa2_mixed", "cfa3", "bifactor", "two_tier"]

TOL_REL = 1e-10
"""Relative tolerance of every quantity (measured: see VALIDATION.md V43)."""
TOL_ABS = 1e-13
"""Absolute floor for entries that are exactly 0 in mirt (slopes fixed at 0)."""


@cache
def _bundle() -> dict[str, Any]:
    raw = json.loads((REFERENCE_DIR / "v43_information_multidim.json").read_text())
    check_reference_meta(raw["meta"])
    return dict(raw)


def _ref(name: str) -> dict[str, Any]:
    return dict(_bundle()["cases"][name])


def _item_model(itemtype: str, n_categories: int) -> ItemModel:
    models: dict[str, ItemModel] = {
        "2PL": TwoPL(),
        "3PL": ThreePL(),
        "4PL": FourPL(),
        "graded": GRM(n_categories=n_categories),
        "gpcm": GPCM(n_categories=n_categories),
        "nominal": Nominal(n_categories=n_categories),
    }
    return models[itemtype]


def _loaded(ref: dict[str, Any], factor: int) -> list[str]:
    """Items whose slope on factor ``factor`` (0-based) is not 0 in the reference."""
    key = f"a{factor + 1}"
    return [item for item in ref["items"] if ref["params"][item].get(key, 0) != 0]


def _spec(name: str) -> Model:
    ref = _ref(name)
    base = Model.from_items(
        {
            item: _item_model(t, k)
            for item, t, k in zip(ref["items"], ref["itemtype"], ref["n_categories"], strict=True)
        }
    )
    factors = ref["factors"]
    if ref["structure"] == "factors":
        return base.with_factors({f: _loaded(ref, m) for m, f in enumerate(factors)})
    if ref["structure"] == "bifactor":
        return base.with_bifactor(
            specific={f: _loaded(ref, m) for m, f in enumerate(factors) if m > 0},
            general=factors[0],
        )
    return base.with_two_tier(
        primary={f: _loaded(ref, m) for m, f in enumerate(factors[:2])},
        specific={f: _loaded(ref, m + 2) for m, f in enumerate(factors[2:])},
    )


@cache
def _at_mirt(name: str) -> FitResult:
    """A ``FitResult`` with every item parameter fixed at the reference values."""
    ref = _ref(name)
    spec = _spec(name)
    k = ref["n_categories"]
    rows = np.asarray(
        [[min(c, kj - 1) for kj in k] for c in range(max(k))] + [[0] * len(k)], dtype=float
    )
    dummy = pd.DataFrame(rows, columns=ref["items"])
    table = spec.parameter_table(dummy)
    values = []
    for item, param, value in zip(table["item"], table["param"], table["value"], strict=True):
        values.append(float(value) if item == "__latent__" else float(ref["params"][item][param]))
    table["value"] = np.asarray(values)
    table["free"] = False
    return fit(spec, dummy, table=table, se=None)


@pytest.mark.parametrize("name", CASES)
def test_information_matrices_match_mirt(name: str) -> None:
    ref = _ref(name)
    res = _at_mirt(name)
    theta = np.asarray(ref["theta"], dtype=float)
    expected = np.asarray(ref["info_matrix"], dtype=float)  # (J, T, D, D)
    for j, item in enumerate(ref["items"]):
        got = information_matrix(res, theta, item=item)
        np.testing.assert_allclose(got, expected[j], rtol=TOL_REL, atol=TOL_ABS)
    np.testing.assert_allclose(
        information_matrix(res, theta), expected.sum(axis=0), rtol=TOL_REL, atol=TOL_ABS
    )


@pytest.mark.parametrize("name", CASES)
def test_directional_information_matches_mirt(name: str) -> None:
    ref = _ref(name)
    res = _at_mirt(name)
    theta = pd.DataFrame(np.asarray(ref["theta"], dtype=float), columns=ref["factors"])
    for block in ref["directions"]:
        direction = np.cos(np.radians(np.asarray(block["degrees"], dtype=float)))
        info = item_information(res, theta, direction=direction)
        assert list(info.columns) == ref["items"]
        assert list(info.index.names) == ref["factors"]
        np.testing.assert_allclose(info.to_numpy(), block["iteminfo"], rtol=TOL_REL, atol=TOL_ABS)
        total = test_information(res, theta, direction=direction)
        np.testing.assert_allclose(total["information"], block["testinfo"], rtol=TOL_REL)
        np.testing.assert_allclose(total["se"], block["se"], rtol=TOL_REL)


@pytest.mark.parametrize("name", CASES)
def test_expected_scores_match_mirt(name: str) -> None:
    ref = _ref(name)
    res = _at_mirt(name)
    theta = np.asarray(ref["theta"], dtype=float)
    out = expected_score(res, theta)
    np.testing.assert_allclose(
        out[ref["items"]].to_numpy(), ref["expected_item"], rtol=TOL_REL, atol=TOL_ABS
    )
    np.testing.assert_allclose(out["total"], ref["expected_test"], rtol=TOL_REL)
