"""V37: information, expected score, reliability and summed-score EAP vs R mirt.

Reference values come from ``validation/r/m7_information_reference.R`` (DESIGN.md §18.2,
VALIDATION.md V37): mirt's ``iteminfo()``, ``testinfo()``, ``expected.item()``,
``expected.test()``, ``marginal_rxx()``, ``empirical_rxx()`` and
``fscores(method = "EAPsum")`` for 2PL, 3PL, GRM, GPCM, nominal and mixed-format tests and a
two-group model. mirt's estimates are put into a ``FitResult`` with every parameter fixed, so
only the statistics are compared.
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
    Model,
    Nominal,
    Normal,
    Quadrature,
    ThreePL,
    TwoPL,
    expected_score,
    fit,
    item_information,
    reliability,
    summed_score_table,
    test_information,
)
from flexpyirt.items import ItemModel
from tests._reference import REFERENCE_DIR, check_reference_meta

pytestmark = pytest.mark.slow

SINGLE = ["twopl", "threepl", "grm", "gpcm", "nominal", "mixed"]
COMPLETE = [*SINGLE, "twogroup"]

TOL_CURVE_REL = 1e-9
"""iteminfo / testinfo / expected.item / expected.test (measured: see VALIDATION.md V37)."""
TOL_MARGINAL_TIGHT = 1e-10
"""Marginal reliability vs integrate(rel.tol = 1e-12)."""
TOL_MARGINAL_DEFAULT = 1e-8
"""Marginal reliability vs marginal_rxx() with R's default integrate() tolerance."""
TOL_EMPIRICAL = 1e-9
TOL_EAPSUM = 1e-9


@cache
def _bundle() -> dict[str, Any]:
    raw = json.loads((REFERENCE_DIR / "v37_information.json").read_text())
    check_reference_meta(raw["meta"])
    return dict(raw)


def _ref(name: str) -> dict[str, Any]:
    raw = _bundle()
    return dict(raw["twogroup"] if name == "twogroup" else raw["cases"][name])


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
    """Unique response patterns (one row each) and their group labels."""
    ref = _ref(name)
    pats = np.asarray(
        [[np.nan if v is None else v for v in row] for row in ref["patterns"]], dtype=float
    )
    groups = np.asarray(ref["pattern_groups"]) if name == "twogroup" else None
    return pd.DataFrame(pats, columns=ref["items"]), groups


def _expanded(name: str) -> tuple[pd.DataFrame, np.ndarray | None]:
    pats, groups = _data(name)
    reps = np.asarray(_ref(name)["counts"], dtype=int)
    data = pats.loc[pats.index.repeat(reps)].reset_index(drop=True)
    return data, None if groups is None else np.repeat(groups, reps)


@cache
def _at_mirt(name: str) -> FitResult:
    """A ``FitResult`` with every parameter fixed at mirt's estimate."""
    ref = _ref(name)
    meta = _bundle()["meta"]
    spec = Model.from_items(
        {
            item: _item_model(t, k)
            for item, t, k in zip(ref["items"], ref["itemtype"], ref["n_categories"], strict=True)
        }
    )
    pats, groups = _data(name)
    if name == "twogroup":
        est = ref["groups"]
        spec = spec.with_groups(
            **{
                g: Normal(mean=est[g]["latent"]["mean"], var=est[g]["latent"]["var"])
                for g in ref["group_names"]
            }
        )
    else:
        est = {None: ref}
    table = spec.parameter_table(pats, group=groups)
    values = []
    for group, item, param in zip(table["group"], table["item"], table["param"], strict=True):
        source = est[group if name == "twogroup" else None]
        if item == "__latent__":
            values.append(float(source["latent"][param]))
        else:
            values.append(float(source["params"][item][param]))
    table["value"] = np.asarray(values)
    table["free"] = False
    quad = Quadrature.equispaced(int(meta["quadpts"]), bounds=tuple(meta["theta_lim"]))
    return fit(spec, pats, group=groups, table=table, quadrature=quad, se=None)


def _blocks(name: str) -> list[tuple[str | None, dict[str, Any]]]:
    """``(group, reference block)`` pairs: one for a single-group case, one per group otherwise."""
    ref = _ref(name)
    if name == "twogroup":
        return [(g, ref["groups"][g]) for g in ref["group_names"]]
    return [(None, ref)]


@pytest.mark.parametrize("name", COMPLETE)
def test_information_and_expected_score_match_mirt(name: str) -> None:
    res = _at_mirt(name)
    for group, ref in _blocks(name):
        theta = np.asarray(ref["theta"], dtype=float)
        info = item_information(res, theta, group=group)
        assert list(info.columns) == _ref(name)["items"]
        np.testing.assert_allclose(info.to_numpy(), ref["iteminfo"], rtol=TOL_CURVE_REL, atol=1e-12)
        total = test_information(res, theta, group=group)
        np.testing.assert_allclose(total["information"], ref["testinfo"], rtol=TOL_CURVE_REL)
        np.testing.assert_allclose(total["se"], 1.0 / np.sqrt(ref["testinfo"]), rtol=TOL_CURVE_REL)
        expected = expected_score(res, theta, group=group)
        np.testing.assert_allclose(
            expected[_ref(name)["items"]].to_numpy(),
            ref["expected_item"],
            rtol=TOL_CURVE_REL,
            atol=1e-12,
        )
        np.testing.assert_allclose(expected["total"], ref["expected_test"], rtol=TOL_CURVE_REL)


@pytest.mark.parametrize("name", SINGLE)
def test_marginal_reliability_matches_mirt(name: str) -> None:
    ref = _ref(name)
    out = reliability(_at_mirt(name))
    assert list(out.columns) == ["marginal"]
    marginal = float(out["marginal"].iloc[0])
    assert marginal == pytest.approx(ref["marginal_rxx_tight"], abs=TOL_MARGINAL_TIGHT)
    assert marginal == pytest.approx(ref["marginal_invariant"], abs=TOL_MARGINAL_TIGHT)
    assert marginal == pytest.approx(ref["marginal_rxx"], abs=TOL_MARGINAL_DEFAULT)


def test_marginal_reliability_two_groups_matches_standardized_mirt_model() -> None:
    ref = _ref("twogroup")
    out = reliability(_at_mirt("twogroup"))
    assert list(out.columns) == ["group", "marginal"]
    assert list(out["group"]) == ref["group_names"]
    for g, marginal in zip(out["group"], out["marginal"], strict=True):
        block = ref["groups"][g]
        assert marginal == pytest.approx(
            block["marginal_rxx_standardized_tight"], abs=TOL_MARGINAL_TIGHT
        )
        assert marginal == pytest.approx(block["marginal_invariant"], abs=TOL_MARGINAL_TIGHT)
        assert marginal == pytest.approx(
            block["marginal_rxx_standardized"], abs=TOL_MARGINAL_DEFAULT
        )
    # mirt's own formula on the focal group (variance != 1) is a different quantity.
    focal = ref["groups"]["focal"]
    assert abs(focal["marginal_rxx_tight"] - focal["marginal_invariant"]) > 1e-3


@pytest.mark.parametrize("name", [*COMPLETE, "twopl_missing"])
def test_empirical_reliability_matches_mirt(name: str) -> None:
    res = _at_mirt(name)
    data, groups = _expanded(name)
    out = reliability(res, data, group=groups)
    if name == "twogroup":
        ref = _ref(name)
        for g, value in zip(out["group"], out["empirical"], strict=True):
            assert value == pytest.approx(ref["groups"][g]["empirical_rxx"], abs=TOL_EMPIRICAL)
    else:
        assert float(out["empirical"].iloc[0]) == pytest.approx(
            _ref(name)["empirical_rxx"], abs=TOL_EMPIRICAL
        )


@pytest.mark.parametrize("name", COMPLETE)
def test_summed_score_table_matches_mirt(name: str) -> None:
    table = summed_score_table(_at_mirt(name))
    for group, ref in _blocks(name):
        got = table if group is None else table[table["group"] == group]
        tab = ref["eapsum"]
        np.testing.assert_array_equal(got["score"].to_numpy(), np.asarray(tab["sum_scores"]))
        np.testing.assert_allclose(got["theta"], tab["eap"], rtol=0, atol=TOL_EAPSUM)
        np.testing.assert_allclose(got["se"], tab["eap_sd"], rtol=0, atol=TOL_EAPSUM)
        np.testing.assert_allclose(
            got["probability"] * ref["n"], tab["expected"], rtol=1e-9, atol=1e-12
        )
        assert float(got["probability"].sum()) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("name", [*COMPLETE, "twopl_missing"])
def test_eapsum_scores_match_mirt(name: str) -> None:
    ref = _ref(name)
    pats, groups = _data(name)
    out = _at_mirt(name).score(pats, method="EAPsum", group=groups)
    assert list(out.columns) == ["theta", "se"]
    eap = np.asarray([np.nan if v is None else v for v in ref["eapsum_full"]["eap"]], dtype=float)
    sd = np.asarray([np.nan if v is None else v for v in ref["eapsum_full"]["eap_sd"]], dtype=float)
    incomplete = pats.isna().any(axis=1).to_numpy()
    # mirt returns NA exactly for the incomplete patterns, and so does flexpyirt.
    np.testing.assert_array_equal(np.isnan(eap), incomplete)
    np.testing.assert_array_equal(out["theta"].isna().to_numpy(), incomplete)
    np.testing.assert_array_equal(out["se"].isna().to_numpy(), incomplete)
    np.testing.assert_allclose(out["theta"], eap, rtol=0, atol=TOL_EAPSUM)
    np.testing.assert_allclose(out["se"], sd, rtol=0, atol=TOL_EAPSUM)
    if name == "twopl_missing":
        assert incomplete.any() and not incomplete.all()
