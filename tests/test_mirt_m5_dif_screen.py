"""V49: DIF screening, Wald DIF tests and DIF effect sizes vs R mirt (DESIGN.md §16.22).

Reference values come from ``validation/r/m5_dif_screen_reference.R``: mirt's ``DIF()`` with
``scheme = "drop"`` / ``"drop_sequential"`` / ``"add_sequential"`` (``seq_stat = 0.05``,
``p.adjust = "BH"``, ``max_run`` 2, 3 and ``Inf``) and ``Wald = TRUE`` on a two-group test of
8 2PL and 4 graded items, ``scheme = "drop"`` on a three-group 2PL test, and ``DRF()``,
``empirical_ES()`` and ``RMSD_DIF()`` at mirt's estimates of a partially invariant and an invariant
two-group model (put into fixed ``FitResult`` objects, so only the statistics are compared).
"""

from __future__ import annotations

import json
import warnings
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest

from flexpyirt import (
    BAEM,
    GRM,
    DIFScreenResult,
    FitResult,
    Model,
    Normal,
    Quadrature,
    TwoPL,
    dif_effect_sizes,
    dif_screen,
    fit,
)
from flexpyirt._dif import adjust_p
from flexpyirt.errors import ConvergenceWarning
from flexpyirt.items import ItemModel
from tests._reference import REFERENCE_DIR, check_reference_meta

pytestmark = pytest.mark.slow

QUAD = Quadrature.equispaced(81, bounds=(-8, 8))
ES_QUAD = Quadrature.equispaced(201, bounds=(-6, 6))
METHOD = BAEM(tol=1e-9, max_iter=20000)
TOL_X2 = 1e-8
"""Absolute; measured in VALIDATION.md V49."""
TOL_P_REL = 1e-3
TOL_LOGLIK_REL = 1e-8
TOL_WALD_REL = 1e-5
"""Relative; Oakes standard errors of two independent fits."""
TOL_EFFECT = 1e-12
"""Effect sizes at mirt's estimates (no estimation involved)."""
TOL_ESSD = 1e-12
TOL_RMSD = 1e-12


@cache
def _raw() -> dict[str, Any]:
    raw = json.loads((REFERENCE_DIR / "v49_dif_screen.json").read_text())
    check_reference_meta(raw["meta"], require_grid_check=True)
    return dict(raw)


def _expand(block: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray]:
    pats = pd.DataFrame(np.asarray(block["patterns"], dtype=float), columns=block["items"])
    counts = np.asarray(block["counts"], dtype=int)
    data = pats.loc[pats.index.repeat(counts)].reset_index(drop=True)
    return data, np.repeat(np.asarray(block["pattern_groups"]), counts)


def _item_models(block: dict[str, Any]) -> dict[str, ItemModel]:
    out: dict[str, ItemModel] = {}
    for item, kind, k in zip(block["items"], block["itemtype"], block["n_categories"], strict=True):
        out[item] = TwoPL() if kind == "2PL" else GRM(n_categories=int(k))
    return out


def _screen_spec() -> Model:
    block = _raw()["screen_data"]
    return Model.from_items(_item_models(block)).with_groups(
        ref=Normal.standard(), focal=Normal.free()
    )


@cache
def _screen(name: str, max_iter: int = 10) -> DIFScreenResult:
    raw = _raw()
    data, group = _expand(raw["screen_data"])
    kwargs: dict[str, Any] = {}
    if name == "all_others_intercepts":
        kwargs["params"] = ["d", "d1", "d2"]
    if name == "sequential_constrain":
        kwargs["anchor_items"] = raw["screen_data"]["anchors"]
    strategy = "all_others" if name.startswith("all_others") else name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return dif_screen(
            _screen_spec(),
            data,
            group=group,
            strategy=strategy,  # type: ignore[arg-type]
            max_iter=max_iter,
            method=METHOD,
            quadrature=QUAD,
            **kwargs,
        )


def _check_table(out: pd.DataFrame, ref: dict[str, Any], *, adj: bool = True) -> None:
    assert list(out["item"]) == list(ref["item"])
    assert out["converged"].all()
    np.testing.assert_array_equal(out["df"], ref["df"])
    np.testing.assert_allclose(out["x2"], ref["x2"], rtol=0, atol=TOL_X2)
    np.testing.assert_allclose(out["p"], ref["p"], rtol=TOL_P_REL, atol=1e-10)
    if adj:
        np.testing.assert_allclose(out["adj_p"], ref["adj_p"], rtol=TOL_P_REL, atol=1e-10)


@pytest.mark.parametrize("name", ["all_others", "all_others_intercepts"])
def test_all_others_matches_mirt_drop(name: str) -> None:
    ref = _raw()["screen"][name]
    out = _screen(name)
    _check_table(out.table, ref)
    flagged = [i for i, a in zip(ref["item"], ref["adj_p"], strict=True) if a < 0.05]
    assert out.flagged_items == flagged


@pytest.mark.parametrize("name", ["sequential_free", "sequential_constrain"])
@pytest.mark.parametrize("run", [0, 1, 2])
def test_sequential_matches_mirt(name: str, run: int) -> None:
    ref_case = _raw()["screen"][name]
    ref = ref_case["runs"][run]
    max_run = ref["max_run"]
    out = _screen(name, 10 if max_run == "Inf" else int(max_run) - 1)
    # mirt reports only the flagged items, each tested against the final anchor set.
    assert out.flagged_items == list(ref["item"])
    final = out.table[out.table["flagged"]].reset_index(drop=True)
    _check_table(final, ref, adj=False)
    np.testing.assert_allclose(
        adjust_p(final["p"].to_numpy(), "BH"), ref["adj_p"], rtol=TOL_P_REL, atol=1e-10
    )
    if max_run == "Inf":
        assert out.converged
        assert out.fit.loglik == pytest.approx(ref_case["final_loglik"], rel=TOL_LOGLIK_REL)
        assert out.fit.n_params == ref_case["final_nest"]


def test_wald_matches_mirt() -> None:
    raw = _raw()
    ref = raw["screen"]["wald"]
    data, group = _expand(raw["screen_data"])
    anchors = raw["screen_data"]["anchors"]
    res = fit(
        _screen_spec().equal_across_groups(items=anchors),
        data,
        group=group,
        method=METHOD,
        quadrature=QUAD,
        se="oakes",
    )
    assert res.loglik == pytest.approx(raw["screen"]["base_loglik"]["anchor_se"], rel=1e-9)
    out = res.dif(items=ref["item"], test="wald", p_adjust="holm")
    assert list(out["item"]) == list(ref["item"])
    np.testing.assert_array_equal(out["df"], ref["df"])
    np.testing.assert_allclose(out["x2"], ref["x2"], rtol=TOL_WALD_REL)
    np.testing.assert_allclose(out["p"], ref["p"], rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(out["adj_p"], ref["adj_p"], rtol=1e-5, atol=1e-10)


@pytest.mark.parametrize("name", ["all_others", "all_others_intercepts"])
def test_three_groups_matches_mirt(name: str) -> None:
    raw = _raw()
    ref = raw["three"][name]
    data, group = _expand(raw["three_data"])
    items = raw["three_data"]["items"]
    spec = Model.uniform(TwoPL(), items=items).with_groups(
        g1=Normal.standard(), g2=Normal.free(), g3=Normal.free()
    )
    out = dif_screen(
        spec,
        data,
        group=group,
        params=["d"] if name.endswith("intercepts") else None,
        method=METHOD,
        quadrature=QUAD,
    )
    _check_table(out.table, ref)


# ---- effect sizes at mirt's estimates --------------------------------------------------------


@cache
def _at_mirt(which: str) -> tuple[FitResult, pd.DataFrame, np.ndarray]:
    raw = _raw()
    block = raw["screen_data"]
    est = raw["effect_sizes"][which]
    spec = Model.from_items(_item_models(block)).with_groups(
        **{
            g: Normal(mean=est[g]["latent"]["mean"], var=est[g]["latent"]["var"])
            for g in block["group_names"]
        }
    )
    data, group = _expand(block)
    table = spec.parameter_table(data, group=group)
    values = []
    for g, item, param in zip(table["group"], table["item"], table["param"], strict=True):
        source = est[g]
        values.append(
            float(
                source["latent"][param] if item == "__latent__" else source["params"][item][param]
            )
        )
    table["value"] = np.asarray(values)
    table["free"] = False
    res = fit(spec, data, group=group, table=table, quadrature=ES_QUAD, se=None)
    return res, data, group


@pytest.mark.parametrize("density", ["focal", "reference"])
def test_signed_unsigned_differences_match_drf(density: str) -> None:
    es = _raw()["effect_sizes"]
    res, _, _ = _at_mirt("partial")
    out = dif_effect_sizes(res, density=density)  # type: ignore[arg-type]
    items = out[out["level"] == "item"]
    ref = es[f"drf_{density}"]
    assert list(items["item"]) == list(ref["item"])
    np.testing.assert_allclose(items["signed_difference"], ref["signed"], rtol=0, atol=TOL_EFFECT)
    np.testing.assert_allclose(
        items["unsigned_difference"], ref["unsigned"], rtol=0, atol=TOL_EFFECT
    )
    test = out[out["level"] == "test"].iloc[0]
    assert test["signed_difference"] == pytest.approx(
        es[f"dtf_{density}"]["signed"], abs=TOL_EFFECT
    )
    assert test["unsigned_difference"] == pytest.approx(
        es[f"dtf_{density}"]["unsigned"], abs=TOL_EFFECT
    )


def test_essd_and_rmsd_match_mirt() -> None:
    es = _raw()["effect_sizes"]
    res, data, group = _at_mirt("partial")
    theta = res.score(data, group=group)["theta"].to_numpy()[group == "focal"]
    # The data are rebuilt from unique patterns, so rows are in another order than in R.
    np.testing.assert_allclose(np.sort(theta), np.sort(es["eap_focal"]), rtol=0, atol=1e-12)
    out = dif_effect_sizes(res, data=data, group=group)
    items = out[out["level"] == "item"]
    np.testing.assert_allclose(
        items["expected_score_standardized_difference"], es["essd"], rtol=0, atol=TOL_ESSD
    )
    test = out[out["level"] == "test"].iloc[0]
    assert test["expected_score_standardized_difference"] == pytest.approx(
        es["etssd"], abs=TOL_ESSD
    )
    np.testing.assert_allclose(
        items["rmsd_reference"], es["rmsd_partial"]["ref"], rtol=0, atol=TOL_RMSD
    )
    np.testing.assert_allclose(
        items["rmsd_focal"], es["rmsd_partial"]["focal"], rtol=0, atol=TOL_RMSD
    )


def test_rmsd_of_invariant_model_matches_mirt() -> None:
    es = _raw()["effect_sizes"]
    res, data, group = _at_mirt("invariant")
    out = dif_effect_sizes(res, data=data, group=group)
    items = out[out["level"] == "item"]
    np.testing.assert_allclose(
        items["rmsd_reference"], es["rmsd_invariant"]["ref"], rtol=0, atol=TOL_RMSD
    )
    np.testing.assert_allclose(
        items["rmsd_focal"], es["rmsd_invariant"]["focal"], rtol=0, atol=TOL_RMSD
    )
    # Invariant parameters: no expected-score differences.
    np.testing.assert_allclose(items["signed_difference"], 0.0, atol=1e-12)
