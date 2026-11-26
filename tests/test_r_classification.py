"""V52: classification accuracy and consistency vs cacIRT and a recomputation from R mirt.

Reference values come from ``validation/r/classification_reference.R`` (DESIGN.md §18.3,
VALIDATION.md V52): cacIRT's ``class.Lee()`` (3PL), ``Lee.poly.D()`` / ``Lee.poly.P()`` (GRM)
and ``Rud.D()`` / ``Rud.P()`` with mirt's information SE, together with an independent
recomputation in R from mirt's ``probtrace()``, ``expected.test()`` and ``testinfo()`` that
also gives kappa, the false positive / negative rates, the per-category indices and the
tables. The estimates (generating values for the 3PL case) are put into a ``FitResult`` with
every parameter fixed, so only the indices are compared.
"""

from __future__ import annotations

import json
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest

from flexpyirt import (
    GRM,
    ClassificationResult,
    FitResult,
    Model,
    Normal,
    Quadrature,
    ThreePL,
    TwoPL,
    classification_accuracy,
    fit,
)
from flexpyirt.items import ItemModel
from tests._reference import REFERENCE_DIR, check_reference_meta

pytestmark = pytest.mark.slow

TOL_D = 1e-12
"""D approach (same nodes and weights as R): measured 2e-14 (VALIDATION.md V52)."""
TOL_P = 1e-12
"""P approach, including the difference of the EAP scores from mirt's: measured 4e-15."""
TOL_EAP = 1e-12
TOL_POLY_CACIRT = 1e-10
"""cacIRT's Lee.poly.D() clips the category probabilities at 1e-4 (measured 4e-12)."""


@cache
def _bundle() -> dict[str, Any]:
    raw = json.loads((REFERENCE_DIR / "v52_classification.json").read_text())
    check_reference_meta(raw["meta"])
    return dict(raw)


def _item_model(itemtype: str, n_categories: int) -> ItemModel:
    if itemtype == "3PL":
        return ThreePL()
    if itemtype == "2PL":
        return TwoPL()
    return GRM(n_categories=n_categories)


def _ref(name: str) -> dict[str, Any]:
    raw = _bundle()
    return dict(raw["twogroup"] if name == "twogroup" else raw["cases"][name])


@cache
def _data(name: str) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Unique response patterns (one row each) and their group labels."""
    ref = _ref(name)
    pats = pd.DataFrame(np.asarray(ref["patterns"], dtype=float), columns=ref["items"])
    groups = np.asarray(ref["pattern_groups"]) if name == "twogroup" else None
    return pats, groups


def _expanded(name: str) -> tuple[pd.DataFrame, np.ndarray | None]:
    pats, groups = _data(name)
    reps = np.asarray(_ref(name)["counts"], dtype=int)
    data = pats.loc[pats.index.repeat(reps)].reset_index(drop=True)
    return data, None if groups is None else np.repeat(groups, reps)


@cache
def _at_r(name: str) -> FitResult:
    """A ``FitResult`` with every parameter fixed at the reference values."""
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
    table = spec.parameter_table(pats, group=groups)
    values = []
    for group, item, param, value in zip(
        table["group"], table["item"], table["param"], table["value"], strict=True
    ):
        source = ref["groups"][group] if name == "twogroup" else ref
        if item == "__latent__":
            values.append(float(source["latent"][param]) if "latent" in source else float(value))
        else:
            values.append(float(source["params"][item].get(param, value)))
    table["value"] = np.asarray(values)
    table["free"] = False
    quad = Quadrature.equispaced(int(meta["quadpts"]), bounds=tuple(meta["theta_lim"]))
    return fit(spec, pats, group=groups, table=table, quadrature=quad, se=None)


def _runs(name: str) -> list[tuple[str | None, int, dict[str, Any]]]:
    ref = _ref(name)
    if name == "twogroup":
        return [
            (g, i, r) for g in ref["group_names"] for i, r in enumerate(ref["groups"][g]["runs"])
        ]
    return [(None, i, r) for i, r in enumerate(ref["runs"])]


RUNS = [(name, i) for name in ("threepl", "grm") for _, i, _ in _runs(name)]


def _result(name: str, run: dict[str, Any]) -> ClassificationResult:
    res = _at_r(name)
    kwargs: dict[str, Any] = {
        "cut_scores": run["cuts"],
        "method": run["method"],
        "scale": run["scale"],
    }
    if run["approach"] == "P":
        data, groups = _expanded(name)
        kwargs["data"] = data
        kwargs["group"] = groups
    return classification_accuracy(res, **kwargs)


def _select(frame: pd.DataFrame, group: str | None) -> pd.DataFrame:
    if group is None:
        return frame
    return frame[frame["group"] == group].reset_index(drop=True)


def _compare(out: ClassificationResult, group: str | None, run: dict[str, Any]) -> None:
    tol = TOL_D if run["approach"] == "D" else TOL_P
    rec = run["recompute"]
    summary = _select(out.summary, group).iloc[0]
    for key in ("accuracy", "consistency", "kappa"):
        assert summary[key] == pytest.approx(rec[key], abs=tol), key
    by_cut = _select(out.by_cut, group)
    np.testing.assert_allclose(by_cut["cut"], np.atleast_1d(run["cuts"]), rtol=0, atol=0)
    converted = [c for c in ("summed_score_cut", "theta_cut") if c in by_cut.columns]
    if converted:
        np.testing.assert_allclose(
            by_cut[converted[0]], np.atleast_1d(run["method_cuts"]), rtol=0, atol=1e-9
        )
    for key, values in rec["by_cut"].items():
        np.testing.assert_allclose(by_cut[key], np.atleast_1d(values), rtol=0, atol=tol)
    by_category = _select(out.by_category, group)
    for key, values in rec["by_category"].items():
        np.testing.assert_allclose(by_category[key], values, rtol=0, atol=tol)
    a = out.accuracy_table if group is None else out.accuracy_table.loc[group]
    b = out.consistency_table if group is None else out.consistency_table.loc[group]
    np.testing.assert_allclose(a.to_numpy(), rec["accuracy_table"], rtol=0, atol=tol)
    np.testing.assert_allclose(b.to_numpy(), rec["consistency_table"], rtol=0, atol=tol)
    if run["approach"] == "D":
        cond = _select(out.conditional, group)
        np.testing.assert_allclose(cond["accuracy"], rec["conditional_accuracy"], atol=tol)
        np.testing.assert_allclose(cond["consistency"], rec["conditional_consistency"], atol=tol)
        if run["method"] == "lee":
            np.testing.assert_allclose(cond["true_score"], run["true_score"], rtol=1e-12)
        else:
            np.testing.assert_allclose(cond["se"], run["se"], rtol=1e-10)
    cac = run.get("cacirt")
    if cac is not None:
        cac_tol = tol if cac["exact"] else max(tol, TOL_POLY_CACIRT)
        assert summary["accuracy"] == pytest.approx(cac["accuracy"], abs=cac_tol)
        assert summary["consistency"] == pytest.approx(cac["consistency"], abs=cac_tol)
        np.testing.assert_allclose(by_cut["accuracy"], cac["per_cut_accuracy"], atol=cac_tol)
        np.testing.assert_allclose(by_cut["consistency"], cac["per_cut_consistency"], atol=cac_tol)


@pytest.mark.parametrize(("name", "index"), RUNS)
def test_single_group_matches_cacirt_and_recomputation(name: str, index: int) -> None:
    run = _ref(name)["runs"][index]
    _compare(_result(name, run), None, run)


@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_two_groups_match_cacirt_and_recomputation(index: int) -> None:
    ref = _ref("twogroup")
    run = ref["groups"]["ref"]["runs"][index]
    out = _result("twogroup", run)
    assert list(out.summary["group"]) == ref["group_names"]
    for g in ref["group_names"]:
        _compare(out, g, ref["groups"][g]["runs"][index])


@pytest.mark.parametrize("name", ["threepl", "grm", "twogroup"])
def test_eap_scores_match_mirt(name: str) -> None:
    """The P approach rests on the EAP scores; they agree with mirt's."""
    pats, groups = _data(name)
    scores = _at_r(name).score(pats, method="EAP", group=groups)
    np.testing.assert_allclose(scores["theta"], _ref(name)["eap"], rtol=0, atol=TOL_EAP)
