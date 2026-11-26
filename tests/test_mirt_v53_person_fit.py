"""V53: person fit (l_z, l_z*, infit / outfit) and item infit / outfit vs R (DESIGN.md §20).

Reference values come from ``validation/r/person_fit_reference.R`` on V37's data and mirt
estimates (2PL, 2PL with 8% missing, 3PL, GRM, GPCM, nominal, mixed format, two groups) plus a
Rasch refit of the V37 2PL data: mirt's ``personfit()`` and ``itemfit(fit_stats = "infit")``,
``sirt::personfit.stat()`` for the Rasch model, and an R recomputation of l_z* from the formulas.
mirt's estimates are put into a ``FitResult`` with every parameter fixed, so only the statistics
(and the ability estimates) are compared.

Two comparisons per case and estimator (EAP, MAP, ML, WLE):

- **at mirt's theta**: the statistics evaluated at mirt's ``fscores()`` estimates (strict: they
  differ only by floating point and by the central differences of the R l_z*);
- **end to end**: ``person_fit()`` / ``item_fit()`` with flexpyirt's own estimates, which agree
  with mirt's to the V13 / V37 tolerances.
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
    Rasch,
    ThreePL,
    TwoPL,
    _personfit,
    fit,
)
from flexpyirt.items import ItemModel
from flexpyirt.result import PersonFitStatistic
from tests._reference import REFERENCE_DIR, check_reference_meta

pytestmark = pytest.mark.slow

CASES = ["twopl", "twopl_missing", "threepl", "grm", "gpcm", "nominal", "mixed", "twogroup"]
METHODS = ["EAP", "MAP", "ML", "WLE"]
PERSON_COLUMNS = ["lz", "infit", "z_infit", "outfit", "z_outfit"]

TOL_AT_THETA = 1e-9
"""Relative tolerance of l_z and infit / outfit at mirt's theta (measured: VALIDATION.md V53)."""
TOL_LZ_STAR = 1e-7
"""Absolute tolerance of l_z* vs the R recomputation with central differences."""
LZ_STAR_THETA_MAX = 10.0
"""l_z* is compared with the R recomputation for |theta| <= 10 only (see the test)."""
TOL_THETA = {"EAP": 1e-8, "MAP": 1e-5, "ML": 1e-4, "WLE": 1e-4}
"""Ability estimates vs mirt (V13 / V37 tolerances)."""
TOL_END_TO_END = 1e-4
"""Absolute tolerance of the statistics with flexpyirt's own ability estimates."""


def _num(values: list[Any]) -> np.ndarray:
    out = np.empty(len(values))
    for i, v in enumerate(values):
        out[i] = np.nan if v is None else np.inf if v == "Inf" else -np.inf if v == "-Inf" else v
    return out


@cache
def _bundle() -> dict[str, Any]:
    raw = json.loads((REFERENCE_DIR / "v53_person_fit.json").read_text())
    check_reference_meta(raw["meta"])
    return dict(raw)


@cache
def _v37() -> dict[str, Any]:
    return dict(json.loads((REFERENCE_DIR / "v37_information.json").read_text()))


def _ref(name: str) -> dict[str, Any]:
    raw = _bundle()
    return dict(raw["twogroup"] if name == "twogroup" else raw["cases"][name])


def _source(name: str) -> dict[str, Any]:
    v37 = _v37()
    return dict(v37["twogroup"] if name == "twogroup" else v37["cases"][name])


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
def _patterns(name: str) -> tuple[pd.DataFrame, np.ndarray | None, np.ndarray]:
    """Unique patterns, their groups and their counts."""
    src = _source(name)
    pats = np.asarray(
        [[np.nan if v is None else v for v in row] for row in src["patterns"]], dtype=float
    )
    groups = np.asarray(src["pattern_groups"]) if name == "twogroup" else None
    return pd.DataFrame(pats, columns=src["items"]), groups, np.asarray(src["counts"], dtype=int)


def _expanded(name: str) -> tuple[pd.DataFrame, np.ndarray | None]:
    pats, groups, counts = _patterns(name)
    data = pats.loc[pats.index.repeat(counts)].reset_index(drop=True)
    return data, None if groups is None else np.repeat(groups, counts)


def _stored(name: str) -> tuple[pd.DataFrame, np.ndarray | None]:
    pats, groups, _ = _patterns(name)
    n = int(_ref(name)["n_store"])
    return pats.iloc[:n].reset_index(drop=True), None if groups is None else groups[:n]


@cache
def _at_mirt(name: str) -> FitResult:
    """A ``FitResult`` with every parameter fixed at mirt's (V37) estimates."""
    src = _source(name)
    meta = _v37()["meta"]
    spec = Model.from_items(
        {
            item: _item_model(t, k)
            for item, t, k in zip(src["items"], src["itemtype"], src["n_categories"], strict=True)
        }
    )
    pats, groups, _ = _patterns(name)
    if name == "twogroup":
        est = src["groups"]
        spec = spec.with_groups(
            **{
                g: Normal(mean=est[g]["latent"]["mean"], var=est[g]["latent"]["var"])
                for g in src["group_names"]
            }
        )
    table = spec.parameter_table(pats, group=groups)
    values = []
    for group, item, param in zip(table["group"], table["item"], table["param"], strict=True):
        block = src["groups"][group] if name == "twogroup" else src
        source = block["latent"] if item == "__latent__" else block["params"][item]
        values.append(float(source[param]))
    table["value"] = np.asarray(values)
    table["free"] = False
    quad = Quadrature.equispaced(int(meta["quadpts"]), bounds=tuple(meta["theta_lim"]))
    return fit(spec, pats, group=groups, table=table, quadrature=quad, se=None)


@cache
def _rasch() -> FitResult:
    ref = _bundle()["rasch_sirt"]
    pats, _, _ = _patterns("twopl")
    spec = Model.uniform(Rasch(), items=list(pats.columns))
    table = spec.parameter_table(pats)
    for j, item in enumerate(pats.columns):
        table.loc[(table["item"] == item) & (table["param"] == "d"), "value"] = ref["d"][j]
    table.loc[(table["item"] == "__latent__") & (table["param"] == "var"), "value"] = ref["var"]
    table["free"] = False
    meta = _v37()["meta"]
    quad = Quadrature.equispaced(int(meta["quadpts"]), bounds=tuple(meta["theta_lim"]))
    return fit(spec, pats, table=table, quadrature=quad, se=None)


def _at_theta(
    res: FitResult, data: pd.DataFrame, groups: np.ndarray | None, method: str, theta: np.ndarray
) -> dict[str, np.ndarray]:
    """Person statistics evaluated at the given ability estimates."""
    _, per_row, _, _ = res._residual_rows(
        data,
        method=method,  # type: ignore[arg-type]
        group=groups,
        item_names=None,
        theta=theta,
    )
    infit, z_infit, outfit, z_outfit = _personfit.mean_square(
        per_row["resid2"], per_row["w"], per_row["c"], axis=1
    )
    return {
        "lz": per_row["lz"],
        "lz_star": per_row["lz_star"],
        "infit": infit,
        "z_infit": z_infit,
        "outfit": outfit,
        "z_outfit": z_outfit,
    }


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("name", CASES)
def test_person_fit_at_mirt_theta(name: str, method: str) -> None:
    """l_z, infit / outfit (mirt personfit) and l_z* (R recomputation) at mirt's theta."""
    ref = _ref(name)["persons"][method]
    data, groups = _stored(name)
    theta = _num(ref["theta"])
    finite = np.isfinite(theta)
    assert finite.sum() > 0.9 * finite.size
    got = _at_theta(_at_mirt(name), data, groups, method, theta)
    for col in PERSON_COLUMNS:
        np.testing.assert_allclose(
            got[col][finite], _num(ref[col])[finite], rtol=TOL_AT_THETA, atol=1e-12, err_msg=col
        )
        assert np.isnan(got[col][~finite]).all()
    complete = finite & data.notna().all(axis=1).to_numpy()
    np.testing.assert_allclose(
        got["z_outfit"][complete], _num(ref["z_outfit_mirt"])[complete], rtol=TOL_AT_THETA
    )
    if method != "EAP":
        # Beyond |theta| = 10 (mirt's ML on 3PL flat tails and on its missing-data extremes,
        # V13) the R central differences of probtrace() lose precision; the 2PL closed form in
        # test_person_fit.py covers flexpyirt there.
        moderate = finite & (np.abs(theta) <= LZ_STAR_THETA_MAX)
        np.testing.assert_allclose(
            got["lz_star"][moderate], _num(ref["lz_star"])[moderate], rtol=0, atol=TOL_LZ_STAR
        )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("name", CASES)
def test_person_fit_end_to_end(name: str, method: str) -> None:
    """``person_fit()`` with flexpyirt's own estimates agrees with the references."""
    ref = _ref(name)["persons"][method]
    data, groups = _stored(name)
    stats: tuple[PersonFitStatistic, ...] = ("lz", "infit", "outfit")
    if method != "EAP":
        stats = (*stats, "lz_star")
    got = _at_mirt(name).person_fit(
        data,
        method=method,  # type: ignore[arg-type]
        group=groups,
        statistics=stats,
    )
    theta_ref = _num(ref["theta"])
    theta = got["theta"].to_numpy()
    both = np.isfinite(theta_ref) & np.isfinite(theta)
    np.testing.assert_allclose(theta[both], theta_ref[both], rtol=0, atol=TOL_THETA[method])
    cols = PERSON_COLUMNS + (["lz_star"] if method != "EAP" else [])
    for col in cols:
        np.testing.assert_allclose(
            got[col].to_numpy()[both], _num(ref[col])[both], atol=TOL_END_TO_END, err_msg=col
        )
    np.testing.assert_array_equal(got["n_items"], data.notna().sum(axis=1))


@pytest.mark.parametrize("method", ["EAP", "MAP", "WLE"])
@pytest.mark.parametrize("name", CASES)
def test_item_infit_outfit(name: str, method: str) -> None:
    """Item infit / outfit vs mirt ``itemfit(fit_stats = "infit")`` (per group for two groups)."""
    data, groups = _expanded(name)
    got = _at_mirt(name).item_fit(
        data,
        statistics=("infit", "outfit"),
        method=method,  # type: ignore[arg-type]
        group=groups,
    )
    ref = _ref(name)["item_fit"][method]
    blocks = [(g, ref[g]) for g in ("ref", "focal")] if name == "twogroup" else [(None, ref)]
    for g, block in blocks:
        part = got if g is None else got[got["group"] == g]
        assert list(part["item"]) == block["item"]
        for col in ("infit", "z_infit", "outfit", "z_outfit"):
            np.testing.assert_allclose(
                part[col].to_numpy(), np.asarray(block[col]), atol=TOL_END_TO_END, err_msg=col
            )
        if data.notna().all(axis=None):  # mirt's own z.outfit is right without missing data
            np.testing.assert_allclose(
                part["z_outfit"].to_numpy(), np.asarray(block["z_outfit_mirt"]), atol=TOL_END_TO_END
            )


def test_rasch_matches_mirt_and_sirt() -> None:
    """Rasch WLE: l_z, infit and outfit vs mirt personfit and sirt personfit.stat."""
    ref = _bundle()["rasch_sirt"]
    pats, _, _ = _patterns("twopl")
    data = pats.iloc[: int(ref["n_store"])].reset_index(drop=True)
    theta = _num(ref["persons"]["WLE"]["theta"])
    got = _at_theta(_rasch(), data, None, "WLE", theta)
    for col in PERSON_COLUMNS:
        np.testing.assert_allclose(
            got[col], _num(ref["persons"]["WLE"][col]), rtol=TOL_AT_THETA, err_msg=col
        )
    for col in ("lz", "infit", "outfit"):
        np.testing.assert_allclose(got[col], np.asarray(ref["sirt"][col]), rtol=TOL_AT_THETA)
    end = _rasch().person_fit(data, method="WLE")
    np.testing.assert_allclose(end["theta"], theta, atol=TOL_THETA["WLE"])
    np.testing.assert_allclose(end["lz"], ref["sirt"]["lz"], atol=TOL_END_TO_END)
    full, _ = _expanded("twopl")
    item = _rasch().item_fit(full, statistics=("infit", "outfit"), method="WLE")
    for col in ("infit", "z_infit", "outfit", "z_outfit"):
        np.testing.assert_allclose(
            item[col], ref["item_fit"]["WLE"][col], atol=TOL_END_TO_END, err_msg=col
        )
