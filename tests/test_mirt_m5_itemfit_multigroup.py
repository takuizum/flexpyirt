"""V15 (multiple groups): S-X2 per group vs R mirt's ``itemfit()`` on ``multipleGroup`` fits.

Reference values come from ``validation/r/m5_itemfit_multigroup_reference.R`` (DESIGN.md §15.45):
``twogroup_2pl`` evaluates mirt's S-X2 at its V4 estimates; ``science_grm_2g`` is the Science
data (GRM) split into two groups by row parity with all items invariant and the focal mean and
variance free. flexpyirt refits both. For ``science_grm_2g`` mirt stops ~2e-4 short in the item
parameters (flexpyirt's log-likelihood is higher), so S-X2 is evaluated at mirt's estimates: the
fitted result with its free parameter vector replaced by mirt's values.
"""

from __future__ import annotations

import dataclasses
import json
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from flexpyirt import BAEM, GRM, FitResult, Model, Normal, Quadrature, TwoPL, fit
from flexpyirt.errors import SpecificationError
from tests._reference import check_reference_meta

REFERENCE_DIR = Path(__file__).parent / "reference"
TOL_SX2_REL = 1e-6
CASES = ["twogroup_2pl", "science_grm_2g"]


@cache
def _bundle() -> dict[str, Any]:
    raw = json.loads((REFERENCE_DIR / "v15_sx2_multigroup.json").read_text())
    check_reference_meta(raw["meta"])
    return dict(raw["cases"])


def _expand(raw: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray]:
    pats = pd.DataFrame(np.asarray(raw["patterns"], dtype=float), columns=raw["items"])
    reps = np.asarray(raw["counts"], dtype=int)
    df = pats.loc[pats.index.repeat(reps)].reset_index(drop=True)
    return df, np.repeat(np.asarray(raw["pattern_groups"]), reps)


@cache
def _case(name: str) -> tuple[FitResult, pd.DataFrame, np.ndarray]:
    ref = _bundle()[name]
    method = BAEM(tol=1e-10, max_iter=50000)
    if name == "twogroup_2pl":
        raw = json.loads((REFERENCE_DIR / f"{ref['source']}.json").read_text())
        df, grp = _expand(raw)
        quad = Quadrature.equispaced(raw["meta"]["quadpts"], bounds=tuple(raw["meta"]["theta_lim"]))
        spec = (
            Model.uniform(TwoPL(), items=raw["items"])
            .with_groups(ref=Normal.standard(), focal=Normal.free())
            .equal_across_groups(items=raw["anchors"])
        )
    else:
        df, grp = _expand(ref)
        quad = Quadrature.equispaced(ref["quadpts"], bounds=tuple(ref["theta_lim"]))
        spec = (
            Model.uniform(GRM(n_categories=4), items=ref["items"])
            .with_groups(ref=Normal.standard(), focal=Normal.free())
            .equal_across_groups(items=ref["items"])
        )
    res = fit(spec, df, group=grp, method=method, quadrature=quad, se=None)
    assert res.converged
    return res, df, grp


def _at_mirt(res: FitResult, estimates: dict[str, Any]) -> FitResult:
    """``res`` with every free parameter set to mirt's estimate (free flags unchanged)."""
    t = res.parameters
    eta = np.array(res._eta, dtype=float)
    for _, row in t[t["free"]].iterrows():
        est = estimates[row["group"]]
        if row["item"] == "__latent__":
            value = est[row["param"]]
        else:
            value = est["items"][row["item"]][{"a": "a1"}.get(row["param"], row["param"])]
        eta[int(row["free_id"])] = value
    return dataclasses.replace(res, _eta=eta)


def test_science_fit_matches_mirt() -> None:
    ref = _bundle()["science_grm_2g"]
    res, _, _ = _case("science_grm_2g")
    assert res.loglik == pytest.approx(ref["loglik"], rel=1e-9)
    assert res.loglik > ref["loglik"]
    latent = res.latent.set_index("group")
    assert latent.loc["focal", "mean"] == pytest.approx(ref["estimates"]["focal"]["mean"], abs=1e-5)
    assert latent.loc["focal", "var"] == pytest.approx(ref["estimates"]["focal"]["var"], abs=1e-5)


@pytest.mark.parametrize("name", CASES)
def test_s_x2_per_group_matches_mirt(name: str) -> None:
    ref = _bundle()[name]
    res, df, grp = _case(name)
    if name == "science_grm_2g":
        res = _at_mirt(res, ref["estimates"])
    out = res.item_fit(df, group=grp, min_cell=ref["mincell"])
    assert list(out.columns) == ["group", "item", "s_x2", "df", "rmsea", "p"]
    assert list(dict.fromkeys(out["group"])) == ["ref", "focal"]
    for g, expected in ref["groups"].items():
        got = out[out["group"] == g]
        assert list(got["item"]) == expected["item"]
        np.testing.assert_array_equal(got["df"].to_numpy(), np.asarray(expected["df"]))
        np.testing.assert_allclose(got["s_x2"], expected["S_X2"], rtol=TOL_SX2_REL)
        np.testing.assert_allclose(got["rmsea"], expected["RMSEA"], rtol=TOL_SX2_REL, atol=1e-12)
        np.testing.assert_allclose(got["p"], expected["p"], rtol=1e-4, atol=1e-12)


def test_requires_groups_and_all_categories_per_group() -> None:
    res, df, grp = _case("science_grm_2g")
    with pytest.raises(SpecificationError, match="group"):
        res.item_fit(df)
    first = df.columns[0]
    sparse = df.assign(**{first: np.where(grp == "focal", np.minimum(df[first], 2), df[first])})
    with pytest.raises(SpecificationError, match="in group 'focal'"):
        res.item_fit(sparse, group=grp)
