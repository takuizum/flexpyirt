"""Numerical agreement of likelihood-ratio DIF tests with R mirt's ``DIF()`` (VALIDATION.md V17).

Reference values come from validation/r/m5_dif_reference.R on the V4 data (2PL, groups
ref/focal, 81 points on [-8, 8]): scheme ``"add"`` from the anchored model (q01–q05 equal, focal
mean and variance free) for ``a`` and ``d`` together and for ``d`` alone, and scheme ``"drop"``
from the fully invariant model.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from flexpyirt import BAEM, FitResult, Model, Normal, Quadrature, TwoPL, fit
from tests._reference import check_reference_meta

REFERENCE_DIR = Path(__file__).parent / "reference"
WIDE = Quadrature.equispaced(81, bounds=(-8, 8))
TOL_X2 = 1e-5
TOL_CRIT = 1e-5
TOL_P = 1e-6


@cache
def _raw() -> dict[str, Any]:
    raw = dict(json.loads((REFERENCE_DIR / "v17_dif.json").read_text()))
    check_reference_meta(raw["meta"], require_grid_check=True)
    return raw


@cache
def _data() -> tuple[pd.DataFrame, np.ndarray]:
    src = json.loads((REFERENCE_DIR / "v4_twogroup_2pl.json").read_text())
    pats = pd.DataFrame(np.asarray(src["patterns"], dtype=float), columns=src["items"])
    counts = np.asarray(src["counts"], dtype=int)
    data = pats.loc[pats.index.repeat(counts)].reset_index(drop=True)
    group = np.repeat(np.asarray(src["pattern_groups"]), counts)
    return data, group


@cache
def _base(scheme: str, squarem: bool = False) -> FitResult:
    data, group = _data()
    items = list(data.columns)
    anchors = _raw()["anchors"] if scheme == "add" else items
    spec = (
        Model.uniform(TwoPL(), items=items)
        .with_groups(ref=Normal.standard(), focal=Normal.free())
        .equal_across_groups(items=anchors)
    )
    return fit(
        spec,
        data,
        group=group,
        method=BAEM(tol=1e-9, max_iter=20000, squarem=squarem),
        quadrature=WIDE,
        se=None,
    )


@cache
def _dif(case: str, squarem: bool = False) -> pd.DataFrame:
    ref = _raw()["cases"][case]
    data, group = _data()
    params = ref["which_par"]
    return _base(ref["scheme"], squarem).dif(
        data, group=group, items=ref["item"], params=list(params), scheme=ref["scheme"]
    )


CASES = ["add_ad", "add_d", "drop_ad"]


@pytest.mark.parametrize("case", CASES)
def test_base_loglik_matches(case: str) -> None:
    ref = _raw()["cases"][case]
    res = _base(ref["scheme"])
    assert res.loglik == pytest.approx(ref["base_loglik"], rel=1e-9)
    assert res.n_params == ref["base_nest"]


@pytest.mark.parametrize("squarem", (False, True))
@pytest.mark.parametrize("case", CASES)
def test_dif_matches_mirt(case: str, squarem: bool) -> None:
    ref = _raw()["cases"][case]
    out = _dif(case, squarem)
    assert list(out["item"]) == list(ref["item"])
    assert out["converged"].all()
    np.testing.assert_array_equal(out["df"], ref["df"])
    np.testing.assert_allclose(out["x2"], ref["x2"], rtol=0, atol=TOL_X2)
    for col in ("aic", "sabic", "hq", "bic"):
        np.testing.assert_allclose(out[col], ref[col], rtol=0, atol=TOL_CRIT)
    np.testing.assert_allclose(out["p"], ref["p"], rtol=1e-4, atol=TOL_P)
    np.testing.assert_allclose(out["adj_p"], ref["adj_p"], rtol=1e-4, atol=TOL_P)
