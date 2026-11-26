"""V54: ``compare_fits()`` against R mirt's ``anova(mod1, mod2, ...)`` (VALIDATION.md V54).

Reference values come from ``validation/r/v54_compare_simulate_reference.R``: Rasch vs 2PL vs
3PL on the V11a data, PCM vs GPCM on the V8b data, and a fully invariant two-group 2PL against
the anchored model and against the model with the last item free (V4 data, 81 points on
[-8, 8]). Each model is refitted here, so the table differences combine the fit agreement
(VALIDATION.md, log-likelihood) and the formulas.
"""

from __future__ import annotations

import json
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest

from flexpyirt import (
    BAEM,
    GPCM,
    PCM,
    FitResult,
    Model,
    Normal,
    Quadrature,
    Rasch,
    ThreePL,
    TwoPL,
    compare_fits,
    fit,
)
from tests._reference import REFERENCE_DIR, check_reference_meta

pytestmark = pytest.mark.slow

TOL_LOGLIK_ABS = 1e-7
"""Log-likelihood and criteria (absolute; measured 5e-11, VALIDATION.md V54)."""
TOL_X2_ABS = 1e-7
"""Likelihood-ratio statistic (absolute; measured 1.4e-11)."""
TOL_P_ABS = 1e-9
"""p-value (absolute; measured 7e-13; mirt prints 0 below about 1e-16)."""
METHOD = BAEM(tol=1e-9, max_iter=20000)
DEFAULT = Quadrature.equispaced(61, bounds=(-6, 6))
WIDE = Quadrature.equispaced(81, bounds=(-8, 8))


@cache
def _raw() -> dict[str, Any]:
    raw = dict(json.loads((REFERENCE_DIR / "v54_compare_fits.json").read_text()))
    check_reference_meta(raw["meta"])
    return raw


def _data(name: str) -> tuple[pd.DataFrame, np.ndarray]:
    src = json.loads((REFERENCE_DIR / f"{name}.json").read_text())
    pats = pd.DataFrame(np.asarray(src["patterns"], dtype=float), columns=src["items"])
    counts = np.asarray(src["counts"], dtype=int)
    data = pats.loc[pats.index.repeat(counts)].reset_index(drop=True)
    return data, np.repeat(np.asarray(src["pattern_groups"]), counts)


def _single(name: str, item_models: list[Any]) -> list[FitResult]:
    data, _ = _data(name)
    items = list(data.columns)
    return [
        fit(Model.uniform(m, items=items), data, method=METHOD, quadrature=DEFAULT, se=None)
        for m in item_models
    ]


def _twogroup(free_items: list[str]) -> FitResult:
    data, group = _data("v4_twogroup_2pl")
    items = list(data.columns)
    spec = (
        Model.uniform(TwoPL(), items=items)
        .with_groups(ref=Normal.standard(), focal=Normal.free())
        .equal_across_groups(items=[i for i in items if i not in free_items])
    )
    return fit(spec, data, group=group, method=METHOD, quadrature=WIDE, se=None)


@cache
def _fits(case: str) -> tuple[FitResult, ...]:
    if case == "dichotomous":
        return tuple(_single("v11_3pl", [Rasch(), TwoPL(), ThreePL()]))
    if case == "polytomous":
        return tuple(_single("v8_sim_gpcm", [PCM(n_categories=4), GPCM(n_categories=4)]))
    items = list(_data("v4_twogroup_2pl")[0].columns)
    if case == "twogroup":
        anchors = set(_raw()["anchors"])
        return (_twogroup([]), _twogroup([i for i in items if i not in anchors]))
    return (_twogroup([]), _twogroup(items[-1:]))


CASES = ["dichotomous", "polytomous", "twogroup", "twogroup_last_free"]


@pytest.mark.filterwarnings("ignore::flexpyirt.errors.HeywoodWarning")
@pytest.mark.parametrize("case", CASES)
def test_compare_fits_matches_mirt_anova(case: str) -> None:
    ref = _raw()["cases"][case]
    table = compare_fits(*_fits(case), names=ref["model"])
    assert list(table["model"]) == ref["model"]
    np.testing.assert_array_equal(table["n_params"].to_numpy(), ref["n_params"])
    for col in ("loglik", "aic", "sabic", "hq", "bic"):
        np.testing.assert_allclose(table[col].to_numpy(), ref[col], rtol=0, atol=TOL_LOGLIK_ABS)
    # mirt leaves the first row empty.
    assert table.loc[0, ["x2", "df", "p"]].isna().all()
    np.testing.assert_allclose(table["x2"].to_numpy()[1:], ref["x2"][1:], atol=TOL_X2_ABS)
    np.testing.assert_array_equal(table["df"].to_numpy()[1:].astype(int), ref["df"][1:])
    np.testing.assert_allclose(table["p"].to_numpy()[1:], ref["p"][1:], atol=TOL_P_ABS)
