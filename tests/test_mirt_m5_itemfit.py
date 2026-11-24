"""V15: Orlando–Thissen S-X2 item fit vs R mirt's ``itemfit()`` (VALIDATION.md, DESIGN.md §15.40).

Reference values come from ``validation/r/m5_itemfit_reference.R``: mirt evaluates S-X2 at its own
estimates from existing references (V3, V5, V8, V11); flexpyirt refits the same data, so the
comparison also absorbs the ~1e-8 agreement of the estimates.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from flexpyirt import (
    BAEM,
    GPCM,
    GRM,
    PCM,
    FitResult,
    Model,
    Normal,
    Quadrature,
    ThreePL,
    TwoPL,
    _itemfit,
    fit,
)
from flexpyirt.errors import SpecificationError
from tests._reference import check_reference_meta

# mirt reference comparison: nightly and release-PR CI only (CONTRIBUTING.md, "Tests").
pytestmark = pytest.mark.slow

REFERENCE_DIR = Path(__file__).parent / "reference"
TOL_SX2_REL = 1e-6
CASES = [
    "grm_science",
    "grm_science_mincell3",
    "gpcm_science",
    "pcm_sim",
    "threepl_sim",
    "fipc_2pl",
]


@cache
def _bundle() -> dict[str, Any]:
    raw = json.loads((REFERENCE_DIR / "v15_sx2.json").read_text())
    check_reference_meta(raw["meta"])
    return dict(raw["cases"])


def _data(source: str) -> tuple[dict[str, Any], pd.DataFrame]:
    raw = json.loads((REFERENCE_DIR / f"{source}.json").read_text())
    pats = pd.DataFrame(np.asarray(raw["patterns"], dtype=float), columns=raw["items"])
    return raw, pats.loc[pats.index.repeat(raw["counts"])].reset_index(drop=True)


@cache
def _fit(source: str) -> FitResult:
    raw, df = _data(source)
    items = raw["items"]
    quad = Quadrature.equispaced(raw["meta"]["quadpts"], bounds=tuple(raw["meta"]["theta_lim"]))
    spec = {
        "v3_science_grm": lambda: Model.uniform(GRM(n_categories=4), items=items),
        "v8_science_gpcm": lambda: Model.uniform(GPCM(n_categories=4), items=items),
        "v8_sim_pcm": lambda: Model.uniform(PCM(n_categories=4), items=items),
        "v11_3pl": lambda: Model.uniform(ThreePL(), items=items),
        "v5_fipc_2pl": lambda: (
            Model.uniform(TwoPL(), items=items)
            .fix_from(pd.DataFrame(raw["bank"]))
            .with_latent(Normal.free())
        ),
    }[source]()
    return fit(spec, df, method=BAEM(tol=1e-9, max_iter=20000), quadrature=quad, se=None)


@pytest.mark.parametrize("name", CASES)
def test_s_x2_matches_mirt(name: str) -> None:
    ref = _bundle()[name]
    _, df = _data(ref["source"])
    res = _fit(ref["source"]).item_fit(df, min_cell=ref["mincell"])
    assert list(res["item"]) == ref["item"]
    np.testing.assert_array_equal(res["df"].to_numpy(), np.asarray(ref["df"]))
    np.testing.assert_allclose(res["s_x2"], ref["S_X2"], rtol=TOL_SX2_REL)
    np.testing.assert_allclose(res["rmsea"], ref["RMSEA"], rtol=TOL_SX2_REL, atol=1e-12)
    np.testing.assert_allclose(res["p"], ref["p"], rtol=1e-4, atol=1e-12)


def test_lord_wingersky_matches_enumeration() -> None:
    """Summed-score likelihoods equal brute-force enumeration of response patterns."""
    rng = np.random.default_rng(0)
    n_categories = [2, 3, 4]
    traces = [rng.dirichlet(np.ones(k), size=5) for k in n_categories]
    lik = _itemfit.lord_wingersky(traces)
    brute = np.zeros_like(lik)
    for y in np.ndindex(*n_categories):
        brute[sum(y)] += np.prod([traces[j][:, y[j]] for j in range(3)], axis=0)
    np.testing.assert_allclose(lik, brute, rtol=1e-14)
    np.testing.assert_allclose(lik.sum(axis=0), 1.0, rtol=1e-14)


def test_expected_tables_are_conditional_probabilities() -> None:
    """Each possible row of E sums to 1 (categories given the summed score)."""
    rng = np.random.default_rng(1)
    traces = [rng.dirichlet(np.ones(k), size=9) for k in (2, 3, 3, 2)]
    w = rng.dirichlet(np.ones(9))
    for e in _itemfit.s_x2_expected_tables(traces, w):
        np.testing.assert_allclose(np.nansum(e, axis=1), 1.0, rtol=1e-13)


def test_item_fit_rejects_unsupported_inputs() -> None:
    _, df = _data("v3_science_grm")
    res = _fit("v3_science_grm")
    with pytest.raises(SpecificationError, match="complete data"):
        res.item_fit(df.assign(**{df.columns[0]: df.iloc[:, 0].where(df.index > 0)}))
    with pytest.raises(SpecificationError, match="every response category"):
        res.item_fit(df.assign(**{df.columns[0]: np.minimum(df.iloc[:, 0], 2)}))
    with pytest.raises(SpecificationError, match="min_cell"):
        res.item_fit(df, min_cell=0)
