"""V56: DavidianCurve and NormalMixture per group (VALIDATION.md V56, DESIGN.md §27, #75).

Reference values come from ``validation/r/v56_multigroup_shapes_reference.R``: mirt
``multipleGroup(customGroup = createGroup(...))`` with flexpyirt's standardized densities
normalized on the grid, polished by Newton steps on an independent R marginal log-likelihood,
with R Richardson-Hessian SEs, EAPs computed in R and S-X2 from ``itemfit()`` on single-group
mirt objects fixed at each group's estimates.

(a) 2PL, three groups, ``DavidianCurve(2)``, anchors q01-q06, focal means / variances free, 3%
missing; (b) 2PL, two groups, ``NormalMixture(2)``, every item invariant; (c) FIPC, two groups,
``DavidianCurve(2)``, items fixed at bank values, every group's mean and variance free.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from flexpyirt import BAEM, DavidianCurve, FitResult, Model, NormalMixture, Quadrature, TwoPL, fit
from flexpyirt._compile import LATENT_ITEM
from tests._reference import check_reference_meta

# mirt reference comparison: nightly and release-PR CI only (CONTRIBUTING.md, "Tests").
pytestmark = pytest.mark.slow

REFERENCE_DIR = Path(__file__).parent / "reference"
CASES = ["v56a_davidian_three_groups", "v56b_mixture_two_groups", "v56c_fipc_davidian"]
QUAD = Quadrature.equispaced(61, bounds=(-6, 6))
TOL_LOGLIK_REL = 1e-8
TOL_PARAM = 1e-4
TOL_SE_REL = 1e-3
TOL_EAP = 1e-5
TOL_SX2 = 1e-4


@cache
def _raw(name: str) -> dict[str, Any]:
    raw = dict(json.loads((REFERENCE_DIR / f"{name}.json").read_text()))
    check_reference_meta(raw["meta"], require_grid_check=True)
    return raw


def _patterns(raw: dict[str, Any]) -> pd.DataFrame:
    rows = [[np.nan if v is None else v for v in row] for row in raw["patterns"]]
    return pd.DataFrame(np.asarray(rows, dtype=float), columns=raw["items"])


def _data(raw: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray]:
    patterns = _patterns(raw)
    counts = np.asarray(raw["counts"], dtype=int)
    data = patterns.loc[patterns.index.repeat(counts)].reset_index(drop=True)
    return data, np.repeat(np.asarray(raw["pattern_groups"]), counts)


def _reference(raw: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(raw["parameters"]).set_index(["group", "item", "param"])


def _spec(raw: dict[str, Any]) -> Model:
    groups = list(raw["groups"])
    k = int(raw["meta"]["k"])
    family = DavidianCurve if raw["meta"]["kind"] == "davidian" else NormalMixture
    base = Model.uniform(TwoPL(), items=raw["items"])
    if raw.get("bank"):
        bank = pd.DataFrame(raw["bank"])
        latents = {g: family(k, free_mean=True, free_var=True) for g in groups}
        return base.fix_from(bank).with_groups(latents)
    focal = {g: family(k, free_mean=True, free_var=True) for g in groups[1:]}
    latents = {groups[0]: family(k)} | focal
    return base.with_groups(latents).equal_across_groups(items=raw["anchors"])


@cache
def _fit(name: str) -> FitResult:
    raw = _raw(name)
    data, group = _data(raw)
    spec = _spec(raw)
    table = spec.parameter_table(data, group=group)
    # Davidian and mixture likelihoods are multimodal in the shape: start each group's latent
    # parameters near the reference optimum (two significant digits), so both programs are
    # compared at the same local maximum.
    ref = _reference(raw)["value"]
    for i, row in table[table["item"] == LATENT_ITEM].iterrows():
        value = float(ref[(row["group"], LATENT_ITEM, row["param"])])
        table.at[i, "value"] = float(f"{value:.2g}")
    return fit(
        spec,
        data,
        group=group,
        table=table,
        quadrature=QUAD,
        method=BAEM(tol=1e-10, max_iter=50000, squarem=True),
        se="oakes",
    )


@pytest.mark.parametrize("name", CASES)
def test_loglik(name: str) -> None:
    raw, res = _raw(name), _fit(name)
    assert res.converged
    assert abs(res.loglik - raw["loglik"]) <= TOL_LOGLIK_REL * abs(raw["loglik"])
    # mirt's own EM solution is no better than the polished optimum.
    assert res.loglik >= raw["loglik_mirt"] - 1e-6


@pytest.mark.parametrize("name", CASES)
def test_parameters_and_se(name: str) -> None:
    raw, res = _raw(name), _fit(name)
    ref = _reference(raw)
    ours = res.parameters.set_index(["group", "item", "param"])
    for key, row in ref.iterrows():
        mine = ours.loc[[key]].iloc[0]
        assert abs(float(mine["estimate"]) - float(row["value"])) < TOL_PARAM, key
        assert bool(mine["free"]) == bool(row["free"]), key
        if row["free"]:
            assert abs(float(mine["se"]) / float(row["se"]) - 1.0) < TOL_SE_REL, key


@pytest.mark.parametrize("name", CASES)
def test_eap(name: str) -> None:
    raw, res = _raw(name), _fit(name)
    scores = res.score(_patterns(raw), group=raw["pattern_groups"], method="EAP")
    np.testing.assert_allclose(scores["theta"], raw["eap"]["theta"], rtol=0, atol=TOL_EAP)
    np.testing.assert_allclose(scores["se"], raw["eap"]["sd"], rtol=0, atol=TOL_EAP)


@pytest.mark.parametrize("name", [n for n in CASES if n != "v56a_davidian_three_groups"])
def test_s_x2_per_group(name: str) -> None:
    raw, res = _raw(name), _fit(name)
    data, group = _data(raw)
    table = res.item_fit(data, group=group).set_index(["group", "item"])
    params = res.parameters
    for g, ref in raw["s_x2"].items():
        in_group = params["group"] == g
        n_lat = int(params.loc[in_group & (params["item"] == LATENT_ITEM), "free"].sum())
        for item, stat, df_mirt in zip(ref["item"], ref["s_x2"], ref["df_mirt"], strict=True):
            n_item = int(params.loc[in_group & (params["item"] == item), "free"].sum())
            assert abs(table.loc[(g, item), "s_x2"] - stat) < TOL_SX2 * max(1.0, stat), (g, item)
            # mirt's objects are fixed (no free parameters); flexpyirt subtracts them.
            assert table.loc[(g, item), "df"] == df_mirt - n_item - n_lat, (g, item)
