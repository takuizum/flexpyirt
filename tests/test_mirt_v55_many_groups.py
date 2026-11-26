"""V55: many groups against R mirt ``multipleGroup()`` (VALIDATION.md V55, DESIGN.md §26, #79).

Reference values come from ``validation/r/v55_many_groups_reference.R``: a 2PL with 10 groups of
unequal size, anchors q01-q06 invariant, q07-q15 free per group, focal means and variances free,
5% missing responses, 81 points on [-8, 8]. With ten groups the fit takes the group-blocked
E-step (``CompiledProblem.blocks``), so this checks it end to end: estimates, log-likelihood,
Oakes SEs and EAP.
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
from flexpyirt._compile import compile_problem, prepare
from tests._reference import check_reference_meta

# mirt reference comparison: nightly and release-PR CI only (CONTRIBUTING.md, "Tests").
pytestmark = pytest.mark.slow

REFERENCE = Path(__file__).parent / "reference" / "v55_many_groups.json"
QUAD = Quadrature.equispaced(81, bounds=(-8, 8))
TOL_LOGLIK_REL = 1e-6
TOL_PARAM = 1e-4
TOL_SE_REL = 1e-3
TOL_EAP = 1e-5


@cache
def _raw() -> dict[str, Any]:
    raw = dict(json.loads(REFERENCE.read_text()))
    check_reference_meta(raw["meta"], require_grid_check=True)
    return raw


def _data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    raw = _raw()
    rows = [[np.nan if v is None else v for v in row] for row in raw["patterns"]]
    patterns = pd.DataFrame(np.asarray(rows, dtype=float), columns=raw["items"])
    counts = np.asarray(raw["counts"], dtype=int)
    data = patterns.loc[patterns.index.repeat(counts)].reset_index(drop=True)
    group = np.repeat(np.asarray(raw["pattern_groups"]), counts)
    return patterns, data, group, np.asarray(raw["pattern_groups"])


def _spec() -> Model:
    raw = _raw()
    groups = list(raw["groups"])
    latents = {groups[0]: Normal.standard()} | {g: Normal.free() for g in groups[1:]}
    return (
        Model.uniform(TwoPL(), items=raw["items"])
        .with_groups(latents)
        .equal_across_groups(items=raw["anchors"])
    )


@cache
def _fit() -> FitResult:
    _, data, group, _ = _data()
    return fit(
        _spec(),
        data,
        group=group,
        method=BAEM(tol=1e-9, max_iter=20000, squarem=True),
        quadrature=QUAD,
        se="oakes",
    )


def test_uses_group_blocks() -> None:
    _, data, group, _ = _data()
    prep = prepare(_spec(), data, group=group, item_names=None, table=None)
    comp = compile_problem(_spec(), prep.y, prep.group, prep.table, QUAD)
    assert comp.problem.blocks is not None
    assert comp.problem.n_groups == 10


def test_loglik() -> None:
    res = _fit()
    assert res.converged
    ref = float(_raw()["loglik"])
    assert abs(res.loglik - ref) <= TOL_LOGLIK_REL * abs(ref)


def test_item_parameters_and_se() -> None:
    res, raw = _fit(), _raw()
    items = res.items.set_index(["group", "item"])
    for g in raw["groups"]:
        for item, ref in raw["estimates"][g]["params"].items():
            row = items.loc[(g, item)]
            for name in ("a", "d"):
                assert abs(row[name] - ref[name]) < TOL_PARAM, (g, item, name)
                assert abs(row[f"{name}_se"] / ref[f"{name}_se"] - 1) < TOL_SE_REL, (g, item, name)


def test_latent_parameters_and_se() -> None:
    res, raw = _fit(), _raw()
    latent = res.latent.set_index("group")
    for g in raw["groups"][1:]:
        ref = raw["estimates"][g]["latent"]
        for name in ("mean", "var"):
            assert abs(latent.loc[g, name] - ref[name]) < TOL_PARAM, (g, name)
            assert abs(latent.loc[g, f"{name}_se"] / ref[f"{name}_se"] - 1) < TOL_SE_REL, (g, name)


def test_eap() -> None:
    res, raw = _fit(), _raw()
    patterns, _, _, pattern_groups = _data()
    scores = res.score(patterns, group=pattern_groups, method="EAP")
    np.testing.assert_allclose(scores["theta"], raw["eap"]["theta"], atol=TOL_EAP)
    np.testing.assert_allclose(scores["se"], raw["eap"]["se"], atol=TOL_EAP)
