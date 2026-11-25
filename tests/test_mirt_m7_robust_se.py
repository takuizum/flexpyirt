"""V35: cross-product and sandwich covariances vs R mirt ``SE.type = "crossprod"`` / ``"sandwich"``.

Reference values come from ``validation/r/m7_robust_se_reference.R`` (mirt's parameter table and
full ``vcov`` matrices). Two kinds of checks (VALIDATION.md V35, DESIGN.md §17.3):

- **At mirt's solution**: flexpyirt's problem is compiled with mirt's estimates, so the covariance
  matrices are compared entry by entry without any difference in the estimates.
- **End to end**: ``fit(se=...)`` from flexpyirt's own starting values, standard errors against
  mirt's.

mirt reports the lower asymptote ``g`` on the logit scale; mirt's matrices are transformed to the
natural scale with the Jacobian ``g (1 - g)`` before comparing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from flexpyirt import (
    BAEM,
    GPCM,
    GRM,
    EmpiricalHistogram,
    FitResult,
    Model,
    Nominal,
    Normal,
    NormalPrior,
    Quadrature,
    ThreePL,
    TwoPL,
    _core,
    _robust_se,
    fit,
)
from flexpyirt._compile import LATENT_ITEM, compile_problem, prepare
from flexpyirt.items import ItemModel
from tests._reference import check_reference_meta

# mirt reference comparison: nightly and release-PR CI only (CONTRIBUTING.md, "Tests").
pytestmark = pytest.mark.slow

REFERENCE_DIR = Path(__file__).parent / "reference"
CASES = [
    "2pl_misfit",
    "mixed",
    "2pl_map",
    "twogroup",
    "fipc_equal",
    "two_factor",
    "three_factor",
]
EH_CASE = "fipc_eh"

TOL_LOGLIK_REL = 1e-9
TOL_CROSS_PRODUCT = 1e-9
"""Entries of the cross-product covariance at mirt's solution, relative to sqrt(v_ii v_jj)."""
TOL_SANDWICH = 1e-6
"""The same for the sandwich, whose bread in mirt is a finite-difference Oakes Hessian."""
TOL_PARAM = 1e-6
TOL_SE_REL = 1e-5
LOOSE = {"three_factor": 100.0}
"""Tolerance factor for the sandwich and the end-to-end checks: mirt fits the three-factor case
with BFGS (its score at the estimates is 6e-6, against 1e-7 with NR), so its Oakes Hessian and
estimates are less exact (observed 1.4e-5, 5.0e-7 and 6.6e-6 against the three tolerances)."""
TOL_EH = 1e-4
"""EH: comparisons that involve mirt's Richardson (finite-difference) covariance."""

_information_oakes = jax.jit(_core.information_oakes)
_information_cross_product = jax.jit(_robust_se.information_cross_product)


@dataclass(frozen=True)
class Case:
    name: str
    raw: dict[str, Any]
    spec: Model
    quadrature: Quadrature

    @property
    def data(self) -> pd.DataFrame:
        patterns = pd.DataFrame(
            np.asarray(self.raw["patterns"], dtype=float), columns=self.raw["items"]
        )
        return patterns.loc[patterns.index.repeat(self.raw["counts"])].reset_index(drop=True)

    @property
    def group(self) -> np.ndarray | None:
        if not self.spec.is_multigroup:
            return None
        return np.repeat(np.asarray(self.raw["pattern_groups"]), self.raw["counts"])


def _spec(name: str, raw: dict[str, Any]) -> Model:
    items = raw["items"]
    if name == "mixed":
        models: dict[str, ItemModel] = {
            "3PL": ThreePL(),
            "graded": GRM(4),
            "gpcm": GPCM(3),
            "nominal": Nominal(3),
        }
        return Model.from_items(
            {item: models[kind] for item, kind in zip(items, raw["itemtypes"], strict=True)}
        )
    spec = Model.uniform(TwoPL(), items=items)
    if name == "2pl_map":
        a_mean, a_sd = raw["prior"]["a"]
        d_mean, d_sd = raw["prior"]["d"]
        return spec.prior(param="a", dist=NormalPrior(a_mean, a_sd)).prior(
            param="d", dist=NormalPrior(d_mean, d_sd)
        )
    if name == "twogroup":
        return spec.with_groups(ref=Normal.standard(), focal=Normal.free()).equal_across_groups(
            items=raw["anchors"]
        )
    if name == "fipc_equal":
        return (
            spec.fix_from(pd.DataFrame(raw["bank"]))
            .with_latent(Normal.free())
            .equal_across_items(items=raw["equal_items"], params=["a"])
        )
    if name in ("two_factor", "three_factor"):
        return spec.with_factors(raw["factors"])
    if name == EH_CASE:
        return spec.fix_from(pd.DataFrame(raw["bank"])).with_latent(EmpiricalHistogram())
    return spec


@cache
def load(name: str) -> Case:
    raw = dict(json.loads((REFERENCE_DIR / f"v35_{name}.json").read_text(encoding="utf-8")))
    check_reference_meta(raw["meta"], require_grid_check=True)
    lower, upper = raw["meta"]["theta_lim"]
    quadrature = Quadrature.equispaced(int(raw["meta"]["quadpts"]), bounds=(lower, upper))
    return Case(name, raw, _spec(name, raw), quadrature)


def _param_name(mirt_name: str, n_factors: int) -> str:
    """Translate a mirt parameter name into flexpyirt's (``COV_ij`` is a correlation here)."""
    if n_factors == 1:
        return {"a1": "a", "MEAN_1": "mean", "COV_11": "var"}.get(mirt_name, mirt_name)
    if mirt_name.startswith("MEAN_"):
        return f"mean_{mirt_name[5:]}"
    if mirt_name.startswith("COV_"):
        i, j = int(mirt_name[4]), int(mirt_name[5])
        return f"var_{i}" if i == j else f"cor_{j}{i}"
    return mirt_name


@dataclass(frozen=True)
class AtMirt:
    """flexpyirt's problem at mirt's estimates, with mirt's ``parnum`` → ``free_id`` map."""

    problem: _core.CompiledProblem
    eta: jax.Array
    table: pd.DataFrame
    free_id_of: dict[int, int]
    logit_scale: np.ndarray
    """Per ``free_id``: ``g (1 - g)`` for a lower asymptote (mirt works with logit g), else 1."""

    def mirt_matrix(self, block: dict[str, Any]) -> np.ndarray:
        """Return mirt's covariance in ``free_id`` order on flexpyirt's natural scale."""
        ids = np.array([self.free_id_of[int(p)] for p in block["parnum"]])
        n_free = int(self.eta.shape[0])
        assert sorted(ids.tolist()) == list(range(n_free)), "mirt and flexpyirt free sets differ"
        out = np.empty((n_free, n_free))
        out[np.ix_(ids, ids)] = np.asarray(block["matrix"], dtype=float)
        scaled: np.ndarray = self.logit_scale[:, None] * out * self.logit_scale[None, :]
        return scaled


@cache
def at_mirt(name: str) -> AtMirt:
    case = load(name)
    prep = prepare(case.spec, case.data, group=case.group, item_names=None, table=None)
    table = prep.table.copy()
    free_id_of: dict[int, int] = {}
    n_free = int(table["free_id"].max()) + 1
    logit_scale = np.ones(n_free)
    for row in case.raw["values"]:
        item = LATENT_ITEM if row["item"] == "GROUP" else row["item"]
        param = _param_name(row["name"], case.spec.n_factors)
        sel = (table["group"] == row["group"]) & (table["item"] == item) & (table["param"] == param)
        if not sel.any():
            assert not row["est"], f"mirt estimates {row} but flexpyirt has no such parameter"
            continue
        assert bool(table.loc[sel, "free"].iloc[0]) == bool(row["est"]), row
        table.loc[sel, "value"] = float(row["value"])
        if row["est"]:
            free_id = int(table.loc[sel, "free_id"].iloc[0])
            free_id_of[int(row["parnum"])] = free_id
            if param == "g":
                logit_scale[free_id] = float(row["value"]) * (1.0 - float(row["value"]))
    weights = None
    if case.spec.is_empirical_histogram:
        np.testing.assert_allclose(case.raw["nodes"], case.quadrature.nodes(), atol=1e-12)
        weights = np.asarray(case.raw["eh_weights"], dtype=float)[None, :]
    comp = compile_problem(
        case.spec, prep.y, prep.group, table, case.quadrature, eh_weights=weights
    )
    return AtMirt(comp.problem, jnp.asarray(comp.eta0), table, free_id_of, logit_scale)


def scaled_difference(got: np.ndarray, expected: np.ndarray) -> float:
    """Largest ``|got_ij - expected_ij| / sqrt(expected_ii expected_jj)``."""
    scale = np.sqrt(np.diag(expected))
    return float(np.max(np.abs(got - expected) / np.outer(scale, scale)))


def covariances_at_mirt(name: str) -> dict[str, np.ndarray]:
    """flexpyirt's Oakes, cross-product and sandwich covariances at mirt's estimates."""
    m = at_mirt(name)
    bread = np.linalg.inv(np.asarray(_information_oakes(m.problem, m.eta)))
    meat = np.asarray(_information_cross_product(m.problem, m.eta))
    return {"oakes": bread, "crossprod": np.linalg.inv(meat), "sandwich": bread @ meat @ bread}


@pytest.mark.parametrize("name", [*CASES, EH_CASE])
def test_loglik_at_mirt_solution(name: str) -> None:
    """The parameter mapping is right: our log-likelihood at mirt's estimates is mirt's."""
    m = at_mirt(name)
    ref = float(load(name).raw["loglik"])
    assert abs(float(_core.marginal_loglik(m.problem, m.eta)) - ref) <= TOL_LOGLIK_REL * abs(ref)


@pytest.mark.parametrize("name", CASES)
def test_cross_product_covariance_at_mirt_solution(name: str) -> None:
    m, case = at_mirt(name), load(name)
    got = covariances_at_mirt(name)["crossprod"]
    ref = m.mirt_matrix(case.raw["vcov"]["crossprod"])
    assert scaled_difference(got, ref) <= TOL_CROSS_PRODUCT


@pytest.mark.parametrize("name", CASES)
def test_sandwich_covariance_at_mirt_solution(name: str) -> None:
    m, case = at_mirt(name), load(name)
    got = covariances_at_mirt(name)["sandwich"]
    ref = m.mirt_matrix(case.raw["vcov"]["sandwich"])
    assert scaled_difference(got, ref) <= TOL_SANDWICH * LOOSE.get(name, 1.0)


def test_empirical_histogram_at_mirt_solution() -> None:
    """EH with the histogram held at mirt's weights (DESIGN.md §15.35).

    mirt's cross-product holds the histogram fixed as well and is compared directly. mirt's
    ``"sandwich"`` uses its Oakes Hessian, which for EH is not the Hessian with the histogram
    fixed (VALIDATION.md V6); the reference for the sandwich is therefore assembled from mirt's
    Richardson covariance (the fixed-histogram Hessian) and its cross-product.
    """
    m, case = at_mirt(EH_CASE), load(EH_CASE)
    got = covariances_at_mirt(EH_CASE)
    cross = m.mirt_matrix(case.raw["vcov"]["crossprod"])
    richardson = m.mirt_matrix(case.raw["vcov"]["Richardson"])
    assert scaled_difference(got["crossprod"], cross) <= TOL_CROSS_PRODUCT
    assert scaled_difference(got["oakes"], richardson) <= TOL_EH
    assembled = richardson @ np.linalg.inv(cross) @ richardson
    assert scaled_difference(got["sandwich"], assembled) <= TOL_EH


@cache
def _fit(name: str, se: str) -> FitResult:
    case = load(name)
    return fit(
        case.spec,
        case.data,
        group=case.group,
        method=BAEM(tol=1e-9, max_iter=50000),
        quadrature=case.quadrature,
        se=se,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("se", ["crossprod", "sandwich"])
@pytest.mark.parametrize("name", CASES)
def test_fit_standard_errors(name: str, se: str) -> None:
    """End to end: ``fit(se=...)`` reproduces mirt's estimates and standard errors."""
    m, case = at_mirt(name), load(name)
    res = _fit(name, se)
    assert res.converged and res.se_method == se and res.vcov is not None
    ref_loglik = float(case.raw["loglik"])
    assert res.loglik >= ref_loglik - 1e-6 * abs(ref_loglik)
    estimates = np.asarray(res._eta)
    loose = LOOSE.get(name, 1.0)
    np.testing.assert_allclose(estimates, np.asarray(m.eta), rtol=0, atol=TOL_PARAM * loose)
    # mirt's matrix is on the logit scale of g at mirt's g; rescale with the fitted g instead.
    ref = m.mirt_matrix(case.raw["vcov"][se])
    np.testing.assert_allclose(
        np.sqrt(np.diag(res.vcov)), np.sqrt(np.diag(ref)), rtol=TOL_SE_REL * loose, atol=0
    )
    table = res.parameters
    free = table["free"].to_numpy()
    np.testing.assert_allclose(
        table.loc[free, "se"], np.sqrt(np.diag(res.vcov))[table.loc[free, "free_id"]]
    )
