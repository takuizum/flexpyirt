"""V57: weighted and cluster-robust sandwich covariances vs R (mirt ``survey.weights`` + numDeriv).

Reference values come from ``validation/r/v57_weighted_cluster_se_reference.R`` (VALIDATION.md V57,
DESIGN.md §17.6). mirt has neither the design-based weighted sandwich nor a cluster-robust one, so
the references are computed in R from mirt's weighted Oakes covariance (the bread) and per-person
scores by ``numDeriv`` (checked there against mirt's ``estfun``). Checks:

- **At mirt's solution** (flexpyirt's problem compiled with mirt's estimates): the middle matrices
  (weighted, cluster, unweighted cluster, singleton clusters) against the R ones, which isolates
  the scores from the finite-difference bread of mirt; ``se="crossprod"`` with weights against
  mirt's ``SE.type = "crossprod"`` with ``survey.weights``; and flexpyirt's parts reproducing mirt's
  own ``SE.type = "sandwich"`` with weights (whose middle matrix is ``sum_i w_i s_i s_i^T``).
- **End to end**: ``fit(..., weights=, cluster=, se="sandwich")`` standard errors against the
  reference ones.
"""

from __future__ import annotations

import importlib
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

from flexpyirt import BAEM, FitResult, Model, Normal, Quadrature, TwoPL, _core, _robust_se, fit
from flexpyirt._compile import LATENT_ITEM, Prepared, compile_problem, prepare
from tests._reference import check_reference_meta

# mirt reference comparison: nightly and release-PR CI only (CONTRIBUTING.md, "Tests").
pytestmark = pytest.mark.slow

_fit_module = importlib.import_module("flexpyirt.fit")
REFERENCE_DIR = Path(__file__).parent / "reference"
CASES = ["school_2pl", "twogroup"]

TOL_MEAT = 1e-7
"""Middle matrices at mirt's solution, relative to sqrt(m_ii m_jj) (numDeriv scores ~1e-9)."""
TOL_CROSS_PRODUCT = 1e-9
"""mirt's crossprod covariance (analytic scores on both sides), relative to sqrt(v_ii v_jj)."""
TOL_SANDWICH = 1e-6
"""Covariances whose bread is mirt's finite-difference Oakes Hessian (as in V35)."""
TOL_PARAM = 1e-6
TOL_SE_REL = 1e-5

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
        return pd.DataFrame(
            np.asarray(self.raw["responses"], dtype=float), columns=self.raw["items"]
        )

    @property
    def group(self) -> np.ndarray | None:
        raw_group = self.raw.get("group")
        return np.asarray(raw_group) if raw_group else None  # R writes NULL as {}

    @property
    def weights(self) -> np.ndarray:
        return np.asarray(self.raw["weights"], dtype=float)

    @property
    def cluster(self) -> np.ndarray:
        return np.asarray(self.raw["cluster"])

    def block(self, key: str) -> dict[str, Any]:
        return dict(self.raw["vcov"][key])


@cache
def load(name: str) -> Case:
    raw = dict(json.loads((REFERENCE_DIR / f"v57_{name}.json").read_text(encoding="utf-8")))
    check_reference_meta(raw["meta"], require_grid_check=True)
    assert raw["meta"]["max_abs_numderiv_minus_estfun"] < 1e-6
    lower, upper = raw["meta"]["theta_lim"]
    quadrature = Quadrature.equispaced(int(raw["meta"]["quadpts"]), bounds=(lower, upper))
    spec = Model.uniform(TwoPL(), items=raw["items"])
    if name == "twogroup":
        spec = spec.with_groups(ref=Normal.standard(), focal=Normal.free()).equal_across_groups(
            items=raw["anchors"]
        )
    return Case(name, raw, spec, quadrature)


_PARAM_NAMES = {"a1": "a", "MEAN_1": "mean", "COV_11": "var"}


@dataclass(frozen=True)
class AtMirt:
    """flexpyirt's problem at mirt's estimates, with mirt's ``parnum`` → ``free_id`` map."""

    case: Case
    prep: Prepared
    problem: _core.CompiledProblem
    eta: jax.Array
    free_id_of: dict[int, int]

    def mirt_matrix(self, key: str) -> np.ndarray:
        """Return a reference matrix in ``free_id`` order."""
        block = self.case.block(key)
        ids = np.array([self.free_id_of[int(p)] for p in block["parnum"]])
        n_free = int(self.eta.shape[0])
        assert sorted(ids.tolist()) == list(range(n_free)), "mirt and flexpyirt free sets differ"
        out = np.empty((n_free, n_free))
        out[np.ix_(ids, ids)] = np.asarray(block["matrix"], dtype=float)
        return out

    def reference_meat(self, key: str) -> np.ndarray:
        """Recover the middle matrix ``M`` of a reference ``V = V_oakes M V_oakes``."""
        bread = np.linalg.inv(self.mirt_matrix("oakes"))
        meat: np.ndarray = bread @ self.mirt_matrix(key) @ bread
        return meat

    def meat(self, *, weights: bool = True, cluster: Any = "case") -> np.ndarray:
        """flexpyirt's middle matrix at mirt's solution (``fit()``'s own helper)."""
        case = self.case
        labels = case.cluster if isinstance(cluster, str) and cluster == "case" else cluster
        prep = prepare(
            case.spec,
            case.data,
            group=case.group,
            item_names=None,
            table=self.prep.table,
            weights=case.weights if weights else None,
            cluster=labels,
        )
        return np.asarray(_fit_module._sandwich_meat(self.problem, self.eta, prep))


@cache
def at_mirt(name: str) -> AtMirt:
    case = load(name)
    prep = prepare(
        case.spec, case.data, group=case.group, item_names=None, table=None, weights=case.weights
    )
    table = prep.table.copy()
    free_id_of: dict[int, int] = {}
    for row in case.raw["values"]:
        item = LATENT_ITEM if row["item"] == "GROUP" else row["item"]
        param = _PARAM_NAMES.get(row["name"], row["name"])
        sel = (table["group"] == row["group"]) & (table["item"] == item) & (table["param"] == param)
        if not sel.any():
            assert not row["est"], f"mirt estimates {row} but flexpyirt has no such parameter"
            continue
        assert bool(table.loc[sel, "free"].iloc[0]) == bool(row["est"]), row
        table.loc[sel, "value"] = float(row["value"])
        if row["est"]:
            free_id_of[int(row["parnum"])] = int(table.loc[sel, "free_id"].iloc[0])
    prep = prepare(
        case.spec,
        case.data,
        group=case.group,
        item_names=None,
        table=table,
        weights=case.weights,
    )
    comp = compile_problem(
        case.spec, prep.y, prep.group, prep.table, case.quadrature, weights=prep.weights
    )
    return AtMirt(case, prep, comp.problem, jnp.asarray(comp.eta0), free_id_of)


def _assert_close(got: np.ndarray, expected: np.ndarray, tol: float) -> None:
    """Entries relative to ``sqrt(e_ii e_jj)`` (the scale of a covariance or information)."""
    scale = np.sqrt(np.outer(np.diag(expected), np.diag(expected)))
    err = np.abs(got - expected) / scale
    assert err.max() < tol, f"max relative error {err.max():.3g} >= {tol:g}"


@pytest.mark.parametrize("name", CASES)
def test_weighted_meat_at_mirts_solution(name: str) -> None:
    """``sum_i w_i^2 s_i s_i^T`` against the numDeriv person scores."""
    point = at_mirt(name)
    _assert_close(point.meat(cluster=None), point.reference_meat("weighted"), TOL_MEAT)


@pytest.mark.parametrize("name", CASES)
def test_cluster_meat_at_mirts_solution(name: str) -> None:
    """``G/(G-1) sum_c u_c u_c^T`` with ``u_c`` the weighted scores summed by school."""
    point = at_mirt(name)
    _assert_close(point.meat(), point.reference_meat("cluster"), TOL_MEAT)
    _assert_close(point.meat(weights=False), point.mirt_matrix("unweighted_cluster_meat"), TOL_MEAT)


@pytest.mark.parametrize("name", CASES)
def test_singleton_clusters_at_mirts_solution(name: str) -> None:
    """One person per cluster is the weighted sandwich times ``N / (N - 1)``."""
    point = at_mirt(name)
    singletons = np.arange(len(point.case.data))
    _assert_close(point.meat(cluster=singletons), point.reference_meat("singleton"), TOL_MEAT)


@pytest.mark.parametrize("name", CASES)
def test_crossprod_with_weights_matches_mirt(name: str) -> None:
    """``se="crossprod"`` with weights is mirt's ``SE.type = "crossprod"`` with survey weights."""
    point = at_mirt(name)
    info = np.asarray(_information_cross_product(point.problem, point.eta))
    _assert_close(np.linalg.inv(info), point.mirt_matrix("mirt_crossprod"), TOL_CROSS_PRODUCT)


@pytest.mark.parametrize("name", CASES)
def test_mirts_weighted_sandwich_uses_the_frequency_weight_meat(name: str) -> None:
    """mirt's ``SE.type = "sandwich"`` with weights is ``H^-1 (sum_i w_i s_i s_i^T) H^-1``.

    This is the reading of mirt on which DESIGN.md §17.6 rests; flexpyirt's Oakes bread and
    weighted cross-product reproduce it, and differ from the design-based sandwich.
    """
    point = at_mirt(name)
    bread = np.linalg.inv(np.asarray(_information_oakes(point.problem, point.eta)))
    _assert_close(bread, point.mirt_matrix("oakes"), TOL_SANDWICH)
    meat = np.asarray(_information_cross_product(point.problem, point.eta))
    _assert_close(bread @ meat @ bread, point.mirt_matrix("mirt_sandwich"), TOL_SANDWICH)
    design = point.mirt_matrix("weighted")
    assert np.max(np.abs(np.diag(design) / np.diag(point.mirt_matrix("mirt_sandwich")) - 1)) > 0.05


@cache
def _fit(name: str, kind: str) -> FitResult:
    case = load(name)
    cluster = case.cluster if kind == "cluster" else None
    return fit(
        case.spec,
        case.data,
        group=case.group,
        weights=case.weights,
        cluster=cluster,
        quadrature=case.quadrature,
        method=BAEM(tol=1e-10, max_iter=20000),
        se="sandwich",
    )


@pytest.mark.parametrize("kind", ["weighted", "cluster"])
@pytest.mark.parametrize("name", CASES)
def test_end_to_end_standard_errors(name: str, kind: str) -> None:
    point = at_mirt(name)
    res = _fit(name, kind)
    assert res.converged and res.se_method == "sandwich"
    assert res._n_clusters == (point.case.raw["meta"]["n_clusters"] if kind == "cluster" else None)
    assert res.loglik == pytest.approx(point.case.raw["loglik"], rel=1e-9)
    np.testing.assert_allclose(res._eta, np.asarray(point.eta), rtol=0, atol=TOL_PARAM)
    assert res.vcov is not None
    expected = np.sqrt(np.diag(point.mirt_matrix(kind)))
    np.testing.assert_allclose(np.sqrt(np.diag(res.vcov)), expected, rtol=TOL_SE_REL)
