"""V6: empirical histogram latent distributions vs R mirt ``dentype = "EH"`` (VALIDATION.md).

mirt's EH does not standardize the histogram, so the comparable fits are FIPC fits, where the
fixed items set the scale and flexpyirt does not standardize either (DESIGN.md §15.34).

On a 61-point grid the EH likelihood is nearly flat in many histogram directions and its maximum
is spiky, so EM creeps for tens of thousands of cycles and mirt (whose stopping rule looks only at
item parameters) stops somewhere on that path. Final weights are therefore not comparable node by
node. V6 instead checks, at mirt's solution: the log-likelihood, that one flexpyirt EM step
reproduces mirt's state, and the standard errors; and that a full flexpyirt fit reaches at least
mirt's log-likelihood. Reference values come from ``validation/r/m3_eh_reference.R``.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from flexpyirt import BAEM, EmpiricalHistogram, FitResult, Model, Quadrature, TwoPL, _core, fit
from flexpyirt._compile import compile_problem, prepare
from flexpyirt.errors import ConvergenceWarning
from tests._reference import check_reference_meta

REFERENCE_DIR = Path(__file__).parent / "reference"
CASES = ["v6_fipc_eh", "v6_twogroup_fipc_eh"]
TOL_SE_REL = 1e-3
QUAD = Quadrature.equispaced(61, lim=(-6, 6))


@dataclass(frozen=True)
class Case:
    name: str
    raw: dict[str, Any]
    spec: Model

    @property
    def groups(self) -> list[str]:
        return list(self.raw["groups"])

    @property
    def data(self) -> pd.DataFrame:
        pats = pd.DataFrame(
            np.asarray(self.raw["patterns"], dtype=float), columns=self.raw["items"]
        )
        return pats.loc[pats.index.repeat(self.raw["counts"])].reset_index(drop=True)

    @property
    def group(self) -> np.ndarray | None:
        if not self.spec.is_multigroup:
            return None
        return np.repeat(np.asarray(self.raw["pattern_groups"]), self.raw["counts"])

    def params(self, group: str, key: str = "estimates") -> pd.DataFrame:
        return pd.DataFrame(self.raw[key][group]).T.astype(float)

    def weights(self, group: str) -> np.ndarray:
        return np.asarray(self.raw["eh_weights"][group], dtype=float)


@cache
def load(name: str) -> Case:
    raw = dict(json.loads((REFERENCE_DIR / f"{name}.json").read_text()))
    check_reference_meta(raw["meta"], require_grid_check=True)
    np.testing.assert_allclose(raw["nodes"], QUAD.nodes(), atol=1e-12)
    spec = Model.uniform(TwoPL(), items=raw["items"]).fix_from(pd.DataFrame(raw["bank"]))
    if name == "v6_fipc_eh":
        spec = spec.with_latent(EmpiricalHistogram())
    else:
        spec = spec.with_groups(ref=EmpiricalHistogram(), focal=EmpiricalHistogram())
        spec = spec.equal_across_groups(items=raw["free_items"])
    return Case(name, raw, spec)


@cache
def _fit(name: str, se: str) -> FitResult:
    """Full fit with SQUAREM (plain EM needs ~70,000 cycles here, as mirt does)."""
    case = load(name)
    return fit(
        case.spec,
        case.data,
        group=case.group,
        method=BAEM(tol=1e-6, max_iter=20000, squarem=True),
        quadrature=QUAD,
        se=se,  # type: ignore[arg-type]
    )


@cache
def _plain_em(name: str) -> FitResult:
    """A bounded run of plain EM, for monotonicity only."""
    case = load(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return fit(
            case.spec,
            case.data,
            group=case.group,
            method=BAEM(tol=1e-9, max_iter=2000, squarem=False),
            quadrature=QUAD,
            se=None,
        )


def _items(res: FitResult, group: str) -> pd.DataFrame:
    items = res.items
    if "group" in items:
        items = items[items["group"] == group]
    return items.set_index("item")


@dataclass(frozen=True)
class AtMirt:
    """flexpyirt's problem evaluated at mirt's item estimates and histogram weights."""

    problem: _core.CompiledProblem
    eta: jnp.ndarray
    table: pd.DataFrame
    weights: np.ndarray


@cache
def at_mirt(name: str) -> AtMirt:
    case = load(name)
    prep = prepare(case.spec, case.data, group=case.group, item_names=None, table=None)
    table = prep.table.copy()
    for g in case.groups:
        for item, row in case.params(g).iterrows():
            for p in ("a", "d"):
                sel = (table["group"] == g) & (table["item"] == item) & (table["param"] == p)
                table.loc[sel, "value"] = row[p]
    weights = np.stack([case.weights(g) for g in case.groups])
    comp = compile_problem(case.spec, prep.y, prep.group, table, QUAD, eh_weights=weights)
    return AtMirt(comp.problem, jnp.asarray(comp.eta0), table, weights)


def test_loglik_at_mirt_solution() -> None:
    """E-step / likelihood with an EH prior: our log-likelihood at mirt's solution is mirt's."""
    for name in CASES:
        ref = float(load(name).raw["loglik"])
        m = at_mirt(name)
        ll = float(_core.marginal_loglik(m.problem, m.eta))
        assert abs(ll - ref) <= 1e-12 * abs(ref)


@pytest.mark.parametrize("name", CASES)
def test_em_map_fixed_point_at_mirt(name: str) -> None:
    """One flexpyirt EM step from mirt's converged state reproduces mirt's state.

    mirt stops when item parameters change by < 1e-9 and reports the histogram from its last
    E-step, one cycle behind its final items; its stopping rule ignores the histogram, which still
    drifts by a few 1e-7 per cycle. Our E-step + KKT histogram update must therefore return mirt's
    weights to 1e-6, and our item M-step must leave the items in place.
    """
    m = at_mirt(name)
    counts, _ = _core.e_step(m.problem, m.eta)
    eta_new, w_new = _core._m_step_eh(m.problem, m.eta, counts, n_steps=20)
    np.testing.assert_allclose(np.asarray(w_new), m.weights, rtol=0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(eta_new), np.asarray(m.eta), rtol=0, atol=1e-6)


@pytest.mark.parametrize("se", ["oakes", "hessian"])
def test_standard_errors_at_mirt_solution(se: str) -> None:
    """Item SEs with the histogram held fixed (DESIGN.md §15.35) vs mirt's Richardson SEs.

    mirt's Richardson SEs use the same definition (``BL.LL`` with the EH prior fixed). mirt's own
    Oakes SEs for EH differ from its Richardson SEs by up to 1.3% (VALIDATION.md, V6), so they are
    not used.
    """
    case, m = load("v6_fipc_eh"), at_mirt("v6_fipc_eh")
    info_fn = _core.information_oakes if se == "oakes" else _core.information_hessian
    se_by_id = np.sqrt(np.diag(np.linalg.inv(np.asarray(info_fn(m.problem, m.eta)))))
    t = m.table
    for item, row in case.params("all", "se_richardson").iterrows():
        for p in ("a", "d"):
            fid = int(t.loc[(t["item"] == item) & (t["param"] == p), "free_id"].iloc[0])
            assert se_by_id[fid] == pytest.approx(row[f"{p}_se"], rel=TOL_SE_REL)


def test_two_group_oakes_equals_hessian_at_mirt_solution() -> None:
    """mirt 1.46.1 fails to compute Richardson SEs for two-group EH; check Oakes == Hessian."""
    m = at_mirt("v6_twogroup_fipc_eh")
    oakes = np.asarray(_core.information_oakes(m.problem, m.eta))
    hessian = np.asarray(_core.information_hessian(m.problem, m.eta))
    np.testing.assert_allclose(oakes, hessian, rtol=1e-8, atol=1e-8 * np.abs(hessian).max())


@pytest.mark.parametrize("name", CASES)
def test_fit_reaches_mirt_loglik(name: str) -> None:
    """A full flexpyirt fit ends at least as high as mirt's (both creep along a flat ridge)."""
    case, res = load(name), _fit(name, "oakes")
    assert res.converged
    ref = float(case.raw["loglik"])
    assert res.loglik >= ref - 1e-7 * abs(ref)
    assert res.loglik - ref <= 1e-5 * abs(ref)
    for g in case.groups:
        ref_items = case.params(g)
        ours = _items(res, g).loc[ref_items.index]
        np.testing.assert_allclose(ours["a"], ref_items["a"], rtol=0, atol=5e-4)
        np.testing.assert_allclose(ours["d"], ref_items["d"], rtol=0, atol=5e-4)


@pytest.mark.parametrize("name", CASES)
def test_em_monotone(name: str) -> None:
    for res in (_plain_em(name), _fit(name, "oakes")):
        ll = res.history["loglik"].to_numpy()
        assert np.all(np.diff(ll) >= -1e-10 * np.abs(ll[1:]))
    assert _fit(name, "oakes").loglik >= _plain_em(name).loglik


def test_focal_histogram_moments() -> None:
    """The focal histogram (generated as 0.5 - 1.2 * standardized chi^2_4) is not standardized."""
    res = _fit("v6_twogroup_fipc_eh", "oakes")
    focal = res.latent[res.latent["group"] == "focal"]
    assert float(focal["mean"].iloc[0]) == pytest.approx(0.5, abs=0.1)
    assert float(focal["var"].iloc[0]) == pytest.approx(1.44, abs=0.2)
    assert float(focal["skewness"].iloc[0]) < -0.5


def test_bank_items_stay_fixed() -> None:
    case, res = load("v6_twogroup_fipc_eh"), _fit("v6_twogroup_fipc_eh", "oakes")
    bank = pd.DataFrame(case.raw["bank"]).set_index("item")
    for g in case.groups:
        ours = _items(res, g).loc[bank.index]
        np.testing.assert_array_equal(ours["a"], bank["a"])
        np.testing.assert_array_equal(ours["d"], bank["d"])
