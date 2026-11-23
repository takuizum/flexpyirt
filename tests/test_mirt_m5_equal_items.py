"""Numerical agreement with R mirt for equality constraints across items (VALIDATION.md V16).

Reference files come from validation/r/m5_equal_items_reference.R (mirt's ``CONSTRAIN``):
V16a one common slope for LSAT7 (2PL), V16b constraints across 2PL and graded items, V16c two
groups with anchors and a common slope within each group (81 points on [-8, 8]).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from flexpyirt import BAEM, GRM, FitResult, Model, Normal, Quadrature, TwoPL, fit
from tests._reference import check_reference_meta

REFERENCE_DIR = Path(__file__).parent / "reference"
SQUAREM = (False, True)
TOL_LOGLIK_REL = 1e-6
TOL_PARAM = 1e-4
TOL_SE_REL = 1e-3
CASES = ["v16_lsat7_common_slope", "v16_mixed_constraints", "v16_twogroup_common_slope"]


@dataclass(frozen=True)
class Case:
    name: str
    raw: dict[str, Any]
    spec: Model
    quadrature: Quadrature

    @property
    def groups(self) -> list[str]:
        return list(self.raw["groups"])

    @property
    def patterns(self) -> pd.DataFrame:
        rows = np.asarray(self.raw["patterns"], dtype=float)
        return pd.DataFrame(rows, columns=self.raw["items"])

    @property
    def data(self) -> pd.DataFrame:
        counts = np.asarray(self.raw["counts"], dtype=int)
        return self.patterns.loc[self.patterns.index.repeat(counts)].reset_index(drop=True)

    @property
    def group(self) -> np.ndarray | None:
        if not self.spec.is_multigroup:
            return None
        counts = np.asarray(self.raw["counts"], dtype=int)
        return np.repeat(np.asarray(self.raw["pattern_groups"]), counts)

    def params(self, group: str, key: str = "estimates") -> dict[str, dict[str, float | None]]:
        return dict(self.raw[key][group]["params"])

    def latent(self, group: str, key: str = "estimates") -> dict[str, float | None]:
        return dict(self.raw[key][group]["latent"])


@cache
def load(name: str) -> Case:
    raw = dict(json.loads((REFERENCE_DIR / f"{name}.json").read_text()))
    check_reference_meta(raw["meta"], require_grid_check=True)
    items = list(raw["items"])
    if name == "v16_mixed_constraints":
        spec = Model.from_items(
            {i: (TwoPL() if k < 6 else GRM(n_categories=3)) for k, i in enumerate(items)}
        )
        quad = Quadrature.equispaced()
    elif name == "v16_twogroup_common_slope":
        spec = (
            Model.uniform(TwoPL(), items=items)
            .with_groups(ref=Normal.standard(), focal=Normal.free())
            .equal_across_groups(items=raw["anchors"])
        )
        quad = Quadrature.equispaced(81, bounds=(-8, 8))
    else:
        spec = Model.uniform(TwoPL(), items=items)
        quad = Quadrature.equispaced()
    for c in raw["constraints"]:
        params = [c["params"]] if isinstance(c["params"], str) else list(c["params"])
        spec = spec.equal_across_items(items=c["items"], params=params)
    return Case(name, raw, spec, quad)


@cache
def _fit(name: str, se: str, squarem: bool = False) -> FitResult:
    case = load(name)
    return fit(
        case.spec,
        case.data,
        group=case.group,
        method=BAEM(tol=1e-9, max_iter=20000, squarem=squarem),
        quadrature=case.quadrature,
        se=se,  # type: ignore[arg-type]
    )


@cache
def _fixed_at_mirt(name: str) -> FitResult:
    """All parameters fixed at mirt's estimates (no EM step moves anything)."""
    case = load(name)
    table = case.spec.parameter_table(case.data, group=case.group)
    values = []
    for g, item, param in zip(table["group"], table["item"], table["param"], strict=True):
        est = case.raw["estimates"][g]
        values.append(est["latent"][param] if item == "__latent__" else est["params"][item][param])
    table["value"] = np.asarray(values, dtype=float)
    table["free"] = False
    return fit(
        case.spec, case.data, group=case.group, table=table, quadrature=case.quadrature, se=None
    )


def _rows(res: FitResult, group: str) -> pd.DataFrame:
    items = res.items
    if "group" in items:
        items = items[items["group"] == group]
    return items.set_index("item")


def _pairs(case: Case, group: str, key: str) -> list[tuple[str, str, float | None]]:
    """``(item, column, mirt value)`` for every parameter or SE column mirt reports."""
    out = []
    for item, vals in case.params(group, key).items():
        for col, v in vals.items():
            out.append((item, col, None if v is None else float(v)))
    return out


def test_n_free_matches_mirt() -> None:
    for name in CASES:
        assert _fit(name, "oakes").n_params == load(name).raw["meta"]["n_free"]


@pytest.mark.parametrize("squarem", SQUAREM)
@pytest.mark.parametrize("name", CASES)
def test_loglik(name: str, squarem: bool) -> None:
    case, res = load(name), _fit(name, "oakes", squarem)
    assert res.converged
    ref = float(case.raw["loglik"])
    assert abs(res.loglik - ref) <= TOL_LOGLIK_REL * abs(ref)
    assert res.loglik >= _fixed_at_mirt(name).loglik - 1e-8


@pytest.mark.parametrize("squarem", SQUAREM)
@pytest.mark.parametrize("name", CASES)
def test_parameters(name: str, squarem: bool) -> None:
    case, res = load(name), _fit(name, "oakes", squarem)
    for g in case.groups:
        ours = _rows(res, g)
        for item, col, ref in _pairs(case, g, "estimates"):
            if col.endswith("_se") or ref is None:
                continue
            assert ours.loc[item, col] == pytest.approx(ref, abs=TOL_PARAM), (g, item, col)
        lat = res.latent.set_index("group").loc[g]
        ref_lat = case.latent(g)
        assert lat["mean"] == pytest.approx(ref_lat["mean"], abs=TOL_PARAM)
        assert lat["var"] == pytest.approx(ref_lat["var"], abs=TOL_PARAM)


@pytest.mark.parametrize("squarem", SQUAREM)
@pytest.mark.parametrize(
    ("name", "se", "key"),
    [
        *[(n, "oakes", "estimates") for n in CASES],
        *[(n, "hessian", "se_richardson") for n in CASES],
    ],
)
def test_standard_errors(name: str, se: str, key: str, squarem: bool) -> None:
    """Oakes vs mirt Oakes, direct Hessian vs mirt Richardson (shared parameters share SEs)."""
    case, res = load(name), _fit(name, se, squarem)
    for g in case.groups:
        ours = _rows(res, g)
        for item, col, ref in _pairs(case, g, key):
            if not col.endswith("_se") or ref is None:
                continue
            assert ours.loc[item, col] == pytest.approx(ref, rel=TOL_SE_REL), (g, item, col)
        lat = res.latent.set_index("group").loc[g]
        for p in ("mean", "var"):
            ref_se = case.latent(g, key)[f"{p}_se"]
            if ref_se is None:
                assert np.isnan(lat[f"{p}_se"])
            else:
                assert lat[f"{p}_se"] == pytest.approx(ref_se, rel=TOL_SE_REL)


@pytest.mark.parametrize("squarem", SQUAREM)
@pytest.mark.parametrize("name", CASES)
def test_em_monotone(name: str, squarem: bool) -> None:
    res = _fit(name, "oakes", squarem)
    ll = res.history["loglik"].to_numpy()
    assert np.all(np.diff(ll) >= -1e-9 * np.abs(ll[1:]))
    assert res.loglik >= ll[-1] - 1e-9 * abs(ll[-1])


@pytest.mark.parametrize("name", CASES)
def test_oakes_equals_hessian(name: str) -> None:
    oakes, hess = _fit(name, "oakes"), _fit(name, "hessian")
    assert oakes.vcov is not None and hess.vcov is not None
    np.testing.assert_allclose(oakes.vcov, hess.vcov, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("name", CASES)
def test_constrained_parameters_are_identical(name: str) -> None:
    case, res = load(name), _fit(name, "oakes")
    for g in case.groups:
        ours = _rows(res, g)
        for c in case.raw["constraints"]:
            params = [c["params"]] if isinstance(c["params"], str) else list(c["params"])
            for p in params:
                vals = ours.loc[c["items"], p].to_numpy()
                assert np.all(vals == vals[0]), (g, c["items"], p)
