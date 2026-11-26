"""Unit tests for the information of multidimensional models (DESIGN.md §18.3).

Agreement with R mirt is in ``tests/test_mirt_m7_information_multidim.py`` (VALIDATION.md V43).
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pandas as pd
import pytest

from flexpyirt import (
    GRM,
    FitResult,
    Model,
    Nominal,
    Normal,
    ThreePL,
    TwoPL,
    expected_score,
    fit,
    information_matrix,
    item_information,
    test_information,
)
from flexpyirt.errors import SpecificationError
from flexpyirt.information import _composite_se
from flexpyirt.items import ItemModel

THETA2 = np.array([[-2.0, 1.0], [0.0, 0.0], [0.7, -1.4], [2.5, 2.0]])

ITEMS: dict[str, tuple[ItemModel, dict[str, float]]] = {
    "x1": (TwoPL(), {"a1": 1.3, "a2": 0.0, "d": -0.4}),
    "x2": (ThreePL(), {"a1": 1.6, "a2": 0.0, "d": 0.3, "g": 0.2, "u": 1.0}),
    "x3": (GRM(n_categories=4), {"a1": 1.1, "a2": 0.0, "d1": 1.2, "d2": 0.1, "d3": -1.1}),
    "y1": (TwoPL(), {"a1": 0.0, "a2": 0.9, "d": 0.5}),
    "y2": (Nominal(n_categories=3), {"a1": 0.0, "a2": 1.2, "ak1": 0.6, "d1": 0.4, "d2": -0.3}),
    "y3": (TwoPL(), {"a1": 0.0, "a2": 1.4, "d": -0.8}),
}
"""Simple structure: ``x*`` on factor ``f1`` only, ``y*`` on ``f2`` only."""

CROSS = {"x1": {"a2": 0.7}, "y1": {"a1": 0.5}}
"""Cross-loadings added to :data:`ITEMS` for the model with cross-loadings."""


def _dummy(models: list[ItemModel], names: list[str]) -> pd.DataFrame:
    k = [m.kernel.n_categories for m in models]
    rows = [[min(c, kj - 1) for kj in k] for c in range(max(k))] + [[0] * len(k)]
    return pd.DataFrame(np.asarray(rows, dtype=float), columns=names)


def _fixed(spec: Model, params: dict[str, dict[str, float]], dummy: pd.DataFrame) -> FitResult:
    table = spec.parameter_table(dummy)
    table["value"] = [
        float(v) if item == "__latent__" else float(params[item].get(param, v))
        for item, param, v in zip(table["item"], table["param"], table["value"], strict=True)
    ]
    table["free"] = False
    return fit(spec, dummy, table=table, se=None)


def _two_factor(cross: bool) -> FitResult:
    params = {item: dict(p) for item, (_, p) in ITEMS.items()}
    loads = {"f1": ["x1", "x2", "x3"], "f2": ["y1", "y2", "y3"]}
    if cross:
        for item, extra in CROSS.items():
            params[item].update(extra)
        loads = {"f1": ["x1", "x2", "x3", "y1"], "f2": ["x1", "y1", "y2", "y3"]}
    spec = Model.from_items({item: m for item, (m, _) in ITEMS.items()}).with_factors(loads)
    return _fixed(spec, params, _dummy([m for m, _ in ITEMS.values()], list(ITEMS)))


@cache
def _simple() -> FitResult:
    return _two_factor(cross=False)


@cache
def _cross() -> FitResult:
    return _two_factor(cross=True)


def _unidimensional(items: list[str]) -> FitResult:
    """The unidimensional model of ``items`` with their slope on their own factor."""
    params = {}
    for item in items:
        _, p = ITEMS[item]
        slope = p["a1"] if item.startswith("x") else p["a2"]
        params[item] = {"a": slope, **{k: v for k, v in p.items() if not k.startswith("a")}}
        params[item].update({k: v for k, v in p.items() if k.startswith("ak")})
    spec = Model.from_items({item: ITEMS[item][0] for item in items})
    return _fixed(spec, params, _dummy([ITEMS[i][0] for i in items], items))


@pytest.mark.slow
def test_simple_structure_reduces_to_unidimensional_information() -> None:
    res = _simple()
    matrices = information_matrix(res, THETA2)
    assert matrices.shape == (len(THETA2), 2, 2)
    np.testing.assert_array_equal(matrices[:, 0, 1], 0.0)
    for factor, items, direction in (
        (0, ["x1", "x2", "x3"], [1.0, 0.0]),
        (1, ["y1", "y2", "y3"], [0.0, 1.0]),
    ):
        uni = _unidimensional(items)
        theta = THETA2[:, factor]
        expected = item_information(uni, theta).to_numpy()
        got = item_information(res, THETA2, direction=direction)
        np.testing.assert_allclose(got[items].to_numpy(), expected, rtol=1e-12)
        others = [i for i in ITEMS if i not in items]
        np.testing.assert_array_equal(got[others].to_numpy(), 0.0)
        # Block-diagonal information: the composite SE along an axis is the sub-test's SE.
        total = test_information(res, THETA2, direction=direction)
        uni_total = test_information(uni, theta)
        np.testing.assert_allclose(total["information"], uni_total["information"], rtol=1e-12)
        np.testing.assert_allclose(total["se"], uni_total["se"], rtol=1e-12)
        np.testing.assert_allclose(
            expected_score(res, THETA2)[items].to_numpy(),
            expected_score(uni, theta)[items].to_numpy(),
            rtol=1e-12,
        )


def test_matrices_are_symmetric_rank_one_and_match_closed_form() -> None:
    res = _cross()
    per_item = np.stack([information_matrix(res, THETA2, item=i) for i in ITEMS], axis=1)
    np.testing.assert_allclose(per_item, np.swapaxes(per_item, -1, -2), rtol=0, atol=0)
    eig = np.linalg.eigvalsh(per_item)
    assert np.all(eig[..., 0] > -1e-14)
    # Every built-in family depends on theta through a' theta only: rank 1.
    assert np.all(eig[..., 0] <= 1e-12 * eig[..., 1])
    np.testing.assert_allclose(information_matrix(res, THETA2), per_item.sum(axis=1), rtol=1e-13)
    # 2PL: I = P (1 - P) a a'.
    a = np.array([1.3, 0.7])
    p = 1.0 / (1.0 + np.exp(-(THETA2 @ a - 0.4)))
    closed = (p * (1.0 - p))[:, None, None] * np.outer(a, a)
    np.testing.assert_allclose(information_matrix(res, THETA2, item="x1"), closed, rtol=1e-12)
    np.testing.assert_allclose(expected_score(res, THETA2)["x1"], p, rtol=1e-12)


def test_directional_information_and_composite_se() -> None:
    res = _cross()
    direction = np.array([0.6, -0.8])
    matrices = information_matrix(res, THETA2)
    info = test_information(res, THETA2, direction=direction)
    np.testing.assert_allclose(
        info["information"], np.einsum("m,tmn,n->t", direction, matrices, direction), rtol=1e-13
    )
    inverse = np.linalg.inv(matrices)
    np.testing.assert_allclose(
        info["se"], np.sqrt(np.einsum("m,tmn,n->t", direction, inverse, direction)), rtol=1e-12
    )
    # The composite SE is at least 1/sqrt(directional information) (Cauchy-Schwarz).
    assert np.all(info["se"] >= 1.0 / np.sqrt(info["information"]) * (1 - 1e-12))
    # The direction is normalized; its sign does not matter.
    pd.testing.assert_frame_equal(test_information(res, THETA2, direction=-5 * direction), info)
    items = item_information(res, THETA2, direction=direction)
    np.testing.assert_allclose(items.sum(axis=1), info["information"], rtol=1e-13)


def test_theta_layout_and_index() -> None:
    res = _cross()
    out = item_information(res, THETA2, direction=[1.0, 1.0])
    assert isinstance(out.index, pd.MultiIndex)
    assert list(out.index.names) == ["f1", "f2"]
    np.testing.assert_array_equal(np.asarray(out.index.tolist()), THETA2)
    # A DataFrame is matched by factor name, whatever its column order.
    frame = pd.DataFrame(THETA2[:, ::-1], columns=["f2", "f1"])
    pd.testing.assert_frame_equal(item_information(res, frame, direction=[1.0, 1.0]), out)
    # A 1-D array of length D is one point.
    one = expected_score(res, THETA2[2])
    assert len(one) == 1
    np.testing.assert_allclose(one.to_numpy(), expected_score(res, THETA2[2:3]).to_numpy())
    assert information_matrix(res, THETA2[2]).shape == (1, 2, 2)


def test_unidimensional_information_matrix_is_the_scalar() -> None:
    uni = _unidimensional(["x1", "x2", "x3"])
    theta = np.array([-1.0, 0.0, 2.0])
    matrices = information_matrix(uni, theta)
    assert matrices.shape == (3, 1, 1)
    np.testing.assert_allclose(
        matrices[:, 0, 0], test_information(uni, theta)["information"], rtol=1e-14
    )
    np.testing.assert_allclose(
        information_matrix(uni, theta, item="x2")[:, 0, 0],
        item_information(uni, theta)["x2"],
        rtol=1e-14,
    )


def test_singular_information_matrix_gives_infinite_se() -> None:
    # A fitted model cannot have all slope vectors parallel (identification refuses it), so
    # the rule is checked on the helper: a rank-1 matrix (rounded as the kernel would) and 0.
    a = np.array([1.1, 0.7])
    rank_one = np.outer(a, a) * 0.23
    rank_one[0, 1] = rank_one[1, 0] = rank_one[0, 1] * (1 + 2e-16)
    regular = rank_one + np.diag([0.0, 1e-3])
    matrices = np.stack([rank_one, np.zeros((2, 2)), regular])
    se = _composite_se(matrices, np.array([1.0, 0.0]))
    assert np.isinf(se[0]) and np.isinf(se[1])
    assert se[2] == pytest.approx(np.sqrt(np.linalg.inv(regular)[0, 0]), rel=1e-12)


def test_bifactor_information_lives_in_the_general_and_own_specific_block() -> None:
    items = [f"q{j}" for j in range(1, 7)]
    spec = Model.uniform(TwoPL(), items=items).with_bifactor(
        specific={"s1": items[:3], "s2": items[3:]}
    )
    params = {
        item: {"a1": 1.0 + 0.1 * j, f"a{2 + j // 3}": 0.4 + 0.15 * j, "d": 0.2 * j - 0.5}
        for j, item in enumerate(items)
    }
    res = _fixed(spec, params, _dummy([TwoPL()] * 6, items))
    theta = np.array([[0.3, -1.0, 0.5], [-2.0, 1.0, 1.5]])
    for j, item in enumerate(items):
        matrix = information_matrix(res, theta, item=item)
        other = 2 if j < 3 else 1
        np.testing.assert_array_equal(matrix[:, other, :], 0.0)
        np.testing.assert_array_equal(matrix[:, :, other], 0.0)
        assert np.all(matrix[:, 0, 0] > 0)
    out = test_information(res, theta, direction=[1.0, 0.0, 0.0])
    assert list(out.index.names) == ["general", "s1", "s2"]
    assert np.all(np.isfinite(out["se"]))


@pytest.mark.slow
def test_multiple_groups_use_the_group_parameters() -> None:
    items = list(ITEMS)
    models = [m for m, _ in ITEMS.values()]
    spec = (
        Model.from_items(dict(zip(items, models, strict=True)))
        .with_factors({"f1": items[:3], "f2": items[3:]})
        .with_groups(ref=Normal(), focal=Normal())
    )
    dummy = _dummy(models, items)
    labels = np.repeat(["ref", "focal"], len(dummy))
    data = pd.concat([dummy, dummy], ignore_index=True)
    table = spec.parameter_table(data, group=labels)
    values = []
    for group, item, param, value in zip(
        table["group"], table["item"], table["param"], table["value"], strict=True
    ):
        if item == "__latent__":
            values.append(float(value))
            continue
        v = float(ITEMS[item][1].get(param, value))
        values.append(2.0 * v if group == "focal" and param.startswith("a") else v)
    table["value"] = values
    table["free"] = False
    res = fit(spec, data, group=labels, table=table, se=None)
    ref = information_matrix(res, THETA2, group="ref")
    np.testing.assert_allclose(ref, information_matrix(_simple(), THETA2), rtol=1e-12)
    focal = item_information(res, THETA2, direction=[1.0, 0.0], group="focal")
    assert not np.allclose(
        focal.to_numpy(),
        item_information(res, THETA2, direction=[1.0, 0.0], group="ref").to_numpy(),
    )
    with pytest.raises(SpecificationError, match="group="):
        information_matrix(res, THETA2)


def test_errors() -> None:
    res = _cross()
    uni = _unidimensional(["x1", "x2", "x3"])
    for fn in (item_information, test_information):
        with pytest.raises(SpecificationError, match="needs direction"):
            fn(res, THETA2)
        with pytest.raises(SpecificationError, match="length 2"):
            fn(res, THETA2, direction=[1.0, 0.0, 0.0])
        with pytest.raises(SpecificationError, match="non-zero"):
            fn(res, THETA2, direction=[0.0, 0.0])
        with pytest.raises(SpecificationError, match="non-zero"):
            fn(res, THETA2, direction=[np.nan, 1.0])
        with pytest.raises(SpecificationError, match="only used for multidimensional"):
            fn(uni, [0.0], direction=[1.0])
    for fn in (information_matrix, expected_score):
        with pytest.raises(SpecificationError, match=r"shape \(T, 2\)"):
            fn(res, np.zeros((3, 3)))
        with pytest.raises(SpecificationError, match=r"shape \(T, 2\)"):
            fn(res, np.zeros((0, 2)))
        with pytest.raises(SpecificationError, match="finite"):
            fn(res, [[0.0, np.inf]])
        with pytest.raises(SpecificationError, match="factor columns"):
            fn(res, pd.DataFrame(THETA2, columns=["f1", "g"]))
        with pytest.raises(SpecificationError, match="array of numbers"):
            fn(res, [["a", "b"]])
    with pytest.raises(SpecificationError, match="Unknown item"):
        information_matrix(res, THETA2, item="nope")
    with pytest.raises(SpecificationError, match="FitResult"):
        information_matrix("not a fit", THETA2)  # type: ignore[arg-type]
