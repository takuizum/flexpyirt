"""Unit tests of common-item screening (DESIGN.md §20; the R comparison is test_r_v50_screening)."""

from __future__ import annotations

import warnings
from functools import cache

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from flexpyirt import (
    GPCM,
    GRM,
    CommonItemScreeningResult,
    FitResult,
    FourPL,
    Model,
    Nominal,
    Quadrature,
    Rasch,
    TwoPL,
    fit,
    screen_common_items,
    stack_screening_tables,
)
from flexpyirt._linking import transform_kernel_params
from flexpyirt._screening import expected_score_rmsd, item_location, robust_z
from flexpyirt.errors import ScreeningWarning, SpecificationError
from flexpyirt.items import ItemModel

A_TRUE, B_TRUE = 1.25, 0.4
N_ITEMS = 10


def test_robust_z_matches_the_formula_with_type7_quantiles() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 100.0, np.nan])
    # Finite values 1, 2, 3, 4, 100: median 3, type-7 quartiles 2 and 4, IQR 2.
    expected = (x - 3.0) / (0.74 * 2.0)
    np.testing.assert_allclose(robust_z(x), expected, equal_nan=True)
    x_even = np.array([0.1, -0.2, 0.4, 0.0])
    q1, q3 = np.quantile(x_even, [0.25, 0.75])
    np.testing.assert_allclose(robust_z(x_even), (x_even - np.median(x_even)) / (0.74 * (q3 - q1)))


def test_robust_z_is_nan_when_the_iqr_vanishes() -> None:
    assert np.all(np.isnan(robust_z(np.full(5, -0.3))))
    assert np.all(np.isnan(robust_z(np.array([np.nan, np.nan]))))


ITEM_CASES: list[tuple[ItemModel, list[float]]] = [
    (TwoPL(), [1.3, -0.4]),
    (FourPL(), [1.1, 0.3, 0.15, 0.95]),
    (GRM(n_categories=4), [1.2, 1.5, 0.2, -1.1]),
    (GPCM(n_categories=4), [0.9, 0.8, 0.9, -0.2]),
    (Nominal(n_categories=3), [1.0, 0.7, 0.4, -0.3]),
]


@pytest.mark.parametrize(("item_model", "params"), ITEM_CASES, ids=lambda v: str(v)[:20])
def test_item_location_transforms_as_a_b_plus_b(item_model: ItemModel, params: list[float]) -> None:
    kernel = item_model.kernel
    xi = jnp.asarray(params, dtype=jnp.float64)
    xi_star = transform_kernel_params(kernel, xi, jnp.float64(A_TRUE), jnp.float64(B_TRUE))
    assert float(item_location(kernel, xi_star)) == pytest.approx(
        A_TRUE * float(item_location(kernel, xi)) + B_TRUE, abs=1e-12
    )


def test_item_location_is_the_mean_threshold() -> None:
    kernel = GRM(n_categories=4).kernel
    xi = jnp.asarray([1.2, 1.5, 0.2, -1.1], dtype=jnp.float64)
    assert float(item_location(kernel, xi)) == pytest.approx(np.mean([-1.5, -0.2, 1.1]) / 1.2)
    nominal = Nominal(n_categories=3).kernel
    xi_n = jnp.asarray([1.0, 0.7, 0.4, -0.3], dtype=jnp.float64)
    assert float(item_location(nominal, xi_n)) == pytest.approx(0.3 / 2.0)


@pytest.mark.parametrize(("item_model", "params"), ITEM_CASES, ids=lambda v: str(v)[:20])
def test_rmsd_compares_source_curve_at_moved_nodes(
    item_model: ItemModel, params: list[float]
) -> None:
    kernel = item_model.kernel
    nodes = jnp.linspace(-4.0, 4.0, 41)
    weights = jnp.exp(-0.5 * nodes**2)
    weights = weights / weights.sum()
    src = jnp.asarray(params, dtype=jnp.float64)
    tgt = src.at[0].multiply(1.1)
    src_star = transform_kernel_params(kernel, src, jnp.float64(A_TRUE), jnp.float64(B_TRUE))
    scores = jnp.arange(kernel.n_categories, dtype=jnp.float64)
    e_moved = jnp.exp(kernel.log_prob(((nodes - B_TRUE) / A_TRUE)[:, None], src)) @ scores
    e_tgt = jnp.exp(kernel.log_prob(nodes[:, None], tgt)) @ scores
    expected = float(jnp.sqrt(jnp.sum(weights * (e_moved - e_tgt) ** 2)))
    got = float(expected_score_rmsd(kernel, src_star, tgt, nodes, weights))
    assert got == pytest.approx(expected, rel=1e-12)
    assert float(expected_score_rmsd(kernel, tgt, tgt, nodes, weights)) == 0.0


# ---- screen_common_items on fixed fits ---------------------------------------------------------


def _fixed_fit(item_model: ItemModel, values: dict[str, list[float]], seed: int) -> FitResult:
    """A fit with every parameter fixed at ``values`` (a small random data set)."""
    names = list(values)
    spec = Model.uniform(item_model, items=names)
    rng = np.random.default_rng(seed)
    n_cat = item_model.kernel.n_categories
    data = pd.DataFrame(rng.integers(0, n_cat, size=(60, len(names))), columns=names)
    table = spec.parameter_table(data)
    vals = []
    for item, param in zip(table["item"], table["param"], strict=True):
        if item == "__latent__":
            vals.append(0.0 if param == "mean" else 1.0)
        else:
            vals.append(values[item][list(item_model.kernel.param_names).index(param)])
    table["value"] = vals
    table["free"] = False
    return fit(spec, data, table=table, quadrature=Quadrature.equispaced(31), se=None)


def _forms(
    drift: dict[str, float], *, rasch: bool = False, noise: bool = False
) -> tuple[FitResult, FitResult]:
    """Target 2PL items and source items on the scale theta_src = (theta - B) / A, with drift.

    The source item ``i`` has ``a_src = A * a``, ``d_src = d + a * B`` (so that transforming it with
    the true constants recovers the target item), and its location shifted by ``drift[i]``.
    ``noise`` adds small estimation-like perturbations (|db| <= 0.05, slopes within 5%), without
    which the IQR of the undrifted items is 0.
    """
    rng = np.random.default_rng(11)
    a = np.ones(N_ITEMS) if rasch else rng.uniform(0.8, 2.0, N_ITEMS)
    b = np.linspace(-1.5, 1.5, N_ITEMS)
    db = rng.uniform(-0.05, 0.05, N_ITEMS) if noise else np.zeros(N_ITEMS)
    da = rng.uniform(0.95, 1.05, N_ITEMS) if noise and not rasch else np.ones(N_ITEMS)
    names = [f"i{j:02d}" for j in range(N_ITEMS)]
    tgt = {n: [float(a[j]), float(-a[j] * b[j])] for j, n in enumerate(names)}
    a_const = 1.0 if rasch else A_TRUE
    src = {}
    for j, n in enumerate(names):
        b_src = (b[j] + db[j] + drift.get(n, 0.0) - B_TRUE) / a_const
        a_src = a[j] * da[j] * a_const
        src[n] = [float(a_src), float(-a_src * b_src)]
    model = Rasch() if rasch else TwoPL()
    return _fixed_fit(model, src, 1), _fixed_fit(model, tgt, 2)


@cache
def _drift_forms() -> tuple[FitResult, FitResult]:
    return _forms({"i02": 0.9, "i07": -0.7}, noise=True)


def test_no_drift_recovers_constants_and_flags_nothing() -> None:
    source, target = _forms({})
    res = screen_common_items(source, target, statistic="displacement")
    assert res.removed_items == ()
    assert pytest.approx(A_TRUE, abs=1e-7) == res.link.A
    assert pytest.approx(B_TRUE, abs=1e-7) == res.link.B
    np.testing.assert_allclose(res.table["displacement"], 0.0, atol=1e-7)
    np.testing.assert_allclose(res.table["rmsd"], 0.0, atol=1e-7)
    assert list(res.history["iteration"]) == [0]


@pytest.mark.parametrize("statistic", ["displacement", "rmsd", "robust_z"])
def test_purification_removes_the_drifted_items_and_recovers_the_constants(
    statistic: str,
) -> None:
    source, target = _drift_forms()
    res = screen_common_items(source, target, statistic=statistic)  # type: ignore[arg-type]
    assert res.removed_items == ("i02", "i07")
    assert pytest.approx(A_TRUE, abs=0.05) == res.link.A
    assert pytest.approx(B_TRUE, abs=0.05) == res.link.B
    kept = res.table.set_index("item")
    assert kept.loc["i02", "difference_b"] == pytest.approx(0.9, abs=0.1)
    assert kept.loc["i07", "difference_b"] == pytest.approx(-0.7, abs=0.1)
    assert kept.loc["i02", "iteration_removed"] == 1
    assert kept.loc["i07", "iteration_removed"] == 2
    assert kept["iteration_removed"].isna().sum() == N_ITEMS - 2
    assert res.kept_items == tuple(n for n in kept.index if n not in {"i02", "i07"})
    assert list(res.history["n_items"]) == [10, 9, 8]
    assert list(res.history["removed"]) == [("i02",), ("i07",), ()]
    assert res.initial_link.common_items == tuple(kept.index)
    assert set(res.flagged_items) == {"i02", "i07"}


def test_removal_all_removes_every_flagged_item_in_one_step() -> None:
    source, target = _drift_forms()
    res = screen_common_items(source, target, statistic="displacement", removal="all")
    assert res.removed_items == ("i02", "i07")
    assert list(res.history["removed"]) == [("i02", "i07"), ()]


def test_purify_false_only_reports() -> None:
    source, target = _drift_forms()
    res = screen_common_items(source, target, statistic="displacement", purify=False)
    assert res.removed_items == ()
    assert res.link is res.initial_link
    assert res.table["kept"].all()
    assert res.table.set_index("item").loc["i02", "flagged"]


def test_max_iter_and_min_items_stop_with_a_warning() -> None:
    source, target = _drift_forms()
    with pytest.warns(ScreeningWarning, match="max_iter=1"):
        res = screen_common_items(source, target, statistic="displacement", max_iter=1)
    assert res.removed_items == ("i02",)
    with pytest.warns(ScreeningWarning, match="min_items=9"):
        res = screen_common_items(source, target, statistic="displacement", min_items=9)
    assert res.removed_items == ("i02",)
    with pytest.warns(ScreeningWarning, match="min_items=9"):
        res = screen_common_items(
            source, target, statistic="displacement", min_items=9, removal="all"
        )
    assert res.removed_items == ("i02",)


def test_rasch_slope_ratio_robust_z_is_nan() -> None:
    source, target = _forms({"i04": 0.8}, rasch=True, noise=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ScreeningWarning)
        res = screen_common_items(source, target)
    assert res.table["robust_z_a"].isna().all()
    assert res.removed_items[0] == "i04"


def test_items_mapping_and_stacking() -> None:
    source, target = _drift_forms()
    res = screen_common_items(
        source, target, items=[f"i{j:02d}" for j in range(6)], statistic="displacement"
    )
    assert list(res.table["item"]) == [f"i{j:02d}" for j in range(6)]
    assert res.removed_items == ("i02",)
    stacked = stack_screening_tables({"2025": res, "2026": res})
    assert list(stacked.columns[:6]) == ["administration", "item", "target_item", "model", "A", "B"]
    assert list(stacked["administration"]) == ["2025"] * 6 + ["2026"] * 6
    assert stacked["A"].iloc[0] == pytest.approx(res.link.A)


def test_result_rendering() -> None:
    source, target = _drift_forms()
    res = screen_common_items(source, target, statistic="displacement")
    assert isinstance(res, CommonItemScreeningResult)
    text = res.summary(digits=3)
    assert "Removed:       i02, i07" in text
    assert "removed=['i02', 'i07']" in str(res)
    assert "Purification steps" in res._repr_html_()
    with pytest.raises(SpecificationError, match="digits"):
        res.summary(digits=-1)


def test_invalid_arguments_raise() -> None:
    source, target = _drift_forms()
    with pytest.raises(SpecificationError, match="statistic"):
        screen_common_items(source, target, statistic="lord")  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match="removal"):
        screen_common_items(source, target, removal="first")  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match="threshold"):
        screen_common_items(source, target, threshold=0.0)
    with pytest.raises(SpecificationError, match="min_items"):
        screen_common_items(source, target, min_items=0)
    with pytest.raises(SpecificationError, match="max_iter"):
        screen_common_items(source, target, max_iter=-1)
    with pytest.raises(SpecificationError, match="purify"):
        screen_common_items(source, target, purify=1)  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match="source must be a FitResult"):
        screen_common_items(object(), target)  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match="non-empty"):
        stack_screening_tables({})
    with pytest.raises(SpecificationError, match="CommonItemScreeningResult"):
        stack_screening_tables({"x": object()})  # type: ignore[dict-item]
