"""Unit tests of the plot data functions and smoke tests of flexpyirt.plotting (DESIGN.md §21).

The numbers are checked against closed forms and brute-force recomputations here and against
mirt / TAM in ``tests/test_mirt_v48_plots.py`` (V48). The figures are only smoke-tested on the
Agg back end: the objects are created and the plotted data equal the DataFrames.
"""

from __future__ import annotations

import subprocess
import sys
from functools import cache
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pytest

from flexpyirt import (
    GRM,
    FitResult,
    Model,
    Normal,
    Quadrature,
    ThreePL,
    TwoPL,
    empirical_item_curve,
    expected_score,
    fit,
    item_probabilities,
    test_information,
    wright_map_table,
)
from flexpyirt.curves import _bin_sizes
from flexpyirt.errors import MissingDependencyError, SpecificationError

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from flexpyirt import plotting

A = [1.2, 0.8, 1.6, 1.0]
D = [0.5, -0.3, 0.0, 1.1]
GRM_D = [1.0, -0.2, -1.4]
GRM_A = 1.3


def _fixed(
    data: pd.DataFrame,
    items: dict[str, Any],
    values: dict[tuple[str | None, str, str], float],
    groups: np.ndarray | None = None,
    latent: dict[str, Normal] | None = None,
) -> FitResult:
    spec = Model.from_items(items)
    if latent is not None:
        spec = spec.with_groups(**latent)
    table = spec.parameter_table(data, group=groups)
    out = []
    for g, item, param, value in zip(
        table["group"], table["item"], table["param"], table["value"], strict=True
    ):
        key = (g if groups is not None else None, item, param)
        out.append(values.get(key, values.get((None, item, param), float(value))))
    table["value"] = np.asarray(out, dtype=float)
    table["free"] = False
    quad = Quadrature.equispaced(41, bounds=(-6.0, 6.0))
    return fit(spec, data, group=groups, table=table, quadrature=quad, se=None)


def _simulate(n: int, seed: int, shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    theta = rng.normal(shift, 1.0, n)
    cols: dict[str, np.ndarray] = {}
    for j, (a, d) in enumerate(zip(A, D, strict=True)):
        cols[f"i{j + 1}"] = (rng.random(n) < 1.0 / (1.0 + np.exp(-(a * theta + d)))).astype(float)
    upper = 1.0 / (1.0 + np.exp(-(GRM_A * theta[:, None] + np.asarray(GRM_D))))
    cols["g1"] = (rng.random(n)[:, None] < upper).sum(axis=1).astype(float)
    return pd.DataFrame(cols)


def _values() -> dict[tuple[str | None, str, str], float]:
    values: dict[tuple[str | None, str, str], float] = {}
    for j, (a, d) in enumerate(zip(A, D, strict=True)):
        values[(None, f"i{j + 1}", "a")] = a
        values[(None, f"i{j + 1}", "d")] = d
    values[(None, "g1", "a")] = GRM_A
    for k, d in enumerate(GRM_D, start=1):
        values[(None, "g1", f"d{k}")] = d
    return values


ITEMS = {**{f"i{j}": TwoPL() for j in range(1, 5)}, "g1": GRM(n_categories=4)}


@cache
def _single() -> tuple[FitResult, pd.DataFrame]:
    data = _simulate(503, 1)
    data.iloc[::7, 0] = np.nan
    return _fixed(data, ITEMS, _values()), data


@cache
def _two_groups() -> tuple[FitResult, pd.DataFrame, np.ndarray]:
    data = pd.concat([_simulate(300, 2), _simulate(300, 3, shift=0.5)], ignore_index=True)
    groups = np.repeat(["ref", "focal"], 300)
    values = _values()
    values[("focal", "i1", "d")] = -0.5  # DIF in item i1
    latent = {"ref": Normal(mean=0.0, var=1.0), "focal": Normal(mean=0.5, var=1.2)}
    return _fixed(data, ITEMS, values, groups, latent), data, groups


# --- numbers ---------------------------------------------------------------------------------


def test_bin_sizes_follow_mirt_rule() -> None:
    assert _bin_sizes(1003, 10).tolist() == [101, *[100] * 8, 102]
    assert _bin_sizes(7, 3).tolist() == [2, 2, 3]
    assert _bin_sizes(9, 1).tolist() == [9]
    assert _bin_sizes(14, 4).tolist() == [4, 3, 3, 4]


def test_item_probabilities_closed_form_and_sum() -> None:
    res, _ = _single()
    theta = np.linspace(-3, 3, 7)
    out = item_probabilities(res, theta)
    assert list(out.columns) == ["item", "category", "theta", "probability"]
    p1 = out[(out["item"] == "i1") & (out["category"] == 1)]["probability"].to_numpy()
    np.testing.assert_allclose(p1, 1.0 / (1.0 + np.exp(-(A[0] * theta + D[0]))), rtol=1e-13)
    sums = out.groupby(["item", "theta"])["probability"].sum()
    np.testing.assert_allclose(sums.to_numpy(), 1.0, atol=1e-14)
    one = item_probabilities(res, 0.0, items="g1")
    assert one["category"].tolist() == [0, 1, 2, 3]


def test_empirical_item_curve_brute_force() -> None:
    res, data = _single()
    out = empirical_item_curve(res, data, n_bins=5)
    eap = res.score(data)["theta"].to_numpy()
    for item, k_j in [("i1", 2), ("g1", 4)]:
        answered = data[item].notna().to_numpy()
        n_j = int(answered.sum())
        order = np.argsort(eap[answered], kind="stable")
        theta_sorted = eap[answered][order]
        y_sorted = data[item].to_numpy()[answered][order].astype(int)
        sizes = _bin_sizes(n_j, 5)
        starts = np.concatenate([[0], np.cumsum(sizes)])
        got = out[out["item"] == item]
        for b in range(5):
            sl = slice(starts[b], starts[b + 1])
            rows = got[got["bin"] == b + 1]
            mean = theta_sorted[sl].mean()
            assert rows["n"].iloc[0] == sizes[b]
            assert rows["theta"].iloc[0] == pytest.approx(mean, abs=1e-14)
            props = np.bincount(y_sorted[sl], minlength=k_j) / sizes[b]
            np.testing.assert_allclose(rows["observed"], props, atol=1e-15)
            model = item_probabilities(res, mean, items=item)["probability"].to_numpy()
            np.testing.assert_allclose(rows["expected"], model, atol=1e-14)
    assert out[out["item"] == "i1"]["n"].drop_duplicates().sum() <= len(data)


def test_empirical_item_curve_two_groups_has_group_column() -> None:
    res, data, groups = _two_groups()
    out = empirical_item_curve(res, data, items=["i1"], n_bins=4, group=groups)
    assert next(iter(out.columns)) == "group"
    assert set(out["group"]) == {"ref", "focal"}
    assert out.groupby("group")["n"].sum().tolist() == [600, 600]  # 2 categories x 300


def test_empirical_item_curve_errors() -> None:
    res, data = _single()
    with pytest.raises(SpecificationError, match="n_bins"):
        empirical_item_curve(res, data, n_bins=0)
    with pytest.raises(SpecificationError, match="Unknown items"):
        empirical_item_curve(res, data, items=["nope"])
    with pytest.raises(SpecificationError, match="fewer than n_bins"):
        empirical_item_curve(res, data.iloc[:5], n_bins=10)


def test_wright_map_locations_closed_forms() -> None:
    res, data = _single()
    table = wright_map_table(res, data)
    assert list(table.persons.columns) == ["theta"]
    assert table.persons.index.equals(data.index)
    items = table.items.set_index(["item", "category"])["location"]
    for j, (a, d) in enumerate(zip(A, D, strict=True)):
        assert items[(f"i{j + 1}", 1)] == pytest.approx(-d / a, abs=1e-12)
    for k, d in enumerate(GRM_D, start=1):
        assert items[("g1", k)] == pytest.approx(-d / GRM_A, abs=1e-12)


def test_wright_map_three_pl_without_crossing_is_nan() -> None:
    data = pd.DataFrame({"x": [0.0, 1.0, 1.0, 0.0], "y": [1.0, 1.0, 0.0, 0.0]})
    values: dict[tuple[str | None, str, str], float] = {
        (None, "x", "a"): 1.0,
        (None, "x", "d"): 0.0,
        (None, "x", "g"): 0.6,
        (None, "y", "a"): 1.5,
        (None, "y", "d"): 0.3,
        (None, "y", "g"): 0.2,
    }
    res = _fixed(data, {"x": ThreePL(), "y": ThreePL()}, values)
    loc = wright_map_table(res, data).items.set_index("item")["location"]
    assert np.isnan(loc["x"])
    # 0.2 + 0.8 * sigmoid(1.5 t + 0.3) = 1/2  ->  t = (logit(0.375) - 0.3) / 1.5
    want = (np.log(0.375 / 0.625) - 0.3) / 1.5
    assert loc["y"] == pytest.approx(want, abs=1e-12)


def test_wright_map_two_groups_and_method() -> None:
    res, data, groups = _two_groups()
    table = wright_map_table(res, data, group=groups, method="MAP")
    assert list(table.persons.columns) == ["group", "theta"]
    assert list(table.persons["group"]) == list(groups)
    loc = table.items.set_index(["group", "item"])["location"]
    assert loc[("ref", "i1")] == pytest.approx(-D[0] / A[0], abs=1e-12)
    assert loc[("focal", "i1")] == pytest.approx(0.5 / A[0], abs=1e-12)
    with pytest.raises(SpecificationError, match="scores persons"):
        wright_map_table(res, data, group=groups, method="ML")  # type: ignore[arg-type]


# --- figures (Agg) --------------------------------------------------------------------------


def test_plot_item_curves_draws_the_dataframes() -> None:
    res, data = _single()
    theta = np.linspace(-3, 3, 31)
    fig = plotting.plot_item_curves(res, ["i1", "g1"], data=data, n_bins=5, theta=theta)
    panels = [ax for ax in fig.axes if ax.get_visible()]
    assert [ax.get_title() for ax in panels] == ["i1", "g1"]
    curves = item_probabilities(res, theta, items=["i1", "g1"])
    emp = empirical_item_curve(res, data, items=["i1", "g1"], n_bins=5)
    i1, g1 = panels
    assert len(i1.lines) == 1 and len(g1.lines) == 4
    want = curves[(curves["item"] == "i1") & (curves["category"] == 1)]["probability"]
    np.testing.assert_array_equal(i1.lines[0].get_ydata(), want.to_numpy())
    points = i1.collections[0].get_offsets()
    rows = emp[(emp["item"] == "i1") & (emp["category"] == 1)]
    np.testing.assert_array_equal(np.asarray(points)[:, 1], rows["observed"].to_numpy())
    for k in range(4):
        want_k = curves[(curves["item"] == "g1") & (curves["category"] == k)]["probability"]
        np.testing.assert_array_equal(g1.lines[k].get_ydata(), want_k.to_numpy())
    plt.close("all")


def test_plot_item_curves_dif_one_line_per_group() -> None:
    res, data, groups = _two_groups()
    fig = plotting.plot_item_curves(res, "i1", data=data, group=groups, n_bins=4)
    (ax,) = [a for a in fig.axes if a.get_visible()]
    assert [line.get_label() for line in ax.lines] == ["ref", "focal"]
    for line, g in zip(ax.lines, ["ref", "focal"], strict=True):
        want = item_probabilities(res, line.get_xdata(), items="i1", group=g)
        np.testing.assert_array_equal(
            line.get_ydata(), want[want["category"] == 1]["probability"].to_numpy()
        )
    assert len(ax.collections) == 2
    _, given = plt.subplots()
    assert plotting.plot_item_curves(res, "i1", ax=given) is given.figure
    with pytest.raises(SpecificationError, match="single item"):
        plotting.plot_item_curves(res, ["i1", "i2"], ax=given)
    with pytest.raises(SpecificationError, match="only used together with data"):
        plotting.plot_item_curves(res, "i1", group=groups)
    plt.close("all")


def test_plot_information_and_expected_score() -> None:
    res, _ = _single()
    theta = np.linspace(-3, 3, 21)
    ax = plotting.plot_information(res, standard_error=True, theta=theta)
    info = test_information(res, theta)
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), info["information"].to_numpy())
    twin = next(a for a in ax.figure.axes if a is not ax)
    np.testing.assert_array_equal(twin.lines[0].get_ydata(), info["se"].to_numpy())
    ax = plotting.plot_information(res, ["i1", "i2"], theta=theta)
    assert [line.get_label() for line in ax.lines] == ["i1", "i2"]
    with pytest.raises(SpecificationError, match="standard_error"):
        plotting.plot_information(res, "i1", standard_error=True)
    ax = plotting.plot_expected_score(res, theta=theta)
    np.testing.assert_array_equal(
        ax.lines[0].get_ydata(), expected_score(res, theta)["total"].to_numpy()
    )
    res2, _, _ = _two_groups()
    assert len(plotting.plot_expected_score(res2).lines) == 2
    assert len(plotting.plot_information(res2, group="focal").lines) == 1
    plt.close("all")


def test_plot_wright_map() -> None:
    res, data = _single()
    fig = plotting.plot_wright_map(res, data, n_bins=12)
    left, right = fig.axes
    table = wright_map_table(res, data)
    widths = [p.get_width() for p in left.patches]  # type: ignore[attr-defined]
    assert sum(widths) == len(data)
    offsets = np.asarray(right.collections[0].get_offsets())
    np.testing.assert_array_equal(offsets[:, 1], table.items["location"].to_numpy())
    res2, data2, groups = _two_groups()
    fig = plotting.plot_wright_map(res2, data2, group=groups)
    assert len(fig.axes[1].collections) == 2
    plt.close("all")


# --- without matplotlib ---------------------------------------------------------------------


def test_missing_matplotlib_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    res, data = _single()
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    with pytest.raises(MissingDependencyError, match=r"flexpyirt\[plot\]"):
        plotting.plot_expected_score(res)
    with pytest.raises(ImportError):
        plotting.plot_wright_map(res, data)


@pytest.mark.slow
def test_import_flexpyirt_does_not_import_matplotlib() -> None:
    code = (
        "import sys; sys.modules['matplotlib'] = None; "
        "import flexpyirt, flexpyirt.plotting; "
        "assert flexpyirt.empirical_item_curve and flexpyirt.plotting.plot_item_curves; "
        "print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "ok"
