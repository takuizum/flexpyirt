r"""Figures of a fitted unidimensional model, drawn with matplotlib (DESIGN.md §20).

matplotlib is an optional dependency (``pip install "flexpyirt[plot]"``). It is imported when a
plot function is called, never by ``import flexpyirt`` or ``import flexpyirt.plotting``; without
it the plot functions raise :class:`~flexpyirt.errors.MissingDependencyError`.

The functions are thin: every number they draw comes from a public function returning a
DataFrame (:func:`~flexpyirt.item_probabilities`, :func:`~flexpyirt.empirical_item_curve`,
:func:`~flexpyirt.item_information`, :func:`~flexpyirt.test_information`,
:func:`~flexpyirt.expected_score`, :func:`~flexpyirt.wright_map_table`), and only those
numbers are validated. They return matplotlib objects and never call ``plt.show()``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np

from flexpyirt.curves import (
    PersonMethod,
    _select_items,
    empirical_item_curve,
    item_probabilities,
    wright_map_table,
)
from flexpyirt.errors import MissingDependencyError, SpecificationError
from flexpyirt.information import (
    _coerce_theta,
    _group_names,
    _require_unidimensional,
    _resolve_group,
    expected_score,
    item_information,
    test_information,
)
from flexpyirt.result import FitResult

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "plot_expected_score",
    "plot_information",
    "plot_item_curves",
    "plot_wright_map",
]

_DEFAULT_THETA = np.linspace(-4.0, 4.0, 161)
r"""Default :math:`\theta` grid of the curves."""
_PANEL_COLUMNS = 3
"""Panels per row of :func:`plot_item_curves`."""
_THETA_LABEL = r"$\theta$"


def _pyplot() -> ModuleType:
    """Import and return ``matplotlib.pyplot``, or raise a clear error when it is missing."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise MissingDependencyError(
            "flexpyirt.plotting needs matplotlib, which is not installed; install it with "
            '`pip install "flexpyirt[plot]"` (or `uv add "flexpyirt[plot]"`).'
        ) from exc
    return plt


def _theta(theta: Any) -> np.ndarray:
    return _DEFAULT_THETA if theta is None else _coerce_theta(theta)


def _groups_to_draw(res: FitResult, group: str | None, what: str) -> list[str | None]:
    """``[group]`` when given, every group of a multiple-group model, else ``[None]``."""
    if group is not None or not res._model.is_multigroup:
        _resolve_group(res, group, what)
        return [group]
    return list(_group_names(res))


def _new_axes(ax: Axes | None) -> Axes:
    if ax is not None:
        return ax
    _, new = _pyplot().subplots()
    return new  # type: ignore[no-any-return]


def plot_item_curves(
    res: FitResult,
    items: str | Sequence[str] | None = None,
    *,
    data: Any = None,
    group: Any = None,
    n_bins: int = 10,
    theta: Any = None,
    item_names: list[str] | None = None,
    ax: Axes | None = None,
) -> Figure:
    r"""Plot item characteristic curves, with observed proportions when ``data`` is given.

    One panel per item. A dichotomous item shows :math:`P_{j1}(\theta)`; a polytomous item
    shows every category :math:`P_{jk}(\theta)`. The curves are
    :func:`~flexpyirt.item_probabilities`; with ``data``, the points are the bin proportions
    :math:`(\bar\theta_b, O_{bk})` of :func:`~flexpyirt.empirical_item_curve`. A multiple-group
    model draws one curve per group (the DIF plot of the item).

    Parameters
    ----------
    res : FitResult
        A fitted unidimensional model.
    items : str or sequence of str, optional
        Items to plot (default: all).
    data : DataFrame or array-like, optional
        Responses; when given, the observed proportions are drawn as points.
    group : array-like, optional
        Group label of each row of ``data`` (required with ``data`` for a multiple-group
        model).
    n_bins : int, default 10
        Number of :math:`\theta` bins of the observed proportions.
    theta : array-like, optional
        :math:`\theta` grid of the curves (default: 161 points on :math:`[-4, 4]`).
    item_names : list of str, optional
        Column names when ``data`` is an array.
    ax : matplotlib Axes, optional
        Axes to draw into; only for a single item.

    Returns
    -------
    matplotlib Figure
        The figure holding the panels (``ax.figure`` when ``ax`` is given).

    Raises
    ------
    MissingDependencyError
        If matplotlib is not installed.
    SpecificationError
        As :func:`~flexpyirt.empirical_item_curve`, and for ``ax`` with more than one item or
        ``group`` / ``item_names`` without ``data``.
    """
    _require_unidimensional(res, "plot_item_curves()")
    plt = _pyplot()
    chosen = _select_items(res, items)
    if ax is not None and len(chosen) != 1:
        raise SpecificationError("ax= draws a single item; pass one item or leave ax=None.")
    if data is None and (group is not None or item_names is not None):
        raise SpecificationError("group= and item_names= are only used together with data.")
    grid = _theta(theta)
    groups = _groups_to_draw(res, None, "plot_item_curves()")
    observed = (
        None
        if data is None
        else empirical_item_curve(
            res, data, items=chosen, n_bins=n_bins, group=group, item_names=item_names
        )
    )
    if ax is not None:
        axes = [ax]
        figure = ax.figure
    else:
        n_cols = min(_PANEL_COLUMNS, len(chosen))
        n_rows = math.ceil(len(chosen) / n_cols)
        figure, grid_axes = plt.subplots(
            n_rows, n_cols, squeeze=False, figsize=(4.0 * n_cols, 3.2 * n_rows)
        )
        axes = list(grid_axes.ravel())
        for unused in axes[len(chosen) :]:
            unused.set_visible(False)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for panel, item in zip(axes, chosen, strict=False):
        for g_pos, g in enumerate(groups):
            curves = item_probabilities(res, grid, items=item, group=g)
            n_categories = int(curves["category"].max()) + 1
            shown = [1] if n_categories == 2 else list(range(n_categories))
            for k_pos, k in enumerate(shown):
                color = colors[(g_pos if len(groups) > 1 else k_pos) % len(colors)]
                curve = curves[curves["category"] == k]
                label = ", ".join(
                    part
                    for part in (g, None if n_categories == 2 else f"category {k}")
                    if part is not None
                )
                panel.plot(
                    curve["theta"].to_numpy(),
                    curve["probability"].to_numpy(),
                    color=color,
                    linestyle=["-", "--", ":", "-."][k_pos % 4] if len(groups) > 1 else "-",
                    label=label or None,
                )
                if observed is not None:
                    rows = observed[(observed["item"] == item) & (observed["category"] == k)]
                    if g is not None:
                        rows = rows[rows["group"] == g]
                    panel.scatter(
                        rows["theta"].to_numpy(), rows["observed"].to_numpy(), color=color, s=14
                    )
        panel.set_title(item)
        panel.set_xlabel(_THETA_LABEL)
        panel.set_ylabel(r"$P(\theta)$")
        panel.set_ylim(-0.02, 1.02)
        if any(not str(line.get_label()).startswith("_") for line in panel.lines):
            panel.legend(fontsize="small")
    return figure  # type: ignore[return-value]


def plot_information(
    res: FitResult,
    items: str | Sequence[str] | None = None,
    *,
    standard_error: bool = False,
    group: str | None = None,
    theta: Any = None,
    ax: Axes | None = None,
) -> Axes:
    r"""Plot the test information (default) or the information of the given items.

    The curves are :func:`~flexpyirt.test_information` (column ``information``) or
    :func:`~flexpyirt.item_information`. With ``standard_error=True`` the conditional standard
    error :math:`I(\theta)^{-1/2}` (column ``se``) is drawn on a second y axis. A
    multiple-group model draws every group unless ``group`` names one.

    Parameters
    ----------
    res : FitResult
        A fitted unidimensional model.
    items : str or sequence of str, optional
        Items whose information is drawn; default: the test information.
    standard_error : bool, default False
        Also draw the standard error of the test information (not with ``items``).
    group : str, optional
        Group name (default: every group of a multiple-group model).
    theta : array-like, optional
        :math:`\theta` grid (default: 161 points on :math:`[-4, 4]`).
    ax : matplotlib Axes, optional
        Axes to draw into (default: a new figure).

    Returns
    -------
    matplotlib Axes
        The axes of the information curves; the standard error is on twin axes made with
        ``ax.twinx()`` (the other axes of ``ax.figure``).

    Raises
    ------
    MissingDependencyError
        If matplotlib is not installed.
    SpecificationError
        For a multidimensional model, an unknown item or group, or ``standard_error`` with
        ``items``.
    """
    _require_unidimensional(res, "plot_information()")
    _pyplot()
    if standard_error and items is not None:
        raise SpecificationError("standard_error=True draws the test information's SE only.")
    chosen = None if items is None else _select_items(res, items)
    grid = _theta(theta)
    groups = _groups_to_draw(res, group, "plot_information()")
    axes = _new_axes(ax)
    twin = axes.twinx() if standard_error else None
    for g in groups:
        prefix = "" if g is None else f"{g}: "
        if chosen is None:
            info = test_information(res, grid, group=g)
            axes.plot(grid, info["information"].to_numpy(), label=f"{prefix}information")
            if twin is not None:
                twin.plot(grid, info["se"].to_numpy(), linestyle="--", label=f"{prefix}SE")
        else:
            info = item_information(res, grid, group=g)
            for item in chosen:
                axes.plot(grid, info[item].to_numpy(), label=f"{prefix}{item}")
    axes.set_xlabel(_THETA_LABEL)
    axes.set_ylabel("information")
    if twin is not None:
        twin.set_ylabel("standard error")
    if len(axes.lines) > 1 or twin is not None:
        handles, labels = axes.get_legend_handles_labels()
        if twin is not None:
            more_handles, more_labels = twin.get_legend_handles_labels()
            handles, labels = handles + more_handles, labels + more_labels
        axes.legend(handles, labels, fontsize="small")
    return axes


def plot_expected_score(
    res: FitResult,
    *,
    group: str | None = None,
    theta: Any = None,
    ax: Axes | None = None,
) -> Axes:
    r"""Plot the test characteristic curve :math:`T(\theta)=\sum_jE_j(\theta)`.

    The curve is the column ``total`` of :func:`~flexpyirt.expected_score`. A multiple-group
    model draws every group unless ``group`` names one.

    Parameters
    ----------
    res : FitResult
        A fitted unidimensional model.
    group : str, optional
        Group name (default: every group of a multiple-group model).
    theta : array-like, optional
        :math:`\theta` grid (default: 161 points on :math:`[-4, 4]`).
    ax : matplotlib Axes, optional
        Axes to draw into (default: a new figure).

    Returns
    -------
    matplotlib Axes

    Raises
    ------
    MissingDependencyError
        If matplotlib is not installed.
    SpecificationError
        For a multidimensional model or an unknown group.
    """
    _require_unidimensional(res, "plot_expected_score()")
    _pyplot()
    grid = _theta(theta)
    groups = _groups_to_draw(res, group, "plot_expected_score()")
    axes = _new_axes(ax)
    for g in groups:
        total = expected_score(res, grid, group=g)["total"].to_numpy()
        axes.plot(grid, total, label=g)
    axes.set_xlabel(_THETA_LABEL)
    axes.set_ylabel("expected test score")
    if len(groups) > 1:
        axes.legend(fontsize="small")
    return axes


def plot_wright_map(
    res: FitResult,
    data: Any,
    *,
    group: Any = None,
    item_names: list[str] | None = None,
    method: PersonMethod = "EAP",
    n_bins: int = 30,
) -> Figure:
    r"""Plot an item–person (Wright) map: the person distribution beside the item locations.

    The left panel is a horizontal histogram of the person scores and the right panel shows
    each item's Thurstonian thresholds on the same :math:`\theta` axis; both come from
    :func:`~flexpyirt.wright_map_table`. A multiple-group model overlays one histogram and one
    set of item markers per group.

    Parameters
    ----------
    res : FitResult
        A fitted unidimensional model.
    data : DataFrame or array-like
        Responses on the fitted items (NaN is missing).
    group : array-like, optional
        Group label of each row of ``data`` (required for a multiple-group model).
    item_names : list of str, optional
        Column names when ``data`` is an array.
    method : {"EAP", "MAP", "WLE"}, default "EAP"
        Scoring method of the persons.
    n_bins : int, default 30
        Number of histogram bins.

    Returns
    -------
    matplotlib Figure
        A figure with two axes sharing the :math:`\theta` (y) axis: persons, then items.

    Raises
    ------
    MissingDependencyError
        If matplotlib is not installed.
    SpecificationError
        As :func:`~flexpyirt.wright_map_table`, or ``n_bins`` that is not a positive integer.
    """
    _require_unidimensional(res, "plot_wright_map()")
    plt = _pyplot()
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise SpecificationError(f"n_bins must be a positive integer, got {n_bins!r}.")
    table = wright_map_table(res, data, group=group, item_names=item_names, method=method)
    figure, (left, right) = plt.subplots(
        1, 2, sharey=True, figsize=(8.0, 5.0), gridspec_kw={"width_ratios": [1, 2]}
    )
    persons, items = table.persons, table.items
    names = list(res._model.item_names)
    groups: list[str | None] = list(_group_names(res)) if res._model.is_multigroup else [None]
    finite = persons["theta"].to_numpy()
    edges = np.histogram_bin_edges(finite[np.isfinite(finite)], bins=n_bins)
    offsets = np.linspace(-0.15, 0.15, len(groups)) if len(groups) > 1 else [0.0]
    for g, offset in zip(groups, offsets, strict=True):
        p = persons if g is None else persons[persons["group"] == g]
        left.hist(
            p["theta"].to_numpy(),
            bins=edges,
            orientation="horizontal",
            histtype="step" if len(groups) > 1 else "bar",
            label=g,
        )
        it = items if g is None else items[items["group"] == g]
        x = np.asarray([names.index(i) for i in it["item"]], dtype=float) + offset
        right.scatter(x, it["location"].to_numpy(), marker="_", s=120, label=g)
    left.invert_xaxis()
    left.set_xlabel("persons")
    left.set_ylabel(_THETA_LABEL)
    right.set_xticks(range(len(names)), names, rotation=90)
    right.set_xlabel("items (Thurstonian thresholds)")
    if len(groups) > 1:
        right.legend(fontsize="small")
    return figure  # type: ignore[no-any-return]
