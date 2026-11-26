r"""Common-item screening for linking and item parameter drift (Python shell, DESIGN.md §20).

:func:`screen_common_items` checks the common items of a unidimensional linking (:func:`link`)
for item parameter drift with the robust z of Huynh & Meyer (2010), the displacement of the item
location or the RMSD between expected item score curves, removes the items that moved and links
again (iterative purification; Candell & Drasgow, 1988). :func:`stack_screening_tables` stacks the
results of several administrations screened against one base form.
"""

from __future__ import annotations

import html
import math
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
import numpy as np
import pandas as pd

from flexpyirt._linking import LinkMethod, transform_kernel_params
from flexpyirt._precision import check_x64
from flexpyirt._screening import expected_score_rmsd_jit, item_location_jit, robust_z
from flexpyirt.errors import ScreeningWarning, SpecificationError
from flexpyirt.items import ItemKernel
from flexpyirt.linking import (
    LinkResult,
    _extract_item_table,
    _fitted_grid,
    _resolve_common_items,
    _resolve_group,
    link,
)
from flexpyirt.result import _HTML_STYLE, FitResult, _frame_to_html_table

__all__ = [
    "CommonItemScreeningResult",
    "ScreeningRemoval",
    "ScreeningStatistic",
    "screen_common_items",
    "stack_screening_tables",
]

ScreeningStatistic = Literal["robust_z", "displacement", "rmsd"]
ScreeningRemoval = Literal["largest", "all"]

_STATISTICS: tuple[str, ...] = ("robust_z", "displacement", "rmsd")
_REMOVALS: tuple[str, ...] = ("largest", "all")
DEFAULT_THRESHOLDS: dict[str, float] = {"robust_z": 1.645, "displacement": 0.3, "rmsd": 0.1}
"""Default flagging thresholds per statistic (DESIGN.md §20)."""

_TABLE_COLUMNS: tuple[str, ...] = (
    "item",
    "target_item",
    "model",
    "a_source",
    "b_source",
    "a_target",
    "b_target",
    "log_ratio_a",
    "difference_b",
    "displacement",
    "robust_z_a",
    "robust_z_b",
    "rmsd",
    "statistic",
    "flagged",
    "kept",
    "iteration_removed",
)


@dataclass(frozen=True)
class _ItemPair:
    """One common item: names, model label, kernel and untransformed parameters."""

    source: str
    target: str
    model: str
    kernel: ItemKernel
    src_params: np.ndarray
    tgt_params: np.ndarray


def _screening_table(
    pairs: list[_ItemPair],
    kept: np.ndarray,
    a_const: float,
    b_const: float,
    nodes: np.ndarray,
    weights: np.ndarray,
    statistic: str,
    threshold: float,
    iteration_removed: list[int | None],
) -> pd.DataFrame:
    """Per-item screening quantities under the linking ``(a_const, b_const)`` (DESIGN.md §20)."""
    a_j = jnp.asarray(a_const, dtype=jnp.float64)
    b_j = jnp.asarray(b_const, dtype=jnp.float64)
    nodes_j = jnp.asarray(nodes, dtype=jnp.float64)
    weights_j = jnp.asarray(weights, dtype=jnp.float64)
    rows: list[dict[str, object]] = []
    for pair in pairs:
        src = jnp.asarray(pair.src_params, dtype=jnp.float64)
        tgt = jnp.asarray(pair.tgt_params, dtype=jnp.float64)
        src_star = transform_kernel_params(pair.kernel, src, a_j, b_j)
        a_src = float(src_star[0])
        a_tgt = float(tgt[0])
        b_src = float(item_location_jit(pair.kernel, src_star))
        b_tgt = float(item_location_jit(pair.kernel, tgt))
        log_ratio = (
            math.log(abs(a_src)) - math.log(abs(a_tgt)) if a_src != 0.0 and a_tgt != 0.0 else np.nan
        )
        rmsd = float(expected_score_rmsd_jit(pair.kernel, src_star, tgt, nodes_j, weights_j))
        rows.append(
            {
                "item": pair.source,
                "target_item": pair.target,
                "model": pair.model,
                "a_source": a_src,
                "b_source": b_src,
                "a_target": a_tgt,
                "b_target": b_tgt,
                "log_ratio_a": log_ratio,
                "difference_b": b_src - b_tgt,
                "displacement": abs(b_src - b_tgt),
                "rmsd": rmsd,
            }
        )
    table = pd.DataFrame(rows)
    table["robust_z_a"] = robust_z(table["log_ratio_a"].to_numpy(dtype=float))
    table["robust_z_b"] = robust_z(table["difference_b"].to_numpy(dtype=float))
    if statistic == "robust_z":
        abs_z = np.abs(table[["robust_z_a", "robust_z_b"]].to_numpy(dtype=float))
        all_nan = np.all(np.isnan(abs_z), axis=1)
        filled = np.where(np.isnan(abs_z), -np.inf, abs_z)
        stat = np.where(all_nan, np.nan, np.max(filled, axis=1))
    else:
        stat = table[statistic].to_numpy(dtype=float)
    table["statistic"] = stat
    table["flagged"] = np.nan_to_num(stat, nan=-np.inf) > threshold
    table["kept"] = kept
    table["iteration_removed"] = pd.array(iteration_removed, dtype="Int64")
    return table.loc[:, list(_TABLE_COLUMNS)]


@dataclass(frozen=True)
class CommonItemScreeningResult:
    r"""Result of :func:`flexpyirt.screen_common_items` (DESIGN.md §20).

    Attributes
    ----------
    table : pandas.DataFrame
        One row per common item under the final linking: ``item`` (source name), ``target_item``,
        ``model``, ``a_source`` / ``b_source`` (source slope and location after the
        transformation), ``a_target`` / ``b_target``, ``log_ratio_a``
        (:math:`\log|a^*| - \log|a^{\mathrm{tgt}}|`), ``difference_b``
        (:math:`b^* - b^{\mathrm{tgt}}`),
        ``displacement`` (:math:`|b^* - b^{\mathrm{tgt}}|`), ``robust_z_a``, ``robust_z_b``
        (against the median and IQR of all common items), ``rmsd``, ``statistic`` (the value
        compared with ``threshold``), ``flagged``, ``kept`` and ``iteration_removed`` (nullable
        integer; the purification step that removed the item).
    link : LinkResult
        Linking on the kept items.
    initial_link : LinkResult
        Linking on all common items (step 0).
    kept_items : tuple of str
        Source names of the items kept in the anchor set.
    removed_items : tuple of str
        Source names of the removed items, in removal order.
    history : pandas.DataFrame
        One row per linking step: ``iteration`` (0 = all common items), ``n_items``, ``A``, ``B``,
        ``criterion``, ``converged`` and ``removed`` (tuple of the items removed after that
        step's statistics).
    statistic : {"robust_z", "displacement", "rmsd"}
        Screening statistic.
    threshold : float
        Flagging threshold.
    removal : {"largest", "all"}
        Removal rule.
    purify : bool
        Whether flagged items were removed.
    """

    table: pd.DataFrame
    link: LinkResult
    initial_link: LinkResult
    kept_items: tuple[str, ...]
    removed_items: tuple[str, ...]
    history: pd.DataFrame
    statistic: ScreeningStatistic
    threshold: float
    removal: ScreeningRemoval
    purify: bool

    @property
    def flagged_items(self) -> tuple[str, ...]:
        """Source names of the items flagged under the final linking (kept or removed)."""
        return tuple(self.table.loc[self.table["flagged"], "item"])

    def summary(self, *, digits: int = 4) -> str:
        """Return a readable plain-text summary of the screening.

        Parameters
        ----------
        digits : int, default 4
            Number of decimal places for floating-point values.

        Returns
        -------
        str
        """
        if isinstance(digits, bool) or not isinstance(digits, int) or digits < 0:
            raise SpecificationError(f"digits must be a non-negative integer, got {digits!r}.")

        def _fmt(v: float) -> str:
            return f"{v:.{digits}f}"

        history = self.history.assign(removed=self.history["removed"].map(", ".join))
        lines = [
            f"flexpyirt CommonItemScreeningResult ({self.statistic} > {self.threshold:g})",
            f"  Common items:  {len(self.table)} ({len(self.kept_items)} kept, "
            f"{len(self.removed_items)} removed)",
            f"  Removed:       {', '.join(self.removed_items) or '(none)'}",
            f"  Final linking: A={self.link.A:.{digits}f}, B={self.link.B:.{digits}f} "
            f"(initial A={self.initial_link.A:.{digits}f}, B={self.initial_link.B:.{digits}f})",
            "",
            "Purification steps:",
            history.to_string(index=False, float_format=_fmt),
            "",
            "Common items (final linking):",
            self.table.to_string(index=False, float_format=_fmt, na_rep="NaN"),
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        """Concise one-line summary."""
        return (
            f"CommonItemScreeningResult(statistic={self.statistic!r}, "
            f"threshold={self.threshold:g}, "
            f"kept={len(self.kept_items)}, removed={list(self.removed_items)}, "
            f"A={self.link.A:.4f}, B={self.link.B:.4f})"
        )

    def _repr_html_(self) -> str:
        """HTML summary for Jupyter notebooks."""
        meta = [
            ("Statistic", f"{self.statistic} > {self.threshold:g}"),
            (
                "Common items",
                f"{len(self.table)} ({len(self.kept_items)} kept, "
                f"{len(self.removed_items)} removed)",
            ),
            ("Removed", ", ".join(self.removed_items) or "(none)"),
            ("Final linking", f"A={self.link.A:.4f}, B={self.link.B:.4f}"),
        ]
        meta_rows = "".join(
            f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in meta
        )
        history = self.history.assign(removed=self.history["removed"].map(", ".join))
        # The HTML renderer formats numeric columns as floats; pandas' NA is not one.
        table = self.table.assign(
            iteration_removed=self.table["iteration_removed"].astype("float64")
        )
        return (
            f"{_HTML_STYLE}"
            '<div class="flexpyirt-result">'
            '<div class="flexpyirt-header">flexpyirt CommonItemScreeningResult</div>'
            f'<table class="flexpyirt-meta"><tbody>{meta_rows}</tbody></table>'
            "<h4>Purification steps</h4>"
            f"{_frame_to_html_table(history, digits=4)}"
            "<h4>Common items (final linking)</h4>"
            f"{_frame_to_html_table(table, digits=4)}"
            "</div>"
        )


def screen_common_items(
    source: FitResult,
    target: FitResult,
    *,
    items: Iterable[str] | Mapping[str, str] | None = None,
    method: LinkMethod = "stocking_lord",
    symmetric: bool = False,
    statistic: ScreeningStatistic = "robust_z",
    threshold: float | None = None,
    purify: bool = True,
    removal: ScreeningRemoval = "largest",
    min_items: int = 2,
    max_iter: int = 10,
    source_group: str | None = None,
    target_group: str | None = None,
) -> CommonItemScreeningResult:
    r"""Screen the common items of a linking for drift and relink without the items that moved.

    Each step links ``source`` onto ``target`` with :func:`link` on the current anchor set,
    transforms the source items (:math:`a^* = a/A`, location :math:`b^* = A\,b + B`) and computes
    per common item (DESIGN.md §20):

    - ``robust_z`` (Huynh & Meyer, 2010): for :math:`x_j = \log|a_j^*| - \log|a_j^{\mathrm{tgt}}|`
      and for :math:`x_j = b_j^* - b_j^{\mathrm{tgt}}`,
      :math:`z_j = (x_j - \operatorname{median} x)
      / (0.74\,\mathrm{IQR}\, x)`
      over all common items; the statistic is :math:`\max(|z_a|, |z_b|)`.
    - ``displacement``: :math:`|b_j^* - b_j^{\mathrm{tgt}}|` in logits.
    - ``rmsd``: :math:`\{\sum_q w_q [E_j^*(\theta_q) - E_j^{\mathrm{tgt}}(\theta_q)]^2\}^{1/2}`,
      the root mean squared difference of the expected item score curves
      :math:`E_j(\theta) = \sum_k k\,P_{jk}(\theta)` on the target group's fitted grid.

    The location :math:`b_j` is the display difficulty for dichotomous items, the mean of the
    display thresholds for GRM and GPCM / PCM, and :math:`-d_{K-1}/((K-1)a)` for nominal items.
    Items whose statistic exceeds ``threshold`` are flagged. With ``purify=True`` the flagged
    item with the largest statistic (``removal="largest"``) or every flagged item
    (``removal="all"``) is removed and the linking is repeated, until no kept item is flagged
    (iterative purification; Candell & Drasgow, 1988).

    Validated against base-R recomputations of the same rules on mirt calibrations (VALIDATION.md
    V50).

    Parameters
    ----------
    source : FitResult
        Fitted unidimensional model on the source (new) scale.
    target : FitResult
        Fitted unidimensional model on the target (base) scale.
    items : iterable of str or mapping of str to str, optional
        Common items, as in :func:`link`. Default: all items with the same name in both fits.
    method : {"stocking_lord", "haebara"}, default "stocking_lord"
        Linking criterion passed to :func:`link`.
    symmetric : bool, default False
        Symmetric linking criterion, passed to :func:`link`.
    statistic : {"robust_z", "displacement", "rmsd"}, default "robust_z"
        Screening statistic.
    threshold : float, optional
        Flagging threshold; defaults to 1.645 (``robust_z``), 0.3 (``displacement``) or 0.1
        (``rmsd``).
    purify : bool, default True
        If ``True``, remove flagged items and relink; if ``False``, only report the statistics
        under the linking on all common items.
    removal : {"largest", "all"}, default "largest"
        Remove the flagged item with the largest statistic per step (ties: first in item order),
        or every flagged item.
    min_items : int, default 2
        Smallest anchor set a removal may leave.
    max_iter : int, default 10
        Maximum number of removal steps.
    source_group, target_group : str, optional
        Group names when ``source`` / ``target`` are multiple-group fits.

    Returns
    -------
    CommonItemScreeningResult
        The per-item ``table``, the final and initial :class:`LinkResult`, the kept and removed
        items and the step ``history``.

    Raises
    ------
    PrecisionError
        If JAX is not configured for 64-bit floating point.
    SpecificationError
        If an input is not a unidimensional :class:`FitResult`, an option is invalid, or
        :func:`link` rejects the common items.

    Warns
    -----
    ScreeningWarning
        If purification stops while kept items are still flagged (``max_iter`` reached or
        ``min_items`` would be undercut).
    """
    check_x64()
    for role, res in (("source", source), ("target", target)):
        if not isinstance(res, FitResult):
            raise SpecificationError(f"{role} must be a FitResult, got {type(res).__name__}.")
        if res._model.n_factors != 1 or res._model.is_reduced:
            raise SpecificationError(
                "screen_common_items() supports unidimensional models only; "
                f"{role} has {res._model.n_factors} factors."
            )
    if statistic not in _STATISTICS:
        raise SpecificationError(f"Unknown statistic={statistic!r}; use one of {_STATISTICS}.")
    if removal not in _REMOVALS:
        raise SpecificationError(f"Unknown removal={removal!r}; use one of {_REMOVALS}.")
    if threshold is None:
        threshold = DEFAULT_THRESHOLDS[statistic]
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not (math.isfinite(threshold) and threshold > 0.0)
    ):
        raise SpecificationError(f"threshold must be a positive finite number, got {threshold!r}.")
    if not isinstance(purify, bool):
        raise SpecificationError(f"purify must be a bool, got {purify!r}.")
    if isinstance(min_items, bool) or not isinstance(min_items, int) or min_items < 1:
        raise SpecificationError(f"min_items must be a positive integer, got {min_items!r}.")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 0:
        raise SpecificationError(f"max_iter must be a non-negative integer, got {max_iter!r}.")

    src_gname, _ = _resolve_group(source, source_group, "source")
    tgt_gname, tgt_gidx = _resolve_group(target, target_group, "target")
    name_pairs = _resolve_common_items(source, target, items)
    src_est, _ = _extract_item_table(source, src_gname)
    tgt_est, _ = _extract_item_table(target, tgt_gname)
    src_lookup = dict(source._model.items)
    pairs = [
        _ItemPair(
            source=s,
            target=t,
            model=src_lookup[s].name,
            kernel=source._model.item_kernel(src_lookup[s]),
            src_params=src_est[s],
            tgt_params=tgt_est[t],
        )
        for s, t in name_pairs
    ]
    nodes, weights = _fitted_grid(target, tgt_gname, tgt_gidx)

    def _link(kept_mask: np.ndarray) -> LinkResult:
        return link(
            source,
            target,
            items={p.source: p.target for p, k in zip(pairs, kept_mask, strict=True) if k},
            method=method,
            symmetric=symmetric,
            source_group=source_group,
            target_group=target_group,
        )

    kept = np.ones(len(pairs), dtype=bool)
    removed_at: list[int | None] = [None] * len(pairs)
    removed_order: list[str] = []
    initial = _link(kept)
    current = initial
    history: list[dict[str, object]] = []
    iteration = 0
    while True:
        table = _screening_table(
            pairs,
            kept,
            float(current.A),
            float(current.B),
            nodes,
            weights,
            statistic,
            float(threshold),
            removed_at,
        )
        step = {
            "iteration": iteration,
            "n_items": int(kept.sum()),
            "A": float(current.A),
            "B": float(current.B),
            "criterion": current.criterion,
            "converged": current.converged,
            "removed": (),
        }
        history.append(step)
        flagged_kept = np.flatnonzero(table["flagged"].to_numpy() & kept)
        if not purify or flagged_kept.size == 0:
            break
        if iteration >= max_iter:
            warnings.warn(
                f"screen_common_items() stopped after max_iter={max_iter} removal steps with "
                f"{flagged_kept.size} kept item(s) still flagged.",
                ScreeningWarning,
                stacklevel=2,
            )
            break
        stat = table["statistic"].to_numpy(dtype=float)
        # Decreasing statistic; the stable sort keeps item order among ties.
        order = flagged_kept[np.argsort(-stat[flagged_kept], kind="stable")]
        candidates = order[:1] if removal == "largest" else order
        n_removable = int(kept.sum()) - min_items
        if n_removable <= 0:
            warnings.warn(
                f"screen_common_items() stopped with {flagged_kept.size} kept item(s) flagged: "
                f"removing more would leave fewer than min_items={min_items} anchor items.",
                ScreeningWarning,
                stacklevel=2,
            )
            break
        to_remove = candidates[:n_removable]
        iteration += 1
        for idx in to_remove:
            kept[idx] = False
            removed_at[idx] = iteration
            removed_order.append(pairs[idx].source)
        step["removed"] = tuple(pairs[idx].source for idx in to_remove)
        current = _link(kept)

    history_df = pd.DataFrame(
        history, columns=["iteration", "n_items", "A", "B", "criterion", "converged", "removed"]
    )
    return CommonItemScreeningResult(
        table=table,
        link=current,
        initial_link=initial,
        kept_items=tuple(p.source for p, k in zip(pairs, kept, strict=True) if k),
        removed_items=tuple(removed_order),
        history=history_df,
        statistic=statistic,
        threshold=float(threshold),
        removal=removal,
        purify=purify,
    )


def stack_screening_tables(results: Mapping[str, CommonItemScreeningResult]) -> pd.DataFrame:
    """Stack the tables of several administrations screened against one base form.

    Parameters
    ----------
    results : mapping of str to CommonItemScreeningResult
        ``{administration: result}``, e.g. each administration's fit screened against the base
        administration with :func:`screen_common_items`.

    Returns
    -------
    pandas.DataFrame
        The ``table`` of every result, in mapping order, with an ``administration`` column first
        and the final linking constants in ``A`` and ``B``.

    Raises
    ------
    SpecificationError
        If ``results`` is empty or holds something other than screening results.
    """
    if not isinstance(results, Mapping) or not results:
        raise SpecificationError(
            "results must be a non-empty {administration: CommonItemScreeningResult} mapping."
        )
    frames = []
    for name, res in results.items():
        if not isinstance(res, CommonItemScreeningResult):
            raise SpecificationError(
                f"results[{name!r}] must be a CommonItemScreeningResult, got {type(res).__name__}."
            )
        frame = res.table.copy()
        frame.insert(0, "administration", str(name))
        frame.insert(4, "A", float(res.link.A))
        frame.insert(5, "B", float(res.link.B))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
