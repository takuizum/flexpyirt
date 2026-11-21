"""Compile layer: Spec + data -> ParameterTable -> CompiledProblem (DESIGN.md §2 ②, §5, §14).

All validation happens here, before anything is jit-compiled, with messages naming the offending
group / item / parameter.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
import pandas as pd

from flexpyirt._core import CompiledProblem, FamilyBlock, LatentBlock, PriorBlock
from flexpyirt.errors import FlexpyirtWarning, IdentificationError, SpecificationError
from flexpyirt.items import FourPLKernel, GradedKernel, ItemKernel
from flexpyirt.latent import EmpiricalHistogram, Normal, eh_project_floor
from flexpyirt.model import LATENT_ITEM
from flexpyirt.priors import PRIOR_TYPES, Beta, LogNormal, NormalPrior, Prior, encode_prior
from flexpyirt.quadrature import Quadrature

if TYPE_CHECKING:
    from flexpyirt.model import Model

__all__ = [
    "LATENT_ITEM",
    "TABLE_COLUMNS",
    "Compiled",
    "Prepared",
    "build_table",
    "coerce_groups",
    "coerce_responses",
    "compile_problem",
    "eh_param_count",
    "eh_standardization_flags",
    "model_parameter_schema",
    "prepare",
    "validate_table",
    "validate_table_constraints",
]

LATENT_PARAMS = ("mean", "var")
TABLE_COLUMNS = (
    "group",
    "item",
    "model",
    "param",
    "value",
    "free",
    "lower",
    "upper",
    "prior",
    "free_id",
)


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------


def coerce_responses(
    data: Any,
    model: Model,
    *,
    item_names: Sequence[str] | None = None,
) -> np.ndarray:
    """Return an ``(N, J)`` int array ordered like the model's items, missing coded ``-1``.

    Accepts a pandas DataFrame, a polars DataFrame (via ``to_pandas``), or a 2-D array-like with
    ``item_names``. Responses are integers ``0..n_cat-1``; NaN is missing (MAR, DESIGN.md §14.8).

    Raises
    ------
    SpecificationError
        Missing or extra item columns, or out-of-range / non-integer values.
    """
    items = model.item_names
    if hasattr(data, "to_pandas") and not isinstance(data, pd.DataFrame):
        data = data.to_pandas()
    if isinstance(data, pd.DataFrame):
        columns = [str(c) for c in data.columns]
        frame = data
    else:
        arr = np.asarray(data, dtype=float)
        if arr.ndim != 2:
            raise SpecificationError(f"Response data must be 2-D, got shape {arr.shape}.")
        if item_names is None:
            if arr.shape[1] != len(items):
                raise SpecificationError(
                    f"Response array has {arr.shape[1]} columns but the model has {len(items)} "
                    "items; pass item_names= to map columns to items."
                )
            item_names = list(items)
        if len(item_names) != arr.shape[1]:
            raise SpecificationError(
                f"item_names has {len(item_names)} entries but the data has {arr.shape[1]} columns."
            )
        columns = [str(c) for c in item_names]
        frame = pd.DataFrame(arr, columns=columns)

    if len(set(columns)) != len(columns):
        raise SpecificationError("Response data has duplicated column names.")
    missing = [i for i in items if i not in columns]
    if missing:
        raise SpecificationError(f"Items in the model but not in the data: {missing}.")
    extra = [c for c in columns if c not in set(items)]
    if extra:
        raise SpecificationError(
            f"Columns in the data but not in the model: {extra}. Drop them before fitting."
        )

    values = frame[list(items)].to_numpy(dtype=float)
    if values.shape[0] == 0:
        raise SpecificationError("Response data has no rows.")
    out = np.full(values.shape, -1, dtype=np.int32)
    for j, (name, item_model) in enumerate(model.items):
        col = values[:, j]
        obs = ~np.isnan(col)
        k = item_model.kernel.n_cat
        ok = np.isin(col[obs], np.arange(k, dtype=float))
        if not ok.all():
            bad = sorted(set(col[obs][~ok].tolist()))[:5]
            raise SpecificationError(
                f"Item {name!r} ({item_model.name}) must have values 0..{k - 1} or NaN; "
                f"found {bad}."
            )
        out[obs, j] = col[obs].astype(np.int32)
    return out


def coerce_groups(
    group: Any, model: Model, n: int, *, require_all: bool = True
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return integer group codes (``(N,)``) and group names in the model's order.

    ``require_all`` (estimation) demands that every declared group appears in the data.
    """
    names = tuple(name for name, _ in model.resolved_groups())
    if not model.is_multigroup:
        if group is not None:
            raise SpecificationError(
                "group= was given but the model has no groups; declare them with with_groups()."
            )
        return np.zeros(n, dtype=np.int32), names
    if group is None:
        raise SpecificationError(f"The model has groups {list(names)}; pass group= labels.")
    labels = np.asarray(group.to_numpy() if hasattr(group, "to_numpy") else group).astype(str)
    if labels.shape != (n,):
        raise SpecificationError(f"group= must have one label per row ({n}), got {labels.shape}.")
    unknown = sorted(set(labels.tolist()) - set(names))
    if unknown:
        raise SpecificationError(f"group= has labels not declared in with_groups(): {unknown}.")
    absent = [g for g in names if g not in set(labels.tolist())]
    if absent and require_all:
        raise SpecificationError(f"Groups declared but absent from the data: {absent}.")
    lookup = {g: i for i, g in enumerate(names)}
    return np.array([lookup[v] for v in labels], dtype=np.int32), names


def compress_patterns(
    y: np.ndarray, group: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return unique (group, pattern) rows, their groups, counts and each row's index (§4.1)."""
    keyed = np.column_stack([group, y])
    uniq, inverse, counts = np.unique(keyed, axis=0, return_inverse=True, return_counts=True)
    return uniq[:, 1:], uniq[:, 0], counts.astype(float), inverse.reshape(-1)


def pad_patterns(
    patterns: np.ndarray, groups: np.ndarray, counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad to the next power-of-two bucket (at least 8) with zero-count rows (DESIGN.md §4.1)."""
    n = patterns.shape[0]
    bucket = max(8, 1 << (n - 1).bit_length())
    pad = bucket - n
    return (
        np.vstack([patterns, np.zeros((pad, patterns.shape[1]), dtype=patterns.dtype)]),
        np.concatenate([groups, np.zeros(pad, dtype=groups.dtype)]),
        np.concatenate([counts, np.zeros(pad)]),
    )


# --------------------------------------------------------------------------------------------
# Parameter table
# --------------------------------------------------------------------------------------------


def _start_values(model: Model, y: np.ndarray) -> dict[str, np.ndarray]:
    obs = y >= 0
    scores = np.where(obs, y, 0)
    total = scores.sum(axis=1)
    out = {}
    for j, (name, item_model) in enumerate(model.items):
        rows = obs[:, j]
        if rows.sum() < 2:
            # Not estimable (reported later by _check_estimable); finite placeholder values (the
            # item model's starts for p = 0.5) keep the table valid when the user fixes this item.
            out[name] = item_model.start_values(np.array([0.0, 1.0]), np.zeros(2))
            continue
        item = y[rows, j].astype(float)
        rest = (total - scores[:, j])[rows].astype(float)
        out[name] = item_model.start_values(item, rest)
    return out


def _prior_center(pri: Prior) -> float | None:
    """Return a representative interior point for ``pri``, or ``None`` if it overflows."""
    if isinstance(pri, NormalPrior):
        return pri.mean
    if isinstance(pri, LogNormal):
        try:
            return math.exp(pri.meanlog)
        except OverflowError:
            return None
    if isinstance(pri, Beta):
        return pri.a / (pri.a + pri.b)
    return None


def _interior_point(lo: float, hi: float, candidate: float | None = None) -> float:
    """Return a finite point strictly inside ``(lo, hi)`` (assuming ``lo < hi``)."""
    if candidate is not None and math.isfinite(candidate) and lo < candidate < hi:
        return candidate
    if math.isfinite(lo) and math.isfinite(hi):
        return 0.5 * (lo + hi)
    if math.isfinite(lo):
        return lo + 1.0
    if math.isfinite(hi):
        return hi - 1.0
    return 0.0


def _order_grm_intercepts(
    d: np.ndarray,
    is_free: Sequence[bool],
    eff_lo: Sequence[float],
    eff_hi: Sequence[float],
) -> np.ndarray:
    """Ensure GRM intercepts are strictly decreasing inside each free parameter's open bounds."""
    m = len(d)
    if m < 2 or all(float(d[j]) > float(d[j + 1]) for j in range(m - 1)):
        return d
    lb = [eff_lo[j] if is_free[j] else float(d[j]) for j in range(m)]
    ub = [eff_hi[j] if is_free[j] else float(d[j]) for j in range(m)]
    for j in range(m - 2, -1, -1):
        if is_free[j]:
            lb[j] = max(lb[j], lb[j + 1])
    for j in range(1, m):
        if is_free[j]:
            ub[j] = min(ub[j], ub[j - 1])
    if any(is_free[j] and not (lb[j] < ub[j]) for j in range(m)):
        return d
    out = d.copy()
    j = 0
    while j < m:
        if not is_free[j]:
            j += 1
            continue
        s = j
        while j < m and is_free[j]:
            j += 1
        e = j - 1
        for idx in range(s, e + 1):
            rem = e - idx + 1
            lo = lb[idx]
            hi = ub[idx] if idx == 0 else min(ub[idx], float(out[idx - 1]))
            if math.isfinite(lo) and math.isfinite(hi):
                out[idx] = lo + (hi - lo) * (rem / (rem + 1.0))
            elif math.isfinite(lo):
                out[idx] = lo + 0.5 * rem
            elif math.isfinite(hi):
                out[idx] = hi - 0.5
            else:
                out[idx] = 0.5 * (rem - 0.5 * (e - s + 2))
    return out


def build_table(model: Model, y: np.ndarray) -> pd.DataFrame:
    """Build the default ParameterTable (DESIGN.md §5.1, §14.1) from the spec and responses."""
    rows: list[dict[str, Any]] = []
    starts = _start_values(model, y)
    fixed = {(item, param): value for item, param, value in model.fixed}
    priors = {(item, param): dist for item, param, dist in model.priors}
    for group_name, latent in model.resolved_groups():
        for name, item_model in model.items:
            lower, upper = item_model.bounds()
            free_default = item_model.default_free()
            param_names = item_model.kernel.param_names
            item_starts = starts[name].copy()
            is_free = [
                free_default[k] and (name, param) not in fixed
                for k, param in enumerate(param_names)
            ]
            eff_lo = [float(v) for v in lower]
            eff_hi = [float(v) for v in upper]
            for k, param in enumerate(param_names):
                if not is_free[k]:
                    item_starts[k] = fixed.get((name, param), float(item_starts[k]))
                    continue
                pri = priors.get((name, param))
                if pri is not None:
                    p_lo, p_hi = pri.support
                    eff_lo[k] = max(eff_lo[k], p_lo)
                    eff_hi[k] = min(eff_hi[k], p_hi)
                lo_k, hi_k = eff_lo[k], eff_hi[k]
                if lo_k < hi_k and not (lo_k < float(item_starts[k]) < hi_k):
                    candidate = _prior_center(pri) if pri is not None else None
                    item_starts[k] = _interior_point(lo_k, hi_k, candidate)
            if isinstance(item_model.kernel, FourPLKernel):
                g_free = is_free[2]
                u_free = is_free[3]
                if g_free and not u_free:
                    hi_g = min(eff_hi[2], float(item_starts[3]))
                    if eff_lo[2] < hi_g and not (eff_lo[2] < float(item_starts[2]) < hi_g):
                        item_starts[2] = _interior_point(eff_lo[2], hi_g)
                elif u_free and not g_free:
                    lo_u = max(eff_lo[3], float(item_starts[2]))
                    if lo_u < eff_hi[3] and not (lo_u < float(item_starts[3]) < eff_hi[3]):
                        item_starts[3] = _interior_point(lo_u, eff_hi[3])
                elif g_free and u_free and not (float(item_starts[2]) < float(item_starts[3])):
                    ov_lo = max(eff_lo[2], eff_lo[3])
                    ov_hi = min(eff_hi[2], eff_hi[3])
                    if ov_lo < ov_hi and math.isfinite(ov_lo) and math.isfinite(ov_hi):
                        item_starts[2] = ov_lo + (ov_hi - ov_lo) / 3.0
                        item_starts[3] = ov_lo + 2.0 * (ov_hi - ov_lo) / 3.0
            elif isinstance(item_model.kernel, GradedKernel):
                item_starts[1:] = _order_grm_intercepts(
                    item_starts[1:], is_free[1:], eff_lo[1:], eff_hi[1:]
                )
            for k, param in enumerate(param_names):
                key = (name, param)
                rows.append(
                    {
                        "group": group_name,
                        "item": name,
                        "model": item_model.name,
                        "param": param,
                        "value": float(item_starts[k]),
                        "free": is_free[k],
                        "lower": float(lower[k]),
                        "upper": float(upper[k]),
                        "prior": priors.get(key),
                    }
                )
        if isinstance(latent, Normal):
            latent_rows: tuple[tuple[str, float, bool, float], ...] = (
                ("mean", latent.mean, latent.free_mean, -np.inf),
                ("var", latent.var, latent.free_var, 0.0),
            )
            for param, value, free, lat_lower in latent_rows:
                rows.append(
                    {
                        "group": group_name,
                        "item": LATENT_ITEM,
                        "model": "Normal",
                        "param": param,
                        "value": float(value),
                        "free": bool(free),
                        "lower": lat_lower,
                        "upper": np.inf,
                        "prior": None,
                    }
                )
    table = pd.DataFrame(rows)
    table["prior"] = table["prior"].astype(object)
    ids = np.full(len(table), -1)
    free_mask = table["free"].to_numpy(dtype=bool)
    ids[free_mask] = np.arange(int(free_mask.sum()))
    # Across-group equalities share the free_id of the first group's row.
    item_col = table["item"].to_numpy(dtype=object)
    param_col = table["param"].to_numpy(dtype=object)
    for items, params in model.equal:
        for item in items:
            mask = (item_col == item) & free_mask
            if params is not None:
                mask &= np.isin(param_col, list(params))
            for param in set(param_col[mask].tolist()):
                rows_p = np.flatnonzero(mask & (param_col == param))
                ids[rows_p] = ids[rows_p[0]]
    table["free_id"] = ids
    return table[list(TABLE_COLUMNS)].astype({"free": bool, "free_id": int})


def model_parameter_schema(model: Model) -> pd.DataFrame:
    """Return canonical ``(group, item, param, model, lower, upper)`` bounds for ``model``."""
    rows: list[tuple[str, str, str, str, float, float]] = []
    for group_name, latent in model.resolved_groups():
        for name, item_model in model.items:
            lower, upper = item_model.bounds()
            for param, lo, hi in zip(item_model.kernel.param_names, lower, upper, strict=True):
                rows.append((group_name, name, param, item_model.name, float(lo), float(hi)))
        if isinstance(latent, Normal):
            rows.append((group_name, LATENT_ITEM, "mean", "Normal", -np.inf, np.inf))
            rows.append((group_name, LATENT_ITEM, "var", "Normal", 0.0, np.inf))
    return pd.DataFrame(rows, columns=["group", "item", "param", "model", "lower", "upper"])


def _validate_value_column(
    t: pd.DataFrame,
    model: Model,
    *,
    value_col: str,
    free_label: str,
    fixed_label: str,
) -> None:
    """Validate finiteness, bounds, prior support, and model ordering for one value column."""
    group_col = t["group"].to_numpy(dtype=object)
    item_col = t["item"].to_numpy(dtype=object)
    param_col = t["param"].to_numpy(dtype=object)
    value = t[value_col].to_numpy(dtype=float)
    lower = t["lower"].to_numpy(dtype=float)
    upper = t["upper"].to_numpy(dtype=float)
    free = t["free"].to_numpy(dtype=bool)
    prior_col = t["prior"].to_numpy(dtype=object)
    for i in range(len(t)):
        where = f"(group={group_col[i]!r}, item={item_col[i]!r}, param={param_col[i]!r})"
        if not np.isfinite(value[i]):
            raise SpecificationError(f"Non-finite {value_col} {value[i]} at {where}.")
        if free[i] and not (lower[i] < value[i] < upper[i]):
            raise SpecificationError(
                f"{free_label} {value[i]} is not strictly inside ({lower[i]}, {upper[i]}) "
                f"at {where}."
            )
        if not free[i] and not (lower[i] <= value[i] <= upper[i]):
            raise SpecificationError(
                f"{fixed_label} {value[i]} is outside [{lower[i]}, {upper[i]}] at {where}."
            )
        if item_col[i] == LATENT_ITEM and param_col[i] == "var" and not value[i] > 0:
            raise SpecificationError(f"Latent variance must be positive at {where}.")
        pri = prior_col[i]
        if pri is not None and free[i]:
            p_lo, p_hi = pri.support
            if not (p_lo < value[i] < p_hi):
                raise SpecificationError(
                    f"{free_label} {value[i]} is outside the support ({p_lo}, {p_hi}) "
                    f"of {pri!r} at {where}."
                )

    # Asymptote ordering g < u for 3PL / 4PL items.
    fourpl_items = {name for name, m in model.items if isinstance(m.kernel, FourPLKernel)}
    if fourpl_items:
        sub = t[t["item"].isin(fourpl_items) & t["param"].isin(["g", "u"])]
        for (grp, itm), rows in sub.groupby(["group", "item"], sort=False):
            vals = dict(zip(rows["param"], rows[value_col], strict=True))
            if not vals["g"] < vals["u"]:
                raise SpecificationError(
                    f"Lower asymptote g ({vals['g']}) must be < upper asymptote u ({vals['u']}) "
                    f"at (group={grp!r}, item={itm!r})."
                )

    # Intercept ordering d1 > d2 > ... > d_{K-1} for GRM items.
    grm_items = {
        name: m.kernel.intercept_params
        for name, m in model.items
        if isinstance(m.kernel, GradedKernel) and m.kernel.n_cat > 2
    }
    if grm_items:
        sub = t[t["item"].isin(grm_items)]
        for (grp, itm), rows in sub.groupby(["group", "item"], sort=False):
            vals = dict(zip(rows["param"], rows[value_col], strict=True))
            intercepts = grm_items[str(itm)]
            ds = [float(vals[p]) for p in intercepts]
            if not all(ds[k] > ds[k + 1] for k in range(len(ds) - 1)):
                raise SpecificationError(
                    f"GRM intercepts ({value_col}) must satisfy "
                    f"{' > '.join(intercepts)} at (group={grp!r}, item={itm!r}), got {ds}."
                )


def _validate_shared_free_ids(t: pd.DataFrame) -> None:
    """Verify rows sharing a ``free_id`` match in item, param, prior, and ``(lower, upper)``."""
    free_rows = t[t["free"]]
    shared = free_rows.groupby("free_id")["item"].nunique()
    if (shared > 1).any():
        fid = int(shared[shared > 1].index[0])
        items = sorted(t.loc[t["free_id"] == fid, "item"].unique())
        raise SpecificationError(
            f"free_id {fid} is shared by items {items}. Equality constraints across items are "
            "not supported yet (planned for M5)."
        )
    within = free_rows.groupby("free_id")["param"].nunique()
    if (within > 1).any():
        fid = int(within[within > 1].index[0])
        raise SpecificationError(f"free_id {fid} is shared by different parameters of one item.")
    for fid_val, rows_f in free_rows.groupby("free_id"):
        priors_f = list(rows_f["prior"])
        first_p = priors_f[0]
        if any(p != first_p for p in priors_f[1:]):
            raise SpecificationError(
                f"Rows sharing free_id {fid_val} have conflicting priors: {priors_f}."
            )
        lowers_f = rows_f["lower"].to_numpy(dtype=float)
        uppers_f = rows_f["upper"].to_numpy(dtype=float)
        if np.any(lowers_f != lowers_f[0]) or np.any(uppers_f != uppers_f[0]):
            raise SpecificationError(f"Rows sharing free_id {fid_val} have conflicting bounds.")


def validate_table_constraints(
    table: pd.DataFrame,
    reference: pd.DataFrame,
    model: Model,
    *,
    check_free_ids: bool = True,
    extra_value_cols: Sequence[tuple[str, str, str]] = (),
) -> pd.DataFrame:
    """Reorder ``table`` to canonical ``(group, item, param)`` order and check domain invariants."""
    keys = ["group", "item", "param"]
    orig_cols = list(table.columns)
    t = table.set_index(keys).loc[reference.set_index(keys).index].reset_index()[orig_cols]

    group_col = t["group"].to_numpy(dtype=object)
    item_col = t["item"].to_numpy(dtype=object)
    param_col = t["param"].to_numpy(dtype=object)
    lower = t["lower"].to_numpy(dtype=float)
    upper = t["upper"].to_numpy(dtype=float)
    ref_lower = reference["lower"].to_numpy(dtype=float)
    ref_upper = reference["upper"].to_numpy(dtype=float)
    free = t["free"].to_numpy(dtype=bool)
    prior_col = t["prior"].to_numpy(dtype=object)
    for i in range(len(t)):
        where = f"(group={group_col[i]!r}, item={item_col[i]!r}, param={param_col[i]!r})"
        if not lower[i] < upper[i]:
            raise SpecificationError(f"lower must be < upper at {where}.")
        if lower[i] < ref_lower[i] or upper[i] > ref_upper[i]:
            raise SpecificationError(
                f"Bounds [{lower[i]}, {upper[i]}] exceed model domain "
                f"[{ref_lower[i]}, {ref_upper[i]}] at {where}."
            )
        pri = prior_col[i]
        if pri is not None:
            if item_col[i] == LATENT_ITEM:
                raise SpecificationError(
                    f"Priors on latent-distribution parameters are not supported at {where}."
                )
            if not isinstance(pri, PRIOR_TYPES):
                raise SpecificationError(
                    f"Invalid prior {pri!r} at {where}; use NormalPrior, LogNormal, Beta or None."
                )
            if free[i]:
                p_lo, p_hi = pri.support
                if not (max(lower[i], p_lo) < min(upper[i], p_hi)):
                    raise SpecificationError(
                        f"Bounds [{lower[i]}, {upper[i]}] do not overlap the support "
                        f"({p_lo}, {p_hi}) of {pri!r} at {where}."
                    )

    _validate_value_column(
        t, model, value_col="value", free_label="Starting value", fixed_label="Fixed value"
    )
    for col, free_label, fixed_label in extra_value_cols:
        _validate_value_column(
            t, model, value_col=col, free_label=free_label, fixed_label=fixed_label
        )
    if check_free_ids:
        _validate_shared_free_ids(t)
    return t


def validate_table(
    table: pd.DataFrame, reference: pd.DataFrame, model: Model, y: np.ndarray, group: np.ndarray
) -> pd.DataFrame:
    """Validate a (possibly user-edited) table against the default one and normalize ``free_id``.

    Free rows with ``free_id < 0`` get fresh ids; ids are renumbered to ``0..n-1`` keeping any
    sharing. Sharing an id across different items is not supported before M5 (DESIGN.md §4.3).
    Also checks estimability from the data and identification (§4.5, §14.5).
    """
    missing_cols = [c for c in TABLE_COLUMNS if c not in table.columns]
    if missing_cols:
        raise SpecificationError(f"Parameter table is missing columns {missing_cols}.")
    t = table[list(TABLE_COLUMNS)].copy().reset_index(drop=True)
    keys = ["group", "item", "param"]
    if t.duplicated(keys).any():
        dup = t.loc[t.duplicated(keys, keep=False), keys].to_dict("records")
        raise SpecificationError(f"Parameter table has duplicated rows: {dup}.")
    got = set(map(tuple, t[keys].to_numpy().tolist()))
    want = set(map(tuple, reference[keys].to_numpy().tolist()))
    if got != want:
        raise SpecificationError(
            "Parameter table rows do not match the model: "
            f"unexpected {sorted(got - want)}, missing {sorted(want - got)}. "
            "Edit values, free flags, bounds or free_id only."
        )
    t["free"] = t["free"].astype(bool)
    t["prior"] = t["prior"].astype(object)
    for col in ("value", "lower", "upper"):
        t[col] = t[col].astype(float)
    t = validate_table_constraints(t, reference, model, check_free_ids=False)
    t["model"] = reference["model"].to_numpy()

    free = t["free"].to_numpy(dtype=bool)
    ids = np.where(free, t["free_id"].fillna(-1).to_numpy(dtype=int), -1)
    next_id = int(ids.max()) + 1 if (ids >= 0).any() else 0
    for i in np.flatnonzero(free & (ids < 0)).tolist():
        ids[i] = next_id
        next_id += 1
    _, ids_free = np.unique(ids[ids >= 0], return_inverse=True)
    ids[ids >= 0] = ids_free
    t["free_id"] = ids.astype(int)

    _validate_shared_free_ids(t)
    _check_estimable(t, model, y, group)
    _check_identification(t, model)
    return t


def _check_estimable(t: pd.DataFrame, model: Model, y: np.ndarray, group: np.ndarray) -> None:
    """Each free item parameter needs every category observed in the groups that use it."""
    names = [g for g, _ in model.resolved_groups()]
    col = {name: j for j, name in enumerate(model.item_names)}
    n_cat = {name: m.kernel.n_cat for name, m in model.items}
    free_items = t[t["free"] & (t["item"] != LATENT_ITEM)]
    for (item, _fid), rows in free_items.groupby(["item", "free_id"]):
        codes = [names.index(g) for g in rows["group"].unique()]
        resp = y[np.isin(group, codes), col[str(item)]]
        seen = set(np.unique(resp[resp >= 0]).tolist())
        absent = [k for k in range(n_cat[str(item)]) if k not in seen]
        if absent:
            where = (
                "" if not model.is_multigroup else f" in groups {sorted(rows['group'].unique())}"
            )
            raise SpecificationError(
                f"Item {item!r}: categories {absent} are never observed{where}, so its parameters "
                "are not estimable. Collapse categories, fix the item, or constrain it."
            )


def eh_standardization_flags(t: pd.DataFrame, model: Model) -> tuple[bool, bool]:
    """Return ``(standardize_mean, standardize_var)`` for an EH model (DESIGN.md §4.4, §15.29).

    The reference group's mean is standardized iff no intercept parameter is fixed in its connected
    component; its variance is standardized iff no nonzero slope parameter is fixed in its
    connected component. Under FIPC (fixed nonzero slopes and fixed intercepts), both are False.
    """
    if not model.is_eh:
        return False, False
    lookup = dict(model.items)
    groups = [g for g, _ in model.resolved_groups()]
    ref = groups[0]
    items = t[t["item"] != LATENT_ITEM]
    role = np.array(
        [
            "slope"
            if p in lookup[i].kernel.slope_params
            else ("intercept" if p in lookup[i].kernel.intercept_params else "other")
            for i, p in zip(items["item"], items["param"], strict=True)
        ]
    )

    flags: dict[str, bool] = {}
    for key, kind in (("var", "slope"), ("mean", "intercept")):
        rows = items[role == kind]
        parent = {g: g for g in groups}

        def find(g: str, parent: dict[str, str] = parent) -> str:
            while parent[g] != g:
                g = parent[g]
            return g

        for _, shared in rows[rows["free"]].groupby("free_id"):
            gs = list(shared["group"].unique())
            for other in gs[1:]:
                parent[find(other)] = find(gs[0])
        fixed_rows = rows[~rows["free"]]
        if kind == "slope":
            fixed_rows = fixed_rows[fixed_rows["value"] != 0]
        fixed_roots = {find(g) for g in fixed_rows["group"]}
        flags[key] = find(ref) not in fixed_roots
    return flags["mean"], flags["var"]


def eh_param_count(t: pd.DataFrame, model: Model, quad: Quadrature) -> int:
    """Return the number of free EH latent-weight parameters for AIC/BIC (DESIGN.md §4.8, §15.30).

    Reference group: ``Q - 1 - int(standardize_mean) - int(standardize_var)`` (``Q - 3`` when both
    mean and variance are standardized, ``Q - 2`` for Rasch/PCM, ``Q - 1`` under FIPC). Every
    focal group: ``Q - 1``. Returns ``0`` for ``Normal`` models.
    """
    if not model.is_eh:
        return 0
    std_mean, std_var = eh_standardization_flags(t, model)
    n_groups = len(model.resolved_groups())
    q = quad.n_points
    ref_params = q - 1 - int(std_mean) - int(std_var)
    focal_params = (n_groups - 1) * (q - 1)
    return ref_params + focal_params


def _check_identification(t: pd.DataFrame, model: Model) -> None:
    """Scale identification rules of DESIGN.md §4.5 and §14.5 (necessary conditions)."""
    lookup = dict(model.items)
    groups = [g for g, _ in model.resolved_groups()]
    items = t[t["item"] != LATENT_ITEM]
    role = np.array(
        [
            "slope"
            if p in lookup[i].kernel.slope_params
            else ("intercept" if p in lookup[i].kernel.intercept_params else "other")
            for i, p in zip(items["item"], items["param"], strict=True)
        ]
    )
    if model.is_eh:
        std_mean, std_var = eh_standardization_flags(t, model)
        lat_free: dict[tuple[str, str], bool] = {}
        for idx, g in enumerate(groups):
            if idx == 0:
                lat_free[(g, "mean")] = not std_mean
                lat_free[(g, "var")] = not std_var
            else:
                lat_free[(g, "mean")] = True
                lat_free[(g, "var")] = True
    else:
        lat_free = {
            (g, p): bool(f)
            for g, i, p, f in zip(t["group"], t["item"], t["param"], t["free"], strict=True)
            if i == LATENT_ITEM
        }

    for latent_param, kind, what in (("var", "slope", "scale"), ("mean", "intercept", "location")):
        rows = items[role == kind]
        parent = {g: g for g in groups}

        def find(g: str, parent: dict[str, str] = parent) -> str:
            while parent[g] != g:
                g = parent[g]
            return g

        for _, shared in rows[rows["free"]].groupby("free_id"):
            gs = list(shared["group"].unique())
            for other in gs[1:]:
                parent[find(other)] = find(gs[0])
        fixed_rows = rows[~rows["free"]]
        if kind == "slope":
            fixed_rows = fixed_rows[fixed_rows["value"] != 0]
        anchored = {find(g) for g in fixed_rows["group"]}
        anchored |= {find(g) for g in groups if not lat_free[(g, latent_param)]}
        for g in groups:
            if lat_free[(g, latent_param)] and find(g) not in anchored:
                where = "" if not model.is_multigroup else f" of group {g!r}"
                fix_hint = (
                    "fix at least one slope (e.g. Rasch or PCM items)"
                    if kind == "slope"
                    else "fix at least one intercept"
                )
                raise IdentificationError(
                    f"The latent {'variance' if latent_param == 'var' else 'mean'}{where} is free "
                    f"but nothing links it to a fixed {what}. Fix it, {fix_hint}, or make items "
                    "equal across groups (anchors)."
                )
        if model.is_multigroup and latent_param == "mean":
            n_groups_per_id = rows[rows["free"]].groupby("free_id")["group"].nunique()
            for g in groups:
                if not lat_free[(g, "mean")]:
                    continue
                own = rows[rows["group"] == g]
                linked = own.loc[own["free"], "free_id"]
                n_anchor = int((n_groups_per_id.reindex(linked).fillna(0) > 1).sum())
                n_anchor += int((~own["free"]).sum())
                if n_anchor == 1:
                    warnings.warn(
                        f"The latent mean of group {g!r} is linked by a single anchor parameter; "
                        "estimates may be unstable. Consider more anchor items.",
                        FlexpyirtWarning,
                        stacklevel=4,
                    )


# --------------------------------------------------------------------------------------------
# Problem
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Prepared:
    """Coerced data and the validated table."""

    y: np.ndarray
    group: np.ndarray
    group_names: tuple[str, ...]
    table: pd.DataFrame


def prepare(
    model: Model,
    data: Any,
    *,
    group: Any,
    item_names: Sequence[str] | None,
    table: pd.DataFrame | None,
) -> Prepared:
    """Coerce data and build/validate the parameter table."""
    y = coerce_responses(data, model, item_names=item_names)
    codes, names = coerce_groups(group, model, y.shape[0])
    default = build_table(model, y)
    final = validate_table(default if table is None else table, default, model, y, codes)
    return Prepared(y=y, group=codes, group_names=names, table=final)


@dataclass(frozen=True)
class Compiled:
    """Compiled problem plus what the Result layer needs to map back."""

    problem: CompiledProblem
    eta0: np.ndarray
    inverse: np.ndarray
    n_patterns: int


def compile_problem(
    model: Model,
    y: np.ndarray,
    group: np.ndarray,
    table: pd.DataFrame,
    quad: Quadrature,
    *,
    eh_weights: np.ndarray | None = None,
) -> Compiled:
    """Turn a validated table and responses into a :class:`CompiledProblem` (§5.2, §14.1–14.2)."""
    patterns, pat_groups, counts, inverse = compress_patterns(y, group)
    n_patterns = patterns.shape[0]
    patterns, pat_groups, counts = pad_patterns(patterns, pat_groups, counts)
    n_free = int(table["free_id"].max()) + 1 if table["free"].any() else 0
    eta0 = np.zeros(n_free)
    prior_kind = np.zeros(n_free, dtype=np.int32)
    prior_p1 = np.zeros(n_free, dtype=float)
    prior_p2 = np.ones(n_free, dtype=float)
    free_rows = table[table["free"]].drop_duplicates("free_id")
    free_ids = free_rows["free_id"].to_numpy(dtype=int)
    eta0[free_ids] = free_rows["value"].to_numpy(dtype=float)
    for fid_val, pri in zip(free_ids.tolist(), free_rows["prior"].tolist(), strict=True):
        k_code, p1, p2 = encode_prior(pri)
        prior_kind[fid_val] = k_code
        prior_p1[fid_val] = p1
        prior_p2[fid_val] = p2

    groups = [g for g, _ in model.resolved_groups()]
    n_groups = len(groups)
    lookup: dict[tuple[str, str, str], tuple[float, int, float, float, Prior | None]] = {
        (g, i, p): (v, fid, lo, hi, pri)
        for g, i, p, v, fid, lo, hi, pri in zip(
            table["group"],
            table["item"],
            table["param"],
            table["value"].to_numpy(dtype=float),
            table["free_id"].to_numpy(dtype=int),
            table["lower"].to_numpy(dtype=float),
            table["upper"].to_numpy(dtype=float),
            table["prior"].tolist(),
            strict=True,
        )
    }

    kernels: list[ItemKernel] = []
    members: dict[ItemKernel, list[int]] = {}
    for j, (_, item_model) in enumerate(model.items):
        k = item_model.kernel
        if k not in members:
            kernels.append(k)
            members[k] = []
        members[k].append(j)

    families = []
    for k in kernels:
        cols = members[k]
        names = [model.items[j][0] for j in cols]
        params = list(k.param_names)
        shape = (n_groups, len(cols), len(params))
        fixed = np.zeros(shape)
        free_index = np.full(shape, -1, dtype=np.int32)
        local_index = np.full(shape, -1, dtype=np.int32)
        slots: list[list[tuple[int, float, float]]] = []
        for jj, item in enumerate(names):
            seen: dict[int, int] = {}
            item_slots: list[tuple[int, float, float]] = []
            for gg, g in enumerate(groups):
                for pp, p in enumerate(params):
                    v, fid, lo, hi, pri = lookup[(g, item, p)]
                    fixed[gg, jj, pp] = v
                    free_index[gg, jj, pp] = fid
                    if fid >= 0:
                        if fid not in seen:
                            seen[fid] = len(item_slots)
                            if pri is not None:
                                p_lo, p_hi = pri.support
                                lo = max(lo, p_lo)
                                hi = min(hi, p_hi)
                            item_slots.append((fid, lo, hi))
                        local_index[gg, jj, pp] = seen[fid]
            slots.append(item_slots)
        n_slots = max(1, max(len(s) for s in slots))
        slot_free = np.full((len(cols), n_slots), -1, dtype=np.int32)
        slot_lo = np.full((len(cols), n_slots), -np.inf)
        slot_hi = np.full((len(cols), n_slots), np.inf)
        for jj, item_slots in enumerate(slots):
            for ss, (fid, lo, hi) in enumerate(item_slots):
                slot_free[jj, ss], slot_lo[jj, ss], slot_hi[jj, ss] = fid, lo, hi
        families.append(
            FamilyBlock(
                columns=jnp.asarray(cols, dtype=jnp.int32),
                fixed_values=jnp.asarray(fixed, dtype=jnp.float64),
                free_index=jnp.asarray(free_index),
                local_index=jnp.asarray(local_index),
                slot_free_index=jnp.asarray(slot_free),
                slot_lower=jnp.asarray(slot_lo, dtype=jnp.float64),
                slot_upper=jnp.asarray(slot_hi, dtype=jnp.float64),
                kernel=k,
            )
        )

    nodes = jnp.asarray(quad.nodes(), dtype=jnp.float64)
    resolved = model.resolved_groups()
    if model.is_eh:
        min_w_list: list[float] = []
        for g_name, dist in resolved:
            assert isinstance(dist, EmpiricalHistogram)
            mw = float(dist.min_weight)
            if mw * quad.n_points >= 1.0:
                raise SpecificationError(
                    f"EmpiricalHistogram min_weight ({mw}) for group {g_name!r} must satisfy "
                    f"min_weight * n_points < 1 (n_points={quad.n_points})."
                )
            min_w_list.append(mw)
        min_w_arr = jnp.asarray(min_w_list, dtype=jnp.float64)
        if eh_weights is not None:
            w_arr = np.asarray(eh_weights, dtype=np.float64)
            if w_arr.shape != (n_groups, quad.n_points):
                raise SpecificationError(
                    f"eh_weights must have shape ({n_groups}, {quad.n_points}), got {w_arr.shape}."
                )
            eh_w = jnp.asarray(w_arr, dtype=jnp.float64)
        else:
            base = jnp.exp(-0.5 * nodes**2)
            eh_w = jnp.stack([eh_project_floor(base, min_w_arr[g]) for g in range(n_groups)])
        std_mean, std_var = eh_standardization_flags(table, model)
        latent = LatentBlock(
            fixed_values=jnp.zeros((n_groups, 2), dtype=jnp.float64),
            free_index=jnp.full((n_groups, 2), -1, dtype=jnp.int32),
            lower=jnp.full((n_groups, 2), -jnp.inf, dtype=jnp.float64),
            upper=jnp.full((n_groups, 2), jnp.inf, dtype=jnp.float64),
            eh_weights=eh_w,
            eh_min_weight=min_w_arr,
            is_eh=True,
            standardize_mean=std_mean,
            standardize_var=std_var,
        )
    else:
        lat_vals = np.array(
            [[lookup[(g, LATENT_ITEM, p)][:4] for p in LATENT_PARAMS] for g in groups], dtype=float
        )
        latent = LatentBlock(
            fixed_values=jnp.asarray(lat_vals[:, :, 0], dtype=jnp.float64),
            free_index=jnp.asarray(lat_vals[:, :, 1].astype(np.int32)),
            lower=jnp.asarray(lat_vals[:, :, 2], dtype=jnp.float64),
            upper=jnp.asarray(lat_vals[:, :, 3], dtype=jnp.float64),
            eh_weights=jnp.ones((n_groups, 1), dtype=jnp.float64),
            eh_min_weight=jnp.zeros((n_groups,), dtype=jnp.float64),
            is_eh=False,
            standardize_mean=False,
            standardize_var=False,
        )
    priors_block = PriorBlock(
        kind=jnp.asarray(prior_kind, dtype=jnp.int32),
        param1=jnp.asarray(prior_p1, dtype=jnp.float64),
        param2=jnp.asarray(prior_p2, dtype=jnp.float64),
    )
    problem = CompiledProblem(
        y=jnp.asarray(patterns, dtype=jnp.int32),
        counts=jnp.asarray(counts, dtype=jnp.float64),
        group=jnp.asarray(pat_groups, dtype=jnp.int32),
        nodes=nodes,
        families=tuple(families),
        latent=latent,
        priors=priors_block,
        n_groups=n_groups,
    )
    return Compiled(problem=problem, eta0=eta0, inverse=inverse, n_patterns=n_patterns)
