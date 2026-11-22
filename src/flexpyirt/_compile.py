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

from flexpyirt._core import (
    CompiledProblem,
    CoupledBlock,
    CoupledMember,
    FamilyBlock,
    LatentBlock,
    PriorBlock,
)
from flexpyirt.errors import FlexpyirtWarning, IdentificationError, SpecificationError
from flexpyirt.items import (
    GPCM,
    GRM,
    PCM,
    FourPL,
    FourPLKernel,
    GradedKernel,
    ItemKernel,
    ItemModel,
    Nominal,
    Rasch,
    ThreePL,
    _asymptote_starts,
    _gpcm_intercept_starts,
    _intercept_start,
    _slope_start,
)
from flexpyirt.latent import (
    DavidianCurve,
    EmpiricalHistogram,
    Normal,
    correlation_matrix_is_pd,
    davidian_param_names,
    eh_project_floor,
    latent_param_names,
)
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


def _non_slope_starts(item_model: ItemModel, item: np.ndarray, a_eff: float) -> list[float]:
    """Return classical starting values for non-slope parameters given effective slope ``a_eff``."""
    p = float(item.mean())
    if isinstance(item_model, (ThreePL, FourPL)):
        g, u = _asymptote_starts(p, free_upper=isinstance(item_model, FourPL))
        return [_intercept_start((p - g) / (u - g), a_eff), g, u]
    if isinstance(item_model, GRM):
        cum = [float((item >= k).mean()) for k in range(1, item_model.n_cat)]
        d = [_intercept_start(prob, a_eff) for prob in cum]
        for k in range(1, len(d)):
            d[k] = min(d[k], d[k - 1] - 0.05)
        return d
    if isinstance(item_model, (GPCM, PCM)):
        return _gpcm_intercept_starts(item, item_model.n_cat, a_eff)
    if isinstance(item_model, Nominal):
        return [
            *(float(k) for k in range(1, item_model.n_cat - 1)),
            *_gpcm_intercept_starts(item, item_model.n_cat, a_eff),
        ]
    return [_intercept_start(p, a_eff)]


def _start_values(model: Model, y: np.ndarray) -> dict[str, np.ndarray]:
    obs = y >= 0
    scores = np.where(obs, y, 0)
    if not model.is_multidimensional:
        total = scores.sum(axis=1)
        out = {}
        for j, (name, item_model) in enumerate(model.items):
            rows = obs[:, j]
            if rows.sum() < 2:
                # Not estimable (reported later by _check_estimable); finite placeholder values
                # (starts for p = 0.5) keep the table valid when the user fixes this item.
                out[name] = item_model.start_values(np.array([0.0, 1.0]), np.zeros(2))
                continue
            item = y[rows, j].astype(float)
            rest = (total - scores[:, j])[rows].astype(float)
            out[name] = item_model.start_values(item, rest)
        return out

    # Multidimensional confirmatory starts (DESIGN.md §4.10, §15.31): positive slopes on loaded
    # factors (from factor-specific item-rest correlations, scaled by 1/sqrt(n_loaded)) and 0 on
    # unloaded factors; intercepts/asymptotes use the effective slope norm ||a||.
    n_factors = model.n_factors
    col_of = {name: j for j, (name, _) in enumerate(model.items)}
    assert model.factors is not None
    factor_totals = [
        scores[:, [col_of[item_name] for item_name in assigned]].sum(axis=1)
        for _, assigned in model.factors
    ]
    out = {}
    for j, (name, item_model) in enumerate(model.items):
        rows = obs[:, j]
        loaded = model.item_factors(name)
        slopes = np.zeros(n_factors, dtype=float)
        if rows.sum() < 2:
            placeholder = np.array([0.0, 1.0])
            for m in loaded:
                slopes[m] = (
                    1.0 if isinstance(item_model, (Rasch, PCM)) else 1.0 / math.sqrt(len(loaded))
                )
            a_eff = max(float(np.linalg.norm(slopes)), 0.25)
            out[name] = np.array([*slopes, *_non_slope_starts(item_model, placeholder, a_eff)])
            continue
        item = y[rows, j].astype(float)
        for m in loaded:
            if isinstance(item_model, (Rasch, PCM)):
                slopes[m] = 1.0
            else:
                rest_m = (factor_totals[m] - scores[:, j])[rows].astype(float)
                slopes[m] = float(
                    np.clip(_slope_start(item, rest_m) / math.sqrt(len(loaded)), 0.25, 3.0)
                )
        a_eff = max(float(np.linalg.norm(slopes)), 0.25)
        out[name] = np.array([*slopes, *_non_slope_starts(item_model, item, a_eff)])
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


def _davidian_phi_starts(degree: int, y: np.ndarray) -> list[float]:
    """Return small skewness-aligned starting angles for ``DavidianCurve`` (DESIGN.md §16.4).

    ``phi = 0`` is a stationary point of the standardized likelihood (the scores in ``phi_1`` and
    ``phi_2`` vanish there), so the start is moved off it. The sign is chosen so that the
    standardized start density has a third standardized moment with the same sign as the
    total-score skewness: to leading order ``+4 phi_1**3`` for ``degree == 1`` (``+0.0039`` at
    ``phi_1 = 0.1``) and ``-8.3 phi_1 phi_2`` for ``degree >= 2`` (``+0.083`` at
    ``(phi_1, phi_2) = (-0.1, 0.1)``).
    """
    obs = y >= 0
    has_obs = obs.any(axis=1)
    sign = 1.0
    if int(has_obs.sum()) >= 3:
        totals = np.where(obs, y, 0).sum(axis=1)[has_obs].astype(float)
        sd = float(np.std(totals))
        if sd > 0.0:
            skew = float(np.mean(((totals - float(np.mean(totals))) / sd) ** 3))
            sign = 1.0 if skew >= 0.0 else -1.0
    if degree == 1:
        return [0.1 * sign]
    return [-0.1 * sign, 0.1, *([0.0] * (degree - 2))]


def build_table(model: Model, y: np.ndarray) -> pd.DataFrame:
    """Build the default ParameterTable (DESIGN.md §5.1, §14.1) from the spec and responses."""
    rows: list[dict[str, Any]] = []
    starts = _start_values(model, y)
    fixed = {(item, param): value for item, param, value in model.fixed}
    priors = {(item, param): dist for item, param, dist in model.priors}
    n_factors = model.n_factors
    all_fixed_slope = all(isinstance(m, (Rasch, PCM)) for _, m in model.items)
    for group_name, latent in model.resolved_groups():
        for name, item_model in model.items:
            kernel = model.item_kernel(item_model)
            base_lo, base_hi = item_model.bounds()
            base_free = item_model.default_free()
            lower = (base_lo[0],) * n_factors + base_lo[1:]
            upper = (base_hi[0],) * n_factors + base_hi[1:]
            loaded_set = set(model.item_factors(name))
            free_default = (
                tuple(m in loaded_set and base_free[0] for m in range(n_factors)) + base_free[1:]
            )
            param_names = kernel.param_names
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
            if isinstance(kernel, FourPLKernel):
                g_idx = n_factors + 1
                u_idx = n_factors + 2
                g_free = is_free[g_idx]
                u_free = is_free[u_idx]
                if g_free and not u_free:
                    hi_g = min(eff_hi[g_idx], float(item_starts[u_idx]))
                    if eff_lo[g_idx] < hi_g and not (
                        eff_lo[g_idx] < float(item_starts[g_idx]) < hi_g
                    ):
                        item_starts[g_idx] = _interior_point(eff_lo[g_idx], hi_g)
                elif u_free and not g_free:
                    lo_u = max(eff_lo[u_idx], float(item_starts[g_idx]))
                    if lo_u < eff_hi[u_idx] and not (
                        lo_u < float(item_starts[u_idx]) < eff_hi[u_idx]
                    ):
                        item_starts[u_idx] = _interior_point(lo_u, eff_hi[u_idx])
                elif (
                    g_free
                    and u_free
                    and not (float(item_starts[g_idx]) < float(item_starts[u_idx]))
                ):
                    ov_lo = max(eff_lo[g_idx], eff_lo[u_idx])
                    ov_hi = min(eff_hi[g_idx], eff_hi[u_idx])
                    if ov_lo < ov_hi and math.isfinite(ov_lo) and math.isfinite(ov_hi):
                        item_starts[g_idx] = ov_lo + (ov_hi - ov_lo) / 3.0
                        item_starts[u_idx] = ov_lo + 2.0 * (ov_hi - ov_lo) / 3.0
            elif isinstance(kernel, GradedKernel):
                item_starts[n_factors:] = _order_grm_intercepts(
                    item_starts[n_factors:],
                    is_free[n_factors:],
                    eff_lo[n_factors:],
                    eff_hi[n_factors:],
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
        if isinstance(latent, EmpiricalHistogram):
            continue
        if isinstance(latent, DavidianCurve):
            phi_starts = _davidian_phi_starts(latent.degree, y)
            latent_rows: list[tuple[str, float, bool, float, float]] = [
                ("mean", latent.mean, latent.free_mean, -np.inf, np.inf),
                ("var", latent.var, latent.free_var, 0.0, np.inf),
                *(
                    (f"phi_{j}", phi_starts[j - 1], True, -np.inf, np.inf)
                    for j in range(1, latent.degree + 1)
                ),
            ]
            lat_model = "Davidian"
        elif not model.is_multidimensional:
            latent_rows = [
                ("mean", latent.mean, latent.free_mean, -np.inf, np.inf),
                ("var", latent.var, latent.free_var, 0.0, np.inf),
            ]
            lat_model = "Normal"
        else:
            # Single group: means 0 and variances 1 (free for all-Rasch/PCM). Multiple groups:
            # each group's Normal spec applies to every factor (DESIGN.md §15.43).
            if model.is_multigroup:
                assert isinstance(latent, Normal)
                mean0, free_mean = latent.mean, latent.free_mean
                var0, free_var = latent.var, latent.free_var
            else:
                mean0, free_mean, var0, free_var = 0.0, False, 1.0, all_fixed_slope
            latent_rows = [
                *(
                    (f"mean_{m}", mean0, free_mean, -np.inf, np.inf)
                    for m in range(1, n_factors + 1)
                ),
                *((f"var_{m}", var0, free_var, 0.0, np.inf) for m in range(1, n_factors + 1)),
                *(
                    (f"cor_{i}{j}", 0.0, model.correlated_factors, -1.0, 1.0)
                    for i in range(1, n_factors + 1)
                    for j in range(i + 1, n_factors + 1)
                ),
            ]
            lat_model = "MVN"
        for param, value, free, lat_lower, lat_upper in latent_rows:
            rows.append(
                {
                    "group": group_name,
                    "item": LATENT_ITEM,
                    "model": lat_model,
                    "param": param,
                    "value": float(value),
                    "free": bool(free),
                    "lower": lat_lower,
                    "upper": lat_upper,
                    "prior": None,
                }
            )
    table = pd.DataFrame(rows)
    table["prior"] = table["prior"].astype(object)
    free_mask = table["free"].to_numpy(dtype=bool)
    item_col = table["item"].to_numpy(dtype=object)
    param_col = table["param"].to_numpy(dtype=object)
    group_col = table["group"].to_numpy(dtype=object)
    # Equalities (across groups, §14.1, and across items, §16.1) merge free rows by union-find.
    parent = np.arange(len(table))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    def union(rows_eq: np.ndarray) -> None:
        if rows_eq.size < 2:
            return
        root = find(int(rows_eq[0]))
        for r in rows_eq[1:].tolist():
            other = find(int(r))
            if other != root:
                parent[max(root, other)] = min(root, other)
                root = min(root, other)

    for items, params in model.equal:
        for item in items:
            mask = (item_col == item) & free_mask
            if params is not None:
                mask &= np.isin(param_col, list(params))
            for param in set(param_col[mask].tolist()):
                union(np.flatnonzero(mask & (param_col == param)))
    for items, params in model.equal_items:
        in_items = np.isin(item_col, list(items)) & free_mask
        for group_name in dict.fromkeys(group_col.tolist()):
            for param in params:
                union(np.flatnonzero(in_items & (group_col == group_name) & (param_col == param)))

    roots = np.array([find(i) for i in range(len(table))])
    # Number the merged rows in order of first appearance (default ids follow the row order).
    ids = np.full(len(table), -1)
    order: dict[int, int] = {}
    for i in np.flatnonzero(free_mask).tolist():
        ids[i] = order.setdefault(int(roots[i]), len(order))
    # A parameter shared across items starts at the mean of its rows' default starting values.
    if model.equal_items:
        values = table["value"].to_numpy(dtype=float).copy()
        for fid in range(len(order)):
            rows_f = np.flatnonzero(ids == fid)
            if np.unique(item_col[rows_f]).size > 1:
                values[rows_f] = float(np.mean(values[rows_f]))
        table["value"] = values
    table["free_id"] = ids
    return table[list(TABLE_COLUMNS)].astype({"free": bool, "free_id": int})


def model_parameter_schema(model: Model) -> pd.DataFrame:
    """Return canonical ``(group, item, param, model, lower, upper)`` bounds for ``model``."""
    rows: list[tuple[str, str, str, str, float, float]] = []
    for group_name, latent in model.resolved_groups():
        for name, item_model in model.items:
            lower, upper = item_model.bounds()
            n_slopes = model.n_factors
            # Multidimensional items repeat the slope bounds for a1..aD (as in build_table).
            lower = (lower[0],) * n_slopes + tuple(lower[1:])
            upper = (upper[0],) * n_slopes + tuple(upper[1:])
            names = model.item_kernel(item_model).param_names
            for param, lo, hi in zip(names, lower, upper, strict=True):
                rows.append((group_name, name, param, item_model.name, float(lo), float(hi)))
        if isinstance(latent, EmpiricalHistogram):
            continue
        if isinstance(latent, DavidianCurve):
            rows.append((group_name, LATENT_ITEM, "mean", "Davidian", -np.inf, np.inf))
            rows.append((group_name, LATENT_ITEM, "var", "Davidian", 0.0, np.inf))
            for j in range(1, latent.degree + 1):
                rows.append((group_name, LATENT_ITEM, f"phi_{j}", "Davidian", -np.inf, np.inf))
            continue
        if not model.is_multidimensional:
            rows.append((group_name, LATENT_ITEM, "mean", "Normal", -np.inf, np.inf))
            rows.append((group_name, LATENT_ITEM, "var", "Normal", 0.0, np.inf))
            continue
        for lat_param in latent_param_names(model.n_factors):
            if lat_param.startswith("var_"):
                lo_hi = (0.0, np.inf)
            elif lat_param.startswith("cor_"):
                lo_hi = (-1.0, 1.0)
            else:
                lo_hi = (-np.inf, np.inf)
            rows.append((group_name, LATENT_ITEM, lat_param, "MVN", *lo_hi))
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
        if (
            item_col[i] == LATENT_ITEM
            and (param_col[i] == "var" or str(param_col[i]).startswith("var_"))
            and not value[i] > 0
        ):
            raise SpecificationError(f"Latent variance must be positive at {where}.")
        if (
            item_col[i] == LATENT_ITEM
            and str(param_col[i]).startswith("cor_")
            and not (-1.0 < value[i] < 1.0)
        ):
            raise SpecificationError(
                f"Latent correlation must lie strictly in (-1, 1), got {value[i]} at {where}."
            )
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
    """Verify rows sharing a ``free_id`` agree in prior and ``(lower, upper)`` (DESIGN §16.1).

    A ``free_id`` may be shared across groups and across items (also of different item models),
    but within one item it maps to a single parameter. Latent-distribution parameters cannot share
    a ``free_id`` with item parameters.
    """
    free_rows = t[t["free"]]
    within = free_rows.groupby(["free_id", "item"])["param"].nunique()
    if (within > 1).any():
        fid, item = within[within > 1].index[0]
        raise SpecificationError(
            f"free_id {int(fid)} is shared by different parameters of item {item!r}."
        )
    is_latent = (free_rows["item"] == LATENT_ITEM).to_numpy(dtype=bool)
    fid_col = free_rows["free_id"].to_numpy(dtype=int)
    both = np.intersect1d(fid_col[is_latent], fid_col[~is_latent])
    if both.size:
        raise SpecificationError(
            f"free_id {int(both[0])} is shared by item and latent-distribution parameters."
        )
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
    sharing, also across items (DESIGN.md §16.1).
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
    # Positive-definite latent correlation matrix and nonzero starting slopes per factor (D > 1).
    if model.is_multidimensional:
        n_factors = model.n_factors
        cor_names = [
            f"cor_{i}{j}" for i in range(1, n_factors + 1) for j in range(i + 1, n_factors + 1)
        ]
        lat_sub = t[t["item"] == LATENT_ITEM]
        for grp_lat, rows_g in lat_sub.groupby("group", sort=False):
            vals_g = dict(zip(rows_g["param"], rows_g["value"], strict=True))
            cors = [float(vals_g[c]) for c in cor_names]
            if not correlation_matrix_is_pd(cors, n_factors):
                raise SpecificationError(
                    f"Latent correlation matrix is not positive definite for group {grp_lat!r}: "
                    f"{dict(zip(cor_names, cors, strict=True))}."
                )
        items_sub = t[t["item"] != LATENT_ITEM]
        for m, fname in enumerate(model.factor_names, start=1):
            rows_m = items_sub[items_sub["param"] == f"a{m}"]
            slope_vals = rows_m["value"].to_numpy(dtype=float)
            # A factor whose loadings are all fixed at 0 is an identification problem, reported
            # as IdentificationError by _check_identification_mirt; here only starting values.
            if rows_m["free"].any() and not np.any(slope_vals != 0.0):
                raise SpecificationError(
                    f"Factor {fname!r} (a{m}) has all starting slopes equal to 0; at least one "
                    "loading must have a nonzero starting value."
                )

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
    """Return ``(standardize_mean, standardize_var)`` for an EH model (DESIGN.md §4.4, §15.34).

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
    """Return the number of free EH latent-weight parameters for AIC/BIC (DESIGN.md §4.8, §15.35).

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
    """Scale identification rules of DESIGN.md §4.5, §14.5, and §15.31 (necessary conditions)."""
    if model.is_multidimensional:
        _check_identification_mirt(t, model)
        return
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


def _check_identification_mirt(t: pd.DataFrame, model: Model) -> None:
    """Identification checks for D > 1 (DESIGN §15.31, §15.43).

    Single group: the per-group checks below. Multiple groups: rotational checks within each
    group, and scale/location linking per factor across groups (:func:`_check_mirt_group_links`).
    """
    if not model.is_multigroup:
        _check_identification_mirt_group(t, model, check_scale=True)
        return
    for g, _ in model.resolved_groups():
        _check_identification_mirt_group(t[t["group"] == g], model, check_scale=False)
    _check_mirt_group_links(t, model)


def _check_mirt_group_links(t: pd.DataFrame, model: Model) -> None:
    """Per factor, every free ``var_m`` / ``mean_m`` must be linked to a fixed scale / location.

    For factor :math:`m`, groups are connected when they share a free slope ``a{m}`` (scale) or a
    free intercept of an item loading on :math:`m` (location) through a common ``free_id`` (anchor
    items). A component is anchored by a fixed nonzero ``a{m}`` (scale), a fixed intercept of an
    item loading on :math:`m` (location), or a group whose ``var_m`` / ``mean_m`` is fixed
    (DESIGN.md §4.5, §15.43).
    """
    groups = [g for g, _ in model.resolved_groups()]
    kernels = {name: model.item_kernel(m) for name, m in model.items}
    items_df = t[t["item"] != LATENT_ITEM]
    lat = t[t["item"] == LATENT_ITEM]
    lat_free = {
        (g, p): bool(f) for g, p, f in zip(lat["group"], lat["param"], lat["free"], strict=True)
    }
    for m, fname in enumerate(model.factor_names, start=1):
        slope = f"a{m}"
        s_rows = items_df[items_df["param"] == slope]
        active = s_rows[s_rows["free"] | (s_rows["value"] != 0.0)]
        active_pairs = set(zip(active["group"], active["item"], strict=True))
        int_mask = np.array(
            [
                (g, i) in active_pairs and p in kernels[i].intercept_params
                for g, i, p in zip(
                    items_df["group"], items_df["item"], items_df["param"], strict=True
                )
            ],
            dtype=bool,
        )
        for latent_param, rows, what in (
            (f"var_{m}", s_rows, "scale"),
            (f"mean_{m}", items_df[int_mask], "location"),
        ):
            parent = {g: g for g in groups}

            def find(g: str, parent: dict[str, str] = parent) -> str:
                while parent[g] != g:
                    g = parent[g]
                return g

            for _, shared in rows[rows["free"]].groupby("free_id"):
                gs = list(shared["group"].unique())
                for other in gs[1:]:
                    parent[find(other)] = find(gs[0])
            fixed = rows[~rows["free"]]
            if what == "scale":
                fixed = fixed[fixed["value"] != 0.0]
            anchored = {find(g) for g in fixed["group"]}
            anchored |= {find(g) for g in groups if not lat_free[(g, latent_param)]}
            for g in groups:
                if lat_free[(g, latent_param)] and find(g) not in anchored:
                    raise IdentificationError(
                        f"The latent {latent_param} of group {g!r} (factor {fname!r}) is free but "
                        f"nothing links it to a fixed {what}: make items loading on {fname!r} "
                        "equal across groups (anchors) or fix them."
                    )


def _check_identification_mirt_group(t: pd.DataFrame, model: Model, *, check_scale: bool) -> None:
    """Necessary scale, location, and rotational identification checks for D > 1 (DESIGN §15.31)."""
    n_factors = model.n_factors
    fnames = model.factor_names
    lookup = {name: model.item_kernel(m) for name, m in model.items}
    lat_rows = t[t["item"] == LATENT_ITEM]
    lat_free = dict(zip(lat_rows["param"], lat_rows["free"].astype(bool), strict=True))
    items_df = t[t["item"] != LATENT_ITEM]

    free_sets: list[set[str]] = []
    fixed_nonzero_sets: list[set[str]] = []
    n_fixed_slopes: list[int] = []
    fixed_vectors: list[np.ndarray] = []
    all_items = list(model.item_names)

    for m, fname in enumerate(fnames, start=1):
        slope = f"a{m}"
        s_rows = items_df[items_df["param"] == slope]
        free_items = set(s_rows.loc[s_rows["free"], "item"].tolist())
        fixed_nz = set(s_rows.loc[(~s_rows["free"]) & (s_rows["value"] != 0.0), "item"].tolist())
        active_items = free_items | fixed_nz
        if not active_items:
            raise IdentificationError(
                f"Factor {fname!r} ({slope}) has no active loadings (all slopes are fixed at 0)."
            )
        if check_scale and lat_free[f"var_{m}"] and not fixed_nz:
            raise IdentificationError(
                f"The latent variance of factor {fname!r} (var_{m}) is free, but no slope "
                f"{slope} is fixed at a nonzero value."
            )
        if check_scale and lat_free[f"mean_{m}"]:
            has_fixed_int = False
            for itm in active_items:
                ints = lookup[itm].intercept_params
                itm_ints = items_df[(items_df["item"] == itm) & (items_df["param"].isin(ints))]
                if (~itm_ints["free"]).any():
                    has_fixed_int = True
                    break
            if not has_fixed_int:
                raise IdentificationError(
                    f"The latent mean of factor {fname!r} (mean_{m}) is free, but no item "
                    "loading on it has a fixed intercept."
                )
        free_sets.append(free_items)
        fixed_nonzero_sets.append(fixed_nz)
        n_fixed_slopes.append(int((~s_rows["free"]).sum()))
        fixed_vals = dict(zip(s_rows["item"], s_rows["value"].astype(float), strict=True))
        fixed_vectors.append(np.array([fixed_vals.get(it, 0.0) for it in all_items]))

    # Necessary confirmatory rotational identification (Bollen, 1989; Reckase, 2009):
    # each factor involved in c_m free correlations needs at least c_m fixed slopes (typically 0s),
    # and no two factors may share an identical free-loading set with no fixed nonzero slopes.
    for m, fname in enumerate(fnames, start=1):
        free_cors = sum(
            1
            for other in range(1, n_factors + 1)
            if other != m and lat_free[f"cor_{min(m, other)}{max(m, other)}"]
        )
        if n_fixed_slopes[m - 1] < free_cors:
            raise IdentificationError(
                f"Factor {fname!r} (a{m}) has {free_cors} free correlation(s) but only "
                f"{n_fixed_slopes[m - 1]} fixed slope(s); confirmatory models need at least "
                f"{free_cors} fixed loading(s) per correlated factor to prevent rotational "
                "indeterminacy."
            )
    for i in range(n_factors):
        for j in range(i + 1, n_factors):
            if (
                free_sets[i] == free_sets[j]
                and not fixed_nonzero_sets[i]
                and not fixed_nonzero_sets[j]
            ):
                raise IdentificationError(
                    f"Factors {fnames[i]!r} and {fnames[j]!r} have identical free loading "
                    "patterns with no fixed nonzero slopes, which is rotationally unidentified."
                )
            # With no free slopes on either factor, proportional fixed loading vectors make the
            # loading matrix rank deficient (e.g. Rasch/PCM items loading on both factors):
            # only a combination of the two factors is identified.
            if not free_sets[i] and not free_sets[j]:
                vi, vj = fixed_vectors[i], fixed_vectors[j]
                if np.linalg.matrix_rank(np.vstack([vi, vj])) < 2:
                    raise IdentificationError(
                        f"Factors {fnames[i]!r} and {fnames[j]!r} have proportional fixed loadings "
                        "and no free loadings, so only a combination of them is identified."
                    )


# --------------------------------------------------------------------------------------------
# Problem
# --------------------------------------------------------------------------------------------


def _coupled_blocks(
    free_index: Sequence[np.ndarray],
    slot_bounds: dict[int, tuple[float, float]],
) -> tuple[CoupledBlock, ...]:
    """Components of items connected by ``free_id``s shared across items (DESIGN.md §16.1).

    ``free_index[f]`` is family ``f``'s ``(G, J_f, P_f)`` free-index array and ``slot_bounds``
    maps each ``free_id`` to its bounds (intersected with its prior's support). Components with a
    single item are left to the item-wise M-step.
    """
    item_keys = [(f, jj) for f, fi in enumerate(free_index) for jj in range(fi.shape[1])]
    parent = list(range(len(item_keys)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner: dict[int, int] = {}
    for idx, (f, jj) in enumerate(item_keys):
        for fid in np.unique(free_index[f][:, jj, :]).tolist():
            if fid < 0:
                continue
            if fid in owner:
                a, b = find(owner[fid]), find(idx)
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                owner[fid] = idx
    components: dict[int, list[int]] = {}
    for idx in range(len(item_keys)):
        components.setdefault(find(idx), []).append(idx)

    blocks: list[CoupledBlock] = []
    for members_idx in components.values():
        if len(members_idx) < 2:
            continue
        slot_of: dict[int, int] = {}
        by_family: dict[int, list[int]] = {}
        for idx in members_idx:
            f, jj = item_keys[idx]
            by_family.setdefault(f, []).append(jj)
            for fid in free_index[f][:, jj, :].ravel().tolist():
                if fid >= 0 and fid not in slot_of:
                    slot_of[fid] = len(slot_of)
        members = []
        for f, positions in by_family.items():
            sub = free_index[f][:, positions, :]
            local = np.array([slot_of.get(int(v), -1) for v in sub.ravel()], dtype=np.int32)
            local = local.reshape(sub.shape)
            members.append(
                CoupledMember(
                    positions=jnp.asarray(positions, dtype=jnp.int32),
                    local_index=jnp.asarray(local, dtype=jnp.int32),
                    family=f,
                )
            )
        fids = list(slot_of)
        blocks.append(
            CoupledBlock(
                slot_free_index=jnp.asarray(fids, dtype=jnp.int32),
                slot_lower=jnp.asarray([slot_bounds[v][0] for v in fids], dtype=jnp.float64),
                slot_upper=jnp.asarray([slot_bounds[v][1] for v in fids], dtype=jnp.float64),
                members=tuple(members),
            )
        )
    return tuple(blocks)


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
        k = model.item_kernel(item_model)
        if k not in members:
            kernels.append(k)
            members[k] = []
        members[k].append(j)

    families = []
    family_free_index: list[np.ndarray] = []
    slot_bounds: dict[int, tuple[float, float]] = {}
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
                            slot_bounds.setdefault(fid, (lo, hi))
                        local_index[gg, jj, pp] = seen[fid]
            slots.append(item_slots)
        n_slots = max(1, max(len(s) for s in slots))
        slot_free = np.full((len(cols), n_slots), -1, dtype=np.int32)
        slot_lo = np.full((len(cols), n_slots), -np.inf)
        slot_hi = np.full((len(cols), n_slots), np.inf)
        for jj, item_slots in enumerate(slots):
            for ss, (fid, lo, hi) in enumerate(item_slots):
                slot_free[jj, ss], slot_lo[jj, ss], slot_hi[jj, ss] = fid, lo, hi
        family_free_index.append(free_index)
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

    nodes = jnp.asarray(quad.grid(model.n_factors), dtype=jnp.float64)
    resolved = model.resolved_groups()
    if model.is_eh and quad.kind != "equispaced":
        raise SpecificationError(
            "EmpiricalHistogram needs an equispaced quadrature grid (its weights are histogram "
            "bins); Gauss-Hermite quadrature is for Normal latent distributions."
        )
    if model.is_davidian and quad.kind != "equispaced":
        raise SpecificationError(
            "DavidianCurve is validated on equispaced quadrature grids only; Gauss-Hermite "
            "quadrature is for Normal latent distributions."
        )
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
            base = jnp.exp(-0.5 * nodes[:, 0] ** 2)
            eh_w = jnp.stack([eh_project_floor(base, min_w_arr[g]) for g in range(n_groups)])
        std_mean, std_var = eh_standardization_flags(table, model)
        if std_var and not std_mean:
            raise IdentificationError(
                "EmpiricalHistogram with fixed intercepts but no fixed nonzero slope in the "
                "reference group's linked items is not supported: the variance-only constraint is "
                "not linear in the weights (DESIGN.md §15.34). Fix at least one slope as well, or "
                "use a Normal latent distribution."
            )
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
            davidian_degree=0,
        )
    elif model.is_davidian:
        degree = model.davidian_degree
        lat_params = davidian_param_names(degree)
        lat_vals = np.array(
            [[lookup[(g, LATENT_ITEM, p)][:4] for p in lat_params] for g in groups], dtype=float
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
            davidian_degree=degree,
        )
    else:
        lat_params = latent_param_names(model.n_factors)
        lat_vals = np.array(
            [[lookup[(g, LATENT_ITEM, p)][:4] for p in lat_params] for g in groups], dtype=float
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
            davidian_degree=0,
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
        node_log_base=jnp.asarray(quad.base_log_weights(model.n_factors), dtype=jnp.float64),
        families=tuple(families),
        latent=latent,
        priors=priors_block,
        n_groups=n_groups,
        coupled=_coupled_blocks(family_free_index, slot_bounds),
    )
    return Compiled(problem=problem, eta0=eta0, inverse=inverse, n_patterns=n_patterns)
