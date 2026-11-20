"""Compile layer: Spec + data -> ParameterTable -> CompiledProblem (DESIGN.md §2 ②, §5).

All validation happens here, before anything is jit-compiled, with messages naming the offending
item / parameter.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
import pandas as pd

from flexpyirt._core import CompiledProblem, FamilyBlock, LatentBlock
from flexpyirt.errors import IdentificationError, SpecificationError
from flexpyirt.items import ItemKernel
from flexpyirt.quadrature import Quadrature

if TYPE_CHECKING:
    from flexpyirt.model import Model

__all__ = [
    "LATENT_ITEM",
    "SINGLE_GROUP",
    "TABLE_COLUMNS",
    "Compiled",
    "build_table",
    "coerce_responses",
    "compile_problem",
    "validate_table",
]

LATENT_ITEM = "__latent__"
SINGLE_GROUP = "all"
LATENT_PARAMS = ("mean", "var")
TABLE_COLUMNS = ("group", "item", "model", "param", "value", "free", "lower", "upper", "free_id")


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------


def coerce_responses(
    data: Any,
    items: Sequence[str],
    *,
    item_names: Sequence[str] | None = None,
    require_variation: bool = True,
) -> np.ndarray:
    """Return an ``(N, J)`` int array of responses ordered like ``items``.

    Accepts a pandas DataFrame, a polars DataFrame (via ``to_pandas``), or a 2-D array-like with
    ``item_names``. In M1 responses must be 0/1 without missing values (DESIGN.md §13.3).

    Raises
    ------
    SpecificationError
        Missing or extra item columns, non-binary values, missing values, or (when
        ``require_variation``, i.e. for estimation) an item on which every respondent gave the
        same response.
    """
    if hasattr(data, "to_pandas") and not isinstance(data, pd.DataFrame):
        data = data.to_pandas()
    if isinstance(data, pd.DataFrame):
        columns = [str(c) for c in data.columns]
        frame = data
    else:
        arr = np.asarray(data)
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
    if np.isnan(values).any():
        bad = [items[int(j)] for j in np.flatnonzero(np.isnan(values).any(axis=0))]
        raise SpecificationError(
            f"Missing responses found in items {bad}. Missing data is supported from milestone M2."
        )
    for j, name in enumerate(items):
        col = values[:, j]
        if not np.isin(col, (0.0, 1.0)).all():
            raise SpecificationError(f"Item {name!r} has values other than 0/1.")
        if require_variation and col.min() == col.max():
            raise SpecificationError(
                f"Item {name!r}: every respondent gave response {int(col[0])}; "
                "its parameters are not estimable."
            )
    return values.astype(np.int32)


def compress_patterns(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return unique response patterns, their counts, and each row's pattern (DESIGN.md §4.1)."""
    patterns, inverse, counts = np.unique(y, axis=0, return_inverse=True, return_counts=True)
    return patterns, counts.astype(float), inverse.reshape(-1)


def pad_patterns(patterns: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pad to the next power-of-two bucket (at least 8) with zero-count rows (DESIGN.md §4.1)."""
    n = patterns.shape[0]
    bucket = max(8, 1 << (n - 1).bit_length())
    pad = bucket - n
    return (
        np.vstack([patterns, np.zeros((pad, patterns.shape[1]), dtype=patterns.dtype)]),
        np.concatenate([counts, np.zeros(pad)]),
    )


# --------------------------------------------------------------------------------------------
# Parameter table
# --------------------------------------------------------------------------------------------


def build_table(model: Model, y: np.ndarray) -> pd.DataFrame:
    """Build the default ParameterTable (DESIGN.md §5.1) from the spec and coerced responses."""
    rows: list[dict[str, Any]] = []
    total = y.sum(axis=1)
    fixed = {(item, param): value for item, param, value in model.fixed}
    for j, (name, item_model) in enumerate(model.items):
        start = item_model.start_values(np.column_stack([y[:, j], total - y[:, j]]))
        lower, upper = item_model.bounds()
        free_default = item_model.default_free()
        for k, param in enumerate(item_model.kernel.param_names):
            key = (name, param)
            rows.append(
                {
                    "group": SINGLE_GROUP,
                    "item": name,
                    "model": item_model.name,
                    "param": param,
                    "value": fixed.get(key, float(start[k])),
                    "free": free_default[k] and key not in fixed,
                    "lower": float(lower[k]),
                    "upper": float(upper[k]),
                }
            )
    latent = model.resolved_latent()
    latent_rows: tuple[tuple[str, float, bool, float], ...] = (
        ("mean", latent.mean, latent.free_mean, -np.inf),
        ("var", latent.var, latent.free_var, 0.0),
    )
    for param, value, free, lat_lower in latent_rows:
        rows.append(
            {
                "group": SINGLE_GROUP,
                "item": LATENT_ITEM,
                "model": "Normal",
                "param": param,
                "value": float(value),
                "free": bool(free),
                "lower": lat_lower,
                "upper": np.inf,
            }
        )
    table = pd.DataFrame(rows)
    table["free_id"] = -1
    table.loc[table["free"], "free_id"] = np.arange(int(table["free"].sum()))
    return table[list(TABLE_COLUMNS)].astype({"free": bool, "free_id": int})


def validate_table(table: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Validate a (possibly user-edited) table against the default one and normalize ``free_id``.

    Free rows with ``free_id < 0`` get fresh ids; ids are renumbered to ``0..n-1`` keeping any
    sharing. Sharing an id across different items is not supported before M5 (DESIGN.md §4.3).
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
    t = t.set_index(keys).loc[reference.set_index(keys).index].reset_index()[list(TABLE_COLUMNS)]
    t["model"] = reference["model"].to_numpy()
    t["free"] = t["free"].astype(bool)
    t["value"] = t["value"].astype(float)
    t["lower"] = t["lower"].astype(float)
    t["upper"] = t["upper"].astype(float)

    item_col = t["item"].to_numpy(dtype=object)
    param_col = t["param"].to_numpy(dtype=object)
    value = t["value"].to_numpy(dtype=float)
    lower = t["lower"].to_numpy(dtype=float)
    upper = t["upper"].to_numpy(dtype=float)
    free = t["free"].to_numpy(dtype=bool)
    for i in range(len(t)):
        where = f"(item={item_col[i]!r}, param={param_col[i]!r})"
        if not np.isfinite(value[i]):
            raise SpecificationError(f"Non-finite value {value[i]} at {where}.")
        if not lower[i] < upper[i]:
            raise SpecificationError(f"lower must be < upper at {where}.")
        if free[i] and not (lower[i] < value[i] < upper[i]):
            raise SpecificationError(
                f"Starting value {value[i]} is not strictly inside ({lower[i]}, {upper[i]}) "
                f"at {where}."
            )
        if item_col[i] == LATENT_ITEM and param_col[i] == "var" and not value[i] > 0:
            raise SpecificationError(f"Latent variance must be positive, got {value[i]}.")

    ids = np.where(free, t["free_id"].fillna(-1).to_numpy(dtype=int), -1)
    next_id = int(ids.max()) + 1 if (ids >= 0).any() else 0
    for i in np.flatnonzero(free & (ids < 0)).tolist():
        ids[i] = next_id
        next_id += 1
    _, ids_free = np.unique(ids[ids >= 0], return_inverse=True)
    ids[ids >= 0] = ids_free
    t["free_id"] = ids.astype(int)

    shared = t[t["free"]].groupby("free_id")["item"].nunique()
    if (shared > 1).any():
        fid = int(shared[shared > 1].index[0])
        items = sorted(t.loc[t["free_id"] == fid, "item"].unique())
        raise SpecificationError(
            f"free_id {fid} is shared by items {items}. Equality constraints across items are "
            "not supported yet (planned for M5)."
        )
    within = t[t["free"]].groupby("free_id")["param"].nunique()
    if (within > 1).any():
        fid = int(within[within > 1].index[0])
        raise SpecificationError(f"free_id {fid} is shared by different parameters of one item.")
    _check_identification(t)
    return t


def _check_identification(t: pd.DataFrame) -> None:
    """Scale identification rules of DESIGN.md §4.5 (single group)."""
    lat = t[t["item"] == LATENT_ITEM].set_index("param")
    items = t[t["item"] != LATENT_ITEM]
    if bool(lat.loc["var", "free"]):
        fixed_slopes = items[(items["param"] == "a") & ~items["free"] & (items["value"] != 0)]
        if fixed_slopes.empty:
            raise IdentificationError(
                "The latent variance is free but every slope is estimated, so the scale is not "
                "identified. Fix the latent variance or fix at least one slope (e.g. Rasch items)."
            )
    if bool(lat.loc["mean", "free"]):
        fixed_intercepts = items[(items["param"] == "d") & ~items["free"]]
        if fixed_intercepts.empty:
            raise IdentificationError(
                "The latent mean is free but every intercept is estimated, so the location is not "
                "identified. Fix the latent mean or fix at least one intercept."
            )


# --------------------------------------------------------------------------------------------
# Problem
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Compiled:
    """Compiled problem plus what the Result layer needs to map back."""

    problem: CompiledProblem
    eta0: np.ndarray
    table: pd.DataFrame
    n_obs: int


def compile_problem(model: Model, y: np.ndarray, table: pd.DataFrame, quad: Quadrature) -> Compiled:
    """Turn a validated table and responses into a :class:`CompiledProblem`."""
    patterns, counts, _ = compress_patterns(y)
    patterns, counts = pad_patterns(patterns, counts)
    n_free = int(table["free_id"].max()) + 1 if table["free"].any() else 0
    eta0 = np.zeros(n_free)
    free_rows = table[table["free"]]
    eta0[free_rows["free_id"].to_numpy(dtype=int)] = free_rows["value"].to_numpy(dtype=float)

    item_rows = table[table["item"] != LATENT_ITEM]
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
        sub = item_rows.set_index(["item", "param"])
        idx = pd.MultiIndex.from_product([names, list(k.param_names)])
        block = sub.loc[idx]
        shape = (len(cols), len(k.param_names))

        def arr(
            col: str, block: pd.DataFrame = block, shape: tuple[int, int] = shape
        ) -> np.ndarray:
            return block[col].to_numpy().reshape(shape)

        families.append(
            FamilyBlock(
                columns=jnp.asarray(cols, dtype=jnp.int32),
                fixed_values=jnp.asarray(arr("value"), dtype=jnp.float64),
                free_index=jnp.asarray(arr("free_id"), dtype=jnp.int32),
                lower=jnp.asarray(arr("lower"), dtype=jnp.float64),
                upper=jnp.asarray(arr("upper"), dtype=jnp.float64),
                kernel=k,
            )
        )

    lat = table[table["item"] == LATENT_ITEM].set_index("param").loc[list(LATENT_PARAMS)]

    def lat_arr(col: str, dtype: Any) -> Any:
        return jnp.asarray(lat[col].to_numpy().reshape(1, 2), dtype=dtype)

    latent = LatentBlock(
        fixed_values=lat_arr("value", jnp.float64),
        free_index=lat_arr("free_id", jnp.int32),
        lower=lat_arr("lower", jnp.float64),
        upper=lat_arr("upper", jnp.float64),
    )
    problem = CompiledProblem(
        y=jnp.asarray(patterns, dtype=jnp.int32),
        counts=jnp.asarray(counts, dtype=jnp.float64),
        group=jnp.zeros(patterns.shape[0], dtype=jnp.int32),
        nodes=jnp.asarray(quad.nodes(), dtype=jnp.float64),
        families=tuple(families),
        latent=latent,
        n_groups=1,
    )
    return Compiled(problem=problem, eta0=eta0, table=table, n_obs=int(y.shape[0]))
