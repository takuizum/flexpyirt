"""Strict JSON serialization and deserialization for :class:`~flexpyirt.FitResult`.

Implements the versioned plain-JSON format specified in DESIGN.md §15.26–§15.28:

- Top-level metadata: ``"format": "flexpyirt.FitResult"``, ``"schema_version": 4`` (versions 1
  and 2, written before multidimensional models and across-item equalities were stored, and
  version 3, written before the naming refactor of §15.46, are still read), and
  ``"flexpyirt_version"``.
- Strict RFC 8259 JSON (``allow_nan=False`` on write; rejecting ``parse_constant`` on read).
  Finite IEEE-754 float64 values use standard ``repr``-exact JSON numbers; non-finite values in
  float-typed fields are encoded explicitly as ``"NaN"``, ``"Infinity"``, or ``"-Infinity"``.
- Explicit name registries for item models, latent distributions, priors, quadrature grids, and
  estimation methods (never ``eval`` or ``pickle``).
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from flexpyirt._compile import (
    TABLE_COLUMNS,
    model_parameter_schema,
    validate_table_constraints,
)
from flexpyirt._precision import check_x64
from flexpyirt.errors import SpecificationError
from flexpyirt.items import GPCM, GRM, PCM, FourPL, ItemModel, Nominal, Rasch, ThreePL, TwoPL
from flexpyirt.latent import (
    MAX_DAVIDIAN_DEGREE,
    DavidianCurve,
    EmpiricalHistogram,
    LatentSpec,
    Normal,
)
from flexpyirt.method import BAEM
from flexpyirt.model import Model
from flexpyirt.priors import BetaPrior, LogNormalPrior, NormalPrior, Prior
from flexpyirt.quadrature import Quadrature

if TYPE_CHECKING:
    from flexpyirt.result import FitResult

__all__ = [
    "FORMAT_TAG",
    "SCHEMA_VERSION",
    "dump_fit_result",
    "load_fit_result",
]

FORMAT_TAG = "flexpyirt.FitResult"
SCHEMA_VERSION = 4
"""Schema version written by :func:`dump_fit_result`.

2 adds the confirmatory factor structure, 3 across-item equalities, and 4 renames the item-model
key ``n_cat`` to ``n_categories`` and the priors ``LogNormal(meanlog, sdlog)`` / ``Beta`` to
``LogNormalPrior(mean_log, sd_log)`` / ``BetaPrior`` (DESIGN.md §15.46).
"""
SUPPORTED_SCHEMA_VERSIONS: frozenset[int] = frozenset({1, 2, 3, 4})
"""Schema versions accepted by :func:`load_fit_result`; version 1 has no factor keys."""

_LEGACY_PRIOR_TYPES = {"LogNormal": "LogNormalPrior", "Beta": "BetaPrior"}
_LEGACY_PRIOR_KEYS = {"meanlog": "mean_log", "sdlog": "sd_log"}

_INT32_MAX = (1 << 31) - 1
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1
_MAX_INT_DIGITS = 20  # len(str(-2**63)) == 20
_MAX_ITEM_CATEGORIES = 1024
_MAX_QUADRATURE_POINTS = 100_000

_SPECIAL_FLOATS: dict[str, float] = {
    "NaN": math.nan,
    "Infinity": math.inf,
    "-Infinity": -math.inf,
}
_VALID_SE_METHODS: tuple[str | None, ...] = (None, "oakes", "hessian")
_PARAM_COLUMNS: tuple[str, ...] = (*TABLE_COLUMNS, "estimate", "se")
_HISTORY_COLUMNS: tuple[str, ...] = (
    "iter",
    "loglik",
    "log_posterior",
    "max_change",
    "em_evaluations",
    "safeguard",
)
_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "format",
        "schema_version",
        "flexpyirt_version",
        "model",
        "quadrature",
        "method",
        "n_obs",
        "se_method",
        "loglik",
        "log_posterior",
        "converged",
        "n_iter",
        "n_em_evaluations",
        "n_safeguards",
        "parameters",
        "vcov",
        "history",
    }
)


# --------------------------------------------------------------------------------------------
# Primitive validation and float encoding
# --------------------------------------------------------------------------------------------


def _reject_json_constant(token: str) -> Any:
    """Reject non-standard JSON constants (``NaN``, ``Infinity``, ``-Infinity``)."""
    raise SpecificationError(
        f"Non-standard JSON constant {token!r} is not allowed; "
        "non-finite floats must be encoded as 'NaN', 'Infinity', or '-Infinity'."
    )


def _parse_json_float(token: str) -> float:
    """Parse a JSON float token, rejecting numeric overflow to ``+/-Infinity``."""
    val = float(token)
    if not math.isfinite(val):
        raise SpecificationError(
            f"JSON number {token!r} overflows float64; "
            "non-finite floats must be encoded as 'NaN', 'Infinity', or '-Infinity'."
        )
    return val


def _parse_json_int(token: str) -> int:
    """Parse a JSON integer token, rejecting integers outside signed 64-bit range."""
    if len(token) > _MAX_INT_DIGITS:
        preview = token if len(token) <= 32 else f"{token[:32]}..."
        raise SpecificationError(f"JSON integer {preview!r} is out of 64-bit signed range.")
    val = int(token)
    if val < _INT64_MIN or val > _INT64_MAX:
        raise SpecificationError(f"JSON integer {token!r} is out of 64-bit signed range.")
    return val


def _encode_float(value: float | np.floating[Any]) -> float | str:
    """Encode a float64 for strict JSON: finite numbers verbatim, non-finite as tagged strings."""
    val = float(value)
    if math.isnan(val):
        return "NaN"
    if math.isinf(val):
        return "Infinity" if val > 0.0 else "-Infinity"
    return val


def _decode_float(raw: Any, *, field_name: str, finite: bool = False) -> float:
    """Decode a float from a JSON number or explicit ``'NaN'/'Infinity'/'-Infinity'`` string."""
    if isinstance(raw, str) and raw in _SPECIAL_FLOATS:
        val = _SPECIAL_FLOATS[raw]
    elif isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise SpecificationError(
            f"Expected a JSON number or 'NaN'/'Infinity'/'-Infinity' for {field_name}, got {raw!r}."
        )
    else:
        try:
            val = float(raw)
        except OverflowError as exc:
            raise SpecificationError(f"{field_name} overflows float64: {raw!r}.") from exc
        if not math.isfinite(val):
            raise SpecificationError(
                f"{field_name} overflows float64; "
                "non-finite floats must be encoded as 'NaN', 'Infinity', or '-Infinity'."
            )
    if finite and not math.isfinite(val):
        raise SpecificationError(f"{field_name} must be finite, got {val!r}.")
    return val


def _require_dict(raw: Any, *, field_name: str) -> dict[str, Any]:
    """Ensure ``raw`` is a JSON object (``dict``)."""
    if not isinstance(raw, dict):
        raise SpecificationError(
            f"Expected a JSON object for {field_name}, got {type(raw).__name__}."
        )
    return raw


def _require_list(raw: Any, *, field_name: str) -> list[Any]:
    """Ensure ``raw`` is a JSON array (``list``)."""
    if not isinstance(raw, list):
        raise SpecificationError(
            f"Expected a JSON array for {field_name}, got {type(raw).__name__}."
        )
    return raw


def _require_str(raw: Any, *, field_name: str) -> str:
    """Ensure ``raw`` is a JSON string."""
    if not isinstance(raw, str):
        raise SpecificationError(f"Expected a string for {field_name}, got {type(raw).__name__}.")
    return raw


def _require_bool(raw: Any, *, field_name: str) -> bool:
    """Ensure ``raw`` is a JSON boolean."""
    if not isinstance(raw, bool):
        raise SpecificationError(f"Expected a boolean for {field_name}, got {type(raw).__name__}.")
    return raw


def _require_int(
    raw: Any,
    *,
    field_name: str,
    min_value: int | None = None,
    max_value: int | None = _INT64_MAX,
) -> int:
    """Ensure ``raw`` is an integer (not ``bool``), with optional lower and upper bounds."""
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise SpecificationError(f"Expected an integer for {field_name}, got {raw!r}.")
    if min_value is not None and raw < min_value:
        raise SpecificationError(f"{field_name} must be >= {min_value}, got {raw}.")
    if max_value is not None and raw > max_value:
        raise SpecificationError(f"{field_name} must be <= {max_value}, got {raw}.")
    return raw


def _check_keys(
    obj: dict[str, Any],
    *,
    required: set[str] | frozenset[str],
    field_name: str,
) -> None:
    """Verify ``obj`` has exactly the ``required`` keys."""
    keys = set(obj.keys())
    missing = sorted(required - keys)
    if missing:
        raise SpecificationError(f"Missing required keys in {field_name}: {missing}.")
    extra = sorted(keys - required)
    if extra:
        raise SpecificationError(f"Unexpected keys in {field_name}: {extra}.")


def _floats_equal(a: float, b: float) -> bool:
    """Compare two floats, treating two NaNs as equal."""
    return (math.isnan(a) and math.isnan(b)) or a == b


# --------------------------------------------------------------------------------------------
# Explicit registries (DESIGN.md §15.28)
# --------------------------------------------------------------------------------------------


def _serialize_simple_item(model: Any) -> dict[str, Any]:
    return {"type": str(model.name)}


def _serialize_polytomous_item(model: Any) -> dict[str, Any]:
    return {"type": str(model.name), "n_categories": int(model.n_categories)}


def _make_simple_item_loader(cls: type[Any]) -> Callable[[dict[str, Any], str], ItemModel]:
    def _load(obj: dict[str, Any], field_name: str) -> ItemModel:
        _check_keys(obj, required={"type"}, field_name=field_name)
        item_model: ItemModel = cls()
        return item_model

    return _load


def _make_polytomous_item_loader(cls: type[Any]) -> Callable[[dict[str, Any], str], ItemModel]:
    def _load(obj: dict[str, Any], field_name: str) -> ItemModel:
        _check_keys(obj, required={"type", "n_categories"}, field_name=field_name)
        n_categories = _require_int(
            obj["n_categories"],
            field_name=f"{field_name}.n_categories",
            max_value=_MAX_ITEM_CATEGORIES,
        )
        item_model: ItemModel = cls(n_categories=n_categories)
        return item_model

    return _load


_ITEM_MODEL_SERIALIZERS: dict[type[Any], Callable[[Any], dict[str, Any]]] = {
    TwoPL: _serialize_simple_item,
    Rasch: _serialize_simple_item,
    ThreePL: _serialize_simple_item,
    FourPL: _serialize_simple_item,
    GRM: _serialize_polytomous_item,
    GPCM: _serialize_polytomous_item,
    PCM: _serialize_polytomous_item,
    Nominal: _serialize_polytomous_item,
}

_ITEM_MODEL_DESERIALIZERS: dict[str, Callable[[dict[str, Any], str], ItemModel]] = {
    TwoPL.name: _make_simple_item_loader(TwoPL),
    Rasch.name: _make_simple_item_loader(Rasch),
    ThreePL.name: _make_simple_item_loader(ThreePL),
    FourPL.name: _make_simple_item_loader(FourPL),
    GRM.name: _make_polytomous_item_loader(GRM),
    GPCM.name: _make_polytomous_item_loader(GPCM),
    PCM.name: _make_polytomous_item_loader(PCM),
    Nominal.name: _make_polytomous_item_loader(Nominal),
}


def _serialize_item_model(item_model: ItemModel) -> dict[str, Any]:
    """Serialize a registered :class:`~flexpyirt.items.ItemModel`."""
    serializer = _ITEM_MODEL_SERIALIZERS.get(type(item_model))
    if serializer is None:
        raise SpecificationError(
            f"Unsupported item model {type(item_model).__name__!r} for JSON serialization."
        )
    return serializer(item_model)


def _deserialize_item_model(raw: Any, *, field_name: str) -> ItemModel:
    """Deserialize an :class:`~flexpyirt.items.ItemModel` by its registered name."""
    obj = _require_dict(raw, field_name=field_name)
    if "type" not in obj:
        raise SpecificationError(f"Missing 'type' key in {field_name}.")
    kind = _require_str(obj["type"], field_name=f"{field_name}.type")
    loader = _ITEM_MODEL_DESERIALIZERS.get(kind)
    if loader is None:
        raise SpecificationError(
            f"Unknown item model type {kind!r} in {field_name}; "
            f"supported: {sorted(_ITEM_MODEL_DESERIALIZERS)}."
        )
    return loader(obj, field_name)


def _serialize_normal_latent(dist: Normal) -> dict[str, Any]:
    return {
        "type": "Normal",
        "mean": _encode_float(dist.mean),
        "var": _encode_float(dist.var),
        "free_mean": bool(dist.free_mean),
        "free_var": bool(dist.free_var),
    }


def _deserialize_normal_latent(obj: dict[str, Any], field_name: str) -> Normal:
    _check_keys(
        obj,
        required={"type", "mean", "var", "free_mean", "free_var"},
        field_name=field_name,
    )
    return Normal(
        mean=_decode_float(obj["mean"], field_name=f"{field_name}.mean", finite=True),
        var=_decode_float(obj["var"], field_name=f"{field_name}.var", finite=True),
        free_mean=_require_bool(obj["free_mean"], field_name=f"{field_name}.free_mean"),
        free_var=_require_bool(obj["free_var"], field_name=f"{field_name}.free_var"),
    )


def _serialize_eh_latent(dist: EmpiricalHistogram) -> dict[str, Any]:
    return {
        "type": "EmpiricalHistogram",
        "min_weight": _encode_float(dist.min_weight),
    }


def _deserialize_eh_latent(obj: dict[str, Any], field_name: str) -> EmpiricalHistogram:
    _check_keys(obj, required={"type", "min_weight"}, field_name=field_name)
    return EmpiricalHistogram(
        min_weight=_decode_float(
            obj["min_weight"], field_name=f"{field_name}.min_weight", finite=True
        )
    )


def _serialize_davidian_latent(dist: DavidianCurve) -> dict[str, Any]:
    return {
        "type": "DavidianCurve",
        "degree": int(dist.degree),
        "mean": _encode_float(dist.mean),
        "var": _encode_float(dist.var),
        "free_mean": bool(dist.free_mean),
        "free_var": bool(dist.free_var),
    }


def _deserialize_davidian_latent(obj: dict[str, Any], field_name: str) -> DavidianCurve:
    _check_keys(
        obj,
        required={"type", "degree", "mean", "var", "free_mean", "free_var"},
        field_name=field_name,
    )
    return DavidianCurve(
        degree=_require_int(
            obj["degree"],
            field_name=f"{field_name}.degree",
            min_value=1,
            max_value=MAX_DAVIDIAN_DEGREE,
        ),
        mean=_decode_float(obj["mean"], field_name=f"{field_name}.mean", finite=True),
        var=_decode_float(obj["var"], field_name=f"{field_name}.var", finite=True),
        free_mean=_require_bool(obj["free_mean"], field_name=f"{field_name}.free_mean"),
        free_var=_require_bool(obj["free_var"], field_name=f"{field_name}.free_var"),
    )


_LATENT_SERIALIZERS: dict[type[Any], Callable[[Any], dict[str, Any]]] = {
    Normal: _serialize_normal_latent,
    EmpiricalHistogram: _serialize_eh_latent,
    DavidianCurve: _serialize_davidian_latent,
}
_LATENT_DESERIALIZERS: dict[str, Callable[[dict[str, Any], str], LatentSpec]] = {
    "Normal": _deserialize_normal_latent,
    "EmpiricalHistogram": _deserialize_eh_latent,
    "DavidianCurve": _deserialize_davidian_latent,
}


def _serialize_latent(dist: LatentSpec) -> dict[str, Any]:
    """Serialize a registered latent distribution."""
    serializer = _LATENT_SERIALIZERS.get(type(dist))
    if serializer is None:
        raise SpecificationError(
            f"Unsupported latent distribution {type(dist).__name__!r} for JSON serialization."
        )
    return serializer(dist)


def _deserialize_latent(raw: Any, *, field_name: str) -> LatentSpec:
    """Deserialize a latent distribution by its registered name."""
    obj = _require_dict(raw, field_name=field_name)
    if "type" not in obj:
        raise SpecificationError(f"Missing 'type' key in {field_name}.")
    kind = _require_str(obj["type"], field_name=f"{field_name}.type")
    loader = _LATENT_DESERIALIZERS.get(kind)
    if loader is None:
        raise SpecificationError(
            f"Unknown latent distribution type {kind!r} in {field_name}; "
            f"supported: {sorted(_LATENT_DESERIALIZERS)}."
        )
    return loader(obj, field_name)


def _serialize_normal_prior(prior: NormalPrior) -> dict[str, Any]:
    return {
        "type": "NormalPrior",
        "mean": _encode_float(prior.mean),
        "sd": _encode_float(prior.sd),
    }


def _serialize_lognormal_prior(prior: LogNormalPrior) -> dict[str, Any]:
    return {
        "type": "LogNormalPrior",
        "mean_log": _encode_float(prior.mean_log),
        "sd_log": _encode_float(prior.sd_log),
    }


def _serialize_beta_prior(prior: BetaPrior) -> dict[str, Any]:
    return {
        "type": "BetaPrior",
        "a": _encode_float(prior.a),
        "b": _encode_float(prior.b),
    }


def _deserialize_normal_prior(obj: dict[str, Any], field_name: str) -> NormalPrior:
    _check_keys(obj, required={"type", "mean", "sd"}, field_name=field_name)
    return NormalPrior(
        mean=_decode_float(obj["mean"], field_name=f"{field_name}.mean", finite=True),
        sd=_decode_float(obj["sd"], field_name=f"{field_name}.sd", finite=True),
    )


def _deserialize_lognormal_prior(obj: dict[str, Any], field_name: str) -> LogNormalPrior:
    _check_keys(obj, required={"type", "mean_log", "sd_log"}, field_name=field_name)
    return LogNormalPrior(
        mean_log=_decode_float(obj["mean_log"], field_name=f"{field_name}.mean_log", finite=True),
        sd_log=_decode_float(obj["sd_log"], field_name=f"{field_name}.sd_log", finite=True),
    )


def _deserialize_beta_prior(obj: dict[str, Any], field_name: str) -> BetaPrior:
    _check_keys(obj, required={"type", "a", "b"}, field_name=field_name)
    return BetaPrior(
        a=_decode_float(obj["a"], field_name=f"{field_name}.a", finite=True),
        b=_decode_float(obj["b"], field_name=f"{field_name}.b", finite=True),
    )


_PRIOR_SERIALIZERS: dict[type[Any], Callable[[Any], dict[str, Any]]] = {
    NormalPrior: _serialize_normal_prior,
    LogNormalPrior: _serialize_lognormal_prior,
    BetaPrior: _serialize_beta_prior,
}
_PRIOR_DESERIALIZERS: dict[str, Callable[[dict[str, Any], str], Prior]] = {
    "NormalPrior": _deserialize_normal_prior,
    "LogNormalPrior": _deserialize_lognormal_prior,
    "BetaPrior": _deserialize_beta_prior,
}


def _serialize_prior(prior: Prior | None) -> dict[str, Any] | None:
    """Serialize a registered prior (or ``None``)."""
    if prior is None:
        return None
    serializer = _PRIOR_SERIALIZERS.get(type(prior))
    if serializer is None:
        raise SpecificationError(
            f"Unsupported prior distribution {type(prior).__name__!r} for JSON serialization."
        )
    return serializer(prior)


def _deserialize_prior(raw: Any, *, field_name: str) -> Prior | None:
    """Deserialize a prior by its registered name (or ``None``)."""
    if raw is None:
        return None
    obj = _require_dict(raw, field_name=field_name)
    if "type" not in obj:
        raise SpecificationError(f"Missing 'type' key in {field_name}.")
    kind = _require_str(obj["type"], field_name=f"{field_name}.type")
    loader = _PRIOR_DESERIALIZERS.get(kind)
    if loader is None:
        raise SpecificationError(
            f"Unknown prior type {kind!r} in {field_name}; "
            f"supported: {sorted(_PRIOR_DESERIALIZERS)}."
        )
    return loader(obj, field_name)


def _upgrade_legacy_names(raw: Any) -> Any:
    """Rename schema 1–3 keys to their schema 4 names (DESIGN.md §15.46); returns a new object.

    Item models: ``n_cat`` -> ``n_categories``. Priors: ``LogNormal`` -> ``LogNormalPrior`` (with
    ``meanlog``/``sdlog`` -> ``mean_log``/``sd_log``) and ``Beta`` -> ``BetaPrior``. Only
    dictionaries carrying an item-model or prior ``type`` are touched.
    """
    if isinstance(raw, list):
        return [_upgrade_legacy_names(v) for v in raw]
    if not isinstance(raw, dict):
        return raw
    out = {k: _upgrade_legacy_names(v) for k, v in raw.items()}
    kind = out.get("type")
    if kind in _LEGACY_PRIOR_TYPES:
        out["type"] = _LEGACY_PRIOR_TYPES[kind]
        out = {_LEGACY_PRIOR_KEYS.get(k, k): v for k, v in out.items()}
    elif isinstance(kind, str) and kind in _ITEM_MODEL_DESERIALIZERS and "n_cat" in out:
        out = {("n_categories" if k == "n_cat" else k): v for k, v in out.items()}
    return out


def _serialize_equispaced_quadrature(quad: Quadrature) -> dict[str, Any]:
    return {
        "type": "equispaced",
        "n_points": int(quad.n_points),
        "lower": _encode_float(quad.lower),
        "upper": _encode_float(quad.upper),
    }


def _deserialize_equispaced_quadrature(obj: dict[str, Any], field_name: str) -> Quadrature:
    _check_keys(obj, required={"type", "n_points", "lower", "upper"}, field_name=field_name)
    return Quadrature(
        n_points=_require_int(
            obj["n_points"],
            field_name=f"{field_name}.n_points",
            max_value=_MAX_QUADRATURE_POINTS,
        ),
        lower=_decode_float(obj["lower"], field_name=f"{field_name}.lower", finite=True),
        upper=_decode_float(obj["upper"], field_name=f"{field_name}.upper", finite=True),
    )


def _serialize_gauss_hermite_quadrature(quad: Quadrature) -> dict[str, Any]:
    return {"type": "gauss_hermite", "n_points": int(quad.n_points)}


def _deserialize_gauss_hermite_quadrature(obj: dict[str, Any], field_name: str) -> Quadrature:
    _check_keys(obj, required={"type", "n_points"}, field_name=field_name)
    return Quadrature.gauss_hermite(
        _require_int(obj["n_points"], field_name=f"{field_name}.n_points", max_value=200)
    )


_QUADRATURE_SERIALIZERS: dict[str, Callable[[Any], dict[str, Any]]] = {
    "equispaced": _serialize_equispaced_quadrature,
    "gauss_hermite": _serialize_gauss_hermite_quadrature,
}
_QUADRATURE_DESERIALIZERS: dict[str, Callable[[dict[str, Any], str], Quadrature]] = {
    "equispaced": _deserialize_equispaced_quadrature,
    "gauss_hermite": _deserialize_gauss_hermite_quadrature,
}


def _serialize_quadrature(quad: Quadrature) -> dict[str, Any]:
    """Serialize a registered :class:`~flexpyirt.Quadrature`."""
    serializer = _QUADRATURE_SERIALIZERS.get(quad.kind) if isinstance(quad, Quadrature) else None
    if serializer is None:
        raise SpecificationError(
            f"Unsupported quadrature {type(quad).__name__!r} for JSON serialization."
        )
    return serializer(quad)


def _deserialize_quadrature(raw: Any, *, field_name: str) -> Quadrature:
    """Deserialize a :class:`~flexpyirt.Quadrature` by its registered name."""
    obj = _require_dict(raw, field_name=field_name)
    if "type" not in obj:
        raise SpecificationError(f"Missing 'type' key in {field_name}.")
    kind = _require_str(obj["type"], field_name=f"{field_name}.type")
    loader = _QUADRATURE_DESERIALIZERS.get(kind)
    if loader is None:
        raise SpecificationError(
            f"Unknown quadrature type {kind!r} in {field_name}; "
            f"supported: {sorted(_QUADRATURE_DESERIALIZERS)}."
        )
    return loader(obj, field_name)


def _serialize_baem(method: BAEM) -> dict[str, Any]:
    return {
        "type": "BAEM",
        "tol": _encode_float(method.tol),
        "max_iter": int(method.max_iter),
        "m_step_newton_steps": int(method.m_step_newton_steps),
        "squarem": bool(method.squarem),
    }


def _deserialize_baem(obj: dict[str, Any], field_name: str) -> BAEM:
    _check_keys(
        obj,
        required={"type", "tol", "max_iter", "m_step_newton_steps", "squarem"},
        field_name=field_name,
    )
    return BAEM(
        tol=_decode_float(obj["tol"], field_name=f"{field_name}.tol", finite=True),
        max_iter=_require_int(
            obj["max_iter"], field_name=f"{field_name}.max_iter", max_value=_INT32_MAX
        ),
        m_step_newton_steps=_require_int(
            obj["m_step_newton_steps"],
            field_name=f"{field_name}.m_step_newton_steps",
            max_value=_INT32_MAX,
        ),
        squarem=_require_bool(obj["squarem"], field_name=f"{field_name}.squarem"),
    )


_METHOD_SERIALIZERS: dict[type[Any], Callable[[Any], dict[str, Any]]] = {
    BAEM: _serialize_baem,
}
_METHOD_DESERIALIZERS: dict[str, Callable[[dict[str, Any], str], BAEM]] = {
    "BAEM": _deserialize_baem,
}


def _serialize_method(method: BAEM) -> dict[str, Any]:
    """Serialize a registered estimation method."""
    serializer = _METHOD_SERIALIZERS.get(type(method))
    if serializer is None:
        raise SpecificationError(
            f"Unsupported estimation method {type(method).__name__!r} for JSON serialization."
        )
    return serializer(method)


def _deserialize_method(raw: Any, *, field_name: str) -> BAEM:
    """Deserialize an estimation method by its registered name."""
    obj = _require_dict(raw, field_name=field_name)
    if "type" not in obj:
        raise SpecificationError(f"Missing 'type' key in {field_name}.")
    kind = _require_str(obj["type"], field_name=f"{field_name}.type")
    loader = _METHOD_DESERIALIZERS.get(kind)
    if loader is None:
        raise SpecificationError(
            f"Unknown estimation method type {kind!r} in {field_name}; "
            f"supported: {sorted(_METHOD_DESERIALIZERS)}."
        )
    return loader(obj, field_name)


# --------------------------------------------------------------------------------------------
# Model, ParameterTable, vcov, history serialization
# --------------------------------------------------------------------------------------------


def _serialize_model(model: Model) -> dict[str, Any]:
    """Serialize a :class:`~flexpyirt.Model` specification."""
    return {
        "items": [
            {"name": name, "model": _serialize_item_model(item_model)}
            for name, item_model in model.items
        ],
        "fixed": [
            {"item": item, "param": param, "value": _encode_float(value)}
            for item, param, value in model.fixed
        ],
        "latent": None if model.latent is None else _serialize_latent(model.latent),
        "groups": (
            None
            if model.groups is None
            else [
                {"name": group_name, "latent": _serialize_latent(dist)}
                for group_name, dist in model.groups
            ]
        ),
        "equal": [
            {
                "items": list(eq_items),
                "params": None if eq_params is None else list(eq_params),
            }
            for eq_items, eq_params in model.equal
        ],
        "priors": [
            {"item": item, "param": param, "dist": _serialize_prior(dist)}
            for item, param, dist in model.priors
        ],
        "factors": (
            None
            if model.factors is None
            else [{"name": name, "items": list(items)} for name, items in model.factors]
        ),
        "correlated_factors": bool(model.correlated_factors),
        "equal_items": [
            {"items": list(eq_items), "params": list(eq_params)}
            for eq_items, eq_params in model.equal_items
        ],
    }


def _deserialize_model(raw: Any, *, schema_version: int = SCHEMA_VERSION) -> Model:
    """Deserialize and validate a :class:`~flexpyirt.Model` specification.

    Schema version 2 adds ``factors`` and ``correlated_factors``; version 1 payloads have neither
    and are unidimensional. Schema version 3 adds ``equal_items`` (equalities across items,
    DESIGN.md §16.1), rebuilt through :meth:`Model.equal_across_items`. A factor structure is
    rebuilt through :meth:`Model.with_factors`, so the same validation applies as when it was
    first declared, and the fixed values, equalities and priors are checked against the
    multidimensional parameter names (``a1``..``aD``).
    """
    obj = _require_dict(raw, field_name="model")
    required = {"items", "fixed", "latent", "groups", "equal", "priors"}
    if schema_version >= 2:
        required |= {"factors", "correlated_factors"}
    if schema_version >= 3:
        required |= {"equal_items"}
    _check_keys(obj, required=required, field_name="model")

    items_raw = _require_list(obj["items"], field_name="model.items")
    items_list: list[tuple[str, ItemModel]] = []
    for idx, entry in enumerate(items_raw):
        entry_obj = _require_dict(entry, field_name=f"model.items[{idx}]")
        _check_keys(entry_obj, required={"name", "model"}, field_name=f"model.items[{idx}]")
        name = _require_str(entry_obj["name"], field_name=f"model.items[{idx}].name")
        im = _deserialize_item_model(entry_obj["model"], field_name=f"model.items[{idx}].model")
        items_list.append((name, im))
    items_tuple = tuple(items_list)
    # Validates non-empty and unique item names.
    base = Model(items=items_tuple)
    lookup = dict(base.items)
    if schema_version >= 2 and obj["factors"] is not None:
        correlated = _require_bool(obj["correlated_factors"], field_name="model.correlated_factors")
        factors_raw = _require_list(obj["factors"], field_name="model.factors")
        factor_map: dict[str, list[str]] = {}
        for idx, entry in enumerate(factors_raw):
            f_obj = _require_dict(entry, field_name=f"model.factors[{idx}]")
            _check_keys(f_obj, required={"name", "items"}, field_name=f"model.factors[{idx}]")
            f_name = _require_str(f_obj["name"], field_name=f"model.factors[{idx}].name")
            if f_name in factor_map:
                raise SpecificationError(f"Duplicated factor name {f_name!r} in model.factors.")
            f_items_raw = _require_list(f_obj["items"], field_name=f"model.factors[{idx}].items")
            f_items = [
                _require_str(v, field_name=f"model.factors[{idx}].items[{k}]")
                for k, v in enumerate(f_items_raw)
            ]
            unknown_f = [i for i in f_items if i not in lookup]
            if unknown_f:
                raise SpecificationError(
                    f"model.factors[{idx}] references unknown items {unknown_f}."
                )
            factor_map[f_name] = f_items
        base = base.with_factors(factor_map, correlated=correlated)
    elif schema_version >= 2:
        _require_bool(obj["correlated_factors"], field_name="model.correlated_factors")

    def _param_names(item: str) -> tuple[str, ...]:
        return base.item_kernel(lookup[item]).param_names

    latent = (
        None
        if obj["latent"] is None
        else _deserialize_latent(obj["latent"], field_name="model.latent")
    )
    groups: tuple[tuple[str, LatentSpec], ...] | None = None
    if obj["groups"] is not None:
        groups_raw = _require_list(obj["groups"], field_name="model.groups")
        if len(groups_raw) < 2:
            raise SpecificationError("model.groups must contain at least two groups.")
        groups_list: list[tuple[str, LatentSpec]] = []
        seen_groups: set[str] = set()
        for idx, entry in enumerate(groups_raw):
            g_obj = _require_dict(entry, field_name=f"model.groups[{idx}]")
            _check_keys(g_obj, required={"name", "latent"}, field_name=f"model.groups[{idx}]")
            g_name = _require_str(g_obj["name"], field_name=f"model.groups[{idx}].name")
            if g_name in seen_groups:
                raise SpecificationError(f"Duplicated group name {g_name!r} in model.groups.")
            seen_groups.add(g_name)
            g_lat = _deserialize_latent(g_obj["latent"], field_name=f"model.groups[{idx}].latent")
            groups_list.append((g_name, g_lat))
        groups = tuple(groups_list)
    if latent is not None and groups is not None:
        raise SpecificationError("model cannot specify both 'latent' and 'groups'.")
    if base.is_multidimensional and (latent is not None or groups is not None):
        raise SpecificationError(
            "model with factors cannot specify 'latent' or 'groups' (multidimensional models "
            "are single-group with a multivariate normal latent distribution)."
        )

    fixed_raw = _require_list(obj["fixed"], field_name="model.fixed")
    fixed_list: list[tuple[str, str, float]] = []
    for idx, entry in enumerate(fixed_raw):
        f_obj = _require_dict(entry, field_name=f"model.fixed[{idx}]")
        _check_keys(f_obj, required={"item", "param", "value"}, field_name=f"model.fixed[{idx}]")
        f_item = _require_str(f_obj["item"], field_name=f"model.fixed[{idx}].item")
        f_param = _require_str(f_obj["param"], field_name=f"model.fixed[{idx}].param")
        f_val = _decode_float(f_obj["value"], field_name=f"model.fixed[{idx}].value", finite=True)
        if f_item not in lookup:
            raise SpecificationError(f"model.fixed[{idx}] references unknown item {f_item!r}.")
        if f_param not in _param_names(f_item):
            raise SpecificationError(
                f"model.fixed[{idx}] references unknown parameter {f_param!r} for item {f_item!r}."
            )
        fixed_list.append((f_item, f_param, f_val))

    equal_raw = _require_list(obj["equal"], field_name="model.equal")
    equal_list: list[tuple[tuple[str, ...], tuple[str, ...] | None]] = []
    for idx, entry in enumerate(equal_raw):
        e_obj = _require_dict(entry, field_name=f"model.equal[{idx}]")
        _check_keys(e_obj, required={"items", "params"}, field_name=f"model.equal[{idx}]")
        eq_items_raw = _require_list(e_obj["items"], field_name=f"model.equal[{idx}].items")
        if not eq_items_raw:
            raise SpecificationError(f"model.equal[{idx}].items cannot be empty.")
        eq_items = tuple(
            _require_str(v, field_name=f"model.equal[{idx}].items[{k}]")
            for k, v in enumerate(eq_items_raw)
        )
        unknown_eq = [i for i in eq_items if i not in lookup]
        if unknown_eq:
            raise SpecificationError(
                f"model.equal[{idx}].items references unknown items {unknown_eq}."
            )
        eq_params: tuple[str, ...] | None = None
        if e_obj["params"] is not None:
            eq_params_raw = _require_list(e_obj["params"], field_name=f"model.equal[{idx}].params")
            eq_params = tuple(
                _require_str(v, field_name=f"model.equal[{idx}].params[{k}]")
                for k, v in enumerate(eq_params_raw)
            )
            for item in eq_items:
                bad = [p for p in eq_params if p not in _param_names(item)]
                if bad:
                    raise SpecificationError(
                        f"model.equal[{idx}] references unknown parameters {bad} for item {item!r}."
                    )
        equal_list.append((eq_items, eq_params))

    priors_raw = _require_list(obj["priors"], field_name="model.priors")
    priors_list: list[tuple[str, str, Prior]] = []
    for idx, entry in enumerate(priors_raw):
        p_obj = _require_dict(entry, field_name=f"model.priors[{idx}]")
        _check_keys(p_obj, required={"item", "param", "dist"}, field_name=f"model.priors[{idx}]")
        p_item = _require_str(p_obj["item"], field_name=f"model.priors[{idx}].item")
        p_param = _require_str(p_obj["param"], field_name=f"model.priors[{idx}].param")
        p_dist = _deserialize_prior(p_obj["dist"], field_name=f"model.priors[{idx}].dist")
        if p_dist is None:
            raise SpecificationError(f"model.priors[{idx}].dist cannot be null.")
        if p_item not in lookup:
            raise SpecificationError(f"model.priors[{idx}] references unknown item {p_item!r}.")
        if p_param not in _param_names(p_item):
            raise SpecificationError(
                f"model.priors[{idx}] references unknown parameter {p_param!r} for item {p_item!r}."
            )
        priors_list.append((p_item, p_param, p_dist))

    with_items = base
    if schema_version >= 3:
        eq_items_raw = _require_list(obj["equal_items"], field_name="model.equal_items")
        for idx, entry in enumerate(eq_items_raw):
            loc = f"model.equal_items[{idx}]"
            ei_obj = _require_dict(entry, field_name=loc)
            _check_keys(ei_obj, required={"items", "params"}, field_name=loc)
            ei_items = [
                _require_str(v, field_name=f"{loc}.items[{k}]")
                for k, v in enumerate(_require_list(ei_obj["items"], field_name=f"{loc}.items"))
            ]
            ei_params = [
                _require_str(v, field_name=f"{loc}.params[{k}]")
                for k, v in enumerate(_require_list(ei_obj["params"], field_name=f"{loc}.params"))
            ]
            unknown_ei = [i for i in ei_items if i not in lookup]
            if unknown_ei:
                raise SpecificationError(f"{loc}.items references unknown items {unknown_ei}.")
            with_items = with_items.equal_across_items(items=ei_items, params=ei_params)

    return replace(
        base,
        fixed=tuple(fixed_list),
        latent=latent,
        groups=groups,
        equal=tuple(equal_list),
        equal_items=with_items.equal_items,
        priors=tuple(priors_list),
    )


def _serialize_parameters(table: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialize ``FitResult.parameters`` as a list of row objects."""
    rows: list[dict[str, Any]] = []
    cols = [table[c].tolist() for c in _PARAM_COLUMNS]
    for grp, item, mdl, param, val, free, lo, hi, pri, fid, est, se in zip(*cols, strict=True):
        rows.append(
            {
                "group": str(grp),
                "item": str(item),
                "model": str(mdl),
                "param": str(param),
                "value": _encode_float(val),
                "free": bool(free),
                "lower": _encode_float(lo),
                "upper": _encode_float(hi),
                "prior": _serialize_prior(pri),
                "free_id": int(fid),
                "estimate": _encode_float(est),
                "se": _encode_float(se),
            }
        )
    return rows


def _deserialize_parameters(raw: Any, model: Model) -> tuple[pd.DataFrame, np.ndarray]:
    """Deserialize ``parameters`` and reconstruct ``_eta`` with full consistency checks."""
    rows_raw = _require_list(raw, field_name="parameters")
    schema = model_parameter_schema(model)
    expected: dict[tuple[str, str, str], str] = {
        (str(g), str(i), str(p)): str(m)
        for g, i, p, m in zip(
            schema["group"], schema["item"], schema["param"], schema["model"], strict=True
        )
    }
    required_keys = frozenset(_PARAM_COLUMNS)

    groups: list[str] = []
    items: list[str] = []
    models: list[str] = []
    params: list[str] = []
    values: list[float] = []
    frees: list[bool] = []
    lowers: list[float] = []
    uppers: list[float] = []
    priors: list[Prior | None] = []
    free_ids: list[int] = []
    estimates: list[float] = []
    ses: list[float] = []

    seen: set[tuple[str, str, str]] = set()

    for idx, entry in enumerate(rows_raw):
        loc = f"parameters[{idx}]"
        row = _require_dict(entry, field_name=loc)
        _check_keys(row, required=required_keys, field_name=loc)

        grp = _require_str(row["group"], field_name=f"{loc}.group")
        itm = _require_str(row["item"], field_name=f"{loc}.item")
        mdl = _require_str(row["model"], field_name=f"{loc}.model")
        prm = _require_str(row["param"], field_name=f"{loc}.param")
        key = (grp, itm, prm)
        if key in seen:
            raise SpecificationError(f"Duplicated parameter row {key} in parameters.")
        if key not in expected:
            raise SpecificationError(
                f"Unexpected parameter row {key} in parameters for this model."
            )
        if mdl != expected[key]:
            raise SpecificationError(
                f"Parameter row {key} has model {mdl!r}, expected {expected[key]!r}."
            )
        seen.add(key)

        val = _decode_float(row["value"], field_name=f"{loc}.value", finite=True)
        is_free = _require_bool(row["free"], field_name=f"{loc}.free")
        lo = _decode_float(row["lower"], field_name=f"{loc}.lower")
        hi = _decode_float(row["upper"], field_name=f"{loc}.upper")
        pri = _deserialize_prior(row["prior"], field_name=f"{loc}.prior")
        fid = _require_int(row["free_id"], field_name=f"{loc}.free_id", max_value=_INT32_MAX)
        if is_free:
            if fid < 0:
                raise SpecificationError(f"Free parameter row {key} must have free_id >= 0.")
        elif fid != -1:
            raise SpecificationError(
                f"Fixed parameter row {key} must have free_id == -1, got {fid}."
            )
        est = _decode_float(row["estimate"], field_name=f"{loc}.estimate", finite=True)
        se_val = _decode_float(row["se"], field_name=f"{loc}.se")
        if not is_free:
            if est != val:
                raise SpecificationError(
                    f"Fixed parameter row {key} has estimate ({est!r}) != value ({val!r})."
                )
            if not math.isnan(se_val):
                raise SpecificationError(
                    f"Fixed parameter row {key} must have se == NaN, got {se_val!r}."
                )
        elif not (math.isnan(se_val) or (math.isfinite(se_val) and se_val >= 0.0)):
            raise SpecificationError(
                f"Free parameter row {key} must have finite non-negative or NaN se, got {se_val!r}."
            )

        groups.append(grp)
        items.append(itm)
        models.append(mdl)
        params.append(prm)
        values.append(val)
        frees.append(is_free)
        lowers.append(lo)
        uppers.append(hi)
        priors.append(pri)
        free_ids.append(fid)
        estimates.append(est)
        ses.append(se_val)

    missing = sorted(set(expected) - seen)
    if missing:
        raise SpecificationError(f"Missing parameter rows for this model: {missing}.")

    frame = pd.DataFrame(
        {
            "group": groups,
            "item": items,
            "model": models,
            "param": params,
            "value": np.array(values, dtype=np.float64),
            "free": np.array(frees, dtype=bool),
            "lower": np.array(lowers, dtype=np.float64),
            "upper": np.array(uppers, dtype=np.float64),
            "prior": pd.Series(priors, dtype=object),
            "free_id": np.array(free_ids, dtype=int),
            "estimate": np.array(estimates, dtype=np.float64),
            "se": np.array(ses, dtype=np.float64),
        }
    )
    frame = validate_table_constraints(
        frame,
        schema,
        model,
        check_free_ids=True,
        extra_value_cols=(("estimate", "Estimate", "Fixed estimate"),),
    )

    free_meta: dict[int, tuple[float, float]] = {}
    free_rows = frame[frame["free"]]
    for fid_val, est_val, se_item in zip(
        free_rows["free_id"].tolist(),
        free_rows["estimate"].tolist(),
        free_rows["se"].tolist(),
        strict=True,
    ):
        prev = free_meta.get(fid_val)
        if prev is None:
            free_meta[fid_val] = (float(est_val), float(se_item))
        else:
            prev_est, prev_se = prev
            if prev_est != est_val:
                raise SpecificationError(
                    f"Rows sharing free_id {fid_val} have conflicting estimates "
                    f"({prev_est!r} vs {est_val!r})."
                )
            if not _floats_equal(prev_se, float(se_item)):
                raise SpecificationError(
                    f"Rows sharing free_id {fid_val} have conflicting standard errors "
                    f"({prev_se!r} vs {se_item!r})."
                )

    n_free = len(free_meta)
    expected_fids = set(range(n_free))
    if set(free_meta.keys()) != expected_fids:
        raise SpecificationError(
            f"Free parameter free_id values must be contiguous 0..{n_free - 1}, "
            f"got {sorted(free_meta.keys())}."
        )
    eta = np.empty(n_free, dtype=np.float64)
    for fid_val, (est_val, _) in free_meta.items():
        eta[fid_val] = est_val

    return frame, eta


def _serialize_vcov(vcov: np.ndarray | None) -> list[list[float | str]] | None:
    """Serialize ``vcov`` as a 2-D JSON list (or ``None``)."""
    if vcov is None:
        return None
    arr = np.asarray(vcov, dtype=np.float64)
    return [[_encode_float(x) for x in row] for row in arr]


def _deserialize_vcov(raw: Any, *, n_free: int) -> np.ndarray | None:
    """Deserialize ``vcov`` and verify its ``(n_free, n_free)`` shape."""
    if raw is None:
        return None
    rows = _require_list(raw, field_name="vcov")
    if len(rows) != n_free:
        raise SpecificationError(
            f"vcov must have shape ({n_free}, {n_free}), got {len(rows)} rows."
        )
    out = np.empty((n_free, n_free), dtype=np.float64)
    for i, row_raw in enumerate(rows):
        row = _require_list(row_raw, field_name=f"vcov[{i}]")
        if len(row) != n_free:
            raise SpecificationError(f"vcov[{i}] must have length {n_free}, got {len(row)}.")
        for j, val in enumerate(row):
            out[i, j] = _decode_float(val, field_name=f"vcov[{i}][{j}]")
    return out


def _serialize_history(history: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialize ``FitResult.history`` as a list of per-iteration objects."""
    rows: list[dict[str, Any]] = []
    cols = [history[c].tolist() for c in _HISTORY_COLUMNS]
    for it, ll, lp, chg, ev, sg in zip(*cols, strict=True):
        rows.append(
            {
                "iter": int(it),
                "loglik": _encode_float(ll),
                "log_posterior": _encode_float(lp),
                "max_change": _encode_float(chg),
                "em_evaluations": int(ev),
                "safeguard": bool(sg),
            }
        )
    return rows


def _deserialize_history(
    raw: Any,
    *,
    n_iter: int,
    n_em_evaluations: int,
    n_safeguards: int,
) -> pd.DataFrame:
    """Deserialize ``FitResult.history`` and verify its length, ordering, and totals."""
    rows_raw = _require_list(raw, field_name="history")
    if len(rows_raw) != n_iter:
        raise SpecificationError(
            f"history length ({len(rows_raw)}) does not match n_iter ({n_iter})."
        )
    required_keys = frozenset(_HISTORY_COLUMNS)
    iters: list[int] = []
    logliks: list[float] = []
    log_posts: list[float] = []
    changes: list[float] = []
    em_evals: list[int] = []
    safeguards: list[bool] = []
    for idx, entry in enumerate(rows_raw):
        loc = f"history[{idx}]"
        row = _require_dict(entry, field_name=loc)
        _check_keys(row, required=required_keys, field_name=loc)
        it_val = _require_int(row["iter"], field_name=f"{loc}.iter", min_value=1)
        if it_val != idx + 1:
            raise SpecificationError(f"{loc}.iter must be {idx + 1}, got {it_val}.")
        iters.append(it_val)
        logliks.append(_decode_float(row["loglik"], field_name=f"{loc}.loglik"))
        log_posts.append(_decode_float(row["log_posterior"], field_name=f"{loc}.log_posterior"))
        changes.append(_decode_float(row["max_change"], field_name=f"{loc}.max_change"))
        ev_val = _require_int(
            row["em_evaluations"],
            field_name=f"{loc}.em_evaluations",
            min_value=0,
            max_value=_INT32_MAX,
        )
        if em_evals and ev_val < em_evals[-1]:
            raise SpecificationError(
                f"{loc}.em_evaluations ({ev_val}) must be non-decreasing (previous {em_evals[-1]})."
            )
        em_evals.append(ev_val)
        safeguards.append(_require_bool(row["safeguard"], field_name=f"{loc}.safeguard"))

    total_safeguards = sum(safeguards)
    if total_safeguards != n_safeguards:
        raise SpecificationError(
            f"sum(history.safeguard) ({total_safeguards}) does not match "
            f"n_safeguards ({n_safeguards})."
        )
    if n_iter == 0:
        if n_em_evaluations != 0:
            raise SpecificationError(
                f"n_em_evaluations must be 0 when n_iter is 0, got {n_em_evaluations}."
            )
    elif em_evals[-1] != n_em_evaluations:
        raise SpecificationError(
            f"Final history.em_evaluations ({em_evals[-1]}) does not match "
            f"n_em_evaluations ({n_em_evaluations})."
        )

    return pd.DataFrame(
        {
            "iter": np.array(iters, dtype=np.int64),
            "loglik": np.array(logliks, dtype=np.float64),
            "log_posterior": np.array(log_posts, dtype=np.float64),
            "max_change": np.array(changes, dtype=np.float64),
            "em_evaluations": np.array(em_evals, dtype=np.int32),
            "safeguard": np.array(safeguards, dtype=bool),
        }
    )


def _serialize_eh_weights(eh_weights: np.ndarray) -> list[list[float | str]]:
    """Serialize ``_eh_weights`` of shape ``(G, Q)`` as a nested JSON list."""
    arr = np.asarray(eh_weights, dtype=np.float64)
    return [[_encode_float(x) for x in row] for row in arr]


def _deserialize_eh_weights(raw: Any, *, model: Model, n_nodes: int) -> np.ndarray:
    """Deserialize and validate ``latent_weights`` of shape ``(n_groups, n_nodes)``."""
    resolved = model.resolved_groups()
    n_groups = len(resolved)
    rows = _require_list(raw, field_name="latent_weights")
    if len(rows) != n_groups:
        raise SpecificationError(
            f"latent_weights must have {n_groups} group rows, got {len(rows)}."
        )
    out = np.empty((n_groups, n_nodes), dtype=np.float64)
    for g, (row_raw, (g_name, dist)) in enumerate(zip(rows, resolved, strict=True)):
        assert isinstance(dist, EmpiricalHistogram)
        mw = float(dist.min_weight)
        if mw * n_nodes >= 1.0:
            raise SpecificationError(
                f"EmpiricalHistogram min_weight ({mw}) for group {g_name!r} must satisfy "
                f"min_weight * n_points < 1 (n_points={n_nodes})."
            )
        row = _require_list(row_raw, field_name=f"latent_weights[{g}]")
        if len(row) != n_nodes:
            raise SpecificationError(
                f"latent_weights[{g}] must have {n_nodes} nodes, got {len(row)}."
            )
        for q, val in enumerate(row):
            w = _decode_float(val, field_name=f"latent_weights[{g}][{q}]", finite=True)
            if not (w > 0.0):
                raise SpecificationError(
                    f"latent_weights[{g}][{q}] must be strictly positive, got {w}."
                )
            if w < mw - 1e-15:
                raise SpecificationError(
                    f"latent_weights[{g}][{q}] ({w}) is below min_weight ({mw}) "
                    f"for group {g_name!r}."
                )
            out[g, q] = w
        total = float(np.sum(out[g]))
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise SpecificationError(f"latent_weights[{g}] must sum to 1, got {total}.")
    return out


# --------------------------------------------------------------------------------------------
# Top-level entry points
# --------------------------------------------------------------------------------------------


def dump_fit_result(
    result: FitResult,
    path: str | os.PathLike[str] | None = None,
    *,
    indent: int | None = 2,
) -> str:
    """Serialize ``result`` to strict JSON and optionally write it to ``path``."""
    if indent is not None and (
        isinstance(indent, bool) or not isinstance(indent, int) or indent < 0
    ):
        raise SpecificationError(f"indent must be a non-negative integer or None, got {indent!r}.")
    import flexpyirt

    payload: dict[str, Any] = {
        "format": FORMAT_TAG,
        "schema_version": SCHEMA_VERSION,
        "flexpyirt_version": str(flexpyirt.__version__),
        "model": _serialize_model(result._model),
        "quadrature": _serialize_quadrature(result._quadrature),
        "method": _serialize_method(result._method),
        "n_obs": int(result.n_obs),
        "se_method": result.se_method,
        "loglik": _encode_float(result.loglik),
        "log_posterior": _encode_float(result.log_posterior),
        "converged": bool(result.converged),
        "n_iter": int(result.n_iter),
        "n_em_evaluations": int(result.n_em_evaluations),
        "n_safeguards": int(result.n_safeguards),
        "parameters": _serialize_parameters(result.parameters),
        "vcov": _serialize_vcov(result.vcov),
        "history": _serialize_history(result.history),
    }
    if result._eh_weights is not None:
        payload["latent_weights"] = _serialize_eh_weights(result._eh_weights)
    text = json.dumps(payload, indent=indent, allow_nan=False)
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text


def _read_utf8_file(path: Path) -> str:
    """Read a UTF-8 file, translating invalid UTF-8 bytes into :class:`SpecificationError`."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise SpecificationError(f"FitResult JSON file {str(path)!r} is not valid UTF-8.") from exc


def _read_source(source: str | os.PathLike[str]) -> str:
    """Read raw JSON text from either a file path or a JSON string."""
    if isinstance(source, os.PathLike):
        return _read_utf8_file(Path(source))
    if not isinstance(source, str):
        raise SpecificationError(
            f"from_json() expects a file path or JSON string, got {type(source).__name__}."
        )
    stripped = source.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return source
    has_newline = "\n" in source or "\r" in source
    if not has_newline:
        candidate = Path(source)
        try:
            is_file = candidate.is_file()
        except OSError:
            is_file = False
        if is_file:
            return _read_utf8_file(candidate)
        looks_like_path = (
            os.sep in source
            or (os.altsep is not None and os.altsep in source)
            or source.lower().endswith(".json")
        )
        if looks_like_path:
            return _read_utf8_file(candidate)
    return source


def load_fit_result(source: str | os.PathLike[str]) -> FitResult:
    """Load a :class:`~flexpyirt.FitResult` from a JSON file path or JSON string."""
    check_x64()
    text = _read_source(source)
    try:
        raw = json.loads(
            text,
            parse_constant=_reject_json_constant,
            parse_float=_parse_json_float,
            parse_int=_parse_json_int,
        )
    except json.JSONDecodeError as exc:
        raise SpecificationError(f"Invalid FitResult JSON: {exc}.") from exc

    obj = _require_dict(raw, field_name="FitResult")

    fmt = obj.get("format")
    if fmt != FORMAT_TAG:
        raise SpecificationError(f"Unsupported format {fmt!r}; expected {FORMAT_TAG!r}.")
    if "schema_version" not in obj:
        raise SpecificationError("Missing 'schema_version' in FitResult JSON.")
    schema_version = _require_int(obj["schema_version"], field_name="schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise SpecificationError(
            f"Unsupported FitResult schema_version {schema_version}; "
            f"supported versions are {sorted(SUPPORTED_SCHEMA_VERSIONS)}."
        )
    allowed_keys = _TOP_LEVEL_KEYS | {"latent_weights"}
    keys = set(obj.keys())
    missing_top = sorted(_TOP_LEVEL_KEYS - keys)
    if missing_top:
        raise SpecificationError(f"Missing required keys in FitResult: {missing_top}.")
    extra_top = sorted(keys - allowed_keys)
    if extra_top:
        raise SpecificationError(f"Unexpected keys in FitResult: {extra_top}.")
    _require_str(obj["flexpyirt_version"], field_name="flexpyirt_version")
    if schema_version < 4:
        obj = _upgrade_legacy_names(obj)

    model = _deserialize_model(obj["model"], schema_version=schema_version)
    quadrature = _deserialize_quadrature(obj["quadrature"], field_name="quadrature")
    eh_weights: np.ndarray | None = None
    if model.is_empirical_histogram:
        if "latent_weights" not in obj:
            raise SpecificationError(
                "Missing required key 'latent_weights' for an EmpiricalHistogram FitResult."
            )
        eh_weights = _deserialize_eh_weights(
            obj["latent_weights"],
            model=model,
            n_nodes=quadrature.n_points,
        )
    elif "latent_weights" in obj:
        raise SpecificationError(
            "Unexpected key 'latent_weights' for a non-EmpiricalHistogram FitResult."
        )
    method = _deserialize_method(obj["method"], field_name="method")
    n_obs = _require_int(obj["n_obs"], field_name="n_obs", min_value=1, max_value=_INT64_MAX)
    se_method = obj["se_method"]
    if se_method not in _VALID_SE_METHODS:
        raise SpecificationError(
            f"Unknown se_method {se_method!r}; expected one of {_VALID_SE_METHODS}."
        )
    loglik = _decode_float(obj["loglik"], field_name="loglik")
    log_posterior = _decode_float(obj["log_posterior"], field_name="log_posterior")
    converged = _require_bool(obj["converged"], field_name="converged")
    n_iter = _require_int(obj["n_iter"], field_name="n_iter", min_value=0, max_value=_INT32_MAX)
    n_em_evaluations = _require_int(
        obj["n_em_evaluations"],
        field_name="n_em_evaluations",
        min_value=0,
        max_value=_INT32_MAX,
    )
    n_safeguards = _require_int(
        obj["n_safeguards"], field_name="n_safeguards", min_value=0, max_value=_INT32_MAX
    )

    parameters, eta = _deserialize_parameters(obj["parameters"], model)
    n_free = int(eta.shape[0])
    vcov = _deserialize_vcov(obj["vcov"], n_free=n_free)
    expect_vcov_none = se_method is None or n_free == 0
    if (vcov is None) != expect_vcov_none:
        if vcov is None:
            raise SpecificationError(
                f"vcov cannot be null when se_method={se_method!r} and n_free={n_free}."
            )
        if se_method is None:
            raise SpecificationError("vcov must be null when se_method is null.")
        raise SpecificationError("vcov must be null when there are no free parameters.")
    if vcov is None and not np.isnan(parameters["se"].to_numpy(dtype=float)).all():
        raise SpecificationError("Parameter standard errors must be NaN when vcov is null.")
    history = _deserialize_history(
        obj["history"],
        n_iter=n_iter,
        n_em_evaluations=n_em_evaluations,
        n_safeguards=n_safeguards,
    )

    from flexpyirt.result import FitResult

    return FitResult(
        parameters=parameters,
        loglik=loglik,
        log_posterior=log_posterior,
        converged=converged,
        n_iter=n_iter,
        n_em_evaluations=n_em_evaluations,
        n_safeguards=n_safeguards,
        history=history,
        vcov=vcov,
        se_method=se_method,
        n_obs=n_obs,
        _model=model,
        _eta=eta,
        _quadrature=quadrature,
        _method=method,
        _eh_weights=eh_weights,
    )
