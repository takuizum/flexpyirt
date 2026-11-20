"""Model specification (Spec layer, DESIGN.md §2 ①)."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

import pandas as pd

from flexpyirt.errors import SpecificationError
from flexpyirt.items import ItemModel, Rasch
from flexpyirt.latent import Normal

__all__ = ["Model"]


@dataclass(frozen=True)
class Model:
    r"""Immutable specification of an IRT model.

    Build with :meth:`uniform` or :meth:`from_items`. Builder methods return a new ``Model``; an
    existing instance is never mutated.

    Attributes
    ----------
    items : tuple of (name, ItemModel)
        Items in order.
    fixed : tuple of (item, param, value)
        Parameters fixed by :meth:`fix`.
    latent : Normal or None
        Latent distribution. ``None`` means the default: :math:`N(0,\sigma^2)` with the variance
        estimated when every item is Rasch, otherwise :math:`N(0,1)` (DESIGN.md §4.5).
    """

    items: tuple[tuple[str, ItemModel], ...]
    fixed: tuple[tuple[str, str, float], ...] = ()
    latent: Normal | None = None

    def __post_init__(self) -> None:
        """Validate item names."""
        if not self.items:
            raise SpecificationError("A model needs at least one item.")
        names = [name for name, _ in self.items]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise SpecificationError(f"Duplicated item names: {dup}.")

    # ---- constructors ------------------------------------------------------------------------

    @classmethod
    def uniform(cls, item_model: ItemModel, *, items: Iterable[Any]) -> Model:
        """Return a model where every item in ``items`` uses ``item_model``."""
        return cls(items=tuple((str(name), item_model) for name in items))

    @classmethod
    def from_items(cls, items: Mapping[str, ItemModel]) -> Model:
        """Return a model from a mapping ``item name -> item model`` (order is kept)."""
        return cls(items=tuple((str(name), m) for name, m in items.items()))

    # ---- builders ----------------------------------------------------------------------------

    @property
    def item_names(self) -> tuple[str, ...]:
        """Item names in order."""
        return tuple(name for name, _ in self.items)

    def fix(self, *, item: str, **values: float) -> Model:
        """Return a model with parameters of ``item`` fixed, e.g. ``fix(item="q01", a=1.2)``.

        Values use the internal (slope–intercept) parameterization.
        """
        lookup = dict(self.items)
        if item not in lookup:
            raise SpecificationError(f"Cannot fix unknown item {item!r}.")
        if not values:
            raise SpecificationError("fix() needs at least one parameter value.")
        params = lookup[item].kernel.param_names
        new = {(i, p): v for i, p, v in self.fixed}
        for param, value in values.items():
            if param not in params:
                raise SpecificationError(
                    f"Item {item!r} ({lookup[item].name}) has no parameter {param!r}; "
                    f"available: {list(params)}."
                )
            if not math.isfinite(float(value)):
                raise SpecificationError(f"Fixed value for {item!r}.{param} must be finite.")
            new[(item, param)] = float(value)
        return replace(self, fixed=tuple((i, p, v) for (i, p), v in new.items()))

    def with_latent(self, dist: Normal) -> Model:
        """Return a model with latent distribution ``dist``."""
        if not isinstance(dist, Normal):
            raise SpecificationError(f"Unsupported latent distribution {dist!r}.")
        return replace(self, latent=dist)

    def resolved_latent(self) -> Normal:
        """Return the latent distribution, applying the default rule of DESIGN.md §4.5."""
        if self.latent is not None:
            return self.latent
        if all(isinstance(m, Rasch) for _, m in self.items):
            return Normal(free_var=True)
        return Normal.standard()

    def parameter_table(
        self, data: Any, *, item_names: Iterable[str] | None = None
    ) -> pd.DataFrame:
        """Return the default parameter table (DESIGN.md §5.1) for ``data``.

        Edit ``value``, ``free``, ``lower``, ``upper`` or ``free_id`` and pass it back with
        ``fit(..., table=edited)``.
        """
        from flexpyirt._compile import build_table, coerce_responses, validate_table

        y = coerce_responses(
            data, self.item_names, item_names=None if item_names is None else list(item_names)
        )
        table = build_table(self, y)
        return validate_table(table, table)
