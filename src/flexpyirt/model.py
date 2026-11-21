"""Model specification (Spec layer, DESIGN.md §2 ①)."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

import pandas as pd

from flexpyirt.errors import SpecificationError
from flexpyirt.items import PCM, GPCMKernel, ItemModel, Rasch
from flexpyirt.latent import Normal
from flexpyirt.priors import PRIOR_TYPES, Prior

__all__ = ["LATENT_ITEM", "SINGLE_GROUP", "Model"]

LATENT_ITEM = "__latent__"
SINGLE_GROUP = "all"

ItemSelector = Iterable[Any] | Callable[[str], bool] | None


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
        Parameters fixed in every group by :meth:`fix` or :meth:`fix_from`.
    latent : Normal or None
        Latent distribution of a single-group model. ``None`` means the default:
        :math:`N(0,\sigma^2)` with the variance estimated when every item is Rasch or PCM,
        otherwise :math:`N(0,1)` (DESIGN.md §4.5, §15.8).
    groups : tuple of (group, Normal) or None
        Groups and their latent distributions for a multiple-group model; the first is the
        reference group (DESIGN.md §14.4).
    equal : tuple of (items, params or None)
        Across-group equality constraints from :meth:`equal_across_groups`.
    priors : tuple of (item, param, Prior)
        Prior distributions on item parameters from :meth:`prior` (applied across all groups;
        shared parameters are counted once and fixed parameters ignore their prior).
    """

    items: tuple[tuple[str, ItemModel], ...]
    fixed: tuple[tuple[str, str, float], ...] = ()
    latent: Normal | None = None
    groups: tuple[tuple[str, Normal], ...] | None = None
    equal: tuple[tuple[tuple[str, ...], tuple[str, ...] | None], ...] = ()
    priors: tuple[tuple[str, str, Prior], ...] = ()

    def __post_init__(self) -> None:
        """Validate item names."""
        if not self.items:
            raise SpecificationError("A model needs at least one item.")
        names = [name for name, _ in self.items]
        if LATENT_ITEM in names:
            raise SpecificationError(f"Item name {LATENT_ITEM!r} is reserved.")
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

    # ---- queries -----------------------------------------------------------------------------

    @property
    def item_names(self) -> tuple[str, ...]:
        """Item names in order."""
        return tuple(name for name, _ in self.items)

    @property
    def is_multigroup(self) -> bool:
        """Whether groups were declared with :meth:`with_groups`."""
        return self.groups is not None

    def resolved_latent(self) -> Normal:
        """Return the single-group latent distribution, applying DESIGN.md §4.5's default rule."""
        if self.latent is not None:
            return self.latent
        if all(isinstance(m, (Rasch, PCM)) for _, m in self.items):
            return Normal(free_var=True)
        return Normal.standard()

    def resolved_groups(self) -> tuple[tuple[str, Normal], ...]:
        """Return ``(group, latent)`` pairs; a single-group model has one group named ``"all"``."""
        if self.groups is None:
            return ((SINGLE_GROUP, self.resolved_latent()),)
        return self.groups

    def _select(self, items: ItemSelector) -> tuple[str, ...]:
        if items is None:
            return self.item_names
        if callable(items):
            return tuple(n for n in self.item_names if items(n))
        chosen = tuple(str(i) for i in items)
        unknown = [i for i in chosen if i not in self.item_names]
        if unknown:
            raise SpecificationError(f"Unknown items: {unknown}.")
        return chosen

    # ---- builders ----------------------------------------------------------------------------

    def fix(self, *, item: str, **values: float) -> Model:
        """Return a model with parameters of ``item`` fixed in every group.

        Values use the internal (slope–intercept) parameterization, e.g.
        ``fix(item="q01", a=1.2, d=0.3)``.
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

    def fix_from(
        self,
        bank: pd.DataFrame,
        *,
        items: ItemSelector = None,
        parameterization: Literal["slope-intercept", "irt"] = "slope-intercept",
    ) -> Model:
        r"""Return a model with item parameters fixed from an item bank (FIPC, DESIGN.md §14.6).

        Parameters
        ----------
        bank : DataFrame
            One row per item with an ``item`` column and parameter columns: ``a``, ``d`` (or
            ``d1..``) for ``"slope-intercept"``; ``a``, ``b`` (or ``b1..``) for ``"irt"``,
            converted with :math:`d=-ab` (or :math:`d_k=-a\sum_{c=1}^{k} b_c` for GPCM/PCM step
            difficulties). ``a`` may be omitted (or NaN) for Rasch and PCM items, or when already
            fixed on the model via :meth:`fix` (whose fixed slope is used to convert ``b`` to
            ``d``). Asymptotes ``g`` (3PL/4PL) and ``u`` (4PL) are given as they are in both
            parameterizations; ``u`` may be omitted (or NaN) for 3PL items (it stays 1), and any
            asymptote already fixed via :meth:`fix` may also be omitted (or NaN). NaN cells are
            treated as not given.
        items : list of str or callable, optional
            Items to fix (default: every bank item that is in the model).
        parameterization : {"slope-intercept", "irt"}
        """
        if "item" not in bank.columns:
            raise SpecificationError("The item bank needs an 'item' column.")
        rows = {str(r["item"]): r for r in bank.to_dict("records")}
        chosen = (
            tuple(i for i in self.item_names if i in rows) if items is None else self._select(items)
        )
        missing = [i for i in chosen if i not in rows]
        if missing:
            raise SpecificationError(f"Items not found in the bank: {missing}.")

        def _has_value(row: Mapping[Any, Any], key: str) -> bool:
            return key in row and bool(pd.notna(row[key]))

        def _missing_message(item: str, row: Mapping[Any, Any], key: str) -> str:
            if key not in row:
                return f"Bank row for {item!r} has no column {key!r}."
            return f"Bank row for {item!r} has missing (NaN) value in column {key!r}."

        lookup = dict(self.items)
        prior_fixed = {(i, p): v for i, p, v in self.fixed}
        model = self
        for item in chosen:
            row = rows[item]
            item_model = lookup[item]
            kernel = item_model.kernel
            slope = float(row["a"]) if _has_value(row, "a") else prior_fixed.get((item, "a"), 1.0)
            is_gpcm = isinstance(kernel, GPCMKernel)
            cum_b = 0.0
            values: dict[str, float] = {}
            for p, free_default in zip(kernel.param_names, item_model.default_free(), strict=True):
                if p in kernel.slope_params:
                    if _has_value(row, p):
                        values[p] = float(row[p])
                    elif free_default and (item, p) not in prior_fixed:
                        raise SpecificationError(_missing_message(item, row, p))
                    continue
                if p in kernel.intercept_params:
                    if parameterization == "slope-intercept":
                        key = p
                    elif parameterization == "irt":
                        key = "b" + p[1:]
                    else:
                        raise SpecificationError(f"Unknown parameterization {parameterization!r}.")
                    if not _has_value(row, key):
                        raise SpecificationError(_missing_message(item, row, key))
                    value = float(row[key])
                    if parameterization == "slope-intercept":
                        values[p] = value
                    elif is_gpcm:
                        cum_b += value
                        values[p] = -slope * cum_b
                    else:
                        values[p] = -slope * value
                    continue
                # Other parameters (asymptotes g, u) are the same in both parameterizations. A
                # parameter fixed by default (u of a 3PL item) or already fixed via fix() may be
                # omitted or NaN.
                if _has_value(row, p):
                    values[p] = float(row[p])
                elif free_default and (item, p) not in prior_fixed:
                    raise SpecificationError(_missing_message(item, row, p))
            model = model.fix(item=item, **values)
        return model

    def with_latent(self, dist: Normal) -> Model:
        """Return a single-group model with latent distribution ``dist``."""
        if not isinstance(dist, Normal):
            raise SpecificationError(f"Unsupported latent distribution {dist!r}.")
        if self.groups is not None:
            raise SpecificationError("This model has groups; set latents with with_groups().")
        return replace(self, latent=dist)

    def with_groups(self, **latents: Normal) -> Model:
        """Return a multiple-group model; keyword order defines the groups, the first is reference.

        Example: ``with_groups(ref=Normal.standard(), focal=Normal.free())``.
        """
        if len(latents) < 2:
            raise SpecificationError("with_groups() needs at least two groups.")
        for name, dist in latents.items():
            if not isinstance(dist, Normal):
                raise SpecificationError(f"Unsupported latent distribution for group {name!r}.")
        if self.latent is not None:
            raise SpecificationError("Use with_groups() instead of with_latent() for groups.")
        return replace(self, groups=tuple(latents.items()))

    def equal_across_groups(
        self, *, items: ItemSelector, params: Iterable[str] | None = None
    ) -> Model:
        """Return a model whose ``params`` (default: all) of ``items`` are equal across groups.

        These are the anchor items that link the groups' scales (DESIGN.md §4.5, §14.1).
        """
        chosen = self._select(items)
        if not chosen:
            raise SpecificationError("equal_across_groups() selected no items.")
        param_tuple = None if params is None else tuple(params)
        if param_tuple is not None:
            lookup = dict(self.items)
            for item in chosen:
                bad = [p for p in param_tuple if p not in lookup[item].kernel.param_names]
                if bad:
                    raise SpecificationError(f"Item {item!r} has no parameters {bad}.")
        return replace(self, equal=(*self.equal, (chosen, param_tuple)))

    def prior(
        self,
        *,
        param: str,
        dist: Prior,
        items: ItemSelector = None,
    ) -> Model:
        """Return a model with prior ``dist`` attached to ``param`` of the selected items.

        The prior is evaluated on the natural parameter scale without a Jacobian term
        (DESIGN.md §4.7). In a multiple-group model it applies to every group; a parameter shared
        across groups (same ``free_id``) is counted once, and a fixed parameter ignores its prior.
        Priors on latent-distribution parameters (``"mean"``, ``"var"``) are not supported.

        Parameters
        ----------
        param : str
            Internal item parameter name (e.g. ``"a"``, ``"d"``, ``"g"``, ``"u"``, ``"d1"``).
        dist : NormalPrior, LogNormal, or Beta
            Prior distribution on the natural parameter scale.
        items : iterable of str or callable, optional
            Items to attach the prior to (default: every item in the model).
        """
        if not isinstance(dist, PRIOR_TYPES):
            raise SpecificationError(
                f"Unsupported prior distribution {dist!r}; use NormalPrior, LogNormal or Beta."
            )
        if param in ("mean", "var"):
            raise SpecificationError(
                f"Priors on latent-distribution parameters ({param!r}) are not supported."
            )
        chosen = self._select(items)
        if not chosen:
            raise SpecificationError("prior() selected no items.")
        lookup = dict(self.items)
        for item in chosen:
            params = lookup[item].kernel.param_names
            if param not in params:
                raise SpecificationError(
                    f"Item {item!r} ({lookup[item].name}) has no parameter {param!r}; "
                    f"available: {list(params)}."
                )
        new = {(i, p): d for i, p, d in self.priors}
        for item in chosen:
            new[(item, param)] = dist
        return replace(self, priors=tuple((i, p, d) for (i, p), d in new.items()))

    def parameter_table(
        self, data: Any, *, group: Any = None, item_names: Iterable[str] | None = None
    ) -> pd.DataFrame:
        """Return the default parameter table (DESIGN.md §5.1) for ``data``.

        Edit ``value``, ``free``, ``lower``, ``upper``, ``prior`` or ``free_id`` and pass it back
        with ``fit(..., table=edited)``.
        """
        from flexpyirt._compile import prepare

        return prepare(
            self,
            data,
            group=group,
            item_names=None if item_names is None else list(item_names),
            table=None,
        ).table
