"""Model specification (Spec layer, DESIGN.md §2 ①)."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

import pandas as pd

from flexpyirt.errors import SpecificationError
from flexpyirt.items import PCM, GPCMKernel, ItemKernel, ItemModel, NominalKernel, Rasch
from flexpyirt.latent import (
    DavidianCurve,
    EmpiricalHistogram,
    LatentSpec,
    Normal,
    NormalMixture,
    latent_param_names,
)
from flexpyirt.priors import PRIOR_TYPES, Prior
from flexpyirt.quadrature import MAX_FACTORS, MAX_KERNEL_FACTORS

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
    equal_items : tuple of (items, params)
        Across-item equality constraints from :meth:`equal_across_items` (each parameter in
        ``params`` is shared by ``items`` within every group; DESIGN.md §16.1).
    priors : tuple of (item, param, Prior)
        Prior distributions on item parameters from :meth:`prior` (applied across all groups;
        shared parameters are counted once and fixed parameters ignore their prior).
    factors : tuple of (factor, tuple of item names) or None
        Confirmatory loading pattern from :meth:`with_factors` for 2–3 dimensional models.
        ``None`` means unidimensional.
    correlated_factors : bool
        Whether factor correlations are free by default when ``factors`` is set.
    bifactor : bool
        Whether ``factors`` is a bifactor structure from :meth:`with_bifactor` (the first factor
        is the general factor; the E-step uses Gibbons–Hedeker dimension reduction, DESIGN.md
        §17.1).
    n_primary_factors : int
        Number of leading primary factors of a two-tier structure from :meth:`with_two_tier`
        (DESIGN.md §17.2); 0 for every other model.
    """

    items: tuple[tuple[str, ItemModel], ...]
    fixed: tuple[tuple[str, str, float], ...] = ()
    latent: LatentSpec | None = None
    groups: tuple[tuple[str, LatentSpec], ...] | None = None
    equal: tuple[tuple[tuple[str, ...], tuple[str, ...] | None], ...] = ()
    equal_items: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = ()
    priors: tuple[tuple[str, str, Prior], ...] = ()
    factors: tuple[tuple[str, tuple[str, ...]], ...] | None = None
    correlated_factors: bool = True
    bifactor: bool = False
    n_primary_factors: int = 0

    def __post_init__(self) -> None:
        """Validate item names and group latent-distribution consistency."""
        if not self.items:
            raise SpecificationError("A model needs at least one item.")
        if self.bifactor and self.factors is None:
            raise SpecificationError("A bifactor model needs factors; use with_bifactor().")
        if self.n_primary_factors < 0 or (self.bifactor and self.n_primary_factors):
            raise SpecificationError(
                "n_primary_factors must be 0 for bifactor models and non-negative otherwise."
            )
        if self.n_primary_factors and (
            self.factors is None or len(self.factors) <= self.n_primary_factors
        ):
            raise SpecificationError(
                "A two-tier model needs primary and specific factors; use with_two_tier()."
            )
        if self.is_reduced and self.groups is not None:
            raise SpecificationError(
                "Bifactor and two-tier models do not support multiple groups yet (DESIGN.md "
                "§17.1, §17.2)."
            )
        names = [name for name, _ in self.items]
        if LATENT_ITEM in names:
            raise SpecificationError(f"Item name {LATENT_ITEM!r} is reserved.")
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise SpecificationError(f"Duplicated item names: {dup}.")
        if self.groups is not None:
            kinds = {type(dist) for _, dist in self.groups}
            for single_only in (DavidianCurve, NormalMixture):
                if single_only in kinds:
                    raise SpecificationError(
                        f"{single_only.__name__} is only supported for single-group models; use "
                        "Model.with_latent() instead of with_groups()."
                    )
            if EmpiricalHistogram in kinds and len(kinds) > 1:
                raise SpecificationError(
                    "Cannot mix EmpiricalHistogram and Normal across groups; when "
                    "EmpiricalHistogram is specified, every group must use EmpiricalHistogram."
                )

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

    @property
    def is_empirical_histogram(self) -> bool:
        """Whether this model uses an :class:`~flexpyirt.EmpiricalHistogram` latent distribution."""
        return isinstance(self.resolved_groups()[0][1], EmpiricalHistogram)

    @property
    def is_davidian(self) -> bool:
        """Whether this model uses a :class:`~flexpyirt.DavidianCurve` latent distribution."""
        return isinstance(self.resolved_groups()[0][1], DavidianCurve)

    @property
    def davidian_degree(self) -> int:
        """Polynomial degree of the Davidian curve, or ``0`` for non-Davidian models."""
        dist = self.resolved_groups()[0][1]
        return dist.degree if isinstance(dist, DavidianCurve) else 0

    @property
    def is_mixture(self) -> bool:
        """Whether this model uses a :class:`~flexpyirt.NormalMixture` latent distribution."""
        return isinstance(self.resolved_groups()[0][1], NormalMixture)

    @property
    def mixture_components(self) -> int:
        """Number of normal-mixture components, or ``0`` for non-mixture models."""
        dist = self.resolved_groups()[0][1]
        return dist.n_components if isinstance(dist, NormalMixture) else 0

    @property
    def is_multidimensional(self) -> bool:
        """Whether confirmatory factors were declared with :meth:`with_factors`."""
        return self.factors is not None

    @property
    def n_factors(self) -> int:
        """Number of latent factors (1 for unidimensional, 2 or 3 with :meth:`with_factors`)."""
        return 1 if self.factors is None else len(self.factors)

    @property
    def factor_names(self) -> tuple[str, ...]:
        """Declared factor names in order (empty tuple for a unidimensional model)."""
        return () if self.factors is None else tuple(name for name, _ in self.factors)

    @property
    def is_reduced(self) -> bool:
        """Whether the E-step uses dimension reduction (bifactor or two-tier; DESIGN.md §17)."""
        return self.bifactor or self.n_primary_factors > 0

    @property
    def n_primary(self) -> int:
        """Number of primary factors of a reduced model.

        1 for bifactor (the general factor), ``n_primary_factors`` for two-tier, and 0 for
        models without dimension reduction.
        """
        return 1 if self.bifactor else self.n_primary_factors

    @property
    def quadrature_dimension(self) -> int:
        """Dimension of the quadrature grid the E-step integrates over.

        ``n_factors`` for product grids; ``n_primary + 1`` for bifactor and two-tier models,
        whose E-step uses the grid of the primary factors and one specific factor (DESIGN.md
        §17.1, §17.2).
        """
        return self.n_primary + 1 if self.is_reduced else self.n_factors

    def latent_parameter_names(self) -> tuple[str, ...]:
        """Return the names of the Normal latent parameters of each group (DESIGN.md §15.30).

        ``("mean", "var")`` for one factor, ``(mean_1..D, var_1..D, cor_ij...)`` for 2–3 factors,
        ``(mean_1..D, var_1..D)`` for bifactor models, whose factors are orthogonal
        (DESIGN.md §17.1), and ``(mean_1..D, var_1..D, cor_ij...)`` with correlations among the
        primary factors only (``i < j <= P``) for two-tier models (DESIGN.md §17.2).
        """
        if self.is_reduced:
            d = self.n_factors
            n_primary = self.n_primary
            return (
                *(f"mean_{m}" for m in range(1, d + 1)),
                *(f"var_{m}" for m in range(1, d + 1)),
                *(
                    f"cor_{i}{j}"
                    for i in range(1, n_primary + 1)
                    for j in range(i + 1, n_primary + 1)
                ),
            )
        return latent_param_names(self.n_factors)

    def item_kernel(self, item_model: ItemModel) -> ItemKernel:
        """Return ``item_model``'s kernel adapted to this model's dimensionality."""
        if self.factors is None:
            return item_model.kernel
        return item_model.kernel.with_factors(len(self.factors))

    def item_factors(self, item: str) -> tuple[int, ...]:
        """Return 0-based factor indices that ``item`` loads on."""
        if self.factors is None:
            return (0,)
        return tuple(idx for idx, (_, assigned) in enumerate(self.factors) if item in assigned)

    def resolved_latent(self) -> LatentSpec:
        """Return the single-group latent distribution, applying DESIGN.md §4.5's default rule."""
        if self.latent is not None:
            return self.latent
        if all(isinstance(m, (Rasch, PCM)) for _, m in self.items):
            return Normal(free_var=True)
        return Normal.standard()

    def resolved_groups(self) -> tuple[tuple[str, LatentSpec], ...]:
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
        params = self.item_kernel(lookup[item]).param_names
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
        parameterization: Literal["slope_intercept", "irt"] = "slope_intercept",
    ) -> Model:
        r"""Return a model with item parameters fixed from an item bank (FIPC, DESIGN.md §14.6).

        Parameters
        ----------
        bank : DataFrame
            One row per item with an ``item`` column and parameter columns: ``a``, ``d`` (or
            ``d1..``) for ``"slope_intercept"``; ``a``, ``b`` (or ``b1..``) for ``"irt"``,
            converted with :math:`d=-ab` (or :math:`d_k=-a\sum_{c=1}^{k} b_c` for GPCM/PCM step
            difficulties). ``a`` may be omitted (or NaN) for Rasch and PCM items, or when already
            fixed on the model via :meth:`fix` (whose fixed slope is used to convert ``b`` to
            ``d``). Asymptotes ``g`` (3PL/4PL) and ``u`` (4PL) are given as they are in both
            parameterizations; ``u`` may be omitted (or NaN) for 3PL items (it stays 1), and any
            asymptote already fixed via :meth:`fix` may also be omitted (or NaN). ``Nominal``
            items take ``a``, ``ak1..ak{K-2}``, and ``d1..d{K-1}`` and support only
            ``"slope_intercept"`` (DESIGN.md §16.6). NaN cells are treated as not given.
        items : list of str or callable, optional
            Items to fix (default: every bank item that is in the model).
        parameterization : {"slope_intercept", "irt"}

        Raises
        ------
        SpecificationError
            If the bank lacks a needed column or item, if ``parameterization="irt"`` is used
            with a multidimensional model, or if ``parameterization="irt"`` is used for a
            ``Nominal`` item (the nominal model has no IRT thresholds).
        """
        if self.is_multidimensional and parameterization == "irt":
            raise SpecificationError(
                "parameterization='irt' is only supported for unidimensional models; "
                "use parameterization='slope-intercept' for multidimensional models."
            )
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
            kernel = self.item_kernel(item_model)
            if parameterization == "irt" and isinstance(kernel, NominalKernel):
                raise SpecificationError(
                    f"parameterization='irt' is not supported for Nominal item {item!r}; "
                    "use parameterization='slope-intercept'."
                )
            slope = float(row["a"]) if _has_value(row, "a") else prior_fixed.get((item, "a"), 1.0)
            is_gpcm = isinstance(kernel, GPCMKernel)
            cum_b = 0.0
            base_free = item_model.default_free()
            loaded_set = set(self.item_factors(item))
            free_defaults = (
                tuple(m in loaded_set and base_free[0] for m in range(self.n_factors))
                + base_free[1:]
            )
            values: dict[str, float] = {}
            for p, free_default in zip(kernel.param_names, free_defaults, strict=True):
                if p in kernel.slope_params:
                    if _has_value(row, p):
                        values[p] = float(row[p])
                    elif free_default and (item, p) not in prior_fixed:
                        raise SpecificationError(_missing_message(item, row, p))
                    continue
                if p in kernel.intercept_params:
                    if parameterization == "slope_intercept":
                        key = p
                    elif parameterization == "irt":
                        key = "b" + p[1:]
                    else:
                        raise SpecificationError(f"Unknown parameterization {parameterization!r}.")
                    if not _has_value(row, key):
                        raise SpecificationError(_missing_message(item, row, key))
                    value = float(row[key])
                    if parameterization == "slope_intercept":
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

    def with_factors(
        self,
        factors: Mapping[str, ItemSelector],
        *,
        correlated: bool = True,
    ) -> Model:
        r"""Return a confirmatory multidimensional model with 2 or 3 latent factors (DESIGN §4.9).

        Each entry ``factor_name: items`` declares which items load on that factor; an item may
        load on multiple factors (cross-loading), and slopes on factors an item does not load on
        are fixed at 0. Item parameters become ``a1``..``aD`` and ``d`` (or ``d1``.. for GRM /
        GPCM / PCM), matching mirt (Chalmers, 2012).

        The latent distribution is multivariate normal with means fixed at 0 and variances fixed
        at 1 (or variances free when every item is :class:`Rasch` or :class:`PCM`), and
        correlations free by default when ``correlated=True`` (as in mirt's ``COV = F1*F2``) or
        fixed at 0 when ``correlated=False``.

        Parameters
        ----------
        factors : mapping of str to iterable of item names or callable
            Factor names (in order, 2 or 3 factors) mapped to the items loading on each factor.
            Every factor must have at least one item, and every model item must load on at least
            one factor.
        correlated : bool, default True
            Whether off-diagonal factor correlations are estimated by default.

        Raises
        ------
        SpecificationError
            If the number of factors is outside ``{2, 3}``, factor names are empty or duplicated,
            any factor is empty or lists unknown/duplicated items, any model item loads on no
            factor, or ``with_groups()`` / ``with_latent()`` is already set.
        """
        if self.is_reduced:
            raise SpecificationError(
                "This model is a bifactor or two-tier model; with_factors() cannot be combined "
                "with with_bifactor() or with_two_tier()."
            )
        if self.groups is not None and any(
            isinstance(dist, EmpiricalHistogram) for _, dist in self.groups
        ):
            raise SpecificationError(
                "EmpiricalHistogram is only supported for unidimensional models."
            )
        if self.latent is not None:
            raise SpecificationError(
                "with_latent() is only supported for unidimensional models; configure factor "
                "correlations with with_factors(..., correlated=...) or edit the ParameterTable."
            )
        if not isinstance(factors, Mapping):
            raise SpecificationError("with_factors() requires a mapping of factor name -> items.")
        n_factors = len(factors)
        if not (2 <= n_factors <= MAX_FACTORS):
            raise SpecificationError(
                f"with_factors() supports 2 to {MAX_FACTORS} factors, got {n_factors}."
            )
        resolved: list[tuple[str, tuple[str, ...]]] = []
        seen_factors: set[str] = set()
        assigned_all: set[str] = set()
        for raw_name, selector in factors.items():
            name = str(raw_name)
            if not name.strip():
                raise SpecificationError("Factor names must be non-empty strings.")
            if name in seen_factors:
                raise SpecificationError(f"Duplicated factor name: {name!r}.")
            seen_factors.add(name)
            if isinstance(selector, str):
                raise SpecificationError(
                    f"Factor {name!r} items must be an iterable of item names or a callable, "
                    f"not a single string {selector!r}."
                )
            chosen = self._select(selector)
            if not chosen:
                raise SpecificationError(f"Factor {name!r} must have at least one item.")
            if len(set(chosen)) != len(chosen):
                dup = sorted({i for i in chosen if chosen.count(i) > 1})
                raise SpecificationError(f"Factor {name!r} lists duplicated items: {dup}.")
            resolved.append((name, chosen))
            assigned_all.update(chosen)

        unassigned = [i for i in self.item_names if i not in assigned_all]
        if unassigned:
            raise SpecificationError(f"Items not assigned to any factor: {unassigned}.")

        candidate = replace(self, factors=tuple(resolved), correlated_factors=bool(correlated))
        self._check_parameter_names(candidate, "with_factors")
        return candidate

    def _check_parameter_names(self, candidate: Model, builder: str) -> None:
        """Check that fixed values, priors and equalities name parameters of ``candidate``."""
        n_factors = candidate.n_factors
        lookup = dict(self.items)
        for item, param, _ in self.fixed:
            valid = candidate.item_kernel(lookup[item]).param_names
            if param not in valid:
                raise SpecificationError(
                    f"Fixed parameter {item!r}.{param} is not valid for a {n_factors}-factor "
                    f"model (available: {list(valid)}); call {builder}() before fix()."
                )
        for item, param, _ in self.priors:
            valid = candidate.item_kernel(lookup[item]).param_names
            if param not in valid:
                raise SpecificationError(
                    f"Prior on {item!r}.{param} is not valid for a {n_factors}-factor model "
                    f"(available: {list(valid)}); call {builder}() before prior()."
                )
        for eq_items, eq_params in self.equal_items:
            for item in eq_items:
                valid = candidate.item_kernel(lookup[item]).param_names
                bad = [p for p in eq_params if p not in valid]
                if bad:
                    raise SpecificationError(
                        f"Equality on {item!r}.{bad} is not valid for a {n_factors}-factor model "
                        f"(available: {list(valid)}); call {builder}() before "
                        "equal_across_items()."
                    )

    def with_bifactor(
        self,
        *,
        specific: Mapping[str, ItemSelector],
        general: str = "general",
    ) -> Model:
        r"""Return a bifactor model: one general factor plus specific factors (DESIGN.md §17.1).

        Every item loads on the general factor and on at most one specific factor; items listed
        under no specific factor load on the general factor only. Factors are orthogonal
        standard normals. Item parameters are ``a1`` (general), ``a2``..``aD`` (specific, in the
        order of ``specific``; slopes on other specific factors are fixed at 0) and ``d`` (or
        ``d1``..), matching the columns of mirt's ``bfactor()`` (Chalmers, 2012).

        The marginal likelihood is integrated by Gibbons and Hedeker's (1992) dimension
        reduction: because item :math:`j` depends on :math:`(\theta_0, \theta_{s(j)})` only,

        .. math::
            L_p=\sum_g w^{(0)}_g\prod_{s}\Big[\sum_t w^{(s)}_t
            \prod_{j\in s}P_j(y_{pj}\mid\theta_g,\theta_t)\Big],

        so each E-step needs one :math:`(1+1)`-dimensional grid per specific factor instead of a
        :math:`(1+S)`-dimensional product grid.

        Parameters
        ----------
        specific : mapping of str to iterable of item names or callable
            Specific factor names (in order, at least one) mapped to their items; each needs at
            least two items and an item may appear under at most one specific factor.
        general : str, default "general"
            Name of the general factor.

        Raises
        ------
        SpecificationError
            If factor names are empty or duplicated, a specific factor has fewer than two items or
            lists unknown items, an item is under two specific factors, the model already has
            factors, groups or a latent distribution, or any item is :class:`Rasch` or
            :class:`PCM` (the factor variances would have to be free; not supported yet).
        """
        if self.factors is not None:
            raise SpecificationError(
                "This model already has factors; call with_bifactor() on a model without "
                "with_factors(), with_bifactor() or with_two_tier()."
            )
        self._check_reducible("with_bifactor", "Bifactor", specific)
        if not isinstance(general, str) or not general.strip():
            raise SpecificationError("The general factor name must be a non-empty string.")
        resolved: list[tuple[str, tuple[str, ...]]] = [(general, self.item_names)]
        resolved.extend(self._resolve_specific(specific, {general}))
        if len(resolved) > MAX_KERNEL_FACTORS:
            raise SpecificationError(
                f"with_bifactor() supports at most {MAX_KERNEL_FACTORS - 1} specific factors, got "
                f"{len(resolved) - 1}."
            )
        candidate = replace(self, factors=tuple(resolved), correlated_factors=False, bifactor=True)
        self._check_parameter_names(candidate, "with_bifactor")
        return candidate

    def _check_reducible(self, builder: str, label: str, specific: object) -> None:
        """Run the checks shared by :meth:`with_bifactor` and :meth:`with_two_tier`."""
        if self.groups is not None:
            raise SpecificationError(
                f"{label} models do not support multiple groups yet (DESIGN.md §17.1, §17.2)."
            )
        if self.latent is not None:
            raise SpecificationError(
                f"with_latent() is not supported for {label.lower()} models; their factors are "
                "standard normals."
            )
        if not isinstance(specific, Mapping) or len(specific) == 0:
            raise SpecificationError(
                f"{builder}() requires a non-empty mapping specific factor name -> items."
            )
        rasch_like = [name for name, m in self.items if isinstance(m, (Rasch, PCM))]
        if rasch_like:
            raise SpecificationError(
                f"{label} models need free slopes; Rasch / PCM items {rasch_like} are not "
                "supported yet (their factor variances would have to be free)."
            )

    def _resolve_specific(
        self, specific: Mapping[str, ItemSelector], seen: set[str]
    ) -> list[tuple[str, tuple[str, ...]]]:
        """Resolve specific factors: at least two items each, each item under at most one."""
        seen = set(seen)
        resolved: list[tuple[str, tuple[str, ...]]] = []
        owner: dict[str, str] = {}
        for raw_name, selector in specific.items():
            name = str(raw_name)
            if not name.strip():
                raise SpecificationError("Factor names must be non-empty strings.")
            if name in seen:
                raise SpecificationError(f"Duplicated factor name: {name!r}.")
            seen.add(name)
            if isinstance(selector, str):
                raise SpecificationError(
                    f"Factor {name!r} items must be an iterable of item names or a callable, "
                    f"not a single string {selector!r}."
                )
            chosen = self._select(selector)
            if len(set(chosen)) != len(chosen):
                dup = sorted({i for i in chosen if chosen.count(i) > 1})
                raise SpecificationError(f"Factor {name!r} lists duplicated items: {dup}.")
            if len(chosen) < 2:
                raise SpecificationError(
                    f"Specific factor {name!r} needs at least two items, got {len(chosen)} "
                    "(a one-item specific slope is not identified)."
                )
            for item in chosen:
                if item in owner:
                    raise SpecificationError(
                        f"Item {item!r} is listed under specific factors {owner[item]!r} and "
                        f"{name!r}; each item loads on at most one specific factor."
                    )
                owner[item] = name
            resolved.append((name, chosen))
        return resolved

    def with_two_tier(
        self,
        *,
        primary: Mapping[str, ItemSelector],
        specific: Mapping[str, ItemSelector],
        correlated: bool = True,
    ) -> Model:
        r"""Return a two-tier model: correlated primary factors plus specific factors (§17.2).

        Every item loads on at least one primary factor (cross-loadings among primary factors
        are allowed) and on at most one specific factor. Primary factors are standard normal
        and may correlate with each other; specific factors are standard normal and orthogonal
        to every other factor. Item parameters are ``a1``..``aP`` (primary, in the order of
        ``primary``), ``a(P+1)``..``aD`` (specific, in the order of ``specific``; slopes an item
        does not load on are fixed at 0) and ``d`` (or ``d1``..), matching the columns of mirt's
        ``bfactor(data, model, model2 = )`` (Chalmers, 2012).

        The marginal likelihood is integrated with Cai's (2010) dimension reduction: with the
        primary product grid :math:`\boldsymbol\theta_g` and its multivariate-normal weights
        :math:`w^{(0)}_g`,

        .. math::
            L_p=\sum_g w^{(0)}_g\prod_{s}\Big[\sum_t w^{(s)}_t
            \prod_{j\in s}P_j(y_{pj}\mid\boldsymbol\theta_g,\theta_t)\Big],

        so each E-step needs one :math:`(P+1)`-dimensional grid per specific factor instead of a
        :math:`(P+S)`-dimensional product grid. A bifactor model (:meth:`with_bifactor`) is the
        case of one primary factor on every item.

        Parameters
        ----------
        primary : mapping of str to iterable of item names or callable
            Primary factor names (in order, 1 or 2 factors) mapped to the items loading on each.
            Every model item must load on at least one primary factor.
        specific : mapping of str to iterable of item names or callable
            Specific factor names (in order, at least one) mapped to their items; each needs at
            least two items and an item may appear under at most one specific factor.
        correlated : bool, default True
            Whether the correlations among primary factors are estimated (mirt's ``COV =
            F1*F2`` in ``model2``); ``False`` fixes them at 0.

        Raises
        ------
        SpecificationError
            If there are no or more than two primary factors, factor names are empty or
            duplicated, a primary factor is empty or lists unknown/duplicated items, an item
            loads on no primary factor, a specific factor has fewer than two items, an item is
            under two specific factors, the model already has factors, groups or a latent
            distribution, or any item is :class:`Rasch` or :class:`PCM`.
        """
        if self.factors is not None:
            raise SpecificationError(
                "This model already has factors; call with_two_tier() on a model without "
                "with_factors(), with_bifactor() or with_two_tier()."
            )
        self._check_reducible("with_two_tier", "Two-tier", specific)
        if not isinstance(primary, Mapping) or len(primary) == 0:
            raise SpecificationError(
                "with_two_tier() requires a non-empty mapping primary factor name -> items."
            )
        if len(primary) > MAX_FACTORS - 1:
            raise SpecificationError(
                f"with_two_tier() supports 1 to {MAX_FACTORS - 1} primary factors, got "
                f"{len(primary)} (the reduced grid has one more dimension than the primary "
                "factors; DESIGN.md §17.2)."
            )
        resolved: list[tuple[str, tuple[str, ...]]] = []
        seen: set[str] = set()
        assigned_all: set[str] = set()
        for raw_name, selector in primary.items():
            name = str(raw_name)
            if not name.strip():
                raise SpecificationError("Factor names must be non-empty strings.")
            if name in seen:
                raise SpecificationError(f"Duplicated factor name: {name!r}.")
            seen.add(name)
            if isinstance(selector, str):
                raise SpecificationError(
                    f"Factor {name!r} items must be an iterable of item names or a callable, "
                    f"not a single string {selector!r}."
                )
            chosen = self._select(selector)
            if not chosen:
                raise SpecificationError(f"Factor {name!r} must have at least one item.")
            if len(set(chosen)) != len(chosen):
                dup = sorted({i for i in chosen if chosen.count(i) > 1})
                raise SpecificationError(f"Factor {name!r} lists duplicated items: {dup}.")
            resolved.append((name, chosen))
            assigned_all.update(chosen)
        unassigned = [i for i in self.item_names if i not in assigned_all]
        if unassigned:
            raise SpecificationError(f"Items not assigned to any primary factor: {unassigned}.")
        n_primary = len(resolved)
        resolved.extend(self._resolve_specific(specific, seen))
        if len(resolved) > MAX_KERNEL_FACTORS:
            raise SpecificationError(
                f"with_two_tier() supports at most {MAX_KERNEL_FACTORS} factors in total, got "
                f"{len(resolved)}."
            )
        candidate = replace(
            self,
            factors=tuple(resolved),
            correlated_factors=bool(correlated) and n_primary > 1,
            n_primary_factors=n_primary,
        )
        self._check_parameter_names(candidate, "with_two_tier")
        return candidate

    def with_latent(self, dist: LatentSpec) -> Model:
        """Return a single-group model with latent distribution ``dist``."""
        if not isinstance(dist, (Normal, EmpiricalHistogram, DavidianCurve, NormalMixture)):
            raise SpecificationError(f"Unsupported latent distribution {dist!r}.")
        if self.groups is not None:
            raise SpecificationError("This model has groups; set latents with with_groups().")
        if self.is_multidimensional:
            raise SpecificationError(
                "with_latent() is only supported for unidimensional models; configure factor "
                "correlations with with_factors(..., correlated=...) or edit the ParameterTable."
            )
        return replace(self, latent=dist)

    def with_groups(self, **latents: LatentSpec) -> Model:
        """Return a multiple-group model; keyword order defines the groups, the first is reference.

        Every group must use ``Normal`` or every group must use ``EmpiricalHistogram`` (mixing
        them across groups raises :class:`~flexpyirt.errors.SpecificationError`; DESIGN.md §4.4).
        :class:`~flexpyirt.DavidianCurve` and :class:`~flexpyirt.NormalMixture` are single-group
        only.

        Examples
        --------
        ``with_groups(ref=Normal.standard(), focal=Normal.free())``
        ``with_groups(ref=EmpiricalHistogram(), focal=EmpiricalHistogram())``

        With :meth:`with_factors` (in either order), each group's ``Normal`` applies to every
        factor: ``Normal.free()`` frees all of that group's factor means and variances, and the
        factor correlations follow ``with_factors(correlated=...)`` in every group (§15.43).
        """
        if len(latents) < 2:
            raise SpecificationError("with_groups() needs at least two groups.")
        for name, dist in latents.items():
            if isinstance(dist, (DavidianCurve, NormalMixture)):
                raise SpecificationError(
                    f"{type(dist).__name__} is only supported for single-group models "
                    f"(group {name!r}); "
                    "use Model.with_latent() instead of with_groups()."
                )
            if not isinstance(dist, (Normal, EmpiricalHistogram)):
                raise SpecificationError(f"Unsupported latent distribution for group {name!r}.")
        kinds = {type(dist) for dist in latents.values()}
        if self.is_multidimensional and EmpiricalHistogram in kinds:
            raise SpecificationError(
                "EmpiricalHistogram is only supported for unidimensional models."
            )
        if EmpiricalHistogram in kinds and len(kinds) > 1:
            raise SpecificationError(
                "Cannot mix EmpiricalHistogram and Normal across groups; when EmpiricalHistogram "
                "is specified, every group must use EmpiricalHistogram."
            )
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
                bad = [
                    p for p in param_tuple if p not in self.item_kernel(lookup[item]).param_names
                ]
                if bad:
                    raise SpecificationError(f"Item {item!r} has no parameters {bad}.")
        return replace(self, equal=(*self.equal, (chosen, param_tuple)))

    def equal_across_items(self, *, items: ItemSelector, params: Iterable[str]) -> Model:
        r"""Return a model where each of ``params`` is one common parameter for ``items``.

        For example ``equal_across_items(items=None, params=["a"])`` gives every item the same
        slope (a "1PL" with the latent variance fixed). The constraint applies within each group
        (as mirt's ``CONSTRAIN``); combine it with :meth:`equal_across_groups` to share across
        groups too. Fixed parameters are not affected. The M-step solves the items connected by
        such constraints jointly (DESIGN.md §4.3, §16.1).

        Parameters
        ----------
        items : iterable of str or callable, or None
            At least two items (``None`` selects every item).
        params : iterable of str
            Internal parameter names (e.g. ``"a"``, ``"a1"``, ``"d"``, ``"d1"``, ``"g"``) that every
            selected item has. Each name is one separate common parameter.
        """
        chosen = self._select(items)
        if len(chosen) < 2:
            raise SpecificationError("equal_across_items() needs at least two items.")
        if isinstance(params, str):
            raise SpecificationError("params must be an iterable of parameter names, not a str.")
        param_tuple = tuple(dict.fromkeys(str(p) for p in params))
        if not param_tuple:
            raise SpecificationError("equal_across_items() needs at least one parameter.")
        lookup = dict(self.items)
        for item in chosen:
            bad = [p for p in param_tuple if p not in self.item_kernel(lookup[item]).param_names]
            if bad:
                raise SpecificationError(f"Item {item!r} has no parameters {bad}.")
        return replace(self, equal_items=(*self.equal_items, (chosen, param_tuple)))

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
        Priors on latent-distribution parameters (``"mean"``, ``"var"``, ``"mean_*"``,
        ``"var_*"``, ``"cor_*"``, ``"phi_*"``, ``"rmean_*"``, ``"rvar_*"``, ``"logit_*"``) are not
        supported.

        Parameters
        ----------
        param : str
            Internal item parameter name (e.g. ``"a"``, ``"a1"``, ``"d"``, ``"g"``, ``"u"``,
            ``"d1"``).
        dist : NormalPrior, LogNormalPrior, BetaPrior, or LogitNormalPrior
            Prior distribution on the natural parameter scale.
        items : iterable of str or callable, optional
            Items to attach the prior to (default: every item in the model).
        """
        if not isinstance(dist, PRIOR_TYPES):
            raise SpecificationError(
                f"Unsupported prior distribution {dist!r}; "
                "use NormalPrior, LogNormalPrior, BetaPrior or LogitNormalPrior."
            )
        latent_prefixes = ("mean_", "var_", "cor_", "phi_", "rmean_", "rvar_", "logit_")
        if param in ("mean", "var") or param.startswith(latent_prefixes):
            raise SpecificationError(
                f"Priors on latent-distribution parameters ({param!r}) are not supported."
            )
        chosen = self._select(items)
        if not chosen:
            raise SpecificationError("prior() selected no items.")
        lookup = dict(self.items)
        for item in chosen:
            params = self.item_kernel(lookup[item]).param_names
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
