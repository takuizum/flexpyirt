r"""IRT true-score and observed-score equating and raw-to-scale tables (DESIGN.md §23).

Result / Python shell layer. :func:`irt_equate` converts the summed scores of a new form to the
summed-score scale of an old form once both are on one :math:`\theta` scale (Lord, 1980; Lord &
Wingersky, 1984; Kolen & Brennan, 2014, §6.5–6.6); :func:`score_distribution` returns the
model-implied summed-score distribution of a form; :func:`scale_table` maps each summed score to
a linear transformation of :math:`\theta`.
"""

from __future__ import annotations

import html
import math
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal, get_args

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from flexpyirt import _equating
from flexpyirt._equating import Form, FormFamily
from flexpyirt._precision import check_x64
from flexpyirt.errors import ConvergenceWarning, SpecificationError
from flexpyirt.items import ItemKernel
from flexpyirt.linking import LinkResult, _extract_item_table, _fitted_grid
from flexpyirt.result import _HTML_STYLE, FitResult, _frame_to_html_table

__all__ = [
    "EquatingMethod",
    "EquatingPopulation",
    "IRTEquatingResult",
    "ScaleMethod",
    "irt_equate",
    "scale_table",
    "score_distribution",
]

EquatingMethod = Literal["true_score", "observed_score"]
"""Methods of :func:`irt_equate`."""
EquatingPopulation = Literal["new", "old", "synthetic"]
"""Populations of observed-score equating in :func:`irt_equate`."""
ScaleMethod = Literal["true_score", "EAPsum"]
"""Methods of :func:`scale_table`."""

_ASYMPTOTE_THETA = 1e8
r"""Own-scale :math:`|\theta|` at which the TCC's asymptotes are evaluated (DESIGN.md §23.2)."""
_MONOTONE_GRID = np.linspace(-20.0, 20.0, 801)
"""Own-scale grid on which true-score methods check that the TCC does not decrease."""
_MONOTONE_TOL = 1e-10
_START_BRACKET = 10.0
_MAX_BRACKET = 1e6
_SCORE_TOL = 1e-13
_MAX_ITER = 300

_form_probs = jax.jit(_equating.form_probs)
_tcc = jax.jit(_equating.test_characteristic)
_information = jax.jit(_equating.form_information)
_solve = jax.jit(_equating.solve_true_score, static_argnames=("score_tol", "max_iter"))
_distribution = jax.jit(_equating.score_distribution)
_eap = jax.jit(_equating.summed_score_eap)


# --------------------------------------------------------------------------------------------
# Forms
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _FormSpec:
    """A form built from one group of one fit, with its population on the common scale."""

    form: Form
    items: tuple[str, ...]
    max_score: int
    nodes: np.ndarray
    weights: np.ndarray


def _require_fit(res: Any, role: str) -> FitResult:
    if not isinstance(res, FitResult):
        raise SpecificationError(f"{role} must be a FitResult, got {type(res).__name__}.")
    if res._model.n_factors != 1:
        raise SpecificationError(
            f"{role} has {res._model.n_factors} factors; IRT equating supports unidimensional "
            "models only."
        )
    return res


def _resolve_group(res: FitResult, group: str | None, argument: str) -> str:
    """Return the name of the group selected by ``group`` (the only group of a single-group fit)."""
    names = [g for g, _ in res._model.resolved_groups()]
    if res._model.is_multigroup:
        if group is None:
            raise SpecificationError(
                f"A multiple-group FitResult needs {argument}=; choose one of {names}."
            )
        if not isinstance(group, str) or group not in names:
            raise SpecificationError(f"Unknown {argument}={group!r}; available groups: {names}.")
        return group
    if group is not None:
        raise SpecificationError(f"{argument}= is only valid for a multiple-group FitResult.")
    return names[0]


def _resolve_items(res: FitResult, items: Iterable[str] | None, argument: str) -> list[str]:
    names = list(res._model.item_names)
    if items is None:
        return names
    if isinstance(items, str):
        raise SpecificationError(f"{argument} must be a list of item names, not a str.")
    chosen = [str(i) for i in items]
    if not chosen:
        raise SpecificationError(f"{argument} must name at least one item.")
    duplicated = sorted({i for i in chosen if chosen.count(i) > 1})
    if duplicated:
        raise SpecificationError(f"Duplicate items in {argument}: {duplicated}.")
    unknown = [i for i in chosen if i not in names]
    if unknown:
        raise SpecificationError(f"Unknown items in {argument}: {unknown}.")
    return chosen


def _build_form(
    res: FitResult,
    items: list[str],
    group: str,
    *,
    slope: float = 1.0,
    intercept: float = 0.0,
) -> _FormSpec:
    r"""Build the form of ``items`` in ``group`` on the scale :math:`A\theta+B` of ``res``."""
    estimates, _ = _extract_item_table(res, group)
    lookup = dict(res._model.items)
    by_kernel: dict[ItemKernel, list[int]] = {}
    kernels: list[ItemKernel] = []
    for position, name in enumerate(items):
        kernel = res._model.item_kernel(lookup[name])
        if not np.all(np.isfinite(estimates[name])):
            raise SpecificationError(f"Item {name!r} has no estimates in group {group!r}.")
        kernels.append(kernel)
        by_kernel.setdefault(kernel, []).append(position)
    families = tuple(
        FormFamily(
            kernel=kernel,
            params=jnp.asarray(np.stack([estimates[items[c]] for c in cols]), dtype=jnp.float64),
            columns=tuple(cols),
        )
        for kernel, cols in by_kernel.items()
    )
    form = Form(
        families=families,
        n_items=len(items),
        max_categories=max(k.n_categories for k in kernels),
        slope=jnp.asarray(slope, dtype=jnp.float64),
        intercept=jnp.asarray(intercept, dtype=jnp.float64),
    )
    group_index = [g for g, _ in res._model.resolved_groups()].index(group)
    nodes, weights = _fitted_grid(res, group, group_index)
    return _FormSpec(
        form=form,
        items=tuple(items),
        max_score=sum(k.n_categories - 1 for k in kernels),
        nodes=slope * np.asarray(nodes, dtype=np.float64) + intercept,
        weights=np.asarray(weights, dtype=np.float64),
    )


def _bucket(n: int) -> int:
    """Return the padded length of ``n`` points: at least 1024 and a power of 2."""
    return max(1024, 1 << max(n - 1, 0).bit_length())


def _on_grid(fn: Any, form: Form, theta: np.ndarray) -> np.ndarray:
    """Evaluate a jitted curve ``fn(form, theta)`` with ``theta`` padded to a bucket length.

    Padding keeps the number of compilations per form small (one per bucket) when the curves are
    evaluated at arrays of many different lengths.
    """
    n = theta.shape[0]
    padded = np.zeros(_bucket(n), dtype=np.float64)
    padded[:n] = theta
    return np.asarray(fn(form, jnp.asarray(padded)), dtype=np.float64)[:n]


def _asymptotes(spec: _FormSpec) -> tuple[float, float]:
    r"""Return :math:`(\tau_{\mathrm{lo}},\tau_{\mathrm{hi}})` of the form's TCC."""
    form = spec.form
    own = np.array([-_ASYMPTOTE_THETA, _ASYMPTOTE_THETA])
    lo, hi = _on_grid(_tcc, form, float(form.slope) * own + float(form.intercept))
    return float(lo), float(hi)


def _check_monotone(spec: _FormSpec, role: str) -> None:
    """Raise if the TCC decreases on the own-scale grid (true-score methods need it monotone)."""
    form = spec.form
    theta = float(form.slope) * _MONOTONE_GRID + float(form.intercept)
    tau = _on_grid(_tcc, form, theta)
    drop = float(np.max(tau[:-1] - tau[1:]))
    if drop > _MONOTONE_TOL * (1.0 + spec.max_score):
        raise SpecificationError(
            f"The test characteristic curve of {role} decreases (by {drop:.3g}) somewhere in "
            "[-20, 20]; true-score methods need a non-decreasing curve (check for negative "
            "slopes or nominal items whose expected score is not monotone)."
        )


def _true_score_theta(spec: _FormSpec, scores: np.ndarray, role: str) -> np.ndarray:
    r"""Solve :math:`\tau(\theta)=x` on the common scale for scores strictly inside the TCC's range.

    Scores at or below the lower asymptote get :math:`-\infty`, scores at or above the upper one
    :math:`+\infty` (DESIGN.md §23.3).
    """
    lo_asym, hi_asym = _asymptotes(spec)
    theta = np.where(scores <= lo_asym, -np.inf, np.inf)
    inside = (scores > lo_asym) & (scores < hi_asym)
    if not np.any(inside):
        return theta
    targets = scores[inside].astype(np.float64)
    form = spec.form
    centre = float(form.intercept)
    half = _START_BRACKET * float(form.slope)
    while True:
        tau_lo, tau_hi = _on_grid(_tcc, form, np.array([centre - half, centre + half]))
        if (tau_lo < targets.min() and tau_hi > targets.max()) or half >= _MAX_BRACKET:
            break
        half *= 2.0
    if not (tau_lo < targets.min() and tau_hi > targets.max()):
        raise SpecificationError(
            f"Some scores of {role} have true-score solutions beyond |theta| = {_MAX_BRACKET:g} "
            "(scores within about 1e-12 of an asymptote of the test characteristic curve)."
        )
    n = targets.size
    solution, residual, _ = _solve(
        form,
        jnp.asarray(targets),
        jnp.full((n,), centre - half, dtype=jnp.float64),
        jnp.full((n,), centre + half, dtype=jnp.float64),
        score_tol=_SCORE_TOL * (1.0 + spec.max_score),
        max_iter=_MAX_ITER,
    )
    residual = np.asarray(residual, dtype=np.float64)
    worst = float(np.max(np.abs(residual)))
    if worst > 1e-8 * (1.0 + spec.max_score):
        warnings.warn(
            f"Solving tau(theta) = x for {role} did not converge (largest |tau - x| = "
            f"{worst:.3e}).",
            ConvergenceWarning,
            stacklevel=3,
        )
    theta[inside] = np.asarray(solution, dtype=np.float64)
    return theta


def _tcc_values(spec: _FormSpec, theta: np.ndarray) -> np.ndarray:
    finite = np.isfinite(theta)
    out = np.full(theta.shape, np.nan)
    if np.any(finite):
        out[finite] = _on_grid(_tcc, spec.form, theta[finite])
    return out


# --------------------------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class IRTEquatingResult:
    r"""Result of :func:`flexpyirt.irt_equate`.

    Attributes
    ----------
    table : DataFrame
        The conversion table: ``score`` (summed score on the new form, ``0..X``), ``equated``
        (its equivalent on the summed-score scale of the old form) and, for the true-score
        method, ``theta`` (the :math:`\theta` on the old form's scale with
        :math:`\tau_X(\theta)=x`; :math:`-\infty` / :math:`+\infty` for scores at or beyond the
        lower / upper asymptote of the new form's test characteristic curve, which are equated
        by the linear interpolation of Kolen & Brennan, 2014, §6.5).
    method : {"true_score", "observed_score"}
        Equating method.
    new_items, old_items : tuple of str
        Items of the new and the old form.
    new_range, old_range : tuple of float
        Lower and upper asymptotes :math:`(\tau_{\mathrm{lo}},\tau_{\mathrm{hi}})` of the test
        characteristic curves of the two forms.
    population : {"new", "old", "synthetic"} or None
        Population of observed-score equating (``None`` for the true-score method).
    synthetic_weight : float or None
        Weight :math:`w_1` of the new form's population in the synthetic population (1 for
        ``"new"``, 0 for ``"old"``; ``None`` for the true-score method).
    new_distribution, old_distribution : DataFrame or None
        Model-implied summed-score distributions (``score``, ``probability``) of the two forms in
        the synthetic population (observed-score method only).
    """

    table: pd.DataFrame
    method: EquatingMethod
    new_items: tuple[str, ...]
    old_items: tuple[str, ...]
    new_range: tuple[float, float]
    old_range: tuple[float, float]
    population: EquatingPopulation | None = None
    synthetic_weight: float | None = None
    new_distribution: pd.DataFrame | None = None
    old_distribution: pd.DataFrame | None = None

    def _meta(self) -> list[tuple[str, str]]:
        rows = [
            ("Method", self.method),
            ("New form", f"{len(self.new_items)} items, scores 0..{len(self.table) - 1}"),
            ("Old form", f"{len(self.old_items)} items"),
        ]
        if self.population is not None:
            rows.append(("Population", f"{self.population} (w_new = {self.synthetic_weight:g})"))
        return rows

    def summary(self, *, digits: int = 4) -> str:
        """Return a readable plain-text summary with the conversion table.

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
        lines = ["flexpyirt IRTEquatingResult"]
        lines += [f"  {k + ':':<12}{v}" for k, v in self._meta()]
        lines += ["", self.table.to_string(index=False, float_format=lambda v: f"{v:.{digits}f}")]
        return "\n".join(lines)

    def __str__(self) -> str:
        """Concise one-line summary."""
        return (
            f"IRTEquatingResult(method={self.method!r}, {len(self.table)} scores, "
            f"{len(self.new_items)} -> {len(self.old_items)} items)"
        )

    def _repr_html_(self) -> str:
        """HTML summary for Jupyter notebooks."""
        meta_rows = "".join(
            f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in self._meta()
        )
        return (
            f"{_HTML_STYLE}"
            '<div class="flexpyirt-result">'
            '<div class="flexpyirt-header">flexpyirt IRTEquatingResult</div>'
            f'<table class="flexpyirt-meta"><tbody>{meta_rows}</tbody></table>'
            f"{_frame_to_html_table(self.table, digits=4)}"
            "</div>"
        )


# --------------------------------------------------------------------------------------------
# Public functions
# --------------------------------------------------------------------------------------------


def _true_score_equate(new: _FormSpec, old: _FormSpec) -> tuple[pd.DataFrame, tuple[float, ...]]:
    """Conversion table of true-score equating with the Kolen–Brennan boundary rules."""
    new_lo, new_hi = _asymptotes(new)
    old_lo, old_hi = _asymptotes(old)
    scores = np.arange(new.max_score + 1, dtype=np.float64)
    theta = _true_score_theta(new, scores, "the new form")
    equated = _tcc_values(old, theta)
    below = scores <= new_lo
    if new_lo > 0.0:
        equated[below] = scores[below] * old_lo / new_lo
    else:
        equated[below] = 0.0
    above = scores >= new_hi
    span = new.max_score - new_hi
    if span > 0.0:
        equated[above] = old_hi + (scores[above] - new_hi) * (old.max_score - old_hi) / span
    else:
        equated[above] = float(old.max_score)
    table = pd.DataFrame({"score": scores.astype(np.int64), "theta": theta, "equated": equated})
    return table, (new_lo, new_hi, old_lo, old_hi)


def _observed_score_equate(
    new: _FormSpec, old: _FormSpec, weight: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the equated scores and the two distributions in the synthetic population."""

    def synthetic(spec: _FormSpec) -> np.ndarray:
        parts = []
        for w_pop, pop in ((weight, new), (1.0 - weight, old)):
            if w_pop == 0.0:
                continue
            dist = _distribution(spec.form, jnp.asarray(pop.nodes), jnp.asarray(pop.weights))
            parts.append(w_pop * np.asarray(dist, dtype=np.float64)[: spec.max_score + 1])
        return np.asarray(np.sum(parts, axis=0), dtype=np.float64)

    f_new = synthetic(new)
    g_old = synthetic(old)
    return _equating.equipercentile(f_new, g_old), f_new, g_old


def irt_equate(
    new: FitResult,
    old: FitResult,
    *,
    method: EquatingMethod = "true_score",
    new_items: Iterable[str] | None = None,
    old_items: Iterable[str] | None = None,
    link: LinkResult | None = None,
    new_group: str | None = None,
    old_group: str | None = None,
    population: EquatingPopulation = "new",
    synthetic_weight: float | None = None,
) -> IRTEquatingResult:
    r"""Equate the summed scores of a new form to an old form by IRT true or observed scores.

    Both forms must be on one :math:`\theta` scale: separate fits after FIPC, a concurrent
    calibration (the same fit twice, with ``new_items`` / ``old_items``), or a ``link`` that
    maps the scale of ``new`` onto that of ``old``, :math:`\theta_Y=A\theta_X+B`. The common
    scale is the old form's; with a link the new form's items are evaluated at
    :math:`(\theta-B)/A` (DESIGN.md §23).

    **True-score equating** (Lord, 1980; Kolen & Brennan, 2014, §6.5). With the test
    characteristic curves :math:`\tau_X(\theta)=\sum_j\sum_kk\,P_{jk}(\theta)` of the two forms,

    .. math:: e_Y(x)=\tau_Y(\theta_x),\qquad \tau_X(\theta_x)=x,

    solved by safeguarded Newton for :math:`\tau_X^{\mathrm{lo}}<x<\tau_X^{\mathrm{hi}}` (the
    sums of the lower and upper asymptotes). Other scores are equated linearly between
    :math:`(0,0)` and :math:`(\tau_X^{\mathrm{lo}},\tau_Y^{\mathrm{lo}})`, and between
    :math:`(\tau_X^{\mathrm{hi}},\tau_Y^{\mathrm{hi}})` and :math:`(X,Y)` (the maximum scores).

    **Observed-score equating** (Lord & Wingersky, 1984; Kolen & Brennan, 2014, §6.6). The
    model-implied summed-score distributions of both forms in the synthetic population
    :math:`w_1\cdot\text{new}+(1-w_1)\cdot\text{old}`,

    .. math:: f(x)=\sum_{p\in\{1,2\}}w_p\sum_qL_X(x\mid\theta_{pq})\,w_{pq},

    with the Lord–Wingersky likelihood :math:`L_X` (:func:`summed_score_distribution`) and the
    fitted latent distribution :math:`\{(\theta_{pq},w_{pq})\}` of each population, are
    equated by the equipercentile method :math:`e_Y(x)=Q^{-1}(P(x))` with the percentile-rank
    functions of Kolen & Brennan (2014, ch. 2).

    Validated against equateIRT, SNSequate, mirt and the equate package (VALIDATION.md V51).
    Standard errors of equated scores are not available.

    Parameters
    ----------
    new : FitResult
        Unidimensional fit containing the new form :math:`X`.
    old : FitResult
        Unidimensional fit containing the old form :math:`Y` (may be ``new``).
    method : {"true_score", "observed_score"}, default "true_score"
        Equating method.
    new_items, old_items : list of str, optional
        Items of each form; default all items of the fit.
    link : LinkResult, optional
        Result of ``link(new, old)`` (``source=new``, ``target=old``, unidimensional).
    new_group, old_group : str, optional
        Group of each form's parameters and population; required for a multiple-group fit.
    population : {"new", "old", "synthetic"}, default "new"
        Population of observed-score equating: the new form's population (:math:`w_1=1`),
        the old form's (:math:`w_1=0`) or the synthetic population with
        :math:`w_1=` ``synthetic_weight``.
    synthetic_weight : float, optional
        Weight :math:`w_1\in[0,1]` of the new form's population; required with
        ``population="synthetic"`` and not allowed otherwise.

    Returns
    -------
    IRTEquatingResult
        The conversion table and, for observed-score equating, the two distributions.

    Raises
    ------
    PrecisionError
        If JAX is not configured for 64-bit floating point.
    SpecificationError
        For a multidimensional fit, unknown or duplicated items, a missing or unknown group, a
        ``link`` that is not a unidimensional ``LinkResult`` between these groups or that does not
        cover the new form's items, invalid ``population`` / ``synthetic_weight``, or (true-score
        method) a test characteristic curve that decreases.
    """
    check_x64()
    new = _require_fit(new, "new")
    old = _require_fit(old, "old")
    if method not in get_args(EquatingMethod):
        raise SpecificationError(
            f"Unknown method={method!r}; use one of {list(get_args(EquatingMethod))}."
        )
    if population not in get_args(EquatingPopulation):
        raise SpecificationError(
            f"Unknown population={population!r}; use one of {list(get_args(EquatingPopulation))}."
        )
    new_group_name = _resolve_group(new, new_group, "new_group")
    old_group_name = _resolve_group(old, old_group, "old_group")
    new_names = _resolve_items(new, new_items, "new_items")
    old_names = _resolve_items(old, old_items, "old_items")
    slope, intercept = 1.0, 0.0
    if link is not None:
        if not isinstance(link, LinkResult):
            raise SpecificationError(f"link must be a LinkResult, got {type(link).__name__}.")
        if link._factor_names:
            raise SpecificationError("link must be a unidimensional LinkResult.")
        if (link._source_group, link._target_group) != (new_group_name, old_group_name):
            raise SpecificationError(
                f"link maps group {link._source_group!r} onto {link._target_group!r}, but the "
                f"forms are in groups {new_group_name!r} (new) and {old_group_name!r} (old); "
                "pass the result of link(new, old)."
            )
        missing = [i for i in new_names if i not in set(link.items["item"])]
        if missing:
            raise SpecificationError(
                f"link does not contain the new form's items {missing}; pass the result of "
                "link(new, old)."
            )
        slope, intercept = float(link.A), float(link.B)
    if method == "observed_score":
        if population == "synthetic":
            if synthetic_weight is None:
                raise SpecificationError('population="synthetic" needs synthetic_weight=.')
            if (
                isinstance(synthetic_weight, bool)
                or not isinstance(synthetic_weight, (int, float))
                or not 0.0 <= float(synthetic_weight) <= 1.0
            ):
                raise SpecificationError(
                    f"synthetic_weight must be a number in [0, 1], got {synthetic_weight!r}."
                )
            weight = float(synthetic_weight)
        else:
            if synthetic_weight is not None:
                raise SpecificationError(
                    'synthetic_weight= is only used with population="synthetic".'
                )
            weight = 1.0 if population == "new" else 0.0
    elif synthetic_weight is not None or population != "new":
        raise SpecificationError(
            'population= and synthetic_weight= are only used with method="observed_score".'
        )

    new_spec = _build_form(new, new_names, new_group_name, slope=slope, intercept=intercept)
    old_spec = _build_form(old, old_names, old_group_name)

    if method == "true_score":
        _check_monotone(new_spec, "the new form")
        _check_monotone(old_spec, "the old form")
        table, (new_lo, new_hi, old_lo, old_hi) = _true_score_equate(new_spec, old_spec)
        return IRTEquatingResult(
            table=table,
            method=method,
            new_items=new_spec.items,
            old_items=old_spec.items,
            new_range=(new_lo, new_hi),
            old_range=(old_lo, old_hi),
        )

    equated, f_new, g_old = _observed_score_equate(new_spec, old_spec, weight)
    new_lo, new_hi = _asymptotes(new_spec)
    old_lo, old_hi = _asymptotes(old_spec)
    return IRTEquatingResult(
        table=pd.DataFrame({"score": np.arange(new_spec.max_score + 1), "equated": equated}),
        method=method,
        new_items=new_spec.items,
        old_items=old_spec.items,
        new_range=(new_lo, new_hi),
        old_range=(old_lo, old_hi),
        population=population,
        synthetic_weight=weight,
        new_distribution=pd.DataFrame(
            {"score": np.arange(new_spec.max_score + 1), "probability": f_new}
        ),
        old_distribution=pd.DataFrame(
            {"score": np.arange(old_spec.max_score + 1), "probability": g_old}
        ),
    )


def score_distribution(
    res: FitResult, *, items: Iterable[str] | None = None, group: str | None = None
) -> pd.DataFrame:
    r"""Return the model-implied summed-score distribution of a form.

    .. math:: f(x)=\sum_qL(x\mid\theta_q)\,w_q

    with the Lord–Wingersky likelihood :math:`L(x\mid\theta)` of the form's items
    (:func:`summed_score_distribution`; Lord & Wingersky, 1984) and the fitted latent
    distribution :math:`\{(\theta_q,w_q)\}` of the group (any latent family). For all items of a
    fit with a ``Normal`` latent distribution it is the ``probability`` column of
    :func:`summed_score_table`, i.e. mirt's ``fscores(method = "EAPsum", full.scores = FALSE)``
    ``expected`` column divided by :math:`N`.

    Parameters
    ----------
    res : FitResult
        A fitted unidimensional model.
    items : list of str, optional
        Items of the form; default all items.
    group : str, optional
        Group whose parameters and population are used; required for a multiple-group model.

    Returns
    -------
    DataFrame
        Columns ``score`` (``0..X``) and ``probability``.

    Raises
    ------
    PrecisionError
        If JAX is not configured for 64-bit floating point.
    SpecificationError
        For a multidimensional model, unknown or duplicated items, or a missing or unknown
        ``group``.
    """
    check_x64()
    res = _require_fit(res, "res")
    spec = _build_form(
        res, _resolve_items(res, items, "items"), _resolve_group(res, group, "group")
    )
    dist = _distribution(spec.form, jnp.asarray(spec.nodes), jnp.asarray(spec.weights))
    return pd.DataFrame(
        {
            "score": np.arange(spec.max_score + 1),
            "probability": np.asarray(dist, dtype=np.float64)[: spec.max_score + 1],
        }
    )


def _check_number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SpecificationError(f"{name} must be a finite number, got {value!r}.")
    if positive and value <= 0:
        raise SpecificationError(f"{name} must be positive, got {value!r}.")
    return float(value)


def _scale_theta(spec: _FormSpec, method: ScaleMethod) -> tuple[np.ndarray, np.ndarray]:
    r"""Return :math:`\theta(x)` and its standard error for every summed score of the form."""
    if method == "true_score":
        _check_monotone(spec, "the form")
        scores = np.arange(spec.max_score + 1, dtype=np.float64)
        theta = _true_score_theta(spec, scores, "the form")
        theta_se = np.full(theta.shape, np.nan)
        finite = np.isfinite(theta)
        if np.any(finite):
            info = _on_grid(_information, spec.form, theta[finite])
            with np.errstate(divide="ignore"):
                theta_se[finite] = 1.0 / np.sqrt(info)
        return theta, theta_se
    _, eap, sd = _eap(spec.form, jnp.asarray(spec.nodes), jnp.asarray(spec.weights))
    return (
        np.asarray(eap, dtype=np.float64)[: spec.max_score + 1],
        np.asarray(sd, dtype=np.float64)[: spec.max_score + 1],
    )


def scale_table(
    res: FitResult,
    *,
    slope: float,
    intercept: float,
    method: ScaleMethod = "true_score",
    items: Iterable[str] | None = None,
    group: str | None = None,
    lowest: float | None = None,
    highest: float | None = None,
    n_digits: int = 0,
) -> pd.DataFrame:
    r"""Return a raw-to-scale conversion table for the summed scores of a form.

    Each summed score :math:`x=0,\dots,X` gets a :math:`\theta` and its standard error, and the
    scale score

    .. math::
        s(x)=\min\{\max\{\text{slope}\cdot\theta(x)+\text{intercept},\ \text{lowest}\},
            \ \text{highest}\},\qquad \mathrm{SE}_s(x)=\text{slope}\cdot\mathrm{SE}_\theta(x).

    - ``method="true_score"``: :math:`\theta(x)` solves :math:`\tau(\theta)=x` on the test
      characteristic curve of the form (Lord, 1980; Kolen & Brennan, 2014, §6.5;
      :math:`-\infty` / :math:`+\infty` at or beyond its lower / upper asymptote) and
      :math:`\mathrm{SE}_\theta=I(\theta)^{-1/2}` with the test information of the form's items.
    - ``method="EAPsum"``: :math:`\theta(x)` is the summed-score EAP and :math:`\mathrm{SE}_\theta`
      the posterior SD under the group's fitted latent distribution (Thissen, Pommerich,
      Billeaud & Williams, 1995), as in :func:`summed_score_table`.

    Parameters
    ----------
    res : FitResult
        A fitted unidimensional model; :math:`\theta` is on its scale.
    slope : float
        Positive slope of the linear transformation of :math:`\theta`.
    intercept : float
        Intercept of the linear transformation.
    method : {"true_score", "EAPsum"}, default "true_score"
        How each summed score is mapped to :math:`\theta`.
    items : list of str, optional
        Items of the form; default all items.
    group : str, optional
        Group whose parameters (and, for ``"EAPsum"``, population) are used; required for a
        multiple-group model.
    lowest, highest : float, optional
        Bounds the scale scores are clipped to (no clipping on a side without a bound, so an
        infinite :math:`\theta` gives an infinite scale score).
    n_digits : int, default 0
        Decimal places of ``scale_score_rounded`` (NumPy ``round``, half to even).

    Returns
    -------
    DataFrame
        Columns ``score``, ``theta``, ``theta_se`` (NaN where :math:`\theta` is infinite),
        ``scale_score``, ``scale_score_se`` (not clipped) and ``scale_score_rounded``.

    Raises
    ------
    PrecisionError
        If JAX is not configured for 64-bit floating point.
    SpecificationError
        For a multidimensional model, unknown or duplicated items, a missing or unknown
        ``group``, a non-positive ``slope``, ``lowest >= highest``, a negative ``n_digits``, or
        (``"true_score"``) a test characteristic curve that decreases.
    """
    check_x64()
    res = _require_fit(res, "res")
    if method not in get_args(ScaleMethod):
        raise SpecificationError(
            f"Unknown method={method!r}; use one of {list(get_args(ScaleMethod))}."
        )
    slope_value = _check_number(slope, "slope", positive=True)
    intercept_value = _check_number(intercept, "intercept")
    low = -np.inf if lowest is None else _check_number(lowest, "lowest")
    high = np.inf if highest is None else _check_number(highest, "highest")
    if low >= high:
        raise SpecificationError(f"lowest ({lowest}) must be below highest ({highest}).")
    if isinstance(n_digits, bool) or not isinstance(n_digits, int) or n_digits < 0:
        raise SpecificationError(f"n_digits must be a non-negative integer, got {n_digits!r}.")
    spec = _build_form(
        res, _resolve_items(res, items, "items"), _resolve_group(res, group, "group")
    )
    scores = np.arange(spec.max_score + 1)
    theta, theta_se = _scale_theta(spec, method)
    scale = np.clip(slope_value * theta + intercept_value, low, high)
    return pd.DataFrame(
        {
            "score": scores,
            "theta": theta,
            "theta_se": theta_se,
            "scale_score": scale,
            "scale_score_se": slope_value * theta_se,
            "scale_score_rounded": np.round(scale, n_digits),
        }
    )
