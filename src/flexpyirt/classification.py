r"""Classification accuracy and consistency for cut scores (DESIGN.md §18.4).

Result / Python shell layer for unidimensional models: :func:`classification_accuracy` takes a
:class:`~flexpyirt.FitResult` and cut scores and returns a :class:`ClassificationResult`. The
JAX kernels are in :mod:`flexpyirt._classification`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, get_args

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from flexpyirt import _classification, _core, _information
from flexpyirt._compile import coerce_groups
from flexpyirt._precision import check_x64
from flexpyirt.errors import SpecificationError
from flexpyirt.information import _group_names, _problem, _require_unidimensional
from flexpyirt.result import FitResult
from flexpyirt.sumscore import summed_score_distribution

__all__ = ["ClassificationMethod", "ClassificationResult", "classification_accuracy"]

ClassificationMethod = Literal["lee", "rudner"]
"""Methods of :func:`classification_accuracy`."""
ClassificationScale = Literal["summed_score", "theta"]
"""Scales the cut scores of :func:`classification_accuracy` can be given on."""
ClassificationScoreMethod = Literal["EAP", "MAP", "ML", "WLE"]
"""Scoring methods for the respondents' estimates of the P method."""

_NATIVE_SCALE: dict[str, str] = {"lee": "summed_score", "rudner": "theta"}
_INVERSE_BRACKET = 40.0
"""Half-width of the bracket of the inverse test characteristic curve."""
_INVERSE_STEPS = 200
"""Bisection steps; the bracket shrinks below the float64 spacing well before this."""

_category_probabilities = jax.jit(_information.category_probabilities)
_item_curves = jax.jit(_information.item_curves)
_lee_probabilities = jax.jit(
    lambda probs, cuts: _classification.lee_probabilities(summed_score_distribution(probs), cuts)
)
_rudner_probabilities = jax.jit(_classification.rudner_probabilities)
_tables = jax.jit(_classification.tables)


@dataclass(frozen=True)
class ClassificationResult:
    r"""Result of :func:`classification_accuracy` (Lee, 2010; Rudner, 2001, 2005).

    The cut scores define the categories ``0 .. C`` (a value at or above a cut is above it).
    In a multiple-group result every DataFrame has a leading ``group`` column (the two tables
    a ``(group, category)`` row index).

    Attributes
    ----------
    method : str
        ``"lee"`` or ``"rudner"``.
    scale : str
        Scale of ``cut_scores``: ``"summed_score"`` or ``"theta"``.
    cut_scores : tuple of float
        The cut scores as given.
    approach : str
        ``"D"`` (integrated over the fitted latent distribution) or ``"P"`` (averaged over the
        respondents' estimates in ``data``).
    summary : pandas.DataFrame
        One row per group: ``accuracy`` :math:`\gamma=\sum_hA_{hh}`, ``consistency``
        :math:`\phi=\sum_hB_{hh}`, ``kappa`` (Cohen's kappa of consistency) and, for the P
        approach, ``n`` (respondents).
    by_category : pandas.DataFrame
        Per category: ``category``, ``true_proportion`` (:math:`\sum_kA_{hk}`), ``proportion``
        (observed, :math:`p_h=\sum_kB_{hk}`), ``accuracy`` (observed :math:`=h` given true
        :math:`=h`) and ``consistency`` (second :math:`=h` given first :math:`=h`).
    by_cut : pandas.DataFrame
        Per cut (the split below / at or above it): ``cut``, ``accuracy``, ``consistency``,
        ``kappa``, ``false_positive`` (true below, observed above) and ``false_negative``;
        ``summed_score_cut`` or ``theta_cut`` when the cuts were converted to the method's scale.
    conditional : pandas.DataFrame
        Per quadrature node (D) or respondent (P, indexed like ``data``): ``theta``,
        ``weight``, ``true_category``, ``accuracy``, ``consistency`` and ``true_score`` (Lee) or
        ``se`` (Rudner).
    accuracy_table : pandas.DataFrame
        :math:`A`: joint probabilities of the true (rows) and observed (columns) categories.
    consistency_table : pandas.DataFrame
        :math:`B`: joint probabilities of the categories of two independent administrations.
    """

    method: str
    scale: str
    cut_scores: tuple[float, ...]
    approach: str
    summary: pd.DataFrame
    by_category: pd.DataFrame
    by_cut: pd.DataFrame
    conditional: pd.DataFrame
    accuracy_table: pd.DataFrame
    consistency_table: pd.DataFrame

    def _overall(self, column: str) -> float:
        if "group" in self.summary.columns:
            raise SpecificationError(
                f"{column} is defined per group in a multiple-group result; use .summary."
            )
        return float(self.summary[column].iloc[0])

    @property
    def accuracy(self) -> float:
        """Overall classification accuracy (single-group results)."""
        return self._overall("accuracy")

    @property
    def consistency(self) -> float:
        """Overall classification consistency (single-group results)."""
        return self._overall("consistency")

    @property
    def kappa(self) -> float:
        """Cohen's kappa of classification consistency (single-group results)."""
        return self._overall("kappa")


def _kappa(agreement: float, marginal: np.ndarray) -> float:
    chance = float(np.sum(marginal**2))
    return (agreement - chance) / (1.0 - chance) if chance < 1.0 else float("nan")


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0.0, numerator / denominator, np.nan)


def _indices(
    accuracy_table: np.ndarray, consistency_table: np.ndarray
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Overall, per-category and per-cut indices from the tables A and B."""
    a, b = accuracy_table, consistency_table
    n_categories = a.shape[0]
    true_prop = a.sum(axis=1)
    observed = b.sum(axis=1)
    consistency = float(np.trace(b))
    overall = {
        "accuracy": float(np.trace(a)),
        "consistency": consistency,
        "kappa": _kappa(consistency, observed),
    }
    by_category = pd.DataFrame(
        {
            "category": np.arange(n_categories),
            "true_proportion": true_prop,
            "proportion": observed,
            "accuracy": _ratio(np.diag(a), true_prop),
            "consistency": _ratio(np.diag(b), observed),
        }
    )
    rows = []
    for k in range(1, n_categories):
        lo, hi = slice(0, k), slice(k, n_categories)
        cut_consistency = float(b[lo, lo].sum() + b[hi, hi].sum())
        rows.append(
            {
                "accuracy": float(a[lo, lo].sum() + a[hi, hi].sum()),
                "consistency": cut_consistency,
                "kappa": _kappa(
                    cut_consistency, np.array([observed[lo].sum(), observed[hi].sum()])
                ),
                "false_positive": float(a[lo, hi].sum()),
                "false_negative": float(a[hi, lo].sum()),
            }
        )
    return overall, by_category, pd.DataFrame(rows)


def _coerce_cuts(cut_scores: Any) -> np.ndarray:
    try:
        cuts = np.atleast_1d(np.asarray(cut_scores, dtype=np.float64))
    except (TypeError, ValueError) as exc:
        raise SpecificationError("cut_scores must be a number or a 1-D array of numbers.") from exc
    if cuts.ndim != 1 or cuts.size == 0:
        raise SpecificationError(
            f"cut_scores must be a number or a non-empty 1-D array, got shape {cuts.shape}."
        )
    if not np.all(np.isfinite(cuts)):
        raise SpecificationError("cut_scores must be finite.")
    if np.any(np.diff(cuts) <= 0.0):
        raise SpecificationError(f"cut_scores must be strictly increasing, got {cuts.tolist()}.")
    return cuts


def _check_summed_cuts(cuts: np.ndarray, max_score: int, what: str) -> None:
    if cuts[0] <= 0.0 or cuts[-1] > max_score:
        raise SpecificationError(
            f"{what} must lie in (0, {max_score}] (the summed scores of the test), got "
            f"{cuts.tolist()}."
        )
    ceilings = np.ceil(cuts)
    if np.any(np.diff(ceilings) == 0.0):
        raise SpecificationError(
            f"{what} {cuts.tolist()} select the same summed scores (the scores at or above "
            f"{ceilings.tolist()}); every category must contain at least one score."
        )


def _true_score(
    prob: _core.CompiledProblem, eta: jax.Array, theta: np.ndarray, group_index: int
) -> np.ndarray:
    r""":math:`\tau(\theta)=\sum_j\sum_kk\,P_{jk}(\theta)` of one group."""
    probs = np.asarray(_category_probabilities(prob, eta, jnp.asarray(theta)))[group_index]
    scores = np.arange(probs.shape[-1], dtype=np.float64)
    return np.asarray(np.einsum("tjk,k->t", probs, scores), dtype=np.float64)


def _inverse_true_score(
    prob: _core.CompiledProblem,
    eta: jax.Array,
    cuts: np.ndarray,
    group_index: int,
    group_label: str,
) -> np.ndarray:
    r"""Solve :math:`\tau(\theta)=c` for each cut by bisection on :math:`[-40, 40]`."""
    lo = np.full(cuts.shape, -_INVERSE_BRACKET)
    hi = np.full(cuts.shape, _INVERSE_BRACKET)
    ends = _true_score(prob, eta, np.array([lo[0], hi[0]]), group_index)
    if np.any(cuts <= ends[0]) or np.any(cuts >= ends[1]):
        raise SpecificationError(
            f"cut_scores {cuts.tolist()} are outside the range of the test characteristic "
            f"curve{group_label} ({ends[0]:.6g}, {ends[1]:.6g}); summed-score cuts cannot be "
            "converted to theta there."
        )
    for _ in range(_INVERSE_STEPS):
        mid = 0.5 * (lo + hi)
        above = _true_score(prob, eta, mid, group_index) >= cuts
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
        if np.all(hi - lo <= 4.0 * np.spacing(np.maximum(np.abs(lo), np.abs(hi)))):
            break
    return np.asarray(0.5 * (lo + hi), dtype=np.float64)


def _points(
    res: FitResult,
    prob: _core.CompiledProblem,
    eta: jax.Array,
    data: Any,
    group: Any,
    item_names: list[str] | None,
    score_method: str,
) -> list[tuple[int, np.ndarray, np.ndarray, pd.Index | None]]:
    """Per group: ``(group index, theta, weights, respondent index or None)``."""
    names = _group_names(res)
    if data is None:
        nodes = np.asarray(prob.nodes[:, 0])
        weights = np.exp(np.asarray(_core._latent_log_weights(prob, eta)))
        weights = weights / weights.sum(axis=1, keepdims=True)
        return [(g, nodes, weights[g], None) for g in range(len(names))]
    scores = res.score(data, method=score_method, group=group, item_names=item_names)  # type: ignore[arg-type]
    theta = scores["theta"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(theta)):
        bad = int(np.sum(~np.isfinite(theta)))
        raise SpecificationError(
            f"{bad} respondent(s) have a non-finite {score_method} estimate (for example ML on "
            "an all-lowest or all-highest pattern, or an all-missing row); use "
            "score_method='EAP' or 'WLE', or drop those rows."
        )
    codes, _ = coerce_groups(group, res._model, len(theta), require_all=False)
    out = []
    for g in range(len(names)):
        members = codes == g
        n_members = int(members.sum())
        if n_members == 0:
            continue
        out.append((g, theta[members], np.full(n_members, 1.0 / n_members), scores.index[members]))
    return out


def classification_accuracy(
    res: FitResult,
    *,
    cut_scores: Any,
    method: ClassificationMethod = "lee",
    scale: ClassificationScale | None = None,
    data: Any = None,
    group: Any = None,
    item_names: list[str] | None = None,
    score_method: ClassificationScoreMethod = "EAP",
) -> ClassificationResult:
    r"""Return the classification accuracy and consistency of cut scores.

    Cut scores :math:`c_1<\dots<c_C` sort respondents into the categories
    :math:`h(v)=\#\{k: v\ge c_k\}=0,\dots,C`. At a true value :math:`\theta` each method gives
    the probabilities :math:`P_h(\theta)` of the observed categories and a true category
    :math:`t(\theta)`:

    - ``"lee"`` (Lee, 2010; cuts on the summed score):
      :math:`P_h(\theta)=\sum_{s:\,h(s)=h}L(s\mid\theta)` with the Lord–Wingersky likelihood
      (every item family) and :math:`t(\theta)=h(\tau(\theta))`,
      :math:`\tau(\theta)=\sum_j\sum_kk\,P_{jk}(\theta)`. A non-integer cut :math:`c` puts the
      scores :math:`s\ge\lceil c\rceil` above it.
    - ``"rudner"`` (Rudner, 2001, 2005; cuts on :math:`\theta`):
      :math:`P_h(\theta)=\Phi((c_{h+1}-\theta)/\mathrm{SE})-\Phi((c_h-\theta)/\mathrm{SE})`
      with :math:`\mathrm{SE}(\theta)=I(\theta)^{-1/2}` and :math:`t(\theta)=h(\theta)`.

    Over points :math:`\theta_n` with weights :math:`w_n` (the fitted quadrature and latent
    weights without ``data``, the D approach; the respondents' estimates with weights
    :math:`1/N` with ``data``, the P approach; Lathrop & Cheng, 2013) the tables

    .. math::
        A_{tk}=\sum_nw_n[t(\theta_n)=t]P_k(\theta_n),\qquad
        B_{hk}=\sum_nw_nP_h(\theta_n)P_k(\theta_n)

    give accuracy :math:`\gamma=\sum_hA_{hh}`, consistency :math:`\phi=\sum_hB_{hh}` and
    Cohen's (1960) kappa :math:`\kappa=(\phi-\phi_c)/(1-\phi_c)`,
    :math:`\phi_c=\sum_h(\sum_kB_{hk})^2`. They are cacIRT's ``class.Lee()`` and
    ``class.Rud()`` ("Simultaneous" for all cuts, "Marginal" per cut).

    Parameters
    ----------
    res : FitResult
        A fitted unidimensional model.
    cut_scores : float or array-like, shape (C,)
        Strictly increasing cut scores; a value at or above a cut is above it. On the summed
        score they must lie in :math:`(0, S_{\max}]` and select different summed scores.
    method : {"lee", "rudner"}, default "lee"
        Lee's summed-score method or Rudner's normal-approximation method.
    scale : {"summed_score", "theta"}, optional
        Scale of ``cut_scores``; default the method's own (summed score for Lee, theta for
        Rudner). Otherwise the cuts are converted through the test characteristic curve
        :math:`\tau` (Lee: :math:`\tau(c)`; Rudner: :math:`\tau^{-1}(c)`, which needs an
        increasing :math:`\tau`).
    data : DataFrame or array-like, optional
        Responses on the fitted items (NaN is missing). When given, the indices are averaged
        over the respondents' estimates (the P approach) instead of integrated over the fitted
        latent distribution (the D approach). Each row counts once: the sampling weights of a
        weighted fit are not applied.
    group : array-like, optional
        Group label of each row of ``data`` (required with ``data`` for a multiple-group
        model). Without ``data``, every group of a multiple-group model is reported.
    item_names : list of str, optional
        Column names when ``data`` is an array.
    score_method : {"EAP", "MAP", "ML", "WLE"}, default "EAP"
        Estimates of the P approach (:meth:`FitResult.score`). Rudner's SE is always the
        information SE :math:`I(\hat\theta)^{-1/2}`.

    Returns
    -------
    ClassificationResult
        Overall, per-category, per-cut and conditional indices and the tables :math:`A` and
        :math:`B`, per group in a multiple-group model.

    Raises
    ------
    PrecisionError
        If JAX is not configured for 64-bit floating point.
    SpecificationError
        For a multidimensional model, an unknown ``method`` / ``scale`` / ``score_method``,
        invalid cut scores, ``group`` / ``item_names`` / ``score_method`` without ``data``, or a
        non-finite estimate in the P approach.
    """
    check_x64()
    _require_unidimensional(res, "classification_accuracy()")
    if method not in get_args(ClassificationMethod):
        raise SpecificationError(
            f"Unknown method={method!r}; use one of {list(get_args(ClassificationMethod))}."
        )
    if scale is None:
        scale = _NATIVE_SCALE[method]  # type: ignore[assignment]
    if scale not in get_args(ClassificationScale):
        raise SpecificationError(
            f"Unknown scale={scale!r}; use one of {list(get_args(ClassificationScale))}."
        )
    if score_method not in get_args(ClassificationScoreMethod):
        raise SpecificationError(
            f"Unknown score_method={score_method!r}; use one of "
            f"{list(get_args(ClassificationScoreMethod))}."
        )
    if data is None and (group is not None or item_names is not None or score_method != "EAP"):
        raise SpecificationError(
            "group=, item_names= and score_method= are only used together with data."
        )
    cuts = _coerce_cuts(cut_scores)
    max_score = sum(m.kernel.n_categories - 1 for _, m in res._model.items)
    if scale == "summed_score":
        _check_summed_cuts(cuts, max_score, "cut_scores")
    native = scale == _NATIVE_SCALE[method]
    converted_column = "summed_score_cut" if method == "lee" else "theta_cut"

    prob, eta = _problem(res)
    names = _group_names(res)
    multigroup = res._model.is_multigroup
    frames: dict[str, list[pd.DataFrame]] = {
        key: [] for key in ("summary", "by_category", "by_cut", "conditional", "a", "b")
    }
    for g, theta, weights, respondents in _points(
        res, prob, eta, data, group, item_names, score_method
    ):
        label = f" of group {names[g]!r}" if multigroup else ""
        if native:
            method_cuts = cuts
        elif method == "lee":
            method_cuts = _true_score(prob, eta, cuts, g)
            _check_summed_cuts(method_cuts, max_score, f"The summed-score cuts{label}")
        else:
            method_cuts = _inverse_true_score(prob, eta, cuts, g, label)
        cuts_j = jnp.asarray(method_cuts)
        theta_j = jnp.asarray(theta)
        conditional: dict[str, Any] = {"theta": theta, "weight": weights}
        if method == "lee":
            probs = np.asarray(_category_probabilities(prob, eta, theta_j))[g]
            p_cat = np.asarray(_lee_probabilities(jnp.asarray(probs), cuts_j))
            true_score = np.einsum("tjk,k->t", probs, np.arange(probs.shape[-1], dtype=np.float64))
            true = np.asarray(_classification.categories(jnp.asarray(true_score), cuts_j))
        else:
            information = np.asarray(_item_curves(prob, eta, theta_j.reshape(-1, 1))[0])[
                g, :, :, 0, 0
            ].sum(axis=1)
            with np.errstate(divide="ignore"):
                se = 1.0 / np.sqrt(information)
            p_cat = np.asarray(_rudner_probabilities(theta_j, jnp.asarray(se), cuts_j))
            true = np.asarray(_classification.categories(theta_j, cuts_j))
        a, b = (
            np.asarray(t)
            for t in _tables(jnp.asarray(p_cat), jnp.asarray(true), jnp.asarray(weights))
        )
        conditional["true_category"] = true
        conditional["accuracy"] = np.take_along_axis(p_cat, true[:, None], axis=1)[:, 0]
        conditional["consistency"] = np.sum(p_cat**2, axis=1)
        if method == "lee":
            conditional["true_score"] = true_score
        else:
            conditional["se"] = se
        overall, by_category, by_cut = _indices(a, b)
        by_cut.insert(0, "cut", cuts)
        if not native:
            by_cut.insert(1, converted_column, method_cuts)
        summary = pd.DataFrame([overall])
        if data is not None:
            summary["n"] = len(theta)
        cond = pd.DataFrame(conditional, index=respondents)
        categories = pd.Index(np.arange(len(cuts) + 1), name="true")
        observed = pd.Index(np.arange(len(cuts) + 1), name="observed")
        a_frame = pd.DataFrame(a, index=categories, columns=observed)
        b_frame = pd.DataFrame(
            b, index=categories.rename("first"), columns=observed.rename("second")
        )
        if multigroup:
            for frame in (summary, by_category, by_cut, cond):
                frame.insert(0, "group", names[g])
            a_frame = pd.concat({names[g]: a_frame}, names=["group"])
            b_frame = pd.concat({names[g]: b_frame}, names=["group"])
        for key, frame in (
            ("summary", summary),
            ("by_category", by_category),
            ("by_cut", by_cut),
            ("conditional", cond),
            ("a", a_frame),
            ("b", b_frame),
        ):
            frames[key].append(frame)

    def stacked(key: str, *, keep_index: bool = False) -> pd.DataFrame:
        return pd.concat(frames[key], ignore_index=not keep_index)

    return ClassificationResult(
        method=method,
        scale=str(scale),
        cut_scores=tuple(float(c) for c in cuts),
        approach="D" if data is None else "P",
        summary=stacked("summary"),
        by_category=stacked("by_category"),
        by_cut=stacked("by_cut"),
        conditional=stacked("conditional", keep_index=data is not None),
        accuracy_table=stacked("a", keep_index=True),
        consistency_table=stacked("b", keep_index=True),
    )
