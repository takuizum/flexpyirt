"""Differential item functioning on observed scores (DESIGN.md §16.20).

:func:`mantel_haenszel` (0 / 1 items, two groups), :func:`mantel` (ordered polytomous items, two
groups), :func:`generalized_mantel_haenszel` (several focal groups, or nominal categories) and
:func:`logistic_dif` (0 / 1 items, uniform and non-uniform DIF). Respondents are matched on the
total score of the anchor items, the studied item included, or on an external matching
variable; every distinct value is a stratum. With ``purify=True`` the items flagged at one step
leave the matching score of the next until the flagged set repeats (Holland & Thayer, 1988;
difR's ``purify = TRUE``). No IRT model is involved and nothing here imports JAX.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm

from flexpyctt import _dif
from flexpyctt.errors import ConvergenceWarning, FlexpycttWarning, SpecificationError

__all__ = [
    "DIFResult",
    "generalized_mantel_haenszel",
    "logistic_dif",
    "mantel",
    "mantel_haenszel",
]

PAdjust = Literal["holm", "BH", "none"]
GroupLabels = pd.Series | np.ndarray | Sequence[object]

#: Effect-size thresholds of the logistic-regression classes (on Nagelkerke's Delta R^2).
ZUMBO_THOMAS = (0.13, 0.26)
JODOIN_GIERL = (0.035, 0.07)


@dataclass(frozen=True)
class DIFResult:
    """Result of an observed-score DIF analysis.

    Attributes
    ----------
    table : pandas.DataFrame
        One row per studied item: ``item``, the statistic and effect sizes of the method,
        ``p``, ``adj_p`` and ``flagged`` (``adj_p < alpha``).
    anchor_items : list of str
        Items whose total forms the matching score of the final step (after purification; each
        studied item is added to it for its own test). Empty with an external ``matching``.
    n_iter : int
        Purification iterations run (0 without purification or when nothing was flagged at
        the first step).
    converged : bool
        Whether purification ended with a repeated flagged set (always ``True`` without
        purification).
    flag_history : pandas.DataFrame
        Items flagged (unadjusted ``p < alpha``) at each step, one row per step (``0`` is the
        test on the initial anchors), one column per studied item.
    method : str
        ``"mantel_haenszel"``, ``"mantel"``, ``"generalized_mantel_haenszel"`` or
        ``"logistic"``.
    """

    table: pd.DataFrame
    anchor_items: list[str]
    n_iter: int
    converged: bool
    flag_history: pd.DataFrame
    method: str

    @property
    def flagged_items(self) -> list[str]:
        """Names of the items flagged in the final table."""
        return [str(i) for i in self.table.loc[self.table["flagged"], "item"]]


# ---------------------------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Prepared:
    names: list[str]
    values: np.ndarray
    member: np.ndarray
    tested: np.ndarray
    anchors: np.ndarray
    matching: np.ndarray | None
    n_groups: int


def _scores(data: pd.DataFrame | np.ndarray, what: str) -> tuple[list[str], np.ndarray]:
    if isinstance(data, pd.DataFrame):
        names = [str(c) for c in data.columns]
        try:
            values = data.to_numpy(dtype=np.float64, na_value=np.nan)
        except (TypeError, ValueError) as exc:
            raise SpecificationError(
                f"{what} needs numeric item scores; could not convert data to float: {exc}"
            ) from exc
    else:
        try:
            values = np.asarray(data, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise SpecificationError(
                f"{what} needs numeric item scores; could not convert data to float: {exc}"
            ) from exc
        if values.ndim != 2:
            raise SpecificationError(
                f"data must be two-dimensional (respondents x items), got shape {values.shape}."
            )
        names = [f"item{j + 1}" for j in range(values.shape[1])]
    if len(set(names)) != len(names):
        raise SpecificationError("Item (column) names must be unique.")
    return names, values


def _select(names: list[str], chosen: Sequence[str] | None, what: str) -> np.ndarray:
    if chosen is None:
        return np.arange(len(names))
    if isinstance(chosen, str):
        chosen = [chosen]
    chosen = [str(c) for c in chosen]
    unknown = [c for c in chosen if c not in names]
    if unknown:
        raise SpecificationError(f"{what} names unknown items: {unknown}.")
    if len(set(chosen)) != len(chosen):
        raise SpecificationError(f"{what} lists an item more than once.")
    if not chosen:
        raise SpecificationError(f"{what} is empty.")
    return np.array(sorted(names.index(c) for c in chosen), dtype=np.int64)


def _prepare(
    data: pd.DataFrame | np.ndarray,
    group: GroupLabels,
    focal: Sequence[object],
    *,
    items: Sequence[str] | None,
    anchor_items: Sequence[str] | None,
    matching: pd.Series | np.ndarray | Sequence[float] | None,
    purify: bool,
    max_iter: int,
    p_adjust: str,
    alpha: float,
    what: str,
) -> _Prepared:
    names, values = _scores(data, what)
    n_obs = values.shape[0]
    labels = np.asarray(group, dtype=object).ravel()
    if labels.shape[0] != n_obs:
        raise SpecificationError(f"group has {labels.shape[0]} labels for {n_obs} respondents.")
    if pd.isna(pd.Series(labels)).any():
        raise SpecificationError("group has missing labels.")
    member = np.zeros(n_obs, dtype=np.int64)
    for code, label in enumerate(focal, start=1):
        rows = labels == label
        if not rows.any():
            raise SpecificationError(f"Focal group {label!r} does not occur in group.")
        member[rows] = code
    if not (member == 0).any():
        raise SpecificationError("No reference respondents: every row belongs to a focal group.")
    if p_adjust not in ("holm", "BH", "none"):
        raise SpecificationError(f"Unknown p_adjust={p_adjust!r}; use 'holm', 'BH' or 'none'.")
    if not 0.0 < alpha < 1.0:
        raise SpecificationError(f"alpha must be in (0, 1), got {alpha}.")
    if max_iter < 1:
        raise SpecificationError(f"max_iter must be at least 1, got {max_iter}.")
    tested = _select(names, items, "items")
    if matching is not None:
        if purify:
            raise SpecificationError(
                "purify=True needs the total-score matching; it cannot be combined with an "
                "external matching variable."
            )
        if anchor_items is not None:
            raise SpecificationError("Give either anchor_items or matching, not both.")
        match = np.asarray(matching, dtype=np.float64).ravel()
        if match.shape[0] != n_obs:
            raise SpecificationError(f"matching has {match.shape[0]} values for {n_obs} rows.")
        if not np.isfinite(match).all():
            raise SpecificationError("matching has missing or non-finite values.")
        anchors = np.zeros(0, dtype=np.int64)
    else:
        match = None
        anchors = _select(names, anchor_items, "anchor_items")
    used = np.union1d(tested, anchors)
    if np.isnan(values[:, used]).any():
        missing = [names[j] for j in used if np.isnan(values[:, j]).any()]
        raise SpecificationError(
            f"{what} needs complete responses on the studied and matching items; missing "
            f"values in {missing}. Drop those rows first."
        )
    return _Prepared(names, values, member, tested, anchors, match, len(focal) + 1)


def _matching_score(prep: _Prepared, item: int, anchors: np.ndarray) -> np.ndarray:
    if prep.matching is not None:
        return prep.matching
    total = prep.values[:, anchors].sum(axis=1)
    if item not in anchors:
        total = total + prep.values[:, item]
    return total


def _require_binary(prep: _Prepared, what: str) -> None:
    sub = prep.values[:, prep.tested]
    if not np.isin(sub, (0.0, 1.0)).all():
        bad = [prep.names[j] for j in prep.tested if not np.isin(prep.values[:, j], (0, 1)).all()]
        raise SpecificationError(f"{what} needs 0 / 1 scores on the studied items; see {bad}.")


Statistic = Callable[[np.ndarray], dict[str, np.ndarray]]


def _run(
    prep: _Prepared, statistic: Statistic, *, purify: bool, max_iter: int, alpha: float
) -> tuple[dict[str, np.ndarray], np.ndarray, int, bool, list[np.ndarray]]:
    """Run ``statistic`` on the anchors, then purify (difR's rule, unadjusted p-values)."""

    def significant(stats: dict[str, np.ndarray]) -> np.ndarray:
        p = stats["p"]
        return np.isfinite(p) & (p < alpha)

    anchors = prep.anchors
    stats = statistic(anchors)
    flags = significant(stats)
    history = [flags]
    n_iter = 0
    converged = True
    if purify and flags.any():
        converged = False
        current = anchors
        while n_iter < max_iter:
            n_iter += 1
            current = anchors[~np.isin(anchors, prep.tested[flags])]
            stats = statistic(current)
            new = significant(stats)
            history.append(new)
            if np.array_equal(new, flags):
                converged = True
                break
            flags = new
        anchors = current
    return stats, anchors, n_iter, converged, history


def _result(
    prep: _Prepared,
    columns: Mapping[str, np.ndarray | list[str | None]],
    run: tuple[dict[str, np.ndarray], np.ndarray, int, bool, list[np.ndarray]],
    *,
    p_adjust: str,
    alpha: float,
    method: str,
) -> DIFResult:
    stats, anchors, n_iter, converged, history = run
    if not converged:
        warnings.warn(
            f"Purification did not converge in {n_iter} iterations; the table is from the last "
            "iteration.",
            ConvergenceWarning,
            stacklevel=3,
        )
    item_names = [prep.names[j] for j in prep.tested]
    table = pd.DataFrame({"item": item_names, **columns})
    table["p"] = stats["p"]
    table["adj_p"] = _dif.adjust_p(stats["p"], p_adjust)
    table["flagged"] = np.isfinite(table["adj_p"].to_numpy()) & (table["adj_p"].to_numpy() < alpha)
    history_frame = pd.DataFrame(np.vstack(history), columns=item_names)
    history_frame.index.name = "step"
    return DIFResult(
        table=table,
        anchor_items=[prep.names[j] for j in anchors],
        n_iter=n_iter,
        converged=converged,
        flag_history=history_frame,
        method=method,
    )


def _warn_sparse(names: list[str]) -> None:
    if names:
        warnings.warn(
            f"Sparse strata for {names}: the common odds ratio is 0, infinite or undefined.",
            FlexpycttWarning,
            stacklevel=3,
        )


# ---------------------------------------------------------------------------------------------
# Mantel–Haenszel
# ---------------------------------------------------------------------------------------------


def _ets_class(
    delta: np.ndarray, delta_se: np.ndarray, p: np.ndarray, alpha: float, rule: str
) -> list[str | None]:
    size = np.abs(delta)
    z = norm.ppf(1.0 - alpha)
    out: list[str | None] = []
    for d, se, pv in zip(size, delta_se, p, strict=True):
        if not np.isfinite(d):
            out.append(None)
            continue
        if rule == "effect_size":
            out.append("A" if d < 1.0 else ("B" if d < 1.5 else "C"))
            continue
        if d < 1.0 or not (np.isfinite(pv) and pv < alpha):
            out.append("A")
        elif d >= 1.5 and np.isfinite(se) and se > 0 and (d - 1.0) / se > z:
            out.append("C")
        else:
            out.append("B")
    return out


def mantel_haenszel(
    data: pd.DataFrame | np.ndarray,
    *,
    group: GroupLabels,
    focal: object,
    items: Sequence[str] | None = None,
    anchor_items: Sequence[str] | None = None,
    matching: pd.Series | np.ndarray | Sequence[float] | None = None,
    purify: bool = False,
    max_iter: int = 10,
    p_adjust: PAdjust = "none",
    alpha: float = 0.05,
    continuity_correction: bool = True,
    classification: Literal["ets", "effect_size"] = "ets",
) -> DIFResult:
    r"""Mantel–Haenszel DIF for 0 / 1 items and two groups (Holland & Thayer, 1988).

    For each studied item the respondents are stratified on the matching score and

    .. math::
        \chi^2_{MH}=\frac{\big(\lvert\sum_kA_k-\sum_kE(A_k)\rvert-\tfrac12\big)^2}
        {\sum_k\operatorname{Var}(A_k)},\qquad
        \hat\alpha_{MH}=\frac{\sum_kA_kD_k/T_k}{\sum_kB_kC_k/T_k},\qquad
        \Delta_{MH}=-2.35\ln\hat\alpha_{MH},

    with :math:`A_k, B_k` the reference respondents answering correctly / incorrectly in
    stratum :math:`k`, :math:`C_k, D_k` the focal ones and :math:`T_k` the stratum size (strata
    of one respondent are left out; Mantel & Haenszel, 1959). :math:`\hat\alpha_{MH}>1`
    (:math:`\Delta_{MH}<0`) means the item favours the reference group. The standard error of
    :math:`\ln\hat\alpha_{MH}` is Phillips & Holland's (1987); :math:`SE(\Delta_{MH})=2.35\,
    SE(\ln\hat\alpha_{MH})`. ETS classes (Zieky, 1993; Zwick, 2012): **A** if
    :math:`\lvert\Delta_{MH}\rvert<1` or :math:`\chi^2_{MH}` is not significant at ``alpha``;
    **C** if :math:`\lvert\Delta_{MH}\rvert\ge1.5` and significantly above 1
    (:math:`(\lvert\Delta_{MH}\rvert-1)/SE>z_{1-\alpha}`); **B** otherwise.
    ``classification="effect_size"`` uses the thresholds only (A < 1 ≤ B < 1.5 ≤ C), the codes
    difR prints. difR: ``difMH()``.

    Parameters
    ----------
    data : pandas.DataFrame or numpy.ndarray
        Respondents x items. Studied items are scored 0 / 1; anchor items may have any numeric
        scores (they only enter the total). No missing values on the items used.
    group : pandas.Series, numpy.ndarray or sequence
        Group label of each respondent.
    focal : object
        Label of the focal group; every other respondent is in the reference group.
    items : sequence of str, optional
        Items to test (default: all).
    anchor_items : sequence of str, optional
        Items whose total is the matching score (default: all). The studied item is always
        part of its own matching score.
    matching : array-like, optional
        An external matching variable (one value per respondent; each distinct value is a
        stratum) instead of the total score. Not with ``anchor_items`` or ``purify``.
    purify : bool, default False
        Iterative purification: items flagged (unadjusted ``p < alpha``) leave the anchors and
        the tests are repeated until the flagged set repeats or ``max_iter`` is reached.
    max_iter : int, default 10
        Maximum number of purification iterations.
    p_adjust : {"none", "holm", "BH"}, default "none"
        Adjustment of the final p-values over the studied items (Holm, 1979; Benjamini &
        Hochberg, 1995). Purification uses unadjusted p-values (difR's ``puriadjType =
        "simple"``).
    alpha : float, default 0.05
        Significance level for ``flagged``, purification and the ETS classes.
    continuity_correction : bool, default True
        Subtract 1/2 from :math:`\lvert\sum A-\sum E(A)\rvert` (Holland & Thayer, 1988; difR).
        ``stats::mantelhaen.test`` skips the correction when the difference is below 1/2.
    classification : {"ets", "effect_size"}, default "ets"
        The full ETS rule, or thresholds on :math:`\lvert\Delta_{MH}\rvert` only.

    Returns
    -------
    DIFResult
        ``table`` columns: ``item``, ``chi2``, ``odds_ratio``, ``log_odds_ratio``,
        ``log_odds_ratio_se``, ``delta``, ``delta_se``, ``ets_class``, ``p``, ``adj_p``,
        ``flagged``.

    Raises
    ------
    SpecificationError
        Missing responses, studied items not 0 / 1, an unknown item or focal label, or
        inconsistent options.
    """
    if classification not in ("ets", "effect_size"):
        raise SpecificationError(
            f"Unknown classification={classification!r}; use 'ets' or 'effect_size'."
        )
    prep = _prepare(
        data,
        group,
        [focal],
        items=items,
        anchor_items=anchor_items,
        matching=matching,
        purify=purify,
        max_iter=max_iter,
        p_adjust=p_adjust,
        alpha=alpha,
        what="mantel_haenszel()",
    )
    _require_binary(prep, "mantel_haenszel()")

    def statistic(anchors: np.ndarray) -> dict[str, np.ndarray]:
        rows = [
            _dif.mantel_haenszel_item(
                prep.values[:, j],
                prep.member,
                _matching_score(prep, j, anchors),
                continuity_correction=continuity_correction,
            )
            for j in prep.tested
        ]
        chi2 = np.array([r["chi2"] for r in rows])
        return {
            "chi2": chi2,
            "odds_ratio": np.array([r["odds_ratio"] for r in rows]),
            "var": np.array([r["var_log_odds_ratio"] for r in rows]),
            "p": _dif.chi2_sf(chi2, 1),
        }

    run = _run(prep, statistic, purify=purify, max_iter=max_iter, alpha=alpha)
    stats = run[0]
    odds = stats["odds_ratio"]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_odds = np.log(odds)
    log_se = np.sqrt(stats["var"])
    delta = -2.35 * log_odds
    delta_se = 2.35 * log_se
    _warn_sparse(
        [prep.names[j] for j, o in zip(prep.tested, odds, strict=True) if not (0 < o < np.inf)]
    )
    columns: dict[str, np.ndarray | list[str | None]] = {
        "chi2": stats["chi2"],
        "odds_ratio": odds,
        "log_odds_ratio": log_odds,
        "log_odds_ratio_se": log_se,
        "delta": delta,
        "delta_se": delta_se,
        "ets_class": _ets_class(delta, delta_se, stats["p"], alpha, classification),
    }
    return _result(prep, columns, run, p_adjust=p_adjust, alpha=alpha, method="mantel_haenszel")


# ---------------------------------------------------------------------------------------------
# Mantel (ordered polytomous items)
# ---------------------------------------------------------------------------------------------


def mantel(
    data: pd.DataFrame | np.ndarray,
    *,
    group: GroupLabels,
    focal: object,
    items: Sequence[str] | None = None,
    anchor_items: Sequence[str] | None = None,
    matching: pd.Series | np.ndarray | Sequence[float] | None = None,
    purify: bool = False,
    max_iter: int = 10,
    p_adjust: PAdjust = "none",
    alpha: float = 0.05,
) -> DIFResult:
    r"""Mantel's (1963) test for ordered polytomous items and two groups.

    Compares, stratum by stratum, the sum of the focal group's item scores with its
    expectation under no DIF (:func:`flexpyctt._dif.mantel_item`):

    .. math::
        \chi^2_M=\frac{\big(\sum_kF_k-\sum_kE(F_k)\big)^2}{\sum_k\operatorname{Var}(F_k)},
        \qquad df=1,

    the scores being the item's numeric values. For 0 / 1 items it equals the Mantel–Haenszel
    statistic without continuity correction. The effect size is the Liu & Agresti (1996)
    cumulative common odds ratio :math:`\hat\psi` (Penfield & Algina, 2003), oriented as
    :math:`\hat\alpha_{MH}` (values above 1 favour the reference group; it reduces to
    :math:`\hat\alpha_{MH}` for 0 / 1 items), with the delta-method standard error of
    :math:`\ln\hat\psi` (large strata). difR: ``difMantel.poly()``, ``liu_agresti_ccor()``
    (which report :math:`1/\hat\psi`).

    Parameters
    ----------
    data, group, focal, items, anchor_items, matching, purify, max_iter, p_adjust, alpha
        As :func:`mantel_haenszel`; the studied items may have any ordered numeric scores.

    Returns
    -------
    DIFResult
        ``table`` columns: ``item``, ``chi2``, ``df``, ``cumulative_odds_ratio``,
        ``log_cumulative_odds_ratio``, ``log_cumulative_odds_ratio_se``, ``p``, ``adj_p``,
        ``flagged``.
    """
    prep = _prepare(
        data,
        group,
        [focal],
        items=items,
        anchor_items=anchor_items,
        matching=matching,
        purify=purify,
        max_iter=max_iter,
        p_adjust=p_adjust,
        alpha=alpha,
        what="mantel()",
    )

    def statistic(anchors: np.ndarray) -> dict[str, np.ndarray]:
        chi2, psi, se = [], [], []
        for j in prep.tested:
            score = _matching_score(prep, j, anchors)
            x = prep.values[:, j]
            chi2.append(_dif.mantel_item(x, prep.member, score))
            ratio, log_se = _dif.liu_agresti_item(x, prep.member, score)
            psi.append(ratio)
            se.append(log_se)
        chi2_arr = np.array(chi2)
        return {
            "chi2": chi2_arr,
            "psi": np.array(psi),
            "se": np.array(se),
            "p": _dif.chi2_sf(chi2_arr, 1),
        }

    run = _run(prep, statistic, purify=purify, max_iter=max_iter, alpha=alpha)
    stats = run[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_psi = np.log(stats["psi"])
    columns: dict[str, np.ndarray | list[str | None]] = {
        "chi2": stats["chi2"],
        "df": np.ones(prep.tested.size, dtype=np.int64),
        "cumulative_odds_ratio": stats["psi"],
        "log_cumulative_odds_ratio": log_psi,
        "log_cumulative_odds_ratio_se": stats["se"],
    }
    return _result(prep, columns, run, p_adjust=p_adjust, alpha=alpha, method="mantel")


# ---------------------------------------------------------------------------------------------
# Generalized Mantel–Haenszel
# ---------------------------------------------------------------------------------------------


def generalized_mantel_haenszel(
    data: pd.DataFrame | np.ndarray,
    *,
    group: GroupLabels,
    focal: Sequence[object],
    items: Sequence[str] | None = None,
    anchor_items: Sequence[str] | None = None,
    matching: pd.Series | np.ndarray | Sequence[float] | None = None,
    purify: bool = False,
    max_iter: int = 10,
    p_adjust: PAdjust = "none",
    alpha: float = 0.05,
) -> DIFResult:
    r"""Generalized Mantel–Haenszel test: several focal groups and / or nominal categories.

    In each stratum the respondents form a groups x categories table (the reference group and
    each focal group; every observed score of the item is a category, unordered). The statistic
    compares the focal-group counts with their expectations given the margins
    (:func:`flexpyctt._dif.generalized_mh_item`; Somes, 1986; Penfield, 2001):

    .. math::
        Q_{GMH}=\Big(\sum_k\mathbf n_k-\mathbf m_k\Big)^\top\Big(\sum_k\mathbf V_k\Big)^{-1}
        \Big(\sum_k\mathbf n_k-\mathbf m_k\Big),\qquad df=(G-1)(C-1).

    With two groups and 0 / 1 items it is :math:`\chi^2_{MH}` without continuity correction.
    difR: ``difGMH()`` (0 / 1 items); ``stats::mantelhaen.test()`` on a groups x categories x
    strata table.

    Parameters
    ----------
    data, group, items, anchor_items, matching, purify, max_iter, p_adjust, alpha
        As :func:`mantel_haenszel`; the studied items may have any numeric scores, treated as
        unordered categories.
    focal : sequence
        Labels of the focal groups (one or more); every other respondent is in the reference
        group.

    Returns
    -------
    DIFResult
        ``table`` columns: ``item``, ``chi2``, ``df``, ``p``, ``adj_p``, ``flagged``.
    """
    if isinstance(focal, (str, bytes)) or not isinstance(focal, Sequence | np.ndarray | pd.Series):
        raise SpecificationError("focal must be a sequence of focal-group labels.")
    focal_list = list(focal)
    if not focal_list or len(set(map(str, focal_list))) != len(focal_list):
        raise SpecificationError("focal must list one or more distinct labels.")
    prep = _prepare(
        data,
        group,
        focal_list,
        items=items,
        anchor_items=anchor_items,
        matching=matching,
        purify=purify,
        max_iter=max_iter,
        p_adjust=p_adjust,
        alpha=alpha,
        what="generalized_mantel_haenszel()",
    )

    def statistic(anchors: np.ndarray) -> dict[str, np.ndarray]:
        out = [
            _dif.generalized_mh_item(
                prep.values[:, j], prep.member, _matching_score(prep, j, anchors), prep.n_groups
            )
            for j in prep.tested
        ]
        chi2 = np.array([o[0] for o in out])
        df = np.array([o[1] for o in out], dtype=np.int64)
        expected_df = np.array(
            [(prep.n_groups - 1) * (np.unique(prep.values[:, j]).size - 1) for j in prep.tested]
        )
        short = [
            prep.names[j] for j, d, e in zip(prep.tested, df, expected_df, strict=True) if 0 < d < e
        ]
        if short:
            warnings.warn(
                f"The covariance matrix is singular for {short}; a generalized inverse was used "
                "and df is its rank.",
                FlexpycttWarning,
                stacklevel=4,
            )
        with np.errstate(invalid="ignore"):
            p = np.where(df > 0, _dif.chi2_sf(chi2, np.maximum(df, 1)), np.nan)
        return {"chi2": chi2, "df": df, "p": p}

    run = _run(prep, statistic, purify=purify, max_iter=max_iter, alpha=alpha)
    columns = {"chi2": run[0]["chi2"], "df": run[0]["df"]}
    return _result(
        prep, columns, run, p_adjust=p_adjust, alpha=alpha, method="generalized_mantel_haenszel"
    )


# ---------------------------------------------------------------------------------------------
# Logistic regression
# ---------------------------------------------------------------------------------------------


def _classes(values: np.ndarray, cuts: tuple[float, float]) -> list[str | None]:
    return [
        None if not np.isfinite(v) else ("A" if v < cuts[0] else ("B" if v < cuts[1] else "C"))
        for v in values
    ]


def logistic_dif(
    data: pd.DataFrame | np.ndarray,
    *,
    group: GroupLabels,
    focal: object,
    items: Sequence[str] | None = None,
    anchor_items: Sequence[str] | None = None,
    matching: pd.Series | np.ndarray | Sequence[float] | None = None,
    purify: bool = False,
    max_iter: int = 10,
    p_adjust: PAdjust = "none",
    alpha: float = 0.05,
    kind: Literal["uniform", "non_uniform", "both"] = "both",
    test: Literal["likelihood_ratio", "wald"] = "likelihood_ratio",
) -> DIFResult:
    r"""Logistic-regression DIF for 0 / 1 items and two groups (Swaminathan & Rogers, 1990).

    With :math:`S` the matching score and :math:`G` the focal indicator, nested models

    .. math::
        M_0:\ \operatorname{logit}P(x=1)=\beta_0+\beta_1S,\quad
        M_1:\ \dots+\beta_2G,\quad
        M_2:\ \dots+\beta_2G+\beta_3SG

    are fitted by maximum likelihood (Newton–Raphson). ``kind="both"`` compares :math:`M_0`
    with :math:`M_2` (:math:`df=2`), ``"uniform"`` :math:`M_0` with :math:`M_1` and
    ``"non_uniform"`` :math:`M_1` with :math:`M_2` (:math:`df=1`), by the likelihood-ratio
    statistic (difference of deviances) or the Wald statistic of the added coefficients in
    the larger model. The effect size is the difference of Nagelkerke's :math:`R^2` of the two
    models (Zumbo, 1999), with
    :math:`R^2_N=(1-e^{(D-D_{\text{null}})/N})/(1-e^{-D_{\text{null}}/N})`;
    classes A / B / C by the thresholds of Zumbo & Thomas (1997; 0.13, 0.26) and of Jodoin &
    Gierl (2001; 0.035, 0.07) on :math:`\Delta R^2` only (significance is in ``flagged``), as
    difR prints them. difR: ``difLogistic()``.

    Parameters
    ----------
    data, group, focal, items, anchor_items, matching, purify, max_iter, p_adjust, alpha
        As :func:`mantel_haenszel`. An external ``matching`` may be continuous.
    kind : {"both", "uniform", "non_uniform"}, default "both"
        Which DIF to test.
    test : {"likelihood_ratio", "wald"}, default "likelihood_ratio"
        The test statistic.

    Returns
    -------
    DIFResult
        ``table`` columns: ``item``, ``chi2``, ``df``, ``delta_r2``, ``zumbo_thomas_class``,
        ``jodoin_gierl_class``, ``group_coefficient``, ``group_coefficient_se``,
        ``interaction_coefficient``, ``interaction_coefficient_se`` (coefficients of :math:`G`
        and :math:`SG` in the larger model; NaN when it has no such term), ``p``, ``adj_p``,
        ``flagged``.

    Warns
    -----
    ConvergenceWarning
        A logistic regression did not converge (for example under complete separation).
    """
    if kind not in ("uniform", "non_uniform", "both"):
        raise SpecificationError(f"Unknown kind={kind!r}; use 'uniform', 'non_uniform' or 'both'.")
    if test not in ("likelihood_ratio", "wald"):
        raise SpecificationError(f"Unknown test={test!r}; use 'likelihood_ratio' or 'wald'.")
    prep = _prepare(
        data,
        group,
        [focal],
        items=items,
        anchor_items=anchor_items,
        matching=matching,
        purify=purify,
        max_iter=max_iter,
        p_adjust=p_adjust,
        alpha=alpha,
        what="logistic_dif()",
    )
    _require_binary(prep, "logistic_dif()")
    n_obs = prep.values.shape[0]
    g = (prep.member == 1).astype(np.float64)
    df = 2 if kind == "both" else 1
    # Columns of the larger / smaller model: 0 intercept, 1 score, 2 group, 3 interaction.
    large = [0, 1, 2] if kind == "uniform" else [0, 1, 2, 3]
    small = [0, 1, 2] if kind == "non_uniform" else [0, 1]
    tested_terms = {"uniform": [2], "non_uniform": [3], "both": [2, 3]}[kind]

    def statistic(anchors: np.ndarray) -> dict[str, np.ndarray]:
        out: dict[str, list[float]] = {k: [] for k in ("chi2", "r2", "bg", "sg", "bi", "si")}
        unconverged = []
        for j in prep.tested:
            y = prep.values[:, j]
            s = _matching_score(prep, j, anchors)
            full = np.column_stack([np.ones(n_obs), s, g, s * g])
            b1, v1, dev1, ok1 = _dif.logistic_fit(y, full[:, large])
            _, _, dev0, ok0 = _dif.logistic_fit(y, full[:, small])
            if not (ok0 and ok1):
                unconverged.append(prep.names[j])
            mean = float(y.mean())
            null_dev = (
                0.0
                if mean in (0.0, 1.0)
                else float(-2.0 * n_obs * (mean * np.log(mean) + (1 - mean) * np.log1p(-mean)))
            )

            def r2(dev: float, null_dev: float = null_dev) -> float:
                top = 1 - np.exp((dev - null_dev) / n_obs)
                return float(top / (1 - np.exp(-null_dev / n_obs)))

            if test == "likelihood_ratio":
                stat = dev0 - dev1
            else:
                pos = [large.index(t) for t in tested_terms]
                coef = b1[pos]
                stat = float(coef @ np.linalg.solve(v1[np.ix_(pos, pos)], coef))
            out["chi2"].append(stat)
            out["r2"].append(r2(dev1) - r2(dev0) if null_dev > 0 else np.nan)
            out["bg"].append(b1[2])
            out["sg"].append(np.sqrt(v1[2, 2]))
            out["bi"].append(b1[3] if 3 in large else np.nan)
            out["si"].append(np.sqrt(v1[3, 3]) if 3 in large else np.nan)
        if unconverged:
            warnings.warn(
                f"Logistic regression did not converge for {unconverged} (separation?).",
                ConvergenceWarning,
                stacklevel=4,
            )
        res = {k: np.array(v, dtype=np.float64) for k, v in out.items()}
        res["p"] = _dif.chi2_sf(res["chi2"], df)
        return res

    run = _run(prep, statistic, purify=purify, max_iter=max_iter, alpha=alpha)
    stats = run[0]
    columns: dict[str, np.ndarray | list[str | None]] = {
        "chi2": stats["chi2"],
        "df": np.full(prep.tested.size, df, dtype=np.int64),
        "delta_r2": stats["r2"],
        "zumbo_thomas_class": _classes(stats["r2"], ZUMBO_THOMAS),
        "jodoin_gierl_class": _classes(stats["r2"], JODOIN_GIERL),
        "group_coefficient": stats["bg"],
        "group_coefficient_se": stats["sg"],
        "interaction_coefficient": stats["bi"],
        "interaction_coefficient_se": stats["si"],
    }
    return _result(prep, columns, run, p_adjust=p_adjust, alpha=alpha, method="logistic")
