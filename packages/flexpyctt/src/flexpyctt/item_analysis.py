"""Classical item analysis (DESIGN.md §16.21).

Scoring with an answer key (:func:`score_responses`), item statistics (:func:`item_analysis`),
coefficients alpha and omega (:func:`coefficient_alpha`, :func:`coefficient_omega`) and option
(distractor) analysis (:func:`distractor_analysis`). NumPy / SciPy / pandas only.
"""

from __future__ import annotations

import warnings
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.special import ndtri
from scipy.stats import f as f_distribution
from scipy.stats import norm

from flexpyctt import _efa
from flexpyctt.correlation import polychoric, smooth_correlation
from flexpyctt.dimensionality import _pearson, _warn_efa
from flexpyctt.errors import (
    FlexpycttWarning,
    HeywoodWarning,
    NotPositiveDefiniteWarning,
    SpecificationError,
)

__all__ = [
    "AlphaResult",
    "OmegaResult",
    "coefficient_alpha",
    "coefficient_omega",
    "distractor_analysis",
    "item_analysis",
    "score_responses",
]

MissingRule = Literal["pairwise", "listwise"]
ItemCorrelation = Literal["pearson", "polyserial"]
OmegaCorrelation = Literal["pearson", "polychoric"]
ScoringMissing = Literal["incorrect", "keep"]

_MISSING_RULES = ("pairwise", "listwise")
_ITEM_CORRELATIONS = ("pearson", "polyserial")
_OMEGA_CORRELATIONS = ("pearson", "polychoric")
_SCORING_MISSING = ("incorrect", "keep")

KeyLike = Mapping[str, object] | pd.Series | Sequence[object] | np.ndarray


# --------------------------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------------------------


def _coerce_scores(
    data: pd.DataFrame | np.ndarray, *, what: str, min_items: int
) -> tuple[list[str], np.ndarray]:
    """Return item names and a float ``(N, J)`` array of item scores (NaN = missing)."""
    if isinstance(data, pd.DataFrame):
        names = [str(c) for c in data.columns]
        try:
            values = data.to_numpy(dtype=np.float64, na_value=np.nan)
        except (TypeError, ValueError) as exc:
            raise SpecificationError(
                f"{what}() needs numeric item scores; could not convert data to float: {exc}"
            ) from exc
    else:
        try:
            values = np.asarray(data, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise SpecificationError(
                f"{what}() needs numeric item scores; could not convert data to float: {exc}"
            ) from exc
        if values.ndim != 2:
            raise SpecificationError(
                f"data must be two-dimensional (respondents x items), got shape {values.shape}."
            )
        names = [f"item{j + 1}" for j in range(values.shape[1])]
    if len(set(names)) != len(names):
        raise SpecificationError("Item (column) names must be unique.")
    if values.shape[1] < min_items:
        raise SpecificationError(
            f"{what}() needs at least {min_items} items, got {values.shape[1]}."
        )
    if np.isinf(values).any():
        raise SpecificationError("Item scores must be finite numbers or NaN (missing).")
    return names, values


def _apply_missing_rule(values: np.ndarray, missing: str) -> np.ndarray:
    """Drop incomplete rows (``"listwise"``) and rows without any response."""
    if missing not in _MISSING_RULES:
        raise SpecificationError(f"missing must be one of {_MISSING_RULES}, got {missing!r}.")
    observed = ~np.isnan(values)
    keep = observed.all(axis=1) if missing == "listwise" else observed.any(axis=1)
    out = values[keep]
    if out.shape[0] < 2:
        raise SpecificationError(
            f"Fewer than two respondents remain after missing={missing!r} "
            f"({out.shape[0]} of {values.shape[0]})."
        )
    return out


def _check_items(names: list[str], values: np.ndarray) -> None:
    """Every item needs two observed values; every pair two joint observations."""
    observed = ~np.isnan(values)
    for j, name in enumerate(names):
        if np.unique(values[observed[:, j], j]).size < 2:
            raise SpecificationError(f"Item {name!r} has fewer than two distinct observed scores.")
    joint = observed.T.astype(np.int64) @ observed.astype(np.int64)
    if (joint < 2).any():
        i, j = np.argwhere(joint < 2)[0]
        raise SpecificationError(
            f"Items {names[i]!r} and {names[j]!r} have fewer than two respondents with both "
            "scores observed."
        )


def _pairwise_covariance(values: np.ndarray) -> np.ndarray:
    r"""Pairwise-complete covariance matrix (R's ``cov(x, use = "pairwise")``).

    For each pair :math:`(i,j)` the means and the :math:`n_{ij}-1` denominator use the
    :math:`n_{ij}` respondents with both scores observed; the diagonal is each item's variance.
    """
    observed = ~np.isnan(values)
    # Centre by the observed means first (covariances are shift invariant) for precision.
    centred = np.where(observed, values - np.nanmean(values, axis=0), 0.0)
    mask = observed.astype(np.float64)
    n_joint = mask.T @ mask
    cross = centred.T @ centred
    sums = centred.T @ mask  # sums[i, j] = sum of x_i over rows where j is observed
    cov = np.asarray((cross - sums * sums.T / n_joint) / (n_joint - 1.0), dtype=np.float64)
    out: np.ndarray = (cov + cov.T) / 2.0
    return out


def _alpha(cov: np.ndarray) -> float:
    k = cov.shape[0]
    return float(k / (k - 1) * (1.0 - np.trace(cov) / cov.sum()))


def _standardized_alpha(corr: np.ndarray) -> float:
    k = corr.shape[0]
    return float(k / (k - 1) * (1.0 - k / corr.sum()))


def _drop(mat: np.ndarray, j: int) -> np.ndarray:
    keep = np.arange(mat.shape[0]) != j
    return mat[np.ix_(keep, keep)]


# --------------------------------------------------------------------------------------------
# Scoring with an answer key
# --------------------------------------------------------------------------------------------


def _as_frame(data: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
        frame.columns = [str(c) for c in frame.columns]
    else:
        arr = np.asarray(data, dtype=object)
        if arr.ndim != 2:
            raise SpecificationError(
                f"data must be two-dimensional (respondents x items), got shape {arr.shape}."
            )
        frame = pd.DataFrame(arr, columns=[f"item{j + 1}" for j in range(arr.shape[1])])
    if len(set(frame.columns)) != frame.shape[1]:
        raise SpecificationError("Item (column) names must be unique.")
    if frame.shape[1] < 1 or frame.shape[0] < 1:
        raise SpecificationError("data must have at least one respondent and one item.")
    return frame


def _as_key(key: KeyLike, items: list[str]) -> dict[str, object]:
    if isinstance(key, (pd.Series, Mapping)):
        mapping = {str(k): v for k, v in key.items()}
    elif isinstance(key, (Sequence, np.ndarray)) and not isinstance(key, (str, bytes)):
        values = list(key)
        if len(values) != len(items):
            raise SpecificationError(
                f"A key given as a sequence needs one entry per item ({len(items)}), "
                f"got {len(values)}."
            )
        mapping = dict(zip(items, values, strict=True))
    else:
        raise SpecificationError(
            "key must be a mapping or Series from item name to correct option, or a sequence "
            f"with one entry per item; got {type(key).__name__}."
        )
    missing_items = [name for name in items if name not in mapping]
    extra_items = [name for name in mapping if name not in items]
    if missing_items:
        raise SpecificationError(f"key has no entry for item(s) {missing_items[:5]}.")
    if extra_items:
        raise SpecificationError(f"key names item(s) not in data: {extra_items[:5]}.")
    for name in items:
        if _is_missing(mapping[name]):
            raise SpecificationError(f"The key of item {name!r} is missing.")
    return {name: mapping[name] for name in items}


def _is_missing(value: object) -> bool:
    out = pd.isna(value)  # type: ignore[call-overload]
    return bool(out) if np.ndim(out) == 0 else False


def _score_frame(frame: pd.DataFrame, key: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(correct, observed)`` boolean arrays and warn about keys never chosen."""
    correct = np.zeros(frame.shape, dtype=bool)
    observed = np.zeros(frame.shape, dtype=bool)
    never_chosen: list[str] = []
    for j, name in enumerate(frame.columns):
        column = frame[name]
        obs = column.notna().to_numpy()
        hit = np.array(
            [ok and bool(value == key[name]) for value, ok in zip(column, obs, strict=True)],
            dtype=bool,
        )
        correct[:, j] = hit
        observed[:, j] = obs
        if obs.any() and not hit.any():
            never_chosen.append(name)
    if never_chosen:
        warnings.warn(
            f"No respondent chose the keyed option of item(s) {never_chosen[:5]}"
            f"{' ...' if len(never_chosen) > 5 else ''}; check that the key uses the same codes "
            "(and types) as the data.",
            FlexpycttWarning,
            stacklevel=3,
        )
    return correct, observed


def score_responses(
    data: pd.DataFrame | np.ndarray,
    *,
    key: KeyLike,
    missing: ScoringMissing = "incorrect",
) -> pd.DataFrame:
    r"""Score raw responses with an answer key.

    The item score is :math:`x_{ij}=1` if respondent :math:`i` chose item :math:`j`'s keyed
    option and 0 otherwise, as psych's ``score.multiple.choice(key, data, score = FALSE)``.

    Parameters
    ----------
    data : pandas.DataFrame or numpy.ndarray
        Respondents x items of chosen options (any codes: numbers or strings); NaN / None is a
        missing (omitted) response.
    key : mapping, pandas.Series or sequence
        Correct option per item: a mapping or Series from item name to option, or a sequence in
        column order. Options are compared with ``==``, so the key must use the data's codes and
        types (``1`` does not match ``"1"``).
    missing : {"incorrect", "keep"}, default "incorrect"
        ``"incorrect"`` scores a missing response 0; ``"keep"`` keeps it missing (NaN).

    Returns
    -------
    pandas.DataFrame
        0 / 1 item scores with the data's index and columns (integer with ``"incorrect"``,
        float with NaN for missing responses with ``"keep"``).

    Raises
    ------
    SpecificationError
        If the key does not name exactly the data's items or a key entry is missing.

    Warns
    -----
    FlexpycttWarning
        If no respondent chose some item's keyed option (often a key with the wrong codes).
    """
    if missing not in _SCORING_MISSING:
        raise SpecificationError(f"missing must be one of {_SCORING_MISSING}, got {missing!r}.")
    frame = _as_frame(data)
    items = list(frame.columns)
    correct, observed = _score_frame(frame, _as_key(key, items))
    if missing == "incorrect":
        return pd.DataFrame(correct.astype(np.int64), index=frame.index, columns=items)
    scores = np.where(observed, correct.astype(np.float64), np.nan)
    return pd.DataFrame(scores, index=frame.index, columns=items)


# --------------------------------------------------------------------------------------------
# Coefficient alpha
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AlphaResult:
    r"""Result of :func:`coefficient_alpha`.

    Attributes
    ----------
    alpha : float
        Coefficient alpha (KR-20 for 0 / 1 items).
    standardized_alpha : float
        Alpha of the standardized items, from the correlation matrix.
    confidence_interval : tuple of float
        Feldt's (lower, upper) interval for ``alpha`` at ``confidence_level``.
    confidence_level : float
        Level of ``confidence_interval``.
    n_items : int
        Number of items.
    n_obs : int
        Respondents used (``"listwise"``: complete rows; ``"pairwise"``: rows with at least one
        response); the :math:`n` of the Feldt interval.
    if_deleted : pandas.DataFrame
        One row per item: ``alpha`` and ``standardized_alpha`` of the other items (NaN with two
        items).
    """

    alpha: float
    standardized_alpha: float
    confidence_interval: tuple[float, float]
    confidence_level: float
    n_items: int
    n_obs: int
    if_deleted: pd.DataFrame


def coefficient_alpha(
    data: pd.DataFrame | np.ndarray,
    *,
    missing: MissingRule = "pairwise",
    confidence_level: float = 0.95,
) -> AlphaResult:
    r"""Coefficient alpha (Cronbach, 1951), standardized alpha and Feldt's interval.

    With :math:`k` items and item covariance matrix :math:`C`,

    .. math::
        \alpha=\frac{k}{k-1}\Bigl(1-\frac{\operatorname{tr}C}{\mathbf 1^\top C\mathbf 1}\Bigr),
        \qquad
        \alpha_{\mathrm{std}}=\frac{k}{k-1}\Bigl(1-\frac{k}{\mathbf 1^\top R\mathbf 1}\Bigr)
        =\frac{k\bar r}{1+(k-1)\bar r}

    with :math:`R` the correlation matrix and :math:`\bar r` its mean off-diagonal entry. For
    0 / 1 items :math:`\alpha` is KR-20 (Kuder & Richardson, 1937). Feldt's interval (Feldt,
    Woodruff & Salih, 1987) at level :math:`1-\gamma`, with :math:`n` respondents, is

    .. math::
        \bigl[1-(1-\alpha)F_{1-\gamma/2},\;1-(1-\alpha)F_{\gamma/2}\bigr],\quad
        F_q=F^{-1}\bigl(q;\,n-1,\,(n-1)(k-1)\bigr).

    This is psych's ``alpha()`` (``total$raw_alpha``, ``std.alpha``, ``alpha.drop``,
    ``feldt``); with ``missing="listwise"`` also ltm's ``cronbach.alpha(na.rm = TRUE)``.

    Parameters
    ----------
    data : pandas.DataFrame or numpy.ndarray
        Respondents x items of item scores (0 / 1 or ordered category scores); NaN is missing.
        At least two items.
    missing : {"pairwise", "listwise"}, default "pairwise"
        ``"pairwise"``: each covariance and correlation from the respondents who answered both
        items (psych's default ``use = "pairwise"``); ``"listwise"``: complete rows only.
    confidence_level : float, default 0.95
        Level of the Feldt interval, in (0, 1).

    Returns
    -------
    AlphaResult

    Raises
    ------
    SpecificationError
        If the data are not numeric, an item is constant, or a pair of items has fewer than two
        joint observations.

    Notes
    -----
    ``if_deleted`` removes the item's row and column from :math:`C` and :math:`R`, as psych.
    Under ``"pairwise"``, :math:`C` and :math:`R` need not be positive semi-definite, and the
    Feldt interval uses every respondent with a response, as psych uses every row.
    """
    if not (0.0 < confidence_level < 1.0):
        raise SpecificationError(f"confidence_level must be in (0, 1), got {confidence_level!r}.")
    names, raw = _coerce_scores(data, what="coefficient_alpha", min_items=2)
    values = _apply_missing_rule(raw, missing)
    _check_items(names, values)
    cov = _pairwise_covariance(values)
    corr = _pearson(values)
    k = len(names)
    n_obs = int(values.shape[0])
    alpha = _alpha(cov)
    gamma = 1.0 - confidence_level
    df1, df2 = n_obs - 1, (n_obs - 1) * (k - 1)
    lower = 1.0 - (1.0 - alpha) * float(f_distribution.ppf(1.0 - gamma / 2.0, df1, df2))
    upper = 1.0 - (1.0 - alpha) * float(f_distribution.ppf(gamma / 2.0, df1, df2))
    if k > 2:
        dropped = [(_alpha(_drop(cov, j)), _standardized_alpha(_drop(corr, j))) for j in range(k)]
    else:
        dropped = [(np.nan, np.nan)] * k
    return AlphaResult(
        alpha=alpha,
        standardized_alpha=_standardized_alpha(corr),
        confidence_interval=(lower, upper),
        confidence_level=float(confidence_level),
        n_items=k,
        n_obs=n_obs,
        if_deleted=pd.DataFrame(
            dropped, index=pd.Index(names, name="item"), columns=["alpha", "standardized_alpha"]
        ),
    )


# --------------------------------------------------------------------------------------------
# Item statistics
# --------------------------------------------------------------------------------------------


def _polyserial_factor(column: np.ndarray) -> float:
    r""":math:`\sigma_y/\sum_k\phi(\tau_k)` of one item over its observed scores."""
    observed = column[~np.isnan(column)]
    _, counts = np.unique(observed, return_counts=True)
    cumulative = np.cumsum(counts)[:-1] / observed.size
    density = norm.pdf(ndtri(cumulative)).sum()
    return float(observed.std(ddof=0) / density)


def item_analysis(
    data: pd.DataFrame | np.ndarray,
    *,
    correlation: ItemCorrelation = "pearson",
    missing: MissingRule = "pairwise",
) -> pd.DataFrame:
    r"""Classical item statistics: difficulty, item–total and item–rest correlations.

    For item :math:`j` with mean :math:`\bar x_j` and largest observed score
    :math:`m_j`, the difficulty is :math:`\bar x_j/m_j` (the proportion correct for 0 / 1
    items). With the item covariance matrix :math:`C` and :math:`c_j=\sum_l C_{jl}`, the
    item–total and item–rest (corrected item–total) correlations are

    .. math::
        r_{j,T}=\frac{c_j}{\sqrt{C_{jj}\,\mathbf 1^\top C\mathbf 1}},\qquad
        r_{j,T-j}=\frac{c_j-C_{jj}}{\sqrt{C_{jj}\,(\mathbf 1^\top C\mathbf 1-2c_j+C_{jj})}},

    the Pearson correlations of the item with the total score :math:`T=\sum_l x_l` and with the
    rest score :math:`T-x_j` when the data are complete (point-biserial for 0 / 1 items).
    With ``correlation="polyserial"`` each is converted to the polyserial correlation (biserial
    for 0 / 1 items) by the two-step ("ad hoc") estimator of Olsson, Drasgow & Dorans (1982),

    .. math::
        \hat\rho_{j}=\frac{r\,\sigma_j}{\sum_{k=1}^{K_j-1}\phi(\tau_{jk})},\qquad
        \tau_{jk}=\Phi^{-1}\bigl(\hat P(x_j\le k\text{-th observed score})\bigr),

    with :math:`\sigma_j` the item's standard deviation (denominator :math:`n`); for 0 / 1 items
    this is :math:`r\sqrt{pq}/\phi(\Phi^{-1}(p))`. This is polycor's ``polyserial(ML = FALSE)``.
    ``alpha_if_deleted`` is coefficient alpha without the item (:func:`coefficient_alpha`).

    Parameters
    ----------
    data : pandas.DataFrame or numpy.ndarray
        Respondents x items of item scores (0 / 1 or ordered category scores, at least 0); NaN is
        missing. At least two items.
    correlation : {"pearson", "polyserial"}, default "pearson"
        Pearson (point-biserial for 0 / 1 items) or polyserial (biserial) item–total and
        item–rest correlations.
    missing : {"pairwise", "listwise"}, default "pairwise"
        ``"pairwise"``: item means, standard deviations, thresholds from each item's observed
        scores and :math:`C` from pairwise-complete covariances; ``"listwise"``: complete rows
        only.

    Returns
    -------
    pandas.DataFrame
        One row per item: ``item``, ``n`` (observed scores), ``n_categories`` (distinct observed
        scores), ``mean``, ``sd`` (denominator :math:`n-1`), ``difficulty``, ``item_total``,
        ``item_rest``, ``alpha_if_deleted`` (NaN with two items).

    Raises
    ------
    SpecificationError
        If the data are not numeric, a score is negative, an item is constant, or a pair of
        items has fewer than two joint observations.

    Warns
    -----
    HeywoodWarning
        If a polyserial estimate is outside [-1, 1] (it is set to ±1, as psych does).

    Notes
    -----
    With missing data the correlations are those implied by the pairwise covariance matrix,
    the quantities from which ``alpha`` is computed: psych's ``alpha()`` reports the same
    ``r.drop`` but computes ``raw.r`` from the sum of each respondent's observed scores, which
    differs from ``item_total`` when responses are missing.
    """
    if correlation not in _ITEM_CORRELATIONS:
        raise SpecificationError(
            f"correlation must be one of {_ITEM_CORRELATIONS}, got {correlation!r}."
        )
    names, raw = _coerce_scores(data, what="item_analysis", min_items=2)
    values = _apply_missing_rule(raw, missing)
    if (values < 0).any():
        raise SpecificationError(
            "item_analysis() needs item scores of at least 0 (difficulty is the mean divided by "
            "the largest score)."
        )
    _check_items(names, values)
    cov = _pairwise_covariance(values)
    k = len(names)
    sums = cov.sum(axis=1)
    total_var = cov.sum()
    diag = np.diag(cov)
    item_total = sums / np.sqrt(diag * total_var)
    item_rest = (sums - diag) / np.sqrt(diag * (total_var - 2.0 * sums + diag))
    if correlation == "polyserial":
        factor = np.array([_polyserial_factor(values[:, j]) for j in range(k)])
        item_total, item_rest = item_total * factor, item_rest * factor
        outside = (np.abs(item_total) > 1.0) | (np.abs(item_rest) > 1.0)
        if outside.any():
            flagged = [names[j] for j in np.flatnonzero(outside)]
            warnings.warn(
                f"Polyserial estimates outside [-1, 1] were set to ±1 for item(s) {flagged[:5]}.",
                HeywoodWarning,
                stacklevel=2,
            )
            item_total = np.clip(item_total, -1.0, 1.0)
            item_rest = np.clip(item_rest, -1.0, 1.0)
    observed = ~np.isnan(values)
    means = np.nanmean(values, axis=0)
    maxima = np.nanmax(values, axis=0)
    alpha_dropped = (
        np.array([_alpha(_drop(cov, j)) for j in range(k)]) if k > 2 else np.full(k, np.nan)
    )
    return pd.DataFrame(
        {
            "item": names,
            "n": observed.sum(axis=0).astype(np.int64),
            "n_categories": [int(np.unique(values[observed[:, j], j]).size) for j in range(k)],
            "mean": means,
            "sd": np.nanstd(values, axis=0, ddof=1),
            "difficulty": means / maxima,
            "item_total": item_total,
            "item_rest": item_rest,
            "alpha_if_deleted": alpha_dropped,
        }
    )


# --------------------------------------------------------------------------------------------
# Coefficient omega
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class OmegaResult:
    r"""Result of :func:`coefficient_omega`.

    Attributes
    ----------
    omega : float
        :math:`\omega_t` of the one-factor model.
    loadings : pandas.Series
        One-factor ULS (minres) loadings of the items as scored, signed to a non-negative sum.
    uniquenesses : pandas.Series
        :math:`1-\lambda_j^2`.
    correlation : pandas.DataFrame
        The correlation matrix the factor model was fitted to, items as scored (smoothed if it
        was not positive definite).
    reversed_items : tuple of str
        Items with a negative loading, reversed before computing ``omega`` (empty if none).
    """

    omega: float
    loadings: pd.Series
    uniquenesses: pd.Series
    correlation: pd.DataFrame
    reversed_items: tuple[str, ...]


def coefficient_omega(
    data: pd.DataFrame | np.ndarray,
    *,
    correlation: OmegaCorrelation = "pearson",
) -> OmegaResult:
    r"""McDonald's omega total from a one-factor model.

    A one-factor model is fitted to the item correlation matrix :math:`R` by unweighted least
    squares (minres, :func:`~flexpyctt.efa`) and

    .. math::
        \omega_t=1-\frac{\sum_j(1-\lambda_j^2)}{\mathbf s^\top R\,\mathbf s},\qquad
        s_j=\operatorname{sign}(\lambda_j)\ (s_j=1\text{ if }\lambda_j=0)

    (McDonald, 1999; in this form Revelle & Zinbarg, 2009), the reliability of the sum of the
    standardized items with the items that load negatively reversed, with the observed total
    variance :math:`\mathbf s^\top R\mathbf s` in the denominator. This is psych's
    ``omega(m = R, nfactors = 1)$omega.tot`` with its default ``flip = TRUE``; when every
    loading is non-negative, :math:`\mathbf s=\mathbf 1`.

    Parameters
    ----------
    data : pandas.DataFrame or numpy.ndarray
        Respondents x items of item scores; NaN is missing. At least three items.
    correlation : {"pearson", "polychoric"}, default "pearson"
        ``"pearson"``: pairwise-complete Pearson correlations; ``"polychoric"``:
        :func:`~flexpyctt.polychoric` (tetrachoric for 0 / 1 items) with its defaults.

    Returns
    -------
    OmegaResult

    Raises
    ------
    SpecificationError
        If the data are not numeric, an item is constant, or a correlation is undefined.

    Warns
    -----
    FlexpycttWarning
        If some items load negatively and were reversed (reverse-score them, or drop them, to
        get the omega of the total as scored).
    NotPositiveDefiniteWarning
        If the correlation matrix was smoothed before the factor analysis.
    HeywoodWarning, ConvergenceWarning
        From the factor analysis.

    Notes
    -----
    Hierarchical omega (:math:`\omega_h`) needs a rotated multi-factor solution and a
    Schmid–Leiman transformation and is not provided.
    """
    if correlation not in _OMEGA_CORRELATIONS:
        raise SpecificationError(
            f"correlation must be one of {_OMEGA_CORRELATIONS}, got {correlation!r}."
        )
    names, values = _coerce_scores(data, what="coefficient_omega", min_items=3)
    values = _apply_missing_rule(values, "pairwise")
    _check_items(names, values)
    if correlation == "polychoric":
        mat = polychoric(pd.DataFrame(values, columns=names)).correlation.to_numpy(dtype=np.float64)
    else:
        mat, smoothed = smooth_correlation(_pearson(values))
        if smoothed:
            warnings.warn(
                "The pairwise Pearson correlation matrix was not positive definite and was "
                "smoothed.",
                NotPositiveDefiniteWarning,
                stacklevel=2,
            )
    solution = _efa.fit_factors(mat, 1, "uls")
    _warn_efa(solution, 1)
    loadings = solution.loadings[:, 0]
    uniquenesses = 1.0 - loadings**2
    signs = np.where(loadings < 0.0, -1.0, 1.0)
    reversed_items = tuple(names[j] for j in np.flatnonzero(signs < 0.0))
    if reversed_items:
        warnings.warn(
            f"Item(s) {list(reversed_items[:5])}{' ...' if len(reversed_items) > 5 else ''} "
            "load negatively and were reversed before computing omega (as psych's omega() "
            "with flip = TRUE).",
            FlexpycttWarning,
            stacklevel=2,
        )
    omega = float(1.0 - uniquenesses.sum() / (signs @ mat @ signs))
    return OmegaResult(
        omega=omega,
        loadings=pd.Series(loadings, index=names, name="loading"),
        uniquenesses=pd.Series(uniquenesses, index=names, name="uniqueness"),
        correlation=pd.DataFrame(mat, index=names, columns=names),
        reversed_items=reversed_items,
    )


# --------------------------------------------------------------------------------------------
# Distractor analysis
# --------------------------------------------------------------------------------------------


def _sorted_options(values: list[Hashable]) -> list[Hashable]:
    try:
        return sorted(values)  # type: ignore[type-var]
    except TypeError:
        return sorted(values, key=lambda v: (type(v).__name__, str(v)))


def _score_groups(total: np.ndarray, n_groups: int) -> np.ndarray:
    """Group index 0..G-1 by R's ``cut(total, quantile(total, 0:G / G), include.lowest = TRUE)``."""
    breaks = np.quantile(total, np.linspace(0.0, 1.0, n_groups + 1))
    if np.unique(breaks).size != breaks.size:
        raise SpecificationError(
            f"The total-score quantiles for n_groups={n_groups} are not distinct "
            f"({np.round(breaks, 3).tolist()}); use fewer groups."
        )
    # (b_{g}, b_{g+1}] for g >= 1 and [b_0, b_1] for the first group.
    return np.clip(np.searchsorted(breaks, total, side="left") - 1, 0, n_groups - 1)


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def distractor_analysis(
    data: pd.DataFrame | np.ndarray,
    *,
    key: KeyLike,
    n_groups: int = 3,
) -> pd.DataFrame:
    r"""Option (distractor) analysis of multiple-choice responses.

    Responses are scored with ``key`` (a missing response scores 0), giving the total score
    :math:`T` and, for item :math:`j`, the rest score :math:`T-x_j`. For every option of every
    item, among the respondents who answered the item, the table gives how many chose it, the
    point-biserial correlation of choosing it (0 / 1) with the rest score, the mean total score
    of those who chose it, and the proportion choosing it in each of ``n_groups`` total-score
    groups. The groups are cut at the quantiles of :math:`T` over all respondents (R's
    ``quantile`` type 7) as ``cut(T, quantile(T, 0:G / G), include.lowest = TRUE)``, so tied
    scores stay in one group. A working distractor has a negative point-biserial and a
    proportion that falls from the low to the high group.

    Parameters
    ----------
    data : pandas.DataFrame or numpy.ndarray
        Respondents x items of chosen options (any codes); NaN / None is missing.
    key : mapping, pandas.Series or sequence
        Correct option per item, as in :func:`score_responses`.
    n_groups : int, default 3
        Number of total-score groups (at least 2).

    Returns
    -------
    pandas.DataFrame
        One row per item and option (observed options and the key, in sorted order):
        ``item``, ``option``, ``is_key``, ``n``, ``proportion`` (of the item's respondents),
        ``point_biserial``, ``mean_score`` (NaN if nobody chose it), ``proportion_group_1``
        ... ``proportion_group_<n_groups>`` (low to high).

    Raises
    ------
    SpecificationError
        If the key does not match the items, ``n_groups`` is not an integer of at least 2, or
        the total-score quantiles are not distinct.

    Warns
    -----
    FlexpycttWarning
        If no respondent chose some item's keyed option.
    """
    if not (isinstance(n_groups, (int, np.integer)) and n_groups >= 2):
        raise SpecificationError(f"n_groups must be an integer of at least 2, got {n_groups!r}.")
    frame = _as_frame(data)
    items = list(frame.columns)
    keys = _as_key(key, items)
    correct, observed = _score_frame(frame, keys)
    scores = correct.astype(np.float64)
    total = scores.sum(axis=1)
    groups = _score_groups(total, int(n_groups))
    rows: list[dict[str, object]] = []
    for j, name in enumerate(items):
        answered = observed[:, j]
        column = frame[name].to_numpy(dtype=object)
        chosen = column[answered]
        options = list(pd.unique(pd.Series(chosen, dtype=object)))
        if not any(bool(o == keys[name]) for o in options):
            options.append(keys[name])
        rest = (total - scores[:, j])[answered]
        total_answered = total[answered]
        group_answered = groups[answered]
        n_answered = int(answered.sum())
        for option in _sorted_options(options):
            picked = np.array([bool(v == option) for v in chosen], dtype=bool)
            count = int(picked.sum())
            row: dict[str, object] = {
                "item": name,
                "option": option,
                "is_key": bool(option == keys[name]),
                "n": count,
                "proportion": count / n_answered if n_answered else float("nan"),
                "point_biserial": _correlation(picked.astype(np.float64), rest),
                "mean_score": float(total_answered[picked].mean()) if count else float("nan"),
            }
            for g in range(int(n_groups)):
                in_group = group_answered == g
                size = int(in_group.sum())
                row[f"proportion_group_{g + 1}"] = (
                    float(picked[in_group].sum() / size) if size else float("nan")
                )
            rows.append(row)
    out = pd.DataFrame(rows)
    out["option"] = out["option"].astype(object)
    return out
