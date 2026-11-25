"""Score distributions: the input of every equating method.

A score distribution is a set of equally spaced possible scores with their frequencies. The
percentile-rank and percentile functions follow Kolen & Brennan (2014, ch. 2).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from flexpyequate.errors import ScoreDistributionError

__all__ = ["BivariateScoreDistribution", "ScoreDistribution"]

FloatArray = NDArray[np.float64]

# Relative tolerance for "equally spaced" and for matching observed scores to possible scores.
_GRID_RTOL = 1e-9


def _as_float_vector(values: Any, name: str) -> FloatArray:
    try:
        array = np.array(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ScoreDistributionError(f"{name} must be numeric.") from error
    if array.ndim != 1:
        raise ScoreDistributionError(f"{name} must be one-dimensional, got shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ScoreDistributionError(f"{name} must be finite (no NaN or infinity).")
    return array


def _validate_scores(values: Any, name: str) -> FloatArray:
    scores = _as_float_vector(values, name)
    if scores.size < 2:
        raise ScoreDistributionError(f"{name} needs at least two possible scores.")
    steps = np.diff(scores)
    if np.any(steps <= 0):
        raise ScoreDistributionError(f"{name} must be strictly increasing.")
    if not np.allclose(steps, steps[0], rtol=_GRID_RTOL, atol=0.0):
        raise ScoreDistributionError(f"{name} must be equally spaced.")
    return scores


def _score_grid(
    observed: FloatArray,
    scores: ArrayLike | None,
    score_range: tuple[float, float] | None,
    increment: float,
    name: str,
) -> FloatArray:
    """Possible scores from explicit ``scores``, a ``score_range`` or the observed range."""
    if scores is not None:
        if score_range is not None:
            raise ScoreDistributionError(f"Give either {name} or its range, not both.")
        return _validate_scores(scores, name)
    if not increment > 0:
        raise ScoreDistributionError(f"increment must be positive, got {increment}.")
    if score_range is None:
        if observed.size == 0:
            raise ScoreDistributionError("No observed scores and no possible scores were given.")
        low, high = float(np.min(observed)), float(np.max(observed))
    else:
        low, high = float(score_range[0]), float(score_range[1])
    n_steps = (high - low) / increment
    if n_steps < 1 - _GRID_RTOL or abs(n_steps - round(n_steps)) > _GRID_RTOL * max(1.0, n_steps):
        raise ScoreDistributionError(
            f"The score range ({low}, {high}) is not a positive multiple of increment {increment}."
        )
    return low + increment * np.arange(round(n_steps) + 1, dtype=np.float64)


def _score_index(observed: FloatArray, scores: FloatArray, name: str) -> NDArray[np.intp]:
    """Index of each observed score among the possible scores."""
    increment = scores[1] - scores[0]
    position = (observed - scores[0]) / increment
    index: NDArray[np.intp] = np.rint(position).astype(np.intp)
    off_grid = (index < 0) | (index >= scores.size) | (np.abs(position - index) > 1e-6)
    if np.any(off_grid):
        bad = observed[off_grid][0]
        raise ScoreDistributionError(
            f"Observed {name} score {bad} is not one of the possible scores "
            f"({scores[0]}, {scores[1]}, ..., {scores[-1]})."
        )
    return index


def _freeze(array: FloatArray) -> FloatArray:
    array = np.array(array, dtype=np.float64)
    array.setflags(write=False)
    return array


def percentile_rank_proportion(
    scores: FloatArray, cumulative: FloatArray, values: FloatArray
) -> FloatArray:
    r"""Percentile-rank function as a proportion in :math:`[0, 1]`.

    A possible score :math:`x_j` is treated as uniform on
    :math:`[x_j - \delta/2,\, x_j + \delta/2)`. With :math:`x^*` the possible score nearest to
    :math:`x`,

    .. math::

        P(x) = F(x^* - \delta) + \frac{x - (x^* - \delta/2)}{\delta}
               \bigl[F(x^*) - F(x^* - \delta)\bigr],

    with :math:`P(x) = 0` for :math:`x < x_1 - \delta/2` and :math:`P(x) = 1` for
    :math:`x \ge x_J + \delta/2` (Kolen & Brennan, 2014, ch. 2).
    """
    increment = scores[1] - scores[0]
    low = scores[0] - increment / 2
    high = scores[-1] + increment / 2
    nearest = np.clip(np.floor((values - low) / increment), 0, scores.size - 1).astype(np.intp)
    below = np.where(nearest > 0, cumulative[np.maximum(nearest - 1, 0)], 0.0)
    fraction = (values - (scores[nearest] - increment / 2)) / increment
    rank = below + fraction * (cumulative[nearest] - below)
    rank = np.where(values < low, 0.0, rank)
    rank = np.where(values >= high, 1.0, rank)
    return np.asarray(np.clip(rank, 0.0, 1.0), dtype=np.float64)


def percentile_from_proportion(
    scores: FloatArray, cumulative: FloatArray, proportions: FloatArray
) -> FloatArray:
    r"""Inverse of the percentile-rank function for proportions in :math:`[0, 1]`.

    .. math::

        P^{-1}_U(p) &= x^*_U - \frac{\delta}{2}
            + \delta\,\frac{p - F(x^*_U - \delta)}{F(x^*_U) - F(x^*_U - \delta)},
            \qquad x^*_U = \min\{x_j : F(x_j) > p\}, \\
        P^{-1}_L(p) &= x^*_L + \frac{\delta}{2}
            + \delta\,\frac{p - F(x^*_L)}{F(x^*_L + \delta) - F(x^*_L)},
            \qquad x^*_L = \max\{x_j : F(x_j) < p\},

    with :math:`P^{-1}_U(1) = x_J + \delta/2`, :math:`P^{-1}_L(0) = x_1 - \delta/2` and
    :math:`P^{-1}(p) = [P^{-1}_U(p) + P^{-1}_L(p)]/2` (Kolen & Brennan, 2014, ch. 2). The
    two versions differ only where zero frequencies make :math:`F` flat.
    """
    increment = scores[1] - scores[0]
    n_scores = scores.size
    low = scores[0] - increment / 2
    high = scores[-1] + increment / 2
    # F extended by F(x_1 - delta) = 0 on the left, so index i + 1 holds F(x_i).
    extended = np.concatenate(([0.0], cumulative))

    # Upper version: first possible score whose cumulative proportion exceeds p.
    first_above = np.searchsorted(cumulative, proportions, side="right")
    at_top = first_above >= n_scores
    i_upper = np.minimum(first_above, n_scores - 1)
    step_upper = extended[i_upper + 1] - extended[i_upper]
    with np.errstate(divide="ignore", invalid="ignore"):
        upper = (
            scores[i_upper]
            - increment / 2
            + increment * (proportions - extended[i_upper]) / step_upper
        )
    upper = np.where(at_top, high, upper)

    # Lower version: last possible score whose cumulative proportion is below p (index -1 stands
    # for the score x_1 - delta, where F = 0).
    last_below = np.searchsorted(cumulative, proportions, side="left") - 1
    at_bottom = proportions <= 0
    i_lower = np.minimum(last_below, n_scores - 2)
    step_lower = extended[i_lower + 2] - extended[i_lower + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        lower = (
            scores[0]
            + increment * i_lower
            + increment / 2
            + increment * (proportions - extended[i_lower + 1]) / step_lower
        )
    lower = np.where(at_bottom, low, lower)
    return np.asarray((upper + lower) / 2, dtype=np.float64)


@dataclass(frozen=True, eq=False)
class ScoreDistribution:
    """Frequencies of the possible scores of one test form.

    Build it with :meth:`from_scores` (one score per examinee) or :meth:`from_frequencies` (a
    frequency table).

    Parameters
    ----------
    scores : array_like of shape (n_scores,)
        Possible scores: strictly increasing and equally spaced. Scores nobody obtained are
        included with frequency zero.
    frequencies : array_like of shape (n_scores,)
        Non-negative frequency of each possible score. Non-integer values are allowed (presmoothed
        distributions).

    Raises
    ------
    ScoreDistributionError
        If the scores are not equally spaced, a frequency is negative or not finite, or all
        frequencies are zero.
    """

    scores: FloatArray
    frequencies: FloatArray

    def __post_init__(self) -> None:
        """Validate, copy and freeze the arrays."""
        scores = _validate_scores(self.scores, "scores")
        frequencies = _as_float_vector(self.frequencies, "frequencies")
        if frequencies.shape != scores.shape:
            raise ScoreDistributionError(
                f"frequencies has {frequencies.size} entries for {scores.size} possible scores."
            )
        if np.any(frequencies < 0):
            raise ScoreDistributionError("frequencies must be non-negative.")
        if not frequencies.sum() > 0:
            raise ScoreDistributionError("frequencies must not all be zero.")
        object.__setattr__(self, "scores", _freeze(scores))
        object.__setattr__(self, "frequencies", _freeze(frequencies))

    @classmethod
    def from_frequencies(
        cls, *, scores: Iterable[float], frequencies: ArrayLike
    ) -> ScoreDistribution:
        """Build a distribution from a frequency table.

        Parameters
        ----------
        scores : array_like of shape (n_scores,)
            Possible scores, equally spaced and increasing.
        frequencies : array_like of shape (n_scores,)
            Frequency of each possible score.
        """
        possible = scores if isinstance(scores, np.ndarray) else np.array(list(scores))
        return cls(scores=possible, frequencies=np.asarray(frequencies, dtype=np.float64))

    @classmethod
    def from_scores(
        cls,
        values: ArrayLike,
        *,
        scores: ArrayLike | None = None,
        score_range: tuple[float, float] | None = None,
        increment: float = 1.0,
    ) -> ScoreDistribution:
        """Tabulate one observed score per examinee.

        Parameters
        ----------
        values : array_like of shape (n_examinees,)
            Observed scores. Missing values are not allowed; drop them first.
        scores : array_like, optional
            The possible scores. Give this or ``score_range``.
        score_range : tuple of float, optional
            Lowest and highest possible score; the possible scores are ``increment`` apart. If
            neither ``scores`` nor ``score_range`` is given, the observed range is used.
        increment : float, default 1.0
            Distance between possible scores when ``scores`` is not given.

        Raises
        ------
        ScoreDistributionError
            If an observed score is not one of the possible scores.
        """
        observed = _as_float_vector(values, "values")
        grid = _score_grid(observed, scores, score_range, increment, "scores")
        index = _score_index(observed, grid, "x")
        counts = np.bincount(index, minlength=grid.size).astype(np.float64)
        return cls(scores=grid, frequencies=counts)

    # ------------------------------------------------------------------ basic quantities
    @property
    def n_scores(self) -> int:
        """Number of possible scores."""
        return int(self.scores.size)

    @property
    def increment(self) -> float:
        """Distance between adjacent possible scores."""
        return float(self.scores[1] - self.scores[0])

    @property
    def n(self) -> float:
        """Total frequency (sample size)."""
        return float(np.cumsum(self.frequencies)[-1])

    @property
    def probabilities(self) -> FloatArray:
        """Relative frequency of each possible score."""
        return np.asarray(self.frequencies / self.n, dtype=np.float64)

    @property
    def cumulative(self) -> FloatArray:
        r"""Cumulative proportion :math:`F(x_j) = \sum_{i \le j} r_i` at each possible score."""
        running = np.cumsum(self.frequencies)
        return np.asarray(running / running[-1], dtype=np.float64)

    @property
    def mean(self) -> float:
        r"""Mean :math:`\mu = \sum_j r_j x_j`."""
        return float(self.probabilities @ self.scores)

    def variance(self, *, ddof: int = 0) -> float:
        r"""Variance :math:`\sigma^2 = \frac{N}{N - \mathrm{ddof}} \sum_j r_j (x_j - \mu)^2`.

        Parameters
        ----------
        ddof : int, default 0
            Delta degrees of freedom. 0 is the population variance used by Kolen & Brennan (2014);
            1 is the unbiased sample variance (R ``equate``).
        """
        if ddof not in (0, 1):
            raise ScoreDistributionError(f"ddof must be 0 or 1, got {ddof}.")
        population = float(self.probabilities @ (self.scores - self.mean) ** 2)
        return population * self.n / (self.n - ddof)

    def sd(self, *, ddof: int = 0) -> float:
        """Return the standard deviation, the square root of :meth:`variance`."""
        return float(np.sqrt(self.variance(ddof=ddof)))

    @property
    def skewness(self) -> float:
        r"""Skewness :math:`\sum_j r_j (x_j - \mu)^3 / \sigma^3` (population form)."""
        centered = self.scores - self.mean
        return float(self.probabilities @ centered**3 / self.variance() ** 1.5)

    @property
    def kurtosis(self) -> float:
        r"""Kurtosis :math:`\sum_j r_j (x_j - \mu)^4 / \sigma^4` (3 for a normal distribution)."""
        centered = self.scores - self.mean
        return float(self.probabilities @ centered**4 / self.variance() ** 2)

    # ------------------------------------------------------------------ percentiles
    def percentile_rank(self, values: ArrayLike) -> FloatArray:
        r"""Percentile rank (0-100) of scores, also between possible scores.

        A possible score :math:`x_j` is treated as uniform on
        :math:`[x_j - \delta/2,\, x_j + \delta/2)`, so that with :math:`x^*` the possible score
        nearest to :math:`x`

        .. math::

            P(x) = 100\Bigl\{F(x^* - \delta) + \frac{x - (x^* - \delta/2)}{\delta}
                   \bigl[F(x^*) - F(x^* - \delta)\bigr]\Bigr\},

        0 below :math:`x_1 - \delta/2` and 100 from :math:`x_J + \delta/2` on (Kolen & Brennan,
        2014, ch. 2). At a possible score this is the percentage below it plus half the
        percentage at it.

        Parameters
        ----------
        values : array_like
            Scores at which to evaluate the percentile rank.

        Returns
        -------
        numpy.ndarray
            Percentile ranks, same shape as ``values``.
        """
        points = np.asarray(values, dtype=np.float64)
        return 100.0 * percentile_rank_proportion(self.scores, self.cumulative, points)

    def percentile(self, ranks: ArrayLike) -> FloatArray:
        r"""Score at given percentile ranks (0-100): the inverse of :meth:`percentile_rank`.

        The mean of the upper and lower versions of Kolen & Brennan (2014, ch. 2),

        .. math::

            P^{-1}_U(p) &= x^*_U - \frac{\delta}{2}
                + \delta\,\frac{p - F(x^*_U - \delta)}{F(x^*_U) - F(x^*_U - \delta)},
                \qquad x^*_U = \min\{x_j : F(x_j) > p\}, \\
            P^{-1}_L(p) &= x^*_L + \frac{\delta}{2}
                + \delta\,\frac{p - F(x^*_L)}{F(x^*_L + \delta) - F(x^*_L)},
                \qquad x^*_L = \max\{x_j : F(x_j) < p\},

        with :math:`p` the rank divided by 100, :math:`P^{-1}_U(1) = x_J + \delta/2` and
        :math:`P^{-1}_L(0) = x_1 - \delta/2`. The two versions differ only where zero frequencies
        make the cumulative distribution flat.

        Parameters
        ----------
        ranks : array_like
            Percentile ranks between 0 and 100.

        Returns
        -------
        numpy.ndarray
            Scores between :math:`x_1 - \delta/2` and :math:`x_J + \delta/2`.
        """
        proportions = np.asarray(ranks, dtype=np.float64) / 100.0
        if np.any(~np.isfinite(proportions)) or np.any((proportions < 0) | (proportions > 1)):
            raise ScoreDistributionError("Percentile ranks must lie between 0 and 100.")
        return percentile_from_proportion(self.scores, self.cumulative, proportions)

    def to_frame(self) -> pd.DataFrame:
        """Return the distribution as a table, one row per possible score.

        Returns
        -------
        pandas.DataFrame
            Columns ``score``, ``frequency``, ``probability``, ``cumulative`` and
            ``percentile_rank``.
        """
        return pd.DataFrame(
            {
                "score": self.scores,
                "frequency": self.frequencies,
                "probability": self.probabilities,
                "cumulative": self.cumulative,
                "percentile_rank": self.percentile_rank(self.scores),
            }
        )

    def __repr__(self) -> str:
        """Return a one-line summary."""
        return (
            f"ScoreDistribution(scores={self.scores[0]:g}..{self.scores[-1]:g} "
            f"by {self.increment:g}, n={self.n:g}, mean={self.mean:.4f}, sd={self.sd():.4f})"
        )


@dataclass(frozen=True, eq=False)
class BivariateScoreDistribution:
    """Joint frequencies of two scores obtained by the same examinees.

    Used for the single-group design, where every examinee takes form X and form Y.

    Parameters
    ----------
    x_scores : array_like of shape (n_x_scores,)
        Possible scores of the first variable (form X), equally spaced and increasing.
    y_scores : array_like of shape (n_y_scores,)
        Possible scores of the second variable (form Y).
    frequencies : array_like of shape (n_x_scores, n_y_scores)
        Non-negative joint frequencies; rows are X scores and columns are Y scores.

    Raises
    ------
    ScoreDistributionError
        If the scores or frequencies are invalid, or a margin has no observation.
    """

    x_scores: FloatArray
    y_scores: FloatArray
    frequencies: FloatArray

    def __post_init__(self) -> None:
        """Validate, copy and freeze the arrays."""
        x_scores = _validate_scores(self.x_scores, "x_scores")
        y_scores = _validate_scores(self.y_scores, "y_scores")
        try:
            frequencies = np.array(self.frequencies, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ScoreDistributionError("frequencies must be numeric.") from error
        if frequencies.shape != (x_scores.size, y_scores.size):
            raise ScoreDistributionError(
                f"frequencies has shape {frequencies.shape}; expected "
                f"({x_scores.size}, {y_scores.size})."
            )
        if not np.all(np.isfinite(frequencies)) or np.any(frequencies < 0):
            raise ScoreDistributionError("frequencies must be finite and non-negative.")
        if not frequencies.sum() > 0:
            raise ScoreDistributionError("frequencies must not all be zero.")
        object.__setattr__(self, "x_scores", _freeze(x_scores))
        object.__setattr__(self, "y_scores", _freeze(y_scores))
        object.__setattr__(self, "frequencies", _freeze(frequencies))

    @classmethod
    def from_frequencies(
        cls, *, x_scores: ArrayLike, y_scores: ArrayLike, frequencies: ArrayLike
    ) -> BivariateScoreDistribution:
        """Build a joint distribution from a two-way frequency table.

        Parameters
        ----------
        x_scores : array_like of shape (n_x_scores,)
            Possible scores of form X (rows of the table).
        y_scores : array_like of shape (n_y_scores,)
            Possible scores of form Y (columns of the table).
        frequencies : array_like of shape (n_x_scores, n_y_scores)
            Joint frequencies.
        """
        return cls(
            x_scores=np.asarray(x_scores, dtype=np.float64),
            y_scores=np.asarray(y_scores, dtype=np.float64),
            frequencies=np.asarray(frequencies, dtype=np.float64),
        )

    @classmethod
    def from_scores(
        cls,
        x_values: ArrayLike,
        y_values: ArrayLike,
        *,
        x_scores: ArrayLike | None = None,
        y_scores: ArrayLike | None = None,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
        x_increment: float = 1.0,
        y_increment: float = 1.0,
    ) -> BivariateScoreDistribution:
        """Cross-tabulate paired observed scores (one pair per examinee).

        Parameters
        ----------
        x_values, y_values : array_like of shape (n_examinees,)
            Observed scores on the two forms, in the same examinee order.
        x_scores, y_scores : array_like, optional
            The possible scores of each form.
        x_range, y_range : tuple of float, optional
            Lowest and highest possible score of each form, used when the possible scores are not
            given. Without either, the observed range is used.
        x_increment, y_increment : float, default 1.0
            Distance between possible scores when they are not given explicitly.
        """
        x_observed = _as_float_vector(x_values, "x_values")
        y_observed = _as_float_vector(y_values, "y_values")
        if x_observed.size != y_observed.size:
            raise ScoreDistributionError(
                f"x_values and y_values must be paired; got {x_observed.size} and "
                f"{y_observed.size} scores."
            )
        x_grid = _score_grid(x_observed, x_scores, x_range, x_increment, "x_scores")
        y_grid = _score_grid(y_observed, y_scores, y_range, y_increment, "y_scores")
        x_index = _score_index(x_observed, x_grid, "x")
        y_index = _score_index(y_observed, y_grid, "y")
        table = np.zeros((x_grid.size, y_grid.size), dtype=np.float64)
        np.add.at(table, (x_index, y_index), 1.0)
        return cls(x_scores=x_grid, y_scores=y_grid, frequencies=table)

    @property
    def n(self) -> float:
        """Total frequency (number of examinees)."""
        return float(self.frequencies.sum())

    @property
    def probabilities(self) -> FloatArray:
        """Joint relative frequencies, shape ``(n_x_scores, n_y_scores)``."""
        return np.asarray(self.frequencies / self.n, dtype=np.float64)

    @property
    def x(self) -> ScoreDistribution:
        """Marginal distribution of the first variable (form X)."""
        return ScoreDistribution(scores=self.x_scores, frequencies=self.frequencies.sum(axis=1))

    @property
    def y(self) -> ScoreDistribution:
        """Marginal distribution of the second variable (form Y)."""
        return ScoreDistribution(scores=self.y_scores, frequencies=self.frequencies.sum(axis=0))

    def covariance(self, *, ddof: int = 0) -> float:
        r"""Covariance of the two scores.

        .. math::

            \sigma_{XY} = \frac{N}{N - \mathrm{ddof}} \sum_{jk} p_{jk} (x_j - \mu_X)(y_k - \mu_Y)

        Parameters
        ----------
        ddof : int, default 0
            Delta degrees of freedom (0: population form, 1: unbiased sample form).
        """
        if ddof not in (0, 1):
            raise ScoreDistributionError(f"ddof must be 0 or 1, got {ddof}.")
        x_centered = self.x_scores - self.x.mean
        y_centered = self.y_scores - self.y.mean
        population = float(x_centered @ self.probabilities @ y_centered)
        return population * self.n / (self.n - ddof)

    @property
    def correlation(self) -> float:
        """Pearson correlation between the two scores."""
        return self.covariance() / (self.x.sd() * self.y.sd())

    def to_frame(self) -> pd.DataFrame:
        """Return the joint distribution in long form.

        Returns
        -------
        pandas.DataFrame
            One row per score pair, with columns ``x_score``, ``y_score``, ``frequency`` and
            ``probability``.
        """
        x_grid, y_grid = np.meshgrid(self.x_scores, self.y_scores, indexing="ij")
        return pd.DataFrame(
            {
                "x_score": x_grid.ravel(),
                "y_score": y_grid.ravel(),
                "frequency": self.frequencies.ravel(),
                "probability": self.probabilities.ravel(),
            }
        )

    def __repr__(self) -> str:
        """Return a one-line summary."""
        return (
            f"BivariateScoreDistribution(x={self.x_scores[0]:g}..{self.x_scores[-1]:g}, "
            f"y={self.y_scores[0]:g}..{self.y_scores[-1]:g}, n={self.n:g}, "
            f"correlation={self.correlation:.4f})"
        )
