"""Observed-score equating of score distributions: identity, mean, linear, equipercentile."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from flexpyequate.distributions import (
    BivariateScoreDistribution,
    FloatArray,
    ScoreDistribution,
    _freeze,
    percentile_from_proportion,
    percentile_rank_proportion,
)
from flexpyequate.errors import SpecificationError

__all__ = ["EquatingMethod", "EquatingResult", "equate"]

EquatingMethod = Literal["identity", "mean", "linear", "equipercentile"]
Design = Literal["equivalent_groups", "single_group"]


@dataclass(frozen=True, eq=False)
class EquatingResult:
    r"""An equating function from form X to the scale of form Y.

    Attributes
    ----------
    method : str
        The equating method (``"identity"``, ``"mean"``, ``"linear"`` or ``"equipercentile"``).
    design : str
        ``"equivalent_groups"`` or ``"single_group"``.
    x, y : ScoreDistribution
        The distributions of form X and form Y that were equated (the margins for a single
        group).
    scores : numpy.ndarray
        The possible scores of form X.
    equated : numpy.ndarray
        Form-Y equivalents :math:`e_Y(x_j)` of the possible scores of form X.
    slope, intercept : float or None
        Coefficients of :math:`e_Y(x) = \mathrm{intercept} + \mathrm{slope}\,x` for the
        identity, mean and linear methods; ``None`` for equipercentile equating.
    ddof : int
        Delta degrees of freedom used for the standard deviations of linear equating.
    """

    method: EquatingMethod
    design: Design
    x: ScoreDistribution
    y: ScoreDistribution
    scores: FloatArray
    equated: FloatArray
    slope: float | None
    intercept: float | None
    ddof: int

    def transform(self, values: ArrayLike) -> FloatArray:
        """Equate arbitrary form-X scores (also between possible scores) to the form-Y scale.

        Parameters
        ----------
        values : array_like
            Form-X scores.

        Returns
        -------
        numpy.ndarray
            Form-Y equivalents, same shape as ``values``. For equipercentile equating, scores
            outside the form-X range map to the ends of the form-Y range.
        """
        points = np.asarray(values, dtype=np.float64)
        if self.slope is not None and self.intercept is not None:
            return np.asarray(self.intercept + self.slope * points, dtype=np.float64)
        return _equipercentile(self.x, self.y, points)

    @property
    def table(self) -> pd.DataFrame:
        """Conversion table: columns ``score`` (form X) and ``equated`` (form-Y equivalent)."""
        return pd.DataFrame({"score": self.scores, "equated": self.equated})

    @property
    def moments(self) -> pd.DataFrame:
        """Moments of form X, form Y and the equated form-X scores.

        Returns
        -------
        pandas.DataFrame
            Rows ``x``, ``y`` and ``equated``; columns ``mean``, ``sd``, ``skewness`` and
            ``kurtosis`` (population forms). The ``equated`` row describes :math:`e_Y(X)`, which
            takes the value :math:`e_Y(x_j)` with the relative frequency of :math:`x_j`.
        """
        weights = self.x.probabilities
        mean = float(weights @ self.equated)
        centered = self.equated - mean
        variance = float(weights @ centered**2)
        rows = {
            "x": [self.x.mean, self.x.sd(), self.x.skewness, self.x.kurtosis],
            "y": [self.y.mean, self.y.sd(), self.y.skewness, self.y.kurtosis],
            "equated": [
                mean,
                float(np.sqrt(variance)),
                float(weights @ centered**3 / variance**1.5) if variance > 0 else float("nan"),
                float(weights @ centered**4 / variance**2) if variance > 0 else float("nan"),
            ],
        }
        return pd.DataFrame.from_dict(
            rows, orient="index", columns=["mean", "sd", "skewness", "kurtosis"]
        )

    def __repr__(self) -> str:
        """Return a one-line summary."""
        line = f"EquatingResult(method={self.method!r}, design={self.design!r}"
        if self.slope is not None and self.intercept is not None:
            line += f", slope={self.slope:.6f}, intercept={self.intercept:.6f}"
        return line + f", n_scores={self.scores.size})"


def _equipercentile(x: ScoreDistribution, y: ScoreDistribution, values: FloatArray) -> FloatArray:
    r"""Equipercentile equivalents :math:`e_Y(x) = Q^{-1}(P(x))`.

    Kolen & Brennan (2014), eq. 2.18.
    """
    rank = percentile_rank_proportion(x.scores, x.cumulative, values)
    return percentile_from_proportion(y.scores, y.cumulative, rank)


def equate(
    x: ScoreDistribution | BivariateScoreDistribution,
    y: ScoreDistribution | None = None,
    *,
    method: EquatingMethod = "equipercentile",
    ddof: int = 0,
) -> EquatingResult:
    r"""Equate form X to the scale of form Y from their score distributions.

    The data collection design follows from the input: two separate distributions are equivalent
    groups; one bivariate distribution is a single group, whose two margins are equated.

    Methods (Kolen & Brennan, 2014, ch. 2), with :math:`\mu` and :math:`\sigma` the means and
    standard deviations, :math:`P` the percentile-rank function of X and :math:`Q^{-1}` the
    percentile function of Y:

    .. math::

        \text{identity:}\quad & e_Y(x) = x \\
        \text{mean:}\quad & e_Y(x) = x - \mu_X + \mu_Y \\
        \text{linear:}\quad & e_Y(x) = \frac{\sigma_Y}{\sigma_X}(x - \mu_X) + \mu_Y \\
        \text{equipercentile:}\quad & e_Y(x) = Q^{-1}\bigl(P(x)\bigr)

    The percentile rank treats each possible score as uniform on an interval one score increment
    wide (see :meth:`ScoreDistribution.percentile_rank` and
    :meth:`ScoreDistribution.percentile`).

    Parameters
    ----------
    x : ScoreDistribution or BivariateScoreDistribution
        Distribution of form X (equivalent groups), or the joint distribution of forms X and Y
        taken by the same examinees (single group).
    y : ScoreDistribution, optional
        Distribution of form Y from an independent, equivalent group. Omit it for a single group.
    method : {"identity", "mean", "linear", "equipercentile"}, default "equipercentile"
        The equating method.
    ddof : int, default 0
        Delta degrees of freedom of the standard deviations in linear equating: 0 (population
        form, as in Kolen & Brennan) or 1 (sample form, as R ``equate``). It matters only when the
        two sample sizes differ.

    Returns
    -------
    EquatingResult
        The conversion table and, for the linear family, slope and intercept.

    Raises
    ------
    SpecificationError
        If the inputs do not match a supported design, the method is unknown, or linear equating
        is asked for a form-X distribution without variance.

    References
    ----------
    Kolen, M. J., & Brennan, R. L. (2014). *Test equating, scaling, and linking* (3rd ed.).
    Springer.

    Examples
    --------
    >>> import flexpyequate as fe
    >>> x = fe.ScoreDistribution.from_frequencies(scores=range(5), frequencies=[4, 6, 6, 3, 1])
    >>> y = fe.ScoreDistribution.from_frequencies(scores=range(5), frequencies=[2, 4, 6, 4, 4])
    >>> round(fe.equate(x, y, method="mean").intercept, 2)
    0.65
    """
    if method not in get_args(EquatingMethod):
        raise SpecificationError(
            f"Unknown method {method!r}; choose one of {', '.join(get_args(EquatingMethod))}."
        )
    if ddof not in (0, 1):
        raise SpecificationError(f"ddof must be 0 or 1, got {ddof}.")
    design: Design
    if isinstance(x, BivariateScoreDistribution):
        if y is not None:
            raise SpecificationError(
                "A bivariate distribution is a single-group design and is equated on its own; "
                "do not pass y. Designs with an anchor test are not supported yet."
            )
        design = "single_group"
        x_margin, y_margin = x.x, x.y
    elif isinstance(x, ScoreDistribution) and isinstance(y, ScoreDistribution):
        design = "equivalent_groups"
        x_margin, y_margin = x, y
    else:
        raise SpecificationError(
            "Pass two ScoreDistribution objects (equivalent groups) or one "
            "BivariateScoreDistribution (single group)."
        )

    slope: float | None = None
    intercept: float | None = None
    if method == "equipercentile":
        equated = _equipercentile(x_margin, y_margin, x_margin.scores)
    else:
        if method == "identity":
            slope, intercept = 1.0, 0.0
        elif method == "mean":
            slope, intercept = 1.0, y_margin.mean - x_margin.mean
        else:
            x_sd = x_margin.sd(ddof=ddof)
            if not x_sd > 0:
                raise SpecificationError(
                    "Linear equating needs a form-X distribution with positive variance."
                )
            slope = y_margin.sd(ddof=ddof) / x_sd
            intercept = y_margin.mean - slope * x_margin.mean
        equated = intercept + slope * x_margin.scores
    return EquatingResult(
        method=method,
        design=design,
        x=x_margin,
        y=y_margin,
        scores=x_margin.scores,
        equated=_freeze(equated),
        slope=slope,
        intercept=intercept,
        ddof=ddof,
    )
