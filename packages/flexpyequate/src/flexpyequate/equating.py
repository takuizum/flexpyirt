"""Observed-score equating of score distributions.

Identity, mean, linear and equipercentile equating for equivalent groups and a single group, and
the methods for non-equivalent groups with an anchor test: linear (Tucker, Levine, Braun-Holland,
chained) and equipercentile (frequency estimation, chained). Standard errors of equating by the
delta method (where a formula exists) and by the bootstrap.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Literal, get_args

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from flexpyequate._neat import (
    Anchor,
    GroupMoments,
    chained_linear_coefficients,
    levine_gamma,
    levine_true_coefficients,
    post_stratified_distributions,
    synthetic_moments,
    tucker_gamma,
)
from flexpyequate._standard_errors import (
    anchor_linear_gradients,
    braun_holland_se,
    lord_equipercentile_se,
    mean_linear_gradients,
    moment_covariance,
    quadratic_form_se,
    univariate_moment_covariance,
)
from flexpyequate.distributions import (
    BivariateScoreDistribution,
    FloatArray,
    ScoreDistribution,
    _freeze,
    percentile_from_proportion,
    percentile_rank_proportion,
)
from flexpyequate.errors import ScoreDistributionError, SparseDataWarning, SpecificationError

__all__ = ["EquatingMethod", "EquatingResult", "StandardError", "equate"]

EquatingMethod = Literal[
    "identity",
    "mean",
    "linear",
    "equipercentile",
    "tucker",
    "levine_observed",
    "levine_true",
    "braun_holland",
    "chained_linear",
    "frequency_estimation",
    "chained_equipercentile",
]
Design = Literal["equivalent_groups", "single_group", "nonequivalent_groups"]
StandardError = Literal["analytic", "analytic_normal", "bootstrap"]

_ANCHOR_METHODS = (
    "tucker",
    "levine_observed",
    "levine_true",
    "braun_holland",
    "chained_linear",
    "frequency_estimation",
    "chained_equipercentile",
)
_SYNTHETIC_METHODS = ("tucker", "levine_observed", "braun_holland", "frequency_estimation")


@dataclass(frozen=True, eq=False)
class EquatingResult:
    r"""An equating function from form X to the scale of form Y.

    Attributes
    ----------
    method : str
        The equating method, as passed to :func:`equate`.
    design : str
        ``"equivalent_groups"``, ``"single_group"`` or ``"nonequivalent_groups"``.
    x, y : ScoreDistribution
        The distributions of form X and form Y that were equated: the margins for a single
        group; for non-equivalent groups, form X in the population that took it and form Y in the
        population that took it. Frequency estimation is the exception: there they are the
        distributions of X and Y in the synthetic population, which is where the equating is
        defined (their total frequency is :math:`N_1 + N_2`).
    scores : numpy.ndarray
        The possible scores of form X.
    equated : numpy.ndarray
        Form-Y equivalents :math:`e_Y(x_j)` of the possible scores of form X.
    slope, intercept : float or None
        Coefficients of :math:`e_Y(x) = \mathrm{intercept} + \mathrm{slope}\,x` for the
        linear methods; ``None`` for the equipercentile methods.
    ddof : int
        Delta degrees of freedom used for the variances and covariances of the linear methods.
    synthetic_weight : float or None
        Weight :math:`w_1` of the population that took form X in the synthetic population
        (Tucker, Levine observed-score, Braun-Holland and frequency-estimation methods); ``None``
        otherwise.
    anchor : str or None
        ``"internal"`` or ``"external"`` for the Levine methods; ``None`` otherwise.
    gamma : tuple of float or None
        :math:`(\gamma_1, \gamma_2)` of the Tucker and Levine methods; ``None`` otherwise.
    synthetic_moments : pandas.DataFrame or None
        Mean and standard deviation of X and Y in the synthetic population (rows ``x`` and ``y``,
        columns ``mean`` and ``sd``) for the Tucker, Levine observed-score and Braun-Holland
        methods; ``None`` otherwise.
    anchor_distributions : tuple of ScoreDistribution or None
        Chained equipercentile equating: the anchor distributions in the population that took X
        and in the population that took Y; ``None`` otherwise.
    see : numpy.ndarray or None
        Standard error of equating at each possible score of form X, if ``standard_error`` was
        asked for; ``None`` otherwise.
    standard_error : str or None
        How ``see`` was obtained: ``"analytic"``, ``"analytic_normal"`` or ``"bootstrap"``.
    n_replications : int or None
        Bootstrap: the number of bootstrap samples that ``see`` is based on (samples that could
        not be equated are left out).
    bootstrap_mean : numpy.ndarray or None
        Bootstrap: mean of the equated scores over the bootstrap samples. Its difference from
        ``equated`` estimates the bias of the equating function.
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
    synthetic_weight: float | None = None
    anchor: Anchor | None = None
    gamma: tuple[float, float] | None = None
    synthetic_moments: pd.DataFrame | None = None
    anchor_distributions: tuple[ScoreDistribution, ScoreDistribution] | None = None
    see: FloatArray | None = None
    standard_error: StandardError | None = None
    n_replications: int | None = None
    bootstrap_mean: FloatArray | None = None

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
        if self.anchor_distributions is not None:
            return _chained_equipercentile(self.x, self.y, *self.anchor_distributions, points)
        return _equipercentile(self.x, self.y, points)

    @property
    def table(self) -> pd.DataFrame:
        """Conversion table: columns ``score`` (form X) and ``equated`` (form-Y equivalent).

        A column ``see`` (standard error of equating) follows when standard errors were computed.
        """
        table = pd.DataFrame({"score": self.scores, "equated": self.equated})
        if self.see is not None:
            table["see"] = self.see
        return table

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

    Kolen & Brennan (2014), ch. 2.
    """
    rank = percentile_rank_proportion(x.scores, x.cumulative, values)
    return percentile_from_proportion(y.scores, y.cumulative, rank)


def _chained_equipercentile(
    x: ScoreDistribution,
    y: ScoreDistribution,
    anchor_1: ScoreDistribution,
    anchor_2: ScoreDistribution,
    values: FloatArray,
) -> FloatArray:
    r"""Chained equipercentile equivalents.

    .. math:: e_Y(x) = Q_2^{-1}\Bigl(H_2\bigl(H_1^{-1}(P_1(x))\bigr)\Bigr)

    X is equated to the anchor in population 1 and the anchor to Y in population 2, with the
    percentile-rank functions :math:`P_1` (X) and :math:`H_1` (anchor) of population 1 and
    :math:`H_2` (anchor) and :math:`Q_2` (Y) of population 2 (Kolen & Brennan, 2014, ch. 5).
    """
    on_anchor = _equipercentile(x, anchor_1, values)
    return _equipercentile(anchor_2, y, on_anchor)


def _weighted_moments(scores: FloatArray, probabilities: FloatArray) -> tuple[float, float]:
    mean = float(probabilities @ scores)
    return mean, float(np.sqrt(probabilities @ (scores - mean) ** 2))


def _synthetic_frame(mean_x: float, sd_x: float, mean_y: float, sd_y: float) -> pd.DataFrame:
    return pd.DataFrame({"mean": [mean_x, mean_y], "sd": [sd_x, sd_y]}, index=["x", "y"])


def _post_stratify(
    x: BivariateScoreDistribution, y: BivariateScoreDistribution, weight: float
) -> tuple[FloatArray, FloatArray]:
    """Synthetic score probabilities of X and Y, warning about anchor scores without data."""
    synthetic_x, synthetic_y, dropped = post_stratified_distributions(
        x.probabilities, y.probabilities, weight
    )
    if dropped > 0:
        warnings.warn(
            "Some anchor scores were observed in one population only; that part of the other "
            f"population (probability up to {dropped:.4g}) is left out of the synthetic "
            "distributions. Presmoothing the joint distributions avoids this.",
            SparseDataWarning,
            stacklevel=5,
        )
    return synthetic_x, synthetic_y


def _equate_with_anchor(
    x: BivariateScoreDistribution,
    y: BivariateScoreDistribution,
    method: EquatingMethod,
    ddof: int,
    synthetic_weight: float | None,
    anchor: Anchor | None,
) -> EquatingResult:
    """Equating for non-equivalent groups with an anchor test."""
    if x.y_scores.shape != y.y_scores.shape or not np.allclose(x.y_scores, y.y_scores):
        raise SpecificationError(
            "The anchor must be the same test in both populations: x and y need the same "
            "possible anchor scores (their second variable)."
        )
    for name, joint in (("x", x), ("y", y)):
        if not (joint.x.variance() > 0 and joint.y.variance() > 0):
            raise SpecificationError(
                f"The total and the anchor score of {name} must both have positive variance."
            )
    weight: float | None = None
    if method in _SYNTHETIC_METHODS:
        weight = x.n / (x.n + y.n) if synthetic_weight is None else float(synthetic_weight)
        if not 0.0 <= weight <= 1.0:
            raise SpecificationError(f"synthetic_weight must lie in [0, 1], got {weight}.")
    uses_levine = method in ("levine_observed", "levine_true")
    if uses_levine and anchor not in ("internal", "external"):
        raise SpecificationError(
            f'method={method!r} needs anchor="internal" (the anchor items count towards the '
            'total score) or anchor="external".'
        )

    if method == "chained_equipercentile":
        anchors = (x.y, y.y)
        equated = _chained_equipercentile(x.x, y.x, *anchors, x.x_scores)
        return EquatingResult(
            method=method,
            design="nonequivalent_groups",
            x=x.x,
            y=y.x,
            scores=x.x_scores,
            equated=_freeze(equated),
            slope=None,
            intercept=None,
            ddof=ddof,
            anchor_distributions=anchors,
        )
    if method == "frequency_estimation":
        assert weight is not None
        synthetic_x, synthetic_y = _post_stratify(x, y, weight)
        total = x.n + y.n
        x_synthetic = ScoreDistribution(scores=x.x_scores, frequencies=synthetic_x * total)
        y_synthetic = ScoreDistribution(scores=y.x_scores, frequencies=synthetic_y * total)
        return EquatingResult(
            method=method,
            design="nonequivalent_groups",
            x=x_synthetic,
            y=y_synthetic,
            scores=x.x_scores,
            equated=_freeze(_equipercentile(x_synthetic, y_synthetic, x.x_scores)),
            slope=None,
            intercept=None,
            ddof=ddof,
            synthetic_weight=weight,
            synthetic_moments=_synthetic_frame(
                x_synthetic.mean, x_synthetic.sd(), y_synthetic.mean, y_synthetic.sd()
            ),
        )

    group_1 = GroupMoments.of(x, ddof)
    group_2 = GroupMoments.of(y, ddof)
    gamma: tuple[float, float] | None = None
    synthetic: pd.DataFrame | None = None

    if method == "chained_linear":
        slope, intercept = chained_linear_coefficients(group_1, group_2)
    elif method == "braun_holland":
        assert weight is not None
        synthetic_x, synthetic_y = _post_stratify(x, y, weight)
        mean_x, sd_x = _weighted_moments(x.x_scores, synthetic_x)
        mean_y, sd_y = _weighted_moments(y.x_scores, synthetic_y)
        slope = sd_y / sd_x
        intercept = mean_y - slope * mean_x
        synthetic = _synthetic_frame(mean_x, sd_x, mean_y, sd_y)
    else:
        if method == "tucker":
            gamma = (tucker_gamma(group_1), tucker_gamma(group_2))
        else:
            assert anchor is not None
            if not (group_1.covariance > 0 and group_2.covariance > 0):
                raise SpecificationError(
                    "Levine's methods need a positive covariance between total and anchor score "
                    "in both populations."
                )
            gamma = (levine_gamma(group_1, anchor), levine_gamma(group_2, anchor))
        if method == "levine_true":
            slope, intercept = levine_true_coefficients(group_1, group_2, *gamma)
        else:
            assert weight is not None
            moments = synthetic_moments(group_1, group_2, gamma[0], gamma[1], weight)
            slope = moments.sd_y / moments.sd_x
            intercept = moments.mean_y - slope * moments.mean_x
            synthetic = _synthetic_frame(moments.mean_x, moments.sd_x, moments.mean_y, moments.sd_y)

    x_margin, y_margin = x.x, y.x
    return EquatingResult(
        method=method,
        design="nonequivalent_groups",
        x=x_margin,
        y=y_margin,
        scores=x_margin.scores,
        equated=_freeze(intercept + slope * x_margin.scores),
        slope=float(slope),
        intercept=float(intercept),
        ddof=ddof,
        synthetic_weight=weight,
        anchor=anchor if uses_levine else None,
        gamma=gamma,
        synthetic_moments=synthetic,
    )


def equate(
    x: ScoreDistribution | BivariateScoreDistribution,
    y: ScoreDistribution | BivariateScoreDistribution | None = None,
    *,
    method: EquatingMethod = "equipercentile",
    ddof: int = 0,
    synthetic_weight: float | None = None,
    anchor: Anchor | None = None,
    standard_error: StandardError | None = None,
    n_replications: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> EquatingResult:
    r"""Equate form X to the scale of form Y from their score distributions.

    The data collection design follows from the input:

    - two ``ScoreDistribution`` objects: equivalent groups;
    - one ``BivariateScoreDistribution``: a single group, whose two margins are equated;
    - two ``BivariateScoreDistribution`` objects: non-equivalent groups with an anchor test. The
      first holds form X and the anchor in the population that took X, the second form Y and the
      anchor in the population that took Y; in both, the anchor is the second variable.

    **Equivalent groups and single group** (Kolen & Brennan, 2014, ch. 2), with :math:`\mu` and
    :math:`\sigma` the means and standard deviations, :math:`P` the percentile-rank function of X
    and :math:`Q^{-1}` the percentile function of Y:

    .. math::

        \text{identity:}\quad & e_Y(x) = x \\
        \text{mean:}\quad & e_Y(x) = x - \mu_X + \mu_Y \\
        \text{linear:}\quad & e_Y(x) = \frac{\sigma_Y}{\sigma_X}(x - \mu_X) + \mu_Y \\
        \text{equipercentile:}\quad & e_Y(x) = Q^{-1}\bigl(P(x)\bigr)

    The percentile rank treats each possible score as uniform on an interval one score increment
    wide (see :meth:`ScoreDistribution.percentile_rank` and
    :meth:`ScoreDistribution.percentile`).

    **Non-equivalent groups with an anchor test** (Kolen & Brennan, 2014, ch. 4). Population 1
    takes X and the anchor V, population 2 takes Y and V. ``"tucker"``, ``"levine_observed"`` and
    ``"braun_holland"`` equate linearly in the synthetic population
    :math:`T = w_1 P_1 + w_2 P_2`,

    .. math::

        l_Y(x) = \frac{\sigma_s(Y)}{\sigma_s(X)}\,[x - \mu_s(X)] + \mu_s(Y),

    with

    .. math::

        \mu_s(X) &= \mu_1(X) - w_2 \gamma_1 [\mu_1(V) - \mu_2(V)] \\
        \mu_s(Y) &= \mu_2(Y) + w_1 \gamma_2 [\mu_1(V) - \mu_2(V)] \\
        \sigma_s^2(X) &= \sigma_1^2(X) - w_2 \gamma_1^2 [\sigma_1^2(V) - \sigma_2^2(V)]
                         + w_1 w_2 \gamma_1^2 [\mu_1(V) - \mu_2(V)]^2 \\
        \sigma_s^2(Y) &= \sigma_2^2(Y) + w_1 \gamma_2^2 [\sigma_1^2(V) - \sigma_2^2(V)]
                         + w_1 w_2 \gamma_2^2 [\mu_1(V) - \mu_2(V)]^2

    for the Tucker and Levine observed-score methods, where

    .. math::

        \text{Tucker:}\quad & \gamma_1 = \frac{\sigma_1(X, V)}{\sigma_1^2(V)} \\
        \text{Levine, internal anchor:}\quad & \gamma_1 = \frac{\sigma_1^2(X)}{\sigma_1(X, V)} \\
        \text{Levine, external anchor:}\quad &
            \gamma_1 = \frac{\sigma_1^2(X) + \sigma_1(X, V)}{\sigma_1^2(V) + \sigma_1(X, V)}

    and :math:`\gamma_2` likewise from Y and V in population 2. The Braun-Holland method takes
    the synthetic means and standard deviations from the frequency-estimation distributions
    :math:`f_s(x) = w_1 f_1(x) + w_2 \sum_v f_1(x \mid v) h_2(v)` and
    :math:`g_s(y) = w_1 \sum_v g_2(y \mid v) h_1(v) + w_2 g_2(y)`. Two methods do not use a
    synthetic population:

    .. math::

        \text{Levine true score:}\quad & l_Y(x) = \frac{\gamma_2}{\gamma_1}[x - \mu_1(X)]
            + \mu_2(Y) + \gamma_2 [\mu_1(V) - \mu_2(V)] \\
        \text{chained linear:}\quad & l_Y(x) = \mu_2(Y) + \frac{\sigma_2(Y)}{\sigma_2(V)}
            \Bigl[\mu_1(V) + \frac{\sigma_1(V)}{\sigma_1(X)}(x - \mu_1(X)) - \mu_2(V)\Bigr]

    The equipercentile methods for this design (Kolen & Brennan, 2014, ch. 5) are
    ``"frequency_estimation"``, :math:`e_Y(x) = Q_s^{-1}(P_s(x))` with the percentile-rank
    functions of the synthetic distributions :math:`f_s` and :math:`g_s` above, and
    ``"chained_equipercentile"``,
    :math:`e_Y(x) = Q_2^{-1}\bigl(H_2(H_1^{-1}(P_1(x)))\bigr)`: X is equated to the anchor in
    population 1 (:math:`P_1`, :math:`H_1`) and the anchor to Y in population 2 (:math:`H_2`,
    :math:`Q_2`). Pass presmoothed joint distributions (:func:`presmooth`) to smooth either.

    **Standard errors of equating** (``standard_error``; Kolen & Brennan, 2014, ch. 7).

    ``"analytic"`` is the delta method without a distributional assumption. The linear methods
    are functions :math:`e_Y(x) = c(\theta) + b(\theta)x` of the means, variances and covariances
    :math:`\theta` of the samples, and

    .. math::

        \mathrm{SE}[\hat e_Y(x)]^2 = g(x)^\top \Omega\, g(x), \qquad
        g(x) = \nabla c + x \nabla b, \qquad \Omega = \operatorname{Cov}(\hat\theta),

    where :math:`N\operatorname{Cov}(\bar A, \bar B) = \sigma_{AB}`,
    :math:`N\operatorname{Cov}(\bar A, s_{BC}) = \mu_{ABC}` and
    :math:`N\operatorname{Cov}(s_{AB}, s_{CD}) = \mu_{ABCD} - \sigma_{AB}\sigma_{CD}` with the
    central moments :math:`\mu` of the observed distributions (Kolen, 1985; Zu & Yuan, 2012).
    For linear equating of equivalent groups this is the formula of Braun & Holland (1982),

    .. math::

        \mathrm{SE}^2 = \sigma_Y^2 \Bigl\{ \frac{1}{N_X} + \frac{1}{N_Y}
        + \Bigl[\frac{\mathrm{sk}(X)}{N_X} + \frac{\mathrm{sk}(Y)}{N_Y}\Bigr] z
        + \Bigl[\frac{\mathrm{ku}(X) - 1}{4 N_X} + \frac{\mathrm{ku}(Y) - 1}{4 N_Y}\Bigr] z^2
        \Bigr\}, \qquad z = \frac{x - \mu_X}{\sigma_X} .

    The Braun-Holland method is differentiated with respect to the cell probabilities of the two
    joint distributions instead. For equipercentile equating of equivalent groups the formula of
    Lord (1982) is used: with :math:`p = P(x)`, :math:`y^*_U = \min\{y_k : G(y_k) > p\}`,
    :math:`G_0 = G(y^*_U - \delta_Y)` and :math:`g = G(y^*_U) - G_0`,

    .. math::

        \mathrm{SE}^2 = \frac{\delta_Y^2}{g^2} \Bigl\{ \frac{p(1 - p)}{N_X}
        + \frac{1}{N_Y}\Bigl[G_0 - p^2 + \frac{(p - G_0)^2}{g}\Bigr] \Bigr\} .

    ``"analytic_normal"`` replaces :math:`\Omega` by its value under (bivariate) normality: third
    moments zero and :math:`\mu_{ABCD} - \sigma_{AB}\sigma_{CD} = \sigma_{AC}\sigma_{BD} +
    \sigma_{AD}\sigma_{BC}`. It exists for the methods that depend on moments only.

    ``"bootstrap"`` resamples :math:`N` examinees with replacement from each sample
    (multinomial counts with the observed relative frequencies; the cells of a joint
    distribution, so that the dependence between its two scores is kept), equates each of the
    ``n_replications`` samples and takes the standard deviation (divisor :math:`B - 1`) of the
    equated scores. It works for every method.

    ===================================== ============ =================== =============
    Method (design)                       ``analytic`` ``analytic_normal`` ``bootstrap``
    ===================================== ============ =================== =============
    identity                              0            0                   0
    mean, linear (equivalent groups)      yes          yes                 yes
    mean, linear (single group)           yes          yes                 yes
    equipercentile (equivalent groups)    yes (Lord)   no                  yes
    equipercentile (single group)         no           no                  yes
    tucker, levine_observed, levine_true  yes          yes                 yes
    chained_linear                        yes          yes                 yes
    braun_holland                         yes          no                  yes
    frequency_estimation                  no           no                  yes
    chained_equipercentile                no           no                  yes
    ===================================== ============ =================== =============

    All of them treat the frequencies as observed. For a presmoothed distribution the analytic
    standard errors and the bootstrap ignore that a model was fitted (the bootstrap then draws
    from the smoothed distribution without refitting it), and the total frequency is taken as
    the sample size.

    Parameters
    ----------
    x : ScoreDistribution or BivariateScoreDistribution
        Distribution of form X (equivalent groups); the joint distribution of forms X and Y taken
        by the same examinees (single group); or the joint distribution of form X and the anchor
        (non-equivalent groups).
    y : ScoreDistribution or BivariateScoreDistribution, optional
        Distribution of form Y from an independent, equivalent group, or the joint distribution
        of form Y and the anchor in the other population. Omit it for a single group.
    method : str, default "equipercentile"
        ``"identity"``, ``"mean"``, ``"linear"`` or ``"equipercentile"`` for equivalent groups
        and a single group; ``"tucker"``, ``"levine_observed"``, ``"levine_true"``,
        ``"braun_holland"``, ``"chained_linear"``, ``"frequency_estimation"`` or
        ``"chained_equipercentile"`` for non-equivalent groups.
    ddof : int, default 0
        Delta degrees of freedom of the variances and covariances in the linear methods: 0
        (population form, as in Kolen & Brennan) or 1 (sample form, as R ``equate``). The
        Braun-Holland method always uses the moments of the synthetic distributions.
    synthetic_weight : float, optional
        Weight :math:`w_1` of the population that took form X, between 0 and 1, for the Tucker,
        Levine observed-score, Braun-Holland and frequency-estimation methods. By default the
        weights are proportional to the sample sizes, :math:`w_1 = N_1 / (N_1 + N_2)`. Other
        methods ignore it.
    anchor : {"internal", "external"}, optional
        Required by the Levine methods: ``"internal"`` if the anchor items count towards the
        total scores, ``"external"`` if the anchor is scored separately. Other methods ignore it.
    standard_error : {"analytic", "analytic_normal", "bootstrap"}, optional
        Compute standard errors of equating at the possible scores of form X (see above). By
        default none are computed. The variances and covariances plugged into the analytic
        formulas follow ``ddof``; third and fourth moments always divide by :math:`N`
        (``ddof=1`` reproduces R ``equate``'s standard error of linear equating for equivalent
        groups).
    n_replications : int, default 1000
        Number of bootstrap samples :math:`B` (R ``equate::bootstrap``'s ``reps``).
    rng : numpy.random.Generator or int, optional
        Random number generator or non-negative integer seed. Required for the bootstrap, so
        that a result can be reproduced; ignored otherwise.

    Returns
    -------
    EquatingResult
        The conversion table and, for the linear methods, slope and intercept; with
        ``standard_error``, also ``see``.

    Raises
    ------
    SpecificationError
        If the inputs do not match a supported design, the method does not belong to the design,
        a required argument is missing, a variance or covariance the method divides by is not
        positive, the requested analytic standard error does not exist for the method, or the
        bootstrap is asked for without ``rng``.

    Warns
    -----
    SparseDataWarning
        Braun-Holland and frequency-estimation methods: an anchor score was observed in one
        population only. Bootstrap: some bootstrap samples could not be equated and were left
        out.

    References
    ----------
    Braun, H. I., & Holland, P. W. (1982). Observed-score test equating: A mathematical analysis
    of some ETS equating procedures. In P. W. Holland & D. B. Rubin (Eds.), *Test equating*
    (pp. 9-49). Academic Press.

    Kolen, M. J. (1985). Standard errors of Tucker equating. *Applied Psychological Measurement,
    9*, 209-223.

    Kolen, M. J., & Brennan, R. L. (2014). *Test equating, scaling, and linking* (3rd ed.).
    Springer.

    Lord, F. M. (1982). The standard error of equipercentile equating. *Journal of Educational
    Statistics, 7*, 165-174.

    Zu, J., & Yuan, K.-H. (2012). Standard error of linear observed-score equating for the NEAT
    design with nonnormally distributed data. *Journal of Educational Measurement, 49*, 190-213.

    Examples
    --------
    >>> import flexpyequate as fe
    >>> x = fe.ScoreDistribution.from_frequencies(scores=range(5), frequencies=[4, 6, 6, 3, 1])
    >>> y = fe.ScoreDistribution.from_frequencies(scores=range(5), frequencies=[2, 4, 6, 4, 4])
    >>> round(fe.equate(x, y, method="mean").intercept, 2)
    0.65
    """
    point = _equate_point(x, y, method, ddof, synthetic_weight, anchor)
    if standard_error is None:
        return point
    if standard_error not in get_args(StandardError):
        raise SpecificationError(
            f"Unknown standard_error {standard_error!r}; choose one of "
            f"{', '.join(get_args(StandardError))}, or None."
        )
    if standard_error == "bootstrap":
        generator = _as_generator(rng)
        if not (
            isinstance(n_replications, (int, np.integer))
            and not isinstance(n_replications, bool)
            and n_replications >= 2
        ):
            raise SpecificationError(
                f"n_replications must be an integer of at least 2, got {n_replications!r}."
            )
        see, mean, n_used = _bootstrap(
            x, y, point, method, ddof, anchor, int(n_replications), generator
        )
        return replace(
            point,
            see=_freeze(see),
            standard_error=standard_error,
            n_replications=n_used,
            bootstrap_mean=_freeze(mean),
        )
    see = _analytic_see(
        x, y, point, method, ddof, anchor, normal=standard_error == "analytic_normal"
    )
    return replace(point, see=_freeze(see), standard_error=standard_error)


def _as_generator(rng: np.random.Generator | int | None) -> np.random.Generator:
    """Return ``rng`` itself, or a new generator seeded with the integer ``rng``."""
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, (int, np.integer)) and not isinstance(rng, bool) and rng >= 0:
        return np.random.default_rng(int(rng))
    raise SpecificationError(
        'standard_error="bootstrap" needs rng: a numpy.random.Generator or a non-negative '
        f"integer seed, got {rng!r}."
    )


def _resample(
    distribution: ScoreDistribution | BivariateScoreDistribution, generator: np.random.Generator
) -> ScoreDistribution | BivariateScoreDistribution:
    """Draw one bootstrap sample: multinomial counts with the observed relative frequencies."""
    size = round(distribution.n)
    cells = np.ravel(distribution.frequencies)
    counts = generator.multinomial(size, cells / cells.sum()).astype(np.float64)
    if isinstance(distribution, BivariateScoreDistribution):
        return BivariateScoreDistribution(
            x_scores=distribution.x_scores,
            y_scores=distribution.y_scores,
            frequencies=counts.reshape(distribution.frequencies.shape),
        )
    return ScoreDistribution(scores=distribution.scores, frequencies=counts)


def _bootstrap(
    x: ScoreDistribution | BivariateScoreDistribution,
    y: ScoreDistribution | BivariateScoreDistribution | None,
    point: EquatingResult,
    method: EquatingMethod,
    ddof: int,
    anchor: Anchor | None,
    n_replications: int,
    generator: np.random.Generator,
) -> tuple[FloatArray, FloatArray, int]:
    r"""Bootstrap standard errors of equating (Kolen & Brennan, 2014, ch. 7).

    Each replication draws :math:`N` examinees with replacement from every sample (that is,
    multinomial counts with the observed relative frequencies; for a joint distribution the
    cells are resampled, which keeps the dependence between the two scores), form X's sample
    first and form Y's second, and equates them with the same method. With
    :math:`\hat e_b(x)` the equated scores of replication :math:`b`,

    .. math::

        \mathrm{SE}_{boot}(x) = \sqrt{\frac{1}{B - 1} \sum_b
        \bigl[\hat e_b(x) - \bar e(x)\bigr]^2} .

    Returns the standard errors, the bootstrap means and the number of replications used.
    """
    for name, source in (("x", x), ("y", y)):
        if source is not None and round(source.n) < 2:
            raise SpecificationError(f"The bootstrap needs at least two examinees in {name}.")
    equated = np.full((n_replications, point.scores.size), np.nan)
    n_sparse = 0
    for replication in range(n_replications):
        x_sample = _resample(x, generator)
        y_sample = None if y is None else _resample(y, generator)
        try:
            with warnings.catch_warnings(record=True) as caught, np.errstate(all="ignore"):
                warnings.simplefilter("always")
                result = _equate_point(
                    x_sample, y_sample, method, ddof, point.synthetic_weight, anchor
                )
        except (SpecificationError, ScoreDistributionError):
            continue
        n_sparse += any(issubclass(item.category, SparseDataWarning) for item in caught)
        equated[replication] = result.equated
    valid = np.all(np.isfinite(equated), axis=1)
    n_used = int(valid.sum())
    if n_used < 2:
        raise SpecificationError(
            "Fewer than two bootstrap samples could be equated; the data are too sparse for "
            f"method={method!r}."
        )
    if n_used < n_replications:
        warnings.warn(
            f"{n_replications - n_used} of {n_replications} bootstrap samples could not be "
            "equated (a score without variance, or a non-positive covariance) and were left out.",
            SparseDataWarning,
            stacklevel=3,
        )
    if n_sparse:
        warnings.warn(
            f"In {n_sparse} of {n_replications} bootstrap samples some anchor scores were "
            "observed in one population only; that part of the other population was left out of "
            "the synthetic distributions.",
            SparseDataWarning,
            stacklevel=3,
        )
    used = equated[valid]
    return (
        np.asarray(used.std(axis=0, ddof=1), dtype=np.float64),
        np.asarray(used.mean(axis=0), dtype=np.float64),
        n_used,
    )


def _block_diagonal(first: FloatArray, second: FloatArray) -> FloatArray:
    size = first.shape[0] + second.shape[0]
    block = np.zeros((size, size))
    block[: first.shape[0], : first.shape[0]] = first
    block[first.shape[0] :, first.shape[0] :] = second
    return block


def _analytic_see(
    x: ScoreDistribution | BivariateScoreDistribution,
    y: ScoreDistribution | BivariateScoreDistribution | None,
    point: EquatingResult,
    method: EquatingMethod,
    ddof: int,
    anchor: Anchor | None,
    *,
    normal: bool,
) -> FloatArray:
    """Delta-method standard errors of equating; see :func:`equate` for what is available."""
    scores = point.scores
    name = "analytic_normal" if normal else "analytic"

    def unavailable(reason: str) -> SpecificationError:
        return SpecificationError(
            f'standard_error="{name}" is not available for method={method!r} with '
            f'{point.design.replace("_", " ")}: {reason} Use standard_error="bootstrap".'
        )

    if method == "identity":
        return np.zeros_like(scores)
    if point.design == "nonequivalent_groups":
        assert isinstance(x, BivariateScoreDistribution)
        assert isinstance(y, BivariateScoreDistribution)
        if method in ("frequency_estimation", "chained_equipercentile"):
            raise unavailable("no formula is implemented for the equipercentile anchor methods.")
        if method == "braun_holland":
            if normal:
                raise unavailable(
                    "the method depends on the whole joint distributions, not on their moments."
                )
            assert point.synthetic_weight is not None
            see = braun_holland_se(
                x.x_scores,
                y.x_scores,
                x.probabilities,
                y.probabilities,
                x.n,
                y.n,
                point.synthetic_weight,
            )
            if see is None:
                raise unavailable("an anchor score was observed in one population only.")
            return see
        assert method in ("tucker", "levine_observed", "levine_true", "chained_linear")
        slope_gradient, intercept_gradient = anchor_linear_gradients(
            method,
            GroupMoments.of(x, ddof),
            GroupMoments.of(y, ddof),
            point.synthetic_weight,
            anchor,
        )
        omega = _block_diagonal(
            moment_covariance(
                x.x_scores, x.y_scores, x.probabilities, x.n, ddof=ddof, normal=normal
            ),
            moment_covariance(
                y.x_scores, y.y_scores, y.probabilities, y.n, ddof=ddof, normal=normal
            ),
        )
        return quadratic_form_se(scores, slope_gradient, intercept_gradient, omega)

    single_group = point.design == "single_group"
    if method == "equipercentile":
        if single_group:
            raise unavailable("Lord's (1982) formula assumes two independent samples.")
        if normal:
            raise unavailable("the formula does not depend on a distributional assumption.")
        return lord_equipercentile_se(
            point.x.scores,
            point.x.cumulative,
            point.x.n,
            point.y.scores,
            point.y.cumulative,
            point.y.n,
        )
    assert method in ("mean", "linear")
    slope_gradient, intercept_gradient = mean_linear_gradients(
        method,
        point.x.mean,
        point.x.variance(ddof=ddof),
        point.y.variance(ddof=ddof),
        single_group=single_group,
    )
    if single_group:
        assert isinstance(x, BivariateScoreDistribution)
        omega = moment_covariance(
            x.x_scores, x.y_scores, x.probabilities, x.n, ddof=ddof, normal=normal
        )
    else:
        omega = _block_diagonal(
            univariate_moment_covariance(
                point.x.scores, point.x.probabilities, point.x.n, ddof=ddof, normal=normal
            ),
            univariate_moment_covariance(
                point.y.scores, point.y.probabilities, point.y.n, ddof=ddof, normal=normal
            ),
        )
    return quadratic_form_se(scores, slope_gradient, intercept_gradient, omega)


def _equate_point(
    x: ScoreDistribution | BivariateScoreDistribution,
    y: ScoreDistribution | BivariateScoreDistribution | None,
    method: EquatingMethod,
    ddof: int,
    synthetic_weight: float | None,
    anchor: Anchor | None,
) -> EquatingResult:
    """Validate the inputs and compute the equating function (no standard errors)."""
    if method not in get_args(EquatingMethod):
        raise SpecificationError(
            f"Unknown method {method!r}; choose one of {', '.join(get_args(EquatingMethod))}."
        )
    if ddof not in (0, 1):
        raise SpecificationError(f"ddof must be 0 or 1, got {ddof}.")
    design: Design
    if isinstance(x, BivariateScoreDistribution) and isinstance(y, BivariateScoreDistribution):
        if method not in _ANCHOR_METHODS:
            raise SpecificationError(
                f"method={method!r} is not available for non-equivalent groups with an anchor "
                f"test; choose one of {', '.join(_ANCHOR_METHODS)}."
            )
        return _equate_with_anchor(x, y, method, ddof, synthetic_weight, anchor)
    if method in _ANCHOR_METHODS:
        raise SpecificationError(
            f"method={method!r} needs two BivariateScoreDistribution objects: form X with the "
            "anchor, and form Y with the anchor."
        )
    if isinstance(x, BivariateScoreDistribution):
        if y is not None:
            raise SpecificationError(
                "Pass one BivariateScoreDistribution (single group) or two (non-equivalent "
                "groups with an anchor test), not a bivariate and a univariate distribution."
            )
        design = "single_group"
        x_margin, y_margin = x.x, x.y
    elif isinstance(x, ScoreDistribution) and isinstance(y, ScoreDistribution):
        design = "equivalent_groups"
        x_margin, y_margin = x, y
    else:
        raise SpecificationError(
            "Pass two ScoreDistribution objects (equivalent groups), one "
            "BivariateScoreDistribution (single group) or two (non-equivalent groups)."
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
