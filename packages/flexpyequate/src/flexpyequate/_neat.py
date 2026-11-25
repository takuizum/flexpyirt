"""Linear equating for non-equivalent groups with an anchor test (Kolen & Brennan, 2014, ch. 4).

Population 1 takes form X and the anchor V; population 2 takes form Y and the same anchor. Pure
functions on moments and probability tables; validation lives in :mod:`flexpyequate.equating`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from flexpyequate.distributions import BivariateScoreDistribution, FloatArray

Anchor = Literal["internal", "external"]


@dataclass(frozen=True)
class GroupMoments:
    """Moments of one population: total score (form X or Y) and anchor score V."""

    mean_total: float
    mean_anchor: float
    variance_total: float
    variance_anchor: float
    covariance: float

    @classmethod
    def of(cls, joint: BivariateScoreDistribution, ddof: int) -> GroupMoments:
        return cls(
            mean_total=joint.x.mean,
            mean_anchor=joint.y.mean,
            variance_total=joint.x.variance(ddof=ddof),
            variance_anchor=joint.y.variance(ddof=ddof),
            covariance=joint.covariance(ddof=ddof),
        )


@dataclass(frozen=True)
class SyntheticMoments:
    """Means and standard deviations of X and Y in the synthetic population."""

    mean_x: float
    mean_y: float
    sd_x: float
    sd_y: float


def tucker_gamma(group: GroupMoments) -> float:
    r"""Regression slope of the total score on the anchor score.

    .. math:: \gamma = \sigma(X, V) / \sigma^2(V)

    Tucker method: the regression of X on V is linear and the same in both populations (Kolen &
    Brennan, 2014, ch. 4).
    """
    return group.covariance / group.variance_anchor


def levine_gamma(group: GroupMoments, anchor: Anchor) -> float:
    r"""Ratio of the true-score standard deviations of the total and the anchor.

    Under the classical congeneric model of Levine's methods (Kolen & Brennan, 2014, ch. 4),

    .. math::

        \gamma = \frac{\sigma^2(X)}{\sigma(X, V)} \quad \text{(internal anchor)}, \qquad
        \gamma = \frac{\sigma^2(X) + \sigma(X, V)}{\sigma^2(V) + \sigma(X, V)}
        \quad \text{(external anchor)}.
    """
    if anchor == "internal":
        return group.variance_total / group.covariance
    return (group.variance_total + group.covariance) / (group.variance_anchor + group.covariance)


def synthetic_moments(
    group_1: GroupMoments, group_2: GroupMoments, gamma_1: float, gamma_2: float, weight_1: float
) -> SyntheticMoments:
    r"""Moments of X and Y in the synthetic population :math:`T = w_1 P_1 + w_2 P_2`.

    .. math::

        \mu_s(X) &= \mu_1(X) - w_2 \gamma_1 [\mu_1(V) - \mu_2(V)] \\
        \mu_s(Y) &= \mu_2(Y) + w_1 \gamma_2 [\mu_1(V) - \mu_2(V)] \\
        \sigma_s^2(X) &= \sigma_1^2(X) - w_2 \gamma_1^2 [\sigma_1^2(V) - \sigma_2^2(V)]
                         + w_1 w_2 \gamma_1^2 [\mu_1(V) - \mu_2(V)]^2 \\
        \sigma_s^2(Y) &= \sigma_2^2(Y) + w_1 \gamma_2^2 [\sigma_1^2(V) - \sigma_2^2(V)]
                         + w_1 w_2 \gamma_2^2 [\mu_1(V) - \mu_2(V)]^2

    (Kolen & Brennan, 2014, ch. 4; the same expressions serve the Tucker and the Levine
    observed-score methods, which differ in :math:`\gamma`).
    """
    weight_2 = 1.0 - weight_1
    mean_gap = group_1.mean_anchor - group_2.mean_anchor
    variance_gap = group_1.variance_anchor - group_2.variance_anchor
    variance_x = (
        group_1.variance_total
        - weight_2 * gamma_1**2 * variance_gap
        + weight_1 * weight_2 * gamma_1**2 * mean_gap**2
    )
    variance_y = (
        group_2.variance_total
        + weight_1 * gamma_2**2 * variance_gap
        + weight_1 * weight_2 * gamma_2**2 * mean_gap**2
    )
    return SyntheticMoments(
        mean_x=group_1.mean_total - weight_2 * gamma_1 * mean_gap,
        mean_y=group_2.mean_total + weight_1 * gamma_2 * mean_gap,
        sd_x=float(np.sqrt(variance_x)),
        sd_y=float(np.sqrt(variance_y)),
    )


def levine_true_coefficients(
    group_1: GroupMoments, group_2: GroupMoments, gamma_1: float, gamma_2: float
) -> tuple[float, float]:
    r"""Slope and intercept of Levine true-score equating applied to observed scores.

    .. math::

        l_Y(x) = \frac{\gamma_2}{\gamma_1}\,[x - \mu_1(X)] + \mu_2(Y)
                 + \gamma_2\,[\mu_1(V) - \mu_2(V)]

    (Kolen & Brennan, 2014, ch. 4). It does not depend on the synthetic population.
    """
    slope = gamma_2 / gamma_1
    intercept = (
        group_2.mean_total
        + gamma_2 * (group_1.mean_anchor - group_2.mean_anchor)
        - slope * group_1.mean_total
    )
    return slope, intercept


def chained_linear_coefficients(
    group_1: GroupMoments, group_2: GroupMoments
) -> tuple[float, float]:
    r"""Slope and intercept of chained linear equating.

    X is linked to V in population 1 and V to Y in population 2:

    .. math::

        l_Y(x) = \mu_2(Y) + \frac{\sigma_2(Y)}{\sigma_2(V)}
                 \Bigl[\mu_1(V) + \frac{\sigma_1(V)}{\sigma_1(X)}\,(x - \mu_1(X)) - \mu_2(V)\Bigr]

    (Kolen & Brennan, 2014, ch. 4).
    """
    x_to_anchor = float(np.sqrt(group_1.variance_anchor / group_1.variance_total))
    anchor_to_y = float(np.sqrt(group_2.variance_total / group_2.variance_anchor))
    slope = anchor_to_y * x_to_anchor
    intercept = group_2.mean_total + anchor_to_y * (
        group_1.mean_anchor - x_to_anchor * group_1.mean_total - group_2.mean_anchor
    )
    return slope, intercept


def post_stratified_distributions(
    joint_1: FloatArray, joint_2: FloatArray, weight_1: float
) -> tuple[FloatArray, FloatArray, float]:
    r"""Score probabilities of X and Y in the synthetic population by frequency estimation.

    With joint probabilities :math:`f_1(x, v)` and :math:`g_2(y, v)` and anchor margins
    :math:`h_1(v)`, :math:`h_2(v)`, and assuming that the conditional distributions given the
    anchor score are the same in both populations,

    .. math::

        f_s(x) &= w_1 f_1(x) + w_2 \sum_v f_1(x \mid v)\, h_2(v) \\
        g_s(y) &= w_1 \sum_v g_2(y \mid v)\, h_1(v) + w_2\, g_2(y)

    (Kolen & Brennan, 2014, ch. 5). Where an anchor score occurs in one population only, the
    conditional distribution is not estimable; that part of the other population is left out and
    the result is renormalized.

    Returns
    -------
    f_s, g_s : numpy.ndarray
        Synthetic score probabilities of X and Y.
    dropped : float
        The largest anchor probability mass that had to be left out (0 if none).
    """
    anchor_1 = joint_1.sum(axis=0)
    anchor_2 = joint_2.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        conditional_1 = np.where(anchor_1 > 0, joint_1 / anchor_1, 0.0)
        conditional_2 = np.where(anchor_2 > 0, joint_2 / anchor_2, 0.0)
    synthetic_x = weight_1 * joint_1.sum(axis=1) + (1.0 - weight_1) * conditional_1 @ anchor_2
    synthetic_y = weight_1 * conditional_2 @ anchor_1 + (1.0 - weight_1) * joint_2.sum(axis=1)
    dropped = max(
        float(anchor_2[anchor_1 <= 0].sum()) if weight_1 < 1 else 0.0,
        float(anchor_1[anchor_2 <= 0].sum()) if weight_1 > 0 else 0.0,
    )
    return (
        np.asarray(synthetic_x / synthetic_x.sum(), dtype=np.float64),
        np.asarray(synthetic_y / synthetic_y.sum(), dtype=np.float64),
        dropped,
    )
