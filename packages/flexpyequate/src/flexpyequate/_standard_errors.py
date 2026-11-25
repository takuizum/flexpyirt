r"""Analytic standard errors of the traditional equating methods (delta method).

Pure functions on moments and probability tables; validation lives in
:mod:`flexpyequate.equating`.

Every linear method except Braun-Holland is a function of the means, variances and covariances
of the samples. With :math:`\theta` those moments, :math:`\Omega = \operatorname{Cov}(\hat\theta)`
and :math:`e_Y(x) = c(\theta) + b(\theta)\,x`,

.. math::

    \operatorname{Var}[\hat e_Y(x)] \approx
    g(x)^\top \Omega\, g(x), \qquad g(x) = \nabla c + x\, \nabla b

(Kolen, 1985; Kolen & Brennan, 2014, ch. 7; Zu & Yuan, 2012). Independent samples contribute
separate blocks of :math:`\Omega`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from flexpyequate._neat import Anchor, GroupMoments
from flexpyequate.distributions import FloatArray, percentile_rank_proportion

# Position of each moment of one sample in its moment vector.
MEAN_FIRST, MEAN_SECOND, VAR_FIRST, COVARIANCE, VAR_SECOND = range(5)
N_MOMENTS = 5

GammaMethod = Literal["tucker", "levine"]


def moment_covariance(
    first_scores: FloatArray,
    second_scores: FloatArray,
    probabilities: FloatArray,
    n: float,
    *,
    ddof: int,
    normal: bool,
) -> FloatArray:
    r"""Covariance matrix of the sample moments of two scores observed on one sample.

    The moments are ordered :math:`(\bar U, \bar V, s_{UU}, s_{UV}, s_{VV})`. With central
    moments :math:`\mu_{abcd} = E[(U - \mu_U)^{\cdot} \cdots]` of the score distribution,

    .. math::

        N \operatorname{Cov}(\bar A, \bar B) &= \sigma_{AB}, \\
        N \operatorname{Cov}(\bar A, s_{BC}) &= \mu_{ABC}, \\
        N \operatorname{Cov}(s_{AB}, s_{CD}) &= \mu_{ABCD} - \sigma_{AB}\,\sigma_{CD}

    to first order, for any distribution (the multinomial delta method; Kolen & Brennan, 2014,
    ch. 7; Zu & Yuan, 2012). Under a bivariate normal distribution (``normal=True``) the third
    moments vanish and :math:`\mu_{ABCD} - \sigma_{AB}\sigma_{CD} =
    \sigma_{AC}\sigma_{BD} + \sigma_{AD}\sigma_{BC}`.

    Parameters
    ----------
    first_scores, second_scores : numpy.ndarray
        Possible scores of U (rows of ``probabilities``) and V (columns).
    probabilities : numpy.ndarray of shape (n_first, n_second)
        Joint probabilities.
    n : float
        Sample size.
    ddof : int
        Delta degrees of freedom of the variances and covariances :math:`\sigma_{AB}` that are
        plugged in. Third and fourth moments always divide by :math:`N`.
    normal : bool
        Use the normal-theory covariance instead of the distribution-free one.

    Returns
    -------
    numpy.ndarray of shape (5, 5)
    """
    row_margin = probabilities.sum(axis=1)
    column_margin = probabilities.sum(axis=0)
    centered_first = first_scores - float(row_margin @ first_scores)
    centered_second = second_scores - float(column_margin @ second_scores)
    ones_first = np.ones_like(centered_first)
    ones_second = np.ones_like(centered_second)
    # Value of each basis function in every cell: u, v, u^2, uv, v^2 (centred).
    basis = np.stack(
        [
            np.outer(centered_first, ones_second),
            np.outer(ones_first, centered_second),
            np.outer(centered_first**2, ones_second),
            np.outer(centered_first, centered_second),
            np.outer(ones_first, centered_second**2),
        ]
    ).reshape(N_MOMENTS, -1)
    products = (basis * probabilities.ravel()) @ basis.T
    scale = n / (n - ddof)
    var_first = float(products[MEAN_FIRST, MEAN_FIRST]) * scale
    covariance = float(products[MEAN_FIRST, MEAN_SECOND]) * scale
    var_second = float(products[MEAN_SECOND, MEAN_SECOND]) * scale
    second_moments = np.array([var_first, covariance, var_second])

    omega = np.zeros((N_MOMENTS, N_MOMENTS))
    omega[:2, :2] = [[var_first, covariance], [covariance, var_second]]
    if normal:
        omega[2:, 2:] = [
            [2 * var_first**2, 2 * var_first * covariance, 2 * covariance**2],
            [
                2 * var_first * covariance,
                var_first * var_second + covariance**2,
                2 * covariance * var_second,
            ],
            [2 * covariance**2, 2 * covariance * var_second, 2 * var_second**2],
        ]
    else:
        omega[:2, 2:] = products[:2, 2:]
        omega[2:, :2] = products[2:, :2]
        omega[2:, 2:] = products[2:, 2:] - np.outer(second_moments, second_moments)
    return np.asarray(omega / n, dtype=np.float64)


def univariate_moment_covariance(
    scores: FloatArray, probabilities: FloatArray, n: float, *, ddof: int, normal: bool
) -> FloatArray:
    r"""Covariance matrix of :math:`(\bar X, s_X^2)`: the 2 x 2 case of :func:`moment_covariance`.

    .. math::

        N \operatorname{Cov} = \begin{pmatrix} \sigma^2 & \mu_3 \\ \mu_3 & \mu_4 - \sigma^4
        \end{pmatrix}
    """
    full = moment_covariance(
        scores, np.zeros(1), probabilities[:, None], n, ddof=ddof, normal=normal
    )
    keep = [MEAN_FIRST, VAR_FIRST]
    return np.asarray(full[np.ix_(keep, keep)], dtype=np.float64)


def quadratic_form_se(
    scores: FloatArray,
    slope_gradient: FloatArray,
    intercept_gradient: FloatArray,
    omega: FloatArray,
) -> FloatArray:
    r"""Return :math:`\sqrt{g(x)^\top \Omega\, g(x)}` with :math:`g(x) = \nabla c + x \nabla b`."""
    gradient = intercept_gradient[None, :] + scores[:, None] * slope_gradient[None, :]
    variance = np.einsum("ik,kl,il->i", gradient, omega, gradient)
    return np.asarray(np.sqrt(np.maximum(variance, 0.0)), dtype=np.float64)


def mean_linear_gradients(
    method: Literal["mean", "linear"],
    mean_x: float,
    var_x: float,
    var_y: float,
    *,
    single_group: bool,
) -> tuple[FloatArray, FloatArray]:
    r"""Gradients of slope and intercept of mean and linear equating.

    For linear equating :math:`b = \sigma_Y / \sigma_X` and :math:`c = \mu_Y - b \mu_X`, so

    .. math::

        \frac{\partial b}{\partial \sigma_X^2} = -\frac{b}{2\sigma_X^2}, \qquad
        \frac{\partial b}{\partial \sigma_Y^2} = \frac{b}{2\sigma_Y^2}, \qquad
        \nabla c = \nabla \mu_Y - b\, \nabla \mu_X - \mu_X \nabla b ;

    for mean equating :math:`b = 1`. With two independent samples these give the standard error
    of Braun & Holland (1982; Kolen & Brennan, 2014, ch. 7),

    .. math::

        \operatorname{Var}[\hat l_Y(x)] \approx \sigma_Y^2 \Bigl\{ \frac{1}{N_X} + \frac{1}{N_Y}
        + \Bigl[\frac{\mathrm{sk}(X)}{N_X} + \frac{\mathrm{sk}(Y)}{N_Y}\Bigr] z
        + \Bigl[\frac{\mathrm{ku}(X) - 1}{4 N_X} + \frac{\mathrm{ku}(Y) - 1}{4 N_Y}\Bigr] z^2
        \Bigr\}, \qquad z = \frac{x - \mu_X}{\sigma_X} .

    Returns
    -------
    slope_gradient, intercept_gradient : numpy.ndarray
        With respect to :math:`(\mu_X, \sigma_X^2, \mu_Y, \sigma_Y^2)` for equivalent groups and
        to :math:`(\mu_X, \mu_Y, \sigma_X^2, \sigma_{XY}, \sigma_Y^2)` for a single group.
    """
    if single_group:
        size, i_mean_x, i_mean_y, i_var_x, i_var_y = (
            5,
            MEAN_FIRST,
            MEAN_SECOND,
            VAR_FIRST,
            VAR_SECOND,
        )
    else:
        size, i_mean_x, i_var_x, i_mean_y, i_var_y = 4, 0, 1, 2, 3
    slope_gradient = np.zeros(size)
    intercept_gradient = np.zeros(size)
    slope = 1.0
    if method == "linear":
        slope = float(np.sqrt(var_y / var_x))
        slope_gradient[i_var_x] = -slope / (2 * var_x)
        slope_gradient[i_var_y] = slope / (2 * var_y)
    intercept_gradient[i_mean_y] = 1.0
    intercept_gradient[i_mean_x] = -slope
    intercept_gradient -= mean_x * slope_gradient
    return slope_gradient, intercept_gradient


@dataclass(frozen=True)
class _Gamma:
    r"""A :math:`\gamma` and its gradient with respect to the five moments of its population."""

    value: float
    gradient: FloatArray


def _gamma(group: GroupMoments, method: GammaMethod, anchor: Anchor | None) -> _Gamma:
    r"""Return :math:`\gamma` of the Tucker or Levine method with its gradient.

    .. math::

        \text{Tucker:}\quad & \gamma = \frac{\sigma_{XV}}{\sigma_V^2}, &
            \frac{\partial\gamma}{\partial\sigma_{XV}} &= \frac{1}{\sigma_V^2}, &
            \frac{\partial\gamma}{\partial\sigma_V^2} &= -\frac{\sigma_{XV}}{\sigma_V^4} \\
        \text{Levine, internal:}\quad & \gamma = \frac{\sigma_X^2}{\sigma_{XV}}, &
            \frac{\partial\gamma}{\partial\sigma_X^2} &= \frac{1}{\sigma_{XV}}, &
            \frac{\partial\gamma}{\partial\sigma_{XV}} &= -\frac{\sigma_X^2}{\sigma_{XV}^2} \\
        \text{Levine, external:}\quad &
            \gamma = \frac{\sigma_X^2 + \sigma_{XV}}{\sigma_V^2 + \sigma_{XV}}, &
            \frac{\partial\gamma}{\partial\sigma_X^2} &= \frac{1}{\sigma_V^2 + \sigma_{XV}}, &
            \frac{\partial\gamma}{\partial\sigma_{XV}} &=
                \frac{\sigma_V^2 - \sigma_X^2}{(\sigma_V^2 + \sigma_{XV})^2}, \quad
            \frac{\partial\gamma}{\partial\sigma_V^2} =
                -\frac{\sigma_X^2 + \sigma_{XV}}{(\sigma_V^2 + \sigma_{XV})^2}
    """
    gradient = np.zeros(N_MOMENTS)
    total, anchor_variance, covariance = (
        group.variance_total,
        group.variance_anchor,
        group.covariance,
    )
    if method == "tucker":
        value = covariance / anchor_variance
        gradient[COVARIANCE] = 1 / anchor_variance
        gradient[VAR_SECOND] = -covariance / anchor_variance**2
    elif anchor == "internal":
        value = total / covariance
        gradient[VAR_FIRST] = 1 / covariance
        gradient[COVARIANCE] = -total / covariance**2
    else:
        denominator = anchor_variance + covariance
        value = (total + covariance) / denominator
        gradient[VAR_FIRST] = 1 / denominator
        gradient[COVARIANCE] = (anchor_variance - total) / denominator**2
        gradient[VAR_SECOND] = -(total + covariance) / denominator**2
    return _Gamma(float(value), gradient)


def _unit(position: int, group: int) -> FloatArray:
    """Return the unit vector of one moment of population 1 (``group=0``) or 2 in the 10-vector."""
    vector = np.zeros(2 * N_MOMENTS)
    vector[group * N_MOMENTS + position] = 1.0
    return vector


def _embed(gradient: FloatArray, group: int) -> FloatArray:
    vector = np.zeros(2 * N_MOMENTS)
    vector[group * N_MOMENTS : (group + 1) * N_MOMENTS] = gradient
    return vector


def anchor_linear_gradients(
    method: Literal["tucker", "levine_observed", "levine_true", "chained_linear"],
    group_1: GroupMoments,
    group_2: GroupMoments,
    weight_1: float | None,
    anchor: Anchor | None,
) -> tuple[FloatArray, FloatArray]:
    r"""Gradients of slope and intercept of the linear anchor-test methods.

    The ten moments are those of population 1, :math:`(\mu_1(X), \mu_1(V), \sigma_1^2(X),
    \sigma_1(X, V), \sigma_1^2(V))`, followed by those of population 2 with Y in place of X.

    **Tucker and Levine observed score** (Kolen, 1985; Kolen & Brennan, 2014, ch. 7). With
    :math:`\Delta_\mu = \mu_1(V) - \mu_2(V)`, :math:`\Delta_\sigma = \sigma_1^2(V) -
    \sigma_2^2(V)` and the synthetic moments of :func:`flexpyequate._neat.synthetic_moments`,

    .. math::

        d\mu_s(X) &= d\mu_1(X) - w_2 \Delta_\mu\, d\gamma_1 - w_2 \gamma_1\, d\Delta_\mu \\
        d\mu_s(Y) &= d\mu_2(Y) + w_1 \Delta_\mu\, d\gamma_2 + w_1 \gamma_2\, d\Delta_\mu \\
        d\sigma_s^2(X) &= d\sigma_1^2(X)
            + 2\gamma_1 (-w_2 \Delta_\sigma + w_1 w_2 \Delta_\mu^2)\, d\gamma_1
            - w_2 \gamma_1^2\, d\Delta_\sigma + 2 w_1 w_2 \gamma_1^2 \Delta_\mu\, d\Delta_\mu \\
        d\sigma_s^2(Y) &= d\sigma_2^2(Y)
            + 2\gamma_2 (w_1 \Delta_\sigma + w_1 w_2 \Delta_\mu^2)\, d\gamma_2
            + w_1 \gamma_2^2\, d\Delta_\sigma + 2 w_1 w_2 \gamma_2^2 \Delta_\mu\, d\Delta_\mu

    and, for :math:`b = \sigma_s(Y)/\sigma_s(X)` and :math:`c = \mu_s(Y) - b\,\mu_s(X)`,

    .. math::

        db = \frac{b}{2}\Bigl[\frac{d\sigma_s^2(Y)}{\sigma_s^2(Y)}
             - \frac{d\sigma_s^2(X)}{\sigma_s^2(X)}\Bigr], \qquad
        dc = d\mu_s(Y) - b\, d\mu_s(X) - \mu_s(X)\, db .

    **Levine true score**: :math:`b = \gamma_2/\gamma_1`,
    :math:`c = \mu_2(Y) + \gamma_2 \Delta_\mu - b\,\mu_1(X)`.

    **Chained linear**: :math:`b = k\,\sigma_1(V)/\sigma_1(X)` with
    :math:`k = \sigma_2(Y)/\sigma_2(V)`, :math:`c = \mu_2(Y) + k \Delta_\mu - b\,\mu_1(X)`.

    Returns
    -------
    slope_gradient, intercept_gradient : numpy.ndarray of shape (10,)
    """
    mean_gap = group_1.mean_anchor - group_2.mean_anchor
    d_mean_gap = _unit(MEAN_SECOND, 0) - _unit(MEAN_SECOND, 1)

    if method == "chained_linear":
        anchor_to_y = float(np.sqrt(group_2.variance_total / group_2.variance_anchor))
        d_anchor_to_y = (anchor_to_y / 2) * (
            _unit(VAR_FIRST, 1) / group_2.variance_total
            - _unit(VAR_SECOND, 1) / group_2.variance_anchor
        )
        slope = anchor_to_y * float(np.sqrt(group_1.variance_anchor / group_1.variance_total))
        d_slope = slope * d_anchor_to_y / anchor_to_y + (slope / 2) * (
            _unit(VAR_SECOND, 0) / group_1.variance_anchor
            - _unit(VAR_FIRST, 0) / group_1.variance_total
        )
        d_intercept = (
            _unit(MEAN_FIRST, 1)
            + mean_gap * d_anchor_to_y
            + anchor_to_y * d_mean_gap
            - group_1.mean_total * d_slope
            - slope * _unit(MEAN_FIRST, 0)
        )
        return d_slope, d_intercept

    gamma_method: GammaMethod = "tucker" if method == "tucker" else "levine"
    gamma_1 = _gamma(group_1, gamma_method, anchor)
    gamma_2 = _gamma(group_2, gamma_method, anchor)
    d_gamma_1 = _embed(gamma_1.gradient, 0)
    d_gamma_2 = _embed(gamma_2.gradient, 1)

    if method == "levine_true":
        slope = gamma_2.value / gamma_1.value
        d_slope = d_gamma_2 / gamma_1.value - slope / gamma_1.value * d_gamma_1
        d_intercept = (
            _unit(MEAN_FIRST, 1)
            + mean_gap * d_gamma_2
            + gamma_2.value * d_mean_gap
            - group_1.mean_total * d_slope
            - slope * _unit(MEAN_FIRST, 0)
        )
        return d_slope, d_intercept

    assert weight_1 is not None
    weight_2 = 1.0 - weight_1
    variance_gap = group_1.variance_anchor - group_2.variance_anchor
    d_variance_gap = _unit(VAR_SECOND, 0) - _unit(VAR_SECOND, 1)
    g1, g2 = gamma_1.value, gamma_2.value
    cross = weight_1 * weight_2 * mean_gap**2

    mean_x = group_1.mean_total - weight_2 * g1 * mean_gap
    variance_x = group_1.variance_total - weight_2 * g1**2 * variance_gap + cross * g1**2
    variance_y = group_2.variance_total + weight_1 * g2**2 * variance_gap + cross * g2**2
    d_mean_x = _unit(MEAN_FIRST, 0) - weight_2 * mean_gap * d_gamma_1 - weight_2 * g1 * d_mean_gap
    d_mean_y = _unit(MEAN_FIRST, 1) + weight_1 * mean_gap * d_gamma_2 + weight_1 * g2 * d_mean_gap
    d_variance_x = (
        _unit(VAR_FIRST, 0)
        + 2 * g1 * (-weight_2 * variance_gap + cross) * d_gamma_1
        - weight_2 * g1**2 * d_variance_gap
        + 2 * weight_1 * weight_2 * g1**2 * mean_gap * d_mean_gap
    )
    d_variance_y = (
        _unit(VAR_FIRST, 1)
        + 2 * g2 * (weight_1 * variance_gap + cross) * d_gamma_2
        + weight_1 * g2**2 * d_variance_gap
        + 2 * weight_1 * weight_2 * g2**2 * mean_gap * d_mean_gap
    )
    slope = float(np.sqrt(variance_y / variance_x))
    d_slope = (slope / 2) * (d_variance_y / variance_y - d_variance_x / variance_x)
    d_intercept = d_mean_y - slope * d_mean_x - mean_x * d_slope
    return d_slope, d_intercept


def braun_holland_se(
    x_scores: FloatArray,
    y_scores: FloatArray,
    joint_1: FloatArray,
    joint_2: FloatArray,
    n_1: float,
    n_2: float,
    weight_1: float,
) -> FloatArray | None:
    r"""Return the standard error of Braun-Holland linear equating (delta method on the cells).

    The synthetic probabilities are those of post-stratification,

    .. math::

        f_s(x_j) = \sum_l \Bigl[w_1 + w_2 \frac{h_{2l}}{h_{1l}}\Bigr] p_{jl}, \qquad
        g_s(y_k) = \sum_l \Bigl[w_2 + w_1 \frac{h_{1l}}{h_{2l}}\Bigr] q_{kl},

    with joint probabilities :math:`p` (X, V in population 1) and :math:`q` (Y, V in population
    2) and anchor margins :math:`h_1`, :math:`h_2`. The equated score
    :math:`e = \mu_s(Y) + b\,(x - \mu_s(X))`, :math:`b = \sigma_s(Y)/\sigma_s(X)`, has

    .. math::

        de = \sum_j A_j\, df_s(x_j) + \sum_k B_k\, dg_s(y_k), \quad
        A_j = -b\,x_j - \frac{b\,(x - \mu_s(X))(x_j - \mu_s(X))^2}{2\sigma_s^2(X)}, \quad
        B_k = y_k + \frac{b\,(x - \mu_s(X))(y_k - \mu_s(Y))^2}{2\sigma_s^2(Y)},

    and, with :math:`\bar A_l = \sum_j A_j p_{jl}` and :math:`\bar B_l = \sum_k B_k q_{kl}`,

    .. math::

        \frac{\partial e}{\partial p_{jl}} &= A_j \Bigl[w_1 + w_2 \frac{h_{2l}}{h_{1l}}\Bigr]
            - w_2 \frac{h_{2l}}{h_{1l}^2} \bar A_l + w_1 \frac{\bar B_l}{h_{2l}}, \\
        \frac{\partial e}{\partial q_{kl}} &= B_k \Bigl[w_2 + w_1 \frac{h_{1l}}{h_{2l}}\Bigr]
            - w_1 \frac{h_{1l}}{h_{2l}^2} \bar B_l + w_2 \frac{\bar A_l}{h_{1l}} .

    The variance is :math:`\sum_{\text{samples}} [\sum p\, (\partial e)^2 - (\sum p\, \partial
    e)^2]/N` (multinomial sampling of each joint distribution).

    Returns
    -------
    numpy.ndarray or None
        Standard errors at ``x_scores``; ``None`` if an anchor score has probability zero in
        exactly one population, where the derivative does not exist.
    """
    anchor_1 = joint_1.sum(axis=0)
    anchor_2 = joint_2.sum(axis=0)
    used = (anchor_1 > 0) | (anchor_2 > 0)
    if np.any(used & ((anchor_1 <= 0) | (anchor_2 <= 0))):
        return None
    p, q = joint_1[:, used], joint_2[:, used]
    h1, h2 = anchor_1[used], anchor_2[used]
    weight_2 = 1.0 - weight_1
    factor_p = weight_1 + weight_2 * h2 / h1
    factor_q = weight_2 + weight_1 * h1 / h2
    synthetic_x = p @ factor_p
    synthetic_y = q @ factor_q
    mean_x = float(synthetic_x @ x_scores)
    mean_y = float(synthetic_y @ y_scores)
    variance_x = float(synthetic_x @ (x_scores - mean_x) ** 2)
    variance_y = float(synthetic_y @ (y_scores - mean_y) ** 2)
    slope = float(np.sqrt(variance_y / variance_x))

    deviation = x_scores - mean_x  # one row of coefficients per equated score
    a = -slope * x_scores[None, :] - slope * np.outer(deviation, (x_scores - mean_x) ** 2) / (
        2 * variance_x
    )
    b = y_scores[None, :] + slope * np.outer(deviation, (y_scores - mean_y) ** 2) / (2 * variance_y)
    a_bar = a @ p  # (n_scores, n_anchor)
    b_bar = b @ q
    gradient_p = (
        a[:, :, None] * factor_p[None, None, :]
        + (-weight_2 * h2 / h1**2 * a_bar + weight_1 * b_bar / h2)[:, None, :]
    )
    gradient_q = (
        b[:, :, None] * factor_q[None, None, :]
        + (-weight_1 * h1 / h2**2 * b_bar + weight_2 * a_bar / h1)[:, None, :]
    )

    def multinomial_variance(gradient: FloatArray, cells: FloatArray, n: float) -> FloatArray:
        first = np.einsum("ijl,jl->i", gradient, cells)
        second = np.einsum("ijl,jl->i", gradient**2, cells)
        return np.asarray((second - first**2) / n, dtype=np.float64)

    variance = multinomial_variance(gradient_p, p, n_1) + multinomial_variance(gradient_q, q, n_2)
    return np.asarray(np.sqrt(np.maximum(variance, 0.0)), dtype=np.float64)


def lord_equipercentile_se(
    x_scores: FloatArray,
    x_cumulative: FloatArray,
    n_x: float,
    y_scores: FloatArray,
    y_cumulative: FloatArray,
    n_y: float,
) -> FloatArray:
    r"""Return the standard error of equipercentile equating for equivalent groups (Lord, 1982).

    With :math:`p = P(x)` the percentile rank of :math:`x` as a proportion,
    :math:`y^*_U = \min\{y_k : G(y_k) > p\}`, :math:`G_0 = G(y^*_U - \delta_Y)` and
    :math:`g = G(y^*_U) - G_0`,

    .. math::

        \operatorname{Var}[\hat e_Y(x)] \approx \frac{\delta_Y^2}{g^2} \Bigl\{
        \frac{p(1 - p)}{N_X} + \frac{1}{N_Y}\Bigl[G_0 - p^2 + \frac{(p - G_0)^2}{g}\Bigr]
        \Bigr\}

    (Lord, 1982; Kolen & Brennan, 2014, ch. 7, where the braces read
    :math:`p(1-p)(N_X + N_Y)/(N_X N_Y) - [G(y^*_U) - p][p - G_0]/(N_Y g)`, which is the same
    expression). The standard error is 0 where :math:`p` is 0 or 1.
    """
    increment = float(y_scores[1] - y_scores[0])
    rank = percentile_rank_proportion(x_scores, x_cumulative, x_scores)
    extended = np.concatenate(([0.0], y_cumulative))
    first_above = np.searchsorted(y_cumulative, rank, side="right")
    inside = (rank > 0) & (first_above < y_scores.size)
    index = np.minimum(first_above, y_scores.size - 1)
    below = extended[index]
    step = extended[index + 1] - below
    with np.errstate(divide="ignore", invalid="ignore"):
        variance = (
            rank * (1 - rank) / n_x + (below - rank**2 + (rank - below) ** 2 / step) / n_y
        ) / step**2
    variance = np.where(inside, variance, 0.0)
    return np.asarray(increment * np.sqrt(np.maximum(variance, 0.0)), dtype=np.float64)
