"""Kernel equating (von Davier, Holland & Thayer, 2004; Lee & von Davier, 2011).

Continuization with a Gaussian, logistic or uniform kernel, bandwidth selection, the equating
function, its standard error (SEE), the standard error of the difference between two equating
functions (SEED) and the percent relative error, for the equivalent-groups, single-group,
counterbalanced and anchor-test (chained and post-stratification) designs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.optimize import minimize_scalar
from scipy.special import expit, ndtr, ndtri

from flexpyequate.distributions import (
    BivariateScoreDistribution,
    FloatArray,
    ScoreDistribution,
    _freeze,
)
from flexpyequate.errors import SpecificationError
from flexpyequate.presmoothing import LoglinearResult, loglinear_c_matrix, saturated_c_matrix

__all__ = [
    "BandwidthMethod",
    "Kernel",
    "KernelEquatingResult",
    "kernel_equate",
    "penalty_bandwidth",
]

Kernel = Literal["gaussian", "logistic", "uniform"]
BandwidthMethod = Literal["penalty", "rule_of_thumb", "double_smoothing", "cross_validation"]
# Variance of the kernel variable V: standard normal, standard logistic, uniform on (-1/2, 1/2).
_KERNEL_VARIANCE: dict[str, float] = {
    "gaussian": 1.0,
    "logistic": float(np.pi**2 / 3.0),
    "uniform": 1.0 / 12.0,
}

_SQRT_2PI = float(np.sqrt(2.0 * np.pi))
# Half-width at which the derivative of the density is inspected (w in ch. 4), in units of the
# score increment.
_DERIVATIVE_OFFSET = 0.25
# Search range of the bandwidth in units of the score increment.
_BANDWIDTH_RANGE = (0.05, 10.0)
_N_GRID = 400
# Pilot bandwidth of the double-smoothing criterion, in units of the score increment (kequate).
_PILOT_BANDWIDTH = 4.0


def _normal_density(z: FloatArray) -> FloatArray:
    return np.asarray(np.exp(-0.5 * z * z) / _SQRT_2PI, dtype=np.float64)


def _kernel_cdf(z: FloatArray, kernel: Kernel) -> FloatArray:
    r"""Distribution function :math:`K(z)` of the kernel variable.

    Standard normal :math:`\Phi(z)`; standard logistic :math:`1/(1 + e^{-z})`; uniform on
    :math:`(-1/2, 1/2)`, :math:`\min\{\max(z + 1/2, 0), 1\}` (Lee & von Davier, 2011).
    """
    if kernel == "gaussian":
        return np.asarray(ndtr(z), dtype=np.float64)
    if kernel == "logistic":
        return np.asarray(expit(z), dtype=np.float64)
    return np.asarray(np.clip(z + 0.5, 0.0, 1.0), dtype=np.float64)


def _kernel_density(z: FloatArray, kernel: Kernel) -> FloatArray:
    r"""Return the kernel density :math:`k(z) = K'(z)`.

    :math:`\phi(z)`; :math:`K(z)[1 - K(z)]`; or 1 on :math:`[-1/2, 1/2)`.
    """
    if kernel == "gaussian":
        return _normal_density(z)
    if kernel == "logistic":
        cdf = expit(z)
        return np.asarray(cdf * (1.0 - cdf), dtype=np.float64)
    return np.asarray((z >= -0.5) & (z < 0.5), dtype=np.float64)


def _kernel_density_derivative(z: FloatArray, kernel: Kernel) -> FloatArray:
    r"""Return :math:`k'(z)`: :math:`-z\phi(z)`, :math:`k(z)[1 - 2K(z)]`, or 0."""
    if kernel == "gaussian":
        return np.asarray(-z * _normal_density(z), dtype=np.float64)
    if kernel == "logistic":
        cdf = expit(z)
        return np.asarray(cdf * (1.0 - cdf) * (1.0 - 2.0 * cdf), dtype=np.float64)
    return np.zeros_like(z)


@dataclass(frozen=True)
class _Continuized:
    r"""A discrete score distribution continuized with a kernel.

    .. math::

        F_h(x) = \sum_j r_j\, K\bigl(R_j(x)\bigr), \qquad
        R_j(x) = \frac{x - a x_j - (1 - a)\mu}{a h}, \qquad
        a^2 = \frac{\sigma^2}{\sigma^2 + \sigma_V^2 h^2}

    (von Davier et al., 2004, ch. 4, for the Gaussian kernel; Lee & von Davier, 2011, for a
    general kernel), with :math:`\mu` and :math:`\sigma^2` the mean and variance of the discrete
    distribution, :math:`K` the distribution function of the kernel variable :math:`V` and
    :math:`\sigma_V^2` its variance. The continuized variable keeps mean and variance.
    """

    scores: FloatArray
    probabilities: FloatArray
    bandwidth: float
    mean: float
    variance: float
    kernel: Kernel = "gaussian"

    @classmethod
    def build(
        cls,
        scores: FloatArray,
        probabilities: FloatArray,
        bandwidth: float,
        kernel: Kernel = "gaussian",
    ) -> _Continuized:
        mean = float(probabilities @ scores)
        variance = float(probabilities @ (scores - mean) ** 2)
        return cls(scores, probabilities, bandwidth, mean, variance, kernel)

    @property
    def shrinkage(self) -> float:
        """The factor :math:`a`."""
        kernel_variance = _KERNEL_VARIANCE[self.kernel]
        return float(np.sqrt(self.variance / (self.variance + kernel_variance * self.bandwidth**2)))

    def standardized(self, values: FloatArray) -> FloatArray:
        """Return :math:`R_j(x)`, shape ``(n_values, n_scores)``."""
        a = self.shrinkage
        centers = a * self.scores + (1.0 - a) * self.mean
        return np.asarray(
            (values[:, None] - centers[None, :]) / (a * self.bandwidth), dtype=np.float64
        )

    def cdf(self, values: FloatArray) -> FloatArray:
        """Return :math:`F_h(x)`."""
        return np.asarray(
            _kernel_cdf(self.standardized(values), self.kernel) @ self.probabilities,
            dtype=np.float64,
        )

    def density(self, values: FloatArray) -> FloatArray:
        r"""Return :math:`f_h(x) = \frac{1}{a h} \sum_j r_j\, k(R_j(x))`."""
        scale = self.shrinkage * self.bandwidth
        weighted = _kernel_density(self.standardized(values), self.kernel) @ self.probabilities
        return np.asarray(weighted / scale, dtype=np.float64)

    def density_derivative(self, values: FloatArray) -> FloatArray:
        r"""Return :math:`f_h'(x) = \frac{1}{(a h)^2} \sum_j r_j\, k'(R_j(x))`."""
        scale = self.shrinkage * self.bandwidth
        z = self.standardized(values)
        return np.asarray(
            (_kernel_density_derivative(z, self.kernel) @ self.probabilities) / scale**2,
            dtype=np.float64,
        )

    def quantile(self, proportions: FloatArray) -> FloatArray:
        """Return :math:`F_h^{-1}(u)` by a bracketed Newton iteration.

        With the uniform kernel :math:`F_h` can be flat (below, above and, for a small bandwidth,
        between the possible scores); the result is then kept inside the support of the
        continuized distribution.
        """
        a = self.shrinkage
        spread = 40.0 * a * self.bandwidth
        centers = a * self.scores + (1.0 - a) * self.mean
        low = np.full(proportions.shape, float(np.min(centers)) - spread)
        high = np.full(proportions.shape, float(np.max(centers)) + spread)
        clipped = np.clip(proportions, 1e-300, 1.0 - 1e-16)
        current = np.clip(self.mean + np.sqrt(self.variance) * ndtri(clipped), low, high)
        for _ in range(200):
            residual = self.cdf(current) - proportions
            low = np.where(residual < 0, current, low)
            high = np.where(residual >= 0, current, high)
            slope = self.density(current)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                proposal = current - residual / slope
            inside = np.isfinite(proposal) & (proposal > low) & (proposal < high)
            updated = np.where(inside, proposal, 0.5 * (low + high))
            done = np.abs(updated - current) <= 1e-14 * (1.0 + np.abs(updated))
            current = updated
            if np.all(done):
                break
        if self.kernel == "uniform":
            half_width = 0.5 * a * self.bandwidth
            current = np.clip(
                current, float(np.min(centers)) - half_width, float(np.max(centers)) + half_width
            )
        return np.asarray(current, dtype=np.float64)

    def cdf_gradient(self, values: FloatArray) -> FloatArray:
        r"""Return the derivatives of :math:`F_h(x)` with respect to the score probabilities.

        .. math::

            \frac{\partial F_h(x)}{\partial r_j} &= K\bigl(R_j(x)\bigr) - M_j(x)\, f_h(x), \\
            M_j(x) &= \tfrac12 (x - \mu)(1 - a^2)\Bigl(\frac{x_j - \mu}{\sigma}\Bigr)^2
                      + (1 - a)\, x_j

        (von Davier et al., 2004, ch. 5; the same expression holds for any kernel, with its own
        :math:`a`, Lee & von Davier, 2011). Shape ``(n_values, n_scores)``.
        """
        a = self.shrinkage
        standardized_scores = (self.scores - self.mean) ** 2 / self.variance
        moment_term = (
            0.5 * (values[:, None] - self.mean) * (1.0 - a * a) * standardized_scores[None, :]
            + (1.0 - a) * self.scores[None, :]
        )
        return np.asarray(
            _kernel_cdf(self.standardized(values), self.kernel)
            - moment_term * self.density(values)[:, None],
            dtype=np.float64,
        )


def _penalty(
    bandwidth: float,
    unit_scores: FloatArray,
    probabilities: FloatArray,
    weight: float,
    kernel: Kernel = "gaussian",
) -> float:
    r"""Bandwidth penalty on scores with unit increment.

    .. math::

        \mathrm{PEN}(h) = \sum_j \bigl(r_j - f_h(x_j)\bigr)^2 + K \sum_j A_j (1 - B_j),

    with :math:`A_j = 1` if :math:`f_h'(x_j - w) < 0`, :math:`B_j = 0` if
    :math:`f_h'(x_j + w) > 0` and :math:`w = 1/4` (von Davier et al., 2004, ch. 4). The density
    of the uniform kernel is piecewise constant, so its second term is zero (Lee & von Davier,
    2011).
    """
    continuized = _Continuized.build(unit_scores, probabilities, bandwidth, kernel)
    value = float(np.sum((probabilities - continuized.density(unit_scores)) ** 2))
    if weight > 0 and kernel != "uniform":
        left = continuized.density_derivative(unit_scores - _DERIVATIVE_OFFSET)
        right = continuized.density_derivative(unit_scores + _DERIVATIVE_OFFSET)
        value += weight * float(np.sum((left < 0) & (right > 0)))
    return value


def _minimize_over_bandwidth(
    criterion: Callable[[float], float], upper: float = _BANDWIDTH_RANGE[1]
) -> float:
    """Minimize a bandwidth criterion: grid search, then a bounded search around the best point.

    The bandwidth is in units of the score increment and is searched between 0.05 and ``upper``
    (10 by default). A grid comes first because the criteria can have several local minima and
    jumps.
    """
    grid = np.geomspace(_BANDWIDTH_RANGE[0], upper, _N_GRID)
    values = np.array([criterion(float(h)) for h in grid])
    best = int(np.argmin(values))
    lower = grid[max(best - 1, 0)]
    upper = grid[min(best + 1, grid.size - 1)]
    refined = minimize_scalar(
        criterion, bounds=(lower, upper), method="bounded", options={"xatol": 1e-12}
    )
    return float(refined.x) if refined.fun <= values[best] else float(grid[best])


def _penalty_bandwidth(
    scores: FloatArray, probabilities: FloatArray, weight: float, kernel: Kernel = "gaussian"
) -> float:
    increment = float(scores[1] - scores[0])
    unit_scores = (scores - scores[0]) / increment
    unit_bandwidth = _minimize_over_bandwidth(
        lambda h: _penalty(h, unit_scores, probabilities, weight, kernel)
    )
    return unit_bandwidth * increment


def _rule_of_thumb_bandwidth(scores: FloatArray, probabilities: FloatArray, n: float) -> float:
    r"""Rule-of-thumb bandwidth of Andersson & von Davier (2014) for the Gaussian kernel.

    .. math:: h = \frac{9\,\sigma}{\sqrt{100\,N^{2/5} - 81}}

    with :math:`\sigma` the standard deviation of the score distribution and :math:`N` the sample
    size: Silverman's rule :math:`h^* = 0.9\,\sigma N^{-1/5}` for a density estimate, adjusted for
    the shrinkage of the continuization (:math:`a h = h^*`).
    """
    mean = float(probabilities @ scores)
    sd = float(np.sqrt(probabilities @ (scores - mean) ** 2))
    return 9.0 * sd / float(np.sqrt(100.0 * n**0.4 - 81.0))


def _gaussian_weights(
    points: FloatArray, unit_scores: FloatArray, mean: float, variance: float, bandwidth: float
) -> FloatArray:
    r"""Return :math:`\phi(R_j(x))/(a h)` for every point (rows) and possible score (columns)."""
    a = float(np.sqrt(variance / (variance + bandwidth**2)))
    z = (points[:, None] - a * unit_scores[None, :] - (1.0 - a) * mean) / (a * bandwidth)
    return np.asarray(_normal_density(z) / (a * bandwidth), dtype=np.float64)


def _double_smoothing(
    bandwidth: float, unit_scores: FloatArray, probabilities: FloatArray, pilot: float
) -> float:
    r"""Double-smoothing criterion of Häggström & Wiberg (2014) on scores with unit increment.

    A pilot density :math:`f_g` with a large bandwidth :math:`g` stands in for the unknown
    density. On the grid :math:`x_l` of the possible scores and the points halfway between them
    (:math:`2J - 1` points),

    .. math::

        \mathrm{DS}(h) = \frac{1}{2J - 1} \sum_l
        \Bigl[\sum_j f_g(x_j)\, \frac{1}{a h}\phi\bigl(R_j(x_l)\bigr) - r^*(x_l)\Bigr]^2,

    where the first term continuizes the pilot values :math:`f_g(x_j)` with bandwidth :math:`h`
    and :math:`r^*(x_l)` is :math:`r_j` at a possible score and :math:`f_g(x_l)` halfway between
    two. :math:`a` and :math:`R_j` use the mean and variance of the score distribution.
    """
    mean = float(probabilities @ unit_scores)
    variance = float(probabilities @ (unit_scores - mean) ** 2)
    grid = np.arange(2 * unit_scores.size - 1) / 2.0 + unit_scores[0]
    pilot_on_grid = _gaussian_weights(grid, unit_scores, mean, variance, pilot) @ probabilities
    target = pilot_on_grid.copy()
    target[::2] = probabilities
    smoothed = _gaussian_weights(grid, unit_scores, mean, variance, bandwidth) @ pilot_on_grid[::2]
    return float(np.mean((smoothed - target) ** 2))


def _cross_validation(
    bandwidth: float, unit_scores: FloatArray, probabilities: FloatArray
) -> float:
    r"""Leave-one-out cross-validation criterion of ``kequate``, unit increment.

    Each score probability is predicted from the others by a kernel-weighted average,

    .. math::

        \mathrm{CV}(h) = \frac{1}{J} \sum_j \bigl(r_j - \hat r_{(j)}\bigr)^2, \qquad
        \hat r_{(j)} = \frac{\sum_{i \ne j} r_i\, \phi\bigl(R_i(x_j)\bigr)}
                             {\sum_{i \ne j} \phi\bigl(R_i(x_j)\bigr)},

    with :math:`a` and :math:`R_i` from the mean and variance of the whole distribution
    (Andersson et al., 2013, ``kequate`` 1.6.4, ``CV = TRUE``). This is not the likelihood
    cross-validation of Liang & von Davier (2014), which splits the sample at random.
    """
    mean = float(probabilities @ unit_scores)
    variance = float(probabilities @ (unit_scores - mean) ** 2)
    weights = _gaussian_weights(unit_scores, unit_scores, mean, variance, bandwidth)
    np.fill_diagonal(weights, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        predicted = (weights @ probabilities) / weights.sum(axis=1)
    if not np.all(np.isfinite(predicted)):
        return float("inf")
    return float(np.mean((probabilities - predicted) ** 2))


def _check_kernel_options(kernel: str, bandwidth_method: str) -> None:
    if kernel not in get_args(Kernel):
        raise SpecificationError(
            f"Unknown kernel {kernel!r}; choose one of {', '.join(get_args(Kernel))}."
        )
    if bandwidth_method not in get_args(BandwidthMethod):
        raise SpecificationError(
            f"Unknown bandwidth_method {bandwidth_method!r}; choose one of "
            f"{', '.join(get_args(BandwidthMethod))}."
        )
    if bandwidth_method != "penalty" and kernel != "gaussian":
        raise SpecificationError(
            f"bandwidth_method={bandwidth_method!r} is defined for the Gaussian kernel only; "
            f'use bandwidth_method="penalty" or give the bandwidths with kernel={kernel!r}.'
        )


def _select_bandwidth(
    scores: FloatArray,
    probabilities: FloatArray,
    n: float | None,
    method: BandwidthMethod,
    kernel: Kernel,
    weight: float,
) -> float:
    """Select the bandwidth of one continuized distribution by the named rule."""
    if method == "penalty":
        return _penalty_bandwidth(scores, probabilities, weight, kernel)
    if method == "rule_of_thumb":
        if n is None or not n > 1:
            raise SpecificationError(
                'bandwidth_method="rule_of_thumb" needs the sample size of the distribution.'
            )
        return _rule_of_thumb_bandwidth(scores, probabilities, n)
    increment = float(scores[1] - scores[0])
    unit_scores = (scores - scores[0]) / increment
    if method == "double_smoothing":
        unit_bandwidth = _minimize_over_bandwidth(
            lambda h: _double_smoothing(h, unit_scores, probabilities, _PILOT_BANDWIDTH),
            upper=_PILOT_BANDWIDTH,
        )
    else:
        unit_bandwidth = _minimize_over_bandwidth(
            lambda h: _cross_validation(h, unit_scores, probabilities)
        )
    return unit_bandwidth * increment


def penalty_bandwidth(
    distribution: ScoreDistribution | LoglinearResult,
    *,
    penalty_weight: float = 1.0,
    kernel: Kernel = "gaussian",
) -> float:
    r"""Bandwidth that minimizes the penalty of von Davier et al. (2004).

    .. math::

        \mathrm{PEN}(h) = \sum_j \bigl(r_j - f_h(x_j)\bigr)^2 + K \sum_j A_j (1 - B_j)

    (ch. 4). The first term makes the continuized density :math:`f_h` close to the score
    probabilities at the possible scores. The second counts the scores around which the density is
    U-shaped: :math:`A_j = 1` if :math:`f_h'(x_j - w) < 0` and :math:`B_j = 0` if
    :math:`f_h'(x_j + w) > 0`, with :math:`w` a quarter of the score increment.

    Scores are measured in units of their increment while the penalty is evaluated (so that
    density and probabilities are comparable), and the bandwidth is returned on the score scale.
    The minimum is searched between 0.05 and 10 increments: a grid search followed by a bounded
    minimization around the best grid point, because the penalty can have several local minima
    and jumps when :math:`K > 0`.

    Parameters
    ----------
    distribution : ScoreDistribution or LoglinearResult
        The score distribution; for a presmoothing result, the smoothed distribution is used.
    penalty_weight : float, default 1.0
        The weight :math:`K` of the second term (``kequate``'s ``KPEN``, ``SNSequate``'s ``Kp``).
        0 drops it.
    kernel : {"gaussian", "logistic", "uniform"}, default "gaussian"
        The kernel of the continuization (see :func:`kernel_equate`). The density of the
        uniform kernel is piecewise constant, so its second penalty term is zero whatever
        ``penalty_weight`` (Lee & von Davier, 2011); its first term jumps where the edge of a
        kernel passes a possible score, and the bandwidth returned is the best point found by
        the grid and the local search.

    Returns
    -------
    float
        The bandwidth :math:`h`.

    References
    ----------
    Lee, Y.-H., & von Davier, A. A. (2011). Equating through alternative kernels. In A. A. von
    Davier (Ed.), *Statistical models for test equating, scaling, and linking* (pp. 159-173).
    Springer.

    von Davier, A. A., Holland, P. W., & Thayer, D. T. (2004). *The kernel method of test
    equating*. Springer.
    """
    if not penalty_weight >= 0:
        raise SpecificationError(f"penalty_weight must be non-negative, got {penalty_weight}.")
    _check_kernel_options(kernel, "penalty")
    scores, probabilities, _, _ = _unpack(distribution, "distribution")
    return _penalty_bandwidth(scores, probabilities, float(penalty_weight), kernel)


KernelDesign = Literal[
    "equivalent_groups", "single_group", "counterbalanced", "chained", "post_stratification"
]
_TWO_SAMPLE_BIVARIATE = ("counterbalanced", "chained", "post_stratification")
# Bandwidth of the "linear" kernel equating function, in standard deviations (as kequate).
_LINEAR_BANDWIDTH_SDS = 1000.0


def _percent_relative_error(
    weights: FloatArray, equated: FloatArray, target: ScoreDistribution
) -> FloatArray:
    orders = np.arange(1, 11)
    equated_moments = np.array([weights @ equated**p for p in orders])
    target_moments = np.array([target.probabilities @ target.scores**p for p in orders])
    return np.asarray(100.0 * (equated_moments - target_moments) / target_moments, dtype=np.float64)


@dataclass(frozen=True, eq=False)
class KernelEquatingResult:
    r"""A kernel equating function from form X to the scale of form Y.

    Attributes
    ----------
    design : str
        ``"equivalent_groups"``, ``"single_group"``, ``"counterbalanced"``, ``"chained"`` or
        ``"post_stratification"``.
    x, y : ScoreDistribution
        The score distributions that were continuized: the score probabilities :math:`r` and
        :math:`s` given by the design (margins, weighted margins, or the synthetic population;
        for chained equating, X in the population that took it and Y in the population that took
        it).
    scores : numpy.ndarray
        The possible scores of form X.
    equated : numpy.ndarray
        Form-Y equivalents :math:`e_Y(x_j)`.
    see : numpy.ndarray
        Standard error of equating at each possible score.
    see_vectors : numpy.ndarray of shape (n_scores, n_columns)
        The vectors whose Euclidean norms are ``see``. Two equating functions computed from the
        same data in the same design family share their columns, which is what :meth:`seed` uses.
    bandwidth_x, bandwidth_y : float
        Bandwidths :math:`h_X` and :math:`h_Y` of the kernel.
    penalty_weight : float
        The weight :math:`K` used when a bandwidth was selected by the penalty.
    kernel : str
        ``"gaussian"``, ``"logistic"`` or ``"uniform"``.
    bandwidth_method : str
        The rule that selects the bandwidths that were not given: ``"penalty"``,
        ``"rule_of_thumb"``, ``"double_smoothing"`` or ``"cross_validation"``.
    x_presmoothed, y_presmoothed : bool
        Whether the first and the second input came from a log-linear model (which determines
        the covariance used for the standard errors). Equal for a single group.
    anchor_distributions : tuple of ScoreDistribution or None
        Chained equating: the anchor distributions in the populations that took X and Y.
    anchor_bandwidths : tuple of float or None
        Chained equating: the bandwidths of the two anchor distributions.
    synthetic_weight : float or None
        Post-stratification: weight of the population that took X in the synthetic population.
    counterbalance_weights : tuple of float or None
        Counterbalanced design: the weights of the first sample in :math:`r` and in :math:`s`.
    """

    design: KernelDesign
    x: ScoreDistribution
    y: ScoreDistribution
    scores: FloatArray
    equated: FloatArray
    see: FloatArray
    see_vectors: FloatArray
    bandwidth_x: float
    bandwidth_y: float
    penalty_weight: float
    x_presmoothed: bool
    y_presmoothed: bool
    anchor_distributions: tuple[ScoreDistribution, ScoreDistribution] | None = None
    anchor_bandwidths: tuple[float, float] | None = None
    synthetic_weight: float | None = None
    counterbalance_weights: tuple[float, float] | None = None
    kernel: Kernel = "gaussian"
    bandwidth_method: BandwidthMethod = "penalty"

    def _continuized(self) -> dict[str, _Continuized]:
        forms = {
            "x": _Continuized.build(
                self.x.scores, self.x.probabilities, self.bandwidth_x, self.kernel
            ),
            "y": _Continuized.build(
                self.y.scores, self.y.probabilities, self.bandwidth_y, self.kernel
            ),
        }
        if self.anchor_distributions is not None and self.anchor_bandwidths is not None:
            for name, anchor, bandwidth in zip(
                ("anchor_x", "anchor_y"),
                self.anchor_distributions,
                self.anchor_bandwidths,
                strict=True,
            ):
                forms[name] = _Continuized.build(
                    anchor.scores, anchor.probabilities, bandwidth, self.kernel
                )
        return forms

    def transform(self, values: ArrayLike) -> FloatArray:
        """Equate arbitrary form-X scores to the form-Y scale.

        Parameters
        ----------
        values : array_like
            Form-X scores.

        Returns
        -------
        numpy.ndarray
            Form-Y equivalents, same shape as ``values``.
        """
        points = np.asarray(values, dtype=np.float64)
        forms = self._continuized()
        flat = np.atleast_1d(points).ravel()
        proportions = forms["x"].cdf(flat)
        if "anchor_x" in forms:
            proportions = forms["anchor_y"].cdf(forms["anchor_x"].quantile(proportions))
        equated = forms["y"].quantile(proportions)
        return np.asarray(equated.reshape(points.shape), dtype=np.float64)

    @property
    def table(self) -> pd.DataFrame:
        """Conversion table: columns ``score``, ``equated`` and ``see``."""
        return pd.DataFrame({"score": self.scores, "equated": self.equated, "see": self.see})

    @property
    def continuized(self) -> pd.DataFrame:
        """Continuized distribution functions at the possible scores of each form.

        Returns
        -------
        pandas.DataFrame
            Columns ``form``, ``score``, ``probability`` (discrete), ``cdf`` and ``density``.
            ``form`` is ``"x"`` or ``"y"``, and for chained equating also ``"anchor_x"`` and
            ``"anchor_y"`` (the anchor in the populations that took X and Y).
        """
        frames = []
        for form, continuized in self._continuized().items():
            frames.append(
                pd.DataFrame(
                    {
                        "form": form,
                        "score": continuized.scores,
                        "probability": continuized.probabilities,
                        "cdf": continuized.cdf(continuized.scores),
                        "density": continuized.density(continuized.scores),
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    @property
    def percent_relative_error(self) -> pd.Series:
        r"""Percent relative error in the first ten moments.

        .. math::

            \mathrm{PRE}(p) = 100\, \frac{\mu_p(e_Y(X)) - \mu_p(Y)}{\mu_p(Y)}, \qquad
            \mu_p(e_Y(X)) = \sum_j r_j\, e_Y(x_j)^p, \quad \mu_p(Y) = \sum_k s_k\, y_k^p

        (von Davier et al., 2004, ch. 4).

        Returns
        -------
        pandas.Series
            Indexed by the moment order 1..10.

        Raises
        ------
        SpecificationError
            For chained equating, where X and Y are observed in different populations; use
            :attr:`chain_percent_relative_error`.
        """
        if self.design == "chained":
            raise SpecificationError(
                "For chained equating the percent relative error is defined for each link; use "
                "chain_percent_relative_error."
            )
        return pd.Series(
            _percent_relative_error(self.x.probabilities, self.equated, self.y),
            index=pd.Index(np.arange(1, 11), name="moment"),
            name="percent_relative_error",
        )

    @property
    def chain_percent_relative_error(self) -> pd.DataFrame:
        """Percent relative error of the two links of chained equating.

        Returns
        -------
        pandas.DataFrame
            Indexed by the moment order 1..10, with columns ``x_to_anchor`` (X equated to the
            anchor in the population that took X) and ``anchor_to_y`` (the anchor equated to Y in
            the population that took Y).

        Raises
        ------
        SpecificationError
            If the design is not chained equating.
        """
        if self.anchor_distributions is None:
            raise SpecificationError("chain_percent_relative_error needs chained equating.")
        forms = self._continuized()
        anchor_1, anchor_2 = self.anchor_distributions
        on_anchor = forms["anchor_x"].quantile(forms["x"].cdf(self.x.scores))
        on_y = forms["y"].quantile(forms["anchor_y"].cdf(anchor_2.scores))
        return pd.DataFrame(
            {
                "x_to_anchor": _percent_relative_error(self.x.probabilities, on_anchor, anchor_1),
                "anchor_to_y": _percent_relative_error(anchor_2.probabilities, on_y, self.y),
            },
            index=pd.Index(np.arange(1, 11), name="moment"),
        )

    def seed(self, other: KernelEquatingResult) -> pd.DataFrame:
        r"""Return the standard error of the difference between two equating functions (SEED).

        For two equating functions :math:`e_1` and :math:`e_2` estimated from the same data,

        .. math::

            \mathrm{SEED}(x) = \sqrt{\operatorname{Var}\bigl(e_1(x) - e_2(x)\bigr)}
                             = \bigl\lVert v_1(x) - v_2(x) \bigr\rVert,

        with :math:`v(x)` the vector whose norm is the SEE (von Davier et al., 2004, ch. 5).
        Typical uses: kernel against linear equating (``linear=True``), two sets of bandwidths,
        or chained against post-stratification equating.

        Parameters
        ----------
        other : KernelEquatingResult
            An equating function computed **from the same inputs**: the same distributions or
            presmoothing results, in the same design or in the other anchor-test design. This
            cannot be checked beyond the shapes; with different data the result is meaningless.

        Returns
        -------
        pandas.DataFrame
            Columns ``score``, ``difference`` (this function minus ``other``) and ``seed``.

        Raises
        ------
        SpecificationError
            If the two results do not have the same X scores and the same sources of sampling
            variation.
        """
        anchor_designs = ("chained", "post_stratification")
        same_family = self.design == other.design or (
            self.design in anchor_designs and other.design in anchor_designs
        )
        if (
            not same_family
            or self.scores.shape != other.scores.shape
            or not np.allclose(self.scores, other.scores)
            or self.see_vectors.shape != other.see_vectors.shape
        ):
            raise SpecificationError(
                "seed() needs two equating functions estimated from the same data: the same "
                "design (or chained and post-stratification), X scores and presmoothing models."
            )
        difference = self.see_vectors - other.see_vectors
        return pd.DataFrame(
            {
                "score": self.scores,
                "difference": self.equated - other.equated,
                "seed": np.sqrt((difference**2).sum(axis=1)),
            }
        )

    def __repr__(self) -> str:
        """Return a one-line summary."""
        return (
            f"KernelEquatingResult(design={self.design!r}, kernel={self.kernel!r}, "
            f"bandwidth_x={self.bandwidth_x:.4f}, "
            f"bandwidth_y={self.bandwidth_y:.4f}, n_scores={self.scores.size})"
        )


def _unpack(
    source: ScoreDistribution | LoglinearResult, name: str
) -> tuple[FloatArray, FloatArray, FloatArray, ScoreDistribution]:
    """Scores, probabilities, C-matrix and distribution of one form."""
    if isinstance(source, LoglinearResult):
        distribution = source.distribution
        if not isinstance(distribution, ScoreDistribution):
            raise SpecificationError(
                f"{name} is a bivariate presmoothing result; the equivalent-groups design needs "
                "the distribution of one score."
            )
        probabilities = distribution.probabilities
        c_matrix = loglinear_c_matrix(probabilities, source.design_matrix)
    elif isinstance(source, ScoreDistribution):
        distribution = source
        probabilities = distribution.probabilities
        c_matrix = saturated_c_matrix(probabilities)
    else:
        raise SpecificationError(
            f"{name} must be a ScoreDistribution or a LoglinearResult from presmooth()."
        )
    if not distribution.variance() > 0:
        raise SpecificationError(f"{name} has no variance; kernel equating is not defined.")
    return distribution.scores, probabilities, c_matrix, distribution


@dataclass(frozen=True)
class _Joint:
    """One sample with two scores: probabilities and their scaled C-matrix.

    ``c`` has shape ``(n_first, n_second, n_columns)`` and already carries the factor
    :math:`N^{-1/2}`, so that the covariance of the estimated joint probabilities is ``c c'``.
    """

    distribution: BivariateScoreDistribution
    probabilities: FloatArray
    c: FloatArray
    presmoothed: bool

    @property
    def n(self) -> float:
        return self.distribution.n


def _unpack_joint(source: object, name: str) -> _Joint:
    if isinstance(source, LoglinearResult):
        distribution = source.distribution
        if not isinstance(distribution, BivariateScoreDistribution):
            raise SpecificationError(
                f"{name} is the presmoothing result of one score; this design needs a joint "
                "distribution of two scores."
            )
        flat = np.ravel(distribution.probabilities)
        c_matrix = loglinear_c_matrix(flat, source.design_matrix)
        presmoothed = True
    elif isinstance(source, BivariateScoreDistribution):
        distribution = source
        flat = np.ravel(distribution.probabilities)
        c_matrix = saturated_c_matrix(flat)
        presmoothed = False
    else:
        raise SpecificationError(
            f"{name} must be a BivariateScoreDistribution or its LoglinearResult from presmooth()."
        )
    if not (distribution.x.variance() > 0 and distribution.y.variance() > 0):
        raise SpecificationError(f"Both scores of {name} need positive variance.")
    shape = distribution.frequencies.shape
    return _Joint(
        distribution=distribution,
        probabilities=distribution.probabilities,
        c=np.asarray(c_matrix.reshape(*shape, -1) / np.sqrt(distribution.n), dtype=np.float64),
        presmoothed=presmoothed,
    )


def _resolve_bandwidth(
    bandwidth: float | None,
    scores: FloatArray,
    probabilities: FloatArray,
    weight: float,
    linear: bool,
    name: str,
    kernel: Kernel = "gaussian",
    method: BandwidthMethod = "penalty",
    n: float | None = None,
) -> float:
    if linear:
        if bandwidth is not None:
            raise SpecificationError(f"Give either linear=True or {name}, not both.")
        mean = float(probabilities @ scores)
        return _LINEAR_BANDWIDTH_SDS * float(np.sqrt(probabilities @ (scores - mean) ** 2))
    if bandwidth is None:
        return _select_bandwidth(scores, probabilities, n, method, kernel, weight)
    if not (np.isfinite(bandwidth) and bandwidth > 0):
        raise SpecificationError(f"{name} must be a positive number, got {bandwidth}.")
    return float(bandwidth)


def _pad(left: FloatArray, right: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Put two blocks from independent samples side by side: ``[left, 0]`` and ``[0, right]``."""
    return (
        np.hstack([left, np.zeros((left.shape[0], right.shape[1]))]),
        np.hstack([np.zeros((right.shape[0], left.shape[1])), right]),
    )


def _as_distribution(scores: FloatArray, probabilities: FloatArray, n: float) -> ScoreDistribution:
    return ScoreDistribution(scores=scores, frequencies=probabilities * n)


def _same_scores(first: FloatArray, second: FloatArray) -> bool:
    return first.shape == second.shape and bool(np.allclose(first, second))


def kernel_equate(
    x: ScoreDistribution | BivariateScoreDistribution | LoglinearResult,
    y: ScoreDistribution | BivariateScoreDistribution | LoglinearResult | None = None,
    *,
    design: KernelDesign | None = None,
    bandwidth_x: float | None = None,
    bandwidth_y: float | None = None,
    anchor_bandwidths: tuple[float | None, float | None] | None = None,
    penalty_weight: float = 1.0,
    linear: bool = False,
    synthetic_weight: float | None = None,
    counterbalance_weights: tuple[float, float] = (0.5, 0.5),
    kernel: Kernel = "gaussian",
    bandwidth_method: BandwidthMethod = "penalty",
) -> KernelEquatingResult:
    r"""Kernel equating of form X to form Y.

    The discrete distributions with score probabilities :math:`r_j` (X) and :math:`s_k` (Y) are
    continuized (von Davier et al., 2004, ch. 4; Lee & von Davier, 2011),

    .. math::

        F_{h_X}(x) = \sum_j r_j\, K\!\left(\frac{x - a_X x_j - (1 - a_X)\mu_X}{a_X h_X}\right),
        \qquad a_X^2 = \frac{\sigma_X^2}{\sigma_X^2 + \sigma_V^2 h_X^2},

    and likewise :math:`G_{h_Y}`, with :math:`K` the distribution function of the kernel and
    :math:`\sigma_V^2` its variance: the standard normal (:math:`\sigma_V^2 = 1`), the standard
    logistic :math:`K(z) = 1/(1 + e^{-z})` (:math:`\pi^2/3`) or the uniform distribution on
    :math:`(-1/2, 1/2)` (:math:`1/12`). The equating function is

    .. math::

        e_Y(x) = G_{h_Y}^{-1}\bigl(F_{h_X}(x)\bigr) .

    **Designs.** The design function gives :math:`r` and :math:`s` from the estimated
    (presmoothed or observed) probabilities (ch. 2 and 3 of the book):

    - ``"equivalent_groups"``: ``x`` and ``y`` are the distributions of X and Y in two
      independent samples; :math:`r` and :math:`s` are their probabilities.
    - ``"single_group"``: ``x`` is the joint distribution :math:`P` of (X, Y) in one sample;
      :math:`r_j = \sum_k p_{jk}`, :math:`s_k = \sum_j p_{jk}`.
    - ``"counterbalanced"``: ``x`` and ``y`` are the joint distributions of (X, Y) in the sample
      that took X first and in the sample that took Y first, with margins
      :math:`r_1, s_1` and :math:`r_2, s_2`; :math:`r = w_X r_1 + (1 - w_X) r_2` and
      :math:`s = w_Y s_1 + (1 - w_Y) s_2`.
    - ``"post_stratification"``: ``x`` is the joint distribution :math:`P` of (X, A) in
      population P and ``y`` that of (Y, A), :math:`Q`, in population Q, with anchor margins
      :math:`t_P`, :math:`t_Q`. For the synthetic population :math:`T = wP + (1 - w)Q`,

      .. math::

          r_j = \sum_l \Bigl[w + (1 - w)\frac{t_{Ql}}{t_{Pl}}\Bigr] p_{jl}, \qquad
          s_k = \sum_l \Bigl[(1 - w) + w\frac{t_{Pl}}{t_{Ql}}\Bigr] q_{kl} .

    - ``"chained"``: the same inputs; X is equated to the anchor in P and the anchor to Y in Q,

      .. math::

          e_Y(x) = G_Q^{-1}\Bigl(H_Q\bigl(H_P^{-1}(F_P(x))\bigr)\Bigr),

      with four continuized distributions: X and A in P, A and Y in Q.

    **Standard error of equating** (delta method, ch. 5):

    .. math::

        \mathrm{SEE}_Y(x) = \bigl\lVert J_{e_Y}(x)\, J_{DF}\, C \bigr\rVert, \qquad
        J_{e_Y}(x) = \frac{1}{G'(e_Y(x))}
        \Bigl(\frac{\partial F(x)}{\partial r},\; -\frac{\partial G(e_Y(x))}{\partial s}\Bigr),
        \qquad
        \frac{\partial F(x)}{\partial r_j} = K\bigl(R_{jX}(x)\bigr) - M_{jX}(x)\, F'(x),

    with :math:`M_{jX}(x) = \tfrac12 (x - \mu_X)(1 - a_X^2)[(x_j - \mu_X)/\sigma_X]^2 +
    (1 - a_X) x_j` for every kernel, where :math:`C C^\top` is the covariance of the estimated
    probabilities of each sample
    (:attr:`LoglinearResult.c_matrix` divided by :math:`\sqrt N` for a presmoothed distribution,
    the multinomial covariance otherwise) and :math:`J_{DF}` the Jacobian of the design function.
    For chained equating the chain rule gives, with :math:`a = H_P^{-1}(F_P(x))`,

    .. math::

        \mathrm{SEE}_Y(x)^2 =
        \Bigl\lVert \frac{H_Q'(a)}{G_Q'(e_Y(x))} \frac{1}{H_P'(a)}
        \Bigl(\frac{\partial F_P(x)}{\partial r_P} J_{r}
        - \frac{\partial H_P(a)}{\partial t_P} J_{t}\Bigr) C_P \Bigr\rVert^2
        + \Bigl\lVert \frac{1}{G_Q'(e_Y(x))}
        \Bigl(\frac{\partial H_Q(a)}{\partial t_Q} J_{t} - \frac{\partial G_Q(e_Y(x))}{\partial s_Q}
        J_{s}\Bigr) C_Q \Bigr\rVert^2 .

    Parameters
    ----------
    x : ScoreDistribution, BivariateScoreDistribution or LoglinearResult
        First input of the design (see above). Pass the result of :func:`presmooth` to equate
        presmoothed distributions, as recommended by von Davier et al.; a plain distribution is
        used as observed.
    y : ScoreDistribution, BivariateScoreDistribution or LoglinearResult, optional
        Second input of the design; omit it for a single group.
    design : str, optional
        ``"equivalent_groups"`` and ``"single_group"`` follow from the inputs. With two joint
        distributions it is required: ``"counterbalanced"``, ``"chained"`` or
        ``"post_stratification"``.
    bandwidth_x, bandwidth_y : float, optional
        Bandwidths :math:`h_X` and :math:`h_Y`. By default each is chosen by
        ``bandwidth_method``.
    anchor_bandwidths : tuple of float, optional
        Chained equating: bandwidths of the anchor in the population that took X and in the
        population that took Y. ``None`` entries are chosen by ``bandwidth_method``.
    penalty_weight : float, default 1.0
        Weight :math:`K` of the second penalty term in the bandwidth selection.
    linear : bool, default False
        Use bandwidths of 1000 standard deviations, which gives the linear equating function of
        the design (and its standard error). Not together with explicit bandwidths, and not
        with the uniform kernel (whose support is bounded).
    synthetic_weight : float, optional
        Post-stratification: weight :math:`w` of the population that took X, between 0 and 1.
        Default :math:`N_P / (N_P + N_Q)`.
    counterbalance_weights : tuple of float, default (0.5, 0.5)
        Counterbalanced design: :math:`(w_X, w_Y)`, the weights of the sample that took X first
        in :math:`r` and in :math:`s`.
    kernel : {"gaussian", "logistic", "uniform"}, default "gaussian"
        The kernel (``kequate``'s ``kernel``; ``SNSequate``'s ``kert``). The logistic kernel is
        the standard logistic and the uniform kernel is uniform on :math:`(-1/2, 1/2)`, as in
        Lee & von Davier (2011) and the defaults of both R packages. With the uniform kernel
        the continuized density is zero outside its support and, for a bandwidth below the
        score increment, between the possible scores; where the density of Y is zero at an
        equated score, ``see`` is ``nan``.
    bandwidth_method : str, default "penalty"
        Rule for the bandwidths that are not given.

        - ``"penalty"``: :func:`penalty_bandwidth` with ``penalty_weight``; any kernel.
        - ``"rule_of_thumb"``: :math:`h = 9\sigma / \sqrt{100 N^{2/5} - 81}` (Andersson & von
          Davier, 2014), with :math:`\sigma` the standard deviation of the distribution that is
          continuized and :math:`N` the size of the sample that took the form. Not for the
          counterbalanced design.
        - ``"double_smoothing"``: minimizes the double-smoothing criterion of Häggström &
          Wiberg (2014),

          .. math::

              \mathrm{DS}(h) = \sum_{l=1}^{2J-1} \Bigl[\sum_j f_g(x_j)\,
              \frac{1}{a h}\phi\bigl(R_j(x^*_l)\bigr) - r^*_l\Bigr]^2,

          over the possible scores and the points halfway between them, where :math:`f_g` is a
          pilot continuization with bandwidth :math:`g` (4 score increments, as in
          ``kequate``) and :math:`r^*_l` is :math:`r_j` at a possible score and :math:`f_g`
          halfway between two. The minimum is searched below the pilot bandwidth.
        - ``"cross_validation"``: minimizes the leave-one-out criterion of ``kequate``,
          :math:`\sum_j (r_j - \hat r_{(j)})^2` with :math:`\hat r_{(j)} = \sum_{i \ne j} r_i
          \phi(R_i(x_j)) / \sum_{i \ne j} \phi(R_i(x_j))`. (Not the likelihood
          cross-validation of Liang & von Davier, 2014, which splits the sample at random.)

        The last three are defined for the Gaussian kernel. The standard errors treat the
        selected bandwidths as fixed, whatever the rule.

    Returns
    -------
    KernelEquatingResult
        Equated scores, standard errors, bandwidths and the percent relative error.

    Raises
    ------
    SpecificationError
        If the inputs do not fit the design, a distribution has no variance, a weight or a
        bandwidth is out of range, or (post-stratification) an anchor score has probability zero
        in one population but not in the other.

    References
    ----------
    Andersson, B., & von Davier, A. A. (2014). Improving the bandwidth selection in kernel
    equating. *Journal of Educational Measurement, 51*, 223-238.

    Häggström, J., & Wiberg, M. (2014). Optimal bandwidth selection in observed-score kernel
    equating. *Journal of Educational Measurement, 51*, 201-211.

    Lee, Y.-H., & von Davier, A. A. (2011). Equating through alternative kernels. In A. A. von
    Davier (Ed.), *Statistical models for test equating, scaling, and linking* (pp. 159-173).
    Springer.

    von Davier, A. A., Holland, P. W., & Thayer, D. T. (2004). *The kernel method of test
    equating*. Springer.
    """
    if not penalty_weight >= 0:
        raise SpecificationError(f"penalty_weight must be non-negative, got {penalty_weight}.")
    weight = float(penalty_weight)
    _check_kernel_options(kernel, bandwidth_method)
    if linear and kernel == "uniform":
        raise SpecificationError(
            "linear=True is not available with the uniform kernel: its support is bounded, so a "
            "large bandwidth does not give the linear equating function. Use the Gaussian or "
            "the logistic kernel for the linear function."
        )

    def is_univariate(source: object) -> bool:
        return isinstance(source, ScoreDistribution) or (
            isinstance(source, LoglinearResult)
            and isinstance(source.distribution, ScoreDistribution)
        )

    def is_bivariate(source: object) -> bool:
        return isinstance(source, BivariateScoreDistribution) or (
            isinstance(source, LoglinearResult)
            and isinstance(source.distribution, BivariateScoreDistribution)
        )

    if is_bivariate(x) and y is None:
        inferred: KernelDesign | None = "single_group"
    elif is_bivariate(x) and is_bivariate(y):
        inferred = None
    elif y is not None and (is_univariate(x) or is_univariate(y) or design is None):
        inferred = "equivalent_groups"
    else:
        inferred = None
    if inferred is None:
        if design not in _TWO_SAMPLE_BIVARIATE or not (is_bivariate(x) and is_bivariate(y)):
            raise SpecificationError(
                "With two joint distributions, design must be 'counterbalanced', 'chained' or "
                "'post_stratification'; with one, omit y (single group); for equivalent groups "
                "pass the distributions of X and Y."
            )
    elif design is not None and design != inferred:
        raise SpecificationError(
            f"design={design!r} does not fit the inputs, which are those of {inferred!r}."
        )
    else:
        design = inferred
    assert design is not None
    if anchor_bandwidths is not None and design != "chained":
        raise SpecificationError("anchor_bandwidths applies to chained equating only.")

    anchors: tuple[ScoreDistribution, ScoreDistribution] | None = None
    anchor_h: tuple[float, float] | None = None
    used_weight: float | None = None
    used_cb: tuple[float, float] | None = None

    if design == "equivalent_groups":
        assert y is not None
        x_scores, r, c_r, x_distribution = _unpack(x, "x")  # type: ignore[arg-type]
        y_scores, s, c_s, y_distribution = _unpack(y, "y")  # type: ignore[arg-type]
        u_r, u_s = _pad(c_r / np.sqrt(x_distribution.n), c_s / np.sqrt(y_distribution.n))
        presmoothed = (isinstance(x, LoglinearResult), isinstance(y, LoglinearResult))
        n_x, n_y = x_distribution.n, y_distribution.n
    elif design == "single_group":
        joint = _unpack_joint(x, "x")
        x_scores, y_scores = joint.distribution.x_scores, joint.distribution.y_scores
        r, s = joint.probabilities.sum(axis=1), joint.probabilities.sum(axis=0)
        u_r, u_s = joint.c.sum(axis=1), joint.c.sum(axis=0)
        x_distribution = _as_distribution(x_scores, r, joint.n)
        y_distribution = _as_distribution(y_scores, s, joint.n)
        presmoothed = (joint.presmoothed, joint.presmoothed)
        n_x = n_y = joint.n
    else:
        first, second = _unpack_joint(x, "x"), _unpack_joint(y, "y")
        presmoothed = (first.presmoothed, second.presmoothed)
        # Sample size behind each continuized distribution (rule-of-thumb bandwidth): the
        # population that took the form. The counterbalanced design mixes two samples.
        n_x, n_y = first.n, second.n
        if design == "counterbalanced" and bandwidth_method == "rule_of_thumb" and not linear:
            raise SpecificationError(
                'bandwidth_method="rule_of_thumb" is not available for the counterbalanced '
                "design, where X and Y are each observed in two samples."
            )
        if design == "counterbalanced":
            if not (
                _same_scores(first.distribution.x_scores, second.distribution.x_scores)
                and _same_scores(first.distribution.y_scores, second.distribution.y_scores)
            ):
                raise SpecificationError(
                    "Counterbalanced design: both samples need the same possible X scores "
                    "(first variable) and Y scores (second variable)."
                )
            w_x, w_y = (float(value) for value in counterbalance_weights)
            if not (0.0 <= w_x <= 1.0 and 0.0 <= w_y <= 1.0):
                raise SpecificationError("counterbalance_weights must lie in [0, 1].")
            used_cb = (w_x, w_y)
            x_scores, y_scores = first.distribution.x_scores, first.distribution.y_scores
            r = w_x * first.probabilities.sum(axis=1) + (1 - w_x) * second.probabilities.sum(axis=1)
            s = w_y * first.probabilities.sum(axis=0) + (1 - w_y) * second.probabilities.sum(axis=0)
            u_r = np.hstack([w_x * first.c.sum(axis=1), (1 - w_x) * second.c.sum(axis=1)])
            u_s = np.hstack([w_y * first.c.sum(axis=0), (1 - w_y) * second.c.sum(axis=0)])
            total = first.n + second.n
            x_distribution = _as_distribution(x_scores, r, total)
            y_distribution = _as_distribution(y_scores, s, total)
        else:
            if not _same_scores(first.distribution.y_scores, second.distribution.y_scores):
                raise SpecificationError(
                    "The anchor must be the same test in both populations: x and y need the "
                    "same possible anchor scores (their second variable)."
                )
            x_scores, y_scores = first.distribution.x_scores, second.distribution.x_scores
            anchor_scores = first.distribution.y_scores
            t_p, t_q = first.probabilities.sum(axis=0), second.probabilities.sum(axis=0)
            if design == "post_stratification":
                used_weight = (
                    first.n / (first.n + second.n)
                    if synthetic_weight is None
                    else float(synthetic_weight)
                )
                if not 0.0 <= used_weight <= 1.0:
                    raise SpecificationError(
                        f"synthetic_weight must lie in [0, 1], got {used_weight}."
                    )
                r, s, u_r, u_s = _post_stratification(first, second, t_p, t_q, used_weight)
                total = first.n + second.n
                x_distribution = _as_distribution(x_scores, r, total)
                y_distribution = _as_distribution(y_scores, s, total)
            else:
                r, s = first.probabilities.sum(axis=1), second.probabilities.sum(axis=1)
                x_distribution = _as_distribution(x_scores, r, first.n)
                y_distribution = _as_distribution(y_scores, s, second.n)
                anchors = (
                    _as_distribution(anchor_scores, t_p, first.n),
                    _as_distribution(anchor_scores, t_q, second.n),
                )

    if not (x_distribution.variance() > 0 and y_distribution.variance() > 0):
        raise SpecificationError("X and Y need positive variance for kernel equating.")
    h_x = _resolve_bandwidth(
        bandwidth_x, x_scores, r, weight, linear, "bandwidth_x", kernel, bandwidth_method, n_x
    )
    h_y = _resolve_bandwidth(
        bandwidth_y, y_scores, s, weight, linear, "bandwidth_y", kernel, bandwidth_method, n_y
    )
    F = _Continuized.build(x_scores, r, h_x, kernel)
    G = _Continuized.build(y_scores, s, h_y, kernel)

    if design == "chained":
        assert anchors is not None
        given = anchor_bandwidths if anchor_bandwidths is not None else (None, None)
        if len(given) != 2:
            raise SpecificationError("anchor_bandwidths must hold two values.")
        h_ap = _resolve_bandwidth(
            given[0],
            anchor_scores,
            t_p,
            weight,
            linear,
            "anchor_bandwidths[0]",
            kernel,
            bandwidth_method,
            n_x,
        )
        h_aq = _resolve_bandwidth(
            given[1],
            anchor_scores,
            t_q,
            weight,
            linear,
            "anchor_bandwidths[1]",
            kernel,
            bandwidth_method,
            n_y,
        )
        anchor_h = (h_ap, h_aq)
        H_P = _Continuized.build(anchor_scores, t_p, h_ap, kernel)
        H_Q = _Continuized.build(anchor_scores, t_q, h_aq, kernel)
        on_anchor = H_P.quantile(F.cdf(x_scores))
        equated = G.quantile(H_Q.cdf(on_anchor))
        density_y = G.density(equated)
        # Population P: X to the anchor, carried to the Y scale by the slope of the second link.
        link_p = F.cdf_gradient(x_scores) @ first.c.sum(axis=1) - H_P.cdf_gradient(
            on_anchor
        ) @ first.c.sum(axis=0)
        # Population Q: the anchor to Y, evaluated at the anchor equivalents.
        link_q = H_Q.cdf_gradient(on_anchor) @ second.c.sum(axis=0) - G.cdf_gradient(
            equated
        ) @ second.c.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = H_Q.density(on_anchor) / density_y / H_P.density(on_anchor)
            see_vectors = np.hstack([slope[:, None] * link_p, link_q / density_y[:, None]])
    else:
        equated = G.quantile(F.cdf(x_scores))
        with np.errstate(divide="ignore", invalid="ignore"):
            see_vectors = (
                F.cdf_gradient(x_scores) @ u_r - G.cdf_gradient(equated) @ u_s
            ) / G.density(equated)[:, None]
    # Uniform kernel: where a continuized density is zero the delta method does not apply.
    see_vectors = np.where(
        np.all(np.isfinite(see_vectors), axis=1, keepdims=True), see_vectors, np.nan
    )

    return KernelEquatingResult(
        design=design,
        x=x_distribution,
        y=y_distribution,
        scores=x_scores,
        equated=_freeze(equated),
        see=_freeze(np.sqrt((see_vectors**2).sum(axis=1))),
        see_vectors=_freeze(see_vectors),
        bandwidth_x=h_x,
        bandwidth_y=h_y,
        penalty_weight=weight,
        x_presmoothed=presmoothed[0],
        y_presmoothed=presmoothed[1],
        anchor_distributions=anchors,
        anchor_bandwidths=anchor_h,
        synthetic_weight=used_weight,
        counterbalance_weights=used_cb,
        kernel=kernel,
        bandwidth_method=bandwidth_method,
    )


def _post_stratification(
    first: _Joint, second: _Joint, t_p: FloatArray, t_q: FloatArray, weight: float
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    r"""Design function of post-stratification equating and its Jacobian times C.

    .. math::

        r_j = \sum_l \Bigl[w + (1 - w)\frac{t_{Ql}}{t_{Pl}}\Bigr] p_{jl}, \qquad
        s_k = \sum_l \Bigl[(1 - w) + w\frac{t_{Pl}}{t_{Ql}}\Bigr] q_{kl}

    (von Davier et al., 2004, ch. 2), with :math:`t_{Pl} = \sum_j p_{jl}` and
    :math:`t_{Ql} = \sum_k q_{kl}`. Differentiating,

    .. math::

        dr_j = \sum_l \Bigl[w + (1 - w)\frac{t_{Ql}}{t_{Pl}}\Bigr] dp_{jl}
               - (1 - w) \sum_l \frac{p_{jl}\, t_{Ql}}{t_{Pl}^2}\, dt_{Pl}
               + (1 - w) \sum_l \frac{p_{jl}}{t_{Pl}}\, dt_{Ql},

    and symmetrically for :math:`s`. Returns :math:`r`, :math:`s` and the matrices
    :math:`J_{DF,r} C`, :math:`J_{DF,s} C` with columns ordered as [sample P, sample Q].
    """
    used = (t_p > 0) | (t_q > 0)
    if np.any(used & ((t_p <= 0) | (t_q <= 0))):
        raise SpecificationError(
            "Post-stratification needs every observed anchor score in both populations; "
            "presmooth the joint distributions (presmooth) first."
        )
    p, q = first.probabilities[:, used], second.probabilities[:, used]
    c_p, c_q = first.c[:, used, :], second.c[:, used, :]
    t_p, t_q = t_p[used], t_q[used]
    ratio_qp, ratio_pq = t_q / t_p, t_p / t_q
    r = p @ (weight + (1 - weight) * ratio_qp)
    s = q @ ((1 - weight) + weight * ratio_pq)
    anchor_c_p, anchor_c_q = c_p.sum(axis=0), c_q.sum(axis=0)  # d t_P, d t_Q

    u_r_p = np.einsum("l,jlc->jc", weight + (1 - weight) * ratio_qp, c_p) - (1 - weight) * (
        (p * (t_q / t_p**2)) @ anchor_c_p
    )
    u_r_q = (1 - weight) * ((p / t_p) @ anchor_c_q)
    u_s_q = np.einsum("l,klc->kc", (1 - weight) + weight * ratio_pq, c_q) - weight * (
        (q * (t_p / t_q**2)) @ anchor_c_q
    )
    u_s_p = weight * ((q / t_q) @ anchor_c_p)
    return (
        np.asarray(r, dtype=np.float64),
        np.asarray(s, dtype=np.float64),
        np.hstack([u_r_p, u_r_q]),
        np.hstack([u_s_p, u_s_q]),
    )
