"""Kernel equating with a Gaussian kernel (von Davier, Holland & Thayer, 2004).

Implemented for the equivalent-groups design: continuization, penalty bandwidth selection,
the equating function, its standard error (SEE) and the percent relative error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.optimize import minimize_scalar
from scipy.special import ndtr, ndtri

from flexpyequate.distributions import FloatArray, ScoreDistribution, _freeze
from flexpyequate.errors import SpecificationError
from flexpyequate.presmoothing import LoglinearResult, loglinear_c_matrix, saturated_c_matrix

__all__ = ["KernelEquatingResult", "kernel_equate", "penalty_bandwidth"]

_SQRT_2PI = float(np.sqrt(2.0 * np.pi))
# Half-width at which the derivative of the density is inspected (w in eq. 4.30), in units of the
# score increment.
_DERIVATIVE_OFFSET = 0.25
# Search range of the bandwidth in units of the score increment.
_BANDWIDTH_RANGE = (0.05, 10.0)
_N_GRID = 400


def _normal_density(z: FloatArray) -> FloatArray:
    return np.asarray(np.exp(-0.5 * z * z) / _SQRT_2PI, dtype=np.float64)


@dataclass(frozen=True)
class _Continuized:
    r"""A discrete score distribution continuized with a Gaussian kernel.

    .. math::

        F_h(x) = \sum_j r_j\, \Phi\bigl(R_j(x)\bigr), \qquad
        R_j(x) = \frac{x - a x_j - (1 - a)\mu}{a h}, \qquad
        a^2 = \frac{\sigma^2}{\sigma^2 + h^2}

    (von Davier et al., 2004, eq. 4.5-4.7), with :math:`\mu` and :math:`\sigma^2` the mean and
    variance of the discrete distribution. The continuized variable keeps both.
    """

    scores: FloatArray
    probabilities: FloatArray
    bandwidth: float
    mean: float
    variance: float

    @classmethod
    def build(cls, scores: FloatArray, probabilities: FloatArray, bandwidth: float) -> _Continuized:
        mean = float(probabilities @ scores)
        variance = float(probabilities @ (scores - mean) ** 2)
        return cls(scores, probabilities, bandwidth, mean, variance)

    @property
    def shrinkage(self) -> float:
        """The factor :math:`a`."""
        return float(np.sqrt(self.variance / (self.variance + self.bandwidth**2)))

    def standardized(self, values: FloatArray) -> FloatArray:
        """Return :math:`R_j(x)`, shape ``(n_values, n_scores)``."""
        a = self.shrinkage
        centers = a * self.scores + (1.0 - a) * self.mean
        return np.asarray(
            (values[:, None] - centers[None, :]) / (a * self.bandwidth), dtype=np.float64
        )

    def cdf(self, values: FloatArray) -> FloatArray:
        """Return :math:`F_h(x)`."""
        return np.asarray(ndtr(self.standardized(values)) @ self.probabilities, dtype=np.float64)

    def density(self, values: FloatArray) -> FloatArray:
        r"""Return :math:`f_h(x) = \frac{1}{a h} \sum_j r_j\, \phi(R_j(x))`."""
        scale = self.shrinkage * self.bandwidth
        weighted = _normal_density(self.standardized(values)) @ self.probabilities
        return np.asarray(weighted / scale, dtype=np.float64)

    def density_derivative(self, values: FloatArray) -> FloatArray:
        r"""Return :math:`f_h'(x) = -\frac{1}{(a h)^2} \sum_j r_j\, R_j(x)\, \phi(R_j(x))`."""
        scale = self.shrinkage * self.bandwidth
        z = self.standardized(values)
        return np.asarray(
            -((z * _normal_density(z)) @ self.probabilities) / scale**2, dtype=np.float64
        )

    def quantile(self, proportions: FloatArray) -> FloatArray:
        """Return :math:`F_h^{-1}(u)` by a bracketed Newton iteration."""
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
        return np.asarray(current, dtype=np.float64)

    def cdf_gradient(self, values: FloatArray) -> FloatArray:
        r"""Return the derivatives of :math:`F_h(x)` with respect to the score probabilities.

        .. math::

            \frac{\partial F_h(x)}{\partial r_j} &= \Phi\bigl(R_j(x)\bigr) - M_j(x)\, f_h(x), \\
            M_j(x) &= \tfrac12 (x - \mu)(1 - a^2)\Bigl(\frac{x_j - \mu}{\sigma}\Bigr)^2
                      + (1 - a)\, x_j

        (von Davier et al., 2004, eq. 5.13-5.15). Shape ``(n_values, n_scores)``.
        """
        a = self.shrinkage
        standardized_scores = (self.scores - self.mean) ** 2 / self.variance
        moment_term = (
            0.5 * (values[:, None] - self.mean) * (1.0 - a * a) * standardized_scores[None, :]
            + (1.0 - a) * self.scores[None, :]
        )
        return np.asarray(
            ndtr(self.standardized(values)) - moment_term * self.density(values)[:, None],
            dtype=np.float64,
        )


def _penalty(
    bandwidth: float, unit_scores: FloatArray, probabilities: FloatArray, weight: float
) -> float:
    r"""Bandwidth penalty on scores with unit increment.

    .. math::

        \mathrm{PEN}(h) = \sum_j \bigl(r_j - f_h(x_j)\bigr)^2 + K \sum_j A_j (1 - B_j),

    with :math:`A_j = 1` if :math:`f_h'(x_j - w) < 0`, :math:`B_j = 0` if
    :math:`f_h'(x_j + w) > 0` and :math:`w = 1/4` (von Davier et al., 2004, eq. 4.27-4.30).
    """
    continuized = _Continuized.build(unit_scores, probabilities, bandwidth)
    value = float(np.sum((probabilities - continuized.density(unit_scores)) ** 2))
    if weight > 0:
        left = continuized.density_derivative(unit_scores - _DERIVATIVE_OFFSET)
        right = continuized.density_derivative(unit_scores + _DERIVATIVE_OFFSET)
        value += weight * float(np.sum((left < 0) & (right > 0)))
    return value


def _penalty_bandwidth(scores: FloatArray, probabilities: FloatArray, weight: float) -> float:
    increment = float(scores[1] - scores[0])
    unit_scores = (scores - scores[0]) / increment
    grid = np.geomspace(*_BANDWIDTH_RANGE, _N_GRID)
    values = np.array([_penalty(h, unit_scores, probabilities, weight) for h in grid])
    best = int(np.argmin(values))
    lower = grid[max(best - 1, 0)]
    upper = grid[min(best + 1, grid.size - 1)]
    refined = minimize_scalar(
        _penalty,
        bounds=(lower, upper),
        args=(unit_scores, probabilities, weight),
        method="bounded",
        options={"xatol": 1e-12},
    )
    unit_bandwidth = float(refined.x) if refined.fun <= values[best] else float(grid[best])
    return unit_bandwidth * increment


def penalty_bandwidth(
    distribution: ScoreDistribution | LoglinearResult, *, penalty_weight: float = 1.0
) -> float:
    r"""Bandwidth of the Gaussian kernel that minimizes the penalty of von Davier et al. (2004).

    .. math::

        \mathrm{PEN}(h) = \sum_j \bigl(r_j - f_h(x_j)\bigr)^2 + K \sum_j A_j (1 - B_j)

    (eq. 4.27-4.30). The first term makes the continuized density :math:`f_h` close to the score
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

    Returns
    -------
    float
        The bandwidth :math:`h`.

    References
    ----------
    von Davier, A. A., Holland, P. W., & Thayer, D. T. (2004). *The kernel method of test
    equating*. Springer.
    """
    if not penalty_weight >= 0:
        raise SpecificationError(f"penalty_weight must be non-negative, got {penalty_weight}.")
    scores, probabilities, _, _ = _unpack(distribution, "distribution")
    return _penalty_bandwidth(scores, probabilities, float(penalty_weight))


@dataclass(frozen=True, eq=False)
class KernelEquatingResult:
    r"""A kernel equating function from form X to the scale of form Y.

    Attributes
    ----------
    design : str
        ``"equivalent_groups"``.
    x, y : ScoreDistribution
        The (possibly presmoothed) score distributions that were continuized.
    scores : numpy.ndarray
        The possible scores of form X.
    equated : numpy.ndarray
        Form-Y equivalents :math:`e_Y(x_j) = G_{h_Y}^{-1}(F_{h_X}(x_j))`.
    see : numpy.ndarray
        Standard error of equating at each possible score.
    bandwidth_x, bandwidth_y : float
        Bandwidths :math:`h_X` and :math:`h_Y` of the Gaussian kernel.
    penalty_weight : float
        The weight :math:`K` used when a bandwidth was selected by the penalty.
    x_presmoothed, y_presmoothed : bool
        Whether the score probabilities came from a log-linear model (which determines the
        covariance used for the standard errors).
    """

    design: Literal["equivalent_groups"]
    x: ScoreDistribution
    y: ScoreDistribution
    scores: FloatArray
    equated: FloatArray
    see: FloatArray
    bandwidth_x: float
    bandwidth_y: float
    penalty_weight: float
    x_presmoothed: bool
    y_presmoothed: bool

    def _continuized(self) -> tuple[_Continuized, _Continuized]:
        return (
            _Continuized.build(self.x.scores, self.x.probabilities, self.bandwidth_x),
            _Continuized.build(self.y.scores, self.y.probabilities, self.bandwidth_y),
        )

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
        x_continuized, y_continuized = self._continuized()
        flat = np.atleast_1d(points).ravel()
        equated = y_continuized.quantile(x_continuized.cdf(flat))
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
            Columns ``form`` (``"x"`` or ``"y"``), ``score``, ``probability`` (discrete),
            ``cdf`` (:math:`F_h` or :math:`G_h`) and ``density``.
        """
        frames = []
        for form, continuized in zip(("x", "y"), self._continuized(), strict=True):
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

        (von Davier et al., 2004, eq. 4.32).

        Returns
        -------
        pandas.Series
            Indexed by the moment order 1..10.
        """
        orders = np.arange(1, 11)
        equated_moments = np.array([self.x.probabilities @ self.equated**p for p in orders])
        target_moments = np.array([self.y.probabilities @ self.y.scores**p for p in orders])
        return pd.Series(
            100.0 * (equated_moments - target_moments) / target_moments,
            index=pd.Index(orders, name="moment"),
            name="percent_relative_error",
        )

    def __repr__(self) -> str:
        """Return a one-line summary."""
        return (
            f"KernelEquatingResult(design={self.design!r}, bandwidth_x={self.bandwidth_x:.4f}, "
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


def _resolve_bandwidth(
    bandwidth: float | None, scores: FloatArray, probabilities: FloatArray, weight: float, name: str
) -> float:
    if bandwidth is None:
        return _penalty_bandwidth(scores, probabilities, weight)
    if not (np.isfinite(bandwidth) and bandwidth > 0):
        raise SpecificationError(f"{name} must be a positive number, got {bandwidth}.")
    return float(bandwidth)


def kernel_equate(
    x: ScoreDistribution | LoglinearResult,
    y: ScoreDistribution | LoglinearResult,
    *,
    bandwidth_x: float | None = None,
    bandwidth_y: float | None = None,
    penalty_weight: float = 1.0,
) -> KernelEquatingResult:
    r"""Kernel equating of form X to form Y for equivalent groups (Gaussian kernel).

    The discrete distributions with score probabilities :math:`r_j` (X) and :math:`s_k` (Y) are
    continuized (von Davier et al., 2004, eq. 4.5-4.7),

    .. math::

        F_{h_X}(x) = \sum_j r_j\, \Phi\!\left(\frac{x - a_X x_j - (1 - a_X)\mu_X}{a_X h_X}\right),
        \qquad a_X^2 = \frac{\sigma_X^2}{\sigma_X^2 + h_X^2},

    and likewise :math:`G_{h_Y}`, and the equating function is

    .. math::

        e_Y(x) = G_{h_Y}^{-1}\bigl(F_{h_X}(x)\bigr) .

    The standard error of equating follows from the delta method (eq. 5.1-5.22):

    .. math::

        \mathrm{SEE}_Y(x) = \frac{1}{G_{h_Y}'(e_Y(x))}
        \sqrt{\tfrac{1}{N}\Bigl\lVert \tfrac{\partial F_{h_X}(x)}{\partial r}^{\!\top} C_r
                \Bigr\rVert^2
              + \tfrac{1}{M}\Bigl\lVert \tfrac{\partial G_{h_Y}(e_Y(x))}{\partial s}^{\!\top} C_s
                \Bigr\rVert^2},

    where :math:`N` and :math:`M` are the sample sizes and :math:`C_r C_r^\top / N` is the
    covariance of the estimated score probabilities: that of the log-linear model for a
    presmoothed distribution (:attr:`LoglinearResult.c_matrix`), the multinomial covariance
    otherwise.

    Parameters
    ----------
    x, y : ScoreDistribution or LoglinearResult
        Score distributions of forms X and Y from independent, equivalent groups. Pass the result
        of :func:`presmooth` to equate presmoothed distributions, as recommended by von Davier et
        al.; a plain ``ScoreDistribution`` is used as observed.
    bandwidth_x, bandwidth_y : float, optional
        Bandwidths :math:`h_X` and :math:`h_Y`. By default each is chosen by
        :func:`penalty_bandwidth`. Large bandwidths (e.g. 1000 standard deviations) give the
        linear equating function.
    penalty_weight : float, default 1.0
        Weight :math:`K` of the second penalty term in the bandwidth selection.

    Returns
    -------
    KernelEquatingResult
        Equated scores, standard errors, bandwidths and the percent relative error.

    Raises
    ------
    SpecificationError
        If an input is not a univariate distribution, has no variance, or a bandwidth is not
        positive.

    References
    ----------
    von Davier, A. A., Holland, P. W., & Thayer, D. T. (2004). *The kernel method of test
    equating*. Springer.
    """
    if not penalty_weight >= 0:
        raise SpecificationError(f"penalty_weight must be non-negative, got {penalty_weight}.")
    x_scores, r, c_r, x_distribution = _unpack(x, "x")
    y_scores, s, c_s, y_distribution = _unpack(y, "y")
    weight = float(penalty_weight)
    h_x = _resolve_bandwidth(bandwidth_x, x_scores, r, weight, "bandwidth_x")
    h_y = _resolve_bandwidth(bandwidth_y, y_scores, s, weight, "bandwidth_y")

    F = _Continuized.build(x_scores, r, h_x)
    G = _Continuized.build(y_scores, s, h_y)
    equated = G.quantile(F.cdf(x_scores))

    # Delta method: rows are the possible X scores.
    gradient_r = F.cdf_gradient(x_scores) @ c_r
    gradient_s = G.cdf_gradient(equated) @ c_s
    variance = (gradient_r**2).sum(axis=1) / x_distribution.n + (gradient_s**2).sum(
        axis=1
    ) / y_distribution.n
    see = np.sqrt(variance) / G.density(equated)

    return KernelEquatingResult(
        design="equivalent_groups",
        x=x_distribution,
        y=y_distribution,
        scores=x_scores,
        equated=_freeze(equated),
        see=_freeze(see),
        bandwidth_x=h_x,
        bandwidth_y=h_y,
        penalty_weight=weight,
        x_presmoothed=isinstance(x, LoglinearResult),
        y_presmoothed=isinstance(y, LoglinearResult),
    )
