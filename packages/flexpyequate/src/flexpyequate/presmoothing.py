"""Polynomial log-linear presmoothing of score distributions (Holland & Thayer, 2000)."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import overload

import numpy as np
import pandas as pd
from scipy.special import gammaln

from flexpyequate.distributions import (
    BivariateScoreDistribution,
    FloatArray,
    ScoreDistribution,
    _freeze,
)
from flexpyequate.errors import ConvergenceWarning, SpecificationError

__all__ = ["LoglinearResult", "presmooth"]


@dataclass(frozen=True, eq=False)
class LoglinearResult:
    r"""A fitted polynomial log-linear model for a score distribution.

    Attributes
    ----------
    distribution : ScoreDistribution or BivariateScoreDistribution
        The presmoothed distribution: the fitted frequencies on the same possible scores. Its
        total equals the observed total.
    observed : ScoreDistribution or BivariateScoreDistribution
        The distribution that was smoothed.
    terms : tuple of str
        Names of the polynomial terms, e.g. ``("x^1", "x^2")`` or ``("x^1", "y^1", "x^1*y^1")``.
    design_matrix : numpy.ndarray of shape (n_cells, n_terms)
        The polynomial terms evaluated at every cell (no intercept column). Scores are rescaled to
        :math:`[-1, 1]` before taking powers; this changes neither the model nor the fit. For a
        bivariate distribution the cells are in row-major order (X varies slowest).
    log_likelihood : float
        Poisson log-likelihood at the fitted frequencies,
        :math:`\sum_j [n_j \log m_j - m_j - \log n_j!]`.
    n_iterations : int
        Number of Newton iterations.
    converged : bool
        Whether the score equations were solved to the requested tolerance.
    """

    distribution: ScoreDistribution | BivariateScoreDistribution
    observed: ScoreDistribution | BivariateScoreDistribution
    terms: tuple[str, ...]
    design_matrix: FloatArray
    log_likelihood: float
    n_iterations: int
    converged: bool

    @property
    def n_parameters(self) -> int:
        """Number of free parameters: the polynomial terms plus the intercept."""
        return len(self.terms) + 1

    @property
    def degrees_of_freedom(self) -> int:
        """Residual degrees of freedom: number of cells minus number of parameters."""
        return int(self.design_matrix.shape[0]) - self.n_parameters

    @property
    def fit_statistics(self) -> pd.Series:
        r"""Goodness-of-fit statistics of the model.

        With observed frequencies :math:`n_j` and fitted frequencies :math:`m_j`:

        .. math::

            G^2 &= 2 \sum_j n_j \log(n_j / m_j) \\
            \chi^2_{\mathrm{Pearson}} &= \sum_j (n_j - m_j)^2 / m_j \\
            \chi^2_{\mathrm{FT}} &= \sum_j \bigl(\sqrt{n_j} + \sqrt{n_j + 1}
                                    - \sqrt{4 m_j + 1}\bigr)^2 \\
            \mathrm{AIC} &= -2 \ell + 2 p

        where :math:`\ell` is the Poisson log-likelihood and :math:`p` the number of parameters
        (Holland & Thayer, 2000). ``AIC`` equals R ``glm``'s for the same Poisson model.

        Returns
        -------
        pandas.Series
            ``likelihood_ratio``, ``pearson``, ``freeman_tukey``, ``degrees_of_freedom``,
            ``n_parameters``, ``log_likelihood`` and ``aic``.
        """
        observed = np.ravel(self.observed.frequencies)
        fitted = np.ravel(self.distribution.frequencies)
        positive = observed > 0
        likelihood_ratio = 2.0 * float(
            np.sum(observed[positive] * np.log(observed[positive] / fitted[positive]))
            - np.sum(observed - fitted)
        )
        pearson = float(np.sum((observed - fitted) ** 2 / fitted))
        freeman_tukey = float(
            np.sum((np.sqrt(observed) + np.sqrt(observed + 1) - np.sqrt(4 * fitted + 1)) ** 2)
        )
        return pd.Series(
            {
                "likelihood_ratio": likelihood_ratio,
                "pearson": pearson,
                "freeman_tukey": freeman_tukey,
                "degrees_of_freedom": float(self.degrees_of_freedom),
                "n_parameters": float(self.n_parameters),
                "log_likelihood": self.log_likelihood,
                "aic": -2.0 * self.log_likelihood + 2.0 * self.n_parameters,
            }
        )

    @property
    def c_matrix(self) -> FloatArray:
        r"""The C-matrix of the fitted score probabilities (Holland & Thayer, 1987).

        With :math:`r` the fitted probabilities, :math:`B` the design matrix and the QR
        factorization :math:`(D_{\sqrt r} - \sqrt r\, r^\top) B = QR`,

        .. math::

            C = D_{\sqrt r}\, Q, \qquad
            \operatorname{Cov}(\hat r) \approx \tfrac{1}{N}\, C C^\top .

        Returns
        -------
        numpy.ndarray of shape (n_cells, n_terms)
            Any matrix with the same :math:`C C^\top` serves equally; the columns themselves are
            not unique.
        """
        probabilities = np.ravel(self.distribution.probabilities)
        return loglinear_c_matrix(probabilities, self.design_matrix)

    def __repr__(self) -> str:
        """Return a one-line summary."""
        stats = self.fit_statistics
        return (
            f"LoglinearResult(terms={len(self.terms)}, "
            f"likelihood_ratio={stats['likelihood_ratio']:.3f}, "
            f"degrees_of_freedom={self.degrees_of_freedom}, aic={stats['aic']:.3f})"
        )


def loglinear_c_matrix(probabilities: FloatArray, design_matrix: FloatArray) -> FloatArray:
    r"""C-matrix :math:`D_{\sqrt r} Q` with :math:`(D_{\sqrt r} - \sqrt r\, r^\top) B = QR`."""
    root = np.sqrt(probabilities)
    projected = root[:, None] * design_matrix - np.outer(root, probabilities @ design_matrix)
    q_factor, _ = np.linalg.qr(projected)
    return np.asarray(root[:, None] * q_factor, dtype=np.float64)


def saturated_c_matrix(probabilities: FloatArray) -> FloatArray:
    r"""C-matrix without presmoothing: :math:`D_{\sqrt r} - r \sqrt r^\top`.

    :math:`C C^\top = D_r - r r^\top`, the multinomial covariance of the sample proportions times
    :math:`N`.
    """
    root = np.sqrt(probabilities)
    return np.asarray(np.diag(root) - np.outer(probabilities, root), dtype=np.float64)


def _rescale(scores: FloatArray) -> FloatArray:
    """Map the possible scores linearly onto [-1, 1]."""
    middle = (scores[0] + scores[-1]) / 2
    half_range = (scores[-1] - scores[0]) / 2
    return np.asarray((scores - middle) / half_range, dtype=np.float64)


def _fit_poisson(
    frequencies: FloatArray, design_matrix: FloatArray, max_iterations: int, tolerance: float
) -> tuple[FloatArray, int, bool]:
    r"""Maximize the Poisson log-likelihood of :math:`\log m = \beta_0 + B\beta` by Newton's method.

    The columns of :math:`[1, B]` are orthonormalized first; the fitted frequencies depend only on
    their span. Convergence: every score equation, :math:`|q^\top (n - m)| / N` for an orthonormal
    column :math:`q`, is below ``tolerance``.
    """
    total = float(frequencies.sum())
    full = np.column_stack([np.ones(frequencies.size), design_matrix])
    basis, triangular = np.linalg.qr(full)
    diagonal = np.abs(np.diag(triangular))
    if np.any(diagonal < 1e-10 * diagonal.max()):
        raise SpecificationError(
            "The polynomial terms are linearly dependent on these scores; lower the degree."
        )

    def log_likelihood(coefficients: FloatArray) -> float:
        linear_predictor = basis @ coefficients
        return float(frequencies @ linear_predictor - np.exp(linear_predictor).sum())

    # Start from the least-squares fit to the log of the (slightly flattened) frequencies.
    coefficients = basis.T @ np.log(frequencies + 0.5)
    current = log_likelihood(coefficients)
    converged = False
    n_iterations = 0
    while True:
        fitted = np.exp(basis @ coefficients)
        gradient = basis.T @ (frequencies - fitted)
        if np.max(np.abs(gradient)) / total < tolerance:
            converged = True
            break
        if n_iterations == max_iterations:
            break
        information = basis.T @ (fitted[:, None] * basis)
        step = np.linalg.solve(information, gradient)
        # Step halving keeps the log-likelihood from decreasing.
        scale = 1.0
        while True:
            candidate = coefficients + scale * step
            with np.errstate(over="ignore"):
                value = log_likelihood(candidate)
            if (np.isfinite(value) and value >= current - 1e-12 * abs(current)) or scale < 1e-10:
                break
            scale /= 2
        coefficients, current = candidate, value
        n_iterations += 1
    fitted = np.exp(basis @ coefficients)
    return np.asarray(fitted, dtype=np.float64), n_iterations, converged


def _poisson_log_likelihood(observed: FloatArray, fitted: FloatArray) -> float:
    return float(np.sum(observed * np.log(fitted) - fitted - gammaln(observed + 1)))


@overload
def presmooth(
    distribution: ScoreDistribution,
    *,
    degree: int = ...,
    max_iterations: int = ...,
    tolerance: float = ...,
) -> LoglinearResult: ...
@overload
def presmooth(
    distribution: BivariateScoreDistribution,
    *,
    degree: tuple[int, int] = ...,
    cross_terms: Sequence[tuple[int, int]] = ...,
    max_iterations: int = ...,
    tolerance: float = ...,
) -> LoglinearResult: ...
def presmooth(
    distribution: ScoreDistribution | BivariateScoreDistribution,
    *,
    degree: int | tuple[int, int] | None = None,
    cross_terms: Sequence[tuple[int, int]] = ((1, 1),),
    max_iterations: int = 100,
    tolerance: float = 1e-12,
) -> LoglinearResult:
    r"""Presmooth a score distribution with a polynomial log-linear model.

    For one score with frequencies :math:`n_j` (Holland & Thayer, 2000),

    .. math::

        \log m_j = \beta_0 + \sum_{i=1}^{C} \beta_i\, x_j^{\,i},

    and for the joint distribution of two scores

    .. math::

        \log m_{jk} = \beta_0 + \sum_{i=1}^{C_X} \beta_{Xi}\, x_j^{\,i}
                      + \sum_{i=1}^{C_Y} \beta_{Yi}\, y_k^{\,i}
                      + \sum_{(a, b)} \beta_{ab}\, x_j^{\,a} y_k^{\,b},

    fitted by maximum likelihood (Newton's method). The fitted distribution has the observed
    total and the same first :math:`C` moments as the observed one; in the bivariate model it
    also preserves the cross moments :math:`E[X^a Y^b]` of the included cross terms.

    Parameters
    ----------
    distribution : ScoreDistribution or BivariateScoreDistribution
        The observed distribution.
    degree : int or tuple of int, optional
        Polynomial degree :math:`C` (default 6) for one score, or :math:`(C_X, C_Y)` (default
        ``(6, 6)``) for a bivariate distribution. It is R ``equate``'s ``degrees`` and the degree
        of the polynomial in a ``glm`` formula.
    cross_terms : sequence of tuple of int, default ((1, 1),)
        Bivariate only: the powers :math:`(a, b)` of the cross terms :math:`x^a y^b`. Pass an
        empty sequence for a model in which the two scores are independent.
    max_iterations : int, default 100
        Maximum number of Newton iterations.
    tolerance : float, default 1e-12
        Convergence tolerance on the score equations (differences between observed and fitted
        moments of the orthonormalized terms, as proportions).

    Returns
    -------
    LoglinearResult
        The presmoothed distribution and the fit statistics.

    Raises
    ------
    SpecificationError
        If a degree is not a positive integer, exceeds what the number of possible scores
        supports, or the cross terms are invalid.

    Warns
    -----
    ConvergenceWarning
        If Newton's method does not converge within ``max_iterations``.

    References
    ----------
    Holland, P. W., & Thayer, D. T. (2000). Univariate and bivariate loglinear models for discrete
    test score distributions. *Journal of Educational and Behavioral Statistics, 25*, 133-183.
    """
    if max_iterations < 1:
        raise SpecificationError("max_iterations must be at least 1.")
    if not tolerance > 0:
        raise SpecificationError("tolerance must be positive.")

    if isinstance(distribution, ScoreDistribution):
        if degree is None:
            degree = 6
        if isinstance(degree, tuple) or not _is_positive_int(degree):
            raise SpecificationError(
                f"degree must be a positive integer for one score, got {degree!r}."
            )
        if degree > distribution.n_scores - 2:
            raise SpecificationError(
                f"degree {degree} is too high for {distribution.n_scores} possible scores "
                f"(at most {distribution.n_scores - 2})."
            )
        scaled = _rescale(distribution.scores)
        design_matrix = np.column_stack([scaled**power for power in range(1, degree + 1)])
        terms = tuple(f"x^{power}" for power in range(1, degree + 1))
        observed = np.asarray(distribution.frequencies)
    elif isinstance(distribution, BivariateScoreDistribution):
        if degree is None:
            degree = (6, 6)
        if (
            not isinstance(degree, tuple)
            or len(degree) != 2
            or not all(_is_positive_int(value) for value in degree)
        ):
            raise SpecificationError(
                f"degree must be a pair of positive integers for two scores, got {degree!r}."
            )
        sizes = distribution.frequencies.shape
        for value, size, label in zip(degree, sizes, ("x", "y"), strict=True):
            if value > size - 1:
                raise SpecificationError(
                    f"degree {value} is too high for {size} possible {label} scores "
                    f"(at most {size - 1})."
                )
        pairs = [tuple(pair) for pair in cross_terms]
        for pair in pairs:
            if len(pair) != 2 or not all(_is_positive_int(value) for value in pair):
                raise SpecificationError(
                    f"cross_terms must be pairs of positive integers, got {pair!r}."
                )
        if len(set(pairs)) != len(pairs):
            raise SpecificationError("cross_terms contains a repeated pair.")
        x_scaled, y_scaled = np.meshgrid(
            _rescale(distribution.x_scores), _rescale(distribution.y_scores), indexing="ij"
        )
        x_flat, y_flat = x_scaled.ravel(), y_scaled.ravel()
        columns = [x_flat**power for power in range(1, degree[0] + 1)]
        columns += [y_flat**power for power in range(1, degree[1] + 1)]
        columns += [x_flat**a * y_flat**b for a, b in pairs]
        design_matrix = np.column_stack(columns)
        terms = (
            tuple(f"x^{power}" for power in range(1, degree[0] + 1))
            + tuple(f"y^{power}" for power in range(1, degree[1] + 1))
            + tuple(f"x^{a}*y^{b}" for a, b in pairs)
        )
        observed = np.ravel(distribution.frequencies)
        if design_matrix.shape[1] > observed.size - 2:
            raise SpecificationError("The model has as many parameters as there are cells.")
    else:
        raise SpecificationError(
            "presmooth() needs a ScoreDistribution or a BivariateScoreDistribution."
        )

    fitted, n_iterations, converged = _fit_poisson(
        observed, design_matrix, max_iterations, tolerance
    )
    if not converged:
        warnings.warn(
            f"Log-linear presmoothing did not converge in {max_iterations} iterations.",
            ConvergenceWarning,
            stacklevel=2,
        )
    smoothed: ScoreDistribution | BivariateScoreDistribution
    if isinstance(distribution, ScoreDistribution):
        smoothed = ScoreDistribution(scores=distribution.scores, frequencies=fitted)
    else:
        smoothed = BivariateScoreDistribution(
            x_scores=distribution.x_scores,
            y_scores=distribution.y_scores,
            frequencies=fitted.reshape(distribution.frequencies.shape),
        )
    return LoglinearResult(
        distribution=smoothed,
        observed=distribution,
        terms=terms,
        design_matrix=_freeze(design_matrix),
        log_likelihood=_poisson_log_likelihood(observed, fitted),
        n_iterations=n_iterations,
        converged=converged,
    )


def _is_positive_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool) and int(value) >= 1
