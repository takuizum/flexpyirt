"""Unit tests for polynomial log-linear presmoothing."""

from __future__ import annotations

import numpy as np
import pytest

from flexpyequate import BivariateScoreDistribution, ScoreDistribution, presmooth
from flexpyequate.errors import ConvergenceWarning, SpecificationError
from flexpyequate.presmoothing import saturated_c_matrix


def _univariate(seed: int = 0, n: int = 1500) -> ScoreDistribution:
    rng = np.random.default_rng(seed)
    values = rng.binomial(30, rng.beta(5, 4, size=n))
    return ScoreDistribution.from_scores(values, score_range=(0, 30))


def _bivariate(seed: int = 1, n: int = 1200) -> BivariateScoreDistribution:
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=n)
    x_values = rng.binomial(12, 1 / (1 + np.exp(-theta)))
    y_values = rng.binomial(15, 1 / (1 + np.exp(-0.9 * theta + 0.2)))
    return BivariateScoreDistribution.from_scores(
        x_values, y_values, x_range=(0, 12), y_range=(0, 15)
    )


@pytest.mark.parametrize("degree", [1, 2, 3, 5, 8])
def test_fitted_distribution_preserves_the_first_moments(degree: int) -> None:
    observed = _univariate()
    result = presmooth(observed, degree=degree)
    smoothed = result.distribution
    assert isinstance(smoothed, ScoreDistribution)
    assert result.converged
    assert smoothed.n == pytest.approx(observed.n, rel=1e-12)
    for power in range(1, degree + 1):
        raw_observed = observed.probabilities @ observed.scores**power
        raw_smoothed = smoothed.probabilities @ smoothed.scores**power
        assert raw_smoothed == pytest.approx(raw_observed, rel=1e-9)
    # The next moment is not preserved (the model is not saturated).
    higher = degree + 1
    assert smoothed.probabilities @ smoothed.scores**higher != pytest.approx(
        observed.probabilities @ observed.scores**higher, rel=1e-9
    )
    assert np.all(smoothed.frequencies > 0)
    assert result.terms == tuple(f"x^{power}" for power in range(1, degree + 1))
    assert result.n_parameters == degree + 1
    assert result.degrees_of_freedom == observed.n_scores - degree - 1


def test_fit_does_not_depend_on_the_location_and_unit_of_the_scores() -> None:
    observed = _univariate()
    moved = ScoreDistribution(scores=100 + 2.5 * observed.scores, frequencies=observed.frequencies)
    np.testing.assert_allclose(
        presmooth(moved, degree=5).distribution.frequencies,
        presmooth(observed, degree=5).distribution.frequencies,
        rtol=1e-9,
    )


def test_fit_statistics_and_nested_models() -> None:
    observed = _univariate()
    low = presmooth(observed, degree=2).fit_statistics
    high = presmooth(observed, degree=6).fit_statistics
    assert high["likelihood_ratio"] < low["likelihood_ratio"]
    assert high["log_likelihood"] > low["log_likelihood"]
    assert low["degrees_of_freedom"] - high["degrees_of_freedom"] == 4
    assert high["aic"] == pytest.approx(-2 * high["log_likelihood"] + 2 * 7)
    assert set(low.index) == {
        "likelihood_ratio",
        "pearson",
        "freeman_tukey",
        "degrees_of_freedom",
        "n_parameters",
        "log_likelihood",
        "aic",
    }


def test_log_linear_data_are_reproduced_exactly() -> None:
    scores = np.arange(11.0)
    expected = 500 * np.exp(-0.5 * ((scores - 4.5) / 2.0) ** 2)
    observed = ScoreDistribution(scores=scores, frequencies=expected)
    result = presmooth(observed, degree=2)
    np.testing.assert_allclose(result.distribution.frequencies, expected, rtol=1e-10)
    assert result.fit_statistics["likelihood_ratio"] == pytest.approx(0, abs=1e-8)


def test_bivariate_preserves_marginal_and_cross_moments() -> None:
    observed = _bivariate()
    result = presmooth(observed, degree=(4, 3), cross_terms=[(1, 1), (2, 1)])
    smoothed = result.distribution
    assert isinstance(smoothed, BivariateScoreDistribution)
    assert result.converged
    assert result.terms == ("x^1", "x^2", "x^3", "x^4", "y^1", "y^2", "y^3", "x^1*y^1", "x^2*y^1")
    assert smoothed.n == pytest.approx(observed.n, rel=1e-12)
    for power in range(1, 5):
        assert smoothed.x.probabilities @ smoothed.x_scores**power == pytest.approx(
            observed.x.probabilities @ observed.x_scores**power, rel=1e-9
        )
    for power in range(1, 4):
        assert smoothed.y.probabilities @ smoothed.y_scores**power == pytest.approx(
            observed.y.probabilities @ observed.y_scores**power, rel=1e-9
        )
    assert smoothed.covariance() == pytest.approx(observed.covariance(), rel=1e-9)
    assert smoothed.correlation == pytest.approx(observed.correlation, rel=1e-9)
    x2y_observed = observed.x_scores**2 @ observed.probabilities @ observed.y_scores
    x2y_smoothed = smoothed.x_scores**2 @ smoothed.probabilities @ smoothed.y_scores
    assert x2y_smoothed == pytest.approx(x2y_observed, rel=1e-9)


def test_bivariate_without_cross_terms_is_independence() -> None:
    observed = _bivariate()
    smoothed = presmooth(observed, degree=(3, 3), cross_terms=()).distribution
    assert isinstance(smoothed, BivariateScoreDistribution)
    np.testing.assert_allclose(
        smoothed.probabilities,
        np.outer(smoothed.x.probabilities, smoothed.y.probabilities),
        atol=1e-12,
    )
    # The margins are the univariate fits.
    np.testing.assert_allclose(
        smoothed.x.frequencies,
        presmooth(observed.x, degree=3).distribution.frequencies,
        rtol=1e-8,
    )


def test_c_matrix_gives_the_asymptotic_covariance() -> None:
    result = presmooth(_univariate(), degree=4)
    probabilities = result.distribution.probabilities
    design = result.design_matrix
    c_matrix = result.c_matrix
    assert c_matrix.shape == (probabilities.size, 4)
    # Sigma = W B (B' W B)^{-1} B' W with W = D_r - r r' (Holland & Thayer, 1987).
    weight = np.diag(probabilities) - np.outer(probabilities, probabilities)
    middle = np.linalg.inv(design.T @ weight @ design)
    expected = weight @ design @ middle @ design.T @ weight
    np.testing.assert_allclose(c_matrix @ c_matrix.T, expected, atol=1e-12)
    # Probabilities sum to one, so every column of C sums to zero.
    np.testing.assert_allclose(c_matrix.sum(axis=0), 0, atol=1e-12)


def test_c_matrix_matches_a_parametric_bootstrap() -> None:
    result = presmooth(_univariate(), degree=3)
    smoothed = result.distribution
    assert isinstance(smoothed, ScoreDistribution)
    probabilities = smoothed.probabilities
    n = 4000
    rng = np.random.default_rng(11)
    draws = []
    for _ in range(2000):
        sample = ScoreDistribution(
            scores=smoothed.scores,
            frequencies=rng.multinomial(n, probabilities).astype(np.float64),
        )
        draws.append(presmooth(sample, degree=3).distribution.probabilities)
    empirical = np.cov(np.array(draws).T) * n
    c_matrix = result.c_matrix
    theoretical = c_matrix @ c_matrix.T
    scale = np.sqrt(np.outer(np.diag(theoretical), np.diag(theoretical)))
    assert np.max(np.abs(empirical - theoretical) / scale) < 0.15
    np.testing.assert_allclose(np.diag(empirical), np.diag(theoretical), rtol=0.15)


def test_saturated_c_matrix_is_the_multinomial_covariance() -> None:
    probabilities = _univariate().probabilities
    c_matrix = saturated_c_matrix(probabilities)
    np.testing.assert_allclose(
        c_matrix @ c_matrix.T,
        np.diag(probabilities) - np.outer(probabilities, probabilities),
        atol=1e-14,
    )


def test_specification_errors() -> None:
    observed = _univariate()
    joint = _bivariate()
    with pytest.raises(SpecificationError, match="positive integer"):
        presmooth(observed, degree=0)
    with pytest.raises(SpecificationError, match="positive integer"):
        presmooth(observed, degree=(2, 2))  # type: ignore[call-overload]
    with pytest.raises(SpecificationError, match="too high"):
        presmooth(observed, degree=30)
    with pytest.raises(SpecificationError, match="pair of positive integers"):
        presmooth(joint, degree=3)  # type: ignore[call-overload]
    with pytest.raises(SpecificationError, match="too high"):
        presmooth(joint, degree=(13, 3))
    with pytest.raises(SpecificationError, match="cross_terms"):
        presmooth(joint, degree=(2, 2), cross_terms=[(0, 1)])
    with pytest.raises(SpecificationError, match="repeated"):
        presmooth(joint, degree=(2, 2), cross_terms=[(1, 1), (1, 1)])
    with pytest.raises(SpecificationError, match="tolerance"):
        presmooth(observed, degree=2, tolerance=0)
    with pytest.raises(SpecificationError, match="ScoreDistribution"):
        presmooth([1, 2, 3], degree=2)  # type: ignore[call-overload]


def test_convergence_warning() -> None:
    with pytest.warns(ConvergenceWarning, match="did not converge"):
        result = presmooth(_univariate(), degree=6, max_iterations=1)
    assert not result.converged
    assert result.n_iterations == 1
