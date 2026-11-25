"""Unit tests for kernel equating (equivalent groups, Gaussian kernel)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from flexpyequate import (
    BivariateScoreDistribution,
    ScoreDistribution,
    equate,
    kernel_equate,
    penalty_bandwidth,
    presmooth,
)
from flexpyequate.errors import SpecificationError
from flexpyequate.kernel import _Continuized, _penalty


def _forms(n_x: int = 2000, n_y: int = 2200) -> tuple[ScoreDistribution, ScoreDistribution]:
    rng = np.random.default_rng(42)
    x_values = rng.binomial(25, rng.beta(5, 4, size=n_x))
    y_values = rng.binomial(30, rng.beta(4, 4, size=n_y))
    return (
        ScoreDistribution.from_scores(x_values, score_range=(0, 25)),
        ScoreDistribution.from_scores(y_values, score_range=(0, 30)),
    )


def _unnormalized(scores: np.ndarray, weights: np.ndarray, bandwidth: float) -> _Continuized:
    """Continuization as a function of free weights: mean and variance are sums over them."""
    mean = float(weights @ scores)
    variance = float(weights @ (scores - mean) ** 2)
    return _Continuized(scores, weights, bandwidth, mean, variance)


@pytest.mark.parametrize("bandwidth", [0.3, 0.6, 2.0, 50.0])
def test_continuization_keeps_mean_and_variance(bandwidth: float) -> None:
    x, _ = _forms()
    continuized = _Continuized.build(x.scores, x.probabilities, bandwidth)
    low, high = -60.0, 90.0
    points = list(x.scores)
    total = quad(
        lambda v: continuized.density(np.array([v]))[0], low, high, points=points, limit=500
    )[0]
    mean = quad(
        lambda v: v * continuized.density(np.array([v]))[0], low, high, points=points, limit=500
    )[0]
    second = quad(
        lambda v: (v - x.mean) ** 2 * continuized.density(np.array([v]))[0],
        low,
        high,
        points=points,
        limit=500,
    )[0]
    assert total == pytest.approx(1.0, abs=1e-8)
    assert mean == pytest.approx(x.mean, abs=1e-7)
    assert second == pytest.approx(x.variance(), abs=1e-6)


def test_density_is_the_derivative_of_the_cdf() -> None:
    x, _ = _forms()
    continuized = _Continuized.build(x.scores, x.probabilities, 0.6)
    points = np.linspace(-1.0, 26.0, 55)
    step = 1e-5
    numeric = (continuized.cdf(points + step) - continuized.cdf(points - step)) / (2 * step)
    np.testing.assert_allclose(continuized.density(points), numeric, rtol=1e-6, atol=1e-10)
    numeric = (continuized.density(points + step) - continuized.density(points - step)) / (2 * step)
    np.testing.assert_allclose(
        continuized.density_derivative(points), numeric, rtol=1e-5, atol=1e-9
    )


def test_quantile_inverts_the_cdf() -> None:
    x, _ = _forms()
    continuized = _Continuized.build(x.scores, x.probabilities, 0.6)
    points = np.linspace(-1.5, 26.5, 40)
    np.testing.assert_allclose(continuized.quantile(continuized.cdf(points)), points, atol=1e-9)


@pytest.mark.parametrize("bandwidth", [0.4, 0.7, 3.0])
def test_cdf_gradient_matches_finite_differences(bandwidth: float) -> None:
    x, _ = _forms()
    scores, weights = x.scores, x.probabilities
    points = np.array([-0.5, 3.0, 11.2, 14.0, 19.7, 25.0])
    analytic = _Continuized.build(scores, weights, bandwidth).cdf_gradient(points)
    step = 1e-6
    numeric = np.empty_like(analytic)
    for j in range(scores.size):
        shift = np.zeros(scores.size)
        shift[j] = step
        up = _unnormalized(scores, weights + shift, bandwidth).cdf(points)
        down = _unnormalized(scores, weights - shift, bandwidth).cdf(points)
        numeric[:, j] = (up - down) / (2 * step)
    np.testing.assert_allclose(analytic, numeric, rtol=0, atol=2e-8)


def test_large_bandwidths_give_linear_equating() -> None:
    x, y = _forms()
    result = kernel_equate(x, y, bandwidth_x=1e4 * x.sd(), bandwidth_y=1e4 * y.sd())
    linear = equate(x, y, method="linear")
    np.testing.assert_allclose(result.equated, linear.equated, atol=1e-6)


def test_small_bandwidths_approach_equipercentile_ranks() -> None:
    # With little smoothing the continuized CDF at x + 1/2 is close to the discrete F(x).
    x, _ = _forms()
    continuized = _Continuized.build(x.scores, x.probabilities, 0.1)
    np.testing.assert_allclose(continuized.cdf(x.scores + 0.5), x.cumulative, atol=1e-4)


def test_equating_is_symmetric() -> None:
    x, y = _forms()
    forward = kernel_equate(x, y, bandwidth_x=0.6, bandwidth_y=0.65)
    backward = kernel_equate(y, x, bandwidth_x=0.65, bandwidth_y=0.6)
    np.testing.assert_allclose(backward.transform(forward.equated), x.scores, atol=1e-8)
    np.testing.assert_allclose(forward.transform(x.scores), forward.equated, atol=1e-12)
    assert forward.transform(12.5).shape == ()
    assert forward.transform([[1.0, 2.0], [3.0, 4.0]]).shape == (2, 2)


def test_equivariance_under_linear_score_transformations() -> None:
    x, y = _forms()
    base = kernel_equate(presmooth(x, degree=4), presmooth(y, degree=4))
    x_moved = ScoreDistribution(scores=100 + 0.5 * x.scores, frequencies=x.frequencies)
    y_moved = ScoreDistribution(scores=-3 + 2.0 * y.scores, frequencies=y.frequencies)
    moved = kernel_equate(presmooth(x_moved, degree=4), presmooth(y_moved, degree=4))
    assert moved.bandwidth_x == pytest.approx(0.5 * base.bandwidth_x, rel=1e-6)
    assert moved.bandwidth_y == pytest.approx(2.0 * base.bandwidth_y, rel=1e-6)
    np.testing.assert_allclose(moved.equated, -3 + 2.0 * base.equated, atol=1e-6)
    np.testing.assert_allclose(moved.see, 2.0 * base.see, rtol=1e-6)


def test_penalty_second_term_counts_u_shapes() -> None:
    # A bimodal distribution: with a small bandwidth the density dips at the empty score.
    scores = np.arange(7.0)
    probabilities = np.array([0.05, 0.25, 0.2, 0.0, 0.2, 0.25, 0.05])
    first = _penalty(0.3, scores, probabilities, 0.0)
    both = _penalty(0.3, scores, probabilities, 1.0)
    assert both - first == pytest.approx(1.0)
    assert _penalty(0.3, scores, probabilities, 2.5) - first == pytest.approx(2.5)
    distribution = ScoreDistribution(scores=scores, frequencies=probabilities * 100)
    unpenalized = penalty_bandwidth(distribution, penalty_weight=0)
    penalized = penalty_bandwidth(distribution, penalty_weight=1)
    assert penalized > unpenalized
    grid = np.geomspace(0.05, 10, 2000)
    for weight, found in ((0.0, unpenalized), (1.0, penalized)):
        best = min(_penalty(h, scores, probabilities, weight) for h in grid)
        assert _penalty(found, scores, probabilities, weight) <= best + 1e-12


def test_default_bandwidths_come_from_the_penalty() -> None:
    x, y = _forms()
    smooth_x, smooth_y = presmooth(x, degree=5), presmooth(y, degree=5)
    result = kernel_equate(smooth_x, smooth_y, penalty_weight=0)
    assert result.bandwidth_x == pytest.approx(penalty_bandwidth(smooth_x, penalty_weight=0))
    assert result.bandwidth_y == pytest.approx(penalty_bandwidth(smooth_y, penalty_weight=0))
    assert 0.3 < result.bandwidth_x < 1.0
    assert result.x_presmoothed and result.y_presmoothed
    assert result.penalty_weight == 0


def test_see_matches_a_parametric_bootstrap() -> None:
    x, y = _forms()
    smooth_x, smooth_y = presmooth(x, degree=4), presmooth(y, degree=4)
    result = kernel_equate(smooth_x, smooth_y, bandwidth_x=0.6, bandwidth_y=0.6)
    r, s = result.x.probabilities, result.y.probabilities
    rng = np.random.default_rng(2026)
    draws = []
    for _ in range(600):
        x_sample = ScoreDistribution(
            scores=x.scores, frequencies=rng.multinomial(int(x.n), r).astype(np.float64)
        )
        y_sample = ScoreDistribution(
            scores=y.scores, frequencies=rng.multinomial(int(y.n), s).astype(np.float64)
        )
        draws.append(
            kernel_equate(
                presmooth(x_sample, degree=4),
                presmooth(y_sample, degree=4),
                bandwidth_x=0.6,
                bandwidth_y=0.6,
            ).equated
        )
    empirical = np.std(np.array(draws), axis=0, ddof=1)
    np.testing.assert_allclose(empirical, result.see, rtol=0.12)


def test_presmoothing_reduces_the_standard_error() -> None:
    x, y = _forms()
    observed = kernel_equate(x, y, bandwidth_x=0.6, bandwidth_y=0.6)
    smoothed = kernel_equate(
        presmooth(x, degree=4), presmooth(y, degree=4), bandwidth_x=0.6, bandwidth_y=0.6
    )
    assert not observed.x_presmoothed
    assert np.all(observed.see > 0) and np.all(smoothed.see > 0)
    assert np.mean(smoothed.see) < np.mean(observed.see)


def test_outputs() -> None:
    x, y = _forms()
    result = kernel_equate(presmooth(x, degree=4), presmooth(y, degree=4))
    assert list(result.table.columns) == ["score", "equated", "see"]
    assert len(result.table) == x.n_scores
    continuized = result.continuized
    assert list(continuized.columns) == ["form", "score", "probability", "cdf", "density"]
    assert len(continuized) == x.n_scores + y.n_scores
    assert np.all(np.diff(continuized.query("form == 'y'")["cdf"]) > 0)
    pre = result.percent_relative_error
    assert list(pre.index) == list(range(1, 11))
    # The first moments of the equated scores are close to those of Y.
    assert np.all(np.abs(pre.iloc[:3]) < 0.1)
    assert "bandwidth_x" in repr(result)
    with pytest.raises(ValueError, match="read-only"):
        result.see[0] = 0.0


def test_specification_errors() -> None:
    x, y = _forms()
    joint = BivariateScoreDistribution.from_frequencies(
        x_scores=[0, 1, 2], y_scores=[0, 1, 2], frequencies=[[5, 2, 1], [2, 6, 2], [1, 2, 5]]
    )
    with pytest.raises(SpecificationError, match="bandwidth_x"):
        kernel_equate(x, y, bandwidth_x=0.0)
    with pytest.raises(SpecificationError, match="bandwidth_y"):
        kernel_equate(x, y, bandwidth_y=float("nan"))
    with pytest.raises(SpecificationError, match="penalty_weight"):
        kernel_equate(x, y, penalty_weight=-1)
    with pytest.raises(SpecificationError, match="penalty_weight"):
        penalty_bandwidth(x, penalty_weight=-1)
    with pytest.raises(SpecificationError, match="bivariate"):
        kernel_equate(presmooth(joint, degree=(1, 1)), y)
    with pytest.raises(SpecificationError, match="must be a ScoreDistribution"):
        kernel_equate(joint, y)
    constant = ScoreDistribution.from_frequencies(scores=[0, 1, 2], frequencies=[0, 9, 0])
    with pytest.raises(SpecificationError, match="no variance"):
        kernel_equate(constant, y)
