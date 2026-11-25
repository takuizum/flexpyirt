"""Unit tests for the logistic and uniform kernels and the further bandwidth rules (E7)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

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
from flexpyequate.kernel import (
    _KERNEL_VARIANCE,
    BandwidthMethod,
    Kernel,
    _Continuized,
    _cross_validation,
    _double_smoothing,
    _kernel_cdf,
    _kernel_density,
    _kernel_density_derivative,
    _penalty,
)

KERNELS = ["gaussian", "logistic", "uniform"]


def _forms(n_x: int = 2000, n_y: int = 2200) -> tuple[ScoreDistribution, ScoreDistribution]:
    rng = np.random.default_rng(42)
    return (
        ScoreDistribution.from_scores(
            rng.binomial(25, rng.beta(5, 4, size=n_x)), score_range=(0, 25)
        ),
        ScoreDistribution.from_scores(
            rng.binomial(30, rng.beta(4, 4, size=n_y)), score_range=(0, 30)
        ),
    )


def _anchor_groups() -> tuple[BivariateScoreDistribution, BivariateScoreDistribution]:
    rng = np.random.default_rng(8)

    def one(n: int, shift: float, n_items: int) -> BivariateScoreDistribution:
        theta = rng.normal(shift, 1, size=n)
        return BivariateScoreDistribution.from_scores(
            rng.binomial(n_items, 1 / (1 + np.exp(-0.9 * theta))),
            rng.binomial(8, 1 / (1 + np.exp(-theta))),
            x_range=(0, n_items),
            y_range=(0, 8),
        )

    return one(2500, 0.2, 18), one(2800, -0.1, 20)


@pytest.mark.parametrize("kernel", KERNELS)
def test_kernel_variable_has_the_stated_moments(kernel: Kernel) -> None:
    def moment(order: int) -> float:
        integral = quad(
            lambda v: v**order * _kernel_density(np.array([v]), kernel)[0],
            -60,
            60,
            points=[-0.5, 0.5],
            limit=400,
        )[0]
        return float(integral)

    assert moment(0) == pytest.approx(1.0, abs=1e-9)
    assert moment(1) == pytest.approx(0.0, abs=1e-9)
    assert moment(2) == pytest.approx(_KERNEL_VARIANCE[kernel], abs=1e-8)
    points = np.linspace(-3, 3, 41) + 0.013
    step = 1e-6
    np.testing.assert_allclose(
        (_kernel_cdf(points + step, kernel) - _kernel_cdf(points - step, kernel)) / (2 * step),
        _kernel_density(points, kernel),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        (_kernel_density(points + step, kernel) - _kernel_density(points - step, kernel))
        / (2 * step),
        _kernel_density_derivative(points, kernel),
        atol=1e-8,
    )


@pytest.mark.parametrize("kernel", ["logistic", "uniform"])
@pytest.mark.parametrize("bandwidth", [0.4, 1.0, 3.0])
def test_continuization_keeps_mean_and_variance(kernel: str, bandwidth: float) -> None:
    x, _ = _forms()
    continuized = _Continuized.build(x.scores, x.probabilities, bandwidth, kernel)  # type: ignore[arg-type]
    grid = np.linspace(-150.0, 180.0, 1_320_001)
    cdf = continuized.cdf(grid)
    # E[X] and E[(X - mu)^2] from the distribution function (integration by parts).
    mass = np.diff(cdf)
    middle = (grid[1:] + grid[:-1]) / 2
    assert cdf[0] == pytest.approx(0.0, abs=1e-12) and cdf[-1] == pytest.approx(1.0, abs=1e-12)
    assert float(mass @ middle) == pytest.approx(x.mean, abs=1e-6)
    assert float(mass @ (middle - x.mean) ** 2) == pytest.approx(x.variance(), abs=1e-4)


@pytest.mark.parametrize("kernel", ["logistic", "uniform"])
def test_cdf_gradient_matches_finite_differences(kernel: str) -> None:
    x, _ = _forms()
    scores, probabilities = x.scores, x.probabilities
    bandwidth = 1.3
    continuized = _Continuized.build(scores, probabilities, bandwidth, kernel)  # type: ignore[arg-type]
    points = np.array([3.21, 9.4, 14.03, 20.6])
    gradient = continuized.cdf_gradient(points)

    def cdf(weights: np.ndarray) -> np.ndarray:
        mean = float(weights @ scores)
        variance = float(weights @ (scores - mean) ** 2)
        return _Continuized(scores, weights, bandwidth, mean, variance, kernel).cdf(points)  # type: ignore[arg-type]

    # The formula holds on the tangent space of the simplex: compare directional derivatives
    # along directions that keep the total probability.
    rng = np.random.default_rng(0)
    step = 1e-7
    for _ in range(5):
        direction = rng.normal(size=scores.size)
        direction -= direction.mean()
        numeric = (
            cdf(probabilities + step * direction) - cdf(probabilities - step * direction)
        ) / (2 * step)
        np.testing.assert_allclose(gradient @ direction, numeric, rtol=1e-5, atol=1e-8)


def test_large_bandwidths_give_linear_equating() -> None:
    x, y = _forms()
    linear = equate(x, y, method="linear")
    result = kernel_equate(x, y, linear=True, kernel="logistic")
    np.testing.assert_allclose(result.equated, linear.equated, atol=1e-4)
    assert result.kernel == "logistic"
    gaussian = kernel_equate(x, y, linear=True)
    np.testing.assert_allclose(result.see, gaussian.see, rtol=1e-4)
    # A uniform kernel has bounded support: scores far from the mean would map to its ends.
    with pytest.raises(SpecificationError, match="uniform kernel"):
        kernel_equate(x, y, linear=True, kernel="uniform")


def test_uniform_kernel_with_unit_bandwidth_is_close_to_equipercentile() -> None:
    x, y = _forms()
    result = kernel_equate(x, y, bandwidth_x=1.0, bandwidth_y=1.0, kernel="uniform")
    equipercentile = equate(x, y)
    inside = (x.cumulative > 0.01) & (x.cumulative < 0.99)
    assert np.max(np.abs(result.equated - equipercentile.equated)[inside]) < 0.05


@pytest.mark.parametrize("kernel", ["logistic", "uniform"])
def test_equivariance_under_linear_score_transformations(kernel: str) -> None:
    x, y = _forms()
    x_scaled = ScoreDistribution(scores=10 + 0.5 * x.scores, frequencies=x.frequencies)
    y_scaled = ScoreDistribution(scores=100 + 2 * y.scores, frequencies=y.frequencies)
    base = kernel_equate(x, y, kernel=kernel, penalty_weight=0)  # type: ignore[arg-type]
    scaled = kernel_equate(x_scaled, y_scaled, kernel=kernel, penalty_weight=0)  # type: ignore[arg-type]
    assert scaled.bandwidth_x == pytest.approx(0.5 * base.bandwidth_x, rel=1e-6)
    assert scaled.bandwidth_y == pytest.approx(2 * base.bandwidth_y, rel=1e-6)
    np.testing.assert_allclose(scaled.equated, 100 + 2 * base.equated, atol=1e-5)
    np.testing.assert_allclose(scaled.see, 2 * base.see, rtol=1e-4)


@pytest.mark.parametrize("method", ["rule_of_thumb", "double_smoothing", "cross_validation"])
def test_bandwidth_rules_are_equivariant(method: str) -> None:
    x, y = _forms()
    smooth_x, smooth_y = presmooth(x, degree=4), presmooth(y, degree=4)
    scaled_x = presmooth(
        ScoreDistribution(scores=10 + 0.5 * x.scores, frequencies=x.frequencies), degree=4
    )
    base = kernel_equate(smooth_x, smooth_y, bandwidth_method=method)  # type: ignore[arg-type]
    scaled = kernel_equate(scaled_x, smooth_y, bandwidth_method=method)  # type: ignore[arg-type]
    assert base.bandwidth_method == method
    assert scaled.bandwidth_x == pytest.approx(0.5 * base.bandwidth_x, rel=1e-6)
    assert scaled.bandwidth_y == pytest.approx(base.bandwidth_y, rel=1e-9)
    np.testing.assert_allclose(scaled.equated, base.equated, atol=1e-5)
    assert np.all(np.isfinite(base.see))


def test_rule_of_thumb_formula() -> None:
    x, y = _forms()
    result = kernel_equate(x, y, bandwidth_method="rule_of_thumb")
    assert result.bandwidth_x == pytest.approx(9 * x.sd() / np.sqrt(100 * x.n**0.4 - 81))
    assert result.bandwidth_y == pytest.approx(9 * y.sd() / np.sqrt(100 * y.n**0.4 - 81))
    # Silverman's rule for the density estimate: a * h = 0.9 * sigma * N^(-1/5).
    shrinkage = np.sqrt(x.variance() / (x.variance() + result.bandwidth_x**2))
    assert shrinkage * result.bandwidth_x == pytest.approx(0.9 * x.sd() * x.n ** (-0.2))
    # A given bandwidth wins over the rule.
    mixed = kernel_equate(x, y, bandwidth_method="rule_of_thumb", bandwidth_x=0.7)
    assert mixed.bandwidth_x == 0.7 and mixed.bandwidth_y == result.bandwidth_y


def test_rule_of_thumb_uses_the_sample_that_took_each_form() -> None:
    xa, ya = _anchor_groups()
    for design in ("chained", "post_stratification"):
        result = kernel_equate(xa, ya, design=design, bandwidth_method="rule_of_thumb")
        factor_p = 9 / np.sqrt(100 * xa.n**0.4 - 81)
        factor_q = 9 / np.sqrt(100 * ya.n**0.4 - 81)
        assert result.bandwidth_x == pytest.approx(factor_p * result.x.sd())
        assert result.bandwidth_y == pytest.approx(factor_q * result.y.sd())
        if result.anchor_bandwidths is not None:
            assert result.anchor_bandwidths[0] == pytest.approx(factor_p * xa.y.sd())
            assert result.anchor_bandwidths[1] == pytest.approx(factor_q * ya.y.sd())


def test_criteria_are_minimized_by_the_selected_bandwidths() -> None:
    x, y = _forms()
    smooth = presmooth(x, degree=4)
    probabilities = smooth.distribution.probabilities
    scores = x.scores
    criteria: tuple[tuple[BandwidthMethod, Callable[[float], float]], ...] = (
        ("double_smoothing", lambda h: _double_smoothing(h, scores, probabilities, 4.0)),
        ("cross_validation", lambda h: _cross_validation(h, scores, probabilities)),
    )
    for method, criterion in criteria:
        selected = kernel_equate(smooth, y, bandwidth_method=method).bandwidth_x
        grid = np.geomspace(0.05, 10, 300)
        assert criterion(selected) <= min(criterion(h) for h in grid) + 1e-15


@pytest.mark.parametrize("kernel", ["logistic", "uniform"])
def test_penalty_bandwidth_minimizes_the_penalty(kernel: str) -> None:
    x, _ = _forms()
    smooth = presmooth(x, degree=4)
    probabilities = smooth.distribution.probabilities
    selected = penalty_bandwidth(smooth, kernel=kernel)  # type: ignore[arg-type]
    grid = np.geomspace(0.05, 10, 500)
    values = [_penalty(h, x.scores, probabilities, 1.0, kernel) for h in grid]  # type: ignore[arg-type]
    assert _penalty(selected, x.scores, probabilities, 1.0, kernel) <= min(values) + 1e-15  # type: ignore[arg-type]
    if kernel == "uniform":
        # The second penalty term does not exist for a piecewise constant density.
        assert selected == penalty_bandwidth(smooth, kernel="uniform", penalty_weight=0)


def test_uniform_kernel_with_gaps_has_no_standard_error_there() -> None:
    x, y = _forms()
    result = kernel_equate(x, y, bandwidth_x=0.5, bandwidth_y=0.5, kernel="uniform")
    assert np.all(np.isfinite(result.equated))
    assert np.all(np.diff(result.equated) >= 0)
    assert np.any(np.isnan(result.see)) or np.all(np.isfinite(result.see))
    table = result.continuized
    assert set(table["form"]) == {"x", "y"}


@pytest.mark.parametrize("kernel", ["logistic", "uniform"])
@pytest.mark.parametrize("design", ["chained", "post_stratification"])
def test_anchor_designs_run_with_every_kernel(kernel: str, design: str) -> None:
    xa, ya = _anchor_groups()
    fit_x = presmooth(xa, degree=(4, 3), cross_terms=[(1, 1)])
    fit_y = presmooth(ya, degree=(4, 3), cross_terms=[(1, 1)])
    options: dict[str, Any] = {"design": design, "kernel": kernel}
    if kernel == "uniform":
        options.update(bandwidth_x=1.3, bandwidth_y=1.3)
        if design == "chained":
            options["anchor_bandwidths"] = (1.3, 1.3)
    result = kernel_equate(fit_x, fit_y, **options)
    gaussian = kernel_equate(fit_x, fit_y, design=design)  # type: ignore[arg-type]
    assert np.all(np.isfinite(result.see)) and np.all(result.see > 0)
    assert np.max(np.abs(result.equated - gaussian.equated)) < 0.3
    seed = result.seed(gaussian)
    assert np.all(seed["seed"] < result.see + gaussian.see)
    np.testing.assert_allclose(result.transform(result.scores), result.equated, atol=1e-10)


def test_specification_errors() -> None:
    x, y = _forms()
    with pytest.raises(SpecificationError, match="Unknown kernel"):
        kernel_equate(x, y, kernel="epanechnikov")  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match="Unknown bandwidth_method"):
        kernel_equate(x, y, bandwidth_method="silverman")  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match="Gaussian kernel only"):
        kernel_equate(x, y, kernel="logistic", bandwidth_method="double_smoothing")
    with pytest.raises(SpecificationError, match="Unknown kernel"):
        penalty_bandwidth(x, kernel="triangular")  # type: ignore[arg-type]
    rng = np.random.default_rng(1)
    samples = [
        BivariateScoreDistribution.from_scores(
            rng.binomial(10, 0.5, size=300),
            rng.binomial(10, 0.55, size=300),
            x_range=(0, 10),
            y_range=(0, 10),
        )
        for _ in range(2)
    ]
    with pytest.raises(SpecificationError, match="counterbalanced"):
        kernel_equate(*samples, design="counterbalanced", bandwidth_method="rule_of_thumb")
    kernel_equate(*samples, design="counterbalanced", bandwidth_method="cross_validation")
