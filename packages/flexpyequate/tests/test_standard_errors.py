"""Unit tests for the standard errors of the traditional equating methods (E7)."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from flexpyequate import BivariateScoreDistribution, ScoreDistribution, equate
from flexpyequate._neat import (
    GroupMoments,
    chained_linear_coefficients,
    levine_gamma,
    levine_true_coefficients,
    synthetic_moments,
    tucker_gamma,
)
from flexpyequate._standard_errors import (
    anchor_linear_gradients,
    moment_covariance,
    univariate_moment_covariance,
)
from flexpyequate.equating import _resample
from flexpyequate.errors import SparseDataWarning, SpecificationError


def _forms(n_x: int = 900, n_y: int = 1100) -> tuple[ScoreDistribution, ScoreDistribution]:
    rng = np.random.default_rng(7)
    return (
        ScoreDistribution.from_scores(
            rng.binomial(20, rng.beta(5, 3, size=n_x)), score_range=(0, 20)
        ),
        ScoreDistribution.from_scores(
            rng.binomial(24, rng.beta(4, 4, size=n_y)), score_range=(0, 24)
        ),
    )


def _single_group(n: int = 1200) -> BivariateScoreDistribution:
    rng = np.random.default_rng(11)
    theta = rng.normal(size=n)
    probability = 1 / (1 + np.exp(-theta))
    return BivariateScoreDistribution.from_scores(
        rng.binomial(15, probability),
        rng.binomial(18, 1 / (1 + np.exp(-0.9 * theta + 0.2))),
        x_range=(0, 15),
        y_range=(0, 18),
    )


def _anchor_groups(
    n_1: int = 1000, n_2: int = 1300, *, internal: bool = False
) -> tuple[BivariateScoreDistribution, BivariateScoreDistribution]:
    rng = np.random.default_rng(3)

    def one(n: int, shift: float, n_items: int) -> BivariateScoreDistribution:
        theta = rng.normal(shift, 1, size=n)
        anchor = rng.binomial(6, 1 / (1 + np.exp(-theta)))
        rest = rng.binomial(n_items, 1 / (1 + np.exp(-0.9 * theta - 0.1)))
        total = rest + anchor if internal else rest
        upper = n_items + 6 if internal else n_items
        return BivariateScoreDistribution.from_scores(
            total, anchor, x_range=(0, upper), y_range=(0, 6)
        )

    return one(n_1, 0.2, 12), one(n_2, -0.1, 14)


# ----------------------------------------------------------------------------- moments
def test_moment_covariance_is_the_multinomial_delta_method() -> None:
    x, _ = _forms()
    centered = x.scores - x.mean
    m2, m3, m4 = (float(x.probabilities @ centered**k) for k in (2, 3, 4))
    omega = univariate_moment_covariance(x.scores, x.probabilities, x.n, ddof=0, normal=False)
    np.testing.assert_allclose(omega * x.n, [[m2, m3], [m3, m4 - m2**2]], rtol=1e-12)
    normal = univariate_moment_covariance(x.scores, x.probabilities, x.n, ddof=0, normal=True)
    np.testing.assert_allclose(normal * x.n, [[m2, 0], [0, 2 * m2**2]], rtol=1e-12, atol=1e-15)


def test_moment_covariance_matches_simulation() -> None:
    joint = _single_group(400)
    rng = np.random.default_rng(5)
    draws = []
    for _ in range(4000):
        sample = _resample(joint, rng)
        assert isinstance(sample, BivariateScoreDistribution)
        draws.append(
            [
                sample.x.mean,
                sample.y.mean,
                sample.x.variance(),
                sample.covariance(),
                sample.y.variance(),
            ]
        )
    empirical = np.cov(np.array(draws).T)
    omega = moment_covariance(
        joint.x_scores, joint.y_scores, joint.probabilities, joint.n, ddof=0, normal=False
    )
    scale = np.sqrt(np.outer(np.diag(omega), np.diag(omega)))
    np.testing.assert_allclose(empirical / scale, omega / scale, atol=0.07)


# ----------------------------------------------------------------------------- gradients
def _coefficients(
    method: str, theta: np.ndarray, weight: float, anchor: Any
) -> tuple[float, float]:
    group_1, group_2 = GroupMoments(*theta[:5]), GroupMoments(*theta[5:])
    # GroupMoments fields: mean_total, mean_anchor, variance_total, variance_anchor, covariance.
    if method == "chained_linear":
        return chained_linear_coefficients(group_1, group_2)
    if method == "tucker":
        gamma = tucker_gamma(group_1), tucker_gamma(group_2)
    else:
        gamma = levine_gamma(group_1, anchor), levine_gamma(group_2, anchor)
    if method == "levine_true":
        return levine_true_coefficients(group_1, group_2, *gamma)
    moments = synthetic_moments(group_1, group_2, *gamma, weight)
    slope = moments.sd_y / moments.sd_x
    return slope, moments.mean_y - slope * moments.mean_x


@pytest.mark.parametrize(
    ("method", "anchor"),
    [
        ("tucker", None),
        ("levine_observed", "internal"),
        ("levine_observed", "external"),
        ("levine_true", "internal"),
        ("levine_true", "external"),
        ("chained_linear", None),
    ],
)
def test_anchor_gradients_match_finite_differences(method: str, anchor: Any) -> None:
    xa, ya = _anchor_groups()
    group_1, group_2 = GroupMoments.of(xa, 0), GroupMoments.of(ya, 0)
    weight = 0.35
    d_slope, d_intercept = anchor_linear_gradients(method, group_1, group_2, weight, anchor)  # type: ignore[arg-type]

    def as_fields(group: GroupMoments) -> list[float]:
        return [
            group.mean_total,
            group.mean_anchor,
            group.variance_total,
            group.variance_anchor,
            group.covariance,
        ]

    fields = np.array(as_fields(group_1) + as_fields(group_2))
    # Position in the gradient (mean, mean, var total, covariance, var anchor) of each field.
    position = np.array([0, 1, 2, 4, 3, 5, 6, 7, 9, 8])
    for index in range(10):
        step = 1e-6 * max(1.0, abs(fields[index]))
        up, down = fields.copy(), fields.copy()
        up[index] += step
        down[index] -= step
        slope_up, intercept_up = _coefficients(method, up, weight, anchor)
        slope_down, intercept_down = _coefficients(method, down, weight, anchor)
        assert d_slope[position[index]] == pytest.approx(
            (slope_up - slope_down) / (2 * step), abs=1e-7
        )
        assert d_intercept[position[index]] == pytest.approx(
            (intercept_up - intercept_down) / (2 * step), abs=1e-6
        )


# ----------------------------------------------------------------------------- closed forms
@pytest.mark.parametrize("ddof", [0, 1])
def test_linear_equivalent_groups_is_the_braun_holland_formula(ddof: int) -> None:
    x, y = _forms()
    result = equate(x, y, method="linear", ddof=ddof, standard_error="analytic")

    def shape(form: ScoreDistribution) -> tuple[float, float]:
        centered = form.scores - form.mean
        variance = form.variance(ddof=ddof)
        return (
            float(form.probabilities @ centered**3) / variance**1.5,
            float(form.probabilities @ centered**4) / variance**2,
        )

    (skew_x, kurt_x), (skew_y, kurt_y) = shape(x), shape(y)
    z = (x.scores - x.mean) / x.sd(ddof=ddof)
    variance = y.variance(ddof=ddof) * (
        1 / x.n
        + 1 / y.n
        + (skew_x / x.n + skew_y / y.n) * z
        + ((kurt_x - 1) / (4 * x.n) + (kurt_y - 1) / (4 * y.n)) * z**2
    )
    assert result.see is not None
    np.testing.assert_allclose(result.see, np.sqrt(variance), rtol=1e-12)
    assert result.standard_error == "analytic"
    assert list(result.table.columns) == ["score", "equated", "see"]


def test_normal_theory_closed_forms() -> None:
    x, y = _forms()
    z = (x.scores - x.mean) / x.sd()
    see = equate(x, y, method="linear", standard_error="analytic_normal").see
    assert see is not None
    np.testing.assert_allclose(
        see, np.sqrt(y.variance() * (1 / x.n + 1 / y.n) * (1 + z**2 / 2)), rtol=1e-12
    )
    mean_see = equate(x, y, method="mean", standard_error="analytic").see
    assert mean_see is not None
    np.testing.assert_allclose(mean_see, np.sqrt(x.variance() / x.n + y.variance() / y.n))

    joint = _single_group()
    rho = joint.correlation
    z = (joint.x_scores - joint.x.mean) / joint.x.sd()
    see = equate(joint, method="linear", standard_error="analytic_normal").see
    assert see is not None
    np.testing.assert_allclose(
        see,
        np.sqrt(joint.y.variance() * (1 - rho) * (z**2 * (1 + rho) + 2) / joint.n),
        rtol=1e-12,
    )
    mean_see = equate(joint, method="mean", standard_error="analytic").see
    assert mean_see is not None
    difference_variance = joint.x.variance() + joint.y.variance() - 2 * joint.covariance()
    np.testing.assert_allclose(mean_see, np.sqrt(difference_variance / joint.n))


def test_single_group_dependence_reduces_the_standard_error() -> None:
    joint = _single_group()
    paired = equate(joint, method="linear", standard_error="analytic").see
    independent = equate(joint.x, joint.y, method="linear", standard_error="analytic").see
    assert paired is not None and independent is not None
    assert np.all(paired < independent)


def test_identity_has_no_sampling_error() -> None:
    x, y = _forms()
    for kind in ("analytic", "analytic_normal"):
        see = equate(x, y, method="identity", standard_error=kind).see
        assert see is not None and np.all(see == 0)
    assert equate(x, y, method="identity").see is None
    assert list(equate(x, y).table.columns) == ["score", "equated"]


def test_lord_formula_by_hand_and_under_rescaling() -> None:
    x = ScoreDistribution.from_frequencies(scores=range(4), frequencies=[10, 30, 40, 20])
    y = ScoreDistribution.from_frequencies(scores=range(4), frequencies=[20, 30, 30, 20])
    result = equate(x, y, standard_error="analytic")
    assert result.see is not None
    # x = 1: P = 0.1 + 0.15 = 0.25; y* = 1 (G = 0.5 > 0.25), G0 = 0.2, g = 0.3.
    p, below, step = 0.25, 0.2, 0.3
    expected = np.sqrt(
        (p * (1 - p) / 100 + (below - p**2 + (p - below) ** 2 / step) / 100) / step**2
    )
    assert result.see[1] == pytest.approx(expected, rel=1e-12)

    stretched = ScoreDistribution.from_frequencies(
        scores=100 + 2.5 * np.arange(4), frequencies=y.frequencies
    )
    rescaled = equate(x, stretched, standard_error="analytic").see
    assert rescaled is not None
    np.testing.assert_allclose(rescaled, 2.5 * result.see, rtol=1e-12)


def test_braun_holland_matches_a_numerical_jacobian() -> None:
    xa, ya = _anchor_groups(400, 500)
    weight = 0.4
    result = equate(
        xa, ya, method="braun_holland", synthetic_weight=weight, standard_error="analytic"
    )
    assert result.see is not None

    def equated(cells_1: np.ndarray, cells_2: np.ndarray) -> np.ndarray:
        first = BivariateScoreDistribution(xa.x_scores, xa.y_scores, cells_1)
        second = BivariateScoreDistribution(ya.x_scores, ya.y_scores, cells_2)
        return equate(first, second, method="braun_holland", synthetic_weight=weight).equated

    def variance(target: int) -> np.ndarray:
        sources = [xa.probabilities, ya.probabilities]
        base = sources[target]
        step = 1e-6
        jacobian = np.zeros((result.scores.size, base.size))
        for cell in np.flatnonzero(base.ravel() > 0):
            flat_shift = np.zeros(base.size)
            flat_shift[cell] = step
            shift = flat_shift.reshape(base.shape)
            up, down = list(sources), list(sources)
            up[target], down[target] = base + shift, base - shift
            jacobian[:, cell] = (equated(*up) - equated(*down)) / (2 * step)
        flat = base.ravel()
        n = (xa.n, ya.n)[target]
        return np.asarray((jacobian**2 @ flat - (jacobian @ flat) ** 2) / n)

    np.testing.assert_allclose(result.see, np.sqrt(variance(0) + variance(1)), rtol=1e-6)


# ----------------------------------------------------------------------------- availability
def test_unavailable_analytic_standard_errors() -> None:
    x, y = _forms()
    joint = _single_group()
    xa, ya = _anchor_groups()
    with pytest.raises(SpecificationError, match="bootstrap"):
        equate(joint, method="equipercentile", standard_error="analytic")
    with pytest.raises(SpecificationError, match="bootstrap"):
        equate(x, y, method="equipercentile", standard_error="analytic_normal")
    for method in ("frequency_estimation", "chained_equipercentile"):
        with pytest.raises(SpecificationError, match="bootstrap"):
            equate(xa, ya, method=method, standard_error="analytic")
    with pytest.raises(SpecificationError, match="moments"):
        equate(xa, ya, method="braun_holland", standard_error="analytic_normal")
    with pytest.raises(SpecificationError, match="Unknown standard_error"):
        equate(x, y, standard_error="jackknife")  # type: ignore[arg-type]


def test_braun_holland_needs_every_anchor_score_in_both_populations() -> None:
    xa, ya = _anchor_groups(300, 300)
    frequencies = np.array(xa.frequencies)
    frequencies[:, 0] = 0
    sparse = BivariateScoreDistribution(xa.x_scores, xa.y_scores, frequencies)
    with (
        pytest.warns(SparseDataWarning),
        pytest.raises(SpecificationError, match="one population only"),
    ):
        equate(sparse, ya, method="braun_holland", standard_error="analytic")


# ----------------------------------------------------------------------------- bootstrap
def test_bootstrap_needs_an_explicit_generator() -> None:
    x, y = _forms()
    with pytest.raises(SpecificationError, match="rng"):
        equate(x, y, standard_error="bootstrap")
    with pytest.raises(SpecificationError, match="rng"):
        equate(x, y, standard_error="bootstrap", rng=-1)
    for bad in (1, 2.5, True):
        with pytest.raises(SpecificationError, match="n_replications"):
            equate(x, y, standard_error="bootstrap", rng=1, n_replications=bad)  # type: ignore[arg-type]


def test_bootstrap_is_reproducible() -> None:
    x, y = _forms()
    first = equate(x, y, standard_error="bootstrap", n_replications=50, rng=123)
    second = equate(
        x, y, standard_error="bootstrap", n_replications=50, rng=np.random.default_rng(123)
    )
    other = equate(x, y, standard_error="bootstrap", n_replications=50, rng=124)
    assert first.see is not None and second.see is not None and other.see is not None
    np.testing.assert_array_equal(first.see, second.see)
    assert not np.array_equal(first.see, other.see)
    assert first.n_replications == 50 and first.standard_error == "bootstrap"
    np.testing.assert_array_equal(first.equated, equate(x, y).equated)


@pytest.mark.parametrize("design", ["equivalent_groups", "single_group", "anchor"])
def test_bootstrap_resampling_mechanics(design: str) -> None:
    """The documented procedure, written out: multinomial counts, X's sample first, then Y's."""
    rng = np.random.default_rng(2026)
    n_replications = 40
    replicates = []
    if design == "equivalent_groups":
        x, y = _forms(300, 350)
        result = equate(
            x, y, method="equipercentile", standard_error="bootstrap",
            n_replications=n_replications, rng=2026,
        )  # fmt: skip
        for _ in range(n_replications):
            x_counts = rng.multinomial(300, x.probabilities).astype(np.float64)
            y_counts = rng.multinomial(350, y.probabilities).astype(np.float64)
            replicates.append(
                equate(
                    ScoreDistribution(x.scores, x_counts), ScoreDistribution(y.scores, y_counts)
                ).equated
            )
    elif design == "single_group":
        joint = _single_group(400)
        result = equate(
            joint, method="linear", standard_error="bootstrap",
            n_replications=n_replications, rng=2026,
        )  # fmt: skip
        for _ in range(n_replications):
            counts = rng.multinomial(400, joint.probabilities.ravel()).astype(np.float64)
            sample = BivariateScoreDistribution(
                joint.x_scores, joint.y_scores, counts.reshape(joint.frequencies.shape)
            )
            replicates.append(equate(sample, method="linear").equated)
    else:
        xa, ya = _anchor_groups(500, 600)
        result = equate(
            xa, ya, method="tucker", standard_error="bootstrap",
            n_replications=n_replications, rng=2026,
        )  # fmt: skip
        for _ in range(n_replications):
            samples = []
            for source, size in ((xa, 500), (ya, 600)):
                counts = rng.multinomial(size, source.probabilities.ravel()).astype(np.float64)
                samples.append(
                    BivariateScoreDistribution(
                        source.x_scores, source.y_scores, counts.reshape(source.frequencies.shape)
                    )
                )
            # The synthetic weight stays that of the observed sample sizes.
            replicates.append(
                equate(*samples, method="tucker", synthetic_weight=500 / 1100).equated
            )
    stacked = np.array(replicates)
    assert result.see is not None and result.bootstrap_mean is not None
    np.testing.assert_allclose(result.see, stacked.std(axis=0, ddof=1), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(result.bootstrap_mean, stacked.mean(axis=0), rtol=1e-12)


@pytest.mark.parametrize(
    ("design", "method", "options"),
    [
        ("equivalent_groups", "mean", {}),
        ("equivalent_groups", "linear", {}),
        ("single_group", "linear", {}),
        ("anchor", "tucker", {}),
        ("anchor", "levine_observed", {"anchor": "external"}),
        ("anchor", "levine_true", {"anchor": "external"}),
        ("anchor", "chained_linear", {}),
        ("anchor", "braun_holland", {}),
    ],
)
def test_bootstrap_agrees_with_the_delta_method(
    design: str, method: str, options: dict[str, Any]
) -> None:
    """Monte Carlo error of a bootstrap SE with B replications is about SE / sqrt(2B)."""
    if design == "equivalent_groups":
        inputs: tuple[Any, ...] = _forms()
    elif design == "single_group":
        inputs = (_single_group(),)
    else:
        inputs = _anchor_groups()
    n_replications = 1500
    analytic = equate(*inputs, method=method, standard_error="analytic", **options).see  # type: ignore[arg-type]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SparseDataWarning)
        bootstrap = equate(
            *inputs,
            method=method,  # type: ignore[arg-type]
            standard_error="bootstrap",
            n_replications=n_replications,
            rng=99,
            **options,
        )
    assert analytic is not None and bootstrap.see is not None
    # Five Monte Carlo standard errors (1.8% each) plus the O(1/N) error of the delta method.
    np.testing.assert_allclose(bootstrap.see, analytic, rtol=5 / np.sqrt(2 * n_replications) + 0.03)
    assert abs(float(np.mean(bootstrap.see / analytic)) - 1) < 0.03


def test_bootstrap_equipercentile_is_close_to_lord() -> None:
    x, y = _forms(3000, 3000)
    analytic = equate(x, y, standard_error="analytic").see
    bootstrap = equate(x, y, standard_error="bootstrap", n_replications=1000, rng=5).see
    assert analytic is not None and bootstrap is not None
    middle = (x.cumulative > 0.05) & (x.cumulative < 0.95)
    ratio = bootstrap[middle] / analytic[middle]
    # Lord's formula treats the percentile rank as a binomial proportion: it is an
    # approximation for discrete scores, so the agreement is rough.
    assert np.all((ratio > 0.6) & (ratio < 1.25))


def test_bootstrap_leaves_out_samples_that_cannot_be_equated() -> None:
    x = ScoreDistribution.from_frequencies(scores=range(3), frequencies=[3, 0, 0.6])
    y = ScoreDistribution.from_frequencies(scores=range(3), frequencies=[2, 3, 2])
    with pytest.warns(SparseDataWarning, match="could not be equated"):
        result = equate(x, y, method="linear", standard_error="bootstrap", n_replications=60, rng=1)
    assert result.n_replications is not None and 2 <= result.n_replications < 60
    assert result.see is not None and np.all(np.isfinite(result.see))
