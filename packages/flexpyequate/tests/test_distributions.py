"""Unit tests for the score-distribution containers and the percentile functions."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from flexpyequate import BivariateScoreDistribution, ScoreDistribution
from flexpyequate.errors import ScoreDistributionError

frequency_lists = st.lists(st.integers(min_value=0, max_value=30), min_size=2, max_size=25).filter(
    lambda values: sum(values) > 0
)


def test_from_scores_tabulates_on_the_given_range() -> None:
    dist = ScoreDistribution.from_scores([1, 1, 3, 4, 4, 4], score_range=(0, 5))
    np.testing.assert_array_equal(dist.scores, np.arange(6.0))
    np.testing.assert_array_equal(dist.frequencies, [0, 2, 0, 1, 3, 0])
    assert dist.n == 6
    assert dist.n_scores == 6
    assert dist.increment == 1.0


def test_from_scores_defaults_to_the_observed_range_and_respects_increment() -> None:
    dist = ScoreDistribution.from_scores([2.5, 3.0, 3.0, 4.5], increment=0.5)
    np.testing.assert_allclose(dist.scores, [2.5, 3.0, 3.5, 4.0, 4.5])
    np.testing.assert_array_equal(dist.frequencies, [1, 2, 0, 0, 1])


def test_moments_match_numpy_on_the_raw_scores() -> None:
    rng = np.random.default_rng(1)
    values = rng.binomial(20, 0.6, size=500)
    dist = ScoreDistribution.from_scores(values, score_range=(0, 20))
    assert dist.mean == pytest.approx(values.mean(), abs=1e-12)
    assert dist.variance() == pytest.approx(values.var(), abs=1e-12)
    assert dist.variance(ddof=1) == pytest.approx(values.var(ddof=1), abs=1e-12)
    assert dist.sd(ddof=1) == pytest.approx(values.std(ddof=1), abs=1e-12)
    centered = values - values.mean()
    assert dist.skewness == pytest.approx((centered**3).mean() / values.std() ** 3, abs=1e-12)
    assert dist.kurtosis == pytest.approx((centered**4).mean() / values.std() ** 4, abs=1e-12)


def test_percentile_rank_at_possible_scores_is_below_plus_half() -> None:
    # Kolen & Brennan (2014), ch. 2: a four-item test with frequencies 2, 3, 2, 2, 1.
    dist = ScoreDistribution.from_frequencies(scores=range(5), frequencies=[2, 3, 2, 2, 1])
    np.testing.assert_allclose(dist.percentile_rank(dist.scores), [10, 35, 60, 80, 95])
    # Between possible scores the rank is piecewise linear on [x - 0.5, x + 0.5).
    np.testing.assert_allclose(
        dist.percentile_rank([-0.5, 0.5, 1.25, 4.5, 7.0]), [0, 20, 42.5, 100, 100]
    )
    np.testing.assert_allclose(dist.percentile_rank([-3.0]), [0])


def test_percentile_inverts_percentile_rank() -> None:
    dist = ScoreDistribution.from_frequencies(scores=range(5), frequencies=[2, 3, 2, 2, 1])
    np.testing.assert_allclose(dist.percentile([10, 35, 60, 80, 95]), dist.scores, atol=1e-12)
    np.testing.assert_allclose(dist.percentile([0, 100]), [-0.5, 4.5])
    np.testing.assert_allclose(dist.percentile([62.0]), [2.1], atol=1e-12)


def test_percentile_averages_upper_and_lower_over_zero_frequencies() -> None:
    # F is flat at 0.3 over scores 1 and 2: the upper percentile is 2.5, the lower 0.5.
    dist = ScoreDistribution.from_frequencies(scores=range(4), frequencies=[3, 0, 0, 7])
    np.testing.assert_allclose(dist.percentile([30.0]), [1.5])
    # No observation at the lowest score: lower = -0.5, upper = 0.5 at rank 0.
    dist = ScoreDistribution.from_frequencies(scores=range(3), frequencies=[0, 5, 5])
    np.testing.assert_allclose(dist.percentile([0.0]), [0.0])
    # ... and at the highest score: upper = 2.5, lower = 1.5 at rank 100.
    dist = ScoreDistribution.from_frequencies(scores=range(3), frequencies=[5, 5, 0])
    np.testing.assert_allclose(dist.percentile([100.0]), [2.0])


@settings(max_examples=200, deadline=None)
@given(
    frequencies=frequency_lists, low=st.integers(-5, 5), increment=st.sampled_from([0.5, 1.0, 2.0])
)
def test_percentile_round_trip_and_monotonicity(
    frequencies: list[int], low: int, increment: float
) -> None:
    scores = low + increment * np.arange(len(frequencies))
    dist = ScoreDistribution.from_frequencies(scores=scores, frequencies=frequencies)
    grid = np.linspace(scores[0] - increment, scores[-1] + increment, 101)
    ranks = dist.percentile_rank(grid)
    assert np.all(np.diff(ranks) >= -1e-10)
    assert ranks[0] == 0 and ranks[-1] == 100
    # P(P^{-1}(p)) = p for every p, whatever the zero frequencies.
    proportions = np.linspace(0, 100, 41)
    back = dist.percentile_rank(dist.percentile(proportions))
    np.testing.assert_allclose(back, proportions, atol=1e-8)
    assert np.all(np.diff(dist.percentile(proportions)) >= -1e-10)


def test_arrays_are_read_only_and_copied() -> None:
    frequencies = np.array([1.0, 2.0, 3.0])
    dist = ScoreDistribution.from_frequencies(scores=[0, 1, 2], frequencies=frequencies)
    frequencies[0] = 99
    assert dist.frequencies[0] == 1
    with pytest.raises(ValueError, match="read-only"):
        dist.frequencies[0] = 5


def test_to_frame_columns() -> None:
    frame = ScoreDistribution.from_frequencies(scores=[0, 1, 2], frequencies=[1, 2, 1]).to_frame()
    assert list(frame.columns) == [
        "score",
        "frequency",
        "probability",
        "cumulative",
        "percentile_rank",
    ]
    np.testing.assert_allclose(frame["cumulative"], [0.25, 0.75, 1.0])


@pytest.mark.parametrize(
    ("scores", "frequencies", "message"),
    [
        ([0, 1, 3], [1, 1, 1], "equally spaced"),
        ([0, 1, 1], [1, 1, 1], "strictly increasing"),
        ([0], [1], "at least two"),
        ([0, 1, 2], [1, 1], "entries"),
        ([0, 1, 2], [1, -1, 1], "non-negative"),
        ([0, 1, 2], [0, 0, 0], "all be zero"),
        ([0, 1, 2], [1, float("nan"), 1], "finite"),
    ],
)
def test_invalid_distributions_raise(
    scores: list[float], frequencies: list[float], message: str
) -> None:
    with pytest.raises(ScoreDistributionError, match=message):
        ScoreDistribution.from_frequencies(scores=scores, frequencies=frequencies)


def test_from_scores_rejects_off_grid_and_missing_values() -> None:
    with pytest.raises(ScoreDistributionError, match="not one of the possible scores"):
        ScoreDistribution.from_scores([0, 1, 7], score_range=(0, 5))
    with pytest.raises(ScoreDistributionError, match="not one of the possible scores"):
        ScoreDistribution.from_scores([0, 1.5, 2], score_range=(0, 5))
    with pytest.raises(ScoreDistributionError, match="finite"):
        ScoreDistribution.from_scores([0, float("nan"), 2], score_range=(0, 5))
    with pytest.raises(ScoreDistributionError, match="not both"):
        ScoreDistribution.from_scores([0, 1], scores=[0, 1, 2], score_range=(0, 2))
    with pytest.raises(ScoreDistributionError, match="multiple of increment"):
        ScoreDistribution.from_scores([0, 1], score_range=(0, 2.5))
    dist = ScoreDistribution.from_frequencies(scores=[0, 1], frequencies=[1, 1])
    with pytest.raises(ScoreDistributionError, match="between 0 and 100"):
        dist.percentile([101.0])


def test_bivariate_margins_and_moments() -> None:
    rng = np.random.default_rng(7)
    theta = rng.normal(size=800)
    x_values = rng.binomial(10, 1 / (1 + np.exp(-theta)))
    y_values = rng.binomial(15, 1 / (1 + np.exp(-theta + 0.3)))
    joint = BivariateScoreDistribution.from_scores(
        x_values, y_values, x_range=(0, 10), y_range=(0, 15)
    )
    assert joint.frequencies.shape == (11, 16)
    assert joint.n == 800
    np.testing.assert_array_equal(joint.x.frequencies, np.bincount(x_values, minlength=11))
    np.testing.assert_array_equal(joint.y.frequencies, np.bincount(y_values, minlength=16))
    assert joint.covariance() == pytest.approx(np.cov(x_values, y_values, ddof=0)[0, 1], abs=1e-12)
    assert joint.covariance(ddof=1) == pytest.approx(np.cov(x_values, y_values)[0, 1], abs=1e-12)
    assert joint.correlation == pytest.approx(np.corrcoef(x_values, y_values)[0, 1], abs=1e-12)
    frame = joint.to_frame()
    assert len(frame) == 11 * 16
    assert frame["frequency"].sum() == 800


def test_bivariate_validation() -> None:
    with pytest.raises(ScoreDistributionError, match="shape"):
        BivariateScoreDistribution.from_frequencies(
            x_scores=[0, 1], y_scores=[0, 1, 2], frequencies=np.ones((3, 2))
        )
    with pytest.raises(ScoreDistributionError, match="paired"):
        BivariateScoreDistribution.from_scores([0, 1], [0, 1, 2])
    with pytest.raises(ScoreDistributionError, match="non-negative"):
        BivariateScoreDistribution.from_frequencies(
            x_scores=[0, 1], y_scores=[0, 1], frequencies=[[1, -1], [1, 1]]
        )
