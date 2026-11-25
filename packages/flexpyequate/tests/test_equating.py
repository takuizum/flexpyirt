"""Unit tests for identity, mean, linear and equipercentile equating."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from flexpyequate import BivariateScoreDistribution, ScoreDistribution, equate
from flexpyequate.equating import EquatingMethod
from flexpyequate.errors import SpecificationError


def _pair() -> tuple[ScoreDistribution, ScoreDistribution]:
    x = ScoreDistribution.from_frequencies(scores=range(5), frequencies=[4, 6, 6, 3, 1])
    y = ScoreDistribution.from_frequencies(scores=range(5), frequencies=[2, 4, 6, 4, 4])
    return x, y


def test_identity_and_mean() -> None:
    x, y = _pair()
    identity = equate(x, y, method="identity")
    np.testing.assert_array_equal(identity.equated, x.scores)
    mean = equate(x, y, method="mean")
    assert mean.slope == 1.0
    assert mean.intercept == pytest.approx(y.mean - x.mean)
    assert mean.moments.loc["equated", "mean"] == pytest.approx(y.mean)
    assert mean.design == "equivalent_groups"


def test_linear_matches_the_first_two_moments() -> None:
    x, y = _pair()
    linear = equate(x, y, method="linear")
    moments = linear.moments
    assert moments.loc["equated", "mean"] == pytest.approx(y.mean, abs=1e-12)
    assert moments.loc["equated", "sd"] == pytest.approx(y.sd(), abs=1e-12)
    # A linear function preserves skewness and kurtosis of X.
    assert moments.loc["equated", "skewness"] == pytest.approx(x.skewness, abs=1e-12)
    assert moments.loc["equated", "kurtosis"] == pytest.approx(x.kurtosis, abs=1e-12)
    np.testing.assert_allclose(
        linear.transform([0.5, 10.0]), [linear.transform(0.5), linear.transform(10.0)]
    )


def test_linear_ddof_matters_only_for_unequal_sample_sizes() -> None:
    x, y = _pair()
    assert equate(x, y, method="linear", ddof=1).slope == pytest.approx(
        equate(x, y, method="linear").slope, abs=1e-14
    )
    bigger = ScoreDistribution.from_frequencies(scores=range(5), frequencies=[4, 8, 12, 8, 8])
    population = equate(x, bigger, method="linear")
    sample = equate(x, bigger, method="linear", ddof=1)
    assert population.slope is not None and sample.slope is not None
    assert sample.slope / population.slope == pytest.approx(np.sqrt((40 / 39) / (20 / 19)))


def test_equipercentile_by_hand() -> None:
    x, y = _pair()
    result = equate(x, y, method="equipercentile")
    # P(x) = .10, .35, .65, .875, .975;  G = .10, .30, .60, .80, 1.
    np.testing.assert_allclose(result.equated, [0.5, 1 + 2 / 3, 2.75, 3.875, 4.375], atol=1e-12)
    assert result.slope is None and result.intercept is None
    assert list(result.table.columns) == ["score", "equated"]
    np.testing.assert_allclose(result.transform(x.scores), result.equated)
    np.testing.assert_allclose(result.transform([-2.0, 99.0]), [-0.5, 4.5])


def test_equipercentile_of_a_distribution_with_itself_is_the_identity() -> None:
    x, _ = _pair()
    np.testing.assert_allclose(equate(x, x).equated, x.scores, atol=1e-12)
    shifted = ScoreDistribution.from_frequencies(
        scores=np.arange(5) * 2.0 + 10, frequencies=x.frequencies
    )
    np.testing.assert_allclose(equate(x, shifted).equated, shifted.scores, atol=1e-12)


@settings(max_examples=150, deadline=None)
@given(
    x_frequencies=st.lists(st.integers(1, 40), min_size=3, max_size=20),
    y_frequencies=st.lists(st.integers(1, 40), min_size=3, max_size=20),
)
def test_equipercentile_properties(x_frequencies: list[int], y_frequencies: list[int]) -> None:
    x = ScoreDistribution.from_frequencies(
        scores=range(len(x_frequencies)), frequencies=x_frequencies
    )
    y = ScoreDistribution.from_frequencies(
        scores=range(len(y_frequencies)), frequencies=y_frequencies
    )
    forward = equate(x, y)
    assert np.all(np.diff(forward.equated) > 0)
    assert forward.equated[0] >= -0.5 and forward.equated[-1] <= len(y_frequencies) - 0.5
    # The equated scores have the percentile ranks of the X scores (all frequencies positive).
    np.testing.assert_allclose(
        y.percentile_rank(forward.equated), x.percentile_rank(x.scores), atol=1e-9
    )
    # Symmetry: equating back returns the original scores.
    backward = equate(y, x)
    np.testing.assert_allclose(backward.transform(forward.equated), x.scores, atol=1e-8)


def test_single_group_uses_the_margins() -> None:
    rng = np.random.default_rng(3)
    x_values = rng.binomial(8, 0.5, size=300)
    y_values = np.clip(x_values + rng.integers(-1, 3, size=300), 0, 10)
    joint = BivariateScoreDistribution.from_scores(
        x_values, y_values, x_range=(0, 8), y_range=(0, 10)
    )
    methods: tuple[EquatingMethod, ...] = ("mean", "linear", "equipercentile")
    for method in methods:
        single = equate(joint, method=method)
        separate = equate(joint.x, joint.y, method=method)
        assert single.design == "single_group"
        np.testing.assert_array_equal(single.equated, separate.equated)


def test_specification_errors() -> None:
    x, y = _pair()
    joint = BivariateScoreDistribution.from_frequencies(
        x_scores=[0, 1], y_scores=[0, 1], frequencies=[[2, 1], [1, 2]]
    )
    with pytest.raises(SpecificationError, match="Unknown method"):
        equate(x, y, method="circle_arc")  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match="not a bivariate and a univariate"):
        equate(joint, y)
    with pytest.raises(SpecificationError, match="two ScoreDistribution"):
        equate(x)
    with pytest.raises(SpecificationError, match="ddof"):
        equate(x, y, method="linear", ddof=2)
    constant = ScoreDistribution.from_frequencies(scores=[0, 1, 2], frequencies=[0, 5, 0])
    with pytest.raises(SpecificationError, match="positive variance"):
        equate(constant, y, method="linear")


def test_result_is_immutable() -> None:
    x, y = _pair()
    result = equate(x, y)
    with pytest.raises(ValueError, match="read-only"):
        result.equated[0] = 0.0
    assert "equipercentile" in repr(result)
