"""Unit tests for kernel equating beyond equivalent groups: SG, CB, chained, PSE and SEED."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from flexpyequate import (
    BivariateScoreDistribution,
    LoglinearResult,
    ScoreDistribution,
    equate,
    kernel_equate,
    presmooth,
)
from flexpyequate.errors import SpecificationError

from .test_anchor_linear import _population


def _single_group(seed: int = 3, n: int = 2500) -> BivariateScoreDistribution:
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=n)
    x_values = rng.binomial(12, 1 / (1 + np.exp(-theta)))
    y_values = rng.binomial(14, 1 / (1 + np.exp(-0.9 * theta + 0.3)))
    return BivariateScoreDistribution.from_scores(
        x_values, y_values, x_range=(0, 12), y_range=(0, 14)
    )


def _neat() -> tuple[LoglinearResult, LoglinearResult]:
    x = _population(1, 3000, 20, 0.2, n_anchor=8)
    y = _population(2, 3500, 22, -0.1, n_anchor=8)
    return (
        presmooth(x, degree=(4, 3), cross_terms=[(1, 1)]),
        presmooth(y, degree=(4, 3), cross_terms=[(1, 1)]),
    )


def test_single_group_under_independence_equals_equivalent_groups() -> None:
    # In the independence model the two margins are estimated independently, so one sample of
    # size N behaves like two equivalent groups of size N.
    joint = _single_group()
    single = kernel_equate(
        presmooth(joint, degree=(4, 4), cross_terms=()), bandwidth_x=0.6, bandwidth_y=0.6
    )
    groups = kernel_equate(
        presmooth(joint.x, degree=4), presmooth(joint.y, degree=4), bandwidth_x=0.6, bandwidth_y=0.6
    )
    assert single.design == "single_group"
    np.testing.assert_allclose(single.equated, groups.equated, atol=1e-7)
    np.testing.assert_allclose(single.see, groups.see, rtol=1e-6)


def test_single_group_dependence_reduces_the_standard_error() -> None:
    joint = _single_group()
    dependent = kernel_equate(presmooth(joint, degree=(4, 4), cross_terms=[(1, 1)]))
    independent = kernel_equate(presmooth(joint, degree=(4, 4), cross_terms=()))
    assert joint.correlation > 0.5
    assert np.all(dependent.see < independent.see)
    observed = kernel_equate(joint)
    assert not observed.x_presmoothed and dependent.x_presmoothed
    assert np.all(observed.see > 0)


def test_counterbalanced_with_identical_samples() -> None:
    smooth = presmooth(_single_group(), degree=(3, 3), cross_terms=[(1, 1)])
    single = kernel_equate(smooth, bandwidth_x=0.6, bandwidth_y=0.6)
    both = kernel_equate(smooth, smooth, design="counterbalanced", bandwidth_x=0.6, bandwidth_y=0.6)
    assert both.counterbalance_weights == (0.5, 0.5)
    np.testing.assert_allclose(both.equated, single.equated, atol=1e-10)
    # Two independent samples of the same size: the variance halves.
    np.testing.assert_allclose(both.see, single.see / np.sqrt(2), rtol=1e-10)
    assert both.x.n == pytest.approx(2 * single.x.n)


def test_counterbalanced_extreme_weights_use_one_sample_per_form() -> None:
    first = _single_group(3)
    second = _single_group(4, n=2000)
    smooth_1 = presmooth(first, degree=(4, 4), cross_terms=())
    smooth_2 = presmooth(second, degree=(4, 4), cross_terms=())
    result = kernel_equate(
        smooth_1,
        smooth_2,
        design="counterbalanced",
        counterbalance_weights=(1.0, 0.0),
        bandwidth_x=0.6,
        bandwidth_y=0.6,
    )
    groups = kernel_equate(
        presmooth(first.x, degree=4),
        presmooth(second.y, degree=4),
        bandwidth_x=0.6,
        bandwidth_y=0.6,
    )
    np.testing.assert_allclose(result.equated, groups.equated, atol=1e-7)
    np.testing.assert_allclose(result.see, groups.see, rtol=1e-6)


def test_linear_kernel_equating_matches_the_linear_methods() -> None:
    x, y = _neat()
    x_joint, y_joint = x.distribution, y.distribution
    assert isinstance(x_joint, BivariateScoreDistribution)
    assert isinstance(y_joint, BivariateScoreDistribution)
    post = kernel_equate(x, y, design="post_stratification", linear=True, synthetic_weight=0.3)
    braun_holland = equate(x_joint, y_joint, method="braun_holland", synthetic_weight=0.3)
    np.testing.assert_allclose(post.equated, braun_holland.equated, atol=1e-7)
    chained = kernel_equate(x, y, design="chained", linear=True)
    chained_linear = equate(x_joint, y_joint, method="chained_linear")
    np.testing.assert_allclose(chained.equated, chained_linear.equated, atol=1e-7)
    assert post.bandwidth_x == pytest.approx(1000 * post.x.sd())
    assert chained.anchor_bandwidths is not None
    assert chained.anchor_bandwidths[0] == pytest.approx(1000 * x_joint.y.sd())


def test_post_stratification_distributions_match_frequency_estimation() -> None:
    x, y = _neat()
    x_joint, y_joint = x.distribution, y.distribution
    assert isinstance(x_joint, BivariateScoreDistribution)
    assert isinstance(y_joint, BivariateScoreDistribution)
    kernel = kernel_equate(x, y, design="post_stratification")
    assert kernel.synthetic_weight == pytest.approx(3000 / 6500)
    estimated = equate(x_joint, y_joint, method="frequency_estimation")
    np.testing.assert_allclose(kernel.x.probabilities, estimated.x.probabilities, atol=1e-13)
    np.testing.assert_allclose(kernel.y.probabilities, estimated.y.probabilities, atol=1e-13)
    # Kernel and equipercentile equating of the same distributions are close.
    assert np.abs(kernel.equated - estimated.equated)[2:-2].max() < 0.1


@pytest.mark.parametrize("design", ["chained", "post_stratification"])
def test_a_form_equated_to_itself_is_the_identity(design: str) -> None:
    x, _ = _neat()
    result = kernel_equate(x, x, design=design, bandwidth_x=0.6, bandwidth_y=0.6)  # type: ignore[arg-type]
    np.testing.assert_allclose(result.equated, result.scores, atol=1e-8)


def test_chained_outputs() -> None:
    x, y = _neat()
    result = kernel_equate(
        x, y, design="chained", bandwidth_x=0.6, bandwidth_y=0.6, anchor_bandwidths=(0.5, None)
    )
    assert result.anchor_bandwidths is not None and result.anchor_bandwidths[0] == 0.5
    assert 0.3 < result.anchor_bandwidths[1] < 1.0
    np.testing.assert_allclose(result.transform(result.scores), result.equated, atol=1e-12)
    assert np.all(np.diff(result.transform(np.linspace(-1, 21, 60))) > 0)
    assert set(result.continuized["form"]) == {"x", "y", "anchor_x", "anchor_y"}
    pre = result.chain_percent_relative_error
    assert list(pre.columns) == ["x_to_anchor", "anchor_to_y"] and len(pre) == 10
    assert np.all(np.abs(pre.iloc[:2].to_numpy()) < 0.1)
    with pytest.raises(SpecificationError, match="chain_percent_relative_error"):
        _ = result.percent_relative_error
    post = kernel_equate(x, y, design="post_stratification")
    with pytest.raises(SpecificationError, match="needs chained"):
        _ = post.chain_percent_relative_error
    assert post.anchor_distributions is None
    assert set(post.continuized["form"]) == {"x", "y"}


def test_seed_properties() -> None:
    x, y = _neat()
    chained = kernel_equate(x, y, design="chained")
    post = kernel_equate(x, y, design="post_stratification")
    linear = kernel_equate(x, y, design="post_stratification", linear=True)
    np.testing.assert_allclose(post.seed(post)["seed"], 0.0, atol=1e-15)
    for first, second in ((chained, post), (post, linear)):
        frame = first.seed(second)
        assert list(frame.columns) == ["score", "difference", "seed"]
        np.testing.assert_allclose(frame["difference"], first.equated - second.equated)
        # Triangle inequalities for the norm of a difference of vectors.
        assert np.all(frame["seed"] <= first.see + second.see + 1e-12)
        assert np.all(frame["seed"] >= np.abs(first.see - second.see) - 1e-12)
        np.testing.assert_allclose(second.seed(first)["seed"], frame["seed"])
    # Two estimates from the same data are positively correlated, so the SEED is far smaller
    # than for independent estimates.
    assert np.all(chained.seed(post)["seed"] < np.sqrt(chained.see**2 + post.see**2))
    groups = kernel_equate(x.distribution.x, y.distribution.x)  # type: ignore[union-attr]
    with pytest.raises(SpecificationError, match="same data"):
        post.seed(groups)
    observed = x.observed
    assert isinstance(observed, BivariateScoreDistribution)
    other_model = kernel_equate(
        presmooth(observed, degree=(3, 3), cross_terms=[(1, 1)]), y, design="post_stratification"
    )
    with pytest.raises(SpecificationError, match="same data"):
        post.seed(other_model)


def test_see_and_seed_match_a_parametric_bootstrap() -> None:
    x, y = _neat()
    x_joint, y_joint = x.distribution, y.distribution
    assert isinstance(x_joint, BivariateScoreDistribution)
    assert isinstance(y_joint, BivariateScoreDistribution)

    def both(first: LoglinearResult, second: LoglinearResult) -> tuple[Any, Any]:
        return (
            kernel_equate(
                first, second, design="post_stratification", bandwidth_x=0.6, bandwidth_y=0.6
            ),
            kernel_equate(
                first,
                second,
                design="chained",
                bandwidth_x=0.6,
                bandwidth_y=0.6,
                anchor_bandwidths=(0.6, 0.6),
            ),
        )

    post, chained = both(x, y)
    rng = np.random.default_rng(99)
    draws_post, draws_chained = [], []
    for _ in range(400):
        samples = []
        for joint in (x_joint, y_joint):
            counts = rng.multinomial(int(joint.n), joint.probabilities.ravel())
            sample = BivariateScoreDistribution(
                x_scores=joint.x_scores,
                y_scores=joint.y_scores,
                frequencies=counts.reshape(joint.frequencies.shape).astype(np.float64),
            )
            samples.append(presmooth(sample, degree=(4, 3), cross_terms=[(1, 1)]))
        post_draw, chained_draw = both(samples[0], samples[1])
        draws_post.append(post_draw.equated)
        draws_chained.append(chained_draw.equated)
    post_draws, chained_draws = np.array(draws_post), np.array(draws_chained)
    np.testing.assert_allclose(post_draws.std(axis=0, ddof=1), post.see, rtol=0.15)
    np.testing.assert_allclose(chained_draws.std(axis=0, ddof=1), chained.see, rtol=0.15)
    np.testing.assert_allclose(
        (chained_draws - post_draws).std(axis=0, ddof=1), chained.seed(post)["seed"], rtol=0.15
    )


def test_specification_errors() -> None:
    x, y = _neat()
    joint = _single_group()
    margin: ScoreDistribution = joint.x
    with pytest.raises(SpecificationError, match="design must be"):
        kernel_equate(x, y)
    with pytest.raises(SpecificationError, match="design must be"):
        kernel_equate(x, y, design="single_group")
    with pytest.raises(SpecificationError, match="does not fit the inputs"):
        kernel_equate(joint, design="chained")
    with pytest.raises(SpecificationError, match="does not fit the inputs"):
        kernel_equate(margin, margin, design="post_stratification")
    with pytest.raises(SpecificationError, match="must be a ScoreDistribution"):
        kernel_equate(joint, margin)
    with pytest.raises(SpecificationError, match="anchor_bandwidths applies"):
        kernel_equate(x, y, design="post_stratification", anchor_bandwidths=(0.5, 0.5))
    with pytest.raises(SpecificationError, match="linear=True or bandwidth_x"):
        kernel_equate(x, y, design="chained", linear=True, bandwidth_x=0.5)
    with pytest.raises(SpecificationError, match="synthetic_weight"):
        kernel_equate(x, y, design="post_stratification", synthetic_weight=1.5)
    with pytest.raises(SpecificationError, match="counterbalance_weights"):
        kernel_equate(joint, joint, design="counterbalanced", counterbalance_weights=(0.5, 2.0))
    with pytest.raises(SpecificationError, match="same possible X scores"):
        kernel_equate(joint, _population(1, 500, 12, 0.0, n_anchor=8), design="counterbalanced")
    with pytest.raises(SpecificationError, match="same possible anchor scores"):
        kernel_equate(x, _population(1, 500, 20, 0.0, n_anchor=10), design="chained")
    # An anchor score observed in one population only: post-stratification needs presmoothing.
    sparse_x = BivariateScoreDistribution.from_frequencies(
        x_scores=[0, 1, 2, 3],
        y_scores=[0, 1, 2],
        frequencies=[[4, 0, 0], [3, 6, 0], [0, 5, 0], [0, 2, 0]],
    )
    full_y = BivariateScoreDistribution.from_frequencies(
        x_scores=[0, 1, 2, 3],
        y_scores=[0, 1, 2],
        frequencies=[[5, 1, 0], [2, 6, 1], [0, 4, 3], [0, 1, 4]],
    )
    with pytest.raises(SpecificationError, match="presmooth"):
        kernel_equate(
            sparse_x, full_y, design="post_stratification", bandwidth_x=0.6, bandwidth_y=0.6
        )
