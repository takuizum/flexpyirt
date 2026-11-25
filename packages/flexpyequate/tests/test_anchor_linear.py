"""Unit tests for linear equating with an anchor test (non-equivalent groups)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from flexpyequate import BivariateScoreDistribution, EquatingResult, equate
from flexpyequate._neat import post_stratified_distributions
from flexpyequate.equating import EquatingMethod
from flexpyequate.errors import SparseDataWarning, SpecificationError

ANCHOR_METHODS: tuple[EquatingMethod, ...] = (
    "tucker",
    "levine_observed",
    "levine_true",
    "braun_holland",
    "chained_linear",
)


def _population(
    seed: int, n: int, n_items: int, shift: float, *, n_anchor: int = 10
) -> BivariateScoreDistribution:
    """Total score with an internal anchor of ``n_anchor`` items."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(shift, 1.0, size=n)
    probability = 1 / (1 + np.exp(-theta))
    anchor = rng.binomial(n_anchor, probability)
    total = anchor + rng.binomial(n_items - n_anchor, probability)
    return BivariateScoreDistribution.from_scores(
        total, anchor, x_range=(0, n_items), y_range=(0, n_anchor)
    )


def _moment(result: EquatingResult, form: str, column: str) -> float:
    assert result.synthetic_moments is not None
    return float(result.synthetic_moments[column].to_numpy()[["x", "y"].index(form)])


def _forms() -> tuple[BivariateScoreDistribution, BivariateScoreDistribution]:
    return _population(1, 3000, 30, 0.2), _population(2, 3500, 32, -0.1)


def _equate(
    x: BivariateScoreDistribution,
    y: BivariateScoreDistribution,
    method: EquatingMethod,
    **options: float,
) -> tuple[float, float]:
    result = equate(x, y, method=method, anchor="internal", **options)  # type: ignore[arg-type]
    assert result.slope is not None and result.intercept is not None
    return result.slope, result.intercept


@pytest.mark.parametrize("method", ANCHOR_METHODS)
def test_a_form_equated_to_itself_is_the_identity(method: EquatingMethod) -> None:
    x, _ = _forms()
    slope, intercept = _equate(x, x, method)
    assert slope == pytest.approx(1.0, abs=1e-12)
    assert intercept == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("method", ["tucker", "levine_observed", "braun_holland"])
@pytest.mark.parametrize("weight", [0.0, 0.3, 1.0])
def test_equal_anchor_distributions_give_ordinary_linear_equating(
    method: EquatingMethod, weight: float
) -> None:
    # Same anchor margin in both populations: the synthetic population is either population.
    x, y = _forms()
    anchor = x.y.probabilities
    conditional_y = y.probabilities / y.y.probabilities
    y_matched = BivariateScoreDistribution.from_frequencies(
        x_scores=y.x_scores, y_scores=y.y_scores, frequencies=conditional_y * anchor * y.n
    )
    slope, intercept = _equate(x, y_matched, method, synthetic_weight=weight)
    plain = equate(x.x, y_matched.x, method="linear")
    assert slope == pytest.approx(plain.slope, rel=1e-10)
    assert intercept == pytest.approx(plain.intercept, abs=1e-9)


def test_synthetic_moments_at_the_extreme_weights() -> None:
    x, y = _forms()
    methods: tuple[EquatingMethod, ...] = ("tucker", "levine_observed", "braun_holland")
    for method in methods:
        all_x = equate(x, y, method=method, anchor="internal", synthetic_weight=1.0)
        assert _moment(all_x, "x", "mean") == pytest.approx(x.x.mean)
        assert _moment(all_x, "x", "sd") == pytest.approx(x.x.sd())
        all_y = equate(x, y, method=method, anchor="internal", synthetic_weight=0.0)
        assert _moment(all_y, "y", "mean") == pytest.approx(y.x.mean)
        assert _moment(all_y, "y", "sd") == pytest.approx(y.x.sd())


def test_default_weight_is_proportional_to_sample_size() -> None:
    x, y = _forms()
    default = equate(x, y, method="tucker")
    assert default.synthetic_weight == pytest.approx(3000 / 6500)
    explicit = equate(x, y, method="tucker", synthetic_weight=3000 / 6500)
    assert default.slope == explicit.slope and default.intercept == explicit.intercept
    assert default.design == "nonequivalent_groups"
    assert default.gamma is not None
    assert default.gamma[0] == pytest.approx(x.covariance() / x.y.variance())
    np.testing.assert_allclose(default.x.frequencies, x.x.frequencies)
    np.testing.assert_allclose(default.y.frequencies, y.x.frequencies)
    np.testing.assert_allclose(default.transform([0.0, 10.0]), default.equated[[0, 10]])


def test_tucker_equated_scores_match_the_synthetic_moments() -> None:
    x, y = _forms()
    result = equate(x, y, method="tucker", synthetic_weight=0.4)
    assert result.slope is not None and result.intercept is not None
    assert result.slope == pytest.approx(_moment(result, "y", "sd") / _moment(result, "x", "sd"))
    assert result.intercept + result.slope * _moment(result, "x", "mean") == pytest.approx(
        _moment(result, "y", "mean")
    )


def test_chained_linear_is_the_composition_of_two_single_group_equatings() -> None:
    x, y = _forms()
    x_to_anchor = equate(x, method="linear")
    anchor_first = BivariateScoreDistribution.from_frequencies(
        x_scores=y.y_scores, y_scores=y.x_scores, frequencies=y.frequencies.T
    )
    anchor_to_y = equate(anchor_first, method="linear")
    chained = equate(x, y, method="chained_linear")
    np.testing.assert_allclose(
        chained.equated, anchor_to_y.transform(x_to_anchor.equated), atol=1e-10
    )
    assert chained.synthetic_weight is None and chained.gamma is None
    assert chained.synthetic_moments is None


def test_levine_true_score_uses_the_ratio_of_gammas() -> None:
    x, y = _forms()
    result = equate(x, y, method="levine_true", anchor="internal")
    gamma_1 = x.x.variance() / x.covariance()
    gamma_2 = y.x.variance() / y.covariance()
    assert result.gamma == pytest.approx((gamma_1, gamma_2))
    assert result.slope == pytest.approx(gamma_2 / gamma_1)
    assert result.synthetic_weight is None and result.anchor == "internal"
    external = equate(x, y, method="levine_true", anchor="external")
    assert external.gamma is not None
    assert external.gamma[0] == pytest.approx(
        (x.x.variance() + x.covariance()) / (x.y.variance() + x.covariance())
    )


def test_levine_recovers_the_true_score_relation_of_parallel_parts() -> None:
    # Classical test theory: with an internal anchor of 10 out of 30 exchangeable items, the
    # true total score is 3 times the true anchor score, so gamma is close to 3.
    x = _population(5, 200_000, 30, 0.0)
    result = equate(x, x, method="levine_observed", anchor="internal")
    assert result.gamma is not None
    assert result.gamma[0] == pytest.approx(3.0, abs=0.03)


def test_ddof_matters_only_for_the_synthetic_variances() -> None:
    x, y = _forms()
    changed: tuple[EquatingMethod, ...] = ("tucker", "levine_observed")
    for method in changed:
        population = _equate(x, y, method)
        sample = _equate(x, y, method, ddof=1)
        assert sample != population
        assert sample[0] == pytest.approx(population[0], rel=1e-3)
    # Ratios of moments of one population (gamma, the chained slopes) do not depend on ddof,
    # and Braun-Holland uses the moments of the synthetic distributions.
    unchanged: tuple[EquatingMethod, ...] = ("levine_true", "chained_linear", "braun_holland")
    for method in unchanged:
        assert _equate(x, y, method, ddof=1) == pytest.approx(_equate(x, y, method), rel=1e-13)


def test_post_stratified_distributions_by_direct_summation() -> None:
    x, y = _forms()
    weight = 0.35
    synthetic_x, synthetic_y, dropped = post_stratified_distributions(
        x.probabilities, y.probabilities, weight
    )
    assert dropped == 0.0
    h1, h2 = x.y.probabilities, y.y.probabilities
    expected_x = np.zeros(x.x_scores.size)
    for j in range(x.x_scores.size):
        conditional = sum(x.probabilities[j, v] / h1[v] * h2[v] for v in range(h1.size))
        expected_x[j] = weight * x.x.probabilities[j] + (1 - weight) * conditional
    np.testing.assert_allclose(synthetic_x, expected_x, atol=1e-14)
    assert synthetic_x.sum() == pytest.approx(1.0) and synthetic_y.sum() == pytest.approx(1.0)
    # The synthetic anchor margin is the same mixture whichever form it is computed from.
    np.testing.assert_allclose(
        (x.probabilities / h1) @ (weight * h1 + (1 - weight) * h2), synthetic_x, atol=1e-14
    )


def test_braun_holland_warns_when_an_anchor_score_is_missing_in_one_population() -> None:
    x_table = np.array([[4.0, 0, 0], [3, 6, 0], [0, 5, 0], [0, 2, 0]])
    y_table = np.array([[5.0, 1, 0], [2, 6, 1], [0, 4, 3], [0, 1, 4]])
    x = BivariateScoreDistribution.from_frequencies(
        x_scores=[0, 1, 2, 3], y_scores=[0, 1, 2], frequencies=x_table
    )
    y = BivariateScoreDistribution.from_frequencies(
        x_scores=[0, 1, 2, 3], y_scores=[0, 1, 2], frequencies=y_table
    )
    with pytest.warns(SparseDataWarning, match="one population only"):
        result = equate(x, y, method="braun_holland")
    assert np.isfinite(result.equated).all()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # With all the weight on the population that has every anchor score nothing is dropped.
        equate(y, x, method="braun_holland", synthetic_weight=0.0)
        equate(x, y, method="tucker")


def test_specification_errors() -> None:
    x, y = _forms()
    with pytest.raises(SpecificationError, match="needs anchor"):
        equate(x, y, method="levine_observed")
    with pytest.raises(SpecificationError, match="needs anchor"):
        equate(x, y, method="levine_true", anchor="both")  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match="synthetic_weight"):
        equate(x, y, method="tucker", synthetic_weight=1.2)
    with pytest.raises(SpecificationError, match="not available for non-equivalent groups"):
        equate(x, y, method="equipercentile")
    with pytest.raises(SpecificationError, match="needs two BivariateScoreDistribution"):
        equate(x.x, y.x, method="tucker")
    with pytest.raises(SpecificationError, match="needs two BivariateScoreDistribution"):
        equate(x, method="chained_linear")
    other_anchor = _population(3, 500, 32, 0.0, n_anchor=12)
    with pytest.raises(SpecificationError, match="same possible anchor scores"):
        equate(x, other_anchor, method="braun_holland")
    # The other methods only use moments, so the anchor may be scored on different scales.
    assert equate(x, other_anchor, method="chained_linear").slope is not None
    constant = BivariateScoreDistribution.from_frequencies(
        x_scores=[0, 1, 2], y_scores=[0, 1], frequencies=[[0, 0], [3, 4], [0, 0]]
    )
    with pytest.raises(SpecificationError, match="positive variance"):
        equate(constant, constant, method="tucker")
