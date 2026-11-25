"""Unit tests for equipercentile equating with an anchor test (non-equivalent groups)."""

from __future__ import annotations

import numpy as np
import pytest

from flexpyequate import BivariateScoreDistribution, equate, presmooth
from flexpyequate.equating import EquatingMethod
from flexpyequate.errors import SparseDataWarning, SpecificationError

from .test_anchor_linear import _forms, _population

METHODS: tuple[EquatingMethod, ...] = ("frequency_estimation", "chained_equipercentile")


@pytest.mark.parametrize("method", METHODS)
def test_a_form_equated_to_itself_is_the_identity(method: EquatingMethod) -> None:
    x, _ = _forms()
    result = equate(x, x, method=method)
    np.testing.assert_allclose(result.equated, x.x_scores, atol=1e-9)
    assert result.slope is None and result.intercept is None
    assert result.design == "nonequivalent_groups"


@pytest.mark.parametrize("weight", [0.0, 0.4, 1.0])
def test_equal_anchor_distributions_give_ordinary_equipercentile_equating(weight: float) -> None:
    x, y = _forms()
    conditional_y = y.probabilities / y.y.probabilities
    y_matched = BivariateScoreDistribution.from_frequencies(
        x_scores=y.x_scores,
        y_scores=y.y_scores,
        frequencies=conditional_y * x.y.probabilities * y.n,
    )
    result = equate(x, y_matched, method="frequency_estimation", synthetic_weight=weight)
    plain = equate(x.x, y_matched.x, method="equipercentile")
    np.testing.assert_allclose(result.equated, plain.equated, atol=1e-9)


def test_frequency_estimation_result_describes_the_synthetic_population() -> None:
    x, y = _forms()
    all_x = equate(x, y, method="frequency_estimation", synthetic_weight=1.0)
    np.testing.assert_allclose(all_x.x.probabilities, x.x.probabilities, atol=1e-14)
    assert not np.allclose(all_x.y.probabilities, y.x.probabilities, atol=1e-4)
    all_y = equate(x, y, method="frequency_estimation", synthetic_weight=0.0)
    np.testing.assert_allclose(all_y.y.probabilities, y.x.probabilities, atol=1e-14)
    default = equate(x, y, method="frequency_estimation")
    assert default.synthetic_weight == pytest.approx(x.n / (x.n + y.n))
    assert default.synthetic_moments is not None
    assert float(default.synthetic_moments["mean"].to_numpy()[0]) == pytest.approx(default.x.mean)
    assert default.anchor_distributions is None and default.gamma is None
    # The equated scores have the synthetic distribution of Y, up to discreteness.
    moments = default.moments
    assert float(moments["mean"].to_numpy()[2]) == pytest.approx(default.y.mean, abs=0.05)
    assert float(moments["sd"].to_numpy()[2]) == pytest.approx(default.y.sd(), abs=0.05)


def test_frequency_estimation_shares_its_synthetic_moments_with_braun_holland() -> None:
    x, y = _forms()
    estimated = equate(x, y, method="frequency_estimation", synthetic_weight=0.3)
    braun_holland = equate(x, y, method="braun_holland", synthetic_weight=0.3)
    assert estimated.synthetic_moments is not None and braun_holland.synthetic_moments is not None
    np.testing.assert_allclose(
        estimated.synthetic_moments.to_numpy(), braun_holland.synthetic_moments.to_numpy()
    )


def test_chained_is_the_composition_of_two_single_group_equatings() -> None:
    x, y = _forms()
    x_to_anchor = equate(x, method="equipercentile")
    anchor_first = BivariateScoreDistribution.from_frequencies(
        x_scores=y.y_scores, y_scores=y.x_scores, frequencies=y.frequencies.T
    )
    anchor_to_y = equate(anchor_first, method="equipercentile")
    chained = equate(x, y, method="chained_equipercentile")
    np.testing.assert_allclose(
        chained.equated, anchor_to_y.transform(x_to_anchor.equated), atol=1e-12
    )
    assert chained.anchor_distributions is not None
    np.testing.assert_allclose(chained.anchor_distributions[0].frequencies, x.y.frequencies)
    np.testing.assert_allclose(chained.anchor_distributions[1].frequencies, y.y.frequencies)
    assert chained.synthetic_weight is None and chained.synthetic_moments is None


@pytest.mark.parametrize("method", METHODS)
def test_transform_and_monotonicity(method: EquatingMethod) -> None:
    x, y = _forms()
    result = equate(x, y, method=method)
    np.testing.assert_allclose(result.transform(x.x_scores), result.equated, atol=1e-12)
    grid = np.linspace(-2.0, 33.0, 200)
    between = result.transform(grid)
    assert np.all(np.diff(between) >= -1e-12)
    assert between[0] == pytest.approx(-0.5) and between[-1] == pytest.approx(32.5)
    assert list(result.table.columns) == ["score", "equated"]


@pytest.mark.parametrize("method", METHODS)
def test_abler_population_on_x_is_adjusted_for(method: EquatingMethod) -> None:
    # Both populations take the same form; only their ability differs. Equating X to "Y" should
    # then be near the identity, unlike equating the margins as if the groups were equivalent.
    # With a 10-item anchor the adjustment is partial (frequency estimation is known to
    # under-adjust for large group differences), so only the improvement is asserted.
    x = _population(11, 20000, 30, 0.4)
    y = _population(12, 20000, 30, -0.2)
    middle = slice(8, 24)
    adjusted = np.abs(equate(x, y, method=method).equated - x.x_scores)[middle]
    naive = np.abs(equate(x.x, y.x, method="equipercentile").equated - x.x_scores)[middle]
    assert naive.mean() > 2.0
    assert adjusted.mean() < 0.5 * naive.mean()


def test_presmoothed_joint_distributions_can_be_equated() -> None:
    x, y = _forms()
    x_smooth = presmooth(x, degree=(4, 4), cross_terms=[(1, 1)]).distribution
    y_smooth = presmooth(y, degree=(4, 4), cross_terms=[(1, 1)]).distribution
    assert isinstance(x_smooth, BivariateScoreDistribution)
    assert isinstance(y_smooth, BivariateScoreDistribution)
    for method in METHODS:
        smooth = equate(x_smooth, y_smooth, method=method).equated
        raw = equate(x, y, method=method).equated
        assert np.all(np.diff(smooth) > 0)
        assert np.abs(smooth - raw)[5:25].max() < 0.5


def test_sparse_anchor_warning_and_errors() -> None:
    x_table = np.array([[4.0, 0, 0], [3, 6, 0], [0, 5, 0], [0, 2, 0]])
    y_table = np.array([[5.0, 1, 0], [2, 6, 1], [0, 4, 3], [0, 1, 4]])
    x = BivariateScoreDistribution.from_frequencies(
        x_scores=[0, 1, 2, 3], y_scores=[0, 1, 2], frequencies=x_table
    )
    y = BivariateScoreDistribution.from_frequencies(
        x_scores=[0, 1, 2, 3], y_scores=[0, 1, 2], frequencies=y_table
    )
    with pytest.warns(SparseDataWarning, match="one population only"):
        result = equate(x, y, method="frequency_estimation")
    assert result.x.probabilities.sum() == pytest.approx(1.0)
    forms = _forms()
    with pytest.raises(SpecificationError, match="synthetic_weight"):
        equate(*forms, method="frequency_estimation", synthetic_weight=-0.1)
    with pytest.raises(SpecificationError, match="needs two BivariateScoreDistribution"):
        equate(forms[0].x, forms[1].x, method="chained_equipercentile")
