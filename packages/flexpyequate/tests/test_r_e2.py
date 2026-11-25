"""E2 against R: polynomial log-linear presmoothing (DESIGN.md §7, case Q3).

Reference values: ``stats::glm(family = poisson)`` with a strict tolerance, and R ``equate``'s
presmoothing, which uses glm's default tolerance (``validation/r/equate_e2_reference.R``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from flexpyequate import BivariateScoreDistribution, ScoreDistribution, equate, presmooth

from ._reference import load_reference

REFERENCE = load_reference("e2_presmoothing")
UNIVARIATE = [(case, fit) for case in REFERENCE["univariate"] for fit in case["fits"]]
UNIVARIATE_IDS = [f"{case['name']}-degree{fit['degree']}" for case, fit in UNIVARIATE]
BIVARIATE = REFERENCE["bivariate"]

FITTED_RTOL = 1e-8  # glm with epsilon = 1e-13
EQUATE_FITTED_RTOL = 1e-5  # equate: glm default tolerance
STATISTIC_RTOL = 1e-7
SMOOTHED_EQUATING_TOL = 1e-5


def _check_statistics(statistics: Any, fit: dict[str, Any]) -> None:
    assert fit["converged"] is True
    for key in ("likelihood_ratio", "pearson", "freeman_tukey", "log_likelihood", "aic"):
        assert statistics[key] == pytest.approx(fit[key], rel=STATISTIC_RTOL), key
    assert statistics["degrees_of_freedom"] == fit["degrees_of_freedom"]


@pytest.mark.parametrize(("case", "fit"), UNIVARIATE, ids=UNIVARIATE_IDS)
def test_univariate(case: dict[str, Any], fit: dict[str, Any]) -> None:
    observed = ScoreDistribution.from_frequencies(
        scores=case["scores"], frequencies=case["frequencies"]
    )
    result = presmooth(observed, degree=fit["degree"])
    assert result.converged
    np.testing.assert_allclose(result.distribution.frequencies, fit["fitted"], rtol=FITTED_RTOL)
    np.testing.assert_allclose(
        result.distribution.frequencies, fit["equate_fitted"], rtol=EQUATE_FITTED_RTOL
    )
    _check_statistics(result.fit_statistics, fit)


@pytest.mark.parametrize("index", range(len(BIVARIATE["fits"])))
def test_bivariate(index: int) -> None:
    fit = BIVARIATE["fits"][index]
    observed = BivariateScoreDistribution.from_frequencies(
        x_scores=BIVARIATE["x_scores"],
        y_scores=BIVARIATE["y_scores"],
        frequencies=BIVARIATE["joint_frequencies"],
    )
    degree = (int(fit["degree"][0]), int(fit["degree"][1]))
    cross_terms = [(int(a), int(b)) for a, b in fit["cross_terms"]]
    result = presmooth(observed, degree=degree, cross_terms=cross_terms)
    assert result.converged
    np.testing.assert_allclose(result.distribution.frequencies, fit["fitted"], rtol=FITTED_RTOL)
    if "equate_fitted" in fit:
        np.testing.assert_allclose(
            result.distribution.frequencies, fit["equate_fitted"], rtol=EQUATE_FITTED_RTOL
        )
    _check_statistics(result.fit_statistics, fit)


@pytest.mark.parametrize("index", range(len(REFERENCE["smoothed_equating"]["results"])))
def test_equipercentile_equating_of_presmoothed_distributions(index: int) -> None:
    data = REFERENCE["smoothed_equating"]
    reference = data["results"][index]
    x = ScoreDistribution.from_frequencies(
        scores=data["x_scores"], frequencies=data["x_frequencies"]
    )
    y = ScoreDistribution.from_frequencies(
        scores=data["y_scores"], frequencies=data["y_frequencies"]
    )
    x_smooth = presmooth(x, degree=reference["degree"]).distribution
    y_smooth = presmooth(y, degree=reference["degree"]).distribution
    assert isinstance(x_smooth, ScoreDistribution)
    assert isinstance(y_smooth, ScoreDistribution)
    result = equate(x_smooth, y_smooth, method="equipercentile")
    np.testing.assert_allclose(
        result.equated, reference["equated"], rtol=0, atol=SMOOTHED_EQUATING_TOL
    )
