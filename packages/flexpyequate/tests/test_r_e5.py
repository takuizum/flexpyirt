"""E5 against R: equipercentile equating with an anchor test (DESIGN.md §7, case Q7).

Reference values (``validation/r/equate_e5_reference.R``): R ``equate`` and the formulas of Kolen &
Brennan (2014) written out in R without rounding.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from flexpyequate import BivariateScoreDistribution, equate, presmooth

from ._reference import load_reference

REFERENCE = load_reference("e5_neat_equipercentile")
CASES = {case["name"]: case for case in REFERENCE["cases"]}
NAMES = sorted(CASES)
WEIGHTED = [
    (name, index) for name in NAMES for index in range(len(CASES[name]["frequency_estimation"]))
]
WEIGHTED_IDS = [
    f"{name}-w{CASES[name]['frequency_estimation'][index]['weight']:.3f}"
    for name, index in WEIGHTED
]
SMOOTHED = REFERENCE["smoothed"]

KB_TOL = 1e-10  # independent, unrounded R implementation
EQUATE_TOL = 5e-7  # equate rounds ranks to 10 decimals
PROBABILITY_TOL = 1e-13
SMOOTHED_EQUATE_TOL = 5e-6  # rounding divided by very small smoothed probabilities
SMOOTHED_END_TO_END_TOL = 5e-6  # same, plus equate's glm at its default tolerance


def _joints(
    case: dict[str, Any], x_key: str = "x_joint_frequencies", y_key: str = "y_joint_frequencies"
) -> tuple[BivariateScoreDistribution, BivariateScoreDistribution]:
    x = BivariateScoreDistribution.from_frequencies(
        x_scores=case["x_scores"], y_scores=case["anchor_scores"], frequencies=case[x_key]
    )
    y = BivariateScoreDistribution.from_frequencies(
        x_scores=case["y_scores"], y_scores=case["anchor_scores"], frequencies=case[y_key]
    )
    return x, y


def _inside(*ranks: Any) -> np.ndarray:
    """Scores at which no percentile rank is exactly 0 or 1 (where equate deviates)."""
    keep = np.ones(len(ranks[0]), dtype=bool)
    for rank in ranks:
        values = np.asarray(rank, dtype=np.float64)
        keep &= (values > 0) & (values < 1)
    return keep


def test_reference_metadata() -> None:
    assert REFERENCE["meta"]["equate_version"]
    assert NAMES == ["external_anchor", "kb_neat"]


@pytest.mark.parametrize(("name", "index"), WEIGHTED, ids=WEIGHTED_IDS)
def test_frequency_estimation(name: str, index: int) -> None:
    case = CASES[name]
    entry = case["frequency_estimation"][index]
    x, y = _joints(case)
    result = equate(x, y, method="frequency_estimation", synthetic_weight=entry["weight"])
    reference = entry["kb"]
    np.testing.assert_allclose(result.equated, reference["equated"], rtol=0, atol=KB_TOL)
    np.testing.assert_allclose(
        result.x.probabilities, reference["synthetic_x"], rtol=0, atol=PROBABILITY_TOL
    )
    np.testing.assert_allclose(
        result.y.probabilities, reference["synthetic_y"], rtol=0, atol=PROBABILITY_TOL
    )
    assert result.x.n == pytest.approx(x.n + y.n)

    # equate at the weight its count-based synthetic distributions actually use.
    at_effective = equate(
        x, y, method="frequency_estimation", synthetic_weight=entry["effective_weight"]
    )
    effective = entry["kb_at_effective_weight"]
    np.testing.assert_allclose(at_effective.equated, effective["equated"], rtol=0, atol=KB_TOL)
    keep = _inside(effective["rank_x"])
    assert keep.sum() >= 30
    np.testing.assert_allclose(
        at_effective.equated[keep],
        np.asarray(entry["equate"]["equated"])[keep],
        rtol=0,
        atol=EQUATE_TOL,
    )


@pytest.mark.parametrize("name", NAMES)
def test_frequency_estimation_default_weight(name: str) -> None:
    case = CASES[name]
    entry = case["frequency_estimation"][0]
    x, y = _joints(case)
    result = equate(x, y, method="frequency_estimation")
    assert result.synthetic_weight == pytest.approx(entry["weight"], abs=1e-15)
    np.testing.assert_allclose(result.equated, entry["kb"]["equated"], rtol=0, atol=KB_TOL)


@pytest.mark.parametrize("name", NAMES)
def test_chained_equipercentile(name: str) -> None:
    case = CASES[name]
    x, y = _joints(case)
    result = equate(x, y, method="chained_equipercentile")
    reference = case["chained"]["kb"]
    np.testing.assert_allclose(result.equated, reference["equated"], rtol=0, atol=KB_TOL)
    keep = _inside(reference["rank_x"], reference["rank_anchor"])
    assert keep.sum() >= 30
    np.testing.assert_allclose(
        result.equated[keep],
        np.asarray(case["chained"]["equate"]["equated"])[keep],
        rtol=0,
        atol=EQUATE_TOL,
    )


@pytest.mark.parametrize("method", ["frequency_estimation", "chained"])
def test_presmoothed_on_the_tables_smoothed_by_equate(method: str) -> None:
    x, y = _joints(SMOOTHED, "x_smoothed", "y_smoothed")
    if method == "chained":
        result = equate(x, y, method="chained_equipercentile")
    else:
        result = equate(x, y, method="frequency_estimation", synthetic_weight=SMOOTHED["weight"])
    reference = SMOOTHED[method]
    np.testing.assert_allclose(
        result.equated, reference["kb_on_equate_smoothing"]["equated"], rtol=0, atol=KB_TOL
    )
    np.testing.assert_allclose(
        result.equated, reference["equate"]["equated"], rtol=0, atol=SMOOTHED_EQUATE_TOL
    )


@pytest.mark.parametrize("method", ["frequency_estimation", "chained"])
def test_presmoothed_end_to_end(method: str) -> None:
    # flexpyequate's own presmoothing, then equating, against equate doing both.
    x, y = _joints(SMOOTHED)
    degree = (int(SMOOTHED["degree"][0]), int(SMOOTHED["degree"][1]))
    cross_terms = [(int(a), int(b)) for a, b in SMOOTHED["cross_terms"]]
    x_smooth = presmooth(x, degree=degree, cross_terms=cross_terms).distribution
    y_smooth = presmooth(y, degree=degree, cross_terms=cross_terms).distribution
    assert isinstance(x_smooth, BivariateScoreDistribution)
    assert isinstance(y_smooth, BivariateScoreDistribution)
    if method == "chained":
        result = equate(x_smooth, y_smooth, method="chained_equipercentile")
    else:
        result = equate(
            x_smooth, y_smooth, method="frequency_estimation", synthetic_weight=SMOOTHED["weight"]
        )
    np.testing.assert_allclose(
        result.equated, SMOOTHED[method]["equate"]["equated"], rtol=0, atol=SMOOTHED_END_TO_END_TOL
    )
