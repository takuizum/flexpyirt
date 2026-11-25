"""E3 against R: kernel equating for equivalent groups (DESIGN.md §7, cases Q4 and Q5).

Reference values (``validation/r/equate_e3_reference.R``):

- ``kequate_book``: kequate with sample sizes times 1e8, which removes its N / (N - 1) variance
  factor (flexpyequate follows von Davier, Holland & Thayer, 2004); SEE rescaled.
- ``kequate_default``: the ordinary kequate call.
- ``independent``: the formulas of the book in R with a strict root tolerance and a
  finite-difference SEE.
- ``snsequate``: SNSequate, which uses the book's variance and second penalty term.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from flexpyequate import (
    KernelEquatingResult,
    LoglinearResult,
    ScoreDistribution,
    kernel_equate,
    penalty_bandwidth,
    presmooth,
)
from flexpyequate.kernel import _penalty

from ._reference import cases_by_name, load_reference

CASES = cases_by_name("e3_kernel")
NAMES = sorted(CASES)
SMOOTHED = [name for name in NAMES if "degree_x" in CASES[name]]

PROBABILITY_RTOL = 1e-8
CDF_TOL = 1e-12
INDEPENDENT_EQUATED_TOL = 1e-9
KEQUATE_EQUATED_TOL = 1e-6  # kequate: uniroot(tol = 1e-6)
SEE_RTOL = 1e-6
PRE_TOL = 1e-5
BANDWIDTH_TOL = 1e-6
PENALTY_RTOL = 1e-10
# The ordinary kequate call differs by its N / (N - 1) variance factor.
DEFAULT_EQUATED_TOL = 5e-3
DEFAULT_SEE_RTOL = 1e-3
DEFAULT_BANDWIDTH_TOL = 1e-4
SNSEQUATE_EQUATED_TOL = 1e-4
SNSEQUATE_SEE_RTOL = 1e-4
SNSEQUATE_BANDWIDTH_TOL = 1e-3  # optim() from a fixed start, default tolerance


def _inputs(
    case: dict[str, Any],
) -> tuple[ScoreDistribution | LoglinearResult, ScoreDistribution | LoglinearResult]:
    x = ScoreDistribution.from_frequencies(
        scores=case["x_scores"], frequencies=case["x_frequencies"]
    )
    y = ScoreDistribution.from_frequencies(
        scores=case["y_scores"], frequencies=case["y_frequencies"]
    )
    if "degree_x" not in case:
        return x, y
    return presmooth(x, degree=case["degree_x"]), presmooth(y, degree=case["degree_y"])


def _fixed(case: dict[str, Any]) -> KernelEquatingResult:
    x, y = _inputs(case)
    return kernel_equate(x, y, bandwidth_x=case["bandwidth_x"], bandwidth_y=case["bandwidth_y"])


def test_reference_metadata() -> None:
    meta = load_reference("e3_kernel")["meta"]
    assert meta["kequate_version"] and meta["snsequate_version"]
    assert meta["sample_size_factor"] == 1e8
    assert len(SMOOTHED) == 3 and len(NAMES) == 4


@pytest.mark.parametrize("name", NAMES)
def test_score_probabilities_and_covariance(name: str) -> None:
    case = CASES[name]
    result = _fixed(case)
    np.testing.assert_allclose(result.x.probabilities, case["r"], rtol=PROBABILITY_RTOL)
    np.testing.assert_allclose(result.y.probabilities, case["s"], rtol=PROBABILITY_RTOL)
    if "covariance_r" in case:
        x, _ = _inputs(case)
        assert isinstance(x, LoglinearResult)
        covariance = x.c_matrix @ x.c_matrix.T / x.observed.n
        reference = np.asarray(case["covariance_r"])
        np.testing.assert_allclose(
            covariance, reference, rtol=0, atol=1e-9 * np.abs(reference).max()
        )


@pytest.mark.parametrize("name", NAMES)
def test_continuized_cdfs(name: str) -> None:
    case = CASES[name]
    continuized = _fixed(case).continuized
    reference = case["kequate_book"]
    np.testing.assert_allclose(
        continuized.query("form == 'x'")["cdf"], reference["cdf_x"], rtol=0, atol=CDF_TOL
    )
    np.testing.assert_allclose(
        continuized.query("form == 'y'")["cdf"], reference["cdf_y"], rtol=0, atol=CDF_TOL
    )
    np.testing.assert_allclose(
        continuized.query("form == 'x'")["density"],
        case["independent"]["density_x"],
        rtol=0,
        atol=CDF_TOL,
    )


@pytest.mark.parametrize("name", NAMES)
def test_equated_scores(name: str) -> None:
    case = CASES[name]
    result = _fixed(case)
    np.testing.assert_allclose(
        result.equated, case["independent"]["equated"], rtol=0, atol=INDEPENDENT_EQUATED_TOL
    )
    np.testing.assert_allclose(
        result.equated, case["kequate_book"]["equated"], rtol=0, atol=KEQUATE_EQUATED_TOL
    )
    np.testing.assert_allclose(
        result.equated, case["kequate_default"]["equated"], rtol=0, atol=DEFAULT_EQUATED_TOL
    )


@pytest.mark.parametrize("name", NAMES)
def test_standard_errors(name: str) -> None:
    case = CASES[name]
    result = _fixed(case)
    np.testing.assert_allclose(result.see, case["kequate_book"]["see"], rtol=SEE_RTOL)
    np.testing.assert_allclose(result.see, case["independent"]["see"], rtol=SEE_RTOL)
    np.testing.assert_allclose(result.see, case["kequate_default"]["see"], rtol=DEFAULT_SEE_RTOL)


@pytest.mark.parametrize("name", NAMES)
def test_percent_relative_error(name: str) -> None:
    case = CASES[name]
    np.testing.assert_allclose(
        _fixed(case).percent_relative_error, case["kequate_book"]["pre"], rtol=0, atol=PRE_TOL
    )


@pytest.mark.parametrize("name", NAMES)
def test_penalty_function(name: str) -> None:
    case = CASES[name]
    result = _fixed(case)
    reference = case["independent"]
    scores = np.asarray(case["x_scores"], dtype=np.float64)
    for weight, key in ((0.0, "penalty_x_weight0"), (1.0, "penalty_x_weight1")):
        values = [
            _penalty(h, scores, result.x.probabilities, weight) for h in reference["penalty_grid"]
        ]
        np.testing.assert_allclose(values, reference[key], rtol=PENALTY_RTOL)


@pytest.mark.parametrize("name", NAMES)
def test_penalty_bandwidths(name: str) -> None:
    case = CASES[name]
    x, y = _inputs(case)
    independent = case["independent"]
    for weight in (0, 1):
        h_x = penalty_bandwidth(x, penalty_weight=weight)
        h_y = penalty_bandwidth(y, penalty_weight=weight)
        assert h_x == pytest.approx(
            independent[f"penalty_bandwidth_x_weight{weight}"], abs=BANDWIDTH_TOL
        )
        assert h_y == pytest.approx(
            independent[f"penalty_bandwidth_y_weight{weight}"], abs=BANDWIDTH_TOL
        )
    # kequate minimizes the first penalty term only (KPEN = 0).
    h_x = penalty_bandwidth(x, penalty_weight=0)
    h_y = penalty_bandwidth(y, penalty_weight=0)
    assert h_x == pytest.approx(case["kequate_book"]["penalty_bandwidth_x"], abs=BANDWIDTH_TOL)
    assert h_y == pytest.approx(case["kequate_book"]["penalty_bandwidth_y"], abs=BANDWIDTH_TOL)
    assert h_x == pytest.approx(
        case["kequate_default"]["penalty_bandwidth_x"], abs=DEFAULT_BANDWIDTH_TOL
    )
    assert h_y == pytest.approx(
        case["kequate_default"]["penalty_bandwidth_y"], abs=DEFAULT_BANDWIDTH_TOL
    )


@pytest.mark.parametrize("name", NAMES)
def test_default_call_against_the_default_kequate_call(name: str) -> None:
    # End to end with selected bandwidths, as a user would run both packages (KPEN = 0).
    case = CASES[name]
    x, y = _inputs(case)
    result = kernel_equate(x, y, penalty_weight=0)
    reference = case["kequate_default"]
    np.testing.assert_allclose(
        result.equated, reference["auto_equated"], rtol=0, atol=DEFAULT_EQUATED_TOL
    )
    np.testing.assert_allclose(result.see, reference["auto_see"], rtol=DEFAULT_SEE_RTOL)


@pytest.mark.parametrize("name", [name for name in NAMES if "snsequate" in CASES[name]])
def test_snsequate(name: str) -> None:
    case = CASES[name]
    reference = case["snsequate"]
    x, y = _inputs(case)
    assert penalty_bandwidth(x, penalty_weight=1) == pytest.approx(
        reference["penalty_bandwidth_x_weight1"], abs=SNSEQUATE_BANDWIDTH_TOL
    )
    assert penalty_bandwidth(y, penalty_weight=1) == pytest.approx(
        reference["penalty_bandwidth_y_weight1"], abs=SNSEQUATE_BANDWIDTH_TOL
    )
    assert penalty_bandwidth(x, penalty_weight=0) == pytest.approx(
        reference["penalty_bandwidth_x_weight0"], abs=SNSEQUATE_BANDWIDTH_TOL
    )
    if reference.get("equated"):
        result = _fixed(case)
        np.testing.assert_allclose(
            result.equated, reference["equated"], rtol=0, atol=SNSEQUATE_EQUATED_TOL
        )
        np.testing.assert_allclose(result.see, reference["see"], rtol=SNSEQUATE_SEE_RTOL)


def test_snsequate_equating_is_available_for_one_case() -> None:
    # SNSequate's ker.eq() fails on the small-sample case (root bracket); keep one that works.
    assert len(CASES["act_math_smoothed"]["snsequate"]["equated"]) == 41
