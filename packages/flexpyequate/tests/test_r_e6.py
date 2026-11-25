"""E6 against R: kernel equating for SG, CB and anchor-test designs (DESIGN.md §7, case Q8).

Reference values (``validation/r/equate_e6_reference.R``): kequate with its N / (N - 1) variance
factor removed (``kequate_book``), the ordinary kequate call (``kequate_default``), and the design
functions of von Davier, Holland & Thayer (2004) written out in R with a finite-difference SEE
(``independent``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from flexpyequate import (
    BivariateScoreDistribution,
    KernelEquatingResult,
    LoglinearResult,
    kernel_equate,
    penalty_bandwidth,
    presmooth,
)

from ._reference import load_reference

REFERENCE = load_reference("e6_kernel_designs")
SG = REFERENCE["single_group"]
CB = REFERENCE["counterbalanced"]
NEAT = REFERENCE["neat"]
PSE = NEAT["post_stratification"]
CE = NEAT["chained"]

PROBABILITY_RTOL = 1e-8
INDEPENDENT_EQUATED_TOL = 1e-9
KEQUATE_EQUATED_TOL = 1e-6  # kequate: uniroot(tol = 1e-6)
SEE_RTOL = 1e-6
SEED_RTOL = 1e-5
LINEAR_EQUATED_TOL = 1e-6
PRE_TOL = 1e-5
BANDWIDTH_TOL = 1e-6
DEFAULT_EQUATED_TOL = 5e-3  # ordinary kequate call: N / (N - 1) variance factor
DEFAULT_SEE_RTOL = 1e-3


def _smooth(
    x_scores: Any, y_scores: Any, frequencies: Any, degree: Any, cross_terms: Any
) -> LoglinearResult:
    joint = BivariateScoreDistribution.from_frequencies(
        x_scores=x_scores, y_scores=y_scores, frequencies=frequencies
    )
    return presmooth(
        joint,
        degree=(int(degree[0]), int(degree[1])),
        cross_terms=[(int(a), int(b)) for a, b in cross_terms],
    )


def _single_group() -> LoglinearResult:
    return _smooth(
        SG["x_scores"], SG["y_scores"], SG["joint_frequencies"], SG["degree"], SG["cross_terms"]
    )


def _counterbalanced() -> tuple[LoglinearResult, LoglinearResult]:
    return (
        _smooth(
            CB["x_scores"], CB["y_scores"], CB["frequencies_12"], CB["degree"], CB["cross_terms"]
        ),
        _smooth(
            CB["x_scores"], CB["y_scores"], CB["frequencies_21"], CB["degree"], CB["cross_terms"]
        ),
    )


def _neat() -> tuple[LoglinearResult, LoglinearResult]:
    return (
        _smooth(
            NEAT["x_scores"],
            NEAT["anchor_scores"],
            NEAT["x_joint_frequencies"],
            NEAT["x_degree"],
            NEAT["cross_terms"],
        ),
        _smooth(
            NEAT["y_scores"],
            NEAT["anchor_scores"],
            NEAT["y_joint_frequencies"],
            NEAT["y_degree"],
            NEAT["cross_terms"],
        ),
    )


def _check(
    result: KernelEquatingResult,
    linear: KernelEquatingResult | None,
    case: dict[str, Any],
) -> None:
    independent = case["independent"]
    np.testing.assert_allclose(
        result.equated, independent["equated"], rtol=0, atol=INDEPENDENT_EQUATED_TOL
    )
    np.testing.assert_allclose(result.see, independent["see"], rtol=SEE_RTOL)
    if "kequate_book" in case:
        book = case["kequate_book"]
        np.testing.assert_allclose(
            result.equated, book["equated"], rtol=0, atol=KEQUATE_EQUATED_TOL
        )
        np.testing.assert_allclose(result.see, book["see"], rtol=SEE_RTOL)
        assert linear is not None
        np.testing.assert_allclose(
            linear.equated, book["linear_equated"], rtol=0, atol=LINEAR_EQUATED_TOL
        )
        np.testing.assert_allclose(linear.see, book["linear_see"], rtol=SEED_RTOL)
        np.testing.assert_allclose(result.seed(linear)["seed"], book["seed_linear"], rtol=SEED_RTOL)
    if "kequate_default" in case:
        default = case["kequate_default"]
        np.testing.assert_allclose(
            result.equated, default["equated"], rtol=0, atol=DEFAULT_EQUATED_TOL
        )
        np.testing.assert_allclose(result.see, default["see"], rtol=DEFAULT_SEE_RTOL)


def test_reference_metadata() -> None:
    assert REFERENCE["meta"]["kequate_version"]
    assert REFERENCE["meta"]["sample_size_factor"] == 1e8


def test_fitted_probabilities() -> None:
    np.testing.assert_allclose(
        _single_group().distribution.probabilities,
        SG["fitted_probabilities"],
        rtol=PROBABILITY_RTOL,
    )
    x, y = _neat()
    np.testing.assert_allclose(
        x.distribution.probabilities, NEAT["x_fitted_probabilities"], rtol=PROBABILITY_RTOL
    )
    np.testing.assert_allclose(
        y.distribution.probabilities, NEAT["y_fitted_probabilities"], rtol=PROBABILITY_RTOL
    )


def test_single_group() -> None:
    smooth = _single_group()
    result = kernel_equate(smooth, bandwidth_x=SG["bandwidth_x"], bandwidth_y=SG["bandwidth_y"])
    assert result.design == "single_group"
    _check(result, kernel_equate(smooth, linear=True), SG)
    np.testing.assert_allclose(
        result.percent_relative_error, SG["kequate_book"]["pre"], rtol=0, atol=PRE_TOL
    )
    automatic = kernel_equate(smooth, penalty_weight=0)
    assert automatic.bandwidth_x == pytest.approx(
        SG["kequate_book"]["penalty_bandwidth_x"], abs=BANDWIDTH_TOL
    )
    assert automatic.bandwidth_y == pytest.approx(
        SG["kequate_book"]["penalty_bandwidth_y"], abs=BANDWIDTH_TOL
    )


def test_single_group_without_presmoothing() -> None:
    joint = BivariateScoreDistribution.from_frequencies(
        x_scores=SG["x_scores"], y_scores=SG["y_scores"], frequencies=SG["joint_frequencies"]
    )
    result = kernel_equate(joint, bandwidth_x=SG["bandwidth_x"], bandwidth_y=SG["bandwidth_y"])
    assert not result.x_presmoothed
    _check(result, None, {"independent": SG["observed"]})


@pytest.mark.parametrize("index", range(len(CB["cases"])))
def test_counterbalanced(index: int) -> None:
    case = CB["cases"][index]
    first, second = _counterbalanced()
    weights = (float(case["weights"][0]), float(case["weights"][1]))
    result = kernel_equate(
        first,
        second,
        design="counterbalanced",
        counterbalance_weights=weights,
        bandwidth_x=CB["bandwidth_x"],
        bandwidth_y=CB["bandwidth_y"],
    )
    linear = kernel_equate(
        first, second, design="counterbalanced", counterbalance_weights=weights, linear=True
    )
    _check(result, linear, case)


@pytest.mark.parametrize("index", range(len(PSE["cases"])))
def test_post_stratification(index: int) -> None:
    case = PSE["cases"][index]
    x, y = _neat()
    options: dict[str, Any] = {"design": "post_stratification", "synthetic_weight": case["weight"]}
    result = kernel_equate(
        x, y, bandwidth_x=PSE["bandwidth_x"], bandwidth_y=PSE["bandwidth_y"], **options
    )
    np.testing.assert_allclose(result.x.probabilities, case["r"], rtol=PROBABILITY_RTOL)
    np.testing.assert_allclose(result.y.probabilities, case["s"], rtol=PROBABILITY_RTOL)
    _check(result, kernel_equate(x, y, linear=True, **options), case)
    book = case["kequate_book"]
    np.testing.assert_allclose(result.percent_relative_error, book["pre"], rtol=0, atol=PRE_TOL)
    automatic = kernel_equate(x, y, penalty_weight=0, **options)
    assert automatic.bandwidth_x == pytest.approx(book["penalty_bandwidth_x"], abs=BANDWIDTH_TOL)
    assert automatic.bandwidth_y == pytest.approx(book["penalty_bandwidth_y"], abs=BANDWIDTH_TOL)


def test_chained() -> None:
    x, y = _neat()
    h = CE["bandwidths"]
    result = kernel_equate(
        x,
        y,
        design="chained",
        bandwidth_x=h["x"],
        bandwidth_y=h["y"],
        anchor_bandwidths=(h["anchor_x"], h["anchor_y"]),
    )
    _check(result, kernel_equate(x, y, design="chained", linear=True), CE)
    book = CE["kequate_book"]
    pre = result.chain_percent_relative_error
    np.testing.assert_allclose(pre["x_to_anchor"], book["pre_x_to_anchor"], rtol=0, atol=PRE_TOL)
    np.testing.assert_allclose(pre["anchor_to_y"], book["pre_anchor_to_y"], rtol=0, atol=PRE_TOL)
    automatic = kernel_equate(x, y, design="chained", penalty_weight=0)
    selected = book["penalty_bandwidths"]
    assert automatic.bandwidth_x == pytest.approx(selected["x"], abs=BANDWIDTH_TOL)
    assert automatic.bandwidth_y == pytest.approx(selected["y"], abs=BANDWIDTH_TOL)
    assert automatic.anchor_bandwidths == pytest.approx(
        (selected["anchor_x"], selected["anchor_y"]), abs=BANDWIDTH_TOL
    )
    assert automatic.anchor_distributions is not None
    assert automatic.anchor_bandwidths is not None
    assert automatic.anchor_bandwidths[0] == pytest.approx(
        penalty_bandwidth(automatic.anchor_distributions[0], penalty_weight=0)
    )


def test_counterbalanced_kequate_with_unequal_weight() -> None:
    # kequate's single weight wcb gives r and s of the weights (wcb, wcb), but its SEE weights
    # the Y margins the other way round and is therefore not the SEE of that equating function
    # unless wcb = 1/2 (DESIGN.md §7, known differences). flexpyequate's SEE for (0.3, 0.3)
    # agrees with the finite-difference SEE (test_counterbalanced); kequate's does not.
    first, second = _counterbalanced()
    result = kernel_equate(
        first,
        second,
        design="counterbalanced",
        counterbalance_weights=(0.3, 0.3),
        bandwidth_x=CB["bandwidth_x"],
        bandwidth_y=CB["bandwidth_y"],
    )
    kequate = CB["kequate_wcb_0.3"]
    np.testing.assert_allclose(result.equated, kequate["equated"], rtol=0, atol=KEQUATE_EQUATED_TOL)
    assert np.max(np.abs(np.asarray(kequate["see"]) / result.see - 1)) > 0.1
