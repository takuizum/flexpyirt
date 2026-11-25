"""E7 against R: logistic and uniform kernels, further bandwidth rules (DESIGN.md §7, Q11, Q12).

Reference values (``validation/r/equate_e7_kernel_reference.R``), for the data and log-linear
models of the E3 and E6 reference files: kequate with its N / (N - 1) variance factor removed
(``kequate_book``), the continuization for a general kernel written out in R with a
finite-difference SEE (``independent``), and ``SNSequate::bandwidth``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from flexpyequate import (
    BivariateScoreDistribution,
    KernelEquatingResult,
    LoglinearResult,
    ScoreDistribution,
    kernel_equate,
    penalty_bandwidth,
    presmooth,
)
from flexpyequate.kernel import (
    _cross_validation,
    _double_smoothing,
    _penalty,
    _select_bandwidth,
)

from ._reference import cases_by_name, load_reference

REFERENCE = load_reference("e7_kernels")
ACT = cases_by_name("e3_kernel")["act_math_smoothed"]
E6 = load_reference("e6_kernel_designs")
KERNELS = ["logistic", "uniform"]
RULES = REFERENCE["bandwidth_rules"]
RULE_CASES = {case["name"]: case for case in RULES["cases"]}

INDEPENDENT_EQUATED_TOL = 1e-9
KEQUATE_EQUATED_TOL = 1e-6  # kequate: uniroot(tol = 1e-6)
SEE_RTOL = 1e-6
SEED_RTOL = 1e-5
CDF_TOL = 1e-12
PRE_TOL = 5e-5  # tenth moments of kequate's equated scores (root tolerance 1e-6)
PENALTY_RTOL = 1e-10
BANDWIDTH_TOL = 1e-6
SNSEQUATE_BANDWIDTH_TOL = 1e-5  # optim(method = "Brent") started at 0.65
CRITERION_RTOL = 1e-10
CROSS_VALIDATION_BANDWIDTH_TOL = 1e-5  # a flat criterion against R's optimize()
RULE_OF_THUMB_RTOL = 1e-10
END_TO_END_SEE_RTOL = 1e-5


def _relative(values: Any, reference: Any) -> float:
    return float(np.max(np.abs(np.asarray(values) / np.asarray(reference) - 1)))


def _equivalent_groups() -> tuple[LoglinearResult, LoglinearResult]:
    x = ScoreDistribution.from_frequencies(scores=ACT["x_scores"], frequencies=ACT["x_frequencies"])
    y = ScoreDistribution.from_frequencies(scores=ACT["y_scores"], frequencies=ACT["y_frequencies"])
    return presmooth(x, degree=ACT["degree_x"]), presmooth(y, degree=ACT["degree_y"])


def _smooth(x_scores: Any, y_scores: Any, frequencies: Any, degree: Any) -> LoglinearResult:
    joint = BivariateScoreDistribution.from_frequencies(
        x_scores=x_scores, y_scores=y_scores, frequencies=frequencies
    )
    return presmooth(joint, degree=(int(degree[0]), int(degree[1])), cross_terms=[(1, 1)])


def _single_group() -> LoglinearResult:
    data = E6["single_group"]
    return _smooth(data["x_scores"], data["y_scores"], data["joint_frequencies"], data["degree"])


def _counterbalanced() -> tuple[LoglinearResult, LoglinearResult]:
    data = E6["counterbalanced"]
    return (
        _smooth(data["x_scores"], data["y_scores"], data["frequencies_12"], data["degree"]),
        _smooth(data["x_scores"], data["y_scores"], data["frequencies_21"], data["degree"]),
    )


def _neat() -> tuple[LoglinearResult, LoglinearResult]:
    data = E6["neat"]
    return (
        _smooth(
            data["x_scores"], data["anchor_scores"], data["x_joint_frequencies"], data["x_degree"]
        ),
        _smooth(
            data["y_scores"], data["anchor_scores"], data["y_joint_frequencies"], data["y_degree"]
        ),
    )


def _check(result: KernelEquatingResult, reference: dict[str, Any]) -> None:
    independent = reference["independent"]
    np.testing.assert_allclose(
        result.equated, independent["equated"], rtol=0, atol=INDEPENDENT_EQUATED_TOL
    )
    assert _relative(result.see, independent["see"]) < SEE_RTOL
    if "kequate_book" in reference:
        book = reference["kequate_book"]
        np.testing.assert_allclose(
            result.equated, book["equated"], rtol=0, atol=KEQUATE_EQUATED_TOL
        )
        assert _relative(result.see, book["see"]) < SEE_RTOL
        if "pre" in book:
            np.testing.assert_allclose(
                result.percent_relative_error, book["pre"], rtol=0, atol=PRE_TOL
            )


def _check_seed(
    result: KernelEquatingResult, linear: KernelEquatingResult, book: dict[str, Any]
) -> None:
    """kequate's SEED is against its Gaussian linear function, whatever the kernel."""
    np.testing.assert_allclose(
        linear.equated, book["linear_equated"], rtol=0, atol=KEQUATE_EQUATED_TOL
    )
    assert _relative(result.seed(linear)["seed"], book["seed_gaussian_linear"]) < SEED_RTOL


def test_reference_metadata() -> None:
    assert REFERENCE["meta"]["kequate_version"]
    assert REFERENCE["meta"]["snsequate_version"]
    assert RULES["pilot_bandwidth"] == 4


# ----------------------------------------------------------------------------- kernels
@pytest.mark.parametrize("kernel", KERNELS)
def test_equivalent_groups(kernel: str) -> None:
    reference = REFERENCE["equivalent_groups"][kernel]
    x, y = _equivalent_groups()
    result = kernel_equate(
        x,
        y,
        bandwidth_x=reference["bandwidth_x"],
        bandwidth_y=reference["bandwidth_y"],
        kernel=kernel,  # type: ignore[arg-type]
    )
    _check(result, reference)
    _check_seed(result, kernel_equate(x, y, linear=True), reference["kequate_book"])
    continuized = result.continuized
    form_x = continuized[continuized["form"] == "x"]
    form_y = continuized[continuized["form"] == "y"]
    book, independent = reference["kequate_book"], reference["independent"]
    np.testing.assert_allclose(form_x["cdf"], book["cdf_x"], rtol=0, atol=1e-11)
    np.testing.assert_allclose(form_y["cdf"], book["cdf_y"], rtol=0, atol=1e-11)
    np.testing.assert_allclose(form_x["cdf"], independent["cdf_x"], rtol=0, atol=CDF_TOL)
    np.testing.assert_allclose(form_x["density"], independent["density_x"], rtol=0, atol=CDF_TOL)


@pytest.mark.parametrize("kernel", KERNELS)
def test_penalty(kernel: str) -> None:
    reference = REFERENCE["equivalent_groups"][kernel]
    x, y = _equivalent_groups()
    scores, probabilities = x.distribution.scores, x.distribution.probabilities
    assert isinstance(x.distribution, ScoreDistribution)
    grid = REFERENCE["equivalent_groups"]["penalty_grid"]
    for weight, source, key in (
        (0.0, "kequate_book", "penalty_x_weight0"),
        (0.0, "independent", "penalty_x_weight0"),
        (1.0, "independent", "penalty_x_weight1"),
    ):
        values = [_penalty(h, scores, probabilities, weight, kernel) for h in grid]  # type: ignore[arg-type]
        assert _relative(values, reference[source][key]) < PENALTY_RTOL
    for weight in (0, 1):
        selected = (
            penalty_bandwidth(x, penalty_weight=weight, kernel=kernel),  # type: ignore[arg-type]
            penalty_bandwidth(y, penalty_weight=weight, kernel=kernel),  # type: ignore[arg-type]
        )
        independent = reference["independent"]
        assert selected[0] == pytest.approx(
            independent[f"penalty_bandwidth_x_weight{weight}"], abs=BANDWIDTH_TOL
        )
        assert selected[1] == pytest.approx(
            independent[f"penalty_bandwidth_y_weight{weight}"], abs=BANDWIDTH_TOL
        )
        if weight == 0:
            book = reference["kequate_book"]
            assert selected[0] == pytest.approx(book["penalty_bandwidth_x"], abs=BANDWIDTH_TOL)
            assert selected[1] == pytest.approx(book["penalty_bandwidth_y"], abs=BANDWIDTH_TOL)
        else:
            sns = reference["snsequate"]
            assert selected[0] == pytest.approx(
                sns["penalty_bandwidth_x_weight1"], abs=SNSEQUATE_BANDWIDTH_TOL
            )
            assert selected[1] == pytest.approx(
                sns["penalty_bandwidth_y_weight1"], abs=SNSEQUATE_BANDWIDTH_TOL
            )


@pytest.mark.parametrize("kernel", KERNELS)
def test_single_group(kernel: str) -> None:
    reference = REFERENCE["single_group"][kernel]
    joint = _single_group()
    result = kernel_equate(
        joint,
        bandwidth_x=reference["bandwidth_x"],
        bandwidth_y=reference["bandwidth_y"],
        kernel=kernel,  # type: ignore[arg-type]
    )
    _check(result, reference)
    _check_seed(result, kernel_equate(joint, linear=True), reference["kequate_book"])


@pytest.mark.parametrize("kernel", KERNELS)
def test_counterbalanced(kernel: str) -> None:
    reference = REFERENCE["counterbalanced"][kernel]
    first, second = _counterbalanced()
    for case in reference["cases"]:
        weights = (float(case["weights"][0]), float(case["weights"][1]))
        result = kernel_equate(
            first,
            second,
            design="counterbalanced",
            counterbalance_weights=weights,
            bandwidth_x=reference["bandwidth_x"],
            bandwidth_y=reference["bandwidth_y"],
            kernel=kernel,  # type: ignore[arg-type]
        )
        _check(result, case)
        if "kequate_book" in case:
            linear = kernel_equate(first, second, design="counterbalanced", linear=True)
            _check_seed(result, linear, case["kequate_book"])


@pytest.mark.parametrize("kernel", KERNELS)
def test_post_stratification(kernel: str) -> None:
    reference = REFERENCE["neat"][kernel]
    xa, ya = _neat()
    for case in reference["post_stratification"]:
        options: dict[str, Any] = {
            "design": "post_stratification",
            "synthetic_weight": case["weight"],
        }
        result = kernel_equate(
            xa,
            ya,
            bandwidth_x=reference["bandwidths"]["x"],
            bandwidth_y=reference["bandwidths"]["y"],
            kernel=kernel,  # type: ignore[arg-type]
            **options,
        )
        _check(result, case)
        _check_seed(result, kernel_equate(xa, ya, linear=True, **options), case["kequate_book"])


@pytest.mark.parametrize("kernel", KERNELS)
def test_chained(kernel: str) -> None:
    reference = REFERENCE["neat"][kernel]
    bandwidths = reference["bandwidths"]
    xa, ya = _neat()
    result = kernel_equate(
        xa,
        ya,
        design="chained",
        bandwidth_x=bandwidths["x"],
        bandwidth_y=bandwidths["y"],
        anchor_bandwidths=(bandwidths["anchor_x"], bandwidths["anchor_y"]),
        kernel=kernel,  # type: ignore[arg-type]
    )
    _check(result, reference["chained"])
    book = reference["chained"]["kequate_book"]
    pre = result.chain_percent_relative_error
    np.testing.assert_allclose(pre["x_to_anchor"], book["pre_x_to_anchor"], rtol=0, atol=PRE_TOL)
    np.testing.assert_allclose(pre["anchor_to_y"], book["pre_anchor_to_y"], rtol=0, atol=PRE_TOL)


# ----------------------------------------------------------------------------- bandwidth rules
@pytest.mark.parametrize("name", sorted(RULE_CASES))
def test_bandwidth_rules(name: str) -> None:
    case = RULE_CASES[name]
    scores = np.asarray(case["scores"], dtype=np.float64)
    probabilities = np.asarray(case["probabilities"], dtype=np.float64)
    n = float(case["n"])
    kequate, independent = case["kequate"], case["independent"]

    double_smoothing = [
        _double_smoothing(h, scores, probabilities, RULES["pilot_bandwidth"]) for h in case["grid"]
    ]
    cross_validation = [_cross_validation(h, scores, probabilities) for h in case["grid"]]
    for reference in (kequate, independent):
        assert _relative(double_smoothing, reference["double_smoothing_values"]) < CRITERION_RTOL
        assert _relative(cross_validation, reference["cross_validation_values"]) < CRITERION_RTOL

    def select(method: str) -> float:
        return _select_bandwidth(scores, probabilities, n, method, "gaussian", 0.0)  # type: ignore[arg-type]

    for reference in (kequate, independent):
        assert select("double_smoothing") == pytest.approx(
            reference["double_smoothing"], abs=BANDWIDTH_TOL
        )
        assert select("cross_validation") == pytest.approx(
            reference["cross_validation"], abs=CROSS_VALIDATION_BANDWIDTH_TOL
        )
    rule_of_thumb = select("rule_of_thumb")
    assert rule_of_thumb == pytest.approx(independent["rule_of_thumb"], rel=RULE_OF_THUMB_RTOL)
    # kequate's standard deviation has the factor sqrt(N / (N - 1)).
    assert rule_of_thumb * np.sqrt(n / (n - 1)) == pytest.approx(
        kequate["rule_of_thumb"], rel=RULE_OF_THUMB_RTOL
    )


@pytest.mark.parametrize("method", ["double_smoothing", "cross_validation"])
def test_bandwidth_rules_end_to_end(method: str) -> None:
    tolerance = BANDWIDTH_TOL if method == "double_smoothing" else CROSS_VALIDATION_BANDWIDTH_TOL
    reference = RULES["end_to_end"]["equivalent_groups"][method]
    x, y = _equivalent_groups()
    result = kernel_equate(x, y, bandwidth_method=method)  # type: ignore[arg-type]
    assert result.bandwidth_x == pytest.approx(reference["bandwidths"]["hx"], abs=tolerance)
    assert result.bandwidth_y == pytest.approx(reference["bandwidths"]["hy"], abs=tolerance)
    np.testing.assert_allclose(result.equated, reference["equated"], rtol=0, atol=1e-5)
    assert _relative(result.see, reference["see"]) < END_TO_END_SEE_RTOL

    reference = RULES["end_to_end"]["chained"][method]
    xa, ya = _neat()
    result = kernel_equate(xa, ya, design="chained", bandwidth_method=method)  # type: ignore[arg-type]
    assert result.anchor_bandwidths is not None
    selected = (result.bandwidth_x, result.bandwidth_y, *result.anchor_bandwidths)
    expected = [reference["bandwidths"][key] for key in ("hxP", "hyQ", "haP", "haQ")]
    np.testing.assert_allclose(selected, expected, rtol=0, atol=tolerance)
    np.testing.assert_allclose(result.equated, reference["equated"], rtol=0, atol=1e-5)
    assert _relative(result.see, reference["see"]) < END_TO_END_SEE_RTOL
