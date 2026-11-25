"""E4 against R: linear equating with an anchor test (DESIGN.md §7, case Q6).

Reference values (``validation/r/equate_e4_reference.R``): R ``equate`` (N - 1 moments, i.e.
``ddof=1``) and the formulas of Kolen & Brennan (2014) written out in R with population moments.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from flexpyequate import BivariateScoreDistribution, EquatingResult, equate

from ._reference import cases_by_name, load_reference

CASES = cases_by_name("e4_neat_linear")
NAMES = sorted(CASES)
WEIGHTED = [(name, index) for name in NAMES for index in range(len(CASES[name]["weights"]))]
WEIGHTED_IDS = [
    f"{name}-w{CASES[name]['weights'][index]['weight']:.3f}" for name, index in WEIGHTED
]

COEFFICIENT_TOL = 1e-10
EQUATED_TOL = 1e-9
PROBABILITY_TOL = 1e-13


def _joints(case: dict[str, Any]) -> tuple[BivariateScoreDistribution, BivariateScoreDistribution]:
    x = BivariateScoreDistribution.from_frequencies(
        x_scores=case["x_scores"],
        y_scores=case["anchor_scores"],
        frequencies=case["x_joint_frequencies"],
    )
    y = BivariateScoreDistribution.from_frequencies(
        x_scores=case["y_scores"],
        y_scores=case["anchor_scores"],
        frequencies=case["y_joint_frequencies"],
    )
    return x, y


def _check(result: EquatingResult, reference: dict[str, Any], *, check_sd: bool = True) -> None:
    assert result.slope == pytest.approx(reference["slope"], abs=COEFFICIENT_TOL)
    assert result.intercept == pytest.approx(reference["intercept"], abs=COEFFICIENT_TOL)
    if "equated" in reference:
        np.testing.assert_allclose(result.equated, reference["equated"], rtol=0, atol=EQUATED_TOL)
    if "gamma" in reference:
        assert result.gamma == pytest.approx(reference["gamma"], abs=COEFFICIENT_TOL)
    if "synthetic_mean" in reference:
        assert result.synthetic_moments is not None
        np.testing.assert_allclose(
            result.synthetic_moments["mean"], reference["synthetic_mean"], rtol=0, atol=1e-10
        )
        if check_sd:
            np.testing.assert_allclose(
                result.synthetic_moments["sd"], reference["synthetic_sd"], rtol=0, atol=1e-10
            )


def test_reference_metadata() -> None:
    assert load_reference("e4_neat_linear")["meta"]["equate_version"]
    assert NAMES == ["external_anchor", "kb_neat"]
    assert CASES["kb_neat"]["anchor"] == "internal"
    assert CASES["external_anchor"]["anchor"] == "external"
    x, y = _joints(CASES["kb_neat"])
    assert (x.n, y.n) == (1655, 1638)


@pytest.mark.parametrize(("name", "index"), WEIGHTED, ids=WEIGHTED_IDS)
@pytest.mark.parametrize("method", ["tucker", "levine_observed"])
def test_tucker_and_levine_observed(name: str, index: int, method: str) -> None:
    case = CASES[name]
    entry = case["weights"][index]
    x, y = _joints(case)
    options: dict[str, Any] = {"synthetic_weight": entry["weight"], "anchor": case["anchor"]}
    _check(equate(x, y, method=method, ddof=1, **options), entry["equate"][method])  # type: ignore[arg-type]
    _check(equate(x, y, method=method, **options), entry["kb"][method])  # type: ignore[arg-type]


@pytest.mark.parametrize(("name", "index"), WEIGHTED, ids=WEIGHTED_IDS)
def test_braun_holland(name: str, index: int) -> None:
    case = CASES[name]
    entry = case["weights"][index]
    x, y = _joints(case)
    result = equate(x, y, method="braun_holland", synthetic_weight=entry["weight"])
    _check(result, entry["kb"]["braun_holland"])
    # equate mixes the populations as counts: its weight of population 1 is
    # ws N1 / (ws N1 + (1 - ws) N2) (DESIGN.md §7, known differences).
    at_equate_weight = equate(
        x, y, method="braun_holland", synthetic_weight=entry["braun_holland_weight"]
    )
    # equate's synthetic standard deviations divide by (total count - 1); the factor is the same
    # for X and Y and cancels in the slope.
    _check(at_equate_weight, entry["equate"]["braun_holland"], check_sd=False)
    _check(at_equate_weight, entry["kb"]["braun_holland_at_equate_weight"])


@pytest.mark.parametrize("name", NAMES)
def test_default_weight(name: str) -> None:
    case = CASES[name]
    entry = case["weights"][0]
    x, y = _joints(case)
    result = equate(x, y, method="tucker")
    assert result.synthetic_weight == pytest.approx(entry["weight"], abs=1e-15)
    _check(result, entry["kb"]["tucker"])


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("method", ["levine_true", "chained_linear"])
def test_levine_true_and_chained_linear(name: str, method: str) -> None:
    case = CASES[name]
    x, y = _joints(case)
    options: dict[str, Any] = {"anchor": case["anchor"]}
    _check(equate(x, y, method=method, ddof=1, **options), case["equate"][method])  # type: ignore[arg-type]
    _check(equate(x, y, method=method, **options), case["kb"][method])  # type: ignore[arg-type]


@pytest.mark.parametrize(("name", "index"), WEIGHTED, ids=WEIGHTED_IDS)
def test_synthetic_distributions(name: str, index: int) -> None:
    from flexpyequate._neat import post_stratified_distributions

    case = CASES[name]
    entry = case["weights"][index]
    x, y = _joints(case)
    synthetic_x, synthetic_y, dropped = post_stratified_distributions(
        x.probabilities, y.probabilities, entry["weight"]
    )
    reference = entry["kb"]["braun_holland"]
    assert dropped == 0.0
    np.testing.assert_allclose(synthetic_x, reference["synthetic_x"], rtol=0, atol=PROBABILITY_TOL)
    np.testing.assert_allclose(synthetic_y, reference["synthetic_y"], rtol=0, atol=PROBABILITY_TOL)
