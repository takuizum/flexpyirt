"""E1 against R: mean, linear and equipercentile equating (DESIGN.md §7, cases Q1 and Q2).

Reference values: R ``equate`` and an independent R implementation of Kolen & Brennan (2014),
eq. 2.14-2.18 (``validation/r/equate_e1_reference.R``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from flexpyequate import BivariateScoreDistribution, EquatingResult, ScoreDistribution, equate

from ._reference import cases_by_name, load_reference

CASES = cases_by_name("e1_equate")

MOMENT_TOL = 1e-10
LINEAR_TOL = 1e-10
RANK_TOL = 1e-12
EQUIPERCENTILE_TOL = 1e-10  # independent, unrounded R implementation
EQUATE_EQUIPERCENTILE_TOL = 1e-7  # equate rounds ranks to 10 decimals


def _distributions(case: dict[str, Any]) -> tuple[ScoreDistribution, ScoreDistribution]:
    x = ScoreDistribution.from_frequencies(
        scores=case["x_scores"], frequencies=case["x_frequencies"]
    )
    y = ScoreDistribution.from_frequencies(
        scores=case["y_scores"], frequencies=case["y_frequencies"]
    )
    return x, y


def _equate(case: dict[str, Any], method: str, ddof: int = 0) -> EquatingResult:
    if case["design"] == "single_group":
        joint = BivariateScoreDistribution.from_frequencies(
            x_scores=case["x_scores"],
            y_scores=case["y_scores"],
            frequencies=case["joint_frequencies"],
        )
        return equate(joint, method=method, ddof=ddof)  # type: ignore[arg-type]
    x, y = _distributions(case)
    return equate(x, y, method=method, ddof=ddof)  # type: ignore[arg-type]


def test_reference_metadata() -> None:
    meta = load_reference("e1_equate")["meta"]
    assert meta["equate_version"]
    assert set(CASES) == {
        "act_math",
        "unequal_lengths",
        "zero_frequencies",
        "score_increments",
        "single_group",
    }


@pytest.mark.parametrize("name", sorted(CASES))
def test_moments(name: str) -> None:
    case = CASES[name]
    x, y = _distributions(case)
    assert x.mean == pytest.approx(case["x_moments"]["mean"], abs=MOMENT_TOL)
    assert y.mean == pytest.approx(case["y_moments"]["mean"], abs=MOMENT_TOL)
    assert x.sd(ddof=1) == pytest.approx(case["x_moments"]["sd_ddof1"], abs=MOMENT_TOL)
    assert y.sd(ddof=1) == pytest.approx(case["y_moments"]["sd_ddof1"], abs=MOMENT_TOL)


@pytest.mark.parametrize("name", sorted(CASES))
def test_percentile_ranks(name: str) -> None:
    case = CASES[name]
    x, y = _distributions(case)
    np.testing.assert_allclose(
        x.percentile_rank(x.scores) / 100, case["percentile_rank_x"], rtol=0, atol=RANK_TOL
    )
    np.testing.assert_allclose(
        y.percentile_rank(y.scores) / 100, case["percentile_rank_y"], rtol=0, atol=RANK_TOL
    )


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("method", ["mean", "linear"])
def test_mean_and_linear(name: str, method: str) -> None:
    case = CASES[name]
    result = _equate(case, method, ddof=1)
    reference = case[method]
    assert result.slope == pytest.approx(reference["slope"], abs=LINEAR_TOL)
    assert result.intercept == pytest.approx(reference["intercept"], abs=LINEAR_TOL)
    np.testing.assert_allclose(result.equated, reference["equated"], rtol=0, atol=LINEAR_TOL)


@pytest.mark.parametrize("name", sorted(CASES))
def test_equipercentile_against_kolen_brennan_formulas(name: str) -> None:
    case = CASES[name]
    result = _equate(case, "equipercentile")
    np.testing.assert_allclose(
        result.equated, case["kb_equipercentile"]["equated"], rtol=0, atol=EQUIPERCENTILE_TOL
    )


@pytest.mark.parametrize("name", sorted(CASES))
def test_equipercentile_against_equate(name: str) -> None:
    case = CASES[name]
    result = _equate(case, "equipercentile")
    # equate returns the end of the Y range where the percentile rank is exactly 0 or 1, instead
    # of the mean of the upper and lower percentiles (DESIGN.md §7, known differences).
    rank = np.asarray(case["percentile_rank_x"])
    inside = (rank > 0) & (rank < 1)
    assert inside.sum() >= 15
    np.testing.assert_allclose(
        result.equated[inside],
        np.asarray(case["equipercentile"]["equated"])[inside],
        rtol=0,
        atol=EQUATE_EQUIPERCENTILE_TOL,
    )


def test_single_group_joint_moments() -> None:
    case = CASES["single_group"]
    joint = BivariateScoreDistribution.from_frequencies(
        x_scores=case["x_scores"], y_scores=case["y_scores"], frequencies=case["joint_frequencies"]
    )
    assert joint.covariance(ddof=1) == pytest.approx(case["covariance_ddof1"], abs=MOMENT_TOL)
    assert joint.correlation == pytest.approx(case["correlation"], abs=MOMENT_TOL)


def test_act_math_published_moments() -> None:
    # Kolen & Brennan (2014), Table 2.6 (raw-score moments of forms X and Y, four decimals).
    x, y = _distributions(CASES["act_math"])
    assert (x.n, y.n) == (4329, 4152)
    assert round(x.mean, 4) == 19.8524
    assert round(y.mean, 4) == 18.9798
    assert round(x.sd(), 4) == 8.2116
    assert round(y.sd(), 4) == 8.9393
