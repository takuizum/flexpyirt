"""E7 against R: standard errors of the traditional methods (DESIGN.md §7, cases Q9 and Q10).

Reference values (``validation/r/equate_e7_standard_errors_reference.R``):

- ``equate``: the analytic standard errors of R ``equate``; ``equate_scaled`` is the same call
  with all frequencies times 1e6, which removes the mixture of N and N - 1 moments in its
  anchor-test standard errors;
- ``independent``: the delta method written out in R with finite-difference Jacobians;
- ``equate_bootstrap``: ``equate::bootstrap``. R's random numbers cannot be reproduced, so the
  bootstrap comparison is statistical: two Monte Carlo estimates of the same standard error.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from flexpyequate import BivariateScoreDistribution, ScoreDistribution, equate
from flexpyequate.errors import SparseDataWarning

from ._reference import load_reference

REFERENCE = load_reference("e7_standard_errors")
EG = {case["name"]: case for case in REFERENCE["equivalent_groups"]}
SG = REFERENCE["single_group"]
NEAT = {case["name"]: case for case in REFERENCE["neat"]}
WEIGHTED = [(name, index) for name in sorted(NEAT) for index in range(len(NEAT[name]["weights"]))]

EQUATE_RTOL = 1e-10  # same formula, same moments
EQUATE_SCALED_RTOL = 1e-8  # frequencies times 1e6: N against N - 1 differs by 1e-9
EQUATE_ORDINARY_RTOL = 2e-3  # equate mixes N and N - 1 moments: O(1 / N)
INDEPENDENT_RTOL = 1e-8  # finite-difference Jacobians
LORD_TOL = 1e-6  # equate rounds percentile ranks to 10 decimals
N_REPLICATIONS = 2000
# Two bootstrap estimates with B replications each differ by about SE / sqrt(B) (one standard
# deviation, 2.2% for B = 2000); the tolerance is about five of them.
BOOTSTRAP_RTOL = 0.12
BOOTSTRAP_MEAN_RTOL = 0.06


def _relative(values: Any, reference: Any) -> float:
    return float(np.max(np.abs(np.asarray(values) / np.asarray(reference) - 1)))


def _forms(case: dict[str, Any]) -> tuple[ScoreDistribution, ScoreDistribution]:
    return (
        ScoreDistribution.from_frequencies(
            scores=case["x_scores"], frequencies=case["x_frequencies"]
        ),
        ScoreDistribution.from_frequencies(
            scores=case["y_scores"], frequencies=case["y_frequencies"]
        ),
    )


def _joints(case: dict[str, Any]) -> tuple[BivariateScoreDistribution, BivariateScoreDistribution]:
    return (
        BivariateScoreDistribution.from_frequencies(
            x_scores=case["x_scores"],
            y_scores=case["anchor_scores"],
            frequencies=case["x_joint_frequencies"],
        ),
        BivariateScoreDistribution.from_frequencies(
            x_scores=case["y_scores"],
            y_scores=case["anchor_scores"],
            frequencies=case["y_joint_frequencies"],
        ),
    )


def _see(*inputs: Any, **options: Any) -> np.ndarray:
    see = equate(*inputs, **options).see
    assert see is not None
    return np.asarray(see)


def test_reference_metadata() -> None:
    assert REFERENCE["meta"]["equate_version"]
    assert sorted(EG) == ["act_math", "score_increments", "unequal_lengths"]
    assert sorted(NEAT) == ["external_anchor", "kb_neat"]
    assert all(case["all_anchor_scores_observed"] for case in NEAT.values())


# ----------------------------------------------------------------------------- analytic
@pytest.mark.parametrize("name", sorted(EG))
def test_linear_equivalent_groups_against_equate(name: str) -> None:
    x, y = _forms(EG[name])
    see = _see(x, y, method="linear", ddof=1, standard_error="analytic")
    assert _relative(see, EG[name]["equate"]["linear"]) < EQUATE_RTOL


@pytest.mark.parametrize("name", sorted(EG))
@pytest.mark.parametrize("method", ["mean", "linear"])
def test_mean_and_linear_equivalent_groups_against_independent(name: str, method: str) -> None:
    x, y = _forms(EG[name])
    reference = EG[name]["independent"][method]
    assert _relative(_see(x, y, method=method, standard_error="analytic"), reference["se"]) < (
        INDEPENDENT_RTOL
    )
    normal = _see(x, y, method=method, standard_error="analytic_normal")
    assert _relative(normal, reference["se_normal"]) < INDEPENDENT_RTOL


@pytest.mark.parametrize("name", sorted(EG))
def test_equipercentile_equivalent_groups_against_equate(name: str) -> None:
    case = EG[name]
    x, y = _forms(case)
    see = _see(x, y, method="equipercentile", standard_error="analytic")
    # equate's formula has no score increment: its values are in units of the Y increment.
    unit = y.increment if case["equate"]["equipercentile_in_increments"] else 1.0
    np.testing.assert_allclose(
        see, unit * np.array(case["equate"]["equipercentile"]), rtol=LORD_TOL, atol=LORD_TOL
    )


@pytest.mark.parametrize("method", ["mean", "linear"])
def test_single_group_against_independent(method: str) -> None:
    joint = BivariateScoreDistribution.from_frequencies(
        x_scores=SG["x_scores"], y_scores=SG["y_scores"], frequencies=SG["joint_frequencies"]
    )
    reference = SG["independent"][method]
    assert _relative(_see(joint, method=method, standard_error="analytic"), reference["se"]) < (
        INDEPENDENT_RTOL
    )
    normal = _see(joint, method=method, standard_error="analytic_normal")
    assert _relative(normal, reference["se_normal"]) < INDEPENDENT_RTOL
    if method == "linear":
        assert _relative(normal, SG["normal_closed_form_linear"]) < 1e-12


@pytest.mark.parametrize(("name", "index"), WEIGHTED)
@pytest.mark.parametrize("method", ["tucker", "levine_observed"])
def test_tucker_and_levine_observed(name: str, index: int, method: str) -> None:
    case = NEAT[name]
    entry = case["weights"][index]
    options: dict[str, Any] = {
        "method": method,
        "synthetic_weight": entry["weight"],
        "anchor": case["anchor"],
    }
    general = _see(*_joints(case), standard_error="analytic", **options)
    normal = _see(*_joints(case), standard_error="analytic_normal", **options)
    scaled = entry["equate_scaled"][method]
    assert _relative(general, scaled["general"]) < EQUATE_SCALED_RTOL
    assert _relative(normal, scaled["normal"]) < EQUATE_SCALED_RTOL
    independent = entry["independent"][method]
    assert _relative(general, independent["se"]) < INDEPENDENT_RTOL
    assert _relative(normal, independent["se_normal"]) < INDEPENDENT_RTOL
    # The ordinary equate call, with either ddof.
    ordinary = entry["equate"][method]
    for ddof in (0, 1):
        see = _see(*_joints(case), standard_error="analytic", ddof=ddof, **options)
        assert _relative(see, ordinary["general"]) < EQUATE_ORDINARY_RTOL
        see = _see(*_joints(case), standard_error="analytic_normal", ddof=ddof, **options)
        assert _relative(see, ordinary["normal"]) < EQUATE_ORDINARY_RTOL


@pytest.mark.parametrize("name", sorted(NEAT))
def test_chained_linear(name: str) -> None:
    case = NEAT[name]
    general = _see(*_joints(case), method="chained_linear", standard_error="analytic")
    normal = _see(*_joints(case), method="chained_linear", standard_error="analytic_normal")
    # Ratios of moments of one population: N against N - 1 cancels, equate agrees as it is.
    for reference in (case["equate"]["chained_linear"], case["equate_scaled"]["chained_linear"]):
        assert _relative(general, reference["general"]) < EQUATE_RTOL
        assert _relative(normal, reference["normal"]) < EQUATE_RTOL
    independent = case["independent"]["chained_linear"]
    assert _relative(general, independent["se"]) < INDEPENDENT_RTOL
    assert _relative(normal, independent["se_normal"]) < INDEPENDENT_RTOL


@pytest.mark.parametrize("name", sorted(NEAT))
def test_levine_true_against_independent(name: str) -> None:
    case = NEAT[name]
    options: dict[str, Any] = {"method": "levine_true", "anchor": case["anchor"]}
    independent = case["independent"]["levine_true"]
    general = _see(*_joints(case), standard_error="analytic", **options)
    normal = _see(*_joints(case), standard_error="analytic_normal", **options)
    assert _relative(general, independent["se"]) < INDEPENDENT_RTOL
    assert _relative(normal, independent["se_normal"]) < INDEPENDENT_RTOL


@pytest.mark.parametrize(("name", "index"), WEIGHTED)
def test_braun_holland_against_independent(name: str, index: int) -> None:
    case = NEAT[name]
    entry = case["weights"][index]
    see = _see(
        *_joints(case),
        method="braun_holland",
        synthetic_weight=entry["weight"],
        standard_error="analytic",
    )
    assert _relative(see, entry["independent"]["braun_holland"]["se"]) < INDEPENDENT_RTOL


# ----------------------------------------------------------------------------- bootstrap
def _bootstrap_check(
    result_see: np.ndarray, reference: dict[str, Any], keep: np.ndarray | None = None
) -> None:
    """Statistical agreement of two bootstrap estimates (ours and equate::bootstrap)."""
    assert reference["reps"] == N_REPLICATIONS
    theirs = np.asarray(reference["se"])
    keep = np.ones(theirs.shape, dtype=bool) if keep is None else keep
    ratio = result_see[keep] / theirs[keep]
    assert np.max(np.abs(ratio - 1)) < BOOTSTRAP_RTOL
    assert abs(float(np.mean(ratio)) - 1) < BOOTSTRAP_MEAN_RTOL


def _central(distribution: ScoreDistribution) -> np.ndarray:
    """Scores away from the ends, where the two packages define the equated score alike."""
    rank = distribution.percentile_rank(distribution.scores) / 100
    return np.asarray((rank > 0.02) & (rank < 0.98))


@pytest.mark.parametrize("name", sorted(EG))
@pytest.mark.parametrize("method", ["linear", "equipercentile"])
def test_bootstrap_equivalent_groups_against_equate(name: str, method: str) -> None:
    case = EG[name]
    x, y = _forms(case)
    # equate's linear equating uses N - 1 variances; the difference is far below Monte Carlo
    # error, ddof=1 is passed for exactness.
    result = equate(
        x,
        y,
        method=method,  # type: ignore[arg-type]
        ddof=1,
        standard_error="bootstrap",
        n_replications=N_REPLICATIONS,
        rng=20261125,
    )
    assert result.see is not None
    keep = None if method == "linear" else _central(x)
    _bootstrap_check(np.asarray(result.see), case["equate_bootstrap"][method], keep)


@pytest.mark.parametrize("method", ["linear", "equipercentile"])
def test_bootstrap_single_group_against_equate(method: str) -> None:
    joint = BivariateScoreDistribution.from_frequencies(
        x_scores=SG["x_scores"], y_scores=SG["y_scores"], frequencies=SG["joint_frequencies"]
    )
    result = equate(
        joint,
        method=method,  # type: ignore[arg-type]
        ddof=1,
        standard_error="bootstrap",
        n_replications=N_REPLICATIONS,
        rng=20261125,
    )
    assert result.see is not None
    keep = None if method == "linear" else _central(joint.x)
    _bootstrap_check(np.asarray(result.see), SG["equate_bootstrap"][method], keep)


BOOTSTRAP_NEAT = [
    (name, method) for name in sorted(NEAT) for method in NEAT[name]["equate_bootstrap"]
]


@pytest.mark.parametrize(("name", "method"), BOOTSTRAP_NEAT)
def test_bootstrap_anchor_methods_against_equate(name: str, method: str) -> None:
    case = NEAT[name]
    xa, ya = _joints(case)
    options: dict[str, Any] = {"anchor": case["anchor"]} if method == "tucker" else {}
    if method == "frequency_estimation":
        # equate mixes counts: its effective weight of population 1 (DESIGN.md §7).
        weight = xa.n / (xa.n + ya.n)
        options["synthetic_weight"] = weight * xa.n / (weight * xa.n + (1 - weight) * ya.n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SparseDataWarning)
        result = equate(
            xa,
            ya,
            method=method,  # type: ignore[arg-type]
            ddof=1,
            standard_error="bootstrap",
            n_replications=N_REPLICATIONS,
            rng=20261125,
            **options,
        )
    assert result.see is not None
    keep = None if method in ("tucker", "chained_linear") else _central(xa.x)
    _bootstrap_check(np.asarray(result.see), case["equate_bootstrap"][method], keep)
