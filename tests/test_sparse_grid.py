"""Smolyak sparse-grid generator of the design study (DESIGN.md §12, open question 6).

The generator lives in ``benchmarks/`` (study code, not shipped). These tests keep the study's
numbers reproducible: exactness up to the rule's degree and agreement with R ``SparseGrid``.
"""

from __future__ import annotations

import itertools
import json
import math

import numpy as np
import pytest
from benchmarks.sparse_grid import smolyak_normal, total_degree_exact

from tests._reference import REFERENCE_DIR

SPARSE_GRID_REFERENCE = REFERENCE_DIR / "sparse_grid" / "gqn.json"


def _normal_moment(k: int) -> int:
    """E[Z^k] for Z ~ N(0, 1): 0 for odd k, (k - 1)!! for even k."""
    return 0 if k % 2 else math.prod(range(k - 1, 0, -2))


@pytest.mark.parametrize(("n_factors", "level"), [(2, 4), (3, 4), (4, 3), (5, 4)])
def test_smolyak_integrates_mvn_moments_up_to_its_degree(n_factors: int, level: int) -> None:
    nodes, weights = smolyak_normal(n_factors, level)
    degree = total_degree_exact(level)
    assert weights.sum() == pytest.approx(1.0, abs=1e-13)
    beyond = 0.0
    for exponents in itertools.product(range(degree + 3), repeat=n_factors):
        if sum(exponents) > degree + 2:
            continue
        estimate = float(weights @ np.prod(nodes ** np.array(exponents), axis=1))
        truth = math.prod(_normal_moment(k) for k in exponents)
        error = abs(estimate - truth) / max(1.0, abs(truth))
        if sum(exponents) <= degree:
            assert error < 1e-11, exponents
        else:
            beyond = max(beyond, error)
    # Not exact beyond the degree (the rule is not a higher-order rule by accident).
    assert beyond > 1e-3


def test_smolyak_correlated_mvn_moments_by_cholesky_transform() -> None:
    sigma = np.array([[1.0, 0.5, 0.3], [0.5, 1.2, 0.4], [0.3, 0.4, 0.8]])
    mean = np.array([0.2, -0.1, 0.4])
    nodes, weights = smolyak_normal(3, 3)  # degree 5
    theta = mean + nodes @ np.linalg.cholesky(sigma).T
    np.testing.assert_allclose(weights @ theta, mean, atol=1e-13)
    centered = theta - mean
    cov = (weights[:, None] * centered).T @ centered
    np.testing.assert_allclose(cov, sigma, atol=1e-13)
    # Fourth moment E[X1^2 X2^2] = s11 s22 + 2 s12^2 (Isserlis, 1918).
    fourth = weights @ (centered[:, 0] ** 2 * centered[:, 1] ** 2)
    assert fourth == pytest.approx(sigma[0, 0] * sigma[1, 1] + 2 * sigma[0, 1] ** 2, abs=1e-12)


def test_smolyak_has_negative_weights_beyond_the_trivial_levels() -> None:
    for n_factors in range(2, 7):
        _, weights = smolyak_normal(n_factors, 1)
        assert weights.tolist() == [1.0]
        _, weights = smolyak_normal(n_factors, 3)
        assert (weights < 0).any()


def test_smolyak_matches_r_sparsegrid_gqn() -> None:
    reference = json.loads(SPARSE_GRID_REFERENCE.read_text())
    assert reference["meta"]["package"] == "SparseGrid"
    for rule in reference["rules"]:
        n_factors = rule["n_factors"]
        ref_nodes = np.asarray(rule["nodes"], dtype=np.float64)
        ref_weights = np.asarray(rule["weights"], dtype=np.float64)
        nodes, weights = smolyak_normal(n_factors, rule["level"])
        assert nodes.shape == ref_nodes.shape
        order_ref = np.lexsort(np.round(ref_nodes, 8).T[::-1])
        order = np.lexsort(np.round(nodes, 8).T[::-1])
        np.testing.assert_allclose(nodes[order], ref_nodes[order_ref], atol=1e-11)
        np.testing.assert_allclose(weights[order], ref_weights[order_ref], atol=1e-12)
    for row in reference["counts"]:
        if row["n_factors"] > 6:  # keep the test fast; 7..10 are in the JSON for the study table
            continue
        _, weights = smolyak_normal(row["n_factors"], row["level"])
        assert len(weights) == row["n_points"], row
