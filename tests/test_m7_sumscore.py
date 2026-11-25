"""Unit tests for the summed-score distribution kernel (DESIGN.md §18.1).

Agreement with R (``poibin``, ``stats::dpois`` and mirt's summed-score tables) is in
``tests/test_r_m7_sumscore.py`` (VALIDATION.md V36).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flexpyirt import _itemfit, summed_score_distribution
from flexpyirt.errors import SpecificationError

N_CATEGORIES = [2, 3, 4, 2, 3]


def _mixed(n_batch: int = 4, seed: int = 0) -> tuple[list[np.ndarray], np.ndarray]:
    """Random category probabilities as per-item traces and as one padded ``(B, J, K)`` array."""
    rng = np.random.default_rng(seed)
    traces = [rng.dirichlet(np.ones(k), size=n_batch) for k in N_CATEGORIES]
    probs = np.zeros((n_batch, len(N_CATEGORIES), max(N_CATEGORIES)))
    for j, tr in enumerate(traces):
        probs[:, j, : tr.shape[1]] = tr
    return traces, probs


def _enumerate(traces: list[np.ndarray]) -> np.ndarray:
    """Brute-force distribution over all response patterns, shape ``(B, S + 1)``."""
    n_categories = [tr.shape[1] for tr in traces]
    out = np.zeros((traces[0].shape[0], sum(k - 1 for k in n_categories) + 1))
    for y in np.ndindex(*n_categories):
        out[:, sum(y)] += np.prod([traces[j][:, y[j]] for j in range(len(traces))], axis=0)
    return out


def _dichotomous(p: np.ndarray) -> np.ndarray:
    return np.stack([1.0 - p, p], axis=-1)


@pytest.mark.parametrize("method", ["recursion", "fft"])
def test_exact_methods_match_enumeration(method: str) -> None:
    traces, probs = _mixed()
    expected = _enumerate(traces)
    got = np.asarray(summed_score_distribution(probs, method=method))  # type: ignore[arg-type]
    total = expected.shape[1]
    np.testing.assert_allclose(got[:, :total], expected, rtol=0, atol=5e-16)
    # Scores above this item set's maximum are padding: exactly 0 for the recursion.
    assert got.shape == (4, len(N_CATEGORIES) * (max(N_CATEGORIES) - 1) + 1)
    np.testing.assert_allclose(got[:, total:], 0.0, atol=0 if method == "recursion" else 5e-16)
    np.testing.assert_allclose(got.sum(axis=1), 1.0, rtol=1e-14)


def test_recursion_has_small_relative_error_in_the_tails() -> None:
    traces, probs = _mixed()
    expected = _enumerate(traces)
    got = np.asarray(summed_score_distribution(probs))[:, : expected.shape[1]]
    np.testing.assert_allclose(got, expected, rtol=1e-13)


def test_log_recursion_matches_and_survives_underflow() -> None:
    traces, probs = _mixed()
    expected = _enumerate(traces)
    log_p = np.asarray(summed_score_distribution(probs, log=True))
    np.testing.assert_allclose(log_p[:, : expected.shape[1]], np.log(expected), rtol=1e-13)
    assert np.all(np.isneginf(log_p[:, expected.shape[1] :]))
    # 2000 items with p = 0.01: P(S = 2000) = 1e-4000 underflows in the linear recursion.
    long = _dichotomous(np.full((1, 2000), 0.01))
    assert float(summed_score_distribution(long)[0, -1]) == 0.0
    log_long = np.asarray(summed_score_distribution(long, log=True))
    assert log_long[0, -1] == pytest.approx(2000 * math.log(0.01), rel=1e-12)
    assert log_long[0, 0] == pytest.approx(2000 * math.log(0.99), rel=1e-12)


def test_binomial_special_case() -> None:
    """Equal success probabilities give the binomial distribution."""
    n, p = 40, 0.3
    expected = np.array([math.comb(n, s) * p**s * (1 - p) ** (n - s) for s in range(n + 1)])
    probs = _dichotomous(np.full(n, p))
    np.testing.assert_allclose(summed_score_distribution(probs), expected, rtol=1e-12)
    np.testing.assert_allclose(
        summed_score_distribution(probs, method="fft"), expected, rtol=0, atol=1e-15
    )


def test_batch_axes_and_dtypes() -> None:
    _, probs = _mixed(n_batch=6)
    stacked = probs.reshape(2, 3, *probs.shape[1:])
    flat = np.asarray(summed_score_distribution(probs))
    np.testing.assert_allclose(
        np.asarray(summed_score_distribution(stacked)).reshape(flat.shape), flat, rtol=1e-14
    )
    np.testing.assert_allclose(summed_score_distribution(probs[0]), flat[0], rtol=1e-14)
    # Integer input (deterministic items) is promoted to floating point.
    certain = np.array([[0, 1], [1, 0], [0, 1]])
    np.testing.assert_array_equal(summed_score_distribution(certain), [0.0, 0.0, 1.0, 0.0])


@pytest.mark.parametrize("method", ["recursion", "fft", "normal", "refined_normal", "poisson"])
def test_jit_vmap_and_grad(method: str) -> None:
    rng = np.random.default_rng(3)
    probs = jnp.asarray(_dichotomous(rng.uniform(0.1, 0.9, size=(5, 12))))

    def kernel(x: jax.Array) -> jax.Array:
        return summed_score_distribution(x, method=method)  # type: ignore[arg-type]

    direct = kernel(probs)
    np.testing.assert_allclose(jax.jit(kernel)(probs), direct, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(jax.vmap(kernel)(probs), direct, rtol=1e-13, atol=1e-15)
    grad = jax.grad(lambda x: kernel(x)[0, 4])(probs)
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_gradient_matches_leave_one_out_distribution() -> None:
    r""":math:`\partial P(S=s)/\partial p_j = P_{-j}(S=s-1) - P_{-j}(S=s)` for dichotomous items."""
    rng = np.random.default_rng(4)
    p = rng.uniform(0.1, 0.9, size=9)
    s, j = 4, 2

    def prob_of_score(success: jax.Array) -> jax.Array:
        return summed_score_distribution(jnp.stack([1.0 - success, success], axis=-1))[s]

    def log_prob_of_score(success: jax.Array) -> jax.Array:
        both = jnp.stack([1.0 - success, success], axis=-1)
        return summed_score_distribution(both, log=True)[s]

    rest = np.asarray(summed_score_distribution(_dichotomous(np.delete(p, j))))
    expected = rest[s - 1] - rest[s]
    grad = jax.grad(prob_of_score)(jnp.asarray(p))
    assert float(grad[j]) == pytest.approx(expected, rel=1e-12)
    log_grad = jax.grad(log_prob_of_score)(jnp.asarray(p))
    assert float(log_grad[j]) == pytest.approx(
        expected / float(prob_of_score(jnp.asarray(p))), rel=1e-11
    )


def test_log_recursion_gradient_is_finite_with_zero_probabilities() -> None:
    _, probs = _mixed()  # padded categories have probability exactly 0
    grad = jax.grad(lambda x: summed_score_distribution(x, log=True)[0, 3])(jnp.asarray(probs))
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_fft_error_is_absolute() -> None:
    """The FFT is accurate to about 1e-16 in absolute terms, so tiny tails are not reliable."""
    probs = _dichotomous(np.full(200, 0.05))
    exact = np.asarray(summed_score_distribution(probs))
    fft = np.asarray(summed_score_distribution(probs, method="fft"))
    np.testing.assert_allclose(fft, exact, rtol=0, atol=1e-14)
    assert exact[150] < 1e-100  # far below what an absolute error of 1e-16 can resolve
    log_fft = np.asarray(summed_score_distribution(probs, method="fft", log=True))
    np.testing.assert_allclose(log_fft[:20], np.log(exact[:20]), rtol=1e-9)


def test_approximations_are_close_and_ordered() -> None:
    rng = np.random.default_rng(5)
    probs = _dichotomous(rng.uniform(0.2, 0.8, size=60))
    exact = np.asarray(summed_score_distribution(probs))
    normal = np.asarray(summed_score_distribution(probs, method="normal"))
    refined = np.asarray(summed_score_distribution(probs, method="refined_normal"))
    err_normal = np.abs(np.cumsum(normal) - np.cumsum(exact)).max()
    err_refined = np.abs(np.cumsum(refined) - np.cumsum(exact)).max()
    assert err_refined <= err_normal < 5e-3
    # Poisson: total-variation error at most sum p_j^2 (Le Cam, 1960), good for small p_j.
    rare = rng.uniform(0.001, 0.02, size=60)
    exact_rare = np.asarray(summed_score_distribution(_dichotomous(rare)))
    poisson = np.asarray(summed_score_distribution(_dichotomous(rare), method="poisson"))
    assert 0.5 * np.abs(poisson - exact_rare).sum() <= np.sum(rare**2)


@pytest.mark.parametrize(
    ("probs", "method", "match"),
    [
        (np.ones(3) / 3, "recursion", "shape"),
        (np.ones((2, 1)), "recursion", "shape"),
        (np.ones((0, 2)), "recursion", "shape"),
        (np.full((3, 2), 0.5), "exact", "Unknown method"),
        (np.full((3, 3), 1 / 3), "normal", "dichotomous"),
        (np.full((3, 3), 1 / 3), "refined_normal", "dichotomous"),
        (np.full((3, 3), 1 / 3), "poisson", "dichotomous"),
    ],
)
def test_rejects(probs: np.ndarray, method: str, match: str) -> None:
    with pytest.raises(SpecificationError, match=match):
        summed_score_distribution(probs, method=method)  # type: ignore[arg-type]


def test_item_fit_recursion_uses_the_kernel() -> None:
    """``_itemfit.lord_wingersky`` keeps its ``(S + 1, Q)`` layout and exact zeros are dropped."""
    traces, _ = _mixed()
    lik = _itemfit.lord_wingersky(traces)
    expected = _enumerate(traces)
    assert lik.shape == (expected.shape[1], 4)
    np.testing.assert_allclose(lik.T, expected, rtol=1e-13)
    assert _itemfit.lord_wingersky([]).shape == (1, 1)
