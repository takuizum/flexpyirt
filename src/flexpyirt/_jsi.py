r"""Jackknife slope index kernels (Edwards, Houts & Cai, 2018; DESIGN.md §16.17).

Removing item :math:`j` is the same as treating its responses as missing: its one-hot rows are
zero, so it drops out of the marginal likelihood while every array keeps its shape. One jit
compilation therefore serves every removed item (DESIGN.md §5).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from flexpyirt import _core


def without_item(prob: _core.CompiledProblem, item: Array) -> _core.CompiledProblem:
    """Return ``prob`` with every response to column ``item`` set to missing."""
    return _core.with_responses(prob, prob.y.at[:, item].set(-1))


@jax.jit
def gradient_without(prob: _core.CompiledProblem, eta: Array, item: Array) -> Array:
    r""":math:`\nabla_\eta[\ell_{(-j)}(\eta)+\log\pi(\eta)]`, the log posterior without item ``j``.

    .. math::
        \ell_{(-j)}(\eta)=\sum_p c_p\log\sum_q w_q\prod_{i\ne j}P_i(y_{pi}\mid\theta_q)
    """
    grad: Array = jax.grad(lambda e: _core.log_posterior(without_item(prob, item), e))(eta)
    return grad


@jax.jit
def gradients_without(prob: _core.CompiledProblem, eta: Array, items: Array) -> Array:
    """:func:`gradient_without` for every entry of ``items`` at one ``eta``, ``(len(items), P)``."""
    grads: Array = jax.vmap(lambda j: gradient_without(prob, eta, j))(items)
    return grads


@jax.jit
def information_without(prob: _core.CompiledProblem, eta: Array, item: Array) -> Array:
    r"""Observed information :math:`-\nabla^2_\eta[\ell_{(-j)}(\eta)+\log\pi(\eta)]`."""
    return _core.information_hessian(without_item(prob, item), eta)


@jax.jit
def log_posterior_without(prob: _core.CompiledProblem, eta: Array, item: Array) -> Array:
    r""":math:`\ell_{(-j)}(\eta)+\log\pi(\eta)`."""
    return _core.log_posterior(without_item(prob, item), eta)


class NewtonOutcome(NamedTuple):
    """Result of :func:`newton_without`.

    ``eta``: the full parameter vector after the steps (removed-only entries unchanged);
    ``vcov``: inverse information on the ``keep`` entries at the last information evaluation;
    ``converged``: whether the last step was below ``tol`` (always ``True`` for a fixed number
    of steps); ``n_steps``: steps taken.
    """

    eta: np.ndarray
    vcov: np.ndarray
    converged: bool
    n_steps: int


def _ascent_direction(info: np.ndarray, grad: np.ndarray) -> np.ndarray:
    """Solve ``info @ step = grad``, adding a ridge until ``info`` is positive definite."""
    scale = max(float(np.mean(np.abs(np.diag(info)))), 1.0)
    ridge = 0.0
    for _ in range(30):
        try:
            chol = np.linalg.cholesky(info + ridge * np.eye(info.shape[0]))
        except np.linalg.LinAlgError:
            ridge = scale * 1e-8 if ridge == 0.0 else ridge * 10.0
            continue
        return np.asarray(np.linalg.solve(chol.T, np.linalg.solve(chol, grad)))
    return grad / scale


def newton_without(
    prob: _core.CompiledProblem,
    eta: np.ndarray,
    item: int,
    keep: np.ndarray,
    *,
    max_steps: int,
    tol: float | None,
) -> NewtonOutcome:
    r"""Newton–Raphson on :math:`\ell_{(-j)}+\log\pi` over the ``keep`` entries of ``eta``.

    .. math::
        \eta^{(t+1)}_{K}=\eta^{(t)}_{K}+\lambda_t\,I_{(-j)}\bigl(\eta^{(t)}\bigr)_{KK}^{-1}
        \nabla_{K}\bigl[\ell_{(-j)}+\log\pi\bigr]\bigl(\eta^{(t)}\bigr)

    with step halving (:math:`\lambda_t\in\{1,\tfrac12,\dots\}`) until the objective does not
    decrease. With ``tol=None`` exactly ``max_steps`` steps are taken (the one- or two-step
    Taylor approximation of the refit); otherwise the iteration stops once the largest step is
    below ``tol`` (the refit). The returned ``vcov`` is the inverse information of the last
    evaluation: at the start point for a single step, at the converged point for the refit.
    """
    item_arr = jnp.asarray(item)
    current = np.asarray(eta, dtype=np.float64).copy()
    objective = float(log_posterior_without(prob, jnp.asarray(current), item_arr))
    vcov = np.full((keep.size, keep.size), np.nan)
    converged = tol is None
    n_steps = 0
    while n_steps < max_steps:
        grad = np.asarray(gradient_without(prob, jnp.asarray(current), item_arr))[keep]
        info = np.asarray(information_without(prob, jnp.asarray(current), item_arr))
        info = info[np.ix_(keep, keep)]
        vcov = _inverse(info)
        step = _ascent_direction(info, grad)
        if tol is not None and float(np.max(np.abs(step))) < tol:
            converged = True
            break
        lam = 1.0
        for _ in range(30):
            trial = current.copy()
            trial[keep] += lam * step
            value = float(log_posterior_without(prob, jnp.asarray(trial), item_arr))
            if np.isfinite(value) and value >= objective - 1e-12 * abs(objective):
                break
            lam *= 0.5
        current, objective = trial, value
        n_steps += 1
    return NewtonOutcome(eta=current, vcov=vcov, converged=converged, n_steps=n_steps)


def _inverse(info: np.ndarray) -> np.ndarray:
    """Inverse of a symmetric information matrix (NaN if it is not positive definite)."""
    try:
        chol = np.linalg.cholesky(info)
    except np.linalg.LinAlgError:
        return np.full(info.shape, np.nan)
    inv_chol = np.linalg.inv(chol)
    return np.asarray(inv_chol.T @ inv_chol)
