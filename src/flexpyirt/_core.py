"""Core layer: pure JAX functions on pytrees (DESIGN.md §2 ③, §4).

Nothing here knows about names, strings or DataFrames. All shapes are static; variable-length
data is padded and masked (zero-count pattern rows, ``-1`` free indices).

Notation (DESIGN.md §3): ``P`` patterns, ``J`` items, ``Q`` nodes, ``G`` groups, ``K`` categories,
``eta`` the free-parameter vector on the natural scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from flexpyirt._newton import newton_ascent
from flexpyirt.items import ItemKernel
from flexpyirt.latent import normal_log_weights

__all__ = [
    "CompiledProblem",
    "EMResult",
    "FamilyBlock",
    "LatentBlock",
    "eap",
    "em",
    "information_hessian",
    "information_oakes",
    "marginal_loglik",
]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FamilyBlock:
    """Items sharing one kernel (DESIGN.md §5.2).

    Attributes
    ----------
    columns : (J_f,) int
        Column of each item in the response matrix.
    fixed_values : (J_f, P_f) float
        Values of fixed parameters (and starting values of free ones).
    free_index : (J_f, P_f) int
        Index into ``eta``, or ``-1`` if fixed.
    lower, upper : (J_f, P_f) float
        Parameter bounds.
    kernel : ItemKernel
        Static log-probability kernel.
    """

    columns: Array
    fixed_values: Array
    free_index: Array
    lower: Array
    upper: Array
    kernel: ItemKernel = field(metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LatentBlock:
    """Normal latent distribution of each group; parameter order is ``(mean, var)``.

    Attributes
    ----------
    fixed_values, free_index, lower, upper : (G, 2)
    """

    fixed_values: Array
    free_index: Array
    lower: Array
    upper: Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CompiledProblem:
    """Everything the Core needs (DESIGN.md §5.2).

    Attributes
    ----------
    y : (P, J) int
        Unique response patterns, missing as ``-1``; padding rows have count 0.
    counts : (P,) float
        Pattern frequencies.
    group : (P,) int
        Group of each pattern.
    nodes : (Q,) float
        Quadrature nodes.
    families : tuple[FamilyBlock, ...]
    latent : LatentBlock
    n_groups : int
        Static number of groups.
    """

    y: Array
    counts: Array
    group: Array
    nodes: Array
    families: tuple[FamilyBlock, ...]
    latent: LatentBlock
    n_groups: int = field(metadata={"static": True})


def _expand(eta: Array, fixed: Array, free_index: Array) -> Array:
    """Return ``eta[free_index]`` where free and ``fixed`` elsewhere (DESIGN.md §13.2)."""
    if eta.shape[0] == 0:  # static: nothing is free
        return fixed
    return jnp.where(free_index >= 0, eta[jnp.maximum(free_index, 0)], fixed)


def _scatter(eta: Array, values: Array, free_index: Array) -> Array:
    """Write ``values`` of free slots back into ``eta`` (fixed slots are dropped)."""
    idx = jnp.where(free_index >= 0, free_index, eta.shape[0])
    return eta.at[idx.ravel()].set(values.ravel(), mode="drop")


def _item_log_probs(prob: CompiledProblem, eta: Array) -> list[Array]:
    """Per family, ``(J_f, Q, K_f)`` log-probabilities on the nodes."""
    out = []
    for fam in prob.families:
        xi = _expand(eta, fam.fixed_values, fam.free_index)
        out.append(jax.vmap(lambda p, k=fam.kernel: k.log_prob(prob.nodes, p))(xi))
    return out


def _one_hots(prob: CompiledProblem) -> list[Array]:
    """Per family, ``(P, J_f, K_f)`` one-hot responses; a missing response is an all-zero row."""
    return [
        jax.nn.one_hot(prob.y[:, fam.columns], fam.kernel.n_cat, dtype=prob.nodes.dtype)
        for fam in prob.families
    ]


def _latent_log_weights(prob: CompiledProblem, eta: Array) -> Array:
    """``(G, Q)`` log quadrature weights."""
    lat = _expand(eta, prob.latent.fixed_values, prob.latent.free_index)
    return jax.vmap(normal_log_weights, in_axes=(None, 0, 0))(prob.nodes, lat[:, 0], lat[:, 1])


def _log_joint(prob: CompiledProblem, eta: Array) -> Array:
    r"""``(P, Q)`` :math:`\ell_{pq}+\log w_{g(p)q}` (DESIGN.md §4.2)."""
    ell = jnp.zeros((prob.y.shape[0], prob.nodes.shape[0]), dtype=prob.nodes.dtype)
    for oh, logp in zip(_one_hots(prob), _item_log_probs(prob, eta), strict=True):
        ell = ell + jnp.einsum("pjk,jqk->pq", oh, logp)
    return ell + _latent_log_weights(prob, eta)[prob.group]


def marginal_loglik(prob: CompiledProblem, eta: Array) -> Array:
    r"""Marginal log-likelihood (Bock & Aitkin, 1981).

    .. math::
        \ell(\eta)=\sum_p c_p\log\sum_q w_{g(p)q}\prod_j P_j(y_{pj}\mid\theta_q)
    """
    return jnp.sum(prob.counts * jax.scipy.special.logsumexp(_log_joint(prob, eta), axis=1))


class ExpectedCounts(NamedTuple):
    r"""E-step output: :math:`\bar r_{jkq}` per family (summed over groups), :math:`\bar n_{gq}`."""

    r: tuple[Array, ...]
    n: Array


def e_step(prob: CompiledProblem, eta: Array) -> tuple[ExpectedCounts, Array]:
    r"""E-step (DESIGN.md §4.2).

    .. math::
        h_{pq}=\mathrm{softmax}_q(\ell_{pq}+\log w_{g(p)q}),\quad
        \bar n_{gq}=\sum_{p\in g}c_ph_{pq},\quad
        \bar r_{jkq}=\sum_p c_ph_{pq}\mathbf 1[y_{pj}=k]

    Returns the expected counts and the marginal log-likelihood at ``eta``.
    """
    log_joint = _log_joint(prob, eta)
    lse = jax.scipy.special.logsumexp(log_joint, axis=1, keepdims=True)
    weighted = prob.counts[:, None] * jnp.exp(log_joint - lse)  # c_p h_pq, (P, Q)
    group_one_hot = jax.nn.one_hot(prob.group, prob.n_groups, dtype=weighted.dtype)
    n_bar = jnp.einsum("pg,pq->gq", group_one_hot, weighted)
    r_bar = tuple(jnp.einsum("pjk,pq->jkq", oh, weighted) for oh in _one_hots(prob))
    loglik = jnp.sum(prob.counts * lse[:, 0])
    return ExpectedCounts(r_bar, n_bar), loglik


def _item_q(kernel: ItemKernel, nodes: Array, r: Array, params: Array) -> Array:
    r""":math:`Q_j(\xi_j)=\sum_q\sum_k\bar r_{jkq}\log P_{jk}(\theta_q;\xi_j)` (DESIGN.md §4.3)."""
    return jnp.sum(r.T * kernel.log_prob(nodes, params))


def _latent_q(nodes: Array, n: Array, params: Array) -> Array:
    r"""Latent part of Q (DESIGN.md §4.4).

    .. math:: Q_g^{\mathrm{lat}}=\sum_q\bar n_{gq}\log w_{gq}(\mu_g,\sigma_g^2)
    """
    return jnp.sum(n * normal_log_weights(nodes, params[0], params[1]))


def expected_complete_loglik(prob: CompiledProblem, eta: Array, counts: ExpectedCounts) -> Array:
    r""":math:`Q(\eta\mid\eta')` with the expected counts computed at :math:`\eta'`."""
    total = jnp.zeros((), dtype=prob.nodes.dtype)
    for fam, r in zip(prob.families, counts.r, strict=True):
        xi = _expand(eta, fam.fixed_values, fam.free_index)
        q = jax.vmap(lambda rj, p, k=fam.kernel: _item_q(k, prob.nodes, rj, p))(r, xi)
        total = total + jnp.sum(q)
    lat = _expand(eta, prob.latent.fixed_values, prob.latent.free_index)
    q_lat = jax.vmap(lambda n, p: _latent_q(prob.nodes, n, p))(counts.n, lat)
    return total + jnp.sum(q_lat)


def m_step(prob: CompiledProblem, eta: Array, counts: ExpectedCounts, n_steps: int) -> Array:
    """M-step: item-wise and group-wise masked Newton, ``vmap``-ed (DESIGN.md §4.3, §4.4)."""
    new_eta = eta
    for fam, r in zip(prob.families, counts.r, strict=True):
        xi = _expand(eta, fam.fixed_values, fam.free_index)

        def solve_item(
            rj: Array, x: Array, free: Array, lo: Array, hi: Array, k: ItemKernel = fam.kernel
        ) -> Array:
            return newton_ascent(lambda p: _item_q(k, prob.nodes, rj, p), x, free, lo, hi, n_steps)

        xi_new = jax.vmap(solve_item)(r, xi, fam.free_index >= 0, fam.lower, fam.upper)
        new_eta = _scatter(new_eta, xi_new, fam.free_index)

    lat_block = prob.latent
    lat = _expand(eta, lat_block.fixed_values, lat_block.free_index)

    def solve_group(n: Array, x: Array, free: Array, lo: Array, hi: Array) -> Array:
        return newton_ascent(lambda p: _latent_q(prob.nodes, n, p), x, free, lo, hi, n_steps)

    lat_new = jax.vmap(solve_group)(
        counts.n, lat, lat_block.free_index >= 0, lat_block.lower, lat_block.upper
    )
    return _scatter(new_eta, lat_new, lat_block.free_index)


class EMResult(NamedTuple):
    """Output of :func:`em`. History arrays have length ``max_iter`` and are NaN past ``n_iter``."""

    eta: Array
    n_iter: Array
    converged: Array
    loglik: Array
    loglik_history: Array
    change_history: Array


def em(prob: CompiledProblem, eta0: Array, *, tol: float, max_iter: int, n_newton: int) -> EMResult:
    """Bock–Aitkin EM in a ``lax.while_loop`` (DESIGN.md §4.6).

    ``loglik_history[t]`` is the log-likelihood at the parameters entering iteration ``t``;
    ``change_history[t]`` the maximum absolute parameter change made by iteration ``t``.
    """
    nan_hist = jnp.full((max_iter,), jnp.nan, dtype=eta0.dtype)

    def cond(state: tuple[Array, Array, Array, Array, Array]) -> Array:
        _, it, change, _, _ = state
        return (it < max_iter) & (change >= tol)

    def body(
        state: tuple[Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array]:
        eta, it, _, ll_hist, ch_hist = state
        counts, ll = e_step(prob, eta)
        new = m_step(prob, eta, counts, n_newton)
        change = jnp.max(jnp.abs(new - eta), initial=0.0)
        return new, it + 1, change, ll_hist.at[it].set(ll), ch_hist.at[it].set(change)

    init = (eta0, jnp.asarray(0), jnp.asarray(jnp.inf, eta0.dtype), nan_hist, nan_hist)
    eta, n_iter, change, ll_hist, ch_hist = jax.lax.while_loop(cond, body, init)
    return EMResult(
        eta=eta,
        n_iter=n_iter,
        converged=change < tol,
        loglik=marginal_loglik(prob, eta),
        loglik_history=ll_hist,
        change_history=ch_hist,
    )


def information_hessian(prob: CompiledProblem, eta: Array) -> Array:
    r"""Observed information :math:`-\nabla^2_\eta\ell(\eta)` by ``jax.hessian`` (DESIGN §4.8)."""
    info: Array = -jax.hessian(lambda e: marginal_loglik(prob, e))(eta)
    return info


def information_oakes(prob: CompiledProblem, eta: Array) -> Array:
    r"""Observed information by Oakes' identity (Oakes, 1999; DESIGN.md §4.8).

    .. math::
        I(\hat\eta)=-\Big[\frac{\partial^2Q(\eta\mid\eta')}{\partial\eta\,\partial\eta^\top}
        +\frac{\partial^2Q(\eta\mid\eta')}{\partial\eta\,\partial\eta'^\top}\Big]_{\eta=\eta'=\hat\eta}
    """

    def q(e: Array, e_prime: Array) -> Array:
        counts, _ = e_step(prob, e_prime)
        return expected_complete_loglik(prob, e, counts)

    h_own, h_cross = jax.jacfwd(jax.grad(q, argnums=0), argnums=(0, 1))(eta, eta)
    info: Array = -(h_own + h_cross)
    return info


def eap(prob: CompiledProblem, eta: Array) -> tuple[Array, Array]:
    r"""EAP and posterior SD of each pattern, using its group's latent distribution as prior.

    .. math::
        \hat\theta_p=\sum_q\theta_q h_{pq},\qquad
        \mathrm{SD}_p=\Big(\sum_q\theta_q^2h_{pq}-\hat\theta_p^2\Big)^{1/2}
    """
    h = jax.nn.softmax(_log_joint(prob, eta), axis=1)
    mean = h @ prob.nodes
    var = h @ prob.nodes**2 - mean**2
    return mean, jnp.sqrt(jnp.maximum(var, 0.0))
