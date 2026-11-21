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
from flexpyirt.priors import PRIOR_NONE, eval_log_prior

__all__ = [
    "CompiledProblem",
    "EMResult",
    "FamilyBlock",
    "LatentBlock",
    "PriorBlock",
    "eap",
    "em",
    "information_hessian",
    "information_oakes",
    "log_posterior",
    "log_prior",
    "marginal_loglik",
    "squarem",
]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FamilyBlock:
    """Items sharing one kernel, for all groups (DESIGN.md §5.2, §14.1–14.2).

    Attributes
    ----------
    columns : (J_f,) int
        Column of each item in the response matrix.
    fixed_values : (G, J_f, P_f) float
        Values of fixed parameters (and starting values of free ones).
    free_index : (G, J_f, P_f) int
        Index into ``eta``, or ``-1`` if fixed.
    local_index : (G, J_f, P_f) int
        Slot in the item's local vector ``u_j`` (length ``S``), or ``-1`` if fixed. Parameters
        equal across groups share a slot.
    slot_free_index : (J_f, S) int
        Index into ``eta`` of each local slot, or ``-1`` for unused (padding) slots.
    slot_lower, slot_upper : (J_f, S) float
        Bounds of each local slot.
    kernel : ItemKernel
        Static log-probability kernel.
    """

    columns: Array
    fixed_values: Array
    free_index: Array
    local_index: Array
    slot_free_index: Array
    slot_lower: Array
    slot_upper: Array
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
class PriorBlock:
    """Branch-free prior specification per free parameter ``eta`` (DESIGN.md §4.7).

    Attributes
    ----------
    kind : (n_free,) int32
        ``0`` none, ``1`` NormalPrior, ``2`` LogNormal, ``3`` Beta.
    param1, param2 : (n_free,) float64
        Distribution parameters (unused entries hold ``0.0`` and ``1.0``).
    """

    kind: Array
    param1: Array
    param2: Array


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
    priors : PriorBlock
        Prior specification per free parameter (counted once per ``free_id``).
    n_groups : int
        Static number of groups.
    """

    y: Array
    counts: Array
    group: Array
    nodes: Array
    families: tuple[FamilyBlock, ...]
    latent: LatentBlock
    priors: PriorBlock
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
    """Per family, ``(G, J_f, Q, K_f)`` log-probabilities on the nodes."""
    out = []
    for fam in prob.families:
        xi = _expand(eta, fam.fixed_values, fam.free_index)  # (G, J_f, P_f)

        def per_item(p: Array, k: ItemKernel = fam.kernel) -> Array:
            return k.log_prob(prob.nodes, p)

        out.append(jax.vmap(jax.vmap(per_item))(xi))
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
    r"""``(P, Q)`` :math:`\ell_{pq}+\log w_{g(p)q}` (DESIGN.md §4.2, §14.3).

    The item part is computed for every group, ``(P, G, Q)``, and each pattern's own group is
    then selected, avoiding a ``(P, J, Q, K)`` intermediate.
    """
    n_pat, n_nodes = prob.y.shape[0], prob.nodes.shape[0]
    ell = jnp.zeros((n_pat, prob.n_groups, n_nodes), dtype=prob.nodes.dtype)
    for oh, logp in zip(_one_hots(prob), _item_log_probs(prob, eta), strict=True):
        ell = ell + jnp.einsum("pjk,gjqk->pgq", oh, logp)
    own = jnp.take_along_axis(ell, prob.group[:, None, None], axis=1)[:, 0, :]
    return own + _latent_log_weights(prob, eta)[prob.group]


def marginal_loglik(prob: CompiledProblem, eta: Array) -> Array:
    r"""Marginal log-likelihood (Bock & Aitkin, 1981).

    .. math::
        \ell(\eta)=\sum_p c_p\log\sum_q w_{g(p)q}\prod_j P_j(y_{pj}\mid\theta_q)
    """
    return jnp.sum(prob.counts * jax.scipy.special.logsumexp(_log_joint(prob, eta), axis=1))


def log_prior(prob: CompiledProblem, eta: Array) -> Array:
    r"""Sum of log priors over all free parameters on the natural scale (DESIGN.md §4.7).

    Returns ``0.0`` when no priors are present or ``eta`` is empty. Each free parameter
    (``free_id``) is counted once; fixed parameters are not in ``eta`` and contribute nothing.
    """
    if eta.shape[0] == 0:
        return jnp.zeros((), dtype=prob.nodes.dtype)
    return jnp.sum(eval_log_prior(eta, prob.priors.kind, prob.priors.param1, prob.priors.param2))


def log_posterior(prob: CompiledProblem, eta: Array) -> Array:
    r"""Log posterior :math:`\ell(\eta) + \log\pi(\eta)` (DESIGN.md §4.7).

    Equals :func:`marginal_loglik` when no priors are attached.
    """
    return marginal_loglik(prob, eta) + log_prior(prob, eta)


class ExpectedCounts(NamedTuple):
    r"""E-step output.

    ``r``: :math:`\bar r_{gjkq}` per family, shape ``(G, J_f, K_f, Q)``; ``n``: :math:`\bar n_{gq}`.
    """

    r: tuple[Array, ...]
    n: Array


def e_step(prob: CompiledProblem, eta: Array) -> tuple[ExpectedCounts, Array]:
    r"""E-step (DESIGN.md §4.2).

    .. math::
        h_{pq}=\mathrm{softmax}_q(\ell_{pq}+\log w_{g(p)q}),\quad
        \bar n_{gq}=\sum_{p\in g}c_ph_{pq},\quad
        \bar r_{gjkq}=\sum_{p\in g}c_ph_{pq}\mathbf 1[y_{pj}=k]

    Returns the expected counts and the marginal log-likelihood at ``eta``.
    """
    log_joint = _log_joint(prob, eta)
    lse = jax.scipy.special.logsumexp(log_joint, axis=1, keepdims=True)
    weighted = prob.counts[:, None] * jnp.exp(log_joint - lse)  # c_p h_pq, (P, Q)
    group_one_hot = jax.nn.one_hot(prob.group, prob.n_groups, dtype=weighted.dtype)
    n_bar = jnp.einsum("pg,pq->gq", group_one_hot, weighted)
    r_bar = tuple(
        jnp.einsum("pg,pjk,pq->gjkq", group_one_hot, oh, weighted) for oh in _one_hots(prob)
    )
    loglik = jnp.sum(prob.counts * lse[:, 0])
    return ExpectedCounts(r_bar, n_bar), loglik


def _item_q(kernel: ItemKernel, nodes: Array, r: Array, params: Array) -> Array:
    r""":math:`Q_{jg}(\xi)=\sum_q\sum_k\bar r_{gjkq}\log P_{jk}(\theta_q;\xi)` for one group (§4.3).

    Missing responses contribute nothing because their one-hot rows are zero.
    """
    return jnp.sum(r.T * kernel.log_prob(nodes, params))


def _latent_q(nodes: Array, n: Array, params: Array) -> Array:
    r"""Latent part of Q (DESIGN.md §4.4).

    .. math:: Q_g^{\mathrm{lat}}=\sum_q\bar n_{gq}\log w_{gq}(\mu_g,\sigma_g^2)
    """
    return jnp.sum(n * normal_log_weights(nodes, params[0], params[1]))


def _local_to_xi(u: Array, local: Array, fixed: Array) -> Array:
    """Item parameters of every group, ``(G, P_f)``, from the item's local vector ``u``."""
    return jnp.where(local >= 0, u[jnp.maximum(local, 0)], fixed)


def expected_complete_loglik(prob: CompiledProblem, eta: Array, counts: ExpectedCounts) -> Array:
    r""":math:`Q(\eta\mid\eta')` with the expected counts computed at :math:`\eta'`."""
    total = jnp.zeros((), dtype=prob.nodes.dtype)
    for fam, r in zip(prob.families, counts.r, strict=True):
        xi = _expand(eta, fam.fixed_values, fam.free_index)  # (G, J_f, P_f)

        def per_item(rj: Array, p: Array, k: ItemKernel = fam.kernel) -> Array:
            return _item_q(k, prob.nodes, rj, p)

        total = total + jnp.sum(jax.vmap(jax.vmap(per_item))(r, xi))
    lat = _expand(eta, prob.latent.fixed_values, prob.latent.free_index)
    q_lat = jax.vmap(lambda n, p: _latent_q(prob.nodes, n, p))(counts.n, lat)
    return total + jnp.sum(q_lat)


def m_step(prob: CompiledProblem, eta: Array, counts: ExpectedCounts, n_steps: int) -> Array:
    """M-step: item-wise and group-wise masked Newton, ``vmap``-ed (DESIGN.md §4.3, §4.4, §14.2).

    Each item is one Newton block over its local vector, which holds the parameters of all
    groups with across-group equalities merged into shared slots. Item-parameter priors are added
    directly to each item's local objective so the item-wise decomposition is preserved and every
    shared parameter's prior is counted once.
    """
    new_eta = eta
    for fam, r in zip(prob.families, counts.r, strict=True):
        u0 = _expand(eta, jnp.zeros(fam.slot_free_index.shape, eta.dtype), fam.slot_free_index)
        slot_free = fam.slot_free_index >= 0
        if eta.shape[0] == 0:
            slot_kind = jnp.full(fam.slot_free_index.shape, PRIOR_NONE, dtype=jnp.int32)
            slot_p1 = jnp.zeros(fam.slot_free_index.shape, dtype=eta.dtype)
            slot_p2 = jnp.ones(fam.slot_free_index.shape, dtype=eta.dtype)
        else:
            safe_fid = jnp.maximum(fam.slot_free_index, 0)
            slot_kind = jnp.where(slot_free, prob.priors.kind[safe_fid], PRIOR_NONE)
            slot_p1 = jnp.where(slot_free, prob.priors.param1[safe_fid], 0.0)
            slot_p2 = jnp.where(slot_free, prob.priors.param2[safe_fid], 1.0)
        r_items = jnp.swapaxes(r, 0, 1)  # (J_f, G, K, Q)
        local = jnp.swapaxes(fam.local_index, 0, 1)  # (J_f, G, P_f)
        fixed = jnp.swapaxes(fam.fixed_values, 0, 1)

        def solve_item(
            rj: Array,
            loc: Array,
            fix: Array,
            u: Array,
            free: Array,
            lo: Array,
            hi: Array,
            sk: Array,
            sp1: Array,
            sp2: Array,
            k: ItemKernel = fam.kernel,
        ) -> Array:
            def q_item(v: Array) -> Array:
                xi = _local_to_xi(v, loc, fix)  # (G, P_f)
                q_lik = jnp.sum(jax.vmap(lambda rg, pg: _item_q(k, prob.nodes, rg, pg))(rj, xi))
                return q_lik + jnp.sum(eval_log_prior(v, sk, sp1, sp2))

            return newton_ascent(q_item, u, free, lo, hi, n_steps)

        u_new = jax.vmap(solve_item)(
            r_items,
            local,
            fixed,
            u0,
            slot_free,
            fam.slot_lower,
            fam.slot_upper,
            slot_kind,
            slot_p1,
            slot_p2,
        )
        new_eta = _scatter(new_eta, u_new, fam.slot_free_index)

    lat_block = prob.latent
    lat = _expand(eta, lat_block.fixed_values, lat_block.free_index)

    def solve_group(n: Array, x: Array, free: Array, lo: Array, hi: Array) -> Array:
        return newton_ascent(lambda p: _latent_q(prob.nodes, n, p), x, free, lo, hi, n_steps)

    lat_new = jax.vmap(solve_group)(
        counts.n, lat, lat_block.free_index >= 0, lat_block.lower, lat_block.upper
    )
    return _scatter(new_eta, lat_new, lat_block.free_index)


class EMResult(NamedTuple):
    """Output of :func:`em` and :func:`squarem`.

    History arrays have length ``max_iter``; entries past ``n_iter`` are NaN (for float arrays),
    0 (for ``em_eval_history``), or False (for ``safeguard_history``).
    """

    eta: Array
    n_iter: Array
    converged: Array
    loglik: Array
    log_posterior: Array
    loglik_history: Array
    log_posterior_history: Array
    change_history: Array
    n_em_evaluations: Array
    em_eval_history: Array
    safeguard_history: Array
    n_safeguards: Array


_SQUAREM_MAX_BACKTRACK = 10
"""Maximum halvings of ``alpha`` towards ``-1`` before falling back to ``theta_2``."""


def _eta_bounds(prob: CompiledProblem, eta: Array) -> tuple[Array, Array]:
    """Per-coordinate ``(lower, upper)`` bounds of ``eta`` from the M-step blocks (§15.13)."""
    lo = jnp.full_like(eta, -jnp.inf)
    hi = jnp.full_like(eta, jnp.inf)
    for fam in prob.families:
        lo = _scatter(lo, fam.slot_lower, fam.slot_free_index)
        hi = _scatter(hi, fam.slot_upper, fam.slot_free_index)
    lo = _scatter(lo, prob.latent.lower, prob.latent.free_index)
    hi = _scatter(hi, prob.latent.upper, prob.latent.free_index)
    return lo, hi


def em(prob: CompiledProblem, eta0: Array, *, tol: float, max_iter: int, n_newton: int) -> EMResult:
    """Bock–Aitkin EM in a ``lax.while_loop`` (DESIGN.md §4.6, §4.7).

    ``loglik_history[t]`` and ``log_posterior_history[t]`` are evaluated at the parameters entering
    iteration ``t``; ``change_history[t]`` is the maximum absolute parameter change made by
    iteration ``t``. EM is monotone in the log posterior (which equals the log-likelihood without
    priors).
    """
    nan_hist = jnp.full((max_iter,), jnp.nan, dtype=eta0.dtype)
    eval_hist = jnp.zeros((max_iter,), dtype=jnp.int32)
    safe_hist = jnp.zeros((max_iter,), dtype=jnp.bool_)
    one = jnp.asarray(1, dtype=jnp.int32)

    def cond(state: tuple[Array, Array, Array, Array, Array, Array, Array]) -> Array:
        _, it, change, _, _, _, _ = state
        return (it < max_iter) & (change >= tol)

    def body(
        state: tuple[Array, Array, Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
        eta, it, _, ll_hist, lp_hist, ch_hist, ev_hist = state
        counts, ll = e_step(prob, eta)
        lp = ll + log_prior(prob, eta)
        new = m_step(prob, eta, counts, n_newton)
        change = jnp.max(jnp.abs(new - eta), initial=0.0)
        next_it = it + one
        return (
            new,
            next_it,
            change,
            ll_hist.at[it].set(ll),
            lp_hist.at[it].set(lp),
            ch_hist.at[it].set(change),
            ev_hist.at[it].set(next_it),
        )

    init = (
        eta0,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(jnp.inf, eta0.dtype),
        nan_hist,
        nan_hist,
        nan_hist,
        eval_hist,
    )
    eta, n_iter, change, ll_hist, lp_hist, ch_hist, ev_hist = jax.lax.while_loop(cond, body, init)
    final_ll = marginal_loglik(prob, eta)
    return EMResult(
        eta=eta,
        n_iter=n_iter,
        converged=change < tol,
        loglik=final_ll,
        log_posterior=final_ll + log_prior(prob, eta),
        loglik_history=ll_hist,
        log_posterior_history=lp_hist,
        change_history=ch_hist,
        n_em_evaluations=n_iter,
        em_eval_history=ev_hist,
        safeguard_history=safe_hist,
        n_safeguards=jnp.asarray(0, dtype=jnp.int32),
    )


def squarem(
    prob: CompiledProblem, eta0: Array, *, tol: float, max_iter: int, n_newton: int
) -> EMResult:
    r"""SQUAREM-accelerated Bock–Aitkin EM (Varadhan & Roland, 2008, scheme S3; DESIGN §4.6, §4.7).

    Each iteration applies the EM map :math:`F` twice to obtain :math:`\theta_1=F(\theta_0)` and
    :math:`\theta_2=F(\theta_1)`, forms :math:`r=\theta_1-\theta_0`,
    :math:`s=\theta_2-\theta_1`, :math:`v=s-r`, and computes the S3 step length

    .. math::
        \alpha = \min\!\left(-\frac{\lVert r\rVert_2}{\lVert v\rVert_2},\,-1\right),
        \qquad
        \theta'(\alpha) = \theta_0 - 2\alpha r + \alpha^2 v
        = \theta_2 - 2\delta s + \delta^2 v \quad (\delta=\alpha+1\le 0).

    Monotonicity and feasibility safeguard: if :math:`\theta'(\alpha)` leaves the strict parameter
    bounds ``(lower, upper)`` or gives a non-finite or lower log posterior than
    :math:`\max(\ell_{\mathrm{post}}(\theta_2),\ell_{\mathrm{post}}(\theta_0))`, :math:`\alpha` is
    stepped back towards :math:`-1` via :math:`\alpha\leftarrow(\alpha-1)/2` (i.e.
    :math:`\delta\leftarrow\delta/2`) up to ``_SQUAREM_MAX_BACKTRACK`` times. An accepted
    :math:`\theta'` is followed by one stabilizing EM step :math:`\theta_{\text{new}}=F(\theta')`
    (3 EM map evaluations); if no backtracked candidate is acceptable,
    :math:`\theta_{\text{new}}=\theta_2` is used directly (2 EM map evaluations). Runs entirely
    inside ``lax.while_loop``.
    """
    lower, upper = _eta_bounds(prob, eta0)
    nan_hist = jnp.full((max_iter,), jnp.nan, dtype=eta0.dtype)
    eval_hist = jnp.zeros((max_iter,), dtype=jnp.int32)
    safe_hist = jnp.zeros((max_iter,), dtype=jnp.bool_)
    zero = jnp.asarray(0, dtype=jnp.int32)
    one = jnp.asarray(1, dtype=jnp.int32)
    two = jnp.asarray(2, dtype=jnp.int32)
    three = jnp.asarray(3, dtype=jnp.int32)

    def _is_acceptable(cand: Array, lp_ref: Array) -> Array:
        in_bounds = jnp.all(jnp.isfinite(cand) & (cand > lower) & (cand < upper))

        def _eval_lp(c: Array) -> Array:
            lp_c = log_posterior(prob, c)
            return jnp.isfinite(lp_c) & (lp_c >= lp_ref)

        accepted: Array = jax.lax.cond(in_bounds, _eval_lp, lambda _: jnp.asarray(False), cand)
        return accepted

    def cond(
        state: tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array],
    ) -> Array:
        _, it, change, _, _, _, _, _, _, _ = state
        return (it < max_iter) & (change >= tol)

    def body(
        state: tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array]:
        eta, it, _, n_evals, n_safe, ll_hist, lp_hist, ch_hist, ev_hist, sf_hist = state
        counts0, ll0 = e_step(prob, eta)
        lp0 = ll0 + log_prior(prob, eta)
        theta1 = m_step(prob, eta, counts0, n_newton)
        counts1, _ = e_step(prob, theta1)
        theta2 = m_step(prob, theta1, counts1, n_newton)

        r = theta1 - eta
        s = theta2 - theta1
        v = s - r
        r_norm = jnp.sqrt(jnp.sum(r * r))
        v_norm = jnp.sqrt(jnp.sum(v * v))
        has_v = v_norm > 0.0
        raw_alpha = jnp.where(has_v, -r_norm / jnp.where(has_v, v_norm, 1.0), -1.0)
        alpha = jnp.where(jnp.isfinite(raw_alpha), jnp.minimum(raw_alpha, -1.0), -1.0)

        def _extrapolate(_unused: None) -> tuple[Array, Array, Array]:
            lp2 = log_posterior(prob, theta2)
            lp_ref = jnp.maximum(lp2, lp0)

            def bt_cond(bt: tuple[Array, Array, Array, Array]) -> Array:
                k, _, ok, _ = bt
                return (~ok) & (k <= _SQUAREM_MAX_BACKTRACK)

            def bt_body(bt: tuple[Array, Array, Array, Array]) -> tuple[Array, Array, Array, Array]:
                k, delta, _, _ = bt
                cand = theta2 - 2.0 * delta * s + (delta * delta) * v
                ok = _is_acceptable(cand, lp_ref)
                next_delta = jnp.where(ok, delta, 0.5 * delta)
                next_k = jnp.where(ok, k, k + one)
                return next_k, next_delta, ok, cand

            init_bt = (zero, alpha + 1.0, jnp.asarray(False), theta2)
            k_out, _, ok, cand = jax.lax.while_loop(bt_cond, bt_body, init_bt)
            fired = k_out > zero
            theta_p = jnp.where(ok, cand, theta2)
            return theta_p, ok, fired

        def _no_extrapolate(_unused: None) -> tuple[Array, Array, Array]:
            return theta2, jnp.asarray(True), jnp.asarray(False)

        theta_prime, do_stabilize, fired = jax.lax.cond(
            alpha < -1.0, _extrapolate, _no_extrapolate, None
        )

        def _stabilize(t: Array) -> tuple[Array, Array]:
            c, _ = e_step(prob, t)
            return m_step(prob, t, c, n_newton), three

        def _fallback(_unused: Array) -> tuple[Array, Array]:
            return theta2, two

        new, evals_added = jax.lax.cond(do_stabilize, _stabilize, _fallback, theta_prime)
        change = jnp.max(jnp.abs(new - eta), initial=0.0)
        next_evals = n_evals + evals_added
        next_safe = n_safe + fired.astype(jnp.int32)
        return (
            new,
            it + one,
            change,
            next_evals,
            next_safe,
            ll_hist.at[it].set(ll0),
            lp_hist.at[it].set(lp0),
            ch_hist.at[it].set(change),
            ev_hist.at[it].set(next_evals),
            sf_hist.at[it].set(fired),
        )

    init = (
        eta0,
        zero,
        jnp.asarray(jnp.inf, eta0.dtype),
        zero,
        zero,
        nan_hist,
        nan_hist,
        nan_hist,
        eval_hist,
        safe_hist,
    )
    eta, n_iter, change, n_evals, n_safe, ll_hist, lp_hist, ch_hist, ev_hist, sf_hist = (
        jax.lax.while_loop(cond, body, init)
    )
    final_ll = marginal_loglik(prob, eta)
    return EMResult(
        eta=eta,
        n_iter=n_iter,
        converged=change < tol,
        loglik=final_ll,
        log_posterior=final_ll + log_prior(prob, eta),
        loglik_history=ll_hist,
        log_posterior_history=lp_hist,
        change_history=ch_hist,
        n_em_evaluations=n_evals,
        em_eval_history=ev_hist,
        safeguard_history=sf_hist,
        n_safeguards=n_safe,
    )


def information_hessian(prob: CompiledProblem, eta: Array) -> Array:
    r"""Observed posterior information :math:`-\nabla^2_\eta[\ell(\eta)+\log\pi(\eta)]` (§4.8).

    Reduces to :math:`-\nabla^2_\eta\ell(\eta)` when no priors are present.
    """
    info: Array = -jax.hessian(lambda e: log_posterior(prob, e))(eta)
    return info


def information_oakes(prob: CompiledProblem, eta: Array) -> Array:
    r"""Observed (posterior) information by Oakes' identity (Oakes, 1999; DESIGN.md §4.8).

    .. math::
        I(\hat\eta)=-\Big[\frac{\partial^2Q_{\mathrm{post}}(\eta\mid\eta')}{\partial\eta\,\partial\eta^\top}
        +\frac{\partial^2Q_{\mathrm{post}}(\eta\mid\eta')}{\partial\eta\,\partial\eta'^\top}\Big]_{\eta=\eta'=\hat\eta}

    where :math:`Q_{\mathrm{post}}(\eta\mid\eta')=Q(\eta\mid\eta')+\log\pi(\eta)`. Adding
    :math:`\log\pi(\eta)` contributes :math:`-\nabla^2_\eta\log\pi(\hat\eta)` and keeps the
    exact identity with :func:`information_hessian`.
    """

    def q(e: Array, e_prime: Array) -> Array:
        counts, _ = e_step(prob, e_prime)
        return expected_complete_loglik(prob, e, counts) + log_prior(prob, e)

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
