"""Core layer: per-pattern scores and the cross-product information (DESIGN.md §4.8, §17.5).

Pure JAX functions on the :class:`~flexpyirt._core.CompiledProblem` pytree. ``fit()`` inverts
:func:`information_cross_product` for ``se="crossprod"`` and combines it with the observed
information of :func:`flexpyirt._core.information_oakes` for ``se="sandwich"``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import jax
import jax.numpy as jnp
from jax import Array

from flexpyirt._core import (
    CompiledProblem,
    _item_log_probs,
    _latent_log_weights,
    _one_hots,
    log_prior,
)

__all__ = [
    "information_cross_product",
    "score_cross_product",
    "score_sum",
]

_CHUNK_ELEMENTS = 8_000_000
"""Target element budget (~64 MB in float64) for the per-chunk reverse-mode workspace."""
_MAX_CHUNK = 1024
"""Maximum number of response patterns differentiated per chunk."""

Carry = TypeVar("Carry")


def _chunk_size(prob: CompiledProblem) -> int:
    """Return the static number of patterns per chunk that bounds the score workspace.

    One pattern's cotangent of the node tables has ``G * Q * (1 + sum_f J_f * K_f)`` elements;
    pulling it back through the item kernels adds about ``P_f`` copies of the item part.
    """
    n_nodes = prob.nodes.shape[0]
    per_pattern = prob.n_groups * n_nodes
    for fam in prob.families:
        n_items, n_params = fam.free_index.shape[1], fam.free_index.shape[2]
        per_pattern += prob.n_groups * n_items * n_nodes * fam.kernel.n_categories * (n_params + 1)
    return int(max(1, min(_MAX_CHUNK, _CHUNK_ELEMENTS // per_pattern)))


def _fold_pattern_scores(
    prob: CompiledProblem,
    eta: Array,
    init: Carry,
    accumulate: Callable[[Carry, Array, Array], Carry],
) -> Carry:
    r"""Fold ``accumulate(carry, scores, counts)`` over pattern chunks of the score matrix.

    ``scores`` is a ``(B, n_free)`` block of per-pattern scores
    :math:`s_p=\nabla_\eta\log L_p(\eta)` and ``counts`` the ``(B,)`` pattern counts (zero on
    padding rows). The node tables :math:`\log P_{gjqk}(\eta)` and :math:`\log w_{gq}(\eta)` are
    evaluated and linearized once (``jax.vjp``); each chunk differentiates its pattern
    log-likelihoods with respect to the tables by reverse mode (``jax.jacrev``) and pulls the
    result back to :math:`\eta`, so neither ``(P, n_free)`` nor ``(P, G, J, Q, K)`` is
    materialized. Chunks that contain only zero-count padding rows are skipped.
    """

    def node_tables(e: Array) -> tuple[tuple[Array, ...], Array]:
        return tuple(_item_log_probs(prob, e)), _latent_log_weights(prob, e)

    tables, pullback = jax.vjp(node_tables, eta)

    def chunk_logliks(
        tabs: tuple[tuple[Array, ...], Array], one_hots: tuple[Array, ...], group: Array
    ) -> Array:
        log_probs, log_weights = tabs
        ell = jnp.zeros((group.shape[0], prob.n_groups, prob.nodes.shape[0]), dtype=eta.dtype)
        for one_hot, log_prob in zip(one_hots, log_probs, strict=True):
            ell = ell + jnp.einsum("pjk,gjqk->pgq", one_hot, log_prob)
        own = jnp.take_along_axis(ell, group[:, None, None], axis=1)[:, 0, :]
        return jax.scipy.special.logsumexp(own + log_weights[group], axis=1)

    chunk = _chunk_size(prob)
    n_patterns = prob.y.shape[0]
    n_chunks = (n_patterns + chunk - 1) // chunk
    pad = n_chunks * chunk - n_patterns

    def chunked(x: Array) -> Array:
        padded = jnp.concatenate([x, jnp.zeros((pad, *x.shape[1:]), dtype=x.dtype)], axis=0)
        return padded.reshape(n_chunks, chunk, *x.shape[1:])

    chunks = (
        tuple(chunked(one_hot) for one_hot in _one_hots(prob)),
        chunked(prob.group),
        chunked(prob.counts),
    )

    def step(carry: Carry, x: tuple[tuple[Array, ...], Array, Array]) -> tuple[Carry, None]:
        one_hots, group, counts = x

        def with_scores(c: Carry) -> Carry:
            cotangents = jax.jacrev(chunk_logliks)(tables, one_hots, group)
            (scores,) = jax.vmap(pullback)(cotangents)
            return accumulate(c, scores, counts)

        return jax.lax.cond(jnp.any(counts > 0), with_scores, lambda c: c, carry), None

    out, _ = jax.lax.scan(step, init, chunks)
    return out


def score_sum(prob: CompiledProblem, eta: Array) -> Array:
    r"""Return the count-weighted sum of the per-pattern scores.

    :math:`\sum_p c_p s_p=\nabla_\eta\ell(\eta)`, which equals ``jax.grad`` of
    :func:`flexpyirt._core.marginal_loglik`; used to check the chunked scores.
    """

    def accumulate(total: Array, scores: Array, counts: Array) -> Array:
        return total + counts @ scores

    return _fold_pattern_scores(prob, eta, jnp.zeros_like(eta), accumulate)


def score_cross_product(prob: CompiledProblem, eta: Array) -> Array:
    r"""Return the cross-product of the per-pattern scores (outer product of gradients).

    .. math::
        B(\eta)=\sum_p c_p\,s_p s_p^\top,\qquad
        s_p=\nabla_\eta\log\sum_q w_{g(p)q}(\eta)\prod_j P_j(y_{pj}\mid\theta_q;\eta)

    Berndt, Hall, Hall & Hausman (1974). The scores come from reverse-mode autodiff of each
    pattern's marginal log-likelihood, in pattern chunks (DESIGN.md §17.5). A pattern of group
    :math:`g` has a zero score for parameters that only enter other groups. For Empirical
    Histogram models the histogram weights are held fixed (they are not in :math:`\eta`;
    DESIGN.md §15.35).
    """
    n_free = eta.shape[0]
    if n_free == 0:
        return jnp.zeros((0, 0), dtype=eta.dtype)

    def accumulate(total: Array, scores: Array, counts: Array) -> Array:
        return total + (counts[:, None] * scores).T @ scores

    return _fold_pattern_scores(prob, eta, jnp.zeros((n_free, n_free), dtype=eta.dtype), accumulate)


def information_cross_product(prob: CompiledProblem, eta: Array) -> Array:
    r"""Return the cross-product estimate of the (posterior) information (DESIGN.md §4.8, §17.5).

    .. math::
        B_{\mathrm{post}}(\hat\eta)=\sum_p c_p\,s_p s_p^\top-\nabla^2_\eta\log\pi(\hat\eta)

    Without priors this is the outer-product-of-gradients estimate of the Fisher information.
    With priors the prior curvature is added exactly as in
    :func:`flexpyirt._core.information_hessian`, so that both estimate the same posterior
    information (the convention of R mirt's ``SE.type = "crossprod"``).
    """
    n_free = eta.shape[0]
    if n_free == 0:
        return jnp.zeros((0, 0), dtype=eta.dtype)
    prior_curvature = -jax.hessian(lambda e: log_prior(prob, e))(eta)
    info: Array = score_cross_product(prob, eta) + prior_curvature
    return info
