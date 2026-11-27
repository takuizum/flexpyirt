r"""Core layer: per-pattern scores and the cross-product information (DESIGN.md §4.8, §17.5).

Pure JAX functions on the :class:`~flexpyirt._core.CompiledProblem` pytree. ``fit()`` inverts
:func:`information_cross_product` for ``se="crossprod"`` and combines it with the observed
information of :func:`flexpyirt._core.information_oakes` for ``se="sandwich"``. Under sampling
weights the sandwich's middle matrix uses squared weights (:func:`score_cross_product` on a problem
whose counts are :math:`\sum_i w_i^2`), and the cluster-robust version sums the scores within
clusters (:func:`score_cluster_cross_product`; DESIGN.md §17.6).
"""

from __future__ import annotations

from collections.abc import Callable

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
    "score_cluster_cross_product",
    "score_cross_product",
    "score_sum",
]

_CHUNK_ELEMENTS = 8_000_000
"""Target element budget (~64 MB in float64) for the per-chunk reverse-mode workspace."""
_MAX_CHUNK = 1024
"""Maximum number of response patterns differentiated per chunk."""


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


def _fold_pattern_scores[Carry](
    prob: CompiledProblem,
    eta: Array,
    init: Carry,
    accumulate: Callable[[Carry, Array, Array, Array], Carry],
    labels: Array | None = None,
) -> Carry:
    r"""Fold ``accumulate(carry, scores, counts, labels)`` over pattern chunks of the score matrix.

    ``scores`` is a ``(B, n_free)`` block of per-pattern scores
    :math:`s_p=\nabla_\eta\log L_p(\eta)`, ``counts`` the ``(B,)`` pattern counts (zero on
    padding rows) and ``labels`` the ``(B,)`` chunk of ``labels`` (zero when not given; the
    cluster of each pattern for :func:`score_cluster_cross_product`). The node tables
    :math:`\log P_{gjqk}(\eta)` and :math:`\log w_{gq}(\eta)` are evaluated and linearized once
    (``jax.vjp``); each chunk differentiates its pattern log-likelihoods with respect to the
    tables by reverse mode (``vmap`` of ``jax.grad`` over the patterns) and pulls the result back
    to :math:`\eta`, so neither ``(P, n_free)`` nor ``(P, G, J, Q, K)`` is materialized. Chunks
    that contain only zero-count padding rows are skipped.
    """

    def node_tables(e: Array) -> tuple[tuple[Array, ...], Array]:
        return tuple(_item_log_probs(prob, e)), _latent_log_weights(prob, e)

    tables, pullback = jax.vjp(node_tables, eta)

    def pattern_loglik(
        tabs: tuple[tuple[Array, ...], Array], one_hots: tuple[Array, ...], group: Array
    ) -> Array:
        """Return one pattern's log-likelihood; ``one_hots`` is ``(J_f, K_f)`` per family."""
        log_probs, log_weights = tabs
        ell = log_weights[group]
        for one_hot, log_prob in zip(one_hots, log_probs, strict=True):
            ell = ell + jnp.einsum("jk,jqk->q", one_hot, log_prob[group])
        return jax.scipy.special.logsumexp(ell)

    table_scores = jax.vmap(jax.grad(pattern_loglik), in_axes=(None, 0, 0))

    n_patterns = prob.y.shape[0]
    chunk = min(_chunk_size(prob), n_patterns)
    n_chunks = (n_patterns + chunk - 1) // chunk
    pad = n_chunks * chunk - n_patterns

    def chunked(x: Array) -> Array:
        padded = jnp.concatenate([x, jnp.zeros((pad, *x.shape[1:]), dtype=x.dtype)], axis=0)
        return padded.reshape(n_chunks, chunk, *x.shape[1:])

    if labels is None:
        labels = jnp.zeros(n_patterns, dtype=jnp.int32)
    chunks = (
        tuple(chunked(one_hot) for one_hot in _one_hots(prob)),
        chunked(prob.group),
        chunked(prob.counts),
        chunked(labels),
    )

    def step(carry: Carry, x: tuple[tuple[Array, ...], Array, Array, Array]) -> tuple[Carry, None]:
        one_hots, group, counts, chunk_labels = x

        def with_scores(c: Carry) -> Carry:
            (scores,) = jax.vmap(pullback)(table_scores(tables, one_hots, group))
            return accumulate(c, scores, counts, chunk_labels)

        return jax.lax.cond(jnp.any(counts > 0), with_scores, lambda c: c, carry), None

    out, _ = jax.lax.scan(step, init, chunks)
    return out


def score_sum(prob: CompiledProblem, eta: Array) -> Array:
    r"""Return the count-weighted sum of the per-pattern scores.

    :math:`\sum_p c_p s_p=\nabla_\eta\ell(\eta)`, which equals ``jax.grad`` of
    :func:`flexpyirt._core.marginal_loglik`; used to check the chunked scores.
    """

    def accumulate(total: Array, scores: Array, counts: Array, _labels: Array) -> Array:
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

    def accumulate(total: Array, scores: Array, counts: Array, _labels: Array) -> Array:
        return total + (counts[:, None] * scores).T @ scores

    return _fold_pattern_scores(prob, eta, jnp.zeros((n_free, n_free), dtype=eta.dtype), accumulate)


def score_cluster_cross_product(
    prob: CompiledProblem, eta: Array, cluster: Array, n_clusters: int
) -> Array:
    r"""Return the cross-product of the per-cluster sums of the weighted scores.

    .. math::
        U_c=\sum_{p:\,\mathrm{cluster}(p)=c} c_p\,s_p,\qquad
        B_{\mathrm{cl}}=\sum_{c=1}^{G}U_c U_c^\top

    Binder (1983); Williams (2000). ``prob`` holds one pattern per (group, cluster, response
    pattern), so ``prob.counts`` is the sum of the weights of its rows (the number of rows when
    unweighted) and ``cluster`` the ``(P,)`` cluster index in ``0..n_clusters-1`` of each pattern
    (any index on zero-count padding rows). ``n_clusters`` is static. The small-sample factor
    :math:`G/(G-1)` and the prior curvature are added by the caller (DESIGN.md §17.6). With one
    pattern per cluster this is :func:`score_cross_product` with squared counts.
    """
    n_free = eta.shape[0]
    if n_free == 0:
        return jnp.zeros((0, 0), dtype=eta.dtype)

    def accumulate(totals: Array, scores: Array, counts: Array, labels: Array) -> Array:
        return totals.at[labels].add(counts[:, None] * scores)

    init = jnp.zeros((n_clusters, n_free), dtype=eta.dtype)
    totals = _fold_pattern_scores(prob, eta, init, accumulate, labels=cluster)
    return totals.T @ totals


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
