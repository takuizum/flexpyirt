r"""Information, expected-score and summed-score EAP kernels (Core layer, DESIGN.md §18.2).

Pure JAX functions on :class:`~flexpyirt._core.CompiledProblem` pytrees for unidimensional
models. The item information comes from automatic differentiation of the category
log-probabilities of each item kernel; there are no hand-written item-family derivatives.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from flexpyirt._core import CompiledProblem, _expand, _item_log_probs, _latent_log_weights
from flexpyirt.items import ItemKernel
from flexpyirt.sumscore import summed_score_distribution

__all__ = ["category_probabilities", "item_curves", "summed_score_eap"]


def item_curves(prob: CompiledProblem, eta: Array, theta: Array) -> tuple[Array, Array]:
    r"""Return the item information and the expected item score of every group at ``theta``.

    .. math::
        I_j(\theta)=\sum_{k=0}^{K_j-1}P_{jk}(\theta)
            \Big(\frac{\partial\log P_{jk}(\theta)}{\partial\theta}\Big)^2,\qquad
        E_j(\theta)=\sum_{k=0}^{K_j-1}k\,P_{jk}(\theta)

    (Birnbaum, 1968; Samejima, 1969). ``theta`` has shape ``(T,)``; both results have shape
    ``(G, T, J)`` with items in column order.
    """
    n_items = prob.y.shape[1]
    shape = (prob.n_groups, theta.shape[0], n_items)
    information = jnp.zeros(shape, dtype=theta.dtype)
    expected = jnp.zeros(shape, dtype=theta.dtype)
    for fam in prob.families:
        xi = _expand(eta, fam.fixed_values, fam.free_index)  # (G, J_f, P_f)
        scores = jnp.arange(fam.kernel.n_categories, dtype=theta.dtype)

        def at_theta(
            t: Array, p: Array, k: ItemKernel = fam.kernel, s: Array = scores
        ) -> tuple[Array, Array]:
            def log_prob_at(u: Array) -> Array:
                return k.log_prob(u.reshape(1, 1), p)[0]

            log_p = log_prob_at(t)
            d_log_p = jax.jacfwd(log_prob_at)(t)
            p_cat = jnp.exp(log_p)
            return jnp.sum(p_cat * d_log_p * d_log_p), jnp.sum(s * p_cat)

        def per_item(p: Array) -> tuple[Array, Array]:
            return jax.vmap(lambda t: at_theta(t, p))(theta)

        info_f, exp_f = jax.vmap(jax.vmap(per_item))(xi)  # (G, J_f, T)
        information = information.at[:, :, fam.columns].set(jnp.swapaxes(info_f, 1, 2))
        expected = expected.at[:, :, fam.columns].set(jnp.swapaxes(exp_f, 1, 2))
    return information, expected


def category_probabilities(prob: CompiledProblem, eta: Array, theta: Array) -> Array:
    r"""Return the category probabilities :math:`P_{jk}(\theta)` of every group at ``theta``.

    .. math::
        P_{jk}(\theta)=\Pr(Y_j=k\mid\theta),\qquad k=0,\dots,K_j-1

    from each item kernel (the trace lines of Lord, 1980). ``theta`` has shape ``(T,)``; the
    result has shape ``(G, T, J, K_max)`` with items in column order and categories above an
    item's own :math:`K_j-1` padded with 0.
    """
    n_items = prob.y.shape[1]
    k_max = max(fam.kernel.n_categories for fam in prob.families)
    out = jnp.zeros((prob.n_groups, theta.shape[0], n_items, k_max), dtype=theta.dtype)
    for fam in prob.families:
        xi = _expand(eta, fam.fixed_values, fam.free_index)  # (G, J_f, P_f)

        def per_item(p: Array, k: ItemKernel = fam.kernel) -> Array:
            return jnp.exp(k.log_prob(theta.reshape(-1, 1), p))  # (T, K_f)

        p_f = jax.vmap(jax.vmap(per_item))(xi)  # (G, J_f, T, K_f)
        out = out.at[:, :, fam.columns, : fam.kernel.n_categories].set(jnp.swapaxes(p_f, 1, 2))
    return out


def summed_score_eap(prob: CompiledProblem, eta: Array) -> tuple[Array, Array, Array]:
    r"""Return the summed-score distribution, EAP and posterior SD of every group.

    With the quadrature nodes :math:`\theta_q`, the latent weights :math:`w_{gq}` and the
    Lord–Wingersky likelihood :math:`L_g(s\mid\theta_q)`
    (:func:`~flexpyirt.sumscore.summed_score_distribution`),

    .. math::
        P_g(s)=\sum_qL_g(s\mid\theta_q)w_{gq},\qquad
        \hat\theta_g(s)=\frac{\sum_q\theta_qL_g(s\mid\theta_q)w_{gq}}{P_g(s)},\qquad
        \mathrm{SD}_g(s)^2=\frac{\sum_q(\theta_q-\hat\theta_g(s))^2L_g(s\mid\theta_q)w_{gq}}{P_g(s)}

    (Thissen, Pommerich, Billeaud & Williams, 1995). The three results have shape
    ``(G, J (K_max - 1) + 1)``; scores above the largest possible score have probability 0, and
    scores with probability 0 have NaN EAP and SD.
    """
    n_items = prob.y.shape[1]
    n_nodes = prob.nodes.shape[0]
    k_max = max(fam.kernel.n_categories for fam in prob.families)
    probs = jnp.zeros((prob.n_groups, n_nodes, n_items, k_max), dtype=prob.nodes.dtype)
    for fam, log_p in zip(prob.families, _item_log_probs(prob, eta), strict=True):
        p_f = jnp.exp(jnp.swapaxes(log_p, 1, 2))  # (G, Q, J_f, K_f)
        probs = probs.at[:, :, fam.columns, : fam.kernel.n_categories].set(p_f)
    lik = summed_score_distribution(probs)  # (G, Q, S)
    weights = jnp.exp(_latent_log_weights(prob, eta))
    weights = weights / jnp.sum(weights, axis=1, keepdims=True)
    joint = lik * weights[:, :, None]
    nodes = prob.nodes[:, 0]
    p_score = jnp.sum(joint, axis=1)  # (G, S)
    positive = p_score > 0.0
    denom = jnp.where(positive, p_score, 1.0)
    mean = jnp.einsum("q,gqs->gs", nodes, joint) / denom
    var = jnp.sum((nodes[None, :, None] - mean[:, None, :]) ** 2 * joint, axis=1) / denom
    sd = jnp.sqrt(jnp.maximum(var, 0.0))
    return p_score, jnp.where(positive, mean, jnp.nan), jnp.where(positive, sd, jnp.nan)
