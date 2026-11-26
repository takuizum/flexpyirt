r"""Classification accuracy and consistency kernels (Core layer, DESIGN.md §18.4).

Pure JAX functions. Cut scores :math:`c_1<\dots<c_C` define the categories
:math:`h(v)=\#\{k: v\ge c_k\}=0,\dots,C`. Each method gives, at true values :math:`\theta_n`,
the probabilities :math:`P_h(\theta_n)` of the observed categories and the true categories
:math:`t(\theta_n)`; :func:`tables` turns them into the joint tables every index comes from.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

__all__ = ["categories", "lee_probabilities", "rudner_probabilities", "tables"]


def categories(values: Array, cuts: Array) -> Array:
    r"""Return the category :math:`h(v)=\#\{k: v\ge c_k\}` of each value, shape ``values.shape``."""
    return jnp.sum(values[..., None] >= cuts, axis=-1)


def lee_probabilities(lik: Array, cuts: Array) -> Array:
    r"""Return the observed-category probabilities of Lee (2010) from the summed-score likelihood.

    .. math:: P_h(\theta)=\sum_{s:\,h(s)=h}L(s\mid\theta)

    with the Lord–Wingersky likelihood ``lik`` of shape ``(..., S)`` (scores ``0 .. S - 1``)
    and summed-score cuts ``cuts`` of shape ``(C,)``; the result has shape ``(..., C + 1)``.
    """
    scores = jnp.arange(lik.shape[-1], dtype=cuts.dtype)
    membership = jax.nn.one_hot(categories(scores, cuts), cuts.shape[0] + 1, dtype=lik.dtype)
    return lik @ membership


def rudner_probabilities(theta: Array, se: Array, cuts: Array) -> Array:
    r"""Return the observed-category probabilities of Rudner (2001, 2005).

    With :math:`\hat\theta\sim N(\theta,\mathrm{SE}^2)`,

    .. math::
        P_h(\theta)=\Phi\Big(\frac{c_{h+1}-\theta}{\mathrm{SE}}\Big)
            -\Phi\Big(\frac{c_h-\theta}{\mathrm{SE}}\Big),\qquad c_0=-\infty,\ c_{C+1}=\infty.

    ``theta`` and ``se`` have shape ``(N,)``, ``cuts`` ``(C,)``; the result is ``(N, C + 1)``.
    """
    below = jax.scipy.stats.norm.cdf((cuts[None, :] - theta[:, None]) / se[:, None])
    zeros = jnp.zeros_like(below[:, :1])
    ones = jnp.ones_like(below[:, :1])
    return jnp.concatenate([below, ones], axis=1) - jnp.concatenate([zeros, below], axis=1)


def tables(probs: Array, true: Array, weights: Array) -> tuple[Array, Array]:
    r"""Return the accuracy table :math:`A` and the consistency table :math:`B`.

    .. math::
        A_{tk}=\sum_nw_n\,[t(\theta_n)=t]\,P_k(\theta_n),\qquad
        B_{hk}=\sum_nw_nP_h(\theta_n)P_k(\theta_n)

    (Lee, 2010): the joint distribution of the (true, observed) categories and of the observed
    categories of two independent administrations. ``probs`` is ``(N, H)``, ``true`` ``(N,)``
    integer categories and ``weights`` ``(N,)`` summing to 1; both tables are ``(H, H)``.
    """
    n_categories = probs.shape[-1]
    weighted = probs * weights[:, None]
    true_one_hot = jax.nn.one_hot(true, n_categories, dtype=probs.dtype)
    return true_one_hot.T @ weighted, probs.T @ weighted
