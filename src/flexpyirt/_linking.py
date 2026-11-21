"""Pure JAX core for differentiable IRT scale linking (DESIGN.md §16.5).

All functions here operate on static-shape pytrees (:class:`LinkingFamily`,
:class:`LinkingProblem`) and JAX arrays only; no strings, DataFrames, or Python-level validation.
Category characteristic curves come directly from each :class:`~flexpyirt.items.ItemKernel`, and
all derivatives (for Newton optimization and Ogasawara delta-method SEs) come from JAX automatic
differentiation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

from flexpyirt._core import _expand
from flexpyirt.errors import SpecificationError
from flexpyirt.items import (
    DichotomousKernel,
    FourPLKernel,
    GPCMKernel,
    GradedKernel,
    ItemKernel,
)

__all__ = [
    "LinkMethod",
    "LinkingFamily",
    "LinkingProblem",
    "linking_criterion",
    "transform_kernel_params",
]

LinkMethod = Literal["haebara", "stocking-lord"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LinkingFamily:
    """Common items sharing one unidimensional :class:`ItemKernel` (Core pytree).

    Attributes
    ----------
    kernel : ItemKernel
        Static unidimensional item kernel shared by all items in this family.
    src_params : Array, shape (J_f, P_f)
        Source-scale item parameter values (used directly for fixed entries).
    tgt_params : Array, shape (J_f, P_f)
        Target-scale item parameter values (used directly for fixed entries).
    src_free_index : Array, shape (J_f, P_f)
        Int32 index into the source common free-parameter vector ``eta_s`` (``-1`` if fixed).
    tgt_free_index : Array, shape (J_f, P_f)
        Int32 index into the target common free-parameter vector ``eta_t`` (``-1`` if fixed).
    """

    kernel: ItemKernel = field(metadata={"static": True})
    src_params: Array
    tgt_params: Array
    src_free_index: Array
    tgt_free_index: Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LinkingProblem:
    r"""Static-shape pytree describing a two-scale characteristic-curve linking problem.

    Attributes
    ----------
    families : tuple of LinkingFamily
        Common items grouped by unidimensional :class:`ItemKernel` (static length/kernels).
    tgt_nodes : Array, shape (Q_t,)
        Target-scale quadrature nodes :math:`\theta_q^{(t)}`.
    tgt_weights : Array, shape (Q_t,)
        Target-scale quadrature weights :math:`w_q^{(t)}` (non-negative, summing to 1).
    src_nodes : Array, shape (Q_s,)
        Source-scale quadrature nodes :math:`\theta_q^{(s)}` (used when ``symmetric=True``).
    src_weights : Array, shape (Q_s,)
        Source-scale quadrature weights :math:`w_q^{(s)}` (used when ``symmetric=True``).
    """

    families: tuple[LinkingFamily, ...]
    tgt_nodes: Array
    tgt_weights: Array
    src_nodes: Array
    src_weights: Array


def transform_kernel_params(
    kernel: ItemKernel,
    params: Array,
    a_const: Array,
    b_const: Array,
) -> Array:
    r"""Transform unidimensional slope–intercept item parameters under :math:`\theta^*=A\theta+B`.

    Substituting :math:`\theta=(\theta^*-B)/A` (:math:`A>0`) leaves category response
    probabilities invariant, :math:`P(y=k\mid\theta^*;\xi^*)=P(y=k\mid\theta;\xi)` (Kolen &
    Brennan, 2014; DESIGN.md §16.5):

    - :class:`~flexpyirt.items.DichotomousKernel` (2PL/Rasch):
      :math:`a^*=a/A`, :math:`d^*=d-aB/A`.
    - :class:`~flexpyirt.items.FourPLKernel` (3PL/4PL):
      :math:`a^*=a/A`, :math:`d^*=d-aB/A`, asymptotes :math:`g^*=g`, :math:`u^*=u` unchanged.
    - :class:`~flexpyirt.items.GradedKernel` (GRM):
      :math:`a^*=a/A`, :math:`d_k^*=d_k-aB/A` for :math:`k=1,\dots,K-1` (preserving
      :math:`d_1>\dots>d_{K-1}`).
    - :class:`~flexpyirt.items.GPCMKernel` (GPCM/PCM):
      :math:`a^*=a/A`, :math:`d_k^*=d_k-k\,aB/A` for :math:`k=1,\dots,K-1` (with :math:`d_0=0`),
      preserving :math:`k\,a^*\theta^*+d_k^*=k\,a\theta+d_k` and step difficulties
      :math:`b_k^*=A\,b_k+B`.

    Parameters
    ----------
    kernel : ItemKernel
        Unidimensional item kernel (``kernel.n_factors == 1``).
    params : Array, shape (P,)
        Item parameters on the source scale in ``kernel.param_names`` order.
    a_const, b_const : Array, scalar
        Linear transformation constants :math:`(A, B)` with :math:`A > 0`.

    Returns
    -------
    Array, shape (P,)
        Transformed item parameters on the target scale.
    """
    a_new = params[0] / a_const
    shift = a_new * b_const  # a * B / A
    if isinstance(kernel, DichotomousKernel):
        return jnp.array([a_new, params[1] - shift], dtype=params.dtype)
    if isinstance(kernel, FourPLKernel):
        return jnp.array([a_new, params[1] - shift, params[2], params[3]], dtype=params.dtype)
    if isinstance(kernel, GradedKernel):
        return jnp.concatenate([a_new[None], params[1:] - shift])
    if isinstance(kernel, GPCMKernel):
        ks = jnp.arange(1, kernel.n_cat, dtype=params.dtype)
        return jnp.concatenate([a_new[None], params[1:] - ks * shift])
    raise SpecificationError(
        f"Scale parameter transformation is not implemented for kernel {type(kernel).__name__}."
    )


def _family_probs(kernel: ItemKernel, xi: Array, nodes_1d: Array) -> Array:
    """Return ``(J_f, Q, K_f)`` category probabilities for one family on 1-D ``nodes_1d``."""
    theta = nodes_1d[:, None]
    return jax.vmap(lambda p: jnp.exp(kernel.log_prob(theta, p)))(xi)


def _haebara_direction(
    families: tuple[LinkingFamily, ...],
    xi_fixed_list: tuple[Array, ...],
    xi_moved_list: tuple[Array, ...],
    nodes: Array,
    moved_nodes: Array,
    weights: Array,
) -> Array:
    r"""Single-direction Haebara squared category-curve difference summed over families.

    .. math::
        \sum_{q=1}^{Q} w_q \sum_{j\in\mathcal{C}} \sum_{k=0}^{K_j-1}
        \Big(P_{jk}^{\text{fixed}}(\theta_q) - P_{jk}^{\text{moved}}(\tilde\theta_q)\Big)^2.
    """
    total_q = jnp.zeros_like(nodes)
    for fam, xi_fixed, xi_moved in zip(families, xi_fixed_list, xi_moved_list, strict=True):
        p_fixed = _family_probs(fam.kernel, xi_fixed, nodes)
        p_moved = _family_probs(fam.kernel, xi_moved, moved_nodes)
        total_q = total_q + jnp.sum((p_fixed - p_moved) ** 2, axis=(0, 2))
    return jnp.sum(weights * total_q)


def _stocking_lord_direction(
    families: tuple[LinkingFamily, ...],
    xi_fixed_list: tuple[Array, ...],
    xi_moved_list: tuple[Array, ...],
    nodes: Array,
    moved_nodes: Array,
    weights: Array,
) -> Array:
    r"""Single-direction Stocking–Lord squared TCC difference summed over families.

    .. math::
        \sum_{q=1}^{Q} w_q \left(
            \sum_{j\in\mathcal{C}} E_j^{\text{fixed}}(\theta_q)
            - \sum_{j\in\mathcal{C}} E_j^{\text{moved}}(\tilde\theta_q)
        \right)^2,\qquad
        E_j(\theta) = \sum_{k=0}^{K_j-1} k\,P_{jk}(\theta).
    """
    tcc_fixed = jnp.zeros_like(nodes)
    tcc_moved = jnp.zeros_like(nodes)
    for fam, xi_fixed, xi_moved in zip(families, xi_fixed_list, xi_moved_list, strict=True):
        cats = jnp.arange(fam.kernel.n_cat, dtype=nodes.dtype)
        p_fixed = _family_probs(fam.kernel, xi_fixed, nodes)
        p_moved = _family_probs(fam.kernel, xi_moved, moved_nodes)
        tcc_fixed = tcc_fixed + jnp.sum(p_fixed * cats[None, None, :], axis=(0, 2))
        tcc_moved = tcc_moved + jnp.sum(p_moved * cats[None, None, :], axis=(0, 2))
    return jnp.sum(weights * (tcc_fixed - tcc_moved) ** 2)


def linking_criterion(
    ab: Array,
    eta_s: Array,
    eta_t: Array,
    prob: LinkingProblem,
    *,
    method: LinkMethod,
    symmetric: bool,
) -> Array:
    r"""Evaluate the Haebara or Stocking–Lord linking criterion at :math:`(A, B)`.

    For linear scale transformation :math:`\theta^* = A\theta + B` (:math:`A > 0`), the
    transformed source curve at :math:`\theta^*` equals the source curve at
    :math:`(\theta^*-B)/A`, and the target curve transformed to the source scale at :math:`\theta`
    equals the target curve at :math:`A\theta+B`:

    - **Haebara** (Haebara, 1980; polytomous form Kim & Lee, 2006):

      .. math::
          H_{\mathrm{fwd}}(A, B) = \sum_{q=1}^{Q_t} w_q^{(t)} \sum_{j\in\mathcal{C}}
              \sum_{k=0}^{K_j-1}\Big(
                  P_{jk}^{\mathrm{tgt}}(\theta_q^{(t)})
                  - P_{jk}^{\mathrm{src}}\big((\theta_q^{(t)}-B)/A\big)
              \Big)^2.

    - **Stocking–Lord** (Stocking & Lord, 1983; polytomous form Kim & Lee, 2006):

      .. math::
          \mathrm{SL}_{\mathrm{fwd}}(A, B) = \sum_{q=1}^{Q_t} w_q^{(t)} \Big(
              \mathrm{TCC}^{\mathrm{tgt}}(\theta_q^{(t)})
              - \mathrm{TCC}^{\mathrm{src}}\big((\theta_q^{(t)}-B)/A\big)
          \Big)^2,

      where
      :math:`\mathrm{TCC}(\theta) = \sum_{j\in\mathcal{C}}\sum_{k=0}^{K_j-1} k\,P_{jk}(\theta)`.

    - **Symmetric variants** (``symmetric=True``; Kim & Lee, 2006): add the reverse-direction
      criterion evaluated on ``prob.src_nodes`` with weights ``prob.src_weights`` comparing source
      curves at :math:`\theta_q^{(s)}` against target curves at :math:`A\theta_q^{(s)} + B`.

    Returns ``NaN`` when :math:`A \le 0` (double-``where`` guarded so derivatives at :math:`A > 0`
    are finite and NaN-free).

    Parameters
    ----------
    ab : Array, shape (2,)
        Linking constants ``[A, B]``.
    eta_s, eta_t : Array
        Common-item free parameter vectors for source and target.
    prob : LinkingProblem
        Static-shape linking problem pytree.
    method : {"haebara", "stocking-lord"}
        Static criterion selector.
    symmetric : bool
        Static flag for the symmetric criterion.

    Returns
    -------
    Array, scalar
        Criterion value (non-negative when :math:`A > 0`, ``NaN`` otherwise).
    """
    a_const, b_const = ab[0], ab[1]
    valid = a_const > 0.0
    safe_a = jnp.where(valid, a_const, 1.0)

    xi_s_list = tuple(_expand(eta_s, fam.src_params, fam.src_free_index) for fam in prob.families)
    xi_t_list = tuple(_expand(eta_t, fam.tgt_params, fam.tgt_free_index) for fam in prob.families)

    dir_fn = _haebara_direction if method == "haebara" else _stocking_lord_direction

    fwd_nodes = (prob.tgt_nodes - b_const) / safe_a
    val = dir_fn(prob.families, xi_t_list, xi_s_list, prob.tgt_nodes, fwd_nodes, prob.tgt_weights)
    if symmetric:
        bwd_nodes = safe_a * prob.src_nodes + b_const
        val = val + dir_fn(
            prob.families, xi_s_list, xi_t_list, prob.src_nodes, bwd_nodes, prob.src_weights
        )
    return jnp.where(valid, val, jnp.nan)
