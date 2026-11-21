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
    "linking_vcov",
    "solve_linking",
    "transform_kernel_params",
]

LinkMethod = Literal["haebara", "stocking-lord"]

_N_HALVINGS = 40
"""Number of backtracking step sizes ``2^-k`` evaluated in parallel during 2-D Newton."""
_MAX_STEP = 1.0
"""Cap on the Euclidean norm of a single Newton step to prevent jumping to flat plateaus."""


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


def solve_linking(
    prob: LinkingProblem,
    eta_s: Array,
    eta_t: Array,
    ab0: Array,
    *,
    method: LinkMethod,
    symmetric: bool,
    tol: float | Array = 1e-10,
    max_iter: int | Array = 200,
) -> tuple[Array, Array, Array, Array, Array]:
    r"""Minimize :func:`linking_criterion` over :math:`A > 0, B \in \mathbb{R}` via Newton.

    Uses ``jax.grad`` and ``jax.hessian`` inside ``lax.while_loop`` with shape-static arrays
    (mirroring :mod:`flexpyirt._newton` and :mod:`flexpyirt._scoring`; DESIGN.md §16.5):

    - If the :math:`2\times 2` Hessian is positive definite, takes the exact Newton step
      :math:`\delta = -H^{-1}g`; otherwise adds Levenberg–Marquardt damping
      :math:`(H + \lambda I)^{-1}g` (falling back to :math:`-g` if still not a finite descent
      direction).
    - Evaluates 40 step sizes :math:`2^{-k}` in parallel with ``vmap``, projecting any candidate
      with :math:`A \le 0` halfway to the boundary (:math:`A/2`), and takes the largest step that
      does not increase the criterion (monotone descent).
    - Near the optimum where objective decrease hits float64 rounding, accepts a full interior
      Newton polish step when it reduces :math:`\|g\|_\infty` by more than half within 1-ULP
      objective noise.

    Parameters
    ----------
    prob : LinkingProblem
        Static-shape linking problem pytree.
    eta_s, eta_t : Array
        Common-item free parameter vectors for source and target.
    ab0 : Array, shape (2,)
        Initial ``[A, B]`` with ``A > 0``.
    method : {"haebara", "stocking-lord"}
        Static criterion selector.
    symmetric : bool
        Static flag for the symmetric criterion.
    tol : float or Array, default 1e-10
        Convergence tolerance on :math:`\|\nabla F(A, B)\|_\infty`.
    max_iter : int or Array, default 200
        Maximum number of Newton iterations.

    Returns
    -------
    ab : Array, shape (2,)
        Estimated ``[A, B]``.
    criterion : Array, scalar
        Criterion value at ``ab``.
    grad_norm : Array, scalar
        :math:`\|\nabla F(A, B)\|_\infty` at ``ab``.
    n_iter : Array, scalar int32
        Number of Newton iterations taken.
    converged : Array, scalar bool
        Whether ``grad_norm <= tol`` with finite criterion and gradient.
    """
    dtype = ab0.dtype
    tol_arr = jnp.asarray(tol, dtype=dtype)
    max_iter_arr = jnp.asarray(max_iter, dtype=jnp.int32)
    steps = 0.5 ** jnp.arange(_N_HALVINGS, dtype=dtype)
    eye2 = jnp.eye(2, dtype=dtype)

    def obj(ab: Array) -> Array:
        return linking_criterion(ab, eta_s, eta_t, prob, method=method, symmetric=symmetric)

    grad_fn = jax.grad(obj)
    hess_fn = jax.hessian(obj)

    v0 = obj(ab0)
    g0 = grad_fn(ab0)
    g0_inf = jnp.max(jnp.abs(g0))
    done0 = (~jnp.isfinite(v0)) | (~jnp.all(jnp.isfinite(g0))) | (g0_inf <= tol_arr)

    def cond(state: tuple[Array, Array, Array, Array, Array]) -> Array:
        _, _, _, it, done = state
        return (~done) & (it < max_iter_arr)

    def body(
        state: tuple[Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array]:
        x, v, g, it, _ = state
        g_inf = jnp.max(jnp.abs(g))
        h = hess_fn(x)
        h = 0.5 * (h + h.T)
        h_finite = jnp.all(jnp.isfinite(h))
        eigs = jnp.linalg.eigvalsh(jnp.where(h_finite, h, eye2))
        is_pd = h_finite & (eigs[0] > 1e-12)
        lam = jnp.where(is_pd, 0.0, jnp.maximum(0.0, -eigs[0]) + 1e-2)
        h_reg = jnp.where(h_finite, h + lam * eye2, eye2)
        d_newton = -jnp.linalg.solve(h_reg, g)
        is_descent = jnp.all(jnp.isfinite(d_newton)) & (jnp.dot(g, d_newton) < 0.0)
        neg_g = jnp.where(jnp.all(jnp.isfinite(g)), -g, jnp.zeros_like(g))
        direction = jnp.where(is_descent, d_newton, neg_g)
        dir_norm = jnp.linalg.norm(direction)
        direction = jnp.where(
            dir_norm > _MAX_STEP,
            direction * (_MAX_STEP / jnp.where(dir_norm > 0.0, dir_norm, 1.0)),
            direction,
        )

        raw = x[None, :] + steps[:, None] * direction[None, :]
        a_cand = jnp.where(raw[:, 0] <= 0.0, 0.5 * x[0], raw[:, 0])
        cands = raw.at[:, 0].set(a_cand)
        vals = jax.vmap(obj)(cands)
        inside = cands[:, 0] > 0.0
        ok = inside & jnp.isfinite(vals) & (vals <= v)
        any_ok = jnp.any(ok)
        best = cands[jnp.argmax(ok)]

        polish_x = cands[0]
        polish_val = vals[0]
        polish_g_inf = jnp.max(jnp.abs(grad_fn(polish_x)))
        polish_ok = (
            (~ok[0])
            & is_pd
            & inside[0]
            & jnp.isfinite(polish_val)
            & (polish_val <= v + 1e-14 * (1.0 + jnp.abs(v)))
            & (polish_g_inf < 0.5 * g_inf)
        )
        moved = polish_ok | any_ok
        x_next = jnp.where(polish_ok, polish_x, jnp.where(any_ok, best, x))
        actually_moved = moved & (jnp.max(jnp.abs(x_next - x)) > 0.0)
        v_next = obj(x_next)
        g_next = grad_fn(x_next)
        g_next_inf = jnp.max(jnp.abs(g_next))
        done = (
            (~jnp.isfinite(v_next))
            | (~jnp.all(jnp.isfinite(g_next)))
            | (g_next_inf <= tol_arr)
            | (~actually_moved)
        )
        return x_next, v_next, g_next, it + jnp.int32(1), done

    init = (ab0, v0, g0, jnp.int32(0), done0)
    x_fin, v_fin, g_fin, it_fin, _ = jax.lax.while_loop(cond, body, init)
    grad_norm = jnp.max(jnp.abs(g_fin))
    converged = jnp.isfinite(v_fin) & jnp.all(jnp.isfinite(g_fin)) & (grad_norm <= tol_arr)
    return x_fin, v_fin, grad_norm, it_fin, converged


def linking_vcov(
    ab_hat: Array,
    eta_s: Array,
    eta_t: Array,
    vcov_s: Array,
    vcov_t: Array,
    prob: LinkingProblem,
    *,
    method: LinkMethod,
    symmetric: bool,
) -> tuple[Array, Array]:
    r"""Asymptotic covariance of :math:`(\hat A, \hat B)` by the delta method (Ogasawara, 2001).

    At the stationary point :math:`g(\hat c, \hat\eta_s, \hat\eta_t) = \nabla_c F = 0` with
    :math:`c = (A, B)`, the implicit function theorem gives the Jacobians

    .. math::
        J_s = \frac{\partial \hat c}{\partial \eta_s^\top} = -H_{cc}^{-1} H_{c,\eta_s},
        \qquad
        J_t = \frac{\partial \hat c}{\partial \eta_t^\top} = -H_{cc}^{-1} H_{c,\eta_t},

    where :math:`H_{cc} = \partial^2 F/\partial c\,\partial c^\top` (``jax.jacfwd``) and
    :math:`H_{c,\eta_s}, H_{c,\eta_t}` (``jax.jacrev``) are computed via JAX autodiff. For two
    independently calibrated forms with common-item free-parameter covariances
    :math:`\Sigma_s, \Sigma_t` (treating the quadrature weights as fixed),

    .. math::
        \operatorname{Cov}(\hat A, \hat B) = J_s \Sigma_s J_s^\top + J_t \Sigma_t J_t^\top.

    Parameters
    ----------
    ab_hat : Array, shape (2,)
        Estimated linking constants ``[A, B]``.
    eta_s, eta_t : Array, shapes (m_s,), (m_t,)
        Common-item free parameters of source and target.
    vcov_s, vcov_t : Array, shapes (m_s, m_s), (m_t, m_t)
        Covariance matrices of ``eta_s`` and ``eta_t``.
    prob : LinkingProblem
        Static-shape linking problem pytree.
    method : {"haebara", "stocking-lord"}
        Static criterion selector.
    symmetric : bool
        Static flag for the symmetric criterion.

    Returns
    -------
    vcov : Array, shape (2, 2)
        Asymptotic covariance matrix of ``(A, B)`` (NaN if :math:`H_{cc}` is not positive
        definite).
    valid : Array, scalar bool
        ``True`` iff :math:`H_{cc}` is finite and positive definite and ``vcov`` is finite.
    """
    dtype = ab_hat.dtype
    eye2 = jnp.eye(2, dtype=dtype)

    def grad_c(ab: Array, es: Array, et: Array) -> Array:
        g: Array = jax.grad(
            lambda x: linking_criterion(x, es, et, prob, method=method, symmetric=symmetric)
        )(ab)
        return g

    h_cc = jax.jacfwd(grad_c, argnums=0)(ab_hat, eta_s, eta_t)
    h_cc = 0.5 * (h_cc + h_cc.T)
    h_finite = jnp.all(jnp.isfinite(h_cc))
    eigs = jnp.linalg.eigvalsh(jnp.where(h_finite, h_cc, eye2))
    is_pd = h_finite & (eigs[0] > 1e-14)
    safe_h = jnp.where(is_pd, h_cc, eye2)

    h_cs = jax.jacrev(grad_c, argnums=1)(ab_hat, eta_s, eta_t)
    h_ct = jax.jacrev(grad_c, argnums=2)(ab_hat, eta_s, eta_t)
    j_s = -jnp.linalg.solve(safe_h, h_cs)
    j_t = -jnp.linalg.solve(safe_h, h_ct)

    cov = j_s @ vcov_s @ j_s.T + j_t @ vcov_t @ j_t.T
    cov = 0.5 * (cov + cov.T)
    valid = is_pd & jnp.all(jnp.isfinite(cov))
    return jnp.where(valid, cov, jnp.nan), valid
