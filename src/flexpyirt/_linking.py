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
    NominalKernel,
)

__all__ = [
    "LinkMethod",
    "LinkTransformation",
    "LinkingFamily",
    "LinkingProblem",
    "linking_constants",
    "linking_criterion",
    "linking_vcov",
    "multidim_linking_criterion",
    "multidim_linking_vcov",
    "n_transformation_params",
    "solve_linking",
    "solve_multidim_linking",
    "transform_kernel_params",
    "transform_multidim_kernel_params",
    "unpack_transformation",
]

LinkMethod = Literal["haebara", "stocking_lord"]
LinkTransformation = Literal["diagonal", "orthogonal", "oblique"]

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
        Common items grouped by :class:`ItemKernel` (static length/kernels).
    tgt_nodes : Array, shape (Q_t,)
        Target-scale quadrature nodes :math:`\theta_q^{(t)}`. For multidimensional linking
        (DESIGN.md §19) both node arrays have shape ``(Q, D)``.
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
    - :class:`~flexpyirt.items.NominalKernel` (nominal response model): :math:`a^*=a/A`, scoring
      coefficients :math:`\mathrm{ak}_k^*=\mathrm{ak}_k` unchanged (:math:`k=1,\dots,K-2`), and
      :math:`d_k^*=d_k-\mathrm{ak}_k\,aB/A` for :math:`k=1,\dots,K-1` (with
      :math:`\mathrm{ak}_{K-1}=K-1`, :math:`d_0=0`), preserving
      :math:`\mathrm{ak}_k a^*\theta^*+d_k^*=\mathrm{ak}_k a\theta+d_k`.

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
        ks = jnp.arange(1, kernel.n_categories, dtype=params.dtype)
        return jnp.concatenate([a_new[None], params[1:] - ks * shift])
    if isinstance(kernel, NominalKernel):
        n_free_scoring = kernel.n_categories - 2
        scoring = jnp.concatenate(
            [
                params[1 : 1 + n_free_scoring],
                jnp.array([float(kernel.n_categories - 1)], dtype=params.dtype),
            ]
        )
        return jnp.concatenate(
            [
                a_new[None],
                params[1 : 1 + n_free_scoring],
                params[1 + n_free_scoring :] - scoring * shift,
            ]
        )
    raise SpecificationError(
        f"Scale parameter transformation is not implemented for kernel {type(kernel).__name__}."
    )


def _family_probs(kernel: ItemKernel, xi: Array, nodes: Array) -> Array:
    """Return ``(J_f, Q, K_f)`` category probabilities for one family on ``(Q, D)`` ``nodes``."""
    return jax.vmap(lambda p: jnp.exp(kernel.log_prob(nodes, p)))(xi)


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
    total_q = jnp.zeros_like(weights)
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
    tcc_fixed = jnp.zeros_like(weights)
    tcc_moved = jnp.zeros_like(weights)
    for fam, xi_fixed, xi_moved in zip(families, xi_fixed_list, xi_moved_list, strict=True):
        cats = jnp.arange(fam.kernel.n_categories, dtype=weights.dtype)
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
    method : {"haebara", "stocking_lord"}
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
    val = dir_fn(
        prob.families,
        xi_t_list,
        xi_s_list,
        prob.tgt_nodes[:, None],
        fwd_nodes[:, None],
        prob.tgt_weights,
    )
    if symmetric:
        bwd_nodes = safe_a * prob.src_nodes + b_const
        val = val + dir_fn(
            prob.families,
            xi_s_list,
            xi_t_list,
            prob.src_nodes[:, None],
            bwd_nodes[:, None],
            prob.src_weights,
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
      direction). The step's Euclidean norm is capped at ``_MAX_STEP = 1`` so that a poor start
      cannot jump onto a flat plateau of the criterion (where curves no longer overlap).
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
    method : {"haebara", "stocking_lord"}
        Static criterion selector.
    symmetric : bool
        Static flag for the symmetric criterion.
    tol : float or Array, default 1e-10
        Absolute convergence tolerance on :math:`\|\nabla F(A, B)\|_\infty`. It is not rescaled:
        :math:`F` uses weights summing to 1 and grows with the number of common items and (for
        Stocking–Lord) the squared score range, so the default is near float64 resolution for
        typical tests; loosen it for very long anchor sets if the solver reports
        non-convergence with a tiny gradient. Non-convergence (including a step that no longer
        decreases :math:`F`) is reported as ``converged = False``.
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
    vcov_st: Array | None = None,
) -> tuple[Array, Array]:
    r"""Asymptotic covariance of :math:`(\hat A, \hat B)` by the delta method (Ogasawara, 2001).

    At the stationary point :math:`g(\hat c, \hat\eta_s, \hat\eta_t) = \nabla_c F = 0` with
    :math:`c = (A, B)`, the implicit function theorem gives the Jacobians

    .. math::
        J_s = \frac{\partial \hat c}{\partial \eta_s^\top} = -H_{cc}^{-1} H_{c,\eta_s},
        \qquad
        J_t = \frac{\partial \hat c}{\partial \eta_t^\top} = -H_{cc}^{-1} H_{c,\eta_t},

    where :math:`H_{cc} = \partial^2 F/\partial c\,\partial c^\top` (``jax.jacfwd``) and
    :math:`H_{c,\eta_s}, H_{c,\eta_t}` (``jax.jacrev``) are computed via JAX autodiff. With
    common-item free-parameter covariances :math:`\Sigma_s, \Sigma_t` and cross-covariance
    :math:`\Sigma_{st} = \operatorname{Cov}(\hat\eta_s, \hat\eta_t)`,

    .. math::
        \operatorname{Cov}(\hat A, \hat B) = J_s \Sigma_s J_s^\top + J_t \Sigma_t J_t^\top
            + J_s \Sigma_{st} J_t^\top + J_t \Sigma_{st}^\top J_s^\top,

    where :math:`\Sigma_{st} = 0` for two independently calibrated forms (the default) and is the
    corresponding block of the joint covariance when both scales come from one (multiple-group)
    calibration, so that shared free parameters (``free_id`` equality constraints) are counted
    consistently.

    The quadrature weights are held fixed. When they are estimates (fitted latent mean/variance or
    empirical-histogram weights) this is exact to first order under a correctly specified model:
    :math:`\partial g / \partial w_q` is a weighted sum of curve residuals times curve
    derivatives, the residuals vanish at the population solution, so :math:`H_{c,w} = 0` there.
    Under item parameter drift / DIF (non-zero population residuals) the weights' sampling
    variability is ignored.

    The result is meaningful only at a stationary point of the criterion; callers must not use it
    when the optimizer did not converge.

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
    method : {"haebara", "stocking_lord"}
        Static criterion selector.
    symmetric : bool
        Static flag for the symmetric criterion.
    vcov_st : Array, shape (m_s, m_t), optional
        Cross-covariance of ``eta_s`` and ``eta_t`` (``None`` means zero: independent forms).

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
    if vcov_st is not None:
        cross = j_s @ vcov_st @ j_t.T
        cov = cov + cross + cross.T
    cov = 0.5 * (cov + cov.T)
    valid = is_pd & jnp.all(jnp.isfinite(cov))
    return jnp.where(valid, cov, jnp.nan), valid


# ---- Multidimensional linking: theta* = A theta + beta (DESIGN.md §19) -------------------------


def n_transformation_params(n_factors: int, transformation: LinkTransformation) -> int:
    r"""Return the number of free transformation parameters :math:`p` (DESIGN.md §19).

    :math:`2D` (``"diagonal"``), :math:`1 + D(D-1)/2 + D` (``"orthogonal"``) or :math:`D^2 + D`
    (``"oblique"``).
    """
    if transformation == "diagonal":
        return 2 * n_factors
    if transformation == "orthogonal":
        return 1 + n_factors * (n_factors - 1) // 2 + n_factors
    return n_factors * n_factors + n_factors


def unpack_transformation(
    c: Array, n_factors: int, transformation: LinkTransformation
) -> tuple[Array, Array, Array]:
    r"""Return :math:`(\mathbf A, \boldsymbol\beta, \text{valid})` from the parameter vector ``c``.

    - ``"diagonal"``: :math:`c = (A_1, \dots, A_D, \beta_1, \dots, \beta_D)`,
      :math:`\mathbf A = \operatorname{diag}(A_m)`, valid iff every :math:`A_m > 0`.
    - ``"orthogonal"`` (Li & Lissitz, 2000): :math:`c = (k, \phi_{12}, \phi_{13}, \dots,
      \boldsymbol\beta)`, :math:`\mathbf A = k\,\exp(\mathbf S(\boldsymbol\phi))` with the
      skew-symmetric :math:`S_{ij} = -\phi_{ij}`, :math:`S_{ji} = \phi_{ij}` (:math:`i < j`,
      lexicographic), a dilation :math:`k > 0` times a rotation (:math:`\det = +1`); valid iff
      :math:`k > 0`.
    - ``"oblique"`` (Oshima, Davey & Lee, 2000): :math:`c = (\operatorname{vec}(\mathbf A)
      \text{ row-major}, \boldsymbol\beta)`, valid iff :math:`\det\mathbf A > 0`.

    Invalid ``c`` returns the identity matrix (double-``where`` guard), so derivatives at valid
    points never see NaN.
    """
    d = n_factors
    eye = jnp.eye(d, dtype=c.dtype)
    if transformation == "diagonal":
        scales = c[:d]
        valid = jnp.all(scales > 0.0)
        return jnp.where(valid, jnp.diag(scales), eye), c[d:], valid
    if transformation == "orthogonal":
        n_angles = d * (d - 1) // 2
        rows, cols = jnp.triu_indices(d, k=1)
        angles = c[1 : 1 + n_angles]
        skew = jnp.zeros((d, d), dtype=c.dtype).at[rows, cols].set(-angles)
        skew = skew.at[cols, rows].set(angles)
        valid = c[0] > 0.0
        dilation = jnp.where(valid, c[0], 1.0)
        return dilation * jax.scipy.linalg.expm(skew), c[1 + n_angles :], valid
    a_mat = c[: d * d].reshape(d, d)
    valid = jnp.linalg.det(a_mat) > 0.0
    return jnp.where(valid, a_mat, eye), c[d * d :], valid


def linking_constants(c: Array, n_factors: int, transformation: LinkTransformation) -> Array:
    r"""Return :math:`(\operatorname{vec}(\mathbf A)\text{ row-major}, \boldsymbol\beta)`."""
    a_mat, beta, _ = unpack_transformation(c, n_factors, transformation)
    return jnp.concatenate([a_mat.reshape(-1), beta])


def transform_multidim_kernel_params(
    kernel: ItemKernel, params: Array, a_mat: Array, beta: Array
) -> Array:
    r"""Transform item parameters under :math:`\theta^*=\mathbf A\theta+\boldsymbol\beta`.

    Substituting :math:`\boldsymbol\theta=\mathbf A^{-1}(\boldsymbol\theta^*-\boldsymbol\beta)`
    into the linear predictor :math:`\mathbf a^\top\boldsymbol\theta+d` leaves every category
    probability invariant with (Oshima, Davey & Lee, 2000; DESIGN.md §19)

    .. math::
        \mathbf a^* = \mathbf A^{-\top}\mathbf a, \qquad
        d_k^* = d_k - m_k\,\mathbf a^{*\top}\boldsymbol\beta,

    where the multiplier :math:`m_k` of the linear predictor in category :math:`k` is 1
    (dichotomous, 3PL/4PL, graded), :math:`k` (GPCM/PCM) or :math:`\mathrm{ak}_k` (nominal, with
    :math:`\mathrm{ak}_{K-1}=K-1`). Asymptotes :math:`g, u` and nominal scoring coefficients are
    unchanged.

    Parameters
    ----------
    kernel : ItemKernel
        Item kernel with ``kernel.n_factors == D``.
    params : Array, shape (P,)
        Source-scale item parameters in ``kernel.param_names`` order.
    a_mat : Array, shape (D, D)
        Nonsingular transformation matrix :math:`\mathbf A`.
    beta : Array, shape (D,)
        Translation :math:`\boldsymbol\beta`.

    Returns
    -------
    Array, shape (P,)
        Item parameters on the target scale.
    """
    d = kernel.n_factors
    slopes = jnp.linalg.solve(a_mat.T, params[:d])
    shift = slopes @ beta
    rest = params[d:]
    if isinstance(kernel, DichotomousKernel):
        return jnp.concatenate([slopes, rest - shift])
    if isinstance(kernel, FourPLKernel):
        return jnp.concatenate([slopes, rest[:1] - shift, rest[1:]])
    if isinstance(kernel, GradedKernel):
        return jnp.concatenate([slopes, rest - shift])
    if isinstance(kernel, GPCMKernel):
        ks = jnp.arange(1, kernel.n_categories, dtype=params.dtype)
        return jnp.concatenate([slopes, rest - ks * shift])
    if isinstance(kernel, NominalKernel):
        n_free_scoring = kernel.n_categories - 2
        scoring = jnp.concatenate(
            [
                rest[:n_free_scoring],
                jnp.array([float(kernel.n_categories - 1)], dtype=params.dtype),
            ]
        )
        return jnp.concatenate(
            [slopes, rest[:n_free_scoring], rest[n_free_scoring:] - scoring * shift]
        )
    raise SpecificationError(
        f"Scale parameter transformation is not implemented for kernel {type(kernel).__name__}."
    )


def multidim_linking_criterion(
    c: Array,
    eta_s: Array,
    eta_t: Array,
    prob: LinkingProblem,
    *,
    method: LinkMethod,
    symmetric: bool,
    transformation: LinkTransformation,
) -> Array:
    r"""Multidimensional Haebara or Stocking–Lord criterion (DESIGN.md §19).

    With :math:`\boldsymbol\theta^*=\mathbf A\boldsymbol\theta+\boldsymbol\beta` mapping the
    source scale onto the target scale and :math:`(\boldsymbol\theta_q^{(t)}, w_q^{(t)})` the
    target product grid (Oshima, Davey & Lee, 2000, test and item characteristic function
    methods; category form Kim & Lee, 2006):

    .. math::
        H_{\mathrm{fwd}} = \sum_q w_q^{(t)} \sum_{j\in\mathcal C}\sum_{k=0}^{K_j-1}\Big(
            P_{jk}^{\mathrm{tgt}}(\boldsymbol\theta_q^{(t)})
            - P_{jk}^{\mathrm{src}}\big(\mathbf A^{-1}(\boldsymbol\theta_q^{(t)}
              -\boldsymbol\beta)\big)\Big)^2,

    .. math::
        \mathrm{SL}_{\mathrm{fwd}} = \sum_q w_q^{(t)}\Big(
            \mathrm{TCC}^{\mathrm{tgt}}(\boldsymbol\theta_q^{(t)})
            - \mathrm{TCC}^{\mathrm{src}}\big(\mathbf A^{-1}(\boldsymbol\theta_q^{(t)}
              -\boldsymbol\beta)\big)\Big)^2,
        \qquad \mathrm{TCC}(\boldsymbol\theta)=\sum_{j\in\mathcal C}\sum_k k\,P_{jk}
        (\boldsymbol\theta).

    ``symmetric=True`` adds the reverse term on the source grid, comparing source curves at
    :math:`\boldsymbol\theta_q^{(s)}` with target curves at
    :math:`\mathbf A\boldsymbol\theta_q^{(s)}+\boldsymbol\beta`. Returns ``NaN`` for an invalid
    ``c`` (see :func:`unpack_transformation`).

    Parameters
    ----------
    c : Array, shape (p,)
        Transformation parameters in the order of :func:`unpack_transformation`.
    eta_s, eta_t : Array
        Common-item free parameter vectors for source and target.
    prob : LinkingProblem
        Linking problem with ``(Q, D)`` node arrays.
    method : {"haebara", "stocking_lord"}
        Static criterion selector.
    symmetric : bool
        Static flag for the symmetric criterion.
    transformation : {"diagonal", "orthogonal", "oblique"}
        Static parameterization of :math:`\mathbf A`.

    Returns
    -------
    Array, scalar
    """
    n_factors = prob.tgt_nodes.shape[1]
    a_mat, beta, valid = unpack_transformation(c, n_factors, transformation)
    xi_s_list = tuple(_expand(eta_s, fam.src_params, fam.src_free_index) for fam in prob.families)
    xi_t_list = tuple(_expand(eta_t, fam.tgt_params, fam.tgt_free_index) for fam in prob.families)
    dir_fn = _haebara_direction if method == "haebara" else _stocking_lord_direction

    fwd_nodes = jnp.linalg.solve(a_mat, (prob.tgt_nodes - beta[None, :]).T).T
    val = dir_fn(prob.families, xi_t_list, xi_s_list, prob.tgt_nodes, fwd_nodes, prob.tgt_weights)
    if symmetric:
        bwd_nodes = prob.src_nodes @ a_mat.T + beta[None, :]
        val = val + dir_fn(
            prob.families, xi_s_list, xi_t_list, prob.src_nodes, bwd_nodes, prob.src_weights
        )
    return jnp.where(valid, val, jnp.nan)


def solve_multidim_linking(
    prob: LinkingProblem,
    eta_s: Array,
    eta_t: Array,
    c0: Array,
    *,
    method: LinkMethod,
    symmetric: bool,
    transformation: LinkTransformation,
    tol: float | Array = 1e-10,
    max_iter: int | Array = 200,
) -> tuple[Array, Array, Array, Array, Array]:
    r"""Minimize :func:`multidim_linking_criterion` over the transformation parameters ``c``.

    The same monotone Newton iteration as :func:`solve_linking` in :math:`p` dimensions: exact
    Newton step when the Hessian is positive definite, Levenberg–Marquardt damping otherwise,
    Euclidean step norm capped at ``_MAX_STEP``, 40 step sizes :math:`2^{-k}` evaluated in
    parallel, and a gradient-polish step at rounding level. Candidates outside the valid region
    (:math:`A_m \le 0`, :math:`k \le 0` or :math:`\det\mathbf A \le 0`) evaluate to ``NaN`` and
    are rejected by the step-size search, so the iterates stay in the connected component of the
    start (no reflections).

    Parameters
    ----------
    prob : LinkingProblem
        Linking problem with ``(Q, D)`` node arrays.
    eta_s, eta_t : Array
        Common-item free parameter vectors for source and target.
    c0 : Array, shape (p,)
        Valid starting transformation parameters.
    method, symmetric, transformation
        Static selectors as in :func:`multidim_linking_criterion`.
    tol : float or Array, default 1e-10
        Absolute tolerance on :math:`\|\nabla F(c)\|_\infty`.
    max_iter : int or Array, default 200
        Maximum number of Newton iterations.

    Returns
    -------
    c : Array, shape (p,)
    criterion : Array, scalar
    grad_norm : Array, scalar
    n_iter : Array, scalar int32
    converged : Array, scalar bool
    """
    dtype = c0.dtype
    n_params = c0.shape[0]
    tol_arr = jnp.asarray(tol, dtype=dtype)
    max_iter_arr = jnp.asarray(max_iter, dtype=jnp.int32)
    steps = 0.5 ** jnp.arange(_N_HALVINGS, dtype=dtype)
    eye = jnp.eye(n_params, dtype=dtype)

    def criterion_at(c: Array) -> Array:
        return multidim_linking_criterion(
            c, eta_s, eta_t, prob, method=method, symmetric=symmetric, transformation=transformation
        )

    grad_fn = jax.grad(criterion_at)
    hess_fn = jax.hessian(criterion_at)

    v0 = criterion_at(c0)
    g0 = grad_fn(c0)
    done0 = (~jnp.isfinite(v0)) | (~jnp.all(jnp.isfinite(g0))) | (jnp.max(jnp.abs(g0)) <= tol_arr)

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
        eigs = jnp.linalg.eigvalsh(jnp.where(h_finite, h, eye))
        is_pd = h_finite & (eigs[0] > 1e-12)
        lam = jnp.where(is_pd, 0.0, jnp.maximum(0.0, -eigs[0]) + 1e-2)
        h_reg = jnp.where(h_finite, h + lam * eye, eye)
        d_newton = -jnp.linalg.solve(h_reg, g)
        is_descent = jnp.all(jnp.isfinite(d_newton)) & (jnp.dot(g, d_newton) < 0.0)
        direction = jnp.where(is_descent, d_newton, -g)
        dir_norm = jnp.linalg.norm(direction)
        direction = jnp.where(
            dir_norm > _MAX_STEP,
            direction * (_MAX_STEP / jnp.where(dir_norm > 0.0, dir_norm, 1.0)),
            direction,
        )

        cands = x[None, :] + steps[:, None] * direction[None, :]
        vals = jax.vmap(criterion_at)(cands)
        ok = jnp.isfinite(vals) & (vals <= v)
        any_ok = jnp.any(ok)
        best = cands[jnp.argmax(ok)]

        polish_x = cands[0]
        polish_val = vals[0]
        polish_g = grad_fn(polish_x)
        polish_ok = (
            (~ok[0])
            & is_pd
            & jnp.isfinite(polish_val)
            & jnp.all(jnp.isfinite(polish_g))
            & (polish_val <= v + 1e-14 * (1.0 + jnp.abs(v)))
            & (jnp.max(jnp.abs(polish_g)) < 0.5 * g_inf)
        )
        x_next = jnp.where(polish_ok, polish_x, jnp.where(any_ok, best, x))
        actually_moved = (polish_ok | any_ok) & (jnp.max(jnp.abs(x_next - x)) > 0.0)
        v_next = criterion_at(x_next)
        g_next = grad_fn(x_next)
        done = (
            (~jnp.isfinite(v_next))
            | (~jnp.all(jnp.isfinite(g_next)))
            | (jnp.max(jnp.abs(g_next)) <= tol_arr)
            | (~actually_moved)
        )
        return x_next, v_next, g_next, it + jnp.int32(1), done

    init = (c0, v0, g0, jnp.int32(0), done0)
    x_fin, v_fin, g_fin, it_fin, _ = jax.lax.while_loop(cond, body, init)
    grad_norm = jnp.max(jnp.abs(g_fin))
    converged = jnp.isfinite(v_fin) & jnp.all(jnp.isfinite(g_fin)) & (grad_norm <= tol_arr)
    return x_fin, v_fin, grad_norm, it_fin, converged


def multidim_linking_vcov(
    c_hat: Array,
    eta_s: Array,
    eta_t: Array,
    vcov_s: Array,
    vcov_t: Array,
    prob: LinkingProblem,
    *,
    method: LinkMethod,
    symmetric: bool,
    transformation: LinkTransformation,
    vcov_st: Array | None = None,
) -> tuple[Array, Array, Array]:
    r"""Delta-method covariance of the multidimensional linking constants (Ogasawara, 2001).

    As in :func:`linking_vcov`, the stationarity condition :math:`\nabla_c F = 0` gives
    :math:`J_s = -H_{cc}^{-1}H_{c,\eta_s}`, :math:`J_t = -H_{cc}^{-1}H_{c,\eta_t}` and

    .. math::
        \operatorname{Cov}(\hat c) = J_s\Sigma_sJ_s^\top + J_t\Sigma_tJ_t^\top
            + J_s\Sigma_{st}J_t^\top + J_t\Sigma_{st}^\top J_s^\top .

    The reported constants :math:`\kappa(c) = (\operatorname{vec}(\mathbf A), \boldsymbol\beta)`
    are a smooth function of :math:`c`, so with :math:`G = \partial\kappa/\partial c^\top`
    (``jax.jacfwd``; DESIGN.md §19)

    .. math::
        \operatorname{Cov}(\hat\kappa) = G\operatorname{Cov}(\hat c)\,G^\top,

    which has rank at most :math:`p` (structural zeros of a diagonal :math:`\mathbf A` have zero
    variance; the entries of an orthogonal :math:`\mathbf A` are linearly dependent to first
    order).

    Parameters
    ----------
    c_hat : Array, shape (p,)
        Estimated transformation parameters.
    eta_s, eta_t, vcov_s, vcov_t, prob, vcov_st
        As in :func:`linking_vcov`.
    method, symmetric, transformation
        Static selectors as in :func:`multidim_linking_criterion`.

    Returns
    -------
    vcov : Array, shape (D*D + D, D*D + D)
        Covariance of :math:`(\operatorname{vec}(\mathbf A)\text{ row-major}, \boldsymbol\beta)`.
    vcov_params : Array, shape (p, p)
        Covariance of ``c``.
    valid : Array, scalar bool
        ``True`` iff :math:`H_{cc}` is finite and positive definite and the result is finite.
    """
    n_factors = prob.tgt_nodes.shape[1]
    eye = jnp.eye(c_hat.shape[0], dtype=c_hat.dtype)

    def grad_c(c: Array, es: Array, et: Array) -> Array:
        g: Array = jax.grad(
            lambda x: multidim_linking_criterion(
                x, es, et, prob, method=method, symmetric=symmetric, transformation=transformation
            )
        )(c)
        return g

    h_cc = jax.jacfwd(grad_c, argnums=0)(c_hat, eta_s, eta_t)
    h_cc = 0.5 * (h_cc + h_cc.T)
    h_finite = jnp.all(jnp.isfinite(h_cc))
    eigs = jnp.linalg.eigvalsh(jnp.where(h_finite, h_cc, eye))
    is_pd = h_finite & (eigs[0] > 1e-14)
    safe_h = jnp.where(is_pd, h_cc, eye)

    j_s = -jnp.linalg.solve(safe_h, jax.jacrev(grad_c, argnums=1)(c_hat, eta_s, eta_t))
    j_t = -jnp.linalg.solve(safe_h, jax.jacrev(grad_c, argnums=2)(c_hat, eta_s, eta_t))
    cov_c = j_s @ vcov_s @ j_s.T + j_t @ vcov_t @ j_t.T
    if vcov_st is not None:
        cross = j_s @ vcov_st @ j_t.T
        cov_c = cov_c + cross + cross.T
    cov_c = 0.5 * (cov_c + cov_c.T)

    jac = jax.jacfwd(lambda x: linking_constants(x, n_factors, transformation))(c_hat)
    cov = jac @ cov_c @ jac.T
    cov = 0.5 * (cov + cov.T)
    valid = is_pd & jnp.all(jnp.isfinite(cov))
    return jnp.where(valid, cov, jnp.nan), jnp.where(valid, cov_c, jnp.nan), valid
