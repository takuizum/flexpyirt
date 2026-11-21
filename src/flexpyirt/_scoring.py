"""Ability scoring kernels (Core layer, DESIGN.md §8, §15.21–§15.25).

Pure JAX functions on :class:`~flexpyirt._core.CompiledProblem` pytrees. Every derivative (score,
observed information, Fisher test information, and its derivative for WLE) comes from JAX
automatic differentiation — no hand-written item-family derivatives. All shapes are static so
every kernel runs inside ``jit`` with compressed, power-of-two-padded response patterns.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import jax
import jax.numpy as jnp
from jax import Array

from flexpyirt._core import CompiledProblem, FamilyBlock, _expand, _log_joint, eap
from flexpyirt.items import ItemKernel

__all__ = [
    "map_scores",
    "ml_scores",
    "plausible_values",
    "wle_scores",
]

_N_HALVINGS = 30
"""Number of backtracking step sizes ``2^-k`` evaluated together in 1-D safeguarded Newton."""
_MAX_STEP = 10.0
"""Cap on a single 1-D Newton or gradient-fallback step to avoid exponential overflow."""
_MAX_ITER = 80
"""Maximum Newton iterations per pattern in ``_newton_1d``."""
_GRAD_TOL = 1e-11
"""First-derivative convergence tolerance for 1-D safeguarded Newton."""
_STEP_TOL = 1e-12
"""Minimum step size below which 1-D safeguarded Newton stops."""
_MAX_THETA = 20.0
"""Absolute bound for ML and WLE optimization (matches mirt's default ``max_theta = 20``)."""
_ML_MAX_SE = 30.0
"""SE threshold above which a flat-tail ML solution is treated as diverging (mirt convention)."""
_PV_GRID_POINTS = 501
"""Number of equispaced nodes for continuous piecewise-linear inverse-CDF PV sampling."""
_PV_LATTICE_POINTS = {2: 61, 3: 25}
"""Points per dimension of the per-pattern PV lattice for ``D = 2, 3`` (DESIGN.md §15.42)."""
_PV_SPAN_SD = 6.0
"""Half-width of the per-pattern PV lattice in posterior SDs around EAP."""


def _expand_items(prob: CompiledProblem, eta: Array) -> tuple[Array, ...]:
    """Per family, ``(G, J_f, P_f)`` item parameters expanded from ``eta``."""
    return tuple(_expand(eta, fam.fixed_values, fam.free_index) for fam in prob.families)


def _expand_latent(prob: CompiledProblem, eta: Array) -> Array:
    """``(G, L_D)`` latent parameters (``(mean, var)`` for ``D = 1``) expanded from ``eta``."""
    return _expand(eta, prob.latent.fixed_values, prob.latent.free_index)


def _pattern_loglik(
    theta: Array,
    y_p: Array,
    g_p: Array,
    families: tuple[FamilyBlock, ...],
    xi_list: tuple[Array, ...],
) -> Array:
    r"""Log-likelihood :math:`\log L(\theta\mid y_p)` for one pattern, dropping missing items."""
    t1 = theta[None]
    total = jnp.zeros((), dtype=theta.dtype)
    for fam, xi in zip(families, xi_list, strict=True):
        y_f = y_p[fam.columns]  # (J_f,)
        xi_g = xi[g_p]  # (J_f, P_f)

        def _lp_item(p: Array, k: ItemKernel = fam.kernel) -> Array:
            return k.log_prob(t1, p)[0]

        lp = jax.vmap(_lp_item)(xi_g)  # (J_f, K_f)
        y_safe = jnp.maximum(y_f, 0)
        picked = lp[jnp.arange(y_f.shape[0]), y_safe]
        total = total + jnp.sum(jnp.where(y_f >= 0, picked, 0.0))
    return total


def _pattern_info(
    theta: Array,
    y_p: Array,
    g_p: Array,
    families: tuple[FamilyBlock, ...],
    xi_list: tuple[Array, ...],
) -> Array:
    r"""Observed-item Fisher test information :math:`I(\theta)` via JAX autodiff.

    .. math::
        I(\theta) = \sum_{j:\,y_{pj}\ge 0}\sum_{k=0}^{K_j-1}
            P_{jk}(\theta)\left(\frac{\partial\log P_{jk}(\theta)}{\partial\theta}\right)^2.
    """
    total = jnp.zeros((), dtype=theta.dtype)
    for fam, xi in zip(families, xi_list, strict=True):
        y_f = y_p[fam.columns]
        xi_g = xi[g_p]

        def _item_info(p: Array, k: ItemKernel = fam.kernel) -> Array:
            def lp_fn(t: Array) -> Array:
                return k.log_prob(t[None], p)[0]

            lp = lp_fn(theta)
            dlp = jax.jacfwd(lp_fn)(theta)
            return jnp.sum(jnp.exp(lp) * (dlp * dlp))

        infos = jax.vmap(_item_info)(xi_g)
        total = total + jnp.sum(jnp.where(y_f >= 0, infos, 0.0))
    return total


def _pattern_flags(
    y_p: Array,
    g_p: Array,
    families: tuple[FamilyBlock, ...],
    xi_list: tuple[Array, ...],
) -> tuple[Array, Array, Array]:
    """Return ``(all_missing, all_low, all_high)`` scalar booleans for one pattern."""
    n_obs = jnp.asarray(0, dtype=jnp.int32)
    n_low = jnp.asarray(0, dtype=jnp.int32)
    n_high = jnp.asarray(0, dtype=jnp.int32)
    for fam, xi in zip(families, xi_list, strict=True):
        y_f = y_p[fam.columns]
        # Every ItemKernel places the slope parameter "a" at index 0.
        # Items with a == 0 carry zero information about theta and are treated like missing.
        a = xi[g_p, :, 0]
        obs = (y_f >= 0) & (a != 0.0)
        top = fam.kernel.n_cat - 1
        low_cat = jnp.where(a > 0.0, 0, top)
        high_cat = jnp.where(a > 0.0, top, 0)
        n_obs = n_obs + jnp.sum(obs, dtype=jnp.int32)
        n_low = n_low + jnp.sum(obs & (y_f == low_cat), dtype=jnp.int32)
        n_high = n_high + jnp.sum(obs & (y_f == high_cat), dtype=jnp.int32)
    all_missing = n_obs == 0
    all_low = (n_obs > 0) & (n_low == n_obs)
    all_high = (n_obs > 0) & (n_high == n_obs)
    return all_missing, all_low, all_high


def _newton_1d(f: Callable[[Array], Array], x0: Array, lo: Array, hi: Array) -> Array:
    r"""Safeguarded 1-D Newton ascent maximizing scalar ``f`` over ``(lo, hi)`` (§15.22).

    First and second derivatives come from JAX autodiff. Falls back to a capped gradient step
    when :math:`f''(x)\ge -10^{-12}` (non-concave regions, e.g. 3PL/4PL) and evaluates step sizes
    :math:`2^{-k}` (:math:`k=0,\dots,29`) in parallel with ``vmap`` so :math:`f` never decreases.
    """
    grad_fn = jax.grad(f)
    hess_fn = jax.grad(grad_fn)
    zero = jnp.asarray(0, dtype=jnp.int32)
    one = jnp.asarray(1, dtype=jnp.int32)
    steps = 0.5 ** jnp.arange(_N_HALVINGS, dtype=x0.dtype)

    def cond(state: tuple[Array, Array, Array]) -> Array:
        _, it, done = state
        return (~done) & (it < _MAX_ITER)

    def body(state: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        x, it, _ = state
        fx = f(x)
        g = grad_fn(x)
        h = hess_fn(x)
        concave = jnp.isfinite(h) & (h < -1e-12)
        newton_dir = jnp.where(concave, -g / jnp.where(concave, h, -1.0), g)
        is_ascent = jnp.isfinite(newton_dir) & (g * newton_dir > 0.0)
        direction = jnp.where(is_ascent, newton_dir, jnp.where(jnp.isfinite(g), g, 0.0))
        direction = jnp.clip(direction, -_MAX_STEP, _MAX_STEP)

        raw = x + steps * direction
        lo_val = jnp.where(jnp.isfinite(lo), lo, x)
        hi_val = jnp.where(jnp.isfinite(hi), hi, x)
        cands = jnp.where(
            raw <= lo,
            x + 0.5 * (lo_val - x),
            jnp.where(raw >= hi, x + 0.5 * (hi_val - x), raw),
        )
        vals = jax.vmap(f)(cands)
        inside = (cands > lo) & (cands < hi)
        ok = inside & jnp.isfinite(vals) & (vals >= fx)
        any_ok = jnp.any(ok)
        best = cands[jnp.argmax(ok)]
        # Near the mode, f(x) is flat to float64 rounding (~sqrt(eps) ~ 1e-8 in x); if strict
        # value ascent hits the 1-ULP floor on a strictly concave region, allow a single Newton
        # polish step when it strictly reduces |g| within 1-ULP objective noise.
        polish_x = cands[0]
        polish_val = vals[0]
        polish_ok = (
            (~ok[0])
            & concave
            & inside[0]
            & jnp.isfinite(polish_val)
            & (polish_val >= fx - 1e-14 * (1.0 + jnp.abs(fx)))
            & (jnp.abs(grad_fn(polish_x)) < 0.5 * jnp.abs(g))
        )
        moved = polish_ok | any_ok
        x_next = jnp.where(polish_ok, polish_x, jnp.where(any_ok, best, x))
        step_size = jnp.abs(x_next - x)
        done = (
            (~jnp.isfinite(fx))
            | (~jnp.isfinite(g))
            | (jnp.abs(g) < _GRAD_TOL)
            | (~moved)
            | (step_size < _STEP_TOL)
        )
        return x_next, it + one, done

    x_fin, _, _ = jax.lax.while_loop(cond, body, (x0, zero, jnp.asarray(False)))
    return x_fin


def _mvn_mean_cov(params: Array, n_factors: int) -> tuple[Array, Array]:
    r"""Mean vector and covariance :math:`\Sigma_{ij}=\rho_{ij}\sqrt{v_iv_j}` from latent params.

    ``params`` follows :func:`~flexpyirt.latent.latent_param_names`:
    ``(mean_1..D, var_1..D, cor_12, cor_13, .., cor_{D-1,D})``.
    """
    means = params[:n_factors]
    sd = jnp.sqrt(params[n_factors : 2 * n_factors])
    corr = jnp.eye(n_factors, dtype=params.dtype)
    k = 2 * n_factors
    for i in range(n_factors):
        for j in range(i + 1, n_factors):
            corr = corr.at[i, j].set(params[k]).at[j, i].set(params[k])
            k += 1
    return means, corr * sd[:, None] * sd[None, :]


def _newton_nd(f: Callable[[Array], Array], x0: Array, bound: float | None = None) -> Array:
    r"""Safeguarded Newton ascent maximizing scalar ``f`` over :math:`\mathbb R^D` (§15.39).

    Gradient and Hessian come from JAX autodiff. The Newton direction
    :math:`-H^{-1}\nabla f` is used when :math:`H` is negative definite (all eigenvalues
    :math:`<-10^{-12}`), otherwise the gradient; the step is capped at norm ``_MAX_STEP`` and the
    step sizes :math:`2^{-k}` (:math:`k=0,\dots,29`) are evaluated in parallel so :math:`f`
    never decreases (the D-dimensional analogue of ``_newton_1d``). With ``bound``, candidates
    outside the open box :math:`|\theta_m| < \text{bound}` are rejected (mirt's ``max_theta``).
    """
    grad_fn = jax.grad(f)
    hess_fn = jax.hessian(f)

    def f_box(x: Array) -> Array:
        if bound is None:
            return f(x)
        inside = jnp.all(jnp.abs(x) < bound)
        return jnp.where(inside, f(jnp.where(inside, x, 0.0)), -jnp.inf)

    zero = jnp.asarray(0, dtype=jnp.int32)
    one = jnp.asarray(1, dtype=jnp.int32)
    steps = 0.5 ** jnp.arange(_N_HALVINGS, dtype=x0.dtype)

    def cond(state: tuple[Array, Array, Array]) -> Array:
        _, it, done = state
        return (~done) & (it < _MAX_ITER)

    def body(state: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        x, it, _ = state
        fx = f(x)
        g = grad_fn(x)
        h = hess_fn(x)
        finite_h = jnp.all(jnp.isfinite(h))
        h_safe = jnp.where(finite_h, h, -jnp.eye(x.shape[0], dtype=x.dtype))
        neg_def = finite_h & (jnp.max(jnp.linalg.eigvalsh(h_safe)) < -1e-12)
        newton_dir = -jnp.linalg.solve(
            jnp.where(neg_def, h_safe, -jnp.eye(x.shape[0], dtype=x.dtype)), g
        )
        direction = jnp.where(neg_def, newton_dir, g)
        direction = jnp.where(jnp.all(jnp.isfinite(direction)), direction, 0.0)
        norm = jnp.sqrt(jnp.sum(direction * direction))
        direction = direction * jnp.minimum(1.0, _MAX_STEP / jnp.where(norm > 0.0, norm, 1.0))

        cands = x[None, :] + steps[:, None] * direction[None, :]
        vals = jax.vmap(f_box)(cands)
        ok = jnp.isfinite(vals) & (vals >= fx)
        any_ok = jnp.any(ok)
        best = cands[jnp.argmax(ok)]
        # Newton polish near the mode, where f is flat to rounding (see _newton_1d).
        polish_ok = (
            (~ok[0])
            & neg_def
            & jnp.isfinite(vals[0])
            & (vals[0] >= fx - 1e-14 * (1.0 + jnp.abs(fx)))
            & (jnp.max(jnp.abs(grad_fn(cands[0]))) < 0.5 * jnp.max(jnp.abs(g)))
        )
        moved = polish_ok | any_ok
        x_next = jnp.where(polish_ok, cands[0], jnp.where(any_ok, best, x))
        step_size = jnp.max(jnp.abs(x_next - x))
        done = (
            (~jnp.isfinite(fx))
            | (~jnp.all(jnp.isfinite(g)))
            | (jnp.max(jnp.abs(g)) < _GRAD_TOL)
            | (~moved)
            | (step_size < _STEP_TOL)
        )
        return x_next, it + one, done

    x_fin, _, _ = jax.lax.while_loop(cond, body, (x0, zero, jnp.asarray(False)))
    return x_fin


def _map_scores_multidim(prob: CompiledProblem, eta: Array) -> tuple[Array, Array]:
    r"""MAP scores for :math:`D\in\{2,3\}` with the multivariate normal latent prior (§15.39).

    .. math::
        f(\boldsymbol\theta) = \log L(\boldsymbol\theta\mid y_p)
            - \tfrac12(\boldsymbol\theta-\boldsymbol\mu)^\top\boldsymbol\Sigma^{-1}
              (\boldsymbol\theta-\boldsymbol\mu),
        \qquad \mathrm{SE}_m = \big[(-\nabla^2 f(\hat{\boldsymbol\theta}))^{-1}\big]_{mm}^{1/2}.

    Starts from EAP. All-missing patterns return the prior mean and
    :math:`\sqrt{\Sigma_{mm}}`.
    """
    n_factors = int(prob.nodes.shape[1])
    theta0, _ = eap(prob, eta)  # (P, D)
    xi_list = _expand_items(prob, eta)
    lat = _expand_latent(prob, eta)

    def solve_one(y_p: Array, g_p: Array, t0: Array) -> tuple[Array, Array]:
        mu, cov = _mvn_mean_cov(lat[g_p], n_factors)
        prec = jnp.linalg.inv(cov)

        def obj(t: Array) -> Array:
            dev = t - mu
            quad: Array = dev @ prec @ dev
            return _pattern_loglik(t, y_p, g_p, prob.families, xi_list) - 0.5 * quad

        theta_hat = _newton_nd(obj, t0)
        h = jax.hessian(obj)(theta_hat)
        finite = jnp.all(jnp.isfinite(h))
        h_safe = jnp.where(finite, h, -jnp.eye(n_factors, dtype=h.dtype))
        neg_def = finite & (jnp.max(jnp.linalg.eigvalsh(h_safe)) < 0.0)
        cov_hat = jnp.linalg.inv(-jnp.where(neg_def, h_safe, -jnp.eye(n_factors, dtype=h.dtype)))
        se = jnp.where(neg_def, jnp.sqrt(jnp.diag(cov_hat)), jnp.nan)
        return theta_hat, se

    return jax.vmap(solve_one)(prob.y, prob.group, theta0)


def map_scores(prob: CompiledProblem, eta: Array) -> tuple[Array, Array]:
    r"""Bayes modal (MAP) ability estimates and SEs per unique pattern (DESIGN.md §8, §15.22).

    Maximizes

    .. math::
        f_{\mathrm{MAP}}(\theta)
        = \log L(\theta\mid y_p) - \frac{(\theta - \mu_{g(p)})^2}{2\sigma_{g(p)}^2}

    using each group's estimated normal latent distribution as a continuous prior (not the
    quadrature grid), starting from EAP. The standard error at the mode is
    :math:`\mathrm{SE} = (-f''_{\mathrm{MAP}}(\hat\theta))^{-1/2}`. Missing responses are dropped
    from :math:`\log L`; an all-missing pattern returns the prior mode :math:`\mu_g` and SD
    :math:`\sigma_g`. Multidimensional problems (:math:`D>1`) use the multivariate normal prior
    (:func:`_map_scores_multidim`).
    """
    if prob.nodes.ndim == 2 and prob.nodes.shape[1] > 1:
        return _map_scores_multidim(prob, eta)
    theta0, _ = eap(prob, eta)
    xi_list = _expand_items(prob, eta)
    lat = _expand_latent(prob, eta)
    neg_inf = jnp.asarray(-jnp.inf, dtype=theta0.dtype)
    pos_inf = jnp.asarray(jnp.inf, dtype=theta0.dtype)

    def solve_one(y_p: Array, g_p: Array, t0: Array) -> tuple[Array, Array]:
        mu = lat[g_p, 0]
        var = lat[g_p, 1]

        def obj(t: Array) -> Array:
            return _pattern_loglik(t, y_p, g_p, prob.families, xi_list) - 0.5 * (t - mu) ** 2 / var

        theta_hat = _newton_1d(obj, t0, neg_inf, pos_inf)
        h = jax.grad(jax.grad(obj))(theta_hat)
        ok_h = jnp.isfinite(h) & (h < 0.0)
        se = jnp.where(ok_h, 1.0 / jnp.sqrt(jnp.where(ok_h, -h, 1.0)), jnp.nan)
        return theta_hat, se

    return jax.vmap(solve_one)(prob.y, prob.group, theta0)


def _pattern_info_matrix(
    theta: Array,
    y_p: Array,
    g_p: Array,
    families: tuple[FamilyBlock, ...],
    xi_list: tuple[Array, ...],
) -> Array:
    r"""Observed-item Fisher test information matrix :math:`\mathbf I(\boldsymbol\theta)` (D, D).

    .. math::
        \mathbf I(\boldsymbol\theta) = \sum_{j:\,y_{pj}\ge 0}\sum_{k}
            P_{jk}(\boldsymbol\theta)\,\nabla\log P_{jk}(\boldsymbol\theta)\,
            \nabla\log P_{jk}(\boldsymbol\theta)^\top,

    the multidimensional analogue of :func:`_pattern_info`, via JAX autodiff.
    """
    n_factors = theta.shape[0]
    total = jnp.zeros((n_factors, n_factors), dtype=theta.dtype)
    for fam, xi in zip(families, xi_list, strict=True):
        y_f = y_p[fam.columns]
        xi_g = xi[g_p]

        def _item_info(p: Array, k: ItemKernel = fam.kernel) -> Array:
            def lp_fn(t: Array) -> Array:
                return k.log_prob(t[None, :], p)[0]

            prob_k = jnp.exp(lp_fn(theta))  # (K,)
            jac = jax.jacfwd(lp_fn)(theta)  # (K, D)
            info: Array = jnp.einsum("k,km,kn->mn", prob_k, jac, jac)
            return info

        infos = jax.vmap(_item_info)(xi_g)  # (J_f, D, D)
        total = total + jnp.sum(jnp.where((y_f >= 0)[:, None, None], infos, 0.0), axis=0)
    return total


def _pattern_flags_multidim(
    y_p: Array, families: tuple[FamilyBlock, ...]
) -> tuple[Array, Array, Array]:
    """``(all_missing, all_low, all_high)`` on raw categories, as mirt does for ``D > 1``."""
    n_obs = jnp.asarray(0, dtype=jnp.int32)
    n_low = jnp.asarray(0, dtype=jnp.int32)
    n_high = jnp.asarray(0, dtype=jnp.int32)
    for fam in families:
        y_f = y_p[fam.columns]
        obs = y_f >= 0
        n_obs = n_obs + jnp.sum(obs, dtype=jnp.int32)
        n_low = n_low + jnp.sum(obs & (y_f == 0), dtype=jnp.int32)
        n_high = n_high + jnp.sum(obs & (y_f == fam.kernel.n_cat - 1), dtype=jnp.int32)
    return n_obs == 0, (n_obs > 0) & (n_low == n_obs), (n_obs > 0) & (n_high == n_obs)


def _inv_neg_hessian_se(h: Array, keep: Array) -> tuple[Array, Array]:
    """SEs from the inverse negative Hessian restricted to ``keep`` dims; ``(se, neg_def)``."""
    n = h.shape[0]
    eye = jnp.eye(n, dtype=h.dtype)
    mask = keep[:, None] & keep[None, :]
    h_sub = jnp.where(mask, h, -eye)
    finite = jnp.all(jnp.isfinite(h_sub))
    h_safe = jnp.where(finite, h_sub, -eye)
    neg_def = finite & (jnp.max(jnp.linalg.eigvalsh(h_safe)) < 0.0)
    cov = jnp.linalg.inv(-jnp.where(neg_def, h_safe, -eye))
    return jnp.sqrt(jnp.diag(cov)), neg_def


def _ml_scores_multidim(prob: CompiledProblem, eta: Array) -> tuple[Array, Array]:
    r"""ML scores for :math:`D\in\{2,3\}` (DESIGN.md §15.41).

    Maximizes :math:`\log L(\boldsymbol\theta\mid y_p)` over :math:`|\theta_m|<20` from EAP.
    Patterns whose observed responses are all in the lowest (highest) category return
    ``-inf`` (``+inf``) on every dimension; a dimension that reaches the bound or has
    :math:`\mathrm{SE}>30` diverges and returns :math:`\pm\infty` with SE ``NaN``, and the
    other dimensions take SEs from the inverse negative Hessian of their own block. All-missing
    patterns return ``NaN``.
    """
    theta0_raw, _ = eap(prob, eta)
    theta0 = jnp.clip(theta0_raw, -_MAX_THETA + 1e-3, _MAX_THETA - 1e-3)
    xi_list = _expand_items(prob, eta)
    n_factors = int(prob.nodes.shape[1])

    def solve_one(y_p: Array, g_p: Array, t0: Array) -> tuple[Array, Array]:
        all_missing, all_low, all_high = _pattern_flags_multidim(y_p, prob.families)

        def obj(t: Array) -> Array:
            return _pattern_loglik(t, y_p, g_p, prob.families, xi_list)

        theta_hat = _newton_nd(obj, t0, bound=_MAX_THETA)
        h = jax.hessian(obj)(theta_hat)
        all_dims = jnp.ones((n_factors,), dtype=bool)
        se_full, nd_full = _inv_neg_hessian_se(h, all_dims)
        near = jnp.abs(theta_hat) >= _MAX_THETA - 1e-2
        diverged = near | (nd_full & (se_full > _ML_MAX_SE))
        se_kept, nd_kept = _inv_neg_hessian_se(h, ~diverged)
        sign = jnp.where(theta_hat < 0.0, -jnp.inf, jnp.inf)
        theta_out = jnp.where(diverged, sign, theta_hat)
        se_out = jnp.where(diverged | ~nd_kept, jnp.nan, se_kept)
        extreme = all_low | all_high
        ext_val = jnp.where(all_low, -jnp.inf, jnp.inf)
        theta_out = jnp.where(all_missing, jnp.nan, jnp.where(extreme, ext_val, theta_out))
        se_out = jnp.where(all_missing | extreme, jnp.nan, se_out)
        return theta_out, se_out

    return jax.vmap(solve_one)(prob.y, prob.group, theta0)


def _wle_scores_multidim(prob: CompiledProblem, eta: Array) -> tuple[Array, Array]:
    r"""WLE scores for :math:`D\in\{2,3\}` (DESIGN.md §15.41; Warm, 1989; Wang, 2015).

    Maximizes the Jeffreys/Firth penalized log-likelihood
    :math:`\log L(\boldsymbol\theta\mid y_p)+\tfrac12\log\det\mathbf I(\boldsymbol\theta)`
    (``mirt:::WLE.mirt``) over :math:`|\theta_m|<20` from EAP; SEs are the square roots of the
    diagonal of the inverse negative Hessian of that objective. All-missing patterns return
    ``NaN``.
    """
    theta0_raw, _ = eap(prob, eta)
    theta0 = jnp.clip(theta0_raw, -_MAX_THETA + 1e-3, _MAX_THETA - 1e-3)
    xi_list = _expand_items(prob, eta)
    n_factors = int(prob.nodes.shape[1])

    def solve_one(y_p: Array, g_p: Array, t0: Array) -> tuple[Array, Array]:
        all_missing, _, _ = _pattern_flags_multidim(y_p, prob.families)

        def obj(t: Array) -> Array:
            info = _pattern_info_matrix(t, y_p, g_p, prob.families, xi_list)
            sign, logdet = jnp.linalg.slogdet(info)
            log_det = jnp.where(sign > 0.0, logdet, jnp.nan)
            return _pattern_loglik(t, y_p, g_p, prob.families, xi_list) + 0.5 * log_det

        theta_hat = _newton_nd(obj, t0, bound=_MAX_THETA)
        h = jax.hessian(obj)(theta_hat)
        se, nd = _inv_neg_hessian_se(h, jnp.ones((n_factors,), dtype=bool))
        ok = (~all_missing) & nd & jnp.all(jnp.isfinite(theta_hat))
        theta_out = jnp.where(all_missing, jnp.nan, theta_hat)
        return theta_out, jnp.where(ok, se, jnp.nan)

    return jax.vmap(solve_one)(prob.y, prob.group, theta0)


def ml_scores(prob: CompiledProblem, eta: Array) -> tuple[Array, Array]:
    r"""Maximum likelihood (ML) ability estimates and SEs per unique pattern (DESIGN.md §8, §15.23).

    Maximizes :math:`\log L(\theta\mid y_p)` over :math:`(-\theta_{\max}, \theta_{\max})` with
    :math:`\theta_{\max}=20`, starting from EAP, and computes the observed-information standard
    error :math:`\mathrm{SE} = (-\partial^2\log L/\partial\theta^2)^{-1/2}` (matching mirt's
    default ``expected.info = FALSE``). Patterns with no finite maximum (all lowest / all highest
    observed categories, or non-concave 3PL/4PL tails where the supremum lies at :math:`\pm\infty`)
    return ``-inf`` / ``+inf`` with SE ``NaN``; all-missing patterns return ``NaN`` with SE ``NaN``.
    Multidimensional problems use :func:`_ml_scores_multidim`.
    """
    if prob.nodes.ndim == 2 and prob.nodes.shape[1] > 1:
        return _ml_scores_multidim(prob, eta)
    theta0_raw, _ = eap(prob, eta)
    theta0 = jnp.clip(theta0_raw, -_MAX_THETA + 1e-3, _MAX_THETA - 1e-3)
    xi_list = _expand_items(prob, eta)
    lo = jnp.asarray(-_MAX_THETA, dtype=theta0.dtype)
    hi = jnp.asarray(_MAX_THETA, dtype=theta0.dtype)

    def solve_one(y_p: Array, g_p: Array, t0: Array) -> tuple[Array, Array]:
        all_missing, all_low, all_high = _pattern_flags(y_p, g_p, prob.families, xi_list)

        def obj(t: Array) -> Array:
            return _pattern_loglik(t, y_p, g_p, prob.families, xi_list)

        theta_hat = _newton_1d(obj, t0, lo, hi)
        h = jax.grad(jax.grad(obj))(theta_hat)
        ok_h = jnp.isfinite(h) & (h < 0.0)
        se_raw = jnp.where(ok_h, 1.0 / jnp.sqrt(jnp.where(ok_h, -h, 1.0)), jnp.inf)

        f_hat = obj(theta_hat)
        f_lo = obj(lo)
        f_hi = obj(hi)
        lo_beats_hat = f_lo > f_hat + 1e-12
        hi_beats_hat = f_hi > f_hat + 1e-12
        endpoint_beats_hat = lo_beats_hat | hi_beats_hat
        diverged = (
            (~ok_h)
            | (se_raw > _ML_MAX_SE)
            | (jnp.abs(theta_hat) >= _MAX_THETA - 1e-2)
            | endpoint_beats_hat
        )
        sign_hat = jnp.where(theta_hat < 0.0, -jnp.inf, jnp.inf)
        div_val = jnp.where(
            endpoint_beats_hat,
            jnp.where(f_lo > f_hi, -jnp.inf, jnp.where(f_hi > f_lo, jnp.inf, sign_hat)),
            sign_hat,
        )
        theta_out = jnp.where(
            all_missing,
            jnp.nan,
            jnp.where(
                all_low,
                -jnp.inf,
                jnp.where(all_high, jnp.inf, jnp.where(diverged, div_val, theta_hat)),
            ),
        )
        se_out = jnp.where(all_missing | all_low | all_high | diverged, jnp.nan, se_raw)
        return theta_out, se_out

    return jax.vmap(solve_one)(prob.y, prob.group, theta0)


def wle_scores(prob: CompiledProblem, eta: Array) -> tuple[Array, Array]:
    r"""Weighted likelihood (WLE) ability estimates and SEs per unique pattern (Warm, 1989).

    Solves

    .. math::
        \frac{\partial\log L(\theta\mid y_p)}{\partial\theta}
        + \frac{J(\theta)}{2 I(\theta)} = 0, \qquad
        J(\theta) = \frac{\partial I(\theta)}{\partial\theta},

    by maximizing the equivalent Jeffreys/Firth (1993) penalized log-likelihood
    :math:`f_{\mathrm{WLE}}(\theta) = \log L(\theta\mid y_p) + \tfrac{1}{2}\log I(\theta)` over
    :math:`(-\theta_{\max}, \theta_{\max})`, where :math:`I(\theta)` is the observed-item Fisher
    test information computed strictly via JAX autodiff (DESIGN.md §8, §15.24). Matches
    ``mirt:::WLE.mirt`` for every item family. Extreme patterns (all lowest / all highest
    categories) have finite estimates; all-missing patterns return ``NaN`` and SE ``NaN``. The
    standard error is :math:`(-f''_{\mathrm{WLE}}(\hat\theta))^{-1/2}`, matching mirt's default
    ``fscores(..., method="WLE", full.scores.SE=TRUE)``. Multidimensional problems use
    :func:`_wle_scores_multidim`.
    """
    if prob.nodes.ndim == 2 and prob.nodes.shape[1] > 1:
        return _wle_scores_multidim(prob, eta)
    theta0_raw, _ = eap(prob, eta)
    theta0 = jnp.clip(theta0_raw, -_MAX_THETA + 1e-3, _MAX_THETA - 1e-3)
    xi_list = _expand_items(prob, eta)
    lo = jnp.asarray(-_MAX_THETA, dtype=theta0.dtype)
    hi = jnp.asarray(_MAX_THETA, dtype=theta0.dtype)

    def solve_one(y_p: Array, g_p: Array, t0: Array) -> tuple[Array, Array]:
        all_missing, _, _ = _pattern_flags(y_p, g_p, prob.families, xi_list)

        def obj(t: Array) -> Array:
            info = _pattern_info(t, y_p, g_p, prob.families, xi_list)
            pos = info > 0.0
            log_info = jnp.where(pos, jnp.log(jnp.where(pos, info, 1.0)), jnp.nan)
            return _pattern_loglik(t, y_p, g_p, prob.families, xi_list) + 0.5 * log_info

        theta_hat = _newton_1d(obj, t0, lo, hi)
        h = jax.grad(jax.grad(obj))(theta_hat)
        ok = (~all_missing) & jnp.isfinite(theta_hat) & jnp.isfinite(h) & (h < 0.0)
        theta_out = jnp.where(all_missing, jnp.nan, theta_hat)
        se_out = jnp.where(ok, 1.0 / jnp.sqrt(jnp.where(ok, -h, 1.0)), jnp.nan)
        return theta_out, se_out

    return jax.vmap(solve_one)(prob.y, prob.group, theta0)


def plausible_values(
    prob: CompiledProblem,
    eta: Array,
    inverse: Array,
    key: Array,
    *,
    n_draws: int,
) -> Array:
    r"""Draw ``n_draws`` plausible values per respondent from the posterior (Mislevy, 1991).

    Evaluates each unique pattern's posterior density on a fine equispaced grid of ``501`` points
    spanning ``[prob.nodes[0], prob.nodes[-1]]``, integrates via the trapezoidal rule to a
    piecewise-linear CDF on :math:`[0, 1]`, and draws continuous samples per respondent by
    inverse-CDF linear interpolation against ``jax.random.uniform(key, (N_pad, n_draws))``
    (DESIGN.md §8, §15.25). Respondents sharing a response pattern reuse the same CDF but draw
    independently. Deterministic given ``key``. Multidimensional problems use
    :func:`_plausible_values_multidim` and return shape ``(N, n_draws, D)``.
    """
    if prob.nodes.ndim == 2 and prob.nodes.shape[1] > 1:
        return _plausible_values_multidim(prob, eta, inverse, key, n_draws=n_draws)
    lo = prob.nodes[0, 0] if prob.nodes.ndim == 2 else prob.nodes[0]
    hi = prob.nodes[-1, 0] if prob.nodes.ndim == 2 else prob.nodes[-1]
    grid = jnp.linspace(lo, hi, _PV_GRID_POINTS, dtype=prob.nodes.dtype)
    fine_nodes = grid[:, None] if prob.nodes.ndim == 2 else grid
    if prob.latent.is_eh:
        nodes_1d = prob.nodes[:, 0] if prob.nodes.ndim == 2 else prob.nodes

        def _interp_w(w_g: Array) -> Array:
            w_fine = jnp.interp(grid, nodes_1d, w_g)
            return w_fine / jnp.sum(w_fine)

        eh_fine = jax.vmap(_interp_w)(prob.latent.eh_weights)
        prob_fine = replace(prob, nodes=fine_nodes, latent=replace(prob.latent, eh_weights=eh_fine))
    else:
        prob_fine = replace(prob, nodes=fine_nodes)
    post = jax.nn.softmax(_log_joint(prob_fine, eta), axis=1)  # (P_pad, Q_pv)

    seg = 0.5 * (post[:, :-1] + post[:, 1:])  # trapezoidal interval mass, (P_pad, Q_pv - 1)
    seg_sum = jnp.sum(seg, axis=1, keepdims=True)
    seg_norm = seg / jnp.where(seg_sum > 0.0, seg_sum, 1.0)
    cdf = jnp.concatenate(
        [jnp.zeros((post.shape[0], 1), dtype=post.dtype), jnp.cumsum(seg_norm, axis=1)],
        axis=1,
    )
    cdf = cdf.at[:, -1].set(1.0)

    u = jax.random.uniform(key, shape=(inverse.shape[0], n_draws), dtype=grid.dtype)
    u = jnp.clip(u, 0.0, 1.0 - 1e-15)

    def draw_one(pat_idx: Array, u_row: Array) -> Array:
        c = cdf[pat_idx]
        right = jnp.clip(jnp.searchsorted(c, u_row, side="right"), 1, _PV_GRID_POINTS - 1)
        left = right - 1
        c_lo = c[left]
        c_hi = c[right]
        dc = c_hi - c_lo
        frac = jnp.where(dc > 0.0, (u_row - c_lo) / jnp.where(dc > 0.0, dc, 1.0), 0.0)
        return grid[left] + frac * (grid[right] - grid[left])

    draws: Array = jax.vmap(draw_one)(inverse, u)
    return draws


def _plausible_values_multidim(
    prob: CompiledProblem,
    eta: Array,
    inverse: Array,
    key: Array,
    *,
    n_draws: int,
) -> Array:
    r"""Plausible values for :math:`D\in\{2,3\}` from a per-pattern posterior lattice (§15.42).

    For each unique pattern, an equispaced lattice of ``_PV_LATTICE_POINTS[D]`` points per
    dimension spans :math:`\mathrm{EAP}_m \pm 6\,\mathrm{SD}_m` (clipped to the fitted grid's
    range). The posterior
    :math:`\log L(\boldsymbol\theta\mid y_p)-\tfrac12(\boldsymbol\theta-\boldsymbol\mu)^\top
    \boldsymbol\Sigma^{-1}(\boldsymbol\theta-\boldsymbol\mu)` is evaluated on it, each draw
    picks a lattice cell from that discrete posterior and adds uniform jitter within the cell
    (width :math:`\Delta_m`), giving continuous draws whose variance exceeds the posterior's by
    :math:`\Delta_m^2/12` (about 0.2% in 2D and 0.6% in 3D of the posterior variance). Draws are
    independent across respondents and deterministic given ``key``.

    Returns
    -------
    Array, shape (N_pad, n_draws, D)
    """
    n_factors = int(prob.nodes.shape[1])
    n_pts = _PV_LATTICE_POINTS[n_factors]
    lo = prob.nodes[0, 0]
    hi = prob.nodes[-1, 0]
    m_eap, sd_eap = eap(prob, eta)  # (P, D)
    a = jnp.clip(m_eap - _PV_SPAN_SD * sd_eap, lo, hi)
    b = jnp.clip(m_eap + _PV_SPAN_SD * sd_eap, lo, hi)
    b = jnp.maximum(b, a + 1e-6)
    xi_list = _expand_items(prob, eta)
    lat = _expand_latent(prob, eta)
    unit = jnp.linspace(0.0, 1.0, n_pts, dtype=prob.nodes.dtype)
    grids = jnp.meshgrid(*([jnp.arange(n_pts)] * n_factors), indexing="ij")
    cell_index = jnp.stack([g.ravel() for g in grids], axis=1)  # (n^D, D)

    def pattern_logits(args: tuple[Array, Array, Array, Array]) -> Array:
        y_p, g_p, a_p, b_p = args
        mu, cov = _mvn_mean_cov(lat[g_p], n_factors)
        prec = jnp.linalg.inv(cov)
        nodes = a_p[None, :] + (b_p - a_p)[None, :] * unit[cell_index]  # (n^D, D)

        def log_post(t: Array) -> Array:
            dev = t - mu
            quad: Array = dev @ prec @ dev
            return _pattern_loglik(t, y_p, g_p, prob.families, xi_list) - 0.5 * quad

        out: Array = jax.vmap(log_post)(nodes)
        return out

    logits = jax.lax.map(pattern_logits, (prob.y, prob.group, a, b), batch_size=8)
    keys = jax.random.split(key, inverse.shape[0])

    def draw_one(args: tuple[Array, Array]) -> Array:
        pat, k = args
        k_cell, k_jit = jax.random.split(k)
        cells = jax.random.categorical(k_cell, logits[pat], shape=(n_draws,))
        idx = cell_index[cells]  # (n_draws, D)
        width = (b[pat] - a[pat]) / (n_pts - 1)
        centers = a[pat][None, :] + width[None, :] * idx
        jitter = (jax.random.uniform(k_jit, (n_draws, n_factors), dtype=width.dtype) - 0.5) * width
        out: Array = centers + jitter
        return out

    draws: Array = jax.lax.map(draw_one, (inverse, keys), batch_size=256)
    return draws
