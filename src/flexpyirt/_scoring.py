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


def _expand_items(prob: CompiledProblem, eta: Array) -> tuple[Array, ...]:
    """Per family, ``(G, J_f, P_f)`` item parameters expanded from ``eta``."""
    return tuple(_expand(eta, fam.fixed_values, fam.free_index) for fam in prob.families)


def _expand_latent(prob: CompiledProblem, eta: Array) -> Array:
    """``(G, 2)`` latent ``(mean, var)`` parameters expanded from ``eta``."""
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
        obs = y_f >= 0
        # Every ItemKernel places the slope parameter "a" at index 0.
        a = xi[g_p, :, 0]
        top = fam.kernel.n_cat - 1
        low_cat = jnp.where(a >= 0.0, 0, top)
        high_cat = jnp.where(a >= 0.0, top, 0)
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
    :math:`\sigma_g`.
    """
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


def ml_scores(prob: CompiledProblem, eta: Array) -> tuple[Array, Array]:
    r"""Maximum likelihood (ML) ability estimates and SEs per unique pattern (DESIGN.md §8, §15.23).

    Maximizes :math:`\log L(\theta\mid y_p)` over :math:`(-\theta_{\max}, \theta_{\max})` with
    :math:`\theta_{\max}=20`, starting from EAP, and computes the observed-information standard
    error :math:`\mathrm{SE} = (-\partial^2\log L/\partial\theta^2)^{-1/2}` (matching mirt's
    default ``expected.info = FALSE``). Patterns with no finite maximum (all lowest / all highest
    observed categories, or non-concave 3PL/4PL tails where the supremum lies at :math:`\pm\infty`)
    return ``-inf`` / ``+inf`` with SE ``NaN``; all-missing patterns return ``NaN`` with SE ``NaN``.
    """
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
    ``fscores(..., method="WLE", full.scores.SE=TRUE)``.
    """
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
    independently. Deterministic given ``key``.
    """
    grid = jnp.linspace(prob.nodes[0], prob.nodes[-1], _PV_GRID_POINTS, dtype=prob.nodes.dtype)
    prob_fine = replace(prob, nodes=grid)
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
