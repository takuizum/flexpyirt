"""Masked, bounded, monotone Newton ascent used by every M-step (DESIGN.md §13.1).

Derivatives come from JAX autodiff; only the update and line search are written here. Everything
is shape-static so it can be ``vmap``-ed over items and run inside ``lax.while_loop``.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array

__all__ = ["newton_ascent"]

_N_HALVINGS = 30
_ACTIVE_EPS = 1e-8
"""A free coordinate this close to a bound, with the gradient pointing out of the box, is held."""


def _newton_step(
    f: Callable[[Array], Array], x: Array, mask: Array, lower: Array, upper: Array
) -> Array:
    """One damped Newton step on the free coordinates ``mask`` that never decreases ``f``."""
    grad = jax.grad(f)(x)
    # epsilon-active set (Bertsekas, 1982): coordinates pressed against a bound are held for this
    # step so that they do not block the others (DESIGN.md §15.2).
    active = ((x - lower <= _ACTIVE_EPS) & (grad < 0)) | ((upper - x <= _ACTIVE_EPS) & (grad > 0))
    m = (mask & ~active).astype(x.dtype)
    g = grad * m
    h = jax.hessian(f)(x)
    # Fixed coordinates: replace their rows/columns by -I so they do not move.
    h = h * jnp.outer(m, m) - jnp.diag(1.0 - m)
    direction = -jnp.linalg.solve(h, g)
    # If H is not negative definite the Newton direction may not be an ascent direction;
    # fall back to the gradient.
    is_ascent = (jnp.dot(g, direction) > 0) & jnp.all(jnp.isfinite(direction))
    direction = jnp.where(is_ascent, direction, g) * m

    steps = 0.5 ** jnp.arange(_N_HALVINGS, dtype=x.dtype)
    raw = x[None, :] + steps[:, None] * direction[None, :]
    # A coordinate that would leave the box moves halfway to its bound instead (the others keep
    # their step); the objective check below still guarantees ascent.
    lo = jnp.where(jnp.isfinite(lower), lower, x)
    hi = jnp.where(jnp.isfinite(upper), upper, x)
    candidates = jnp.where(
        raw <= lower, x + 0.5 * (lo - x), jnp.where(raw >= upper, x + 0.5 * (hi - x), raw)
    )
    values = jax.vmap(f)(candidates)
    inside = (candidates > lower) & (candidates < upper)
    feasible = jnp.all(inside | ~mask[None, :], axis=1)
    ok = feasible & jnp.isfinite(values) & (values >= f(x))
    first = jnp.argmax(ok)  # largest acceptable step
    return jnp.where(jnp.any(ok), candidates[first], x)


def _to_unconstrained(x: Array, kind: Array, lower: Array, upper: Array) -> Array:
    r"""Map natural ``x`` to the unconstrained :math:`z` of DESIGN.md §15.47 (``kind`` per slot).

    ``kind``: 0 identity, 1 :math:`\operatorname{logit}((x-l)/(u-l))`, 2 :math:`\log(x-l)`,
    3 :math:`\log(u-x)`. Double-``where`` guards keep unselected branches finite.
    """
    lo = jnp.where(jnp.isfinite(lower), lower, 0.0)
    hi = jnp.where(jnp.isfinite(upper), upper, 1.0)
    width = jnp.where(kind == 1, hi - lo, 1.0)
    p = jnp.where(kind == 1, (x - lo) / width, 0.5)
    logit = jnp.log(p) - jnp.log1p(-p)
    above = jnp.where(kind == 2, x - lo, 1.0)
    below = jnp.where(kind == 3, hi - x, 1.0)
    return jnp.where(
        kind == 1,
        logit,
        jnp.where(kind == 2, jnp.log(above), jnp.where(kind == 3, jnp.log(below), x)),
    )


def _to_natural(z: Array, kind: Array, lower: Array, upper: Array) -> Array:
    r"""Inverse of :func:`_to_unconstrained`.

    :math:`l+(u-l)\sigma(z)`, :math:`l+e^z` or :math:`u-e^z` by ``kind``; identity for kind 0.
    """
    lo = jnp.where(jnp.isfinite(lower), lower, 0.0)
    hi = jnp.where(jnp.isfinite(upper), upper, 1.0)
    safe = jnp.where(kind == 0, 0.0, z)
    return jnp.where(
        kind == 1,
        lo + (hi - lo) * jax.nn.sigmoid(safe),
        jnp.where(kind == 2, lo + jnp.exp(safe), jnp.where(kind == 3, hi - jnp.exp(safe), z)),
    )


def _transform_kind(mask: Array, lower: Array, upper: Array) -> Array:
    """Per-slot transform code (DESIGN.md §15.47); fixed slots and unbounded slots get 0."""
    lo_fin = jnp.isfinite(lower)
    hi_fin = jnp.isfinite(upper)
    kind = jnp.where(lo_fin & hi_fin, 1, jnp.where(lo_fin, 2, jnp.where(hi_fin, 3, 0))).astype(
        jnp.int32
    )
    return jnp.where(mask, kind, 0)


def _unconstrained_step(
    f: Callable[[Array], Array], x: Array, mask: Array, lower: Array, upper: Array
) -> Array:
    """One damped Newton step on the logit/log scale of the bounded coordinates (§15.47).

    The iterate stays on the natural scale; the derivatives are those of ``f`` composed with the
    inverse transform, and candidates are compared with ``f(x)`` itself, so a rejected step
    returns ``x`` exactly and ``f`` never decreases.
    """
    kind = _transform_kind(mask, lower, upper)
    z = _to_unconstrained(x, kind, lower, upper)

    def f_z(v: Array) -> Array:
        return f(_to_natural(v, kind, lower, upper))

    m = mask.astype(x.dtype)
    g = jax.grad(f_z)(z) * m
    h = jax.hessian(f_z)(z)
    h = h * jnp.outer(m, m) - jnp.diag(1.0 - m)
    direction = -jnp.linalg.solve(h, g)
    is_ascent = (jnp.dot(g, direction) > 0) & jnp.all(jnp.isfinite(direction))
    direction = jnp.where(is_ascent, direction, g) * m

    steps = 0.5 ** jnp.arange(_N_HALVINGS, dtype=x.dtype)
    z_cand = z[None, :] + steps[:, None] * direction[None, :]
    candidates = jax.vmap(lambda v: _to_natural(v, kind, lower, upper))(z_cand)
    # Fixed slots keep their exact value (no round trip through the transform).
    candidates = jnp.where(mask[None, :], candidates, x[None, :])
    values = jax.vmap(f)(candidates)
    inside = (candidates > lower) & (candidates < upper)
    feasible = jnp.all(inside | ~mask[None, :], axis=1)
    ok = feasible & jnp.isfinite(values) & (values >= f(x))
    first = jnp.argmax(ok)
    return jnp.where(jnp.any(ok), candidates[first], x)


def newton_ascent(
    f: Callable[[Array], Array],
    x0: Array,
    mask: Array,
    lower: Array,
    upper: Array,
    n_steps: int,
    *,
    unconstrained: bool = False,
) -> Array:
    r"""Maximize ``f`` over the coordinates where ``mask`` is true, starting from ``x0``.

    Each step solves :math:`H\delta=-g` on the free coordinates (gradient and Hessian by
    ``jax.grad`` / ``jax.hessian``), falls back to :math:`\delta=g` if :math:`\delta` is not an
    ascent direction, and takes the largest step :math:`2^{-k}\delta` (:math:`k<30`) that stays
    strictly inside ``(lower, upper)`` on the free coordinates and does not decrease ``f``. Hence
    :math:`f(x_{\text{out}})\ge f(x_0)` always holds, which keeps EM monotone.

    Bounds (DESIGN.md §15.2): a coordinate whose step would leave the box moves halfway to the
    bound instead, and a coordinate within ``1e-8`` of a bound whose gradient points outwards is
    held for that step (an :math:`\varepsilon`-active set; Bertsekas, 1982), so a parameter that
    converges to its bound does not stop the item's other parameters from moving.

    With ``unconstrained=True`` (``BAEM(m_step_scale="unconstrained")``, DESIGN.md §15.47), each
    free coordinate with a finite bound is stepped on
    :math:`\operatorname{logit}((x-l)/(u-l))`, :math:`\log(x-l)` or :math:`\log(u-x)` instead,
    with derivatives of :math:`f\circ T^{-1}` by autodiff (no Jacobian term: the objective is
    unchanged) and no active set. Interior maximizers are the same; boundary maxima are only
    approached.

    Parameters
    ----------
    f : callable
        Scalar objective of a 1-D array.
    x0, mask, lower, upper : Array, shape (n,)
    n_steps : int
        Number of Newton steps (static).
    unconstrained : bool
        Step on the unconstrained scale of the bounded coordinates (static; default ``False``).

    Returns
    -------
    Array, shape (n,)
    """
    step = _unconstrained_step if unconstrained else _newton_step

    def body(_: int, x: Array) -> Array:
        return step(f, x, mask, lower, upper)

    out: Array = jax.lax.fori_loop(0, n_steps, body, x0)
    return out
