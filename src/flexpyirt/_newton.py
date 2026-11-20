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


def newton_ascent(
    f: Callable[[Array], Array],
    x0: Array,
    mask: Array,
    lower: Array,
    upper: Array,
    n_steps: int,
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

    Parameters
    ----------
    f : callable
        Scalar objective of a 1-D array.
    x0, mask, lower, upper : Array, shape (n,)
    n_steps : int
        Number of Newton steps (static).

    Returns
    -------
    Array, shape (n,)
    """

    def body(_: int, x: Array) -> Array:
        return _newton_step(f, x, mask, lower, upper)

    out: Array = jax.lax.fori_loop(0, n_steps, body, x0)
    return out
