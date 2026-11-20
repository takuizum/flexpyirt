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


def _newton_step(
    f: Callable[[Array], Array], x: Array, mask: Array, lower: Array, upper: Array
) -> Array:
    """One damped Newton step on the free coordinates ``mask`` that never decreases ``f``."""
    m = mask.astype(x.dtype)
    g = jax.grad(f)(x) * m
    h = jax.hessian(f)(x)
    # Fixed coordinates: replace their rows/columns by -I so they do not move.
    h = h * jnp.outer(m, m) - jnp.diag(1.0 - m)
    direction = -jnp.linalg.solve(h, g)
    # If H is not negative definite the Newton direction may not be an ascent direction;
    # fall back to the gradient.
    is_ascent = (jnp.dot(g, direction) > 0) & jnp.all(jnp.isfinite(direction))
    direction = jnp.where(is_ascent, direction, g) * m

    steps = 0.5 ** jnp.arange(_N_HALVINGS, dtype=x.dtype)
    candidates = x[None, :] + steps[:, None] * direction[None, :]
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
