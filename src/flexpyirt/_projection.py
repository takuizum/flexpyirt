r"""Kernels of projective IRT (DESIGN.md §32).

Pure functions on arrays; the checks, names and tables live in :mod:`flexpyirt.projection`.

* :func:`marginal_item_counts` sums a multidimensional E-table over every factor except the
  target, giving the marginal expected table of the MML-PIRT approach (Chalmers, Falk & Reise,
  2026).
* :func:`kernel_scale` gives the closed-form logistic approximations
  (Ip, 2010; Stucky, Thissen & Edelen, 2013) written for any latent correlation matrix.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import Array

from flexpyirt import _core

LOGISTIC_KERNEL_CONSTANT = 16.0 * math.sqrt(3.0) / (15.0 * math.pi)
r"""Ip's (2010) constant :math:`k=16\sqrt3/(15\pi)\approx0.5881` of the logistic kernel."""
NORMAL_OGIVE_CONSTANT = 1.0 / 1.702
r""":math:`1/D` with :math:`D=1.702`, the normal-ogive scaling of Stucky et al. (2013)."""


def node_map(coordinates: Array, nodes_1d: Array) -> Array:
    """One-hot ``(..., Q, n)`` map of each node's target coordinate to the one-dimensional grid.

    ``coordinates`` holds the target factor's value at every node of a (product or block) grid;
    every value is one of ``nodes_1d`` up to rounding, so the nearest node is exact.
    """
    idx = jnp.argmin(jnp.abs(coordinates[..., None] - nodes_1d), axis=-1)
    return jax.nn.one_hot(idx, nodes_1d.shape[0], dtype=nodes_1d.dtype)


def marginal_item_counts(
    prob: _core.CompiledProblem, eta: Array, factor: int, nodes_1d: Array
) -> tuple[tuple[Array, ...], Array]:
    r"""Marginal expected table on the target factor's grid (Chalmers, Falk & Reise, 2026).

    With the E-step of the parent model at :math:`\hat\eta`,

    .. math::
        \bar r_{jk}(\theta_q)=\sum_{\boldsymbol\theta_m:\ \theta_{mf}=\theta_q}
        \bar r_{jk}(\boldsymbol\theta_m;\hat\eta),

    the sum over every node whose target coordinate is :math:`\theta_q` (each item on its own
    block grid in a bifactor or two-tier model, DESIGN.md §17.1). :math:`\bar n` is marginalized
    the same way from the primary (or product) grid.

    Returns ``(r, n)``: per family ``(G, J_f, K_f, n)`` and ``(G, n)``.
    """
    counts, _ = _core.e_step(prob, eta)
    r_out = []
    for f, r in enumerate(counts.r):
        nodes, axis = _core._family_nodes(prob, f)
        onehot = node_map(nodes[..., factor], nodes_1d)
        if axis is None:
            r_out.append(jnp.einsum("gjkq,qn->gjkn", r, onehot))
        else:
            r_out.append(jnp.einsum("gjkq,jqn->gjkn", r, onehot))
    n_nodes = prob.nodes.shape[0]
    n_map = node_map(prob.nodes[:, factor], nodes_1d)
    n_marg = jnp.einsum("gq,qn->gn", counts.n[:, :n_nodes], n_map)
    return tuple(r_out), n_marg


def kernel_scale(slopes: Array, phi: Array, factor: int, constant: float) -> tuple[Array, Array]:
    r"""Projected slope and intercept scale of the logistic approximations.

    For :math:`\boldsymbol\theta\sim N(\mathbf 0,\boldsymbol\Phi)` with :math:`\Phi_{ff}=1`,
    :math:`\mathbf a^\top\boldsymbol\theta\mid\theta_f` is normal with mean
    :math:`(\mathbf a^\top\boldsymbol\Phi_{\cdot f})\theta_f` and variance
    :math:`v=\mathbf a^\top\boldsymbol\Phi\mathbf a-(\mathbf a^\top\boldsymbol\Phi_{\cdot f})^2`.
    Approximating the logistic by a normal ogive (or the logistic kernel) and integrating gives

    .. math::
        a^*=s\,\mathbf a^\top\boldsymbol\Phi_{\cdot f},\qquad d^*_k=s\,d_k,\qquad
        s=\big(1+c^2v\big)^{-1/2},

    with :math:`c=16\sqrt3/(15\pi)` (Ip, 2010, the logistic kernel approximation) or
    :math:`c=1/1.702` (Stucky, Thissen & Edelen, 2013, through the factor-analytic loadings).
    With one nuisance factor and :math:`\rho=0` this is Ip's (2010) formula
    :math:`a^*=a(1+k^2a_s^2)^{-1/2}`.

    Returns ``(a_star, s)``.
    """
    loading = slopes @ phi[:, factor]
    variance = slopes @ phi @ slopes - loading**2
    scale = 1.0 / jnp.sqrt(1.0 + constant**2 * jnp.maximum(variance, 0.0))
    return scale * loading, scale
