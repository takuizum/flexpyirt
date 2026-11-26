r"""Pure numerical core of common-item screening for linking (DESIGN.md §21).

Functions here take item kernels, parameter arrays and quadrature grids only; names, tables and
validation live in :mod:`flexpyirt.screening`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from flexpyirt.items import ItemKernel, NominalKernel

__all__ = [
    "ROBUST_Z_SCALE",
    "expected_score_rmsd",
    "item_location",
    "robust_z",
]

ROBUST_Z_SCALE = 0.74
r"""Factor :math:`0.74 \approx 1/1.349` turning an IQR into a normal standard deviation."""

_IQR_FLOOR = 1e-12
"""IQRs at or below this value give NaN robust z (e.g. the slope ratios of Rasch items)."""


def item_location(kernel: ItemKernel, params: Array) -> Array:
    r"""One location per item, transforming as :math:`b^* = A\,b + B` (DESIGN.md §21).

    - 2PL / Rasch / 3PL / 4PL: the display difficulty :math:`b = -d/a`.
    - GRM, GPCM / PCM: the mean of the display thresholds,
      :math:`\bar b = \frac{1}{K-1}\sum_{k=1}^{K-1} b_k` (step difficulties for GPCM).
    - Nominal: :math:`-d_{K-1}/((K-1)\,a)`, the crossing point of the first and last category
      curves (as in the Mean/Mean start of :func:`flexpyirt.link`).

    Parameters
    ----------
    kernel : ItemKernel
        Unidimensional item kernel.
    params : Array, shape (P,)
        Item parameters in ``kernel.param_names`` order.

    Returns
    -------
    Array, scalar
        The item location.
    """
    if isinstance(kernel, NominalKernel):
        return -params[-1] / ((kernel.n_categories - 1) * params[0])
    return jnp.mean(kernel.irt_thresholds(params))


def expected_score_rmsd(
    kernel: ItemKernel,
    src_params: Array,
    tgt_params: Array,
    nodes: Array,
    weights: Array,
) -> Array:
    r"""Root mean squared difference of two expected item score curves (DESIGN.md §21).

    .. math::
        \mathrm{RMSD} = \Big\{\sum_q w_q\,\big[E^{*}(\theta_q) - E^{\mathrm{tgt}}(\theta_q)\big]^2
        \Big\}^{1/2},
        \qquad E(\theta) = \sum_{k=0}^{K-1} k\,P_k(\theta),

    with the source item already transformed to the target scale (``src_params`` is
    :math:`\xi^*`), so that :math:`E^*(\theta) = E^{\mathrm{src}}((\theta - B)/A)`.

    Parameters
    ----------
    kernel : ItemKernel
        Unidimensional item kernel shared by the two items.
    src_params, tgt_params : Array, shape (P,)
        Transformed source and target item parameters.
    nodes : Array, shape (Q,)
        Target-scale quadrature nodes.
    weights : Array, shape (Q,)
        Non-negative weights summing to 1.

    Returns
    -------
    Array, scalar
        The RMSD on the item's score scale (0 to ``K - 1``).
    """
    theta = nodes[:, None]
    scores = jnp.arange(kernel.n_categories, dtype=nodes.dtype)
    e_src = jnp.exp(kernel.log_prob(theta, src_params)) @ scores
    e_tgt = jnp.exp(kernel.log_prob(theta, tgt_params)) @ scores
    return jnp.sqrt(jnp.sum(weights * (e_src - e_tgt) ** 2))


expected_score_rmsd_jit = jax.jit(expected_score_rmsd, static_argnums=0)
item_location_jit = jax.jit(item_location, static_argnums=0)


def robust_z(values: np.ndarray) -> np.ndarray:
    r"""Robust z of Huynh & Meyer (2010) over all common items.

    .. math::
        z_j = \frac{x_j - \operatorname{median}_i x_i}{0.74\,\mathrm{IQR}_i(x_i)},

    with quantiles by linear interpolation (R ``quantile(type = 7)``). The median and IQR are
    taken over every common item, kept or removed (DESIGN.md §21).

    Parameters
    ----------
    values : ndarray, shape (J,)
        Per-item differences :math:`x_j` (NaN entries are skipped in the median and IQR).

    Returns
    -------
    ndarray, shape (J,)
        Robust z; all NaN when there are no finite values or the IQR is at or below ``1e-12``.
    """
    ref = values[np.isfinite(values)]
    if ref.size == 0:
        return np.full(values.shape, np.nan)
    q1, med, q3 = np.quantile(ref, [0.25, 0.5, 0.75])
    iqr = float(q3 - q1)
    if not iqr > _IQR_FLOOR:
        return np.full(values.shape, np.nan)
    return np.asarray((values - med) / (ROBUST_Z_SCALE * iqr), dtype=np.float64)
