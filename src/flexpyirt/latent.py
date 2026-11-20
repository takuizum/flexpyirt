"""Latent (population) distributions (Spec layer) and their quadrature weights (Core layer)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
from jax import Array

from flexpyirt.errors import SpecificationError

__all__ = ["Normal", "normal_log_weights"]


@dataclass(frozen=True)
class Normal:
    r"""Normal latent distribution :math:`\theta\sim N(\mu,\sigma^2)` on a quadrature grid.

    Parameters
    ----------
    mean, var : float
        Value of the mean and variance; the starting value when the parameter is free.
    free_mean, free_var : bool
        Whether the mean / variance is estimated.

    Examples
    --------
    >>> Normal.standard()          # N(0, 1), nothing estimated
    >>> Normal(free_var=True)      # N(0, sigma^2), variance estimated (Rasch default)
    >>> Normal.free()              # mean and variance estimated
    """

    mean: float = 0.0
    var: float = 1.0
    free_mean: bool = False
    free_var: bool = False

    def __post_init__(self) -> None:
        """Validate values."""
        if not math.isfinite(self.mean):
            raise SpecificationError(f"Normal mean must be finite, got {self.mean}.")
        if not (math.isfinite(self.var) and self.var > 0):
            raise SpecificationError(
                f"Normal variance must be positive and finite, got {self.var}."
            )

    @classmethod
    def standard(cls) -> Normal:
        """Return the fixed standard normal :math:`N(0,1)`."""
        return cls()

    @classmethod
    def free(cls, *, mean: float = 0.0, var: float = 1.0) -> Normal:
        """Return a normal with both mean and variance estimated (starting at ``mean``, ``var``)."""
        return cls(mean=mean, var=var, free_mean=True, free_var=True)


def normal_log_weights(nodes: Array, mean: Array, var: Array) -> Array:
    r"""Log quadrature weights of a normal density normalized on the grid.

    .. math::
        \log w_q = \log\phi(\theta_q;\mu,\sigma^2) - \log\sum_{q'}\phi(\theta_{q'};\mu,\sigma^2)

    The normalizing constant of :math:`\phi` cancels, so only the kernel
    :math:`-(\theta_q-\mu)^2/(2\sigma^2)` is evaluated (DESIGN.md §4.4).

    Parameters
    ----------
    nodes : Array, shape (Q,)
    mean, var : Array, scalar

    Returns
    -------
    Array, shape (Q,)
    """
    kernel = -0.5 * (nodes - mean) ** 2 / var
    return jax.nn.log_softmax(kernel)
