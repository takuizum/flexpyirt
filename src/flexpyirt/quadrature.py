"""Quadrature grids for the latent variable."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from flexpyirt.errors import SpecificationError

__all__ = ["Quadrature"]


@dataclass(frozen=True)
class Quadrature:
    """Quadrature grid for a unidimensional latent variable.

    Use :meth:`equispaced` to construct. The weights are not stored here: they come from the latent
    distribution evaluated on the nodes (DESIGN.md §4.9).
    """

    n_points: int = 61
    lower: float = -6.0
    upper: float = 6.0

    def __post_init__(self) -> None:
        """Validate the grid."""
        if self.n_points < 3:
            raise SpecificationError(f"Quadrature needs at least 3 points, got {self.n_points}.")
        if not (
            math.isfinite(self.lower) and math.isfinite(self.upper) and self.lower < self.upper
        ):
            raise SpecificationError(
                "Quadrature limits must be finite with lower < upper, "
                f"got ({self.lower}, {self.upper})."
            )

    @classmethod
    def equispaced(
        cls, n_points: int = 61, *, lim: tuple[float, float] = (-6.0, 6.0)
    ) -> Quadrature:
        """Return ``n_points`` equally spaced nodes on ``lim`` (endpoints included).

        The default (61 points on [-6, 6]) matches mirt's unidimensional default.
        """
        return cls(n_points=n_points, lower=float(lim[0]), upper=float(lim[1]))

    def nodes(self) -> np.ndarray:
        """Return the nodes, shape ``(n_points,)``."""
        return np.linspace(self.lower, self.upper, self.n_points)
