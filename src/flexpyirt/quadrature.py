"""Quadrature grids for the latent variable."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from flexpyirt.errors import SpecificationError

__all__ = ["DEFAULT_QUADPTS", "MAX_FACTORS", "Quadrature"]

MAX_FACTORS = 3
"""Maximum number of latent dimensions supported with product quadrature (DESIGN.md §4.9)."""

DEFAULT_QUADPTS: dict[int, int] = {1: 61, 2: 31, 3: 15}
"""Default equispaced points per dimension, matching mirt's ``select_quadpts``."""


@dataclass(frozen=True)
class Quadrature:
    """Equispaced quadrature grid (product grid in 2–3 dimensions).

    Use :meth:`equispaced` or :meth:`default` to construct. The weights are not stored here: they
    come from the latent distribution evaluated on the nodes (DESIGN.md §4.9).
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
        """Return ``n_points`` equally spaced nodes per dimension on ``lim`` (endpoints included).

        The default (61 points on [-6, 6]) matches mirt's unidimensional default. For 2-D and 3-D
        models, :meth:`default` selects 31 and 15 points per dimension respectively (matching
        mirt's defaults).
        """
        return cls(n_points=n_points, lower=float(lim[0]), upper=float(lim[1]))

    @classmethod
    def default(cls, n_factors: int = 1, *, lim: tuple[float, float] = (-6.0, 6.0)) -> Quadrature:
        """Return the default equispaced grid for ``n_factors`` in ``{1, 2, 3}`` (DESIGN §4.9).

        Uses 61 points for 1 factor, 31 per dimension (:math:`31^2 = 961`) for 2 factors, and 15
        per dimension (:math:`15^3 = 3375`) for 3 factors, matching mirt's ``select_quadpts``.
        """
        if n_factors not in DEFAULT_QUADPTS:
            raise SpecificationError(
                f"Product quadrature supports 1 to {MAX_FACTORS} factors, got {n_factors}."
            )
        return cls.equispaced(DEFAULT_QUADPTS[n_factors], lim=lim)

    def nodes(self) -> np.ndarray:
        """Return the 1-D nodes along each dimension, shape ``(n_points,)``."""
        return np.linspace(self.lower, self.upper, self.n_points)

    def grid(self, n_factors: int = 1) -> np.ndarray:
        """Return the product quadrature grid of shape ``(n_points**n_factors, n_factors)``."""
        if not (1 <= n_factors <= MAX_FACTORS):
            raise SpecificationError(
                f"Product quadrature supports 1 to {MAX_FACTORS} factors, got {n_factors}."
            )
        base = self.nodes()
        if n_factors == 1:
            return base[:, None]
        meshes = np.meshgrid(*([base] * n_factors), indexing="ij")
        return np.column_stack([m.ravel() for m in meshes])
