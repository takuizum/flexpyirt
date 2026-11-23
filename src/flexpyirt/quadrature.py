"""Quadrature grids for the latent variable."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from flexpyirt.errors import SpecificationError

__all__ = ["DEFAULT_N_POINTS", "MAX_FACTORS", "Quadrature"]

MAX_FACTORS = 3
"""Maximum number of latent dimensions supported with product quadrature (DESIGN.md §4.9)."""

DEFAULT_N_POINTS: dict[int, int] = {1: 61, 2: 31, 3: 15}
"""Default equispaced points per dimension, matching mirt's ``select_quadpts``."""


@dataclass(frozen=True)
class Quadrature:
    """Quadrature grid (product grid in 2–3 dimensions): equispaced or Gauss–Hermite.

    Use :meth:`equispaced`, :meth:`default` or :meth:`gauss_hermite` to construct. The weights are
    not stored here: they come from the latent distribution evaluated on the nodes, times a fixed
    per-node base weight that is 1 for equispaced grids (DESIGN.md §4.9, §15.44).
    """

    n_points: int = 61
    lower: float = -6.0
    upper: float = 6.0
    kind: str = "equispaced"

    def __post_init__(self) -> None:
        """Validate the grid."""
        if self.kind not in ("equispaced", "gauss_hermite"):
            raise SpecificationError(f"Unknown quadrature kind {self.kind!r}.")
        if self.kind == "gauss_hermite":
            if not 3 <= self.n_points <= 200:
                raise SpecificationError(
                    f"Gauss-Hermite quadrature needs 3 to 200 points, got {self.n_points}."
                )
            return
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
        cls, n_points: int = 61, *, bounds: tuple[float, float] = (-6.0, 6.0)
    ) -> Quadrature:
        """Return ``n_points`` equally spaced nodes per dimension on ``bounds`` (endpoints kept).

        ``bounds = (lower, upper)`` is mirt's ``theta_lim``. The default (61 points on [-6, 6])
        matches mirt's unidimensional default. For 2-D and 3-D models, :meth:`default` selects 31
        and 15 points per dimension respectively (matching mirt's defaults).
        """
        return cls(n_points=n_points, lower=float(bounds[0]), upper=float(bounds[1]))

    @classmethod
    def gauss_hermite(cls, n_points: int = 21) -> Quadrature:
        r"""Return ``n_points`` Gauss–Hermite nodes per dimension for normal latent distributions.

        Nodes are :math:`\theta_q=\sqrt2\,x_q` with base weights :math:`w_q/\sqrt\pi`, where
        :math:`(x_q, w_q)` are the Gauss–Hermite rule for :math:`e^{-x^2}`, so
        :math:`\sum_q w_q f(\theta_q)/\sqrt\pi\approx E_{N(0,1)}[f]` exactly for polynomials of
        degree :math:`< 2n` (Golub & Welsch, 1969). A ``Normal`` latent with mean :math:`\mu` and
        variance :math:`\sigma^2` reweights the same fixed nodes by
        :math:`\phi(\theta_q;\mu,\sigma^2)/\phi(\theta_q;0,1)` (normalized), so free means and
        variances keep the EM structure (DESIGN.md §15.44). Not available with
        ``EmpiricalHistogram``. Product rules are used in 2–3 dimensions.
        """
        x, _ = np.polynomial.hermite.hermgauss(n_points)
        span = math.sqrt(2.0) * float(np.max(np.abs(x)))
        return cls(n_points=n_points, lower=-span, upper=span, kind="gauss_hermite")

    @classmethod
    def default(
        cls, n_factors: int = 1, *, bounds: tuple[float, float] = (-6.0, 6.0)
    ) -> Quadrature:
        """Return the default equispaced grid for ``n_factors`` in ``{1, 2, 3}`` (DESIGN §4.9).

        Uses 61 points for 1 factor, 31 per dimension (:math:`31^2 = 961`) for 2 factors, and 15
        per dimension (:math:`15^3 = 3375`) for 3 factors, matching mirt's ``select_quadpts``.
        """
        if n_factors not in DEFAULT_N_POINTS:
            raise SpecificationError(
                f"Product quadrature supports 1 to {MAX_FACTORS} factors, got {n_factors}."
            )
        return cls.equispaced(DEFAULT_N_POINTS[n_factors], bounds=bounds)

    def nodes(self) -> np.ndarray:
        """Return the 1-D nodes along each dimension, shape ``(n_points,)``."""
        if self.kind == "gauss_hermite":
            x, _ = np.polynomial.hermite.hermgauss(self.n_points)
            nodes: np.ndarray = math.sqrt(2.0) * x
            return nodes
        return np.linspace(self.lower, self.upper, self.n_points)

    def base_log_weights(self, n_factors: int = 1) -> np.ndarray:
        r"""Return the fixed log base weight of every node, shape ``(n_points**n_factors,)``.

        0 for equispaced grids. For Gauss–Hermite,
        :math:`\sum_m\big[\log(w_{q_m}/\sqrt\pi)+\tfrac12\theta_{q_m}^2\big]`, the log ratio of
        the Gauss–Hermite weight to the standard normal density (up to a constant), which the
        latent density on the node is added to before normalizing (DESIGN.md §15.44).
        """
        if self.kind == "equispaced":
            return np.zeros(self.n_points**n_factors)
        x, w = np.polynomial.hermite.hermgauss(self.n_points)
        per_dim = np.log(w / math.sqrt(math.pi)) + x**2  # theta^2 / 2 = x^2
        if n_factors == 1:
            return per_dim
        meshes = np.meshgrid(*([per_dim] * n_factors), indexing="ij")
        out: np.ndarray = np.sum(np.stack([m.ravel() for m in meshes], axis=1), axis=1)
        return out

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
