"""Quadrature grids for the latent variable."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri

from flexpyirt.errors import SpecificationError

__all__ = ["DEFAULT_N_POINTS", "MAX_FACTORS", "MAX_KERNEL_FACTORS", "MAX_QMC_FACTORS", "Quadrature"]

MAX_FACTORS = 3
"""Maximum number of latent dimensions supported with product quadrature (DESIGN.md §4.9)."""

MAX_QMC_FACTORS = 5
"""Maximum number of latent dimensions supported with quasi-Monte Carlo quadrature (§28)."""

_QMC_MIN_POINTS = 64
_QMC_MAX_POINTS = 200_000
_HALTON_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)

MAX_KERNEL_FACTORS = 32
"""Maximum number of slopes per item kernel; bifactor models reach it with 31 specific factors
because their E-step uses (1+1)-dimensional grids (DESIGN.md §17.1)."""

DEFAULT_N_POINTS: dict[int, int] = {1: 61, 2: 31, 3: 15}
"""Default equispaced points per dimension, matching mirt's ``select_quadpts``."""


@dataclass(frozen=True)
class Quadrature:
    """Quadrature grid: equispaced or Gauss–Hermite product grids, or quasi-Monte Carlo points.

    Use :meth:`equispaced`, :meth:`default`, :meth:`gauss_hermite` or :meth:`qmc` to construct.
    The weights are not stored here: they come from the latent distribution evaluated on the nodes,
    times a fixed per-node base weight that is 1 for equispaced grids (DESIGN.md §4.9, §15.44,
    §28). For product grids ``n_points`` counts the points per dimension; for ``"qmc"`` it is the
    total number of points.
    """

    n_points: int = 61
    lower: float = -6.0
    upper: float = 6.0
    kind: str = "equispaced"
    proposal_mean: tuple[float, ...] | None = None
    proposal_cov: tuple[tuple[float, ...], ...] | None = None

    def __post_init__(self) -> None:
        """Validate the grid."""
        if self.kind != "qmc" and (self.proposal_mean is not None or self.proposal_cov is not None):
            raise SpecificationError("A proposal mean/covariance is only used by qmc quadrature.")
        if self.kind not in ("equispaced", "gauss_hermite", "qmc"):
            raise SpecificationError(f"Unknown quadrature kind {self.kind!r}.")
        if self.kind == "qmc":
            if not _QMC_MIN_POINTS <= self.n_points <= _QMC_MAX_POINTS:
                raise SpecificationError(
                    f"Quasi-Monte Carlo quadrature needs {_QMC_MIN_POINTS} to {_QMC_MAX_POINTS} "
                    f"points, got {self.n_points}."
                )
            if (self.proposal_mean is None) != (self.proposal_cov is None):
                raise SpecificationError("Give both the proposal mean and covariance, or neither.")
            if self.proposal_mean is not None:
                _proposal_cholesky(self.proposal_mean, self.proposal_cov)
            return
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
    def qmc(
        cls,
        n_points: int = 5000,
        *,
        mean: Sequence[float] | np.ndarray | None = None,
        cov: Sequence[Sequence[float]] | np.ndarray | None = None,
    ) -> Quadrature:
        r"""Return ``n_points`` fixed quasi-Monte Carlo nodes for normal latent distributions.

        The nodes are the Halton points (Halton, 1960) :math:`\mathbf u_q`, :math:`q=1,\dots,Q`
        (component :math:`m` is the radical inverse of :math:`q` in the :math:`m`-th prime base),
        mapped to :math:`\mathbf z_q=\Phi^{-1}(\mathbf u_q)`, with base weights :math:`1/Q`, so
        :math:`\frac1Q\sum_q f(\mathbf z_q)\approx E_{N(0,I)}[f]` with error of order
        :math:`(\log Q)^D/Q` for smooth :math:`f`. A multivariate ``Normal`` latent reweights
        the fixed nodes by the ratio of its density to the proposal's (normalized), as for
        :meth:`gauss_hermite`, so the weights are positive and the EM structure (exact latent
        M-step, monotonicity, Oakes SEs) is unchanged (DESIGN.md §28). mirt's
        ``method = "QMCEM"`` instead moves its nodes every cycle, so the two agree up to QMC
        error. Multidimensional ``Normal`` models only (not unidimensional, bifactor or two-tier
        models).

        The reweighting has finite variance only when the latent covariance is not much wider than
        the proposal (every eigenvalue of :math:`\boldsymbol\Sigma` below twice the proposal's).
        ``mean`` and ``cov`` move the fixed nodes to :math:`\mathbf m+\mathbf L\mathbf z_q`
        (:math:`\mathbf C=\mathbf L\mathbf L^\top`) with base weights
        :math:`-\log Q-\log\phi(\boldsymbol\theta_q;\mathbf m,\mathbf C)`. Without them,
        :func:`~flexpyirt.fit` fits once on :math:`N(\mathbf 0,\mathbf I)` nodes and refits on
        nodes moved to that fit's latent mean and covariance (DESIGN.md §28).
        """
        if (mean is None) != (cov is None):
            raise SpecificationError("Give both mean and cov, or neither.")
        if mean is None:
            return cls(n_points=int(n_points), kind="qmc")
        mean_t = tuple(float(m) for m in mean)
        cov_t = tuple(tuple(float(c) for c in row) for row in np.asarray(cov, dtype=float))
        return cls(n_points=int(n_points), kind="qmc", proposal_mean=mean_t, proposal_cov=cov_t)

    def with_proposal(
        self, mean: Sequence[float] | np.ndarray, cov: Sequence[Sequence[float]] | np.ndarray
    ) -> Quadrature:
        """Return the same qmc quadrature with its fixed nodes moved to ``N(mean, cov)``."""
        if self.kind != "qmc":
            raise SpecificationError("with_proposal() is only for qmc quadrature.")
        return Quadrature.qmc(self.n_points, mean=mean, cov=cov)

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
                f"Product quadrature supports 1 to {MAX_FACTORS} factors, got {n_factors}; "
                "pass quadrature=Quadrature.qmc() for more factors."
            )
        return cls.equispaced(DEFAULT_N_POINTS[n_factors], bounds=bounds)

    def nodes(self) -> np.ndarray:
        """Return the 1-D nodes along each dimension, shape ``(n_points,)``."""
        if self.kind == "qmc":
            raise SpecificationError(
                "Quasi-Monte Carlo quadrature has no per-dimension axis; use grid(n_factors)."
            )
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
        if self.kind == "qmc":
            z = _halton_normal(self.n_points, n_factors)
            qmc_base: np.ndarray = -math.log(self.n_points) + 0.5 * np.sum(z * z, axis=1)
            if self.proposal_mean is not None:
                chol = self._proposal(n_factors)[1]
                qmc_base = qmc_base + float(np.sum(np.log(np.diag(chol))))
            return qmc_base
        x, w = np.polynomial.hermite.hermgauss(self.n_points)
        per_dim = np.log(w / math.sqrt(math.pi)) + x**2  # theta^2 / 2 = x^2
        if n_factors == 1:
            return per_dim
        meshes = np.meshgrid(*([per_dim] * n_factors), indexing="ij")
        out: np.ndarray = np.sum(np.stack([m.ravel() for m in meshes], axis=1), axis=1)
        return out

    def grid(self, n_factors: int = 1) -> np.ndarray:
        r"""Return the grid: ``(n_points**n_factors, n_factors)``, or ``(n_points, D)`` for qmc.

        Product grids for ``"equispaced"`` and ``"gauss_hermite"``; the ``n_points`` Halton nodes
        :math:`\Phi^{-1}(\mathbf u_q)` for ``"qmc"`` (:meth:`qmc`).
        """
        if self.kind == "qmc":
            z = _halton_normal(self.n_points, n_factors)
            if self.proposal_mean is None:
                return z
            mean, chol = self._proposal(n_factors)
            moved: np.ndarray = mean[None, :] + z @ chol.T
            return moved
        if not (1 <= n_factors <= MAX_FACTORS):
            raise SpecificationError(
                f"Product quadrature supports 1 to {MAX_FACTORS} factors, got {n_factors}."
            )
        base = self.nodes()
        if n_factors == 1:
            return base[:, None]
        meshes = np.meshgrid(*([base] * n_factors), indexing="ij")
        return np.column_stack([m.ravel() for m in meshes])

    def _proposal(self, n_factors: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the proposal mean and Cholesky factor, checking the dimension."""
        mean, chol = _proposal_cholesky(self.proposal_mean, self.proposal_cov)
        if mean.shape[0] != n_factors:
            raise SpecificationError(
                f"The qmc proposal has {mean.shape[0]} dimensions, but the model has {n_factors}."
            )
        return mean, chol


def _proposal_cholesky(
    mean: Sequence[float] | None, cov: Sequence[Sequence[float]] | None
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a proposal ``N(mean, cov)`` and return ``(mean, cholesky(cov))``."""
    m = np.asarray(mean, dtype=np.float64)
    c = np.asarray(cov, dtype=np.float64)
    if m.ndim != 1 or c.shape != (m.shape[0], m.shape[0]):
        raise SpecificationError(
            f"The qmc proposal needs a mean of length D and a D x D covariance, got shapes "
            f"{m.shape} and {c.shape}."
        )
    if not (np.all(np.isfinite(m)) and np.all(np.isfinite(c)) and np.allclose(c, c.T)):
        raise SpecificationError("The qmc proposal covariance must be finite and symmetric.")
    try:
        chol = np.linalg.cholesky(c)
    except np.linalg.LinAlgError as err:
        raise SpecificationError("The qmc proposal covariance must be positive definite.") from err
    return m, chol


def _radical_inverse(index: np.ndarray, base: int) -> np.ndarray:
    """Return the radical inverse of positive integers ``index`` in ``base`` (van der Corput)."""
    out = np.zeros(index.shape, dtype=np.float64)
    rest = index.copy()
    scale = 1.0 / base
    while np.any(rest > 0):
        out += (rest % base) * scale
        rest //= base
        scale /= base
    return out


def _halton_normal(n_points: int, n_factors: int) -> np.ndarray:
    r"""Return ``n_points`` Halton points mapped by :math:`\Phi^{-1}`, shape ``(n_points, D)``.

    Point :math:`q=1,\dots,Q` has component :math:`m` equal to the radical inverse of :math:`q` in
    the :math:`m`-th prime (Halton, 1960); :math:`q=0` (the origin of the unit cube, which maps to
    :math:`-\infty`) is skipped. Every component lies in :math:`(0,1)`.
    """
    if not 1 <= n_factors <= len(_HALTON_BASES):
        raise SpecificationError(
            f"Quasi-Monte Carlo quadrature supports 1 to {len(_HALTON_BASES)} dimensions, "
            f"got {n_factors}."
        )
    index = np.arange(1, n_points + 1, dtype=np.int64)
    u = np.column_stack([_radical_inverse(index, b) for b in _HALTON_BASES[:n_factors]])
    grid: np.ndarray = ndtri(u)
    return grid
