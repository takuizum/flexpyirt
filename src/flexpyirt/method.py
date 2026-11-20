"""Estimation method settings."""

from __future__ import annotations

from dataclasses import dataclass

from flexpyirt.errors import SpecificationError

__all__ = ["BAEM"]


@dataclass(frozen=True)
class BAEM:
    """Settings of the Bock–Aitkin EM algorithm (Bock & Aitkin, 1981).

    Parameters
    ----------
    tol : float
        Convergence when the maximum absolute change of the free parameters between two EM
        iterations is below ``tol`` (same definition as mirt's ``TOL``; DESIGN.md §4.6).
    max_iter : int
        Maximum number of EM iterations.
    m_step_newton_steps : int
        Newton steps per M-step. The M-step need not be solved exactly; each step never decreases
        the expected complete-data log-likelihood, so EM stays monotone (generalized EM).
    """

    tol: float = 1e-4
    max_iter: int = 2000
    m_step_newton_steps: int = 3

    def __post_init__(self) -> None:
        """Validate settings."""
        if not self.tol > 0:
            raise SpecificationError(f"tol must be positive, got {self.tol}.")
        if self.max_iter < 1:
            raise SpecificationError(f"max_iter must be >= 1, got {self.max_iter}.")
        if self.m_step_newton_steps < 1:
            raise SpecificationError(
                f"m_step_newton_steps must be >= 1, got {self.m_step_newton_steps}."
            )
