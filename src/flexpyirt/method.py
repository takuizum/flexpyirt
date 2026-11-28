"""Estimation method settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from flexpyirt.errors import SpecificationError

__all__ = ["BAEM"]


@dataclass(frozen=True)
class BAEM:
    r"""Settings of the Bock–Aitkin EM algorithm (Bock & Aitkin, 1981).

    Parameters
    ----------
    tol : float
        Convergence when the maximum absolute change of the free parameters between successive
        accepted iterates is below ``tol`` (same definition as mirt's ``TOL``; DESIGN.md §4.6).
    max_iter : int
        Maximum number of accepted EM (or SQUAREM) iterations.
    m_step_newton_steps : int
        Newton steps per M-step. The M-step need not be solved exactly; each step never decreases
        the expected complete-data log-likelihood, so EM stays monotone (generalized EM).
    squarem : bool
        If ``True``, accelerate the EM map with SQUAREM scheme S3 (Varadhan & Roland, 2008) and a
        monotonicity/feasibility safeguard (DESIGN.md §4.6). Defaults to ``False``.
    m_step_scale : {"natural", "unconstrained"}
        Scale on which the M-step Newton steps the bounded parameters (DESIGN.md §15.2, §15.47).
        ``"natural"`` (default) steps every parameter on its natural scale with a bounded active
        set, so boundary solutions (e.g. :math:`g=0`) are reached and flagged by
        :class:`~flexpyirt.errors.HeywoodWarning`. ``"unconstrained"`` steps each free parameter
        with a finite bound on :math:`\operatorname{logit}((x-l)/(u-l))` (both bounds finite,
        e.g. :math:`g`, :math:`u` and factor correlations), :math:`\log(x-l)` (lower bound only,
        e.g. latent variances) or :math:`\log(u-x)`; unbounded parameters such as slopes are
        not transformed. The objective, the estimates at an interior solution, the parameter
        table, the priors, the standard errors and SQUAREM are unchanged; a boundary solution is
        only approached and may need many more iterations.
    anderson : bool
        If ``True``, accelerate the EM map :math:`F` with Anderson acceleration (Anderson, 1965;
        Henderson & Varadhan, 2019) on the unconstrained coordinates :math:`z=T(\eta)` of
        ``m_step_scale="unconstrained"`` (logit/log of the bounded parameters). With residual
        :math:`g_k=T(F(\eta_k))-z_k` and the last :math:`m=5` differences
        :math:`\Delta Z, \Delta G` of :math:`z` and :math:`g`,
        :math:`z_{k+1}=z_k+g_k-(\Delta Z+\Delta G)\gamma_k` with
        :math:`\gamma_k=\arg\min_\gamma\lVert g_k-\Delta G\gamma\rVert_2`. Each iteration
        costs one EM map evaluation (SQUAREM costs three plus the candidate checks). A candidate
        that leaves the bounds or lowers the log posterior is discarded and EM restarts from the
        last EM step, so the log posterior never decreases. Convergence needs the change of the
        accepted iterate and of the EM step from it below ``tol`` on two consecutive iterations
        (DESIGN.md §15.49). Cannot be combined with ``squarem``. Defaults to ``False``.
    """

    tol: float = 1e-4
    max_iter: int = 2000
    m_step_newton_steps: int = 3
    squarem: bool = False
    m_step_scale: Literal["natural", "unconstrained"] = "natural"
    anderson: bool = False

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
        if not isinstance(self.squarem, bool):
            raise SpecificationError(f"squarem must be a bool, got {type(self.squarem).__name__}.")
        if not isinstance(self.anderson, bool):
            raise SpecificationError(
                f"anderson must be a bool, got {type(self.anderson).__name__}."
            )
        if self.anderson and self.squarem:
            raise SpecificationError(
                "squarem and anderson are alternative accelerations; set at most one to True."
            )
        if self.m_step_scale not in ("natural", "unconstrained"):
            raise SpecificationError(
                f"m_step_scale must be 'natural' or 'unconstrained', got {self.m_step_scale!r}."
            )
