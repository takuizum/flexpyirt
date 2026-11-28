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
    asymptote_m_step : {"joint", "mixture"}
        M-step of 3PL / 4PL items (DESIGN.md §15.49). ``"joint"`` (default) maximizes the usual
        expected complete-data log-likelihood over :math:`(a, d, g, u)` jointly by Newton.
        ``"mixture"`` (EMM; Zheng, Meng, Guo & Liu, 2018) treats each response as a guess, a slip
        or an ability-based response, which makes the asymptote update closed form and the
        slope–intercept update a weighted 2PL fit; both parts are concave. The marginal
        likelihood, its maximum, the priors and the standard errors are unchanged; the number of
        iterations differs. Other item models, and items linked across items, are unaffected.
    """

    tol: float = 1e-4
    max_iter: int = 2000
    m_step_newton_steps: int = 3
    squarem: bool = False
    m_step_scale: Literal["natural", "unconstrained"] = "natural"
    asymptote_m_step: Literal["joint", "mixture"] = "joint"

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
        if self.m_step_scale not in ("natural", "unconstrained"):
            raise SpecificationError(
                f"m_step_scale must be 'natural' or 'unconstrained', got {self.m_step_scale!r}."
            )
        if self.asymptote_m_step not in ("joint", "mixture"):
            raise SpecificationError(
                f"asymptote_m_step must be 'joint' or 'mixture', got {self.asymptote_m_step!r}."
            )
