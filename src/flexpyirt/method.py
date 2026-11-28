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
    warmup_dtype : {"float32"} or None
        If ``"float32"``, first run EM (or SQUAREM) on a ``float32`` copy of the problem until the
        largest change is below :attr:`warmup_tol`, then continue in ``float64`` on the original
        problem from there until the largest change is below ``tol`` (DESIGN.md §28). The
        estimates, log-likelihood and standard errors all come from the ``float64`` stage, so
        they meet the same stopping rule as a pure ``float64`` fit; they agree with it to about
        ``tol`` (the two runs stop at different points of the same EM path, not bit for bit).
        If the ``float32`` stage returns non-finite values, it is discarded and the ``float64``
        stage starts from the usual starting values. ``None`` (default) runs in ``float64``
        only. The ``float32`` stage pays off on hardware with slow ``float64`` arithmetic (most
        GPUs); on a CPU it saves about 10–30% of plain EM time on large problems and nothing
        with ``squarem=True``, and it adds a second compilation. With a likelihood that has
        several maxima (``NormalMixture``, 3PL) the warm start may lead to a different maximum
        than the classical starting values would.
    """

    tol: float = 1e-4
    max_iter: int = 2000
    m_step_newton_steps: int = 3
    squarem: bool = False
    m_step_scale: Literal["natural", "unconstrained"] = "natural"
    warmup_dtype: Literal["float32"] | None = None

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
        if self.warmup_dtype not in (None, "float32"):
            raise SpecificationError(
                f"warmup_dtype must be 'float32' or None, got {self.warmup_dtype!r}."
            )

    @property
    def warmup_tol(self) -> float:
        r"""Stopping tolerance of the ``float32`` warmup, :math:`\max(10\,\mathrm{tol},10^{-3})`.

        ``float32`` resolves about :math:`10^{-7}` relative, so the warmup stops well above that
        and leaves the last digits to the ``float64`` stage (DESIGN.md §28).
        """
        return max(10.0 * self.tol, 1e-3)
