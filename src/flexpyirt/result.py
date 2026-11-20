"""Estimation result (Result layer, DESIGN.md §2 ④).

M0 placeholder: only the type exists. Parameter tables, standard errors, log-likelihood and
scoring are added from M1 on (API.md §7).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["FitResult"]


@dataclass(frozen=True)
class FitResult:
    """Result of :func:`flexpyirt.fit`."""
