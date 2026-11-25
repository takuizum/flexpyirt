"""Exception and warning classes raised by flexpyequate.

All errors raised intentionally by the library derive from :class:`FlexpyequateError`, and all
warnings from :class:`FlexpyequateWarning`, so callers can catch them as a family.
"""

from __future__ import annotations

__all__ = [
    "ConvergenceWarning",
    "FlexpyequateError",
    "FlexpyequateWarning",
    "ScoreDistributionError",
    "SpecificationError",
]


class FlexpyequateError(Exception):
    """Base class for all errors raised by flexpyequate."""


class ScoreDistributionError(FlexpyequateError, ValueError):
    """Scores or frequencies do not form a valid score distribution.

    Examples: possible scores that are not equally spaced, negative frequencies, an observed score
    that is not one of the possible scores, or a distribution without any observation.
    """


class SpecificationError(FlexpyequateError, ValueError):
    """The arguments of an equating call are inconsistent with each other or with the data.

    Examples: a single-group method given two separate distributions, an unknown method name, or a
    polynomial degree that the number of possible scores cannot support.
    """


class FlexpyequateWarning(UserWarning):
    """Base class for all warnings issued by flexpyequate."""


class ConvergenceWarning(FlexpyequateWarning):
    """An iterative procedure stopped before reaching its convergence criterion."""
