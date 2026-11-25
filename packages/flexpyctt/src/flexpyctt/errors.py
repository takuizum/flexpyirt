"""Exception and warning classes raised by flexpyctt.

All errors raised intentionally by the library derive from :class:`FlexpycttError`, and all
warnings from :class:`FlexpycttWarning`, so callers can catch them as a family. flexpyirt
re-exports these classes from ``flexpyirt.errors`` (DESIGN.md §16.18).
"""

from __future__ import annotations

__all__ = [
    "ConvergenceWarning",
    "FlexpycttError",
    "FlexpycttWarning",
    "HeywoodWarning",
    "NotPositiveDefiniteWarning",
    "SpecificationError",
]


class FlexpycttError(Exception):
    """Base class for all errors raised by flexpyctt."""


class SpecificationError(FlexpycttError, ValueError):
    """The request is inconsistent with itself or with the data.

    Examples: an option outside its range, non-numeric responses, or an item with a single
    observed category. Raised before any computation starts.
    """


class FlexpycttWarning(UserWarning):
    """Base class for all warnings emitted by flexpyctt."""


class ConvergenceWarning(FlexpycttWarning):
    """Estimation stopped without meeting the convergence criterion (e.g. ``max_iter`` reached)."""


class HeywoodWarning(FlexpycttWarning):
    """A boundary solution was reached (e.g. a correlation at 1 or a uniqueness at its bound)."""


class NotPositiveDefiniteWarning(FlexpycttWarning):
    """A correlation matrix (e.g. a pairwise polychoric matrix) was not positive definite.

    It was smoothed to a positive definite matrix.
    """
