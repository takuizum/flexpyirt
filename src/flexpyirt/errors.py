"""Exception and warning classes raised by flexpyirt.

All errors raised intentionally by the library derive from :class:`FlexpyirtError`, and all
warnings from :class:`FlexpyirtWarning`, so callers can catch them as a family (DESIGN.md §10).
"""

from __future__ import annotations

__all__ = [
    "ConvergenceWarning",
    "FlexpyirtError",
    "FlexpyirtWarning",
    "HeywoodWarning",
    "IdentificationError",
    "PrecisionError",
    "SpecificationError",
]


class FlexpyirtError(Exception):
    """Base class for all errors raised by flexpyirt."""


class SpecificationError(FlexpyirtError, ValueError):
    """The model specification is inconsistent with itself or with the data.

    Examples: fixing a parameter of an item that does not exist, or a category count that does not
    match the observed responses. Raised before estimation starts.
    """


class IdentificationError(FlexpyirtError, ValueError):
    """The model is not identified.

    Example: freeing the mean and variance of a focal group without any anchor item. Raised before
    estimation starts.
    """


class PrecisionError(FlexpyirtError, RuntimeError):
    """64-bit floating point is not enabled in JAX.

    flexpyirt requires ``jax_enable_x64``. It never changes this global setting
    on the user's behalf.
    """


class FlexpyirtWarning(UserWarning):
    """Base class for all warnings emitted by flexpyirt."""


class ConvergenceWarning(FlexpyirtWarning):
    """Estimation stopped without meeting the convergence criterion (e.g. ``max_iter`` reached)."""


class HeywoodWarning(FlexpyirtWarning):
    """A boundary solution was reached (e.g. guessing parameter near 0, diverging slope)."""
