"""Exception and warning classes raised by flexpyirt.

All errors raised intentionally by the library derive from :class:`FlexpyirtError`, and all
warnings from :class:`FlexpyirtWarning`, so callers can catch them as a family (DESIGN.md §10).

The classes shared with flexpyctt (``SpecificationError`` and the warnings) are defined there and
re-exported here; ``FlexpyirtError`` and ``FlexpyirtWarning`` are flexpyctt's base classes under
flexpyirt's names, so catching them also catches what a flexpyctt function raises
(DESIGN.md §16.18).
"""

from __future__ import annotations

from flexpyctt.errors import (
    ConvergenceWarning,
    FlexpycttError,
    FlexpycttWarning,
    HeywoodWarning,
    NotPositiveDefiniteWarning,
    SpecificationError,
)

__all__ = [
    "ConvergenceWarning",
    "FlexpyirtError",
    "FlexpyirtWarning",
    "HeywoodWarning",
    "IdentificationError",
    "NotPositiveDefiniteWarning",
    "PrecisionError",
    "SpecificationError",
]


FlexpyirtError = FlexpycttError
"""Base class for all errors raised by flexpyirt (``flexpyctt.FlexpycttError``)."""

FlexpyirtWarning = FlexpycttWarning
"""Base class for all warnings emitted by flexpyirt (``flexpyctt.FlexpycttWarning``)."""


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
