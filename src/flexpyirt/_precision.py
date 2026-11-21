"""Check that JAX runs in 64-bit floating point.

The check is performed at the entry of :func:`flexpyirt.fit` and :meth:`flexpyirt.FitResult.score`,
not at import time, so that importing flexpyirt never depends on or alters the user's global JAX
configuration (CLAUDE.md, principle 4).
"""

from __future__ import annotations

import jax
import numpy as np

from flexpyirt.errors import PrecisionError

__all__ = ["check_x64", "x64_enabled"]

_MESSAGE = (
    "flexpyirt requires 64-bit floating point, but JAX is running in 32-bit mode. "
    'Call `jax.config.update("jax_enable_x64", True)` before creating any JAX arrays '
    "(or set the environment variable JAX_ENABLE_X64=1)."
)


def x64_enabled() -> bool:
    """Return whether JAX currently uses 64-bit floating point.

    Returns
    -------
    bool
        ``True`` if a NumPy ``float64`` request is kept as ``float64`` by JAX.
    """
    return bool(jax.dtypes.canonicalize_dtype(np.float64) == np.dtype(np.float64))


def check_x64() -> None:
    """Raise if JAX is not configured for 64-bit floating point.

    Raises
    ------
    PrecisionError
        If ``jax_enable_x64`` is disabled.
    """
    if not x64_enabled():
        raise PrecisionError(_MESSAGE)
