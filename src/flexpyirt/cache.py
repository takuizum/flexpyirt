"""Opt-in persistent JAX compilation cache (DESIGN.md §28).

Importing flexpyirt never changes the global JAX configuration (CLAUDE.md, principle 4).
:func:`enable_compilation_cache` is the one explicit call that does: it points JAX's persistent
compilation cache at a directory, so a later Python process that fits the same model shapes
loads the compiled XLA programs from disk instead of compiling them again.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import jax

from flexpyirt.errors import SpecificationError

__all__ = ["default_cache_dir", "enable_compilation_cache"]


def default_cache_dir() -> Path:
    """Return the default cache directory: the user's OS cache directory plus ``flexpyirt/jax``.

    ``$XDG_CACHE_HOME`` (else ``~/.cache``) on Linux and other POSIX systems,
    ``~/Library/Caches`` on macOS and ``%LOCALAPPDATA%`` on Windows.

    Returns
    -------
    pathlib.Path
        The directory (not created).
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "flexpyirt" / "jax"


def enable_compilation_cache(
    path: str | os.PathLike[str] | None = None,
    *,
    min_compile_time: float = 0.0,
) -> Path:
    """Turn on JAX's persistent compilation cache for this process.

    JAX compiles each fitting program (EM, SQUAREM, standard errors) the first time it sees a
    model's shapes, which takes about 0.5–3 s per program on a CPU. With the cache on, the
    compiled programs are written to ``path``, and any later process that calls this function
    with the same ``path`` and fits a model of the same shapes (same item types, number of items,
    number of groups, quadrature points and padded number of response patterns) reads them back
    instead. Results are identical: the cache stores the same compiled program.

    This changes JAX's global configuration (``jax_compilation_cache_dir`` and
    ``jax_persistent_cache_min_compile_time_secs``) for the whole process, including your own JAX
    code, which is why flexpyirt never does it on import. Call it once, before the first fit.

    Parameters
    ----------
    path : str or os.PathLike, optional
        Cache directory, created if needed. Defaults to :func:`default_cache_dir`.
        Anyone who can write to this directory can make your process run arbitrary code, so do
        not share it with untrusted users.
    min_compile_time : float, default 0.0
        Only programs that took at least this many seconds to compile are written to the cache
        (JAX's own default, 1 s, would skip most of flexpyirt's programs).

    Returns
    -------
    pathlib.Path
        The cache directory in use.

    Raises
    ------
    SpecificationError
        If ``min_compile_time`` is negative.

    Examples
    --------
    >>> import flexpyirt as fp
    >>> cache_dir = fp.enable_compilation_cache()  # doctest: +SKIP
    """
    if min_compile_time < 0:
        raise SpecificationError(f"min_compile_time must be >= 0, got {min_compile_time}.")
    directory = Path(default_cache_dir() if path is None else path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(directory))  # type: ignore[no-untyped-call]
    jax.config.update(  # type: ignore[no-untyped-call]
        "jax_persistent_cache_min_compile_time_secs", float(min_compile_time)
    )
    return directory
