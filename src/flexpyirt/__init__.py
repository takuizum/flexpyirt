"""flexpyirt: Bock–Aitkin EM estimation for item response theory models, built on JAX."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from flexpyirt import errors
from flexpyirt.fit import fit
from flexpyirt.model import Model
from flexpyirt.result import FitResult

try:
    __version__ = version("flexpyirt")
except PackageNotFoundError:  # pragma: no cover - only when running from an unbuilt source tree
    __version__ = "0.0.0+unknown"

__all__ = ["FitResult", "Model", "__version__", "errors", "fit"]
