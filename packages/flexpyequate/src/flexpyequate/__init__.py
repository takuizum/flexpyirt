"""flexpyequate: equating of test score distributions, checked against R."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from flexpyequate import errors
from flexpyequate.distributions import BivariateScoreDistribution, ScoreDistribution
from flexpyequate.equating import EquatingResult, equate
from flexpyequate.kernel import KernelEquatingResult, kernel_equate, penalty_bandwidth
from flexpyequate.presmoothing import LoglinearResult, presmooth

try:
    __version__ = version("flexpyequate")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0.0.0"

__all__ = [
    "BivariateScoreDistribution",
    "EquatingResult",
    "KernelEquatingResult",
    "LoglinearResult",
    "ScoreDistribution",
    "__version__",
    "equate",
    "errors",
    "kernel_equate",
    "penalty_bandwidth",
    "presmooth",
]
