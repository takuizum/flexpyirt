"""flexpyctt: model-free item and test analysis (NumPy / SciPy / pandas, no JAX)."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from flexpyctt import errors
from flexpyctt.correlation import PolychoricResult, polychoric
from flexpyctt.dimensionality import (
    ComparisonDataResult,
    DetectResult,
    DimtestResult,
    EFAResult,
    EmpiricalKaiserResult,
    ExploratoryDetectResult,
    HullResult,
    MAPResult,
    ParallelAnalysisResult,
    comparison_data,
    detect,
    dimtest,
    efa,
    empirical_kaiser,
    exploratory_detect,
    hull,
    parallel_analysis,
    velicer_map,
)

try:
    __version__ = version("flexpyctt")
except PackageNotFoundError:  # pragma: no cover - only when running from an unbuilt source tree
    __version__ = "0.0.0+unknown"

__all__ = [
    "ComparisonDataResult",
    "DetectResult",
    "DimtestResult",
    "EFAResult",
    "EmpiricalKaiserResult",
    "ExploratoryDetectResult",
    "HullResult",
    "MAPResult",
    "ParallelAnalysisResult",
    "PolychoricResult",
    "__version__",
    "comparison_data",
    "detect",
    "dimtest",
    "efa",
    "empirical_kaiser",
    "errors",
    "exploratory_detect",
    "hull",
    "parallel_analysis",
    "polychoric",
    "velicer_map",
]
