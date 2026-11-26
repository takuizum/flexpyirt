"""flexpyctt: model-free item and test analysis (NumPy / SciPy / pandas, no JAX)."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from flexpyctt import errors
from flexpyctt.correlation import PolychoricResult, polychoric
from flexpyctt.dif import (
    DIFResult,
    generalized_mantel_haenszel,
    logistic_dif,
    mantel,
    mantel_haenszel,
)
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
from flexpyctt.item_analysis import (
    AlphaResult,
    OmegaResult,
    coefficient_alpha,
    coefficient_omega,
    distractor_analysis,
    item_analysis,
    score_responses,
)

try:
    __version__ = version("flexpyctt")
except PackageNotFoundError:  # pragma: no cover - only when running from an unbuilt source tree
    __version__ = "0.0.0+unknown"

__all__ = [
    "AlphaResult",
    "ComparisonDataResult",
    "DIFResult",
    "DetectResult",
    "DimtestResult",
    "EFAResult",
    "EmpiricalKaiserResult",
    "ExploratoryDetectResult",
    "HullResult",
    "MAPResult",
    "OmegaResult",
    "ParallelAnalysisResult",
    "PolychoricResult",
    "__version__",
    "coefficient_alpha",
    "coefficient_omega",
    "comparison_data",
    "detect",
    "dimtest",
    "distractor_analysis",
    "efa",
    "empirical_kaiser",
    "errors",
    "exploratory_detect",
    "generalized_mantel_haenszel",
    "hull",
    "item_analysis",
    "logistic_dif",
    "mantel",
    "mantel_haenszel",
    "parallel_analysis",
    "polychoric",
    "score_responses",
    "velicer_map",
]
