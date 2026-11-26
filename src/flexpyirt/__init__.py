"""flexpyirt: Bock–Aitkin EM estimation for item response theory models, built on JAX.

The model-free dimensionality tools (``parallel_analysis``, ``efa``, ``detect``, ...) are defined
in flexpyctt and re-exported here; ``polychoric`` is flexpyirt's JAX implementation
(DESIGN.md §16.18).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from flexpyirt import errors
from flexpyirt.correlation import PolychoricResult, polychoric
from flexpyirt.dif import DIFScreenResult, dif_effect_sizes, dif_screen
from flexpyirt.dimensionality import (
    ComparisonDataResult,
    DetectResult,
    DimensionalityResult,
    DimtestResult,
    EFAResult,
    EmpiricalKaiserResult,
    ExploratoryDetectResult,
    HullResult,
    JSIResult,
    LocalDependenceResult,
    MAPResult,
    ModelParallelAnalysisResult,
    ParallelAnalysisResult,
    comparison_data,
    detect,
    dimensionality,
    dimtest,
    efa,
    empirical_kaiser,
    exploratory_detect,
    hull,
    model_parallel_analysis,
    parallel_analysis,
    velicer_map,
)
from flexpyirt.fit import fit
from flexpyirt.information import (
    expected_score,
    item_information,
    reliability,
    summed_score_table,
    test_information,
)
from flexpyirt.items import GPCM, GRM, PCM, FourPL, Nominal, Rasch, ThreePL, TwoPL
from flexpyirt.latent import DavidianCurve, EmpiricalHistogram, Normal, NormalMixture
from flexpyirt.linking import LinkResult, link
from flexpyirt.method import BAEM
from flexpyirt.model import Model
from flexpyirt.priors import BetaPrior, LogNormalPrior, NormalPrior
from flexpyirt.quadrature import Quadrature
from flexpyirt.result import FitResult
from flexpyirt.sumscore import summed_score_distribution

try:
    __version__ = version("flexpyirt")
except PackageNotFoundError:  # pragma: no cover - only when running from an unbuilt source tree
    __version__ = "0.0.0+unknown"

__all__ = [
    "BAEM",
    "GPCM",
    "GRM",
    "PCM",
    "BetaPrior",
    "ComparisonDataResult",
    "DIFScreenResult",
    "DavidianCurve",
    "DetectResult",
    "DimensionalityResult",
    "DimtestResult",
    "EFAResult",
    "EmpiricalHistogram",
    "EmpiricalKaiserResult",
    "ExploratoryDetectResult",
    "FitResult",
    "FourPL",
    "HullResult",
    "JSIResult",
    "LinkResult",
    "LocalDependenceResult",
    "LogNormalPrior",
    "MAPResult",
    "Model",
    "ModelParallelAnalysisResult",
    "Nominal",
    "Normal",
    "NormalMixture",
    "NormalPrior",
    "ParallelAnalysisResult",
    "PolychoricResult",
    "Quadrature",
    "Rasch",
    "ThreePL",
    "TwoPL",
    "__version__",
    "comparison_data",
    "detect",
    "dif_effect_sizes",
    "dif_screen",
    "dimensionality",
    "dimtest",
    "efa",
    "empirical_kaiser",
    "errors",
    "expected_score",
    "exploratory_detect",
    "fit",
    "hull",
    "item_information",
    "link",
    "model_parallel_analysis",
    "parallel_analysis",
    "polychoric",
    "reliability",
    "summed_score_distribution",
    "summed_score_table",
    "test_information",
    "velicer_map",
]
