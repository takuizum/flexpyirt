"""flexpyirt: Bock–Aitkin EM estimation for item response theory models, built on JAX."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from flexpyirt import errors
from flexpyirt.correlation import PolychoricResult, polychoric
from flexpyirt.dimensionality import (
    EmpiricalKaiserResult,
    MAPResult,
    ParallelAnalysisResult,
    empirical_kaiser,
    parallel_analysis,
    velicer_map,
)
from flexpyirt.fit import fit
from flexpyirt.items import GPCM, GRM, PCM, FourPL, Nominal, Rasch, ThreePL, TwoPL
from flexpyirt.latent import DavidianCurve, EmpiricalHistogram, Normal, NormalMixture
from flexpyirt.linking import LinkResult, link
from flexpyirt.method import BAEM
from flexpyirt.model import Model
from flexpyirt.priors import BetaPrior, LogNormalPrior, NormalPrior
from flexpyirt.quadrature import Quadrature
from flexpyirt.result import FitResult

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
    "DavidianCurve",
    "EmpiricalHistogram",
    "EmpiricalKaiserResult",
    "FitResult",
    "FourPL",
    "LinkResult",
    "LogNormalPrior",
    "MAPResult",
    "Model",
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
    "empirical_kaiser",
    "errors",
    "fit",
    "link",
    "parallel_analysis",
    "polychoric",
    "velicer_map",
]
