"""flexpyirt: Bock–Aitkin EM estimation for item response theory models, built on JAX."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from flexpyirt import errors
from flexpyirt.fit import fit
from flexpyirt.items import GPCM, GRM, PCM, FourPL, Nominal, Rasch, ThreePL, TwoPL
from flexpyirt.latent import EmpiricalHistogram, Normal
from flexpyirt.method import BAEM
from flexpyirt.model import Model
from flexpyirt.priors import Beta, LogNormal, NormalPrior
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
    "Beta",
    "EmpiricalHistogram",
    "FitResult",
    "FourPL",
    "LogNormal",
    "Model",
    "Nominal",
    "Normal",
    "NormalPrior",
    "Quadrature",
    "Rasch",
    "ThreePL",
    "TwoPL",
    "__version__",
    "errors",
    "fit",
]
