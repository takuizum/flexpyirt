"""Dimensionality diagnostics that use a fitted model, and the ensemble (DESIGN.md §16.8, §16.18).

:func:`model_parallel_analysis` and :func:`dimensionality` are defined here, with the result
classes of :meth:`flexpyirt.FitResult.local_dependence` and :meth:`flexpyirt.FitResult.jsi`. The
model-free tools (:func:`parallel_analysis`, :func:`velicer_map`, :func:`empirical_kaiser`,
:func:`efa`, :func:`hull`, :func:`comparison_data`, :func:`detect`, :func:`exploratory_detect`,
:func:`dimtest`) live in flexpyctt, which does not need JAX; they are re-exported here and from
``flexpyirt`` so existing imports keep working.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flexpyctt import polychoric
from flexpyctt.correlation import _coerce_ordinal
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
    _descending_eigenvalues,
    _eigenvalues,
    _observed_correlation,
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

from flexpyirt._compile import coerce_responses
from flexpyirt._precision import check_x64
from flexpyirt.errors import FlexpyirtWarning, SpecificationError

if TYPE_CHECKING:
    from flexpyirt.result import FitResult

__all__ = [
    "ComparisonDataResult",
    "DetectResult",
    "DimensionalityResult",
    "DimtestResult",
    "EFAResult",
    "EmpiricalKaiserResult",
    "ExploratoryDetectResult",
    "HullResult",
    "JSIResult",
    "LocalDependenceResult",
    "MAPResult",
    "ModelParallelAnalysisResult",
    "ParallelAnalysisResult",
    "comparison_data",
    "detect",
    "dimensionality",
    "dimtest",
    "efa",
    "empirical_kaiser",
    "exploratory_detect",
    "hull",
    "model_parallel_analysis",
    "parallel_analysis",
    "velicer_map",
]


def _rng_from_key(key: jax.Array) -> np.random.Generator:
    """Return a NumPy generator seeded with the data of a JAX PRNG key.

    flexpyctt's functions take a :class:`numpy.random.Generator`; the functions here take a JAX
    key, split it as before and derive one generator per flexpyctt call (DESIGN.md §16.18).
    """
    data = jax.random.key_data(key) if jnp.issubdtype(key.dtype, jax.dtypes.prng_key) else key
    return np.random.default_rng([int(word) for word in np.asarray(data).ravel()])


# ----------------------------------------------------------------------------- local dependence
@dataclass(frozen=True)
class LocalDependenceResult:
    """Result of :meth:`flexpyirt.FitResult.local_dependence`.

    Attributes
    ----------
    pairs : pandas.DataFrame
        One row per item pair (per group and pair, with a leading ``group`` column, in a
        multiple-group model): ``item_i``, ``item_j``, ``n`` (respondents answering both),
        ``q3``, ``q3_star``, ``x2``, ``g2``, ``df``, ``p_x2``, ``p_g2``, ``cramer_v``. A pair
        that nobody answered together (``n = 0``, e.g. in a booklet design) has NaN statistics.
    q3 : pandas.DataFrame
        Item x item Q3 matrix (unit diagonal); in a multiple-group model one matrix per group
        stacked on a ``(group, item)`` row index.
    q3_mean : float or pandas.Series
        Mean off-diagonal Q3 (subtracted in ``q3_star``; over the pairs that have a Q3, which
        needs two respondents answering both items); in a multiple-group model a Series
        indexed by group, each group's ``q3_star`` using its own mean.
    """

    pairs: pd.DataFrame
    q3: pd.DataFrame
    q3_mean: float | pd.Series

    @property
    def max_q3_star(self) -> float:
        """Largest Q3* over pairs, and over groups (Christensen, Makransky & Horton, 2017)."""
        return float(self.pairs["q3_star"].max())


JSIMethod = Literal["refit", "newton", "one_step"]


@dataclass(frozen=True)
class JSIResult:
    r"""Result of :meth:`flexpyirt.FitResult.jsi` (Edwards, Houts & Cai, 2018).

    Attributes
    ----------
    matrix : pandas.DataFrame
        Removed item (rows) x item (columns): row :math:`j`, column :math:`i` is
        :math:`(\hat a_i-\hat a_i^{(-j)})/\mathrm{SE}(\hat a_i^{(-j)})`, the standardized drop in
        item :math:`i`'s slope when item :math:`j` is removed (mirt's
        ``residuals(type = "JSI", fold = FALSE)``). The diagonal is NaN, as is every column of
        an item without a free slope.
    pairs : pandas.DataFrame
        One row per item pair :math:`i<j`: ``item_i``, ``item_j``, ``jsi_ij`` (item
        :math:`i`'s slope when :math:`j` is removed), ``jsi_ji`` and ``folded``
        (:math:`\mathrm{JSI}_{ij}+\mathrm{JSI}_{ji}`, mirt's ``fold = TRUE``).
    method : str
        ``"refit"``, ``"newton"`` or ``"one_step"``.
    converged : pandas.Series
        Per removed item, whether its refit converged (always ``True`` for the approximations).
    """

    matrix: pd.DataFrame
    pairs: pd.DataFrame
    method: str
    converged: pd.Series


# ----------------------------------------------------------------------------- model-based PA
ModelEigenType = Literal["max", "smc", "pca"]
_MODEL_EIGEN_TYPES: tuple[str, ...] = ("max", "smc", "pca")


def _model_eigenvalues(mat: np.ndarray, eigen_type: str) -> np.ndarray:
    """Return descending eigenvalues (``"max"``: largest off-diagonal r on the diagonal)."""
    if eigen_type == "max":
        out = mat.copy()
        np.fill_diagonal(out, np.max(mat[np.triu_indices_from(mat, k=1)]))
        return _descending_eigenvalues(out)
    return _eigenvalues(mat, eigen_type)


@dataclass(frozen=True)
class ModelParallelAnalysisResult:
    r"""Result of :func:`model_parallel_analysis`.

    Attributes
    ----------
    statistic : float
        Second eigenvalue :math:`T` of the observed matrix.
    p_value : float
        :math:`(1+\#\{b: T_b\ge T\})/(B+1)` over the usable replications.
    observed : numpy.ndarray
        Observed eigenvalues (descending).
    replications : numpy.ndarray
        ``(B, p)`` eigenvalues of the data simulated from the fitted model.
    eigen_type : str
        The eigenvalue type used.
    n_skipped : int
        Replications whose correlations could not be estimated (excluded).
    """

    statistic: float
    p_value: float
    observed: np.ndarray
    replications: np.ndarray
    eigen_type: str
    n_skipped: int


def model_parallel_analysis(
    fit: FitResult,
    data: pd.DataFrame | np.ndarray,
    *,
    key: jax.Array,
    n_replications: int = 100,
    eigen_type: ModelEigenType = "max",
    zero_cell_correction: float = 0.5,
    item_names: list[str] | None = None,
) -> ModelParallelAnalysisResult:
    r"""Model-based (modified) parallel analysis of a fitted unidimensional model.

    Drasgow & Lissak (1983): the second eigenvalue :math:`T` of the observed polychoric
    (tetrachoric) matrix is compared with its distribution in data simulated from the fitted
    unidimensional model, :math:`T_1,\dots,T_B`; a small

    .. math::
        p=\frac{1+\#\{b: T_b\ge T\}}{B+1}

    indicates more second-dimension structure than the model produces. Simulated data have the
    same size and missingness pattern as ``data``; :math:`\theta\sim N(\hat\mu,\hat\sigma^2)`.
    With ``eigen_type="max"`` every diagonal element is the largest off-diagonal correlation, as
    in ltm's ``unidimTest``; ``"smc"`` and ``"pca"`` are as in :func:`parallel_analysis`. The
    correlations are flexpyctt's (NumPy) polychoric correlations.

    The parameters are estimated from the same data, so under a correct model the
    :math:`p`-value is conservative (tends to be large; in a check with 30 simulated 2PL data
    sets its mean was 0.66 and 1 of 30 was below 0.05). A small :math:`p` is evidence against
    unidimensionality; a large one is weak evidence for it.

    Parameters
    ----------
    fit : FitResult
        Single-group unidimensional fit with a ``Normal`` latent distribution.
    data : pandas.DataFrame or numpy.ndarray
        Responses on the fitted items (the fitting data); missing as NaN.
    key : jax.Array
        JAX PRNG key for the simulations.
    n_replications : int, default 100
        Number of simulated data sets :math:`B`.
    eigen_type : {"max", "smc", "pca"}, default "max"
        Diagonal of the matrix whose eigenvalues are used.
    zero_cell_correction : float, default 0.5
        Passed to :func:`flexpyctt.polychoric`.
    item_names : list of str, optional
        Column names when ``data`` is an array.

    Returns
    -------
    ModelParallelAnalysisResult

    Raises
    ------
    SpecificationError
        Unsupported model, invalid options or data.
    PrecisionError
        If JAX is not running in 64-bit mode.

    Warns
    -----
    FlexpyirtWarning
        If some simulated data sets had to be skipped.
    """
    check_x64()
    fit._require_unweighted("model_parallel_analysis()")
    fit._require_unidimensional_normal("model_parallel_analysis()")
    if eigen_type not in _MODEL_EIGEN_TYPES:
        raise SpecificationError(
            f"eigen_type must be one of {_MODEL_EIGEN_TYPES}, got {eigen_type!r}."
        )
    if not (isinstance(n_replications, (int, np.integer)) and n_replications >= 2):
        raise SpecificationError(
            f"n_replications must be an integer of at least 2, got {n_replications!r}."
        )
    y = coerce_responses(data, fit._model, item_names=item_names)
    names = [name for name, _ in fit._model.items]
    if len(names) < 3:
        raise SpecificationError("model_parallel_analysis() needs at least 3 items.")
    values = np.where(y >= 0, y, np.nan).astype(np.float64)
    observed_corr = _observed_correlation(values, names, "polychoric", zero_cell_correction)
    observed = _model_eigenvalues(observed_corr, eigen_type)
    missing = y < 0
    rows: list[np.ndarray] = []
    n_skipped = 0
    for k in jax.random.split(key, int(n_replications)):
        sim = fit._simulate_responses(k, missing)
        try:
            corr = _observed_correlation(sim, names, "polychoric", zero_cell_correction, quiet=True)
        except SpecificationError:
            n_skipped += 1
            continue
        rows.append(_model_eigenvalues(corr, eigen_type))
    if len(rows) < 2:
        raise SpecificationError("Fewer than two simulated data sets could be estimated.")
    if n_skipped:
        warnings.warn(
            f"{n_skipped} of {n_replications} simulated data sets were skipped (an item was "
            "constant among jointly observed respondents).",
            FlexpyirtWarning,
            stacklevel=2,
        )
    replications = np.vstack(rows)
    statistic = float(observed[1])
    p_value = float((1 + np.sum(replications[:, 1] >= statistic)) / (replications.shape[0] + 1))
    return ModelParallelAnalysisResult(
        statistic=statistic,
        p_value=p_value,
        observed=observed,
        replications=replications,
        eigen_type=eigen_type,
        n_skipped=n_skipped,
    )


# ----------------------------------------------------------------------------- ensemble
@dataclass(frozen=True)
class DimensionalityResult:
    """Result of :func:`dimensionality`.

    Attributes
    ----------
    number_of_dimensions : pandas.DataFrame
        Question (A), one row per method: ``method``, ``n_factors``, ``detail``, ``source``.
    unidimensionality : pandas.DataFrame
        Question (B), one row per method: ``method``, ``statistic``,
        ``threshold``, ``supports_unidimensionality`` (``None`` when unavailable), ``detail``,
        ``source``.
    n_factors_mode : tuple of int
        Most frequent suggested number(s) of factors (several on ties).
    n_factors_median : float
        Median suggested number of factors.
    n_agreeing : int
        Number of methods suggesting a mode value.
    n_supporting : int
        Number of available question-(B) methods supporting essential unidimensionality.
    n_unidimensionality_methods : int
        Number of available question-(B) methods.
    results : dict of str to object
        The individual results (``polychoric``, ``parallel_analysis``, ``velicer_map``,
        ``empirical_kaiser``, ``hull``, ``comparison_data``, ``exploratory_detect``,
        ``dimtest`` and, with ``fit``,
        ``model_parallel_analysis``, ``local_dependence``, ``m2``; ``None`` when unavailable).
    """

    number_of_dimensions: pd.DataFrame
    unidimensionality: pd.DataFrame
    n_factors_mode: tuple[int, ...]
    n_factors_median: float
    n_agreeing: int
    n_supporting: int
    n_unidimensionality_methods: int
    results: dict[str, object]

    def summary(self) -> str:
        """Return a plain-text report of every method and both consensus lines."""
        lines = ["Number of dimensions (question A)"]
        for row in self.number_of_dimensions.itertuples(index=False):
            lines.append(f"  {row.method:<32} {row.n_factors:>3}   {row.detail}")
        mode = ", ".join(str(k) for k in self.n_factors_mode)
        lines.append(
            f"  consensus: mode {mode} ({self.n_agreeing} of {len(self.number_of_dimensions)} "
            f"methods), median {self.n_factors_median:g}"
        )
        if len(self.unidimensionality):
            lines.append("Essential unidimensionality (question B)")
            for rec in self.unidimensionality.to_dict("records"):
                flag = rec["supports_unidimensionality"]
                verdict = "n/a" if flag is None else ("supports" if flag else "against")
                lines.append(f"  {rec['method']!s:<32} {verdict:<9} {rec['detail']}")
            lines.append(
                f"  consensus: {self.n_supporting} of {self.n_unidimensionality_methods} "
                "available methods support essential unidimensionality"
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        """Return :meth:`summary`."""
        return self.summary()


def dimensionality(
    data: pd.DataFrame | np.ndarray,
    *,
    key: jax.Array,
    fit: FitResult | None = None,
    n_replications: int = 100,
    zero_cell_correction: float = 0.5,
    alpha: float = 0.05,
    q3_star_threshold: float = 0.2,
    rmsea_threshold: float = 0.05,
    detect_threshold: float = 0.2,
    max_clusters: int = 4,
    item_names: list[str] | None = None,
) -> DimensionalityResult:
    r"""Run several dimensionality diagnostics and report them side by side.

    No single method is reliable on its own (Auerswald & Moshagen, 2019), so this reports every
    method and two separate consensus lines (DESIGN.md §16.8):

    (A) **How many dimensions?** :func:`parallel_analysis` (polychoric, PCA eigenvalues,
    permutation reference, 95th percentile), :func:`velicer_map` (revised, fourth power) and
    :func:`empirical_kaiser` and :func:`hull` (CAF, ULS; with at least 6 items), all on one
    polychoric matrix, and :func:`comparison_data` (Pearson; complete data only). The consensus
    is the mode (all values
    on a tie) and the median of the suggested numbers, with the number of methods agreeing.

    (B) **Is essential unidimensionality tenable?**

    * :func:`exploratory_detect` (complete data only): supports when the validation-half DETECT
      of the selected partition is below ``detect_threshold`` (0.2; Jang & Roussos, 2007;
      Zhang, 2007);
    * :func:`dimtest` (complete binary data only): supports when :math:`p\ge` ``alpha``
      (conservative: its type I error is below nominal in DESIGN.md §16.16's study).

    Only with ``fit`` (a single-group unidimensional model with a ``Normal`` latent
    distribution):

    * :func:`model_parallel_analysis`: supports when :math:`p\ge` ``alpha``;
    * :meth:`FitResult.local_dependence`: supports when the largest Q3* is below
      ``q3_star_threshold`` (0.2 is a common rule; Christensen, Makransky & Horton, 2017, show
      that critical values depend on the test and sample);
    * :meth:`FitResult.m2`: supports when RMSEA\ :sub:`2` :math:`\le` ``rmsea_threshold``
      (close fit, Maydeu-Olivares & Joe, 2014); it checks the whole model, not only its
      dimensionality, and is unavailable with missing data.

    The consensus is the number of available methods supporting essential unidimensionality.
    Methods are not weighted.

    Parameters
    ----------
    data : pandas.DataFrame or numpy.ndarray
        Respondents x items, ordered category codes; NaN is missing. With ``fit``, the fitting
        data (the fitted items, in any column order when a DataFrame).
    key : jax.Array
        JAX PRNG key; split for the parallel analyses, the Hull bound, the DETECT split, the
        comparison data and DIMTEST. The model-free methods receive a NumPy generator seeded
        from their key.
    fit : FitResult, optional
        Fitted unidimensional model for question (B).
    n_replications : int, default 100
        Replications of both parallel analyses.
    zero_cell_correction : float, default 0.5
        Passed to :func:`flexpyctt.polychoric`.
    alpha : float, default 0.05
        Level for :func:`model_parallel_analysis` and :func:`dimtest`.
    q3_star_threshold : float, default 0.2
        Largest acceptable Q3*.
    rmsea_threshold : float, default 0.05
        Largest acceptable RMSEA\ :sub:`2` of :meth:`FitResult.m2`.
    detect_threshold : float, default 0.2
        Largest DETECT (validation half) still read as essential unidimensionality.
    max_clusters : int, default 4
        Largest number of clusters searched by :func:`exploratory_detect`.
    item_names : list of str, optional
        Column names when ``data`` is an array and ``fit`` is given.

    Returns
    -------
    DimensionalityResult

    Raises
    ------
    SpecificationError
        Invalid options or data, or a ``fit`` that question (B) does not support.
    PrecisionError
        If JAX is not running in 64-bit mode.
    """
    check_x64()
    for name, value, lo, hi in (
        ("alpha", alpha, 0.0, 1.0),
        ("q3_star_threshold", q3_star_threshold, 0.0, 1.0),
        ("rmsea_threshold", rmsea_threshold, 0.0, 1.0),
        ("detect_threshold", detect_threshold, 0.0, np.inf),
    ):
        if not (lo < float(value) < hi):
            raise SpecificationError(f"{name} must be in ({lo:g}, {hi:g}), got {value!r}.")
    key_a, key_b, key_c, key_d, key_e, key_f = jax.random.split(key, 6)
    results: dict[str, object] = {}

    if fit is not None:
        fit._require_unweighted("dimensionality(fit=...)")
        fit._require_unidimensional_normal("dimensionality(fit=...)")
        y = coerce_responses(data, fit._model, item_names=item_names)
        names = [name for name, _ in fit._model.items]
        frame = pd.DataFrame(np.where(y >= 0, y, np.nan).astype(np.float64), columns=names)
    else:
        names, values = _coerce_ordinal(data)
        frame = pd.DataFrame(values, columns=names)

    pc = polychoric(frame, zero_cell_correction=zero_cell_correction)
    pa = parallel_analysis(
        frame,
        rng=_rng_from_key(key_a),
        n_replications=n_replications,
        zero_cell_correction=zero_cell_correction,
    )
    vm = velicer_map(pc)
    ekc = empirical_kaiser(pc)
    results.update(polychoric=pc, parallel_analysis=pa, velicer_map=vm, empirical_kaiser=ekc)
    number = pd.DataFrame(
        [
            {
                "method": "parallel analysis (PCA, 95%)",
                "n_factors": pa.n_factors,
                "detail": (
                    f"eigenvalues {np.round(pa.observed[:3], 3).tolist()} vs reference "
                    f"{np.round(pa.reference[:3], 3).tolist()}"
                ),
                "source": "Horn (1965); Glorfeld (1995); Garrido, Abad & Ponsoda (2013)",
            },
            {
                "method": "MAP (revised, 4th power)",
                "n_factors": vm.n_factors,
                "detail": f"minimum {np.nanmin(vm.statistic):.4g} at m = {vm.n_factors}",
                "source": "Velicer, Eaton & Fava (2000)",
            },
            {
                "method": "empirical Kaiser criterion",
                "n_factors": ekc.n_factors,
                "detail": f"n = {ekc.n_obs}",
                "source": "Braeken & van Assen (2017)",
            },
        ]
    )
    if len(names) >= 6:
        hl = hull(pc, rng=_rng_from_key(key_d))
        results["hull"] = hl
        on = hl.table[hl.table["on_hull"]]
        number = pd.concat(
            [
                number,
                pd.DataFrame(
                    [
                        {
                            "method": "Hull (CAF, ULS)",
                            "n_factors": hl.n_factors,
                            "detail": (
                                f"J = {hl.max_factors}, on hull k = {on['n_factors'].tolist()}"
                            ),
                            "source": "Lorenzo-Seva, Timmerman & Kiers (2011)",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    else:
        results["hull"] = None
    if not frame.isna().to_numpy().any():
        cd = comparison_data(frame, rng=_rng_from_key(key_e))
        results["comparison_data"] = cd
        number = pd.concat(
            [
                number,
                pd.DataFrame(
                    [
                        {
                            "method": "comparison data (Pearson)",
                            "n_factors": cd.n_factors,
                            "detail": (
                                "mean eigenvalue RMSE "
                                f"{np.round(cd.rmse.mean().to_numpy(), 4).tolist()}"
                            ),
                            "source": "Ruscio & Roche (2012)",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    else:
        results["comparison_data"] = None
    suggested = number["n_factors"].to_numpy()
    values_k, counts_k = np.unique(suggested, return_counts=True)
    mode = tuple(int(v) for v in values_k[counts_k == np.max(counts_k)])

    rows: list[dict[str, object]] = []
    n_items = len(names)
    if frame.isna().to_numpy().any() or n_items < 3:
        results["exploratory_detect"] = None
        rows.append(
            {
                "method": "DETECT (exploratory, validation half)",
                "statistic": float("nan"),
                "threshold": detect_threshold,
                "supports_unidimensionality": None,
                "detail": "not available: DETECT needs complete data",
                "source": "Zhang & Stout (1999); Jang & Roussos (2007)",
            }
        )
    else:
        xd = exploratory_detect(
            frame, max_clusters=min(max_clusters, n_items), rng=_rng_from_key(key_c)
        )
        results["exploratory_detect"] = xd
        best = xd.table[xd.table["n_clusters"] == xd.n_clusters].to_dict("records")[0]
        value = float(best["DETECT_validation"])
        rows.append(
            {
                "method": "DETECT (exploratory, validation half)",
                "statistic": value,
                "threshold": detect_threshold,
                "supports_unidimensionality": bool(value < detect_threshold),
                "detail": (
                    f"DETECT {value:.3f} for {xd.n_clusters} clusters ({best['cluster_sizes']}), "
                    f"ASSI {float(best['ASSI_validation']):.3f}, "
                    f"RATIO {float(best['RATIO_validation']):.3f}"
                ),
                "source": "Zhang & Stout (1999); Jang & Roussos (2007)",
            }
        )
    binary = (not frame.isna().to_numpy().any()) and bool(
        np.isin(frame.to_numpy(), (0.0, 1.0)).all()
    )
    if binary and n_items >= 4:
        dt = dimtest(frame, rng=_rng_from_key(key_f))
        results["dimtest"] = dt
        rows.append(
            {
                "method": "DIMTEST",
                "statistic": dt.p_value,
                "threshold": alpha,
                "supports_unidimensionality": bool(dt.p_value >= alpha),
                "detail": f"T = {dt.statistic:.3f}, p = {dt.p_value:.3g}, AT {dt.assessment_items}",
                "source": "Stout (1987); Stout, Froelich & Gao (2001)",
            }
        )
    else:
        results["dimtest"] = None
        rows.append(
            {
                "method": "DIMTEST",
                "statistic": float("nan"),
                "threshold": alpha,
                "supports_unidimensionality": None,
                "detail": "not available: DIMTEST needs complete binary data",
                "source": "Stout (1987); Stout, Froelich & Gao (2001)",
            }
        )
    if fit is not None:
        mpa = model_parallel_analysis(
            fit,
            frame,
            key=key_b,
            n_replications=n_replications,
            zero_cell_correction=zero_cell_correction,
        )
        ld = fit.local_dependence(frame)
        results.update(model_parallel_analysis=mpa, local_dependence=ld)
        rows.append(
            {
                "method": "model-based parallel analysis",
                "statistic": mpa.p_value,
                "threshold": alpha,
                "supports_unidimensionality": bool(mpa.p_value >= alpha),
                "detail": f"second eigenvalue {mpa.statistic:.4g}, p = {mpa.p_value:.3g}",
                "source": "Drasgow & Lissak (1983)",
            }
        )
        worst = ld.pairs.iloc[int(np.nanargmax(ld.pairs["q3_star"].to_numpy()))]
        rows.append(
            {
                "method": "largest Q3*",
                "statistic": ld.max_q3_star,
                "threshold": q3_star_threshold,
                "supports_unidimensionality": bool(ld.max_q3_star < q3_star_threshold),
                "detail": f"{ld.max_q3_star:.3f} ({worst['item_i']}, {worst['item_j']})",
                "source": "Yen (1984); Christensen, Makransky & Horton (2017)",
            }
        )
        try:
            m2 = fit.m2(frame)
        except SpecificationError as exc:
            results["m2"] = None
            rows.append(
                {
                    "method": "M2 RMSEA",
                    "statistic": float("nan"),
                    "threshold": rmsea_threshold,
                    "supports_unidimensionality": None,
                    "detail": f"not available: {exc}",
                    "source": "Maydeu-Olivares & Joe (2014)",
                }
            )
        else:
            results["m2"] = m2
            rmsea = float(m2["rmsea"].iloc[0])
            rows.append(
                {
                    "method": "M2 RMSEA",
                    "statistic": rmsea,
                    "threshold": rmsea_threshold,
                    # NaN when M2 has no degrees of freedom: not available, not a vote against.
                    "supports_unidimensionality": (
                        bool(rmsea <= rmsea_threshold) if np.isfinite(rmsea) else None
                    ),
                    "detail": (
                        f"RMSEA = {rmsea:.4f}, M2 p = {float(m2['p'].iloc[0]):.3g}"
                        if np.isfinite(rmsea)
                        else f"not available: M2 has df = {int(m2['df'].iloc[0])}"
                    ),
                    "source": "Maydeu-Olivares & Joe (2014)",
                }
            )
    unidim = pd.DataFrame(
        rows,
        columns=[
            "method",
            "statistic",
            "threshold",
            "supports_unidimensionality",
            "detail",
            "source",
        ],
    )
    available = [v for v in unidim["supports_unidimensionality"] if v is not None]
    return DimensionalityResult(
        number_of_dimensions=number,
        unidimensionality=unidim,
        n_factors_mode=mode,
        n_factors_median=float(np.median(suggested)),
        n_agreeing=int(np.max(counts_k)),
        n_supporting=int(sum(bool(v) for v in available)),
        n_unidimensionality_methods=len(available),
        results=results,
    )
