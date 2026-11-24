"""Number-of-dimensions diagnostics (Python shell layer, DESIGN.md §16.8, §16.10).

Public entry points :func:`parallel_analysis`, :func:`velicer_map` and :func:`empirical_kaiser`,
each returning a frozen result object. All three answer question (A) of DESIGN.md §16.8, "how many
dimensions?"; they are diagnostics and never feed a fit.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.stats import norm

from flexpyirt import _comparison_data, _detect, _dimtest, _efa
from flexpyirt._compile import coerce_responses
from flexpyirt._precision import check_x64
from flexpyirt.correlation import (
    PolychoricResult,
    _coerce_ordinal,
    polychoric,
    smooth_correlation,
)
from flexpyirt.errors import (
    ConvergenceWarning,
    FlexpyirtWarning,
    HeywoodWarning,
    NotPositiveDefiniteWarning,
    SpecificationError,
)

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

EigenType = Literal["pca", "smc"]
ReferenceKind = Literal["permutation", "normal"]
CorrelationKind = Literal["polychoric", "pearson"]

_EIGEN_TYPES: tuple[str, ...] = ("pca", "smc")
_REFERENCES: tuple[str, ...] = ("permutation", "normal")
_CORRELATIONS: tuple[str, ...] = ("polychoric", "pearson")


# ----------------------------------------------------------------------------- helpers
def _as_correlation(
    correlation: pd.DataFrame | np.ndarray | PolychoricResult,
) -> tuple[list[str], np.ndarray]:
    """Return item names and a validated symmetric unit-diagonal ``(p, p)`` matrix."""
    if isinstance(correlation, PolychoricResult):
        frame = correlation.correlation
        names, mat = [str(c) for c in frame.columns], frame.to_numpy(dtype=np.float64)
    elif isinstance(correlation, pd.DataFrame):
        names = [str(c) for c in correlation.columns]
        mat = correlation.to_numpy(dtype=np.float64)
    else:
        mat = np.asarray(correlation, dtype=np.float64)
        names = [f"item{j + 1}" for j in range(mat.shape[-1])] if mat.ndim == 2 else []
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or mat.shape[0] < 3:
        raise SpecificationError(
            f"correlation must be a square matrix with at least 3 variables, got shape {mat.shape}."
        )
    if not np.isfinite(mat).all():
        raise SpecificationError("correlation must be finite.")
    if not np.allclose(mat, mat.T, rtol=0.0, atol=1e-10):
        raise SpecificationError("correlation must be symmetric.")
    if not np.allclose(np.diag(mat), 1.0, rtol=0.0, atol=1e-10):
        raise SpecificationError("correlation must have a unit diagonal.")
    return names, (mat + mat.T) / 2.0


def _descending_eigenvalues(mat: np.ndarray) -> np.ndarray:
    return np.linalg.eigvalsh(mat)[::-1].copy()


def _reduced(mat: np.ndarray) -> np.ndarray:
    r"""Reduced correlation matrix :math:`R-\operatorname{diag}(1-h_j)` with SMC communalities.

    :math:`h_j=1-1/(R^{-1})_{jj}` (squared multiple correlations; Humphreys & Ilgen, 1969).
    """
    smc = 1.0 - 1.0 / np.diag(np.linalg.inv(mat))
    out = mat.copy()
    np.fill_diagonal(out, smc)
    return out


def _eigenvalues(mat: np.ndarray, eigen_type: str) -> np.ndarray:
    return _descending_eigenvalues(mat if eigen_type == "pca" else _reduced(mat))


def _count_leading(observed: np.ndarray, reference: np.ndarray) -> int:
    """Count the leading ``observed`` values strictly above ``reference`` (up to the first miss)."""
    above = observed > reference
    return int(np.argmin(above)) if not above.all() else int(above.size)


# ----------------------------------------------------------------------------- EKC
@dataclass(frozen=True)
class EmpiricalKaiserResult:
    """Result of :func:`empirical_kaiser`.

    Attributes
    ----------
    n_factors : int
        Number of leading eigenvalues above their reference values.
    eigenvalues : numpy.ndarray
        Eigenvalues of the correlation matrix, descending.
    reference : numpy.ndarray
        EKC reference eigenvalues :math:`l_j`.
    n_obs : int
        Sample size used.
    """

    n_factors: int
    eigenvalues: np.ndarray
    reference: np.ndarray
    n_obs: int


def _default_n_obs(correlation: object, n_obs: int | None) -> int:
    if n_obs is not None:
        if not (isinstance(n_obs, (int, np.integer)) and n_obs >= 1):
            raise SpecificationError(f"n_obs must be a positive integer, got {n_obs!r}.")
        return int(n_obs)
    if isinstance(correlation, PolychoricResult):
        counts = correlation.n_obs.to_numpy()
        return int(counts[np.triu_indices_from(counts, k=1)].min())
    raise SpecificationError("n_obs is required unless correlation is a PolychoricResult.")


def empirical_kaiser(
    correlation: pd.DataFrame | np.ndarray | PolychoricResult,
    *,
    n_obs: int | None = None,
) -> EmpiricalKaiserResult:
    r"""Empirical Kaiser criterion (Braeken & van Assen, 2017).

    The :math:`j`-th eigenvalue :math:`\lambda_j` of the :math:`p\times p` correlation matrix is
    compared with

    .. math::
        l_j=\max\Big\{\Big(1+\sqrt{p/n}\Big)^2\,
        \frac{p-\sum_{i<j}\lambda_i}{p-j+1},\;1\Big\},

    the Marčenko–Pastur upper edge for the variance left after the first :math:`j-1`
    eigenvalues, bounded below by Kaiser's 1. The number of factors is the number of leading
    eigenvalues with :math:`\lambda_j>l_j`.

    Parameters
    ----------
    correlation : pandas.DataFrame, numpy.ndarray or PolychoricResult
        Correlation matrix (at least 3 variables).
    n_obs : int, optional
        Sample size :math:`n`. Required unless ``correlation`` is a :class:`PolychoricResult`,
        whose smallest pairwise number of observations is then used.

    Returns
    -------
    EmpiricalKaiserResult
    """
    _, mat = _as_correlation(correlation)
    n = _default_n_obs(correlation, n_obs)
    values = _descending_eigenvalues(mat)
    p = values.size
    factor = (1.0 + np.sqrt(p / n)) ** 2
    before = np.concatenate([[0.0], np.cumsum(values)[:-1]])
    reference = np.maximum(factor * (p - before) / (p - np.arange(p)), 1.0)
    return EmpiricalKaiserResult(
        n_factors=_count_leading(values, reference),
        eigenvalues=values,
        reference=reference,
        n_obs=n,
    )


# ----------------------------------------------------------------------------- MAP
@dataclass(frozen=True)
class MAPResult:
    r"""Result of :func:`velicer_map`.

    Attributes
    ----------
    n_factors : int
        The number of components :math:`m` minimizing the average partial correlation.
    statistic : numpy.ndarray
        :math:`f_m` for :math:`m=0,\dots,p-1` (index = number of components partialled out);
        NaN from the first :math:`m` at which some partial variance is at most :math:`10^{-10}`
        (the partial correlations are then undefined; EFA.dimensions stops there too).
    power : int
        2 (Velicer, 1976) or 4 (Velicer, Eaton & Fava, 2000).
    """

    n_factors: int
    statistic: np.ndarray
    power: int


def velicer_map(
    correlation: pd.DataFrame | np.ndarray | PolychoricResult,
    *,
    power: Literal[2, 4] = 4,
) -> MAPResult:
    r"""Velicer's minimum average partial (MAP) test.

    With :math:`A_m` the loadings of the first :math:`m` principal components
    (:math:`A_m=V_m\Lambda_m^{1/2}`), the partial covariance is :math:`C_m=R-A_mA_m^\top` and
    the partial correlations :math:`R^*_m=D_m^{-1/2}C_mD_m^{-1/2}` (:math:`D_m=\operatorname{diag}
    C_m`). The statistic

    .. math::
        f_m=\frac{1}{p(p-1)}\sum_{i\ne j}\big(r^*_{m,ij}\big)^{q},\qquad m=0,\dots,p-1,

    (:math:`R^*_0=R`) is minimized over :math:`m`; :math:`q=2` is the original test (Velicer,
    1976), :math:`q=4` the revised one (Velicer, Eaton & Fava, 2000; O'Connor, 2000). psych's
    ``vss()`` reports :math:`q=2` for :math:`m\ge1` only. EFA.dimensions' ``MAP()`` agrees for
    :math:`q=2` but computes its "4th power" statistic as
    :math:`(\operatorname{tr}(R_m^{*4})-p)/(p(p-1))`, a matrix power, not the elementwise power.

    Parameters
    ----------
    correlation : pandas.DataFrame, numpy.ndarray or PolychoricResult
        Correlation matrix (at least 3 variables).
    power : {2, 4}, default 4
        Power :math:`q` of the partial correlations.

    Returns
    -------
    MAPResult
    """
    if power not in (2, 4):
        raise SpecificationError(f"power must be 2 or 4, got {power!r}.")
    _, mat = _as_correlation(correlation)
    p = mat.shape[0]
    values, vectors = np.linalg.eigh(mat)
    order = np.argsort(values)[::-1]
    loadings = vectors[:, order] * np.sqrt(np.clip(values[order], 0.0, None))
    off = ~np.eye(p, dtype=bool)
    stats = np.full(p, np.nan)
    for m in range(p):
        partial = mat - loadings[:, :m] @ loadings[:, :m].T
        if np.diag(partial).min() <= 1e-10:
            break  # a variable is fully explained; later partial correlations are undefined
        scale = 1.0 / np.sqrt(np.diag(partial))
        rstar = partial * scale[:, None] * scale[None, :]
        stats[m] = np.sum(rstar[off] ** power) / (p * (p - 1))
    return MAPResult(n_factors=int(np.nanargmin(stats)), statistic=stats, power=int(power))


# ----------------------------------------------------------------------------- parallel analysis
@dataclass(frozen=True)
class ParallelAnalysisResult:
    """Result of :func:`parallel_analysis`.

    Attributes
    ----------
    n_factors : int
        Number of leading observed eigenvalues above the reference.
    observed : numpy.ndarray
        Observed eigenvalues (descending) of the ``eigen_type`` matrix.
    reference : numpy.ndarray
        Per position, the ``quantile`` (or the mean when ``quantile=None``) of the reference
        eigenvalues.
    reference_mean : numpy.ndarray
        Per position, the mean of the reference eigenvalues.
    replications : numpy.ndarray
        ``(n_replications, p)`` reference eigenvalues (each row descending).
    correlation : pandas.DataFrame
        The observed correlation matrix.
    eigen_type, reference_kind, correlation_kind : str
        The settings used.
    quantile : float or None
        The quantile used (``None`` = mean).
    n_skipped : int
        Permutation replications whose correlations could not be estimated (see Notes of
        :func:`parallel_analysis`); they are excluded.
    """

    n_factors: int
    observed: np.ndarray
    reference: np.ndarray
    reference_mean: np.ndarray
    replications: np.ndarray
    correlation: pd.DataFrame
    eigen_type: str
    reference_kind: str
    correlation_kind: str
    quantile: float | None
    n_skipped: int


def _pearson(values: np.ndarray) -> np.ndarray:
    """Pairwise-complete Pearson correlation of a float array with NaN."""
    frame = pd.DataFrame(values)
    mat = frame.corr(method="pearson", min_periods=2).to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(mat).all():
        raise SpecificationError(
            "A Pearson correlation is undefined (an item is constant among jointly observed "
            "respondents)."
        )
    np.fill_diagonal(mat, 1.0)
    return mat


def _observed_correlation(
    values: np.ndarray,
    names: list[str],
    correlation_kind: str,
    zero_cell_correction: float,
    *,
    quiet: bool = False,
) -> np.ndarray:
    """Smoothed polychoric or Pearson correlation; ``quiet`` silences flexpyirt warnings."""
    with warnings.catch_warnings():
        if quiet:
            warnings.simplefilter("ignore", FlexpyirtWarning)
        if correlation_kind == "polychoric":
            res = polychoric(
                pd.DataFrame(values, columns=names), zero_cell_correction=zero_cell_correction
            )
            return res.correlation.to_numpy()
        mat, smoothed = smooth_correlation(_pearson(values))
        if smoothed:
            warnings.warn(
                "The Pearson matrix was not positive definite and was smoothed.",
                NotPositiveDefiniteWarning,
                stacklevel=3,
            )
        return mat


def _permutations(key: jax.Array, n_replications: int, p: int, n_obs: int) -> np.ndarray:
    """``(n_replications, p, n_obs)`` independent permutations of the rows, one per item."""

    def one(k: jax.Array) -> jax.Array:
        return jax.vmap(lambda kk: jax.random.permutation(kk, n_obs))(jax.random.split(k, p))

    return np.asarray(jax.vmap(one)(jax.random.split(key, n_replications)))


def _normal_reference(
    key: jax.Array, n_obs: int, p: int, n_replications: int, eigen_type: str, chunk: int
) -> np.ndarray:
    """Eigenvalues of Pearson correlations of i.i.d. standard normal ``(n_obs, p)`` data."""

    def one(k: jax.Array) -> jax.Array:
        x = jax.random.normal(k, (n_obs, p), dtype=jnp.float64)
        x = x - x.mean(axis=0)
        cov = x.T @ x
        sd = jnp.sqrt(jnp.diag(cov))
        corr = cov / sd[:, None] / sd[None, :]
        if eigen_type == "smc":
            smc = 1.0 - 1.0 / jnp.diag(jnp.linalg.inv(corr))
            corr = corr - jnp.diag(jnp.diag(corr)) + jnp.diag(smc)
        values: jax.Array = jnp.linalg.eigvalsh(corr)
        return values[::-1]

    batched = jax.jit(jax.vmap(one))
    keys = jax.random.split(key, n_replications)
    out = [np.asarray(batched(keys[s : s + chunk])) for s in range(0, n_replications, chunk)]
    return np.concatenate(out, axis=0)


def parallel_analysis(
    data: pd.DataFrame | np.ndarray,
    *,
    key: jax.Array,
    correlation: CorrelationKind = "polychoric",
    eigen_type: EigenType = "pca",
    reference: ReferenceKind = "permutation",
    n_replications: int = 100,
    quantile: float | None = 0.95,
    zero_cell_correction: float = 0.5,
) -> ParallelAnalysisResult:
    r"""Horn's (1965) parallel analysis.

    The eigenvalues :math:`\lambda_1\ge\dots\ge\lambda_p` of the observed correlation matrix
    (principal components, ``eigen_type="pca"``, or the reduced matrix with squared multiple
    correlations on the diagonal, ``"smc"``) are compared with those of reference data that share
    the sample size and the number of variables but have no common factors. The number of factors
    is the number of leading :math:`\lambda_j` above the reference's ``quantile`` at position
    :math:`j` (Glorfeld, 1995; the mean when ``quantile=None``, as in Horn, 1965).

    Two references are available:

    * ``"permutation"`` (default): every item's responses are permuted independently, which keeps
      each item's marginal distribution (and missingness) and removes the associations; the
      correlation (polychoric or Pearson, as for the data) is recomputed for each replication.
      For ordinal data this is the reference studied by Garrido, Abad & Ponsoda (2013).
    * ``"normal"``: Pearson correlations of i.i.d. standard normal data of the same size
      (Horn, 1965). With incomplete data the number of rows is used.

    Parameters
    ----------
    data : pandas.DataFrame or numpy.ndarray
        Respondents x items, ordered category codes (or continuous scores for
        ``correlation="pearson"``); NaN is missing.
    key : jax.Array
        JAX PRNG key (``jax.random.key(seed)``) for the reference data.
    correlation : {"polychoric", "pearson"}, default "polychoric"
        Correlation of the data (and of permutation replications). Polychoric correlations use
        :func:`~flexpyirt.polychoric` with smoothing; Pearson correlations are pairwise complete
        and smoothed the same way when not positive definite.
    eigen_type : {"pca", "smc"}, default "pca"
        Eigenvalues of the correlation matrix, or of the reduced matrix (SMC communalities).
    reference : {"permutation", "normal"}, default "permutation"
        Reference data, see above.
    n_replications : int, default 100
        Number of reference data sets.
    quantile : float or None, default 0.95
        Quantile of the reference eigenvalues in (0, 1); ``None`` uses the mean.
    zero_cell_correction : float, default 0.5
        Passed to :func:`~flexpyirt.polychoric` for the data and every replication.

    Returns
    -------
    ParallelAnalysisResult

    Raises
    ------
    SpecificationError
        Invalid options or data (see :func:`~flexpyirt.polychoric`), fewer than 3 items.
    PrecisionError
        If JAX is not running in 64-bit mode.

    Warns
    -----
    FlexpyirtWarning
        If some permutation replications had to be skipped.

    Notes
    -----
    A permutation replication is skipped (``n_skipped``) when, with missing data, a pair of
    permuted items has no jointly observed respondent or one of them is constant among those.
    Warnings raised while estimating replication correlations (boundary pairs, smoothing) are
    not repeated.
    """
    check_x64()
    if correlation not in _CORRELATIONS:
        raise SpecificationError(
            f"correlation must be one of {_CORRELATIONS}, got {correlation!r}."
        )
    if eigen_type not in _EIGEN_TYPES:
        raise SpecificationError(f"eigen_type must be one of {_EIGEN_TYPES}, got {eigen_type!r}.")
    if reference not in _REFERENCES:
        raise SpecificationError(f"reference must be one of {_REFERENCES}, got {reference!r}.")
    if not (isinstance(n_replications, (int, np.integer)) and n_replications >= 2):
        raise SpecificationError(
            f"n_replications must be an integer of at least 2, got {n_replications!r}."
        )
    if quantile is not None and not (0.0 < float(quantile) < 1.0):
        raise SpecificationError(f"quantile must be in (0, 1) or None, got {quantile!r}.")
    names, values = _coerce_ordinal(data)
    n_obs, p = values.shape
    if p < 3:
        raise SpecificationError("parallel_analysis() needs at least 3 items.")

    observed_corr = _observed_correlation(values, names, correlation, zero_cell_correction)
    observed = _eigenvalues(observed_corr, eigen_type)

    n_skipped = 0
    if reference == "normal":
        replications = _normal_reference(
            key, n_obs, p, int(n_replications), eigen_type, chunk=max(1, int(2e7 // (n_obs * p)))
        )
    else:
        orders = _permutations(key, int(n_replications), p, n_obs)
        rows: list[np.ndarray] = []
        for r in range(int(n_replications)):
            permuted = np.take_along_axis(values, orders[r].T, axis=0)
            try:
                corr = _observed_correlation(
                    permuted, names, correlation, zero_cell_correction, quiet=True
                )
            except SpecificationError:
                n_skipped += 1
                continue
            rows.append(_eigenvalues(corr, eigen_type))
        if len(rows) < 2:
            raise SpecificationError(
                "Fewer than two permutation replications could be estimated; with this much "
                'missing data use reference="normal".'
            )
        replications = np.vstack(rows)
        if n_skipped:
            warnings.warn(
                f"{n_skipped} of {n_replications} permutation replications were skipped "
                "(a permuted pair had no usable jointly observed responses).",
                FlexpyirtWarning,
                stacklevel=2,
            )

    mean = replications.mean(axis=0)
    ref = mean if quantile is None else np.quantile(replications, float(quantile), axis=0)
    return ParallelAnalysisResult(
        n_factors=_count_leading(observed, ref),
        observed=observed,
        reference=ref,
        reference_mean=mean,
        replications=replications,
        correlation=pd.DataFrame(observed_corr, index=names, columns=names),
        eigen_type=eigen_type,
        reference_kind=reference,
        correlation_kind=correlation,
        quantile=None if quantile is None else float(quantile),
        n_skipped=n_skipped,
    )


# ----------------------------------------------------------------------------- local dependence
@dataclass(frozen=True)
class LocalDependenceResult:
    """Result of :meth:`flexpyirt.FitResult.local_dependence`.

    Attributes
    ----------
    pairs : pandas.DataFrame
        One row per item pair: ``item_i``, ``item_j``, ``n`` (respondents answering both),
        ``q3``, ``q3_star``, ``x2``, ``g2``, ``df``, ``p_x2``, ``p_g2``, ``cramer_v``.
    q3 : pandas.DataFrame
        Item x item Q3 matrix (unit diagonal).
    q3_mean : float
        Mean off-diagonal Q3 (subtracted in ``q3_star``).
    """

    pairs: pd.DataFrame
    q3: pd.DataFrame
    q3_mean: float

    @property
    def max_q3_star(self) -> float:
        """Largest Q3* over pairs (Christensen, Makransky & Horton, 2017)."""
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
    in ltm's ``unidimTest``; ``"smc"`` and ``"pca"`` are as in :func:`parallel_analysis`.

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
        Passed to :func:`~flexpyirt.polychoric`.
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
        comparison data and DIMTEST.
    fit : FitResult, optional
        Fitted unidimensional model for question (B).
    n_replications : int, default 100
        Replications of both parallel analyses.
    zero_cell_correction : float, default 0.5
        Passed to :func:`~flexpyirt.polychoric`.
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
        key=key_a,
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
        hl = hull(pc, key=key_d)
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
        cd = comparison_data(frame, key=key_e)
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
        xd = exploratory_detect(frame, max_clusters=min(max_clusters, n_items), key=key_c)
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
        dt = dimtest(frame, key=key_f)
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
        worst = ld.pairs.iloc[int(np.argmax(ld.pairs["q3_star"].to_numpy()))]
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
                    "supports_unidimensionality": bool(rmsea <= rmsea_threshold),
                    "detail": f"RMSEA = {rmsea:.4f}, M2 p = {float(m2['p'].iloc[0]):.3g}",
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


# ----------------------------------------------------------------------------- DETECT
def _complete_scores(data: pd.DataFrame | np.ndarray, what: str) -> tuple[list[str], np.ndarray]:
    names, values = _coerce_ordinal(data)
    if np.isnan(values).any():
        raise SpecificationError(
            f"{what} needs complete data: the total and rest scores are undefined with missing "
            "responses."
        )
    if values.shape[1] < 3:
        raise SpecificationError(f"{what} needs at least 3 items.")
    return names, values


def _index_frame(indices: dict[str, tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [indices[name] for name in _detect.INDEX_NAMES],
        index=list(_detect.INDEX_NAMES),
        columns=["unweighted", "weighted"],
    )


@dataclass(frozen=True)
class DetectResult:
    """Result of :func:`detect`.

    Attributes
    ----------
    indices : pandas.DataFrame
        Rows ``DETECT``, ``ASSI``, ``RATIO``, ``MADCOV100``, ``MCOV100``; columns
        ``unweighted`` and ``weighted`` (pairs weighted by the square root of their sample size;
        identical with complete data).
    conditional_covariance : pandas.DataFrame
        Item x item conditional covariances (zero diagonal).
    clusters : pandas.Series
        The item partition used.
    n_obs : int
        Number of respondents.
    """

    indices: pd.DataFrame
    conditional_covariance: pd.DataFrame
    clusters: pd.Series
    n_obs: int

    @property
    def detect(self) -> float:
        """DETECT index (unweighted)."""
        return float(self.indices["unweighted"].to_numpy()[0])

    @property
    def assi(self) -> float:
        """ASSI (unweighted)."""
        return float(self.indices["unweighted"].to_numpy()[1])

    @property
    def ratio(self) -> float:
        """RATIO (unweighted)."""
        return float(self.indices["unweighted"].to_numpy()[2])


def _coerce_clusters(clusters: object, names: list[str]) -> np.ndarray:
    if isinstance(clusters, (pd.Series, Mapping)):
        mapping = dict(clusters.items())
        missing = [n for n in names if n not in mapping]
        extra = [k for k in mapping if str(k) not in names]
        if missing or extra:
            raise SpecificationError(
                f"clusters must name every item exactly once; missing {missing}, unknown {extra}."
            )
        labels = [mapping[n] for n in names]
    else:
        labels = list(clusters)  # type: ignore[call-overload]
        if len(labels) != len(names):
            raise SpecificationError(f"clusters has {len(labels)} labels for {len(names)} items.")
    _, codes = np.unique(np.asarray(labels, dtype=object).astype(str), return_inverse=True)
    return np.asarray(codes, dtype=np.int64)


def detect(
    data: pd.DataFrame | np.ndarray,
    clusters: pd.Series | Mapping[str, object] | Sequence[object] | np.ndarray,
) -> DetectResult:
    r"""Confirmatory DETECT for a given partition of the items (Zhang & Stout, 1999).

    The conditional covariance of each pair is the average of the covariances given the total
    score and given the rest score (:func:`flexpyirt._detect.conditional_covariances`), and

    .. math::
        \mathrm{DETECT}=\frac{100}{\binom J2}\sum_{i<j}\delta_{ij}\,\widehat{\mathrm{ccov}}_{ij},
        \qquad \delta_{ij}=\begin{cases}1&\text{same cluster}\\-1&\text{otherwise,}\end{cases}

    together with ASSI and RATIO (Zhang, 2007), as sirt's ``conf.detect(...,
    use_sum_score = TRUE)``. Under essential unidimensionality the conditional covariances are
    near (slightly below) zero, so DETECT is near zero or negative for any partition. A common
    classification (Jang & Roussos, 2007; Zhang, 2007) is DETECT < 0.2 essentially
    unidimensional, 0.2–0.4 weak, 0.4–1 moderate, > 1 strong multidimensionality, and ASSI > 0.25
    or RATIO > 0.36 an essential deviation from unidimensionality for the partition.

    Parameters
    ----------
    data : pandas.DataFrame or numpy.ndarray
        Respondents x items, complete numeric item scores.
    clusters : pandas.Series, dict, list or numpy.ndarray
        Cluster label of every item: a mapping item name -> label, or labels in column order.

    Returns
    -------
    DetectResult

    Raises
    ------
    SpecificationError
        Missing responses, fewer than 3 items, or clusters that do not match the items.
    """
    names, values = _complete_scores(data, "detect()")
    codes = _coerce_clusters(clusters, names)
    ccov = _detect.conditional_covariances(values)
    indices = _detect.detect_indices(ccov, codes)
    return DetectResult(
        indices=_index_frame(indices),
        conditional_covariance=pd.DataFrame(ccov, index=names, columns=names),
        clusters=pd.Series(codes + 1, index=names, name="cluster"),
        n_obs=int(values.shape[0]),
    )


@dataclass(frozen=True)
class ExploratoryDetectResult:
    """Result of :func:`exploratory_detect`.

    Attributes
    ----------
    table : pandas.DataFrame
        One row per number of clusters ``k``: ``n_clusters``, ``cluster_sizes`` and the
        unweighted ``DETECT``, ``ASSI``, ``RATIO`` on the estimation half (``*_estimation``) and
        the validation half (``*_validation``).
    clusters : dict of int to pandas.Series
        The partition into ``k`` clusters (labels numbered by first appearance).
    n_clusters : int
        ``k`` with the largest estimation-half DETECT (sirt's choice).
    estimation_rows, validation_rows : numpy.ndarray
        Row positions of the two halves.
    """

    table: pd.DataFrame
    clusters: dict[int, pd.Series]
    n_clusters: int
    estimation_rows: np.ndarray
    validation_rows: np.ndarray


def exploratory_detect(
    data: pd.DataFrame | np.ndarray,
    *,
    max_clusters: int = 4,
    key: jax.Array | None = None,
    estimation_rows: np.ndarray | list[int] | None = None,
) -> ExploratoryDetectResult:
    r"""Exploratory DETECT with a cross-validated partition search.

    The rows are split into an estimation half and a validation half. On the estimation half
    the items are clustered by Ward's method on :math:`\max(\mathrm{ccov})-\mathrm{ccov}`
    (sirt's ``expl.detect``); every partition into :math:`k=2,\dots,` ``max_clusters`` clusters
    is then evaluated by DETECT on both halves (Zhang & Stout, 1999, recommend judging the
    selected partition on data not used to select it). The reported ``n_clusters`` maximizes
    the estimation-half DETECT, as in sirt; compare it with the validation columns.

    Parameters
    ----------
    data : pandas.DataFrame or numpy.ndarray
        Respondents x items, complete numeric item scores.
    max_clusters : int, default 4
        Largest number of clusters (at least 2, at most the number of items).
    key : jax.Array, optional
        JAX PRNG key for the random split into halves (``floor(N/2)`` estimation rows).
    estimation_rows : array-like of int, optional
        Row positions of the estimation half instead of a random split (the rest validate). An
        empty validation half is not allowed. Exactly one of ``key`` and ``estimation_rows`` is
        needed.

    Returns
    -------
    ExploratoryDetectResult

    Raises
    ------
    SpecificationError
        Invalid options or data.
    """
    names, values = _complete_scores(data, "exploratory_detect()")
    n_obs, n_items = values.shape
    if not (isinstance(max_clusters, (int, np.integer)) and 2 <= max_clusters <= n_items):
        raise SpecificationError(
            f"max_clusters must be an integer in [2, {n_items}], got {max_clusters!r}."
        )
    if (key is None) == (estimation_rows is None):
        raise SpecificationError("Pass exactly one of key= and estimation_rows=.")
    if estimation_rows is None:
        assert key is not None
        perm = np.asarray(jax.random.permutation(key, n_obs))
        est = np.sort(perm[: n_obs // 2])
    else:
        est = np.unique(np.asarray(estimation_rows, dtype=np.int64))
        if est.size == 0 or est.min() < 0 or est.max() >= n_obs:
            raise SpecificationError("estimation_rows must be row positions within the data.")
    val = np.setdiff1d(np.arange(n_obs), est)
    if val.size == 0:
        raise SpecificationError("The validation half is empty.")
    ccov_est = _detect.conditional_covariances(values[est])
    ccov_val = _detect.conditional_covariances(values[val])
    ks = list(range(2, int(max_clusters) + 1))
    partitions = _detect.ward_clusters(ccov_est, ks)
    rows = []
    clusters: dict[int, pd.Series] = {}
    for k in ks:
        labels = partitions[k]
        clusters[k] = pd.Series(labels, index=names, name="cluster")
        on_est = _detect.detect_indices(ccov_est, labels)
        on_val = _detect.detect_indices(ccov_val, labels)
        rows.append(
            {
                "n_clusters": k,
                "cluster_sizes": "-".join(str(int(s)) for s in np.bincount(labels)[1:]),
                **{f"{n}_estimation": on_est[n][0] for n in ("DETECT", "ASSI", "RATIO")},
                **{f"{n}_validation": on_val[n][0] for n in ("DETECT", "ASSI", "RATIO")},
            }
        )
    table = pd.DataFrame(rows)
    best = int(table["n_clusters"].to_numpy()[np.argmax(table["DETECT_estimation"].to_numpy())])
    return ExploratoryDetectResult(
        table=table,
        clusters=clusters,
        n_clusters=best,
        estimation_rows=est,
        validation_rows=val,
    )


# ----------------------------------------------------------------------------- EFA and Hull
EFAMethod = Literal["uls", "ml"]
HullFitIndex = Literal["caf", "cfi", "rmsea"]
_EFA_METHODS: tuple[str, ...] = ("uls", "ml")
_HULL_FIT: tuple[str, ...] = ("caf", "cfi", "rmsea")


@dataclass(frozen=True)
class EFAResult:
    r"""Result of :func:`efa`.

    Attributes
    ----------
    loadings : pandas.DataFrame
        Unrotated loadings (items x factors), each column signed to a non-negative sum.
    uniquenesses : pandas.Series
        Estimated uniquenesses :math:`\psi_i`.
    communalities : pandas.Series
        :math:`h_i^2=\sum_f\lambda_{if}^2`.
    method : str
        ``"uls"`` or ``"ml"``.
    converged : bool
        Whether the optimizer met its criterion.
    heywood : bool
        Whether some uniqueness is at the lower bound 0.005.
    objective : float
        Minimized objective.
    fit : dict of str to float
        ``chi2`` (Bartlett-corrected), ``df``, ``p``, ``caf``, ``cfi``, ``rmsea`` when ``n_obs``
        was given; otherwise only ``caf`` and ``df``.
    """

    loadings: pd.DataFrame
    uniquenesses: pd.Series
    communalities: pd.Series
    method: str
    converged: bool
    heywood: bool
    objective: float
    fit: dict[str, float]


def efa(
    correlation: pd.DataFrame | np.ndarray | PolychoricResult,
    *,
    n_factors: int,
    method: EFAMethod = "uls",
    n_obs: int | None = None,
) -> EFAResult:
    r"""Unrotated exploratory factor analysis of a correlation matrix.

    The uniquenesses :math:`\Psi` minimize the ULS (minres) criterion
    :math:`\tfrac12\lVert R-\Psi-\Lambda\Lambda^\top\rVert_F^2` (Harman & Jones, 1966) or the
    ML discrepancy (Jöreskog, 1967), with :math:`\Lambda` from the leading eigenvectors of
    :math:`R-\Psi` (ULS) or :math:`\Psi^{-1/2}R\Psi^{-1/2}` (ML), by L-BFGS-B with analytic
    gradients from :math:`\psi_i=1-\mathrm{SMC}_i`, bounded to :math:`[0.005,\max(1,
    \mathrm{SMC})]`. This is psych's ``fa(fm = "minres" / "ml", rotate = "none")`` and EFAtools'
    ``EFA(method = "ULS" / "ML", rotation = "none")``. Fit indices follow EFAtools (CAF,
    Bartlett-corrected :math:`\chi^2`, CFI, RMSEA; see :func:`flexpyirt._efa.fit_indices`).

    Parameters
    ----------
    correlation : pandas.DataFrame, numpy.ndarray or PolychoricResult
        Positive definite correlation matrix.
    n_factors : int
        Number of factors (at least 1, with positive degrees of freedom).
    method : {"uls", "ml"}, default "uls"
        Estimator.
    n_obs : int, optional
        Sample size for :math:`\chi^2`, CFI and RMSEA (default for a ``PolychoricResult``: its
        smallest pairwise :math:`n`).

    Returns
    -------
    EFAResult

    Raises
    ------
    SpecificationError
        Invalid options or a matrix that is not positive definite.

    Warns
    -----
    ConvergenceWarning
        If the optimizer did not converge.
    HeywoodWarning
        If a uniqueness is at its lower bound.
    """
    names, mat = _as_correlation(correlation)
    if method not in _EFA_METHODS:
        raise SpecificationError(f"method must be one of {_EFA_METHODS}, got {method!r}.")
    p = mat.shape[0]
    max_k = _efa.max_identified_factors(p)
    if not (isinstance(n_factors, (int, np.integer)) and 1 <= n_factors <= max_k):
        raise SpecificationError(
            f"n_factors must be an integer in [1, {max_k}] for {p} variables (positive "
            f"degrees of freedom), got {n_factors!r}."
        )
    if np.linalg.eigvalsh(mat).min() <= 0:
        raise SpecificationError("efa() needs a positive definite correlation matrix.")
    n = None
    if n_obs is not None or isinstance(correlation, PolychoricResult):
        n = _default_n_obs(correlation, n_obs)
    sol = _efa.fit_factors(mat, int(n_factors), method)
    _warn_efa(sol, int(n_factors))
    fit: dict[str, float]
    if n is None:
        full = _efa.fit_indices(sol.loadings, mat, 2)
        fit = {"caf": full["caf"], "df": full["df"]}
    else:
        fit = _efa.fit_indices(sol.loadings, mat, n)
    columns = [f"F{f + 1}" for f in range(int(n_factors))]
    h2 = np.sum(sol.loadings**2, axis=1)
    return EFAResult(
        loadings=pd.DataFrame(sol.loadings, index=names, columns=columns),
        uniquenesses=pd.Series(sol.uniquenesses, index=names, name="uniqueness"),
        communalities=pd.Series(h2, index=names, name="communality"),
        method=method,
        converged=sol.converged,
        heywood=bool(np.any(sol.uniquenesses <= _efa.UNIQUENESS_FLOOR * (1 + 1e-9))),
        objective=sol.objective,
        fit=fit,
    )


def _warn_efa(sol: _efa.FactorSolution, k: int) -> None:
    if not sol.converged:
        warnings.warn(
            f"The {k}-factor solution did not converge.", ConvergenceWarning, stacklevel=3
        )
    if np.any(sol.uniquenesses <= _efa.UNIQUENESS_FLOOR * (1 + 1e-9)):
        warnings.warn(
            f"The {k}-factor solution has a uniqueness at its lower bound (Heywood case).",
            HeywoodWarning,
            stacklevel=3,
        )


@dataclass(frozen=True)
class HullResult:
    """Result of :func:`hull`.

    Attributes
    ----------
    n_factors : int
        Selected number of factors.
    table : pandas.DataFrame
        Rows ``k = 0..max_factors``: ``n_factors``, ``fit`` (CAF, CFI or 1 - RMSEA), ``df``,
        ``st`` (scree test value, NaN off the hull's interior), ``on_hull``, ``heywood`` and
        ``converged`` (of each EFA solution).
    fit_index : str
        ``"caf"``, ``"cfi"`` or ``"rmsea"``.
    method : str
        EFA estimator.
    max_factors : int
        Largest number of factors fitted.
    """

    n_factors: int
    table: pd.DataFrame
    fit_index: str
    method: str
    max_factors: int


def hull(
    correlation: pd.DataFrame | np.ndarray | PolychoricResult,
    *,
    n_obs: int | None = None,
    max_factors: int | None = None,
    key: jax.Array | None = None,
    method: EFAMethod = "uls",
    fit_index: HullFitIndex = "caf",
    n_replications: int = 1000,
) -> HullResult:
    r"""Hull method for the number of factors (Lorenzo-Seva, Timmerman & Kiers, 2011).

    Factor solutions with :math:`k=0,\dots,J` factors are placed by their fit :math:`f_k` and
    degrees of freedom :math:`df_k`; the solutions on the upper convex boundary are kept and the
    selected :math:`k` maximizes

    .. math::
        st_k=\frac{(f_k-f_{k-1})/(df_k-df_{k-1})}{(f_{k+1}-f_k)/(df_{k+1}-df_k)},

    the point after which more factors buy little fit. Fit is CAF (common part accounted for,
    :math:`1-\mathrm{KMO}` of the residual matrix), CFI or :math:`1-\mathrm{RMSEA}`; the
    :math:`k=0` row uses the correlation matrix itself. This is EFAtools' ``HULL()``. By default
    :math:`J` is one more than parallel analysis (SMC eigenvalues of normal data, mean rule, as
    EFAtools), capped at the largest :math:`k` with positive degrees of freedom and at least 3.

    Parameters
    ----------
    correlation : pandas.DataFrame, numpy.ndarray or PolychoricResult
        Positive definite correlation matrix of at least 6 variables.
    n_obs : int, optional
        Sample size (needed for CFI and RMSEA and for the default ``max_factors``; default for a
        ``PolychoricResult``: its smallest pairwise :math:`n`).
    max_factors : int, optional
        :math:`J`. If omitted, from parallel analysis with ``key``.
    key : jax.Array, optional
        JAX PRNG key for the default ``max_factors``.
    method : {"uls", "ml"}, default "uls"
        EFA estimator (see :func:`efa`).
    fit_index : {"caf", "cfi", "rmsea"}, default "caf"
        Fit measure.
    n_replications : int, default 1000
        Parallel-analysis data sets for the default ``max_factors``.

    Returns
    -------
    HullResult

    Raises
    ------
    SpecificationError
        Invalid options, fewer than 6 variables, or missing ``n_obs`` / ``key`` when needed.

    Warns
    -----
    HeywoodWarning
        If the selected solution has a Heywood case or did not converge (as EFAtools).
    FlexpyirtWarning
        If fewer than three solutions are on the hull.
    """
    _, mat = _as_correlation(correlation)
    p = mat.shape[0]
    if p < 6:
        raise SpecificationError("hull() needs at least 6 variables.")
    if method not in _EFA_METHODS:
        raise SpecificationError(f"method must be one of {_EFA_METHODS}, got {method!r}.")
    if fit_index not in _HULL_FIT:
        raise SpecificationError(f"fit_index must be one of {_HULL_FIT}, got {fit_index!r}.")
    if np.linalg.eigvalsh(mat).min() <= 0:
        raise SpecificationError("hull() needs a positive definite correlation matrix.")
    needs_n = fit_index != "caf" or max_factors is None
    n = None
    if n_obs is not None or isinstance(correlation, PolychoricResult):
        n = _default_n_obs(correlation, n_obs)
    elif needs_n:
        raise SpecificationError(
            "n_obs is required for CFI / RMSEA and for the default max_factors."
        )
    cap = _efa.max_identified_factors(p)
    if max_factors is None:
        if key is None:
            raise SpecificationError("Pass max_factors= or a key= for parallel analysis.")
        assert n is not None
        reps = _normal_reference(key, n, p, int(n_replications), "smc", chunk=500)
        n_pa = _count_leading(_eigenvalues(mat, "smc"), reps.mean(axis=0))
        j = min(n_pa + 1, cap)
        j = max(j, min(3, cap))
    else:
        if not (isinstance(max_factors, (int, np.integer)) and 1 <= max_factors <= cap):
            raise SpecificationError(
                f"max_factors must be an integer in [1, {cap}] for {p} variables, "
                f"got {max_factors!r}."
            )
        j = int(max_factors)
    n_eval = n if n is not None else 2
    rows = [{**_efa.null_row(mat, n_eval), "heywood": False, "converged": True}]
    for k in range(1, j + 1):
        sol = _efa.fit_factors(mat, k, method)
        idx = _efa.fit_indices(sol.loadings, mat, n_eval)
        rows.append(
            {
                "caf": idx["caf"],
                "cfi": idx["cfi"],
                "rmsea": 1.0 - idx["rmsea"],
                "df": idx["df"],
                "heywood": bool(np.any(sol.uniquenesses <= _efa.UNIQUENESS_FLOOR * (1 + 1e-9))),
                "converged": sol.converged,
            }
        )
    fit = np.array([r[fit_index] for r in rows], dtype=np.float64)
    df = np.array([r["df"] for r in rows], dtype=np.float64)
    selected, st, on_hull = _efa.hull_select(fit, df)
    if int(np.sum(on_hull)) < 3:
        warnings.warn(
            "Fewer than three solutions are on the hull; the solution with the best fit is "
            "returned.",
            FlexpyirtWarning,
            stacklevel=2,
        )
    table = pd.DataFrame(
        {
            "n_factors": np.arange(j + 1),
            "fit": fit,
            "df": df,
            "st": st,
            "on_hull": on_hull,
            "heywood": [bool(r["heywood"]) for r in rows],
            "converged": [bool(r["converged"]) for r in rows],
        }
    )
    if rows[selected]["heywood"] or not rows[selected]["converged"]:
        warnings.warn(
            f"The selected {selected}-factor solution has a Heywood case or did not converge; "
            "cross-check the number of factors with other criteria.",
            HeywoodWarning,
            stacklevel=2,
        )
    return HullResult(
        n_factors=selected, table=table, fit_index=fit_index, method=method, max_factors=j
    )


# ----------------------------------------------------------------------------- comparison data
@dataclass(frozen=True)
class ComparisonDataResult:
    """Result of :func:`comparison_data`.

    Attributes
    ----------
    n_factors : int
        Suggested number of factors.
    rmse : pandas.DataFrame
        Eigenvalue RMSE of every bootstrap sample (rows) for each tested number of factors
        (columns ``1..``).
    p_values : pandas.Series
        For ``k >= 2``, the one-sided rank-sum :math:`p` of "RMSE with ``k`` factors is smaller
        than with ``k - 1``".
    population_rmse : pandas.Series
        RMSE of the generated populations' correlations against the observed ones.
    max_factors : int
        Largest number of factors that could be tested.
    """

    n_factors: int
    rmse: pd.DataFrame
    p_values: pd.Series
    population_rmse: pd.Series
    max_factors: int


def comparison_data(
    data: pd.DataFrame | np.ndarray,
    *,
    key: jax.Array,
    max_factors: int | None = None,
    n_population: int = 10000,
    n_samples: int = 500,
    alpha: float = 0.30,
    max_iter: int = 50,
) -> ComparisonDataResult:
    r"""Comparison-data method for the number of factors (Ruscio & Roche, 2012).

    For :math:`k=1,2,\dots`, a population of ``n_population`` cases with :math:`k` common
    factors that reproduces the observed Pearson correlations and every variable's observed
    distribution is generated (Ruscio & Kaczetow, 2008); ``n_samples`` samples of the data's
    size are drawn from it and the RMSE between their eigenvalues and the observed eigenvalues is
    recorded. Adding the :math:`k`-th factor is kept while the RMSE with :math:`k` factors is
    smaller than with :math:`k-1` by a one-sided Mann–Whitney test at ``alpha`` (0.30, as
    recommended); the suggestion is the last kept :math:`k`. This is EFAtools' ``CD()``.

    Parameters
    ----------
    data : pandas.DataFrame or numpy.ndarray
        Respondents x variables, complete numeric data (Pearson correlations).
    key : jax.Array
        JAX PRNG key.
    max_factors : int, optional
        Largest :math:`k` (default and cap: the largest :math:`k` with positive degrees of
        freedom).
    n_population : int, default 10000
        Size of each comparison population.
    n_samples : int, default 500
        Samples drawn from each population.
    alpha : float, default 0.30
        Level of the sequential tests.
    max_iter : int, default 50
        Iterations of the principal axis factoring inside the generator.

    Returns
    -------
    ComparisonDataResult

    Raises
    ------
    SpecificationError
        Missing responses, a constant variable, too few variables, or invalid options.
    PrecisionError
        If JAX is not running in 64-bit mode.
    """
    check_x64()
    _, values = _coerce_ordinal(data)
    if np.isnan(values).any():
        raise SpecificationError("comparison_data() needs complete data.")
    n_obs, p = values.shape
    if np.any(np.ptp(values, axis=0) == 0):
        raise SpecificationError("Every variable must have at least two distinct values.")
    cap = _efa.max_identified_factors(p)
    if cap < 1:
        raise SpecificationError("comparison_data() needs enough variables for one factor.")
    if max_factors is None:
        k_max = cap
    elif isinstance(max_factors, (int, np.integer)) and max_factors >= 1:
        k_max = min(int(max_factors), cap)
    else:
        raise SpecificationError(f"max_factors must be a positive integer, got {max_factors!r}.")
    for name, value in (("n_population", n_population), ("n_samples", n_samples)):
        if not (isinstance(value, (int, np.integer)) and value >= 2):
            raise SpecificationError(f"{name} must be an integer of at least 2, got {value!r}.")
    if not (0.0 < float(alpha) < 1.0):
        raise SpecificationError(f"alpha must be in (0, 1), got {alpha!r}.")

    corr = np.corrcoef(values, rowvar=False)
    observed = _descending_eigenvalues(corr)
    rmse: dict[int, np.ndarray] = {}
    p_values: dict[int, float] = {}
    population_rmse: dict[int, float] = {}
    chunk = max(1, int(2e7 // (n_obs * p)))
    k = 1  # as EFAtools: k is kept while adding it helps; the result is the last kept k
    for k_key in jax.random.split(key, k_max):
        k_draw, k_shared, k_unique, k_boot = jax.random.split(k_key, 4)
        rows = np.asarray(jax.random.randint(k_draw, (int(n_population), p), 0, n_obs))
        dists = np.sort(values[rows, np.arange(p)], axis=0)
        if np.any(dists[0] == dists[-1]):
            raise SpecificationError(
                "Resampling a variable's distribution gave a constant column; increase "
                "n_population or collapse very rare categories."
            )
        shared = np.asarray(jax.random.normal(k_shared, (int(n_population), k), dtype=jnp.float64))
        unique = np.asarray(jax.random.normal(k_unique, (int(n_population), p), dtype=jnp.float64))
        pop, pop_rmse = _comparison_data.generate_population(
            corr, dists, shared, unique, max_iter=int(max_iter)
        )
        population_rmse[k] = pop_rmse
        boot = np.asarray(jax.random.randint(k_boot, (int(n_samples), n_obs), 0, int(n_population)))
        rmse[k] = np.concatenate(
            [
                _comparison_data.eigenvalue_rmse(pop[boot[s : s + chunk]], observed)
                for s in range(0, int(n_samples), chunk)
            ]
        )
        if k > 1:
            p_values[k] = _comparison_data.wilcoxon_less(rmse[k], rmse[k - 1])
            if not p_values[k] < float(alpha):
                break
        k += 1
    return ComparisonDataResult(
        n_factors=k - 1,
        rmse=pd.DataFrame(rmse),
        p_values=pd.Series(p_values, name="p", dtype=np.float64),
        population_rmse=pd.Series(population_rmse, name="population_rmse"),
        max_factors=k_max,
    )


# ----------------------------------------------------------------------------- DIMTEST
@dataclass(frozen=True)
class DimtestResult:
    r"""Result of :func:`dimtest`.

    Attributes
    ----------
    statistic : float
        Bias-corrected :math:`T=(T_L-\bar T_G)/\sqrt{1+1/B}`, standard normal under essential
        unidimensionality.
    p_value : float
        One-sided :math:`1-\Phi(T)`.
    t_l : float
        :math:`T_L` on the test sample.
    t_g : numpy.ndarray
        :math:`T_L` of the ``B`` bootstrap data sets.
    assessment_items, partitioning_items : list of str
        The AT and PT items.
    n_groups : int
        Partitioning-score groups used on the test sample.
    training_rows, test_rows : numpy.ndarray
        Row positions used to select the AT (empty when ``assessment_items`` is given) and to test.
    """

    statistic: float
    p_value: float
    t_l: float
    t_g: np.ndarray
    assessment_items: list[str]
    partitioning_items: list[str]
    n_groups: int
    training_rows: np.ndarray
    test_rows: np.ndarray


def _select_assessment_items(x: np.ndarray, max_size: int) -> np.ndarray:
    """AT = the cluster (Ward on conditional covariances) with the largest DETECT vs the rest."""
    n_items = x.shape[1]
    ccov = _detect.conditional_covariances(x)
    partitions = _detect.ward_clusters(ccov, list(range(2, min(6, n_items) + 1)))
    best: tuple[float, tuple[int, ...]] | None = None
    seen: set[tuple[int, ...]] = set()
    for labels in partitions.values():
        for label in np.unique(labels):
            members = tuple(int(i) for i in np.flatnonzero(labels == label))
            if not (2 <= len(members) <= max_size) or members in seen:
                continue
            seen.add(members)
            split = np.where(np.isin(np.arange(n_items), members), 1, 2)
            value = _detect.detect_indices(ccov, split)["DETECT"][0]
            if best is None or value > best[0]:
                best = (value, members)
    if best is None:
        raise SpecificationError("No assessment subtest of the allowed size could be formed.")
    return np.asarray(best[1], dtype=np.int64)


def dimtest(
    data: pd.DataFrame | np.ndarray,
    *,
    key: jax.Array,
    assessment_items: Sequence[str] | None = None,
    training_fraction: float = 1 / 3,
    n_bootstrap: int = 100,
    min_group: int = 20,
    max_assessment_size: int | None = None,
) -> DimtestResult:
    r"""DIMTEST: test essential unidimensionality (Stout, 1987; Stout, Froelich & Gao, 2001).

    The items are split into an assessment subtest (AT) and a partitioning subtest (PT). Given
    the PT score, the AT items' conditional covariances are near zero under essential
    unidimensionality and positive when the AT measures a dimension the PT does not;
    :math:`T_L` (Nandakumar & Stout, 1993; :func:`flexpyirt._dimtest.t_l_statistic`) sums their
    standardized group-wise totals. Its bias is estimated by the mean :math:`\bar T_G` of
    :math:`T_L` over ``n_bootstrap`` unidimensional data sets generated from kernel-smoothed item
    response functions of the test sample (Ramsay, 1991, bandwidth :math:`1.1N^{-1/5}`;
    :math:`\theta\sim N(0,1)`), and

    .. math::
        T=\frac{T_L-\bar T_G}{\sqrt{1+1/B}},\qquad p=1-\Phi(T)

    (Stout, Froelich & Gao, 2001). Without ``assessment_items`` the AT is chosen on a random
    ``training_fraction`` of the respondents (Froelich & Habing, 2008, use a separate subsample):
    among the clusters of a Ward clustering of conditional covariances (:func:`exploratory_detect`)
    with 2 to ``max_assessment_size`` items, the one whose split from the rest has the largest
    DETECT; the test then uses the other respondents only. This AT search is in the spirit of
    Froelich & Habing's conditional-covariance selection, not their exact procedure. The test is
    accepted on a simulation study (no reference implementation outside DIMPACK; DESIGN.md
    §16.16).

    Parameters
    ----------
    data : pandas.DataFrame or numpy.ndarray
        Respondents x items, complete binary (0/1) responses.
    key : jax.Array
        JAX PRNG key (split, AT search, bootstrap).
    assessment_items : sequence of str, optional
        The AT items (then all respondents are used for the test; choose them without looking at
        these data, or the test is biased).
    training_fraction : float, default 1/3
        Share of respondents used to select the AT.
    n_bootstrap : int, default 100
        Bootstrap data sets :math:`B`.
    min_group : int, default 20
        Smallest PT-score group used.
    max_assessment_size : int, optional
        Largest AT (default half the items).

    Returns
    -------
    DimtestResult

    Raises
    ------
    SpecificationError
        Non-binary or incomplete data, fewer than 4 items, invalid options, or no usable
        PT-score group.
    PrecisionError
        If JAX is not running in 64-bit mode.
    """
    check_x64()
    names, values = _coerce_ordinal(data)
    if np.isnan(values).any() or not np.isin(values, (0.0, 1.0)).all():
        raise SpecificationError("dimtest() needs complete binary (0/1) responses.")
    n_obs, n_items = values.shape
    if n_items < 4:
        raise SpecificationError("dimtest() needs at least 4 items.")
    max_size = n_items // 2 if max_assessment_size is None else int(max_assessment_size)
    if not 2 <= max_size <= n_items - 2:
        raise SpecificationError(
            f"max_assessment_size must be in [2, {n_items - 2}], got {max_assessment_size!r}."
        )
    if not (isinstance(n_bootstrap, (int, np.integer)) and n_bootstrap >= 2):
        raise SpecificationError(
            f"n_bootstrap must be an integer of at least 2, got {n_bootstrap!r}."
        )
    if not (isinstance(min_group, (int, np.integer)) and min_group >= 2):
        raise SpecificationError(f"min_group must be an integer of at least 2, got {min_group!r}.")
    k_split, k_ties, k_boot = jax.random.split(key, 3)
    if assessment_items is None:
        if not (0.0 < float(training_fraction) < 1.0):
            raise SpecificationError(
                f"training_fraction must be in (0, 1), got {training_fraction!r}."
            )
        perm = np.asarray(jax.random.permutation(k_split, n_obs))
        n_train = int(np.floor(n_obs * float(training_fraction)))
        if n_train < 2 or n_obs - n_train < 2:
            raise SpecificationError("Too few respondents for the training / test split.")
        train = np.sort(perm[:n_train])
        test = np.sort(perm[n_train:])
        at_idx = _select_assessment_items(values[train], max_size)
    else:
        chosen = list(dict.fromkeys(str(i) for i in assessment_items))
        unknown = [i for i in chosen if i not in names]
        if unknown:
            raise SpecificationError(f"Unknown assessment_items: {unknown}.")
        if not 2 <= len(chosen) <= n_items - 2:
            raise SpecificationError("The AT needs at least 2 items and the PT at least 2.")
        at_idx = np.asarray([names.index(i) for i in chosen], dtype=np.int64)
        train = np.zeros(0, dtype=np.int64)
        test = np.arange(n_obs)
    pt_idx = np.setdiff1d(np.arange(n_items), at_idx)
    x = values[test]
    t_l, n_groups = _dimtest.t_l_statistic(x[:, at_idx], x[:, pt_idx].sum(axis=1), int(min_group))
    if n_groups == 0:
        raise SpecificationError(
            f"No partitioning-score group has at least min_group={min_group} respondents."
        )
    n_test = x.shape[0]
    grid = np.linspace(-4.0, 4.0, 161)
    tie_noise = np.asarray(jax.random.uniform(k_ties, (n_test,), dtype=jnp.float64))
    irfs = _dimtest.kernel_irfs(x, tie_noise, grid, 1.1 * n_test ** (-0.2))
    t_g = []
    for kb in jax.random.split(k_boot, int(n_bootstrap)):
        k_theta, k_unif = jax.random.split(kb)
        theta = np.asarray(jax.random.normal(k_theta, (n_test,), dtype=jnp.float64))
        unif = np.asarray(jax.random.uniform(k_unif, (n_test, n_items), dtype=jnp.float64))
        sim = _dimtest.simulate_from_irfs(irfs, grid, theta, unif)
        value, groups = _dimtest.t_l_statistic(
            sim[:, at_idx], sim[:, pt_idx].sum(axis=1), int(min_group)
        )
        if groups > 0:
            t_g.append(value)
    t_g_arr = np.asarray(t_g)
    statistic = float((t_l - t_g_arr.mean()) / np.sqrt(1.0 + 1.0 / t_g_arr.size))
    return DimtestResult(
        statistic=statistic,
        p_value=float(norm.sf(statistic)),
        t_l=float(t_l),
        t_g=t_g_arr,
        assessment_items=[names[i] for i in at_idx],
        partitioning_items=[names[i] for i in pt_idx],
        n_groups=int(n_groups),
        training_rows=train,
        test_rows=test,
    )
