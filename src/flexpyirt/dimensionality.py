"""Number-of-dimensions diagnostics (Python shell layer, DESIGN.md §16.8, §16.10).

Public entry points :func:`parallel_analysis`, :func:`velicer_map` and :func:`empirical_kaiser`,
each returning a frozen result object. All three answer question (A) of DESIGN.md §16.8, "how many
dimensions?"; they are diagnostics and never feed a fit.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from flexpyirt._compile import coerce_responses
from flexpyirt._precision import check_x64
from flexpyirt.correlation import (
    PolychoricResult,
    _coerce_ordinal,
    polychoric,
    smooth_correlation,
)
from flexpyirt.errors import FlexpyirtWarning, NotPositiveDefiniteWarning, SpecificationError

if TYPE_CHECKING:
    from flexpyirt.result import FitResult

__all__ = [
    "EmpiricalKaiserResult",
    "LocalDependenceResult",
    "MAPResult",
    "ModelParallelAnalysisResult",
    "ParallelAnalysisResult",
    "empirical_kaiser",
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
