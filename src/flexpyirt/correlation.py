"""Polychoric and tetrachoric correlations (Python shell layer, DESIGN.md §16.9).

Public entry point :func:`polychoric` and result object :class:`PolychoricResult`. The jit kernels
(bivariate normal distribution function, per-pair maximum likelihood) are in
:mod:`flexpyirt._polychoric`.

flexpyctt has a NumPy / SciPy implementation of the same estimator (``flexpyctt.polychoric``,
used by the dimensionality tools); both return flexpyctt's :class:`PolychoricResult`
(DESIGN.md §16.18).
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flexpyctt.correlation import PolychoricResult
from scipy.special import ndtri

from flexpyirt._polychoric import fit_pairs
from flexpyirt._precision import check_x64
from flexpyirt.errors import (
    ConvergenceWarning,
    HeywoodWarning,
    NotPositiveDefiniteWarning,
    SpecificationError,
)

__all__ = ["PolychoricResult", "polychoric", "smooth_correlation"]

# |rho| is kept at or below tanh(_Z_MAX) = 1 - 2.25e-7; a pair whose maximum lies beyond is flagged.
_Z_MAX = 8.0
_SMOOTH_EIGEN_TOL = 1e-12

_fit_pairs = jax.jit(fit_pairs, static_argnames=("z_max", "tol", "max_iter"))


def smooth_correlation(correlation: np.ndarray) -> tuple[np.ndarray, bool]:
    r"""Smooth a symmetric matrix with unit diagonal to a positive definite correlation matrix.

    If the smallest eigenvalue is below machine epsilon, eigenvalues below
    :math:`10^{-12}` are replaced by :math:`10^{-10}`, all eigenvalues are rescaled to sum to the
    number of variables :math:`p`, the matrix is rebuilt and rescaled to unit diagonal:

    .. math::
        \tilde R = D^{-1/2}\, V \operatorname{diag}(\tilde\lambda) V^\top D^{-1/2},\quad
        \tilde\lambda_i = \frac{p\,\lambda_i^*}{\sum_l \lambda_l^*},\quad
        \lambda_i^*=\begin{cases}100\cdot10^{-12} & \lambda_i<10^{-12}\\ \lambda_i&\text{else}
        \end{cases}

    with :math:`D` the diagonal of the rebuilt matrix. This is psych's ``cor.smooth``
    (Revelle, 2024; Wothke, 1993).

    Parameters
    ----------
    correlation : numpy.ndarray
        ``(p, p)`` symmetric matrix with unit diagonal.

    Returns
    -------
    tuple of numpy.ndarray and bool
        The (possibly) smoothed matrix and whether smoothing was applied.
    """
    values, vectors = np.linalg.eigh(correlation)
    if values.min() >= np.finfo(np.float64).eps:
        return correlation, False
    values = np.where(values < _SMOOTH_EIGEN_TOL, 100.0 * _SMOOTH_EIGEN_TOL, values)
    values = values * correlation.shape[0] / values.sum()
    rebuilt = (vectors * values) @ vectors.T
    scale = 1.0 / np.sqrt(np.diag(rebuilt))
    smoothed = rebuilt * scale[:, None] * scale[None, :]
    np.fill_diagonal(smoothed, 1.0)
    return (smoothed + smoothed.T) / 2.0, True


def _coerce_ordinal(data: pd.DataFrame | np.ndarray) -> tuple[list[str], np.ndarray]:
    """Return item names and a float ``(N, J)`` array (NaN = missing) of ordinal responses."""
    if isinstance(data, pd.DataFrame):
        names = [str(c) for c in data.columns]
        try:
            values = data.to_numpy(dtype=np.float64, na_value=np.nan)
        except (TypeError, ValueError) as exc:
            raise SpecificationError(
                "polychoric() needs numeric responses (ordered category codes); "
                f"could not convert data to float: {exc}"
            ) from exc
    else:
        try:
            values = np.asarray(data, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise SpecificationError(
                f"polychoric() needs numeric responses; could not convert data to float: {exc}"
            ) from exc
        if values.ndim != 2:
            raise SpecificationError(
                f"data must be two-dimensional (respondents x items), got shape {values.shape}."
            )
        names = [f"item{j + 1}" for j in range(values.shape[1])]
    if len(set(names)) != len(names):
        raise SpecificationError("Item (column) names must be unique.")
    if values.shape[1] < 2:
        raise SpecificationError("polychoric() needs at least two items.")
    if np.isinf(values).any():
        raise SpecificationError("Responses must be finite numbers or NaN (missing).")
    return names, values


def polychoric(
    data: pd.DataFrame | np.ndarray,
    *,
    smooth: bool = True,
    zero_cell_correction: float = 0.5,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> PolychoricResult:
    r"""Estimate the polychoric (tetrachoric for binary items) correlation matrix.

    Each item :math:`j` is taken as a discretized standard normal variable :math:`Y_j^*`, with
    thresholds :math:`\tau_{j1}<\dots<\tau_{j,K_j-1}` estimated from the item's marginal
    proportions, :math:`\hat\tau_{jk}=\Phi^{-1}(\hat P(Y_j\le k))`. For each pair the correlation
    :math:`\rho` of :math:`(Y_i^*, Y_j^*)` then maximizes the two-step log-likelihood

    .. math::
        \ell(\rho)=\sum_{a,b} n_{ab}\log\pi_{ab}(\rho),\quad
        \pi_{ab}=\Phi_2(\tau_{ia},\tau_{jb};\rho)-\Phi_2(\tau_{i,a-1},\tau_{jb};\rho)
        -\Phi_2(\tau_{ia},\tau_{j,b-1};\rho)+\Phi_2(\tau_{i,a-1},\tau_{j,b-1};\rho)

    (Olsson, 1979, two-step estimator; :math:`\Phi_2` by Genz, 2004), maximized by Newton's
    method in :math:`\operatorname{atanh}\rho` with analytic derivatives. lavaan's ``lavCor``
    with ordered variables computes the same estimator; psych's ``polychoric`` approximates it.

    Parameters
    ----------
    data : pandas.DataFrame or numpy.ndarray
        Respondents x items. Responses are ordered category codes (any numbers; their ascending
        order is the category order); NaN is missing. Each item needs at least two observed
        categories. Unobserved codes are not categories: an item observed as ``{0, 2}`` is
        binary.
    smooth : bool, default True
        Smooth a matrix that is not positive definite with :func:`smooth_correlation` (psych's
        ``cor.smooth``), with a :class:`~flexpyirt.errors.NotPositiveDefiniteWarning`.
    zero_cell_correction : float, default 0.5
        For a 2 x 2 table with exactly one empty cell, add this value to the two cells of the
        diagonal containing the empty cell and subtract it from the other two, keeping the
        margins (lavaan's ``zero.add[1]`` with ``zero.keep.margins = TRUE``). Without it the
        maximum is at :math:`|\rho|=1`. 0 disables the correction. Larger tables are never
        corrected.
    tol : float, default 1e-10
        Convergence criterion on the gradient in :math:`\operatorname{atanh}\rho`, relative to
        the pair's number of observations.
    max_iter : int, default 100
        Maximum number of Newton iterations per pair.

    Returns
    -------
    PolychoricResult
        The correlation matrix, thresholds and per-pair diagnostics.

    Raises
    ------
    SpecificationError
        If the data are not numeric, an item has fewer than two observed categories, or a pair of
        items has no respondent with both responses observed (or, among those, one item is
        constant).
    PrecisionError
        If JAX is not running in 64-bit mode.

    Warns
    -----
    HeywoodWarning
        If some pair's maximum is at :math:`|\rho|\to 1`.
    ConvergenceWarning
        If some pair did not converge within ``max_iter``.
    NotPositiveDefiniteWarning
        If the matrix was smoothed.

    Notes
    -----
    Missing responses are handled by available cases: thresholds use every observed response to
    the item, and each pair's table uses respondents who answered both items (psych's
    ``polychoric`` default and lavaan's ``missing = "pairwise"``). This is consistent under
    MCAR. Two-step estimation fixes the thresholds of a pair at their marginal estimates, so they
    are shared by every pair involving the item.
    """
    check_x64()
    if not (np.isfinite(zero_cell_correction) and zero_cell_correction >= 0.0):
        raise SpecificationError(
            f"zero_cell_correction must be a non-negative number, got {zero_cell_correction!r}."
        )
    if not (np.isfinite(tol) and tol > 0.0):
        raise SpecificationError(f"tol must be a positive number, got {tol!r}.")
    if not (isinstance(max_iter, int) and max_iter >= 1):
        raise SpecificationError(f"max_iter must be a positive integer, got {max_iter!r}.")
    names, values = _coerce_ordinal(data)
    n_items = len(names)

    # Recode each item to 0..K_j-1 over its observed values; -1 = missing.
    codes = np.full(values.shape, -1, dtype=np.int64)
    categories: dict[str, np.ndarray] = {}
    thresholds: dict[str, np.ndarray] = {}
    n_categories = np.zeros(n_items, dtype=np.int64)
    for j, name in enumerate(names):
        col = values[:, j]
        observed = ~np.isnan(col)
        uniq, inverse = np.unique(col[observed], return_inverse=True)
        if uniq.size < 2:
            raise SpecificationError(
                f"Item {name!r} has {uniq.size} observed categor{'y' if uniq.size == 1 else 'ies'};"
                " a polychoric correlation needs at least two."
            )
        codes[observed, j] = inverse
        categories[name] = uniq
        n_categories[j] = uniq.size
        counts = np.bincount(inverse, minlength=uniq.size).astype(np.float64)
        thresholds[name] = ndtri(np.cumsum(counts)[:-1] / counts.sum())

    k_max = int(np.max(n_categories))
    bounds = np.full((n_items, k_max + 1), np.inf)
    bounds[:, 0] = -np.inf
    for j, name in enumerate(names):
        bounds[j, 1 : n_categories[j]] = thresholds[name]

    rows, cols = np.triu_indices(n_items, k=1)
    n_pairs = rows.size
    tables = np.zeros((n_pairs, k_max, k_max), dtype=np.float64)
    start = np.zeros(n_pairs, dtype=np.float64)
    corrected = np.zeros(n_pairs, dtype=bool)
    for p in range(n_pairs):
        i, j = int(rows[p]), int(cols[p])
        both = (codes[:, i] >= 0) & (codes[:, j] >= 0)
        ci, cj = codes[both, i], codes[both, j]
        if ci.size == 0:
            raise SpecificationError(
                f"Items {names[i]!r} and {names[j]!r} have no respondent with both responses "
                "observed."
            )
        if np.unique(ci).size < 2 or np.unique(cj).size < 2:
            raise SpecificationError(
                f"Among respondents who answered both {names[i]!r} and {names[j]!r}, one item is "
                "constant; their polychoric correlation is not identified."
            )
        table = np.zeros((k_max, k_max), dtype=np.float64)
        np.add.at(table, (ci, cj), 1.0)
        if n_categories[i] == 2 and n_categories[j] == 2 and zero_cell_correction > 0.0:
            empty = np.argwhere(table[:2, :2] == 0.0)
            if empty.shape[0] == 1:
                sign = 1.0 if empty[0, 0] == empty[0, 1] else -1.0
                adj = sign * zero_cell_correction * np.array([[1.0, -1.0], [-1.0, 1.0]])
                table[:2, :2] += adj
                corrected[p] = True
        tables[p] = table
        pearson = np.corrcoef(ci, cj)[0, 1]
        start[p] = np.arctanh(np.clip(pearson if np.isfinite(pearson) else 0.0, -0.9, 0.9))

    fitted = _fit_pairs(
        jnp.asarray(start),
        jnp.asarray(bounds[rows]),
        jnp.asarray(bounds[cols]),
        jnp.asarray(tables),
        z_max=_Z_MAX,
        tol=float(tol),
        max_iter=int(max_iter),
    )
    rho = np.asarray(fitted.rho, dtype=np.float64)
    converged = np.asarray(fitted.converged, dtype=bool)
    at_boundary = np.asarray(fitted.at_boundary, dtype=bool)

    def square(upper: np.ndarray, diagonal: float | int | bool, dtype: type) -> np.ndarray:
        out: np.ndarray = np.empty((n_items, n_items), dtype=dtype)
        out[rows, cols] = upper
        out[cols, rows] = upper
        np.fill_diagonal(out, diagonal)
        return out

    raw = square(rho, 1.0, np.float64)
    observed_counts = (codes >= 0).astype(np.int64)
    n_obs = observed_counts.T @ observed_counts

    def frame(mat: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(mat, index=names, columns=names)

    min_eigenvalue = float(np.linalg.eigvalsh(raw).min())
    final, smoothed = smooth_correlation(raw) if smooth else (raw, False)

    pair_names = [f"{names[i]}-{names[j]}" for i, j in zip(rows, cols, strict=True)]
    if at_boundary.any():
        flagged = [pair_names[p] for p in np.flatnonzero(at_boundary)]
        warnings.warn(
            f"{len(flagged)} pair(s) have their maximum at |rho| -> 1 ({', '.join(flagged[:5])}"
            f"{', ...' if len(flagged) > 5 else ''}); typically a table with empty cells.",
            HeywoodWarning,
            stacklevel=2,
        )
    if not converged.all():
        flagged = [pair_names[p] for p in np.flatnonzero(~converged)]
        warnings.warn(
            f"{len(flagged)} pair(s) did not converge in max_iter={max_iter} iterations "
            f"({', '.join(flagged[:5])}{', ...' if len(flagged) > 5 else ''}).",
            ConvergenceWarning,
            stacklevel=2,
        )
    if smoothed:
        warnings.warn(
            f"The polychoric matrix was not positive definite (smallest eigenvalue "
            f"{min_eigenvalue:.3g}) and was smoothed.",
            NotPositiveDefiniteWarning,
            stacklevel=2,
        )

    return PolychoricResult(
        correlation=frame(final),
        unsmoothed_correlation=frame(raw),
        thresholds=thresholds,
        categories=categories,
        n_obs=frame(n_obs),
        converged=frame(square(converged, True, bool)),
        at_boundary=frame(square(at_boundary, False, bool)),
        zero_cell_corrected=frame(square(corrected, False, bool)),
        smoothed=smoothed,
        min_eigenvalue=min_eigenvalue,
    )
