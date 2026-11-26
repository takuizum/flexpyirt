"""Model-comparison table over several fits of the same data (Result layer, DESIGN.md §24.1).

Pure post-fit Python: every number comes from ``FitResult.loglik``, ``FitResult.n_params`` and
the sample size; nothing is refitted and nothing runs inside ``jit``.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from flexpyirt._dif import information_criteria
from flexpyirt.errors import FlexpyirtWarning, SpecificationError
from flexpyirt.result import FitResult

__all__ = ["COMPARE_COLUMNS", "compare_fits"]

COMPARE_COLUMNS = (
    "model",
    "n_params",
    "loglik",
    "aic",
    "sabic",
    "hq",
    "bic",
    "x2",
    "df",
    "p",
)
"""Columns of the table returned by :func:`compare_fits`."""

LOGLIK_ORDER_TOL = 1e-8
"""Decrease of the log-likelihood (in absolute value) tolerated before warning about the order."""


def _sample_size(result: FitResult) -> float:
    """Return the :math:`N` of BIC: the number of rows or the sum of the weights (DESIGN §17.4)."""
    return float(result.n_obs if result._sum_weights is None else result._sum_weights)


def compare_fits(*results: FitResult, names: Sequence[str] | None = None) -> pd.DataFrame:
    r"""Compare several fits of the same data by information criteria and likelihood-ratio tests.

    One row per fit, in the order given. With :math:`\ell` the marginal log-likelihood,
    :math:`k` the number of free parameters (``FitResult.n_params``) and :math:`N` the number
    of respondents,

    .. math::
        \mathrm{AIC}=-2\ell+2k,\quad \mathrm{SABIC}=-2\ell+k\log\frac{N+2}{24},\quad
        \mathrm{HQ}=-2\ell+2k\log\log N,\quad \mathrm{BIC}=-2\ell+k\log N

    (Akaike, 1974; Sclove, 1987; Hannan & Quinn, 1979; Schwarz, 1978), and each row from the
    second on is tested against the row before it by the likelihood-ratio statistic

    .. math::
        X^2=2(\ell_m-\ell_{m-1}),\qquad \mathit{df}=k_m-k_{m-1},\qquad
        p=\Pr(\chi^2_{\mathit{df}}>X^2)

    (Wilks, 1938). This is R mirt's ``anova(mod1, mod2, ...)`` (Chalmers, 2012).

    Pass the fits from the most restricted to the least restricted model. The test is valid
    only when each model is nested in the next one (the restricted model is the larger model
    with some parameters fixed or set equal) and the restriction is not on the boundary of the
    parameter space; neither can be checked here, so it is the caller's responsibility. A test
    of a variance fixed at 0, of a mixture against fewer components or of a ``DavidianCurve``
    against a ``Normal`` latent distribution does not follow :math:`\chi^2_{\mathit{df}}`, and
    no boundary correction is applied; compare such models by the information criteria only.

    Parameters
    ----------
    *results : FitResult
        Two or more fits of the same responses, ordered from the most restricted model to the
        least restricted.
    names : sequence of str, optional
        Labels for the ``model`` column (default ``"model1"``, ``"model2"``, ...).

    Returns
    -------
    DataFrame
        Columns ``model``, ``n_params``, ``loglik``, ``aic``, ``sabic``, ``hq``, ``bic``,
        ``x2``, ``df`` and ``p``. ``x2``, ``df`` and ``p`` compare each row with the previous
        one; they are NaN in the first row, and ``p`` is NaN when ``df <= 0``.

    Raises
    ------
    SpecificationError
        If fewer than two fits are given, an argument is not a ``FitResult``, ``names`` does
        not have one distinct label per fit, or the fits do not have the same number of
        respondents (or, for fits with ``weights=``, the same sum of weights).

    Warns
    -----
    FlexpyirtWarning
        When a row has ``df <= 0`` (the fits are not ordered from the most restricted model),
        when the log-likelihood decreases while parameters are added (a non-nested pair or a
        local maximum), or when a fit has active priors (the criteria use the marginal
        log-likelihood at the MAP estimate, as ``FitResult.aic`` warns).

    Notes
    -----
    For fits with sampling weights :math:`N` is the sum of the weights, as in
    ``FitResult.bic``. The models' data are not compared: fits of different responses with the
    same number of rows give a table without meaning.

    Examples
    --------
    >>> rasch = fit(Model.uniform(Rasch(), items=cols), data)
    >>> twopl = fit(Model.uniform(TwoPL(), items=cols), data)
    >>> compare_fits(rasch, twopl, names=["Rasch", "2PL"])
    """
    if len(results) < 2:
        raise SpecificationError("compare_fits() needs at least two FitResult objects.")
    bad = [type(r).__name__ for r in results if not isinstance(r, FitResult)]
    if bad:
        raise SpecificationError(f"compare_fits() takes FitResult objects, got {bad}.")
    if names is None:
        labels = [f"model{m}" for m in range(1, len(results) + 1)]
    else:
        if isinstance(names, str):
            raise SpecificationError("names must be a sequence of str, not a single str.")
        labels = [str(n) for n in names]
        if len(labels) != len(results):
            raise SpecificationError(
                f"names has {len(labels)} labels but {len(results)} fits were given."
            )
        if len(set(labels)) != len(labels):
            raise SpecificationError(f"names must be distinct, got {labels}.")

    sizes = [_sample_size(r) for r in results]
    if any(not math.isclose(s, sizes[0], rel_tol=1e-12, abs_tol=0.0) for s in sizes):
        raise SpecificationError(
            "compare_fits() needs fits of the same data; the numbers of respondents (sums of "
            f"weights for weighted fits) differ: {dict(zip(labels, sizes, strict=True))}."
        )
    with_priors = [lab for lab, r in zip(labels, results, strict=True) if r._has_priors]
    if with_priors:
        warnings.warn(
            f"Fits {with_priors} have priors: the criteria and likelihood-ratio tests use the "
            "marginal log-likelihood at the MAP estimate rather than the MLE; interpret with "
            "caution.",
            FlexpyirtWarning,
            stacklevel=2,
        )

    n = sizes[0]
    rows: list[dict[str, object]] = []
    not_increasing: list[str] = []
    decreasing: list[str] = []
    for m, (label, res) in enumerate(zip(labels, results, strict=True)):
        k = res.n_params
        row: dict[str, object] = {
            "model": label,
            "n_params": k,
            "loglik": float(res.loglik),
            **information_criteria(float(res.loglik), k, n),
            "x2": np.nan,
            "df": np.nan,
            "p": np.nan,
        }
        if m > 0:
            prev = results[m - 1]
            df = k - prev.n_params
            x2 = 2.0 * (float(res.loglik) - float(prev.loglik))
            row["x2"] = x2
            row["df"] = df
            if df <= 0:
                not_increasing.append(f"{labels[m - 1]} -> {label} (df = {df})")
            else:
                row["p"] = float(stats.chi2.sf(max(x2, 0.0), df))
                if x2 < -2.0 * LOGLIK_ORDER_TOL:
                    decreasing.append(f"{labels[m - 1]} -> {label} (X2 = {x2:.6g})")
        rows.append(row)
    if not_increasing:
        warnings.warn(
            "The number of parameters does not increase for "
            + ", ".join(not_increasing)
            + "; order the fits from the most restricted model to the least restricted. "
            "p is NaN for these rows.",
            FlexpyirtWarning,
            stacklevel=2,
        )
    if decreasing:
        warnings.warn(
            "The log-likelihood decreases although parameters are added for "
            + ", ".join(decreasing)
            + "; the models may not be nested, or a fit stopped at a local maximum.",
            FlexpyirtWarning,
            stacklevel=2,
        )
    out = pd.DataFrame(rows, columns=list(COMPARE_COLUMNS))
    out["df"] = out["df"].astype("Int64")
    return out
