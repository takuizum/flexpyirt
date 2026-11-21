"""Estimation entry point."""

from __future__ import annotations

import warnings
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from flexpyirt import _core
from flexpyirt._compile import LATENT_ITEM, compile_problem, prepare
from flexpyirt._precision import check_x64
from flexpyirt.errors import (
    ConvergenceWarning,
    FlexpyirtWarning,
    HeywoodWarning,
    SpecificationError,
)
from flexpyirt.method import BAEM
from flexpyirt.model import Model
from flexpyirt.quadrature import Quadrature
from flexpyirt.result import FitResult

__all__ = ["fit"]

SEMethod = Literal["oakes", "hessian"]

HEYWOOD_BOUND_TOL = 1e-4
"""Distance to a finite bound below which a free parameter is reported as a boundary solution."""
HEYWOOD_MAX_SLOPE = 20.0
"""Absolute slope above which a free slope is reported as diverging."""
SQUAREM_FALLBACK_WARN_MIN = 5
"""Minimum full fallbacks to ``theta_2`` before warning about frequent SQUAREM fallbacks."""
SQUAREM_FALLBACK_WARN_RATE = 0.5
"""Fallback fraction above which a ``ConvergenceWarning`` is emitted on a converged fit."""

_em = jax.jit(_core.em, static_argnames=("max_iter", "n_newton"))
_squarem = jax.jit(_core.squarem, static_argnames=("max_iter", "n_newton"))
_information = {
    "oakes": jax.jit(_core.information_oakes),
    "hessian": jax.jit(_core.information_hessian),
}


def fit(
    spec: Model,
    data: Any,
    *,
    group: Any = None,
    table: pd.DataFrame | None = None,
    item_names: list[str] | None = None,
    method: BAEM | None = None,
    quadrature: Quadrature | None = None,
    se: SEMethod | None = "oakes",
) -> FitResult:
    r"""Estimate an IRT model by MMLE or MAP with Bock–Aitkin EM.

    Bock & Aitkin (1981). When item-parameter priors are attached via :meth:`Model.prior` (or the
    ``prior`` column of ``table``), EM finds the Bayes modal (MAP) estimate on the natural
    parameter scale without a Jacobian term (DESIGN.md §4.7). Standard errors come from Oakes'
    identity (Oakes, 1999) or the direct Hessian of the log-likelihood (or log posterior with
    priors), both by automatic differentiation (DESIGN.md §4.8).

    Parameters
    ----------
    spec : Model
        Model specification.
    data : DataFrame or array-like
        Responses (persons × items) coded ``0..n_cat-1``; NaN is missing (MAR). Columns must be
        exactly the model's items.
    group : array-like, optional
        Group label of each row for a multiple-group model (see ``Model.with_groups``).
    table : DataFrame, optional
        Edited parameter table from ``spec.parameter_table(data)``; takes precedence over
        ``spec``.
    item_names : list of str, optional
        Column names when ``data`` is an array.
    method : BAEM, optional
        EM settings (default ``BAEM()``).
    quadrature : Quadrature, optional
        Quadrature grid (default: 61 points on [-6, 6] for 1 factor, 31 per dimension for 2
        factors, and 15 per dimension for 3 factors, matching mirt's defaults).
    se : {"oakes", "hessian"} or None
        Standard-error method, or ``None`` to skip.

    Returns
    -------
    FitResult

    Raises
    ------
    PrecisionError
        If JAX is not configured for 64-bit floating point.
    SpecificationError, IdentificationError
        If the specification is invalid, before estimation starts.
    """
    check_x64()
    if se not in (None, "oakes", "hessian"):
        raise SpecificationError(f"Unknown se={se!r}; use 'oakes', 'hessian' or None.")
    method = BAEM() if method is None else method
    quadrature = Quadrature.default(spec.n_factors) if quadrature is None else quadrature

    prep = prepare(spec, data, group=group, item_names=item_names, table=table)
    final_table = prep.table
    compiled = compile_problem(spec, prep.y, prep.group, final_table, quadrature)
    problem = compiled.problem

    runner = _squarem if method.squarem else _em
    out = runner(
        problem,
        jnp.asarray(compiled.eta0),
        tol=method.tol,
        max_iter=method.max_iter,
        n_newton=method.m_step_newton_steps,
    )
    eta = np.asarray(out.eta)
    converged = bool(out.converged)
    n_iter = int(out.n_iter)
    n_em_evaluations = int(out.n_em_evaluations)
    n_safeguards = int(out.n_safeguards)
    n_fallbacks = 3 * n_iter - n_em_evaluations if method.squarem else 0
    if not converged:
        extra = (
            f"; SQUAREM safeguard fired {n_safeguards}/{n_iter} times "
            f"({n_fallbacks} fallbacks to EM)"
            if method.squarem and n_safeguards > 0
            else ""
        )
        warnings.warn(
            f"EM did not converge in {method.max_iter} iterations (tol={method.tol}{extra}).",
            ConvergenceWarning,
            stacklevel=2,
        )
    elif (
        method.squarem
        and n_fallbacks >= SQUAREM_FALLBACK_WARN_MIN
        and n_iter > 0
        and n_fallbacks / n_iter > SQUAREM_FALLBACK_WARN_RATE
    ):
        warnings.warn(
            f"SQUAREM fell back to plain EM frequently ({n_fallbacks}/{n_iter} iterations; "
            f"safeguard intervened {n_safeguards} times); consider plain EM "
            "(BAEM(squarem=False)).",
            ConvergenceWarning,
            stacklevel=2,
        )

    vcov = None
    se_values = np.full(eta.shape[0], np.nan)
    if se is not None and eta.shape[0] > 0:
        info = np.asarray(_information[se](problem, out.eta))
        vcov = _invert_information(info)
        se_values = np.sqrt(np.clip(np.diag(vcov), 0.0, None))

    params = final_table.copy()
    free = params["free"].to_numpy()
    fid = params["free_id"].to_numpy()
    estimate = params["value"].to_numpy(dtype=float).copy()
    se_column = np.full(len(params), np.nan)
    estimate[free] = eta[fid[free]]
    se_column[free] = se_values[fid[free]]
    params["estimate"] = estimate
    params["se"] = se_column
    _warn_heywood(params, spec)

    history = pd.DataFrame(
        {
            "iter": np.arange(1, n_iter + 1),
            "loglik": np.asarray(out.loglik_history)[:n_iter],
            "log_posterior": np.asarray(out.log_posterior_history)[:n_iter],
            "max_change": np.asarray(out.change_history)[:n_iter],
            "em_evaluations": np.asarray(out.em_eval_history)[:n_iter],
            "safeguard": np.asarray(out.safeguard_history)[:n_iter],
        }
    )
    return FitResult(
        parameters=params,
        loglik=float(out.loglik),
        log_posterior=float(out.log_posterior),
        converged=converged,
        n_iter=n_iter,
        n_em_evaluations=n_em_evaluations,
        n_safeguards=n_safeguards,
        history=history,
        vcov=vcov,
        se_method=se,
        n_obs=int(prep.y.shape[0]),
        _model=spec,
        _eta=eta,
        _quadrature=quadrature,
    )


def _warn_heywood(params: pd.DataFrame, spec: Model) -> None:
    """Warn about boundary solutions among the free item parameters (DESIGN.md §15.6).

    A free parameter within ``HEYWOOD_BOUND_TOL`` of a finite bound (model domain, table bound,
    or attached prior support) or a free slope with ``|a| > HEYWOOD_MAX_SLOPE`` triggers one
    ``HeywoodWarning`` naming every affected item.
    """
    items = params[(params["item"] != LATENT_ITEM) & params["free"]]
    if items.empty:
        return
    kernels = {name: spec.item_kernel(m) for name, m in spec.items}
    est = items["estimate"].to_numpy(dtype=float)
    lower = items["lower"].to_numpy(dtype=float).copy()
    upper = items["upper"].to_numpy(dtype=float).copy()
    for idx, pri in enumerate(items["prior"].tolist()):
        if pri is not None:
            p_lo, p_hi = pri.support
            lower[idx] = max(lower[idx], p_lo)
            upper[idx] = min(upper[idx], p_hi)
    at_bound = (est - lower < HEYWOOD_BOUND_TOL) | (upper - est < HEYWOOD_BOUND_TOL)
    is_slope = np.array(
        [p in kernels[i].slope_params for i, p in zip(items["item"], items["param"], strict=True)],
        dtype=bool,
    )
    steep = is_slope & (np.abs(est) > HEYWOOD_MAX_SLOPE)
    flagged = items[at_bound | steep]
    if flagged.empty:
        return
    labels = [
        f"{i}.{p}={v:.4g}" + (f" (group {g!r})" if spec.is_multigroup else "")
        for g, i, p, v in zip(
            flagged["group"], flagged["item"], flagged["param"], flagged["estimate"], strict=True
        )
    ]
    warnings.warn(
        f"Boundary (Heywood) solution for items {sorted(set(flagged['item']))}: "
        f"{', '.join(labels)}. A parameter within {HEYWOOD_BOUND_TOL} of a finite bound "
        f"or a slope beyond ±{HEYWOOD_MAX_SLOPE} is weakly identified by the data; consider "
        "fixing it (e.g. g=0 or u=1), relaxing its bounds/prior, or using a simpler item model.",
        HeywoodWarning,
        stacklevel=3,
    )


def _invert_information(info: np.ndarray) -> np.ndarray:
    """Invert the information matrix, warning (and returning NaN) if it is not positive definite."""
    sym = 0.5 * (info + info.T)
    try:
        chol = np.linalg.cholesky(sym)
    except np.linalg.LinAlgError:
        warnings.warn(
            "The information matrix is not positive definite; standard errors are NaN.",
            FlexpyirtWarning,
            stacklevel=3,
        )
        return np.full_like(sym, np.nan)
    inv_chol = np.linalg.inv(chol)
    vcov: np.ndarray = inv_chol.T @ inv_chol
    return vcov
