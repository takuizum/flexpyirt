"""Estimation entry point."""

from __future__ import annotations

import warnings
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from flexpyirt import _core
from flexpyirt._compile import build_table, coerce_responses, compile_problem, validate_table
from flexpyirt._precision import check_x64
from flexpyirt.errors import ConvergenceWarning, FlexpyirtWarning, SpecificationError
from flexpyirt.method import BAEM
from flexpyirt.model import Model
from flexpyirt.quadrature import Quadrature
from flexpyirt.result import FitResult

__all__ = ["fit"]

SEMethod = Literal["oakes", "hessian"]

_em = jax.jit(_core.em, static_argnames=("max_iter", "n_newton"))
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
    r"""Estimate an IRT model by marginal maximum likelihood with Bock–Aitkin EM.

    Bock & Aitkin (1981). Standard errors from Oakes' identity (Oakes, 1999) or the Hessian of the
    marginal log-likelihood, both by automatic differentiation (DESIGN.md §4.8).

    Parameters
    ----------
    spec : Model
        Model specification.
    data : DataFrame or array-like
        Responses (persons × items), 0/1. Columns must be exactly the model's items.
    group : array-like, optional
        Group labels for multiple-group estimation (not supported before M2).
    table : DataFrame, optional
        Edited parameter table from ``spec.parameter_table(data)``; takes precedence over
        ``spec``.
    item_names : list of str, optional
        Column names when ``data`` is an array.
    method : BAEM, optional
        EM settings (default ``BAEM()``).
    quadrature : Quadrature, optional
        Quadrature grid (default 61 equally spaced points on [-6, 6]).
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
    if group is not None:
        raise NotImplementedError("Multiple-group estimation is planned for milestone M2.")
    if se not in (None, "oakes", "hessian"):
        raise SpecificationError(f"Unknown se={se!r}; use 'oakes', 'hessian' or None.")
    method = BAEM() if method is None else method
    quadrature = Quadrature.equispaced() if quadrature is None else quadrature

    y = coerce_responses(data, spec.item_names, item_names=item_names)
    default = build_table(spec, y)
    final_table = validate_table(default if table is None else table, default)
    compiled = compile_problem(spec, y, final_table, quadrature)
    problem = compiled.problem

    out = _em(
        problem,
        jnp.asarray(compiled.eta0),
        tol=method.tol,
        max_iter=method.max_iter,
        n_newton=method.m_step_newton_steps,
    )
    eta = np.asarray(out.eta)
    converged = bool(out.converged)
    n_iter = int(out.n_iter)
    if not converged:
        warnings.warn(
            f"EM did not converge in {method.max_iter} iterations (tol={method.tol}).",
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

    history = pd.DataFrame(
        {
            "iter": np.arange(1, n_iter + 1),
            "loglik": np.asarray(out.loglik_history)[:n_iter],
            "max_change": np.asarray(out.change_history)[:n_iter],
        }
    )
    return FitResult(
        parameters=params,
        loglik=float(out.loglik),
        converged=converged,
        n_iter=n_iter,
        history=history,
        vcov=vcov,
        se_method=se,
        n_obs=compiled.n_obs,
        _model=spec,
        _eta=eta,
        _quadrature=quadrature,
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
