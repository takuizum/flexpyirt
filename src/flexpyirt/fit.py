"""Estimation entry point."""

from __future__ import annotations

import warnings
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from flexpyirt import _core, _robust_se
from flexpyirt._compile import LATENT_ITEM, compile_problem, prepare
from flexpyirt._precision import check_x64
from flexpyirt.errors import (
    ConvergenceWarning,
    FlexpyirtWarning,
    HeywoodWarning,
    SpecificationError,
)
from flexpyirt.items import ItemKernel, NominalKernel
from flexpyirt.latent import mixture_components, mixture_param_names
from flexpyirt.method import BAEM
from flexpyirt.model import Model
from flexpyirt.quadrature import Quadrature
from flexpyirt.result import FitResult

__all__ = ["fit"]

SEMethod = Literal["oakes", "hessian", "sandwich", "crossprod"]
_SE_METHODS: tuple[str, ...] = ("oakes", "hessian", "sandwich", "crossprod")

HEYWOOD_BOUND_TOL = 1e-4
"""Distance to a finite bound below which a free parameter is reported as a boundary solution."""
HEYWOOD_MAX_SLOPE = 20.0
"""Absolute slope above which a free slope is reported as diverging."""
NOMINAL_MIN_SLOPE = 1e-3
"""Slope norm below which a nominal item with a free slope is reported as degenerate."""
NOMINAL_MAX_PARAM = 50.0
"""Absolute ``ak`` or ``d`` above which a nominal item is reported as degenerate."""
MIXTURE_MIN_WEIGHT = 1e-3
"""Mixing proportion below which a ``NormalMixture`` component is reported as degenerate."""
MIXTURE_COLLAPSE_TOL = 1e-3
"""Mean and variance distance below which two mixture components are reported as collapsed."""
SQUAREM_FALLBACK_WARN_MIN = 5
"""Minimum full fallbacks to ``theta_2`` before warning about frequent SQUAREM fallbacks."""
SQUAREM_FALLBACK_WARN_RATE = 0.5
"""Fallback fraction above which a ``ConvergenceWarning`` is emitted on a converged fit."""

_em = jax.jit(_core.em, static_argnames=("max_iter", "n_newton"))
_squarem = jax.jit(_core.squarem, static_argnames=("max_iter", "n_newton"))
_information = {
    "oakes": jax.jit(_core.information_oakes),
    "hessian": jax.jit(_core.information_hessian),
    "crossprod": jax.jit(_robust_se.information_cross_product),
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
    priors), both by automatic differentiation (DESIGN.md §4.8). ``se="sandwich"`` and
    ``se="crossprod"`` use the cross-product of the per-pattern scores (DESIGN.md §17.5).

    Parameters
    ----------
    spec : Model
        Model specification.
    data : DataFrame or array-like
        Responses (persons × items) coded ``0..K-1``; NaN is missing (MAR). Columns must be
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
    se : {"oakes", "hessian", "sandwich", "crossprod"} or None
        Standard-error method, or ``None`` to skip. ``"oakes"`` (Oakes, 1999) and ``"hessian"``
        invert the observed information :math:`H`. ``"crossprod"`` inverts the cross-product of
        the per-pattern scores :math:`B=\sum_p c_p s_p s_p^\top` (mirt
        ``SE.type = "crossprod"``), and ``"sandwich"`` is the robust covariance
        :math:`H^{-1}BH^{-1}` (Huber, 1967; White, 1982; mirt ``SE.type = "sandwich"``), which
        stays valid when the model is misspecified. With priors, the prior curvature is added
        to :math:`B` as it is to :math:`H`. Both are unavailable for ``DavidianCurve`` and
        ``NormalMixture`` latent distributions.

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
    if se is not None and se not in _SE_METHODS:
        raise SpecificationError(
            f"Unknown se={se!r}; use 'oakes', 'hessian', 'sandwich', 'crossprod' or None."
        )
    if se in ("sandwich", "crossprod") and (spec.is_davidian or spec.is_mixture):
        latent_name = "DavidianCurve" if spec.is_davidian else "NormalMixture"
        raise SpecificationError(
            f"se={se!r} is not available with a {latent_name} latent distribution; "
            "use se='oakes' or se='hessian'."
        )
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
    eh_weights = (
        np.asarray(out.eh_weights, dtype=np.float64) if spec.is_empirical_histogram else None
    )
    problem_fin = (
        _core._with_eh_weights(problem, out.eh_weights) if spec.is_empirical_histogram else problem
    )
    if eh_weights is not None:
        _check_eh_reference_moments(problem, eh_weights[0])
    if se is not None and eta.shape[0] > 0:
        if se == "sandwich":
            bread = _invert_information(np.asarray(_information["oakes"](problem_fin, out.eta)))
            meat = np.asarray(_information["crossprod"](problem_fin, out.eta))
            vcov = bread @ (0.5 * (meat + meat.T)) @ bread
            vcov = 0.5 * (vcov + vcov.T)
        else:
            vcov = _invert_information(np.asarray(_information[se](problem_fin, out.eta)))
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
    if spec.is_mixture:
        _warn_mixture_degenerate(params, spec, quadrature)

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
        _method=method,
        _eh_weights=eh_weights,
    )


def _warn_heywood(params: pd.DataFrame, spec: Model) -> None:
    """Warn about boundary solutions among the free item parameters (DESIGN.md §15.6).

    A free parameter within ``HEYWOOD_BOUND_TOL`` of a finite bound (model domain, table bound,
    or attached prior support) or a free slope with ``|a| > HEYWOOD_MAX_SLOPE`` triggers one
    ``HeywoodWarning`` naming every affected item.
    """
    kernels = {name: spec.item_kernel(m) for name, m in spec.items}
    _warn_nominal_degenerate(params, kernels, spec.is_multigroup)
    items = params[(params["item"] != LATENT_ITEM) & params["free"]]
    if items.empty:
        return
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


def _warn_nominal_degenerate(
    params: pd.DataFrame, kernels: dict[str, ItemKernel], multigroup: bool
) -> None:
    r"""Warn about degenerate nominal items (DESIGN.md §16.6).

    A ``Nominal`` item with ``n_categories >= 3`` and a free slope is degenerate when its slope norm
    :math:`\lVert\mathbf a\rVert < 10^{-3}` (the item carries no information about
    :math:`\theta` and :math:`\mathrm{ak}_1..\mathrm{ak}_{K-2}` are not identified: the saddle
    that traps mirt's Newton–Raphson from its default starting values) or when some
    :math:`|\mathrm{ak}_k|` or :math:`|d_k|` exceeds ``NOMINAL_MAX_PARAM`` (diverging
    category parameters). One ``HeywoodWarning`` names every affected item.
    """
    rows = params[params["item"] != LATENT_ITEM]
    labels: list[str] = []
    items: set[str] = set()
    for (group, item), sub in rows.groupby(["group", "item"], sort=False):
        kernel = kernels[str(item)]
        if not isinstance(kernel, NominalKernel) or kernel.n_categories < 3:
            continue
        if not bool(sub.loc[sub["param"].isin(kernel.slope_params), "free"].any()):
            continue
        value = dict(zip(sub["param"], sub["estimate"].astype(float), strict=True))
        slope = float(np.sqrt(sum(value[p] ** 2 for p in kernel.slope_params)))
        others = [abs(value[p]) for p in kernel.param_names if p not in kernel.slope_params]
        largest = max(others)
        if slope < NOMINAL_MIN_SLOPE or largest > NOMINAL_MAX_PARAM:
            where = f" (group {group!r})" if multigroup else ""
            labels.append(f"{item}: |a|={slope:.3g}, max|ak|,|d|={largest:.4g}{where}")
            items.add(str(item))
    if labels:
        warnings.warn(
            f"Degenerate nominal items {sorted(items)}: {'; '.join(labels)}. A slope below "
            f"{NOMINAL_MIN_SLOPE} leaves the scoring coefficients unidentified, and scoring "
            f"coefficients or intercepts beyond ±{NOMINAL_MAX_PARAM} indicate diverging "
            "categories; check for sparse categories, consider other starting values, fix the "
            "scoring coefficients (e.g. ak_k = k, the GPCM), or use a simpler item model.",
            HeywoodWarning,
            stacklevel=4,
        )


def _warn_mixture_degenerate(params: pd.DataFrame, spec: Model, quadrature: Quadrature) -> None:
    r"""Warn about degenerate ``NormalMixture`` solutions (DESIGN.md §16.7).

    With the standardized components :math:`(\pi_c,\mu_c,\sigma_c^2)`, one ``HeywoodWarning``
    lists every component with :math:`\pi_c<` ``MIXTURE_MIN_WEIGHT`` (its location and scale
    are unidentified), :math:`\sigma_c` below half the smallest node spacing (narrower than the grid
    can resolve), or :math:`\mu_c` outside the grid, and every pair of components whose means and
    variances both differ by less than ``MIXTURE_COLLAPSE_TOL`` (a collapsed mixture).
    """
    names = list(mixture_param_names(spec.mixture_components))
    lat = params[params["item"] == LATENT_ITEM].set_index("param")["estimate"]
    pi, means, variances = (
        np.asarray(x, dtype=float)
        for x in mixture_components(jnp.asarray(lat.loc[names].to_numpy(dtype=float)))
    )
    nodes = np.sort(np.asarray(quadrature.nodes(), dtype=float))
    spacing = float(np.min(np.diff(nodes)))
    issues: list[str] = []
    for c in range(len(pi)):
        label = f"component {c + 1} (pi={pi[c]:.3g}, mean={means[c]:.4g}, var={variances[c]:.3g})"
        if pi[c] < MIXTURE_MIN_WEIGHT:
            issues.append(f"{label}: weight below {MIXTURE_MIN_WEIGHT}")
        if not variances[c] >= (0.5 * spacing) ** 2:
            issues.append(f"{label}: sd below half the grid spacing {spacing:.3g}")
        if not nodes[0] <= means[c] <= nodes[-1]:
            issues.append(f"{label}: mean outside the grid [{nodes[0]:g}, {nodes[-1]:g}]")
    for c in range(len(pi)):
        for c2 in range(c + 1, len(pi)):
            if (
                abs(means[c] - means[c2]) < MIXTURE_COLLAPSE_TOL
                and abs(variances[c] - variances[c2]) < MIXTURE_COLLAPSE_TOL
            ):
                issues.append(f"components {c + 1} and {c2 + 1} coincide")
    if issues:
        warnings.warn(
            "Degenerate normal-mixture solution: "
            + "; ".join(issues)
            + ". The mixture is not identified there (a component can be dropped or its "
            "parameters are not determined by the data); try other starting values (means=, "
            "variances=, weights=) or fewer components.",
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


EH_MOMENT_TOL = 1e-8
"""Tolerance on the standardized reference EH moments checked after fitting (DESIGN §15.34)."""


def _check_eh_reference_moments(problem: _core.CompiledProblem, w_ref: np.ndarray) -> None:
    """Warn if the reference EH histogram misses its mean-0 / variance-1 constraints."""
    lat = problem.latent
    if not (lat.standardize_mean or lat.standardize_var):
        return
    nodes = np.asarray(problem.nodes, dtype=np.float64).reshape(-1)
    mean = float(np.sum(w_ref * nodes))
    second = float(np.sum(w_ref * nodes**2))
    bad = abs(float(np.sum(w_ref)) - 1.0) > EH_MOMENT_TOL or abs(mean) > EH_MOMENT_TOL
    if lat.standardize_var:
        bad = bad or abs(second - 1.0) > EH_MOMENT_TOL
    if bad:
        warnings.warn(
            "The reference group's empirical histogram does not meet its identification "
            f"constraints (sum={np.sum(w_ref):.10g}, mean={mean:.3g}, E[theta^2]={second:.10g}); "
            "the moment-constrained M-step did not converge. Widen the quadrature range.",
            ConvergenceWarning,
            stacklevel=3,
        )
