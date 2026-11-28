r"""Projective IRT: a unidimensional model on one factor of a multidimensional fit (DESIGN.md §28).

A multidimensional model (typically a bifactor model) is projected onto one of its factors by
integrating the other factors out of every item response function (Ip, 2010). The projected
item parameters describe each item on the target factor alone, with the local dependence
caused by the nuisance factors absorbed into the item parameters. :func:`project` estimates
them in three ways, for 2PL and GRM items (the logistic kernels the methods were derived and
checked for):

* ``"MML"``: maximum marginal likelihood on the marginal expected table of the parent model
  (MML-PIRT; Chalmers, Falk & Reise, 2026), with standard errors from the information of the
  projected objective.
* ``"logistic_kernel"``: the closed-form logistic kernel approximation (Ip, 2010).
* ``"normal_ogive"``: the closed-form approximation through the normal-ogive (factor-analytic)
  loadings with :math:`D=1.702` (Stucky, Thissen & Edelen, 2013).
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from flexpyirt import _core, _projection
from flexpyirt._compile import TABLE_COLUMNS, compile_problem, prepare
from flexpyirt._precision import check_x64
from flexpyirt.errors import ConvergenceWarning, SpecificationError
from flexpyirt.items import ItemKernel
from flexpyirt.latent import Normal
from flexpyirt.model import LATENT_ITEM, Model
from flexpyirt.result import FitResult

__all__ = ["project"]

ProjectionMethod = Literal["MML", "logistic_kernel", "normal_ogive"]
_METHODS: tuple[str, ...] = ("MML", "logistic_kernel", "normal_ogive")
ProjectionSE = Literal["oakes", "complete", "hessian"]
_SE_METHODS: tuple[str, ...] = ("oakes", "complete", "hessian")
_CONSTANTS = {
    "logistic_kernel": _projection.LOGISTIC_KERNEL_CONSTANT,
    "normal_ogive": _projection.NORMAL_OGIVE_CONSTANT,
}
_ITEM_MODELS = ("2PL", "GRM")
"""Item models that can be projected: the logistic kernels of Ip (2010), Stucky et al. (2013)
and Chalmers et al. (2026) (M2PL to 2PL, MGRM to GRM)."""
_INTERCEPT = re.compile(r"d[0-9]*")
"""Intercepts rescaled by the closed-form projections (``d`` and the GRM's ``d1..d{K-1}``)."""

MML_TOL = 1e-10
"""Largest absolute parameter change at which the projected Newton iterations stop."""
MML_MAX_ITER = 500
"""Maximum number of projected Newton iterations (``MML_NEWTON_STEPS`` steps each)."""
MML_NEWTON_STEPS = 5
"""Newton steps per item in one projected iteration."""
_HISTORY_COLUMNS = (
    "iter",
    "loglik",
    "log_posterior",
    "max_change",
    "em_evaluations",
    "safeguard",
)

_m_step_items = jax.jit(_core._m_step_items, static_argnames=("n_steps",))
_marginal_item_counts = jax.jit(_projection.marginal_item_counts, static_argnames=("factor",))
_marginal_loglik = jax.jit(_core.marginal_loglik)


def project(
    result: FitResult,
    data: Any,
    *,
    method: ProjectionMethod = "MML",
    factor: str | None = None,
    item_names: list[str] | None = None,
    se: ProjectionSE | None = "oakes",
) -> FitResult:
    r"""Project a multidimensional model onto one factor (projective IRT).

    Integrating the nuisance factors :math:`\boldsymbol\theta_{-f}` out of item :math:`j`'s
    response function gives its marginal trace on the target factor :math:`\theta_f`
    (Ip, 2010),

    .. math::
        \bar P_j(y\mid\theta_f)=\int P_j(y\mid\theta_f,\boldsymbol\theta_{-f})\,
        g(\boldsymbol\theta_{-f}\mid\theta_f)\,d\boldsymbol\theta_{-f},

    which the projected (unidimensional) item model :math:`P_j(y\mid\theta_f;\xi^*_j)` of the
    same family approximates. The projected parameters :math:`\xi^*_j` keep the scale of the
    target factor and absorb the local dependence that the nuisance factors cause, so a
    unidimensional model with them scores :math:`\theta_f` (for example the general factor of a
    bifactor model) on the parent's metric.

    * ``method="MML"`` (Chalmers, Falk & Reise, 2026): the parent's E-step at its estimates
      :math:`\hat\eta` is summed over the nuisance factors,
      :math:`\bar r_{jk}(\theta_q)=\sum_{\boldsymbol\theta_m:\theta_{mf}=\theta_q}
      \bar r_{jk}(\boldsymbol\theta_m;\hat\eta)`, and each item maximizes
      :math:`\sum_q\sum_k\bar r_{jk}(\theta_q)\log P_{jk}(\theta_q;\xi^*_j)` once (the table is
      not updated).
    * ``method="logistic_kernel"`` (Ip, 2010) and ``method="normal_ogive"`` (Stucky, Thissen &
      Edelen, 2013): closed forms
      :math:`a^*=s\,\mathbf a^\top\boldsymbol\Phi_{\cdot f}` and :math:`d^*_k=s\,d_k` with
      :math:`s=(1+c^2v)^{-1/2}`,
      :math:`v=\mathbf a^\top\boldsymbol\Phi\mathbf a-(\mathbf a^\top\boldsymbol\Phi_{\cdot f})^2`
      (the variance of the nuisance part given :math:`\theta_f`), and :math:`c=16\sqrt3/(15\pi)`
      or :math:`c=1/1.702`.

    Parameters
    ----------
    result : FitResult
        A single-group multidimensional fit with 2PL and GRM items and a ``Normal`` latent
        distribution: a bifactor or two-tier model (:meth:`Model.with_bifactor`,
        :meth:`Model.with_two_tier`) or a confirmatory model (:meth:`Model.with_factors`).
    data : DataFrame or array-like
        The responses ``result`` was fitted to. ``"MML"`` needs them for the E-step; every method
        uses them to lay out the projected model and for its log-likelihood.
    method : {"MML", "logistic_kernel", "normal_ogive"}
        Estimator (mirt ``pirt(estimator = "MML" / "LKA" / "FA")``).
    factor : str, optional
        Factor to project onto (mirt ``project``). Default: the first factor, the general factor
        of a bifactor model. In a bifactor or two-tier model it must be a primary factor. Its
        mean must be fixed at 0 and its variance at 1.
    item_names : list of str, optional
        Column names when ``data`` is an array.
    se : {"oakes", "complete", "hessian"} or None
        Covariance of the projected parameters, or ``None`` to skip. For ``"MML"``, with
        :math:`I_{\bar r}` the negative Hessian of the objective above (the marginal table held
        fixed), :math:`I_c` the same on the projected model's own E-step and :math:`I_o` the
        projected model's observed information: ``"complete"`` inverts :math:`I_{\bar r}`
        (mirt ``SE.type = "complete"``; treats the table as known and is the smallest),
        ``"hessian"`` inverts :math:`I_o` (mirt ``"Richardson"``), and ``"oakes"`` inverts
        :math:`I_{\bar r}-(I_c-I_o)`, Oakes' (1999) identity with the cross term (the missing
        information) from the projected model (mirt's default ``"Oakes"``). None of them uses
        the parent's covariance. For the closed forms any value other than ``None`` gives the
        delta method on the parent's covariance (``None`` when the parent has none).

    Returns
    -------
    FitResult
        The projected unidimensional model with :math:`N(0,1)` on the target factor, on the
        parent's one-dimensional grid. Every parameter is free or fixed as in a default
        unidimensional model (also when the parent fixed it, so a parent fixed at an item
        bank's values can be projected); priors are dropped.
        ``loglik`` is the marginal log-likelihood of ``data`` under the projected model; it is
        not maximized, so likelihood-ratio tests against the parent do not apply. ``se_method``
        is ``"projection_oakes"``, ``"projection_complete"``, ``"projection_hessian"`` (MML) or
        ``"projection_delta"`` (closed forms). There are no EM iterations:
        ``n_iter`` is 0, the history is empty, and ``converged`` says whether the Newton
        iterations of ``"MML"`` converged (always true for the closed forms).

    Raises
    ------
    SpecificationError
        For a unidimensional, multiple-group, weighted or non-Normal parent, a factor that cannot
        be projected onto, or an item model other than 2PL or GRM.

    Notes
    -----
    In bifactor models the closed forms equal mirt's ``pirt(estimator = "LKA")`` and
    ``pirt(estimator = "FA")``. With correlated factors they use the conditional variance above;
    mirt's ``"LKA"`` ignores the correlation and its ``"FA"`` uses the pattern loadings, so the
    values differ there.

    References
    ----------
    Chalmers, R. P., Falk, C. F., & Reise, S. P. (2026). A general approach for estimating
    projective IRT models. *Applied Psychological Measurement*.

    Ip, E. H. (2010). Empirically indistinguishable multidimensional IRT and locally dependent
    unidimensional item response models. *British Journal of Mathematical and Statistical
    Psychology*, 63(2), 395–416.

    Oakes, D. (1999). Direct calculation of the information matrix via the EM algorithm.
    *Journal of the Royal Statistical Society: Series B*, 61(2), 479–482.

    Stucky, B. D., Thissen, D., & Edelen, M. O. (2013). Using logistic approximations of
    marginal trace lines to develop short assessments. *Applied Psychological Measurement*,
    37(1), 41–57.
    """
    check_x64()
    if se is not None and se not in _SE_METHODS:
        raise SpecificationError(f"Unknown se={se!r}; use 'oakes', 'complete', 'hessian' or None.")
    if method not in _METHODS:
        raise SpecificationError(
            f"Unknown method={method!r}; use 'MML', 'logistic_kernel' or 'normal_ogive'."
        )
    parent = _check_parent(result)
    target = _target_factor(parent, factor)
    _check_target_scale(result, target)

    proj_spec = Model(items=parent.items, latent=Normal())
    prep = prepare(parent, data, group=None, item_names=item_names, table=None)
    proj_table = prepare(proj_spec, data, group=None, item_names=item_names, table=None).table
    proj_table = _projected_layout(result, proj_table, target)

    eta_parent = jnp.asarray(result._eta)
    if method == "MML":
        start = _closed_form_values(
            result, target, _projection.LOGISTIC_KERNEL_CONSTANT, _row_keys(proj_table)
        )
        proj_table["value"] = _starting_values(proj_table, start(eta_parent))
        compiled = compile_problem(proj_spec, prep.y, prep.group, proj_table, result._quadrature)
        problem = compiled.problem
        parent_problem = compile_problem(
            parent, prep.y, prep.group, result.parameters[list(TABLE_COLUMNS)], result._quadrature
        ).problem
        nodes_1d = jnp.asarray(result._quadrature.nodes(), dtype=jnp.float64)
        r_marg, n_marg = _marginal_item_counts(parent_problem, eta_parent, target, nodes_1d)
        counts = _align_counts(parent_problem, problem, r_marg, n_marg)
        eta, converged = _maximize(problem, jnp.asarray(compiled.eta0), counts)
        vcov = None
        if se is not None and eta.shape[0] > 0:
            vcov = _invert(_mml_information(problem, eta, counts, se))
        se_method = None if se is None else f"projection_{se}"
    else:
        values = _closed_form_values(result, target, _CONSTANTS[method], _row_keys(proj_table))
        full = np.asarray(values(eta_parent))
        free = proj_table["free"].to_numpy()
        fid = proj_table["free_id"].to_numpy()
        order = np.argsort(fid[free])
        free_rows = np.flatnonzero(free)[order]
        proj_table["value"] = np.where(free, proj_table["value"].to_numpy(dtype=float), full)
        eta = jnp.asarray(full[free_rows])
        problem = compile_problem(
            proj_spec, prep.y, prep.group, proj_table, result._quadrature
        ).problem
        converged = True
        vcov = None
        if se is not None and result.vcov is not None and free_rows.size > 0:
            jac = np.asarray(jax.jacfwd(lambda e: values(e)[free_rows])(eta_parent))
            vcov = jac @ np.asarray(result.vcov) @ jac.T
            vcov = 0.5 * (vcov + vcov.T)
        se_method = "projection_delta" if vcov is not None else None

    eta_np = np.asarray(eta)
    se_values = (
        np.full(eta_np.shape[0], np.nan)
        if vcov is None
        else np.sqrt(np.clip(np.diag(vcov), 0.0, None))
    )
    params = proj_table.copy()
    free = params["free"].to_numpy()
    fid = params["free_id"].to_numpy()
    estimate = params["value"].to_numpy(dtype=float).copy()
    se_column = np.full(len(params), np.nan)
    estimate[free] = eta_np[fid[free]]
    se_column[free] = se_values[fid[free]]
    params["estimate"] = estimate
    params["se"] = se_column
    loglik = float(_marginal_loglik(problem, eta))
    return FitResult(
        parameters=params,
        loglik=loglik,
        log_posterior=loglik,
        converged=converged,
        n_iter=0,
        n_em_evaluations=0,
        n_safeguards=0,
        history=pd.DataFrame({name: [] for name in _HISTORY_COLUMNS}),
        vcov=vcov,
        se_method=se_method,
        n_obs=int(prep.y.shape[0]),
        _model=proj_spec,
        _eta=eta_np,
        _quadrature=result._quadrature,
        _method=result._method,
    )


# ---- checks -----------------------------------------------------------------------------------


def _check_parent(result: FitResult) -> Model:
    """Return the parent's specification after checking that it can be projected."""
    if not isinstance(result, FitResult):
        raise SpecificationError(f"project() needs a FitResult, got {type(result).__name__}.")
    model = result._model
    if not model.is_multidimensional:
        raise SpecificationError("project() needs a multidimensional model; this one has 1 factor.")
    if model.is_multigroup:
        raise SpecificationError("project() supports single-group models only.")
    if model.is_empirical_histogram or model.is_davidian or model.is_mixture:
        raise SpecificationError("project() supports Normal latent distributions only.")
    if result._sum_weights is not None:
        raise SpecificationError(
            "project() does not support models fitted with sampling weights (weights=)."
        )
    unsupported = sorted({m.name for _, m in model.items if m.name not in _ITEM_MODELS})
    if unsupported:
        raise SpecificationError(
            f"project() supports 2PL and GRM items only, not {unsupported}: projective IRT "
            "approximates the marginal trace of a logistic kernel (DESIGN.md §28.1)."
        )
    return model


def _target_factor(model: Model, factor: str | None) -> int:
    """Return the 0-based index of the factor to project onto."""
    names = model.factor_names
    if factor is None:
        return 0
    if factor not in names:
        raise SpecificationError(f"Unknown factor {factor!r}; the model's factors are {names}.")
    index = names.index(factor)
    if model.is_reduced and index >= model.n_primary:
        kind = "general" if model.bifactor else "primary"
        raise SpecificationError(
            f"Factor {factor!r} is a specific factor; a bifactor or two-tier model can be "
            f"projected onto its {kind} factor{'s' if model.n_primary > 1 else ''} "
            f"{names[: model.n_primary]} only."
        )
    return index


def _check_target_scale(result: FitResult, target: int) -> None:
    """Raise unless the target factor's mean is 0 and its variance 1 (the projected metric)."""
    rows = result.parameters[result.parameters["item"] == LATENT_ITEM]
    estimate = dict(zip(rows["param"], rows["estimate"].to_numpy(dtype=float), strict=True))
    mean = float(estimate[f"mean_{target + 1}"])
    var = float(estimate[f"var_{target + 1}"])
    if mean != 0.0 or var != 1.0:
        raise SpecificationError(
            f"The target factor must have mean 0 and variance 1 (got mean {mean:.6g}, variance "
            f"{var:.6g}); the projected model is on the N(0, 1) metric."
        )


# ---- tables -----------------------------------------------------------------------------------


def _parent_name(kernel: ItemKernel, proj_kernel: ItemKernel, param: str, target: int) -> str:
    """Name of the parent parameter that a projected parameter comes from."""
    if param in proj_kernel.slope_params:
        return kernel.slope_params[target]
    return param


def _projected_layout(result: FitResult, table: pd.DataFrame, target: int) -> pd.DataFrame:
    """Set ``value``, ``free`` and ``free_id`` of the projected table (DESIGN.md §28.2).

    Every parameter has the projected model's default status (``table`` is that model's default
    table), whether or not it was fixed in the parent, so a parent fixed at an item bank's values
    is projected too; the parent's estimates are the starting values. Parameters are not shared
    across items and priors are dropped.
    """
    parent = result._model
    rows = result.parameters
    parent_value = {
        (str(i), str(p)): float(v)
        for i, p, v in zip(rows["item"], rows["param"], rows["estimate"], strict=True)
    }
    models = dict(parent.items)
    free = table["free"].to_numpy(dtype=bool).copy()
    values = table["value"].to_numpy(dtype=float).copy()
    for pos, (item, param) in enumerate(zip(table["item"], table["param"], strict=True)):
        if item == LATENT_ITEM:
            continue
        kernel = parent.item_kernel(models[item])
        proj_kernel = models[item].kernel
        values[pos] = parent_value[(item, _parent_name(kernel, proj_kernel, param, target))]
    out = table.copy()
    out["free"] = free
    out["value"] = values
    out["free_id"] = -1
    out.loc[free, "free_id"] = np.arange(int(free.sum()))
    out["prior"] = None
    return out


def _closed_form_values(
    result: FitResult, target: int, constant: float, keys: list[tuple[str, str]]
) -> Callable[[jax.Array], jax.Array]:
    r"""Return ``eta_parent -> values of the projected rows keys`` from the closed forms.

    ``keys`` are the ``(item, param)`` rows of the projected table; latent rows get mean 0 and
    variance 1. The slope is :math:`a^*` and the intercepts are scaled by :math:`s`.
    """
    parent = result._model
    rows = result.parameters
    index = {
        (str(i), str(p)): k
        for k, (i, p) in enumerate(zip(rows["item"], rows["param"], strict=True))
    }
    fixed = jnp.asarray(rows["value"].to_numpy(dtype=float))
    free_rows = np.flatnonzero(rows["free"].to_numpy())
    free_ids = rows["free_id"].to_numpy()[free_rows]
    n_factors = parent.n_factors
    latent_names = set(parent.latent_parameter_names())
    models = dict(parent.items)
    kernels = {name: parent.item_kernel(m) for name, m in parent.items}

    def cov_matrix(full: jax.Array) -> jax.Array:
        var = jnp.stack([full[index[(LATENT_ITEM, f"var_{m + 1}")]] for m in range(n_factors)])
        sd = jnp.sqrt(var)
        cov = jnp.diag(var)
        for i in range(n_factors):
            for j in range(i + 1, n_factors):
                name = f"cor_{i + 1}{j + 1}"
                if name in latent_names:
                    c = full[index[(LATENT_ITEM, name)]] * sd[i] * sd[j]
                    cov = cov.at[i, j].set(c).at[j, i].set(c)
        return cov

    def values(eta: jax.Array) -> jax.Array:
        full = fixed.at[free_rows].set(eta[free_ids]) if free_rows.size else fixed
        phi = cov_matrix(full)
        projected: dict[str, tuple[jax.Array, jax.Array]] = {}
        for name, kernel in kernels.items():
            slopes = full[np.asarray([index[(name, s)] for s in kernel.slope_params])]
            projected[name] = _projection.kernel_scale(slopes, phi, target, constant)
        out = []
        for item, param in keys:
            if item == LATENT_ITEM:
                out.append(jnp.asarray(0.0 if param == "mean" else 1.0))
                continue
            kernel = kernels[item]
            source = full[index[(item, _parent_name(kernel, models[item].kernel, param, target))]]
            a_star, scale = projected[item]
            if param in models[item].kernel.slope_params:
                out.append(a_star)
            elif _INTERCEPT.fullmatch(param):
                out.append(scale * source)
            else:
                out.append(source)
        return jnp.stack(out)

    return values


def _row_keys(table: pd.DataFrame) -> list[tuple[str, str]]:
    """``(item, param)`` of every row of a table, in order."""
    return [(str(i), str(p)) for i, p in zip(table["item"], table["param"], strict=True)]


def _starting_values(table: pd.DataFrame, values: jax.Array) -> np.ndarray:
    """Use ``values`` (projected-table row order) as starting values of the free rows."""
    start = np.asarray(values, dtype=float)
    out = table["value"].to_numpy(dtype=float).copy()
    free = table["free"].to_numpy()
    out[free] = start[free]
    return out


# ---- MML --------------------------------------------------------------------------------------


def _align_counts(
    parent: _core.CompiledProblem,
    problem: _core.CompiledProblem,
    r_marg: tuple[jax.Array, ...],
    n_marg: jax.Array,
) -> _core.ExpectedCounts:
    """Reorder the parent's per-family marginal tables to the projected problem's families."""
    per_item: dict[int, jax.Array] = {}
    for fam, r in zip(parent.families, r_marg, strict=True):
        for pos, col in enumerate(np.asarray(fam.columns)):
            per_item[int(col)] = r[:, pos]
    r_out = tuple(
        jnp.stack([per_item[int(col)] for col in np.asarray(fam.columns)], axis=1)
        for fam in problem.families
    )
    return _core.ExpectedCounts(r_out, n_marg)


def _maximize(
    problem: _core.CompiledProblem, eta0: jax.Array, counts: _core.ExpectedCounts
) -> tuple[jax.Array, bool]:
    """Maximize the projected objective with the item-wise masked Newton of the M-step (§4.3)."""
    eta = eta0
    for _ in range(MML_MAX_ITER):
        new = _m_step_items(problem, eta, counts, n_steps=MML_NEWTON_STEPS)
        change = float(jnp.max(jnp.abs(new - eta))) if eta.shape[0] else 0.0
        eta = new
        if change < MML_TOL:
            return eta, True
    warnings.warn(
        f"The projected MML objective did not converge in {MML_MAX_ITER} iterations.",
        ConvergenceWarning,
        stacklevel=3,
    )
    return eta, False


_information_oakes = jax.jit(_core.information_oakes)


def _complete_information(
    problem: _core.CompiledProblem, eta: jax.Array, counts: _core.ExpectedCounts
) -> np.ndarray:
    r""":math:`-\nabla^2_\eta Q(\eta)` with the expected table ``counts`` held fixed."""
    hess = jax.hessian(lambda e: _core.expected_complete_loglik(problem, e, counts))(eta)
    return -np.asarray(hess)


def _mml_information(
    problem: _core.CompiledProblem,
    eta: jax.Array,
    counts: _core.ExpectedCounts,
    se: str,
) -> np.ndarray:
    r"""Information of the MML-PIRT estimates (DESIGN.md §28.3).

    With :math:`I_{\bar r}=-\nabla^2Q^*` on the parent's marginal table,
    :math:`I_c=-\nabla^2Q(\eta\mid\eta)` on the projected model's own E-step and
    :math:`I_o` its observed information (Oakes, 1999):

    * ``"complete"``: :math:`I_{\bar r}` (mirt ``SE.type = "complete"``);
    * ``"hessian"``: :math:`I_o` (mirt ``"Richardson"``);
    * ``"oakes"``: :math:`I_{\bar r}-(I_c-I_o)`, Oakes' identity with the second-derivative term
      from the marginal table and the cross term (the missing information) from the projected
      model (mirt's default ``"Oakes"``, also its ``"Louis"``).
    """
    fixed_table = _complete_information(problem, eta, counts)
    if se == "complete":
        return fixed_table
    observed = np.asarray(_information_oakes(problem, eta))
    if se == "hessian":
        return observed
    own_counts, _ = _core.e_step(problem, eta)
    missing = _complete_information(problem, eta, own_counts) - observed
    info: np.ndarray = fixed_table - missing
    return info


def _invert(info: np.ndarray) -> np.ndarray:
    from flexpyirt.fit import _invert_information

    vcov: np.ndarray = _invert_information(info)
    return vcov
