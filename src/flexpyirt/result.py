"""Estimation result (Result layer, DESIGN.md §2 ④)."""

from __future__ import annotations

import html
import os
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from functools import cache
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array

from flexpyirt import _core, _information, _itemfit, _local_dependence, _m2, _scoring
from flexpyirt._compile import (
    LATENT_ITEM,
    coerce_groups,
    coerce_responses,
    compile_problem,
    eh_param_count,
)
from flexpyirt._json import dump_fit_result, load_fit_result
from flexpyirt._precision import check_x64
from flexpyirt.errors import FlexpyirtWarning, SpecificationError
from flexpyirt.items import ItemKernel
from flexpyirt.latent import (
    davidian_param_names,
    eh_moments,
    latent_param_names,
    mixture_components,
    mixture_param_names,
)
from flexpyirt.method import BAEM
from flexpyirt.quadrature import Quadrature

if TYPE_CHECKING:
    from flexpyirt._dif import DIFScheme, PAdjust
    from flexpyirt.dimensionality import JSIMethod, JSIResult, LocalDependenceResult
    from flexpyirt.model import Model

__all__ = ["FitResult"]

_JSI_APPROXIMATION_MIN_ITEMS = 10
"""Below this many items ``jsi(method="newton" | "one_step")`` warns (DESIGN.md §16.17)."""

ScoreMethod = Literal["EAP", "MAP", "ML", "WLE", "PV", "EAPsum"]
_SCORE_METHODS: tuple[str, ...] = ("EAP", "MAP", "ML", "WLE", "PV", "EAPsum")

_eap = jax.jit(_core.eap)
_map = jax.jit(_scoring.map_scores)
_ml = jax.jit(_scoring.ml_scores)
_wle = jax.jit(_scoring.wle_scores)
_pv = jax.jit(_scoring.plausible_values, static_argnames=("n_draws",))
_summed_score_eap = jax.jit(_information.summed_score_eap)


def _is_prng_key(key: Any) -> bool:
    """Return True iff ``key`` is a JAX PRNG key (``jax.random.key`` or ``PRNGKey``)."""
    if not isinstance(key, jax.Array):
        return False
    if jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
        return key.ndim == 0
    return bool(key.dtype == jnp.uint32 and key.shape == (2,))


@jax.jit
def _mixture_val_and_jac(params: Array) -> tuple[Array, Array]:
    """Standardized mixture components ``(pi, means, variances)`` stacked, and their Jacobian."""

    def fn(p: Array) -> Array:
        return jnp.concatenate(mixture_components(p))

    return fn(params), jax.jacfwd(fn)(params)


def _mixture_derived(
    names: Sequence[str], est: np.ndarray, fids: np.ndarray, vcov: np.ndarray | None
) -> dict[str, float]:
    r"""Standardized components of a fitted ``NormalMixture`` with delta-method SEs (§16.7).

    Returns ``pi_c``, ``mean_c``, ``var_c`` (:math:`c=1,\dots,k`) and their ``*_se`` from
    :math:`J\Sigma J^\top` restricted to the free latent parameters (Jacobian by JAX forward-mode
    autodiff; Oehlert, 1992). SEs are NaN without ``vcov``.
    """
    k = (len(names) + 1) // 3
    vals, jac = _mixture_val_and_jac(jnp.asarray(est, dtype=jnp.float64))
    vals_np = np.asarray(vals, dtype=float)
    jac_np = np.asarray(jac, dtype=float)
    free = fids >= 0
    ses = np.full(vals_np.shape, np.nan)
    if vcov is not None and free.any():
        j_free = jac_np[:, free]
        v_free = vcov[np.ix_(fids[free], fids[free])]
        ses = np.sqrt(np.clip(np.diag(j_free @ v_free @ j_free.T), 0.0, None))
    out: dict[str, float] = {}
    for block, label in enumerate(("pi", "mean", "var")):
        for c in range(k):
            out[f"{label}_{c + 1}"] = float(vals_np[block * k + c])
            out[f"{label}_{c + 1}_se"] = float(ses[block * k + c])
    return out


@cache
def _irt_val_and_jac(kernel: ItemKernel) -> Callable[[Array], tuple[Array, Array]]:
    """Return a jitted ``(irt_thresholds, jacfwd(irt_thresholds))`` pair for ``kernel``."""
    jac_fn = jax.jacfwd(kernel.irt_thresholds)

    @jax.jit
    def fn(params: Array) -> tuple[Array, Array]:
        return kernel.irt_thresholds(params), jac_fn(params)

    return fn


def _delta_irt_thresholds(
    kernel: ItemKernel,
    xi: np.ndarray,
    fids: np.ndarray,
    vcov: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute display IRT thresholds and delta-method SEs for one item (DESIGN.md §7).

    For display transformation :math:`g(\xi)` with Jacobian :math:`J = \partial g / \partial\xi`
    computed via JAX forward-mode automatic differentiation, the asymptotic variance of each
    threshold is

    .. math::
        \operatorname{Var}\big(g_m(\hat\xi)\big) \approx J_m\,\Sigma_\xi\,J_m^\top,

    restricted to the free parameters entering :math:`g_m` (fixed parameters contribute zero
    variance; if no parameter entering :math:`g_m` is free, or ``vcov`` is ``None``, the SE is
    NaN). Casella & Berger (2002); Oehlert (1992).
    """
    vals_j, jac_j = _irt_val_and_jac(kernel)(jnp.asarray(xi, dtype=jnp.float64))
    b_vals = np.asarray(vals_j, dtype=float)
    jac = np.asarray(jac_j, dtype=float)
    b_ses = np.full(b_vals.shape, np.nan, dtype=float)
    if vcov is None:
        return b_vals, b_ses
    for m, deps in enumerate(kernel.irt_dependency_indices):
        active = [p for p in deps if fids[p] >= 0]
        if not active:
            continue
        active_fids = [int(fids[p]) for p in active]
        j_m = jac[m, active]
        v_m = vcov[np.ix_(active_fids, active_fids)]
        var_m = float(j_m @ v_m @ j_m)
        b_ses[m] = np.nan if np.isnan(var_m) else float(np.sqrt(max(var_m, 0.0)))
    return b_vals, b_ses


_PRIORS_IC_NOTE = (
    "Note: Priors are active (MAP); AIC and BIC use the marginal log-likelihood evaluated at the "
    "MAP estimate rather than the MLE."
)

_HTML_STYLE = """<style>
.flexpyirt-result {
  font-family: system-ui, -apple-system, sans-serif;
  font-size: 13px;
  line-height: 1.45;
  margin: 0.5em 0;
}
.flexpyirt-result h4 {
  margin: 0.8em 0 0.35em 0;
  font-size: 13px;
  font-weight: 600;
}
.flexpyirt-result .flexpyirt-header {
  font-size: 15px;
  font-weight: 600;
  margin-bottom: 0.5em;
}
.flexpyirt-result .flexpyirt-meta {
  border-collapse: collapse;
  margin-bottom: 0.6em;
}
.flexpyirt-result .flexpyirt-meta th {
  text-align: left;
  font-weight: 600;
  padding: 2px 12px 2px 0;
  opacity: 0.8;
  white-space: nowrap;
}
.flexpyirt-result .flexpyirt-meta td {
  text-align: left;
  padding: 2px 16px 2px 0;
}
.flexpyirt-result .flexpyirt-note {
  margin: 0.4em 0 0.6em 0;
  padding: 6px 10px;
  border-left: 3px solid #d29922;
  background: rgba(210, 153, 34, 0.12);
  font-size: 12px;
}
.flexpyirt-result .flexpyirt-table-wrap {
  overflow-x: auto;
  max-height: 420px;
  margin-bottom: 0.6em;
}
.flexpyirt-result table.flexpyirt-table {
  border-collapse: collapse;
  font-variant-numeric: tabular-nums;
}
.flexpyirt-result table.flexpyirt-table th,
.flexpyirt-result table.flexpyirt-table td {
  padding: 3px 8px;
  border-bottom: 1px solid rgba(127, 127, 127, 0.25);
  white-space: nowrap;
}
.flexpyirt-result table.flexpyirt-table th {
  background: rgba(127, 127, 127, 0.10);
  font-weight: 600;
  text-align: right;
}
.flexpyirt-result table.flexpyirt-table th.text,
.flexpyirt-result table.flexpyirt-table td.text {
  text-align: left;
}
.flexpyirt-result table.flexpyirt-table td.num {
  text-align: right;
}
.flexpyirt-result table.flexpyirt-table td.na {
  text-align: right;
  opacity: 0.55;
}
</style>"""


def _frame_to_html_table(df: pd.DataFrame, *, digits: int = 4) -> str:
    """Render a DataFrame as an escaped plain HTML table with tabular numeric formatting."""
    is_num = [pd.api.types.is_numeric_dtype(df[col].dtype) for col in df.columns]
    head_cells = [
        f'<th class="{"num" if num else "text"}">{html.escape(str(col))}</th>'
        for col, num in zip(df.columns, is_num, strict=True)
    ]
    body_rows: list[str] = []
    for row in df.itertuples(index=False, name=None):
        cells: list[str] = []
        for val, num in zip(row, is_num, strict=True):
            if num:
                fval = float(val)
                if np.isnan(fval):
                    cells.append('<td class="na">NaN</td>')
                else:
                    cells.append(f'<td class="num">{fval:.{digits}f}</td>')
            else:
                cells.append(f'<td class="text">{html.escape(str(val))}</td>')
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    return (
        '<div class="flexpyirt-table-wrap"><table class="flexpyirt-table">'
        f"<thead><tr>{''.join(head_cells)}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )


@dataclass(frozen=True)
class FitResult:
    """Result of :func:`flexpyirt.fit`.

    Attributes
    ----------
    parameters : DataFrame
        The parameter table (DESIGN.md §5.1) with ``estimate`` and ``se`` columns added.
    loglik : float
        Marginal log-likelihood at the estimates (likelihood only, excluding priors).
    log_posterior : float
        Log posterior (marginal log-likelihood plus sum of free-parameter log priors) at the
        estimates; equals ``loglik`` when no priors are active (DESIGN.md §4.7).
    converged : bool
        Whether the EM convergence criterion was met.
    n_iter : int
        Number of accepted EM (or SQUAREM) iterations.
    n_em_evaluations : int
        Total number of EM map evaluations (E-step + M-step). Equal to ``n_iter`` for plain EM,
        and 2 or 3 per accepted iterate for SQUAREM.
    n_safeguards : int
        Number of SQUAREM iterations on which the monotonicity/feasibility safeguard fired (0 for
        plain EM).
    history : DataFrame
        Columns ``iter``, ``loglik`` (at the parameters entering the iteration),
        ``log_posterior`` (at the parameters entering the iteration; equals ``loglik`` without
        priors and is non-decreasing across iterations), ``max_change``, ``em_evaluations``
        (cumulative EM map evaluations through the iteration), and ``safeguard`` (whether the
        SQUAREM safeguard fired on that iteration).
    vcov : ndarray or None
        Covariance matrix of the free parameters in ``free_id`` order (``None`` if ``se=None``).
    se_method : str or None
        How ``vcov`` was computed.
    n_obs : int
        Number of respondents.
    """

    parameters: pd.DataFrame
    loglik: float
    log_posterior: float
    converged: bool
    n_iter: int
    n_em_evaluations: int
    n_safeguards: int
    history: pd.DataFrame
    vcov: np.ndarray | None
    se_method: str | None
    n_obs: int
    _model: Model = field(repr=False)
    _eta: np.ndarray = field(repr=False)
    _quadrature: Quadrature = field(repr=False)
    _method: BAEM = field(default_factory=BAEM, repr=False)
    _eh_weights: np.ndarray | None = field(default=None, repr=False)

    # ---- summaries ---------------------------------------------------------------------------

    @property
    def _has_priors(self) -> bool:
        """Whether at least one free parameter has an active prior."""
        free = self.parameters[self.parameters["free"]]
        return any(p is not None for p in free["prior"])

    @property
    def n_params(self) -> int:
        """Number of free parameters (including EH latent-weight parameters; DESIGN §4.8)."""
        return int(self._eta.shape[0]) + eh_param_count(
            self.parameters, self._model, self._quadrature
        )

    def _aic_value(self) -> float:
        return -2.0 * self.loglik + 2.0 * self.n_params

    def _bic_value(self) -> float:
        return -2.0 * self.loglik + self.n_params * float(np.log(self.n_obs))

    @property
    def aic(self) -> float:
        r""":math:`-2\ell+2k` (warns with ``FlexpyirtWarning`` under MAP)."""
        if self._has_priors:
            warnings.warn(
                "AIC uses the marginal log-likelihood evaluated at the MAP estimate rather than "
                "the MLE; interpret with caution when priors are present.",
                FlexpyirtWarning,
                stacklevel=2,
            )
        return self._aic_value()

    @property
    def bic(self) -> float:
        r""":math:`-2\ell+k\log N` (warns with ``FlexpyirtWarning`` under MAP)."""
        if self._has_priors:
            warnings.warn(
                "BIC uses the marginal log-likelihood evaluated at the MAP estimate rather than "
                "the MLE; interpret with caution when priors are present.",
                FlexpyirtWarning,
                stacklevel=2,
            )
        return self._bic_value()

    @property
    def items(self) -> pd.DataFrame:
        r"""One row per item (per group in a multiple-group model).

        Estimates and SEs by internal parameter, plus the display IRT thresholds and their
        delta-method SEs (DESIGN.md §7): :math:`b = -d/a` (2PL/Rasch/3PL/4PL),
        :math:`b_k = -d_k/a` (GRM), and step difficulties :math:`b_k = -(d_k - d_{k-1})/a` with
        :math:`d_0 = 0` (GPCM/PCM), with corresponding ``b_se`` / ``bk_se`` columns computed from
        :attr:`vcov` using JAX automatic differentiation for the Jacobian. Fixed parameters
        contribute no variance; when ``vcov`` is ``None`` or all parameters entering a threshold
        are fixed, its SE is NaN.
        """
        p = self.parameters[self.parameters["item"] != LATENT_ITEM]
        index = ["group", "item"]
        est = p.pivot(index=index, columns="param", values="estimate")
        se = p.pivot(index=index, columns="param", values="se").add_suffix("_se")
        fids_piv = p.pivot(index=index, columns="param", values="free_id")
        params = list(dict.fromkeys(p["param"]))
        groups = list(dict.fromkeys(p["group"]))
        order = pd.MultiIndex.from_product([groups, list(self._model.item_names)], names=index)
        out = pd.concat([est[params], se[[f"{c}_se" for c in params]]], axis=1).loc[order]
        models = p.drop_duplicates(index).set_index(index)["model"]
        out.insert(0, "model", models.loc[order])

        kernels = {name: self._model.item_kernel(m) for name, m in self._model.items}
        irt_cols = list(
            dict.fromkeys(
                name for _, m in self._model.items for name in self._model.item_kernel(m).irt_names
            )
        )
        if irt_cols:
            b_data: dict[str, np.ndarray] = {
                col: np.full(len(order), np.nan, dtype=float)
                for col in (*irt_cols, *(f"{c}_se" for c in irt_cols))
            }
            est_ord = est.loc[order]
            fids_ord = fids_piv.loc[order]
            for row_idx, (grp, itm) in enumerate(order):
                kernel = kernels[itm]
                p_names = list(kernel.param_names)
                xi = est_ord.loc[(grp, itm), p_names].to_numpy(dtype=float)
                fids = fids_ord.loc[(grp, itm), p_names].to_numpy(dtype=int)
                b_vals, b_ses = _delta_irt_thresholds(kernel, xi, fids, self.vcov)
                for name, val, se_val in zip(kernel.irt_names, b_vals, b_ses, strict=True):
                    b_data[name][row_idx] = val
                    b_data[f"{name}_se"][row_idx] = se_val
            for col, arr in b_data.items():
                out[col] = arr

        out.columns.name = None
        out = out.reset_index()
        if not self._model.is_multigroup:
            out = out.drop(columns="group")
        return out

    @property
    def latent(self) -> pd.DataFrame:
        r"""Latent distribution summary per group.

        For ``Normal`` models: latent means, variances, correlations (for D > 1) and their SEs.
        For ``NormalMixture`` models: ``mean``, ``var`` and the raw shape ``rmean_c``, ``rvar_c``,
        ``logit_c`` with SEs, followed by the standardized components ``pi_c``, ``mean_c``,
        ``var_c`` (:math:`c=1,\dots,k`, on the :math:`\theta` scale) with delta-method SEs
        (DESIGN.md §16.7).
        For ``EmpiricalHistogram`` models: columns ``group``, ``mean``, ``var``, ``sd``,
        ``skewness``, ``kurtosis`` (Fisher's excess kurtosis :math:`\mu_4/\sigma^4 - 3`, 0 for
        a normal distribution; DESIGN.md §15.38). See :attr:`latent_weights` for the full
        ``(group, node, weight)`` table.
        """
        if self._model.is_empirical_histogram:
            assert self._eh_weights is not None
            nodes = self._quadrature.nodes()
            groups = [g for g, _ in self._model.resolved_groups()]
            rows = []
            for idx, g_name in enumerate(groups):
                mean, var, sd, skew, kurt = eh_moments(nodes, self._eh_weights[idx])
                rows.append(
                    {
                        "group": g_name,
                        "mean": mean,
                        "var": var,
                        "sd": sd,
                        "skewness": skew,
                        "kurtosis": kurt,
                    }
                )
            return pd.DataFrame(rows)
        p = self.parameters[self.parameters["item"] == LATENT_ITEM]
        groups = list(dict.fromkeys(p["group"]))
        if self._model.is_davidian:
            lat_cols = list(davidian_param_names(self._model.davidian_degree))
        elif self._model.is_mixture:
            lat_cols = list(mixture_param_names(self._model.mixture_components))
        else:
            lat_cols = list(latent_param_names(self._model.n_factors))
        est = p.pivot(index="group", columns="param", values="estimate")[lat_cols]
        se = p.pivot(index="group", columns="param", values="se")[lat_cols].add_suffix("_se")
        out = pd.concat([est, se], axis=1).loc[groups]
        out.columns.name = None
        out = out.reset_index()
        if self._model.is_mixture:
            fid = p.pivot(index="group", columns="param", values="free_id")[lat_cols]
            derived = [
                _mixture_derived(
                    lat_cols,
                    est.loc[g].to_numpy(dtype=float),
                    fid.loc[g].to_numpy(dtype=int),
                    self.vcov,
                )
                for g in groups
            ]
            out = pd.concat([out, pd.DataFrame(derived)], axis=1)
        return out

    @property
    def latent_weights(self) -> pd.DataFrame | None:
        """Per-group EH weights ``(group, node, weight)``, or ``None`` for ``Normal`` models."""
        if not self._model.is_empirical_histogram:
            return None
        assert self._eh_weights is not None
        nodes = self._quadrature.nodes()
        groups = [g for g, _ in self._model.resolved_groups()]
        n_nodes = len(nodes)
        return pd.DataFrame(
            {
                "group": np.repeat(groups, n_nodes),
                "node": np.tile(nodes, len(groups)),
                "weight": self._eh_weights.reshape(-1),
            }
        )

    # ---- scoring -----------------------------------------------------------------------------

    def score(
        self,
        data: Any,
        *,
        method: ScoreMethod = "EAP",
        group: Any = None,
        item_names: list[str] | None = None,
        key: Array | None = None,
        n_draws: int = 5,
    ) -> pd.DataFrame:
        r"""Score respondents on the fitted model.

        Supports six scoring methods (DESIGN.md §8, §15.21–§15.25, §18.2). All but ``"EAPsum"``
        drop missing responses from the likelihood:

        - ``"EAP"``: expected a posteriori :math:`\hat\theta=\sum_q\theta_q h_{pq}` and posterior
          SD on the fitted model's quadrature grid using the respondent's group's estimated latent
          distribution as prior (Bock & Mislevy, 1982).
        - ``"MAP"``: Bayes modal estimate maximizing
          :math:`\log L(\theta\mid y) + \log\phi(\theta;\mu_g,\sigma_g^2)` with the group's
          estimated normal latent distribution as a continuous prior, and SE
          :math:`(-\partial^2/\partial\theta^2)^{-1/2}` at the mode. All-missing rows return
          :math:`(\mu_g, \sigma_g)`. For multidimensional models the prior is the estimated
          multivariate normal and SEs are the square roots of the diagonal of
          :math:`(-\nabla^2)^{-1}` at the mode.
        - ``"ML"``: maximum likelihood estimate maximizing :math:`\log L(\theta\mid y)`, with
          observed-information SE :math:`(-\partial^2\log L/\partial\theta^2)^{-1/2}`. Patterns
          with no finite maximum (all lowest / all highest observed categories, or diverging
          3PL/4PL tails) return ``-inf`` / ``+inf`` with SE ``NaN``; all-missing rows return
          ``NaN`` with SE ``NaN``.
        - ``"WLE"``: weighted likelihood estimate (Warm, 1989; Firth, 1993) solving
          :math:`\partial\log L/\partial\theta + J(\theta)/(2I(\theta)) = 0` with observed-item
          Fisher test information :math:`I(\theta)` and :math:`J(\theta)=\partial I/\partial\theta`
          from JAX autodiff, equivalently maximizing
          :math:`\log L(\theta\mid y) + \tfrac{1}{2}\log I(\theta)`. Finite on extreme patterns;
          all-missing rows return ``NaN`` with SE ``NaN``. SE is the inverse square root of the
          negative second derivative of the WLE objective (matching mirt's default).
        - ``"PV"``: ``n_draws`` independent plausible values per respondent drawn from the
          posterior on a fine equispaced grid with piecewise-linear inverse-CDF interpolation
          (Mislevy, 1991). Requires an explicit JAX PRNG ``key`` and is deterministic given
          ``key``.
        - ``"EAPsum"``: summed-score EAP (Thissen, Pommerich, Billeaud & Williams, 1995): the
          posterior mean and SD of :math:`\theta` given the respondent's summed score, looked up
          in :func:`flexpyirt.summed_score_table` (of the respondent's group). A respondent with
          any missing response gets ``NaN`` (as in mirt), because a summed score over fewer
          items is not on the scale of the table. Unidimensional models with ``Normal`` latent
          distributions only.

        Parameters
        ----------
        data : DataFrame or array-like
            Responses on the fitted items (NaN is missing).
        method : {"EAP", "MAP", "ML", "WLE", "PV", "EAPsum"}, default "EAP"
            Scoring method.
        group : array-like, optional
            Group labels (required for a multiple-group model).
        item_names : list of str, optional
            Column names when ``data`` is an array.
        key : jax.Array, optional
            JAX PRNG key (e.g. ``jax.random.key(0)``). Required when ``method="PV"``; must be
            ``None`` for deterministic methods.
        n_draws : int, default 5
            Number of plausible values per respondent when ``method="PV"`` (must be ``>= 1``).

        Returns
        -------
        DataFrame
            For unidimensional ``"EAP"``, ``"MAP"``, ``"ML"``, ``"WLE"`` and ``"EAPsum"``: columns
            ``theta`` and ``se``. For unidimensional ``"PV"``: columns ``pv1`` ..
            ``pv{n_draws}``. For multidimensional ``"EAP"`` and ``"MAP"``: columns
            ``theta_<factor>`` and ``se_<factor>`` for each declared factor (``"ML"``, ``"WLE"``
            and ``"PV"`` raise :class:`SpecificationError` for ``D > 1``). One row per
            respondent; index follows ``data`` when it is a DataFrame.

        Raises
        ------
        PrecisionError
            If JAX is not configured for 64-bit floating point.
        SpecificationError
            If ``method`` is unknown, ``key`` is missing or invalid for ``"PV"``, ``key`` is
            supplied for a non-PV method, ``n_draws`` is not a positive integer, or
            ``method="EAPsum"`` is used with a multidimensional model or a latent distribution
            other than ``Normal``.
        """
        check_x64()
        if method not in _SCORE_METHODS:
            raise SpecificationError(
                f"Unsupported scoring method {method!r}; use one of {_SCORE_METHODS}."
            )
        if method == "MAP" and self._model.is_empirical_histogram:
            raise SpecificationError(
                "method='MAP' requires a continuous latent prior and is not supported with "
                "EmpiricalHistogram; use method='EAP', 'ML', 'WLE', or 'PV'."
            )
        if method == "MAP" and self._model.is_davidian:
            raise SpecificationError(
                "method='MAP' is not supported with DavidianCurve (the density can be multimodal "
                "and vanish at polynomial roots); use method='EAP', 'ML', 'WLE', or 'PV'."
            )
        if method == "MAP" and self._model.is_mixture:
            raise SpecificationError(
                "method='MAP' is not supported with NormalMixture (the mixture prior can be "
                "multimodal, so the posterior mode need not be unique); use method='EAP', 'ML', "
                "'WLE', or 'PV'."
            )
        if method == "EAPsum" and (
            self._model.is_multidimensional
            or self._model.is_empirical_histogram
            or self._model.is_davidian
            or self._model.is_mixture
        ):
            raise SpecificationError(
                "method='EAPsum' supports unidimensional models with Normal latent "
                "distributions only."
            )
        if method == "PV":
            if key is None:
                raise SpecificationError(
                    "method='PV' requires an explicit JAX PRNG key (e.g. key=jax.random.key(0))."
                )
            if not _is_prng_key(key):
                raise SpecificationError(
                    "key must be a JAX PRNG key from jax.random.key() or jax.random.PRNGKey()."
                )
            if isinstance(n_draws, bool) or not isinstance(n_draws, int) or n_draws < 1:
                raise SpecificationError(f"n_draws must be a positive integer, got {n_draws!r}.")
        elif key is not None:
            raise SpecificationError(f"key= is only used with method='PV', not {method!r}.")

        y = coerce_responses(data, self._model, item_names=item_names)
        codes, _ = coerce_groups(group, self._model, y.shape[0], require_all=False)
        compiled = compile_problem(
            self._model, y, codes, self.parameters, self._quadrature, eh_weights=self._eh_weights
        )
        eta = jnp.asarray(self._eta)
        n = compiled.n_patterns
        inverse = compiled.inverse
        index = data.index if isinstance(data, pd.DataFrame) else None

        if method == "PV":
            assert key is not None
            n_obs = int(inverse.shape[0])
            bucket = max(8, 1 << (n_obs - 1).bit_length())
            inv_pad = np.zeros(bucket, dtype=np.int32)
            inv_pad[:n_obs] = inverse
            draws = np.asarray(
                _pv(compiled.problem, eta, jnp.asarray(inv_pad), key, n_draws=n_draws)
            )[:n_obs]
            if self._model.is_multidimensional:
                fnames = self._model.factor_names
                cols_md = {
                    f"pv{m}_{f}": draws[:, m - 1, i]
                    for m in range(1, n_draws + 1)
                    for i, f in enumerate(fnames)
                }
                return pd.DataFrame(cols_md, index=index)
            cols = [f"pv{m}" for m in range(1, n_draws + 1)]
            return pd.DataFrame(draws, columns=cols, index=index)

        if method == "EAPsum":
            _, table_theta, table_se = (
                np.asarray(a) for a in _summed_score_eap(compiled.problem, eta)
            )
            complete = (y >= 0).all(axis=1)
            sums = np.where(y >= 0, y, 0).sum(axis=1)
            return pd.DataFrame(
                {
                    "theta": np.where(complete, table_theta[codes, sums], np.nan),
                    "se": np.where(complete, table_se[codes, sums], np.nan),
                },
                index=index,
            )

        scorer = {"EAP": _eap, "MAP": _map, "ML": _ml, "WLE": _wle}[method]
        theta, se = scorer(compiled.problem, eta)
        if self._model.is_multidimensional:
            theta_obs = np.asarray(theta)[:n][inverse]
            se_obs = np.asarray(se)[:n][inverse]
            fnames = self._model.factor_names
            cols_dict = {
                **{f"theta_{f}": theta_obs[:, i] for i, f in enumerate(fnames)},
                **{f"se_{f}": se_obs[:, i] for i, f in enumerate(fnames)},
            }
            return pd.DataFrame(cols_dict, index=index)
        return pd.DataFrame(
            {"theta": np.asarray(theta)[:n][inverse], "se": np.asarray(se)[:n][inverse]},
            index=index,
        )

    def item_fit(
        self,
        data: Any,
        *,
        min_cell: float = 1.0,
        group: Any = None,
        item_names: list[str] | None = None,
    ) -> pd.DataFrame:
        r"""Orlando–Thissen S-X2 item fit statistics (DESIGN.md §15.40).

        For each item :math:`j`, compares observed and model-implied counts of each category at
        each summed score :math:`s=1,\dots,S-1` (Orlando & Thissen, 2000, 2003):

        .. math::
            E_j[s,k] = N_s\,\frac{\sum_q P_{jk}(\theta_q)\,L_{-j}(s-k\mid\theta_q)\,w_q}
                                  {\sum_q L(s\mid\theta_q)\,w_q},\qquad
            S\text{-}X^2_j=\sum_{s,k}\frac{(O_{jsk}-E_{jsk})^2}{E_{jsk}},

        where :math:`L` and :math:`L_{-j}` (item :math:`j` left out) come from the
        Lord–Wingersky recursion, :math:`w_q` are the fitted latent weights on the fitted
        quadrature grid, and :math:`N_s` is the number of respondents with summed score
        :math:`s`. Sparse cells are collapsed exactly as mirt's ``itemfit()`` does, and
        :math:`df_j = \#\text{cells} - \#\text{rows} - p_j - p_{\text{latent}}`
        (free parameters of item :math:`j` and of the latent distribution, mirt's convention).

        Parameters
        ----------
        data : DataFrame or array-like
            Complete responses on the fitted items (typically the fitting data).
        min_cell : float, default 1.0
            Minimum expected count per cell before collapsing (mirt's ``min_cell``).
        group : array-like, optional
            Group labels (required for a multiple-group model). S-X2 is computed within each
            group from that group's responses, item parameters and latent distribution, with
            :math:`df` net of that group's free item and latent parameters (shared anchors count
            in every group), as mirt's ``itemfit()`` does for ``multipleGroup`` fits.
        item_names : list of str, optional
            Column names when ``data`` is an array.

        Returns
        -------
        DataFrame
            One row per item (per group and item, with a leading ``group`` column, in a
            multiple-group model): ``item``, ``s_x2``, ``df``, ``rmsea``, ``p``. Items whose
            collapsed table leaves ``df <= 0`` get ``df = 0`` and NaN statistics.

        Raises
        ------
        PrecisionError
            If JAX is not configured for 64-bit floating point.
        SpecificationError
            For multidimensional, ``EmpiricalHistogram``, ``DavidianCurve`` or ``NormalMixture``
            models (not yet supported), data with missing responses, or an item with an unobserved
            category (within a group).
        """
        check_x64()
        model = self._model
        if (
            model.is_multidimensional
            or model.is_empirical_histogram
            or model.is_davidian
            or model.is_mixture
        ):
            raise SpecificationError(
                "item_fit() supports unidimensional models with Normal latent distributions only."
            )
        if isinstance(min_cell, bool) or not (isinstance(min_cell, (int, float)) and min_cell > 0):
            raise SpecificationError(f"min_cell must be a positive number, got {min_cell!r}.")
        y = coerce_responses(data, model, item_names=item_names)
        if (y < 0).any():
            raise SpecificationError(
                "item_fit() requires complete data; S-X2 is not defined with missing responses."
            )
        codes, group_names = coerce_groups(group, model, y.shape[0], require_all=False)
        n_categories = [m.kernel.n_categories for _, m in model.items]
        compiled = compile_problem(model, y, codes, self.parameters, self._quadrature)
        prob = compiled.problem
        eta = jnp.asarray(self._eta)
        log_probs = [np.exp(np.asarray(lp)) for lp in _core._item_log_probs(prob, eta)]
        all_weights = np.exp(np.asarray(_core._latent_log_weights(prob, eta)))
        params = self.parameters
        rows = []
        for g, g_name in enumerate(group_names):
            y_g = y[codes == g]
            if y_g.shape[0] == 0:
                continue
            for j, (name, _) in enumerate(model.items):
                if np.unique(y_g[:, j]).size != n_categories[j]:
                    where = f" in group {g_name!r}" if model.is_multigroup else ""
                    raise SpecificationError(
                        f"Item {name!r} does not have every response category observed{where}; "
                        "S-X2 needs all categories (as in mirt)."
                    )
            traces: list[np.ndarray] = [np.empty(0)] * len(model.items)
            for fam, lp_np in zip(prob.families, log_probs, strict=True):
                for pos, col in enumerate(np.asarray(fam.columns)):
                    traces[int(col)] = lp_np[g, pos]  # (Q, K_j)
            observed = _itemfit.observed_tables(y_g, n_categories)
            expected = _itemfit.s_x2_expected_tables(traces, all_weights[g])
            n_s = observed[0].sum(axis=1)
            in_group = params["group"] == g_name
            is_latent = params["item"] == LATENT_ITEM
            n_free_latent = int(params.loc[in_group & is_latent, "free"].sum())
            for j, (name, _) in enumerate(model.items):
                o, e = _itemfit.collapse_cells(observed[j], expected[j] * n_s[:, None], min_cell)
                n_free_item = int(params.loc[in_group & (params["item"] == name), "free"].sum())
                stat, df, rmsea, pval = _itemfit.s_x2(
                    o, e, n_free_item, n_free_latent, y_g.shape[0]
                )
                row = {"item": name, "s_x2": stat, "df": df, "rmsea": rmsea, "p": pval}
                rows.append({"group": g_name, **row} if model.is_multigroup else row)
        return pd.DataFrame(rows)

    def _require_unidimensional_normal(self, what: str) -> None:
        """Raise unless the model is single-group, unidimensional with a ``Normal`` latent."""
        model = self._model
        if (
            model.is_multidimensional
            or model.is_multigroup
            or model.is_empirical_histogram
            or model.is_davidian
            or model.is_mixture
        ):
            raise SpecificationError(
                f"{what} supports single-group unidimensional models with a Normal latent "
                "distribution only."
            )

    def _traces(
        self, prob: _core.CompiledProblem, nodes: np.ndarray | None = None
    ) -> list[np.ndarray]:
        """Per item, ``(Q, K_j)`` category probabilities of group 0 at ``nodes`` (default: grid)."""
        if nodes is not None:
            prob = replace(prob, nodes=jnp.asarray(nodes, dtype=jnp.float64).reshape(-1, 1))
        log_probs = _core._item_log_probs(prob, jnp.asarray(self._eta))
        traces: list[np.ndarray] = [np.empty(0)] * len(self._model.items)
        for fam, lp in zip(prob.families, log_probs, strict=True):
            lp_np = np.exp(np.asarray(lp))
            for pos, col in enumerate(np.asarray(fam.columns)):
                traces[int(col)] = lp_np[0, pos]
        return traces

    def _compiled_for(self, y: np.ndarray) -> _core.CompiledProblem:
        codes, _ = coerce_groups(None, self._model, y.shape[0], require_all=False)
        return compile_problem(self._model, y, codes, self.parameters, self._quadrature).problem

    def _simulate_responses(self, key: Any, missing: np.ndarray) -> np.ndarray:
        r"""Simulate responses from the fitted model, NaN where ``missing`` (``(N, J)`` bool).

        :math:`\theta_i\sim N(\hat\mu,\hat\sigma^2)` (the fitted latent distribution), then each
        response is drawn from :math:`P_{jk}(\theta_i)` by inversion of a uniform draw.
        """
        latent = self.latent.iloc[0]
        n, n_items = missing.shape
        k_theta, k_unif = jax.random.split(key)
        theta = float(latent["mean"]) + np.sqrt(float(latent["var"])) * np.asarray(
            jax.random.normal(k_theta, (n,), dtype=jnp.float64)
        )
        unif = np.asarray(jax.random.uniform(k_unif, (n, n_items), dtype=jnp.float64))
        prob = self._compiled_for(np.zeros((1, n_items), dtype=np.int64))
        traces = self._traces(prob, theta)
        out = np.empty((n, n_items), dtype=np.float64)
        for j, trace in enumerate(traces):
            cum = np.cumsum(trace, axis=1)[:, :-1]
            out[:, j] = (unif[:, j : j + 1] > cum).sum(axis=1)
        out[missing] = np.nan
        return out

    def local_dependence(
        self,
        data: Any,
        *,
        item_names: list[str] | None = None,
    ) -> LocalDependenceResult:
        r"""Local-dependence statistics for every item pair (DESIGN.md §16.11).

        * **Q3** (Yen, 1984): the correlation over respondents of the residuals
          :math:`d_{ij}=x_{ij}-E(X_j\mid\hat\theta_i)` with EAP :math:`\hat\theta_i`
          (pairwise-complete), and **Q3\*** :math:`=Q3-\overline{Q3}` with
          :math:`\overline{Q3}` the mean off-diagonal Q3 (Christensen, Makransky & Horton, 2017).
        * **LD** :math:`X^2` and :math:`G^2` (Chen & Thissen, 1997) comparing each pair's
          table with :math:`E_{ab}=n_{ij}\sum_q P_{ia}(\theta_q)P_{jb}(\theta_q)w_q` on the
          fitted grid, with :math:`df=(K_i-1)(K_j-1)`, :math:`p`-values and the signed
          Cramér's :math:`V` (sign of :math:`\gamma_O-\gamma_E`), as mirt's
          ``residuals(type = "LD" / "LDG2" / "Q3")``.

        Parameters
        ----------
        data : DataFrame or array-like
            Responses on the fitted items (typically the fitting data); missing as NaN.
        item_names : list of str, optional
            Column names when ``data`` is an array.

        Returns
        -------
        LocalDependenceResult

        Raises
        ------
        PrecisionError
            If JAX is not configured for 64-bit floating point.
        SpecificationError
            For multiple-group, multidimensional or non-``Normal`` latent models, or invalid
            data.
        """
        from flexpyirt.dimensionality import LocalDependenceResult

        check_x64()
        self._require_unidimensional_normal("local_dependence()")
        y = coerce_responses(data, self._model, item_names=item_names)
        names = [name for name, _ in self._model.items]
        prob = self._compiled_for(y)
        eap = self.score(pd.DataFrame(np.where(y >= 0, y, np.nan), columns=names))[
            "theta"
        ].to_numpy(dtype=np.float64)
        expected = np.column_stack(
            [t @ np.arange(t.shape[1], dtype=np.float64) for t in self._traces(prob, eap)]
        )
        residuals = np.where(y >= 0, y - expected, np.nan)
        q3 = _local_dependence.q3_matrix(residuals)
        upper = np.triu_indices(len(names), k=1)
        q3_mean = float(np.mean(q3[upper]))
        weights = np.exp(np.asarray(_core._latent_log_weights(prob, jnp.asarray(self._eta))))[0]
        ld = _local_dependence.ld_pairs(y, self._traces(prob), weights)
        pairs = pd.DataFrame(
            {
                "item_i": [names[int(i)] for i in ld["i"]],
                "item_j": [names[int(j)] for j in ld["j"]],
                "n": ld["n"].astype(np.int64),
                "q3": q3[upper],
                "q3_star": q3[upper] - q3_mean,
                "x2": ld["x2"],
                "g2": ld["g2"],
                "df": ld["df"].astype(np.int64),
                "p_x2": ld["p_x2"],
                "p_g2": ld["p_g2"],
                "cramer_v": ld["cramer_v"],
            }
        )
        return LocalDependenceResult(
            pairs=pairs,
            q3=pd.DataFrame(q3, index=names, columns=names),
            q3_mean=q3_mean,
        )

    def jsi(
        self,
        data: Any,
        *,
        item_names: list[str] | None = None,
        method: JSIMethod = "refit",
        newton_steps: int = 1,
        tol: float = 1e-8,
        max_iter: int = 100,
    ) -> JSIResult:
        r"""Jackknife slope index (Edwards, Houts & Cai, 2018; DESIGN.md §16.17).

        For every item :math:`j` the model is re-estimated without item :math:`j`, and for
        every other item :math:`i`

        .. math::
            \mathrm{JSI}_{ji}=\frac{\hat a_i-\hat a_i^{(-j)}}
            {\mathrm{SE}\bigl(\hat a_i^{(-j)}\bigr)}.

        A large positive value means item :math:`i`'s slope was inflated by item :math:`j`, a
        sign of local dependence between the two. There is no fixed cutoff: look for values that
        stand out from the rest of the matrix.

        Removing item :math:`j` is done by treating its responses as missing, so the latent
        distribution, constraints across items and priors of the remaining parameters stay as
        fitted. The three methods differ only in how :math:`\hat\eta^{(-j)}` is obtained, all
        starting from the full-model estimates :math:`\hat\eta`:

        * ``"refit"`` (as mirt's ``residuals(type = "JSI")``): Newton–Raphson with step
          halving on :math:`\ell_{(-j)}` until the largest step is below ``tol``; the SE is
          from the observed information at the refit.
        * ``"newton"``: ``newton_steps`` Newton steps (a Taylor approximation of the refit),
          the SE from the information at the last evaluation. One step usually agrees with
          the refit to a few hundredths; two steps to about 0.01.
        * ``"one_step"``: one step with the full model's covariance matrix,
          :math:`\hat\eta^{(-j)}\approx\hat\eta+V\nabla\ell_{(-j)}(\hat\eta)`, and the full
          model's SE in the denominator. It needs only :math:`J` gradients (one ``vmap``), but
          the different denominator makes it differ from the refit by up to about 0.1 in small
          samples.

        Parameters
        ----------
        data : DataFrame or array-like
            The fitting data; missing as NaN.
        item_names : list of str, optional
            Column names when ``data`` is an array.
        method : {"refit", "newton", "one_step"}
            How the estimates without each item are obtained.
        newton_steps : int
            Steps for ``method="newton"`` (at least 1).
        tol : float
            Convergence tolerance on the largest step for ``method="refit"``.
        max_iter : int
            Maximum Newton steps for ``method="refit"``.

        Returns
        -------
        JSIResult

        Raises
        ------
        PrecisionError
            If JAX is not configured for 64-bit floating point.
        SpecificationError
            For multiple-group, multidimensional or non-``Normal`` latent models, fewer than
            three items, invalid options, or ``method="one_step"`` without a covariance matrix.

        Warns
        -----
        ConvergenceWarning
            If a refit does not converge in ``max_iter`` steps.
        """
        from flexpyirt import _jsi
        from flexpyirt.dimensionality import JSIResult
        from flexpyirt.errors import ConvergenceWarning

        check_x64()
        self._require_unidimensional_normal("jsi()")
        if method not in ("refit", "newton", "one_step"):
            raise SpecificationError(
                f"Unknown method={method!r}; use 'refit', 'newton' or 'one_step'."
            )
        if method == "newton" and (int(newton_steps) != newton_steps or newton_steps < 1):
            raise SpecificationError("newton_steps must be a positive integer.")
        if method == "refit" and (not tol > 0 or int(max_iter) != max_iter or max_iter < 1):
            raise SpecificationError("tol must be positive and max_iter a positive integer.")
        if method == "one_step" and self.vcov is None:
            raise SpecificationError("method='one_step' needs a fit with standard errors (se).")
        names = [name for name, _ in self._model.items]
        n_items = len(names)
        if n_items < 3:
            raise SpecificationError("jsi() needs at least three items.")
        if method != "refit" and n_items < _JSI_APPROXIMATION_MIN_ITEMS:
            warnings.warn(
                f"With {n_items} items, removing one changes the estimates too much for "
                f"method={method!r} to approximate the refit; use method='refit'.",
                FlexpyirtWarning,
                stacklevel=2,
            )
        y = coerce_responses(data, self._model, item_names=item_names)
        prob = self._compiled_for(y)
        eta = np.asarray(self._eta, dtype=np.float64)

        params = self.parameters
        free = params["free"].to_numpy(dtype=bool)
        fid = params["free_id"].to_numpy(dtype=np.int64)
        item_col = params["item"].to_numpy()
        lookup = dict(self._model.items)
        slope_id = np.full(n_items, -1, dtype=np.int64)
        for k, name in enumerate(names):
            slope = self._model.item_kernel(lookup[name]).slope_params
            if not slope:
                continue
            row = (item_col == name) & (params["param"].to_numpy() == slope[0]) & free
            if row.any():
                slope_id[k] = fid[row][0]
        a_full = np.where(slope_id >= 0, eta[np.maximum(slope_id, 0)], np.nan)

        out = np.full((n_items, n_items), np.nan)
        converged = np.ones(n_items, dtype=bool)
        if method == "one_step":
            assert self.vcov is not None
            grads = np.asarray(_jsi.gradients_without(prob, jnp.asarray(eta), jnp.arange(n_items)))
            vcov_full = np.asarray(self.vcov, dtype=np.float64)
            se_full = np.sqrt(np.clip(np.diag(vcov_full), 0.0, None))
        for j, name in enumerate(names):
            keep = np.unique(fid[free & (item_col != name)])
            if method == "one_step":
                grad = np.zeros_like(eta)
                grad[keep] = grads[j, keep]
                eta_j = eta + vcov_full @ grad
                se_j = se_full
            else:
                outcome = _jsi.newton_without(
                    prob,
                    eta,
                    j,
                    keep,
                    max_steps=int(max_iter if method == "refit" else newton_steps),
                    tol=tol if method == "refit" else None,
                )
                converged[j] = outcome.converged
                eta_j = outcome.eta
                se_j = np.full(eta.size, np.nan)
                se_j[keep] = np.sqrt(np.clip(np.diag(outcome.vcov), 0.0, None))
            for i in range(n_items):
                sid = slope_id[i]
                if i == j or sid < 0 or sid not in keep:
                    continue
                out[j, i] = (a_full[i] - eta_j[sid]) / se_j[sid]
        if not converged.all():
            bad = [names[k] for k in np.flatnonzero(~converged)]
            warnings.warn(
                f"The refit without items {bad} did not converge in {max_iter} steps.",
                ConvergenceWarning,
                stacklevel=2,
            )
        upper_i, upper_j = np.triu_indices(n_items, k=1)
        pairs = pd.DataFrame(
            {
                "item_i": [names[int(i)] for i in upper_i],
                "item_j": [names[int(j)] for j in upper_j],
                "jsi_ij": out[upper_j, upper_i],
                "jsi_ji": out[upper_i, upper_j],
            }
        )
        pairs["folded"] = pairs["jsi_ij"] + pairs["jsi_ji"]
        return JSIResult(
            matrix=pd.DataFrame(out, index=pd.Index(names, name="removed"), columns=names),
            pairs=pairs,
            method=method,
            converged=pd.Series(converged, index=names, name="converged"),
        )

    def m2(
        self,
        data: Any,
        *,
        item_names: list[str] | None = None,
        ci: float = 0.90,
    ) -> pd.DataFrame:
        r"""Compute the :math:`M_2` / :math:`M_2^*` limited-information fit statistic (§16.3).

        Computes the Maydeu-Olivares & Joe (2005, 2006) limited-information test statistic on the
        first- and second-order item-score moments — the :math:`J` item means :math:`E[y_j]` and
        :math:`\binom{J}{2}` cross-products :math:`E[y_i y_j]` (:math:`i < j`), matching R mirt's
        default ``M2(type = "M2*")`` (for dichotomous items this is identically :math:`M_2`):

        .. math::
            M_2 = N\,(\hat p_2 - \pi_2)^\top C_2\,(\hat p_2 - \pi_2),\qquad
            C_2 = \Delta_c(\Delta_c^\top \Xi_2 \Delta_c)^{-1}\Delta_c^\top,\qquad
            df = J + \binom{J}{2} - q,

        where :math:`\Delta_c` is an orthonormal basis for the orthogonal complement of
        :math:`\Delta_2 = \partial\pi_2 / \partial\eta^\top` (over all free item and latent
        parameters, shared parameters once), :math:`\Xi_2` is the asymptotic covariance of
        :math:`\sqrt{N}(\hat p_2 - \pi_2)` under local independence, and :math:`q` is
        :attr:`n_params`. Also reports RMSEA with its noncentral-:math:`\chi^2` confidence
        interval (MacCallum, Browne & Sugawara, 1996), SRMSR from item-score residual
        correlations, and for dichotomous models TLI and CFI against the independence (null)
        model (polytomous models return ``NaN`` for ``tli`` and ``cfi`` because the null
        :math:`\Delta_2` is rank-deficient under item-score moments; see DESIGN.md §16.3).

        Parameters
        ----------
        data : DataFrame or array-like
            Complete responses on the fitted items (typically the fitting data).
        item_names : list of str, optional
            Column names when ``data`` is an array.
        ci : float, default 0.90
            Confidence level strictly in ``(0, 1)`` for the RMSEA confidence interval. The
            output columns follow ``ci`` (e.g. ``rmsea_5`` and ``rmsea_95`` for ``ci=0.90``).

        Returns
        -------
        DataFrame
            One-row DataFrame with columns ``m2``, ``df``, ``p``, ``rmsea``, ``rmsea_<lo>``,
            ``rmsea_<hi>``, ``srmsr``, ``tli``, ``cfi``. When ``df <= 0`` or :math:`\Delta_2` is
            rank-deficient (the model is not locally identified from first- and second-order
            moments), a :class:`FlexpyirtWarning` is emitted and ``m2``, ``p``, RMSEA, its CI,
            ``tli`` and ``cfi`` are ``NaN`` (``df`` and ``srmsr`` remain finite).

        Raises
        ------
        PrecisionError
            If JAX is not configured for 64-bit floating point.
        SpecificationError
            For multidimensional, multiple-group, ``EmpiricalHistogram``, ``NormalMixture``, or MAP
            (with priors) models, missing responses, invalid ``ci``, or fewer than 2 items or
            respondents.
        """
        check_x64()
        model = self._model
        if (
            model.is_multidimensional
            or model.is_multigroup
            or model.is_empirical_histogram
            or model.is_mixture
        ):
            raise SpecificationError(
                "m2() supports single-group unidimensional models with a Normal latent "
                "distribution only."
            )
        if self._has_priors:
            raise SpecificationError(
                "m2() requires an MLE fit without priors (MAP fits are not supported)."
            )
        if isinstance(ci, bool) or not isinstance(ci, (int, float)) or not (0.0 < float(ci) < 1.0):
            raise SpecificationError(f"ci must be strictly between 0 and 1, got {ci!r}.")
        ci_f = float(ci)
        y = coerce_responses(data, model, item_names=item_names)
        if (y < 0).any():
            raise SpecificationError(
                "m2() requires complete data; M2 is not defined with missing responses."
            )
        if y.shape[0] < 2:
            raise SpecificationError("m2() requires at least 2 respondents.")
        if y.shape[1] < 2:
            raise SpecificationError("m2() requires at least 2 items.")

        codes, _ = coerce_groups(None, model, y.shape[0], require_all=False)
        compiled = compile_problem(model, y, codes, self.parameters, self._quadrature)
        n_categories = [m.kernel.n_categories for _, m in model.items]
        res = _m2.m2_statistic(
            compiled.problem,
            jnp.asarray(self._eta),
            y,
            n_params=self.n_params,
            n_categories=n_categories,
            level=ci_f,
        )
        lo_col, hi_col = _m2.ci_column_names(ci_f)
        return pd.DataFrame(
            [
                {
                    "m2": res.m2,
                    "df": res.df,
                    "p": res.p,
                    "rmsea": res.rmsea,
                    lo_col: res.rmsea_lower,
                    hi_col: res.rmsea_upper,
                    "srmsr": res.srmsr,
                    "tli": res.tli,
                    "cfi": res.cfi,
                }
            ]
        )

    def dif(
        self,
        data: Any,
        *,
        group: Any,
        items: Iterable[str],
        params: Sequence[str] | None = None,
        scheme: DIFScheme = "add",
        p_adjust: PAdjust = "holm",
        item_names: list[str] | None = None,
    ) -> pd.DataFrame:
        r"""Likelihood-ratio DIF tests across the groups of a multiple-group model (§16.2).

        For each item, refits the model with its ``params`` made equal across groups
        (``scheme="add"``, from this fit as the baseline) or freed in every group
        (``scheme="drop"``), with the same estimation settings and quadrature, and compares the
        marginal log-likelihoods (Thissen, Steinberg & Wainer, 1993):

        .. math::
            X^2 = 2\,(\ell_{\text{free}} - \ell_{\text{constrained}}),\qquad
            df = |k_{\text{base}} - k_{\text{new}}|,\qquad p = P(\chi^2_{df} > X^2).

        Parameters
        ----------
        data : DataFrame or array-like
            The responses this result was fitted to.
        group : array-like
            Group label of each row (as passed to :func:`~flexpyirt.fit`).
        items : iterable of str
            Items to test, one refit each.
        params : sequence of str, optional
            Internal parameters to test (default: every parameter of each item).
        scheme : {"add", "drop"}, default "add"
            ``"add"`` constrains the item's parameters to equality; ``"drop"`` frees them.
        p_adjust : {"holm", "BH", "none"}, default "holm"
            Multiple-testing adjustment of ``adj_p`` over the tested items (Holm, 1979;
            Benjamini & Hochberg, 1995); ``"holm"`` is mirt's default.
        item_names : list of str, optional
            Column names when ``data`` is an array.

        Returns
        -------
        DataFrame
            One row per item: ``item``, ``converged``, ``aic``, ``sabic``, ``hq``, ``bic``
            (information criteria of the model with the item free minus the constrained one, as
            mirt's ``DIF()``; negative values favour DIF), ``x2``, ``df``, ``p`` and ``adj_p``.

        Raises
        ------
        SpecificationError
            For single-group models, MAP fits with priors, unknown items or parameters, or an
            item with nothing to test under ``scheme``.
        IdentificationError
            If a refitted model is not identified (e.g. freeing the only anchors).
        """
        from flexpyirt._dif import dif_tests

        return dif_tests(
            self,
            data,
            group=group,
            items=items,
            params=params,
            scheme=scheme,
            p_adjust=p_adjust,
            item_names=item_names,
        )

    def _model_label(self) -> str:
        """Concise description of item models in the specification."""
        counts: dict[str, int] = {}
        for _, m in self._model.items:
            tag = (
                f"{m.name}({m.kernel.n_categories})"
                if m.name in ("GRM", "GPCM", "PCM", "Nominal")
                else m.name
            )
            counts[tag] = counts.get(tag, 0) + 1
        if len(counts) == 1:
            return next(iter(counts))
        return ", ".join(f"{name} ({count})" for name, count in counts.items())

    def _groups_label(self) -> str:
        """Concise description of the model's groups."""
        groups = [g for g, _ in self._model.resolved_groups()]
        if self._model.is_multigroup:
            return f"{len(groups)} ({', '.join(groups)})"
        return "1 (single group)"

    def _convergence_label(self) -> str:
        """Concise description of convergence, iterations, and EM map evaluations."""
        status = "Converged" if self.converged else "NOT converged"
        evals = f"{self.n_em_evaluations} EM evaluations"
        if self.n_safeguards > 0:
            evals += f", {self.n_safeguards} SQUAREM safeguards"
        return f"{status} ({self.n_iter} iterations, {evals})"

    def _quadrature_label(self) -> str:
        """Concise description of the quadrature grid."""
        quad = self._quadrature
        d = self._model.n_factors
        if d == 1:
            return f"{quad.n_points} equispaced nodes on [{quad.lower:g}, {quad.upper:g}]"
        total = quad.n_points**d
        return (
            f"{quad.n_points}^{d} = {total} product equispaced nodes "
            f"on [{quad.lower:g}, {quad.upper:g}]^{d}"
        )

    def _factors_label(self) -> str:
        """Concise description of latent factors for multidimensional models."""
        fnames = self._model.factor_names
        return f"{len(fnames)} ({', '.join(fnames)})"

    def summary(self, *, digits: int = 4) -> str:
        """Return a readable plain-text summary of the fitted model.

        Includes the model specification, sample/item/parameter counts, groups, convergence and
        EM evaluation counts, log-likelihood, log-posterior, AIC and BIC (with a note when priors
        are active), the latent distribution table, and the item parameter table with standard
        errors (including delta-method ``b_se`` / ``bk_se`` for unidimensional models).

        Parameters
        ----------
        digits : int, default 4
            Number of decimal places for floating-point values.

        Returns
        -------
        str
        """
        if isinstance(digits, bool) or not isinstance(digits, int) or digits < 0:
            raise SpecificationError(f"digits must be a non-negative integer, got {digits!r}.")
        est_kind = "MAP, priors active" if self._has_priors else "MMLE"
        se_label = "None" if self.se_method is None else self.se_method
        quad_label = self._quadrature_label()
        n_items = len(self._model.item_names)
        aic_val = self._aic_value()
        bic_val = self._bic_value()

        def _fmt(v: float) -> str:
            return f"{v:.{digits}f}"

        latent_str = self.latent.to_string(index=False, float_format=_fmt, na_rep="NaN")
        items_str = self.items.to_string(index=False, float_format=_fmt, na_rep="NaN")

        lines = [
            "flexpyirt FitResult",
            "===================",
            f"Model:          {self._model_label()} ({est_kind})",
            f"Data:           {self.n_obs} observations, {n_items} items, "
            f"{self.n_params} free parameters",
            *(
                [f"Factors:        {self._factors_label()}"]
                if self._model.is_multidimensional
                else []
            ),
            f"Groups:         {self._groups_label()}",
            f"Convergence:    {self._convergence_label()}",
            f"SE method:      {se_label}",
            f"Quadrature:     {quad_label}",
            "",
            "Fit statistics",
            "--------------",
            f"Log-likelihood: {self.loglik:.{digits}f}",
            f"Log-posterior:  {self.log_posterior:.{digits}f}",
            f"AIC:            {aic_val:.{digits}f}",
            f"BIC:            {bic_val:.{digits}f}",
        ]
        if self._has_priors:
            lines.append(_PRIORS_IC_NOTE)
        if self._model.is_empirical_histogram:
            latent_title = "Latent distribution (EmpiricalHistogram)"
        elif self._model.is_davidian:
            latent_title = (
                f"Latent distribution (DavidianCurve, degree={self._model.davidian_degree})"
            )
        elif self._model.is_mixture:
            latent_title = (
                "Latent distribution "
                f"(NormalMixture, n_components={self._model.mixture_components})"
            )
        else:
            latent_title = "Latent distribution"
        lines.extend(
            [
                "",
                latent_title,
                "-" * len(latent_title),
                latent_str,
            ]
        )
        if self.latent_weights is not None:
            weights_str = self.latent_weights.to_string(
                index=False, float_format=_fmt, na_rep="NaN"
            )
            lines.extend(
                [
                    "",
                    "Empirical histogram weights",
                    "---------------------------",
                    weights_str,
                ]
            )
        lines.extend(
            [
                "",
                "Item parameters",
                "---------------",
                items_str,
            ]
        )
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Return an HTML rendering of the summary for Jupyter notebooks (no external assets)."""
        digits = 4
        est_kind = "MAP, priors active" if self._has_priors else "MMLE"
        se_label = "None" if self.se_method is None else self.se_method
        quad_label = self._quadrature_label()
        n_items = len(self._model.item_names)

        meta_rows = [
            ("Model", f"{self._model_label()} ({est_kind})"),
            (
                "Data",
                f"{self.n_obs} observations, {n_items} items, {self.n_params} free parameters",
            ),
            *([("Factors", self._factors_label())] if self._model.is_multidimensional else []),
            ("Groups", self._groups_label()),
            ("Convergence", self._convergence_label()),
            ("SE method", se_label),
            ("Quadrature", quad_label),
            ("Log-likelihood", f"{self.loglik:.{digits}f}"),
            ("Log-posterior", f"{self.log_posterior:.{digits}f}"),
            ("AIC", f"{self._aic_value():.{digits}f}"),
            ("BIC", f"{self._bic_value():.{digits}f}"),
        ]
        meta_html = "".join(
            f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in meta_rows
        )
        note_html = (
            f'<div class="flexpyirt-note">{html.escape(_PRIORS_IC_NOTE)}</div>'
            if self._has_priors
            else ""
        )
        if self._model.is_empirical_histogram:
            latent_heading = "Latent distribution (EmpiricalHistogram)"
        elif self._model.is_davidian:
            latent_heading = (
                f"Latent distribution (DavidianCurve, degree={self._model.davidian_degree})"
            )
        elif self._model.is_mixture:
            latent_heading = (
                "Latent distribution "
                f"(NormalMixture, n_components={self._model.mixture_components})"
            )
        else:
            latent_heading = "Latent distribution"
        latent_html = _frame_to_html_table(self.latent, digits=digits)
        weights_section = (
            "<h4>Empirical histogram weights</h4>"
            f"{_frame_to_html_table(self.latent_weights, digits=digits)}"
            if self.latent_weights is not None
            else ""
        )
        items_html = _frame_to_html_table(self.items, digits=digits)
        return (
            f"{_HTML_STYLE}"
            '<div class="flexpyirt-result">'
            '<div class="flexpyirt-header">flexpyirt FitResult</div>'
            f'<table class="flexpyirt-meta"><tbody>{meta_html}</tbody></table>'
            f"{note_html}"
            f"<h4>{html.escape(latent_heading)}</h4>"
            f"{latent_html}"
            f"{weights_section}"
            "<h4>Item parameters</h4>"
            f"{items_html}"
            "</div>"
        )

    def __repr__(self) -> str:
        """Short text summary."""
        status = "converged" if self.converged else "NOT converged"
        post = f", log_posterior={self.log_posterior:.4f}" if self._has_priors else ""
        return (
            f"FitResult(loglik={self.loglik:.4f}{post}, n_params={self.n_params}, "
            f"n_obs={self.n_obs}, {status} in {self.n_iter} iterations, se={self.se_method})"
        )

    # ---- JSON I/O ----------------------------------------------------------------------------

    def to_json(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        indent: int | None = 2,
    ) -> str:
        """Serialize this result to a strict JSON string, optionally writing it to ``path``.

        Writes plain, versioned RFC 8259 JSON (no ``pickle``) with top-level
        ``"format": "flexpyirt.FitResult"``, ``"schema_version": 1``, and the ``flexpyirt``
        version (DESIGN.md §15.26–§15.28). Finite IEEE-754 float64 values are written as
        ``repr``-exact JSON numbers and non-finite values (``NaN``, ``+inf``, ``-inf``) are
        encoded explicitly as ``"NaN"``, ``"Infinity"``, and ``"-Infinity"`` in float-typed
        fields so the output is valid strict JSON and round-trips bit-for-bit.

        Parameters
        ----------
        path : str or os.PathLike, optional
            If provided, also write the UTF-8 JSON text to this path.
        indent : int or None, default 2
            Indentation level passed to :func:`json.dumps` (``None`` for compact output).

        Returns
        -------
        str
            The serialized JSON string.

        Raises
        ------
        SpecificationError
            If ``indent`` is invalid or the result contains an unregistered model/prior type.
        """
        return dump_fit_result(self, path=path, indent=indent)

    @classmethod
    def from_json(cls, source: str | os.PathLike[str]) -> FitResult:
        """Load a :class:`FitResult` from a JSON file path or JSON string without refitting.

        Reconstructs the model specification, validated parameter table (including estimates, SEs,
        and priors), covariance matrix, convergence history, quadrature grid, and estimation
        settings through explicit name registries (never ``eval`` or ``pickle``; DESIGN.md
        §15.26–§15.28). Calls :func:`~flexpyirt._precision.check_x64` at entry because scoring
        requires 64-bit floating point.

        Parameters
        ----------
        source : str or os.PathLike
            Path to a UTF-8 JSON file written by :meth:`to_json`, or the JSON string itself.

        Returns
        -------
        FitResult
            Reconstructed result supporting all attributes, summaries, and :meth:`score`.

        Raises
        ------
        PrecisionError
            If JAX is not configured for 64-bit floating point.
        SpecificationError
            If ``source`` is not valid strict JSON, has an unknown format tag, schema version,
            or registered type name, or fails internal consistency validation.
        FileNotFoundError
            If ``source`` refers to a file path that does not exist.
        """
        return load_fit_result(source)
