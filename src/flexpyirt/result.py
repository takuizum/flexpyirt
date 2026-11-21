"""Estimation result (Result layer, DESIGN.md §2 ④)."""

from __future__ import annotations

import html
import os
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array

from flexpyirt import _core, _scoring
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
from flexpyirt.latent import eh_moments, latent_param_names
from flexpyirt.method import BAEM
from flexpyirt.quadrature import Quadrature

if TYPE_CHECKING:
    from flexpyirt.model import Model

__all__ = ["FitResult"]

ScoreMethod = Literal["EAP", "MAP", "ML", "WLE", "PV"]
_SCORE_METHODS: tuple[str, ...] = ("EAP", "MAP", "ML", "WLE", "PV")

_eap = jax.jit(_core.eap)
_map = jax.jit(_scoring.map_scores)
_ml = jax.jit(_scoring.ml_scores)
_wle = jax.jit(_scoring.wle_scores)
_pv = jax.jit(_scoring.plausible_values, static_argnames=("n_draws",))


def _is_prng_key(key: Any) -> bool:
    """Return True iff ``key`` is a JAX PRNG key (``jax.random.key`` or ``PRNGKey``)."""
    if not isinstance(key, jax.Array):
        return False
    if jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
        return key.ndim == 0
    return bool(key.dtype == jnp.uint32 and key.shape == (2,))


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
    for m, deps in enumerate(kernel.irt_dep_indices):
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


def _df_to_html_table(df: pd.DataFrame, *, digits: int = 4) -> str:
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
        For ``EmpiricalHistogram`` models: columns ``group``, ``mean``, ``var``, ``sd``,
        ``skewness``, ``kurtosis`` (Fisher's excess kurtosis :math:`\mu_4/\sigma^4 - 3`, 0 for
        a normal distribution; DESIGN.md §15.38). See :attr:`latent_weights` for the full
        ``(group, node, weight)`` table.
        """
        if self._model.is_eh:
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
        lat_cols = list(latent_param_names(self._model.n_factors))
        est = p.pivot(index="group", columns="param", values="estimate")[lat_cols]
        se = p.pivot(index="group", columns="param", values="se")[lat_cols].add_suffix("_se")
        out = pd.concat([est, se], axis=1).loc[groups]
        out.columns.name = None
        return out.reset_index()

    @property
    def latent_weights(self) -> pd.DataFrame | None:
        """Per-group EH weights ``(group, node, weight)``, or ``None`` for ``Normal`` models."""
        if not self._model.is_eh:
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

        Supports five scoring methods (DESIGN.md §8, §15.21–§15.25), all dropping missing
        responses from the likelihood:

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

        Parameters
        ----------
        data : DataFrame or array-like
            Responses on the fitted items (NaN is missing).
        method : {"EAP", "MAP", "ML", "WLE", "PV"}, default "EAP"
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
            For unidimensional ``"EAP"``, ``"MAP"``, ``"ML"``, and ``"WLE"``: columns ``theta``
            and ``se``. For unidimensional ``"PV"``: columns ``pv1`` .. ``pv{n_draws}``. For
            multidimensional ``"EAP"`` and ``"MAP"``: columns ``theta_<factor>`` and
            ``se_<factor>`` for each declared factor (``"ML"``, ``"WLE"`` and ``"PV"`` raise
            :class:`SpecificationError` for ``D > 1``). One row per respondent; index follows
            ``data`` when it is a DataFrame.

        Raises
        ------
        PrecisionError
            If JAX is not configured for 64-bit floating point.
        SpecificationError
            If ``method`` is unknown, ``key`` is missing or invalid for ``"PV"``, ``key`` is
            supplied for a non-PV method, or ``n_draws`` is not a positive integer.
        """
        check_x64()
        if method not in _SCORE_METHODS:
            raise SpecificationError(
                f"Unsupported scoring method {method!r}; use one of {_SCORE_METHODS}."
            )
        if method == "MAP" and self._model.is_eh:
            raise SpecificationError(
                "method='MAP' requires a continuous latent prior and is not supported with "
                "EmpiricalHistogram; use method='EAP', 'ML', 'WLE', or 'PV'."
            )
        if self._model.is_multidimensional and method not in ("EAP", "MAP"):
            raise SpecificationError(
                f"Scoring method {method!r} is only implemented for unidimensional models; "
                "use method='EAP' or 'MAP' for multidimensional models."
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
            draws = _pv(compiled.problem, eta, jnp.asarray(inv_pad), key, n_draws=n_draws)
            cols = [f"pv{m}" for m in range(1, n_draws + 1)]
            return pd.DataFrame(np.asarray(draws)[:n_obs], columns=cols, index=index)

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

    def _model_label(self) -> str:
        """Concise description of item models in the specification."""
        counts: dict[str, int] = {}
        for _, m in self._model.items:
            tag = f"{m.name}({m.kernel.n_cat})" if m.name in ("GRM", "GPCM", "PCM") else m.name
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
        latent_title = (
            "Latent distribution (EmpiricalHistogram)"
            if self._model.is_eh
            else "Latent distribution"
        )
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
        latent_heading = (
            "Latent distribution (EmpiricalHistogram)"
            if self._model.is_eh
            else "Latent distribution"
        )
        latent_html = _df_to_html_table(self.latent, digits=digits)
        weights_section = (
            "<h4>Empirical histogram weights</h4>"
            f"{_df_to_html_table(self.latent_weights, digits=digits)}"
            if self.latent_weights is not None
            else ""
        )
        items_html = _df_to_html_table(self.items, digits=digits)
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
