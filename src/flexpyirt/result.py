"""Estimation result (Result layer, DESIGN.md §2 ④)."""

from __future__ import annotations

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

from flexpyirt import _core
from flexpyirt._compile import LATENT_ITEM, coerce_groups, coerce_responses, compile_problem
from flexpyirt.errors import FlexpyirtWarning, SpecificationError
from flexpyirt.items import ItemKernel
from flexpyirt.quadrature import Quadrature

if TYPE_CHECKING:
    from flexpyirt.model import Model

__all__ = ["FitResult"]

_eap = jax.jit(_core.eap)


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

    # ---- summaries ---------------------------------------------------------------------------

    @property
    def _has_priors(self) -> bool:
        """Whether at least one free parameter has an active prior."""
        free = self.parameters[self.parameters["free"]]
        return any(p is not None for p in free["prior"])

    @property
    def n_params(self) -> int:
        """Number of free parameters."""
        return int(self._eta.shape[0])

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
        return -2.0 * self.loglik + 2.0 * self.n_params

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
        return -2.0 * self.loglik + self.n_params * float(np.log(self.n_obs))

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

        kernels = {name: m.kernel for name, m in self._model.items}
        irt_cols = list(
            dict.fromkeys(name for _, m in self._model.items for name in m.kernel.irt_names)
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
        """One row per group: latent ``mean``, ``var`` and their SEs."""
        p = self.parameters[self.parameters["item"] == LATENT_ITEM]
        groups = list(dict.fromkeys(p["group"]))
        est = p.pivot(index="group", columns="param", values="estimate")[["mean", "var"]]
        se = p.pivot(index="group", columns="param", values="se")[["mean", "var"]].add_suffix("_se")
        out = pd.concat([est, se], axis=1).loc[groups]
        out.columns.name = None
        return out.reset_index()

    # ---- scoring -----------------------------------------------------------------------------

    def score(
        self,
        data: Any,
        *,
        method: Literal["EAP"] = "EAP",
        group: Any = None,
        item_names: list[str] | None = None,
    ) -> pd.DataFrame:
        r"""Score respondents.

        Parameters
        ----------
        data : DataFrame or array-like
            Responses on the fitted items (NaN is missing).
        method : {"EAP"}
            Expected a posteriori with the respondent's group's estimated latent distribution as
            prior; ``se`` is the posterior SD (Bock & Mislevy, 1982). Other methods arrive in M4.
        group : array-like, optional
            Group labels (required for a multiple-group model).
        item_names : list of str, optional
            Column names when ``data`` is an array.

        Returns
        -------
        DataFrame
            Columns ``theta`` and ``se``, one row per respondent (index follows ``data`` when it
            is a DataFrame).
        """
        if method != "EAP":
            raise SpecificationError(f"Unsupported scoring method {method!r}; M1 supports 'EAP'.")
        y = coerce_responses(data, self._model, item_names=item_names)
        codes, _ = coerce_groups(group, self._model, y.shape[0], require_all=False)
        compiled = compile_problem(self._model, y, codes, self.parameters, self._quadrature)
        theta, sd = _eap(compiled.problem, jax.numpy.asarray(self._eta))
        n = compiled.n_patterns
        inverse = compiled.inverse
        index = data.index if isinstance(data, pd.DataFrame) else None
        return pd.DataFrame(
            {"theta": np.asarray(theta)[:n][inverse], "se": np.asarray(sd)[:n][inverse]},
            index=index,
        )

    def __repr__(self) -> str:
        """Short text summary."""
        status = "converged" if self.converged else "NOT converged"
        post = f", log_posterior={self.log_posterior:.4f}" if self._has_priors else ""
        return (
            f"FitResult(loglik={self.loglik:.4f}{post}, n_params={self.n_params}, "
            f"n_obs={self.n_obs}, {status} in {self.n_iter} iterations, se={self.se_method})"
        )
