"""Estimation result (Result layer, DESIGN.md §2 ④)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import jax
import numpy as np
import pandas as pd

from flexpyirt import _core
from flexpyirt._compile import LATENT_ITEM, coerce_groups, coerce_responses, compile_problem
from flexpyirt.errors import FlexpyirtWarning, SpecificationError
from flexpyirt.quadrature import Quadrature

if TYPE_CHECKING:
    from flexpyirt.model import Model

__all__ = ["FitResult"]

_eap = jax.jit(_core.eap)


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
        """One row per item (per group in a multiple-group model).

        Estimates and SEs by internal parameter, plus the display IRT thresholds: ``b = -d/a``
        (2PL/Rasch/3PL/4PL), ``bk = -dk/a`` (GRM), and step difficulties
        ``bk = -(dk - d{k-1})/a`` with ``d0 = 0`` (GPCM/PCM). Display-only; their SEs (delta
        method) arrive in M4.
        """
        p = self.parameters[self.parameters["item"] != LATENT_ITEM]
        index = ["group", "item"]
        est = p.pivot(index=index, columns="param", values="estimate")
        se = p.pivot(index=index, columns="param", values="se").add_suffix("_se")
        params = list(dict.fromkeys(p["param"]))
        groups = list(dict.fromkeys(p["group"]))
        order = pd.MultiIndex.from_product([groups, list(self._model.item_names)], names=index)
        out = pd.concat([est[params], se[[f"{c}_se" for c in params]]], axis=1).loc[order]
        models = p.drop_duplicates(index).set_index(index)["model"]
        out.insert(0, "model", models.loc[order])
        if "a" in params:
            is_gpcm = out["model"].isin(("GPCM", "PCM"))
            for c in params:
                if c.startswith("d"):
                    step = out[c]
                    if c not in ("d", "d1"):
                        prev = f"d{int(c[1:]) - 1}"
                        if prev in out.columns:
                            step = step.where(~is_gpcm, out[c] - out[prev])
                    out["b" + c[1:]] = -step / out["a"]
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
