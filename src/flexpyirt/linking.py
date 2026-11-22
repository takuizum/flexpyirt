"""Differentiable IRT scale linking (Result / Python shell layer, DESIGN.md §16.5).

Public entry point :func:`link` and result object :class:`LinkResult` for characteristic-curve
linking of two separately calibrated unidimensional scales via common items (Haebara, 1980;
Stocking & Lord, 1983; Kim & Lee, 2006; Kolen & Brennan, 2014), with delta-method standard errors
from the implicit function theorem via JAX automatic differentiation (Ogasawara, 2001).
"""

from __future__ import annotations

import html
import math
import re
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, overload

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from flexpyirt._compile import LATENT_ITEM
from flexpyirt._core import _latent_log_weights_group
from flexpyirt._linking import (
    LinkingFamily,
    LinkingProblem,
    LinkMethod,
    linking_vcov,
    solve_linking,
    transform_kernel_params,
)
from flexpyirt._precision import check_x64
from flexpyirt.errors import ConvergenceWarning, SpecificationError
from flexpyirt.items import ItemKernel
from flexpyirt.latent import davidian_param_names, latent_param_names
from flexpyirt.result import _HTML_STYLE, FitResult, _frame_to_html_table

__all__ = ["LinkMethod", "LinkResult", "link"]

_LINK_METHODS: tuple[str, ...] = ("haebara", "stocking_lord")
_PV_COL_RE = re.compile(r"^pv\d+$")

_solve_linking = jax.jit(solve_linking, static_argnames=("method", "symmetric"))
_linking_vcov = jax.jit(linking_vcov, static_argnames=("method", "symmetric"))


def _resolve_group(res: FitResult, group: str | None, role: str) -> tuple[str, int]:
    """Return ``(group_name, group_index)`` for ``source`` or ``target``, validating selection."""
    groups = [g for g, _ in res._model.resolved_groups()]
    if res._model.is_multigroup:
        if group is None:
            raise SpecificationError(
                f"{role} is a multiple-group FitResult; pass {role}_group= to select one of "
                f"{groups}."
            )
        if not isinstance(group, str) or group not in groups:
            raise SpecificationError(f"Unknown {role}_group={group!r}; available groups: {groups}.")
        return group, groups.index(group)
    if group is not None:
        raise SpecificationError(
            f"{role}_group= is only valid when {role} is a multiple-group FitResult."
        )
    return groups[0], 0


def _resolve_common_items(
    source: FitResult,
    target: FitResult,
    items: Iterable[str] | Mapping[str, str] | None,
) -> list[tuple[str, str]]:
    """Return validated ``(source_item, target_item)`` pairs of common items."""
    src_names = set(source._model.item_names)
    tgt_names = set(target._model.item_names)

    if items is None:
        pairs = [(name, name) for name in source._model.item_names if name in tgt_names]
        if not pairs:
            raise SpecificationError(
                "No common items found between source and target; pass items= as a "
                "{source_item: target_item} mapping if item names differ."
            )
        return pairs

    if isinstance(items, str):
        raise SpecificationError(
            "items must be an iterable of item names or a {source: target} mapping, not a str."
        )

    if isinstance(items, Mapping):
        if not items:
            raise SpecificationError("link() needs at least one common item.")
        pairs = [(str(k), str(v)) for k, v in items.items()]
        bad_src = [s for s, _ in pairs if s not in src_names]
        if bad_src:
            raise SpecificationError(f"Unknown source items in items=: {bad_src}.")
        bad_tgt = [t for _, t in pairs if t not in tgt_names]
        if bad_tgt:
            raise SpecificationError(f"Unknown target items in items=: {bad_tgt}.")
        seen_tgt: set[str] = set()
        dup_tgt: list[str] = []
        for _, t in pairs:
            if t in seen_tgt and t not in dup_tgt:
                dup_tgt.append(t)
            seen_tgt.add(t)
        if dup_tgt:
            raise SpecificationError(f"Duplicate target items in items= mapping: {dup_tgt}.")
        return pairs

    chosen = list(dict.fromkeys(str(i) for i in items))
    if not chosen:
        raise SpecificationError("link() needs at least one common item.")
    bad_src = [i for i in chosen if i not in src_names]
    if bad_src:
        raise SpecificationError(f"Unknown source items in items=: {bad_src}.")
    bad_tgt = [i for i in chosen if i not in tgt_names]
    if bad_tgt:
        raise SpecificationError(f"Unknown target items in items=: {bad_tgt}.")
    return [(i, i) for i in chosen]


def _fitted_grid(res: FitResult, group_name: str, group_idx: int) -> tuple[np.ndarray, np.ndarray]:
    r"""Return ``(nodes, weights)`` of ``res``'s fitted unidimensional latent distribution.

    The weights are those the EM integrates with (DESIGN.md §4.4, §15.44, §16.4): the fitted
    histogram for ``EmpiricalHistogram``; otherwise
    :math:`w_q=\operatorname{softmax}_q(b_q+\log g(\theta_q;\hat{\boldsymbol\xi}))` with the
    fixed node base weights :math:`b_q` (0 on equispaced grids, nonzero for Gauss–Hermite) and
    the fitted latent density :math:`g` (``Normal`` or the standardized ``DavidianCurve``).
    """
    nodes = res._quadrature.nodes().astype(np.float64)
    if res._model.is_empirical_histogram:
        assert res._eh_weights is not None
        w = np.asarray(res._eh_weights[group_idx], dtype=np.float64)
    else:
        lat = res.parameters[
            (res.parameters["group"] == group_name) & (res.parameters["item"] == LATENT_ITEM)
        ].set_index("param")["estimate"]
        degree = res._model.davidian_degree
        names = davidian_param_names(degree) if degree > 0 else latent_param_names(1)
        params = jnp.asarray(lat.loc[list(names)].to_numpy(dtype=np.float64))
        base = jnp.asarray(res._quadrature.base_log_weights(1), dtype=jnp.float64)
        logw = _latent_log_weights_group(
            jnp.asarray(nodes, dtype=jnp.float64)[:, None], base, params, degree
        )
        w = np.exp(np.asarray(logw, dtype=np.float64))
    return nodes, w / float(np.sum(w))


def _coerce_1d_float(arr: Any, name: str) -> np.ndarray:
    """Validate and convert ``arr`` to a non-empty finite 1-D float64 NumPy array."""
    try:
        out = np.asarray(arr, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SpecificationError(f"{name} must be a 1-D numeric array.") from exc
    if out.ndim != 1 or out.size == 0:
        raise SpecificationError(f"{name} must be a non-empty 1-D array, got shape {out.shape}.")
    if not np.all(np.isfinite(out)):
        raise SpecificationError(f"{name} must contain only finite values.")
    return out


def _resolve_grids(
    source: FitResult,
    src_gname: str,
    src_gidx: int,
    target: FitResult,
    tgt_gname: str,
    tgt_gidx: int,
    nodes: Any,
    weights: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(tgt_nodes, tgt_weights, src_nodes, src_weights)`` normalized to sum to 1."""
    if nodes is None and weights is None:
        tgt_nodes, tgt_weights = _fitted_grid(target, tgt_gname, tgt_gidx)
        src_nodes, src_weights = _fitted_grid(source, src_gname, src_gidx)
        return tgt_nodes, tgt_weights, src_nodes, src_weights

    if nodes is not None:
        u_nodes = _coerce_1d_float(nodes, "nodes")
        if weights is None:
            u_weights = np.full(u_nodes.shape, 1.0 / u_nodes.size, dtype=np.float64)
        else:
            u_weights = _coerce_1d_float(weights, "weights")
            if u_weights.shape != u_nodes.shape:
                raise SpecificationError(
                    f"weights shape {u_weights.shape} must match nodes shape {u_nodes.shape}."
                )
            if np.any(u_weights < 0.0) or float(np.sum(u_weights)) <= 0.0:
                raise SpecificationError("weights must be non-negative with a positive sum.")
            u_weights = u_weights / float(np.sum(u_weights))
        return u_nodes, u_weights, u_nodes, u_weights

    # nodes is None and weights is not None
    tgt_nodes, _ = _fitted_grid(target, tgt_gname, tgt_gidx)
    u_weights = _coerce_1d_float(weights, "weights")
    if u_weights.shape != tgt_nodes.shape:
        raise SpecificationError(
            f"weights length {u_weights.size} must match target's quadrature length "
            f"{tgt_nodes.size} when nodes= is omitted."
        )
    if np.any(u_weights < 0.0) or float(np.sum(u_weights)) <= 0.0:
        raise SpecificationError("weights must be non-negative with a positive sum.")
    u_weights = u_weights / float(np.sum(u_weights))
    return tgt_nodes, u_weights, tgt_nodes, u_weights


def _extract_item_table(
    res: FitResult, group_name: str
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return ``(estimates, free_ids)`` dicts keyed by item name for one group."""
    sub = res.parameters[
        (res.parameters["group"] == group_name) & (res.parameters["item"] != LATENT_ITEM)
    ]
    est_piv = sub.pivot(index="item", columns="param", values="estimate")
    fid_piv = sub.pivot(index="item", columns="param", values="free_id")
    est_map: dict[str, np.ndarray] = {}
    fid_map: dict[str, np.ndarray] = {}
    for name, item_model in res._model.items:
        pnames = list(res._model.item_kernel(item_model).param_names)
        est_map[name] = est_piv.loc[name, pnames].to_numpy(dtype=np.float64)
        fid_map[name] = fid_piv.loc[name, pnames].to_numpy(dtype=np.int32)
    return est_map, fid_map


def _mean_mean_start(
    pairs: list[tuple[str, str]],
    kernels: dict[str, ItemKernel],
    src_est: dict[str, np.ndarray],
    tgt_est: dict[str, np.ndarray],
) -> np.ndarray:
    r"""Mean/Mean starting values :math:`(A_0, B_0)` (Loyd & Hoover, 1980; Kolen & Brennan, 2014).

    .. math::
        A_0 = \frac{\sum_{j\in\mathcal{C}} |a_j^{\mathrm{src}}|}
                   {\sum_{j\in\mathcal{C}} |a_j^{\mathrm{tgt}}|},
        \qquad
        B_0 = \bar b^{\mathrm{tgt}} - A_0 \bar b^{\mathrm{src}},

    computed across all common items (falling back to ``(1.0, 0.0)`` when slopes vanish). These
    are starting values only: absolute slopes are summed, and :math:`\bar b` pools every display
    threshold of every common item (one per dichotomous item, :math:`K_j - 1` per polytomous
    item), so polytomous items weigh more than in per-item Mean/Mean averages.
    """
    sum_s = sum(abs(float(src_est[s][0])) for s, _ in pairs)
    sum_t = sum(abs(float(tgt_est[t][0])) for _, t in pairs)
    a0 = sum_s / sum_t if (sum_s > 1e-8 and sum_t > 1e-8) else 1.0
    if not (math.isfinite(a0) and a0 > 0.0):
        a0 = 1.0

    b_src: list[float] = []
    b_tgt: list[float] = []
    for s, t in pairs:
        if abs(float(src_est[s][0])) > 1e-8 and abs(float(tgt_est[t][0])) > 1e-8:
            k = kernels[s]
            bs = np.asarray(k.irt_thresholds(jnp.asarray(src_est[s])), dtype=float)
            bt = np.asarray(k.irt_thresholds(jnp.asarray(tgt_est[t])), dtype=float)
            if np.all(np.isfinite(bs)) and np.all(np.isfinite(bt)):
                b_src.extend(bs.tolist())
                b_tgt.extend(bt.tolist())
    b0 = float(np.mean(b_tgt) - a0 * np.mean(b_src)) if b_src else 0.0
    if not math.isfinite(b0):
        b0 = 0.0
    return np.array([a0, b0], dtype=np.float64)


def _transformed_items_table(
    source: FitResult,
    src_est: dict[str, np.ndarray],
    common_src: set[str],
    a_const: float,
    b_const: float,
) -> pd.DataFrame:
    """Transform all source items to the target scale, adding display IRT thresholds."""
    a_arr = jnp.asarray(a_const, dtype=jnp.float64)
    b_arr = jnp.asarray(b_const, dtype=jnp.float64)

    param_cols = list(
        dict.fromkeys(
            p for _, m in source._model.items for p in source._model.item_kernel(m).param_names
        )
    )
    irt_cols = list(
        dict.fromkeys(
            p for _, m in source._model.items for p in source._model.item_kernel(m).irt_names
        )
    )
    rows: list[dict[str, Any]] = []
    for name, item_model in source._model.items:
        kernel = source._model.item_kernel(item_model)
        xi_star = transform_kernel_params(
            kernel, jnp.asarray(src_est[name], dtype=jnp.float64), a_arr, b_arr
        )
        b_star = kernel.irt_thresholds(xi_star)
        row: dict[str, Any] = {
            "item": name,
            "model": item_model.name,
            "common": name in common_src,
        }
        for col in param_cols:
            row[col] = math.nan
        for p, val in zip(kernel.param_names, np.asarray(xi_star, dtype=float), strict=True):
            row[p] = float(val)
        for col in irt_cols:
            row[col] = math.nan
        for p, val in zip(kernel.irt_names, np.asarray(b_star, dtype=float), strict=True):
            row[p] = float(val)
        rows.append(row)
    return pd.DataFrame(rows, columns=["item", "model", "common", *param_cols, *irt_cols])


@dataclass(frozen=True)
class LinkResult:
    r"""Result of :func:`flexpyirt.link` for linear transformation :math:`\theta^* = A\theta + B`.

    Attributes
    ----------
    A : float
        Slope linking constant (:math:`A > 0`).
    B : float
        Intercept linking constant.
    A_se : float
        Delta-method standard error of ``A`` (Ogasawara, 2001), or ``NaN`` if unavailable.
    B_se : float
        Delta-method standard error of ``B`` (Ogasawara, 2001), or ``NaN`` if unavailable.
    vcov : ndarray or None
        ``(2, 2)`` asymptotic covariance matrix of ``(A, B)``, or ``None`` if unavailable (the
        optimizer did not converge, ``se=None`` on a side with free common-item parameters, all
        common parameters are fixed, or the criterion Hessian is not positive definite).
    criterion : float
        Minimized Haebara or Stocking–Lord criterion value at ``(A, B)``.
    method : {"haebara", "stocking_lord"}
        Linking criterion used.
    symmetric : bool
        Whether the symmetric (two-direction) criterion was minimized.
    converged : bool
        Whether the Newton optimizer met ``grad_norm <= tol``.
    n_iter : int
        Number of Newton iterations performed.
    grad_norm : float
        Infinity norm :math:`\|\nabla F(A, B)\|_\infty` at the returned ``(A, B)``.
    common_items : tuple of str
        Common item names (on the source scale) used in the linking criterion.
    items : DataFrame
        All source items transformed onto the target scale (columns ``item``, ``model``,
        ``common``, internal slope–intercept parameters ``a``, ``d``/``d1..``, ``g``, ``u``, and
        display IRT thresholds ``b``/``b1..``). Directly usable with :meth:`Model.fix_from`.
        For fixed-slope models (Rasch, PCM) the transformed slope is :math:`a^* = 1/A`, not 1:
        the items keep their response curves on the target scale, so ``fix_from`` fixes the slope
        at :math:`1/A` (they are unit-slope items again only when :math:`A = 1`).
    """

    A: float
    B: float
    A_se: float
    B_se: float
    vcov: np.ndarray | None
    criterion: float
    method: LinkMethod
    symmetric: bool
    converged: bool
    n_iter: int
    grad_norm: float
    common_items: tuple[str, ...]
    items: pd.DataFrame
    _source_group: str = field(repr=False)
    _target_group: str = field(repr=False)

    @overload
    def transform_theta(self, theta: float) -> float: ...

    @overload
    def transform_theta(self, theta: pd.Series[Any]) -> pd.Series[Any]: ...

    @overload
    def transform_theta(self, theta: pd.DataFrame) -> pd.DataFrame: ...

    @overload
    def transform_theta(self, theta: np.ndarray | jax.Array) -> np.ndarray: ...

    def transform_theta(self, theta: Any) -> Any:
        r"""Transform ability estimates from the source scale to the target scale.

        Applies :math:`\theta^* = A\theta + B` to a ``theta`` column and to every plausible-value
        column ``pv1``, ``pv2``, ... (both are transformed when both are present), and
        :math:`\mathrm{SE}^* = A\,\mathrm{SE}` to an ``se`` column. :math:`(A, B)` are treated as
        known, so the transformed SEs exclude linking error (the sampling variability of
        :math:`(\hat A, \hat B)` summarized by :attr:`vcov`).

        Parameters
        ----------
        theta : float, ndarray, Series, or DataFrame
            Source-scale ability scores. Accepts:

            - a :class:`pandas.DataFrame` returned by :meth:`FitResult.score` (either columns
              ``theta`` and optional ``se``, or plausible-value columns ``pv1``, ``pv2``, ...);
            - a :class:`pandas.Series`;
            - a scalar float/int;
            - a NumPy / JAX array or array-like.

        Returns
        -------
        float, ndarray, Series, or DataFrame
            Transformed scores on the target scale (same container type as ``theta``).

        Raises
        ------
        SpecificationError
            If ``theta`` is a boolean or a DataFrame lacking ``theta`` / ``pv*`` columns.
        """
        if isinstance(theta, bool):
            raise SpecificationError("transform_theta() does not accept a boolean.")
        if isinstance(theta, pd.DataFrame):
            out = theta.copy()
            pv_cols = [c for c in out.columns if isinstance(c, str) and _PV_COL_RE.match(c)]
            if "theta" not in out.columns and not pv_cols:
                raise SpecificationError(
                    "DataFrame passed to transform_theta() must contain a 'theta' column or "
                    "plausible-value columns ('pv1', ...)."
                )
            if "theta" in out.columns:
                out["theta"] = self.A * out["theta"].astype(float) + self.B
                if "se" in out.columns:
                    out["se"] = self.A * out["se"].astype(float)
            for col in pv_cols:
                out[col] = self.A * out[col].astype(float) + self.B
            return out
        if isinstance(theta, pd.Series):
            return self.A * theta.astype(float) + self.B
        if isinstance(theta, (int, float, np.number)):
            return float(self.A * float(theta) + self.B)
        arr = np.asarray(theta, dtype=np.float64)
        return self.A * arr + self.B

    def _constants_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"constant": "A", "estimate": self.A, "se": self.A_se},
                {"constant": "B", "estimate": self.B, "se": self.B_se},
            ]
        )

    def summary(self, *, digits: int = 4) -> str:
        """Return a readable plain-text summary of the linking result.

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

        def _fmt(v: float) -> str:
            return f"{v:.{digits}f}"

        sym_label = "symmetric" if self.symmetric else "non-symmetric"
        status = "Converged" if self.converged else "NOT converged"
        const_str = self._constants_table().to_string(index=False, float_format=_fmt, na_rep="NaN")
        items_str = self.items.to_string(index=False, float_format=_fmt, na_rep="NaN")
        lines = [
            f"flexpyirt LinkResult ({self.method}, {sym_label})",
            f"  Common items:  {len(self.common_items)} / {len(self.items)} source items",
            f"  Criterion:     {self.criterion:.{max(digits, 6)}g}",
            f"  Convergence:   {status} ({self.n_iter} iterations, max|grad|={self.grad_norm:.2e})",
            "",
            "Linking constants (theta* = A * theta + B):",
            const_str,
            "",
            "Transformed source items (target scale):",
            items_str,
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        """Concise one-line summary."""
        sym = "symmetric" if self.symmetric else "non-symmetric"
        status = "converged" if self.converged else "NOT converged"
        return (
            f"LinkResult(method={self.method!r}, {sym}, A={self.A:.4f}, B={self.B:.4f}, "
            f"criterion={self.criterion:.4g}, {status} in {self.n_iter} iters)"
        )

    def _repr_html_(self) -> str:
        """HTML summary for Jupyter notebooks."""
        sym = "symmetric" if self.symmetric else "non-symmetric"
        status = "Converged" if self.converged else "NOT converged"
        meta = [
            ("Method", f"{self.method} ({sym})"),
            ("Common items", f"{len(self.common_items)} / {len(self.items)} source items"),
            ("Criterion", f"{self.criterion:.6g}"),
            ("Convergence", f"{status} ({self.n_iter} iterations, max|grad|={self.grad_norm:.2e})"),
        ]
        meta_rows = "".join(
            f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in meta
        )
        const_html = _frame_to_html_table(self._constants_table(), digits=4)
        items_html = _frame_to_html_table(self.items, digits=4)
        return (
            f"{_HTML_STYLE}"
            '<div class="flexpyirt-result">'
            '<div class="flexpyirt-header">flexpyirt LinkResult</div>'
            f'<table class="flexpyirt-meta"><tbody>{meta_rows}</tbody></table>'
            "<h4>Linking constants (theta* = A * theta + B)</h4>"
            f"{const_html}"
            "<h4>Transformed source items (target scale)</h4>"
            f"{items_html}"
            "</div>"
        )


def link(
    source: FitResult,
    target: FitResult,
    *,
    items: Iterable[str] | Mapping[str, str] | None = None,
    method: LinkMethod = "stocking_lord",
    symmetric: bool = False,
    nodes: Any = None,
    weights: Any = None,
    source_group: str | None = None,
    target_group: str | None = None,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> LinkResult:
    r"""Link a separately calibrated ``source`` scale onto ``target`` via common items.

    Estimates linear transformation constants :math:`(A, B)` with :math:`A > 0` such that
    :math:`\theta^* = A\theta + B` maps the ``source`` (new) scale onto the ``target`` (base)
    scale by minimizing a characteristic-curve criterion over common items (Kolen & Brennan, 2014;
    DESIGN.md §16.5):

    - ``method="stocking_lord"`` (Stocking & Lord, 1983; polytomous form Kim & Lee, 2006):
      weighted squared difference of common-item test characteristic curves
      :math:`\mathrm{TCC}(\theta) = \sum_{j\in\mathcal{C}}\sum_{k=0}^{K_j-1} k\,P_{jk}(\theta)`.
    - ``method="haebara"`` (Haebara, 1980; polytomous form Kim & Lee, 2006):
      weighted sum over common items and categories of squared category-curve differences
      :math:`[P_{jk}^{\mathrm{tgt}}(\theta) - P_{jk}^{\mathrm{src}}((\theta-B)/A)]^2`.
    - ``symmetric=True`` (Kim & Lee, 2006): adds the reverse-direction squared difference
      comparing source curves on the source scale against target curves at :math:`A\theta + B`.

    Haebara sums over all :math:`K_j` categories, including both categories of a dichotomous
    item (the Kim & Lee, 2006 convention). For a dichotomous item this is
    :math:`2[P^{\mathrm{tgt}} - P^{\mathrm{src}}]^2`, twice Haebara's (1980) original term: the
    solution is unchanged when all common items are dichotomous, reported criterion values are
    doubled, and in mixed-format sets dichotomous items carry the Kim & Lee weight.

    Derivatives for Newton optimization and delta-method standard errors (Ogasawara, 2001) are
    computed by JAX automatic differentiation. The standard errors assume ``source`` and
    ``target`` are independent calibrations; when the same :class:`FitResult` is passed as both
    (two groups of one multiple-group fit), the cross-covariance block of its joint ``vcov`` is
    included so shared (equality-constrained) parameters are handled correctly. Two *distinct*
    ``FitResult`` objects fitted to overlapping data are treated as independent. Quadrature
    weights are held fixed, which is exact to first order under a correctly specified model
    (DESIGN.md §16.5). No standard errors are reported when the optimizer did not converge.

    Validated against base-R references (VALIDATION.md V20) for 2PL, 3PL, GRM, GPCM, and mixed
    2PL + GPCM common items; the parameter transforms and criteria for Rasch, 4PL and PCM are
    covered by exact-recovery unit tests only.

    Parameters
    ----------
    source : FitResult
        Fitted unidimensional model on the source (new) scale.
    target : FitResult
        Fitted unidimensional model on the target (base) scale.
    items : iterable of str or mapping of str to str, optional
        Common items to link on. Default (``None``) uses all items sharing the same name in
        ``source`` and ``target``. Pass a ``{source_item: target_item}`` mapping when common items
        have different names in the two forms. Every common pair must share the same item kernel
        and number of categories.
    method : {"stocking_lord", "haebara"}, default "stocking_lord"
        Characteristic-curve linking criterion.
    symmetric : bool, default False
        If ``True``, minimize the symmetric (two-direction) criterion (Kim & Lee, 2006).
    nodes : array-like, optional
        1-D quadrature nodes :math:`\theta_q` at which curves are compared. Defaults to the target
        group's fitted quadrature grid (and the source group's grid for the reverse direction when
        ``symmetric=True``).
    weights : array-like, optional
        Non-negative quadrature weights :math:`w_q` matching ``nodes`` (normalized to sum to 1).
        Defaults to the fitted latent distribution weights when ``nodes`` is omitted, or uniform
        weights ``1 / len(nodes)`` when ``nodes`` is given without ``weights``.
    source_group : str, optional
        Group name when ``source`` is a multiple-group :class:`FitResult`.
    target_group : str, optional
        Group name when ``target`` is a multiple-group :class:`FitResult`.
    tol : float, default 1e-10
        Absolute (not rescaled) convergence tolerance on :math:`\|\nabla F(A, B)\|_\infty`; the
        criterion uses weights summing to 1 and grows with the number of common items, so the
        default suits typical anchor sets and may need loosening for very long ones.
    max_iter : int, default 200
        Maximum number of Newton iterations.

    Returns
    -------
    LinkResult
        Estimated ``A``, ``B``, delta-method SEs (``A_se``, ``B_se``, ``vcov``), minimized
        ``criterion``, convergence diagnostics, transformed ``items`` table, and
        :meth:`LinkResult.transform_theta`.

    Raises
    ------
    PrecisionError
        If JAX is not configured for 64-bit floating point.
    SpecificationError
        If either input is not a unidimensional :class:`FitResult`, a multiple-group fit is
        passed without selecting a group, no common items exist, common items have mismatched
        kernels/categories, or options/nodes/weights are invalid.
    """
    check_x64()
    if not isinstance(source, FitResult):
        raise SpecificationError(f"source must be a FitResult, got {type(source).__name__}.")
    if not isinstance(target, FitResult):
        raise SpecificationError(f"target must be a FitResult, got {type(target).__name__}.")
    if source._model.is_multidimensional or target._model.is_multidimensional:
        raise SpecificationError("link() supports unidimensional models only.")
    if method not in _LINK_METHODS:
        raise SpecificationError(f"Unknown method={method!r}; use one of {_LINK_METHODS}.")
    if not isinstance(symmetric, bool):
        raise SpecificationError(f"symmetric must be a bool, got {symmetric!r}.")
    if (
        isinstance(tol, bool)
        or not isinstance(tol, (int, float))
        or not (math.isfinite(tol) and tol > 0.0)
    ):
        raise SpecificationError(f"tol must be a positive finite number, got {tol!r}.")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise SpecificationError(f"max_iter must be a positive integer, got {max_iter!r}.")

    src_gname, src_gidx = _resolve_group(source, source_group, "source")
    tgt_gname, tgt_gidx = _resolve_group(target, target_group, "target")
    pairs = _resolve_common_items(source, target, items)

    src_lookup = dict(source._model.items)
    tgt_lookup = dict(target._model.items)
    common_kernels: dict[str, ItemKernel] = {}
    for s_name, t_name in pairs:
        sk = source._model.item_kernel(src_lookup[s_name])
        tk = target._model.item_kernel(tgt_lookup[t_name])
        if sk != tk:
            raise SpecificationError(
                f"Common item {s_name!r} (source: {src_lookup[s_name].name}, "
                f"n_categories={sk.n_categories}) and {t_name!r} (target: "
                f"{tgt_lookup[t_name].name}, n_categories={tk.n_categories}) do not "
                "share the same item kernel and category count."
            )
        common_kernels[s_name] = sk

    tgt_nodes, tgt_weights, src_nodes, src_weights = _resolve_grids(
        source, src_gname, src_gidx, target, tgt_gname, tgt_gidx, nodes, weights
    )
    src_est, src_fid = _extract_item_table(source, src_gname)
    tgt_est, tgt_fid = _extract_item_table(target, tgt_gname)

    src_fids = sorted({int(f) for s, _ in pairs for f in src_fid[s] if f >= 0})
    tgt_fids = sorted({int(f) for _, t in pairs for f in tgt_fid[t] if f >= 0})
    src_fid_pos = {fid: idx for idx, fid in enumerate(src_fids)}
    tgt_fid_pos = {fid: idx for idx, fid in enumerate(tgt_fids)}

    eta_s = (
        np.asarray(source._eta[src_fids], dtype=np.float64)
        if src_fids
        else np.zeros((0,), dtype=np.float64)
    )
    eta_t = (
        np.asarray(target._eta[tgt_fids], dtype=np.float64)
        if tgt_fids
        else np.zeros((0,), dtype=np.float64)
    )

    by_kernel: dict[ItemKernel, list[tuple[str, str]]] = {}
    for s_name, t_name in pairs:
        by_kernel.setdefault(common_kernels[s_name], []).append((s_name, t_name))

    families: list[LinkingFamily] = []
    for kernel, k_pairs in by_kernel.items():
        s_params = np.stack([src_est[s] for s, _ in k_pairs], axis=0)
        t_params = np.stack([tgt_est[t] for _, t in k_pairs], axis=0)
        s_idx = np.stack(
            [
                np.array([src_fid_pos.get(int(f), -1) for f in src_fid[s]], dtype=np.int32)
                for s, _ in k_pairs
            ],
            axis=0,
        )
        t_idx = np.stack(
            [
                np.array([tgt_fid_pos.get(int(f), -1) for f in tgt_fid[t]], dtype=np.int32)
                for _, t in k_pairs
            ],
            axis=0,
        )
        families.append(
            LinkingFamily(
                kernel=kernel,
                src_params=jnp.asarray(s_params, dtype=jnp.float64),
                tgt_params=jnp.asarray(t_params, dtype=jnp.float64),
                src_free_index=jnp.asarray(s_idx, dtype=jnp.int32),
                tgt_free_index=jnp.asarray(t_idx, dtype=jnp.int32),
            )
        )

    prob = LinkingProblem(
        families=tuple(families),
        tgt_nodes=jnp.asarray(tgt_nodes, dtype=jnp.float64),
        tgt_weights=jnp.asarray(tgt_weights, dtype=jnp.float64),
        src_nodes=jnp.asarray(src_nodes, dtype=jnp.float64),
        src_weights=jnp.asarray(src_weights, dtype=jnp.float64),
    )
    eta_s_j = jnp.asarray(eta_s, dtype=jnp.float64)
    eta_t_j = jnp.asarray(eta_t, dtype=jnp.float64)
    ab0 = jnp.asarray(_mean_mean_start(pairs, common_kernels, src_est, tgt_est), dtype=jnp.float64)

    ab_hat_j, crit_j, grad_norm_j, n_iter_j, conv_j = _solve_linking(
        prob,
        eta_s_j,
        eta_t_j,
        ab0,
        method=method,
        symmetric=symmetric,
        tol=float(tol),
        max_iter=int(max_iter),
    )
    ab_hat = np.asarray(ab_hat_j, dtype=float)
    a_const, b_const = float(ab_hat[0]), float(ab_hat[1])
    criterion = float(crit_j)
    grad_norm = float(grad_norm_j)
    n_iter = int(n_iter_j)
    converged = bool(conv_j)

    if not converged:
        warnings.warn(
            f"link({method!r}, symmetric={symmetric}) did not converge in {n_iter} iterations "
            f"(max|grad| = {grad_norm:.3e} > tol = {tol:.3e}).",
            ConvergenceWarning,
            stacklevel=2,
        )

    # Delta-method standard errors (Ogasawara, 2001) when covariances are available. They are only
    # valid at a stationary point, so a non-converged solution reports no SEs.
    vcov_out: np.ndarray | None = None
    a_se = math.nan
    b_se = math.nan
    has_free = bool(src_fids or tgt_fids)
    have_src_cov = (not src_fids) or (source.vcov is not None)
    have_tgt_cov = (not tgt_fids) or (target.vcov is not None)
    same_fit = source is target
    if converged and has_free and have_src_cov and have_tgt_cov:
        vcov_s = (
            np.asarray(source.vcov[np.ix_(src_fids, src_fids)], dtype=np.float64)
            if src_fids and source.vcov is not None
            else np.zeros((0, 0), dtype=np.float64)
        )
        vcov_t = (
            np.asarray(target.vcov[np.ix_(tgt_fids, tgt_fids)], dtype=np.float64)
            if tgt_fids and target.vcov is not None
            else np.zeros((0, 0), dtype=np.float64)
        )
        # One calibration on both sides (e.g. two groups of a multiple-group fit): the two scales'
        # estimates are correlated (shared free_ids, common latent/anchor parameters), so the
        # joint covariance's cross block enters; independent fits have zero cross-covariance.
        vcov_st = (
            jnp.asarray(source.vcov[np.ix_(src_fids, tgt_fids)], dtype=jnp.float64)
            if same_fit and src_fids and tgt_fids and source.vcov is not None
            else None
        )
        cov_j, valid_j = _linking_vcov(
            ab_hat_j,
            eta_s_j,
            eta_t_j,
            jnp.asarray(vcov_s, dtype=jnp.float64),
            jnp.asarray(vcov_t, dtype=jnp.float64),
            prob,
            method=method,
            symmetric=symmetric,
            vcov_st=vcov_st,
        )
        if bool(valid_j):
            vcov_out = np.asarray(cov_j, dtype=float)
            a_se = math.sqrt(max(float(vcov_out[0, 0]), 0.0))
            b_se = math.sqrt(max(float(vcov_out[1, 1]), 0.0))

    common_src_names = tuple(s for s, _ in pairs)
    items_df = _transformed_items_table(source, src_est, set(common_src_names), a_const, b_const)

    return LinkResult(
        A=a_const,
        B=b_const,
        A_se=a_se,
        B_se=b_se,
        vcov=vcov_out,
        criterion=criterion,
        method=method,
        symmetric=symmetric,
        converged=converged,
        n_iter=n_iter,
        grad_norm=grad_norm,
        common_items=common_src_names,
        items=items_df,
        _source_group=src_gname,
        _target_group=tgt_gname,
    )
