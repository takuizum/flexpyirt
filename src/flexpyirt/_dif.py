"""Multiple-group likelihood-ratio and Wald DIF tests (Result layer, DESIGN.md §16.2, §16.20).

Post-fit Python orchestration: each likelihood-ratio test refits the model with one change to the
parameter table (its parameters made equal across groups, or freed), and compares the two
marginal log-likelihoods; a Wald test uses the covariance matrix of the fit and needs no refit.
Nothing here runs inside ``jit`` except the refits themselves.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from flexpyirt._compile import TABLE_COLUMNS
from flexpyirt.errors import ConvergenceWarning, SpecificationError

if TYPE_CHECKING:
    from flexpyirt.result import FitResult

__all__ = [
    "DIFScheme",
    "DIFTest",
    "PAdjust",
    "adjust_p",
    "dif_tests",
    "information_criteria",
    "item_rows",
    "set_shared",
    "wald_statistic",
]

DIFScheme = Literal["add", "drop"]
DIFTest = Literal["likelihood_ratio", "wald"]
PAdjust = Literal["holm", "BH", "none"]

DIF_COLUMNS = (
    "item",
    "converged",
    "aic",
    "sabic",
    "hq",
    "bic",
    "x2",
    "df",
    "p",
    "adj_p",
)
WALD_COLUMNS = ("item", "x2", "df", "p", "adj_p")


def adjust_p(p: np.ndarray, method: PAdjust) -> np.ndarray:
    r"""Adjust p-values for multiple tests, as R's ``p.adjust`` (NaN entries are left out).

    With :math:`p_{(1)}\le\dots\le p_{(m)}`:

    - ``"holm"`` (Holm, 1979): :math:`\tilde p_{(i)}=\max_{k\le i}\min\{1,(m-k+1)\,p_{(k)}\}`
      (mirt's ``DIF()`` default);
    - ``"BH"`` (Benjamini & Hochberg, 1995): :math:`\tilde p_{(i)}=\min_{k\ge i}\min\{1,
      \tfrac{m}{k}\,p_{(k)}\}`;
    - ``"none"``: unchanged.
    """
    p = np.asarray(p, dtype=float)
    if method == "none":
        return p.copy()
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    m = ok.size
    if m == 0:
        return out
    order = ok[np.argsort(p[ok], kind="stable")]
    ranks = np.arange(1, m + 1)
    if method == "holm":
        adj = np.maximum.accumulate((m - ranks + 1) * p[order])
    elif method == "BH":
        adj = np.minimum.accumulate((p[order] * m / ranks)[::-1])[::-1]
    else:
        raise SpecificationError(f"Unknown p_adjust={method!r}; use 'holm', 'BH' or 'none'.")
    out[order] = np.minimum(adj, 1.0)
    return out


def information_criteria(loglik: float, k: float, n: float) -> dict[str, float]:
    r"""AIC, sample-size adjusted BIC, Hannan–Quinn and BIC of one fit (mirt's definitions).

    .. math::
        \mathrm{AIC}=-2\ell+2k,\quad \mathrm{SABIC}=-2\ell+k\log\frac{N+2}{24},\quad
        \mathrm{HQ}=-2\ell+2k\log\log N,\quad \mathrm{BIC}=-2\ell+k\log N

    (Akaike, 1974; Sclove, 1987; Hannan & Quinn, 1979; Schwarz, 1978).
    """
    m2 = -2.0 * loglik
    return {
        "aic": m2 + 2.0 * k,
        "sabic": m2 + k * float(np.log((n + 2.0) / 24.0)),
        "hq": m2 + 2.0 * k * float(np.log(np.log(n))),
        "bic": m2 + k * float(np.log(n)),
    }


def set_shared(table: pd.DataFrame, rows: np.ndarray, *, shared: bool) -> pd.DataFrame:
    """Give the free ``rows`` (one item, several groups) one ``free_id`` per param, or one each."""
    t = table.copy()
    next_id = int(t["free_id"].max()) + 1
    sub = t.loc[rows]
    for _, idx in sub.groupby("param", sort=False).groups.items():
        if shared:
            t.loc[idx, "free_id"] = int(t.loc[idx, "free_id"].min())
        else:
            for k, i in enumerate(idx):
                if k > 0:
                    t.loc[i, "free_id"] = next_id
                    next_id += 1
    return t


def item_rows(table: pd.DataFrame, item: str, params: Sequence[str]) -> np.ndarray:
    """Positions of the free rows of ``params`` of ``item`` (all groups) in ``table``."""
    mask = (table["item"] == item) & table["param"].isin(list(params)) & table["free"]
    return np.flatnonzero(mask.to_numpy())


def wald_statistic(
    eta: np.ndarray, vcov: np.ndarray, table: pd.DataFrame, rows: np.ndarray
) -> tuple[float, int]:
    r"""Wald statistic for the equality across groups of the parameters in ``rows``.

    For each parameter (``param``) among ``rows`` with distinct ``free_id``\ s
    :math:`u_1,\dots,u_m` across groups, the contrasts :math:`\eta_{u_1}-\eta_{u_k}`
    (:math:`k=2,\dots,m`) are the rows of :math:`\mathbf L`, and

    .. math::
        W=(\mathbf L\boldsymbol\eta)^\top(\mathbf L\mathbf V\mathbf L^\top)^{-1}
          (\mathbf L\boldsymbol\eta),\qquad df=\text{number of rows of }\mathbf L

    (Wald, 1943; Lord, 1980), as mirt's ``wald()`` behind ``DIF(Wald = TRUE)``. Returns
    ``(W, df)``; ``df = 0`` when every parameter is already shared, and ``W`` is NaN when
    :math:`\mathbf L\mathbf V\mathbf L^\top` is singular.
    """
    sub = table.iloc[rows]
    contrasts = []
    for _, ids in sub.groupby("param", sort=False)["free_id"]:
        unique = list(dict.fromkeys(int(v) for v in ids))
        for other in unique[1:]:
            row = np.zeros(eta.shape[0])
            row[unique[0]] = 1.0
            row[other] = -1.0
            contrasts.append(row)
    if not contrasts:
        return float("nan"), 0
    contrast = np.vstack(contrasts)
    diff = contrast @ eta
    middle = contrast @ vcov @ contrast.T
    try:
        stat = float(diff @ np.linalg.solve(middle, diff))
    except np.linalg.LinAlgError:
        stat = float("nan")
    return stat, len(contrasts)


def _wald_tests(
    result: FitResult, tested: list[str], params: Sequence[str] | None, p_adjust: PAdjust
) -> pd.DataFrame:
    """Wald DIF tests of :meth:`FitResult.dif` with ``test="wald"`` (DESIGN.md §16.20)."""
    if result.vcov is None:
        raise SpecificationError(
            "dif(test='wald') needs the covariance matrix of the estimates; fit with se= "
            "(e.g. se='oakes')."
        )
    model = result._model
    lookup = dict(model.items)
    table = result.parameters
    eta = np.asarray(result._eta, dtype=float)
    vcov = np.asarray(result.vcov, dtype=float)
    rows_out = []
    for item in tested:
        names = model.item_kernel(lookup[item]).param_names
        chosen = names if params is None else tuple(params)
        bad = [p for p in chosen if p not in names]
        if bad:
            raise SpecificationError(f"Item {item!r} has no parameters {bad}.")
        stat, df = wald_statistic(eta, vcov, table, item_rows(table, item, chosen))
        if df == 0:
            raise SpecificationError(
                f"Item {item!r}: parameters {list(chosen)} are already equal across groups; "
                "nothing to test."
            )
        p = float(stats.chi2.sf(stat, df)) if np.isfinite(stat) else float("nan")
        rows_out.append({"item": item, "x2": stat, "df": int(df), "p": p})
    out = pd.DataFrame(rows_out)
    out["adj_p"] = adjust_p(out["p"].to_numpy(), p_adjust)
    return out[list(WALD_COLUMNS)]


def dif_tests(
    result: FitResult,
    data: Any,
    *,
    group: Any,
    items: Iterable[str],
    params: Sequence[str] | None,
    scheme: DIFScheme,
    test: DIFTest = "likelihood_ratio",
    p_adjust: PAdjust,
    item_names: list[str] | None,
) -> pd.DataFrame:
    """Run the likelihood-ratio or Wald DIF tests of :meth:`FitResult.dif` (see there)."""
    from flexpyirt.fit import fit

    result._require_unweighted("dif()")
    model = result._model
    if not model.is_multigroup:
        raise SpecificationError("dif() needs a multiple-group model (Model.with_groups()).")
    if result._has_priors:
        raise SpecificationError(
            "dif() compares marginal likelihoods; it is not defined for MAP fits with priors."
        )
    if scheme not in ("add", "drop"):
        raise SpecificationError(f"Unknown scheme={scheme!r}; use 'add' or 'drop'.")
    if test not in ("likelihood_ratio", "wald"):
        raise SpecificationError(f"Unknown test={test!r}; use 'likelihood_ratio' or 'wald'.")
    if p_adjust not in ("holm", "BH", "none"):
        raise SpecificationError(f"Unknown p_adjust={p_adjust!r}; use 'holm', 'BH' or 'none'.")
    if isinstance(items, str):
        raise SpecificationError("items must be an iterable of item names, not a str.")
    tested = list(dict.fromkeys(str(i) for i in items))
    if not tested:
        raise SpecificationError("dif() needs at least one item to test.")
    unknown = [i for i in tested if i not in model.item_names]
    if unknown:
        raise SpecificationError(f"Unknown items: {unknown}.")
    if isinstance(params, str):
        raise SpecificationError("params must be an iterable of parameter names, not a str.")
    if test == "wald":
        if scheme != "add":
            raise SpecificationError(
                "dif(test='wald') tests parameters that are free across groups; use scheme='add'."
            )
        return _wald_tests(result, tested, params, p_adjust)
    if data is None or group is None:
        raise SpecificationError("dif() with the likelihood-ratio test needs data and group=.")
    base = result.parameters[list(TABLE_COLUMNS)].copy()
    # Start every refit at the base estimates (same fixed point, fewer iterations).
    base["value"] = result.parameters["estimate"].to_numpy(dtype=float)
    k_base = result.n_params
    n = result.n_obs
    base_crit = information_criteria(result.loglik, k_base, n)
    lookup = dict(model.items)

    rows_out = []
    for item in tested:
        names = model.item_kernel(lookup[item]).param_names
        chosen = names if params is None else tuple(params)
        bad = [p for p in chosen if p not in names]
        if bad:
            raise SpecificationError(f"Item {item!r} has no parameters {bad}.")
        mask = ((base["item"] == item) & base["param"].isin(list(chosen)) & base["free"]).to_numpy()
        rows = np.flatnonzero(mask)
        table = set_shared(base, rows, shared=(scheme == "add"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            new = fit(
                model,
                data,
                group=group,
                table=table,
                item_names=item_names,
                method=result._method,
                quadrature=result._quadrature,
                se=None,
            )
        for w in caught:
            if not issubclass(w.category, ConvergenceWarning):
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
        k_new = new.n_params
        df = abs(k_base - k_new)
        if df == 0:
            raise SpecificationError(
                f"Item {item!r}: parameters {list(chosen)} are already "
                f"{'equal across groups' if scheme == 'add' else 'free in every group'}; "
                f"nothing to test with scheme={scheme!r}."
            )
        new_crit = information_criteria(new.loglik, k_new, n)
        # mirt's sign: criteria of the model with the item free minus the constrained one, so a
        # negative difference favours DIF in both schemes.
        if scheme == "add":
            ll_free, ll_con, crit_free, crit_con = result.loglik, new.loglik, base_crit, new_crit
        else:
            ll_free, ll_con, crit_free, crit_con = new.loglik, result.loglik, new_crit, base_crit
        x2 = max(0.0, 2.0 * (ll_free - ll_con))
        rows_out.append(
            {
                "item": item,
                "converged": bool(new.converged),
                **{key: crit_free[key] - crit_con[key] for key in ("aic", "sabic", "hq", "bic")},
                "x2": x2,
                "df": int(df),
                "p": float(stats.chi2.sf(x2, df)),
            }
        )
    out = pd.DataFrame(rows_out)
    out["adj_p"] = adjust_p(out["p"].to_numpy(), p_adjust)
    return out[list(DIF_COLUMNS)]
