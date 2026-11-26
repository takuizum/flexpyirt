r"""DIF workflow for multiple-group models: screening, anchor purification and effect sizes.

Result / Python shell layer (DESIGN.md §16.22). :func:`dif_screen` tests every item with one of
three anchor strategies and returns the final partially invariant fit; :func:`dif_effect_sizes`
quantifies the differences of a fitted multiple-group model. Only the refits run in ``jit``.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy import stats

from flexpyirt import _core
from flexpyirt._compile import TABLE_COLUMNS, coerce_groups, coerce_responses, compile_problem
from flexpyirt._dif import DIFTest, PAdjust, adjust_p, item_rows, set_shared, wald_statistic
from flexpyirt._precision import check_x64
from flexpyirt.errors import ConvergenceWarning, SpecificationError
from flexpyirt.fit import SEMethod, fit
from flexpyirt.method import BAEM
from flexpyirt.model import Model
from flexpyirt.quadrature import Quadrature
from flexpyirt.result import FitResult

__all__ = ["DIFScreenResult", "DIFStrategy", "dif_effect_sizes", "dif_screen"]

DIFStrategy = Literal["all_others", "sequential_free", "sequential_constrain"]
EffectDensity = Literal["focal", "reference"]

TABLE_OUT = ("item", "x2", "df", "p", "adj_p", "flagged", "converged")
HISTORY_OUT = ("round", "item", "retested", "x2", "df", "p", "adj_p", "flagged")

_e_step = jax.jit(_core.e_step)
_latent_log_weights = jax.jit(_core._latent_log_weights)


@dataclass(frozen=True)
class DIFScreenResult:
    """Result of :func:`dif_screen` (DESIGN.md §16.22).

    Attributes
    ----------
    table : DataFrame
        One row per tested item: ``item``, ``x2``, ``df``, ``p``, ``adj_p``, ``flagged``,
        ``converged``. For ``"all_others"`` the tests of the only round (each item with all others
        as anchors); for the sequential strategies each item tested against the final anchor set
        (the item free versus constrained, every other item as in the final model). ``adj_p`` is
        adjusted over the table; ``flagged`` is the decision of the procedure. ``converged`` is
        whether the refits of the row's test converged.
    flagged_items : list of str
        Items flagged for DIF, in model order.
    anchor_items : list of str
        Items whose parameters are equal across groups in the final model (every item not
        flagged, including the untested ones), in model order.
    history : DataFrame
        One row per round and tested item: ``round``, ``item``, ``retested`` (tested in that
        round; otherwise the latest statistic is carried over), ``x2``, ``df``, ``p``, ``adj_p``
        (adjusted over the tested items in that round), ``flagged``.
    fit : FitResult
        The final model: the tested parameters of the flagged items free across groups,
        everything else equal; fitted with ``se=``.
    strategy : str
        The anchor strategy.
    test : str
        ``"likelihood_ratio"`` or ``"wald"``.
    n_rounds : int
        Number of rounds of tests.
    converged : bool
        Whether the procedure reached a round that reproduced the flags of the round before
        (always ``True`` for ``"all_others"``).
    """

    table: pd.DataFrame
    flagged_items: list[str]
    anchor_items: list[str]
    history: pd.DataFrame
    fit: FitResult
    strategy: str
    test: str
    n_rounds: int
    converged: bool


def _check_choice(name: str, value: Any, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise SpecificationError(
            f"Unknown {name}={value!r}; use one of {', '.join(repr(c) for c in choices)}."
        )


def _item_list(model: Model, value: Iterable[str] | None, name: str) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        raise SpecificationError(f"{name} must be an iterable of item names, not a str.")
    out = list(dict.fromkeys(str(i) for i in value))
    unknown = [i for i in out if i not in model.item_names]
    if unknown:
        raise SpecificationError(f"Unknown items in {name}: {unknown}.")
    return out


class _Screen:
    """Fits cached by the set of free tested items, and the test of one item in a state."""

    def __init__(
        self,
        spec: Model,
        data: Any,
        group: Any,
        item_names: list[str] | None,
        base: pd.DataFrame,
        rows: dict[str, np.ndarray],
        test: DIFTest,
        method: BAEM | None,
        quadrature: Quadrature | None,
        se_test: SEMethod | None,
    ) -> None:
        self.spec, self.data, self.group, self.item_names = spec, data, group, item_names
        self.base, self.rows, self.test = base, rows, test
        self.method, self.quadrature, self.se_test = method, quadrature, se_test
        self.cache: dict[tuple[frozenset[str], str | None], FitResult] = {}
        self.start: FitResult | None = None

    def table_for(self, free: frozenset[str]) -> pd.DataFrame:
        table = self.base
        for item in sorted(free):
            table = set_shared(table, self.rows[item], shared=False)
        if self.start is not None:
            table = table.copy()
            table["value"] = self.start.parameters["estimate"].to_numpy(dtype=float)
        return table

    def fit_for(self, free: frozenset[str], se: SEMethod | None) -> FitResult:
        key = (free, se)
        if key not in self.cache:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                res = fit(
                    self.spec,
                    self.data,
                    group=self.group,
                    table=self.table_for(free),
                    item_names=self.item_names,
                    method=self.method,
                    quadrature=self.quadrature,
                    se=se,
                )
            for w in caught:
                if not issubclass(w.category, ConvergenceWarning):
                    warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
            self.cache[key] = res
        return self.cache[key]

    def run(self, item: str, free: frozenset[str]) -> dict[str, Any]:
        """Test ``item`` with every other tested item free or constrained as in ``free``."""
        if self.test == "wald":
            big = self.fit_for(free | {item}, self.se_test)
            assert big.vcov is not None
            stat, df = wald_statistic(
                np.asarray(big._eta, dtype=float),
                np.asarray(big.vcov, dtype=float),
                big.parameters,
                item_rows(big.parameters, item, list(self.base_params[item])),
            )
            converged = bool(big.converged)
        else:
            big = self.fit_for(free | {item}, None)
            small = self.fit_for(free - {item}, None)
            stat = max(0.0, 2.0 * (big.loglik - small.loglik))
            df = big.n_params - small.n_params
            converged = bool(big.converged and small.converged)
        p = float(stats.chi2.sf(stat, df)) if df > 0 and np.isfinite(stat) else float("nan")
        return {"x2": float(stat), "df": int(df), "p": p, "converged": converged}

    base_params: dict[str, tuple[str, ...]]


def dif_screen(
    spec: Model,
    data: Any,
    *,
    group: Any,
    items: Iterable[str] | None = None,
    anchor_items: Iterable[str] | None = None,
    strategy: DIFStrategy = "all_others",
    test: DIFTest = "likelihood_ratio",
    params: Sequence[str] | None = None,
    p_adjust: PAdjust = "BH",
    alpha: float = 0.05,
    max_iter: int = 10,
    method: BAEM | None = None,
    quadrature: Quadrature | None = None,
    se: SEMethod | None = None,
    item_names: list[str] | None = None,
) -> DIFScreenResult:
    r"""Screen every item of a multiple-group model for DIF, with anchor purification.

    The models are built from ``spec`` (its :meth:`Model.equal_across_groups` calls are
    replaced): every parameter of an untested item and every parameter outside ``params`` is
    equal across groups, and the ``params`` of a tested item are equal or free per group
    according to the procedure. With :math:`F` the set of free tested items, item :math:`i` is
    tested by comparing :math:`F\cup\{i\}` with :math:`F\setminus\{i\}`:

    .. math::
        X^2=2\,(\ell_{F\cup\{i\}}-\ell_{F\setminus\{i\}}),\qquad
        df=k_{F\cup\{i\}}-k_{F\setminus\{i\}}

    (Thissen, Steinberg & Wainer, 1993), or with the Wald statistic of the fit of
    :math:`F\cup\{i\}` (Wald, 1943; Lord, 1980). Each round adjusts the latest :math:`p` of
    every tested item over the tested items and flags :math:`\tilde p<\alpha`. Strategies
    (Candell & Drasgow, 1988; Kopf, Zeileis & Strobl, 2015), named after mirt's ``DIF()``
    schemes they reproduce:

    - ``"all_others"`` (mirt ``"drop"`` from the fully invariant model): one round,
      :math:`F=\varnothing`; each item is tested with all others as anchors.
    - ``"sequential_free"`` (``"drop_sequential"``): round 1 as ``"all_others"``; then
      :math:`F` = the flagged items and the unflagged items are retested, until the flags
      repeat.
    - ``"sequential_constrain"`` (``"add_sequential"``): starts from the anchors (the untested
      items) with every tested item free; then :math:`F` = the flagged items and the flagged
      items are retested, until the flags repeat.

    Parameters
    ----------
    spec : Model
        A unidimensional multiple-group model (:meth:`Model.with_groups`), typically with the
        focal latent distributions free (``Normal.free()``). No priors and no
        :meth:`Model.equal_across_items` constraints.
    data : DataFrame or array-like
        Responses.
    group : array-like
        Group label of each row.
    items : iterable of str, optional
        Items to test (default: every item not in ``anchor_items``).
    anchor_items : iterable of str, optional
        Items that are always equal across groups and not tested. Items in neither argument
        are anchors too. ``"sequential_constrain"`` needs at least one anchor item.
    strategy : {"all_others", "sequential_free", "sequential_constrain"}, default "all_others"
        Anchor strategy (see above).
    test : {"likelihood_ratio", "wald"}, default "likelihood_ratio"
        The Wald test fits the model with the tested item free once, with standard errors
        (``se``, default ``"oakes"``), instead of two fits; with ``"sequential_constrain"`` one
        fit tests every item of a round.
    params : sequence of str, optional
        Internal parameters tested for DIF (default: every parameter). Names an item does not
        have are ignored for that item; every tested item must have at least one.
    p_adjust : {"BH", "holm", "none"}, default "BH"
        Adjustment of the :math:`p`-values over the tested items (Benjamini & Hochberg, 1995;
        Holm, 1979).
    alpha : float, default 0.05
        Items with adjusted :math:`p<\alpha` are flagged (mirt's numeric ``seq_stat``).
    max_iter : int, default 10
        Maximum number of rounds of the sequential strategies (mirt's ``max_run`` minus 1).
    method, quadrature : optional
        Estimation settings of every fit (as in :func:`~flexpyirt.fit`).
    se : {"oakes", "hessian", "sandwich", "crossprod"}, optional
        Standard errors of the final fit (and of the Wald fits; ``"oakes"`` when ``None``).
    item_names : list of str, optional
        Column names when ``data`` is an array.

    Returns
    -------
    DIFScreenResult

    Raises
    ------
    PrecisionError
        If JAX is not configured for 64-bit floating point.
    SpecificationError
        For a single-group or multidimensional model, priors, equality constraints across
        items, unknown or overlapping items, an item without any of ``params``, no anchor item
        with ``"sequential_constrain"``, or invalid options.
    IdentificationError
        If a model of the procedure is not identified (e.g. every item flagged).

    Warns
    -----
    ConvergenceWarning
        If a sequential strategy has not repeated its flags after ``max_iter`` rounds.

    See Also
    --------
    FitResult.dif : tests of chosen items against a model you build.
    dif_effect_sizes : effect sizes of the final fit (``DIFScreenResult.fit``).
    """
    check_x64()
    if not isinstance(spec, Model):
        raise SpecificationError(f"dif_screen() needs a Model, got {type(spec).__name__}.")
    if not spec.is_multigroup:
        raise SpecificationError("dif_screen() needs a multiple-group model (Model.with_groups()).")
    if spec.is_multidimensional or spec.is_reduced:
        raise SpecificationError("dif_screen() supports unidimensional models only.")
    if spec.priors:
        raise SpecificationError(
            "dif_screen() compares marginal likelihoods; it is not defined with priors."
        )
    if spec.equal_items:
        raise SpecificationError(
            "dif_screen() does not support equality constraints across items "
            "(Model.equal_across_items())."
        )
    _check_choice("strategy", strategy, ("all_others", "sequential_free", "sequential_constrain"))
    _check_choice("test", test, ("likelihood_ratio", "wald"))
    _check_choice("p_adjust", p_adjust, ("holm", "BH", "none"))
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise SpecificationError(f"alpha must be in (0, 1), got {alpha!r}.")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise SpecificationError(f"max_iter must be a positive integer, got {max_iter!r}.")
    anchors_in = _item_list(spec, anchor_items, "anchor_items") or []
    tested_in = _item_list(spec, items, "items")
    if tested_in is None:
        tested_in = [i for i in spec.item_names if i not in anchors_in]
    overlap = [i for i in tested_in if i in anchors_in]
    if overlap:
        raise SpecificationError(f"Items both tested and anchors: {overlap}.")
    tested = [i for i in spec.item_names if i in tested_in]
    if not tested:
        raise SpecificationError("dif_screen() needs at least one item to test.")
    if strategy == "sequential_constrain" and len(tested) == len(spec.item_names):
        raise SpecificationError(
            "strategy='sequential_constrain' starts from anchor items; pass anchor_items= "
            "(or leave some items out of items=)."
        )
    if isinstance(params, str):
        raise SpecificationError("params must be an iterable of parameter names, not a str.")

    lookup = dict(spec.items)
    chosen: dict[str, tuple[str, ...]] = {}
    for item in tested:
        names = spec.item_kernel(lookup[item]).param_names
        chosen[item] = names if params is None else tuple(p for p in params if p in names)
        if not chosen[item]:
            raise SpecificationError(
                f"Item {item!r} has none of the parameters {list(params or ())}."
            )

    invariant = replace(spec, equal=()).equal_across_groups(items=None)
    base = invariant.parameter_table(data, group=group, item_names=item_names)
    base = base[list(TABLE_COLUMNS)].reset_index(drop=True)
    rows = {item: item_rows(base, item, chosen[item]) for item in tested}
    se_test: SEMethod | None = (se or "oakes") if test == "wald" else None
    screen = _Screen(
        invariant, data, group, item_names, base, rows, test, method, quadrature, se_test
    )
    screen.base_params = chosen

    free = frozenset(tested) if strategy == "sequential_constrain" else frozenset()
    flags = free
    latest: dict[str, dict[str, Any]] = {}
    history: list[dict[str, Any]] = []
    converged = True
    n_rounds = 0
    for round_no in range(1, max_iter + 1):
        n_rounds = round_no
        screen.start = screen.fit_for(free, se_test if test == "wald" else None)
        if round_no == 1:
            candidates = tested
        elif strategy == "sequential_free":
            candidates = [i for i in tested if i not in free]
        else:
            candidates = [i for i in tested if i in free]
        for item in candidates:
            latest[item] = screen.run(item, free)
        adj = adjust_p(np.array([latest[i]["p"] for i in tested]), p_adjust)
        new_flags = frozenset(i for i, a in zip(tested, adj, strict=True) if a < alpha)
        for item, a in zip(tested, adj, strict=True):
            history.append(
                {
                    "round": round_no,
                    "item": item,
                    "retested": item in candidates,
                    **{k: latest[item][k] for k in ("x2", "df", "p")},
                    "adj_p": float(a),
                    "flagged": item in new_flags,
                }
            )
        if strategy == "all_others":
            flags = new_flags
            break
        if new_flags == flags:
            break
        flags = new_flags
        free = flags
        if round_no == max_iter:
            converged = False
            warnings.warn(
                f"dif_screen(strategy={strategy!r}) did not repeat its flags within "
                f"max_iter={max_iter} rounds; the result uses the flags of the last round.",
                ConvergenceWarning,
                stacklevel=2,
            )

    if strategy == "all_others":
        rows_out = [{"item": i, **latest[i]} for i in tested]
    else:
        screen.start = screen.fit_for(flags, se_test if test == "wald" else None)
        rows_out = [{"item": i, **screen.run(i, flags)} for i in tested]
    table = pd.DataFrame(rows_out)
    table["adj_p"] = adjust_p(table["p"].to_numpy(), p_adjust)
    table["flagged"] = table["item"].isin(flags)
    final = screen.fit_for(flags, se if se is not None else se_test)
    return DIFScreenResult(
        table=table[list(TABLE_OUT)],
        flagged_items=[i for i in spec.item_names if i in flags],
        anchor_items=[i for i in spec.item_names if i not in flags],
        history=pd.DataFrame(history, columns=list(HISTORY_OUT)),
        fit=final,
        strategy=strategy,
        test=test,
        n_rounds=n_rounds,
        converged=converged,
    )


def _effect_problem(
    res: FitResult, y: np.ndarray, codes: np.ndarray
) -> tuple[_core.CompiledProblem, jax.Array]:
    compiled = compile_problem(
        res._model, y, codes, res.parameters, res._quadrature, eh_weights=res._eh_weights
    )
    return compiled.problem, jnp.asarray(res._eta)


def _expected_scores(res: FitResult, theta: np.ndarray, group_index: int) -> np.ndarray:
    """``(T, J)`` expected item scores of one group at ``theta``."""
    from flexpyirt.information import _curves

    return _curves(res, theta, group_index)[1]


def _essd(focal: np.ndarray, reference: np.ndarray) -> float:
    r"""Cohen's :math:`d` of two equal-size samples (Meade, 2010).

    .. math::
        d=(\bar x-\bar y)\big/\sqrt{(s_x^2+s_y^2)/2}
    """
    if focal.shape[0] < 2:
        return float("nan")
    pooled = np.sqrt((np.var(focal, ddof=1) + np.var(reference, ddof=1)) / 2.0)
    if pooled == 0:
        return float("nan")
    return float((focal.mean() - reference.mean()) / pooled)


def _rmsd(
    res: FitResult, y: np.ndarray, codes: np.ndarray, expected_nodes: np.ndarray
) -> np.ndarray:
    r"""``(G, J)`` RMSD of every group and item (expected-score form, DESIGN.md §16.22).

    .. math::
        \mathrm{RMSD}_{gj}=\Big(\sum_qw_{gq}\big(\hat E_{gj}(\theta_q)-E_{gj}(\theta_q)\big)^2
        \Big)^{1/2},\qquad \hat E_{gj}(\theta_q)=\frac{\sum_kk\,\bar r_{gjkq}}{\sum_k\bar r_{gjkq}}

    (von Davier, 2005; Köhler, Robitzsch & Hartig, 2020; mirt's ``RMSD_DIF(probfun = FALSE)``).
    ``expected_nodes`` is ``(G, Q, J)``.
    """
    prob, eta = _effect_problem(res, y, codes)
    counts, _ = _e_step(prob, eta)
    weights = np.exp(np.asarray(_latent_log_weights(prob, eta)))  # (G, Q)
    n_groups, _, n_items = expected_nodes.shape
    out = np.full((n_groups, n_items), np.nan)
    for fam, r_bar in zip(prob.families, counts.r, strict=True):
        r_np = np.asarray(r_bar)  # (G, J_f, K_f, Q)
        n_cat = r_np.shape[2]
        scores = np.arange(n_cat, dtype=float)
        for pos, col in enumerate(np.asarray(fam.columns)):
            r_item = r_np[:, pos]  # (G, K_f, Q); padded categories carry zero counts
            total = r_item.sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                observed = np.einsum("k,gkq->gq", scores, r_item) / total
            diff2 = (observed - expected_nodes[:, :, int(col)]) ** 2
            out[:, int(col)] = np.sqrt(np.sum(weights * diff2, axis=1))
    return out


def dif_effect_sizes(
    result: FitResult | DIFScreenResult,
    *,
    data: Any = None,
    group: Any = None,
    focal: str | Iterable[str] | None = None,
    items: Iterable[str] | None = None,
    density: EffectDensity = "focal",
    item_names: list[str] | None = None,
) -> pd.DataFrame:
    r"""DIF and DTF effect sizes of a fitted multiple-group model (DESIGN.md §16.22).

    For each focal group :math:`f` against the reference group :math:`r` (the first group) and
    item :math:`j`, with :math:`D_j(\theta)=E_{fj}(\theta)-E_{rj}(\theta)` the difference of
    the expected item scores :math:`E_{gj}(\theta)=\sum_kk\,P_{gjk}(\theta)` and :math:`w_q`
    the normalized fitted latent weights of the focal (``density="reference"``: reference) group
    on the fitted grid :math:`\theta_q`:

    .. math::
        \text{signed}_j=\sum_qw_qD_j(\theta_q),\qquad
        \text{unsigned}_j=\sum_qw_q\,|D_j(\theta_q)|

    (Chalmers, 2018; mirt's ``DRF(DIF = TRUE, best_fitting = TRUE)`` ``sDIF`` / ``uDIF``).
    With ``data``, also

    - the expected-score standardized difference (Meade, 2010; mirt's ``empirical_ES()``
      ``ESSD``) at the EAP scores :math:`\hat\theta_i` of the focal respondents,
      :math:`\big(\overline{E_{fj}(\hat\theta)}-\overline{E_{rj}(\hat\theta)}\big)/
      \sqrt{(s_f^2+s_r^2)/2}` (sample variances);
    - the root mean squared deviation of the reference and the focal group (expected-score form
      of von Davier, 2005, and Köhler, Robitzsch & Hartig, 2020; mirt's
      ``RMSD_DIF(probfun = FALSE)``),
      :math:`\big(\sum_qw_{gq}(\hat E_{gj}(\theta_q)-E_{gj}(\theta_q))^2\big)^{1/2}` with the
      posterior expected item score :math:`\hat E_{gj}` of the group's data. It measures how well
      a group's parameters fit that group, so it detects DIF for items whose parameters are
      equal across groups (fit the invariant model).

    One more row per focal group (``level="test"``) uses the expected score summed over
    ``items`` (signed / unsigned DTF, the expected test score standardized difference; RMSD is
    NaN). Signs are focal minus reference.

    Parameters
    ----------
    result : FitResult or DIFScreenResult
        A unidimensional multiple-group fit (for a :class:`DIFScreenResult`, its ``fit``).
    data : DataFrame or array-like, optional
        The responses of the fit; adds ``expected_score_standardized_difference``,
        ``rmsd_reference`` and ``rmsd_focal``.
    group : array-like, optional
        Group labels of ``data`` (required with ``data``).
    focal : str or iterable of str, optional
        Focal groups (default: every group but the first).
    items : iterable of str, optional
        Items (default: every item); the test row sums over them.
    density : {"focal", "reference"}, default "focal"
        Latent density that weights the differences.
    item_names : list of str, optional
        Column names when ``data`` is an array.

    Returns
    -------
    DataFrame
        One row per focal group and item, then a test row per focal group: ``group`` (the
        focal group), ``level`` (``"item"`` / ``"test"``), ``item`` (``"test"`` on the test
        row), ``signed_difference``, ``unsigned_difference`` and, with ``data``,
        ``expected_score_standardized_difference``, ``rmsd_reference``, ``rmsd_focal``.

    Raises
    ------
    PrecisionError
        If JAX is not configured for 64-bit floating point.
    SpecificationError
        For a single-group, multidimensional or weighted fit, unknown groups or items, the
        reference group as focal, or ``data`` without ``group``.
    """
    check_x64()
    res = result.fit if isinstance(result, DIFScreenResult) else result
    if not isinstance(res, FitResult):
        raise SpecificationError(
            f"dif_effect_sizes() needs a FitResult or DIFScreenResult, got {type(result).__name__}."
        )
    model = res._model
    if not model.is_multigroup:
        raise SpecificationError("dif_effect_sizes() needs a multiple-group model.")
    if model.is_multidimensional or model.is_reduced:
        raise SpecificationError("dif_effect_sizes() supports unidimensional models only.")
    _check_choice("density", density, ("focal", "reference"))
    groups = [g for g, _ in model.resolved_groups()]
    reference = groups[0]
    if focal is None:
        focal_groups = groups[1:]
    else:
        focal_groups = [focal] if isinstance(focal, str) else list(dict.fromkeys(focal))
        unknown = [g for g in focal_groups if g not in groups]
        if unknown:
            raise SpecificationError(f"Unknown focal groups {unknown}; groups: {groups}.")
        if reference in focal_groups:
            raise SpecificationError(
                f"{reference!r} is the reference group (the first group); it cannot be focal."
            )
        if not focal_groups:
            raise SpecificationError("focal= selected no groups.")
    chosen = _item_list(model, items, "items") or list(model.item_names)
    if not chosen:
        raise SpecificationError("items= selected no items.")
    columns = [model.item_names.index(i) for i in chosen]

    nodes = np.asarray(res._quadrature.nodes(), dtype=float).reshape(-1)
    n_groups = len(groups)
    placeholder = np.zeros((n_groups, len(model.items)), dtype=np.int64)
    prob, eta = _effect_problem(res, placeholder, np.arange(n_groups, dtype=np.int32))
    weights = np.exp(np.asarray(_latent_log_weights(prob, eta)))  # (G, Q)
    weights = weights / weights.sum(axis=1, keepdims=True)
    expected = np.stack([_expected_scores(res, nodes, g) for g in range(n_groups)])  # (G, Q, J)

    with_data = data is not None
    if with_data:
        res._require_unweighted("dif_effect_sizes()")
        if group is None:
            raise SpecificationError("dif_effect_sizes() with data= needs group=.")
        y = coerce_responses(data, model, item_names=item_names)
        codes, _ = coerce_groups(group, model, y.shape[0], require_all=False)
        rmsd = _rmsd(res, y, codes, expected)
        theta = res.score(data, group=group, item_names=item_names, method="EAP")["theta"]
        theta_np = np.asarray(theta, dtype=float)

    ref_index = 0
    out = []
    for name in focal_groups:
        f = groups.index(name)
        w = weights[f] if density == "focal" else weights[ref_index]
        diff = expected[f][:, columns] - expected[ref_index][:, columns]  # (Q, J_sel)
        test_diff = diff.sum(axis=1)
        if with_data:
            theta_f = theta_np[(codes == f) & np.isfinite(theta_np)]
            e_foc = _expected_scores(res, theta_f, f)[:, columns]
            e_ref = _expected_scores(res, theta_f, ref_index)[:, columns]
        for k, item in enumerate(chosen):
            row: dict[str, Any] = {
                "group": name,
                "level": "item",
                "item": item,
                "signed_difference": float(np.sum(w * diff[:, k])),
                "unsigned_difference": float(np.sum(w * np.abs(diff[:, k]))),
            }
            if with_data:
                row["expected_score_standardized_difference"] = _essd(e_foc[:, k], e_ref[:, k])
                row["rmsd_reference"] = float(rmsd[ref_index, columns[k]])
                row["rmsd_focal"] = float(rmsd[f, columns[k]])
            out.append(row)
        test_row: dict[str, Any] = {
            "group": name,
            "level": "test",
            "item": "test",
            "signed_difference": float(np.sum(w * test_diff)),
            "unsigned_difference": float(np.sum(w * np.abs(test_diff))),
        }
        if with_data:
            test_row["expected_score_standardized_difference"] = _essd(
                e_foc.sum(axis=1), e_ref.sum(axis=1)
            )
            test_row["rmsd_reference"] = float("nan")
            test_row["rmsd_focal"] = float("nan")
        out.append(test_row)
    return pd.DataFrame(out)
