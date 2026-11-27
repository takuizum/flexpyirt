r"""The numbers behind the plots: trace lines, empirical item curves and Wright-map tables.

Result / Python shell layer (DESIGN.md §21) for unidimensional models. Every function takes a
:class:`~flexpyirt.FitResult` first and returns DataFrames; :mod:`flexpyirt.plotting` draws
them, so what a figure shows is exactly what these functions return.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from flexpyirt import _information
from flexpyirt._compile import coerce_groups, coerce_responses
from flexpyirt._precision import check_x64
from flexpyirt.errors import SpecificationError
from flexpyirt.information import (
    _coerce_theta,
    _group_names,
    _problem,
    _require_unidimensional,
    _resolve_group,
)
from flexpyirt.result import FitResult

__all__ = [
    "WrightMapTable",
    "empirical_item_curve",
    "item_probabilities",
    "wright_map_table",
]

_THRESHOLD_BOUND = 20.0
"""Thurstonian thresholds are searched on :math:`[-20, 20]`."""
_THRESHOLD_GRID = 4001
"""Points of the bracketing grid of the threshold search (spacing 0.01)."""
_BISECTION_STEPS = 64
"""Bisection steps after bracketing (the interval shrinks below one ulp long before)."""

_category_probabilities = jax.jit(_information.category_probabilities)

PersonMethod = Literal["EAP", "MAP", "WLE"]
_PERSON_METHODS: tuple[str, ...] = ("EAP", "MAP", "WLE")


class WrightMapTable(NamedTuple):
    r"""The two tables of an item–person (Wright) map, returned by :func:`wright_map_table`.

    Attributes
    ----------
    persons : DataFrame
        One row per respondent: ``theta`` (and a leading ``group`` column in a multiple-group
        model); the index follows ``data`` when it is a DataFrame.
    items : DataFrame
        One row per item threshold: ``item``, ``category`` (:math:`k=1,\\dots,K_j-1`) and
        ``location`` (and a leading ``group`` column in a multiple-group model).
    """

    persons: pd.DataFrame
    items: pd.DataFrame


def _select_items(res: FitResult, items: str | Sequence[str] | None) -> list[str]:
    names = list(res._model.item_names)
    if items is None:
        return names
    chosen = [items] if isinstance(items, str) else list(items)
    if not chosen:
        raise SpecificationError("items= must name at least one item.")
    unknown = [i for i in chosen if i not in names]
    if unknown:
        raise SpecificationError(f"Unknown items {unknown}; the model has {names}.")
    if len(set(chosen)) != len(chosen):
        raise SpecificationError(f"items= has duplicates: {chosen}.")
    return chosen


def _probabilities(res: FitResult, theta: np.ndarray) -> np.ndarray:
    """``(G, T, J, K_max)`` category probabilities at ``theta`` (0 above an item's own K)."""
    prob, eta = _problem(res)
    return np.asarray(_category_probabilities(prob, eta, jnp.asarray(theta)))


def _n_categories(res: FitResult) -> list[int]:
    return [int(m.kernel.n_categories) for _, m in res._model.items]


def item_probabilities(
    res: FitResult,
    theta: Any,
    *,
    items: str | Sequence[str] | None = None,
    group: str | None = None,
) -> pd.DataFrame:
    r"""Return the category response functions (trace lines) of a fitted unidimensional model.

    .. math::
        P_{jk}(\theta)=\Pr(Y_j=k\mid\theta),\qquad k=0,\dots,K_j-1

    for every item family (Lord, 1980). It is mirt's ``probtrace()``.

    Parameters
    ----------
    res : FitResult
        A fitted unidimensional model.
    theta : float or array-like, shape (T,)
        Latent trait values.
    items : str or sequence of str, optional
        Items to include (default: all, in the model's order).
    group : str, optional
        Group name; required for a multiple-group model.

    Returns
    -------
    DataFrame
        Long format, one row per item, category and value of ``theta`` (in that order):
        ``item``, ``category``, ``theta``, ``probability``.

    Raises
    ------
    PrecisionError
        If JAX is not configured for 64-bit floating point.
    SpecificationError
        For a multidimensional model, an unknown item, a missing or unknown ``group``, or a
        ``theta`` that is not a finite number or 1-D array.
    """
    check_x64()
    _require_unidimensional(res, "item_probabilities()")
    chosen = _select_items(res, items)
    group_index = _resolve_group(res, group, "item_probabilities()")
    values = _coerce_theta(theta)
    probs = _probabilities(res, values)[group_index]  # (T, J, K_max)
    names = list(res._model.item_names)
    n_categories = _n_categories(res)
    frames = []
    for item in chosen:
        j = names.index(item)
        k_j = n_categories[j]
        frames.append(
            pd.DataFrame(
                {
                    "item": item,
                    "category": np.repeat(np.arange(k_j), values.size),
                    "theta": np.tile(values, k_j),
                    "probability": probs[:, j, :k_j].T.ravel(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _bin_sizes(n: int, n_bins: int) -> np.ndarray:
    r"""Sizes of ``n_bins`` consecutive bins of ``n`` sorted respondents (mirt's rule).

    Every bin gets :math:`\\lfloor n/B\\rfloor` respondents; of the remainder
    :math:`c=n\\bmod B`, :math:`\\lfloor c/2\\rfloor` go to the first bin and the rest to the last.
    """
    sizes = np.full(n_bins, n // n_bins, dtype=np.int64)
    rest = n % n_bins
    sizes[0] += rest // 2
    sizes[-1] += rest - rest // 2
    return sizes


def empirical_item_curve(
    res: FitResult,
    data: Any,
    *,
    items: str | Sequence[str] | None = None,
    n_bins: int = 10,
    group: Any = None,
    item_names: list[str] | None = None,
) -> pd.DataFrame:
    r"""Return observed category proportions in bins of EAP :math:`\theta` beside the model.

    For item :math:`j`, the :math:`n_j` respondents who answered it are sorted by their EAP
    score :math:`\hat\theta_i` (ties in row order) and cut into :math:`B` consecutive bins of
    :math:`\lfloor n_j/B\rfloor` respondents; of the remainder :math:`c=n_j\bmod B`,
    :math:`\lfloor c/2\rfloor` go to the first bin and the rest to the last, as in mirt's
    ``itemfit()``. For bin :math:`b` with respondents :math:`\mathcal B_b`,

    .. math::
        \bar\theta_b=\frac1{n_b}\sum_{i\in\mathcal B_b}\hat\theta_i,\qquad
        O_{bk}=\frac1{n_b}\sum_{i\in\mathcal B_b}[y_{ij}=k],\qquad
        E_{bk}=P_{jk}(\bar\theta_b)

    (Bock, 1972; Yen, 1981). :math:`n_bO_{bk}` and :math:`n_bE_{bk}` are the observed and
    expected tables of mirt's ``itemfit(fit_stats = "X2", return.tables = TRUE)`` without its
    cell collapsing, and :math:`(\bar\theta_b, O_{bk})` are the points of
    ``itemfit(empirical.plot = )``.

    Parameters
    ----------
    res : FitResult
        A fitted unidimensional model.
    data : DataFrame or array-like
        Responses on the fitted items (NaN is missing; a respondent enters the bins of the items
        they answered). Each row counts once: the sampling
        weights of a weighted fit are not applied.
    items : str or sequence of str, optional
        Items to include (default: all, in the model's order).
    n_bins : int, default 10
        Number of bins :math:`B` (mirt's ``group.bins``).
    group : array-like, optional
        Group label of each row of ``data`` (required for a multiple-group model). Bins are
        formed within each group, with that group's item parameters.
    item_names : list of str, optional
        Column names when ``data`` is an array.

    Returns
    -------
    DataFrame
        One row per (group,) item, bin and category: ``item``, ``bin`` (1, ..., B), ``theta``
        (:math:`\bar\theta_b`), ``n`` (:math:`n_b`), ``category``, ``observed``
        (:math:`O_{bk}`) and ``expected`` (:math:`E_{bk}`), with a leading ``group`` column in a
        multiple-group model.

    Raises
    ------
    PrecisionError
        If JAX is not configured for 64-bit floating point.
    SpecificationError
        For a multidimensional model, an unknown item, ``n_bins`` that is not a positive
        integer, or an item (in a group) answered by fewer than ``n_bins`` respondents.
    """
    check_x64()
    _require_unidimensional(res, "empirical_item_curve()")
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise SpecificationError(f"n_bins must be a positive integer, got {n_bins!r}.")
    chosen = _select_items(res, items)
    model = res._model
    y = coerce_responses(data, model, item_names=item_names)
    codes, group_names = coerce_groups(group, model, y.shape[0], require_all=False)
    eap = res.score(data, method="EAP", group=group, item_names=item_names)["theta"].to_numpy()
    names = list(model.item_names)
    n_categories = _n_categories(res)
    frames = []
    for g, g_name in enumerate(group_names):
        in_group = codes == g
        if not in_group.any():
            continue
        y_g, eap_g = y[in_group], eap[in_group]
        bins: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for item in chosen:
            j = names.index(item)
            answered = y_g[:, j] >= 0
            n_j = int(answered.sum())
            if n_j < n_bins:
                where = f" in group {g_name!r}" if model.is_multigroup else ""
                raise SpecificationError(
                    f"Item {item!r} has {n_j} responses{where}, fewer than n_bins={n_bins}."
                )
            order = np.argsort(eap_g[answered], kind="stable")
            theta_sorted = eap_g[answered][order]
            y_sorted = y_g[answered, j][order]
            label = np.repeat(np.arange(n_bins), _bin_sizes(n_j, n_bins))
            counts = np.bincount(label, minlength=n_bins)
            means = np.bincount(label, weights=theta_sorted, minlength=n_bins) / counts
            observed = np.zeros((n_bins, n_categories[j]))
            np.add.at(observed, (label, y_sorted), 1.0)
            bins[item] = (means, counts, observed / counts[:, None])
        all_means = np.concatenate([b[0] for b in bins.values()])
        # One call for every bin of every chosen item in this group.
        probs = _probabilities(res, all_means)[g]  # (len(chosen) * B, J, K_max)
        for pos, (item, (means, counts, observed)) in enumerate(bins.items()):
            j = names.index(item)
            k_j = n_categories[j]
            expected = probs[pos * n_bins : (pos + 1) * n_bins, j, :k_j]
            frame = pd.DataFrame(
                {
                    "item": item,
                    "bin": np.repeat(np.arange(1, n_bins + 1), k_j),
                    "theta": np.repeat(means, k_j),
                    "n": np.repeat(counts, k_j),
                    "category": np.tile(np.arange(k_j), n_bins),
                    "observed": observed.ravel(),
                    "expected": expected.ravel(),
                }
            )
            if model.is_multigroup:
                frame.insert(0, "group", g_name)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _thurstonian_thresholds(res: FitResult, group_index: int) -> list[np.ndarray]:
    r"""Per item, the roots of :math:`P(Y_j\ge k\mid\theta)=1/2`, :math:`k=1,\dots,K_j-1`.

    The first sign change on an equispaced grid of :math:`[-20, 20]` is refined by bisection;
    NaN where :math:`P(Y_j\ge k\mid\theta)-1/2` does not change sign on the interval.
    """
    n_categories = _n_categories(res)
    grid = np.linspace(-_THRESHOLD_BOUND, _THRESHOLD_BOUND, _THRESHOLD_GRID)
    upper = np.flip(np.cumsum(np.flip(_probabilities(res, grid)[group_index], -1), -1), -1)
    targets: list[tuple[int, int]] = []  # (item, k) with a bracket
    lo_list: list[float] = []
    hi_list: list[float] = []
    out = [np.full(k - 1, np.nan) for k in n_categories]
    for j, k_j in enumerate(n_categories):
        for k in range(1, k_j):
            f = upper[:, j, k] - 0.5
            change = np.nonzero(np.sign(f[:-1]) != np.sign(f[1:]))[0]
            exact = np.nonzero(f == 0.0)[0]
            if exact.size and (not change.size or exact[0] <= change[0]):
                out[j][k - 1] = grid[exact[0]]
                continue
            if not change.size:
                continue
            targets.append((j, k))
            lo_list.append(grid[change[0]])
            hi_list.append(grid[change[0] + 1])
    if targets:
        lo, hi = np.asarray(lo_list), np.asarray(hi_list)
        item_idx = np.asarray([t[0] for t in targets])
        cat_idx = np.asarray([t[1] for t in targets])
        rows = np.arange(len(targets))

        def excess(t: np.ndarray) -> np.ndarray:
            p = _probabilities(res, t)[group_index]  # (T, J, K_max)
            tail = np.flip(np.cumsum(np.flip(p, -1), -1), -1)
            return np.asarray(tail[rows, item_idx, cat_idx] - 0.5)

        f_lo = excess(lo)
        for _ in range(_BISECTION_STEPS):
            mid = 0.5 * (lo + hi)
            f_mid = excess(mid)
            same = np.sign(f_mid) == np.sign(f_lo)
            lo = np.where(same, mid, lo)
            f_lo = np.where(same, f_mid, f_lo)
            hi = np.where(same, hi, mid)
        for (j, k), value in zip(targets, 0.5 * (lo + hi), strict=True):
            out[j][k - 1] = value
    return out


def wright_map_table(
    res: FitResult,
    data: Any,
    *,
    group: Any = None,
    item_names: list[str] | None = None,
    method: PersonMethod = "EAP",
) -> WrightMapTable:
    r"""Return the person estimates and item locations of an item–person (Wright) map.

    Persons are placed at their scores :math:`\hat\theta_i` (``FitResult.score``). Item
    :math:`j` is placed at its Thurstonian thresholds, the :math:`K_j-1` values

    .. math::
        \tau_{jk}:\quad P(Y_j\ge k\mid\tau_{jk})=\sum_{c\ge k}P_{jc}(\tau_{jk})=\tfrac12,
        \qquad k=1,\dots,K_j-1

    (Wright & Masters, 1982; Wu, Adams, Wilson & Haldane, 2007, ``tam.threshold``). For 2PL /
    Rasch items :math:`\tau_{j1}=b_j=-d_j/a_j` and for GRM items :math:`\tau_{jk}=b_{jk}`; for
    3PL / 4PL items it is where the probability of a correct response is one half (not
    :math:`b`, where it is :math:`(g+u)/2`), and for GPCM / PCM items the thresholds differ from
    the step difficulties. :math:`P(Y_j\ge k\mid\theta)` increases with :math:`\theta` for every
    ordered family with :math:`a>0`; the first root on :math:`[-20, 20]` is returned (for nominal
    items, whose categories are not ordered, this is the first crossing of the cumulative
    probability in the data's category order), and NaN when there is none (e.g. a 3PL item with
    :math:`g\ge1/2`).

    Parameters
    ----------
    res : FitResult
        A fitted unidimensional model.
    data : DataFrame or array-like
        Responses on the fitted items (NaN is missing). Each row counts once: the sampling
        weights of a weighted fit are not applied.
    group : array-like, optional
        Group label of each row of ``data`` (required for a multiple-group model).
    item_names : list of str, optional
        Column names when ``data`` is an array.
    method : {"EAP", "MAP", "WLE"}, default "EAP"
        Scoring method of the persons (finite for every response pattern).

    Returns
    -------
    WrightMapTable
        ``persons`` (column ``theta``) and ``items`` (columns ``item``, ``category``,
        ``location``), each with a leading ``group`` column in a multiple-group model; in that
        case ``items`` has the thresholds of every group, computed from that group's
        parameters.

    Raises
    ------
    PrecisionError
        If JAX is not configured for 64-bit floating point.
    SpecificationError
        For a multidimensional model or an unsupported ``method``.
    """
    check_x64()
    _require_unidimensional(res, "wright_map_table()")
    if method not in _PERSON_METHODS:
        raise SpecificationError(
            f"wright_map_table() scores persons with one of {_PERSON_METHODS}, got {method!r}."
        )
    model = res._model
    scores = res.score(data, method=method, group=group, item_names=item_names)
    persons = pd.DataFrame({"theta": scores["theta"].to_numpy()}, index=scores.index)
    groups = _group_names(res)
    if model.is_multigroup:
        codes, _ = coerce_groups(group, model, len(scores), require_all=False)
        persons.insert(0, "group", np.asarray(groups, dtype=object)[codes])
    names = list(model.item_names)
    frames = []
    for g, g_name in enumerate(groups):
        thresholds = _thurstonian_thresholds(res, g)
        frame = pd.DataFrame(
            {
                "item": [name for name, t in zip(names, thresholds, strict=True) for _ in t],
                "category": np.concatenate([np.arange(1, t.size + 1) for t in thresholds]),
                "location": np.concatenate(thresholds),
            }
        )
        if model.is_multigroup:
            frame.insert(0, "group", g_name)
        frames.append(frame)
    return WrightMapTable(persons=persons, items=pd.concat(frames, ignore_index=True))
