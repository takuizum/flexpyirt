r"""Information functions, expected scores, reliability and the summed-score EAP table.

Result / Python shell layer (DESIGN.md §18.2) for unidimensional models. Every function takes
a :class:`~flexpyirt.FitResult` as its first argument and returns a DataFrame.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from flexpyirt import _core, _information
from flexpyirt._compile import coerce_groups, compile_problem
from flexpyirt._precision import check_x64
from flexpyirt.errors import SpecificationError
from flexpyirt.result import FitResult

__all__ = [
    "expected_score",
    "item_information",
    "reliability",
    "summed_score_table",
    "test_information",
]

_MARGINAL_N_POINTS = 801
"""Points of the dedicated grid of the marginal-reliability integral."""
_MARGINAL_SPAN_SD = 8.0
"""Half-width of that grid in latent standard deviations."""

_item_curves = jax.jit(_information.item_curves)
_summed_score_eap = jax.jit(_information.summed_score_eap)


def _require_unidimensional(res: FitResult, what: str) -> None:
    if not isinstance(res, FitResult):
        raise SpecificationError(f"{what} needs a FitResult, got {type(res).__name__}.")
    if res._model.is_multidimensional:
        raise SpecificationError(
            f"{what} supports unidimensional models only; information for multidimensional "
            "models is not available yet."
        )


def _require_normal(res: FitResult, what: str) -> None:
    model = res._model
    if model.is_empirical_histogram or model.is_davidian or model.is_mixture:
        raise SpecificationError(
            f"{what} supports Normal latent distributions only (not EmpiricalHistogram, "
            "DavidianCurve or NormalMixture)."
        )


def _group_names(res: FitResult) -> list[str]:
    return [g for g, _ in res._model.resolved_groups()]


def _resolve_group(res: FitResult, group: str | None, what: str) -> int:
    """Return the index of the group called ``group`` (0 for a single-group model)."""
    names = _group_names(res)
    if res._model.is_multigroup:
        if group is None:
            raise SpecificationError(
                f"{what} needs group= for a multiple-group model; choose one of {names}."
            )
        if not isinstance(group, str) or group not in names:
            raise SpecificationError(f"Unknown group={group!r}; available groups: {names}.")
        return names.index(group)
    if group is not None:
        raise SpecificationError("group= is only valid for a multiple-group model.")
    return 0


def _coerce_theta(theta: Any) -> np.ndarray:
    """Validate ``theta`` and return it as a non-empty finite 1-D float64 array."""
    try:
        values = np.atleast_1d(np.asarray(theta, dtype=np.float64))
    except (TypeError, ValueError) as exc:
        raise SpecificationError("theta must be a number or a 1-D array of numbers.") from exc
    if values.ndim != 1 or values.size == 0:
        raise SpecificationError(
            f"theta must be a number or a non-empty 1-D array, got shape {values.shape}."
        )
    if not np.all(np.isfinite(values)):
        raise SpecificationError("theta must be finite.")
    return values


def _problem(res: FitResult) -> tuple[_core.CompiledProblem, jax.Array]:
    """Compile ``res`` on one placeholder response pattern per group."""
    model = res._model
    n_groups = len(model.resolved_groups())
    y = np.zeros((n_groups, len(model.items)), dtype=np.int64)
    codes = np.arange(n_groups)
    compiled = compile_problem(
        model, y, codes, res.parameters, res._quadrature, eh_weights=res._eh_weights
    )
    return compiled.problem, jnp.asarray(res._eta)


def _curves(res: FitResult, theta: np.ndarray, group_index: int) -> tuple[np.ndarray, np.ndarray]:
    """``(T, J)`` item information and expected item scores of one group at ``theta``."""
    prob, eta = _problem(res)
    information, expected = _item_curves(prob, eta, jnp.asarray(theta))
    return np.asarray(information[group_index]), np.asarray(expected[group_index])


def item_information(res: FitResult, theta: Any, *, group: str | None = None) -> pd.DataFrame:
    r"""Return the item information functions of a fitted unidimensional model.

    The Fisher information of item :math:`j` about :math:`\theta` is

    .. math::
        I_j(\theta)=\sum_{k=0}^{K_j-1}P_{jk}(\theta)
            \Big(\frac{\partial\log P_{jk}(\theta)}{\partial\theta}\Big)^2

    (Birnbaum, 1968; Samejima, 1969), computed for every item family by automatic
    differentiation of the category log-probabilities. It is what mirt's ``iteminfo()``
    returns.

    Parameters
    ----------
    res : FitResult
        A fitted unidimensional model.
    theta : float or array-like, shape (T,)
        Latent trait values.
    group : str, optional
        Group name; required for a multiple-group model.

    Returns
    -------
    DataFrame
        One row per value of ``theta`` (the index, named ``theta``) and one column per item.

    Raises
    ------
    PrecisionError
        If JAX is not configured for 64-bit floating point.
    SpecificationError
        For a multidimensional model, a missing or unknown ``group``, or a ``theta`` that is
        not a finite number or 1-D array.
    """
    check_x64()
    _require_unidimensional(res, "item_information()")
    group_index = _resolve_group(res, group, "item_information()")
    values = _coerce_theta(theta)
    information, _ = _curves(res, values, group_index)
    return pd.DataFrame(
        information, index=pd.Index(values, name="theta"), columns=list(res._model.item_names)
    )


def test_information(res: FitResult, theta: Any, *, group: str | None = None) -> pd.DataFrame:
    r"""Return the test information function and the conditional standard error.

    .. math::
        I(\theta)=\sum_jI_j(\theta),\qquad \mathrm{SE}(\theta)=I(\theta)^{-1/2}

    with the item information :math:`I_j` of :func:`item_information` (Birnbaum, 1968). It is
    mirt's ``testinfo()``.

    Parameters
    ----------
    res : FitResult
        A fitted unidimensional model.
    theta : float or array-like, shape (T,)
        Latent trait values.
    group : str, optional
        Group name; required for a multiple-group model.

    Returns
    -------
    DataFrame
        One row per value of ``theta`` (the index, named ``theta``) with the columns
        ``information`` and ``se`` (infinite where the information is 0).

    Raises
    ------
    PrecisionError
        If JAX is not configured for 64-bit floating point.
    SpecificationError
        For a multidimensional model, a missing or unknown ``group``, or a ``theta`` that is
        not a finite number or 1-D array.
    """
    check_x64()
    _require_unidimensional(res, "test_information()")
    group_index = _resolve_group(res, group, "test_information()")
    values = _coerce_theta(theta)
    information, _ = _curves(res, values, group_index)
    total = information.sum(axis=1)
    with np.errstate(divide="ignore"):
        se = 1.0 / np.sqrt(total)
    return pd.DataFrame({"information": total, "se": se}, index=pd.Index(values, name="theta"))


# pytest would otherwise collect this public function as a test when a test module imports it.
test_information.__test__ = False  # type: ignore[attr-defined]


def expected_score(res: FitResult, theta: Any, *, group: str | None = None) -> pd.DataFrame:
    r"""Return the expected item scores and the test characteristic curve.

    .. math::
        E_j(\theta)=\sum_{k=0}^{K_j-1}k\,P_{jk}(\theta),\qquad T(\theta)=\sum_jE_j(\theta)

    with categories scored :math:`0,\dots,K_j-1` as in the data. They are mirt's
    ``expected.item()`` and ``expected.test()``.

    Parameters
    ----------
    res : FitResult
        A fitted unidimensional model.
    theta : float or array-like, shape (T,)
        Latent trait values.
    group : str, optional
        Group name; required for a multiple-group model.

    Returns
    -------
    DataFrame
        One row per value of ``theta`` (the index, named ``theta``), one column per item with
        its expected score, and the column ``total`` with the expected test score.

    Raises
    ------
    PrecisionError
        If JAX is not configured for 64-bit floating point.
    SpecificationError
        For a multidimensional model, a missing or unknown ``group``, a ``theta`` that is not
        a finite number or 1-D array, or an item named ``total``.
    """
    check_x64()
    _require_unidimensional(res, "expected_score()")
    names = list(res._model.item_names)
    if "total" in names:
        raise SpecificationError(
            "expected_score() uses the column name 'total' for the test score; rename the "
            "item called 'total'."
        )
    group_index = _resolve_group(res, group, "expected_score()")
    values = _coerce_theta(theta)
    _, expected = _curves(res, values, group_index)
    out = pd.DataFrame(expected, index=pd.Index(values, name="theta"), columns=names)
    out["total"] = expected.sum(axis=1)
    return out


def _marginal_reliability(res: FitResult) -> list[float]:
    r"""Per group, :math:`\int\sigma^2I/(\sigma^2I+1)\,\phi(\theta;\mu,\sigma^2)\,d\theta`."""
    means = res.latent["mean"].to_numpy(dtype=np.float64)
    variances = res.latent["var"].to_numpy(dtype=np.float64)
    z = np.linspace(-_MARGINAL_SPAN_SD, _MARGINAL_SPAN_SD, _MARGINAL_N_POINTS)
    weights = np.exp(-0.5 * z * z)
    weights /= weights.sum()
    out = []
    for g in range(len(_group_names(res))):
        mean, var = float(means[g]), float(variances[g])
        information, _ = _curves(res, mean + np.sqrt(var) * z, g)
        scaled = var * information.sum(axis=1)
        out.append(float(np.sum(weights * scaled / (scaled + 1.0))))
    return out


def reliability(
    res: FitResult,
    data: Any = None,
    *,
    group: Any = None,
    item_names: list[str] | None = None,
) -> pd.DataFrame:
    r"""Return the marginal and, with ``data``, the empirical reliability of EAP scores.

    The marginal reliability of a group with latent distribution :math:`N(\mu,\sigma^2)` and
    test information :math:`I(\theta)` is the expected conditional reliability

    .. math::
        \bar\rho=\int\frac{\sigma^2I(\theta)}{\sigma^2I(\theta)+1}\,
            \phi(\theta;\mu,\sigma^2)\,d\theta

    (Green, Bock, Humphreys, Linn & Reckase, 1984; Thissen & Wainer, 2001). It equals mirt's
    ``marginal_rxx()`` for :math:`\mu=0,\ \sigma^2=1` and, unlike it, is invariant under a
    linear change of the latent scale. The integral is evaluated on 801 equispaced points on
    :math:`\mu\pm8\sigma`, independently of the quadrature of the fit.

    The empirical reliability uses the EAP scores :math:`\hat\theta_i` and posterior standard
    deviations :math:`\mathrm{SE}_i` of the respondents in ``data``:

    .. math::
        \hat\rho=\frac{s^2_{\hat\theta}}{s^2_{\hat\theta}+\overline{\mathrm{SE}^2}}

    with the sample variance :math:`s^2_{\hat\theta}` (denominator :math:`N-1`) and the mean
    posterior variance :math:`\overline{\mathrm{SE}^2}`, as mirt's ``empirical_rxx()``. In a
    multiple-group model both are computed within each group.

    Parameters
    ----------
    res : FitResult
        A fitted unidimensional model with ``Normal`` latent distributions.
    data : DataFrame or array-like, optional
        Responses on the fitted items (NaN is missing). When given, the empirical reliability
        of their EAP scores is added.
    group : array-like, optional
        Group label of each row of ``data`` (required with ``data`` for a multiple-group
        model).
    item_names : list of str, optional
        Column names when ``data`` is an array.

    Returns
    -------
    DataFrame
        One row per group (with a leading ``group`` column in a multiple-group model): the
        column ``marginal`` and, with ``data``, ``empirical`` (NaN for a group with fewer than
        two respondents in ``data``).

    Raises
    ------
    PrecisionError
        If JAX is not configured for 64-bit floating point.
    SpecificationError
        For a multidimensional model, a latent distribution other than ``Normal``, or
        ``group`` / ``item_names`` without ``data``.
    """
    check_x64()
    _require_unidimensional(res, "reliability()")
    _require_normal(res, "reliability()")
    if data is None and (group is not None or item_names is not None):
        raise SpecificationError("group= and item_names= are only used together with data.")
    names = _group_names(res)
    columns: dict[str, Any] = {}
    if res._model.is_multigroup:
        columns["group"] = names
    columns["marginal"] = _marginal_reliability(res)
    if data is not None:
        scores = res.score(data, method="EAP", group=group, item_names=item_names)
        codes, _ = coerce_groups(group, res._model, len(scores), require_all=False)
        theta = scores["theta"].to_numpy()
        se = scores["se"].to_numpy()
        empirical = []
        for g in range(len(names)):
            in_group = codes == g
            if int(in_group.sum()) < 2:
                empirical.append(float("nan"))
                continue
            score_var = float(np.var(theta[in_group], ddof=1))
            error_var = float(np.mean(se[in_group] ** 2))
            total = score_var + error_var
            empirical.append(score_var / total if total > 0.0 else float("nan"))
        columns["empirical"] = empirical
    return pd.DataFrame(columns)


def summed_score_table(res: FitResult) -> pd.DataFrame:
    r"""Return the summed-score EAP table and the model-implied summed-score distribution.

    For each summed score :math:`s=0,\dots,\sum_j(K_j-1)`, with the fitted quadrature nodes
    :math:`\theta_q`, the fitted latent weights :math:`w_q` and the Lord–Wingersky likelihood
    :math:`L(s\mid\theta_q)` (:func:`summed_score_distribution`),

    .. math::
        P(s)=\sum_qL(s\mid\theta_q)w_q,\qquad
        \hat\theta(s)=\frac{\sum_q\theta_qL(s\mid\theta_q)w_q}{P(s)},\qquad
        \mathrm{SD}(s)^2=\frac{\sum_q(\theta_q-\hat\theta(s))^2L(s\mid\theta_q)w_q}{P(s)}

    (Thissen, Pommerich, Billeaud & Williams, 1995). It is the table of mirt's
    ``fscores(method = "EAPsum", full.scores = FALSE)``, whose ``expected`` column is
    :math:`N\,P(s)`.

    Parameters
    ----------
    res : FitResult
        A fitted unidimensional model with ``Normal`` latent distributions.

    Returns
    -------
    DataFrame
        One row per summed score (per group and score, with a leading ``group`` column, in a
        multiple-group model): ``score``, ``theta`` (the EAP), ``se`` (the posterior standard
        deviation) and ``probability`` (:math:`P(s)`).

    Raises
    ------
    PrecisionError
        If JAX is not configured for 64-bit floating point.
    SpecificationError
        For a multidimensional model or a latent distribution other than ``Normal``.
    """
    check_x64()
    _require_unidimensional(res, "summed_score_table()")
    _require_normal(res, "summed_score_table()")
    prob, eta = _problem(res)
    max_score = sum(m.kernel.n_categories - 1 for _, m in res._model.items)
    p_score, theta, se = (np.asarray(a)[:, : max_score + 1] for a in _summed_score_eap(prob, eta))
    frames = []
    for g, name in enumerate(_group_names(res)):
        frame = pd.DataFrame(
            {
                "score": np.arange(max_score + 1),
                "theta": theta[g],
                "se": se[g],
                "probability": p_score[g],
            }
        )
        if res._model.is_multigroup:
            frame.insert(0, "group", name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
