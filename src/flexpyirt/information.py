r"""Information functions, expected scores, reliability and the summed-score EAP table.

Result / Python shell layer (DESIGN.md §18.2, §18.3). Every function takes a
:class:`~flexpyirt.FitResult` as its first argument. Information and expected scores serve
unidimensional and multidimensional models; reliability and the summed-score table are
unidimensional.
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
    "information_matrix",
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
        raise SpecificationError(f"{what} supports unidimensional models only.")


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


def _require_fit(res: FitResult, what: str) -> None:
    if not isinstance(res, FitResult):
        raise SpecificationError(f"{what} needs a FitResult, got {type(res).__name__}.")


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


def _coerce_theta_points(res: FitResult, theta: Any) -> tuple[np.ndarray, pd.Index]:
    """Return ``theta`` as a ``(T, D)`` float64 array and the index of the result.

    Unidimensional models take a number or 1-D array (index ``theta``); multidimensional models
    take a ``(T, D)`` array, a DataFrame whose columns are the factor names, or one point as a
    1-D array of length ``D`` (a ``MultiIndex`` named by the factors).
    """
    model = res._model
    if not model.is_multidimensional:
        values = _coerce_theta(theta)
        return values[:, None], pd.Index(values, name="theta")
    names = list(model.factor_names)
    n_dims = len(names)
    if isinstance(theta, pd.DataFrame):
        if sorted(map(str, theta.columns)) != sorted(names) or len(theta.columns) != n_dims:
            raise SpecificationError(
                f"A theta DataFrame must have exactly the factor columns {names}, got "
                f"{list(theta.columns)}."
            )
        theta = theta[names]
    try:
        values = np.asarray(theta, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SpecificationError("theta must be an array of numbers.") from exc
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != n_dims:
        raise SpecificationError(
            f"theta of a model with {n_dims} factors must have shape (T, {n_dims}) (or be one "
            f"point of length {n_dims}), got shape {np.shape(theta)}."
        )
    if not np.all(np.isfinite(values)):
        raise SpecificationError("theta must be finite.")
    index = pd.MultiIndex.from_arrays([values[:, m] for m in range(n_dims)], names=names)
    return values, index


def _coerce_direction(res: FitResult, direction: Any, what: str) -> np.ndarray | None:
    """Return the normalized direction (``None`` for a unidimensional model)."""
    model = res._model
    if not model.is_multidimensional:
        if direction is not None:
            raise SpecificationError(
                f"direction= is only used for multidimensional models; {what} of a "
                "unidimensional model takes no direction."
            )
        return None
    names = list(model.factor_names)
    if direction is None:
        raise SpecificationError(
            f"{what} of a multidimensional model needs direction= (a vector of length "
            f"{len(names)} over the factors {names}); use information_matrix() for the full "
            "matrix."
        )
    try:
        values = np.asarray(direction, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SpecificationError("direction must be a 1-D array of numbers.") from exc
    if values.shape != (len(names),):
        raise SpecificationError(
            f"direction must have length {len(names)} (one entry per factor {names}), got "
            f"shape {values.shape}."
        )
    norm = float(np.linalg.norm(values)) if np.all(np.isfinite(values)) else float("nan")
    if not np.isfinite(norm) or norm == 0.0:
        raise SpecificationError("direction must be finite and non-zero.")
    return values / norm


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


def _matrices(res: FitResult, theta: np.ndarray, group_index: int) -> tuple[np.ndarray, np.ndarray]:
    """``(T, J, D, D)`` item information matrices and ``(T, J)`` expected item scores."""
    prob, eta = _problem(res)
    information, expected = _item_curves(prob, eta, jnp.asarray(theta))
    return np.asarray(information[group_index]), np.asarray(expected[group_index])


def _curves(res: FitResult, theta: np.ndarray, group_index: int) -> tuple[np.ndarray, np.ndarray]:
    """``(T, J)`` item information and expected item scores of a unidimensional model."""
    information, expected = _matrices(res, theta[:, None], group_index)
    return information[:, :, 0, 0], expected


def _directional(matrices: np.ndarray, direction: np.ndarray | None) -> np.ndarray:
    r"""Return :math:`\mathbf u^\top\mathbf I\mathbf u` on the last two axes."""
    if direction is None:
        return matrices[..., 0, 0]
    return np.asarray(np.einsum("m,...mn,n->...", direction, matrices, direction))


def _composite_se(matrices: np.ndarray, direction: np.ndarray) -> np.ndarray:
    r"""Return :math:`\sqrt{\mathbf u^\top\mathbf I^{-1}\mathbf u}` per point.

    ``inf`` where the matrix is numerically singular: its smallest eigenvalue is at most
    ``D * eps`` times its largest (or the matrix is 0).
    """
    n_dims = matrices.shape[-1]
    eigvals = np.linalg.eigvalsh(matrices)
    regular = eigvals[:, 0] > n_dims * np.finfo(np.float64).eps * eigvals[:, -1]
    out = np.full(matrices.shape[0], np.inf)
    for t in np.flatnonzero(regular):
        out[t] = float(np.sqrt(direction @ np.linalg.solve(matrices[t], direction)))
    return out


def item_information(
    res: FitResult, theta: Any, *, direction: Any = None, group: str | None = None
) -> pd.DataFrame:
    r"""Return the item information functions of a fitted model.

    The Fisher information of item :math:`j` about :math:`\theta` is

    .. math::
        I_j(\theta)=\sum_{k=0}^{K_j-1}P_{jk}(\theta)
            \Big(\frac{\partial\log P_{jk}(\theta)}{\partial\theta}\Big)^2

    (Birnbaum, 1968; Samejima, 1969), computed for every item family by automatic
    differentiation of the category log-probabilities. It is what mirt's ``iteminfo()``
    returns. For a multidimensional model it is the directional information

    .. math::
        I_{j,\mathbf u}(\boldsymbol\theta)=\mathbf u^\top\mathbf I_j(\boldsymbol\theta)\mathbf u

    along the unit vector :math:`\mathbf u` = ``direction / ||direction||``, with the item
    information matrix :math:`\mathbf I_j` of :func:`information_matrix` (Reckase & McKinley,
    1991). mirt's ``iteminfo(..., degrees = )`` is ``direction = cos(degrees)`` when the angles
    satisfy :math:`\sum_m\cos^2\alpha_m=1` (mirt does not normalize the direction).

    Parameters
    ----------
    res : FitResult
        A fitted model (unidimensional, confirmatory multidimensional, bifactor or two-tier).
    theta : float or array-like
        Latent trait values: shape ``(T,)`` for a unidimensional model; for a model with
        ``D`` factors an array of shape ``(T, D)``, a DataFrame whose columns are the factor
        names, or one point of length ``D``.
    direction : array-like, shape (D,), optional
        Direction in the latent space (normalized to unit length; mirt ``degrees`` →
        ``cos(degrees)``). Required for a multidimensional model, not allowed for a
        unidimensional one.
    group : str, optional
        Group name; required for a multiple-group model.

    Returns
    -------
    DataFrame
        One row per point of ``theta`` and one column per item. The index is named ``theta``
        for a unidimensional model and is a ``MultiIndex`` with the factor names as level names
        for a multidimensional model.

    Raises
    ------
    PrecisionError
        If JAX is not configured for 64-bit floating point.
    SpecificationError
        For a missing or unknown ``group``, an invalid ``theta``, or a ``direction`` that is
        missing (multidimensional), given (unidimensional), of the wrong length or zero.
    """
    check_x64()
    _require_fit(res, "item_information()")
    group_index = _resolve_group(res, group, "item_information()")
    values, index = _coerce_theta_points(res, theta)
    direction_u = _coerce_direction(res, direction, "item_information()")
    matrices, _ = _matrices(res, values, group_index)
    return pd.DataFrame(
        _directional(matrices, direction_u), index=index, columns=list(res._model.item_names)
    )


def test_information(
    res: FitResult, theta: Any, *, direction: Any = None, group: str | None = None
) -> pd.DataFrame:
    r"""Return the test information function and the conditional standard error.

    .. math::
        I(\theta)=\sum_jI_j(\theta),\qquad \mathrm{SE}(\theta)=I(\theta)^{-1/2}

    with the item information :math:`I_j` of :func:`item_information` (Birnbaum, 1968). It is
    mirt's ``testinfo()``. For a multidimensional model, with the test information matrix
    :math:`\mathbf I=\sum_j\mathbf I_j` and the unit vector :math:`\mathbf u`,

    .. math::
        I_{\mathbf u}(\boldsymbol\theta)=\mathbf u^\top\mathbf I(\boldsymbol\theta)\mathbf u,
        \qquad
        \mathrm{SE}_{\mathbf u}(\boldsymbol\theta)
            =\sqrt{\mathbf u^\top\mathbf I(\boldsymbol\theta)^{-1}\mathbf u}

    (Reckase & McKinley, 1991; Reckase, 2009): the directional information and the asymptotic
    standard error of the ML estimate of the composite :math:`\mathbf u^\top\boldsymbol\theta`
    (not :math:`I_{\mathbf u}^{-1/2}`, which assumes the orthogonal coordinates known and is what
    mirt's ``plot(type = "SE")`` draws; it is ``1 / np.sqrt(information)``).

    Parameters
    ----------
    res : FitResult
        A fitted model (unidimensional, confirmatory multidimensional, bifactor or two-tier).
    theta : float or array-like
        Latent trait values, as in :func:`item_information`.
    direction : array-like, shape (D,), optional
        Direction in the latent space (normalized to unit length). Required for a
        multidimensional model, not allowed for a unidimensional one.
    group : str, optional
        Group name; required for a multiple-group model.

    Returns
    -------
    DataFrame
        One row per point of ``theta`` (indexed as in :func:`item_information`) with the
        columns ``information`` and ``se`` (infinite where the information is 0 or, for a
        multidimensional model, where the information matrix is numerically singular).

    Raises
    ------
    PrecisionError
        If JAX is not configured for 64-bit floating point.
    SpecificationError
        As :func:`item_information`.
    """
    check_x64()
    _require_fit(res, "test_information()")
    group_index = _resolve_group(res, group, "test_information()")
    values, index = _coerce_theta_points(res, theta)
    direction_u = _coerce_direction(res, direction, "test_information()")
    matrices, _ = _matrices(res, values, group_index)
    total_matrix = matrices.sum(axis=1)
    total = _directional(total_matrix, direction_u)
    if direction_u is None:
        with np.errstate(divide="ignore"):
            se = 1.0 / np.sqrt(total)
    else:
        se = _composite_se(total_matrix, direction_u)
    return pd.DataFrame({"information": total, "se": se}, index=index)


# pytest would otherwise collect this public function as a test when a test module imports it.
test_information.__test__ = False  # type: ignore[attr-defined]


def information_matrix(
    res: FitResult, theta: Any, *, item: str | None = None, group: str | None = None
) -> np.ndarray:
    r"""Return the test (or one item's) Fisher information matrix at each point of ``theta``.

    .. math::
        \mathbf I_j(\boldsymbol\theta)=\sum_{k=0}^{K_j-1}P_{jk}(\boldsymbol\theta)\,
            \nabla_{\boldsymbol\theta}\log P_{jk}(\boldsymbol\theta)\,
            \nabla_{\boldsymbol\theta}\log P_{jk}(\boldsymbol\theta)^\top,\qquad
        \mathbf I(\boldsymbol\theta)=\sum_j\mathbf I_j(\boldsymbol\theta)

    (Reckase, 2009, ch. 5), by automatic differentiation of the category log-probabilities.
    It is mirt's ``iteminfo(..., multidim_matrix = TRUE)``, summed over items for the test.

    Parameters
    ----------
    res : FitResult
        A fitted model (unidimensional, confirmatory multidimensional, bifactor or two-tier).
    theta : float or array-like
        Latent trait values, as in :func:`item_information`.
    item : str, optional
        Item name; by default the test information matrix (the sum over items).
    group : str, optional
        Group name; required for a multiple-group model.

    Returns
    -------
    ndarray, shape (T, D, D)
        One symmetric positive semi-definite matrix per point of ``theta``, rows and columns
        in the order of the model's factors (``D = 1`` for a unidimensional model).

    Raises
    ------
    PrecisionError
        If JAX is not configured for 64-bit floating point.
    SpecificationError
        For a missing or unknown ``group``, an invalid ``theta`` or an unknown ``item``.
    """
    check_x64()
    _require_fit(res, "information_matrix()")
    group_index = _resolve_group(res, group, "information_matrix()")
    values, _ = _coerce_theta_points(res, theta)
    names = list(res._model.item_names)
    if item is not None and item not in names:
        raise SpecificationError(f"Unknown item={item!r}; available items: {names}.")
    matrices, _ = _matrices(res, values, group_index)
    if item is None:
        return matrices.sum(axis=1)
    return matrices[:, names.index(item)].copy()


def expected_score(res: FitResult, theta: Any, *, group: str | None = None) -> pd.DataFrame:
    r"""Return the expected item scores and the test characteristic curve.

    .. math::
        E_j(\theta)=\sum_{k=0}^{K_j-1}k\,P_{jk}(\theta),\qquad T(\theta)=\sum_jE_j(\theta)

    with categories scored :math:`0,\dots,K_j-1` as in the data, and :math:`\theta` a vector
    for a multidimensional model. They are mirt's ``expected.item()`` and ``expected.test()``.

    Parameters
    ----------
    res : FitResult
        A fitted model (unidimensional, confirmatory multidimensional, bifactor or two-tier).
    theta : float or array-like
        Latent trait values, as in :func:`item_information`.
    group : str, optional
        Group name; required for a multiple-group model.

    Returns
    -------
    DataFrame
        One row per point of ``theta`` (indexed as in :func:`item_information`), one column
        per item with its expected score, and the column ``total`` with the expected test
        score.

    Raises
    ------
    PrecisionError
        If JAX is not configured for 64-bit floating point.
    SpecificationError
        For a missing or unknown ``group``, an invalid ``theta``, or an item named ``total``.
    """
    check_x64()
    _require_fit(res, "expected_score()")
    names = list(res._model.item_names)
    if "total" in names:
        raise SpecificationError(
            "expected_score() uses the column name 'total' for the test score; rename the "
            "item called 'total'."
        )
    group_index = _resolve_group(res, group, "expected_score()")
    values, index = _coerce_theta_points(res, theta)
    _, expected = _matrices(res, values, group_index)
    out = pd.DataFrame(expected, index=index, columns=names)
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
        of their EAP scores is added. Each row counts once: the sampling
        weights of a weighted fit are not applied.
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
