r"""Simulation of response data from a fitted or a fully specified model (DESIGN.md §24.2).

Respondent :math:`i` of group :math:`g` gets :math:`\boldsymbol\theta_i` drawn from the group's
latent distribution (or given), and each response is drawn from the item's category
probabilities by inversion of one uniform number,

.. math::
    X_{ij}=\min\{k:\ u_{ij}\le F_{jk}(\boldsymbol\theta_i)\},\qquad
    F_{jk}=\sum_{c=0}^{k}P_{jc},\quad u_{ij}\sim U(0,1)

(Devroye, 1986, §III.2). The category probabilities are the fitted kernels' own (the same
``_core._item_log_probs`` the E-step uses), evaluated at the drawn points instead of the grid.
Random numbers come from an explicit JAX PRNG key, so a draw is reproducible from ``key``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from flexpyirt import _core
from flexpyirt._compile import LATENT_ITEM, build_table, compile_problem
from flexpyirt._precision import check_x64
from flexpyirt.errors import SpecificationError
from flexpyirt.latent import (
    davidian_hermite_coeffs,
    davidian_moments,
    davidian_param_names,
    mixture_components,
    mixture_param_names,
)
from flexpyirt.model import Model
from flexpyirt.quadrature import MAX_FACTORS, Quadrature
from flexpyirt.result import FitResult, _is_prng_key

__all__ = ["SimulatedData", "simulate_responses"]

_item_log_probs = jax.jit(_core._item_log_probs)

DAVIDIAN_GRID_HALF_WIDTH = 12.0
r"""Half-width of the :math:`z` grid on which the Davidian density is inverted."""
DAVIDIAN_GRID_POINTS = 24001
r"""Number of points of that grid (spacing :math:`10^{-3}`)."""


@dataclass(frozen=True)
class SimulatedData:
    r"""Responses simulated by :func:`simulate_responses`.

    Attributes
    ----------
    responses : DataFrame
        One row per simulated respondent and one ``int64`` column per item (the model's item
        names), coded ``0..K-1``; it can be passed to :func:`flexpyirt.fit` as it is.
    theta : ndarray
        The respondents' latent traits: shape ``(N,)`` for a unidimensional model and
        ``(N, D)`` otherwise (columns in the order of the model's factors).
    group : ndarray or None
        Group label of each row for a multiple-group model (pass it as ``fit(..., group=)``),
        ``None`` for a single-group model.
    """

    responses: pd.DataFrame
    theta: np.ndarray
    group: np.ndarray | None


@dataclass(frozen=True)
class _Source:
    """What the simulation needs from a fit or from a model with given parameters."""

    model: Model
    table: pd.DataFrame
    eta: np.ndarray
    quadrature: Quadrature
    eh_weights: np.ndarray | None


def _source_from_fit(res: FitResult) -> _Source:
    return _Source(
        model=res._model,
        table=res.parameters,
        eta=np.asarray(res._eta, dtype=np.float64),
        quadrature=res._quadrature,
        eh_weights=res._eh_weights,
    )


def _dummy_responses(model: Model) -> np.ndarray:
    """Responses with every category of every item, so ``build_table``'s start values exist."""
    k_max = max(m.kernel.n_categories for _, m in model.items)
    rows = [[min(k, m.kernel.n_categories - 1) for _, m in model.items] for k in range(k_max)]
    return np.asarray(rows * 2, dtype=np.int32)


def _source_from_model(spec: Model, parameters: pd.DataFrame) -> _Source:
    """Fix every item parameter from ``parameters`` (as ``Model.fix_from``) and check the latent."""
    if not isinstance(parameters, pd.DataFrame):
        raise SpecificationError(
            "parameters must be a DataFrame (an item bank as in Model.fix_from()), got "
            f"{type(parameters).__name__}."
        )
    for _, latent in spec.resolved_groups():
        name = type(latent).__name__
        if name in ("EmpiricalHistogram", "DavidianCurve"):
            raise SpecificationError(
                f"Simulation from a Model needs every latent parameter in the specification; "
                f"a {name} has its shape only after fitting. Simulate from a FitResult, or "
                "pass theta=."
            )
    model = spec.fix_from(parameters)
    table = build_table(model, _dummy_responses(model))
    free_items = table[(table["item"] != LATENT_ITEM) & table["free"]]
    if not free_items.empty:
        missing = sorted(
            {f"{i}.{p}" for i, p in zip(free_items["item"], free_items["param"], strict=True)}
        )
        raise SpecificationError(
            f"parameters gives no value for {missing}; every item parameter must be given."
        )
    table = table.copy()
    table["estimate"] = table["value"].to_numpy(dtype=float)
    n_free = int(table["free_id"].max()) + 1 if table["free"].any() else 0
    eta = np.zeros(n_free)
    free = table[table["free"]]
    eta[free["free_id"].to_numpy(dtype=int)] = free["value"].to_numpy(dtype=float)
    return _Source(
        model=model,
        table=table,
        eta=eta,
        quadrature=(
            Quadrature.default(model.quadrature_dimension)
            if model.quadrature_dimension <= MAX_FACTORS
            else Quadrature.qmc()
        ),
        eh_weights=None,
    )


def _category_probabilities(src: _Source, theta: np.ndarray) -> list[np.ndarray]:
    """Per item, ``(G, N, K_j)`` category probabilities of every group at ``theta`` ``(N, D)``."""
    model = src.model
    y = np.zeros((1, len(model.items)), dtype=np.int32)
    prob = compile_problem(model, y, np.zeros(1, dtype=np.int32), src.table, src.quadrature).problem
    # Without the reduction the kernels read ``nodes`` for bifactor / two-tier models too.
    prob = replace(prob, nodes=jnp.asarray(theta, dtype=jnp.float64), reduction=None)
    log_probs = _item_log_probs(prob, jnp.asarray(src.eta))
    out: list[np.ndarray] = [np.empty(0)] * len(model.items)
    for fam, lp in zip(prob.families, log_probs, strict=True):
        probs = np.exp(np.asarray(lp))
        for pos, col in enumerate(np.asarray(fam.columns)):
            out[int(col)] = probs[:, pos]
    return out


def responses_from_uniforms(probs: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    r"""Invert the category cumulative probabilities at uniform numbers.

    .. math::
        X_i=\min\{k:\ u_i\le F_k\}=\#\{k<K-1:\ u_i>F_k\},\qquad F_k=\sum_{c\le k}P_c

    (Devroye, 1986, §III.2). The last cumulative probability is never compared, so round-off
    in :math:`\sum_cP_c` cannot give a category beyond :math:`K-1`.

    Parameters
    ----------
    probs : ndarray, shape (N, K)
        Category probabilities of one item for each respondent.
    uniforms : ndarray, shape (N,)
        Uniform numbers in :math:`[0, 1)`.

    Returns
    -------
    ndarray of int64, shape (N,)
    """
    cum = np.cumsum(probs, axis=1)[:, :-1]
    return np.asarray((uniforms[:, None] > cum).sum(axis=1), dtype=np.int64)


def _latent_values(src: _Source, group: str) -> dict[str, float]:
    t = src.table
    rows = t[(t["item"] == LATENT_ITEM) & (t["group"] == group)]
    return dict(zip(rows["param"], rows["estimate"].to_numpy(dtype=float), strict=True))


def _normal_draws(values: Mapping[str, float], n_factors: int, key: Any, n: int) -> np.ndarray:
    r"""Draw :math:`\boldsymbol\theta=\boldsymbol\mu+\mathbf L\mathbf z` (multivariate normal).

    :math:`\mathbf{LL}^\top=\boldsymbol\Sigma` and :math:`\mathbf z\sim N(\mathbf 0,\mathbf I)`.
    """
    if n_factors == 1:
        mean = np.array([values["mean"]])
        cov = np.array([[values["var"]]])
    else:
        mean = np.array([values[f"mean_{m}"] for m in range(1, n_factors + 1)])
        sd = np.sqrt([values[f"var_{m}"] for m in range(1, n_factors + 1)])
        corr = np.eye(n_factors)
        for i in range(1, n_factors + 1):
            for j in range(i + 1, n_factors + 1):
                corr[i - 1, j - 1] = corr[j - 1, i - 1] = values.get(f"cor_{i}{j}", 0.0)
        cov = corr * np.outer(sd, sd)
    chol = np.linalg.cholesky(cov)
    z = np.asarray(jax.random.normal(key, (n, n_factors), dtype=jnp.float64))
    return np.asarray(mean[None, :] + z @ chol.T)


def _mixture_draws(values: Mapping[str, float], n_components: int, key: Any, n: int) -> np.ndarray:
    r"""Component :math:`c\sim\mathrm{Cat}(\boldsymbol\pi)`, then :math:`N(\mu_c,\sigma_c^2)`."""
    names = mixture_param_names(n_components)
    pi, means, variances = (
        np.asarray(x, dtype=float)
        for x in mixture_components(jnp.asarray([values[p] for p in names], dtype=jnp.float64))
    )
    k_comp, k_norm = jax.random.split(key)
    comp = np.asarray(
        jax.random.choice(k_comp, n_components, shape=(n,), p=jnp.asarray(pi / pi.sum()))
    )
    z = np.asarray(jax.random.normal(k_norm, (n,), dtype=jnp.float64))
    out: np.ndarray = (means[comp] + np.sqrt(variances[comp]) * z)[:, None]
    return out


def _davidian_draws(values: Mapping[str, float], degree: int, key: Any, n: int) -> np.ndarray:
    r"""Invert the CDF of :math:`h(z)=P_k(z)^2\varphi(z)` on a fine grid, then standardize.

    :math:`\theta=\mu+\sigma(z-\mu_Z)/\sigma_Z` (DESIGN.md §16.4). The CDF is the trapezoidal
    integral of :math:`h` on ``DAVIDIAN_GRID_POINTS`` points of
    :math:`[-12, 12]`, inverted by linear interpolation.
    """
    names = davidian_param_names(degree)
    params = np.array([values[p] for p in names])
    coeffs = davidian_hermite_coeffs(jnp.asarray(params[2:], dtype=jnp.float64))
    mu_z, var_z = (float(x) for x in davidian_moments(coeffs))
    c = np.asarray(coeffs, dtype=float)
    z = np.linspace(-DAVIDIAN_GRID_HALF_WIDTH, DAVIDIAN_GRID_HALF_WIDTH, DAVIDIAN_GRID_POINTS)
    basis = [np.ones_like(z), z]
    for j in range(2, degree + 1):
        basis.append((z * basis[-1] - math.sqrt(j - 1) * basis[-2]) / math.sqrt(j))
    poly = np.stack(basis[: degree + 1], axis=1) @ c
    density = poly**2 * np.exp(-0.5 * z**2)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(z))])
    cdf /= cdf[-1]
    u = np.asarray(jax.random.uniform(key, (n,), dtype=jnp.float64))
    z_draw = np.interp(u, cdf, z)
    theta: np.ndarray = params[0] + math.sqrt(params[1]) * (z_draw - mu_z) / math.sqrt(var_z)
    return theta[:, None]


def _histogram_draws(nodes: np.ndarray, weights: np.ndarray, key: Any, n: int) -> np.ndarray:
    """Draw grid nodes with the fitted empirical-histogram weights (a discrete distribution)."""
    w = np.asarray(weights, dtype=float)
    idx = np.asarray(jax.random.choice(key, w.shape[0], shape=(n,), p=jnp.asarray(w / w.sum())))
    out: np.ndarray = np.asarray(nodes, dtype=float)[idx][:, None]
    return out


def _draw_theta(src: _Source, g_index: int, group: str, key: Any, n: int) -> np.ndarray:
    """``(n, D)`` draws from the latent distribution of group ``g_index``."""
    model = src.model
    if model.is_empirical_histogram:
        assert src.eh_weights is not None
        return _histogram_draws(src.quadrature.nodes(), src.eh_weights[g_index], key, n)
    values = _latent_values(src, group)
    if model.is_davidian:
        return _davidian_draws(values, model.davidian_degree, key, n)
    if model.is_mixture:
        return _mixture_draws(values, model.mixture_components, key, n)
    return _normal_draws(values, model.n_factors, key, n)


def _positive_int(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise SpecificationError(f"{what} must be a positive integer, got {value!r}.")
    return int(value)


def _group_sizes(model: Model, n_obs: Any) -> list[int]:
    """Rows per group in the model's group order."""
    groups = [g for g, _ in model.resolved_groups()]
    if isinstance(n_obs, Mapping):
        if not model.is_multigroup:
            raise SpecificationError(
                "n_obs may be a mapping of group sizes only for a multiple-group model."
            )
        unknown = sorted(str(g) for g in n_obs if str(g) not in groups)
        if unknown:
            raise SpecificationError(f"n_obs names groups not in the model: {unknown}.")
        sizes = {str(g): n for g, n in n_obs.items()}
        out = []
        for g in groups:
            n = sizes.get(g, 0)
            if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) < 0:
                raise SpecificationError(f"n_obs[{g!r}] must be a non-negative integer, got {n!r}.")
            out.append(int(n))
        if sum(out) == 0:
            raise SpecificationError("n_obs gives no respondent in any group.")
        return out
    n = _positive_int(n_obs, "n_obs")
    return [n] * len(groups)


def _given_theta(model: Model, theta: Any, group: Any, n_obs: Any) -> tuple[np.ndarray, np.ndarray]:
    """Validate given abilities (and group labels); return ``(N, D)`` and group codes."""
    arr = np.asarray(theta, dtype=float)
    d = model.n_factors
    if arr.ndim == 1 and d == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[1] != d or arr.shape[0] == 0:
        expected = "(N,) or (N, 1)" if d == 1 else f"(N, {d})"
        raise SpecificationError(f"theta must have shape {expected}, got {np.shape(theta)}.")
    if not np.all(np.isfinite(arr)):
        raise SpecificationError("theta must be finite.")
    n = arr.shape[0]
    if n_obs is not None and (isinstance(n_obs, Mapping) or _positive_int(n_obs, "n_obs") != n):
        raise SpecificationError(
            f"n_obs must be omitted (or equal {n}, the number of rows of theta) when theta is "
            "given."
        )
    groups = [g for g, _ in model.resolved_groups()]
    if not model.is_multigroup:
        if group is not None:
            raise SpecificationError(
                "group= was given but the model has no groups; declare them with with_groups()."
            )
        return arr, np.zeros(n, dtype=np.int32)
    if group is None:
        raise SpecificationError(
            f"The model has groups {groups}; pass group= labels, one per row of theta."
        )
    labels = np.asarray(group.to_numpy() if hasattr(group, "to_numpy") else group).astype(str)
    if labels.shape != (n,):
        raise SpecificationError(f"group= must have one label per row ({n}), got {labels.shape}.")
    unknown = sorted(set(labels.tolist()) - set(groups))
    if unknown:
        raise SpecificationError(f"group= has labels not declared in with_groups(): {unknown}.")
    lookup = {g: i for i, g in enumerate(groups)}
    return arr, np.array([lookup[v] for v in labels], dtype=np.int32)


def simulate_responses(
    source: FitResult | Model,
    *,
    n_obs: int | Mapping[str, int] | None = None,
    key: jax.Array,
    parameters: pd.DataFrame | None = None,
    theta: Any = None,
    group: Any = None,
) -> SimulatedData:
    r"""Simulate item responses from a fitted model or from a model with given parameters.

    Each respondent's :math:`\boldsymbol\theta_i` is drawn from the latent distribution of
    their group (or taken from ``theta``), and each response from the item's category
    probabilities :math:`P_{jk}(\boldsymbol\theta_i)` by inversion of a uniform number:

    .. math::
        X_{ij}=\min\{k:\ u_{ij}\le F_{jk}(\boldsymbol\theta_i)\},\qquad
        F_{jk}=\sum_{c=0}^{k}P_{jc}(\boldsymbol\theta_i),\quad u_{ij}\sim U(0,1)

    (Devroye, 1986, §III.2). The latent draws follow the fitted (or specified) distribution:

    - ``Normal``: :math:`N(\boldsymbol\mu,\boldsymbol\Sigma)` with the fitted means, variances
      and correlations (bifactor factors are orthogonal; two-tier models correlate the
      primary factors only);
    - ``NormalMixture``: a component :math:`c` with probability :math:`\pi_c`, then
      :math:`N(\mu_c,\sigma_c^2)` (the standardized components, DESIGN.md §16.7);
    - ``DavidianCurve``: the continuous density :math:`f_k(\theta)` (DESIGN.md §16.4), by
      inversion of its cumulative distribution function computed on a fine grid;
    - ``EmpiricalHistogram``: the quadrature nodes with the fitted weights. The fitted
      distribution is this discrete distribution, so the draws take the grid's values only.

    Parameters
    ----------
    source : FitResult or Model
        A fit, whose estimates are used, or a model specification whose item parameters are
        given by ``parameters``.
    n_obs : int or mapping of str to int, optional
        Number of respondents. For a multiple-group model an integer is the number per group,
        and a mapping gives each group's number (groups left out get none). Required unless
        ``theta`` is given.
    key : jax.Array
        JAX PRNG key (``jax.random.key(0)`` or ``jax.random.PRNGKey(0)``). The same key gives
        the same data.
    parameters : DataFrame, optional
        With a ``Model`` only (and then required): an item bank as for
        :meth:`Model.fix_from` in the slope–intercept parameterization (an ``item`` column and
        one column per internal parameter: ``a`` or ``a1``.., ``d`` or ``d1``.., ``g``, ``u``,
        ``ak1``..). Every item parameter must be given. The latent distribution is the
        specification's: a ``Normal``'s ``mean`` and ``var`` (factor correlations 0 for
        ``with_factors()`` models) or a ``NormalMixture``'s ``means``, ``variances`` and
        ``weights``; pass ``theta`` for any other latent distribution.
    theta : array-like, optional
        Latent traits to use instead of draws, shape ``(N,)`` (or ``(N, 1)``) for one factor
        and ``(N, D)`` otherwise; ``n_obs`` is then omitted.
    group : array-like, optional
        With ``theta`` for a multiple-group model: the group label of each row of ``theta``.

    Returns
    -------
    SimulatedData
        ``responses`` (a DataFrame with the item names as columns), ``theta`` and ``group``.
        Rows are ordered by group (in the model's group order) when ``theta`` is drawn.

    Raises
    ------
    PrecisionError
        If JAX is not configured for 64-bit floating point.
    SpecificationError
        If ``key`` is not a PRNG key, ``n_obs`` / ``theta`` / ``group`` / ``parameters`` do
        not fit the model, or a ``Model`` with an ``EmpiricalHistogram`` or ``DavidianCurve``
        latent distribution is given without ``theta`` (their shape exists only after
        fitting).

    Notes
    -----
    Missing responses are not simulated: to mimic a test design or a missing-data mechanism,
    set entries of ``responses`` to NaN afterwards.

    Examples
    --------
    >>> spec = Model.uniform(TwoPL(), items=cols)
    >>> sim = simulate_responses(fit(spec, data), n_obs=1000, key=jax.random.key(1))
    >>> refit = fit(spec, sim.responses)
    """
    check_x64()
    if not _is_prng_key(key):
        raise SpecificationError(
            "key must be a JAX PRNG key from jax.random.key() or jax.random.PRNGKey()."
        )
    if isinstance(source, FitResult):
        if parameters is not None:
            raise SpecificationError(
                "parameters= is used with a Model only; a FitResult has its own estimates."
            )
        src = _source_from_fit(source)
    elif isinstance(source, Model):
        if parameters is None:
            raise SpecificationError(
                "Simulation from a Model needs parameters= (an item bank as in "
                "Model.fix_from()) for the item parameters."
            )
        if theta is None:
            src = _source_from_model(source, parameters)
        else:
            src = _source_from_model(_without_latent_shape(source), parameters)
    else:
        raise SpecificationError(
            f"source must be a FitResult or a Model, got {type(source).__name__}."
        )
    model = src.model
    groups = [g for g, _ in model.resolved_groups()]
    k_theta, k_unif = jax.random.split(key)

    if theta is None:
        if group is not None:
            raise SpecificationError("group= is used with theta= only; use n_obs per group.")
        if n_obs is None:
            raise SpecificationError("n_obs is required unless theta is given.")
        sizes = _group_sizes(model, n_obs)
        keys = jax.random.split(k_theta, len(groups))
        draws = [
            _draw_theta(src, g_idx, g_name, keys[g_idx], n)
            for g_idx, (g_name, n) in enumerate(zip(groups, sizes, strict=True))
            if n > 0
        ]
        theta_arr = np.concatenate(draws, axis=0)
        codes = np.repeat(np.arange(len(groups), dtype=np.int32), sizes)
    else:
        theta_arr, codes = _given_theta(model, theta, group, n_obs)

    n = theta_arr.shape[0]
    probs = _category_probabilities(src, theta_arr)
    uniforms = np.asarray(jax.random.uniform(k_unif, (n, len(model.items)), dtype=jnp.float64))
    rows = np.arange(n)
    columns = {
        name: responses_from_uniforms(p[codes, rows], uniforms[:, j])
        for j, ((name, _), p) in enumerate(zip(model.items, probs, strict=True))
    }
    return SimulatedData(
        responses=pd.DataFrame(columns),
        theta=theta_arr[:, 0].copy() if model.n_factors == 1 else theta_arr,
        group=np.asarray(groups, dtype=object)[codes] if model.is_multigroup else None,
    )


def _without_latent_shape(model: Model) -> Model:
    """With ``theta`` given, the latent distribution is not used: drop shapes needing a fit."""
    from flexpyirt.latent import DavidianCurve, EmpiricalHistogram, Normal

    def plain(dist: Any) -> Any:
        return Normal() if isinstance(dist, (DavidianCurve, EmpiricalHistogram)) else dist

    if model.groups is not None:
        return replace(model, groups=tuple((g, plain(d)) for g, d in model.groups))
    if model.latent is not None:
        return replace(model, latent=plain(model.latent))
    return model
