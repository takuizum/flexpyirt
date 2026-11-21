"""Latent (population) distributions (Spec layer) and their quadrature weights (Core layer)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from flexpyirt.errors import SpecificationError

__all__ = [
    "EmpiricalHistogram",
    "LatentSpec",
    "Normal",
    "correlation_matrix_is_pd",
    "eh_constrained_weights",
    "eh_moments",
    "eh_project_floor",
    "latent_param_names",
    "mvn_log_weights",
    "normal_log_weights",
]


@dataclass(frozen=True)
class Normal:
    r"""Normal latent distribution :math:`\theta\sim N(\mu,\sigma^2)` on a quadrature grid.

    Parameters
    ----------
    mean, var : float
        Value of the mean and variance; the starting value when the parameter is free.
    free_mean, free_var : bool
        Whether the mean / variance is estimated.

    Examples
    --------
    >>> Normal.standard()          # N(0, 1), nothing estimated
    >>> Normal(free_var=True)      # N(0, sigma^2), variance estimated (Rasch default)
    >>> Normal.free()              # mean and variance estimated
    """

    mean: float = 0.0
    var: float = 1.0
    free_mean: bool = False
    free_var: bool = False

    def __post_init__(self) -> None:
        """Validate values."""
        if not math.isfinite(self.mean):
            raise SpecificationError(f"Normal mean must be finite, got {self.mean}.")
        if not (math.isfinite(self.var) and self.var > 0):
            raise SpecificationError(
                f"Normal variance must be positive and finite, got {self.var}."
            )

    @classmethod
    def standard(cls) -> Normal:
        """Return the fixed standard normal :math:`N(0,1)`."""
        return cls()

    @classmethod
    def free(cls, *, mean: float = 0.0, var: float = 1.0) -> Normal:
        """Return a normal with both mean and variance estimated (starting at ``mean``, ``var``)."""
        return cls(mean=mean, var=var, free_mean=True, free_var=True)


def normal_log_weights(nodes: Array, mean: Array, var: Array) -> Array:
    r"""Log quadrature weights of a normal density normalized on the grid.

    .. math::
        \log w_q = \log\phi(\theta_q;\mu,\sigma^2) - \log\sum_{q'}\phi(\theta_{q'};\mu,\sigma^2)

    The normalizing constant of :math:`\phi` cancels, so only the kernel
    :math:`-(\theta_q-\mu)^2/(2\sigma^2)` is evaluated (DESIGN.md §4.4).

    Parameters
    ----------
    nodes : Array, shape (Q,)
    mean, var : Array, scalar

    Returns
    -------
    Array, shape (Q,)
    """
    kernel = -0.5 * (nodes - mean) ** 2 / var
    return jax.nn.log_softmax(kernel)


@dataclass(frozen=True, kw_only=True)
class EmpiricalHistogram:
    r"""Empirical histogram (EH) nonparametric latent distribution (Woods, 2007).

    Represents each group's latent distribution by discrete weights
    :math:`w_{g1},\dots,w_{gQ}` on the fixed quadrature grid :math:`\theta_1,\dots,\theta_Q`
    satisfying :math:`w_{gq}\ge\tau>0` and :math:`\sum_q w_{gq}=1` (DESIGN.md §4.4, §15.34).
    During EM, each group's EH M-step solves
    :math:`\max_{w\in\Delta^{Q-1},\,w_q\ge\tau}\sum_q \bar n_{gq}\log w_q` via exact KKT
    water-filling (reducing identically to :math:`\bar n_{gq}/N_g` when no node hits the floor).
    Unless the scale is pinned by fixed item parameters (FIPC), the reference group's M-step adds
    the constraints :math:`\sum_q w_q\theta_q=0` and :math:`\sum_q w_q\theta_q^2=1` (mean only
    for Rasch/PCM), so its histogram has mean 0 and variance 1 after every iteration and EM stays
    monotone (:func:`eh_constrained_weights`).

    Parameters
    ----------
    min_weight : float, default 1e-10
        Strict positive floor :math:`\tau` on every quadrature node's weight so empty nodes keep
        finite log-weights. Must satisfy :math:`0 < \tau < 1/Q` for the quadrature grid used.
    """

    min_weight: float = 1e-10

    def __post_init__(self) -> None:
        """Validate ``min_weight``."""
        if isinstance(self.min_weight, bool) or not isinstance(self.min_weight, (int, float)):
            raise SpecificationError(
                f"EmpiricalHistogram min_weight must be a float, got {self.min_weight!r}."
            )
        val = float(self.min_weight)
        if not (math.isfinite(val) and 0.0 < val < 1.0):
            raise SpecificationError(
                "EmpiricalHistogram min_weight must be finite and in (0, 1), "
                f"got {self.min_weight}."
            )


LatentSpec = Normal | EmpiricalHistogram
"""Union of supported latent distribution specifications."""


def eh_project_floor(counts: Array, min_weight: Array | float) -> Array:
    r"""Exact KKT maximizer of :math:`\sum_q \bar n_q\log w_q` on the floored simplex.

    Solves

    .. math::
        \max_{w\in\Delta^{Q-1},\;w_q\ge\tau}\sum_{q=1}^Q \bar n_q \log w_q,

    whose unique KKT solution is :math:`w_q=\max(\tau,\,c^*\bar n_q)` with :math:`c^*>0` chosen
    so :math:`\sum_q w_q=1` (DESIGN.md §4.4, §15.34). Reduces identically to
    :math:`\bar n_q / \sum_{q'}\bar n_{q'}` whenever no node hits the floor :math:`\tau`.
    """
    n_nodes = counts.shape[0]
    total = jnp.sum(counts)
    r = jnp.where(total > 0.0, counts / jnp.where(total > 0.0, total, 1.0), 1.0 / n_nodes)
    order = jnp.argsort(r)
    r_sorted = r[order]
    suf = jnp.cumsum(r_sorted[::-1])[::-1]
    ks = jnp.arange(n_nodes, dtype=r.dtype)
    rem = 1.0 - ks * min_weight
    c_k = jnp.where(suf > 0.0, rem / jnp.where(suf > 0.0, suf, 1.0), 0.0)
    above = (c_k * r_sorted) > min_weight
    has_above = jnp.any(above)
    k_opt = jnp.argmax(above)
    c_opt = c_k[k_opt]
    w = jnp.where(has_above, jnp.maximum(min_weight, c_opt * r), jnp.full_like(r, 1.0 / n_nodes))
    return w / jnp.sum(w)


_EH_DUAL_MAX_ITER = 200
"""Maximum dual Newton iterations in :func:`eh_constrained_weights`."""
_EH_DUAL_GRAD_TOL = 1e-13
"""Convergence tolerance on the moment residuals in :func:`eh_constrained_weights`."""


def eh_constrained_weights(
    counts: Array, nodes: Array, min_weight: Array | float, n_moments: int
) -> Array:
    r"""Maximize :math:`\sum_q \bar n_q\log w_q` on the floored simplex with moment constraints.

    Solves the reference-group EH M-step under the identification of DESIGN.md §4.4 / §15.34:

    .. math::
        \max_{w}\ \sum_{q=1}^Q p_q\log w_q
        \quad\text{s.t.}\quad w_q\ge\tau,\quad \sum_q w_q=1,\quad
        \sum_q w_q\theta_q=0\ (n_{\text{moments}}\ge 1),\quad
        \sum_q w_q\theta_q^2=1\ (n_{\text{moments}}=2),

    with :math:`p_q=\bar n_q/\sum_r\bar n_r`. With :math:`\phi_q=(1,\theta_q,\theta_q^2)` (first
    :math:`K=n_{\text{moments}}+1` entries) and multipliers :math:`\lambda\in\mathbb R^K`, the
    KKT solution is :math:`w_q(\lambda)=\max\{\tau,\ p_q/(\lambda^\top\phi_q)\}`, where
    :math:`\lambda` minimizes the smooth convex dual

    .. math::
        G(\lambda)=\sum_q\big[p_q\log w_q(\lambda)-(\lambda^\top\phi_q)\,w_q(\lambda)\big]
        +\lambda^\top c,\qquad c=(1,0,1)_{1:K},

    over :math:`\{\lambda:\lambda^\top\phi_q>0\ \forall q\}` (Boyd & Vandenberghe, 2004,
    §5.2). :math:`G` is minimized by damped Newton with backtracking from
    :math:`\lambda=(1,0,\dots)`; :math:`\nabla G=c-\sum_q w_q\phi_q` is the constraint residual,
    so the returned weights satisfy the moment constraints to about ``1e-13``. The objective is
    the EM latent Q-function, so the M-step is exact and EM stays monotone. ``n_moments=0``
    reduces to :func:`eh_project_floor`. Used as an I-projection (``counts`` = candidate weights)
    for SQUAREM candidates (§15.36).

    The minimizer is interior whenever every :math:`p_q>0` (E-step counts and softmax candidates
    always are): :math:`G\to\infty` as any :math:`\lambda^\top\phi_q\downarrow 0`. If some
    :math:`p_q=0` and the optimum needs extra mass on those nodes, the dual infimum lies on the
    boundary; Newton then stops early and the constraints are not met, which ``fit()`` reports
    (DESIGN.md §15.34).

    Parameters
    ----------
    counts : (Q,) float
        Expected counts :math:`\bar n_q \ge 0` (or candidate weights).
    nodes : (Q,) float
        Quadrature nodes :math:`\theta_q`.
    min_weight : float
        Floor :math:`\tau`.
    n_moments : {0, 1, 2}
        Static number of moment constraints (mean 0; mean 0 and variance 1).
    """
    if n_moments == 0:
        return eh_project_floor(counts, min_weight)
    dtype = counts.dtype
    total = jnp.sum(counts)
    p = counts / total
    k = n_moments + 1
    phi = jnp.stack([jnp.ones_like(nodes), nodes, nodes**2], axis=1)[:, :k]  # (Q, K)
    c = jnp.asarray([1.0, 0.0, 1.0], dtype=dtype)[:k]

    def weights(lam: Array) -> tuple[Array, Array, Array]:
        sq = phi @ lam
        safe = jnp.where(sq > 0.0, sq, 1.0)
        free = p > min_weight * safe
        w = jnp.where(free, p / safe, min_weight)
        return w, sq, free

    def dual(lam: Array) -> Array:
        w, sq, _ = weights(lam)
        logw = jnp.log(w)
        val = jnp.sum(jnp.where(p > 0.0, p * logw, 0.0) - sq * w) + lam @ c
        return jnp.where(jnp.all(sq > 0.0), val, jnp.inf)

    def grad_hess(lam: Array) -> tuple[Array, Array]:
        w, sq, free = weights(lam)
        g = c - phi.T @ w
        curv = jnp.where(free, p / jnp.where(sq > 0.0, sq, 1.0) ** 2, 0.0)
        h = (phi * curv[:, None]).T @ phi
        return g, h

    def cond(state: tuple[Array, Array, Array, Array]) -> Array:
        _, g, it, stalled = state
        return (jnp.max(jnp.abs(g)) > _EH_DUAL_GRAD_TOL) & (it < _EH_DUAL_MAX_ITER) & ~stalled

    def body(state: tuple[Array, Array, Array, Array]) -> tuple[Array, Array, Array, Array]:
        lam, g, it, _ = state
        _, h = grad_hess(lam)
        step = -jnp.linalg.solve(h, g)
        f0 = dual(lam)
        slope = g @ step

        g_inf = jnp.max(jnp.abs(g))

        def armijo_fails(t: Array) -> Array:
            # Near the optimum the dual decrease falls below rounding; a feasible step that
            # shrinks the constraint residual is then accepted instead.
            cand = lam + t * step
            f1 = dual(cand)
            g1, _ = grad_hess(cand)
            shrinks = jnp.isfinite(f1) & (jnp.max(jnp.abs(g1)) < 0.5 * g_inf)
            accept: Array = (f1 <= f0 + 1e-4 * t * slope) | shrinks
            return ~accept

        t = jax.lax.while_loop(
            lambda t: armijo_fails(t) & (t > 1e-12), lambda t: 0.5 * t, jnp.ones((), dtype=dtype)
        )
        ok = ~armijo_fails(t)
        lam_new = jnp.where(ok, lam + t * step, lam)
        g_new, _ = grad_hess(lam_new)
        return lam_new, g_new, it + 1, ~ok

    lam0 = jnp.zeros((k,), dtype=dtype).at[0].set(1.0)
    g0, _ = grad_hess(lam0)
    init = (lam0, g0, jnp.zeros((), dtype=jnp.int32), jnp.zeros((), dtype=jnp.bool_))
    lam, _, _, _ = jax.lax.while_loop(cond, body, init)
    w, _, _ = weights(lam)
    return w


def eh_moments(nodes: np.ndarray, weights: np.ndarray) -> tuple[float, float, float, float, float]:
    r"""Return ``(mean, var, sd, skewness, kurtosis)`` of discrete EH weights on ``nodes``.

    .. math::
        \mu = \sum_q w_q\theta_q,\quad
        \sigma^2 = \sum_q w_q(\theta_q-\mu)^2,\quad
        \gamma_1 = \frac{\sum_q w_q(\theta_q-\mu)^3}{\sigma^3},\quad
        \gamma_2 = \frac{\sum_q w_q(\theta_q-\mu)^4}{\sigma^4} - 3

    where :math:`\gamma_2` is Fisher's excess kurtosis (0 for a normal distribution).
    """
    th = np.asarray(nodes, dtype=float)
    w = np.asarray(weights, dtype=float)
    mean = float(np.sum(w * th))
    centered = th - mean
    var = float(np.sum(w * centered**2))
    if var <= 0.0:
        return mean, max(var, 0.0), 0.0, math.nan, math.nan
    sd = math.sqrt(var)
    skewness = float(np.sum(w * centered**3)) / (sd**3)
    kurtosis = float(np.sum(w * centered**4)) / (var**2) - 3.0
    return mean, var, sd, skewness, kurtosis


def latent_param_names(n_factors: int) -> tuple[str, ...]:
    """Return the latent parameter names for ``n_factors`` in ``{1, 2, 3}``.

    For ``n_factors == 1``, returns ``("mean", "var")``. For ``n_factors > 1``, returns
    ``("mean_1", ..., "mean_D", "var_1", ..., "var_D", "cor_12", ...)`` with correlation pairs in
    lexicographic order ``i < j``.
    """
    if n_factors == 1:
        return ("mean", "var")
    if n_factors not in (2, 3):
        raise SpecificationError(f"Multivariate normal supports 1 to 3 factors, got {n_factors}.")
    means = tuple(f"mean_{i}" for i in range(1, n_factors + 1))
    vars_ = tuple(f"var_{i}" for i in range(1, n_factors + 1))
    cors = tuple(
        f"cor_{i}{j}" for i in range(1, n_factors + 1) for j in range(i + 1, n_factors + 1)
    )
    return (*means, *vars_, *cors)


def correlation_matrix_is_pd(cors: Sequence[float], n_factors: int) -> bool:
    """Return True iff the ``n_factors`` × ``n_factors`` correlation matrix is positive definite."""
    if n_factors == 1:
        return True
    if n_factors == 2:
        r12 = float(cors[0])
        return math.isfinite(r12) and abs(r12) < 1.0
    if n_factors == 3:
        r12, r13, r23 = float(cors[0]), float(cors[1]), float(cors[2])
        if not (math.isfinite(r12) and math.isfinite(r13) and math.isfinite(r23)):
            return False
        m2 = 1.0 - r12 * r12
        det = 1.0 - r12 * r12 - r13 * r13 - r23 * r23 + 2.0 * r12 * r13 * r23
        return m2 > 0.0 and det > 0.0
    raise SpecificationError(f"Multivariate normal supports 1 to 3 factors, got {n_factors}.")


def mvn_log_weights(nodes: Array, params: Array) -> Array:
    r"""Log quadrature weights of a :math:`D`-dimensional normal density normalized on the grid.

    .. math::
        \log w_q = \log\phi_D(\boldsymbol\theta_q; \boldsymbol\mu, \boldsymbol\Sigma)
                 - \log\sum_{q'}\phi_D(\boldsymbol\theta_{q'}; \boldsymbol\mu, \boldsymbol\Sigma),
        \qquad
        \Sigma_{ii} = v_i > 0,\quad \Sigma_{ij} = \rho_{ij}\sqrt{v_i v_j}\ (i \neq j).

    The normalizing constant :math:`(2\pi)^{-D/2}|\boldsymbol\Sigma|^{-1/2}` is independent of
    :math:`q` and cancels inside ``log_softmax``. For :math:`D = 1` this dispatches to
    :func:`normal_log_weights` unchanged. For :math:`D \in \{2, 3\}`, positive definiteness of the
    correlation matrix is checked by Sylvester's criterion (leading principal minors :math:`> 0`)
    with double-``where`` guards so derivatives at valid interior points never see NaN, while
    non-positive-definite candidates return NaN and are rejected by the Newton line search and
    SQUAREM safeguard (DESIGN.md §4.4, §15.30).

    Parameters
    ----------
    nodes : Array, shape (Q, D) or (Q,)
        Product quadrature nodes.
    params : Array, shape (2*D + D*(D-1)//2,)
        Latent parameters in :func:`latent_param_names` order.

    Returns
    -------
    Array, shape (Q,)
    """
    n_factors = 1 if nodes.ndim == 1 else int(nodes.shape[1])
    if n_factors == 1:
        t = nodes if nodes.ndim == 1 else nodes[:, 0]
        return normal_log_weights(t, params[0], params[1])

    means = params[:n_factors]
    vars_ = params[n_factors : 2 * n_factors]
    ok_var = jnp.all(vars_ > 0.0)
    safe_vars = jnp.where(vars_ > 0.0, vars_, 1.0)
    z = (nodes - means[None, :]) / jnp.sqrt(safe_vars)[None, :]  # (Q, D)

    if n_factors == 2:
        r12 = params[4]
        det = 1.0 - r12 * r12
        is_pd = ok_var & (det > 0.0)
        safe_det = jnp.where(is_pd, det, 1.0)
        quad = (z[:, 0] ** 2 - 2.0 * r12 * z[:, 0] * z[:, 1] + z[:, 1] ** 2) / safe_det
        logw = jax.nn.log_softmax(-0.5 * quad)
        return jnp.where(is_pd, logw, jnp.nan)

    if n_factors == 3:
        r12, r13, r23 = params[6], params[7], params[8]
        m2 = 1.0 - r12 * r12
        det = 1.0 - r12 * r12 - r13 * r13 - r23 * r23 + 2.0 * r12 * r13 * r23
        is_pd = ok_var & (m2 > 0.0) & (det > 0.0)
        one = jnp.ones((), dtype=params.dtype)
        corr = jnp.array([[one, r12, r13], [r12, one, r23], [r13, r23, one]])
        safe_corr = jnp.where(is_pd, corr, jnp.eye(3, dtype=params.dtype))
        chol = jnp.linalg.cholesky(safe_corr)
        y = jax.scipy.linalg.solve_triangular(chol, z.T, lower=True)
        quad = jnp.sum(y * y, axis=0)
        logw = jax.nn.log_softmax(-0.5 * quad)
        return jnp.where(is_pd, logw, jnp.nan)

    raise SpecificationError(f"Multivariate normal supports 1 to 3 factors, got {n_factors}.")
