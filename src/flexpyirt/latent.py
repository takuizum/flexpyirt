"""Latent (population) distributions (Spec layer) and their quadrature weights (Core layer)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from flexpyirt.errors import SpecificationError

__all__ = [
    "Normal",
    "correlation_matrix_is_pd",
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
