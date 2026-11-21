"""Prior distributions on item parameters (Spec layer) and branch-free evaluation (Core layer).

Priors are defined on the **natural parameter scale** without a Jacobian term (DESIGN.md §4.7):
evaluating :meth:`log_pdf` at an item parameter value directly gives its contribution to the log
posterior. The normal prior is named :class:`NormalPrior` to avoid clashing with the latent
distribution :class:`~flexpyirt.Normal` (API.md §7).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from flexpyirt.errors import SpecificationError

__all__ = [
    "PRIOR_BETA",
    "PRIOR_LOGNORMAL",
    "PRIOR_NONE",
    "PRIOR_NORMAL",
    "Beta",
    "LogNormal",
    "NormalPrior",
    "Prior",
    "encode_prior",
    "eval_log_prior",
]

PRIOR_NONE = 0
PRIOR_NORMAL = 1
PRIOR_LOGNORMAL = 2
PRIOR_BETA = 3

_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


def _validate_finite(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SpecificationError(f"{name} must be a finite real number, got {value!r}.")


def _validate_positive(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not (math.isfinite(value) and value > 0)
    ):
        raise SpecificationError(f"{name} must be positive and finite, got {value!r}.")


def _as_float_array(x: ArrayLike) -> Array:
    dtype = jnp.result_type(x, 0.0)
    return jnp.asarray(x, dtype=dtype)


@dataclass(frozen=True)
class NormalPrior:
    r"""Normal prior distribution on the natural parameter scale.

    .. math::
        \log\pi(x) = -\frac{1}{2}\log(2\pi) - \log\sigma - \frac{(x - \mu)^2}{2\sigma^2},
        \qquad x \in (-\infty, \infty).

    Corresponds to ``norm`` in R ``mirt`` (Mislevy, 1986; Chalmers, 2012). Named
    :class:`NormalPrior` to avoid clashing with the latent distribution :class:`~flexpyirt.Normal`.

    Parameters
    ----------
    mean : float
        Prior mean :math:`\mu` (finite).
    sd : float
        Prior standard deviation :math:`\sigma` (positive and finite).
    """

    mean: float
    sd: float

    def __post_init__(self) -> None:
        """Validate prior hyperparameters."""
        _validate_finite(self.mean, "NormalPrior mean")
        _validate_positive(self.sd, "NormalPrior sd")

    @property
    def support(self) -> tuple[float, float]:
        r"""Open support :math:`(-\infty, \infty)`."""
        return (-math.inf, math.inf)

    def log_pdf(self, x: ArrayLike) -> Array:
        """Evaluate the log probability density at ``x``."""
        arr = _as_float_array(x)
        z = (arr - self.mean) / self.sd
        return -_HALF_LOG_2PI - jnp.log(jnp.asarray(self.sd, dtype=arr.dtype)) - 0.5 * z * z


@dataclass(frozen=True)
class LogNormal:
    r"""Log-normal prior distribution on the natural parameter scale.

    .. math::
        \log\pi(x) = -\log x - \log s - \frac{1}{2}\log(2\pi)
                     - \frac{(\log x - \mu)^2}{2s^2},
        \qquad x > 0,

    and :math:`-\infty` for :math:`x \le 0`. Parameterized on the log scale
    (:math:`\log X \sim N(\mu, s^2)`), matching R ``dlnorm`` and ``lnorm`` in ``mirt``
    (Mislevy, 1986; Chalmers, 2012). Commonly used for item slopes :math:`a`.

    Parameters
    ----------
    meanlog : float
        Mean :math:`\mu` of :math:`\log X` (finite).
    sdlog : float
        Standard deviation :math:`s` of :math:`\log X` (positive and finite).
    """

    meanlog: float
    sdlog: float

    def __post_init__(self) -> None:
        """Validate prior hyperparameters."""
        _validate_finite(self.meanlog, "LogNormal meanlog")
        _validate_positive(self.sdlog, "LogNormal sdlog")

    @property
    def support(self) -> tuple[float, float]:
        r"""Open support :math:`(0, \infty)`."""
        return (0.0, math.inf)

    def log_pdf(self, x: ArrayLike) -> Array:
        """Evaluate the log probability density at ``x`` (``-inf`` for ``x <= 0``)."""
        arr = _as_float_array(x)
        valid = arr > 0.0
        safe_x = jnp.where(valid, arr, 1.0)
        log_x = jnp.log(safe_x)
        z = (log_x - self.meanlog) / self.sdlog
        log_sd = jnp.log(jnp.asarray(self.sdlog, dtype=arr.dtype))
        val = -log_x - log_sd - _HALF_LOG_2PI - 0.5 * z * z
        return jnp.where(valid, val, -jnp.inf)


@dataclass(frozen=True)
class Beta:
    r"""Beta prior distribution on the natural parameter scale.

    .. math::
        \log\pi(x) = (a - 1)\log x + (b - 1)\log(1 - x) - \log B(a, b),
        \qquad 0 < x < 1,

    and :math:`-\infty` outside :math:`(0, 1)`. Evaluated directly on the natural-scale parameter
    without a logit Jacobian (DESIGN.md §4.7), matching R ``dbeta`` and the log-prior objective
    of ``expbeta`` in ``mirt`` (Swaminathan & Gifford, 1986; Chalmers, 2012). Commonly used for
    3PL/4PL asymptotes :math:`g` and :math:`u`.

    Parameters
    ----------
    a, b : float
        Positive finite shape parameters (:math:`a > 0`, :math:`b > 0`). For a unimodal interior
        prior mode at :math:`(a - 1) / (a + b - 2)`, choose :math:`a > 1` and :math:`b > 1`.
    """

    a: float
    b: float

    def __post_init__(self) -> None:
        """Validate prior hyperparameters."""
        _validate_positive(self.a, "Beta a")
        _validate_positive(self.b, "Beta b")

    @property
    def support(self) -> tuple[float, float]:
        """Open support :math:`(0, 1)`."""
        return (0.0, 1.0)

    def log_pdf(self, x: ArrayLike) -> Array:
        """Evaluate the log probability density at ``x`` (``-inf`` outside ``(0, 1)``)."""
        arr = _as_float_array(x)
        valid = (arr > 0.0) & (arr < 1.0)
        safe_x = jnp.where(valid, arr, 0.5)
        a_arr = jnp.asarray(self.a, dtype=arr.dtype)
        b_arr = jnp.asarray(self.b, dtype=arr.dtype)
        val = (
            (a_arr - 1.0) * jnp.log(safe_x)
            + (b_arr - 1.0) * jnp.log1p(-safe_x)
            - jax.scipy.special.betaln(a_arr, b_arr)
        )
        return jnp.where(valid, val, -jnp.inf)


Prior = NormalPrior | LogNormal | Beta
PRIOR_TYPES: tuple[type[NormalPrior], type[LogNormal], type[Beta]] = (
    NormalPrior,
    LogNormal,
    Beta,
)


def encode_prior(prior: Prior | None) -> tuple[int, float, float]:
    """Encode a prior object as ``(kind, param1, param2)`` for the Core layer."""
    if prior is None:
        return (PRIOR_NONE, 0.0, 1.0)
    if isinstance(prior, NormalPrior):
        return (PRIOR_NORMAL, float(prior.mean), float(prior.sd))
    if isinstance(prior, LogNormal):
        return (PRIOR_LOGNORMAL, float(prior.meanlog), float(prior.sdlog))
    if isinstance(prior, Beta):
        return (PRIOR_BETA, float(prior.a), float(prior.b))
    raise SpecificationError(
        f"Unsupported prior distribution {prior!r}; use NormalPrior, LogNormal, Beta or None."
    )


def eval_log_prior(x: Array, kind: Array, param1: Array, param2: Array) -> Array:
    r"""Branch-free elementwise log-prior evaluation for jit (DESIGN.md §4.7).

    Uses the double-``where`` pattern on every branch so that unselected branches and unused
    padding slots (``kind == PRIOR_NONE``) never evaluate :math:`\log 0` or negative logs and
    contribute zero value, gradient and Hessian. Out-of-support values for an active
    :class:`LogNormal` or :class:`Beta` prior return :math:`-\infty` (rejected by the M-step
    line search and SQUAREM safeguard).
    """
    is_norm = kind == PRIOR_NORMAL
    sd_n = jnp.where(is_norm, param2, 1.0)
    z_n = (x - param1) / sd_n
    lp_norm = jnp.where(is_norm, -_HALF_LOG_2PI - jnp.log(sd_n) - 0.5 * z_n * z_n, 0.0)

    is_ln = kind == PRIOR_LOGNORMAL
    valid_ln = is_ln & (x > 0.0)
    sd_ln = jnp.where(is_ln, param2, 1.0)
    safe_x_ln = jnp.where(valid_ln, x, 1.0)
    log_x = jnp.log(safe_x_ln)
    z_ln = (log_x - param1) / sd_ln
    lp_ln_val = -log_x - jnp.log(sd_ln) - _HALF_LOG_2PI - 0.5 * z_ln * z_ln
    lp_ln = jnp.where(valid_ln, lp_ln_val, jnp.where(is_ln, -jnp.inf, 0.0))

    is_beta = kind == PRIOR_BETA
    valid_beta = is_beta & (x > 0.0) & (x < 1.0)
    a_b = jnp.where(is_beta, param1, 1.0)
    b_b = jnp.where(is_beta, param2, 1.0)
    safe_x_b = jnp.where(valid_beta, x, 0.5)
    lp_beta_val = (
        (a_b - 1.0) * jnp.log(safe_x_b)
        + (b_b - 1.0) * jnp.log1p(-safe_x_b)
        - jax.scipy.special.betaln(a_b, b_b)
    )
    lp_beta = jnp.where(valid_beta, lp_beta_val, jnp.where(is_beta, -jnp.inf, 0.0))

    return lp_norm + lp_ln + lp_beta
