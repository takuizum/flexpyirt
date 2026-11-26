"""Prior distributions on item parameters (Spec layer) and branch-free evaluation (Core layer).

Priors are defined on the **natural parameter scale** without a Jacobian term (DESIGN.md §4.7):
evaluating :meth:`log_pdf` at an item parameter value directly gives its contribution to the log
posterior. :class:`LogitNormalPrior` is a density on :math:`(0, 1)` whose formula contains the
logit Jacobian because that is part of the density itself, unless ``mode_scale="logit"`` asks for
the logit-metric posterior mode (DESIGN.md §15.48). The normal prior is
named :class:`NormalPrior` to avoid clashing with the latent distribution
:class:`~flexpyirt.Normal` (API.md §7).
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any, Literal, get_args

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from flexpyirt.errors import SpecificationError

__all__ = [
    "PRIOR_BETA",
    "PRIOR_LOGITNORMAL",
    "PRIOR_LOGITNORMAL_LOGIT",
    "PRIOR_LOGNORMAL",
    "PRIOR_NONE",
    "PRIOR_NORMAL",
    "BetaPrior",
    "LogNormalPrior",
    "LogitNormalPrior",
    "NormalPrior",
    "Prior",
    "encode_prior",
    "eval_log_prior",
]

PRIOR_NONE = 0
PRIOR_NORMAL = 1
PRIOR_LOGNORMAL = 2
PRIOR_BETA = 3
PRIOR_LOGITNORMAL = 4
PRIOR_LOGITNORMAL_LOGIT = 5

LogitModeScale = Literal["natural", "logit"]
_LOGIT_MODE_SCALES: tuple[str, ...] = get_args(LogitModeScale)

_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)

RealScalar = float | numbers.Real | np.integer[Any] | np.floating[Any]


def _validate_finite(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise SpecificationError(f"{name} must be a finite real number, got {value!r}.")
    try:
        out = float(value)
    except OverflowError:
        raise SpecificationError(f"{name} must be a finite real number, got {value!r}.") from None
    if not math.isfinite(out):
        raise SpecificationError(f"{name} must be a finite real number, got {value!r}.")
    return out


def _validate_positive(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise SpecificationError(f"{name} must be positive and finite, got {value!r}.")
    try:
        out = float(value)
    except OverflowError:
        raise SpecificationError(f"{name} must be positive and finite, got {value!r}.") from None
    if not (math.isfinite(out) and out > 0.0):
        raise SpecificationError(f"{name} must be positive and finite, got {value!r}.")
    return out


def _as_float_array(x: ArrayLike) -> Array:
    dtype = jnp.result_type(x, 0.0)
    return jnp.asarray(x, dtype=dtype)


def _log_beta(a: Array, b: Array) -> Array:
    """Log Beta function via ``gammaln`` (full float64 precision; ``betaln`` is ~1e-7 in JAX)."""
    return (
        jax.scipy.special.gammaln(a)
        + jax.scipy.special.gammaln(b)
        - jax.scipy.special.gammaln(a + b)
    )


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

    def __init__(self, mean: RealScalar, sd: RealScalar) -> None:
        """Validate prior hyperparameters and store them as Python floats."""
        object.__setattr__(self, "mean", _validate_finite(mean, "NormalPrior mean"))
        object.__setattr__(self, "sd", _validate_positive(sd, "NormalPrior sd"))

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
class LogNormalPrior:
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
    mean_log : float
        Mean :math:`\mu` of :math:`\log X` (finite).
    sd_log : float
        Standard deviation :math:`s` of :math:`\log X` (positive and finite).
    """

    mean_log: float
    sd_log: float

    def __init__(self, mean_log: RealScalar, sd_log: RealScalar) -> None:
        """Validate prior hyperparameters and store them as Python floats."""
        object.__setattr__(self, "mean_log", _validate_finite(mean_log, "LogNormal mean_log"))
        object.__setattr__(self, "sd_log", _validate_positive(sd_log, "LogNormal sd_log"))

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
        z = (log_x - self.mean_log) / self.sd_log
        log_sd = jnp.log(jnp.asarray(self.sd_log, dtype=arr.dtype))
        val = -log_x - log_sd - _HALF_LOG_2PI - 0.5 * z * z
        return jnp.where(valid, val, -jnp.inf)


@dataclass(frozen=True)
class BetaPrior:
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

    def __init__(self, a: RealScalar, b: RealScalar) -> None:
        """Validate prior hyperparameters and store them as Python floats."""
        object.__setattr__(self, "a", _validate_positive(a, "Beta a"))
        object.__setattr__(self, "b", _validate_positive(b, "Beta b"))

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
            - _log_beta(a_arr, b_arr)
        )
        return jnp.where(valid, val, -jnp.inf)


@dataclass(frozen=True)
class LogitNormalPrior:
    r"""Logit-normal prior distribution on :math:`(0, 1)` (natural parameter scale).

    :math:`\operatorname{logit}X \sim N(\mu, s^2)` (Aitchison & Shen, 1980), so the density of
    :math:`X` itself contains the Jacobian of the logit:

    .. math::
        \log\pi(x) = -\log x - \log(1 - x) - \log s - \frac{1}{2}\log(2\pi)
                     - \frac{(\operatorname{logit} x - \mu)^2}{2s^2},
        \qquad 0 < x < 1,

    and :math:`-\infty` outside :math:`(0, 1)`. Used for 3PL/4PL asymptotes :math:`g` and
    :math:`u`; :math:`\mu=\operatorname{logit}(1/4)\approx-1.0986` centers :math:`g` on the
    chance rate of four-option items.

    flexMIRT's ``Guessing : Normal(mu, s)`` prior (Houts & Cai, flexMIRT user's manual) and R
    ``mirt``'s ``PRIOR = (..., g, norm, mu, s)`` (Chalmers, 2012) are the same distribution of
    :math:`g`, but both maximize the posterior over :math:`\lambda=\operatorname{logit} g`, which
    drops the :math:`-\log x - \log(1 - x)` term from the objective. ``mode_scale="logit"``
    evaluates that objective term instead,

    .. math::
        \log\tilde\pi(x) = -\log s - \frac{1}{2}\log(2\pi)
                     - \frac{(\operatorname{logit} x - \mu)^2}{2s^2},
        \qquad 0 < x < 1,

    the normal log density of :math:`\lambda` at :math:`\operatorname{logit} x` (not a density of
    :math:`x`), so the MAP estimate is the logit-metric posterior mode of flexMIRT and mirt and
    ``log_posterior`` equals theirs (DESIGN.md §15.48). The default ``"natural"`` gives the
    natural-scale posterior mode of the proper density, as for every other prior (DESIGN.md §4.7).

    Parameters
    ----------
    mean_logit : float
        Mean :math:`\mu` of :math:`\operatorname{logit} X` (finite).
    sd_logit : float
        Standard deviation :math:`s` of :math:`\operatorname{logit} X` (positive and finite).
    mode_scale : {"natural", "logit"}, default "natural"
        Scale on which the posterior mode is taken: ``"natural"`` includes the logit Jacobian
        (density of :math:`X`), ``"logit"`` omits it (flexMIRT / mirt ``norm`` on ``g``).
    """

    mean_logit: float
    sd_logit: float
    mode_scale: LogitModeScale

    def __init__(
        self,
        mean_logit: RealScalar,
        sd_logit: RealScalar,
        *,
        mode_scale: LogitModeScale = "natural",
    ) -> None:
        """Validate prior hyperparameters and store them as Python floats."""
        object.__setattr__(
            self, "mean_logit", _validate_finite(mean_logit, "LogitNormalPrior mean_logit")
        )
        object.__setattr__(
            self, "sd_logit", _validate_positive(sd_logit, "LogitNormalPrior sd_logit")
        )
        if not isinstance(mode_scale, str) or mode_scale not in _LOGIT_MODE_SCALES:
            raise SpecificationError(
                f"LogitNormalPrior mode_scale must be one of {list(_LOGIT_MODE_SCALES)}, "
                f"got {mode_scale!r}."
            )
        object.__setattr__(self, "mode_scale", mode_scale)

    @property
    def support(self) -> tuple[float, float]:
        """Open support :math:`(0, 1)`."""
        return (0.0, 1.0)

    def log_pdf(self, x: ArrayLike) -> Array:
        r"""Evaluate the log prior term at ``x`` (``-inf`` outside ``(0, 1)``).

        For ``mode_scale="natural"`` this is the log density of :math:`X`; for ``"logit"`` it is
        :math:`\log\tilde\pi(x)`, the log density of :math:`\operatorname{logit} X` at
        :math:`\operatorname{logit} x` (no Jacobian).
        """
        arr = _as_float_array(x)
        active = arr > -jnp.inf
        return _logitnormal_log_pdf(
            arr,
            jnp.asarray(self.mean_logit, dtype=arr.dtype),
            jnp.asarray(self.sd_logit, dtype=arr.dtype),
            active,
            active & (self.mode_scale == "natural"),
        )


def _logitnormal_log_pdf(
    x: Array, mean: Array, sd: Array, active: Array, with_jacobian: Array
) -> Array:
    r"""Logit-normal log prior where ``active`` (0 elsewhere), ``-inf`` outside ``(0, 1)``.

    :math:`\log\phi(\operatorname{logit}x;\mu,s) - [\log x + \log(1-x)]`, the bracket only where
    ``with_jacobian`` (DESIGN.md §15.48).
    """
    valid = active & (x > 0.0) & (x < 1.0)
    safe_x = jnp.where(valid, x, 0.5)
    safe_sd = jnp.where(active, sd, 1.0)
    log_x = jnp.log(safe_x)
    log_1mx = jnp.log1p(-safe_x)
    z = (log_x - log_1mx - mean) / safe_sd
    jacobian = jnp.where(with_jacobian, -log_x - log_1mx, 0.0)
    val = jacobian - jnp.log(safe_sd) - _HALF_LOG_2PI - 0.5 * z * z
    return jnp.where(valid, val, jnp.where(active, -jnp.inf, 0.0))


Prior = NormalPrior | LogNormalPrior | BetaPrior | LogitNormalPrior
PRIOR_TYPES: tuple[
    type[NormalPrior], type[LogNormalPrior], type[BetaPrior], type[LogitNormalPrior]
] = (
    NormalPrior,
    LogNormalPrior,
    BetaPrior,
    LogitNormalPrior,
)


def encode_prior(prior: Prior | None) -> tuple[int, float, float]:
    """Encode a prior object as ``(kind, param1, param2)`` for the Core layer."""
    if prior is None:
        return (PRIOR_NONE, 0.0, 1.0)
    if isinstance(prior, NormalPrior):
        return (PRIOR_NORMAL, float(prior.mean), float(prior.sd))
    if isinstance(prior, LogNormalPrior):
        return (PRIOR_LOGNORMAL, float(prior.mean_log), float(prior.sd_log))
    if isinstance(prior, BetaPrior):
        return (PRIOR_BETA, float(prior.a), float(prior.b))
    if isinstance(prior, LogitNormalPrior):
        kind = PRIOR_LOGITNORMAL if prior.mode_scale == "natural" else PRIOR_LOGITNORMAL_LOGIT
        return (kind, float(prior.mean_logit), float(prior.sd_logit))
    raise SpecificationError(
        f"Unsupported prior distribution {prior!r}; "
        "use NormalPrior, LogNormalPrior, BetaPrior, LogitNormalPrior or None."
    )


def eval_log_prior(x: Array, kind: Array, param1: Array, param2: Array) -> Array:
    r"""Branch-free elementwise log-prior evaluation for jit (DESIGN.md §4.7).

    Uses the double-``where`` pattern on every branch so that unselected branches and unused
    padding slots (``kind == PRIOR_NONE``) never evaluate :math:`\log 0` or negative logs and
    contribute zero value, gradient and Hessian. Out-of-support values for an active
    :class:`LogNormalPrior`, :class:`BetaPrior` or :class:`LogitNormalPrior` prior return
    :math:`-\infty` (rejected by the M-step line search and SQUAREM safeguard).
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
        (a_b - 1.0) * jnp.log(safe_x_b) + (b_b - 1.0) * jnp.log1p(-safe_x_b) - _log_beta(a_b, b_b)
    )
    lp_beta = jnp.where(valid_beta, lp_beta_val, jnp.where(is_beta, -jnp.inf, 0.0))

    is_logitn_nat = kind == PRIOR_LOGITNORMAL
    is_logitn = is_logitn_nat | (kind == PRIOR_LOGITNORMAL_LOGIT)
    lp_logitn = _logitnormal_log_pdf(x, param1, param2, is_logitn, is_logitn_nat)

    return lp_norm + lp_ln + lp_beta + lp_logitn
