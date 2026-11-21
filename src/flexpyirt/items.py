"""Item response models (Spec layer) and their log-probability kernels (Core layer).

An *item model* (``TwoPL``, ``Rasch``, ``ThreePL``, ``FourPL``, ``GRM``, ``GPCM``, ``PCM``) is
what the user writes in a specification. It owns the default free/fixed pattern and starting
values. Its *kernel* holds only the pure log-probability function used inside jit. Item models
that share a kernel (``Rasch`` and ``TwoPL``; ``ThreePL`` and ``FourPL``; ``GRM`` with the same
number of categories; ``GPCM`` and ``PCM`` with the same number of categories) are estimated
together as one "family" (DESIGN.md §4.3, §6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from flexpyirt.errors import SpecificationError

__all__ = [
    "GPCM",
    "GRM",
    "PCM",
    "DichotomousKernel",
    "FourPL",
    "FourPLKernel",
    "GPCMKernel",
    "GradedKernel",
    "ItemKernel",
    "ItemModel",
    "Rasch",
    "ThreePL",
    "TwoPL",
]


class ItemKernel(Protocol):
    """Pure log-probability function of an item family (Core layer).

    Implementations must be hashable (they are static arguments of jit).
    """

    @property
    def param_names(self) -> tuple[str, ...]:
        """Internal parameter names, in order."""
        ...

    @property
    def n_cat(self) -> int:
        """Number of response categories (coded ``0..n_cat-1``)."""
        ...

    @property
    def slope_params(self) -> tuple[str, ...]:
        """Parameters that carry the latent scale (used by identification checks)."""
        ...

    @property
    def intercept_params(self) -> tuple[str, ...]:
        """Parameters that carry the latent location (used by identification checks)."""
        ...

    @property
    def irt_names(self) -> tuple[str, ...]:
        """Display IRT threshold/difficulty names, in order (e.g. ``("b",)``)."""
        ...

    @property
    def irt_dep_indices(self) -> tuple[tuple[int, ...], ...]:
        """Indices into ``param_names`` that enter each display threshold in ``irt_names``."""
        ...

    def irt_thresholds(self, params: Array) -> Array:
        """Return display IRT thresholds/difficulties of shape ``(len(irt_names),)``."""
        ...

    def log_prob(self, theta: Array, params: Array) -> Array:
        """Return ``(Q, n_cat)`` log-probabilities for nodes ``theta`` of shape ``(Q,)``."""
        ...

    def affine_transform(self, params: Array, mean: Array, sd: Array) -> Array:
        r"""Transform ``params`` under :math:`\theta^*=(\theta-m)/s`."""
        ...

    def __hash__(self) -> int:
        """Kernels must be hashable (static arguments of jit)."""
        ...


@dataclass(frozen=True)
class DichotomousKernel:
    r"""Two-parameter logistic kernel in slope–intercept form.

    .. math::
        P(y=1\mid\theta) = \sigma(a\theta + d), \qquad
        \log P(y=1\mid\theta) = \log\sigma(a\theta+d),\quad
        \log P(y=0\mid\theta) = \log\sigma(-(a\theta+d)).

    Birnbaum (1968); slope–intercept parameterization as in mirt (Chalmers, 2012).
    """

    @property
    def param_names(self) -> tuple[str, ...]:
        """``("a", "d")``."""
        return ("a", "d")

    @property
    def n_cat(self) -> int:
        """Two categories."""
        return 2

    @property
    def slope_params(self) -> tuple[str, ...]:
        """``("a",)``."""
        return ("a",)

    @property
    def intercept_params(self) -> tuple[str, ...]:
        """``("d",)``."""
        return ("d",)

    @property
    def irt_names(self) -> tuple[str, ...]:
        """``("b",)``."""
        return ("b",)

    @property
    def irt_dep_indices(self) -> tuple[tuple[int, ...], ...]:
        """``((0, 1),)`` — ``b`` depends on ``(a, d)``."""
        return ((0, 1),)

    def irt_thresholds(self, params: Array) -> Array:
        r"""Return IRT difficulty :math:`b = -d / a` (Birnbaum, 1968)."""
        return jnp.array([-params[1] / params[0]])

    def log_prob(self, theta: Array, params: Array) -> Array:
        """Return ``(Q, 2)`` log-probabilities of responses 0 and 1."""
        z = params[0] * theta + params[1]
        return jnp.stack([jax.nn.log_sigmoid(-z), jax.nn.log_sigmoid(z)], axis=-1)

    def affine_transform(self, params: Array, mean: Array, sd: Array) -> Array:
        r"""Return :math:`(a s,\; d + a m)` under :math:`\theta^* = (\theta - m) / s`."""
        return jnp.stack([params[0] * sd, params[1] + params[0] * mean])


@dataclass(frozen=True)
class FourPLKernel:
    r"""Four-parameter logistic kernel in slope–intercept form, shared by 3PL and 4PL items.

    .. math::
        P(y=1\mid\theta) = g + (u-g)\,\sigma(a\theta+d),\qquad 0\le g<u\le 1,

    computed in log space as (DESIGN.md §15.3)

    .. math::
        \log P(y=1) &= \operatorname{logaddexp}\big(\log g,\ \log(u-g)+\log\sigma(z)\big),\\
        \log P(y=0) &= \operatorname{logaddexp}\big(\log(1-u),\ \log(u-g)+\log\sigma(-z)\big),
        \qquad z=a\theta+d.

    At :math:`g=0` the first log-probability is :math:`\log(u-g)+\log\sigma(z)` and at :math:`u=1`
    the second is :math:`\log(u-g)+\log\sigma(-z)`, selected with a double ``where`` so that
    neither :math:`\log 0` nor its infinite derivative is ever evaluated: with :math:`g=0`,
    :math:`u=1` the kernel returns exactly the 2PL log-probabilities, and gradients and Hessians
    with respect to every parameter (including the fixed boundary ones) are finite. A candidate
    with :math:`u\le g` gives NaN, which the M-step line search rejects.

    Birnbaum (1968) for the 3PL; Barton & Lord (1981) for the upper asymptote; slope–intercept
    parameterization and parameter names as in mirt (Chalmers, 2012).
    """

    @property
    def param_names(self) -> tuple[str, ...]:
        """``("a", "d", "g", "u")``."""
        return ("a", "d", "g", "u")

    @property
    def n_cat(self) -> int:
        """Two categories."""
        return 2

    @property
    def slope_params(self) -> tuple[str, ...]:
        """``("a",)``."""
        return ("a",)

    @property
    def intercept_params(self) -> tuple[str, ...]:
        """``("d",)``; the asymptotes ``g`` and ``u`` carry neither scale nor location."""
        return ("d",)

    @property
    def irt_names(self) -> tuple[str, ...]:
        """``("b",)``."""
        return ("b",)

    @property
    def irt_dep_indices(self) -> tuple[tuple[int, ...], ...]:
        """``((0, 1),)`` — ``b`` depends on ``(a, d)`` only."""
        return ((0, 1),)

    def irt_thresholds(self, params: Array) -> Array:
        r"""Return IRT difficulty :math:`b = -d / a` (Birnbaum, 1968; Barton & Lord, 1981)."""
        return jnp.array([-params[1] / params[0]])

    def log_prob(self, theta: Array, params: Array) -> Array:
        """Return ``(Q, 2)`` log-probabilities of responses 0 and 1."""
        a, d, g, u = params[0], params[1], params[2], params[3]
        z = a * theta + d
        log_span = jnp.log(u - g)
        core1 = log_span + jax.nn.log_sigmoid(z)
        core0 = log_span + jax.nn.log_sigmoid(-z)
        # Double where: the log's argument is replaced by 1 where the asymptote term vanishes, so
        # the unselected branch is finite and contributes a zero (not NaN) gradient.
        has_g = g > 0
        has_rest = u < 1
        log_g = jnp.log(jnp.where(has_g, g, 1.0))
        log_rest = jnp.log(jnp.where(has_rest, 1.0 - u, 1.0))
        log_p1 = jnp.where(has_g, jnp.logaddexp(log_g, core1), core1)
        log_p0 = jnp.where(has_rest, jnp.logaddexp(log_rest, core0), core0)
        valid = (g >= 0) & (u <= 1) & (u > g)
        out = jnp.stack([log_p0, log_p1], axis=-1)
        return jnp.where(valid, out, jnp.nan)

    def affine_transform(self, params: Array, mean: Array, sd: Array) -> Array:
        r"""Return :math:`(a s,\; d + a m,\; g,\; u)` under :math:`\theta^* = (\theta - m) / s`."""
        return jnp.stack([params[0] * sd, params[1] + params[0] * mean, params[2], params[3]])


def _log1mexp(x: Array) -> Array:
    r"""Return :math:`\log(1-e^{x})` for :math:`x<0`, stably and with NaN-free gradients.

    Mächler (2012): use :math:`\log(-\mathrm{expm1}(x))` for :math:`x>-\log 2` and
    :math:`\mathrm{log1p}(-e^x)` otherwise. Each branch gets an input clamped to its own safe
    domain so the unused branch never produces inf/NaN (whose gradient would leak through
    ``where``). For :math:`x\ge 0` the result is NaN, which the M-step line search rejects.
    """
    near = x > -jnp.log(2.0)
    x_near = jnp.where(near, x, -1.0)
    x_far = jnp.where(near, -1.0, x)
    out = jnp.where(near, jnp.log(-jnp.expm1(x_near)), jnp.log1p(-jnp.exp(x_far)))
    return jnp.where(x < 0, out, jnp.nan)


@dataclass(frozen=True)
class GradedKernel:
    r"""Graded response kernel (Samejima, 1969) in slope–intercept form.

    .. math::
        P(y\ge k\mid\theta)=\sigma(a\theta+d_k),\quad k=1,\dots,K-1,\qquad d_1>\dots>d_{K-1},

        P(y=k\mid\theta)=P(y\ge k\mid\theta)-P(y\ge k+1\mid\theta).

    Log-probabilities: :math:`\log\sigma(-z_1)` for :math:`k=0`, :math:`\log\sigma(z_{K-1})` for
    :math:`k=K-1`, and
    :math:`\log\sigma(z_k)+\mathrm{log1mexp}(\log\sigma(z_{k+1})-\log\sigma(z_k))` in between
    (DESIGN.md §14.7). Disordered intercepts give NaN.
    """

    n_cat: int

    def __post_init__(self) -> None:
        """Validate the number of categories."""
        if self.n_cat < 2:
            raise SpecificationError(f"GRM needs at least 2 categories, got {self.n_cat}.")

    @property
    def param_names(self) -> tuple[str, ...]:
        """``("a", "d1", ..., "d{K-1}")``."""
        return ("a", *(f"d{k}" for k in range(1, self.n_cat)))

    @property
    def slope_params(self) -> tuple[str, ...]:
        """``("a",)``."""
        return ("a",)

    @property
    def intercept_params(self) -> tuple[str, ...]:
        """``("d1", ..., "d{K-1}")``."""
        return self.param_names[1:]

    @property
    def irt_names(self) -> tuple[str, ...]:
        """``("b1", ..., "b{K-1}")``."""
        return tuple(f"b{k}" for k in range(1, self.n_cat))

    @property
    def irt_dep_indices(self) -> tuple[tuple[int, ...], ...]:
        """Each ``bk`` depends on ``(a, dk)``."""
        return tuple((0, k) for k in range(1, self.n_cat))

    def irt_thresholds(self, params: Array) -> Array:
        r"""Return IRT thresholds :math:`b_k = -d_k / a` (Samejima, 1969)."""
        return -params[1:] / params[0]

    def log_prob(self, theta: Array, params: Array) -> Array:
        """Return ``(Q, n_cat)`` category log-probabilities."""
        z = params[0] * theta[:, None] + params[None, 1:]  # (Q, K-1)
        log_cum = jax.nn.log_sigmoid(z)  # log P(y >= k), k = 1..K-1
        first = jax.nn.log_sigmoid(-z[:, :1])
        last = log_cum[:, -1:]
        middle = log_cum[:, :-1] + _log1mexp(log_cum[:, 1:] - log_cum[:, :-1])
        return jnp.concatenate([first, middle, last], axis=1)

    def affine_transform(self, params: Array, mean: Array, sd: Array) -> Array:
        r"""Return :math:`(a s,\; d_1 + a m, \dots, d_{K-1} + a m)`."""
        return jnp.concatenate([params[:1] * sd, params[1:] + params[0] * mean])


@dataclass(frozen=True)
class GPCMKernel:
    r"""Generalized partial credit kernel (Muraki, 1992) in slope–intercept form.

    .. math::
        P(y=k\mid\theta) = \frac{\exp(k\,a\theta + d_k)}{\sum_{c=0}^{K-1}\exp(c\,a\theta + d_c)},
        \qquad k=0,\dots,K-1,\quad d_0=0,

    with fixed scoring coefficients :math:`\mathrm{ak}_k=k` and :math:`d_0=0` (not a parameter),
    matching mirt's ``itemtype="gpcm"`` (Chalmers, 2012). Computed with ``jax.nn.log_softmax``;
    there is no ordering constraint on :math:`d_1,\dots,d_{K-1}`. Shared by :class:`GPCM` and
    :class:`PCM` items with the same ``n_cat`` (DESIGN.md §15.8).
    """

    n_cat: int

    def __post_init__(self) -> None:
        """Validate the number of categories."""
        if self.n_cat < 2:
            raise SpecificationError(f"GPCM needs at least 2 categories, got {self.n_cat}.")

    @property
    def param_names(self) -> tuple[str, ...]:
        """``("a", "d1", ..., "d{K-1}")``."""
        return ("a", *(f"d{k}" for k in range(1, self.n_cat)))

    @property
    def slope_params(self) -> tuple[str, ...]:
        """``("a",)``."""
        return ("a",)

    @property
    def intercept_params(self) -> tuple[str, ...]:
        """``("d1", ..., "d{K-1}")``."""
        return self.param_names[1:]

    @property
    def irt_names(self) -> tuple[str, ...]:
        """``("b1", ..., "b{K-1}")``."""
        return tuple(f"b{k}" for k in range(1, self.n_cat))

    @property
    def irt_dep_indices(self) -> tuple[tuple[int, ...], ...]:
        r"""``b1`` depends on ``(a, d1)``; ``bk`` (:math:`k\ge 2`) on ``(a, d{k-1}, dk)``."""
        return ((0, 1), *((0, k - 1, k) for k in range(2, self.n_cat)))

    def irt_thresholds(self, params: Array) -> Array:
        r"""Return step difficulties :math:`b_k = -(d_k - d_{k-1}) / a` with :math:`d_0 = 0`.

        Muraki (1992); Masters (1982).
        """
        intercepts = jnp.concatenate([jnp.zeros((1,), dtype=params.dtype), params[1:]])
        return -jnp.diff(intercepts) / params[0]

    def log_prob(self, theta: Array, params: Array) -> Array:
        """Return ``(Q, n_cat)`` category log-probabilities."""
        categories = jnp.arange(self.n_cat, dtype=theta.dtype)
        intercepts = jnp.concatenate([jnp.zeros((1,), dtype=params.dtype), params[1:]])
        eta = params[0] * theta[:, None] * categories[None, :] + intercepts[None, :]
        return jax.nn.log_softmax(eta, axis=-1)

    def affine_transform(self, params: Array, mean: Array, sd: Array) -> Array:
        r"""Return :math:`(a s,\; d_1 + a m, \dots, d_{K-1} + (K-1) a m)`."""
        ks = jnp.arange(1, self.n_cat, dtype=params.dtype)
        return jnp.concatenate([params[:1] * sd, params[1:] + ks * params[0] * mean])


class ItemModel(Protocol):
    """An item model as used in a specification (Spec layer)."""

    name: ClassVar[str]

    @property
    def kernel(self) -> ItemKernel:
        """Log-probability kernel; items with equal kernels are estimated as one family."""
        ...

    def default_free(self) -> tuple[bool, ...]:
        """Whether each kernel parameter is estimated by default."""
        ...

    def start_values(self, item: np.ndarray, rest: np.ndarray) -> np.ndarray:
        """Classical starting values from one item's observed responses and their rest scores.

        ``item`` and ``rest`` have the same length and contain observed responses only.
        """
        ...

    def bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Lower and upper bounds of each kernel parameter."""
        ...


def _slope_start(item: np.ndarray, rest: np.ndarray) -> float:
    r"""Return the starting slope from the item–rest correlation (DESIGN.md §4.10).

    :math:`a^{(0)}=\mathrm{clip}(1.7\,r/\sqrt{1-r^2},0.25,3)`.
    """
    if item.std() == 0 or rest.std() == 0:
        return 1.0
    r = float(np.clip(np.corrcoef(item, rest)[0, 1], -0.95, 0.95))
    return float(np.clip(1.7 * r / np.sqrt(1.0 - r * r), 0.25, 3.0))


def _intercept_start(p: float, a: float) -> float:
    r"""Return :math:`\mathrm{logit}(p)\sqrt{1+(a/1.7)^2}`, :math:`p` clipped to [0.01, 0.99]."""
    p = float(np.clip(p, 0.01, 0.99))
    return float(np.log(p / (1.0 - p)) * np.sqrt(1.0 + (a / 1.7) ** 2))


def _unbounded(n: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    return (-np.inf,) * n, (np.inf,) * n


@dataclass(frozen=True)
class TwoPL:
    r"""Two-parameter logistic model, :math:`P(y=1\mid\theta)=\sigma(a\theta+d)`.

    Both the slope :math:`a` and the intercept :math:`d` are estimated. The IRT difficulty is
    :math:`b=-d/a`.
    """

    name: ClassVar[str] = "2PL"

    @property
    def kernel(self) -> DichotomousKernel:
        """Shared dichotomous kernel."""
        return DichotomousKernel()

    def default_free(self) -> tuple[bool, ...]:
        """Slope and intercept are free."""
        return (True, True)

    def start_values(self, item: np.ndarray, rest: np.ndarray) -> np.ndarray:
        """Return starting values (DESIGN.md §4.10)."""
        a = _slope_start(item, rest)
        return np.array([a, _intercept_start(item.mean(), a)])

    def bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Unbounded."""
        return _unbounded(2)


@dataclass(frozen=True)
class Rasch:
    r"""Rasch model, :math:`P(y=1\mid\theta)=\sigma(\theta+d)` (Rasch, 1960).

    The slope is fixed at 1 and the scale is carried by the latent variance, which is estimated
    by default in a single-group model (DESIGN.md §4.5), as in mirt's ``itemtype="Rasch"``.
    """

    name: ClassVar[str] = "Rasch"

    @property
    def kernel(self) -> DichotomousKernel:
        """Shared dichotomous kernel."""
        return DichotomousKernel()

    def default_free(self) -> tuple[bool, ...]:
        """Slope fixed (at 1), intercept free."""
        return (False, True)

    def start_values(self, item: np.ndarray, rest: np.ndarray) -> np.ndarray:
        """Return starting values; the slope is 1."""
        return np.array([1.0, _intercept_start(item.mean(), 1.0)])

    def bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Unbounded."""
        return _unbounded(2)


def _asymptote_starts(p: float, *, free_upper: bool) -> tuple[float, float]:
    r"""Return starting :math:`(g^{(0)}, u^{(0)})` for proportion correct :math:`p` (DESIGN §15.4).

    :math:`g^{(0)}=\min(0.2,\,p/2)` and :math:`u^{(0)}=\max(0.95,\,(1+p)/2)` (4PL) or 1 (3PL), so
    that :math:`g^{(0)}<p<u^{(0)}` with :math:`p` clipped to :math:`[0.01, 0.99]`.
    """
    p = float(np.clip(p, 0.01, 0.99))
    g = min(0.2, 0.5 * p)
    u = max(0.95, 0.5 * (1.0 + p)) if free_upper else 1.0
    return g, u


def _guessing_start_values(item: np.ndarray, rest: np.ndarray, *, free_upper: bool) -> np.ndarray:
    r"""Return starting :math:`(a, d, g, u)` for 3PL / 4PL items (DESIGN.md §4.10, §15.4).

    :math:`a^{(0)}` as for the 2PL; :math:`d^{(0)}` as for the 2PL but from the proportion
    correct rescaled to the span of the asymptotes, :math:`(p-g^{(0)})/(u^{(0)}-g^{(0)})`.
    """
    p = float(item.mean())
    g, u = _asymptote_starts(p, free_upper=free_upper)
    a = _slope_start(item, rest)
    return np.array([a, _intercept_start((p - g) / (u - g), a), g, u])


_ASYMPTOTE_BOUNDS = ((-np.inf, -np.inf, 0.0, 0.0), (np.inf, np.inf, 1.0, 1.0))


@dataclass(frozen=True)
class ThreePL:
    r"""Three-parameter logistic model, :math:`P(y=1\mid\theta)=g+(1-g)\,\sigma(a\theta+d)`.

    Birnbaum (1968). The slope :math:`a`, intercept :math:`d` and lower asymptote ("guessing")
    :math:`g` are estimated on their natural scale with :math:`0<g<1`; the upper asymptote
    :math:`u` is fixed at 1. Same model and parameter names as mirt's ``itemtype="3PL"`` (mirt
    estimates :math:`g` on the logit scale; the maximum is the same). Shares
    :class:`FourPLKernel` with :class:`FourPL`. The IRT difficulty is :math:`b=-d/a`.
    """

    name: ClassVar[str] = "3PL"

    @property
    def kernel(self) -> FourPLKernel:
        """Shared four-parameter kernel."""
        return FourPLKernel()

    def default_free(self) -> tuple[bool, ...]:
        """Return free flags: ``a``, ``d``, ``g`` free; ``u`` fixed (at 1)."""
        return (True, True, True, False)

    def start_values(self, item: np.ndarray, rest: np.ndarray) -> np.ndarray:
        """Return starting values (DESIGN.md §15.4); ``u`` is 1."""
        return _guessing_start_values(item, rest, free_upper=False)

    def bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return parameter bounds: ``a``, ``d`` unbounded; ``g``, ``u`` in ``[0, 1]``."""
        return _ASYMPTOTE_BOUNDS


@dataclass(frozen=True)
class FourPL:
    r"""Four-parameter logistic model, :math:`P(y=1\mid\theta)=g+(u-g)\,\sigma(a\theta+d)`.

    Barton & Lord (1981). All of :math:`a, d, g, u` are estimated on their natural scale with
    :math:`0<g<1`, :math:`0<u<1` (and :math:`g<u`). Same model and parameter names as mirt's
    ``itemtype="4PL"``. Shares :class:`FourPLKernel` with :class:`ThreePL`.
    """

    name: ClassVar[str] = "4PL"

    @property
    def kernel(self) -> FourPLKernel:
        """Shared four-parameter kernel."""
        return FourPLKernel()

    def default_free(self) -> tuple[bool, ...]:
        """All four parameters are free."""
        return (True, True, True, True)

    def start_values(self, item: np.ndarray, rest: np.ndarray) -> np.ndarray:
        """Return starting values (DESIGN.md §15.4)."""
        return _guessing_start_values(item, rest, free_upper=True)

    def bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return parameter bounds: ``a``, ``d`` unbounded; ``g``, ``u`` in ``[0, 1]``."""
        return _ASYMPTOTE_BOUNDS


@dataclass(frozen=True)
class GRM:
    r"""Graded response model (Samejima, 1969) with ``n_cat`` ordered categories ``0..n_cat-1``.

    :math:`P(y\ge k\mid\theta)=\sigma(a\theta+d_k)` with :math:`d_1>\dots>d_{K-1}`; the IRT
    thresholds are :math:`b_k=-d_k/a`. Same parameterization as mirt's ``itemtype="graded"``.
    """

    n_cat: int
    name: ClassVar[str] = "GRM"

    def __post_init__(self) -> None:
        """Validate the number of categories."""
        if self.n_cat < 2:
            raise SpecificationError(f"GRM needs at least 2 categories, got {self.n_cat}.")

    @property
    def kernel(self) -> GradedKernel:
        """Graded kernel with this number of categories."""
        return GradedKernel(self.n_cat)

    def default_free(self) -> tuple[bool, ...]:
        """All parameters are free."""
        return (True,) * self.n_cat

    def start_values(self, item: np.ndarray, rest: np.ndarray) -> np.ndarray:
        r"""Return starting values (DESIGN.md §4.10).

        :math:`a^{(0)}` from the item–rest correlation and
        :math:`d_k^{(0)}=\mathrm{logit}(P(y\ge k))\sqrt{1+(a^{(0)}/1.7)^2}`.
        """
        a = _slope_start(item, rest)
        cum = [float((item >= k).mean()) for k in range(1, self.n_cat)]
        d = [_intercept_start(p, a) for p in cum]
        # keep strictly decreasing even after clipping
        for k in range(1, len(d)):
            d[k] = min(d[k], d[k - 1] - 0.05)
        return np.array([a, *d])

    def bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Unbounded (ordering is enforced by the line search)."""
        return _unbounded(self.n_cat)


def _gpcm_intercept_starts(item: np.ndarray, n_cat: int, a: float) -> list[float]:
    r"""Return starting :math:`d_1,\dots,d_{K-1}` for GPCM/PCM (DESIGN.md §4.10, §15.9).

    Each step :math:`\delta_c^{(0)}=\mathrm{logit}(p_c)\sqrt{1+(a/1.7)^2}` uses the cumulative
    proportion :math:`p_c=\hat P(y\ge c)` (clipped to :math:`[0.01, 0.99]`), and
    :math:`d_k^{(0)}=\sum_{c=1}^{k}\delta_c^{(0)}`. Reduces to :func:`_intercept_start` for
    ``n_cat == 2``.
    """
    steps = [_intercept_start(float((item >= c).mean()), a) for c in range(1, n_cat)]
    return [float(v) for v in np.cumsum(steps)]


@dataclass(frozen=True)
class GPCM:
    r"""Generalized partial credit model (Muraki, 1992) with ``n_cat`` categories ``0..n_cat-1``.

    .. math::
        P(y=k\mid\theta) = \frac{\exp(k\,a\theta + d_k)}{\sum_{c=0}^{K-1}\exp(c\,a\theta + d_c)},
        \qquad d_0 = 0.

    Slope :math:`a` and intercepts :math:`d_1,\dots,d_{K-1}` are estimated with no ordering
    constraint; step difficulties displayed in :attr:`FitResult.items` are
    :math:`b_k=-(d_k-d_{k-1})/a` with :math:`d_0=0`. Same parameterization as mirt's
    ``itemtype="gpcm"`` (Chalmers, 2012). With ``n_cat=2`` this is identical to :class:`TwoPL`.
    Shares :class:`GPCMKernel` with :class:`PCM` of the same ``n_cat``.
    """

    n_cat: int
    name: ClassVar[str] = "GPCM"

    def __post_init__(self) -> None:
        """Validate the number of categories."""
        if self.n_cat < 2:
            raise SpecificationError(f"GPCM needs at least 2 categories, got {self.n_cat}.")

    @property
    def kernel(self) -> GPCMKernel:
        """GPCM kernel with this number of categories."""
        return GPCMKernel(self.n_cat)

    def default_free(self) -> tuple[bool, ...]:
        """All parameters are free."""
        return (True,) * self.n_cat

    def start_values(self, item: np.ndarray, rest: np.ndarray) -> np.ndarray:
        """Return starting values (DESIGN.md §4.10, §15.9)."""
        a = _slope_start(item, rest)
        return np.array([a, *_gpcm_intercept_starts(item, self.n_cat, a)])

    def bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Unbounded."""
        return _unbounded(self.n_cat)


@dataclass(frozen=True)
class PCM:
    r"""Partial credit model (Masters, 1982) with ``n_cat`` categories ``0..n_cat-1``.

    .. math::
        P(y=k\mid\theta) = \frac{\exp(k\theta + d_k)}{\sum_{c=0}^{K-1}\exp(c\theta + d_c)},
        \qquad d_0 = 0.

    GPCM with slope fixed at :math:`a=1`; step difficulties are :math:`b_k=-(d_k-d_{k-1})` with
    :math:`d_0=0`. In a single-group model whose items are all :class:`Rasch` or :class:`PCM`, the
    latent variance is estimated by default (DESIGN.md §4.5, §15.8), matching mirt's
    ``itemtype="Rasch"`` on polytomous data. With ``n_cat=2`` this is identical to :class:`Rasch`.
    Shares :class:`GPCMKernel` with :class:`GPCM` of the same ``n_cat``.
    """

    n_cat: int
    name: ClassVar[str] = "PCM"

    def __post_init__(self) -> None:
        """Validate the number of categories."""
        if self.n_cat < 2:
            raise SpecificationError(f"PCM needs at least 2 categories, got {self.n_cat}.")

    @property
    def kernel(self) -> GPCMKernel:
        """Shared GPCM kernel with this number of categories."""
        return GPCMKernel(self.n_cat)

    def default_free(self) -> tuple[bool, ...]:
        """Slope fixed (at 1), intercepts free."""
        return (False, *(True,) * (self.n_cat - 1))

    def start_values(self, item: np.ndarray, rest: np.ndarray) -> np.ndarray:
        """Return starting values (DESIGN.md §4.10, §15.9); the slope is 1."""
        return np.array([1.0, *_gpcm_intercept_starts(item, self.n_cat, 1.0)])

    def bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Unbounded."""
        return _unbounded(self.n_cat)
