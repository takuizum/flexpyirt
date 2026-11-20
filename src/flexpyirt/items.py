"""Item response models (Spec layer) and their log-probability kernels (Core layer).

An *item model* (``TwoPL``, ``Rasch``, ``GRM``) is what the user writes in a specification. It owns
the default free/fixed pattern and starting values. Its *kernel* holds only the pure
log-probability function used inside jit. Item models that share a kernel (``Rasch`` and
``TwoPL``; ``GRM`` with the same number of categories) are estimated together as one "family"
(DESIGN.md §4.3, §6).
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
    "GRM",
    "DichotomousKernel",
    "GradedKernel",
    "ItemKernel",
    "ItemModel",
    "Rasch",
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

    def log_prob(self, theta: Array, params: Array) -> Array:
        """Return ``(Q, n_cat)`` log-probabilities for nodes ``theta`` of shape ``(Q,)``."""
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

    def log_prob(self, theta: Array, params: Array) -> Array:
        """Return ``(Q, 2)`` log-probabilities of responses 0 and 1."""
        z = params[0] * theta + params[1]
        return jnp.stack([jax.nn.log_sigmoid(-z), jax.nn.log_sigmoid(z)], axis=-1)


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

    def log_prob(self, theta: Array, params: Array) -> Array:
        """Return ``(Q, n_cat)`` category log-probabilities."""
        z = params[0] * theta[:, None] + params[None, 1:]  # (Q, K-1)
        log_cum = jax.nn.log_sigmoid(z)  # log P(y >= k), k = 1..K-1
        first = jax.nn.log_sigmoid(-z[:, :1])
        last = log_cum[:, -1:]
        middle = log_cum[:, :-1] + _log1mexp(log_cum[:, 1:] - log_cum[:, :-1])
        return jnp.concatenate([first, middle, last], axis=1)


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
