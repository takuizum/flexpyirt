"""Item response models (Spec layer) and their log-probability kernels (Core layer).

An *item model* (``TwoPL``, ``Rasch``) is what the user writes in a specification. It owns the
default free/fixed pattern and starting values. Its *kernel* holds only the pure log-probability
function used inside jit. Item models that share a kernel (``Rasch`` and ``TwoPL``) are estimated
together as one "family" (DESIGN.md §4.3, §6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

__all__ = ["DichotomousKernel", "ItemKernel", "ItemModel", "Rasch", "TwoPL"]


class ItemKernel(Protocol):
    """Pure log-probability function of an item family (Core layer).

    Implementations must be hashable (they are static arguments of jit).
    """

    param_names: ClassVar[tuple[str, ...]]
    n_cat: ClassVar[int]

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

    param_names: ClassVar[tuple[str, ...]] = ("a", "d")
    n_cat: ClassVar[int] = 2

    def log_prob(self, theta: Array, params: Array) -> Array:
        """Return ``(Q, 2)`` log-probabilities of responses 0 and 1."""
        z = params[0] * theta + params[1]
        return jnp.stack([jax.nn.log_sigmoid(-z), jax.nn.log_sigmoid(z)], axis=-1)


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

    def start_values(self, y: np.ndarray) -> np.ndarray:
        """Classical starting values from one item's responses and the rest score.

        ``y`` has shape ``(N, 2)``: column 0 is the item response (0/1), column 1 the rest score.
        """
        ...

    def bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Lower and upper bounds of each kernel parameter."""
        ...


def _dichotomous_start(y: np.ndarray, *, slope: float | None) -> np.ndarray:
    r"""Return starting values for 2PL / Rasch (DESIGN.md §4.10).

    :math:`a^{(0)}=\mathrm{clip}(1.7\,r/\sqrt{1-r^2},0.25,3)`,
    :math:`d^{(0)}=\mathrm{logit}(p)\sqrt{1+(a^{(0)}/1.7)^2}`, with :math:`r` the item–rest
    point-biserial correlation and :math:`p` the proportion correct clipped to ``[0.01, 0.99]``.
    """
    item, rest = y[:, 0].astype(float), y[:, 1].astype(float)
    p = float(np.clip(item.mean(), 0.01, 0.99))
    if slope is None:
        sd_item, sd_rest = item.std(), rest.std()
        r = 0.0 if sd_item == 0 or sd_rest == 0 else float(np.corrcoef(item, rest)[0, 1])
        r = float(np.clip(r, -0.95, 0.95))
        a = float(np.clip(1.7 * r / np.sqrt(1.0 - r * r), 0.25, 3.0))
    else:
        a = slope
    d = float(np.log(p / (1.0 - p)) * np.sqrt(1.0 + (a / 1.7) ** 2))
    return np.array([a, d])


_UNBOUNDED = ((-np.inf, -np.inf), (np.inf, np.inf))


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

    def start_values(self, y: np.ndarray) -> np.ndarray:
        """See :func:`_dichotomous_start`."""
        return _dichotomous_start(y, slope=None)

    def bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Unbounded."""
        return _UNBOUNDED


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

    def start_values(self, y: np.ndarray) -> np.ndarray:
        """See :func:`_dichotomous_start`; the slope is 1."""
        return _dichotomous_start(y, slope=1.0)

    def bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Unbounded."""
        return _UNBOUNDED
