r"""IRT equating kernels (Core layer, DESIGN.md §20).

Pure JAX functions on the :class:`Form` pytree: the category probabilities of a test form on a
common :math:`\theta` scale, its test characteristic curve (TCC) and test information, a
Newton–bisection solver for :math:`\tau(\theta)=x`, and model-implied summed-score distributions
and EAPs on a latent grid. The equipercentile functions at the end are plain NumPy (they act on
two short probability vectors outside ``jit``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from flexpyirt.items import ItemKernel
from flexpyirt.sumscore import summed_score_distribution

__all__ = [
    "Form",
    "FormFamily",
    "equipercentile",
    "form_information",
    "form_probs",
    "percentile_rank",
    "score_distribution",
    "solve_true_score",
    "summed_score_eap",
    "test_characteristic",
]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FormFamily:
    """Items of a form that share one unidimensional :class:`ItemKernel` (Core pytree).

    Attributes
    ----------
    kernel : ItemKernel
        Static item kernel.
    params : Array, shape (J_f, P_f)
        Kernel parameters of the items, on the form's own scale.
    columns : tuple of int
        Static positions of the items in the form.
    """

    kernel: ItemKernel = field(metadata={"static": True})
    params: Array
    columns: tuple[int, ...] = field(metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Form:
    r"""A test form evaluated on a common scale (Core pytree).

    The items are evaluated at their own scale's :math:`\theta_X=(\theta-B)/A`, so
    :math:`P_{jk}(\theta)=P_{jk}((\theta-B)/A;\xi_j)` (DESIGN.md §20.2).

    Attributes
    ----------
    families : tuple of FormFamily
        Items grouped by kernel.
    n_items : int
        Static number of items.
    max_categories : int
        Static largest number of categories.
    slope : Array, shape ()
        :math:`A` of :math:`\theta=A\theta_X+B` (1 without a link).
    intercept : Array, shape ()
        :math:`B` (0 without a link).
    """

    families: tuple[FormFamily, ...]
    n_items: int = field(metadata={"static": True})
    max_categories: int = field(metadata={"static": True})
    slope: Array
    intercept: Array


def form_probs(form: Form, theta: Array) -> Array:
    r"""Return the category probabilities :math:`P_{jk}(\theta)` of every item.

    ``theta`` has shape ``(T,)`` on the common scale; the result has shape ``(T, J, K_max)``,
    items with fewer categories padded with zeros.
    """
    own = (theta - form.intercept) / form.slope
    out = jnp.zeros((theta.shape[0], form.n_items, form.max_categories), dtype=theta.dtype)
    for fam in form.families:

        def per_item(p: Array, k: ItemKernel = fam.kernel) -> Array:
            return jnp.exp(k.log_prob(own[:, None], p))  # (T, K_f)

        p_f = jax.vmap(per_item)(fam.params)  # (J_f, T, K_f)
        cols = jnp.asarray(fam.columns)
        out = out.at[:, cols, : fam.kernel.n_categories].set(jnp.swapaxes(p_f, 0, 1))
    return out


def test_characteristic(form: Form, theta: Array) -> Array:
    r"""Return the test characteristic curve.

    .. math:: \tau(\theta)=\sum_j\sum_{k=0}^{K_j-1}k\,P_{jk}(\theta)

    (Lord, 1980). ``theta`` has shape ``(T,)``; the result has shape ``(T,)``.
    """
    scores = jnp.arange(form.max_categories, dtype=theta.dtype)
    return jnp.einsum("tjk,k->t", form_probs(form, theta), scores)


# pytest would otherwise collect this function as a test when a test module imports it.
test_characteristic.__test__ = False  # type: ignore[attr-defined]


def form_information(form: Form, theta: Array) -> Array:
    r"""Return the test information of the form's items on the common scale.

    .. math::
        I(\theta)=\sum_j\sum_kP_{jk}(\theta)
            \Big(\frac{\partial\log P_{jk}(\theta)}{\partial\theta}\Big)^2

    (Birnbaum, 1968; Samejima, 1969), by forward-mode autodiff of the category log-probabilities
    as in ``_information.item_curves``. ``theta`` has shape ``(T,)``.
    """
    total = jnp.zeros(theta.shape, dtype=theta.dtype)
    for fam in form.families:

        def at_theta(t: Array, p: Array, k: ItemKernel = fam.kernel) -> Array:
            def log_prob_at(u: Array) -> Array:
                return k.log_prob(((u - form.intercept) / form.slope).reshape(1, 1), p)[0]

            log_p = log_prob_at(t)
            d_log_p = jax.jacfwd(log_prob_at)(t)
            return jnp.sum(jnp.exp(log_p) * d_log_p * d_log_p)

        def per_item(p: Array) -> Array:
            return jax.vmap(lambda t: at_theta(t, p))(theta)

        total = total + jnp.sum(jax.vmap(per_item)(fam.params), axis=0)
    return total


def solve_true_score(
    form: Form,
    targets: Array,
    lower: Array,
    upper: Array,
    *,
    score_tol: float,
    max_iter: int,
) -> tuple[Array, Array, Array]:
    r"""Solve :math:`\tau(\theta_x)=x` for every target by safeguarded Newton (Newton–bisection).

    Each target starts at the midpoint of its bracket ``[lower, upper]``, which must satisfy
    :math:`\tau(\text{lower})<x<\tau(\text{upper})`. A Newton step
    :math:`\theta-(\tau(\theta)-x)/\tau'(\theta)` that leaves the current bracket, or a
    non-positive derivative, is replaced by bisection; the bracket shrinks at every step
    (Press et al., 2007, §9.4, ``rtsafe``). :math:`\tau'` comes from ``jax.jvp``. Iteration stops
    when every :math:`|\tau(\theta)-x|\le` ``score_tol`` or every bracket is narrower than
    :math:`10^{-14}(1+|\theta|)`, or after ``max_iter`` steps.

    Returns
    -------
    theta : Array, shape (S,)
        The solutions.
    residual : Array, shape (S,)
        :math:`\tau(\theta)-x` at the solutions.
    n_iter : Array, shape ()
        Number of steps taken.
    """

    def tau(t: Array) -> Array:
        return test_characteristic(form, t)

    def width_ok(lo: Array, hi: Array, t: Array) -> Array:
        return hi - lo <= 1e-14 * (1.0 + jnp.abs(t))

    def cond(state: tuple[Array, Array, Array, Array, Array]) -> Array:
        t, lo, hi, resid, it = state
        done = (jnp.abs(resid) <= score_tol) | width_ok(lo, hi, t)
        return (it < max_iter) & ~jnp.all(done)

    def body(
        state: tuple[Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array]:
        t, lo, hi, resid, it = state
        value, slope = jax.jvp(tau, (t,), (jnp.ones_like(t),))
        resid = value - targets
        lo = jnp.where(resid < 0.0, t, lo)
        hi = jnp.where(resid > 0.0, t, hi)
        newton = t - resid / jnp.where(slope > 0.0, slope, 1.0)
        inside = (slope > 0.0) & (newton > lo) & (newton < hi) & jnp.isfinite(newton)
        step = jnp.where(inside, newton, 0.5 * (lo + hi))
        # A converged entry stays where it is.
        done = (jnp.abs(resid) <= score_tol) | width_ok(lo, hi, t)
        return jnp.where(done, t, step), lo, hi, resid, it + 1

    start = 0.5 * (lower + upper)
    init = (start, lower, upper, jnp.full_like(start, jnp.inf), jnp.asarray(0))
    t, _, _, _, n_iter = jax.lax.while_loop(cond, body, init)
    return t, tau(t) - targets, n_iter


def score_distribution(form: Form, nodes: Array, weights: Array) -> Array:
    r"""Return the model-implied summed-score distribution in a population.

    .. math:: f(x)=\sum_qL(x\mid\theta_q)\,w_q

    with the Lord–Wingersky likelihood :math:`L(x\mid\theta)` of
    :func:`~flexpyirt.sumscore.summed_score_distribution` (Lord & Wingersky, 1984; Thissen,
    Pommerich, Billeaud & Williams, 1995). ``nodes`` (common scale) and ``weights`` (summing to
    1) have shape ``(Q,)``; the result has shape ``(J (K_max - 1) + 1,)``.
    """
    lik = summed_score_distribution(form_probs(form, nodes))  # (Q, S)
    return jnp.einsum("qs,q->s", lik, weights)


def summed_score_eap(form: Form, nodes: Array, weights: Array) -> tuple[Array, Array, Array]:
    r"""Return the summed-score distribution, EAP and posterior SD of :math:`\theta`.

    .. math::
        f(x)=\sum_qL(x\mid\theta_q)w_q,\qquad
        \hat\theta(x)=\frac{\sum_q\theta_qL(x\mid\theta_q)w_q}{f(x)},\qquad
        \mathrm{SD}(x)^2=\frac{\sum_q(\theta_q-\hat\theta(x))^2L(x\mid\theta_q)w_q}{f(x)}

    (Thissen, Pommerich, Billeaud & Williams, 1995). Scores with :math:`f(x)=0` have NaN EAP and
    SD.
    """
    joint = summed_score_distribution(form_probs(form, nodes)) * weights[:, None]  # (Q, S)
    prob = jnp.sum(joint, axis=0)
    positive = prob > 0.0
    denom = jnp.where(positive, prob, 1.0)
    mean = nodes @ joint / denom
    var = jnp.sum((nodes[:, None] - mean[None, :]) ** 2 * joint, axis=0) / denom
    sd = jnp.sqrt(jnp.maximum(var, 0.0))
    return prob, jnp.where(positive, mean, jnp.nan), jnp.where(positive, sd, jnp.nan)


def percentile_rank(probabilities: np.ndarray) -> np.ndarray:
    r"""Return the percentile ranks (as proportions) of the integer scores ``0..S``.

    .. math:: P(x)=F(x-1)+\tfrac12f(x),\qquad F(-1)=0

    with the discrete distribution :math:`f` and its cumulative distribution :math:`F`, each
    score being uniform on :math:`[x-\tfrac12,x+\tfrac12)` (Kolen & Brennan, 2014, ch. 2).
    """
    cumulative = np.cumsum(probabilities)
    below = np.concatenate(([0.0], cumulative[:-1]))
    return np.asarray(below + 0.5 * probabilities, dtype=np.float64)


def equipercentile(new_probabilities: np.ndarray, old_probabilities: np.ndarray) -> np.ndarray:
    r"""Return the equipercentile equivalents :math:`e_Y(x)=Q^{-1}(P(x))` of ``x = 0..X``.

    With the percentile-rank function :math:`P` of :func:`percentile_rank` for the new form and
    the cumulative distribution :math:`G` of the old form (scores ``0..Y``),

    .. math::
        Q^{-1}_U(p)&=\frac{p-G(y_U-1)}{G(y_U)-G(y_U-1)}+y_U-\tfrac12,
            \qquad y_U=\min\{y:G(y)>p\},\\
        Q^{-1}_L(p)&=\frac{p-G(y_L)}{G(y_L+1)-G(y_L)}+y_L+\tfrac12,
            \qquad y_L=\max\{y:G(y)<p\},

    with :math:`G(-1)=0`, :math:`Q^{-1}_U(1)=Y+\tfrac12`, :math:`Q^{-1}_L(0)=-\tfrac12`, and
    :math:`Q^{-1}=(Q^{-1}_U+Q^{-1}_L)/2` (Kolen & Brennan, 2014, ch. 2; the two versions differ
    only where :math:`g(y)=0`). The probabilities are normalized to sum to 1 first.
    """
    f = np.asarray(new_probabilities, dtype=np.float64)
    g = np.asarray(old_probabilities, dtype=np.float64)
    f = f / f.sum()
    g = g / g.sum()
    p = percentile_rank(f)
    n_old = g.size
    cumulative = np.cumsum(g)
    cumulative[-1] = 1.0
    extended = np.concatenate(([0.0], cumulative))  # extended[y + 1] = G(y)
    scores = np.arange(n_old, dtype=np.float64)

    first_above = np.searchsorted(cumulative, p, side="right")
    at_top = first_above >= n_old
    i_upper = np.minimum(first_above, n_old - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        upper = (p - extended[i_upper]) / (extended[i_upper + 1] - extended[i_upper])
    upper = np.where(at_top, n_old - 0.5, upper + scores[i_upper] - 0.5)

    last_below = np.searchsorted(cumulative, p, side="left") - 1
    at_bottom = p <= 0.0
    i_lower = np.clip(last_below, -1, n_old - 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        lower = (p - extended[i_lower + 1]) / (extended[i_lower + 2] - extended[i_lower + 1])
    lower = np.where(at_bottom, -0.5, lower + i_lower + 0.5)
    return np.asarray(0.5 * (upper + lower), dtype=np.float64)
