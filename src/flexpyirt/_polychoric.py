r"""Polychoric / tetrachoric correlations: jit kernels (functional core, DESIGN.md §16.9).

The bivariate normal distribution function follows Genz (2004), the two-step maximum likelihood
estimator of the correlation follows Olsson (1979). Everything here is a pure function of arrays;
names, validation and warnings live in :mod:`flexpyirt.correlation`.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr

__all__ = ["PairFit", "bivariate_normal_cdf", "fit_pairs", "pair_log_likelihood"]

_TWO_PI = 2.0 * jnp.pi
# Finite stand-in for infinite limits inside the formulas; Phi(+-40) is exactly 1 / 0 in float64.
_HUGE = 40.0

# 20-point Gauss-Legendre rule on [-1, 1], positive half (Genz, 2004, uses it for |r| >= 0.75;
# here it is used for every r, which is at least as accurate as the 6- and 12-point rules).
_GL_X = (
    0.9931285991850949,
    0.9639719272779138,
    0.9122344282513259,
    0.8391169718222188,
    0.7463319064601508,
    0.6360536807265150,
    0.5108670019508271,
    0.3737060887154196,
    0.2277858511416451,
    0.07652652113349733,
)
_GL_W = (
    0.01761400713915212,
    0.04060142980038694,
    0.06267204833410906,
    0.08327674157670475,
    0.1019301198172404,
    0.1181945319615184,
    0.1316886384491766,
    0.1420961093183821,
    0.1491729864726037,
    0.1527533871307259,
)


def _nodes_weights() -> tuple[jax.Array, jax.Array]:
    """Return the 20-point rule mapped to [0, 2] (``1 -+ x``) as in Genz's ``BVNU``."""
    x = jnp.asarray(_GL_X, dtype=jnp.float64)
    w = jnp.asarray(_GL_W, dtype=jnp.float64)
    return jnp.concatenate([1.0 - x, 1.0 + x]), jnp.concatenate([w, w])


def _upper_finite(h: jax.Array, k: jax.Array, r: jax.Array) -> jax.Array:
    r"""Genz's ``BVNU``: :math:`P(X>h, Y>k)` for finite ``h``, ``k`` and :math:`|r|\le 1`.

    For :math:`|r|<0.925` it integrates Plackett's identity in :math:`\arcsin r`,

    .. math::
        P(X>h,Y>k)=\Phi(-h)\Phi(-k)+\frac{1}{2\pi}\int_0^{\arcsin r}
        \exp\Big(-\frac{h^2+k^2-2hk\sin\theta}{2\cos^2\theta}\Big)\,d\theta,

    and for :math:`|r|\ge 0.925` it uses Drezner & Wesolowsky's (1990) expansion around
    :math:`|r|=1` as refined by Genz (2004). Both branches are evaluated and one is selected,
    so the function has no data-dependent Python control flow.
    """
    x, w = _nodes_weights()
    hk = h * k

    # |r| < 0.925.
    hs = 0.5 * (h * h + k * k)
    asr = 0.5 * jnp.arcsin(jnp.clip(r, -0.925, 0.925))
    sn = jnp.sin(asr[..., None] * x)
    small = jnp.sum(w * jnp.exp((sn * hk[..., None] - hs[..., None]) / (1.0 - sn * sn)), axis=-1)
    small = small * asr / _TWO_PI + ndtr(-h) * ndtr(-k)

    # |r| >= 0.925: flip the sign of k for negative r.
    kk = jnp.where(r < 0.0, -k, k)
    hkk = jnp.where(r < 0.0, -hk, hk)
    one_minus_r2 = jnp.maximum(1.0 - r * r, 0.0)
    bs = (h - kk) ** 2
    c = (4.0 - hkk) / 8.0
    d = (12.0 - hkk) / 16.0
    safe_as = jnp.where(one_minus_r2 > 0.0, one_minus_r2, 1.0)
    safe_a = jnp.sqrt(safe_as)
    asr_big = -(bs / safe_as + hkk) / 2.0
    first = jnp.where(
        asr_big > -100.0,
        safe_a
        * jnp.exp(jnp.minimum(asr_big, 0.0))
        * (1.0 - c * (bs - safe_as) * (1.0 - d * bs / 5.0) / 3.0 + c * d * safe_as * safe_as / 5.0),
        0.0,
    )
    b = jnp.sqrt(bs)
    safe_hkk = jnp.where(hkk > -100.0, hkk, 0.0)
    second = jnp.where(
        hkk > -100.0,
        jnp.exp(-safe_hkk / 2.0)
        * jnp.sqrt(_TWO_PI)
        * ndtr(-b / safe_a)
        * b
        * (1.0 - c * bs * (1.0 - d * bs / 5.0) / 3.0),
        0.0,
    )
    big = first - second
    half_a = safe_a / 2.0
    xs = (half_a[..., None] * x) ** 2
    rs = jnp.sqrt(1.0 - xs)
    asr_x = -(bs[..., None] / xs + hkk[..., None]) / 2.0
    term = jnp.where(
        asr_x > -100.0,
        half_a[..., None]
        * w
        * jnp.exp(jnp.minimum(asr_x, 0.0))
        * (
            jnp.exp(-hkk[..., None] * (1.0 - rs) / (2.0 * (1.0 + rs))) / rs
            - (1.0 + c[..., None] * xs * (1.0 + d[..., None] * xs))
        ),
        0.0,
    )
    big = -(big + jnp.sum(term, axis=-1)) / _TWO_PI
    big = jnp.where(one_minus_r2 > 0.0, big, 0.0)
    lower = jnp.where(h < 0.0, ndtr(kk) - ndtr(h), ndtr(-h) - ndtr(-kk))
    big = jnp.where(
        r > 0.0,
        big + ndtr(-jnp.maximum(h, kk)),
        jnp.where(h >= kk, -big, lower - big),
    )

    out = jnp.where(jnp.abs(r) < 0.925, small, big)
    return jnp.clip(out, 0.0, 1.0)


def _cdf_value(h: jax.Array, k: jax.Array, r: jax.Array) -> jax.Array:
    r"""Return :math:`\Phi_2(h,k;r)=P(X\le h,Y\le k)`, handling infinite limits exactly."""
    hc = jnp.clip(h, -_HUGE, _HUGE)
    kc = jnp.clip(k, -_HUGE, _HUGE)
    core = _upper_finite(-hc, -kc, r)
    return jnp.where(
        (h == -jnp.inf) | (k == -jnp.inf),
        0.0,
        jnp.where(h == jnp.inf, ndtr(kc), jnp.where(k == jnp.inf, ndtr(hc), core)),
    )


def bivariate_normal_density(h: jax.Array, k: jax.Array, r: jax.Array) -> jax.Array:
    r"""Evaluate the standard bivariate normal density.

    .. math::
        \phi_2(h,k;r)=\frac{1}{2\pi\sqrt{1-r^2}}
        \exp\Big(-\frac{h^2-2rhk+k^2}{2(1-r^2)}\Big),

    which is 0 when ``h`` or ``k`` is infinite.
    """
    hc = jnp.clip(h, -_HUGE, _HUGE)
    kc = jnp.clip(k, -_HUGE, _HUGE)
    s = 1.0 - r * r
    q = (hc * hc - 2.0 * r * hc * kc + kc * kc) / (2.0 * s)
    dens = jnp.exp(-q) / (_TWO_PI * jnp.sqrt(s))
    return jnp.where(jnp.isinf(h) | jnp.isinf(k), 0.0, dens)


def _partial_h(h: jax.Array, k: jax.Array, r: jax.Array) -> jax.Array:
    r""":math:`\partial\Phi_2/\partial h=\phi(h)\,\Phi\big((k-rh)/\sqrt{1-r^2}\big)`."""
    hc = jnp.clip(h, -_HUGE, _HUGE)
    kc = jnp.clip(k, -_HUGE, _HUGE)
    phi = jnp.exp(-0.5 * hc * hc) / jnp.sqrt(_TWO_PI)
    val = phi * ndtr((kc - r * hc) / jnp.sqrt(1.0 - r * r))
    return jnp.where(jnp.isinf(h) | (k == -jnp.inf), 0.0, val)


@jax.custom_jvp
def bivariate_normal_cdf(h: jax.Array, k: jax.Array, r: jax.Array) -> jax.Array:
    r"""Evaluate the standard bivariate normal distribution function :math:`\Phi_2(h,k;r)`.

    .. math::
        \Phi_2(h,k;r)=P(X\le h, Y\le k),\quad (X,Y)\sim N\Big(0,\begin{pmatrix}1&r\\r&1
        \end{pmatrix}\Big),

    computed with Genz's (2004) algorithm (20-point Gauss–Legendre rule; Drezner & Wesolowsky,
    1990, for :math:`|r|\ge 0.925`). Infinite limits are exact. The derivatives are analytic
    (custom JVP): :math:`\partial_r\Phi_2=\phi_2(h,k;r)` (Plackett, 1954) and
    :math:`\partial_h\Phi_2=\phi(h)\Phi((k-rh)/\sqrt{1-r^2})`. Requires :math:`|r|<1`.
    """
    return _cdf_value(h, k, r)


@bivariate_normal_cdf.defjvp
def _bivariate_normal_cdf_jvp(
    primals: tuple[jax.Array, jax.Array, jax.Array],
    tangents: tuple[jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array]:
    h, k, r = primals
    dh, dk, dr = tangents
    out = _cdf_value(h, k, r)
    grad = (
        _partial_h(h, k, r) * dh + _partial_h(k, h, r) * dk + bivariate_normal_density(h, k, r) * dr
    )
    return out, grad


def cell_probabilities(bounds_row: jax.Array, bounds_col: jax.Array, rho: jax.Array) -> jax.Array:
    r"""Cell probabilities of the discretized bivariate normal (Olsson, 1979, eq. 2).

    .. math::
        \pi_{ab}=\Phi_2(\tau_a,\tau'_b)-\Phi_2(\tau_{a-1},\tau'_b)
        -\Phi_2(\tau_a,\tau'_{b-1})+\Phi_2(\tau_{a-1},\tau'_{b-1}),

    with ``bounds_row`` :math:`=(-\infty,\tau_1,\dots,\tau_{K-1},\infty,\dots)` of length
    ``K_max + 1`` (padding repeats :math:`+\infty`, so padded cells have probability 0).
    """
    cdf = bivariate_normal_cdf(bounds_row[:, None], bounds_col[None, :], rho)
    return cdf[1:, 1:] - cdf[:-1, 1:] - cdf[1:, :-1] + cdf[:-1, :-1]


def pair_log_likelihood(
    z: jax.Array, bounds_row: jax.Array, bounds_col: jax.Array, counts: jax.Array
) -> jax.Array:
    r"""Two-step log-likelihood of one item pair in :math:`z=\operatorname{atanh}\rho`.

    .. math::
        \ell(\rho)=\sum_{a,b} n_{ab}\log\pi_{ab}(\rho)

    (Olsson, 1979, eq. 3, thresholds fixed at their marginal estimates). Cells with
    :math:`n_{ab}=0` contribute 0.
    """
    probs = cell_probabilities(bounds_row, bounds_col, jnp.tanh(z))
    observed = counts > 0.0
    safe = jnp.where(observed, jnp.maximum(probs, 1e-300), 1.0)
    return jnp.sum(jnp.where(observed, counts * jnp.log(safe), 0.0))


# (z, log-likelihood, gradient, iteration, done) and (step, log-likelihood, halvings).
_NewtonState = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]
_Halving = tuple[jax.Array, jax.Array, jax.Array]


class PairFit(NamedTuple):
    """Result of :func:`fit_pairs` for each pair (leading axis)."""

    rho: jax.Array
    log_likelihood: jax.Array
    gradient: jax.Array
    n_iter: jax.Array
    converged: jax.Array
    at_boundary: jax.Array


def _fit_one(
    z0: jax.Array,
    bounds_row: jax.Array,
    bounds_col: jax.Array,
    counts: jax.Array,
    *,
    z_max: float,
    tol: float,
    max_iter: int,
) -> PairFit:
    """Maximize :func:`pair_log_likelihood` over ``z`` in ``[-z_max, z_max]`` by Newton's method."""

    def loglik(z: jax.Array) -> jax.Array:
        return pair_log_likelihood(z, bounds_row, bounds_col, counts)

    grad_fn = jax.grad(loglik)
    hess_fn = jax.grad(grad_fn)

    def projected_gradient(z: jax.Array, g: jax.Array) -> jax.Array:
        pushes_out = ((z >= z_max) & (g > 0.0)) | ((z <= -z_max) & (g < 0.0))
        return jnp.where(pushes_out, 0.0, g)

    def cond(state: _NewtonState) -> jax.Array:
        _, _, _, it, done = state
        return (~done) & (it < max_iter)

    def body(state: _NewtonState) -> _NewtonState:
        z, ll, _, it, _ = state
        g = grad_fn(z)
        hz = hess_fn(z)
        # Newton step when concave; otherwise a unit gradient step.
        step = jnp.where(hz < 0.0, -g / jnp.where(hz < 0.0, hz, -1.0), jnp.sign(g) * 0.5)
        step = jnp.clip(step, -2.0, 2.0)

        def halve(carry: _Halving) -> _Halving:
            s, _, n = carry
            s = s / 2.0
            zn = jnp.clip(z + s, -z_max, z_max)
            return s, loglik(zn), n + 1

        def worse(carry: _Halving) -> jax.Array:
            _, lln, n = carry
            return (lln < ll - 1e-12 * jnp.abs(ll)) & (n < 40)

        z_try = jnp.clip(z + step, -z_max, z_max)
        step, ll_new, _ = jax.lax.while_loop(worse, halve, (step, loglik(z_try), jnp.int32(0)))
        z_new = jnp.clip(z + step, -z_max, z_max)
        g_new = grad_fn(z_new)
        pg = projected_gradient(z_new, g_new)
        n_total = jnp.sum(counts)
        done = (jnp.abs(pg) <= tol * n_total) | (jnp.abs(z_new - z) <= 1e-14)
        return z_new, ll_new, g_new, it + 1, done

    ll0 = loglik(z0)
    z, ll, g, it, done = jax.lax.while_loop(
        cond, body, (z0, ll0, grad_fn(z0), jnp.int32(0), jnp.asarray(False))
    )
    # The log-likelihood can increase monotonically towards |rho| = 1 (e.g. a 2 x 2 table with an
    # empty cell); its gradient in z then decays exponentially and the loop stops early. The
    # supremum is at the bound whenever the bound is at least as likely as the stationary point.
    z_bound = jnp.where(z >= 0.0, z_max, -z_max)
    ll_bound = loglik(z_bound)
    at_bound = ll_bound >= ll
    z = jnp.where(at_bound, z_bound, z)
    ll = jnp.where(at_bound, ll_bound, ll)
    g = jnp.where(at_bound, grad_fn(z_bound), g)
    done = done | at_bound
    return PairFit(
        rho=jnp.tanh(z),
        log_likelihood=ll,
        gradient=g,
        n_iter=it,
        converged=done,
        at_boundary=at_bound,
    )


def fit_pairs(
    z0: jax.Array,
    bounds_row: jax.Array,
    bounds_col: jax.Array,
    counts: jax.Array,
    *,
    z_max: float,
    tol: float,
    max_iter: int,
) -> PairFit:
    """Fit every pair (leading axis) with :func:`_fit_one` under ``vmap``."""

    def one(z: jax.Array, br: jax.Array, bc: jax.Array, n: jax.Array) -> PairFit:
        return _fit_one(z, br, bc, n, z_max=z_max, tol=tol, max_iter=max_iter)

    return jax.vmap(one)(z0, bounds_row, bounds_col, counts)
