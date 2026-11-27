import os, time, numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"
import jax, jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def t(fn, *a, n=30):
    jax.block_until_ready(fn(*a))
    w = []
    c = []
    for _ in range(n):
        t0 = time.perf_counter()
        c0 = time.process_time()
        jax.block_until_ready(fn(*a))
        w.append(time.perf_counter() - t0)
        c.append(time.process_time() - c0)
    return f"wall={1e3 * np.min(w):6.2f} cpu={1e3 * np.median(c):6.2f}"


rng = np.random.default_rng(0)
for P, J, Q, D in [(24576, 40, 61, 1), (6144, 20, 961, 2), (6144, 30, 61, 1)]:
    Y = jnp.asarray((rng.uniform(size=(P, J)) < 0.5).astype(float))
    th = jnp.asarray(rng.normal(size=(Q, D)))
    A = jnp.asarray(rng.normal(size=(J, D)))
    d = jnp.asarray(rng.normal(size=J))
    z = (th @ A.T + d).T  # (J,Q)
    lp1 = jax.nn.log_sigmoid(z)
    lp0 = jax.nn.log_sigmoid(-z)
    oh = jnp.stack([1 - Y, Y], -1)
    lp = jnp.stack([lp0, lp1], -1)  # (J,Q,2)
    gen = jax.jit(lambda oh, lp: jnp.einsum("pjk,jqk->pq", oh, lp))
    half = jax.jit(lambda Y, lp0, lp1: jnp.sum(lp0, 0) + Y @ (lp1 - lp0))
    rank = jax.jit(lambda Y, A, d, th, lp0: jnp.sum(lp0, 0) + (Y @ d)[:, None] + (Y @ A) @ th.T)
    W = jax.nn.softmax(gen(oh, lp), axis=1)

    def s2(x):
        lse = jax.scipy.special.logsumexp(x, axis=1, keepdims=True)
        return jnp.exp(x - lse), lse

    def s1(x):
        m = jnp.max(x, axis=1, keepdims=True)
        e = jnp.exp(x - m)
        s = jnp.sum(e, axis=1, keepdims=True)
        return e / s, m + jnp.log(s)

    L = gen(oh, lp)
    rgen = jax.jit(lambda oh, W: jnp.einsum("pjk,pq->jkq", oh, W))
    rhalf = jax.jit(lambda Y, W: Y.T @ W)
    print(
        P,
        J,
        Q,
        D,
        "| lj gen",
        t(gen, oh, lp),
        "half",
        t(half, Y, lp0, lp1),
        "rank",
        t(rank, Y, A, d, th, lp0),
    )
    print(
        "   | soft2",
        t(jax.jit(s2), L),
        "soft1",
        t(jax.jit(s1), L),
        "| r gen",
        t(rgen, oh, W),
        "half",
        t(rhalf, Y, W),
        flush=True,
    )
