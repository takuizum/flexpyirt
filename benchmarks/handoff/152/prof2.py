import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from prof import *  # noqa  (runs nothing if no argv cases)
import jax.numpy as jnp
from flexpyirt import _core

for name in sys.argv[2:]:
    m, df, g, q = case(name)
    prep = prepare(m, df, group=g, item_names=None, table=None)
    comp = compile_problem(m, prep.y, prep.group, prep.table, q)
    prob, eta = comp.problem, jnp.asarray(comp.eta0)
    lj = jax.jit(_core._log_joint)(prob, eta)
    P, Q = lj.shape

    def soft2(x):
        lse = jax.scipy.special.logsumexp(x, axis=1, keepdims=True)
        return jnp.exp(x - lse), lse

    def soft1(x):
        mx = jnp.max(x, axis=1, keepdims=True)
        e = jnp.exp(x - mx)
        s = jnp.sum(e, axis=1, keepdims=True)
        return e / s, mx + jnp.log(s)

    W = jax.jit(soft2)(lj)[0] * prob.counts[:, None]
    oh = _core._one_hots(prob)
    goh = jax.nn.one_hot(prob.group, prob.n_groups, dtype=W.dtype)
    f3 = jax.jit(lambda oh, W: jnp.einsum("pg,pjk,pq->gjkq", goh, oh, W))
    f2 = jax.jit(lambda oh, W: jnp.einsum("pjk,pq->jkq", oh, W))
    y1 = (prob.y[:, prob.families[0].columns] == 1).astype(W.dtype)
    f1 = jax.jit(lambda y1, W: y1.T @ W)
    fexp = jax.jit(jnp.exp)
    print(
        f"{name} P={P} Q={Q} soft2={t(jax.jit(soft2), lj):.2f} soft1={t(jax.jit(soft1), lj):.2f} exp={t(fexp, lj):.2f} "
        f"r3={t(f3, oh[0], W):.2f} r2={t(f2, oh[0], W):.2f} y1W={t(f1, y1, W):.2f}",
        flush=True,
    )
