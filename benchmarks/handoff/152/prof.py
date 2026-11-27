"""Time jitted E-step pieces. Usage: python prof.py <repo> case..."""

import os
import sys
import time

os.environ["JAX_PLATFORMS"] = "cpu"
repo = sys.argv[1]
sys.path.insert(0, os.path.dirname(__file__))
import jax

jax.config.update("jax_enable_x64", True)
import numpy as np

from flexpyirt import _core
from flexpyirt._compile import compile_problem, prepare
from micro import case


def t(fn, *a, n=20):
    jax.block_until_ready(fn(*a))
    best = []
    for _ in range(n):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*a))
        best.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(best))


for name in [] if __name__ != "__main__" else sys.argv[2:]:
    m, df, g, q = case(name)
    prep = prepare(m, df, group=g, item_names=None, table=None)
    comp = compile_problem(m, prep.y, prep.group, prep.table, q)
    prob, eta = comp.problem, jax.numpy.asarray(comp.eta0)
    if eta is None:
        import jax.numpy as jnp

        eta = jnp.asarray(comp.eta)
    lj = jax.jit(_core._log_joint)
    es = jax.jit(_core.e_step)
    ml = jax.jit(_core.marginal_loglik)

    c, _ = es(prob, eta)
    print(
        f"{name:8s} P={prob.y.shape[0]} log_joint={t(lj, prob, eta):7.2f}ms "
        f"marg={t(ml, prob, eta):7.2f}ms e_step={t(es, prob, eta):7.2f}ms",
        flush=True,
    )
