import sys

sys.path.insert(0, "tests")
import jax, jax.numpy as jnp, numpy as np

jax.config.update("jax_enable_x64", True)
from test_estep_fastpaths import _data, _spec, _problem, _generic
from flexpyirt import _core

_em = jax.jit(_core.em, static_argnames=("max_iter", "n_newton", "unconstrained"))
_squarem = jax.jit(_core.squarem, static_argnames=("max_iter", "n_newton", "unconstrained"))
for ng in (1, 3):
    df, g = _data(ng, missing=ng == 3)
    prob, eta0 = _problem(_spec(ng), df, g if ng > 1 else None)
    gp = _generic(prob)
    for name, run in (("em", _em), ("sq", _squarem)):
        for it in (5, 25):
            a = run(gp, eta0, tol=0.0, max_iter=it, n_newton=3)
            b = run(gp, eta0 * (1 + 1e-14), tol=0.0, max_iter=it, n_newton=3)
            c = run(prob, eta0, tol=0.0, max_iter=it, n_newton=3)
            print(
                ng,
                name,
                it,
                "generic-vs-perturbed",
                float(jnp.max(jnp.abs(a.eta - b.eta))),
                "fast-vs-generic",
                float(jnp.max(jnp.abs(a.eta - c.eta))),
                flush=True,
            )
