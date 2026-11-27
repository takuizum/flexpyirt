# ruff: noqa
"""Time e_step / m_step / oakes on a many-group benchmark case."""

import sys, time
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, ".")
from benchmarks.data import CASES_BY_NAME, simulate_case
from benchmarks.run_flexpyirt import _build_spec
from flexpyirt import Quadrature
from flexpyirt._compile import compile_problem, prepare
import flexpyirt._core as core

name = sys.argv[1] if len(sys.argv) > 1 else "manygroups_2pl_g200"
case = CASES_BY_NAME[name]
df = simulate_case(case)
spec = _build_spec(case)
group = df.pop("group").to_numpy() if "group" in df else None
prep = prepare(spec, df, group=group, item_names=None, table=None)
comp = compile_problem(
    spec, prep.y, prep.group, prep.table, Quadrature.equispaced(61, bounds=(-6, 6))
)
prob = comp.problem
eta = jnp.asarray(comp.eta0)
print("P", prob.y.shape, "blocks", None if prob.blocks is None else prob.blocks.rows.shape)


def t(label, f, *a, n=5):
    out = f(*a)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(n):
        out = f(*a)
        jax.block_until_ready(out)
    print(f"{label}: {(time.perf_counter() - t0) / n * 1e3:.1f} ms")
    return out


estep = jax.jit(core.e_step)
counts, ll = t("e_step", estep, prob, eta)
mstep = jax.jit(lambda p, e, c: core.m_step(p, e, c, 20))
t("m_step", mstep, prob, eta, counts)
if "--oakes" in sys.argv:
    t("oakes", jax.jit(core.information_oakes), prob, eta, n=1)
