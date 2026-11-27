"""Time information_oakes / crossprod / fused alone, cold and warm. prof.py <datarepo> <case> which..."""
import os, sys, time, json
os.environ["JAX_PLATFORMS"] = "cpu"
repo = sys.argv[1]; case_name = sys.argv[2]; which = sys.argv[3:]
sys.path.insert(1, repo)
import jax; jax.config.update("jax_enable_x64", True)
import numpy as np, pandas as pd, jax.numpy as jnp
import flexpyirt
from flexpyirt import BAEM, Quadrature, fit, _core, _robust_se
from flexpyirt._compile import prepare, compile_problem
from benchmarks.data import CASES_BY_NAME
from benchmarks.run_flexpyirt import _build_spec
case = CASES_BY_NAME[case_name]
df = pd.read_csv(f"{repo}/benchmarks/data/{case.name}.csv")
group = df["group"].to_numpy() if case.n_groups > 1 else None
resp = df.drop(columns=["group"]) if case.n_groups > 1 else df
spec = _build_spec(case)
quad = Quadrature.equispaced(61, bounds=(-6, 6))
res = fit(spec, resp, group=group, method=BAEM(tol=1e-4, max_iter=2000), quadrature=quad, se=None)
prep = prepare(spec, resp, group=group, item_names=None, table=None)
p = compile_problem(spec, prep.y, prep.group, prep.table, quad).problem
e = jnp.asarray(res._eta)
fns = {"oakes": _core.information_oakes, "crossprod": _robust_se.information_cross_product,
       "fused": getattr(_core, "information_oakes_and_cross_product", None)}
out = {"case": case_name, "src": flexpyirt.__file__.split("/src/")[0][-12:]}
for w in which:
    fn = jax.jit(fns[w])
    t0 = time.perf_counter(); r = jax.block_until_ready(fn(p, e)); cold = time.perf_counter() - t0
    ts = []
    for _ in range(7):
        t0 = time.perf_counter(); jax.block_until_ready(fn(p, e)); ts.append(time.perf_counter() - t0)
    out[w] = {"cold": round(cold, 3), "warm": round(float(np.min(ts)), 4), "med": round(float(np.median(ts)), 4)}
    od = os.environ.get("OUTDIR")
    if od:
        np.save(f"{od}/{case_name}_{w}.npy", np.asarray(r[0] if w == "fused" else r))
        if w == "fused": np.save(f"{od}/{case_name}_fusedcross.npy", np.asarray(r[1]))
print(json.dumps(out), flush=True)
