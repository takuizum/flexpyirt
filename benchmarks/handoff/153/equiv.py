# ruff: noqa
"""Fit many-group / multigroup cases and dump estimates, SEs, loglik and iterations as JSON.

Run from a checkout's root: python equiv.py OUT.json
"""

import json, sys
import jax

jax.config.update("jax_enable_x64", True)
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "tests")
from benchmarks.data import CASES_BY_NAME, simulate_case
from benchmarks.run_flexpyirt import _build_spec
from flexpyirt import BAEM, Quadrature, fit

out = {}
QUAD = Quadrature.equispaced(61, bounds=(-6, 6))
cases = []
for name in [
    "twogroup_2pl_n6000_j30",
    "manygroups_2pl_g10",
    "manygroups_2pl_g50",
    "manygroups_2pl_g200",
]:
    case = CASES_BY_NAME[name]
    df = simulate_case(case)
    group = df.pop("group").to_numpy()
    cases.append((name, _build_spec(case), df, group, QUAD))
from test_many_groups import QUAD as TQ, _data, _spec

for n_anchors in (4, 8):
    df, labels = _data(3)
    cases.append((f"test_grm_missing_g3_a{n_anchors}", _spec(3, n_anchors), df, labels, TQ))
for name, spec, df, group, quad in cases:
    for squarem in (False, True):
        se = "oakes" if name != "manygroups_2pl_g200" or squarem else "none"
        res = fit(
            spec,
            df,
            group=group,
            method=BAEM(squarem=squarem, max_iter=5000),
            quadrature=quad,
            se=se,
        )
        p = res.parameters
        key = f"{name}|squarem={squarem}"
        out[key] = {
            "loglik": float(res.loglik),
            "n_iter": int(res.n_iter),
            "converged": bool(res.converged),
            "estimate": p["estimate"].to_numpy().tolist(),
            "se": p["se"].to_numpy(dtype=float).tolist() if "se" in p else None,
        }
        print(key, out[key]["loglik"], out[key]["n_iter"], flush=True)
json.dump(out, open(sys.argv[1], "w"))
