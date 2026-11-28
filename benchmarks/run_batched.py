"""Benchmark batched multi-model estimation against fitting the models one by one (#157).

Two workloads, each in a fresh subprocess (``JAX_PLATFORMS=cpu``) so the cold call includes
compilation; warm times are the median of ``--repeats`` further calls in the same process:

- ``dif30``: ``FitResult.dif(scheme="drop")`` likelihood-ratio tests of all 30 items of a
  two-group 2PL (2000 respondents per group, focal mean and variance free): 30 refits.
- ``mix16``: 16 starts of a 2PL with a two-component ``NormalMixture`` latent density
  (2000 respondents, 20 items): ``fit(table=...)`` once per start (``loop``) vs. the same 16
  starts in one batch (``batched``: the default, a sequential ``lax.map`` on CPU;
  ``batched_vmap``: ``vmap``), via the parameter tables, so all variants use identical starts.

Works on revisions without the batched code (then only the ``loop`` variants run), so the same
script measures a base checkout. Run from the repository root::

    uv run python benchmarks/run_batched.py --out results.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

WORKLOADS = ("dif30", "mix16")


def _dif_data() -> tuple[Any, Any]:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(157)
    n_items, n_per = 30, 2000
    a = rng.uniform(0.8, 2.0, n_items)
    d = rng.normal(0.0, 1.0, n_items)
    out = []
    for g, mu in ((0, 0.0), (1, 0.3)):
        dd = d.copy()
        if g == 1:
            dd[:3] += 0.6
        theta = rng.normal(mu, 1.0, (n_per, 1))
        out.append((rng.random((n_per, n_items)) < 1 / (1 + np.exp(-(theta * a + dd)))) * 1.0)
    df = pd.DataFrame(np.vstack(out), columns=[f"i{k:02d}" for k in range(n_items)])
    return df, np.repeat(["r", "f"], n_per)


def _mix_data() -> Any:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(1570)
    n_obs, n_items = 2000, 20
    cls = rng.random(n_obs) < 0.35
    theta = np.where(cls, rng.normal(2.0, 0.6, n_obs), rng.normal(0.0, 1.0, n_obs))
    a = rng.uniform(0.9, 1.9, n_items)
    d = np.linspace(-1.5, 2.0, n_items)
    y = (rng.random((n_obs, n_items)) < 1 / (1 + np.exp(-(theta[:, None] * a + d)))) * 1.0
    return pd.DataFrame(y, columns=[f"q{k:02d}" for k in range(n_items)])


def _time(fn: Any, repeats: int) -> dict[str, float]:
    t0 = time.perf_counter()
    fn()
    cold = time.perf_counter() - t0
    warm = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        warm.append(time.perf_counter() - t0)
    return {"cold_s": cold, "warm_s": statistics.median(warm)}


def _child(workload: str, variant: str, repeats: int) -> dict[str, Any]:
    import jax

    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
    import numpy as np

    from flexpyirt import BAEM, Model, Normal, NormalMixture, Quadrature, TwoPL, fit

    warnings.simplefilter("ignore")
    method = BAEM(tol=1e-4, max_iter=2000)
    quad = Quadrature.equispaced(61, bounds=(-6, 6))
    extra: dict[str, Any] = {}
    if workload == "dif30":
        df, grp = _dif_data()
        spec = (
            Model.uniform(TwoPL(), items=df.columns)
            .with_groups(r=Normal.standard(), f=Normal.free())
            .equal_across_groups(items=list(df.columns))
        )
        res = fit(spec, df, group=grp, method=method, quadrature=quad, se=None)

        def run() -> None:
            out = res.dif(df, group=grp, items=list(df.columns), scheme="drop")
            extra["x2"] = out["x2"].tolist()

        timing = _time(run, repeats)
    else:
        df = _mix_data()
        spec = Model.uniform(TwoPL(), items=df.columns).with_latent(NormalMixture(2))
        table = spec.parameter_table(df)
        rng = np.random.default_rng(16)
        tables = []
        free = table["free"].to_numpy()
        for m in range(16):
            t = table.copy()
            if m > 0:
                vals = t["value"].to_numpy(dtype=float).copy()
                unbounded = free & ~np.isfinite(t["lower"].to_numpy(dtype=float))
                vals[unbounded] += 0.5 * rng.standard_normal(int(unbounded.sum()))
                t["value"] = vals
            tables.append(t)
        if variant == "loop":

            def run() -> None:
                fits = [
                    fit(spec, df, table=t, method=method, quadrature=quad, se=None) for t in tables
                ]
                extra["loglik"] = [f.loglik for f in fits]
                extra["n_iter"] = [f.n_iter for f in fits]

        else:
            from flexpyirt.fit import _fit_tables

            def run() -> None:
                fits = _fit_tables(
                    spec,
                    df,
                    tables=tables,
                    group=None,
                    item_names=None,
                    method=method,
                    quadrature=quad,
                    se=None,
                    **({"parallel": True} if variant == "batched_vmap" else {}),
                )
                extra["loglik"] = [f.loglik for f in fits]
                extra["n_iter"] = [f.n_iter for f in fits]

        timing = _time(run, repeats)
    return {"workload": workload, "variant": variant, **timing, **extra}


def main(argv: list[str] | None = None) -> int:
    """Run every workload in its own subprocess and print / save the timings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=False)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--child", nargs=2, metavar=("WORKLOAD", "VARIANT"))
    args = parser.parse_args(argv)
    if args.child:
        print(json.dumps(_child(args.child[0], args.child[1], args.repeats)))
        return 0
    try:
        import flexpyirt._batch  # noqa: F401

        has_batch = True
    except ImportError:
        has_batch = False
    jobs = [("dif30", "dif"), ("mix16", "loop")]
    if has_batch:
        jobs += [("mix16", "batched"), ("mix16", "batched_vmap")]
    results = []
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    for workload, variant in jobs:
        cmd = [sys.executable, __file__, "--child", workload, variant]
        out = subprocess.check_output(
            [*cmd, "--repeats", str(args.repeats)],
            env=env,
            text=True,
        )
        row = json.loads(out.strip().splitlines()[-1])
        print(
            f"{workload:6s} {variant:8s} cold {row['cold_s']:8.2f} s  warm {row['warm_s']:8.2f} s",
            flush=True,
        )
        results.append(row)
    if args.out:
        args.out.write_text(json.dumps({"has_batch": has_batch, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
