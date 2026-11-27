"""Micro benchmark: warm fit time for a few cases. Usage: python micro.py <repo> [case ...]."""

import json
import os
import statistics
import sys
import time

os.environ["JAX_PLATFORMS"] = "cpu"
repo = sys.argv[1]
sys.path.insert(0, repo)
import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
import pandas as pd

from flexpyirt import BAEM, GPCM, GRM, Model, Quadrature, ThreePL, TwoPL, fit

DATA = os.path.join(repo, "benchmarks", "data")


def sim2d(n, j, seed=1):
    rng = np.random.default_rng(seed)
    th = rng.multivariate_normal([0, 0], [[1, 0.4], [0.4, 1]], size=n)
    a = rng.uniform(0.8, 2.0, size=(j, 2))
    a[: j // 2, 1] = 0
    a[j // 2 :, 0] = 0
    d = rng.normal(size=j)
    p = 1 / (1 + np.exp(-(th @ a.T + d)))
    y = (rng.uniform(size=p.shape) < p).astype(int)
    return pd.DataFrame(y, columns=[f"i{k}" for k in range(j)])


def simgpcm(n, j, k=4, seed=2):
    rng = np.random.default_rng(seed)
    th = rng.normal(size=n)
    y = np.clip(np.round(th[:, None] * 1.2 + 1.5 + rng.normal(size=(n, j))), 0, k - 1).astype(int)
    return pd.DataFrame(y, columns=[f"i{k}" for k in range(j)])


def case(name):
    q = Quadrature.equispaced(61, bounds=(-6, 6))
    if name == "2pl20k":
        df = pd.read_csv(f"{DATA}/2pl_n20000_j40.csv")
        return Model.uniform(TwoPL(), items=df.columns), df, None, q
    if name == "2pl5k":
        df = pd.read_csv(f"{DATA}/2pl_n5000_j40.csv")
        return Model.uniform(TwoPL(), items=df.columns), df, None, q
    if name == "3pl5k":
        df = pd.read_csv(f"{DATA}/3pl_n5000_j30.csv")
        return Model.uniform(ThreePL(), items=df.columns), df, None, q
    if name == "grm":
        df = pd.read_csv(f"{DATA}/grm4_n2000_j20.csv")
        return Model.uniform(GRM(n_categories=4), items=df.columns), df, None, q
    if name == "gpcm":
        df = simgpcm(5000, 20)
        return Model.uniform(GPCM(n_categories=4), items=df.columns), df, None, q
    if name == "mirt2d":
        df = sim2d(5000, 20)
        cols = list(df.columns)
        m = Model.uniform(TwoPL(), items=cols).with_factors({"f1": cols[:10], "f2": cols[10:]})
        return m, df, None, Quadrature.equispaced(31, bounds=(-6, 6))
    if name == "2grp":
        df = pd.read_csv(f"{DATA}/twogroup_2pl_n6000_j30.csv")
        from flexpyirt import Normal

        g = df.pop("group").to_numpy()
        items = list(df.columns)
        m = (
            Model.uniform(TwoPL(), items=items)
            .with_groups(ref=Normal.standard(), focal=Normal.free())
            .equal_across_groups(items=items[:10])
        )
        return m, df, g, q
    raise KeyError(name)


if __name__ == "__main__":
    out = {}
    for name in sys.argv[2:] or ["2pl20k", "3pl5k", "grm", "gpcm", "mirt2d", "2grp"]:
        m, df, g, q = case(name)
        meth = BAEM(tol=1e-4, max_iter=2000, squarem=False)
        t0 = time.perf_counter()
        r = fit(m, df, group=g, method=meth, quadrature=q, se=None)
        cold = time.perf_counter() - t0
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            r = fit(m, df, group=g, method=meth, quadrature=q, se=None)
            _ = float(r.loglik)
            ts.append(time.perf_counter() - t0)
        out[name] = dict(cold=cold, warm=statistics.median(ts), loglik=float(r.loglik))
        print(
            f"{name:8s} cold={cold:7.2f} warm={statistics.median(ts):7.3f} ll={r.loglik:.10f}",
            flush=True,
        )
    json.dump(out, sys.stdout if False else open(os.devnull, "w"))
