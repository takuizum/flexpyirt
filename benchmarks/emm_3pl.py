"""Benchmark of the 3PL / 4PL mixture M-step (``BAEM(asymptote_m_step="mixture")``, EMM).

Two parts:

- ``speed``: iterations, EM evaluations and warm wall time of ``"joint"`` and ``"mixture"`` on
  the 3PL benchmark case (``3pl_n5000_j30``) and on the 4PL validation data (``v11_4pl``), with
  and without SQUAREM, at ``tol=1e-4`` (the benchmark tolerance) and ``tol=1e-6``.
- ``small``: replicated small-sample 3PL simulations (the setting where Zheng et al., 2018 report
  that EMM converges where MMLE/EM does not): convergence rate, iterations, boundary / extreme
  estimates, the log-likelihood reached and recovery of the true parameters.

Usage::

    uv run python benchmarks/emm_3pl.py speed
    uv run python benchmarks/emm_3pl.py small --n-persons 500 --n-items 10 --reps 25
    uv run python benchmarks/emm_3pl.py report  # merges benchmarks/data/emm_*.json into Markdown

Raw JSON goes to ``benchmarks/data/`` (git-ignored).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from flexpyirt import BAEM, Model, ThreePL, fit  # noqa: E402
from flexpyirt.errors import FlexpyirtWarning  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from benchmarks.data import CASES, simulate_case  # noqa: E402

DATA = ROOT / "data"
MODES = ("joint", "mixture")


def _fit(spec: Model, df: pd.DataFrame, method: BAEM, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FlexpyirtWarning)
        return fit(spec, df, method=method, se=None, **kwargs)


def _speed_case(name: str, spec: Model, df: pd.DataFrame, quadrature: Any, repeats: int) -> list:
    rows = []
    for tol in (1e-4, 1e-6):
        for squarem in (False, True):
            ref_ll = None
            for mode in MODES:
                method = BAEM(tol=tol, max_iter=20000, squarem=squarem, asymptote_m_step=mode)
                res = _fit(spec, df, method, quadrature=quadrature)  # compile
                times = []
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    res = _fit(spec, df, method, quadrature=quadrature)
                    times.append(time.perf_counter() - t0)
                if ref_ll is None:
                    ref_ll = res.loglik
                rows.append(
                    {
                        "case": name,
                        "tol": tol,
                        "squarem": squarem,
                        "mode": mode,
                        "converged": res.converged,
                        "n_iter": res.n_iter,
                        "n_em_evaluations": res.n_em_evaluations,
                        "warm_s": statistics.median(times),
                        "loglik": res.loglik,
                        "dloglik_vs_joint": res.loglik - ref_ll,
                    }
                )
                print(rows[-1], flush=True)
    return rows


def run_speed(repeats: int) -> None:
    """Time both M-steps on the 3PL benchmark case and the V11 4PL data."""
    from tests.test_mirt_m3 import load

    from flexpyirt import Quadrature

    case = next(c for c in CASES if c.name == "3pl_n5000_j30")
    df = simulate_case(case)
    spec = Model.uniform(ThreePL(), items=df.columns)
    rows = _speed_case(case.name, spec, df, Quadrature.equispaced(), repeats)
    v11 = load("v11_4pl")
    rows += _speed_case("v11_4pl", v11.spec, v11.data, v11.quadrature, repeats)
    DATA.mkdir(exist_ok=True)
    (DATA / "emm_speed.json").write_text(json.dumps(rows, indent=1))


def _simulate_3pl(n: int, j: int, seed: int) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    a = rng.lognormal(0.0, 0.3, j)
    d = rng.uniform(-1.5, 1.0, j)
    g = rng.uniform(0.1, 0.3, j)
    theta = rng.normal(0.0, 1.0, (n, 1))
    p = g + (1.0 - g) / (1.0 + np.exp(-(theta * a + d)))
    y = (rng.random(p.shape) < p).astype(np.int64)
    cols = [f"q{k:02d}" for k in range(1, j + 1)]
    return pd.DataFrame(y, columns=cols), {"a": a, "d": d, "g": g}


def run_small(n: int, j: int, reps: int, squarem: bool, max_iter: int) -> None:
    """Fit replicated small-sample 3PL simulations with both M-steps."""
    rows = []
    for rep in range(reps):
        df, truth = _simulate_3pl(n, j, seed=10_000 * n + 100 * j + rep)
        spec = Model.uniform(ThreePL(), items=df.columns)
        for mode in MODES:
            method = BAEM(tol=1e-4, max_iter=max_iter, squarem=squarem, asymptote_m_step=mode)
            t0 = time.perf_counter()
            res = _fit(spec, df, method)
            elapsed = time.perf_counter() - t0
            it = res.items
            a, d, g = (it[p].to_numpy() for p in ("a", "d", "g"))
            rows.append(
                {
                    "n": n,
                    "j": j,
                    "rep": rep,
                    "squarem": squarem,
                    "mode": mode,
                    "converged": res.converged,
                    "n_iter": res.n_iter,
                    "n_em_evaluations": res.n_em_evaluations,
                    "seconds": elapsed,
                    "loglik": res.loglik,
                    "n_g_boundary": int(np.sum((g < 1e-4) | (g > 1 - 1e-4))),
                    "n_extreme_a": int(np.sum(np.abs(a) > 10)),
                    "max_abs_a": float(np.max(np.abs(a))),
                    "rmse_a": float(np.sqrt(np.mean((a - truth["a"]) ** 2))),
                    "rmse_d": float(np.sqrt(np.mean((d - truth["d"]) ** 2))),
                    "rmse_g": float(np.sqrt(np.mean((g - truth["g"]) ** 2))),
                }
            )
            print(rows[-1], flush=True)
    DATA.mkdir(exist_ok=True)
    tag = f"n{n}_j{j}_{'sq' if squarem else 'em'}"
    (DATA / f"emm_small_{tag}.json").write_text(json.dumps(rows, indent=1))


def _markdown(frame: pd.DataFrame) -> str:
    """Return ``frame`` as a GitHub Markdown table (no ``tabulate`` dependency)."""
    cols = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += [
        "| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)
    ]
    return "\n".join(lines)


def report() -> None:
    """Merge the raw JSON files into a Markdown report."""
    out = []
    speed = DATA / "emm_speed.json"
    if speed.exists():
        s = pd.DataFrame(json.loads(speed.read_text()))
        out.append("## Speed (warm, median of repeats)\n")
        cols = ["case", "tol", "squarem", "mode", "converged", "n_iter", "n_em_evaluations"]
        s["warm_s"] = s["warm_s"].round(2)
        s["dloglik_vs_joint"] = s["dloglik_vs_joint"].map(lambda x: f"{x:+.1e}")
        out.append(_markdown(s[[*cols, "warm_s", "dloglik_vs_joint"]]))
    files = sorted(DATA.glob("emm_small_*.json"))
    if files:
        sm = pd.concat([pd.DataFrame(json.loads(f.read_text())) for f in files])
        best = sm.groupby(["n", "j", "rep", "squarem"])["loglik"].transform("max")
        sm["below_best"] = sm["loglik"] < best - 1e-3
        agg = (
            sm.groupby(["n", "j", "squarem", "mode"])
            .agg(
                reps=("rep", "size"),
                converged=("converged", "mean"),
                median_iter=("n_iter", "median"),
                max_iter=("n_iter", "max"),
                median_s=("seconds", "median"),
                fits_g_boundary=("n_g_boundary", lambda x: int((x > 0).sum())),
                fits_extreme_a=("n_extreme_a", lambda x: int((x > 0).sum())),
                lower_loglik=("below_best", "sum"),
                rmse_a=("rmse_a", "median"),
                rmse_d=("rmse_d", "median"),
                rmse_g=("rmse_g", "median"),
            )
            .reset_index()
        )
        out.append("\n## Small samples (3PL, tol=1e-4)\n")
        out.append(_markdown(agg.round(3)))
    text = "\n".join(out) + "\n"
    (DATA / "emm_report.md").write_text(text)
    print(text)


def main() -> None:
    """Parse the command line and run one part."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("speed")
    sp.add_argument("--repeats", type=int, default=3)
    sm = sub.add_parser("small")
    sm.add_argument("--n-persons", type=int, default=500)
    sm.add_argument("--n-items", type=int, default=10)
    sm.add_argument("--reps", type=int, default=25)
    sm.add_argument("--squarem", action="store_true")
    sm.add_argument("--max-iter", type=int, default=2000)
    sub.add_parser("report")
    args = parser.parse_args()
    if args.cmd == "speed":
        run_speed(args.repeats)
    elif args.cmd == "small":
        run_small(args.n_persons, args.n_items, args.reps, args.squarem, args.max_iter)
    else:
        report()


if __name__ == "__main__":
    main()
