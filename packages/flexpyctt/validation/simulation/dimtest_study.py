"""DIMTEST simulation study (VALIDATION.md V30): type I error and power.

Run from the repository root:
  uv run python packages/flexpyctt/validation/simulation/dimtest_study.py
Writes dimtest_study.json next to this file. Not run in CI (minutes); the test suite runs a
small version of the same conditions.

Conditions: 2PL data (slopes U(0.8, 1.8), intercepts N(0, 0.8)) with J items in two halves on
correlated dimensions; rho = 1 is unidimensional (type I error), rho < 1 two-dimensional
(power). Each replication draws new item parameters and data and runs flexpyctt.dimtest() with
its defaults (AT chosen on a third of the respondents, B = 100).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from flexpyctt import dimtest

OUT = Path(__file__).with_suffix(".json")


def simulate(rng: np.random.Generator, n: int, n_items: int, rho: float) -> np.ndarray:
    """2PL responses with two halves of items on dimensions correlated ``rho``."""
    theta = rng.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], size=n)
    a = rng.uniform(0.8, 1.8, n_items)
    d = rng.normal(0.0, 0.8, n_items)
    dim = np.repeat([0, 1], n_items // 2)
    eta = a * theta[:, dim] + d
    return (rng.uniform(size=(n, n_items)) < 1.0 / (1.0 + np.exp(-eta))).astype(np.float64)


def run(n: int, n_items: int, rho: float, reps: int, seed: int) -> dict[str, object]:
    """Run ``reps`` replications of one condition and summarize T and the rejection rates."""
    rng = np.random.default_rng(seed)
    stats, ps = [], []
    for r in range(reps):
        res = dimtest(simulate(rng, n, n_items, rho), rng=seed * 1000 + r)
        stats.append(res.statistic)
        ps.append(res.p_value)
    t = np.asarray(stats)
    p = np.asarray(ps)
    return {
        "n": n,
        "n_items": n_items,
        "rho": rho,
        "reps": reps,
        "mean_T": float(t.mean()),
        "sd_T": float(t.std(ddof=1)),
        "reject_05": float(np.mean(p < 0.05)),
        "reject_10": float(np.mean(p < 0.10)),
    }


def main() -> None:
    """Run every condition and write the JSON summary."""
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    conditions = [
        (1000, 20, 1.0),
        (2000, 20, 1.0),
        (2000, 30, 1.0),
        (1000, 20, 0.7),
        (2000, 20, 0.7),
        (2000, 30, 0.8),
    ]
    results = []
    for i, (n, j, rho) in enumerate(conditions):
        t0 = time.time()
        row = run(n, j, rho, reps, seed=3000 + i)
        row["seconds"] = round(time.time() - t0, 1)
        print(json.dumps(row))
        results.append(row)
    OUT.write_text(json.dumps({"reps": reps, "conditions": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
