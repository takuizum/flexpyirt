"""Benchmark flexpyirt wall time on CPU (cold and warm) across models, SQUAREM, and SE options.

Each ``(case, squarem, se)`` combination is executed in a fresh Python subprocess with
``JAX_PLATFORMS=cpu`` so that the **cold** call genuinely includes first-time JAX/XLA compilation
and cannot reuse another configuration's in-process JIT cache. Subsequent calls in the same
subprocess measure **warm** wall time (median of ``--repeats``, default 3).

Run from the repository root::

    uv run python benchmarks/run_flexpyirt.py
    uv run python benchmarks/run_flexpyirt.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import warnings
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any, Literal

# Allow both `python benchmarks/run_flexpyirt.py` and `python -m benchmarks.run_flexpyirt`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.data import (  # noqa: E402
    CASES_BY_NAME,
    DEFAULT_DATA_DIR,
    BenchmarkCase,
    generate_datasets,
    resolve_cases,
)

DEFAULT_OUT_PATH = DEFAULT_DATA_DIR / "flexpyirt_raw.json"
DEFAULT_TOL = 1e-4
DEFAULT_MAX_ITER = 2000
DEFAULT_QUADPTS = 61
DEFAULT_THETA_LIM = (-6.0, 6.0)
DEFAULT_WARM_REPEATS = 3

SEArg = Literal["none", "oakes"]


def _cpu_model() -> str:
    """Return a human-readable CPU model string."""
    if sys.platform == "darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
            if out:
                return out
        except OSError:
            pass
    elif sys.platform.startswith("linux"):
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.lower().startswith("model name"):
                    _, _, value = line.partition(":")
                    if value.strip():
                        return value.strip()
    return platform.processor() or platform.machine()


def _ram_gb() -> float | None:
    """Return installed physical RAM in GiB if available."""
    if sys.platform == "darwin":
        try:
            raw = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            return round(int(raw) / (1024**3), 2)
        except (OSError, ValueError):
            return None
    if sys.platform.startswith("linux"):
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return round(int(parts[1]) / (1024**2), 2)
    return None


def _machine_label() -> str:
    """Return a short label for the result file name, derived from the CPU model.

    The hostname is deliberately not recorded: results are committed to a public repository.
    """
    raw = _cpu_model().lower()
    for noise in ("(r)", "(tm)", "cpu", "processor", "@"):
        raw = raw.replace(noise, " ")
    words = [w for w in raw.replace("-", " ").split() if w.isalnum() and "ghz" not in w]
    label = "-".join(words[:4])
    return label or platform.machine().lower() or "machine"


def _build_spec(case: BenchmarkCase) -> Any:
    from flexpyirt import GRM, Model, Normal, ThreePL, TwoPL

    items = case.item_names
    if case.model == "2pl":
        return Model.uniform(TwoPL(), items=items)
    if case.model == "grm":
        return Model.uniform(GRM(n_cat=case.n_cat), items=items)
    if case.model == "3pl":
        return Model.uniform(ThreePL(), items=items)
    if case.model == "twogroup_2pl":
        return (
            Model.uniform(TwoPL(), items=items)
            .with_groups(ref=Normal.standard(), focal=Normal.free())
            .equal_across_groups(items=case.anchors)
        )
    raise ValueError(f"Unknown model kind: {case.model}")


def _run_worker(
    case_name: str,
    *,
    data_dir: Path,
    squarem: bool,
    se_arg: SEArg,
    warm_repeats: int,
    tol: float,
    max_iter: int,
    quadpts: int,
    theta_lim: tuple[float, float],
) -> dict[str, Any]:
    """Execute one cold + ``warm_repeats`` warm fits inside the current (fresh) process."""
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax

    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

    import numpy as np
    import pandas as pd

    from flexpyirt import BAEM, Quadrature, fit

    case = CASES_BY_NAME[case_name]
    df = pd.read_csv(data_dir / f"{case.name}.csv")
    if case.n_groups > 1:
        group: Any = df["group"].to_numpy()
        resp = df.drop(columns=["group"])
    else:
        group = None
        resp = df

    spec = _build_spec(case)
    method = BAEM(tol=tol, max_iter=max_iter, squarem=squarem)
    quad = Quadrature.equispaced(quadpts, lim=theta_lim)
    se_opt: Literal["oakes"] | None = None if se_arg == "none" else "oakes"

    def _one_call() -> tuple[float, Any, list[str]]:
        captured: list[str] = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t0 = time.perf_counter()
            res = fit(spec, resp, group=group, method=method, quadrature=quad, se=se_opt)
            # FitResult materializes numpy arrays (blocking JAX execution); ensure host floats
            # are read before stopping the timer.
            _ = float(res.loglik)
            if res.vcov is not None:
                _ = float(res.vcov[0, 0])
            t1 = time.perf_counter()
        assert isinstance(res._eta, np.ndarray)
        for w in caught:
            captured.append(f"{w.category.__name__}: {w.message}")
        return t1 - t0, res, captured

    cold_s, cold_res, cold_warnings = _one_call()
    warm_runs_s: list[float] = []
    for _ in range(warm_repeats):
        dt, warm_res, _ = _one_call()
        warm_runs_s.append(dt)
        if abs(warm_res.loglik - cold_res.loglik) > 1e-10 * max(1.0, abs(cold_res.loglik)):
            raise RuntimeError(
                f"Cold/warm loglik mismatch for {case_name}: {cold_res.loglik} vs {warm_res.loglik}"
            )

    return {
        "case": case_name,
        "squarem": squarem,
        "se": se_arg,
        "cold_s": cold_s,
        "warm_s": statistics.median(warm_runs_s),
        "warm_min_s": min(warm_runs_s),
        "warm_max_s": max(warm_runs_s),
        "warm_runs_s": warm_runs_s,
        "n_iter": int(cold_res.n_iter),
        "n_em_evaluations": int(cold_res.n_em_evaluations),
        "n_safeguards": int(cold_res.n_safeguards),
        "converged": bool(cold_res.converged),
        "loglik": float(cold_res.loglik),
        "n_params": int(cold_res.n_params),
        "n_obs": int(cold_res.n_obs),
        "warnings": cold_warnings,
        "jax_devices": [repr(d) for d in jax.devices()],
    }


def _collect_meta(
    *,
    tol: float,
    max_iter: int,
    quadpts: int,
    theta_lim: tuple[float, float],
    warm_repeats: int,
    quick: bool,
    jax_devices: list[str],
) -> dict[str, Any]:
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    import numpy as np
    import pandas as pd
    import scipy

    import flexpyirt

    return {
        "date": date.today().isoformat(),
        "machine_label": _machine_label(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "python_version": platform.python_version(),
        "flexpyirt_version": flexpyirt.__version__,
        "jax_version": jax.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "pandas_version": pd.__version__,
        "jax_platforms": "cpu",
        "jax_devices": jax_devices or [repr(d) for d in jax.devices()],
        "settings": {
            "tol": tol,
            "max_iter": max_iter,
            "quadpts": quadpts,
            "theta_lim": list(theta_lim),
            "warm_repeats": warm_repeats,
            "quick": quick,
        },
    }


def run_benchmark(
    cases: Sequence[BenchmarkCase],
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    out_path: Path = DEFAULT_OUT_PATH,
    warm_repeats: int = DEFAULT_WARM_REPEATS,
    tol: float = DEFAULT_TOL,
    max_iter: int = DEFAULT_MAX_ITER,
    quadpts: int = DEFAULT_QUADPTS,
    theta_lim: tuple[float, float] = DEFAULT_THETA_LIM,
    quick: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run all flexpyirt benchmark configurations and write raw JSON output."""
    manifest_path = data_dir / "manifest.json"
    missing_csv = [c for c in cases if not (data_dir / f"{c.name}.csv").exists()]
    if not manifest_path.exists() or missing_csv:
        generate_datasets(cases, data_dir=data_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_by_name = {entry["name"]: entry for entry in manifest["cases"]}

    script_path = Path(__file__).resolve()
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    records: list[dict[str, Any]] = []
    jax_devices: list[str] = []

    for case in cases:
        for se_arg in ("none", "oakes"):
            for squarem in (False, True):
                cmd = [
                    sys.executable,
                    str(script_path),
                    "--worker",
                    "--case",
                    case.name,
                    "--data-dir",
                    str(data_dir),
                    "--squarem",
                    "true" if squarem else "false",
                    "--se",
                    se_arg,
                    "--repeats",
                    str(warm_repeats),
                    "--tol",
                    str(tol),
                    "--max-iter",
                    str(max_iter),
                    "--quadpts",
                    str(quadpts),
                    "--theta-lim",
                    str(theta_lim[0]),
                    str(theta_lim[1]),
                ]
                proc = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    env=env,
                )
                # Take the last non-empty line in case JAX prints startup notes to stdout.
                lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
                rec = json.loads(lines[-1])
                if not jax_devices and rec.get("jax_devices"):
                    jax_devices = list(rec["jax_devices"])
                records.append(rec)
                if verbose:
                    mode = "SQUAREM" if squarem else "plain  "
                    warn_tag = f" [warnings={len(rec['warnings'])}]" if rec["warnings"] else ""
                    print(
                        f"{case.name:24s} se={se_arg:5s} {mode} "
                        f"cold={rec['cold_s']:6.3f}s warm={rec['warm_s']:6.3f}s "
                        f"iter={rec['n_iter']:4d} em_evals={rec['n_em_evaluations']:4d} "
                        f"ll={rec['loglik']:.4f} conv={rec['converged']}{warn_tag}",
                        flush=True,
                    )

    payload: dict[str, Any] = {
        "meta": _collect_meta(
            tol=tol,
            max_iter=max_iter,
            quadpts=quadpts,
            theta_lim=theta_lim,
            warm_repeats=warm_repeats,
            quick=quick,
            jax_devices=jax_devices,
        ),
        "cases": [manifest_by_name[c.name] for c in cases if c.name in manifest_by_name],
        "runs": records,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=list(CASES_BY_NAME.keys()),
        help="Subset of benchmark cases to run.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer cases (unless --cases is given) and 1 warm repeat.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Number of warm repeats after the cold call (default: 3; 1 in --quick).",
    )
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--quadpts", type=int, default=DEFAULT_QUADPTS)
    parser.add_argument(
        "--theta-lim",
        type=float,
        nargs=2,
        default=list(DEFAULT_THETA_LIM),
        metavar=("LOW", "HIGH"),
    )
    # Internal worker flags
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--squarem", choices=("true", "false"), default=None, help=argparse.SUPPRESS
    )
    parser.add_argument("--se", choices=("none", "oakes"), default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    args = _parse_args(argv)
    theta_lim = (float(args.theta_lim[0]), float(args.theta_lim[1]))
    if args.worker:
        if args.case is None or args.squarem is None or args.se is None:
            raise SystemExit("--worker requires --case, --squarem, and --se.")
        rec = _run_worker(
            args.case,
            data_dir=args.data_dir,
            squarem=(args.squarem == "true"),
            se_arg=args.se,
            warm_repeats=1 if args.repeats is None else args.repeats,
            tol=args.tol,
            max_iter=args.max_iter,
            quadpts=args.quadpts,
            theta_lim=theta_lim,
        )
        print(json.dumps(rec))
        return

    repeats = (1 if args.quick else DEFAULT_WARM_REPEATS) if args.repeats is None else args.repeats
    if not args.quick and repeats < 3:
        raise SystemExit("Full benchmark requires --repeats >= 3 (use --quick for 1 repeat).")
    cases = resolve_cases(args.cases, quick=args.quick)
    run_benchmark(
        cases,
        data_dir=args.data_dir,
        out_path=args.out,
        warm_repeats=repeats,
        tol=args.tol,
        max_iter=args.max_iter,
        quadpts=args.quadpts,
        theta_lim=theta_lim,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
