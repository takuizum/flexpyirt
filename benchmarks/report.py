"""Merge flexpyirt and R mirt benchmark outputs into JSON and Markdown reports.

Reads raw JSON files produced by ``benchmarks/run_flexpyirt.py`` and ``benchmarks/run_mirt.R``
and writes ``benchmarks/results/<date>-<machine-label>.json`` and
``benchmarks/results/<date>-<machine-label>.md``.

Run from the repository root::

    uv run python benchmarks/report.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.data import DEFAULT_DATA_DIR  # noqa: E402

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_FLEX_RAW = DEFAULT_DATA_DIR / "flexpyirt_raw.json"
DEFAULT_MIRT_RAW = DEFAULT_DATA_DIR / "mirt_raw.json"


def _fmt_sec(val: float | None) -> str:
    if val is None or not math.isfinite(val):
        return "—"
    return f"{val:.3f}"


def _fmt_mb(val: float | None) -> str:
    if val is None or not math.isfinite(val):
        return "—"
    return f"{val:.0f}"


def _fmt_ratio(num: float | None, den: float | None) -> str:
    if num is None or den is None or den <= 0.0 or not (math.isfinite(num) and math.isfinite(den)):
        return "—"
    return f"{num / den:.2f}×"


def _fmt_delta_ll(ll_a: float | None, ll_b: float | None) -> str:
    if ll_a is None or ll_b is None or not (math.isfinite(ll_a) and math.isfinite(ll_b)):
        return "—"
    return f"{abs(ll_a - ll_b):.2e}"


def _safe_ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den <= 0.0 or not (math.isfinite(num) and math.isfinite(den)):
        return None
    return num / den


def _safe_abs_diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or not (math.isfinite(a) and math.isfinite(b)):
        return None
    return abs(a - b)


def merge_results(
    flex_raw: dict[str, Any],
    mirt_raw: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge flexpyirt and optional mirt raw benchmark payloads into one structured dict."""
    flex_by_key: dict[tuple[str, str, bool], dict[str, Any]] = {
        (r["case"], r["se"], bool(r["squarem"])): r
        for r in flex_raw["runs"]
        if not r.get("anderson", False)
    }
    mirt_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    if mirt_raw is not None:
        mirt_by_key = {(r["case"], r["se"], r["algo"]): r for r in mirt_raw["runs"]}

    cases = list(flex_raw["cases"])
    rows: list[dict[str, Any]] = []
    for case in cases:
        name = case["name"]
        for se in ("none", "oakes"):
            plain = flex_by_key.get((name, se, False))
            sq = flex_by_key.get((name, se, True))
            m_def = mirt_by_key.get((name, se, "default"))
            m_nr = mirt_by_key.get((name, se, "nr_none"))
            if plain is None and sq is None and m_def is None and m_nr is None:
                continue

            plain_warm = plain["warm_s"] if plain else None
            plain_cold = plain["cold_s"] if plain else None
            sq_warm = sq["warm_s"] if sq else None
            sq_cold = sq["cold_s"] if sq else None
            m_def_s = m_def["elapsed_s"] if m_def else None
            m_nr_s = m_nr["elapsed_s"] if m_nr else None

            plain_ll = plain["loglik"] if plain else None
            sq_ll = sq["loglik"] if sq else None
            m_def_ll = m_def["loglik"] if m_def else None
            m_nr_ll = m_nr["loglik"] if m_nr else None

            ref_flex = plain if plain is not None else sq
            rows.append(
                {
                    "case": name,
                    "label": case["label"],
                    "se": se,
                    "n_params": ref_flex["n_params"] if ref_flex is not None else None,
                    "n_patterns": case.get("n_patterns"),
                    "flexpyirt_plain": plain,
                    "flexpyirt_squarem": sq,
                    "mirt_default": m_def,
                    "mirt_nr_none": m_nr,
                    "ratios": {
                        "plain_warm_vs_mirt_default": _safe_ratio(plain_warm, m_def_s),
                        "squarem_warm_vs_mirt_default": _safe_ratio(sq_warm, m_def_s),
                        "plain_cold_vs_mirt_default": _safe_ratio(plain_cold, m_def_s),
                        "squarem_cold_vs_mirt_default": _safe_ratio(sq_cold, m_def_s),
                        "plain_warm_vs_mirt_nr_none": _safe_ratio(plain_warm, m_nr_s),
                        "squarem_warm_vs_mirt_nr_none": _safe_ratio(sq_warm, m_nr_s),
                        "squarem_warm_vs_plain_warm": _safe_ratio(sq_warm, plain_warm),
                    },
                    "abs_delta_loglik": {
                        "plain_vs_mirt_default": _safe_abs_diff(plain_ll, m_def_ll),
                        "squarem_vs_mirt_default": _safe_abs_diff(sq_ll, m_def_ll),
                        "plain_vs_mirt_nr_none": _safe_abs_diff(plain_ll, m_nr_ll),
                        "squarem_vs_plain": _safe_abs_diff(sq_ll, plain_ll),
                    },
                }
            )

    return {
        "meta": {
            "flexpyirt": flex_raw["meta"],
            "mirt": mirt_raw["meta"] if mirt_raw is not None else None,
        },
        "cases": cases,
        "results": rows,
    }


def render_markdown(merged: dict[str, Any]) -> str:
    """Render a human-readable Markdown report from ``merge_results`` output."""
    fmeta = merged["meta"]["flexpyirt"]
    mmeta = merged["meta"]["mirt"]
    fset = fmeta["settings"]

    lines: list[str] = [
        f"# Benchmark results: {fmeta['date']} ({fmeta['machine_label']})",
        "",
        "## Environment",
        "",
        f"- **Machine**: `{fmeta['machine_label']}` ({fmeta['platform']})",
        f"- **CPU**: {fmeta['cpu_model']} ({fmeta['cpu_count']} logical cores"
        + (f", {fmeta['ram_gb']} GiB RAM" if fmeta.get("ram_gb") is not None else "")
        + ")",
        f"- **Python / flexpyirt**: Python {fmeta['python_version']}, "
        f"flexpyirt `{fmeta['flexpyirt_version']}`, JAX `{fmeta['jax_version']}` "
        f"(`JAX_PLATFORMS=cpu`, devices: `{', '.join(fmeta['jax_devices'])}`), "
        f"NumPy `{fmeta['numpy_version']}`, SciPy `{fmeta['scipy_version']}`, "
        f"pandas `{fmeta['pandas_version']}`",
    ]
    if mmeta is not None:
        mset = mmeta["settings"]
        lines.append(f"- **R / mirt**: {mmeta['r_version']}, mirt `{mmeta['mirt_version']}`")
        mirt_rep_note = f", mirt repeats: {mset['repeats']}"
    else:
        lines.append("- **R / mirt**: not run")
        mirt_rep_note = ""

    lines.extend(
        [
            f"- **Settings**: `TOL = {fset['tol']}`, quadrature `{fset['quadpts']}` points on "
            f"`[{fset['theta_lim'][0]:g}, {fset['theta_lim'][1]:g}]`, "
            f"max iterations `{fset['max_iter']}`, flexpyirt warm repeats: {fset['warm_repeats']}"
            + mirt_rep_note
            + (" *(quick mode)*" if fset.get("quick") else ""),
            "",
            "## Wall time (seconds)",
            "",
            "Cold time is the first `fit()` call in a fresh Python process (includes JAX/XLA "
            "compilation). Warm time is the median of repeated calls after the first. "
            "Ratios `< 1.00×` mean flexpyirt (or SQUAREM) is faster; `> 1.00×` means slower.",
            "",
            "| Case | SE | plain cold (s) | plain warm (s) | SQUAREM cold (s) | SQUAREM warm (s) "
            "| mirt default (s) | mirt NR/none (s) | plain cold / mirt def "
            "| plain warm / mirt def | SQUAREM warm / mirt def | SQUAREM / plain (warm) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for row in merged["results"]:
        fp = row["flexpyirt_plain"]
        fs = row["flexpyirt_squarem"]
        md = row["mirt_default"]
        mn = row["mirt_nr_none"]
        se_label = "None" if row["se"] == "none" else "Oakes"
        lines.append(
            "| "
            + " | ".join(
                [
                    row["label"],
                    se_label,
                    _fmt_sec(fp["cold_s"] if fp else None),
                    _fmt_sec(fp["warm_s"] if fp else None),
                    _fmt_sec(fs["cold_s"] if fs else None),
                    _fmt_sec(fs["warm_s"] if fs else None),
                    _fmt_sec(md["elapsed_s"] if md else None),
                    _fmt_sec(mn["elapsed_s"] if mn else None),
                    _fmt_ratio(fp["cold_s"] if fp else None, md["elapsed_s"] if md else None),
                    _fmt_ratio(fp["warm_s"] if fp else None, md["elapsed_s"] if md else None),
                    _fmt_ratio(fs["warm_s"] if fs else None, md["elapsed_s"] if md else None),
                    _fmt_ratio(fs["warm_s"] if fs else None, fp["warm_s"] if fp else None),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Iterations and log-likelihood sanity check",
            "",
            "SQUAREM reports `iter (EM evals, safeguards)`. `|Δ ll|` is the absolute difference "
            "in marginal log-likelihood against mirt default (at `TOL = 1e-4`).",
            "",
            "| Case | SE | Patterns | Params | plain iter | SQUAREM iter (evals, sg) "
            "| mirt def iter | mirt NR/none iter | |Δ ll| plain vs mirt "
            "| |Δ ll| SQUAREM vs mirt | Status / notes |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )

    for row in merged["results"]:
        fp = row["flexpyirt_plain"]
        fs = row["flexpyirt_squarem"]
        md = row["mirt_default"]
        mn = row["mirt_nr_none"]
        se_label = "None" if row["se"] == "none" else "Oakes"
        plain_it = str(fp["n_iter"]) if fp else "—"
        sq_it = f"{fs['n_iter']} ({fs['n_em_evaluations']}, {fs['n_safeguards']})" if fs else "—"
        m_def_it = str(md["iterations"]) if md else "—"
        m_nr_it = str(mn["iterations"]) if mn else "—"

        notes: list[str] = []
        for tag, rec in (
            ("plain", fp),
            ("SQUAREM", fs),
            ("mirt-def", md),
            ("mirt-NR", mn),
        ):
            if rec is None:
                continue
            if not rec.get("converged", True):
                notes.append(f"{tag}: NOT converged")
            if rec.get("warnings"):
                notes.append(f"{tag}: {len(rec['warnings'])} warning(s)")

        lines.append(
            "| "
            + " | ".join(
                [
                    row["label"],
                    se_label,
                    str(row["n_patterns"]) if row.get("n_patterns") is not None else "—",
                    str(row["n_params"]) if row.get("n_params") is not None else "—",
                    plain_it,
                    sq_it,
                    m_def_it,
                    m_nr_it,
                    _fmt_delta_ll(fp["loglik"] if fp else None, md["loglik"] if md else None),
                    _fmt_delta_ll(fs["loglik"] if fs else None, md["loglik"] if md else None),
                    "; ".join(notes) if notes else "ok",
                ]
            )
            + " |"
        )

    # Compile time and peak memory (recorded since #79; older raw files have neither).
    mem_rows = [
        row
        for row in merged["results"]
        if any(
            rec is not None and ("peak_rss_mb" in rec or "peak_mb" in rec)
            for rec in (row["flexpyirt_plain"], row["flexpyirt_squarem"], row["mirt_default"])
        )
    ]
    if mem_rows:
        lines.extend(
            [
                "",
                "## Compile time and peak memory",
                "",
                "flexpyirt compile time is cold minus warm time; peak memory is the worker "
                "process's peak resident set size (`ru_maxrss`, includes the JAX runtime). mirt "
                'memory is R\'s `gc()` "max used" since a reset before the fit (R heap only).',
                "",
                "| Case | SE | plain compile (s) | plain peak RSS (MiB) | SQUAREM compile (s) "
                "| SQUAREM peak RSS (MiB) | mirt default R heap (MiB) |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in mem_rows:
            fp = row["flexpyirt_plain"]
            fs = row["flexpyirt_squarem"]
            md = row["mirt_default"]
            se_label = "None" if row["se"] == "none" else "Oakes"
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["label"],
                        se_label,
                        _fmt_sec(fp.get("compile_s") if fp else None),
                        _fmt_mb(fp.get("peak_rss_mb") if fp else None),
                        _fmt_sec(fs.get("compile_s") if fs else None),
                        _fmt_mb(fs.get("peak_rss_mb") if fs else None),
                        _fmt_mb(md.get("peak_mb") if md else None),
                    ]
                )
                + " |"
            )

    # Record mirt's resolved default optimizer & acceleration per case for transparency.
    if mmeta is not None:
        mirt_opts: list[str] = []
        seen: set[str] = set()
        for row in merged["results"]:
            md = row["mirt_default"]
            if md is not None and row["case"] not in seen:
                seen.add(row["case"])
                mirt_opts.append(
                    f"`{row['case']}`: optimizer=`{md['optimizer_used']}`, "
                    f"accelerate=`{md['accelerate_used']}`"
                )
        if mirt_opts:
            lines.extend(
                [
                    "",
                    "## mirt default settings resolved per case",
                    "",
                    *[f"- {item}" for item in mirt_opts],
                ]
            )

    # Record any warnings verbatim if present.
    warn_lines: list[str] = []
    for row in merged["results"]:
        se_label = "None" if row["se"] == "none" else "Oakes"
        for tag, key in (
            ("flexpyirt plain", "flexpyirt_plain"),
            ("flexpyirt SQUAREM", "flexpyirt_squarem"),
            ("mirt default", "mirt_default"),
            ("mirt NR/none", "mirt_nr_none"),
        ):
            rec = row[key]
            if rec and rec.get("warnings"):
                for msg in rec["warnings"]:
                    warn_lines.append(f"- `{row['case']}` (`se={se_label}`, {tag}): {msg}")
    if warn_lines:
        lines.extend(["", "## Warnings recorded during runs", "", *warn_lines])

    lines.append("")
    return "\n".join(lines)


def write_report(
    flex_path: Path = DEFAULT_FLEX_RAW,
    *,
    mirt_path: Path | None = DEFAULT_MIRT_RAW,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    stem: str | None = None,
) -> tuple[Path, Path]:
    """Merge raw JSON files and write ``<stem>.json`` and ``<stem>.md`` to ``results_dir``."""
    flex_raw = json.loads(flex_path.read_text(encoding="utf-8"))
    mirt_raw: dict[str, Any] | None = None
    if mirt_path is not None and mirt_path.exists():
        mirt_raw = json.loads(mirt_path.read_text(encoding="utf-8"))

    merged = merge_results(flex_raw, mirt_raw)
    fmeta = merged["meta"]["flexpyirt"]
    report_stem = stem or f"{fmeta['date']}-{fmeta['machine_label']}"
    results_dir.mkdir(parents=True, exist_ok=True)
    json_out = results_dir / f"{report_stem}.json"
    md_out = results_dir / f"{report_stem}.md"
    json_out.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    md_out.write_text(render_markdown(merged), encoding="utf-8")
    return json_out, md_out


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flex", type=Path, default=DEFAULT_FLEX_RAW)
    parser.add_argument(
        "--mirt",
        type=Path,
        default=DEFAULT_MIRT_RAW,
        help="Path to mirt_raw.json (ignored if --no-mirt is passed or file is missing).",
    )
    parser.add_argument("--no-mirt", action="store_true", help="Report flexpyirt results only.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--stem",
        type=str,
        default=None,
        help="Output filename stem (default: <date>-<machine-label>).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    args = _parse_args(argv)
    mirt_path = None if args.no_mirt else args.mirt
    json_out, md_out = write_report(
        args.flex,
        mirt_path=mirt_path,
        results_dir=args.results_dir,
        stem=args.stem,
    )
    print(f"wrote {json_out}")
    print(f"wrote {md_out}")


if __name__ == "__main__":
    main()
