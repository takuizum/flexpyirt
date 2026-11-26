"""Compare regenerated mirt references with the committed ones (VALIDATION.md, mirt versions).

Walks two directories of reference JSON (``tests/reference/*.json``) leaf by leaf and reports every
number that moved by more than the drift tolerance of its kind, every structural change (keys or
array lengths added / removed, a number that became ``null`` or ``"Inf"``) and every file added or
removed. ``meta`` blocks (versions, SHAs, iteration counts) are not compared.

Drift tolerances, from the tolerance table of VALIDATION.md (the strictest entry of each kind, so a
change that could move a comparison test is always reported):

- standard errors, variances of estimators (key contains ``se`` as a word, ``vcov``):
  ``1e-3`` relative;
- log-likelihoods and fit statistics (key contains ``loglik``, ``log_lik``, ``logpost``,
  ``log_post``, ``log_prior``, ``aic``, ``bic``, ``m2``, ``deviance``): ``1e-6`` relative;
- everything else (parameters, latent moments, scores, probabilities, information):
  ``1e-5`` absolute plus ``1e-6`` relative.

Usage::

    uv run python scripts/compare_references.py --base <committed dir> --new tests/reference \
        [--report drift.md]

Exits with status 1 when anything drifted, 0 otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SE_KEY = re.compile(r"(^|[_.])se($|[_.\d])|vcov|^se\d", re.IGNORECASE)
FIT_KEY = re.compile(
    r"loglik|log_lik|logpost|log_post|log_prior|logprior|aic|bic|(^|_)m2|deviance", re.IGNORECASE
)
SKIP_KEYS = {"meta"}

SE_RTOL = 1e-3
FIT_RTOL = 1e-6
ATOL = 1e-5
RTOL = 1e-6


@dataclass
class FileReport:
    """Differences found in one reference file."""

    name: str
    compared: int = 0
    drifts: list[tuple[str, Any, Any, float]] = field(default_factory=list)
    structural: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        """Whether anything in this file drifted."""
        return bool(self.drifts or self.structural)


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if value in ("Inf", "-Inf", "NaN", "NA"):
        return float(value.replace("NA", "nan").replace("Inf", "inf"))
    return None


def _tolerance(path: str) -> tuple[float, float, str]:
    keys = [k for k in path.split("/") if k and not k.isdigit()]
    last = keys[-1] if keys else ""
    if any(SE_KEY.search(k) for k in keys[-2:]):
        return 0.0, SE_RTOL, "se"
    if FIT_KEY.search(last):
        return 0.0, FIT_RTOL, "fit"
    return ATOL, RTOL, "value"


def _compare(base: Any, new: Any, path: str, report: FileReport) -> None:
    if isinstance(base, dict) and isinstance(new, dict):
        for key in sorted(set(base) | set(new)):
            if key in SKIP_KEYS:
                continue
            sub = f"{path}/{key}"
            if key not in new:
                report.structural.append(f"{sub}: removed")
            elif key not in base:
                report.structural.append(f"{sub}: added")
            else:
                _compare(base[key], new[key], sub, report)
        return
    if isinstance(base, list) and isinstance(new, list):
        if len(base) != len(new):
            report.structural.append(f"{path}: length {len(base)} -> {len(new)}")
            return
        for i, (b, n) in enumerate(zip(base, new, strict=True)):
            _compare(b, n, f"{path}/{i}", report)
        return
    b_num, n_num = _as_number(base), _as_number(new)
    if b_num is None or n_num is None:
        if base != new:
            report.structural.append(f"{path}: {base!r} -> {new!r}")
        return
    report.compared += 1
    if math.isnan(b_num) and math.isnan(n_num):
        return
    if math.isinf(b_num) or math.isinf(n_num) or math.isnan(b_num) or math.isnan(n_num):
        if b_num != n_num:
            report.structural.append(f"{path}: {base!r} -> {new!r}")
        return
    atol, rtol, _ = _tolerance(path)
    diff = abs(n_num - b_num)
    if diff > atol + rtol * abs(b_num):
        report.drifts.append((path, base, new, diff / max(abs(b_num), 1e-300)))


def compare_dirs(base_dir: Path, new_dir: Path) -> tuple[list[FileReport], list[str], list[str]]:
    """Compare every ``*.json`` of two reference directories.

    Parameters
    ----------
    base_dir : Path
        Committed references.
    new_dir : Path
        Regenerated references.

    Returns
    -------
    reports : list of FileReport
        One report per file present in both directories.
    added, removed : list of str
        Files only in ``new_dir`` / only in ``base_dir``.
    """
    base = {p.name: p for p in base_dir.glob("*.json")}
    new = {p.name: p for p in new_dir.glob("*.json")}
    reports = []
    for name in sorted(set(base) & set(new)):
        report = FileReport(name)
        _compare(json.loads(base[name].read_text()), json.loads(new[name].read_text()), "", report)
        reports.append(report)
    return reports, sorted(set(new) - set(base)), sorted(set(base) - set(new))


def _mirt_provenance(path: Path) -> str:
    meta = json.loads(path.read_text()).get("meta", {})
    return f"mirt {meta.get('mirt_version')} ({meta.get('mirt_sha')})"


def format_report(
    reports: list[FileReport],
    added: list[str],
    removed: list[str],
    *,
    base_dir: Path,
    new_dir: Path,
    max_rows: int = 15,
) -> str:
    """Format the comparison as Markdown."""
    changed = [r for r in reports if r.changed]
    lines = ["## mirt reference drift", ""]
    sample = next(iter(sorted(new_dir.glob("*.json"))), None)
    sample_base = next(iter(sorted(base_dir.glob("*.json"))), None)
    if sample is not None and sample_base is not None:
        lines += [
            f"- committed: {_mirt_provenance(sample_base)}",
            f"- regenerated: {_mirt_provenance(sample)}",
        ]
    lines += [
        f"- files compared: {len(reports)}; changed: {len(changed)}; "
        f"added: {len(added)}; removed: {len(removed)}",
        f"- tolerances: SE {SE_RTOL:g} rel., log-likelihood / fit {FIT_RTOL:g} rel., "
        f"other values {ATOL:g} abs. + {RTOL:g} rel.",
        "",
    ]
    for name in added:
        lines.append(f"- added: `{name}`")
    for name in removed:
        lines.append(f"- removed: `{name}`")
    for report in changed:
        worst = max((d[3] for d in report.drifts), default=0.0)
        lines += [
            "",
            f"### `{report.name}`: {len(report.drifts)} of {report.compared} numbers drifted "
            f"(max rel. {worst:.3g}), {len(report.structural)} structural",
            "",
        ]
        if report.drifts:
            lines += ["| path | committed | regenerated | rel. diff |", "|---|---|---|---|"]
            for path, b, n, rel in sorted(report.drifts, key=lambda d: -d[3])[:max_rows]:
                lines.append(f"| `{path}` | {b!r} | {n!r} | {rel:.3g} |")
            if len(report.drifts) > max_rows:
                lines.append(f"| ... {len(report.drifts) - max_rows} more | | | |")
        for item in report.structural[:max_rows]:
            lines.append(f"- `{item}`")
        if len(report.structural) > max_rows:
            lines.append(f"- ... {len(report.structural) - max_rows} more")
    if not changed and not added and not removed:
        lines.append("No drift.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Run the comparison from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", type=Path, required=True, help="committed references")
    parser.add_argument("--new", type=Path, required=True, help="regenerated references")
    parser.add_argument("--report", type=Path, help="write the Markdown report here")
    args = parser.parse_args(argv)
    reports, added, removed = compare_dirs(args.base, args.new)
    text = format_report(reports, added, removed, base_dir=args.base, new_dir=args.new)
    if args.report is not None:
        args.report.write_text(text)
    sys.stdout.write(text)
    drifted = any(r.changed for r in reports) or bool(added or removed)
    return 1 if drifted else 0


if __name__ == "__main__":
    raise SystemExit(main())
