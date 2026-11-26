"""Deterministic simulated datasets for the flexpyirt vs. R mirt benchmark suite.

Writes one CSV per benchmark case plus ``manifest.json`` into ``benchmarks/data/`` (git-ignored)
so Python (``run_flexpyirt.py``) and R (``run_mirt.R``) fit identical response matrices.

Run from the repository root::

    uv run python benchmarks/data.py
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"

ModelKind = Literal["2pl", "grm", "3pl", "twogroup_2pl", "manygroups_2pl"]


@dataclass(frozen=True)
class BenchmarkCase:
    """Definition of one benchmark dataset.

    Attributes
    ----------
    name : str
        Machine-readable case identifier (used for CSV filenames and JSON keys).
    label : str
        Human-readable label for Markdown tables.
    model : {"2pl", "grm", "3pl", "twogroup_2pl", "manygroups_2pl"}
        Model family. ``manygroups_2pl`` has ``n_groups`` groups with every item invariant
        across groups and free focal means and variances (DESIGN.md §26).
    n_persons : int
        Total number of respondents, split equally across the groups.
    n_items : int
        Number of items.
    n_categories : int
        Number of response categories (2 for dichotomous, 4 for GRM).
    seed : int
        NumPy ``default_rng`` seed.
    n_groups : int
        Number of groups (1 for single-group, 2 for ``twogroup_2pl``).
    n_anchors : int
        Number of anchor items shared across groups (0 for single-group).
    """

    name: str
    label: str
    model: ModelKind
    n_persons: int
    n_items: int
    n_categories: int
    seed: int
    n_groups: int = 1
    n_anchors: int = 0

    @property
    def item_names(self) -> list[str]:
        """Return item column names ``q01..q{J}``."""
        return [f"q{j:02d}" for j in range(1, self.n_items + 1)]

    @property
    def group_names(self) -> list[str]:
        """Return the group labels in model order (the first is the reference group)."""
        if self.model == "twogroup_2pl":
            return ["ref", "focal"]
        if self.model == "manygroups_2pl":
            return [f"s{g:03d}" for g in range(1, self.n_groups + 1)]
        return ["all"]

    @property
    def anchors(self) -> list[str]:
        """Return anchor item names (first ``n_anchors`` items)."""
        return self.item_names[: self.n_anchors]


CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        name="2pl_n1000_j20",
        label="2PL (1000 × 20)",
        model="2pl",
        n_persons=1000,
        n_items=20,
        n_categories=2,
        seed=202601,
    ),
    BenchmarkCase(
        name="2pl_n5000_j40",
        label="2PL (5000 × 40)",
        model="2pl",
        n_persons=5000,
        n_items=40,
        n_categories=2,
        seed=202602,
    ),
    BenchmarkCase(
        name="2pl_n20000_j40",
        label="2PL (20000 × 40)",
        model="2pl",
        n_persons=20000,
        n_items=40,
        n_categories=2,
        seed=202603,
    ),
    BenchmarkCase(
        name="grm4_n2000_j20",
        label="GRM K=4 (2000 × 20)",
        model="grm",
        n_persons=2000,
        n_items=20,
        n_categories=4,
        seed=202604,
    ),
    BenchmarkCase(
        name="3pl_n5000_j30",
        label="3PL (5000 × 30)",
        model="3pl",
        n_persons=5000,
        n_items=30,
        n_categories=2,
        seed=202605,
    ),
    BenchmarkCase(
        name="twogroup_2pl_n6000_j30",
        label="2-group 2PL (2×3000 × 30)",
        model="twogroup_2pl",
        n_persons=6000,
        n_items=30,
        n_categories=2,
        seed=202606,
        n_groups=2,
        n_anchors=10,
    ),
)

MANY_GROUP_CASES: tuple[BenchmarkCase, ...] = tuple(
    BenchmarkCase(
        name=f"manygroups_2pl_g{g}",
        label=f"{g}-group 2PL ({g}×200 × 20)",
        model="manygroups_2pl",
        n_persons=200 * g,
        n_items=20,
        n_categories=2,
        seed=202610 + g,
        n_groups=g,
        n_anchors=20,
    )
    for g in (10, 50, 200)
)
"""Many-group cases (#79, DESIGN.md §26); not in the default suite (``--many-groups``)."""

CASES_BY_NAME: dict[str, BenchmarkCase] = {c.name: c for c in (*CASES, *MANY_GROUP_CASES)}
QUICK_CASE_NAMES: tuple[str, ...] = ("2pl_n1000_j20", "grm4_n2000_j20")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-x)), dtype=np.float64)


def simulate_case(case: BenchmarkCase) -> pd.DataFrame:
    """Simulate a deterministic response DataFrame for ``case``.

    For ``twogroup_2pl`` the first column is ``"group"`` (values ``"ref"`` and ``"focal"``)
    followed by item columns ``q01..q{J}``. Single-group cases contain only the item columns.
    """
    rng = np.random.default_rng(case.seed)
    n, j = case.n_persons, case.n_items
    cols = case.item_names

    if case.model == "2pl":
        a = rng.lognormal(0.0, 0.3, j)
        d = rng.normal(0.0, 1.0, j)
        theta = rng.normal(0.0, 1.0, (n, 1))
        p = _sigmoid(theta * a[None, :] + d[None, :])
        y = (rng.random(p.shape) < p).astype(np.int64)
        df = pd.DataFrame(y, columns=cols)
    elif case.model == "grm":
        k = case.n_categories
        a = rng.lognormal(0.0, 0.3, j)
        d1 = rng.normal(1.2, 0.3, j)
        gaps = rng.uniform(0.8, 1.4, (j, k - 2))
        d = np.column_stack([d1, d1[:, None] - np.cumsum(gaps, axis=1)])
        theta = rng.normal(0.0, 1.0, (n, 1))
        cum = _sigmoid(theta[:, :, None] * a[None, :, None] + d[None, :, :])
        y = (rng.random((n, j, 1)) < cum).sum(axis=2).astype(np.int64)
        df = pd.DataFrame(y, columns=cols)
    elif case.model == "3pl":
        # Moderate-to-high slopes and hard-to-moderate difficulties keep asymptotes interior
        # (matching the V11a design in validation/r/m3_reference.R).
        a = rng.lognormal(0.4, 0.2, j)
        d = rng.uniform(-1.5, 0.5, j)
        g = rng.uniform(0.12, 0.25, j)
        theta = rng.normal(0.0, 1.0, (n, 1))
        p = g[None, :] + (1.0 - g[None, :]) * _sigmoid(theta * a[None, :] + d[None, :])
        y = (rng.random(p.shape) < p).astype(np.int64)
        df = pd.DataFrame(y, columns=cols)
    elif case.model == "twogroup_2pl":
        n_per = n // 2
        a_ref = rng.lognormal(0.0, 0.3, j)
        d_ref = rng.normal(0.0, 1.0, j)
        a_foc = a_ref.copy()
        d_foc = d_ref.copy()
        n_non = j - case.n_anchors
        d_foc[case.n_anchors :] = d_ref[case.n_anchors :] + rng.normal(0.0, 0.4, n_non)
        a_foc[case.n_anchors :] = a_ref[case.n_anchors :] * np.exp(rng.normal(0.0, 0.15, n_non))

        theta_ref = rng.normal(0.0, 1.0, (n_per, 1))
        theta_foc = rng.normal(0.5, float(np.sqrt(1.2)), (n_per, 1))
        p_ref = _sigmoid(theta_ref * a_ref[None, :] + d_ref[None, :])
        p_foc = _sigmoid(theta_foc * a_foc[None, :] + d_foc[None, :])
        y_ref = (rng.random(p_ref.shape) < p_ref).astype(np.int64)
        y_foc = (rng.random(p_foc.shape) < p_foc).astype(np.int64)
        y = np.vstack([y_ref, y_foc])
        df = pd.DataFrame(y, columns=cols)
        df.insert(0, "group", np.repeat(["ref", "focal"], n_per))
    elif case.model == "manygroups_2pl":
        n_per = n // case.n_groups
        a = rng.lognormal(0.0, 0.3, j)
        d = rng.normal(0.0, 1.0, j)
        means = np.concatenate([[0.0], rng.normal(0.0, 0.5, case.n_groups - 1)])
        sds = np.concatenate([[1.0], rng.uniform(0.8, 1.25, case.n_groups - 1)])
        theta = means[:, None] + sds[:, None] * rng.normal(0.0, 1.0, (case.n_groups, n_per))
        p = _sigmoid(theta.reshape(-1, 1) * a[None, :] + d[None, :])
        y = (rng.random(p.shape) < p).astype(np.int64)
        df = pd.DataFrame(y, columns=cols)
        df.insert(0, "group", np.repeat(case.group_names, n_per))
    else:  # pragma: no cover - exhaustive Literal guard
        raise ValueError(f"Unknown model kind: {case.model}")

    _validate_categories(df, case)
    return df


def _validate_categories(df: pd.DataFrame, case: BenchmarkCase) -> None:
    expected = set(range(case.n_categories))
    if case.n_groups > 1:
        for grp, sub in df.groupby("group", sort=False):
            for col in case.item_names:
                obs = set(sub[col].unique().tolist())
                if obs != expected:
                    raise RuntimeError(
                        f"Case {case.name} group {grp!r} item {col} observed {sorted(obs)}, "
                        f"expected {sorted(expected)}."
                    )
    else:
        for col in case.item_names:
            obs = set(df[col].unique().tolist())
            if obs != expected:
                raise RuntimeError(
                    f"Case {case.name} item {col} observed {sorted(obs)}, "
                    f"expected {sorted(expected)}."
                )


def _count_unique_patterns(df: pd.DataFrame) -> int:
    return len(df.drop_duplicates())


def resolve_cases(
    case_names: Sequence[str] | None, *, quick: bool = False, many_groups: bool = False
) -> list[BenchmarkCase]:
    """Select benchmark cases from CLI arguments."""
    if many_groups and not case_names:
        return list(MANY_GROUP_CASES)
    if case_names:
        unknown = [name for name in case_names if name not in CASES_BY_NAME]
        if unknown:
            raise ValueError(
                f"Unknown case(s): {unknown}. Available: {list(CASES_BY_NAME.keys())}."
            )
        return [CASES_BY_NAME[name] for name in case_names]
    if quick:
        return [CASES_BY_NAME[name] for name in QUICK_CASE_NAMES]
    return list(CASES)


def generate_datasets(
    cases: Sequence[BenchmarkCase] | None = None,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> dict[str, object]:
    """Generate CSV files and ``manifest.json`` for ``cases`` into ``data_dir``."""
    selected = list(CASES) if cases is None else list(cases)
    data_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    for case in selected:
        df = simulate_case(case)
        csv_name = f"{case.name}.csv"
        csv_path = data_dir / csv_name
        df.to_csv(csv_path, index=False)
        entry: dict[str, object] = {
            **asdict(case),
            "csv": csv_name,
            "items": case.item_names,
            "anchors": case.anchors,
            "groups": case.group_names,
            "n_patterns": _count_unique_patterns(df),
        }
        entries.append(entry)

    manifest: dict[str, object] = {"cases": entries}
    manifest_path = data_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Output directory for CSV files and manifest.json (default: benchmarks/data).",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=list(CASES_BY_NAME.keys()),
        help="Specific cases to generate (default: all cases, or quick subset with --quick).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Generate only the quick benchmark subset.",
    )
    parser.add_argument(
        "--many-groups",
        action="store_true",
        help="Generate only the many-group cases (G = 10, 50, 200).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    args = _parse_args(argv)
    cases = resolve_cases(args.cases, quick=args.quick, many_groups=args.many_groups)
    manifest = generate_datasets(cases, data_dir=args.data_dir)
    for entry in manifest["cases"]:  # type: ignore[attr-defined]
        print(
            f"wrote {args.data_dir / entry['csv']} "
            f"(N={entry['n_persons']}, J={entry['n_items']}, "
            f"patterns={entry['n_patterns']})"
        )


if __name__ == "__main__":
    main()
