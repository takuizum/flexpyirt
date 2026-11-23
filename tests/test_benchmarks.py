"""Smoke test for the Python benchmark pipeline (runs in --quick mode on the smallest case)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks import data as bench_data
from benchmarks import report as bench_report
from benchmarks import run_flexpyirt


@pytest.mark.slow
def test_python_benchmark_pipeline_quick_smallest_case(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"
    flex_out = data_dir / "flexpyirt_raw.json"

    bench_data.main(["--data-dir", str(data_dir), "--cases", "2pl_n1000_j20"])
    assert (data_dir / "2pl_n1000_j20.csv").exists()
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    assert [c["name"] for c in manifest["cases"]] == ["2pl_n1000_j20"]

    run_flexpyirt.main(
        [
            "--data-dir",
            str(data_dir),
            "--out",
            str(flex_out),
            "--quick",
            "--cases",
            "2pl_n1000_j20",
        ]
    )
    raw = json.loads(flex_out.read_text(encoding="utf-8"))
    assert raw["meta"]["jax_platforms"] == "cpu"
    assert raw["meta"]["jax_devices"]
    assert raw["meta"]["settings"]["quick"] is True
    assert len(raw["runs"]) == 4

    seen = set()
    logliks: list[float] = []
    for rec in raw["runs"]:
        seen.add((rec["case"], rec["se"], rec["squarem"]))
        assert rec["converged"] is True
        assert rec["cold_s"] > 0.0
        assert rec["warm_s"] > 0.0
        assert rec["n_iter"] >= 1
        assert rec["n_em_evaluations"] >= rec["n_iter"]
        logliks.append(float(rec["loglik"]))
    assert seen == {
        ("2pl_n1000_j20", "none", False),
        ("2pl_n1000_j20", "none", True),
        ("2pl_n1000_j20", "oakes", False),
        ("2pl_n1000_j20", "oakes", True),
    }
    assert max(logliks) - min(logliks) < 1e-3

    bench_report.main(
        [
            "--flex",
            str(flex_out),
            "--no-mirt",
            "--results-dir",
            str(results_dir),
            "--stem",
            "smoke",
        ]
    )
    merged = json.loads((results_dir / "smoke.json").read_text(encoding="utf-8"))
    assert len(merged["results"]) == 2
    assert merged["meta"]["mirt"] is None

    md = (results_dir / "smoke.md").read_text(encoding="utf-8")
    assert "2PL (1000 × 20)" in md
    assert "| plain cold (s) |" in md
    assert "Oakes" in md
