"""Provenance of the mirt references (VALIDATION.md, "R and mirt versions").

Every ``tests/reference/*.json`` is generated with R mirt installed from GitHub
(philchalmers/mirt) and records its version and commit SHA; every generator script loads mirt
through ``validation/r/setup.R`` and is run by ``validation/r/run_references.sh``. Also tests
``scripts/compare_references.py``, which the scheduled reference check uses.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests._reference import REFERENCE_DIR

ROOT = Path(__file__).resolve().parents[1]
R_DIR = ROOT / "validation" / "r"
SHA = re.compile(r"[0-9a-f]{40}")


def _metas(node: Any, path: str = "") -> Iterator[tuple[str, dict[str, Any]]]:
    if isinstance(node, dict):
        meta = node.get("meta")
        if isinstance(meta, dict):
            yield path or "/", meta
        for key, value in node.items():
            if key != "meta":
                yield from _metas(value, f"{path}/{key}")


REFERENCES = sorted(REFERENCE_DIR.glob("*.json"))
MIRT_SCRIPTS = sorted(
    p for p in R_DIR.glob("*_reference.R") if "library(mirt)" in p.read_text(encoding="utf-8")
)


@pytest.mark.parametrize("path", REFERENCES, ids=lambda p: p.stem)
def test_reference_records_mirt_commit(path: Path) -> None:
    raw = json.loads(path.read_text())
    metas = dict(_metas(raw))
    top = metas["/"]
    assert isinstance(top.get("mirt_version"), str), f"{path.name}: meta.mirt_version missing"
    sha = top.get("mirt_sha")
    assert isinstance(sha, str) and SHA.fullmatch(sha), (
        f"{path.name}: meta.mirt_sha must be the 40-hex commit of a GitHub mirt, got {sha!r} "
        "(regenerate with validation/r/install_mirt.R + run_references.sh)"
    )
    for where, meta in metas.items():
        if "mirt_version" in meta:
            assert meta["mirt_version"] == top["mirt_version"], f"{path.name}{where}"
            assert meta.get("mirt_sha") == sha, f"{path.name}{where}: mirt_sha differs from top"


def test_mirt_scripts_found() -> None:
    assert len(MIRT_SCRIPTS) >= 40


@pytest.mark.parametrize("path", MIRT_SCRIPTS, ids=lambda p: p.stem)
def test_mirt_script_uses_shared_setup(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    setup = text.find('source(file.path("validation", "r", "setup.R"))')
    assert setup >= 0, f"{path.name} must source validation/r/setup.R"
    assert setup < text.find("library(mirt)"), f"{path.name}: source setup.R before library(mirt)"
    assert "mirt_commit_sha()" in text, f"{path.name} must record mirt_sha = mirt_commit_sha()"
    assert 'packageVersion("mirt")' not in text, f"{path.name}: use mirt_version_string()"


def test_run_references_lists_every_mirt_script() -> None:
    runner = (R_DIR / "run_references.sh").read_text(encoding="utf-8")
    stages = re.findall(r"^STAGE\d=\((.*?)\)", runner, flags=re.MULTILINE | re.DOTALL)
    listed = [name for stage in stages for name in stage.split()]
    expected = sorted(p.name.removesuffix("_reference.R") for p in MIRT_SCRIPTS)
    assert sorted(listed) == expected


def _load_compare() -> ModuleType:
    path = ROOT / "scripts" / "compare_references.py"
    spec = importlib.util.spec_from_file_location("compare_references", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["compare_references"] = module
    spec.loader.exec_module(module)
    return module


cr = _load_compare()


def _write(directory: Path, name: str, payload: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(payload))


def test_compare_references_tolerances(tmp_path: Path) -> None:
    base = {
        "meta": {"mirt_version": "1.48", "mirt_sha": "a" * 40, "iterations": 10},
        "loglik": -1000.0,
        "params": {"q1": {"a": 1.0, "a_se": 0.1, "d": "Inf"}},
        "eap": [0.5, -0.5],
    }
    new = json.loads(json.dumps(base))
    new["meta"] = {"mirt_version": "1.49", "mirt_sha": "b" * 40, "iterations": 12}
    new["loglik"] = -1000.0005  # 5e-7 relative: within 1e-6
    new["params"]["q1"]["a_se"] = 0.10005  # 5e-4 relative: within 1e-3
    new["eap"] = [0.500005, -0.5]  # 5e-6 absolute: within 1e-5
    _write(tmp_path / "base", "v.json", base)
    _write(tmp_path / "new", "v.json", new)
    reports, added, removed = cr.compare_dirs(tmp_path / "base", tmp_path / "new")
    assert not reports[0].changed and not added and not removed
    assert reports[0].compared == 6

    new["loglik"] = -1000.01
    new["params"]["q1"]["a_se"] = 0.1002
    new["params"]["q1"]["d"] = 12.0
    new["eap"] = [0.5001, -0.5, 0.0]
    _write(tmp_path / "new", "v.json", new)
    _write(tmp_path / "new", "w.json", base)
    reports, added, removed = cr.compare_dirs(tmp_path / "base", tmp_path / "new")
    (report,) = reports
    assert [d[0] for d in report.drifts] == ["/loglik", "/params/q1/a_se"]
    assert report.structural == ["/eap: length 2 -> 3", "/params/q1/d: 'Inf' -> 12.0"]
    assert added == ["w.json"] and removed == []
    text = cr.format_report(
        reports, added, removed, base_dir=tmp_path / "base", new_dir=tmp_path / "new"
    )
    assert "`v.json`: 2 of 4 numbers drifted" in text and "added: `w.json`" in text
    assert cr.main(["--base", str(tmp_path / "base"), "--new", str(tmp_path / "new")]) == 1
    assert cr.main(["--base", str(tmp_path / "base"), "--new", str(tmp_path / "base")]) == 0
