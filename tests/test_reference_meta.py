"""Regression tests for mirt reference JSON metadata checks across all loaders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests import (
    _reference,
    test_mirt_m2,
    test_mirt_m3,
    test_mirt_m3_gpcm,
    test_mirt_m3_priors,
    test_mirt_m4_irtpars,
)
from tests._reference import REFERENCE_DIR, check_reference_meta


def test_check_reference_meta_accepts_valid_and_rejects_invalid() -> None:
    valid_full: dict[str, Any] = {
        "case": "demo",
        "converged": True,
        "converged_richardson": True,
        "theta_lim": [-6, 6],
        "grid_check": [-6, 6],
    }
    check_reference_meta(valid_full, require_richardson=True, require_grid_check=True)

    valid_minimal: dict[str, Any] = {
        "case": "demo_min",
        "converged": True,
        "theta_lim": [-6, 6],
    }
    check_reference_meta(valid_minimal)
    with pytest.raises(AssertionError, match="Richardson"):
        check_reference_meta(valid_minimal, require_richardson=True)
    with pytest.raises(AssertionError, match="grid_check"):
        check_reference_meta(valid_minimal, require_grid_check=True)

    with pytest.raises(AssertionError, match="Oakes"):
        check_reference_meta({**valid_full, "converged": False})
    with pytest.raises(AssertionError, match="Oakes"):
        check_reference_meta({"case": "missing_conv", "theta_lim": [-6, 6]})
    with pytest.raises(AssertionError, match="Richardson"):
        check_reference_meta({**valid_full, "converged_richardson": False})
    with pytest.raises(AssertionError, match="grid_check"):
        check_reference_meta({**valid_full, "grid_check": [-4, 4]})


def test_all_committed_reference_jsons_pass_meta_checks() -> None:
    files = sorted(REFERENCE_DIR.glob("*.json"))
    assert len(files) >= 13
    for path in files:
        raw = json.loads(path.read_text())
        check_reference_meta(raw["meta"])


@pytest.mark.parametrize(
    ("module", "case_name"),
    [
        (_reference, "v1_lsat7_2pl"),
        (test_mirt_m2, "v3_science_grm"),
        (test_mirt_m3, "v11_3pl"),
        (test_mirt_m3_gpcm, "v8_science_gpcm"),
        (test_mirt_m3_priors, "v7_2pl_map"),
    ],
)
@pytest.mark.parametrize(
    "bad_patch",
    [
        {"converged": False},
        {"converged_richardson": False},
        {"grid_check": [-4, 4]},
    ],
)
def test_every_mirt_loader_rejects_bad_reference_meta(
    module: Any,
    case_name: str,
    bad_patch: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = json.loads((REFERENCE_DIR / f"{case_name}.json").read_text())
    raw["meta"].update(bad_patch)
    (tmp_path / f"{case_name}.json").write_text(json.dumps(raw))

    monkeypatch.setattr(module, "REFERENCE_DIR", tmp_path)
    clear = getattr(module.load, "cache_clear", None)
    if clear is not None:
        clear()
    try:
        with pytest.raises(AssertionError):
            module.load(case_name)
    finally:
        if clear is not None:
            clear()


@pytest.mark.parametrize(
    ("target", "bad_patch"),
    [
        ("top", {"converged": False}),
        ("top", {"grid_check": [-4, 4]}),
        ("case", {"converged": False}),
        ("case", {"grid_check": [-4, 4]}),
    ],
)
def test_v12_irtpars_loader_rejects_bad_reference_meta(
    target: str,
    bad_patch: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = json.loads((REFERENCE_DIR / test_mirt_m4_irtpars.REFERENCE_FILE).read_text())
    if target == "top":
        raw["meta"].update(bad_patch)
    else:
        raw["cases"]["lsat7_2pl"]["meta"].update(bad_patch)
    (tmp_path / test_mirt_m4_irtpars.REFERENCE_FILE).write_text(json.dumps(raw))

    monkeypatch.setattr(test_mirt_m4_irtpars, "REFERENCE_DIR", tmp_path)
    test_mirt_m4_irtpars._bundle.cache_clear()
    test_mirt_m4_irtpars.load.cache_clear()
    try:
        with pytest.raises(AssertionError):
            test_mirt_m4_irtpars.load("lsat7_2pl")
    finally:
        test_mirt_m4_irtpars._bundle.cache_clear()
        test_mirt_m4_irtpars.load.cache_clear()
