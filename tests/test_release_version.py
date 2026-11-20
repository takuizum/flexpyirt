"""Tests for scripts/release_version.py (release flow helper)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "release_version.py"
    spec = importlib.util.spec_from_file_location("release_version", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["release_version"] = module
    spec.loader.exec_module(module)
    return module


rv = _load()


@pytest.mark.parametrize(
    ("title", "level"),
    [
        ("release(major): drop Python 3.11", "major"),
        ("release(minor): add GRM", "minor"),
        ("release(patch): fix SE", "patch"),
        ("  release(patch): trailing spaces ok  ", "patch"),
    ],
)
def test_parse_title(title: str, level: str) -> None:
    assert rv.parse_title(title) == level


@pytest.mark.parametrize(
    "title",
    [
        "release(minor):",
        "release(minor) add GRM",
        "Release(minor): x",
        "release(huge): x",
        "feat: x",
    ],
)
def test_parse_title_rejects(title: str) -> None:
    with pytest.raises(ValueError, match="release"):
        rv.parse_title(title)


@pytest.mark.parametrize(
    ("version", "level", "expected"),
    [
        ("0.0.1", "patch", "0.0.2"),
        ("0.0.1", "minor", "0.1.0"),
        ("0.3.7", "major", "1.0.0"),
        ("1.9.9", "minor", "1.10.0"),
    ],
)
def test_bump_version(version: str, level: str, expected: str) -> None:
    assert rv.bump_version(version, level) == expected


def test_bump_rejects_non_semver() -> None:
    with pytest.raises(ValueError):
        rv.bump_version("1.0", "patch")


def test_write_version_only_touches_project_table(tmp_path: Path) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_text(
        '[build-system]\nversion = "keep"\n\n[project]\nname = "x"\nversion = "0.0.1"\n'
        'description = "y"\n\n[tool.other]\nversion = "keep-too"\n'
    )
    rv.write_version("0.2.0", path)
    text = path.read_text()
    assert 'version = "0.2.0"' in text
    assert text.count('"keep"') == 1 and text.count('"keep-too"') == 1
    assert rv.read_version(path) == "0.2.0"


def test_cli_next_initial_keeps_current(capsys: pytest.CaptureFixture[str]) -> None:
    current = rv.read_version()
    assert rv.main(["next", "--bump", "minor", "--initial"]) == 0
    assert capsys.readouterr().out.strip() == current
    assert rv.main(["next", "--bump", "patch"]) == 0
    assert capsys.readouterr().out.strip() == rv.bump_version(current, "patch")


def test_cli_parse_title_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert rv.main(["parse-title", "not a release"]) == 1
    assert "error:" in capsys.readouterr().err


# ---- scripts/check_pr.py -------------------------------------------------------------------


def _load_check_pr() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "check_pr.py"
    spec = importlib.util.spec_from_file_location("check_pr", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_pr"] = module
    spec.loader.exec_module(module)
    return module


cp = _load_check_pr()


@pytest.fixture
def issues(monkeypatch: pytest.MonkeyPatch) -> None:
    states = {12: "open", 13: "closed"}
    monkeypatch.setattr(cp, "issue_state", lambda repo, n: states.get(n))


def test_release_pr_rules(issues: None) -> None:
    assert cp.check("main", "develop", "release(minor): add GRM", "", "o/r") == []
    errs = cp.check("main", "feature/x", "add GRM", "", "o/r")
    assert len(errs) == 2


def test_feature_pr_rules(issues: None) -> None:
    assert cp.check("develop", "feature/docs-readme", "Docs", "", "o/r") == []
    assert cp.check("develop", "Feature/Docs", "Docs", "", "o/r")
    assert cp.check("develop", "chore/x", "x", "", "o/r")
    assert cp.check("other", "feature/x", "x", "", "o/r")


def test_fix_pr_rules(issues: None) -> None:
    assert cp.check("develop", "fix/12-oakes-se", "Fix SE", "Closes #12", "o/r") == []
    assert cp.check("develop", "fix/12-oakes-se", "Fix SE", "fixes #12\nmore", "o/r") == []
    assert any("Closes #12" in e for e in cp.check("develop", "fix/12-x", "t", "See #12", "o/r"))
    assert any(
        "Closes #12" in e for e in cp.check("develop", "fix/12-x", "t", "Closes #120", "o/r")
    )
    assert any("closed" in e for e in cp.check("develop", "fix/13-x", "t", "Closes #13", "o/r"))
    assert any(
        "does not exist" in e for e in cp.check("develop", "fix/99-x", "t", "Closes #99", "o/r")
    )
    assert cp.check("develop", "fix/oakes", "t", "", "o/r")
