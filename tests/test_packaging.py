"""flexpyirt's dependency on flexpyctt, which is released from its own repository (§16.24).

These checks read the repository's configuration only (no JAX, no fits).
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
# Present in a checkout; the sdist leaves .github out.
needs_workflows = pytest.mark.skipif(not WORKFLOWS.exists(), reason="not a repository checkout")


def _pyproject() -> dict[str, Any]:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _requirement_names(requirements: list[str]) -> set[str]:
    return {re.split(r"[\s<>=!~\[;]", r, maxsplit=1)[0].lower() for r in requirements}


def test_flexpyctt_is_required_within_one_minor_version() -> None:
    """Before 1.0 a flexpyctt minor release may break the API, so the range stays inside one."""
    (requirement,) = [
        r for r in _pyproject()["project"]["dependencies"] if r.startswith("flexpyctt")
    ]
    match = re.fullmatch(r"flexpyctt>=0\.(\d+)(?:\.\d+)?,<0\.(\d+)", requirement)
    assert match is not None, requirement
    assert int(match.group(2)) == int(match.group(1)) + 1


def test_both_come_from_the_index_and_flexpyequate_is_for_development_only() -> None:
    config = _pyproject()
    uv = config["tool"].get("uv", {})
    assert "workspace" not in uv
    assert not {"flexpyctt", "flexpyequate"} & set(uv.get("sources", {}))
    assert not (ROOT / "packages").exists()
    runtime = _requirement_names(config["project"]["dependencies"])
    for extra in config["project"]["optional-dependencies"].values():
        runtime |= _requirement_names(extra)
    assert "flexpyctt" in runtime
    # tests/test_equating.py uses flexpyequate as an independent check; flexpyirt does not.
    assert "flexpyequate" not in runtime
    assert "flexpyequate" in _requirement_names(config["dependency-groups"]["dev"])


@needs_workflows
def test_release_workflow_publishes_flexpyirt_only() -> None:
    text = (WORKFLOWS / "release.yml").read_text(encoding="utf-8")
    assert "--package" not in text
    assert text.count("uses: pypa/gh-action-pypi-publish@release/v1") == 1
    assert text.count("id-token: write") == 1
    assert "url: https://pypi.org/project/flexpyirt/" in text
