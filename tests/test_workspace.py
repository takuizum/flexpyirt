"""The uv workspace: every distribution is built, tested and released the same way (§16.18).

These checks read the repository's configuration only (no JAX, no fits): a distribution added
under ``packages/`` must be wired into CI and the release workflow, share the version tag, and
the JAX-free ones must not depend on JAX or on flexpyirt.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
MEMBERS = sorted(p.parent for p in (ROOT / "packages").glob("*/pyproject.toml"))
WORKFLOWS = ROOT / ".github" / "workflows"
# Present in a checkout; the sdist leaves .github out.
needs_workflows = pytest.mark.skipif(not WORKFLOWS.exists(), reason="not a repository checkout")


def _pyproject(directory: Path) -> dict[str, Any]:
    with (directory / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _requirement_names(requirements: list[str]) -> set[str]:
    return {re.split(r"[\s<>=!~\[;]", r, maxsplit=1)[0].lower() for r in requirements}


def test_members_are_the_expected_distributions() -> None:
    assert [m.name for m in MEMBERS] == ["flexpyctt", "flexpyequate"]
    assert _pyproject(ROOT)["tool"]["uv"]["workspace"]["members"] == ["packages/*"]


@pytest.mark.parametrize("member", MEMBERS, ids=lambda m: m.name)
def test_member_is_a_jax_free_distribution_with_the_shared_version(member: Path) -> None:
    config = _pyproject(member)
    project = config["project"]
    assert project["name"] == member.name
    # NumPy / SciPy / pandas only: no JAX, and no other distribution of the workspace.
    assert _requirement_names(project["dependencies"]) == {"numpy", "scipy", "pandas"}
    assert "optional-dependencies" not in project
    # One version for every distribution: the repository's vX.Y.Z tag.
    assert project["dynamic"] == ["version"]
    version = config["tool"]["hatch"]["version"]
    assert version["source"] == "vcs"
    assert version["raw-options"]["root"] == "../.."
    assert config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        f"src/{member.name}"
    ]
    for name in ("README.md", "README.ja.md", "LICENSE", f"src/{member.name}/py.typed"):
        assert (member / name).exists(), name


def test_flexpyirt_depends_on_flexpyctt_only() -> None:
    """flexpyequate is a development dependency of the root, never a runtime one."""
    config = _pyproject(ROOT)
    runtime = _requirement_names(config["project"]["dependencies"])
    for extra in config["project"]["optional-dependencies"].values():
        runtime |= _requirement_names(extra)
    assert "flexpyctt" in runtime
    assert "flexpyequate" not in runtime
    assert "flexpyequate" in _requirement_names(config["dependency-groups"]["dev"])


@needs_workflows
@pytest.mark.parametrize("name", ["flexpyirt", *(m.name for m in MEMBERS)])
def test_release_workflow_builds_and_publishes(name: str) -> None:
    text = (WORKFLOWS / "release.yml").read_text(encoding="utf-8")
    # The dry run (plan job) and the publish job both build it.
    assert text.count(f"uv build --package {name}\n") == 2
    # Both jobs check the file names of all distributions.
    loops = re.findall(r"for name in ([a-z ]+); do", text)
    assert len(loops) == 2 and all(name in loop.split() for loop in loops)
    # Its files go to a PyPI job of its own, which uploads only them.
    assert f"cp dist/{name}-* dist-{name}/" in text
    assert text.count(f"name: dist-{name}\n") == 2  # upload and download
    assert f"url: https://pypi.org/project/{name}/" in text


@needs_workflows
def test_release_workflow_publishes_flexpyctt_before_flexpyirt() -> None:
    text = (WORKFLOWS / "release.yml").read_text(encoding="utf-8")
    jobs = dict(re.findall(r"^  ([a-z-]+):\n    needs: \[([a-z, -]+)\]", text, re.M))
    assert jobs["pypi-flexpyctt"] == "plan, publish"
    assert jobs["pypi-flexpyequate"] == "plan, publish"
    assert jobs["pypi"] == "plan, publish, pypi-flexpyctt"
    assert text.count("uses: pypa/gh-action-pypi-publish@release/v1") == 3
    assert text.count("id-token: write") == 3


@needs_workflows
@pytest.mark.parametrize("member", MEMBERS, ids=lambda m: m.name)
def test_ci_tests_member_with_and_without_jax(member: Path) -> None:
    text = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    name = member.name
    assert f"run: uv run pytest packages/{name} " in text
    assert f"name: {name} without JAX (${{{{ matrix.python-version }}}})" in text
    assert f"run: uv sync --locked --package {name}\n" in text
    assert f"run: uv run --no-sync pytest packages/{name} " in text
    assert f"uv build --package {name}\n" in text
    # The required status checks keep their names.
    assert re.search(r"^  test:\n", text, re.M)
