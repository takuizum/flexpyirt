"""Release helper for the develop -> main flow (see CONTRIBUTING.md).

Commands
--------
``parse-title TITLE``
    Print the bump level (``major`` / ``minor`` / ``patch``) of a release PR title of the form
    ``release(<level>): <summary>``; exit 1 if the title does not match.
``next --bump LEVEL [--initial]``
    Print the version to release. With ``--initial`` (no ``v*`` tag exists yet) the current
    version is released as is; otherwise it is bumped.
``write VERSION``
    Write ``VERSION`` into ``[project].version`` of pyproject.toml, keeping the rest of the file.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path
from typing import Literal

Bump = Literal["major", "minor", "patch"]

TITLE_RE = re.compile(r"^release\((major|minor|patch)\): \S.*$")
VERSION_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def parse_title(title: str) -> Bump:
    """Return the bump level of a release PR title, or raise ``ValueError``."""
    m = TITLE_RE.match(title.strip())
    if m is None:
        raise ValueError(
            f"Release PR title must look like 'release(major|minor|patch): summary', got {title!r}."
        )
    level = m.group(1)
    assert level in ("major", "minor", "patch")
    return level  # type: ignore[return-value]


def bump_version(version: str, level: Bump) -> str:
    """Return ``version`` bumped at ``level`` (semantic versioning, lower parts reset to 0)."""
    m = VERSION_RE.match(version)
    if m is None:
        raise ValueError(f"Version must be MAJOR.MINOR.PATCH, got {version!r}.")
    major, minor, patch = (int(x) for x in m.groups())
    if level == "major":
        return f"{major + 1}.0.0"
    if level == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def read_version(path: Path = PYPROJECT) -> str:
    """Return ``[project].version`` of a pyproject.toml."""
    return str(tomllib.loads(path.read_text())["project"]["version"])


def write_version(version: str, path: Path = PYPROJECT) -> None:
    """Replace ``[project].version`` in a pyproject.toml, keeping formatting elsewhere."""
    if VERSION_RE.match(version) is None:
        raise ValueError(f"Version must be MAJOR.MINOR.PATCH, got {version!r}.")
    text = path.read_text()
    project = re.search(r"^\[project\]\s*$(.*?)(?=^\[|\Z)", text, flags=re.M | re.S)
    if project is None:
        raise ValueError("pyproject.toml has no [project] table.")
    body, n = re.subn(
        r'^version\s*=\s*"[^"]*"', f'version = "{version}"', project.group(1), count=1, flags=re.M
    )
    if n != 1:
        raise ValueError("[project] table has no version line.")
    path.write_text(text[: project.start(1)] + body + text[project.end(1) :])


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_title = sub.add_parser("parse-title")
    p_title.add_argument("title")
    p_next = sub.add_parser("next")
    p_next.add_argument("--bump", choices=["major", "minor", "patch"], required=True)
    p_next.add_argument("--initial", action="store_true")
    p_write = sub.add_parser("write")
    p_write.add_argument("version")
    args = parser.parse_args(argv)

    try:
        if args.command == "parse-title":
            print(parse_title(args.title))
        elif args.command == "next":
            current = read_version()
            print(current if args.initial else bump_version(current, args.bump))
        else:
            write_version(args.version)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
