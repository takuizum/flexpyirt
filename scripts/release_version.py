"""Release helper for the develop -> main flow (see CONTRIBUTING.md).

The package version comes from git tags (hatch-vcs), so releasing means creating the next
``vX.Y.Z`` tag on main; nothing is committed.

Commands
--------
``parse-title TITLE``
    Print the bump level (``major`` / ``minor`` / ``patch``) of a release PR title of the form
    ``release(<level>): <summary>``; exit 1 if the title does not match.
``next --bump LEVEL [--latest vX.Y.Z]``
    Print the version to release: ``--latest`` bumped at ``LEVEL``, or ``INITIAL_VERSION`` when
    there is no release tag yet.
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Literal

Bump = Literal["major", "minor", "patch"]

INITIAL_VERSION = "0.0.1"
TITLE_RE = re.compile(r"^release\((major|minor|patch)\): \S.*$")
VERSION_RE = re.compile(r"^v?(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


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
    """Return ``version`` (``X.Y.Z`` or ``vX.Y.Z``) bumped at ``level``, without the ``v``."""
    m = VERSION_RE.match(version)
    if m is None:
        raise ValueError(f"Version must be MAJOR.MINOR.PATCH, got {version!r}.")
    major, minor, patch = (int(x) for x in m.groups())
    if level == "major":
        return f"{major + 1}.0.0"
    if level == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def next_version(level: Bump, latest: str | None) -> str:
    """Return the next release version; the first release is ``INITIAL_VERSION``."""
    return INITIAL_VERSION if not latest else bump_version(latest, level)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Release helper (see module docstring).")
    sub = parser.add_subparsers(dest="command", required=True)
    p_title = sub.add_parser("parse-title")
    p_title.add_argument("title")
    p_next = sub.add_parser("next")
    p_next.add_argument("--bump", choices=["major", "minor", "patch"], required=True)
    p_next.add_argument("--latest", default="")
    args = parser.parse_args(argv)
    try:
        if args.command == "parse-title":
            print(parse_title(args.title))
        else:
            print(next_version(args.bump, args.latest or None))
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
