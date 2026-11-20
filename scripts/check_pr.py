"""Check that a pull request follows the branch flow in CONTRIBUTING.md.

Reads ``BASE``, ``HEAD``, ``TITLE``, ``BODY`` and ``REPO`` from the environment (set by
.github/workflows/pr-rules.yml) and exits 1 with a message for each violation.

Rules
-----
- PRs into ``main`` come from ``develop`` and have a title ``release(major|minor|patch): ...``.
- PRs into ``develop`` come from ``feature/<slug>`` or ``fix/<issue>-<slug>``.
- A ``fix/<issue>-...`` PR links that issue with a closing keyword (``Closes #<issue>``), and the
  issue exists and is open.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

TITLE_RE = re.compile(r"^release\((major|minor|patch)\): \S")
FEATURE_RE = re.compile(r"^feature/[a-z0-9][a-z0-9._-]*$")
FIX_RE = re.compile(r"^fix/(\d+)-[a-z0-9][a-z0-9._-]*$")
CLOSING_RE = r"\b(close[sd]?|fix(e[sd])?|resolve[sd]?)\s+#{n}\b"


def issue_state(repo: str, number: int) -> str | None:
    """Return ``"open"`` / ``"closed"`` for an issue, or ``None`` if it is missing or a PR."""
    out = subprocess.run(
        ["gh", "api", f"repos/{repo}/issues/{number}"], capture_output=True, text=True, check=False
    )
    if out.returncode != 0:
        return None
    data = json.loads(out.stdout)
    if "pull_request" in data:
        return None
    return str(data["state"])


def check(base: str, head: str, title: str, body: str, repo: str) -> list[str]:
    """Return the list of rule violations (empty if the PR is fine)."""
    errors: list[str] = []
    if base == "main":
        if head != "develop":
            errors.append(f"PRs into main must come from develop (got {head!r}).")
        if TITLE_RE.match(title.strip()) is None:
            errors.append(
                "Release PR title must be 'release(major|minor|patch): summary' "
                f"(got {title!r}); the level decides the version bump."
            )
    elif base == "develop":
        fix = FIX_RE.match(head)
        if fix is None and FEATURE_RE.match(head) is None:
            errors.append(
                f"Branch {head!r} must be named feature/<slug> or fix/<issue>-<slug> "
                "(lowercase letters, digits, '.', '_', '-')."
            )
        if fix is not None:
            n = int(fix.group(1))
            if re.search(CLOSING_RE.format(n=n), body or "", flags=re.I) is None:
                errors.append(
                    f"Fix PRs must link their issue: add 'Closes #{n}' to the description."
                )
            state = issue_state(repo, n)
            if state is None:
                errors.append(f"Issue #{n} (from the branch name) does not exist.")
            elif state != "open":
                errors.append(f"Issue #{n} is {state}; fix branches address open issues.")
    else:
        errors.append(f"PRs must target develop (or main for releases), not {base!r}.")
    return errors


def main() -> int:
    """Entry point for the workflow."""
    env = os.environ
    errors = check(env["BASE"], env["HEAD"], env.get("TITLE", ""), env.get("BODY", ""), env["REPO"])
    for e in errors:
        print(f"::error::{e}")
    if not errors:
        print("PR follows the branch flow.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
