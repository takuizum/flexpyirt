"""MkDocs hook: build the site from the same Markdown files that GitHub renders.

- Drop the in-page language switcher line (``English | [日本語](X.ja.md)``); the site has its own.
- Replace ``<!-- include: packages/<name>/README.md -->`` with that README (without its title and
  language switcher), so a package's PyPI page is also its page on the site. On GitHub the comment
  is invisible and the page links to the README instead.
- Point links that leave ``docs/`` (``../benchmarks/...``) to the file on GitHub.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

GITHUB_BLOB = "https://github.com/takuizum/flexpyirt/blob/main/"
ROOT = Path(__file__).resolve().parents[1]
_SWITCHER = re.compile(
    r"^(English \| \[日本語\]\([^)]*\)|\[English\]\([^)]*\) \| 日本語)\s*$", re.M
)
_INCLUDE = re.compile(r"^<!-- include: (packages/[\w-]+/README(?:\.ja)?\.md) -->\s*$", re.M)
_TITLE = re.compile(r"\A# [^\n]*\n")
_OUTSIDE = re.compile(r"\]\(\.\./([^)]+)\)")


def included_readme(path: str) -> str:
    """Return a package README without its title line and language switcher.

    Parameters
    ----------
    path
        Path of the README relative to the repository root.
    """
    text = (ROOT / path).read_text(encoding="utf-8")
    text, n_titles = _TITLE.subn("", text, count=1)
    text, n_switchers = _SWITCHER.subn("", text, count=1)
    if n_titles != 1 or n_switchers != 1:
        raise ValueError(f"{path}: expected a '# title' first line and a language switcher line")
    return text.strip() + "\n"


def on_page_markdown(markdown: str, **kwargs: Any) -> str:
    """Rewrite one page's Markdown before it is rendered."""
    markdown = _SWITCHER.sub("", markdown, count=1)
    markdown = _INCLUDE.sub(lambda m: included_readme(m.group(1)), markdown)
    return _OUTSIDE.sub(lambda m: f"]({GITHUB_BLOB}{m.group(1)})", markdown)
