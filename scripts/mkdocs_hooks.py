"""MkDocs hook: build the site from the same Markdown files that GitHub renders.

- Drop the in-page language switcher line (``English | [日本語](X.ja.md)``); the site has its own.
- Point links that leave ``docs/`` (``../benchmarks/...``) to the file on GitHub.
"""

from __future__ import annotations

import re
from typing import Any

GITHUB_BLOB = "https://github.com/takuizum/flexpyirt/blob/main/"
_SWITCHER = re.compile(
    r"^(English \| \[日本語\]\([^)]*\)|\[English\]\([^)]*\) \| 日本語)\s*$", re.M
)
_OUTSIDE = re.compile(r"\]\(\.\./([^)]+)\)")


def on_page_markdown(markdown: str, **kwargs: Any) -> str:
    """Rewrite one page's Markdown before it is rendered."""
    markdown = _SWITCHER.sub("", markdown, count=1)
    return _OUTSIDE.sub(lambda m: f"]({GITHUB_BLOB}{m.group(1)})", markdown)
