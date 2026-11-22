"""MkDocs hook: render Sphinx-style ``:math:`...``` roles from docstrings with MathJax.

mkdocstrings renders the role as ``:math:<code>...</code>``; turn it into an arithmatex span so
the MathJax configuration in ``docs/javascripts/mathjax.js`` typesets it.
"""

from __future__ import annotations

import re
from typing import Any

_MATH_ROLE = re.compile(r":math:<code>(.*?)</code>", re.DOTALL)


def on_page_content(html: str, **kwargs: Any) -> str:
    """Replace ``:math:`` roles with inline arithmatex spans."""
    return _MATH_ROLE.sub(r'<span class="arithmatex">\\(\1\\)</span>', html)
