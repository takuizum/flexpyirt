"""Package-level checks: public names, version, README and the no-JAX rule (issue #87)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import flexpyequate

README_DIR = Path(__file__).resolve().parents[1]
BLOCK_RE = re.compile(r"^```python\n(.*?)^```", re.M | re.S)


def test_public_names_are_importable() -> None:
    for name in flexpyequate.__all__:
        assert hasattr(flexpyequate, name), name
    assert flexpyequate.__version__


def test_import_does_not_load_jax_or_the_other_distributions() -> None:
    """flexpyequate must work where JAX, flexpyirt and flexpyctt are not installed."""
    code = (
        "import sys, flexpyequate; "
        "bad = [m for m in ('jax', 'flexpyirt', 'flexpyctt') if m in sys.modules]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def _code_only(block: str) -> list[str]:
    """Return the block's lines without comments and blank lines."""
    lines = [re.sub(r"\s+#.*$", "", line).rstrip() for line in block.splitlines()]
    return [line for line in lines if line.strip() and not line.strip().startswith("#")]


def test_readme_languages_have_the_same_code() -> None:
    english, japanese = (
        [_code_only(b) for b in BLOCK_RE.findall((README_DIR / n).read_text(encoding="utf-8"))]
        for n in ("README.md", "README.ja.md")
    )
    assert english
    assert english == japanese
