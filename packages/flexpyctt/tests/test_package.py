"""Package-level checks: public names, version, README and the no-JAX rule (DESIGN.md §16.18)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

import flexpyctt


def test_public_names_are_importable() -> None:
    for name in flexpyctt.__all__:
        assert hasattr(flexpyctt, name), name
    assert flexpyctt.__version__


def test_import_does_not_load_jax_or_flexpyirt() -> None:
    """flexpyctt must work where JAX is not installed, so it never imports it."""
    code = (
        "import sys, flexpyctt, flexpyctt.dimensionality, flexpyctt.correlation; "
        "bad = [m for m in ('jax', 'flexpyirt') if m in sys.modules]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


README_DIR = Path(__file__).resolve().parents[1]
BLOCK_RE = re.compile(r"^```python\n(.*?)^```", re.M | re.S)


def _code_only(block: str) -> list[str]:
    """Return the block's lines without comments and blank lines."""
    lines = [re.sub(r"\s+#.*$", "", line).rstrip() for line in block.splitlines()]
    return [line for line in lines if line.strip() and not line.strip().startswith("#")]


@pytest.mark.parametrize("name", ["README.md", "README.ja.md"])
def test_readme_code_runs(name: str, capsys: pytest.CaptureFixture[str]) -> None:
    blocks = BLOCK_RE.findall((README_DIR / name).read_text(encoding="utf-8"))
    assert blocks, f"{name} has no python blocks"
    namespace: dict[str, object] = {"__name__": f"readme:{name}"}
    for code in blocks:
        exec(compile(code, name, "exec"), namespace)
    capsys.readouterr()
    assert namespace["pa"].n_factors == 2  # type: ignore[attr-defined]


def test_readme_languages_have_the_same_code() -> None:
    english, japanese = (
        [_code_only(b) for b in BLOCK_RE.findall((README_DIR / n).read_text(encoding="utf-8"))]
        for n in ("README.md", "README.ja.md")
    )
    assert english == japanese
