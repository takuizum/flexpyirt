"""Keep the documentation honest.

- Every ```python block in README.md and docs/guide.md runs, top to bottom, in one namespace per
  file (later blocks may use names from earlier ones).
- Every public name in ``flexpyirt.__all__`` and ``flexpyirt.errors.__all__`` is documented in
  docs/reference.md.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest

import flexpyirt
from flexpyirt import errors

ROOT = Path(__file__).resolve().parents[1]
BLOCK_RE = re.compile(r"^```python\n(.*?)^```", re.M | re.S)


def _blocks(path: Path) -> list[str]:
    return BLOCK_RE.findall(path.read_text())


@pytest.mark.parametrize("doc", ["README.md", "docs/guide.md"])
def test_code_blocks_run(doc: str, capsys: pytest.CaptureFixture[str]) -> None:
    blocks = _blocks(ROOT / doc)
    assert blocks, f"{doc} has no python blocks"
    namespace: dict[str, object] = {"__name__": f"docs:{doc}"}
    with warnings.catch_warnings():
        warnings.simplefilter("error", errors.ConvergenceWarning)
        for i, code in enumerate(blocks):
            try:
                exec(compile(code, f"{doc}[block {i}]", "exec"), namespace)
            except Exception as e:  # pragma: no cover - message for the failing block
                raise AssertionError(f"{doc} block {i} failed:\n{code}") from e
    capsys.readouterr()


def test_reference_covers_public_api() -> None:
    text = (ROOT / "docs" / "reference.md").read_text()
    names = [n for n in flexpyirt.__all__ if not n.startswith("_") and n != "errors"]
    names += list(errors.__all__)
    missing = [n for n in names if not re.search(rf"^#+ .*`{re.escape(n)}\b", text, re.M)]
    assert not missing, f"Not documented in docs/reference.md: {missing}"
