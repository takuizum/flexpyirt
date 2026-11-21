"""Keep the documentation honest, in both languages.

- Every ```python block in README / docs/guide (English and Japanese) runs, top to bottom, in one
  namespace per file (later blocks may use names from earlier ones).
- English and Japanese versions have the same code (comments and docstrings aside).
- Every document has its counterpart (``X.md`` <-> ``X.ja.md``).
- Every public name in ``flexpyirt.__all__`` and ``flexpyirt.errors.__all__`` is documented in
  docs/reference.md and docs/reference.ja.md.
- Japanese documents link to Japanese documents (except the language switcher).
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
RUNNABLE = ["README.md", "README.ja.md", "docs/guide.md", "docs/guide.ja.md"]
PAIRED = [
    "README",
    "CONTRIBUTING",
    "benchmarks/README",
    *(f"docs/{p.stem}" for p in (ROOT / "docs").glob("*.md")),
]


def _blocks(path: Path) -> list[str]:
    return BLOCK_RE.findall(path.read_text())


def _code_only(block: str) -> list[str]:
    """Return the block's lines without comments, docstring lines and blank lines."""
    out = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith('"""'):
            continue
        out.append(re.sub(r"\s+#.*$", "", line).rstrip())
    return out


@pytest.mark.parametrize("doc", RUNNABLE)
def test_code_blocks_run(doc: str, capsys: pytest.CaptureFixture[str]) -> None:
    blocks = _blocks(ROOT / doc)
    assert blocks, f"{doc} has no python blocks"
    namespace: dict[str, object] = {"__name__": f"docs:{doc}"}
    with warnings.catch_warnings():
        warnings.simplefilter("error", errors.ConvergenceWarning)
        warnings.simplefilter("error", errors.HeywoodWarning)
        for i, code in enumerate(blocks):
            try:
                exec(compile(code, f"{doc}[block {i}]", "exec"), namespace)
            except Exception as e:  # pragma: no cover - message for the failing block
                raise AssertionError(f"{doc} block {i} failed:\n{code}") from e
    capsys.readouterr()


@pytest.mark.parametrize("stem", sorted({p.removesuffix(".ja") for p in PAIRED}))
def test_documents_come_in_pairs(stem: str) -> None:
    en, ja = ROOT / f"{stem}.md", ROOT / f"{stem}.ja.md"
    if not en.exists() and not ja.exists():
        pytest.skip(f"{stem} is not on this branch")
    assert en.exists(), f"{en.name} is missing (counterpart of {ja.name})"
    assert ja.exists(), f"{ja.name} is missing (counterpart of {en.name})"


@pytest.mark.parametrize("stem", sorted({p.removesuffix(".ja") for p in PAIRED}))
def test_translations_have_the_same_code(stem: str) -> None:
    en, ja = ROOT / f"{stem}.md", ROOT / f"{stem}.ja.md"
    if not (en.exists() and ja.exists()):
        pytest.skip(f"{stem} is not on this branch")
    en_blocks = [_code_only(b) for b in _blocks(en)]
    ja_blocks = [_code_only(b) for b in _blocks(ja)]
    assert len(en_blocks) == len(ja_blocks), f"{stem}: different number of python blocks"
    for i, (a, b) in enumerate(zip(en_blocks, ja_blocks, strict=True)):
        assert a == b, f"{stem} block {i} differs between English and Japanese"


@pytest.mark.parametrize("doc", ["docs/reference.md", "docs/reference.ja.md"])
def test_reference_covers_public_api(doc: str) -> None:
    text = (ROOT / doc).read_text()
    names = [n for n in flexpyirt.__all__ if not n.startswith("_") and n != "errors"]
    names += list(errors.__all__)
    missing = [n for n in names if not re.search(rf"^#+ .*`{re.escape(n)}\b", text, re.M)]
    assert not missing, f"Not documented in {doc}: {missing}"


@pytest.mark.parametrize("doc", sorted(str(p.relative_to(ROOT)) for p in ROOT.rglob("*.ja.md")))
def test_japanese_docs_link_to_japanese_docs(doc: str) -> None:
    doc_path = ROOT / doc
    lines = doc_path.read_text().splitlines()
    body = [line for line in lines if not line.startswith("[English](")]
    links = re.findall(r"\]\(([^)#\s]+\.md)(?:#[^)]*)?\)", "\n".join(body))
    results_dir = (ROOT / "benchmarks" / "results").resolve()
    english = []
    for link in links:
        if link.endswith(".ja.md"):
            continue
        # Generated benchmark result tables under benchmarks/results/ are language-neutral.
        target = (doc_path.parent / link).resolve()
        if results_dir in target.parents:
            continue
        english.append(link)
    assert not english, f"{doc} links to English documents: {english}"
