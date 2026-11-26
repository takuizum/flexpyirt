"""Regenerate the figures shown in the user guide (docs/images/plots/*.png; DESIGN.md §21).

Runs the ```python blocks of docs/guide.md top to bottom (as tests/test_docs.py does) until the
figures of the guide exist, then saves them, so the pictures are exactly what the guide's code
draws. Run from the repository root after changing the plotting code or those guide sections::

    uv run python scripts/guide_figures.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "guide.md"
OUT = ROOT / "docs" / "images" / "plots"
BLOCK_RE = re.compile(r"^```python\n(.*?)^```", re.M | re.S)
FIGURES = {
    "item_curves": "fig_curves",
    "information": "ax_information",
    "expected_score": "ax_expected",
    "wright_map": "fig_wright",
    "dif": "fig_dif",
}
"""File stem -> name of the Figure or Axes in the guide's namespace."""


def main() -> None:
    """Run the guide's code blocks and save its figures."""
    namespace: dict[str, Any] = {"__name__": "guide_figures"}
    for i, code in enumerate(BLOCK_RE.findall(GUIDE.read_text(encoding="utf-8"))):
        exec(compile(code, f"guide.md[block {i}]", "exec"), namespace)
        if all(name in namespace for name in FIGURES.values()):
            break
    else:
        missing = [name for name in FIGURES.values() if name not in namespace]
        raise SystemExit(f"docs/guide.md does not define {missing}")
    OUT.mkdir(parents=True, exist_ok=True)
    for stem, name in FIGURES.items():
        obj = namespace[name]
        figure = getattr(obj, "figure", obj)
        figure.tight_layout()
        path = OUT / f"{stem}.png"
        figure.savefig(path, dpi=80)
        print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
