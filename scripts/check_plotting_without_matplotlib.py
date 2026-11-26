"""Check flexpyirt without matplotlib (CI, isolated environment with no extras; DESIGN.md §20).

``import flexpyirt`` and ``import flexpyirt.plotting`` must work, the DataFrame functions behind
the plots must work, and every plot function must raise ``MissingDependencyError`` naming the
``flexpyirt[plot]`` extra. Exits 1 with a message otherwise. Run with
``python scripts/check_plotting_without_matplotlib.py`` in an environment without matplotlib.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import flexpyirt  # noqa: E402
import flexpyirt.plotting as plotting  # noqa: E402
from flexpyirt import Model, TwoPL, fit  # noqa: E402
from flexpyirt.errors import MissingDependencyError  # noqa: E402


def main() -> int:
    """Run the checks; return the exit status."""
    if importlib.util.find_spec("matplotlib") is not None:
        print("matplotlib is installed; this check needs an environment without it.")
        return 1
    if "matplotlib" in sys.modules:
        print("import flexpyirt imported matplotlib.")
        return 1
    rng = np.random.default_rng(0)
    data = pd.DataFrame((rng.random((200, 4)) < 0.6).astype(float), columns=list("abcd"))
    res = fit(Model.uniform(TwoPL(), items=list("abcd")), data, se=None)
    flexpyirt.empirical_item_curve(res, data, n_bins=4)
    flexpyirt.wright_map_table(res, data)
    calls: dict[str, Callable[[], Any]] = {
        "plot_item_curves": lambda: plotting.plot_item_curves(res),
        "plot_information": lambda: plotting.plot_information(res),
        "plot_expected_score": lambda: plotting.plot_expected_score(res),
        "plot_wright_map": lambda: plotting.plot_wright_map(res, data),
    }
    failed = []
    for name, call in calls.items():
        try:
            call()
        except MissingDependencyError as exc:
            if "flexpyirt[plot]" not in str(exc):
                failed.append(f"{name}: message does not name the extra: {exc}")
        else:
            failed.append(f"{name}: no MissingDependencyError")
    for line in failed:
        print(line)
    if not failed:
        print("ok: flexpyirt works without matplotlib and the plot functions raise clearly")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
