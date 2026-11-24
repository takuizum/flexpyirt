"""Test-session configuration.

The test suite (not the library) enables 64-bit JAX globally, as a user would.

Scheduling (CONTRIBUTING.md, "Tests"):

- pytest-xdist (``--dist loadfile``) runs a module on one worker, so the JAX compilations and the
  ``functools.cache``-d fits of a module are reused. A heavy test that needs its own fit belongs in
  a module of its own, so it runs on another worker.
- ``--shard K/N`` runs only the modules of shard K (1-based) of N. Modules are packed into shards
  by the seconds in ``tests/durations.json`` (longest first, each into the least loaded shard).
- ``--write-durations PATH`` writes the measured seconds per module, the input for
  ``tests/durations.json``.
- The terminal summary lists tests that are not marked ``slow`` but took longer than
  ``SLOW_SECONDS``.
"""

from __future__ import annotations

import gc
import json
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import jax
import pytest

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

DURATIONS = Path(__file__).parent / "durations.json"
SLOW_SECONDS = 5.0
"""A test (with its share of a cached fit) slower than this must be marked ``slow``."""

_module_seconds: defaultdict[str, float] = defaultdict(float)
_unmarked_slow: dict[str, float] = {}


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("flexpyirt")
    group.addoption(
        "--shard",
        default=None,
        metavar="K/N",
        help="Run only shard K (1-based) of N; modules are balanced by tests/durations.json.",
    )
    group.addoption(
        "--write-durations",
        default=None,
        metavar="PATH",
        help="Write the measured seconds per test module to PATH (JSON).",
    )


def _module(item: pytest.Item) -> str:
    return item.path.stem


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Record each test's module and ``slow`` mark in its reports (read in the main process)."""
    for item in items:
        item.user_properties.append(("test_module", _module(item)))
        item.user_properties.append(("slow", item.get_closest_marker("slow") is not None))


def _parse_shard(value: str) -> tuple[int, int]:
    try:
        k, n = (int(part) for part in value.split("/"))
    except ValueError:
        raise pytest.UsageError(f"--shard must look like K/N, got {value!r}") from None
    if not 1 <= k <= n:
        raise pytest.UsageError(f"--shard needs 1 <= K <= N, got {value!r}")
    return k, n


def assign_shards(seconds: dict[str, float], n_shards: int) -> dict[str, int]:
    """Pack modules into ``n_shards`` shards: longest first, each into the least loaded shard.

    Ties are broken by name so every xdist worker gets the same assignment.
    """
    load = [0.0] * n_shards
    shard_of = {}
    for name in sorted(seconds, key=lambda g: (-seconds[g], g)):
        target = min(range(n_shards), key=lambda s: (load[s], s))
        shard_of[name] = target
        load[target] += seconds[name]
    return shard_of


class _ShardPlugin:
    """Deselects the modules of other shards (after ``-m``/``-k`` selection)."""

    def __init__(self, k: int, n: int) -> None:
        self.k, self.n = k, n

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(
        self, config: pytest.Config, items: list[pytest.Item]
    ) -> None:
        known = json.loads(DURATIONS.read_text(encoding="utf-8")) if DURATIONS.exists() else {}
        counts: defaultdict[str, int] = defaultdict(int)
        for item in items:
            counts[_module(item)] += 1
        # A module missing from durations.json is weighted as 1 s per test.
        seconds = {g: float(known.get(g, c)) for g, c in counts.items()}
        shard_of = assign_shards(seconds, self.n)
        keep = [item for item in items if shard_of[_module(item)] == self.k - 1]
        drop = [item for item in items if shard_of[_module(item)] != self.k - 1]
        if drop:
            config.hook.pytest_deselected(items=drop)
            items[:] = keep


def pytest_configure(config: pytest.Config) -> None:
    shard = config.getoption("--shard")
    if shard:
        config.pluginmanager.register(_ShardPlugin(*_parse_shard(shard)), "flexpyirt-shard")


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    props = dict(report.user_properties)
    if "test_module" not in props:
        return
    _module_seconds[str(props["test_module"])] += report.duration
    if report.when == "call" and not props["slow"] and report.duration > SLOW_SECONDS:
        _unmarked_slow[report.nodeid] = report.duration


def pytest_sessionfinish(session: pytest.Session) -> None:
    path = session.config.getoption("--write-durations")
    if path and not hasattr(session.config, "workerinput"):
        rounded = {g: round(s, 1) for g, s in sorted(_module_seconds.items())}
        Path(path).write_text(json.dumps(rounded, indent=2) + "\n", encoding="utf-8")


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    if not _unmarked_slow:
        return
    terminalreporter.section(f"tests over {SLOW_SECONDS:g} s not marked slow")
    for nodeid, seconds in sorted(_unmarked_slow.items(), key=lambda kv: -kv[1]):
        terminalreporter.write_line(f"{seconds:8.1f}s  {nodeid}")


@pytest.fixture(autouse=True, scope="module")
def _release_jax_compilations() -> Iterator[None]:
    """Drop compiled JAX executables after each test module.

    Compiled programs otherwise accumulate for the whole session; with the full suite this grew
    past the memory of the GitHub-hosted runner (the Python 3.11 job was killed with a runner
    shutdown signal in ``tests/test_mirt_m4_scoring.py``). Later modules recompile what they use
    (from the persistent compilation cache when ``JAX_COMPILATION_CACHE_DIR`` is set, as in CI).
    """
    yield
    jax.clear_caches()  # type: ignore[no-untyped-call]
    gc.collect()
