"""Test-session configuration.

The test suite (not the library) enables 64-bit JAX globally, as a user would.
"""

import gc
from collections.abc import Iterator

import jax
import pytest

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]


@pytest.fixture(autouse=True, scope="module")
def _release_jax_compilations() -> Iterator[None]:
    """Drop compiled JAX executables after each test module.

    Compiled programs otherwise accumulate for the whole session; with the full suite this grew
    past the memory of the GitHub-hosted runner (the Python 3.11 job was killed with a runner
    shutdown signal in ``tests/test_mirt_m4_scoring.py``). Later modules recompile what they use.
    """
    yield
    jax.clear_caches()  # type: ignore[no-untyped-call]
    gc.collect()
