"""Cold-start work (#154, DESIGN.md §28): packed pattern compression, the phased SQUAREM loop
and the opt-in persistent compilation cache."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from flexpyirt import BAEM, Model, TwoPL, enable_compilation_cache, fit
from flexpyirt._compile import _pack_rows, _unique_rows, compress_patterns
from flexpyirt.cache import default_cache_dir
from flexpyirt.errors import SpecificationError


def _reference_compress(
    y: np.ndarray, group: np.ndarray, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The pre-#154 implementation (``np.unique`` over whole rows)."""
    keyed = np.column_stack([group, y])
    uniq, inverse, n_rows = np.unique(keyed, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    counts = (
        n_rows.astype(float)
        if weights is None
        else np.bincount(inverse, weights=weights, minlength=uniq.shape[0]).astype(float)
    )
    return uniq[:, 1:], uniq[:, 0], counts, inverse


def _assert_same(a: tuple[np.ndarray, ...], b: tuple[np.ndarray, ...]) -> None:
    for x, z in zip(a, b, strict=True):
        assert x.dtype == z.dtype or (x.dtype.kind == z.dtype.kind == "i")
        np.testing.assert_array_equal(x, z)


@settings(max_examples=200, deadline=None)
@given(
    data=st.data(),
    n_rows=st.integers(1, 60),
    n_items=st.integers(1, 80),
    n_categories=st.integers(2, 40),
    n_groups=st.integers(1, 5),
    weighted=st.booleans(),
)
def test_compress_patterns_matches_np_unique(
    data: st.DataObject,
    n_rows: int,
    n_items: int,
    n_categories: int,
    n_groups: int,
    weighted: bool,
) -> None:
    """Same unique rows, order, groups, counts and inverse as ``np.unique(axis=0)``.

    Few rows and few categories make duplicates likely; many items or categories span several
    64-bit words (the ``lexsort`` path).
    """
    y = data.draw(
        hnp.arrays(np.int32, (n_rows, n_items), elements=st.integers(-1, n_categories - 1))
    )
    group = data.draw(hnp.arrays(np.int32, (n_rows,), elements=st.integers(0, n_groups - 1)))
    if data.draw(st.booleans()):
        # Repeat rows so the duplicate handling is exercised on wide rows too.
        y = np.concatenate([y, y[::-1]])
        group = np.concatenate([group, group[::-1]])
    weights = (
        data.draw(hnp.arrays(np.float64, (y.shape[0],), elements=st.floats(0.0, 10.0, width=64)))
        if weighted
        else None
    )
    _assert_same(compress_patterns(y, group, weights), _reference_compress(y, group, weights))


@pytest.mark.parametrize(
    ("n_rows", "n_items", "n_categories", "n_groups"),
    [(2000, 20, 2, 1), (3000, 40, 2, 3), (2000, 30, 5, 50), (500, 300, 3, 2), (4000, 10, 2, 1)],
)
def test_compress_patterns_matches_np_unique_larger(
    n_rows: int, n_items: int, n_categories: int, n_groups: int
) -> None:
    rng = np.random.default_rng(n_rows + n_items)
    y = rng.integers(-1, n_categories, size=(n_rows, n_items)).astype(np.int32)
    group = rng.integers(0, n_groups, size=n_rows).astype(np.int32)
    _assert_same(compress_patterns(y, group), _reference_compress(y, group))


def test_pack_rows_word_layout() -> None:
    """Binary items with missing need 2 bits each: 32 items per word, plus the group field."""
    rng = np.random.default_rng(0)
    y = rng.integers(-1, 2, size=(100, 64)).astype(np.int32)
    group = np.zeros(100, dtype=np.int32)
    packed = _pack_rows(np.column_stack([group, y]))
    # 1 bit (constant group) + 64 x 2 bits = 129 bits -> 3 words.
    assert packed.shape == (100, 3)
    assert packed.dtype == np.uint64


def test_unique_rows_extreme_values() -> None:
    """Values far apart (wide fields) still sort and compare exactly."""
    keyed = np.array(
        [[0, 2**31 - 1, -(2**31)], [0, -(2**31), 2**31 - 1], [0, 2**31 - 1, -(2**31)], [1, 0, 0]],
        dtype=np.int32,
    )
    uniq, inverse, n_rows = _unique_rows(keyed)
    ref_u, ref_i, ref_n = np.unique(keyed, axis=0, return_inverse=True, return_counts=True)
    np.testing.assert_array_equal(uniq, ref_u)
    np.testing.assert_array_equal(inverse, ref_i.reshape(-1))
    np.testing.assert_array_equal(n_rows, ref_n)


def test_default_cache_dir_is_under_user_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    if sys.platform not in ("win32", "darwin"):
        monkeypatch.setenv("XDG_CACHE_HOME", "/some/cache")
        assert default_cache_dir() == Path("/some/cache/flexpyirt/jax")
    assert default_cache_dir().parts[-2:] == ("flexpyirt", "jax")


def test_enable_compilation_cache_rejects_negative_time() -> None:
    with pytest.raises(SpecificationError, match="min_compile_time"):
        enable_compilation_cache("unused", min_compile_time=-1.0)


_CACHE_SCRIPT = textwrap.dedent(
    """
    import json, sys
    import jax
    jax.config.update("jax_enable_x64", True)
    import numpy as np
    import flexpyirt as fp
    before = jax.config.jax_compilation_cache_dir
    jax.jit(lambda x: x + 1.0)(1.0)  # JAX already compiled something before the call
    path = fp.enable_compilation_cache(sys.argv[1])
    rng = np.random.default_rng(1)
    theta = rng.normal(size=(300, 1))
    a = np.linspace(0.8, 1.6, 6)
    d = np.linspace(-1.0, 1.0, 6)
    y = (rng.random((300, 6)) < 1 / (1 + np.exp(-(theta * a + d)))).astype(float)
    items = [f"i{j}" for j in range(6)]
    res = fp.fit(
        fp.Model.uniform(fp.TwoPL(), items=items), y, item_names=items,
        method=fp.BAEM(squarem=True), se=None,
    )
    print(json.dumps({"before": before, "path": str(path),
                      "loglik": res.loglik, "eta": res._eta.tolist()}))
    """
)


def _run_cache_script(cache_dir: Path) -> dict[str, object]:
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    out = subprocess.run(
        [sys.executable, "-c", _CACHE_SCRIPT, str(cache_dir)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    result: dict[str, object] = json.loads(out.stdout.strip().splitlines()[-1])
    return result


@pytest.mark.slow
def test_enable_compilation_cache_round_trip(tmp_path: Path) -> None:
    """A second process reads the programs the first wrote and gets bit-identical results."""
    cache_dir = tmp_path / "cache"
    first = _run_cache_script(cache_dir)
    assert first["before"] is None  # importing flexpyirt leaves the JAX config alone
    assert first["path"] == str(cache_dir.resolve())
    written = sorted(p.name for p in cache_dir.iterdir())
    assert written, "no compiled program was written to the cache"
    second = _run_cache_script(cache_dir)
    assert sorted(p.name for p in cache_dir.iterdir()) == written  # all hits, nothing new
    assert second["loglik"] == first["loglik"]
    assert second["eta"] == first["eta"]


@pytest.mark.slow
def test_squarem_phased_loop_converges() -> None:
    """The phased SQUAREM loop needs fewer EM evaluations than plain EM and reaches the same
    optimum (DESIGN.md §4.6, §28.1)."""
    rng = np.random.default_rng(3)
    theta = rng.normal(size=(800, 1))
    a = np.linspace(0.7, 2.0, 8)
    d = np.linspace(-1.5, 1.5, 8)
    y = (rng.random((800, 8)) < 1 / (1 + np.exp(-(theta * a + d)))).astype(float)
    items = [f"i{j}" for j in range(8)]
    spec = Model.uniform(TwoPL(), items=items)
    plain = fit(spec, y, item_names=items, method=BAEM(tol=1e-7), se=None)
    sq = fit(spec, y, item_names=items, method=BAEM(tol=1e-7, squarem=True), se=None)
    assert sq.n_em_evaluations < plain.n_em_evaluations
    np.testing.assert_allclose(sq._eta, plain._eta, atol=1e-5)
    assert abs(sq.loglik - plain.loglik) < 1e-8
