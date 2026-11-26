"""Group-blocked E-step for many groups (DESIGN.md §26, #79).

The blocked path must reproduce the flat path (every pattern under every group, then its own
group selected) up to summation order: expected counts, log-likelihood, Oakes and Hessian
information and EAP at a fixed parameter vector, and fitted estimates and SEs.
"""

from __future__ import annotations

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import flexpyirt._compile as compile_mod
import flexpyirt._core as core
from flexpyirt import BAEM, GRM, Model, Normal, Quadrature, TwoPL, fit
from flexpyirt._compile import build_group_blocks, compile_problem, prepare
from flexpyirt.errors import SpecificationError

QUAD = Quadrature.equispaced(41, bounds=(-6, 6))
TOL_CORE = 1e-10


@cache
def _data(n_groups: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Unequal group sizes, a GRM item and 5% missing responses."""
    rng = np.random.default_rng(79)
    n_items = 8
    slopes = rng.uniform(0.8, 2.0, n_items)
    intercepts = rng.normal(0.0, 1.0, n_items)
    blocks, labels = [], []
    for g in range(n_groups):
        theta = rng.normal(0.3 * g / n_groups, 1.0 + 0.2 * g / n_groups, 60 + 25 * g)
        prob = 1.0 / (1.0 + np.exp(-(slopes * theta[:, None] + intercepts)))
        y = (rng.uniform(size=prob.shape) < prob).astype(float)
        y[:, -1] += (rng.uniform(size=theta.shape) < prob[:, -1]).astype(float)  # 3 categories
        y[rng.uniform(size=y.shape) < 0.05] = np.nan
        blocks.append(y)
        labels += [f"g{g}"] * theta.size
    df = pd.DataFrame(np.vstack(blocks), columns=[f"i{j}" for j in range(n_items)])
    return df, np.array(labels)


def _spec(n_groups: int) -> Model:
    items = [f"i{j}" for j in range(8)]
    latents = {"g0": Normal.standard()} | {f"g{g}": Normal.free() for g in range(1, n_groups)}
    return (
        Model.from_items({**{i: TwoPL() for i in items[:-1]}, items[-1]: GRM(n_categories=3)})
        .with_groups(latents)
        .equal_across_groups(items=items[:4])
    )


def _compiled(n_groups: int, *, blocked: bool) -> tuple[core.CompiledProblem, jax.Array]:
    df, labels = _data(n_groups)
    spec = _spec(n_groups)
    prep = prepare(spec, df, group=labels, item_names=None, table=None)
    old = compile_mod.GROUP_BLOCK_MIN_GROUPS
    compile_mod.GROUP_BLOCK_MIN_GROUPS = 2 if blocked else 10**9
    try:
        comp = compile_problem(spec, prep.y, prep.group, prep.table, QUAD)
    finally:
        compile_mod.GROUP_BLOCK_MIN_GROUPS = old
    assert (comp.problem.blocks is not None) == blocked
    # A non-trivial point: perturb every free parameter (variances stay positive).
    eta = jnp.asarray(comp.eta0) + 0.05 * jnp.sin(jnp.arange(comp.eta0.shape[0]) + 1.0)
    return comp.problem, eta


def _close(a: jax.Array, b: jax.Array, tol: float = TOL_CORE) -> None:
    a_np, b_np = np.asarray(a), np.asarray(b)
    scale = max(1.0, float(np.max(np.abs(a_np))))
    np.testing.assert_allclose(a_np, b_np, rtol=0.0, atol=tol * scale)


@pytest.mark.parametrize("n_groups", [2, 3])
def test_blocked_e_step_matches_flat(n_groups: int) -> None:
    flat, eta = _compiled(n_groups, blocked=False)
    blocked, _ = _compiled(n_groups, blocked=True)
    counts_f, ll_f = jax.jit(core.e_step)(flat, eta)
    counts_b, ll_b = jax.jit(core.e_step)(blocked, eta)
    _close(ll_f, ll_b)
    _close(counts_f.n, counts_b.n)
    for r_f, r_b in zip(counts_f.r, counts_b.r, strict=True):
        _close(r_f, r_b)
    _close(core.marginal_loglik(flat, eta), core.marginal_loglik(blocked, eta))
    # Real pattern rows only: flat padding rows are in no block (DESIGN.md §26).
    n_real = int(np.sum(np.asarray(flat.counts) > 0))
    for a, b in zip(core.eap(flat, eta), core.eap(blocked, eta), strict=True):
        _close(a[:n_real], b[:n_real])


@pytest.mark.slow
def test_blocked_information_matches_flat() -> None:
    flat, eta = _compiled(3, blocked=False)
    blocked, _ = _compiled(3, blocked=True)
    _close(core.information_oakes(flat, eta), core.information_oakes(blocked, eta), 1e-9)
    _close(core.information_hessian(flat, eta), core.information_hessian(blocked, eta), 1e-9)


def test_blocked_oakes_sub_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blocks larger than the Oakes chunk budget are split into sub-chunks."""
    blocked, eta = _compiled(3, blocked=True)
    full = core.information_oakes(blocked, eta)
    monkeypatch.setattr(core, "_OAKES_CHUNK_ELEMENTS", 1)
    _close(full, core.information_oakes(blocked, eta), 1e-9)


@pytest.mark.slow
def test_blocked_fit_matches_flat(monkeypatch: pytest.MonkeyPatch) -> None:
    df, labels = _data(3)
    spec = _spec(3)
    method = BAEM(tol=1e-9, max_iter=5000)
    monkeypatch.setattr(compile_mod, "GROUP_BLOCK_MIN_GROUPS", 10**9)
    res_f = fit(spec, df, group=labels, method=method, quadrature=QUAD)
    monkeypatch.setattr(compile_mod, "GROUP_BLOCK_MIN_GROUPS", 2)
    res_b = fit(spec, df, group=labels, method=method, quadrature=QUAD)
    assert res_f.converged and res_b.converged
    assert abs(res_f.loglik - res_b.loglik) < 1e-8
    p_f, p_b = res_f.parameters, res_b.parameters
    np.testing.assert_allclose(p_b["estimate"], p_f["estimate"], atol=1e-6)
    np.testing.assert_allclose(p_b["se"], p_f["se"], rtol=1e-5)
    s_f = res_f.score(df, group=labels)
    s_b = res_b.score(df, group=labels)
    np.testing.assert_allclose(s_b["theta"], s_f["theta"], atol=1e-6)


def test_build_group_blocks_layout() -> None:
    groups = np.repeat(np.arange(4), [3, 0, 40, 9])
    n_rows = 64
    blk = build_group_blocks(groups, 4, n_rows)
    rows = np.asarray(blk.rows)
    grp = np.asarray(blk.group)
    last = np.asarray(blk.last)
    real = rows[rows < n_rows]
    np.testing.assert_array_equal(np.sort(real), np.arange(groups.size))  # each row once
    for b in range(rows.shape[0]):
        in_block = rows[b][rows[b] < n_rows]
        assert np.all(groups[in_block] == grp[b])
    # Group-major order; one ``last`` block per non-empty group, the empty group has none.
    used = np.any(rows < n_rows, axis=1)
    assert np.all(np.diff(grp[used]) >= 0)
    assert sorted(grp[last].tolist()) == [0, 2, 3]
    assert not np.any(last[~used])
    for g in (0, 2, 3):
        assert np.flatnonzero(used & (grp == g))[-1] == np.flatnonzero(last & (grp == g))[0]


def test_with_groups_accepts_mapping() -> None:
    items = ["a", "b", "c"]
    groups = {"ref": Normal.standard()} | {f"s{k}": Normal.free() for k in range(3)}
    spec = Model.uniform(TwoPL(), items=items).with_groups(groups)
    assert [g for g, _ in spec.resolved_groups()] == ["ref", "s0", "s1", "s2"]
    assert spec.with_groups(groups) == spec.with_groups(**groups)
    with pytest.raises(SpecificationError, match="mapping or keywords"):
        Model.uniform(TwoPL(), items=items).with_groups(groups, extra=Normal.free())
