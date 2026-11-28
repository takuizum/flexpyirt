"""Batched estimation of many models in one kernel (DESIGN.md §28)."""

from __future__ import annotations

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from flexpyirt import BAEM, Model, Normal, TwoPL, fit
from flexpyirt._batch import chunk_size, perturbed_starts, run_batch, stack_problems
from flexpyirt._compile import compile_problem, prepare
from flexpyirt._core import CompiledProblem, _eta_bounds
from flexpyirt._dif import TABLE_COLUMNS, set_shared
from flexpyirt.errors import SpecificationError
from flexpyirt.fit import _em, _fit_tables, _squarem
from flexpyirt.quadrature import Quadrature

QUAD = Quadrature.equispaced(41, bounds=(-6, 6))
METHOD = BAEM(tol=1e-7, max_iter=3000)
_ETA_ATOL = 1e-5
"""Agreement of converged estimates (EM with ``tol=1e-7`` stops about ``tol/(1-rate)`` away)."""


def _sim(n_per: int = 400, n_items: int = 5, seed: int = 3) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.8, 1.6, size=n_items)
    d = rng.normal(0.0, 0.8, size=n_items)
    out = []
    for shift, mu in ((0.0, 0.0), (0.8, 0.3)):
        dd = d.copy()
        dd[-1] += shift
        theta = rng.normal(mu, 1.0, size=(n_per, 1))
        out.append((rng.random((n_per, n_items)) < 1 / (1 + np.exp(-(theta * a + dd)))) * 1.0)
    df = pd.DataFrame(np.vstack(out), columns=[f"i{k}" for k in range(n_items)])
    return df, np.repeat(["r", "f"], n_per)


@cache
def _design() -> tuple[Model, pd.DataFrame, np.ndarray, list[pd.DataFrame]]:
    """Invariant 2PL with two groups and one table per item freeing that item (DIF refits)."""
    df, grp = _sim()
    spec = (
        Model.uniform(TwoPL(), items=df.columns)
        .with_groups(r=Normal.standard(), f=Normal.free())
        .equal_across_groups(items=list(df.columns))
    )
    base = spec.parameter_table(df, group=grp)[list(TABLE_COLUMNS)].reset_index(drop=True)
    tables = []
    for item in df.columns:
        rows = np.flatnonzero(((base["item"] == item) & base["free"]).to_numpy())
        tables.append(set_shared(base, rows, shared=False))
    return spec, df, grp, [base, *tables]


def _compiled(
    spec: Model, df: pd.DataFrame, grp: np.ndarray, table: pd.DataFrame
) -> tuple[CompiledProblem, np.ndarray]:
    prep = prepare(spec, df, group=grp, item_names=None, table=table)
    comp = compile_problem(spec, prep.y, prep.group, prep.table, QUAD)
    return comp.problem, comp.eta0


def _single(prob: CompiledProblem, eta0: np.ndarray, method: BAEM = METHOD) -> object:
    runner = _squarem if method.squarem else _em
    return runner(
        prob, jnp.asarray(eta0), tol=method.tol, max_iter=method.max_iter, n_newton=3
    )


# --------------------------------------------------------------------------------------------
# Stacking
# --------------------------------------------------------------------------------------------


def test_stack_pads_eta_and_slots_and_shares_identical_leaves() -> None:
    spec, df, grp, tables = _design()
    comps = [_compiled(spec, df, grp, t) for t in tables[:3]]
    stack = stack_problems([p for p, _ in comps], [e for _, e in comps])
    assert stack is not None
    n_free = [e.shape[0] for _, e in comps]
    assert n_free[0] < n_free[1] == n_free[2]
    np.testing.assert_array_equal(stack.n_free, n_free)
    assert stack.eta0.shape == (3, max(n_free))
    np.testing.assert_array_equal(stack.eta0[0, n_free[0] :], 0.0)
    # Responses, counts, nodes and groups are shared, not copied per model.
    n_shared = sum(not b for b in stack.batched)
    assert n_shared >= 4
    assert len(stack.stacked) == sum(stack.batched)
    # The items freed in some model are not pooled over groups in the batch.
    prob = jax.tree_util.tree_unflatten(stack.treedef, _leaves(stack, 0))
    assert prob.families[0].invariant == (False, False, True, True, True)


def _leaves(stack: object, m: int) -> list[object]:
    it_shared = iter(stack.shared)  # type: ignore[attr-defined]
    it_stacked = iter(stack.stacked)  # type: ignore[attr-defined]
    return [
        next(it_stacked)[m] if b else next(it_shared)
        for b in stack.batched  # type: ignore[attr-defined]
    ]


def test_unstackable_structures_return_none() -> None:
    spec, df, grp, tables = _design()
    prob, eta0 = _compiled(spec, df, grp, tables[0])
    one = Model.uniform(TwoPL(), items=df.columns)
    prob1, eta1 = _compiled(one, df, None, one.parameter_table(df))  # type: ignore[arg-type]
    assert stack_problems([prob, prob1], [eta0, eta1]) is None


def test_chunk_size_is_balanced_and_bounded() -> None:
    spec, df, grp, tables = _design()
    prob, _ = _compiled(spec, df, grp, tables[0])
    assert chunk_size(prob, 7) == 7
    assert chunk_size(prob, 7, budget=1) == 1
    assert chunk_size(prob, 1, budget=10**12) == 1
    # Smallest budget that fits two models = twice the per-model working set.
    lo, hi = 1, 10**12
    while lo < hi:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if chunk_size(prob, 2, budget=mid) == 2 else (mid + 1, hi)
    per_model = lo // 2
    # Room for 3 of 7 models: chunks of 3, 3, 1 are balanced to 3, 2, 2 (padded to size 3).
    assert chunk_size(prob, 7, budget=3 * per_model) == 3
    assert chunk_size(prob, 8, budget=7 * per_model) == 4


# --------------------------------------------------------------------------------------------
# Batched runs equal single runs
# --------------------------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(("squarem", "parallel"), [(False, True), (True, True), (False, False)])
def test_batched_runs_equal_single_runs(squarem: bool, parallel: bool) -> None:
    spec, df, grp, tables = _design()
    method = BAEM(tol=1e-7, max_iter=3000, squarem=squarem)
    comps = [_compiled(spec, df, grp, t) for t in tables]
    stack = stack_problems([p for p, _ in comps], [e for _, e in comps])
    assert stack is not None
    runs = run_batch(
        stack,
        tol=method.tol,
        max_iter=method.max_iter,
        n_newton=3,
        accelerate=squarem,
        parallel=parallel,
    )
    for (prob, eta0), got in zip(comps, runs, strict=True):
        ref = _single(prob, eta0, method)
        assert got.eta.shape == eta0.shape
        assert bool(got.converged) and bool(ref.converged)  # type: ignore[attr-defined]
        # The same fixed point within the convergence tolerance. Batched kernels round
        # differently, so the last change can cross ``tol`` one iteration earlier or later.
        # SQUAREM's safeguard compares log posteriors that differ by ~1e-12 near convergence,
        # so its decisions (and the iteration count) can flip with rounding.
        if not squarem:
            assert abs(int(got.n_iter) - int(ref.n_iter)) <= 1  # type: ignore[attr-defined]
        np.testing.assert_allclose(got.eta, ref.eta, atol=_ETA_ATOL)  # type: ignore[attr-defined]
        assert float(got.loglik) == pytest.approx(float(ref.loglik), abs=1e-7)  # type: ignore[attr-defined]
        n = min(int(got.n_iter), int(ref.n_iter), 5 if squarem else 10**9)  # type: ignore[attr-defined]
        assert np.isnan(got.loglik_history[int(got.n_iter) :]).all()
        np.testing.assert_allclose(
            got.loglik_history[:n],
            np.asarray(ref.loglik_history)[:n],  # type: ignore[attr-defined]
            atol=1e-6,
        )


@pytest.mark.slow
def test_converged_slice_is_frozen() -> None:
    """A model started at its fixed point stops after one iteration while the others go on."""
    spec, df, grp, tables = _design()
    prob, eta0 = _compiled(spec, df, grp, tables[1])
    ref = _single(prob, eta0)
    at_solution = np.asarray(ref.eta)  # type: ignore[attr-defined]
    stack = stack_problems([prob, prob], [at_solution, eta0])
    assert stack is not None
    runs = run_batch(
        stack,
        tol=METHOD.tol,
        max_iter=METHOD.max_iter,
        n_newton=3,
        accelerate=False,
        parallel=True,
    )
    frozen = _single(prob, at_solution)
    assert int(runs[0].n_iter) == int(frozen.n_iter) <= 2  # type: ignore[attr-defined]
    assert int(runs[1].n_iter) == int(ref.n_iter) > 10  # type: ignore[attr-defined]
    np.testing.assert_allclose(runs[0].eta, np.asarray(frozen.eta), atol=1e-12)  # type: ignore[attr-defined]


@pytest.mark.slow
def test_uneven_chunks_reuse_one_kernel() -> None:
    spec, df, grp, tables = _design()
    comps = [_compiled(spec, df, grp, t) for t in tables]
    stack = stack_problems([p for p, _ in comps], [e for _, e in comps])
    assert stack is not None
    opts = {
        "tol": METHOD.tol,
        "max_iter": METHOD.max_iter,
        "n_newton": 3,
        "accelerate": False,
        "parallel": True,
    }
    whole = run_batch(stack, **opts)  # type: ignore[arg-type]
    chunked = run_batch(stack, batch_size=4, **opts)  # type: ignore[arg-type]
    for a, b in zip(whole, chunked, strict=True):
        assert abs(int(a.n_iter) - int(b.n_iter)) <= 1
        np.testing.assert_allclose(a.eta, b.eta, atol=_ETA_ATOL)


@pytest.mark.slow
def test_resampled_counts_batch() -> None:
    """Bootstrap-style: the same patterns with resampled counts, stacked on ``counts`` only."""
    spec, df, grp, tables = _design()
    prob, eta0 = _compiled(spec, df, grp, tables[0])
    rng = np.random.default_rng(0)
    counts = np.asarray(prob.counts)
    probs = []
    for _ in range(3):
        new = rng.multinomial(int(counts.sum()), counts / counts.sum()).astype(float)
        probs.append(type(prob)(**{**prob.__dict__, "counts": jnp.asarray(new)}))
    stack = stack_problems(probs, [eta0] * 3)
    assert stack is not None
    assert sum(stack.batched) == 1  # only the counts differ
    runs = run_batch(stack, tol=METHOD.tol, max_iter=METHOD.max_iter, n_newton=3, accelerate=False)
    for p, got in zip(probs, runs, strict=True):
        ref = _single(p, eta0)
        assert abs(int(got.n_iter) - int(ref.n_iter)) <= 1  # type: ignore[attr-defined]
        np.testing.assert_allclose(got.eta, np.asarray(ref.eta), atol=_ETA_ATOL)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------------------------
# Shell: _fit_tables and FitResult.dif()
# --------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_fit_tables_equals_fit() -> None:
    spec, df, grp, tables = _design()
    got = _fit_tables(
        spec,
        df,
        tables=tables[:3],
        group=grp,
        item_names=None,
        method=METHOD,
        quadrature=QUAD,
        se="oakes",
    )
    for table, res in zip(tables[:3], got, strict=True):
        ref = fit(spec, df, group=grp, table=table, method=METHOD, quadrature=QUAD, se="oakes")
        assert abs(res.n_iter - ref.n_iter) <= 1
        assert res.converged and ref.converged
        assert res.loglik == pytest.approx(ref.loglik, abs=1e-7)
        np.testing.assert_allclose(res._eta, ref._eta, atol=_ETA_ATOL)
        np.testing.assert_allclose(res.parameters["se"], ref.parameters["se"], atol=1e-5)
        assert res.n_params == ref.n_params
        assert res.starts is None


def test_fit_tables_rejects_robust_se() -> None:
    spec, df, grp, tables = _design()
    with pytest.raises(SpecificationError, match="batched fits support"):
        _fit_tables(
            spec,
            df,
            tables=tables[:2],
            group=grp,
            item_names=None,
            method=METHOD,
            quadrature=QUAD,
            se="sandwich",
        )


@pytest.mark.slow
def test_dif_lrt_equals_one_by_one_refits() -> None:
    spec, df, grp, tables = _design()
    res = fit(spec, df, group=grp, method=METHOD, quadrature=QUAD, se=None)
    model = (
        Model.uniform(TwoPL(), items=df.columns)
        .with_groups(r=Normal.standard(), f=Normal.free())
        .equal_across_groups(items=["i0", "i1"])
    )
    free = fit(model, df, group=grp, method=METHOD, quadrature=QUAD, se=None)
    out = free.dif(df, group=grp, items=["i2", "i3", "i4"])
    base = free.parameters[list(TABLE_COLUMNS)].copy()
    base["value"] = free.parameters["estimate"].to_numpy(dtype=float)
    for item, x2 in zip(out["item"], out["x2"], strict=True):
        rows = np.flatnonzero(((base["item"] == item) & base["free"]).to_numpy())
        table = set_shared(base, rows, shared=True)
        con = fit(model, df, group=grp, table=table, method=METHOD, quadrature=QUAD, se=None)
        assert x2 == pytest.approx(max(0.0, 2 * (free.loglik - con.loglik)), abs=1e-6)
    assert res.converged


# --------------------------------------------------------------------------------------------
# Multi-start
# --------------------------------------------------------------------------------------------


def test_perturbed_starts_stay_inside_bounds() -> None:
    spec, df, grp, tables = _design()
    prob, eta0 = _compiled(spec, df, grp, tables[0])
    starts = perturbed_starts(prob, eta0, n_starts=50, scale=2.0, key=jax.random.key(3))
    assert starts.shape == (50, eta0.shape[0])
    np.testing.assert_array_equal(starts[0], eta0)
    lo, hi = (np.asarray(b) for b in _eta_bounds(prob, jnp.asarray(eta0)))
    assert (starts > lo).all() and (starts < hi).all()
    assert np.std(starts[1:], axis=0).min() > 0.1
    again = perturbed_starts(prob, eta0, n_starts=50, scale=2.0, key=jax.random.key(3))
    np.testing.assert_array_equal(starts, again)
    one = perturbed_starts(prob, eta0, n_starts=1, scale=2.0, key=jax.random.key(3))
    np.testing.assert_array_equal(one, eta0[None])


@pytest.mark.slow
def test_multi_start_picks_the_best_start() -> None:
    spec, df, grp, tables = _design()
    key = jax.random.key(7)
    res = fit(
        spec, df, group=grp, method=METHOD, quadrature=QUAD, se="oakes", n_starts=4, key=key
    )
    starts = res.starts
    assert starts is not None
    assert list(starts.columns) == ["start", "loglik", "log_posterior", "converged", "n_iter", "best"]
    assert starts["best"].sum() == 1
    best = starts[starts["best"]].iloc[0]
    assert res.loglik == pytest.approx(best["loglik"], abs=1e-12)
    assert res.n_iter == best["n_iter"]
    top = starts.loc[starts["converged"], "log_posterior"].max()
    assert best["log_posterior"] >= top - 1e-6
    # Ties within rounding go to the earliest start (here start 1).
    assert int(best["start"]) == 1
    # Each start equals a single run from the same starting values.
    prob, eta0 = _compiled(spec, df, grp, tables[0])
    vals = perturbed_starts(prob, eta0, n_starts=4, scale=0.5, key=key)
    for k in range(4):
        ref = _single(prob, vals[k])
        assert abs(int(starts["n_iter"].iloc[k]) - int(ref.n_iter)) <= 1  # type: ignore[attr-defined]
        assert starts["loglik"].iloc[k] == pytest.approx(float(ref.loglik), abs=1e-7)  # type: ignore[attr-defined]
    # This likelihood is unimodal: every start reaches the single-start solution.
    single = fit(spec, df, group=grp, method=METHOD, quadrature=QUAD, se="oakes")
    assert single.starts is None
    assert res.loglik == pytest.approx(single.loglik, abs=1e-7)
    np.testing.assert_allclose(res.parameters["estimate"], single.parameters["estimate"], atol=1e-4)
    np.testing.assert_allclose(res.parameters["se"], single.parameters["se"], atol=1e-4)


def test_multi_start_argument_errors() -> None:
    spec, df, grp, _ = _design()
    with pytest.raises(SpecificationError, match="PRNG key"):
        fit(spec, df, group=grp, n_starts=2)
    with pytest.raises(SpecificationError, match="n_starts"):
        fit(spec, df, group=grp, n_starts=0)
    with pytest.raises(SpecificationError, match="n_starts"):
        fit(spec, df, group=grp, n_starts=True)  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match="start_scale"):
        fit(spec, df, group=grp, n_starts=2, start_scale=0.0, key=jax.random.key(0))


@pytest.mark.slow
def test_small_batch_matches_single_runs() -> None:
    """Fast check: two starts of a one-group 2PL, batched vs. one by one."""
    df, _ = _sim(n_per=150, n_items=4)
    spec = Model.uniform(TwoPL(), items=df.columns)
    prob, eta0 = _compiled(spec, df, None, spec.parameter_table(df))  # type: ignore[arg-type]
    method = BAEM(tol=1e-5, max_iter=500)
    starts = perturbed_starts(prob, eta0, n_starts=2, scale=0.5, key=jax.random.key(1))
    stack = stack_problems([prob, prob], list(starts))
    assert stack is not None and not any(stack.batched)
    runs = run_batch(
        stack, tol=method.tol, max_iter=method.max_iter, n_newton=3, accelerate=False, parallel=True
    )
    for start, got in zip(starts, runs, strict=True):
        ref = _single(prob, start, method)
        assert abs(int(got.n_iter) - int(ref.n_iter)) <= 1  # type: ignore[attr-defined]
        np.testing.assert_allclose(got.eta, np.asarray(ref.eta), atol=1e-3)  # type: ignore[attr-defined]
        assert float(got.loglik) == pytest.approx(float(ref.loglik), abs=1e-6)  # type: ignore[attr-defined]
