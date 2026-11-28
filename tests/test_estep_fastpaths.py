"""E-step fast paths (DESIGN.md §4.1, §4.2, #152).

A complete family (no missing response in a real pattern row) drops category 0 from the
pattern contraction; one group skips the group one-hot; patterns are padded to
``{1, 1.5} * 2**k`` buckets. Each must reproduce the generic path (every family not complete)
up to summation order: expected counts, log-likelihood and posterior at a fixed parameter
vector, and fitted estimates and SEs.
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import flexpyirt._compile as compile_mod
from flexpyirt import (
    BAEM,
    GPCM,
    GRM,
    Model,
    Normal,
    Quadrature,
    ThreePL,
    TwoPL,
    _core,
    fit,
)
from flexpyirt._compile import compile_problem, pad_patterns, pattern_bucket, prepare
from flexpyirt._jsi import without_item

QUAD = Quadrature.equispaced(41, bounds=(-6, 6))
TOL = 1e-12


@cache
def _data(n_groups: int, missing: bool, seed: int = 152) -> tuple[pd.DataFrame, np.ndarray]:
    """Items i0-i5 dichotomous, i6-i7 with 3 categories; missing responses only in i6-i7."""
    rng = np.random.default_rng(seed)
    slopes = rng.uniform(0.8, 2.0, 8)
    intercepts = rng.normal(0.0, 1.0, 8)
    blocks, labels = [], []
    for g in range(n_groups):
        theta = rng.normal(0.3 * g / n_groups, 1.0, 150 + 40 * g)
        prob = 1.0 / (1.0 + np.exp(-(slopes * theta[:, None] + intercepts)))
        y = (rng.uniform(size=prob.shape) < prob).astype(float)
        y[:, 6:] += (rng.uniform(size=(theta.size, 2)) < prob[:, 6:]).astype(float)
        if missing:
            y[:, 6:][rng.uniform(size=(theta.size, 2)) < 0.1] = np.nan
        blocks.append(y)
        labels += [f"g{g}"] * theta.size
    df = pd.DataFrame(np.vstack(blocks), columns=[f"i{j}" for j in range(8)])
    return df, np.array(labels)


def _spec(n_groups: int, polytomous: str = "grm") -> Model:
    items = [f"i{j}" for j in range(8)]
    poly = GRM(n_categories=3) if polytomous == "grm" else GPCM(n_categories=3)
    spec = Model.from_items({i: TwoPL() if j < 6 else poly for j, i in enumerate(items)})
    if n_groups == 1:
        return spec
    latents = {"g0": Normal.standard()} | {f"g{g}": Normal.free() for g in range(1, n_groups)}
    return spec.with_groups(latents).equal_across_groups(items=items[:3])


def _problem(
    spec: Model, df: pd.DataFrame, group: np.ndarray | None
) -> tuple[_core.CompiledProblem, jax.Array]:
    prep = prepare(spec, df, group=group, item_names=None, table=None)
    comp = compile_problem(spec, prep.y, prep.group, prep.table, QUAD)
    return comp.problem, jnp.asarray(comp.eta0)


def _generic(prob: _core.CompiledProblem) -> _core.CompiledProblem:
    """``prob`` with every family on the generic one-hot path."""
    return replace(prob, families=tuple(replace(f, complete=False) for f in prob.families))


def _perturbed(eta: jax.Array) -> jax.Array:
    """Move away from the starting values so every table entry matters."""
    rng = np.random.default_rng(1)
    return eta + 0.1 * jnp.asarray(rng.normal(size=eta.shape))


def _assert_close(fast: object, generic: object, tol: float = TOL) -> None:
    for a, b in zip(
        jax.tree_util.tree_leaves(fast), jax.tree_util.tree_leaves(generic), strict=True
    ):
        a, b = np.asarray(a), np.asarray(b)
        scale = max(1.0, float(np.max(np.abs(b)))) if b.size else 1.0
        np.testing.assert_allclose(a, b, rtol=0, atol=tol * scale)


# ---- buckets ------------------------------------------------------------------------------


def test_pattern_buckets() -> None:
    expected = {1: 8, 8: 8, 9: 12, 12: 12, 13: 16, 17: 24, 25: 32, 19_859: 24_576, 24_577: 32_768}
    for n, bucket in expected.items():
        assert pattern_bucket(n) == bucket
    for n in range(1, 5000):
        b = pattern_bucket(n)
        assert n <= b <= max(8, 1.5 * n)
    y, g, c = pad_patterns(np.ones((13, 2), dtype=int), np.zeros(13, dtype=int), np.ones(13))
    assert y.shape == (16, 2) and c[13:].sum() == 0.0 and g.shape == (16,)


# ---- flags --------------------------------------------------------------------------------


def test_complete_flags() -> None:
    df, _ = _data(1, missing=True)
    prob, _ = _problem(_spec(1), df, None)
    assert [f.complete for f in prob.families] == [True, False]
    df, _ = _data(1, missing=False)
    prob, _ = _problem(_spec(1), df, None)
    assert [f.complete for f in prob.families] == [True, True]
    # Replacing the responses resets every flag.
    reset = _core.with_responses(prob)
    assert not any(f.complete for f in reset.families)
    assert not any(f.complete for f in without_item(prob, jnp.asarray(0)).families)


# ---- fixed parameters ---------------------------------------------------------------------


@pytest.mark.parametrize("n_groups", [1, 2, 5])
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("polytomous", ["grm", "gpcm"])
def test_e_step_matches_generic(n_groups: int, missing: bool, polytomous: str) -> None:
    df, group = _data(n_groups, missing)
    prob, eta = _problem(_spec(n_groups, polytomous), df, group if n_groups > 1 else None)
    assert (prob.blocks is not None) == (n_groups > 1)
    assert prob.families[0].complete
    eta = _perturbed(eta)
    fast = jax.jit(_core._e_step_details)(prob, eta)
    generic = jax.jit(_core._e_step_details)(_generic(prob), eta)
    _assert_close(fast, generic)
    _assert_close(
        jax.jit(_core.marginal_loglik)(prob, eta),
        jax.jit(_core.marginal_loglik)(_generic(prob), eta),
    )


def test_e_step_matches_generic_flat_many_groups() -> None:
    """The flat path with several groups (blocks off) keeps the group one-hot."""
    df, group = _data(3, missing=False)
    prob, eta = _problem(_spec(3), df, group)
    prob = replace(prob, blocks=None)
    eta = _perturbed(eta)
    _assert_close(
        jax.jit(_core._e_step_details)(prob, eta),
        jax.jit(_core._e_step_details)(_generic(prob), eta),
    )


def test_e_step_matches_generic_3pl_multidimensional() -> None:
    rng = np.random.default_rng(3)
    theta = rng.normal(size=(300, 2))
    a = rng.uniform(0.5, 1.5, size=(8, 2))
    p = 0.15 + 0.85 / (1 + np.exp(-(theta @ a.T + rng.normal(size=8))))
    df = pd.DataFrame(
        (rng.uniform(size=p.shape) < p).astype(int), columns=[f"i{j}" for j in range(8)]
    )
    items = list(df.columns)
    spec = Model.uniform(ThreePL(), items=items).with_factors({"f1": items[:4], "f2": items[4:]})
    prep = prepare(spec, df, group=None, item_names=None, table=None)
    comp = compile_problem(spec, prep.y, prep.group, prep.table, Quadrature.equispaced(15))
    prob, eta = comp.problem, _perturbed(jnp.asarray(comp.eta0))
    assert prob.families[0].complete
    _assert_close(
        jax.jit(_core._e_step_details)(prob, eta),
        jax.jit(_core._e_step_details)(_generic(prob), eta),
    )


def test_rank_d_dispatch() -> None:
    """Only complete families of the exact built-in 2PL / GPCM kernels take the rank-D split."""
    from flexpyirt.items import DichotomousKernel

    class Custom(DichotomousKernel):
        pass

    df, _ = _data(1, missing=True)
    prob, _ = _problem(_spec(1, "gpcm"), df, None)
    assert [_core._is_rank_d(prob, f) for f in prob.families] == [True, False]
    custom = replace(prob.families[0], kernel=Custom())
    assert not _core._is_rank_d(prob, custom)


@pytest.mark.parametrize("polytomous", ["grm", "gpcm"])
def test_e_step_matches_generic_two_factors(polytomous: str) -> None:
    df, _ = _data(1, missing=False)
    items = list(df.columns)
    spec = _spec(1, polytomous).with_factors({"f1": items[:5], "f2": items[3:]})
    prep = prepare(spec, df, group=None, item_names=None, table=None)
    comp = compile_problem(spec, prep.y, prep.group, prep.table, Quadrature.equispaced(15))
    prob, eta = comp.problem, _perturbed(jnp.asarray(comp.eta0))
    assert _core._is_rank_d(prob, prob.families[0])
    _assert_close(
        jax.jit(_core._e_step_details)(prob, eta),
        jax.jit(_core._e_step_details)(_generic(prob), eta),
    )


# ---- fits ---------------------------------------------------------------------------------

_em = jax.jit(_core.em, static_argnames=("max_iter", "n_newton", "unconstrained"))


@pytest.mark.parametrize("n_groups", [1, 3])
def test_em_step_and_information_match_generic(n_groups: int) -> None:
    """One EM iteration, and Oakes and Hessian information at a fixed point.

    Longer trajectories are not compared: the generic path itself moves by up to 1e-3 after 25
    SQUAREM iterations when the start is perturbed by 1e-14 (relative), so they test the EM
    map's sensitivity, not the E-step.
    """
    df, group = _data(n_groups, missing=n_groups == 3)
    prob, eta0 = _problem(_spec(n_groups), df, group if n_groups > 1 else None)
    fast = _em(prob, eta0, tol=0.0, max_iter=1, n_newton=3)
    generic = _em(_generic(prob), eta0, tol=0.0, max_iter=1, n_newton=3)
    _assert_close(fast.eta, generic.eta, tol=1e-10)
    _assert_close(fast.loglik, generic.loglik)
    eta = _perturbed(eta0)
    for info in (_core.information_oakes, _core.information_hessian):
        _assert_close(jax.jit(info)(prob, eta), jax.jit(info)(_generic(prob), eta), tol=1e-10)


def test_fit_matches_generic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Converged fits agree to the convergence tolerance (the stopping iteration may differ)."""
    df, _ = _data(1, missing=False)
    method = BAEM(tol=1e-8, max_iter=3000)
    fast = fit(_spec(1), df, method=method, quadrature=QUAD, se="oakes")
    with monkeypatch.context() as m:
        m.setattr(compile_mod, "_family_complete", lambda *_: False)
        generic = fit(_spec(1), df, method=method, quadrature=QUAD, se="oakes")
    assert fast.converged and generic.converged
    np.testing.assert_allclose(fast.loglik, generic.loglik, rtol=1e-12)
    p_f, p_g = fast.parameters, generic.parameters
    np.testing.assert_allclose(p_f["estimate"], p_g["estimate"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(p_f["se"], p_g["se"], rtol=1e-5)
