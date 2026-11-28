"""The mixture (EMM) M-step of 3PL / 4PL items, ``BAEM(asymptote_m_step="mixture")`` (§15.49).

The mixture objective has the gradient of the usual one at the previous iterate, so both M-steps
reach the same maximum. The mirt reference cases with the option on are in ``test_mirt_m3.py``.
"""

from __future__ import annotations

import json
import warnings
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from flexpyirt import (
    BAEM,
    BetaPrior,
    FitResult,
    FourPL,
    Model,
    Normal,
    ThreePL,
    TwoPL,
    fit,
)
from flexpyirt._core import _item_q, _mixture_asymptote_start, _mixture_item_q
from flexpyirt._json import SCHEMA_VERSION
from flexpyirt.errors import FlexpyirtWarning, SpecificationError
from flexpyirt.items import FourPLKernel

NODES = jnp.linspace(-6.0, 6.0, 31)


def _simulate(
    a: np.ndarray, d: np.ndarray, g: np.ndarray, u: np.ndarray, n: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, 1.0, size=(n, 1))
    p = g + (u - g) / (1.0 + np.exp(-(theta * a + d)))
    y = (rng.random(p.shape) < p).astype(int)
    return pd.DataFrame(y, columns=[f"q{j + 1:02d}" for j in range(len(a))])


def _counts(seed: int) -> jnp.ndarray:
    """Random positive expected counts ``(2, Q)``."""
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.uniform(0.0, 30.0, size=(2, NODES.shape[0])))


# ---- options ----------------------------------------------------------------------------------


def test_default_is_joint() -> None:
    assert BAEM().asymptote_m_step == "joint"


@pytest.mark.parametrize("value", ["emm", "Mixture", "", None])
def test_invalid_value_raises(value: object) -> None:
    with pytest.raises(SpecificationError, match="asymptote_m_step"):
        BAEM(asymptote_m_step=value)  # type: ignore[arg-type]


# ---- the objective ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "params",
    [
        [1.3, -0.4, 0.2, 1.0],  # 3PL
        [0.8, 0.6, 0.15, 0.9],  # 4PL
        [1.1, 0.2, 0.0, 1.0],  # 2PL limit
        [1.1, 0.2, 0.0, 0.85],  # upper asymptote only
    ],
)
def test_gradient_equals_usual_objective_at_previous_iterate(params: list[float]) -> None:
    """Fisher's identity: both objectives have the observed-data gradient at ``xi'``."""
    kernel = FourPLKernel()
    xi = jnp.asarray(params)
    r = _counts(1)
    grad_usual = jax.grad(lambda p: _item_q(kernel, NODES, r, p))(xi)
    grad_mix = jax.grad(lambda p: _mixture_item_q(kernel, NODES, r, p, xi))(xi)
    np.testing.assert_allclose(np.asarray(grad_mix), np.asarray(grad_usual), rtol=1e-10, atol=1e-9)


def test_objective_is_a_minorizer() -> None:
    r"""Q^mix(xi | xi') - Q(xi | xi') is maximal at xi = xi' (EM's information inequality)."""
    kernel = FourPLKernel()
    old = jnp.asarray([1.0, -0.2, 0.2, 0.95])
    r = _counts(2)

    def gap(p: jnp.ndarray) -> jnp.ndarray:
        return _mixture_item_q(kernel, NODES, r, p, old) - _item_q(kernel, NODES, r, p)

    rng = np.random.default_rng(3)
    for _ in range(20):
        p = old + jnp.asarray(rng.normal(0.0, [0.3, 0.3, 0.05, 0.03]))
        assert float(gap(p)) <= float(gap(old)) + 1e-9


@pytest.mark.parametrize(
    ("free", "expected_slot"),
    [((True, True, True, False), 2), ((True, True, False, True), 3), ((True,) * 4, 2)],
)
def test_closed_form_asymptote_zeroes_its_gradient(
    free: tuple[bool, ...], expected_slot: int
) -> None:
    kernel = FourPLKernel()
    old = jnp.asarray([1.2, -0.3, 0.18, 1.0 if not free[3] else 0.92])
    old = old.at[2].set(0.18 if free[2] else 0.1)
    r = _counts(4)[None]  # one group
    loc = jnp.asarray([[0, 1, 2 if free[2] else -1, 3 if free[3] else -1]])
    mask = jnp.asarray(free)
    lo = jnp.asarray([-jnp.inf, -jnp.inf, 0.0, 0.0])
    hi = jnp.asarray([jnp.inf, jnp.inf, 1.0, 1.0])
    start = _mixture_asymptote_start(kernel, NODES, r, old[None], loc, old, mask, lo, hi)
    np.testing.assert_array_equal(np.asarray(start[:2]), np.asarray(old[:2]))
    assert float(start[expected_slot]) != float(old[expected_slot])
    grad = jax.grad(lambda p: _mixture_item_q(kernel, NODES, r[0], p, old))(start)
    for s in (2, 3):
        if free[s]:
            assert abs(float(grad[s])) < 1e-8


# ---- fits -------------------------------------------------------------------------------------


A = np.array([1.4, 1.0, 1.7, 1.2, 0.9, 1.5, 1.1, 1.3])
D = np.array([-0.8, 0.2, -0.3, 0.6, -1.2, 0.0, 0.4, -0.5])
G = np.array([0.2, 0.15, 0.25, 0.1, 0.2, 0.18, 0.22, 0.12])


def _both(spec: Model, df: pd.DataFrame, **kwargs: object) -> tuple[FitResult, FitResult]:
    out = []
    for mode in ("joint", "mixture"):
        method = BAEM(tol=1e-8, max_iter=20000, squarem=True, asymptote_m_step=mode)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FlexpyirtWarning)
            out.append(fit(spec, df, method=method, se=None, **kwargs))  # type: ignore[arg-type]
    return out[0], out[1]


def _assert_same(joint: FitResult, mix: FitResult) -> None:
    """Same maximum. The mixture M-step converges more slowly, so at the same ``tol`` it stops
    farther from the maximum: the tolerances are those of ``tol=1e-8`` at a rate near 1."""
    assert mix.converged
    assert mix.log_posterior == pytest.approx(joint.log_posterior, abs=5e-5)
    for p in ("a", "d", "g", "u"):
        if p in joint.items:
            np.testing.assert_allclose(mix.items[p], joint.items[p], rtol=0, atol=2e-3)
    ll = mix.history["loglik"].to_numpy()
    assert np.all(np.diff(ll) >= -1e-9 * np.abs(ll[1:]))


@cache
def _data_3pl() -> pd.DataFrame:
    return _simulate(A, D, G, np.ones_like(A), n=3000, seed=11)


@pytest.mark.slow
def test_3pl_same_maximum() -> None:
    df = _data_3pl()
    joint, mix = _both(Model.uniform(ThreePL(), items=df.columns), df)
    assert joint.converged
    _assert_same(joint, mix)


@pytest.mark.slow
def test_4pl_and_mixed_models_same_maximum() -> None:
    u = np.array([0.95, 0.9, 0.94, 0.92, 0.97, 0.9, 0.93, 0.96])
    df = _simulate(A, D, G, u, n=4000, seed=12)
    items = list(df.columns)
    # q03 is 3PL: as a 4PL item its slope diverges here (no finite maximum with 8 items).
    four = [items[0], items[1], items[3]]
    models = {c: FourPL() if c in four else ThreePL() for c in items[:7]}
    spec = Model.from_items({**models, items[7]: TwoPL()}).fix(item=items[0], g=0.2)
    joint, mix = _both(spec, df)
    assert joint.converged
    _assert_same(joint, mix)


@pytest.mark.slow
def test_priors_and_groups_same_maximum() -> None:
    df = _data_3pl()
    group = np.repeat(["ref", "focal"], len(df) // 2)
    spec = (
        Model.uniform(ThreePL(), items=df.columns)
        .with_groups(ref=Normal.standard(), focal=Normal.free())
        .equal_across_groups(items=list(df.columns[:6]))
        .equal_across_groups(items=list(df.columns[6:]), params=["a", "d"])
        .prior(param="g", dist=BetaPrior(5.0, 17.0), items=list(df.columns))
    )
    joint, mix = _both(spec, df, group=group)
    assert joint.converged
    _assert_same(joint, mix)


# ---- JSON -------------------------------------------------------------------------------------


def test_json_round_trip_and_default_key_absent() -> None:
    df = _data_3pl().iloc[:400, :4]
    spec = Model.uniform(ThreePL(), items=df.columns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FlexpyirtWarning)
        mix = fit(spec, df, method=BAEM(tol=1e-3, asymptote_m_step="mixture"), se=None)
        joint = fit(spec, df, method=BAEM(tol=1e-3), se=None)
    raw = json.loads(mix.to_json())
    assert raw["schema_version"] == SCHEMA_VERSION == 10
    assert raw["method"]["asymptote_m_step"] == "mixture"
    assert FitResult.from_json(mix.to_json())._method.asymptote_m_step == "mixture"
    assert "asymptote_m_step" not in json.loads(joint.to_json())["method"]
    assert FitResult.from_json(joint.to_json())._method.asymptote_m_step == "joint"
    # A version-9 file cannot carry the key.
    raw["schema_version"] = 9
    with pytest.raises(SpecificationError, match="asymptote_m_step"):
        FitResult.from_json(json.dumps(raw))
