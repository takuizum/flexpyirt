"""Unit and property tests of the M1 kernels, Core functions and Compile layer."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from flexpyirt import BAEM, Model, Normal, Quadrature, Rasch, TwoPL, _core, fit
from flexpyirt._compile import compile_problem, prepare
from flexpyirt._newton import newton_ascent
from flexpyirt.errors import HeywoodWarning, IdentificationError, SpecificationError
from flexpyirt.items import DichotomousKernel
from tests._reference import load


def _simulate(a: np.ndarray, d: np.ndarray, n: int, seed: int, sd: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, sd, size=(n, 1))
    p = 1.0 / (1.0 + np.exp(-(theta * a + d)))
    y = (rng.random(p.shape) < p).astype(int)
    return pd.DataFrame(y, columns=[f"i{j}" for j in range(len(a))])


def _compiled(spec: Model, df: pd.DataFrame) -> tuple[_core.CompiledProblem, jax.Array]:
    prep = prepare(spec, df, group=None, item_names=None, table=None)
    comp = compile_problem(spec, prep.y, prep.group, prep.table, Quadrature.equispaced())
    return comp.problem, jnp.asarray(comp.eta0)


# ---- kernels -------------------------------------------------------------------------------


@settings(deadline=None)  # the first example includes JAX compilation
@given(
    a=st.floats(-4, 4),
    d=st.floats(-6, 6),
)
def test_dichotomous_probabilities_sum_to_one(a: float, d: float) -> None:
    theta = jnp.linspace(-6, 6, 13)
    lp = DichotomousKernel().log_prob(theta, jnp.array([a, d]))
    np.testing.assert_allclose(jnp.exp(lp).sum(axis=1), 1.0, rtol=1e-12)
    np.testing.assert_allclose(
        jnp.exp(lp[:, 1]), 1.0 / (1.0 + np.exp(-(a * theta + d))), rtol=1e-12
    )


# ---- E-step --------------------------------------------------------------------------------


def test_e_step_conserves_counts() -> None:
    df = load("v1_lsat7_2pl").data
    prob, eta = _compiled(Model.uniform(TwoPL(), items=df.columns), df)
    counts, ll = _core.e_step(prob, eta)
    assert float(counts.n.sum()) == pytest.approx(len(df))
    # every respondent answered every item: sum over categories equals the node counts
    np.testing.assert_allclose(
        counts.r[0].sum(axis=2), np.broadcast_to(counts.n[:, None, :], (1, 5, 61))
    )
    assert float(ll) == pytest.approx(float(_core.marginal_loglik(prob, eta)))


def test_oakes_identity_holds_away_from_the_mle() -> None:
    """Oakes' identity is exact at any eta, not only at the MLE."""
    df = load("v2_lsat7_rasch").data
    prob, eta = _compiled(Model.uniform(Rasch(), items=df.columns), df)
    eta = eta + 0.1
    np.testing.assert_allclose(
        _core.information_oakes(prob, eta), _core.information_hessian(prob, eta), rtol=1e-8
    )


# ---- Newton ----------------------------------------------------------------------------------


def test_newton_reaches_interior_optimum_and_keeps_fixed_coordinates() -> None:
    def f(x: jax.Array) -> jax.Array:
        return -((x[0] - 3.0) ** 2) - (x[1] + 1.0) ** 2 - (x[2] - 5.0) ** 2

    x0 = jnp.array([0.0, 0.0, 0.0])
    mask = jnp.array([True, False, True])
    inf = jnp.full(3, jnp.inf)
    x = newton_ascent(f, x0, mask, -inf, inf, 5)
    np.testing.assert_allclose(x, [3.0, 0.0, 5.0], atol=1e-12)


def test_newton_stays_inside_bounds() -> None:
    """Iterates stay strictly inside the bounds and approach a bound that holds the optimum."""

    def f(x: jax.Array) -> jax.Array:
        return -((x[0] - 5.0) ** 2)

    x = newton_ascent(
        f, jnp.array([0.0]), jnp.array([True]), jnp.array([-1.0]), jnp.array([2.0]), 20
    )
    assert 2.0 - 1e-5 < float(x[0]) < 2.0


def test_newton_bound_does_not_block_other_coordinates() -> None:
    """A coordinate pressed against its bound is held while the others still converge."""

    def f(x: jax.Array) -> jax.Array:
        return -((x[0] + 1.0) ** 2) - (x[1] - 3.0) ** 2 - 0.5 * x[0] * x[1]

    x = newton_ascent(
        f,
        jnp.array([0.5, 0.0]),
        jnp.array([True, True]),
        jnp.array([0.0, -jnp.inf]),
        jnp.array([jnp.inf, jnp.inf]),
        60,
    )
    assert 0.0 < float(x[0]) < 1e-7
    # with x0 at its bound 0, the optimum in x1 is 3
    assert float(x[1]) == pytest.approx(3.0, abs=1e-6)


def test_newton_never_decreases_nonconcave_objective() -> None:
    def f(x: jax.Array) -> jax.Array:
        return jnp.sin(3 * x[0]) + jnp.cos(2 * x[1]) * x[0]

    x0 = jnp.array([0.3, -1.2])
    mask = jnp.array([True, True])
    inf = jnp.full(2, jnp.inf)
    x = newton_ascent(f, x0, mask, -inf, inf, 10)
    assert float(f(x)) >= float(f(x0))


# ---- invariance ------------------------------------------------------------------------------


def test_invariant_to_row_and_item_order() -> None:
    case = load("v1_sim_2pl")
    df = case.data
    method = BAEM(tol=1e-9, max_iter=5000)
    base = fit(Model.uniform(TwoPL(), items=case.items), df, method=method, se=None)
    rng = np.random.default_rng(1)
    rows = df.iloc[rng.permutation(len(df))]
    cols = list(rng.permutation(case.items))
    perm = fit(Model.uniform(TwoPL(), items=cols), rows[cols], method=method, se=None)
    a = base.items.set_index("item").loc[case.items, ["a", "d"]]
    b = perm.items.set_index("item").loc[case.items, ["a", "d"]]
    np.testing.assert_allclose(a, b, atol=1e-7)
    assert perm.loglik == pytest.approx(base.loglik, rel=1e-12)


def test_starting_values_do_not_matter() -> None:
    case = load("v1_lsat7_2pl")
    spec = Model.uniform(TwoPL(), items=case.items)
    method = BAEM(tol=1e-9, max_iter=5000)
    base = fit(spec, case.data, method=method, se=None)
    table = spec.parameter_table(case.data)
    items = table["item"] != "__latent__"
    table.loc[items & (table["param"] == "a"), "value"] = 0.5
    table.loc[items & (table["param"] == "d"), "value"] = 0.0
    other = fit(spec, case.data, table=table, method=method, se=None)
    np.testing.assert_allclose(
        base.items[["a", "d"]].to_numpy(), other.items[["a", "d"]].to_numpy(), atol=1e-6
    )


# ---- monotonicity (property) ---------------------------------------------------------------


@settings(max_examples=8, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    seed=st.integers(0, 2**31 - 1),
    rasch=st.booleans(),
)
def test_em_is_monotone(seed: int, rasch: bool) -> None:
    rng = np.random.default_rng(seed)
    a = np.ones(6) if rasch else rng.lognormal(0.0, 0.4, 6)
    d = rng.normal(0.0, 1.0, 6)
    df = _simulate(a, d, 400, seed, sd=1.3 if rasch else 1.0)
    if (df.nunique() < 2).any():
        return
    spec = Model.uniform(Rasch() if rasch else TwoPL(), items=df.columns)
    with warnings.catch_warnings():
        # Random small-N 2PL draws can legitimately hit a finite-sample Heywood slope; EM
        # monotonicity must still hold.
        warnings.simplefilter("ignore", HeywoodWarning)
        res = fit(spec, df, method=BAEM(tol=1e-7, max_iter=300), se=None)
    ll = res.history["loglik"].to_numpy()
    assert np.all(np.diff(ll) >= -1e-9 * np.abs(ll[1:]))
    assert res.loglik >= ll[-1] - 1e-9 * abs(ll[-1])


# ---- validation and identification -----------------------------------------------------------


@pytest.fixture
def small() -> pd.DataFrame:
    return load("v1_lsat7_2pl").data


def test_free_variance_with_free_slopes_is_not_identified(small: pd.DataFrame) -> None:
    spec = Model.uniform(TwoPL(), items=small.columns).with_latent(Normal(free_var=True))
    with pytest.raises(IdentificationError, match="variance"):
        fit(spec, small)


def test_free_mean_without_fixed_intercept_is_not_identified(small: pd.DataFrame) -> None:
    spec = Model.uniform(Rasch(), items=small.columns).with_latent(Normal.free())
    with pytest.raises(IdentificationError, match="mean"):
        fit(spec, small)


def test_fixing_a_slope_identifies_free_variance(small: pd.DataFrame) -> None:
    spec = (
        Model.uniform(TwoPL(), items=small.columns)
        .fix(item="Item.1", a=1.0)
        .with_latent(Normal(free_var=True))
    )
    res = fit(spec, small, se=None)
    assert res.converged
    assert res.items.set_index("item").loc["Item.1", "a"] == 1.0


def test_default_latent_rule() -> None:
    assert Model.uniform(Rasch(), items=["x", "y"]).resolved_latent() == Normal(free_var=True)
    assert Model.uniform(TwoPL(), items=["x", "y"]).resolved_latent() == Normal.standard()
    mixed = Model.from_items({"x": Rasch(), "y": TwoPL()})
    assert mixed.resolved_latent() == Normal.standard()


def test_specification_errors(small: pd.DataFrame) -> None:
    spec = Model.uniform(TwoPL(), items=small.columns)
    with pytest.raises(SpecificationError, match="unknown item"):
        spec.fix(item="nope", a=1.0)
    with pytest.raises(SpecificationError, match="no parameter"):
        spec.fix(item="Item.1", g=0.2)
    with pytest.raises(SpecificationError, match="not in the model"):
        fit(spec, small.assign(extra=0))
    with pytest.raises(SpecificationError, match="not in the data"):
        fit(spec, small.drop(columns="Item.1"))
    with pytest.raises(SpecificationError, match=r"values 0\.\.1"):
        fit(spec, small.replace({1: 2}))
    with pytest.raises(SpecificationError, match="never observed"):
        fit(spec, small.assign(**{"Item.1": 1}))
    with pytest.raises(SpecificationError, match="Duplicated"):
        Model.uniform(TwoPL(), items=["x", "x"])
    with pytest.raises(SpecificationError, match="reserved"):
        Model.uniform(TwoPL(), items=["x", "__latent__"])
    with pytest.raises(SpecificationError, match="reserved"):
        Model.from_items({"__latent__": TwoPL()})


def test_free_id_shared_across_items_is_rejected(small: pd.DataFrame) -> None:
    spec = Model.uniform(TwoPL(), items=small.columns)
    table = spec.parameter_table(small)
    slopes = (table["param"] == "a") & (table["item"] != "__latent__")
    table.loc[slopes, "free_id"] = 0
    with pytest.raises(SpecificationError, match="M5"):
        fit(spec, small, table=table)


def test_table_rows_must_match_model(small: pd.DataFrame) -> None:
    spec = Model.uniform(TwoPL(), items=small.columns)
    table = spec.parameter_table(small)
    with pytest.raises(SpecificationError, match="do not match"):
        fit(spec, small, table=table.iloc[1:])


def test_numpy_input_with_item_names(small: pd.DataFrame) -> None:
    spec = Model.uniform(TwoPL(), items=small.columns)
    a = fit(spec, small, se=None)
    b = fit(spec, small.to_numpy(), item_names=list(small.columns), se=None)
    assert a.loglik == pytest.approx(b.loglik, rel=1e-12)
