"""Unit tests for bifactor models with Gibbons–Hedeker dimension reduction (DESIGN.md §17.1).

The reduced E-step is checked against brute-force integration over the full ``(1 + S)``-dimensional
product grid, Oakes' identity against the Hessian, and a one-specific-factor bifactor model against
the validated two-dimensional confirmatory path (``with_factors(..., correlated=False)``).
"""

from __future__ import annotations

import json
from dataclasses import replace
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from flexpyirt import (
    BAEM,
    GRM,
    PCM,
    FitResult,
    Model,
    Normal,
    Quadrature,
    Rasch,
    ThreePL,
    TwoPL,
    _core,
    fit,
)
from flexpyirt._compile import compile_problem, prepare
from flexpyirt.errors import SpecificationError
from flexpyirt.items import ItemModel

ITEMS = [f"i{k}" for k in range(8)]
SPECIFIC = {"s1": ITEMS[:3], "s2": ITEMS[3:6]}  # i6, i7 load on the general factor only


def _sim(n: int = 600, seed: int = 11) -> pd.DataFrame:
    """2PL items i0-i5 and i7, a 3-category GRM item i6; 8% missing at random."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=(n, 3))
    spec_of = [1, 1, 1, 2, 2, 2, 0, 0]
    a0 = rng.uniform(1.0, 1.8, 8)
    a_s = np.where(np.asarray(spec_of) > 0, rng.uniform(0.6, 1.2, 8), 0.0)
    z = a0 * theta[:, [0]] + a_s * theta[:, spec_of] + rng.normal(0.0, 0.7, 8)
    y = (rng.random((n, 8)) < 1.0 / (1.0 + np.exp(-z))).astype(float)
    y[:, 6] = np.digitize(z[:, 6] + rng.logistic(size=n), [-0.8, 0.8])
    y[rng.random(y.shape) < 0.08] = np.nan
    return pd.DataFrame(y, columns=ITEMS)


def _spec() -> Model:
    models: dict[str, ItemModel] = {name: TwoPL() for name in ITEMS}
    models["i6"] = GRM(3)
    return Model.from_items(models).with_bifactor(specific=SPECIFIC, general="g")


def _problem(
    spec: Model, data: pd.DataFrame, quad: Quadrature
) -> tuple[_core.CompiledProblem, np.ndarray]:
    prep = prepare(spec, data, group=None, item_names=None, table=None)
    comp = compile_problem(spec, prep.y, prep.group, prep.table, quad)
    return comp.problem, comp.eta0


def _full_grid(prob: _core.CompiledProblem, quad: Quadrature) -> tuple[jax.Array, jax.Array]:
    """Brute-force ``(n^D, D)`` product grid and its standard-normal log weights."""
    n_factors = prob.nodes.shape[1]
    assert prob.reduction is not None
    grid = jnp.asarray(quad.grid(n_factors))
    x = prob.reduction.nodes_1d
    logw1 = jax.nn.log_softmax(jnp.asarray(quad.base_log_weights(1)) - 0.5 * x**2)
    meshes = jnp.meshgrid(*([logw1] * n_factors), indexing="ij")
    logw = sum(m.ravel() for m in meshes)
    return grid, jnp.asarray(logw)


def _full_log_joint(
    prob: _core.CompiledProblem, quad: Quadrature, eta: jax.Array
) -> tuple[jax.Array, jax.Array]:
    grid, logw = _full_grid(prob, quad)
    full = replace(prob, nodes=grid, node_log_base=jnp.zeros(grid.shape[0]), reduction=None)
    ell = sum(
        jnp.einsum("pjk,gjqk->pq", oh, lp)
        for oh, lp in zip(_core._one_hots(full), _core._item_log_probs(full, eta), strict=True)
    )
    return ell + logw, grid


@cache
def _setup() -> tuple[_core.CompiledProblem, jax.Array, Quadrature]:
    quad = Quadrature.equispaced(9)
    prob, eta0 = _problem(_spec(), _sim(n=300), quad)
    rng = np.random.default_rng(0)
    eta = jnp.asarray(eta0 + 0.2 * rng.standard_normal(eta0.shape))
    return prob, eta, quad


# ---- specification -------------------------------------------------------------------------


def test_with_bifactor_structure() -> None:
    spec = _spec()
    assert spec.bifactor and spec.n_factors == 3 and spec.quadrature_dimension == 2
    assert spec.factor_names == ("g", "s1", "s2")
    assert spec.item_factors("i0") == (0, 1) and spec.item_factors("i7") == (0,)
    assert spec.latent_parameter_names() == (
        "mean_1",
        "mean_2",
        "mean_3",
        "var_1",
        "var_2",
        "var_3",
    )
    table = spec.parameter_table(_sim())
    items = table[table["item"] != "__latent__"].set_index(["item", "param"])
    assert items.loc[("i0", "a3"), "value"] == 0.0 and not items.loc[("i0", "a3"), "free"]
    assert items.loc[("i4", "a3"), "free"] and not items.loc[("i4", "a2"), "free"]
    assert not items.loc[("i7", "a2"), "free"] and items.loc[("i7", "a1"), "free"]
    latent = table[table["item"] == "__latent__"]
    assert list(latent["param"]) == list(spec.latent_parameter_names())
    assert not latent["free"].any()


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda m: m.with_bifactor(specific={}), "non-empty"),
        (lambda m: m.with_bifactor(specific={"s": ["i0"]}), "at least two items"),
        (
            lambda m: m.with_bifactor(specific={"s": ["i0", "i1"], "t": ["i1", "i2"]}),
            "at most one specific",
        ),
        (lambda m: m.with_bifactor(specific={"s": ["i0", "nope"]}), "Unknown items"),
        (lambda m: m.with_bifactor(specific={"g": ["i0", "i1"]}, general="g"), "Duplicated"),
        (lambda m: m.with_bifactor(specific={"s": "i0"}), "single string"),
        (lambda m: m.with_bifactor(specific={"s": ["i0", "i1"]}, general=" "), "non-empty"),
        (
            lambda m: m.with_bifactor(specific={"s": ["i0", "i1"]}).with_factors(
                {"a": ["i0"], "b": ["i1"]}
            ),
            "cannot be combined",
        ),
        (
            lambda m: m.with_factors({"a": ITEMS[:4], "b": ITEMS[4:]}).with_bifactor(
                specific={"s": ["i0", "i1"]}
            ),
            "already has factors",
        ),
        (
            lambda m: m.with_bifactor(specific={"s": ["i0", "i1"]}).with_groups(
                a=Normal.standard(), b=Normal.free()
            ),
            "multiple groups",
        ),
        (
            lambda m: m.with_groups(a=Normal.standard(), b=Normal.free()).with_bifactor(
                specific={"s": ["i0", "i1"]}
            ),
            "multiple groups",
        ),
        (
            lambda m: m.with_latent(Normal.standard()).with_bifactor(specific={"s": ["i0", "i1"]}),
            "with_latent",
        ),
        (lambda m: m.fix(item="i0", a=1.0).with_bifactor(specific={"s": ["i0", "i1"]}), "a1"),
    ],
)
def test_with_bifactor_rejects(build: object, match: str) -> None:
    spec = Model.uniform(TwoPL(), items=ITEMS)
    with pytest.raises(SpecificationError, match=match):
        build(spec)  # type: ignore[operator]


@pytest.mark.parametrize("item_model", [Rasch(), PCM(3)])
def test_with_bifactor_rejects_fixed_slopes(item_model: object) -> None:
    spec = Model.uniform(item_model, items=ITEMS)  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match="free slopes"):
        spec.with_bifactor(specific={"s": ["i0", "i1"]})


def test_free_latent_parameter_rejected() -> None:
    spec, data = _spec(), _sim()
    table = spec.parameter_table(data)
    table.loc[(table["item"] == "__latent__") & (table["param"] == "var_2"), "free"] = True
    with pytest.raises(SpecificationError, match="free latent"):
        fit(spec, data, table=table, quadrature=Quadrature.equispaced(7), se=None)


def test_default_quadrature_matches_mirt_bfactor() -> None:
    """mirt's ``bfactor()`` uses ``select_quadpts(nfact - nspec + 1) = 31`` points."""
    res = fit(_spec(), _sim(n=200), method=BAEM(tol=1e-2), se=None)
    assert res._quadrature == Quadrature.equispaced(31)


# ---- kernel against brute force ------------------------------------------------------------


def test_marginal_loglik_matches_full_grid() -> None:
    prob, eta, quad = _setup()
    log_joint, _ = _full_log_joint(prob, quad, eta)
    expected = jnp.sum(prob.counts * jax.scipy.special.logsumexp(log_joint, axis=1))
    got = _core.marginal_loglik(prob, eta)
    assert float(got) == pytest.approx(float(expected), rel=1e-13, abs=1e-9)


def test_eap_matches_full_grid() -> None:
    prob, eta, quad = _setup()
    log_joint, grid = _full_log_joint(prob, quad, eta)
    h = jax.nn.softmax(log_joint, axis=1)
    mean = h @ grid
    sd = jnp.sqrt(h @ grid**2 - mean**2)
    got_mean, got_sd = _core.eap(prob, eta)
    np.testing.assert_allclose(got_mean, mean, rtol=0, atol=1e-12)
    np.testing.assert_allclose(got_sd, sd, rtol=0, atol=1e-12)


def test_expected_counts_match_full_grid() -> None:
    """Block counts summed over the grid equal the full-grid counts of each item and category."""
    prob, eta, quad = _setup()
    log_joint, _ = _full_log_joint(prob, quad, eta)
    h = jax.nn.softmax(log_joint, axis=1)
    counts, loglik = _core.e_step(prob, eta)
    assert float(loglik) == pytest.approx(float(_core.marginal_loglik(prob, eta)), rel=1e-14)
    for oh, r in zip(_core._one_hots(prob), counts.r, strict=True):
        expected = jnp.einsum("p,pjk,pq->jk", prob.counts, oh, h)
        np.testing.assert_allclose(jnp.sum(r[0], axis=-1), expected, rtol=1e-12, atol=1e-9)
    n_total = float(jnp.sum(prob.counts))
    np.testing.assert_allclose(jnp.sum(counts.n, axis=-1), n_total, rtol=1e-13)


@pytest.mark.slow
def test_fisher_identity_and_oakes_equal_hessian() -> None:
    prob, eta, _ = _setup()
    counts, _ = _core.e_step(prob, eta)
    grad_q = jax.grad(lambda e: _core.expected_complete_loglik(prob, e, counts))(eta)
    grad_l = jax.grad(lambda e: _core.marginal_loglik(prob, e))(eta)
    np.testing.assert_allclose(grad_q, grad_l, rtol=0, atol=1e-9)
    info_oakes = _core.information_oakes(prob, eta)
    info_hess = _core.information_hessian(prob, eta)
    np.testing.assert_allclose(info_oakes, info_hess, rtol=0, atol=1e-9)


def test_m_step_does_not_decrease_q_or_loglik() -> None:
    prob, eta, _ = _setup()
    counts, ll0 = _core.e_step(prob, eta)
    new_eta = _core.m_step(prob, eta, counts, 3)
    q0 = _core.expected_complete_loglik(prob, eta, counts)
    q1 = _core.expected_complete_loglik(prob, new_eta, counts)
    assert float(q1) >= float(q0) - 1e-10
    assert float(_core.marginal_loglik(prob, new_eta)) >= float(ll0) - 1e-10


def test_general_only_block_is_exact() -> None:
    """A model whose items all have a specific factor gives the same loglik with no dummy items."""
    data = _sim()[ITEMS[:6]]
    spec = Model.uniform(TwoPL(), items=ITEMS[:6]).with_bifactor(specific=SPECIFIC)
    quad = Quadrature.equispaced(7)
    prob, eta0 = _problem(spec, data, quad)
    log_joint, _ = _full_log_joint(prob, quad, jnp.asarray(eta0))
    expected = jnp.sum(prob.counts * jax.scipy.special.logsumexp(log_joint, axis=1))
    assert float(_core.marginal_loglik(prob, jnp.asarray(eta0))) == pytest.approx(
        float(expected), rel=1e-13
    )


# ---- one specific factor equals the 2D confirmatory path ----------------------------------


@cache
def _one_specific() -> tuple[FitResult, FitResult]:
    data = _sim(n=800, seed=5)[ITEMS[:6]]
    base = Model.uniform(TwoPL(), items=ITEMS[:6])
    bif = base.with_bifactor(specific={"s": ITEMS[:3]}, general="g")
    conf = base.with_factors({"g": ITEMS[:6], "s": ITEMS[:3]}, correlated=False)
    quad = Quadrature.equispaced(15)
    method = BAEM(tol=1e-9, max_iter=20000)
    return (
        fit(bif, data, quadrature=quad, method=method, se="oakes"),
        fit(conf, data, quadrature=quad, method=method, se="oakes"),
    )


@pytest.mark.slow
def test_one_specific_factor_equals_confirmatory_2d() -> None:
    bif, conf = _one_specific()
    assert bif.converged and conf.converged
    assert bif.loglik == pytest.approx(conf.loglik, rel=1e-12)
    cols = ["a1", "a2", "d", "a1_se", "a2_se", "d_se"]
    np.testing.assert_allclose(
        bif.items[cols].to_numpy(dtype=float),
        conf.items[cols].to_numpy(dtype=float),
        rtol=1e-5,
        atol=1e-6,
    )
    data = _sim(n=800, seed=5)[ITEMS[:6]].iloc[:40]
    np.testing.assert_allclose(
        bif.score(data).to_numpy(), conf.score(data).to_numpy(), rtol=0, atol=1e-5
    )


# ---- fitting, SQUAREM, JSON and unsupported features ---------------------------------------


@cache
def _fitted(squarem: bool = False) -> FitResult:
    return fit(
        _spec(),
        _sim(),
        quadrature=Quadrature.equispaced(11),
        method=BAEM(tol=1e-8, max_iter=20000, squarem=squarem),
        se="oakes",
    )


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:EM did not converge")
def test_fit_converges_and_squarem_agrees() -> None:
    plain, accel = _fitted(False), _fitted(True)
    assert plain.converged and accel.converged
    hist = plain.history["log_posterior"].to_numpy()
    assert np.all(np.diff(hist[np.isfinite(hist)]) >= -1e-8)
    assert accel.loglik == pytest.approx(plain.loglik, abs=1e-6)
    cols = [c for c in plain.items.columns if c.startswith(("a", "d")) and not c.endswith("_se")]
    np.testing.assert_allclose(
        accel.items[cols].to_numpy(dtype=float),
        plain.items[cols].to_numpy(dtype=float),
        atol=1e-4,
    )
    hess = fit(
        _spec(),
        _sim(),
        quadrature=Quadrature.equispaced(11),
        table=plain.parameters.assign(value=plain.parameters["estimate"]),
        method=BAEM(tol=1e-8, max_iter=1),
        se="hessian",
    )
    se_cols = [c for c in plain.items.columns if c.endswith("_se")]
    np.testing.assert_allclose(
        hess.items[se_cols].to_numpy(dtype=float),
        plain.items[se_cols].to_numpy(dtype=float),
        rtol=1e-4,
    )


@pytest.mark.slow
def test_json_round_trip_and_summary() -> None:
    res = _fitted(False)
    payload = json.loads(res.to_json())
    assert payload["model"]["bifactor"] is True
    loaded = FitResult.from_json(json.dumps(payload))
    assert loaded._model == res._model
    pd.testing.assert_frame_equal(loaded.items, res.items, check_exact=True)
    pd.testing.assert_frame_equal(loaded.latent, res.latent, check_exact=True)
    data = _sim().iloc[:20]
    pd.testing.assert_frame_equal(loaded.score(data), res.score(data), check_exact=True)
    assert "bifactor: general g; specific s1, s2" in res.summary()
    assert list(res.score(data).columns) == [
        "theta_g",
        "theta_s1",
        "theta_s2",
        "se_g",
        "se_s1",
        "se_s2",
    ]
    bad = json.loads(res.to_json())
    bad["model"]["factors"][0]["items"] = ITEMS[:6]
    with pytest.raises(SpecificationError, match="general factor"):
        FitResult.from_json(json.dumps(bad))
    old = json.loads(res.to_json())
    old["schema_version"] = 4
    with pytest.raises(SpecificationError, match="bifactor"):
        FitResult.from_json(json.dumps(old))


@pytest.mark.slow
@pytest.mark.parametrize("method", ["MAP", "ML", "WLE"])
def test_unsupported_scoring_methods(method: str) -> None:
    with pytest.raises(SpecificationError, match="bifactor"):
        _fitted(False).score(_sim().iloc[:5], method=method)  # type: ignore[arg-type]


def test_three_pl_bifactor_compiles() -> None:
    """Every item family that supports factors works with the reduction (here 3PL + GRM)."""
    models: dict[str, ItemModel] = {name: ThreePL() for name in ITEMS}
    models["i6"] = GRM(3)
    spec = Model.from_items(models).with_bifactor(specific=SPECIFIC)
    quad = Quadrature.equispaced(7)
    prob, eta0 = _problem(spec, _sim(n=200), quad)
    eta = jnp.asarray(eta0)
    log_joint, _ = _full_log_joint(prob, quad, eta)
    expected = jnp.sum(prob.counts * jax.scipy.special.logsumexp(log_joint, axis=1))
    assert float(_core.marginal_loglik(prob, eta)) == pytest.approx(float(expected), rel=1e-13)
