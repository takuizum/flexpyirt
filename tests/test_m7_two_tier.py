"""Unit tests for two-tier models (DESIGN.md §17.2).

The reduced E-step (Cai, 2010) is checked against brute-force integration over the full
``(P + S)``-dimensional product grid; the agreement with mirt's ``bfactor(model2 = )`` is in
``tests/test_mirt_m7_two_tier.py`` (VALIDATION.md V33).
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
    TwoPL,
    _core,
    fit,
)
from flexpyirt._compile import compile_problem, prepare
from flexpyirt.errors import IdentificationError, SpecificationError
from flexpyirt.items import ItemModel

ITEMS = [f"i{k}" for k in range(10)]
# i4 cross-loads on both primary factors; i8 and i9 have no specific factor.
PRIMARY = {"p1": ITEMS[:5], "p2": ITEMS[4:]}
SPECIFIC = {"s1": ITEMS[:3], "s2": ITEMS[5:8]}
RHO = 0.45


def _sim(n: int = 600, seed: int = 21) -> pd.DataFrame:
    """2PL items and a 3-category GRM item (i6); 6% missing at random."""
    rng = np.random.default_rng(seed)
    cov = np.eye(4)
    cov[0, 1] = cov[1, 0] = RHO
    theta = rng.multivariate_normal(np.zeros(4), cov, size=n)
    a = np.zeros((10, 4))
    a[:5, 0] = rng.uniform(1.0, 1.7, 5)
    a[4:, 1] = rng.uniform(1.0, 1.7, 6)
    a[:3, 2] = rng.uniform(0.6, 1.1, 3)
    a[5:8, 3] = rng.uniform(0.6, 1.1, 3)
    z = theta @ a.T + rng.normal(0.0, 0.6, 10)
    y = (rng.random((n, 10)) < 1.0 / (1.0 + np.exp(-z))).astype(float)
    y[:, 6] = np.digitize(z[:, 6] + rng.logistic(size=n), [-0.8, 0.8])
    y[rng.random(y.shape) < 0.06] = np.nan
    return pd.DataFrame(y, columns=ITEMS)


def _spec(*, correlated: bool = True) -> Model:
    models: dict[str, ItemModel] = {name: TwoPL() for name in ITEMS}
    models["i6"] = GRM(3)
    return Model.from_items(models).with_two_tier(
        primary=PRIMARY, specific=SPECIFIC, correlated=correlated
    )


def _problem(
    spec: Model, data: pd.DataFrame, quad: Quadrature
) -> tuple[_core.CompiledProblem, np.ndarray]:
    prep = prepare(spec, data, group=None, item_names=None, table=None)
    comp = compile_problem(spec, prep.y, prep.group, prep.table, quad)
    return comp.problem, comp.eta0


def _full_log_joint(prob: _core.CompiledProblem, eta: jax.Array) -> tuple[jax.Array, jax.Array]:
    r"""Brute-force log joint on the full ``(n^D, D)`` grid.

    The weights are the product of the primary weights (multivariate normal normalized on the
    primary grid) and each specific factor's one-dimensional weights.
    """
    red = prob.reduction
    assert red is not None
    n_primary, n_specific = red.n_primary, red.n_specific
    x = red.nodes_1d
    n_pts = x.shape[0]
    log_wp, log_wb = _core._bifactor_log_weights(prob, eta)
    axes = [x] * (n_primary + n_specific)
    grid = jnp.stack([m.ravel() for m in jnp.meshgrid(*axes, indexing="ij")], axis=1)
    shape = (n_pts,) * (n_primary + n_specific)
    logw = log_wp[0].reshape((n_pts,) * n_primary + (1,) * n_specific)
    for s in range(n_specific):
        view = [1] * (n_primary + n_specific)
        view[n_primary + s] = n_pts
        logw = logw + log_wb[0, s].reshape(view)
    logw = jnp.broadcast_to(logw, shape).ravel()
    full = replace(prob, nodes=grid, node_log_base=jnp.zeros(grid.shape[0]), reduction=None)
    ell = sum(
        jnp.einsum("pjk,gjqk->pq", oh, lp)
        for oh, lp in zip(_core._one_hots(full), _core._item_log_probs(full, eta), strict=True)
    )
    return ell + logw, grid


@cache
def _setup() -> tuple[_core.CompiledProblem, jax.Array, Quadrature]:
    quad = Quadrature.equispaced(7)
    prob, eta0 = _problem(_spec(), _sim(n=300), quad)
    rng = np.random.default_rng(0)
    eta = np.asarray(eta0 + 0.2 * rng.standard_normal(eta0.shape))
    eta[int(prob.latent.free_index[0, -1])] = 0.4  # cor_12
    return prob, jnp.asarray(eta), quad


# ---- specification -------------------------------------------------------------------------


def test_with_two_tier_structure() -> None:
    spec = _spec()
    assert not spec.bifactor and spec.is_reduced
    assert spec.n_primary_factors == 2 and spec.n_primary == 2
    assert spec.n_factors == 4 and spec.quadrature_dimension == 3
    assert spec.factor_names == ("p1", "p2", "s1", "s2")
    assert spec.item_factors("i0") == (0, 2) and spec.item_factors("i4") == (0, 1)
    assert spec.item_factors("i9") == (1,)
    names = ("mean_1", "mean_2", "mean_3", "mean_4", "var_1", "var_2", "var_3", "var_4", "cor_12")
    assert spec.latent_parameter_names() == names
    table = spec.parameter_table(_sim())
    items = table[table["item"] != "__latent__"].set_index(["item", "param"])
    assert items.loc[("i0", "a1"), "free"] and items.loc[("i0", "a3"), "free"]
    assert not items.loc[("i0", "a2"), "free"] and items.loc[("i0", "a2"), "value"] == 0.0
    assert items.loc[("i4", "a1"), "free"] and items.loc[("i4", "a2"), "free"]
    assert not items.loc[("i9", "a4"), "free"]
    latent = table[table["item"] == "__latent__"].set_index("param")
    assert tuple(latent.index) == names
    assert list(latent.index[latent["free"]]) == ["cor_12"]
    uncorrelated = _spec(correlated=False).parameter_table(_sim())
    assert not uncorrelated.loc[uncorrelated["item"] == "__latent__", "free"].any()


def test_default_quadrature_matches_mirt_bfactor() -> None:
    """mirt's ``bfactor()`` uses ``select_quadpts(nfact - nspec + 1) = 15`` points for P = 2."""
    assert Quadrature.default(_spec().quadrature_dimension) == Quadrature.equispaced(15)


def _base() -> Model:
    return Model.uniform(TwoPL(), items=ITEMS)


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda: _base().with_two_tier(primary={}, specific=SPECIFIC), "primary"),
        (lambda: _base().with_two_tier(primary=PRIMARY, specific={}), "specific"),
        (
            lambda: _base().with_two_tier(
                primary={"a": ITEMS[:4], "b": ITEMS[4:7], "c": ITEMS[7:]}, specific=SPECIFIC
            ),
            "1 to 2 primary",
        ),
        (
            lambda: _base().with_two_tier(primary={"a": ITEMS[:5]}, specific=SPECIFIC),
            "not assigned to any primary",
        ),
        (
            lambda: _base().with_two_tier(primary=PRIMARY, specific={"p1": ITEMS[:3]}),
            "Duplicated factor name",
        ),
        (
            lambda: _base().with_two_tier(primary=PRIMARY, specific={"s": ITEMS[:1]}),
            "at least two items",
        ),
        (
            lambda: _base().with_two_tier(
                primary=PRIMARY, specific={"s": ITEMS[:3], "t": ITEMS[2:5]}
            ),
            "at most one specific",
        ),
        (
            lambda: _spec().with_factors({"a": ITEMS[:5], "b": ITEMS[5:]}),
            "two-tier",
        ),
        (lambda: _spec().with_bifactor(specific=SPECIFIC), "already has factors"),
        (
            lambda: (
                _base()
                .with_factors({"a": ITEMS[:5], "b": ITEMS[5:]})
                .with_two_tier(primary=PRIMARY, specific=SPECIFIC)
            ),
            "already has factors",
        ),
        (
            lambda: (
                _base()
                .with_groups(x=Normal(), y=Normal(free_mean=True, free_var=True))
                .with_two_tier(primary=PRIMARY, specific=SPECIFIC)
            ),
            "multiple groups",
        ),
    ],
)
def test_with_two_tier_rejects(build: object, match: str) -> None:
    with pytest.raises(SpecificationError, match=match):
        build()  # type: ignore[operator]


@pytest.mark.parametrize("item_model", [Rasch(), PCM(3)])
def test_with_two_tier_rejects_fixed_slopes(item_model: ItemModel) -> None:
    spec = Model.uniform(item_model, items=ITEMS)
    with pytest.raises(SpecificationError, match="Rasch / PCM"):
        spec.with_two_tier(primary=PRIMARY, specific=SPECIFIC)


def test_free_latent_moment_rejected() -> None:
    table = _spec().parameter_table(_sim())
    mask = (table["item"] == "__latent__") & (table["param"] == "var_3")
    table.loc[mask, "free"] = True
    table.loc[mask, "free_id"] = int(table["free_id"].max()) + 1
    with pytest.raises(SpecificationError, match="free latent means or variances"):
        fit(_spec(), _sim(), table=table, se=None)


def test_identical_primary_loadings_are_unidentified() -> None:
    """Two primary factors on the same items with a free correlation are a rotation."""
    spec = _base().with_two_tier(primary={"p1": ITEMS, "p2": ITEMS}, specific={"s": ITEMS[:3]})
    with pytest.raises(IdentificationError):
        fit(spec, _sim().fillna(0.0).clip(upper=1.0), se=None)


# ---- kernel against brute force ------------------------------------------------------------


def test_marginal_loglik_matches_full_grid() -> None:
    prob, eta, _ = _setup()
    log_joint, _ = _full_log_joint(prob, eta)
    expected = jnp.sum(prob.counts * jax.scipy.special.logsumexp(log_joint, axis=1))
    got = _core.marginal_loglik(prob, eta)
    assert float(got) == pytest.approx(float(expected), rel=1e-13, abs=1e-9)


def test_eap_matches_full_grid() -> None:
    prob, eta, _ = _setup()
    log_joint, grid = _full_log_joint(prob, eta)
    h = jax.nn.softmax(log_joint, axis=1)
    mean = h @ grid
    sd = jnp.sqrt(h @ grid**2 - mean**2)
    got_mean, got_sd = _core.eap(prob, eta)
    np.testing.assert_allclose(got_mean, mean, rtol=0, atol=1e-12)
    np.testing.assert_allclose(got_sd, sd, rtol=0, atol=1e-12)


def test_expected_counts_match_full_grid() -> None:
    """Block counts summed over the grid equal the full-grid counts; marginals are consistent."""
    prob, eta, quad = _setup()
    red = prob.reduction
    assert red is not None
    log_joint, grid = _full_log_joint(prob, eta)
    h = jax.nn.softmax(log_joint, axis=1)
    counts, loglik = _core.e_step(prob, eta)
    assert float(loglik) == pytest.approx(float(_core.marginal_loglik(prob, eta)), rel=1e-14)
    for oh, r in zip(_core._one_hots(prob), counts.r, strict=True):
        expected = jnp.einsum("p,pjk,pq->jk", prob.counts, oh, h)
        np.testing.assert_allclose(jnp.sum(r[0], axis=-1), expected, rtol=1e-12, atol=1e-9)
    n_pts = quad.n_points
    n_prim = n_pts**2
    # The primary marginal of the flat ``n`` equals the full-grid posterior summed over the
    # specific dimensions (the primary index is major in the full grid as well).
    full_n = prob.counts @ h
    primary_n = full_n.reshape(n_prim, -1).sum(axis=1)
    np.testing.assert_allclose(counts.n[0, :n_prim], primary_n, rtol=1e-11, atol=1e-10)
    blocks = counts.n[0, n_prim:].reshape(red.n_specific + 1, n_pts)
    full_4d = full_n.reshape((n_pts,) * 4)
    np.testing.assert_allclose(blocks[0], full_4d.sum(axis=(0, 1, 3)), rtol=1e-11, atol=1e-10)
    np.testing.assert_allclose(blocks[1], full_4d.sum(axis=(0, 1, 2)), rtol=1e-11, atol=1e-10)
    np.testing.assert_allclose(grid.shape, (n_pts**4, 4))


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


@pytest.mark.slow
def test_m_step_moves_the_correlation_and_does_not_decrease_q_or_loglik() -> None:
    prob, eta, _ = _setup()
    counts, ll0 = _core.e_step(prob, eta)
    new_eta = _core.m_step(prob, eta, counts, 3)
    cor = int(prob.latent.free_index[0, -1])
    assert float(new_eta[cor]) != float(eta[cor]) and abs(float(new_eta[cor])) < 1.0
    q0 = _core.expected_complete_loglik(prob, eta, counts)
    q1 = _core.expected_complete_loglik(prob, new_eta, counts)
    assert float(q1) >= float(q0) - 1e-10
    assert float(_core.marginal_loglik(prob, new_eta)) >= float(ll0) - 1e-10


def test_one_primary_factor_equals_bifactor() -> None:
    """``P = 1`` with the primary factor on every item is the bifactor model."""
    data = _sim(n=300)
    base = Model.uniform(TwoPL(), items=ITEMS[:6] + ITEMS[7:])
    data = data[list(base.item_names)]
    specific = {"s1": ITEMS[:3], "s2": ITEMS[7:]}
    two_tier = base.with_two_tier(primary={"g": list(base.item_names)}, specific=specific)
    bifactor = base.with_bifactor(specific=specific, general="g")
    assert two_tier.latent_parameter_names() == bifactor.latent_parameter_names()
    assert two_tier.quadrature_dimension == 2
    quad = Quadrature.equispaced(9)
    prob_t, eta0 = _problem(two_tier, data, quad)
    prob_b, _ = _problem(bifactor, data, quad)
    eta = jnp.asarray(eta0)
    assert float(_core.marginal_loglik(prob_t, eta)) == pytest.approx(
        float(_core.marginal_loglik(prob_b, eta)), rel=1e-14
    )
    np.testing.assert_allclose(_core.eap(prob_t, eta)[0], _core.eap(prob_b, eta)[0], atol=1e-13)


# ---- the 3D confirmatory path --------------------------------------------------------------


@cache
def _one_specific() -> tuple[FitResult, FitResult]:
    """Two primary factors and one specific factor: also a 3D confirmatory model."""
    items = ITEMS[:6] + ITEMS[7:]
    data = _sim(n=800, seed=5)[items]
    base = Model.uniform(TwoPL(), items=items)
    primary = {"p1": items[:5], "p2": items[4:]}
    two_tier = base.with_two_tier(primary=primary, specific={"s": items[:3]})
    table = base.with_factors({**primary, "s": items[:3]}).parameter_table(data)
    lat = table["item"] == "__latent__"
    fixed = lat & table["param"].isin(["cor_13", "cor_23"])
    table.loc[fixed, "free"] = False
    table.loc[fixed, "free_id"] = -1
    conf = base.with_factors({**primary, "s": items[:3]})
    quad = Quadrature.equispaced(11)
    method = BAEM(tol=1e-9, max_iter=20000)
    return (
        fit(two_tier, data, quadrature=quad, method=method, se="oakes"),
        fit(conf, data, table=table, quadrature=quad, method=method, se="oakes"),
    )


@pytest.mark.slow
def test_one_specific_factor_equals_confirmatory_3d() -> None:
    two_tier, conf = _one_specific()
    assert two_tier.converged and conf.converged
    assert two_tier.loglik == pytest.approx(conf.loglik, rel=1e-12)
    cols = ["a1", "a2", "a3", "d", "a1_se", "a2_se", "a3_se", "d_se"]
    np.testing.assert_allclose(
        two_tier.items[cols].to_numpy(dtype=float),
        conf.items[cols].to_numpy(dtype=float),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        two_tier.latent[["cor_12", "cor_12_se"]].to_numpy(dtype=float),
        conf.latent[["cor_12", "cor_12_se"]].to_numpy(dtype=float),
        rtol=1e-5,
    )
    data = _sim(n=800, seed=5)[list(two_tier._model.item_names)].iloc[:40]
    np.testing.assert_allclose(
        two_tier.score(data).to_numpy(), conf.score(data).to_numpy(), rtol=0, atol=1e-5
    )


# ---- fitting, SQUAREM, JSON and unsupported features ---------------------------------------


@cache
def _fitted(squarem: bool = False) -> FitResult:
    return fit(
        _spec(),
        _sim(),
        quadrature=Quadrature.equispaced(9),
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
    assert 0.0 < float(plain.latent["cor_12"].iloc[0]) < 1.0
    assert float(accel.latent["cor_12"].iloc[0]) == pytest.approx(
        float(plain.latent["cor_12"].iloc[0]), abs=1e-4
    )
    hess = fit(
        _spec(),
        _sim(),
        quadrature=Quadrature.equispaced(9),
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
    assert float(hess.latent["cor_12_se"].iloc[0]) == pytest.approx(
        float(plain.latent["cor_12_se"].iloc[0]), rel=1e-4
    )


@pytest.mark.slow
def test_json_round_trip_and_summary() -> None:
    res = _fitted(False)
    payload = json.loads(res.to_json())
    assert payload["schema_version"] == 6
    assert payload["model"]["n_primary_factors"] == 2 and payload["model"]["bifactor"] is False
    loaded = FitResult.from_json(json.dumps(payload))
    assert loaded._model == res._model
    pd.testing.assert_frame_equal(loaded.items, res.items, check_exact=True)
    pd.testing.assert_frame_equal(loaded.latent, res.latent, check_exact=True)
    data = _sim().iloc[:20]
    pd.testing.assert_frame_equal(loaded.score(data), res.score(data), check_exact=True)
    assert "two-tier: primary p1, p2; specific s1, s2" in res.summary()
    assert list(res.score(data).columns) == [
        f"{kind}_{name}" for kind in ("theta", "se") for name in ("p1", "p2", "s1", "s2")
    ]
    bad = json.loads(res.to_json())
    bad["model"]["n_primary_factors"] = 4
    with pytest.raises(SpecificationError, match="n_primary_factors"):
        FitResult.from_json(json.dumps(bad))
    old = json.loads(res.to_json())
    old["schema_version"] = 5
    with pytest.raises(SpecificationError, match="n_primary_factors"):
        FitResult.from_json(json.dumps(old))


@pytest.mark.slow
@pytest.mark.parametrize("method", ["MAP", "ML", "WLE"])
def test_unsupported_scoring_methods(method: str) -> None:
    with pytest.raises(SpecificationError, match="two-tier"):
        _fitted(False).score(_sim().iloc[:5], method=method)  # type: ignore[arg-type]
