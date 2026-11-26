"""V54: ``simulate_responses()`` (VALIDATION.md V54).

Random streams cannot match R's, so the checks are

- deterministic: for the same item parameters, :math:`\\theta` and uniform numbers, the category
  probabilities equal mirt's ``probtrace()`` and the responses equal those obtained in R by
  inverting the cumulative probabilities (``validation/r/v54_compare_simulate_reference.R``);
- distributional: category proportions and the summed-score distribution of a large sample
  agree with the model-implied marginals within Monte Carlo error, and the latent draws follow
  each latent distribution (``Normal`` with 1, 2 and bifactor factors, ``NormalMixture``,
  ``DavidianCurve``, ``EmpiricalHistogram``; two groups);
- recovery: ``fit()`` on simulated data returns the generating parameters within sampling error.
"""

from __future__ import annotations

import json
from functools import cache
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from flexpyirt import (
    BAEM,
    GPCM,
    GRM,
    DavidianCurve,
    EmpiricalHistogram,
    FitResult,
    FourPL,
    Model,
    Nominal,
    Normal,
    NormalMixture,
    Rasch,
    SimulatedData,
    ThreePL,
    TwoPL,
    fit,
    simulate_responses,
    summed_score_distribution,
)
from flexpyirt.errors import SpecificationError
from flexpyirt.items import ItemModel
from flexpyirt.latent import davidian_log_weights, mixture_components
from flexpyirt.simulate import (
    _category_probabilities,
    _source_from_fit,
    _source_from_model,
    responses_from_uniforms,
)
from tests._reference import REFERENCE_DIR, check_reference_meta

TOL_PROBTRACE = 1e-12
Z_MAX = 4.5
"""Largest tolerated |z| of a Monte Carlo comparison (many comparisons per test)."""


@cache
def _raw() -> dict[str, Any]:
    raw = dict(json.loads((REFERENCE_DIR / "v54_simulate.json").read_text()))
    check_reference_meta(raw["meta"])
    return raw


def _item_model(itemtype: str, n_categories: int) -> ItemModel:
    if itemtype == "Rasch":
        return Rasch()
    if itemtype == "2PL":
        return TwoPL()
    if itemtype == "3PL":
        return ThreePL()
    if itemtype == "4PL":
        return FourPL()
    if itemtype == "graded":
        return GRM(n_categories=n_categories)
    if itemtype == "gpcm":
        return GPCM(n_categories=n_categories)
    return Nominal(n_categories=n_categories)


def _reference_model(case: dict[str, Any]) -> tuple[Model, pd.DataFrame]:
    items = case["items"]
    spec = Model.from_items(
        {
            i: _item_model(t, k)
            for i, t, k in zip(items, case["itemtype"], case["n_categories"], strict=True)
        }
    )
    if "factors" in case:
        spec = spec.with_factors(case["factors"])
    bank = pd.DataFrame([{"item": i, **case["params"][i]} for i in items])
    return spec, bank


# ---- deterministic ------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["unidimensional", "two_factor"])
def test_inversion_matches_r(name: str) -> None:
    case = _raw()[name]
    spec, bank = _reference_model(case)
    theta = np.asarray(case["theta"], dtype=float)
    uniforms = np.asarray(case["u"], dtype=float)
    probs = _category_probabilities(_source_from_model(spec, bank), theta)
    expected = np.asarray(case["responses"], dtype=np.int64)
    for j, item in enumerate(case["items"]):
        np.testing.assert_allclose(
            probs[j][0], np.asarray(case["probtrace"][item]), rtol=0, atol=TOL_PROBTRACE
        )
        np.testing.assert_array_equal(
            responses_from_uniforms(probs[j][0], uniforms[:, j]), expected[:, j]
        )


def test_given_theta_uses_the_model() -> None:
    case = _raw()["unidimensional"]
    spec, bank = _reference_model(case)
    theta = np.asarray(case["theta"], dtype=float)[:, 0]
    sim = simulate_responses(spec, parameters=bank, theta=theta, key=jax.random.key(0))
    assert list(sim.responses.columns) == case["items"]
    assert all(dt == np.int64 for dt in sim.responses.dtypes)
    np.testing.assert_array_equal(sim.theta, theta)
    assert sim.group is None
    for j, k in enumerate(case["n_categories"]):
        assert sim.responses.iloc[:, j].between(0, k - 1).all()


def test_reproducible_from_key() -> None:
    case = _raw()["unidimensional"]
    spec, bank = _reference_model(case)
    one = simulate_responses(spec, parameters=bank, n_obs=500, key=jax.random.key(7))
    two = simulate_responses(spec, parameters=bank, n_obs=500, key=jax.random.key(7))
    other = simulate_responses(spec, parameters=bank, n_obs=500, key=jax.random.PRNGKey(8))
    pd.testing.assert_frame_equal(one.responses, two.responses)
    np.testing.assert_array_equal(one.theta, two.theta)
    assert not one.responses.equals(other.responses)


# ---- distributional -----------------------------------------------------------------------------


def _hermite(n: int = 81) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.hermite_e.hermegauss(n)
    return nodes, weights / weights.sum()


def _check_marginals(
    sim: SimulatedData, src_probs: list[np.ndarray], weights: np.ndarray, rows: np.ndarray
) -> None:
    """Category proportions and summed scores of ``rows`` vs the quadrature marginals."""
    n = int(rows.sum())
    for j, p in enumerate(src_probs):
        expected = weights @ p
        counts = np.bincount(sim.responses.iloc[rows, j], minlength=p.shape[1])
        z = (counts / n - expected) / np.sqrt(expected * (1 - expected) / n)
        assert np.max(np.abs(z)) < Z_MAX, (j, z)
    k_max = max(p.shape[1] for p in src_probs)
    padded = np.stack([np.pad(p, ((0, 0), (0, k_max - p.shape[1]))) for p in src_probs], axis=1)
    expected = weights @ np.asarray(summed_score_distribution(padded))
    scores = sim.responses.iloc[rows].sum(axis=1).to_numpy()
    counts = np.bincount(scores, minlength=expected.shape[0])
    ok = expected > 1e-4
    z = (counts[ok] / n - expected[ok]) / np.sqrt(expected[ok] * (1 - expected[ok]) / n)
    assert np.max(np.abs(z)) < Z_MAX, z


MIXED: dict[str, ItemModel] = {
    "r1": Rasch(),
    "t1": TwoPL(),
    "t2": ThreePL(),
    "f1": FourPL(),
    "g1": GRM(n_categories=4),
    "p1": GPCM(n_categories=3),
    "n1": Nominal(n_categories=4),
}
MIXED_BANK = pd.DataFrame(
    [
        {"item": "r1", "d": 0.4},
        {"item": "t1", "a": 1.5, "d": -0.3},
        {"item": "t2", "a": 1.7, "d": 0.1, "g": 0.2},
        {"item": "f1", "a": 1.2, "d": 0.5, "g": 0.1, "u": 0.92},
        {"item": "g1", "a": 1.1, "d1": 1.8, "d2": 0.2, "d3": -1.3},
        {"item": "p1", "a": 0.8, "d1": 0.6, "d2": 0.1},
        {"item": "n1", "a": 1.0, "ak1": 0.5, "ak2": 2.2, "d1": 0.4, "d2": 0.8, "d3": -0.2},
    ]
)


@pytest.mark.slow
def test_marginals_normal_from_model() -> None:
    spec = Model.from_items(MIXED).with_latent(Normal(mean=0.3, var=1.4))
    sim = simulate_responses(spec, parameters=MIXED_BANK, n_obs=60000, key=jax.random.key(1))
    assert sim.theta.mean() == pytest.approx(0.3, abs=Z_MAX * np.sqrt(1.4 / 60000))
    nodes, weights = _hermite()
    probs = _category_probabilities(
        _source_from_model(spec, MIXED_BANK), 0.3 + np.sqrt(1.4) * nodes[:, None]
    )
    _check_marginals(sim, [p[0] for p in probs], weights, np.ones(60000, dtype=bool))


@pytest.mark.slow
def test_marginals_two_groups() -> None:
    spec = Model.from_items(MIXED).with_groups(
        ref=Normal.standard(), focal=Normal(mean=-0.5, var=0.6)
    )
    sim = simulate_responses(
        spec, parameters=MIXED_BANK, n_obs={"ref": 30000, "focal": 40000}, key=jax.random.key(2)
    )
    assert sim.group is not None
    assert (sim.group == "ref").sum() == 30000 and (sim.group == "focal").sum() == 40000
    src = _source_from_model(spec, MIXED_BANK)
    nodes, weights = _hermite()
    for g, (mean, var) in enumerate([(0.0, 1.0), (-0.5, 0.6)]):
        probs = _category_probabilities(src, mean + np.sqrt(var) * nodes[:, None])
        rows = sim.group == ["ref", "focal"][g]
        assert sim.theta[rows].var() == pytest.approx(var, rel=0.05)
        _check_marginals(sim, [p[g] for p in probs], weights, rows)


@pytest.mark.slow
def test_two_factor_and_bifactor_latent_draws() -> None:
    items = [f"q{j}" for j in range(6)]
    bank = pd.DataFrame(
        {"item": items, "a1": [1.0] * 3 + [0.0] * 3, "a2": [0.0] * 3 + [1.0] * 3, "d": 0.0}
    )
    spec = Model.uniform(TwoPL(), items=items).with_factors({"A": items[:3], "B": items[3:]})
    sim = simulate_responses(spec, parameters=bank, n_obs=40000, key=jax.random.key(3))
    assert sim.theta.shape == (40000, 2)
    np.testing.assert_allclose(np.cov(sim.theta.T), np.eye(2), atol=0.03)

    bifactor = Model.uniform(TwoPL(), items=items).with_bifactor(
        specific={"s1": items[:3], "s2": items[3:]}
    )
    bank_bf = pd.DataFrame(
        {
            "item": items,
            "a1": 1.2,
            "a2": [0.8] * 3 + [0.0] * 3,
            "a3": [0.0] * 3 + [0.8] * 3,
            "d": np.linspace(-1, 1, 6),
        }
    )
    sim_bf = simulate_responses(bifactor, parameters=bank_bf, n_obs=40000, key=jax.random.key(4))
    assert sim_bf.theta.shape == (40000, 3)
    np.testing.assert_allclose(np.cov(sim_bf.theta.T), np.eye(3), atol=0.03)
    # Pairwise association: items sharing a specific factor are more correlated.
    corr = sim_bf.responses.corr().to_numpy()
    assert corr[0, 1] > corr[0, 4] + 0.05


def _ks_ok(sample: np.ndarray, cdf_at: Any) -> None:
    stat = stats.kstest(sample, cdf_at).statistic
    assert stat < 1.95 / np.sqrt(sample.shape[0]), stat  # 0.1% critical value


@pytest.mark.slow
def test_mixture_latent_draws() -> None:
    spec = Model.uniform(TwoPL(), items=["a", "b"]).with_latent(
        NormalMixture(2, means=[0.0, 2.0], variances=[1.0, 0.5], weights=[0.7, 0.3])
    )
    bank = pd.DataFrame({"item": ["a", "b"], "a": [1.0, 1.0], "d": [0.0, 0.0]})
    sim = simulate_responses(spec, parameters=bank, n_obs=50000, key=jax.random.key(5))
    src = _source_from_model(spec, bank)
    lat = src.table[src.table["item"] == "__latent__"].set_index("param")["estimate"]
    pi, means, variances = (
        np.asarray(x) for x in mixture_components(jnp.asarray(lat.to_numpy(dtype=float)))
    )
    np.testing.assert_allclose([sim.theta.mean(), sim.theta.var()], [0.0, 1.0], atol=0.03)
    _ks_ok(
        sim.theta,
        lambda t: sum(pi[c] * stats.norm.cdf(t, means[c], np.sqrt(variances[c])) for c in range(2)),
    )


@cache
def _davidian_fit() -> FitResult:
    items = [f"q{j}" for j in range(12)]
    bank = pd.DataFrame({"item": items, "a": 1.5, "d": np.linspace(-2, 2, 12)})
    gen = Model.uniform(TwoPL(), items=items).with_latent(
        NormalMixture(2, means=[0.0, 2.5], variances=[1.0, 0.5], weights=[0.7, 0.3])
    )
    data = simulate_responses(gen, parameters=bank, n_obs=3000, key=jax.random.key(6)).responses
    return fit(Model.uniform(TwoPL(), items=items).with_latent(DavidianCurve(2)), data, se=None)


@pytest.mark.slow
def test_davidian_latent_draws() -> None:
    res = _davidian_fit()
    sim = simulate_responses(res, n_obs=50000, key=jax.random.key(7))
    lat = res.parameters[res.parameters["item"] == "__latent__"]["estimate"].to_numpy()
    grid = np.linspace(-12, 12, 48001)
    w = np.exp(np.asarray(davidian_log_weights(jnp.asarray(grid), jnp.asarray(lat))))
    cdf = np.cumsum(w)
    _ks_ok(sim.theta, lambda t: np.interp(t, grid, cdf))
    np.testing.assert_allclose([sim.theta.mean(), sim.theta.var()], [0.0, 1.0], atol=0.03)


@pytest.mark.slow
def test_empirical_histogram_latent_draws() -> None:
    res = fit(
        Model.uniform(TwoPL(), items=list(_davidian_fit().items["item"])).with_latent(
            EmpiricalHistogram()
        ),
        simulate_responses(_davidian_fit(), n_obs=3000, key=jax.random.key(8)).responses,
        se=None,
    )
    sim = simulate_responses(res, n_obs=50000, key=jax.random.key(9))
    nodes = res._quadrature.nodes()
    assert np.isin(sim.theta, nodes).all()
    weights = res.latent_weights
    assert weights is not None
    expected = weights["weight"].to_numpy()
    counts = np.array([(sim.theta == x).sum() for x in nodes])
    ok = expected > 1e-3
    z = (counts[ok] / 50000 - expected[ok]) / np.sqrt(expected[ok] * (1 - expected[ok]) / 50000)
    assert np.max(np.abs(z)) < Z_MAX


# ---- recovery -----------------------------------------------------------------------------------


@pytest.mark.slow
def test_recovery_mixed_format() -> None:
    spec = Model.from_items(MIXED)
    sim = simulate_responses(spec, parameters=MIXED_BANK, n_obs=20000, key=jax.random.key(10))
    res = fit(spec, sim.responses, method=BAEM(tol=1e-8, max_iter=5000))
    est = res.parameters[res.parameters["free"] & (res.parameters["item"] != "__latent__")]
    truth = MIXED_BANK.set_index("item")
    true = np.array([truth.loc[i, p] for i, p in zip(est["item"], est["param"], strict=True)])
    z = (est["estimate"].to_numpy() - true) / est["se"].to_numpy()
    assert np.max(np.abs(z)) < Z_MAX, z


@pytest.mark.slow
def test_recovery_two_groups_latent() -> None:
    items = [f"q{j}" for j in range(15)]
    bank = pd.DataFrame(
        {"item": items, "a": np.linspace(0.8, 2.0, 15), "d": np.linspace(-1.5, 1.5, 15)}
    )
    gen = Model.uniform(TwoPL(), items=items).with_groups(
        ref=Normal.standard(), focal=Normal(mean=0.5, var=1.3)
    )
    sim = simulate_responses(gen, parameters=bank, n_obs=8000, key=jax.random.key(11))
    spec = (
        Model.uniform(TwoPL(), items=items)
        .with_groups(ref=Normal.standard(), focal=Normal.free())
        .equal_across_groups(items=items)
    )
    res = fit(spec, sim.responses, group=sim.group)
    focal = res.latent.set_index("group").loc["focal"].to_dict()
    for param, true in (("mean", 0.5), ("var", 1.3)):
        z = (focal[param] - true) / focal[f"{param}_se"]
        assert abs(z) < Z_MAX, (param, z)


@pytest.mark.slow
def test_from_fit_matches_from_model() -> None:
    """A FitResult and a Model with the same parameters simulate the same data from one key."""
    items = ["a", "b", "c", "d"]
    bank = pd.DataFrame({"item": items, "a": [1.0, 1.2, 0.8, 1.5], "d": [0.0, 0.5, -0.5, 1.0]})
    spec = Model.uniform(TwoPL(), items=items)
    data = simulate_responses(spec, parameters=bank, n_obs=400, key=jax.random.key(12))
    res = fit(spec, data.responses, se=None)
    fitted_bank = (
        res.parameters[res.parameters["item"] != "__latent__"]
        .pivot(index="item", columns="param", values="estimate")
        .reset_index()
    )
    one = simulate_responses(res, n_obs=300, key=jax.random.key(13))
    two = simulate_responses(spec, parameters=fitted_bank, n_obs=300, key=jax.random.key(13))
    pd.testing.assert_frame_equal(one.responses, two.responses)
    assert _source_from_fit(res).model is res._model


# ---- argument checks ----------------------------------------------------------------------------


def test_argument_errors() -> None:
    spec = Model.from_items(MIXED)
    key = jax.random.key(0)
    with pytest.raises(SpecificationError, match="PRNG key"):
        simulate_responses(spec, parameters=MIXED_BANK, n_obs=10, key=0)  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match="needs parameters"):
        simulate_responses(spec, n_obs=10, key=key)
    with pytest.raises(SpecificationError, match="n_obs is required"):
        simulate_responses(spec, parameters=MIXED_BANK, key=key)
    with pytest.raises(SpecificationError, match="positive integer"):
        simulate_responses(spec, parameters=MIXED_BANK, n_obs=0, key=key)
    with pytest.raises(SpecificationError, match="every item parameter"):
        simulate_responses(
            Model.uniform(TwoPL(), items=["a", "b"]),
            parameters=pd.DataFrame({"item": ["a"], "a": [1.0], "d": [0.0]}),
            n_obs=5,
            key=key,
        )
    with pytest.raises(SpecificationError, match="theta must have shape"):
        simulate_responses(spec, parameters=MIXED_BANK, theta=np.zeros((3, 2)), key=key)
    with pytest.raises(SpecificationError, match="n_obs must be omitted"):
        simulate_responses(spec, parameters=MIXED_BANK, theta=np.zeros(3), n_obs=4, key=key)
    with pytest.raises(SpecificationError, match="has its shape only after fitting"):
        simulate_responses(
            spec.with_latent(DavidianCurve(2)), parameters=MIXED_BANK, n_obs=5, key=key
        )
    groups = spec.with_groups(ref=Normal.standard(), focal=Normal.standard())
    with pytest.raises(SpecificationError, match="pass group="):
        simulate_responses(groups, parameters=MIXED_BANK, theta=np.zeros(3), key=key)
    with pytest.raises(SpecificationError, match="not in the model"):
        simulate_responses(groups, parameters=MIXED_BANK, n_obs={"other": 3}, key=key)
    sim = simulate_responses(
        groups, parameters=MIXED_BANK, theta=np.zeros(3), group=["focal", "ref", "focal"], key=key
    )
    assert list(sim.group) == ["focal", "ref", "focal"]  # type: ignore[arg-type]
    # theta given: a DavidianCurve shape is not needed.
    simulate_responses(
        spec.with_latent(DavidianCurve(2)), parameters=MIXED_BANK, theta=np.zeros(2), key=key
    )
