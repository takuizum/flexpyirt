"""Fixed-node quasi-Monte Carlo quadrature (DESIGN.md §28, #164).

mirt's ``QMCEM`` moves its nodes every cycle, so it agrees with these fits only up to QMC error
and is not used as a tight reference here. The checks are exact identities of the node
construction and reweighting, a 3-D fit against a fine Gauss–Hermite product grid, and a 4-D fit
with independent factors against separate unidimensional fits (the exact likelihood factorizes).
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from flexpyirt import BAEM, EmpiricalHistogram, FitResult, Model, Quadrature, TwoPL, fit
from flexpyirt.errors import SpecificationError
from flexpyirt.latent import (
    correlation_matrix_is_positive_definite,
    latent_param_names,
    mvn_log_weights,
)


def _simple_structure(
    seed: int, n_factors: int, per_factor: int, n_persons: int, rho: float
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    rng = np.random.default_rng(seed)
    n_items = n_factors * per_factor
    a = np.zeros((n_items, n_factors))
    for m in range(n_factors):
        a[m * per_factor : (m + 1) * per_factor, m] = rng.uniform(1.0, 2.0, per_factor)
    d = rng.normal(size=n_items)
    sigma = np.full((n_factors, n_factors), rho)
    np.fill_diagonal(sigma, 1.0)
    theta = rng.multivariate_normal(np.zeros(n_factors), sigma, n_persons)
    y = (rng.random((n_persons, n_items)) < 1.0 / (1.0 + np.exp(-(theta @ a.T + d)))).astype(float)
    df = pd.DataFrame(y, columns=[f"q{i}" for i in range(n_items)])
    blocks = {
        f"F{m + 1}": [f"q{i}" for i in range(m * per_factor, (m + 1) * per_factor)]
        for m in range(n_factors)
    }
    return df, blocks


def test_halton_nodes_are_the_radical_inverses() -> None:
    quad = Quadrature.qmc(64)
    grid = quad.grid(3)
    assert grid.shape == (64, 3)
    u = stats.norm.cdf(grid)
    # q = 1, 2, 3 in bases 2, 3, 5 (Halton, 1960).
    np.testing.assert_allclose(
        u[:3], [[1 / 2, 1 / 3, 1 / 5], [1 / 4, 2 / 3, 2 / 5], [3 / 4, 1 / 9, 3 / 5]], atol=1e-14
    )
    assert np.all((u > 0.0) & (u < 1.0))
    np.testing.assert_allclose(
        quad.base_log_weights(3), -np.log(64) + 0.5 * np.sum(grid**2, axis=1), atol=1e-14
    )
    with pytest.raises(SpecificationError, match="per-dimension axis"):
        quad.nodes()


def test_standard_nodes_integrate_normal_moments_to_qmc_accuracy() -> None:
    grid = Quadrature.qmc(5000).grid(4)
    # Halton points from q = 1 are slightly biased low (about 2e-3 here); a random sample of
    # this size would have a standard error of 1.4e-2.
    assert np.abs(grid.mean(axis=0)).max() < 5e-3
    np.testing.assert_allclose(np.cov(grid.T, bias=True), np.eye(4), atol=1e-2)


def test_proposal_equal_to_latent_gives_equal_weights() -> None:
    mean = np.array([0.2, -0.1, 0.3, 0.0])
    sigma = np.array(
        [[1.0, 0.5, 0.3, 0.2], [0.5, 1.2, 0.4, 0.1], [0.3, 0.4, 0.9, 0.3], [0.2, 0.1, 0.3, 1.1]]
    )
    quad = Quadrature.qmc(1000, mean=mean, cov=sigma)
    grid = quad.grid(4)
    np.testing.assert_allclose(
        grid, mean + Quadrature.qmc(1000).grid(4) @ np.linalg.cholesky(sigma).T, atol=1e-13
    )
    sd = np.sqrt(np.diag(sigma))
    corr = sigma / np.outer(sd, sd)
    params = np.concatenate([mean, np.diag(sigma), corr[np.triu_indices(4, 1)]])
    logw = np.asarray(mvn_log_weights(jnp.asarray(grid), jnp.asarray(params)))
    # base + log phi(theta; mean, sigma) is constant: the reweighting is exactly 1/Q.
    weights = np.asarray(jax.nn.softmax(jnp.asarray(quad.base_log_weights(4) + logw)))
    np.testing.assert_allclose(weights, np.full(1000, 1e-3), rtol=1e-10)
    assert quad.with_proposal(mean, sigma) == quad


def test_general_dimension_mvn_log_weights() -> None:
    rng = np.random.default_rng(0)
    nodes = rng.normal(size=(50, 5))
    mean = rng.normal(scale=0.2, size=5)
    variances = rng.uniform(0.6, 1.5, size=5)
    raw = rng.normal(size=(5, 5))
    cov = raw @ raw.T + 5 * np.eye(5)
    corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
    params = np.concatenate([mean, variances, corr[np.triu_indices(5, 1)]])
    sigma = corr * np.sqrt(np.outer(variances, variances))
    expected = stats.multivariate_normal(mean, sigma).logpdf(nodes)
    expected -= np.log(np.sum(np.exp(expected - expected.max()))) + expected.max()
    got = np.asarray(mvn_log_weights(jnp.asarray(nodes), jnp.asarray(params)))
    np.testing.assert_allclose(got, expected, atol=1e-12)
    grad = jax.grad(lambda p: jnp.sum(mvn_log_weights(jnp.asarray(nodes), p)[:5]))(
        jnp.asarray(params)
    )
    assert np.all(np.isfinite(np.asarray(grad)))
    bad = params.copy()
    bad[10:] = 0.9
    bad[10] = -0.9  # cor_12 = -0.9 with every other correlation 0.9: not positive definite
    assert not correlation_matrix_is_positive_definite(bad[10:].tolist(), 5)
    assert np.all(np.isnan(np.asarray(mvn_log_weights(jnp.asarray(nodes), jnp.asarray(bad)))))
    assert correlation_matrix_is_positive_definite(params[10:].tolist(), 5)
    assert len(latent_param_names(5)) == 5 + 5 + 10


def test_specification_errors() -> None:
    with pytest.raises(SpecificationError, match="64 to 200000"):
        Quadrature.qmc(10)
    with pytest.raises(SpecificationError, match="both"):
        Quadrature.qmc(100, mean=[0.0, 0.0])
    with pytest.raises(SpecificationError, match="positive definite"):
        Quadrature.qmc(100, mean=[0.0, 0.0], cov=[[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(SpecificationError, match="only used by qmc"):
        Quadrature(n_points=21, proposal_mean=(0.0,), proposal_cov=((1.0,),))
    with pytest.raises(SpecificationError, match=r"2 dimensions, but the model has 3"):
        Quadrature.qmc(100, mean=[0.0, 0.0], cov=np.eye(2)).grid(3)
    with pytest.raises(SpecificationError, match=r"Quadrature\.qmc"):
        Quadrature.default(4)
    df, blocks = _simple_structure(1, 2, 3, 200, 0.3)
    one = Model.uniform(TwoPL(), items=df.columns)
    with pytest.raises(SpecificationError, match="multidimensional"):
        fit(one, df, quadrature=Quadrature.qmc(100))
    with pytest.raises(SpecificationError, match="multidimensional"):
        fit(one.with_latent(EmpiricalHistogram()), df, quadrature=Quadrature.qmc(100))
    bifactor = Model.uniform(TwoPL(), items=df.columns).with_bifactor(general="G", specific=blocks)
    with pytest.raises(SpecificationError, match="bifactor"):
        fit(bifactor, df, quadrature=Quadrature.qmc(100))
    with pytest.raises(SpecificationError, match="2 to 5 factors"):
        Model.uniform(TwoPL(), items=[f"i{k}" for k in range(6)]).with_factors(
            {f"F{m}": [f"i{m}"] for m in range(6)}
        )


@pytest.mark.slow
def test_two_factor_fit_scores_and_json() -> None:
    df, blocks = _simple_structure(2, 2, 3, 300, 0.3)
    spec = Model.uniform(TwoPL(), items=df.columns).with_factors(blocks)
    res = fit(spec, df, method=BAEM(tol=1e-4), quadrature=Quadrature.qmc(256))
    assert res.converged
    quad = res._quadrature
    assert quad.kind == "qmc" and quad.proposal_mean is not None
    # Second stage: nodes moved to the first stage's latent N(0, R).
    cor = float(res.latent["cor_12"].iloc[0])
    assert quad.proposal_cov is not None
    assert abs(quad.proposal_cov[0][1] - cor) < 0.05
    assert "256 quasi-Monte Carlo (Halton) nodes" in res.summary()
    for method in ("EAP", "MAP", "WLE"):
        assert np.isfinite(res.score(df.iloc[:5], method=method).to_numpy()).all()
    loaded = FitResult.from_json(res.to_json())
    assert loaded._quadrature == quad
    pd.testing.assert_frame_equal(loaded.score(df.iloc[:20]), res.score(df.iloc[:20]))
    stored = json.loads(res.to_json())["quadrature"]
    assert stored["type"] == "qmc" and stored["sequence"] == "halton" and len(stored["cov"]) == 2


@pytest.mark.slow
def test_three_factor_fit_matches_fine_gauss_hermite_product_grid() -> None:
    df, blocks = _simple_structure(5, 3, 5, 2000, 0.5)
    spec = Model.uniform(TwoPL(), items=df.columns).with_factors(blocks)
    ref = fit(spec, df, method=BAEM(tol=1e-8), quadrature=Quadrature.gauss_hermite(21))
    res = fit(spec, df, method=BAEM(tol=1e-8), quadrature=Quadrature.qmc(5000))
    cols = ["a1", "a2", "a3", "d"]
    # Measured 5.1e-3 (parameters) and 1.2e-3 (correlations); the single-stage fit on
    # N(0, I) nodes is off by 5e-2 because rho = 0.5 makes the reweighting variance infinite.
    np.testing.assert_allclose(res.items[cols].to_numpy(), ref.items[cols].to_numpy(), atol=1e-2)
    for c in ("cor_12", "cor_13", "cor_23"):
        assert float(res.latent[c].iloc[0]) == pytest.approx(float(ref.latent[c].iloc[0]), abs=3e-3)
    assert res.loglik == pytest.approx(ref.loglik, abs=0.2)


@pytest.mark.slow
def test_four_independent_factors_match_separate_unidimensional_fits() -> None:
    df, blocks = _simple_structure(11, 4, 5, 2000, 0.0)
    spec = Model.uniform(TwoPL(), items=df.columns).with_factors(blocks, correlated=False)
    res = fit(spec, df, method=BAEM(tol=1e-8), quadrature=Quadrature.qmc(5000))
    items = res.items.set_index("item") if "item" in res.items.columns else res.items
    loglik = 0.0
    for m, cols in enumerate(blocks.values(), start=1):
        ref = fit(
            Model.uniform(TwoPL(), items=cols),
            df[cols],
            method=BAEM(tol=1e-10),
            quadrature=Quadrature.gauss_hermite(61),
        )
        loglik += ref.loglik
        ref_items = ref.items.set_index("item") if "item" in ref.items.columns else ref.items
        np.testing.assert_allclose(
            items.loc[cols, f"a{m}"].to_numpy(), ref_items.loc[cols, "a"].to_numpy(), atol=1.5e-2
        )
        np.testing.assert_allclose(
            items.loc[cols, "d"].to_numpy(), ref_items.loc[cols, "d"].to_numpy(), atol=1e-2
        )
    assert res.loglik == pytest.approx(loglik, abs=0.5)
    with pytest.raises(SpecificationError, match="up to 3 factors"):
        res.score(df.iloc[:2], method="PV", key=jax.random.key(0))
