"""Gauss–Hermite quadrature (DESIGN.md §15.44).

mirt does not use Gauss–Hermite in EM, so the checks are against numerical integration
(``scipy.integrate.quad``) and against fits on very fine equispaced grids.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from flexpyirt import (
    BAEM,
    EmpiricalHistogram,
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
from flexpyirt.errors import SpecificationError


def _sim_2pl(seed: int, n: int, j: int, mean: float, var: float) -> tuple[pd.DataFrame, ...]:
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.8, 1.8, j)
    d = rng.normal(size=j)
    theta = rng.normal(mean, math.sqrt(var), n)
    y = (rng.random((n, j)) < 1.0 / (1.0 + np.exp(-(np.outer(theta, a) + d)))).astype(float)
    return pd.DataFrame(y, columns=[f"i{k}" for k in range(j)]), a, d


def test_nodes_and_base_weights_reproduce_standard_normal_moments() -> None:
    quad = Quadrature.gauss_hermite(21)
    nodes = quad.nodes()
    w = np.exp(quad.base_log_weights() - 0.5 * nodes**2)
    w /= w.sum()
    for k, moment in ((0, 1.0), (1, 0.0), (2, 1.0), (3, 0.0), (4, 3.0), (6, 15.0)):
        assert np.sum(w * nodes**k) == pytest.approx(moment, abs=1e-12)
    grid = quad.grid(2)
    base2 = quad.base_log_weights(2)
    assert grid.shape == (441, 2) and base2.shape == (441,)
    np.testing.assert_allclose(
        base2.reshape(21, 21), quad.base_log_weights()[:, None] + quad.base_log_weights()[None, :]
    )
    assert np.all(Quadrature.equispaced().base_log_weights() == 0.0)


def test_marginal_loglik_matches_numerical_integration() -> None:
    """Reweighted GH nodes integrate a non-standard normal latent (mean 0.3, var 1.4)."""
    df, a, d = _sim_2pl(0, 400, 8, 0.3, 1.4)
    spec = Model.uniform(TwoPL(), items=df.columns).with_latent(Normal(mean=0.3, var=1.4))
    pats = df.drop_duplicates().iloc[:30].reset_index(drop=True)

    def exact(row: np.ndarray) -> float:
        def f(t: float) -> float:
            z = a * t + d
            lp = np.where(row == 1, -np.logaddexp(0.0, -z), -np.logaddexp(0.0, z))
            return float(np.exp(lp.sum()) * stats.norm.pdf(t, 0.3, math.sqrt(1.4)))

        return math.log(integrate.quad(f, -np.inf, np.inf, epsabs=0, epsrel=1e-13, limit=200)[0])

    expected = np.array([exact(r) for r in pats.to_numpy()])
    prep = prepare(spec, pats, group=None, item_names=None, table=None)
    table = prep.table.copy()
    for k, item in enumerate(df.columns):
        table.loc[(table["item"] == item) & (table["param"] == "a"), "value"] = a[k]
        table.loc[(table["item"] == item) & (table["param"] == "d"), "value"] = d[k]
    table["free"] = False
    comp = compile_problem(spec, prep.y, prep.group, table, Quadrature.gauss_hermite(61))
    log_joint = _core._log_joint(comp.problem, jnp.asarray(comp.eta0))
    got = np.asarray(jax.scipy.special.logsumexp(log_joint, axis=1))[: comp.n_patterns]
    np.testing.assert_allclose(got[comp.inverse], expected, rtol=1e-9)


def test_rasch_free_variance_matches_fine_equispaced_fit() -> None:
    rng = np.random.default_rng(1)
    d = rng.normal(size=10)
    theta = rng.normal(0.0, math.sqrt(1.6), 2000)
    y = (rng.random((2000, 10)) < 1.0 / (1.0 + np.exp(-(theta[:, None] + d)))).astype(float)
    df = pd.DataFrame(y, columns=[f"i{k}" for k in range(10)])
    spec = Model.uniform(Rasch(), items=df.columns)
    method = BAEM(tol=1e-10, max_iter=5000)
    ref = fit(spec, df, method=method, quadrature=Quadrature.equispaced(401, lim=(-12, 12)))
    res = fit(spec, df, method=method, quadrature=Quadrature.gauss_hermite(41))
    assert res.loglik == pytest.approx(ref.loglik, abs=1e-7)
    np.testing.assert_allclose(res.items["d"], ref.items["d"], atol=1e-8)
    assert float(res.latent["var"].iloc[0]) == pytest.approx(
        float(ref.latent["var"].iloc[0]), abs=1e-8
    )
    np.testing.assert_allclose(res.items["d_se"], ref.items["d_se"], rtol=1e-7)


def test_two_dimensional_product_rule_matches_fine_grid() -> None:
    rng = np.random.default_rng(2)
    a = np.zeros((8, 2))
    a[:4, 0] = rng.uniform(1.0, 1.5, 4)
    a[4:, 1] = rng.uniform(1.0, 1.5, 4)
    d = rng.normal(size=8)
    theta = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.4], [0.4, 1.0]], 1500)
    y = (rng.random((1500, 8)) < 1.0 / (1.0 + np.exp(-(theta @ a.T + d)))).astype(float)
    df = pd.DataFrame(y, columns=[f"q{k}" for k in range(8)])
    spec = Model.uniform(TwoPL(), items=df.columns).with_factors(
        {"F1": list(df.columns[:4]), "F2": list(df.columns[4:])}
    )
    ref = fit(spec, df, method=BAEM(tol=1e-9), quadrature=Quadrature.equispaced(121, lim=(-8, 8)))
    res = fit(spec, df, method=BAEM(tol=1e-9), quadrature=Quadrature.gauss_hermite(31))
    assert res.loglik == pytest.approx(ref.loglik, abs=1e-5)
    cols = ["a1", "a2", "d"]
    np.testing.assert_allclose(res.items[cols].to_numpy(), ref.items[cols].to_numpy(), atol=1e-5)
    assert float(res.latent["cor_12"].iloc[0]) == pytest.approx(
        float(ref.latent["cor_12"].iloc[0]), abs=1e-6
    )


def test_scoring_json_and_errors() -> None:
    df, _, _ = _sim_2pl(3, 500, 8, 0.0, 1.0)
    spec = Model.uniform(TwoPL(), items=df.columns)
    res = fit(spec, df, quadrature=Quadrature.gauss_hermite(31))
    for method in ("EAP", "MAP", "WLE"):
        scores = res.score(df.iloc[:20], method=method)  # type: ignore[arg-type]
        assert np.isfinite(scores.to_numpy()).all()
    assert not res.score(df.iloc[:20], method="ML")["theta"].isna().any()  # +/-inf allowed
    pv = res.score(df.iloc[:5], method="PV", key=jax.random.key(0), n_draws=3)
    assert np.isfinite(pv.to_numpy()).all()
    loaded = FitResult.from_json(res.to_json())
    assert loaded._quadrature == res._quadrature
    pd.testing.assert_frame_equal(loaded.score(df), res.score(df), check_exact=True)
    with pytest.raises(SpecificationError, match="equispaced"):
        fit(spec.with_latent(EmpiricalHistogram()), df, quadrature=Quadrature.gauss_hermite(21))
    with pytest.raises(SpecificationError, match="3 to 200"):
        Quadrature.gauss_hermite(2)
