"""``LogitNormalPrior`` (DESIGN §15.48, V45).

R mirt has no logit-normal prior on the natural scale (its ``norm`` prior on ``g`` acts on the
internal logit(g) without the Jacobian, like flexMIRT's ``Guessing : Normal``), so V45 is an
independent numerical check: the density against SciPy (normalization and moments of
logit X by quadrature), and the MAP estimate against a stationarity condition written here from
the formula, with the likelihood gradient from the mirt-validated marginal log-likelihood.
"""

from __future__ import annotations

import math
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from flexpyirt import (
    BAEM,
    FitResult,
    LogitNormalPrior,
    Model,
    NormalPrior,
    ThreePL,
    TwoPL,
    _core,
    fit,
)
from flexpyirt._compile import _prior_center, compile_problem, prepare
from flexpyirt.errors import SpecificationError
from flexpyirt.priors import PRIOR_LOGITNORMAL, encode_prior, eval_log_prior
from flexpyirt.quadrature import Quadrature


def _reference_log_pdf(x: np.ndarray, mean: float, sd: float) -> np.ndarray:
    """log N(logit x; mean, sd) - log x - log(1 - x) (change of variables)."""
    logit = np.log(x) - np.log1p(-x)
    return stats.norm.logpdf(logit, mean, sd) - np.log(x) - np.log1p(-x)


# ---- density ----------------------------------------------------------------------------------


@pytest.mark.parametrize(("mean", "sd"), [(-1.0986, 0.5), (0.0, 1.0), (1.5, 2.5), (-4.0, 0.2)])
def test_log_pdf_matches_change_of_variables(mean: float, sd: float) -> None:
    x = np.array([1e-6, 0.01, 0.1, 0.25, 0.5, 0.8, 0.999])
    got = np.asarray(LogitNormalPrior(mean, sd).log_pdf(x))
    np.testing.assert_allclose(got, _reference_log_pdf(x, mean, sd), rtol=1e-13, atol=1e-12)


@pytest.mark.parametrize(("mean", "sd"), [(-1.0986, 0.5), (0.5, 1.7)])
def test_density_integrates_to_one_with_logit_moments(mean: float, sd: float) -> None:
    prior = LogitNormalPrior(mean, sd)

    def pdf(x: float) -> float:
        return math.exp(float(prior.log_pdf(x)))

    total, _ = integrate.quad(pdf, 0.0, 1.0, epsabs=1e-13, epsrel=1e-13, limit=200)
    m1, _ = integrate.quad(
        lambda x: math.log(x / (1 - x)) * pdf(x), 0.0, 1.0, epsabs=1e-13, limit=200
    )
    m2, _ = integrate.quad(
        lambda x: (math.log(x / (1 - x)) - mean) ** 2 * pdf(x), 0.0, 1.0, epsabs=1e-13, limit=200
    )
    assert total == pytest.approx(1.0, abs=1e-10)
    assert m1 == pytest.approx(mean, abs=1e-9)
    assert m2 == pytest.approx(sd**2, rel=1e-9)


def test_outside_support_is_minus_inf() -> None:
    out = np.asarray(LogitNormalPrior(0.0, 1.0).log_pdf(np.array([-0.1, 0.0, 1.0, 1.5])))
    assert np.all(out == -np.inf)
    assert LogitNormalPrior(0.0, 1.0).support == (0.0, 1.0)


@pytest.mark.parametrize(
    ("mean", "sd"), [(math.inf, 1.0), (0.0, 0.0), (0.0, -1.0), (0.0, math.nan), (True, 1.0)]
)
def test_invalid_hyperparameters(mean: object, sd: object) -> None:
    with pytest.raises(SpecificationError):
        LogitNormalPrior(mean, sd)  # type: ignore[arg-type]


def test_core_evaluation_matches_and_is_branch_safe() -> None:
    kind, p1, p2 = encode_prior(LogitNormalPrior(-1.2, 0.7))
    assert kind == PRIOR_LOGITNORMAL
    x = jnp.asarray([0.05, 0.2, 0.6])
    k = jnp.full(3, kind, dtype=jnp.int32)
    got = eval_log_prior(x, k, jnp.full(3, p1), jnp.full(3, p2))
    np.testing.assert_allclose(got, LogitNormalPrior(-1.2, 0.7).log_pdf(x), rtol=1e-15)
    # Inactive slots (other kinds, padding) at x = 0 contribute nothing and stay differentiable.
    kinds = jnp.asarray([0, 1, PRIOR_LOGITNORMAL], dtype=jnp.int32)
    x0 = jnp.asarray([0.0, 0.0, 0.3])

    def total(v: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(eval_log_prior(v, kinds, jnp.asarray([0.0, 0.0, -1.2]), jnp.ones(3)))

    grad = np.asarray(jax.grad(total)(x0))
    assert np.all(np.isfinite(grad))
    assert grad[0] == 0.0


# ---- attachment -------------------------------------------------------------------------------


def _sim_3pl(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = np.array([1.3, 1.1, 1.6, 0.9, 1.4, 1.2, 1.0, 1.5])
    d = np.array([-0.6, 0.2, -1.1, 0.5, 0.0, -0.3, 0.8, -0.9])
    g = np.array([0.2, 0.25, 0.15, 0.22, 0.18, 0.2, 0.24, 0.16])
    theta = rng.normal(size=(n, 1))
    p = g + (1 - g) / (1 + np.exp(-(theta * a + d)))
    return pd.DataFrame((rng.random(p.shape) < p).astype(int), columns=[f"q{j}" for j in range(8)])


def test_prior_support_becomes_the_m_step_box() -> None:
    """On a slope the (0, 1) support is intersected with the box; the table keeps the domain."""
    df = _sim_3pl(300, 1)
    spec = Model.uniform(TwoPL(), items=df.columns).prior(
        param="a", dist=LogitNormalPrior(0.4, 0.5), items=["q0"]
    )
    prep = prepare(spec, df, group=None, item_names=None, table=None)
    row = prep.table[(prep.table["item"] == "q0") & (prep.table["param"] == "a")].iloc[0]
    assert (row["lower"], row["upper"]) == (-math.inf, math.inf)
    assert 0.0 < row["value"] < 1.0
    comp = compile_problem(spec, prep.y, prep.group, prep.table, Quadrature.equispaced())
    lower, upper = _core._eta_bounds(comp.problem, jnp.asarray(comp.eta0))
    fid = int(row["free_id"])
    assert (float(lower[fid]), float(upper[fid])) == (0.0, 1.0)
    assert _prior_center(LogitNormalPrior(0.4, 0.5)) == pytest.approx(1 / (1 + math.exp(-0.4)))


# ---- V45: MAP stationarity --------------------------------------------------------------------


MEAN, SD = -1.3863, 0.6  # logit(1/5)


@cache
def _map_fit(scale: str) -> tuple[Model, pd.DataFrame, FitResult]:
    df = _sim_3pl(1200, 45)
    spec = (
        Model.uniform(ThreePL(), items=df.columns)
        .prior(param="g", dist=LogitNormalPrior(MEAN, SD))
        .prior(param="d", dist=NormalPrior(0.0, 3.0))
    )
    res = fit(
        spec,
        df,
        method=BAEM(tol=1e-10, max_iter=20000, m_step_scale=scale),  # type: ignore[arg-type]
        se=None,
    )
    return spec, df, res


@pytest.mark.slow
@pytest.mark.parametrize("scale", ["natural", "unconstrained"])
def test_v45_map_is_stationary_point_of_independent_objective(scale: str) -> None:
    r"""d/deta [l(eta) + sum log pi(eta)] = 0 at the MAP, with the prior gradients from the formula.

    .. math::
        \frac{d}{dg}\log\pi(g)=-\frac1g+\frac1{1-g}-\frac{\operatorname{logit}g-\mu}{s^2g(1-g)},
        \qquad \frac{d}{dd}\log\pi(d)=-\frac{d}{9}.
    """
    spec, df, res = _map_fit(scale)
    assert res.converged
    prep = prepare(spec, df, group=None, item_names=None, table=None)
    comp = compile_problem(spec, prep.y, prep.group, prep.table, Quadrature.equispaced())
    eta = jnp.asarray(res._eta)
    grad = np.array(jax.grad(lambda e: _core.marginal_loglik(comp.problem, e))(eta))
    table = prep.table[prep.table["free"]]
    for fid, param in zip(table["free_id"], table["param"], strict=True):
        x = float(res._eta[fid])
        if param == "g":
            logit = math.log(x / (1 - x))
            grad[fid] += -1 / x + 1 / (1 - x) - (logit - MEAN) / (SD**2 * x * (1 - x))
        elif param == "d":
            grad[fid] += -x / 9.0
    assert np.max(np.abs(grad)) < 1e-6
    # A strict local maximum of the log posterior.
    hess = np.asarray(jax.hessian(lambda e: _core.log_posterior(comp.problem, e))(eta))
    assert np.max(np.linalg.eigvalsh(hess)) < 0.0


@pytest.mark.slow
def test_v45_scales_agree_and_jacobian_moves_g_away_from_one_half() -> None:
    """Both M-step scales give one MAP; flexMIRT's logit-scale objective gives g nearer 1/2.

    flexMIRT's ``Guessing : Normal(mean, sd)`` maximizes l + log N(logit g; mean, sd), i.e. the
    same kernel without the -log g - log(1 - g) term (DESIGN §15.48). That mode is computed here
    by Newton's method on the same likelihood.
    """
    spec, df, nat = _map_fit("natural")
    _, _, unc = _map_fit("unconstrained")
    np.testing.assert_allclose(unc._eta, nat._eta, rtol=0, atol=1e-6)

    prep = prepare(spec, df, group=None, item_names=None, table=None)
    comp = compile_problem(spec, prep.y, prep.group, prep.table, Quadrature.equispaced())
    table = prep.table[prep.table["free"]]
    g_ids = jnp.asarray(table.loc[table["param"] == "g", "free_id"].to_numpy())
    d_ids = jnp.asarray(table.loc[table["param"] == "d", "free_id"].to_numpy())

    def neg_flexmirt_objective(e: jnp.ndarray) -> jnp.ndarray:
        g = e[g_ids]
        logit = jnp.log(g) - jnp.log1p(-g)
        lp = jnp.sum(-0.5 * ((logit - MEAN) / SD) ** 2) + jnp.sum(-0.5 * (e[d_ids] / 3.0) ** 2)
        return -(_core.marginal_loglik(comp.problem, e) + lp)

    # Newton from our MAP (the two modes are close): exact gradient and Hessian by autodiff.
    grad_fn = jax.jit(jax.grad(neg_flexmirt_objective))
    hess_fn = jax.jit(jax.hessian(neg_flexmirt_objective))
    x = jnp.asarray(nat._eta)
    for _ in range(20):
        x = x - jnp.linalg.solve(hess_fn(x), grad_fn(x))
    assert float(jnp.max(jnp.abs(grad_fn(x)))) < 1e-8
    assert np.min(np.linalg.eigvalsh(np.asarray(hess_fn(x)))) > 0.0
    x_flex = np.asarray(x)
    g_flex = x_flex[np.asarray(g_ids)]
    g_ours = nat._eta[np.asarray(g_ids)]
    assert np.all(np.abs(g_ours - 0.5) > np.abs(g_flex - 0.5))
    assert np.max(np.abs(g_ours - g_flex)) > 1e-3
