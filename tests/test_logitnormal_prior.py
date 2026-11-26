"""``LogitNormalPrior`` (DESIGN §15.48, V45).

R mirt has no logit-normal prior on the natural scale (its ``norm`` prior on ``g`` acts on the
internal logit(g) without the Jacobian, like flexMIRT's ``Guessing : Normal``), so for
``mode_scale="natural"`` V45 is an independent numerical check: the density against SciPy
(normalization and moments of logit X by quadrature), and the MAP estimate against a stationarity
condition written here from the formula, with the likelihood gradient from the mirt-validated
marginal log-likelihood. ``mode_scale="logit"`` is checked the same way against the logit-metric
objective, against the flexMIRT mode computed here by Newton's method, and against R mirt's
``PRIOR = (..., g, norm, mu, s)`` fit (``tests/reference/v45_3pl_logit_norm_map.json``).
"""

from __future__ import annotations

import json
import math
from functools import cache
from pathlib import Path
from typing import Any

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
from flexpyirt._json import _deserialize_prior, _serialize_prior
from flexpyirt.errors import SpecificationError
from flexpyirt.priors import (
    PRIOR_LOGITNORMAL,
    PRIOR_LOGITNORMAL_LOGIT,
    encode_prior,
    eval_log_prior,
)
from flexpyirt.quadrature import Quadrature
from tests._reference import check_reference_meta


def _reference_log_pdf(x: np.ndarray, mean: float, sd: float) -> np.ndarray:
    """log N(logit x; mean, sd) - log x - log(1 - x) (change of variables)."""
    logit = np.log(x) - np.log1p(-x)
    out: np.ndarray = stats.norm.logpdf(logit, mean, sd) - np.log(x) - np.log1p(-x)
    return out


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


# ---- mode_scale="logit" ------------------------------------------------------------------------


def test_logit_mode_scale_drops_the_jacobian() -> None:
    """log pi~(x) = log N(logit x; mean, sd): the logit-metric objective term (DESIGN §15.48)."""
    x = np.array([1e-6, 0.01, 0.1, 0.25, 0.5, 0.8, 0.999])
    prior = LogitNormalPrior(-1.0986, 0.5, mode_scale="logit")
    logit = np.log(x) - np.log1p(-x)
    np.testing.assert_allclose(
        np.asarray(prior.log_pdf(x)), stats.norm.logpdf(logit, -1.0986, 0.5), rtol=1e-13
    )
    natural = np.asarray(LogitNormalPrior(-1.0986, 0.5).log_pdf(x))
    np.testing.assert_allclose(
        natural - np.asarray(prior.log_pdf(x)), -np.log(x) - np.log1p(-x), rtol=1e-12
    )
    out = np.asarray(prior.log_pdf(np.array([-0.1, 0.0, 1.0, 1.5])))
    assert np.all(out == -np.inf)
    assert prior.support == (0.0, 1.0)
    assert prior != LogitNormalPrior(-1.0986, 0.5)
    assert LogitNormalPrior(0.0, 1.0).mode_scale == "natural"


@pytest.mark.parametrize("bad", ["Logit", "unconstrained", None, 1])
def test_invalid_mode_scale(bad: object) -> None:
    with pytest.raises(SpecificationError, match="mode_scale"):
        LogitNormalPrior(0.0, 1.0, mode_scale=bad)  # type: ignore[arg-type]


def test_core_evaluation_logit_mode() -> None:
    prior = LogitNormalPrior(-1.2, 0.7, mode_scale="logit")
    kind, p1, p2 = encode_prior(prior)
    assert (kind, p1, p2) == (PRIOR_LOGITNORMAL_LOGIT, -1.2, 0.7)
    x = jnp.asarray([0.05, 0.2, 0.6])
    got = eval_log_prior(x, jnp.full(3, kind, dtype=jnp.int32), jnp.full(3, p1), jnp.full(3, p2))
    np.testing.assert_allclose(got, prior.log_pdf(x), rtol=1e-15)
    # Mixed with the natural code and padding: inactive slots stay zero and differentiable.
    kinds = jnp.asarray([0, PRIOR_LOGITNORMAL, PRIOR_LOGITNORMAL_LOGIT], dtype=jnp.int32)
    x0 = jnp.asarray([0.0, 0.3, 0.3])

    def total(v: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(eval_log_prior(v, kinds, jnp.full(3, -1.2), jnp.full(3, 0.7)))

    grad = np.asarray(jax.grad(total)(x0))
    assert grad[0] == 0.0
    # d/dx log N(logit x) = -(logit x - mu) / (s^2 x (1 - x)); natural adds -1/x + 1/(1 - x).
    lam = math.log(0.3 / 0.7)
    g_logit = -(lam + 1.2) / (0.49 * 0.3 * 0.7)
    assert grad[2] == pytest.approx(g_logit, rel=1e-13)
    assert grad[1] == pytest.approx(g_logit - 1 / 0.3 + 1 / 0.7, rel=1e-13)


@pytest.mark.parametrize("mode_scale", ["natural", "logit"])
def test_json_round_trip_of_mode_scale(mode_scale: str) -> None:
    prior = LogitNormalPrior(-1.2, 0.7, mode_scale=mode_scale)  # type: ignore[arg-type]
    obj = _serialize_prior(prior)
    assert obj == {
        "type": "LogitNormalPrior",
        "mean_logit": -1.2,
        "sd_logit": 0.7,
        "mode_scale": mode_scale,
    }
    assert _deserialize_prior(obj, field_name="prior") == prior


def test_json_without_mode_scale_reads_as_natural() -> None:
    obj = {"type": "LogitNormalPrior", "mean_logit": -1.2, "sd_logit": 0.7}
    assert _deserialize_prior(obj, field_name="prior") == LogitNormalPrior(-1.2, 0.7)
    with pytest.raises(SpecificationError, match="mode_scale"):
        _deserialize_prior({**obj, "mode_scale": "jacobian"}, field_name="prior")
    with pytest.raises(SpecificationError):
        _deserialize_prior({**obj, "mode_scale": 1}, field_name="prior")


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
def _map_fit(scale: str, mode_scale: str = "natural") -> tuple[Model, pd.DataFrame, FitResult]:
    df = _sim_3pl(1200, 45)
    prior = LogitNormalPrior(MEAN, SD, mode_scale=mode_scale)  # type: ignore[arg-type]
    spec = (
        Model.uniform(ThreePL(), items=df.columns)
        .prior(param="g", dist=prior)
        .prior(param="d", dist=NormalPrior(0.0, 3.0))
    )
    res = fit(
        spec,
        df,
        method=BAEM(tol=1e-10, max_iter=20000, m_step_scale=scale),  # type: ignore[arg-type]
        se="hessian",
    )
    return spec, df, res


def _ids(spec: Model, df: pd.DataFrame) -> tuple[_core.CompiledProblem, np.ndarray, np.ndarray]:
    prep = prepare(spec, df, group=None, item_names=None, table=None)
    comp = compile_problem(spec, prep.y, prep.group, prep.table, Quadrature.equispaced())
    table = prep.table[prep.table["free"]]
    g_ids = table.loc[table["param"] == "g", "free_id"].to_numpy()
    d_ids = table.loc[table["param"] == "d", "free_id"].to_numpy()
    return comp.problem, g_ids, d_ids


@pytest.mark.slow
@pytest.mark.parametrize("mode_scale", ["natural", "logit"])
@pytest.mark.parametrize("scale", ["natural", "unconstrained"])
def test_v45_map_is_stationary_point_of_independent_objective(scale: str, mode_scale: str) -> None:
    r"""d/deta [l(eta) + sum log pi(eta)] = 0 at the MAP, with the prior gradients from the formula.

    .. math::
        \frac{d}{dg}\log\pi(g)=[-\tfrac1g+\tfrac1{1-g}]
            -\frac{\operatorname{logit}g-\mu}{s^2g(1-g)},
        \qquad \frac{d}{dd}\log\pi(d)=-\frac{d}{9},

    the bracket only for ``mode_scale="natural"``.
    """
    spec, df, res = _map_fit(scale, mode_scale)
    assert res.converged
    problem, g_ids, d_ids = _ids(spec, df)
    eta = jnp.asarray(res._eta)
    grad = np.array(jax.grad(lambda e: _core.marginal_loglik(problem, e))(eta))
    for fid in g_ids:
        x = float(res._eta[fid])
        logit = math.log(x / (1 - x))
        grad[fid] += -(logit - MEAN) / (SD**2 * x * (1 - x))
        if mode_scale == "natural":
            grad[fid] += -1 / x + 1 / (1 - x)
    for fid in d_ids:
        grad[fid] += -float(res._eta[fid]) / 9.0
    assert np.max(np.abs(grad)) < 1e-6
    # A strict local maximum of the log posterior.
    hess = np.asarray(jax.hessian(lambda e: _core.log_posterior(problem, e))(eta))
    assert np.max(np.linalg.eigvalsh(hess)) < 0.0


def _flexmirt_mode(spec: Model, df: pd.DataFrame, start: np.ndarray) -> tuple[np.ndarray, Any]:
    """flexMIRT's logit-metric MAP, by Newton's method on lambda = logit g (independent code).

    flexMIRT's ``Guessing : Normal(mean, sd)`` maximizes l + log N(lambda; mean, sd) over
    ``lambda = logit g`` (DESIGN §15.48). Returns the mode on the natural scale and the negative
    log posterior as a function of the logit-scale vector (g slots replaced by lambda).
    """
    problem, g_ids, d_ids = _ids(spec, df)
    g_idx = jnp.asarray(g_ids)
    d_idx = jnp.asarray(d_ids)

    def neg_objective(v: jnp.ndarray) -> jnp.ndarray:
        lam = v[g_idx]
        e = v.at[g_idx].set(jax.nn.sigmoid(lam))
        lp = jnp.sum(_normal_logpdf(lam, MEAN, SD)) + jnp.sum(_normal_logpdf(e[d_idx], 0.0, 3.0))
        return -(_core.marginal_loglik(problem, e) + lp)

    grad_fn = jax.jit(jax.grad(neg_objective))
    hess_fn = jax.jit(jax.hessian(neg_objective))
    v = jnp.asarray(start).at[g_idx].set(jnp.log(start[g_ids] / (1 - start[g_ids])))
    for _ in range(20):
        v = v - jnp.linalg.solve(hess_fn(v), grad_fn(v))
    assert float(jnp.max(jnp.abs(grad_fn(v)))) < 1e-8
    assert np.min(np.linalg.eigvalsh(np.asarray(hess_fn(v)))) > 0.0
    x = np.asarray(v.at[g_idx].set(jax.nn.sigmoid(v[g_idx])))
    return x, (neg_objective, v, g_ids)


def _normal_logpdf(x: jnp.ndarray, mean: float, sd: float) -> jnp.ndarray:
    """Normal log density (jax), written out for the independent objective."""
    return -0.5 * math.log(2 * math.pi) - math.log(sd) - 0.5 * ((x - mean) / sd) ** 2


@pytest.mark.slow
def test_v45_scales_agree_and_jacobian_moves_g_away_from_one_half() -> None:
    """Both M-step scales give one MAP; flexMIRT's logit-scale objective gives g nearer 1/2."""
    spec, df, nat = _map_fit("natural")
    _, _, unc = _map_fit("unconstrained")
    np.testing.assert_allclose(unc._eta, nat._eta, rtol=0, atol=1e-6)

    x_flex, _ = _flexmirt_mode(spec, df, nat._eta)
    _, g_ids, _ = _ids(spec, df)
    g_flex = x_flex[g_ids]
    g_ours = nat._eta[g_ids]
    assert np.all(np.abs(g_ours - 0.5) > np.abs(g_flex - 0.5))
    assert np.max(np.abs(g_ours - g_flex)) > 1e-3


@pytest.mark.slow
@pytest.mark.parametrize("scale", ["natural", "unconstrained"])
def test_v45_logit_mode_scale_reproduces_flexmirt_mode_and_se(scale: str) -> None:
    r"""``mode_scale="logit"``: MAP, log posterior and SEs of flexMIRT's logit-metric objective.

    The SE of :math:`g` is the delta-method SE of the logit-scale estimate,
    :math:`\mathrm{SE}(g)=g(1-g)\,\mathrm{SE}(\lambda)`, from the inverse logit-scale Hessian of
    the independent objective (DESIGN §15.48).
    """
    _, df, nat = _map_fit("natural")
    spec, _, res = _map_fit(scale, "logit")
    assert res.converged
    x_flex, (neg_objective, v_hat, g_ids) = _flexmirt_mode(spec, df, nat._eta)
    np.testing.assert_allclose(res._eta, x_flex, rtol=0, atol=1e-7)
    assert res.log_posterior == pytest.approx(-float(neg_objective(v_hat)), abs=1e-8)
    # flexMIRT's mode differs from the natural-scale one (V45: up to ~0.04 in g).
    assert np.max(np.abs(res._eta[g_ids] - nat._eta[g_ids])) > 1e-3

    cov_logit = np.linalg.inv(np.asarray(jax.hessian(neg_objective)(v_hat)))
    se_ref = np.sqrt(np.diag(cov_logit))
    g = x_flex[g_ids]
    se_ref[g_ids] *= g * (1 - g)
    assert res.vcov is not None
    np.testing.assert_allclose(np.sqrt(np.diag(res.vcov)), se_ref, rtol=1e-6)


# ---- V45: mirt PRIOR = (..., g, norm, mu, s) ------------------------------------------------


REFERENCE = Path(__file__).parent / "reference" / "v45_3pl_logit_norm_map.json"


@cache
def _mirt_case() -> tuple[dict[str, Any], pd.DataFrame, Model]:
    raw = dict(json.loads(REFERENCE.read_text()))
    check_reference_meta(raw["meta"], require_grid_check=True)
    patterns = pd.DataFrame(np.asarray(raw["patterns"], dtype=float), columns=raw["items"])
    counts = np.asarray(raw["counts"], dtype=int)
    data = patterns.loc[patterns.index.repeat(counts)].reset_index(drop=True)
    g_prior, d_prior = raw["priors"]["g"], raw["priors"]["d"]
    assert g_prior["mode_scale"] == "logit"
    spec = (
        Model.uniform(ThreePL(), items=raw["items"])
        .prior(
            param="g",
            dist=LogitNormalPrior(g_prior["mean_logit"], g_prior["sd_logit"], mode_scale="logit"),
        )
        .prior(param="d", dist=NormalPrior(d_prior["mean"], d_prior["sd"]))
    )
    return raw, data, spec


@cache
def _mirt_fit(scale: str) -> FitResult:
    _, data, spec = _mirt_case()
    return fit(
        spec,
        data,
        method=BAEM(tol=1e-9, max_iter=20000, m_step_scale=scale),  # type: ignore[arg-type]
        se="oakes",
    )


@pytest.mark.slow
@pytest.mark.parametrize("scale", ["natural", "unconstrained"])
def test_v45_logit_mode_scale_matches_mirt_norm_prior_on_g(scale: str) -> None:
    """``mode_scale="logit"`` reproduces mirt's ``norm`` prior on its internal logit(g)."""
    raw, _, _ = _mirt_case()
    res = _mirt_fit(scale)
    assert res.converged
    assert res.loglik == pytest.approx(raw["loglik"], rel=1e-9)
    assert res.log_posterior == pytest.approx(raw["log_posterior"], rel=1e-9)
    ref = pd.DataFrame(raw["estimates"]["all"]["params"]).T.loc[raw["items"]]
    ours = res.items.set_index("item").loc[ref.index]
    for p in ("a", "d", "g"):
        np.testing.assert_allclose(ours[p], ref[p].astype(float), rtol=0, atol=1e-6)
        np.testing.assert_allclose(ours[f"{p}_se"], ref[f"{p}_se"].astype(float), rtol=1e-5)
