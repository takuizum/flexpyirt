"""Regression and equivalence tests for structured Oakes and chunked Hessian information.

Keeps the original ``jax.jacfwd(jax.grad(q))`` Oakes implementation and direct ``jax.hessian``
implementation as private reference functions and verifies that :func:`_core.information_oakes`
and :func:`_core.information_hessian` match them to relative 1e-10 across single-group,
multiple-group, GRM, 3PL, 4PL, GPCM, PCM, FIPC, missing-data, MAP/prior, and edge-case models.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from flexpyirt import (
    BAEM,
    GPCM,
    GRM,
    PCM,
    BetaPrior,
    FourPL,
    LogNormalPrior,
    Model,
    Normal,
    NormalPrior,
    Quadrature,
    Rasch,
    ThreePL,
    TwoPL,
    _core,
    fit,
)
from flexpyirt._compile import compile_problem, prepare
from flexpyirt.errors import HeywoodWarning
from tests import test_mirt_m2, test_mirt_m3, test_mirt_m3_gpcm, test_mirt_m3_priors
from tests._reference import load as load_m1


def _reference_information_oakes(prob: _core.CompiledProblem, eta: jax.Array) -> jax.Array:
    """Original Oakes implementation via full ``jax.jacfwd(jax.grad(q))`` (kept for testing)."""

    def q(e: jax.Array, e_prime: jax.Array) -> jax.Array:
        counts, _ = _core.e_step(prob, e_prime)
        return _core.expected_complete_loglik(prob, e, counts) + _core.log_prior(prob, e)

    h_own, h_cross = jax.jacfwd(jax.grad(q, argnums=0), argnums=(0, 1))(eta, eta)
    info: jax.Array = -(h_own + h_cross)
    return info


def _reference_information_hessian(prob: _core.CompiledProblem, eta: jax.Array) -> jax.Array:
    """Original direct-Hessian implementation via ``jax.hessian`` (kept for testing)."""
    info: jax.Array = -jax.hessian(lambda e: _core.log_posterior(prob, e))(eta)
    return info


def _compile_case(
    spec: Model,
    df: pd.DataFrame,
    *,
    group: np.ndarray | None = None,
    quad: Quadrature | None = None,
    table: pd.DataFrame | None = None,
) -> tuple[_core.CompiledProblem, jax.Array]:
    quadrature = Quadrature.equispaced() if quad is None else quad
    prep = prepare(spec, df, group=group, item_names=None, table=table)
    comp = compile_problem(spec, prep.y, prep.group, prep.table, quadrature)
    return comp.problem, jnp.asarray(comp.eta0)


def _assert_matches_reference(prob: _core.CompiledProblem, eta: jax.Array) -> None:
    """Assert fast Oakes and chunked Hessian agree with reference implementations to rel 1e-10."""
    ref_oakes = np.asarray(jax.jit(_reference_information_oakes)(prob, eta))
    got_oakes = np.asarray(jax.jit(_core.information_oakes)(prob, eta))
    ref_hess = np.asarray(jax.jit(_reference_information_hessian)(prob, eta))
    got_hess = np.asarray(jax.jit(_core.information_hessian)(prob, eta))

    scale = max(float(np.max(np.abs(ref_oakes))), 1.0) if ref_oakes.size else 1.0
    np.testing.assert_allclose(got_oakes, ref_oakes, rtol=1e-10, atol=1e-12 * scale)
    np.testing.assert_allclose(got_hess, ref_hess, rtol=1e-10, atol=1e-12 * scale)
    np.testing.assert_allclose(got_oakes, got_hess, rtol=1e-10, atol=1e-12 * scale)


VALIDATION_CASES = [
    "v1_lsat7_2pl",
    "v2_lsat7_rasch",
    "v3_science_grm",
    "v4_twogroup_2pl",
    "v5_fipc_2pl",
    "v9_missing_2pl",
    "v11_3pl",
    "v8_science_gpcm",
    "v8_sim_pcm",
    "v7_2pl_map",
    "v7_2pl_lnorm_map",
    "v7_3pl_map",
]


def _load_fitted_case(name: str) -> tuple[_core.CompiledProblem, jax.Array]:
    """Load a validation case and return its compiled problem and fitted ``eta``."""
    if name in ("v1_lsat7_2pl", "v2_lsat7_rasch"):
        c1 = load_m1(name)
        spec = Model.uniform(TwoPL() if name == "v1_lsat7_2pl" else Rasch(), items=c1.items)
        res = fit(spec, c1.data, method=BAEM(tol=1e-8), se=None)
        prob, _ = _compile_case(spec, c1.data)
        return prob, jnp.asarray(res._eta)

    if name in ("v3_science_grm", "v4_twogroup_2pl", "v5_fipc_2pl", "v9_missing_2pl"):
        c2 = test_mirt_m2.load(name)
        res = fit(
            c2.spec,
            c2.data,
            group=c2.group,
            method=BAEM(tol=1e-8, squarem=True),
            quadrature=c2.quadrature,
            se=None,
        )
        prob, _ = _compile_case(c2.spec, c2.data, group=c2.group, quad=c2.quadrature)
        return prob, jnp.asarray(res._eta)

    if name == "v11_3pl":
        c3 = test_mirt_m3.load(name)
        with warnings.catch_warnings():
            warnings.simplefilter("error", HeywoodWarning)
            res3 = fit(
                c3.spec,
                c3.data,
                method=BAEM(tol=1e-8, squarem=True),
                quadrature=c3.quadrature,
                se=None,
            )
        prob3, _ = _compile_case(c3.spec, c3.data, quad=c3.quadrature)
        return prob3, jnp.asarray(res3._eta)

    if name in ("v8_science_gpcm", "v8_sim_pcm"):
        cg = test_mirt_m3_gpcm.load(name)
        resg = fit(
            cg.spec,
            cg.data,
            method=BAEM(tol=1e-8, squarem=True),
            quadrature=cg.quadrature,
            se=None,
        )
        probg, _ = _compile_case(cg.spec, cg.data, quad=cg.quadrature)
        return probg, jnp.asarray(resg._eta)

    if name in ("v7_2pl_map", "v7_2pl_lnorm_map", "v7_3pl_map"):
        cp = test_mirt_m3_priors.load(name)
        with warnings.catch_warnings():
            warnings.simplefilter("error", HeywoodWarning)
            resp = fit(
                cp.spec,
                cp.data,
                method=BAEM(tol=1e-8, squarem=True),
                quadrature=cp.quadrature,
                se=None,
            )
        probp, _ = _compile_case(cp.spec, cp.data, quad=cp.quadrature)
        return probp, jnp.asarray(resp._eta)

    raise KeyError(name)


@pytest.mark.parametrize("name", VALIDATION_CASES)
def test_oakes_matches_reference_on_validation_cases(name: str) -> None:
    """New Oakes and chunked Hessian agree with reference implementations across V1–V11."""
    prob, eta = _load_fitted_case(name)
    _assert_matches_reference(prob, eta)


def test_oakes_matches_reference_mixed_multigroup_missing_priors_away_from_mle() -> None:
    """Exact match on mixed families + 2 groups + anchors + missing data + all 3 prior types."""
    rng = np.random.default_rng(123)
    n = 400
    df = pd.DataFrame(
        {
            "m_2pl": rng.integers(0, 2, n).astype(float),
            "m_rasch": rng.integers(0, 2, n).astype(float),
            "m_3pl": rng.integers(0, 2, n).astype(float),
            "m_4pl": rng.integers(0, 2, n).astype(float),
            "m_grm": rng.integers(0, 4, n).astype(float),
            "m_gpcm": rng.integers(0, 3, n).astype(float),
            "m_pcm": rng.integers(0, 3, n).astype(float),
        }
    )
    df[rng.random(df.shape) < 0.12] = np.nan
    grp = np.array(["ref"] * (n // 2) + ["focal"] * (n // 2))
    spec = (
        Model.from_items(
            {
                "m_2pl": TwoPL(),
                "m_rasch": Rasch(),
                "m_3pl": ThreePL(),
                "m_4pl": FourPL(),
                "m_grm": GRM(4),
                "m_gpcm": GPCM(3),
                "m_pcm": PCM(3),
            }
        )
        .with_groups(ref=Normal.standard(), focal=Normal.free())
        .equal_across_groups(items=["m_2pl", "m_grm", "m_gpcm"])
        .prior(param="a", dist=LogNormalPrior(0.2, 0.3), items=["m_2pl", "m_3pl", "m_4pl", "m_grm"])
        .prior(param="d", dist=NormalPrior(0.0, 2.0), items=["m_2pl", "m_rasch", "m_3pl"])
        .prior(param="g", dist=BetaPrior(4.0, 16.0), items=["m_3pl", "m_4pl"])
        .prior(param="u", dist=BetaPrior(18.0, 2.0), items=["m_4pl"])
    )
    prob, eta = _compile_case(spec, df, group=grp)
    _assert_matches_reference(prob, eta)


def test_oakes_zero_free_parameters() -> None:
    """All-fixed model (n_free == 0) returns an empty (0, 0) matrix cleanly."""
    c1 = load_m1("v1_lsat7_2pl")
    spec = Model.uniform(TwoPL(), items=c1.items)
    table = spec.parameter_table(c1.data)
    table["free"] = False
    prob, eta = _compile_case(spec, c1.data, table=table)
    assert eta.shape == (0,)
    assert _core.information_oakes(prob, eta).shape == (0, 0)
    assert _core.information_hessian(prob, eta).shape == (0, 0)


def test_oakes_non_power_of_two_patterns_and_high_slopes() -> None:
    """Directly constructed CompiledProblem with non-power-of-two P and peaked posteriors."""
    rng = np.random.default_rng(7)
    n, j = 60, 12
    a = rng.uniform(4.0, 8.0, size=(1, j))
    d = rng.normal(0.0, 1.5, size=(1, j))
    theta = rng.normal(0.0, 1.0, size=(n, 1))
    y = (rng.random((n, j)) < 1.0 / (1.0 + np.exp(-(theta * a + d)))).astype(float)
    df = pd.DataFrame(y, columns=[f"i{k}" for k in range(j)])
    spec = Model.uniform(TwoPL(), items=list(df.columns))
    prob, _ = _compile_case(spec, df)
    # Slice to a non-power-of-two pattern count (17)
    prob_17 = _core.CompiledProblem(
        y=prob.y[:17],
        counts=prob.counts[:17],
        group=prob.group[:17],
        nodes=prob.nodes,
        families=prob.families,
        latent=prob.latent,
        priors=prob.priors,
        node_log_base=prob.node_log_base,
        n_groups=prob.n_groups,
    )
    eta = jnp.asarray(np.column_stack([a.ravel(), d.ravel()]).ravel())
    _assert_matches_reference(prob_17, eta)


def test_oakes_prior_hessian_is_added_once_per_shared_free_id() -> None:
    """Prior curvature is added to Oakes information and counted once for shared free_ids."""
    rng = np.random.default_rng(99)
    n = 300
    df = pd.DataFrame({f"i{k}": rng.integers(0, 2, n).astype(float) for k in range(4)})
    grp = np.array(["g1"] * (n // 2) + ["g2"] * (n // 2))
    base_spec = (
        Model.uniform(TwoPL(), items=list(df.columns))
        .with_groups(g1=Normal.standard(), g2=Normal.free())
        .equal_across_groups(items=["i0", "i1"])
    )
    prior_spec = base_spec.prior(param="a", dist=NormalPrior(1.0, 0.5), items=["i0", "i2"])

    prob_base, eta = _compile_case(base_spec, df, group=grp)
    prob_prior, _ = _compile_case(prior_spec, df, group=grp)
    info_base = np.asarray(jax.jit(_core.information_oakes)(prob_base, eta))
    info_prior = np.asarray(jax.jit(_core.information_oakes)(prob_prior, eta))

    # NormalPrior(1.0, 0.5) has constant negative second derivative 1 / 0.5^2 = 4.0.
    # i0.a is shared across both groups (1 free_id -> +4.0 once); i2.a is free per group (+4.0 x 2).
    diff = info_prior - info_base
    off_diag = diff - np.diag(np.diag(diff))
    np.testing.assert_allclose(off_diag, 0.0, atol=1e-12)
    diag = np.diag(diff)
    assert np.sum(np.isclose(diag, 4.0, rtol=1e-12)) == 3
    assert np.sum(np.isclose(diag, 0.0, atol=1e-12)) == len(diag) - 3
    _assert_matches_reference(prob_prior, eta)


def test_oakes_many_chunks_two_groups_and_padding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise multi-chunk accumulation with several groups and all-padding chunks.

    With the default chunk size every other test fits in one chunk. Shrinking the chunk forces:
    several chunks per group, chunks belonging entirely to the other group (skipped by the
    `lax.cond` on `jnp.any(mask_g)`), and trailing chunks made only of zero-count padding rows.
    The function is called without `jax.jit` so the patched constant is read at trace time.
    """
    rng = np.random.default_rng(11)
    a = np.array([1.2, 0.8, 1.5, 1.0, 0.9, 1.3])
    d = np.array([0.5, -0.4, 0.0, 1.0, -0.8, 0.3])

    def sim(n: int, mu: float) -> np.ndarray:
        theta = rng.normal(mu, 1.0, size=(n, 1))
        y: np.ndarray = (rng.random((n, a.size)) < 1.0 / (1.0 + np.exp(-(theta * a + d)))).astype(
            float
        )
        return y

    df = pd.DataFrame(
        np.vstack([sim(500, 0.0), sim(500, 0.6)]), columns=[f"i{j}" for j in range(6)]
    )
    df = df.mask(rng.random(df.shape) < 0.1)  # some missing responses
    group = np.repeat(["ref", "focal"], 500)
    spec = (
        Model.uniform(TwoPL(), items=df.columns)
        .with_groups(ref=Normal.standard(), focal=Normal.free())
        .equal_across_groups(items=["i0", "i1", "i2"])
    )
    prob, eta = _compile_case(spec, df, group=group)
    n_rows = int(prob.y.shape[0])
    n_real = int(np.count_nonzero(np.asarray(prob.counts)))
    chunk = 8
    assert n_rows // chunk >= 4, "need several chunks"
    assert n_rows - n_real >= chunk, "need at least one all-padding chunk"
    eta = eta + 0.05  # away from the starting point; Oakes' identity holds at any eta

    monkeypatch.setattr(_core, "_OAKES_MAX_CHUNK", chunk)
    got = np.asarray(_core.information_oakes(prob, eta))
    ref = np.asarray(jax.jit(_reference_information_oakes)(prob, eta))
    scale = max(float(np.max(np.abs(ref))), 1.0)
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12 * scale)
