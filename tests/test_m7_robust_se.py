"""Unit tests for the cross-product and sandwich standard errors (DESIGN.md §17.5).

The agreement with R mirt is in ``tests/test_mirt_m7_robust_se.py`` (VALIDATION.md V35).
"""

from __future__ import annotations

import sys
import warnings
from collections.abc import Callable
from functools import cache
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from flexpyirt import (
    BAEM,
    GPCM,
    GRM,
    BetaPrior,
    DavidianCurve,
    EmpiricalHistogram,
    FitResult,
    LogNormalPrior,
    Model,
    Nominal,
    Normal,
    NormalMixture,
    NormalPrior,
    Quadrature,
    ThreePL,
    TwoPL,
    _core,
    _robust_se,
    fit,
    link,
)
from flexpyirt._compile import compile_problem, prepare
from flexpyirt.errors import SpecificationError

METHOD = BAEM(tol=1e-7, max_iter=5000)
SE_METHODS: tuple[Literal["oakes", "crossprod", "sandwich"], ...] = (
    "oakes",
    "crossprod",
    "sandwich",
)


def _simulate_2pl(
    seed: int, n: int, n_items: int, *, guess: float = 0.0, mean: float = 0.0, sd: float = 1.0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    theta = mean + sd * rng.normal(size=n)
    slope = rng.lognormal(0.0, 0.25, n_items)
    intercept = rng.normal(0.0, 0.8, n_items)
    prob = guess + (1.0 - guess) / (1.0 + np.exp(-(theta[:, None] * slope + intercept)))
    data = (rng.random((n, n_items)) < prob).astype(float)
    return pd.DataFrame(data, columns=[f"q{j + 1:02d}" for j in range(n_items)])


def _simulate_ordinal(seed: int, n: int, n_items: int, n_categories: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=n)
    cuts = np.sort(rng.normal(0.0, 1.0, (n_items, n_categories - 1)), axis=1)
    latent = theta[:, None] * rng.lognormal(0.0, 0.2, n_items) + rng.logistic(size=(n, n_items))
    data = (latent[:, :, None] > cuts[None]).sum(axis=2).astype(float)
    return pd.DataFrame(data, columns=[f"p{j + 1}" for j in range(n_items)])


def _problem(
    spec: Model,
    data: pd.DataFrame,
    *,
    group: np.ndarray | None = None,
    quadrature: Quadrature | None = None,
) -> tuple[_core.CompiledProblem, jax.Array]:
    """Compile ``spec`` and return the problem with a perturbed starting vector."""
    quadrature = Quadrature.default(spec.n_factors) if quadrature is None else quadrature
    prep = prepare(spec, data, group=group, item_names=None, table=None)
    comp = compile_problem(spec, prep.y, prep.group, prep.table, quadrature)
    return comp.problem, jnp.asarray(comp.eta0)


def _dense_scores(prob: _core.CompiledProblem, eta: jax.Array) -> np.ndarray:
    """``(P, n_free)`` scores by forward-mode autodiff of all pattern log-likelihoods at once."""

    def pattern_logliks(p: _core.CompiledProblem, e: jax.Array) -> jax.Array:
        return jax.scipy.special.logsumexp(_core._log_joint(p, e), axis=1)

    return np.asarray(jax.jit(jax.jacfwd(pattern_logliks, argnums=1))(prob, eta))


def _jit(function: Callable[..., jax.Array]) -> Callable[..., jax.Array]:
    """Jit through a fresh wrapper so that a patched chunk budget is traced again."""
    return jax.jit(lambda prob, eta: function(prob, eta))


@cache
def _named_problem(name: str) -> tuple[_core.CompiledProblem, jax.Array]:
    if name == "2pl_missing":
        two_pl = _simulate_2pl(1, 400, 6)
        two_pl.iloc[::7, 2] = np.nan
        return _problem(Model.uniform(TwoPL(), items=two_pl.columns), two_pl)
    if name == "mixed_priors":
        mixed = pd.concat(
            [
                _simulate_2pl(2, 300, 3, guess=0.2),
                _simulate_ordinal(3, 300, 2, 4),
                _simulate_ordinal(4, 300, 2, 3).rename(columns={"p1": "g1", "p2": "g2"}),
                _simulate_ordinal(5, 300, 1, 3).rename(columns={"p1": "n1"}),
            ],
            axis=1,
        )
        binary = ["q01", "q02", "q03"]
        spec = (
            Model.from_items(
                {
                    **{c: ThreePL() for c in binary},
                    "p1": GRM(4),
                    "p2": GRM(4),
                    "g1": GPCM(3),
                    "g2": GPCM(3),
                    "n1": Nominal(3),
                }
            )
            .prior(param="a", dist=LogNormalPrior(0.0, 0.5), items=binary)
            .prior(param="g", dist=BetaPrior(5, 17), items=binary)
            .prior(param="d", dist=NormalPrior(0.0, 2.0), items=binary)
        )
        return _problem(spec, mixed)
    if name == "2pl_priors":
        data = _simulate_2pl(9, 300, 5)
        spec = (
            Model.uniform(TwoPL(), items=data.columns)
            .prior(param="a", dist=LogNormalPrior(0.0, 0.5), items=["q01", "q02", "q03"])
            .prior(param="d", dist=NormalPrior(0.0, 2.0), items=["q01", "q02"])
        )
        return _problem(spec, data)
    if name == "two_group":
        grouped = pd.concat([_simulate_2pl(6, 250, 6), _simulate_2pl(6, 350, 6, mean=0.4, sd=1.2)])
        labels = np.repeat(["ref", "focal"], [250, 350])
        spec = (
            Model.uniform(TwoPL(), items=grouped.columns)
            .with_groups(ref=Normal.standard(), focal=Normal.free())
            .equal_across_groups(items=["q01", "q02", "q03"])
            .equal_across_items(items=["q05", "q06"], params=["a"])
        )
        return _problem(spec, grouped, group=labels)
    if name == "two_factor":
        two_dim = _simulate_2pl(7, 300, 6)
        spec = Model.uniform(TwoPL(), items=two_dim.columns).with_factors(
            {"F1": ["q01", "q02", "q03", "q04"], "F2": ["q04", "q05", "q06"]}
        )
        return _problem(spec, two_dim, quadrature=Quadrature.equispaced(11))
    assert name == "fipc_eh"
    fipc = _simulate_2pl(8, 300, 6, mean=0.3)
    bank = pd.DataFrame({"item": ["q01", "q02"], "a": [1.0, 1.2], "d": [0.0, 0.5]})
    spec = (
        Model.uniform(TwoPL(), items=fipc.columns).fix_from(bank).with_latent(EmpiricalHistogram())
    )
    return _problem(spec, fipc, quadrature=Quadrature.equispaced(21))


PROBLEMS = [
    "2pl_missing",
    pytest.param("mixed_priors", marks=pytest.mark.slow),  # four kernels: ~10 s of compilation
    "two_group",
    pytest.param("two_factor", marks=pytest.mark.slow),
    "fipc_eh",
]


@pytest.mark.parametrize("name", PROBLEMS)
@pytest.mark.parametrize("budget", [None, 20_000])
def test_chunked_scores_match_dense_autodiff(
    name: str, budget: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The chunked reverse-mode scores equal a dense forward-mode Jacobian, for any chunk size."""
    prob, eta = _named_problem(name)
    if budget is not None:
        monkeypatch.setattr(_robust_se, "_CHUNK_ELEMENTS", budget)
        assert _robust_se._chunk_size(prob) < prob.y.shape[0]
    scores = _dense_scores(prob, eta)
    counts = np.asarray(prob.counts)
    cross = np.asarray(_jit(_robust_se.score_cross_product)(prob, eta))
    expected = (counts[:, None] * scores).T @ scores
    np.testing.assert_allclose(cross, expected, rtol=0, atol=1e-10 * np.abs(expected).max())
    total = np.asarray(_jit(_robust_se.score_sum)(prob, eta))
    gradient = np.asarray(jax.grad(lambda e: _core.marginal_loglik(prob, e))(eta))
    np.testing.assert_allclose(
        total, gradient, rtol=0, atol=1e-9 * max(1.0, np.abs(gradient).max())
    )


def test_prior_curvature_is_added_like_in_the_hessian() -> None:
    """``B_post - B`` is minus the Hessian of the log prior (zero where there is no prior)."""
    prob, eta = _named_problem("2pl_priors")
    with_prior = np.asarray(_jit(_robust_se.information_cross_product)(prob, eta))
    without = np.asarray(_jit(_robust_se.score_cross_product)(prob, eta))
    curvature = -np.asarray(jax.hessian(lambda e: _core.log_prior(prob, e))(eta))
    np.testing.assert_allclose(with_prior - without, curvature, rtol=0, atol=1e-9)
    has_prior = np.asarray(prob.priors.kind) != 0
    assert has_prior.any() and not has_prior.all()
    assert np.all(np.diag(curvature)[has_prior] != 0.0)
    assert np.all(curvature[~has_prior] == 0.0)


def test_other_groups_parameters_have_zero_scores() -> None:
    """A reference-group pattern has a zero score for the focal latent mean and variance."""
    prob, eta = _named_problem("two_group")
    scores = _dense_scores(prob, eta)
    latent_ids = np.asarray(prob.latent.free_index)
    focal_ids = latent_ids[1][latent_ids[1] >= 0]
    assert focal_ids.size == 2
    in_ref = (np.asarray(prob.group) == 0) & (np.asarray(prob.counts) > 0)
    assert np.all(scores[np.ix_(in_ref, focal_ids)] == 0.0)
    assert np.all(np.abs(scores[np.ix_(~in_ref, focal_ids)]).sum(axis=0) > 0.0)


@cache
def _fits(case: str, se: str | None) -> FitResult:
    assert case == "misfit"  # strong guessing fitted by a 2PL
    data = _simulate_2pl(12, 3000, 6, guess=0.3)
    spec = Model.uniform(TwoPL(), items=data.columns)
    return fit(spec, data, method=METHOD, se=se)  # type: ignore[arg-type]


def test_information_equality_under_a_correct_model() -> None:
    """With a correct model and N = 20,000, ``B`` equals ``H`` within sampling error.

    The relative sampling error of an entry of ``B / N`` is of order ``N**-0.5 = 0.7%`` times a
    kurtosis factor; the three standard errors must agree within 3%, and ``B`` with ``H`` within
    10% in the metric of ``H`` (the 12 eigenvalues of ``H^{-1/2} B H^{-1/2}``, whose extremes
    spread more than single entries; observed 0.955 to 1.071).
    """
    data = _simulate_2pl(11, 20_000, 6)
    spec = Model.uniform(TwoPL(), items=data.columns)
    res = fit(spec, data, method=METHOD, se="oakes")
    assert res.vcov is not None
    prob, _ = _problem(spec, data)
    cross_product = np.asarray(_jit(_robust_se.information_cross_product)(prob, res._eta))
    se_oakes = np.sqrt(np.diag(res.vcov))
    se_cross = np.sqrt(np.diag(np.linalg.inv(cross_product)))
    se_sandwich = np.sqrt(np.diag(res.vcov @ cross_product @ res.vcov))
    np.testing.assert_allclose(se_cross, se_oakes, rtol=0.03)
    np.testing.assert_allclose(se_sandwich, se_oakes, rtol=0.03)
    half = np.linalg.cholesky(res.vcov)  # H^{-1} = L L^T
    eigenvalues = np.linalg.eigvalsh(half.T @ cross_product @ half)
    assert eigenvalues.min() > 0.9 and eigenvalues.max() < 1.1


def test_sandwich_differs_under_misspecification() -> None:
    """Fitting a 2PL to data with guessing 0.3 breaks the information equality visibly."""
    oakes, sandwich = _fits("misfit", "oakes"), _fits("misfit", "sandwich")
    assert oakes.vcov is not None and sandwich.vcov is not None
    np.testing.assert_allclose(sandwich.parameters["estimate"], oakes.parameters["estimate"])
    ratio = np.sqrt(np.diag(sandwich.vcov) / np.diag(oakes.vcov))
    assert np.abs(ratio - 1.0).max() > 0.05


def test_sandwich_is_oakes_when_cross_product_is_the_information(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forcing ``B = H`` turns ``H^{-1} B H^{-1}`` into the Oakes covariance."""
    fit_module = sys.modules["flexpyirt.fit"]
    data = _simulate_2pl(13, 500, 5)
    spec = Model.uniform(TwoPL(), items=data.columns).prior(param="a", dist=NormalPrior(1.0, 0.5))
    oakes = fit(spec, data, method=METHOD, se="oakes")
    monkeypatch.setitem(fit_module._information, "crossprod", fit_module._information["oakes"])
    sandwich = fit(spec, data, method=METHOD, se="sandwich")
    cross = fit(spec, data, method=METHOD, se="crossprod")
    assert oakes.vcov is not None and sandwich.vcov is not None and cross.vcov is not None
    scale = np.abs(oakes.vcov).max()
    np.testing.assert_allclose(sandwich.vcov, oakes.vcov, rtol=0, atol=1e-10 * scale)
    np.testing.assert_allclose(cross.vcov, oakes.vcov, rtol=0, atol=1e-10 * scale)
    assert sandwich.se_method == "sandwich" and cross.se_method == "crossprod"


def test_fit_composes_the_sandwich_from_its_parts() -> None:
    """``vcov`` of ``se="sandwich"`` is ``V_oakes B_post V_oakes``, ``B_post = V_crossprod^-1``."""
    oakes, cross, sandwich = (_fits("misfit", se) for se in ("oakes", "crossprod", "sandwich"))
    assert oakes.vcov is not None and cross.vcov is not None and sandwich.vcov is not None
    expected = oakes.vcov @ np.linalg.inv(cross.vcov) @ oakes.vcov
    np.testing.assert_allclose(sandwich.vcov, expected, rtol=0, atol=1e-9 * np.abs(expected).max())
    np.testing.assert_allclose(sandwich.vcov, sandwich.vcov.T, rtol=0, atol=0)
    for res, vcov in ((cross, cross.vcov), (sandwich, sandwich.vcov)):
        table = res.parameters
        free = table["free"].to_numpy()
        np.testing.assert_allclose(
            table.loc[free, "se"], np.sqrt(np.diag(vcov))[table.loc[free, "free_id"]]
        )


@pytest.mark.parametrize("se", ["crossprod", "sandwich"])
def test_replicated_rows_are_counts(se: str) -> None:
    """Doubling every row doubles ``B`` and ``H``: the covariance halves exactly.

    Patterns are compressed with their counts, so this is the statement that a pattern observed
    ``c`` times contributes ``c`` identical person-level terms.
    """
    data = _simulate_2pl(14, 300, 5)
    spec = Model.uniform(TwoPL(), items=data.columns)
    tight = BAEM(tol=1e-10, max_iter=20000)
    once = fit(spec, data, method=tight, se=se)  # type: ignore[arg-type]
    shuffled = pd.concat([data, data]).sample(frac=1.0, random_state=0).reset_index(drop=True)
    twice = fit(spec, shuffled, method=tight, se=se)  # type: ignore[arg-type]
    assert once.vcov is not None and twice.vcov is not None
    np.testing.assert_allclose(twice.parameters["estimate"], once.parameters["estimate"], atol=1e-8)
    np.testing.assert_allclose(2.0 * twice.vcov, once.vcov, rtol=1e-6)


def test_display_standard_errors_use_the_chosen_covariance() -> None:
    """``b_se`` is the delta-method SE of ``b = -d / a`` under the sandwich covariance."""
    oakes, sandwich = _fits("misfit", "oakes"), _fits("misfit", "sandwich")
    assert sandwich.vcov is not None
    table = sandwich.parameters
    items = sandwich.items.set_index("item")
    for name in items.index:
        rows = table[table["item"] == name].set_index("param")
        a, d = rows.loc["a", "estimate"], rows.loc["d", "estimate"]
        ids = [int(rows.loc["a", "free_id"]), int(rows.loc["d", "free_id"])]
        jacobian = np.array([d / a**2, -1.0 / a])
        expected = np.sqrt(jacobian @ sandwich.vcov[np.ix_(ids, ids)] @ jacobian)
        assert items.loc[name, "b_se"] == pytest.approx(expected, rel=1e-8)
    assert not np.allclose(items["b_se"], oakes.items.set_index("item")["b_se"], rtol=0.02)


def test_linking_uses_the_chosen_covariance() -> None:
    """``link()`` propagates ``vcov``: the sandwich fit gives other linking SEs than Oakes."""
    target = fit(
        Model.uniform(TwoPL(), items=_simulate_2pl(12, 10, 6).columns),
        _simulate_2pl(15, 2000, 6, guess=0.3, mean=0.5, sd=1.2),
        method=METHOD,
        se="sandwich",
    )
    robust = link(_fits("misfit", "sandwich"), target)
    plain = link(_fits("misfit", "oakes"), target)
    assert abs(robust.A - plain.A) < 1e-10
    assert np.isfinite(robust.A_se) and np.isfinite(robust.B_se)
    assert abs(robust.A_se / plain.A_se - 1.0) > 0.01


def test_json_round_trip_keeps_the_se_method() -> None:
    for se in ("crossprod", "sandwich"):
        res = _fits("misfit", se)
        back = FitResult.from_json(res.to_json())
        assert back.se_method == se
        assert back.vcov is not None and res.vcov is not None
        np.testing.assert_array_equal(back.vcov, res.vcov)
        assert f"se={se}" in repr(back)


def test_empirical_histogram_keeps_the_item_only_covariance() -> None:
    """EH: ``vcov`` covers the free item parameters, with the histogram held fixed."""
    data = _simulate_2pl(16, 800, 6, mean=0.2)
    bank = pd.DataFrame(
        {"item": ["q01", "q02", "q03"], "a": [1.0, 1.1, 0.9], "d": [0.0, 0.3, -0.2]}
    )
    spec = (
        Model.uniform(TwoPL(), items=data.columns).fix_from(bank).with_latent(EmpiricalHistogram())
    )
    method = BAEM(tol=1e-4, max_iter=2000, squarem=True)
    quadrature = Quadrature.equispaced(21)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = {
            se: fit(spec, data, method=method, quadrature=quadrature, se=se) for se in SE_METHODS
        }
    n_free = 2 * (6 - 3)
    for res in results.values():
        assert res.vcov is not None and res.vcov.shape == (n_free, n_free)
        assert np.all(np.isfinite(res.vcov)) and np.all(np.diag(res.vcov) > 0)
    oakes, cross, sandwich = (results[k].vcov for k in ("oakes", "crossprod", "sandwich"))
    assert oakes is not None and cross is not None and sandwich is not None
    expected = oakes @ np.linalg.inv(cross) @ oakes
    np.testing.assert_allclose(sandwich, expected, rtol=0, atol=1e-9 * np.abs(expected).max())


@pytest.mark.parametrize("se", ["crossprod", "sandwich"])
def test_davidian_and_mixture_are_rejected_before_fitting(se: str) -> None:
    data = _simulate_2pl(17, 50, 5)
    for latent, label in ((DavidianCurve(2), "DavidianCurve"), (NormalMixture(2), "NormalMixture")):
        spec = Model.uniform(TwoPL(), items=data.columns).with_latent(latent)
        with pytest.raises(SpecificationError, match=f"se='{se}' is not available with a {label}"):
            fit(spec, data, se=se)  # type: ignore[arg-type]


@pytest.mark.parametrize("se", ["crossprod", "sandwich"])
def test_bifactor_and_two_tier_are_rejected_before_fitting(se: str) -> None:
    """Dimension-reduced models are not covered (DESIGN.md §17.5): the scores use the full grid."""
    data = _simulate_2pl(18, 50, 6)
    base = Model.uniform(TwoPL(), items=data.columns)
    reduced = (
        base.with_bifactor(specific={"s1": ["q01", "q02", "q03"], "s2": ["q04", "q05", "q06"]}),
        base.with_two_tier(
            primary={"p1": ["q01", "q02", "q03"], "p2": ["q04", "q05", "q06"]},
            specific={"s1": ["q01", "q04"], "s2": ["q02", "q05"]},
        ),
    )
    for spec in reduced:
        assert spec.is_reduced
        with pytest.raises(SpecificationError, match="not available for bifactor and two-tier"):
            fit(spec, data, se=se)  # type: ignore[arg-type]


@pytest.mark.parametrize("se", ["crossprod", "sandwich"])
def test_weights_are_rejected_before_fitting(se: str) -> None:
    """The cross-product does not account for sampling weights yet (DESIGN.md §17.5)."""
    data = _simulate_2pl(19, 50, 6)
    spec = Model.uniform(TwoPL(), items=data.columns)
    with pytest.raises(SpecificationError, match="not available with weights="):
        fit(spec, data, se=se, weights=np.ones(len(data)))  # type: ignore[arg-type]


def test_unknown_se_lists_the_options() -> None:
    data = _simulate_2pl(17, 50, 5)
    spec = Model.uniform(TwoPL(), items=data.columns)
    with pytest.raises(SpecificationError, match="'oakes', 'hessian', 'sandwich', 'crossprod'"):
        fit(spec, data, se="louis")  # type: ignore[arg-type]


def test_jsi_one_step_needs_the_observed_information() -> None:
    """``jsi(method="one_step")`` takes a Newton step with ``vcov``; a robust one is refused."""
    data = _simulate_2pl(12, 3000, 6, guess=0.3)
    with pytest.raises(SpecificationError, match="se='oakes' or se='hessian', not se='sandwich'"):
        _fits("misfit", "sandwich").jsi(data, method="one_step")
