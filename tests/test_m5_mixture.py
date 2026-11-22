"""Unit tests for normal-mixture latent distributions (DESIGN.md §4.4, §16.7)."""

from __future__ import annotations

import json
import warnings
from functools import cache
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from flexpyirt import (
    BAEM,
    FitResult,
    Model,
    Normal,
    NormalMixture,
    NormalPrior,
    Quadrature,
    Rasch,
    TwoPL,
    fit,
    link,
)
from flexpyirt._compile import compile_problem, prepare
from flexpyirt._core import information_hessian, information_oakes
from flexpyirt.errors import HeywoodWarning, IdentificationError, SpecificationError
from flexpyirt.latent import (
    MAX_MIXTURE_COMPONENTS,
    mixture_components,
    mixture_log_weights,
    mixture_param_names,
    normal_log_weights,
)
from flexpyirt.linking import _fitted_grid


def _bimodal_2pl_data(
    *, seed: int = 20261122, n_obs: int = 1500, n_items: int = 16
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """0.65 N(0, 1) + 0.35 N(2.2, 0.6^2) trait, 2PL items."""
    rng = np.random.default_rng(seed)
    cls = rng.random(n_obs) < 0.35
    theta = np.where(cls, rng.normal(2.2, 0.6, n_obs), rng.normal(0.0, 1.0, n_obs))
    a = rng.uniform(0.9, 1.9, size=n_items)
    d = np.linspace(-1.2, 2.2, n_items)
    probs = 1.0 / (1.0 + np.exp(-(theta[:, None] * a[None, :] + d[None, :])))
    y = (rng.random((n_obs, n_items)) < probs).astype(int)
    cols = [f"q{j + 1:02d}" for j in range(n_items)]
    return pd.DataFrame(y, columns=cols), a, d


@cache
def _fitted_mix2(squarem: bool = True, se: str = "oakes") -> tuple[pd.DataFrame, FitResult]:
    df, _, _ = _bimodal_2pl_data()
    spec = Model.uniform(TwoPL(), items=df.columns).with_latent(NormalMixture(2))
    res = fit(
        spec,
        df,
        method=BAEM(tol=1e-8, max_iter=3000, squarem=squarem),
        se=se,  # type: ignore[arg-type]
    )
    return df, res


def _random_params(rng: np.random.Generator, k: int) -> np.ndarray:
    out = [rng.normal(0.2, 0.5), rng.uniform(0.5, 2.0)]
    for _ in range(2, k + 1):
        out.extend([rng.normal(0.0, 2.0), rng.uniform(0.2, 2.5), rng.normal(0.0, 1.0)])
    return np.asarray(out)


# --------------------------------------------------------------------------------------------
# Spec validation
# --------------------------------------------------------------------------------------------


def test_spec_validation_and_param_names() -> None:
    m = NormalMixture(2)
    assert (m.n_components, m.mean, m.var, m.free_mean, m.free_var) == (2, 0.0, 1.0, False, False)
    assert m.means is None and m.variances is None and m.weights is None
    f = NormalMixture.free(3, mean=0.5, var=2.0, means=[0, 1, 2], weights=[0.5, 0.25, 0.25])
    assert (f.free_mean, f.free_var, f.mean, f.var) == (True, True, 0.5, 2.0)
    assert f.means == (0.0, 1.0, 2.0) and f.weights == (0.5, 0.25, 0.25)
    assert hash(NormalMixture(2, means=[0.0, 1.0])) == hash(NormalMixture(2, means=(0.0, 1.0)))

    for bad_k in (0, 1, MAX_MIXTURE_COMPONENTS + 1, True, 2.0, "2"):
        with pytest.raises(SpecificationError, match="n_components"):
            NormalMixture(bad_k)  # type: ignore[arg-type]
        with pytest.raises(SpecificationError, match="n_components"):
            mixture_param_names(bad_k)  # type: ignore[arg-type]
    for bad in (float("nan"), float("inf"), True, "0"):
        with pytest.raises(SpecificationError, match="mean"):
            NormalMixture(2, mean=bad)  # type: ignore[arg-type]
    for bad in (0.0, -1.0, float("nan"), "1"):
        with pytest.raises(SpecificationError, match="var"):
            NormalMixture(2, var=bad)  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match="free_mean"):
        NormalMixture(2, free_mean=1)  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match="free_var"):
        NormalMixture(2, free_var="no")  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match=r"means.*2 entries"):
        NormalMixture(2, means=[0.0, 1.0, 2.0])
    with pytest.raises(SpecificationError, match=r"means.*finite"):
        NormalMixture(2, means=[0.0, float("nan")])
    with pytest.raises(SpecificationError, match=r"means.*sequence"):
        NormalMixture(2, means="01")  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match=r"variances.*floats"):
        NormalMixture(2, variances=[1.0, True])
    with pytest.raises(SpecificationError, match="variances must be positive"):
        NormalMixture(2, variances=[1.0, 0.0])
    with pytest.raises(SpecificationError, match="weights must be positive"):
        NormalMixture(2, weights=[1.0, 0.0])
    with pytest.raises(SpecificationError, match="sum to 1"):
        NormalMixture(2, weights=[0.5, 0.6])

    assert mixture_param_names(2) == ("mean", "var", "rmean_2", "rvar_2", "logit_2")
    assert len(mixture_param_names(4)) == 2 + 3 * 3


def test_raw_starts_map_any_affine_scale_to_the_raw_gauge() -> None:
    assert NormalMixture(2).raw_starts() == pytest.approx((1.5, 1.0, 0.0))
    assert NormalMixture(4).raw_starts() == pytest.approx(
        (1.5, 1.0, 0.0, -1.5, 1.0, 0.0, 3.0, 1.0, 0.0)
    )
    shape = NormalMixture(
        3, means=[-1.0, 1.0, 3.0], variances=[0.25, 1.0, 0.5], weights=[0.5, 0.3, 0.2]
    )
    # The same shape written in another affine scale (theta' = 2 theta + 1) gives the same start.
    scaled = NormalMixture(
        3, means=[-1.0, 3.0, 7.0], variances=[1.0, 4.0, 2.0], weights=[0.5, 0.3, 0.2]
    )
    assert shape.raw_starts() == pytest.approx(scaled.raw_starts())
    assert shape.raw_starts() == pytest.approx((4.0, 4.0, np.log(0.6), 8.0, 2.0, np.log(0.4)))
    df, _, _ = _bimodal_2pl_data(n_obs=100, n_items=4)
    tbl = Model.uniform(TwoPL(), items=df.columns).with_latent(shape).parameter_table(df)
    lat = tbl[tbl["item"] == "__latent__"]
    assert list(lat["param"]) == list(mixture_param_names(3))
    assert list(lat["free"]) == [False, False] + [True] * 6
    assert list(lat["model"].unique()) == ["Mixture"]
    np.testing.assert_allclose(lat["value"].to_numpy()[2:], shape.raw_starts())


def test_model_unsupported_combinations() -> None:
    items = ["q1", "q2", "q3", "q4"]
    base = Model.uniform(TwoPL(), items=items)
    with pytest.raises(SpecificationError, match="single-group"):
        base.with_groups(ref=NormalMixture(2), focal=NormalMixture(2))
    with pytest.raises(SpecificationError, match="single-group"):
        base.with_groups(ref=Normal.standard(), focal=NormalMixture(2))
    with pytest.raises(SpecificationError, match="single-group"):
        Model(items=base.items, groups=(("ref", NormalMixture(2)), ("focal", NormalMixture(2))))
    with pytest.raises(SpecificationError, match="unidimensional"):
        base.with_factors({"F1": ["q1", "q2"], "F2": ["q3", "q4"]}).with_latent(NormalMixture(2))
    with pytest.raises(SpecificationError, match="with_latent"):
        base.with_latent(NormalMixture(2)).with_factors({"F1": ["q1", "q2"], "F2": ["q3", "q4"]})
    with pytest.raises(SpecificationError, match="latent"):
        base.with_latent(NormalMixture(2)).prior(param="rmean_2", dist=NormalPrior(0.0, 1.0))

    df, _, _ = _bimodal_2pl_data(n_obs=200, n_items=4)
    spec = Model.uniform(TwoPL(), items=df.columns).with_latent(NormalMixture(2))
    tbl = spec.parameter_table(df)
    tbl["prior"] = pd.Series(
        [
            NormalPrior(0.0, 1.0) if p == "logit_2" else cur
            for p, cur in zip(tbl["param"], tbl["prior"], strict=True)
        ],
        dtype=object,
    )
    with pytest.raises(SpecificationError, match="Priors on latent"):
        fit(spec, df, table=tbl, se=None)
    bad = spec.parameter_table(df)
    bad.loc[bad["param"] == "rvar_2", ["value", "free"]] = [0.0, False]
    with pytest.raises(SpecificationError, match="variance must be positive"):
        fit(spec, df, table=bad, se=None)


# --------------------------------------------------------------------------------------------
# Density, standardization and grid weights
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("k", [2, 3, 5])
def test_components_are_exactly_standardized(k: int) -> None:
    rng = np.random.default_rng(k)
    for _ in range(5):
        params = _random_params(rng, k)
        pi, means, variances = (np.asarray(x) for x in mixture_components(jnp.asarray(params)))
        assert pi.sum() == pytest.approx(1.0, abs=1e-15)
        assert pi[1:] / pi[0] == pytest.approx(np.exp(params[4::3]), rel=1e-12)
        assert float(np.sum(pi * means)) == pytest.approx(params[0], abs=1e-12)
        total_var = float(np.sum(pi * (variances + means**2)) - params[0] ** 2)
        assert total_var == pytest.approx(params[1], rel=1e-12)
        # Ratios to component 1 reproduce the raw gauge (rmean_c, rvar_c).
        sd1 = np.sqrt(variances[0])
        np.testing.assert_allclose((means[1:] - means[0]) / sd1, params[2::3], rtol=1e-12)
        np.testing.assert_allclose(variances[1:] / variances[0], params[3::3], rtol=1e-12)


@pytest.mark.parametrize("k", [2, 3, 4])
def test_log_weights_match_direct_mixture_of_normalized_components(k: int) -> None:
    rng = np.random.default_rng(10 + k)
    nodes = np.linspace(-6.0, 6.0, 61)
    params = _random_params(rng, k)
    logw = np.asarray(mixture_log_weights(jnp.asarray(nodes), jnp.asarray(params)))
    assert np.exp(logw).sum() == pytest.approx(1.0, abs=1e-14)
    pi, means, variances = (np.asarray(x) for x in mixture_components(jnp.asarray(params)))
    direct = np.zeros_like(nodes)
    for c in range(k):
        dens = np.exp(-0.5 * (nodes - means[c]) ** 2 / variances[c])
        direct += pi[c] * dens / dens.sum()
    np.testing.assert_allclose(np.exp(logw), direct, rtol=1e-12, atol=1e-300)
    # (Q, 1) nodes give the same weights.
    np.testing.assert_allclose(
        np.asarray(mixture_log_weights(jnp.asarray(nodes[:, None]), jnp.asarray(params))), logw
    )


def test_identical_components_reduce_to_normal_on_both_grids() -> None:
    # rmean_c = 0, rvar_c = 1: every component is N(mean, var), whatever the logits.
    params = jnp.asarray([0.3, 1.7, 0.0, 1.0, 0.4, 0.0, 1.0, -1.1])
    eq = Quadrature.equispaced(61)
    nodes = jnp.asarray(eq.nodes())
    np.testing.assert_allclose(
        np.asarray(mixture_log_weights(nodes, params)),
        np.asarray(normal_log_weights(nodes, jnp.asarray(0.3), jnp.asarray(1.7))),
        rtol=0,
        atol=1e-13,
    )
    gh = Quadrature.gauss_hermite(21)
    base = jnp.asarray(gh.base_log_weights(1))
    gh_nodes = jnp.asarray(gh.nodes())
    expected = jax.nn.log_softmax(
        base + normal_log_weights(gh_nodes, jnp.asarray(0.3), jnp.asarray(1.7))
    )
    np.testing.assert_allclose(
        np.asarray(mixture_log_weights(gh_nodes, params, base)),
        np.asarray(expected),
        rtol=0,
        atol=1e-13,
    )


def test_infeasible_parameters_give_nan_and_gradients_are_finite() -> None:
    nodes = jnp.linspace(-6.0, 6.0, 61)
    good = jnp.asarray([0.0, 1.0, 2.0, 0.4, -0.5])
    for bad in (good.at[3].set(0.0), good.at[3].set(-0.1), good.at[1].set(0.0)):
        assert bool(jnp.all(jnp.isnan(mixture_log_weights(nodes, bad))))
    counts = jnp.exp(mixture_log_weights(nodes, good)) * 1000.0
    grad = jax.grad(lambda p: jnp.sum(counts * mixture_log_weights(nodes, p)))(good)
    hess = jax.hessian(lambda p: jnp.sum(counts * mixture_log_weights(nodes, p)))(good)
    assert bool(jnp.all(jnp.isfinite(grad))) and bool(jnp.all(jnp.isfinite(hess)))
    # The fitted counts are a stationary point of the latent Q in the free shape parameters.
    np.testing.assert_allclose(np.asarray(grad[2:]), 0.0, atol=1e-9)


# --------------------------------------------------------------------------------------------
# Estimation
# --------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_em_squarem_monotone_same_optimum_and_oakes_equals_hessian() -> None:
    df, res_sq = _fitted_mix2()
    _, res_em = _fitted_mix2(squarem=False, se="hessian")
    assert res_sq.converged and res_em.converged
    for res in (res_em, res_sq):
        ll = res.history["loglik"].to_numpy()
        assert np.all(np.diff(ll) >= -1e-10 * np.abs(ll[1:]))
    assert res_sq.n_em_evaluations < res_em.n_em_evaluations
    assert res_sq.loglik == pytest.approx(res_em.loglik, rel=1e-11)
    np.testing.assert_allclose(res_sq._eta, res_em._eta, rtol=0, atol=2e-5)

    spec = res_sq._model
    prep = prepare(spec, df, group=None, item_names=None, table=res_sq.parameters)
    comp = compile_problem(spec, prep.y, prep.group, prep.table, res_sq._quadrature)
    eta = jnp.asarray(res_sq._eta)
    oakes = np.asarray(information_oakes(comp.problem, eta))
    hess = np.asarray(information_hessian(comp.problem, eta))
    np.testing.assert_allclose(oakes, hess, rtol=1e-9, atol=1e-9 * np.abs(hess).max())

    # A two-component mixture is at least as good as the nested Normal model.
    res_nm = fit(Model.uniform(TwoPL(), items=df.columns), df, se=None)
    assert res_sq.loglik > res_nm.loglik + 5.0


def test_relabelled_starts_reach_the_same_mixture() -> None:
    df, res = _fitted_mix2()
    spec = Model.uniform(TwoPL(), items=df.columns)
    lat = res.latent
    for mix in (
        NormalMixture(2, means=[0.0, -1.5]),
        NormalMixture(2, means=[0.0, 3.0], variances=[1.0, 0.3], weights=[0.8, 0.2]),
        NormalMixture(2, means=[2.0, 0.0], variances=[0.3, 1.0], weights=[0.3, 0.7]),
    ):
        other = fit(
            spec.with_latent(mix), df, method=BAEM(tol=1e-8, max_iter=3000, squarem=True), se=None
        )
        assert other.loglik == pytest.approx(res.loglik, rel=1e-11)
        np.testing.assert_allclose(other.items["a"], res.items["a"], atol=2e-5)
        # Components may be relabelled; compare them sorted by mean.
        o_lat = other.latent
        for col in ("mean", "var", "pi"):
            ours = sorted(
                zip(
                    [float(o_lat[f"mean_{c}"].iloc[0]) for c in (1, 2)],
                    [float(o_lat[f"{col}_{c}"].iloc[0]) for c in (1, 2)],
                    strict=True,
                )
            )
            ref = sorted(
                zip(
                    [float(lat[f"mean_{c}"].iloc[0]) for c in (1, 2)],
                    [float(lat[f"{col}_{c}"].iloc[0]) for c in (1, 2)],
                    strict=True,
                )
            )
            np.testing.assert_allclose(np.array(ours)[:, 1], np.array(ref)[:, 1], atol=2e-5)


def test_latent_table_derived_components_and_delta_ses() -> None:
    _, res = _fitted_mix2()
    lat = res.latent
    cols = list(lat.columns)
    for name in ("mean", "var", "rmean_2", "rvar_2", "logit_2", "pi_1", "pi_2", "mean_1", "var_2"):
        assert name in cols and f"{name}_se" in cols
    row = lat.iloc[0]
    assert row["pi_1"] + row["pi_2"] == pytest.approx(1.0, abs=1e-14)
    assert row["pi_1_se"] == pytest.approx(row["pi_2_se"], rel=1e-10)
    assert row["pi_1"] * row["mean_1"] + row["pi_2"] * row["mean_2"] == pytest.approx(
        0.0, abs=1e-12
    )
    assert np.isnan(row["mean_se"]) and np.isnan(row["var_se"])
    # The bimodal truth (0.65 N(0,1) + 0.35 N(2.2, 0.36)) is recovered on the standardized scale.
    assert sorted([row["pi_1"], row["pi_2"]]) == pytest.approx([0.35, 0.65], abs=0.06)
    # Delta-method SE of pi_2 from the logit: pi_1 pi_2 * se(logit_2).
    assert row["pi_2_se"] == pytest.approx(row["pi_1"] * row["pi_2"] * row["logit_2_se"], rel=1e-8)
    assert "NormalMixture, n_components=2" in res.summary()
    assert "NormalMixture, n_components=2" in res._repr_html_()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        no_se = fit(
            res._model, _fitted_mix2()[0], table=res.parameters, se=None, method=BAEM(max_iter=1)
        )
    assert np.isnan(float(no_se.latent["pi_1_se"].iloc[0]))


def test_identification_rasch_and_fipc() -> None:
    df, a_true, d_true = _bimodal_2pl_data(n_obs=800, n_items=8)
    with pytest.raises(IdentificationError, match="variance"):
        fit(
            Model.uniform(TwoPL(), items=df.columns).with_latent(NormalMixture(2, free_var=True)),
            df,
            se=None,
        )
    with pytest.raises(IdentificationError, match="mean"):
        fit(
            Model.uniform(TwoPL(), items=df.columns).with_latent(NormalMixture(2, free_mean=True)),
            df,
            se=None,
        )
    res_rasch = fit(
        Model.uniform(Rasch(), items=df.columns).with_latent(NormalMixture(2, free_var=True)),
        df,
        method=BAEM(tol=1e-6, max_iter=2000, squarem=True),
        se="oakes",
    )
    assert res_rasch.converged
    assert np.isnan(float(res_rasch.latent["mean_se"].iloc[0]))
    assert float(res_rasch.latent["var_se"].iloc[0]) > 0.0

    bank = pd.DataFrame({"item": list(df.columns[:4]), "a": a_true[:4], "d": d_true[:4]})
    res_fipc = fit(
        Model.uniform(TwoPL(), items=df.columns).fix_from(bank).with_latent(NormalMixture.free(2)),
        df,
        method=BAEM(tol=1e-6, max_iter=2000, squarem=True),
        se="oakes",
    )
    assert res_fipc.converged
    lat = res_fipc.latent.iloc[0]
    assert lat["mean_se"] > 0.0 and lat["var_se"] > 0.0
    # Moments of the generating trait 0.65 N(0, 1) + 0.35 N(2.2, 0.36).
    true_mean = 0.35 * 2.2
    true_var = 0.65 * 1.0 + 0.35 * 0.36 + 0.65 * 0.35 * 2.2**2
    assert lat["mean"] == pytest.approx(true_mean, abs=0.25)
    assert lat["var"] == pytest.approx(true_var, rel=0.3)


def test_fixed_identical_components_match_normal_fit() -> None:
    df, _, _ = _bimodal_2pl_data(n_obs=600, n_items=6)
    spec_mix = Model.uniform(TwoPL(), items=df.columns).with_latent(NormalMixture(2))
    tbl = spec_mix.parameter_table(df)
    for param, value in (("rmean_2", 0.0), ("rvar_2", 1.0), ("logit_2", 0.3)):
        tbl.loc[tbl["param"] == param, ["value", "free"]] = [value, False]
    with pytest.warns(HeywoodWarning, match="coincide"):
        res_mix = fit(spec_mix, df, table=tbl, method=BAEM(tol=1e-8, squarem=True), se="oakes")
    res_nm = fit(Model.uniform(TwoPL(), items=df.columns), df, method=BAEM(tol=1e-8, squarem=True))
    assert res_mix.loglik == pytest.approx(res_nm.loglik, rel=1e-12)
    np.testing.assert_allclose(res_mix.items["a"], res_nm.items["a"], atol=1e-7)
    assert res_mix.vcov is not None and res_nm.vcov is not None
    np.testing.assert_allclose(res_mix.vcov, res_nm.vcov, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        ({"rmean_2": 1.0, "rvar_2": 1.0, "logit_2": -9.0}, "weight below"),
        ({"rmean_2": 2.0, "rvar_2": 1e-4, "logit_2": 0.0}, "half the grid spacing"),
        ({"rmean_2": 40.0, "rvar_2": 1.0, "logit_2": -6.0}, "outside the grid"),
    ],
)
def test_degenerate_solutions_warn(values: dict[str, float], match: str) -> None:
    df, _, _ = _bimodal_2pl_data(n_obs=300, n_items=5)
    spec = Model.uniform(TwoPL(), items=df.columns).with_latent(NormalMixture(2))
    tbl = spec.parameter_table(df)
    for param, value in values.items():
        tbl.loc[tbl["param"] == param, ["value", "free"]] = [value, False]
    with pytest.warns(HeywoodWarning, match=match):
        fit(spec, df, table=tbl, method=BAEM(tol=1e-4), se=None)


def test_well_separated_fit_does_not_warn() -> None:
    df, res = _fitted_mix2()
    with warnings.catch_warnings():
        warnings.simplefilter("error", HeywoodWarning)
        fit(res._model, df, method=BAEM(tol=1e-6, max_iter=3000, squarem=True), se=None)


def test_three_components_and_gauss_hermite() -> None:
    df, res = _fitted_mix2()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", HeywoodWarning)
        res3 = fit(
            Model.uniform(TwoPL(), items=df.columns).with_latent(NormalMixture(3)),
            df,
            method=BAEM(tol=1e-6, max_iter=3000, squarem=True),
            se=None,
        )
    assert res3.converged
    assert res3.loglik >= res.loglik - 1e-6
    assert res3.n_params == res.n_params + 3
    pis = [float(res3.latent[f"pi_{c}"].iloc[0]) for c in (1, 2, 3)]
    assert sum(pis) == pytest.approx(1.0, abs=1e-14)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", HeywoodWarning)
        res_gh = fit(
            res._model,
            df,
            method=BAEM(tol=1e-7, max_iter=3000, squarem=True),
            quadrature=Quadrature.gauss_hermite(61),
            se=None,
        )
    assert res_gh.converged
    assert res_gh.loglik == pytest.approx(res.loglik, rel=1e-5)


# --------------------------------------------------------------------------------------------
# Scoring, other result methods, linking and JSON
# --------------------------------------------------------------------------------------------


def test_scoring_and_rejected_methods() -> None:
    df, res = _fitted_mix2()
    sub = df.iloc[:25]
    eap = res.score(sub, method="EAP")
    nodes = res._quadrature.nodes()
    lat = res.parameters[res.parameters["item"] == "__latent__"]["estimate"].to_numpy()
    logw = np.asarray(mixture_log_weights(jnp.asarray(nodes), jnp.asarray(lat)))
    a = res.items["a"].to_numpy()
    d = res.items["d"].to_numpy()
    y = sub.to_numpy()
    z = nodes[:, None] * a[None, :] + d[None, :]
    logp1 = -np.log1p(np.exp(-z))
    logp0 = -np.log1p(np.exp(z))
    ll = y @ logp1.T + (1 - y) @ logp0.T + logw[None, :]
    post = np.exp(ll - ll.max(axis=1, keepdims=True))
    post /= post.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(eap["theta"].to_numpy(), post @ nodes, rtol=0, atol=1e-12)

    ml = res.score(sub, method="ML")
    wle = res.score(sub, method="WLE")
    assert ml.shape == wle.shape == (25, 2)
    pv1 = res.score(sub, method="PV", key=jax.random.key(3), n_draws=4)
    pv2 = res.score(sub, method="PV", key=jax.random.key(3), n_draws=4)
    pd.testing.assert_frame_equal(pv1, pv2)
    with pytest.raises(SpecificationError, match="NormalMixture"):
        res.score(sub, method="MAP")
    with pytest.raises(SpecificationError, match="item_fit"):
        res.item_fit(df)
    with pytest.raises(SpecificationError, match="m2"):
        res.m2(df)


def test_link_uses_fitted_mixture_weights() -> None:
    df, res = _fitted_mix2()
    nodes, w = _fitted_grid(res, "all", 0)
    lat = res.parameters[res.parameters["item"] == "__latent__"]["estimate"].to_numpy()
    expected = np.exp(np.asarray(mixture_log_weights(jnp.asarray(nodes), jnp.asarray(lat))))
    np.testing.assert_allclose(w, expected, rtol=1e-12, atol=1e-300)
    out = link(res, res, items=list(df.columns))
    assert abs(out.A - 1.0) < 1e-8
    assert abs(out.B) < 1e-8


def test_json_roundtrip(tmp_path: Path) -> None:
    df, res = _fitted_mix2()
    path = tmp_path / "mix.json"
    res.to_json(path)
    back = FitResult.from_json(path)
    assert back.loglik == res.loglik
    pd.testing.assert_frame_equal(back.latent, res.latent)
    pd.testing.assert_frame_equal(back.score(df.iloc[:5]), res.score(df.iloc[:5]))

    spec = Model.uniform(Rasch(), items=df.columns).with_latent(
        NormalMixture(
            3,
            var=1.5,
            free_var=True,
            means=[0.0, 1.0, -1.0],
            variances=[1.0, 0.5, 2.0],
            weights=[0.5, 0.3, 0.2],
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        one = fit(spec, df, method=BAEM(max_iter=3), se=None)
    back2 = FitResult.from_json(one.to_json())
    assert back2._model == spec
    pd.testing.assert_frame_equal(back2.latent, one.latent)


def test_malformed_mixture_json() -> None:
    _, res = _fitted_mix2()
    payload = json.loads(res.to_json())

    def load_with(match: str, *, drop: str | None = None, **update: Any) -> None:
        bad = json.loads(json.dumps(payload))
        latent: dict[str, Any] = bad["model"]["latent"]
        if drop is not None:
            latent.pop(drop)
        latent.update(update)
        with pytest.raises(SpecificationError, match=match):
            FitResult.from_json(json.dumps(bad))

    for key in ("n_components", "mean", "var", "free_mean", "free_var", "means", "weights"):
        load_with(key, drop=key)
    load_with("extra", extra=1)
    for bad_k in (1, MAX_MIXTURE_COMPONENTS + 1, True, 2.5, "2"):
        load_with("n_components", n_components=bad_k)
    load_with("var", var=0.0)
    load_with("means", means="0,1")
    load_with("weights", weights=[0.5, 0.6])
    load_with("variances", variances=[1.0, "x"])
    # Latent rows must match the declared number of components.
    load_with("rmean_3|parameter", n_components=3)
