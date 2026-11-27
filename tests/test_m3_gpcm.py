"""Unit and property tests for GPCM and PCM kernels, latent defaults, display, and mixed models."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from flexpyirt import BAEM, GPCM, GRM, PCM, FourPL, Model, Normal, Rasch, ThreePL, TwoPL, fit
from flexpyirt.errors import HeywoodWarning, IdentificationError, SpecificationError
from flexpyirt.items import DichotomousKernel, GPCMKernel, ItemModel


def _sim_gpcm(
    a: np.ndarray,
    d: np.ndarray,
    n: int,
    seed: int,
    *,
    mu: float = 0.0,
    var: float = 1.0,
) -> pd.DataFrame:
    """Simulate responses from GPCM with intercepts d of shape (J, K-1), d0 = 0."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(mu, np.sqrt(var), size=n)
    n_items, n_steps = d.shape
    k = np.arange(n_steps + 1, dtype=float)
    y = np.empty((n, n_items), dtype=float)
    for j in range(n_items):
        d_full = np.concatenate([[0.0], d[j]])
        logits = np.outer(theta * a[j], k) + d_full[None, :]
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        cdf = np.cumsum(probs, axis=1)
        u = rng.random(n)[:, None]
        y[:, j] = (u >= cdf).sum(axis=1)
    return pd.DataFrame(y, columns=[f"q{j:02d}" for j in range(1, n_items + 1)])


# ---- GPCMKernel ----------------------------------------------------------------------------


@pytest.mark.slow
@settings(deadline=None)
@given(
    a=st.floats(-3.0, 3.0),
    d1=st.floats(-4.0, 4.0),
    d2=st.floats(-4.0, 4.0),
    d3=st.floats(-4.0, 4.0),
)
def test_gpcm_probabilities_sum_to_one(a: float, d1: float, d2: float, d3: float) -> None:
    """Probabilities sum to 1 and match the GPCM formula with no ordering constraint on d."""
    theta = jnp.linspace(-6.0, 6.0, 25)
    params = jnp.array([a, d1, d2, d3])
    kernel = GPCMKernel(4)
    lp = kernel.log_prob(theta, params)
    p = np.exp(np.asarray(lp))
    np.testing.assert_allclose(p.sum(axis=1), 1.0, rtol=1e-12)

    th = np.asarray(theta)
    raw = np.column_stack([np.zeros_like(th), a * th + d1, 2 * a * th + d2, 3 * a * th + d3])
    raw -= raw.max(axis=1, keepdims=True)
    expected = np.exp(raw) / np.exp(raw).sum(axis=1, keepdims=True)
    np.testing.assert_allclose(p, expected, rtol=1e-12, atol=1e-15)

    def obj(x: jax.Array) -> jax.Array:
        return kernel.log_prob(theta, x).sum()

    grad = np.asarray(jax.grad(obj)(params))
    hess = np.asarray(jax.hessian(obj)(params))
    assert np.isfinite(grad).all()
    assert np.isfinite(hess).all()


def test_gpcm_and_pcm_n_cat_validation() -> None:
    for bad in (1, 0, -2):
        with pytest.raises(SpecificationError, match="at least 2"):
            GPCMKernel(bad)
        with pytest.raises(SpecificationError, match="at least 2"):
            GPCM(bad)
        with pytest.raises(SpecificationError, match="at least 2"):
            PCM(bad)


def test_gpcm_two_categories_equals_2pl_kernel() -> None:
    theta = jnp.linspace(-6.0, 6.0, 61)
    for a, d in [(1.2, -0.5), (0.5, 2.0), (2.5, -3.0), (-0.8, 0.0)]:
        params = jnp.array([a, d])
        lp_2pl = DichotomousKernel().log_prob(theta, params)
        lp_gpcm = GPCMKernel(2).log_prob(theta, params)
        np.testing.assert_allclose(np.asarray(lp_gpcm), np.asarray(lp_2pl), rtol=1e-12, atol=1e-14)


@pytest.mark.slow
def test_gpcm_two_categories_equals_2pl_fit() -> None:
    """With n_categories = 2, GPCM matches 2PL and PCM matches Rasch in estimates, SEs, and EAP."""
    a = np.array([1.1, 1.4, 0.9, 1.3, 1.0, 1.2])
    d = np.array([[-0.6], [0.3], [-0.2], [0.5], [0.0], [-0.4]])
    df = _sim_gpcm(a, d, n=800, seed=101)
    method = BAEM(tol=1e-9, max_iter=5000)

    res_2pl = fit(Model.uniform(TwoPL(), items=df.columns), df, method=method, se="oakes")
    res_gpcm = fit(Model.uniform(GPCM(2), items=df.columns), df, method=method, se="oakes")

    assert res_gpcm.converged
    assert res_gpcm.loglik == pytest.approx(res_2pl.loglik, rel=1e-12)
    np.testing.assert_allclose(res_gpcm.items["a"], res_2pl.items["a"], atol=1e-8)
    np.testing.assert_allclose(res_gpcm.items["d1"], res_2pl.items["d"], atol=1e-8)
    np.testing.assert_allclose(res_gpcm.items["b1"], res_2pl.items["b"], atol=1e-8)
    np.testing.assert_allclose(res_gpcm.items["a_se"], res_2pl.items["a_se"], rtol=1e-7)
    np.testing.assert_allclose(res_gpcm.items["d1_se"], res_2pl.items["d_se"], rtol=1e-7)
    np.testing.assert_allclose(res_gpcm.score(df)["theta"], res_2pl.score(df)["theta"], atol=1e-8)

    res_rasch = fit(Model.uniform(Rasch(), items=df.columns), df, method=method, se="oakes")
    res_pcm = fit(Model.uniform(PCM(2), items=df.columns), df, method=method, se="oakes")
    assert res_pcm.converged
    assert res_pcm.loglik == pytest.approx(res_rasch.loglik, rel=1e-12)
    np.testing.assert_allclose(res_pcm.items["d1"], res_rasch.items["d"], atol=1e-8)
    np.testing.assert_allclose(res_pcm.latent["var"], res_rasch.latent["var"], atol=1e-8)
    np.testing.assert_allclose(res_pcm.latent["var_se"], res_rasch.latent["var_se"], rtol=1e-7)


def test_gpcm_and_pcm_starting_values() -> None:
    rng = np.random.default_rng(7)
    item_bin = rng.integers(0, 2, size=200).astype(float)
    rest = rng.integers(0, 10, size=200).astype(float)
    np.testing.assert_allclose(
        GPCM(2).start_values(item_bin, rest), TwoPL().start_values(item_bin, rest), rtol=1e-14
    )
    np.testing.assert_allclose(
        PCM(2).start_values(item_bin, rest), Rasch().start_values(item_bin, rest), rtol=1e-14
    )

    item_poly = rng.integers(0, 4, size=200).astype(float)
    starts_gpcm = GPCM(4).start_values(item_poly, rest)
    starts_pcm = PCM(4).start_values(item_poly, rest)
    assert starts_gpcm.shape == (4,) and np.isfinite(starts_gpcm).all()
    assert starts_pcm.shape == (4,) and starts_pcm[0] == 1.0 and np.isfinite(starts_pcm).all()


# ---- Latent default rule & identification --------------------------------------------------


def test_pcm_default_latent_rule() -> None:
    assert Model.uniform(PCM(4), items=["q1", "q2"]).resolved_latent() == Normal(free_var=True)
    assert Model.from_items({"q1": Rasch(), "q2": PCM(4)}).resolved_latent() == Normal(
        free_var=True
    )
    assert Model.from_items({"q1": PCM(4), "q2": GPCM(4)}).resolved_latent() == Normal.standard()
    assert Model.from_items({"q1": PCM(4), "q2": TwoPL()}).resolved_latent() == Normal.standard()
    assert Model.uniform(GPCM(4), items=["q1", "q2"]).resolved_latent() == Normal.standard()
    assert (
        Model.uniform(PCM(4), items=["q1", "q2"]).with_latent(Normal.standard()).resolved_latent()
        == Normal.standard()
    )


def test_gpcm_and_pcm_identification() -> None:
    a = np.ones(4)
    d = np.array([[0.8, 0.5, -0.7]] * 4)
    df = _sim_gpcm(a, d, n=400, seed=202)
    with pytest.raises(IdentificationError, match="variance"):
        fit(Model.uniform(GPCM(4), items=df.columns).with_latent(Normal(free_var=True)), df)
    with pytest.raises(IdentificationError, match="mean"):
        fit(Model.uniform(PCM(4), items=df.columns).with_latent(Normal.free()), df)


def test_gpcm_unobserved_category_is_rejected() -> None:
    df = pd.DataFrame({"x": [0, 1, 2, 0, 1, 2], "y": [0, 1, 1, 0, 2, 2]}, dtype=float)
    df.loc[df["x"] == 2, "x"] = 1
    with pytest.raises(SpecificationError, match=r"categories \[2\]"):
        fit(Model.uniform(GPCM(3), items=["x", "y"]), df, se=None)
    with pytest.raises(SpecificationError, match=r"categories \[2\]"):
        fit(Model.uniform(PCM(3), items=["x", "y"]), df, se=None)


# ---- Step difficulties, mixed models, and fix_from -----------------------------------------


@pytest.mark.slow
def test_mixed_gpcm_pcm_twopl_grm_model() -> None:
    """Mix GPCM, PCM, 2PL, Rasch, and GRM items (and distinct n_categories) in one model."""
    rng = np.random.default_rng(303)
    n = 1200
    theta = rng.normal(0.0, 1.0, size=n)

    def sim_2pl(a: float, d: float) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-(a * theta + d)))
        return (rng.random(n) < p).astype(float)

    def sim_grm(a: float, ds: list[float]) -> np.ndarray:
        cum = 1.0 / (1.0 + np.exp(-(a * theta[:, None] + np.array(ds)[None, :])))
        probs = np.column_stack([1.0 - cum[:, 0], -np.diff(cum, axis=1), cum[:, -1]])
        cdf = np.cumsum(probs, axis=1)
        return np.asarray((rng.random(n)[:, None] >= cdf).sum(axis=1), dtype=float)

    def sim_gpcm_item(a: float, ds: list[float]) -> np.ndarray:
        k = np.arange(len(ds) + 1, dtype=float)
        logits = np.outer(theta * a, k) + np.array([0.0, *ds])[None, :]
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        cdf = np.cumsum(probs, axis=1)
        return np.asarray((rng.random(n)[:, None] >= cdf).sum(axis=1), dtype=float)

    df = pd.DataFrame(
        {
            "b1": sim_2pl(1.2, 0.3),
            "b2": sim_2pl(1.0, -0.4),
            "g1": sim_grm(1.3, [1.5, 0.2, -1.2]),
            "p1": sim_gpcm_item(1.1, [1.2, 1.0, -0.4]),
            "p2": sim_gpcm_item(1.0, [0.9, 0.8, -0.5]),
            "p3": sim_gpcm_item(1.4, [0.7, -0.3]),
        }
    )
    item_models: dict[str, ItemModel] = {
        "b1": TwoPL(),
        "b2": Rasch(),
        "g1": GRM(4),
        "p1": GPCM(4),
        "p2": PCM(4),
        "p3": GPCM(3),
    }
    spec = Model.from_items(item_models)
    with warnings.catch_warnings():
        warnings.simplefilter("error", HeywoodWarning)
        res = fit(spec, df, method=BAEM(tol=1e-7, max_iter=5000), se="oakes")

    assert res.converged
    items = res.items.set_index("item")
    assert list(items["model"]) == ["2PL", "Rasch", "GRM", "GPCM", "PCM", "GPCM"]

    # 2PL / Rasch display unchanged: b = -d / a
    for name in ("b1", "b2"):
        assert float(items["b"].loc[name]) == pytest.approx(
            -float(items["d"].loc[name]) / float(items["a"].loc[name])
        )
        assert items.loc[name, ["b1", "b2", "b3"]].isna().all()

    # GRM display unchanged: bk = -dk / a
    for k in (1, 2, 3):
        assert float(items[f"b{k}"].loc["g1"]) == pytest.approx(
            -float(items[f"d{k}"].loc["g1"]) / float(items["a"].loc["g1"])
        )

    # GPCM / PCM step difficulties: bk = -(dk - d{k-1}) / a with d0 = 0
    for name, n_steps in (("p1", 3), ("p2", 3), ("p3", 2)):
        a_hat = float(items["a"].loc[name])
        prev_d = 0.0
        for k in range(1, n_steps + 1):
            dk = float(items[f"d{k}"].loc[name])
            assert float(items[f"b{k}"].loc[name]) == pytest.approx(-(dk - prev_d) / a_hat)
            prev_d = dk
    assert float(items["a"].loc["p2"]) == 1.0 and np.isnan(float(items["a_se"].loc["p2"]))
    assert np.isnan(float(items["b3"].loc["p3"]))

    scores = res.score(df.iloc[:10])
    assert len(scores) == 10 and scores["theta"].notna().all()


def test_fix_from_gpcm_and_pcm() -> None:
    """fix_from converts GPCM/PCM IRT step difficulties via d_k = -a * sum_{c=1}^k b_c."""
    bank_irt = pd.DataFrame(
        [
            {"item": "q01", "a": 1.5, "b1": -1.0, "b2": 0.2, "b3": 1.4},
            {"item": "q02", "b1": -0.8, "b2": 0.1, "b3": 0.9},  # PCM may omit a
        ]
    )
    spec = Model.from_items({"q01": GPCM(4), "q02": PCM(4), "q03": GPCM(4)}).fix_from(
        bank_irt, parameterization="irt"
    )
    fixed = {(i, p): v for i, p, v in spec.fixed}
    assert fixed == {
        ("q01", "a"): 1.5,
        ("q01", "d1"): pytest.approx(-1.5 * (-1.0)),
        ("q01", "d2"): pytest.approx(-1.5 * (-1.0 + 0.2)),
        ("q01", "d3"): pytest.approx(-1.5 * (-1.0 + 0.2 + 1.4)),
        ("q02", "d1"): pytest.approx(0.8),
        ("q02", "d2"): pytest.approx(-(-0.8 + 0.1)),
        ("q02", "d3"): pytest.approx(-(-0.8 + 0.1 + 0.9)),
    }

    # GPCM requires 'a' in bank
    bank_no_a = pd.DataFrame({"item": ["q01"], "b1": [-0.5], "b2": [0.5]})
    with pytest.raises(SpecificationError, match=r"'q01'.*'a'"):
        Model.uniform(GPCM(3), items=["q01", "q02"]).fix_from(bank_no_a, parameterization="irt")

    # Round-trip from FitResult.items -> fix_from(parameterization="irt")
    a = np.array([1.2, 1.0, 0.9, 1.3])
    d = np.array([[1.1, 0.8, -0.6], [0.7, 0.5, -0.9], [0.9, 0.2, -0.8], [1.3, 1.1, -0.3]])
    df = _sim_gpcm(a, d, n=600, seed=404)
    res = fit(Model.uniform(GPCM(4), items=df.columns), df, method=BAEM(tol=1e-7), se=None)
    rt = Model.uniform(GPCM(4), items=df.columns).fix_from(res.items, parameterization="irt")
    rt_fixed = {(i, p): v for i, p, v in rt.fixed}
    items_df = res.items.set_index("item")
    for name in df.columns:
        assert rt_fixed[(name, "a")] == pytest.approx(float(items_df["a"].loc[name]))
        for k in (1, 2, 3):
            assert rt_fixed[(name, f"d{k}")] == pytest.approx(float(items_df[f"d{k}"].loc[name]))


def test_fix_from_irt_uses_prior_fixed_slope() -> None:
    """fix_from(parameterization='irt') uses a slope fixed via .fix() when 'a' is omitted."""
    # Finding scenario 1: PCM / Rasch with overridden slope via .fix() and bank omitting 'a'.
    bank_steps = pd.DataFrame([{"item": "q1", "b1": -0.5, "b2": 0.5}])
    spec_pcm = (
        Model.uniform(PCM(3), items=["q1", "q2"])
        .fix(item="q1", a=1.5)
        .fix_from(bank_steps, parameterization="irt")
    )
    assert {(i, p): v for i, p, v in spec_pcm.fixed} == {
        ("q1", "a"): 1.5,
        ("q1", "d1"): pytest.approx(-1.5 * (-0.5)),
        ("q1", "d2"): pytest.approx(-1.5 * (-0.5 + 0.5)),
    }

    bank_dich = pd.DataFrame([{"item": "q1", "b": 0.4}])
    spec_rasch = (
        Model.uniform(Rasch(), items=["q1", "q2"])
        .fix(item="q1", a=1.5)
        .fix_from(bank_dich, parameterization="irt")
    )
    assert {(i, p): v for i, p, v in spec_rasch.fixed} == {
        ("q1", "a"): 1.5,
        ("q1", "d"): pytest.approx(-1.5 * 0.4),
    }

    # Finding scenario 2: GPCM with slope already fixed via .fix() and bank omitting 'a'.
    spec_gpcm = (
        Model.uniform(GPCM(3), items=["q1", "q2"])
        .fix(item="q1", a=1.5)
        .fix_from(bank_steps, parameterization="irt")
    )
    assert {(i, p): v for i, p, v in spec_gpcm.fixed} == {
        ("q1", "a"): 1.5,
        ("q1", "d1"): pytest.approx(-1.5 * (-0.5)),
        ("q1", "d2"): pytest.approx(-1.5 * (-0.5 + 0.5)),
    }

    # Every other item model that converts b -> d (2PL, 3PL, 4PL, GRM), including NaN 'a'.
    spec_2pl = (
        Model.uniform(TwoPL(), items=["q1", "q2"])
        .fix(item="q1", a=1.5)
        .fix_from(pd.DataFrame([{"item": "q1", "a": np.nan, "b": 0.4}]), parameterization="irt")
    )
    assert {(i, p): v for i, p, v in spec_2pl.fixed} == {
        ("q1", "a"): 1.5,
        ("q1", "d"): pytest.approx(-1.5 * 0.4),
    }

    spec_3pl = (
        Model.uniform(ThreePL(), items=["q1", "q2"])
        .fix(item="q1", a=1.5)
        .fix_from(pd.DataFrame([{"item": "q1", "b": 0.4, "g": 0.2}]), parameterization="irt")
    )
    assert {(i, p): v for i, p, v in spec_3pl.fixed} == {
        ("q1", "a"): 1.5,
        ("q1", "d"): pytest.approx(-1.5 * 0.4),
        ("q1", "g"): 0.2,
    }

    spec_4pl = (
        Model.uniform(FourPL(), items=["q1", "q2"])
        .fix(item="q1", a=1.5, u=0.9)
        .fix_from(pd.DataFrame([{"item": "q1", "b": 0.4, "g": 0.15}]), parameterization="irt")
    )
    assert {(i, p): v for i, p, v in spec_4pl.fixed} == {
        ("q1", "a"): 1.5,
        ("q1", "u"): 0.9,
        ("q1", "d"): pytest.approx(-1.5 * 0.4),
        ("q1", "g"): 0.15,
    }

    spec_grm = (
        Model.uniform(GRM(3), items=["q1", "q2"])
        .fix(item="q1", a=1.5)
        .fix_from(pd.DataFrame([{"item": "q1", "b1": -0.5, "b2": 0.8}]), parameterization="irt")
    )
    assert {(i, p): v for i, p, v in spec_grm.fixed} == {
        ("q1", "a"): 1.5,
        ("q1", "d1"): pytest.approx(-1.5 * (-0.5)),
        ("q1", "d2"): pytest.approx(-1.5 * 0.8),
    }

    # Explicit 'a' in the bank overrides a previously fixed slope and is used for b -> d.
    spec_override = (
        Model.uniform(GPCM(3), items=["q1", "q2"])
        .fix(item="q1", a=1.5)
        .fix_from(
            pd.DataFrame([{"item": "q1", "a": 2.0, "b1": -0.5, "b2": 0.5}]),
            parameterization="irt",
        )
    )
    assert {(i, p): v for i, p, v in spec_override.fixed} == {
        ("q1", "a"): 2.0,
        ("q1", "d1"): pytest.approx(-2.0 * (-0.5)),
        ("q1", "d2"): pytest.approx(-2.0 * (-0.5 + 0.5)),
    }

    # Without prior .fix(a=...), free-slope models still require 'a' in the bank.
    for item_model, bank in (
        (TwoPL(), bank_dich),
        (ThreePL(), pd.DataFrame([{"item": "q1", "b": 0.4, "g": 0.2}])),
        (FourPL(), pd.DataFrame([{"item": "q1", "b": 0.4, "g": 0.15, "u": 0.9}])),
        (GRM(3), bank_steps),
        (GPCM(3), bank_steps),
    ):
        with pytest.raises(SpecificationError, match=r"'q1'.*'a'"):
            Model.uniform(item_model, items=["q1", "q2"]).fix_from(bank, parameterization="irt")
