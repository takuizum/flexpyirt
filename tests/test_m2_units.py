"""Unit tests for M2 features: GRM kernel, groups, equality, FIPC, identification, missing data."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from flexpyirt import BAEM, GRM, Model, Normal, TwoPL, fit
from flexpyirt.errors import FlexpyirtWarning, IdentificationError, SpecificationError
from flexpyirt.items import GradedKernel


def _sim(n: int, seed: int, a: np.ndarray, d: np.ndarray, mu: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    theta = rng.normal(mu, 1.0, size=(n, 1))
    y = (rng.random((n, len(a))) < 1 / (1 + np.exp(-(theta * a + d)))).astype(float)
    return pd.DataFrame(y, columns=[f"i{j}" for j in range(len(a))])


A = np.array([1.0, 1.3, 0.8, 1.1, 0.9, 1.2])
D = np.array([0.5, -0.3, 1.0, 0.0, -0.8, 0.4])


@pytest.fixture(scope="module")
def two_groups() -> tuple[pd.DataFrame, np.ndarray]:
    ref = _sim(600, 1, A, D)
    foc = _sim(600, 2, A, D, mu=0.5)
    return pd.concat([ref, foc], ignore_index=True), np.repeat(["ref", "focal"], 600)


# ---- GRM kernel ----------------------------------------------------------------------------


@settings(deadline=None)
@given(
    a=st.floats(0.2, 3.0),
    d1=st.floats(-3, 4),
    gaps=st.lists(st.floats(0.05, 3.0), min_size=2, max_size=2),
)
def test_grm_probabilities(a: float, d1: float, gaps: list[float]) -> None:
    d = np.array([d1, d1 - gaps[0], d1 - gaps[0] - gaps[1]])
    theta = jnp.linspace(-6, 6, 25)
    lp = GradedKernel(4).log_prob(theta, jnp.array([a, *d]))
    p = np.exp(np.asarray(lp))
    np.testing.assert_allclose(p.sum(axis=1), 1.0, rtol=1e-10)
    cum = 1 / (1 + np.exp(-(a * np.asarray(theta)[:, None] + d)))
    np.testing.assert_allclose(p[:, 1:].sum(axis=1), cum[:, 0], rtol=1e-8, atol=1e-14)
    grad = jax.grad(lambda x: GradedKernel(4).log_prob(theta, x).sum())(jnp.array([a, *d]))
    assert np.isfinite(np.asarray(grad)).all()


def test_grm_disordered_intercepts_give_nan() -> None:
    lp = GradedKernel(3).log_prob(jnp.array([0.0]), jnp.array([1.0, -1.0, 1.0]))
    assert np.isnan(np.asarray(lp)).any()


def test_grm_two_categories_equals_2pl() -> None:
    theta = jnp.linspace(-4, 4, 9)
    params = jnp.array([1.3, 0.4])
    np.testing.assert_allclose(
        GradedKernel(2).log_prob(theta, params), TwoPL().kernel.log_prob(theta, params), rtol=1e-12
    )


def test_grm_unobserved_category_is_rejected() -> None:
    df = pd.DataFrame({"x": [0, 1, 2, 0, 1, 2], "y": [0, 1, 1, 0, 2, 2]}, dtype=float)
    df.loc[df["x"] == 2, "x"] = 1  # category 2 never observed for x
    with pytest.raises(SpecificationError, match=r"categories \[2\]"):
        fit(Model.uniform(GRM(n_categories=3), items=["x", "y"]), df)


# ---- groups and equality -------------------------------------------------------------------


def test_equal_across_groups_shares_free_ids(two_groups: tuple[pd.DataFrame, np.ndarray]) -> None:
    df, grp = two_groups
    spec = (
        Model.uniform(TwoPL(), items=df.columns)
        .with_groups(ref=Normal.standard(), focal=Normal.free())
        .equal_across_groups(items=["i0", "i1"])
    )
    t = spec.parameter_table(df, group=grp).set_index(["group", "item", "param"])
    fid = t["free_id"].to_dict()
    assert fid[("ref", "i0", "a")] == fid[("focal", "i0", "a")]
    assert fid[("ref", "i5", "d")] != fid[("focal", "i5", "d")]


def test_two_identical_groups_reproduce_single_group(
    two_groups: tuple[pd.DataFrame, np.ndarray],
) -> None:
    df, grp = two_groups
    single = fit(Model.uniform(TwoPL(), items=df.columns), df, se=None, method=BAEM(tol=1e-9))
    multi = fit(
        Model.uniform(TwoPL(), items=df.columns)
        .with_groups(g1=Normal.standard(), g2=Normal.standard())
        .equal_across_groups(items=None),
        df,
        group=np.where(grp == "ref", "g1", "g2"),
        se=None,
        method=BAEM(tol=1e-9),
    )
    assert multi.loglik == pytest.approx(single.loglik, rel=1e-12)
    a = single.items.set_index("item")[["a", "d"]]
    b = multi.items[multi.items["group"] == "g2"].set_index("item")[["a", "d"]]
    np.testing.assert_allclose(a, b, atol=1e-7)


def test_focal_free_without_anchors_is_not_identified(
    two_groups: tuple[pd.DataFrame, np.ndarray],
) -> None:
    df, grp = two_groups
    spec = Model.uniform(TwoPL(), items=df.columns).with_groups(
        ref=Normal.standard(), focal=Normal.free()
    )
    with pytest.raises(IdentificationError, match="group 'focal'"):
        fit(spec, df, group=grp)


def test_intercept_only_anchors_do_not_identify_variance(
    two_groups: tuple[pd.DataFrame, np.ndarray],
) -> None:
    df, grp = two_groups
    spec = (
        Model.uniform(TwoPL(), items=df.columns)
        .with_groups(ref=Normal.standard(), focal=Normal.free())
        .equal_across_groups(items=["i0", "i1"], params=["d"])
    )
    with pytest.raises(IdentificationError, match="variance of group 'focal'"):
        fit(spec, df, group=grp)


def test_single_anchor_warns(two_groups: tuple[pd.DataFrame, np.ndarray]) -> None:
    df, grp = two_groups
    spec = (
        Model.uniform(TwoPL(), items=df.columns)
        .with_groups(ref=Normal.standard(), focal=Normal.free())
        .equal_across_groups(items=["i0"])
    )
    with pytest.warns(FlexpyirtWarning, match="single anchor"):
        spec.parameter_table(df, group=grp)


def test_group_label_errors(two_groups: tuple[pd.DataFrame, np.ndarray]) -> None:
    df, grp = two_groups
    spec = (
        Model.uniform(TwoPL(), items=df.columns)
        .with_groups(ref=Normal.standard(), focal=Normal.free())
        .equal_across_groups(items=["i0", "i1", "i2"])
    )
    with pytest.raises(SpecificationError, match="pass group="):
        fit(spec, df)
    with pytest.raises(SpecificationError, match="not declared"):
        fit(spec, df, group=np.where(grp == "ref", "other", grp))
    with pytest.raises(SpecificationError, match="absent"):
        fit(spec, df, group=np.full(len(df), "ref"))


def test_with_groups_validation() -> None:
    spec = Model.uniform(TwoPL(), items=["x", "y"])
    with pytest.raises(SpecificationError, match="at least two"):
        spec.with_groups(only=Normal.standard())
    with pytest.raises(SpecificationError, match="with_groups"):
        spec.with_groups(a=Normal.standard(), b=Normal.free()).with_latent(Normal.free())


# ---- FIPC ----------------------------------------------------------------------------------


def test_fix_from_irt_parameterization() -> None:
    bank = pd.DataFrame({"item": ["x", "y"], "a": [1.5, 0.8], "b": [0.4, -1.0]})
    spec = Model.uniform(TwoPL(), items=["x", "y", "z"]).fix_from(bank, parameterization="irt")
    fixed = {(i, p): v for i, p, v in spec.fixed}
    assert fixed[("x", "d")] == pytest.approx(-1.5 * 0.4)
    assert fixed[("y", "a")] == 0.8
    assert ("z", "a") not in fixed


def test_fix_from_grm_irt() -> None:
    bank = pd.DataFrame({"item": ["g"], "a": [2.0], "b1": [-1.0], "b2": [0.5]})
    spec = Model.uniform(GRM(n_categories=3), items=["g", "h"]).fix_from(
        bank, parameterization="irt"
    )
    fixed = {(i, p): v for i, p, v in spec.fixed}
    assert fixed[("g", "d1")] == pytest.approx(2.0) and fixed[("g", "d2")] == pytest.approx(-1.0)


def test_fipc_without_fixed_items_is_not_identified() -> None:
    df = _sim(500, 3, A, D)
    with pytest.raises(IdentificationError):
        fit(Model.uniform(TwoPL(), items=df.columns).with_latent(Normal.free()), df)


# ---- missing data --------------------------------------------------------------------------


def test_missing_data_is_ignored_in_likelihood() -> None:
    """Data with 20% MCAR holes fit and score; a fully missing item is rejected."""
    df = _sim(800, 4, A, D)
    rng = np.random.default_rng(5)
    holes = df.mask(rng.random(df.shape) < 0.2)
    res = fit(Model.uniform(TwoPL(), items=df.columns), holes, se=None)
    assert res.converged
    scores = res.score(holes)
    assert scores["theta"].notna().all()
    holes["i0"] = np.nan
    with pytest.raises(SpecificationError, match="never observed"):
        fit(Model.uniform(TwoPL(), items=df.columns), holes)


def test_unanswered_rows_get_the_prior() -> None:
    df = _sim(500, 6, A, D)
    res = fit(Model.uniform(TwoPL(), items=df.columns), df, se=None)
    empty = pd.DataFrame(np.full((1, len(A)), np.nan), columns=df.columns)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = res.score(empty)
    assert s["theta"].iloc[0] == pytest.approx(0.0, abs=1e-10)
    assert s["se"].iloc[0] == pytest.approx(1.0, abs=1e-6)
