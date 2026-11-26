"""Davidian curves and normal mixtures per group (DESIGN.md §27, #75).

Internal checks; the comparison with mirt ``multipleGroup(customGroup = ...)`` and an
independent R likelihood is V56 (``test_mirt_v56_multigroup_shapes.py``).
"""

from __future__ import annotations

import json
import warnings
from functools import cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from flexpyirt import (
    BAEM,
    DavidianCurve,
    FitResult,
    Model,
    Normal,
    NormalMixture,
    Quadrature,
    TwoPL,
    fit,
    simulate_responses,
)
from flexpyirt._compile import LATENT_ITEM, compile_problem, prepare
from flexpyirt._core import _latent_log_weights
from flexpyirt.errors import HeywoodWarning, SpecificationError
from flexpyirt.linking import _fitted_grid

QUAD = Quadrature.equispaced(41, bounds=(-6, 6))
KINDS = ["davidian", pytest.param("mixture", marks=pytest.mark.slow)]
ITEMS = [f"q{j:02d}" for j in range(12)]


@cache
def _data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(75)
    slopes = rng.uniform(1.3, 2.3, len(ITEMS))
    intercepts = rng.normal(0.0, 0.8, len(ITEMS))
    ref = np.concatenate([rng.normal(-1.1, 0.5, 500), rng.normal(1.1, 0.5, 450)])
    foc = np.concatenate([rng.normal(-0.5, 0.5, 300), rng.normal(1.5, 0.5, 500)])
    blocks = []
    for theta in (ref, foc):
        prob = 1.0 / (1.0 + np.exp(-(slopes * theta[:, None] + intercepts)))
        blocks.append((rng.uniform(size=prob.shape) < prob).astype(float))
    df = pd.DataFrame(np.vstack(blocks), columns=ITEMS)
    return df, np.repeat(["ref", "foc"], [ref.size, foc.size])


def _davidian_spec(degree: int = 2) -> Model:
    return (
        Model.uniform(TwoPL(), items=ITEMS)
        .with_groups(
            ref=DavidianCurve(degree),
            foc=DavidianCurve(degree, free_mean=True, free_var=True),
        )
        .equal_across_groups(items=None)
    )


def _mixture_spec() -> Model:
    return (
        Model.uniform(TwoPL(), items=ITEMS)
        .with_groups(
            ref=NormalMixture(2, means=(-1.0, 1.0), variances=(0.25, 0.25)),
            foc=NormalMixture(
                2, free_mean=True, free_var=True, means=(-1.0, 1.0), variances=(0.25, 0.25)
            ),
        )
        .equal_across_groups(items=None)
    )


@cache
def _fit(kind: str, se: str = "oakes") -> FitResult:
    df, group = _data()
    spec = _davidian_spec() if kind == "davidian" else _mixture_spec()
    return fit(
        spec,
        df,
        group=group,
        quadrature=QUAD,
        method=BAEM(tol=1e-8, max_iter=5000, squarem=True),
        se=se,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("kind", KINDS)
def test_fit_latent_rows_per_group(kind: str) -> None:
    res = _fit(kind)
    assert res.converged
    latent = res.latent.set_index("group")
    assert list(latent.index) == ["ref", "foc"]
    moments = latent[["mean", "var", "mean_se"]].to_numpy(dtype=float)  # rows: ref, foc
    assert moments[0, 0] == 0.0 and moments[0, 1] == 1.0
    assert np.isnan(moments[0, 2]) and np.isfinite(moments[1, 2])
    prefixes = ("phi_", "rmean_", "rvar_", "logit_")
    shape = [str(c) for c in latent.columns if str(c).startswith(prefixes)]
    shape = [c for c in shape if not c.endswith("_se")]
    table = latent[shape].to_numpy(dtype=float)
    assert shape and np.isfinite(table).all()
    assert table[0, 0] != table[1, 0]  # per-group shapes


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["davidian", "mixture"])
def test_oakes_equals_hessian(kind: str) -> None:
    se_o = _fit(kind).parameters["se"].to_numpy()
    se_h = _fit(kind, "hessian").parameters["se"].to_numpy()
    np.testing.assert_allclose(se_o, se_h, rtol=1e-6)


def test_davidian_with_zero_angles_is_the_normal_model() -> None:
    """Fixing every ``phi_*`` at 0 gives exactly the two-group Normal model (DESIGN.md §27)."""
    df, group = _data()
    spec = _davidian_spec()
    table = spec.parameter_table(df, group=group)
    phi = (table["item"] == LATENT_ITEM) & table["param"].str.startswith("phi_")
    table.loc[phi, "value"] = 0.0
    table.loc[phi, "free"] = False
    method = BAEM(tol=1e-9, max_iter=5000)
    res_d = fit(spec, df, group=group, table=table, quadrature=QUAD, method=method, se=None)
    normal = (
        Model.uniform(TwoPL(), items=ITEMS)
        .with_groups(ref=Normal.standard(), foc=Normal.free())
        .equal_across_groups(items=None)
    )
    res_n = fit(normal, df, group=group, quadrature=QUAD, method=method, se=None)
    assert abs(res_d.loglik - res_n.loglik) < 1e-8
    lat_d = res_d.latent.set_index("group")
    lat_n = res_n.latent.set_index("group")
    np.testing.assert_allclose(lat_d[["mean", "var"]], lat_n[["mean", "var"]], atol=1e-6)
    np.testing.assert_allclose(res_d.items[["a", "d"]], res_n.items[["a", "d"]], atol=1e-6)


@pytest.mark.parametrize("kind", KINDS)
def test_scoring_item_fit_and_linking_use_each_group(kind: str) -> None:
    res = _fit(kind)
    df, group = _data()
    prep = prepare(res._model, df, group=group, item_names=None, table=res.parameters)
    prob = compile_problem(res._model, prep.y, prep.group, prep.table, QUAD).problem
    weights = np.exp(np.asarray(_latent_log_weights(prob, jnp.asarray(res._eta))))
    for g, name in enumerate(["ref", "foc"]):
        nodes, w = _fitted_grid(res, name, g)  # link()'s default weights
        np.testing.assert_allclose(w, weights[g], atol=1e-14)
        expected_eap = float(nodes @ w)  # an empty response pattern has the prior mean
        empty = pd.DataFrame([[np.nan] * len(ITEMS)], columns=ITEMS)
        assert abs(res.score(empty, group=[name])["theta"].iloc[0] - expected_eap) < 1e-10
    fit_table = res.item_fit(df, group=group)
    assert set(fit_table["group"]) == {"ref", "foc"}
    assert np.all(np.isfinite(fit_table["s_x2"]))
    # Degrees of freedom subtract the group's free latent parameters (shape, plus the focal
    # group's mean and variance).
    df_by_group = fit_table.groupby("group")["df"].sum()
    n_cells_equal = fit_table.pivot(index="item", columns="group", values="df")
    assert n_cells_equal.shape == (len(ITEMS), 2) and df_by_group.notna().all()


def test_simulation_and_json_round_trip(tmp_path: Path) -> None:
    res = _fit("davidian")
    sim = simulate_responses(res, n_obs={"ref": 4000, "foc": 4000}, key=jax.random.key(1))
    theta = pd.Series(np.asarray(sim.theta).reshape(-1)).groupby(np.asarray(sim.group))
    latent = res.latent.set_index("group")
    for name in ("ref", "foc"):
        assert abs(theta.mean()[name] - latent.loc[name, "mean"]) < 0.06
    path = tmp_path / "fit.json"
    res.to_json(path)
    assert json.loads(path.read_text())["schema_version"] == 8
    back = FitResult.from_json(path)
    assert back._model == res._model
    pd.testing.assert_frame_equal(back.latent, res.latent)


@pytest.mark.slow
def test_mixture_degenerate_warning_names_the_group() -> None:
    df, group = _data()
    spec = (
        Model.uniform(TwoPL(), items=ITEMS)
        .with_groups(ref=NormalMixture(2), foc=NormalMixture(2, free_mean=True, free_var=True))
        .equal_across_groups(items=None)
    )
    table = spec.parameter_table(df, group=group)
    weight = (table["group"] == "foc") & (table["param"] == "logit_2")
    table.loc[weight, "value"] = -14.0
    table.loc[weight, "free"] = False  # a focal component of weight ~1e-6
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit(spec, df, group=group, table=table, quadrature=QUAD, se=None)
    messages = [str(w.message) for w in caught if issubclass(w.category, HeywoodWarning)]
    assert any("group 'foc' component 2" in m for m in messages)


def test_fipc_frees_every_group() -> None:
    """FIPC with Davidian groups: items fixed at the anchored fit's values, every latent free."""
    df, group = _data()
    anchored = _fit("davidian")
    bank = anchored.items.drop_duplicates("item")[["item", "a", "d"]]
    spec = (
        Model.uniform(TwoPL(), items=ITEMS)
        .fix_from(bank)
        .with_groups(
            ref=DavidianCurve(2, free_mean=True, free_var=True),
            foc=DavidianCurve(2, free_mean=True, free_var=True),
        )
    )
    table = spec.parameter_table(df, group=group)
    assert not table.loc[table["item"] != LATENT_ITEM, "free"].any()
    assert table.loc[table["item"] == LATENT_ITEM, "free"].all()
    # Start at the anchored latent estimates (Davidian likelihoods are multimodal in the shape):
    # that point is feasible for FIPC with exactly the anchored log-likelihood, so FIPC can only
    # improve on it, and with the reference group at N(0, 1) it stays there.
    estimate = anchored.parameters.set_index(["group", "item", "param"])["estimate"]
    latent_rows = table["item"] == LATENT_ITEM
    table.loc[latent_rows, "value"] = [
        estimate[(g, LATENT_ITEM, p)]
        for g, p in zip(
            table.loc[latent_rows, "group"], table.loc[latent_rows, "param"], strict=True
        )
    ]
    res = fit(spec, df, group=group, table=table, quadrature=QUAD, method=BAEM(tol=1e-9), se=None)
    assert res.converged
    assert res.loglik >= anchored.loglik - 1e-8
    latent = res.latent.set_index("group")
    ref = anchored.latent.set_index("group")
    np.testing.assert_allclose(latent[["mean", "var"]], ref[["mean", "var"]], atol=1e-3)


def test_unsupported_with_these_families() -> None:
    res = _fit("davidian")
    df, group = _data()
    with pytest.raises(SpecificationError, match="MAP"):
        res.score(df.head(2), group=group[:2], method="MAP")
    with pytest.raises(SpecificationError, match="m2"):
        res.m2(df, group=group)
