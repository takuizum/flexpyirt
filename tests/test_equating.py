"""Unit tests of IRT equating and scale tables (DESIGN.md §23; R references in test_r_equating).

Fits have every parameter fixed, so they only carry item parameters and latent distributions.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import flexpyequate
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from flexpyirt import (
    GPCM,
    GRM,
    FitResult,
    FourPL,
    IRTEquatingResult,
    LinkResult,
    Model,
    Nominal,
    Normal,
    Quadrature,
    ThreePL,
    TwoPL,
    expected_score,
    fit,
    irt_equate,
    item_information,
    scale_table,
    score_distribution,
    summed_score_table,
)
from flexpyirt._equating import (
    Form,
    FormFamily,
    equipercentile,
    percentile_rank,
    solve_true_score,
    test_characteristic,
)
from flexpyirt.equating import _FormSpec, _true_score_equate
from flexpyirt.errors import SpecificationError
from flexpyirt.items import FourPLKernel, ItemModel

ITEMS: dict[str, tuple[ItemModel, dict[str, float]]] = {
    "p1": (TwoPL(), {"a": 1.2, "d": 0.4}),
    "p2": (TwoPL(), {"a": 0.8, "d": -0.6}),
    "p3": (TwoPL(), {"a": 1.6, "d": 0.1}),
    "t1": (ThreePL(), {"a": 1.4, "d": -0.3, "g": 0.2, "u": 1.0}),
    "t2": (ThreePL(), {"a": 1.0, "d": 0.5, "g": 0.15, "u": 1.0}),
    "f1": (FourPL(), {"a": 1.3, "d": 0.2, "g": 0.1, "u": 0.9}),
    "f2": (FourPL(), {"a": 0.9, "d": -0.4, "g": 0.05, "u": 0.85}),
    "g1": (GRM(n_categories=4), {"a": 1.5, "d1": 1.5, "d2": 0.0, "d3": -1.4}),
    "g2": (GRM(n_categories=3), {"a": 1.1, "d1": 0.7, "d2": -0.9}),
    "c1": (GPCM(n_categories=3), {"a": 0.9, "d1": 0.5, "d2": 0.2}),
    "n1": (Nominal(n_categories=3), {"a": 1.0, "ak1": 3.0, "d1": -0.5, "d2": 1.0}),
    "neg": (TwoPL(), {"a": -0.7, "d": 0.2}),
}
MONOTONE = [i for i in ITEMS if i not in ("n1", "neg")]
QUAD = Quadrature.equispaced(61, bounds=(-6.0, 6.0))


def _fixed_fit(
    names: list[str], groups: dict[str, dict[str, float]] | None = None, shift: float = 0.0
) -> FitResult:
    """A fit with every parameter fixed: ``ITEMS`` (intercepts shifted by ``shift`` in group 2)."""
    spec = Model.from_items({i: ITEMS[i][0] for i in names})
    rng = np.random.default_rng(7)
    n = 240
    data = pd.DataFrame(
        {i: rng.permutation(np.arange(n) % ITEMS[i][0].kernel.n_categories) for i in names}
    )
    group = None
    if groups is not None:
        spec = spec.with_groups(
            **{g: Normal(mean=v["mean"], var=v["var"]) for g, v in groups.items()}
        )
        group = np.repeat(list(groups), n // len(groups))
    table = spec.parameter_table(data, group=group)
    values = []
    for g, item, param in zip(table["group"], table["item"], table["param"], strict=True):
        if item == "__latent__":
            latent = {"mean": 0.0, "var": 1.0} if groups is None else groups[str(g)]
            values.append(latent[str(param)])
        else:
            value = ITEMS[item][1][param]
            if groups is not None and g == list(groups)[1] and param.startswith("d"):
                value += shift
            values.append(value)
    table["value"] = np.asarray(values, dtype=np.float64)
    table["free"] = False
    return fit(spec, data, group=group, table=table, quadrature=QUAD, se=None)


@cache
def _single() -> FitResult:
    return _fixed_fit(list(ITEMS))


@cache
def _two_groups() -> FitResult:
    return _fixed_fit(
        ["p1", "p2", "p3", "t1", "g1"],
        groups={"ref": {"mean": 0.0, "var": 1.0}, "focal": {"mean": 0.5, "var": 1.44}},
        shift=0.3,
    )


# ---- equipercentile step ---------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("zeros", [False, True])
def test_equipercentile_matches_flexpyequate(seed: int, zeros: bool) -> None:
    rng = np.random.default_rng(seed)
    f = rng.dirichlet(np.ones(15))
    g = rng.dirichlet(np.ones(21))
    if zeros:
        f[[0, 7]] = 0.0
        g[[0, 4, 5, 20]] = 0.0
    x = flexpyequate.ScoreDistribution.from_frequencies(scores=range(15), frequencies=f)
    y = flexpyequate.ScoreDistribution.from_frequencies(scores=range(21), frequencies=g)
    want = flexpyequate.equate(x, y, method="equipercentile").table["equated"].to_numpy()
    np.testing.assert_allclose(equipercentile(f, g), want, rtol=0, atol=1e-12)


def test_equipercentile_textbook_values() -> None:
    # The worked example checked against equate::equate(type = "equipercentile").
    f = np.array([0.1, 0.2, 0.3, 0.2, 0.15, 0.05])
    g = np.array([0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05])
    np.testing.assert_allclose(percentile_rank(f), [0.05, 0.2, 0.45, 0.7, 0.875, 0.975])
    np.testing.assert_allclose(
        equipercentile(f, g), [0.5, 1.75, 2.833333333333333, 3.75, 4.75, 6.0], atol=1e-14
    )


def test_equipercentile_of_identical_distributions_is_identity() -> None:
    f = np.random.default_rng(3).dirichlet(np.ones(12))
    np.testing.assert_allclose(equipercentile(f, f), np.arange(12), atol=1e-13)


# ---- kernels ---------------------------------------------------------------------------------


def _four_pl_form(params: np.ndarray, slope: float = 1.0, intercept: float = 0.0) -> Form:
    return Form(
        families=(FormFamily(FourPLKernel(), jnp.asarray(params), tuple(range(len(params)))),),
        n_items=len(params),
        max_categories=2,
        slope=jnp.asarray(slope),
        intercept=jnp.asarray(intercept),
    )


def test_tcc_and_solver_kernels() -> None:
    params = np.array([[1.2, 0.3, 0.2, 0.95], [0.7, -1.0, 0.1, 1.0], [2.0, 0.5, 0.0, 0.9]])
    form = _four_pl_form(params, slope=1.3, intercept=-0.2)
    theta = jnp.linspace(-3.0, 3.0, 7)
    own = (theta + 0.2) / 1.3
    z = params[:, 0][None, :] * np.asarray(own)[:, None] + params[:, 1][None, :]
    p = params[:, 2] + (params[:, 3] - params[:, 2]) / (1.0 + np.exp(-z))
    np.testing.assert_allclose(test_characteristic(form, theta), p.sum(axis=1), rtol=1e-14)

    targets = jnp.asarray([0.31, 1.0, 1.9, 2.84])
    solve = jax.jit(solve_true_score, static_argnames=("score_tol", "max_iter"))
    t, residual, n_iter = solve(
        form,
        targets,
        jnp.full(4, -100.0),
        jnp.full(4, 100.0),
        score_tol=1e-13,
        max_iter=300,
    )
    assert np.max(np.abs(np.asarray(residual))) <= 1e-12
    assert int(n_iter) < 300
    np.testing.assert_allclose(test_characteristic(form, t), targets, atol=1e-12)


# ---- score_distribution and scale_table vs summed_score_table ----------------------------


@pytest.mark.slow
def test_score_distribution_matches_summed_score_table() -> None:
    res = _single()
    got = score_distribution(res)
    want = summed_score_table(res)
    np.testing.assert_array_equal(got["score"], want["score"])
    np.testing.assert_allclose(got["probability"], want["probability"], rtol=1e-12, atol=1e-17)
    assert got["probability"].sum() == pytest.approx(1.0, abs=1e-13)


@pytest.mark.slow
def test_score_distribution_of_items_and_groups() -> None:
    res = _two_groups()
    for group in ("ref", "focal"):
        got = score_distribution(res, group=group)
        want = summed_score_table(res)
        want = want[want["group"] == group]
        np.testing.assert_allclose(got["probability"], want["probability"], rtol=1e-12, atol=1e-17)
    part = score_distribution(res, items=["p1", "g1"], group="ref")
    assert list(part["score"]) == [0, 1, 2, 3, 4]
    assert part["probability"].sum() == pytest.approx(1.0, abs=1e-13)


@pytest.mark.slow
def test_scale_table_eapsum_matches_summed_score_table() -> None:
    res = _single()
    got = scale_table(res, slope=15.0, intercept=100.0, method="EAPsum", n_digits=1)
    want = summed_score_table(res)
    np.testing.assert_allclose(got["theta"], want["theta"], rtol=1e-11)
    np.testing.assert_allclose(got["theta_se"], want["se"], rtol=1e-11)
    np.testing.assert_allclose(got["scale_score"], 15.0 * want["theta"] + 100.0, rtol=1e-12)
    np.testing.assert_allclose(got["scale_score_se"], 15.0 * want["se"], rtol=1e-11)
    np.testing.assert_array_equal(got["scale_score_rounded"], np.round(got["scale_score"], 1))


@pytest.mark.slow
def test_scale_table_true_score_bounds() -> None:
    res = _single()
    items = ["p1", "t1", "t2", "f1", "g1"]
    lo = 0.2 + 0.15 + 0.1
    hi = 1 + 1 + 1 + 0.9 + 3
    unbounded = scale_table(res, slope=10.0, intercept=50.0, items=items)
    assert list(unbounded["score"]) == list(range(8))
    assert np.isneginf(unbounded["theta"].iloc[0])
    assert np.all(np.isposinf(unbounded["theta"][unbounded["score"] >= hi]))
    assert np.all(
        np.isfinite(unbounded["theta"][(unbounded["score"] > lo) & (unbounded["score"] < hi)])
    )
    assert np.isneginf(unbounded["scale_score"].iloc[0])
    assert np.isnan(unbounded["theta_se"].iloc[0])
    clipped = scale_table(res, slope=10.0, intercept=50.0, items=items, lowest=20, highest=80)
    assert clipped["scale_score"].iloc[0] == 20.0
    assert clipped["scale_score"].iloc[-1] == 80.0
    assert clipped["scale_score"].is_monotonic_increasing
    inside = np.isfinite(clipped["theta"]).to_numpy()
    info = item_information(res, clipped["theta"][inside].to_numpy())[items].sum(axis=1)
    np.testing.assert_allclose(
        clipped["theta_se"][inside], 1.0 / np.sqrt(info.to_numpy()), rtol=1e-12
    )
    np.testing.assert_allclose(clipped["scale_score_se"], 10.0 * clipped["theta_se"], rtol=1e-15)


# ---- true-score equating ---------------------------------------------------------------------


@pytest.mark.slow
def test_true_score_equating_identity() -> None:
    res = _single()
    out = irt_equate(res, res, new_items=MONOTONE, old_items=MONOTONE)
    assert isinstance(out, IRTEquatingResult)
    np.testing.assert_allclose(out.table["equated"], out.table["score"], atol=1e-11)


@pytest.mark.slow
def test_true_score_boundary_rules_with_4pl() -> None:
    res = _single()
    new_items = ["t1", "f1", "f2", "p2"]  # lower 0.35, upper 3.75, max 4
    old_items = ["t2", "f2", "g2", "p3"]  # lower 0.2, upper 4.85, max 5
    out = irt_equate(res, res, new_items=new_items, old_items=old_items)
    table = out.table.set_index("score")
    new_lo, new_hi = out.new_range
    old_lo, old_hi = out.old_range
    assert (new_lo, new_hi) == pytest.approx((0.35, 3.75), abs=1e-12)
    assert (old_lo, old_hi) == pytest.approx((0.2, 4.85), abs=1e-12)
    assert table.loc[0, "equated"] == 0.0
    theta_column = table["theta"].to_numpy()
    assert np.isneginf(theta_column[0])
    assert np.isposinf(theta_column[4])
    assert table.loc[4, "equated"] == pytest.approx(5.0, abs=1e-14)
    # Scores inside: tau_old(theta) with tau_new(theta) = x.
    theta = table.loc[[1, 2, 3], "theta"].to_numpy()
    e = expected_score(res, theta)
    np.testing.assert_allclose(e[new_items].sum(axis=1), [1, 2, 3], atol=1e-11)
    np.testing.assert_allclose(
        table.loc[[1, 2, 3], "equated"], e[old_items].sum(axis=1), atol=1e-12
    )


def test_true_score_interpolation_beyond_both_asymptotes() -> None:
    # New form: three 4PL items with g = 0.3, u = 0.5 (range 0.9..1.5, maximum 3); old form: two
    # items with g = 0.2, u = 0.6 and one 2PL (range 0.4..2.2, maximum 3).
    new = _FormSpec(
        form=_four_pl_form(
            np.array([[1.0, 0.0, 0.3, 0.5], [1.5, 0.5, 0.3, 0.5], [0.8, -0.5, 0.3, 0.5]])
        ),
        items=("x1", "x2", "x3"),
        max_score=3,
        nodes=np.zeros(1),
        weights=np.ones(1),
    )
    old = _FormSpec(
        form=_four_pl_form(
            np.array([[1.2, 0.2, 0.2, 0.6], [0.9, -0.3, 0.2, 0.6], [1.1, 0.4, 0.0, 1.0]])
        ),
        items=("y1", "y2", "y3"),
        max_score=3,
        nodes=np.zeros(1),
        weights=np.ones(1),
    )
    table, (new_lo, new_hi, old_lo, old_hi) = _true_score_equate(new, old)
    assert (new_lo, new_hi) == pytest.approx((0.9, 1.5), abs=1e-12)
    assert (old_lo, old_hi) == pytest.approx((0.4, 2.2), abs=1e-12)
    assert list(table["score"]) == [0, 1, 2, 3]
    # Score 1 is inside (0.9, 1.5): solved; scores 2 and 3 lie on the upper segment.
    assert np.isfinite(table["theta"][1])
    np.testing.assert_allclose(
        table["equated"][2:], old_hi + (np.array([2.0, 3.0]) - new_hi) * (3 - old_hi) / (3 - new_hi)
    )
    assert table["equated"].iloc[-1] == pytest.approx(3.0)
    assert np.all(np.isposinf(table["theta"][2:]))
    assert table["equated"][0] == 0.0 and np.isneginf(table["theta"][0])


def _link(a: float, b: float, items: list[str], groups: tuple[str, str] = ("all", "all")) -> Any:
    """A ``LinkResult`` with given constants (only ``A``, ``B``, items and groups are used)."""
    return LinkResult(
        A=a,
        B=b,
        A_se=np.nan,
        B_se=np.nan,
        vcov=None,
        criterion=0.0,
        method="stocking_lord",
        symmetric=False,
        transformation="diagonal",
        converged=True,
        n_iter=0,
        grad_norm=0.0,
        common_items=(),
        items=pd.DataFrame({"item": items}),
        _source_group=groups[0],
        _target_group=groups[1],
    )


@pytest.mark.slow
def test_link_evaluates_new_items_at_the_transformed_theta() -> None:
    res = _single()
    new_items, old_items = ["p1", "p2", "g1", "c1"], MONOTONE[:6]
    out = irt_equate(
        res, res, link=_link(1.3, -0.4, new_items), new_items=new_items, old_items=old_items
    )
    theta = out.table["theta"].to_numpy()[1:-1]
    e_new = expected_score(res, (theta + 0.4) / 1.3)[new_items].sum(axis=1).to_numpy()
    np.testing.assert_allclose(e_new, out.table["score"].to_numpy()[1:-1], atol=1e-11)
    e_old = expected_score(res, theta)[old_items].sum(axis=1).to_numpy()
    np.testing.assert_allclose(out.table["equated"].to_numpy()[1:-1], e_old, atol=1e-11)
    # Observed score: the new population's nodes move by A theta + B; with A = 1, B = 0 the link
    # changes nothing.
    plain = irt_equate(res, res, method="observed_score", new_items=new_items, old_items=old_items)
    same = irt_equate(
        res,
        res,
        method="observed_score",
        new_items=new_items,
        old_items=old_items,
        link=_link(1.0, 0.0, new_items),
    )
    np.testing.assert_allclose(same.table["equated"], plain.table["equated"], atol=1e-14)


@pytest.mark.slow
def test_multigroup_forms_and_populations() -> None:
    res = _two_groups()
    new_items, old_items = ["p1", "p2", "g1"], ["p3", "t1", "g1"]
    common: dict[str, Any] = {
        "method": "observed_score",
        "new_items": new_items,
        "old_items": old_items,
        "new_group": "ref",
        "old_group": "focal",
    }
    on_new = irt_equate(res, res, **common)
    on_old = irt_equate(res, res, population="old", **common)
    mixed = irt_equate(res, res, population="synthetic", synthetic_weight=0.3, **common)
    f_ref = score_distribution(res, items=new_items, group="ref")["probability"].to_numpy()
    g_focal = score_distribution(res, items=old_items, group="focal")["probability"].to_numpy()
    assert on_new.new_distribution is not None and on_old.old_distribution is not None
    np.testing.assert_allclose(on_new.new_distribution["probability"], f_ref, rtol=1e-13)
    np.testing.assert_allclose(on_old.old_distribution["probability"], g_focal, rtol=1e-13)
    assert mixed.new_distribution is not None and on_old.new_distribution is not None
    np.testing.assert_allclose(
        mixed.new_distribution["probability"],
        0.3 * f_ref + 0.7 * on_old.new_distribution["probability"].to_numpy(),
        rtol=1e-13,
    )
    with pytest.raises(SpecificationError, match="new_group="):
        irt_equate(res, res)


@pytest.mark.slow
def test_observed_score_identity_and_populations() -> None:
    res = _single()
    out = irt_equate(res, res, method="observed_score", new_items=MONOTONE, old_items=MONOTONE)
    np.testing.assert_allclose(out.table["equated"], out.table["score"], atol=1e-11)
    assert out.population == "new" and out.synthetic_weight == 1.0
    old = irt_equate(res, res, method="observed_score", new_items=["p1", "t1"], population="old")
    assert old.synthetic_weight == 0.0
    # Nominal items are fine for observed-score equating.
    nominal = irt_equate(res, res, method="observed_score", new_items=["n1", "p1"])
    assert np.all(np.diff(nominal.table["equated"]) > 0)


# ---- errors and display ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"method": "kernel"}, "Unknown method"),
        ({"population": "old"}, "only used with method"),
        ({"method": "observed_score", "population": "synthetic"}, "needs synthetic_weight"),
        (
            {"method": "observed_score", "population": "synthetic", "synthetic_weight": 1.5},
            "in \\[0, 1\\]",
        ),
        ({"method": "observed_score", "synthetic_weight": 0.5}, "only used with population"),
        ({"new_items": ["zz"]}, "Unknown items"),
        ({"new_items": ["p1", "p1"]}, "Duplicate"),
        ({"new_items": "p1"}, "not a str"),
        ({"new_items": ["neg", "p1"]}, "decreases"),
        ({"new_items": ["n1", "p1"]}, "decreases"),
        ({"new_group": "x"}, "only valid for a multiple-group"),
        ({"link": "x"}, "LinkResult"),
    ],
)
@pytest.mark.slow
def test_irt_equate_errors(kwargs: dict[str, Any], match: str) -> None:
    res = _single()
    with pytest.raises(SpecificationError, match=match):
        irt_equate(res, res, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"slope": 0.0, "intercept": 0.0}, "positive"),
        ({"slope": 1.0, "intercept": np.inf}, "finite"),
        ({"slope": 1.0, "intercept": 0.0, "lowest": 5, "highest": 5}, "below"),
        ({"slope": 1.0, "intercept": 0.0, "n_digits": -1}, "n_digits"),
        ({"slope": 1.0, "intercept": 0.0, "method": "EAP"}, "Unknown method"),
        ({"slope": 1.0, "intercept": 0.0, "items": ["neg"]}, "decreases"),
    ],
)
@pytest.mark.slow
def test_scale_table_errors(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(SpecificationError, match=match):
        scale_table(_single(), **kwargs)


@pytest.mark.slow
def test_non_fit_inputs_raise() -> None:
    with pytest.raises(SpecificationError, match="FitResult"):
        irt_equate("a", _single())  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match="FitResult"):
        score_distribution(None)  # type: ignore[arg-type]


@pytest.mark.slow
def test_link_must_match_groups_and_items() -> None:
    res = _single()
    with pytest.raises(SpecificationError, match="does not contain"):
        irt_equate(res, res, link=_link(1.0, 0.0, ["p1"]))
    with pytest.raises(SpecificationError, match="maps group"):
        irt_equate(res, res, link=_link(1.0, 0.0, list(ITEMS), ("a", "b")))


@pytest.mark.slow
def test_result_display() -> None:
    res = _single()
    out = irt_equate(res, res, method="observed_score", new_items=["p1", "t1", "g1"])
    assert "observed_score" in str(out)
    text = out.summary(digits=2)
    assert "Population" in text and "equated" in text
    assert "<table" in out._repr_html_()
    with pytest.raises(SpecificationError, match="digits"):
        out.summary(digits=-1)
