"""Unit tests for information, expected score, reliability and summed-score EAP (DESIGN.md §18.2).

Agreement with R mirt is in ``tests/test_mirt_m7_information.py`` (VALIDATION.md V37).
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pandas as pd
import pytest

from flexpyirt import (
    BAEM,
    GPCM,
    GRM,
    EmpiricalHistogram,
    FitResult,
    FourPL,
    Model,
    Nominal,
    Normal,
    ThreePL,
    TwoPL,
    expected_score,
    fit,
    item_information,
    reliability,
    summed_score_table,
    test_information,
)
from flexpyirt.errors import SpecificationError
from flexpyirt.items import ItemModel

THETA = np.array([-3.0, -1.2, 0.0, 0.7, 2.5])

MIXED: dict[str, tuple[ItemModel, dict[str, float]]] = {
    "two": (TwoPL(), {"a": 1.3, "d": -0.4}),
    "three": (ThreePL(), {"a": 1.7, "d": 0.3, "g": 0.2, "u": 1.0}),
    "four": (FourPL(), {"a": 1.1, "d": 0.1, "g": 0.15, "u": 0.95}),
    "grm": (GRM(n_categories=4), {"a": 1.4, "d1": 1.2, "d2": 0.1, "d3": -1.1}),
    "gpcm": (GPCM(n_categories=3), {"a": 0.9, "d1": 0.5, "d2": -0.2}),
    "nominal": (Nominal(n_categories=3), {"a": 1.2, "ak1": 0.6, "d1": 0.4, "d2": -0.3}),
}


def _fixed(
    items: dict[str, tuple[ItemModel, dict[str, float]]],
    *,
    groups: dict[str, tuple[float, float]] | None = None,
    scale: dict[str, tuple[float, float]] | None = None,
) -> FitResult:
    """A ``FitResult`` with every parameter fixed.

    ``groups`` maps group names to the latent ``(mean, var)``; ``scale`` maps group names to
    ``(slope factor, intercept shift per unit slope)`` applied to that group's item parameters.
    """
    spec = Model.from_items({name: model for name, (model, _) in items.items()})
    k_max = max(model.kernel.n_categories for model, _ in items.values())
    rows = np.asarray(
        [
            [min(c, model.kernel.n_categories - 1) for model, _ in items.values()]
            for c in range(k_max)
        ],
        dtype=float,
    )
    labels = None
    if groups is None:
        dummy = pd.DataFrame(rows, columns=list(items))
    else:
        spec = spec.with_groups(**{g: Normal(mean=m, var=v) for g, (m, v) in groups.items()})
        dummy = pd.DataFrame(np.vstack([rows] * len(groups)), columns=list(items))
        labels = np.repeat(list(groups), k_max)
    table = spec.parameter_table(dummy, group=labels)
    values = []
    for group, item, param, value in zip(
        table["group"], table["item"], table["param"], table["value"], strict=True
    ):
        if item == "__latent__":
            values.append(float(value))
            continue
        par = dict(items[item][1])
        if scale is not None and group in scale:
            factor, shift = scale[group]
            a = par["a"]
            par = {
                k: (v * factor if k == "a" else v + a * shift if k.startswith("d") else v)
                for k, v in par.items()
            }
        values.append(par[param])
    table["value"] = np.asarray(values)
    table["free"] = False
    return fit(spec, dummy, group=labels, table=table, se=None)


@cache
def _mixed() -> FitResult:
    return _fixed(MIXED)


def _probs(res: FitResult, theta: np.ndarray) -> list[np.ndarray]:
    """Per item, ``(T, K_j)`` category probabilities of a single-group fit at ``theta``."""
    prob = res._compiled_for(np.zeros((1, len(res._model.items)), dtype=np.int64))
    return res._traces(prob, np.asarray(theta, dtype=float))


def test_information_closed_forms_2pl_3pl() -> None:
    info = item_information(_mixed(), THETA)
    assert info.index.name == "theta"
    np.testing.assert_array_equal(info.index.to_numpy(), THETA)
    assert list(info.columns) == list(MIXED)
    a, d = MIXED["two"][1]["a"], MIXED["two"][1]["d"]
    p = 1.0 / (1.0 + np.exp(-(a * THETA + d)))
    np.testing.assert_allclose(info["two"], a * a * p * (1.0 - p), rtol=1e-12)
    par = MIXED["three"][1]
    a, g = par["a"], par["g"]
    p_star = 1.0 / (1.0 + np.exp(-(a * THETA + par["d"])))
    p = g + (1.0 - g) * p_star
    # Birnbaum (1968): I = a^2 (Q / P) ((P - g) / (1 - g))^2.
    np.testing.assert_allclose(
        info["three"], a * a * (1.0 - p) / p * ((p - g) / (1.0 - g)) ** 2, rtol=1e-12
    )


def test_information_matches_finite_differences_for_every_family() -> None:
    res = _mixed()
    info = item_information(res, THETA)
    h = 1e-5
    center, up, down = _probs(res, THETA), _probs(res, THETA + h), _probs(res, THETA - h)
    for j, name in enumerate(MIXED):
        slope = (up[j] - down[j]) / (2.0 * h)
        np.testing.assert_allclose(info[name], (slope**2 / center[j]).sum(axis=1), rtol=1e-8)


def test_test_information_expected_score_and_scalar_theta() -> None:
    res = _mixed()
    info = item_information(res, THETA)
    total = test_information(res, THETA)
    assert list(total.columns) == ["information", "se"]
    np.testing.assert_allclose(total["information"], info.sum(axis=1), rtol=1e-14)
    np.testing.assert_allclose(total["se"], 1.0 / np.sqrt(total["information"]), rtol=1e-14)
    expected = expected_score(res, THETA)
    assert list(expected.columns) == [*MIXED, "total"]
    for name, p in zip(MIXED, _probs(res, THETA), strict=True):
        np.testing.assert_allclose(expected[name], p @ np.arange(p.shape[1]), rtol=1e-13)
    np.testing.assert_allclose(expected["total"], expected[list(MIXED)].sum(axis=1), rtol=1e-14)
    # The test characteristic curve is increasing here and bounded by the maximum score.
    assert np.all(np.diff(expected["total"]) > 0)
    assert expected["total"].max() < 1 + 1 + 1 + 3 + 2 + 2
    one = test_information(res, 0.7)
    assert one.shape == (1, 2)
    assert float(one["information"].iloc[0]) == pytest.approx(float(total["information"].iloc[3]))


def test_summed_score_table_matches_enumeration() -> None:
    res = _mixed()
    table = summed_score_table(res)
    assert list(table.columns) == ["score", "theta", "se", "probability"]
    n_categories = [model.kernel.n_categories for model, _ in MIXED.values()]
    max_score = sum(k - 1 for k in n_categories)
    np.testing.assert_array_equal(table["score"], np.arange(max_score + 1))
    nodes = res._quadrature.nodes()
    weights = np.exp(-0.5 * nodes**2)
    weights /= weights.sum()
    traces = _probs(res, nodes)
    joint = np.zeros((max_score + 1, nodes.size))
    for y in np.ndindex(*n_categories):
        lik = np.prod([traces[j][:, y[j]] for j in range(len(traces))], axis=0)
        joint[sum(y)] += lik * weights
    p_score = joint.sum(axis=1)
    eap = joint @ nodes / p_score
    sd = np.sqrt((joint * (nodes[None, :] - eap[:, None]) ** 2).sum(axis=1) / p_score)
    np.testing.assert_allclose(table["probability"], p_score, rtol=1e-12)
    np.testing.assert_allclose(table["theta"], eap, rtol=0, atol=1e-12)
    np.testing.assert_allclose(table["se"], sd, rtol=0, atol=1e-12)
    assert float(table["probability"].sum()) == pytest.approx(1.0, abs=1e-13)
    # Law of total expectation: the EAPs average to the prior mean.
    assert float((table["probability"] * table["theta"]).sum()) == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.diff(table["theta"]) > 0)


def test_eapsum_scores_look_up_the_table_and_incomplete_rows_are_nan() -> None:
    res = _mixed()
    table = summed_score_table(res)
    data = pd.DataFrame(
        [[0, 0, 0, 0, 0, 0], [1, 1, 1, 3, 2, 2], [1, 0, 1, 2, 0, 1], [1, np.nan, 1, 2, 0, 1]],
        columns=list(MIXED),
        index=["low", "high", "mid", "incomplete"],
    )
    out = res.score(data, method="EAPsum")
    assert list(out.columns) == ["theta", "se"]
    assert list(out.index) == list(data.index)
    for label, score in [("low", 0), ("high", 10), ("mid", 5)]:
        assert out.loc[label, "theta"] == table["theta"].iloc[score]
        assert out.loc[label, "se"] == table["se"].iloc[score]
    assert out.loc[["incomplete"]].isna().all(axis=None)


def test_reliability_bounds_and_more_items_is_more_reliable() -> None:
    short = _fixed({k: MIXED[k] for k in ["two", "grm"]})
    full = _mixed()
    r_short = float(reliability(short)["marginal"].iloc[0])
    r_full = float(reliability(full)["marginal"].iloc[0])
    assert 0.0 < r_short < r_full < 1.0
    rng = np.random.default_rng(3)
    theta = rng.normal(size=4000)
    data = pd.DataFrame(
        {
            name: (rng.random((theta.size, 1)) > np.cumsum(p, axis=1)[:, :-1]).sum(axis=1)
            for name, p in zip(MIXED, _probs(full, theta), strict=True)
        }
    ).astype(float)
    out = reliability(full, data)
    assert list(out.columns) == ["marginal", "empirical"]
    empirical = float(out["empirical"].iloc[0])
    assert 0.0 < empirical < 1.0
    # Data simulated from the model: the two estimates of the same quantity are close.
    assert empirical == pytest.approx(r_full, abs=0.03)
    scores = full.score(data)
    var = float(np.var(scores["theta"].to_numpy(), ddof=1))
    err = float(np.mean(scores["se"].to_numpy() ** 2))
    assert empirical == pytest.approx(var / (var + err), rel=1e-13)
    assert np.isnan(reliability(full, data.iloc[:1])["empirical"].iloc[0])


MOVED_MEAN, MOVED_VAR = 0.6, 2.2
MOVED_ITEMS = {k: MIXED[k] for k in ["two", "three", "four", "grm"]}


@cache
def _moved() -> FitResult:
    """Two groups that are the same model on two latent scales.

    Group "moved" is group "base" on the scale theta' = mean + sd * theta:
    a' = a / sd, d' = d - a mean / sd (dichotomous and GRM items).
    """
    sd = float(np.sqrt(MOVED_VAR))
    return _fixed(
        MOVED_ITEMS,
        groups={"base": (0.0, 1.0), "moved": (MOVED_MEAN, MOVED_VAR)},
        scale={"moved": (1.0 / sd, -MOVED_MEAN / sd)},
    )


def test_marginal_reliability_is_invariant_to_the_latent_scale() -> None:
    mean, var, sd = MOVED_MEAN, MOVED_VAR, float(np.sqrt(MOVED_VAR))
    res = _moved()
    out = reliability(res)
    assert list(out.columns) == ["group", "marginal"]
    assert list(out["group"]) == ["base", "moved"]
    assert out["marginal"].iloc[1] == pytest.approx(out["marginal"].iloc[0], abs=1e-13)
    assert out["marginal"].iloc[0] == pytest.approx(
        float(reliability(_fixed(MOVED_ITEMS))["marginal"].iloc[0]), abs=1e-14
    )
    # Information transforms with the square of the scale; expected scores are unchanged.
    base = test_information(res, THETA, group="base")["information"].to_numpy()
    moved = test_information(res, mean + sd * THETA, group="moved")["information"].to_numpy()
    np.testing.assert_allclose(moved * var, base, rtol=1e-12)
    np.testing.assert_allclose(
        expected_score(res, mean + sd * THETA, group="moved").to_numpy(),
        expected_score(res, THETA, group="base").to_numpy(),
        rtol=1e-12,
    )


def test_summed_score_table_and_scores_per_group() -> None:
    mean, sd = MOVED_MEAN, float(np.sqrt(MOVED_VAR))
    res = _moved()
    # The summed-score distribution is the same and the EAPs are on the moved scale, up to
    # the truncation of N(0.6, 2.2) by the fitted grid on [-6, 6].
    table = summed_score_table(res)
    assert list(table.columns) == ["group", "score", "theta", "se", "probability"]
    t_base = table[table["group"] == "base"].reset_index(drop=True)
    t_moved = table[table["group"] == "moved"].reset_index(drop=True)
    np.testing.assert_allclose(t_moved["probability"], t_base["probability"], rtol=2e-3)
    np.testing.assert_allclose(t_moved["theta"], mean + sd * t_base["theta"], atol=2e-2)
    # Persons are scored with their own group's table.
    data = pd.DataFrame([[1, 0, 1, 3]] * 2, columns=list(MOVED_ITEMS))
    scores = res.score(data, method="EAPsum", group=["base", "moved"])
    assert scores["theta"].iloc[0] == t_base["theta"].iloc[5]
    assert scores["theta"].iloc[1] == t_moved["theta"].iloc[5]
    assert scores["theta"].iloc[0] != scores["theta"].iloc[1]
    emp = reliability(res, data, group=["base", "moved"])
    assert list(emp.columns) == ["group", "marginal", "empirical"]
    assert emp["empirical"].isna().all()


def test_group_and_theta_validation() -> None:
    single = _mixed()
    multi = _fixed(MIXED, groups={"ref": (0.0, 1.0), "focal": (0.3, 1.4)})
    for fn in (item_information, test_information, expected_score):
        with pytest.raises(SpecificationError, match="needs group="):
            fn(multi, THETA)
        with pytest.raises(SpecificationError, match="Unknown group"):
            fn(multi, THETA, group="other")
        with pytest.raises(SpecificationError, match="only valid for a multiple-group"):
            fn(single, THETA, group="ref")
        with pytest.raises(SpecificationError, match="finite"):
            fn(single, [0.0, np.nan])
        with pytest.raises(SpecificationError, match="1-D"):
            fn(single, np.zeros((2, 2)))
        with pytest.raises(SpecificationError, match="1-D"):
            fn(single, [])
        with pytest.raises(SpecificationError, match="number"):
            fn(single, "abc")
        with pytest.raises(SpecificationError, match="FitResult"):
            fn("not a fit", THETA)  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match="together with data"):
        reliability(single, group=["a"])
    with pytest.raises(SpecificationError, match="'total'"):
        expected_score(_fixed({"total": MIXED["two"], "x": MIXED["grm"]}), THETA)


@pytest.mark.slow
def test_multidimensional_and_non_normal_models() -> None:
    rng = np.random.default_rng(11)
    items = [f"q{j}" for j in range(1, 7)]
    theta = rng.normal(size=(400, 1))
    data = pd.DataFrame(
        (rng.random((400, 6)) < 1 / (1 + np.exp(-(theta + np.linspace(-1, 1, 6))))).astype(float),
        columns=items,
    )
    method = BAEM(max_iter=500, tol=1e-3)
    two_factor = fit(
        Model.uniform(TwoPL(), items=items).with_factors({"f1": items[:3], "f2": items[3:]}),
        data,
        method=method,
        se=None,
    )
    for fn in (item_information, test_information, expected_score):
        with pytest.raises(SpecificationError, match="unidimensional"):
            fn(two_factor, THETA)
    with pytest.raises(SpecificationError, match="unidimensional"):
        reliability(two_factor)
    with pytest.raises(SpecificationError, match="unidimensional"):
        summed_score_table(two_factor)
    with pytest.raises(SpecificationError, match="EAPsum"):
        two_factor.score(data, method="EAPsum")

    histogram = fit(
        Model.uniform(TwoPL(), items=items).with_latent(EmpiricalHistogram()),
        data,
        method=method,
        se=None,
    )
    # Information and expected scores do not involve the latent distribution.
    info = test_information(histogram, THETA)
    assert np.all(info["information"] > 0)
    assert expected_score(histogram, THETA)["total"].between(0, 6).all()
    with pytest.raises(SpecificationError, match="Normal latent"):
        reliability(histogram)
    with pytest.raises(SpecificationError, match="Normal latent"):
        summed_score_table(histogram)
    with pytest.raises(SpecificationError, match="Normal latent"):
        histogram.score(data, method="EAPsum")


@pytest.mark.parametrize("structure", ["bifactor", "two_tier"])
def test_bifactor_and_two_tier_models_are_rejected(structure: str) -> None:
    items = [f"q{j}" for j in range(1, 9)]
    base = Model.uniform(TwoPL(), items=items)
    if structure == "bifactor":
        spec = base.with_bifactor(specific={"s1": items[:4], "s2": items[4:]})
    else:
        spec = base.with_two_tier(
            primary={"p1": items[:4], "p2": items[4:]},
            specific={"s1": items[:2] + items[4:6], "s2": items[2:4] + items[6:]},
        )
    assert spec.is_multidimensional
    dummy = pd.DataFrame(np.vstack([np.zeros(8), np.ones(8)]), columns=items)
    table = spec.parameter_table(dummy)
    table["free"] = False
    res = fit(spec, dummy, table=table, se=None)
    for fn in (item_information, test_information, expected_score):
        with pytest.raises(SpecificationError, match="unidimensional"):
            fn(res, THETA)
    with pytest.raises(SpecificationError, match="unidimensional"):
        reliability(res)
    with pytest.raises(SpecificationError, match="unidimensional"):
        reliability(res, dummy)
    with pytest.raises(SpecificationError, match="unidimensional"):
        summed_score_table(res)
    with pytest.raises(SpecificationError, match="EAPsum"):
        res.score(dummy, method="EAPsum")
