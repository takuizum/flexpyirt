"""Classification accuracy and consistency without R (DESIGN.md §18.4, VALIDATION.md V52).

Independent checks of :func:`flexpyirt.classification_accuracy`: brute-force enumeration of all
response patterns of a short mixed-format test (item probabilities written out here in NumPy),
a Monte Carlo simulation of two parallel administrations, the identities between the indices,
the cut-scale conversions and the input errors. The comparison with cacIRT and a
recomputation in R is in ``test_r_classification.py``.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from flexpyirt import (
    BAEM,
    GPCM,
    GRM,
    FitResult,
    Model,
    Normal,
    Quadrature,
    ThreePL,
    TwoPL,
    classification_accuracy,
    expected_score,
    fit,
    test_information,
)
from flexpyirt.errors import SpecificationError
from flexpyirt.items import ItemModel

# Mixed-format test: two 2PL, one 3PL, one GRM (K = 3), one GPCM (K = 3); S_max = 7.
MIXED: dict[str, tuple[ItemModel, dict[str, float]]] = {
    "a": (TwoPL(), {"a": 1.3, "d": 0.4}),
    "b": (TwoPL(), {"a": 0.8, "d": -0.6}),
    "c": (ThreePL(), {"a": 1.6, "d": -0.2, "g": 0.2}),
    "d": (GRM(n_categories=3), {"a": 1.1, "d1": 1.0, "d2": -0.7}),
    "e": (GPCM(n_categories=3), {"a": 0.9, "d1": 0.5, "d2": 0.3}),
}


def _fixed(
    items: dict[str, tuple[ItemModel, dict[str, float]]],
    *,
    quadrature: Quadrature | None = None,
    groups: dict[str, Normal] | None = None,
) -> FitResult:
    """A ``FitResult`` with every parameter fixed at the given values."""
    spec = Model.from_items({name: model for name, (model, _) in items.items()})
    # Placeholder responses with every category observed (in each group); the values do not
    # matter because nothing is estimated.
    n_rows = 6
    data = pd.DataFrame(
        {
            name: np.arange(n_rows) % getattr(model, "n_categories", 2)
            for name, (model, _) in items.items()
        }
    )
    group = None
    if groups is not None:
        spec = spec.with_groups(**groups)
        data = pd.concat([data] * len(groups), ignore_index=True)
        group = np.repeat(list(groups), n_rows)
    table = spec.parameter_table(data, group=group)
    values = [
        value if item == "__latent__" else items[item][1].get(param, value)
        for item, param, value in zip(table["item"], table["param"], table["value"], strict=True)
    ]
    table["value"] = np.asarray(values, dtype=float)
    table["free"] = False
    return fit(spec, data, group=group, table=table, quadrature=quadrature, se=None)


def _numpy_probs(theta: np.ndarray) -> list[np.ndarray]:
    """Category probabilities of the MIXED items, written out independently of flexpyirt."""

    def logistic(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    out = []
    for model, par in MIXED.values():
        z = par["a"] * theta
        if isinstance(model, ThreePL):
            p1 = par["g"] + (1.0 - par["g"]) * logistic(z + par["d"])
            out.append(np.stack([1.0 - p1, p1], axis=-1))
        elif isinstance(model, TwoPL):
            p1 = logistic(z + par["d"])
            out.append(np.stack([1.0 - p1, p1], axis=-1))
        elif isinstance(model, GRM):
            upper = np.stack(
                [np.ones_like(z), logistic(z + par["d1"]), logistic(z + par["d2"]), 0 * z], -1
            )
            out.append(upper[:, :-1] - upper[:, 1:])
        else:  # GPCM: P_k proportional to exp(k a theta + d_k), d_0 = 0
            logits = np.stack([0 * z, z + par["d1"], 2 * z + par["d2"]], axis=-1)
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            out.append(e / e.sum(axis=1, keepdims=True))
    return out


@pytest.fixture(scope="module")
def mixed() -> FitResult:
    return _fixed(MIXED)


@cache
def _fine() -> FitResult:
    """The MIXED items on a fine grid, for the Monte Carlo comparisons."""
    return _fixed(MIXED, quadrature=Quadrature.equispaced(2401))


def _category(values: Any, cuts: Sequence[float]) -> np.ndarray:
    return np.sum(np.asarray(values)[..., None] >= np.asarray(cuts), axis=-1)


@pytest.mark.parametrize("cuts", [[3], [2, 5], [1.5, 3.2, 6]])
def test_lee_matches_brute_force_enumeration(cuts: list[float], mixed: FitResult) -> None:
    """All 2 x 2 x 2 x 3 x 3 = 72 response patterns at every node give the same tables."""
    out = classification_accuracy(mixed, cut_scores=cuts)
    cond = out.conditional
    theta = cond["theta"].to_numpy()
    weights = cond["weight"].to_numpy()
    expected_weights = norm.pdf(theta)
    np.testing.assert_allclose(weights, expected_weights / expected_weights.sum(), rtol=1e-12)
    probs = _numpy_probs(theta)
    n_categories = len(cuts) + 1
    p_cat = np.zeros((theta.size, n_categories))
    for pattern in itertools.product(*(range(p.shape[1]) for p in probs)):
        p_pattern = np.prod([p[:, k] for p, k in zip(probs, pattern, strict=True)], axis=0)
        p_cat[:, _category(np.array(sum(pattern)), cuts)] += p_pattern
    true_score = sum(p @ np.arange(p.shape[1]) for p in probs)
    true = _category(true_score, cuts)
    a = np.zeros((n_categories, n_categories))
    for t in range(n_categories):
        a[t] = (weights[true == t, None] * p_cat[true == t]).sum(axis=0)
    b = p_cat.T @ (weights[:, None] * p_cat)
    np.testing.assert_allclose(cond["true_score"], true_score, rtol=1e-12)
    np.testing.assert_array_equal(cond["true_category"], true)
    np.testing.assert_allclose(cond["accuracy"], p_cat[np.arange(theta.size), true], atol=1e-14)
    np.testing.assert_allclose(cond["consistency"], (p_cat**2).sum(axis=1), atol=1e-14)
    np.testing.assert_allclose(out.accuracy_table.to_numpy(), a, atol=1e-14)
    np.testing.assert_allclose(out.consistency_table.to_numpy(), b, atol=1e-14)
    assert out.accuracy == pytest.approx(np.trace(a), abs=1e-14)
    assert out.consistency == pytest.approx(np.trace(b), abs=1e-14)
    chance = np.sum(b.sum(axis=1) ** 2)
    assert out.kappa == pytest.approx((np.trace(b) - chance) / (1 - chance), abs=1e-13)


def test_rudner_matches_direct_normal_probabilities(mixed: FitResult) -> None:
    cuts = [-0.4, 0.9]
    res = mixed
    out = classification_accuracy(res, cut_scores=cuts, method="rudner")
    cond = out.conditional
    theta = cond["theta"].to_numpy()
    se = test_information(res, theta)["se"].to_numpy()
    np.testing.assert_allclose(cond["se"], se, rtol=1e-13)
    bounds = np.array([-np.inf, *cuts, np.inf])
    p_cat = np.diff(norm.cdf((bounds[None, :] - theta[:, None]) / se[:, None]), axis=1)
    true = _category(theta, cuts)
    np.testing.assert_allclose(cond["accuracy"], p_cat[np.arange(theta.size), true], atol=1e-14)
    np.testing.assert_allclose(cond["consistency"], (p_cat**2).sum(axis=1), atol=1e-14)


def _indices_identities(out: Any) -> None:
    a = out.accuracy_table.to_numpy()
    b = out.consistency_table.to_numpy()
    assert a.sum() == pytest.approx(1.0, abs=1e-13)
    assert b.sum() == pytest.approx(1.0, abs=1e-13)
    np.testing.assert_allclose(b, b.T, atol=1e-15)
    # Both tables have the observed-category distribution as their observed margin.
    np.testing.assert_allclose(a.sum(axis=0), b.sum(axis=1), atol=1e-13)
    cat = out.by_category
    np.testing.assert_allclose(cat["proportion"], b.sum(axis=1), atol=1e-15)
    np.testing.assert_allclose(cat["true_proportion"], a.sum(axis=1), atol=1e-15)
    cut = out.by_cut
    np.testing.assert_allclose(
        cut["accuracy"] + cut["false_positive"] + cut["false_negative"], 1.0, atol=1e-13
    )
    # Overall accuracy is at most each per-cut accuracy (a correct overall category is
    # correct for every cut).
    assert np.all(cut["accuracy"] >= out.accuracy - 1e-13)
    assert np.all(cut["consistency"] >= out.consistency - 1e-13)


@pytest.mark.parametrize("method", ["lee", "rudner"])
@pytest.mark.parametrize("data", [False, True])
def test_identities_between_indices(method: str, data: bool, mixed: FitResult) -> None:
    res = mixed
    cuts = [2, 5] if method == "lee" else [-0.4, 0.9]
    kwargs: dict[str, Any] = {}
    if data:
        rng = np.random.default_rng(3)
        kwargs["data"] = pd.DataFrame(
            {
                name: rng.integers(
                    0, model.n_categories if hasattr(model, "n_categories") else 2, 40
                )
                for name, (model, _) in MIXED.items()
            },
            index=pd.RangeIndex(100, 140),
        )
    out = classification_accuracy(res, cut_scores=cuts, method=method, **kwargs)  # type: ignore[arg-type]
    _indices_identities(out)
    if data:
        assert out.approach == "P"
        assert list(out.conditional.index) == list(range(100, 140))
        assert int(out.summary["n"].iloc[0]) == 40
        np.testing.assert_allclose(out.conditional["weight"], 1 / 40)
        assert out.accuracy == pytest.approx(out.conditional["accuracy"].mean(), abs=1e-14)
        assert out.consistency == pytest.approx(out.conditional["consistency"].mean(), abs=1e-14)
    else:
        assert out.approach == "D"
        cond = out.conditional
        assert out.accuracy == pytest.approx(cond["weight"] @ cond["accuracy"], abs=1e-14)


@pytest.mark.parametrize("method", ["lee", "rudner"])
def test_one_cut_per_cut_equals_overall(method: str, mixed: FitResult) -> None:
    out = classification_accuracy(mixed, cut_scores=3 if method == "lee" else 0.2, method=method)  # type: ignore[arg-type]
    row = out.by_cut.iloc[0]
    for key in ("accuracy", "consistency", "kappa"):
        assert row[key] == pytest.approx(out.summary[key].iloc[0], abs=1e-15)


def test_scale_conversions_go_through_the_test_characteristic_curve(mixed: FitResult) -> None:
    res = mixed
    theta_cuts = [-0.5, 0.7]
    lee_theta = classification_accuracy(res, cut_scores=theta_cuts, scale="theta")
    tcc = expected_score(res, theta_cuts)["total"].to_numpy()
    np.testing.assert_allclose(lee_theta.by_cut["summed_score_cut"], tcc, rtol=1e-14)
    lee_summed = classification_accuracy(res, cut_scores=tcc)
    pd.testing.assert_frame_equal(lee_theta.summary, lee_summed.summary, rtol=1e-14)

    summed_cuts = [2.0, 4.5]
    rudner_summed = classification_accuracy(
        res, cut_scores=summed_cuts, method="rudner", scale="summed_score"
    )
    inverse = rudner_summed.by_cut["theta_cut"].to_numpy()
    np.testing.assert_allclose(expected_score(res, inverse)["total"], summed_cuts, atol=1e-12)
    rudner_theta = classification_accuracy(res, cut_scores=inverse, method="rudner")
    pd.testing.assert_frame_equal(rudner_summed.summary, rudner_theta.summary, rtol=1e-14)


def test_non_integer_cut_uses_the_ceiling(mixed: FitResult) -> None:
    res = mixed
    a = classification_accuracy(res, cut_scores=[2.01, 4.99])
    b = classification_accuracy(res, cut_scores=[3, 5])
    # The observed categories are the same; only the true categories (tau vs the cut) differ.
    pd.testing.assert_frame_equal(
        a.by_category[["proportion", "consistency"]], b.by_category[["proportion", "consistency"]]
    )


@pytest.mark.slow
def test_monte_carlo_two_administrations() -> None:
    """Simulated parallel administrations agree with the D approach within Monte Carlo error.

    The quadrature is fine (2401 nodes) because the conditional accuracy jumps where the true
    category changes, which makes the D approach's error of the order of the node spacing.
    """
    res = _fine()
    cuts = [2, 5]
    out = classification_accuracy(res, cut_scores=cuts)
    rng = np.random.default_rng(20261126)
    n = 200_000
    theta = rng.standard_normal(n)
    probs = _numpy_probs(theta)

    def administer() -> np.ndarray:
        scores = np.zeros(n, dtype=int)
        for p in probs:
            u = rng.random(n)[:, None]
            scores += (u > np.cumsum(p, axis=1)[:, :-1]).sum(axis=1)
        return _category(scores, cuts)

    first, second = administer(), administer()
    true = _category(sum(p @ np.arange(p.shape[1]) for p in probs), cuts)
    for observed, expected in (
        (np.mean(first == true), out.accuracy),
        (np.mean(first == second), out.consistency),
    ):
        se = np.sqrt(expected * (1.0 - expected) / n)
        assert abs(observed - expected) < 4.5 * se
    joint = pd.crosstab(true, first, normalize=True).to_numpy()
    np.testing.assert_allclose(joint, out.accuracy_table.to_numpy(), atol=4.5 * np.sqrt(0.25 / n))


@pytest.mark.slow
def test_monte_carlo_rudner() -> None:
    res = _fine()
    cuts = [-0.4, 0.9]
    out = classification_accuracy(res, cut_scores=cuts, method="rudner")
    rng = np.random.default_rng(7)
    n = 200_000
    theta = rng.standard_normal(n)
    se = np.interp(theta, out.conditional["theta"], out.conditional["se"])
    first = _category(theta + se * rng.standard_normal(n), cuts)
    second = _category(theta + se * rng.standard_normal(n), cuts)
    true = _category(theta, cuts)
    for observed, expected in (
        (np.mean(first == true), out.accuracy),
        (np.mean(first == second), out.consistency),
    ):
        assert abs(observed - expected) < 4.5 * np.sqrt(expected * (1.0 - expected) / n)


@pytest.mark.slow
def test_multiple_groups_use_each_groups_distribution(mixed: FitResult) -> None:
    groups = {"ref": Normal(mean=0.0, var=1.0), "focal": Normal(mean=0.8, var=1.5)}
    res = _fixed(MIXED, groups=groups)
    out = classification_accuracy(res, cut_scores=[2, 5])
    assert list(out.summary["group"]) == ["ref", "focal"]
    assert list(out.accuracy_table.index.get_level_values("group").unique()) == ["ref", "focal"]
    ref = classification_accuracy(mixed, cut_scores=[2, 5])
    pd.testing.assert_frame_equal(
        out.summary[out.summary["group"] == "ref"].drop(columns="group").reset_index(drop=True),
        ref.summary,
        rtol=1e-13,
    )
    focal = out.conditional[out.conditional["group"] == "focal"]
    weights = norm.pdf(focal["theta"], 0.8, np.sqrt(1.5))
    np.testing.assert_allclose(focal["weight"], weights / weights.sum(), rtol=1e-12)
    with pytest.raises(SpecificationError, match="per group"):
        _ = out.accuracy
    data = pd.DataFrame({name: [0, 1, 1, 0] for name in MIXED})
    labels = ["focal", "focal", "focal", "focal"]
    only_focal = classification_accuracy(res, cut_scores=[2, 5], data=data, group=labels)
    assert list(only_focal.summary["group"]) == ["focal"]


def test_errors(mixed: FitResult) -> None:
    res = mixed
    with pytest.raises(SpecificationError, match="strictly increasing"):
        classification_accuracy(res, cut_scores=[4, 2])
    with pytest.raises(SpecificationError, match="finite"):
        classification_accuracy(res, cut_scores=[np.nan])
    with pytest.raises(SpecificationError, match=r"\(0, 7\]"):
        classification_accuracy(res, cut_scores=[8])
    with pytest.raises(SpecificationError, match=r"\(0, 7\]"):
        classification_accuracy(res, cut_scores=[0])
    with pytest.raises(SpecificationError, match="same summed scores"):
        classification_accuracy(res, cut_scores=[2.2, 2.7])
    with pytest.raises(SpecificationError, match="Unknown method"):
        classification_accuracy(res, cut_scores=[2], method="livingston")  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match="Unknown scale"):
        classification_accuracy(res, cut_scores=[2], scale="percent")  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match="only used together with data"):
        classification_accuracy(res, cut_scores=[2], score_method="ML")
    with pytest.raises(SpecificationError, match="outside the range"):
        classification_accuracy(res, cut_scores=[0.1], method="rudner", scale="summed_score")


@pytest.mark.slow
def test_multidimensional_model_is_rejected() -> None:
    items = ["x", "y", "z", "w"]
    rng = np.random.default_rng(5)
    data = pd.DataFrame(rng.integers(0, 2, (200, 4)), columns=items)
    two_factor = fit(
        Model.uniform(TwoPL(), items=items).with_factors({"f1": items[:2], "f2": items[2:]}),
        data,
        method=BAEM(max_iter=5, tol=1e-3),
        se=None,
    )
    with pytest.raises(SpecificationError, match="unidimensional"):
        classification_accuracy(two_factor, cut_scores=[2])


@pytest.mark.slow
def test_non_finite_estimates_are_rejected(mixed: FitResult) -> None:
    all_lowest = pd.DataFrame({name: [0, 1] for name in MIXED})
    with pytest.raises(SpecificationError, match="non-finite ML"):
        classification_accuracy(mixed, cut_scores=[2], data=all_lowest, score_method="ML")
