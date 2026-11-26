"""IRT true-score / observed-score equating and scale tables against R (VALIDATION.md V51).

Reference values come from ``validation/r/v51_irt_equating_reference.R`` ->
``tests/reference/v51_irt_equating.json``: equateIRT (``score_compute``, TSE and OSE) and
SNSequate (``irt.eq(method = "TS")``) for 20 + 20 dichotomous 3PL / 2PL items with a link, and,
for mixed polytomous forms (GRM, GPCM, 3PL, 2PL), references assembled from mirt
(``expected.test()``, ``probtrace()``, ``testinfo()``, ``fscores(method = "EAPsum")``) and
``equate::equate(type = "equipercentile")``. Every parameter is fixed at the reference values, so
only the equating is under test.

The fast tier builds the forms straight from the reference parameters (no fit) and checks the
kernels and the conversion rules; the slow tier repeats the comparisons end to end through
``irt_equate()``, ``score_distribution()`` and ``scale_table()`` on fits fixed at those values.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from flexpyirt import (
    GPCM,
    GRM,
    FitResult,
    LinkResult,
    Model,
    Quadrature,
    ThreePL,
    TwoPL,
    fit,
    irt_equate,
    scale_table,
    score_distribution,
)
from flexpyirt._equating import Form, FormFamily
from flexpyirt.equating import (
    _FormSpec,
    _observed_score_equate,
    _scale_theta,
    _true_score_equate,
)
from flexpyirt.items import ItemKernel, ItemModel
from tests._reference import check_reference_meta

REFERENCE_FILE = Path(__file__).parent / "reference" / "v51_irt_equating.json"

# equateIRT's Newton stops at |d theta| < 1e-6, after which the quadratic convergence leaves
# about 1e-12; the true-score solutions here are good to 1e-13 in the score.
TOL_TSE = 1e-9
# SNSequate's uniroot() uses the default tolerance (about 1.2e-4 in theta).
TOL_TSE_SNSEQUATE_INTERIOR = 1e-3
# Exact formulas (linear interpolation, distributions, equipercentile).
TOL_EXACT = 1e-11
# equate::equate() rounds the percentile ranks and the cumulative distribution to 10 decimals
# (equate:::equip: round(px(x), 10), round(fx(y), 10)), i.e. about 1e-10 / g(y) in the score.
TOL_EQUATE = 1e-8
# uniroot(tol = 1e-13) and mirt's expected.test().
TOL_POLY_THETA = 1e-9


def _value(v: Any) -> float:
    if isinstance(v, str):
        return {"Inf": np.inf, "-Inf": -np.inf, "NaN": np.nan, "NA": np.nan}[v]
    return float(v)


def _array(values: list[Any]) -> np.ndarray:
    return np.asarray([_value(v) for v in values], dtype=np.float64)


@cache
def _reference() -> dict[str, Any]:
    raw = dict(json.loads(REFERENCE_FILE.read_text()))
    check_reference_meta(raw["meta"])
    return raw


def _item_model(itemtype: str, n_categories: int) -> ItemModel:
    if itemtype == "2PL":
        return TwoPL()
    if itemtype == "3PL":
        return ThreePL()
    if itemtype == "GRM":
        return GRM(n_categories=n_categories)
    if itemtype == "GPCM":
        return GPCM(n_categories=n_categories)
    raise ValueError(itemtype)


def _n_categories(params: dict[str, float]) -> int:
    """2 for 2PL / 3PL (intercept ``d``), else one more than the intercepts ``d1, d2, ...``."""
    return 2 if "d" in params else 1 + sum(1 for p in params if p.startswith("d"))


def _spec(case: str, side: str) -> _FormSpec:
    """The form of one side built from the reference parameters, without a fit."""
    ref = _reference()
    form = ref[case][side]
    items = list(form["items"])
    by_kernel: dict[ItemKernel, list[int]] = {}
    n_cats = []
    for position, item in enumerate(items):
        params = form["params"][item]
        n_cats.append(_n_categories(params))
        kernel = _item_model(str(form["itemtypes"][item]), n_cats[-1]).kernel
        by_kernel.setdefault(kernel, []).append(position)
    families = tuple(
        FormFamily(
            kernel=kernel,
            params=jnp.asarray(
                [[float(form["params"][items[c]][p]) for p in kernel.param_names] for c in cols]
            ),
            columns=tuple(cols),
        )
        for kernel, cols in by_kernel.items()
    )
    slope, intercept = 1.0, 0.0
    if side == "new":
        slope, intercept = float(ref[case]["link"]["A"]), float(ref[case]["link"]["B"])
    return _FormSpec(
        form=Form(
            families=families,
            n_items=len(items),
            max_categories=max(n_cats),
            slope=jnp.asarray(slope),
            intercept=jnp.asarray(intercept),
        ),
        items=tuple(items),
        max_score=sum(k - 1 for k in n_cats),
        nodes=slope * np.asarray(ref["nodes"]) + intercept,
        weights=np.asarray(ref["weights"]),
    )


def test_kernels_dichotomous_true_score() -> None:
    ref = _reference()["dichotomous"]
    table, (new_lo, _, old_lo, _) = _true_score_equate(
        _spec("dichotomous", "new"), _spec("dichotomous", "old")
    )
    assert new_lo == pytest.approx(ref["sum_guessing"]["new"], abs=1e-12)
    assert old_lo == pytest.approx(ref["sum_guessing"]["old"], abs=1e-12)
    e = ref["tse_equateirt"]
    scores = np.asarray(e["score"])
    inside = scores < 20
    got = table.set_index("score").loc[scores[inside]]
    np.testing.assert_allclose(got["equated"], np.asarray(e["equated"])[inside], atol=TOL_TSE)
    theta_old = ref["link"]["A"] * np.asarray(e["theta_new_scale"])[inside] + ref["link"]["B"]
    np.testing.assert_allclose(got["theta"], theta_old, atol=TOL_TSE)
    sns = _array(ref["tse_snsequate"]["equated"])
    exact = (table["score"] <= new_lo) | (table["score"] == 20)
    np.testing.assert_allclose(table["equated"][exact], sns[exact], rtol=0, atol=TOL_EXACT)
    np.testing.assert_allclose(
        table["equated"][~exact], sns[~exact], rtol=0, atol=TOL_TSE_SNSEQUATE_INTERIOR
    )


@pytest.mark.parametrize("population", ["new", "old", "synthetic"])
def test_kernels_dichotomous_observed_score(population: str) -> None:
    ref = _reference()["dichotomous"]["ose_equateirt"][population]
    equated, _, _ = _observed_score_equate(
        _spec("dichotomous", "new"), _spec("dichotomous", "old"), float(ref["w"])
    )
    np.testing.assert_allclose(equated, ref["equated"], rtol=0, atol=TOL_EXACT)


def test_kernels_polytomous_true_score() -> None:
    ref = _reference()["polytomous"]["tse"]
    table, ranges = _true_score_equate(_spec("polytomous", "new"), _spec("polytomous", "old"))
    np.testing.assert_allclose(table["theta"], _array(ref["theta"]), atol=TOL_POLY_THETA)
    np.testing.assert_allclose(table["equated"], _array(ref["equated"]), atol=TOL_POLY_THETA)
    assert ranges[:2] == pytest.approx((0.0, 18.0), abs=1e-12)


@pytest.mark.parametrize("population", ["new", "old", "synthetic"])
def test_kernels_polytomous_observed_score(population: str) -> None:
    ref = _reference()["polytomous"]
    want = ref["ose"][population]
    equated, f, g = _observed_score_equate(
        _spec("polytomous", "new"), _spec("polytomous", "old"), float(want["w"])
    )
    np.testing.assert_allclose(f, want["f"], rtol=1e-11)
    np.testing.assert_allclose(g, want["g"], rtol=1e-11)
    np.testing.assert_allclose(equated, want["equated"], rtol=0, atol=TOL_EQUATE)
    if population == "new":
        np.testing.assert_allclose(f, ref["eapsum_expected_over_n"]["new"], rtol=1e-11)
    if population == "old":
        np.testing.assert_allclose(g, ref["eapsum_expected_over_n"]["old"], rtol=1e-11)


@pytest.mark.parametrize("method", ["true_score", "EAPsum"])
def test_kernels_scale_theta(method: str) -> None:
    want = _reference()["polytomous"]["scale_old"][
        "true_score" if method == "true_score" else "eapsum"
    ]
    theta, theta_se = _scale_theta(_spec("polytomous", "old"), method)  # type: ignore[arg-type]
    tol = TOL_POLY_THETA if method == "true_score" else 1e-10
    np.testing.assert_allclose(theta, _array(want["theta"]), rtol=tol, atol=tol)
    np.testing.assert_allclose(theta_se, _array(want["theta_se"]), rtol=tol, atol=tol)


@cache
def _fixed(case: str, side: str) -> FitResult:
    """A FitResult with every parameter fixed at the reference values (latent N(0, 1))."""
    ref = _reference()
    form = ref[case][side]
    items = list(form["items"])
    n_cats = {i: _n_categories(form["params"][i]) for i in items}
    spec = Model.from_items({i: _item_model(str(form["itemtypes"][i]), n_cats[i]) for i in items})
    rng = np.random.default_rng(51)
    # Every category observed; the responses do not matter with all parameters fixed.
    data = pd.DataFrame({i: rng.permutation(np.arange(300) % n_cats[i]) for i in items})
    table = spec.parameter_table(data)
    values = []
    for item, param in zip(table["item"], table["param"], strict=True):
        if item == "__latent__":
            values.append({"mean": 0.0, "var": 1.0}[str(param)])
        else:
            values.append(float(form["params"][item][param]))
    table["value"] = np.asarray(values, dtype=np.float64)
    table["free"] = False
    meta = ref["meta"]
    quad = Quadrature.equispaced(int(meta["quadpts"]), bounds=tuple(meta["theta_lim"]))
    return fit(spec, data, table=table, quadrature=quad, se=None)


@cache
def _link(case: str) -> LinkResult:
    """A ``LinkResult`` from the new form's scale to the old one's with the reference constants."""
    ref = _reference()[case]
    return LinkResult(
        A=float(ref["link"]["A"]),
        B=float(ref["link"]["B"]),
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
        items=pd.DataFrame({"item": ref["new"]["items"]}),
        _source_group="all",
        _target_group="all",
    )


@pytest.mark.slow
def test_fixed_fits_reproduce_the_grid() -> None:
    ref = _reference()
    nodes = np.asarray(ref["nodes"])
    weights = np.asarray(ref["weights"])
    from flexpyirt.linking import _fitted_grid

    res = _fixed("dichotomous", "new")
    got_nodes, got_weights = _fitted_grid(res, "all", 0)
    np.testing.assert_allclose(got_nodes, nodes, rtol=0, atol=1e-14)
    np.testing.assert_allclose(got_weights, weights, rtol=1e-12, atol=0)


@pytest.mark.slow
def test_dichotomous_true_score_against_equateirt_and_snsequate() -> None:
    ref = _reference()["dichotomous"]
    new, old = _fixed("dichotomous", "new"), _fixed("dichotomous", "old")
    res = irt_equate(new, old, link=_link("dichotomous"))
    table = res.table
    assert list(table["score"]) == list(range(21))
    np.testing.assert_allclose(res.new_range[0], ref["sum_guessing"]["new"], atol=1e-12)
    np.testing.assert_allclose(res.old_range[0], ref["sum_guessing"]["old"], atol=1e-12)

    # Interior scores (sum of guessing < x < 20): equateIRT's Newton solution.
    e = ref["tse_equateirt"]
    scores = np.asarray(e["score"])
    inside = scores < 20
    got = table.set_index("score").loc[scores[inside]]
    np.testing.assert_allclose(got["equated"], np.asarray(e["equated"])[inside], atol=TOL_TSE)
    a, b = ref["link"]["A"], ref["link"]["B"]
    theta_old = a * np.asarray(e["theta_new_scale"])[inside] + b
    np.testing.assert_allclose(got["theta"], theta_old, atol=TOL_TSE)

    # Every score: SNSequate (exact below the guessing sum and at 20; loose inside).
    s = ref["tse_snsequate"]
    sns = _array(s["equated"])
    exact = (table["score"] <= ref["sum_guessing"]["new"]) | (table["score"] == 20)
    assert int(exact.sum()) == 4
    np.testing.assert_allclose(table["equated"][exact], sns[exact], rtol=0, atol=TOL_EXACT)
    np.testing.assert_allclose(
        table["equated"][~exact], sns[~exact], rtol=0, atol=TOL_TSE_SNSEQUATE_INTERIOR
    )
    assert np.all(np.isneginf(table["theta"][table["score"] <= 2]))
    assert np.isposinf(table["theta"].iloc[-1])


@pytest.mark.slow
@pytest.mark.parametrize("population", ["new", "old", "synthetic"])
def test_dichotomous_observed_score_against_equateirt(population: str) -> None:
    ref = _reference()["dichotomous"]["ose_equateirt"][population]
    new, old = _fixed("dichotomous", "new"), _fixed("dichotomous", "old")
    weight = 0.5 if population == "synthetic" else None
    res = irt_equate(
        new,
        old,
        method="observed_score",
        link=_link("dichotomous"),
        population=population,  # type: ignore[arg-type]
        synthetic_weight=weight,
    )
    assert res.synthetic_weight == pytest.approx(float(ref["w"]))
    np.testing.assert_allclose(res.table["equated"], ref["equated"], rtol=0, atol=TOL_EXACT)


@pytest.mark.slow
def test_polytomous_true_score_against_mirt() -> None:
    ref = _reference()["polytomous"]
    new, old = _fixed("polytomous", "new"), _fixed("polytomous", "old")
    res = irt_equate(new, old, link=_link("polytomous"))
    tse = ref["tse"]
    np.testing.assert_array_equal(res.table["score"], tse["score"])
    np.testing.assert_allclose(res.table["theta"], _array(tse["theta"]), atol=TOL_POLY_THETA)
    np.testing.assert_allclose(res.table["equated"], _array(tse["equated"]), atol=TOL_POLY_THETA)
    assert res.new_range == pytest.approx((0.0, 18.0), abs=1e-12)
    assert res.old_range[0] == pytest.approx(ref["sum_guessing"]["old"], abs=1e-12)


@pytest.mark.slow
def test_polytomous_distributions_against_mirt() -> None:
    ref = _reference()["polytomous"]
    new, old = _fixed("polytomous", "new"), _fixed("polytomous", "old")
    for res, side in ((new, "new"), (old, "old")):
        got = score_distribution(res)["probability"].to_numpy()
        want = np.asarray(ref["eapsum_expected_over_n"][side])
        np.testing.assert_allclose(got, want, rtol=1e-11, atol=1e-16)


@pytest.mark.slow
@pytest.mark.parametrize("population", ["new", "old", "synthetic"])
def test_polytomous_observed_score_against_equate(population: str) -> None:
    ref = _reference()["polytomous"]["ose"][population]
    new, old = _fixed("polytomous", "new"), _fixed("polytomous", "old")
    weight = 0.5 if population == "synthetic" else None
    res = irt_equate(
        new,
        old,
        method="observed_score",
        link=_link("polytomous"),
        population=population,  # type: ignore[arg-type]
        synthetic_weight=weight,
    )
    assert res.new_distribution is not None and res.old_distribution is not None
    np.testing.assert_allclose(res.new_distribution["probability"], ref["f"], rtol=1e-11)
    np.testing.assert_allclose(res.old_distribution["probability"], ref["g"], rtol=1e-11)
    np.testing.assert_allclose(res.table["equated"], ref["equated"], rtol=0, atol=TOL_EQUATE)


@pytest.mark.slow
@pytest.mark.parametrize("method", ["true_score", "EAPsum"])
def test_scale_table_against_mirt(method: str) -> None:
    ref = _reference()["polytomous"]["scale_old"]
    want = ref["true_score" if method == "true_score" else "eapsum"]
    got = scale_table(
        _fixed("polytomous", "old"),
        slope=ref["slope"],
        intercept=ref["intercept"],
        lowest=ref["lowest"],
        highest=ref["highest"],
        method=method,  # type: ignore[arg-type]
    )
    np.testing.assert_array_equal(got["score"], want["score"])
    tol = TOL_POLY_THETA if method == "true_score" else 1e-10
    for column in ("theta", "theta_se", "scale_score", "scale_score_se"):
        np.testing.assert_allclose(
            got[column], _array(want[column]), rtol=tol, atol=tol, err_msg=column
        )
    np.testing.assert_array_equal(got["scale_score_rounded"], _array(want["scale_score_rounded"]))
