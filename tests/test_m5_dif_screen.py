"""Unit tests for the DIF workflow: Wald tests, dif_screen() and dif_effect_sizes() (§16.20)."""

from __future__ import annotations

from functools import cache

import numpy as np
import pandas as pd
import pytest

from flexpyirt import (
    BAEM,
    GRM,
    DIFScreenResult,
    FitResult,
    LogNormalPrior,
    Model,
    Normal,
    Quadrature,
    TwoPL,
    dif_effect_sizes,
    dif_screen,
    expected_score,
    fit,
)
from flexpyirt._dif import wald_statistic
from flexpyirt.errors import ConvergenceWarning, SpecificationError

QUAD = Quadrature.equispaced(61, bounds=(-6, 6))
METHOD = BAEM(tol=1e-8, max_iter=5000)
ITEMS = [f"i{k}" for k in range(6)]


@cache
def _data() -> tuple[pd.DataFrame, np.ndarray]:
    """Two groups of 700, six 2PL items, intercept DIF on i5, focal N(0.4, 1)."""
    rng = np.random.default_rng(1)
    a = rng.uniform(0.8, 1.6, 6)
    d = rng.normal(0.0, 0.8, 6)
    blocks = []
    for mu, dd in ((0.0, d), (0.4, d + np.r_[0, 0, 0, 0, 0, 0.9])):
        theta = rng.normal(mu, 1.0, (700, 1))
        blocks.append((rng.random((700, 6)) < 1 / (1 + np.exp(-(theta * a + dd)))).astype(float))
    return pd.DataFrame(np.vstack(blocks), columns=ITEMS), np.repeat(["r", "f"], 700)


def _spec() -> Model:
    return Model.uniform(TwoPL(), items=ITEMS).with_groups(r=Normal.standard(), f=Normal.free())


@cache
def _anchored(se: str | None = "oakes") -> FitResult:
    df, grp = _data()
    spec = _spec().equal_across_groups(items=["i0", "i1"])
    return fit(spec, df, group=grp, method=METHOD, quadrature=QUAD, se=se)  # type: ignore[arg-type]


@cache
def _screen(strategy: str, test: str = "likelihood_ratio") -> DIFScreenResult:
    df, grp = _data()
    anchors = ["i0", "i1"] if strategy == "sequential_constrain" else None
    return dif_screen(
        _spec(),
        df,
        group=grp,
        anchor_items=anchors,
        strategy=strategy,  # type: ignore[arg-type]
        test=test,  # type: ignore[arg-type]
        method=METHOD,
        quadrature=QUAD,
    )


def test_wald_statistic_by_hand() -> None:
    eta = np.array([1.0, 0.5, 1.4, 0.1, 9.0])
    rng = np.random.default_rng(0)
    m = rng.normal(size=(5, 5))
    vcov = m @ m.T + np.eye(5)
    table = pd.DataFrame(
        {
            "group": ["r", "r", "f", "f", "g"],
            "item": ["x"] * 5,
            "param": ["a", "d", "a", "d", "d"],
            "free_id": [0, 1, 2, 3, 1],
        }
    )
    stat, df = wald_statistic(eta, vcov, table, np.arange(5))
    contrast = np.array([[1.0, 0, -1, 0, 0], [0, 1, 0, -1, 0]])
    diff = contrast @ eta
    expected = diff @ np.linalg.inv(contrast @ vcov @ contrast.T) @ diff
    assert df == 2
    assert stat == pytest.approx(expected, rel=1e-12)
    stat, df = wald_statistic(eta, vcov, table, np.array([1, 4]))
    assert np.isnan(stat) and df == 0


@pytest.mark.slow
def test_wald_dif_close_to_likelihood_ratio() -> None:
    res = _anchored()
    df, grp = _data()
    wald = res.dif(items=["i4", "i5"], test="wald")
    assert list(wald.columns) == ["item", "x2", "df", "p", "adj_p"]
    lr = res.dif(df, group=grp, items=["i4", "i5"])
    np.testing.assert_array_equal(wald["df"], lr["df"])
    assert wald["p"].iloc[1] < 1e-6
    # Asymptotically equivalent; here within 25% on the DIF item.
    assert wald["x2"].iloc[1] == pytest.approx(lr["x2"].iloc[1], rel=0.25)
    one = res.dif(items=["i5"], params=["d"], test="wald")
    assert int(one["df"].iloc[0]) == 1


def test_wald_dif_errors() -> None:
    res = _anchored()
    with pytest.raises(SpecificationError, match="already equal"):
        res.dif(items=["i0"], test="wald")
    with pytest.raises(SpecificationError, match="scheme='add'"):
        res.dif(items=["i5"], test="wald", scheme="drop")
    with pytest.raises(SpecificationError, match="test="):
        res.dif(items=["i5"], test="score")  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match="needs data"):
        res.dif(items=["i5"])
    with pytest.raises(SpecificationError, match="se="):
        _anchored(None).dif(items=["i5"], test="wald")


@pytest.mark.slow
def test_all_others_matches_manual_drop_tests() -> None:
    out = _screen("all_others")
    assert list(out.table.columns) == [
        "item",
        "x2",
        "df",
        "p",
        "adj_p",
        "flagged",
        "converged",
    ]
    assert "i5" in out.flagged_items
    assert out.n_rounds == 1 and out.converged
    assert out.anchor_items == [i for i in ITEMS if i not in out.flagged_items]
    # The same tests from FitResult.dif(scheme="drop") on the fully invariant model.
    df, grp = _data()
    inv = fit(
        _spec().equal_across_groups(items=None),
        df,
        group=grp,
        method=METHOD,
        quadrature=QUAD,
        se=None,
    )
    manual = inv.dif(df, group=grp, items=ITEMS, scheme="drop", p_adjust="BH")
    np.testing.assert_allclose(out.table["x2"], manual["x2"], atol=1e-5)
    np.testing.assert_array_equal(out.table["df"], manual["df"])
    np.testing.assert_allclose(out.table["adj_p"], manual["adj_p"], rtol=1e-4, atol=1e-8)
    np.testing.assert_array_equal(out.table["flagged"], manual["adj_p"] < 0.05)
    # The final fit frees exactly the flagged items.
    params = out.fit.parameters
    for item in ITEMS:
        rows = params[(params["item"] == item) & params["free"]]
        n_ids = rows.groupby("param")["free_id"].nunique()
        assert (n_ids == (2 if item in out.flagged_items else 1)).all()


@pytest.mark.slow
def test_sequential_free_history_and_final_table() -> None:
    out = _screen("sequential_free")
    hist = out.history
    assert list(hist.columns) == ["round", "item", "retested", "x2", "df", "p", "adj_p", "flagged"]
    assert hist["round"].max() == out.n_rounds
    first = hist[hist["round"] == 1]
    assert first["retested"].all()
    last = hist[hist["round"] == out.n_rounds]
    assert set(last.loc[last["flagged"], "item"]) == set(out.flagged_items)
    if out.n_rounds > 1:
        later = hist[hist["round"] == 2]
        flagged_1 = set(first.loc[first["flagged"], "item"])
        assert set(later.loc[~later["retested"], "item"]) == flagged_1
    # Unflagged items in the final table are the last round's retests (same final anchors).
    final = out.table.set_index("item")
    for item in ITEMS:
        if item not in out.flagged_items and out.n_rounds > 1:
            row = last.set_index("item").loc[item]
            assert final.loc[item, "x2"] == pytest.approx(row["x2"], abs=1e-9)


@pytest.mark.slow
def test_sequential_constrain_wald_uses_one_fit_per_round() -> None:
    out = _screen("sequential_constrain", "wald")
    assert out.test == "wald"
    assert list(out.table["item"]) == ITEMS[2:]
    assert "i5" in out.flagged_items
    assert {"i0", "i1"} <= set(out.anchor_items)
    assert out.fit.vcov is not None
    # Round 1 is FitResult.dif(test="wald") on the model with only the anchors equal.
    round1 = out.history[out.history["round"] == 1].reset_index(drop=True)
    manual = _anchored().dif(items=ITEMS[2:], test="wald", p_adjust="BH")
    np.testing.assert_allclose(round1["x2"], manual["x2"], rtol=1e-4)


@pytest.mark.slow
def test_max_iter_warns() -> None:
    df, grp = _data()
    with pytest.warns(ConvergenceWarning, match="max_iter"):
        out = dif_screen(
            _spec(),
            df,
            group=grp,
            strategy="sequential_free",
            max_iter=1,
            method=METHOD,
            quadrature=QUAD,
        )
    assert not out.converged and out.n_rounds == 1


def test_dif_screen_errors() -> None:
    df, grp = _data()
    spec = _spec()
    single = Model.uniform(TwoPL(), items=ITEMS)
    with pytest.raises(SpecificationError, match="multiple-group"):
        dif_screen(single, df, group=grp)
    with pytest.raises(SpecificationError, match="priors"):
        dif_screen(spec.prior(param="a", dist=LogNormalPrior(0.0, 0.5)), df, group=grp)
    with pytest.raises(SpecificationError, match="across items"):
        dif_screen(spec.equal_across_items(items=["i0", "i1"], params=["a"]), df, group=grp)
    with pytest.raises(SpecificationError, match="strategy"):
        dif_screen(spec, df, group=grp, strategy="lasso")  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match="test"):
        dif_screen(spec, df, group=grp, test="score")  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match="p_adjust"):
        dif_screen(spec, df, group=grp, p_adjust="bonferroni")  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match="alpha"):
        dif_screen(spec, df, group=grp, alpha=1.5)
    with pytest.raises(SpecificationError, match="max_iter"):
        dif_screen(spec, df, group=grp, max_iter=0)
    with pytest.raises(SpecificationError, match="not a str"):
        dif_screen(spec, df, group=grp, items="i5")
    with pytest.raises(SpecificationError, match="Unknown items"):
        dif_screen(spec, df, group=grp, anchor_items=["nope"])
    with pytest.raises(SpecificationError, match="both tested and anchors"):
        dif_screen(spec, df, group=grp, items=["i1", "i2"], anchor_items=["i1"])
    with pytest.raises(SpecificationError, match="anchor_items"):
        dif_screen(spec, df, group=grp, strategy="sequential_constrain")
    with pytest.raises(SpecificationError, match="none of the parameters"):
        dif_screen(spec, df, group=grp, params=["d1"])
    with pytest.raises(SpecificationError, match="not a str"):
        dif_screen(spec, df, group=grp, params="d")


# ---- effect sizes ----------------------------------------------------------------------------


def _fixed_two_group(
    shift: float, slope_f: float = 1.2
) -> tuple[FitResult, pd.DataFrame, np.ndarray]:
    """A fixed 2PL + GRM two-group fit with DIF on q2 only (no estimation)."""
    rng = np.random.default_rng(3)
    items = ["q0", "q1", "q2"]
    spec = Model.from_items({"q0": TwoPL(), "q1": GRM(n_categories=3), "q2": TwoPL()}).with_groups(
        ref=Normal.standard(), foc=Normal(mean=0.5, var=1.5)
    )
    y = rng.integers(0, 2, size=(400, 3)).astype(float)
    y[:, 1] = rng.integers(0, 3, size=400)
    grp = np.repeat(["ref", "foc"], 200)
    df = pd.DataFrame(y, columns=items)
    table = spec.parameter_table(df, group=grp)
    values = {
        ("q0", "a"): 1.1,
        ("q0", "d"): 0.2,
        ("q1", "a"): 1.4,
        ("q1", "d1"): 1.0,
        ("q1", "d2"): -0.8,
        ("q2", "a"): 1.2,
        ("q2", "d"): -0.3,
    }
    for k, (g, item, param) in enumerate(
        zip(table["group"], table["item"], table["param"], strict=True)
    ):
        if item == "__latent__":
            continue
        v = values[(item, param)]
        if g == "foc" and item == "q2":
            v = v + shift if param == "d" else slope_f
        table.loc[k, "value"] = v
    table["free"] = False
    res = fit(spec, df, group=grp, table=table, quadrature=QUAD, se=None)
    return res, df, grp


def test_effect_sizes_by_hand() -> None:
    res, _, _ = _fixed_two_group(shift=-0.7)
    out = dif_effect_sizes(res)
    assert list(out.columns) == [
        "group",
        "level",
        "item",
        "signed_difference",
        "unsigned_difference",
    ]
    assert list(out["level"]) == ["item"] * 3 + ["test"]
    nodes = QUAD.nodes()
    w = np.exp(-0.5 * (nodes - 0.5) ** 2 / 1.5)
    w /= w.sum()
    e_ref = expected_score(res, nodes, group="ref")
    e_foc = expected_score(res, nodes, group="foc")
    diff = (e_foc["q2"] - e_ref["q2"]).to_numpy()
    q2 = out[out["item"] == "q2"].iloc[0]
    assert q2["signed_difference"] == pytest.approx(np.sum(w * diff), abs=1e-12)
    assert q2["unsigned_difference"] == pytest.approx(np.sum(w * np.abs(diff)), abs=1e-12)
    np.testing.assert_allclose(out["signed_difference"].iloc[:2], 0.0, atol=1e-14)
    test = out[out["level"] == "test"].iloc[0]
    assert test["signed_difference"] == pytest.approx(out["signed_difference"].iloc[:3].sum())
    ref = dif_effect_sizes(res, density="reference", items=["q2"])
    w_ref = np.exp(-0.5 * nodes**2)
    w_ref /= w_ref.sum()
    assert ref["signed_difference"].iloc[0] == pytest.approx(np.sum(w_ref * diff), abs=1e-12)
    assert len(ref) == 2


def test_effect_sizes_with_data() -> None:
    res, df, grp = _fixed_two_group(shift=-0.7)
    out = dif_effect_sizes(res, data=df, group=grp, focal="foc")
    assert {"expected_score_standardized_difference", "rmsd_reference", "rmsd_focal"} <= set(
        out.columns
    )
    item = out[out["level"] == "item"].set_index("item")
    assert item.loc["q0", "expected_score_standardized_difference"] == 0.0  # no difference
    assert (item["expected_score_standardized_difference"].loc[["q2"]] < 0).all()
    assert (item[["rmsd_reference", "rmsd_focal"]] >= 0).all().all()
    assert np.isnan(out[out["level"] == "test"]["rmsd_focal"].iloc[0])
    # ESSD by hand at the focal EAP scores.
    theta = res.score(df, group=grp)["theta"].to_numpy()[grp == "foc"]
    e_f = expected_score(res, theta, group="foc")["q2"].to_numpy()
    e_r = expected_score(res, theta, group="ref")["q2"].to_numpy()
    essd = (e_f.mean() - e_r.mean()) / np.sqrt((e_f.var(ddof=1) + e_r.var(ddof=1)) / 2)
    assert item.loc["q2", "expected_score_standardized_difference"] == pytest.approx(essd)


def test_effect_size_errors() -> None:
    res, df, _ = _fixed_two_group(shift=0.0)
    with pytest.raises(SpecificationError, match="reference group"):
        dif_effect_sizes(res, focal="ref")
    with pytest.raises(SpecificationError, match="Unknown focal"):
        dif_effect_sizes(res, focal=["nope"])
    with pytest.raises(SpecificationError, match="density"):
        dif_effect_sizes(res, density="marginal")  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match="group="):
        dif_effect_sizes(res, data=df)
    with pytest.raises(SpecificationError, match="Unknown items"):
        dif_effect_sizes(res, items=["zz"])
    with pytest.raises(SpecificationError, match="FitResult"):
        dif_effect_sizes("x")  # type: ignore[arg-type]
    single = fit(
        Model.uniform(TwoPL(), items=["q0", "q2"]), df[["q0", "q2"]], quadrature=QUAD, se=None
    )
    with pytest.raises(SpecificationError, match="multiple-group"):
        dif_effect_sizes(single)
