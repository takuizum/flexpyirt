"""Unit tests for likelihood-ratio DIF tests (DESIGN.md §16.2)."""

from __future__ import annotations

import warnings
from functools import cache

import numpy as np
import pandas as pd
import pytest

from flexpyirt import (
    BAEM,
    GRM,
    EmpiricalHistogram,
    FitResult,
    LogNormalPrior,
    Model,
    Normal,
    Quadrature,
    TwoPL,
    fit,
)
from flexpyirt._dif import adjust_p
from flexpyirt.errors import (
    ConvergenceWarning,
    FlexpyirtWarning,
    IdentificationError,
    SpecificationError,
)

WIDE = Quadrature.equispaced(81, bounds=(-8, 8))
METHOD = BAEM(tol=1e-8, max_iter=5000)


def test_adjust_p_matches_r() -> None:
    """Values from R 4.4 ``p.adjust(c(.01, .04, .03, NA, .2, .005, .5), method)``."""
    p = np.array([0.01, 0.04, 0.03, np.nan, 0.2, 0.005, 0.5])
    holm = [0.05, 0.12, 0.12, np.nan, 0.4, 0.03, 0.5]
    bh = [0.03, 0.06, 0.06, np.nan, 0.24, 0.03, 0.5]
    np.testing.assert_allclose(adjust_p(p, "holm"), holm, rtol=1e-15)
    np.testing.assert_allclose(adjust_p(p, "BH"), bh, rtol=1e-15)
    np.testing.assert_array_equal(adjust_p(p, "none"), p)
    assert np.isnan(adjust_p(np.array([np.nan]), "holm")).all()


def _sim(n_per: int = 700, seed: int = 4, n_categories: int = 2) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    j = 6
    a = rng.uniform(0.8, 1.6, size=j)
    if n_categories == 2:
        d = rng.normal(0.0, 0.8, size=j)
        d_f = d.copy()
        d_f[5] += 0.9  # DIF on the last item
    else:
        d = np.sort(rng.normal(0.0, 1.0, size=(j, n_categories - 1)), axis=1)[:, ::-1]
        d_f = d.copy()
        d_f[5] += 0.9
    out = []
    for dd, mu in ((d, 0.0), (d_f, 0.4)):
        theta = rng.normal(mu, 1.0, size=(n_per, 1))
        if n_categories == 2:
            y = (rng.random((n_per, j)) < 1 / (1 + np.exp(-(theta * a + dd)))).astype(float)
        else:
            z = theta[:, :, None] * a[None, :, None] + dd[None]
            y = (rng.random((n_per, j))[:, :, None] < 1 / (1 + np.exp(-z))).sum(axis=2)
            y = y.astype(float)
        out.append(y)
    df = pd.DataFrame(np.vstack(out), columns=[f"i{k}" for k in range(j)])
    return df, np.repeat(["r", "f"], n_per)


@cache
def _base(n_categories: int = 2) -> tuple[FitResult, pd.DataFrame, np.ndarray]:
    df, grp = _sim(n_categories=n_categories)
    item = TwoPL() if n_categories == 2 else GRM(n_categories=n_categories)
    spec = (
        Model.uniform(item, items=df.columns)
        .with_groups(r=Normal.standard(), f=Normal.free())
        .equal_across_groups(items=["i0", "i1", "i2"])
    )
    return fit(spec, df, group=grp, method=METHOD, quadrature=WIDE, se=None), df, grp


def test_add_scheme_flags_the_dif_item_and_is_consistent() -> None:
    res, df, grp = _base()
    out = res.dif(df, group=grp, items=["i3", "i4", "i5"])
    assert list(out.columns) == [
        "item",
        "converged",
        "aic",
        "sabic",
        "hq",
        "bic",
        "x2",
        "df",
        "p",
        "adj_p",
    ]
    assert out["converged"].all()
    assert (out["df"] == 2).all()
    np.testing.assert_allclose(out["aic"], -out["x2"] + 2 * out["df"], atol=1e-9)
    n = df.shape[0]
    np.testing.assert_allclose(out["bic"], -out["x2"] + out["df"] * np.log(n), atol=1e-9)
    by_item = dict(zip(out["item"], out["p"], strict=True))
    assert by_item["i5"] < 1e-3
    assert (out["adj_p"] >= out["p"]).all()
    # Refitting the constrained model by hand gives the same statistic.
    table = res.parameters[list(res.parameters.columns[:10])].copy()
    rows = (table["item"] == "i5") & table["free"]
    for _, idx in table[rows].groupby("param").groups.items():
        table.loc[idx, "free_id"] = int(table.loc[idx, "free_id"].min())
    con = fit(res._model, df, group=grp, table=table, method=METHOD, quadrature=WIDE, se=None)
    x2 = 2 * (res.loglik - con.loglik)
    assert float(out["x2"].iloc[2]) == pytest.approx(x2, abs=1e-6)


def test_drop_scheme_and_params_subset() -> None:
    res, df, grp = _base()
    out = res.dif(df, group=grp, items=["i1", "i2"], params=["d"], scheme="drop")
    assert (out["df"] == 1).all()
    assert (out["x2"] >= 0).all()
    np.testing.assert_allclose(out["aic"], -out["x2"] + 2.0, atol=1e-9)


def test_grm_dif_runs() -> None:
    res, df, grp = _base(n_categories=3)
    out = res.dif(df, group=grp, items=["i5"], params=["d1", "d2"])
    assert int(out["df"].iloc[0]) == 2
    assert out["p"].iloc[0] < 1e-3


def test_errors() -> None:
    res, df, grp = _base()
    with pytest.raises(SpecificationError, match="already equal"):
        res.dif(df, group=grp, items=["i0"])
    with pytest.raises(SpecificationError, match="already"):
        res.dif(df, group=grp, items=["i5"], scheme="drop")
    with pytest.raises(SpecificationError, match="Unknown items"):
        res.dif(df, group=grp, items=["nope"])
    with pytest.raises(SpecificationError, match="no parameters"):
        res.dif(df, group=grp, items=["i5"], params=["g"])
    with pytest.raises(SpecificationError, match="not a str"):
        res.dif(df, group=grp, items="i5")
    with pytest.raises(SpecificationError, match="scheme"):
        res.dif(df, group=grp, items=["i5"], scheme="both")  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match="p_adjust"):
        res.dif(df, group=grp, items=["i5"], p_adjust="fdr")  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match="at least one item"):
        res.dif(df, group=grp, items=[])
    single = fit(Model.uniform(TwoPL(), items=df.columns), df, method=METHOD, se=None)
    with pytest.raises(SpecificationError, match="multiple-group"):
        single.dif(df, group=grp, items=["i5"])


def test_freeing_all_anchors_is_unidentified() -> None:
    df, grp = _sim()
    spec = (
        Model.uniform(TwoPL(), items=df.columns)
        .with_groups(r=Normal.standard(), f=Normal.free())
        .equal_across_groups(items=["i0"])
    )
    with pytest.warns(FlexpyirtWarning, match="single anchor"):
        res = fit(spec, df, group=grp, method=METHOD, quadrature=WIDE, se=None)
    with pytest.raises(IdentificationError):
        res.dif(df, group=grp, items=["i0"], scheme="drop")


def test_map_fits_are_rejected() -> None:
    df, grp = _sim()
    spec = (
        Model.uniform(TwoPL(), items=df.columns)
        .with_groups(r=Normal.standard(), f=Normal.free())
        .equal_across_groups(items=["i0", "i1", "i2"])
        .prior(param="a", dist=LogNormalPrior(0.0, 0.5))
    )
    res = fit(spec, df, group=grp, method=METHOD, quadrature=WIDE, se=None)
    with pytest.raises(SpecificationError, match="priors"):
        res.dif(df, group=grp, items=["i5"])


def test_eh_dif_runs() -> None:
    df, grp = _sim()
    spec = (
        Model.uniform(TwoPL(), items=df.columns)
        .with_groups(r=EmpiricalHistogram(), f=EmpiricalHistogram())
        .equal_across_groups(items=["i0", "i1", "i2"])
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)  # EH creeps (VALIDATION.md V6)
        res = fit(spec, df, group=grp, method=BAEM(tol=1e-4, max_iter=500), se=None)
    out = res.dif(df, group=grp, items=["i5"], params=["d"])
    assert int(out["df"].iloc[0]) == 1
    assert out["x2"].iloc[0] >= 0
