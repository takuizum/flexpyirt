"""Person fit and residual item fit: unit tests and the l_z* null distribution (DESIGN.md §23).

The comparison with R (mirt ``personfit()`` / ``itemfit()``, sirt, the R recomputation of l_z*)
is in ``test_mirt_v53_person_fit.py``. Here: closed forms for 2PL items computed independently in
NumPy, argument checks, and a simulation showing that l_z* is close to N(0, 1) under the model
where l_z is not (Snijders, 2001).
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from flexpyirt import GRM, FitResult, Model, Normal, TwoPL, fit
from flexpyirt.errors import SpecificationError

J = 20
A = np.linspace(0.8, 2.0, J)
D = np.linspace(-1.5, 1.5, J)


def _simulate(n: int, seed: int, *, missing: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=n)
    y = (rng.uniform(size=(n, J)) < expit(theta[:, None] * A + D)).astype(float)
    y[rng.uniform(size=y.shape) < missing] = np.nan
    return pd.DataFrame(y, columns=[f"i{j:02d}" for j in range(J)])


@cache
def _true_model_fit(n: int = 400, seed: int = 0, missing: float = 0.05) -> FitResult:
    """A 2PL ``FitResult`` with every parameter fixed at the generating values."""
    data = _simulate(n, seed, missing=missing)
    spec = Model.uniform(TwoPL(), items=list(data.columns))
    table = spec.parameter_table(data)
    for j, item in enumerate(data.columns):
        table.loc[(table["item"] == item) & (table["param"] == "a"), "value"] = A[j]
        table.loc[(table["item"] == item) & (table["param"] == "d"), "value"] = D[j]
    table["free"] = False
    return fit(spec, data, table=table, se=None)


def _floored(var: np.ndarray) -> np.ndarray:
    """mirt's guard: W = sqrt(1e-5) where W^2 < 1e-5 (DESIGN.md §23)."""
    return np.where(var**2 < 1e-5, np.sqrt(1e-5), var)


def _closed_form(y: np.ndarray, theta: np.ndarray, method: str) -> dict[str, np.ndarray]:
    """2PL l_z, l_z* (Snijders, 2001, eqs. for w_j and r_j) and Wright–Masters mean squares."""
    p = expit(theta[:, None] * A + D)
    q = 1.0 - p
    obs = ~np.isnan(y)
    yy = np.where(obs, y, 0.0)
    w = np.log(p / q)
    dp = A * p * q
    r = dp / (p * q)
    l0 = np.where(obs, yy * np.log(p) + (1 - yy) * np.log(q), 0).sum(1)
    e0 = np.where(obs, p * np.log(p) + q * np.log(q), 0).sum(1)
    v0 = np.where(obs, p * q * w**2, 0).sum(1)
    cn = np.where(obs, dp * w, 0).sum(1) / np.where(obs, dp * r, 0).sum(1)
    w_tilde = w - cn[:, None] * r
    tau2 = np.where(obs, w_tilde**2 * p * q, 0).sum(1)
    info = np.where(obs, dp * r, 0).sum(1)
    if method == "ML":
        r0 = np.zeros_like(theta)
    elif method == "MAP":
        r0 = -theta
    else:  # WLE, 2PL: I'(theta) = sum_j a_j^3 p q (q - p)
        r0 = np.where(obs, A**3 * p * q * (q - p), 0).sum(1) / (2 * info)
    kurt = p * q * (q**3 + p**3)
    var = _floored(p * q)
    resid2 = np.where(obs, (yy - p) ** 2, np.nan)
    n = obs.sum(1)
    outfit = np.nansum(resid2 / var, 1) / n
    infit = np.nansum(resid2, 1) / np.where(obs, var, 0).sum(1)
    q_out = np.sqrt(np.where(obs, kurt / var**2, 0).sum(1) / n**2 - 1 / n)
    q_in = np.sqrt(np.where(obs, kurt - var**2, 0).sum(1)) / np.where(obs, var, 0).sum(1)
    return {
        "lz": (l0 - e0) / np.sqrt(v0),
        "lz_star": (l0 - e0 + cn * r0) / np.sqrt(tau2),
        "infit": infit,
        "z_infit": (np.cbrt(infit) - 1) * 3 / np.minimum(q_in, 1.4142)
        + np.minimum(q_in, 1.4142) / 3,
        "outfit": outfit,
        "z_outfit": (np.cbrt(outfit) - 1) * 3 / np.minimum(q_out, 1.4142)
        + np.minimum(q_out, 1.4142) / 3,
        "n_items": n,
    }


@pytest.mark.parametrize("method", ["ML", "MAP", "WLE"])
def test_person_fit_matches_2pl_closed_form(method: str) -> None:
    res = _true_model_fit()
    data = _simulate(400, 0, missing=0.05)
    got = res.person_fit(
        data,
        method=method,  # type: ignore[arg-type]
        statistics=("lz", "lz_star", "infit", "outfit"),
    )
    theta = got["theta"].to_numpy()
    finite = np.isfinite(theta)
    want = _closed_form(data.to_numpy(), np.where(finite, theta, 0.0), method)
    for col in ("lz", "lz_star", "infit", "z_infit", "outfit", "z_outfit"):
        np.testing.assert_allclose(got[col].to_numpy()[finite], want[col][finite], rtol=1e-10)
        assert np.isnan(got[col].to_numpy()[~finite]).all()
    np.testing.assert_array_equal(got["n_items"].to_numpy(), want["n_items"])


def test_item_fit_matches_2pl_closed_form() -> None:
    """Item infit / outfit over respondents, from the same residual moments."""
    res = _true_model_fit()
    data = _simulate(400, 0, missing=0.05)
    theta = res.score(data, method="WLE")["theta"].to_numpy()
    y = data.to_numpy()
    p = expit(theta[:, None] * A + D)
    kurt = p * (1 - p) * ((1 - p) ** 3 + p**3)
    var = _floored(p * (1 - p))
    obs = ~np.isnan(y)
    resid2 = np.where(obs, (np.nan_to_num(y) - p) ** 2, np.nan)
    n = obs.sum(0)
    outfit = np.nansum(resid2 / var, 0) / n
    infit = np.nansum(resid2, 0) / np.where(obs, var, 0).sum(0)
    q_out = np.minimum(np.sqrt(np.where(obs, kurt / var**2, 0).sum(0) / n**2 - 1 / n), 1.4142)
    got = res.item_fit(data, statistics=("infit", "outfit"), method="WLE")
    assert list(got.columns) == ["item", "infit", "z_infit", "outfit", "z_outfit"]
    np.testing.assert_allclose(got["infit"], infit, rtol=1e-10)
    np.testing.assert_allclose(got["outfit"], outfit, rtol=1e-10)
    np.testing.assert_allclose(
        got["z_outfit"], (np.cbrt(outfit) - 1) * 3 / q_out + q_out / 3, rtol=1e-9
    )


@pytest.mark.slow
def test_lz_star_null_distribution_is_standard_normal() -> None:
    """Under the model l_z* has mean ~0 and SD ~1 (ML and WLE); l_z at theta-hat has SD < 1."""
    n = 3000
    res = _true_model_fit(n, 1, 0.0)
    data = _simulate(n, 1)
    for method in ("ML", "WLE"):
        got = res.person_fit(data, method=method, statistics=("lz", "lz_star"))
        lz_star = got["lz_star"].dropna().to_numpy()
        lz = got["lz"].dropna().to_numpy()
        assert lz_star.size > 0.95 * n
        se_mean = 1.0 / np.sqrt(lz_star.size)
        assert abs(lz_star.mean()) < 4 * se_mean + 0.05
        assert 0.93 < lz_star.std() < 1.07
        assert lz.std() < 0.93
        assert lz_star.std() > lz.std()


def test_default_item_fit_is_unchanged_and_columns_merge() -> None:
    res = _true_model_fit(300, 2, 0.0)
    data = _simulate(300, 2)
    default = res.item_fit(data)
    assert list(default.columns) == ["item", "s_x2", "df", "rmsea", "p"]
    both = res.item_fit(data, statistics=["s_x2", "outfit"])
    assert list(both.columns) == ["item", "s_x2", "df", "rmsea", "p", "outfit", "z_outfit"]
    pd.testing.assert_frame_equal(both[default.columns], default)


def test_person_fit_columns_and_index() -> None:
    res = _true_model_fit()
    data = _simulate(400, 0, missing=0.05).iloc[:20]
    data.index = pd.RangeIndex(100, 120)
    got = res.person_fit(data)
    assert list(got.columns) == [
        "theta",
        "lz",
        "infit",
        "z_infit",
        "outfit",
        "z_outfit",
        "n_items",
    ]
    assert got.index.equals(data.index)
    got = res.person_fit(data.to_numpy(), item_names=list(data.columns), statistics="lz")
    assert list(got.columns) == ["theta", "lz", "n_items"]
    empty = data.iloc[:1] * np.nan
    row = res.person_fit(empty, method="WLE").iloc[0]
    assert row["n_items"] == 0 and np.isnan(row["lz"]) and np.isnan(row["infit"])


@pytest.mark.slow
def test_multigroup_grm_runs_per_group() -> None:
    rng = np.random.default_rng(5)
    n, items = 600, [f"g{j}" for j in range(6)]
    groups = np.repeat(["ref", "focal"], n // 2)
    theta = rng.normal(size=n) + np.where(groups == "focal", 0.5, 0.0)
    cuts = np.array([-1.0, 0.0, 1.0])
    y = (theta[:, None, None] * 1.3 + rng.logistic(size=(n, 6, 1)) > cuts).sum(-1)
    data = pd.DataFrame(y.astype(float), columns=items)
    spec = (
        Model.uniform(GRM(n_categories=4), items=items)
        .with_groups(ref=Normal.standard(), focal=Normal.free())
        .equal_across_groups(items=items)
    )
    res = fit(spec, data, group=groups, se=None)
    pf = res.person_fit(data, group=groups, method="MAP", statistics=("lz", "lz_star"))
    assert pf["lz_star"].notna().all()
    item = res.item_fit(data, group=groups, statistics=("s_x2", "infit"))
    assert list(item.columns[:2]) == ["group", "item"]
    assert item.shape[0] == 12 and item["infit"].between(0.5, 1.5).all()


def test_argument_checks() -> None:
    res = _true_model_fit()
    data = _simulate(400, 0, missing=0.05)
    with pytest.raises(SpecificationError, match="lz_star"):
        res.person_fit(data, statistics=("lz_star",))
    with pytest.raises(SpecificationError, match="unknown statistics"):
        res.person_fit(data, statistics=("lz", "u3"))  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match="at least one"):
        res.person_fit(data, statistics=())
    with pytest.raises(SpecificationError, match="point estimate"):
        res.person_fit(data, method="EAPsum")
    with pytest.raises(SpecificationError, match="unknown statistics"):
        res.item_fit(data, statistics=("x2",))  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match="complete data"):
        res.item_fit(data, statistics=("s_x2", "infit"))
