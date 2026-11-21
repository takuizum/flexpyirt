"""Unit, property, safeguard, and EM-evaluation reduction tests for SQUAREM (DESIGN §4.6, §15)."""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from flexpyirt import (
    BAEM,
    GPCM,
    GRM,
    FourPL,
    Model,
    Normal,
    Quadrature,
    Rasch,
    ThreePL,
    TwoPL,
    _core,
    fit,
)
from flexpyirt._compile import compile_problem, prepare
from flexpyirt.errors import ConvergenceWarning, HeywoodWarning, SpecificationError
from flexpyirt.items import ItemModel
from tests import test_mirt_m1, test_mirt_m2, test_mirt_m3_gpcm
from tests._reference import load as load_m1


def _simulate_2pl(a: np.ndarray, d: np.ndarray, n: int, seed: int, sd: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, sd, size=(n, 1))
    p = 1.0 / (1.0 + np.exp(-(theta * a + d)))
    y = (rng.random(p.shape) < p).astype(int)
    return pd.DataFrame(y, columns=[f"i{j}" for j in range(len(a))])


def _simulate_grm(a: np.ndarray, d: np.ndarray, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, 1.0, size=(n, 1))
    cum = 1.0 / (1.0 + np.exp(-(theta[:, :, None] * a[None, :, None] + d[None, :, :])))
    u = rng.random((n, len(a), 1))
    y = (u < cum).sum(axis=2)
    return pd.DataFrame(y, columns=[f"g{j}" for j in range(len(a))])


def _simulate_gpcm(a: np.ndarray, d: np.ndarray, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, 1.0, size=n)
    n_cat = d.shape[1] + 1
    cats = np.arange(n_cat, dtype=float)
    y = np.zeros((n, len(a)), dtype=int)
    for j in range(len(a)):
        intercepts = np.concatenate([[0.0], d[j]])
        eta = np.outer(theta, a[j] * cats) + intercepts[None, :]
        eta -= eta.max(axis=1, keepdims=True)
        probs = np.exp(eta)
        probs /= probs.sum(axis=1, keepdims=True)
        cdf = np.cumsum(probs, axis=1)
        y[:, j] = (rng.random((n, 1)) > cdf).sum(axis=1)
    return pd.DataFrame(y, columns=[f"p{j}" for j in range(len(a))])


# ---- BAEM settings and bounds ---------------------------------------------------------------


def test_baem_squarem_option_default_and_validation() -> None:
    assert BAEM().squarem is False
    assert BAEM(squarem=True).squarem is True
    with pytest.raises(SpecificationError, match="squarem"):
        BAEM(squarem="S3")  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match="squarem"):
        BAEM(squarem=1)  # type: ignore[arg-type]


def test_eta_bounds_matches_parameter_table() -> None:
    df = load_m1("v1_lsat7_2pl").data
    spec = Model.from_items(
        {
            "Item.1": ThreePL(),
            "Item.2": FourPL(),
            "Item.3": TwoPL(),
            "Item.4": Rasch(),
            "Item.5": Rasch(),
        }
    ).with_latent(Normal(free_var=True))
    table = spec.parameter_table(df)
    # Apply custom bounds to one free slope to verify user bounds propagate too.
    row = (table["item"] == "Item.3") & (table["param"] == "a")
    table.loc[row, "lower"] = 0.1
    table.loc[row, "upper"] = 5.0
    prep = prepare(spec, df, group=None, item_names=None, table=table)
    comp = compile_problem(spec, prep.y, prep.group, prep.table, Quadrature.equispaced())
    lo, hi = _core._eta_bounds(comp.problem, jnp.asarray(comp.eta0))

    free_rows = prep.table[prep.table["free"]].drop_duplicates("free_id").sort_values("free_id")
    np.testing.assert_array_equal(np.asarray(lo), free_rows["lower"].to_numpy(dtype=float))
    np.testing.assert_array_equal(np.asarray(hi), free_rows["upper"].to_numpy(dtype=float))


def test_eta_bounds_and_squarem_with_zero_free_parameters() -> None:
    case = load_m1("v1_lsat7_2pl")
    spec = Model.uniform(TwoPL(), items=case.items)
    table = spec.parameter_table(case.data)
    table["free"] = False
    prep = prepare(spec, case.data, group=None, item_names=None, table=table)
    comp = compile_problem(spec, prep.y, prep.group, prep.table, Quadrature.equispaced())
    lo, hi = _core._eta_bounds(comp.problem, jnp.asarray(comp.eta0))
    assert lo.shape == (0,) and hi.shape == (0,)

    plain = fit(spec, case.data, table=table, method=BAEM(squarem=False), se=None)
    sq = fit(spec, case.data, table=table, method=BAEM(squarem=True), se=None)
    assert sq.converged
    assert sq.n_iter == 1
    assert sq.n_safeguards == 0
    assert sq.loglik == pytest.approx(plain.loglik, rel=1e-15)


# ---- history and counters -------------------------------------------------------------------


def test_history_columns_and_counters_plain_vs_squarem() -> None:
    case = load_m1("v1_lsat7_2pl")
    spec = Model.uniform(TwoPL(), items=case.items)

    plain = fit(spec, case.data, method=BAEM(tol=1e-8, squarem=False), se=None)
    assert list(plain.history.columns) == [
        "iter",
        "loglik",
        "max_change",
        "em_evaluations",
        "safeguard",
    ]
    assert plain.n_em_evaluations == plain.n_iter
    assert plain.n_safeguards == 0
    np.testing.assert_array_equal(plain.history["em_evaluations"], plain.history["iter"])
    assert not plain.history["safeguard"].any()

    sq = fit(spec, case.data, method=BAEM(tol=1e-8, squarem=True), se=None)
    assert list(sq.history.columns) == list(plain.history.columns)
    assert sq.converged
    assert sq.n_em_evaluations == int(sq.history["em_evaluations"].iloc[-1])
    assert sq.n_safeguards == int(sq.history["safeguard"].sum())
    inc = np.diff(np.concatenate([[0], sq.history["em_evaluations"].to_numpy()]))
    assert set(inc.tolist()).issubset({2, 3})
    assert int((inc == 2).sum()) == 3 * sq.n_iter - sq.n_em_evaluations


# ---- monotonicity property (Hypothesis) -----------------------------------------------------


@settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    seed=st.integers(0, 2**31 - 1),
    kind=st.sampled_from(["2pl", "rasch", "grm", "3pl", "gpcm"]),
)
def test_squarem_is_monotone(seed: int, kind: str) -> None:
    rng = np.random.default_rng(seed)
    if kind in ("2pl", "rasch", "3pl"):
        a = np.ones(6) if kind == "rasch" else rng.lognormal(0.1, 0.35, 6)
        d = rng.normal(0.0, 1.0, 6)
        df = _simulate_2pl(a, d, 400, seed, sd=1.3 if kind == "rasch" else 1.0)
        if (df.nunique() < 2).any():
            return
        item_models: dict[str, ItemModel] = {"2pl": TwoPL(), "rasch": Rasch(), "3pl": ThreePL()}
        spec = Model.uniform(item_models[kind], items=df.columns)
    elif kind == "grm":
        a = rng.lognormal(0.1, 0.3, 5)
        gaps = rng.uniform(0.5, 1.2, (5, 2))
        d1 = rng.normal(0.8, 0.3, 5)
        d = np.column_stack([d1, d1 - gaps[:, 0], d1 - gaps[:, 0] - gaps[:, 1]])
        df = _simulate_grm(a, d, 500, seed)
        if any(df[c].nunique() < 4 for c in df.columns):
            return
        spec = Model.uniform(GRM(n_cat=4), items=df.columns)
    else:
        a = rng.lognormal(0.1, 0.3, 5)
        d = rng.normal(0.0, 0.8, (5, 3))
        df = _simulate_gpcm(a, d, 500, seed)
        if any(df[c].nunique() < 4 for c in df.columns):
            return
        spec = Model.uniform(GPCM(n_cat=4), items=df.columns)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", HeywoodWarning)
        warnings.simplefilter("ignore", ConvergenceWarning)
        res = fit(spec, df, method=BAEM(tol=1e-7, max_iter=200, squarem=True), se=None)

    ll = res.history["loglik"].to_numpy()
    assert np.all(np.diff(ll) >= -1e-9 * np.abs(ll[1:]))
    assert res.loglik >= ll[-1] - 1e-9 * abs(ll[-1])
    assert res.n_em_evaluations == int(res.history["em_evaluations"].iloc[-1])
    assert res.n_safeguards == int(res.history["safeguard"].sum())

    free = res.parameters[res.parameters["free"]]
    assert (free["estimate"] > free["lower"]).all()
    assert (free["estimate"] < free["upper"]).all()


# ---- safeguard path tests -------------------------------------------------------------------


def test_safeguard_fires_for_threepl_near_lower_bound() -> None:
    """2PL data fitted with 3PL pushes g -> 0 so extrapolation leaves bounds and fires safeguard."""
    a = np.array([1.5, 1.8, 1.3, 2.0, 1.4, 1.7, 1.6, 1.9])
    d = np.array([-1.0, -0.5, 0.0, 0.5, -0.8, 0.3, -0.3, 0.8])
    df = _simulate_2pl(a, d, 1500, seed=42)
    spec = Model.uniform(ThreePL(), items=df.columns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", HeywoodWarning)
        warnings.simplefilter("ignore", ConvergenceWarning)
        res = fit(spec, df, method=BAEM(tol=1e-6, max_iter=300, squarem=True), se=None)

    assert res.n_safeguards > 0
    assert bool(res.history["safeguard"].any())
    ll = res.history["loglik"].to_numpy()
    assert np.all(np.diff(ll) >= -1e-9 * np.abs(ll[1:]))
    assert res.loglik >= ll[-1] - 1e-9 * abs(ll[-1])
    g_vals = res.items["g"].to_numpy()
    assert np.all((g_vals > 0.0) & (g_vals < 1.0))


def test_safeguard_fires_for_latent_variance_near_zero() -> None:
    """Near-zero latent variance: SQUAREM keeps var > 0 and the log-likelihood monotone.

    Whether the safeguard actually fires here depends on the exact iterate path, which differs
    across platforms (it fired on macOS but not on Linux CI), so only the guarantees are asserted.
    `test_safeguard_fires_for_threepl_near_lower_bound` covers the safeguard path itself.
    """
    a = np.ones(8)
    d = np.linspace(-1.2, 1.2, 8)
    df = _simulate_2pl(a, d, 1200, seed=7, sd=0.05)
    spec = Model.uniform(Rasch(), items=df.columns)
    table = spec.parameter_table(df)
    table.loc[(table["item"] == "__latent__") & (table["param"] == "var"), "value"] = 0.05
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        res = fit(
            spec,
            df,
            table=table,
            method=BAEM(tol=1e-8, max_iter=300, squarem=True),
            se=None,
        )

    assert float(res.latent.iloc[0]["var"]) > 0.0
    ll = res.history["loglik"].to_numpy()
    assert np.all(np.diff(ll) >= -1e-9 * np.abs(ll[1:]))
    assert res.loglik >= ll[-1] - 1e-9 * abs(ll[-1])


def test_non_convergence_warning_mentions_safeguard_when_triggered() -> None:
    case = test_mirt_m2.load("v3_science_grm")
    with pytest.warns(ConvergenceWarning, match="SQUAREM safeguard fired"):
        res = fit(
            case.spec,
            case.data,
            method=BAEM(tol=1e-12, max_iter=5, squarem=True),
            quadrature=case.quadrature,
            se=None,
        )
    assert not res.converged
    assert res.n_safeguards > 0


# ---- clear reduction in EM evaluations on slow cases ----------------------------------------


@pytest.mark.parametrize(
    "case_id",
    ["v2_lsat7_rasch", "v2_sim_rasch", "v4_twogroup_2pl", "v8_science_gpcm"],
)
def test_squarem_reduces_em_evaluations_on_slow_cases(case_id: str) -> None:
    """SQUAREM cuts EM map evaluations clearly (to < 0.75x) while matching the fixed point.

    The exact counts vary across platforms (e.g. GPCM/Science: 96 of 251 on macOS, 105 of 209 on
    Linux CI), so the threshold leaves room for that while still requiring a clear reduction.
    """
    if case_id.startswith("v2_"):
        c1 = load_m1(case_id)
        spec, data, group, quad = (
            test_mirt_m1._spec(c1),
            c1.data,
            None,
            Quadrature.equispaced(),
        )
    elif case_id.startswith("v4_"):
        c2 = test_mirt_m2.load(case_id)
        spec, data, group, quad = c2.spec, c2.data, c2.group, c2.quadrature
    else:
        c3 = test_mirt_m3_gpcm.load(case_id)
        spec, data, group, quad = c3.spec, c3.data, None, c3.quadrature

    plain = fit(
        spec,
        data,
        group=group,
        method=BAEM(tol=1e-9, max_iter=20000, squarem=False),
        quadrature=quad,
        se=None,
    )
    sq = fit(
        spec,
        data,
        group=group,
        method=BAEM(tol=1e-9, max_iter=20000, squarem=True),
        quadrature=quad,
        se=None,
    )
    assert plain.converged and sq.converged
    assert sq.n_em_evaluations < 0.75 * plain.n_em_evaluations
    np.testing.assert_allclose(sq._eta, plain._eta, rtol=0, atol=1e-6)
    assert sq.loglik == pytest.approx(plain.loglik, rel=1e-10)


def test_converged_fit_warns_when_squarem_falls_back_frequently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The converged-but-frequent-fallback ConvergenceWarning path of fit() (DESIGN §15.13).

    Forcing the real algorithm to fall back on most iterations is fragile, so the jitted runner is
    wrapped to report 6 accepted iterates with 12 EM evaluations, i.e. 3*6 - 12 = 6 fallbacks
    (>= SQUAREM_FALLBACK_WARN_MIN and > SQUAREM_FALLBACK_WARN_RATE of the iterations).
    """
    import importlib

    # `flexpyirt.fit` is also the name of the exported function, so look the module up directly.
    fit_module = importlib.import_module("flexpyirt.fit")

    real = fit_module._squarem

    def fake(*args: object, **kwargs: object) -> object:
        out = real(*args, **kwargs)
        return out._replace(
            n_iter=jnp.asarray(6, dtype=jnp.int32),
            n_em_evaluations=jnp.asarray(12, dtype=jnp.int32),
            converged=jnp.asarray(True),
        )

    monkeypatch.setattr(fit_module, "_squarem", fake)
    case = load_m1("v1_lsat7_2pl")
    with pytest.warns(ConvergenceWarning, match="SQUAREM fell back to plain EM frequently"):
        res = fit(
            Model.uniform(TwoPL(), items=case.items),
            case.data,
            method=BAEM(squarem=True),
            se=None,
        )
    assert res.converged


def test_converged_fit_does_not_warn_with_rare_fallbacks() -> None:
    case = load_m1("v1_lsat7_2pl")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        res = fit(Model.uniform(TwoPL(), items=case.items), case.data, method=BAEM(squarem=True))
    assert res.converged
