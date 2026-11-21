"""Unit tests for M4 scoring methods: MAP, ML, WLE, and plausible values (PV)."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from flexpyirt import (
    BAEM,
    GPCM,
    GRM,
    PCM,
    FitResult,
    FourPL,
    Model,
    Normal,
    Quadrature,
    Rasch,
    ThreePL,
    TwoPL,
    fit,
)
from flexpyirt.errors import SpecificationError

A_2PL = np.array([1.2, 0.9, 1.5, 1.1, 0.8, 1.4, 1.0, 1.3])
D_2PL = np.array([0.8, -0.6, 0.3, -1.0, 0.5, -0.2, 1.1, -0.4])
ITEMS_2PL = [f"q{j:02d}" for j in range(1, len(A_2PL) + 1)]


def _sim_2pl(
    n: int, seed: int, a: np.ndarray, d: np.ndarray, mu: float = 0.0, sd: float = 1.0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    theta = rng.normal(mu, sd, size=(n, 1))
    p = 1.0 / (1.0 + np.exp(-(theta * a + d)))
    y = (rng.random((n, len(a))) < p).astype(float)
    return pd.DataFrame(y, columns=[f"q{j:02d}" for j in range(1, len(a) + 1)])


def _fixed_2pl(
    a: np.ndarray,
    d: np.ndarray,
    *,
    mean: float = 0.0,
    var: float = 1.0,
    quadrature: Quadrature | None = None,
) -> FitResult:
    """Build a 2PL FitResult with exact fixed item and latent parameters."""
    items = [f"q{j:02d}" for j in range(1, len(a) + 1)]
    dummy = pd.DataFrame(np.vstack([np.zeros(len(a)), np.ones(len(a))]), columns=items)
    spec = Model.uniform(TwoPL(), items=items).with_latent(Normal(mean=mean, var=var))
    table = spec.parameter_table(dummy)
    vals = []
    for item, param in zip(table["item"], table["param"], strict=True):
        if item == "__latent__":
            vals.append(mean if param == "mean" else var)
        else:
            idx = items.index(item)
            vals.append(float(a[idx] if param == "a" else d[idx]))
    table["value"] = vals
    table["free"] = False
    return fit(spec, dummy, table=table, quadrature=quadrature, se=None)


@pytest.fixture(scope="module")
def fitted_2pl() -> tuple[FitResult, pd.DataFrame]:
    df = _sim_2pl(600, 101, A_2PL, D_2PL)
    res = fit(Model.uniform(TwoPL(), items=df.columns), df, method=BAEM(tol=1e-6), se=None)
    return res, df


# ---- Error handling & API validation -------------------------------------------------------


def test_score_validation_errors(fitted_2pl: tuple[FitResult, pd.DataFrame]) -> None:
    res, df = fitted_2pl
    small = df.iloc[:5]
    key = jax.random.key(0)

    with pytest.raises(SpecificationError, match="Unsupported scoring method"):
        res.score(small, method="INVALID")  # type: ignore[arg-type]

    with pytest.raises(SpecificationError, match="requires an explicit JAX PRNG key"):
        res.score(small, method="PV")

    with pytest.raises(SpecificationError, match="JAX PRNG key"):
        res.score(small, method="PV", key=42)  # type: ignore[arg-type]

    with pytest.raises(SpecificationError, match="JAX PRNG key"):
        res.score(small, method="PV", key=jnp.array([1.0, 2.0]))

    for bad_draws in (0, -3, True, 2.5):
        with pytest.raises(SpecificationError, match="n_draws"):
            res.score(small, method="PV", key=key, n_draws=bad_draws)  # type: ignore[arg-type]

    for det_method in ("EAP", "MAP", "ML", "WLE"):
        with pytest.raises(SpecificationError, match="only used with method='PV'"):
            res.score(small, method=det_method, key=key)


def test_score_raises_precision_error_without_x64() -> None:
    """FitResult.score() checks float64 at entry and raises PrecisionError when x64 is off."""
    code = textwrap.dedent(
        """
        import jax
        import numpy as np
        import pandas as pd
        import flexpyirt
        from flexpyirt.errors import PrecisionError

        jax.config.update("jax_enable_x64", True)
        items = ["i1", "i2"]
        df = pd.DataFrame([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]], columns=items)
        spec = flexpyirt.Model.uniform(flexpyirt.TwoPL(), items=items)
        table = spec.parameter_table(df)
        table["value"] = [
            1.0 if p == "a" else (0.0 if p == "d" else (0.0 if p == "mean" else 1.0))
            for p in table["param"]
        ]
        table["free"] = False
        res = flexpyirt.fit(spec, df, table=table, se=None)

        jax.config.update("jax_enable_x64", False)
        assert not jax.config.jax_enable_x64
        for method in ("EAP", "MAP", "ML", "WLE", "PV"):
            kwargs = {"key": jax.random.key(0)} if method == "PV" else {}
            try:
                res.score(df, method=method, **kwargs)
            except PrecisionError as exc:
                assert "jax_enable_x64" in str(exc)
            else:
                raise SystemExit(f"PrecisionError not raised for method={method!r}")
        """
    )
    env = {**os.environ, "JAX_ENABLE_X64": "0", "JAX_PLATFORMS": "cpu"}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


# ---- MAP vs ML & analytic checks -----------------------------------------------------------


def test_map_with_flat_prior_matches_ml_on_finite_patterns() -> None:
    """As prior variance -> inf, MAP approaches ML in both theta and SE on finite patterns."""
    res_flat = _fixed_2pl(A_2PL, D_2PL, mean=0.0, var=1e8)
    pats = pd.DataFrame(
        [
            [0, 0, 1, 0, 1, 0, 0, 1],
            [1, 0, 1, 1, 0, 1, 0, 0],
            [1, 1, 1, 0, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 1, 0],
        ],
        columns=ITEMS_2PL,
        dtype=float,
    )
    ml = res_flat.score(pats, method="ML")
    mp = res_flat.score(pats, method="MAP")
    np.testing.assert_allclose(mp["theta"], ml["theta"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(mp["se"], ml["se"], rtol=1e-6, atol=0)


def test_map_uses_continuous_prior_not_grid() -> None:
    """MAP maximizes with the exact continuous normal prior even on a coarse grid."""
    res_coarse = _fixed_2pl(
        A_2PL, D_2PL, mean=0.7, var=1.5, quadrature=Quadrature.equispaced(15, lim=(-6, 6))
    )
    res_fine = _fixed_2pl(
        A_2PL, D_2PL, mean=0.7, var=1.5, quadrature=Quadrature.equispaced(81, lim=(-8, 8))
    )
    pats = pd.DataFrame(
        [[0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 1, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]],
        columns=ITEMS_2PL,
        dtype=float,
    )
    s_coarse = res_coarse.score(pats, method="MAP")
    s_fine = res_fine.score(pats, method="MAP")
    np.testing.assert_allclose(s_coarse["theta"], s_fine["theta"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(s_coarse["se"], s_fine["se"], rtol=1e-10, atol=0)


def test_map_all_missing_returns_prior_mode_and_sd() -> None:
    res = _fixed_2pl(A_2PL, D_2PL, mean=-0.45, var=1.44)
    empty = pd.DataFrame(np.full((2, len(A_2PL)), np.nan), columns=ITEMS_2PL)
    s = res.score(empty, method="MAP")
    np.testing.assert_allclose(s["theta"], -0.45, atol=1e-10)
    np.testing.assert_allclose(s["se"], 1.2, atol=1e-10)


# ---- ML extreme patterns & 3PL diverging tail ----------------------------------------------


def test_ml_extreme_patterns_and_all_missing() -> None:
    res = _fixed_2pl(A_2PL, D_2PL)
    pats = pd.DataFrame(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],  # all lowest -> -inf
            [1, 1, 1, 1, 1, 1, 1, 1],  # all highest -> +inf
            [np.nan] * 8,  # all missing -> NaN
            [0, np.nan, 0, 0, np.nan, 0, 0, 0],  # observed all lowest -> -inf
            [1, 1, np.nan, 1, 1, 1, np.nan, 1],  # observed all highest -> +inf
            [1, 0, 1, 0, 1, 0, 1, 0],  # interior -> finite
        ],
        columns=ITEMS_2PL,
        dtype=float,
    )
    ml = res.score(pats, method="ML")
    assert ml["theta"].iloc[0] == -np.inf and np.isnan(ml["se"].iloc[0])
    assert ml["theta"].iloc[1] == np.inf and np.isnan(ml["se"].iloc[1])
    assert np.isnan(ml["theta"].iloc[2]) and np.isnan(ml["se"].iloc[2])
    assert ml["theta"].iloc[3] == -np.inf and np.isnan(ml["se"].iloc[3])
    assert ml["theta"].iloc[4] == np.inf and np.isnan(ml["se"].iloc[4])
    assert np.isfinite(ml["theta"].iloc[5]) and np.isfinite(ml["se"].iloc[5])
    assert ml["se"].iloc[5] > 0


def test_ml_3pl_diverging_low_score_pattern_is_neg_inf() -> None:
    """In 3PL, a single lucky guess among hard items can have supremum at -inf."""
    items = [f"c{j}" for j in range(1, 11)]
    dummy = pd.DataFrame(np.vstack([np.zeros(10), np.ones(10)]), columns=items)
    spec = Model.uniform(ThreePL(), items=items)
    table = spec.parameter_table(dummy)
    vals = []
    for item, param in zip(table["item"], table["param"], strict=True):
        if item == "__latent__":
            vals.append(0.0 if param == "mean" else 1.0)
        elif param == "a":
            vals.append(2.0)
        elif param == "d":
            vals.append(-1.0)
        elif param == "g":
            vals.append(0.25)
        else:
            vals.append(1.0)
    table["value"] = vals
    table["free"] = False
    res = fit(spec, dummy, table=table, se=None)

    pats = pd.DataFrame(
        [
            [0] * 10,
            [1] + [0] * 9,  # 1/10 with g=0.25 -> logL decreases with theta
            [1] * 6 + [0] * 4,  # clearly interior
            [1] * 10,
        ],
        columns=items,
        dtype=float,
    )
    ml = res.score(pats, method="ML")
    assert ml["theta"].iloc[0] == -np.inf and np.isnan(ml["se"].iloc[0])
    assert ml["theta"].iloc[1] == -np.inf and np.isnan(ml["se"].iloc[1])
    assert np.isfinite(ml["theta"].iloc[2]) and np.isfinite(ml["se"].iloc[2])
    assert ml["theta"].iloc[3] == np.inf and np.isnan(ml["se"].iloc[3])


def _fixed_3pl(a: list[float], d: list[float], g: list[float]) -> FitResult:
    """Build a 3PL FitResult with exact fixed item and standard-normal latent parameters."""
    items = [f"c{j}" for j in range(1, len(a) + 1)]
    dummy = pd.DataFrame(np.vstack([np.zeros(len(a)), np.ones(len(a))]), columns=items)
    spec = Model.uniform(ThreePL(), items=items)
    table = spec.parameter_table(dummy)
    vals = []
    for item, param in zip(table["item"], table["param"], strict=True):
        if item == "__latent__":
            vals.append(0.0 if param == "mean" else 1.0)
        else:
            idx = items.index(item)
            if param == "a":
                vals.append(float(a[idx]))
            elif param == "d":
                vals.append(float(d[idx]))
            elif param == "g":
                vals.append(float(g[idx]))
            else:
                vals.append(1.0)
    table["value"] = vals
    table["free"] = False
    return fit(spec, dummy, table=table, se=None)


def test_ml_diverged_sign_uses_theta_hat_unless_endpoint_dominates() -> None:
    """Diverged ML uses sign(theta_hat) unless an endpoint actually exceeds f(theta_hat).

    Regression test: in 3PL, f(-20) > f(+20) holds for nearly every non-all-1 pattern because the
    lower asymptote is finite while P(y=0) -> 0 as theta -> +inf. When a positive interior mode
    diverges via SE > 30 with f(theta_hat) > max(f(-20), f(+20)), ML must return +inf (matching
    mirt:::ML), not -inf. Conversely, when f(-20) strictly exceeds f(theta_hat), ML returns -inf.
    """
    a = [2.0, 2.0, 2.0, 2.0, 1.0]
    g = [0.25] * 5
    items = [f"c{j}" for j in range(1, 6)]
    pats = pd.DataFrame(
        [
            [1, 1, 1, 1, 0],  # 4 easy items right, 1 very hard item wrong -> positive mode
            [0, 0, 0, 0, 1],  # 4 easy items wrong, 1 very hard item right -> -inf dominates
        ],
        columns=items,
        dtype=float,
    )

    # At d5 = -12, the positive mode is still finite (SE < 30).
    res_fin = _fixed_3pl(a, [1.0, 1.0, 1.0, 1.0, -12.0], g)
    ml_fin = res_fin.score(pats, method="ML")
    assert ml_fin["theta"].iloc[0] > 4.0 and np.isfinite(ml_fin["se"].iloc[0])
    assert ml_fin["theta"].iloc[1] == -np.inf and np.isnan(ml_fin["se"].iloc[1])

    # At d5 = -13, the positive mode has SE > 30 with f(theta_hat) >> f(-20) > f(+20): expect +inf.
    res_div = _fixed_3pl(a, [1.0, 1.0, 1.0, 1.0, -13.0], g)
    ml_div = res_div.score(pats, method="ML")
    assert ml_div["theta"].iloc[0] == np.inf and np.isnan(ml_div["se"].iloc[0])
    # When the -inf endpoint genuinely dominates f(theta_hat), expect -inf.
    assert ml_div["theta"].iloc[1] == -np.inf and np.isnan(ml_div["se"].iloc[1])


def test_zero_slope_items_treated_as_missing_for_pattern_flags() -> None:
    """Items with slope a == 0 carry no information about theta and act like missing responses."""
    a = np.array([0.0, 1.2, 1.0, 1.5, 0.9])
    d = np.array([0.0, -0.5, 0.0, 0.5, -0.2])
    items = [f"q{j:02d}" for j in range(1, 6)]
    res = _fixed_2pl(a, d)

    pats = pd.DataFrame(
        [
            [0, np.nan, np.nan, np.nan, np.nan],  # only a == 0 item observed -> all-missing
            [1, np.nan, np.nan, np.nan, np.nan],  # only a == 0 item observed -> all-missing
            [1, 0, 0, 0, 0],  # all informative items lowest despite y1 = 1 -> -inf
            [0, 1, 1, 1, 1],  # all informative items highest despite y1 = 0 -> +inf
            [0, 1, 0, 1, 0],  # mixed informative items -> same as y1 missing
            [np.nan, 1, 0, 1, 0],
        ],
        columns=items,
        dtype=float,
    )

    ml = res.score(pats, method="ML")
    assert np.isnan(ml["theta"].iloc[0]) and np.isnan(ml["se"].iloc[0])
    assert np.isnan(ml["theta"].iloc[1]) and np.isnan(ml["se"].iloc[1])
    assert ml["theta"].iloc[2] == -np.inf and np.isnan(ml["se"].iloc[2])
    assert ml["theta"].iloc[3] == np.inf and np.isnan(ml["se"].iloc[3])
    np.testing.assert_allclose(ml["theta"].iloc[4], ml["theta"].iloc[5], rtol=0, atol=1e-12)
    np.testing.assert_allclose(ml["se"].iloc[4], ml["se"].iloc[5], rtol=0, atol=1e-12)

    wle = res.score(pats, method="WLE")
    assert np.isnan(wle["theta"].iloc[0]) and np.isnan(wle["se"].iloc[0])
    assert np.isnan(wle["theta"].iloc[1]) and np.isnan(wle["se"].iloc[1])
    assert np.isfinite(wle["theta"].iloc[2]) and np.isfinite(wle["se"].iloc[2])
    assert np.isfinite(wle["theta"].iloc[3]) and np.isfinite(wle["se"].iloc[3])
    np.testing.assert_allclose(wle["theta"].iloc[4], wle["theta"].iloc[5], rtol=0, atol=1e-12)
    np.testing.assert_allclose(wle["se"].iloc[4], wle["se"].iloc[5], rtol=0, atol=1e-12)


# ---- WLE finite on extremes & stationary equation check ------------------------------------


@pytest.mark.parametrize(
    ("model_cls", "n_cat", "params_by_name"),
    [
        (TwoPL, 2, {"a": 1.4, "d": -0.3}),
        (Rasch, 2, {"a": 1.0, "d": 0.2}),
        (ThreePL, 2, {"a": 1.6, "d": -0.5, "g": 0.2, "u": 1.0}),
        (FourPL, 2, {"a": 1.7, "d": 0.1, "g": 0.15, "u": 0.9}),
        (GRM, 4, {"a": 1.3, "d1": 1.5, "d2": 0.1, "d3": -1.4}),
        (GPCM, 4, {"a": 1.1, "d1": 0.7, "d2": 0.2, "d3": -0.9}),
        (PCM, 4, {"a": 1.0, "d1": 0.8, "d2": 0.0, "d3": -1.1}),
    ],
)
def test_wle_finite_on_extremes_and_satisfies_warm_equation(
    model_cls: type,
    n_cat: int,
    params_by_name: dict[str, float],
) -> None:
    """WLE is finite on all-lowest/all-highest patterns and solves dL/dtheta + I'/(2I) = 0."""
    item_model = model_cls(n_cat=n_cat) if n_cat > 2 else model_cls()
    items = [f"x{j}" for j in range(5)]
    dummy = pd.DataFrame(np.tile(np.arange(n_cat), (5, 1)).T, columns=items, dtype=float)
    spec = Model.uniform(item_model, items=items)
    table = spec.parameter_table(dummy)
    vals = []
    for item, param in zip(table["item"], table["param"], strict=True):
        if item == "__latent__":
            vals.append(0.0 if param == "mean" else 1.0)
        else:
            offset = (items.index(item) - 2) * 0.25
            base = params_by_name[param]
            vals.append(base + offset if param.startswith("d") else base)
    table["value"] = vals
    table["free"] = False
    res = fit(spec, dummy, table=table, se=None)

    pats = pd.DataFrame(
        [
            [0] * 5,  # all lowest
            [n_cat - 1] * 5,  # all highest
            [0, n_cat - 1, 0, n_cat - 1, 0],  # mixed
            [np.nan] * 5,  # all missing -> NaN
        ],
        columns=items,
        dtype=float,
    )
    wle = res.score(pats, method="WLE")
    assert np.isfinite(wle["theta"].iloc[:3]).all()
    assert np.isfinite(wle["se"].iloc[:3]).all()
    assert (wle["se"].iloc[:3] > 0).all()
    assert wle["theta"].iloc[0] < wle["theta"].iloc[2] < wle["theta"].iloc[1]
    assert np.isnan(wle["theta"].iloc[3]) and np.isnan(wle["se"].iloc[3])

    # Directly verify Warm's first-order condition d log L / d theta + I'(theta) / (2 I(theta)) = 0
    kernel = item_model.kernel
    p_names = list(kernel.param_names)
    item_tbl = res.items.set_index("item")
    xi = jnp.asarray(item_tbl.loc[items, p_names].to_numpy(dtype=float))
    for row_idx in range(3):
        y_row = jnp.asarray(pats.iloc[row_idx].to_numpy(dtype=int))

        def loglik(t: jax.Array, y_r: jax.Array = y_row) -> jax.Array:
            lp = jax.vmap(lambda p: kernel.log_prob(t[None], p)[0])(xi)
            return jnp.sum(lp[jnp.arange(5), y_r])

        def info(t: jax.Array) -> jax.Array:
            def one(p: jax.Array) -> jax.Array:
                lp = kernel.log_prob(t[None], p)[0]
                dlp = jax.jacfwd(lambda s: kernel.log_prob(s[None], p)[0])(t)
                return jnp.sum(jnp.exp(lp) * dlp * dlp)

            return jnp.sum(jax.vmap(one)(xi))

        th = jnp.asarray(float(wle["theta"].iloc[row_idx]))
        grad_ll = float(jax.grad(loglik)(th))
        info_val = float(info(th))
        j_val = float(jax.grad(info)(th))
        assert abs(grad_ll + j_val / (2.0 * info_val)) < 1e-8


# ---- Plausible values (PV) -----------------------------------------------------------------


def test_pv_reproducibility_keys_and_independence(
    fitted_2pl: tuple[FitResult, pd.DataFrame],
) -> None:
    res, df = fitted_2pl
    # Duplicate the same pattern across 4 rows: draws must be independent across respondents.
    dup = pd.concat([df.iloc[[0]]] * 4, ignore_index=True)
    dup.index = pd.Index(["r1", "r2", "r3", "r4"])

    k1 = jax.random.key(123)
    k2 = jax.random.key(456)
    pv1 = res.score(dup, method="PV", key=k1, n_draws=7)
    pv1_again = res.score(dup, method="PV", key=k1, n_draws=7)
    pv2 = res.score(dup, method="PV", key=k2, n_draws=7)

    assert list(pv1.columns) == [f"pv{m}" for m in range(1, 8)]
    assert list(pv1.index) == ["r1", "r2", "r3", "r4"]
    pd.testing.assert_frame_equal(pv1, pv1_again)
    assert not np.allclose(pv1.to_numpy(), pv2.to_numpy())
    # Identical patterns still receive independent draws across respondents.
    assert not np.allclose(pv1.iloc[0].to_numpy(), pv1.iloc[1].to_numpy())

    # Legacy PRNGKey is also accepted.
    pv_legacy = res.score(dup, method="PV", key=jax.random.PRNGKey(123), n_draws=3)
    assert pv_legacy.shape == (4, 3)
    assert np.isfinite(pv_legacy.to_numpy()).all()


def test_pv_moments_match_eap_within_mc_error() -> None:
    """With large n_draws, PV sample mean and SD match EAP and posterior SD within MC error."""
    res = _fixed_2pl(
        A_2PL, D_2PL, mean=0.3, var=1.2, quadrature=Quadrature.equispaced(81, lim=(-8, 8))
    )
    pats = pd.DataFrame(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 1, 1, 0, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, np.nan, 0, np.nan, 1, 0, np.nan, 1],
            [np.nan] * 8,  # all missing -> draws from N(0.3, 1.2)
        ],
        columns=ITEMS_2PL,
        dtype=float,
    )
    eap_df = res.score(pats, method="EAP")
    n_draws = 8000
    pv_df = res.score(pats, method="PV", key=jax.random.key(2026), n_draws=n_draws)
    draws = pv_df.to_numpy()
    pv_mean = draws.mean(axis=1)
    pv_sd = draws.std(axis=1, ddof=1)

    # Monte-Carlo SE of the mean is sd / sqrt(n_draws) <= 0.015; allow 5 * MC_SE.
    mc_se_mean = eap_df["se"].to_numpy() / np.sqrt(n_draws)
    np.testing.assert_allclose(pv_mean, eap_df["theta"].to_numpy(), atol=5.0 * mc_se_mean.max())
    # Monte-Carlo SE of the SD is ~sd / sqrt(2 * n_draws); allow 6 * MC_SE.
    mc_se_sd = eap_df["se"].to_numpy() / np.sqrt(2.0 * n_draws)
    np.testing.assert_allclose(pv_sd, eap_df["se"].to_numpy(), atol=6.0 * mc_se_sd.max())


# ---- Multiple groups, missing data, array inputs, and invariance ---------------------------


def test_all_methods_with_groups_missing_and_permutation_invariance() -> None:
    ref = _sim_2pl(300, 10, A_2PL, D_2PL, mu=0.0, sd=1.0)
    d_foc = D_2PL.copy()
    d_foc[4:] += np.array([0.4, -0.3, 0.2, -0.5])
    foc = _sim_2pl(300, 11, A_2PL, d_foc, mu=0.6, sd=1.15)
    df = pd.concat([ref, foc], ignore_index=True)
    grp = np.repeat(["ref", "focal"], 300)
    spec = (
        Model.uniform(TwoPL(), items=df.columns)
        .with_groups(ref=Normal.standard(), focal=Normal.free())
        .equal_across_groups(items=ITEMS_2PL[:4])
    )
    res = fit(
        spec,
        df,
        group=grp,
        method=BAEM(tol=1e-6),
        quadrature=Quadrature.equispaced(81, lim=(-8, 8)),
        se=None,
    )

    sample = pd.DataFrame(
        [
            [1, 0, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [1, np.nan, 1, 0, np.nan, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        columns=ITEMS_2PL,
        index=pd.Index(["a", "b", "c", "d"]),
        dtype=float,
    )
    sample_grp = ["ref", "focal", "focal", "ref"]

    for meth in ("EAP", "MAP", "ML", "WLE"):
        out = res.score(sample, method=meth, group=sample_grp)
        assert list(out.index) == ["a", "b", "c", "d"]
        # Same pattern scored in different groups uses group-specific item & latent params.
        assert out["theta"].iloc[0] != out["theta"].iloc[1]
        # Array input with item_names gives identical values.
        arr_out = res.score(sample.to_numpy(), method=meth, group=sample_grp, item_names=ITEMS_2PL)
        np.testing.assert_allclose(
            out["theta"].to_numpy(), arr_out["theta"].to_numpy(), equal_nan=True
        )
        np.testing.assert_allclose(out["se"].to_numpy(), arr_out["se"].to_numpy(), equal_nan=True)
        # Permutation invariance.
        perm = [3, 1, 0, 2]
        perm_out = res.score(sample.iloc[perm], method=meth, group=[sample_grp[i] for i in perm])
        np.testing.assert_allclose(
            out["theta"].to_numpy()[perm], perm_out["theta"].to_numpy(), equal_nan=True
        )
        np.testing.assert_allclose(
            out["se"].to_numpy()[perm], perm_out["se"].to_numpy(), equal_nan=True
        )

    # PV with groups and missing data
    pv = res.score(sample, method="PV", group=sample_grp, key=jax.random.key(7), n_draws=4)
    assert pv.shape == (4, 4)
    assert list(pv.index) == ["a", "b", "c", "d"]
    assert np.isfinite(pv.to_numpy()).all()
    # Focal group draws should have higher mean than ref group draws on large n_draws.
    pv_big = res.score(
        sample.iloc[:2], method="PV", group=["ref", "focal"], key=jax.random.key(9), n_draws=2000
    )
    assert pv_big.iloc[1].mean() > pv_big.iloc[0].mean()
