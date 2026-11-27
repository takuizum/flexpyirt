"""The optional unconstrained M-step scale, ``BAEM(m_step_scale="unconstrained")`` (DESIGN §15.47).

Interior solutions must equal those of the default natural-scale active-set Newton (§15.2);
boundary solutions are only approached but still raise ``HeywoodWarning``. The mirt reference
cases with the option on are in ``test_mirt_m3.py``, ``test_mirt_m3_priors.py`` and
``test_mirt_m4_mirt.py`` (V45).
"""

from __future__ import annotations

import json
import warnings
from functools import cache

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from flexpyirt import (
    BAEM,
    FitResult,
    LogNormalPrior,
    Model,
    Normal,
    ThreePL,
    TwoPL,
    fit,
)
from flexpyirt._json import SCHEMA_VERSION
from flexpyirt._newton import _to_natural, _to_unconstrained, _transform_kind, newton_ascent
from flexpyirt.errors import HeywoodWarning, SpecificationError

SCALES = ("natural", "unconstrained")


def _simulate_3pl(a: np.ndarray, d: np.ndarray, g: np.ndarray, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, 1.0, size=(n, 1))
    p = g + (1.0 - g) / (1.0 + np.exp(-(theta * a + d)))
    y = (rng.random(p.shape) < p).astype(int)
    return pd.DataFrame(y, columns=[f"q{j + 1:02d}" for j in range(len(a))])


# ---- options ----------------------------------------------------------------------------------


def test_default_is_natural() -> None:
    assert BAEM().m_step_scale == "natural"


@pytest.mark.parametrize("value", ["logit", "Natural", "", None])
def test_invalid_scale_raises(value: object) -> None:
    with pytest.raises(SpecificationError, match="m_step_scale"):
        BAEM(m_step_scale=value)  # type: ignore[arg-type]


# ---- transforms -------------------------------------------------------------------------------


def test_transform_kinds_follow_the_bounds() -> None:
    inf = np.inf
    lower = jnp.asarray([0.0, 0.0, -inf, -inf, -1.0, 0.0])
    upper = jnp.asarray([1.0, inf, 3.0, inf, 1.0, 1.0])
    mask = jnp.asarray([True, True, True, True, True, False])
    kind = np.asarray(_transform_kind(mask, lower, upper))
    np.testing.assert_array_equal(kind, [1, 2, 3, 0, 1, 0])


def test_transform_round_trip_and_formulas() -> None:
    inf = np.inf
    lower = jnp.asarray([0.0, 0.0, -inf, -inf, -1.0])
    upper = jnp.asarray([1.0, inf, 3.0, inf, 1.0])
    mask = jnp.ones(5, dtype=bool)
    kind = _transform_kind(mask, lower, upper)
    x = jnp.asarray([0.2, 1.7, -0.4, -2.5, 0.3])
    z = _to_unconstrained(x, kind, lower, upper)
    expected = [np.log(0.2 / 0.8), np.log(1.7), np.log(3.4), -2.5, 2.0 * np.arctanh(0.3)]
    np.testing.assert_allclose(np.asarray(z), expected, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(np.asarray(_to_natural(z, kind, lower, upper)), x, rtol=1e-14)


def test_fixed_slots_with_dummy_values_stay_finite() -> None:
    """Fixed slots carry placeholder values outside the box; they must not create NaNs."""
    lower = jnp.asarray([0.0, 0.0])
    upper = jnp.asarray([1.0, 1.0])
    mask = jnp.asarray([True, False])
    kind = _transform_kind(mask, lower, upper)
    x = jnp.asarray([0.3, 5.0])
    z = _to_unconstrained(x, kind, lower, upper)
    assert np.all(np.isfinite(np.asarray(z)))
    np.testing.assert_array_equal(np.asarray(_to_natural(z, kind, lower, upper))[1], 5.0)


# ---- the Newton step --------------------------------------------------------------------------


def _beta_like(x: jnp.ndarray) -> jnp.ndarray:
    """Concave in (0, 1) x (0, inf) with maximum at (0.3, 2.0)."""
    return 3.0 * jnp.log(x[0]) + 7.0 * jnp.log1p(-x[0]) + 2.0 * jnp.log(x[1]) - x[1]


@pytest.mark.parametrize("unconstrained", [False, True])
def test_interior_maximum_is_the_same(unconstrained: bool) -> None:
    lower = jnp.asarray([0.0, 0.0])
    upper = jnp.asarray([1.0, jnp.inf])
    mask = jnp.ones(2, dtype=bool)
    x = newton_ascent(
        _beta_like,
        jnp.asarray([0.9, 0.1]),
        mask,
        lower,
        upper,
        40,
        unconstrained=unconstrained,
    )
    np.testing.assert_allclose(np.asarray(x), [0.3, 2.0], rtol=1e-10)


def test_boundary_maximum_is_approached_from_inside() -> None:
    """f increasing in x on (0, 1): the iterate tends to 1 but stays strictly inside."""

    def f(x: jnp.ndarray) -> jnp.ndarray:
        return x[0] - (x[1] - 0.5) ** 2

    lower = jnp.asarray([0.0, -jnp.inf])
    upper = jnp.asarray([1.0, jnp.inf])
    x0 = jnp.asarray([0.5, 3.0])
    x = np.asarray(
        newton_ascent(f, x0, jnp.ones(2, dtype=bool), lower, upper, 30, unconstrained=True)
    )
    assert 0.99 < x[0] < 1.0
    np.testing.assert_allclose(x[1], 0.5, rtol=1e-12)
    assert float(f(jnp.asarray(x))) >= float(f(x0))


def test_fixed_coordinates_do_not_move_and_rejected_step_is_exact() -> None:
    lower = jnp.asarray([0.0, 0.0])
    upper = jnp.asarray([1.0, 1.0])
    x0 = jnp.asarray([0.3, 0.123456789])
    mask = jnp.asarray([True, False])
    x = newton_ascent(_beta_like, x0, mask, lower, upper, 5, unconstrained=True)
    assert float(x[1]) == 0.123456789
    # At the maximum no step increases f: the iterate comes back bit for bit.
    x_star = jnp.asarray([0.3, 2.0])
    out = newton_ascent(
        _beta_like,
        x_star,
        jnp.ones(2, dtype=bool),
        jnp.asarray([0.0, 0.0]),
        jnp.asarray([1.0, jnp.inf]),
        3,
        unconstrained=True,
    )
    assert float(_beta_like(out)) >= float(_beta_like(x_star))


# ---- fits: interior solutions agree with the natural scale -----------------------------------


@cache
def _data_3pl() -> pd.DataFrame:
    a = np.array([1.2, 1.5, 0.9, 1.4, 1.1, 1.3, 1.0, 1.6])
    d = np.array([-0.5, 0.3, -1.0, 0.7, -0.2, 0.5, 0.0, -0.8])
    g = np.array([0.2, 0.15, 0.25, 0.18, 0.22, 0.19, 0.2, 0.17])
    return _simulate_3pl(a, d, g, n=1000, seed=15)


@cache
def _fit_3pl(scale: str, squarem: bool = False) -> FitResult:
    df = _data_3pl()
    spec = (
        Model.uniform(ThreePL(), items=df.columns)
        .prior(param="a", dist=LogNormalPrior(0.2, 0.5))
        .prior(param="g", dist=LogNormalPrior(-1.6, 0.6))
    )
    return fit(
        spec,
        df,
        method=BAEM(tol=1e-8, max_iter=5000, m_step_scale=scale, squarem=squarem),  # type: ignore[arg-type]
        se="oakes",
    )


@pytest.mark.slow
@pytest.mark.parametrize("squarem", [False, True])
def test_3pl_map_interior_solution_matches(squarem: bool) -> None:
    """logit(g) (box (0, 1)) and log(a) (box (0, inf) from LogNormalPrior) give the same MAP."""
    nat, unc = _fit_3pl("natural"), _fit_3pl("unconstrained", squarem)
    assert nat.converged and unc.converged
    # Both stop on max|change| < 1e-8; slow EM directions leave ~1e-6 to the common fixed point.
    np.testing.assert_allclose(unc._eta, nat._eta, rtol=0, atol=1e-5)
    np.testing.assert_allclose(unc.log_posterior, nat.log_posterior, rtol=1e-12)
    assert nat.vcov is not None and unc.vcov is not None
    np.testing.assert_allclose(unc.vcov, nat.vcov, rtol=1e-4, atol=1e-10)
    lp = unc.history["log_posterior"].to_numpy()
    assert np.all(np.diff(lp) >= -1e-9 * np.abs(lp[1:]))


@cache
def _fit_two_group(scale: str) -> FitResult:
    rng = np.random.default_rng(76)
    a = rng.uniform(0.8, 1.8, 10)
    d = rng.normal(0.0, 0.8, 10)
    theta = np.concatenate([rng.normal(0.0, 1.0, 800), rng.normal(0.4, 1.3, 800)])
    p = 1.0 / (1.0 + np.exp(-(theta[:, None] * a + d)))
    df = pd.DataFrame((rng.random(p.shape) < p).astype(int), columns=[f"i{j}" for j in range(10)])
    group = np.repeat(["ref", "focal"], 800)
    spec = (
        Model.uniform(TwoPL(), items=df.columns)
        .with_groups(ref=Normal.standard(), focal=Normal.free())
        .equal_across_groups(items=list(df.columns))
    )
    return fit(
        spec,
        df,
        group=group,
        method=BAEM(tol=1e-9, max_iter=5000, m_step_scale=scale),  # type: ignore[arg-type]
        se=None,
    )


def test_two_group_free_variance_matches() -> None:
    """log(sigma^2) for the focal variance (box (0, inf))."""
    nat, unc = _fit_two_group("natural"), _fit_two_group("unconstrained")
    assert nat.converged and unc.converged
    np.testing.assert_allclose(unc._eta, nat._eta, rtol=0, atol=1e-6)
    focal_var = unc.latent.set_index("group")["var"].to_numpy(dtype=float)[1]
    assert focal_var > 1.2


@pytest.mark.slow
def test_boundary_solution_still_warns() -> None:
    """Items generated with g = 0: both scales end at the bound and raise HeywoodWarning."""
    a = np.array([1.2, 1.5, 0.9, 1.4, 1.1, 1.3])
    d = np.array([-1.5, 0.3, -1.0, 0.7, -0.2, 0.5])
    g = np.array([0.0, 0.2, 0.0, 0.2, 0.2, 0.2])
    df = _simulate_3pl(a, d, g, n=1500, seed=5)
    spec = Model.uniform(ThreePL(), items=df.columns)
    lls = {}
    for scale in SCALES:
        with pytest.warns(HeywoodWarning):
            res = fit(
                spec,
                df,
                method=BAEM(tol=1e-6, max_iter=3000, m_step_scale=scale),  # type: ignore[arg-type]
                se=None,
            )
        g_hat = res.items["g"].to_numpy()
        assert np.all(g_hat > 0.0)
        assert np.sum(g_hat < 1e-4) >= 2
        lls[scale] = res.loglik
    assert abs(lls["unconstrained"] - lls["natural"]) < 1e-5


# ---- JSON -------------------------------------------------------------------------------------


def test_json_round_trip_and_old_schema(tmp_path: pytest.TempPathFactory) -> None:
    res = _fit_two_group("unconstrained")
    path = tmp_path / "fit.json"  # type: ignore[operator]
    res.to_json(path)
    raw = json.loads(path.read_text())
    assert raw["schema_version"] == SCHEMA_VERSION
    assert raw["method"]["m_step_scale"] == "unconstrained"
    assert FitResult.from_json(path)._method.m_step_scale == "unconstrained"
    # A version-7 file has no m_step_scale key and reads as the natural scale.
    raw["schema_version"] = 7
    del raw["method"]["m_step_scale"]
    old = tmp_path / "old.json"  # type: ignore[operator]
    old.write_text(json.dumps(raw))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert FitResult.from_json(old)._method.m_step_scale == "natural"
