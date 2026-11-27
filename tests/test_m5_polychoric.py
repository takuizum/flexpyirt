"""D1: polychoric / tetrachoric correlations (DESIGN.md §16.9, VALIDATION.md V24).

Unit tests of the bivariate normal distribution function and the shell layer, and the reference
comparison against lavaan's ``lavCor`` (tight) and psych's ``polychoric`` (loose) from
``tests/reference/flexpyctt/v24_polychoric.json`` (a copy of flexpyctt's file, generated in the
flexpyctt repository by ``validation/r/m5_polychoric_reference.R``). These tests cover
flexpyirt's JAX implementation; flexpyctt's NumPy / SciPy implementation has the same tests in its own suite, and
the two are required to agree here (DESIGN.md §16.18).
"""

from __future__ import annotations

import json
import warnings
from functools import cache
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from flexpyctt import polychoric as numpy_polychoric
from scipy.integrate import quad
from scipy.stats import norm

from flexpyirt import PolychoricResult, polychoric
from flexpyirt._polychoric import bivariate_normal_cdf, pair_log_likelihood
from flexpyirt.correlation import smooth_correlation
from flexpyirt.errors import (
    HeywoodWarning,
    NotPositiveDefiniteWarning,
    PrecisionError,
    SpecificationError,
)
from tests._reference import CTT_REFERENCE_DIR, check_reference_meta

LAVAAN_TOL = 1e-6
PSYCH_TOL = 1e-3


def _plackett_cdf(h: float, k: float, r: float) -> float:
    """Independent reference: Phi(h) Phi(k) + integral_0^r phi_2(h, k; t) dt (Plackett, 1954)."""

    def density(t: float) -> float:
        s = 1.0 - t * t
        return float(np.exp(-(h * h - 2 * t * h * k + k * k) / (2 * s)) / (2 * np.pi * np.sqrt(s)))

    integral = quad(density, 0.0, r, epsabs=1e-15, epsrel=1e-13, limit=500)[0]
    return float(norm.cdf(h) * norm.cdf(k) + integral)


@pytest.mark.parametrize(
    "r", [-0.9999, -0.97, -0.93, -0.925, -0.8, -0.3, 0.0, 0.1, 0.6, 0.9249, 0.95, 0.999]
)
def test_bivariate_normal_cdf_matches_plackett_integral(r: float) -> None:
    rng = np.random.default_rng(int(1000 * (r + 1)))
    pts = rng.normal(scale=2.0, size=(25, 2))
    got = np.asarray(bivariate_normal_cdf(jnp.asarray(pts[:, 0]), jnp.asarray(pts[:, 1]), r))
    want = np.array([_plackett_cdf(h, k, r) for h, k in pts])
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-14)


def test_bivariate_normal_cdf_infinite_limits() -> None:
    inf = jnp.inf
    h = jnp.array([inf, -inf, 0.7, 0.3, inf, -inf])
    k = jnp.array([0.5, 0.2, inf, -inf, inf, -inf])
    got = np.asarray(bivariate_normal_cdf(h, k, 0.4))
    want = np.array([norm.cdf(0.5), 0.0, norm.cdf(0.7), 0.0, 1.0, 0.0])
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-15)


def test_bivariate_normal_cdf_gradients_are_analytic() -> None:
    h, k, r = 0.3, -0.4, 0.6
    grads = jax.grad(bivariate_normal_cdf, argnums=(0, 1, 2))(h, k, r)
    s = np.sqrt(1 - r * r)
    want = (
        norm.pdf(h) * norm.cdf((k - r * h) / s),
        norm.pdf(k) * norm.cdf((h - r * k) / s),
        np.exp(-(h * h - 2 * r * h * k + k * k) / (2 * s * s)) / (2 * np.pi * s),
    )
    np.testing.assert_allclose(np.asarray(grads), want, rtol=1e-13)


def test_smooth_correlation_leaves_positive_definite_matrix() -> None:
    mat = np.array([[1.0, 0.3], [0.3, 1.0]])
    out, done = smooth_correlation(mat)
    assert not done
    np.testing.assert_array_equal(out, mat)


@cache
def _reference() -> dict[str, Any]:
    raw = dict(json.loads((CTT_REFERENCE_DIR / "v24_polychoric.json").read_text(encoding="utf-8")))
    # mirt only supplies the LSAT7 / Science data here; the values come from lavaan and psych.
    check_reference_meta(raw["meta"], mirt_estimates=False)
    return raw


def _frame(case: dict[str, Any]) -> pd.DataFrame:
    rows = [[np.nan if v is None else v for v in row] for row in case["data"]]
    return pd.DataFrame(rows, columns=case["items"], dtype=float)


def _cases() -> list[str]:
    return [c["name"] for c in _reference()["cases"]]


def _case(name: str) -> dict[str, Any]:
    return next(c for c in _reference()["cases"] if c["name"] == name)


@cache
def _fit(name: str) -> PolychoricResult:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return polychoric(_frame(_case(name)), smooth=False)


@pytest.mark.parametrize("name", _cases())
def test_polychoric_matches_lavaan(name: str) -> None:
    case = _case(name)
    res = _fit(name)
    assert len(_frame(case)) == case["n"]
    assert res.converged.to_numpy().all()
    assert not res.at_boundary.to_numpy().any()
    np.testing.assert_allclose(
        res.correlation.to_numpy(), np.asarray(case["lavaan_rho"]), rtol=0, atol=LAVAAN_TOL
    )
    thresholds = np.concatenate([res.thresholds[item] for item in case["items"]])
    np.testing.assert_allclose(thresholds, np.asarray(case["lavaan_th"]), rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", _cases())
def test_polychoric_maximizes_the_two_step_likelihood(name: str) -> None:
    """The gradient of the pair log-likelihood vanishes at the estimate (in atanh rho)."""
    case = _case(name)
    res = _fit(name)
    df = _frame(case)
    items = case["items"]
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            ya, yb = df[items[a]].to_numpy(), df[items[b]].to_numpy()
            both = ~np.isnan(ya) & ~np.isnan(yb)
            ca = np.searchsorted(res.categories[items[a]], ya[both])
            cb = np.searchsorted(res.categories[items[b]], yb[both])
            ka, kb = res.categories[items[a]].size, res.categories[items[b]].size
            table = np.zeros((ka, kb))
            np.add.at(table, (ca, cb), 1.0)
            if res.zero_cell_corrected.iloc[a, b]:
                empty = np.argwhere(table == 0)[0]
                sign = 1.0 if empty[0] == empty[1] else -1.0
                table += sign * 0.5 * np.array([[1.0, -1.0], [-1.0, 1.0]])
            row = np.concatenate([[-np.inf], res.thresholds[items[a]], [np.inf]])
            col = np.concatenate([[-np.inf], res.thresholds[items[b]], [np.inf]])
            z = np.arctanh(float(res.correlation.to_numpy()[a, b]))
            grad = jax.grad(pair_log_likelihood)(
                jnp.float64(z), jnp.asarray(row), jnp.asarray(col), jnp.asarray(table)
            )
            assert abs(float(grad)) <= 1e-7 * table.sum()


@pytest.mark.parametrize("name", ["polychoric_mixed", "lsat7", "science", "tetrachoric_sim"])
def test_polychoric_agrees_with_psych(name: str) -> None:
    """psych's optimize() tolerance (~1.2e-4) limits this check; zero-cell-corrected pairs skipped.

    Missing-data cases are not compared: with unequal category counts psych sets
    ``global = FALSE`` and takes each pair's thresholds from the pair's table margins.
    """
    case = _case(name)
    res = _fit(name)
    keep = ~res.zero_cell_corrected.to_numpy()
    diff = np.abs(res.correlation.to_numpy() - np.asarray(case["psych_rho"]))
    assert diff[keep].max() <= PSYCH_TOL


@pytest.mark.parametrize("name", _cases())
def test_jax_and_numpy_implementations_agree(name: str) -> None:
    """Genz quadrature + autodiff (flexpyirt) vs Owen's T + analytic derivatives (flexpyctt)."""
    data = _frame(_case(name))
    jax_res = _fit(name)
    numpy_res = numpy_polychoric(data, smooth=False)
    assert isinstance(numpy_res, PolychoricResult)
    np.testing.assert_allclose(
        numpy_res.correlation.to_numpy(), jax_res.correlation.to_numpy(), rtol=0, atol=1e-9
    )
    for item in data.columns:
        np.testing.assert_array_equal(numpy_res.thresholds[item], jax_res.thresholds[item])
    for flag in ("converged", "at_boundary", "zero_cell_corrected", "n_obs"):
        pd.testing.assert_frame_equal(getattr(numpy_res, flag), getattr(jax_res, flag))


def test_jax_and_numpy_implementations_agree_at_the_boundary_and_when_smoothing() -> None:
    data = pd.DataFrame({"a": [0, 0, 0, 1, 1, 1, 1, 0], "b": [0, 0, 1, 1, 1, 1, 1, 0]})
    with pytest.warns(HeywoodWarning):
        jax_res = polychoric(data, zero_cell_correction=0.0)
    with pytest.warns(HeywoodWarning):
        numpy_res = numpy_polychoric(data, zero_cell_correction=0.0)
    np.testing.assert_array_equal(numpy_res.at_boundary.to_numpy(), jax_res.at_boundary.to_numpy())
    np.testing.assert_allclose(
        numpy_res.correlation.to_numpy(), jax_res.correlation.to_numpy(), rtol=0, atol=1e-12
    )


def test_zero_cell_correction_is_applied_like_lavaan() -> None:
    res = _fit("tetrachoric_zero_cell")
    assert res.zero_cell_corrected.to_numpy().sum() > 0


def test_smoothing_matches_psych_cor_smooth() -> None:
    ref = _reference()["smooth"]
    out, done = smooth_correlation(np.asarray(ref["input"]))
    assert done
    np.testing.assert_allclose(out, np.asarray(ref["output"]), rtol=0, atol=1e-12)
    assert np.linalg.eigvalsh(out).min() > 0


def test_polychoric_smooths_a_pairwise_matrix_that_is_not_positive_definite() -> None:
    """Three items observed in disjoint pairs with inconsistent signs (A~B, B~C, A~-C)."""
    rng = np.random.default_rng(7)
    n = 400
    blocks = []
    for sign in (1.0, 1.0, -1.0):
        z = rng.normal(size=n)
        e = rng.normal(size=(n, 2)) * 0.3
        blocks.append(((z + e[:, 0]) > 0).astype(float))
        blocks.append(((sign * z + e[:, 1]) > 0).astype(float))
    nan = np.full(n, np.nan)
    data = pd.DataFrame(
        {
            "A": np.concatenate([blocks[0], nan, blocks[4]]),
            "B": np.concatenate([blocks[1], blocks[2], nan]),
            "C": np.concatenate([nan, blocks[3], blocks[5]]),
        }
    )
    with pytest.warns(NotPositiveDefiniteWarning):
        res = polychoric(data)
    assert res.smoothed
    assert res.min_eigenvalue < 0
    assert np.linalg.eigvalsh(res.correlation.to_numpy()).min() > 0
    np.testing.assert_allclose(np.diag(res.correlation.to_numpy()), 1.0)
    assert res.n_obs.loc["A", "B"] == n and res.n_obs.loc["A", "A"] == 2 * n
    unsmoothed = polychoric(data, smooth=False)
    assert not unsmoothed.smoothed
    np.testing.assert_array_equal(
        unsmoothed.correlation.to_numpy(), res.unsmoothed_correlation.to_numpy()
    )


def test_boundary_without_zero_cell_correction() -> None:
    data = pd.DataFrame({"a": [0, 0, 0, 1, 1, 1, 1, 0], "b": [0, 0, 1, 1, 1, 1, 1, 0]})
    with pytest.warns(HeywoodWarning):
        res = polychoric(data, zero_cell_correction=0.0)
    assert res.at_boundary.loc["a", "b"]
    assert res.correlation.to_numpy()[0, 1] > 0.9999
    corrected = polychoric(data)
    assert corrected.zero_cell_corrected.loc["a", "b"]
    assert not corrected.at_boundary.loc["a", "b"]


def test_category_codes_only_need_order() -> None:
    rng = np.random.default_rng(3)
    base = rng.integers(0, 3, size=(300, 3)).astype(float)
    base[:, 1] = np.clip(base[:, 0] + rng.integers(-1, 2, size=300), 0, 2)
    recoded = pd.DataFrame({"x": base[:, 0] * 10 - 5, "y": base[:, 1] ** 2, "z": base[:, 2]})
    plain = polychoric(pd.DataFrame(base, columns=["x", "y", "z"]))
    np.testing.assert_allclose(
        polychoric(recoded).correlation.to_numpy(), plain.correlation.to_numpy(), atol=1e-14
    )
    np.testing.assert_array_equal(plain.categories["y"], [0.0, 1.0, 2.0])
    array_res = polychoric(base)
    assert list(array_res.correlation.columns) == ["item1", "item2", "item3"]


@pytest.mark.parametrize(
    ("data", "match"),
    [
        (pd.DataFrame({"a": [0, 1, 1], "b": [1, 1, 1]}), "observed category"),
        (pd.DataFrame({"a": ["x", "y"], "b": [0, 1]}), "numeric"),
        (pd.DataFrame({"a": [0, 1, 0, 1]}), "at least two items"),
        (
            pd.DataFrame({"a": [0, 1, np.nan, np.nan], "b": [np.nan, np.nan, 0, 1]}),
            "no respondent",
        ),
        (
            pd.DataFrame({"a": [0, 1, 0, 1], "b": [0, 0, 1, np.nan], "c": [1, 1, np.nan, 0]}),
            "constant",
        ),
        (np.zeros(5), "two-dimensional"),
        (np.array([[0.0, 1.0], [np.inf, 0.0], [1.0, 1.0]]), "finite"),
    ],
)
def test_invalid_data(data: Any, match: str) -> None:
    with pytest.raises(SpecificationError, match=match):
        polychoric(data)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"zero_cell_correction": -0.5}, "zero_cell_correction"),
        ({"tol": 0.0}, "tol"),
        ({"max_iter": 0}, "max_iter"),
    ],
)
def test_invalid_options(kwargs: dict[str, Any], match: str) -> None:
    data = pd.DataFrame({"a": [0, 1, 0, 1, 1], "b": [0, 1, 1, 1, 0]})
    with pytest.raises(SpecificationError, match=match):
        polychoric(data, **kwargs)


def test_requires_x64() -> None:
    data = pd.DataFrame({"a": [0, 1, 0, 1, 1], "b": [0, 1, 1, 1, 0]})
    jax.config.update("jax_enable_x64", False)  # type: ignore[no-untyped-call]
    try:
        with pytest.raises(PrecisionError):
            polychoric(data)
    finally:
        jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
