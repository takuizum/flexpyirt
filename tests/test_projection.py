"""Unit tests for projective IRT (``project()``, DESIGN.md §32).

The mirt comparison (V61) is in ``test_mirt_v61_projection.py``. Here: the closed forms against
numerical integration and Ip's (2010) worked example, the marginal table against the confirmatory
path, the two-tier path, the result object, and the errors.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from scipy import special, stats

from flexpyirt import (
    BAEM,
    GPCM,
    GRM,
    FitResult,
    Model,
    Quadrature,
    ThreePL,
    TwoPL,
    fit,
    project,
)
from flexpyirt._projection import (
    LOGISTIC_KERNEL_CONSTANT,
    NORMAL_OGIVE_CONSTANT,
    kernel_scale,
)
from flexpyirt.errors import SpecificationError
from flexpyirt.items import ItemModel

QUAD = Quadrature.equispaced(21, bounds=(-6.0, 6.0))
ITEMS = [f"i{k}" for k in range(8)]


def _simulate(
    slopes: np.ndarray, intercepts: np.ndarray, n: int, seed: int, cov: np.ndarray | None = None
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_factors = slopes.shape[1]
    cov = np.eye(n_factors) if cov is None else cov
    theta = rng.multivariate_normal(np.zeros(n_factors), cov, size=n)
    z = theta @ slopes.T + intercepts
    y = (rng.random(z.shape) < special.expit(z)).astype(float)
    return pd.DataFrame(y, columns=[f"i{k}" for k in range(slopes.shape[0])])


@cache
def _one_specific() -> tuple[pd.DataFrame, FitResult, FitResult]:
    """Bifactor model with one specific factor and the same model as a 2-factor CFA."""
    slopes = np.zeros((8, 2))
    slopes[:, 0] = np.linspace(0.9, 1.8, 8)
    slopes[:4, 1] = [1.2, 0.9, 1.4, 1.0]
    data = _simulate(slopes, np.linspace(-1.0, 1.0, 8), 600, 3)
    data.iloc[::17, 2] = np.nan
    base = Model.uniform(TwoPL(), items=ITEMS)
    bifactor = fit(
        base.with_bifactor(specific={"s": ITEMS[:4]}, general="g"),
        data,
        quadrature=QUAD,
        method=BAEM(tol=1e-10),
    )
    cfa = fit(
        base.with_factors({"g": ITEMS, "s": ITEMS[:4]}, correlated=False),
        data,
        quadrature=QUAD,
        method=BAEM(tol=1e-10),
    )
    return data, bifactor, cfa


def _value(rows: pd.DataFrame, item: str, param: str) -> float:
    """Estimate of ``(item, param)`` in a frame from :func:`_item_rows`."""
    return float(rows["estimate"].to_dict()[(item, param)])


def _item_rows(result: FitResult) -> pd.DataFrame:
    rows = result.parameters[result.parameters["item"] != "__latent__"]
    return rows.set_index(["item", "param"])


# ---- closed forms ---------------------------------------------------------------------------


def test_logistic_kernel_reproduces_ip_example() -> None:
    """Ip (2010) / Chalmers et al. (2026): a = 2.63, a_s = 3.07, d = -1.49 -> 1.274, -0.721."""
    a_star, scale = kernel_scale(jnp.asarray([2.63, 3.07]), jnp.eye(2), 0, LOGISTIC_KERNEL_CONSTANT)
    assert float(a_star) == pytest.approx(1.274, abs=5e-4)
    assert float(scale * -1.49) == pytest.approx(-0.721, abs=1e-3)


@pytest.mark.parametrize("target", [0, 1, 2])
def test_closed_form_is_exact_for_the_normal_ogive(target: int) -> None:
    r"""With :math:`P=\Phi(c(\mathbf a^\top\boldsymbol\theta+d))` the closed form is exact."""
    slopes = np.array([1.3, 0.8, -0.6])
    d = 0.4
    phi = np.array([[1.0, 0.3, 0.2], [0.3, 1.0, -0.4], [0.2, -0.4, 1.0]])
    c = NORMAL_OGIVE_CONSTANT
    a_star, scale = kernel_scale(jnp.asarray(slopes), jnp.asarray(phi), target, c)
    others = [m for m in range(3) if m != target]
    s_ff = phi[target, target]
    cond_cov = (
        phi[np.ix_(others, others)] - np.outer(phi[others, target], phi[target, others]) / s_ff
    )
    nodes, weights = np.polynomial.hermite_e.hermegauss(40)
    weights = weights / weights.sum()
    chol = np.linalg.cholesky(cond_cov)
    z = np.stack(np.meshgrid(nodes, nodes, indexing="ij"), axis=-1).reshape(-1, 2)
    w = np.outer(weights, weights).ravel()
    for theta_f in (-1.5, 0.0, 0.7):
        rest = (phi[others, target] / s_ff) * theta_f + z @ chol.T
        eta = slopes[target] * theta_f + rest @ slopes[others] + d
        marginal = float(np.sum(w * stats.norm.cdf(c * eta)))
        approx = float(stats.norm.cdf(c * (float(a_star) * theta_f + float(scale) * d)))
        assert marginal == pytest.approx(approx, abs=1e-10)


def test_normal_ogive_matches_standardized_loadings() -> None:
    """Stucky et al. (2013): a* = 1.702 lambda_1 / sqrt(1 - lambda_1^2) for orthogonal factors."""
    a = np.array([1.7, 1.1, 0.0])
    lam = (a / 1.702) / np.sqrt(1.0 + np.sum((a / 1.702) ** 2))
    a_star, _ = kernel_scale(jnp.asarray(a), jnp.eye(3), 0, NORMAL_OGIVE_CONSTANT)
    assert float(a_star) == pytest.approx(1.702 * lam[0] / np.sqrt(1.0 - lam[0] ** 2), rel=1e-12)


# ---- MML ------------------------------------------------------------------------------------


@pytest.mark.slow
def test_mml_bifactor_equals_confirmatory_path() -> None:
    """The block-grid and product-grid marginal tables give the same projection."""
    data, bifactor, cfa = _one_specific()
    p_bif = project(bifactor, data)
    p_cfa = project(cfa, data)
    a, b = _item_rows(p_bif), _item_rows(p_cfa)
    np.testing.assert_allclose(a["estimate"], b["estimate"], atol=2e-5)
    np.testing.assert_allclose(a["se"], b["se"], rtol=1e-3)
    assert p_bif.converged
    assert p_bif.se_method == "projection_oakes"


@pytest.mark.slow
def test_mml_is_close_to_closed_forms_and_attenuates_slopes() -> None:
    data, bifactor, _ = _one_specific()
    mml = _item_rows(project(bifactor, data))
    lka = _item_rows(project(bifactor, data, method="logistic_kernel"))
    parent = _item_rows(bifactor)
    np.testing.assert_allclose(mml["estimate"], lka["estimate"], atol=0.05)
    for item in ITEMS[:4]:
        # Items with a specific loading are attenuated; general-only items are not.
        assert _value(mml, item, "a") < _value(parent, item, "a1")
    for item in ITEMS[4:]:
        assert _value(lka, item, "a") == pytest.approx(_value(parent, item, "a1"), rel=1e-12)


@pytest.mark.slow
def test_mml_marginal_table_matches_category_counts() -> None:
    """Summing the marginal table over nodes gives each item's observed category counts."""
    from flexpyirt import _core
    from flexpyirt._compile import TABLE_COLUMNS, compile_problem, prepare
    from flexpyirt._projection import marginal_item_counts

    data, bifactor, _ = _one_specific()
    prep = prepare(bifactor._model, data, group=None, item_names=None, table=None)
    prob = compile_problem(
        bifactor._model, prep.y, prep.group, bifactor.parameters[list(TABLE_COLUMNS)], QUAD
    ).problem
    r_marg, n_marg = marginal_item_counts(
        prob, jnp.asarray(bifactor._eta), 0, jnp.asarray(QUAD.nodes())
    )
    observed = np.array([[(data[i] == k).sum() for k in (0, 1)] for i in ITEMS])
    np.testing.assert_allclose(np.asarray(r_marg[0][0]).sum(axis=-1), observed, rtol=1e-10)
    assert float(np.asarray(n_marg).sum()) == pytest.approx(len(data), rel=1e-10)
    counts, _ = _core.e_step(prob, jnp.asarray(bifactor._eta))
    assert np.asarray(counts.r[0]).shape[-1] == QUAD.n_points**2


@pytest.mark.slow
def test_two_tier_projection_onto_each_primary_factor() -> None:
    slopes = np.zeros((8, 4))
    slopes[:4, 0] = [1.2, 1.4, 1.0, 1.3]
    slopes[4:, 1] = [1.1, 1.5, 1.2, 0.9]
    slopes[3, 1] = 0.6
    slopes[[0, 1, 4, 5], 2] = 0.9
    slopes[[2, 3, 6, 7], 3] = 0.8
    cov = np.eye(4)
    cov[0, 1] = cov[1, 0] = 0.5
    data = _simulate(slopes, np.linspace(-0.8, 0.8, 8), 800, 5, cov)
    spec = Model.uniform(TwoPL(), items=ITEMS).with_two_tier(
        primary={"f1": ITEMS[:4], "f2": ITEMS[3:]},
        specific={"s1": ["i0", "i1", "i4", "i5"], "s2": ["i2", "i3", "i6", "i7"]},
    )
    parent = fit(
        spec,
        data,
        quadrature=Quadrature.equispaced(11, bounds=(-5.0, 5.0)),
        method=BAEM(squarem=True, max_iter=5000),
    )
    for name in ("f1", "f2"):
        mml = _item_rows(project(parent, data, factor=name))
        lka = _item_rows(project(parent, data, factor=name, method="logistic_kernel"))
        # Items without a loading on the target factor still relate to it via the correlation.
        assert (mml.xs("a", level="param")["estimate"] > 0.2).all()
        np.testing.assert_allclose(mml["estimate"], lka["estimate"], atol=0.1)
        assert np.isfinite(lka["se"]).all()


# ---- result object --------------------------------------------------------------------------


@pytest.mark.slow
def test_projected_result_is_a_working_unidimensional_fit(tmp_path: Path) -> None:
    data, bifactor, _ = _one_specific()
    projected = project(bifactor, data)
    assert not projected._model.is_multidimensional
    assert projected.n_params == 16
    assert np.isfinite(projected.loglik)
    scores = projected.score(data.iloc[:5])
    assert scores.shape[0] == 5
    path = tmp_path / "projected.json"
    projected.to_json(path)
    loaded = FitResult.from_json(path)
    assert loaded.se_method == "projection_oakes"
    np.testing.assert_allclose(
        loaded.parameters["estimate"], projected.parameters["estimate"], rtol=1e-12
    )
    no_se = project(bifactor, data, se=None)
    assert no_se.vcov is None and no_se.se_method is None


# ---- errors ---------------------------------------------------------------------------------


@cache
def _small_fit(item_model: str) -> tuple[pd.DataFrame, FitResult]:
    slopes = np.zeros((6, 2))
    slopes[:, 0] = 1.2
    slopes[:3, 1] = 0.8
    data = _simulate(slopes, np.zeros(6), 300, 2)
    items = ITEMS[:6]
    if item_model == "gpcm":
        data["i5"] = data["i5"] + (data["i4"] == 1) * data["i5"]
        models: dict[str, ItemModel] = {i: TwoPL() for i in items[:5]} | {"i5": GPCM(3)}
        spec = Model.from_items(models).with_bifactor(specific={"s": items[:3]})
    elif item_model == "3pl":
        models = {i: TwoPL() for i in items[1:]} | {"i0": ThreePL()}
        spec = Model.from_items(models).with_bifactor(specific={"s": items[:3]})
    elif item_model == "unidimensional":
        spec = Model.uniform(TwoPL(), items=items)
    else:
        spec = Model.uniform(TwoPL(), items=items).with_bifactor(specific={"s": items[:3]})
    return data, fit(spec, data, quadrature=QUAD, se=None)


@pytest.mark.slow
def test_rejects_unidimensional_parent() -> None:
    data, result = _small_fit("unidimensional")
    with pytest.raises(SpecificationError, match="multidimensional"):
        project(result, data)


@pytest.mark.slow
def test_rejects_specific_factor_and_unknown_names() -> None:
    data, result = _small_fit("bifactor")
    with pytest.raises(SpecificationError, match="specific factor"):
        project(result, data, factor="s")
    with pytest.raises(SpecificationError, match="Unknown factor"):
        project(result, data, factor="nope")
    with pytest.raises(SpecificationError, match="Unknown method"):
        project(result, data, method="LKA")  # type: ignore[arg-type]


@pytest.mark.slow
@pytest.mark.parametrize(("item_model", "name"), [("3pl", "3PL"), ("gpcm", "GPCM")])
@pytest.mark.parametrize("method", ["MML", "logistic_kernel", "normal_ogive"])
def test_rejects_item_models_other_than_2pl_and_grm(
    item_model: str, name: str, method: str
) -> None:
    """Projection is defined for the logistic kernels (2PL, GRM) only, whatever the method."""
    data, result = _small_fit(item_model)
    with pytest.raises(SpecificationError, match=name):
        project(result, data, method=method)  # type: ignore[arg-type]


@pytest.mark.slow
def test_closed_form_without_parent_covariance_has_no_se() -> None:
    data, result = _small_fit("bifactor")
    projected = project(result, data, method="normal_ogive")
    assert projected.vcov is None and projected.se_method is None
    assert projected.n_iter == 0 and projected.converged


@pytest.mark.slow
def test_grm_projection_scales_every_threshold() -> None:
    rng = np.random.default_rng(4)
    theta = rng.normal(size=(400, 2))
    z = 1.3 * theta[:, [0]] + np.where(np.arange(6) < 3, 0.9, 0.0) * theta[:, [1]]
    y = np.stack([np.digitize(z[:, j] + rng.logistic(size=400), [-1.0, 1.0]) for j in range(6)], 1)
    data = pd.DataFrame(y.astype(float), columns=ITEMS[:6])
    spec = Model.uniform(GRM(3), items=ITEMS[:6]).with_bifactor(specific={"s": ITEMS[:3]})
    parent = fit(spec, data, quadrature=QUAD, se=None)
    rows = _item_rows(project(parent, data, method="logistic_kernel"))
    prow = _item_rows(parent)
    ratio = _value(rows, "i0", "a") / _value(prow, "i0", "a1")
    for p in ("d1", "d2"):
        assert _value(rows, "i0", p) == pytest.approx(ratio * _value(prow, "i0", p), rel=1e-12)
