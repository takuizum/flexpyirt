"""Multidimensional scale linking, theta* = A theta + beta, against base R (VALIDATION V40).

Reference values come from ``validation/r/m5_linking_extensions_reference.R`` ->
``tests/reference/v40_linking_multidim.json``. R ``mirt`` does not implement scale linking: each
form of a confirmatory two- or three-factor model is calibrated with mirt (``TOL=1e-10``, Oakes
SEs, factor means 0, variances 1, free correlations; 31 nodes per dimension for two factors and 15
for three, on [-6, 6]), and the multidimensional Haebara and Stocking–Lord criteria (Oshima, Davey
& Lee, 2000; non-symmetric and symmetric) are minimized in base R with ``optim`` over a diagonal,
orthogonal (Li & Lissitz, 2000) or oblique ``A``, using ``mirt::probtrace()`` and
``mirt::expected.item()``. Ogasawara (2001) delta-method covariances of ``(vec(A), beta)`` come
from category probabilities written out in R and nested central finite differences.

The checks on mirt's own item estimates and covariance blocks need no calibration; those of the
two-factor 2PL case run in the fast tier. Checks through :func:`flexpyirt.link` are slow.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from flexpyirt import BAEM, GRM, FitResult, Model, Quadrature, TwoPL, fit, link
from flexpyirt._linking import (
    LinkingFamily,
    LinkingProblem,
    linking_constants,
    multidim_linking_vcov,
    solve_multidim_linking,
)
from flexpyirt.items import ItemKernel, ItemModel
from flexpyirt.latent import mvn_log_weights
from tests._reference import check_reference_meta

REFERENCE_DIR = Path(__file__).parent / "reference"
REFERENCE_FILE = "v40_linking_multidim.json"

CASES = ["v40d_2pl_2d", "v40e_grm_2d", "v40f_2pl_3d"]
FAST_CASE = "v40d_2pl_2d"
TRANSFORMATIONS = ["diagonal", "orthogonal", "oblique"]
CRITERIA = ["stocking_lord", "stocking_lord_sym", "haebara", "haebara_sym"]
VARIANTS = [f"{t}_{c}" for t in TRANSFORMATIONS for c in CRITERIA]

# Linking on identical mirt item estimates: limited by the R reference's optimization accuracy
# (central-FD gradient polish).
TOL_CONST_DIRECT = 1e-8
TOL_CRIT_REL_DIRECT = 1e-9
# Delta-method covariance on identical inputs: limited by the R reference's nested central
# finite differences (step 1e-5); absolute tolerance relative to the largest element.
TOL_VCOV_REL_DIRECT = 1e-4
# End-to-end: both forms calibrated in flexpyirt.
TOL_CONST_FULL = 2e-4
TOL_CRIT_REL_FULL = 1e-3
TOL_VCOV_REL_FULL = 2e-2

_solve = jax.jit(solve_multidim_linking, static_argnames=("method", "symmetric", "transformation"))
_vcov = jax.jit(multidim_linking_vcov, static_argnames=("method", "symmetric", "transformation"))


def _grid(case_names: list[str], *, fast: bool = True) -> list[Any]:
    """(case, variant) pairs; only the two-factor 2PL case is in the fast tier."""
    return [
        pytest.param(c, v, marks=() if (fast and c == FAST_CASE) else pytest.mark.slow)
        for c in case_names
        for v in VARIANTS
    ]


@cache
def _bundle() -> dict[str, Any]:
    raw = dict(json.loads((REFERENCE_DIR / REFERENCE_FILE).read_text()))
    check_reference_meta(raw["meta"], require_grid_check=True)
    return raw


@cache
def _case(name: str) -> dict[str, Any]:
    raw = dict(_bundle()["cases"][name])
    check_reference_meta(raw["meta"], require_grid_check=True)
    return raw


def _ref_constants(ref: dict[str, Any]) -> np.ndarray:
    """Reference ``(vec(A) row-major, beta)``."""
    return np.concatenate([np.asarray(ref["A"], dtype=float).reshape(-1), np.ravel(ref["beta"])])


def _form_model(form: dict[str, Any]) -> Model:
    def item_model(item: str) -> ItemModel:
        if form["itemtypes"][item] == "graded":
            return GRM(n_categories=int(form["n_cats"][item]))
        return TwoPL()

    return Model.from_items({item: item_model(item) for item in form["items"]}).with_factors(
        {f: list(items) for f, items in form["factors"].items()}
    )


def _form_data(form: dict[str, Any]) -> pd.DataFrame:
    pats = pd.DataFrame(np.asarray(form["patterns"], dtype=int), columns=form["items"])
    counts = np.asarray(form["counts"], dtype=int)
    return pats.loc[pats.index.repeat(counts)].reset_index(drop=True)


def _quadrature(case: dict[str, Any]) -> Quadrature:
    meta = case["meta"]
    return Quadrature.equispaced(int(meta["quadpts"]), bounds=tuple(meta["theta_lim"]))


@cache
def _fixed_at_mirt(name: str, side: str) -> FitResult:
    """Create a FitResult with all parameters fixed at mirt's estimates (zero calibration diff)."""
    case = _case(name)
    form = case[side]
    spec = _form_model(form)
    data = _form_data(form)
    table = spec.parameter_table(data)
    vals = [
        float(form["latent"][str(param)])
        if item == "__latent__"
        else float(form["params"][item][param])
        for item, param in zip(table["item"], table["param"], strict=True)
    ]
    table["value"] = np.asarray(vals, dtype=float)
    table["free"] = False
    return fit(spec, data, table=table, quadrature=_quadrature(case), se=None)


@cache
def _fit_form(name: str, side: str) -> FitResult:
    """Calibrate one form from scratch in flexpyirt with tight tolerance and Oakes SEs."""
    case = _case(name)
    form = case[side]
    return fit(
        _form_model(form),
        _form_data(form),
        method=BAEM(tol=1e-9, max_iter=50000, squarem=True),
        quadrature=_quadrature(case),
        se="oakes",
    )


@cache
def _direct_problem(name: str) -> tuple[LinkingProblem, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Linking problem on mirt's common-item estimates with mirt's covariance blocks."""
    case = _case(name)
    n_factors = int(case["meta"]["n_factors"])
    nodes = jnp.asarray(_quadrature(case).grid(n_factors), dtype=jnp.float64)

    def weights(form: dict[str, Any]) -> jax.Array:
        spec = _form_model(form)
        params = jnp.asarray([float(form["latent"][p]) for p in spec.latent_parameter_names()])
        return jnp.exp(mvn_log_weights(nodes, params))

    spec = _form_model(case["target"])
    by_kernel: dict[ItemKernel, list[str]] = {}
    for item in case["common_items"]:
        by_kernel.setdefault(spec.item_kernel(dict(spec.items)[item]), []).append(item)

    sides = {}
    for side in ("source", "target"):
        cv = case["common_vcov"][side]
        pos = {(str(i), str(p)): k for k, (i, p) in enumerate(cv["labels"])}
        sides[side] = (
            pos,
            jnp.asarray(cv["values"], dtype=jnp.float64),
            jnp.asarray(np.asarray(cv["vcov"], dtype=float).reshape(len(pos), len(pos))),
        )

    families = []
    for kernel, items in by_kernel.items():
        arrays = {}
        for side in ("source", "target"):
            pos = sides[side][0]
            params = np.array(
                [[float(case[side]["params"][i][p]) for p in kernel.param_names] for i in items]
            )
            index = np.array(
                [[pos.get((i, p), -1) for p in kernel.param_names] for i in items], dtype=np.int32
            )
            arrays[side] = (jnp.asarray(params), jnp.asarray(index))
        families.append(
            LinkingFamily(
                kernel=kernel,
                src_params=arrays["source"][0],
                tgt_params=arrays["target"][0],
                src_free_index=arrays["source"][1],
                tgt_free_index=arrays["target"][1],
            )
        )
    prob = LinkingProblem(
        families=tuple(families),
        tgt_nodes=nodes,
        tgt_weights=weights(case["target"]),
        src_nodes=nodes,
        src_weights=weights(case["source"]),
    )
    return prob, sides["source"][1], sides["target"][1], sides["source"][2], sides["target"][2]


def _neutral_start(n_factors: int, transformation: str) -> jax.Array:
    """The identity transformation in each parameterization."""
    if transformation == "diagonal":
        return jnp.concatenate([jnp.ones(n_factors), jnp.zeros(n_factors)])
    if transformation == "orthogonal":
        return jnp.zeros(1 + n_factors * (n_factors - 1) // 2 + n_factors).at[0].set(1.0)
    return jnp.concatenate([jnp.eye(n_factors).reshape(-1), jnp.zeros(n_factors)])


def _assert_vcov_close(actual: np.ndarray, expected: np.ndarray, rel: float) -> None:
    """Element-wise agreement, with tolerance relative to the largest reference element."""
    np.testing.assert_allclose(actual, expected, rtol=rel, atol=rel * float(np.max(expected)))


def test_reference_structure() -> None:
    """Fixed zero slopes are not free parameters; cross-loading items have both slopes."""
    case = _case(FAST_CASE)
    labels = {(i, p) for i, p in case["common_vcov"]["source"]["labels"]}
    assert ("c01", "a1") in labels and ("c01", "a2") not in labels
    assert ("c05", "a2") in labels and ("c05", "a1") not in labels
    assert ("c09", "a1") in labels and ("c09", "a2") in labels
    assert float(case["source"]["params"]["c01"]["a2"]) == 0.0
    for name in CASES:
        d = int(_case(name)["meta"]["n_factors"])
        for variant in VARIANTS:
            ref = _case(name)["linking"][variant]
            assert np.asarray(ref["A"]).shape == (d, d)
            assert np.asarray(ref["vcov"]).shape == (d * d + d, d * d + d)
            if ref["transformation"] == "orthogonal":  # A = k R with R a proper rotation
                a_mat = np.asarray(ref["A"], dtype=float)
                dilation = float(np.asarray(ref["par"], dtype=float)[0])
                np.testing.assert_allclose(a_mat.T @ a_mat, dilation**2 * np.eye(d), atol=1e-12)
                assert np.linalg.det(a_mat) > 0.0


@pytest.mark.parametrize(("case_name", "variant"), _grid(CASES))
def test_constants_criterion_and_vcov_on_mirt_estimates_match_r_reference(
    case_name: str, variant: str
) -> None:
    """On mirt's item estimates and covariance blocks: (A, beta), criterion and delta-method vcov.

    The solver starts from the identity transformation, not from the reference solution.
    """
    case = _case(case_name)
    ref = case["linking"][variant]
    n_factors = int(case["meta"]["n_factors"])
    prob, eta_s, eta_t, vcov_s, vcov_t = _direct_problem(case_name)
    kwargs = {
        "method": ref["method"],
        "symmetric": bool(ref["symmetric"]),
        "transformation": ref["transformation"],
    }
    c_hat, crit, gnorm, _, conv = _solve(
        prob,
        eta_s,
        eta_t,
        _neutral_start(n_factors, ref["transformation"]),
        tol=1e-11,
        max_iter=200,
        **kwargs,
    )
    assert bool(conv), f"not converged: max|grad| = {float(gnorm):.2e}"
    constants = np.asarray(linking_constants(c_hat, n_factors, ref["transformation"]))
    np.testing.assert_allclose(constants, _ref_constants(ref), atol=TOL_CONST_DIRECT)
    np.testing.assert_allclose(
        np.asarray(c_hat), np.asarray(ref["par"], dtype=float), atol=TOL_CONST_DIRECT
    )
    assert float(crit) == pytest.approx(float(ref["criterion"]), rel=TOL_CRIT_REL_DIRECT)

    cov, cov_c, valid = _vcov(c_hat, eta_s, eta_t, vcov_s, vcov_t, prob, **kwargs)
    assert bool(valid)
    _assert_vcov_close(np.asarray(cov), np.asarray(ref["vcov"], dtype=float), TOL_VCOV_REL_DIRECT)
    _assert_vcov_close(
        np.asarray(cov_c), np.asarray(ref["vcov_par"], dtype=float), TOL_VCOV_REL_DIRECT
    )


@pytest.mark.parametrize("case_name", CASES)
def test_estimates_recover_the_simulated_transformation(case_name: str) -> None:
    """Diagonal and oblique optima are within 4 delta-method SEs of the simulated (A, beta).

    The simulated transformation is diagonal with unequal scales, which the orthogonal family
    (one common dilation) does not contain.
    """
    case = _case(case_name)
    truth = np.concatenate(
        [np.asarray(case["truth"]["A"], dtype=float).reshape(-1), np.ravel(case["truth"]["beta"])]
    )
    for variant in VARIANTS:
        ref = case["linking"][variant]
        if ref["transformation"] == "orthogonal":
            continue
        se = np.sqrt(np.clip(np.diag(np.asarray(ref["vcov"], dtype=float)), 0.0, None))
        free = se > 0.0
        err = np.abs(_ref_constants(ref) - truth)
        assert np.all(err[free] < 4.0 * se[free]), (variant, err, se)
        assert np.all(err[~free] == 0.0)


@pytest.mark.slow
@pytest.mark.parametrize(("case_name", "variant"), _grid(CASES, fast=False))
def test_link_on_mirt_estimates_matches_r_reference(case_name: str, variant: str) -> None:
    """``link()`` on FitResults fixed at mirt's estimates, from the library's starting values."""
    case = _case(case_name)
    ref = case["linking"][variant]
    n_factors = int(case["meta"]["n_factors"])
    linked = link(
        _fixed_at_mirt(case_name, "source"),
        _fixed_at_mirt(case_name, "target"),
        items=case["common_items"],
        method=ref["method"],
        symmetric=bool(ref["symmetric"]),
        transformation=ref["transformation"],
    )
    assert linked.converged
    assert linked.transformation == ref["transformation"]
    np.testing.assert_allclose(linked.A, np.asarray(ref["A"], dtype=float), atol=TOL_CONST_DIRECT)
    np.testing.assert_allclose(linked.B, np.ravel(ref["beta"]), atol=TOL_CONST_DIRECT)
    assert linked.criterion == pytest.approx(float(ref["criterion"]), rel=TOL_CRIT_REL_DIRECT)
    # Transformed common items reproduce the source curves at A^{-1} (theta - beta).
    src = case["source"]
    spec = _form_model(src)
    theta = jnp.asarray(np.random.default_rng(0).normal(size=(25, n_factors)))
    theta_star = theta @ jnp.asarray(linked.A).T + jnp.asarray(linked.B)
    items = linked.items.set_index("item")
    for name in case["common_items"]:
        kernel = spec.item_kernel(dict(spec.items)[name])
        xi_src = jnp.asarray([float(src["params"][name][p]) for p in kernel.param_names])
        xi_star = jnp.asarray([float(items.loc[name, p]) for p in kernel.param_names])
        np.testing.assert_allclose(
            np.asarray(kernel.log_prob(theta_star, xi_star)),
            np.asarray(kernel.log_prob(theta, xi_src)),
            atol=1e-12,
        )


@pytest.mark.slow
@pytest.mark.parametrize(("case_name", "variant"), _grid(CASES, fast=False))
def test_end_to_end_fit_and_link_matches_r_reference(case_name: str, variant: str) -> None:
    """Fit both forms in flexpyirt, link, and compare (A, beta), criterion and SEs."""
    case = _case(case_name)
    ref = case["linking"][variant]
    n_factors = int(case["meta"]["n_factors"])
    source = _fit_form(case_name, "source")
    target = _fit_form(case_name, "target")
    assert source.converged and target.converged
    for res, side in ((source, "source"), (target, "target")):
        assert res.loglik == pytest.approx(float(case[side]["loglik"]), rel=1e-6)
    linked = link(
        source,
        target,
        items=case["common_items"],
        method=ref["method"],
        symmetric=bool(ref["symmetric"]),
        transformation=ref["transformation"],
    )
    assert linked.converged
    np.testing.assert_allclose(linked.A, np.asarray(ref["A"], dtype=float), atol=TOL_CONST_FULL)
    np.testing.assert_allclose(linked.B, np.ravel(ref["beta"]), atol=TOL_CONST_FULL)
    assert linked.criterion == pytest.approx(float(ref["criterion"]), rel=TOL_CRIT_REL_FULL)
    assert linked.vcov is not None
    ref_vcov = np.asarray(ref["vcov"], dtype=float)
    _assert_vcov_close(linked.vcov, ref_vcov, TOL_VCOV_REL_FULL)
    ses = np.sqrt(np.clip(np.diag(linked.vcov), 0.0, None))
    np.testing.assert_allclose(linked.A_se.reshape(-1), ses[: n_factors**2], atol=0.0)
    np.testing.assert_allclose(linked.B_se, ses[n_factors**2 :], atol=0.0)
