"""Numerical agreement of IRT scale linking with base-R references on mirt fits (VALIDATION V20).

Reference values come from ``validation/r/m5_linking_reference.R`` ->
``tests/reference/v20_linking.json``. Because R ``mirt`` does not implement scale linking, each
form is calibrated with mirt (``optimizer="NR"``, ``TOL=1e-10``, 61 nodes on [-6, 6], Oakes SEs;
latent N(0, 1) except the V20e source form, fixed at N(0.3, 1.5)), and the Haebara (1980) and
Stocking–Lord (1983) criteria (non-symmetric and Kim & Lee, 2006 symmetric variants) plus
Ogasawara (2001) delta-method SEs are computed in base R directly from the formulas using
``mirt::probtrace()`` and ``mirt::expected.item()``.

Tests separate linking accuracy (linking directly on mirt's item estimates, and delta-method SEs
from mirt's own covariance blocks) from end-to-end calibration + linking accuracy (fitting both
forms in flexpyirt and linking).
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

from flexpyirt import BAEM, GPCM, GRM, FitResult, Model, Quadrature, ThreePL, TwoPL, fit, link
from flexpyirt._linking import LinkingFamily, LinkingProblem, linking_vcov, solve_linking
from flexpyirt.items import ItemKernel, ItemModel
from flexpyirt.latent import normal_log_weights
from tests._reference import check_reference_meta

# mirt reference comparison: nightly and release-PR CI only (CONTRIBUTING.md, "Tests").
pytestmark = pytest.mark.slow

REFERENCE_DIR = Path(__file__).parent / "reference"
REFERENCE_FILE = "v20_linking.json"

CASES = ["v20a_2pl", "v20b_grm", "v20c_gpcm", "v20d_3pl", "v20e_2pl_srcgrid", "v20f_mixed"]
VARIANTS = ["stocking_lord", "stocking_lord_sym", "haebara", "haebara_sym"]

# Linking on identical mirt item estimates (no calibration difference): limited by the R
# reference's own optimization accuracy (central-FD gradient polish, max |grad| <= 1.9e-11).
TOL_AB_DIRECT = 1e-9
TOL_CRIT_REL_DIRECT = 1e-9
# Delta-method SEs on identical inputs (mirt's Oakes covariance blocks): limited by the R
# reference's nested central finite differences (step 1e-5).
TOL_SE_REL_DIRECT = 1e-5
# End-to-end: both forms calibrated in flexpyirt (item parameters agree with mirt to ~1e-6).
TOL_AB_FULL = 1e-6
TOL_CRIT_REL_FULL = 1e-5
TOL_SE_REL = 1e-3

_solve = jax.jit(solve_linking, static_argnames=("method", "symmetric"))
_vcov = jax.jit(linking_vcov, static_argnames=("method", "symmetric"))


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


def _item_model(itemtype: str, n_categories: int) -> ItemModel:
    if itemtype == "2PL":
        return TwoPL()
    if itemtype == "3PL":
        return ThreePL()
    if itemtype == "graded":
        return GRM(n_categories=n_categories)
    if itemtype == "gpcm":
        return GPCM(n_categories=n_categories)
    raise ValueError(f"Unsupported reference itemtype {itemtype!r}")


def _form_model(form: dict[str, Any]) -> Model:
    return Model.from_items(
        {
            item: _item_model(str(form["itemtypes"][item]), int(form["n_cats"][item]))
            for item in form["items"]
        }
    )


def _form_data(form: dict[str, Any]) -> pd.DataFrame:
    pats = pd.DataFrame(np.asarray(form["patterns"], dtype=int), columns=form["items"])
    counts = np.asarray(form["counts"], dtype=int)
    return pats.loc[pats.index.repeat(counts)].reset_index(drop=True)


def _latent_value(form: dict[str, Any], param: str) -> float:
    return float(form["latent_mean"] if param == "mean" else form["latent_var"])


@cache
def _fixed_at_mirt(name: str, side: str) -> FitResult:
    """Create a FitResult with all parameters fixed at mirt's estimates (zero calibration diff)."""
    case = _case(name)
    form = case[side]
    meta = case["meta"]
    spec = _form_model(form)
    data = _form_data(form)
    quad = Quadrature.equispaced(int(meta["quadpts"]), bounds=tuple(meta["theta_lim"]))
    table = spec.parameter_table(data)
    vals = []
    for item, param in zip(table["item"], table["param"], strict=True):
        if item == "__latent__":
            vals.append(_latent_value(form, str(param)))
        else:
            vals.append(float(form["params"][item][param]))
    table["value"] = np.asarray(vals, dtype=float)
    table["free"] = False
    return fit(spec, data, table=table, quadrature=quad, se=None)


@cache
def _fit_form(name: str, side: str) -> FitResult:
    """Calibrate one form from scratch in flexpyirt with tight tolerance and Oakes SEs.

    The latent distribution is fixed at the reference calibration's (N(0, 1), or N(0.3, 1.5) for
    the V20e source form).
    """
    case = _case(name)
    form = case[side]
    meta = case["meta"]
    spec = _form_model(form)
    data = _form_data(form)
    quad = Quadrature.equispaced(int(meta["quadpts"]), bounds=tuple(meta["theta_lim"]))
    table = spec.parameter_table(data)
    is_latent = table["item"] == "__latent__"
    table.loc[is_latent, "value"] = [
        _latent_value(form, str(p)) for p in table.loc[is_latent, "param"]
    ]
    return fit(
        spec,
        data,
        table=table,
        method=BAEM(tol=1e-9, max_iter=20000, squarem=True),
        quadrature=quad,
        se="oakes",
    )


@cache
def _direct_problem(name: str) -> tuple[LinkingProblem, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Linking problem on mirt's common-item estimates with mirt's covariance blocks.

    Returns ``(prob, eta_s, eta_t, vcov_s, vcov_t)``; the free-parameter order is the reference's
    item-major ``common_vcov`` labels (natural scale, ``g`` not logit).
    """
    case = _case(name)
    meta = case["meta"]
    nodes = Quadrature.equispaced(int(meta["quadpts"]), bounds=tuple(meta["theta_lim"])).nodes()
    nodes_j = jnp.asarray(nodes, dtype=jnp.float64)

    def weights(form: dict[str, Any]) -> jax.Array:
        return jnp.exp(
            normal_log_weights(
                nodes_j,
                jnp.asarray(form["latent_mean"], dtype=jnp.float64),
                jnp.asarray(form["latent_var"], dtype=jnp.float64),
            )
        )

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
        tgt_nodes=nodes_j,
        tgt_weights=weights(case["target"]),
        src_nodes=nodes_j,
        src_weights=weights(case["source"]),
    )
    return prob, sides["source"][1], sides["target"][1], sides["source"][2], sides["target"][2]


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("side", ("target", "source"))
def test_form_calibration_matches_mirt(case_name: str, side: str) -> None:
    case = _case(case_name)
    form = case[side]
    res = _fit_form(case_name, side)
    assert res.converged
    ref_ll = float(form["loglik"])
    assert res.loglik == pytest.approx(ref_ll, rel=1e-6)
    items_df = res.items.set_index("item")
    for item, pvals in form["params"].items():
        for col, ref_val in pvals.items():
            if col.endswith("_se") or ref_val is None:
                continue
            assert items_df.loc[item, col] == pytest.approx(float(ref_val), abs=1e-4)


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("variant", VARIANTS)
def test_linking_on_mirt_estimates_matches_r_reference(case_name: str, variant: str) -> None:
    """Given identical mirt item estimates, A, B, and criterion match the R reference tightly."""
    case = _case(case_name)
    ref = case["linking"][variant]
    src = _fixed_at_mirt(case_name, "source")
    tgt = _fixed_at_mirt(case_name, "target")
    linked = link(
        src,
        tgt,
        items=case["common_items"],
        method=ref["method"],
        symmetric=bool(ref["symmetric"]),
    )
    assert linked.converged
    assert list(linked.common_items) == list(case["common_items"])
    assert pytest.approx(float(ref["A"]), abs=TOL_AB_DIRECT) == linked.A
    assert pytest.approx(float(ref["B"]), abs=TOL_AB_DIRECT) == linked.B
    assert linked.criterion == pytest.approx(
        float(ref["criterion"]), rel=TOL_CRIT_REL_DIRECT, abs=1e-12
    )


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("variant", VARIANTS)
def test_ogasawara_se_on_mirt_vcov_matches_r_reference(case_name: str, variant: str) -> None:
    """Delta-method vcov from mirt's own estimates and Oakes covariance blocks matches base R."""
    case = _case(case_name)
    ref = case["linking"][variant]
    prob, eta_s, eta_t, vcov_s, vcov_t = _direct_problem(case_name)
    method, symmetric = ref["method"], bool(ref["symmetric"])
    ab, _, _, _, conv = _solve(
        prob,
        eta_s,
        eta_t,
        jnp.array([float(ref["A"]), float(ref["B"])]),
        method=method,
        symmetric=symmetric,
        tol=1e-12,
    )
    assert bool(conv)
    np.testing.assert_allclose(np.asarray(ab), [ref["A"], ref["B"]], atol=TOL_AB_DIRECT)
    cov, valid = _vcov(ab, eta_s, eta_t, vcov_s, vcov_t, prob, method=method, symmetric=symmetric)
    assert bool(valid)
    cov_np = np.asarray(cov)
    np.testing.assert_allclose(
        cov_np, np.asarray(ref["vcov"], dtype=float), rtol=TOL_SE_REL_DIRECT, atol=0.0
    )
    assert np.sqrt(cov_np[0, 0]) == pytest.approx(float(ref["A_se"]), rel=TOL_SE_REL_DIRECT)
    assert np.sqrt(cov_np[1, 1]) == pytest.approx(float(ref["B_se"]), rel=TOL_SE_REL_DIRECT)


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("start", [(0.25, -3.0), (4.0, 3.0), (0.5, 2.5), (3.0, -2.0)])
def test_newton_converges_from_far_starts_on_fitted_estimates(
    case_name: str, variant: str, start: tuple[float, float]
) -> None:
    """From far-off (A, B) starts the solver still reaches the reference optimum (real data)."""
    ref = _case(case_name)["linking"][variant]
    prob, eta_s, eta_t, _, _ = _direct_problem(case_name)
    ab, _, gnorm, _, conv = _solve(
        prob,
        eta_s,
        eta_t,
        jnp.asarray(start, dtype=jnp.float64),
        method=ref["method"],
        symmetric=bool(ref["symmetric"]),
        tol=1e-10,
        max_iter=200,
    )
    assert bool(conv), f"not converged from {start}: max|grad| = {float(gnorm):.2e}"
    np.testing.assert_allclose(np.asarray(ab), [ref["A"], ref["B"]], atol=TOL_AB_DIRECT)


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("variant", VARIANTS)
def test_end_to_end_fit_and_link_matches_r_reference(case_name: str, variant: str) -> None:
    """Fit both forms in flexpyirt, link, and compare A, B, criterion, and Ogasawara SEs."""
    case = _case(case_name)
    ref = case["linking"][variant]
    src = _fit_form(case_name, "source")
    tgt = _fit_form(case_name, "target")
    linked = link(
        src,
        tgt,
        items=case["common_items"],
        method=ref["method"],
        symmetric=bool(ref["symmetric"]),
    )
    assert linked.converged
    assert pytest.approx(float(ref["A"]), abs=TOL_AB_FULL) == linked.A
    assert pytest.approx(float(ref["B"]), abs=TOL_AB_FULL) == linked.B
    assert linked.criterion == pytest.approx(
        float(ref["criterion"]), rel=TOL_CRIT_REL_FULL, abs=1e-10
    )
    assert linked.vcov is not None
    assert linked.vcov.shape == (2, 2)
    np.testing.assert_allclose(linked.vcov, linked.vcov.T, atol=1e-15)
    assert np.all(np.linalg.eigvalsh(linked.vcov) > 0.0)
    assert linked.A_se == pytest.approx(float(ref["A_se"]), rel=TOL_SE_REL)
    assert linked.B_se == pytest.approx(float(ref["B_se"]), rel=TOL_SE_REL)
    np.testing.assert_allclose(linked.vcov, np.asarray(ref["vcov"], dtype=float), rtol=TOL_SE_REL)
