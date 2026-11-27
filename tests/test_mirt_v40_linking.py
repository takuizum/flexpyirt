"""Scale linking for 4PL, nominal and 2PL + nominal common items against base R (VALIDATION V40).

Reference values come from ``validation/r/m5_linking_extensions_reference.R`` ->
``tests/reference/v40_linking_unidim.json``. As in V20 (``tests/test_mirt_m5_linking.py``), R
``mirt`` does not implement scale linking: each form is calibrated with mirt (``optimizer="NR"``,
``TOL=1e-10``, 61 nodes on [-6, 6], Oakes SEs, latent N(0, 1)), and the Haebara (1980) and
Stocking–Lord (1983) criteria (non-symmetric and Kim & Lee, 2006 symmetric variants) plus
Ogasawara (2001) delta-method covariances are computed in base R from the formulas using
``mirt::probtrace()`` and ``mirt::expected.item()``.

The tests on mirt's own item estimates and covariance blocks need no calibration and run in the
fast tier; the checks through :func:`flexpyirt.link` (all parameters fixed at mirt's estimates,
and end-to-end calibration + linking) are slow.
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

from flexpyirt import BAEM, FitResult, FourPL, Model, Nominal, Quadrature, TwoPL, fit, link
from flexpyirt._linking import LinkingFamily, LinkingProblem, linking_vcov, solve_linking
from flexpyirt.items import ItemKernel, ItemModel
from flexpyirt.latent import normal_log_weights
from tests._reference import check_reference_meta

REFERENCE_DIR = Path(__file__).parent / "reference"
REFERENCE_FILE = "v40_linking_unidim.json"

CASES = ["v40a_4pl", "v40b_nominal", "v40c_mixed"]
VARIANTS = ["stocking_lord", "stocking_lord_sym", "haebara", "haebara_sym"]

# Linking on identical mirt item estimates: limited by the R reference's own optimization
# accuracy (central-FD gradient polish, max |grad| <= 1.0e-11).
TOL_AB_DIRECT = 1e-9
TOL_CRIT_REL_DIRECT = 1e-9
# Delta-method covariance on identical inputs: limited by the R reference's nested central finite
# differences (step 1e-5).
TOL_SE_REL_DIRECT = 1e-5
# End-to-end: both forms calibrated in flexpyirt.
TOL_AB_FULL = 1e-6
TOL_CRIT_REL_FULL = 1e-5
TOL_SE_REL = 1e-4

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


def _ref_ab(ref: dict[str, Any]) -> list[float]:
    """Reference ``[A, B]`` (stored as a 1 x 1 matrix and a length-1 vector)."""
    return [float(np.ravel(ref["A"])[0]), float(np.ravel(ref["beta"])[0])]


def _item_model(itemtype: str, n_categories: int) -> ItemModel:
    if itemtype == "2PL":
        return TwoPL()
    if itemtype == "4PL":
        return FourPL()
    if itemtype == "nominal":
        return Nominal(n_categories=n_categories)
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
            vals.append(float(form["latent"][str(param)]))
        else:
            vals.append(float(form["params"][item][param]))
    table["value"] = np.asarray(vals, dtype=float)
    table["free"] = False
    return fit(spec, data, table=table, quadrature=quad, se=None)


@cache
def _fit_form(name: str, side: str) -> FitResult:
    """Calibrate one form from scratch in flexpyirt with tight tolerance and Oakes SEs."""
    case = _case(name)
    form = case[side]
    meta = case["meta"]
    quad = Quadrature.equispaced(int(meta["quadpts"]), bounds=tuple(meta["theta_lim"]))
    return fit(
        _form_model(form),
        _form_data(form),
        method=BAEM(tol=1e-9, max_iter=50000, squarem=True),
        quadrature=quad,
        se="oakes",
    )


@cache
def _direct_problem(name: str) -> tuple[LinkingProblem, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Linking problem on mirt's common-item estimates with mirt's covariance blocks.

    Returns ``(prob, eta_s, eta_t, vcov_s, vcov_t)``; the free-parameter order is the reference's
    item-major ``common_vcov`` labels (natural scale: ``g`` and ``u`` are not logits).
    """
    case = _case(name)
    meta = case["meta"]
    nodes = Quadrature.equispaced(int(meta["quadpts"]), bounds=tuple(meta["theta_lim"])).nodes()
    nodes_j = jnp.asarray(nodes, dtype=jnp.float64)

    def weights(form: dict[str, Any]) -> jax.Array:
        return jnp.exp(
            normal_log_weights(
                nodes_j,
                jnp.asarray(form["latent"]["mean"], dtype=jnp.float64),
                jnp.asarray(form["latent"]["var"], dtype=jnp.float64),
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


def test_reference_covers_every_free_common_parameter() -> None:
    """The reference covariance blocks hold all free common parameters, including ``u``/``ak``."""
    params_4pl = {p for _, p in _case("v40a_4pl")["common_vcov"]["source"]["labels"]}
    assert params_4pl == {"a", "d", "g", "u"}
    params_nominal = {p for _, p in _case("v40b_nominal")["common_vcov"]["source"]["labels"]}
    assert params_nominal == {"a", "ak1", "ak2", "d1", "d2", "d3"}
    # Some estimated scoring coefficients are negative (unordered categories: not GPCM-like).
    form = _case("v40b_nominal")["target"]
    assert min(float(form["params"][i]["ak1"]) for i in form["items"]) < 0.0


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("start", [(1.0, 0.0), (0.4, -2.0), (3.0, 2.5)])
def test_constants_and_criterion_on_mirt_estimates_match_r_reference(
    case_name: str, variant: str, start: tuple[float, float]
) -> None:
    """On identical mirt item estimates, (A, B) and the criterion match base R from any start."""
    ref = _case(case_name)["linking"][variant]
    prob, eta_s, eta_t, _, _ = _direct_problem(case_name)
    ab, crit, gnorm, _, conv = _solve(
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
    np.testing.assert_allclose(np.asarray(ab), _ref_ab(ref), atol=TOL_AB_DIRECT)
    assert float(crit) == pytest.approx(float(ref["criterion"]), rel=TOL_CRIT_REL_DIRECT)


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("variant", VARIANTS)
def test_ogasawara_vcov_on_mirt_vcov_matches_r_reference(case_name: str, variant: str) -> None:
    """Delta-method vcov from mirt's own estimates and Oakes covariance blocks matches base R."""
    ref = _case(case_name)["linking"][variant]
    prob, eta_s, eta_t, vcov_s, vcov_t = _direct_problem(case_name)
    method, symmetric = ref["method"], bool(ref["symmetric"])
    ab, _, _, _, conv = _solve(
        prob,
        eta_s,
        eta_t,
        jnp.asarray(_ref_ab(ref)),
        method=method,
        symmetric=symmetric,
        tol=1e-12,
    )
    assert bool(conv)
    cov, valid = _vcov(ab, eta_s, eta_t, vcov_s, vcov_t, prob, method=method, symmetric=symmetric)
    assert bool(valid)
    np.testing.assert_allclose(
        np.asarray(cov), np.asarray(ref["vcov"], dtype=float), rtol=TOL_SE_REL_DIRECT, atol=0.0
    )


@pytest.mark.parametrize("case_name", CASES)
def test_estimates_recover_the_simulated_transformation(case_name: str) -> None:
    """The reference optimum is within 4 delta-method SEs of the simulated (A, B) = (1.2, 0.5)."""
    case = _case(case_name)
    truth = np.concatenate(
        [np.ravel(np.asarray(case["truth"][key], dtype=float)) for key in ("A", "beta")]
    )
    for variant in VARIANTS:
        ref = case["linking"][variant]
        est = np.asarray(_ref_ab(ref))
        se = np.sqrt(np.diag(np.asarray(ref["vcov"], dtype=float)))
        assert np.all(np.abs(est - truth) < 4.0 * se), (variant, est, se)


@pytest.mark.slow
@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("variant", VARIANTS)
def test_link_on_mirt_estimates_matches_r_reference(case_name: str, variant: str) -> None:
    """``link()`` on FitResults fixed at mirt's estimates (default Mean/Mean start)."""
    case = _case(case_name)
    ref = case["linking"][variant]
    linked = link(
        _fixed_at_mirt(case_name, "source"),
        _fixed_at_mirt(case_name, "target"),
        items=case["common_items"],
        method=ref["method"],
        symmetric=bool(ref["symmetric"]),
    )
    assert linked.converged
    assert pytest.approx(_ref_ab(ref)[0], abs=TOL_AB_DIRECT) == linked.A
    assert pytest.approx(_ref_ab(ref)[1], abs=TOL_AB_DIRECT) == linked.B
    assert linked.criterion == pytest.approx(float(ref["criterion"]), rel=TOL_CRIT_REL_DIRECT)
    # Transformed common items reproduce the source curves at (theta - B) / A.
    src = case["source"]
    spec = _form_model(src)
    theta = jnp.linspace(-4.0, 4.0, 33)[:, None]
    items = linked.items.set_index("item")
    for name in case["common_items"]:
        kernel = spec.item_kernel(dict(spec.items)[name])
        xi_src = jnp.asarray([float(src["params"][name][p]) for p in kernel.param_names])
        xi_star = jnp.asarray([float(items.loc[name, p]) for p in kernel.param_names])
        np.testing.assert_allclose(
            np.asarray(kernel.log_prob(linked.A * theta + linked.B, xi_star)),
            np.asarray(kernel.log_prob(theta, xi_src)),
            atol=1e-12,
        )


@pytest.mark.slow
@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("side", ("target", "source"))
def test_form_calibration_matches_mirt(case_name: str, side: str) -> None:
    form = _case(case_name)[side]
    res = _fit_form(case_name, side)
    assert res.converged
    assert res.loglik == pytest.approx(float(form["loglik"]), rel=1e-6)
    items_df = res.items.set_index("item")
    for item, pvals in form["params"].items():
        for col, ref_val in pvals.items():
            assert items_df.loc[item, col] == pytest.approx(float(ref_val), abs=1e-4)


@pytest.mark.slow
@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("variant", VARIANTS)
def test_end_to_end_fit_and_link_matches_r_reference(case_name: str, variant: str) -> None:
    """Fit both forms in flexpyirt, link, and compare A, B, criterion, and Ogasawara SEs."""
    case = _case(case_name)
    ref = case["linking"][variant]
    linked = link(
        _fit_form(case_name, "source"),
        _fit_form(case_name, "target"),
        items=case["common_items"],
        method=ref["method"],
        symmetric=bool(ref["symmetric"]),
    )
    assert linked.converged
    assert pytest.approx(_ref_ab(ref)[0], abs=TOL_AB_FULL) == linked.A
    assert pytest.approx(_ref_ab(ref)[1], abs=TOL_AB_FULL) == linked.B
    assert linked.criterion == pytest.approx(float(ref["criterion"]), rel=TOL_CRIT_REL_FULL)
    assert linked.vcov is not None
    np.testing.assert_allclose(linked.vcov, np.asarray(ref["vcov"], dtype=float), rtol=TOL_SE_REL)
