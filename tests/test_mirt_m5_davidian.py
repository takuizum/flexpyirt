"""V19: Davidian-curve latent distributions vs R mirt + dcurver (VALIDATION.md).

mirt 1.46.1's built-in ``dentype = "Davidian-k"`` applies Woods' linear-interpolation
``standardizeQuadrature`` to expected counts before running ``nlminb`` on unnormalized
``dcurver::ddc``, breaks EM monotonicity, disables ``SE = TRUE``, and forbids free mean/variance
(DESIGN.md §16.4, VALIDATION.md V19). Each reference JSON (from
``validation/r/m5_davidian_reference.R``) therefore contains:

1. ``exact``: exact standardized Davidian MMLE evaluated inside ``mirt`` via ``createGroup()``
   (calling ``dcurver::ddc``) with ``SE.type = "Oakes"`` and ``fscores(...,
   use_dentype_estimate = TRUE)``, plus exact R central-difference (Richardson) Hessian SEs.
2. ``mirt_builtin`` (for V19a and V19b): built-in ``mirt(..., dentype = "Davidian-k",
   optimizer = "NR", accelerate = "none", TOL = 1e-10)`` estimates, ``logLik``, ``Prior``, and
   ``fscores(..., use_dentype_estimate = TRUE)``, used to verify exact density / log-likelihood /
   EAP agreement at mirt's parameters and that flexpyirt's exact MMLE reaches a strictly higher
   log-likelihood.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from flexpyirt import (
    BAEM,
    GRM,
    DavidianCurve,
    FitResult,
    Model,
    Quadrature,
    TwoPL,
    fit,
)
from flexpyirt._compile import LATENT_ITEM
from flexpyirt.latent import davidian_log_weights
from tests._reference import check_reference_meta

# mirt reference comparison: nightly and release-PR CI only (CONTRIBUTING.md, "Tests").
pytestmark = pytest.mark.slow

REFERENCE_DIR = Path(__file__).parent / "reference"
CASES = ["v19_davidian_2pl_k3", "v19_davidian_grm_k2", "v19_davidian_fipc_k3"]
BUILTIN_CASES = ["v19_davidian_2pl_k3", "v19_davidian_grm_k2"]

TOL_LOGLIK_REL = 1e-6
TOL_PARAM_ABS = 1e-4
TOL_SE_REL = 1e-3
TOL_EAP_ABS = 1e-5


@dataclass(frozen=True)
class Case:
    name: str
    raw: dict[str, Any]
    spec: Model
    quad: Quadrature

    @property
    def degree(self) -> int:
        return int(self.raw["meta"]["degree"])

    @property
    def n_categories(self) -> int:
        return int(self.raw["meta"]["n_cat"])

    @property
    def d_names(self) -> tuple[str, ...]:
        return (
            ("d",)
            if self.n_categories == 2
            else tuple(f"d{k}" for k in range(1, self.n_categories))
        )

    @property
    def data(self) -> pd.DataFrame:
        pats = pd.DataFrame(
            np.asarray(self.raw["patterns"], dtype=float), columns=self.raw["items"]
        )
        return pats.loc[pats.index.repeat(self.raw["counts"])].reset_index(drop=True)

    @property
    def eap_patterns(self) -> pd.DataFrame:
        return pd.DataFrame(
            np.asarray(self.raw["eap_patterns"], dtype=float), columns=self.raw["items"]
        )


@cache
def load(name: str) -> Case:
    raw = dict(json.loads((REFERENCE_DIR / f"{name}.json").read_text()))
    check_reference_meta(raw["meta"], require_grid_check=True)
    meta = raw["meta"]
    lim = meta["theta_lim"]
    quad = Quadrature.equispaced(int(meta["quadpts"]), bounds=(float(lim[0]), float(lim[1])))
    degree = int(meta["degree"])
    n_categories = int(meta["n_cat"])
    item_model = TwoPL() if meta["itemtype"] == "2PL" else GRM(n_categories=n_categories)
    spec = Model.uniform(item_model, items=raw["items"])
    if "bank" in raw:
        spec = spec.fix_from(pd.DataFrame(raw["bank"]))
        spec = spec.with_latent(DavidianCurve.free(degree))
    else:
        spec = spec.with_latent(DavidianCurve(degree))
    return Case(name=name, raw=raw, spec=spec, quad=quad)


@cache
def fitted(name: str, squarem: bool, se: str) -> FitResult:
    case = load(name)
    return fit(
        case.spec,
        case.data,
        method=BAEM(tol=1e-8, max_iter=2000, squarem=squarem),
        quadrature=case.quad,
        se=se,  # type: ignore[arg-type]
    )


def _fixed_eval(
    case: Case,
    items_dict: dict[str, dict[str, float]],
    mean: float,
    var: float,
    phi_vals: list[float],
) -> FitResult:
    """Return a FitResult with every parameter fixed at the supplied values (no EM steps)."""
    tbl = case.spec.parameter_table(case.data)
    tbl["free"] = False
    tbl["free_id"] = -1
    for item_name, row in items_dict.items():
        for p, v in row.items():
            sel = (tbl["item"] == item_name) & (tbl["param"] == p)
            tbl.loc[sel, "value"] = float(v)
    tbl.loc[(tbl["item"] == LATENT_ITEM) & (tbl["param"] == "mean"), "value"] = float(mean)
    tbl.loc[(tbl["item"] == LATENT_ITEM) & (tbl["param"] == "var"), "value"] = float(var)
    for j, val in enumerate(phi_vals, start=1):
        tbl.loc[(tbl["item"] == LATENT_ITEM) & (tbl["param"] == f"phi_{j}"), "value"] = float(val)
    return fit(case.spec, case.data, table=tbl, quadrature=case.quad, se=None)


# --------------------------------------------------------------------------------------------
# Exact standardized Davidian MMLE agreement (mirt createGroup Oakes + R Richardson Hessian)
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("squarem", [False, True])
def test_exact_loglik_params_and_monotonicity(name: str, squarem: bool) -> None:
    case = load(name)
    res = fitted(name, squarem=squarem, se="oakes")
    ex = case.raw["exact"]
    assert res.converged

    # Strict EM / SQUAREM monotonicity.
    ll_hist = res.history["loglik"].to_numpy()
    assert np.all(np.diff(ll_hist) >= -1e-9 * np.abs(ll_hist[1:]))

    # Marginal log-likelihood agreement.
    ref_ll = float(ex["loglik"])
    assert abs(res.loglik - ref_ll) <= TOL_LOGLIK_REL * abs(ref_ll)

    # Free item parameter agreement.
    items_df = res.items.set_index("item")
    for item_name, ref_row in ex["items"].items():
        for p in ("a", *case.d_names):
            assert float(items_df.loc[item_name, p]) == pytest.approx(
                float(ref_row[p]), abs=TOL_PARAM_ABS
            )

    # Latent parameter agreement (mean, var, phi_1..phi_k).
    lat_df = res.latent
    ref_lat = ex["latent"]
    assert float(lat_df["mean"].iloc[0]) == pytest.approx(float(ref_lat["mean"]), abs=TOL_PARAM_ABS)
    assert float(lat_df["var"].iloc[0]) == pytest.approx(float(ref_lat["var"]), abs=TOL_PARAM_ABS)
    for j in range(1, case.degree + 1):
        assert float(lat_df[f"phi_{j}"].iloc[0]) == pytest.approx(
            float(ref_lat[f"phi_{j}"]), abs=TOL_PARAM_ABS
        )


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("squarem", [False, True])
def test_oakes_se_vs_mirt_and_hessian_se_vs_r_richardson(name: str, squarem: bool) -> None:
    case = load(name)
    res_o = fitted(name, squarem=squarem, se="oakes")
    res_h = fitted(name, squarem=squarem, se="hessian")
    ex = case.raw["exact"]

    # Internal Oakes == Hessian identity to ~1e-6 relative (typically ~1e-12).
    assert res_o.vcov is not None and res_h.vcov is not None
    np.testing.assert_allclose(
        res_o.vcov, res_h.vcov, rtol=1e-6, atol=1e-8 * np.abs(res_h.vcov).max()
    )

    # Oakes SEs vs mirt(customGroup, SE.type="Oakes").
    items_o = res_o.items.set_index("item")
    for item_name, ref_se in ex["items_se_oakes"].items():
        for p in ("a", *case.d_names):
            assert float(items_o.loc[item_name, f"{p}_se"]) == pytest.approx(
                float(ref_se[f"{p}_se"]), rel=TOL_SE_REL
            )
    lat_o = res_o.latent
    ref_lat = ex["latent"]
    for p in ("mean", "var", *(f"phi_{j}" for j in range(1, case.degree + 1))):
        ref_val = ref_lat[f"{p}_se"]
        if ref_val is not None:
            assert float(lat_o[f"{p}_se"].iloc[0]) == pytest.approx(float(ref_val), rel=TOL_SE_REL)

    # Hessian SEs vs R exact Richardson numerical Hessian.
    items_h = res_h.items.set_index("item")
    for item_name, ref_se in ex["items_se_hessian"].items():
        for p in ("a", *case.d_names):
            assert float(items_h.loc[item_name, f"{p}_se"]) == pytest.approx(
                float(ref_se[f"{p}_se"]), rel=TOL_SE_REL
            )
    lat_h = res_h.latent
    ref_lat_h = ex["latent_hessian"]
    for p in ("mean", "var", *(f"phi_{j}" for j in range(1, case.degree + 1))):
        ref_val = ref_lat_h[f"{p}_se"]
        if ref_val is not None:
            assert float(lat_h[f"{p}_se"].iloc[0]) == pytest.approx(float(ref_val), rel=TOL_SE_REL)


@pytest.mark.parametrize("name", CASES)
def test_eap_at_exact_reference_parameters(name: str) -> None:
    case = load(name)
    ex = case.raw["exact"]
    ref_lat = ex["latent"]
    phi_vals = [float(ref_lat[f"phi_{j}"]) for j in range(1, case.degree + 1)]
    fixed_res = _fixed_eval(
        case,
        ex["items"],
        mean=float(ref_lat["mean"]),
        var=float(ref_lat["var"]),
        phi_vals=phi_vals,
    )
    assert abs(fixed_res.loglik - float(ex["loglik"])) <= 1e-12 * abs(float(ex["loglik"]))
    sc = fixed_res.score(case.eap_patterns, method="EAP")
    np.testing.assert_allclose(sc["theta"].to_numpy(), ex["eap"]["theta"], rtol=0, atol=TOL_EAP_ABS)
    np.testing.assert_allclose(sc["se"].to_numpy(), ex["eap"]["se"], rtol=0, atol=TOL_EAP_ABS)


# --------------------------------------------------------------------------------------------
# Built-in mirt(..., dentype="Davidian-k") agreement at mirt's parameters & superiority of MMLE
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", BUILTIN_CASES)
def test_builtin_mirt_davidian_density_loglik_eap_and_mmle_gain(name: str) -> None:
    """At mirt's built-in Davidian-k parameters (with mean=mu_z, var=var_z), flexpyirt reproduces
    mirt's Prior, logLik, and EAP to ~1e-14; a full exact MMLE fit reaches strictly higher loglik.
    """
    case = load(name)
    mb = case.raw["mirt_builtin"]
    phi_cos = [float(x) for x in mb["phi_cos"]]
    mu_z = float(mb["mu_z"])
    var_z = float(mb["var_z"])

    # Exact grid weight match with mirt's dcurver::ddc / sum.
    import jax.numpy as jnp

    params_vec = jnp.asarray([mu_z, var_z, *phi_cos], dtype=jnp.float64)
    w_ours = np.exp(
        np.asarray(
            davidian_log_weights(jnp.asarray(case.quad.nodes(), dtype=jnp.float64), params_vec)
        )
    )
    np.testing.assert_allclose(w_ours, np.asarray(mb["prior"], dtype=float), rtol=0, atol=1e-12)

    # Exact logLik and EAP match at mirt's parameters.
    fixed_res = _fixed_eval(case, mb["items"], mean=mu_z, var=var_z, phi_vals=phi_cos)
    ref_ll = float(mb["loglik"])
    assert abs(fixed_res.loglik - ref_ll) <= 1e-12 * abs(ref_ll)
    sc = fixed_res.score(case.eap_patterns, method="EAP")
    np.testing.assert_allclose(sc["theta"].to_numpy(), mb["eap"]["theta"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(sc["se"].to_numpy(), mb["eap"]["se"], rtol=0, atol=1e-12)

    # Full exact MMLE fit achieves strictly higher marginal log-likelihood than mirt's
    # standardizeQuadrature algorithm.
    full_res = fitted(name, squarem=True, se="oakes")
    assert full_res.loglik > ref_ll + 10.0
