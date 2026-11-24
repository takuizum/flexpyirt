"""V22: normal-mixture latent distributions vs R mirt (VALIDATION.md V22, DESIGN.md §16.7).

flexpyirt's ``NormalMixture`` uses mirt's class parameterization for the shape (raw class 1 is
N(0, 1); class c has ``MEAN_1 = rmean_c``, ``COV_11 = rvar_c``, ``PI = logit_c``) and standardizes
the mixture analytically, so on the grid theta_q it is mirt's
``multipleGroup(dentype = "mixture-k", invariance = <all items, free_means, free_var>)`` on the
image grid z_q = M + sqrt(V) theta_q. Each reference JSON (``validation/r/m5_mixture_reference.R``)
holds:

1. ``exact``: flexpyirt's discretized MMLE re-coded in R and Newton-polished from mirt's solution,
   with R Richardson-Hessian SEs and EAPs.
2. ``mirt``: built-in mirt refitted on the image grid (``TOL = 1e-10``, NR, no acceleration), its
   estimates and EAPs mapped to flexpyirt's scale, and its SEs on its own scale (mirt 1.46.1's
   Oakes SEs for mixtures are not observed-information SEs; see VALIDATION.md V22).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from flexpyirt import BAEM, GPCM, GRM, FitResult, Model, NormalMixture, Quadrature, TwoPL, fit
from flexpyirt._compile import LATENT_ITEM
from flexpyirt.items import ItemModel
from flexpyirt.latent import mixture_components, mixture_param_names
from tests._reference import check_reference_meta

# mirt reference comparison: nightly and release-PR CI only (CONTRIBUTING.md, "Tests").
pytestmark = pytest.mark.slow

REFERENCE_DIR = Path(__file__).parent / "reference"
CASES = ["v22_mixture_2pl_k2", "v22_mixture_grm_k2", "v22_mixture_gpcm_k3"]

TOL_LOGLIK_REL = 1e-6
TOL_PARAM_ABS = 1e-4
TOL_SE_REL = 1e-3
TOL_EAP_ABS = 1e-5


@dataclass(frozen=True)
class Case:
    name: str
    raw: dict[str, Any]

    @property
    def meta(self) -> dict[str, Any]:
        return dict(self.raw["meta"])

    @property
    def k(self) -> int:
        return int(self.meta["n_components"])

    @property
    def item_model(self) -> ItemModel:
        n_categories = int(self.meta["n_cat"])
        kind = self.meta["itemtype"]
        if kind == "2PL":
            return TwoPL()
        if kind == "graded":
            return GRM(n_categories=n_categories)
        return GPCM(n_categories=n_categories)

    @property
    def quad(self) -> Quadrature:
        lim = self.meta["theta_lim"]
        return Quadrature.equispaced(
            int(self.meta["quadpts"]), bounds=(float(lim[0]), float(lim[1]))
        )

    @property
    def d_names(self) -> tuple[str, ...]:
        n_categories = int(self.meta["n_cat"])
        return ("d",) if n_categories == 2 else tuple(f"d{k}" for k in range(1, n_categories))

    @property
    def spec(self) -> Model:
        return self.spec_from(NormalMixture(self.k))

    def spec_from(self, mixture: NormalMixture) -> Model:
        return Model.uniform(self.item_model, items=self.raw["items"]).with_latent(mixture)

    @property
    def mirt_labelled_start(self) -> NormalMixture:
        """Default-free start whose raw component 1 is mirt's class 1 (a different gauge)."""
        lat = self.raw["mirt"]["latent"]
        cs = range(2, self.k + 1)
        logits = np.array([0.0, *(float(lat[f"logit_{c}"]) for c in cs)])
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        means = [0.0, *(round(float(lat[f"rmean_{c}"]), 1) for c in cs)]
        variances = [1.0, *(round(float(lat[f"rvar_{c}"]), 1) or 0.1 for c in cs)]
        return NormalMixture(self.k, means=means, variances=variances, weights=list(weights))

    @property
    def data(self) -> pd.DataFrame:
        pats = np.asarray(self.raw["patterns"], dtype=int)
        return pd.DataFrame(np.repeat(pats, self.raw["counts"], axis=0), columns=self.raw["items"])

    @property
    def eap_patterns(self) -> pd.DataFrame:
        return pd.DataFrame(np.asarray(self.raw["eap_patterns"]), columns=self.raw["items"])

    def latent_vector(self, block: str) -> np.ndarray:
        lat = self.raw[block]["latent"]
        names = mixture_param_names(self.k)
        return np.array([0.0, 1.0, *(float(lat[p]) for p in names[2:])])


@cache
def load(name: str) -> Case:
    raw = dict(json.loads((REFERENCE_DIR / f"{name}.json").read_text()))
    check_reference_meta(raw["meta"], require_grid_check=True)
    return Case(name=name, raw=raw)


@cache
def fitted(name: str, squarem: bool, se: str | None, mirt_labels: bool = False) -> FitResult:
    """Fit from flexpyirt's default start, or from a start with mirt's component labelling."""
    case = load(name)
    return fit(
        case.spec_from(case.mirt_labelled_start) if mirt_labels else case.spec,
        case.data,
        method=BAEM(tol=1e-9, max_iter=20000, squarem=squarem),
        quadrature=case.quad,
        se=se,  # type: ignore[arg-type]
    )


def _fixed_eval(case: Case, block: str) -> FitResult:
    """FitResult with every parameter fixed at the reference block's values (no EM steps)."""
    tbl = case.spec.parameter_table(case.data)
    tbl["free"] = False
    tbl["free_id"] = -1
    for item_name, row in case.raw[block]["items"].items():
        for p, v in row.items():
            tbl.loc[(tbl["item"] == item_name) & (tbl["param"] == p), "value"] = float(v)
    for p, v in case.raw[block]["latent"].items():
        tbl.loc[(tbl["item"] == LATENT_ITEM) & (tbl["param"] == p), "value"] = float(v)
    return fit(case.spec, case.data, table=tbl, quadrature=case.quad, se=None)


def _sorted_components(params: np.ndarray) -> np.ndarray:
    """Standardized components ``(pi, mean, var)`` as rows sorted by mean (label invariant)."""
    pi, means, variances = (np.asarray(x) for x in mixture_components(jnp.asarray(params)))
    order = np.argsort(means)
    return np.stack([pi[order], means[order], variances[order]])


def _latent_estimates(res: FitResult) -> np.ndarray:
    lat = res.parameters[res.parameters["item"] == LATENT_ITEM].set_index("param")["estimate"]
    return lat.loc[list(mixture_param_names(res._model.mixture_components))].to_numpy(dtype=float)


# --------------------------------------------------------------------------------------------
# Reference self-consistency: the R discretization equals mirt's, and the two blocks agree
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", CASES)
def test_reference_blocks_agree(name: str) -> None:
    case = load(name)
    ex, mi = case.raw["exact"], case.raw["mirt"]
    assert mi["converged"] is True
    assert ex["max_grad"] < 2e-5
    # R's mirt-scale likelihood reproduces mirt's logLik; its standardized likelihood at mirt's
    # mapped estimates equals mirt's logLik (the grids correspond).
    assert mi["loglik_r_mirt_scale"] == pytest.approx(mi["loglik"], rel=1e-12)
    assert mi["loglik_std_at_mirt"] == pytest.approx(mi["loglik"], rel=1e-12)
    assert ex["loglik"] == pytest.approx(mi["loglik"], rel=1e-10)
    for item_name, row in ex["items"].items():
        for p, v in row.items():
            assert mi["items"][item_name][p] == pytest.approx(v, abs=1e-6)
    for p, v in ex["latent"].items():
        assert mi["latent"][p] == pytest.approx(v, abs=1e-6)


# --------------------------------------------------------------------------------------------
# Estimates, log-likelihood and EM monotonicity
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("squarem", [False, True])
def test_loglik_and_estimates_vs_exact_and_mirt(name: str, squarem: bool) -> None:
    case = load(name)
    res = fitted(name, squarem=squarem, se=None)
    assert res.converged
    ll_hist = res.history["loglik"].to_numpy()
    assert np.all(np.diff(ll_hist) >= -1e-9 * np.abs(ll_hist[1:]))

    for block in ("exact", "mirt"):
        ref = case.raw[block]
        assert abs(res.loglik - float(ref["loglik"])) <= TOL_LOGLIK_REL * abs(float(ref["loglik"]))
        items_df = res.items.set_index("item")
        for item_name, row in ref["items"].items():
            for p in ("a", *case.d_names):
                assert float(items_df.loc[item_name, p]) == pytest.approx(
                    float(row[p]), abs=TOL_PARAM_ABS
                )
        # Standardized components, compared sorted by mean (labels are arbitrary).
        np.testing.assert_allclose(
            _sorted_components(_latent_estimates(res)),
            _sorted_components(case.latent_vector(block)),
            rtol=0,
            atol=TOL_PARAM_ABS,
        )
    # flexpyirt's maximum is not below the exact R maximum.
    assert res.loglik >= float(case.raw["exact"]["loglik"]) - 1e-8 * abs(res.loglik)


@pytest.mark.parametrize("name", CASES)
def test_raw_shape_equals_mirt_class_parameters(name: str) -> None:
    """With mirt's labelling (a gauge choice), the raw shape is mirt's MEAN_1, COV_11 and PI."""
    case = load(name)
    res = fitted(name, squarem=True, se=None, mirt_labels=True)
    assert res.loglik == pytest.approx(fitted(name, squarem=True, se=None).loglik, rel=1e-11)
    np.testing.assert_allclose(
        _latent_estimates(res)[2:], case.latent_vector("mirt")[2:], rtol=0, atol=TOL_PARAM_ABS
    )


# --------------------------------------------------------------------------------------------
# Standard errors
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", CASES)
def test_ses_vs_r_hessian_and_oakes_equals_hessian(name: str) -> None:
    case = load(name)
    res_o = fitted(name, squarem=True, se="oakes", mirt_labels=True)
    res_h = fitted(name, squarem=True, se="hessian", mirt_labels=True)
    assert res_o.vcov is not None and res_h.vcov is not None
    np.testing.assert_allclose(
        res_o.vcov, res_h.vcov, rtol=1e-6, atol=1e-8 * np.abs(res_h.vcov).max()
    )
    ex = case.raw["exact"]
    for res in (res_o, res_h):
        items_df = res.items.set_index("item")
        for item_name, row in ex["items_se_hessian"].items():
            for p in ("a", *case.d_names):
                assert float(items_df.loc[item_name, f"{p}_se"]) == pytest.approx(
                    float(row[p]), rel=TOL_SE_REL
                )
        lat = res.latent.iloc[0]
        for p, v in ex["latent_se_hessian"].items():
            assert float(lat[f"{p}_se"]) == pytest.approx(float(v), rel=TOL_SE_REL)


@pytest.mark.parametrize("name", CASES)
def test_mirt_oakes_ses_for_mixtures_are_not_observed_information(name: str) -> None:
    """Document mirt 1.46.1's mixture Oakes SEs (VALIDATION.md V22): on mirt's own scale the
    latent SEs from the R Richardson Hessian of mirt's discretized likelihood equal flexpyirt's
    (the raw shape coordinates are shared), while mirt's Oakes SEs are several times smaller and
    close to its complete-data SEs (class membership treated as known)."""
    case = load(name)
    mi = case.raw["mirt"]
    n_lat = 3 * (case.k - 1)
    se_h = np.asarray(mi["se_hessian_mirt_scale"])[-n_lat:]
    se_o = np.asarray(mi["se_oakes_mirt_scale"])[-n_lat:]
    se_c = np.asarray(mi["se_complete_mirt_scale"])[-n_lat:]
    res = fitted(name, squarem=True, se="hessian", mirt_labels=True)
    lat = res.latent.iloc[0]
    ours = np.array([float(lat[f"{p}_se"]) for p in mixture_param_names(case.k)[2:]])
    np.testing.assert_allclose(ours, se_h, rtol=TOL_SE_REL)
    assert np.all(se_o < 0.6 * se_h)
    assert np.all(se_c <= se_o * 1.05)


# --------------------------------------------------------------------------------------------
# EAP
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("block", ["exact", "mirt"])
def test_eap_at_reference_parameters(name: str, block: str) -> None:
    case = load(name)
    fixed = _fixed_eval(case, block)
    ref_ll = float(case.raw[block]["loglik"])
    assert abs(fixed.loglik - ref_ll) <= 1e-9 * abs(ref_ll)
    sc = fixed.score(case.eap_patterns, method="EAP")
    eap = case.raw[block]["eap"]
    np.testing.assert_allclose(sc["theta"].to_numpy(), eap["theta"], rtol=0, atol=TOL_EAP_ABS)
    np.testing.assert_allclose(sc["se"].to_numpy(), eap["se"], rtol=0, atol=TOL_EAP_ABS)


@pytest.mark.parametrize("name", CASES)
def test_eap_at_flexpyirt_estimates(name: str) -> None:
    case = load(name)
    res = fitted(name, squarem=True, se=None)
    sc = res.score(case.eap_patterns, method="EAP")
    eap = case.raw["mirt"]["eap"]
    np.testing.assert_allclose(sc["theta"].to_numpy(), eap["theta"], rtol=0, atol=TOL_EAP_ABS)
    np.testing.assert_allclose(sc["se"].to_numpy(), eap["se"], rtol=0, atol=TOL_EAP_ABS)
