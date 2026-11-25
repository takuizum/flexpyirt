"""Unit tests for differentiable IRT scale linking (DESIGN.md §16.5)."""

from __future__ import annotations

import math
import subprocess
import sys

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
    EmpiricalHistogram,
    FitResult,
    FourPL,
    LinkResult,
    Model,
    Nominal,
    Normal,
    Quadrature,
    Rasch,
    ThreePL,
    TwoPL,
    fit,
    link,
)
from flexpyirt._linking import (
    LinkingFamily,
    LinkingProblem,
    LinkMethod,
    linking_criterion,
    solve_linking,
    transform_kernel_params,
)
from flexpyirt.errors import ConvergenceWarning, SpecificationError
from flexpyirt.items import (
    DichotomousKernel,
    FourPLKernel,
    GPCMKernel,
    GradedKernel,
    ItemModel,
    NominalKernel,
)

METHODS: tuple[LinkMethod, ...] = ("stocking_lord", "haebara")


def _simulate(
    model: Model,
    params: dict[str, np.ndarray],
    n_obs: int,
    seed: int,
    *,
    mean: float = 0.0,
    sd: float = 1.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    theta = rng.normal(mean, sd, n_obs)
    th_j = jnp.asarray(theta[:, None], dtype=jnp.float64)
    cols: dict[str, np.ndarray] = {}
    for name, item_model in model.items:
        kernel = model.item_kernel(item_model)
        probs = np.asarray(jnp.exp(kernel.log_prob(th_j, jnp.asarray(params[name]))))
        cdf = np.cumsum(probs, axis=1)
        u = rng.random(n_obs)[:, None]
        cols[name] = np.clip((u > cdf).sum(axis=1), 0, kernel.n_categories - 1).astype(int)
    return pd.DataFrame(cols)


def _make_fixed_fit(
    model: Model,
    params: dict[str, np.ndarray],
    *,
    latent_mean: float = 0.0,
    latent_var: float = 1.0,
    quadrature: Quadrature | None = None,
    seed: int = 1,
) -> FitResult:
    """Create a FitResult with exact fixed item parameters (for noiseless linking checks)."""
    df = _simulate(model, params, n_obs=120, seed=seed, mean=latent_mean, sd=math.sqrt(latent_var))
    # Ensure every category appears at least once so parameter_table succeeds cleanly.
    for name, item_model in model.items:
        k = item_model.kernel.n_categories
        for cat in range(k):
            if cat not in df[name].values:
                df.loc[cat, name] = cat
    table = model.parameter_table(df)
    vals = []
    for item, param in zip(table["item"], table["param"], strict=True):
        if item == "__latent__":
            vals.append(latent_mean if param == "mean" else latent_var)
        else:
            pnames = model.item_kernel(dict(model.items)[item]).param_names
            vals.append(float(params[item][pnames.index(param)]))
    table["value"] = np.asarray(vals, dtype=float)
    table["free"] = False
    return fit(model, df, table=table, quadrature=quadrature, se=None)


# ---- Parameter transform and curve invariance across all 7 unidimensional models -----------------


@pytest.mark.parametrize(
    ("item_model", "params"),
    [
        (TwoPL(), np.array([1.4, -0.5])),
        (Rasch(), np.array([1.0, 0.7])),
        (ThreePL(), np.array([1.6, -0.3, 0.2, 1.0])),
        (FourPL(), np.array([1.5, 0.4, 0.15, 0.88])),
        (GRM(n_categories=4), np.array([1.3, 1.4, 0.1, -1.2])),
        (GPCM(n_categories=4), np.array([1.2, 0.8, 0.2, -0.9])),
        (PCM(n_categories=3), np.array([1.0, 0.5, -0.4])),
        (Nominal(n_categories=4), np.array([1.1, 2.3, -0.4, 0.6, 0.9, -0.7])),
        (Nominal(n_categories=2), np.array([0.8, 0.3])),
    ],
)
@pytest.mark.parametrize(("a_const", "b_const"), [(1.35, 0.6), (0.75, -1.1), (2.2, 2.5)])
def test_transform_kernel_params_preserves_icc_and_irt_thresholds(
    item_model: ItemModel, params: np.ndarray, a_const: float, b_const: float
) -> None:
    kernel = item_model.kernel
    theta_src = jnp.linspace(-5.0, 5.0, 51)
    theta_tgt = a_const * theta_src + b_const
    p_orig = jnp.exp(kernel.log_prob(theta_src[:, None], jnp.asarray(params)))
    xi_star = transform_kernel_params(
        kernel, jnp.asarray(params), jnp.asarray(a_const), jnp.asarray(b_const)
    )
    p_star = jnp.exp(kernel.log_prob(theta_tgt[:, None], xi_star))
    np.testing.assert_allclose(np.asarray(p_star), np.asarray(p_orig), rtol=0, atol=1e-14)

    # Display IRT thresholds transform linearly: b* = A * b + B.
    b_orig = np.asarray(kernel.irt_thresholds(jnp.asarray(params)))
    b_star = np.asarray(kernel.irt_thresholds(xi_star))
    np.testing.assert_allclose(b_star, a_const * b_orig + b_const, rtol=1e-13, atol=1e-13)


def test_transform_kernel_params_preserves_asymptotes_and_grm_order() -> None:
    xi_4pl = jnp.array([1.5, -0.4, 0.18, 0.91])
    out_4pl = transform_kernel_params(FourPLKernel(), xi_4pl, jnp.array(1.4), jnp.array(-0.7))
    assert float(out_4pl[2]) == pytest.approx(0.18, abs=1e-15)
    assert float(out_4pl[3]) == pytest.approx(0.91, abs=1e-15)

    xi_grm = jnp.array([1.2, 1.5, 0.2, -1.1])
    out_grm = transform_kernel_params(GradedKernel(4), xi_grm, jnp.array(1.6), jnp.array(0.9))
    assert np.all(np.diff(np.asarray(out_grm[1:])) < 0)

    # GPCM d_k shifts by -k * a * B / A (k = 1..K-1).
    xi_gpcm = jnp.array([1.2, 0.5, -0.2, -1.0])
    out_gpcm = transform_kernel_params(GPCMKernel(4), xi_gpcm, jnp.array(1.5), jnp.array(0.6))
    shift = 1.2 * 0.6 / 1.5
    np.testing.assert_allclose(
        np.asarray(out_gpcm),
        np.array([1.2 / 1.5, 0.5 - shift, -0.2 - 2 * shift, -1.0 - 3 * shift]),
        atol=1e-15,
    )


def test_transform_kernel_params_nominal_keeps_scoring_coefficients() -> None:
    # Nominal: a* = a / A, ak* = ak, d_k* = d_k - ak_k * a * B / A with ak_{K-1} = K - 1.
    xi = jnp.array([1.1, 2.3, -0.4, 0.6, 0.9, -0.7])
    out = transform_kernel_params(NominalKernel(4), xi, jnp.array(1.5), jnp.array(0.6))
    shift = 1.1 * 0.6 / 1.5
    np.testing.assert_allclose(
        np.asarray(out),
        np.array([1.1 / 1.5, 2.3, -0.4, 0.6 - 2.3 * shift, 0.9 + 0.4 * shift, -0.7 - 3.0 * shift]),
        atol=1e-15,
    )


# ---- Autodiff gradient finite-difference check ---------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("symmetric", [False, True])
def test_criterion_autodiff_gradient_matches_finite_differences(
    method: LinkMethod, symmetric: bool
) -> None:
    nodes = jnp.linspace(-4.0, 4.0, 21)
    weights = jnp.exp(-0.5 * nodes**2)
    weights = weights / jnp.sum(weights)
    fam1 = LinkingFamily(
        kernel=DichotomousKernel(),
        src_params=jnp.array([[1.3, -0.4], [0.9, 0.6]]),
        tgt_params=jnp.array([[1.1, 0.2], [0.8, 0.9]]),
        src_free_index=jnp.array([[0, 1], [2, 3]], dtype=jnp.int32),
        tgt_free_index=jnp.array([[0, 1], [2, 3]], dtype=jnp.int32),
    )
    fam2 = LinkingFamily(
        kernel=GPCMKernel(3),
        src_params=jnp.array([[1.2, 0.6, -0.3]]),
        tgt_params=jnp.array([[1.0, 0.9, 0.2]]),
        src_free_index=jnp.array([[4, 5, 6]], dtype=jnp.int32),
        tgt_free_index=jnp.array([[4, 5, 6]], dtype=jnp.int32),
    )
    prob = LinkingProblem(
        families=(fam1, fam2),
        tgt_nodes=nodes,
        tgt_weights=weights,
        src_nodes=nodes,
        src_weights=weights,
    )
    eta_s = jnp.array([1.3, -0.4, 0.9, 0.6, 1.2, 0.6, -0.3])
    eta_t = jnp.array([1.1, 0.2, 0.8, 0.9, 1.0, 0.9, 0.2])
    ab = jnp.array([1.15, 0.35])

    def fn(x: jax.Array) -> jax.Array:
        return linking_criterion(x, eta_s, eta_t, prob, method=method, symmetric=symmetric)

    grad_ad = np.asarray(jax.grad(fn)(ab))
    assert np.all(np.isfinite(grad_ad))
    eps = 1e-6
    grad_fd = np.zeros(2)
    for i in range(2):
        e = jnp.zeros(2).at[i].set(eps)
        grad_fd[i] = float((fn(ab + e) - fn(ab - e)) / (2.0 * eps))
    np.testing.assert_allclose(grad_ad, grad_fd, rtol=1e-7, atol=1e-9)

    # Non-positive A returns NaN without polluting valid A > 0 derivatives.
    assert math.isnan(float(fn(jnp.array([0.0, 0.0]))))
    assert math.isnan(float(fn(jnp.array([-0.5, 0.0]))))


# ---- Identity and exact transformation recovery across all 7 models + mixed formats --------------


@pytest.mark.slow
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("symmetric", [False, True])
def test_identity_source_equals_target_mixed_format(method: LinkMethod, symmetric: bool) -> None:
    model = Model.from_items(
        {
            "i2pl": TwoPL(),
            "irasch": Rasch(),
            "i3pl": ThreePL(),
            "i4pl": FourPL(),
            "igrm": GRM(n_categories=4),
            "igpcm": GPCM(n_categories=4),
            "ipcm": PCM(n_categories=3),
        }
    )
    params = {
        "i2pl": np.array([1.3, -0.5]),
        "irasch": np.array([1.0, 0.4]),
        "i3pl": np.array([1.5, -0.2, 0.2, 1.0]),
        "i4pl": np.array([1.4, 0.3, 0.15, 0.9]),
        "igrm": np.array([1.2, 1.5, 0.1, -1.3]),
        "igpcm": np.array([1.1, 0.7, 0.1, -0.8]),
        "ipcm": np.array([1.0, 0.6, -0.5]),
    }
    res = _make_fixed_fit(model, params)
    linked = link(res, res, method=method, symmetric=symmetric)
    assert isinstance(linked, LinkResult)
    assert linked.converged
    assert pytest.approx(1.0, abs=1e-12) == linked.A
    assert pytest.approx(0.0, abs=1e-12) == linked.B
    assert linked.criterion == pytest.approx(0.0, abs=1e-24)
    assert linked.n_iter == 0
    assert linked.vcov is None  # all parameters fixed
    assert math.isnan(linked.A_se) and math.isnan(linked.B_se)


@pytest.mark.slow
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("symmetric", [False, True])
def test_exact_recovery_of_known_transformation_all_models(
    method: LinkMethod, symmetric: bool
) -> None:
    """When source items are exact inverse transforms of target items, recover (A, B) exactly."""
    a_true, b_true = 1.35, -0.65
    common_spec: dict[str, ItemModel] = {
        "c_2pl": TwoPL(),
        "c_3pl": ThreePL(),
        "c_4pl": FourPL(),
        "c_grm": GRM(n_categories=4),
        "c_gpcm": GPCM(n_categories=4),
        "c_nom": Nominal(n_categories=4),
    }
    tgt_model = Model.from_items({**common_spec, "t_only": TwoPL()})
    src_model = Model.from_items({**common_spec, "s_only": GRM(n_categories=3)})

    tgt_params = {
        "c_2pl": np.array([1.2, -0.4]),
        "c_3pl": np.array([1.5, 0.3, 0.2, 1.0]),
        "c_4pl": np.array([1.3, -0.2, 0.12, 0.92]),
        "c_grm": np.array([1.1, 1.4, 0.2, -1.1]),
        "c_gpcm": np.array([1.0, 0.8, 0.1, -0.9]),
        "c_nom": np.array([0.9, 2.2, 0.6, 0.7, 0.4, -0.8]),
        "t_only": np.array([0.9, 0.5]),
    }
    # Target -> Source is the inverse transform (1/A, -B/A).
    inv_a = jnp.asarray(1.0 / a_true)
    inv_b = jnp.asarray(-b_true / a_true)
    src_params = {
        name: np.asarray(
            transform_kernel_params(
                tgt_model.item_kernel(item_model), jnp.asarray(tgt_params[name]), inv_a, inv_b
            )
        )
        for name, item_model in common_spec.items()
    }
    src_params["s_only"] = np.array([1.4, 0.9, -0.6])

    target = _make_fixed_fit(tgt_model, tgt_params, seed=10)
    source = _make_fixed_fit(src_model, src_params, seed=11)

    linked = link(source, target, method=method, symmetric=symmetric)
    assert linked.converged
    assert pytest.approx(a_true, abs=1e-11) == linked.A
    assert pytest.approx(b_true, abs=1e-11) == linked.B
    assert linked.criterion == pytest.approx(0.0, abs=1e-22)
    assert linked.common_items == tuple(common_spec.keys())
    assert list(linked.items["item"]) == [*common_spec.keys(), "s_only"]
    assert list(linked.items["common"]) == [True, True, True, True, True, True, False]

    # Common items transformed onto the target scale reproduce target parameters identically.
    items_idx = linked.items.set_index("item")
    for name, item_model in common_spec.items():
        for p, val in zip(item_model.kernel.param_names, tgt_params[name], strict=True):
            assert items_idx.loc[name, p] == pytest.approx(float(val), abs=1e-11)

    # Transformed item table works directly with Model.fix_from() in both parameterizations.
    # (The nominal model has no IRT parameterization, so it is left out of the "irt" check.)
    fipc_si = Model.from_items(common_spec).fix_from(
        linked.items, parameterization="slope_intercept"
    )
    assert len(fipc_si.fixed) == sum(len(m.kernel.param_names) for m in common_spec.values())
    common_spec = {k: v for k, v in common_spec.items() if k != "c_nom"}
    fipc_si = Model.from_items(common_spec).fix_from(
        linked.items, parameterization="slope_intercept"
    )
    fipc_irt = Model.from_items(common_spec).fix_from(linked.items, parameterization="irt")
    assert len(fipc_si.fixed) == len(fipc_irt.fixed)


@pytest.mark.slow
def test_newton_converges_from_perturbed_starting_point() -> None:
    """Verify solve_linking converges from a far-off initial (A, B), not just Mean/Mean."""
    nodes = jnp.linspace(-5.0, 5.0, 41)
    weights = jnp.exp(-0.5 * nodes**2)
    weights = weights / jnp.sum(weights)
    a_true, b_true = 1.4, 0.8
    tgt_p = jnp.array([[1.2, -0.6], [0.9, 0.4], [1.5, 1.0], [1.1, -1.1]])
    src_p = jax.vmap(
        lambda p: transform_kernel_params(
            DichotomousKernel(), p, jnp.asarray(1.0 / a_true), jnp.asarray(-b_true / a_true)
        )
    )(tgt_p)
    fam = LinkingFamily(
        kernel=DichotomousKernel(),
        src_params=src_p,
        tgt_params=tgt_p,
        src_free_index=jnp.full(src_p.shape, -1, dtype=jnp.int32),
        tgt_free_index=jnp.full(tgt_p.shape, -1, dtype=jnp.int32),
    )
    prob = LinkingProblem(
        families=(fam,), tgt_nodes=nodes, tgt_weights=weights, src_nodes=nodes, src_weights=weights
    )
    empty = jnp.zeros((0,), dtype=jnp.float64)
    for method in METHODS:
        for symmetric in (False, True):
            ab_hat, crit, gnorm, n_iter, conv = solve_linking(
                prob,
                empty,
                empty,
                jnp.array([0.4, -1.8]),
                method=method,
                symmetric=symmetric,
                tol=1e-11,
                max_iter=200,
            )
            assert bool(conv)
            assert int(n_iter) > 0
            assert float(gnorm) <= 1e-11
            assert float(crit) < 1e-20
            np.testing.assert_allclose(np.asarray(ab_hat), [a_true, b_true], atol=1e-10)


# ---- Symmetric reversibility, item mapping, custom nodes/weights, multigroup, EH ----------------


@pytest.fixture(scope="module")
def fitted_2pl_pair() -> tuple[FitResult, FitResult, pd.DataFrame]:
    """Two separately calibrated 2PL forms with 6 common items and realistic group shift."""
    common = [f"c{j:02d}" for j in range(1, 7)]
    tgt_items = [*common, "t01", "t02", "t03", "t04"]
    src_items = [*common, "s01", "s02", "s03", "s04"]
    rng = np.random.default_rng(2026)
    true_all = {
        name: np.array([float(rng.lognormal(0.1, 0.25)), float(rng.normal(0.0, 0.9))])
        for name in dict.fromkeys([*tgt_items, *src_items])
    }
    tgt_model = Model.uniform(TwoPL(), items=tgt_items)
    src_model = Model.uniform(TwoPL(), items=src_items)
    df_tgt = _simulate(tgt_model, true_all, n_obs=800, seed=101, mean=0.0, sd=1.0)
    df_src = _simulate(src_model, true_all, n_obs=800, seed=102, mean=0.4, sd=1.25)
    fast = BAEM(tol=1e-8, squarem=True)
    res_tgt = fit(tgt_model, df_tgt, method=fast, se="oakes")
    res_src = fit(src_model, df_src, method=fast, se="oakes")
    return res_src, res_tgt, df_src


@pytest.mark.slow
@pytest.mark.parametrize("method", METHODS)
def test_symmetric_linking_is_exactly_reversible(
    fitted_2pl_pair: tuple[FitResult, FitResult, pd.DataFrame], method: LinkMethod
) -> None:
    res_src, res_tgt, _ = fitted_2pl_pair
    fwd = link(res_src, res_tgt, method=method, symmetric=True)
    rev = link(res_tgt, res_src, method=method, symmetric=True)
    assert fwd.converged and rev.converged
    assert pytest.approx(1.0 / fwd.A, rel=1e-10) == rev.A
    assert pytest.approx(-fwd.B / fwd.A, abs=1e-10) == rev.B
    assert rev.criterion == pytest.approx(fwd.criterion, rel=1e-10)


@pytest.mark.slow
def test_items_mapping_and_custom_nodes_weights(
    fitted_2pl_pair: tuple[FitResult, FitResult, pd.DataFrame],
) -> None:
    res_src, res_tgt, _ = fitted_2pl_pair
    # Subset via iterable and identical mapping give identical constants.
    sub = ["c01", "c02", "c03", "c04"]
    r_list = link(res_src, res_tgt, items=sub)
    r_map = link(res_src, res_tgt, items={k: k for k in sub})
    assert pytest.approx(r_map.A, abs=1e-14) == r_list.A
    assert pytest.approx(r_map.B, abs=1e-14) == r_list.B

    # Custom nodes without weights uses uniform weights; unnormalized weights give identical result.
    grid = np.linspace(-4.0, 4.0, 41)
    r_unif = link(res_src, res_tgt, nodes=grid)
    r_ones = link(res_src, res_tgt, nodes=grid, weights=np.full(41, 7.5))
    assert pytest.approx(r_ones.A, abs=1e-14) == r_unif.A
    assert pytest.approx(r_ones.B, abs=1e-14) == r_unif.B
    assert r_unif.criterion == pytest.approx(r_ones.criterion, rel=1e-14)

    # Custom weights alone uses target's fitted quadrature nodes.
    w_t = np.ones(res_tgt._quadrature.n_points)
    r_wonly = link(res_src, res_tgt, weights=w_t)
    r_explicit = link(res_src, res_tgt, nodes=res_tgt._quadrature.nodes(), weights=w_t)
    assert pytest.approx(r_explicit.A, abs=1e-14) == r_wonly.A
    assert pytest.approx(r_explicit.B, abs=1e-14) == r_wonly.B


def test_transform_theta_all_input_types(
    fitted_2pl_pair: tuple[FitResult, FitResult, pd.DataFrame],
) -> None:
    res_src, res_tgt, df_src = fitted_2pl_pair
    linked = link(res_src, res_tgt)
    a, b = linked.A, linked.B

    # Scalar
    assert linked.transform_theta(0.5) == pytest.approx(a * 0.5 + b)
    assert isinstance(linked.transform_theta(1), float)

    # Array & JAX array
    arr = np.array([-1.0, 0.0, 2.0])
    np.testing.assert_allclose(linked.transform_theta(arr), a * arr + b)
    np.testing.assert_allclose(linked.transform_theta(jnp.asarray(arr)), a * arr + b)

    # Series
    ser = pd.Series([0.2, -0.8], index=["p1", "p2"], name="ability")
    out_ser = linked.transform_theta(ser)
    assert isinstance(out_ser, pd.Series)
    assert list(out_ser.index) == ["p1", "p2"]
    np.testing.assert_allclose(out_ser.to_numpy(), a * ser.to_numpy() + b)

    # EAP DataFrame (theta + se)
    eap_df = res_src.score(df_src.head(10), method="EAP")
    eap_t = linked.transform_theta(eap_df)
    assert isinstance(eap_t, pd.DataFrame)
    np.testing.assert_allclose(eap_t["theta"], a * eap_df["theta"] + b)
    np.testing.assert_allclose(eap_t["se"], a * eap_df["se"])

    # PV DataFrame (pv1..pv3)
    pv_df = res_src.score(df_src.head(10), method="PV", key=jax.random.key(0), n_draws=3)
    pv_t = linked.transform_theta(pv_df)
    for col in ("pv1", "pv2", "pv3"):
        np.testing.assert_allclose(pv_t[col], a * pv_df[col] + b)

    # A frame with both theta/se and plausible values transforms all of them.
    both = eap_df.join(pv_df[["pv1", "pv2"]])
    both_t = linked.transform_theta(both)
    np.testing.assert_allclose(both_t["theta"], a * eap_df["theta"] + b)
    np.testing.assert_allclose(both_t["se"], a * eap_df["se"])
    for col in ("pv1", "pv2"):
        np.testing.assert_allclose(both_t[col], a * pv_df[col] + b)


def test_fixed_slope_items_transform_to_slope_one_over_a() -> None:
    """Rasch / PCM source items get a* = 1/A on the target scale and fix_from keeps that slope."""
    a_true, b_true = 1.3, 0.4
    src_spec: dict[str, ItemModel] = {
        "r1": Rasch(),
        "r2": Rasch(),
        "r3": Rasch(),
        "p1": PCM(n_categories=3),
        "p2": PCM(n_categories=3),
    }
    src_model = Model.from_items(src_spec)
    src_params = {
        "r1": np.array([1.0, -0.5]),
        "r2": np.array([1.0, 0.3]),
        "r3": np.array([1.0, 1.1]),
        "p1": np.array([1.0, 0.6, -0.2]),
        "p2": np.array([1.0, 0.1, -0.9]),
    }
    # The target holds the same items on the target scale (slope 1/A), so it needs 2PL / GPCM.
    tgt_spec: dict[str, ItemModel] = {
        n: (TwoPL() if isinstance(m, Rasch) else GPCM(n_categories=3)) for n, m in src_spec.items()
    }
    tgt_model = Model.from_items(tgt_spec)
    tgt_params = {
        n: np.asarray(
            transform_kernel_params(
                src_model.item_kernel(m),
                jnp.asarray(src_params[n]),
                jnp.asarray(a_true),
                jnp.asarray(b_true),
            )
        )
        for n, m in src_spec.items()
    }
    source = _make_fixed_fit(src_model, src_params, seed=3)
    target = _make_fixed_fit(tgt_model, tgt_params, seed=4)
    # Kernels differ only in which parameters are free, so Rasch/2PL and PCM/GPCM share a kernel.
    linked = link(source, target)
    assert pytest.approx(a_true, abs=1e-10) == linked.A
    assert pytest.approx(b_true, abs=1e-10) == linked.B
    tbl = linked.items.set_index("item")
    np.testing.assert_allclose(tbl["a"].to_numpy(dtype=float), 1.0 / a_true, atol=1e-10)

    # fix_from on the Rasch / PCM model fixes the slope at 1/A (not at 1).
    fixed = dict(((i, p), v) for i, p, v in src_model.fix_from(linked.items).fixed)
    for name in src_spec:
        assert fixed[(name, "a")] == pytest.approx(1.0 / a_true, abs=1e-10)


def test_summary_str_and_html(
    fitted_2pl_pair: tuple[FitResult, FitResult, pd.DataFrame],
) -> None:
    res_src, res_tgt, _ = fitted_2pl_pair
    linked = link(res_src, res_tgt, method="haebara", symmetric=True)
    text = linked.summary(digits=3)
    assert "LinkResult (haebara, symmetric)" in text
    assert "Linking constants" in text
    assert "Transformed source items" in text
    assert "LinkResult(method='haebara'" in str(linked)
    html_str = linked._repr_html_()
    assert "flexpyirt-result" in html_str and "<table" in html_str


# ---- Ogasawara (2001) delta-method SE validation (finite differences + Monte Carlo) -------------


@pytest.mark.slow
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("symmetric", [False, True])
def test_ogasawara_se_matches_central_finite_differences(
    fitted_2pl_pair: tuple[FitResult, FitResult, pd.DataFrame],
    method: LinkMethod,
    symmetric: bool,
) -> None:
    """Validate implicit-function-theorem delta-method vcov against FD of the full optimizer."""
    res_src, res_tgt, _ = fitted_2pl_pair
    linked = link(res_src, res_tgt, method=method, symmetric=symmetric)
    assert linked.vcov is not None
    assert np.isfinite(linked.A_se) and linked.A_se > 0
    assert np.isfinite(linked.B_se) and linked.B_se > 0

    # Rebuild the exact LinkingProblem and differentiate solve_linking wrt (eta_s, eta_t).
    common = list(linked.common_items)
    src_sub = res_src.parameters[res_src.parameters["item"].isin(common)]
    tgt_sub = res_tgt.parameters[res_tgt.parameters["item"].isin(common)]
    src_fids = sorted(int(f) for f in src_sub["free_id"].unique() if f >= 0)
    tgt_fids = sorted(int(f) for f in tgt_sub["free_id"].unique() if f >= 0)
    assert res_src.vcov is not None and res_tgt.vcov is not None
    vcov_s = res_src.vcov[np.ix_(src_fids, src_fids)]
    vcov_t = res_tgt.vcov[np.ix_(tgt_fids, tgt_fids)]

    eta_s = jnp.asarray(res_src._eta[src_fids], dtype=jnp.float64)
    eta_t = jnp.asarray(res_tgt._eta[tgt_fids], dtype=jnp.float64)
    s_pos = {fid: i for i, fid in enumerate(src_fids)}
    t_pos = {fid: i for i, fid in enumerate(tgt_fids)}

    def _pack(res: FitResult, pos: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
        sub = res.parameters[res.parameters["item"] != "__latent__"]
        est_p = sub.pivot(index="item", columns="param", values="estimate")
        fid_p = sub.pivot(index="item", columns="param", values="free_id")
        vals = np.stack([est_p.loc[i, ["a", "d"]].to_numpy(dtype=float) for i in common])
        fids = np.stack(
            [
                np.array(
                    [pos[int(f)] for f in fid_p.loc[i, ["a", "d"]].to_numpy(dtype=int)],
                    dtype=np.int32,
                )
                for i in common
            ]
        )
        return vals, fids

    s_vals, s_idx = _pack(res_src, s_pos)
    t_vals, t_idx = _pack(res_tgt, t_pos)
    nodes = jnp.asarray(res_tgt._quadrature.nodes(), dtype=jnp.float64)
    w = jax.nn.softmax(-0.5 * nodes**2)
    prob = LinkingProblem(
        families=(
            LinkingFamily(
                kernel=DichotomousKernel(),
                src_params=jnp.asarray(s_vals),
                tgt_params=jnp.asarray(t_vals),
                src_free_index=jnp.asarray(s_idx),
                tgt_free_index=jnp.asarray(t_idx),
            ),
        ),
        tgt_nodes=nodes,
        tgt_weights=w,
        src_nodes=nodes,
        src_weights=w,
    )
    ab0 = jnp.array([linked.A, linked.B])
    solver = jax.jit(
        lambda es, et: solve_linking(
            prob, es, et, ab0, method=method, symmetric=symmetric, tol=1e-12, max_iter=200
        )[0]
    )

    eps = 1e-6
    j_s_fd = np.zeros((2, len(src_fids)))
    for i in range(len(src_fids)):
        e = jnp.zeros_like(eta_s).at[i].set(eps)
        j_s_fd[:, i] = np.asarray(solver(eta_s + e, eta_t) - solver(eta_s - e, eta_t)) / (2 * eps)
    j_t_fd = np.zeros((2, len(tgt_fids)))
    for i in range(len(tgt_fids)):
        e = jnp.zeros_like(eta_t).at[i].set(eps)
        j_t_fd[:, i] = np.asarray(solver(eta_s, eta_t + e) - solver(eta_s, eta_t - e)) / (2 * eps)

    cov_fd = j_s_fd @ vcov_s @ j_s_fd.T + j_t_fd @ vcov_t @ j_t_fd.T
    np.testing.assert_allclose(linked.vcov, cov_fd, rtol=1e-6, atol=1e-10)
    assert linked.A_se == pytest.approx(math.sqrt(cov_fd[0, 0]), rel=1e-6)
    assert linked.B_se == pytest.approx(math.sqrt(cov_fd[1, 1]), rel=1e-6)


def test_ogasawara_se_monte_carlo_linearization_check(
    fitted_2pl_pair: tuple[FitResult, FitResult, pd.DataFrame],
) -> None:
    """Monte Carlo linearization check of the Ogasawara delta-method vcov.

    Draws common-item parameters from N(eta_hat, vcov) of each calibration and re-solves the
    linking with the quadrature weights held fixed. This checks the delta-method linearization
    under its own assumptions (normal parameter estimates, fixed weights); it is not a bootstrap
    of the calibrations (no data are resampled and no models are refitted).
    """
    res_src, res_tgt, _ = fitted_2pl_pair
    linked = link(res_src, res_tgt, method="stocking_lord", symmetric=False)
    assert linked.vcov is not None and res_src.vcov is not None and res_tgt.vcov is not None

    common = list(linked.common_items)
    src_fids = sorted(
        int(f)
        for f in res_src.parameters[res_src.parameters["item"].isin(common)]["free_id"].unique()
        if f >= 0
    )
    tgt_fids = sorted(
        int(f)
        for f in res_tgt.parameters[res_tgt.parameters["item"].isin(common)]["free_id"].unique()
        if f >= 0
    )
    vcov_s = res_src.vcov[np.ix_(src_fids, src_fids)]
    vcov_t = res_tgt.vcov[np.ix_(tgt_fids, tgt_fids)]
    s_pos = {fid: i for i, fid in enumerate(src_fids)}
    t_pos = {fid: i for i, fid in enumerate(tgt_fids)}

    def _idx(res: FitResult, pos: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
        sub = res.parameters[res.parameters["item"] != "__latent__"]
        est_p = sub.pivot(index="item", columns="param", values="estimate")
        fid_p = sub.pivot(index="item", columns="param", values="free_id")
        return (
            np.stack([est_p.loc[i, ["a", "d"]].to_numpy(dtype=float) for i in common]),
            np.stack(
                [
                    np.array(
                        [pos[int(f)] for f in fid_p.loc[i, ["a", "d"]].to_numpy(dtype=int)],
                        dtype=np.int32,
                    )
                    for i in common
                ]
            ),
        )

    s_vals, s_idx = _idx(res_src, s_pos)
    t_vals, t_idx = _idx(res_tgt, t_pos)
    nodes = jnp.asarray(res_tgt._quadrature.nodes(), dtype=jnp.float64)
    w = jax.nn.softmax(-0.5 * nodes**2)
    prob = LinkingProblem(
        families=(
            LinkingFamily(
                kernel=DichotomousKernel(),
                src_params=jnp.asarray(s_vals),
                tgt_params=jnp.asarray(t_vals),
                src_free_index=jnp.asarray(s_idx),
                tgt_free_index=jnp.asarray(t_idx),
            ),
        ),
        tgt_nodes=nodes,
        tgt_weights=w,
        src_nodes=nodes,
        src_weights=w,
    )
    ab0 = jnp.array([linked.A, linked.B])
    batch_solve = jax.jit(
        jax.vmap(
            lambda es, et: solve_linking(
                prob, es, et, ab0, method="stocking_lord", symmetric=False, tol=1e-10, max_iter=100
            )[0]
        )
    )
    rng = np.random.default_rng(77)
    n_boot = 2000
    es_draws = rng.multivariate_normal(res_src._eta[src_fids], vcov_s, size=n_boot)
    et_draws = rng.multivariate_normal(res_tgt._eta[tgt_fids], vcov_t, size=n_boot)
    ab_draws = np.asarray(batch_solve(jnp.asarray(es_draws), jnp.asarray(et_draws)))
    boot_se = np.std(ab_draws, axis=0, ddof=1)
    assert linked.A_se == pytest.approx(float(boot_se[0]), rel=0.06)
    assert linked.B_se == pytest.approx(float(boot_se[1]), rel=0.06)


@pytest.mark.slow
def test_se_none_and_multigroup_and_eh(
    fitted_2pl_pair: tuple[FitResult, FitResult, pd.DataFrame],
) -> None:
    res_src, res_tgt, df_src = fitted_2pl_pair
    # When either side has se=None on free common items, vcov is None and SEs are NaN.
    res_nose = fit(res_src._model, df_src, method=BAEM(tol=1e-6, squarem=True), se=None)
    linked_nose = link(res_nose, res_tgt)
    assert linked_nose.vcov is None
    assert math.isnan(linked_nose.A_se) and math.isnan(linked_nose.B_se)

    # Linking a free source fit to a fixed-bank target (0 free target params) still gives SEs.
    tgt_fixed = _make_fixed_fit(
        res_tgt._model,
        {row["item"]: np.array([row["a"], row["d"]]) for _, row in res_tgt.items.iterrows()},
    )
    linked_bank = link(res_src, tgt_fixed)
    assert linked_bank.vcov is not None
    assert np.isfinite(linked_bank.A_se) and linked_bank.A_se > 0

    # EmpiricalHistogram fit works as target / source (uses EH weights by default).
    eh_spec = Model.uniform(TwoPL(), items=list(df_src.columns)).with_latent(EmpiricalHistogram())
    res_eh = fit(
        eh_spec,
        df_src,
        method=BAEM(tol=1e-4, max_iter=300, squarem=True),
        quadrature=Quadrature.equispaced(21, bounds=(-4, 4)),
        se=None,
    )
    linked_eh = link(res_src, res_eh, symmetric=True)
    assert linked_eh.converged and linked_eh.A > 0

    # Multiple-group FitResult works when source_group= / target_group= is selected explicitly.
    mg_spec = (
        Model.uniform(TwoPL(), items=list(df_src.columns))
        .with_groups(ref=Normal.standard(), focal=Normal.free())
        .equal_across_groups(items=["c01", "c02", "c03"])
    )
    grp = np.repeat(["ref", "focal"], len(df_src) // 2)
    res_mg = fit(mg_spec, df_src, group=grp, method=BAEM(tol=1e-5, squarem=True), se=None)
    with pytest.raises(SpecificationError, match="source_group"):
        link(res_mg, res_tgt)
    with pytest.raises(SpecificationError, match="target_group"):
        link(res_src, res_mg)
    with pytest.raises(SpecificationError, match="Unknown source_group"):
        link(res_mg, res_tgt, source_group="missing")
    linked_mg = link(res_mg, res_tgt, source_group="focal")
    assert linked_mg.converged and linked_mg.A > 0


# ---- Validation and error handling ---------------------------------------------------------------


@pytest.mark.slow
def test_errors_and_validation(
    fitted_2pl_pair: tuple[FitResult, FitResult, pd.DataFrame],
) -> None:
    res_src, res_tgt, df_src = fitted_2pl_pair

    with pytest.raises(SpecificationError, match="FitResult"):
        link("bad", res_tgt)  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match="FitResult"):
        link(res_src, "bad")  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match="Unknown method"):
        link(res_src, res_tgt, method="mean-sigma")  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match="symmetric"):
        link(res_src, res_tgt, symmetric="yes")  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match="tol"):
        link(res_src, res_tgt, tol=0.0)
    with pytest.raises(SpecificationError, match="max_iter"):
        link(res_src, res_tgt, max_iter=0)
    with pytest.raises(SpecificationError, match="source_group"):
        link(res_src, res_tgt, source_group="all")
    with pytest.raises(SpecificationError, match="target_group"):
        link(res_src, res_tgt, target_group="all")
    with pytest.raises(SpecificationError, match="not a str"):
        link(res_src, res_tgt, items="c01")
    with pytest.raises(SpecificationError, match="at least one common item"):
        link(res_src, res_tgt, items=[])
    with pytest.raises(SpecificationError, match="Unknown source items"):
        link(res_src, res_tgt, items=["nope"])
    with pytest.raises(SpecificationError, match="Unknown target items"):
        link(res_src, res_tgt, items={"c01": "nope"})
    with pytest.raises(SpecificationError, match="Duplicate target items"):
        link(res_src, res_tgt, items={"c01": "c01", "c02": "c01"})
    with pytest.raises(SpecificationError, match="1-D"):
        link(res_src, res_tgt, nodes=np.ones((2, 2)))
    with pytest.raises(SpecificationError, match="finite"):
        link(res_src, res_tgt, nodes=[0.0, np.nan])
    with pytest.raises(SpecificationError, match="shape"):
        link(res_src, res_tgt, nodes=[0.0, 1.0], weights=[1.0])
    with pytest.raises(SpecificationError, match="non-negative"):
        link(res_src, res_tgt, nodes=[0.0, 1.0], weights=[-1.0, 2.0])
    with pytest.raises(SpecificationError, match="non-negative"):
        link(res_src, res_tgt, nodes=[0.0, 1.0], weights=[0.0, 0.0])
    with pytest.raises(SpecificationError, match="weights length"):
        link(res_src, res_tgt, weights=[0.5, 0.5])

    # No common items at all.
    disjoint = _make_fixed_fit(
        Model.uniform(TwoPL(), items=["x1", "x2"]),
        {"x1": np.array([1.0, 0.0]), "x2": np.array([1.2, -0.5])},
    )
    with pytest.raises(SpecificationError, match="No common items"):
        link(disjoint, res_tgt)

    # Mismatched item kernels / categories on common items.
    m3pl = _make_fixed_fit(
        Model.uniform(ThreePL(), items=["c01", "c02"]),
        {"c01": np.array([1.2, 0.0, 0.2, 1.0]), "c02": np.array([1.0, -0.3, 0.15, 1.0])},
    )
    with pytest.raises(SpecificationError, match="same item kernel"):
        link(m3pl, res_tgt)

    mgrm3 = _make_fixed_fit(
        Model.uniform(GRM(n_categories=3), items=["p1", "p2"]),
        {"p1": np.array([1.1, 0.8, -0.7]), "p2": np.array([1.3, 0.5, -0.9])},
    )
    mgrm4 = _make_fixed_fit(
        Model.uniform(GRM(n_categories=4), items=["p1", "p2"]),
        {"p1": np.array([1.1, 1.0, 0.0, -1.0]), "p2": np.array([1.3, 0.9, -0.1, -1.1])},
    )
    mgpcm3 = _make_fixed_fit(
        Model.uniform(GPCM(n_categories=3), items=["p1", "p2"]),
        {"p1": np.array([1.1, 0.5, -0.4]), "p2": np.array([1.3, 0.2, -0.6])},
    )
    with pytest.raises(SpecificationError, match="same item kernel"):
        link(mgrm3, mgrm4)
    with pytest.raises(SpecificationError, match="same item kernel"):
        link(mgrm3, mgpcm3)

    # Multidimensional FitResult rejected.
    md_spec = Model.uniform(TwoPL(), items=list(df_src.columns)[:6]).with_factors(
        {"F1": list(df_src.columns)[:3], "F2": list(df_src.columns)[3:6]}
    )
    with pytest.warns(ConvergenceWarning):
        res_md = fit(
            md_spec,
            df_src.iloc[:200, :6],
            method=BAEM(tol=1e-2, max_iter=5),
            quadrature=Quadrature.equispaced(7, bounds=(-3, 3)),
            se=None,
        )
    with pytest.raises(SpecificationError, match="unidimensional"):
        link(res_md, res_tgt)
    with pytest.raises(SpecificationError, match="unidimensional"):
        link(res_src, res_md)

    # transform_theta errors & summary digits validation.
    linked = link(res_src, res_tgt)
    with pytest.raises(SpecificationError, match="boolean"):
        linked.transform_theta(True)
    with pytest.raises(SpecificationError, match="theta"):
        linked.transform_theta(pd.DataFrame({"other": [1.0]}))
    with pytest.raises(SpecificationError, match="digits"):
        linked.summary(digits=-1)

    # Non-convergence warning when max_iter=1 is too small from an impossible tolerance.
    with pytest.warns(ConvergenceWarning, match="did not converge"):
        r_nc = link(res_src, res_tgt, tol=1e-30, max_iter=1)
    assert not r_nc.converged
    # Delta-method SEs are only valid at a stationary point: none are reported here.
    assert r_nc.vcov is None
    assert math.isnan(r_nc.A_se) and math.isnan(r_nc.B_se)


@pytest.fixture(scope="module")
def multigroup_anchor_fit() -> FitResult:
    """Two-group 2PL calibration with 4 equality-constrained anchors and 4 group-specific items."""
    items = [f"c{j}" for j in range(8)]
    rng = np.random.default_rng(11)
    true = {
        n: np.array([float(rng.lognormal(0.1, 0.25)), float(rng.normal(0.0, 0.9))]) for n in items
    }
    base = Model.uniform(TwoPL(), items=items)
    df = pd.concat(
        [
            _simulate(base, true, n_obs=1500, seed=21),
            _simulate(base, true, n_obs=1500, seed=22, mean=0.5, sd=1.2),
        ],
        ignore_index=True,
    )
    spec = base.with_groups(ref=Normal.standard(), focal=Normal.free()).equal_across_groups(
        items=items[:4]
    )
    grp = np.repeat(["ref", "focal"], 1500)
    return fit(spec, df, group=grp, method=BAEM(tol=1e-8, squarem=True), se="oakes")


@pytest.mark.slow
def test_same_multigroup_fit_uses_joint_vcov(multigroup_anchor_fit: FitResult) -> None:
    """One FitResult on both sides: cross-covariance from the joint vcov enters the SEs."""
    res = multigroup_anchor_fit
    assert res.vcov is not None
    anchors = ["c0", "c1", "c2", "c3"]
    # Shared (equality-constrained) anchors give (A, B) = (1, 0) for every parameter value, so the
    # delta-method covariance must vanish; ignoring the cross block would give SEs ~0.03-0.04.
    shared = link(res, res, source_group="focal", target_group="ref", items=anchors)
    assert shared.converged
    assert pytest.approx(1.0, abs=1e-10) == shared.A
    assert pytest.approx(0.0, abs=1e-10) == shared.B
    assert shared.vcov is not None
    assert np.max(np.abs(shared.vcov)) < 1e-12

    # Group-specific items: compare against the explicit joint-covariance delta method, where the
    # Jacobians come from autodiff through the converged solver and the joint vcov block includes
    # the cross-covariance between the two groups' parameters.
    other = ["c4", "c5", "c6", "c7"]
    linked = link(res, res, source_group="focal", target_group="ref", items=other)
    assert linked.converged and linked.vcov is not None
    params = res.parameters[res.parameters["item"].isin(other)]
    fid = {
        (str(g), str(i), str(p)): int(f)
        for g, i, p, f in zip(
            params["group"],
            params["item"],
            params["param"],
            params["free_id"].to_numpy(dtype=int),
            strict=True,
        )
        if f >= 0
    }
    src_ids = [fid[("focal", i, p)] for i in other for p in ("a", "d")]
    tgt_ids = [fid[("ref", i, p)] for i in other for p in ("a", "d")]
    n = len(other)
    fam = LinkingFamily(
        kernel=DichotomousKernel(),
        src_params=jnp.asarray(res._eta[src_ids].reshape(n, 2)),
        tgt_params=jnp.asarray(res._eta[tgt_ids].reshape(n, 2)),
        src_free_index=jnp.arange(2 * n, dtype=jnp.int32).reshape(n, 2),
        tgt_free_index=jnp.arange(2 * n, dtype=jnp.int32).reshape(n, 2),
    )
    nodes = jnp.asarray(res._quadrature.nodes(), dtype=jnp.float64)
    lat = res.parameters[
        (res.parameters["item"] == "__latent__") & (res.parameters["group"] == "focal")
    ]
    lat_est = dict(zip(lat["param"], lat["estimate"].to_numpy(dtype=float), strict=True))
    mu, var = lat_est["mean"], lat_est["var"]
    w_t = jax.nn.softmax(-0.5 * nodes**2)
    w_s = jax.nn.softmax(-0.5 * (nodes - mu) ** 2 / var)
    prob = LinkingProblem(
        families=(fam,), tgt_nodes=nodes, tgt_weights=w_t, src_nodes=nodes, src_weights=w_s
    )

    def ab_of(eta_joint: jax.Array) -> jax.Array:
        es, et = eta_joint[: 2 * n], eta_joint[2 * n :]
        return solve_linking(
            prob,
            es,
            et,
            jnp.array([linked.A, linked.B]),
            method="stocking_lord",
            symmetric=False,
            tol=1e-13,
            max_iter=50,
        )[0]

    eta_joint = jnp.asarray(res._eta[src_ids + tgt_ids])
    # Implicit-function Jacobian via central differences of the converged solution.
    eps = 1e-6
    cols = []
    for k in range(4 * n):
        e = jnp.zeros(4 * n).at[k].set(eps)
        cols.append((ab_of(eta_joint + e) - ab_of(eta_joint - e)) / (2 * eps))
    jac = np.stack([np.asarray(c) for c in cols], axis=1)
    joint = res.vcov[np.ix_(src_ids + tgt_ids, src_ids + tgt_ids)]
    expected = jac @ joint @ jac.T
    np.testing.assert_allclose(linked.vcov, expected, rtol=1e-5, atol=1e-12)
    # The cross block is not negligible here: dropping it changes the covariance.
    indep = (
        jac[:, : 2 * n] @ joint[: 2 * n, : 2 * n] @ jac[:, : 2 * n].T
        + jac[:, 2 * n :] @ joint[2 * n :, 2 * n :] @ jac[:, 2 * n :].T
    )
    assert not np.allclose(indep, expected, rtol=1e-3, atol=0.0)


@pytest.mark.slow
def test_link_requires_x64() -> None:
    code = (
        "import flexpyirt\n"
        "from flexpyirt.errors import PrecisionError\n"
        "try:\n"
        "    flexpyirt.link(None, None)\n"
        "except PrecisionError:\n"
        "    pass\n"
        "else:\n"
        "    raise SystemExit('PrecisionError was not raised')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr + proc.stdout


# ---- Default integration grid = the fitted latent distribution (issue #39) ----------------------


@pytest.mark.parametrize(("mean", "var"), [(0.0, 1.0), (0.4, 1.6)])
def test_fitted_grid_keeps_gauss_hermite_base_weights(mean: float, var: float) -> None:
    from flexpyirt.linking import _fitted_grid

    n = 21
    res = _make_fixed_fit(
        Model.uniform(TwoPL(), items=[f"i{j}" for j in range(4)]),
        {f"i{j}": np.array([1.2, 0.3 * j - 0.4]) for j in range(4)},
        latent_mean=mean,
        latent_var=var,
        quadrature=Quadrature.gauss_hermite(n),
    )
    nodes, w = _fitted_grid(res, str(res.parameters["group"].iloc[0]), 0)
    # Independent rule: GH base weights reweighted by phi(theta; mean, var) / phi(theta; 0, 1).
    x, gw = np.polynomial.hermite.hermgauss(n)
    np.testing.assert_allclose(nodes, np.sqrt(2.0) * x, rtol=0, atol=1e-13)
    expected = gw * np.exp(-0.5 * (nodes - mean) ** 2 / var + 0.5 * nodes**2)
    np.testing.assert_allclose(w, expected / expected.sum(), rtol=1e-12, atol=1e-15)
    if (mean, var) == (0.0, 1.0):  # GH integrates N(0, 1) moments exactly
        assert abs(float(w @ nodes)) < 1e-13
        assert float(w @ nodes**2) == pytest.approx(1.0, abs=1e-13)


@pytest.mark.slow
def test_fitted_grid_uses_the_fitted_davidian_density() -> None:
    from flexpyirt import DavidianCurve
    from flexpyirt.latent import davidian_log_weights, davidian_param_names
    from flexpyirt.linking import _fitted_grid

    rng = np.random.default_rng(20261122)
    theta = (rng.chisquare(4.0, size=800) - 4.0) / np.sqrt(8.0)
    a, d = rng.uniform(0.9, 1.8, size=10), np.linspace(-1.6, 1.6, 10)
    y = (rng.random((800, 10)) < 1 / (1 + np.exp(-(theta[:, None] * a + d)))).astype(int)
    df = pd.DataFrame(y, columns=[f"q{j:02d}" for j in range(10)])
    spec = Model.uniform(TwoPL(), items=df.columns).with_latent(DavidianCurve(2))
    res = fit(spec, df, method=BAEM(tol=1e-7, max_iter=1000, squarem=True), se=None)

    nodes, w = _fitted_grid(res, str(res.parameters["group"].iloc[0]), 0)
    lat = res.parameters[res.parameters["item"] == "__latent__"].set_index("param")["estimate"]
    params = jnp.asarray(lat.loc[list(davidian_param_names(2))].to_numpy(dtype=float))
    expected = np.exp(np.asarray(davidian_log_weights(jnp.asarray(nodes)[:, None], params)))
    np.testing.assert_allclose(w, expected / expected.sum(), rtol=1e-12, atol=1e-15)
    normal = np.exp(-0.5 * (nodes - lat["mean"]) ** 2 / lat["var"])
    assert np.max(np.abs(w - normal / normal.sum())) > 1e-3  # skewed: not N(mean, var)
