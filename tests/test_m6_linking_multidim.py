"""Unit tests for multidimensional scale linking, theta* = A theta + beta (DESIGN.md §19)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from flexpyirt import GPCM, GRM, FitResult, FourPL, Model, Nominal, Quadrature, TwoPL, fit, link
from flexpyirt._linking import (
    LinkingFamily,
    LinkingProblem,
    LinkMethod,
    LinkTransformation,
    linking_constants,
    linking_criterion,
    multidim_linking_criterion,
    multidim_linking_vcov,
    n_transformation_params,
    solve_multidim_linking,
    transform_multidim_kernel_params,
    unpack_transformation,
)
from flexpyirt.errors import SpecificationError
from flexpyirt.items import ItemKernel, ItemModel

METHODS: tuple[LinkMethod, ...] = ("stocking_lord", "haebara")
TRANSFORMATIONS: tuple[LinkTransformation, ...] = ("diagonal", "orthogonal", "oblique")

_solve = jax.jit(solve_multidim_linking, static_argnames=("method", "symmetric", "transformation"))
_vcov = jax.jit(multidim_linking_vcov, static_argnames=("method", "symmetric", "transformation"))


def _rotation_2d(angle: float) -> np.ndarray:
    return np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])


def _true_params(transformation: LinkTransformation, n_factors: int) -> np.ndarray:
    """A valid, clearly non-trivial parameter vector ``c`` of each family."""
    beta = np.array([0.4, -0.3, 0.25])[:n_factors]
    if transformation == "diagonal":
        return np.concatenate([np.array([1.3, 0.8, 1.1])[:n_factors], beta])
    if transformation == "orthogonal":
        angles = np.array([0.35, -0.2, 0.15])[: n_factors * (n_factors - 1) // 2]
        return np.concatenate([[1.2], angles, beta])
    a_mat = np.array([[1.2, 0.3, -0.1], [-0.2, 0.9, 0.15], [0.1, -0.25, 1.1]])
    return np.concatenate([a_mat[:n_factors, :n_factors].reshape(-1), beta])


def _target_items(n_factors: int) -> list[tuple[ItemKernel, np.ndarray]]:
    """Common items of every kernel with slopes spanning all factor directions."""
    rng = np.random.default_rng(5)
    models: list[tuple[ItemModel, list[float]]] = [
        (TwoPL(), [0.3]),
        (TwoPL(), [-0.6]),
        (TwoPL(), [0.9]),
        (FourPL(), [-0.2, 0.15, 0.9]),
        (GRM(n_categories=4), [1.1, 0.1, -1.0]),
        (GPCM(n_categories=3), [0.5, -0.4]),
        (Nominal(n_categories=4), [1.8, -0.3, 0.6, 0.2, -0.7]),
        (TwoPL(), [0.0]),
    ]
    out = []
    for i, (item_model, rest) in enumerate(models):
        slopes = rng.uniform(0.5, 1.4, n_factors)
        if i < n_factors:  # some simple-structure items
            slopes = np.where(np.arange(n_factors) == i, slopes, 0.0)
        out.append((item_model.kernel.with_factors(n_factors), np.concatenate([slopes, rest])))
    return out


def _problem(
    n_factors: int, c_true: np.ndarray, transformation: LinkTransformation, *, free: bool = False
) -> tuple[LinkingProblem, jax.Array, jax.Array]:
    """Noiseless problem: source items are the exact inverse transforms of the target items.

    With ``free=True`` every item parameter is a free parameter (for the delta method).
    """
    a_mat, beta, valid = unpack_transformation(jnp.asarray(c_true), n_factors, transformation)
    assert bool(valid)
    a_inv = jnp.linalg.inv(a_mat)
    families = []
    eta_s: list[float] = []
    eta_t: list[float] = []
    for kernel, tgt in _target_items(n_factors):
        src = np.asarray(
            transform_multidim_kernel_params(kernel, jnp.asarray(tgt), a_inv, -a_inv @ beta)
        )
        if free:
            s_idx = np.arange(len(eta_s), len(eta_s) + src.size, dtype=np.int32)
            t_idx = np.arange(len(eta_t), len(eta_t) + tgt.size, dtype=np.int32)
            eta_s.extend(src.tolist())
            eta_t.extend(tgt.tolist())
        else:
            s_idx = t_idx = np.full(src.size, -1, dtype=np.int32)
        families.append(
            LinkingFamily(
                kernel=kernel,
                src_params=jnp.asarray(src[None, :]),
                tgt_params=jnp.asarray(tgt[None, :]),
                src_free_index=jnp.asarray(s_idx[None, :]),
                tgt_free_index=jnp.asarray(t_idx[None, :]),
            )
        )
    quad = Quadrature.equispaced(11 if n_factors == 2 else 7, bounds=(-4.0, 4.0))
    nodes = jnp.asarray(quad.grid(n_factors))
    w_t = jnp.exp(-0.5 * jnp.sum(nodes**2, axis=1))
    w_s = jnp.exp(-0.5 * jnp.sum((nodes - 0.2) ** 2, axis=1) / 1.3)
    prob = LinkingProblem(
        families=tuple(families),
        tgt_nodes=nodes,
        tgt_weights=w_t / jnp.sum(w_t),
        src_nodes=nodes,
        src_weights=w_s / jnp.sum(w_s),
    )
    return prob, jnp.asarray(eta_s, dtype=jnp.float64), jnp.asarray(eta_t, dtype=jnp.float64)


# ---- Transformation of item parameters -----------------------------------------------------------


@pytest.mark.parametrize("n_factors", [2, 3])
@pytest.mark.parametrize("transformation", TRANSFORMATIONS)
def test_transformed_items_reproduce_source_curves(
    n_factors: int, transformation: LinkTransformation
) -> None:
    """P(theta*; xi*) = P(theta; xi) with theta* = A theta + beta, for every kernel."""
    c = jnp.asarray(_true_params(transformation, n_factors))
    a_mat, beta, valid = unpack_transformation(c, n_factors, transformation)
    assert bool(valid)
    theta = jnp.asarray(np.random.default_rng(1).normal(size=(40, n_factors)))
    theta_star = theta @ a_mat.T + beta
    for kernel, params in _target_items(n_factors):
        xi_star = transform_multidim_kernel_params(kernel, jnp.asarray(params), a_mat, beta)
        np.testing.assert_allclose(
            np.asarray(kernel.log_prob(theta_star, xi_star)),
            np.asarray(kernel.log_prob(theta, jnp.asarray(params))),
            atol=1e-12,
        )
        # Slopes transform by A^{-T}; asymptotes and nominal scoring coefficients do not change.
        np.testing.assert_allclose(
            np.asarray(a_mat.T @ xi_star[:n_factors]), params[:n_factors], atol=1e-13
        )
        for name in ("g", "u", "ak1", "ak2"):
            if name in kernel.param_names:
                idx = kernel.param_names.index(name)
                assert float(xi_star[idx]) == params[idx]


def test_transformation_families() -> None:
    """Shapes, validity regions and the closed forms of the three families of A."""
    assert [n_transformation_params(2, t) for t in TRANSFORMATIONS] == [4, 4, 6]
    assert [n_transformation_params(3, t) for t in TRANSFORMATIONS] == [6, 7, 12]

    a_mat, beta, valid = unpack_transformation(jnp.array([1.3, 0.8, 0.4, -0.3]), 2, "diagonal")
    np.testing.assert_allclose(np.asarray(a_mat), np.diag([1.3, 0.8]))
    np.testing.assert_allclose(np.asarray(beta), [0.4, -0.3])
    assert bool(valid)
    assert not bool(unpack_transformation(jnp.array([1.3, -0.8, 0.4, -0.3]), 2, "diagonal")[2])

    # Orthogonal, D = 2: k [[cos, -sin], [sin, cos]].
    a_mat, _, valid = unpack_transformation(jnp.array([1.2, 0.35, 0.0, 0.0]), 2, "orthogonal")
    np.testing.assert_allclose(np.asarray(a_mat), 1.2 * _rotation_2d(0.35), atol=1e-14)
    assert bool(valid)
    assert not bool(unpack_transformation(jnp.array([-1.2, 0.35, 0.0, 0.0]), 2, "orthogonal")[2])
    # Orthogonal, D = 3: a dilation times a proper rotation.
    a3, _, _ = unpack_transformation(jnp.asarray(_true_params("orthogonal", 3)), 3, "orthogonal")
    rot = np.asarray(a3) / 1.2
    np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1e-13)
    assert np.linalg.det(rot) == pytest.approx(1.0, abs=1e-13)
    # phi_12 alone rotates the first two axes, as in D = 2.
    a3, _, _ = unpack_transformation(
        jnp.array([1.0, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0]), 3, "orthogonal"
    )
    np.testing.assert_allclose(np.asarray(a3)[:2, :2], _rotation_2d(0.35), atol=1e-14)

    # Oblique: row-major vec(A); det <= 0 is invalid.
    c = jnp.array([1.2, 0.3, -0.2, 0.9, 0.4, -0.3])
    a_mat, _, valid = unpack_transformation(c, 2, "oblique")
    np.testing.assert_allclose(np.asarray(a_mat), [[1.2, 0.3], [-0.2, 0.9]])
    assert bool(valid)
    np.testing.assert_allclose(np.asarray(linking_constants(c, 2, "oblique")), np.asarray(c))
    assert not bool(
        unpack_transformation(jnp.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0]), 2, "oblique")[2]
    )


# ---- Criterion -----------------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("symmetric", [False, True])
def test_one_factor_criterion_equals_unidimensional_criterion(
    method: LinkMethod, symmetric: bool
) -> None:
    """With D = 1 the multidimensional criterion is the criterion of DESIGN §16.5."""
    kernels: list[tuple[ItemKernel, list[float], list[float]]] = [
        (TwoPL().kernel, [1.2, -0.3], [1.0, 0.1]),
        (Nominal(n_categories=3).kernel, [0.9, 0.4, 0.5, -0.6], [1.1, 0.7, 0.2, -0.1]),
    ]
    nodes = jnp.linspace(-4.0, 4.0, 21)
    weights = jnp.exp(-0.5 * nodes**2)
    weights = weights / jnp.sum(weights)
    none = jnp.full((1, 1), -1, dtype=jnp.int32)

    def problem(grid: jax.Array) -> LinkingProblem:
        return LinkingProblem(
            families=tuple(
                LinkingFamily(
                    kernel=k,
                    src_params=jnp.asarray([s]),
                    tgt_params=jnp.asarray([t]),
                    src_free_index=jnp.broadcast_to(none, (1, len(s))),
                    tgt_free_index=jnp.broadcast_to(none, (1, len(t))),
                )
                for k, s, t in kernels
            ),
            tgt_nodes=grid,
            tgt_weights=weights,
            src_nodes=grid,
            src_weights=weights,
        )

    empty = jnp.zeros((0,))
    ab = jnp.array([1.3, -0.4])
    uni = linking_criterion(ab, empty, empty, problem(nodes), method=method, symmetric=symmetric)
    for transformation in TRANSFORMATIONS:
        multi = multidim_linking_criterion(
            ab,
            empty,
            empty,
            problem(nodes[:, None]),
            method=method,
            symmetric=symmetric,
            transformation=transformation,
        )
        assert float(multi) == pytest.approx(float(uni), rel=1e-13)


@pytest.mark.parametrize("transformation", TRANSFORMATIONS)
def test_criterion_gradient_matches_finite_differences(transformation: LinkTransformation) -> None:
    prob, eta_s, eta_t = _problem(2, _true_params(transformation, 2), transformation)
    c = jnp.asarray(_true_params(transformation, 2)) + 0.07

    def value(x: jax.Array) -> jax.Array:
        return multidim_linking_criterion(
            x, eta_s, eta_t, prob, method="haebara", symmetric=True, transformation=transformation
        )

    grad = np.asarray(jax.grad(value)(c))
    eps = 1e-6
    fd = np.array(
        [
            float(value(c.at[i].add(eps)) - value(c.at[i].add(-eps))) / (2 * eps)
            for i in range(c.shape[0])
        ]
    )
    np.testing.assert_allclose(grad, fd, rtol=1e-6, atol=1e-9)
    # Outside the valid region the criterion is NaN (rejected by the step-size search).
    invalid = {"diagonal": c.at[0].set(-1.0), "orthogonal": c.at[0].set(-1.0)}.get(
        transformation, jnp.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0])
    )
    assert math.isnan(float(value(invalid)))


# ---- Recovery of a known transformation ----------------------------------------------------------


def _recovery_grid() -> list[object]:
    """Every (factors, family, criterion, symmetric); one cheap combination per family is fast."""
    out: list[object] = []
    for n_factors in (2, 3):
        for transformation in TRANSFORMATIONS:
            for method in METHODS:
                for symmetric in (False, True):
                    fast = n_factors == 2 and method == "haebara" and symmetric
                    out.append(
                        pytest.param(
                            n_factors,
                            transformation,
                            method,
                            symmetric,
                            marks=() if fast else pytest.mark.slow,
                        )
                    )
    return out


@pytest.mark.parametrize(("n_factors", "transformation", "method", "symmetric"), _recovery_grid())
def test_solver_recovers_known_transformation_from_perturbed_start(
    n_factors: int, transformation: LinkTransformation, method: LinkMethod, symmetric: bool
) -> None:
    """Source items are exact inverse transforms: the optimum is the known (A, beta), F = 0."""
    c_true = _true_params(transformation, n_factors)
    prob, eta_s, eta_t = _problem(n_factors, c_true, transformation)
    rng = np.random.default_rng(11)
    c0 = jnp.asarray(c_true + rng.uniform(-0.12, 0.12, c_true.size))
    c_hat, crit, _, n_iter, conv = _solve(
        prob,
        eta_s,
        eta_t,
        c0,
        method=method,
        symmetric=symmetric,
        transformation=transformation,
        tol=1e-11,
        max_iter=200,
    )
    assert bool(conv)
    assert int(n_iter) > 0
    np.testing.assert_allclose(np.asarray(c_hat), c_true, atol=1e-8)
    assert float(crit) < 1e-16


# ---- Delta-method covariance ---------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("transformation", TRANSFORMATIONS)
@pytest.mark.parametrize("method", METHODS)
def test_delta_method_jacobian_matches_resolving_the_linking(
    transformation: LinkTransformation, method: LinkMethod
) -> None:
    """Cov(constants) = G J Sigma J' G' with J checked by re-solving at perturbed estimates."""
    c_true = _true_params(transformation, 2)
    prob, eta_s, eta_t = _problem(2, c_true, transformation, free=True)
    rng = np.random.default_rng(3)
    # Noisy estimates so that the optimum is not a zero-residual point.
    eta_s = eta_s + jnp.asarray(rng.normal(0.0, 0.03, eta_s.shape[0]))
    eta_t = eta_t + jnp.asarray(rng.normal(0.0, 0.03, eta_t.shape[0]))
    kwargs = {"method": method, "symmetric": True, "transformation": transformation}

    def solve(es: jax.Array, et: jax.Array) -> np.ndarray:
        c_hat, _, _, _, conv = _solve(prob, es, et, jnp.asarray(c_true), tol=1e-12, **kwargs)
        assert bool(conv)
        return np.asarray(linking_constants(c_hat, 2, transformation))

    c_hat, _, _, _, conv = _solve(prob, eta_s, eta_t, jnp.asarray(c_true), tol=1e-12, **kwargs)
    assert bool(conv)
    sd_s = rng.uniform(0.02, 0.05, eta_s.shape[0])
    sd_t = rng.uniform(0.02, 0.05, eta_t.shape[0])
    cov, cov_c, valid = _vcov(
        c_hat, eta_s, eta_t, jnp.diag(sd_s**2), jnp.diag(sd_t**2), prob, **kwargs
    )
    assert bool(valid)

    eps = 1e-5
    jac_s = np.stack(
        [
            (solve(eta_s.at[i].add(eps), eta_t) - solve(eta_s.at[i].add(-eps), eta_t)) / (2 * eps)
            for i in range(eta_s.shape[0])
        ],
        axis=1,
    )
    jac_t = np.stack(
        [
            (solve(eta_s, eta_t.at[i].add(eps)) - solve(eta_s, eta_t.at[i].add(-eps))) / (2 * eps)
            for i in range(eta_t.shape[0])
        ],
        axis=1,
    )
    expected = jac_s @ np.diag(sd_s**2) @ jac_s.T + jac_t @ np.diag(sd_t**2) @ jac_t.T
    np.testing.assert_allclose(np.asarray(cov), expected, rtol=2e-5, atol=1e-11)

    # Rank is at most the number of free transformation parameters.
    n_free = n_transformation_params(2, transformation)
    assert np.asarray(cov_c).shape == (n_free, n_free)
    assert np.linalg.matrix_rank(np.asarray(cov), tol=1e-12) == n_free
    if transformation == "diagonal":  # structural zeros A12, A21 have zero variance
        np.testing.assert_array_equal(np.asarray(cov)[[1, 2], :], 0.0)


# ---- link() ---------------------------------------------------------------------------------------

_ITEMS: dict[str, ItemModel] = {
    "c1": TwoPL(),
    "c2": TwoPL(),
    "c3": TwoPL(),
    "c4": TwoPL(),
    "c5": GRM(n_categories=3),
    "c6": GPCM(n_categories=3),
    "c7": Nominal(n_categories=3),
    "c8": FourPL(),
}
_FACTORS = {"verbal": ["c1", "c2", "c5", "c6", "c8"], "math": ["c3", "c4", "c5", "c7", "c8"]}
_TARGET: dict[str, list[float]] = {
    "c1": [1.2, 0.0, 0.3],
    "c2": [0.9, 0.0, -0.5],
    "c3": [0.0, 1.1, 0.2],
    "c4": [0.0, 1.4, -0.4],
    "c5": [0.8, 0.7, 0.9, -0.8],
    "c6": [1.0, 0.0, 0.4, -0.3],
    "c7": [0.0, 0.9, 0.6, 0.5, -0.2],
    "c8": [0.9, 1.1, -0.1, 0.15, 0.9],
}


def _fixed_fit(params: dict[str, list[float]], *, cor: float, seed: int) -> FitResult:
    """Two-factor FitResult with every parameter fixed (all slopes included, zero or not)."""
    model = Model.from_items(_ITEMS).with_factors(_FACTORS)
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(
        {name: rng.integers(0, m.kernel.n_categories, 60) for name, m in _ITEMS.items()}
    )
    table = model.parameter_table(data)
    latent = {"mean_1": 0.0, "mean_2": 0.0, "var_1": 1.0, "var_2": 1.0, "cor_12": cor}
    values = []
    for item, param in zip(table["item"], table["param"], strict=True):
        if item == "__latent__":
            values.append(latent[str(param)])
        else:
            names = model.item_kernel(_ITEMS[str(item)]).param_names
            values.append(params[str(item)][names.index(str(param))])
    table["value"] = np.asarray(values, dtype=float)
    table["free"] = False
    return fit(model, data, table=table, quadrature=Quadrature.equispaced(15), se=None)


def _source_params(a_mat: np.ndarray, beta: np.ndarray) -> dict[str, list[float]]:
    model = Model.from_items(_ITEMS).with_factors(_FACTORS)
    a_inv = np.linalg.inv(a_mat)
    return {
        name: np.asarray(
            transform_multidim_kernel_params(
                model.item_kernel(_ITEMS[name]),
                jnp.asarray(_TARGET[name]),
                jnp.asarray(a_inv),
                jnp.asarray(-a_inv @ beta),
            )
        ).tolist()
        for name in _ITEMS
    }


_KNOWN: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "diagonal": (np.diag([1.3, 0.8]), np.array([0.4, -0.3])),
    "orthogonal": (1.2 * _rotation_2d(0.35), np.array([0.4, -0.3])),
    "oblique": (np.array([[1.2, 0.3], [-0.2, 0.9]]), np.array([0.4, -0.3])),
}


@pytest.mark.slow
@pytest.mark.parametrize("transformation", TRANSFORMATIONS)
@pytest.mark.parametrize("method", METHODS)
def test_link_recovers_known_transformation_and_transforms_items(
    transformation: LinkTransformation, method: LinkMethod
) -> None:
    a_true, beta_true = _KNOWN[transformation]
    target = _fixed_fit(_TARGET, cor=0.3, seed=1)
    source = _fixed_fit(_source_params(a_true, beta_true), cor=0.1, seed=2)
    linked = link(source, target, method=method, symmetric=True, transformation=transformation)
    assert linked.converged
    assert linked.transformation == transformation
    assert linked.A.shape == (2, 2) and linked.B.shape == (2,)
    np.testing.assert_allclose(linked.A, a_true, atol=1e-9)
    np.testing.assert_allclose(linked.B, beta_true, atol=1e-9)
    assert linked.criterion == pytest.approx(0.0, abs=1e-18)
    assert linked.vcov is None  # every parameter is fixed
    assert np.all(np.isnan(linked.A_se)) and np.all(np.isnan(linked.B_se))
    # The transformed source items are the target items.
    items = linked.items.set_index("item")
    model = Model.from_items(_ITEMS).with_factors(_FACTORS)
    for name, item_model in _ITEMS.items():
        for p, value in zip(model.item_kernel(item_model).param_names, _TARGET[name], strict=True):
            assert items.loc[name, p] == pytest.approx(value, abs=1e-9)

    # A larger family contains the smaller one: oblique recovers a diagonal or orthogonal A.
    if transformation != "oblique":
        wide = link(source, target, method=method, transformation="oblique")
        np.testing.assert_allclose(wide.A, a_true, atol=1e-9)

    # Symmetric linking is reversible: swapping the forms gives (A^{-1}, -A^{-1} beta).
    back = link(target, source, method=method, symmetric=True, transformation=transformation)
    np.testing.assert_allclose(back.A, np.linalg.inv(a_true), atol=1e-9)
    np.testing.assert_allclose(back.B, -np.linalg.inv(a_true) @ beta_true, atol=1e-9)


@pytest.mark.slow
def test_link_multidim_transform_theta_summary_and_errors() -> None:
    a_true, beta_true = _KNOWN["oblique"]
    target = _fixed_fit(_TARGET, cor=0.3, seed=1)
    source = _fixed_fit(_source_params(a_true, beta_true), cor=0.1, seed=2)
    oblique = link(source, target, method="haebara", transformation="oblique")
    a_diag, beta_diag = _KNOWN["diagonal"]
    diagonal = link(
        _fixed_fit(_source_params(a_diag, beta_diag), cor=0.3, seed=2), target, method="haebara"
    )
    assert diagonal.transformation == "diagonal"

    theta = np.array([[0.2, -0.5], [1.0, 0.3], [-1.2, 0.8]])
    np.testing.assert_allclose(
        oblique.transform_theta(theta), theta @ oblique.A.T + oblique.B, atol=1e-14
    )
    scores = pd.DataFrame(
        {
            "theta_verbal": theta[:, 0],
            "theta_math": theta[:, 1],
            "se_verbal": [0.3, 0.4, 0.5],
            "se_math": [0.2, 0.3, 0.4],
        }
    )
    out = oblique.transform_theta(scores)
    np.testing.assert_allclose(
        out[["theta_verbal", "theta_math"]].to_numpy(), theta @ oblique.A.T + oblique.B
    )
    assert out[["se_verbal", "se_math"]].isna().all().all()  # needs the posterior covariance
    out = diagonal.transform_theta(scores)
    np.testing.assert_allclose(out["se_verbal"], np.array([0.3, 0.4, 0.5]) * diagonal.A[0, 0])
    np.testing.assert_allclose(out["se_math"], np.array([0.2, 0.3, 0.4]) * diagonal.A[1, 1])
    assert scores["theta_verbal"].iloc[0] == 0.2  # input is not modified
    pv = pd.DataFrame(
        {"pv1_verbal": [0.1], "pv1_math": [0.2], "pv2_verbal": [0.3], "pv2_math": [0.4]}
    )
    out = oblique.transform_theta(pv)
    np.testing.assert_allclose(
        out[["pv2_verbal", "pv2_math"]].to_numpy()[0], oblique.A @ [0.3, 0.4] + oblique.B
    )
    for bad in (0.5, pd.Series([0.1, 0.2]), np.zeros((3, 3)), pd.DataFrame({"theta": [0.0]})):
        with pytest.raises(SpecificationError):
            oblique.transform_theta(bad)

    text = oblique.summary()
    assert "A[1,2]" in text and "B[2]" in text and "oblique" in text
    assert "oblique" in str(oblique) and "A[2,1]" in oblique._repr_html_()

    # Custom (Q, 2) nodes and weights.
    nodes = Quadrature.equispaced(9, bounds=(-3.0, 3.0)).grid(2)
    custom = link(source, target, method="haebara", transformation="oblique", nodes=nodes)
    np.testing.assert_allclose(custom.A, a_true, atol=1e-9)

    with pytest.raises(SpecificationError, match="transformation"):
        link(source, target, transformation="rotation")  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match=r"shape \(Q, 2\)"):
        link(source, target, nodes=np.linspace(-3, 3, 9))
    with pytest.raises(SpecificationError, match="weights"):
        link(source, target, nodes=nodes, weights=np.ones(5))
    one_factor = fit(
        Model.from_items({"c1": TwoPL(), "c2": TwoPL(), "c3": TwoPL()}),
        pd.DataFrame(np.random.default_rng(0).integers(0, 2, (80, 3)), columns=["c1", "c2", "c3"]),
        se=None,
    )
    with pytest.raises(SpecificationError, match="same number of factors"):
        link(one_factor, target)
    # transformation= has no effect on unidimensional models (the three families coincide).
    same = link(one_factor, one_factor, transformation="oblique")
    assert isinstance(same.A, float) and same.transformation == "diagonal"
    assert pytest.approx(1.0, abs=1e-10) == same.A
