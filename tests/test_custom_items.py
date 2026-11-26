"""User-defined item models through the public ItemModel / ItemKernel contract (DESIGN.md §20).

Re-implementations of the 2PL and the GRM, written only against the public protocol, must give the
built-in results to machine precision: estimates, standard errors, display thresholds, scores and
the fit statistics that read the kernel. The kernels repeat the built-in arithmetic operation for
operation, so the compiled programs are the same and the results agree bit for bit; the tolerance
below only leaves room for floating-point reassociation by the compiler.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any, ClassVar, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from jax import Array

from flexpyirt import (
    BAEM,
    GRM,
    FitResult,
    Model,
    Quadrature,
    TwoPL,
    fit,
    item_information,
    link,
)
from flexpyirt.errors import SpecificationError
from flexpyirt.items import (
    _REGISTRY,
    ItemKernel,
    ItemModel,
    register_item_model,
)

EXACT: dict[str, Any] = {"rtol": 1e-12, "atol": 1e-12}


def _slope_names(n_factors: int) -> tuple[str, ...]:
    return ("a",) if n_factors == 1 else tuple(f"a{m}" for m in range(1, n_factors + 1))


def _linear(theta: Array, slopes: Array, n_factors: int) -> Array:
    # The contract promises nodes of shape (Q, D) on every code path (DESIGN.md §20).
    assert theta.ndim == 2 and theta.shape[1] == n_factors, theta.shape
    if n_factors == 1:
        return slopes[0] * theta[:, 0]
    return theta @ slopes


@dataclass(frozen=True)
class LogisticKernel:
    """2PL kernel written against the public protocol (no display thresholds)."""

    n_factors: int = 1
    n_categories: ClassVar[int] = 2

    def with_factors(self, n_factors: int) -> LogisticKernel:
        return LogisticKernel(n_factors)

    @property
    def param_names(self) -> tuple[str, ...]:
        return (*_slope_names(self.n_factors), "d")

    @property
    def slope_params(self) -> tuple[str, ...]:
        return _slope_names(self.n_factors)

    @property
    def intercept_params(self) -> tuple[str, ...]:
        return ("d",)

    def log_prob(self, theta: Array, params: Array) -> Array:
        z = _linear(theta, params[: self.n_factors], self.n_factors) + params[self.n_factors]
        return jnp.stack([jax.nn.log_sigmoid(-z), jax.nn.log_sigmoid(z)], axis=-1)


@dataclass(frozen=True)
class Logistic:
    name: ClassVar[str] = "Logistic"

    @property
    def kernel(self) -> LogisticKernel:
        return LogisticKernel()

    def default_free(self) -> tuple[bool, ...]:
        return (True, True)

    def start_values(self, item: np.ndarray, rest: np.ndarray) -> np.ndarray:
        return TwoPL().start_values(item, rest)

    def bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        return (-np.inf, -np.inf), (np.inf, np.inf)


def _log1mexp(x: Array) -> Array:
    near = x > -jnp.log(2.0)
    x_near = jnp.where(near, x, -1.0)
    x_far = jnp.where(near, -1.0, x)
    out = jnp.where(near, jnp.log(-jnp.expm1(x_near)), jnp.log1p(-jnp.exp(x_far)))
    return jnp.where(x < 0, out, jnp.nan)


@dataclass(frozen=True)
class CumulativeKernel:
    """Graded response kernel written against the public protocol, with display thresholds."""

    n_categories: int
    n_factors: int = 1

    def with_factors(self, n_factors: int) -> CumulativeKernel:
        return CumulativeKernel(self.n_categories, n_factors)

    @property
    def param_names(self) -> tuple[str, ...]:
        return (*_slope_names(self.n_factors), *(f"d{k}" for k in range(1, self.n_categories)))

    @property
    def slope_params(self) -> tuple[str, ...]:
        return _slope_names(self.n_factors)

    @property
    def intercept_params(self) -> tuple[str, ...]:
        return self.param_names[self.n_factors :]

    @property
    def irt_names(self) -> tuple[str, ...]:
        return tuple(f"b{k}" for k in range(1, self.n_categories)) if self.n_factors == 1 else ()

    @property
    def irt_dependency_indices(self) -> tuple[tuple[int, ...], ...]:
        if self.n_factors > 1:
            return ()
        return tuple((0, k) for k in range(1, self.n_categories))

    def irt_thresholds(self, params: Array) -> Array:
        return -params[1:] / params[0]

    def log_prob(self, theta: Array, params: Array) -> Array:
        d = self.n_factors
        z = _linear(theta, params[:d], d)[:, None] + params[None, d:]
        log_cum = jax.nn.log_sigmoid(z)
        first = jax.nn.log_sigmoid(-z[:, :1])
        last = log_cum[:, -1:]
        middle = log_cum[:, :-1] + _log1mexp(log_cum[:, 1:] - log_cum[:, :-1])
        return jnp.concatenate([first, middle, last], axis=1)


@dataclass(frozen=True)
class Cumulative:
    n_categories: int
    name: ClassVar[str] = "Cumulative"

    @property
    def kernel(self) -> CumulativeKernel:
        return CumulativeKernel(self.n_categories)

    def default_free(self) -> tuple[bool, ...]:
        return (True,) * self.n_categories

    def start_values(self, item: np.ndarray, rest: np.ndarray) -> np.ndarray:
        return GRM(self.n_categories).start_values(item, rest)

    def bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        return (-np.inf,) * self.n_categories, (np.inf,) * self.n_categories


def test_custom_classes_satisfy_the_protocols() -> None:
    kernel: ItemKernel = CumulativeKernel(4)
    model: ItemModel = Cumulative(4)
    assert hash(kernel) == hash(CumulativeKernel(4))
    assert model.kernel == kernel


@cache
def _data() -> pd.DataFrame:
    rng = np.random.default_rng(86)
    n = 400
    theta = rng.normal(size=n)
    a = rng.uniform(0.8, 2.0, 6)
    d = rng.normal(size=6)
    binary = rng.random((n, 6)) < 1.0 / (1.0 + np.exp(-(np.outer(theta, a) + d)))
    cuts = np.array([-1.0, 0.2, 1.3])
    cont = np.outer(theta, np.ones(4)) * 1.3 + rng.logistic(size=(n, 4))
    poly = np.searchsorted(cuts, cont)
    data = np.column_stack([binary, poly]).astype(float)
    data[rng.random(data.shape) < 0.03] = np.nan
    names = [f"x{j}" for j in range(1, 7)] + [f"s{j}" for j in range(1, 5)]
    return pd.DataFrame(data, columns=names)


def _spec(custom: bool) -> Model:
    df = _data()
    items: dict[str, Any] = {}
    for name in df.columns:
        if name.startswith("x"):
            items[name] = Logistic() if custom else TwoPL()
        else:
            items[name] = Cumulative(4) if custom else GRM(4)
    return Model.from_items(items)


@cache
def _fits() -> tuple[FitResult, FitResult]:
    df = _data()
    return fit(_spec(False), df, se="oakes"), fit(_spec(True), df, se="oakes")


def test_estimates_and_standard_errors_match_builtins() -> None:
    builtin, custom = _fits()
    assert custom.loglik == pytest.approx(builtin.loglik, rel=1e-14)
    assert len(custom.history) == len(builtin.history)
    pb, pc = builtin.parameters, custom.parameters
    assert list(pb["param"]) == list(pc["param"])
    np.testing.assert_allclose(pc["estimate"], pb["estimate"], **EXACT)
    np.testing.assert_allclose(pc["se"], pb["se"], **EXACT)
    assert set(pc.loc[pc["item"].str.match(r"[xs]\d"), "model"]) == {"Logistic", "Cumulative"}


def test_display_thresholds_come_from_the_optional_members() -> None:
    builtin, custom = _fits()
    ib, ic = builtin.items.set_index("item"), custom.items.set_index("item")
    cols = ["b1", "b2", "b3", "b1_se", "b2_se", "b3_se"]
    grm_items = [f"s{j}" for j in range(1, 5)]
    np.testing.assert_allclose(ic.loc[grm_items, cols], ib.loc[grm_items, cols], **EXACT)
    # The custom 2PL kernel has no irt_* members, so its items have no display "b".
    assert "b" not in ic.columns
    assert ic.loc["x1", ["b1", "b2", "b3"]].isna().all()


@pytest.mark.parametrize(
    "method",
    [
        "EAP",
        "MAP",
        pytest.param("ML", marks=pytest.mark.slow),
        pytest.param("WLE", marks=pytest.mark.slow),
    ],
)
def test_scores_match_builtins(method: Literal["EAP", "MAP", "ML", "WLE"]) -> None:
    builtin, custom = _fits()
    df = _data()
    sb = builtin.score(df, method=method)
    sc = custom.score(df, method=method)
    np.testing.assert_allclose(sc.to_numpy(dtype=float), sb.to_numpy(dtype=float), **EXACT)


@pytest.mark.slow
def test_item_fit_m2_and_information_match_builtins() -> None:
    builtin, custom = _fits()
    df = _data()
    complete = df.dropna()
    fb, fc = builtin.item_fit(complete), custom.item_fit(complete)
    np.testing.assert_allclose(
        fc[["s_x2", "df", "p"]].to_numpy(float), fb[["s_x2", "df", "p"]].to_numpy(float), **EXACT
    )
    mb, mc = builtin.m2(complete), custom.m2(complete)
    np.testing.assert_allclose(mc.to_numpy(float), mb.to_numpy(float), **EXACT)
    theta = np.linspace(-3.0, 3.0, 7)
    np.testing.assert_allclose(
        item_information(custom, theta=theta).to_numpy(float),
        item_information(builtin, theta=theta).to_numpy(float),
        **EXACT,
    )


@pytest.mark.slow
def test_multidimensional_custom_model_matches_builtin() -> None:
    df = _data()
    factors = {"F1": [f"x{j}" for j in range(1, 7)], "F2": [f"s{j}" for j in range(1, 5)]}
    options: dict[str, Any] = {
        "method": BAEM(tol=1e-8),
        "quadrature": Quadrature(n_points=15, lower=-5.0, upper=5.0),
    }
    builtin = fit(_spec(False).with_factors(factors), df, **options)
    custom = fit(_spec(True).with_factors(factors), df, **options)
    # Starting values of the custom non-slope parameters come from start_values(), not from the
    # built-in effective-slope rule, so the two runs reach the same maximum along different paths.
    assert custom.loglik == pytest.approx(builtin.loglik, abs=1e-8)
    np.testing.assert_allclose(
        custom.parameters["estimate"], builtin.parameters["estimate"], atol=1e-5
    )


@pytest.fixture
def registry() -> Any:
    saved = dict(_REGISTRY)
    yield _REGISTRY
    _REGISTRY.clear()
    _REGISTRY.update(saved)


def test_json_needs_registration(registry: Any) -> None:
    _, custom = _fits()
    with pytest.raises(SpecificationError, match="register_item_model"):
        custom.to_json()
    register_item_model(Logistic)
    register_item_model(Cumulative)
    text = custom.to_json()
    restored = FitResult.from_json(text)
    assert restored._model == custom._model
    pd.testing.assert_frame_equal(restored.parameters, custom.parameters)
    registry.clear()
    with pytest.raises(SpecificationError, match="must be registered"):
        FitResult.from_json(text)


def test_registration_with_custom_encoding(registry: Any) -> None:
    class Plain:
        name: ClassVar[str] = "PlainLogistic"

        def __init__(self, scale: float = 1.0) -> None:
            self.scale = scale

        def __eq__(self, other: object) -> bool:
            return isinstance(other, Plain) and other.scale == self.scale

        def __hash__(self) -> int:
            return hash(self.scale)

        kernel = LogisticKernel()

        def default_free(self) -> tuple[bool, ...]:
            return (True, True)

        def start_values(self, item: np.ndarray, rest: np.ndarray) -> np.ndarray:
            return TwoPL().start_values(item, rest)

        def bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
            return (-np.inf, -np.inf), (np.inf, np.inf)

    with pytest.raises(SpecificationError, match="not a dataclass"):
        register_item_model(Plain)
    register_item_model(
        Plain,
        to_dict=lambda m: {"scale": m.scale},
        from_dict=lambda fields: Plain(**fields),
    )
    df = _data()[["x1", "x2", "x3", "x4"]]
    res = fit(Model.uniform(Plain(2.0), items=df.columns), df)
    restored = FitResult.from_json(res.to_json())
    assert restored._model.items[0][1] == Plain(2.0)


def test_registration_errors(registry: Any) -> None:
    with pytest.raises(SpecificationError, match="built-in"):
        register_item_model(TwoPL)

    @dataclass(frozen=True)
    class Other(Logistic):
        pass

    register_item_model(Logistic)
    register_item_model(Logistic)  # idempotent
    with pytest.raises(SpecificationError, match="already registered"):
        register_item_model(Other)
    with pytest.raises(SpecificationError, match="both"):
        register_item_model(Cumulative, to_dict=lambda m: {})


def test_contract_is_checked_when_building_a_model() -> None:
    @dataclass(frozen=True)
    class SlopeLast(Logistic):
        name: ClassVar[str] = "SlopeLast"

        @property
        def kernel(self) -> Any:
            @dataclass(frozen=True)
            class Swapped(LogisticKernel):
                @property
                def param_names(self) -> tuple[str, ...]:
                    return ("d", "a")

            return Swapped()

    with pytest.raises(SpecificationError, match="exactly one slope, first"):
        Model.uniform(SlopeLast(), items=["i1", "i2"])

    @dataclass(frozen=True)
    class ShortBounds(Logistic):
        name: ClassVar[str] = "ShortBounds"

        def bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
            return (-np.inf,), (np.inf,)

    with pytest.raises(SpecificationError, match="bounds"):
        Model.uniform(ShortBounds(), items=["i1", "i2"])

    @dataclass(frozen=True)
    class Shadow(Logistic):
        name: ClassVar[str] = "2PL"

    with pytest.raises(SpecificationError, match="built-in name"):
        Model.uniform(Shadow(), items=["i1", "i2"])


def test_unsupported_features_raise() -> None:
    builtin, custom = _fits()
    with pytest.raises(SpecificationError, match="user-defined"):
        link(custom, builtin)
    bank = pd.DataFrame({"item": ["x1"], "a": [1.0], "b": [0.0]})
    with pytest.raises(SpecificationError, match="user-defined"):
        _spec(True).fix_from(bank, items=["x1"], parameterization="irt")
