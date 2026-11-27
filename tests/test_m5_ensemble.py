"""The dimensionality() ensemble (DESIGN.md §16.8, §16.12).

Each row reuses a validated method (V24-V30), so these tests check the assembly: the same keys
(as NumPy generators for the flexpyctt methods) and inputs reach each method, the two consensus
lines are computed as documented, and missing methods are reported rather than dropped.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
from functools import cache
from typing import Any

import jax
import numpy as np
import pandas as pd
import pytest

from flexpyctt import hull, polychoric
from flexpyirt import (
    BAEM,
    DimensionalityResult,
    FitResult,
    Model,
    TwoPL,
    comparison_data,
    dimensionality,
    dimtest,
    empirical_kaiser,
    exploratory_detect,
    fit,
    model_parallel_analysis,
    parallel_analysis,
    velicer_map,
)
from flexpyirt.dimensionality import _rng_from_key
from flexpyirt.errors import SpecificationError
from tests._reference import CTT_REFERENCE_DIR, REFERENCE_DIR


@cache
def _lsat7() -> pd.DataFrame:
    raw = json.loads((REFERENCE_DIR / "v26_local_dependence.json").read_text())
    case = next(c for c in raw["cases"] if c["name"] == "lsat7_2pl")
    return pd.DataFrame(case["data"], columns=case["items"], dtype=float)


@cache
def _fit() -> FitResult:
    data = _lsat7()
    return fit(Model.uniform(TwoPL(), items=list(data.columns)), data, method=BAEM(tol=1e-8))


@cache
def _ensemble() -> DimensionalityResult:
    return dimensionality(_lsat7(), key=jax.random.key(7), fit=_fit(), n_replications=20)


def test_ensemble_reuses_the_individual_methods() -> None:
    res = _ensemble()
    data = _lsat7()
    key_a, key_b, key_c, _, key_e, key_f = jax.random.split(jax.random.key(7), 6)
    pa = parallel_analysis(data, rng=_rng_from_key(key_a), n_replications=20)
    pc = polychoric(data)
    assert list(res.number_of_dimensions["n_factors"]) == [
        pa.n_factors,
        velicer_map(pc).n_factors,
        empirical_kaiser(pc).n_factors,
        comparison_data(data, rng=_rng_from_key(key_e)).n_factors,
    ]
    mpa = model_parallel_analysis(_fit(), data, key=key_b, n_replications=20)
    row = res.unidimensionality.set_index("method").loc["model-based parallel analysis"]
    assert row["statistic"] == pytest.approx(mpa.p_value)
    assert row["supports_unidimensionality"] == (mpa.p_value >= 0.05)
    ld = _fit().local_dependence(data)
    q3 = res.unidimensionality.set_index("method").loc["largest Q3*"]
    assert q3["statistic"] == pytest.approx(ld.max_q3_star)
    m2 = res.unidimensionality.set_index("method").loc["M2 RMSEA"]
    assert m2["statistic"] == pytest.approx(float(_fit().m2(data)["rmsea"].iloc[0]))
    xd = exploratory_detect(data, rng=_rng_from_key(key_c))
    det = res.unidimensionality.set_index("method").loc["DETECT (exploratory, validation half)"]
    best = xd.table[xd.table["n_clusters"] == xd.n_clusters].to_dict("records")[0]
    assert det["statistic"] == pytest.approx(float(best["DETECT_validation"]))
    assert det["supports_unidimensionality"] == (float(best["DETECT_validation"]) < 0.2)
    dt = dimtest(data, rng=_rng_from_key(key_f))
    row_dt = res.unidimensionality.set_index("method").loc["DIMTEST"]
    assert row_dt["statistic"] == pytest.approx(dt.p_value)
    assert set(res.results) == {
        "comparison_data",
        "dimtest",
        "exploratory_detect",
        "hull",
        "polychoric",
        "parallel_analysis",
        "velicer_map",
        "empirical_kaiser",
        "model_parallel_analysis",
        "local_dependence",
        "m2",
    }


def test_ensemble_includes_hull_with_six_or_more_items() -> None:
    raw = json.loads((CTT_REFERENCE_DIR / "v27_detect.json").read_text(encoding="utf-8"))
    case = raw["cases"][0]
    data = pd.DataFrame(case["data"], columns=case["items"], dtype=float)
    res = dimensionality(data, key=jax.random.key(4), n_replications=10)
    row = res.number_of_dimensions.set_index("method").loc["Hull (CAF, ULS)"]
    expected = hull(polychoric(data), rng=_rng_from_key(jax.random.split(jax.random.key(4), 6)[3]))
    assert row["n_factors"] == expected.n_factors == 2
    assert res.results["hull"] is not None


def test_rng_from_key_is_deterministic_and_accepts_both_key_kinds() -> None:
    first = _rng_from_key(jax.random.key(3)).integers(0, 10**9, size=4)
    np.testing.assert_array_equal(
        first, _rng_from_key(jax.random.key(3)).integers(0, 10**9, size=4)
    )
    assert not np.array_equal(first, _rng_from_key(jax.random.key(4)).integers(0, 10**9, size=4))
    legacy = _rng_from_key(jax.random.PRNGKey(3)).integers(0, 10**9, size=4)
    np.testing.assert_array_equal(first, legacy)


def test_moved_tools_are_reexported_from_flexpyctt() -> None:
    """flexpyirt re-exports flexpyctt's functions and classes; polychoric stays the JAX version."""
    import flexpyctt
    import flexpyirt

    moved = [n for n in flexpyctt.__all__ if n not in ("__version__", "errors", "polychoric")]
    for name in moved:
        assert getattr(flexpyirt, name) is getattr(flexpyctt, name), name
    assert flexpyirt.polychoric is not flexpyctt.polychoric
    assert flexpyirt.errors.SpecificationError is flexpyctt.errors.SpecificationError
    assert flexpyirt.errors.FlexpyirtWarning is flexpyctt.errors.FlexpycttWarning
    assert issubclass(flexpyirt.errors.IdentificationError, flexpyctt.errors.FlexpycttError)


def test_ensemble_consensus_lines() -> None:
    res = _ensemble()
    k = res.number_of_dimensions["n_factors"].to_numpy()
    values, counts = np.unique(k, return_counts=True)
    assert res.n_factors_mode == tuple(int(v) for v in values[counts == np.max(counts)])
    assert res.n_agreeing == np.max(counts)
    assert res.n_factors_median == float(np.median(k))
    flags = list(res.unidimensionality["supports_unidimensionality"])
    assert res.n_unidimensionality_methods == 5
    assert res.n_supporting == sum(bool(f) for f in flags)
    text = res.summary()
    assert "question A" in text and "question B" in text
    assert str(res) == text
    for method in res.number_of_dimensions["method"]:
        assert method in text


def test_ensemble_reports_ties_without_breaking_them(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("flexpyirt.dimensionality")
    real_map, real_ekc, real_cd = (
        module.velicer_map,
        module.empirical_kaiser,
        module.comparison_data,
    )

    def map_two(*args: Any, **kwargs: Any) -> Any:
        return dataclasses.replace(real_map(*args, **kwargs), n_factors=2)

    def ekc_three(*args: Any, **kwargs: Any) -> Any:
        return dataclasses.replace(real_ekc(*args, **kwargs), n_factors=3)

    def cd_four(*args: Any, **kwargs: Any) -> Any:
        return dataclasses.replace(real_cd(*args, **kwargs), n_factors=4)

    monkeypatch.setattr(module, "velicer_map", map_two)
    monkeypatch.setattr(module, "empirical_kaiser", ekc_three)
    monkeypatch.setattr(module, "comparison_data", cd_four)
    res = dimensionality(_lsat7(), key=jax.random.key(0), n_replications=5)
    pa = int(res.number_of_dimensions["n_factors"].iloc[0])
    values, counts = np.unique([pa, 2, 3, 4], return_counts=True)
    assert res.n_factors_mode == tuple(int(v) for v in values[counts == np.max(counts)])
    assert res.n_agreeing == int(np.max(counts))
    assert len(res.n_factors_mode) >= 3


def test_ensemble_without_fit_has_only_the_detect_row() -> None:
    res = dimensionality(_lsat7(), key=jax.random.key(1), n_replications=5)
    assert list(res.unidimensionality["method"]) == [
        "DETECT (exploratory, validation half)",
        "DIMTEST",
    ]
    assert res.n_unidimensionality_methods == 2
    assert "question B" in res.summary()


def test_ensemble_with_missing_data_keeps_m2() -> None:
    data = _lsat7().copy()
    data.iloc[::25, 2] = np.nan
    res_fit = fit(Model.uniform(TwoPL(), items=list(data.columns)), data, se=None)
    res = dimensionality(data, key=jax.random.key(2), fit=res_fit, n_replications=5)
    row = res.unidimensionality.set_index("method").loc["M2 RMSEA"]
    assert isinstance(row["supports_unidimensionality"], bool)
    assert res.results["m2"] is not None
    assert row["statistic"] == pytest.approx(float(res_fit.m2(data)["rmsea"].iloc[0]))
    assert res.results["exploratory_detect"] is None
    assert res.results["comparison_data"] is None
    assert res.results["dimtest"] is None
    assert res.n_unidimensionality_methods == 3


def test_ensemble_m2_without_degrees_of_freedom_is_not_a_vote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Issue #145: a NaN RMSEA (M2 with df <= 0) was counted as a vote against unidimensionality.
    real_m2 = FitResult.m2

    def m2_without_df(self: FitResult, *args: Any, **kwargs: Any) -> pd.DataFrame:
        table = real_m2(self, *args, **kwargs).copy()
        table["df"] = 0
        table[["m2", "p", "rmsea", "rmsea_5", "rmsea_95", "tli", "cfi"]] = np.nan
        return table

    monkeypatch.setattr(FitResult, "m2", m2_without_df)
    res = dimensionality(_lsat7(), key=jax.random.key(3), fit=_fit(), n_replications=5)
    row = res.unidimensionality.set_index("method").loc["M2 RMSEA"]
    assert row["supports_unidimensionality"] is None
    assert "df = 0" in row["detail"]
    assert res.n_unidimensionality_methods == _ensemble().n_unidimensionality_methods - 1


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"alpha": 0.0}, "alpha"),
        ({"q3_star_threshold": 1.5}, "q3_star_threshold"),
        ({"rmsea_threshold": -0.1}, "rmsea_threshold"),
        ({"detect_threshold": 0.0}, "detect_threshold"),
    ],
)
def test_ensemble_invalid_thresholds(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(SpecificationError, match=match):
        dimensionality(_lsat7(), key=jax.random.key(0), **kwargs)
