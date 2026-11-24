"""The dimensionality() ensemble (DESIGN.md §16.8, §16.12).

Each row reuses a validated method (V24-V26), so these tests check the assembly: the same keys
and inputs reach each method, the two consensus lines are computed as documented, and missing
methods are reported rather than dropped.
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

from flexpyirt import (
    BAEM,
    DimensionalityResult,
    FitResult,
    Model,
    TwoPL,
    dimensionality,
    empirical_kaiser,
    fit,
    model_parallel_analysis,
    parallel_analysis,
    polychoric,
    velicer_map,
)
from flexpyirt.errors import SpecificationError
from tests._reference import REFERENCE_DIR


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
    key_a, key_b = jax.random.split(jax.random.key(7))
    pa = parallel_analysis(data, key=key_a, n_replications=20)
    pc = polychoric(data)
    assert list(res.number_of_dimensions["n_factors"]) == [
        pa.n_factors,
        velicer_map(pc).n_factors,
        empirical_kaiser(pc).n_factors,
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
    assert set(res.results) == {
        "polychoric",
        "parallel_analysis",
        "velicer_map",
        "empirical_kaiser",
        "model_parallel_analysis",
        "local_dependence",
        "m2",
    }


def test_ensemble_consensus_lines() -> None:
    res = _ensemble()
    k = res.number_of_dimensions["n_factors"].to_numpy()
    values, counts = np.unique(k, return_counts=True)
    assert res.n_factors_mode == tuple(int(v) for v in values[counts == np.max(counts)])
    assert res.n_agreeing == np.max(counts)
    assert res.n_factors_median == float(np.median(k))
    flags = list(res.unidimensionality["supports_unidimensionality"])
    assert res.n_unidimensionality_methods == 3
    assert res.n_supporting == sum(bool(f) for f in flags)
    text = res.summary()
    assert "question A" in text and "question B" in text
    assert str(res) == text
    for method in res.number_of_dimensions["method"]:
        assert method in text


def test_ensemble_reports_ties_without_breaking_them(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("flexpyirt.dimensionality")
    real_map, real_ekc = module.velicer_map, module.empirical_kaiser

    def map_two(*args: Any, **kwargs: Any) -> Any:
        return dataclasses.replace(real_map(*args, **kwargs), n_factors=2)

    def ekc_three(*args: Any, **kwargs: Any) -> Any:
        return dataclasses.replace(real_ekc(*args, **kwargs), n_factors=3)

    monkeypatch.setattr(module, "velicer_map", map_two)
    monkeypatch.setattr(module, "empirical_kaiser", ekc_three)
    res = dimensionality(_lsat7(), key=jax.random.key(0), n_replications=5)
    pa = int(res.number_of_dimensions["n_factors"].iloc[0])
    assert res.n_factors_mode == tuple(sorted({pa, 2, 3}))
    assert res.n_agreeing == (2 if pa in (2, 3) else 1)


def test_ensemble_without_fit_has_no_unidimensionality_rows() -> None:
    res = dimensionality(_lsat7(), key=jax.random.key(1), n_replications=5)
    assert res.unidimensionality.empty
    assert res.n_unidimensionality_methods == 0
    assert res.n_supporting == 0
    assert "question B" not in res.summary()


def test_ensemble_marks_m2_unavailable_with_missing_data() -> None:
    data = _lsat7().copy()
    data.iloc[::25, 2] = np.nan
    res_fit = fit(Model.uniform(TwoPL(), items=list(data.columns)), data, se=None)
    res = dimensionality(data, key=jax.random.key(2), fit=res_fit, n_replications=5)
    row = res.unidimensionality.set_index("method").loc["M2 RMSEA"]
    assert row["supports_unidimensionality"] is None
    assert "not available" in row["detail"]
    assert res.results["m2"] is None
    assert res.n_unidimensionality_methods == 2


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"alpha": 0.0}, "alpha"),
        ({"q3_star_threshold": 1.5}, "q3_star_threshold"),
        ({"rmsea_threshold": -0.1}, "rmsea_threshold"),
    ],
)
def test_ensemble_invalid_thresholds(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(SpecificationError, match=match):
        dimensionality(_lsat7(), key=jax.random.key(0), **kwargs)
