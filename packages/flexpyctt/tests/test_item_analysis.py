"""Classical item analysis (DESIGN.md §16.21) against psych, ltm, polycor and base R (V46)."""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest

from flexpyctt import (
    coefficient_alpha,
    coefficient_omega,
    distractor_analysis,
    item_analysis,
    score_responses,
)
from flexpyctt.errors import (
    FlexpycttWarning,
    HeywoodWarning,
    NotPositiveDefiniteWarning,
    SpecificationError,
)
from reference_data import load_reference

REF = load_reference("v46_item_analysis")
CASES = {case["name"]: case for case in REF["cases"]}
MC_CASES = ["mc_complete", "mc_missing"]
COMPLETE = {"mc_complete", "poly_complete"}

# Closed forms agree to rounding; psych's minres stops earlier than flexpyctt's optimizer.
CLOSED_FORM = 1e-12
OMEGA = 1e-6
LOADINGS = 1e-5


def _frame(rows: list[list[Any]], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [[np.nan if v is None else v for v in row] for row in rows], columns=columns
    )


def _scores(case: dict[str, Any]) -> pd.DataFrame:
    return _frame(case["scores"], case["items"])


def _close(actual: object, expected: object, atol: float) -> None:
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=0.0,
        atol=atol,
    )


# --------------------------------------------------------------------------------------------
# R references
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("missing", ["pairwise", "listwise"])
def test_alpha_matches_psych(name: str, missing: str) -> None:
    case = CASES[name]
    ref = case[missing]
    result = coefficient_alpha(_scores(case), missing=missing)  # type: ignore[arg-type]
    _close(result.alpha, ref["raw_alpha"], CLOSED_FORM)
    _close(result.standardized_alpha, ref["std_alpha"], CLOSED_FORM)
    _close(result.if_deleted["alpha"], ref["drop_raw_alpha"], CLOSED_FORM)
    _close(result.if_deleted["standardized_alpha"], ref["drop_std_alpha"], CLOSED_FORM)
    _close(result.confidence_interval, [ref["feldt_lower"], ref["feldt_upper"]], CLOSED_FORM)
    assert result.n_obs == ref["n_obs"]
    assert result.n_items == len(case["items"])
    if missing == "listwise":
        assert result.n_obs == case["n_listwise"]


@pytest.mark.parametrize("name", list(CASES))
def test_alpha_matches_ltm(name: str) -> None:
    """ltm's raw alpha mixes complete-row and available-case variances, so complete data only."""
    case = CASES[name]
    result = coefficient_alpha(_scores(case), missing="listwise")
    _close(result.standardized_alpha, case["ltm_std_alpha"], CLOSED_FORM)
    if name in COMPLETE:
        _close(result.alpha, case["ltm_alpha"], CLOSED_FORM)
    else:
        assert abs(result.alpha - case["ltm_alpha"]) > 1e-4


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("missing", ["pairwise", "listwise"])
def test_item_statistics_match_psych(name: str, missing: str) -> None:
    case = CASES[name]
    ref = case[missing]
    table = item_analysis(_scores(case), missing=missing)  # type: ignore[arg-type]
    assert table["item"].tolist() == case["items"]
    np.testing.assert_array_equal(table["n"], ref["n"])
    _close(table["mean"], ref["mean"], CLOSED_FORM)
    _close(table["sd"], ref["sd"], CLOSED_FORM)
    _close(table["item_rest"], ref["r_drop"], CLOSED_FORM)
    _close(table["item_total"], ref["c_item_total"], CLOSED_FORM)
    _close(table["alpha_if_deleted"], ref["drop_raw_alpha"], CLOSED_FORM)
    if name in COMPLETE or missing == "listwise":
        _close(table["item_total"], ref["raw_r"], CLOSED_FORM)
    else:
        # psych's raw.r uses rowSums(na.rm = TRUE) as the total (DESIGN.md §16.21).
        assert np.max(np.abs(table["item_total"] - ref["raw_r"])) > 1e-3
    values = _scores(case).to_numpy(dtype=np.float64)
    if missing == "listwise":
        values = values[~np.isnan(values).any(axis=1)]
    _close(table["difficulty"], np.nanmean(values, axis=0) / np.nanmax(values, axis=0), 1e-15)


@pytest.mark.parametrize("name", sorted(COMPLETE))
def test_point_biserial_and_polyserial_match_r(name: str) -> None:
    case = CASES[name]
    ref = case["polyserial"]
    scores = _scores(case)
    pearson = item_analysis(scores)
    poly = item_analysis(scores, correlation="polyserial")
    _close(pearson["item_total"], ref["point_biserial_total"], CLOSED_FORM)
    _close(pearson["item_rest"], ref["point_biserial_rest"], CLOSED_FORM)
    _close(poly["item_total"], ref["polycor_total"], CLOSED_FORM)
    _close(poly["item_rest"], ref["polycor_rest"], CLOSED_FORM)
    _close(poly["item_total"], ref["psych_total"], CLOSED_FORM)
    _close(poly["alpha_if_deleted"], pearson["alpha_if_deleted"], 0.0)
    if "psych_biserial_total" in ref:
        n = len(scores)
        _close(pearson["item_total"], ref["ltm_point_biserial_total"], CLOSED_FORM)
        # psych::biserial divides by the n - 1 SD of the total while using p and q.
        _close(poly["item_total"] * np.sqrt((n - 1) / n), ref["psych_biserial_total"], CLOSED_FORM)


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("missing", ["pairwise", "listwise"])
def test_polyserial_factor(name: str, missing: str) -> None:
    """With missing data the two-step factor uses each item's observed scores (R formula)."""
    case = CASES[name]
    ref = case["polyserial" if missing == "pairwise" else "polyserial_listwise"]
    scores = _scores(case)
    poly = item_analysis(scores, correlation="polyserial", missing=missing)  # type: ignore[arg-type]
    pearson = item_analysis(scores, missing=missing)  # type: ignore[arg-type]
    _close(poly["item_total"] / pearson["item_total"], ref["factor"], 1e-12)
    _close(poly["item_rest"] / pearson["item_rest"], ref["factor"], 1e-12)


@pytest.mark.parametrize("name", list(CASES))
def test_omega_matches_psych(name: str) -> None:
    case = CASES[name]
    scores = _scores(case)
    kinds: tuple[Literal["pearson", "polychoric"], ...] = ("pearson", "polychoric")
    for kind in kinds:
        ref = case[f"omega_{kind}"]
        result = coefficient_omega(scores, correlation=kind)
        corr_tol = CLOSED_FORM if kind == "pearson" else 1e-6  # lavaan's polychoric: V24
        _close(result.correlation, ref["correlation"], corr_tol)
        _close(result.omega, ref["omega_tot"], OMEGA)
        _close(ref["omega_from_fa"], ref["omega_tot"], 1e-12)  # psych's omega.tot is this formula
        _close(result.loadings, ref["fa_loadings"], LOADINGS)
        _close(result.uniquenesses, 1.0 - result.loadings**2, 0.0)
        assert result.reversed_items == ()


def test_omega_reverses_negative_items_as_psych() -> None:
    ref = REF["reversed_item"]
    scores = _frame(ref["scores"], ref["items"])
    with pytest.warns(FlexpycttWarning, match="reversed"):
        result = coefficient_omega(scores)
    _close(result.correlation, ref["correlation"], CLOSED_FORM)
    _close(result.omega, ref["omega_tot"], OMEGA)
    _close(result.loadings, ref["fa_loadings"], LOADINGS)
    reversed_items = tuple(n for n, k in zip(ref["items"], ref["key"], strict=True) if k < 0)
    assert result.reversed_items == reversed_items == ("y2",)
    # psych's flip = FALSE (the total as scored) is the formula without reversing.
    _close(ref["omega_from_fa"], ref["omega_tot_unflipped"], 1e-12)
    assert ref["omega_tot"] - ref["omega_tot_unflipped"] > 0.1
    # Reverse-scoring the item by hand gives the same omega with nothing reversed.
    rescored = scores.assign(y2=4 - scores["y2"])
    by_hand = coefficient_omega(rescored)
    assert by_hand.reversed_items == ()
    _close(by_hand.omega, result.omega, 1e-9)


def test_omega_three_items() -> None:
    ref = REF["three_items"]
    scores = pd.DataFrame(ref["scores"], dtype=np.float64)
    result = coefficient_omega(scores)
    _close(result.omega, ref["omega_pearson"]["omega_tot"], OMEGA)
    _close(result.loadings, ref["omega_pearson"]["fa_loadings"], LOADINGS)


@pytest.mark.parametrize("name", MC_CASES)
def test_scoring_matches_psych(name: str) -> None:
    case = CASES[name]
    raw = _frame(case["raw"], case["items"])
    expected = _frame(case["keyed_scores"], case["items"])
    kept = score_responses(raw, key=case["key"], missing="keep")
    np.testing.assert_array_equal(kept.to_numpy(), expected.to_numpy())
    zero = score_responses(raw, key=case["key"])
    assert (zero.dtypes == np.int64).all()
    np.testing.assert_array_equal(zero.to_numpy(), expected.fillna(0).to_numpy())
    # The reliability statistics were computed on psych's scores.
    np.testing.assert_array_equal(kept.to_numpy(), _scores(case).to_numpy())


@pytest.mark.parametrize("name", MC_CASES)
def test_option_statistics_match_r(name: str) -> None:
    case = CASES[name]
    ref = case["options"]
    n_groups = ref["n_groups"]
    raw = _frame(case["raw"], case["items"])
    table = distractor_analysis(raw, key=case["key"], n_groups=n_groups)
    rows = pd.DataFrame(ref["rows"])
    assert table.columns.tolist() == [
        "item",
        "option",
        "is_key",
        "n",
        "proportion",
        "point_biserial",
        "mean_score",
        *[f"proportion_group_{g + 1}" for g in range(n_groups)],
    ]
    assert table["item"].tolist() == rows["item"].tolist()
    np.testing.assert_array_equal(table["option"].astype(float), rows["option"])
    np.testing.assert_array_equal(table["is_key"], rows["is_key"])
    np.testing.assert_array_equal(table["n"], rows["n"])
    for column in ("proportion", "point_biserial", "mean_score"):
        _close(table[column], rows[column].astype(float), CLOSED_FORM)
    groups = np.array(rows["group_proportions"].tolist(), dtype=np.float64)
    _close(table[[f"proportion_group_{g + 1}" for g in range(n_groups)]], groups, CLOSED_FORM)
    # Option proportions are psych's responseFrequency (of the respondents who answered).
    freq = ref_frequency(case)
    expected = [
        freq.loc[item, str(int(option))]
        for item, option in zip(table["item"], table["option"], strict=True)
    ]
    _close(table["proportion"], expected, CLOSED_FORM)


def ref_frequency(case: dict[str, Any]) -> pd.DataFrame:
    rf = case["response_frequency"]
    return _frame(rf["values"], rf["options"]).set_axis(case["items"])


# --------------------------------------------------------------------------------------------
# Behaviour
# --------------------------------------------------------------------------------------------


def _mc_data() -> tuple[pd.DataFrame, dict[str, str]]:
    rng = np.random.default_rng(3)
    theta = rng.normal(size=400)
    columns = {}
    for j in range(8):
        right = rng.random(400) < 1.0 / (1.0 + np.exp(-(1.5 * theta - (j - 4) / 3)))
        columns[f"q{j}"] = np.where(right, "B", rng.choice(["A", "C", "D"], 400)).astype(object)
    raw = pd.DataFrame(columns)
    raw.iloc[[0, 5, 9], 2] = None
    return raw, {name: "B" for name in raw.columns}


def test_scoring_with_string_options_and_sequence_key() -> None:
    raw, key = _mc_data()
    kept = score_responses(raw, key=key, missing="keep")
    assert kept.iloc[[0, 5, 9], 2].isna().all()
    assert all(dtype == np.float64 for dtype in kept.dtypes)
    by_position = score_responses(raw.to_numpy(), key=["B"] * raw.shape[1])
    np.testing.assert_array_equal(by_position.to_numpy(), score_responses(raw, key=key).to_numpy())
    assert by_position.columns.tolist() == [f"item{j + 1}" for j in range(raw.shape[1])]
    series = score_responses(raw, key=pd.Series(key))
    pd.testing.assert_frame_equal(series, score_responses(raw, key=key))


def test_scoring_errors_and_wrong_codes() -> None:
    raw, key = _mc_data()
    with pytest.raises(SpecificationError, match="no entry"):
        score_responses(raw, key={"q0": "B"})
    with pytest.raises(SpecificationError, match="not in data"):
        score_responses(raw, key={**key, "zz": "A"})
    with pytest.raises(SpecificationError, match="one entry per item"):
        score_responses(raw, key=["B", "B"])
    with pytest.raises(SpecificationError, match="missing"):
        score_responses(raw, key={**key, "q0": None})
    with pytest.raises(SpecificationError, match="missing must be"):
        score_responses(raw, key=key, missing="omit")  # type: ignore[arg-type]
    numbers = raw.replace({"A": 1, "B": 2, "C": 3, "D": 4})
    with pytest.warns(FlexpycttWarning, match="keyed option"):
        score_responses(numbers, key=key)


def test_kr20_and_missing_rules_agree_on_complete_data() -> None:
    raw, key = _mc_data()
    scores = score_responses(raw, key=key)
    pairwise = coefficient_alpha(scores)
    listwise = coefficient_alpha(scores, missing="listwise")
    assert pairwise.alpha == pytest.approx(listwise.alpha, abs=1e-14)
    p = scores.mean().to_numpy()
    k = scores.shape[1]
    kr20 = k / (k - 1) * (1 - np.sum(p * (1 - p)) / scores.sum(axis=1).var(ddof=0))
    assert pairwise.alpha == pytest.approx(kr20, abs=1e-12)
    pd.testing.assert_frame_equal(item_analysis(scores), item_analysis(scores, missing="listwise"))
    table = item_analysis(scores)
    _close(table["difficulty"], p, 1e-15)
    assert (table["n_categories"] == 2).all()


def test_listwise_drops_incomplete_rows() -> None:
    raw, key = _mc_data()
    scores = score_responses(raw, key=key, missing="keep")
    listwise = coefficient_alpha(scores, missing="listwise")
    complete = coefficient_alpha(scores.dropna())
    assert listwise.alpha == pytest.approx(complete.alpha, abs=1e-15)
    assert listwise.n_obs == len(scores) - 3
    assert coefficient_alpha(scores).n_obs == len(scores)


def test_confidence_level_and_two_items() -> None:
    raw, key = _mc_data()
    scores = score_responses(raw, key=key)
    wide = coefficient_alpha(scores, confidence_level=0.99).confidence_interval
    narrow = coefficient_alpha(scores, confidence_level=0.90).confidence_interval
    assert wide[0] < narrow[0] < narrow[1] < wide[1]
    two = coefficient_alpha(scores.iloc[:, :2])
    assert two.if_deleted.isna().all().all()
    assert item_analysis(scores.iloc[:, :2])["alpha_if_deleted"].isna().all()
    with pytest.raises(SpecificationError, match="confidence_level"):
        coefficient_alpha(scores, confidence_level=1.0)


def test_item_order_invariance() -> None:
    raw, key = _mc_data()
    scores = score_responses(raw, key=key, missing="keep")
    order = list(reversed(scores.columns))
    a = item_analysis(scores).set_index("item")
    b = item_analysis(scores[order]).set_index("item").loc[a.index]
    pd.testing.assert_frame_equal(a, b, atol=1e-14, rtol=0.0)
    assert coefficient_omega(scores).omega == pytest.approx(
        coefficient_omega(scores[order]).omega, abs=1e-9
    )


def test_input_errors() -> None:
    raw, key = _mc_data()
    scores = score_responses(raw, key=key)
    with pytest.raises(SpecificationError, match="at least 2 items"):
        coefficient_alpha(scores.iloc[:, :1])
    with pytest.raises(SpecificationError, match="at least 3 items"):
        coefficient_omega(scores.iloc[:, :2])
    with pytest.raises(SpecificationError, match="distinct"):
        item_analysis(scores.assign(q0=1))
    with pytest.raises(SpecificationError, match="at least 0"):
        item_analysis(scores - 1)
    with pytest.raises(SpecificationError, match="numeric"):
        coefficient_alpha(raw)
    with pytest.raises(SpecificationError, match="missing must"):
        item_analysis(scores, missing="complete")  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match="correlation must"):
        item_analysis(scores, correlation="spearman")  # type: ignore[arg-type]
    with pytest.raises(SpecificationError, match="correlation must"):
        coefficient_omega(scores, correlation="polyserial")  # type: ignore[arg-type]
    sparse = scores.astype(float)
    sparse.iloc[2:, 0] = np.nan
    sparse.iloc[:2, 1] = np.nan
    with pytest.raises(SpecificationError, match="both scores observed"):
        coefficient_alpha(sparse)


def test_polyserial_outside_unit_interval_is_clipped() -> None:
    # A two-item "test" of identical binary items: the biserial with the total exceeds 1.
    x = pd.DataFrame({"a": [0, 0, 0, 1, 1, 1, 1, 1], "b": [0, 0, 0, 1, 1, 1, 1, 1]})
    with pytest.warns(HeywoodWarning, match="outside"):
        table = item_analysis(x, correlation="polyserial")
    assert (table["item_total"] == 1.0).all()


def test_omega_smooths_a_pairwise_matrix() -> None:
    # Each pair of a, b, c is observed on its own block of rows: r_ab = r_bc = 1 but r_ac = -1.
    rng = np.random.default_rng(5)
    z = rng.normal(size=(3, 40))
    nan = np.full(40, np.nan)
    x = pd.DataFrame(
        {
            "a": np.r_[z[0], nan, z[2]],
            "b": np.r_[z[0], z[1], nan],
            "c": np.r_[nan, z[1], -z[2]],
            "d": rng.normal(size=120),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", HeywoodWarning)
        warnings.filterwarnings("ignore", message=".*load negatively")
        with pytest.warns(NotPositiveDefiniteWarning):
            result = coefficient_omega(x)
    assert np.linalg.eigvalsh(result.correlation.to_numpy()).min() > 0


def test_distractor_table() -> None:
    raw, key = _mc_data()
    raw["q7"] = raw["q7"].replace({"B": "C"})  # nobody chooses q7's key
    with pytest.warns(FlexpycttWarning, match="keyed option"):
        table = distractor_analysis(raw, key=key, n_groups=4)
    q7 = table[table["item"] == "q7"].set_index("option")
    assert q7.loc["B", "n"] == 0 and q7.loc["B", "is_key"]
    assert q7.loc[["B"], ["point_biserial", "mean_score"]].isna().all().all()
    keyed = table[table["is_key"] & (table["item"] != "q7")]
    assert (keyed["point_biserial"] > 0).all()
    assert (keyed["proportion_group_4"] > keyed["proportion_group_1"]).all()
    per_item = table.groupby("item")["proportion"].sum()
    _close(per_item, np.ones(len(per_item)), 1e-12)
    q2 = table[table["item"] == "q2"]
    assert q2["n"].sum() == len(raw) - 3
    with pytest.raises(SpecificationError, match="n_groups"):
        distractor_analysis(raw, key=key, n_groups=1)
    with warnings.catch_warnings(), pytest.raises(SpecificationError, match="not distinct"):
        warnings.simplefilter("ignore", FlexpycttWarning)
        distractor_analysis(raw.iloc[:5], key=key, n_groups=4)
