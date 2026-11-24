"""V21: multiple-group 2D MIRT vs R mirt, case ``mg2d_cross``, plain EM (one fit, ~2-3 min).

Checks and tolerances: ``tests/_mirt_multigroup.py``.
"""

import pytest

from tests import _mirt_multigroup as mg

pytestmark = pytest.mark.slow
CASE = "mg2d_cross"


def test_loglik_and_estimates() -> None:
    mg.check_loglik_and_estimates(CASE, squarem=False)


def test_standard_errors() -> None:
    mg.check_standard_errors(CASE)


def test_focal_covariance_se() -> None:
    mg.check_focal_covariance_se(CASE)


def test_mirt_stops_short_on_one_ridge() -> None:
    mg.check_mirt_stops_short_on_one_ridge(CASE)


def test_scoring_and_json_round_trip() -> None:
    mg.check_scoring_and_json_round_trip(CASE)
