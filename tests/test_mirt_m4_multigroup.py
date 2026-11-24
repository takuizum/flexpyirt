"""V21: multiple-group confirmatory 2D MIRT vs R mirt — checks that need no fit.

The checks and their tolerances are in ``tests/_mirt_multigroup.py``; each fit has its own module
(``test_mirt_m4_multigroup_*.py``) so the fits run on different workers and CI shards.
"""

from tests import _mirt_multigroup as mg


def test_unlinked_focal_factor_is_identification_error() -> None:
    mg.check_unlinked_focal_factor_is_identification_error()
