"""V21: multiple-group 2D MIRT, ``mg2d_simple``: EM is monotone and Oakes SE equal Hessian SE.

Needs its own plain-EM fit and a Hessian fit (~4 min together), so it is a module of its own.
Checks: ``tests/_mirt_multigroup.py``.
"""

import pytest

from tests import _mirt_multigroup as mg

pytestmark = pytest.mark.slow


def test_monotone_and_oakes_equals_hessian() -> None:
    mg.check_monotone_and_oakes_equals_hessian("mg2d_simple")
