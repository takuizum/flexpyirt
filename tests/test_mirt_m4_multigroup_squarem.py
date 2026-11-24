"""V21: multiple-group 2D MIRT vs R mirt with SQUAREM, both cases (two fits, ~1 min each).

Checks and tolerances: ``tests/_mirt_multigroup.py``.
"""

import pytest

from tests import _mirt_multigroup as mg

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("name", mg.CASES)
def test_loglik_and_estimates(name: str) -> None:
    mg.check_loglik_and_estimates(name, squarem=True)
