"""Estimation entry point."""

from __future__ import annotations

from typing import Any

from flexpyirt._precision import check_x64
from flexpyirt.model import Model
from flexpyirt.result import FitResult

__all__ = ["fit"]


def fit(spec: Model, data: Any, *, group: Any = None) -> FitResult:
    """Estimate an IRT model by Bock–Aitkin EM (Bock & Aitkin, 1981).

    Parameters
    ----------
    spec : Model
        Model specification.
    data : array-like or DataFrame
        Response matrix (persons × items). Missing responses are NaN.
    group : array-like, optional
        Group label of each person, for multiple-group estimation.

    Returns
    -------
    FitResult
        Estimation result.

    Raises
    ------
    PrecisionError
        If JAX is not configured for 64-bit floating point.
    NotImplementedError
        Always, until estimation is implemented in milestone M1.
    """
    check_x64()
    raise NotImplementedError("Estimation is not implemented yet (planned for milestone M1).")
