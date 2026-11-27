"""M0 acceptance tests: the package imports and exposes its skeleton types."""

import dataclasses
import os
import subprocess
import sys
import textwrap

import pytest

import flexpyirt
from flexpyirt import FitResult, Model, TwoPL, errors, fit
from flexpyirt._precision import check_x64, x64_enabled


def test_public_api() -> None:
    assert set(flexpyirt.__all__) >= {"Model", "FitResult", "fit", "errors"}
    assert isinstance(flexpyirt.__version__, str)


def test_spec_and_result_are_frozen() -> None:
    assert dataclasses.is_dataclass(Model)
    assert dataclasses.is_dataclass(FitResult)
    model = Model.uniform(TwoPL(), items=["i1", "i2"])
    with pytest.raises(dataclasses.FrozenInstanceError):
        model.foo = 1  # type: ignore[attr-defined]


def test_error_hierarchy() -> None:
    for cls in (
        errors.SpecificationError,
        errors.IdentificationError,
        errors.PrecisionError,
    ):
        assert issubclass(cls, errors.FlexpyirtError)
    assert issubclass(errors.SpecificationError, ValueError)
    assert issubclass(errors.IdentificationError, ValueError)
    for warn in (errors.ConvergenceWarning, errors.HeywoodWarning):
        assert issubclass(warn, errors.FlexpyirtWarning)
        assert issubclass(warn, UserWarning)


def test_x64_check_passes_when_enabled() -> None:
    assert x64_enabled()
    check_x64()


def test_fit_is_keyword_only_and_rejects_undeclared_groups() -> None:
    spec = Model.uniform(TwoPL(), items=["i1", "i2"])
    data = [[0, 1], [1, 0]]
    with pytest.raises(TypeError):
        fit(spec, data, None)  # type: ignore[call-arg]
    with pytest.raises(errors.SpecificationError, match="with_groups"):
        fit(spec, data, group=[0, 1])


def _run_isolated(code: str, *, x64: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "JAX_ENABLE_X64": x64, "JAX_PLATFORMS": "cpu"}
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.slow
def test_import_does_not_touch_global_x64() -> None:
    proc = _run_isolated(
        """
        import jax
        import flexpyirt
        assert not jax.config.jax_enable_x64
        """,
        x64="0",
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.slow
def test_fit_raises_precision_error_without_x64() -> None:
    proc = _run_isolated(
        """
        import flexpyirt
        from flexpyirt.errors import PrecisionError
        try:
            flexpyirt.fit(flexpyirt.Model.uniform(flexpyirt.TwoPL(), items=['i1', 'i2']), [[0, 1]])
        except PrecisionError as e:
            assert "jax_enable_x64" in str(e)
        else:
            raise SystemExit("PrecisionError not raised")
        """,
        x64="0",
    )
    assert proc.returncode == 0, proc.stderr
