"""Test-session configuration.

The test suite (not the library) enables 64-bit JAX globally, as a user would.
"""

import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
