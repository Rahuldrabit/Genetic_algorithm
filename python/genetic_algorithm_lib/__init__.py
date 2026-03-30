"""Python bindings for the Genetic Algorithm framework.

This package is a thin wrapper around the compiled extension module ``_core``.
After installation, users should:

    import genetic_algorithm_lib as ga

"""

from __future__ import annotations

# The extension is expected to live at ``genetic_algorithm_lib._core`` in wheels.
# The fallback supports local non-wheel builds where it may be top-level.
try:
    from ._core import *  # type: ignore  # noqa: F401,F403
except ImportError:  # pragma: no cover
    from _core import *  # type: ignore  # noqa: F401,F403


def __getattr__(name: str):
    if name == "__version__":
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("genetic-algorithm-lib")
        except PackageNotFoundError:  # pragma: no cover
            return "0.0.0"

    raise AttributeError(name)
