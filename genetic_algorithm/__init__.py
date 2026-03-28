"""Python package alias for the compiled :mod:`ga` extension."""

import ga as _ga

if hasattr(_ga, "__all__"):
    __all__ = _ga.__all__
else:
    __all__ = [name for name in dir(_ga) if not name.startswith("_")]

globals().update({name: getattr(_ga, name) for name in __all__})
