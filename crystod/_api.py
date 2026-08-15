"""Lazy attribute machinery for the public API namespaces.

The domain API modules (``crystod.salc``, ``crystod.phonon``,
``crystod.group``, ``crystod.bz``, ``crystod.md``, ``crystod.mag``,
``crystod.mol``) are thin, curated views over the implementation modules.
Because many implementation modules import phonopy/spgrep (and patch
spglib compatibility) at import time, the views resolve their attributes
lazily via PEP 562: ``import crystod.phonon`` is instant, and the heavy
machinery is only pulled in when an attribute is first used.
"""

from __future__ import annotations

import functools
import importlib
import inspect


def _library_errors(function):
    """Report bad input as ``ValueError`` instead of exiting the process.

    The implementation modules double as command-line entry points and
    report bad input the way a command line wants it, by raising
    ``SystemExit``. That is right for a command and wrong for a library:
    ``SystemExit`` derives from ``BaseException``, so a caller's
    ``except Exception`` does not catch it and the whole program dies.
    The API namespaces translate it at their boundary.
    """

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except SystemExit as exc:
            message = " ".join(str(exc).split())
            raise ValueError(
                message.removeprefix("ERROR: ") or "invalid input"
            ) from None

    return wrapper


def lazy_namespace(module_globals: dict, exports: dict):
    """Build ``__getattr__``, ``__dir__``, and ``__all__`` for an API module.

    ``exports`` maps each public name to ``(implementation_module, attribute)``
    with the implementation module given relative to the ``crystod`` package.
    """
    module_name = module_globals["__name__"]

    def __getattr__(name: str):
        try:
            target_module, attribute = exports[name]
        except KeyError:
            raise AttributeError(
                f"module {module_name!r} has no attribute {name!r}"
            ) from None
        module = importlib.import_module(f".{target_module}", "crystod")
        value = getattr(module, attribute)
        if inspect.isfunction(value):
            value = _library_errors(value)
        module_globals[name] = value  # cache; later access bypasses __getattr__
        return value

    def __dir__():
        dunders = {name for name in module_globals if name.startswith("__")}
        return sorted(dunders | set(exports))

    return __getattr__, __dir__, sorted(exports)
