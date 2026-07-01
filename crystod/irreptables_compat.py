"""Compatibility loader for old and new ``irreptables`` package layouts."""

from __future__ import annotations


def _ensure_irrep_utility_compat() -> None:
    """Backfill helpers expected by newer irreptables releases."""
    try:
        import irrep.utility as utility
    except Exception:
        return

    if not hasattr(utility, "log_message"):
        def log_message(message: str, verbosity: int = 0, threshold: int = 0) -> None:
            if verbosity >= threshold:
                print(message)

        utility.log_message = log_message


def load_irreptables():
    """Return ``(IrrepTable, Irrep)`` across irreptables package versions."""
    top_level_error = None
    nested_error = None

    try:
        from irreptables import IrrepTable, Irrep
        return _wrap_irreptable(IrrepTable), Irrep
    except Exception as exc:
        top_level_error = exc

    _ensure_irrep_utility_compat()

    try:
        from irreptables.irreps import IrrepTable, Irrep
        return _wrap_irreptable(IrrepTable), Irrep
    except Exception as exc:
        nested_error = exc

    raise ImportError(
        "Failed to import IrrepTable / Irrep from irreptables. "
        "Tried both `from irreptables import ...` and "
        "`from irreptables.irreps import ...`. "
        f"Top-level error: {top_level_error!r}. "
        f"Nested-module error: {nested_error!r}."
    ) from nested_error or top_level_error


def _wrap_irreptable(irrep_table_cls):
    """Adapt constructor differences between irreptables releases."""

    class CompatIrrepTable(irrep_table_cls):
        def __init__(self, SGnumber, spinor, *args, **kwargs):
            super().__init__(str(SGnumber), spinor, *args, **kwargs)

    CompatIrrepTable.__name__ = irrep_table_cls.__name__
    CompatIrrepTable.__qualname__ = irrep_table_cls.__qualname__
    return CompatIrrepTable
