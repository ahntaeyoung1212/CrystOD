"""Compatibility helpers for phonopy importing newer spglib layouts."""

from __future__ import annotations

import sys
import types


def ensure_spglib_compat() -> None:
    """Provide ``spglib.spglib`` for phonopy versions that still import it."""
    if "spglib.spglib" in sys.modules:
        return

    try:
        import spglib
    except Exception:
        return

    dataset_cls = getattr(spglib, "SpglibDataset", None)
    if dataset_cls is None:
        try:
            from spglib.spg import SpglibDataset as dataset_cls
        except Exception:
            return

    shim = types.ModuleType("spglib.spglib")
    shim.SpglibDataset = dataset_cls
    sys.modules["spglib.spglib"] = shim
