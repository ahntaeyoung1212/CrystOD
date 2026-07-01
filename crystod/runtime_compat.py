"""Compatibility helpers for phonopy / spgrep API differences."""

from __future__ import annotations

from collections.abc import Mapping


try:
    from spgrep.rep.representation import get_character  # type: ignore
except Exception:
    from spgrep.representation import get_character  # type: ignore

try:
    from spgrep.symmetry.group import get_little_group  # type: ignore
except Exception:
    from spgrep.group import get_little_group  # type: ignore


class SymmetryDatasetAdapter(Mapping):
    """Provide stable item/attribute access without triggering deprecated dict APIs."""

    def __init__(self, dataset):
        self._dataset = dataset

    def __getitem__(self, key):
        if hasattr(self._dataset, key):
            return getattr(self._dataset, key)
        if isinstance(self._dataset, Mapping):
            return self._dataset[key]
        return getattr(self._dataset, key)

    def __iter__(self):
        if isinstance(self._dataset, Mapping):
            return iter(self._dataset)
        return iter(vars(self._dataset))

    def __len__(self):
        if isinstance(self._dataset, Mapping):
            return len(self._dataset)
        return len(vars(self._dataset))

    def __getattr__(self, name):
        try:
            return self[name]
        except Exception as exc:
            raise AttributeError(name) from exc


def get_symmetry_dataset(symmetry):
    """Return a mapping-like symmetry dataset across phonopy versions."""
    dataset = None

    getter = getattr(symmetry, "get_dataset", None)
    if callable(getter):
        dataset = getter()
    elif hasattr(symmetry, "dataset"):
        dataset = symmetry.dataset
    elif hasattr(symmetry, "_dataset"):
        dataset = symmetry._dataset

    if dataset is None:
        raise AttributeError("Could not obtain symmetry dataset from phonopy Symmetry.")
    if isinstance(dataset, Mapping) or hasattr(dataset, "__dict__"):
        return SymmetryDatasetAdapter(dataset)
    raise TypeError(f"Unsupported symmetry dataset type: {type(dataset)!r}")


def get_pointgroup_symbol(symmetry):
    """Return the point-group symbol across phonopy versions."""
    getter = getattr(symmetry, "get_pointgroup", None)
    if callable(getter):
        return getter()

    value = getattr(symmetry, "pointgroup_symbol", None)
    if value is None:
        raise AttributeError("Could not obtain point-group symbol from phonopy Symmetry.")
    return value


def get_chemical_symbols(atoms):
    """Return chemical symbols across phonopy versions."""
    if hasattr(atoms, "symbols"):
        return list(atoms.symbols)

    getter = getattr(atoms, "get_chemical_symbols", None)
    if callable(getter):
        return getter()

    raise AttributeError("Could not obtain chemical symbols from PhonopyAtoms.")


def get_scaled_positions(atoms):
    """Return scaled positions across phonopy versions."""
    if hasattr(atoms, "scaled_positions"):
        return atoms.scaled_positions

    getter = getattr(atoms, "get_scaled_positions", None)
    if callable(getter):
        return getter()

    raise AttributeError("Could not obtain scaled positions from PhonopyAtoms.")
