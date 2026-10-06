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


class QpointsResultAdapter:
    """Expose a legacy q-points dict through the new attribute API."""

    _KEYS = ("frequencies", "eigenvectors", "group_velocities", "dynamical_matrices")

    def __init__(self, qpoints_dict):
        self._dict = qpoints_dict

    def __getattr__(self, name):
        if name in self._KEYS:
            try:
                return self._dict[name]
            except (KeyError, TypeError) as exc:
                raise AttributeError(name) from exc
        raise AttributeError(name)


def get_qpoints_result(phonon):
    """Return the q-points result object of a run_qpoints() call.

    phonopy exposes it as the ``qpoints`` property; older versions only offer
    the deprecated ``get_qpoints_dict()``, whose dict is wrapped so that
    callers can always use ``.frequencies`` / ``.eigenvectors`` /
    ``.group_velocities`` / ``.dynamical_matrices``.
    """
    if hasattr(type(phonon), "qpoints"):
        result = phonon.qpoints
        if result is None:
            raise RuntimeError("Phonopy.run_qpoints() has to be done.")
        if hasattr(result, "frequencies"):
            return result

    getter = getattr(phonon, "get_qpoints_dict", None)
    if callable(getter):
        return QpointsResultAdapter(getter())

    raise AttributeError("Could not obtain q-points results from phonopy Phonopy.")


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


def get_spacegroup_type(spacegroup_type):
    """Return a stable space-group type object across spglib versions."""
    if spacegroup_type is None:
        raise AttributeError("Could not obtain space-group type from spglib.")

    required_attrs = (
        "international_short",
        "international",
        "international_full",
        "hall_number",
        "number",
    )
    if all(hasattr(spacegroup_type, attr) for attr in required_attrs):
        return spacegroup_type

    if isinstance(spacegroup_type, Mapping):
        return SymmetryDatasetAdapter(spacegroup_type)

    raise TypeError(f"Unsupported space-group type: {type(spacegroup_type)!r}")


def get_spacegroup_irreps_from_primitive_symmetry(rotations, translations, kpoint, **kwargs):
    """spgrep's function of the same name, robust at 2k != 0 k points.

    spgrep (0.6.0) ends ``enumerate_small_representations`` and
    ``enumerate_unitary_irreps`` with ``frobenius_schur_indicator``, i.e.
    (1/|G|) sum_g tr D(g)^2 over the coset representatives only, and raises
    ``ValueError("Given representation is not irreducible: indicator=...")``
    when that rounds to 2 or more.  The sum is the Frobenius-Schur indicator
    only when 2k = 0: for 2k != 0 the average over the lattice translations,
    which makes the indicator vanish, is missing (and for the projective
    co-group irreps the factor system is not real), so the number is
    origin-dependent.  At W of I2_12_12_1 (No. 24, also WA) and Ibca (No. 73)
    it is 2 for the valid two-dimensional small irrep (every 2_1 screw gives
    tr Gamma(R)^2 e^(-4 pi i k.tau) = (-2)(-1) = 2).  On that error the small
    irreps are rebuilt from the same spgrep steps (factor system, projective
    co-group irreps by the 'Neto' chain, translation phases) without the
    check, and are verified to be irreducible, mutually inequivalent and
    complete.  Results are unchanged wherever spgrep itself succeeds.

    Args:
        rotations: Primitive-cell rotations, shape ``(order, 3, 3)``.
        translations: Primitive-cell translations, shape ``(order, 3)``.
        kpoint: k vector in the primitive reciprocal basis.
        **kwargs: Passed to spgrep (``method``, ``atol``, ...).  The rebuild
            uses the same ``atol`` and ``max_num_random_generations``, and
            always the 'Neto' chain.

    Returns:
        ``(irreps, mapping_little_group)`` as returned by spgrep.
    """
    import numpy as np
    from spgrep.core import get_spacegroup_irreps_from_primitive_symmetry as _irreps

    try:
        return _irreps(rotations=rotations, translations=translations, kpoint=kpoint, **kwargs)
    except ValueError as exc:
        kpoint = np.asarray(kpoint, dtype=float)
        if (
            "not irreducible" not in str(exc)
            or kwargs.get("real", False)
            or np.allclose(2 * kpoint, np.rint(2 * kpoint), atol=1e-8)
        ):
            raise
        error = exc
    try:
        from spgrep.symmetry.enumerate import (  # type: ignore
            enumerate_unitary_irreps_from_solvable_group_chain,
            purify_irrep_value,
        )
        from spgrep.symmetry.group import (  # type: ignore
            get_cayley_table,
            get_factor_system_from_little_group,
        )
        from spgrep.symmetry.pointgroup import get_pointgroup_chain_generators  # type: ignore
    except Exception:
        from spgrep.group import get_cayley_table, get_factor_system_from_little_group  # type: ignore
        from spgrep.irreps import (  # type: ignore
            enumerate_unitary_irreps_from_solvable_group_chain,
            purify_irrep_value,
        )
        from spgrep.pointgroup import get_pointgroup_chain_generators  # type: ignore

    # the tolerances the caller gave spgrep (spgrep's defaults otherwise)
    atol = {"atol": kwargs["atol"]} if "atol" in kwargs else {}
    chain_options = dict(atol)
    if "max_num_random_generations" in kwargs:
        chain_options["max_num_random_generations"] = kwargs["max_num_random_generations"]
    little_rotations, little_translations, mapping = get_little_group(
        rotations, translations, kpoint, **atol
    )
    factor_system = get_factor_system_from_little_group(
        little_rotations, little_translations, kpoint
    )
    # spgrep's 'Neto' route of enumerate_unitary_irreps, without its final
    # indicator check (it misfires on the projective co-group irreps too,
    # depending on the origin)
    cogroup_irreps = enumerate_unitary_irreps_from_solvable_group_chain(
        get_cayley_table(little_rotations),
        factor_system,
        get_pointgroup_chain_generators(little_rotations),
        **chain_options,
    )
    cogroup_irreps = [purify_irrep_value(rep, **atol) for rep in cogroup_irreps]
    phases = np.exp(-2j * np.pi * np.asarray(little_translations, dtype=float) @ kpoint)
    irreps = [np.asarray(rep) * phases[:, None, None] for rep in cogroup_irreps]
    order = len(little_rotations)
    # orthogonality of the characters over the coset representatives (the
    # translation phases cancel in conj(chi_a) chi_b): <chi_a, chi_b> =
    # order for a = b (irreducible), 0 otherwise (inequivalent)
    characters = np.array([np.trace(rep, axis1=1, axis2=2) for rep in irreps])
    gram = np.conj(characters) @ characters.T
    orthogonal = np.allclose(gram, order * np.eye(len(irreps)), atol=1e-6)
    complete = sum(rep.shape[1] ** 2 for rep in irreps) == order
    if not (complete and orthogonal):
        raise error
    return irreps, mapping
