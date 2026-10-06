"""Spectroscopic activity of the Gamma-point phonons: IR, Raman, silent, acoustic.

A phonon at the zone centre is infrared (IR) active when its irrep occurs in
the vector representation of the point group (it carries a dipole moment),
and Raman active when its irrep occurs in the symmetric square of the vector
representation (it modulates the polarizability, a symmetric second-rank
tensor). A mode that is neither is silent. The three rigid translations of
the crystal are the acoustic modes; their irreps are those of the vector
representation, so they are reported as ``acoustic`` and not as IR active.
This is the selection-rule table of the Bilbao SAM program, computed from the
characters alone:

- ``n_IR = (1/|G|) sum_g conj(chi(g)) tr R_g``;
- ``n_Raman = (1/|G|) sum_g conj(chi(g)) (tr(R_g)^2 + tr(R_g^2)) / 2``,

with ``R_g`` the point-group rotations (proper and improper alike; the trace
of an integer rotation matrix does not depend on the lattice basis).

``crystod-phonon --irreps`` adds the activity of every degenerate set at
Gamma to ``phonon_irreps.yaml``, and ``crystod-phonon --vibration --qpoint GM``
to every mode space; :func:`gamma_mode_activities` is the library form of
both. The module also holds the quantities built on the same irreps:

- :func:`gamma_raman_tensors` -- the symmetry-allowed form of the Raman
  tensor of every Raman-active irrep (``--raman-tensor``);
- :func:`wyckoff_orbit_decomposition` -- the Gamma irreps carried by each
  orbit of equivalent atoms (the Wyckoff-position breakdown of Bilbao SAM);
- :func:`mulliken_symbols` -- the Mulliken symbol of every ISO-IR Gamma
  irrep, from phonopy's point-group character tables;
- :func:`mode_effective_charges` and :func:`dielectric_response` -- with the
  Born effective charges of a BORN file, the mode effective charges, the
  contribution of every IR set to the static dielectric tensor, the acoustic
  sum rule of the charges and the Lyddane-Sachs-Teller check
  (``--irreps --nac``).
"""

from __future__ import annotations

import itertools
import re
import weakref
from dataclasses import dataclass, field, replace

import numpy as np

_DIM_SUFFIX = re.compile(r"\((\d+)\)$")
_INTEGER_TOLERANCE = 1e-2
_KIND_ORDER = ("IR", "Raman", "silent", "acoustic", "unknown")

# e^2 / (eps_vac * amu) expressed for a primitive-cell volume in Angstrom^3
# and an ordinary frequency in THz (omega = 2 pi nu):
#   Delta eps = DIELECTRIC_FACTOR * S / (Omega * nu^2),  S in e^2/amu.
_ELEMENTARY_CHARGE = 1.602176634e-19  # C
_VACUUM_PERMITTIVITY = 8.8541878128e-12  # F/m
_ATOMIC_MASS_UNIT = 1.66053906660e-27  # kg
DIELECTRIC_FACTOR = (
    _ELEMENTARY_CHARGE**2
    / (_VACUUM_PERMITTIVITY * _ATOMIC_MASS_UNIT)
    / (1e-30 * (2.0 * np.pi * 1e12) ** 2)
)

# components of a symmetric tensor, in the order the Raman tensors use
_TENSOR_COMPONENTS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


@dataclass(frozen=True)
class Activity:
    """Spectroscopic activity of one irrep block at Gamma.

    One record per degenerate phonon set (force-constant route) or per mode
    space (structure-only route). :func:`classify_gamma_irreps` fills the
    multiplicities; the caller adds the acoustic flag and the identification
    (labels, bands, frequency) with :func:`dataclasses.replace`.

    Attributes:
        n_ir: Multiplicity of the block in the vector representation (> 0:
            IR active); ``-1`` when the characters of the block are not those
            of a representation (a degenerate set split by too tight a
            tolerance), so that nothing can be said.
        n_raman: Multiplicity in the symmetric square of the vector
            representation (> 0: Raman active); ``-1`` as for ``n_ir``.
        dimension: Dimension of the block (its character at the identity).
        acoustic: ``True`` when every band of the block is a rigid
            translation of the crystal.
        labels: ISO-IR label(s) of the block without the dimension suffix
            (``"GM4-"``); more than one when the set holds several irreps
            (the acoustic set of a polar crystal, where the three
            translations span two irreps at 0 THz by translation invariance,
            or an accidental degeneracy).
        band_indices: 1-based band indices of the block (force-constant
            route only).
        frequency: Frequency in THz of the first band (force-constant route
            only).
        parts: The block split per irrep (force-constant route):
            ``(label, copies, kind)`` triples, ``kind`` one of ``"IR"``,
            ``"Raman"``, ``"IR+Raman"``, ``"silent"`` or ``"acoustic"``. The
            acoustic bands of a set are the copies of the irreps of the
            vector representation they span; the rest is optical. Empty when
            the set could not be decomposed.

    Example:
        >>> from crystod.phonon_activity import Activity
        >>> Activity(n_ir=1, n_raman=0, dimension=3).activity
        ('IR',)
        >>> Activity(n_ir=1, n_raman=0, dimension=3, acoustic=True).activity
        ('acoustic',)
    """

    n_ir: int
    n_raman: int
    dimension: int = 0
    acoustic: bool = False
    labels: tuple[str, ...] = ()
    band_indices: tuple[int, ...] = ()
    frequency: float | None = None
    parts: tuple[tuple[str, int, str], ...] = ()

    @property
    def activity(self) -> tuple[str, ...]:
        """``('IR',)``, ``('Raman',)``, ``('IR', 'Raman')``, ``('silent',)``,
        ``('acoustic',)``, or ``('unknown',)`` for a block that is not a
        representation; a block holding several irreps lists every kind
        that occurs in it (``('IR', 'silent', 'acoustic')``)."""
        if self.acoustic:
            return ("acoustic",)
        if self.n_ir < 0 or self.n_raman < 0:
            return ("unknown",)
        if self.parts:
            kinds = {part for _, _, kind in self.parts for part in kind.split("+")}
            return tuple(kind for kind in _KIND_ORDER if kind in kinds)
        return _optical_kind(self.n_ir, self.n_raman)

    @property
    def ir_active(self) -> bool:
        """``True`` for an optical block that occurs in the vector representation."""
        return not self.acoustic and self.n_ir > 0

    @property
    def raman_active(self) -> bool:
        """``True`` for an optical block that occurs in the symmetric square."""
        return not self.acoustic and self.n_raman > 0


def _optical_kind(n_ir: int, n_raman: int) -> tuple[str, ...]:
    kinds = tuple(name for name, count in (("IR", n_ir), ("Raman", n_raman)) if count > 0)
    return kinds or ("silent",)


def _as_multiplicity(value: complex, what: str, index: int) -> int:
    """Round a character inner product to the non-negative integer it must be."""
    rounded = int(round(value.real))
    if abs(value - rounded) > _INTEGER_TOLERANCE or rounded < 0:
        raise ValueError(
            f"{what} multiplicity of irrep block {index + 1} is {value.real:.4f}"
            f"{value.imag:+.4f}i, not a non-negative integer: the characters are "
            "not those of a representation of the rotations given."
        )
    return rounded


def classify_gamma_irreps(rotations, characters) -> list[Activity]:
    """IR and Raman multiplicities of irrep blocks at Gamma, from characters.

    Args:
        rotations: The point-group rotations at Gamma, shape ``(n_ops, 3, 3)``,
            integer matrices in any lattice basis (only ``tr R`` and
            ``tr R^2`` are used, which do not depend on the basis); the
            rotation parts of the space-group operations of a primitive cell.
        characters: The characters of the blocks over the same ordered list of
            rotations: one vector of length ``n_ops`` (a single block) or an
            array of shape ``(n_blocks, n_ops)``; complex values allowed.

    Returns:
        One :class:`Activity` per block with ``n_ir``, ``n_raman`` and
        ``dimension`` filled (``acoustic`` is ``False``; the caller decides it,
        since the acoustic modes share their irreps with IR-active ones).

    Raises:
        ValueError: If the shapes do not match, or a multiplicity is not a
            non-negative integer (the characters are not those of a
            representation of these rotations).

    Example:
        >>> import numpy as np
        >>> from crystod.phonon_activity import classify_gamma_irreps
        >>> rotations = [np.eye(3, dtype=int), -np.eye(3, dtype=int)]   # -1
        >>> [a.activity for a in classify_gamma_irreps(rotations, [[1, 1], [1, -1]])]
        [('Raman',), ('IR',)]
    """
    rotations = np.asarray(rotations, dtype=float)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError(f"rotations must have shape (n_ops, 3, 3), got {rotations.shape}")
    table = np.asarray(characters, dtype=complex)
    if table.ndim == 1:
        table = table[None, :]
    order = len(rotations)
    if table.ndim != 2 or table.shape[1] != order:
        raise ValueError(
            f"characters must have {order} entries per block (one per rotation), "
            f"got shape {np.asarray(characters).shape}"
        )
    traces = np.trace(rotations, axis1=1, axis2=2)
    traces_squared = np.trace(rotations @ rotations, axis1=1, axis2=2)
    vector = traces
    symmetric_square = (traces**2 + traces_squared) / 2.0
    is_identity = np.all(np.isclose(rotations, np.eye(3)), axis=(1, 2))
    identity = int(np.flatnonzero(is_identity)[0]) if is_identity.any() else None
    activities = []
    for index, chi in enumerate(table):
        n_ir = _as_multiplicity(np.vdot(chi, vector) / order, "IR", index)
        n_raman = _as_multiplicity(np.vdot(chi, symmetric_square) / order, "Raman", index)
        dimension = int(round(chi[identity].real)) if identity is not None else 0
        activities.append(Activity(n_ir=n_ir, n_raman=n_raman, dimension=dimension))
    return activities


def _irreps_frequencies(irreps):
    """The frequencies of a phonopy ``IrReps`` object (THz)."""
    frequencies = getattr(irreps, "frequencies", None)
    return irreps._freqs if frequencies is None else frequencies


def _strip_dimension(label: str) -> str:
    return _DIM_SUFFIX.sub("", label)


def _is_gamma(qpoint) -> bool:
    q = np.asarray(qpoint, dtype=float)
    return bool(np.allclose(q, np.rint(q), atol=1e-8))


def _natural_key(label: str):
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", label))


# ---------------------------------------------------------------------------
# the irreps at Gamma, labeled, on the rotations of a primitive cell
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class _GammaIrreps:
    """The irreps of the point group at Gamma of one primitive cell.

    ``rotations`` are the rotation parts in the primitive lattice basis,
    ``cartesian`` the same rotations in the Cartesian axes of the cell,
    ``characters[i]`` the character of irrep ``labels[i]`` over them, and
    ``counts[i]`` its multiplicity in the displacement representation.
    """

    rotations: np.ndarray
    cartesian: np.ndarray
    labels: tuple[str, ...]
    characters: np.ndarray
    dimensions: tuple[int, ...]
    counts: tuple[int, ...]
    vibrations: object = field(repr=False)


def _gamma_irreps(vibrations) -> _GammaIrreps:
    """The labeled Gamma irreps of a :class:`SymmetryOnlyVibrations`: the
    spgrep irreps and ISO-IR labels of its mode spaces at Gamma, built once
    per object (the Wyckoff orbits and the Mulliken symbols of one
    ``--vibration`` run share it)."""
    cached = getattr(vibrations, "_gamma_irreps_table", None)
    if isinstance(cached, _GammaIrreps):
        return cached
    table = _build_gamma_irreps(vibrations)
    try:
        vibrations._gamma_irreps_table = table
    except AttributeError:
        pass
    return table


def _build_gamma_irreps(vibrations) -> _GammaIrreps:
    gamma = [0.0, 0.0, 0.0]
    irreps, vibration_rep, mapping = vibrations.get_vibration_rep(gamma)
    labels = vibrations.get_irrep_labels(gamma, irreps, mapping)
    rotations = np.asarray(vibrations.rotations)[mapping]
    lattice_t = np.asarray(vibrations.primitive_cell.cell, dtype=float).T
    cartesian = np.array([lattice_t @ rotation @ np.linalg.inv(lattice_t) for rotation in rotations])
    characters = np.array([np.trace(np.asarray(irrep), axis1=1, axis2=2) for irrep in irreps])
    vibration_characters = np.trace(vibration_rep, axis1=1, axis2=2)
    counts = tuple(
        _as_multiplicity(np.vdot(chi, vibration_characters) / len(rotations), "displacement", i)
        for i, chi in enumerate(characters)
    )
    return _GammaIrreps(
        rotations=rotations,
        cartesian=cartesian,
        labels=tuple(_strip_dimension(label) for label in labels),
        characters=characters,
        dimensions=tuple(int(np.asarray(irrep).shape[1]) for irrep in irreps),
        counts=counts,
        vibrations=vibrations,
    )


def _phonon_vibrations(phonon):
    """A :class:`SymmetryOnlyVibrations` on phonopy's primitive cell as it is
    (its Cartesian axes are those of the input unit cell), labeled in the
    ISO-IR frame of the input unit cell, like the labels of ``--irreps``."""
    from .vibration_modes import SymmetryOnlyVibrations

    return SymmetryOnlyVibrations(
        phonon.primitive,
        symprec=phonon.primitive_symmetry.tolerance,
        standardize=False,
        input_cell=phonon.unitcell,
    )


_PHONON_TABLES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _phonon_gamma_table(phonon) -> _GammaIrreps:
    """:func:`_gamma_irreps` of :func:`_phonon_vibrations`, built once per
    phonopy primitive cell (``label_phonon_modes`` at Gamma, the activity
    records and the Raman tensors of one ``--irreps`` run share it)."""
    primitive = phonon.primitive
    key = (id(phonon.unitcell), float(phonon.primitive_symmetry.tolerance))
    try:
        cached = _PHONON_TABLES.get(primitive)
    except TypeError:
        cached = None
    if cached is not None and cached[0] == key:
        return cached[1]
    table = _gamma_irreps(_phonon_vibrations(phonon))
    try:
        _PHONON_TABLES[primitive] = (key, table)
    except TypeError:
        pass
    return table


def _input_axes_vibrations(cell, symprec: float):
    """A :class:`SymmetryOnlyVibrations` on a primitive cell with the
    Cartesian axes of ``cell`` (the cell itself when it is primitive, else
    phonopy's primitive cell of it, which keeps the axes), labeled in the
    ISO-IR frame of ``cell``."""
    from .vibration_modes import SymmetryOnlyVibrations

    if _is_primitive(cell, symprec):
        return SymmetryOnlyVibrations(cell, symprec=symprec, standardize=False)
    from phonopy.structure.cells import get_primitive, guess_primitive_matrix

    matrix = guess_primitive_matrix(cell, symprec=symprec)
    primitive = get_primitive(cell, matrix, symprec=symprec)
    return SymmetryOnlyVibrations(
        primitive, symprec=symprec, standardize=False, input_cell=cell
    )


def _characters_on(table: _GammaIrreps, rotations) -> np.ndarray | None:
    """The characters of ``table`` over another ordered list of the same
    rotations (phonopy's ``_rotations_at_q``), or None when the lists differ."""
    position = {tuple(np.asarray(r, dtype=int).ravel()): i for i, r in enumerate(table.rotations)}
    try:
        order = [position[tuple(np.asarray(r, dtype=int).ravel())] for r in rotations]
    except KeyError:
        return None
    if len(set(order)) != len(table.rotations):
        return None
    return table.characters[:, order]


# ---------------------------------------------------------------------------
# activity of the phonopy degenerate sets and of the --vibration mode spaces
# ---------------------------------------------------------------------------

def _split_set(chi, characters, table, labels, n_acoustic, n_bands):
    """``(label, copies, kind)`` parts of one degenerate set, or ``()``.

    The set's character is decomposed into the labeled Gamma irreps; the
    ``n_acoustic`` acoustic bands are assigned to the irreps of the vector
    representation (one copy of each, in the order the labels list them, as
    far as they occur), the remaining copies are optical with the kind of
    their own irrep.
    """
    order = len(chi)
    counts = []
    for index, irrep_chi in enumerate(characters):
        value = np.vdot(irrep_chi, chi) / order
        copies = int(round(value.real))
        if abs(value - copies) > _INTEGER_TOLERANCE or copies < 0:
            return ()
        if copies:
            counts.append((index, copies))
    if sum(copies * table.dimensions[i] for i, copies in counts) != n_bands:
        return ()
    names = [table.labels[i] for i, _ in counts]
    if labels:
        wanted = [_strip_dimension(text) for text in labels]
        if sorted(wanted) != sorted(names):
            return ()
        counts.sort(key=lambda item: wanted.index(table.labels[item[0]]))
    records = classify_gamma_irreps(table.rotations, [table.characters[i] for i, _ in counts])
    parts = []
    remaining = n_acoustic
    for (index, copies), record in zip(counts, records):
        dimension = table.dimensions[index]
        acoustic = 0
        if record.n_ir > 0 and remaining >= dimension:
            acoustic = min(copies, record.n_ir, remaining // dimension)
            remaining -= acoustic * dimension
        if acoustic:
            parts.append((table.labels[index], acoustic, "acoustic"))
        if copies > acoustic:
            kind = "+".join(_optical_kind(record.n_ir, record.n_raman))
            parts.append((table.labels[index], copies - acoustic, kind))
    return tuple(parts)


def activities_from_phonopy_irreps(phonon, labels=None) -> list[Activity]:
    """Activity of every degenerate set of ``phonon.irreps`` at Gamma.

    The force-constant route: ``phonon.set_irreps([0, 0, 0], tol)`` (or
    :func:`crystod.phonon.get_irrep_labels` at Gamma, which calls it) must
    have run. The characters of each degenerate set over the little-group
    rotations (``phonon.irreps.characters``, ``_rotations_at_q``) are
    classified with :func:`classify_gamma_irreps`. The number of acoustic
    bands of a set is the sum of the rigid-translation weights of its
    eigenvectors (invariant inside a degenerate subspace); the set is
    acoustic when all its bands are. A set holding several irreps is split
    per irrep (:attr:`Activity.parts`): its acoustic bands go to the irreps
    of the vector representation, the rest keep the activity of their own
    irrep. A set whose characters are not a representation gets
    ``n_ir = n_raman = -1`` (activity ``unknown``) instead of an exception.

    Args:
        phonon: A ``phonopy.Phonopy`` object with force constants, its irreps
            set at Gamma.
        labels: Optional ISO-IR labels per degenerate set as
            :func:`crystod.phonon.get_irrep_labels` returns them (``"GM4-(3)"``
            lists, or ``None`` for an unlabeled set).

    Returns:
        One :class:`Activity` per degenerate set, in band order, with labels,
        1-based band indices, the frequency and the per-irrep parts filled.

    Raises:
        ValueError: If ``phonon.irreps`` is not set at a Gamma point.
    """
    from .phonon_subgroups import _translation_weights

    irreps = getattr(phonon, "irreps", None)
    if irreps is None or not _is_gamma(irreps.qpoint):
        raise ValueError("phonon.irreps must be set at Gamma (phonon.set_irreps([0, 0, 0]))")
    rotations = np.asarray(irreps._rotations_at_q)
    frequencies = np.asarray(_irreps_frequencies(irreps))
    weights = _translation_weights(phonon, np.asarray(irreps.eigenvectors))
    try:
        table = _phonon_gamma_table(phonon)
        characters = _characters_on(table, rotations)
    except Exception:
        table, characters = None, None
    activities = []
    for index, (bands, chi) in enumerate(zip(irreps.band_indices, irreps.characters)):
        set_labels = labels[index] if labels is not None and labels[index] else None
        try:
            record = classify_gamma_irreps(rotations, [chi])[0]
        except ValueError:
            record = Activity(n_ir=-1, n_raman=-1, dimension=len(bands))
        n_acoustic = int(round(float(np.sum(weights[list(bands)]))))
        parts: tuple = ()
        if characters is not None and record.n_ir >= 0:
            parts = _split_set(
                np.asarray(chi, dtype=complex), characters, table, set_labels,
                n_acoustic, len(bands),
            )
        names: tuple[str, ...] = ()
        if set_labels:
            names = tuple(_strip_dimension(text) for text in set_labels)
        activities.append(
            replace(
                record,
                acoustic=n_acoustic >= len(bands),
                labels=names,
                band_indices=tuple(int(band) + 1 for band in bands),
                frequency=float(frequencies[bands[0]]),
                parts=parts,
            )
        )
    return activities


def mode_space_activities(vibrations, qpoint=(0.0, 0.0, 0.0)) -> list[Activity]:
    """Activity of every mode space of ``crystod-phonon --vibration`` at Gamma.

    The structure-only route: the spgrep irreps of the point group at Gamma
    (:meth:`~crystod.vibration_modes.SymmetryOnlyVibrations.get_vibration_rep`)
    are classified from their traces, and each irrep is repeated as many
    times as it occurs in the displacement representation -- the order and
    count of the spaces of
    :meth:`~crystod.vibration_modes.SymmetryOnlyVibrations.describe_mode_spaces`.
    Without force constants the acoustic modes are not one of the spaces
    (each space of a repeated irrep is a symmetry-adapted pattern, not a
    normal mode), so ``acoustic`` stays ``False``; the acoustic sets are
    counted by :func:`format_activity_summary` with ``symmetry_only=True``.

    Args:
        vibrations: A :class:`crystod.vibration_modes.SymmetryOnlyVibrations`.
        qpoint: A Gamma point (any reciprocal lattice vector) in the
            primitive reciprocal basis.

    Returns:
        One :class:`Activity` per mode space, with its label and dimension.

    Raises:
        ValueError: If ``qpoint`` is not a Gamma point.
    """
    if not _is_gamma(qpoint):
        raise ValueError(f"activities are defined at Gamma only, got q = {list(qpoint)}")
    q = [float(value) for value in qpoint]
    irreps, vibration_rep, mapping = vibrations.get_vibration_rep(q)
    labels = vibrations.get_irrep_labels(q, irreps, mapping)
    rotations = np.asarray(vibrations.rotations)[mapping]
    characters = [np.trace(np.asarray(irrep), axis1=1, axis2=2) for irrep in irreps]
    records = classify_gamma_irreps(rotations, characters)
    vibration_characters = np.trace(vibration_rep, axis1=1, axis2=2)
    spaces = []
    for index, (record, chi, label) in enumerate(zip(records, characters, labels)):
        count = _as_multiplicity(
            np.vdot(chi, vibration_characters) / len(rotations), "displacement", index
        )
        named = replace(record, labels=(_strip_dimension(label),))
        spaces.extend([named] * count)
    return spaces


def _read_cell(cell):
    """A ``PhonopyAtoms`` from a structure or a POSCAR path."""
    if isinstance(cell, (str, bytes)) or hasattr(cell, "__fspath__"):
        from .vasp_io import read_poscar_cell

        try:
            return read_poscar_cell(str(cell))
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot read the structure {cell}: {exc}") from None
    return cell


def gamma_mode_activities(
    phonon_or_cell,
    *,
    degeneracy_tolerance: float = 1e-3,
    symprec: float = 1e-5,
) -> list[Activity]:
    """IR / Raman / silent / acoustic activity of the Gamma-point phonons.

    The library form of the activity lines of ``crystod-phonon --irreps``
    (from force constants) and ``crystod-phonon --vibration --qpoint GM``
    (from the structure alone).

    Args:
        phonon_or_cell: A ``phonopy.Phonopy`` object with force constants
            (one record per degenerate set, acoustic sets flagged), or a
            crystal structure -- a ``phonopy.structure.atoms.PhonopyAtoms`` or
            a POSCAR path -- (one record per mode space of ``--vibration``).
            A primitive structure is analysed on its own axes; any other is
            first reduced to the spglib primitive cell (the activities do not
            depend on the axes).
        degeneracy_tolerance: Frequency tolerance (THz) of the degenerate sets
            (force-constant route; ``--tolerance`` of ``--irreps``).
        symprec: Symmetry tolerance of the structure-only route.

    Returns:
        A list of :class:`Activity`; :func:`format_activity_summary` turns it
        into the one-line summary the commands print.

    Raises:
        RuntimeError: If the modes at Gamma cannot be labeled (force-constant
            route).
        ValueError: If the structure cannot be read.

    Example:
        >>> import phonopy
        >>> from crystod import phonon
        >>> from crystod.examples import example_path
        >>> ph = phonopy.load(
        ...     unitcell_filename=example_path("221_PPOSCAR_SrTiO3"),
        ...     force_sets_filename=example_path("FORCE_SETS_SrTiO3"),
        ...     supercell_matrix=[4, 4, 4], primitive_matrix="auto", is_nac=False)
        >>> activities = phonon.gamma_mode_activities(ph)
        >>> [(a.labels, a.activity) for a in activities][:2]
        [(('GM4-',), ('acoustic',)), (('GM4-',), ('IR',))]
        >>> phonon.format_activity_summary(activities)
        'GM4- x4 (3 IR, 1 acoustic), GM5- x1 (silent)'
    """
    if hasattr(phonon_or_cell, "set_irreps"):
        from .phonon_subgroups import label_phonon_modes

        modes = label_phonon_modes(
            phonon_or_cell, (0.0, 0.0, 0.0), degeneracy_tolerance=degeneracy_tolerance
        )
        # label_phonon_modes leaves phonon.irreps at Gamma
        return activities_from_phonopy_irreps(
            phonon_or_cell, [list(mode.labels) or None for mode in modes]
        )

    from .vibration_modes import SymmetryOnlyVibrations

    cell = _read_cell(phonon_or_cell)
    vibrations = SymmetryOnlyVibrations(
        cell, symprec=symprec, standardize=not _is_primitive(cell, symprec)
    )
    return mode_space_activities(vibrations, (0.0, 0.0, 0.0))


def _is_primitive(cell, symprec: float) -> bool:
    """``True`` when spglib finds no smaller primitive cell than ``cell``."""
    import spglib

    primitive = spglib.find_primitive(cell.totuple(), symprec=symprec)
    return primitive is None or len(primitive[2]) == len(cell.numbers)


def format_activity_summary(activities, *, symmetry_only: bool = False, mulliken=None) -> str:
    """One-line summary of the Gamma activities, counted per irrep label.

    ``"GM4- x4 (3 IR, 1 acoustic), GM5- x1 (silent)"``: each label with the
    number of times it occurs, then how many of those are acoustic and what
    the optical ones are (``IR``, ``Raman``, ``IR+Raman``, ``silent``). A set
    holding several irreps counts per irrep through its
    :attr:`Activity.parts` (each copy with its own kind); without parts it
    counts once under each of its labels with the set's activity. An
    unlabeled set appears as ``?``. The labels appear in the order of their
    first appearance (band order of the sets; inside a set holding several
    irreps, the labeler's order).

    Args:
        activities: :class:`Activity` records, e.g. from
            :func:`gamma_mode_activities`.
        symmetry_only: The records are mode spaces without force constants
            (``--vibration``): the acoustic sets are not identified among
            them, and the count of each label in the vector representation
            (``n_ir``) is reported as acoustic.
        mulliken: Optional ``{label: Mulliken symbol}`` map
            (:func:`mulliken_symbols`); each label found in it is followed by
            its symbol in square brackets (``"GM4- [T1u] x4 (...)"``).

    Returns:
        The summary string.

    Example:
        >>> from crystod.phonon_activity import Activity, format_activity_summary
        >>> t1u = Activity(n_ir=1, n_raman=0, dimension=3, labels=("GM4-",))
        >>> t2u = Activity(n_ir=0, n_raman=0, dimension=3, labels=("GM5-",))
        >>> format_activity_summary([t1u] * 4 + [t2u], symmetry_only=True)
        'GM4- x4 (3 IR, 1 acoustic), GM5- x1 (silent)'
    """
    return ", ".join(_activity_summary_parts(activities, symmetry_only, mulliken))


def _activity_summary_parts(activities, symmetry_only: bool = False, mulliken=None) -> list[str]:
    """The per-label entries of :func:`format_activity_summary`, one string
    each (``--vibration`` prints them one per line)."""
    order: list[str] = []
    tallies: dict[str, dict[str, int]] = {}
    n_ir_of: dict[str, int] = {}

    def add(name: str, kind: str, count: int) -> None:
        if name not in tallies:
            order.append(name)
            tallies[name] = {}
        tallies[name][kind] = tallies[name].get(kind, 0) + count

    for record in activities:
        if record.parts and not (symmetry_only or record.n_ir < 0):
            for name, copies, kind in record.parts:
                add(name, kind, copies)
            continue
        names = record.labels or ("?",)
        kind = "+".join(record.activity)
        for name in names:
            if len(names) == 1:
                n_ir_of.setdefault(name, record.n_ir)
            add(name, kind, 1)
    parts = []
    for name in order:
        counts = dict(tallies[name])
        total = sum(counts.values())
        if symmetry_only and n_ir_of.get(name, 0) > 0:
            moved = min(n_ir_of[name], total)
            for kind in list(counts):
                if "IR" in kind.split("+"):
                    taken = min(moved, counts[kind])
                    counts[kind] -= taken
                    moved -= taken
                    if not counts[kind]:
                        del counts[kind]
            counts["acoustic"] = counts.get("acoustic", 0) + min(n_ir_of[name], total)
        kinds = sorted(counts, key=lambda kind: (kind == "acoustic", kind))
        if len(kinds) == 1:
            detail = kinds[0]
        else:
            detail = ", ".join(f"{counts[kind]} {kind}" for kind in kinds)
        parts.append(f"{_with_mulliken(name, mulliken)} x{total} ({detail})")
    return parts


# ---------------------------------------------------------------------------
# Mulliken symbols of the Gamma irreps
# ---------------------------------------------------------------------------

_MULLIKEN_TOLERANCE = 1e-3


def _with_mulliken(label: str, mulliken) -> str:
    """``label`` followed by its Mulliken symbol in square brackets when
    ``mulliken`` has one for it (the dimension suffix is ignored)."""
    if mulliken:
        symbol = mulliken.get(_strip_dimension(label))
        if symbol:
            return f"{label} [{symbol}]"
    return label


def _rotation_classes(rotations) -> list[int]:
    """The conjugacy class (0, 1, ...) of every rotation of a point group."""
    matrices = [np.asarray(rotation, dtype=int) for rotation in rotations]
    position = {tuple(matrix.ravel()): index for index, matrix in enumerate(matrices)}
    inverses = [np.rint(np.linalg.inv(matrix)).astype(int) for matrix in matrices]
    classes = [-1] * len(matrices)
    count = 0
    for index, matrix in enumerate(matrices):
        if classes[index] >= 0:
            continue
        for element, inverse in zip(matrices, inverses):
            image = position.get(tuple((element @ matrix @ inverse).ravel()))
            if image is not None:
                classes[image] = count
        count += 1
    return classes


def _rotation_type(matrix) -> tuple[int, int]:
    matrix = np.asarray(matrix, dtype=float)
    return int(round(np.linalg.det(matrix))), int(round(np.trace(matrix)))


def _phonopy_class_symbols(rotations, transformation, pointgroup: str):
    """``(entry, symbols)``: phonopy's character-table entry of the point
    group and the class symbol of every rotation, or ``(None, None)``.

    As ``phonopy.phonon.irreps.IrReps`` does it: the rotations are carried to
    the conventional standard setting, ``P W P^-1`` with ``P`` the spglib
    transformation matrix, and looked up in the ``mapping_table`` of the
    entries of the point group. A setting no entry covers (the twofold axes
    of P312, P3_112, P3_212) falls back on the conjugacy classes, matched to
    the table classes by rotation type (determinant, trace) and class size
    where that match is unique."""
    from phonopy.phonon.character_table import character_table

    transformation = np.asarray(transformation, dtype=float)
    inverse = np.linalg.inv(transformation)
    conventional = [
        np.rint(transformation @ np.asarray(rotation, dtype=float) @ inverse).astype(int)
        for rotation in rotations
    ]
    entries = character_table.get(pointgroup) or ()
    for entry in entries:
        symbols = []
        for rotation in conventional:
            found = None
            for name, matrices in entry["mapping_table"].items():
                if any((np.asarray(matrix) == rotation).all() for matrix in matrices):
                    found = name
                    break
            if found is None:
                break
            symbols.append(found)
        if len(symbols) == len(conventional):
            return entry, symbols
    if not entries:
        return None, None
    entry = entries[0]
    table_types: dict[tuple, str] = {}
    for name, matrices in entry["mapping_table"].items():
        key = (*_rotation_type(matrices[0]), len(matrices))
        if key in table_types:
            return None, None
        table_types[key] = name
    classes = _rotation_classes(conventional)
    sizes = {index: classes.count(index) for index in set(classes)}
    symbols = []
    for rotation, index in zip(conventional, classes):
        name = table_types.get((*_rotation_type(rotation), sizes[index]))
        if name is None:
            return None, None
        symbols.append(name)
    if len(set(symbols)) != len(sizes):
        return None, None
    return entry, symbols


def _mulliken_of_table(table: _GammaIrreps) -> dict[str, str]:
    """``{label: symbol}`` of the irreps of ``table`` (see
    :func:`mulliken_symbols`); a label no row matches is left out, with one
    warning per call."""
    import warnings

    dataset = table.vibrations.spglib_dataset
    pointgroup = str(dataset["pointgroup"]).strip()
    try:
        entry, symbols = _phonopy_class_symbols(
            table.rotations, dataset["transformation_matrix"], pointgroup
        )
    except Exception:
        entry, symbols = None, None
    if entry is None:
        warnings.warn(
            f"no Mulliken symbols: the rotations of point group {pointgroup} do not "
            "fit phonopy's character table",
            stacklevel=3,
        )
        return {}
    rotation_list = list(entry["rotation_list"])
    sizes = np.array([len(entry["mapping_table"][name]) for name in rotation_list], dtype=float)
    rows = {name: np.array(row, dtype=complex) for name, row in entry["character_table"].items()}

    def symbol_of(chi) -> str | None:
        averaged = np.zeros(len(rotation_list), dtype=complex)
        for name, value in zip(symbols, chi):
            averaged[rotation_list.index(name)] += value
        averaged /= sizes
        for name, row in rows.items():
            if np.abs(averaged - row).max() < _MULLIKEN_TOLERANCE:
                return name
        return None

    found: dict[str, str] = {}
    unmatched = []
    characters = np.asarray(table.characters, dtype=complex)
    for index, (label, chi) in enumerate(zip(table.labels, characters)):
        if label in found:
            continue
        symbol = symbol_of(chi)
        if symbol is None and np.abs(chi.imag).max() > 1e-6:
            # a complex irrep: its symbol is that of the physically irreducible
            # sum with its conjugate (E of 3, Eg of -3, E1 of 6, ...)
            for other, other_chi in enumerate(characters):
                if other != index and np.allclose(other_chi, chi.conj(), atol=1e-6):
                    symbol = symbol_of(chi + other_chi)
                    if symbol is not None:
                        partner = table.labels[other]
                        found[partner] = symbol
                        pair = "".join(sorted((label, partner), key=_natural_key))
                        found[pair] = symbol
                    break
        if symbol is None:
            unmatched.append(label)
        else:
            found[label] = symbol
    if unmatched:
        warnings.warn(
            f"no Mulliken symbol of point group {pointgroup} matches the characters of "
            f"{', '.join(unmatched)}",
            stacklevel=3,
        )
    return found


def mulliken_symbols(source, *, symprec: float = 1e-5) -> dict[str, str]:
    """Mulliken symbols of the Gamma irreps, keyed by their ISO-IR labels.

    The point-group character tables of phonopy
    (``phonopy.phonon.character_table``, the ones phonopy's own ``IrReps``
    labels its Gamma modes with) name the ISO-IR Gamma irreps of the
    structure: the point-group rotations are carried to the conventional
    standard setting of spglib exactly as phonopy does it, each is assigned
    to its class through the ``mapping_table`` of the table, and the
    character of every irrep, averaged over each class, is compared with the
    rows of the table (tolerance 1e-3). The symbols therefore follow
    phonopy's axis convention, the one of its own ``ir_label`` output: B1/B2
    (and B1/B2/B3) refer to the axes of the conventional standard setting.
    A complex irrep and its conjugate, which together form one physically
    irreducible representation, both get the symbol of that representation
    (``E`` of the point group 3, ``Eg`` of -3), and so does their combined
    label (``"GM2GM3"``, the label of the Raman tensors of such a pair). An
    irrep no row matches is left out and a warning is issued; the function
    never raises for that reason.

    Args:
        source: A :class:`crystod.vibration_modes.SymmetryOnlyVibrations`, a
            ``phonopy.Phonopy`` object (the irreps and labels of
            ``--irreps``), or a crystal structure -- a
            ``phonopy.structure.atoms.PhonopyAtoms`` or a POSCAR path.
        symprec: Symmetry tolerance of the structure route.

    Returns:
        ``{label: symbol}`` with the labels without the dimension suffix,
        e.g. ``{"GM1": "A1", "GM4": "B1", "GM5": "E2", "GM6": "E1", ...}`` for
        wurtzite ZnO; every Gamma irrep of the point group is a key, not only
        those that occur among the phonons.

    Example:
        >>> from crystod import phonon
        >>> from crystod.examples import example_path
        >>> symbols = phonon.mulliken_symbols(example_path("221_PPOSCAR_SrTiO3"))
        >>> symbols["GM4-"], symbols["GM5-"]
        ('T1u', 'T2u')
    """
    if hasattr(source, "set_irreps"):
        table = _phonon_gamma_table(source)
    elif hasattr(source, "get_vibration_rep"):
        table = _gamma_irreps(source)
    else:
        table = _gamma_irreps(_input_axes_vibrations(_read_cell(source), symprec))
    return _mulliken_of_table(table)


# ---------------------------------------------------------------------------
# Raman tensors
# ---------------------------------------------------------------------------

def _symmetric_tensor_matrix(rotation) -> np.ndarray:
    """The 6x6 matrix of ``T -> R T R^T`` on symmetric tensors, in the
    Frobenius-orthonormal coordinates (xx, yy, zz, sqrt2 yz, sqrt2 xz,
    sqrt2 xy)."""
    basis = []
    for i, j in _TENSOR_COMPONENTS:
        tensor = np.zeros((3, 3))
        if i == j:
            tensor[i, i] = 1.0
        else:
            tensor[i, j] = tensor[j, i] = 1.0 / np.sqrt(2.0)
        basis.append(tensor)
    matrix = np.zeros((6, 6))
    for column, tensor in enumerate(basis):
        image = rotation @ tensor @ rotation.T
        for row, other in enumerate(basis):
            matrix[row, column] = np.sum(image * other)
    return matrix


def _orthonormal_to_components(vector) -> np.ndarray:
    """Frobenius-orthonormal coordinates -> tensor components
    (xx, yy, zz, yz, xz, xy)."""
    vector = np.asarray(vector, dtype=float).copy()
    vector[3:] /= np.sqrt(2.0)
    return vector


def _sparsest_vector(space, tol: float = 1e-8):
    """The unit vector of the row space of ``space`` with the fewest nonzero
    components (supports tried in the order xx, yy, zz, yz, xz, xy)."""
    for size in range(1, 7):
        for support in itertools.combinations(range(6), size):
            outside = [i for i in range(6) if i not in support]
            if outside:
                _, values, rows = np.linalg.svd(space[:, outside].T)
                rank = int(np.sum(values > tol))
                if rank >= space.shape[0]:
                    continue
                coefficients = rows[rank]
            else:
                coefficients = np.eye(space.shape[0])[0]
            vector = coefficients @ space
            if np.linalg.norm(vector) > tol:
                return vector / np.linalg.norm(vector)
    return None


def _canonical_basis(space, tol: float = 1e-8) -> list[np.ndarray]:
    """A canonical basis of a real subspace of the symmetric tensors.

    Sparsest first: repeatedly the vector of the remaining space with the
    fewest nonzero components (supports tried in the order xx, yy, zz, yz,
    xz, xy), then the remaining space is reduced to its orthogonal
    complement (Frobenius metric). For the cubic Eg pair this gives
    ``xx - yy`` and ``xx + yy - 2 zz``, for T2g the three off-diagonal
    pairs. Each vector is scaled so that its first nonzero component is 1,
    and the list is ordered by that component, wider tensors first.
    """
    remaining = np.asarray(space, dtype=float)
    chosen = []
    while remaining.shape[0]:
        found = _sparsest_vector(remaining, tol)
        chosen.append(found)
        projected = remaining - np.outer(remaining @ found, found)
        _, values, rows = np.linalg.svd(projected)
        remaining = rows[: int(np.sum(values > 1e-6))]
    tensors = []
    for vector in chosen:
        components = _orthonormal_to_components(vector)
        components[np.abs(components) < 1e-9] = 0.0
        first = int(np.flatnonzero(np.abs(components) > 1e-9)[0])
        tensors.append(components / components[first])

    def key(components):
        support = np.flatnonzero(np.abs(components) > 1e-9)
        return (int(support[0]), -len(support), tuple(support))

    tensors.sort(key=key)
    return [_components_to_tensor(components) for components in tensors]


def _tensor_to_orthonormal(tensor) -> np.ndarray:
    """3x3 symmetric tensor -> Frobenius-orthonormal coordinates."""
    tensor = np.asarray(tensor, dtype=float)
    vector = np.array([tensor[i, j] for i, j in _TENSOR_COMPONENTS])
    vector[3:] *= np.sqrt(2.0)
    return vector


def _null_space(matrix, tol: float = 1e-8) -> np.ndarray:
    """Orthonormal rows spanning the null space of ``matrix``."""
    _, values, rows = np.linalg.svd(matrix)
    scale = max(float(values[0]) if len(values) else 0.0, 1.0)
    rank = int(np.sum(values > tol * scale))
    return rows[rank:]


def _raman_partner_forms(rotations, space, dimension: int, tol: float = 1e-6):
    """Per-partner Raman tensors of a repeated irrep with shared constants.

    ``space`` (orthonormal rows, Frobenius coordinates) is the isotypic
    subspace of a ``dimension``-dimensional real irrep. A sparsest tensor
    ``T*`` of it whose orbit ``span{R_g T* R_g^T}`` has the dimension of the
    irrep fixes a real orthogonal form ``D(g)`` of the irrep (``u_1 = T*``
    completed to an orthonormal basis of the orbit). Every equivariant map
    ``Phi`` from that irrep into the tensors (``M(g) Phi = Phi D(g)``) is a
    possible Raman tensor of a mode: partner ``j`` is ``Phi u_j``. The maps
    form a space of dimension ``n`` (the multiplicity; twice that for an
    irrep of complex type); its basis is chosen so that partner 1 runs over
    the canonical basis of its tensors, one constant each.

    Returns:
        A list with one ``(n_constants, 3, 3)`` array per partner (the
        coefficient tensor of each constant), or None when no such ``T*``
        is found.
    """
    matrices = np.array([_symmetric_tensor_matrix(rotation) for rotation in rotations])
    candidates = [_sparsest_vector(space)]
    candidates += [_tensor_to_orthonormal(tensor) for tensor in _canonical_basis(space)]
    basis = None
    for candidate in candidates:
        if candidate is None:
            continue
        start = candidate / np.linalg.norm(candidate)
        _, values, rows = np.linalg.svd(matrices @ start)
        orbit = rows[: int(np.sum(values > tol))]
        if len(orbit) != dimension:
            continue
        rest = orbit - np.outer(orbit @ start, start)
        _, values, rows = np.linalg.svd(rest)
        basis = np.column_stack([start] + list(rows[: dimension - 1]))
        break
    if basis is None:
        return None
    blocks = []
    identity6, identity_d = np.eye(6), np.eye(dimension)
    for matrix in matrices:
        represented = basis.T @ matrix @ basis
        blocks.append(np.kron(identity_d, matrix) - np.kron(represented.T, identity6))
    maps = [row.reshape((6, dimension), order="F") for row in _null_space(np.vstack(blocks))]
    if not maps:
        return None
    first = np.array([phi[:, 0] for phi in maps])
    _, values, rows = np.linalg.svd(first)
    first_space = rows[: int(np.sum(values > tol))]
    if len(first_space) != len(maps):
        return None
    columns = np.column_stack([phi[:, 0] for phi in maps])
    chosen = []
    for tensor in _canonical_basis(first_space):
        target = _tensor_to_orthonormal(tensor)
        alpha = np.linalg.lstsq(columns, target, rcond=None)[0]
        chosen.append(sum(a * phi for a, phi in zip(alpha, maps)))
    forms = []
    for j in range(dimension):
        coefficients = np.array(
            [_orthonormal_to_components(phi[:, j]) for phi in chosen]
        ).T  # (6 components, n_constants)
        coefficients[np.abs(coefficients) < 1e-9] = 0.0
        nonzero = np.flatnonzero(np.abs(coefficients.ravel()) > 1e-9)
        if j and len(nonzero) and coefficients.ravel()[nonzero[0]] < 0:
            coefficients = -coefficients
        forms.append(
            np.array([_components_to_tensor(coefficients[:, c]) for c in range(len(chosen))])
        )
    return forms


def _components_to_tensor(components) -> np.ndarray:
    tensor = np.zeros((3, 3))
    for value, (i, j) in zip(components, _TENSOR_COMPONENTS):
        tensor[i, j] = tensor[j, i] = value
    return tensor


def raman_tensor_basis(rotations_cartesian, character) -> list[np.ndarray]:
    """Basis of the symmetric second-rank tensors that transform as an irrep.

    The character projector ``(d/|G|) sum_g conj(chi(g)) (R_g x R_g)`` on the
    six-dimensional space of symmetric tensors (``T -> R T R^T``) gives the
    isotypic subspace of the irrep, of dimension ``d * n_Raman``; it is
    returned in the canonical basis of the sparsest tensors (see
    ``crystod-phonon --raman-tensor``). A complex character is combined with
    its conjugate (the physically irreducible pair), which gives a real
    subspace.

    Args:
        rotations_cartesian: The point-group rotations in Cartesian
            coordinates, shape ``(n_ops, 3, 3)`` (orthogonal matrices).
        character: The character of the irrep over the same rotations.

    Returns:
        The basis tensors as 3x3 arrays, each scaled so that its first
        nonzero component (in the order xx, yy, zz, yz, xz, xy) is 1; empty
        for a Raman-inactive irrep.

    Raises:
        ValueError: If a rotation is not orthogonal.

    Example:
        >>> import numpy as np
        >>> from crystod.phonon_activity import raman_tensor_basis
        >>> ops = [np.eye(3), np.diag([-1.0, -1.0, 1.0])]    # point group 2 (z)
        >>> [t.tolist() for t in raman_tensor_basis(ops, [1, -1])]
        [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
    """
    rotations = np.asarray(rotations_cartesian, dtype=float)
    for rotation in rotations:
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
            raise ValueError("the Cartesian rotations must be orthogonal matrices")
    chi = np.asarray(character, dtype=complex)
    if np.abs(chi.imag).max() > 1e-6:
        chi = chi + chi.conj()
    identity = np.all(np.isclose(rotations, np.eye(3)), axis=(1, 2))
    dimension = chi[int(np.flatnonzero(identity)[0])].real if identity.any() else 1.0
    projector = np.zeros((6, 6))
    for value, rotation in zip(chi, rotations):
        projector += np.real(np.conj(value)) * _symmetric_tensor_matrix(rotation)
    projector *= dimension / len(rotations)
    _, values, rows = np.linalg.svd(projector)
    space = rows[: int(np.sum(values > 1e-6))]
    if not len(space):
        return []
    return _canonical_basis(space)


@dataclass(frozen=True, eq=False)
class RamanTensors:
    """The symmetry-allowed Raman tensors of one Raman-active irrep.

    Attributes:
        label: ISO-IR label of the irrep; a complex irrep is listed together
            with its conjugate (``"GM2GM3"``).
        dimension: Dimension of the (physically irreducible) irrep.
        multiplicity: Its multiplicity in the symmetric square of the vector
            representation.
        tensors: ``dimension * multiplicity`` basis tensors (3x3 arrays) in
            the Cartesian axes of the input cell, a basis of the isotypic
            subspace of the irrep.
        partners: For an irrep of dimension >= 2 that occurs more than once
            (E of 3m, -3m, 32, 3, -3), the Raman tensor of each partner with
            constants shared across the partners: one array of shape
            ``(n_constants, 3, 3)`` per partner, the coefficient tensor of
            each constant (Loudon's form, e.g. ``[[c, 0, 0], [0, -c, d],
            [0, d, 0]]`` and ``[[0, -c, -d], [-c, 0, 0], [-d, 0, 0]]`` for
            Eg of -3m). Empty otherwise, where ``tensors`` are the partners
            (multiplicity 1) or the irrep is one-dimensional.
    """

    label: str
    dimension: int
    multiplicity: int
    tensors: tuple
    partners: tuple = ()


def _raman_records(table: _GammaIrreps, only_present: bool = True) -> list[RamanTensors]:
    records = classify_gamma_irreps(table.rotations, table.characters)
    done: set[int] = set()
    found = []
    for index, record in enumerate(records):
        if index in done or record.n_raman <= 0:
            continue
        if only_present and not table.counts[index]:
            continue
        chi = table.characters[index]
        label, dimension = table.labels[index], table.dimensions[index]
        if np.abs(chi.imag).max() > 1e-6:
            for other, other_chi in enumerate(table.characters):
                if other != index and np.allclose(other_chi, chi.conj(), atol=1e-6):
                    done.add(other)
                    label = "".join(sorted((label, table.labels[other]), key=_natural_key))
                    dimension += table.dimensions[other]
                    break
        done.add(index)
        tensors = raman_tensor_basis(table.cartesian, chi)
        multiplicity = len(tensors) // max(dimension, 1)
        partners: tuple = ()
        if dimension >= 2 and multiplicity >= 2:
            _, _, rows = np.linalg.svd(np.array([_tensor_to_orthonormal(t) for t in tensors]))
            forms = _raman_partner_forms(table.cartesian, rows[: len(tensors)], dimension)
            if forms is not None:
                partners = tuple(forms)
        found.append(
            RamanTensors(
                label=label,
                dimension=dimension,
                multiplicity=multiplicity,
                tensors=tuple(tensors),
                partners=partners,
            )
        )
    found.sort(key=lambda item: _natural_key(item.label))
    return found


def gamma_raman_tensors(phonon_or_cell, *, symprec: float = 1e-5) -> list[RamanTensors]:
    """Raman tensors of the Raman-active Gamma phonon irreps.

    For every irrep that occurs among the Gamma phonons and in the symmetric
    square of the vector representation, the basis of the symmetric
    second-rank tensors that transform as that irrep, in the Cartesian axes
    of the input cell (the polarizability derivative of a mode of the irrep
    is a combination of them); for an irrep of dimension >= 2 that occurs
    more than once (E of the trigonal point groups) also the tensor of each
    partner with constants shared across the partners
    (:attr:`RamanTensors.partners`). The irreps and their ISO-IR labels are those
    of ``--irreps`` / ``--vibration``; the rotations are the point-group
    operations of a primitive cell with the axes of the input cell,
    ``R = L^T W (L^T)^-1`` with ``L`` the lattice vectors as rows.

    Args:
        phonon_or_cell: A ``phonopy.Phonopy`` object (its primitive cell
            keeps the axes of the input unit cell), or a crystal structure --
            a ``phonopy.structure.atoms.PhonopyAtoms`` or a POSCAR path.
        symprec: Symmetry tolerance of the structure route.

    Returns:
        One :class:`RamanTensors` per Raman-active irrep, in label order;
        empty when no Gamma phonon is Raman active (cubic perovskites).

    Example:
        >>> from crystod import phonon
        >>> records = phonon.gamma_raman_tensors("227_PPOSCAR_Si")   # diamond Si
        >>> [(record.label, len(record.tensors)) for record in records]
        [('GM5+', 3)]
        >>> print(phonon.format_raman_tensors(records)[1])
          GM5+ (3 tensors):  [[0, 0, 0], [0, 0, a], [0, a, 0]]   [[0, 0, a], [0, 0, 0], [a, 0, 0]]   [[0, a, 0], [a, 0, 0], [0, 0, 0]]
    """
    if hasattr(phonon_or_cell, "set_irreps"):
        return _raman_records(_phonon_gamma_table(phonon_or_cell))
    vibrations = _input_axes_vibrations(_read_cell(phonon_or_cell), symprec)
    return _raman_records(_gamma_irreps(vibrations))


def _nice_number(value: float) -> str | None:
    """``value`` as a short exact expression (``2``, ``-1/2``,
    ``sqrt(3)/2``), or None."""
    for root in (1, 2, 3, 6):
        scaled = value / np.sqrt(root)
        for denominator in range(1, 13):
            numerator = scaled * denominator
            if abs(numerator - round(numerator)) < 1e-6 and round(numerator) != 0:
                numerator = int(round(numerator))
                sign = "-" if numerator < 0 else ""
                numerator = abs(numerator)
                if root == 1:
                    text = f"{numerator}" if denominator == 1 else f"{numerator}/{denominator}"
                else:
                    text = f"sqrt({root})" if numerator == 1 else f"{numerator}*sqrt({root})"
                    if denominator != 1:
                        text += f"/{denominator}"
                return sign + text
    return None


def _format_entry(value: float, letter: str) -> str:
    if abs(value) < 1e-9:
        return "0"
    text = _nice_number(value)
    if text is None:
        return f"{value:.4f}*{letter}"
    if text == "1":
        return letter
    if text == "-1":
        return f"-{letter}"
    return f"{text}*{letter}"


def format_tensor(tensor, letter: str = "a") -> str:
    """A 3x3 tensor with one free constant, as ``[[a, 0, 0], [0, -a, 0], ...]``.

    Args:
        tensor: The 3x3 array of coefficients.
        letter: The name of the free constant.

    Returns:
        The tensor as nested lists, entries rationalized where possible
        (``2*a``, ``sqrt(3)/2*a``), else with four decimals.

    Example:
        >>> import numpy as np
        >>> from crystod.phonon_activity import format_tensor
        >>> format_tensor(np.diag([1.0, 1.0, -2.0]))
        '[[a, 0, 0], [0, a, 0], [0, 0, -2*a]]'
    """
    rows = [", ".join(_format_entry(float(value), letter) for value in row) for row in tensor]
    return "[" + ", ".join(f"[{row}]" for row in rows) + "]"


def _format_combination(coefficients, letters) -> str:
    """``sum_c coefficients[c] * letters[c]`` as ``a - sqrt(3)*b``."""
    terms = [(float(value), letter) for value, letter in zip(coefficients, letters)
             if abs(value) > 1e-9]
    if not terms:
        return "0"
    text = ""
    for position, (value, letter) in enumerate(terms):
        magnitude = _format_entry(abs(value), letter)
        if position == 0:
            text = f"-{magnitude}" if value < 0 else magnitude
        else:
            text += f" - {magnitude}" if value < 0 else f" + {magnitude}"
    return text


def format_tensor_form(form) -> str:
    """A 3x3 tensor with several constants, as ``[[a, 0, 0], [0, -a, b], ...]``.

    Args:
        form: Array of shape ``(n_constants, 3, 3)``: the coefficient tensor
            of each constant (``a``, ``b``, ...).

    Returns:
        The tensor as nested lists, each entry a combination of the constants.

    Example:
        >>> import numpy as np
        >>> from crystod.phonon_activity import format_tensor_form
        >>> form = np.zeros((2, 3, 3))
        >>> form[0] = np.diag([1.0, -1.0, 0.0]); form[1][1, 2] = form[1][2, 1] = 1.0
        >>> format_tensor_form(form)
        '[[a, 0, 0], [0, -a, b], [0, b, 0]]'
    """
    form = np.asarray(form, dtype=float)
    letters = _constant_names(len(form))
    rows = [
        ", ".join(_format_combination(form[:, i, j], letters) for j in range(3))
        for i in range(3)
    ]
    return "[" + ", ".join(f"[{row}]" for row in rows) + "]"


def _constant_names(count: int) -> list[str]:
    import string

    letters = string.ascii_lowercase
    return [letters[i] if i < len(letters) else f"c{i + 1}" for i in range(count)]


def format_raman_tensors(records, *, mulliken=None) -> list[str]:
    """The lines ``--raman-tensor`` prints.

    Args:
        records: :class:`RamanTensors` from :func:`gamma_raman_tensors`.
        mulliken: Optional ``{label: Mulliken symbol}`` map
            (:func:`mulliken_symbols`); each label found in it is followed by
            its symbol in square brackets (``"GM5+ [T2g] (3 tensors): ..."``).

    Returns:
        A header line and one line per irrep, or the line saying that no
        Gamma phonon is Raman active. An irrep with
        :attr:`RamanTensors.partners` gets one tensor per partner with the
        constants ``a, b, ...`` shared across them; any other irrep its basis
        tensors side by side, each with its own free constant ``a``.
    """
    lines = ["Raman tensors (Cartesian axes of the input cell):"]
    if not records:
        lines.append("  no Raman-active irrep at Gamma")
        return lines
    for record in records:
        if record.partners:
            names = ", ".join(_constant_names(len(record.partners[0])))
            tensors = "   ".join(format_tensor_form(form) for form in record.partners)
            lines.append(
                f"  {_with_mulliken(record.label, mulliken)} ({len(record.partners)} partners, "
                f"constants {names}):  "
                f"{tensors}"
            )
            continue
        count = len(record.tensors)
        noun = "tensor" if count == 1 else "tensors"
        tensors = "   ".join(format_tensor(tensor) for tensor in record.tensors)
        lines.append(f"  {_with_mulliken(record.label, mulliken)} ({count} {noun}):  {tensors}")
    return lines


# ---------------------------------------------------------------------------
# Wyckoff-orbit breakdown
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WyckoffOrbit:
    """The Gamma irreps carried by one orbit of equivalent atoms.

    Attributes:
        element: Chemical symbol of the orbit.
        wyckoff: Wyckoff letter (spglib, standard setting).
        multiplicity: Multiplicity of the Wyckoff position (in the
            conventional cell).
        atoms: 1-based indices of the orbit's atoms in the primitive cell of
            the analysis.
        irreps: ``(label, copies)`` pairs: the decomposition of the
            displacement representation of the orbit at Gamma.
    """

    element: str
    wyckoff: str
    multiplicity: int
    atoms: tuple[int, ...]
    irreps: tuple[tuple[str, int], ...]

    def __str__(self) -> str:
        terms = " + ".join(
            f"{copies} {label}" if copies > 1 else label for label, copies in self.irreps
        )
        return f"Wyckoff orbit {self.element} ({self.multiplicity}{self.wyckoff}): {terms}"


def wyckoff_orbit_decomposition(vibrations) -> list[WyckoffOrbit]:
    """The Gamma phonon irreps per orbit of equivalent atoms (Bilbao SAM).

    The displacement representation restricted to one orbit has the
    character ``chi(g) = (atoms of the orbit that g maps onto themselves,
    modulo lattice translations) * tr R_g`` at Gamma; it is decomposed into
    the labeled Gamma irreps of the mode spaces of ``--vibration``. The sum
    over the orbits is the full mode-space list.

    Args:
        vibrations: A :class:`crystod.vibration_modes.SymmetryOnlyVibrations`
            (any primitive cell; the orbits are spglib's
            ``equivalent_atoms`` of it).

    Returns:
        One :class:`WyckoffOrbit` per orbit, in atom order; the irreps of an
        orbit in natural label order (``GM1, GM2, ..., GM10``), as in the sum
        line of :func:`format_wyckoff_orbits`.

    Raises:
        ValueError: If the cell of ``vibrations`` is not a primitive cell
            (e.g. a supercell analysed with ``standardize=False``).

    Example:
        >>> from crystod import phonon
        >>> from crystod.examples import example_path
        >>> from crystod.vasp_io import read_poscar_cell
        >>> vib = phonon.SymmetryOnlyVibrations(
        ...     read_poscar_cell(example_path("221_PPOSCAR_SrTiO3")), standardize=False)
        >>> for orbit in phonon.wyckoff_orbit_decomposition(vib):
        ...     print(orbit)
        Wyckoff orbit Sr (1a): GM4-
        Wyckoff orbit Ti (1b): GM4-
        Wyckoff orbit O (3c): 2 GM4- + GM5-
    """
    table = _gamma_irreps(vibrations)
    cell = vibrations.primitive_cell
    dataset = vibrations.spglib_dataset
    equivalent = np.asarray(dataset["equivalent_atoms"])
    wyckoffs = list(dataset["wyckoffs"])
    ratio = abs(np.linalg.det(np.asarray(dataset["std_lattice"], dtype=float))) / abs(
        np.linalg.det(np.asarray(cell.cell, dtype=float))
    )
    centring = int(round(ratio))
    if centring < 1 or abs(ratio - centring) > 1e-3:
        raise ValueError(
            f"the cell is not a primitive cell of its crystal (conventional/primitive "
            f"volume ratio {ratio:.4f} is not a positive integer)"
        )
    positions = np.asarray(cell.scaled_positions, dtype=float)
    symbols = list(cell.symbols)
    translations = np.asarray(vibrations.translations)
    _, _, mapping = vibrations.get_vibration_rep([0.0, 0.0, 0.0])
    translations = translations[mapping]
    traces = np.trace(table.rotations, axis1=1, axis2=2)
    order = len(table.rotations)
    orbits = []
    for representative in dict.fromkeys(equivalent.tolist()):
        atoms = [i for i, value in enumerate(equivalent) if value == representative]
        fixed = np.zeros(order)
        for g, (rotation, translation) in enumerate(zip(table.rotations, translations)):
            for atom in atoms:
                image = rotation @ positions[atom] + translation - positions[atom]
                if np.allclose(image, np.rint(image), atol=1e-4):
                    fixed[g] += 1
        chi = fixed * traces
        irreps = []
        for index, irrep_chi in enumerate(table.characters):
            copies = _as_multiplicity(np.vdot(irrep_chi, chi) / order, "orbit", index)
            if copies:
                irreps.append((table.labels[index], copies))
        irreps.sort(key=lambda item: _natural_key(item[0]))
        orbits.append(
            WyckoffOrbit(
                element=symbols[representative],
                wyckoff=str(wyckoffs[representative]),
                multiplicity=len(atoms) * centring,
                atoms=tuple(atom + 1 for atom in atoms),
                irreps=tuple(irreps),
            )
        )
    return orbits


def format_wyckoff_orbits(orbits, mode_space_labels, *, mulliken=None) -> list[str]:
    """The lines of the Wyckoff-orbit breakdown of ``--vibration --qpoint GM``.

    Args:
        orbits: :class:`WyckoffOrbit` records.
        mode_space_labels: The labels of the mode spaces (``"GM4-(3)"`` or
            without the suffix), one per space.
        mulliken: Optional ``{label: Mulliken symbol}`` map
            (:func:`mulliken_symbols`); in the orbit lines each label found in
            it is followed by its symbol in square brackets (``"GM1 [A1]"``).

    Returns:
        One line per orbit and a check line comparing the summed
        decomposition with the mode-space list.
    """
    lines = ["Wyckoff-orbit breakdown (Gamma):"]
    for orbit in orbits:
        if not mulliken:
            lines.append(f"  {orbit}")
            continue
        terms = " + ".join(
            f"{copies} {_with_mulliken(label, mulliken)}" if copies > 1
            else _with_mulliken(label, mulliken)
            for label, copies in orbit.irreps
        )
        lines.append(
            f"  Wyckoff orbit {orbit.element} ({orbit.multiplicity}{orbit.wyckoff}): {terms}"
        )
    total: dict[str, int] = {}
    for orbit in orbits:
        for label, copies in orbit.irreps:
            total[label] = total.get(label, 0) + copies
    spaces: dict[str, int] = {}
    for label in mode_space_labels:
        name = _strip_dimension(label)
        spaces[name] = spaces.get(name, 0) + 1
    terms = " + ".join(
        f"{copies} {label}" if copies > 1 else label
        for label, copies in sorted(total.items(), key=lambda item: _natural_key(item[0]))
    )
    verdict = "equals" if total == spaces else "DOES NOT equal"
    lines.append(f"  sum over the orbits: {terms} ({verdict} the mode-space list)")
    return lines


# ---------------------------------------------------------------------------
# Born effective charges: mode effective charges and the dielectric tensor
# ---------------------------------------------------------------------------

def _real_eigenvectors(vectors, frequencies=None, tol: float = 1e-4) -> np.ndarray:
    """Real eigenvectors spanning the same Gamma eigenspaces.

    The dynamical matrix at Gamma is real, so every eigenspace has a real
    orthonormal basis. Each band first gets the global phase that makes its
    largest component real; a run of consecutive bands whose frequencies
    agree within ``tol`` THz (a degenerate subspace) that is still complex is
    replaced by the first left singular vectors of ``[Re V, Im V]``, a real
    orthonormal basis of the same subspace. A run whose real and imaginary
    parts span more than the run (not conjugation closed) is left as it is.
    """
    vectors = np.array(vectors, dtype=complex)
    for band in range(vectors.shape[1]):
        column = vectors[:, band]
        largest = column[np.argmax(np.abs(column))]
        if abs(largest) > 0:
            vectors[:, band] = column * (abs(largest) / largest)
    if frequencies is None:
        return vectors
    frequencies = np.asarray(frequencies, dtype=float)
    start = 0
    n_bands = vectors.shape[1]
    while start < n_bands:
        stop = start + 1
        while stop < n_bands and abs(frequencies[stop] - frequencies[start]) < tol:
            stop += 1
        block = vectors[:, start:stop]
        if np.abs(block.imag).max(initial=0.0) > 1e-10:
            size = stop - start
            stacked = np.hstack([block.real, block.imag])
            left, values, _ = np.linalg.svd(stacked, full_matrices=False)
            if np.sum(values > 1e-6) == size:
                vectors[:, start:stop] = left[:, :size]
        start = stop
    return vectors


def mode_effective_charges(phonon, eigenvectors=None, frequencies=None) -> np.ndarray:
    """Mode effective charges of the Gamma phonons.

    ``Z~_{m,alpha} = sum_{kappa beta} Z*_{kappa, alpha beta} e_m(kappa beta)
    / sqrt(M_kappa)`` with ``e_m`` the eigenvector of the dynamical matrix
    (the displacement is ``e / sqrt(M)``), ``M`` in amu and ``Z*`` the Born
    effective charges of ``phonon.nac_params`` (phonopy's convention:
    ``Z*[kappa, alpha, beta]`` with ``alpha`` the field and ``beta`` the
    displacement direction, atoms in the order of the primitive cell). The
    eigenvectors are made real first: the global phase of every eigenvector
    is fixed so that its largest component is real, and, when the
    frequencies are known, every degenerate subspace gets a real orthonormal
    basis (the dynamical matrix at Gamma is real), so that the charges are
    real and ``sum_m |Z~_m|^2`` over a degenerate set does not depend on the
    eigenvectors phonopy chose inside it.

    Args:
        phonon: A ``phonopy.Phonopy`` object with force constants and NAC
            parameters (loaded with a BORN file).
        eigenvectors: The Gamma eigenvectors as columns, shape
            ``(3 n_atoms, 3 n_atoms)``; ``phonon.irreps.eigenvectors`` when
            the irreps are set at Gamma, else computed (without the
            non-analytical term).
        frequencies: The frequencies (THz) of ``eigenvectors``, used to find
            the degenerate subspaces; taken with the eigenvectors when those
            are not given.

    Returns:
        The charges in e/sqrt(amu), shape ``(3 n_atoms, 3)``, in band order.

    Raises:
        ValueError: If ``phonon`` carries no Born effective charges.
    """
    nac = getattr(phonon, "nac_params", None)
    if not nac or nac.get("born") is None:
        raise ValueError("the phonopy object carries no Born effective charges (BORN)")
    if eigenvectors is None:
        irreps = getattr(phonon, "irreps", None)
        if irreps is not None and _is_gamma(irreps.qpoint):
            eigenvectors = irreps.eigenvectors
            frequencies = _irreps_frequencies(irreps)
        else:
            from .phonon_subgroups import _qpoints_result

            phonon.run_qpoints([[0.0, 0.0, 0.0]], with_eigenvectors=True)
            all_frequencies, all_vectors = _qpoints_result(phonon)
            eigenvectors, frequencies = all_vectors[0], all_frequencies[0]
    vectors = _real_eigenvectors(eigenvectors, frequencies)
    born = np.asarray(nac["born"], dtype=float)
    masses = np.asarray(phonon.primitive.masses, dtype=float)
    if born.shape != (len(masses), 3, 3):
        raise ValueError(
            f"Born charges of shape {born.shape} do not match the {len(masses)} "
            "atoms of the primitive cell"
        )
    displacements = vectors.reshape(len(masses), 3, -1) / np.sqrt(masses)[:, None, None]
    return np.einsum("kab,kbm->ma", born, displacements)


@dataclass(frozen=True, eq=False)
class DielectricSet:
    """The dielectric quantities of one degenerate set at Gamma.

    Attributes:
        activity: The :class:`Activity` of the set.
        charges: Mode effective charges (e/sqrt(amu)) of its bands, shape
            ``(n_bands, 3)``, from the real eigenvectors of
            :func:`mode_effective_charges`.
        oscillator_strength: ``S = sum_m Re(Z~_m Z~_m^*)`` over the optical
            bands of the set, 3x3 in e^2/amu (basis independent).
        contribution: ``Delta eps = DIELECTRIC_FACTOR * S / (Omega nu^2)``,
            3x3 (zero for an acoustic set).
    """

    activity: Activity
    charges: np.ndarray
    oscillator_strength: np.ndarray
    contribution: np.ndarray

    @property
    def charge_norm(self) -> float:
        """``sqrt(sum_m |Z~_m|^2)`` over all bands of the set (independent
        of the eigenvector basis inside a degenerate subspace)."""
        return float(np.sqrt(np.sum(np.abs(self.charges) ** 2)))


@dataclass(frozen=True, eq=False)
class DielectricResponse:
    """Static dielectric response of the Gamma phonons from Born charges.

    Attributes:
        eps_inf: Electronic dielectric tensor (BORN), 3x3.
        eps_static: ``eps_inf + sum of the set contributions``, 3x3.
        sets: One :class:`DielectricSet` per degenerate set, in band order.
        volume: Primitive-cell volume in Angstrom^3.
        born_sum: ``sum_kappa Z*_kappa`` (zero by the acoustic sum rule), 3x3.
        lst: ``(direction, product, ratio)`` of the Lyddane-Sachs-Teller
            check, or None: ``product`` is ``prod (nu_LO / nu_TO)^2`` over
            the optical bands with the non-analytical term for q -> 0 along
            ``direction`` (reduced reciprocal coordinates), ``ratio`` is
            ``q.eps_0.q / q.eps_inf.q``.
        imaginary_bands: 1-based bands of the optical modes with an
            imaginary frequency and a nonzero mode effective charge: they
            enter ``eps_static`` with a negative sign (``nu^2 < 0``), so that
            ``eps_static`` is then not the static dielectric tensor of a
            stable structure.
    """

    eps_inf: np.ndarray
    eps_static: np.ndarray
    sets: tuple
    volume: float
    born_sum: np.ndarray
    lst: tuple | None = None
    imaginary_bands: tuple[int, ...] = ()

    @property
    def born_sum_deviation(self) -> float:
        """Largest absolute element of ``sum_kappa Z*_kappa``."""
        return float(np.abs(self.born_sum).max())


def _lst_check(phonon, eps_inf, eps_static, direction):
    """``(direction, prod (nu_LO/nu_TO)^2, q.eps_0.q / q.eps_inf.q)`` or None.

    The generalized Lyddane-Sachs-Teller relation: the product over all
    optical bands of the squared frequencies with the non-analytical term
    (q -> 0 along ``direction``) over those without it equals the ratio of
    the static and electronic dielectric constants along q. The acoustic
    bands (largest translation weights) are left out of both products, and
    squared frequencies keep their sign (imaginary modes).
    """
    from .phonon_subgroups import _qpoints_result, _translation_weights

    results = []
    for q_direction in (None, direction):
        phonon.run_qpoints([[0.0, 0.0, 0.0]], with_eigenvectors=True,
                           nac_q_direction=q_direction)
        all_frequencies, all_vectors = _qpoints_result(phonon)
        frequencies = np.asarray(all_frequencies[0], dtype=float)
        weights = _translation_weights(phonon, all_vectors[0])
        optical = np.sort(np.argsort(weights)[:-3])
        squared = np.sign(frequencies[optical]) * frequencies[optical] ** 2
        if np.any(np.abs(squared) < 1e-6):
            return None
        results.append(squared)
    product = float(np.prod(results[1] / results[0]))
    reciprocal = np.linalg.inv(np.asarray(phonon.primitive.cell, dtype=float)).T
    q_cart = np.asarray(direction, dtype=float) @ reciprocal
    q_cart /= np.linalg.norm(q_cart)
    ratio = float(q_cart @ eps_static @ q_cart / (q_cart @ eps_inf @ q_cart))
    return (tuple(float(v) for v in direction), product, ratio)


def dielectric_response(phonon, activities=None, *, lst_direction=(1, 0, 0)) -> DielectricResponse:
    """Mode effective charges and the static dielectric tensor at Gamma.

    For every degenerate set the oscillator strength
    ``S_{alpha beta} = sum_m Z~_{m,alpha} Z~_{m,beta}`` and its contribution
    ``Delta eps = C S / (Omega nu^2)`` to the static dielectric tensor, with
    ``Omega`` the primitive-cell volume (Angstrom^3), ``nu`` the frequency
    (THz) and ``C = e^2 / (eps_vac amu) / (Angstrom^3 (2 pi THz)^2)
    = 4.4225e4`` (:data:`DIELECTRIC_FACTOR`, for ``S`` in e^2/amu);
    ``eps_0 = eps_inf + sum over the optical bands``. The acoustic bands of a
    set (the ``n`` largest rigid-translation weights, ``n`` their rounded
    sum) are left out. Also the acoustic sum rule of the Born charges and
    the Lyddane-Sachs-Teller check (:class:`DielectricResponse`).

    Args:
        phonon: A ``phonopy.Phonopy`` object with force constants, loaded
            with the NAC parameters (``is_nac=True`` and a BORN file).
        activities: The :class:`Activity` records of the Gamma sets from the
            same irreps run (``phonon.irreps`` still at Gamma); computed with
            :func:`gamma_mode_activities` when omitted.
        lst_direction: Direction of q -> 0 (reduced reciprocal coordinates)
            for the Lyddane-Sachs-Teller check; None skips it.

    Returns:
        A :class:`DielectricResponse`.

    Raises:
        ValueError: If ``phonon`` carries no Born effective charges.

    Example:
        >>> import phonopy
        >>> from crystod import phonon
        >>> ph = phonopy.load(unitcell_filename="221_PPOSCAR_ScF3",
        ...     force_sets_filename="FORCE_SETS", born_filename="BORN",
        ...     supercell_matrix=[4, 4, 4], primitive_matrix="auto", is_nac=True)
        >>> response = phonon.dielectric_response(ph)
        >>> round(float(response.eps_static[0, 0]), 3), round(response.lst[1], 3)
        (7.592, 3.324)
    """
    from .phonon_subgroups import _translation_weights

    nac = getattr(phonon, "nac_params", None)
    if not nac or nac.get("born") is None:
        raise ValueError("the phonopy object carries no Born effective charges (BORN)")
    if activities is None:
        activities = gamma_mode_activities(phonon)
    irreps = phonon.irreps
    frequencies = np.asarray(_irreps_frequencies(irreps), dtype=float)
    eigenvectors = _real_eigenvectors(irreps.eigenvectors, frequencies)
    charges = mode_effective_charges(phonon, eigenvectors, frequencies)
    weights = _translation_weights(phonon, eigenvectors)
    volume = float(abs(np.linalg.det(np.asarray(phonon.primitive.cell, dtype=float))))
    eps_inf = np.asarray(nac["dielectric"], dtype=float)
    sets = []
    imaginary = []
    total = np.zeros((3, 3))
    for record in activities:
        bands = [band - 1 for band in record.band_indices]
        n_acoustic = int(round(float(np.sum(weights[bands]))))
        by_weight = sorted(bands, key=lambda band: -weights[band])
        optical = sorted(by_weight[n_acoustic:])
        strength = np.zeros((3, 3))
        contribution = np.zeros((3, 3))
        for band in optical:
            outer = np.real(np.outer(charges[band], charges[band].conj()))
            strength += outer
            nu = frequencies[band]
            if nu < -1e-6 and np.linalg.norm(charges[band]) > 1e-3:
                imaginary.append(band + 1)
            if abs(nu) > 1e-6:
                contribution += DIELECTRIC_FACTOR * outer / (volume * np.sign(nu) * nu**2)
        total += contribution
        sets.append(
            DielectricSet(
                activity=record,
                charges=np.real(charges[bands]),
                oscillator_strength=strength,
                contribution=contribution,
            )
        )
    eps_static = eps_inf + total
    born_sum = np.sum(np.asarray(nac["born"], dtype=float), axis=0)
    lst = None
    if lst_direction is not None:
        lst = _lst_check(phonon, eps_inf, eps_static, lst_direction)
    return DielectricResponse(
        eps_inf=eps_inf,
        eps_static=eps_static,
        sets=tuple(sets),
        volume=volume,
        born_sum=born_sum,
        lst=lst,
        imaginary_bands=tuple(sorted(imaginary)),
    )


def format_dielectric_table(response: DielectricResponse) -> list[str]:
    """The lines ``crystod-phonon --irreps --nac`` prints after the activity
    summary: one row per Gamma set (irrep, frequency, activity, ``|Z~|``,
    the diagonal of ``Delta eps``), the diagonals of eps_inf, of the sum of
    the contributions and of eps_0, a note when imaginary optical modes
    with a dipole contribute (with a negative sign), the acoustic sum rule
    of the Born charges and the Lyddane-Sachs-Teller check.

    Args:
        response: The :class:`DielectricResponse`.

    Returns:
        The lines.
    """
    lines = [
        "Gamma-point dielectric response (Born effective charges from BORN):",
        f"  {'set':>3}  {'irrep':<14} {'freq (THz)':>10}  {'activity':<20} "
        f"{'|Z~| (e/sqrt(amu))':>18}  Delta eps (xx, yy, zz)",
    ]
    for index, item in enumerate(response.sets, start=1):
        record = item.activity
        label = "+".join(record.labels) or "?"
        activity = "+".join(record.activity)
        if record.acoustic:
            delta = "-"
        else:
            delta = "  ".join(f"{round(value, 4) + 0.0:8.4f}" for value in np.diag(item.contribution))
        frequency = record.frequency if abs(record.frequency) >= 5e-5 else 0.0
        lines.append(
            f"  {index:>3}  {label:<14} {frequency:10.4f}  {activity:<20} "
            f"{item.charge_norm:18.4f}  {delta}"
        )

    def diagonal(matrix) -> str:
        return "  ".join(f"{round(value, 4) + 0.0:8.4f}" for value in np.diag(matrix))

    lines.append(f"  eps_inf (diag):        {diagonal(response.eps_inf)}")
    lines.append(f"  sum Delta eps (diag):  {diagonal(response.eps_static - response.eps_inf)}")
    lines.append(f"  eps_0 (diag):          {diagonal(response.eps_static)}")
    if response.imaginary_bands:
        count = len(response.imaginary_bands)
        bands = " ".join(str(band) for band in response.imaginary_bands)
        lines.append(
            f"  note: {count} imaginary optical band(s) (bands {bands}) contribute with "
            "negative sign; eps_0 is not a static dielectric constant of an unstable "
            "structure"
        )
    lines.append(
        f"  sum_kappa Z*_kappa: max |deviation| = {response.born_sum_deviation:.2e} "
        "(acoustic sum rule)"
    )
    if response.lst is not None:
        direction, product, ratio = response.lst
        q = " ".join(f"{value:g}" for value in direction)
        lines.append(
            f"  LST check: prod (nu_LO/nu_TO)^2 = {product:.4f}, eps_0/eps_inf = {ratio:.4f} "
            f"(q -> 0 along [{q}])"
        )
    return lines
