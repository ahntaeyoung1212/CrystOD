"""
Symmetry-adapted phonon eigenvectors at one q point for crystod.

The shared solver of ``crystod-phonon --modulation``
(:class:`crystod.modulation.SymmetryAdaptedModulation`) and ``--vector``
(:func:`crystod.phonon_vector.build_symmetry_adapted_modes`), and, with no
dynamical matrix, of the structures ``--vibration`` writes
(:func:`solve_symmetry_adapted_spaces`). The dynamical
matrix of a phonopy object is block-diagonalized in the spgrep
irrep-projected basis of its primitive cell, so that the partners of a
degenerate level transform with the irrep matrices -- at a
time-reversal-invariant q, real patterns along directions that symmetry
operations fix up to sign -- instead of the arbitrary linear combinations a
plain eigensolver returns.

Conventions (phonopy's; everything runs on ``phonon.primitive`` as it is --
same atom order, positions, lattice and Cartesian frame, no
re-standardization):

* phonopy's dynamical matrix uses the atom-position phase convention: an
  eigenvector ``e`` at q describes the displacement
  ``u_j(R) = Re[A e_j exp(2 pi i q.(R + x_j))] / sqrt(m_j)`` of atom j in the
  cell at lattice translation R (``x_j`` the scaled position of atom j);
* the displacement representation of the little group of q
  (:meth:`crystod.vibration_modes.SymmetryOnlyVibrations.get_vibration_rep`)
  acts on those same vectors and commutes with the dynamical matrix exactly
  as phonopy returns it, so the matrix is used as is, never re-phased;
* the rows returned by spgrep's ``project_to_irrep`` are kets: for the
  ``(dim, 3 * n_atoms)`` array ``B`` of one irrep occurrence,
  ``Gamma(g) @ B.T == B.T @ d(g)`` with ``d`` the spgrep irrep matrices, so
  the dynamical matrix in that basis is ``conj(B) @ D @ B.T``.

Every occurrence of one irrep transforms with the same matrices, so all
occurrences form one cluster; the dynamical matrix couples them through a
scalar per pair (Schur), and the cluster reduces to a small generalized
Hermitian eigenproblem with the occurrence-overlap matrix (spgrep does not
orthogonalize repeated occurrences to each other).

The construction runs in the lattice gauge ``w_j = e_j exp(2 pi i q.x_j)``,
in which the displacement of atom j in the cell at R is
``Re[A w_j exp(2 pi i q.R)] / sqrt(m_j)``. At a time-reversal-invariant q
(``2q`` a reciprocal lattice vector) the dynamical matrix and the
representation are real in that gauge, and the partners of every degenerate
level are returned real (see :func:`solve_symmetry_adapted_modes`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cmp_to_key

import numpy as np
from numpy.typing import NDArray

from .spglib_compat import ensure_spglib_compat

ensure_spglib_compat()

import spglib
from phonopy.structure.atoms import PhonopyAtoms
from spgrep.representation import get_character, project_to_irrep

from .vibration_modes import SymmetryOnlyVibrations

# sqrt(eV/A^2/AMU) -> THz (VASP units); used when the phonopy object does not
# state its own conversion factor.
FREQUENCY_CONVERSION_THZ = 15.633302

# numerical tolerances of the construction (dynamical matrices in phonopy
# units, representation matrices and basis vectors of unit scale)
_SCHUR_TOLERANCE = 1e-6
_EIGEN_TOLERANCE = 1e-6
_REALITY_TOLERANCE = 1e-6
# the displacement-based tie-breaks compare products of unit-vector components,
# which carry the ~1e-8 noise of the eigenvectors: smaller differences are ties
_DISPLACEMENT_TOLERANCE = 1e-6


class NonPrimitiveCellError(ValueError):
    """The primitive cell of a phonopy object is not a primitive cell.

    Raised when the space group of ``phonon.primitive`` contains pure
    translations: the irreps of spgrep and the Bloch phases of the
    displacement representation are defined on a primitive cell only.
    """


def is_time_reversal_invariant(qpoint, atol: float = 1e-8) -> bool:
    """True when q equals -q up to a reciprocal lattice vector (2q integer)."""
    doubled = 2.0 * np.asarray(qpoint, dtype=float)
    return bool(np.allclose(doubled, np.rint(doubled), atol=atol))


@dataclass
class SymmetryAdaptedModes:
    """The symmetry-adapted modes at one q point, sorted by frequency.

    Attributes:
        qpoint: q in the reciprocal basis of ``phonon.primitive``.
        vibrations: The symmetry analysis of ``phonon.primitive`` as it is
            (``primitive_cell``, ``rotations``, ``translations``,
            ``rotations_cartesian``).
        irreps: The spgrep irreps of the little group of q.
        vibration_basis: spgrep's irrep-projected basis, one
            ``(dim, 3 * n_atoms)`` array per irrep occurrence; rows are kets
            in phonopy's (atom-position) phase convention.
        frequencies: Frequencies in THz, negative for imaginary modes.
        degeneracies: Dimension of the level each mode belongs to: the irrep
            dimension, or at a time-reversal-invariant q the physically
            irreducible dimension (twice the irrep dimension for a complex
            irrep paired with its conjugate by time reversal, or for the
            degenerate pair of levels of a pseudo-real irrep).
        eigenvectors: Unit eigenvectors of phonopy's dynamical matrix in
            phonopy's convention (mass-weighted, atom-position phase).
        lattice_vectors: The same vectors in the lattice gauge,
            ``w_j = e_j exp(2 pi i q.x_j)``; real at a time-reversal-invariant
            q.
        freezing_vectors: ``v_j = w_j / sqrt(m_j)`` normalized over the
            primitive cell: the displacement of atom j in the cell at lattice
            translation R is ``Re(v_j exp(2 pi i q.R))`` per unit amplitude.
        time_reversal_invariant: Whether ``2q`` is a reciprocal lattice
            vector.
        irrep_indices: Index into ``irreps`` of the irrep of each mode.
    """

    qpoint: NDArray[np.float64]
    vibrations: SymmetryOnlyVibrations
    irreps: list
    vibration_basis: list[NDArray[np.complex128]]
    frequencies: NDArray[np.float64]
    degeneracies: list[int]
    eigenvectors: list[NDArray[np.complex128]]
    lattice_vectors: list[NDArray[np.complex128]]
    freezing_vectors: list[NDArray[np.complex128]]
    time_reversal_invariant: bool
    irrep_indices: list[int] = field(default_factory=list)


@dataclass
class _Level:
    """One eigenvalue of one irrep cluster with its partner set (columns).

    ``sources`` are the indices, in the flat list of projected spaces (irreps
    in order, the occurrences of each in order), of the spaces the level
    stands for in the symmetry-only construction
    (:func:`solve_symmetry_adapted_spaces`, where every occurrence is one
    level until a complex irrep is merged with its conjugate); the columns
    belong to them in that order, an equal number each.
    """

    eigenvalue: float
    irrep_index: int
    vectors: NDArray[np.complex128]  # (3 * n_atoms, dim), lattice gauge
    degeneracy: int
    sources: tuple[int, ...] = ()


@dataclass
class _LittleGroup:
    """The little group of a time-reversal-invariant q, for isotropy subgroups.

    ``rotations``/``translations`` are the coset representatives the
    representation matrices belong to (primitive basis). At such a q a lattice
    translation L acts on every mode as ``exp(-2 pi i q.L) = +-1``; the
    translations acting as +1 form the lattice ``supercell`` (columns, in the
    primitive basis) of every frozen single-q structure, and ``half`` is one
    lattice vector acting as -1 (None at q = 0). ``setting_rotations`` are the
    rotations in spglib's conventional basis (``P R P^-1``), which does not
    depend on the primitive basis of the input, and ``inverse_axes`` inverts
    the matrix of unit vectors along the conventional axes (rows, in the
    frame of the input): the tie-breaks between equivalent choices use them,
    together with the atoms (``positions``, scaled, as the lattice gauge
    uses them) and ``parity`` (``2q`` mod 2: a lattice translation L acts as
    ``(-1)^(parity.L)``).
    """

    rotations: NDArray[np.int_]
    translations: NDArray[np.float64]
    lattice: NDArray[np.float64]
    symprec: float
    supercell: NDArray[np.int_]
    half: NDArray[np.int_] | None
    setting_rotations: NDArray[np.int_]
    inverse_axes: NDArray[np.float64]
    positions: NDArray[np.float64]
    parity: NDArray[np.int_]
    _pairs: tuple | None = field(default=None, repr=False)

    @classmethod
    def at(
        cls,
        qpoint,
        rotations,
        translations,
        lattice,
        symprec,
        transformation,
        inverse_axes,
        positions,
    ) -> "_LittleGroup":
        parity = np.rint(2.0 * np.asarray(qpoint, dtype=float)).astype(int) % 2
        supercell = np.eye(3, dtype=int)
        half = None
        if parity.any():
            # {L : q.L integer} = {L : parity.L even}: 2 e_k and e_i - parity_i e_k
            k = int(np.flatnonzero(parity)[0])
            for i in range(3):
                supercell[:, i] = 2 * supercell[:, k] if i == k else supercell[:, i]
            for i in range(3):
                if i != k:
                    supercell[k, i] = -parity[i]
            half = np.eye(3, dtype=int)[k]
        transformation = np.asarray(transformation, dtype=float)
        setting = [
            transformation @ rotation @ np.linalg.inv(transformation) for rotation in rotations
        ]
        return cls(
            rotations=np.asarray(rotations, dtype=int),
            translations=np.asarray(translations, dtype=float),
            lattice=np.asarray(lattice, dtype=float),
            symprec=symprec,
            supercell=supercell,
            half=half,
            setting_rotations=np.rint(np.array(setting)).astype(int),
            inverse_axes=np.asarray(inverse_axes, dtype=float),
            positions=np.asarray(positions, dtype=float),
            parity=parity,
        )

    def neighbour_pairs(self) -> tuple:
        """Every pair of different atoms at its shortest separations.

        For each ordered pair (i, j), i != j, the images ``x_j + L`` nearest to
        ``x_i`` (all of them when several are equally near,
        :func:`_nearest_images`): the arrays ``first``, ``second`` (atom
        indices) and ``signs`` (``(-1)^(parity.L)``, the factor that turns the
        lattice-gauge product ``w_i w_j`` into the product of the
        displacements of those two atoms), ordered by (i, j, the separation
        along the conventional axes). The pairs are physical
        (an origin shift, a rigid rotation or another primitive basis gives
        the same ones), and so are the signed products; with every pair
        present they fix a real pattern up to its sign. Computed once.
        """
        if self._pairs is not None:
            return self._pairs
        first, second, lattice_vectors = _nearest_images(
            self.positions, self.lattice, self.inverse_axes
        )
        signs = np.where((lattice_vectors @ self.parity) % 2, -1.0, 1.0)
        self._pairs = (first, second, signs)
        return self._pairs


def _nearest_images(
    positions: NDArray[np.float64],
    lattice: NDArray[np.float64],
    inverse_axes: NDArray[np.float64],
) -> tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]]:
    """Every ordered pair of different atoms at its shortest separations.

    For each pair (i, j), i != j, the images ``x_j + L`` nearest to ``x_i``
    (all of them when several are equally near), ordered by (i, j, the
    separation along the conventional axes): the atom indices ``first`` and
    ``second`` and the lattice vectors ``L`` (rows, primitive basis).
    """
    positions = np.asarray(positions, dtype=float)
    n_atoms = len(positions)
    offsets = np.array(list(np.ndindex(7, 7, 7)), dtype=float) - 3.0
    pairs = []
    for i in range(n_atoms):
        difference = positions - positions[i]
        translations = -np.floor(difference + 0.5)[:, None, :] + offsets[None, :, :]
        separations = (difference[:, None, :] + translations) @ lattice
        distances = np.linalg.norm(separations, axis=2)
        for j in range(n_atoms):
            if j == i:
                continue
            nearest = np.flatnonzero(distances[j] <= distances[j].min() + 1e-4)
            for index in nearest:
                lattice_vector = tuple(np.rint(translations[j, index]).astype(int))
                along_axes = tuple(np.round(separations[j, index] @ inverse_axes, 3))
                pairs.append((i, j, along_axes, lattice_vector))
    pairs.sort(key=lambda pair: pair[:3])
    return (
        np.array([pair[0] for pair in pairs], dtype=int),
        np.array([pair[1] for pair in pairs], dtype=int),
        np.array([pair[3] for pair in pairs], dtype=int).reshape(-1, 3),
    )


def _axes_components(vector: NDArray, inverse_axes: NDArray[np.float64]) -> NDArray[np.float64]:
    """Real displacements (n_atoms, 3) resolved along the conventional axes."""
    return np.asarray(vector).real.reshape(-1, 3) @ inverse_axes


def _partner_descriptor(vector: NDArray, group: "_LittleGroup") -> NDArray[np.float64]:
    """Origin-independent description of a real partner, blind to its sign.

    Three blocks of quadratic quantities, all read along the conventional axes
    (so a rigid rotation changes nothing): the displacement tensor
    ``sum_j u_j u_j^T``; the tensor ``u_j u_j^T`` of every atom, in atom
    order; and the products ``u_i u_j^T`` of the displacements of every pair
    of atoms at their shortest separations
    (:meth:`_LittleGroup.neighbour_pairs`), which together fix the pattern up
    to its sign. A sign of the partner, and the sign ``(-1)^(2q.n_j)`` an
    origin shift gives atom j when it moves it into another cell, cancel from
    every product, so equivalent partners (domains of one direction,
    identical in every symmetry key) are told apart the same way whatever the
    origin of the cell.
    """
    components = _axes_components(vector, group.inverse_axes)
    upper = np.triu_indices(3)
    per_atom = np.einsum("ja,jb->jab", components, components)
    first, second, signs = group.neighbour_pairs()
    pairs = signs[:, None, None] * np.einsum(
        "pa,pb->pab", components[first], components[second]
    )
    return np.concatenate(
        [per_atom.sum(axis=0)[upper], per_atom[:, upper[0], upper[1]].ravel(), pairs.ravel()]
    )


def _compare_descriptors(
    first: NDArray, second: NDArray, tolerance: float = _DISPLACEMENT_TOLERANCE
) -> int:
    """Lexicographic comparison, larger entries first; round-off ties are ties."""
    differences = np.flatnonzero(np.abs(first - second) > tolerance)
    if not len(differences):
        return 0
    return -1 if first[differences[0]] > second[differences[0]] else 1


def _relative_sign(
    vector: NDArray, reference: NDArray, group: "_LittleGroup"
) -> float | None:
    """Sign of ``vector`` against a real partner of the same level, or None.

    The products of the displacements of one atom in the two partners (then
    of pairs of atoms, :meth:`_LittleGroup.neighbour_pairs`), along the
    conventional axes: the first (near-)largest of them is made positive.
    An origin shift flips both factors of every product together, so the
    relative sign -- which decides the domain of a sum of partners at one q
    -- does not depend on the origin. None when every product vanishes.
    """
    components = _axes_components(vector, group.inverse_axes)
    reference_components = _axes_components(reference, group.inverse_axes)
    products = np.einsum("ja,jb->jab", reference_components, components).ravel()
    if np.abs(products).max() < _DISPLACEMENT_TOLERANCE:
        first, second, signs = group.neighbour_pairs()
        if not len(signs):
            return None
        products = (signs[:, None, None] * np.einsum(
            "pa,pb->pab", reference_components[first], components[second]
        )).ravel()
        if np.abs(products).max() < _DISPLACEMENT_TOLERANCE:
            return None
    return float(np.sign(products[_first_largest(np.abs(products))]) or 1.0)


def _integer_kernel_vector(matrix: NDArray[np.int_]) -> NDArray[np.int_] | None:
    """Primitive integer vector spanning the kernel of a rank-2 integer 3x3 matrix."""
    rows = [row for row in matrix if np.any(row)]
    for first in range(len(rows)):
        for second in range(first + 1, len(rows)):
            vector = np.cross(rows[first], rows[second])
            if np.any(vector):
                vector = vector // np.gcd.reduce(np.abs(vector[vector != 0]))
                # a direction, not an orientation: first nonzero entry positive
                return vector if vector[np.flatnonzero(vector)[0]] > 0 else -vector
    return None


def _isotropy_key(signs: list[int], group: _LittleGroup) -> tuple:
    """Setting-independent description of the isotropy subgroup of one partner.

    ``signs[i]`` is +1 or -1 when little-group operation i maps the partner
    onto plus or minus itself, 0 otherwise. At a time-reversal-invariant q a
    sign -1 is undone by the half translation, so the subgroup that leaves the
    frozen partner invariant is known exactly, without building a structure.
    The key is (minus the order of its point group, its space-group number,
    the lattice period along the axis of each rotation and the normal of each
    mirror): the space-group type of the subgroup and its orientation against
    the lattice, none of which depends on the origin, the orientation or the
    lattice basis of the input cell. Sorting by it puts the largest point
    group first and, among equal ones, the lowest space-group number.
    """
    inverse = np.linalg.inv(group.supercell)
    super_lattice = group.supercell.T @ group.lattice
    rotations, translations = [], []
    for sign, rotation, translation in zip(signs, group.rotations, group.translations):
        if sign == 0:
            continue
        if sign < 0:
            if group.half is None:
                continue
            translation = translation + group.half
        rotations.append(np.rint(inverse @ rotation @ group.supercell).astype(int))
        shifted = inverse @ translation
        translations.append(shifted - np.floor(shifted + 1e-9))
    try:
        found = spglib.get_spacegroup_type_from_symmetry(
            np.array(rotations, dtype="intc"),
            np.array(translations, dtype=float),
            lattice=super_lattice,
            symprec=group.symprec,
        )
        number = int(found.number) if found is not None else 0
    except Exception:  # noqa: BLE001 - the key is a tie-break, never fatal
        number = 0
    axes = []
    identity = np.eye(3, dtype=int)
    for rotation in rotations:
        determinant = int(round(np.linalg.det(rotation)))
        if np.array_equal(rotation, determinant * identity):
            continue  # identity and inversion have no axis
        axis = _integer_kernel_vector(rotation - determinant * identity)
        length = float(np.linalg.norm(axis @ super_lattice)) if axis is not None else 0.0
        axes.append((determinant, int(np.trace(rotation)), round(length, 3)))
    return (-len(rotations), number, tuple(sorted(axes)))


def _conversion_factor(phonon) -> float:
    factor = getattr(phonon, "unit_conversion_factor", None)
    try:
        factor = float(factor)
    except (TypeError, ValueError):
        return FREQUENCY_CONVERSION_THZ
    return factor if factor > 0 else FREQUENCY_CONVERSION_THZ


def _check_primitive(vibrations: SymmetryOnlyVibrations, n_atoms: int) -> None:
    """Raise NonPrimitiveCellError when the cell has pure translations."""
    identity = np.eye(3, dtype=int)
    n_translations = sum(
        1 for rotation in vibrations.rotations if np.array_equal(rotation, identity)
    )
    if n_translations > 1:
        raise NonPrimitiveCellError(
            f"the primitive cell of the phonopy object ({n_atoms} atoms) is not "
            f"primitive at the symmetry tolerance {vibrations.symprec:g}: its "
            f"symmetry contains {n_translations} pure translations, so a "
            f"{n_atoms // n_translations}-atom primitive cell exists. Build "
            'the phonopy object with primitive_matrix="auto" (on the command '
            "line: -c <unit cell> with FORCE_SETS, or a phonopy yaml written "
            "with PRIMITIVE_AXES = AUTO); if it already is, the translations "
            "are pseudo-symmetries of this tolerance: use a smaller one "
            "(--tolerance)."
        )


def _phase_by_largest_component(vector: NDArray[np.complex128]) -> complex:
    """Unit phase that makes the first (near-)largest component real positive.

    Components within 1e-6 of the largest magnitude count as ties and the
    first of them wins, so that round-off does not decide the choice.
    """
    magnitudes = np.abs(vector)
    largest = magnitudes.max()
    if largest < 1e-14:
        return 1.0 + 0.0j
    index = int(np.flatnonzero(magnitudes >= largest - 1e-6 * max(1.0, largest))[0])
    return np.conj(vector[index]) / magnitudes[index]


def _first_largest(values: NDArray[np.float64]) -> int:
    """Index of the first entry within 1e-6 of the largest one (round-off ties)."""
    largest = values.max()
    return int(np.flatnonzero(values >= largest - 1e-6 * max(1.0, largest))[0])


def _axes_sign(vector: NDArray[np.complex128], inverse_axes: NDArray[np.float64]) -> float:
    """Sign making the largest component of Re(vector) along the crystal axes positive.

    The displacement of every atom is resolved along the unit vectors of the
    conventional lattice vectors (``inverse_axes`` inverts their matrix, see
    :func:`_conventional_axes`), which neither a rigid rotation of the input
    cell nor another choice of its primitive basis changes; for a cell in its
    standard orientation these are the Cartesian components. The first
    (near-)largest component decides.
    """
    components = (vector.real.reshape(-1, 3) @ inverse_axes).ravel()
    if np.abs(components).max() < 1e-14:
        return 1.0
    return float(np.sign(components[_first_largest(np.abs(components))]) or 1.0)


def _conventional_axes(vibrations: SymmetryOnlyVibrations) -> NDArray[np.float64]:
    """Unit vectors (rows) along spglib's conventional axes, in the input's frame."""
    lattice = np.asarray(vibrations.primitive_cell.cell, dtype=float)
    transformation = np.asarray(vibrations.spglib_dataset["transformation_matrix"], dtype=float)
    # (a_s, b_s, c_s) = (a, b, c) P^-1, not idealized: the input's orientation
    conventional = np.linalg.inv(transformation).T @ lattice
    return conventional / np.linalg.norm(conventional, axis=1)[:, None]


def _bloch_phase(vector: NDArray[np.complex128], inverse_axes: NDArray[np.float64]) -> complex:
    """Unit phase fixing the global phase of a complex (Bloch) pattern.

    The reference is one atom: the first, in atom order, of the atoms with the
    (near-)largest displacement whose trajectory is not a circle. Its complex
    displacement w is turned so that ``w.w`` (not ``w^H w``) is real and
    positive: ``Re(w)`` is then the major axis of the ellipse the atom moves
    on, and the atom sits at the end of it in the cell at R = 0. Neither the
    atom nor the angle depends on the Cartesian frame or the lattice basis of
    the cell (an origin shift that moves the atom into another cell
    translates the structure by that lattice vector). The remaining sign (a
    shift of the modulation by half a period) makes the largest component of
    ``Re(w)`` along the crystal axes positive (:func:`_axes_sign`). When every
    atom moves on a circle, the largest component of the vector is made real
    and positive instead.
    """
    atoms = vector.reshape(-1, 3)
    norms = np.linalg.norm(atoms, axis=1)
    largest = norms.max()
    if largest < 1e-14:
        return 1.0 + 0.0j
    order = sorted(range(len(norms)), key=lambda atom: (-round(norms[atom] / largest, 6), atom))
    for atom in order:
        if norms[atom] < 1e-6 * largest:
            break
        square = atoms[atom] @ atoms[atom]
        if abs(square) > 1e-4 * norms[atom] ** 2:
            phase = np.exp(-0.5j * np.angle(square))
            return phase * _axes_sign(phase * atoms[atom], inverse_axes)
    return _phase_by_largest_component(vector)


def _symmetric_unitary_sqrt(matrix: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Symmetric unitary Z with Z @ Z == matrix, for a symmetric unitary matrix.

    The square root is taken as a function of the matrix (so it inherits the
    symmetry), after a global phase that keeps every eigenvalue away from the
    branch cut on the negative real axis.
    """
    import scipy.linalg

    if matrix.shape[0] == 1:
        return np.sqrt(matrix.astype(complex))
    angles = np.sort(np.angle(np.linalg.eigvals(matrix)))
    gaps = np.diff(np.concatenate([angles, [angles[0] + 2.0 * np.pi]]))
    widest = int(np.argmax(gaps))
    shift = np.pi - (angles[widest] + 0.5 * gaps[widest])  # widest gap onto -1
    schur_form, schur_vectors = scipy.linalg.schur(np.exp(1j * shift) * matrix, output="complex")
    root = (schur_vectors * np.sqrt(np.diag(schur_form))) @ schur_vectors.conj().T
    root = root * np.exp(-0.5j * shift)
    return 0.5 * (root + root.T)


def _takagi_real_basis(vectors: NDArray[np.complex128]) -> NDArray[np.complex128] | None:
    """Unitary Z making ``vectors @ Z`` real, for a conjugation-invariant span.

    ``vectors`` has orthonormal columns spanning a space that complex
    conjugation maps onto itself, so ``conj(vectors) = vectors @ K`` with K
    symmetric unitary; then Z Z^T = K gives ``conj(vectors @ Z) = vectors @ Z``
    (for one column, Z is the common phase ``exp(-i arg(v.v) / 2)``). Returns
    None when the span is not conjugation invariant.
    """
    conjugation = vectors.conj().T @ vectors.conj()
    if np.abs(vectors @ conjugation - vectors.conj()).max() > _REALITY_TOLERANCE:
        return None
    return _symmetric_unitary_sqrt(0.5 * (conjugation + conjugation.T))


def _real_form(irrep: NDArray[np.complex128]) -> NDArray[np.complex128] | None:
    """Unitary T with ``T^H d(g) T`` real for every g, or None.

    The irrep must be of real type: equivalent to its complex conjugate
    through an intertwiner U (``d(g) U = U conj(d(g))``) with
    ``U conj(U) = +1``, which makes U symmetric; T is the symmetric square
    root of U (then ``conj(T^H d T) = T^H d T``). Real matrices give the
    identity, so a real spgrep irrep is kept exactly as it is. Returns None
    for complex and pseudo-real irreps.
    """
    dim = irrep.shape[1]
    if np.abs(irrep.imag).max() < 1e-10:
        return np.eye(dim, dtype=complex)
    intertwiner = None
    for seed_index in range(dim * dim):
        seed = np.zeros((dim, dim), dtype=complex)
        seed[seed_index // dim, seed_index % dim] = 1.0
        # d(g) X conj(d(g))^-1 = d(g) X d(g)^T, averaged over the group
        averaged = np.einsum("gij,jk,glk->il", irrep, seed, irrep) / len(irrep)
        if np.linalg.norm(averaged) > 1e-6:
            intertwiner = averaged / np.sqrt(np.trace(averaged @ averaged.conj().T).real / dim)
            break
    if intertwiner is None:
        return None
    if not np.allclose(intertwiner @ intertwiner.conj(), np.eye(dim), atol=1e-8):
        return None  # pseudo-real: U conj(U) = -1, no real form exists
    root = _symmetric_unitary_sqrt(intertwiner)
    transformed = np.einsum("ji,gjk,kl->gil", root.conj(), irrep, root)
    if np.abs(transformed.imag).max() > 1e-8:
        return None
    return root


def _real_occurrences(
    occurrences: list[NDArray[np.complex128]],
) -> list[NDArray[np.complex128]] | None:
    """Real orthonormal occurrences spanning the same isotypic space.

    ``occurrences`` (rows are lattice-gauge kets) transform with one set of
    real irrep matrices and span a conjugation-invariant space; complex
    conjugation then acts on the occurrence coefficients only, identically
    for every partner. A Loewdin orthonormalization followed by a symmetric
    unitary (Takagi) transformation of the occurrences makes every partner
    real without mixing partners -- for a single occurrence, one common
    phase. Returns None when the span is not conjugation invariant.
    """
    first = np.array([occurrence[0] for occurrence in occurrences]).T
    overlap = first.conj().T @ first
    values, vectors = np.linalg.eigh(0.5 * (overlap + overlap.conj().T))
    if values.min() < 1e-8:
        return None
    inverse_sqrt = (vectors / np.sqrt(values)) @ vectors.conj().T
    mixing = _takagi_real_basis(first @ inverse_sqrt)
    if mixing is None:
        return None
    combination = inverse_sqrt @ mixing
    realified = [
        sum(combination[a, b] * occurrences[a] for a in range(len(occurrences)))
        for b in range(len(occurrences))
    ]
    if any(np.abs(occurrence.imag).max() > _REALITY_TOLERANCE for occurrence in realified):
        return None
    return [occurrence.real + 0j for occurrence in realified]


def _canonical_occurrences(
    occurrences: list[NDArray[np.complex128]],
    qpoint: NDArray[np.float64],
    positions: NDArray[np.float64],
    inverse_axes: NDArray[np.float64],
    pairs,
    real: bool,
) -> list[NDArray[np.complex128]]:
    """Orthonormal occurrences of one irrep chosen by the displacements, not by spgrep.

    ``occurrences`` (rows are lattice-gauge kets, all with the same irrep
    matrices) span the isotypic space of the irrep; which basis of it spgrep
    returns depends on its irrep matrices and on the unit vectors it happens
    to project, and so on how q is written (q, q + G, -q). Every combination
    ``sum_a c_a occurrences[a]`` with the same coefficients for all partners
    is again an occurrence, and for any operator X the matrix
    ``F_ab = sum_mu <occurrence_a[mu]| X |occurrence_b[mu]>`` depends only on
    the average of X over the little group (the sum over the partners of a
    unitary irrep is invariant), so its eigenvectors are a property of the
    isotypic space, whatever basis spgrep picked. The space is split by the
    eigenvalues of such matrices, larger first, each part that is still
    degenerate by the next one: the displacement norm of each atom in atom
    order (the orbit an occurrence lives on, in the order of the first atom
    of each orbit -- the order spgrep's projection gives them), the tensor
    of the displacement of each atom along the conventional axes, then the
    products of the displacements of the pairs of atoms at their shortest
    separations (``pairs()``: :func:`_nearest_images`, each with its Bloch
    factor ``exp(2 pi i q.L)``). None of these depends on the origin, the
    Cartesian frame or the lattice basis of the cell, or on how q is written.
    A part that no form separates any further keeps the orthonormal basis it
    has (the Loewdin combination of spgrep's occurrences where nothing splits
    the space at all): at a time-reversal-invariant q the forms of a
    pseudo-real irrep, which commute with its quaternionic structure, leave
    pairs of occurrences together, and such an irrep is then made real as a
    whole (:func:`_realify_remaining_levels`), which does not use this basis.
    The global phase (sign) of every occurrence is
    left to the conventions applied afterwards. ``real`` states that the
    occurrences are real, which keeps them real.
    """
    stacked = np.array(occurrences, dtype=complex)  # (multiplicity, dim, 3 * n_atoms)
    multiplicity, dim = stacked.shape[:2]
    overlap = np.einsum("aui,bui->ab", stacked.conj(), stacked) / dim
    overlap = 0.5 * (overlap + overlap.conj().T)
    if real:
        overlap = overlap.real
    values, vectors = np.linalg.eigh(overlap)
    if values.min() < 1e-8:
        raise RuntimeError("Irrep spaces of one irrep are linearly dependent.")
    inverse_sqrt = (vectors / np.sqrt(values)) @ vectors.conj().T
    stacked = np.einsum("ab,aui->bui", inverse_sqrt, stacked)
    if multiplicity == 1:
        return [stacked[0]]
    cartesian = stacked.reshape(multiplicity, dim, -1, 3)
    components = cartesian @ inverse_axes
    n_atoms = cartesian.shape[2]
    tensor_components = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))

    def splitting_forms(block, start):
        """(index, form on ``block``) of the forms from ``start`` on that split it.

        The forms are numbered in the order they are applied (atom norms,
        atom tensors, pair products) and built in batches restricted to the
        occurrences of ``block``; only those with an eigenvalue gap are
        returned, the first ones first.
        """
        block_cartesian = np.einsum("ab,auJk->buJk", block, cartesian)
        block_components = np.einsum("ab,auJk->buJk", block, components)

        def batches():
            yield np.arange(n_atoms), np.einsum(
                "iuJk,juJk->Jij", block_cartesian.conj(), block_cartesian
            )
            yield n_atoms + np.arange(6 * n_atoms), np.stack([
                np.einsum("iuJ,juJ->Jij", block_components[..., k].conj(),
                          block_components[..., l])
                for k, l in tensor_components
            ], axis=1).reshape(-1, block.shape[1], block.shape[1])
            weights = np.einsum("iuJk,iuJk->J", block_cartesian.conj(), block_cartesian).real
            support = weights > 1e-12 * weights.max()
            first, second, lattice_vectors = pairs()
            selected = np.flatnonzero(support[first] & support[second])
            phases = np.exp(2j * np.pi * (lattice_vectors @ np.asarray(qpoint, dtype=float)))
            for chunk in np.array_split(selected, max(1, len(selected) // 512)):
                if not len(chunk):
                    continue
                products = phases[chunk, None, None, None, None] * np.einsum(
                    "iuPk,juPl->Pklij",
                    block_components[:, :, first[chunk]].conj(),
                    block_components[:, :, second[chunk]],
                )
                indices = 7 * n_atoms + 9 * np.repeat(chunk, 9) + np.tile(np.arange(9), len(chunk))
                yield indices, products.reshape(-1, block.shape[1], block.shape[1])

        for indices, forms in batches():
            keep = indices >= start
            indices, forms = indices[keep], forms[keep]
            if not len(forms):
                continue
            forms = 0.5 * (forms + np.conj(np.swapaxes(forms, 1, 2)))
            if real:
                forms = forms.real
            gaps = np.diff(np.linalg.eigvalsh(forms), axis=1).max(axis=1)
            for position in np.flatnonzero(gaps > _DISPLACEMENT_TOLERANCE):
                yield int(indices[position]), forms[position]

    def split(block, start):
        if block.shape[1] == 1:
            return [block]
        for index, form in splitting_forms(block, start):
            values, rotation = np.linalg.eigh(form)
            values, rotation = values[::-1], rotation[:, ::-1]
            breaks = np.flatnonzero(np.abs(np.diff(values)) > _DISPLACEMENT_TOLERANCE) + 1
            if not len(breaks):
                continue
            # the earlier forms did not split the block: each part goes on from here
            return [
                piece
                for part in np.split(np.arange(len(values)), breaks)
                for piece in split(block @ rotation[:, part], index + 1)
            ]
        return [block]  # not split any further: keep the basis it has

    basis = np.hstack(split(np.eye(multiplicity, dtype=float if real else complex), 0))
    return list(np.einsum("ab,aui->bui", basis, stacked))


def _solve_irrep_cluster(
    occurrences: list[NDArray[np.complex128]],
    matrix: NDArray[np.complex128],
    real: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Eigenvalues and occurrence coefficients of one irrep cluster.

    ``occurrences`` are the occurrences of one irrep (rows are kets, all with
    the same irrep matrices). By Schur, the dynamical matrix and the overlap
    between two occurrences are scalar multiples of the identity; the cluster
    reduces to ``C c = lambda O c``. Returns the eigenvalues (ascending) and
    the coefficients as columns, normalized to ``c^H O c = 1`` so that every
    partner ``sum_a c_a occurrences[a][mu]`` has unit norm. ``real`` states
    that the occurrences and the matrix are real, which keeps the
    coefficients real as well.
    """
    dim = occurrences[0].shape[0]
    multiplicity = len(occurrences)
    scale = max(1.0, float(np.abs(matrix).max()))
    coupling = np.zeros((multiplicity, multiplicity), dtype=complex)
    overlap = np.zeros((multiplicity, multiplicity), dtype=complex)
    identity = np.eye(dim)
    for a in range(multiplicity):
        for b in range(multiplicity):
            # rows are kets: <a, mu| D |b, nu> = (conj(S_a) D S_b^T)[mu, nu]
            block = occurrences[a].conj() @ matrix @ occurrences[b].T
            gram = occurrences[a].conj() @ occurrences[b].T
            coupling[a, b] = np.trace(block) / dim
            overlap[a, b] = np.trace(gram) / dim
            if np.abs(block - coupling[a, b] * identity).max() > _SCHUR_TOLERANCE * scale:
                raise RuntimeError(
                    "Coupling between irrep spaces is not scalar: the symmetry "
                    "analysis does not match the dynamical matrix."
                )
            if np.abs(gram - overlap[a, b] * identity).max() > _SCHUR_TOLERANCE:
                raise RuntimeError("Overlap between irrep spaces is not scalar.")
    coupling = 0.5 * (coupling + coupling.conj().T)
    overlap = 0.5 * (overlap + overlap.conj().T)
    if real:
        coupling, overlap = coupling.real, overlap.real

    # Loewdin orthonormalization of the occurrences, then an ordinary
    # Hermitian eigenproblem: C c = lambda O c with X = O^(-1/2), c = X u
    overlap_values, overlap_vectors = np.linalg.eigh(overlap)
    if overlap_values.min() < 1e-8:
        raise RuntimeError("Irrep spaces of one irrep are linearly dependent.")
    inverse_sqrt = (overlap_vectors / np.sqrt(overlap_values)) @ overlap_vectors.conj().T
    reduced = inverse_sqrt @ coupling @ inverse_sqrt
    eigenvalues, rotations = np.linalg.eigh(0.5 * (reduced + reduced.conj().T))
    eigenvalues = eigenvalues.real

    # Preserve the symmetry-adapted basis when the cluster is numerically
    # degenerate: re-diagonalizing an exactly degenerate cluster can pick an
    # arbitrary rotated basis. The Loewdin-orthonormalized occurrences stay
    # as close as possible to the projected spaces.
    if multiplicity > 1 and np.allclose(eigenvalues, eigenvalues.mean(), atol=1e-10, rtol=1e-8):
        eigenvalues = np.full(multiplicity, eigenvalues.mean())
        rotations = np.eye(multiplicity)

    coefficients = np.asarray(inverse_sqrt @ rotations, dtype=complex)
    # the phase of an eigenvector is arbitrary (and LAPACK-dependent): fix it
    for column in range(multiplicity):
        coefficients[:, column] *= _phase_by_largest_component(coefficients[:, column])
    return eigenvalues, coefficients


def _group_by_eigenvalue(levels: list[_Level]) -> list[list[_Level]]:
    """Consecutive levels (one irrep, ascending) sharing one eigenvalue."""
    groups: list[list[_Level]] = []
    for level in levels:
        if groups and np.isclose(level.eigenvalue, groups[-1][0].eigenvalue, atol=1e-8, rtol=1e-8):
            groups[-1].append(level)
        else:
            groups.append([level])
    return groups


def _realify_group_as_whole(group: list[_Level]) -> _Level:
    """One real level from a degenerate group (partner structure not kept)."""
    stacked = np.hstack([level.vectors for level in group])
    mixing = _takagi_real_basis(stacked)
    if mixing is None:
        raise RuntimeError(
            "A degenerate level at a time-reversal-invariant q is not closed "
            "under complex conjugation."
        )
    realified = (stacked @ mixing).real + 0j
    return _Level(
        group[0].eigenvalue,
        group[0].irrep_index,
        realified,
        realified.shape[1],
        sum((level.sources for level in group), ()),
    )


def _realify_remaining_levels(
    levels: list[_Level],
    irreps: list[NDArray[np.complex128]],
    inverse_axes: NDArray[np.float64],
    wrap: NDArray[np.complex128],
) -> list[_Level]:
    """Real partners for the levels whose irrep has no real form.

    At a time-reversal-invariant q, a complex irrep and its conjugate carry
    the same levels (time reversal): the partner set W of the irrep and the
    one of its conjugate are merged into one level of twice the dimension,
    ``[sqrt(2) Re(W), sqrt(2) Im(W)]`` -- the partners of the physically
    irreducible (real) representation. The remaining non-real levels
    (pseudo-real irreps, whose levels come in degenerate pairs) are merged
    and made real per degenerate group, which keeps each level an eigenspace
    but not its partner structure. Levels that are already real pass
    through; the order of the list is kept.
    """
    by_irrep: dict[int, list[_Level]] = {}
    for level in levels:
        if np.abs(level.vectors.imag).max() > _REALITY_TOLERANCE:
            by_irrep.setdefault(level.irrep_index, []).append(level)
    characters = {index: get_character(irreps[index]) for index in by_irrep}

    replaced: dict[int, _Level | None] = {}  # id(level) -> merged level, None: dropped
    done: set[int] = set()
    for irrep_index, irrep_levels in by_irrep.items():
        if irrep_index in done:
            continue
        done.add(irrep_index)
        conjugate_index = next(
            (
                other
                for other in by_irrep
                if other not in done
                and np.allclose(characters[other], np.conj(characters[irrep_index]), atol=1e-6)
            ),
            None,
        )
        if conjugate_index is not None:
            partner_levels = by_irrep[conjugate_index]
            if len(partner_levels) == len(irrep_levels) and all(
                np.isclose(level.eigenvalue, partner.eigenvalue, atol=1e-8, rtol=1e-6)
                for level, partner in zip(irrep_levels, partner_levels)
            ):
                done.add(conjugate_index)
                for level, partner in zip(irrep_levels, partner_levels):
                    vectors = level.vectors * _bloch_phase(
                        level.vectors[:, 0] * wrap, inverse_axes
                    )
                    merged = np.sqrt(2.0) * np.hstack([vectors.real, vectors.imag]) + 0j
                    replaced[id(level)] = _Level(
                        level.eigenvalue,
                        level.irrep_index,
                        merged,
                        merged.shape[1],
                        level.sources + partner.sources,
                    )
                    replaced[id(partner)] = None
                continue
        for group in _group_by_eigenvalue(irrep_levels):
            replaced[id(group[0])] = _realify_group_as_whole(group)
            for level in group[1:]:
                replaced[id(level)] = None

    result: list[_Level] = []
    for level in levels:
        if id(level) not in replaced:
            result.append(level)
        elif replaced[id(level)] is not None:
            result.append(replaced[id(level)])
    return result


def _maximal_cliques(adjacency: list[set[int]]) -> list[frozenset[int]]:
    """Maximal cliques of a small undirected graph (Bron-Kerbosch)."""
    cliques: list[frozenset[int]] = []

    def expand(clique: set[int], candidates: set[int], excluded: set[int]) -> None:
        if not candidates and not excluded:
            cliques.append(frozenset(clique))
            return
        for vertex in sorted(candidates):
            expand(clique | {vertex}, candidates & adjacency[vertex], excluded & adjacency[vertex])
            candidates = candidates - {vertex}
            excluded = excluded | {vertex}

    expand(set(), set(range(len(adjacency))), set())
    return cliques


def _partner_signs(matrices: list[NDArray[np.float64]], column: int, tolerance: float) -> list[int]:
    """+1/-1 where the operation maps basis direction ``column`` onto +-itself, else 0."""
    signs = []
    for matrix in matrices:
        image = matrix[:, column]
        sign = int(np.sign(image[column])) if abs(abs(image[column]) - 1.0) < tolerance else 0
        if sign and np.abs(image - sign * np.eye(len(image))[column]).max() > tolerance:
            sign = 0
        signs.append(sign)
    return signs


def _canonical_partner_basis(
    vectors: NDArray[np.float64],
    representation: NDArray[np.float64],
    group: _LittleGroup,
) -> NDArray[np.float64]:
    """Real partners along symmetry-dictated directions, independent of setting.

    ``vectors`` (real orthonormal columns, lattice gauge) span one degenerate
    level and transform with real orthogonal matrices ``A(g)`` under the
    little-group operations; which orthonormal basis of the level spgrep's
    matrices single out depends on the origin of the cell (spgrep computes
    the irreps from the translation parts). The basis is therefore chosen
    from the level itself: the joint eigenvectors of a maximal set of
    commuting involutions (operations with ``R^2 = 1`` whose ``A(g)`` is
    symmetric and not a multiple of the identity) whose joint eigenspaces are
    all one-dimensional -- each such direction is fixed, up to sign, by those
    operations. Bases the group permutes up to sign (all partners equivalent,
    e.g. the cubic axes of a three-dimensional irrep) are preferred; among
    the rest, the basis whose single partners freeze into the most symmetric
    isotropy subgroups (:func:`_isotropy_key`: the largest point group, then
    the lowest space-group number, then the orientation against the
    lattice), and the partners are listed in that order. Both keys are
    properties of the frozen structures, so neither the choice nor the order
    depends on the origin, the Cartesian frame or the lattice basis of the
    cell. Partners whose subgroups are equivalent (domains of one direction,
    e.g. the x and y partners of a tetragonal E level) are ordered by the
    conventional axes (a, b, c) of spglib's setting where those tell them
    apart: by the axis along which they displace the atoms the most, else
    (when two axes tie) by the axis of the highest-order rotation that fixes
    them, then by the rotation parts of those operations. Where all of these
    coincide (domains whose symmetry elements have the same orientations but
    lie elsewhere, such as the two Pnma partners of an X level of diamond),
    the displacements decide (:func:`_partner_descriptor`: the displacement
    tensor, then the tensor of each atom in atom order, then the products of
    the displacements of pairs of atoms, the larger first), which again depends
    neither on the origin nor on the orientation of the cell -- never the
    order in which the eigensolver returns them, which follows the signs the
    lattice translations of the input origin give the operations. Which of
    the equivalent partners comes first is a convention. When no such set
    separates the level (a complex irrep merged with its conjugate, a
    pseudo-real irrep), the finest joint eigenspaces any set gives are kept
    and split further by the displacements (:func:`_split_by_displacements`):
    those partners are not along symmetry-dictated directions, but they too
    do not depend on the origin.
    """
    dim = vectors.shape[1]
    if dim < 2:
        return vectors
    tolerance = 1e-6
    rotations = group.setting_rotations
    matrices = [vectors.T @ rep @ vectors for rep in representation]
    identity = np.eye(3, dtype=int)
    order = sorted(
        range(len(rotations)), key=lambda index: tuple(np.asarray(rotations[index]).ravel())
    )
    keys: list[tuple[int, ...]] = []
    involutions: list[NDArray[np.float64]] = []
    for index in order:
        rotation = np.asarray(rotations[index], dtype=int)
        matrix = matrices[index]
        if np.array_equal(rotation, identity) or not np.array_equal(rotation @ rotation, identity):
            continue
        if np.abs(matrix - matrix.T).max() > tolerance:
            continue  # A(g)^2 = -1: no real eigenvectors
        if np.abs(matrix - np.trace(matrix) / dim * np.eye(dim)).max() < tolerance:
            continue
        if any(
            min(np.abs(matrix - other).max(), np.abs(matrix + other).max()) < tolerance
            for other in involutions
        ):
            continue
        keys.append(tuple(rotation.ravel()))
        involutions.append(matrix)

    def commute(first: NDArray[np.float64], second: NDArray[np.float64]) -> bool:
        return bool(np.abs(first @ second - second @ first).max() < tolerance)

    adjacency = [
        {
            other
            for other in range(len(involutions))
            if other != vertex and commute(involutions[vertex], involutions[other])
        }
        for vertex in range(len(involutions))
    ]
    best = None
    partial = None  # the finest split into joint eigenspaces, when none is complete
    for clique in _maximal_cliques(adjacency) if involutions else []:
        members = sorted(clique)
        # generic weights: the eigenvectors of the sum are the joint ones
        combined = sum(np.sqrt(2.0 + member) * involutions[member] for member in members)
        values, basis = np.linalg.eigh(0.5 * (combined + combined.T))
        if np.min(np.diff(values)) < tolerance:
            # a joint eigenspace is not one-dimensional
            breaks = np.flatnonzero(np.diff(values) >= tolerance) + 1
            blocks = np.split(np.arange(dim), breaks)
            rank = (-len(blocks), tuple(sorted(keys[member] for member in members)))
            if partial is None or rank < partial[0]:
                partial = (rank, [basis[:, block] for block in blocks])
            continue
        in_basis = [basis.T @ matrix @ basis for matrix in matrices]
        # does the group permute the basis directions (up to sign)?
        monomial = all(
            np.all(np.minimum(np.abs(matrix), np.abs(np.abs(matrix) - 1.0)) <= tolerance)
            for matrix in in_basis
        )
        partner_keys = [
            _isotropy_key(_partner_signs(in_basis, column, tolerance), group)
            for column in range(dim)
        ]
        rank = (
            not monomial,
            tuple(sorted(partner_keys)),
            tuple(sorted(keys[member] for member in members)),
        )
        if best is None or rank < best[0]:
            best = (rank, basis, partner_keys, in_basis)
    if best is None:
        # no commuting set separates the level (a complex irrep merged with its
        # conjugate, a pseudo-real irrep, a level without such operations):
        # keep the finest joint eigenspaces one gives, and split what is left
        # by the displacements -- a basis as independent of the origin as the
        # rest, though not along symmetry-dictated directions
        blocks = partial[1] if partial is not None else [np.eye(dim)]
        basis = np.hstack([
            block @ _split_by_displacements(vectors @ block, group) for block in blocks
        ])
        in_basis = [basis.T @ matrix @ basis for matrix in matrices]
        partner_keys = [
            _isotropy_key(_partner_signs(in_basis, column, tolerance), group)
            for column in range(dim)
        ]
    else:
        _, basis, partner_keys, in_basis = best

    # order the partners by their isotropy subgroups; equivalent ones (domains
    # of one direction) by the crystal axis they displace the atoms along
    # (a, b, c: the x, y, z partners of a cubic triplet in that order), else,
    # when that is not one axis, by the axis their subgroup's highest-order
    # rotation lies along (octahedral rotations about a, b, c), then by the
    # rotations that fix them
    def stabilizer(column: int) -> list[int]:
        signs = _partner_signs(in_basis, column, tolerance)
        # a sign -1 is undone by the half translation (q != 0 only)
        return [
            index
            for index, sign in enumerate(signs)
            if sign > 0 or (sign < 0 and group.half is not None)
        ]

    def axis_key(column: int) -> tuple[int, int]:
        best_key = (0, 3)
        for index in stabilizer(column):
            rotation = rotations[index]
            if round(np.linalg.det(rotation)) != 1 or np.array_equal(rotation, identity):
                continue
            axis = _integer_kernel_vector(rotation - identity)
            if axis is None or np.count_nonzero(axis) != 1:
                continue  # not along a conventional axis
            order = next(
                n for n in range(2, 7)
                if np.array_equal(np.linalg.matrix_power(rotation, n), identity)
            )
            best_key = min(best_key, (-order, int(np.flatnonzero(axis)[0])))
        return best_key

    def direction_key(column: int) -> int:
        # the crystal axis the partner displaces the atoms along the most,
        # when one axis clearly does (3 when two tie, as for a rotation)
        components = (basis[:, column] @ vectors.T).reshape(-1, 3) @ group.inverse_axes
        weights = np.sum(components**2, axis=0)
        first, second = np.argsort(-weights, kind="stable")[:2]
        if weights[first] - weights[second] <= 1e-6 * weights[first]:
            return 3
        return int(first)

    def stabilizer_key(column: int) -> tuple:
        return tuple(sorted(tuple(rotations[index].ravel()) for index in stabilizer(column)))

    keys_by_column = [
        (partner_keys[column], direction_key(column), axis_key(column), stabilizer_key(column))
        for column in range(dim)
    ]
    # partners that tie on every symmetry key are ordered by how they displace
    # the atoms, computed only when needed
    descriptors: dict[int, NDArray[np.float64]] = {}

    def descriptor(column: int) -> NDArray[np.float64]:
        if column not in descriptors:
            descriptors[column] = _partner_descriptor(basis[:, column] @ vectors.T, group)
        return descriptors[column]

    def compare(first: int, second: int) -> int:
        if keys_by_column[first] != keys_by_column[second]:
            return -1 if keys_by_column[first] < keys_by_column[second] else 1
        return _compare_descriptors(descriptor(first), descriptor(second))

    columns = sorted(range(dim), key=cmp_to_key(compare))
    return vectors @ basis[:, columns]


def _split_by_displacements(
    vectors: NDArray[np.float64], group: _LittleGroup
) -> NDArray[np.float64]:
    """Rotation of the columns of ``vectors`` onto a basis fixed by the displacements.

    The quadratic forms of :func:`_partner_descriptor` -- ``u_j . u_j`` of
    every atom in atom order, then ``u_i . u_j`` of every pair of atoms at
    their shortest separations, then, for what those leave degenerate, the
    products ``u_ja u_jb`` of the displacements of every atom along the
    conventional axes a, b (atoms that move on circles, as in the real space
    of a complex irrep and its conjugate, which the scalar products cannot
    tell apart) -- are origin-independent, so their eigenvectors are too:
    the space is split by the eigenvalues of the first form, each part that
    is still degenerate by the next form, and so on, larger eigenvalues
    first, until every part is one-dimensional (or the forms run out). A
    part a form does not split keeps its basis, which round-off in a
    degenerate eigensolution would otherwise turn at random. Used where
    symmetry does not single out a basis.
    """
    dim = vectors.shape[1]
    if dim < 2:
        return np.eye(dim)
    components = np.array(
        [_axes_components(vectors[:, column], group.inverse_axes) for column in range(dim)]
    )  # (dim, n_atoms, 3)

    def forms():
        for atom in range(components.shape[1]):
            yield components[:, atom, :] @ components[:, atom, :].T
        first, second, signs = group.neighbour_pairs()
        for i, j, sign in zip(first, second, signs):
            product = sign * components[:, i, :] @ components[:, j, :].T
            yield 0.5 * (product + product.T)
        for atom in range(components.shape[1]):
            for a, b in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)):
                product = np.outer(components[:, atom, a], components[:, atom, b])
                yield 0.5 * (product + product.T)

    blocks = [np.eye(dim)]
    for form in forms():
        if all(block.shape[1] == 1 for block in blocks):
            break
        refined = []
        for block in blocks:
            if block.shape[1] == 1:
                refined.append(block)
                continue
            values, rotation = np.linalg.eigh(block.T @ form @ block)
            values, rotation = values[::-1], rotation[:, ::-1]
            breaks = np.flatnonzero(np.abs(np.diff(values)) > _DISPLACEMENT_TOLERANCE) + 1
            if not len(breaks):
                refined.append(block)
                continue
            parts = np.split(np.arange(len(values)), breaks)
            refined.extend(block @ rotation[:, part] for part in parts)
        blocks = refined
    return np.hstack(blocks)


def _fix_partner_signs(
    vectors: NDArray[np.float64], group: _LittleGroup, wrap: NDArray[np.complex128]
) -> NDArray[np.float64]:
    """Signs of the real partners (columns) of one level at a time-reversal-invariant q.

    At Gamma every partner is signed on its own: its largest component along
    the crystal axes positive (:func:`_axes_sign`), which neither an origin
    shift nor a rigid rotation of the input changes. At any other such q,
    reversing a partner is the same as translating it by a lattice vector
    that acts as -1: the sign of the first partner only selects which of two
    translated copies is written (it is taken the same way, with the atoms in
    the cell [0, 1), so it depends on the origin), and every further partner
    is signed against the ones before it (:func:`_relative_sign`), so that
    the domain a sum of partners freezes into does not depend on the origin.
    """
    signed = np.array(vectors, dtype=float)
    for column in range(signed.shape[1]):
        sign = None
        if group.half is not None:
            for reference in range(column):
                sign = _relative_sign(signed[:, column], signed[:, reference], group)
                if sign is not None:
                    break
        if sign is None:
            sign = _axes_sign(signed[:, column] * wrap, group.inverse_axes)
        signed[:, column] *= sign
    return signed


def _symmetry_adapted_levels(
    vibrations: SymmetryOnlyVibrations,
    q: NDArray[np.float64],
    irreps: list,
    vibration_rep: NDArray[np.complex128],
    mapping_little_group: NDArray[np.int_],
    occurrences_by_irrep: list[list[NDArray[np.complex128]]],
    lattice_matrix: NDArray[np.complex128],
    symprec: float,
    canonical_occurrences: bool = False,
) -> list[_Level]:
    """The levels of every irrep cluster, partners fixed by the conventions.

    The construction shared by :func:`solve_symmetry_adapted_modes` (with the
    dynamical matrix) and :func:`solve_symmetry_adapted_spaces` (with a zero
    matrix): ``occurrences_by_irrep`` are spgrep's projected rows of each
    irrep (kets, atom-position phase convention) on
    ``vibrations.primitive_cell``, ``lattice_matrix`` the matrix in the
    lattice gauge. ``canonical_occurrences`` replaces the occurrences of a
    repeated irrep by the basis :func:`_canonical_occurrences` fixes from the
    displacements before the cluster is solved (for the zero matrix, where
    they are the levels). Returns the levels (vectors in the lattice gauge,
    columns the partners) in the order of the irreps and, within one irrep,
    of the eigenvalues; at a time-reversal-invariant q the levels of a
    complex or pseudo-real irrep are merged into real ones in place of the
    first of them.
    """
    positions = np.asarray(vibrations.primitive_cell.scaled_positions, dtype=float)
    phase = np.repeat(np.exp(2j * np.pi * (positions @ q)), 3)
    time_reversal = is_time_reversal_invariant(q)
    inverse_axes = np.linalg.inv(_conventional_axes(vibrations))
    # the sign and phase conventions read the vectors with every atom in the
    # cell [0, 1) up to round-off (w_j exp(-2 pi i q.n_j), n_j the integer
    # part): an atom written at 1 - 1e-12 instead of 0 is the same atom and
    # must not change the convention
    wrap = np.repeat(np.exp(-2j * np.pi * (np.floor(positions + 1e-6) @ q)), 3)
    nearest: list = []

    def pairs():
        if not nearest:
            nearest.append(_nearest_images(
                positions, np.asarray(vibrations.primitive_cell.cell, dtype=float), inverse_axes
            ))
        return nearest[0]

    levels: list[_Level] = []
    first_source = 0
    for irrep_index, occurrences in enumerate(occurrences_by_irrep):
        if not occurrences:
            continue
        lattice_occurrences = [occurrence * phase[None, :] for occurrence in occurrences]
        real = False
        if time_reversal:
            real_form = _real_form(np.asarray(irreps[irrep_index]))
            if real_form is not None:
                # columns B^T T transform with the real matrices T^H d T
                transformed = [real_form.T @ occurrence for occurrence in lattice_occurrences]
                realified = _real_occurrences(transformed)
                if realified is not None:
                    lattice_occurrences, real = realified, True
        if canonical_occurrences and len(lattice_occurrences) > 1:
            lattice_occurrences = _canonical_occurrences(
                lattice_occurrences, q, positions, inverse_axes, pairs, real
            )
        eigenvalues, coefficients = _solve_irrep_cluster(
            lattice_occurrences, lattice_matrix, real=real
        )
        dim = occurrences[0].shape[0]
        for column, eigenvalue in enumerate(eigenvalues):
            vectors = sum(
                coefficients[a, column] * lattice_occurrences[a].T
                for a in range(len(lattice_occurrences))
            )
            if real:
                vectors = vectors.real + 0j
            levels.append(
                _Level(float(eigenvalue), irrep_index, vectors, dim, (first_source + column,))
            )
        first_source += len(occurrences)

    if time_reversal:
        levels = _realify_remaining_levels(levels, irreps, inverse_axes, wrap)
        lattice_rep = (phase[None, :, None] * vibration_rep * phase.conj()[None, None, :]).real
        little_group = _LittleGroup.at(
            q,
            np.asarray(vibrations.rotations)[mapping_little_group],
            np.asarray(vibrations.translations)[mapping_little_group],
            vibrations.primitive_cell.cell,
            symprec,
            vibrations.spglib_dataset["transformation_matrix"],
            inverse_axes,
            vibrations.primitive_cell.scaled_positions,
        )
        for level in levels:
            vectors = _canonical_partner_basis(level.vectors.real, lattice_rep, little_group)
            level.vectors = _fix_partner_signs(vectors, little_group, wrap) + 0j
    else:
        # The global phase of a Bloch wave is a free parameter here (it shifts
        # the modulation along the lattice) and is fixed by convention only:
        # one common phase per partner set, which keeps the partners
        # transforming with the irrep matrices, taken from the first partner
        # with a rule that a rigid rotation of the input does not change and
        # an origin shift changes at most by a lattice translation (see
        # _bloch_phase) -- and the lattice-gauge vectors of q and q + G are
        # the same, so they give the same phase.
        for level in levels:
            convention = _bloch_phase(level.vectors[:, 0] * wrap, inverse_axes)
            level.vectors = level.vectors * convention
    return levels


def solve_symmetry_adapted_spaces(
    vibrations: SymmetryOnlyVibrations, qpoint, irrep_keys=None
) -> list[NDArray[np.complex128]]:
    """Symmetry-adapted partners of every projected space, without force data.

    The engine of ``crystod-phonon --vibration`` when it writes a structure:
    the construction of :func:`solve_symmetry_adapted_modes` with a zero
    dynamical matrix, on ``vibrations.primitive_cell`` as it is. With nothing
    to diagonalize, every occurrence of an irrep in the displacement
    representation is one level. Which basis of the isotypic space of a
    repeated irrep spgrep returns depends on its irrep matrices and on the
    unit vectors it happens to project, and so on how q is written; the
    occurrences are therefore replaced by an orthonormal basis fixed by the
    displacements (:func:`_canonical_occurrences`: the orbit each lives on,
    in atom order, then how the atoms move along the crystal axes, then the
    products of the displacements of neighbouring atoms), listed in that
    order. The partners of every space then follow the solver's
    conventions: at a time-reversal-invariant q (``2q`` a reciprocal lattice
    vector) they are real in the lattice gauge, orthonormal and turned onto
    the directions symmetry operations fix up to sign, chosen and ordered by
    the isotropy subgroups they freeze into (:func:`_canonical_partner_basis`)
    and signed by :func:`_fix_partner_signs`, so a single partner freezes
    into an isotropy subgroup of its irrep, whatever the origin, orientation
    and lattice basis of the cell. A complex irrep and its conjugate, which
    time reversal joins into one real space, share the partners of that
    space: the first half of them belongs to the space whose irrep comes
    first in ``irrep_keys`` (by default, in the order of the irreps). At any
    other q every partner set is a complex Bloch wave whose global phase is a
    convention (:func:`_bloch_phase`).

    At a time-reversal-invariant q the k-th space of an irrep therefore holds
    the same partners however q is written (q, q + G or -q), provided
    ``irrep_keys`` name the irreps the same way at each (ISO-IR labels do);
    the position of the irrep in the list follows spgrep's order at the q
    given, which can differ. At any other q the occurrences are fixed the
    same way, but the relative phases of the partners of a larger irrep
    follow the irrep matrices spgrep builds at the q given. Where the forms
    of :func:`_canonical_occurrences` leave a part of an isotypic space
    degenerate, its basis is the Loewdin combination of spgrep's copies.

    These are unit-norm symmetry-adapted displacement patterns, not normal
    modes: without force constants nothing selects a combination of the
    spaces of a repeated irrep, and no mass weighting enters.

    Args:
        vibrations: The symmetry analysis of the cell
            (:class:`crystod.vibration_modes.SymmetryOnlyVibrations`); the
            Bloch phases use the positions of its ``primitive_cell``, the cell
            the structures are built on.
        qpoint: q in the reciprocal basis of ``vibrations.primitive_cell``.
        irrep_keys: Optional sort keys, one per irrep of
            ``vibrations.get_vibration_rep(qpoint)`` (e.g. the natural-sorted
            ISO-IR labels): they decide which of a complex irrep and its
            conjugate receives the first half of the partners they share.

    Returns:
        One ``(dim, 3 * n_atoms)`` array per projected space, in the order,
        and with the dimensions, of
        :meth:`~crystod.vibration_modes.SymmetryOnlyVibrations.describe_mode_spaces`;
        rows are unit vectors in the lattice gauge ``w_j`` (the displacement
        of atom j in the cell at lattice translation R is
        ``Re(w_j exp(2 pi i q.R))`` per unit amplitude), real at a
        time-reversal-invariant q.

    Raises:
        RuntimeError: If the projection does not span the displacement space.
    """
    q = np.asarray(qpoint, dtype=float)
    irreps, vibration_rep, mapping_little_group = vibrations.get_vibration_rep(q)
    occurrences_by_irrep = [
        [np.asarray(space, dtype=complex) for space in project_to_irrep(vibration_rep, irrep)]
        for irrep in irreps
    ]
    n_dof = vibration_rep.shape[1]
    if sum(space.shape[0] for spaces in occurrences_by_irrep for space in spaces) != n_dof:
        raise RuntimeError("Irrep projection does not span the full vibration space.")
    levels = _symmetry_adapted_levels(
        vibrations, q, irreps, vibration_rep, mapping_little_group,
        occurrences_by_irrep, np.zeros((n_dof, n_dof), dtype=complex), vibrations.symprec,
        canonical_occurrences=True,
    )
    irrep_of_source = [
        index for index, spaces in enumerate(occurrences_by_irrep) for _ in spaces
    ]
    keys = list(irrep_keys) if irrep_keys is not None else list(range(len(irreps)))
    partners: list[NDArray[np.complex128] | None] = [None] * len(irrep_of_source)
    for level in levels:
        # the spaces of a level in the order of their irreps' keys (stable:
        # the occurrences of one irrep keep their order)
        sources = sorted(level.sources, key=lambda source: keys[irrep_of_source[source]])
        chunks = np.split(level.vectors, len(sources), axis=1)
        for source, chunk in zip(sources, chunks):
            rows = np.array(chunk.T, dtype=complex)
            partners[source] = rows / np.linalg.norm(rows, axis=1)[:, None]
    if any(rows is None for rows in partners):
        raise RuntimeError("A projected space was lost in the symmetry-adapted construction.")
    return partners


def solve_symmetry_adapted_modes(phonon, qpoint, symprec: float = 1e-5) -> SymmetryAdaptedModes:
    """Symmetry-adapted eigenvectors of phonopy's dynamical matrix at q.

    The displacement representation of the little group of q on
    ``phonon.primitive`` (as it is) is projected onto the spgrep irreps; all
    occurrences of one irrep form one cluster, solved as a generalized
    eigenproblem with the occurrence overlap, and every eigenvector is a
    combination ``sum_a c_a B_a[mu]`` of projected rows (kets), so that the
    partners of a degenerate level transform with the irrep matrices. The
    result is verified against phonopy: the frequencies must reproduce its
    spectrum, every vector must be an eigenvector of its dynamical matrix, and
    the vectors must be orthonormal.

    At a time-reversal-invariant q the partners of every degenerate level are
    real in the lattice gauge ``w_j = e_j exp(2 pi i q.x_j)`` and orthonormal,
    so a single partner is a real displacement pattern that freezes in at its
    full amplitude: the irrep is brought to a real form when spgrep's
    matrices are complex, and the occurrences are made real before the
    cluster is solved (one common phase per partner set for a single
    occurrence); a complex irrep and its time-reversal partner are combined
    into the real and imaginary parts of one partner set of twice the
    dimension. The partners are then turned onto directions that symmetry
    operations fix up to sign, chosen from the level itself and ordered by
    the isotropy subgroups they freeze into (see
    :func:`_canonical_partner_basis`), so that which directions come out, and
    in which order, depends neither on the origin, the Cartesian frame or the
    lattice basis of the cell, nor on how q is written (q and q + G give the
    same vectors); equivalent partners that no symmetry key separates are
    ordered by their displacements atom by atom, so there the order of the
    atoms counts. A partner that one of those operations reverses (possible
    at Gamma, where no lattice translation undoes the sign) freezes into a
    lower-symmetry direction than the one it keeps: not every partner is a
    special direction (the second partner of a Gamma E pair of a trigonal or
    hexagonal crystal can be the generic one). Which of several equivalent partners
    (domains of one direction) comes first, and the signs, are conventions
    (:func:`_fix_partner_signs`): at Gamma each partner has its largest
    component along the crystal axes positive; at any other such q a sign
    only picks one of two copies of the pattern translated by a lattice
    vector (which one depends on the origin), and the partners after the
    first are signed relative to the ones before, so that a sum of partners
    of one level freezes into the same domain for every origin. Nothing ties
    the signs of different q points (arms of a star) to each other.

    At any other q the global phase of a partner set is a convention only
    (see :func:`_bloch_phase`; deterministic, unchanged by a rigid rotation
    of the input and changed by an origin shift at most by a lattice
    translation, but not a physical choice): it shifts the modulation along
    the lattice and is not controlled, and a single partner of a degenerate
    level is a complex Bloch partner (circularly polarized in general) whose
    frozen structure depends on it, down to P1.

    Args:
        phonon: A ``phonopy.Phonopy`` object with force constants, built with
            ``primitive_matrix="auto"`` (its primitive cell must be primitive).
        qpoint: q in the reciprocal basis of ``phonon.primitive``.
        symprec: Symmetry tolerance of the spglib/spgrep analysis.

    Returns:
        A :class:`SymmetryAdaptedModes` record, modes sorted by frequency with
        a stable sort, so that mode i is band i of phonopy's sorted spectrum.

    Raises:
        NonPrimitiveCellError: If ``phonon.primitive`` is not primitive.
        ValueError: If the phonopy object carries no force constants.
        RuntimeError: If the construction does not reproduce phonopy's
            solution.
    """
    q = np.asarray(qpoint, dtype=float)
    dynamical_matrix = phonon.dynamical_matrix
    if dynamical_matrix is None:
        raise ValueError(
            "the phonopy object carries no force constants "
            "(FORCE_SETS/FORCE_CONSTANTS missing?)."
        )
    primitive = phonon.primitive
    cell = PhonopyAtoms(
        numbers=primitive.numbers,
        scaled_positions=primitive.scaled_positions,
        cell=primitive.cell,
    )
    n_atoms = len(cell.numbers)
    # standardize=False: symmetry, Bloch phases and Cartesian rotations in the
    # frame, origin and atom order of the dynamical matrix
    vibrations = SymmetryOnlyVibrations(cell=cell, symprec=symprec, standardize=False)
    _check_primitive(vibrations, n_atoms)
    irreps, vibration_rep, mapping_little_group = vibrations.get_vibration_rep(q)

    dynamical_matrix.run(q)
    matrix = np.array(dynamical_matrix.dynamical_matrix, dtype=complex)
    n_dof = matrix.shape[0]

    vibration_basis: list[NDArray[np.complex128]] = []
    occurrences_by_irrep: list[list[NDArray[np.complex128]]] = []
    for irrep in irreps:
        occurrences = [
            np.asarray(space, dtype=complex) for space in project_to_irrep(vibration_rep, irrep)
        ]
        occurrences_by_irrep.append(occurrences)
        vibration_basis.extend(occurrences)
    if sum(space.shape[0] for space in vibration_basis) != n_dof:
        raise RuntimeError("Irrep projection does not span the full vibration space.")

    # lattice gauge: w = phase * e; the matrix becomes phase_i D_ij conj(phase_j)
    phase = np.repeat(np.exp(2j * np.pi * (np.asarray(cell.scaled_positions, dtype=float) @ q)), 3)
    lattice_matrix = phase[:, None] * matrix * phase.conj()[None, :]
    time_reversal = is_time_reversal_invariant(q)
    levels = _symmetry_adapted_levels(
        vibrations, q, irreps, vibration_rep, mapping_little_group,
        occurrences_by_irrep, lattice_matrix, symprec,
    )

    factor = _conversion_factor(phonon)
    frequencies: list[float] = []
    degeneracies: list[int] = []
    irrep_indices: list[int] = []
    lattice_vectors: list[NDArray[np.complex128]] = []
    for level in levels:
        eigenvalue = level.eigenvalue
        frequency = float(np.sign(eigenvalue) * np.sqrt(abs(eigenvalue)) * factor)
        for component in range(level.vectors.shape[1]):
            frequencies.append(frequency)
            degeneracies.append(level.degeneracy)
            irrep_indices.append(level.irrep_index)
            lattice_vectors.append(np.array(level.vectors[:, component], dtype=complex))

    order = np.argsort(frequencies, kind="stable")
    frequencies = [frequencies[index] for index in order]
    degeneracies = [degeneracies[index] for index in order]
    irrep_indices = [irrep_indices[index] for index in order]
    lattice_vectors = [lattice_vectors[index] for index in order]
    eigenvectors = [phase.conj() * vector for vector in lattice_vectors]

    # Verify against the plain phonopy solution before trusting the result.
    reference = np.sort(np.linalg.eigvalsh(0.5 * (matrix + matrix.conj().T)).real)
    reference = np.sign(reference) * np.sqrt(np.abs(reference)) * factor
    if not np.allclose(frequencies, reference, atol=1e-3):
        raise RuntimeError("Symmetry-adapted frequencies do not match the phonopy spectrum.")
    residual_tolerance = _EIGEN_TOLERANCE * max(1.0, float(np.abs(matrix).max()))
    for frequency, vector in zip(frequencies, eigenvectors):
        eigenvalue = np.sign(frequency) * (frequency / factor) ** 2
        if np.linalg.norm(matrix @ vector - eigenvalue * vector) > residual_tolerance:
            raise RuntimeError(
                "A symmetry-adapted mode is not an eigenvector of the dynamical matrix."
            )
    stacked = np.array(eigenvectors)
    if np.abs(stacked.conj() @ stacked.T - np.eye(n_dof)).max() > 1e-6:
        raise RuntimeError("The symmetry-adapted modes are not orthonormal.")

    masses = np.repeat(np.asarray(primitive.masses, dtype=float), 3)
    freezing_vectors = []
    for vector in lattice_vectors:
        freezing = vector / np.sqrt(masses)
        freezing_vectors.append(freezing / np.linalg.norm(freezing))

    return SymmetryAdaptedModes(
        qpoint=q,
        vibrations=vibrations,
        irreps=list(irreps),
        vibration_basis=vibration_basis,
        frequencies=np.array(frequencies, dtype=float),
        degeneracies=degeneracies,
        eigenvectors=eigenvectors,
        lattice_vectors=lattice_vectors,
        freezing_vectors=freezing_vectors,
        time_reversal_invariant=time_reversal,
        irrep_indices=irrep_indices,
    )
