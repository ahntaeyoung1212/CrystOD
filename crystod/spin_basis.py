"""
Symmetry-adapted spin (magnetic) basis workflow for crystod.

Treats the spins on the sites of a selected element as axial-vector degrees
of freedom, decomposes them into irreps of the space group at a q-point by
projection (SALC), separates ferromagnetic (net-moment) from
antiferromagnetic (net-zero) combinations, assigns cluster-multipole ranks
(dipole, octupole, ...), and exports each basis vector as a VESTA file with
spin arrows.

The method follows the cluster multipole / symmetry-adapted multipole moment
(SAMM) framework of M.-T. Suzuki et al., Phys. Rev. B 95, 094406 (2017) and
Phys. Rev. B 99, 174407 (2019).
"""

from __future__ import annotations

from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    RawDescriptionHelpFormatter,
    RawTextHelpFormatter,
)
import warnings
from dataclasses import dataclass, field as dataclass_field
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray

from .spglib_compat import ensure_spglib_compat

ensure_spglib_compat()

from .runtime_compat import get_spacegroup_irreps_from_primitive_symmetry
from spgrep.representation import project_to_irrep

from .phonon_vector import _find_intertwiner, reduced_formula, write_vesta_with_arrows
from .vibration_modes import SymmetryOnlyVibrations


class MyHelpFormatter(
    RawTextHelpFormatter,
    RawDescriptionHelpFormatter,
    ArgumentDefaultsHelpFormatter,
):
    pass


desc = """
Construct symmetry-adapted spin bases (cluster multipoles / SAMM) for the
sites of a selected element: the 3N-dimensional axial-vector (spin) space is
decomposed into space-group irreps at a q-point, ferromagnetic (dipole) and
antiferromagnetic (net moment = 0) combinations are separated, and each basis
vector is exported as a VESTA file with spin arrows.

# Command Examples:
crystod-mag -c 221_PPOSCAR_AlNi3 --element Ni --qpoint 0 0 0
crystod-mag -c 221_PPOSCAR_AlNi3 --element Ni --qpoint GM --format qe
"""

#: Names of the magnetic multipole ranks used in the ``crystod-mag`` output
#: (rank ``p`` -> ``2**p``-pole): ``{1: "dipole", 3: "octupole", ...}``.
MULTIPOLE_NAMES = {
    0: "monopole",
    1: "dipole",
    2: "quadrupole",
    3: "octupole",
    4: "hexadecapole",
    5: "32pole",
    6: "64pole",
}


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=desc, formatter_class=MyHelpFormatter)
    parser.add_argument("--poscar", default="POSCAR", help="POSCAR path.")
    parser.add_argument(
        "--element",
        required=True,
        help="Magnetic element whose sites carry the spins, e.g. Ni.",
    )
    parser.add_argument(
        "--qpoint",
        nargs="+",
        default=None,
        help="Either a high-symmetry label such as GM/X/M/R or three primitive reciprocal coordinates. "
        "When omitted, the spin-multipole irreps at all special k points are listed (as in --salc).",
    )
    parser.add_argument(
        "--show-spin-direction",
        action="store_true",
        help="Print the spin direction [x, y, z] of every atom for each basis vector "
        "(with a noncollinear magnetization input in the selected --format).",
    )
    parser.add_argument(
        "--format",
        choices=["vasp", "qe"],
        default="vasp",
        help="Format of the noncollinear magnetization input printed with "
        "--show-spin-direction: vasp = MAGMOM line, qe = Quantum ESPRESSO "
        "starting_magnetization/angle1/angle2 (&SYSTEM).",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=1.5,
        help="Arrow length in Angstroms given to the largest spin of each basis vector.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Symmetry tolerance, a distance in Angstrom (spglib and site matching).",
    )
    parser.add_argument(
        "--conventional",
        action="store_true",
        help="Export the spin structures in the conventional cell instead of the "
        "primitive cell (primitive-to-conventional matrix from the detected "
        "centring, as in crystod-phonon --vector).",
    )
    parser.add_argument(
        "--classify-only",
        action="store_true",
        help="Print the structure, the decomposition and the table of magnetic "
        "groups and collinear classes only (no VESTA files, no spin listings).",
    )
    return parser


def _parse_coordinate(value: str) -> float:
    try:
        return float(Fraction(value))
    except ValueError:
        return float(value)


# ---------------------------------------------------------------------------
# Axial-vector (spin) representation on the selected sites
# ---------------------------------------------------------------------------

#: Fractional tolerance of the site matching before ``--tolerance`` was
#: threaded into it; a site within it still matches whatever the tolerance.
_SITE_MATCH_FRACTIONAL = 1e-5


def _match_site(transformed, candidates, lattice, tolerance: float) -> int | None:
    """Index into ``candidates`` of the site that ``transformed`` lands on, or None.

    The site nearest to ``transformed`` modulo lattice translations
    (Cartesian distance, the metric of spglib's ``symprec``), accepted when
    it lies within ``tolerance`` Angstrom of it or, as before ``--tolerance``
    reached the site matching, within ``_SITE_MATCH_FRACTIONAL`` in every
    fractional coordinate.
    """
    diff = np.asarray(candidates, dtype=float) - np.asarray(transformed, dtype=float)[None, :]
    residual = diff - np.rint(diff)
    distances = np.linalg.norm(residual @ np.asarray(lattice, dtype=float), axis=1)
    nearest = int(np.argmin(distances))
    if distances[nearest] <= tolerance or (
        np.abs(residual[nearest]) < _SITE_MATCH_FRACTIONAL
    ).all():
        return nearest
    return None


def _symmetrized_site_positions(vibrations: SymmetryOnlyVibrations, site_indices: list[int],
                                symprec: float | None = None) -> NDArray[np.float64]:
    """Fractional positions of the selected sites, symmetrized by the space group.

    Each site is replaced by the average of its preimages under all the
    operations (``x_a - mean_g R^-1 eps_g``, ``eps_g`` the mismatch of the
    image of site a on its matched site), the structure that the operations
    map exactly onto itself. The Bloch phases of the spin representation and
    the lattice gauge are built from these positions, so a structure within
    ``--tolerance`` of its ideal positions gives the representation, bases
    and spin fields of the ideal structure. The input positions are returned
    unchanged when the operations already map them onto each other (every
    mismatch below 1e-10) or when a site has no image within the tolerance.
    """
    positions = np.asarray(vibrations.primitive_cell.scaled_positions, dtype=float)[site_indices]
    lattice = np.asarray(vibrations.primitive_cell.cell, dtype=float)
    if symprec is None:
        symprec = float(getattr(vibrations, "symprec", 1e-5))
    shifts = np.zeros_like(positions)
    off_ideal = False
    rotations = np.asarray(vibrations.rotations)
    translations = np.asarray(vibrations.translations, dtype=float)
    for rotation, translation in zip(rotations, translations):
        inverse = np.linalg.inv(rotation)
        for a, position in enumerate(positions):
            transformed = rotation @ position + translation
            b = _match_site(transformed, positions, lattice, symprec)
            if b is None:
                return positions
            mismatch = transformed - positions[b]
            mismatch = mismatch - np.rint(mismatch)
            off_ideal = off_ideal or bool(np.abs(mismatch).max() > 1e-10)
            shifts[a] += inverse @ mismatch
    if not off_ideal:
        return positions
    return positions - shifts / len(rotations)


def get_spin_representation(
    vibrations: SymmetryOnlyVibrations,
    site_indices: list[int],
    qpoint: list[float],
    *,
    symprec: float | None = None,
):
    """Irreps and the axial-vector (spin) representation on selected sites at q.

    This is the first step of ``crystod-mag`` (at the ``--qpoint`` given, or at
    every special k point in survey mode): the spins of the ``site_indices``
    atoms of the primitive cell are treated as axial vectors, and the little
    group of ``qpoint`` acts on the 3N-dimensional spin space by the same
    construction as the vibration (polar-vector) representation of
    ``crystod-phonon``, with the Cartesian rotation ``R`` replaced by
    ``det(R) * R`` and the site permutation restricted to the selected sites.
    The Bloch-phase convention is that of spgrep, so the returned matrices can
    be projected onto the returned irreps directly with
    ``spgrep.representation.project_to_irrep``.

    Args:
        vibrations: Symmetry container of the primitive cell
            (``crystod.phonon.SymmetryOnlyVibrations``); its ``rotations``,
            ``translations`` and ``primitive_cell`` define the space group.
        site_indices: Indices (into ``vibrations.primitive_cell``) of the N
            atoms that carry a spin, e.g. every atom of the magnetic element.
        qpoint: The q point in primitive reciprocal coordinates, e.g.
            ``[0, 0, 0]`` or ``[0.5, 0.5, 0.5]``.
        symprec: Distance tolerance (Angstrom) within which an operation
            must map a site onto a site; None (default) takes the tolerance
            the space group was found with (``vibrations.symprec``, the
            ``--tolerance`` of ``crystod-mag``), so that every operation
            spglib accepts permutes the sites. Sites within 1e-5 in every
            fractional coordinate always match. The Bloch phases are built
            from the site positions symmetrized by the space group, so a
            structure off its ideal positions within the tolerance gives the
            representation of the ideal structure (for a structure at its
            ideal positions they are the input positions).

    Returns:
        tuple: ``(irreps, spin_rep, mapping)``. ``irreps`` is the list of spgrep
        irreps of the little group at ``qpoint`` (each an array of shape
        ``(order, dim, dim)``); ``spin_rep`` is the complex array of shape
        ``(order, 3N, 3N)`` holding the spin-representation matrix of every
        little-group operation, in the order of the irrep matrices; ``mapping``
        is the integer array that indexes those operations in
        ``vibrations.rotations`` (what ``get_irrep_labels`` and
        ``get_multipole_rank_lists`` expect).

    Example:
        >>> from phonopy.interface.vasp import read_vasp
        >>> from crystod import mag, phonon
        >>> from crystod.examples import example_path
        >>> cell = read_vasp(example_path("221_PPOSCAR_SrTiO3"))
        >>> vib = phonon.SymmetryOnlyVibrations(cell, standardize=False)
        >>> symbols = vib.primitive_cell.symbols
        >>> sites = [i for i, s in enumerate(symbols) if s == "O"]   # 3 sites
        >>> q = [0, 0, 0]
        >>> irreps, spin_rep, mapping = mag.get_spin_representation(vib, sites, q)
        >>> spin_rep.shape
        (48, 9, 9)
        >>> vib.get_irrep_labels(q, irreps, mapping)[6]
        'GM4+(3)'
    """
    positions = vibrations.primitive_cell.scaled_positions
    lattice = np.asarray(vibrations.primitive_cell.cell, dtype=float)
    if symprec is None:
        symprec = float(getattr(vibrations, "symprec", 1e-5))
    irreps, mapping = get_spacegroup_irreps_from_primitive_symmetry(
        rotations=vibrations.rotations,
        translations=vibrations.translations,
        kpoint=qpoint,
    )
    little_rotations = vibrations.rotations[mapping]
    little_translations = vibrations.translations[mapping]

    n_sites = len(site_indices)
    site_positions = np.asarray(positions, dtype=float)[site_indices]
    # the Bloch phases from the symmetrized positions (the input positions
    # for a structure at its ideal positions)
    ideal = _symmetrized_site_positions(vibrations, site_indices, symprec)
    representation = []
    for rotation, translation in zip(little_rotations, little_translations):
        permutation = np.zeros((n_sites, n_sites), dtype=complex)
        for a, index_a in enumerate(site_indices):
            transformed = rotation @ positions[index_a] + translation
            b = _match_site(transformed, site_positions, lattice, symprec)
            if b is None:
                continue
            phase = np.dot(
                qpoint,
                np.dot(np.linalg.inv(rotation), ideal[b] - translation) - ideal[b],
            )
            permutation[b, a] = np.exp(2j * np.pi * phase)
        representation.append(permutation)
    # Cartesian axial part per little-group operation
    lattice_t = np.transpose(vibrations.primitive_cell.cell)
    lattice_t_inv = np.linalg.inv(lattice_t)
    spin_rep = []
    for op_index, rotation in enumerate(little_rotations):
        cartesian = lattice_t @ rotation @ lattice_t_inv
        axial = np.linalg.det(cartesian) * cartesian
        spin_rep.append(np.kron(representation[op_index], axial))
    return irreps, np.array(spin_rep, dtype=complex), mapping


# ---------------------------------------------------------------------------
# Magnetic multipole ranks per irrep (Suzuki PRB 95, Table III logic)
# ---------------------------------------------------------------------------

def get_multipole_rank_lists(
    vibrations: SymmetryOnlyVibrations,
    little_rotations: NDArray[np.int_],
    irreps,
    max_rank: int = 9,
) -> list[list[int]]:
    """Ranks at which each irrep occurs in the magnetic multipole representations.

    ``crystod-mag`` uses these lists at q = 0 to name the symmetry-adapted spin
    bases: the spaces of an irrep are named in the order of the ranks in its
    list, so the ferromagnetic combination (cluster dipole, rank 1) comes
    first and the antiferromagnetic ones follow as octupole, ... (the logic of
    Table III of M.-T. Suzuki et al., Phys. Rev. B 95, 094406). The rank-``p``
    magnetic (time-odd, axial) multipole transforms as the angular-momentum-
    ``p`` representation with parity ``(-1)**(p + 1)`` under inversion: the
    dipole (``p = 1``) and octupole (``p = 3``) are parity-even, the magnetic
    quadrupole (``p = 2``) is parity-odd, and so on. The multiplicity of every
    irrep is obtained by character orthogonality against the rotation
    characters ``sin((p + 1/2) theta) / sin(theta / 2)`` of the proper part
    of every little-group operation.

    Args:
        vibrations: Symmetry container of the primitive cell
            (``crystod.phonon.SymmetryOnlyVibrations``); its lattice converts
            the integer rotations to Cartesian form.
        little_rotations: Integer rotation matrices of the little-group
            operations, in the order of the irrep matrices: normally
            ``vibrations.rotations[mapping]`` with the ``mapping`` returned by
            ``get_spin_representation``.
        irreps: The spgrep irreps of the little group, as returned by
            ``get_spin_representation``.
        max_rank: Highest rank ``p`` included in the search.

    Returns:
        One list per irrep (same order as ``irreps``), holding the ranks
        ``1 <= p <= max_rank`` at which the irrep appears in the rank-``p``
        magnetic multipole representation, in increasing order and repeated
        according to the multiplicity. ``MULTIPOLE_NAMES`` translates ranks
        to names.

    Example:
        >>> # vib, sites, q: see get_spin_representation (O sites of SrTiO3)
        >>> irreps, spin_rep, mapping = mag.get_spin_representation(vib, sites, q)
        >>> labels = vib.get_irrep_labels(q, irreps, mapping)
        >>> rotations = vib.rotations[mapping]
        >>> ranks = mag.get_multipole_rank_lists(vib, rotations, irreps)
        >>> dict(zip(labels, ranks))["GM4+(3)"]
        [1, 3, 5, 5, 7, 7, 9, 9, 9]
    """
    from .runtime_compat import get_character

    lattice_t = np.transpose(vibrations.primitive_cell.cell)
    lattice_t_inv = np.linalg.inv(lattice_t)
    operations = [lattice_t @ rotation @ lattice_t_inv for rotation in little_rotations]
    characters = [np.array(get_character(irrep), dtype=complex) for irrep in irreps]
    order = len(operations)

    rank_lists: list[list[int]] = [[] for _ in irreps]
    for p in range(1, max_rank + 1):
        chi_p = []
        for operation in operations:
            determinant = float(np.sign(np.linalg.det(operation)))
            proper = operation * determinant
            cos_theta = np.clip((np.trace(proper) - 1.0) / 2.0, -1.0, 1.0)
            theta = float(np.arccos(cos_theta))
            if theta < 1e-8:
                base = 2 * p + 1
            else:
                base = np.sin((p + 0.5) * theta) / np.sin(theta / 2.0)
            parity = ((-1) ** (p + 1)) if determinant < 0 else 1
            chi_p.append(parity * base)
        chi_p = np.array(chi_p)
        for index, character in enumerate(characters):
            count = int(round(float(np.real(np.dot(chi_p, np.conj(character)))) / order))
            rank_lists[index].extend([p] * max(count, 0))
    return rank_lists


# ---------------------------------------------------------------------------
# Basis post-processing
# ---------------------------------------------------------------------------

def _realify(space: NDArray[np.complex128]) -> NDArray[np.float64] | None:
    """Return a real orthonormal basis of the row space, or None when the
    space is genuinely complex.

    First each row is rotated by a global phase. When that fails (e.g. the
    circular-basis partners (x + iy, x - iy) of a 2-dim irrep), the rows are
    recombined unitarily: Gram-Schmidt over the real and imaginary parts gives
    a real basis whenever the space is closed under complex conjugation.
    Collapsing such a space with np.real instead would fold both partners onto
    the same vector and lose a dimension.
    """
    rows = []
    for row in space:
        pivot = row[np.argmax(np.abs(row))]
        aligned = row * np.exp(-1j * np.angle(pivot))
        if np.abs(aligned.imag).max() > 1e-8:
            rows = None
            break
        rows.append(aligned.real)
    if rows is not None:
        return np.array(rows)

    dim = space.shape[0]
    basis: list[NDArray[np.float64]] = []
    for row in space:
        for part in (np.real(row), np.imag(row)):
            residual = np.array(part, dtype=float)
            for vector in basis:
                residual -= np.dot(vector, residual) * vector
            norm = np.linalg.norm(residual)
            if norm > 1e-6:
                basis.append(residual / norm)
            if len(basis) == dim:
                break
        if len(basis) == dim:
            break
    if len(basis) < dim:
        return None
    candidate = np.array(basis)
    # the real basis must span the same complex space (closure under conjugation)
    projection = space @ candidate.T @ candidate
    if np.abs(projection - np.asarray(space)).max() > 1e-6:
        return None
    return candidate


def _net_moments(space: NDArray[np.float64], n_sites: int) -> NDArray[np.float64]:
    return np.array(
        [np.sum(space[c].reshape(n_sites, 3), axis=0) for c in range(space.shape[0])]
    )


def separate_ferro_combination(
    spaces: list[NDArray[np.float64]],
    representation: NDArray[np.complex128],
    n_sites: int,
    realify_aligned: bool = True,
) -> tuple[NDArray[np.float64] | None, list[NDArray[np.float64]]]:
    """Split the ferromagnetic combination off the spaces of one irrep.

    At q = 0 an irrep can occur several times in the spin representation
    (``2 x GM4+`` for the three Ni sites of AlNi3); ``crystod-mag`` calls this
    function once per irrep label with all of its projected spaces. The spaces
    are first aligned to a common partner basis (an intertwiner with
    ``representation`` is found for every space, and complex circular bases
    are recombined into real ones), then the net moments ``sum_i S_i`` of the
    aligned spaces are combined into one unit vector: its unitary completion
    gives the unique combination that carries a net moment (the cluster
    dipole, printed as ``FM``) and the orthogonal combinations with
    ``sum_i S_i = 0`` (``AFM``, net moment zero). ``crystod-mag`` names the
    AFM combinations by the remaining entries of the irrep's rank list
    (``get_multipole_rank_lists``): octupole, ... for an irrep even under the
    inversion, quadrupole, hexadecapole, ... for an odd one (Cr2O3, Cr:
    GM1- quadrupole, GM2- hexadecapole). When no space carries a net moment,
    or the completion does not separate cleanly, the aligned spaces are
    returned unchanged as antiferromagnetic.

    Args:
        spaces: Real basis arrays of shape ``(dim, 3N)`` (one row per partner
            of the irrep; ``3N`` = three Cartesian components of N spins) that
            all carry the same irrep, e.g. the output of
            ``spgrep.representation.project_to_irrep`` for one irrep.
        representation: The spin-representation matrices of shape
            ``(order, 3N, 3N)`` returned by ``get_spin_representation``; used
            to align the spaces to a common basis.
        n_sites: The number N of spin-carrying sites.
        realify_aligned: Rotate every aligned row by its own phase to make
            it real (needed for complex spaces). Pass False for spaces that
            are already real and transform with one set of real matrices,
            where a per-row phase would flip the sign of single partners.

    Returns:
        tuple: ``(ferro, afm)``. ``ferro`` is the basis array of shape
        ``(dim, 3N)`` with a net moment (``None`` when no combination has one)
        and ``afm`` is the list of the remaining net-zero basis arrays;
        together they span the same space as the input.

    Example:
        >>> # spin_rep, irreps, sites: see get_spin_representation (SrTiO3, O)
        >>> import numpy as np
        >>> from spgrep.representation import project_to_irrep
        >>> gm4p = irreps[6]                                     # GM4+, twice
        >>> spaces = [np.real(s) for s in project_to_irrep(spin_rep, gm4p)]
        >>> ferro, afm = mag.separate_ferro_combination(spaces, spin_rep, 3)
        >>> ferro.shape, len(afm)
        ((3, 9), 1)
        >>> bool(np.abs(afm[0].reshape(3, 3, 3).sum(axis=1)).max() < 1e-8)
        True
    """
    if len(spaces) == 1:
        nets = _net_moments(spaces[0], n_sites)
        if np.abs(nets).max() < 1e-8:
            return None, [spaces[0]]
        return spaces[0], []

    aligned = [spaces[0]]
    for space in spaces[1:]:
        intertwiner = _find_intertwiner(representation, space, spaces[0])
        if intertwiner is None:
            aligned.append(space)
        elif not realify_aligned:
            aligned.append(np.real(intertwiner.conj().T @ space))
        else:
            realified = _realify(intertwiner.conj().T @ space)
            aligned.append(realified if realified is not None else space)

    # net-moment magnitude of each space (same per component after alignment)
    net_vectors = np.array([_net_moments(space, n_sites)[0] for space in aligned])
    norms = np.linalg.norm(net_vectors, axis=1)
    if norms.max() < 1e-8:
        return None, aligned

    reference = net_vectors[np.argmax(norms)] / norms.max()
    coefficients = net_vectors @ reference  # signed magnitudes
    coefficients = coefficients / np.linalg.norm(coefficients)

    # unitary completion: first row = FM combination, others = AFM
    matrix = np.eye(len(aligned))
    matrix[0] = coefficients
    q_matrix, _ = np.linalg.qr(matrix.T)
    q_matrix = q_matrix.T
    if np.dot(q_matrix[0], coefficients) < 0:
        q_matrix[0] *= -1

    combos = [
        np.array([sum(q_matrix[m, a] * aligned[a][c] for a in range(len(aligned)))
                  for c in range(aligned[0].shape[0])])
        for m in range(len(aligned))
    ]
    ferro = combos[0]
    afm = []
    for combo in combos[1:]:
        if np.abs(_net_moments(combo, n_sites)).max() > 1e-6:
            # could not cleanly separate; keep as-is
            return None, aligned
        afm.append(combo)
    return ferro, afm


def _direction_tag(space_component: NDArray[np.float64], n_sites: int) -> tuple[str, float] | None:
    """(tag, sign) with an x/y/z/111-style tag when all spins of the component
    are collinear; sign canonicalizes the basis vector so that the common axis
    has its first nonzero Cartesian component positive."""
    vectors = space_component.reshape(n_sites, 3)
    norms = np.linalg.norm(vectors, axis=1)
    active = vectors[norms > 1e-8]
    if len(active) == 0:
        return None
    axis = active[0] / np.linalg.norm(active[0])
    for vector in active[1:]:
        cross = np.linalg.norm(np.cross(axis, vector / np.linalg.norm(vector)))
        if cross > 1e-6:
            return None
    # canonical sign: first nonzero component of the axis positive
    sign = 1.0
    for value in axis:
        if abs(value) > 1e-8:
            if value < 0:
                sign = -1.0
                axis = -axis
            break
    # nice integer direction
    scaled = axis / np.max(np.abs(axis))
    integers = []
    for value in scaled:
        fraction = Fraction(float(value)).limit_denominator(6)
        if abs(float(fraction) - value) > 1e-6:
            return None
        integers.append(fraction)
    denominator = np.lcm.reduce([f.denominator for f in integers])
    integers = [int(f * denominator) for f in integers]
    if integers == [1, 0, 0]:
        return "x", sign
    if integers == [0, 1, 0]:
        return "y", sign
    if integers == [0, 0, 1]:
        return "z", sign
    return "".join(str(v) for v in integers).replace("-", "m"), sign


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _print_qe_noncollinear(symbols: list[str], atom_spins: list, element: str) -> None:
    """Print pw.x noncollinear magnetization lines (&SYSTEM) for one basis vector.

    Quantum ESPRESSO specifies noncollinear moments per atom TYPE via
    starting_magnetization / angle1 / angle2, so the magnetic element is split
    into one type per distinct spin direction and the atom membership of each
    type is printed as a comment.
    """
    type_keys: list[tuple] = []
    type_of_atom: list[int] = []
    for symbol, spin in zip(symbols, atom_spins):
        if symbol == element:
            key = (symbol, tuple(np.round(spin, 4)))
        else:
            key = (symbol,)
        if key not in type_keys:
            type_keys.append(key)
        type_of_atom.append(type_keys.index(key) + 1)

    labels = []
    magnetic_counter = 0
    for key in type_keys:
        if len(key) == 2:
            magnetic_counter += 1
            labels.append(f"{element}{magnetic_counter}")
        else:
            labels.append(key[0])

    print("    Quantum ESPRESSO noncollinear magnetization (&SYSTEM):")
    print("      noncolin = .true.")
    for type_index, key in enumerate(type_keys, start=1):
        label = labels[type_index - 1]
        members = [
            f"{symbols[i]}{i + 1}"
            for i, atom_type in enumerate(type_of_atom)
            if atom_type == type_index
        ]
        if len(key) == 1:
            print(f"      ! type {type_index} ({label}): {' '.join(members)} -- non-magnetic")
            continue
        spin = np.array(key[1], dtype=float)
        magnitude = float(np.linalg.norm(spin))
        if magnitude < 1e-8:
            theta = phi = 0.0
        else:
            theta = float(np.degrees(np.arccos(np.clip(spin[2] / magnitude, -1.0, 1.0))))
            phi = float(np.degrees(np.arctan2(spin[1], spin[0])))
        print(
            f"      ! type {type_index} ({label}): {' '.join(members)}  "
            f"S = [{spin[0]:.4f}, {spin[1]:.4f}, {spin[2]:.4f}]"
        )
        print(f"      starting_magnetization({type_index}) = {magnitude:.4f}")
        print(f"      angle1({type_index}) = {theta:.4f}")
        print(f"      angle2({type_index}) = {phi:.4f}")
    print(
        "      ! angle1 = polar angle from the z axis (deg); "
        "angle2 = azimuth from the x axis in the xy plane (deg)."
    )
    print(
        f"      ! Split {element} into the types above in ATOMIC_SPECIES / "
        "ATOMIC_POSITIONS to realize this spin arrangement."
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.classify_only and args.qpoint is None:
        raise SystemExit("ERROR: --classify-only needs --qpoint.")

    from .star_of_k import read_poscar_or_exit

    cell = read_poscar_or_exit(args.poscar)
    vibrations = SymmetryOnlyVibrations(cell=cell, symprec=args.tolerance, standardize=False)
    try:
        # survey and --qpoint alike (spgrep refuses the operations of a
        # conventional cell or a supercell)
        _require_primitive(vibrations)
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from None
    primitive = vibrations.primitive_cell
    symbols = list(primitive.symbols)
    positions = primitive.scaled_positions

    site_indices = [i for i, s in enumerate(symbols) if s == args.element]
    if not site_indices:
        available = ", ".join(sorted(set(symbols)))
        raise SystemExit(f"ERROR: element '{args.element}' is not in this POSCAR ({available}).")
    n_sites = len(site_indices)

    print("\n* Structure *")
    print(f"  Space group: {vibrations.spglib_dataset['international']} "
          f"(#{vibrations.spglib_dataset['number']})")
    print(f"  Magnetic sites: {args.element} x {n_sites}")
    for k, index in enumerate(site_indices):
        print(f"    {args.element}{k + 1}: {np.round(positions[index], 6).tolist()}")

    if args.qpoint is None:
        # survey mode (as in --salc without --kpoint): spin-multipole irreps
        # at every special k point of the space group
        from phonopy.structure.cells import get_primitive_matrix_by_centring

        from .irreptables_compat import load_irreptables
        from .phonon_irreps import get_irt_special_points
        from .runtime_compat import get_character

        IrrepTable, _ = load_irreptables()
        dataset = vibrations.spglib_dataset
        irt_table = IrrepTable(dataset["number"], spinor=False)
        primitive_matrix = get_primitive_matrix_by_centring(dataset["international"][0])
        # named in the frame of the labels, the table's own basis being spglib's
        q_names, q_list = vibrations._special_points_in_label_frame(
            *get_irt_special_points(irt_table, primitive_matrix)
        )

        print(f"\n* Spin (axial-vector) Irreducible Representations of {args.element} sites *")
        for name, q in zip(q_names, q_list):
            irreps, spin_rep, mapping = get_spin_representation(vibrations, site_indices, q)
            labels = vibrations.get_irrep_labels(q, irreps, mapping)
            rep_characters = np.array([np.trace(matrix) for matrix in spin_rep])
            counts = {}
            for index, irrep in enumerate(irreps):
                characters = np.array(get_character(irrep), dtype=complex)
                count = int(round(float(np.real(np.dot(rep_characters, np.conj(characters)))) / len(spin_rep)))
                if count > 0:
                    counts[index] = count
            # a complex irrep and its conjugate at a time-reversal-invariant
            # point: one physically irreducible space, under the pair label
            # (as with --qpoint)
            pairs = (
                _conjugate_pairs(irreps, labels, set(counts))
                if np.allclose(2.0 * np.asarray(q), np.rint(2.0 * np.asarray(q))) else {}
            )
            seconds = set(pairs.values())
            terms = []
            for index, count in counts.items():
                if index in seconds:
                    continue
                label = labels[index]
                if index in pairs:
                    label = _pair_label(label, labels[pairs[index]])
                terms.append(f"{count}.0 [{label}]")
            print(f"  k point (primitive): {name} {np.round(q, 6).tolist()}")
            print(f"    {' + '.join(terms)}")
        print("\nUse --qpoint to obtain the symmetry-adapted spin bases, MAGMOM lines, and VESTA files at one k point.")
        return

    try:
        analysis = analyze_spin_basis(
            element=args.element,
            qpoint=args.qpoint,
            symprec=args.tolerance,
            conventional=args.conventional,
            vibrations=vibrations,
        )
    except ValueError as exc:
        raise SystemExit(f"ERROR: {' '.join(str(exc).split())}") from None
    _print_analysis(analysis, args)


def _print_analysis(analysis: "SpinBasisAnalysis", args) -> None:
    """Print the q-point part of the crystod-mag report from an analysis."""
    from .magnetic_group import (
        ALTERMAGNET_REFERENCE,
        format_magnetic_group,
        format_magnetic_point_group,
    )

    element = analysis.element
    n_sites = analysis.n_sites
    is_gamma = analysis.is_gamma
    print("\n* Selected Q point *")
    print(f"  Selected q-point: {analysis.qpoint_label} = {analysis.qpoint}")

    print("\n* Spin (axial-vector) representation *")
    print(f"  Spin (axial-vector) representation on {element} sites: "
          f"{3 * n_sites} dimensions")
    print("  Decomposition: " + " + ".join(
        (f"{count} x {label}" if count > 1 else label)
        for label, count in analysis.decomposition.items()
    ))
    if analysis.total_dim != 3 * n_sites:
        print(f"  WARNING: projected dimensions ({analysis.total_dim}) do not span the full space.")

    print("\n* Symmetry-adapted spin bases *")
    for label, kind, rank_name, space in analysis.entries:
        rank_text = f", {rank_name}" if rank_name else ""
        print(f"  {label}: dim {space.shape[0]} [{kind}{rank_text}]")

    supercell_sizes = analysis.supercell_sizes
    if not is_gamma:
        print("\n* Magnetic cell *")
        print(
            f"  Magnetic (commensurate) supercell: "
            f"{supercell_sizes[0]}x{supercell_sizes[1]}x{supercell_sizes[2]}"
        )
    if analysis.conventional_centring is not None:
        multiples = analysis.conventional_multiples
        if is_gamma:
            print("\n* Magnetic cell *")
        print(
            f"  Display cell: conventional ({analysis.conventional_centring} centring), "
            f"{multiples[0]}x{multiples[1]}x{multiples[2]} cells"
        )
    if not is_gamma:
        print("  the magnetic space group refers to the real spin field of the magnetic "
              "cell (global phase fixed as the VESTA export)")

    symbols = analysis.symbols
    supercell_symbols = analysis.supercell_symbols
    formula = analysis.formula
    written: list[str] = []
    for entry_index, (label, kind, rank_name, space) in enumerate(analysis.entries):
        if args.classify_only:
            break
        header = f"{label} [{kind}" + (f", {rank_name}]" if rank_name else "]")
        print(f"\n* {header} *")
        for component in analysis.components:
            if component.entry_index != entry_index:
                continue
            tag = component.tag
            field = component.field
            if field is None:
                print(f"  component {tag}: vanishing real spin field; skipped.")
                continue

            print(f"  component {tag}:")
            for k in range(n_sites):
                s = field[0, k]
                print(f"    {element}{k + 1}: S = [{s[0]: .4f}, {s[1]: .4f}, {s[2]: .4f}]")
            net = component.net_moment
            if is_gamma:
                print(f"    net moment: {np.round(net, 6).tolist()}")
            else:
                print(f"    net moment over the magnetic supercell: {np.round(net, 6).tolist()}")
            if component.collinear is not None:
                group = component.group
                if group is None:
                    print("    magnetic space group: not identified by spglib")
                else:
                    print(f"    magnetic space group: {format_magnetic_group(group)}")
                    print(f"    magnetic point group: {format_magnetic_point_group(group)}")
                if component.responses is not None:
                    print(f"    allowed by symmetry: {component.responses.summary()}")
                print(f"    collinear order: {component.collinear.text()}")

            if args.show_spin_direction:
                magmom_parts = []
                atom_spins = []
                for atom_index, site in enumerate(analysis.supercell_site_of_atom):
                    if site is None:
                        magmom_parts.append("0 0 0")
                        atom_spins.append(np.zeros(3))
                    else:
                        cell_index = atom_index // len(symbols)
                        s = field[cell_index, site]
                        magmom_parts.append(f"{s[0]:.4f} {s[1]:.4f} {s[2]:.4f}")
                        atom_spins.append(np.array(s, dtype=float))
                if args.conventional:
                    cell_note = " (conventional cell)" if is_gamma else " (conventional magnetic cell)"
                else:
                    cell_note = "" if is_gamma else f" ({supercell_sizes.tolist()} magnetic supercell)"
                print(f"    spin directions (all atoms, POSCAR order){cell_note}:")
                for atom_index, s_text in enumerate(magmom_parts):
                    print(f"      {supercell_symbols[atom_index]}{atom_index + 1}: "
                          f"[{s_text.replace(' ', ', ')}]")
                if args.format == "qe":
                    _print_qe_noncollinear(supercell_symbols, atom_spins, element)
                else:
                    print(f"    MAGMOM = {'   '.join(magmom_parts)}")

            # VESTA export (file name fixed by analyze_spin_basis)
            rank_tag = rank_name if rank_name else ("FM" if kind == "FM" else analysis.qpoint_label)
            name = component.vesta_name

            arrows = np.zeros((len(supercell_symbols), 3))
            for atom_index, site in enumerate(analysis.supercell_site_of_atom):
                if site is not None:
                    cell_index = atom_index // len(symbols)
                    arrows[atom_index] = field[cell_index, site] * args.amplitude
            write_vesta_with_arrows(
                filepath=name,
                lattice=analysis.supercell_lattice,
                scaled_positions=analysis.supercell_positions,
                symbols=supercell_symbols,
                arrows_cartesian=arrows,
                title=f"{formula} spin basis: {label} {rank_tag} {tag}",
            )
            written.append(name)

    print()
    print("\n".join(format_magnetic_summary(analysis)))

    if written:
        print("\n* Output files *")
        for name in written:
            print(f"  Spin basis written to: {name}")
    print()
    afm_net_zero = all(
        np.linalg.norm(c.net_moment) < 1e-6
        for c in analysis.components if c.kind == "AFM" and c.field is not None
    )
    if is_gamma and afm_net_zero:
        print("All AFM bases satisfy sum_i S_i = 0; FM entries carry the cluster dipole moment.")
    print("Reference: M.-T. Suzuki et al., PRB 95, 094406 (2017); PRB 99, 174407 (2019).")
    print(f"Reference (collinear classes): {ALTERMAGNET_REFERENCE}.")


# ---------------------------------------------------------------------------
# Data layer: the q-point analysis as data (CLI, API and MCP tool)
# ---------------------------------------------------------------------------

def _all_finite(ferro, afm_list) -> bool:
    arrays = ([] if ferro is None else [ferro]) + list(afm_list)
    return all(np.all(np.isfinite(np.asarray(array))) for array in arrays)


_PARTNER_TOLERANCE = 1e-6


def _separation_consistent(ferro, afm_list, n_sites: int) -> bool:
    """Finite bases, a net moment on every FM row and none on any AFM row."""
    if not _all_finite(ferro, afm_list):
        return False
    if any(np.abs(_net_moments(np.real(afm), n_sites)).max() > _PARTNER_TOLERANCE
           for afm in afm_list):
        return False
    if ferro is not None:
        nets = _net_moments(np.real(ferro), n_sites)
        if (np.linalg.norm(nets, axis=1) < _PARTNER_TOLERANCE).any():
            return False
    return True


def _partner_matrices(space, representation) -> NDArray[np.float64]:
    """Real matrices ``S G(g) S^T`` of the operations on the partners (rows) of ``space``.

    The rows are taken at unit norm (the partners of one copy are orthogonal
    with a common norm, which a combination of non-orthogonal copies of the
    irrep need not make 1).
    """
    space = np.real(np.asarray(space))
    norms = np.linalg.norm(space, axis=1)
    space = space / np.where(norms > 1e-12, norms, 1.0)[:, None]
    return np.einsum("ui,gij,vj->guv", space, np.real(representation), space)


def _partners_fixed_by_symmetry(space, representation) -> bool:
    """True when every partner of a real space lies along a direction symmetry fixes.

    A partner (row) qualifies when some operation maps it onto plus or minus
    itself without acting as a multiple of the identity on the whole space
    (a two-fold axis or a mirror of a hexagonal E irrep, the four-fold axis
    of a cubic E irrep); the partners of a 45-degree mixture of such
    directions do not. One-dimensional spaces always qualify.
    """
    space = np.asarray(space)
    dim = space.shape[0]
    if dim < 2:
        return True
    if np.abs(np.imag(space)).max() > 1e-8:
        return False
    tolerance = _PARTNER_TOLERANCE
    identity = np.eye(dim)
    matrices = [
        matrix for matrix in _partner_matrices(space, representation)
        if np.abs(matrix - matrix[0, 0] * identity).max() > tolerance
    ]
    for partner in range(dim):
        if not any(
            abs(abs(matrix[partner, partner]) - 1.0) < tolerance
            and np.abs(np.delete(matrix[:, partner], partner)).max() < tolerance
            for matrix in matrices
        ):
            return False
    return True


def _is_signed_permutation(matrix: NDArray[np.float64]) -> bool:
    """True when ``matrix`` only reorders and negates (to round-off)."""
    magnitudes = np.abs(np.asarray(matrix))
    snapped = np.rint(magnitudes)
    return bool(
        np.abs(magnitudes - snapped).max() < 1e-6
        and np.array_equal(snapped @ snapped.T, np.eye(len(snapped)))
    )


def _separation_clean(ferro, afm_list, representation, n_sites: int) -> bool:
    """A consistent FM/AFM separation whose partners all lie along symmetry-fixed directions."""
    if not _separation_consistent(ferro, afm_list, n_sites):
        return False
    spaces = ([] if ferro is None else [ferro]) + list(afm_list)
    return all(_partners_fixed_by_symmetry(space, representation) for space in spaces)


def _conventional_rows(vibrations: SymmetryOnlyVibrations) -> NDArray[np.float64]:
    """spglib's conventional lattice vectors (rows) in the basis of the input cell.

    In a trigonal or hexagonal crystal the three in-plane lattice vectors
    a, b and -a-b (six of them in a hexagonal lattice) are equivalent, and
    which of them spglib calls a depends on round-off in the input, so on a
    rigid rotation of the cell. Here the choice is made from the input
    basis alone: among the rotations about c that map the lattice onto
    itself, the one whose a has the fewest nonzero input indices, then the
    lexicographically largest ones, is taken -- the input a of a hexagonal
    cell, a - b of a rhombohedral primitive cell in the obverse setting
    (the a axis of the standard hexagonal cell of both). Other crystal
    systems keep spglib's axes.
    """
    import itertools

    dataset = vibrations.spglib_dataset
    transformation = np.asarray(dataset["transformation_matrix"], dtype=float)
    rows = np.linalg.inv(transformation).T
    if not 143 <= int(dataset["number"]) <= 194:
        return rows
    conventional = rows @ np.asarray(vibrations.primitive_cell.cell, dtype=float)
    metric = conventional @ conventional.T
    tolerance = 1e-3 * np.abs(metric).max()
    inverse_rows = np.linalg.inv(rows)
    best = None
    for entries in itertools.product((-1, 0, 1), repeat=4):
        turn = np.array([[entries[0], entries[1], 0], [entries[2], entries[3], 0], [0, 0, 1]])
        if round(np.linalg.det(turn)) != 1:
            continue
        if np.abs(turn @ metric @ turn.T - metric).max() > tolerance:
            continue  # not a rotation of the lattice about c
        in_primitive = inverse_rows @ turn @ rows
        if np.abs(in_primitive - np.rint(in_primitive)).max() > 1e-6:
            continue  # a symmetry of the hexagonal net, not of the (rhombohedral) lattice
        candidate = turn @ rows
        first = np.rint(candidate[0]).astype(int)
        key = (int(np.count_nonzero(first)), tuple(-first))
        if best is None or key < best[0]:
            best = (key, candidate)
    return rows if best is None else best[1]


def _spin_little_group(vibrations: SymmetryOnlyVibrations, mapping, site_indices: list[int]):
    """The little group of q = 0 on the magnetic sites, for the partner conventions.

    A ``crystod.symmetry_adapted_modes._LittleGroup`` built as
    ``crystod-phonon --vibration`` builds it, with the positions of the
    magnetic sites only (the spin vectors have three components per site),
    on the conventional axes of :func:`_conventional_rows`.

    Returns:
        ``(group, conventional)``: the little group, and the conventional
        lattice vectors (rows, in the input's Cartesian frame).
    """
    from .symmetry_adapted_modes import _LittleGroup

    rows = _conventional_rows(vibrations)
    # (a_s, b_s, c_s) = (a, b, c) P^-1 as in _conventional_axes, not idealized
    conventional = rows @ np.asarray(vibrations.primitive_cell.cell, dtype=float)
    axes = conventional / np.linalg.norm(conventional, axis=1)[:, None]
    positions = np.asarray(vibrations.primitive_cell.scaled_positions, dtype=float)[site_indices]
    group = _LittleGroup.at(
        np.zeros(3),
        np.asarray(vibrations.rotations)[mapping],
        np.asarray(vibrations.translations)[mapping],
        vibrations.primitive_cell.cell,
        vibrations.symprec,
        np.linalg.inv(rows.T),
        np.linalg.inv(axes),
        positions,
    )
    return group, conventional


def _partner_frame(conventional: NDArray[np.float64]) -> NDArray[np.float64]:
    """Orthonormal frame (rows) of the partner conventions.

    The conventional a axis, the in-plane direction perpendicular to it
    (along ``c x a``: Cartesian y of a standard hexagonal cell, b of an
    orthogonal one) and the third direction, in the input's Cartesian frame.
    """
    first = conventional[0] / np.linalg.norm(conventional[0])
    second = np.cross(conventional[2], first)
    second /= np.linalg.norm(second)
    return np.array([first, second, np.cross(first, second)])


def _canonical_spin_partners(space, representation, group, conventional) -> NDArray[np.float64]:
    """Orthogonal rotation of the partners of a real space onto symmetry-fixed directions.

    Adapted from ``crystod.symmetry_adapted_modes._canonical_partner_basis``
    (the partner choice of ``crystod-phonon --vibration``): the joint
    eigenvectors of a maximal set of commuting involutions (operations with
    ``R^2 = 1`` whose matrix on the partners is symmetric and not a multiple
    of the identity) whose joint eigenspaces are all one-dimensional, so
    every partner is fixed up to sign by those operations. Bases the group
    permutes up to sign come first (the x, y, z partners of a cubic T
    irrep), then the bases whose partners have the most symmetric isotropy
    subgroups (``_isotropy_key``), as in the source. Two changes for spins:
    among sets that tie on those keys (the three equivalent two-fold axes
    of a trigonal crystal, the three mirror pairs of a hexagonal one), the
    set holding the operation whose axis (mirror normal) lies closest to
    the conventional a axis, to the in-plane direction perpendicular to it
    (``c x a``) or to the third direction of that orthonormal frame is
    taken, in this order (then the face diagonals), so that the partners
    of an in-plane E irrep point along the a axis and perpendicular to it
    in the 321 and the 312 settings alike; and the partners are ordered
    first by the direction of that frame their moments point along the
    most (along a before perpendicular to it), then by their isotropy
    subgroups and the further keys of the source. Only the matrices of the
    operations on the partners and the moments of ``space`` enter, so
    every copy of the irrep that transforms with the same matrices gets
    the same rotation.

    Args:
        space: Real orthonormal partners (rows) of one copy of the irrep,
            shape ``(dim, 3N)``.
        representation: The spin representation ``(order, 3N, 3N)`` at q = 0.
        group: The little group from :func:`_spin_little_group`.
        conventional: The conventional lattice vectors (rows, Cartesian)
            from :func:`_spin_little_group`.

    Returns:
        The orthogonal ``(dim, dim)`` matrix whose columns are the new
        partners in the coordinates of the old ones (``basis.T @ space`` are
        the new rows); a signed permutation is returned exactly.
    """
    from functools import cmp_to_key

    from .symmetry_adapted_modes import (
        _compare_descriptors,
        _integer_kernel_vector,
        _isotropy_key,
        _maximal_cliques,
        _partner_descriptor,
        _partner_signs,
        _split_by_displacements,
    )

    vectors = np.real(np.asarray(space)).T  # (3N, dim), columns
    dim = vectors.shape[1]
    if dim < 2:
        return np.eye(dim)
    tolerance = _PARTNER_TOLERANCE
    rotations = group.setting_rotations
    matrices = list(_partner_matrices(space, representation))
    identity = np.eye(3, dtype=int)
    frame = _partner_frame(conventional)

    def operation_axis(index: int) -> NDArray[np.int_] | None:
        """Rotation axis, or mirror normal, of an operation in the conventional basis."""
        rotation = np.asarray(rotations[index], dtype=int)
        determinant = int(round(np.linalg.det(rotation)))
        if np.array_equal(rotation, determinant * identity):
            return None
        return _integer_kernel_vector(rotation - determinant * identity)

    def axis_rank(index: int) -> tuple:
        # how closely the axis (mirror normal) lies along the a axis, the
        # in-plane direction perpendicular to it or the third direction of
        # ``frame`` (Cartesian, so the [120] two-fold of a 312 setting counts
        # as perpendicular to a), then which of them, then the indices
        axis = operation_axis(index)
        if axis is None:
            return (4.0,)
        direction = axis @ conventional
        direction = direction / np.linalg.norm(direction)
        overlaps = np.abs(frame @ direction)
        along = int(np.argmax(overlaps))
        return (
            round(1.0 - float(overlaps[along]), 6),
            along,
            int(np.count_nonzero(axis)),
            tuple(-abs(int(value)) for value in axis),
        )

    order = sorted(
        range(len(rotations)), key=lambda index: tuple(np.asarray(rotations[index]).ravel())
    )
    keys: list[tuple[int, ...]] = []
    members_of: list[list[int]] = []  # operations giving each involution (up to sign)
    involutions: list[NDArray[np.float64]] = []
    for index in order:
        rotation = np.asarray(rotations[index], dtype=int)
        matrix = matrices[index]
        if np.array_equal(rotation, identity) or not np.array_equal(rotation @ rotation, identity):
            continue
        if np.abs(matrix - matrix.T).max() > tolerance:
            continue
        if np.abs(matrix - np.trace(matrix) / dim * np.eye(dim)).max() < tolerance:
            continue
        duplicate = next(
            (
                position for position, other in enumerate(involutions)
                if min(np.abs(matrix - other).max(), np.abs(matrix + other).max()) < tolerance
            ),
            None,
        )
        if duplicate is not None:
            members_of[duplicate].append(index)
            continue
        keys.append(tuple(rotation.ravel()))
        members_of.append([index])
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
    partial = None
    for clique in _maximal_cliques(adjacency) if involutions else []:
        members = sorted(clique)
        combined = sum(np.sqrt(2.0 + member) * involutions[member] for member in members)
        values, basis = np.linalg.eigh(0.5 * (combined + combined.T))
        if np.min(np.diff(values)) < tolerance:
            breaks = np.flatnonzero(np.diff(values) >= tolerance) + 1
            blocks = np.split(np.arange(dim), breaks)
            rank = (-len(blocks), tuple(sorted(keys[member] for member in members)))
            if partial is None or rank < partial[0]:
                partial = (rank, [basis[:, block] for block in blocks])
            continue
        in_basis = [basis.T @ matrix @ basis for matrix in matrices]
        monomial = all(
            np.all(np.minimum(np.abs(matrix), np.abs(np.abs(matrix) - 1.0)) <= tolerance)
            for matrix in in_basis
        )
        partner_keys = [
            _isotropy_key(_partner_signs(in_basis, column, tolerance), group)
            for column in range(dim)
        ]
        axes = min(axis_rank(index) for member in members for index in members_of[member])
        rank = (
            not monomial,
            tuple(sorted(partner_keys)),
            axes,
            tuple(sorted(keys[member] for member in members)),
        )
        if best is None or rank < best[0]:
            best = (rank, basis, partner_keys, in_basis)
    if best is None:
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

    def stabilizer(column: int) -> list[int]:
        signs = _partner_signs(in_basis, column, tolerance)
        return [index for index, sign in enumerate(signs) if sign > 0]

    def axis_key(column: int) -> tuple[int, int]:
        best_key = (0, 3)
        for index in stabilizer(column):
            rotation = rotations[index]
            if round(np.linalg.det(rotation)) != 1 or np.array_equal(rotation, identity):
                continue
            axis = _integer_kernel_vector(rotation - identity)
            if axis is None or np.count_nonzero(axis) != 1:
                continue
            period = next(
                n for n in range(2, 7)
                if np.array_equal(np.linalg.matrix_power(rotation, n), identity)
            )
            best_key = min(best_key, (-period, int(np.flatnonzero(axis)[0])))
        return best_key

    def direction_key(column: int) -> int:
        # the direction of ``frame`` the moments of the partner point along
        # the most (3 when two directions tie)
        components = (basis[:, column] @ vectors.T).reshape(-1, 3) @ frame.T
        weights = np.sum(components**2, axis=0)
        first, second = np.argsort(-weights, kind="stable")[:2]
        if weights[first] - weights[second] <= 1e-6 * weights[first]:
            return 3
        return int(first)

    def stabilizer_key(column: int) -> tuple:
        return tuple(sorted(tuple(rotations[index].ravel()) for index in stabilizer(column)))

    keys_by_column = [
        (direction_key(column), partner_keys[column], axis_key(column), stabilizer_key(column))
        for column in range(dim)
    ]
    descriptors: dict[int, NDArray[np.float64]] = {}

    def descriptor(column: int) -> NDArray[np.float64]:
        if column not in descriptors:
            descriptors[column] = _partner_descriptor(basis[:, column] @ vectors.T, group)
        return descriptors[column]

    def compare(first: int, second: int) -> int:
        if keys_by_column[first] != keys_by_column[second]:
            return -1 if keys_by_column[first] < keys_by_column[second] else 1
        return _compare_descriptors(descriptor(first), descriptor(second))

    basis = basis[:, sorted(range(dim), key=cmp_to_key(compare))]
    snapped = np.rint(basis)
    if np.abs(basis - snapped).max() < 1e-8 and np.array_equal(
        np.abs(snapped) @ np.abs(snapped).T, np.eye(dim)
    ):
        basis = snapped  # a signed permutation: exact, the values are only reordered
    return basis


def _canonical_spin_spaces(spin_rep, irrep, n_sites: int, group, conventional, pairs):
    """FM/AFM-separated spin bases of one irrep at q = 0 with canonical partners.

    The route taken when the copies of an irrep, made real one by one, do not
    separate cleanly (spgrep gives the E irreps of trigonal and hexagonal
    groups, and of rutile, in a complex basis; each copy then gets its own
    arbitrary real basis). The irrep is brought to a real form
    (``crystod.symmetry_adapted_modes._real_form``) and the spin
    representation projected onto it; the copies are made real together
    (``_real_occurrences``), so that every copy transforms with the same real
    matrices and :func:`separate_ferro_combination` can combine them partner
    by partner. Two or more AFM copies are then replaced by the basis that
    the moments fix (``_canonical_occurrences``: the sites each copy lives
    on, the moments along the crystal axes, the products of the moments of
    neighbouring sites), and the partners of all copies are turned by one
    common rotation onto the directions symmetry fixes
    (:func:`_canonical_spin_partners`); each partner is signed so that its
    largest moment component along the crystal axes is positive.

    Returns:
        ``(ferro, afm)`` as :func:`separate_ferro_combination`, or None when
        the irrep has no real form or the copies cannot be made real.
    """
    from .symmetry_adapted_modes import (
        _axes_sign,
        _canonical_occurrences,
        _real_form,
        _real_occurrences,
    )

    irrep = np.asarray(irrep)
    root = _real_form(irrep)
    if root is None:
        return None
    real_irrep = np.einsum("ji,gjk,kl->gil", root.conj(), irrep, root)
    occurrences = _real_occurrences(
        [np.asarray(space) for space in project_to_irrep(spin_rep, real_irrep)]
    )
    if occurrences is None:
        return None
    occurrences = [np.real(occurrence) for occurrence in occurrences]
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        ferro, afm_list = separate_ferro_combination(
            occurrences, spin_rep, n_sites, realify_aligned=False
        )
    if not _all_finite(ferro, afm_list):
        return None
    afm_list = [np.real(afm) for afm in afm_list]
    if len(afm_list) > 1:
        try:
            afm_list = [
                np.real(space) for space in _canonical_occurrences(
                    [afm + 0j for afm in afm_list], np.zeros(3), group.positions,
                    group.inverse_axes, pairs, real=True,
                )
            ]
        except RuntimeError:
            return None
    reference = np.real(ferro) if ferro is not None else afm_list[0]
    basis = _canonical_spin_partners(reference, spin_rep, group, conventional)

    permutation = None
    if np.array_equal(np.abs(basis) @ np.abs(basis).T, np.eye(len(basis))) and np.array_equal(
        basis, np.rint(basis)
    ):
        # a signed permutation: reorder and negate rows, which keeps every
        # value (and the sign of every zero) exact
        permutation = [int(np.flatnonzero(basis[:, column])[0]) for column in range(len(basis))]

    def turned(space: NDArray[np.float64]) -> NDArray[np.float64]:
        space = np.real(space)
        if permutation is not None:
            rows = np.array([
                space[row] if basis[row, column] > 0 else -space[row]
                for column, row in enumerate(permutation)
            ])
        else:
            rows = basis.T @ space
        return np.array([row * _axes_sign(row, group.inverse_axes) for row in rows])

    return (None if ferro is None else turned(ferro)), [turned(afm) for afm in afm_list]


def _natural_key(label: str) -> tuple:
    """Sort key that orders the digits of a label as numbers (GM2 before GM10)."""
    import re

    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", label))


def _conjugate_pairs(irreps, labels: list[str], present: set[int]) -> dict[int, int]:
    """The complex-type irreps present in the spin representation, with their conjugates.

    A complex-type irrep (complex characters) and the irrep with the
    conjugate characters, which time reversal joins into one physically
    irreducible real representation; the pair is recognized from the
    characters, not from the labels.

    Returns:
        ``{first: second}`` (indices into ``irreps``), ``first`` the member
        whose label sorts first in the natural order (as in
        ``crystod-phonon --vibration``, whose first half of the shared real
        partners belongs to that member).
    """
    from .runtime_compat import get_character

    characters = {index: np.array(get_character(irreps[index]), dtype=complex)
                  for index in present}
    pairs: dict[int, int] = {}
    used: set[int] = set()
    for index in sorted(present):
        if index in used or np.abs(characters[index].imag).max() < 1e-6:
            continue
        partner = next(
            (other for other in sorted(present)
             if other != index and other not in used
             and np.allclose(characters[other], characters[index].conj(), atol=1e-6)),
            None,
        )
        if partner is None:
            continue
        used |= {index, partner}
        first, second = sorted(
            (index, partner), key=lambda i: _natural_key(labels[i].split("(")[0])
        )
        pairs[first] = second
    return pairs


def _pair_label(label: str, partner: str) -> str:
    """``GM3+GM4+(2)``: the ISOTROPY pair label of a complex irrep and its
    conjugate (natural order, as ``crystod-group --parent`` and
    ``crystod-phonon`` write it) with the real dimension of the pair."""
    names = sorted((label.split("(")[0], partner.split("(")[0]), key=_natural_key)

    def dimension(text: str) -> int:
        return int(text.split("(")[1].rstrip(")")) if "(" in text else 1

    return f"{names[0]}{names[1]}({dimension(label) + dimension(partner)})"


def _frame_phase(row: NDArray[np.complex128], n_sites: int, frame: NDArray[np.float64],
                 ferro: bool) -> complex | None:
    """Unit phase that turns the real part of a complex copy onto the axes of ``frame``.

    The copy ``row`` of a complex one-dimensional irrep W spans, with its
    real and imaginary parts, a real two-dimensional space on which every
    operation acts as a rotation (W(g) is a phase), so no symmetry element
    singles out a direction in it; the convention fixes the phase instead.
    With a net moment (``ferro``), the net moment of the real part points
    along the first axis of ``frame`` it has a component on (along +a: the
    conventional a axis, then ``c x a``, then the third direction). Without
    one, the real part carries as much of the moments along the first axis
    of ``frame`` as it can (the sum over the sites of the squared moment
    components along that axis is largest), the next axis deciding when the
    moments along an axis are circular. None when no axis decides.
    """
    moments = row.reshape(n_sites, 3) @ np.asarray(frame).T  # components along the axes
    if ferro:
        net = moments.sum(axis=0)
        scale = np.linalg.norm(net)
        for axis in range(3):
            if scale > 1e-12 and abs(net[axis]) > 1e-6 * scale:
                return np.conj(net[axis]) / abs(net[axis])
    squares = (moments**2).sum(axis=0)
    weights = (np.abs(moments) ** 2).sum(axis=0)
    total = weights.sum()
    for axis in range(3):
        if weights[axis] > 1e-6 * total and abs(squares[axis]) > 1e-4 * weights[axis]:
            return np.exp(-0.5j * np.angle(squares[axis]))
    # every site's moment turns on a circle as the phase changes: the first
    # site (in site order) with the largest moment gets its moment along the
    # first axis it has a component on
    norms = np.linalg.norm(moments, axis=1)
    if norms.max() < 1e-12:
        return None
    site = int(np.flatnonzero(norms >= norms.max() * (1.0 - 1e-6))[0])
    for axis in range(3):
        if abs(moments[site, axis]) > 1e-6 * norms[site]:
            return np.conj(moments[site, axis]) / abs(moments[site, axis])
    return None


def _real_pair_space(row: NDArray[np.complex128], n_sites: int, frame: NDArray[np.float64],
                     inverse_axes: NDArray[np.float64], ferro: bool) -> NDArray[np.float64]:
    """The real two-dimensional space ``sqrt(2) [Re w, Im w]`` of one complex copy ``w``.

    The global phase of ``w`` is fixed by :func:`_frame_phase` (else by
    ``symmetry_adapted_modes._bloch_phase``), and each of the two partners
    is signed with its largest moment component along the crystal axes
    positive (``_axes_sign``).
    """
    from .symmetry_adapted_modes import _axes_sign, _bloch_phase

    row = np.asarray(row, dtype=complex)
    row = row / np.linalg.norm(row)
    phase = _frame_phase(row, n_sites, frame, ferro)
    row = row * (phase if phase is not None else _bloch_phase(row, inverse_axes))
    space = np.sqrt(2.0) * np.array([row.real, row.imag])
    return np.array([part * _axes_sign(part, inverse_axes) for part in space])


def _real_pair_rows(space: NDArray[np.complex128], n_sites: int, frame: NDArray[np.float64],
                    inverse_axes: NDArray[np.float64]) -> NDArray[np.float64]:
    """The real space of one copy of a complex irrep and its conjugate at q != 0.

    At a time-reversal-invariant q the representation is real in the lattice
    gauge, so the conjugate of a copy W of a ``d``-dimensional complex irrep
    is a copy of the conjugate irrep, and the two span one real
    ``2d``-dimensional space, listed under the pair label:
    ``sqrt(2) [Re w_1, ..., Re w_d, Im w_1, ..., Im w_d]`` (the merged
    partners of ``crystod-phonon --vibration``, the first half for the first
    member of the pair). The irrep matrices fix the relative phases of the
    rows, so one global phase is chosen, from the first row as in
    :func:`_real_pair_space`, and each partner is signed with its largest
    moment component along the crystal axes positive. A one-dimensional copy
    gives :func:`_real_pair_space`.
    """
    from .symmetry_adapted_modes import _axes_sign, _bloch_phase

    space = np.asarray(space, dtype=complex)
    if space.shape[0] == 1:
        return _real_pair_space(space[0], n_sites, frame, inverse_axes, False)
    first = space[0] / np.linalg.norm(space[0])
    phase = _frame_phase(first, n_sites, frame, False)
    rows = space * (phase if phase is not None else _bloch_phase(first, inverse_axes))
    real = np.sqrt(2.0) * np.vstack([rows.real, rows.imag])
    return np.array([part * _axes_sign(part, inverse_axes) for part in real])


def _merged_pair_spaces(copies, n_sites: int, group, conventional, pairs):
    """FM/AFM-separated real spin bases of a complex irrep and its conjugate at q = 0.

    A complex-type irrep W and its conjugate carry the same real spin
    arrangements: the real and imaginary parts of a copy span one real
    two-dimensional space, the partners of the physically irreducible
    representation (``sqrt(2) [Re W, Im W]``, the real partners that
    ``crystod-phonon --vibration`` gives the two members of the pair). The
    net moments are separated on the complex copies of W, where every
    complex combination of copies is again a copy: the combinations with a
    net moment (one for a one-dimensional irrep, the cluster dipole) span the
    complement of those without, the AFM ones, which are replaced by the
    basis that the moments fix (``_canonical_occurrences``, as for repeated
    irreps elsewhere). Every copy is then made real
    (:func:`_real_pair_space`): every operation acts on the two partners as
    the same rotation whatever the phase of the copy, and no symmetry element
    fixes a direction in the space, so the phase is a convention -- the net
    moment of the first partner of an FM copy along the conventional a axis,
    the moments of the first partner of an AFM copy along it as much as
    possible (:func:`_frame_phase`).

    Args:
        copies: The complex copies of W (``project_to_irrep``), each of shape
            ``(1, 3N)``.
        n_sites: The number N of spin-carrying sites.
        group, conventional: From :func:`_spin_little_group`.
        pairs: The neighbour-pair callback of ``_canonical_occurrences``.

    Returns:
        ``(ferro_list, afm_list)``: the real ``(2, 3N)`` spaces with and
        without a net moment, or None when a copy is not one-dimensional.
    """
    from .symmetry_adapted_modes import _canonical_occurrences

    copies = [np.asarray(copy, dtype=complex) for copy in copies]
    if any(copy.shape[0] != 1 for copy in copies):
        return None
    rows = np.array([copy[0] for copy in copies])
    orthonormal, _ = np.linalg.qr(rows.T)
    rows = orthonormal.T
    nets = rows.reshape(len(rows), n_sites, 3).sum(axis=1)
    _, values, right = np.linalg.svd(nets.T)
    rank = int(np.sum(values > 1e-6))
    combinations = right.conj() @ rows  # row k: sum_a conj(right[k, a]) rows[a]
    ferro_rows = list(combinations[:rank])
    afm_rows = list(combinations[rank:])
    if len(afm_rows) > 1:
        afm_rows = [
            occurrence[0] for occurrence in _canonical_occurrences(
                [row[None, :] for row in afm_rows], np.zeros(3), group.positions,
                group.inverse_axes, pairs, real=False,
            )
        ]
    frame = _partner_frame(conventional)
    return (
        [_real_pair_space(row, n_sites, frame, group.inverse_axes, True) for row in ferro_rows],
        [_real_pair_space(row, n_sites, frame, group.inverse_axes, False) for row in afm_rows],
    )


def _uniform_moment_spaces(spaces, n_sites: int, frame, group, pairs):
    """FM/AFM separation of the copies of one irrep from the uniform moments.

    The route for an irrep whose copies do not separate into one FM copy and
    net-zero AFM copies, because the irrep occurs more than once in a single
    moment (the moments perpendicular to the two-fold axis of a monoclinic
    crystal, any moment of a triclinic one): the arrangements with a net
    moment are the projections onto the isotypic space of the uniform
    moments along the axes of ``frame`` (taken in that order and made
    orthonormal; each has its net moment along its own axis), and the AFM
    ones are the rest of the space, whose net moments vanish because they are
    orthogonal to every uniform moment. For a one-dimensional irrep the
    space is first split into the moments along each axis of ``frame`` when
    symmetry keeps them apart (the two-fold axis of a monoclinic crystal and
    the axes perpendicular to it), so every arrangement keeps its moments
    along one axis; two or more AFM arrangements of one part are replaced by
    the basis that the moments fix (``_canonical_occurrences``) and signed
    with their largest moment component along the crystal axes positive.

    Returns:
        A list of ``(kind, space)`` pairs (FM first): one space per
        arrangement for a one-dimensional irrep; for a larger irrep one FM
        space with every FM arrangement and the AFM rest in parts of the
        irrep dimension. None when the copies do not span a real space.
    """
    from .symmetry_adapted_modes import _axes_sign, _canonical_occurrences

    spaces = [np.asarray(space) for space in spaces]
    dim = spaces[0].shape[0]
    total = sum(space.shape[0] for space in spaces)
    stacked = np.vstack([part for space in spaces for part in (space.real, space.imag)])
    _, values, right = np.linalg.svd(stacked, full_matrices=False)
    if len(values) < total or values[total - 1] < 1e-8 or (
        len(values) > total and values[total] > 1e-6
    ):
        return None
    basis = right[:total]

    def separate(space: NDArray[np.float64], directions) -> tuple[list, list]:
        """FM rows (projected uniform moments) and the orthonormal rest of ``space``."""
        ferro_rows: list[NDArray[np.float64]] = []
        for direction in directions:
            vector = space.T @ (space @ np.tile(direction, n_sites))
            for row in ferro_rows:
                vector = vector - (row @ vector) * row
            norm = np.linalg.norm(vector)
            if norm > 1e-6:
                ferro_rows.append(vector / norm)
        rest = space
        if ferro_rows:
            ferro = np.array(ferro_rows)
            rest = space - (space @ ferro.T) @ ferro
        _, _, right = np.linalg.svd(rest, full_matrices=False)
        return ferro_rows, list(right[: len(space) - len(ferro_rows)])

    if dim > 1:
        ferro_rows, afm_rows = separate(basis, frame)
        ordered = [("FM", np.array(ferro_rows))] if ferro_rows else []
        ordered += [("AFM", np.array(afm_rows[start:start + dim]))
                    for start in range(0, len(afm_rows), dim)]
        return ordered

    # a one-dimensional irrep: every operation acts on the moments along an
    # axis of ``frame`` alone when it maps that axis onto +-itself (the
    # two-fold axis of a monoclinic crystal and the plane perpendicular to
    # it), and the space then splits into the moments along each axis
    parts = []
    for axis in frame:
        along = np.kron(np.eye(n_sites), np.outer(axis, axis))
        projected = basis @ along
        if np.abs(projected - projected @ basis.T @ basis).max() > 1e-6:
            parts = []
            break
        _, values, right = np.linalg.svd(projected, full_matrices=False)
        rank = int(np.sum(values > 1e-6))
        if rank:
            parts.append((axis, right[:rank]))
    if sum(len(part) for _, part in parts) != total:
        parts = [(None, basis)]
    ferro_rows, afm_rows = [], []
    for axis, part in parts:
        ferro_part, afm_part = separate(part, frame if axis is None else [axis])
        ferro_rows += ferro_part
        if len(afm_part) > 1:
            afm_part = [
                np.real(occurrence[0]) for occurrence in _canonical_occurrences(
                    [row[None, :] + 0j for row in afm_part], np.zeros(3), group.positions,
                    group.inverse_axes, pairs, real=True,
                )
            ]
        afm_rows += [row * _axes_sign(row, group.inverse_axes) for row in afm_part]
    return ([("FM", row[None, :]) for row in ferro_rows]
            + [("AFM", row[None, :]) for row in afm_rows])


def _require_primitive(vibrations: SymmetryOnlyVibrations) -> None:
    """Refuse a cell that is not primitive.

    ``crystod-mag`` analyses the cell as given (``standardize=False``: the
    spin listings and MAGMOM follow its atom order), and the irreps of the
    little groups need the operations of a primitive cell; a conventional
    cell or a supercell has pure translations among its operations.

    Raises:
        ValueError: The space group of the cell holds pure translations.
    """
    rotations = np.asarray(vibrations.rotations)
    n_lattice = int(np.sum(np.all(rotations == np.eye(3, dtype=int), axis=(1, 2))))
    if n_lattice > 1:
        n_atoms = len(vibrations.primitive_cell.symbols)
        raise ValueError(
            f"the input cell is not primitive ({n_atoms} atoms, the primitive cell has "
            f"{n_atoms // n_lattice}); crystod-mag uses the cell as given (the spin listings "
            "and MAGMOM follow its atom order) and needs a primitive cell, such as the "
            "PPOSCAR of phonopy --symmetry (--conventional then exports the spin structures "
            "in the conventional cell)."
        )


#: Largest multiple of a cell along one axis in the magnetic cell at q != 0.
_MAX_CELL_MULTIPLE = 12
#: Tolerance on q . T being an integer for a translation T of the magnetic cell.
_COMMENSURATE_TOLERANCE = 1e-6


def _commensurability_error(qpoint, cell_matrix, conventional: bool) -> str | None:
    """Why the magnetic cell ``cell_matrix`` does not fit q, or None when it does.

    The cell is commensurate when ``q . T`` is an integer for each of its
    vectors ``T`` (rows of ``cell_matrix``, primitive units), so that the
    Bloch phase is the same in every copy of the cell. The cell is built from
    the denominators of q up to ``_MAX_CELL_MULTIPLE``; a q with a larger
    denominator (``0.15 = 3/20``) or none does not fit, and the message
    names the smallest commensurate supercell of the primitive cell.
    """
    qpoint = np.asarray(qpoint, dtype=float)
    products = np.asarray(cell_matrix, dtype=float) @ qpoint
    if np.abs(products - np.rint(products)).max() < _COMMENSURATE_TOLERANCE:
        return None
    sizes = []
    for value in qpoint:
        size = Fraction(float(value)).limit_denominator(1000).denominator
        product = size * float(value)
        sizes.append(size if abs(product - round(product)) < _COMMENSURATE_TOLERANCE else None)
    kind = "conventional" if conventional else "primitive"
    text = (f"q-point {np.round(qpoint, 6).tolist()} is not commensurate with a magnetic "
            f"cell of at most {_MAX_CELL_MULTIPLE} {kind} cells along each axis")
    advice = (f" Choose q with fractional coordinates of denominators up to "
              f"{_MAX_CELL_MULTIPLE}.")
    if None in sizes:
        return text + "; no supercell of up to 1000 primitive cells along each axis is " \
            "commensurate." + advice
    smallest = "x".join(str(size) for size in sizes)
    if conventional and max(sizes) <= _MAX_CELL_MULTIPLE:
        return text + f"; the {smallest} primitive supercell is commensurate (run without " \
            "--conventional)."
    return text + "; the smallest commensurate supercell along the primitive axes is " \
        f"{smallest}." + advice


@dataclass(eq=False)
class SpinBasisComponent:
    """One exported component of a symmetry-adapted spin basis.

    Attributes:
        entry_index: Index of the basis in ``SpinBasisAnalysis.entries``.
        label: The ISO-IR irrep label (``"GM4+(3)"``).
        kind: ``"FM"``, ``"AFM"`` or ``"AFM (q != 0)"``.
        rank_name: Multipole name at q = 0 (``"dipole"``), else None.
        tag: Component name as in the VESTA file name: ``"x"``, ``"y"``,
            ``"z"``, ``"111"``, ``"1m10"`` for collinear components, else
            ``"comp<N>"`` (the N-th basis vector of the space, counted
            from 1).
        field: Real spin field ``(n_cells, N, 3)`` over the magnetic cell,
            largest moment 1; None when the real field vanishes.
        net_moment: Net moment over the magnetic cell.
        group: :class:`crystod.magnetic_group.MagneticGroupInfo` of the
            arrangement (None without classification; its
            ``identified`` is False when spglib finds the operations but
            cannot name the group).
        responses: :class:`crystod.magnetic_group.MagneticResponses`.
        collinear: :class:`crystod.magnetic_group.CollinearClass`.
        file_tag: The tag made unique within the analysis as in the VESTA
            file name (``"z"``, ``"z_2"`` for the second ``z`` of the same
            irrep and multipole); empty when the field vanishes.
        vesta_name: The VESTA file name ``crystod-mag`` writes for this
            component; empty when the field vanishes.
    """

    entry_index: int
    label: str
    kind: str
    rank_name: str | None
    tag: str
    field: NDArray[np.float64] | None
    net_moment: NDArray[np.float64] | None = None
    group: object = None
    responses: object = None
    collinear: object = None
    file_tag: str = ""
    vesta_name: str = ""

    @property
    def header(self) -> str:
        """``"GM4+(3) [FM, dipole]"`` -- the block title of ``crystod-mag``."""
        return f"{self.label} [{self.kind}" + (f", {self.rank_name}]" if self.rank_name else "]")


@dataclass(eq=False)
class SpinBasisAnalysis:
    """The symmetry-adapted spin bases at one q point, with their magnetic
    groups and collinear classes (what ``crystod-mag --qpoint`` prints).

    Attributes:
        element: The magnetic element.
        space_group: International symbol of the space group.
        space_group_number: Its number.
        site_indices: Indices of the magnetic sites in the primitive cell.
        site_positions: Their fractional positions.
        qpoint_label: Name of the q point (``"GAMMA"``, ``"X"``).
        qpoint: Its primitive reciprocal coordinates.
        is_gamma: True at q = 0.
        decomposition: ``{irrep label: multiplicity}`` of the spin
            representation, in the order of the irreps; at a
            time-reversal-invariant q a complex irrep and its conjugate
            count once under their pair label (``"GM3+GM4+(2)"``).
        total_dim: Sum of the projected dimensions (``3 N`` normally).
        entries: The bases: ``(label, kind, rank_name, space)`` with the
            basis array ``space`` of shape ``(dim, 3 N)``.
        supercell_sizes: Diagonal of the commensurate magnetic supercell.
        cell_matrix: The magnetic (display) cell in primitive units, rows.
        conventional_centring: The centring letter with ``conventional``,
            else None.
        conventional_multiples: Conventional cells along each axis, or None.
        cell_translations: Primitive translations inside the magnetic cell.
        supercell_lattice: Lattice of the magnetic cell (rows, Angstrom).
        supercell_symbols: Symbols of the atoms of the magnetic cell.
        supercell_positions: Their fractional positions.
        supercell_site_of_atom: Magnetic-site index of every atom, or None.
        components: The exported components
            (:class:`SpinBasisComponent`), in output order.
        formula: Reduced formula (file names).
        symbols: Symbols of the primitive cell.
        pair_labels: The labels of the complex irreps merged with their
            conjugates (``"GM3+GM4+(2)"``, ``"Z3+Z4+(2)"``) that have a basis.
    """

    element: str
    space_group: str
    space_group_number: int
    site_indices: list
    site_positions: NDArray[np.float64]
    qpoint_label: str
    qpoint: list
    is_gamma: bool
    decomposition: dict
    total_dim: int
    entries: list
    supercell_sizes: NDArray[np.int_]
    cell_matrix: NDArray[np.int_]
    conventional_centring: str | None
    conventional_multiples: NDArray[np.int_] | None
    cell_translations: list
    supercell_lattice: NDArray[np.float64]
    supercell_symbols: list
    supercell_positions: NDArray[np.float64]
    supercell_site_of_atom: list
    components: list = dataclass_field(default_factory=list)
    formula: str = ""
    symbols: list = dataclass_field(default_factory=list)
    pair_labels: list = dataclass_field(default_factory=list)

    @property
    def n_sites(self) -> int:
        """Number of magnetic sites in the primitive cell."""
        return len(self.site_indices)


def analyze_spin_basis(
    cell=None,
    element: str = "",
    qpoint=("GM",),
    *,
    symprec: float = 1e-5,
    conventional: bool = False,
    classify: bool = True,
    vibrations: SymmetryOnlyVibrations | None = None,
) -> SpinBasisAnalysis:
    """Symmetry-adapted spin bases at one q point, classified.

    The computation behind ``crystod-mag --qpoint``: the spin
    representation is projected onto the irreps of the little group, the
    FM/AFM combinations are separated at q = 0 (copies of an irrep that do
    not separate cleanly when made real one by one, or whose partners do
    not lie along directions symmetry fixes -- in a trigonal or hexagonal
    crystal: not along the a axis of the input cell and perpendicular to
    it, :func:`_canonical_spin_partners` -- are redone through the real
    form of the irrep with canonical partners: :func:`_canonical_spin_spaces`;
    an irrep that occurs more than once in a single moment is separated by
    the uniform moments, :func:`_uniform_moment_spaces`; a complex irrep and
    its conjugate are merged into one real two-dimensional space under their
    pair label, :func:`_merged_pair_spaces`, and so at the other
    time-reversal-invariant q points, where a pair of d-dimensional irreps
    gives a real 2d-dimensional space, :func:`_real_pair_rows`), every basis
    vector is resolved into its exported components (x/y/z/111 or comp<N>),
    the real spin field of each component over the magnetic cell is built
    (lattice gauge: the spin of site j in the cell at T is
    ``Re(w_j exp(2 pi i q.T))``; global phase as in the VESTA export,
    largest moment 1), and with ``classify``
    every component gets its magnetic space group
    (:func:`crystod.magnetic_group.magnetic_space_group`), the responses
    its magnetic point group allows
    (:func:`crystod.magnetic_group.macroscopic_responses`) and its
    collinear class (:func:`crystod.magnetic_group.classify_collinear`).

    Args:
        cell: The crystal (phonopy ``PhonopyAtoms``, e.g. from
            ``phonopy.interface.vasp.read_vasp``); not needed when
            ``vibrations`` is given.
        element: Element whose sites carry the spins (``"Ni"``).
        qpoint: ``--qpoint`` tokens: one label (``"GM"``, ``"X"``) or three
            coordinates in the primitive reciprocal basis (fractions such as
            ``"1/2"`` allowed); a single string is one label.
        symprec: Symmetry tolerance (spglib, Angstrom); the sites are matched
            within the tolerance of ``vibrations`` (this one when the cell is
            given).
        conventional: Build the magnetic cell from the conventional cell
            (``crystod-mag --conventional``).
        classify: Compute the magnetic groups, responses and classes.
        vibrations: A prebuilt ``SymmetryOnlyVibrations(cell,
            standardize=False)`` (the CLI passes its own).

    Returns:
        A :class:`SpinBasisAnalysis`.

    Raises:
        SystemExit: The element is not in the cell
            (``ValueError`` through ``crystod.mag``).
        ValueError: Unknown q-point label; the cell is not primitive; q is
            not commensurate with a magnetic cell of at most 12 cells along
            each axis (the message names the smallest commensurate
            supercell).

    Example:
        >>> from phonopy.interface.vasp import read_vasp
        >>> from crystod import mag
        >>> cell = read_vasp("example/test_POSCARs/136_PPOSCAR_TiO2")
        >>> result = mag.analyze_spin_basis(cell, "Ti", "GM")
        >>> [(c.header, c.tag, c.collinear.short_name) for c in result.components][-1]
        ('GM2+(1) [AFM, octupole]', 'z', 'altermagnet (d-wave)')
    """
    from .magnetic_group import (
        classify_collinear,
        macroscopic_responses,
        magnetic_cell_translations,
        magnetic_space_group,
    )

    if vibrations is None:
        vibrations = SymmetryOnlyVibrations(cell=cell, symprec=symprec, standardize=False)
    _require_primitive(vibrations)
    primitive = vibrations.primitive_cell
    symbols = list(primitive.symbols)
    positions = primitive.scaled_positions
    lattice = np.array(primitive.cell)
    site_indices = [i for i, s in enumerate(symbols) if s == element]
    if not site_indices:
        available = ", ".join(sorted(set(symbols)))
        raise SystemExit(f"ERROR: element '{element}' is not in this POSCAR ({available}).")
    n_sites = len(site_indices)

    tokens = [qpoint] if isinstance(qpoint, str) else [str(token) for token in qpoint]
    qpoint_label, qpoint = vibrations.resolve_qpoint(tokens)
    is_gamma = np.allclose(qpoint, 0.0)

    irreps, spin_rep, mapping = get_spin_representation(vibrations, site_indices, qpoint)
    irrep_labels = vibrations.get_irrep_labels(qpoint, irreps, mapping)

    # the projected rows carry the Bloch factor of each site's position
    # (spgrep's convention, as phonopy's eigenvectors); the spin of site j in
    # the cell at lattice translation T is Re(w_j exp(2 pi i q.T)) with
    # w_j = v_j exp(2 pi i q.x_j), the lattice gauge, real at a
    # time-reversal-invariant q for the irreps of real type (x_j symmetrized
    # as in get_spin_representation: the input positions of a structure at
    # its ideal positions)
    ideal_positions = _symmetrized_site_positions(vibrations, site_indices)
    gauge = np.repeat(np.exp(2j * np.pi * (ideal_positions @ np.asarray(qpoint))), 3)

    raw_spaces: list[NDArray] = []
    raw_labels: list[str] = []
    raw_irrep_indices: list[int] = []
    raw_copies: dict[int, list[NDArray]] = {}
    for irrep_index, (irrep, label) in enumerate(zip(irreps, irrep_labels)):
        for space in project_to_irrep(spin_rep, irrep):
            space = np.asarray(space, dtype=complex) * gauge[None, :]
            raw_copies.setdefault(irrep_index, []).append(space)
            realified = _realify(space)
            raw_spaces.append(realified if realified is not None else space)
            raw_labels.append(label)
            raw_irrep_indices.append(irrep_index)

    total_dim = sum(space.shape[0] for space in raw_spaces)

    # at a time-reversal-invariant q (q = 0 and the other points with 2q a
    # reciprocal lattice vector) a complex irrep and its conjugate form one
    # physically irreducible real space, listed once under their pair label
    time_reversal_invariant = np.allclose(2.0 * np.asarray(qpoint), np.rint(2.0 * np.asarray(qpoint)))
    conjugate_pairs = (
        _conjugate_pairs(irreps, irrep_labels, set(raw_irrep_indices))
        if time_reversal_invariant else {}
    )
    second_members = {second: first for first, second in conjugate_pairs.items()}

    def display_label(irrep_index: int) -> str:
        first = second_members.get(irrep_index, irrep_index)
        if first in conjugate_pairs:
            return _pair_label(irrep_labels[first], irrep_labels[conjugate_pairs[first]])
        return irrep_labels[irrep_index]

    counted: dict[str, int] = {}
    for label, irrep_index in zip(raw_labels, raw_irrep_indices):
        if irrep_index in second_members:
            continue
        name = display_label(irrep_index)
        counted[name] = counted.get(name, 0) + 1

    # FM/AFM separation (Gamma only) and multipole-rank naming
    entries = []  # (label, kind, rank_name, space)
    if is_gamma:
        rank_lists = get_multipole_rank_lists(
            vibrations, vibrations.rotations[mapping], irreps
        )
        label_to_irrep_index: dict[str, int] = {}
        for label, irrep_index in zip(raw_labels, raw_irrep_indices):
            label_to_irrep_index.setdefault(label, irrep_index)
        little_group = None
        conventional_lattice = None
        hexagonal_family = 143 <= int(vibrations.spglib_dataset["number"]) <= 194
        nearest: list = []

        def pairs():
            from .symmetry_adapted_modes import _nearest_images

            if not nearest:
                nearest.append(_nearest_images(
                    little_group.positions, lattice, little_group.inverse_axes
                ))
            return nearest[0]

        def add_entries(label: str, irrep_index: int, ordered) -> None:
            # multipole ranks of this irrep in increasing order
            # (FM = lowest rank = dipole, then octupole, ...)
            ranks = rank_lists[irrep_index]
            for position, (kind, space) in enumerate(ordered):
                rank = ranks[position] if position < len(ranks) else None
                rank_name = MULTIPOLE_NAMES.get(rank, f"rank{rank}") if rank else "multipole"
                entries.append((label, kind, rank_name, space))

        for label in dict.fromkeys(raw_labels):
            irrep_index = label_to_irrep_index[label]
            if irrep_index in second_members:
                continue  # listed with the first member of its pair
            if irrep_index in conjugate_pairs:
                if little_group is None:
                    little_group, conventional_lattice = _spin_little_group(
                        vibrations, mapping, site_indices
                    )
                merged = _merged_pair_spaces(
                    raw_copies[irrep_index], n_sites, little_group, conventional_lattice, pairs,
                )
                if merged is not None:
                    ferro_list, afm_list = merged
                    add_entries(display_label(irrep_index), irrep_index,
                                [("FM", space) for space in ferro_list]
                                + [("AFM", space) for space in afm_list])
                    continue
            group = [space for space, l in zip(raw_spaces, raw_labels) if l == label]
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                ferro, afm_list = separate_ferro_combination(group, spin_rep, n_sites)
            clean = _separation_clean(ferro, afm_list, spin_rep, n_sites)
            if clean and hexagonal_family and group[0].shape[0] > 1:
                # symmetry-fixed partners, but possibly not the canonical ones:
                # which of the equivalent in-plane axes (a, b, a+b and the
                # directions perpendicular to them) the copies made real one
                # by one pick depends on the orientation of the input cell;
                # kept only when the canonical choice merely reorders or
                # negates them (other crystal systems keep them as they are)
                if little_group is None:
                    little_group, conventional_lattice = _spin_little_group(
                        vibrations, mapping, site_indices
                    )
                reference = np.real(ferro if ferro is not None else afm_list[0])
                reference = reference / np.linalg.norm(reference, axis=1)[:, None]
                clean = _is_signed_permutation(_canonical_spin_partners(
                    reference, spin_rep, little_group, conventional_lattice
                ))
            if not clean:
                # the copies made real one by one are not aligned or not along
                # the canonical symmetry-fixed directions (2-dim irreps that
                # spgrep gives in a complex basis: E of trigonal and hexagonal
                # groups, rutile GM5+): project onto the real form of the
                # irrep, separate the copies partner by partner and turn the
                # partners onto the directions symmetry fixes
                if little_group is None:
                    little_group, conventional_lattice = _spin_little_group(
                        vibrations, mapping, site_indices
                    )
                canonical = _canonical_spin_spaces(
                    spin_rep, irreps[label_to_irrep_index[label]], n_sites, little_group,
                    conventional_lattice, pairs,
                )
                if canonical is not None:
                    n_copies = len(canonical[1]) + (canonical[0] is not None)
                    better = (_separation_consistent(*canonical, n_sites)
                              or not _all_finite(ferro, afm_list))
                    if n_copies == len(group) and better:
                        ferro, afm_list = canonical
            ordered = []
            if ferro is not None:
                ordered.append(("FM", np.real(ferro)))
            ordered.extend(("AFM", np.real(afm)) for afm in afm_list)
            if not _separation_consistent(ferro, afm_list, n_sites):
                # the irrep occurs more than once in a single moment (more
                # than one FM combination: monoclinic and triclinic crystals)
                if little_group is None:
                    little_group, conventional_lattice = _spin_little_group(
                        vibrations, mapping, site_indices
                    )
                # the frame of the partner conventions; a triclinic crystal
                # has no preferred axes, and the Cartesian axes are taken
                triclinic = int(vibrations.spglib_dataset["number"]) <= 2
                frame = np.eye(3) if triclinic else _partner_frame(conventional_lattice)
                uniform = _uniform_moment_spaces(group, n_sites, frame, little_group, pairs)
                if uniform is not None:
                    ordered = uniform
            add_entries(label, irrep_index, ordered)
    else:
        frame = inverse_axes = None
        if conjugate_pairs:
            pair_group, conventional_lattice = _spin_little_group(
                vibrations, mapping, site_indices
            )
            frame = _partner_frame(conventional_lattice)
            inverse_axes = pair_group.inverse_axes
        for label, space, irrep_index in zip(raw_labels, raw_spaces, raw_irrep_indices):
            if irrep_index in second_members:
                continue  # listed with the first member of its pair
            if irrep_index in conjugate_pairs:
                # the real and imaginary parts of the copy (lattice gauge,
                # where the representation is real) span one real space of
                # twice the dimension, as at q = 0, under the pair label of
                # the decomposition
                space = _real_pair_rows(np.asarray(space), n_sites, frame, inverse_axes)
                label = display_label(irrep_index)
            entries.append((label, "AFM (q != 0)", None, space))

    # commensurate (magnetic) supercell for q != 0
    if is_gamma:
        supercell_sizes = np.array([1, 1, 1], dtype=int)
    else:
        supercell_sizes = np.array(
            [
                1 if abs(component) < 1e-10
                else Fraction(float(component)).limit_denominator(_MAX_CELL_MULTIPLE).denominator
                for component in qpoint
            ],
            dtype=int,
        )

    # Display cell for the VESTA/MAGMOM output: rows of cell_matrix are the
    # display-cell lattice vectors in primitive units. With --conventional the
    # conventional cell is used (times commensurate multiples for q != 0);
    # wrapping an atom by a display-cell vector leaves its Bloch phase
    # unchanged because q . S_row is an integer by construction.
    centring = None
    multiples = None
    if conventional:
        from .phonon_vector import get_commensurate_supercell_matrix, get_conventional_matrix

        centring = vibrations.spglib_dataset["international"][0]
        base_matrix = get_conventional_matrix(centring)
        cell_matrix = np.array(get_commensurate_supercell_matrix(qpoint, base_matrix), dtype=int)
        multiples = np.rint(
            np.diag(cell_matrix @ np.linalg.inv(np.array(base_matrix, dtype=float)))
        ).astype(int)
    else:
        cell_matrix = np.diag(supercell_sizes)
    # the sizes come from the denominators of q up to _MAX_CELL_MULTIPLE: a q
    # that does not fit (0.15 = 3/20 in a 1x1x7 cell) would leave a slice of
    # the modulation, with a spurious net moment, MSG and class
    problem = _commensurability_error(qpoint, cell_matrix, conventional)
    if problem is not None:
        raise ValueError(problem)

    inverse_cell = np.linalg.inv(np.array(cell_matrix, dtype=float))
    cell_translations = magnetic_cell_translations(cell_matrix)

    supercell_lattice = np.array(cell_matrix, dtype=float) @ lattice
    supercell_symbols: list[str] = []
    supercell_positions: list[NDArray[np.float64]] = []
    supercell_site_of_atom: list[int | None] = []  # magnetic-site index or None
    for translation in cell_translations:
        for i, symbol in enumerate(symbols):
            supercell_symbols.append(symbol)
            frac_cell = (positions[i] + translation) @ inverse_cell
            supercell_positions.append(frac_cell - np.floor(frac_cell))
            supercell_site_of_atom.append(site_indices.index(i) if i in site_indices else None)
    supercell_positions = np.array(supercell_positions)
    cell_phases = [np.exp(2j * np.pi * np.dot(qpoint, t)) for t in cell_translations]
    type_numbers = {symbol: n + 1 for n, symbol in enumerate(dict.fromkeys(supercell_symbols))}
    numbers = [type_numbers[symbol] for symbol in supercell_symbols]

    components: list[SpinBasisComponent] = []
    for entry_index, (label, kind, rank_name, space) in enumerate(entries):
        component_tags = []
        component_vectors = []
        for c in range(space.shape[0]):
            vector = np.asarray(space[c])
            result = _direction_tag(np.real(vector), n_sites) if np.abs(np.imag(vector)).max() < 1e-8 else None
            if result is None:
                component_tags.append(f"comp{c + 1}")
                component_vectors.append(vector)
            else:
                tag, sign = result
                component_tags.append(tag)
                component_vectors.append(sign * vector)

        export_list = list(zip(component_tags, component_vectors))
        # combined x+y+z configuration for 3-dim spaces with x/y/z components
        if sorted(component_tags) == ["x", "y", "z"]:
            combined = np.sum(component_vectors, axis=0) / np.sqrt(len(component_vectors))
            export_list.append(("111", combined))

        for tag, vector in export_list:
            # spin field over the magnetic supercell: S_i(T) = Re(S_i exp(2 pi i q.T))
            field = np.array(
                [
                    np.asarray(vector).reshape(n_sites, 3) * phase
                    for phase in cell_phases
                ]
            )  # (n_cells, n_sites, 3)
            sum_of_squares = np.sum(field * field)
            if abs(sum_of_squares) > 1e-12:
                field = field * np.exp(-0.5j * np.angle(sum_of_squares))
            field = np.real(field)
            max_norm = float(np.max(np.linalg.norm(field, axis=2)))
            if max_norm < 1e-12:
                components.append(SpinBasisComponent(entry_index, label, kind, rank_name, tag, None))
                continue
            field = field / max_norm
            net = np.sum(field[0], axis=0) if is_gamma else np.sum(field, axis=(0, 1))
            components.append(
                SpinBasisComponent(entry_index, label, kind, rank_name, tag, field, net)
            )

    # VESTA file names (and the tags made unique the same way)
    formula = reduced_formula(symbols)
    conv_tag = "_conv" if conventional else ""
    used_names: set[str] = set()
    for component in components:
        if component.field is None:
            continue
        short_label = component.label.split("(")[0].strip()
        rank_tag = component.rank_name if component.rank_name else (
            "FM" if component.kind == "FM" else qpoint_label)
        name = f"POSCAR_{formula}_spin_{short_label}_{rank_tag}_{component.tag}{conv_tag}.vesta"
        file_tag = component.tag
        if name in used_names:
            stem = name[: -len(".vesta")]
            suffix = 2
            while f"{stem}_{suffix}.vesta" in used_names:
                suffix += 1
            name = f"{stem}_{suffix}.vesta"
            file_tag = f"{component.tag}_{suffix}"
        used_names.add(name)
        component.vesta_name = name
        component.file_tag = file_tag

    for component in components:
        field = component.field
        if not classify or field is None:
            continue
        if not np.all(np.isfinite(field)):
            continue
        # spglib cell of the magnetic cell: the spins on the magnetic sites,
        # zero on the other atoms
        magmoms = np.zeros((len(supercell_symbols), 3))
        for atom_index, site in enumerate(supercell_site_of_atom):
            if site is not None:
                magmoms[atom_index] = field[atom_index // len(symbols), site]
        try:
            component.group = magnetic_space_group(
                supercell_lattice, supercell_positions, numbers, magmoms, symprec=symprec
            )
        except SystemExit:
            component.group = None
        if component.group is not None:
            component.responses = macroscopic_responses(component.group)
        component.collinear = classify_collinear(
            vibrations.rotations, vibrations.translations, positions, site_indices,
            cell_matrix, field, lattice, symprec=vibrations.symprec,
        )

    dataset = vibrations.spglib_dataset
    return SpinBasisAnalysis(
        element=element,
        space_group=str(dataset["international"]),
        space_group_number=int(dataset["number"]),
        site_indices=site_indices,
        site_positions=np.array([positions[i] for i in site_indices]),
        qpoint_label=qpoint_label,
        qpoint=qpoint,
        is_gamma=bool(is_gamma),
        decomposition=counted,
        total_dim=total_dim,
        entries=entries,
        supercell_sizes=supercell_sizes,
        cell_matrix=np.asarray(cell_matrix, dtype=int),
        conventional_centring=centring,
        conventional_multiples=multiples,
        cell_translations=cell_translations,
        supercell_lattice=supercell_lattice,
        supercell_symbols=supercell_symbols,
        supercell_positions=supercell_positions,
        supercell_site_of_atom=supercell_site_of_atom,
        components=components,
        formula=formula,
        symbols=symbols,
        pair_labels=[display_label(first) for first in conjugate_pairs],
    )


def magnetic_summary_rows(analysis: SpinBasisAnalysis) -> list[tuple[str, ...]]:
    """Rows of the table of magnetic groups and collinear classes.

    Args:
        analysis: A classified :class:`SpinBasisAnalysis`.

    Returns:
        One tuple per exported component with a nonvanishing field:
        ``(basis, component, MSG (BNS), type, MPG, net magnetization,
        class)``, e.g. ``("GM2+(1) [AFM, octupole]", "z",
        "P4_2'/mnm' (136.499)", "III", "4'/mmm'", "none",
        "altermagnet (d-wave)")``. The component is the tag of the VESTA
        file name (``"z_2"`` for the second ``z`` of the same basis). A
        group that spglib cannot identify, and an unclassified component
        (FM/AFM separation failed), give ``"-"`` in the cells concerned;
        no component is left out.
    """
    rows = []
    for component in analysis.components:
        if component.field is None or component.collinear is None:
            continue
        group = component.group
        if group is not None and group.identified:
            msg_cells = (f"{group.bns_symbol} ({group.bns_number})", group.type_name,
                         str(group.magnetic_point_group))
        else:
            msg_cells = ("-", "-", "-")
        if component.responses is not None:
            magnetization = component.responses.magnetization_direction
        else:
            magnetization = "-"
        rows.append((
            component.header,
            component.file_tag or component.tag,
            *msg_cells,
            magnetization,
            component.collinear.short_name,
        ))
    return rows


MAGNETIC_SUMMARY_HEADERS = (
    "basis", "component", "MSG (BNS)", "type", "MPG", "net magnetization", "class",
)


def magnetic_summary_notes(analysis: SpinBasisAnalysis) -> list[str]:
    """The notes under the summary table (frames, tags, warnings).

    Args:
        analysis: A classified :class:`SpinBasisAnalysis`.

    Returns:
        Note lines, unindented: how to read the columns, the frames of the
        symbols and directions, the pair labels of complex irreps merged with
        their conjugates, and a line when spglib could not identify a group.
    """
    lines = [
        "MSG: magnetic space group from the moment directions (BNS number, type I-IV);",
        "MPG: magnetic point group; net magnetization: the direction allowed by symmetry;",
        "class: from the signs of the moments along their common axis only (nonrelativistic).",
        "MSG and MPG symbols are in the standard BNS setting; directions are in the "
        "Cartesian axes of the input cell.",
    ]
    if any(c.field is not None and c.tag.startswith("comp") for c in analysis.components):
        lines.append("component: the tag of the listing and of the VESTA file name "
                     "(comp<N>: the N-th basis vector of its space).")
    for pair in analysis.pair_labels:
        lines.append(f"{pair}: a complex irrep and its conjugate, merged into one real "
                     "space (the physically irreducible representation).")
    if any(c.collinear is not None and c.group is not None and not c.group.identified
           for c in analysis.components):
        lines.append("MSG '-': spglib returned the operations but did not identify the group; "
                     "the responses come from the operations.")
    return lines


def format_magnetic_summary(analysis: SpinBasisAnalysis) -> list[str]:
    """The ``* Magnetic groups and collinear classes *`` block of crystod-mag.

    Args:
        analysis: A classified :class:`SpinBasisAnalysis`.

    Returns:
        The block as lines: title, unindented table, and the notes
        (:func:`magnetic_summary_notes`).
    """
    rows = [MAGNETIC_SUMMARY_HEADERS] + magnetic_summary_rows(analysis)
    widths = [max(len(row[c]) for row in rows) for c in range(len(rows[0]))]
    lines = ["* Magnetic groups and collinear classes *"]
    lines += ["  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip()
              for row in rows]
    lines += [""] + magnetic_summary_notes(analysis)
    return lines


if __name__ == "__main__":
    main()
