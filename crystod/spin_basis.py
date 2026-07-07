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
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray

from .spglib_compat import ensure_spglib_compat

ensure_spglib_compat()

from spgrep.core import get_spacegroup_irreps_from_primitive_symmetry
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
crystod --spin-basis --poscar 221_PPOSCAR_AlNi3 --element Ni --qpoint 0 0 0
crystod --spin-basis --poscar 221_PPOSCAR_AlNi3 --element Ni --qpoint GM --show-spin-direction
"""

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
        "(with a VASP noncollinear MAGMOM line).",
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
        help="Symmetry tolerance.",
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

def get_spin_representation(
    vibrations: SymmetryOnlyVibrations,
    site_indices: list[int],
    qpoint: list[float],
):
    """Irreps and the axial-vector representation on the selected sites at q.

    Same construction as the vibration representation, with the Cartesian part
    replaced by det(R) * R (spins are axial vectors) and the permutation
    restricted to the selected sites.
    """
    positions = vibrations.primitive_cell.scaled_positions
    irreps, mapping = get_spacegroup_irreps_from_primitive_symmetry(
        rotations=vibrations.rotations,
        translations=vibrations.translations,
        kpoint=qpoint,
    )
    little_rotations = vibrations.rotations[mapping]
    little_translations = vibrations.translations[mapping]

    n_sites = len(site_indices)
    representation = []
    for rotation, translation in zip(little_rotations, little_translations):
        permutation = np.zeros((n_sites, n_sites), dtype=complex)
        for a, index_a in enumerate(site_indices):
            transformed = rotation @ positions[index_a] + translation
            for b, index_b in enumerate(site_indices):
                diff = transformed - positions[index_b]
                if (np.abs(diff - np.rint(diff)) < 1e-5).all():
                    phase = np.dot(
                        qpoint,
                        np.dot(np.linalg.inv(rotation), positions[index_b] - translation)
                        - positions[index_b],
                    )
                    permutation[b, a] = np.exp(2j * np.pi * phase)
                    break
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
    """For each irrep, the ranks p at which it appears in the magnetic
    (time-odd, axial) rank-p multipole representation, in increasing order
    with multiplicity.

    The rank-p magnetic multipole transforms as the angular-momentum-p
    representation with parity (-1)^(p+1) under inversion: dipole (p=1) and
    octupole (p=3) are parity-even, the magnetic quadrupole (p=2) is
    parity-odd, etc.
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
    """Rotate each basis row by a global phase and return the real form,
    or None when the space is genuinely complex."""
    rows = []
    for row in space:
        pivot = row[np.argmax(np.abs(row))]
        aligned = row * np.exp(-1j * np.angle(pivot))
        if np.abs(aligned.imag).max() > 1e-8:
            return None
        rows.append(aligned.real)
    return np.array(rows)


def _net_moments(space: NDArray[np.float64], n_sites: int) -> NDArray[np.float64]:
    return np.array(
        [np.sum(space[c].reshape(n_sites, 3), axis=0) for c in range(space.shape[0])]
    )


def separate_ferro_combination(
    spaces: list[NDArray[np.float64]],
    representation: NDArray[np.complex128],
    n_sites: int,
) -> tuple[NDArray[np.float64] | None, list[NDArray[np.float64]]]:
    """Within a set of aligned spaces carrying the same irrep, split off the
    unique combination with a net moment (cluster dipole); the rest are
    net-zero (AFM). Returns (ferro_space_or_None, afm_spaces)."""
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

def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    from .star_of_k import read_poscar_or_exit

    cell = read_poscar_or_exit(args.poscar)
    vibrations = SymmetryOnlyVibrations(cell=cell, symprec=args.tolerance, standardize=False)
    primitive = vibrations.primitive_cell
    symbols = list(primitive.symbols)
    positions = primitive.scaled_positions
    lattice = np.array(primitive.cell)

    site_indices = [i for i, s in enumerate(symbols) if s == args.element]
    if not site_indices:
        available = ", ".join(sorted(set(symbols)))
        raise SystemExit(f"ERROR: element '{args.element}' is not in this POSCAR ({available}).")
    n_sites = len(site_indices)

    print(f"Space group: {vibrations.spglib_dataset['international']} "
          f"(#{vibrations.spglib_dataset['number']})")
    print(f"Magnetic sites: {args.element} x {n_sites}")
    for k, index in enumerate(site_indices):
        print(f"  {args.element}{k + 1}: {np.round(positions[index], 6).tolist()}")

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
        q_names, q_list = get_irt_special_points(irt_table, primitive_matrix)

        print(f"\n * Spin (axial-vector) Irreducible Representations of {args.element} sites *")
        for name, q in zip(q_names, q_list):
            irreps, spin_rep, mapping = get_spin_representation(vibrations, site_indices, q)
            labels = vibrations.get_irrep_labels(q, irreps, mapping)
            rep_characters = np.array([np.trace(matrix) for matrix in spin_rep])
            terms = []
            for irrep, label in zip(irreps, labels):
                characters = np.array(get_character(irrep), dtype=complex)
                count = int(round(float(np.real(np.dot(rep_characters, np.conj(characters)))) / len(spin_rep)))
                if count > 0:
                    terms.append(f"{count}.0 [{label}]")
            print(f"\n * k point (primitive) * \n {name} {q}")
            print(f"   {' + '.join(terms)}")
        print("\nUse --qpoint to obtain the symmetry-adapted spin bases, MAGMOM lines, and VESTA files at one k point.")
        return

    qpoint_label, qpoint = vibrations.resolve_qpoint(args.qpoint)
    is_gamma = np.allclose(qpoint, 0.0)
    print(f"\nSelected q-point: {qpoint_label} = {qpoint}")

    irreps, spin_rep, mapping = get_spin_representation(vibrations, site_indices, qpoint)
    irrep_labels = vibrations.get_irrep_labels(qpoint, irreps, mapping)

    raw_spaces: list[NDArray] = []
    raw_labels: list[str] = []
    raw_irrep_indices: list[int] = []
    for irrep_index, (irrep, label) in enumerate(zip(irreps, irrep_labels)):
        for space in project_to_irrep(spin_rep, irrep):
            realified = _realify(space)
            raw_spaces.append(realified if realified is not None else space)
            raw_labels.append(label)
            raw_irrep_indices.append(irrep_index)

    total_dim = sum(space.shape[0] for space in raw_spaces)
    print(f"\nSpin (axial-vector) representation on {args.element} sites: "
          f"{3 * n_sites} dimensions")
    counted: dict[str, int] = {}
    for label, space in zip(raw_labels, raw_spaces):
        counted[label] = counted.get(label, 0) + 1
    print("Decomposition: " + " + ".join(
        (f"{count} x {label}" if count > 1 else label) for label, count in counted.items()
    ))
    if total_dim != 3 * n_sites:
        print(f"WARNING: projected dimensions ({total_dim}) do not span the full space.")

    # FM/AFM separation (Gamma only) and multipole-rank naming
    entries = []  # (label, kind, rank_name, space)
    if is_gamma:
        rank_lists = get_multipole_rank_lists(
            vibrations, vibrations.rotations[mapping], irreps
        )
        label_to_irrep_index: dict[str, int] = {}
        for label, irrep_index in zip(raw_labels, raw_irrep_indices):
            label_to_irrep_index.setdefault(label, irrep_index)

        for label in dict.fromkeys(raw_labels):
            group = [space for space, l in zip(raw_spaces, raw_labels) if l == label]
            ferro, afm_list = separate_ferro_combination(group, spin_rep, n_sites)
            ordered = []
            if ferro is not None:
                ordered.append(("FM", np.real(ferro)))
            ordered.extend(("AFM", np.real(afm)) for afm in afm_list)

            # assign multipole ranks of this irrep in increasing order
            # (FM = lowest rank = dipole, then octupole, ...)
            ranks = rank_lists[label_to_irrep_index[label]]
            for position, (kind, space) in enumerate(ordered):
                rank = ranks[position] if position < len(ranks) else None
                rank_name = MULTIPOLE_NAMES.get(rank, f"rank{rank}") if rank else "multipole"
                entries.append((label, kind, rank_name, space))
    else:
        for label, space in zip(raw_labels, raw_spaces):
            entries.append((label, "AFM (q != 0)", None, space))

    print("\nSymmetry-adapted spin bases:")
    for label, kind, rank_name, space in entries:
        rank_text = f", {rank_name}" if rank_name else ""
        print(f"  {label}: dim {space.shape[0]} [{kind}{rank_text}]")

    # commensurate (magnetic) supercell for q != 0
    if is_gamma:
        supercell_sizes = np.array([1, 1, 1], dtype=int)
    else:
        supercell_sizes = np.array(
            [
                1 if abs(component) < 1e-10
                else Fraction(float(component)).limit_denominator(12).denominator
                for component in qpoint
            ],
            dtype=int,
        )
        print(
            f"\nMagnetic (commensurate) supercell: "
            f"{supercell_sizes[0]}x{supercell_sizes[1]}x{supercell_sizes[2]}"
        )
    cell_translations = [
        np.array(t)
        for t in np.ndindex(int(supercell_sizes[0]), int(supercell_sizes[1]), int(supercell_sizes[2]))
    ]
    supercell_lattice = np.diag(supercell_sizes) @ lattice
    supercell_symbols: list[str] = []
    supercell_positions: list[NDArray[np.float64]] = []
    supercell_site_of_atom: list[int | None] = []  # magnetic-site index or None
    for translation in cell_translations:
        for i, symbol in enumerate(symbols):
            supercell_symbols.append(symbol)
            supercell_positions.append((positions[i] + translation) / supercell_sizes)
            supercell_site_of_atom.append(site_indices.index(i) if i in site_indices else None)
    supercell_positions = np.array(supercell_positions)
    cell_phases = [np.exp(2j * np.pi * np.dot(qpoint, t)) for t in cell_translations]

    # detailed output + VESTA export
    formula = reduced_formula(symbols)
    used_names: set[str] = set()
    print()
    for label, kind, rank_name, space in entries:
        short_label = label.split("(")[0].strip()
        component_tags = []
        component_vectors = []
        for c in range(space.shape[0]):
            vector = np.asarray(space[c])
            result = _direction_tag(np.real(vector), n_sites) if np.abs(np.imag(vector)).max() < 1e-8 else None
            if result is None:
                component_tags.append(f"comp{c}")
                component_vectors.append(vector)
            else:
                tag, sign = result
                component_tags.append(tag)
                component_vectors.append(sign * vector)

        header = f"{label} [{kind}" + (f", {rank_name}]" if rank_name else "]")
        print(f"=== {header} ===")
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
                print(f"  component {tag}: vanishing real spin field; skipped.")
                continue
            field = field / max_norm

            print(f"  component {tag}:")
            for k in range(n_sites):
                s = field[0, k]
                print(f"    {args.element}{k + 1}: S = [{s[0]: .4f}, {s[1]: .4f}, {s[2]: .4f}]")
            if is_gamma:
                net = np.sum(field[0], axis=0)
                print(f"    net moment: {np.round(net, 6).tolist()}")
            else:
                net = np.sum(field, axis=(0, 1))
                print(f"    net moment over the magnetic supercell: {np.round(net, 6).tolist()}")

            if args.show_spin_direction:
                magmom_parts = []
                for atom_index, site in enumerate(supercell_site_of_atom):
                    if site is None:
                        magmom_parts.append("0 0 0")
                    else:
                        cell_index = atom_index // len(symbols)
                        s = field[cell_index, site]
                        magmom_parts.append(f"{s[0]:.4f} {s[1]:.4f} {s[2]:.4f}")
                cell_note = "" if is_gamma else f" ({supercell_sizes.tolist()} magnetic supercell)"
                print(f"    spin directions (all atoms, POSCAR order){cell_note}:")
                for atom_index, s_text in enumerate(magmom_parts):
                    print(f"      {supercell_symbols[atom_index]}{atom_index + 1}: "
                          f"[{s_text.replace(' ', ', ')}]")
                print(f"    MAGMOM = {'   '.join(magmom_parts)}")

            # VESTA export
            rank_tag = rank_name if rank_name else ("FM" if kind == "FM" else qpoint_label)
            name = f"POSCAR_{formula}_spin_{short_label}_{rank_tag}_{tag}.vesta"
            if name in used_names:
                stem = name[: -len(".vesta")]
                suffix = 2
                while f"{stem}_{suffix}.vesta" in used_names:
                    suffix += 1
                name = f"{stem}_{suffix}.vesta"
            used_names.add(name)

            arrows = np.zeros((len(supercell_symbols), 3))
            for atom_index, site in enumerate(supercell_site_of_atom):
                if site is not None:
                    cell_index = atom_index // len(symbols)
                    arrows[atom_index] = field[cell_index, site] * args.amplitude
            write_vesta_with_arrows(
                filepath=name,
                lattice=supercell_lattice,
                scaled_positions=supercell_positions,
                symbols=supercell_symbols,
                arrows_cartesian=arrows,
                title=f"{formula} spin basis: {label} {rank_tag} {tag}",
            )
            print(f"    written to: {name}")
        print()

    if is_gamma:
        print("All AFM bases satisfy sum_i S_i = 0; FM entries carry the cluster dipole moment.")
    print("Reference: M.-T. Suzuki et al., PRB 95, 094406 (2017); PRB 99, 174407 (2019).")


if __name__ == "__main__":
    main()
