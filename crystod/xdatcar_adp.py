"""
XDATCAR -> ADP (Anisotropic Displacement Parameters) workflow for crystod.

Computes time-averaged atomic positions and symmetry-constrained ADPs (U_ij)
from a molecular-dynamics XDATCAR trajectory and writes them as a CIF file.
Based on script/xdatcar_to_adp.py by Sato (Mochizuki group).
"""

from __future__ import annotations

from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    RawDescriptionHelpFormatter,
    RawTextHelpFormatter,
)
from collections import Counter, defaultdict

import numpy as np
import spglib
from numpy.typing import NDArray

from .runtime_compat import SymmetryDatasetAdapter


class MyHelpFormatter(
    RawTextHelpFormatter,
    RawDescriptionHelpFormatter,
    ArgumentDefaultsHelpFormatter,
):
    pass


desc = """
Compute anisotropic displacement parameters (ADPs) from an MD trajectory
(XDATCAR) and write the time-averaged structure with symmetry-constrained
U_ij tensors as a CIF file.

# Command Example:
crystod-md --adp --dim 4 4 4 --start-step 1000 --xdatcar XDATCAR --output ADP.cif
"""

# U-component order used throughout: [U11, U22, U33, U12, U13, U23]
U_INDICES = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
U_NAMES = ["U11", "U22", "U33", "U12", "U13", "U23"]


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=desc, formatter_class=MyHelpFormatter)
    parser.add_argument(
        "--dim",
        required=True,
        type=str,
        help='MD supercell dimension relative to the unit cell, e.g. "4 4 4" (diagonal only).',
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=0,
        help="First MD step used in the analysis (earlier steps are discarded as equilibration).",
    )
    parser.add_argument(
        "--xdatcar",
        type=str,
        default="XDATCAR",
        help="Input XDATCAR path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ADP.cif",
        help="Output CIF path.",
    )
    parser.add_argument(
        "--symprec",
        "--tolerance",
        dest="symprec",
        type=float,
        default=0.1,
        help="Symmetry tolerance for spglib on the time-averaged structure.",
    )
    parser.add_argument(
        "--grouping-tolerance",
        type=float,
        default=0.1,
        help="Tolerance for grouping supercell atoms into unit-cell sites.",
    )
    return parser


# =============================================================================
# XDATCAR reader (handles both single-header and repeated-header/NpT formats)
# =============================================================================

def _read_header(lines: list[str], start: int) -> tuple[NDArray[np.float64], list[str], int, int]:
    scale = float(lines[start + 1].split()[0])
    lattice = scale * np.array(
        [lines[start + 2 + k].split()[:3] for k in range(3)], dtype=float
    )
    symbols = lines[start + 5].split()
    counts = [int(value) for value in lines[start + 6].split()]
    chem_formula = [symbol for symbol, count in zip(symbols, counts) for _ in range(count)]
    return lattice, chem_formula, sum(counts), start + 7


def read_xdatcar(path: str) -> tuple[list[str], list[NDArray[np.float64]], NDArray[np.float64]]:
    """Read an XDATCAR trajectory.

    Returns (chem_formula, per-frame lattices, fractional coordinates with
    shape (n_frames, n_atoms, 3)).
    """
    with open(path) as fp:
        lines = fp.read().splitlines()

    lattice, chem_formula, n_atoms, index = _read_header(lines, 0)
    lattices: list[NDArray[np.float64]] = []
    frames: list[NDArray[np.float64]] = []
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.lower().startswith("direct"):
            block = lines[index + 1 : index + 1 + n_atoms]
            try:
                coords = np.array(" ".join(block).split(), dtype=float).reshape(n_atoms, -1)[:, :3]
            except ValueError:
                # truncated or corrupt trailing frame
                break
            frames.append(coords)
            lattices.append(lattice)
            index += 1 + n_atoms
        else:
            # repeated header (variable-cell / NpT trajectory)
            if index + 7 > len(lines):
                break
            lattice, chem_formula, n_atoms, index = _read_header(lines, index)

    if not frames:
        raise ValueError(f"No configurations found in '{path}'.")
    return chem_formula, lattices, np.array(frames)


# =============================================================================
# ADP symmetry-constraint helpers
# =============================================================================

def build_symmetry_projector(rotations) -> NDArray[np.float64]:
    """6x6 projector onto U tensors invariant under the site-symmetry rotations.

    Acts on the component vector [U11, U22, U33, U12, U13, U23] (fractional
    axes); applying it yields the symmetry-constrained U matrix.
    """
    averaged = np.zeros((6, 6))
    for rotation in rotations:
        transform = np.zeros((6, 6))
        for src_index, (i, j) in enumerate(U_INDICES):
            u_basis = np.zeros((3, 3))
            u_basis[i, j] = 1.0
            u_basis[j, i] = 1.0
            u_transformed = rotation @ u_basis @ rotation.T
            for dst_index, (di, dj) in enumerate(U_INDICES):
                transform[dst_index, src_index] = u_transformed[di, dj]
        averaged += transform
    return averaged / len(rotations)


def apply_symmetry_constraints(u_cryst: NDArray[np.float64], projector: NDArray[np.float64]) -> NDArray[np.float64]:
    u_vector = np.array([u_cryst[i, j] for i, j in U_INDICES])
    p = projector @ u_vector
    return np.array(
        [
            [p[0], p[3], p[4]],
            [p[3], p[1], p[5]],
            [p[4], p[5], p[2]],
        ]
    )


def get_constraint_description(projector: NDArray[np.float64], tol: float = 1e-6) -> str:
    constraints = []
    for i in range(6):
        unit = np.zeros(6)
        unit[i] = 1.0
        projected = projector @ unit
        if abs(projected[i]) < tol:
            constraints.append(f"{U_NAMES[i]}=0")
        else:
            for j in range(i + 1, 6):
                if abs(projected[j]) > tol:
                    if abs(projected[j] - projected[i]) < tol:
                        constraints.append(f"{U_NAMES[i]}={U_NAMES[j]}")
                    elif abs(projected[j] + projected[i]) < tol:
                        constraints.append(f"{U_NAMES[i]}=-{U_NAMES[j]}")
    return ", ".join(constraints) if constraints else "no constraint"


def get_site_symmetry_operations(coords, rotations, translations, symprec: float = 0.1):
    site_operations = []
    for rotation, translation in zip(rotations, translations):
        transformed = (rotation @ coords + translation) % 1.0
        diff = transformed - coords
        diff = diff - np.round(diff)
        if np.linalg.norm(diff) < symprec:
            site_operations.append(rotation)
    return site_operations


def rotation_to_xyz(rotation, translation) -> str:
    """Convert a rotation matrix and translation vector to a CIF xyz string."""
    axes = ["x", "y", "z"]
    result = []
    for i in range(3):
        terms = []
        for j in range(3):
            if rotation[i, j] == 1:
                terms.append(axes[j])
            elif rotation[i, j] == -1:
                terms.append(f"-{axes[j]}")
            elif rotation[i, j] != 0:
                terms.append(f"{rotation[i, j]}*{axes[j]}")
        if abs(translation[i]) > 1e-6:
            frac = translation[i]
            for value, text in ((0.5, "1/2"), (0.25, "1/4"), (0.75, "3/4"), (1 / 3, "1/3"), (2 / 3, "2/3")):
                if abs(frac - value) < 1e-6:
                    terms.append(text)
                    break
            else:
                terms.append(f"{frac:.4f}")
        if not terms:
            terms = ["0"]
        result.append("+".join(terms).replace("+-", "-"))
    return ", ".join(result)


def _circular_mean(values: NDArray[np.float64], period: float = 1.0) -> float:
    """Mean of periodic values (period-aware, robust against boundary wrapping)."""
    angles = values * 2 * np.pi / period
    mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    return (mean_angle / (2 * np.pi) * period) % period


# =============================================================================
# Main workflow
# =============================================================================

def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    supercell_size = np.array([int(value) for value in args.dim.split()], dtype=int)
    if supercell_size.shape != (3,):
        raise SystemExit('ERROR: --dim requires three integers, e.g. --dim "4 4 4".')

    print(f"Supercell size : {supercell_size.tolist()}")
    print(f"Start step     : {args.start_step}")
    print(f"Input file     : {args.xdatcar}")
    print(f"Output file    : {args.output}")

    # Step 1: read XDATCAR
    print("\nReading XDATCAR... (this may take a while)")
    chem_formula, lattices, all_coordinates = read_xdatcar(args.xdatcar)

    if args.start_step >= len(all_coordinates):
        raise SystemExit(
            f"ERROR: --start-step {args.start_step} exceeds the number of MD steps "
            f"({len(all_coordinates)})."
        )
    coordinates = all_coordinates[args.start_step :]
    n_steps, n_atoms_super, _ = coordinates.shape
    lattice_vec_super = lattices[args.start_step]
    lattice_vec_unit = lattice_vec_super / supercell_size[:, np.newaxis]

    composition = Counter(chem_formula)
    print("\nSupercell info:")
    print(f"  atoms          : {n_atoms_super}")
    print(f"  composition    : {dict(composition)}")
    print(f"  analyzed steps : {n_steps}")

    # Step 2: time-averaged coordinates (circular mean per axis)
    avg_coords = np.zeros((n_atoms_super, 3))
    for i in range(n_atoms_super):
        for axis in range(3):
            avg_coords[i, axis] = _circular_mean(coordinates[:, i, axis])

    # Step 3: fold into one unit cell of the supercell
    fold_thresholds = 1.0 / supercell_size
    folded_coords = avg_coords % fold_thresholds

    # Step 4: group supercell atoms into unit-cell sites
    grouped_indices = []
    remaining_indices = set(range(n_atoms_super))
    adjusted_tolerance = args.grouping_tolerance / np.mean(supercell_size)
    while remaining_indices:
        ref_index = min(remaining_indices)
        ref_coord = folded_coords[ref_index]
        ref_element = chem_formula[ref_index]
        group = []
        for index in list(remaining_indices):
            if chem_formula[index] != ref_element:
                continue
            diff = ref_coord - folded_coords[index]
            for axis in range(3):
                threshold = fold_thresholds[axis]
                diff[axis] = diff[axis] - np.round(diff[axis] / threshold) * threshold
            if np.linalg.norm(diff) < adjusted_tolerance:
                group.append(index)
                remaining_indices.remove(index)
        grouped_indices.append(group)

    print(f"\nNumber of unit-cell sites (groups): {len(grouped_indices)}")
    print(f"Expected replicas per group       : {int(np.prod(supercell_size))}")

    # Step 5: averaged unit-cell coordinate of each group
    group_unitcell_coords = []
    group_elements = []
    for atom_indices in grouped_indices:
        group_elements.append(chem_formula[atom_indices[0]])
        avg_folded = np.array(
            [
                _circular_mean(folded_coords[atom_indices][:, axis], period=fold_thresholds[axis])
                for axis in range(3)
            ]
        )
        group_unitcell_coords.append((avg_folded * supercell_size) % 1.0)
    group_unitcell_coords = np.array(group_unitcell_coords)

    # Step 6: symmetry analysis of the averaged unit cell
    unique_elements = sorted(set(group_elements))
    element_to_number = {element: i + 1 for i, element in enumerate(unique_elements)}
    atom_numbers = [element_to_number[element] for element in group_elements]
    cell = (lattice_vec_unit, group_unitcell_coords, atom_numbers)
    raw_dataset = spglib.get_symmetry_dataset(cell, symprec=args.symprec)
    if raw_dataset is None:
        raise SystemExit("ERROR: spglib could not find any symmetry for the averaged structure.")
    dataset = SymmetryDatasetAdapter(raw_dataset)

    spacegroup = dataset["international"]
    spacegroup_number = dataset["number"]
    rotations = np.array(dataset["rotations"])
    translations = np.array(dataset["translations"])
    equivalent_atoms = np.array(dataset["equivalent_atoms"])

    print(f"\nSpace group: {spacegroup} (No. {spacegroup_number})")
    print(f"Atoms in unit cell   : {len(group_unitcell_coords)}")
    print(f"Symmetry operations  : {len(rotations)}")

    wyckoff_ids_sorted = sorted(set(equivalent_atoms))
    wyckoff_to_cif_index = {wid: i for i, wid in enumerate(wyckoff_ids_sorted)}
    print(f"Asymmetric-unit sites: {len(wyckoff_ids_sorted)}")

    asym_sites = []
    for cif_index, wyckoff_id in enumerate(wyckoff_ids_sorted):
        element = group_elements[wyckoff_id]
        coords = group_unitcell_coords[wyckoff_id]
        multiplicity = int(np.sum(equivalent_atoms == wyckoff_id))
        asym_sites.append(
            {
                "label": f"{element}{cif_index}",
                "element": element,
                "frac_coords": coords,
                "multiplicity": multiplicity,
            }
        )
        print(
            f"  {asym_sites[-1]['label']}: mult={multiplicity}, "
            f"coords=({coords[0]:.5f}, {coords[1]:.5f}, {coords[2]:.5f})"
        )

    # ADP constraints per Wyckoff position
    wyckoff_constraints = {}
    wyckoff_projectors = {}
    print("\nADP constraints per Wyckoff position:")
    for wyckoff_id in wyckoff_ids_sorted:
        coords = group_unitcell_coords[wyckoff_id]
        site_operations = get_site_symmetry_operations(coords, rotations, translations, args.symprec)
        projector = build_symmetry_projector(site_operations)
        wyckoff_projectors[wyckoff_id] = projector
        constraint = get_constraint_description(projector)
        wyckoff_constraints[wyckoff_id] = constraint
        element = group_elements[wyckoff_id]
        print(
            f"  Wyckoff {wyckoff_id} ({element}): site-symmetry order={len(site_operations)}, {constraint}"
        )

    # Step 9: unwrap trajectories into unit-cell Cartesian coordinates
    lattice_inv = np.linalg.inv(lattice_vec_unit)
    all_atom_cart_coords = []
    all_atom_group_ids = []
    for group_index, atom_indices in enumerate(grouped_indices):
        for atom_index in atom_indices:
            supercell_frac = coordinates[:, atom_index, :] % 1.0
            unitcell_frac = supercell_frac @ lattice_vec_super @ lattice_inv
            steps_diff = np.diff(unitcell_frac, axis=0)
            steps_diff -= np.round(steps_diff)
            unwrapped = np.empty_like(unitcell_frac)
            unwrapped[0] = unitcell_frac[0] - np.round(unitcell_frac[0])
            if n_steps > 1:
                unwrapped[1:] = unwrapped[0] + np.cumsum(steps_diff, axis=0)
            all_atom_cart_coords.append(unwrapped @ lattice_vec_unit)
            all_atom_group_ids.append(group_index)
    all_atom_cart_coords = np.array(all_atom_cart_coords)
    all_atom_group_ids = np.array(all_atom_group_ids)
    print(f"\nCoordinate unwrapping done: {len(all_atom_cart_coords)} atoms x {n_steps} steps")

    # Step 10: map every atom onto its Wyckoff representative
    all_atom_wyckoff_ids = []
    all_atom_symop_ids = []
    for atom_index in range(len(all_atom_cart_coords)):
        group_index = all_atom_group_ids[atom_index]
        wyckoff_id = equivalent_atoms[group_index]
        all_atom_wyckoff_ids.append(wyckoff_id)
        coord = group_unitcell_coords[group_index]
        representative = group_unitcell_coords[wyckoff_id]
        found = None
        for symop_index, (rotation, translation) in enumerate(zip(rotations, translations)):
            transformed = (rotation @ coord + translation) % 1.0
            diff = transformed - representative
            diff = diff - np.round(diff)
            if np.linalg.norm(diff) < 0.01:
                found = symop_index
                break
        all_atom_symop_ids.append(found)

    # Step 11: collect displacements rotated into the representative frame
    volume = np.linalg.det(lattice_vec_unit)
    a_star = np.cross(lattice_vec_unit[1], lattice_vec_unit[2]) / volume
    b_star = np.cross(lattice_vec_unit[2], lattice_vec_unit[0]) / volume
    c_star = np.cross(lattice_vec_unit[0], lattice_vec_unit[1]) / volume
    reciprocal_unit = np.array(
        [
            a_star / np.linalg.norm(a_star),
            b_star / np.linalg.norm(b_star),
            c_star / np.linalg.norm(c_star),
        ]
    )

    wyckoff_displacements = defaultdict(list)
    for atom_index in range(len(all_atom_cart_coords)):
        symop_index = all_atom_symop_ids[atom_index]
        if symop_index is None:
            continue
        atom_cart = all_atom_cart_coords[atom_index]
        displacements = atom_cart - np.mean(atom_cart, axis=0)
        rotation_cart = lattice_vec_unit.T @ rotations[symop_index] @ np.linalg.inv(lattice_vec_unit.T)
        wyckoff_displacements[all_atom_wyckoff_ids[atom_index]].append(displacements @ rotation_cart.T)
    for wyckoff_id in list(wyckoff_displacements.keys()):
        wyckoff_displacements[wyckoff_id] = np.concatenate(wyckoff_displacements[wyckoff_id], axis=0)

    # Step 12: ADP tensors
    wyckoff_averaged_u_cryst = {}
    print(f"\n{'Site':<10} {'Ueq (A^2)':<12} {'Constraint':<35}")
    print("-" * 60)
    for wyckoff_id in wyckoff_ids_sorted:
        displacements_all = wyckoff_displacements[wyckoff_id]
        u_cart = np.cov(displacements_all.T, bias=True)
        u_cryst_before = reciprocal_unit @ u_cart @ reciprocal_unit.T
        u_cryst = apply_symmetry_constraints(u_cryst_before, wyckoff_projectors[wyckoff_id])
        wyckoff_averaged_u_cryst[wyckoff_id] = u_cryst

        label = asym_sites[wyckoff_to_cif_index[wyckoff_id]]["label"]
        u_eq = float(np.mean(np.real(np.linalg.eigvals(u_cart))))
        print(f"{label:<10} {u_eq:<12.6f} {wyckoff_constraints[wyckoff_id]:<35}")

    # Step 13: CIF output
    a, b, c = np.linalg.norm(lattice_vec_unit, axis=1)
    alpha = np.degrees(np.arccos(np.dot(lattice_vec_unit[1], lattice_vec_unit[2]) / (b * c)))
    beta = np.degrees(np.arccos(np.dot(lattice_vec_unit[0], lattice_vec_unit[2]) / (a * c)))
    gamma = np.degrees(np.arccos(np.dot(lattice_vec_unit[0], lattice_vec_unit[1]) / (a * b)))

    element_counts = Counter(group_elements)
    formula = "".join(
        f"{element}{count}" if count > 1 else element for element, count in sorted(element_counts.items())
    )
    formula_sum = " ".join(f"{element}{count}" for element, count in sorted(element_counts.items()))

    cif = []
    cif.append("# generated using crystod --xdatcar2adp")
    cif.append(f"data_{formula}")
    cif.append(f"_symmetry_space_group_name_H-M   '{spacegroup}'")
    cif.append(f"_cell_length_a   {a:.8f}")
    cif.append(f"_cell_length_b   {b:.8f}")
    cif.append(f"_cell_length_c   {c:.8f}")
    cif.append(f"_cell_angle_alpha   {alpha:.8f}")
    cif.append(f"_cell_angle_beta   {beta:.8f}")
    cif.append(f"_cell_angle_gamma   {gamma:.8f}")
    cif.append(f"_symmetry_Int_Tables_number   {spacegroup_number}")
    cif.append(f"_chemical_formula_structural   {formula}")
    cif.append(f"_chemical_formula_sum   '{formula_sum}'")
    cif.append(f"_cell_volume   {volume:.8f}")
    cif.append("_cell_formula_units_Z   1")
    cif.append("")
    cif.append("loop_")
    cif.append(" _symmetry_equiv_pos_site_id")
    cif.append(" _symmetry_equiv_pos_as_xyz")
    for i, (rotation, translation) in enumerate(zip(rotations, translations)):
        cif.append(f"  {i + 1}  '{rotation_to_xyz(rotation, translation)}'")
    cif.append("")
    cif.append("loop_")
    cif.append(" _atom_site_type_symbol")
    cif.append(" _atom_site_label")
    cif.append(" _atom_site_symmetry_multiplicity")
    cif.append(" _atom_site_fract_x")
    cif.append(" _atom_site_fract_y")
    cif.append(" _atom_site_fract_z")
    cif.append(" _atom_site_occupancy")
    for site in asym_sites:
        coords = site["frac_coords"]
        cif.append(
            f"  {site['element']}  {site['label']}  {site['multiplicity']}  "
            f"{coords[0]:.8f}  {coords[1]:.8f}  {coords[2]:.8f}  1"
        )
    cif.append("")
    cif.append("loop_")
    cif.append(" _atom_site_aniso_label")
    cif.append(" _atom_site_aniso_U_11")
    cif.append(" _atom_site_aniso_U_22")
    cif.append(" _atom_site_aniso_U_33")
    cif.append(" _atom_site_aniso_U_23")
    cif.append(" _atom_site_aniso_U_13")
    cif.append(" _atom_site_aniso_U_12")
    for wyckoff_id in wyckoff_ids_sorted:
        u = wyckoff_averaged_u_cryst[wyckoff_id]
        label = asym_sites[wyckoff_to_cif_index[wyckoff_id]]["label"]
        cif.append(
            f" {label:<8}{u[0, 0]:<10.5f}{u[1, 1]:<10.5f}{u[2, 2]:<10.5f}"
            f"{u[1, 2]:<10.5f}{u[0, 2]:<10.5f}{u[0, 1]:<10.5f}"
        )

    with open(args.output, "w") as fp:
        fp.write("\n".join(cif))

    print(f"\nSaved: {args.output}")
    print(f"Asymmetric-unit sites: {len(wyckoff_ids_sorted)}")


if __name__ == "__main__":
    main()
