"""
Symmetry-only vibration basis workflow for crystod.
"""

from __future__ import annotations

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, RawDescriptionHelpFormatter, RawTextHelpFormatter
from fractions import Fraction
from pathlib import Path

import numpy as np
import spglib
from ase import Atoms
from ase.io import write as ase_write
from numpy.typing import NDArray

from phonopy.structure.cells import get_primitive_matrix_by_centring

from .irreptables_compat import load_irreptables
from .runtime_compat import (
    SymmetryDatasetAdapter,
    get_character,
    get_chemical_symbols,
    get_little_group,
    get_scaled_positions,
)
from .spglib_compat import ensure_spglib_compat

ensure_spglib_compat()

from phonopy.interface.calculator import read_crystal_structure
from phonopy.structure.atoms import PhonopyAtoms
from spgrep.core import get_spacegroup_irreps_from_primitive_symmetry
from spgrep.representation import project_to_irrep

IrrepTable, Irrep = load_irreptables()


class MyHelpFormatter(
    RawTextHelpFormatter,
    RawDescriptionHelpFormatter,
    ArgumentDefaultsHelpFormatter,
):
    pass


desc = """
Construct symmetry-allowed vibration basis vectors without phonon force data.

# Command Examples:
python3 vibration_beta.py --poscar example/test_POSCARs/221_PPOSCAR_ScF3 --qpoint 0.5 0.5 0.5
python3 vibration_beta.py --poscar example/test_POSCARs/221_PPOSCAR_ScF3 --qpoint R --mode-index 2 --component-index 0 --output POSCAR_vibration
"""


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=desc, formatter_class=MyHelpFormatter)
    parser.add_argument(
        "--poscar",
        default="POSCAR",
        help="POSCAR path.",
    )
    parser.add_argument(
        "--qpoint",
        nargs="+",
        default=None,
        help="Either a high-symmetry label such as GM/X/M/R or three primitive reciprocal coordinates.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Symmetry tolerance.",
    )
    parser.add_argument(
        "--list-qpoints",
        action="store_true",
        help="Only list available high-symmetry q-points and exit.",
    )
    parser.add_argument(
        "--mode-index",
        type=int,
        default=None,
        help="Irrep-grouped mode-space index to inspect.",
    )
    parser.add_argument(
        "--component-index",
        type=int,
        default=0,
        help="Component index inside the selected degenerate mode space.",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.3,
        help="Amplitude used when writing a displaced structure.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output POSCAR path for the selected mode/component.",
    )
    parser.add_argument(
        "--export-npz",
        default=None,
        help="Optional .npz path to save positions, displacements, symbols, and lattice.",
    )
    return parser


class _CoreRepresentation:
    def __init__(self, cell: PhonopyAtoms, symprec: float = 1e-5, standardize: bool = True):
        if standardize:
            primitive_lattice, primitive_pos, primitive_numbers = spglib.standardize_cell(
                cell.totuple(),
                to_primitive=True,
                symprec=symprec,
            )
            self.primitive_cell = PhonopyAtoms(
                numbers=primitive_numbers,
                scaled_positions=primitive_pos,
                cell=primitive_lattice,
            )
            print("\n ### Inputed cell was converted into primitive cell. ###")
        else:
            # Keep the input cell as-is (it must already be primitive). This
            # preserves the caller's atom positions so that phase conventions
            # stay consistent with an externally built dynamical matrix.
            self.primitive_cell = cell

        dataset = SymmetryDatasetAdapter(
            spglib.get_symmetry_dataset(self.primitive_cell.totuple(), symprec=symprec)
        )
        self.spglib_dataset = dataset
        self.rotations = dataset.rotations
        self.translations = dataset.translations

    def get_modified_permutation_rep(
        self,
        rotation: NDArray[np.int_],
        translation: NDArray[np.float64],
        kpoint: list[float],
    ) -> NDArray[np.complex128]:
        positions = get_scaled_positions(self.primitive_cell)
        num_atom = len(positions)
        matrix = np.zeros((num_atom, num_atom), dtype=complex)
        for i, pos_in in enumerate(positions):
            pos_rot = np.dot(rotation, pos_in) + translation
            for j, pos_out in enumerate(positions):
                diff = pos_rot - pos_out
                if (abs(diff - np.rint(diff)) < 1e-5).all():
                    phase_factor = np.dot(
                        kpoint,
                        np.dot(np.linalg.inv(rotation), pos_out - translation) - pos_out,
                    )
                    matrix[j, i] = np.exp(2j * np.pi * phase_factor)
        return matrix

    def get_permutation_reps_at_k(
        self,
        little_rotations: NDArray[np.int_],
        little_translations: NDArray[np.float64],
        kpoint: list[float],
    ) -> NDArray[np.complex128]:
        return np.array(
            [
                self.get_modified_permutation_rep(rotation, translation, kpoint)
                for rotation, translation in zip(little_rotations, little_translations)
            ],
            dtype=np.complex128,
        )

    def get_little_group(self, kpoint: list[float]):
        return get_little_group(
            rotations=self.rotations,
            translations=self.translations,
            kpoint=kpoint,
        )


class SymmetryOnlyVibrations(_CoreRepresentation):
    def __init__(self, cell: PhonopyAtoms, symprec: float = 1e-5, standardize: bool = True):
        super().__init__(cell=cell, symprec=symprec, standardize=standardize)
        lattice_t = np.transpose(self.primitive_cell.cell)
        lattice_t_inv = np.linalg.inv(lattice_t)
        self.rotations_cartesian = np.array(
            [lattice_t @ rotation @ lattice_t_inv for rotation in self.rotations],
            dtype=np.complex128,
        )

    def get_high_symmetry_qpoints(self) -> dict[str, list[float]]:
        import seekpath
        import warnings

        structure = (
            self.primitive_cell.cell,
            self.primitive_cell.scaled_positions,
            self.primitive_cell.numbers,
        )
        path_data = seekpath.get_path(structure, symprec=1e-5)
        if not np.allclose(self.primitive_cell.cell, path_data["primitive_lattice"], atol=1e-4):
            warnings.warn(
                "The primitive cell from seekpath does not match the spglib primitive cell. "
                "The q-point coordinates might need a basis transformation.",
                stacklevel=2,
            )
        return path_data["point_coords"]

    def resolve_qpoint(self, raw_qpoint: list[str]) -> tuple[str, list[float]]:
        qpoint_map = self.get_high_symmetry_qpoints()
        alias_map = {
            "GM": "GAMMA",
            "G": "GAMMA",
            "Γ": "GAMMA",
        }

        if len(raw_qpoint) == 1:
            requested = raw_qpoint[0].strip().upper()
            requested = alias_map.get(requested, requested)
            if requested in qpoint_map:
                return requested, list(qpoint_map[requested])
            available = ", ".join(sorted(qpoint_map))
            raise ValueError(
                f"Unknown q-point label '{raw_qpoint[0]}'. Available labels: {available}"
            )

        if len(raw_qpoint) != 3:
            raise ValueError("--qpoint must be either one label or three coordinates.")

        qpoint = [float(value) for value in raw_qpoint]
        matched_label = None
        for label, coords in qpoint_map.items():
            if np.allclose(qpoint, coords, atol=1e-8):
                matched_label = label
                break
        return matched_label or "custom", qpoint

    def get_vibration_rep(self, kpoint: list[float]):
        irreps, mapping_little_group = get_spacegroup_irreps_from_primitive_symmetry(
            rotations=self.rotations,
            translations=self.translations,
            kpoint=kpoint,
        )
        little_rotations = self.rotations[mapping_little_group]
        little_translations = self.translations[mapping_little_group]
        permutation_matrices = self.get_permutation_reps_at_k(
            little_rotations=little_rotations,
            little_translations=little_translations,
            kpoint=kpoint,
        )
        cartesian_rep = self.rotations_cartesian[mapping_little_group]
        vibration_rep = np.array(
            [
                np.kron(permutation_matrix, cartesian_rotation)
                for permutation_matrix, cartesian_rotation in zip(permutation_matrices, cartesian_rep)
            ],
            dtype=np.complex128,
        )
        return irreps, vibration_rep, mapping_little_group

    def get_vibration_basis(
        self,
        irreps,
        vibration_rep,
        irrep_labels: list[str] | None = None,
    ) -> tuple[list[NDArray[np.complex128]], list[str]]:
        basis_vectors: list[NDArray[np.complex128]] = []
        basis_labels: list[str] = []
        fallback_labels = irrep_labels or [f"irrep_{index + 1}({irrep.shape[1]})" for index, irrep in enumerate(irreps)]
        for irrep, irrep_label in zip(irreps, fallback_labels):
            projected_spaces = project_to_irrep(vibration_rep, irrep)
            basis_vectors.extend(projected_spaces)
            basis_labels.extend([irrep_label] * len(projected_spaces))
        return basis_vectors, basis_labels

    def _get_irt_irreps_at_q(self, qpoint: list[float], irt_table, prim_mat) -> list[Irrep]:
        irreps_at_q = []
        prim_inv = np.linalg.inv(prim_mat)
        conventional_q = np.array(qpoint) @ prim_inv
        for irrep_at_q in irt_table.irreps:
            if np.allclose(irrep_at_q.k, conventional_q):
                irreps_at_q.append(irrep_at_q)
        return irreps_at_q

    def _get_mapping_to_irt(
        self,
        irt_little_rotations: NDArray[np.int_],
        found_little_rotations: NDArray[np.int_],
        prim_mat: NDArray[np.float64],
    ) -> list[int]:
        conventional_little_rotations = prim_mat @ found_little_rotations @ np.linalg.inv(prim_mat)
        mapping_to_irt = []
        for irt_rotation in irt_little_rotations:
            for index, rotation in enumerate(conventional_little_rotations):
                if np.allclose(irt_rotation, rotation):
                    mapping_to_irt.append(index)
                    break
        return mapping_to_irt

    def get_irrep_labels(
        self,
        qpoint: list[float],
        irreps,
        mapping_little_group: NDArray[np.int_],
    ) -> list[str]:
        generic_labels = [f"irrep_{index + 1}({irrep.shape[1]})" for index, irrep in enumerate(irreps)]
        try:
            irt_table = IrrepTable(self.spglib_dataset["number"], spinor=False)
        except Exception:
            return generic_labels

        prim_mat = get_primitive_matrix_by_centring(self.spglib_dataset["international"][0])
        irt_irreps = self._get_irt_irreps_at_q(qpoint, irt_table, prim_mat)
        if not irt_irreps:
            return generic_labels

        irt_little_rotations = np.array(
            [irt_table.symmetries[index - 1].R for index in irt_irreps[0].characters.keys()]
        )
        found_little_rotations = self.rotations[mapping_little_group]
        mapping_to_irt = self._get_mapping_to_irt(irt_little_rotations, found_little_rotations, prim_mat)
        if len(mapping_to_irt) != len(irt_little_rotations):
            return generic_labels

        resolved_labels: list[str] = []
        used_irt_labels: set[str] = set()
        for generic_label, irrep in zip(generic_labels, irreps):
            spgrep_character = np.array(get_character(irrep), dtype=complex)[mapping_to_irt]
            best_label = generic_label
            best_overlap = -1.0
            for irt_irrep in irt_irreps:
                irt_label = f"{irt_irrep.name}({irt_irrep.dim})"
                irt_character = np.array(list(irt_irrep.characters.values()), dtype=complex)
                overlap = np.abs(
                    np.dot(spgrep_character, np.conjugate(irt_character)) / irt_irrep.nsym
                )
                if overlap > best_overlap:
                    best_label = irt_label
                    best_overlap = float(overlap)
            if best_overlap < 0.9:
                best_label = generic_label
            elif best_label in used_irt_labels:
                best_label = f"{best_label} [{generic_label}]"
            used_irt_labels.add(best_label)
            resolved_labels.append(best_label)
        return resolved_labels

    def describe_mode_spaces(
        self,
        qpoint: list[float],
    ) -> tuple[object, list[NDArray[np.complex128]], list[str]]:
        irreps, vibration_rep, mapping_little_group = self.get_vibration_rep(qpoint)
        irrep_labels = self.get_irrep_labels(qpoint, irreps, mapping_little_group)
        basis_spaces, basis_space_labels = self.get_vibration_basis(irreps, vibration_rep, irrep_labels)
        return irreps, basis_spaces, basis_space_labels

    def get_supercell_size(self, qpoint: list[float]) -> tuple[int, int, int]:
        sizes = []
        for component in qpoint:
            if abs(component) < 1e-10:
                sizes.append(1)
            else:
                sizes.append(Fraction(float(component)).limit_denominator(6).denominator)
        return tuple(sizes)

    def get_supercell_displacements(
        self,
        qpoint: list[float],
        mode_vector: NDArray[np.complex128],
        supercell_size: tuple[int, int, int],
    ):
        primitive = self.primitive_cell
        n_atoms = len(primitive.scaled_positions)
        lattice = primitive.cell
        frac_pos = primitive.scaled_positions
        symbols_prim = get_chemical_symbols(primitive)
        mode = mode_vector.reshape((-1, 3))
        n1, n2, n3 = supercell_size

        all_positions = []
        all_displacements = []
        all_symbols = []

        for i1 in range(n1):
            for i2 in range(n2):
                for i3 in range(n3):
                    translation_frac = np.array([i1, i2, i3])
                    phase = np.exp(2j * np.pi * np.dot(qpoint, translation_frac))
                    for atom_index in range(n_atoms):
                        pos_frac = frac_pos[atom_index] + translation_frac
                        pos_cart = pos_frac @ lattice
                        all_positions.append(pos_cart)
                        displacement = np.real(mode[atom_index] * phase)
                        all_displacements.append(displacement)
                        all_symbols.append(symbols_prim[atom_index])

        supercell_lattice = lattice.copy()
        supercell_lattice[0] *= n1
        supercell_lattice[1] *= n2
        supercell_lattice[2] *= n3
        return (
            np.array(all_positions),
            np.array(all_displacements),
            all_symbols,
            supercell_lattice,
        )

    def write_displaced_structure(
        self,
        positions: NDArray[np.float64],
        displacements: NDArray[np.float64],
        symbols: list[str],
        supercell_lattice: NDArray[np.float64],
        amplitude: float,
        output_path: str,
    ) -> None:
        atoms = Atoms(
            symbols=symbols,
            positions=positions + amplitude * displacements,
            cell=supercell_lattice,
            pbc=True,
        )
        ase_write(output_path, atoms, format="vasp", direct=True)


def _print_high_symmetry_qpoints(qpoints: dict[str, list[float]]) -> None:
    print("Available high-symmetry q-points:")
    for label, coords in qpoints.items():
        print(f"  {label:8s} {coords}")


def _print_mode_spaces(basis_spaces: list[NDArray[np.complex128]], irrep_labels: list[str]) -> None:
    print("Irrep-grouped vibration spaces:")
    for mode_index, (space, irrep_label) in enumerate(zip(basis_spaces, irrep_labels)):
        dim = space.shape[0]
        print(
            f"  Mode Space {mode_index:2d}: irrep = {irrep_label}, dimension = {dim}, "
            f"component indices = 0..{dim - 1}"
        )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cell, _ = read_crystal_structure(args.poscar, interface_mode="vasp")
    vibrations = SymmetryOnlyVibrations(cell=cell, symprec=args.tolerance)

    qpoint_map = vibrations.get_high_symmetry_qpoints()
    _print_high_symmetry_qpoints(qpoint_map)
    if args.list_qpoints:
        return

    if not args.qpoint:
        raise ValueError("--qpoint is required unless --list-qpoints is used.")

    qpoint_label, qpoint = vibrations.resolve_qpoint(args.qpoint)
    print(f"\nSelected q-point: {qpoint_label} = {qpoint}")

    irreps, basis_spaces, irrep_labels = vibrations.describe_mode_spaces(qpoint)
    print(f"Number of irrep-grouped vibration spaces: {len(basis_spaces)}")
    _print_mode_spaces(basis_spaces, irrep_labels)

    if args.mode_index is None:
        print(
            "\nUse --mode-index and optionally --component-index to inspect a specific basis vector."
        )
        return

    if args.mode_index < 0 or args.mode_index >= len(basis_spaces):
        raise IndexError(
            f"Mode space index {args.mode_index} is out of range [0, {len(basis_spaces) - 1}]."
        )

    selected_space = basis_spaces[args.mode_index]
    if args.component_index < 0 or args.component_index >= selected_space.shape[0]:
        raise IndexError(
            f"Component index {args.component_index} is out of range [0, {selected_space.shape[0] - 1}]."
        )

    mode_vector = selected_space[args.component_index]
    supercell_size = vibrations.get_supercell_size(qpoint)
    print(f"\nSelected mode space: {args.mode_index}")
    print(f"Selected irrep     : {irrep_labels[args.mode_index]}")
    print(f"Selected component : {args.component_index}")
    print(f"Commensurate supercell size: {supercell_size}")

    positions, displacements, symbols, supercell_lattice = vibrations.get_supercell_displacements(
        qpoint=qpoint,
        mode_vector=mode_vector,
        supercell_size=supercell_size,
    )
    norms = np.linalg.norm(displacements, axis=1)
    print(f"Supercell atom count: {len(symbols)}")
    print(f"Displacement norm range (unit amplitude): min={norms.min():.6f}, max={norms.max():.6f}")
    print("First 5 displacement vectors:")
    for index in range(min(5, len(displacements))):
        print(
            f"  {index:2d} {symbols[index]:2s} "
            f"pos={np.round(positions[index], 6).tolist()} "
            f"disp={np.round(displacements[index], 6).tolist()}"
        )

    if args.export_npz:
        np.savez(
            args.export_npz,
            positions=positions,
            displacements=displacements,
            symbols=np.array(symbols, dtype=object),
            supercell_lattice=supercell_lattice,
            qpoint=np.array(qpoint, dtype=float),
            qpoint_label=np.array(qpoint_label, dtype=object),
            mode_index=np.array(args.mode_index),
            component_index=np.array(args.component_index),
            irrep_labels=np.array(irrep_labels, dtype=object),
            selected_irrep_label=np.array(irrep_labels[args.mode_index], dtype=object),
            mode_space_dimensions=np.array([space.shape[0] for space in basis_spaces], dtype=int),
            selected_mode_dimension=np.array(selected_space.shape[0], dtype=int),
            amplitude=np.array(args.amplitude, dtype=float),
        )
        print(f"Saved mode data to: {args.export_npz}")

    if args.output:
        vibrations.write_displaced_structure(
            positions=positions,
            displacements=displacements,
            symbols=symbols,
            supercell_lattice=supercell_lattice,
            amplitude=args.amplitude,
            output_path=args.output,
        )
        print(f"Saved displaced structure to: {args.output}")


if __name__ == "__main__":
    main()
