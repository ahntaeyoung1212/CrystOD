"""
Symmetry-adapted phonon modulation workflow for crystod.
"""

from __future__ import annotations

from dataclasses import dataclass
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, RawDescriptionHelpFormatter, RawTextHelpFormatter
from fractions import Fraction
from pathlib import Path
import re

import numpy as np
import phonopy
import spglib
from ase import Atoms
from ase.io import write as ase_write
from numpy.typing import NDArray

from .runtime_compat import (
    SymmetryDatasetAdapter,
    get_chemical_symbols,
    get_little_group,
    get_scaled_positions,
)
from .spglib_compat import ensure_spglib_compat

ensure_spglib_compat()

from phonopy.structure.atoms import PhonopyAtoms
from spgrep.core import get_spacegroup_irreps_from_primitive_symmetry
from spgrep.representation import project_to_irrep


class MyHelpFormatter(
    RawTextHelpFormatter,
    RawDescriptionHelpFormatter,
    ArgumentDefaultsHelpFormatter,
):
    pass


desc = """
Generate modulated crystal structures from symmetry-adapted phonon modes.

# Command Example:
crystod --modulation --yaml phonopy_params.yaml --qpoint 0.5 0.5 0.5 --mode 0 1 2 --amplitude 0.3
crystod --modulation --yaml phonopy_params.yaml --qpoint1 0 0.5 0.5 --mode1 0 --qpoint2 0.5 0 0.5 --mode2 0 --output POSCAR_combined
"""


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=desc, formatter_class=MyHelpFormatter)
    parser.add_argument(
        "--yaml",
        dest="yaml_path",
        default="phonopy_params.yaml",
        help="Path to phonopy_params.yaml(.xz).",
    )
    parser.add_argument(
        "--qpoint",
        nargs=3,
        type=float,
        help="Target q-point in primitive reciprocal coordinates.",
    )
    parser.add_argument(
        "--mode",
        nargs="+",
        type=int,
        help="Mode index or indices to apply after the mode table is shown.",
    )
    parser.add_argument(
        "--amplitude",
        nargs="+",
        type=float,
        default=[0.3],
        help="Modulation amplitude(s) in Angstroms.",
    )
    parser.add_argument(
        "--output",
        default="POSCAR_modulated",
        help="Output POSCAR path.",
    )
    parser.add_argument(
        "--tolerance",
        "--symprec",
        dest="symprec",
        type=float,
        default=1e-5,
        help="Symmetry tolerance.",
    )
    return parser


@dataclass
class ModulationTerm:
    qpoint: list[float]
    mode_indices: list[int]
    amplitudes: list[float]


@dataclass
class PreparedModulationTerm:
    modulation: SymmetryAdaptedModulation
    mode_indices: list[int]
    amplitudes: list[float]


class _CoreRepresentation:
    def __init__(self, cell: PhonopyAtoms, symprec: float = 1e-5):
        self.input_cell = cell
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


class _Vibrations(_CoreRepresentation):
    def __init__(self, cell: PhonopyAtoms, symprec: float = 1e-5):
        super().__init__(cell=cell, symprec=symprec)
        lattice_t = np.transpose(self.primitive_cell.cell)
        lattice_t_inv = np.linalg.inv(lattice_t)
        self.rotations_cartesian = np.array(
            [lattice_t @ rotation @ lattice_t_inv for rotation in self.rotations],
            dtype=np.complex128,
        )

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
        return irreps, vibration_rep

    def get_vibration_basis(self, irreps, vibration_rep) -> list[NDArray[np.complex128]]:
        basis_vectors: list[NDArray[np.complex128]] = []
        for irrep in irreps:
            basis_vectors.extend(project_to_irrep(vibration_rep, irrep))
        return basis_vectors


class SymmetryAdaptedModulation:
    def __init__(self, yaml_path: str, qpoint: list[float], symprec: float = 1e-5) -> None:
        self.qpoint = np.array(qpoint, dtype=float)
        self.symprec = symprec
        self.phonon = phonopy.load(yaml_path)
        dynamical_matrix = self.phonon.dynamical_matrix

        primitive = self.phonon.primitive
        primitive_atoms = PhonopyAtoms(
            numbers=primitive.numbers,
            scaled_positions=primitive.scaled_positions,
            cell=primitive.cell,
        )

        self.vibrations = _Vibrations(cell=primitive_atoms, symprec=symprec)
        irreps, vibration_rep = self.vibrations.get_vibration_rep(qpoint)
        vibration_basis = self.vibrations.get_vibration_basis(irreps, vibration_rep)
        self.irreps = irreps
        self.vibration_basis = vibration_basis

        dynamical_matrix.run(qpoint)
        raw_matrix = dynamical_matrix.dynamical_matrix.copy()
        primitive_positions = self.vibrations.primitive_cell.scaled_positions
        self.n_atoms = len(primitive_positions)

        phase = np.exp(2j * np.pi * np.dot(primitive_positions, qpoint))
        modified_matrix = np.zeros_like(raw_matrix)
        for i in range(self.n_atoms):
            for j in range(self.n_atoms):
                modified_matrix[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = (
                    raw_matrix[3 * i : 3 * i + 3, 3 * j : 3 * j + 3]
                    * np.conj(phase[i])
                    * phase[j]
                )

        unitary = np.vstack(vibration_basis)
        block_matrix = unitary @ modified_matrix @ unitary.conj().T

        self.mode_info: list[dict[str, float | int]] = []
        self.mode_vectors: list[NDArray[np.complex128]] = []
        cursor = 0
        for block_index, basis_block in enumerate(vibration_basis):
            block_dim = basis_block.shape[0]
            submatrix = block_matrix[cursor : cursor + block_dim, cursor : cursor + block_dim]
            block_eigvals, block_eigvecs = np.linalg.eigh(submatrix)
            block_eigvals = block_eigvals.real

            # Preserve the symmetry-adapted basis when a block is numerically
            # degenerate. Re-diagonalizing an exactly degenerate block can pick
            # an arbitrary rotated basis and lower the apparent symmetry of an
            # individual mode, even though the irrep subspace is unchanged.
            mean_eigval = float(np.mean(block_eigvals))
            if np.allclose(block_eigvals, mean_eigval, atol=1e-10, rtol=1e-8):
                block_eigvals = np.full(block_dim, mean_eigval, dtype=float)
                block_eigvecs = np.eye(block_dim, dtype=complex)

            for component_index in range(block_dim):
                eigval = float(block_eigvals[component_index])
                frequency = np.sign(eigval) * np.sqrt(np.abs(eigval)) * 15.633302

                mode_vector = np.zeros(3 * self.n_atoms, dtype=complex)
                for basis_index in range(block_dim):
                    mode_vector += block_eigvecs[basis_index, component_index] * basis_block[basis_index]

                self.mode_info.append(
                    {
                        "frequency_THz": frequency,
                        "irrep_block_index": block_index,
                        "component_index": component_index,
                        "degeneracy": block_dim,
                    }
                )
                self.mode_vectors.append(mode_vector)
            cursor += block_dim

        sort_indices = np.argsort([float(info["frequency_THz"]) for info in self.mode_info])
        self.mode_info = [self.mode_info[index] for index in sort_indices]
        self.mode_vectors = [self.mode_vectors[index] for index in sort_indices]

    @property
    def n_modes(self) -> int:
        return len(self.mode_info)

    def print_mode_info(self) -> None:
        print(f"Phonon modes at q = {self.qpoint}")
        print(f"{'Mode':>5s}  {'Freq (THz)':>12s}  {'Irrep Block':>12s}  {'Degeneracy':>11s}")
        print("-" * 50)
        for mode_index, info in enumerate(self.mode_info):
            print(
                f"{mode_index:5d}  {float(info['frequency_THz']):12.4f}  "
                f"{int(info['irrep_block_index']):12d}  {int(info['degeneracy']):11d}"
            )

    @staticmethod
    def get_commensurate_supercell_sizes(qpoint: list[float] | NDArray[np.float64]) -> NDArray[np.int_]:
        sizes = []
        for component in qpoint:
            if abs(component) < 1e-10:
                sizes.append(1)
            else:
                sizes.append(Fraction(float(component)).limit_denominator(12).denominator)
        return np.array(sizes, dtype=int)

    def _get_commensurate_supercell_matrix(self) -> NDArray[np.int_]:
        return np.diag(self.get_commensurate_supercell_sizes(self.qpoint)).astype(int)

    def get_modulated_structure(
        self,
        mode_indices: list[int],
        amplitudes: list[float],
    ) -> Atoms:
        for mode_index in mode_indices:
            if mode_index < 0 or mode_index >= self.n_modes:
                raise IndexError(f"Mode index {mode_index} is out of range [0, {self.n_modes - 1}].")

        supercell_matrix = self._get_commensurate_supercell_matrix()
        n1, n2, n3 = np.diag(supercell_matrix)

        primitive = self.vibrations.primitive_cell
        lattice = primitive.cell
        frac_pos = primitive.scaled_positions
        symbols_prim = get_chemical_symbols(primitive)

        positions = []
        displacements_total = []
        all_symbols = []

        for i1 in range(n1):
            for i2 in range(n2):
                for i3 in range(n3):
                    translation_frac = np.array([i1, i2, i3])
                    phase = np.exp(2j * np.pi * np.dot(self.qpoint, translation_frac))
                    for atom_index in range(self.n_atoms):
                        pos_frac = frac_pos[atom_index] + translation_frac
                        positions.append(pos_frac @ lattice)

                        displacement = np.zeros(3)
                        for mode_index, amplitude in zip(mode_indices, amplitudes):
                            mode_vector = self.mode_vectors[mode_index].reshape(self.n_atoms, 3)
                            displacement += np.real(mode_vector[atom_index] * phase) * amplitude

                        displacements_total.append(displacement)
                        all_symbols.append(symbols_prim[atom_index])

        positions = np.array(positions)
        displacements_total = np.array(displacements_total)
        modulated_positions = positions + displacements_total

        supercell_lattice = lattice.copy()
        supercell_lattice[0] *= n1
        supercell_lattice[1] *= n2
        supercell_lattice[2] *= n3

        unique_symbols = []
        for symbol in symbols_prim:
            if symbol not in unique_symbols:
                unique_symbols.append(symbol)

        sorted_indices = []
        for symbol in unique_symbols:
            for atom_index, atom_symbol in enumerate(all_symbols):
                if atom_symbol == symbol:
                    sorted_indices.append(atom_index)

        sorted_symbols = [all_symbols[index] for index in sorted_indices]
        sorted_positions = modulated_positions[sorted_indices]
        return Atoms(sorted_symbols, sorted_positions, cell=supercell_lattice, pbc=True)

    def write_modulated_poscar(
        self,
        filepath: str,
        mode_indices: list[int],
        amplitudes: list[float],
    ) -> Atoms:
        atoms = self.get_modulated_structure(mode_indices=mode_indices, amplitudes=amplitudes)
        ase_write(filepath, atoms, format="vasp", direct=True)
        print(f"Modulated structure written to: {filepath}")
        return atoms

    @staticmethod
    def analyze_symmetry(atoms: Atoms, symprec: float = 0.1) -> dict[str, str | int]:
        cell = (
            atoms.cell.array,
            atoms.get_scaled_positions(),
            atoms.numbers,
        )
        dataset = SymmetryDatasetAdapter(spglib.get_symmetry_dataset(cell, symprec=symprec))
        info = {
            "international": dataset.international,
            "number": dataset.number,
            "hall": dataset.hall,
        }
        print(f"Space group: {info['international']} (#{info['number']})")
        print(f"Hall symbol: {info['hall']}")
        return info


def _normalize_amplitudes(mode_indices: list[int], amplitudes: list[float]) -> list[float]:
    if len(amplitudes) == 1:
        return amplitudes * len(mode_indices)
    if len(amplitudes) != len(mode_indices):
        raise ValueError(
            f"Number of amplitudes ({len(amplitudes)}) must match number of modes ({len(mode_indices)}) "
            "or be a single value."
        )
    return amplitudes


def _parse_numbered_modulation_terms(extra_argv: list[str]) -> list[ModulationTerm]:
    if not extra_argv:
        return []

    grouped: dict[int, dict[str, list[str]]] = {}
    index = 0
    pattern = re.compile(r"^--(qpoint|mode|amplitude)(\d+)$")

    while index < len(extra_argv):
        token = extra_argv[index]
        match = pattern.fullmatch(token)
        if not match:
            raise ValueError(f"Unrecognized modulation argument: {token}")

        key, suffix = match.group(1), int(match.group(2))
        index += 1
        values: list[str] = []
        while index < len(extra_argv) and not extra_argv[index].startswith("--"):
            values.append(extra_argv[index])
            index += 1

        if not values:
            raise ValueError(f"{token} requires value(s).")
        grouped.setdefault(suffix, {})[key] = values

    terms: list[ModulationTerm] = []
    for suffix in sorted(grouped):
        entry = grouped[suffix]
        if "qpoint" not in entry:
            raise ValueError(f"--qpoint{suffix} is required when using numbered modulation arguments.")
        if "mode" not in entry:
            raise ValueError(f"--mode{suffix} is required when using numbered modulation arguments.")
        if len(entry["qpoint"]) != 3:
            raise ValueError(f"--qpoint{suffix} requires exactly three coordinates.")

        qpoint = [float(value) for value in entry["qpoint"]]
        mode_indices = [int(value) for value in entry["mode"]]
        raw_amplitudes = [float(value) for value in entry.get("amplitude", ["0.3"])]
        amplitudes = _normalize_amplitudes(mode_indices, raw_amplitudes)
        terms.append(
            ModulationTerm(
                qpoint=qpoint,
                mode_indices=mode_indices,
                amplitudes=amplitudes,
            )
        )

    return terms


def _build_combined_modulated_structure(terms: list[PreparedModulationTerm]) -> Atoms:
    if not terms:
        raise ValueError("At least one modulation term is required.")

    reference = terms[0].modulation.vibrations.primitive_cell
    lattice = reference.cell
    frac_pos = reference.scaled_positions
    symbols_prim = get_chemical_symbols(reference)
    n_atoms = len(frac_pos)

    for term in terms[1:]:
        primitive = term.modulation.vibrations.primitive_cell
        if not np.allclose(primitive.cell, lattice):
            raise ValueError("All modulation terms must share the same primitive lattice.")
        if not np.allclose(primitive.scaled_positions, frac_pos):
            raise ValueError("All modulation terms must share the same primitive positions.")
        if get_chemical_symbols(primitive) != symbols_prim:
            raise ValueError("All modulation terms must share the same primitive species ordering.")

    supercell_sizes = np.ones(3, dtype=int)
    for term in terms:
        supercell_sizes = np.lcm(
            supercell_sizes,
            SymmetryAdaptedModulation.get_commensurate_supercell_sizes(term.modulation.qpoint),
        )
    n1, n2, n3 = supercell_sizes

    prepared_terms: list[tuple[np.ndarray, list[np.ndarray], list[float]]] = []
    for term in terms:
        reshaped_vectors = [term.modulation.mode_vectors[index].reshape(n_atoms, 3) for index in term.mode_indices]
        prepared_terms.append((term.modulation.qpoint, reshaped_vectors, term.amplitudes))

    positions = []
    displacements_total = []
    all_symbols = []

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                translation_frac = np.array([i1, i2, i3], dtype=float)
                for atom_index in range(n_atoms):
                    pos_frac = frac_pos[atom_index] + translation_frac
                    positions.append(pos_frac @ lattice)

                    displacement = np.zeros(3)
                    for qpoint, mode_vectors, amplitudes in prepared_terms:
                        phase = np.exp(2j * np.pi * np.dot(qpoint, translation_frac))
                        for mode_vector, amplitude in zip(mode_vectors, amplitudes):
                            displacement += np.real(mode_vector[atom_index] * phase) * amplitude

                    displacements_total.append(displacement)
                    all_symbols.append(symbols_prim[atom_index])

    positions = np.array(positions)
    displacements_total = np.array(displacements_total)
    modulated_positions = positions + displacements_total

    supercell_lattice = lattice.copy()
    supercell_lattice[0] *= n1
    supercell_lattice[1] *= n2
    supercell_lattice[2] *= n3

    unique_symbols: list[str] = []
    for symbol in symbols_prim:
        if symbol not in unique_symbols:
            unique_symbols.append(symbol)

    sorted_indices = []
    for symbol in unique_symbols:
        for atom_index, atom_symbol in enumerate(all_symbols):
            if atom_symbol == symbol:
                sorted_indices.append(atom_index)

    sorted_symbols = [all_symbols[index] for index in sorted_indices]
    sorted_positions = modulated_positions[sorted_indices]
    return Atoms(sorted_symbols, sorted_positions, cell=supercell_lattice, pbc=True)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args, extra_argv = parser.parse_known_args(argv)
    yaml_path = Path(args.yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"File '{yaml_path}' does not exist.")

    try:
        numbered_terms = _parse_numbered_modulation_terms(extra_argv)
    except ValueError as exc:
        parser.error(str(exc))

    if numbered_terms:
        if args.qpoint is not None or args.mode is not None or args.amplitude != [0.3]:
            parser.error(
                "Use either --qpoint/--mode/--amplitude or numbered sets such as "
                "--qpoint1/--mode1/--amplitude1, but not both."
            )
        terms = numbered_terms
    else:
        if args.qpoint is None:
            parser.error("--modulation requires --qpoint, or numbered arguments such as --qpoint1.")
        if args.mode is None:
            parser.error("--modulation requires --mode, or numbered arguments such as --mode1.")
        terms = [
            ModulationTerm(
                qpoint=args.qpoint,
                mode_indices=args.mode,
                amplitudes=_normalize_amplitudes(args.mode, args.amplitude),
            )
        ]

    modulation_cache: dict[tuple[float, float, float], SymmetryAdaptedModulation] = {}
    prepared_terms: list[PreparedModulationTerm] = []

    for term_index, term in enumerate(terms, start=1):
        qpoint_key = tuple(float(value) for value in term.qpoint)
        modulation = modulation_cache.get(qpoint_key)
        if modulation is None:
            print(f"Loading '{yaml_path}' at q = {term.qpoint}...")
            modulation = SymmetryAdaptedModulation(
                yaml_path=str(yaml_path),
                qpoint=term.qpoint,
                symprec=args.symprec,
            )
            print()
            modulation.print_mode_info()
            modulation_cache[qpoint_key] = modulation
            if term_index != len(terms):
                print()

        for mode_index in term.mode_indices:
            if mode_index < 0 or mode_index >= modulation.n_modes:
                raise IndexError(f"Mode index {mode_index} is out of range [0, {modulation.n_modes - 1}].")

        prepared_terms.append(
            PreparedModulationTerm(
                modulation=modulation,
                mode_indices=term.mode_indices,
                amplitudes=term.amplitudes,
            )
        )

    print("\nGenerating modulated structure...")
    if len(prepared_terms) == 1:
        print(f"  q-point: {prepared_terms[0].modulation.qpoint.tolist()}")
        print(f"  Modes: {prepared_terms[0].mode_indices}")
        print(f"  Amplitudes (A): {prepared_terms[0].amplitudes}")
    else:
        for term_index, term in enumerate(prepared_terms, start=1):
            print(
                f"  Term {term_index}: q = {term.modulation.qpoint.tolist()}, "
                f"modes = {term.mode_indices}, amplitudes (A) = {term.amplitudes}"
            )

    if len(prepared_terms) == 1:
        term = prepared_terms[0]
        atoms = term.modulation.write_modulated_poscar(
            filepath=args.output,
            mode_indices=term.mode_indices,
            amplitudes=term.amplitudes,
        )
    else:
        atoms = _build_combined_modulated_structure(prepared_terms)
        ase_write(args.output, atoms, format="vasp", direct=True)
        print(f"Modulated structure written to: {args.output}")

    print("\nSymmetry of the generated structure:")
    SymmetryAdaptedModulation.analyze_symmetry(atoms, symprec=0.1)


if __name__ == "__main__":
    main()
