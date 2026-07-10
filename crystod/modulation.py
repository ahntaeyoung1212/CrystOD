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

from .operations import parse_qpoint_token
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
crystod-phonon --modulation --yaml phonopy_params.yaml --qpoint 0.5 0.5 0.5                      (list modes and star of q only)
crystod-phonon --modulation --yaml phonopy_params.yaml --qpoint 0.5 0.5 0.5 --mode 1 2 3 --amplitude 0.3
crystod-phonon --modulation --yaml phonopy_params.yaml --qpoint1 0 0.5 0.5 --mode1 1 --qpoint2 0.5 0 0.5 --mode2 1 --output POSCAR_combined
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
        type=parse_qpoint_token,
        help="Target q-point in primitive reciprocal coordinates "
        "(fractions such as 1/3 are allowed).",
    )
    parser.add_argument(
        "--mode",
        nargs="+",
        type=int,
        help="Mode number(s) to apply, 1-based as in the printed mode table.\n"
        "If omitted, only the mode table and the star of q are printed.",
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
        default=None,
        help="Output POSCAR path (default: MPOSCAR_{q}_{mode}_{irrep}_{subgroup}).",
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


def _find_intertwiner(
    rep: NDArray[np.complex128],
    space_s: NDArray[np.complex128],
    space_r: NDArray[np.complex128],
) -> NDArray[np.complex128] | None:
    """Unitary aligning the irrep basis of space_s to that of space_r.

    Both spaces must carry equivalent irreps of the same (projective)
    representation ``rep``; returns None when they are inequivalent.
    """
    dim = space_s.shape[0]
    d_s = np.array([space_s @ G @ space_s.conj().T for G in rep])
    d_r = np.array([space_r @ G @ space_r.conj().T for G in rep])
    for seed_index in range(dim * dim):
        seed = np.zeros((dim, dim))
        seed[seed_index // dim, seed_index % dim] = 1.0
        averaged = sum(a @ seed @ b.conj().T for a, b in zip(d_s, d_r)) / len(rep)
        if np.linalg.norm(averaged) > 1e-6:
            u, _, vh = np.linalg.svd(averaged)
            return u @ vh
    return None


def _irrep_filename_tag(labels: list[str]) -> str:
    """Compact irrep tag for file names: drop the dimension suffix "(n)" and
    join distinct labels with '+' (e.g. "GM1(1), GM5(2)" -> "GM1+GM5")."""
    cleaned: list[str] = []
    for label in labels:
        for part in label.split(","):
            part = re.sub(r"\(\d+\)", "", part).strip()
            if part and part != "-" and part not in cleaned:
                cleaned.append(part)
    return "+".join(cleaned)


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

        spaces = vibration_basis
        dims = [space.shape[0] for space in spaces]
        if sum(dims) != modified_matrix.shape[0]:
            raise RuntimeError("Irrep projection does not span the full vibration space.")
        offsets = np.cumsum([0] + dims)
        n_spaces = len(spaces)
        stacked = np.vstack(spaces)
        block_matrix = stacked @ modified_matrix @ stacked.conj().T

        # Spaces carrying equivalent irreps couple through the dynamical
        # matrix; diagonalizing each projected block on its own would drop
        # that coupling and give wrong frequencies whenever an irrep occurs
        # more than once at q. Group coupled spaces into clusters and
        # diagonalize per cluster (same construction as crystod-phonon --vector).
        coupled = np.zeros((n_spaces, n_spaces), dtype=bool)
        for s in range(n_spaces):
            for t in range(n_spaces):
                sub = block_matrix[offsets[s] : offsets[s + 1], offsets[t] : offsets[t + 1]]
                coupled[s, t] = bool(np.abs(sub).max() > 1e-6)
        clusters: list[list[int]] = []
        seen: set[int] = set()
        for s in range(n_spaces):
            if s in seen:
                continue
            stack, cluster = [s], []
            while stack:
                u = stack.pop()
                if u in seen:
                    continue
                seen.add(u)
                cluster.append(u)
                stack.extend(v for v in range(n_spaces) if coupled[u, v] and v not in seen)
            clusters.append(sorted(cluster))

        self.mode_info: list[dict[str, float | int]] = []
        self.mode_vectors: list[NDArray[np.complex128]] = []
        for cluster in clusters:
            dim = dims[cluster[0]]
            if any(dims[index] != dim for index in cluster):
                raise RuntimeError("Coupled irrep spaces with different dimensions.")
            multiplicity = len(cluster)

            aligned = [spaces[cluster[0]]]
            for index in cluster[1:]:
                intertwiner = _find_intertwiner(vibration_rep, spaces[index], spaces[cluster[0]])
                if intertwiner is None:
                    raise RuntimeError("Coupled irrep spaces are not equivalent.")
                aligned.append(intertwiner.conj().T @ spaces[index])

            # After alignment every coupling block is a scalar multiple of the
            # identity (Schur), so the cluster reduces to one multiplicity-sized
            # Hermitian matrix shared by all irrep components.
            coupling = np.zeros((multiplicity, multiplicity), dtype=complex)
            for a in range(multiplicity):
                for b in range(multiplicity):
                    sub = aligned[a] @ modified_matrix @ aligned[b].conj().T
                    if np.abs(sub - np.eye(dim) * np.trace(sub) / dim).max() > 1e-6:
                        raise RuntimeError("Coupling between irrep spaces is not scalar.")
                    coupling[a, b] = np.trace(sub) / dim
            eigenvalues, eigenvectors = np.linalg.eigh(coupling)
            eigenvalues = eigenvalues.real

            # Preserve the symmetry-adapted basis when the cluster is
            # numerically degenerate. Re-diagonalizing an exactly degenerate
            # cluster can pick an arbitrary rotated basis and lower the
            # apparent symmetry of an individual mode.
            if multiplicity > 1 and np.allclose(eigenvalues, eigenvalues.mean(), atol=1e-10, rtol=1e-8):
                eigenvalues = np.full(multiplicity, eigenvalues.mean())
                eigenvectors = np.eye(multiplicity, dtype=complex)

            for w in range(multiplicity):
                eigval = float(eigenvalues[w])
                frequency = np.sign(eigval) * np.sqrt(np.abs(eigval)) * 15.633302
                for component in range(dim):
                    # Rows of the projected basis are bras; keep this module's
                    # convention that the displacement is Re(vector * e^{2 pi i q.R}),
                    # so the bra-space combination uses conjugated coefficients.
                    mode_vector = np.zeros(3 * self.n_atoms, dtype=complex)
                    for a in range(multiplicity):
                        mode_vector += np.conj(eigenvectors[a, w]) * aligned[a][component]
                    self.mode_info.append(
                        {
                            "frequency_THz": frequency,
                            "degeneracy": dim,
                        }
                    )
                    self.mode_vectors.append(mode_vector)

        sort_indices = np.argsort([float(info["frequency_THz"]) for info in self.mode_info], kind="stable")
        self.mode_info = [self.mode_info[index] for index in sort_indices]
        self.mode_vectors = [self.mode_vectors[index] for index in sort_indices]
        self._mode_labels: list[str] | None = None
        self._q_label: str | None = None

        # Verify against the plain phonopy spectrum before trusting the result.
        reference = np.sort(np.linalg.eigvalsh(modified_matrix).real)
        reference = np.sign(reference) * np.sqrt(np.abs(reference)) * 15.633302
        frequencies = [float(info["frequency_THz"]) for info in self.mode_info]
        if not np.allclose(frequencies, reference, atol=1e-3):
            raise RuntimeError("Symmetry-adapted frequencies do not match the phonopy spectrum.")
        for info, mode_vector in zip(self.mode_info, self.mode_vectors):
            eigenvalue = np.sign(info["frequency_THz"]) * (info["frequency_THz"] / 15.633302) ** 2
            ket = np.conj(mode_vector)
            if np.linalg.norm(modified_matrix @ ket - eigenvalue * ket) > 1e-6:
                raise RuntimeError("A symmetry-adapted mode is not an eigenvector of the dynamical matrix.")

    @property
    def n_modes(self) -> int:
        return len(self.mode_info)

    def get_mode_labels(self) -> list[str]:
        """Per-mode irrep labels (e.g. 'X3-(1)'); '-' when labeling is unavailable.

        Uses the irreptables-based labeling of crystod-phonon --vector/--irreps.
        The label of band i applies to mode i because the symmetry-adapted
        frequencies are verified to match the plain phonopy spectrum.
        """
        if self._mode_labels is None:
            labels = ["-"] * self.n_modes
            try:
                from .phonon_vector import _get_mode_labels
                from .runtime_compat import get_symmetry_dataset

                dataset = get_symmetry_dataset(self.phonon.symmetry)
                _, labels = _get_mode_labels(
                    [float(value) for value in self.qpoint],
                    self.phonon,
                    dataset,
                    degeneracy_tolerance=1e-3,
                )
            except Exception:
                pass
            self._mode_labels = labels
        return self._mode_labels

    def get_q_label(self) -> str:
        """Short q label for file names: the CDML name (e.g. 'X') when q lies
        in the star of a tabulated special point, else 'q_<coordinates>'."""
        if self._q_label is None:
            label = "q" + "".join(f"_{value:g}" for value in self.qpoint).replace("/", "o")
            try:
                from phonopy.structure.cells import get_primitive_matrix_by_centring

                from .irreptables_compat import load_irreptables
                from .phonon_irreps import find_star_representative, get_irt_special_points
                from .runtime_compat import get_symmetry_dataset

                irrep_table_cls, _ = load_irreptables()
                dataset = get_symmetry_dataset(self.phonon.primitive_symmetry)
                irt_table = irrep_table_cls(dataset["number"], spinor=False)
                prim_mat = get_primitive_matrix_by_centring(dataset["international"][0])
                q_names, q_list = get_irt_special_points(irt_table, prim_mat)
                representative = find_star_representative(
                    self.qpoint, dataset["rotations"], q_names, q_list
                )
                if representative is not None:
                    label = representative[0]
            except Exception:
                pass
            self._q_label = label
        return self._q_label

    def print_mode_info(self) -> None:
        labels = self.get_mode_labels()
        print(f"Phonon modes at q = {self.qpoint}")
        print(f"{'Mode':>5s}  {'Freq (THz)':>12s}  {'Irrep':>12s}  {'Degeneracy':>11s}")
        print("-" * 50)
        for mode_index, info in enumerate(self.mode_info):
            print(
                f"{mode_index + 1:5d}  {float(info['frequency_THz']):12.4f}  "
                f"{labels[mode_index]:>12s}  {int(info['degeneracy']):11d}"
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
                raise SystemExit(
                    f"ERROR: mode number {mode_index + 1} is out of range "
                    f"[1, {self.n_modes}] (numbering is 1-based)."
                )

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

        qpoint = [parse_qpoint_token(value) for value in entry["qpoint"]]
        mode_indices = [int(value) - 1 for value in entry["mode"]]
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


def _default_output_name(prepared_terms: list[PreparedModulationTerm], spacegroup: str) -> str:
    """Auto output name: MPOSCAR_{q}_{mode}_{irrep}_{subgroup}, one
    {q}_{mode}_{irrep} group per modulation term."""
    parts = ["MPOSCAR"]
    for term in prepared_terms:
        labels = term.modulation.get_mode_labels()
        unique_labels: list[str] = []
        for index in term.mode_indices:
            if labels[index] not in unique_labels:
                unique_labels.append(labels[index])
        parts.append(term.modulation.get_q_label())
        parts.append("mode" + "+".join(str(index + 1) for index in term.mode_indices))
        irrep_tag = _irrep_filename_tag(unique_labels)
        if irrep_tag:
            parts.append(irrep_tag)
    parts.append(spacegroup.replace("/", "").replace(" ", ""))
    return "_".join(parts)


def _load_modulation_with_report(
    yaml_path: Path,
    qpoint: list[float],
    symprec: float,
) -> SymmetryAdaptedModulation:
    """Load the modulation at q and print the mode table and the star of q."""
    print(f"Loading '{yaml_path}' at q = {qpoint}...")
    modulation = SymmetryAdaptedModulation(
        yaml_path=str(yaml_path),
        qpoint=qpoint,
        symprec=symprec,
    )
    print()
    modulation.print_mode_info()

    from .star_of_k import print_star_of_k

    print("\nStar of q (arms related by the space-group rotations):")
    print_star_of_k(
        rotations=modulation.vibrations.rotations,
        translations=modulation.vibrations.translations,
        kpoint=[float(value) for value in qpoint],
        indent="  ",
    )
    return modulation


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
            # Preview: show the mode table and the star of q so that a mode
            # can be chosen, without generating a modulated structure.
            _load_modulation_with_report(yaml_path, args.qpoint, args.symprec)
            print(
                "\nNo --mode given. Choose mode number(s) from the table above and rerun with"
                "\n--mode (and optionally --amplitude) to generate a modulated structure."
            )
            return
        mode_indices = [value - 1 for value in args.mode]
        terms = [
            ModulationTerm(
                qpoint=args.qpoint,
                mode_indices=mode_indices,
                amplitudes=_normalize_amplitudes(mode_indices, args.amplitude),
            )
        ]

    modulation_cache: dict[tuple[float, float, float], SymmetryAdaptedModulation] = {}
    prepared_terms: list[PreparedModulationTerm] = []

    for term_index, term in enumerate(terms, start=1):
        qpoint_key = tuple(float(value) for value in term.qpoint)
        modulation = modulation_cache.get(qpoint_key)
        if modulation is None:
            modulation = _load_modulation_with_report(yaml_path, term.qpoint, args.symprec)
            modulation_cache[qpoint_key] = modulation
            if term_index != len(terms):
                print()

        for mode_index in term.mode_indices:
            if mode_index < 0 or mode_index >= modulation.n_modes:
                raise SystemExit(
                    f"ERROR: mode number {mode_index + 1} is out of range "
                    f"[1, {modulation.n_modes}] (numbering is 1-based)."
                )

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
        print(f"  Modes: {[index + 1 for index in prepared_terms[0].mode_indices]}")
        print(f"  Amplitudes (A): {prepared_terms[0].amplitudes}")
    else:
        for term_index, term in enumerate(prepared_terms, start=1):
            print(
                f"  Term {term_index}: q = {term.modulation.qpoint.tolist()}, "
                f"modes = {[index + 1 for index in term.mode_indices]}, "
                f"amplitudes (A) = {term.amplitudes}"
            )

    if len(prepared_terms) == 1:
        term = prepared_terms[0]
        atoms = term.modulation.get_modulated_structure(
            mode_indices=term.mode_indices,
            amplitudes=term.amplitudes,
        )
    else:
        atoms = _build_combined_modulated_structure(prepared_terms)

    print("\nSymmetry of the generated structure:")
    symmetry = SymmetryAdaptedModulation.analyze_symmetry(atoms, symprec=0.1)

    output_path = args.output or _default_output_name(prepared_terms, str(symmetry["international"]))
    ase_write(output_path, atoms, format="vasp", direct=True)
    print(f"\nModulated structure written to: {output_path}")


if __name__ == "__main__":
    main()
