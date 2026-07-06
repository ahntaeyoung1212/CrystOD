"""
__author__ = "Hiroki Koiso, Yasuhide Mochizuki"
__copyright__ = "Copyright 2026, Mochizuki group"
__version__ = "3.1"
__maintainer__ = "Yasuhide Mochizuki"
__email__ = "mochizuki@rs.tus.ac.jp"
__status__ = "Development"
__released_date__ = "May 20, 2024"
__last_update__= "June 29, 2026"
"""

from __future__ import annotations

from .spglib_compat import ensure_spglib_compat

ensure_spglib_compat()

from phonopy.structure.atoms import PhonopyAtoms
from phonopy.interface.calculator import read_crystal_structure
from phonopy.structure.symmetry import Symmetry
from spglib import get_spacegroup_type_from_symmetry, standardize_cell
from .irreptables_compat import load_irreptables
from .runtime_compat import (
    get_character,
    get_chemical_symbols,
    get_little_group,
    get_scaled_positions,
    get_symmetry_dataset,
)
from .operations import characterize_rotation, get_seitz_symbol
from argparse import ArgumentParser, RawTextHelpFormatter, RawDescriptionHelpFormatter, ArgumentDefaultsHelpFormatter
import numpy as np
from numpy.typing import NDArray
from typing import Optional
import itertools
import re
from fractions import Fraction

IrrepTable, Irrep = load_irreptables()

### PARSER STRUCTURE ###
class MyHelpFormatter(RawTextHelpFormatter, RawDescriptionHelpFormatter, ArgumentDefaultsHelpFormatter):
    pass
desc = """
This program calculates the irreducible representations of elemental's crystal orbitals.

# Command Example:
python3 crystal_orbital.py --poscar POSCAR_ScF3_Pm-3m --element F --orbital p
"""
### --------------- ###


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=desc, formatter_class=MyHelpFormatter)
    parser.add_argument("--poscar", "-poscar", dest="poscar", type=str, default="POSCAR",
                        help="POSCAR.")
    parser.add_argument("--element", "-element", dest="element", required=True, type=str, default=None,
                        help="Element.")
    parser.add_argument("--orbital", "-orbital", dest="orbital", required=True, type=str, default=None,
                        help="Orbital. You can choose from following; s, p, d, f, g, h, i.")
    parser.add_argument("--kpoint", "-kpoint", dest="kpoint", type=float, nargs=3, required=False,
                        help="k-point, kx, ky, kz.")
    parser.add_argument("--spinor", "-spinor", dest="spinor", action="store_true",
                        help="Calculate double space group representations.")
    parser.add_argument("--tolerance", "-tolerance", dest="tolerance", type=float, default=0.00001,
                        help="Symmetry tolerance to search primitive cell.")
    parser.add_argument("--show-irrep-table", "-show-table", dest="table", action="store_true",
                        help="Show table of irreps.")
    return parser


def similarity_transformation(rot: NDArray[np.float_], mat: NDArray[np.float_]) -> NDArray[np.float_]:
    """Similarity transformation by R x M x R^-1."""
    return rot @ mat @ np.linalg.inv(rot)


def _canonicalize_component(value: float, tol: float = 1e-6, max_denominator: int = 48) -> float:
    """Snap a float to a nearby simple rational when possible."""
    nearest_integer = round(value)
    if abs(value - nearest_integer) < tol:
        return float(nearest_integer)

    fraction = Fraction(value).limit_denominator(max_denominator)
    snapped = float(fraction)
    if abs(value - snapped) < tol:
        return snapped

    return float(value)


def canonicalize_kpoint(kpoint: list[float], tol: float = 1e-6, max_denominator: int = 48) -> list[float]:
    """Normalize k-point coordinates to stable simple fractions."""
    normalized = [_canonicalize_component(float(value), tol=tol, max_denominator=max_denominator) for value in kpoint]
    return [0.0 if abs(value) < tol else value for value in normalized]


def format_kpoint(kpoint: list[float], decimals: int = 2) -> list[float]:
    """Format a k-point for display using plain Python floats."""
    return [float(np.round(float(value), decimals)) for value in kpoint]


def irrep_label_sort_key(label: str) -> tuple:
    """Natural ordering for physical irrep labels."""
    pattern = re.compile(
        r"^(?P<spinor>-?)(?P<kpoint>[A-Z]+)(?P<index>\d+)(?P<parity>[+-]?)(?:\((?P<dim>\d+)\))$"
    )
    match = pattern.match(label)
    if not match:
        return (label, 10**9, 10**9, 10**9, label)

    spinor = 0 if match.group("spinor") == "-" else 1
    kpoint = match.group("kpoint")
    index = int(match.group("index"))
    parity = match.group("parity")
    dim = int(match.group("dim"))
    parity_order = {"+": 0, "-": 1, "": 2}.get(parity, 3)
    return (kpoint, spinor, index, parity_order, dim, label)


def get_label_overrides(
    space_group_number: int,
    kpoint_name: Optional[str],
    spinor: bool,
) -> dict[str, str]:
    """Return manual label overrides for known Bilbao/irreptables naming mismatches."""
    return {}

class CrystalOrbital:
    def __init__(
            self, 
            cell: PhonopyAtoms,
            symprec: float = 0.00001,
            spior: bool = False,
            ):
        self._inputed_cell = cell
        # Convert inputed cell to primitive cell.
        (primitive_lattice, primitive_pos, primitive_numbers) = standardize_cell(cell.totuple(), 
                                                                                 to_primitive=True, 
                                                                                 symprec=symprec)
        self.primitive_cell = PhonopyAtoms(numbers=primitive_numbers, 
                                           scaled_positions=primitive_pos, 
                                           cell=primitive_lattice)
        print("\n ### Inputed cell was converted into primitive cell. ###")
        self.spinor = spior
        self.symprec = symprec

        # Get symmetry informations
        symmetry = Symmetry(self.primitive_cell)
        dataset = get_symmetry_dataset(symmetry)
        self.spglib_dataset = dataset
        self._spglib_rotations = dataset['rotations']
        self._spglib_translations = dataset['translations']
        self.transformation_matrix = dataset['transformation_matrix'] # primitive matrix
        self.irt_character_table = IrrepTable(dataset['number'], self.spinor)

        self.rotations, self.translations = self._sort_symmetry_operations_in_order_of_irt(
            self._spglib_rotations,
            self._spglib_translations,
        )
        self.seitz_symbols = [get_seitz_symbol(r, self.transformation_matrix) for r in self.rotations]

    def _sort_symmetry_operations_in_order_of_irt(
        self,
        spglib_R: NDArray[np.int_],
        spglib_t: NDArray[np.float_],
    ) -> tuple[NDArray[np.int_], NDArray[np.float_]]:
        """Sort symmetry operations found by spglib in the irreptables order."""
        irt_conv_R = np.array([sym.R for sym in self.irt_character_table.symmetries], dtype=float)
        irt_prim_R = similarity_transformation(np.linalg.inv(self.transformation_matrix), irt_conv_R)

        sorted_R = []
        sorted_t = []
        used_indices = set()
        for irt_R in np.rint(irt_prim_R).astype(int):
            found = False
            for i, R in enumerate(spglib_R):
                if i in used_indices:
                    continue
                if (irt_R == R).all():
                    sorted_R.append(R)
                    sorted_t.append(spglib_t[i])
                    used_indices.add(i)
                    found = True
                    break
            if not found:
                raise ValueError("Failed to sort symmetry operations in irreptables order.")
        return np.array(sorted_R), np.array(sorted_t)

    def get_irt_irreps_at_k(self, k: list[float]) -> list[Irrep]:
        """Get irreps at the k point from irreptables."""
        k = canonicalize_kpoint(k)
        trans_inv = np.linalg.inv(self.transformation_matrix)
        conventional_k = np.array(k) @ trans_inv
        irreps_at_k = []
        for irrep_at_k in self.irt_character_table.irreps:
            if np.allclose(irrep_at_k.k, conventional_k, atol=1e-6):
                irreps_at_k.append(irrep_at_k)
        return irreps_at_k

    def get_kpoint_name(self, k: list[float]) -> Optional[str]:
        """Get the special k-point name from irreptables if available."""
        k = canonicalize_kpoint(k)
        irreps_at_k = self.get_irt_irreps_at_k(k)
        if irreps_at_k:
            return irreps_at_k[0].kpname
        return None

    def get_irt_special_points(self) -> tuple[list[str], list[list[float]]]:
        """Get unique special k-points from irreptables in primitive basis."""
        primitive_kpoints = []
        kpoint_names = []
        for irrep in self.irt_character_table.irreps:
            primitive_k = canonicalize_kpoint(list(np.array(irrep.k) @ self.transformation_matrix))
            if primitive_k not in primitive_kpoints:
                primitive_kpoints.append(primitive_k)
                kpoint_names.append(irrep.kpname)
        return kpoint_names, primitive_kpoints

    def get_irrep_labels(
        self,
        k: list[float],
        irreps,
        mapping_little_group: NDArray[np.int_],
    ) -> dict[str, str]:
        """Map spgrep irreps to irreptables labels by comparing characters."""
        k = canonicalize_kpoint(k)
        irt_irreps = self.get_irt_irreps_at_k(k)
        if not irt_irreps:
            return {
                f"irrep_{i+1}({irrep.shape[1]})": f"irrep_{i+1}({irrep.shape[1]})"
                for i, irrep in enumerate(irreps)
            }

        def match_direct(
            lhs: NDArray[np.complex128],
            rhs: NDArray[np.complex128],
            atol: float = 1e-5,
        ) -> Optional[NDArray[np.complex128]]:
            if np.allclose(lhs, rhs, atol=atol):
                return rhs
            return None

        def match_with_possible_conjugation(
            lhs: NDArray[np.complex128],
            rhs: NDArray[np.complex128],
            atol: float = 1e-5,
        ) -> Optional[NDArray[np.complex128]]:
            direct = match_direct(lhs, rhs, atol=atol)
            if direct is not None:
                return direct
            rhs_conj = np.conj(rhs)
            if np.allclose(lhs, rhs_conj, atol=atol):
                return rhs_conj
            return None

        def resolve_spinor_phase_convention_block(
            group_generic: list[str],
            group_physical: list[str],
            atol: float = 1e-5,
        ) -> Optional[dict[str, tuple[str, NDArray[np.complex128]]]]:
            """Resolve 1D spinor irreps allowing an operation-wise phase convention shift."""
            if not group_generic or len(group_generic) != len(group_physical):
                return None

            def search_assignment(
                generic_candidates: dict[str, list[tuple[str, NDArray[np.complex128]]]],
            ) -> Optional[dict[str, tuple[str, NDArray[np.complex128]]]]:
                ordered = sorted(generic_candidates, key=lambda label: len(generic_candidates[label]))
                used_physical = set()
                assignment: dict[str, tuple[str, NDArray[np.complex128]]] = {}

                def backtrack(index: int) -> bool:
                    if index == len(ordered):
                        return True
                    generic_label = ordered[index]
                    candidates = sorted(generic_candidates[generic_label], key=lambda item: item[0])
                    for physical_label, transformed_chars in candidates:
                        if physical_label in used_physical:
                            continue
                        used_physical.add(physical_label)
                        assignment[generic_label] = (physical_label, transformed_chars)
                        if backtrack(index + 1):
                            return True
                        used_physical.remove(physical_label)
                        del assignment[generic_label]
                    return False

                if backtrack(0):
                    return assignment
                return None

            for anchor_generic in group_generic:
                anchor_spgrep = spgrep_character_map[anchor_generic]
                for anchor_physical in group_physical:
                    for anchor_use_conjugate in [0, 1]:
                        anchor_irt = (
                            np.conj(irt_character_map[anchor_physical])
                            if anchor_use_conjugate
                            else irt_character_map[anchor_physical]
                        )
                        nonzero_mask = np.abs(anchor_irt) > atol
                        if not np.any(nonzero_mask):
                            continue

                        phase_vector = np.ones_like(anchor_irt, dtype=complex)
                        phase_vector[nonzero_mask] = (
                            anchor_spgrep[nonzero_mask] / anchor_irt[nonzero_mask]
                        )
                        if not np.allclose(
                            np.abs(phase_vector[nonzero_mask]), 1.0, atol=atol
                        ):
                            continue
                        phase_vector[nonzero_mask] /= np.abs(phase_vector[nonzero_mask])

                        generic_candidates: dict[str, list[tuple[str, NDArray[np.complex128]]]] = {}
                        is_consistent = True
                        for generic_label in group_generic:
                            generic_chars = spgrep_character_map[generic_label]
                            candidates = []
                            for physical_label in group_physical:
                                for use_conjugate in [0, 1]:
                                    physical_chars = (
                                        np.conj(irt_character_map[physical_label])
                                        if use_conjugate
                                        else irt_character_map[physical_label]
                                    )
                                    transformed_chars = phase_vector * physical_chars
                                    if np.allclose(
                                        generic_chars, transformed_chars, atol=atol
                                    ):
                                        candidates.append((physical_label, transformed_chars))
                            if not candidates:
                                is_consistent = False
                                break
                            generic_candidates[generic_label] = candidates
                        if not is_consistent:
                            continue

                        assignment = search_assignment(generic_candidates)
                        if assignment is not None:
                            return assignment

            return None

        irt_character_map = {}
        for irt_irrep in irt_irreps:
            irt_label = f"{irt_irrep.name}({irt_irrep.dim})"
            irt_character_map[irt_label] = np.array(
                [irt_irrep.characters[idx + 1] for idx in mapping_little_group],
                dtype=complex,
            )

        spgrep_character_map = {}
        label_map = {}
        matched_character_map = {}
        used_labels = set()
        for i, irrep in enumerate(irreps):
            generic_label = f"irrep_{i+1}({irrep.shape[1]})"
            spgrep_character_map[generic_label] = np.array(get_character(irrep), dtype=complex)
            label_map[generic_label] = generic_label

        ordered_spgrep_labels = sorted(
            spgrep_character_map.keys(),
            key=lambda label: int(label.split("(")[1][:-1]),
            reverse=True,
        )

        # First, try strict direct character matching without conjugation.
        # For spinor irreps this is important: starting with conjugation-aware matching
        # immediately makes conjugate pairs ambiguous even when an exact label match
        # exists in the IRT table.
        for generic_label in ordered_spgrep_labels:
            spgrep_chars = spgrep_character_map[generic_label]
            spgrep_dim = int(generic_label.split("(")[1][:-1])
            matched_candidates = []
            for irt_label, irt_chars in irt_character_map.items():
                if irt_label in used_labels:
                    continue
                irt_dim = int(irt_label.split("(")[1][:-1])
                if irt_dim != spgrep_dim:
                    continue
                transformed_chars = match_direct(spgrep_chars, irt_chars)
                if transformed_chars is not None:
                    matched_candidates.append((irt_label, transformed_chars))
            if len(matched_candidates) == 1:
                irt_label, transformed_chars = matched_candidates[0]
                label_map[generic_label] = irt_label
                matched_character_map[generic_label] = transformed_chars
                used_labels.add(irt_label)

        if not self.spinor:
            progress = True
            while progress:
                progress = False
                unresolved_labels = [
                    label for label in ordered_spgrep_labels if label_map[label] == label
                ]
                if not unresolved_labels:
                    break

                unresolved_dims = sorted(
                    {int(label.split("(")[1][:-1]) for label in unresolved_labels},
                    reverse=True,
                )
                for irrep_dim in unresolved_dims:
                    group_generic = [
                        label for label in unresolved_labels
                        if int(label.split("(")[1][:-1]) == irrep_dim
                    ]
                    group_physical = [
                        label for label in sorted(irt_character_map.keys())
                        if label not in used_labels and int(label.split("(")[1][:-1]) == irrep_dim
                    ]
                    if len(group_generic) != len(group_physical) or not group_generic or len(group_generic) > 6:
                        continue

                    solutions = []
                    for perm in itertools.permutations(group_physical):
                        transformed = {
                            generic_label: irt_character_map[irt_label]
                            for generic_label, irt_label in zip(group_generic, perm)
                        }
                        is_consistent = True

                        for generic_label in group_generic:
                            for reference_label, reference_chars in matched_character_map.items():
                                spgrep_invariant = (
                                    spgrep_character_map[generic_label]
                                    * np.conj(spgrep_character_map[reference_label])
                                )
                                irt_invariant = (
                                    transformed[generic_label]
                                    * np.conj(reference_chars)
                                )
                                if not np.allclose(spgrep_invariant, irt_invariant, atol=1e-5):
                                    is_consistent = False
                                    break
                            if not is_consistent:
                                break
                        if not is_consistent:
                            continue

                        for i, left_label in enumerate(group_generic):
                            for right_label in group_generic[i + 1:]:
                                spgrep_invariant = (
                                    spgrep_character_map[left_label]
                                    * np.conj(spgrep_character_map[right_label])
                                )
                                irt_invariant = (
                                    transformed[left_label]
                                    * np.conj(transformed[right_label])
                                )
                                if not np.allclose(spgrep_invariant, irt_invariant, atol=1e-5):
                                    is_consistent = False
                                    break
                            if not is_consistent:
                                break
                        if is_consistent:
                            solutions.append((perm, transformed))

                    if not solutions:
                        continue

                    solutions.sort(key=lambda item: item[0])
                    perm, transformed = solutions[0]
                    for generic_label, irt_label in zip(group_generic, perm):
                        label_map[generic_label] = irt_label
                        matched_character_map[generic_label] = transformed[generic_label]
                        used_labels.add(irt_label)
                        progress = True

            kpoint_name = self.get_kpoint_name(k)
            overrides = get_label_overrides(
                space_group_number=self.spglib_dataset["number"],
                kpoint_name=kpoint_name,
                spinor=self.spinor,
            )
            if overrides:
                label_map = {
                    generic_label: overrides.get(resolved_label, resolved_label)
                    for generic_label, resolved_label in label_map.items()
                }

            unresolved_generic = [
                label for label in ordered_spgrep_labels if label_map[label] == label
            ]
            unresolved_physical = sorted(
                [label for label in irt_character_map.keys() if label not in used_labels],
                key=irrep_label_sort_key,
            )
            if unresolved_generic and len(unresolved_generic) == len(unresolved_physical):
                unresolved_generic = sorted(
                    unresolved_generic,
                    key=lambda label: (
                        int(label.split("(")[1][:-1]),
                        int(label.split("_")[1].split("(")[0]),
                    ),
                )
                for generic_label, physical_label in zip(unresolved_generic, unresolved_physical):
                    label_map[generic_label] = physical_label
            return label_map

        # For spinor irreps, direct character matching can fail because each irrep may
        # differ by a projective gauge. Gauge-invariant quotients chi_a * chi_b^* are
        # still comparable, so resolve the remaining labels by matching those quotient
        # patterns against irreptables.
        progress = True
        while progress:
            progress = False
            unresolved_labels = [
                label for label in ordered_spgrep_labels if label_map[label] == label
            ]
            if not unresolved_labels:
                break

            unresolved_dims = sorted(
                {int(label.split("(")[1][:-1]) for label in unresolved_labels},
                reverse=True,
            )
            for irrep_dim in unresolved_dims:
                group_generic = [
                    label for label in unresolved_labels
                    if int(label.split("(")[1][:-1]) == irrep_dim
                ]
                group_physical = [
                    label for label in sorted(irt_character_map.keys())
                    if label not in used_labels and int(label.split("(")[1][:-1]) == irrep_dim
                ]
                if len(group_generic) != len(group_physical) or not group_generic:
                    continue

                if len(group_generic) > 6:
                    continue

                solutions = []
                for perm in itertools.permutations(group_physical):
                    for conjugation_bits in itertools.product([0, 1], repeat=len(perm)):
                        transformed = {}
                        for generic_label, irt_label, use_conjugate in zip(
                            group_generic, perm, conjugation_bits
                        ):
                            transformed[generic_label] = (
                                np.conj(irt_character_map[irt_label])
                                if use_conjugate
                                else irt_character_map[irt_label]
                            )

                        is_consistent = True

                        # Compare quotients against already-resolved references.
                        for generic_label in group_generic:
                            for reference_label, reference_chars in matched_character_map.items():
                                spgrep_invariant = (
                                    spgrep_character_map[generic_label]
                                    * np.conj(spgrep_character_map[reference_label])
                                )
                                irt_invariant = (
                                    transformed[generic_label]
                                    * np.conj(reference_chars)
                                )
                                if not np.allclose(spgrep_invariant, irt_invariant, atol=1e-5):
                                    is_consistent = False
                                    break
                            if not is_consistent:
                                break
                        if not is_consistent:
                            continue

                        # Compare pairwise quotients inside the unresolved block.
                        for i, left_label in enumerate(group_generic):
                            for right_label in group_generic[i + 1:]:
                                spgrep_invariant = (
                                    spgrep_character_map[left_label]
                                    * np.conj(spgrep_character_map[right_label])
                                )
                                irt_invariant = (
                                    transformed[left_label]
                                    * np.conj(transformed[right_label])
                                )
                                if not np.allclose(spgrep_invariant, irt_invariant, atol=1e-5):
                                    is_consistent = False
                                    break
                            if not is_consistent:
                                break
                        if is_consistent:
                            solutions.append((perm, conjugation_bits, transformed))

                if not solutions:
                    continue

                solutions.sort(key=lambda item: (item[0], item[1]))
                perm, _, transformed = solutions[0]
                for generic_label, irt_label in zip(group_generic, perm):
                    label_map[generic_label] = irt_label
                    matched_character_map[generic_label] = transformed[generic_label]
                    used_labels.add(irt_label)
                    progress = True

        progress = True
        while progress:
            progress = False
            unresolved_labels = [
                label for label in ordered_spgrep_labels if label_map[label] == label
            ]
            if not unresolved_labels:
                break

            unresolved_dims = sorted(
                {int(label.split("(")[1][:-1]) for label in unresolved_labels},
                reverse=True,
            )
            for irrep_dim in unresolved_dims:
                if irrep_dim != 1:
                    continue
                group_generic = [
                    label for label in unresolved_labels
                    if int(label.split("(")[1][:-1]) == irrep_dim
                ]
                group_physical = [
                    label for label in sorted(irt_character_map.keys())
                    if label not in used_labels and int(label.split("(")[1][:-1]) == irrep_dim
                ]
                if len(group_generic) != len(group_physical) or not group_generic:
                    continue

                assignment = resolve_spinor_phase_convention_block(
                    group_generic=group_generic,
                    group_physical=group_physical,
                )
                if assignment is None:
                    continue

                for generic_label, (physical_label, transformed_chars) in assignment.items():
                    label_map[generic_label] = physical_label
                    matched_character_map[generic_label] = transformed_chars
                    used_labels.add(physical_label)
                    progress = True

        kpoint_name = self.get_kpoint_name(k)
        overrides = get_label_overrides(
            space_group_number=self.spglib_dataset["number"],
            kpoint_name=kpoint_name,
            spinor=self.spinor,
        )
        if overrides:
            label_map = {
                generic_label: overrides.get(resolved_label, resolved_label)
                for generic_label, resolved_label in label_map.items()
            }

        return label_map

    def get_target_element_positions(self, element: str) -> list[int]:
        all_chemical_symbols = get_chemical_symbols(self.primitive_cell)
        if not element in all_chemical_symbols:
            raise ValueError(
                f"Element \"{element}\" is not in the inputed cell."
                )
        target_element_positions = [i for i, symbol in enumerate(all_chemical_symbols) if symbol == element]
        return target_element_positions

    def _get_rotations_at_k(
        self,
        rotations: NDArray[np.int_],
        translations: NDArray[np.float_],
        k: list[float],
    ) -> tuple[NDArray[np.int_], NDArray[np.float_]]:
        """Get little-group operations using a stable integer-lattice test."""
        k = canonicalize_kpoint(k)
        rotations_at_k = []
        translations_at_k = []
        for rotation, translation in zip(rotations, translations):
            diff = np.dot(k, rotation) - k
            if (abs(diff - np.rint(diff)) < 1e-5).all():
                rotations_at_k.append(rotation)
                normalized_translation = np.array(translation, dtype=float).copy()
                for i in range(3):
                    if abs(normalized_translation[i] - 1.0) < 1e-5:
                        normalized_translation[i] = 0.0
                translations_at_k.append(normalized_translation)
        return np.array(rotations_at_k), np.array(translations_at_k)

    def get_modified_permutation_rep(self, 
                                     r: NDArray[np.int_], 
                                     t: NDArray[np.float_], 
                                     k: list[float, float, float]
                                     ) -> NDArray[np.complex128]:
        """Get permutation matrix at the k point"""
        pos = get_scaled_positions(self.primitive_cell)
        num_atom = len(pos)
        matrix = np.zeros((num_atom, num_atom), dtype=complex)
        for i, p1 in enumerate(pos):
            p_rot = np.dot(r, p1) + t  # i -> j
            for j, p2 in enumerate(pos):
                diff = p_rot - p2  # Rx_i + t - x_j
                if (abs(diff - np.rint(diff)) < 1e-5).all():
                    phase_factor = np.dot(
                        k, np.dot(np.linalg.inv(r), p2 - t) - p2
                    )
                    matrix[j, i] = np.exp(2j * np.pi * phase_factor)
        return matrix
    
    def get_permutation_reps_at_k(self,
                                  little_rotations,
                                  little_translations,
                                  k: list[float]
                                  ) -> tuple[NDArray[np.int_], NDArray[np.float_], NDArray[np.complex128]]:
        """Get permutation matrices at given k point.
        
        Parameter
        ---------
        mapping_little_group: NDArray, (little_group_order, )
        k: list, (3, )
            [kx, ky, kz], coordinates of the k point in primitive basis.

        Returns
        -------
        little_rotations: NDArray[np.int_]
            Rotations in primitive basis at k.
        little_translations: NDArray[np.float_]
            Translations in primitive basis at k.
        permutation_matrices: NDArray[np.complex128]
            Permutation matrices of symmetry operations at k.
        """
        permutation_matrices = []
        for r, t in zip(little_rotations, little_translations):
            permutation_matrix = self.get_modified_permutation_rep(r, t, k)
            permutation_matrices.append(permutation_matrix)
        permutation_matrices = np.array(permutation_matrices)
        return permutation_matrices
    
    def get_little_group(self, k: list[float]):
        """Same as spgrep.group.get_little_group."""
        k = canonicalize_kpoint(k)
        little_rotations, little_translations, mapping_little_group = get_little_group(
            rotations=self.rotations, 
            translations=self.translations,
            kpoint=k
        )
        return little_rotations, little_translations

    def get_little_group_symbol(self,
                                k: list[float]
                                ) -> str:
        """Get little group symbol of k from rotations and translations at k."""
        little_rotations, little_translations = self._get_rotations_at_k(self.rotations, self.translations, k)
        site_sym_k = get_spacegroup_type_from_symmetry(little_rotations, little_translations)
        if hasattr(site_sym_k, "international_short"):
            international_short = site_sym_k.international_short
        else:
            international_short = site_sym_k["international_short"]
        if hasattr(site_sym_k, "number"):
            number = site_sym_k.number
        else:
            number = site_sym_k["number"]
        return f"{international_short} ({number})"

    def get_atomic_orbital_characters(self,
                                      rotations: NDArray[np.int_],
                                      orbital: str
                                      ) -> dict[str: float]:
        """Calculate characters of the atomic orbital for each rotation.
        
        Return
        ------
        result : dict[str: float]
            {key = rotaion: value = character}
        """
        # azimuthal number (l)
        orbital_azimuthal_num = {"s" : 0, "p" : 1, "d" : 2, "f" : 3, "g" : 4, "h" : 5, "i" : 6}

        if not orbital in orbital_azimuthal_num.keys():
            raise ValueError(
                f"Orbital \"{orbital}\" cannot be analyzed"
                )
        characters = []
        l = orbital_azimuthal_num[orbital]

        for r in rotations:
            is_proper, rot_order = characterize_rotation(r)
            alpha = 2 * np.pi / rot_order
            if rot_order == 1:
                character = 2 * l + 1
            else:
                character = np.sin((l + 1/2) * alpha) / np.sin(alpha / 2)
            if not is_proper:
                character = character * (-1) ** l
            
            if self.spinor:
                # spinor representation
                if rot_order == 1:
                    character = character * 2
                else:
                    character = character * 2 * np.cos(alpha / 2)
            characters.append(character)
        return np.array(characters)

    def get_permutation_characters(self, 
                                   element: str, 
                                   permutation_matrices: NDArray[np.complex128]
                                   ) -> NDArray[np.complex128]:
        """Calculate permutaion characters of target atoms."""
        elemet_positions = self.get_target_element_positions(element)
        range_strat, range_end = elemet_positions[0], elemet_positions[-1]+1

        permutation_characters = []
        for i in range(permutation_matrices.shape[0]):
            rep = permutation_matrices[i][range_strat : range_end, range_strat : range_end]
            character = np.trace(rep)
            permutation_characters.append(character)
        return np.array(permutation_characters, dtype=complex)
        
    def calc_reducible_characters(self,
                                  k: list[float],
                                  element: str,
                                  orbital: str,
                                  mapping_little_group: NDArray[np.int_],
                                  ) -> NDArray[np.complex128]:
        """Combine the characters of permutations with the characters of atomic orbital."""
        k = canonicalize_kpoint(k)
        little_rotations = self.rotations[mapping_little_group]
        little_translations = self.translations[mapping_little_group]
        permutation_matrices = self.get_permutation_reps_at_k(little_rotations, little_translations, k)
        a_o_charac = self.get_atomic_orbital_characters(little_rotations, orbital)
        perm_charac = self.get_permutation_characters(element, permutation_matrices)
        reducible_characters = perm_charac * a_o_charac
        return reducible_characters

    def irreducible_decomposition(self, 
                                  k: list[float, float, float], 
                                  element: str, 
                                  orbital: str
                                  ) -> tuple[NDArray[np.int_], object, dict[str: float], dict[str, str]]:
        """Get irreps of crystal orbital at k.

        Parameter
        ---------
        k: list[float, float, float]
            [kx, ky, kz], k point in primtive basis.
        element: str
            Element.
        orbital: str
            Orbital.

        Returns
        -------
        little_group_symbol: str
            Little group of k.
        result : dict[str: float]
            {key = irrep(dimension): value = number of the irrep}
        """
        k = canonicalize_kpoint(k)
        # get irreps
        if self.spinor:
            from spgrep.core import get_spacegroup_spinor_irreps_from_primitive_symmetry
            irreps, little_spinor_factor_system, little_unitary_rotations, mapping_little_group = get_spacegroup_spinor_irreps_from_primitive_symmetry(
                                                                          lattice=self.primitive_cell.cell,
                                                                          rotations=self.rotations,
                                                                          translations=self.translations,
                                                                          kpoint=k
                                                                          )
        else:
            from spgrep.core import get_spacegroup_irreps_from_primitive_symmetry
            irreps, mapping_little_group = get_spacegroup_irreps_from_primitive_symmetry(
                                                                   rotations=self.rotations,
                                                                   translations=self.translations,
                                                                   kpoint=k
                                                                   )
        reducible_characters = self.calc_reducible_characters(
            k=k,
            element=element,
            orbital=orbital,
            mapping_little_group=mapping_little_group,
        )
        # irreducible decomposition
        result = {}
        for i, irrep in enumerate(irreps):
            irrep_characters = get_character(irrep)
            num = np.dot(
                reducible_characters, np.conj(irrep_characters)
                ) / irrep.shape[0]
            result[f'irrep_{i+1}({irrep.shape[1]})'] = np.round(num.real, 2)
        irrep_labels = self.get_irrep_labels(k, irreps, mapping_little_group)
        return mapping_little_group, irreps, result, irrep_labels


def make_irrep_table(crystal_orbital, irreps, mapping_little_group, irrep_labels):
    from pandas import DataFrame, set_option
    set_option('display.max_columns', 50)
    set_option('display.width', 1000)
    little_r_symbol = [crystal_orbital.seitz_symbols[idx] for idx in mapping_little_group]
    irrep_table = DataFrame(columns=little_r_symbol)
    for i, irrep in enumerate(irreps):
        generic_label = f"irrep_{i+1}({irrep.shape[1]})"
        resolved_label = irrep_labels[generic_label]
        row_label = generic_label if resolved_label == generic_label else f"{generic_label} = {resolved_label}"
        irrep_table.loc[row_label] = list(np.round(get_character(irrep), 4))
    #irrep_table.to_csv("irrep_table.csv")
    irrep_table.to_excel("irrep_table.xlsx")
    print(irrep_table)


def sort_irrep_items(items):
    """Sort irrep labels by index, then parity (+ before -), then dimension."""
    def sort_key(item):
        label, _ = item
        return irrep_label_sort_key(label)

    return sorted(items, key=sort_key)


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    cell, _ = read_crystal_structure(args.poscar, interface_mode='vasp')
    crystal_orbital = CrystalOrbital(cell=cell, symprec=args.tolerance, spior=args.spinor)

    elemet_positions = crystal_orbital.get_target_element_positions(args.element)
    range_strat, range_end = elemet_positions[0], elemet_positions[-1]+1
    wyckoff_letters = crystal_orbital.spglib_dataset['wyckoffs'][range_strat:range_end]
    site_symmetry_symbols = crystal_orbital.spglib_dataset['site_symmetry_symbols'][range_strat:range_end]

    if args.kpoint is None:
        print(f"\n * Space group *\n {crystal_orbital.spglib_dataset['international']} ({crystal_orbital.spglib_dataset['number']})\n")
        print(f" * Element (number of atoms) *\n {args.element} ({len(wyckoff_letters)})\n")
        print(f" * Wyckoff letters and site symmetry letters *")
        print(f" {wyckoff_letters}\n {site_symmetry_symbols}\n")
        print(f" * Atomic Orbital *\n {args.orbital}\n")
        print(" * Crystal Orbitals *")

        kpoint_names, kpoints = crystal_orbital.get_irt_special_points()
        for kpoint_name, kpoint in zip(kpoint_names, kpoints):
            try:
                mapping_little_group, irreps, bandreps, irrep_labels = crystal_orbital.irreducible_decomposition(
                    k=kpoint,
                    element=args.element,
                    orbital=args.orbital,
                )
            except Exception as error:
                print(f"ERROR at {kpoint_name} {kpoint}")
                print(f"{error}\n")
                continue

            display_bandreps = {irrep_labels[key]: value for key, value in bandreps.items()}
            sorted_bandreps = sort_irrep_items(
                [(key, value) for key, value in display_bandreps.items() if value > 0]
            )
            irreps_result = "+".join([f" {value} [{key}] " for key, value in sorted_bandreps])
            little_group = crystal_orbital.get_little_group_symbol(kpoint)
            rounded_kpoint = format_kpoint(kpoint)

            print(f" k point (primitive):  {kpoint_name} {rounded_kpoint}")
            print(f" little group of k  :  {little_group}")
            print(f" irreps             : {irreps_result}\n")
        return

    print(f"\n * Space group *\n {crystal_orbital.spglib_dataset['international']} ({crystal_orbital.spglib_dataset['number']})\n")
    print(f" * Orbital (number of atoms) *\n {args.element}_{args.orbital} ({len(wyckoff_letters)})\n")
    print(f" * Position *")
    print(f" wyckoff letters      : {wyckoff_letters}")
    print(f" site symmetry letters: {site_symmetry_symbols}\n")
    print(f" * Spinor *\n {args.spinor}\n")

    mapping_little_group, irreps, bandreps, irrep_labels = crystal_orbital.irreducible_decomposition(
        k=args.kpoint,
        element=args.element,
        orbital=args.orbital,
    )

    display_bandreps = {irrep_labels[key]: value for key, value in bandreps.items()}

    kpoint_name = crystal_orbital.get_kpoint_name(args.kpoint)
    formatted_input_kpoint = format_kpoint(args.kpoint)
    if kpoint_name is None:
        print(f" * k point (primitive) * \n {formatted_input_kpoint}\n")
    else:
        print(f" * k point (primitive) * \n {kpoint_name} {formatted_input_kpoint}\n")

    if args.table:
        little_group = crystal_orbital.get_little_group_symbol(args.kpoint)
        print(" * IrRep Table * ")
        print(f" little group: {little_group}")
        print(" table:")
        make_irrep_table(crystal_orbital, irreps, mapping_little_group, irrep_labels)
        print("")

    print(" * Atomic Band Irreducible Representations *")
    sorted_bandreps = sort_irrep_items(
        [(key, value) for key, value in display_bandreps.items() if value > 0]
    )
    irreps_result = "+".join([f" {value} [{key}] " for key, value in sorted_bandreps])
    print(irreps_result + "\n")


if __name__ == "__main__":
    main()
