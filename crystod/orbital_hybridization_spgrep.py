"""
__author__ = "Hiroki Koiso"
__copyright__ = "Copyright 2024, Nakajima group"
__version__ = "1.1.0"
__maintainer__ = "Hiroki Koiso"
__email__ = "koiso.h.aa@m.titech.ac.jp"
__status__ = "Development"
__released_date__ = "June 11, 2024"
__last_update__= "Augst 26, 2024"
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
from argparse import ArgumentParser, RawTextHelpFormatter, RawDescriptionHelpFormatter, ArgumentDefaultsHelpFormatter
import numpy as np
from numpy.typing import NDArray
from typing import Optional
import itertools
import re

IrrepTable, Irrep = load_irreptables()

### PARSER STRUCTURE ###
class MyHelpFormatter(RawTextHelpFormatter, RawDescriptionHelpFormatter, ArgumentDefaultsHelpFormatter):
    pass
desc = """
This program assists in the drawing of crystal orbital diagrams.
POSCAR and atomic orbitals are input, and the atomic orbitals that consist of the crystal orbitals at each k are displayed.

# Command Example:
python3 orbital_hybridization.py --poscar POSCAR_ScF3_Pm-3m --orbital Sc_s Sc_p Sc_d F_s F_p
"""
### --------------- ###


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=desc, formatter_class=MyHelpFormatter)
    parser.add_argument("--poscar", "-poscar", dest="poscar", type=str, default="POSCAR",
                        help="POSCAR.")
    parser.add_argument("--orbital", "-orbital", dest="orbital", nargs='*', required=True, type=str, default=None,
                        help="Elements and orbitals like Sc_s. You can choose from following orbitals; s, p, d, f, g, h, i.")
    parser.add_argument("--kpoint", "-kpoint", dest="kpoint", type=float, nargs=3, required=False,
                        help="k-point.")
    parser.add_argument("--show-irrep-table", "-show-table", dest="table", action="store_true",
                        help="Show table of irreps.")
    parser.add_argument("--tolerance", "-tolerance", dest="tolerance", type=float, default=0.00001,
                        help="Symmetry tolerance to search primitive cell.")
    parser.add_argument("--spinor", "-spinor", dest="spinor", action="store_true",
                        help="Calculate double space group representations.")
    return parser


def similarity_transformation(rot: NDArray[np.float_], mat: NDArray[np.float_]) -> NDArray[np.float_]:
    """Similarity transformation by R x M x R^-1."""
    return rot @ mat @ np.linalg.inv(rot)


def format_kpoint(kpoint: list[float], decimals: int = 2) -> list[float]:
    """Format a k-point for display using plain Python floats."""
    return [float(np.round(float(value), decimals)) for value in kpoint]


def characterize_rotation(rotation: NDArray[np.int_]) -> tuple[bool, int]:
    is_proper = True
    no_inv_r = rotation
    r_det = np.linalg.det(rotation)
    if int(r_det) == -1:
        is_proper = False
        # remove invention.
        no_inv_r = -1 * no_inv_r
    # get the rotation order.
    e = np.eye(3, 3, dtype=int)
    rot_order = 0
    test_r = no_inv_r
    for _ in range(7):
        rot_order += 1
        if (test_r == e).all():
            break
        test_r = no_inv_r @ test_r
    if rot_order not in [1, 2, 3, 4, 6]:
        ValueError("This rotation matrix is not a symmetry operation of the space group.")
    return is_proper, rot_order
    
def get_seitz_symbol(rotation: NDArray[np.int_],
                     trans_mat: NDArray[np.float_]
                     ) -> str:
    """Labelling the rotation part of the symmetry operation with Seitz notation.

    Parameters
    ----------
    rotation: ndarray, (3, 3)
        Rotation part of the symmetry operation.
    trans_mat: ndarray, (3, 3)
        Transformation matrix.

    Returns
    -------
    is_proper: bool
        Proper or not.
    rot_order: int
        Rotaion order.
    symbol: str
        Seitz notation of the rotaion.
    """
    r_trans = trans_mat @ rotation @ np.linalg.inv(trans_mat)
    r = (np.rint(r_trans)).astype(np.int_)
    is_proper, rot_order = characterize_rotation(r)
    if (rot_order == 1):
        if is_proper:
            return '1'
        else:
            return '-1'
    else:
        if is_proper:
            no_inv_r = r
        else:
            # remove invention.
            no_inv_r = -1 * r

    # get the characteristic direction.
    # characteristic direction is the eigenvector with the eigenvalue of 1.
    eig_vals, eig_vecs = np.linalg.eig(no_inv_r)
    eig_vecs = eig_vecs.astype(np.complex128)
    for idx, w in enumerate(eig_vals):
        if abs(w - 1.0) < 1e-8:
            v = eig_vecs[:, idx]
            # round the eigen vector to an integer ratio.
            uniques = np.unique(v).real # sort values
            round_num = uniques[0]
            if abs(round_num) < 1e-8: # check round_num is 0 or not
                round_num = uniques[1]
            direction = np.rint(v.real / round_num).astype(np.int_)
            break
    # get the sign of rotation and make the symbol.
    if (rot_order == 2):
        if is_proper:
            return '2_' + ''.join([str(xi) for xi in direction])
        else:
            return 'm_' + ''.join([str(xi) for xi in direction])
    else: # for rot_order is 3, 4, or 6.
        def get_rot_sign(r, direction):
            """Getting sign of the rotation.
            In this function, rot_axis is the axis of positive rotation.
            For further details about computationing rot_axis, see:
            Technical Concepts: Orientation, Rotation, Velocity and Acceleration and the SRM, P. Berner, Version 2.0, 2008,
            <http://sedris.org/wg8home/Documents/WG80485.pdf>, pp. 33, 61-63.
            """
            rot_axis = np.array([r[2,1] - r[1,2],
                                 r[0,2] - r[2,0],
                                 r[1,0] - r[0,1]])
            # Compare the direction of rotational axis and the characteristic direction by inner product.
            if np.dot(rot_axis, direction) > 0:
                return '+'
            else:
                return '-'
        if is_proper:
            rot_sign = get_rot_sign(no_inv_r, direction)
            return str(rot_order) + '^' + rot_sign + '_' + ''.join([str(xi) for xi in direction])
        else:
            rot_sign = get_rot_sign(no_inv_r, direction)
            return '-' + str(rot_order) + '^' + rot_sign + '_' + ''.join([str(xi) for xi in direction])

class CrystalOrbital:
    def __init__(
            self, 
            cell: PhonopyAtoms,
            symprec: float = 0.001,
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
        trans_inv = np.linalg.inv(self.transformation_matrix)
        conventional_k = np.array(k) @ trans_inv
        irreps_at_k = []
        for irrep_at_k in self.irt_character_table.irreps:
            if np.allclose(irrep_at_k.k, conventional_k, atol=1e-6):
                irreps_at_k.append(irrep_at_k)
        return irreps_at_k

    def get_kpoint_name(self, k: list[float]) -> Optional[str]:
        """Get the special k-point name from irreptables if available."""
        irreps_at_k = self.get_irt_irreps_at_k(k)
        if irreps_at_k:
            return irreps_at_k[0].kpname
        return None

    def get_irt_special_points(self) -> tuple[list[str], list[list[float]]]:
        """Get unique special k-points from irreptables in primitive basis."""
        kpoint_names = []
        primitive_kpoints = []
        for irrep in self.irt_character_table.irreps:
            primitive_k = list(np.round(np.array(irrep.k) @ self.transformation_matrix, 6))
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
        irt_irreps = self.get_irt_irreps_at_k(k)
        if not irt_irreps:
            return {
                f"irrep_{i+1}({irrep.shape[1]})": f"irrep_{i+1}({irrep.shape[1]})"
                for i, irrep in enumerate(irreps)
            }

        def match_with_possible_conjugation(
            lhs: NDArray[np.complex128],
            rhs: NDArray[np.complex128],
            atol: float = 1e-5,
        ) -> Optional[NDArray[np.complex128]]:
            if np.allclose(lhs, rhs, atol=atol):
                return rhs
            rhs_conj = np.conj(rhs)
            if np.allclose(lhs, rhs_conj, atol=atol):
                return rhs_conj
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
                transformed_chars = match_with_possible_conjugation(spgrep_chars, irt_chars)
                if transformed_chars is not None:
                    matched_candidates.append((irt_label, transformed_chars))
            if len(matched_candidates) == 1:
                irt_label, transformed_chars = matched_candidates[0]
                label_map[generic_label] = irt_label
                matched_character_map[generic_label] = transformed_chars
                used_labels.add(irt_label)

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

        return label_map

    def get_target_element_positions(self, element: str) -> list[int]:
        all_chemical_symbols = get_chemical_symbols(self.primitive_cell)
        if not element in all_chemical_symbols:
            raise ValueError(
                f"Element \"{element}\" is not in the inputed cell."
                )
        target_element_positions = [i for i, symbol in enumerate(all_chemical_symbols) if symbol == element]
        return target_element_positions

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
        little_rotations, little_translations = self.get_little_group(k)
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
                                  orbital: str
                                  ) -> NDArray[np.complex128]:
        """Combine the characters of permutations with the characters of atomic orbital."""
        little_rotations, little_translations = self.get_little_group(k)
        permutation_matrices = self.get_permutation_reps_at_k(little_rotations, little_translations, k)
        a_o_charac = self.get_atomic_orbital_characters(little_rotations, orbital)
        perm_charac = self.get_permutation_characters(element, permutation_matrices)
        reducible_characters = perm_charac * a_o_charac
        return reducible_characters
    
    def get_irreps(self, k: list[float]):
        """Get irreps at k."""
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
        return irreps, mapping_little_group

    def irreducible_decomposition(self, 
                                  k: list[float], 
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
        irreps, mapping_little_group = self.get_irreps(k)
        reducible_characters = self.calc_reducible_characters(k=k, element=element, orbital=orbital)
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


def make_irrep_table(crystal_orbital, k, irreps, mapping_little_group, irrep_labels):
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
    pattern = re.compile(
        r"^(?P<spinor>-?)(?P<kpoint>[A-Z]+)(?P<index>\d+)(?P<parity>[+-]?)(?:\((?P<dim>\d+)\))$"
    )

    def sort_key(item):
        label, _ = item
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

    return sorted(items, key=sort_key)

def get_result_at_k(crystal_orbital, k, elements, orbitals):
    irreps, mapping_little_group = crystal_orbital.get_irreps(k)
    result = {f'irrep_{i+1}({irrep.shape[1]})': [] for i, irrep in enumerate(irreps)}
    irrep_labels = crystal_orbital.get_irrep_labels(k, irreps, mapping_little_group)
    for element, orbital in zip(elements, orbitals):
        mapping_little_group, irreps, bandreps, current_labels = crystal_orbital.irreducible_decomposition(k=k, element=element, orbital=orbital)
        for symbol, num in bandreps.items():
            if float(num) != 0.0:
                result[symbol].append(f"{element}({orbital})")
    labeled_result = {
        irrep_labels[key]: value
        for key, value in result.items()
    }
    return mapping_little_group, irreps, labeled_result, irrep_labels


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    cell, _ = read_crystal_structure(args.poscar, interface_mode='vasp')
    crystal_orbital = CrystalOrbital(cell=cell, symprec=args.tolerance, spior=args.spinor)

    input_orbitals = args.orbital
    elements = []
    orbitals = []
    for in_orbit in input_orbitals:
        e, o = in_orbit.split('_')
        elements.append(e)
        orbitals.append(o)

    wyckoff_letters = crystal_orbital.spglib_dataset['wyckoffs']
    site_symmetry_symbols = crystal_orbital.spglib_dataset['site_symmetry_symbols']

    print(f"\n * Space group *\n {crystal_orbital.spglib_dataset['international']} ({crystal_orbital.spglib_dataset['number']})\n")
    print(f" * Position *")
    print(f" wyckoff letters      : {wyckoff_letters}")
    print(f" site symmetry letters: {site_symmetry_symbols}\n")
    print(f" * Atomic Orbital *\n {args.orbital}\n")
    print(f" * Spinor *\n {args.spinor}\n")

    if args.kpoint is None:
        kpoint_names, kpoints = crystal_orbital.get_irt_special_points()
        print(" * Result *")
        for kpoint_name, kpoint in zip(kpoint_names, kpoints):
            mapping_little_group, irreps, labeled_result, irrep_labels = get_result_at_k(
                crystal_orbital,
                kpoint,
                elements,
                orbitals,
            )
            little_group = crystal_orbital.get_little_group_symbol(kpoint)
            print(f" k point (primitive):  {kpoint_name} {format_kpoint(kpoint)}")
            print(f" little group of k  :  {little_group}")
            if args.table:
                print(" * IrRep Table * ")
                print(f" little group: {little_group}")
                print(" table:")
                make_irrep_table(crystal_orbital, kpoint, irreps, mapping_little_group, irrep_labels)
            sorted_result = sort_irrep_items(list(labeled_result.items()))
            show_result = '\n'.join([f" {key}: {' '.join(value)}" for key, value in sorted_result])
            print(show_result)
            print("")
        return

    kpoint_name = crystal_orbital.get_kpoint_name(args.kpoint)
    formatted_kpoint = format_kpoint(args.kpoint)
    if kpoint_name is None:
        print(f" * k point (primitive) * \n {formatted_kpoint}\n")
    else:
        print(f" * k point (primitive) * \n {kpoint_name} {formatted_kpoint}\n")

    mapping_little_group, irreps, labeled_result, irrep_labels = get_result_at_k(
        crystal_orbital,
        args.kpoint,
        elements,
        orbitals,
    )

    if args.table:
        little_group = crystal_orbital.get_little_group_symbol(args.kpoint)
        print(" * IrRep Table * ")
        print(f" little group: {little_group}")
        print(" table:")
        make_irrep_table(crystal_orbital, args.kpoint, irreps, mapping_little_group, irrep_labels)
        print("")

    print(" * Result *")
    sorted_result = sort_irrep_items(list(labeled_result.items()))
    show_result = '\n'.join([f" {key}: {' '.join(value)}" for key, value in sorted_result])
    print(show_result)
    print("\n")


if __name__ == "__main__":
    main()
