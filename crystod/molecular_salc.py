"""Molecular point-group detection and molecular SALCs (crystod-mol).

Molecular counterpart of the crystalline SALC analysis:

- ``--symmetry``: detect the point group of a molecule (XYZ file) with
  pymatgen's ``PointGroupAnalyzer`` (the molecular analogue of
  ``phonopy --symmetry`` for crystals).
- ``--element EL --orbital ORB``: build the site-permutation representation
  of the selected element's sites, multiply its characters by the characters
  of the atomic orbital (s, p, d, f), decompose the product into the irreps
  of the molecular point group, and project out the explicit SALCs with the
  projection-operator technique (permutation x real-orbital Wigner-D
  representation).

The detected Schoenflies point group is mapped to its Hermann-Mauguin symbol
and connected to the same character tables used by ``crystod-group``
(``--decompose``/``--ligand-field``), so the irrep labels are identical across
the molecular and crystalline analyses.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from .decompose_irrep import decompose, get_character_table
from .ligand_field import ORBITAL_AZIMUTHAL_NUMBER, get_orbital_characters
from .operations import wigner_D_real

# Schoenflies -> Hermann-Mauguin for the 32 crystallographic point groups.
SCHOENFLIES_TO_HM = {
    "C1": "1", "Ci": "-1", "C2": "2", "Cs": "m", "C2h": "2/m",
    "D2": "222", "C2v": "mm2", "D2h": "mmm",
    "C4": "4", "S4": "-4", "C4h": "4/m", "D4": "422", "C4v": "4mm",
    "D2d": "-42m", "D4h": "4/mmm",
    "C3": "3", "S6": "-3", "C3i": "-3", "D3": "32", "C3v": "3m", "D3d": "-3m",
    "C6": "6", "C3h": "-6", "C6h": "6/m", "D6": "622", "C6v": "6mm",
    "D3h": "-6m2", "D6h": "6/mmm",
    "T": "23", "Th": "m-3", "O": "432", "Td": "-43m", "Oh": "m-3m",
}

ORBITAL_LABELS = {
    0: ["s"],
    1: ["px", "py", "pz"],
    2: ["dxy", "dyz", "dz2", "dxz", "dx2-y2"],
    3: ["fx(x2-3y2)", "fy(3x2-y2)", "fz(x2-y2)", "fxyz", "fxz2", "fyz2", "fz3"],
}

# Conventional hexagonal basis (rows are lattice vectors) used to convert the
# fractional rotation matrices of the trigonal/hexagonal character tables to
# Cartesian; every other crystal family already stores orthogonal matrices.
_HEXAGONAL_BASIS = np.array(
    [[1.0, 0.0, 0.0], [-0.5, np.sqrt(3.0) / 2.0, 0.0], [0.0, 0.0, 1.0]]
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Molecular point-group symmetry and molecular SALCs."
    )
    parser.add_argument("--xyz", required=True, metavar="FILE",
                        help="Molecule file in XYZ format.")
    parser.add_argument("--symmetry", action="store_true",
                        help="Only detect and print the molecular point group.")
    parser.add_argument("--element", default=None,
                        help="Element whose sites carry the orbitals, e.g. H.")
    parser.add_argument("--orbital", default=None,
                        help=f"Orbital shell: one of {', '.join(ORBITAL_AZIMUTHAL_NUMBER)}.")
    parser.add_argument("--tolerance", type=float, default=0.3,
                        help="Distance tolerance (Angstrom) for the symmetry "
                             "detection (default: 0.3, as in pymatgen).")
    parser.add_argument("--show-matrix", action="store_true",
                        help="Print the site-permutation matrix of every "
                             "symmetry operation.")
    parser.add_argument("--align", action="store_true",
                        help="Rotate the molecule into the standard "
                             "point-group orientation (principal axis along z) "
                             "before the SALC analysis.")
    parser.add_argument("--visualize", action="store_true",
                        help="Write the SALCs as an interactive 3D HTML viewer "
                             "(same viewer as crystod --visualize).")
    parser.add_argument("--output", default=None, metavar="FILE",
                        help="Output HTML path for --visualize "
                             "(default: SALC_{molecule}_{element}_{orbital}.html).")
    parser.add_argument("--bond", action="append", nargs=3, default=None,
                        metavar=("EL1", "EL2", "MAX"),
                        help="Draw bonds between EL1 and EL2 atoms up to MAX "
                             "Angstroms in the viewer (repeatable), e.g. --bond N H 1.2.")
    return parser


# ------------------------------------------------------------------ molecule


def load_molecule(path: str):
    if not os.path.isfile(path):
        raise SystemExit(f"ERROR: molecule file not found: {path}")
    try:
        from pymatgen.core import Molecule
    except ImportError as exc:
        raise SystemExit(
            "ERROR: crystod-mol requires pymatgen (pip install pymatgen)."
        ) from exc
    return Molecule.from_file(path).get_centered_molecule()


def get_symmetry(molecule, tolerance: float):
    """Return (schoenflies_symbol, unique symmetry operations)."""
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer

    analyzer = PointGroupAnalyzer(molecule, tolerance=tolerance)
    seen = {}
    for operation in analyzer.get_symmetry_operations():
        rotation = np.asarray(operation.rotation_matrix, dtype=float)
        seen[tuple(np.round(rotation, 6).ravel())] = rotation
    return str(analyzer.get_pointgroup()), list(seen.values())


# ------------------------------------------- class assignment (table matching)


def _table_operations_cartesian(character_table: dict):
    """All group elements of the character table in Cartesian coordinates,
    with their class labels."""
    operations, class_labels = [], []
    for class_name in character_table["rotation_list"]:
        for matrix in np.asarray(character_table["mapping_table"][class_name], dtype=float):
            operations.append(matrix)
            class_labels.append(class_name)
    operations = np.array(operations)

    identity_like = all(
        np.allclose(op @ op.T, np.eye(3), atol=1e-8) for op in operations
    )
    if not identity_like:
        # trigonal/hexagonal tables: fractional matrices in hexagonal axes
        basis = _HEXAGONAL_BASIS
        to_cart = basis.T
        operations = np.array(
            [to_cart @ op @ np.linalg.inv(to_cart) for op in operations]
        )
        if not all(np.allclose(op @ op.T, np.eye(3), atol=1e-8) for op in operations):
            raise SystemExit("ERROR: could not orthogonalize the character-table matrices.")
    return operations, class_labels


def _axis_angle(rotation: np.ndarray):
    """(det, rotation angle, axis) of an O(3) operation; axis is None for E/i."""
    det = 1.0 if np.linalg.det(rotation) > 0 else -1.0
    proper = det * rotation
    angle = float(np.arccos(np.clip((np.trace(proper) - 1.0) / 2.0, -1.0, 1.0)))
    if angle < 1e-6:
        return det, angle, None
    eigenvalues, eigenvectors = np.linalg.eig(proper)
    index = int(np.argmin(np.abs(eigenvalues - 1.0)))
    axis = np.real(eigenvectors[:, index])
    return det, angle, axis / np.linalg.norm(axis)


def _signature(info) -> tuple:
    return (int(round(info[0])), round(info[1], 3))


def _completed_frame(v1: np.ndarray, v2: np.ndarray | None) -> np.ndarray:
    """Right-handed orthonormal frame with v1 as first column."""
    if v2 is None or abs(abs(float(v1 @ v2)) - 1.0) < 1e-6:
        v2 = np.eye(3)[int(np.argmin(np.abs(v1)))]
    v2 = v2 - (v2 @ v1) * v1
    v2 = v2 / np.linalg.norm(v2)
    return np.column_stack([v1, v2, np.cross(v1, v2)])


def _match_operations(mol_ops, table_ops, class_labels):
    """Match the molecular operations to the character-table operations.

    Finds an orthogonal transform U with {U R U^T} = {table ops} by aligning
    a primary and a secondary symmetry axis, then matches element by element.
    Returns ``(U, matched_indices)`` where ``table_ops[matched_indices[i]]``
    corresponds to ``mol_ops[i]``.
    """
    if len(mol_ops) != len(table_ops):
        raise SystemExit(
            f"ERROR: molecular group order ({len(mol_ops)}) does not match the "
            f"character-table order ({len(table_ops)})."
        )
    mol_info = [_axis_angle(op) for op in mol_ops]
    table_info = [_axis_angle(op) for op in table_ops]

    def assign(U: np.ndarray) -> list | None:
        indices = []
        for op in mol_ops:
            transformed = U @ op @ U.T
            distances = [np.abs(transformed - t).max() for t in table_ops]
            index = int(np.argmin(distances))
            if distances[index] > 1e-2 or index in indices:
                return None
            indices.append(index)
        return indices

    # sort candidate primary ops: proper rotations of highest order first
    axed = [i for i, info in enumerate(mol_info) if info[2] is not None]
    if not axed:  # C1 or Ci
        indices = assign(np.eye(3))
        if indices is None:
            raise SystemExit("ERROR: could not match the trivial point group.")
        return np.eye(3), indices
    axed.sort(key=lambda i: (mol_info[i][0] < 0, mol_info[i][1]))
    a = axed[0]
    v1 = mol_info[a][2]
    secondary = [
        i for i in axed
        if abs(float(mol_info[i][2] @ v1)) < 1.0 - 1e-6
    ]
    secondary.sort(key=lambda i: (mol_info[i][0] < 0, mol_info[i][1]))

    candidates_a = [
        j for j, info in enumerate(table_info)
        if info[2] is not None and _signature(info) == _signature(mol_info[a])
    ]
    for j in candidates_a:
        for sign1 in (1.0, -1.0):
            w1 = sign1 * table_info[j][2]
            if not secondary:  # uniaxial group: any completion works
                U = _completed_frame(w1, None) @ _completed_frame(v1, None).T
                indices = assign(U)
                if indices is not None:
                    return U, indices
                continue
            b = secondary[0]
            v2 = mol_info[b][2]
            for k, info in enumerate(table_info):
                if info[2] is None or _signature(info) != _signature(mol_info[b]):
                    continue
                for sign2 in (1.0, -1.0):
                    w2 = sign2 * info[2]
                    if abs(abs(float(w2 @ w1)) - 1.0) < 1e-6:
                        continue
                    frame_w = _completed_frame(w1, w2)
                    frame_v = _completed_frame(v1, v2)
                    for handedness in (1.0, -1.0):
                        frame = frame_w.copy()
                        frame[:, 2] *= handedness
                        U = frame @ frame_v.T
                        indices = assign(U)
                        if indices is not None:
                            return U, indices
    raise SystemExit(
        "ERROR: could not align the molecular symmetry operations with the "
        "character table (try adjusting --tolerance)."
    )


# --------------------------------------------------- permutation representation


def get_permutation_matrices(operations, coordinates, tolerance: float):
    """Permutation matrix P(g) of every operation on the given sites
    (P[i, j] = 1 when g maps site j onto site i)."""
    matrices = []
    for rotation in operations:
        mapped = coordinates @ rotation.T
        matrix = np.zeros((len(coordinates), len(coordinates)))
        for j, position in enumerate(mapped):
            distances = np.linalg.norm(coordinates - position, axis=1)
            i = int(np.argmin(distances))
            if distances[i] > tolerance:
                raise SystemExit(
                    "ERROR: a symmetry operation does not map the selected "
                    "sites onto themselves (try adjusting --tolerance)."
                )
            matrix[i, j] = 1.0
        if not np.allclose(matrix.sum(axis=0), 1.0) or not np.allclose(matrix.sum(axis=1), 1.0):
            raise SystemExit("ERROR: site mapping is not a permutation.")
        matrices.append(matrix)
    return matrices


def _class_characters(values_per_operation, operation_classes, rotation_list):
    """Collapse per-operation characters onto the classes, checking constancy."""
    characters = {}
    for value, class_name in zip(values_per_operation, operation_classes):
        if class_name in characters and abs(characters[class_name] - value) > 1e-6:
            raise SystemExit(
                f"ERROR: inconsistent characters inside class {class_name}."
            )
        characters[class_name] = value
    return [characters[class_name] for class_name in rotation_list]


# ----------------------------------------------------------- SALC projection


def _rref_orthogonal(rows, tol=1e-8):
    """Numerically stable canonical basis: RREF followed by Gram-Schmidt."""
    # Gaussian elimination (RREF)
    matrix = np.array([np.asarray(row, dtype=float) for row in rows])
    n_rows, n_cols = matrix.shape
    r = 0
    for c in range(n_cols):
        if r >= n_rows:
            break
        pivot = int(np.argmax(np.abs(matrix[r:, c]))) + r
        if abs(matrix[pivot, c]) < tol:
            continue
        matrix[[r, pivot]] = matrix[[pivot, r]]
        matrix[r] = matrix[r] / matrix[r, c]
        for other in range(n_rows):
            if other != r and abs(matrix[other, c]) > tol:
                matrix[other] -= matrix[other, c] * matrix[r]
        r += 1
    matrix = matrix[:r]
    # Gram-Schmidt on the canonical rows
    basis = []
    for row in matrix:
        vector = row.copy()
        for prior in basis:
            vector -= (vector @ prior) * prior
        norm = np.linalg.norm(vector)
        if norm > tol:
            basis.append(vector / norm)
    return basis


def project_salcs(operations, operation_classes, permutations, azimuthal, character_table):
    """Explicit SALCs of the permutation x orbital representation per irrep."""
    rotation_list = list(character_table["rotation_list"])
    order = len(operations)
    dimension = permutations[0].shape[0] * (2 * azimuthal + 1)
    gamma = [
        np.kron(permutation, wigner_D_real(azimuthal, rotation))
        for rotation, permutation in zip(operations, permutations)
    ]
    salcs = {}
    for irrep_name, chars in character_table["character_table"].items():
        chi = {cls: c for cls, c in zip(rotation_list, np.atleast_1d(chars))}
        projector = np.zeros((dimension, dimension))
        for matrix, class_name in zip(gamma, operation_classes):
            projector += float(np.real(chi[class_name])) * matrix
        projector *= float(np.real(chi["E"])) / order
        eigenvalues, eigenvectors = np.linalg.eigh((projector + projector.T) / 2.0)
        columns = [eigenvectors[:, i] for i in range(dimension) if eigenvalues[i] > 0.5]
        if columns:
            salcs[irrep_name] = _rref_orthogonal(columns)
    return salcs


def _pretty_coefficients(vector):
    """Scale a SALC vector to small integers when possible."""
    # tolerances are set by the precision of typical XYZ geometries (~1e-4),
    # not by machine precision: the irrep decomposition itself is exact
    vector = np.asarray(vector, dtype=float)
    vector = vector / np.max(np.abs(vector))
    vector = np.where(np.abs(vector) < 1e-3, 0.0, vector)
    nonzero = vector[vector != 0.0]
    if len(nonzero) and nonzero[0] < 0.0:
        vector = -vector
    for k in range(1, 25):
        if np.allclose(k * vector, np.round(k * vector), atol=2e-3):
            integers = np.round(k * vector).astype(int)
            divisor = np.gcd.reduce(np.abs(integers[integers != 0]))
            return (integers // divisor).astype(float), True
    return vector, False


def format_salc(vector, term_labels) -> str:
    coefficients, is_integer = _pretty_coefficients(vector)
    parts = []
    for coefficient, label in zip(coefficients, term_labels):
        if coefficient == 0.0:
            continue
        magnitude = abs(coefficient)
        if is_integer and magnitude == 1.0:
            body = label
        elif is_integer:
            body = f"{int(magnitude)} {label}"
        else:
            body = f"{magnitude:.3f} {label}"
        parts.append(("- " if coefficient < 0 else "+ ") + body)
    text = " ".join(parts)
    return text[2:] if text.startswith("+ ") else text.replace("- ", "-", 1) if text.startswith("- ") else text


# ------------------------------------------------------------- visualization


class _MoleculeBoxAdapter:
    """Presents the molecule-in-a-box to the crystalline SALC viewer."""

    def __init__(self, cell):
        self.primitive_cell = cell

    def get_supercell_size(self, _kpoint):
        return (1, 1, 1)


def write_molecule_visualization(
    output_path: str,
    molecule,
    alignment: np.ndarray,
    align: bool,
    site_indices: list[int],
    azimuthal: int,
    salcs: dict,
    character_table: dict,
    title: str,
    info: dict,
    bonds: list[tuple[str, str, float]] | None,
) -> None:
    """Write the molecular SALCs as a standalone HTML viewer.

    Reuses the crystalline SALC viewer (crystod --visualize) by placing the
    molecule in a large vacuum box at the Gamma point (all Bloch phases 1);
    the box edges are hidden and the compass shows the Cartesian x/y/z axes.
    """
    from phonopy.structure.atoms import PhonopyAtoms

    from .visualize_basis import write_html_visualization

    positions = np.array([site.coords for site in molecule])
    if align:
        positions = positions @ alignment.T
    span = positions.max(axis=0) - positions.min(axis=0)
    box = float(max(span.max() + 10.0, 12.0))
    centered = positions - (positions.max(axis=0) + positions.min(axis=0)) / 2.0
    cell = PhonopyAtoms(
        symbols=[site.specie.symbol for site in molecule],
        positions=centered + box / 2.0,
        cell=np.diag([box, box, box]),
    )

    irrep_names = [name for name in character_table["character_table"] if name in salcs]
    basis_spaces = [np.array(salcs[name], dtype=complex) for name in irrep_names]
    write_html_visualization(
        output_path=output_path,
        orbitals=_MoleculeBoxAdapter(cell),
        basis_spaces=basis_spaces,
        basis_labels=irrep_names,
        element_indices=list(site_indices),
        l=azimuthal,
        kpoint=[0.0, 0.0, 0.0],
        title=title,
        info=info,
        bonds=bonds,
        draw_cell=False,
        axis_names="xyz",
    )


# -------------------------------------------------------------------- report


def _hm_symbol(schoenflies: str) -> str | None:
    return SCHOENFLIES_TO_HM.get(schoenflies)


def _class_summary(character_table: dict) -> str:
    multiplicities = [
        np.asarray(character_table["mapping_table"][name]).shape[0]
        for name in character_table["rotation_list"]
    ]
    return ", ".join(
        f"{multiplicity}{name}" if multiplicity > 1 else name
        for name, multiplicity in zip(character_table["rotation_list"], multiplicities)
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.symmetry and (args.element is None or args.orbital is None):
        parser.error("either use --symmetry, or give both --element and --orbital.")

    molecule = load_molecule(args.xyz)
    schoenflies, operations = get_symmetry(molecule, args.tolerance)
    hm = _hm_symbol(schoenflies)

    formula = molecule.composition.reduced_formula
    print("\n* Molecule *")
    print(f"{args.xyz} ({formula}, {len(molecule)} atoms)")
    print("\n* Point group *")
    if hm is not None:
        print(f"{schoenflies} (Hermann-Mauguin: {hm})")
    elif "*" in schoenflies:
        print(f"{schoenflies} (linear molecule; continuous non-crystallographic group)")
    else:
        print(f"{schoenflies} (non-crystallographic point group)")

    if args.symmetry:
        if hm is not None:
            character_table = get_character_table(hm)
            _match_operations(operations, *_table_operations_cartesian(character_table))
            print(f"\n* Symmetry operations ({len(operations)}) *")
            print(_class_summary(character_table))
        print()
        return

    if hm is None:
        raise SystemExit(
            "ERROR: the SALC analysis supports the 32 crystallographic point "
            "groups; this molecule's group is "
            f"{schoenflies}. For linear molecules, analyze a finite subgroup "
            "instead (e.g. mmm for D*h) with crystod-group --decompose."
        )

    orbital = args.orbital.strip().lower()
    if orbital not in ORBITAL_AZIMUTHAL_NUMBER:
        raise SystemExit(
            f"ERROR: orbital '{args.orbital}' is not supported. "
            f"Choose from: {', '.join(ORBITAL_AZIMUTHAL_NUMBER)}"
        )
    azimuthal = ORBITAL_AZIMUTHAL_NUMBER[orbital]
    if azimuthal not in ORBITAL_LABELS:
        raise SystemExit("ERROR: explicit SALCs are supported for s, p, d, and f orbitals.")

    element = args.element.strip()
    site_indices = [
        i for i, site in enumerate(molecule) if site.specie.symbol == element
    ]
    if not site_indices:
        available = ", ".join(sorted({site.specie.symbol for site in molecule}))
        raise SystemExit(
            f'ERROR: element "{element}" is not in the molecule (available: {available}).'
        )
    coordinates = np.array([molecule[i].coords for i in site_indices])
    site_labels = [f"{element}{n + 1}" for n in range(len(site_indices))]

    character_table = get_character_table(hm)
    table_ops, table_classes = _table_operations_cartesian(character_table)
    alignment, matched_indices = _match_operations(operations, table_ops, table_classes)
    operation_classes = [table_classes[i] for i in matched_indices]
    # snap the operations onto the exact character-table matrices so that the
    # group closure (and hence the projector) is exact, independent of the
    # numerical precision of the input geometry
    if args.align:
        operations = [table_ops[i] for i in matched_indices]
        coordinates = coordinates @ alignment.T
    else:
        operations = [alignment.T @ table_ops[i] @ alignment for i in matched_indices]
    permutations = get_permutation_matrices(operations, coordinates, args.tolerance)

    frame_note = "standard point-group frame" if args.align else "center-of-mass frame"
    print(f"\n* Target sites ({element}, {len(site_indices)} sites; {frame_note}) *")
    for label, position in zip(site_labels, coordinates):
        print(f"{label}: ({position[0]: .6f}, {position[1]: .6f}, {position[2]: .6f})")

    if args.show_matrix:
        print("\n* Site-permutation matrices *")
        for class_name, permutation in zip(operation_classes, permutations):
            print(f"{class_name}:")
            print(np.asarray(permutation, dtype=int))

    rotation_list = list(character_table["rotation_list"])
    multiplicities = [
        np.asarray(character_table["mapping_table"][name]).shape[0]
        for name in rotation_list
    ]
    class_headers = [
        f"{multiplicity}{name}" if multiplicity > 1 else name
        for name, multiplicity in zip(rotation_list, multiplicities)
    ]

    perm_characters = _class_characters(
        [float(np.trace(p)) for p in permutations], operation_classes, rotation_list
    )
    orbital_characters = get_orbital_characters(orbital, character_table)
    orbital_row = [orbital_characters[name] for name in rotation_list]
    total = [p * o for p, o in zip(perm_characters, orbital_row)]

    width = max(len(header) for header in class_headers) + 2
    print(f"\n* Reducible representation ({element} sites x {orbital} orbital) *")
    print("class:".ljust(16) + "".join(header.rjust(width) for header in class_headers))
    print("chi(perm):".ljust(16) + "".join(f"{value:.0f}".rjust(width) for value in perm_characters))
    print(f"chi({orbital}):".ljust(16) + "".join(f"{value:.0f}".rjust(width) for value in orbital_row))
    print("chi(total):".ljust(16) + "".join(f"{value:.0f}".rjust(width) for value in total))

    results = decompose(total, character_table, multiplicities)
    decomposition = " + ".join(
        f"{count}({irrep})" for irrep, count in results.items() if count > 0
    )
    print("\n* Decomposition *")
    print(f"Gamma = {decomposition}")

    orbital_names = ORBITAL_LABELS[azimuthal]
    term_labels = [
        f"{orbital_name}({site_label})"
        for site_label in site_labels
        for orbital_name in orbital_names
    ]
    salcs = project_salcs(
        operations, operation_classes, permutations, azimuthal, character_table
    )
    axes_note = (
        "orbital axes = standard point-group axes"
        if args.align
        else "orbital axes = input Cartesian axes"
    )
    print(f"\n* SALCs ({axes_note}) *")
    for irrep_name in character_table["character_table"]:
        if irrep_name not in salcs:
            continue
        formatted = ", ".join(format_salc(vector, term_labels) for vector in salcs[irrep_name])
        print(f"{irrep_name}: [{formatted}]")

    if args.visualize:
        bonds = None
        if args.bond:
            try:
                bonds = [(el1, el2, float(cutoff)) for el1, el2, cutoff in args.bond]
            except ValueError:
                raise SystemExit("ERROR: --bond expects EL1 EL2 MAX, e.g. --bond N H 1.2.")
        stem = os.path.splitext(os.path.basename(args.xyz))[0]
        output_path = args.output or f"SALC_{stem}_{element}_{orbital}.html"
        info = {
            "formula": formula,
            "point_group": f"{schoenflies} ({hm})",
            "element_orbital": f"{element} {orbital}",
            "decomposition": f"Gamma = {decomposition}",
            "supercell": "",
            "real_coefficient": True,
        }
        write_molecule_visualization(
            output_path,
            molecule,
            alignment,
            args.align,
            site_indices,
            azimuthal,
            salcs,
            character_table,
            f"Molecular SALC: {formula} {element} {orbital}",
            info,
            bonds,
        )
        print(f"\nSALC viewer written to {output_path}")
    print()


if __name__ == "__main__":
    main()
