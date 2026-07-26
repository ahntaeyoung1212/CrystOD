"""
Symmetry-adapted crystal-orbital (SALC) basis construction and visualization.

Builds the reducible representation kron(permutation, Wigner-D_real) for the
selected element/orbital at a k point, projects it onto the little-group
irreps, prints the SALC coefficients, and optionally writes a standalone
interactive 3D HTML visualization of the orbital arrangement.
"""

from __future__ import annotations

import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, RawDescriptionHelpFormatter, RawTextHelpFormatter
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .operations import wigner_D_real
from .runtime_compat import get_character, get_chemical_symbols
from .spglib_compat import ensure_spglib_compat

ensure_spglib_compat()

from phonopy.interface.calculator import read_crystal_structure
from spgrep.core import get_spacegroup_irreps_from_primitive_symmetry
from spgrep.representation import project_to_irrep

from .crystal_orbital_spgrep import format_kpoint, sort_irrep_items
from .vibration_modes import SymmetryOnlyVibrations


class MyHelpFormatter(
    RawTextHelpFormatter,
    RawDescriptionHelpFormatter,
    ArgumentDefaultsHelpFormatter,
):
    pass


desc = """
Visualize symmetry-adapted crystal-orbital (SALC) basis functions.

# Command Examples:
crystod --visualize-basis --poscar 221_PPOSCAR_ScF3 --element F --orbital p --kpoint 0 0 0
crystod --visualize-basis --poscar 221_PPOSCAR_ScF3 --element F --orbital p --kpoint GM --output salc_F_p.html
"""

ORBITAL_L = {"s": 0, "p": 1, "d": 2, "f": 3}

ORBITAL_COMPONENT_NAMES = {
    0: ["s"],
    1: ["p_x", "p_y", "p_z"],
    2: ["d_xy", "d_yz", "d_z2", "d_xz", "d_x2-y2"],
    3: [
        "f_x(x2-3y2)",
        "f_y(3x2-y2)",
        "f_z(x2-y2)",
        "f_xyz",
        "f_xz2",
        "f_yz2",
        "f_z3",
    ],
}


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=desc, formatter_class=MyHelpFormatter)
    parser.add_argument("--poscar", default="POSCAR", help="POSCAR path.")
    parser.add_argument(
        "--element",
        required=True,
        help="Target element symbol (or 'all').",
    )
    parser.add_argument(
        "--orbital",
        required=True,
        choices=sorted(ORBITAL_L),
        help="Atomic orbital: s, p, d, or f.",
    )
    parser.add_argument(
        "--kpoint",
        nargs="+",
        required=True,
        help="Either a high-symmetry label such as GM/X/M/R or three primitive reciprocal coordinates.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Symmetry tolerance.",
    )
    parser.add_argument(
        "--mode-index",
        type=int,
        default=None,
        help="Only print/visualize the selected irrep-grouped SALC space (1-based).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output HTML path for the interactive 3D visualization "
        "(default: SALC_{element}_{orbital}_{kpoint}.html).",
    )
    parser.add_argument(
        "--bond",
        nargs=3,
        action="append",
        default=None,
        metavar=("EL1", "EL2", "MAX"),
        help="Draw bonds between EL1 and EL2 atoms up to MAX Angstroms, plus the "
        "VESTA-style coordination polyhedra around the EL1 atoms "
        "(repeatable), e.g. --bond Sc F 2.3.",
    )
    parser.add_argument(
        "--real-coefficient",
        action="store_true",
        help=(
            "Re-combine degenerate SALC components into real-coefficient form\n"
            "when the irrep space allows it (real-type irreps). The spanned\n"
            "space is unchanged; only the basis choice within it is rotated."
        ),
    )
    parser.add_argument(
        "--conventional",
        action="store_true",
        help="Display the SALC in the conventional cell instead of the primitive "
        "cell (primitive-to-conventional matrix from the detected centring, "
        "as in crystod-phonon --vector).",
    )
    return parser


def orbital_angular_values(l: int, unit_vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate the real orbital angular parts on unit vectors.

    Returns an array of shape (2l+1, n_points) ordered consistently with
    ``operations.complex_to_real_transform_orbital``.
    """
    x, y, z = unit_vectors[:, 0], unit_vectors[:, 1], unit_vectors[:, 2]
    r2 = x * x + y * y + z * z
    s3 = np.sqrt(3.0)
    if l == 0:
        return np.stack([np.ones_like(x)])
    if l == 1:
        return np.stack([x, y, z])
    if l == 2:
        return np.stack(
            [s3 * x * y, s3 * y * z, (3 * z * z - r2) / 2, s3 * x * z, s3 * (x * x - y * y) / 2]
        )
    if l == 3:
        c1 = np.sqrt(5.0 / 8.0)
        c2 = np.sqrt(15.0) / 2.0
        c3 = np.sqrt(15.0)
        c4 = np.sqrt(3.0 / 8.0)
        return np.stack(
            [
                c1 * x * (x * x - 3 * y * y),
                c1 * y * (3 * x * x - y * y),
                c2 * z * (x * x - y * y),
                c3 * x * y * z,
                c4 * x * (5 * z * z - r2),
                c4 * y * (5 * z * z - r2),
                z * (5 * z * z - 3 * r2) / 2,
            ]
        )
    raise ValueError(f"Unsupported azimuthal quantum number l={l}.")


class SymmetryAdaptedOrbitalBasis(SymmetryOnlyVibrations):
    """SALC basis construction for one element/orbital at a k point."""

    def get_element_indices(self, element: str) -> list[int]:
        symbols = get_chemical_symbols(self.primitive_cell)
        if element.lower() == "all":
            return list(range(len(symbols)))
        indices = [index for index, symbol in enumerate(symbols) if symbol == element]
        if not indices:
            raise ValueError(f"Element '{element}' is not in the inputed cell.")
        return indices

    def get_orbital_rep(self, kpoint: list[float], element: str, l: int):
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
        element_indices = self.get_element_indices(element)
        index_grid = np.ix_(element_indices, element_indices)
        wigner_matrices = [
            wigner_D_real(l, np.real(self.rotations_cartesian[index]))
            for index in mapping_little_group
        ]
        orbital_rep = np.array(
            [
                np.kron(permutation_matrix[index_grid], wigner_matrix)
                for permutation_matrix, wigner_matrix in zip(permutation_matrices, wigner_matrices)
            ],
            dtype=np.complex128,
        )
        return irreps, orbital_rep, mapping_little_group, element_indices

    def decompose_orbital_rep(self, irreps, orbital_rep, irrep_labels: list[str]) -> dict[str, float]:
        rep_characters = np.array([np.trace(matrix) for matrix in orbital_rep])
        multiplicities: dict[str, float] = {}
        for irrep, label in zip(irreps, irrep_labels):
            irrep_characters = np.array(get_character(irrep), dtype=complex)
            count = np.dot(rep_characters, np.conjugate(irrep_characters)) / len(irrep_characters)
            multiplicities[label] = float(np.round(count.real, 2))
        return multiplicities

    def get_orbital_basis(self, irreps, orbital_rep, irrep_labels: list[str]):
        basis_spaces: list[NDArray[np.complex128]] = []
        basis_labels: list[str] = []
        for irrep, irrep_label in zip(irreps, irrep_labels):
            projected_spaces = project_to_irrep(orbital_rep, irrep)
            basis_spaces.extend(projected_spaces)
            basis_labels.extend([irrep_label] * len(projected_spaces))
        return basis_spaces, basis_labels


def _phase_normalize(vector: NDArray[np.complex128], tol: float = 1e-8) -> NDArray[np.complex128]:
    """Remove the arbitrary global phase: make the largest coefficient real positive."""
    pivot = int(np.argmax(np.abs(vector)))
    magnitude = abs(vector[pivot])
    if magnitude < tol:
        return vector
    return vector * (vector[pivot].conjugate() / magnitude)


def realify_basis_space(
    space: NDArray[np.complex128],
    tol: float = 1e-6,
) -> tuple[NDArray[np.complex128], bool]:
    """Rotate a degenerate SALC space to real-coefficient basis vectors.

    A real basis exists whenever the projected space W is closed under complex
    conjugation (guaranteed for real-type irreps when k = -k mod G*). In that
    case Re(v) and Im(v) of every component v lie in W, so a real orthonormal
    basis of W is extracted from them by Gram-Schmidt. The span is unchanged;
    only the unitary basis choice within the irrep space is rotated.

    Returns (new_space, True) on success, or (space unchanged, False) when no
    real basis exists (complex-type irrep / conjugation leaves the space).
    """
    space = np.asarray(space, dtype=np.complex128)
    dimension = space.shape[0]

    # Orthonormal row basis of W and conjugation-closure check.
    _, singular_values, row_basis = np.linalg.svd(space, full_matrices=False)
    row_basis = row_basis[singular_values > tol]
    projector = row_basis.conj().T @ row_basis
    for vector in space:
        conjugated = vector.conj()
        if np.linalg.norm(conjugated - conjugated @ projector) > tol * max(
            1.0, np.linalg.norm(conjugated)
        ):
            return space, False

    # Real candidates in insertion order (keeps sparse, intuitive combinations).
    real_vectors: list[NDArray[np.float64]] = []
    for vector in space:
        vector = _phase_normalize(vector)
        for candidate in (np.real(vector), np.imag(vector)):
            candidate = candidate.astype(float).copy()
            for chosen in real_vectors:
                candidate -= np.dot(chosen, candidate) * chosen
            norm = np.linalg.norm(candidate)
            if norm > tol:
                real_vectors.append(candidate / norm)
            if len(real_vectors) == dimension:
                break
        if len(real_vectors) == dimension:
            break

    if len(real_vectors) != dimension:
        return space, False

    # Fix the sign convention: largest-magnitude coefficient positive.
    new_space = []
    for vector in real_vectors:
        pivot = int(np.argmax(np.abs(vector)))
        if vector[pivot] < 0:
            vector = -vector
        new_space.append(vector)
    return np.array(new_space, dtype=np.complex128), True


def _format_coefficient(value: complex, tol: float = 1e-6) -> str:
    real = 0.0 if abs(value.real) < tol else float(np.round(value.real, 4))
    imag = 0.0 if abs(value.imag) < tol else float(np.round(value.imag, 4))
    if imag == 0.0:
        return f"{real:+.4f}"
    return f"({real:+.4f}{imag:+.4f}j)"


def _print_salc_coefficients(
    basis_spaces: list[NDArray[np.complex128]],
    basis_labels: list[str],
    element: str,
    element_indices: list[int],
    l: int,
    mode_index: int | None,
) -> None:
    component_names = ORBITAL_COMPONENT_NAMES[l]
    n_components = len(component_names)
    print(" * SALC basis functions (irrep-grouped) *")
    for space_index, (space, label) in enumerate(zip(basis_spaces, basis_labels)):
        if mode_index is not None and space_index != mode_index:
            continue
        print(f" Mode Space {space_index + 1}: irrep = {label}, dimension = {space.shape[0]}")
        for component_index, vector in enumerate(space):
            print(f"   component {component_index + 1}:")
            for atom_slot, atom_index in enumerate(element_indices):
                coefficients = vector[atom_slot * n_components : (atom_slot + 1) * n_components]
                if np.max(np.abs(coefficients)) < 1e-6:
                    continue
                terms = ", ".join(
                    f"{name}: {_format_coefficient(value)}"
                    for name, value in zip(component_names, coefficients)
                    if abs(value) > 1e-6
                )
                print(f"     {element}{atom_index + 1} (atom {atom_index}): {terms}")
        print("")


# --------------------------------------------------------------------------
# 3D HTML visualization
# --------------------------------------------------------------------------

_FALLBACK_ATOM_COLORS = {
    "H": "#f0f0f0", "C": "#555555", "N": "#3050f8", "O": "#ff0d0d",
    "F": "#90e050", "Na": "#ab5cf2", "Mg": "#8aff00", "Al": "#bfa6a6",
    "Si": "#f0c8a0", "P": "#ff8000", "S": "#ffff30", "Cl": "#1ff01f",
    "K": "#8f40d4", "Ca": "#3dff00", "Sc": "#e6e6e6", "Ti": "#bfc2c7",
    "V": "#a6a6ab", "Cr": "#8a99c7", "Mn": "#9c7ac7", "Fe": "#e06633",
    "Co": "#f090a0", "Ni": "#50d050", "Cu": "#c88033", "Zn": "#7d80b0",
    "Sr": "#00ff00", "Ba": "#00c900", "O2": "#ff0d0d",
}


def _rgb_to_hex(rgb: list[int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


@lru_cache(maxsize=1)
def _load_atom_colors() -> dict[str, str]:
    color_path = Path(__file__).with_name("vesta_element_rgb.json")
    try:
        payload = json.loads(color_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(_FALLBACK_ATOM_COLORS)

    colors = dict(_FALLBACK_ATOM_COLORS)
    for symbol, rgb in payload.items():
        if symbol.startswith("_") or rgb is None:
            continue
        if (
            isinstance(rgb, list)
            and len(rgb) == 3
            and all(isinstance(channel, int) and 0 <= channel <= 255 for channel in rgb)
        ):
            colors[symbol] = _rgb_to_hex(rgb)
    return colors


def _lattice_edge_traces(lattice: NDArray[np.float64], supercell_size: tuple[int, int, int]):
    n1, n2, n3 = supercell_size
    cell = np.array(
        [lattice[0] * n1, lattice[1] * n2, lattice[2] * n3],
        dtype=float,
    )
    corners = [
        np.zeros(3), cell[0], cell[1], cell[2],
        cell[0] + cell[1], cell[0] + cell[2], cell[1] + cell[2],
        cell[0] + cell[1] + cell[2],
    ]
    edges = [
        (0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4), (2, 6),
        (3, 5), (3, 6), (4, 7), (5, 7), (6, 7),
    ]
    xs, ys, zs = [], [], []
    for start, end in edges:
        xs.extend([corners[start][0], corners[end][0], None])
        ys.extend([corners[start][1], corners[end][1], None])
        zs.extend([corners[start][2], corners[end][2], None])
    return {
        "type": "scatter3d",
        "mode": "lines",
        "x": xs, "y": ys, "z": zs,
        "line": {"color": "#888888", "width": 2},
        "hoverinfo": "skip",
        "showlegend": False,
    }


def _atom_traces(
    positions: NDArray[np.float64],
    symbols: list[str],
) -> list[dict]:
    atom_colors = _load_atom_colors()
    traces = []
    for symbol in sorted(set(symbols)):
        indices = [index for index, s in enumerate(symbols) if s == symbol]
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "x": positions[indices, 0].tolist(),
                "y": positions[indices, 1].tolist(),
                "z": positions[indices, 2].tolist(),
                "marker": {
                    "size": 6,
                    "color": atom_colors.get(symbol, "#cccccc"),
                    "line": {"color": "#333333", "width": 1},
                },
                "name": symbol,
                "text": [f"{symbol}{index}" for index in indices],
                "hoverinfo": "text",
            }
        )
    return traces


def _orbital_surface(
    center: NDArray[np.float64],
    coefficients: NDArray[np.complex128],
    l: int,
    scale: float,
    n_theta: int = 22,
    n_phi: int = 44,
) -> dict | None:
    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    unit = np.stack(
        [
            np.sin(theta_grid) * np.cos(phi_grid),
            np.sin(theta_grid) * np.sin(phi_grid),
            np.cos(theta_grid),
        ],
        axis=-1,
    ).reshape(-1, 3)
    angular = orbital_angular_values(l, unit)
    values = np.real(np.tensordot(coefficients, angular, axes=(0, 0)))
    max_value = float(np.max(np.abs(values)))
    if max_value < 1e-8:
        return None
    radius = scale * np.abs(values) / max_value
    points = unit * radius[:, None] + center[None, :]
    shape = theta_grid.shape
    # sign-only coloring (VESTA style: + yellow, - blue) and 3-decimal
    # coordinates keep the standalone HTML small.
    sign = (values >= 0).astype(int)
    return {
        "type": "surface",
        "x": np.round(points[:, 0], 3).reshape(shape).tolist(),
        "y": np.round(points[:, 1], 3).reshape(shape).tolist(),
        "z": np.round(points[:, 2], 3).reshape(shape).tolist(),
        "surfacecolor": sign.reshape(shape).tolist(),
        "cmin": 0,
        "cmax": 1,
        "colorscale": [[0.0, "#26c6da"], [1.0, "#ffeb3b"]],
        "showscale": False,
        # opaque by default: WebGL depth testing then resolves front/back
        # correctly (translucent surfaces cannot be depth-sorted by plotly)
        "opacity": 1.0,
        "lighting": {
            "ambient": 0.55,
            "diffuse": 0.75,
            "specular": 0.4,
            "roughness": 0.5,
            "fresnel": 0.1,
        },
        "hoverinfo": "skip",
        "showlegend": False,
    }


def _axis_arrow_set(
    lattice: NDArray[np.float64],
    names: list[str],
    colors: tuple[str, str, str],
    arrow_length: float,
    line_width: int,
    cone_size: float,
    label_radius: float,
    label_size: int,
) -> list[dict]:
    """One set of three compass arrows (lines + cones + one text trace)."""
    traces: list[dict] = []
    label_positions = []
    label_colors = []
    for vector, color in zip(lattice, colors):
        direction = vector / np.linalg.norm(vector)
        tip = direction * arrow_length
        traces.append(
            {
                "type": "scatter3d",
                "scene": "scene2",
                "mode": "lines",
                "x": [0, round(float(tip[0]), 3)],
                "y": [0, round(float(tip[1]), 3)],
                "z": [0, round(float(tip[2]), 3)],
                "line": {"color": color, "width": line_width},
                "hoverinfo": "skip",
                "showlegend": False,
            }
        )
        traces.append(
            {
                "type": "cone",
                "scene": "scene2",
                "x": [round(float(tip[0]), 3)],
                "y": [round(float(tip[1]), 3)],
                "z": [round(float(tip[2]), 3)],
                "u": [round(float(direction[0]), 3)],
                "v": [round(float(direction[1]), 3)],
                "w": [round(float(direction[2]), 3)],
                "anchor": "tail",
                "sizemode": "absolute",
                "sizeref": cone_size,
                "colorscale": [[0.0, color], [1.0, color]],
                "showscale": False,
                "hoverinfo": "skip",
                "showlegend": False,
            }
        )
        label_positions.append(direction * label_radius)
        label_colors.append(color)
    traces.append(
        {
            "type": "scatter3d",
            "scene": "scene2",
            "mode": "text",
            "x": [round(float(p[0]), 3) for p in label_positions],
            "y": [round(float(p[1]), 3) for p in label_positions],
            "z": [round(float(p[2]), 3) for p in label_positions],
            "text": names,
            "textfont": {"size": label_size, "color": label_colors},
            "hoverinfo": "skip",
            "showlegend": False,
        }
    )
    return traces


def _axis_traces(
    lattice: NDArray[np.float64],
    axis_names: str = "abc",
    conventional_lattice: NDArray[np.float64] | None = None,
) -> list[dict]:
    """VESTA-style a/b/c compass (a red, b green, c blue).

    The compass lives in a small second scene pinned to the lower-left corner
    of the viewport; its camera is synchronized to the main scene by the page
    JavaScript, so it always shows the current orientation like VESTA.

    `lattice` holds the primitive lattice vectors. With --conventional,
    `conventional_lattice` is given as well and BOTH sets are drawn: the
    primitive vectors as shorter pastel arrows labeled a_prim/b_prim/c_prim,
    and the conventional vectors (the displayed cell) as full-color arrows
    labeled a_conv/b_conv/c_conv (the qualifier is set as a true subscript --
    plotly renders the <sub> tag in text traces)."""
    strong = ("#d62728", "#2ca02c", "#1f77b4")
    if conventional_lattice is None:
        return _axis_arrow_set(
            lattice, list(axis_names), strong,
            arrow_length=1.0, line_width=8, cone_size=0.3,
            label_radius=1.45, label_size=16,
        )
    # both label rings sit well away from the origin: axes that point close to
    # the viewing direction project into a small circle around the centre
    # (fcc down [111]: every primitive vector is a face diagonal tilted only
    # 35 deg off the camera axis), and labels crowded there overlap each other
    pastel = ("#ff9896", "#98df8a", "#aec7e8")
    traces = _axis_arrow_set(
        lattice, [f"{name}<sub>prim</sub>" for name in axis_names], pastel,
        arrow_length=0.62, line_width=5, cone_size=0.2,
        label_radius=1.35, label_size=11,
    )
    traces.extend(
        _axis_arrow_set(
            conventional_lattice,
            [f"{name}<sub>conv</sub>" for name in axis_names], strong,
            arrow_length=1.3, line_width=8, cone_size=0.3,
            label_radius=1.95, label_size=15,
        )
    )
    return traces


def write_html_visualization(
    output_path: str,
    orbitals: SymmetryAdaptedOrbitalBasis,
    basis_spaces: list[NDArray[np.complex128]],
    basis_labels: list[str],
    element_indices: list[int],
    l: int,
    kpoint: list[float],
    title: str,
    mode_index: int | None = None,
    info: dict | None = None,
    bonds: list[tuple[str, str, float]] | None = None,
    conventional: bool = False,
    draw_cell: bool = True,
    axis_names: str = "abc",
) -> None:
    """Write the standalone SALC viewer page.

    The page layout (left control/mode sidebar + central 3D viewport) is
    modeled after the phonon website by Henrique Miranda
    (https://henriquemiranda.github.io/phononwebsite/, BSD-3-Clause); the 3D
    rendering itself uses plotly.

    ``bonds`` is a list of (element_1, element_2, max_length_A): bonds within
    the cutoff are drawn as in VESTA, and the coordination polyhedra around
    the element_1 atoms are rendered as translucent convex hulls.
    """
    primitive = orbitals.primitive_cell
    lattice = np.array(primitive.cell, dtype=float)
    frac_positions = np.array(primitive.scaled_positions, dtype=float)
    symbols = get_chemical_symbols(primitive)

    # Display cell: rows of cell_matrix are the display-cell lattice vectors
    # in the primitive basis — a diagonal (commensurate) supercell of the
    # primitive cell by default, or the conventional cell (times commensurate
    # multiples) with --conventional, as in crystod-phonon --vector.
    if conventional:
        from .phonon_vector import get_commensurate_supercell_matrix, get_conventional_matrix

        centring = orbitals.spglib_dataset["international"][0]
        base_matrix = get_conventional_matrix(centring)
        cell_matrix = np.array(get_commensurate_supercell_matrix(kpoint, base_matrix), dtype=int)
        multiples = np.rint(
            np.diag(cell_matrix @ np.linalg.inv(np.array(base_matrix, dtype=float)))
        ).astype(int)
        cell_description = (
            f"conventional ({centring} centring), "
            f"{multiples[0]} x {multiples[1]} x {multiples[2]} cells"
        )
    else:
        n1, n2, n3 = orbitals.get_supercell_size(kpoint)
        cell_matrix = np.diag([n1, n2, n3]).astype(int)
        cell_description = f"primitive, {n1} x {n2} x {n3} cells"

    display_lattice = np.array(cell_matrix, dtype=float) @ lattice
    inverse_cell = np.linalg.inv(np.array(cell_matrix, dtype=float))

    # Atoms displayed in the display cell, with VESTA-style boundary
    # completion: an atom with fractional coordinate 0 along a display-cell
    # axis is also drawn at 1 (carrying the Bloch phase of its full primitive
    # translation), so that bonds and coordination polyhedra at the cell
    # boundary are not cut off.
    boundary_eps = 1e-6
    corner_shifts = [np.zeros(3)]
    for axis in range(3):
        corner_shifts = corner_shifts + [shift + cell_matrix[axis] for shift in corner_shifts]
    corner_array = np.array(corner_shifts, dtype=float)
    t_low = np.floor(corner_array.min(axis=0)).astype(int) - 1
    t_high = np.ceil(corner_array.max(axis=0)).astype(int) + 1

    atom_entries = []  # (atom_index, translation in primitive-cell units)
    for t1 in range(t_low[0], t_high[0] + 1):
        for t2 in range(t_low[1], t_high[1] + 1):
            for t3 in range(t_low[2], t_high[2] + 1):
                base_translation = np.array([t1, t2, t3], dtype=float)
                for atom_index in range(len(frac_positions)):
                    frac_cell = (frac_positions[atom_index] + base_translation) @ inverse_cell
                    if np.any(frac_cell < -boundary_eps) or np.any(frac_cell >= 1 - boundary_eps):
                        continue
                    duplicate_axes = [
                        axis for axis in range(3) if abs(frac_cell[axis]) < boundary_eps
                    ]
                    combos = [()]
                    for axis in duplicate_axes:
                        combos = combos + [combo + (axis,) for combo in combos]
                    for combo in combos:
                        translation = base_translation.copy()
                        for axis in combo:
                            translation = translation + cell_matrix[axis]
                        atom_entries.append((atom_index, translation))

    all_positions = []
    all_symbols = []
    target_slots = []  # (atom_slot_in_element_list, cartesian position, phase)
    for atom_index, translation in atom_entries:
        position = (frac_positions[atom_index] + translation) @ lattice
        all_positions.append(position)
        all_symbols.append(symbols[atom_index])
        if atom_index in element_indices:
            phase = np.exp(2j * np.pi * np.dot(kpoint, translation))
            target_slots.append((element_indices.index(atom_index), position, phase))
    all_positions = np.array(all_positions)

    nearest = np.inf
    for slot_a in range(len(all_positions)):
        for slot_b in range(slot_a + 1, len(all_positions)):
            separation = float(np.linalg.norm(all_positions[slot_a] - all_positions[slot_b]))
            if separation > 1e-3:
                nearest = min(nearest, separation)
    lobe_scale = 0.45 * nearest if np.isfinite(nearest) else 1.0

    # VESTA-style bonds and coordination polyhedra. As in VESTA, neighbors are
    # searched in the periodic images of the displayed supercell, and image
    # atoms that participate in a bond are added to the display (with lobes
    # when they belong to the target element), so that coordination polyhedra
    # at the cell boundary are complete.
    atom_colors = _load_atom_colors()
    bond_traces = []
    polyhedra_traces = []
    if bonds:
        supercell_lattice = display_lattice
        image_offsets = [
            np.array([o1, o2, o3], dtype=float)
            for o1 in (-1, 0, 1)
            for o2 in (-1, 0, 1)
            for o3 in (-1, 0, 1)
        ]
        n_base = len(all_positions)
        base_positions = np.array(all_positions)
        extra_atoms: dict[tuple, np.ndarray] = {}  # (entry index, offset) -> position

        for el1, el2, max_length in bonds:
            segments: list = []
            centers: dict[int, list] = {}
            for a in range(n_base):
                if all_symbols[a] != el1:
                    continue
                for b in range(n_base):
                    if all_symbols[b] != el2:
                        continue
                    for offset in image_offsets:
                        position_b = base_positions[b] + offset @ supercell_lattice
                        separation = float(np.linalg.norm(base_positions[a] - position_b))
                        if not (1e-3 < separation <= max_length):
                            continue
                        centers.setdefault(a, []).append(position_b)
                        segments.append((base_positions[a], position_b))
                        if np.any(offset):
                            extra_atoms[(b, tuple(int(o) for o in offset))] = position_b

            if segments:
                xs: list = []
                ys: list = []
                zs: list = []
                for position_a, position_b in segments:
                    xs.extend([float(position_a[0]), float(position_b[0]), None])
                    ys.extend([float(position_a[1]), float(position_b[1]), None])
                    zs.extend([float(position_a[2]), float(position_b[2]), None])
                bond_traces.append(
                    {
                        "type": "scatter3d",
                        "mode": "lines",
                        "x": xs,
                        "y": ys,
                        "z": zs,
                        "line": {"color": "#7a7a7a", "width": 5},
                        "name": f"{el1}-{el2} bonds",
                        "hoverinfo": "skip",
                    }
                )
            show_legend = True
            for neighbor_positions in centers.values():
                if len(neighbor_positions) < 4:
                    continue  # a convex hull needs at least four ligands
                points = np.array(neighbor_positions, dtype=float)
                polyhedra_traces.append(
                    {
                        "type": "mesh3d",
                        "x": points[:, 0].tolist(),
                        "y": points[:, 1].tolist(),
                        "z": points[:, 2].tolist(),
                        "alphahull": 0,
                        "opacity": 0.35,
                        "color": atom_colors.get(el1, "#cccccc"),
                        "flatshading": True,
                        "name": f"{el1} polyhedra",
                        "legendgroup": f"poly-{el1}",
                        "showlegend": show_legend,
                        "hoverinfo": "skip",
                    }
                )
                show_legend = False

        # add the bonded image atoms to the display (deduplicated by position)
        all_positions_list = [np.array(position) for position in all_positions]
        for (entry_slot, offset), position in extra_atoms.items():
            if any(np.linalg.norm(position - existing) < 1e-6 for existing in all_positions_list):
                continue
            all_positions_list.append(position)
            atom_index, translation = atom_entries[entry_slot]
            all_symbols.append(symbols[atom_index])
            if atom_index in element_indices:
                image_translation = translation + np.array(offset, dtype=float) @ np.array(
                    cell_matrix, dtype=float
                )
                phase = np.exp(2j * np.pi * np.dot(kpoint, image_translation))
                target_slots.append((element_indices.index(atom_index), position, phase))
        all_positions = np.array(all_positions_list)

    atom_traces = _atom_traces(all_positions, all_symbols)
    static_traces = [_lattice_edge_traces(display_lattice, (1, 1, 1))] if draw_cell else []
    static_traces.extend(
        _axis_traces(
            lattice, axis_names,
            conventional_lattice=display_lattice if conventional else None,
        )
    )
    cell_end = len(static_traces)
    static_traces.extend(atom_traces)
    atoms_end = len(static_traces)
    static_traces.extend(bond_traces)
    bonds_end = len(static_traces)
    static_traces.extend(polyhedra_traces)
    n_static = len(static_traces)

    component_names = ORBITAL_COMPONENT_NAMES[l]
    n_components = len(component_names)

    dynamic_traces = []
    dynamic_centers = []  # lobe centers, for the view-depth opacity fade
    mode_specs = []  # dicts describing each selectable (mode space, component)
    for space_index, (space, label) in enumerate(zip(basis_spaces, basis_labels)):
        if mode_index is not None and space_index != mode_index:
            continue
        for component_index, vector in enumerate(space):
            start = len(dynamic_traces)
            for atom_slot, position, phase in target_slots:
                coefficients = (
                    vector[atom_slot * n_components : (atom_slot + 1) * n_components] * phase
                )
                surface = _orbital_surface(position, coefficients, l, lobe_scale)
                if surface is not None:
                    dynamic_traces.append(surface)
                    dynamic_centers.append([round(float(value), 2) for value in position])
            mode_specs.append(
                {
                    "label": f"Mode {space_index + 1} [{label}] comp {component_index + 1}",
                    "space": space_index + 1,
                    "irrep": label,
                    "component": component_index + 1,
                    "start": start,
                    "count": len(dynamic_traces) - start,
                }
            )

    all_traces = static_traces + dynamic_traces
    if mode_specs:
        first = mode_specs[0]
        for offset, trace in enumerate(all_traces):
            dynamic_offset = offset - n_static
            trace["visible"] = (
                offset < n_static
                or first["start"] <= dynamic_offset < first["start"] + first["count"]
            )

    layout = {
        "scene": {
            # full plot area: without an explicit domain, plotly grid-splits
            # the width between this scene and the compass scene2, squeezing
            # the structure into the left half of the viewport
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "aspectmode": "data",
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "bgcolor": "#ffffff",
            # initial zoom: eye 2.5 (the old default) leaves the structure
            # too small; 2.5 / 1.5 shows it 1.5x larger while still keeping
            # the orbital lobes inside the viewport
            "camera": {"eye": {"x": 1.6667, "y": 1.6667, "z": 1.6667}},
        },
        # small camera-synced a/b/c compass in the lower-left corner
        # (--conventional draws six arrows — primitive AND conventional
        # vectors — so the compass gets a larger corner box there)
        "scene2": {
            "domain": (
                {"x": [0.02, 0.26], "y": [0.02, 0.34]}
                if conventional
                else {"x": [0.03, 0.19], "y": [0.02, 0.22]}
            ),
            "aspectmode": "cube",
            "xaxis": {"visible": False, "range": [-2.2, 2.2]},
            "yaxis": {"visible": False, "range": [-2.2, 2.2]},
            "zaxis": {"visible": False, "range": [-2.2, 2.2]},
            "bgcolor": "rgba(0,0,0,0)",
            "dragmode": False,
        },
        "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
        "paper_bgcolor": "#ffffff",
        "legend": {"x": 0.99, "y": 0.95, "xanchor": "right"},
        "showlegend": True,
    }

    info = dict(info or {})
    info.setdefault("supercell", cell_description)
    bond_summary = "; ".join(
        f"{el1}&ndash;{el2} &le; {max_length:g} &Aring;" for el1, el2, max_length in bonds or []
    )

    info_rows = "".join(
        f"<tr><td>{name}</td><td>{value}</td></tr>"
        for name, value in (
            ("Compound", info.get("formula", "")),
            ("Space group", info.get("space_group", "")),
            ("Point group", info.get("point_group", "")),
            ("Orbitals", info.get("element_orbital", "")),
            ("k point", info.get("kpoint", "")),
            ("Display cell", info.get("supercell", "")),
            ("Bonds", bond_summary),
            ("Basis form", "real coefficients" if info.get("real_coefficient") else "complex (Bloch) coefficients"),
        )
        if value
    )
    bond_controls = ""
    if bond_traces or polyhedra_traces:
        bond_controls = (
            "  <div class=\"control\"><label><input type=\"checkbox\" id=\"show-bonds\" checked\n"
            "           onchange=\"applyVisibility()\"/> show bonds</label></div>\n"
            "  <div class=\"control\"><label><input type=\"checkbox\" id=\"show-poly\" checked\n"
            "           onchange=\"applyVisibility()\"/> show polyhedra</label></div>\n"
        )
    mode_rows = "".join(
        f'<tr class="mode-row" data-index="{row_index}" onclick="setMode({row_index})">'
        f'<td>{spec["space"]}</td><td class="irrep">{spec["irrep"]}</td><td>{spec["component"]}</td></tr>'
        for row_index, spec in enumerate(mode_specs)
    )

    html = (
        "<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"utf-8\"/>\n"
        f"<title>{title}</title>\n"
        "<script src=\"https://cdn.plot.ly/plotly-2.32.0.min.js\"></script>\n"
        "<style>\n"
        "  * { box-sizing: border-box; }\n"
        "  body { margin: 0; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;\n"
        "         color: #333; height: 100vh; display: flex; flex-direction: column; }\n"
        "  #topbar { background: #2c3e50; color: #ecf0f1; padding: 8px 16px;\n"
        "            display: flex; justify-content: space-between; align-items: baseline; }\n"
        "  #topbar .brand { font-size: 18px; font-weight: bold; }\n"
        "  #topbar .brand small { font-weight: normal; opacity: 0.8; margin-left: 8px; }\n"
        "  #topbar .page-title { font-size: 13px; opacity: 0.9; }\n"
        "  #container { flex: 1; display: flex; min-height: 0; }\n"
        "  #sidebar { width: 320px; min-width: 320px; overflow-y: auto; background: #f7f7f7;\n"
        "             border-right: 1px solid #ddd; padding: 12px 16px; font-size: 13px; }\n"
        "  #sidebar h2 { font-size: 13px; text-transform: uppercase; letter-spacing: 0.05em;\n"
        "                color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 3px;\n"
        "                margin: 18px 0 8px; }\n"
        "  #sidebar h2:first-child { margin-top: 4px; }\n"
        "  table.info { width: 100%; border-collapse: collapse; }\n"
        "  table.info td { padding: 2px 4px; vertical-align: top; }\n"
        "  table.info td:first-child { color: #777; width: 40%; }\n"
        "  .mono { font-family: Menlo, Consolas, monospace; font-size: 12px;\n"
        "          background: #fff; border: 1px solid #e0e0e0; padding: 6px; border-radius: 3px; }\n"
        "  table#mode-table { width: 100%; border-collapse: collapse; background: #fff;\n"
        "                     border: 1px solid #e0e0e0; }\n"
        "  table#mode-table th { background: #2c3e50; color: #fff; font-weight: normal;\n"
        "                        padding: 4px 6px; font-size: 12px; text-align: left; }\n"
        "  table#mode-table td { padding: 4px 6px; border-top: 1px solid #eee; cursor: pointer; }\n"
        "  table#mode-table td.irrep { font-family: Menlo, Consolas, monospace; }\n"
        "  tr.mode-row:hover { background: #eaf1f8; }\n"
        "  tr.mode-row.active { background: #d5e5f5; font-weight: bold; }\n"
        "  .control { margin: 6px 0; display: flex; align-items: center; gap: 8px; }\n"
        "  .control label { flex: 1; }\n"
        "  .credit { margin-top: 24px; padding-top: 8px; border-top: 1px solid #ddd;\n"
        "            font-size: 11px; color: #888; }\n"
        "  .credit a { color: #2c6aa0; }\n"
        "  #main { flex: 1; display: flex; flex-direction: column; min-width: 0; }\n"
        "  #mode-title { padding: 8px 14px; font-size: 15px; border-bottom: 1px solid #eee;\n"
        "                background: #fff; }\n"
        "  #mode-title .irrep { font-family: Menlo, Consolas, monospace; color: #2c6aa0; }\n"
        "  #plot { flex: 1; min-height: 0; }\n"
        "</style>\n</head>\n<body>\n"
        "<div id=\"topbar\">\n"
        "  <div class=\"brand\">CrystOD<small>Symmetry-Adapted Linear Combination (SALC) viewer</small></div>\n"
        f"  <div class=\"page-title\">{title}</div>\n"
        "</div>\n"
        "<div id=\"container\">\n"
        "<div id=\"sidebar\">\n"
        "  <h2>Structure</h2>\n"
        f"  <table class=\"info\">{info_rows}</table>\n"
        "  <h2>Irreps of SALC</h2>\n"
        f"  <div class=\"mono\">{info.get('decomposition', '')}</div>\n"
        "  <h2>SALC basis (click to show)</h2>\n"
        "  <table id=\"mode-table\">\n"
        "    <tr><th>Mode</th><th>Irrep</th><th>Comp.</th></tr>\n"
        f"    {mode_rows}\n"
        "  </table>\n"
        "  <h2>Display</h2>\n"
        "  <div class=\"control\"><label>Lobe opacity</label>\n"
        "    <input type=\"range\" id=\"opacity\" min=\"0.1\" max=\"1.0\" step=\"0.05\" value=\"1.0\"\n"
        "           oninput=\"setOpacity(this.value)\"/></div>\n"
        "  <div class=\"control\"><label><input type=\"checkbox\" id=\"show-cell\" checked\n"
        f"           onchange=\"applyVisibility()\"/> {'show cell edges &amp; ' + axis_names + ' axes' if draw_cell else 'show ' + axis_names + ' axes'}</label></div>\n"
        "  <div class=\"control\"><label><input type=\"checkbox\" id=\"show-atoms\" checked\n"
        "           onchange=\"applyVisibility()\"/> show atoms</label></div>\n"
        + bond_controls +
        "  <div class=\"credit\">Viewer layout inspired by the\n"
        "    <a href=\"https://henriquemiranda.github.io/phononwebsite/\" target=\"_blank\">phonon website</a>\n"
        "    by Henrique Miranda (BSD-3-Clause). 3D rendering by plotly.</div>\n"
        "</div>\n"
        "<div id=\"main\">\n"
        "  <div id=\"mode-title\"></div>\n"
        "  <div id=\"plot\"></div>\n"
        "</div>\n"
        "</div>\n"
        "<script>\n"
        f"var data = {json.dumps(all_traces)};\n"
        f"var layout = {json.dumps(layout)};\n"
        f"var N_STATIC = {n_static};\n"
        f"var CELL_END = {cell_end};\n"
        f"var ATOMS_END = {atoms_end};\n"
        f"var BONDS_END = {bonds_end};\n"
        f"var MODES = {json.dumps(mode_specs)};\n"
        f"var DYNAMIC_CENTERS = {json.dumps(dynamic_centers)};\n"
        "var currentMode = 0;\n"
        "var baseOpacity = 1.0;\n"
        "var plotDiv = document.getElementById('plot');\n"
        "function isChecked(id) {\n"
        "  var element = document.getElementById(id);\n"
        "  return element ? element.checked : true;\n"
        "}\n"
        "function applyVisibility() {\n"
        "  var spec = MODES[currentMode];\n"
        "  var visible = data.map(function (_, index) {\n"
        "    if (index < CELL_END) { return isChecked('show-cell'); }\n"
        "    if (index < ATOMS_END) { return isChecked('show-atoms'); }\n"
        "    if (index < BONDS_END) { return isChecked('show-bonds'); }\n"
        "    if (index < N_STATIC) { return isChecked('show-poly'); }\n"
        "    var dynamicOffset = index - N_STATIC;\n"
        "    return spec && dynamicOffset >= spec.start && dynamicOffset < spec.start + spec.count;\n"
        "  });\n"
        "  Plotly.restyle('plot', {visible: visible}, visible.map(function (_, i) { return i; }));\n"
        "}\n"
        "function setMode(index) {\n"
        "  currentMode = index;\n"
        "  var rows = document.querySelectorAll('tr.mode-row');\n"
        "  rows.forEach(function (row, rowIndex) {\n"
        "    row.classList.toggle('active', rowIndex === index);\n"
        "  });\n"
        "  var spec = MODES[index];\n"
        "  document.getElementById('mode-title').innerHTML =\n"
        "    'Mode ' + spec.space + ' <span class=\"irrep\">' + spec.irrep + '</span> — component ' + spec.component;\n"
        "  applyVisibility();\n"
        "}\n"
        "function currentCamera() {\n"
        "  var scene = plotDiv._fullLayout && plotDiv._fullLayout.scene;\n"
        "  return (scene && scene.camera && scene.camera.eye)\n"
        "    ? scene.camera\n"
        "    : {eye: {x: 1.25, y: 1.25, z: 1.25}, up: {x: 0, y: 0, z: 1}};\n"
        "}\n"
        "function syncCompass(camera) {\n"
        "  // update ONLY the compass scene through plotly's internal setViewport:\n"
        "  // calling Plotly.relayout during a drag would redraw the main scene\n"
        "  // from the (stale) stored camera and cancel the rotation in progress.\n"
        "  var eye = camera.eye;\n"
        "  var radius = Math.sqrt(eye.x * eye.x + eye.y * eye.y + eye.z * eye.z) || 1;\n"
        "  var scale = 2.0 / radius;  // compass orientation follows the view, size stays fixed\n"
        "  var sceneLayout = plotDiv._fullLayout && plotDiv._fullLayout.scene2;\n"
        "  var sceneObject = sceneLayout && sceneLayout._scene;\n"
        "  if (!sceneObject) { return; }\n"
        "  var scaledEye = {x: eye.x * scale, y: eye.y * scale, z: eye.z * scale};\n"
        "  var upVector = camera.up\n"
        "    ? {x: camera.up.x, y: camera.up.y, z: camera.up.z}\n"
        "    : {x: 0, y: 0, z: 1};\n"
        "  sceneLayout.camera.eye = scaledEye;\n"
        "  sceneLayout.camera.up = upVector;\n"
        "  sceneLayout.camera.center = {x: 0, y: 0, z: 0};\n"
        "  sceneLayout.camera.projection = sceneLayout.camera.projection || {type: 'perspective'};\n"
        "  // persist into the user layout too: on drag end plotly rebuilds the\n"
        "  // compass scene from plotDiv.layout, which would reset the camera\n"
        "  if (plotDiv.layout && plotDiv.layout.scene2) {\n"
        "    plotDiv.layout.scene2.camera = {eye: scaledEye, up: upVector, center: {x: 0, y: 0, z: 0}};\n"
        "  }\n"
        "  try { sceneObject.setViewport(sceneLayout); } catch (error) { /* keep the drag alive */ }\n"
        "}\n"
        "// Fully opaque lobes (the default) get correct front/back occlusion from\n"
        "// the WebGL depth test. When the user makes them translucent, plotly\n"
        "// cannot depth-sort transparent surfaces, so a depth cue is applied\n"
        "// instead: lobes far from the camera are drawn fainter than near ones.\n"
        "function depthFade(camera) {\n"
        "  if (!DYNAMIC_CENTERS.length) { return; }\n"
        "  var indices = [];\n"
        "  for (var i = N_STATIC; i < data.length; i++) { indices.push(i); }\n"
        "  if (!indices.length) { return; }\n"
        "  if (baseOpacity >= 0.99) {\n"
        "    Plotly.restyle('plot', {opacity: 1.0}, indices);\n"
        "    return;\n"
        "  }\n"
        "  var eye = camera.eye;\n"
        "  var radius = Math.sqrt(eye.x * eye.x + eye.y * eye.y + eye.z * eye.z) || 1;\n"
        "  var vx = eye.x / radius, vy = eye.y / radius, vz = eye.z / radius;\n"
        "  var depths = DYNAMIC_CENTERS.map(function (c) { return c[0] * vx + c[1] * vy + c[2] * vz; });\n"
        "  var minDepth = Math.min.apply(null, depths);\n"
        "  var maxDepth = Math.max.apply(null, depths);\n"
        "  var span = (maxDepth - minDepth) || 1;\n"
        "  var opacities = depths.map(function (d) {\n"
        "    return Math.round(baseOpacity * (0.2 + 0.8 * (d - minDepth) / span) * 100) / 100;\n"
        "  });\n"
        "  Plotly.restyle('plot', {opacity: opacities}, indices);\n"
        "}\n"
        "function setOpacity(value) {\n"
        "  baseOpacity = Number(value);\n"
        "  depthFade(currentCamera());\n"
        "}\n"
        "Plotly.newPlot('plot', data, layout, {responsive: true, displaylogo: false}).then(function () {\n"
        "  if (MODES.length) { setMode(0); }\n"
        "  syncCompass(currentCamera());\n"
        "  depthFade(currentCamera());\n"
        "  // sync the compass continuously while dragging; refresh the depth cue on release\n"
        "  plotDiv.on('plotly_relayouting', function (event) {\n"
        "    if (event['scene.camera']) { syncCompass(event['scene.camera']); }\n"
        "  });\n"
        "  plotDiv.on('plotly_relayout', function (event) {\n"
        "    if (event['scene.camera']) {\n"
        "      syncCompass(event['scene.camera']);\n"
        "      depthFade(event['scene.camera']);\n"
        "    }\n"
        "  });\n"
        "});\n"
        "</script>\n</body>\n</html>\n"
    )
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(html)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    l = ORBITAL_L[args.orbital]

    from .star_of_k import read_poscar_or_exit, resolve_kpoint_input

    cell = read_poscar_or_exit(args.poscar)
    orbitals = SymmetryAdaptedOrbitalBasis(cell=cell, symprec=args.tolerance)

    kpoint_label, kpoint = resolve_kpoint_input(orbitals, args.kpoint)

    element_indices = orbitals.get_element_indices(args.element)
    wyckoff_letters = [orbitals.spglib_dataset["wyckoffs"][index] for index in element_indices]
    site_symmetry_symbols = [
        orbitals.spglib_dataset["site_symmetry_symbols"][index] for index in element_indices
    ]

    print(
        f"\n * Space group *\n {orbitals.spglib_dataset['international']} "
        f"({orbitals.spglib_dataset['number']})\n"
    )
    print(f" * Orbital (number of atoms) *\n {args.element}_{args.orbital} ({len(element_indices)})\n")
    print(" * Position *")
    print(f" wyckoff letters      : {wyckoff_letters}")
    print(f" site symmetry letters: {site_symmetry_symbols}\n")
    print(f" * k point (primitive) * \n {kpoint_label} {format_kpoint(kpoint)}\n")

    irreps, orbital_rep, mapping_little_group, element_indices = orbitals.get_orbital_rep(
        kpoint=kpoint,
        element=args.element,
        l=l,
    )
    irrep_labels = orbitals.get_irrep_labels(kpoint, irreps, mapping_little_group)

    multiplicities = orbitals.decompose_orbital_rep(irreps, orbital_rep, irrep_labels)
    sorted_multiplicities = sort_irrep_items(
        [(key, value) for key, value in multiplicities.items() if value > 0]
    )
    decomposition = "+".join(f" {value} [{key}] " for key, value in sorted_multiplicities)
    print(" * Irreducible Decomposition *")
    print(decomposition + "\n")

    basis_spaces, basis_labels = orbitals.get_orbital_basis(irreps, orbital_rep, irrep_labels)

    if args.real_coefficient:
        realified_spaces = []
        for space, label in zip(basis_spaces, basis_labels):
            new_space, success = realify_basis_space(space)
            realified_spaces.append(new_space)
            if not success:
                print(
                    f" NOTE: no real-coefficient basis exists for {label} "
                    "(space not closed under complex conjugation); left unchanged."
                )
        basis_spaces = realified_spaces

    if args.mode_index is not None and not (1 <= args.mode_index <= len(basis_spaces)):
        raise SystemExit(
            f"ERROR: --mode-index {args.mode_index} is out of range "
            f"[1, {len(basis_spaces)}] (numbering is 1-based)."
        )
    selected_space_index = None if args.mode_index is None else args.mode_index - 1

    _print_salc_coefficients(
        basis_spaces=basis_spaces,
        basis_labels=basis_labels,
        element=args.element,
        element_indices=element_indices,
        l=l,
        mode_index=selected_space_index,
    )

    if args.output:
        output_path = args.output
    else:
        if kpoint_label and kpoint_label != "custom":
            kpoint_tag = "GM" if kpoint_label.upper() == "GAMMA" else kpoint_label
        else:
            kpoint_tag = "_".join(f"{value:g}" for value in kpoint)
        mode_tag = f"_mode{args.mode_index}" if args.mode_index is not None else ""
        conv_tag = "_conv" if args.conventional else ""
        output_path = f"SALC_{args.element}_{args.orbital}_{kpoint_tag}{mode_tag}{conv_tag}.html"

    from collections import Counter

    composition = Counter(get_chemical_symbols(orbitals.primitive_cell))
    formula = "".join(
        f"{symbol}{count if count > 1 else ''}" for symbol, count in composition.items()
    )
    info = {
        "formula": formula,
        "space_group": f"{orbitals.spglib_dataset['international']} (#{orbitals.spglib_dataset['number']})",
        "element_orbital": f"{args.element}_{args.orbital}",
        "kpoint": f"{kpoint_label} {format_kpoint(kpoint)}",
        "decomposition": decomposition,
        "real_coefficient": args.real_coefficient,
    }

    bond_specs: list[tuple[str, str, float]] = []
    for el1, el2, max_text in args.bond or []:
        try:
            bond_specs.append((el1, el2, float(max_text)))
        except ValueError:
            raise SystemExit(
                f"ERROR: --bond expects a numeric maximum length in Angstroms, got '{max_text}'."
            )

    title = f"SALC: {args.element}_{args.orbital} at {kpoint_label} {format_kpoint(kpoint)}"
    write_html_visualization(
        output_path=output_path,
        orbitals=orbitals,
        basis_spaces=basis_spaces,
        basis_labels=basis_labels,
        element_indices=element_indices,
        l=l,
        kpoint=kpoint,
        title=title,
        mode_index=selected_space_index,
        info=info,
        bonds=bond_specs,
        conventional=args.conventional,
    )
    print(f"Saved 3D visualization to: {output_path}")


if __name__ == "__main__":
    main()
