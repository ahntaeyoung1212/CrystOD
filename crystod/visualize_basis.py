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
        help="Only print/visualize the selected irrep-grouped SALC space.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional standalone HTML path for the interactive 3D visualization.",
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
        print(f" Mode Space {space_index}: irrep = {label}, dimension = {space.shape[0]}")
        for component_index, vector in enumerate(space):
            print(f"   component {component_index}:")
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
    n_theta: int = 30,
    n_phi: int = 60,
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
    return {
        "type": "surface",
        "x": points[:, 0].reshape(shape).tolist(),
        "y": points[:, 1].reshape(shape).tolist(),
        "z": points[:, 2].reshape(shape).tolist(),
        "surfacecolor": np.sign(values).reshape(shape).tolist(),
        "cmin": -1,
        "cmax": 1,
        "colorscale": [[0.0, "#3b6fd4"], [0.5, "#dddddd"], [1.0, "#d43b3b"]],
        "showscale": False,
        "opacity": 0.85,
        "hoverinfo": "skip",
        "showlegend": False,
    }


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
) -> None:
    primitive = orbitals.primitive_cell
    lattice = np.array(primitive.cell, dtype=float)
    frac_positions = np.array(primitive.scaled_positions, dtype=float)
    symbols = get_chemical_symbols(primitive)
    supercell_size = orbitals.get_supercell_size(kpoint)
    n1, n2, n3 = supercell_size

    all_positions = []
    all_symbols = []
    target_slots = []  # (atom_slot_in_element_list, cartesian position, phase)
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                translation = np.array([i1, i2, i3], dtype=float)
                phase = np.exp(2j * np.pi * np.dot(kpoint, translation))
                for atom_index in range(len(frac_positions)):
                    position = (frac_positions[atom_index] + translation) @ lattice
                    all_positions.append(position)
                    all_symbols.append(symbols[atom_index])
                    if atom_index in element_indices:
                        target_slots.append((element_indices.index(atom_index), position, phase))
    all_positions = np.array(all_positions)

    nearest = np.inf
    for slot_a in range(len(all_positions)):
        for slot_b in range(slot_a + 1, len(all_positions)):
            nearest = min(nearest, float(np.linalg.norm(all_positions[slot_a] - all_positions[slot_b])))
    lobe_scale = 0.45 * nearest if np.isfinite(nearest) else 1.0

    static_traces = [_lattice_edge_traces(lattice, supercell_size)]
    static_traces.extend(_atom_traces(all_positions, all_symbols))
    n_static = len(static_traces)

    component_names = ORBITAL_COMPONENT_NAMES[l]
    n_components = len(component_names)

    dynamic_traces = []
    button_specs = []  # (label, trace start, trace count)
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
            button_specs.append(
                (
                    f"Mode {space_index} [{label}] comp {component_index}",
                    start,
                    len(dynamic_traces) - start,
                )
            )

    buttons = []
    n_dynamic = len(dynamic_traces)
    for label, start, count in button_specs:
        visibility = [True] * n_static + [False] * n_dynamic
        for offset in range(count):
            visibility[n_static + start + offset] = True
        buttons.append(
            {
                "label": label,
                "method": "update",
                "args": [{"visible": visibility}, {"title": f"{title} — {label}"}],
            }
        )

    all_traces = static_traces + dynamic_traces
    initial_visible = buttons[0]["args"][0]["visible"] if buttons else [True] * n_static
    for trace, visible in zip(all_traces, initial_visible):
        trace["visible"] = visible

    layout = {
        "title": {"text": f"{title} — {button_specs[0][0]}" if button_specs else title},
        "scene": {
            "aspectmode": "data",
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
        },
        "margin": {"l": 0, "r": 0, "t": 60, "b": 0},
        "updatemenus": [
            {
                "buttons": buttons,
                "direction": "down",
                "x": 0.0,
                "y": 1.0,
                "xanchor": "left",
                "yanchor": "top",
            }
        ]
        if buttons
        else [],
        "legend": {"x": 1.0, "y": 0.9},
    }

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
</head>
<body>
<div id="plot" style="width:100vw;height:96vh;"></div>
<script>
var data = {json.dumps(all_traces)};
var layout = {json.dumps(layout)};
Plotly.newPlot("plot", data, layout, {{responsive: true}});
</script>
</body>
</html>
"""
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

    _print_salc_coefficients(
        basis_spaces=basis_spaces,
        basis_labels=basis_labels,
        element=args.element,
        element_indices=element_indices,
        l=l,
        mode_index=args.mode_index,
    )

    if args.output:
        title = f"SALC: {args.element}_{args.orbital} at {kpoint_label} {format_kpoint(kpoint)}"
        write_html_visualization(
            output_path=args.output,
            orbitals=orbitals,
            basis_spaces=basis_spaces,
            basis_labels=basis_labels,
            element_indices=element_indices,
            l=l,
            kpoint=kpoint,
            title=title,
            mode_index=args.mode_index,
        )
        print(f"Saved 3D visualization to: {args.output}")


if __name__ == "__main__":
    main()
