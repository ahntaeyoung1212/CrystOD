"""
Supercell Brillouin-zone plot: interactive 3D HTML view of the first
Brillouin zone of a unit cell together with the (smaller) Brillouin zone of
a transformed (super)lattice, tiled at the supercell reciprocal-lattice
points folded into the unit-cell BZ.

Based on `script/supercell_BZ.py` by Hiroki Koiso (Nakajima group, 2023).
"""

from __future__ import annotations

import os
from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    RawDescriptionHelpFormatter,
    RawTextHelpFormatter,
)
from fractions import Fraction
from itertools import product

import numpy as np
from numpy.typing import NDArray

from .brillouin_zone import get_brillouin_zone_3d, write_html


class MyHelpFormatter(
    RawTextHelpFormatter,
    RawDescriptionHelpFormatter,
    ArgumentDefaultsHelpFormatter,
):
    pass


desc = """
Plot the first Brillouin zone of a unit cell together with the Brillouin
zone of a transformed (super)lattice as an interactive 3D HTML file.
The supercell BZ is tiled at every supercell reciprocal-lattice point that
folds into the unit-cell BZ (i.e. the unit-cell q-points that fold onto the
Gamma point of the supercell).

# Command Example:
crystod-bz -c 221_PPOSCAR_ScF3 --trans-mat "0 1 2  -1 0 2  1 -1 2" --output BZ_supercell.html
"""


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=desc, formatter_class=MyHelpFormatter)
    parser.add_argument("--poscar", default="POSCAR", help="POSCAR path (unit cell).")
    parser.add_argument(
        "--trans-mat",
        "--trans-matrix",
        dest="trans_mat",
        required=True,
        type=str,
        help='Unit-cell to supercell transformation matrix (row-wise), e.g. "0 1 2  -1 0 2  1 -1 2". Fractions such as 1/2 are allowed.',
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output HTML path. Default: BZ_supercell_{POSCAR name}.html in the current directory.",
    )
    return parser


def parse_transformation_matrix(text: str) -> NDArray[np.float64]:
    values = [float(Fraction(token)) for token in text.split()]
    if len(values) != 9:
        raise SystemExit(
            f'ERROR: --trans-mat requires nine numbers ("t11 t12 t13  t21 ..."), got {len(values)}.'
        )
    matrix = np.array(values, dtype=float).reshape(3, 3)
    if abs(np.linalg.det(matrix)) < 1e-10:
        raise SystemExit("ERROR: the transformation matrix is singular.")
    return matrix


def get_folded_gamma_points(trans_mat: NDArray[np.float64], rec_lat: NDArray[np.float64]) -> NDArray[np.float64]:
    """Supercell reciprocal-lattice points folded into the unit-cell BZ.

    Returns their Cartesian coordinates (one per equivalence class modulo the
    unit-cell reciprocal lattice; det(T) points including Gamma). These are
    exactly the unit-cell q-points that fold onto Gamma of the supercell.
    """
    n_classes = int(round(abs(np.linalg.det(trans_mat))))
    inv_t_transpose = np.linalg.inv(trans_mat).T

    # fractional coordinates (unit-cell reciprocal basis) of m @ rec_super
    unique_fracs: list[NDArray[np.float64]] = []
    search = max(2, n_classes)
    for m in product(range(-search, search + 1), repeat=3):
        frac = (np.array(m, dtype=float) @ inv_t_transpose) % 1.0
        frac = np.where(frac > 1.0 - 1e-8, 0.0, frac)
        if not any(np.allclose(frac, known, atol=1e-8) for known in unique_fracs):
            unique_fracs.append(frac)
        if len(unique_fracs) == n_classes:
            break

    # fold each representative into the Wigner-Seitz cell (minimum image)
    neighbors = np.array(list(product((-1, 0, 1), repeat=3)), dtype=float) @ rec_lat
    folded = []
    for frac in unique_fracs:
        point = frac @ rec_lat
        shifted = point - neighbors
        point = shifted[np.argmin(np.linalg.norm(shifted, axis=1))]
        folded.append(point)
    return np.array(folded)


def build_supercell_bz_traces(
    rec_lat: NDArray[np.float64],
    rec_super_lat: NDArray[np.float64],
    centers: NDArray[np.float64],
) -> list[dict]:
    traces: list[dict] = []

    # Unit-cell reciprocal basis (black, dotted)
    basis_labels = ["<i>b<sub>1</sub></i>", "<i>b<sub>2</sub></i>", "<i>b<sub>3</sub></i>"]
    for label, basis in zip(basis_labels, rec_lat):
        bx, by, bz = (float(value) for value in basis)
        traces.append(
            {
                "type": "scatter3d",
                "x": [0.0, bx],
                "y": [0.0, by],
                "z": [0.0, bz],
                "mode": "lines+text",
                "line": {"color": "black", "width": 3, "dash": "dot"},
                "text": ["", label],
                "textfont": {"color": "black", "size": 20},
                "opacity": 0.8,
                "hoverinfo": "skip",
            }
        )

    # Supercell reciprocal basis (colored, solid)
    basis_colors = ["red", "green", "blue"]
    for color, label, basis in zip(basis_colors, basis_labels, rec_super_lat):
        bx, by, bz = (float(value) for value in basis)
        traces.append(
            {
                "type": "scatter3d",
                "x": [0.0, bx],
                "y": [0.0, by],
                "z": [0.0, bz],
                "mode": "lines+text",
                "line": {"color": color, "width": 6},
                "text": ["", label],
                "textfont": {"color": color, "size": 25},
                "opacity": 0.8,
                "hoverinfo": "skip",
            }
        )

    # Unit-cell BZ edges (black, dotted)
    _, edges, _ = get_brillouin_zone_3d(rec_lat)
    for edge in edges:
        traces.append(
            {
                "type": "scatter3d",
                "x": edge[:, 0].tolist(),
                "y": edge[:, 1].tolist(),
                "z": edge[:, 2].tolist(),
                "mode": "lines",
                "line": {"color": "black", "width": 3, "dash": "dot"},
                "opacity": 0.8,
                "hoverinfo": "skip",
            }
        )

    # Supercell BZ edges (red) at every folded supercell reciprocal-lattice point
    _, super_edges, _ = get_brillouin_zone_3d(rec_super_lat)
    for center in centers:
        cx, cy, cz = (float(value) for value in center)
        for edge in super_edges:
            traces.append(
                {
                    "type": "scatter3d",
                    "x": (edge[:, 0] + cx).tolist(),
                    "y": (edge[:, 1] + cy).tolist(),
                    "z": (edge[:, 2] + cz).tolist(),
                    "mode": "lines",
                    "line": {"color": "red", "width": 3},
                    "opacity": 0.8,
                    "hoverinfo": "skip",
                }
            )

    # Markers at the folded centers, with unit-cell fractional coordinates on hover
    centers_frac = centers @ np.linalg.inv(rec_lat)
    traces.append(
        {
            "type": "scatter3d",
            "x": centers[:, 0].tolist(),
            "y": centers[:, 1].tolist(),
            "z": centers[:, 2].tolist(),
            "mode": "markers",
            "marker": {"color": "red", "size": 4},
            "customdata": centers_frac.tolist(),
            "hovertemplate": (
                "folds onto supercell Gamma<br>q-position: (%{customdata[0]:.3f}, "
                "%{customdata[1]:.3f}, %{customdata[2]:.3f})<extra></extra>"
            ),
            "opacity": 1,
        }
    )
    return traces


def _format_fraction(value: float) -> str:
    fraction = Fraction(value).limit_denominator(24)
    if abs(float(fraction) - value) > 1e-6:
        return f"{value:.4f}"
    return str(fraction)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    from .star_of_k import read_poscar_or_exit

    cell = read_poscar_or_exit(args.poscar)
    lattice = np.array(cell.cell, dtype=float)
    trans_mat = parse_transformation_matrix(args.trans_mat)

    super_lattice = trans_mat @ lattice
    # Koiso convention: reciprocal lattice without the 2*pi factor.
    rec_lat = np.linalg.inv(lattice).T
    rec_super_lat = np.linalg.inv(super_lattice).T
    n_cells = abs(np.linalg.det(trans_mat))

    print("Transformation matrix (unit cell -> supercell):")
    for row in trans_mat:
        print(f"  [{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f}]")
    print(f"Volume ratio |det T| = {n_cells:g}")
    print("\nSupercell lattice (rows):")
    for row in super_lattice:
        print(f"  [{row[0]:10.6f} {row[1]:10.6f} {row[2]:10.6f}]")

    centers = get_folded_gamma_points(trans_mat, rec_lat)
    centers_frac = centers @ np.linalg.inv(rec_lat)
    print(f"\nUnit-cell q-points folding onto the supercell Gamma point ({len(centers)}):")
    for frac in centers_frac:
        text = ", ".join(_format_fraction(float(value)) for value in frac)
        print(f"  ({text})")

    traces = build_supercell_bz_traces(rec_lat, rec_super_lat, centers)

    output = args.output
    if output is None:
        output = f"BZ_supercell_{os.path.basename(args.poscar)}.html"
    title = f"Brillouin zones: {os.path.basename(args.poscar)} (black: unit cell, red: supercell)"
    write_html(traces, output, title)
    print(f"\nWrote supercell Brillouin-zone visualization: {output}")


if __name__ == "__main__":
    main()
