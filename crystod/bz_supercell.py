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

from .brillouin_zone import _selling_reduce, get_brillouin_zone_3d, write_html


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
        help='Unit-cell to supercell transformation matrix (row-wise, integer entries), e.g. "0 1 2  -1 0 2  1 -1 2".',
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output HTML path. Default: BZ_supercell_{POSCAR name}.html in the current directory.",
    )
    return parser


def parse_transformation_matrix(text: str) -> NDArray[np.float64]:
    """Parse a ``--trans-mat`` string into a ``(3, 3)`` transformation matrix.

    The nine numbers are read row-wise, as ``crystod-bz --trans-mat`` takes
    them. The matrix ``T`` maps the unit-cell lattice ``A`` (row vectors) to
    the supercell lattice ``T @ A``, so ``|det T|`` is the number of unit
    cells in the supercell. The entries must be integers (a fraction that
    equals an integer, such as ``4/2``, is accepted): with a non-integer
    entry ``T @ A`` is not a superlattice of the unit cell, and the folding
    onto its Gamma point is undefined.

    Args:
        text: Nine whitespace-separated numbers, e.g.
            ``"0 1 2  -1 0 2  1 -1 2"``.

    Returns:
        The ``(3, 3)`` float matrix.

    Raises:
        SystemExit: The string does not hold exactly nine numbers, an entry
            is not an integer, or the matrix is singular (``ValueError``
            when called through ``crystod.bz``).
        ValueError: A token is neither a number nor a fraction.

    Example:
        >>> from crystod import bz
        >>> bz.parse_transformation_matrix("2 0 0  0 2 0  0 0 2").tolist()
        [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
    """
    tokens = text.split()
    values = [Fraction(token) for token in tokens]
    if len(values) != 9:
        raise SystemExit(
            f'ERROR: --trans-mat requires nine numbers ("t11 t12 t13  t21 ..."), got {len(values)}.'
        )
    non_integer = [token for token, value in zip(tokens, values) if value.denominator != 1]
    if non_integer:
        raise SystemExit(
            "ERROR: --trans-mat must have integer entries (the supercell vectors are "
            "integer combinations of the unit-cell vectors); got "
            + ", ".join(non_integer)
            + "."
        )
    matrix = np.array([float(value) for value in values], dtype=float).reshape(3, 3)
    if abs(np.linalg.det(matrix)) < 1e-10:
        raise SystemExit("ERROR: the transformation matrix is singular.")
    return matrix


def get_folded_gamma_points(trans_mat: NDArray[np.float64], rec_lat: NDArray[np.float64]) -> NDArray[np.float64]:
    """Unit-cell q points that fold onto the Gamma point of a supercell.

    The reciprocal lattice of the supercell ``trans_mat @ lattice`` is
    ``|det T|`` times denser than that of the unit cell, and its points fall
    into ``|det T|`` classes modulo the unit-cell reciprocal lattice. One
    representative per class, brought into the unit-cell Brillouin zone by
    the minimum-image rule, is a unit-cell q point that becomes Gamma in the
    supercell; ``crystod-bz --trans-mat`` prints these points and tiles the
    supercell zone at each of them. The minimum image is searched on the
    Selling-reduced basis as well, so it is the nearest image for any basis
    of the unit cell (among equally near images, the first of the 27
    ``{-1, 0, 1}`` images in the input basis is kept).

    Args:
        trans_mat: ``(3, 3)`` unit-cell to supercell transformation matrix
            (rows, integer entries), e.g. from ``parse_transformation_matrix``.
        rec_lat: ``(3, 3)`` reciprocal lattice of the unit cell, rows
            ``b1``, ``b2``, ``b3`` (the command uses ``inv(lattice).T``).

    Returns:
        ``(|det T|, 3)`` array of Cartesian coordinates, Gamma included;
        ``points @ inv(rec_lat)`` gives them in fractional coordinates of
        the unit-cell reciprocal basis.

    Raises:
        SystemExit: ``trans_mat`` has a non-integer entry (``ValueError``
            when called through ``crystod.bz``).

    Example:
        >>> import numpy as np
        >>> from crystod import bz
        >>> rec_lat = np.linalg.inv(4.07 * np.eye(3)).T
        >>> T = bz.parse_transformation_matrix("1 1 0  -1 1 0  0 0 1")
        >>> points = bz.get_folded_gamma_points(T, rec_lat)
        >>> np.round(points @ np.linalg.inv(rec_lat), 3).tolist()
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]]
    """
    trans_mat = np.asarray(trans_mat, dtype=float)
    rec_lat = np.asarray(rec_lat, dtype=float)
    if not np.allclose(trans_mat, np.rint(trans_mat), atol=1e-8):
        raise SystemExit(
            "ERROR: the transformation matrix must have integer entries "
            "(a supercell of the unit cell)."
        )
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
    # The 27 images of an unreduced basis (a sheared or very oblique cell) can
    # miss the nearest one: also wrap the point into the parallelepiped of the
    # Selling-reduced basis and search 5x5x5 images around it there. An image
    # found that way replaces the first choice only when it is strictly
    # nearer, so ties keep the image of the 27-point scan.
    reduced, _ = _selling_reduce(rec_lat)
    reduced_inv = np.linalg.inv(reduced)
    reduced_neighbors = (
        np.array(list(product(range(-2, 3), repeat=3)), dtype=float) @ reduced
    )
    folded = []
    for frac in unique_fracs:
        point = frac @ rec_lat
        shifted = point - neighbors
        norms = np.linalg.norm(shifted, axis=1)
        best = shifted[np.argmin(norms)]
        wrapped = point - np.rint(point @ reduced_inv) @ reduced
        candidates = wrapped - reduced_neighbors
        candidate_norms = np.linalg.norm(candidates, axis=1)
        if candidate_norms.min() < norms.min() * (1.0 - 1e-9):
            best = candidates[np.argmin(candidate_norms)]
        folded.append(best)
    return np.array(folded)


def build_supercell_bz_traces(
    rec_lat: NDArray[np.float64],
    rec_super_lat: NDArray[np.float64],
    centers: NDArray[np.float64],
) -> list[dict]:
    """Build the Plotly traces of a unit-cell zone tiled with supercell zones.

    This is the figure of ``crystod-bz --trans-mat``: the unit-cell reciprocal
    basis and zone edges (black, dotted), the supercell reciprocal basis
    (red, green, blue) and a copy of the supercell Brillouin zone (red)
    centred at every point of ``centers``, with a marker there whose hover
    text gives the unit-cell fractional coordinates. As in
    ``build_bz_traces``, the result is a list of plain ``scatter3d`` trace
    dictionaries.

    Args:
        rec_lat: ``(3, 3)`` reciprocal lattice of the unit cell (rows).
        rec_super_lat: ``(3, 3)`` reciprocal lattice of the supercell (rows),
            ``inv(trans_mat @ lattice).T`` in the command's convention.
        centers: ``(n, 3)`` Cartesian centres of the supercell zones, normally
            the output of ``get_folded_gamma_points``.

    Returns:
        A list of Plotly ``scatter3d`` trace dictionaries.

    Example:
        >>> import numpy as np
        >>> from crystod import bz
        >>> lattice = 4.07 * np.eye(3)
        >>> T = bz.parse_transformation_matrix("2 0 0  0 2 0  0 0 2")
        >>> rec_lat = np.linalg.inv(lattice).T
        >>> rec_super_lat = np.linalg.inv(T @ lattice).T
        >>> centers = bz.get_folded_gamma_points(T, rec_lat)
        >>> traces = bz.build_supercell_bz_traces(rec_lat, rec_super_lat, centers)
        >>> len(centers), len(traces)
        (8, 61)
    """
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

    print("\n* Transformation matrix (unit cell -> supercell) *")
    for row in trans_mat:
        print(f"  [{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f}]")
    print(f"  Volume ratio |det T| = {n_cells:g}")
    print("\n* Supercell lattice (rows) *")
    for row in super_lattice:
        print(f"  [{row[0]:10.6f} {row[1]:10.6f} {row[2]:10.6f}]")

    centers = get_folded_gamma_points(trans_mat, rec_lat)
    centers_frac = centers @ np.linalg.inv(rec_lat)
    print(f"\n* Unit-cell q-points folding onto the supercell Gamma point ({len(centers)}) *")
    for frac in centers_frac:
        text = ", ".join(_format_fraction(float(value)) for value in frac)
        print(f"  ({text})")

    traces = build_supercell_bz_traces(rec_lat, rec_super_lat, centers)

    output = args.output
    if output is None:
        output = f"BZ_supercell_{os.path.basename(args.poscar)}.html"
    title = f"Brillouin zones: {os.path.basename(args.poscar)} (black: unit cell, red: supercell)"
    write_html(traces, output, title)
    print("\n* Output files *")
    print(f"  Wrote supercell Brillouin-zone visualization: {output}")


if __name__ == "__main__":
    main()
