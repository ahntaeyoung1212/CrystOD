"""
Brillouin-zone plot: interactive 3D HTML view of the first Brillouin zone
with an automatically generated high-symmetry k-path (seekpath).

Based on `script/brillouin_zone_plot.py` by Hiroki Koiso (Nakajima group, 2023);
BZ construction via Voronoi decomposition follows Qijing Zheng
(http://staff.ustc.edu.cn/~zqj/posts/howto-plot-brillouin-zone/).
"""

from __future__ import annotations

import json
import os
from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    RawDescriptionHelpFormatter,
    RawTextHelpFormatter,
)
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray


class MyHelpFormatter(
    RawTextHelpFormatter,
    RawDescriptionHelpFormatter,
    ArgumentDefaultsHelpFormatter,
):
    pass


desc = """
Plot the first Brillouin zone as an interactive 3D HTML file.

By default, the space group of the POSCAR is detected and the recommended
high-symmetry k-path is generated automatically with seekpath.
A custom path can be given instead with --band/--label.

# Command Examples:
crystod-bz -c 221_PPOSCAR_ScF3
crystod-bz -c 221_PPOSCAR_ScF3 --output BZ_ScF3_Pm-3m.html
crystod-bz -c 221_PPOSCAR_ScF3 \\
    --band "0 0 0  0 1/2 0  1/2 1/2 0  0 0 0  1/2 1/2 1/2  0 1/2 0, 1/2 1/2 0  1/2 1/2 1/2" \\
    --band-labels "GM X M GM R X  M R"
"""


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=desc, formatter_class=MyHelpFormatter)
    parser.add_argument("--poscar", default="POSCAR", help="POSCAR path.")
    parser.add_argument(
        "--band",
        default=None,
        help=(
            "Optional manual band path. Comma-separated continuous segments,\n"
            'each a whitespace-separated list of fractional coordinates, e.g.\n'
            '"0 0 0  0 1/2 0  1/2 1/2 0, 1/2 1/2 0  1/2 1/2 1/2".\n'
            "If omitted, the path is generated automatically with seekpath."
        ),
    )
    parser.add_argument(
        "--label",
        "--band-labels",
        dest="label",
        default=None,
        help='Optional labels for the manual band path, e.g. "GM X M GM R X M R".',
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output HTML path. Default: BZ_{POSCAR name}.html in the current directory.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Symmetry tolerance forwarded to seekpath/spglib.",
    )
    return parser


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _selling_reduce(basis: NDArray) -> tuple[NDArray, NDArray]:
    """Selling-reduce a 3D lattice basis (obtuse superbase).

    The superbase ``v0 = -(b1 + b2 + b3), v1 = b1, v2 = b2, v3 = b3`` is
    reduced until all six products ``vi . vj`` (i != j) are <= 0: while some
    ``vi . vj > 0``, ``vk`` and ``vl`` (the other two) become ``vk + vi`` and
    ``vl + vi`` and ``vi`` becomes ``-vi``, which lowers ``sum |v|^2`` by
    ``4 vi . vj``. For the reduced basis ``(v1, v2, v3)`` every
    Voronoi-relevant lattice vector is ``+-`` a sum of a subset of the
    superbase (Conway and Sloane, Proc. R. Soc. Lond. A 436, 55 (1992)), so
    its coefficients lie in ``{-1, 0, 1}``: the 27-point Voronoi construction
    of ``get_brillouin_zone_3d`` is exact for it. A basis that already is an
    obtuse superbase (cubic, tetragonal, orthorhombic, the fcc and bcc
    primitive bases, ...) is returned unchanged.

    Args:
        basis: ``(3, 3)`` array whose rows are the basis vectors.

    Returns:
        The tuple ``(reduced, U)`` where ``reduced`` is the ``(3, 3)`` reduced
        basis (rows) and ``U`` the unimodular integer matrix with
        ``reduced = U @ basis``.
    """
    basis = np.asarray(basis, dtype=float)
    # superbase as integer combinations of the input basis vectors
    coefficients = np.array(
        [[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int64
    )
    vectors = coefficients @ basis
    threshold = 1e-10 * float(np.max(np.einsum("ij,ij->i", vectors, vectors)))
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    for _ in range(100000):
        gram = vectors @ vectors.T
        i, j = max(pairs, key=lambda pair: gram[pair])
        if gram[i, j] <= threshold:
            break
        k, l = (m for m in range(4) if m not in (i, j))
        coefficients[k] += coefficients[i]
        coefficients[l] += coefficients[i]
        coefficients[i] = -coefficients[i]
        vectors = coefficients @ basis
    else:  # pragma: no cover - sum |v|^2 decreases at every step
        raise RuntimeError("Selling reduction of the reciprocal basis did not converge.")
    unimodular = coefficients[1:]
    if np.array_equal(unimodular, np.eye(3, dtype=np.int64)):
        return basis.copy(), unimodular
    return unimodular @ basis, unimodular


def _merge_close_vertices(vertices: NDArray, faces: list, tol: float):
    """Merge vertices closer than ``tol`` and drop faces that collapse.

    ``faces`` are lists of row indices of ``vertices`` in cyclic order. Returns
    ``(merged vertices, faces as index lists into them)``; with no pair closer
    than ``tol`` the vertices and faces come back unchanged.
    """
    n = len(vertices)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if n > 1 and tol > 0.0:
        distances = np.linalg.norm(vertices[:, None, :] - vertices[None, :, :], axis=-1)
        for i, j in zip(*np.nonzero(np.triu(distances < tol, k=1))):
            root_i, root_j = find(int(i)), find(int(j))
            if root_i != root_j:
                parent[max(root_i, root_j)] = min(root_i, root_j)

    cluster_of: dict[int, int] = {}
    members: list[list[int]] = []
    for i in range(n):
        root = find(i)
        if root not in cluster_of:
            cluster_of[root] = len(members)
            members.append([])
        members[cluster_of[root]].append(i)
    merged = np.array(
        [vertices[m[0]] if len(m) == 1 else vertices[m].mean(axis=0) for m in members]
    ).reshape(-1, 3)

    merged_faces = []
    for face in faces:
        ids = [cluster_of[find(i)] for i in face]
        # drop consecutive repeats around the polygon
        ids = [v for index, v in enumerate(ids) if v != ids[index - 1]] if len(ids) > 1 else ids
        if len(set(ids)) >= 3:
            merged_faces.append(ids)
    return merged, merged_faces


def get_brillouin_zone_3d(
    rec_lat: NDArray, tolerance: float = 1e-5
) -> tuple[NDArray, list, list]:
    """Construct the first Brillouin zone of a reciprocal lattice.

    The first Brillouin zone is the Wigner-Seitz cell of the reciprocal
    lattice. It is found by a Voronoi decomposition (scipy) of the 3x3x3
    block of reciprocal-lattice points around the origin: the Voronoi cell
    of the origin is the zone. The block is built on the Selling-reduced
    basis (``_selling_reduce``), for which it contains every lattice vector
    that bounds the zone, so any basis of the lattice (a sheared supercell
    matrix, a raw monoclinic cell with an obtuse angle far from 90 degrees)
    gives the same polyhedron; its volume is checked against
    ``|det rec_lat|``. Only the construction uses the reduced basis: the
    vertices are Cartesian, and ``cartesian @ inv(rec_lat)`` gives them in
    the input basis. This is the polyhedron ``crystod-bz`` draws, for the
    unit cell and, with ``--trans-mat``, for the supercell as well.

    Args:
        rec_lat: ``(3, 3)`` array whose rows are the reciprocal basis vectors
            ``b1``, ``b2``, ``b3``. Any overall scale is accepted; the command
            uses ``inv(lattice).T`` (no factor of 2 pi), for which
            ``cartesian @ inv(rec_lat)`` are fractional coordinates.
        tolerance: Corners closer than ``tolerance`` times the length of
            the shortest reciprocal lattice vector are merged into one, and
            faces that shrink to fewer than three corners are dropped, so a
            lattice that is cubic up to noise (e.g. ``1e-6`` A in the POSCAR)
            gives the cube and not a polyhedron with tiny extra faces. ``0``
            keeps every corner Qhull returns.

    Returns:
        The tuple ``(vertices, ridges, facets)`` where ``vertices`` is an
        ``(N, 3)`` array with the Cartesian coordinates of the zone corners,
        ``ridges`` is a list with one ``(M + 1, 3)`` array per facet holding
        the closed polyline of its edges (the first vertex is repeated at the
        end), and ``facets`` is the same list without the repeated vertex.

    Raises:
        RuntimeError: The polyhedron does not have the volume ``|det rec_lat|``
            of the reciprocal unit cell (not expected for a non-singular
            ``rec_lat``; the check guards the construction).

    Example:
        >>> import numpy as np
        >>> from crystod import bz
        >>> rec_lat = np.linalg.inv(4.07 * np.eye(3)).T   # cubic, a = 4.07 A
        >>> vertices, ridges, facets = bz.get_brillouin_zone_3d(rec_lat)
        >>> len(vertices), len(facets)
        (8, 6)
        >>> np.allclose(np.abs(vertices @ np.linalg.inv(rec_lat)), 0.5)
        True
        >>> sheared = np.linalg.inv(np.array([[1, 2, 0], [0, 1, 0], [0, 0, 1]])
        ...                         @ (4.07 * np.eye(3))).T   # same lattice
        >>> vertices, ridges, facets = bz.get_brillouin_zone_3d(sheared)
        >>> len(vertices), len(facets)
        (8, 6)
    """
    from scipy.spatial import ConvexHull, Voronoi

    rec_lat = np.asarray(rec_lat, dtype=float)
    assert rec_lat.shape == (3, 3)
    expected_volume = abs(float(np.linalg.det(rec_lat)))
    reduced, _ = _selling_reduce(rec_lat)

    # The 27-point block of the reduced basis is enough (_selling_reduce); the
    # 125-point block is a fallback that the volume check would call for.
    volume = float("nan")
    for shell in (1, 2):
        grid = np.mgrid[-shell : shell + 1, -shell : shell + 1, -shell : shell + 1]
        px, py, pz = np.tensordot(reduced, grid, axes=[0, 0])
        points = np.c_[px.ravel(), py.ravel(), pz.ravel()]
        # index of the central point [0, 0, 0] (13 for the 3x3x3 block)
        centre = (len(points) - 1) // 2
        vor = Voronoi(points)

        faces: list[list[int]] = []
        bz_vertices: list[int] = []
        for pid, rid in zip(vor.ridge_points, vor.ridge_vertices):
            if pid[0] == centre or pid[1] == centre:
                faces.append(list(rid))
                bz_vertices += rid
        bz_vertices = list(set(bz_vertices))
        if -1 in bz_vertices:  # pragma: no cover - the central cell is bounded
            continue
        volume = ConvexHull(vor.vertices[bz_vertices]).volume
        if abs(volume - expected_volume) <= 1e-6 * expected_volume:
            break
    else:
        raise RuntimeError(
            "the Voronoi cell of the reciprocal lattice does not have the volume "
            f"|det B| = {expected_volume:.6g} (got {volume:.6g})."
        )

    # merge corners that coincide within the tolerance (near-degenerate zones);
    # the shortest lattice vector is a subset sum of the obtuse superbase
    position = {vertex: index for index, vertex in enumerate(bz_vertices)}
    subset_sums = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0],
                            [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=float) @ reduced
    shortest = float(np.min(np.linalg.norm(subset_sums, axis=1)))
    vertices, merged_faces = _merge_close_vertices(
        vor.vertices[bz_vertices],
        [[position[vertex] for vertex in face] for face in faces],
        tolerance * shortest,
    )
    bz_facets = [vertices[face] for face in merged_faces]
    bz_ridges = [vertices[face + [face[0]]] for face in merged_faces]
    return vertices, bz_ridges, bz_facets


def _split_list(values: list, n: int):
    for i in range(0, len(values), n):
        yield values[i : i + n]


def parse_manual_band(band: str) -> list[NDArray]:
    """Parse a ``--band`` string into k-path segments.

    The string has the format of the ``crystod-bz --band`` option: continuous
    segments separated by commas, each a whitespace-separated list of
    fractional coordinates, three numbers per k point; fractions such as
    ``1/2`` are accepted. The coordinates refer to the reciprocal basis of
    the lattice the path is drawn on, which for the command is the input
    cell as given.

    Args:
        band: The path string, e.g.
            ``"0 0 0  0 1/2 0  1/2 1/2 0, 1/2 1/2 0  1/2 1/2 1/2"``.

    Returns:
        One ``(N_i, 3)`` float array per comma-separated segment, in the
        order given; empty segments (a trailing comma) are skipped.

    Raises:
        SystemExit: The number of values in a segment is not a multiple of
            3, a segment has fewer than two k points, or the string holds no
            k point at all (``ValueError`` when called through ``crystod.bz``).
        ValueError: A token is neither a number nor a fraction.

    Example:
        >>> from crystod import bz
        >>> path = "0 0 0  1/2 1/2 0, 1/2 1/2 0  1/2 1/2 1/2"
        >>> segments = bz.parse_manual_band(path)
        >>> len(segments), segments[0].tolist()
        (2, [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    """
    segments = []
    for part in band.split(","):
        tokens = part.split()
        if not tokens:
            continue
        if len(tokens) % 3 != 0:
            raise SystemExit(
                f"ERROR: --band segment '{part.strip()}' does not contain a multiple of 3 coordinates."
            )
        values = [float(Fraction(token)) for token in tokens]
        segment = np.array(list(_split_list(values, 3)), dtype=float)
        if len(segment) < 2:
            raise SystemExit(
                f"ERROR: --band segment '{part.strip()}' needs at least 2 k points."
            )
        segments.append(segment)
    if not segments:
        raise SystemExit("ERROR: --band contains no k points.")
    return segments


GREEK = {
    "GAMMA": "\u0393",
    "GM": "\u0393",
    "DELTA": "\u0394",
    "SIGMA": "\u03a3",
    "LAMBDA": "\u039b",
}


def prettify_label(label: str) -> str:
    """Convert a seekpath k-point label into its display form.

    ``crystod-bz`` places these labels next to the k-path markers of the HTML
    plot: ``GAMMA`` (or ``GM``), ``DELTA``, ``SIGMA`` and ``LAMBDA`` become the
    Greek letters, and a ``_`` suffix becomes an HTML subscript, so ``X_1``
    turns into ``X<sub>1</sub>``. Any other label is returned unchanged.

    Args:
        label: A seekpath-style label such as ``"GAMMA"``, ``"X_1"`` or
            ``"SIGMA_0"``.

    Returns:
        The label as an HTML fragment for Plotly text.

    Example:
        >>> from crystod import bz
        >>> bz.prettify_label("GAMMA"), bz.prettify_label("SIGMA_0")
        ('Γ', 'Σ<sub>0</sub>')
    """
    if "_" in label:
        stem, _, subscript = label.partition("_")
        return f"{GREEK.get(stem, stem)}<sub>{subscript}</sub>"
    return GREEK.get(label, label)


def get_seekpath_kpath(cell, tolerance: float):
    """Generate the recommended high-symmetry k path of a cell with seekpath.

    This is the automatic path of ``crystod-bz -c POSCAR`` without ``--band``.
    seekpath standardizes the cell first, so the coordinates refer to the
    reciprocal basis of the seekpath standardized primitive cell, which is
    returned alongside; when that cell differs from the input, the command
    prints a note and draws the zone for the standardized cell.

    Args:
        cell (PhonopyAtoms): The crystal structure as phonopy's
            ``PhonopyAtoms`` (any object with ``cell``, ``scaled_positions``
            and ``numbers`` attributes works).
        tolerance: Symmetry tolerance forwarded to seekpath and spglib
            (``--tolerance``; the command uses ``1e-5``).

    Returns:
        The tuple ``(segments, label_segments, primitive_lattice, symbol, number)``
        where ``segments`` is a list of ``(N, 3)`` arrays of fractional k
        coordinates, one per continuous piece of the path, ``label_segments``
        the matching lists of ``N`` seekpath labels (``GAMMA``, ``X``, ...),
        ``primitive_lattice`` the ``(3, 3)`` row-vector lattice of the
        standardized primitive cell, and ``symbol``/``number`` the
        international symbol and number of the detected space group.

    Raises:
        SystemExit: seekpath is not installed (``ValueError`` when called
            through ``crystod.bz``).

    Example:
        >>> from phonopy.interface.vasp import read_vasp
        >>> from crystod import bz
        >>> from crystod.examples import example_path
        >>> cell = read_vasp(example_path("221_PPOSCAR_ScF3"))
        >>> segments, labels, lattice, symbol, number = bz.get_seekpath_kpath(
        ...     cell, 1e-5)
        >>> symbol, number
        ('Pm-3m', 221)
        >>> labels
        [['GAMMA', 'X', 'M', 'GAMMA', 'R', 'X'], ['R', 'M']]
    """
    try:
        import seekpath
    except ImportError:
        raise SystemExit(
            "ERROR: seekpath is required for automatic k-path generation.\n"
            "       Install it with `pip install seekpath`, or supply --band/--label manually."
        )

    from .runtime_compat import get_scaled_positions

    lattice = np.array(cell.cell, dtype=float)
    positions = np.array(get_scaled_positions(cell), dtype=float)
    numbers = list(cell.numbers)

    result = seekpath.get_path((lattice, positions, numbers), symprec=tolerance)

    point_coords = result["point_coords"]
    path = result["path"]

    # Group consecutive (start, end) pairs into continuous segments.
    label_segments: list[list[str]] = []
    for start, end in path:
        if label_segments and label_segments[-1][-1] == start:
            label_segments[-1].append(end)
        else:
            label_segments.append([start, end])

    segments = [
        np.array([point_coords[label] for label in labels], dtype=float)
        for labels in label_segments
    ]
    return (
        segments,
        label_segments,
        np.array(result["primitive_lattice"], dtype=float),
        result["spacegroup_international"],
        result["spacegroup_number"],
    )


# ---------------------------------------------------------------------------
# Plotly trace construction (plain dicts; rendered via CDN plotly.js)
# ---------------------------------------------------------------------------
def build_bz_traces(
    rec_lat: NDArray,
    segments: list[NDArray] | None,
    label_segments: list[list[str]] | None,
) -> list[dict]:
    """Build the Plotly traces of a Brillouin zone with an optional k path.

    This is the figure ``crystod-bz`` writes for the unit-cell zone: the
    reciprocal basis vectors ``b1``, ``b2``, ``b3`` (red, green, blue), the
    zone edges and corners (black; hovering a corner shows its fractional
    coordinates) and, when ``segments`` is given, the k path (goldenrod)
    with a marker and label at every k point. The traces are plain
    dictionaries of ``scatter3d`` specifications, ready for
    ``plotly.graph_objects.Figure(data=traces)`` or for ``json.dumps`` into
    a page that loads plotly.js, which is what the command does.

    Args:
        rec_lat: ``(3, 3)`` reciprocal lattice, rows ``b1``, ``b2``, ``b3``
            (see ``get_brillouin_zone_3d``).
        segments: k-path segments as ``(N_i, 3)`` arrays of fractional
            coordinates in the basis ``rec_lat``, as returned by
            ``get_seekpath_kpath`` or ``parse_manual_band``; ``None`` draws
            the zone alone.
        label_segments: One list of ``N_i`` labels per segment, shown after
            ``prettify_label``; ``None`` leaves the markers unlabelled.

    Returns:
        A list of Plotly ``scatter3d`` trace dictionaries.

    Example:
        >>> import numpy as np
        >>> from crystod import bz
        >>> rec_lat = np.linalg.inv(4.07 * np.eye(3)).T
        >>> path = bz.parse_manual_band("0 0 0  1/2 0 0  1/2 1/2 0  0 0 0")
        >>> traces = bz.build_bz_traces(rec_lat, path, [["GM", "X", "M", "GM"]])
        >>> len(traces), traces[-1]["text"]
        (12, ['Γ', 'X', 'M', 'Γ'])
    """
    traces: list[dict] = []

    # Reciprocal basis vectors
    basis_colors = ["red", "green", "blue"]
    basis_labels = ["<i>b<sub>1</sub></i>", "<i>b<sub>2</sub></i>", "<i>b<sub>3</sub></i>"]
    for color, label, basis in zip(basis_colors, basis_labels, rec_lat):
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
                "textfont": {"color": color, "size": 30},
                "opacity": 0.8,
                "hoverinfo": "skip",
            }
        )

    # BZ edges and vertices
    vertices, edges, _ = get_brillouin_zone_3d(rec_lat)
    for edge in edges:
        traces.append(
            {
                "type": "scatter3d",
                "x": edge[:, 0].tolist(),
                "y": edge[:, 1].tolist(),
                "z": edge[:, 2].tolist(),
                "mode": "lines",
                "line": {"color": "black", "width": 5},
                "opacity": 0.8,
                "hoverinfo": "skip",
            }
        )
    vertices_frac = vertices @ np.linalg.inv(rec_lat)
    traces.append(
        {
            "type": "scatter3d",
            "x": vertices[:, 0].tolist(),
            "y": vertices[:, 1].tolist(),
            "z": vertices[:, 2].tolist(),
            "mode": "markers",
            "marker": {"color": "black", "size": 3},
            "customdata": vertices_frac.tolist(),
            "hovertemplate": (
                "q-position: (%{customdata[0]:.3f}, "
                "%{customdata[1]:.3f}, %{customdata[2]:.3f})<extra></extra>"
            ),
            "opacity": 1,
        }
    )

    # Band path
    if segments:
        all_points_cart: list[list[float]] = []
        all_points_frac: list[list[float]] = []
        all_labels: list[str] = []
        for index, segment in enumerate(segments):
            cartesian = segment @ rec_lat
            traces.append(
                {
                    "type": "scatter3d",
                    "x": cartesian[:, 0].tolist(),
                    "y": cartesian[:, 1].tolist(),
                    "z": cartesian[:, 2].tolist(),
                    "mode": "lines",
                    "line": {"color": "goldenrod", "width": 10},
                    "opacity": 0.8,
                    "hoverinfo": "skip",
                }
            )
            all_points_cart.extend(cartesian.tolist())
            all_points_frac.extend(segment.tolist())
            if label_segments is not None:
                all_labels.extend(prettify_label(label) for label in label_segments[index])

        points = np.array(all_points_cart, dtype=float)
        marker_trace = {
            "type": "scatter3d",
            "x": points[:, 0].tolist(),
            "y": points[:, 1].tolist(),
            "z": points[:, 2].tolist(),
            "mode": "markers+text" if all_labels else "markers",
            "marker": {"color": "red", "size": 3},
            "textfont": {"color": "black", "size": 25},
            "customdata": all_points_frac,
            "hovertemplate": (
                "q-position: (%{customdata[0]:.3f}, "
                "%{customdata[1]:.3f}, %{customdata[2]:.3f})<extra></extra>"
            ),
            "opacity": 1,
        }
        if all_labels:
            marker_trace["text"] = all_labels
        traces.append(marker_trace)

    return traces


def write_html(traces: list[dict], output: str, title: str) -> None:
    """Write Plotly traces into a stand-alone HTML page (plotly.js from the CDN)."""
    layout = {
        "title": {"text": title},
        "showlegend": False,
        "scene": {
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "aspectmode": "data",
        },
        "margin": {"l": 0, "r": 0, "t": 40, "b": 0},
    }
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
</head>
<body>
<div id="plot" style="width:100vw;height:95vh;"></div>
<script>
var data = {json.dumps(traces)};
var layout = {json.dumps(layout)};
Plotly.newPlot("plot", data, layout, {{responsive: true}});
</script>
</body>
</html>
"""
    with open(output, "w") as handle:
        handle.write(html)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.label and not args.band:
        parser.error("--label requires --band.")

    from .star_of_k import read_poscar_or_exit

    cell = read_poscar_or_exit(args.poscar)
    input_lattice = np.array(cell.cell, dtype=float)

    if args.band:
        # Manual path: coordinates refer to the reciprocal basis of the input POSCAR.
        segments = parse_manual_band(args.band)
        label_segments = None
        if args.label:
            labels = args.label.split()
            total_points = sum(len(segment) for segment in segments)
            if len(labels) != total_points:
                raise SystemExit(
                    f"ERROR: --label has {len(labels)} labels but --band has {total_points} k points."
                )
            label_segments = []
            cursor = 0
            for segment in segments:
                label_segments.append(labels[cursor : cursor + len(segment)])
                cursor += len(segment)
        plot_lattice = input_lattice
        title = f"First Brillouin zone: {os.path.basename(args.poscar)}"
        print("\n* Band path *")
        print(f"Manual band path with {len(segments)} segment(s).")
    else:
        segments, label_segments, primitive_lattice, sg_symbol, sg_number = get_seekpath_kpath(
            cell, args.tolerance
        )
        plot_lattice = primitive_lattice
        title = f"First Brillouin zone: {os.path.basename(args.poscar)} — {sg_symbol} (#{sg_number})"

        print("\n* Structure *")
        print(f"Space group: {sg_symbol} (#{sg_number})")
        if not np.allclose(primitive_lattice, input_lattice, atol=1e-4):
            print(
                "NOTE: the input cell differs from the seekpath standardized primitive cell;\n"
                "      the BZ and k-path are drawn for the standardized primitive cell."
            )
        print("\n* Recommended k-path (seekpath) *")
        seen: set[str] = set()
        for labels, segment in zip(label_segments, segments):
            for label, coords in zip(labels, segment):
                if label not in seen:
                    seen.add(label)
                    print(
                        f"  {label:<8s} ({coords[0]: .4f}, {coords[1]: .4f}, {coords[2]: .4f})"
                    )
        path_text = "   ".join("-".join(labels) for labels in label_segments)
        print(f"  Path: {path_text}")

    # Koiso convention: reciprocal lattice without the 2*pi factor.
    rec_lat = np.linalg.inv(plot_lattice).T

    traces = build_bz_traces(rec_lat, segments, label_segments)

    output = args.output
    if output is None:
        output = f"BZ_{os.path.basename(args.poscar)}.html"
    write_html(traces, output, title)
    print("\n* Output files *")
    print(f"  Wrote Brillouin-zone visualization: {output}")


if __name__ == "__main__":
    main()
