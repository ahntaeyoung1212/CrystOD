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
crystod --bz --poscar 221_PPOSCAR_ScF3
crystod --bz --poscar 221_PPOSCAR_ScF3 --output BZ_ScF3_Pm-3m.html
crystod --bz --poscar 221_PPOSCAR_ScF3 \\
    --band "0 0 0  0 1/2 0  1/2 1/2 0  0 0 0  1/2 1/2 1/2  0 1/2 0, 1/2 1/2 0  1/2 1/2 1/2" \\
    --label "GM X M GM R X  M R"
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
def get_brillouin_zone_3d(rec_lat: NDArray) -> tuple[NDArray, list, list]:
    """Construct the first Brillouin zone (Wigner-Seitz cell of the
    reciprocal lattice) by Voronoi decomposition.

    Parameters
    ----------
    rec_lat : ndarray, shape=(3, 3)
        Reciprocal lattice row vectors [b1, b2, b3]^T.

    Returns
    -------
    vertices : ndarray
        Cartesian coordinates of the BZ vertices.
    ridges : list of ndarray
        Closed polylines (edges) of each BZ facet.
    facets : list of ndarray
        Vertices of each BZ facet.
    """
    from scipy.spatial import Voronoi

    rec_lat = np.asarray(rec_lat, dtype=float)
    assert rec_lat.shape == (3, 3)

    px, py, pz = np.tensordot(rec_lat, np.mgrid[-1:2, -1:2, -1:2], axes=[0, 0])
    points = np.c_[px.ravel(), py.ravel(), pz.ravel()]
    vor = Voronoi(points)

    bz_facets = []
    bz_ridges = []
    bz_vertices: list[int] = []
    # Index 13 is the central point [0, 0, 0] of the 3x3x3 lattice grid.
    for pid, rid in zip(vor.ridge_points, vor.ridge_vertices):
        if pid[0] == 13 or pid[1] == 13:
            bz_ridges.append(vor.vertices[np.r_[rid, [rid[0]]]])
            bz_facets.append(vor.vertices[rid])
            bz_vertices += rid

    bz_vertices = list(set(bz_vertices))
    return vor.vertices[bz_vertices], bz_ridges, bz_facets


def _split_list(values: list, n: int):
    for i in range(0, len(values), n):
        yield values[i : i + n]


def parse_manual_band(band: str) -> list[NDArray]:
    """Parse a --band string into a list of (N_i, 3) fractional-coordinate arrays."""
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
    """Convert seekpath-style labels (GAMMA, X_1, SIGMA_0) into display form."""
    if "_" in label:
        stem, _, subscript = label.partition("_")
        return f"{GREEK.get(stem, stem)}<sub>{subscript}</sub>"
    return GREEK.get(label, label)


def get_seekpath_kpath(cell, tolerance: float):
    """Run seekpath on a PhonopyAtoms cell and return
    (segments, label_segments, primitive_lattice, spacegroup_symbol, spacegroup_number).

    Each segment is an (N, 3) array of fractional coordinates in the
    reciprocal basis of the seekpath standardized primitive cell; the
    corresponding label segment is a list of N seekpath labels.
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
        print(f"Manual band path with {len(segments)} segment(s).")
    else:
        segments, label_segments, primitive_lattice, sg_symbol, sg_number = get_seekpath_kpath(
            cell, args.tolerance
        )
        plot_lattice = primitive_lattice
        title = f"First Brillouin zone: {os.path.basename(args.poscar)} — {sg_symbol} (#{sg_number})"

        print(f"Space group: {sg_symbol} (#{sg_number})")
        if not np.allclose(primitive_lattice, input_lattice, atol=1e-4):
            print(
                "NOTE: the input cell differs from the seekpath standardized primitive cell;\n"
                "      the BZ and k-path are drawn for the standardized primitive cell."
            )
        print("\nRecommended k-path (seekpath):")
        seen: set[str] = set()
        for labels, segment in zip(label_segments, segments):
            for label, coords in zip(labels, segment):
                if label not in seen:
                    seen.add(label)
                    print(
                        f"  {label:<8s} ({coords[0]: .4f}, {coords[1]: .4f}, {coords[2]: .4f})"
                    )
        path_text = "   ".join("-".join(labels) for labels in label_segments)
        print(f"\nPath: {path_text}")

    # Koiso convention: reciprocal lattice without the 2*pi factor.
    rec_lat = np.linalg.inv(plot_lattice).T

    traces = build_bz_traces(rec_lat, segments, label_segments)

    output = args.output
    if output is None:
        output = f"BZ_{os.path.basename(args.poscar)}.html"
    write_html(traces, output, title)
    print(f"\nWrote Brillouin-zone visualization: {output}")


if __name__ == "__main__":
    main()
