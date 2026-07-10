"""
Element-projected phonon fatband workflow for crystod.

Computes the phonon band structure with eigenvectors directly from
POSCAR + FORCE_SETS (or FORCE_CONSTANTS) along an automatic seekpath
high-symmetry k-path and plots one fatband per element, where the dot
size is the element-projected phonon density (sum of |eigenvector|^2
over the element's atoms). Plotting style based on
script/phonon_fatband.py by Hiroki Koiso.
"""

from __future__ import annotations

from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    RawDescriptionHelpFormatter,
    RawTextHelpFormatter,
)

import numpy as np
from numpy.typing import NDArray

from .spglib_compat import ensure_spglib_compat

ensure_spglib_compat()

from phonopy import load
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

from .brillouin_zone import get_seekpath_kpath, parse_manual_band
from .phonon_vector import DEFAULT_ELEMENT, VESTA_ELEMENTS


class MyHelpFormatter(
    RawTextHelpFormatter,
    RawDescriptionHelpFormatter,
    ArgumentDefaultsHelpFormatter,
):
    pass


desc = """
Plot element-projected phonon fatbands from POSCAR + FORCE_SETS
(or FORCE_CONSTANTS with --readfc). The high-symmetry k-path is generated
automatically with seekpath (or given manually with --band/--label) and one
fatband_<element>.pdf is written per element.

# Command Examples:
crystod-phonon --fatband -c 221_PPOSCAR_ScF3 --dim 4 4 4
crystod-phonon --fatband -c 221_PPOSCAR_ScF3 --dim 4 4 4 --element F
"""

GREEK_MPL = {"GAMMA": r"$\Gamma$", "DELTA": r"$\Delta$", "SIGMA": r"$\Sigma$", "LAMBDA": r"$\Lambda$"}


def _element_plot_style(symbol: str) -> tuple[tuple[float, float, float], float]:
    """VESTA default color of the element and a matching scatter alpha.

    Pale or grayish VESTA colors (high luminance) get a higher opacity so the
    fatband stays visible against the black band lines.
    """
    _, red, green, blue = VESTA_ELEMENTS.get(symbol, DEFAULT_ELEMENT)
    color = (red / 255, green / 255, blue / 255)
    luminance = (0.299 * red + 0.587 * green + 0.114 * blue) / 255
    saturation = (max(red, green, blue) - min(red, green, blue)) / max(max(red, green, blue), 1)
    alpha = 0.1
    alpha += 0.25 * max(0.0, luminance - 0.5) / 0.5  # pale colors
    alpha += 0.15 * max(0.0, (1.0 - saturation) - 0.7) / 0.3  # grayish colors
    return color, min(alpha, 0.5)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=desc, formatter_class=MyHelpFormatter)
    parser.add_argument(
        "--dim",
        required=True,
        type=str,
        help="Supercell dimension used for the force calculation.",
    )
    parser.add_argument(
        "--poscar",
        type=str,
        default="POSCAR",
        help="POSCAR path.",
    )
    parser.add_argument(
        "--readfc",
        action="store_true",
        help="Read FORCE_CONSTANTS instead of FORCE_SETS.",
    )
    parser.add_argument(
        "--nac",
        action="store_true",
        help="Apply the non-analytical term correction (LO/TO splitting) using a BORN file.",
    )
    parser.add_argument(
        "--element",
        default=None,
        help="Plot only this element (default: one fatband per element).",
    )
    parser.add_argument(
        "--band",
        default=None,
        help='Optional manual band path, e.g. "0 0 0  0 1/2 0  1/2 1/2 0, 1/2 1/2 0  1/2 1/2 1/2".',
    )
    parser.add_argument(
        "--label",
        default=None,
        help='Optional labels for the manual band path, e.g. "GM X M M R".',
    )
    parser.add_argument(
        "--npoints",
        type=int,
        default=51,
        help="Number of q-points per band-path segment leg.",
    )
    parser.add_argument(
        "--projection-direction",
        dest="direction",
        type=str,
        default=None,
        help='Optional projection direction in reduced coordinates, e.g. "0 0 1".',
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output prefix (default 'fatband', giving fatband_<element>.pdf).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Symmetry tolerance forwarded to seekpath.",
    )
    return parser


def _prettify_label_mpl(label: str) -> str:
    if "_" in label:
        stem, _, subscript = label.partition("_")
        return f"{GREEK_MPL.get(stem, stem)}$_{{{subscript}}}$"
    return GREEK_MPL.get(label, label)


def _build_tick_positions(
    distances: list[NDArray[np.float64]],
    path_connections: list[bool],
    label_segments: list[list[str]],
) -> tuple[list[float], list[str]]:
    """Tick positions/labels at every band-path vertex; 'X|M' at discontinuities."""
    flat_labels = [label for segment in label_segments for label in segment]
    ticks: list[float] = []
    tick_labels: list[str] = []
    label_index = 0
    for sub_index, sub_distances in enumerate(distances):
        start, end = float(sub_distances[0]), float(sub_distances[-1])
        if not ticks or start > ticks[-1] + 1e-10:
            # new continuous segment starts here
            ticks.append(start)
            tick_labels.append(_prettify_label_mpl(flat_labels[label_index]))
        label_index += 1
        connected = sub_index < len(path_connections) and path_connections[sub_index]
        ticks.append(end)
        if connected:
            tick_labels.append(_prettify_label_mpl(flat_labels[label_index]))
        else:
            # discontinuity (or path end): may need combined 'X|M' label
            if label_index + 1 < len(flat_labels) and sub_index + 1 < len(distances):
                left = _prettify_label_mpl(flat_labels[label_index])
                right = _prettify_label_mpl(flat_labels[label_index + 1])
                tick_labels.append(left if left == right else f"{left}|{right}")
                label_index += 1
            else:
                tick_labels.append(_prettify_label_mpl(flat_labels[label_index]))
    return ticks, tick_labels


def _plot_fatband(
    element: str,
    title: str,
    color: tuple[float, float, float],
    alpha: float,
    distances: list[NDArray[np.float64]],
    frequencies: list[NDArray[np.float64]],
    projections: list[NDArray[np.float64]],
    ticks: list[float],
    tick_labels: list[str],
    n_element_atoms: int,
    output_path: str,
) -> None:
    """One fatband figure (style based on Koiso's phonon_fatband.py)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    cm = 1 / 2.54
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.direction"] = "in"

    fig = plt.figure(figsize=(7.5 * cm, 6.0 * cm), dpi=480, facecolor="w")
    ax = plt.subplot()
    ax.set_title(title, fontsize=9)

    all_distances = np.concatenate(distances)
    ax.set_xlim(float(all_distances.min()), float(all_distances.max()))
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(labelsize=7, width=0.5)
    ax.set_ylabel("Frequency (THz)", labelpad=5, fontsize=8)
    ax.axhline(y=0, linestyle="--", color="black", lw=0.5)
    for side in ("top", "bottom", "left", "right"):
        ax.spines[side].set_linewidth(0.5)
    for tick in ticks[1:-1]:
        ax.axvline(x=tick, ls="dotted", color="black", lw=0.5)

    for sub_distances, sub_frequencies, sub_projections in zip(distances, frequencies, projections):
        for band in range(sub_frequencies.shape[1]):
            ax.plot(sub_distances, sub_frequencies[:, band], lw=0.5, color="black")
            ax.scatter(
                sub_distances,
                sub_frequencies[:, band],
                s=sub_projections[:, band] * 15 / n_element_atoms,
                color=color,
                alpha=alpha,
            )

    plt.tight_layout(pad=0.5)
    plt.savefig(output_path)
    plt.close(fig)


def compute_band_structure(args):
    """Load phonopy data, resolve the k-path, and run the band structure
    with eigenvectors and band connection.

    `args` must carry dim/poscar/readfc/nac/band/label/npoints/tolerance.
    Returns (phonon, band_dict, ticks, tick_labels) where band_dict is the
    phonopy band-structure dict (distances/frequencies/eigenvectors/qpoints
    as lists over continuous sub-paths).
    """
    supercell_mat = [float(n) for n in args.dim.split()]
    if args.readfc:
        force_sets = None
        force_constants = "./FORCE_CONSTANTS"
    else:
        force_sets = "./FORCE_SETS"
        force_constants = None

    # NAC is controlled explicitly: phonopy.load defaults to is_nac=True and
    # would silently pick up a BORN file in the working directory otherwise.
    if args.nac:
        import os

        if not os.path.isfile("BORN"):
            raise SystemExit(
                "ERROR: --nac requires a BORN file (Born effective charges and dielectric\n"
                "       tensor) in the current directory, e.g. generated with phonopy-vasp-born."
            )

    phonon = load(
        supercell_matrix=supercell_mat,
        primitive_matrix="auto",
        unitcell_filename=args.poscar,
        force_sets_filename=force_sets,
        force_constants_filename=force_constants,
        is_nac=args.nac,
        born_filename="BORN" if args.nac else None,
    )
    if args.nac:
        print("NAC (LO/TO splitting) enabled: Born effective charges read from BORN.")
    primitive = phonon.primitive
    symbols = list(primitive.symbols)

    # band path: manual --band/--label, or automatic seekpath k-path
    if args.band:
        segments = parse_manual_band(args.band)
        if args.label:
            labels_flat = args.label.split()
            expected = sum(len(segment) for segment in segments)
            if len(labels_flat) != expected:
                raise SystemExit(
                    f"ERROR: --label needs {expected} labels for this --band path, "
                    f"but {len(labels_flat)} were given."
                )
            label_segments = []
            cursor = 0
            for segment in segments:
                label_segments.append(labels_flat[cursor : cursor + len(segment)])
                cursor += len(segment)
        else:
            label_segments = [[""] * len(segment) for segment in segments]
    else:
        segments, label_segments, seekpath_lattice, spacegroup, spacegroup_number = (
            get_seekpath_kpath(primitive, args.tolerance)
        )
        print(f"Space group: {spacegroup} (#{spacegroup_number})")
        path_text = "  ".join(
            "-".join(_prettify_label_mpl(label) for label in labels) for labels in label_segments
        )
        print(f"k-path (seekpath): {path_text}")
        if not np.allclose(np.array(primitive.cell), seekpath_lattice, atol=1e-4):
            print(
                "NOTE: the input cell differs from the seekpath standardized primitive cell;\n"
                "      the k-path coordinates refer to the standardized primitive cell."
            )

    band_paths = [np.array(segment, dtype=float) for segment in segments]
    qpoints, path_connections = get_band_qpoints_and_path_connections(
        band_paths, npoints=args.npoints
    )
    print(f"\nComputing phonon band structure with eigenvectors "
          f"({sum(len(q) for q in qpoints)} q-points)...")
    phonon.run_band_structure(
        qpoints,
        path_connections=path_connections,
        with_eigenvectors=True,
        is_band_connection=True,
    )
    band = phonon.get_band_structure_dict()
    distances = [np.array(d) for d in band["distances"]]
    ticks, tick_labels = _build_tick_positions(distances, list(path_connections), label_segments)
    return phonon, band, ticks, tick_labels


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    phonon, band, ticks, tick_labels = compute_band_structure(args)
    primitive = phonon.primitive
    symbols = list(primitive.symbols)
    distances = [np.array(d) for d in band["distances"]]
    frequencies = [np.array(f) for f in band["frequencies"]]
    eigenvectors = band["eigenvectors"]

    # optional projection direction (reduced coordinates -> Cartesian unit vector)
    direction = None
    if args.direction is not None:
        fractional = np.array(args.direction.split(), dtype=float)
        cartesian = fractional @ np.array(primitive.cell)
        direction = cartesian / np.linalg.norm(cartesian)
        print(f"Projection direction (Cartesian): {np.round(direction, 6).tolist()}")

    # per-atom projected weight: sum_axes |e_atom|^2 (optionally along `direction`)
    atom_weights = []
    for sub_eigenvectors in eigenvectors:
        sub = np.array(sub_eigenvectors)  # (n_q, 3N, n_bands), eigenvector in columns
        n_q, n_dof, n_bands = sub.shape
        per_atom = sub.reshape(n_q, n_dof // 3, 3, n_bands)
        if direction is not None:
            projected = np.einsum("qacb,c->qab", per_atom, direction)
            weights = np.abs(projected) ** 2
        else:
            weights = np.sum(np.abs(per_atom) ** 2, axis=2)
        atom_weights.append(weights)  # (n_q, n_atoms, n_bands)

    # elements to plot
    unique_elements = []
    for symbol in symbols:
        if symbol not in unique_elements:
            unique_elements.append(symbol)
    if args.element is not None:
        if args.element not in unique_elements:
            raise SystemExit(
                f"ERROR: element '{args.element}' is not in this compound ({', '.join(unique_elements)})."
            )
        unique_elements = [args.element]

    if args.output:
        prefix = args.output
    else:
        prefix = "fatband_nac" if args.nac else "fatband"

    for element in unique_elements:
        atom_indices = [i for i, symbol in enumerate(symbols) if symbol == element]
        projections = [weights[:, atom_indices, :].sum(axis=1) for weights in atom_weights]
        color, alpha = _element_plot_style(element)
        output_path = f"{prefix}_{element}.pdf"
        _plot_fatband(
            element=element,
            title=f"{element} (NAC)" if args.nac else element,
            color=color,
            alpha=alpha,
            distances=distances,
            frequencies=frequencies,
            projections=projections,
            ticks=ticks,
            tick_labels=tick_labels,
            n_element_atoms=len(atom_indices),
            output_path=output_path,
        )
        print(f"Fatband for {element} written to: {output_path}")


if __name__ == "__main__":
    main()
