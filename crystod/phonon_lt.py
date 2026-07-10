"""
Longitudinal/transverse-resolved phonon band workflow for crystod.

Computes the phonon band structure with eigenvectors directly from
POSCAR + FORCE_SETS (or FORCE_CONSTANTS) along an automatic seekpath
high-symmetry k-path and colors each band by its longitudinal character
(red = longitudinal, blue = transverse). Based on script/LT_phonon_band.py
maintained by Hiroki Koiso, after Qijing Zheng
(http://staff.ustc.edu.cn/~zqj/posts/Phonopy-Rutile-TiO2/).
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

from .phonon_fatband import compute_band_structure


class MyHelpFormatter(
    RawTextHelpFormatter,
    RawDescriptionHelpFormatter,
    ArgumentDefaultsHelpFormatter,
):
    pass


desc = """
Plot the phonon band structure colored by longitudinal/transverse character
(red = longitudinal, blue = transverse), from POSCAR + FORCE_SETS
(or FORCE_CONSTANTS with --readfc). The high-symmetry k-path is generated
automatically with seekpath (or given manually with --band/--label).

# Command Examples:
crystod-phonon --lt -c 221_PPOSCAR_ScF3 --dim 4 4 4
crystod-phonon --lt -c 221_PPOSCAR_ScF3 --dim 4 4 4 --nac
"""


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
        "--output",
        default=None,
        help="Output PDF path (default phonon_band_LT.pdf, or phonon_band_LT_nac.pdf with --nac).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Symmetry tolerance forwarded to seekpath.",
    )
    return parser


def get_longitudinal_ratio(
    qpoints: NDArray[np.float64],
    eigenvectors: NDArray[np.complex128],
    reciprocal_lattice: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Longitudinal character per (q-point, band): sqrt(sum_atoms |q_hat . e_atom|^2).

    1 for a purely longitudinal mode, 0 for a purely transverse one;
    0.5 (neutral) at the Gamma point where no propagation direction exists.
    """
    n_q, n_dof, n_bands = eigenvectors.shape
    per_atom = eigenvectors.reshape(n_q, n_dof // 3, 3, n_bands)
    ratio = np.full((n_q, n_bands), 0.5)
    for iq in range(n_q):
        q_cart = qpoints[iq] @ reciprocal_lattice
        q_norm = np.linalg.norm(q_cart)
        if q_norm < 1e-10:
            continue
        q_hat = q_cart / q_norm
        longitudinal = np.einsum("acb,c->ab", per_atom[iq], q_hat)  # (n_atoms, n_bands)
        ratio[iq] = np.linalg.norm(longitudinal, axis=0)
    return ratio


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    phonon, band, ticks, tick_labels = compute_band_structure(args)
    primitive = phonon.primitive
    # Koiso convention: reciprocal lattice without the 2*pi factor (direction only).
    reciprocal_lattice = np.linalg.inv(np.array(primitive.cell)).T

    distances = [np.array(d) for d in band["distances"]]
    frequencies = [np.array(f) for f in band["frequencies"]]
    ratios = [
        get_longitudinal_ratio(np.array(q), np.array(e), reciprocal_lattice)
        for q, e in zip(band["qpoints"], band["eigenvectors"])
    ]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.ticker import AutoMinorLocator
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    cm = 1 / 2.54
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.direction"] = "in"

    fig = plt.figure(figsize=(11.07 * cm, 8.31 * cm), dpi=480, facecolor="w")
    ax = plt.subplot()

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    mappable = matplotlib.cm.ScalarMappable(cmap="bwr", norm=norm)
    mappable.set_array(np.concatenate([r.ravel() for r in ratios]))

    for sub_distances, sub_frequencies, sub_ratios in zip(distances, frequencies, ratios):
        for band_index in range(sub_frequencies.shape[1]):
            x = sub_distances
            y = sub_frequencies[:, band_index]
            z = sub_ratios[:, band_index]
            ax.plot(x, y, lw=1.0, color="k", alpha=0.6)
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            collection = LineCollection(
                segments,
                colors=[mappable.to_rgba(value) for value in (z[1:] + z[:-1]) / 2.0],
            )
            collection.set_linewidth(1.0)
            ax.add_collection(collection)

    all_distances = np.concatenate(distances)
    ax.set_xlim(float(all_distances.min()), float(all_distances.max()))
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(labelsize=8, width=0.5)
    ax.set_ylabel("Frequency (THz)", labelpad=5, fontsize=9)
    if args.nac:
        ax.set_title("L/T character (NAC)", fontsize=9)
    ax.axhline(y=0, linestyle="--", color="black", lw=0.5)
    for tick in ticks[1:-1]:
        ax.axvline(x=tick, ls="dotted", color="black", alpha=0.8, lw=0.5)

    divider = make_axes_locatable(ax)
    ax_cbar = divider.append_axes("right", size="3%", pad=0.02)
    cbar = plt.colorbar(mappable, cax=ax_cbar, ticks=[0, 1])
    cbar.set_ticklabels(["T", "L"])
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout(pad=0.5)
    if args.output:
        output_path = args.output
    else:
        output_path = "phonon_band_LT_nac.pdf" if args.nac else "phonon_band_LT.pdf"
    plt.savefig(output_path)
    plt.close(fig)
    print(f"L/T-resolved phonon band written to: {output_path}")


if __name__ == "__main__":
    main()
