"""
__author__ = "Hiroki Koiso, Yasuhide Mochizuki"
__copyright__ = "Copyright 2026, Mochizuki group"
__version__ = "1.0"
__maintainer__ = "Hiroki Koiso, Yasuhide Mochizuki"
__email__ = "mochizuki@rs.tus.ac.jp"
__status__ = "Development"
__released_date__ = "November 2, 2024"
__last_update__= "June 29, 2026"
"""

from __future__ import annotations

from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    RawDescriptionHelpFormatter,
    RawTextHelpFormatter,
)

import numpy as np
import warnings
from numpy.typing import NDArray

from .spglib_compat import ensure_spglib_compat

ensure_spglib_compat()

from phonopy import load
from phonopy.structure.cells import get_primitive_matrix_by_centring

from .irreptables_compat import load_irreptables
from .operations import snap_qpoint
from .runtime_compat import get_symmetry_dataset

IrrepTable, Irrep = load_irreptables()


class MyHelpFormatter(
    RawTextHelpFormatter,
    RawDescriptionHelpFormatter,
    ArgumentDefaultsHelpFormatter,
):
    pass


desc = """
This program identify the CDML notations for the phonon irreducible representations.
POSCAR and FORCE_STES must exist in the directory where this code runs.

# Command Example:
python3 phonon_irreps.py --dim "2 2 2"
"""


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=desc, formatter_class=MyHelpFormatter)
    parser.add_argument(
        "--dim",
        "-dim",
        dest="dim",
        required=True,
        type=str,
        help="Supercell dimension.",
    )
    parser.add_argument(
        "--poscar",
        "-poscar",
        dest="poscar",
        type=str,
        default="POSCAR",
        help="POSCAR.",
    )
    parser.add_argument(
        "--readfc",
        "-readfc",
        dest="readfc",
        action="store_true",
        help="Read FORCE_CONSTANS.",
    )
    parser.add_argument(
        "--tolerance",
        "-tol",
        dest="tol",
        type=float,
        default=1e-3,
        help="Degeneracy tolerance.",
    )
    parser.add_argument(
        "--all-irreps",
        dest="all_irreps",
        action="store_true",
        help="Additionally label the phonon irreps at the midpoints of the\n"
        "seekpath k-path segments (the symmetry lines DT, Z, SM, ...;\n"
        "ISO-IR labels). Slower than the default special-points-only survey.",
    )
    return parser


def format_qpoint(q, decimals: int = 6) -> list[float]:
    """Format a q-point for yaml output using plain Python floats."""
    return [float(np.round(float(value), decimals)) for value in q]


def get_irt_special_points(irt_table, prim_mat) -> tuple[list[str], list[list[float]]]:
    """Get unique special q-points from irreptables in primitive basis.

    Coordinates are snapped to exact fractions (1/3 stays 1/3, not 0.333333):
    decimal-rounded values break the little-group detection and irreptables
    lookups downstream.
    """
    q_list = []
    q_names = []
    for irrep in irt_table.irreps:
        q_primitive = snap_qpoint(np.dot(irrep.k, prim_mat))
        if q_primitive not in q_list:
            q_list.append(q_primitive)
            q_names.append(irrep.kpname)
    return q_names, q_list


def find_star_representative(
    qpoint: list[float] | NDArray[np.float64],
    rotations: NDArray[np.int_],
    q_names: list[str],
    q_list: list[list[float]],
) -> tuple[str, list[float]] | None:
    """Map q onto the tabulated arm of its star.

    irreptables lists only one representative arm per special point (e.g. only
    (1/2, 1/2, 0) for the three M arms of Pm-3m), so a direct coordinate lookup
    fails for the other arms. Returns (label, representative q) when some
    space-group rotation sends q onto a tabulated point (k' = k R, modulo
    reciprocal-lattice translations); None otherwise. ``rotations`` must be in
    the same (primitive) basis as q and the tabulated points.
    """
    qpoint = np.asarray(qpoint, dtype=float)
    for name, q_special in zip(q_names, q_list):
        target = np.asarray(q_special, dtype=float)
        for rotation in rotations:
            diff = qpoint @ rotation - target
            if (np.abs(diff - np.rint(diff)) < 1e-8).all():
                return name, list(q_special)
    return None


def get_irt_irreps_at_q(
    q: list[float], irt_table, prim_mat, warn: bool = True
) -> list[Irrep]:
    """Get irreps at the q-point from irreptables."""
    irreps_at_q = []
    prim_inv = np.linalg.inv(prim_mat)
    conventional_q = np.array(q) @ prim_inv
    for irrep_at_q in irt_table.irreps:
        if np.allclose(irrep_at_q.k, conventional_q):
            irreps_at_q.append(irrep_at_q)
    if not irreps_at_q and warn:
        warnings.warn(f"No irreps at {q} in irreptables!", stacklevel=2)
    return irreps_at_q


def get_mapping_to_irt(
    irt_little_r: NDArray[np.int_],
    found_little_r: NDArray[np.int_],
    prim_mat,
) -> list[int]:
    """Get mapping from phonopy little-group rotations to irreptables order."""
    conv_little_r = prim_mat @ found_little_r @ np.linalg.inv(prim_mat)
    mapping_to_irt = []
    for irt_r in irt_little_r:
        for i, r in enumerate(conv_little_r):
            if np.allclose(irt_r, r):
                mapping_to_irt.append(i)
                break
    return mapping_to_irt


def _get_isoir_band_labels(
    q: list[float],
    phonon,
    phonon_irreps,
) -> list[list[str] | None] | None:
    """ISO-IR (Miller-Love) labels per degenerate band set, or None.

    Fallback for q points absent from the irreptables (BCS) tables; the
    phonopy band-set characters are decomposed against the ISO-IR small
    irreps (they can be reducible under accidental degeneracy).
    """
    from .isoir import get_isoir_band_decompositions

    primitive = phonon.primitive
    cell = (primitive.cell, primitive.scaled_positions, primitive.numbers)
    dataset = get_symmetry_dataset(phonon.primitive_symmetry)
    rotations = getattr(phonon_irreps, "_rotations_at_q")
    translations = getattr(phonon_irreps, "_translations_at_q")
    decompositions = get_isoir_band_decompositions(
        dataset["number"],
        cell,
        phonon.primitive_symmetry.tolerance,
        q,
        rotations,
        translations,
        list(phonon_irreps.characters),
    )
    labels: list[list[str] | None] = [
        None if decomposed is None
        else [f"{label}({dim})" for label, _, dim in decomposed[0]]
        for decomposed in decompositions
    ]
    return labels if any(label is not None for label in labels) else None


def get_irrep_labels(
    q: list[float],
    phonon,
    irt_table,
    prim_mat,
    degeneracy_tolerance: float,
) -> tuple[list[list[str] | None], list[list[int]], NDArray[np.float64]]:
    """Get irrep labels, band indices, and frequencies at q."""
    phonon.set_irreps(q=np.array(q), degeneracy_tolerance=degeneracy_tolerance)
    phonon_irreps = phonon.irreps
    irt_irreps = get_irt_irreps_at_q(np.array(q), irt_table, prim_mat, warn=False)

    if not irt_irreps:
        # Not in irreptables (e.g. a symmetry line/plane or generic q):
        # fall back to the ISO-IR (ISOTROPY) tables, which cover every
        # k-vector type.  Labels then follow the Miller-Love convention.
        isoir_labels = _get_isoir_band_labels(q, phonon, phonon_irreps)
        if isoir_labels is not None:
            band_indices = phonon_irreps.band_indices
            frequencies = getattr(phonon_irreps, "frequencies", None)
            if frequencies is None:
                frequencies = getattr(phonon_irreps, "_freqs")
            return isoir_labels, band_indices, frequencies
        # only warn when the ISO-IR fallback could not label the q point
        warnings.warn(f"No irreps at {q} in irreptables!", stacklevel=2)
        raise ValueError(f"no irrep labels available at {q}")

    irt_little_r = [irt_table.symmetries[i - 1].R for i in irt_irreps[0].characters.keys()]
    phonon_little_r = getattr(phonon_irreps, "_rotations_at_q")
    mapping_to_irt = get_mapping_to_irt(irt_little_r, phonon_little_r, prim_mat)

    band_indices = phonon_irreps.band_indices
    frequencies = getattr(phonon_irreps, "frequencies", None)
    if frequencies is None:
        # phonopy >= 2.21 dropped the public property; fall back to the internal array.
        frequencies = getattr(phonon_irreps, "_freqs")
    phonon_irreps_characters = phonon_irreps.characters

    labels: list[list[str] | None] = []
    for phonon_irrep_charac in phonon_irreps_characters:
        found = False
        label = []
        for irt_irrep in irt_irreps:
            irt_irrep_character = np.array(list(irt_irrep.characters.values()))
            overlap = np.dot(
                phonon_irrep_charac[mapping_to_irt],
                np.conjugate(irt_irrep_character),
            ) / irt_irrep.nsym
            if overlap > 0.9:
                label.append(f"{irt_irrep.name}({irt_irrep.dim})")
                found = True
        labels.append(label if found else None)

    assert len(labels) == len(band_indices)
    return labels, band_indices, frequencies


def _seekpath_path_midpoints(phonon) -> tuple[str | None, list[tuple[str, str, list[float]]]]:
    """Midpoints of the seekpath k-path segments, labeled via ISO-IR.

    For every segment of the automatic seekpath k-path (e.g. GM-X of Pm-3m)
    the midpoint of the two endpoint coordinates is computed (a point on the
    connecting symmetry line, e.g. DT (0, 1/4, 0)) and labeled with its
    ISO-IR k-vector-type letter.  Returns (path string, [(label, segment,
    midpoint), ...]); midpoints are skipped with a warning when the seekpath
    primitive cell does not match the phonopy primitive cell.
    """
    import seekpath

    primitive = phonon.primitive
    cell = (primitive.cell, primitive.scaled_positions, primitive.numbers)
    path_data = seekpath.get_path(cell, symprec=1e-5)
    if not np.allclose(primitive.cell, path_data["primitive_lattice"], atol=1e-4):
        warnings.warn(
            "The seekpath primitive cell does not match the phonopy primitive "
            "cell; k-path midpoints are skipped.",
            stacklevel=2,
        )
        return None, []

    def display(name: str) -> str:
        return "GM" if name == "GAMMA" else name

    coords = path_data["point_coords"]
    segments = path_data["path"]

    # compress consecutive segments into a path string like GM-X-M-GM-R-X | R-M
    parts: list[list[str]] = []
    for start, end in segments:
        if parts and parts[-1][-1] == start:
            parts[-1].append(end)
        else:
            parts.append([start, end])
    path_string = " | ".join("-".join(display(n) for n in part) for part in parts)

    dataset = get_symmetry_dataset(phonon.primitive_symmetry)
    from .isoir import get_isoir_kpoint_name

    midpoints: list[tuple[str, str, list[float]]] = []
    seen: set[tuple[float, ...]] = set()
    for start, end in segments:
        midpoint = snap_qpoint(
            (np.asarray(coords[start], dtype=float) + np.asarray(coords[end], dtype=float)) / 2.0
        )
        key = tuple(np.round(midpoint, 8))
        if key in seen:
            continue
        seen.add(key)
        label = get_isoir_kpoint_name(
            dataset["number"], cell, phonon.primitive_symmetry.tolerance, midpoint
        )
        if label is None:
            label = "q" + "".join(f"_{value:g}" for value in midpoint)
        midpoints.append((label, f"{display(start)}-{display(end)}", list(midpoint)))
    return path_string, midpoints


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    supercell_mat = [float(n) for n in args.dim.split()]
    if args.readfc:
        force_stes = None
        force_constans = "./FORCE_CONSTANTS"
    else:
        force_stes = "./FORCE_SETS"
        force_constans = None

    phonon = load(
        supercell_matrix=supercell_mat,
        primitive_matrix="auto",
        unitcell_filename=args.poscar,
        force_sets_filename=force_stes,
        force_constants_filename=force_constans,
    )

    dataset = get_symmetry_dataset(phonon.symmetry)
    irt_table = IrrepTable(dataset["number"], spinor=False)
    prim_mat = get_primitive_matrix_by_centring(dataset["international"][0])

    q_names, q_list = get_irt_special_points(irt_table, prim_mat)
    path_string, path_midpoints = None, []
    if args.all_irreps:
        try:
            path_string, path_midpoints = _seekpath_path_midpoints(phonon)
        except Exception:
            pass
    yaml_name = "phonon_irreps_all.yaml" if args.all_irreps else "phonon_irreps.yaml"
    with open(yaml_name, "w") as fp:
        fp.write(f"space_group: {dataset['international']}\n")
        fp.write("special_points:\n")
        for qname, q in zip(q_names, q_list):
            fp.write(f"- # {qname}\n")
            fp.write(f"  q_position: {format_qpoint(q)}\n")
        fp.write("\n")
        if path_midpoints:
            fp.write(f"k_path: {path_string}  # seekpath\n")
            fp.write("path_midpoints:  # midpoints of the k-path segments, ISO-IR k-vector types\n")
            for label, segment, midpoint in path_midpoints:
                fp.write(f"- # {label} (midpoint of {segment})\n")
                fp.write(f"  q_position: {format_qpoint(midpoint)}\n")
            fp.write("\n")
        fp.write("irreps:\n")
        for qname, q in zip(q_names, q_list):
            fp.write(f"- q_label: {qname}\n")
            fp.write(f"  q_position: {format_qpoint(q)}\n")
            labels, band_indices, freqs = get_irrep_labels(
                q=q,
                phonon=phonon,
                irt_table=irt_table,
                prim_mat=prim_mat,
                degeneracy_tolerance=args.tol,
            )
            for i, index in enumerate(band_indices):
                fp.write(f"  - # {' '.join([str(idx + 1) for idx in index])}\n")
                fp.write(f"    irrep_label: {labels[i]}\n")
                fp.write(f"    frequency: %14.10f\n" % (freqs[index[0]]))
            fp.write("\n")
        for label, segment, midpoint in path_midpoints:
            fp.write(f"- q_label: {label}\n")
            fp.write(f"  segment: {segment}\n")
            fp.write(f"  q_position: {format_qpoint(midpoint)}\n")
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="No irreps at")
                    labels_mid, band_indices_mid, freqs_mid = get_irrep_labels(
                        q=midpoint,
                        phonon=phonon,
                        irt_table=irt_table,
                        prim_mat=prim_mat,
                        degeneracy_tolerance=args.tol,
                    )
            except Exception:
                fp.write("  # irrep labeling failed at this q point\n\n")
                continue
            for i, index in enumerate(band_indices_mid):
                fp.write(f"  - # {' '.join([str(idx + 1) for idx in index])}\n")
                fp.write(f"    irrep_label: {labels_mid[i]}\n")
                fp.write(f"    frequency: %14.10f\n" % (freqs_mid[index[0]]))
            fp.write("\n")
    print(f"Phonon irreps written to: {yaml_name}")


if __name__ == "__main__":
    main()
