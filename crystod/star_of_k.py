"""
Star of k: orbit of a k point under the space-group rotations.

Provides a standalone CLI mode and reusable helpers for the modulation mode
(multi-q modulations combine arms of the same star).
"""

from __future__ import annotations

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, RawDescriptionHelpFormatter, RawTextHelpFormatter

import numpy as np
from numpy.typing import NDArray

from phonopy.structure.cells import get_primitive_matrix_by_centring

from .operations import get_seitz_symbol
from .runtime_compat import get_little_group
from .spglib_compat import ensure_spglib_compat

ensure_spglib_compat()

from phonopy.interface.calculator import read_crystal_structure

from .vibration_modes import SymmetryOnlyVibrations


class MyHelpFormatter(
    RawTextHelpFormatter,
    RawDescriptionHelpFormatter,
    ArgumentDefaultsHelpFormatter,
):
    pass


desc = """
Display the star of k: the set of inequivalent k points generated from a given
k point by the space-group rotations (k' = k R, modulo reciprocal lattice).

# Command Examples:
crystod --star-of-k --poscar 221_PPOSCAR_ScF3 --kpoint 0.5 0.5 0
crystod --star-of-k --poscar 221_PPOSCAR_ScF3 --kpoint M
"""


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=desc, formatter_class=MyHelpFormatter)
    parser.add_argument("--poscar", default="POSCAR", help="POSCAR path.")
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
    return parser


def resolve_kpoint_input(structure: SymmetryOnlyVibrations, raw_kpoint: list[str]) -> tuple[str, list[float]]:
    """Resolve a k-point label or coordinates, tolerating a missing seekpath."""
    try:
        return structure.resolve_qpoint(raw_kpoint)
    except Exception:
        if len(raw_kpoint) == 3:
            return "custom", [float(value) for value in raw_kpoint]
        raise


def _wrap_to_unit(kpoint: NDArray[np.float64]) -> NDArray[np.float64]:
    """Wrap a k point into [-0.5, 0.5) for display and comparison."""
    wrapped = np.remainder(np.asarray(kpoint, dtype=float) + 0.5, 1.0) - 0.5
    wrapped[np.abs(wrapped + 0.5) < 1e-8] = 0.5
    return wrapped


def _k_equivalent(k_a: NDArray[np.float64], k_b: NDArray[np.float64], atol: float = 1e-8) -> bool:
    diff = np.asarray(k_a, dtype=float) - np.asarray(k_b, dtype=float)
    return bool((np.abs(diff - np.rint(diff)) < atol).all())


def compute_star(
    rotations: NDArray[np.int_],
    translations: NDArray[np.float64],
    kpoint: list[float],
) -> list[dict]:
    """Compute the star of k.

    Returns one entry per arm:
      {"kpoint": wrapped arm coordinates,
       "representative_index": index of the coset-representative rotation,
       "operation_indices": all rotation indices sending k to this arm}
    """
    kpoint = np.asarray(kpoint, dtype=float)
    arms: list[dict] = []
    for index, rotation in enumerate(rotations):
        k_image = kpoint @ rotation
        for arm in arms:
            if _k_equivalent(k_image, arm["kpoint_raw"]):
                arm["operation_indices"].append(index)
                break
        else:
            arms.append(
                {
                    "kpoint_raw": k_image,
                    "kpoint": _wrap_to_unit(k_image),
                    "representative_index": index,
                    "operation_indices": [index],
                }
            )
    for arm in arms:
        del arm["kpoint_raw"]
    return arms


def format_star_lines(
    arms: list[dict],
    seitz_symbols: list[str] | None = None,
    indent: str = " ",
) -> list[str]:
    lines = []
    for arm_index, arm in enumerate(arms):
        coords = np.round(arm["kpoint"], 6)
        coords_text = "[" + ", ".join(f"{value:+.4f}".rstrip("0").rstrip(".") for value in coords) + "]"
        if seitz_symbols is not None:
            representative = seitz_symbols[arm["representative_index"]]
            lines.append(f"{indent}arm {arm_index + 1}: k = {coords_text}   (representative: {representative})")
        else:
            lines.append(f"{indent}arm {arm_index + 1}: k = {coords_text}")
    return lines


def print_star_of_k(
    rotations: NDArray[np.int_],
    translations: NDArray[np.float64],
    kpoint: list[float],
    seitz_symbols: list[str] | None = None,
    indent: str = " ",
) -> list[dict]:
    """Print the star of k and return the computed arms."""
    arms = compute_star(rotations, translations, kpoint)
    _, _, mapping_little_group = get_little_group(
        rotations=rotations,
        translations=translations,
        kpoint=kpoint,
    )
    order = len(rotations)
    little_order = len(mapping_little_group)
    print(f"{indent}|G| = {order}, |G_k| = {little_order}, |star of k| = {len(arms)}")
    for line in format_star_lines(arms, seitz_symbols, indent=indent):
        print(line)
    return arms


def read_poscar_or_exit(poscar_path: str):
    """Read a POSCAR file, exiting with a clear message when it cannot be read."""
    import os

    if not os.path.isfile(poscar_path):
        raise SystemExit(f"ERROR: No POSCAR named {poscar_path}!")
    cell, _ = read_crystal_structure(poscar_path, interface_mode="vasp")
    if cell is None:
        raise SystemExit(f"ERROR: failed to read POSCAR file: '{poscar_path}'")
    return cell


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cell = read_poscar_or_exit(args.poscar)
    structure = SymmetryOnlyVibrations(cell=cell, symprec=args.tolerance)

    kpoint_label, kpoint = resolve_kpoint_input(structure, args.kpoint)

    dataset = structure.spglib_dataset
    print(f"\n * Space group *\n {dataset['international']} ({dataset['number']})\n")
    print(f" * k point (primitive) *\n {kpoint_label} {np.round(kpoint, 6).tolist()}\n")

    transformation_matrix = get_primitive_matrix_by_centring(dataset["international"][0])
    seitz_symbols = [
        get_seitz_symbol(rotation, transformation_matrix) for rotation in structure.rotations
    ]

    print(" * Star of k *")
    print_star_of_k(
        rotations=structure.rotations,
        translations=structure.translations,
        kpoint=kpoint,
        seitz_symbols=seitz_symbols,
    )
    print("")


if __name__ == "__main__":
    main()
