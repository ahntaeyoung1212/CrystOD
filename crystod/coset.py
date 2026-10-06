"""
Coset decomposition display.

Point-group mode : decompose G into left cosets g H of a subgroup H.
Space-group mode : decompose the rotation group of G into right cosets G_k g
                   of the little co-group of a k point (one coset per arm of
                   the star of k).
"""

from __future__ import annotations

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, RawDescriptionHelpFormatter, RawTextHelpFormatter
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray

from phonopy.structure.cells import get_primitive_matrix_by_centring

from .basis_function import _get_character_table, _resolve_space_group_type
from .irreptables_compat import load_irreptables
from .operations import get_seitz_symbol
from .runtime_compat import get_little_group

IrrepTable, Irrep = load_irreptables()


class MyHelpFormatter(
    RawTextHelpFormatter,
    RawDescriptionHelpFormatter,
    ArgumentDefaultsHelpFormatter,
):
    pass


desc = """
Display coset decompositions.

# Command Examples:
crystod-group --coset --point-group m-3m --subgroup 4/mmm
crystod-group --coset --space-group Pm-3m --kpoint 0.5 0.5 0
"""


def _parse_fractional_float(value: str) -> float:
    try:
        return float(Fraction(value))
    except Exception:
        return float(value)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=desc, formatter_class=MyHelpFormatter)
    parser.add_argument(
        "--point-group",
        "-pg",
        default=None,
        help="Point-group label of the full group G, e.g. m-3m.",
    )
    parser.add_argument(
        "--subgroup",
        default=None,
        help="Point-group label of the subgroup H (required with --point-group).",
    )
    parser.add_argument(
        "--space-group",
        "-sg",
        default=None,
        help="Space-group symbol in standard setting, e.g. Pm-3m.",
    )
    parser.add_argument(
        "--kpoint",
        nargs=3,
        type=_parse_fractional_float,
        default=None,
        help="Primitive-basis k point (required with --space-group).",
    )
    return parser


def _point_group_operations(point_group: str) -> tuple[list[NDArray[np.int_]], list[str]]:
    """Return all rotation matrices and per-operation labels of a point group."""
    ct = _get_character_table(point_group)
    rotations: list[NDArray[np.int_]] = []
    labels: list[str] = []
    for class_name in ct["rotation_list"]:
        class_rotations = [np.array(rotation, dtype=int) for rotation in ct["mapping_table"][class_name]]
        for op_index, rotation in enumerate(class_rotations, start=1):
            rotations.append(rotation)
            if len(class_rotations) == 1:
                labels.append(class_name)
            else:
                labels.append(f"{class_name}#{op_index}")
    return rotations, labels


def _find_index(matrix: NDArray, matrices: list[NDArray]) -> int:
    for index, candidate in enumerate(matrices):
        if np.array_equal(matrix, candidate):
            return index
    return -1


def _left_coset_decomposition(
    group: list[NDArray[np.int_]],
    subgroup_indices: list[int],
) -> list[list[int]]:
    """Decompose G into left cosets g H; returns lists of G-element indices."""
    order = len(group)
    visited = np.zeros(order, dtype=bool)
    cosets: list[list[int]] = []

    first = list(subgroup_indices)
    for index in first:
        visited[index] = True
    cosets.append(first)

    while not visited.all():
        representative_index = int(np.flatnonzero(~visited)[0])
        representative = group[representative_index]
        coset: list[int] = []
        for subgroup_index in subgroup_indices:
            product = representative @ group[subgroup_index]
            product_index = _find_index(product, group)
            if product_index < 0:
                raise RuntimeError("Group is not closed under multiplication.")
            if visited[product_index]:
                raise RuntimeError("Coset overlap detected; H is not a subgroup of G.")
            visited[product_index] = True
            coset.append(product_index)
        cosets.append(coset)
    return cosets


def format_point_group_cosets(point_group: str, subgroup: str) -> str:
    """Text report of ``crystod-group --coset --pg G --subgroup H``.

    Args:
        point_group: Label of the group G, e.g. ``"m-3m"``.
        subgroup: Label of the subgroup H, e.g. ``"4/mmm"``.

    Returns:
        The ``* Groups *`` and ``* Coset decomposition G = sum_i g_i H *``
        blocks as one string (leading newline, no trailing newline).

    Raises:
        ValueError: H is not a subgroup of G in the tabulated axes, or |H|
            does not divide |G|; the message is one sentence.
    """
    group_rotations, group_labels = _point_group_operations(point_group)
    subgroup_rotations, _ = _point_group_operations(subgroup)

    subgroup_indices = []
    missing = []
    for rotation in subgroup_rotations:
        index = _find_index(rotation, group_rotations)
        if index < 0:
            missing.append(rotation)
        else:
            subgroup_indices.append(index)

    group_text = f"G = {point_group} (order {len(group_rotations)})"
    subgroup_text = f"H = {subgroup} (order {len(subgroup_rotations)})"
    if missing:
        raise ValueError(
            f"{subgroup_text} is not a subgroup of {group_text} in the tabulated "
            f"axes: {len(missing)} of its operations are not in G (the "
            "character-table settings may orient the axes differently, and the "
            "coset decomposition needs H in the axes of G).")
    if len(group_rotations) % len(subgroup_rotations) != 0:
        raise ValueError(
            f"|H| does not divide |G| for {subgroup_text} and {group_text} "
            "(Lagrange's theorem), so H is not a subgroup of G.")

    lines = ["", "* Groups *", f" {group_text}", f" {subgroup_text}", ""]

    cosets = _left_coset_decomposition(group_rotations, subgroup_indices)
    index_of_group = len(group_rotations) // len(subgroup_rotations)
    lines.append("* Coset decomposition G = sum_i g_i H *")
    lines.append(f" index [G:H] = {index_of_group}")
    for coset_number, coset in enumerate(cosets, start=1):
        representative_label = group_labels[coset[0]] if coset_number > 1 else "E"
        members = ", ".join(group_labels[index] for index in coset)
        lines.append(f" coset {coset_number} (representative: {representative_label}):")
        lines.append(f"   {{ {members} }}")
    return "\n".join(lines)


def _space_group_primitive_symmetry(space_group_symbol: str):
    sg_type = _resolve_space_group_type(space_group_symbol)
    irt_table = IrrepTable(sg_type.number, spinor=False)
    primitive_matrix = np.array(
        get_primitive_matrix_by_centring(sg_type.international_short[0]),
        dtype=float,
    )
    primitive_matrix_inv = np.linalg.inv(primitive_matrix)

    conventional_rotations = np.array([sym.R for sym in irt_table.symmetries], dtype=float)
    conventional_translations = np.array([sym.t for sym in irt_table.symmetries], dtype=float)
    primitive_rotations = np.rint(
        np.array(
            [primitive_matrix_inv @ rotation @ primitive_matrix for rotation in conventional_rotations]
        )
    ).astype(int)
    # column convention x_conv = M x_prim, matching the rotation conversion above
    primitive_translations = np.mod((primitive_matrix_inv @ conventional_translations.T).T, 1.0)
    primitive_translations[np.isclose(primitive_translations, 1.0, atol=1e-8)] = 0.0
    return sg_type, primitive_matrix, primitive_rotations, primitive_translations


def format_space_group_cosets(space_group_symbol: str, kpoint: list[float]) -> str:
    """Text report of ``crystod-group --coset --sg SG --kpoint K``.

    Args:
        space_group_symbol: Space-group symbol or number, e.g. ``"Pm-3m"``.
        kpoint: k point in the primitive basis.

    Returns:
        The ``* Space group *``, ``* k point (primitive) *``,
        ``* Little co-group G_k *`` and ``* Coset decomposition ... *``
        blocks (one coset per arm of the star of k) as one string (leading
        newline, no trailing newline).
    """
    from .star_of_k import compute_star

    sg_type, primitive_matrix, rotations, translations = _space_group_primitive_symmetry(
        space_group_symbol
    )
    seitz_symbols = [get_seitz_symbol(rotation, primitive_matrix) for rotation in rotations]

    _, _, mapping_little_group = get_little_group(
        rotations=rotations,
        translations=translations,
        kpoint=kpoint,
    )

    little_members = ", ".join(seitz_symbols[index] for index in mapping_little_group)
    arms = compute_star(rotations, translations, kpoint)
    lines = [
        "",
        "* Space group *",
        f" {sg_type.international_short} ({sg_type.number})",
        "",
        "* k point (primitive) *",
        f" {np.round(np.asarray(kpoint, dtype=float), 6).tolist()}",
        "",
        "* Little co-group G_k *",
        f" order |G_k| = {len(mapping_little_group)} (|G| = {len(rotations)})",
        f" {{ {little_members} }}",
        "",
        "* Coset decomposition G = sum_i G_k g_i (one coset per arm of the star) *",
        f" index [G:G_k] = |star of k| = {len(arms)}",
    ]
    for arm_index, arm in enumerate(arms, start=1):
        coords = np.round(arm["kpoint"], 6)
        coords_text = "[" + ", ".join(f"{value:+g}" for value in coords) + "]"
        representative = seitz_symbols[arm["representative_index"]]
        members = ", ".join(seitz_symbols[index] for index in arm["operation_indices"])
        lines.append(f" coset {arm_index} (representative: {representative}, k arm = {coords_text}):")
        lines.append(f"   {{ {members} }}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if bool(args.point_group) == bool(args.space_group):
        parser.error("Specify exactly one of --point-group or --space-group.")

    if args.point_group:
        if not args.subgroup:
            parser.error("--point-group requires --subgroup for the coset decomposition.")
        try:
            text = format_point_group_cosets(args.point_group, args.subgroup)
        except ValueError as error:
            raise SystemExit(f"ERROR: {error}") from None
        print(text)
        return

    if args.kpoint is None:
        parser.error("--space-group requires --kpoint.")
    print(format_space_group_cosets(args.space_group, args.kpoint))


if __name__ == "__main__":
    main()
