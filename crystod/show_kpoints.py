"""Show the special (high-symmetry) k points of a space group.

``crystod-bz --show-kpoint --space-group Pnma`` prints the special k points of
the space group in the primitive reciprocal basis (the same ISO-IR convention
as the irrep analyses of ``crystod``, ``crystod-phonon --irreps``,
``crystod-mag`` and ``crystod-group``).  For centred lattices (F, I, C, A,
R), whose conventional and primitive cells differ, the coordinates in the
conventional reciprocal basis are printed as well.  The table is that of the
ISO-IR standard setting of the space group; a non-standard setting symbol
(``Pbnm``, ``Amam``, ``Bbmm``) is accepted, and a NOTE says that the
coordinates refer to the standard setting.
"""

from __future__ import annotations

import argparse
from fractions import Fraction

import numpy as np
from phonopy.structure.cells import get_primitive_matrix_by_centring

from .basis_function import IrrepTable, _resolve_space_group_type


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Show the special k points of a space group (ISO-IR convention)."
    )
    parser.add_argument(
        "--space-group",
        required=True,
        help='Space-group symbol (e.g. "Pnma", "Fm-3m") or Hermann-Mauguin full symbol.',
    )
    return parser


def _format_coordinate(value: float) -> str:
    fraction = Fraction(value).limit_denominator(24)
    if fraction.denominator == 1:
        return str(fraction.numerator)
    return f"{fraction.numerator}/{fraction.denominator}"


def _format_kpoint(kpoint) -> str:
    return "(" + ", ".join(_format_coordinate(v) for v in kpoint) + ")"


def _standard_setting(sg_type):
    """spglib record of the ISO-IR standard setting of the group of ``sg_type``."""
    import spglib

    from .isoir import iso_hall_number
    from .runtime_compat import get_spacegroup_type

    return get_spacegroup_type(spglib.get_spacegroup_type(iso_hall_number(sg_type.number)))


def _same_setting(first, second) -> bool:
    """Do two space-group records have the same axes and centring?

    The origin choice is ignored (it does not move k vectors): ``Fd-3m``
    with origin choice 1 and 2 share the full symbol ``F 4_1/d -3 2/m``.
    """
    return "".join(first.international_full.split()) == "".join(
        second.international_full.split()
    )


def get_special_kpoints(space_group_symbol: str):
    """Special (high-symmetry) k points of a space group in the ISO-IR convention.

    This is the table behind ``crystod-bz --show-kpoint --space-group Pnma``.
    The k points and their labels (``GM``, ``X``, ``M``, ``R``, ...) come from
    the bundled ISO-IR tables (``CIR_data``), so they are the points at which
    ``crystod``, ``crystod-phonon --irreps``, ``crystod-mag`` and
    ``crystod-group`` label irreps. Coordinates are given in the primitive
    reciprocal basis; for centred lattices (F, I, C, A, R) the
    conventional-basis coordinates are returned as well, since the two
    bases differ there. Both refer to the ISO-IR standard setting of the
    group (origin choice 2, monoclinic b-unique cell choice 1, hexagonal
    axes): a non-standard setting symbol such as ``Pbnm`` or ``Amam`` selects
    the group, and the table is that of ``Pnma`` or ``Cmcm``.

    Args:
        space_group_symbol: Space-group symbol (``"Pnma"``, ``"Fm-3m"`` or the
            full symbol ``"P m -3 m"``) or the number as a string (``"221"``).

    Returns:
        The tuple ``(sg_type, names, primitive_kpoints, conventional_kpoints)``
        where ``sg_type`` is the spglib ``SpaceGroupType`` record of the
        ISO-IR standard setting of the resolved group, the setting the
        coordinates refer to (``sg_type.number``, ``sg_type.international_short``),
        ``names`` the ISO-IR labels ordered with ``GM`` first and then by the
        sum of the absolute primitive coordinates and by name,
        ``primitive_kpoints`` the matching ``[kx, ky, kz]`` lists in the
        primitive reciprocal basis, and ``conventional_kpoints`` the same
        points in the conventional reciprocal basis, or ``None`` for
        primitive (P) lattices, where the two bases coincide.

    Raises:
        SystemExit: The symbol or number is not a standard space group
            (``ValueError`` when called through ``crystod.bz``).

    Example:
        >>> from crystod import bz
        >>> sg_type, names, primitive, conventional = bz.get_special_kpoints("Fm-3m")
        >>> sg_type.number, names
        (225, ['GM', 'X', 'L', 'W'])
        >>> primitive[1], conventional[1]
        ([0.5, 0.0, 0.5], [0.0, 1.0, 0.0])
        >>> bz.get_special_kpoints("Amam")[0].international_short   # setting of Cmcm
        'Cmcm'
    """
    sg_type = _standard_setting(_resolve_space_group_type(space_group_symbol))
    irt_table = IrrepTable(sg_type.number, spinor=False)
    # The k vectors are tabulated in the ISO-IR standard setting (irt_table.name
    # is its symbol), so the centring comes from there and not from the symbol
    # given: C for --sg Amam and --sg Bbmm (B is no phonopy centring), both Cmcm.
    primitive_matrix = np.array(
        get_primitive_matrix_by_centring(irt_table.name[0]),
        dtype=float,
    )

    names: list[str] = []
    conventional: list[list[float]] = []
    primitive: list[list[float]] = []
    for irrep in irt_table.irreps:
        k_conventional = [float(np.round(float(v), 6)) for v in np.array(irrep.k, dtype=float)]
        k_primitive = [
            float(np.round(float(v), 6))
            for v in np.array(irrep.k, dtype=float) @ primitive_matrix
        ]
        if k_primitive not in primitive:
            names.append(irrep.kpname)
            conventional.append(k_conventional)
            primitive.append(k_primitive)

    # GM first, then by distance from GM (sum of |k|), then alphabetically.
    order = sorted(
        range(len(names)),
        key=lambda i: (names[i] != "GM", sum(abs(v) for v in primitive[i]), names[i]),
    )
    names = [names[i] for i in order]
    conventional = [conventional[i] for i in order]
    primitive = [primitive[i] for i in order]

    is_primitive_lattice = np.allclose(primitive_matrix, np.eye(3))
    return sg_type, names, primitive, (None if is_primitive_lattice else conventional)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    sg_type, names, primitive, conventional = get_special_kpoints(args.space_group)
    requested = _resolve_space_group_type(args.space_group)

    print("* Space group *")
    print(f"{sg_type.international_short} (No. {sg_type.number})")
    if not _same_setting(requested, sg_type):
        print(
            f'NOTE: "{args.space_group}" names the non-standard setting '
            f"{requested.international_full} of space group No. {sg_type.number}.\n"
            "      The labels and coordinates below refer to the ISO-IR standard setting\n"
            f"      {sg_type.international_full} (axes and centring of "
            f"{sg_type.international_short}), not to the requested axes."
        )
    print()
    print("* K points (primitive) *")
    for name, kpoint in zip(names, primitive):
        print(f"{name}: {_format_kpoint(kpoint)}")
    if conventional is not None:
        print()
        print("* K points (conventional) *")
        for name, kpoint in zip(names, conventional):
            print(f"{name}: {_format_kpoint(kpoint)}")


if __name__ == "__main__":
    main()
