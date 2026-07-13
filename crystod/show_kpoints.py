"""Show the special (high-symmetry) k points of a space group.

``crystod-bz --show-kpoint --space-group Pnma`` prints the special k points of
the space group in the primitive reciprocal basis (the same CDML convention as
``--salc``, ``--phonon-irrep``, and ``--basis-function``).  For centred
lattices (F, I, C, A, B, R), whose conventional and primitive cells differ,
the coordinates in the conventional reciprocal basis are printed as well.
"""

from __future__ import annotations

import argparse
from fractions import Fraction

import numpy as np
from phonopy.structure.cells import get_primitive_matrix_by_centring

from .basis_function import IrrepTable, _resolve_space_group_type


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Show the special k points of a space group (CDML convention)."
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


def get_special_kpoints(space_group_symbol: str):
    """Special k points of a space group from irreptables.

    Returns ``(sg_type, names, primitive_kpoints, conventional_kpoints)``.
    ``conventional_kpoints`` is ``None`` for primitive (P) lattices, where the
    two bases coincide.
    """
    sg_type = _resolve_space_group_type(space_group_symbol)
    irt_table = IrrepTable(sg_type.number, spinor=False)
    primitive_matrix = np.array(
        get_primitive_matrix_by_centring(sg_type.international_short[0]),
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

    print("* Space group *")
    print(f"{sg_type.international_short} (No. {sg_type.number})")
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
