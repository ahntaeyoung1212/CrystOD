"""
Ligand-field splitting workflow for crystod.

Decomposes an atomic orbital (s, p, d, f, ...) into the irreps of a point
group, i.e. the crystal-field / ligand-field splitting of the orbital in the
given point-symmetric environment. Based on script/ligand_field_spliting.py
by Hiroki Koiso (2023).
"""

from __future__ import annotations

import re
from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    RawDescriptionHelpFormatter,
    RawTextHelpFormatter,
)

import numpy as np

from .decompose_irrep import decompose, get_character_table


class MyHelpFormatter(
    RawTextHelpFormatter,
    RawDescriptionHelpFormatter,
    ArgumentDefaultsHelpFormatter,
):
    pass


desc = """
Calculate the ligand-field splitting of an atomic orbital in a point-symmetric
field: the (2l+1)-dimensional orbital representation is decomposed into the
irreps of the selected point group.

# Command Examples:
crystod-group --ligand-field d --point-group 4/mmm
crystod-group --ligand-field f --point-group m-3m
"""

ORBITAL_AZIMUTHAL_NUMBER = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5, "i": 6}


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=desc, formatter_class=MyHelpFormatter)
    parser.add_argument(
        "--point-group",
        "-pg",
        dest="point_group",
        required=True,
        type=str,
        help="Point group, e.g. 4/mmm.",
    )
    parser.add_argument(
        "--orbital",
        required=True,
        type=str,
        help=f"Orbital: one of {', '.join(ORBITAL_AZIMUTHAL_NUMBER)}.",
    )
    return parser


def _rotation_order(label: str) -> float:
    """Rotation order n from a class label such as C4, S6, C2', C2''."""
    digits = [float(value) for value in re.findall(r"\d", label)]
    order = digits[0]
    for value in digits[1:]:
        order /= value
    return order


def get_orbital_characters(orbital: str, character_table: dict) -> dict[str, int]:
    """Characters of the (2l+1)-dimensional orbital representation per class.

    Uses the standard angular-momentum character formulas:
    chi(C(a)) = sin((l+1/2)a)/sin(a/2), chi(S(a)) = cos((l+1/2)a)/cos(a/2),
    chi(E) = 2l+1, chi(i) = (-1)^l (2l+1), chi(sigma) = chi(S(0)) = 1.
    """
    l = ORBITAL_AZIMUTHAL_NUMBER[orbital]
    characters: dict[str, int] = {}
    for rotation in character_table["rotation_list"]:
        if rotation == "E":
            character = 2 * l + 1
        elif rotation == "i":
            character = (-1) ** l * (2 * l + 1)
        elif "sg" in rotation:
            # mirror plane: chi = 1 for every l
            character = (-1) ** l * np.sin((l + 0.5) * np.pi) / np.sin(np.pi / 2)
        else:
            alpha = 2 * np.pi / _rotation_order(rotation)
            if "C" in rotation:
                character = np.sin((l + 0.5) * alpha) / np.sin(alpha / 2)
            elif "S" in rotation:
                character = np.cos((l + 0.5) * alpha) / np.cos(alpha / 2)
            else:
                raise ValueError(f"Unrecognized symmetry-operation label '{rotation}'.")
        characters[rotation] = round(float(character))
    return characters


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    orbital = args.orbital.strip().lower()
    if orbital not in ORBITAL_AZIMUTHAL_NUMBER:
        raise SystemExit(
            f"ERROR: orbital '{args.orbital}' is not supported. "
            f"Choose from: {', '.join(ORBITAL_AZIMUTHAL_NUMBER)}"
        )

    character_table = get_character_table(args.point_group)
    orbital_characters = get_orbital_characters(orbital, character_table)

    multiplicities = [np.array(ops).shape[0] for ops in character_table["mapping_table"].values()]

    print(f"\n* Point group *\n{args.point_group}\n")
    print(f"* Orbital *\n{orbital}\n")
    print(
        f"* Reducible representation of the {orbital} orbital "
        f"in the {args.point_group} field *\n"
    )
    for (rotation, character), multiplicity in zip(orbital_characters.items(), multiplicities):
        print(f"{multiplicity}{rotation}: {character}")

    results = decompose(list(orbital_characters.values()), character_table, multiplicities)
    result = " + ".join(f"{count}({irrep})" for irrep, count in results.items() if count > 0)

    print("\n* Result *")
    print(result)
    print()


if __name__ == "__main__":
    main()
