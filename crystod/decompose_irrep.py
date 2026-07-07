"""
Reducible-representation decomposition workflow for crystod.

Interactive decomposition of a reducible representation into point-group
irreps from user-supplied characters. Based on script/decomose_to_irreps.py
by Hiroki Koiso (2023).
"""

from __future__ import annotations

from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    RawDescriptionHelpFormatter,
    RawTextHelpFormatter,
)

import numpy as np
from phonopy.phonon.character_table import character_table as all_character_tables


class MyHelpFormatter(
    RawTextHelpFormatter,
    RawDescriptionHelpFormatter,
    ArgumentDefaultsHelpFormatter,
):
    pass


desc = """
Decompose a reducible representation into irreducible representations of a
point group. The characters of the reducible representation are entered
interactively for each symmetry-operation class (or given at once with
--characters).

# Command Examples:
crystod --decompose-irrep --point-group 3m
crystod --decompose-irrep --point-group 3m --characters 3 0 1
"""


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=desc, formatter_class=MyHelpFormatter)
    parser.add_argument(
        "--point-group",
        "-pg",
        dest="point_group",
        required=True,
        type=str,
        help="Point group, e.g. 3m.",
    )
    parser.add_argument(
        "--characters",
        nargs="+",
        type=float,
        default=None,
        help="Characters of the reducible representation, one per class, "
        "in the order of the class prompt (skips interactive input).",
    )
    return parser


def get_character_table(point_group: str) -> dict:
    try:
        return all_character_tables[point_group][0]
    except KeyError:
        available = ", ".join(all_character_tables.keys())
        raise SystemExit(
            f'ERROR: "{point_group}" is not in the available point groups.\n'
            f"Choose from: {available}"
        )


def decompose(
    characters: list[float],
    character_table: dict,
    multiplicities: list[int],
) -> dict[str, int]:
    """Number of times each irrep appears in the reducible representation."""
    multiplicity = np.array(multiplicities, dtype=float)
    reducible = np.array(characters, dtype=float)
    results: dict[str, int] = {}
    for irrep_name, irrep_characters in character_table["character_table"].items():
        irrep = np.array(irrep_characters, dtype=float)
        count = round(float(np.sum(multiplicity * irrep * reducible)) / multiplicity.sum())
        results[irrep_name] = count
    return results


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    character_table = get_character_table(args.point_group)

    multiplicities = [np.array(ops).shape[0] for ops in character_table["mapping_table"].values()]
    class_labels = [
        f"{multiplicity}{operation}"
        for operation, multiplicity in zip(character_table["rotation_list"], multiplicities)
    ]

    print(f"\n* Point group *\n{args.point_group}\n")
    print("* Reducible representation *\n")

    if args.characters is not None:
        if len(args.characters) != len(class_labels):
            raise SystemExit(
                f"ERROR: {len(class_labels)} characters are required for classes "
                f"{class_labels}, but {len(args.characters)} were given."
            )
        characters = list(args.characters)
        for label, value in zip(class_labels, characters):
            print(f"{label}: {value:g}")
    else:
        characters = [float(input(f"{label}: ")) for label in class_labels]

    results = decompose(characters, character_table, multiplicities)
    result = " + ".join(f"{count}({irrep})" for irrep, count in results.items() if count > 0)

    print("\n* Result *")
    print(result if result else "(no irrep: the characters are inconsistent with this point group)")
    print()


if __name__ == "__main__":
    main()
