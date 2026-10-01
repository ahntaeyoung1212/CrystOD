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
crystod-group --decompose --point-group 3m
crystod-group --decompose --point-group 3m --characters 3 0 1
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
    """Character table of a crystallographic point group (phonopy data).

    The table every point-group mode of ``crystod-group`` starts from
    (``--table``, ``--decompose``, ``--ligand-field``, ``--product --pg``,
    ``--multiplet``).

    Args:
        point_group: Hermann-Mauguin point-group label as used by phonopy,
            e.g. ``"m-3m"``, ``"4/mmm"``, ``"3m"``.

    Returns:
        The phonopy character-table dict with the keys ``"rotation_list"``
        (class labels in order), ``"character_table"``
        (``{irrep: characters per class}``) and ``"mapping_table"``
        (``{class label: rotation matrices}``, whose lengths are the class
        sizes).

    Raises:
        SystemExit: Unknown point-group label; the message lists the
            available labels (``ValueError`` when called through
            ``crystod.group``).

    Example:
        >>> from crystod import group
        >>> ct = group.get_character_table("m-3m")
        >>> ct["rotation_list"]
        ('E', 'C3', 'C2', 'C4', 'C4^2', 'i', 'S4', 'S6', 'sgh', 'sgd')
        >>> ct["character_table"]["T2g"]
        (3, 0, 1, -1, -1, 3, -1, 0, -1, 1)
    """
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
    """Multiplicity of every irrep in a reducible representation.

    The reduction formula ``n_i = (1/|G|) sum_C |C| chi_i(C) chi(C)`` over
    the classes ``C`` (real characters); the computation behind
    ``crystod-group --decompose`` and the last step of ``--ligand-field``
    and ``--multiplet``.

    Args:
        characters: Characters of the reducible representation, one per
            class in the order of ``character_table["rotation_list"]``.
        character_table: The table from ``get_character_table``.
        multiplicities: Class sizes in the same order (the lengths of the
            entries of ``character_table["mapping_table"]``).

    Returns:
        ``{irrep label: multiplicity}`` over every irrep of the table
        (zeros included), rounded to integers.

    Example:
        >>> from crystod import group
        >>> ct = group.get_character_table("3m")
        >>> sizes = [len(ops) for ops in ct["mapping_table"].values()]
        >>> group.decompose([3, 0, 1], ct, sizes)
        {'A1': 1, 'A2': 0, 'E': 1}
    """
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
