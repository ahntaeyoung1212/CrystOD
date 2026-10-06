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


def class_names(character_table: dict) -> list[str]:
    """Class labels in the order of the characters.

    Args:
        character_table: The table from ``get_character_table``.

    Returns:
        ``character_table["rotation_list"]`` as a list (point group 1 stores
        the single string ``"E"``).
    """
    names = character_table["rotation_list"]
    return [names] if isinstance(names, str) else list(names)


def class_sizes(character_table: dict) -> list[int]:
    """Class sizes in the order of the characters.

    The sizes are looked up by class label, because the keys of
    ``character_table["mapping_table"]`` may come in another order than
    ``character_table["rotation_list"]`` (point groups -4, 222, mmm,
    6/mmm).

    Args:
        character_table: The table from ``get_character_table``.

    Returns:
        ``[|C| for C in rotation_list]``.

    Example:
        >>> from crystod import group
        >>> from crystod.decompose_irrep import class_sizes
        >>> ct = group.get_character_table("-4")
        >>> ct["rotation_list"], class_sizes(ct)
        (('E', 'S4', 'C2'), [1, 2, 1])
    """
    return [
        int(np.asarray(character_table["mapping_table"][name]).shape[0])
        for name in class_names(character_table)
    ]


def irrep_character_rows(character_table: dict) -> dict[str, np.ndarray]:
    """Irrep characters as 1-d float arrays (point group 1 stores scalars).

    Args:
        character_table: The table from ``get_character_table``.

    Returns:
        ``{irrep: characters in the order of rotation_list}``.
    """
    return {
        name: np.atleast_1d(np.asarray(values, dtype=float))
        for name, values in character_table["character_table"].items()
    }


def decompose(
    characters: list[float],
    character_table: dict,
    multiplicities: list[int] | None = None,
) -> dict[str, int]:
    """Multiplicity of every irrep in a reducible representation.

    The reduction formula
    ``n_i = sum_C |C| chi_i(C) chi(C) / sum_C |C| chi_i(C)^2`` over the
    classes ``C`` (real characters); the computation behind
    ``crystod-group --decompose`` and the last step of ``--ligand-field``
    and ``--multiplet``.  The denominator is ``|G|`` for an ordinary irrep
    and ``2|G|`` for the entries that the tables store as physically
    irreducible real pairs of complex-conjugate irreps (E of 3, 4, -4, 23,
    E1/E2 of 6, ...), so such a pair counts once per real copy.  The
    representation must be real (orbital, product and polynomial spaces
    are).

    Args:
        characters: Characters of the reducible representation, one per
            class in the order of ``character_table["rotation_list"]``.
        character_table: The table from ``get_character_table``.
        multiplicities: Class sizes in the same order; by default
            ``class_sizes(character_table)``.

    Returns:
        ``{irrep label: multiplicity}`` over every irrep of the table
        (zeros included), rounded to integers.

    Example:
        >>> from crystod import group
        >>> ct = group.get_character_table("3m")
        >>> group.decompose([3, 0, 1], ct)
        {'A1': 1, 'A2': 0, 'E': 1}
        >>> group.decompose([2, 0, -2], group.get_character_table("4"))
        {'A': 0, 'B': 0, 'E': 1}
    """
    if multiplicities is None:
        multiplicities = class_sizes(character_table)
    sizes = np.array(multiplicities, dtype=float)
    reducible = np.atleast_1d(np.array(characters, dtype=float))
    results: dict[str, int] = {}
    for irrep_name, irrep in irrep_character_rows(character_table).items():
        norm = float(np.sum(sizes * irrep * irrep))
        results[irrep_name] = round(float(np.sum(sizes * irrep * reducible)) / norm)
    return results


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    character_table = get_character_table(args.point_group)

    multiplicities = class_sizes(character_table)
    class_labels = [
        f"{multiplicity}{operation}"
        for operation, multiplicity in zip(class_names(character_table), multiplicities)
    ]

    print(f"\n* Point group *\n{args.point_group}\n")
    print("* Reducible representation *")

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
