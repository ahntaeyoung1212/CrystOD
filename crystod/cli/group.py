"""crystod-group: representation-theory calculator for point and space groups.

Merges the former group-theory flat modes into one sectioned command:

- ``--product IRREP...``   -- direct-product decomposition (old ``--direct-product``);
- ``--table``              -- display the point-group character table
  (old ``--direct-product --show-irrep-table`` without irreps);
- ``--decompose``          -- reducible-representation decomposition
  (old ``--decompose-irrep``);
- ``--ligand-field ORB``   -- ligand-field splitting of an atomic orbital
  (old ``--ligand-field-split``);
- ``--basis FUNC...``      -- classify polynomial basis functions
  (old ``--basis-function``);
- ``--generate-basis``     -- auto-generate 1st-3rd order polynomial bases
  (old ``--generate-basis-function``);
- ``--coset``              -- coset decompositions (old ``--show-coset``).

No structure file is needed: the group is selected with --pg/--point-group or
--sg/--space-group.
"""

from __future__ import annotations

from argparse import ArgumentParser, RawTextHelpFormatter
from fractions import Fraction

desc = """\
Representation-theory calculator for point and space groups (no structure
file needed; select the group with --pg/--point-group or --sg/--space-group).

# Command Examples:
crystod-group --product T2g T2g T1u --pg m-3m
crystod-group --table --pg 3m
crystod-group --decompose --pg 3m --characters 3 0 1
crystod-group --ligand-field d --pg m-3m
crystod-group --basis x y z --pg m-3m
crystod-group --basis x y z --sg Pm-3m --kpoint 0 0 0
crystod-group --generate-basis --pg m-3m --order 1 2 3
crystod-group --coset --pg m-3m --subgroup 4/mmm
crystod-group --coset --sg Pm-3m --kpoint 0.5 0.5 0
"""


def _parse_fractional_float(value: str) -> float:
    try:
        return float(Fraction(value))
    except Exception:
        return float(value)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="crystod-group", description=desc, formatter_class=RawTextHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--product",
        nargs="+",
        default=None,
        metavar="IRREP",
        help="Decompose the direct product of the given irreps, e.g. --product T2g T2g T1u.",
    )
    mode.add_argument(
        "--table",
        action="store_true",
        help="Display the character table of the point group.",
    )
    mode.add_argument(
        "--decompose",
        action="store_true",
        help="Decompose a reducible representation into irreps from its characters "
        "(interactive, or via --characters).",
    )
    mode.add_argument(
        "--ligand-field",
        dest="ligand_field",
        default=None,
        metavar="ORBITAL",
        help="Ligand-field splitting of an atomic orbital (s/p/d/f/...), e.g. --ligand-field d.",
    )
    mode.add_argument(
        "--basis",
        nargs="+",
        default=None,
        metavar="FUNC",
        help="Classify polynomial basis functions, e.g. --basis x y z (polar) or "
        "--basis Rx Ry Rz (axial).",
    )
    mode.add_argument(
        "--generate-basis",
        action="store_true",
        help="Auto-generate 1st-3rd order polynomial basis functions per irrep.",
    )
    mode.add_argument(
        "--coset",
        action="store_true",
        help="Coset decomposition of a point group by a subgroup, or of a space "
        "group by the little co-group at k.",
    )

    parser.add_argument(
        "--point-group",
        "--pg",
        dest="point_group",
        default=None,
        help="Point-group label, e.g. m-3m.",
    )
    parser.add_argument(
        "--space-group",
        "--sg",
        dest="space_group",
        default=None,
        help="Space-group symbol, e.g. Pm-3m (for --basis/--generate-basis/--coset).",
    )
    parser.add_argument(
        "--kpoint",
        nargs=3,
        type=_parse_fractional_float,
        default=None,
        help="k-point in the primitive basis (space-group modes).",
    )
    parser.add_argument(
        "--subgroup",
        default=None,
        help="Subgroup point-group label for --coset with --point-group, e.g. 4/mmm.",
    )
    parser.add_argument(
        "--order",
        nargs="+",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Polynomial order(s) for --generate-basis.",
    )
    parser.add_argument(
        "--characters",
        nargs="+",
        type=float,
        default=None,
        help="Characters of the reducible representation for --decompose "
        "(skips interactive input).",
    )
    parser.add_argument(
        "--show-irrep-table",
        action="store_true",
        help="Also display the irrep/character table in --product/--basis/--generate-basis mode.",
    )
    return parser


_DASH_VALUE_FLAGS = ("--point-group", "--pg", "--subgroup")


def _merge_dash_values(argv: list[str]) -> list[str]:
    """Merge crystallographic values starting with '-' (e.g. -43m, -3m, -1)
    into their flag as --flag=value, so argparse does not mistake them for
    options."""
    merged: list[str] = []
    skip_next = False
    for index, token in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if (
            token in _DASH_VALUE_FLAGS
            and index + 1 < len(argv)
            and argv[index + 1].startswith("-")
            and not argv[index + 1].startswith("--")
        ):
            merged.append(f"{token}={argv[index + 1]}")
            skip_next = True
        else:
            merged.append(token)
    return merged


def main(argv: list[str] | None = None) -> None:
    import sys

    parser = build_parser()
    if argv is None:
        argv = sys.argv[1:]
    args = parser.parse_args(_merge_dash_values(list(argv)))

    def require_point_group(mode_name: str) -> None:
        if not args.point_group:
            parser.error(f"{mode_name} requires --pg/--point-group.")

    def require_exactly_one_group(mode_name: str) -> None:
        if bool(args.point_group) == bool(args.space_group):
            parser.error(
                f"{mode_name} requires exactly one of --pg/--point-group or --sg/--space-group."
            )

    if args.product:
        require_point_group("--product")

        dispatch_argv = [f"--point-group={args.point_group}", "--irreps", *args.product]
        if args.show_irrep_table:
            dispatch_argv.append("--show-irrep-table")

        from ..direct_product import main as direct_product_main

        direct_product_main(dispatch_argv)
        return

    if args.table:
        require_point_group("--table")

        from ..direct_product import main as direct_product_main

        direct_product_main([f"--point-group={args.point_group}", "--show-irrep-table"])
        return

    if args.decompose:
        require_point_group("--decompose")

        dispatch_argv = [f"--point-group={args.point_group}"]
        if args.characters:
            dispatch_argv.extend(["--characters", *[str(value) for value in args.characters]])

        from ..decompose_irrep import main as decompose_irrep_main

        decompose_irrep_main(dispatch_argv)
        return

    if args.ligand_field:
        require_point_group("--ligand-field")

        from ..ligand_field import main as ligand_field_main

        ligand_field_main([f"--point-group={args.point_group}", "--orbital", args.ligand_field])
        return

    if args.basis is not None:
        require_exactly_one_group("--basis")
        if args.kpoint is not None and args.space_group is None:
            parser.error("--basis uses --kpoint only with --sg/--space-group.")

        dispatch_argv = ["--basis-function", *args.basis]
        if args.point_group:
            dispatch_argv.append(f"--point-group={args.point_group}")
        else:
            dispatch_argv.extend(["--space-group", args.space_group])
            if args.kpoint is not None:
                dispatch_argv.extend(["--kpoint", *[str(value) for value in args.kpoint]])
        if args.show_irrep_table:
            dispatch_argv.append("--show-irrep-table")

        from ..basis_function import main as basis_function_main

        basis_function_main(dispatch_argv)
        return

    if args.generate_basis:
        require_exactly_one_group("--generate-basis")
        if args.space_group and args.kpoint is None:
            parser.error("--generate-basis with --sg/--space-group requires --kpoint.")

        dispatch_argv = []
        if args.point_group:
            dispatch_argv.append(f"--point-group={args.point_group}")
        else:
            dispatch_argv.extend(["--space-group", args.space_group])
            dispatch_argv.extend(["--kpoint", *[str(value) for value in args.kpoint]])
        if args.order:
            dispatch_argv.extend(["--order", *[str(value) for value in args.order]])
        if args.show_irrep_table:
            dispatch_argv.append("--show-irrep-table")

        from ..generate_basis_function import main as generate_basis_function_main

        generate_basis_function_main(dispatch_argv)
        return

    # --coset
    require_exactly_one_group("--coset")
    if args.point_group and not args.subgroup:
        parser.error("--coset with --pg/--point-group requires --subgroup.")
    if args.space_group and args.kpoint is None:
        parser.error("--coset with --sg/--space-group requires --kpoint.")

    dispatch_argv: list[str] = []
    if args.point_group:
        dispatch_argv.extend([f"--point-group={args.point_group}", f"--subgroup={args.subgroup}"])
    else:
        dispatch_argv.extend(["--space-group", args.space_group])
        dispatch_argv.extend(["--kpoint", *[str(value) for value in args.kpoint]])

    from ..coset import main as coset_main

    coset_main(dispatch_argv)


if __name__ == "__main__":
    main()
