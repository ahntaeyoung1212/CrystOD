"""crystod-group: representation-theory calculator for point and space groups.

The representation-theory modes:

- ``--product IRREP...``   -- direct-product decomposition;
- ``--table``              -- display the point-group character table;
- ``--decompose``          -- reducible-representation decomposition;
- ``--ligand-field ORB``   -- ligand-field splitting of an atomic orbital;
- ``--basis FUNC...``      -- classify polynomial basis functions;
- ``--generate-basis``     -- auto-generate 1st-3rd order polynomial bases;
- ``--coset``              -- coset decompositions.

No structure file is needed: the group is selected with --pg/--point-group or
--sg/--space-group.
"""

from __future__ import annotations

import re
from argparse import ArgumentParser, RawTextHelpFormatter
from fractions import Fraction

from .common import CRYSTOD_CITATION, banner

desc = """\
Representation-theory calculator for point and space groups (no structure
file needed; select the group with --pg/--point-group or --sg/--space-group).

# Command Examples:
crystod-group --product T2g T2g T1u --pg m-3m
crystod-group --product R4- R5+ --sg Pm-3m
crystod-group --multiplet T2g2 --pg m-3m [--orbital d]
crystod-group --poscar2cif -c PPOSCAR [--tolerance 0.01]
crystod-group --cif2poscar -c FILE.cif [--conventional]
crystod-group --parent Pm-3m --irrep R4+              (--supergroup is an alias)
crystod-group --parent Pm-3m --kpoint GM              (every irrep of the k point)
crystod-group --supergroup-cif 221.cif --subgroup-cif 140.cif
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


def _kpoint_tokens(values: list[str]) -> list[str]:
    """Split quoted values, so that --kpoint "0 1/2 0" equals --kpoint 0 1/2 0."""
    return [token for value in values for token in value.replace(",", " ").split()]


def _kpoint_coordinates(parser: ArgumentParser, values: list[str]) -> list[float]:
    """The three --kpoint coordinates of the modes that take no k-point name
    (every mode but --parent), with the messages argparse gave for nargs=3."""
    tokens = _kpoint_tokens(values)
    if len(tokens) > 3:
        parser.error(f"unrecognized arguments: {' '.join(tokens[3:])}")
    if len(tokens) == 1 and not tokens[0][:1].isdigit() and tokens[0][:1] not in "+-.":
        parser.error(
            "argument --kpoint: expected 3 arguments (three coordinates in the "
            f"primitive basis; a k-point name such as {tokens[0]} is only "
            "accepted with --parent)"
        )
    if len(tokens) != 3:
        parser.error("argument --kpoint: expected 3 arguments")
    coordinates = []
    for token in tokens:
        try:
            coordinates.append(_parse_fractional_float(token))
        except ValueError:
            parser.error(
                f"argument --kpoint: invalid _parse_fractional_float value: {token!r}"
            )
    return coordinates


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="crystod-group",
        description=f"{banner()}\n\n{desc}",
        epilog=CRYSTOD_CITATION,
        formatter_class=RawTextHelpFormatter,
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
        help="Display the character table of the point group, or, with\n"
        "--space-group and --kpoint, of the little group of k (ISO-IR labels\n"
        "at tabulated k points and on lines/planes).",
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
    mode.add_argument(
        "--multiplet",
        nargs="+",
        default=None,
        metavar="IRREP^N",
        help="Multi-electron term symbols (spin multiplicity + spatial irrep)\n"
        "of an electron configuration, e.g. --multiplet T2g2 --pg m-3m gives\n"
        "^3T1g + ^1A1g + ^1Eg + ^1T2g (T2g2 = T2g^2, no quoting needed;\n"
        "several shells: T2g2 Eg1).",
    )
    mode.add_argument(
        "--poscar2cif",
        action="store_true",
        help="Convert a POSCAR into a Bilbao-style CIF (<POSCAR>.cif),\n"
        "e.g. --poscar2cif -c PPOSCAR [--tolerance 0.01].",
    )
    mode.add_argument(
        "--cif2poscar",
        action="store_true",
        help="Convert a CIF into a POSCAR (input path without .cif;\n"
        "primitive cell by default, --conventional for the conventional cell),\n"
        "e.g. --cif2poscar -c FILE.cif.",
    )
    mode.add_argument(
        "--supergroup-cif",
        dest="supergroup_cif",
        default=None,
        metavar="FILE",
        help="Symmetry-mode (AMPLIMODES-style) analysis: decompose the\n"
        "distortion between a high-symmetry structure (this file) and a\n"
        "low-symmetry structure (--subgroup-cif) into parent irreps with\n"
        "mode amplitudes, e.g. --supergroup-cif 221.cif --subgroup-cif 140.cif.",
    )
    mode.add_argument(
        "--parent",
        # the value is the PARENT group and the output is its subgroups;
        # --supergroup was the original name and stays as an alias, but next
        # to --supergroup-cif (a different analysis) it is easy to confuse
        "--supergroup",
        dest="parent",
        default=None,
        metavar="SG",
        help="Isotropy subgroups: which space group results when a distortion\n"
        "with a given irrep (and order-parameter direction) condenses in the\n"
        "parent group SG, e.g. --parent Pm-3m --irrep GM4- [--order-parameter 0 0 a].\n"
        "With --kpoint instead of --irrep, every irrep of that k point is\n"
        "listed in one table, e.g. --parent Pm-3m --kpoint GM.\n"
        "--supergroup SG is an alias.",
    )

    parser.add_argument(
        "--irrep",
        nargs="+",
        default=None,
        metavar="IR",
        help="ISO-IR irrep label(s) for --parent, e.g. GM4- or R4+;\n"
        "several labels (e.g. --irrep X3- X2+) enumerate the isotropy\n"
        "subgroups of the coupled order parameters. For every irrep of one\n"
        "k point, give --kpoint instead.",
    )
    parser.add_argument(
        "--order-parameter",
        nargs="+",
        default=None,
        metavar="C",
        help='Order-parameter direction for --parent, e.g. 0 0 a or a a 0\n'
        "(letters = free parameters). Omit to list every direction.",
    )
    parser.add_argument(
        "--orbital",
        default=None,
        help="Parent atomic orbital (s/p/d/f/...) for --multiplet: checks the\n"
        "occupied shells against its ligand-field splitting and computes the\n"
        "Coulomb multiplet energies (Racah/Slater parameters).",
    )
    parser.add_argument(
        "--cell",
        "-c",
        "--poscar",
        "--cif",
        dest="cell",
        default=None,
        metavar="FILE",
        help="Input structure file: POSCAR for --poscar2cif, CIF for --cif2poscar.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Symmetry tolerance (symprec, Angstrom) for --poscar2cif/"
        "--cif2poscar (default: 0.01).",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Output path for --poscar2cif/--cif2poscar (defaults: <POSCAR>.cif "
        "/ input without .cif) or for --multiplet --visualize.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="For --multiplet (with --orbital): write the exact term "
        "eigenstates as an interactive HTML page (orbital box diagrams with "
        "the Slater-determinant expansion of every term).",
    )
    parser.add_argument(
        "--conventional",
        action="store_true",
        help="For --cif2poscar: write the conventional cell instead of the\n"
        "primitive cell. For --supergroup-cif: write the per-irrep mode\n"
        "VESTA files in the parent conventional basis (_conv suffix).",
    )
    parser.add_argument(
        "--subgroup-cif",
        dest="subgroup_cif",
        default=None,
        metavar="FILE",
        help="Low-symmetry (distorted) structure file for --supergroup-cif.",
    )

    parser.add_argument(
        "--point-group",
        "--pointgroup",
        "--pg",
        dest="point_group",
        default=None,
        help="Point-group label, e.g. m-3m.",
    )
    parser.add_argument(
        "--space-group",
        "--spacegroup",
        "--sg",
        dest="space_group",
        default=None,
        help="Space-group symbol or number, e.g. Pm-3m or 221\n"
        "(for --table/--basis/--generate-basis/--coset).",
    )
    parser.add_argument(
        "--kpoint",
        # one name or three coordinates: validated per mode in main(), since
        # only --parent takes a name
        nargs="+",
        default=None,
        metavar="K",
        help="k-point in the primitive basis (space-group modes): three\n"
        "coordinates, fractions allowed (e.g. 0.5 0.5 0 or 1/2 1/2 0).\n"
        "With --parent, the k point whose irreps are all listed: its\n"
        "ISO-IR name (GM, R, X, M, ...) or the coordinates of any arm of\n"
        "its star.",
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


_DASH_VALUE_FLAGS = ("--point-group", "--pointgroup", "--pg", "--subgroup")
_NEGATIVE_FRACTION = re.compile(r"-\d+/\d+")


def _merge_dash_values(argv: list[str]) -> list[str]:
    """Merge crystallographic values starting with '-' (e.g. -43m, -3m, -1)
    into their flag as --flag=value, so argparse does not mistake them for
    options; negative fractions of --kpoint (-1/2 1/2 0) are kept as values."""
    merged: list[str] = []
    skip_next = False
    in_kpoint = False
    for index, token in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if token.startswith("--"):
            in_kpoint = token == "--kpoint"
        if (
            token in _DASH_VALUE_FLAGS
            and index + 1 < len(argv)
            and argv[index + 1].startswith("-")
            and not argv[index + 1].startswith("--")
        ):
            merged.append(f"{token}={argv[index + 1]}")
            skip_next = True
        elif in_kpoint and _NEGATIVE_FRACTION.fullmatch(token):
            # argparse takes -0.5 for a number but -1/2 for an option; a
            # leading space marks it as a value (_kpoint_tokens strips it)
            merged.append(" " + token)
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

    if args.parent:
        if args.point_group or args.space_group:
            parser.error("--parent replaces --pg/--sg; give the parent space "
                         "group directly as --parent SG.")
        if args.irrep and args.kpoint is not None:
            parser.error("--parent takes either --irrep (one irrep, or coupled "
                         "irreps) or --kpoint (every irrep of a k point), not both.")
        if args.kpoint is not None and args.order_parameter:
            parser.error("--order-parameter selects a direction of one --irrep; "
                         "it is not used with --kpoint.")
        if not args.irrep and args.kpoint is None:
            message = ("--parent requires --irrep or --kpoint (e.g. --irrep GM4- "
                       "for one irrep, --kpoint GM for every irrep of a k point).")
            try:
                from ..isotropy_subgroup import _available_kpoints
                from ..spacegroup_product import SpaceGroupIrrepAlgebra

                algebra = SpaceGroupIrrepAlgebra(args.parent)
                message += (
                    f"\nk points of {algebra.sg_type.international_short} "
                    f"(No. {algebra.sg_type.number}), primitive basis: "
                    f"{_available_kpoints(algebra)}"
                )
            except SystemExit as exc:
                # unknown space group: say so instead of listing k points
                message += "\n" + str(exc).removeprefix("ERROR: ")
            parser.error(message)
        if args.kpoint is not None:
            # one token: the isotropy module splits it again (a name or
            # three coordinates), and negative fractions survive its parser
            dispatch_argv = [f"--parent={args.parent}",
                             f"--kpoint={' '.join(_kpoint_tokens(args.kpoint))}"]
        else:
            dispatch_argv = [f"--parent={args.parent}", "--irrep", *args.irrep]
            if args.order_parameter:
                dispatch_argv.append("--order-parameter")
                dispatch_argv.extend(args.order_parameter)

        from ..isotropy_subgroup import main as isotropy_subgroup_main

        isotropy_subgroup_main(dispatch_argv)
        return
    if args.irrep or args.order_parameter:
        parser.error("--irrep/--order-parameter are only used with --parent.")
    if args.kpoint is not None:
        args.kpoint = _kpoint_coordinates(parser, args.kpoint)

    if args.supergroup_cif:
        if not args.subgroup_cif:
            parser.error("--supergroup-cif requires --subgroup-cif "
                         "(the low-symmetry structure).")
        if args.point_group or args.space_group:
            parser.error("--supergroup-cif does not use --pg/--sg.")

        dispatch_argv = [f"--supergroup-cif={args.supergroup_cif}",
                         f"--subgroup-cif={args.subgroup_cif}"]
        if args.tolerance is not None:
            dispatch_argv.append(f"--tolerance={args.tolerance}")
        if args.conventional:
            dispatch_argv.append("--conventional")

        from ..symmetry_mode import main as symmetry_mode_main

        symmetry_mode_main(dispatch_argv)
        return
    if args.subgroup_cif:
        parser.error("--subgroup-cif is only used with --supergroup-cif.")

    if args.poscar2cif:
        if not args.cell:
            parser.error("--poscar2cif requires -c/--cell (the POSCAR file).")
        if args.point_group or args.space_group:
            parser.error("--poscar2cif does not use --pg/--sg.")

        dispatch_argv = [f"--cell={args.cell}"]
        if args.tolerance is not None:
            dispatch_argv.append(f"--tolerance={args.tolerance}")
        if args.output:
            dispatch_argv.append(f"--output={args.output}")

        from ..poscar2cif import main as poscar2cif_main

        poscar2cif_main(dispatch_argv)
        return
    if args.cif2poscar:
        if not args.cell:
            parser.error("--cif2poscar requires -c/--cell (the CIF file).")
        if args.point_group or args.space_group:
            parser.error("--cif2poscar does not use --pg/--sg.")

        dispatch_argv = [f"--cell={args.cell}"]
        if args.tolerance is not None:
            dispatch_argv.append(f"--tolerance={args.tolerance}")
        if args.conventional:
            dispatch_argv.append("--conventional")
        if args.output:
            dispatch_argv.append(f"--output={args.output}")

        from ..poscar2cif import cif2poscar_main

        cif2poscar_main(dispatch_argv)
        return
    if (args.cell or args.tolerance is not None or args.conventional
            or (args.output and not args.multiplet)):
        parser.error("-c/--cell, --tolerance, --output, and --conventional are "
                     "only used with --poscar2cif/--cif2poscar/--supergroup-cif "
                     "(--output also with --multiplet --visualize).")

    if args.multiplet:
        require_point_group("--multiplet")
        if args.space_group:
            parser.error("--multiplet works with point groups only (--pg/--point-group).")
        for name, value in (
            ("--kpoint", args.kpoint),
            ("--subgroup", args.subgroup),
            ("--order", args.order),
            ("--characters", args.characters),
        ):
            if value is not None:
                parser.error(f"{name} is not used with --multiplet.")

        dispatch_argv = [f"--point-group={args.point_group}", "--config", *args.multiplet]
        if args.orbital:
            dispatch_argv.extend(["--orbital", args.orbital])
        if args.visualize:
            dispatch_argv.append("--visualize")
        if args.output:
            dispatch_argv.extend(["--output", args.output])

        from ..multiplet import main as multiplet_main

        multiplet_main(dispatch_argv)
        return
    if args.orbital:
        parser.error("--orbital is only used with --multiplet.")
    if args.visualize:
        parser.error("--visualize is only used with --multiplet.")

    if args.product:
        require_exactly_one_group("--product")
        for name, value in (
            ("--kpoint", args.kpoint),
            ("--subgroup", args.subgroup),
            ("--order", args.order),
            ("--characters", args.characters),
        ):
            if value is not None:
                parser.error(f"{name} is not used with --product.")

        if args.space_group:
            if args.show_irrep_table:
                parser.error("--show-irrep-table is not available for space-group products.")
            from ..spacegroup_product import main as spacegroup_product_main

            spacegroup_product_main(
                [f"--space-group={args.space_group}", "--irreps", *args.product]
            )
            return

        dispatch_argv = [f"--point-group={args.point_group}", "--irreps", *args.product]
        if args.show_irrep_table:
            dispatch_argv.append("--show-irrep-table")

        from ..direct_product import main as direct_product_main

        direct_product_main(dispatch_argv)
        return

    if args.table:
        if args.space_group:
            # character table of the little group of k for a space group
            # (ISO-IR labels at tabulated k points and on lines/planes)
            if args.kpoint is None:
                parser.error("--table with --space-group requires --kpoint.")

            from ..basis_function import format_spacegroup_table

            print(format_spacegroup_table(args.space_group, args.kpoint))
            return
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
