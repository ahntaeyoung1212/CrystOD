"""crystod-group: representation-theory calculator for point and space groups.

The representation-theory modes:

- ``--product IRREP...``   -- direct-product decomposition (with
  ``--symmetric`` / ``--antisymmetric``: the squares of one irrep);
- ``--jahn-teller IRREP``  -- Jahn-Teller active modes (with two irreps: the
  symmetry-allowed pseudo-Jahn-Teller coupling modes);
- ``--table``              -- display the point-group character table;
- ``--decompose``          -- reducible-representation decomposition;
- ``--ligand-field ORB``   -- ligand-field splitting of an atomic orbital;
- ``--basis FUNC...``      -- classify polynomial basis functions;
- ``--generate-basis``     -- auto-generate 1st-3rd order polynomial bases;
- ``--tensor KIND``        -- symmetry-allowed form of a property tensor
  (Neumann's principle; ``--pg``, ``--sg`` or ``-c POSCAR``);
- ``--coset``              -- coset decompositions;
- ``--correlate``          -- point-group correlation tables (with ``--pg``
  and ``--subgroup``), subduction to an isotropy subgroup (with
  ``--parent``) and compatibility relations along a line (with ``--sg``).

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
crystod-group --product T2g T2g --pg m-3m --symmetric [--antisymmetric]
crystod-group --product R4+ R4+ --sg Pm-3m --symmetric [--antisymmetric]
crystod-group --jahn-teller Eg --pg m-3m              (Eg T2g: pseudo-Jahn-Teller)
crystod-group --multiplet T2g2 --pg m-3m [--orbital d]
crystod-group --poscar2cif -c PPOSCAR [--tolerance 0.01]
crystod-group --cif2poscar -c FILE.cif [--conventional]
crystod-group --parent Pm-3m --irrep R4+              (--supergroup is an alias)
crystod-group --parent Pm-3m --kpoint GM              (every irrep of the k point)
crystod-group --parent Pm-3m --irrep R4+ --invariants [--degree 4]
crystod-group --parent I4/mmm --irrep X2+ X3- GM5- --invariants --degree 3
crystod-group --parent Pm-3m --irrep R4+ --order-parameter a 0 0 --secondary
crystod-group --parent Pm-3m --child I4/mcm [--size N] [--index N] [--kpoint R]
crystod-group --parent Pm-3m --child Pnma --size 4 --coupled [--secondary]
crystod-group --parent Pm-3m --irrep R4+ M3+ --graph [--output FILE] [--graph-dot]
crystod-group --correlate --pg m-3m --subgroup 4/mmm
crystod-group --correlate --parent Pm-3m --irrep R4+ --order-parameter 0 0 a [--irrep-list R4+ R5+]
crystod-group --correlate --sg Pm-3m --kpoint GM X [--line DT]
crystod-group --supergroup-cif 221.cif --subgroup-cif 140.cif [--output-dir DIR | --no-files]
crystod-group --table --pg 3m
crystod-group --decompose --pg 3m --characters 3 0 1
crystod-group --ligand-field d --pg m-3m
crystod-group --basis x y z --pg m-3m
crystod-group --basis x y z --sg Pm-3m --kpoint 0 0 0
crystod-group --generate-basis --pg m-3m --order 1 2 3
crystod-group --tensor piezoelectric --pg 4mm         (dielectric, elastic, raman, ...)
crystod-group --tensor "e[V2]" --sg P4_32_12          (a Jahn symbol; or -c POSCAR)
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
        "--jahn-teller",
        dest="jahn_teller",
        nargs="+",
        default=None,
        metavar="IRREP",
        help="Jahn-Teller active modes of a degenerate level, [IR x IR] minus\n"
        "the identity irrep, e.g. --jahn-teller Eg --pg m-3m; with two irreps,\n"
        "the symmetry-allowed pseudo-Jahn-Teller coupling modes of IR1 x IR2\n"
        "(a necessary condition only; point groups only).",
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
        "--tensor",
        default=None,
        metavar="KIND",
        help="Symmetry-allowed form of a property tensor (Neumann's principle):\n"
        "independent components, matrix form (Nye/Voigt) and relations in\n"
        "the point group of --pg, --sg or -c POSCAR. KIND is dielectric,\n"
        "pyroelectric, piezoelectric, elastic, compliance, gyration, raman\n"
        "(the Raman tensor of every Raman-active irrep) or a Jahn symbol\n"
        "(V, [V2], V[V2], [[V2]2], e[V2], ...; e = axial, a = time-reversal\n"
        'odd; quote brackets in the shell), e.g. --tensor piezoelectric --pg 4mm.',
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
    mode.add_argument(
        "--correlate",
        action="store_true",
        help="Correlation of irreps. With --pg G --subgroup H: the correlation\n"
        "table of the point group G, one column per inequivalent orientation\n"
        "of H, e.g. --pg m-3m --subgroup 4/mmm gives T2g -> B2g + Eg.\n"
        "With --parent SG --irrep IR\n"
        "--order-parameter C...: the irreps of the Gamma point of the isotropy\n"
        "subgroup H that parent irreps become (--irrep-list, default: the\n"
        "--irrep labels), e.g. R4+ (0,0,a) -> GM1+ + GM5+ of I4/mcm. With\n"
        "--sg SG --kpoint K0 K1: the compatibility relations of the small\n"
        "irreps at K0 and K1 along the line joining them (one name: every\n"
        "line from K0), e.g. --sg Pm-3m --kpoint GM X gives GM4- -> DT1 + DT5.",
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
        help='Order-parameter direction for --parent, e.g. 0 0 a or a -a 0\n'
        "(letters = free parameters, -a = minus a, 2a or 1/2a = a multiple;\n"
        'also one string as printed: "a;-a;0"). Omit to list every direction.',
    )
    parser.add_argument(
        "--invariants",
        action="store_true",
        help="For --parent with --irrep: list the invariant polynomials of\n"
        "the order-parameter components (the terms of the Landau free\n"
        "energy) degree by degree, checked against the Molien series,\n"
        "instead of the isotropy subgroups. Several irreps give the direct\n"
        "sum by multidegree and its lowest-order coupling term. With\n"
        "--order-parameter, also the free energy restricted to that direction.",
    )
    parser.add_argument(
        "--secondary",
        action="store_true",
        help="For --parent with --irrep and --order-parameter: list the\n"
        "secondary order parameters of the isotropy subgroup (irreps at Gamma\n"
        "and at the special k points with a component fixed by it), with\n"
        "their direction, lowest coupling term and type.",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=None,
        metavar="N",
        help="Highest degree of the polynomials for --invariants and of the\n"
        "coupling terms for --secondary (default: 4).",
    )
    parser.add_argument(
        "--child",
        default=None,
        metavar="H",
        help="For --parent: reverse lookup, list every irrep stratum at the\n"
        "special k points whose isotropy subgroup has type H (symbol or\n"
        "number), with its direction, size, index and setting, e.g.\n"
        "--parent Pm-3m --child I4/mcm. The table of all strata of the\n"
        "parent is built once and cached (CRYSTOD_CACHE_DIR, default\n"
        "~/.cache/crystod). --kpoint K [K ...] keeps those k points only.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        metavar="N",
        help="For --child: keep the strata with this cell size only.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        metavar="N",
        help="For --child: keep the strata with this index [G:H] only.",
    )
    parser.add_argument(
        "--no-cache",
        dest="no_cache",
        action="store_true",
        help="For --child: rebuild the isotropy table instead of reading the\n"
        "cached one (the cache file is rewritten).",
    )
    parser.add_argument(
        "--coupled",
        action="store_true",
        help="For --child: also search pairs of coupled irreps, first two\n"
        "irreps at the same k point, then at different k points, and list\n"
        "the coupled strata of type H (both irreps needed), e.g.\n"
        "--parent Pm-3m --child Pnma --size 4 --coupled gives R4+ + M3+.\n"
        "With --secondary, the secondary order parameters of every answer.",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="For --parent with --irrep: write the group-subgroup graph of\n"
        "the isotropy subgroups (the parent, every stratum of the irreps\n"
        "and of their direct sum, and the kernel), layered by index, as an\n"
        "HTML page SUBGROUP_<SG>_<irreps>.html (--output to rename); edges\n"
        "from the parent are dashed when the Landau or Lifshitz condition\n"
        "forbids a continuous transition.",
    )
    parser.add_argument(
        "--graph-dot",
        dest="graph_dot",
        action="store_true",
        help="For --graph: also write the graph as a Graphviz .dot file.",
    )
    parser.add_argument(
        "--irrep-list",
        dest="irrep_list",
        nargs="+",
        default=None,
        metavar="IR",
        help="For --correlate --parent: the parent irreps subduced to the Gamma\n"
        "point of the isotropy subgroup (any tabulated irrep; default: the\n"
        "--irrep labels), e.g. --irrep-list R4+ R5+ M3+ GM4-.",
    )
    parser.add_argument(
        "--line",
        default=None,
        metavar="L",
        help="For --correlate --sg --kpoint K0 K1: the ISO-IR name of the line\n"
        "(e.g. DT) when several lines join the two points (default: the line\n"
        "with the largest little group, then the shortest).",
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
        help="Input structure file: POSCAR for --poscar2cif and --tensor,\n"
        "CIF for --cif2poscar.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Symmetry tolerance (symprec, Angstrom) for --poscar2cif/"
        "--cif2poscar (default: 0.01)\nand for --tensor -c (default: 1e-5).",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Output path for --poscar2cif/--cif2poscar (defaults: <POSCAR>.cif "
        "/ input without .cif), for --multiplet --visualize or for --graph.",
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
        "--output-dir",
        dest="output_dir",
        default=None,
        metavar="DIR",
        help="For --supergroup-cif: directory for the decomposition table\n"
        "(sym_mode_<formula>) and the per-irrep VESTA files, created if\n"
        "missing (default: the current directory).",
    )
    parser.add_argument(
        "--no-files",
        dest="no_files",
        action="store_true",
        help="For --supergroup-cif: print the analysis only and write no\n"
        "table or VESTA files.",
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
        "its star. With --correlate --sg, one or two special-point names.",
    )
    parser.add_argument(
        "--subgroup",
        default=None,
        help="Subgroup point-group label for --coset and --correlate with\n"
        "--point-group, e.g. 4/mmm.",
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
    parser.add_argument(
        "--symmetric",
        action="store_true",
        help="For --product with two identical irreps: also decompose the\n"
        "symmetric square [IR x IR] (point or space group).",
    )
    parser.add_argument(
        "--antisymmetric",
        action="store_true",
        help="For --product with two identical irreps: also decompose the\n"
        "antisymmetric square {IR x IR} (point or space group).",
    )
    return parser


_DASH_VALUE_FLAGS = ("--point-group", "--pointgroup", "--pg", "--subgroup", "--child")
_NEGATIVE_FRACTION = re.compile(r"-\d+/\d+")


def _merge_dash_values(argv: list[str]) -> list[str]:
    """Merge crystallographic values starting with '-' (e.g. -43m, -3m, -1)
    into their flag as --flag=value, so argparse does not mistake them for
    options; negative fractions of --kpoint (-1/2 1/2 0) and negative
    --order-parameter components (a -a 0, -0.5a) are kept as values."""
    merged: list[str] = []
    skip_next = False
    in_kpoint = False
    in_order_parameter = False
    for index, token in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if token.startswith("--"):
            in_kpoint = token == "--kpoint"
            # any unambiguous abbreviation (--order alone is --order N)
            name = token.split("=", 1)[0]
            in_order_parameter = (len(name) > len("--order")
                                  and "--order-parameter".startswith(name))
        elif in_order_parameter and token.startswith("-"):
            # every token up to the next --option is a component; -a,
            # -0.5a, -b are taken for options by argparse, so a leading
            # space marks them as values (stripped by the isotropy module)
            merged.append(" " + token)
            continue
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


# options --correlate does not use (attribute name, flag), besides the ones
# each of its two forms checks itself
_CORRELATE_UNUSED = (
    ("invariants", "--invariants"), ("secondary", "--secondary"), ("degree", "--degree"),
    ("child", "--child"), ("size", "--size"), ("index", "--index"),
    ("no_cache", "--no-cache"), ("coupled", "--coupled"), ("graph", "--graph"),
    ("graph_dot", "--graph-dot"), ("cell", "-c/--cell"),
    ("tolerance", "--tolerance"), ("output", "--output"), ("conventional", "--conventional"),
    ("subgroup_cif", "--subgroup-cif"), ("orbital", "--orbital"),
    ("visualize", "--visualize"), ("subgroup", "--subgroup"), ("order", "--order"),
    ("characters", "--characters"), ("show_irrep_table", "--show-irrep-table"),
    ("symmetric", "--symmetric"), ("antisymmetric", "--antisymmetric"),
)


def _correlate_unused(parser: ArgumentParser, args, extra=(), exclude=()) -> None:
    for attribute, flag in _CORRELATE_UNUSED + tuple(extra):
        if attribute in exclude:
            continue
        value = getattr(args, attribute, None)
        if value is not None and value is not False:
            parser.error(f"{flag} is not used with --correlate.")


def _correlate_parent_argv(parser: ArgumentParser, args) -> list[str]:
    """Checked dispatch arguments of --correlate --parent (subduction)."""
    if args.kpoint is not None or args.line is not None:
        parser.error("--kpoint/--line are only used with --correlate --sg SG.")
    _correlate_unused(parser, args)
    if not args.irrep or not args.order_parameter:
        parser.error("--correlate --parent requires --irrep and --order-parameter "
                     "(the direction whose isotropy subgroup H the irreps are "
                     "subduced to), e.g. --irrep R4+ --order-parameter 0 0 a.")
    argv = [f"--parent={args.parent}", "--irrep", *args.irrep,
            "--order-parameter", *args.order_parameter]
    if args.irrep_list:
        argv += ["--irrep-list", *args.irrep_list]
    return argv


def _correlate_pg_argv(parser: ArgumentParser, args) -> list[str]:
    """Checked dispatch arguments of --correlate --pg (correlation table)."""
    if args.space_group:
        parser.error("--correlate takes either --pg G --subgroup H (correlation table) "
                     "or --sg SG --kpoint K0 K1 (compatibility relations), not both.")
    if (args.irrep or args.order_parameter or args.irrep_list or args.kpoint is not None
            or args.line is not None):
        parser.error("--irrep/--order-parameter/--irrep-list/--kpoint/--line are not "
                     "used with --correlate --pg G.")
    if not args.subgroup:
        parser.error("--correlate --pg requires --subgroup H (the subgroup type), "
                     "e.g. --pg m-3m --subgroup 4/mmm.")
    _correlate_unused(parser, args, exclude=("subgroup",))
    return [f"--pg={args.point_group}", f"--subgroup={args.subgroup}"]


def _correlate_sg_argv(parser: ArgumentParser, args) -> list[str]:
    """Checked dispatch arguments of --correlate --sg (compatibility)."""
    if not args.space_group or args.point_group:
        parser.error("--correlate requires --parent SG --irrep IR --order-parameter C... "
                     "(subduction to the isotropy subgroup), --sg SG --kpoint K0 K1 "
                     "(compatibility relations along a line) or --pg G --subgroup H "
                     "(point-group correlation table).")
    if args.irrep or args.order_parameter or args.irrep_list:
        parser.error("--irrep/--order-parameter/--irrep-list are only used with "
                     "--correlate --parent SG.")
    _correlate_unused(parser, args)
    tokens = _kpoint_tokens(args.kpoint) if args.kpoint is not None else []
    if len(tokens) not in (1, 2):
        parser.error("--correlate --sg requires --kpoint with two special-point names "
                     "(e.g. --kpoint GM X), or one name for every line from it.")
    argv = [f"--sg={args.space_group}", "--kpoint", *tokens]
    if args.line is not None:
        argv.append(f"--line={args.line}")
    return argv


def main(argv: list[str] | None = None) -> None:
    import sys

    parser = build_parser()
    if argv is None:
        argv = sys.argv[1:]
    # --correlate shares the mode group with --parent; next to --parent it
    # selects the subduction form of --parent, so it is taken off here
    correlate_parent = "--correlate" in argv and any(
        token.split("=", 1)[0] in ("--parent", "--supergroup") for token in argv)
    if correlate_parent:
        argv = [token for token in argv if token != "--correlate"]
    names = [token.split("=", 1)[0] for token in argv if token.startswith("--")]
    if "--child" in names and not {"--parent", "--supergroup"} & set(names):
        # before argparse, which would only ask for a mode
        raise SystemExit("ERROR: --child needs --parent SG (the parent space group), "
                         "e.g. --parent Pm-3m --child I4/mcm.")
    args = parser.parse_args(_merge_dash_values(list(argv)))

    def require_point_group(mode_name: str) -> None:
        if not args.point_group:
            parser.error(f"{mode_name} requires --pg/--point-group.")

    def require_exactly_one_group(mode_name: str) -> None:
        if bool(args.point_group) == bool(args.space_group):
            parser.error(
                f"{mode_name} requires exactly one of --pg/--point-group or --sg/--space-group."
            )

    if (args.symmetric or args.antisymmetric) and not args.product:
        parser.error("--symmetric/--antisymmetric are only used with --product "
                     "(two identical irreps, e.g. --product T2g T2g).")
    if (args.output_dir is not None or args.no_files) and not args.supergroup_cif:
        parser.error("--output-dir and --no-files are only used with --supergroup-cif.")
    if args.output_dir is not None and args.no_files:
        parser.error("--output-dir and --no-files exclude each other "
                     "(--no-files writes nothing).")

    if args.parent:
        if args.point_group or args.space_group:
            parser.error("--parent replaces --pg/--sg; give the parent space "
                         "group directly as --parent SG.")
        if correlate_parent:
            from ..correlation import main as correlation_main

            correlation_main(_correlate_parent_argv(parser, args))
            return
        if args.irrep_list is not None or args.line is not None:
            parser.error("--irrep-list and --line are only used with --correlate.")
        child_only = [flag for flag, used in (
            ("--size", args.size is not None), ("--index", args.index is not None),
            ("--no-cache", args.no_cache), ("--coupled", args.coupled)) if used]
        if child_only and args.child is None:
            verb = "is" if len(child_only) == 1 else "are"
            parser.error(f"{'/'.join(child_only)} {verb} only used with --parent SG "
                         "--child H.")
        graph_only = [flag for flag, used in (
            ("--graph-dot", args.graph_dot),
            ("--output", args.output is not None and not args.graph)) if used]
        if graph_only and not args.graph:
            parser.error(f"{'/'.join(graph_only)} is only used with --parent SG "
                         "--irrep IR [IR2 ...] --graph.")
        if args.child is not None:
            other = [flag for flag, used in (
                ("--irrep", args.irrep), ("--order-parameter", args.order_parameter),
                ("--invariants", args.invariants), ("--graph", args.graph),
                ("--secondary", args.secondary and not args.coupled),
                ("--degree", args.degree is not None and not args.secondary)) if used]
            if other:
                parser.error(f"--child lists the irreps that give a subgroup type; "
                             f"{'/'.join(other)} is not used with it"
                             + (" (--secondary needs --coupled)." if args.secondary
                                else "."))
            if args.degree is not None and not 2 <= args.degree <= 12:
                parser.error("--secondary needs --degree between 2 and 12.")
            dispatch_argv = [f"--parent={args.parent}", f"--child={args.child}"]
            if args.size is not None:
                dispatch_argv.append(f"--size={args.size}")
            if args.index is not None:
                dispatch_argv.append(f"--index={args.index}")
            if args.kpoint is not None:
                dispatch_argv.append(f"--kpoint={' '.join(_kpoint_tokens(args.kpoint))}")
            if args.no_cache:
                dispatch_argv.append("--no-cache")
            if args.coupled:
                dispatch_argv.append("--coupled")
            if args.secondary:
                dispatch_argv.append("--secondary")
            if args.degree is not None:
                dispatch_argv.append(f"--degree={args.degree}")

            from ..isotropy_table import main as isotropy_table_main

            isotropy_table_main(dispatch_argv)
            return
        if args.irrep and args.kpoint is not None:
            parser.error("--parent takes either --irrep (one irrep, or coupled "
                         "irreps) or --kpoint (every irrep of a k point), not both.")
        if args.kpoint is not None and args.order_parameter:
            parser.error("--order-parameter selects a direction of one --irrep; "
                         "it is not used with --kpoint.")
        if args.degree is not None and not (args.invariants or args.secondary):
            parser.error("--degree is only used with --invariants or --secondary.")
        if args.invariants:
            if args.kpoint is not None:
                parser.error("--invariants needs --irrep; it is not used "
                             "with --kpoint.")
            if not args.irrep:
                parser.error("--invariants requires --irrep (one irrep label, "
                             "e.g. --irrep R4+, or several for a direct sum).")
        if args.secondary:
            if args.kpoint is not None:
                parser.error("--secondary needs --irrep and --order-parameter; "
                             "it is not used with --kpoint.")
            if not args.irrep:
                parser.error("--secondary requires --irrep and --order-parameter.")
            if not args.order_parameter:
                parser.error("--secondary requires --order-parameter (the "
                             "direction whose isotropy subgroup is analyzed).")
        if args.degree is not None:
            if not 1 <= args.degree <= 12:
                parser.error("--degree must be between 1 and 12.")
            if args.secondary and args.degree < 2:
                parser.error("--secondary needs --degree 2 or higher (a coupling "
                             "term has degree >= 2).")
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
        if args.graph:
            other = [flag for flag, used in (
                ("--kpoint", args.kpoint is not None),
                ("--order-parameter", args.order_parameter),
                ("--invariants", args.invariants), ("--secondary", args.secondary),
                ("--degree", args.degree is not None)) if used]
            if other:
                parser.error(f"--graph draws every stratum of the --irrep labels; "
                             f"{'/'.join(other)} is not used with it.")
            dispatch_argv = [f"--parent={args.parent}", "--irrep", *args.irrep]
            if args.output is not None:
                dispatch_argv.append(f"--output={args.output}")
            if args.graph_dot:
                dispatch_argv.append("--graph-dot")

            from ..subgroup_graph import main as subgroup_graph_main

            subgroup_graph_main(dispatch_argv)
            return
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
            if args.invariants:
                dispatch_argv.append("--invariants")
            if args.secondary:
                dispatch_argv.append("--secondary")
            if args.degree is not None:
                dispatch_argv.append(f"--degree={args.degree}")

        from ..isotropy_subgroup import main as isotropy_subgroup_main

        isotropy_subgroup_main(dispatch_argv)
        return
    if args.correlate:
        from ..correlation import main as correlation_main

        if args.point_group:
            correlation_main(_correlate_pg_argv(parser, args))
            return
        correlation_main(_correlate_sg_argv(parser, args))
        return
    if args.irrep_list is not None or args.line is not None:
        parser.error("--irrep-list and --line are only used with --correlate.")
    if (args.child is not None or args.size is not None or args.index is not None
            or args.no_cache or args.coupled):
        parser.error("--child/--size/--index/--no-cache/--coupled are only used "
                     "with --parent SG --child H.")
    if args.graph or args.graph_dot:
        parser.error("--graph/--graph-dot are only used with --parent SG --irrep IR "
                     "[IR2 ...].")
    if args.irrep or args.order_parameter:
        parser.error("--irrep/--order-parameter are only used with --parent.")
    if args.invariants or args.secondary or args.degree is not None:
        parser.error("--invariants/--secondary/--degree are only used with "
                     "--parent SG --irrep IR.")
    if args.kpoint is not None:
        args.kpoint = _kpoint_coordinates(parser, args.kpoint)

    if args.tensor is not None:
        sources = [flag for flag, used in (("--pg", args.point_group),
                                           ("--sg", args.space_group),
                                           ("-c", args.cell)) if used]
        if len(sources) != 1:
            parser.error("--tensor requires exactly one of --pg/--point-group, "
                         "--sg/--space-group or -c/--cell (a POSCAR), e.g. "
                         "--tensor piezoelectric --pg 4mm.")
        for flag, value in (("--kpoint", args.kpoint), ("--subgroup", args.subgroup),
                            ("--order", args.order), ("--characters", args.characters),
                            ("--output", args.output), ("--orbital", args.orbital),
                            ("--subgroup-cif", args.subgroup_cif)):
            if value is not None:
                parser.error(f"{flag} is not used with --tensor.")
        for flag, value in (("--show-irrep-table", args.show_irrep_table),
                            ("--conventional", args.conventional),
                            ("--visualize", args.visualize)):
            if value:
                parser.error(f"{flag} is not used with --tensor.")
        if args.tolerance is not None and not args.cell:
            parser.error("--tolerance is only used with --tensor -c POSCAR.")
        dispatch_argv = [f"--kind={args.tensor}"]
        if args.point_group:
            dispatch_argv.append(f"--point-group={args.point_group}")
        elif args.space_group:
            dispatch_argv.append(f"--space-group={args.space_group}")
        else:
            dispatch_argv.append(f"--cell={args.cell}")
            if args.tolerance is not None:
                dispatch_argv.append(f"--tolerance={args.tolerance}")

        from ..tensor_form import main as tensor_form_main

        tensor_form_main(dispatch_argv)
        return

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
        if args.output_dir is not None:
            dispatch_argv.append(f"--output-dir={args.output_dir}")
        if args.no_files:
            dispatch_argv.append("--no-files")

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
        square_flags = [flag for flag, wanted in (("--symmetric", args.symmetric),
                                                  ("--antisymmetric", args.antisymmetric))
                        if wanted]
        if square_flags:
            factors = [token for value in args.product for token in value.split()]
            if len(factors) != 2 or factors[0] != factors[1]:
                example = "R4+ R4+" if args.space_group else "T2g T2g"
                parser.error(f"{'/'.join(square_flags)} needs two identical irreps "
                             f"(e.g. --product {example}); got: {' '.join(factors)}.")

        if args.space_group:
            if args.show_irrep_table:
                parser.error("--show-irrep-table is not available for space-group products.")
            from ..spacegroup_product import main as spacegroup_product_main

            spacegroup_product_main(
                [f"--space-group={args.space_group}", "--irreps", *args.product,
                 *square_flags]
            )
            return

        dispatch_argv = [f"--point-group={args.point_group}", "--irreps", *args.product,
                         *square_flags]
        if args.show_irrep_table:
            dispatch_argv.append("--show-irrep-table")

        from ..direct_product import main as direct_product_main

        direct_product_main(dispatch_argv)
        return

    if args.jahn_teller:
        if args.space_group:
            parser.error("--jahn-teller works with point groups only (--pg/--point-group).")
        require_point_group("--jahn-teller")
        for name, value in (
            ("--kpoint", args.kpoint),
            ("--subgroup", args.subgroup),
            ("--order", args.order),
            ("--characters", args.characters),
        ):
            if value is not None:
                parser.error(f"{name} is not used with --jahn-teller.")
        if args.show_irrep_table:
            parser.error("--show-irrep-table is not used with --jahn-teller.")
        levels = [token for value in args.jahn_teller for token in value.split()]
        if len(levels) > 2:
            parser.error("--jahn-teller takes one irrep (Jahn-Teller) or two "
                         "(pseudo-Jahn-Teller), e.g. --jahn-teller Eg or "
                         "--jahn-teller Eg T2g.")

        from ..direct_product import jahn_teller_main

        jahn_teller_main([f"--point-group={args.point_group}", "--irreps", *levels])
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
