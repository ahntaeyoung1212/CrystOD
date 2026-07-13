"""crystod-bz: Brillouin-zone plotting (unit cell, optionally with a supercell).

Merges the former ``crystod --bz`` and ``crystod --bz-supercell`` modes into
one sectioned command. The behaviour is selected by ``--trans-mat``:

- identity matrix (the default) -> unit-cell BZ with an automatic (seekpath)
  or manual (``--band``/``--band-labels``) high-symmetry k-path
  (backed by :mod:`crystod.brillouin_zone`);
- non-identity matrix -> unit-cell BZ drawn together with the first BZ of the
  transformed (super)lattice, tiled at the |det T| unit-cell q-points folding
  onto the supercell Gamma point (backed by :mod:`crystod.bz_supercell`).
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser, RawTextHelpFormatter
from fractions import Fraction

from .common import add_cell_argument, add_output_argument, banner

desc = """\
Plot the first Brillouin zone of a crystal structure as an interactive 3D HTML file.

With the default (identity) --trans-mat, the unit-cell BZ is drawn with an
automatic (seekpath) or manual (--band/--band-labels) high-symmetry k-path.
With a non-identity --trans-mat, the unit-cell BZ (default: 1 0 0  0 1 0  0 0 1)
is drawn together with the first BZ of the transformed (super)lattice, including
the |det T| unit-cell q-points that fold onto the supercell Gamma point.

With --show-kpoint, no plot is produced; instead the special (high-symmetry)
k points of the space group given by --space-group are printed in the
primitive reciprocal basis (and additionally in the conventional basis for
centred lattices such as Fm-3m, where the two definitions differ).

# Command Examples:
crystod-bz -c 227_PPOSCAR_Si
crystod-bz -c 221_PPOSCAR_ScF3 --output BZ_ScF3_Pm-3m.html
crystod-bz --show-kpoint --space-group Pnma
crystod-bz --show-kpoint --space-group Fm-3m
crystod-bz -c 221_PPOSCAR_ScF3 \\
    --band "0 0 0  0 1/2 0  1/2 1/2 0  0 0 0  1/2 1/2 1/2  0 1/2 0, 1/2 1/2 0  1/2 1/2 1/2" \\
    --band-labels "GM X M GM R X  M R"
crystod-bz -c 221_PPOSCAR_ScF3 --trans-mat "0 1 2  -1 0 2  1 -1 2"
"""

IDENTITY_MATRIX = "1 0 0  0 1 0  0 0 1"


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="crystod-bz",
        description=f"{banner()}\n\n{desc}",
        formatter_class=RawTextHelpFormatter,
    )
    add_cell_argument(parser)
    parser.add_argument(
        "--show-kpoint",
        "--show-kpoints",
        dest="show_kpoint",
        action="store_true",
        help=(
            "Print the special k points of the space group given by --space-group\n"
            "(primitive basis; conventional basis as well for centred lattices)\n"
            "instead of plotting a Brillouin zone."
        ),
    )
    parser.add_argument(
        "--space-group",
        default=None,
        help='Space-group symbol for --show-kpoint, e.g. "Pnma" or "Fm-3m".',
    )
    parser.add_argument(
        "--trans-mat",
        "--trans-matrix",
        dest="trans_mat",
        default=IDENTITY_MATRIX,
        metavar="MATRIX",
        help=(
            "Unit-cell to supercell transformation matrix (row-wise, nine numbers),\n"
            'e.g. "0 1 2  -1 0 2  1 -1 2". Fractions such as 1/2 are allowed.\n'
            f'Default: identity ("{IDENTITY_MATRIX}") -> plot the unit-cell BZ only.\n'
            "Any non-identity matrix -> plot the unit-cell BZ together with the\n"
            "first BZ of the transformed (super)lattice."
        ),
    )
    parser.add_argument(
        "--band",
        default=None,
        help=(
            "Optional manual band path (unit-cell BZ mode only). Comma-separated\n"
            "continuous segments, each a whitespace-separated list of fractional\n"
            'coordinates, e.g. "0 0 0  0 1/2 0  1/2 1/2 0, 1/2 1/2 0  1/2 1/2 1/2".\n'
            "If omitted, the path is generated automatically with seekpath."
        ),
    )
    parser.add_argument(
        "--band-labels",
        "--label",
        dest="band_labels",
        default=None,
        help='Optional labels for the manual band path, e.g. "GM X M GM R X M R".',
    )
    add_output_argument(
        parser,
        "Output HTML path. Default: BZ_{structure name}.html "
        "(BZ_supercell_{structure name}.html with a non-identity --trans-mat).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Symmetry tolerance forwarded to seekpath/spglib (unit-cell BZ mode only).",
    )
    return parser


def _parse_trans_mat(parser: ArgumentParser, text: str) -> list[float]:
    try:
        values = [float(Fraction(token)) for token in text.split()]
    except (ValueError, ZeroDivisionError):
        parser.error(f'--trans-mat could not be parsed as numbers: "{text}"')
    if len(values) != 9:
        parser.error(
            f'--trans-mat requires nine numbers ("t11 t12 t13  t21 ..."), got {len(values)}.'
        )
    return values


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.show_kpoint:
        if not args.space_group:
            parser.error("--show-kpoint requires --space-group.")
        from ..show_kpoints import main as show_kpoints_main

        show_kpoints_main([f"--space-group={args.space_group}"])
        return
    if args.space_group:
        parser.error("--space-group is only available with --show-kpoint.")

    if args.band_labels and not args.band:
        parser.error("--band-labels requires --band.")

    is_identity = _parse_trans_mat(parser, args.trans_mat) == [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ]

    if is_identity:
        dispatch_argv = ["--poscar", args.cell]
        if args.band:
            dispatch_argv.extend(["--band", args.band])
        if args.band_labels:
            dispatch_argv.extend(["--label", args.band_labels])
        if args.output:
            dispatch_argv.extend(["--output", args.output])
        if args.tolerance is not None:
            dispatch_argv.extend(["--tolerance", str(args.tolerance)])

        from ..brillouin_zone import main as brillouin_zone_main

        brillouin_zone_main(dispatch_argv)
        return

    if args.band or args.band_labels:
        parser.error(
            "--band/--band-labels are only available in unit-cell BZ mode "
            "(identity --trans-mat)."
        )
    if args.tolerance is not None:
        print("NOTE: --tolerance is not used in supercell-BZ mode.", file=sys.stderr)

    dispatch_argv = ["--poscar", args.cell, "--trans-mat", args.trans_mat]
    if args.output:
        dispatch_argv.extend(["--output", args.output])

    from ..bz_supercell import main as bz_supercell_main

    bz_supercell_main(dispatch_argv)


if __name__ == "__main__":
    main()
