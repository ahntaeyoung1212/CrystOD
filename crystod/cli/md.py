"""crystod-md: molecular-dynamics trajectory analyses.

Modes:

- ``--adp``     -- the former ``crystod --xdatcar2adp``: time-averaged
  structure and symmetry-constrained atomic displacement parameters (ADPs)
  from an MD trajectory, written as a CIF file
  (backed by :mod:`crystod.xdatcar_adp`);
- ``--summary`` -- time-averaged lattice parameters and cell volume of the
  trajectory (backed by :mod:`crystod.md_summary`).

``--format`` selects the trajectory format; only ``vasp`` (XDATCAR) is
implemented, with LAMMPS support planned as a future format.
"""

from __future__ import annotations

import os
from argparse import ArgumentParser, RawTextHelpFormatter

from .common import add_output_argument, banner

desc = """\
Analyze an MD trajectory:
--adp     computes the time-averaged structure and symmetry-constrained
          atomic displacement parameters (ADPs, U_ij) and writes them as a CIF file;
--summary reports the time-averaged lattice parameters (a, b, c, alpha, beta,
          gamma) and cell volume with standard deviations.

# Command Examples:
crystod-md --adp --dim 4 4 4 --start-step 1000
crystod-md --adp --dim="4 4 4" --start-step 1000 --xdatcar XDATCAR --output ADP_ScF3_300K.cif --format vasp
crystod-md --adp --dim 4 0 0  0 4 0  0 0 4 --start-step 1000
crystod-md --summary --start-step 1000 --xdatcar XDATCAR --format vasp
"""


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="crystod-md",
        description=f"{banner()}\n\n{desc}",
        formatter_class=RawTextHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--adp",
        action="store_true",
        help="Compute atomic displacement parameters (ADPs, U_ij) as a CIF file.",
    )
    mode.add_argument(
        "--summary",
        action="store_true",
        help="Summarize the trajectory: time-averaged lattice parameters and volume.",
    )
    parser.add_argument(
        "--dim",
        nargs="+",
        default=None,
        metavar="N",
        help=(
            "MD supercell dimension relative to the unit cell (--adp mode): three\n"
            "diagonal values or a nine-value diagonal matrix, quoted or unquoted,\n"
            'e.g. --dim 4 4 4, --dim="4 4 4", --dim 4 0 0 0 4 0 0 0 4,\n'
            '--dim="4 0 0  0 4 0  0 0 4". Non-diagonal matrices are not supported.'
        ),
    )
    parser.add_argument(
        "--format",
        choices=["vasp"],
        default="vasp",
        help="Trajectory format (default: vasp = XDATCAR). LAMMPS support is planned.",
    )
    parser.add_argument(
        "--xdatcar",
        default="XDATCAR",
        metavar="FILE",
        help="Input XDATCAR path (default: XDATCAR).",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=0,
        metavar="N",
        help="First MD step used in the analysis; earlier steps are discarded "
        "as equilibration (default: 0).",
    )
    parser.add_argument(
        "--end-step",
        type=int,
        default=None,
        metavar="N",
        help="Last MD step used in the analysis, inclusive (--summary mode; default: last step).",
    )
    add_output_argument(parser, "Output CIF path in --adp mode (default: ADP.cif).")
    parser.add_argument(
        "--tolerance",
        "--symprec",
        dest="tolerance",
        type=float,
        default=None,
        help="Symmetry tolerance for spglib on the time-averaged structure "
        "(--adp mode; default: 0.1).",
    )
    parser.add_argument(
        "--grouping-tolerance",
        type=float,
        default=None,
        metavar="TOL",
        help="Tolerance for grouping supercell atoms into unit-cell sites "
        "(--adp mode; default: 0.1).",
    )
    return parser


def _parse_dim(parser: ArgumentParser, tokens: list[str]) -> str:
    """Normalize --dim to the three diagonal values as one string ("4 4 4").

    Accepts three diagonal values or a nine-value diagonal matrix, quoted
    ("4 4 4" arrives as one token) or unquoted (separate tokens).
    """
    flat = " ".join(tokens).split()
    try:
        values = [int(token) for token in flat]
    except ValueError:
        parser.error(f'--dim requires integers, got: {" ".join(flat)}')
    if len(values) == 3:
        diagonal = values
    elif len(values) == 9:
        off_diagonal = [values[i] for i in range(9) if i not in (0, 4, 8)]
        if any(off_diagonal):
            parser.error(
                "--dim with nine values must be a diagonal matrix; "
                "non-diagonal MD supercells are not supported."
            )
        diagonal = [values[0], values[4], values[8]]
    else:
        parser.error(f"--dim requires three or nine integers, got {len(values)}.")
    if any(value <= 0 for value in diagonal):
        parser.error("--dim diagonal values must be positive integers.")
    return " ".join(str(value) for value in diagonal)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not os.path.isfile(args.xdatcar):
        parser.error(f"trajectory file not found: {args.xdatcar}")

    if args.adp:
        if not args.dim:
            parser.error("--adp requires --dim.")
        if args.end_step is not None:
            parser.error("--end-step is only available with --summary.")

        dim = _parse_dim(parser, args.dim)

        dispatch_argv = [
            "--dim",
            dim,
            "--xdatcar",
            args.xdatcar,
            "--start-step",
            str(args.start_step),
        ]
        if args.output:
            dispatch_argv.extend(["--output", args.output])
        if args.tolerance is not None:
            dispatch_argv.extend(["--symprec", str(args.tolerance)])
        if args.grouping_tolerance is not None:
            dispatch_argv.extend(["--grouping-tolerance", str(args.grouping_tolerance)])

        from ..xdatcar_adp import main as xdatcar_adp_main

        xdatcar_adp_main(dispatch_argv)
        return

    # --summary
    if args.dim:
        parser.error("--summary does not use --dim.")
    if args.output:
        parser.error("--summary does not use --output.")
    if args.tolerance is not None or args.grouping_tolerance is not None:
        parser.error("--summary does not use --tolerance/--grouping-tolerance.")

    dispatch_argv = [
        "--xdatcar",
        args.xdatcar,
        "--start-step",
        str(args.start_step),
    ]
    if args.end_step is not None:
        dispatch_argv.extend(["--end-step", str(args.end_step)])

    from ..md_summary import main as md_summary_main

    md_summary_main(dispatch_argv)


if __name__ == "__main__":
    main()
