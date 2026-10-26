"""crystod-md: molecular-dynamics trajectory analyses.

Modes:

- ``--adp``     -- time-averaged structure and symmetry-constrained
  atomic displacement parameters (ADPs)
  from an MD trajectory, written as a CIF file
  (backed by :mod:`crystod.xdatcar_adp`);
- ``--summary`` -- time-averaged lattice parameters and cell volume of the
  trajectory (backed by :mod:`crystod.md_summary`);
- ``--project`` -- projection of every frame onto the symmetry-adapted
  displacement patterns of the parent cell (``-c``): order-parameter time
  series Q(t) per irrep and q point, their statistics and histograms
  (backed by :mod:`crystod.md_projection`); with ``--yaml`` (force
  constants) the basis is the phonon eigenvectors at each q, one normal
  coordinate per mode, with effective frequencies at ``--temperature``.

``--format`` selects the trajectory format; only ``vasp`` (XDATCAR) is
implemented, with LAMMPS support planned as a future format.
"""

from __future__ import annotations

import os
from argparse import ArgumentParser, RawTextHelpFormatter

from .common import (
    CRYSTOD_CITATION,
    add_cell_argument,
    add_output_argument,
    banner,
    mark_negative_fractions,
)

desc = """\
Analyze an MD trajectory:
--adp     computes the time-averaged structure and symmetry-constrained
          atomic displacement parameters (ADPs, U_ij) and writes them as a CIF file;
--summary reports the time-averaged lattice parameters (a, b, c, alpha, beta,
          gamma) and cell volume with standard deviations;
--project projects every frame onto the symmetry-adapted displacement patterns
          of the parent cell (-c) and reports the order-parameter time series
          Q(t) per irrep and q point with their statistics; with --yaml
          (phonopy_params.yaml) onto the phonon eigenvectors at each q, one
          normal coordinate per mode (effective frequencies with --temperature).

# Command Examples:
crystod-md --adp --dim 4 4 4 --start-step 1000
crystod-md --adp --dim="4 4 4" --start-step 1000 --xdatcar XDATCAR --output ADP_ScF3_300K.cif --format vasp
crystod-md --adp --dim 4 0 0  0 4 0  0 0 4 --start-step 1000
crystod-md --summary --start-step 1000 --xdatcar XDATCAR --format vasp
crystod-md --project -c 221_PPOSCAR_ScF3 --dim 4 4 4 --start-step 1000 --qpoint R M
crystod-md --project --yaml phonopy_params.yaml --dim 4 4 4 --start-step 1000 --qpoint R M GM --temperature 300
crystod-md --project -c 221_PPOSCAR_ScF3 --dim 2 2 2 --qpoint M --star
"""


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="crystod-md",
        description=f"{banner()}\n\n{desc}",
        epilog=CRYSTOD_CITATION,
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
    mode.add_argument(
        "--project",
        action="store_true",
        help="Project the trajectory onto the symmetry-adapted displacement patterns\n"
        "of the parent cell (-c), or onto its phonon eigenvectors (--yaml): time\n"
        "series per irrep/mode and q point, statistics, CSV/PDF.",
    )
    add_cell_argument(
        parser,
        " Parent primitive cell of --project (required there unless --yaml is\n"
        "given); the MD cell is --dim times this cell with the same orientation.",
        default=None,
    )
    parser.add_argument(
        "--yaml",
        default=None,
        metavar="FILE",
        help="--project: phonopy_params.yaml with force constants; project onto the\n"
        "phonon eigenvectors at each q (mass-weighted normal coordinates, modes\n"
        "numbered as crystod-phonon --modulation). Its primitive cell is the parent\n"
        "cell: -c is not needed, and if given must be that cell.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        metavar="T",
        help="--project --yaml: temperature of the run in K; adds the effective\n"
        "frequency nu_eff = sqrt(k_B T/<a^2>)/(2 pi) of every mode.",
    )
    parser.add_argument(
        "--star",
        action="store_true",
        help="--project: project every arm of the star of each requested q\n"
        "(--qpoint M gives the three M arms of a cubic parent).",
    )
    parser.add_argument(
        "--dim",
        nargs="+",
        default=None,
        metavar="N",
        help=(
            "MD supercell dimension relative to the unit cell (--adp, --project): three\n"
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
        help="Last MD step used in the analysis, inclusive (--summary, --project;\n"
        "default: last step).",
    )
    parser.add_argument(
        "--qpoint",
        nargs="+",
        default=None,
        metavar="Q",
        help="q points of --project: labels (R M X GM), coordinates (1/2 1/2 1/2),\n"
        "or 'all' for every q commensurate with --dim (default: the tabulated\n"
        "special points commensurate with --dim).",
    )
    parser.add_argument(
        "--irrep",
        nargs="+",
        default=None,
        metavar="IR",
        help="Keep only these ISO-IR irreps in --project, e.g. --irrep R4+ M3+\n"
        "(default: all).",
    )
    parser.add_argument(
        "--mean-lattice",
        action="store_true",
        help="--project: convert the displacements of every frame to Cartesian\n"
        "with the time-averaged cell (default: each frame's own cell).",
    )
    parser.add_argument(
        "--npz",
        action="store_true",
        help="--project: also write PREFIX.npz (steps, column names, Q arrays).",
    )
    parser.add_argument(
        "--no-histogram",
        action="store_true",
        help="--project: do not write the histogram PDF.",
    )
    add_output_argument(
        parser,
        "Output CIF path in --adp mode (default: ADP.cif); output file prefix\n"
        "in --project mode (default: MDPROJ_<formula>, with --yaml\n"
        "MDPROJ_<formula>_phonon).",
    )
    parser.add_argument(
        "--tolerance",
        "--symprec",
        dest="tolerance",
        type=float,
        default=None,
        help="Symmetry tolerance for spglib on the time-averaged structure\n"
        "(--adp mode; default: 0.1), or on the parent cell (--project; default: 1e-5).",
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


_PROJECT_ONLY = (
    ("cell", "-c/--cell"),
    ("yaml", "--yaml"),
    ("temperature", "--temperature"),
    ("star", "--star"),
    ("qpoint", "--qpoint"),
    ("irrep", "--irrep"),
    ("mean_lattice", "--mean-lattice"),
    ("npz", "--npz"),
    ("no_histogram", "--no-histogram"),
)


def _refuse_project_options(parser: ArgumentParser, args, mode: str) -> None:
    for dest, flag in _PROJECT_ONLY:
        if getattr(args, dest):
            parser.error(f"{flag} is only available with --project, not with {mode}.")


def _project(parser: ArgumentParser, args) -> None:
    if not args.cell and not args.yaml:
        parser.error("--project requires -c/--cell (the parent primitive cell) or --yaml "
                     "(phonopy_params.yaml).")
    if args.temperature is not None and not args.yaml:
        parser.error("--temperature requires --yaml (the effective frequencies are compared "
                     "with the harmonic phonon frequencies).")
    if args.yaml and not os.path.isfile(args.yaml):
        parser.error(f"phonopy yaml not found: {args.yaml}")
    if not args.dim:
        parser.error("--project requires --dim.")
    if args.grouping_tolerance is not None:
        parser.error("--project does not use --grouping-tolerance.")
    dim = _parse_dim(parser, args.dim)
    dispatch_argv = [
        "--dim",
        dim,
        "--xdatcar",
        args.xdatcar,
        "--start-step",
        str(args.start_step),
    ]
    if args.cell:
        dispatch_argv.extend(["--cell", args.cell])
    if args.yaml:
        dispatch_argv.extend(["--yaml", args.yaml])
    if args.temperature is not None:
        dispatch_argv.extend(["--temperature", repr(args.temperature)])
    if args.star:
        dispatch_argv.append("--star")
    if args.end_step is not None:
        dispatch_argv.extend(["--end-step", str(args.end_step)])
    if args.qpoint:
        dispatch_argv.extend(["--qpoint", *args.qpoint])
    if args.irrep:
        dispatch_argv.extend(["--irrep", *args.irrep])
    if args.mean_lattice:
        dispatch_argv.append("--mean-lattice")
    if args.npz:
        dispatch_argv.append("--npz")
    if args.no_histogram:
        dispatch_argv.append("--no-histogram")
    if args.output:
        dispatch_argv.extend(["--output", args.output])
    if args.tolerance is not None:
        dispatch_argv.extend(["--symprec", str(args.tolerance)])

    from ..md_projection import main as md_projection_main

    md_projection_main(dispatch_argv)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    if argv is None:
        import sys

        argv = sys.argv[1:]
    args = parser.parse_args(mark_negative_fractions(list(argv)))

    if not os.path.isfile(args.xdatcar):
        parser.error(f"trajectory file not found: {args.xdatcar}")

    if args.project:
        _project(parser, args)
        return

    _refuse_project_options(parser, args, "--adp" if args.adp else "--summary")

    if args.adp:
        if not args.dim:
            parser.error("--adp requires --dim.")
        if args.end_step is not None:
            parser.error("--end-step is only available with --summary and --project.")

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
