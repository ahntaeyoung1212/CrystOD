"""crystod-phonon: phonon analyses.

Merges the former phonon-related flat modes into one sectioned command:

- ``--irreps``     -- phonon irrep labeling (old ``--phonon-irrep``);
- ``--fatband``    -- element-projected phonon fatbands (old ``--phonon-fatband``);
- ``--lt``         -- longitudinal/transverse-resolved bands (old ``--phonon-lt``);
- ``--vector``     -- eigenvector VESTA export (old ``--phonon-vector``);
- ``--modulation`` -- modulated structures (old ``--modulation``);
- ``--vibration``  -- symmetry-only vibration bases, no force data
  (old ``--vibration``);
- ``--subgroup``   -- isotropy subgroups reachable from the imaginary modes.
"""

from __future__ import annotations

from argparse import ArgumentParser, RawTextHelpFormatter

from .common import CRYSTOD_CITATION, add_cell_argument, add_output_argument, banner

desc = """\
Phonon analyses from phonopy force data: a unit cell with FORCE_SETS (or
FORCE_CONSTANTS with --readfc), or a phonopy_params.yaml — except --vibration,
which needs only the crystal structure.

# Command Examples:
crystod-phonon --irreps --dim "4 4 4" -c 221_PPOSCAR_SrTiO3 --readfc
crystod-phonon --fatband --dim "4 4 4" -c 221_PPOSCAR_ScF3 --nac
crystod-phonon --lt --dim "4 4 4" -c 221_PPOSCAR_ScF3
crystod-phonon --vector --dim "4 4 4" -c 227_PPOSCAR_Si --readfc --qpoint GM
crystod-phonon --modulation -c 221_PPOSCAR_ScF3 --qpoint 0.5 0.5 0.5   (list modes and star of q only)
crystod-phonon --modulation -c 221_PPOSCAR_ScF3 --qpoint 0.5 0.5 0.5 --mode 1 2 3 --amplitude 0.3
crystod-phonon --modulation --yaml phonopy_params.yaml --qpoint 0.5 0.5 0.5 --mode 1 2 3 --amplitude 0.3
crystod-phonon --vibration -c 221_PPOSCAR_ScF3 --qpoint R
crystod-phonon --subgroup --dim "4 4 4" -c 221_PPOSCAR_ScF3   (scan every commensurate q)
crystod-phonon --subgroup --dim "4 4 4" -c 221_PPOSCAR_SrTiO3 --qpoint R --modulate
crystod-phonon --subgroup --yaml phonopy_params.yaml --qpoint R
"""


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="crystod-phonon",
        description=f"{banner()}\n\n{desc}",
        epilog=CRYSTOD_CITATION,
        formatter_class=RawTextHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--irreps",
        action="store_true",
        help="Label the phonon modes with space-group irreps (writes phonon_irreps.yaml).",
    )
    mode.add_argument(
        "--fatband",
        action="store_true",
        help="Plot element-projected phonon fatbands (PDF per element).",
    )
    mode.add_argument(
        "--lt",
        action="store_true",
        help="Plot the phonon band colored by longitudinal/transverse character.",
    )
    mode.add_argument(
        "--vector",
        action="store_true",
        help="Export phonon eigenvectors as VESTA files with displacement arrows.",
    )
    mode.add_argument(
        "--modulation",
        action="store_true",
        help="Generate a modulated structure from symmetry-adapted phonon modes.",
    )
    mode.add_argument(
        "--vibration",
        action="store_true",
        help="Construct symmetry-allowed vibration bases without phonon force data.",
    )
    mode.add_argument(
        "--subgroup",
        action="store_true",
        help="List the isotropy subgroups reachable from the imaginary phonon modes.",
    )

    # sentinel default: --modulation has to tell "-c was given" (drive the run
    # from the structure file + FORCE_SETS) from "-c was left out" (fall back to
    # phonopy_params.yaml, as before); every other mode substitutes "POSCAR"
    add_cell_argument(parser, default=None)
    parser.add_argument(
        "--dim",
        nargs="+",
        default=None,
        metavar="N",
        help=(
            "Supercell dimension of the force calculation: three diagonal values\n"
            "or a nine-value diagonal matrix, quoted or unquoted, e.g.\n"
            '--dim 4 4 4, --dim="4 4 4", --dim 4 0 0 0 4 0 0 0 4.\n'
            "Required for --irreps/--fatband/--lt/--vector, and for --subgroup\n"
            "unless --yaml is given."
        ),
    )
    parser.add_argument(
        "--readfc",
        action="store_true",
        help="Read FORCE_CONSTANTS instead of FORCE_SETS.",
    )
    parser.add_argument(
        "--nac",
        action="store_true",
        help="Apply the non-analytical term correction (LO/TO splitting, BORN file) "
        "in --fatband/--lt mode.",
    )
    parser.add_argument(
        "--element",
        default=None,
        help="Restrict --fatband output to one element.",
    )
    parser.add_argument(
        "--band",
        default=None,
        help=(
            "Optional manual band path for --fatband/--lt, comma-separated continuous\n"
            'segments, e.g. "0 0 0  0 1/2 0  1/2 1/2 0, 1/2 1/2 0  1/2 1/2 1/2".\n'
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
    parser.add_argument(
        "--npoints",
        type=int,
        default=None,
        help="Number of q-points per band-path leg for --fatband/--lt (default: 51).",
    )
    parser.add_argument(
        "--projection-direction",
        dest="projection_direction",
        default=None,
        help='Projection direction in reduced coordinates for --fatband, e.g. "0 0 1".',
    )
    parser.add_argument(
        "--qpoint",
        nargs="+",
        default=None,
        help="q-point for --vector/--modulation/--vibration/--subgroup: three coordinates\n"
        "or a high-symmetry label when supported. In --subgroup mode it is optional;\n"
        "without it every q point commensurate with the supercell is scanned.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=-0.1,
        help="Frequency in THz below which a mode counts as imaginary in --subgroup mode.",
    )
    parser.add_argument(
        "--mode",
        nargs="+",
        type=int,
        default=None,
        help="Mode number(s) for --vector/--modulation (1-based, as in the printed mode table).\n"
        "If omitted in --modulation, only the mode table and the star of q are printed.",
    )
    parser.add_argument(
        "--mode-index",
        type=int,
        default=None,
        help="Mode-space number for --vibration (1-based).",
    )
    parser.add_argument(
        "--component-index",
        type=int,
        default=None,
        help="Component number inside the selected mode space for --vibration (1-based).",
    )
    parser.add_argument(
        "--amplitude",
        nargs="+",
        type=float,
        default=None,
        help="Amplitude(s) in Angstroms for --vector/--modulation/--vibration.",
    )
    parser.add_argument(
        "--yaml",
        # sentinel default: --subgroup has to tell "given" from "not given",
        # and --modulation resolves the documented default in its own branch
        default=None,
        help="phonopy_params.yaml(.xz) path for --modulation, and for --subgroup\n"
        "in place of --dim/-c (default: phonopy_params.yaml).",
    )
    parser.add_argument(
        "--conventional",
        action="store_true",
        help="Output the conventional cell instead of the primitive cell in --vector mode.",
    )
    parser.add_argument(
        "--keep-q-coords",
        dest="keep_q_coords",
        action="store_true",
        help="In --vector/--modulation, name output files of a non-special q with its\n"
        "coordinates (q_<coords>) instead of the ISO-IR k-vector-type label, so scans\n"
        "along one symmetry line do not overwrite each other.",
    )
    parser.add_argument(
        "--all-irreps",
        dest="all_irreps",
        action="store_true",
        help="In --irreps, additionally label the phonon irreps at the midpoints of the\n"
        "seekpath k-path segments (the symmetry lines DT, Z, SM, ...; ISO-IR labels).\n"
        "Slower than the default special-points-only survey.",
    )
    parser.add_argument(
        "--modulate",
        action="store_true",
        help="In --subgroup, also generate the distorted structure of every\n"
        "order-parameter direction (the --modulation step, run automatically;\n"
        "--amplitude sets the amplitude, default 0.3 Angstrom).",
    )
    parser.add_argument(
        "--list-qpoints",
        action="store_true",
        help="List available high-symmetry q-points in --vibration mode.",
    )
    parser.add_argument(
        "--export-npz",
        default=None,
        metavar="FILE",
        help="Optional .npz export path for --vibration mode data.",
    )
    add_output_argument(parser, "Output path (mode-dependent default).")
    parser.add_argument(
        "--tolerance",
        # --t used to reach --tolerance by abbreviation; --threshold made that
        # ambiguous, so the old spelling is kept as an explicit option string
        "--t",
        "--tol",
        type=float,
        default=None,
        help="Symmetry tolerance forwarded to the selected analysis.",
    )
    return parser


def _parse_dim(parser: ArgumentParser, tokens: list[str]) -> str:
    """Normalize --dim to the three diagonal values as one string ("4 4 4")."""
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
                "non-diagonal supercells are not supported."
            )
        diagonal = [values[0], values[4], values[8]]
    else:
        parser.error(f"--dim requires three or nine integers, got {len(values)}.")
    if any(value <= 0 for value in diagonal):
        parser.error("--dim diagonal values must be positive integers.")
    return " ".join(str(value) for value in diagonal)


def main(argv: list[str] | None = None) -> None:
    import sys

    parser = build_parser()
    if argv is None:
        argv = sys.argv[1:]
    args, unknown = parser.parse_known_args(list(argv))

    if unknown and not args.modulation:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")

    if args.band_labels and not args.band:
        parser.error("--band-labels requires --band.")

    # -c carries a sentinel default so --modulation can tell it apart from "not
    # given"; every other mode wants the documented POSCAR default
    cell = args.cell if args.cell is not None else "POSCAR"
    dim = _parse_dim(parser, args.dim) if args.dim else None

    if args.irreps:
        if not dim:
            parser.error("--irreps requires --dim.")

        dispatch_argv = ["--dim", dim, "--poscar", cell]
        if args.readfc:
            dispatch_argv.append("--readfc")
        if args.tolerance is not None:
            dispatch_argv.extend(["--tolerance", str(args.tolerance)])
        if args.all_irreps:
            dispatch_argv.append("--all-irreps")

        from ..phonon_irreps import main as phonon_irreps_main

        phonon_irreps_main(dispatch_argv)
        return

    if args.fatband:
        if not dim:
            parser.error("--fatband requires --dim.")

        dispatch_argv = ["--dim", dim, "--poscar", cell]
        if args.readfc:
            dispatch_argv.append("--readfc")
        if args.nac:
            dispatch_argv.append("--nac")
        if args.element:
            dispatch_argv.extend(["--element", args.element])
        if args.band:
            dispatch_argv.extend(["--band", args.band])
        if args.band_labels:
            dispatch_argv.extend(["--label", args.band_labels])
        if args.npoints is not None:
            dispatch_argv.extend(["--npoints", str(args.npoints)])
        if args.projection_direction:
            dispatch_argv.extend(["--projection-direction", args.projection_direction])
        if args.output:
            dispatch_argv.extend(["--output", args.output])
        if args.tolerance is not None:
            dispatch_argv.extend(["--tolerance", str(args.tolerance)])

        from ..phonon_fatband import main as phonon_fatband_main

        phonon_fatband_main(dispatch_argv)
        return

    if args.lt:
        if not dim:
            parser.error("--lt requires --dim.")

        dispatch_argv = ["--dim", dim, "--poscar", cell]
        if args.readfc:
            dispatch_argv.append("--readfc")
        if args.nac:
            dispatch_argv.append("--nac")
        if args.band:
            dispatch_argv.extend(["--band", args.band])
        if args.band_labels:
            dispatch_argv.extend(["--label", args.band_labels])
        if args.npoints is not None:
            dispatch_argv.extend(["--npoints", str(args.npoints)])
        if args.output:
            dispatch_argv.extend(["--output", args.output])
        if args.tolerance is not None:
            dispatch_argv.extend(["--tolerance", str(args.tolerance)])

        from ..phonon_lt import main as phonon_lt_main

        phonon_lt_main(dispatch_argv)
        return

    if args.vector:
        if not dim:
            parser.error("--vector requires --dim.")
        if not args.qpoint:
            parser.error("--vector requires --qpoint.")

        dispatch_argv = ["--dim", dim, "--poscar", cell]
        if args.readfc:
            dispatch_argv.append("--readfc")
        if args.conventional:
            dispatch_argv.append("--conventional")
        if args.keep_q_coords:
            dispatch_argv.append("--keep-q-coords")
        dispatch_argv.extend(["--qpoint", *[str(value) for value in args.qpoint]])
        if args.mode:
            dispatch_argv.extend(["--mode", *[str(value) for value in args.mode]])
        if args.amplitude:
            dispatch_argv.extend(["--amplitude", str(args.amplitude[0])])
        if args.output:
            dispatch_argv.extend(["--output", args.output])
        if args.tolerance is not None:
            dispatch_argv.extend(["--tolerance", str(args.tolerance)])

        from ..phonon_vector import main as phonon_vector_main

        phonon_vector_main(dispatch_argv)
        return

    if args.modulation:
        has_numbered_modulation_args = any(
            token.startswith("--qpoint") or token.startswith("--mode") or token.startswith("--amplitude")
            for token in unknown
        )
        if not args.qpoint and not has_numbered_modulation_args:
            parser.error("--modulation requires --qpoint, or numbered arguments such as --qpoint1.")

        # --yaml, or the structure file with FORCE_SETS/FORCE_CONSTANTS; with
        # neither, modulation.py falls back to phonopy_params.yaml as documented
        dispatch_argv: list[str] = []
        if args.yaml is not None:
            if args.cell is not None or dim or args.readfc:
                parser.error(
                    "--modulation takes either --yaml or -c (with FORCE_SETS/"
                    "FORCE_CONSTANTS), not both."
                )
            dispatch_argv.extend(["--yaml", args.yaml])
        else:
            if args.cell is not None:
                dispatch_argv.extend(["--poscar", args.cell])
            if dim:
                dispatch_argv.extend(["--dim", dim])
            if args.readfc:
                dispatch_argv.append("--readfc")
        if args.qpoint:
            dispatch_argv.extend(["--qpoint", *[str(value) for value in args.qpoint]])
        if args.mode:
            dispatch_argv.extend(["--mode", *[str(value) for value in args.mode]])
        if args.amplitude:
            dispatch_argv.extend(["--amplitude", *[str(value) for value in args.amplitude]])
        if args.output:
            dispatch_argv.extend(["--output", args.output])
        if args.tolerance is not None:
            dispatch_argv.extend(["--tolerance", str(args.tolerance)])
        if args.keep_q_coords:
            dispatch_argv.append("--keep-q-coords")
        dispatch_argv.extend(unknown)

        from ..modulation import main as modulation_main

        modulation_main(dispatch_argv)
        return

    if args.subgroup:
        explicit_yaml = args.yaml is not None
        if dim and explicit_yaml:
            parser.error("--subgroup takes either --dim (with -c) or --yaml, not both.")
        if not dim and not explicit_yaml:
            parser.error(
                "--subgroup requires --dim (the supercell of the force calculation) "
                "or --yaml phonopy_params.yaml."
            )
        if args.output:
            parser.error("--output is not used by --subgroup (it prints a table).")
        if args.amplitude and not args.modulate:
            parser.error("--amplitude is only used by --subgroup with --modulate.")

        dispatch_argv = []
        if dim:
            dispatch_argv.extend(["--dim", dim, "--poscar", cell])
            if args.readfc:
                dispatch_argv.append("--readfc")
        else:
            dispatch_argv.extend(["--yaml", args.yaml])
        if args.qpoint:
            dispatch_argv.extend(["--qpoint", *[str(value) for value in args.qpoint]])
        if args.threshold is not None:
            dispatch_argv.extend(["--threshold", str(args.threshold)])
        if args.tolerance is not None:
            dispatch_argv.extend(["--tolerance", str(args.tolerance)])
        if args.modulate:
            dispatch_argv.append("--modulate")
            if args.amplitude:
                dispatch_argv.extend(["--amplitude", str(args.amplitude[0])])

        from ..phonon_subgroups import main as phonon_subgroups_main

        phonon_subgroups_main(dispatch_argv)
        return

    # --vibration
    if not args.qpoint and not args.list_qpoints:
        parser.error("--vibration requires --qpoint unless --list-qpoints is used.")

    dispatch_argv = ["--poscar", cell]
    if args.qpoint:
        dispatch_argv.extend(["--qpoint", *[str(value) for value in args.qpoint]])
    if args.list_qpoints:
        dispatch_argv.append("--list-qpoints")
    if args.mode_index is not None:
        dispatch_argv.extend(["--mode-index", str(args.mode_index)])
    if args.component_index is not None:
        dispatch_argv.extend(["--component-index", str(args.component_index)])
    if args.amplitude:
        dispatch_argv.extend(["--amplitude", *[str(value) for value in args.amplitude]])
    if args.output:
        dispatch_argv.extend(["--output", args.output])
    if args.export_npz is not None:
        dispatch_argv.extend(["--export-npz", args.export_npz])
    if args.tolerance is not None:
        dispatch_argv.extend(["--tolerance", str(args.tolerance)])

    from ..vibration_modes import main as vibration_main

    vibration_main(dispatch_argv)


if __name__ == "__main__":
    main()
