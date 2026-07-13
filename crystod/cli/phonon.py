"""crystod-phonon: phonon analyses.

Merges the former phonon-related flat modes into one sectioned command:

- ``--irreps``     -- phonon irrep labeling (old ``--phonon-irrep``);
- ``--fatband``    -- element-projected phonon fatbands (old ``--phonon-fatband``);
- ``--lt``         -- longitudinal/transverse-resolved bands (old ``--phonon-lt``);
- ``--vector``     -- eigenvector VESTA export (old ``--phonon-vector``);
- ``--modulation`` -- modulated structures (old ``--modulation``);
- ``--vibration``  -- symmetry-only vibration bases, no force data
  (old ``--vibration``).
"""

from __future__ import annotations

from argparse import ArgumentParser, RawTextHelpFormatter

from .common import add_cell_argument, add_output_argument, banner

desc = """\
Phonon analyses from phonopy force data (FORCE_SETS, or FORCE_CONSTANTS with
--readfc, or phonopy_params.yaml for --modulation) — except --vibration,
which needs only the crystal structure.

# Command Examples:
crystod-phonon --irreps --dim "4 4 4" -c 221_PPOSCAR_SrTiO3 --readfc
crystod-phonon --fatband --dim "4 4 4" -c 221_PPOSCAR_ScF3 --nac
crystod-phonon --lt --dim "4 4 4" -c 221_PPOSCAR_ScF3
crystod-phonon --vector --dim "4 4 4" -c 227_PPOSCAR_Si --readfc --qpoint GM
crystod-phonon --modulation --yaml phonopy_params.yaml --qpoint 0.5 0.5 0.5   (list modes and star of q only)
crystod-phonon --modulation --yaml phonopy_params.yaml --qpoint 0.5 0.5 0.5 --mode 1 2 3 --amplitude 0.3
crystod-phonon --vibration -c 221_PPOSCAR_ScF3 --qpoint R
"""


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="crystod-phonon",
        description=f"{banner()}\n\n{desc}",
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

    add_cell_argument(parser)
    parser.add_argument(
        "--dim",
        nargs="+",
        default=None,
        metavar="N",
        help=(
            "Supercell dimension of the force calculation: three diagonal values\n"
            "or a nine-value diagonal matrix, quoted or unquoted, e.g.\n"
            '--dim 4 4 4, --dim="4 4 4", --dim 4 0 0 0 4 0 0 0 4.\n'
            "Required for --irreps/--fatband/--lt/--vector."
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
        help="q-point for --vector/--modulation/--vibration: three coordinates or a "
        "high-symmetry label when supported.",
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
        default="phonopy_params.yaml",
        help="phonopy_params.yaml(.xz) path for --modulation (default: phonopy_params.yaml).",
    )
    parser.add_argument(
        "--conventional",
        action="store_true",
        help="Output the conventional cell instead of the primitive cell in --vector mode.",
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

    dim = _parse_dim(parser, args.dim) if args.dim else None

    if args.irreps:
        if not dim:
            parser.error("--irreps requires --dim.")

        dispatch_argv = ["--dim", dim, "--poscar", args.cell]
        if args.readfc:
            dispatch_argv.append("--readfc")
        if args.tolerance is not None:
            dispatch_argv.extend(["--tolerance", str(args.tolerance)])

        from ..phonon_irreps import main as phonon_irreps_main

        phonon_irreps_main(dispatch_argv)
        return

    if args.fatband:
        if not dim:
            parser.error("--fatband requires --dim.")

        dispatch_argv = ["--dim", dim, "--poscar", args.cell]
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

        dispatch_argv = ["--dim", dim, "--poscar", args.cell]
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

        dispatch_argv = ["--dim", dim, "--poscar", args.cell]
        if args.readfc:
            dispatch_argv.append("--readfc")
        if args.conventional:
            dispatch_argv.append("--conventional")
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

        dispatch_argv = ["--yaml", args.yaml]
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
        dispatch_argv.extend(unknown)

        from ..modulation import main as modulation_main

        modulation_main(dispatch_argv)
        return

    # --vibration
    if not args.qpoint and not args.list_qpoints:
        parser.error("--vibration requires --qpoint unless --list-qpoints is used.")

    dispatch_argv = ["--poscar", args.cell]
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
