from __future__ import annotations

from argparse import ArgumentParser
from fractions import Fraction


def _parse_fractional_float(value: str) -> float:
    try:
        return float(Fraction(value))
    except Exception:
        return float(value)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="crystod",
        description="Crystal orbital / SALC, basis-function, phonon-irrep, vibration, modulation, and direct-product command line interface.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--salc",
        action="store_true",
        help="Run crystal-orbital SALC analysis.",
    )
    mode.add_argument(
        "--phonon-irrep",
        action="store_true",
        help="Run phonon irrep labeling.",
    )
    mode.add_argument(
        "--direct-product",
        action="store_true",
        help="Run direct-product decomposition for point-group irreps.",
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
        "--basis-function",
        nargs="+",
        default=None,
        help="Classify polynomial basis functions in a point group, e.g. x y z.",
    )

    parser.add_argument("--poscar", default="POSCAR", help="POSCAR path.")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Tolerance forwarded to the selected workflow.",
    )

    parser.add_argument("--element", help="Target element for elemental SALC mode.")
    parser.add_argument("--orbital", help="Target orbital for elemental SALC mode.")
    parser.add_argument(
        "--atomic-orbital",
        nargs="*",
        default=None,
        help="Atomic-orbital list for hybridization SALC mode, e.g. Ni_d O_p.",
    )
    parser.add_argument(
        "--kpoint",
        nargs=3,
        type=_parse_fractional_float,
        default=None,
        help="Optional k-point in primitive basis.",
    )
    parser.add_argument(
        "--spinor",
        action="store_true",
        help="Use double-group / spinor representations.",
    )
    parser.add_argument(
        "--show-irrep-table",
        action="store_true",
        help="Show irrep table when supported.",
    )

    parser.add_argument(
        "--dim",
        help='Supercell dimension for phonon irrep mode, e.g. "4 4 4".',
    )
    parser.add_argument(
        "--readfc",
        action="store_true",
        help="Read FORCE_CONSTANTS instead of FORCE_SETS in phonon irrep mode.",
    )
    parser.add_argument(
        "--point-group",
        help="Point-group label for direct-product mode, e.g. m-3m.",
    )
    parser.add_argument(
        "--space-group",
        help="Space-group symbol for basis-function mode, e.g. Pm-3m.",
    )
    parser.add_argument(
        "--irreps",
        nargs="+",
        default=None,
        help="Irrep labels for direct-product mode, e.g. T2g T2g T1u.",
    )
    parser.add_argument(
        "--yaml",
        default="phonopy_params.yaml",
        help="phonopy_params.yaml(.xz) path for modulation mode.",
    )
    parser.add_argument(
        "--qpoint",
        nargs="+",
        default=None,
        help="q-point for modulation or vibration mode, either as three coordinates or a high-symmetry label when supported.",
    )
    parser.add_argument(
        "--mode",
        nargs="+",
        type=int,
        default=None,
        help="Mode index or indices for modulation mode.",
    )
    parser.add_argument(
        "--amplitude",
        nargs="+",
        type=float,
        default=None,
        help="Modulation amplitude(s) in Angstroms for modulation mode.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for modulation or vibration mode.",
    )
    parser.add_argument(
        "--list-qpoints",
        action="store_true",
        help="List available high-symmetry q-points in vibration mode.",
    )
    parser.add_argument(
        "--mode-index",
        type=int,
        default=None,
        help="Mode-space index for vibration mode.",
    )
    parser.add_argument(
        "--component-index",
        type=int,
        default=None,
        help="Component index inside the selected mode space for vibration mode.",
    )
    parser.add_argument(
        "--export-npz",
        default=None,
        help="Optional .npz export path for vibration mode data.",
    )
    return parser


def _append_optional_flag(argv: list[str], enabled: bool, flag: str) -> None:
    if enabled:
        argv.append(flag)


def _append_optional_value(argv: list[str], flag: str, value) -> None:
    if value is not None:
        argv.extend([flag, str(value)])


def _append_optional_kpoint(argv: list[str], kpoint) -> None:
    if kpoint is not None:
        argv.extend(["--kpoint", *[str(value) for value in kpoint]])


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)

    if unknown and not args.modulation:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")

    if args.salc:
        if args.element and args.atomic_orbital:
            parser.error("--salc accepts either elemental mode or hybridization mode, not both.")

        if args.element:
            if not args.orbital:
                parser.error("--salc with --element requires --orbital.")
            dispatch_argv = [
                "--poscar",
                args.poscar,
                "--element",
                args.element,
                "--orbital",
                args.orbital,
            ]
            _append_optional_kpoint(dispatch_argv, args.kpoint)
            _append_optional_flag(dispatch_argv, args.spinor, "--spinor")
            _append_optional_flag(dispatch_argv, args.show_irrep_table, "--show-irrep-table")
            _append_optional_value(dispatch_argv, "--tolerance", args.tolerance)

            from .crystal_orbital_spgrep import main as crystal_orbital_main

            crystal_orbital_main(dispatch_argv)
            return

        if args.atomic_orbital:
            if args.orbital:
                parser.error("--salc hybridization mode uses --atomic-orbital instead of --orbital.")
            dispatch_argv = [
                "--poscar",
                args.poscar,
                "--orbital",
                *args.atomic_orbital,
            ]
            _append_optional_kpoint(dispatch_argv, args.kpoint)
            _append_optional_flag(dispatch_argv, args.spinor, "--spinor")
            _append_optional_flag(dispatch_argv, args.show_irrep_table, "--show-irrep-table")
            _append_optional_value(dispatch_argv, "--tolerance", args.tolerance)

            from .orbital_hybridization_spgrep import main as orbital_hybridization_main

            orbital_hybridization_main(dispatch_argv)
            return

        parser.error("--salc requires either --element/--orbital or --atomic-orbital.")

    if args.phonon_irrep:
        if not args.dim:
            parser.error("--phonon-irrep requires --dim.")
        if args.element or args.orbital or args.atomic_orbital or args.kpoint or args.spinor or args.show_irrep_table:
            parser.error("--phonon-irrep does not use SALC-specific options.")

        dispatch_argv = [
            "--dim",
            args.dim,
            "--poscar",
            args.poscar,
        ]
        _append_optional_flag(dispatch_argv, args.readfc, "--readfc")
        _append_optional_value(dispatch_argv, "--tolerance", args.tolerance)

        from .phonon_irreps import main as phonon_irreps_main

        phonon_irreps_main(dispatch_argv)
        return

    if args.direct_product:
        if not args.point_group:
            parser.error("--direct-product requires --point-group.")
        if (
            args.poscar != "POSCAR"
            or args.tolerance is not None
            or args.element
            or args.orbital
            or args.atomic_orbital
            or args.kpoint
            or args.spinor
            or args.dim
            or args.readfc
        ):
            parser.error("--direct-product does not use SALC- or phonon-specific options.")
        if not args.irreps and not args.show_irrep_table:
            parser.error("--direct-product requires --irreps and/or --show-irrep-table.")

        dispatch_argv = ["--point-group", args.point_group]
        if args.irreps:
            dispatch_argv.extend(["--irreps", *args.irreps])
        _append_optional_flag(dispatch_argv, args.show_irrep_table, "--show-irrep-table")

        from .direct_product import main as direct_product_main

        direct_product_main(dispatch_argv)
        return

    if args.basis_function is not None:
        if bool(args.point_group) == bool(args.space_group):
            parser.error("--basis-function requires exactly one of --point-group or --space-group.")
        if (
            args.poscar != "POSCAR"
            or args.tolerance is not None
            or args.element
            or args.orbital
            or args.atomic_orbital
            or (args.kpoint is not None and args.space_group is None)
            or args.spinor
            or args.dim
            or args.readfc
            or args.irreps
            or args.yaml != "phonopy_params.yaml"
            or args.qpoint is not None
            or args.mode is not None
            or args.amplitude is not None
            or args.output is not None
            or args.list_qpoints
            or args.mode_index is not None
            or args.component_index is not None
            or args.export_npz is not None
        ):
            parser.error("--basis-function uses --point-group or --space-group, and --kpoint only with --space-group.")

        dispatch_argv = [
            "--basis-function",
            *args.basis_function,
        ]
        if args.point_group:
            dispatch_argv.extend(["--point-group", args.point_group])
        if args.space_group:
            dispatch_argv.extend(["--space-group", args.space_group])
            if args.kpoint is None:
                parser.error("--basis-function with --space-group requires --kpoint.")
            dispatch_argv.extend(["--kpoint", *[str(value) for value in args.kpoint]])
        _append_optional_flag(dispatch_argv, args.show_irrep_table, "--show-irrep-table")

        from .basis_function import main as basis_function_main

        basis_function_main(dispatch_argv)
        return

    if args.modulation:
        if (
            args.poscar != "POSCAR"
            or args.element
            or args.orbital
            or args.atomic_orbital
            or args.kpoint
            or args.spinor
            or args.show_irrep_table
            or args.dim
            or args.readfc
            or args.point_group
            or args.irreps
        ):
            parser.error("--modulation does not use SALC-, phonon-irrep-, or direct-product-specific options.")
        has_numbered_modulation_args = any(
            token.startswith("--qpoint") or token.startswith("--mode") or token.startswith("--amplitude")
            for token in unknown
        )
        if not args.qpoint and not has_numbered_modulation_args:
            parser.error("--modulation requires --qpoint, or numbered arguments such as --qpoint1.")
        if not args.mode and not has_numbered_modulation_args:
            parser.error("--modulation requires --mode, or numbered arguments such as --mode1.")

        dispatch_argv = [
            "--yaml",
            args.yaml,
        ]
        if args.qpoint:
            dispatch_argv.extend(["--qpoint", *[str(value) for value in args.qpoint]])
        if args.mode:
            dispatch_argv.extend(["--mode", *[str(value) for value in args.mode]])
        if args.amplitude:
            dispatch_argv.extend(["--amplitude", *[str(value) for value in args.amplitude]])
        if args.output is not None:
            dispatch_argv.extend(["--output", args.output])
        if args.tolerance is not None:
            dispatch_argv.extend(["--tolerance", str(args.tolerance)])
        dispatch_argv.extend(unknown)

        from .modulation import main as modulation_main

        modulation_main(dispatch_argv)
        return

    if args.vibration:
        if (
            args.element
            or args.orbital
            or args.atomic_orbital
            or args.kpoint
            or args.spinor
            or args.show_irrep_table
            or args.dim
            or args.readfc
            or args.point_group
            or args.irreps
            or args.yaml != "phonopy_params.yaml"
            or args.mode is not None
        ):
            parser.error("--vibration does not use SALC-, phonon-irrep-, modulation-, or direct-product-specific options.")
        if not args.qpoint and not args.list_qpoints:
            parser.error("--vibration requires --qpoint unless --list-qpoints is used.")

        dispatch_argv = [
            "--poscar",
            args.poscar,
        ]
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
        if args.output is not None:
            dispatch_argv.extend(["--output", args.output])
        if args.export_npz is not None:
            dispatch_argv.extend(["--export-npz", args.export_npz])
        if args.tolerance is not None:
            dispatch_argv.extend(["--tolerance", str(args.tolerance)])

        from .vibration_modes import main as vibration_main

        vibration_main(dispatch_argv)
        return

    parser.error("Please specify either --salc, --phonon-irrep, --direct-product, --modulation, --vibration, or --basis-function.")
