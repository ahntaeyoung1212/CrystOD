"""crystod: the main command — crystal-orbital SALC analysis.

The flagship analysis needs no mode flag: give a structure and an orbital
selection and the SALC irrep decomposition runs directly.

- ``--element EL --orbital ORB``  -- elemental SALC (old ``--salc``);
- ``--atomic-orbital A_x B_y``    -- hybridization SALC
  (old ``--salc --atomic-orbital``);
- ``--visualize``                 -- SALC coefficients + interactive 3D HTML
  (old ``--visualize-basis``);
- ``--star-of-k``                 -- symmetry information: star of a k point
  (old flat spelling unchanged).

Old flat modes (``--salc``, ``--phonon-irrep``, ``--bz``, ...) are detected
and routed to :mod:`crystod.cli.legacy`, which keeps them working through
v0.3.x with deprecation notices.
"""

from __future__ import annotations

from argparse import ArgumentParser, RawTextHelpFormatter
from fractions import Fraction

from .common import add_cell_argument, add_output_argument

# Flat mode flags of the pre-v0.3.0 interface, removed in v0.3.0: each maps
# to the guidance shown in the error message. --star-of-k is absent on
# purpose: it kept its spelling and is handled by the new parser directly.
REMOVED_MODE_FLAGS = {
    "--salc": "crystod -c POSCAR --element EL --orbital ORB (or --atomic-orbital EL_ORB ...)",
    "--visualize-basis": "crystod --visualize -c POSCAR --element EL --orbital ORB --kpoint ...",
    "--phonon-irrep": "crystod-phonon --irreps",
    "--phonon-fatband": "crystod-phonon --fatband",
    "--phonon-lt": "crystod-phonon --lt",
    "--phonon-vector": "crystod-phonon --vector",
    "--modulation": "crystod-phonon --modulation",
    "--vibration": "crystod-phonon --vibration",
    "--spin-basis": "crystod-mag",
    "--ligand-field-split": "crystod-group --ligand-field ORBITAL",
    "--decompose-irrep": "crystod-group --decompose",
    "--direct-product": "crystod-group --product (character table: --table)",
    "--basis-function": "crystod-group --basis",
    "--generate-basis-function": "crystod-group --generate-basis",
    "--show-coset": "crystod-group --coset",
    "--bz": "crystod-bz",
    "--bz-supercell": "crystod-bz --trans-mat ...",
    "--xdatcar2adp": "crystod-md --adp",
}

desc = """\
Crystal-orbital SALC analysis (the CrystOD main command). No mode flag is
needed: select the orbitals and the irrep decomposition runs directly.

# Command Examples:
crystod -c 221_PPOSCAR_SrTiO3 --element Ti --orbital d
crystod -c 221_PPOSCAR_SrTiO3 --element Ti --orbital d --kpoint 0 0 0 --spinor --show-irrep-table
crystod -c 221_PPOSCAR_SrTiO3 --atomic-orbital Ti_d O_p --kpoint 0 0 0
crystod -c 221_PPOSCAR_ScF3 --element F --orbital p --kpoint 0 0 0 --visualize
crystod --star-of-k -c 221_PPOSCAR_ScF3 --kpoint 0.5 0.5 0
"""

epilog = """\
Sectioned commands (see crystod-<section> --help):
  crystod-phonon   phonon analyses (--irreps/--fatband/--lt/--vector/--modulation/--vibration)
  crystod-group    point/space-group calculator (--product/--table/--decompose/
                   --ligand-field/--basis/--generate-basis/--coset)
  crystod-mag      symmetry-adapted spin bases (MAGMOM / QE noncollinear input)
  crystod-md       MD-trajectory analyses (--adp/--summary)
  crystod-bz       Brillouin-zone plots (unit cell, or + supercell via --trans-mat)

The pre-v0.3.0 flat modes (--salc, --phonon-irrep, --bz, ...) still work and
print their sectioned replacement; they will be removed in v1.0.
"""


def _parse_fractional_float(value: str) -> float:
    try:
        return float(Fraction(value))
    except Exception:
        return float(value)


def build_parser() -> ArgumentParser:
    from .. import __version__

    parser = ArgumentParser(
        prog="crystod",
        description=desc,
        epilog=epilog,
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"CrystOD {__version__}"
    )
    add_cell_argument(parser)
    parser.add_argument(
        "--element",
        default=None,
        help="Target element for elemental SALC analysis, e.g. Ti.",
    )
    parser.add_argument(
        "--orbital",
        default=None,
        help="Target orbital for elemental SALC analysis, e.g. d.",
    )
    parser.add_argument(
        "--atomic-orbital",
        nargs="+",
        default=None,
        metavar="EL_ORB",
        help="Atomic-orbital list for hybridization analysis, e.g. Ti_d O_p "
        "(alternative to --element/--orbital).",
    )
    parser.add_argument(
        "--kpoint",
        nargs="+",
        default=None,
        help="k-point: three primitive reciprocal coordinates (fractions such as 1/2\n"
        "are allowed), or a high-symmetry label such as GM/X/M/R in\n"
        "--star-of-k/--visualize mode. When omitted in SALC mode, all special\n"
        "k points are analyzed.",
    )
    parser.add_argument(
        "--spinor",
        action="store_true",
        help="Use double-group / spinor representations.",
    )
    parser.add_argument(
        "--show-irrep-table",
        action="store_true",
        help="Show the little-group irrep table at the selected k point.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Construct and visualize the SALC basis functions as an interactive "
        "3D HTML file (requires --element/--orbital/--kpoint).",
    )
    parser.add_argument(
        "--star-of-k",
        action="store_true",
        help="Display the star of a k point for the space group of the structure "
        "(requires --kpoint).",
    )
    parser.add_argument(
        "--mode-index",
        type=int,
        default=None,
        help="Mode-space number for --visualize (1-based).",
    )
    parser.add_argument(
        "--bond",
        nargs=3,
        action="append",
        default=None,
        metavar=("EL1", "EL2", "MAX"),
        help="Draw bonds between EL1 and EL2 atoms up to MAX Angstroms, plus the\n"
        "VESTA-style coordination polyhedra around the EL1 atoms, in --visualize\n"
        "mode (repeatable), e.g. --bond Sc F 2.3.",
    )
    parser.add_argument(
        "--conventional",
        action="store_true",
        help="Display the SALC in the conventional cell instead of the primitive\n"
        "cell (--visualize mode).",
    )
    parser.add_argument(
        "--real-coefficient",
        action="store_true",
        help="Re-combine degenerate SALC components into real-coefficient form "
        "(--visualize mode).",
    )
    add_output_argument(parser, "Output HTML path for --visualize.")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Symmetry tolerance forwarded to the selected analysis.",
    )
    return parser


def _normalize_kpoint(parser: ArgumentParser, tokens: list[str], allow_label: bool) -> list[str]:
    """Return the k-point as float strings, or as a label where supported."""
    if len(tokens) == 3:
        try:
            return [str(_parse_fractional_float(token)) for token in tokens]
        except (ValueError, ZeroDivisionError):
            pass
    if allow_label and len(tokens) == 1:
        return list(tokens)
    message = "--kpoint requires three coordinates such as 0 0 0 (fractions allowed)"
    if allow_label:
        message += " or a high-symmetry label such as GM/X/M/R"
    parser.error(message + ".")


def main(argv: list[str] | None = None) -> None:
    import sys

    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)

    # Pre-v0.3.0 flat invocation -> clear removal error with the replacement.
    for token in argv:
        flag = token.split("=", 1)[0]
        if flag in REMOVED_MODE_FLAGS:
            raise SystemExit(
                f"ERROR: '{flag}' was removed in v0.3.0. Use the sectioned command instead:\n"
                f"  {REMOVED_MODE_FLAGS[flag]}\n"
                "See the README (Command Summary) for the full new interface."
            )

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.star_of_k:
        if args.visualize:
            parser.error("--star-of-k and --visualize cannot be combined.")
        if args.kpoint is None:
            parser.error("--star-of-k requires --kpoint.")

        kpoint = _normalize_kpoint(parser, args.kpoint, allow_label=True)
        dispatch_argv = ["--poscar", args.cell, "--kpoint", *kpoint]
        if args.tolerance is not None:
            dispatch_argv.extend(["--tolerance", str(args.tolerance)])

        from ..star_of_k import main as star_of_k_main

        star_of_k_main(dispatch_argv)
        return

    if args.bond and not args.visualize:
        parser.error("--bond is only available with --visualize.")
    if args.conventional and not args.visualize:
        parser.error("--conventional is only available with --visualize.")

    if args.visualize:
        if not args.element or not args.orbital:
            parser.error("--visualize requires --element and --orbital.")
        if args.kpoint is None:
            parser.error("--visualize requires --kpoint.")

        kpoint = _normalize_kpoint(parser, args.kpoint, allow_label=True)
        dispatch_argv = [
            "--poscar",
            args.cell,
            "--element",
            args.element,
            "--orbital",
            args.orbital,
            "--kpoint",
            *kpoint,
        ]
        if args.tolerance is not None:
            dispatch_argv.extend(["--tolerance", str(args.tolerance)])
        if args.mode_index is not None:
            dispatch_argv.extend(["--mode-index", str(args.mode_index)])
        if args.output:
            dispatch_argv.extend(["--output", args.output])
        if args.real_coefficient:
            dispatch_argv.append("--real-coefficient")
        for el1, el2, max_length in args.bond or []:
            dispatch_argv.extend(["--bond", el1, el2, max_length])
        if args.conventional:
            dispatch_argv.append("--conventional")

        from ..visualize_basis import main as visualize_basis_main

        visualize_basis_main(dispatch_argv)
        return

    if args.atomic_orbital:
        if args.element or args.orbital:
            parser.error(
                "hybridization analysis uses --atomic-orbital alone "
                "(not combined with --element/--orbital)."
            )

        dispatch_argv = ["--poscar", args.cell, "--orbital", *args.atomic_orbital]
        if args.kpoint is not None:
            dispatch_argv.extend(
                ["--kpoint", *_normalize_kpoint(parser, args.kpoint, allow_label=False)]
            )
        if args.spinor:
            dispatch_argv.append("--spinor")
        if args.show_irrep_table:
            dispatch_argv.append("--show-irrep-table")
        if args.tolerance is not None:
            dispatch_argv.extend(["--tolerance", str(args.tolerance)])

        from ..orbital_hybridization_spgrep import main as orbital_hybridization_main

        orbital_hybridization_main(dispatch_argv)
        return

    if args.element or args.orbital:
        if not args.element or not args.orbital:
            parser.error("elemental SALC analysis requires both --element and --orbital.")

        dispatch_argv = [
            "--poscar",
            args.cell,
            "--element",
            args.element,
            "--orbital",
            args.orbital,
        ]
        if args.kpoint is not None:
            dispatch_argv.extend(
                ["--kpoint", *_normalize_kpoint(parser, args.kpoint, allow_label=False)]
            )
        if args.spinor:
            dispatch_argv.append("--spinor")
        if args.show_irrep_table:
            dispatch_argv.append("--show-irrep-table")
        if args.tolerance is not None:
            dispatch_argv.extend(["--tolerance", str(args.tolerance)])

        from ..crystal_orbital_spgrep import main as crystal_orbital_main

        crystal_orbital_main(dispatch_argv)
        return

    parser.error(
        "nothing to do: give --element ELEMENT --orbital ORBITAL (SALC), "
        "--atomic-orbital EL_ORB ... (hybridization), --visualize, or --star-of-k.\n"
        "Domain analyses live in the sectioned commands: crystod-phonon, "
        "crystod-group, crystod-mag, crystod-md, crystod-bz."
    )


if __name__ == "__main__":
    main()
