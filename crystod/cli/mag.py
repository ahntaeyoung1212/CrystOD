"""crystod-mag: magnetism analyses.

The default (and currently only) analysis is the former ``crystod
--spin-basis`` mode: symmetry-adapted spin bases (cluster multipoles / SAMM)
for the sites of a magnetic element (backed by :mod:`crystod.spin_basis`).

Differences from the old flat command:

- per-atom spin directions and the noncollinear magnetization input are
  printed by default (no ``--show-spin-direction`` needed);
- ``--format`` selects the input format: ``vasp`` (MAGMOM line, default) or
  ``qe`` (Quantum ESPRESSO starting_magnetization/angle1/angle2).
"""

from __future__ import annotations

from argparse import ArgumentParser, RawTextHelpFormatter

from .common import CRYSTOD_CITATION, add_cell_argument, banner

desc = """\
Construct symmetry-adapted spin bases (cluster multipoles / SAMM) for the
sites of a magnetic element: the spin space is decomposed into space-group
irreps at a q-point, ferromagnetic (dipole) and antiferromagnetic (net moment
= 0) combinations are separated, and each basis vector is exported as a VESTA
file with spin arrows. Per-atom spin directions and a ready-to-paste
noncollinear magnetization input (VASP MAGMOM or Quantum ESPRESSO
starting_magnetization/angle1/angle2) are printed for every basis vector.

# Command Examples:
crystod-mag -c 221_PPOSCAR_AlNi3 --element Ni --qpoint 0 0 0
crystod-mag -c 221_PPOSCAR_AlNi3 --element Ni --qpoint 0 0 0 --format qe
crystod-mag -c 221_PPOSCAR_AlNi3 --element Ni --qpoint R
crystod-mag -c 221_PPOSCAR_AlNi3 --element Ni    # survey: all special k points
"""


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="crystod-mag",
        description=f"{banner()}\n\n{desc}",
        epilog=CRYSTOD_CITATION,
        formatter_class=RawTextHelpFormatter,
    )
    add_cell_argument(parser)
    parser.add_argument(
        "--element",
        required=True,
        help="Magnetic element whose sites carry the spins, e.g. Ni.",
    )
    parser.add_argument(
        "--qpoint",
        nargs="+",
        default=None,
        help=(
            "Either a high-symmetry label such as GM/X/M/R or three primitive\n"
            "reciprocal coordinates. When omitted, the spin-multipole irreps at\n"
            "all special k points are listed (survey mode)."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["vasp", "qe"],
        default="vasp",
        help=(
            "Format of the printed noncollinear magnetization input:\n"
            "vasp = MAGMOM line (default), qe = Quantum ESPRESSO\n"
            "starting_magnetization/angle1/angle2 (&SYSTEM)."
        ),
    )
    parser.add_argument(
        "--show-spin-direction",
        action="store_true",
        help="No effect (spin directions are always printed); kept for "
        "compatibility with the old crystod --spin-basis.",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=None,
        metavar="A",
        help="Arrow length in Angstroms given to the largest spin of each "
        "basis vector in the VESTA export (default: 1.5).",
    )
    parser.add_argument(
        "--conventional",
        action="store_true",
        help="Export the spin structures (VESTA files, spin listings) in the "
        "conventional cell instead of the primitive cell.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Symmetry tolerance (default: 1e-5).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    dispatch_argv = [
        "--poscar",
        args.cell,
        "--element",
        args.element,
        "--show-spin-direction",
        "--format",
        args.format,
    ]
    if args.qpoint:
        dispatch_argv.extend(["--qpoint", *args.qpoint])
    if args.conventional:
        dispatch_argv.append("--conventional")
    if args.amplitude is not None:
        dispatch_argv.extend(["--amplitude", str(args.amplitude)])
    if args.tolerance is not None:
        dispatch_argv.extend(["--tolerance", str(args.tolerance)])

    from ..spin_basis import main as spin_basis_main

    spin_basis_main(dispatch_argv)


if __name__ == "__main__":
    main()
