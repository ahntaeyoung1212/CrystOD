"""crystod-mol: molecular symmetry and molecular SALCs.

Molecular counterpart of the crystalline analyses:

- ``--symmetry``            -- detect the molecular point group (pymatgen),
  the molecular analogue of ``phonopy --symmetry`` for crystals;
- ``--element EL --orbital ORB`` -- molecular SALCs: characters of the
  site-permutation representation are multiplied by the orbital characters
  and decomposed into point-group irreps, and the explicit SALCs are
  projected out (backed by :mod:`crystod.molecular_salc`).
"""

from __future__ import annotations

from argparse import ArgumentParser, RawTextHelpFormatter

from .common import banner

desc = """\
Analyze the point-group symmetry of a molecule (XYZ file) and construct its
molecular SALCs (symmetry-adapted linear combinations of atomic orbitals).

With --symmetry, the molecular point group is detected with pymatgen
(Schoenflies and Hermann-Mauguin symbols, symmetry operations by class) --
the molecular analogue of `phonopy --symmetry` for crystals.

With --element/--orbital, the site-permutation representation of the selected
element's sites is built, its characters are multiplied by the characters of
the atomic orbital (s/p/d/f), the product is decomposed into the irreps of the
molecular point group, and the explicit SALCs are printed. The irrep labels
come from the same point-group character tables as crystod-group.

# Command Examples:
crystod-mol --symmetry --xyz XYZ_O2.xyz
crystod-mol --symmetry --xyz XYZ_NH3.xyz --tolerance 0.01
crystod-mol --xyz XYZ_NH3.xyz --element H --orbital s
crystod-mol --xyz XYZ_CH4.xyz --element H --orbital s --show-matrix
crystod-mol --xyz XYZ_NH3.xyz --element H --orbital p --visualize --bond N H 1.2
"""


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="crystod-mol",
        description=f"{banner()}\n\n{desc}",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--xyz",
        required=True,
        metavar="FILE",
        help="Molecule file in XYZ format.",
    )
    parser.add_argument(
        "--symmetry",
        action="store_true",
        help="Only detect and print the molecular point group.",
    )
    parser.add_argument(
        "--element",
        default=None,
        help="Element whose sites carry the orbitals (SALC mode), e.g. H.",
    )
    parser.add_argument(
        "--orbital",
        default=None,
        help="Orbital shell for the SALC mode: s, p, d, or f.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.3,
        metavar="TOL",
        help="Distance tolerance (Angstrom) for the symmetry detection\n"
             "(default: 0.3, as in pymatgen).",
    )
    parser.add_argument(
        "--show-matrix",
        action="store_true",
        help="Print the site-permutation matrix of every symmetry operation\n"
             "(SALC mode).",
    )
    parser.add_argument(
        "--align",
        action="store_true",
        help="Rotate the molecule into the standard point-group orientation\n"
             "(principal axis along z) before the SALC analysis, so the SALC\n"
             "coefficients refer to the textbook axis convention.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Write the SALCs as an interactive 3D HTML viewer (SALC mode) --\n"
             "the same standalone viewer as crystod --visualize.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        metavar="FILE",
        help="Output HTML path for --visualize\n"
             "(default: SALC_{molecule}_{element}_{orbital}.html).",
    )
    parser.add_argument(
        "--bond",
        action="append",
        nargs=3,
        default=None,
        metavar=("EL1", "EL2", "MAX"),
        help="Draw bonds between EL1 and EL2 atoms up to MAX Angstroms in the\n"
             "viewer (repeatable), e.g. --bond N H 1.2.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.symmetry and (args.element or args.orbital):
        parser.error("--symmetry cannot be combined with --element/--orbital.")
    if not args.symmetry and (args.element is None or args.orbital is None):
        parser.error("either use --symmetry, or give both --element and --orbital.")
    if not args.visualize and (args.output or args.bond):
        parser.error("--output/--bond are only available with --visualize.")
    if args.symmetry and args.visualize:
        parser.error("--visualize is only available in SALC mode (--element/--orbital).")

    dispatch_argv = ["--xyz", args.xyz, "--tolerance", str(args.tolerance)]
    if args.symmetry:
        dispatch_argv.append("--symmetry")
    else:
        dispatch_argv.extend(["--element", args.element, "--orbital", args.orbital])
        if args.show_matrix:
            dispatch_argv.append("--show-matrix")
        if args.align:
            dispatch_argv.append("--align")
        if args.visualize:
            dispatch_argv.append("--visualize")
        if args.output:
            dispatch_argv.extend(["--output", args.output])
        for bond in args.bond or []:
            dispatch_argv.extend(["--bond", *bond])

    from ..molecular_salc import main as molecular_salc_main

    molecular_salc_main(dispatch_argv)


if __name__ == "__main__":
    main()
