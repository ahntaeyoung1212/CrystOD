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

With --diagram, the molecular-orbital diagram of a single-center molecule
(central atom + ligands, e.g. NH3, CH4, SF6) is constructed from symmetry and
overlap alone: the ligand SALCs and central-atom orbitals are combined per
irrep with the Wolfsberg-Helmholz approximation over exact single-zeta-STO
overlap integrals (symmetry-adapted extended Hueckel), and the result is
written as an interactive HTML diagram (ligand AOs | ligand SALCs | MOs |
central AOs, with correlation lines and electron filling).

With --diagram --ao-left/--ao-right (no --pyscf), the molecule is instead
split into two arbitrary submolecules by formula (e.g. H6 + C6 for benzene)
and the same extended-Hueckel machinery draws a three-column diagram
(left fragment MOs | molecule MOs | right fragment MOs): fragment levels are
the eigenstates of the fragment's own (H, S) sub-block in the one molecular
AO space, and the molecular MOs are projected onto them through the shared
overlap matrix.

With --diagram --pyscf, the diagram becomes quantitative: three PySCF SCF
calculations in one AO space (the molecule, and the two fragments with ghost
basis functions on the removed atoms, i.e. counterpoise-consistent) give the
pre-bonding fragment levels (left | molecule | right) and the exact
projection of every molecular MO onto them. --ao-left/--ao-right select any
fragment partition by formula; the default is ligand cage | central atom.
Options --basis/--theory/--xc/--charge/--spin follow script/calc_pyscf.py.

# Command Examples:
crystod-mol --symmetry --xyz XYZ_O2.xyz
crystod-mol --symmetry --xyz XYZ_NH3.xyz --tolerance 0.01
crystod-mol --xyz XYZ_NH3.xyz --element H --orbital s
crystod-mol --xyz XYZ_CH4.xyz --element H --orbital s --show-matrix
crystod-mol --xyz XYZ_NH3.xyz --element H --orbital p --visualize --bond N H 1.2
crystod-mol --diagram --xyz XYZ_NH3.xyz
crystod-mol --diagram --xyz XYZ_SF6.xyz --center S --output SF6_MO.html
crystod-mol --diagram --xyz XYZ_C6H6.xyz --ao-left H6 --ao-right C6
crystod-mol --diagram --xyz XYZ_H2O.xyz --pyscf
crystod-mol --diagram --xyz XYZ_O2.xyz --pyscf --spin 2 --ao-left O --ao-right O
crystod-mol --diagram --xyz XYZ_CH3OH.xyz --pyscf --ao-left H4 --ao-right CO
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
        "--diagram",
        action="store_true",
        help="Molecular-orbital diagram of a single-center molecule from\n"
             "symmetry + overlap (symmetry-adapted extended Hueckel); writes\n"
             "an interactive HTML diagram by default.",
    )
    parser.add_argument(
        "--center",
        default=None,
        metavar="EL",
        help="Element of the central atom for --diagram\n"
             "(default: the atom closest to the molecular center).",
    )
    parser.add_argument(
        "--pyscf",
        action="store_true",
        help="Quantitative --diagram from three PySCF SCF calculations\n"
             "(molecule + two fragments in the full molecular basis).",
    )
    parser.add_argument(
        "--basis",
        default="def2-svp",
        type=str.lower,
        metavar="BASIS",
        help="PySCF basis set for --pyscf (default: def2-svp).",
    )
    parser.add_argument(
        "--theory",
        default="scf",
        choices=["scf", "dft"],
        help="PySCF level of theory for --pyscf (default: scf = Hartree-Fock).",
    )
    parser.add_argument(
        "--xc",
        default="b3lyp",
        type=str.lower,
        metavar="XC",
        help="Exchange-correlation functional for --pyscf --theory dft\n"
             "(default: b3lyp).",
    )
    parser.add_argument(
        "--charge",
        type=int,
        default=0,
        help="Total charge of the molecule for --pyscf (default: 0).",
    )
    parser.add_argument(
        "--spin",
        type=int,
        default=None,
        metavar="2S",
        help="Molecular spin 2S for --pyscf (default: 0 or 1 by electron\n"
             "parity; e.g. --spin 2 for triplet O2).",
    )
    parser.add_argument(
        "--ao-left",
        default=None,
        metavar="FORMULA",
        help="Left-fragment formula for --diagram, e.g. H4 or O.\n"
             "Without --pyscf: two-fragment extended-Hueckel diagram\n"
             "(both --ao-left and --ao-right required).\n"
             "With --pyscf: fragment partition (default: the ligand atoms).",
    )
    parser.add_argument(
        "--ao-right",
        default=None,
        metavar="FORMULA",
        help="Right-fragment formula for --diagram, e.g. CO or O\n"
             "(with --pyscf, default: the central atom).",
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

    if args.symmetry and args.diagram:
        parser.error("--symmetry cannot be combined with --diagram.")
    if args.diagram and (args.element or args.orbital):
        parser.error("--diagram cannot be combined with --element/--orbital.")
    if args.center and not args.diagram:
        parser.error("--center is only available with --diagram.")
    if args.pyscf and not args.diagram:
        parser.error("--pyscf is only available with --diagram.")
    if not args.pyscf and (args.spin is not None or args.charge):
        parser.error("--charge/--spin require --diagram --pyscf.")
    if not args.diagram and (args.ao_left or args.ao_right):
        parser.error("--ao-left/--ao-right require --diagram.")
    if args.diagram:
        if args.show_matrix or args.align or args.visualize or args.bond:
            parser.error("--show-matrix/--align/--visualize/--bond are not "
                         "available with --diagram (the HTML diagram is "
                         "written by default).")
        dispatch_argv = ["--xyz", args.xyz, "--tolerance", str(args.tolerance)]
        if args.center:
            dispatch_argv.extend(["--center", args.center])
        if args.output:
            dispatch_argv.extend(["--output", args.output])
        if args.ao_left:
            dispatch_argv.extend(["--ao-left", args.ao_left])
        if args.ao_right:
            dispatch_argv.extend(["--ao-right", args.ao_right])
        if args.pyscf:
            dispatch_argv.extend(["--pyscf", "--basis", args.basis,
                                  "--theory", args.theory, "--xc", args.xc,
                                  "--charge", str(args.charge)])
            if args.spin is not None:
                dispatch_argv.extend(["--spin", str(args.spin)])

        from ..mo_diagram import main as mo_diagram_main

        mo_diagram_main(dispatch_argv)
        return
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
