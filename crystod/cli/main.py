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

from .common import (
    add_cell_argument,
    add_output_argument,
    banner,
    print_crystod_citation,
)

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
crystod --diagram -c 221_PPOSCAR_SrTiO3 --co-left SrTi --co-right O3
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

The pre-v0.3.0 flat modes (--salc, --phonon-irrep, --bz, ...) were removed in
v0.3.0; invoking one prints the equivalent sectioned command.

If you use CrystOD in your research, please cite:
  H. Koiso and Y. Mochizuki et al., Phys. Rev. B 110, 064104 (2024). https://doi.org/10.1103/PhysRevB.110.064104
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
        description=f"{banner()}\n\n{desc}",
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
        "or Ti-d O-p (either separator; alternative to --element/--orbital).",
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
        "--diagram",
        action="store_true",
        help="Crystal-orbital diagram: per-k-point energy diagrams of the two\n"
        "fragment sublattices given by --co-left/--co-right (full valence\n"
        "basis, e.g. --co-left SrTi --co-right O3) from symmetry +\n"
        "extended-Hueckel Bloch overlaps, written as interactive HTML\n"
        "(CrystOD_{cell}.html). Every level carries a hover wave-function\n"
        "sketch of all its atomic-orbital components.",
    )
    parser.add_argument(
        "--co-left",
        nargs="+",
        default=None,
        metavar="FORMULA",
        help="Left fragment sublattice of the --diagram (formula of the\n"
        "elements, e.g. SrTi); all valence shells of these atoms are used.",
    )
    parser.add_argument(
        "--co-right",
        nargs="+",
        default=None,
        metavar="FORMULA",
        help="Right fragment sublattice of the --diagram (e.g. O3).",
    )
    parser.add_argument(
        "--electrons",
        type=float,
        default=None,
        help="Electrons per primitive cell in the --diagram\n"
        "(default: all electrons of the neutral atoms).",
    )
    parser.add_argument(
        "--oxidation",
        nargs="+",
        default=None,
        metavar="EL=Q",
        help="Formal oxidation states of the --diagram fragments: they set\n"
        "the removed-sublattice point charges AND the fragment-column\n"
        "electron counts (with --onsite only the counts remain in use),\n"
        "e.g. Sr=+2 Ti=+4 O=-2 (default: guessed with pymatgen).",
    )
    parser.add_argument(
        "--pyscf",
        action="store_true",
        help="Make the --diagram quantitative with three periodic PySCF\n"
        "calculations (left sublattice, right sublattice, crystal) that share\n"
        "one AO space: the removed sublattice stays as ghost basis functions\n"
        "and acts through its formal-charge point lattice, so every fragment\n"
        "level is a real pre-bonding electronic state.",
    )
    parser.add_argument(
        "--basis",
        default=None,
        help="PySCF Gaussian basis for --diagram --pyscf\n"
        "(default: gth-dzvp-molopt-sr).",
    )
    parser.add_argument(
        "--pseudo",
        default=None,
        help="PySCF GTH pseudopotential for --diagram --pyscf\n"
        "(default: gth-pbe).",
    )
    parser.add_argument(
        "--xc",
        default=None,
        help="Exchange-correlation functional for --diagram --pyscf, or 'hf'\n"
        "(default: pbe).",
    )
    parser.add_argument(
        "--kmesh",
        type=int,
        nargs=3,
        default=None,
        metavar=("N1", "N2", "N3"),
        help="Regular k-mesh of the --diagram --pyscf self-consistent step\n"
        "(default: round(8 Angstrom / |a_i|), i.e. 2 2 2 for a ~4 Angstrom cell).",
    )
    parser.add_argument(
        "--ke-cutoff",
        type=float,
        default=None,
        help="FFT-grid cutoff in Hartree for --diagram --pyscf (default: 200).",
    )
    parser.add_argument(
        "--no-symmetrize",
        action="store_true",
        help="For --diagram --pyscf: keep the raw SCF eigenvalues instead of\n"
        "re-diagonalizing the group-averaged Fock (debug; shows the grid-broken\n"
        "degeneracies).",
    )
    parser.add_argument(
        "--max-l",
        type=int,
        default=None,
        help="For --diagram --pyscf: drop basis shells with l above this from\n"
        "every element (e.g. 2 removes the f polarization functions).",
    )
    parser.add_argument(
        "--no-ghost",
        action="store_true",
        help="For --diagram --pyscf: exclude the removed sublattice's basis\n"
        "functions from the fragment calculations (hard constraint -- no\n"
        "fragment wave function on the removed atoms; default keeps them as\n"
        "counterpoise ghosts and reports the ghost weight per level).",
    )
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="For --diagram --pyscf: keep each calculation's own G=0 reference\n"
        "instead of the deep-level (XPS-style) column alignment.",
    )
    parser.add_argument(
        "--projection",
        choices=("lowdin", "mulliken"),
        default=None,
        help="For --diagram --pyscf: population measure for the sketch lobe\n"
        "sizes and the per-(element, shell) rows -- Loewdin |S^(1/2)c|^2\n"
        "(default; non-negative, sums to 100%%) or Mulliken Re[c*(Sc)].",
    )
    parser.add_argument(
        "--chk",
        default=None,
        metavar="FILE",
        help="For --diagram --pyscf: WAVECAR-style restart file -- written\n"
        "after the SCFs if missing, read (skipping all three SCFs) if present;\n"
        "the defining parameters are verified before reuse.",
    )
    parser.add_argument(
        "--chk-info",
        default=None,
        metavar="FILE",
        help="Print the calculation conditions stored in a --chk checkpoint\n"
        "(binary npz): method, k-mesh, fragments, stored densities, and a\n"
        "ready-to-paste option string that reproduces them.",
    )
    parser.add_argument(
        "--onsite",
        action="store_true",
        help="For --diagram --pyscf: single-Hamiltonian mode -- only the\n"
        "crystal SCF runs, and the fragment columns are the per-shell\n"
        "on-site multiplets <phi|F|phi> of its Fock (tight-binding on-site\n"
        "energies, one level per induced irrep; no point charges, no\n"
        "reference alignment). A full-run --chk is reused.",
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
        help="Display in the conventional cell instead of the primitive cell:\n"
        "the SALC in --visualize mode, the hover wave-function sketches in\n"
        "--diagram mode.",
    )
    parser.add_argument(
        "--real-coefficient",
        action="store_true",
        help="Re-combine degenerate SALC components into real-coefficient form "
        "(--visualize mode).",
    )
    parser.add_argument(
        "--dos",
        action="store_true",
        help="With --pyscf: DOS/PDOS and partial charges from the (restarted)\n"
        "density matrix, diagonalized non-self-consistently on --dos-kmesh\n"
        "(VASP-style two-step; reuses a --chk file, so no new SCF is needed).",
    )
    parser.add_argument(
        "--dos-kmesh",
        type=int,
        nargs=3,
        default=None,
        metavar=("N1", "N2", "N3"),
        help="Dense k mesh of the --dos band step (default 8 8 8).",
    )
    parser.add_argument(
        "--band",
        action="store_true",
        help="With --pyscf: electronic band structure along the automatic\n"
        "seekpath high-symmetry path, diagonalized non-self-consistently\n"
        "from the (restarted) density matrix (VASP-style two-step; reuses\n"
        "a --chk file, so no new SCF is needed).",
    )
    parser.add_argument(
        "--fatband",
        action="store_true",
        help="With --band: element- and (element, l)-projected fatbands\n"
        "(one overview page in VESTA colors + one page per element with\n"
        "the s/p/d/f breakdown; --projection picks the measure).",
    )
    parser.add_argument(
        "--band-points",
        type=int,
        default=None,
        metavar="N",
        help="k points per path leg of --band (default 41).",
    )
    parser.add_argument(
        "--sublattice",
        nargs="+",
        default=None,
        metavar="FORMULA",
        help="For --visualize --pyscf: show the PySCF levels of this fragment\n"
        "sublattice (formal-charge ions + ghost basis + point-charge lattice,\n"
        "as in --diagram --pyscf), e.g. --sublattice Sc or --sublattice F3;\n"
        "omit for the full crystal's levels.",
    )
    parser.add_argument(
        "--window",
        nargs=2,
        type=float,
        default=None,
        metavar=("LO", "HI"),
        help="For --visualize --pyscf and --band: energy window in eV\n"
        "(visualize default: HOMO-15 .. LUMO+10 of the displayed column;\n"
        "band default: the band range clipped to -25 .. 15).",
    )
    parser.add_argument(
        "--align",
        choices=("vbm", "absolute"),
        default=None,
        help="For --band/--dos (--pyscf): energy reference of the plots --\n"
        "the valence-band maximum (default) or the raw absolute scale.",
    )
    parser.add_argument(
        "--diagonalize",
        action="store_true",
        help="For --visualize --pyscf: canonicalize degenerate partners\n"
        "(RREF) so the drawn s/p/d/f combinations are axis-aligned instead\n"
        "of the SCF's arbitrary unitary mixture; energies are unchanged.",
    )
    parser.add_argument(
        "--valence-only",
        action="store_true",
        help="For --visualize --pyscf: drop semicore shells (occupied\n"
        "fragment bands > 12 eV below the crystal VBM, e.g. Sc 3s/3p and\n"
        "F 2s of ScF3) from the drawn wave functions -- their admixture in\n"
        "a valence level is the on-site orthogonality tail whose radial\n"
        "node makes a bonding sigma level look antibonding; levels the\n"
        "shell dominates (the semicore bands themselves) keep it.",
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

    if args.chk_info:
        from ..crystal_orbital_pyscf import describe_chk

        describe_chk(args.chk_info)
        return

    # these guards must run before ANY mode branch returns
    if args.fatband and not args.band:
        parser.error("--fatband is only used with --band.")
    if args.band_points is not None and not args.band:
        parser.error("--band-points is only used with --band.")
    if args.align is not None and not (args.band or args.dos):
        parser.error("--align is only used with --band or --dos (--pyscf).")
    if args.bond and not args.visualize:
        parser.error("--bond is only available with --visualize.")
    if args.conventional and not (args.visualize or args.diagram):
        parser.error("--conventional is only available with --visualize "
                     "or --diagram.")

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

    if args.band:
        if not args.pyscf:
            parser.error("--band requires --pyscf (it reads the PySCF "
                         "density matrix).")
        for flag, value in (("--dos", args.dos),
                            ("--diagram", args.diagram),
                            ("--visualize", args.visualize)):
            if value:
                parser.error(f"--band and {flag} are separate runs; call "
                             "them one at a time (they can share the same "
                             "--chk).")
        if args.dos_kmesh is not None:
            parser.error("--dos-kmesh is only used with --dos (the band "
                         "step runs on the seekpath line, --band-points "
                         "per leg).")
        for flag, value in (("--sublattice", args.sublattice),
                            ("--no-align", args.no_align),
                            ("--no-symmetrize", args.no_symmetrize),
                            ("--diagonalize", args.diagonalize),
                            ("--valence-only", args.valence_only),
                            ("--kpoint", args.kpoint),
                            ("--real-coefficient", args.real_coefficient),
                            ("--bond", args.bond),
                            ("--conventional", args.conventional)):
            if value:
                parser.error(f"{flag} is not used with --band.")
        dispatch_argv = ["--poscar", args.cell]
        if args.co_left:
            dispatch_argv.extend(["--co-left", *args.co_left])
        if args.co_right:
            dispatch_argv.extend(["--co-right", *args.co_right])
        if args.fatband:
            dispatch_argv.append("--fatband")
        if args.band_points is not None:
            dispatch_argv.extend(["--band-points", str(args.band_points)])
        if args.align is not None:
            dispatch_argv.extend(["--align", args.align])
        if args.window is not None:
            dispatch_argv.extend(["--window", *map(str, args.window)])
        if args.oxidation:
            dispatch_argv.extend(["--oxidation", *args.oxidation])
        if args.electrons is not None:
            dispatch_argv.extend(["--electrons", str(args.electrons)])
        if args.output is not None:
            dispatch_argv.extend(["--output", args.output])
        if args.tolerance is not None:
            dispatch_argv.extend(["--tolerance", str(args.tolerance)])
        for flag, value in (("--basis", args.basis),
                            ("--pseudo", args.pseudo), ("--xc", args.xc)):
            if value is not None:
                dispatch_argv.extend([flag, value])
        if args.kmesh is not None:
            dispatch_argv.extend(["--kmesh", *map(str, args.kmesh)])
        if args.ke_cutoff is not None:
            dispatch_argv.extend(["--ke-cutoff", str(args.ke_cutoff)])
        if args.no_ghost:
            dispatch_argv.append("--no-ghost")
        if args.max_l is not None:
            dispatch_argv.extend(["--max-l", str(args.max_l)])
        if args.projection is not None:
            dispatch_argv.extend(["--projection", args.projection])
        if args.chk is not None:
            dispatch_argv.extend(["--chk", args.chk])
        if args.onsite:
            dispatch_argv.append("--onsite")

        from ..band_pyscf import main as band_pyscf_main

        band_pyscf_main(dispatch_argv)
        return

    if args.dos:
        if not args.pyscf:
            parser.error("--dos requires --pyscf (it reads the PySCF "
                         "density matrix).")
        for flag, value in (("--diagram", args.diagram),
                            ("--visualize", args.visualize)):
            if value:
                parser.error(f"--dos and {flag} are separate runs; call "
                             "them one at a time (they can share the same "
                             "--chk).")
        dispatch_argv = ["--poscar", args.cell]
        if args.co_left:
            dispatch_argv.extend(["--co-left", *args.co_left])
        if args.co_right:
            dispatch_argv.extend(["--co-right", *args.co_right])
        if args.dos_kmesh is not None:
            dispatch_argv.extend(["--dos-kmesh", *map(str, args.dos_kmesh)])
        if args.align is not None:
            dispatch_argv.extend(["--align", args.align])
        if args.oxidation:
            dispatch_argv.extend(["--oxidation", *args.oxidation])
        if args.electrons is not None:
            dispatch_argv.extend(["--electrons", str(args.electrons)])
        if args.output is not None:
            dispatch_argv.extend(["--output", args.output])
        if args.tolerance is not None:
            dispatch_argv.extend(["--tolerance", str(args.tolerance)])
        for flag, value in (("--basis", args.basis),
                            ("--pseudo", args.pseudo), ("--xc", args.xc)):
            if value is not None:
                dispatch_argv.extend([flag, value])
        if args.kmesh is not None:
            dispatch_argv.extend(["--kmesh", *map(str, args.kmesh)])
        if args.ke_cutoff is not None:
            dispatch_argv.extend(["--ke-cutoff", str(args.ke_cutoff)])
        if args.no_ghost:
            dispatch_argv.append("--no-ghost")
        if args.max_l is not None:
            dispatch_argv.extend(["--max-l", str(args.max_l)])
        if args.projection is not None:
            dispatch_argv.extend(["--projection", args.projection])
        if args.chk is not None:
            dispatch_argv.extend(["--chk", args.chk])
        if args.onsite:
            # the DOS only ever uses the crystal density, so the
            # single-SCF mode (and its crystal-only chk files) fit here too
            dispatch_argv.append("--onsite")

        from ..dos_pyscf import main as dos_pyscf_main

        dos_pyscf_main(dispatch_argv)
        return

    if args.visualize:
        if args.onsite:
            parser.error(
                "--onsite is not supported with --visualize: the SALC "
                "viewer draws the fragment SCF eigenstates; use --onsite "
                "with --diagram --pyscf or --dos --pyscf.")
        if args.pyscf:
            # PySCF eigen-levels in the SALC viewer: all special k points
            # automatically, one page per k; --sublattice picks a fragment
            dispatch_argv = ["--poscar", args.cell]
            if args.sublattice:
                dispatch_argv.extend(["--sublattice", *args.sublattice])
            for el1, el2, max_length in args.bond or []:
                dispatch_argv.extend(["--bond", el1, el2, max_length])
            if args.real_coefficient:
                dispatch_argv.append("--real-coefficient")
            if args.kpoint is not None:
                if len(args.kpoint) != 1:
                    parser.error(
                        "--visualize --pyscf --kpoint takes a special-point "
                        "label such as GM/X/M/R (pages are written per "
                        "special k point; omit it for all of them)."
                    )
                dispatch_argv.extend(["--kpoint", args.kpoint[0]])
            if args.window is not None:
                dispatch_argv.extend(["--window", *map(str, args.window)])
            if args.diagonalize:
                dispatch_argv.append("--diagonalize")
            if args.valence_only:
                dispatch_argv.append("--valence-only")
            if args.oxidation:
                dispatch_argv.extend(["--oxidation", *args.oxidation])
            if args.electrons is not None:
                dispatch_argv.extend(["--electrons", str(args.electrons)])
            if args.output is not None:
                dispatch_argv.extend(["--output", args.output])
            if args.tolerance is not None:
                dispatch_argv.extend(["--tolerance", str(args.tolerance)])
            for flag, value in (("--basis", args.basis),
                                ("--pseudo", args.pseudo),
                                ("--xc", args.xc)):
                if value is not None:
                    dispatch_argv.extend([flag, value])
            if args.kmesh is not None:
                dispatch_argv.extend(["--kmesh", *map(str, args.kmesh)])
            if args.ke_cutoff is not None:
                dispatch_argv.extend(["--ke-cutoff", str(args.ke_cutoff)])
            if args.no_align:
                dispatch_argv.append("--no-align")
            if args.no_ghost:
                dispatch_argv.append("--no-ghost")
            if args.no_symmetrize:
                dispatch_argv.append("--no-symmetrize")
            if args.max_l is not None:
                dispatch_argv.extend(["--max-l", str(args.max_l)])
            if args.projection is not None:
                dispatch_argv.extend(["--projection", args.projection])
            if args.chk is not None:
                dispatch_argv.extend(["--chk", args.chk])
            if args.conventional:
                dispatch_argv.append("--conventional")

            from ..visualize_pyscf import main as visualize_pyscf_main

            visualize_pyscf_main(dispatch_argv)
            return

        if args.sublattice or args.window is not None:
            parser.error("--sublattice/--window require --visualize --pyscf.")
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

    if args.diagram:
        if not (args.co_left and args.co_right):
            parser.error(
                "--diagram requires --co-left and --co-right with the two "
                "fragment sublattices (full valence basis), e.g. "
                "--co-left SrTi --co-right O3; the hover wave-function "
                "sketches are always embedded."
            )
        dispatch_argv = ["--poscar", args.cell,
                         "--co-left", *args.co_left,
                         "--co-right", *args.co_right]
        if args.atomic_orbital:
            parser.error(
                "--diagram no longer takes --atomic-orbital: the hover "
                "wave-function sketches are always embedded, for every level, "
                "with all of its atomic-orbital components."
            )
        if args.oxidation:
            dispatch_argv.extend(["--oxidation", *args.oxidation])
        if args.kpoint is not None:
            if len(args.kpoint) != 1:
                parser.error(
                    "--diagram --kpoint takes a special-point label such as "
                    "GM/X/M/R (the diagram is drawn per special k point)."
                )
            dispatch_argv.extend(["--kpoint", args.kpoint[0]])
        if args.electrons is not None:
            dispatch_argv.extend(["--electrons", str(args.electrons)])
        if args.output is not None:
            dispatch_argv.extend(["--output", args.output])
        if args.tolerance is not None:
            dispatch_argv.extend(["--tolerance", str(args.tolerance)])
        if args.conventional:
            dispatch_argv.append("--conventional")

        if not args.pyscf and args.onsite:
            parser.error("--onsite needs --pyscf: the extended-Hueckel "
                         "columns already come from the one shared "
                         "Hamiltonian.")
        if args.pyscf:
            for flag, value in (("--basis", args.basis), ("--pseudo", args.pseudo),
                                ("--xc", args.xc)):
                if value is not None:
                    dispatch_argv.extend([flag, value])
            if args.kmesh is not None:
                dispatch_argv.extend(["--kmesh", *map(str, args.kmesh)])
            if args.ke_cutoff is not None:
                dispatch_argv.extend(["--ke-cutoff", str(args.ke_cutoff)])
            if args.no_align:
                dispatch_argv.append("--no-align")
            if args.no_ghost:
                dispatch_argv.append("--no-ghost")
            if args.no_symmetrize:
                dispatch_argv.append("--no-symmetrize")
            if args.max_l is not None:
                dispatch_argv.extend(["--max-l", str(args.max_l)])
            if args.projection is not None:
                dispatch_argv.extend(["--projection", args.projection])
            if args.chk is not None:
                dispatch_argv.extend(["--chk", args.chk])
            if args.onsite:
                dispatch_argv.append("--onsite")

            from ..crystal_orbital_pyscf import main as pyscf_diagram_main

            pyscf_diagram_main(dispatch_argv)
            print_crystod_citation()
            return

        from ..crystal_orbital_diagram import main as crystal_diagram_main

        crystal_diagram_main(dispatch_argv)
        print_crystod_citation()
        return
    for flag, value in (("--pyscf", args.pyscf), ("--basis", args.basis),
                        ("--pseudo", args.pseudo), ("--kmesh", args.kmesh),
                        ("--ke-cutoff", args.ke_cutoff),
                        ("--no-align", args.no_align),
                        ("--no-ghost", args.no_ghost),
                        ("--no-symmetrize", args.no_symmetrize),
                        ("--max-l", args.max_l is not None),
                        ("--projection", args.projection),
                        ("--chk", args.chk),
                        ("--sublattice", args.sublattice),
                        ("--diagonalize", args.diagonalize),
                        ("--valence-only", args.valence_only)):
        if value:
            parser.error(f"{flag} is only used with --diagram or "
                         "--visualize --pyscf.")
    if args.onsite:
        parser.error("--onsite is only used with --diagram --pyscf, "
                     "--dos --pyscf or --band --pyscf.")
    if args.window is not None:
        parser.error("--window is only used with --visualize --pyscf "
                     "or --band.")
    if args.electrons is not None:
        parser.error("--electrons is only used with --diagram.")
    if args.co_left or args.co_right:
        parser.error("--co-left/--co-right are only used with --diagram.")
    if args.oxidation:
        parser.error("--oxidation is only used with --diagram.")

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
