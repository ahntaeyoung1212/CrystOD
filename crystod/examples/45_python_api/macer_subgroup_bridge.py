"""Using the CrystOD API from a structure-search workflow (macer example).

This is the symmetry-lowering step of ``macer phonopy tree``
(``macer/phonopy/familytree.py``, ``_get_stable_structure``) rewritten on top
of the CrystOD API. Instead of freezing in each imaginary eigenvector one by
one and asking spglib afterwards what came out, the imaginary levels are
labeled with their irreps and the isotropy subgroups of those irreps are
enumerated -- every order-parameter direction, independent of the arbitrary
basis the diagonalization happens to return.

Run it in a directory holding POSCAR + FORCE_SETS, e.g.

    python macer_subgroup_bridge.py --dim 4 4 4 -c 221_PPOSCAR_SrTiO3

For the equivalent one-liner without any Python, see
``crystod-phonon --subgroup``.
"""

from __future__ import annotations

import argparse

import phonopy

from crystod.phonon import scan_imaginary_modes


def candidate_children(poscar, dim, threshold=-0.1, symprec=1e-4):
    """Space groups reachable from the imaginary modes of ``poscar``.

    Returns one entry per (imaginary level, order-parameter direction):
    ``(frequency, q label, irrep, direction, subgroup symbol, cell multiple)``.
    """
    phonon = phonopy.load(
        supercell_matrix=dim,
        # "auto" keeps a zone-boundary instability at its true q point; with an
        # identity primitive matrix it folds onto the supercell Gamma point,
        # where it carries no irrep of the parent space group.
        primitive_matrix="auto",
        unitcell_filename=poscar,
        force_sets_filename="FORCE_SETS",
        symprec=symprec,
    )

    children = []
    for result in scan_imaginary_modes(phonon, threshold=threshold):
        mode = result.mode
        for sub in result.subgroups:
            children.append(
                (mode.frequency, mode.qpoint_label or str(mode.qpoint),
                 "+".join(mode.labels), sub.direction, sub.symbol, sub.size)
            )
    return children


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--poscar", default="POSCAR")
    parser.add_argument("--dim", nargs=3, type=int, default=[1, 1, 1])
    parser.add_argument("--threshold", type=float, default=-0.1)
    args = parser.parse_args()

    children = candidate_children(args.poscar, args.dim, args.threshold)
    if not children:
        print("dynamically stable: no imaginary mode at any commensurate q point")
        return

    print(f"{'freq (THz)':>11}  {'q':<4} {'irrep':<8} {'direction':<10} "
          f"{'subgroup':<10} cell")
    for frequency, q, irrep, direction, symbol, size in children:
        print(f"{frequency:11.4f}  {q:<4} {irrep:<8} {direction:<10} "
              f"{symbol:<10} x{size}")

    print()
    print("Each direction is one candidate child phase. Generate its distorted")
    print("structure with crystod-phonon --modulation (the default output name")
    print("carries the irrep and the resulting space group), then relax it.")


if __name__ == "__main__":
    main()
