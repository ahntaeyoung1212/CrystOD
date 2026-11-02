"""Regenerate the representation-matrix gallery figures of doc/crystod.md.

Reproduces the imshow galleries of matsym/get_basis_functions.ipynb (H. Koiso)
for ScF3 (Pm-3m) at the Gamma point: one panel per little-group operation,
red = +1, blue = -1, white = 0.

Run from the repository root:
    python doc/make_rep_matrix_figures.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from phonopy.interface.calculator import read_crystal_structure
from spglib import get_symmetry_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSCAR = os.path.join(ROOT, "example", "test_POSCARs", "221_PPOSCAR_ScF3")
OUTDIR = os.path.join(ROOT, "doc", "images")


def quadratic_representation(rotation):
    """6x6 matrix of a 3x3 Cartesian rotation on (x^2, y^2, z^2, xy, yz, zx)."""
    monomials = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0)]
    rep = np.zeros((6, 6))
    for col, (i, j) in enumerate(monomials):
        coeff = np.outer(rotation[:, i], rotation[:, j])
        rep[0, col] = coeff[0, 0]
        rep[1, col] = coeff[1, 1]
        rep[2, col] = coeff[2, 2]
        rep[3, col] = coeff[0, 1] + coeff[1, 0]
        rep[4, col] = coeff[1, 2] + coeff[2, 1]
        rep[5, col] = coeff[2, 0] + coeff[0, 2]
    return rep


def plot_gallery(matrices, path):
    nrows, ncols = 6, 8
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows))
    for index, ax in enumerate(axes.ravel()):
        ax.imshow(matrices[index], cmap="bwr", vmin=-1, vmax=1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    cell, _ = read_crystal_structure(POSCAR, interface_mode="vasp")
    dataset = get_symmetry_dataset(cell.totuple(), symprec=1e-4)

    # Gamma point of Pm-3m: the little group is the full 48-operation group.
    # Cartesian rotation matrices (identical to the fractional ones for cubic P).
    lattice = np.transpose(cell.cell)
    rotations = lattice @ dataset.rotations @ np.linalg.inv(lattice)
    rotations = np.rint(rotations).astype(int)
    assert len(rotations) == 48

    plot_gallery(rotations, os.path.join(OUTDIR, "rep_matrices_linear.png"))
    plot_gallery([quadratic_representation(r) for r in rotations],
                 os.path.join(OUTDIR, "rep_matrices_quadratic.png"))


if __name__ == "__main__":
    main()
