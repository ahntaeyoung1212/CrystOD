"""Write the two small trajectories of the 40_md_projection example.

XDATCAR_synthetic: 2x2x2 ScF3 (../test_POSCARs/221_PPOSCAR_ScF3) with
component 1 of R4+ frozen as 0.3 sin(2 pi t/50) A and the M3+ pattern at
M (1/2, 1/2, 0) as 0.17 cos(2 pi t/37) A, 100 frames, fixed cell, atoms in
a shuffled order.

XDATCAR_I4mcm_1: the --modulation structure
../33_modulation/ScF3_Pm-3m/POSCAR_I4mcm_3 (R4+, --mode 3, the rotation
about c, amplitude 0.3 A) as a one-frame trajectory.

The fractional coordinates of XDATCAR_synthetic are written with 14
decimals: with the usual 8, the rounding leaves about 1e-8 A on the
components that should be zero, above the 1e-9 bound of the checks.

Usage (in this directory):
    python make_xdatcar.py [--output-dir DIR]
"""

import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE = os.path.dirname(HERE)


def write_xdatcar(path, lattice, symbols, frames, species_order):
    """Write a fixed-cell XDATCAR with the species in ``species_order``."""
    order = [i for element in species_order for i, s in enumerate(symbols) if s == element]
    counts = [symbols.count(element) for element in species_order]
    lines = ["ScF3 2x2x2 synthetic", "1.0"]
    lines += ["  " + "  ".join(f"{v:.10f}" for v in row) for row in lattice]
    lines += ["  " + "  ".join(species_order), "  " + "  ".join(map(str, counts))]
    for t, frame in enumerate(frames, start=1):
        lines.append(f"Direct configuration= {t:5d}")
        lines += ["  " + "  ".join(f"{v:.14f}" for v in frame[i]) for i in order]
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    """Write XDATCAR_synthetic and XDATCAR_I4mcm_1 into --output-dir."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", default=HERE,
                        help="directory to write into (default: this example directory)")
    args = parser.parse_args()

    from crystod.star_of_k import read_poscar_or_exit
    from crystod.vibration_modes import SymmetryOnlyVibrations

    parent = read_poscar_or_exit(os.path.join(EXAMPLE, "test_POSCARs", "221_PPOSCAR_ScF3"))
    vibrations = SymmetryOnlyVibrations(parent, standardize=False)
    q_r, q_m = [0.5, 0.5, 0.5], [0.5, 0.5, 0.0]
    row_r = vibrations.get_symmetry_adapted_spaces(q_r)[2][0]  # R4+(3), space 3, component 1
    row_m = vibrations.get_symmetry_adapted_spaces(q_m)[1][0]  # M3+(1), space 2, component 1
    positions, pattern_r, symbols, lattice = vibrations.get_supercell_displacements(
        q_r, row_r, (2, 2, 2))
    _, pattern_m, _, _ = vibrations.get_supercell_displacements(q_m, row_m, (2, 2, 2))
    t = np.arange(100)
    amp_r = 0.3 * np.sin(2 * np.pi * t / 50)
    amp_m = 0.17 * np.cos(2 * np.pi * t / 37)
    cartesian = (positions[None] + amp_r[:, None, None] * pattern_r[None]
                 + amp_m[:, None, None] * pattern_m[None])
    frames = (cartesian @ np.linalg.inv(lattice)) % 1.0
    shuffle = np.random.default_rng(38).permutation(len(symbols))
    write_xdatcar(os.path.join(args.output_dir, "XDATCAR_synthetic"), lattice,
                  [symbols[i] for i in shuffle], frames[:, shuffle], ["F", "Sc"])

    with open(os.path.join(EXAMPLE, "33_modulation", "ScF3_Pm-3m", "POSCAR_I4mcm_3")) as handle:
        lines = handle.read().splitlines()
    with open(os.path.join(args.output_dir, "XDATCAR_I4mcm_1"), "w") as handle:
        handle.write("\n".join(lines[:7] + ["Direct configuration=     1"] + lines[8:40]) + "\n")
    print("Written: XDATCAR_synthetic, XDATCAR_I4mcm_1")


if __name__ == "__main__":
    main()
