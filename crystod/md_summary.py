"""Summary statistics of an MD trajectory (XDATCAR).

Reports the time-averaged lattice parameters (a, b, c, alpha, beta, gamma)
and cell volume, with standard deviations, over a selected step range —
useful for extracting the equilibrium cell from an NpT trajectory.

Based on ``script/md_summary.py`` by Yasuhide Mochizuki (the original reads a
pre-extracted md.csv; this port computes the lattice statistics directly from
the XDATCAR — total energies and temperatures are not stored in XDATCAR).
"""

from __future__ import annotations

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from collections import Counter

import numpy as np

desc = """
Summarize an MD trajectory (XDATCAR): time-averaged lattice parameters
(a, b, c, alpha, beta, gamma) and cell volume with standard deviations.

# Command Example:
crystod-md --summary --start-step 1000 --xdatcar XDATCAR
"""


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=desc, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument(
        "--xdatcar",
        type=str,
        default="XDATCAR",
        help="Input XDATCAR path.",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=0,
        help="First MD step used in the analysis (earlier steps are discarded as equilibration).",
    )
    parser.add_argument(
        "--end-step",
        type=int,
        default=None,
        help="Last MD step used in the analysis (inclusive; default: last step).",
    )
    return parser


def _lattice_parameters(lattice: np.ndarray) -> tuple[float, float, float, float, float, float]:
    a_vec, b_vec, c_vec = lattice
    a = float(np.linalg.norm(a_vec))
    b = float(np.linalg.norm(b_vec))
    c = float(np.linalg.norm(c_vec))
    alpha = float(np.degrees(np.arccos(np.clip(np.dot(b_vec, c_vec) / (b * c), -1.0, 1.0))))
    beta = float(np.degrees(np.arccos(np.clip(np.dot(a_vec, c_vec) / (a * c), -1.0, 1.0))))
    gamma = float(np.degrees(np.arccos(np.clip(np.dot(a_vec, b_vec) / (a * b), -1.0, 1.0))))
    return a, b, c, alpha, beta, gamma


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    from .xdatcar_adp import read_xdatcar

    print(f"Input file : {args.xdatcar}")
    print("\nReading XDATCAR... (this may take a while)")
    chem_formula, lattices, all_coordinates = read_xdatcar(args.xdatcar)

    n_total = len(all_coordinates)
    if args.start_step >= n_total:
        raise SystemExit(
            f"ERROR: --start-step {args.start_step} exceeds the number of MD steps ({n_total})."
        )
    end = n_total if args.end_step is None else args.end_step + 1
    if end <= args.start_step:
        raise SystemExit(
            f"ERROR: --end-step {args.end_step} must not be smaller than --start-step {args.start_step}."
        )
    end = min(end, n_total)

    selected = np.array(lattices[args.start_step : end], dtype=float)
    n_steps = len(selected)
    n_atoms = all_coordinates.shape[1]
    composition = Counter(chem_formula)

    parameters = np.array([_lattice_parameters(lattice) for lattice in selected])
    volumes = np.abs(np.linalg.det(selected))

    print("\nTrajectory info:")
    print(f"  atoms          : {n_atoms}")
    print(f"  composition    : {dict(composition)}")
    print(f"  total steps    : {n_total}")
    print(f"  analyzed steps : {n_steps} (step {args.start_step} .. {args.start_step + n_steps - 1})")

    print("\nTime-averaged cell (mean +/- std):")
    labels = ["a (A)", "b (A)", "c (A)", "alpha (deg)", "beta (deg)", "gamma (deg)"]
    for label, values in zip(labels, parameters.T):
        print(f"  {label:<12}: {values.mean():14.6f} +/- {values.std():.6f}")
    print(f"  {'V (A^3)':<12}: {volumes.mean():14.6f} +/- {volumes.std():.6f}")
    print(f"  {'V/atom (A^3)':<12}: {volumes.mean() / n_atoms:14.6f}")

    print(
        "\nNOTE: XDATCAR stores no energies or temperatures; "
        "Etot/T averages require OSZICAR/vasprun.xml."
    )


if __name__ == "__main__":
    main()
