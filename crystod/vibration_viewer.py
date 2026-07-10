"""
Helper viewer for crystod vibration .npz exports.
"""

from __future__ import annotations

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, RawDescriptionHelpFormatter, RawTextHelpFormatter
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write as ase_write


class MyHelpFormatter(
    RawTextHelpFormatter,
    RawDescriptionHelpFormatter,
    ArgumentDefaultsHelpFormatter,
):
    pass


desc = """
Open a crystod vibration .npz export, print its contents, and optionally
write a displaced structure or an animation trajectory.

# Command Examples:
python3 vibration_viewer.py --npz vibration.npz
python3 vibration_viewer.py --npz vibration.npz --write-poscar POSCAR_view
python3 vibration_viewer.py --npz vibration.npz --write-trajectory vibration.xyz
"""


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=desc, formatter_class=MyHelpFormatter)
    parser.add_argument(
        "--npz",
        required=True,
        help="Path to a .npz file exported by crystod-phonon --vibration.",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=None,
        help="Override the stored amplitude when writing structures.",
    )
    parser.add_argument(
        "--write-poscar",
        default=None,
        help="Optional POSCAR path for one displaced snapshot.",
    )
    parser.add_argument(
        "--write-trajectory",
        default=None,
        help="Optional trajectory path such as .xyz or .extxyz for a vibration animation.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=21,
        help="Number of frames for trajectory output.",
    )
    return parser


def _as_text(value) -> str:
    if isinstance(value, np.ndarray) and value.shape == ():
        return str(value.item())
    return str(value)


def _load_npz(path: str) -> dict[str, object]:
    raw = np.load(path, allow_pickle=True)
    payload: dict[str, object] = {}
    for key in raw.files:
        value = raw[key]
        payload[key] = value.item() if isinstance(value, np.ndarray) and value.shape == () else value
    return payload


def _make_atoms(payload: dict[str, object], amplitude: float) -> Atoms:
    positions = np.asarray(payload["positions"], dtype=float)
    displacements = np.asarray(payload["displacements"], dtype=float)
    symbols = list(np.asarray(payload["symbols"], dtype=object))
    lattice = np.asarray(payload["supercell_lattice"], dtype=float)
    return Atoms(
        symbols=symbols,
        positions=positions + amplitude * displacements,
        cell=lattice,
        pbc=True,
    )


def _write_trajectory(payload: dict[str, object], amplitude: float, frames: int, output_path: str) -> None:
    positions = np.asarray(payload["positions"], dtype=float)
    displacements = np.asarray(payload["displacements"], dtype=float)
    symbols = list(np.asarray(payload["symbols"], dtype=object))
    lattice = np.asarray(payload["supercell_lattice"], dtype=float)
    images = []
    phase_values = np.linspace(0.0, 2.0 * np.pi, frames, endpoint=False)
    for phase in phase_values:
        scale = amplitude * np.sin(phase)
        images.append(
            Atoms(
                symbols=symbols,
                positions=positions + scale * displacements,
                cell=lattice,
                pbc=True,
            )
        )
    ase_write(output_path, images)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(f"File '{npz_path}' does not exist.")

    payload = _load_npz(str(npz_path))
    amplitude = float(args.amplitude if args.amplitude is not None else payload.get("amplitude", 0.3))

    print(f"Loaded: {npz_path}")
    print(f"q-point label : {_as_text(payload.get('qpoint_label', 'unknown'))}")
    print(f"q-point       : {np.round(np.asarray(payload['qpoint'], dtype=float), 6).tolist()}")
    print(f"mode space    : {payload.get('mode_index', 'n/a')}")
    print(f"component     : {payload.get('component_index', 'n/a')}")
    print(f"irrep label   : {_as_text(payload.get('selected_irrep_label', 'unknown'))}")
    print(f"amplitude     : {amplitude}")
    print(f"atoms         : {len(np.asarray(payload['symbols'], dtype=object))}")

    displacements = np.asarray(payload["displacements"], dtype=float)
    norms = np.linalg.norm(displacements, axis=1)
    print(f"displacement norms (unit amplitude): min={norms.min():.6f}, max={norms.max():.6f}")

    if args.write_poscar:
        atoms = _make_atoms(payload, amplitude)
        ase_write(args.write_poscar, atoms, format="vasp", direct=True)
        print(f"Wrote displaced structure: {args.write_poscar}")

    if args.write_trajectory:
        _write_trajectory(payload, amplitude, args.frames, args.write_trajectory)
        print(f"Wrote vibration trajectory: {args.write_trajectory}")


if __name__ == "__main__":
    main()
