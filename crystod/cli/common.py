"""Shared helpers for the sectioned CrystOD commands.

Option conventions (phonopy-aligned):

- ``-c`` / ``--cell`` selects the crystal structure file; ``--poscar`` is
  kept as a hidden alias for backward compatibility.
- ``-o`` / ``--output`` selects the output path.
"""

from __future__ import annotations

from argparse import ArgumentParser


def add_cell_argument(parser: ArgumentParser, help_suffix: str = "") -> None:
    """Add the phonopy-style structure-file option (-c/--cell, alias --poscar)."""
    parser.add_argument(
        "-c",
        "--cell",
        "--poscar",
        dest="cell",
        default="POSCAR",
        metavar="FILE",
        help=f"Crystal structure file in VASP POSCAR format (default: POSCAR).{help_suffix}",
    )


def add_output_argument(parser: ArgumentParser, help_text: str) -> None:
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        default=None,
        metavar="FILE",
        help=help_text,
    )
