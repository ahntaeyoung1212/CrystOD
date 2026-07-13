"""Shared helpers for the sectioned CrystOD commands.

Option conventions (phonopy-aligned):

- ``-c`` / ``--cell`` selects the crystal structure file; ``--poscar`` is
  kept as a hidden alias for backward compatibility.
- ``-o`` / ``--output`` selects the output path.
"""

from __future__ import annotations

from argparse import ArgumentParser

# "CrystOD" rendered in the figlet "standard" font. Embedded as a literal so
# the banner has no runtime dependency on pyfiglet.
_BANNER_ART = r"""
  ____                _    ___  ____
 / ___|_ __ _   _ ___| |_ / _ \|  _ \
| |   | '__| | | / __| __| | | | | | |
| |___| |  | |_| \__ \ |_| |_| | |_| |
 \____|_|   \__, |___/\__|\___/|____/
            |___/
"""


def banner() -> str:
    """CrystOD ASCII-art banner with the subtitle and current version.

    Used as the leading block of the main command's ``--help`` description.
    """
    from .. import __version__

    return f"{_BANNER_ART.strip(chr(10))}\nCrystal Orbital Diagram   version {__version__}"


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
