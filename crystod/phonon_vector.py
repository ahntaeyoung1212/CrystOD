"""
Phonon eigenvector visualization workflow for crystod.

Given POSCAR + FORCE_SETS (or FORCE_CONSTANTS), diagonalize the dynamical
matrix at a selected q-point, list the modes with their irrep labels, and
export selected eigenvectors as VESTA files with displacement arrows.

The VESTA export follows the approach of Phonopy_VESTA
(A. P. Roy et al., Phys. Rev. Lett. 132, 026701 (2024),
https://doi.org/10.1103/PhysRevLett.132.026701): a complete VESTA file is
written (VESTA ignores vectors in files missing its style sections) with
VECTR/VECTT arrow entries placed between SITET and SPLAN.
"""

from __future__ import annotations

from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    RawDescriptionHelpFormatter,
    RawTextHelpFormatter,
)
import re
from fractions import Fraction
from math import gcd
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import warnings

from .spglib_compat import ensure_spglib_compat

ensure_spglib_compat()

from phonopy import load
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import get_primitive_matrix_by_centring, get_supercell
from spgrep.representation import project_to_irrep

from .irreptables_compat import load_irreptables
from .phonon_irreps import get_irrep_labels, get_irt_special_points
from .runtime_compat import get_symmetry_dataset
from .vibration_modes import SymmetryOnlyVibrations

IrrepTable, Irrep = load_irreptables()

# sqrt(eV/A^2/AMU) -> THz, same constant as crystod-phonon --modulation.
FREQUENCY_CONVERSION_THZ = 15.633302


class MyHelpFormatter(
    RawTextHelpFormatter,
    RawDescriptionHelpFormatter,
    ArgumentDefaultsHelpFormatter,
):
    pass


desc = """
Visualize phonon eigenvectors as VESTA files with displacement arrows.
POSCAR and FORCE_SETS (or FORCE_CONSTANTS with --readfc) must exist in the
directory where this code runs, exactly as in --phonon-irrep mode.

# Command Examples:
crystod-phonon --vector --dim "4 4 4" -c 227_PPOSCAR_Si --qpoint GM
crystod-phonon --vector --dim "4 4 4" -c 227_PPOSCAR_Si --qpoint GM              # all modes
crystod-phonon --vector --dim "4 4 4" -c 227_PPOSCAR_Si --qpoint GM --mode 4 5 6 # summed pattern
crystod-phonon --vector --dim "4 4 4" -c 227_PPOSCAR_Si --qpoint 0.5 0 0.5 --mode 1
"""

GAMMA_ALIASES = {"G", "GM", "GAMMA", "Γ"}

# VESTA default per-element atomic radius and color (from VESTA's elements.ini).
VESTA_ELEMENTS: dict[str, tuple[float, int, int, int]] = {
    "H": (0.46, 255, 204, 204),
    "D": (0.46, 204, 204, 255),
    "He": (1.22, 252, 233, 207),
    "Li": (1.57, 134, 224, 116),
    "Be": (1.12, 95, 216, 123),
    "B": (0.81, 32, 162, 15),
    "C": (0.77, 129, 73, 41),
    "N": (0.74, 176, 186, 230),
    "O": (0.74, 255, 3, 0),
    "F": (0.72, 176, 186, 230),
    "Ne": (1.60, 255, 56, 181),
    "Na": (1.91, 250, 221, 61),
    "Mg": (1.60, 252, 124, 22),
    "Al": (1.43, 129, 179, 214),
    "Si": (1.18, 27, 59, 250),
    "P": (1.10, 193, 156, 195),
    "S": (1.04, 255, 250, 0),
    "Cl": (0.99, 50, 252, 3),
    "Ar": (1.92, 207, 254, 197),
    "K": (2.35, 161, 34, 247),
    "Ca": (1.97, 91, 150, 190),
    "Sc": (1.64, 182, 99, 172),
    "Ti": (1.47, 120, 202, 255),
    "V": (1.35, 230, 26, 0),
    "Cr": (1.29, 0, 0, 158),
    "Mn": (1.37, 169, 9, 158),
    "Fe": (1.26, 181, 114, 0),
    "Co": (1.25, 0, 0, 175),
    "Ni": (1.25, 184, 188, 190),
    "Cu": (1.28, 34, 71, 221),
    "Zn": (1.37, 143, 144, 130),
    "Ga": (1.53, 159, 228, 116),
    "Ge": (1.22, 126, 111, 166),
    "As": (1.21, 117, 208, 87),
    "Se": (1.04, 154, 239, 16),
    "Br": (1.14, 127, 49, 3),
    "Kr": (1.98, 250, 193, 243),
    "Rb": (2.50, 255, 0, 153),
    "Sr": (2.15, 0, 255, 39),
    "Y": (1.82, 103, 152, 142),
    "Zr": (1.60, 0, 255, 0),
    "Nb": (1.47, 76, 179, 118),
    "Mo": (1.40, 180, 134, 176),
    "Tc": (1.35, 205, 175, 203),
    "Ru": (1.34, 207, 184, 174),
    "Rh": (1.34, 206, 210, 171),
    "Pd": (1.37, 194, 196, 185),
    "Ag": (1.44, 184, 188, 190),
    "Cd": (1.52, 243, 31, 220),
    "In": (1.67, 215, 129, 187),
    "Sn": (1.58, 155, 143, 186),
    "Sb": (1.41, 216, 131, 80),
    "Te": (1.37, 173, 162, 82),
    "I": (1.33, 143, 31, 139),
    "Xe": (2.18, 155, 161, 248),
    "Cs": (2.72, 15, 255, 185),
    "Ba": (2.24, 30, 240, 45),
    "La": (1.88, 90, 196, 73),
    "Ce": (1.82, 209, 253, 6),
    "Pr": (1.82, 253, 226, 6),
    "Nd": (1.82, 252, 142, 7),
    "Pm": (1.81, 0, 0, 245),
    "Sm": (1.81, 253, 6, 125),
    "Eu": (2.06, 251, 8, 213),
    "Gd": (1.79, 192, 4, 255),
    "Tb": (1.77, 113, 4, 254),
    "Dy": (1.77, 49, 6, 253),
    "Ho": (1.76, 7, 66, 251),
    "Er": (1.75, 73, 115, 59),
    "Tm": (1.00, 0, 0, 224),
    "Yb": (1.94, 39, 253, 244),
    "Lu": (1.72, 38, 253, 181),
    "Hf": (1.59, 180, 180, 89),
    "Ta": (1.47, 183, 155, 86),
    "W": (1.41, 142, 138, 128),
    "Re": (1.37, 179, 177, 142),
    "Os": (1.35, 201, 177, 121),
    "Ir": (1.36, 201, 207, 115),
    "Pt": (1.39, 204, 198, 191),
    "Au": (1.44, 254, 179, 56),
    "Hg": (1.55, 211, 184, 204),
    "Tl": (1.71, 150, 137, 109),
    "Pb": (1.75, 83, 83, 91),
    "Bi": (1.82, 210, 48, 248),
    "Po": (1.77, 0, 0, 255),
    "At": (0.62, 0, 0, 255),
    "Rn": (0.80, 255, 255, 0),
    "Fr": (1.00, 0, 0, 0),
    "Ra": (2.35, 110, 170, 89),
    "Ac": (2.03, 100, 158, 115),
    "Th": (1.80, 38, 254, 120),
    "Pa": (1.63, 41, 251, 53),
    "U": (1.56, 122, 162, 170),
    "Np": (1.56, 76, 76, 76),
    "Pu": (1.64, 76, 76, 76),
    "Am": (1.73, 76, 76, 76),
}
DEFAULT_ELEMENT = (0.80, 76, 76, 76)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=desc, formatter_class=MyHelpFormatter)
    parser.add_argument(
        "--dim",
        required=True,
        type=str,
        help="Supercell dimension used for the force calculation.",
    )
    parser.add_argument(
        "--poscar",
        type=str,
        default="POSCAR",
        help="POSCAR path.",
    )
    parser.add_argument(
        "--readfc",
        action="store_true",
        help="Read FORCE_CONSTANTS instead of FORCE_SETS.",
    )
    parser.add_argument(
        "--qpoint",
        nargs="+",
        required=True,
        help="Either a high-symmetry label such as GM/X/L or three primitive reciprocal coordinates.",
    )
    parser.add_argument(
        "--mode",
        nargs="+",
        type=int,
        default=None,
        help="Mode number(s) (1-based, sorted by frequency) to export as one summed VESTA "
        "file. When omitted, ALL modes are exported as individual VESTA files.",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=1.5,
        help="Arrow length in Angstroms given to the largest atomic displacement of each mode.",
    )
    parser.add_argument(
        "--conventional",
        action="store_true",
        help="Output the VESTA file in the conventional cell instead of the primitive cell.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .vesta path (auto-generated as POSCAR_<formula>_<qlabel>_mode<N>.vesta when omitted).",
    )
    parser.add_argument(
        "--tolerance",
        "-tol",
        dest="tol",
        type=float,
        default=1e-3,
        help="Degeneracy tolerance for irrep labeling.",
    )
    return parser


def _parse_coordinate(value: str) -> float:
    try:
        return float(Fraction(value))
    except ValueError:
        return float(value)


def resolve_qpoint(
    raw_qpoint: list[str],
    q_names: list[str],
    q_list: list[list[float]],
) -> tuple[str, list[float]]:
    """Resolve --qpoint tokens into (label, primitive-basis coordinates)."""
    if len(raw_qpoint) == 1:
        requested = raw_qpoint[0].strip().upper()
        if requested in GAMMA_ALIASES:
            requested = "GM"
        for name, q in zip(q_names, q_list):
            if name.upper() == requested:
                return name, list(q)
        available = ", ".join(q_names)
        raise ValueError(
            f"Unknown q-point label '{raw_qpoint[0]}'. Available labels: {available}"
        )

    if len(raw_qpoint) != 3:
        raise ValueError("--qpoint must be either one label or three coordinates.")

    qpoint = [_parse_coordinate(value) for value in raw_qpoint]
    for name, q in zip(q_names, q_list):
        if np.allclose(qpoint, q, atol=1e-8):
            return name, qpoint
    label = "q" + "".join(f"_{value:g}" for value in qpoint).replace("/", "o")
    return label, qpoint


def reduced_formula(symbols: list[str]) -> str:
    """Reduced chemical formula preserving the first-appearance order, e.g. Si, SrTiO3."""
    unique: list[str] = []
    counts: dict[str, int] = {}
    for symbol in symbols:
        if symbol not in counts:
            unique.append(symbol)
            counts[symbol] = 0
        counts[symbol] += 1
    divisor = 0
    for symbol in unique:
        divisor = gcd(divisor, counts[symbol])
    parts = []
    for symbol in unique:
        count = counts[symbol] // divisor
        parts.append(symbol if count == 1 else f"{symbol}{count}")
    return "".join(parts)


def get_conventional_matrix(centring: str) -> NDArray[np.int_]:
    """Integer primitive-to-conventional matrix; rows are the conventional
    lattice vectors expressed in the primitive basis (e.g. cubic-F:
    [[-1, 1, 1], [1, -1, 1], [1, 1, -1]])."""
    primitive_matrix = np.array(get_primitive_matrix_by_centring(centring), dtype=float)
    conventional = np.linalg.inv(primitive_matrix)
    conventional_int = np.rint(conventional).astype(int)
    if not np.allclose(conventional, conventional_int, atol=1e-8):
        raise ValueError(f"Non-integer primitive-to-conventional matrix for centring '{centring}'.")
    return conventional_int


def get_commensurate_supercell_matrix(
    qpoint: list[float],
    base_matrix: NDArray[np.int_],
) -> NDArray[np.int_]:
    """Smallest diagonal multiple of base_matrix commensurate with q.

    base_matrix rows are the base-cell (primitive or conventional) lattice
    vectors in the primitive basis; the returned matrix S satisfies
    q . S_row in Z for every row, so the Bloch phase is periodic over the
    supercell L_super = S @ L_primitive.
    """
    q_base = np.array(base_matrix, dtype=float) @ np.array(qpoint, dtype=float)
    sizes = []
    for component in q_base:
        if abs(component - round(component)) < 1e-10:
            sizes.append(1)
        else:
            sizes.append(Fraction(float(component)).limit_denominator(12).denominator)
    return np.diag(sizes) @ np.array(base_matrix, dtype=int)


def _lattice_parameters(lattice: NDArray[np.float64]) -> tuple[float, float, float, float, float, float]:
    lengths = np.linalg.norm(lattice, axis=1)
    a, b, c = lengths
    alpha = np.degrees(np.arccos(np.dot(lattice[1], lattice[2]) / (b * c)))
    beta = np.degrees(np.arccos(np.dot(lattice[0], lattice[2]) / (a * c)))
    gamma = np.degrees(np.arccos(np.dot(lattice[0], lattice[1]) / (a * b)))
    return a, b, c, alpha, beta, gamma


_VESTA_HEADER_SECTIONS = """GROUP
1 1 P 1
SYMOP
 0.000000  0.000000  0.000000  1  0  0   0  1  0   0  0  1   1
 -1.0 -1.0 -1.0  0 0 0  0 0 0  0 0 0
TRANM 0
 0.000000  0.000000  0.000000  1  0  0   0  1  0   0  0  1
LTRANSL
 -1
 0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
LORIENT
 -1   0   0   0   0
 1.000000  0.000000  0.000000  1.000000  0.000000  0.000000
 0.000000  0.000000  1.000000  0.000000  0.000000  1.000000
LMATRIX
 1.000000  0.000000  0.000000  0.000000
 0.000000  1.000000  0.000000  0.000000
 0.000000  0.000000  1.000000  0.000000
 0.000000  0.000000  0.000000  1.000000
 0.000000  0.000000  0.000000"""

_VESTA_TAIL_SECTIONS = """SPLAN
  0   0   0   0
LBLAT
 -1
LBLSP
 -1
DLATM
 -1
DLBND
 -1
DLPLY
 -1
PLN2D
  0   0   0   0"""

_VESTA_STYLE_SECTIONS = """SCENE
 1.000000  0.000000  0.000000  0.000000
 0.000000  1.000000  0.000000  0.000000
 0.000000  0.000000  1.000000  0.000000
 0.000000  0.000000  0.000000  1.000000
  0.000   0.000
  0.000
  1.000
HBOND 0 2

STYLE
DISPF 37753794
MODEL   0  1  0
SURFS   0  1  1
SECTS  96  1
FORMS   0  1
ATOMS   0  0  1
BONDS   1
POLYS   1
VECTS 1.000000
FORMP
  1  1.0   0   0   0
ATOMP
 24  24   0  50  2.0   0
BONDP
  1  16  0.250  2.000 127 127 127
POLYP
 204 1  1.000 180 180 180
ISURF
  0   0   0   0
TEX3P
  1 0.00000E+000 1.00000E+000
SECTP
  1 0.00000E+000 1.00000E+000 0.00000E+000
HKLPP
  92 0  1.000   0 128 255
UCOLP
   0   1  1.000   0   0   0
COMPS 1
LABEL 1    12  1.000 0
PROJT 0  0.962
BKGRC
 255 255 255
DPTHQ 1 -0.5000  3.5000
LIGHT0 1
 1.000000  0.000000  0.000000  0.000000
 0.000000  1.000000  0.000000  0.000000
 0.000000  0.000000  1.000000  0.000000
 0.000000  0.000000  0.000000  1.000000
 0.000000  0.000000 20.000000  0.000000
 0.000000  0.000000 -1.000000
  26  26  26 255
 179 179 179 255
 255 255 255 255
LIGHT1
 1.000000  0.000000  0.000000  0.000000
 0.000000  1.000000  0.000000  0.000000
 0.000000  0.000000  1.000000  0.000000
 0.000000  0.000000  0.000000  1.000000
 0.000000  0.000000 20.000000  0.000000
 0.000000  0.000000 -1.000000
   0   0   0   0
   0   0   0   0
   0   0   0   0
LIGHT2
 1.000000  0.000000  0.000000  0.000000
 0.000000  1.000000  0.000000  0.000000
 0.000000  0.000000  1.000000  0.000000
 0.000000  0.000000  0.000000  1.000000
 0.000000  0.000000 20.000000  0.000000
 0.000000  0.000000 -1.000000
   0   0   0   0
   0   0   0   0
   0   0   0   0
LIGHT3
 1.000000  0.000000  0.000000  0.000000
 0.000000  1.000000  0.000000  0.000000
 0.000000  0.000000  1.000000  0.000000
 0.000000  0.000000  0.000000  1.000000
 0.000000  0.000000 20.000000  0.000000
 0.000000  0.000000 -1.000000
   0   0   0   0
   0   0   0   0
   0   0   0   0
ATOMM
 204 204 204 255
  25.600
BONDM
 255 255 255 255
 128.000
POLYM
 255 255 255 255
 128.000
SURFM
   0   0   0 255
 128.000
FORMM
 255 255 255 255
 128.000
HKLPM
 255 255 255 255
 128.000"""


def write_vesta_with_arrows(
    filepath: str,
    lattice: NDArray[np.float64],
    scaled_positions: NDArray[np.float64],
    symbols: list[str],
    arrows_cartesian: NDArray[np.float64],
    title: str,
    arrow_rgb: tuple[int, int, int] = (255, 0, 0),
    arrow_radius: float = 0.5,
) -> None:
    """Write a complete VESTA file with per-atom displacement arrows.

    Arrow components are written in the VESTA vector convention: values along
    the a/b/c axis directions with the modulus in Angstroms (equal to Cartesian
    components for cubic cells).
    """
    a, b, c, alpha, beta, gamma = _lattice_parameters(lattice)
    axis_lengths = np.array([a, b, c], dtype=float)
    arrows_axis = (arrows_cartesian @ np.linalg.inv(lattice)) * axis_lengths

    lines: list[str] = []
    lines.append("#VESTA_FORMAT_VERSION 3.3.0")
    lines.append("")
    lines.append("")
    lines.append("CRYSTAL")
    lines.append("")
    lines.append("TITLE")
    lines.append(title)
    lines.append("")
    lines.append(_VESTA_HEADER_SECTIONS)
    lines.append("CELLP")
    lines.append(f" {a:10.6f} {b:10.6f} {c:10.6f} {alpha:10.6f} {beta:10.6f} {gamma:10.6f}")
    lines.append("  0.000000   0.000000   0.000000   0.000000   0.000000   0.000000")

    site_names: list[str] = []
    site_counter: dict[str, int] = {}
    for symbol in symbols:
        site_counter[symbol] = site_counter.get(symbol, 0) + 1
        site_names.append(f"{symbol}{site_counter[symbol]}")

    lines.append("STRUC")
    for index, (symbol, site_name, frac) in enumerate(zip(symbols, site_names, scaled_positions), start=1):
        lines.append(
            f"  {index} {symbol:9s} {site_name:4s} 1.0000 "
            f"{frac[0]:10.6f} {frac[1]:10.6f} {frac[2]:10.6f}    1a       1"
        )
        lines.append("                            0.000000   0.000000   0.000000  0.00")
    lines.append("  0 0 0 0 0 0 0")

    lines.append("THERI 0")
    for index, site_name in enumerate(site_names, start=1):
        lines.append(f"  {index} {site_name:>10s}  1.000000")
    lines.append("  0 0 0")

    lines.append("SHAPE")
    lines.append("  0       0       0       0   0.000000  0   192   192   192   192")
    lines.append("BOUND")
    lines.append("       0        1         0        1         0        1")
    lines.append("  0   0   0   0  0")
    lines.append("SBOND")
    lines.append("  0 0 0 0")

    lines.append("SITET")
    for index, (symbol, site_name) in enumerate(zip(symbols, site_names), start=1):
        radius, red, green, blue = VESTA_ELEMENTS.get(symbol, DEFAULT_ELEMENT)
        lines.append(
            f"  {index} {site_name:>10s}  {radius:.4f} {red:3d} {green:3d} {blue:3d} "
            f"{red:3d} {green:3d} {blue:3d} 204  0"
        )
    lines.append("  0 0 0 0 0 0")

    lines.append("VECTR")
    for index, arrow in enumerate(arrows_axis, start=1):
        lines.append(f"{index:5d}{arrow[0]:10.5f}{arrow[1]:10.5f}{arrow[2]:10.5f}")
        lines.append(f"{index:5d} 0 0 0 0")
        lines.append("  0 0 0 0 0")
    lines.append("0 0 0 0 0")

    lines.append("VECTT")
    red, green, blue = arrow_rgb
    for index in range(1, len(arrows_axis) + 1):
        lines.append(f"{index:5d}  {arrow_radius:.3f} {red:3d} {green:3d} {blue:3d} 0")
    lines.append("0 0 0 0 0")

    lines.append(_VESTA_TAIL_SECTIONS)

    lines.append("ATOMT")
    seen: list[str] = []
    for symbol in symbols:
        if symbol in seen:
            continue
        seen.append(symbol)
        radius, red, green, blue = VESTA_ELEMENTS.get(symbol, DEFAULT_ELEMENT)
        lines.append(
            f"  {len(seen)} {symbol:>10s}  {radius:.4f} {red:3d} {green:3d} {blue:3d} "
            f"{red:3d} {green:3d} {blue:3d} 204"
        )
    lines.append("  0 0 0 0 0 0")

    lines.append(_VESTA_STYLE_SECTIONS)
    lines.append("")

    with open(filepath, "w") as fp:
        fp.write("\n".join(lines))


def _irrep_filename_tag(labels: list[str]) -> str:
    """Compact irrep tag for file names: drop the dimension suffix "(n)" and
    join distinct labels with '+' (e.g. "GM1(1), GM5(2)" -> "GM1+GM5")."""
    cleaned: list[str] = []
    for label in labels:
        for part in label.split(","):
            part = re.sub(r"\(\d+\)", "", part).strip()
            if part and part != "-" and part not in cleaned:
                cleaned.append(part)
    return "+".join(cleaned)


def _output_path(
    output: str | None,
    formula: str,
    q_label: str,
    mode_file_string: str,
    irrep_tag: str,
    conventional: bool = False,
) -> str:
    if output is not None:
        path = Path(output)
        if path.suffix.lower() != ".vesta":
            path = path.with_suffix(path.suffix + ".vesta") if path.suffix else path.with_suffix(".vesta")
        return str(path)
    suffix = "_conv" if conventional else ""
    irrep_part = f"_{irrep_tag}" if irrep_tag else ""
    return f"POSCAR_{formula}_{q_label}_mode{mode_file_string}{irrep_part}{suffix}.vesta"


def _find_intertwiner(
    rep: NDArray[np.complex128],
    space_s: NDArray[np.complex128],
    space_r: NDArray[np.complex128],
) -> NDArray[np.complex128] | None:
    """Unitary aligning the irrep basis of space_s to that of space_r.

    Both spaces must carry equivalent irreps of the same (projective)
    representation ``rep``; returns None when they are inequivalent.
    """
    dim = space_s.shape[0]
    d_s = np.array([space_s @ G @ space_s.conj().T for G in rep])
    d_r = np.array([space_r @ G @ space_r.conj().T for G in rep])
    for seed_index in range(dim * dim):
        seed = np.zeros((dim, dim))
        seed[seed_index // dim, seed_index % dim] = 1.0
        averaged = sum(a @ seed @ b.conj().T for a, b in zip(d_s, d_r)) / len(rep)
        if np.linalg.norm(averaged) > 1e-6:
            u, _, vh = np.linalg.svd(averaged)
            return u @ vh
    return None


def build_symmetry_adapted_modes(
    phonon,
    qpoint: list[float],
    symprec: float = 1e-5,
) -> list[tuple[float, NDArray[np.complex128]]]:
    """Eigenvectors of the dynamical matrix at q, symmetry-adapted within degenerate subspaces.

    The dynamical matrix is block-diagonalized in the spgrep irrep-projected
    basis (the same construction as crystod-phonon --modulation), so that degenerate
    modes come out along symmetry-dictated directions instead of the arbitrary
    linear combinations returned by a plain eigensolver. Returns a list of
    (frequency_THz, mode_vector) sorted by frequency; the vectors are exact
    eigenvectors of the phonopy dynamical matrix at q. Raises RuntimeError when
    the construction cannot reproduce the phonopy spectrum.
    """
    primitive = phonon.primitive
    cell = PhonopyAtoms(
        numbers=primitive.numbers,
        scaled_positions=primitive.scaled_positions,
        cell=primitive.cell,
    )
    # standardize=False keeps phonopy's atom positions so that the projected
    # basis and the dynamical matrix share one phase convention.
    vibrations = SymmetryOnlyVibrations(cell=cell, symprec=symprec, standardize=False)
    irreps, vibration_rep, _ = vibrations.get_vibration_rep(qpoint)

    spaces: list[NDArray[np.complex128]] = []
    for irrep in irreps:
        spaces.extend(project_to_irrep(vibration_rep, irrep))

    dynamical_matrix = phonon.dynamical_matrix
    dynamical_matrix.run(qpoint)
    matrix = dynamical_matrix.dynamical_matrix.copy()

    dims = [space.shape[0] for space in spaces]
    if sum(dims) != matrix.shape[0]:
        raise RuntimeError("Irrep projection does not span the full vibration space.")

    offsets = np.cumsum([0] + dims)
    n_spaces = len(spaces)
    stacked = np.vstack(spaces)
    block_matrix = stacked @ matrix @ stacked.conj().T

    # Spaces carrying equivalent irreps may couple; group them into clusters.
    coupled = np.zeros((n_spaces, n_spaces), dtype=bool)
    for s in range(n_spaces):
        for t in range(n_spaces):
            sub = block_matrix[offsets[s] : offsets[s + 1], offsets[t] : offsets[t + 1]]
            coupled[s, t] = bool(np.abs(sub).max() > 1e-6)
    clusters: list[list[int]] = []
    seen: set[int] = set()
    for s in range(n_spaces):
        if s in seen:
            continue
        stack, cluster = [s], []
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            cluster.append(u)
            stack.extend(v for v in range(n_spaces) if coupled[u, v] and v not in seen)
        clusters.append(sorted(cluster))

    modes: list[tuple[float, NDArray[np.complex128]]] = []
    for cluster in clusters:
        dim = dims[cluster[0]]
        if any(dims[index] != dim for index in cluster):
            raise RuntimeError("Coupled irrep spaces with different dimensions.")
        multiplicity = len(cluster)

        aligned = [spaces[cluster[0]]]
        for index in cluster[1:]:
            intertwiner = _find_intertwiner(vibration_rep, spaces[index], spaces[cluster[0]])
            if intertwiner is None:
                raise RuntimeError("Coupled irrep spaces are not equivalent.")
            aligned.append(intertwiner.conj().T @ spaces[index])

        # After alignment every coupling block is a scalar multiple of the
        # identity (Schur), so the cluster reduces to one multiplicity-sized
        # Hermitian matrix shared by all irrep components.
        coupling = np.zeros((multiplicity, multiplicity), dtype=complex)
        for a in range(multiplicity):
            for b in range(multiplicity):
                sub = aligned[a] @ matrix @ aligned[b].conj().T
                if np.abs(sub - np.eye(dim) * np.trace(sub) / dim).max() > 1e-6:
                    raise RuntimeError("Coupling between irrep spaces is not scalar.")
                coupling[a, b] = np.trace(sub) / dim
        eigenvalues, eigenvectors = np.linalg.eigh(coupling)
        eigenvalues = eigenvalues.real

        # Preserve the symmetry-adapted basis when the cluster is numerically
        # degenerate, exactly as crystod-phonon --modulation does.
        if multiplicity > 1 and np.allclose(eigenvalues, eigenvalues.mean(), atol=1e-10, rtol=1e-8):
            eigenvalues = np.full(multiplicity, eigenvalues.mean())
            eigenvectors = np.eye(multiplicity, dtype=complex)

        for w in range(multiplicity):
            frequency = float(
                np.sign(eigenvalues[w]) * np.sqrt(abs(eigenvalues[w])) * FREQUENCY_CONVERSION_THZ
            )
            for component in range(dim):
                # Rows of the projected basis are bras; eigenvectors of the
                # dynamical matrix are their complex conjugates.
                vector = np.zeros(matrix.shape[0], dtype=complex)
                for a in range(multiplicity):
                    vector += eigenvectors[a, w] * np.conj(aligned[a][component])
                modes.append((frequency, vector))

    order = np.argsort([mode[0] for mode in modes], kind="stable")
    modes = [modes[index] for index in order]

    # Verify against the plain phonopy solution before trusting the result.
    reference = np.sort(np.linalg.eigvalsh(matrix).real)
    reference = np.sign(reference) * np.sqrt(np.abs(reference)) * FREQUENCY_CONVERSION_THZ
    if not np.allclose([m[0] for m in modes], reference, atol=1e-3):
        raise RuntimeError("Symmetry-adapted frequencies do not match the phonopy spectrum.")
    for frequency, vector in modes:
        eigenvalue = np.sign(frequency) * (frequency / FREQUENCY_CONVERSION_THZ) ** 2
        if np.linalg.norm(matrix @ vector - eigenvalue * vector) > 1e-6:
            raise RuntimeError("A symmetry-adapted mode is not an eigenvector of the dynamical matrix.")
    return modes


def get_supercell_displacement_field(
    phonon,
    qpoint: list[float],
    supercell_matrix: NDArray[np.int_],
    mode_vector: NDArray[np.complex128],
):
    """Complex displacement field of one mode on the commensurate supercell.

    supercell_matrix rows are the supercell lattice vectors in the primitive
    basis (L_super = S @ L_primitive). Follows phonopy's Modulation convention:
    u_j(l) = e_j exp(2 pi i q.(R_l + tau_j)) / sqrt(m_j), with the overall
    phase chosen to maximize the real part.
    Returns (supercell, displacements) where displacements has shape (n_atoms, 3).
    """
    primitive = phonon.primitive
    cell = PhonopyAtoms(
        numbers=primitive.numbers,
        scaled_positions=primitive.scaled_positions,
        cell=primitive.cell,
    )
    matrix = np.array(supercell_matrix, dtype=int)
    # phonopy's get_supercell builds L_super = M.T @ L_primitive.
    supercell = get_supercell(cell, matrix.T)
    s2uu = [supercell.u2u_map[x] for x in supercell.s2u_map]
    # positions in the primitive basis: x_primitive = x_super @ S
    coefs = np.exp(
        2j * np.pi * np.dot(np.dot(supercell.scaled_positions, matrix), qpoint)
    ) / np.sqrt(supercell.masses)
    u = np.array(
        [mode_vector[3 * s2uu[i] : 3 * s2uu[i] + 3] * coefs[i] for i in range(len(supercell.masses))]
    )
    # Global phase maximizing the real-part norm (phonopy MODULATION-style).
    sum_of_squares = np.sum(u * u)
    if abs(sum_of_squares) > 1e-12:
        u = u * np.exp(-0.5j * np.angle(sum_of_squares))
    return supercell, u


def _get_mode_labels(
    qpoint: list[float],
    phonon,
    dataset,
    degeneracy_tolerance: float,
) -> tuple[NDArray[np.float64], list[str]]:
    """Frequencies and per-mode irrep labels at q; labels fall back to '-' silently."""
    n_modes = 3 * len(phonon.primitive)
    try:
        irt_table = IrrepTable(dataset["number"], spinor=False)
        prim_mat = get_primitive_matrix_by_centring(dataset["international"][0])
        labels, band_indices, frequencies = get_irrep_labels(
            q=qpoint,
            phonon=phonon,
            irt_table=irt_table,
            prim_mat=prim_mat,
            degeneracy_tolerance=degeneracy_tolerance,
        )
        mode_labels = ["-"] * n_modes
        for label, indices in zip(labels, band_indices):
            text = ", ".join(label) if label else "-"
            for band_index in indices:
                mode_labels[band_index] = text
        return frequencies, mode_labels
    except Exception:
        phonon.run_qpoints([qpoint])
        frequencies = phonon.get_qpoints_dict()["frequencies"][0]
        return frequencies, ["-"] * n_modes


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    supercell_mat = [float(n) for n in args.dim.split()]
    if args.readfc:
        force_sets = None
        force_constants = "./FORCE_CONSTANTS"
    else:
        force_sets = "./FORCE_SETS"
        force_constants = None

    phonon = load(
        supercell_matrix=supercell_mat,
        primitive_matrix="auto",
        unitcell_filename=args.poscar,
        force_sets_filename=force_sets,
        force_constants_filename=force_constants,
    )

    dataset = get_symmetry_dataset(phonon.symmetry)
    prim_mat = get_primitive_matrix_by_centring(dataset["international"][0])
    try:
        irt_table = IrrepTable(dataset["number"], spinor=False)
        q_names, q_list = get_irt_special_points(irt_table, prim_mat)
    except Exception:
        q_names, q_list = [], []

    formula = reduced_formula(list(phonon.primitive.symbols))

    # the header + mode table is also saved as a text file (report_lines)
    report_lines: list[str] = []

    def echo(line: str = "") -> None:
        print(line)
        report_lines.append(line)

    if q_names:
        echo(f"Space group: {dataset['international']} (#{dataset['number']})")
        echo("Available high-symmetry q-points:")
        for name, q in zip(q_names, q_list):
            echo(f"  {name:8s} {q}")

    q_label, qpoint = resolve_qpoint(args.qpoint, q_names, q_list)
    echo(f"\nSelected q-point: {q_label} = {qpoint}")

    # Symmetry-adapted eigenvectors: degenerate modes are aligned along
    # symmetry-dictated directions (same construction as crystod-phonon --modulation)
    # instead of the arbitrary combinations a plain eigensolver returns.
    try:
        symmetry_adapted_modes = build_symmetry_adapted_modes(phonon, qpoint)
        frequencies = np.array([frequency for frequency, _ in symmetry_adapted_modes])
        mode_vectors = [vector for _, vector in symmetry_adapted_modes]
    except Exception as exc:
        warnings.warn(
            f"Symmetry-adapted mode construction failed ({exc}); falling back to plain "
            "phonopy eigenvectors, which may look tilted within degenerate subspaces.",
            stacklevel=2,
        )
        dynamical_matrix = phonon.dynamical_matrix
        dynamical_matrix.run(qpoint)
        eigenvalues, eigenvector_matrix = np.linalg.eigh(dynamical_matrix.dynamical_matrix)
        eigenvalues = eigenvalues.real
        frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * FREQUENCY_CONVERSION_THZ
        mode_vectors = [eigenvector_matrix[:, band] for band in range(len(frequencies))]

    _, mode_labels = _get_mode_labels(qpoint, phonon, dataset, args.tol)
    echo(f"\nPhonon modes at q = {q_label}")
    echo(f"{'Mode':>5s}  {'Freq (THz)':>12s}  Irrep")
    echo("-" * 40)
    for mode_index, frequency in enumerate(frequencies):
        echo(f"{mode_index + 1:5d}  {frequency:12.4f}  {mode_labels[mode_index]}")

    report_path = f"phonon_modes_{formula}_{q_label}.txt"
    with open(report_path, "w") as handle:
        handle.write("\n".join(report_lines) + "\n")
    print(f"\nMode table written to: {report_path}")

    n_modes = len(frequencies)
    if args.mode is None:
        if args.output:
            raise SystemExit("ERROR: --output requires --mode (one summed output file).")
        mode_groups = [[index] for index in range(n_modes)]
        print(f"\nNo --mode given; exporting all {n_modes} modes as individual VESTA files.")
    else:
        selected: list[int] = []
        for value in args.mode:
            if value < 1 or value > n_modes:
                raise SystemExit(
                    f"ERROR: mode number {value} is out of range [1, {n_modes}] (numbering is 1-based)."
                )
            selected.append(value - 1)
        mode_groups = [selected]

    if args.conventional:
        centring = dataset["international"][0]
        base_matrix = get_conventional_matrix(centring)
        print(f"\nConventional-cell output (centring {centring}); primitive-to-conventional matrix:")
        for row in base_matrix:
            print(f"  {row.tolist()}")
    else:
        base_matrix = np.eye(3, dtype=int)

    supercell_matrix = get_commensurate_supercell_matrix(qpoint, base_matrix)
    if not np.array_equal(supercell_matrix, base_matrix):
        multiples = np.rint(np.diag(supercell_matrix @ np.linalg.inv(base_matrix))).astype(int)
        print(
            f"\nCommensurate supercell for visualization: "
            f"{multiples[0]}x{multiples[1]}x{multiples[2]} "
            f"{'conventional' if args.conventional else 'primitive'} cells"
        )
    elif not args.conventional:
        print("\nCommensurate supercell for visualization: 1x1x1")

    # As in --modulation, multiple modes selected with --mode are summed into
    # one displacement pattern (each mode with unit weight). Without --mode,
    # every mode is exported individually.
    pad = len(str(n_modes))  # zero-pad mode numbers in file names so ls sorts them
    for group in mode_groups:
        supercell = None
        total_displacements = None
        for mode_index in group:
            supercell, complex_field = get_supercell_displacement_field(
                phonon=phonon,
                qpoint=qpoint,
                supercell_matrix=supercell_matrix,
                mode_vector=mode_vectors[mode_index],
            )
            displacements = complex_field.real
            max_norm = float(np.max(np.linalg.norm(displacements, axis=1)))
            if max_norm < 1e-12:
                print(f"Mode {mode_index + 1}: real part of the eigenvector vanished; contributes nothing.")
            print(
                f"  + mode {mode_index + 1}: {mode_labels[mode_index]}, "
                f"{frequencies[mode_index]:.4f} THz"
            )
            total_displacements = displacements if total_displacements is None else total_displacements + displacements

        max_norm = float(np.max(np.linalg.norm(total_displacements, axis=1)))
        if max_norm < 1e-12:
            print("The summed displacement pattern vanished; nothing to write.")
            continue
        arrows = total_displacements * (args.amplitude / max_norm)

        mode_numbers = [mode_index + 1 for mode_index in group]
        mode_string = "+".join(str(number) for number in mode_numbers)
        mode_file_string = "+".join(f"{number:0{pad}d}" for number in mode_numbers)
        unique_labels: list[str] = []
        for mode_index in group:
            if mode_labels[mode_index] not in unique_labels:
                unique_labels.append(mode_labels[mode_index])
        if len(group) == 1:
            title = (
                f"{formula} {q_label} mode {mode_string} "
                f"[{unique_labels[0]}] {frequencies[group[0]]:.4f} THz"
            )
        else:
            title = f"{formula} {q_label} modes {mode_string} [{', '.join(unique_labels)}]"
        if args.conventional:
            title += " (conventional cell)"

        output_path = _output_path(
            args.output,
            formula,
            q_label,
            mode_file_string,
            _irrep_filename_tag(unique_labels),
            args.conventional,
        )
        write_vesta_with_arrows(
            filepath=output_path,
            lattice=np.array(supercell.cell, dtype=float),
            scaled_positions=np.array(supercell.scaled_positions, dtype=float),
            symbols=list(supercell.symbols),
            arrows_cartesian=arrows,
            title=title,
        )
        if len(group) == 1:
            print(f"Mode {mode_string} written to: {output_path}")
        else:
            print(f"Sum of modes {mode_string} written to: {output_path}")

    print(
        f"\nArrows are scaled so the largest displacement is {args.amplitude:g} A; "
        "adjust arrow size in VESTA via Edit > Vectors or Properties > Vectors if needed."
    )


if __name__ == "__main__":
    main()
