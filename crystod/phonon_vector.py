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

from .irreptables_compat import load_irreptables
# _find_intertwiner is not used here; crystod.spin_basis imports it from this module
from .modulation import _find_intertwiner, _irrep_filename_tag  # noqa: F401
from .operations import parse_qpoint_token
from .phonon_irreps import (
    _special_points_in_label_frame,
    check_nac_loaded,
    find_star_representative,
    get_irrep_labels,
    get_irt_special_points,
    nac_load_options,
)
from .runtime_compat import get_qpoints_result, get_symmetry_dataset
from .symmetry_adapted_modes import solve_symmetry_adapted_modes
from .vasp_io import read_poscar_cell

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
        "--keep-q-coords",
        dest="keep_q_coords",
        action="store_true",
        help="Name output files of a non-special q with its coordinates "
        "(q_<coords>) instead of the ISO-IR k-vector-type label, so scans "
        "along one symmetry line do not overwrite each other.",
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
    parser.add_argument(
        "--nac",
        dest="nac",
        action="store_true",
        help="Apply the non-analytical term correction with ./BORN (without it\n"
        "a BORN file is not read).",
    )
    return parser


def _parse_coordinate(value: str) -> float:
    try:
        return parse_qpoint_token(value)
    except ValueError:
        return float(value)


def resolve_qpoint(
    raw_qpoint: list[str],
    q_names: list[str],
    q_list: list[list[float]],
    rotations: NDArray[np.int_] | None = None,
    isoir_context: tuple | None = None,
) -> tuple[str, list[float]]:
    """Resolve ``--qpoint`` tokens into a label and primitive-basis coordinates.

    This is how ``crystod-phonon --vector`` and ``--subgroup`` read their
    ``--qpoint`` argument: a single token is the label of a tabulated special
    point (``GM``, ``G``, ``GAMMA`` and the Greek capital gamma all mean
    Gamma); three tokens are coordinates in the primitive reciprocal basis,
    fractions such as ``1/3`` allowed. With ``isoir_context``, coordinates
    are labeled with their ISO-IR k-vector type in the frame of the labels
    (the star's name at a special point, ``PA`` at the -k partner of P,
    ``U``, ``B``, ``DT`` on a line or plane). Otherwise, or when the ISO-IR
    data give nothing, they are labeled with the name of the special point
    they coincide with; with ``rotations``, any arm of a tabulated star is
    labeled with the star's name, not only the tabulated arm; anything else
    is ``q_<coords>``. The commands pass the list in the frame of the labels
    (``phonon_irreps._special_points_in_label_frame``), so a label given on
    the command line names the point the labels call by it.

    Args:
        raw_qpoint: The tokens, one label or three coordinate strings.
        q_names: Labels of the tabulated special points.
        q_list: Their coordinates, as returned by
            :func:`crystod.phonon.get_irt_special_points`.
        rotations: Space-group rotations in the primitive basis, shape
            ``(n_ops, 3, 3)``, or ``None`` to label exact matches only.
        isoir_context: ``(space-group number, primitive cell tuple, symprec)``
            for the ISO-IR fallback label, optionally followed by the input
            cell tuple whose ISO-IR frame the label refers to, or ``None``
            for ``q_<coords>``.

    Returns:
        ``(label, qpoint)`` with ``qpoint`` a list of three floats. The
        coordinates are the ones that were given (not the tabulated arm), so
        the label names the star while the modes are computed at the
        requested q.

    Raises:
        ValueError: For an unknown label, or a token count other than one or
            three.

    Example:
        >>> from crystod import phonon
        >>> names = ["GM", "R", "X", "M"]
        >>> points = [[0, 0, 0], [0.5, 0.5, 0.5], [0, 0.5, 0], [0.5, 0.5, 0]]
        >>> phonon.resolve_qpoint(["R"], names, points)
        ('R', [0.5, 0.5, 0.5])
        >>> # rotations: the primitive-cell rotations of the space group
        >>> phonon.resolve_qpoint(["1/2", "0", "0"], names, points, rotations)
        ('X', [0.5, 0.0, 0.0])
    """
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
    if isoir_context is not None:
        # the ISO-IR k-vector type in the frame of the labels first
        from .isoir import get_isoir_kpoint_name

        sgnum, cell, symprec = isoir_context[:3]
        input_cell = isoir_context[3] if len(isoir_context) > 3 else None
        isoir_name = get_isoir_kpoint_name(
            sgnum, cell, symprec, qpoint, input_cell=input_cell
        )
        if isoir_name is not None:
            return isoir_name, qpoint
    for name, q in zip(q_names, q_list):
        if np.allclose(qpoint, q, atol=1e-8):
            return name, qpoint
    if rotations is not None:
        representative = find_star_representative(qpoint, rotations, q_names, q_list)
        if representative is not None:
            return representative[0], qpoint
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
    [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]).

    phonopy's primitive matrix M acts on COLUMN-vector lattices,
    (a_p, b_p, c_p) = (a_c, b_c, c_c) M, i.e. L_p = M^T L_c for the
    row-vector lattices used throughout CrystOD -- so the row matrix here
    is inv(M)^T, NOT inv(M).  The two coincide for the symmetric P/F/I
    matrices, but bare inv(M) drew a wrong (non-conventional) cell for R
    centring (an oblique det-3 supercell instead of the hexagonal cell)
    and axis-swapped cells for A/C."""
    primitive_matrix = np.array(get_primitive_matrix_by_centring(centring), dtype=float)
    conventional = np.linalg.inv(primitive_matrix).T
    conventional_int = np.rint(conventional).astype(int)
    if not np.allclose(conventional, conventional_int, atol=1e-8):
        raise ValueError(f"Non-integer primitive-to-conventional matrix for centring '{centring}'.")
    return conventional_int


def get_commensurate_supercell_matrix(
    qpoint: list[float],
    base_matrix: NDArray[np.int_],
) -> NDArray[np.int_]:
    """Smallest diagonal multiple of ``base_matrix`` commensurate with q.

    ``crystod-phonon --vector`` draws a mode on the smallest supercell over
    which its Bloch phase is periodic: each base-cell axis is multiplied by
    the denominator of the corresponding component of q in the base
    reciprocal basis (denominators up to 12).

    Args:
        qpoint: Fractional coordinates of q in the primitive reciprocal basis.
        base_matrix: Rows are the base-cell (primitive or conventional)
            lattice vectors in the primitive basis: the identity for the
            primitive cell, or the matrix of ``get_conventional_matrix`` for
            the conventional cell.

    Returns:
        Integer matrix ``S`` whose rows are the supercell lattice vectors in
        the primitive basis (``L_super = S @ L_primitive``), satisfying
        ``q . S_row in Z`` for every row.

    Example:
        >>> import numpy as np
        >>> from crystod import phonon
        >>> base = np.eye(3, dtype=int)
        >>> phonon.get_commensurate_supercell_matrix([0.0, 0.5, 0.0], base)
        array([[1, 0, 0],
               [0, 2, 0],
               [0, 0, 1]])
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

    The output of ``crystod-phonon --vector``: the structure is written as a
    P1 cell with one VECTR/VECTT arrow per atom, following the Phonopy_VESTA
    approach (A. P. Roy et al., Phys. Rev. Lett. 132, 026701 (2024)): a
    complete file including VESTA's style sections, since VESTA ignores
    vectors in files that lack them. Arrow components are written in the
    VESTA vector convention: values along the a/b/c axis directions with the
    modulus in Angstroms (equal to Cartesian components for cubic cells).

    Args:
        filepath: Output path (``.vesta``).
        lattice: Lattice vectors as rows, in Angstroms, shape ``(3, 3)``.
        scaled_positions: Fractional atomic coordinates, shape
            ``(n_atoms, 3)``.
        symbols: Chemical symbols of the atoms (radius and color are taken
            from VESTA's element defaults).
        arrows_cartesian: Cartesian arrow vectors in Angstroms, shape
            ``(n_atoms, 3)``.
        title: The VESTA title line.
        arrow_rgb: Arrow color as an RGB triple (default red).
        arrow_radius: Arrow radius in VESTA units.

    Returns:
        None. The file is written to ``filepath``.

    Example:
        Draw the first R-point mode of SrTiO3 on its 2x2x2 supercell (``ph``
        as in :func:`crystod.phonon.label_phonon_modes`)::

            import numpy as np
            from crystod import phonon
            from crystod.phonon_vector import get_supercell_displacement_field

            q = [0.5, 0.5, 0.5]
            modes = phonon.build_symmetry_adapted_modes(ph, q)
            supercell_matrix = phonon.get_commensurate_supercell_matrix(
                q, np.eye(3, dtype=int))
            supercell, field = get_supercell_displacement_field(
                ph, q, supercell_matrix, modes[0][1])
            arrows = field.real * (1.5 / np.linalg.norm(field.real, axis=1).max())
            phonon.write_vesta_with_arrows(
                "SrTiO3_R_mode1.vesta", np.array(supercell.cell),
                np.array(supercell.scaled_positions), list(supercell.symbols),
                arrows, title="SrTiO3 R mode 1")
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


def build_symmetry_adapted_modes(
    phonon,
    qpoint: list[float],
    symprec: float = 1e-5,
) -> list[tuple[float, NDArray[np.complex128]]]:
    """Symmetry-adapted eigenvectors of the dynamical matrix at q.

    The dynamical matrix is block-diagonalized in the spgrep irrep-projected
    basis of the primitive cell, so that the partners of a degenerate level
    transform with the irrep matrices instead of coming out as the arbitrary
    linear combinations a plain eigensolver returns -- the solver shared with
    ``crystod-phonon --modulation``
    (:func:`crystod.symmetry_adapted_modes.solve_symmetry_adapted_modes`).
    ``crystod-phonon --vector`` exports these vectors as VESTA arrows. The
    result is verified against the plain phonopy solution: the frequencies
    must match, every vector must be an eigenvector of the dynamical matrix,
    and the vectors must be orthonormal. At a time-reversal-invariant q
    (``2q`` a reciprocal lattice vector) the partners of a degenerate level
    are real displacement patterns (``e_j exp(2 pi i q.x_j)`` real).

    Args:
        phonon: A ``phonopy.Phonopy`` object with force constants, built with
            ``primitive_matrix="auto"``; its primitive cell is used as-is (no
            standardization), so that the projected basis and the dynamical
            matrix share one phase convention.
        qpoint: Fractional coordinates of q in the primitive reciprocal basis.
        symprec: Symmetry tolerance of the spglib/spgrep analysis.

    Returns:
        List of ``(frequency_THz, mode_vector)`` sorted by frequency (negative
        for imaginary modes); ``mode_vector`` has ``3 * n_atoms`` complex
        components in the mass-weighted phonopy convention (atom-position
        phase), as phonopy's own eigenvectors.

    Raises:
        ValueError: If the primitive cell of ``phonon`` is not primitive
            (:class:`crystod.symmetry_adapted_modes.NonPrimitiveCellError`).
        RuntimeError: If the construction cannot reproduce the phonopy
            spectrum (the projection does not span the vibration space, or
            the symmetry analysis does not match the dynamical matrix).

    Example:
        >>> from crystod import phonon
        >>> # ph: the SrTiO3 object of the label_phonon_modes example
        >>> modes = phonon.build_symmetry_adapted_modes(ph, [0.5, 0.5, 0.5])
        >>> [round(frequency, 4) for frequency, _ in modes[:4]]
        [-1.0867, -1.0867, -1.0867, 3.9891]
        >>> modes[0][1].shape
        (15,)
    """
    modes = solve_symmetry_adapted_modes(phonon, qpoint, symprec=symprec)
    return [
        (float(frequency), vector)
        for frequency, vector in zip(modes.frequencies, modes.eigenvectors)
    ]


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
        # The ISO-IR tables tabulate one arm per star; map q onto that arm so every
        # arm gets labels. The spectra of star arms are identical band by band,
        # so the labels at the representative apply to the modes at q.
        label_q = list(qpoint)
        q_names, q_list = get_irt_special_points(irt_table, prim_mat)
        rotations = get_symmetry_dataset(phonon.primitive_symmetry)["rotations"]
        representative = find_star_representative(qpoint, rotations, q_names, q_list)
        if representative is not None:
            label_q = representative[1]
        labels, band_indices, frequencies = get_irrep_labels(
            q=label_q,
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
        frequencies = get_qpoints_result(phonon).frequencies[0]
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
        unitcell=read_poscar_cell(args.poscar),
        force_sets_filename=force_sets,
        force_constants_filename=force_constants,
        **nac_load_options(args.nac),
    )
    nac_note = check_nac_loaded(phonon, args.nac, "BORN")
    if nac_note:
        print(nac_note)

    dataset = get_symmetry_dataset(phonon.symmetry)
    prim_mat = get_primitive_matrix_by_centring(dataset["international"][0])
    try:
        irt_table = IrrepTable(dataset["number"], spinor=False)
        q_names, q_list = _special_points_in_label_frame(phonon, irt_table, prim_mat)
    except Exception:
        q_names, q_list = [], []

    formula = reduced_formula(list(phonon.primitive.symbols))

    # the header + mode table is also saved as a text file (report_lines, in
    # its own layout); the terminal shows the same lines in titled blocks
    report_lines: list[str] = []
    # "written to" notices, printed together in the final block
    output_notices: list[str] = []

    if q_names:
        report_lines.append(f"Space group: {dataset['international']} (#{dataset['number']})")
        report_lines.append("Available high-symmetry q-points:")
        print("\n* Structure *")
        print(f"  {report_lines[0]}")
        print("\n* Available high-symmetry q-points *")
        for name, q in zip(q_names, q_list):
            report_lines.append(f"  {name:8s} {np.round(q, 6).tolist()}")
            print(report_lines[-1])

    try:
        primitive_rotations = get_symmetry_dataset(phonon.primitive_symmetry)["rotations"]
    except Exception:
        primitive_rotations = None
    if args.keep_q_coords:
        isoir_context = None
    else:
        try:
            primitive = phonon.primitive
            unitcell = phonon.unitcell
            isoir_context = (
                dataset["number"],
                (primitive.cell, primitive.scaled_positions, primitive.numbers),
                phonon.primitive_symmetry.tolerance,
                (unitcell.cell, unitcell.scaled_positions, unitcell.numbers),
            )
        except Exception:
            isoir_context = None
    q_label, qpoint = resolve_qpoint(
        args.qpoint, q_names, q_list,
        rotations=primitive_rotations, isoir_context=isoir_context,
    )
    report_lines.append(f"\nSelected q-point: {q_label} = {qpoint}")
    print("\n* Selected Q point *")
    print(f"  Selected q-point: {q_label} = {qpoint}")

    # Symmetry-adapted eigenvectors: the partners of degenerate modes transform
    # with the irrep matrices (same construction as crystod-phonon --modulation)
    # instead of being the arbitrary combinations a plain eigensolver returns.
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
    report_lines.append(f"\nPhonon modes at q = {q_label}")
    print(f"\n* Phonon modes at q = {q_label} *")
    table_start = len(report_lines)
    report_lines.append(f"{'Mode':>5s}  {'Freq (THz)':>12s}  Irrep")
    report_lines.append("-" * 40)
    for mode_index, frequency in enumerate(frequencies):
        report_lines.append(f"{mode_index + 1:5d}  {frequency:12.4f}  {mode_labels[mode_index]}")
    print("\n".join(report_lines[table_start:]))

    report_path = f"phonon_modes_{formula}_{q_label}.txt"
    with open(report_path, "w") as handle:
        handle.write("\n".join(report_lines) + "\n")
    output_notices.append(f"Mode table written to: {report_path}")

    n_modes = len(frequencies)
    if args.mode is None:
        if args.output:
            print("\n* Output files *")
            for notice in output_notices:
                print(f"  {notice}")
            raise SystemExit("ERROR: --output requires --mode (one summed output file).")
        mode_groups = [[index] for index in range(n_modes)]
        print("\n* Displacement patterns *")
        print(f"  No --mode given; exporting all {n_modes} modes as individual VESTA files.")
    else:
        selected: list[int] = []
        for value in args.mode:
            if value < 1 or value > n_modes:
                raise SystemExit(
                    f"ERROR: mode number {value} is out of range [1, {n_modes}] (numbering is 1-based)."
                )
            selected.append(value - 1)
        mode_groups = [selected]
        print("\n* Displacement patterns *")

    if args.conventional:
        centring = dataset["international"][0]
        base_matrix = get_conventional_matrix(centring)
        print(f"  Conventional-cell output (centring {centring}); primitive-to-conventional matrix:")
        for row in base_matrix:
            print(f"    {row.tolist()}")
    else:
        base_matrix = np.eye(3, dtype=int)

    supercell_matrix = get_commensurate_supercell_matrix(qpoint, base_matrix)
    if not np.array_equal(supercell_matrix, base_matrix):
        multiples = np.rint(np.diag(supercell_matrix @ np.linalg.inv(base_matrix))).astype(int)
        print(
            f"  Commensurate supercell for visualization: "
            f"{multiples[0]}x{multiples[1]}x{multiples[2]} "
            f"{'conventional' if args.conventional else 'primitive'} cells"
        )
    elif not args.conventional:
        print("  Commensurate supercell for visualization: 1x1x1")

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
                print(f"  Mode {mode_index + 1}: real part of the eigenvector vanished; contributes nothing.")
            print(
                f"  + mode {mode_index + 1}: {mode_labels[mode_index]}, "
                f"{frequencies[mode_index]:.4f} THz"
            )
            total_displacements = displacements if total_displacements is None else total_displacements + displacements

        max_norm = float(np.max(np.linalg.norm(total_displacements, axis=1)))
        if max_norm < 1e-12:
            print("  The summed displacement pattern vanished; nothing to write.")
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
            output_notices.append(f"Mode {mode_string} written to: {output_path}")
        else:
            output_notices.append(f"Sum of modes {mode_string} written to: {output_path}")

    print(
        f"  Arrows are scaled so the largest displacement is {args.amplitude:g} A; "
        "adjust arrow size in VESTA via Edit > Vectors or Properties > Vectors if needed."
    )
    print("\n* Output files *")
    for notice in output_notices:
        print(f"  {notice}")


if __name__ == "__main__":
    main()
