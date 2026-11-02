"""Surface slabs cut from a bulk crystal structure (``crystod-slab``).

For a Miller index ``(h k l)`` pymatgen's
:class:`~pymatgen.core.surface.SlabGenerator` builds the oriented unit cell
of the bulk, whose ``a`` and ``b`` vectors span the ``(h k l)`` plane, stacks
whole oriented cells along ``c`` until their height reaches ``slab_size`` (the
atoms then span that height less the gap at the cut) and adds at least
``vacuum_size`` of vacuum, likewise in whole cells, and by default centers
the slab in the cell.  Every possible termination is a shift of the cut along
``c``; ``SlabGenerator.get_slabs`` keeps one slab per distinct termination
(terminations that give the same slab are merged).  With ``max_index`` the
indices are the ones with ``|h|, |k|, |l| <= max_index`` that are distinct
under the Laue group of the crystal (its point group and the inversion that
turns a slab over), found in the axes of the cell as given.

Each slab is described by its number of atoms, its thickness (the extent of
the atoms along the surface normal), the vacuum (the cell height along the
normal minus the thickness), the in-plane area and its formula, and by two
yes/no properties, both as pymatgen defines them:

- *polar*: the slab decorated with oxidation states has a dipole per area
  along the normal above 1e-3 e/A (``Slab.is_polar``).  The states are
  guessed from the bulk composition (``Composition.oxi_state_guesses``) or
  given; with no states (a metal, an unusual stoichiometry) the polarity is
  unknown (``None``).
- *symmetric*: an operation of the slab's point group turns the surface
  normal over -- inversion, a mirror or glide plane parallel to the surface,
  a two-fold rotation or screw axis lying in it -- so the two faces are
  equivalent (``Slab.is_symmetric``).

``symmetrize=True`` asks for symmetric slabs instead (pymatgen's
``get_slabs(symmetrize=True)``, with the faces compared at the symmetry
tolerance of the run): atoms are removed from one face of every asymmetric
slab until the two faces are equivalent, which changes the stoichiometry and
can leave the slab with a net formal charge (:attr:`SlabRecord.charge`).

The slabs are written as VASP POSCAR files (direct coordinates, one block
per element in the order of the bulk POSCAR) named
``POSCAR_slab_{formula}_{h}{k}{l}_t{n}`` (``_s{n}`` for symmetrized slabs).

Originally ``script/slabkit.py`` (R. Kiribayashi, 2024); the paper to cite
for ``crystod-slab`` is R. Kiribayashi et al., ACS Appl. Mater. Interfaces
17, 61872 (2025).  The slab construction is pymatgen's, as there; the polar
check is now pymatgen's dipole test and every termination is reported.
"""

from __future__ import annotations

import itertools
import math
import os
import re
import textwrap
import warnings
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral, Real
from pathlib import Path

import numpy as np

#: Minimum slab height in Angstrom (pymatgen ``min_slab_size``).
DEFAULT_SLAB_SIZE = 10.0
#: Minimum vacuum height in Angstrom (pymatgen ``min_vacuum_size``).
DEFAULT_VACUUM_SIZE = 15.0
#: Symmetry tolerance in Angstrom of the space group, the Laue group of the
#: Miller indices and the symmetric-slab test.
DEFAULT_TOLERANCE = 0.1
#: Columns of the terminal output.
OUTPUT_WIDTH = 100

# get_slabs(tol=..., ftol=...): pymatgen's own defaults, fixed here (tol is a
# fractional tolerance of the primitive-cell search and the slab matching,
# ftol the clustering distance in Angstrom of the atomic planes)
_SLAB_TOL = 0.1
_SLAB_FTOL = 0.1
# a net charge (e per cell) or a bulk charge (e per formula unit) below this is zero
_CHARGE_TOL = 1e-6

#: The papers to cite for ``crystod-slab`` (that of the author of the
#: original slabkit.py) and for pymatgen's slab construction.
SLAB_CITATION_LINES = (
    "If you use crystod-slab in your research, please cite:",
    "  R. Kiribayashi, K. Sunada, T. Isobe, K. Kobayashi, T. Nagai, H. Ishiguro, Y. Mochizuki,",
    "  and A. Nakajima, \"Preparation and Antiviral Activity of La2CuO4 and Y2Cu2O5 with",
    "  Mechanistic Insights from First Principles\", ACS Appl. Mater. Interfaces 17, 61872-61887",
    "  (2025). https://doi.org/10.1021/acsami.5c17978",
    "The slabs are built with pymatgen's SlabGenerator:",
    "  W. Sun and G. Ceder, Surf. Sci. 617, 53 (2013); R. Tran et al., Sci. Data 3, 160080 (2016).",
)
SLAB_CITATION = "\n".join(SLAB_CITATION_LINES)

#: What the columns of :func:`format_table` mean (printed under the table).
TABLE_LEGEND = (
    "thickness: extent of the atoms along the surface normal (whole oriented cells are stacked "
    "to the minimum slab height, and the atoms span them less the gap at the cut); vacuum: cell "
    "height along the normal minus the thickness (both in A); area: in-plane cell area (A^2); "
    "shift: pymatgen's termination shift (fractional c of the oriented cell); polar: dipole per "
    "area from the oxidation states (Slab.is_polar; ? = no states); symmetric: an operation of "
    "the slab turns the normal over (e.g. inversion, a mirror or glide plane parallel to the "
    "surface, a 2-fold axis in it), so the two faces are equivalent (Slab.is_symmetric)."
)
#: Appended to :data:`TABLE_LEGEND` when the table has a charge column.
CHARGE_LEGEND = "charge: net formal charge per cell (e) from the oxidation states."

# point group -> Laue group (the point group with the inversion added)
_LAUE_GROUPS = {
    "1": "-1", "-1": "-1", "2": "2/m", "m": "2/m", "2/m": "2/m",
    "222": "mmm", "mm2": "mmm", "mmm": "mmm",
    "4": "4/m", "-4": "4/m", "4/m": "4/m",
    "422": "4/mmm", "4mm": "4/mmm", "-42m": "4/mmm", "-4m2": "4/mmm", "4/mmm": "4/mmm",
    "3": "-3", "-3": "-3",
    "32": "-3m", "321": "-3m", "312": "-3m", "3m": "-3m", "3m1": "-3m", "31m": "-3m",
    "-3m": "-3m", "-3m1": "-3m", "-31m": "-3m",
    "6": "6/m", "-6": "6/m", "6/m": "6/m",
    "622": "6/mmm", "6mm": "6/mmm", "-6m2": "6/mmm", "-62m": "6/mmm", "6/mmm": "6/mmm",
    "23": "m-3", "m-3": "m-3", "432": "m-3m", "-43m": "m-3m", "m-3m": "m-3m",
}


def _fail(message: str):
    raise SystemExit(f"ERROR: {message}")


def _one_line(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    return text.removeprefix("ERROR: ") or type(exc).__name__


def _check_tolerance(tolerance) -> float:
    # spglib crashes the interpreter on a negative or NaN symprec
    if not (isinstance(tolerance, Real) and math.isfinite(tolerance) and tolerance > 0):
        _fail(f"the symmetry tolerance must be a positive number of Angstrom (got {tolerance}).")
    return float(tolerance)


def _check_size(value, what: str) -> float:
    if not (isinstance(value, Real) and math.isfinite(value) and value > 0):
        _fail(f"the minimum {what} must be positive (got {value}).")
    return float(value)


def _check_file(path: str, what: str) -> None:
    if os.path.isdir(path):
        _fail(f"{path} is a directory, not {'an' if what[0] in 'aeiou' else 'a'} {what}.")
    if not os.path.isfile(path):
        _fail(f"no {what} {path}.")


# ------------------------------------------------------------------ Miller indices
def format_miller_index(index) -> str:
    """``"(1 0 -1)"``: an index in parentheses, the components separated by spaces."""
    return "(" + " ".join(str(int(i)) for i in index) + ")"


def _index_tokens(index) -> list[str]:
    if isinstance(index, str):
        text = index
    else:
        try:
            parts = list(index)
        except TypeError:
            parts = [index]
        text = " ".join(str(part) for part in parts)
    text = text.strip().strip("()[]")
    return [token for token in re.split(r"[\s,]+", text) if token]


def parse_miller_index(index) -> tuple[int, int, int]:
    """A Miller index ``(h, k, l)`` from three integers or a Miller-Bravais index.

    Args:
        index: ``"1 0 0"``, ``"1,0,0"``, ``["1", "0", "0"]``, ``(1, 0, 0)``,
            or four integers ``(h k i l)`` of hexagonal axes, of which the
            third must be ``i = -(h + k)`` and is dropped:
            ``["1", "0", "-1", "0"]`` -> ``(1, 0, 0)``.

    Returns:
        ``(h, k, l)`` as given (not divided by a common factor).

    Raises:
        SystemExit: Not three or four integers, a four-index input with
            ``i != -(h + k)``, or the zero vector (``ValueError`` through
            :mod:`crystod.slab`).

    Example:
        >>> from crystod import slab
        >>> slab.parse_miller_index(["1", "0", "-1", "0"])
        (1, 0, 0)
    """
    tokens = _index_tokens(index)
    shown = " ".join(tokens) or "nothing"
    values = []
    for token in tokens:
        if not re.fullmatch(r"[+-]?\d+", token):
            _fail(f"Miller indices are integers; got '{token}' in '{shown}'.")
        values.append(int(token))
    if len(values) not in (3, 4):
        _fail(f"a Miller index is h k l, or h k i l on hexagonal axes; got '{shown}'.")
    if len(values) == 4:
        h, k, i, l = values
        if i != -(h + k):
            _fail(f"({shown}) is not a Miller-Bravais index (h k i l): i = {i}, "
                  f"but -(h + k) = {-(h + k)}.")
        values = [h, k, l]
    if not any(values):
        _fail(f"({shown}) is not a plane: give at least one nonzero index.")
    return tuple(values)


def _read_index_entries(path) -> list[tuple[tuple[int, int, int], list[str]]]:
    """The indices of a list file with the tokens each line gave (four for ``h k i l``)."""
    path = os.fspath(path)
    _check_file(path, "index list file")
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        _fail(f"cannot read the index list file {path}: {_one_line(exc)}.")
    entries = []
    for number, line in enumerate(lines, start=1):
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        try:
            entries.append((parse_miller_index(text), _index_tokens(text)))
        except SystemExit as exc:
            _fail(f"{path}, line {number}: {_one_line(exc)}")
    if not entries:
        _fail(f"{path} holds no Miller index (one per line: 1 0 0, 1,0,0 or 1 0 -1 0).")
    return entries


def read_index_list(path) -> list[tuple[int, int, int]]:
    """The Miller indices of a list file, one per line.

    A line holds ``1 0 0``, ``1,0,0`` or a four-index ``1 0 -1 0``; ``#``
    starts a comment, blank lines are skipped.

    Raises:
        SystemExit: The file is missing or unreadable, a line is not a
            Miller index (the message names the line number), or the file
            holds no index (``ValueError`` through :mod:`crystod.slab`).
    """
    return [index for index, _ in _read_index_entries(path)]


def _check_max_index(max_index) -> int:
    if isinstance(max_index, bool) or not isinstance(max_index, Integral) or max_index < 1:
        _fail(f"the maximum Miller index must be a positive integer (got {max_index}).")
    return int(max_index)


def _rotations(structure, tolerance: float) -> list[np.ndarray]:
    """The rotations of the space group of ``structure`` in the fractional
    coordinates of its own cell (the identity alone when spglib finds none)."""
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            operations = SpacegroupAnalyzer(
                structure, symprec=tolerance).get_symmetry_operations(cartesian=False)
        except Exception:  # spglib found no symmetry at this tolerance
            operations = []
    unique = {tuple(np.rint(op.rotation_matrix).astype(int).ravel()) for op in operations}
    unique.add((1, 0, 0, 0, 1, 0, 0, 0, 1))
    return [np.array(rotation, dtype=int).reshape(3, 3) for rotation in sorted(unique)]


def _miller_orbit(index, rotations) -> set[tuple[int, int, int]]:
    """The indices equivalent to ``index``: ``h R`` for every rotation ``R``
    (a plane ``h . x = n`` goes to ``(h R^-1) . x' = n'``, and the group holds
    ``R^-1`` with ``R``) and their negatives (a slab has both faces)."""
    h = np.array(index, dtype=int)
    orbit = set()
    for rotation in rotations:
        image = tuple(int(i) for i in h @ rotation)
        orbit.add(image)
        orbit.add(tuple(-i for i in image))
    return orbit


def distinct_miller_indices(structure, max_index: int,
                            tolerance: float = DEFAULT_TOLERANCE) -> list[tuple[int, int, int]]:
    """The Miller indices with ``|h|, |k|, |l| <= max_index`` that are distinct
    under the Laue group of the crystal.

    Two indices are the same surface when a rotation of the space group of
    ``structure`` (found by spglib at ``tolerance``), possibly followed by the
    inversion that turns a slab over, maps one onto the other.  This is done
    in the axes of the cell as given, so it holds for every setting (a
    primitive rhombohedral cell, a monoclinic cell of pseudo-orthorhombic
    metric, ...).  The indices are reduced by their common factor, and the
    first index of each family is kept in pymatgen's order
    (``get_symmetrically_distinct_miller_indices``: low indices first), with
    whose list this one agrees for a crystal that is not trigonal and whose
    lattice has no more symmetry than the crystal.  pymatgen merges indices
    under the symmetry of the lattice (and, for a trigonal crystal, of its
    hexagonal setting), which on a primitive rhombohedral cell lists one
    surface twice and misses another, and on a lattice of higher symmetry
    than the crystal misses surfaces.

    Example:
        >>> from crystod import slab
        >>> from crystod.examples import example_path
        >>> s = slab.load_structure(example_path("221_PPOSCAR_SrTiO3"))
        >>> slab.distinct_miller_indices(s, 1)
        [(1, 1, 1), (1, 1, 0), (1, 0, 0)]
    """
    max_index = _check_max_index(max_index)
    rotations = _rotations(structure, _check_tolerance(tolerance))
    values = list(range(-max_index, max_index + 1))[::-1]
    candidates = sorted((index for index in itertools.product(values, values, values)
                         if any(index)), key=lambda index: max(abs(i) for i in index))
    seen: set[tuple[int, int, int]] = set()
    distinct = []
    for index in candidates:
        index = _reduced(index)
        if index in seen:
            continue
        seen |= _miller_orbit(index, rotations)
        distinct.append(index)
    return distinct


def _turns_over(index, rotations) -> bool:
    """Some rotation of the bulk maps the normal ``index`` onto its reverse."""
    h = np.array(index, dtype=int)
    return any(np.array_equal(h @ rotation, -h) for rotation in rotations)


# ----------------------------------------------------------------- the structure
def _composition_formula(structure, order=None) -> str:
    """The reduced formula with the elements in ``order`` (default: the order of
    the sites), without pymatgen's grouping: ``YBa2Cu3O6``, not ``Ba2Y(CuO2)3``."""
    counts = Counter(site.specie.symbol for site in structure)
    divisor = math.gcd(*counts.values())
    reduced = Counter({symbol: n // divisor for symbol, n in counts.items()})
    order = tuple(order or dict.fromkeys(site.specie.symbol for site in structure))
    return _formula(reduced, order)


def load_structure(path, conventional: bool = False, tolerance: float = DEFAULT_TOLERANCE):
    """Read the bulk structure (POSCAR; CIF and other pymatgen formats work too).

    The structure is read by pymatgen in VASP's reading of a POSCAR
    (:func:`crystod.vasp_io.poscar_structure`), stripped of site properties
    and oxidation states, and checked to be ordered.

    Args:
        path: The structure file.
        conventional: Return the standardized conventional cell
            (``SpacegroupAnalyzer.get_conventional_standard_structure``) instead
            of the cell as given; Miller indices then refer to its axes.  The
            sites keep the element order of the file.
        tolerance: Symmetry tolerance in Angstrom of the standardization.

    Returns:
        A ``pymatgen.core.Structure``.

    Raises:
        SystemExit: The file is missing or unreadable, or a site is partially
            occupied or shared by several species (``ValueError`` through
            :mod:`crystod.slab`).
    """
    from pymatgen.core import Structure
    from pymatgen.io.vasp.inputs import Poscar

    from .vasp_io import poscar_structure

    path = os.fspath(path)
    if not os.path.exists(path):
        _fail(f"no structure file {path} (give it with -c FILE).")
    _check_file(path, "structure file")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            structure = Structure.from_file(path)
        except Exception:  # an unrecognized file name: try it as a POSCAR
            try:
                structure = Poscar.from_file(path, read_velocities=False).structure
            except Exception as exc:  # pymatgen raises several types for a bad file
                _fail(f"cannot read {path} as a POSCAR or CIF: {_one_line(exc)}.")
    structure = poscar_structure(structure, path)
    disordered = [site for site in structure if not site.is_ordered]
    if disordered:
        first = disordered[0]
        where = ", ".join(f"{v:.4g}" for v in first.frac_coords)
        more = f" (1 of {len(disordered)})" if len(disordered) > 1 else ""
        _fail(f"{path}: disordered site {first.species_string} at ({where}){more}; "
              "slabs need an ordered structure.")
    symbols = [site.specie.symbol for site in structure]
    plain = Structure(structure.lattice, symbols, structure.frac_coords)
    if conventional:
        plain = conventional_cell(plain, tolerance)
    return plain


def conventional_cell(structure, tolerance: float = DEFAULT_TOLERANCE, species_order=None):
    """The standardized conventional cell of ``structure`` (``--conventional``).

    pymatgen's ``SpacegroupAnalyzer.get_conventional_standard_structure`` at
    ``tolerance``; the sites are sorted (stably) by element in
    ``species_order``, by default the element order of ``structure``.

    Raises:
        SystemExit: No conventional cell at this tolerance (``ValueError``
            through :mod:`crystod.slab`).
    """
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    tolerance = _check_tolerance(tolerance)
    order = list(species_order or dict.fromkeys(site.specie.symbol for site in structure))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            standard = SpacegroupAnalyzer(
                structure, symprec=tolerance).get_conventional_standard_structure()
        except Exception as exc:  # spglib found no symmetry at this tolerance
            _fail(f"the conventional standard cell could not be found at tolerance "
                  f"{tolerance:g} A ({_one_line(exc)}).")
    rank = {symbol: n for n, symbol in enumerate(order)}
    sites = sorted(standard, key=lambda site: rank.get(site.specie.symbol, len(rank)))
    return Structure(standard.lattice, [site.specie.symbol for site in sites],
                     [site.frac_coords for site in sites])


@dataclass(frozen=True)
class CellInfo:
    """Symmetry facts of a bulk cell, as ``crystod-slab`` prints them.

    Attributes:
        formula: Reduced formula, elements in the order of the sites
            (``SrTiO3``; ``YBa2Cu3O6``, never pymatgen's ``Ba2Y(CuO2)3``).
        space_group: International symbol found by spglib (``Pm-3m``).
        space_group_number: Its number (221).
        n_atoms: Atoms in the cell.
        n_conventional: Atoms in the standardized conventional cell.
        is_conventional: The cell has the lattice parameters and the number
            of atoms of the conventional standard cell, so its Miller indices
            are those of that cell.
        species: The elements in the order of their first site.
        lattice: ``(a, b, c, alpha, beta, gamma)`` in Angstrom and degrees.
        point_group: The crystal point group (``m-3m``).
        laue_group: Its Laue group, under which :func:`distinct_miller_indices`
            merges indices (``m-3m``; ``-3m`` for ``R-3c``).
    """

    formula: str
    space_group: str
    space_group_number: int
    n_atoms: int
    n_conventional: int
    is_conventional: bool
    species: tuple[str, ...]
    lattice: tuple[float, float, float, float, float, float]
    point_group: str = "1"
    laue_group: str = "-1"


def describe_cell(structure, tolerance: float = DEFAULT_TOLERANCE) -> CellInfo:
    """Space group of ``structure`` and how it compares with the conventional cell."""
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    tolerance = _check_tolerance(tolerance)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            analyzer = SpacegroupAnalyzer(structure, symprec=tolerance)
            symbol = analyzer.get_space_group_symbol()
            number = int(analyzer.get_space_group_number())
            point_group = analyzer.get_point_group_symbol()
            standard = analyzer.get_conventional_standard_structure()
        except Exception:  # spglib found no symmetry at this tolerance
            symbol, number, point_group, standard = "P1", 1, "1", structure
    lattice = structure.lattice
    same = (len(standard) == len(structure)
            and np.allclose(lattice.abc, standard.lattice.abc, atol=max(tolerance, 1e-4))
            and np.allclose(lattice.angles, standard.lattice.angles, atol=0.5))
    return CellInfo(
        formula=_composition_formula(structure),
        space_group=symbol,
        space_group_number=number,
        n_atoms=len(structure),
        n_conventional=len(standard),
        is_conventional=bool(same),
        species=tuple(dict.fromkeys(site.specie.symbol for site in structure)),
        lattice=(*(float(x) for x in lattice.abc), *(float(x) for x in lattice.angles)),
        point_group=point_group,
        laue_group=_LAUE_GROUPS.get(point_group, "?"),
    )


# -------------------------------------------------------------- oxidation states
def guess_oxidation_states(structure) -> dict[str, float] | None:
    """pymatgen's most likely oxidation states of the bulk composition.

    ``None`` when pymatgen has no charge-balanced guess, or only the all-zero
    one of a metal: the dipole, hence the polarity, is then unknown.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            guesses = structure.composition.oxi_state_guesses()
        except Exception:  # an element without oxidation-state data
            guesses = ()
    if not guesses:
        return None
    states = {str(element): float(value) for element, value in guesses[0].items()}
    if not any(states.values()):
        return None
    return states


def parse_oxidation_states(tokens) -> dict[str, float]:
    """``{"Sr": 2.0, "O": -2.0}`` from ``["Sr=2", "O=-2"]`` (or ``"Sr=2 O=-2"``).

    A state may be a fraction, ``Fe=8/3``.

    Raises:
        SystemExit: A token that is not ``EL=Q`` with a number ``Q``, an
            unknown element, or an element given twice (``ValueError``
            through :mod:`crystod.slab`).
    """
    from pymatgen.core.periodic_table import Element

    if isinstance(tokens, str):
        tokens = tokens.split()
    states: dict[str, float] = {}
    for token in tokens:
        symbol, equals, value = str(token).partition("=")
        symbol = symbol.strip()
        try:
            charge = float(Fraction(value.strip()))
        except (ValueError, ZeroDivisionError):
            charge = math.nan
        if not equals or not symbol or not math.isfinite(charge):
            _fail(f"oxidation states (--oxidation) are EL=Q pairs such as Sr=2 Ti=4 O=-2; "
                  f"got '{token}'.")
        if not Element.is_valid_symbol(symbol):
            _fail(f"oxidation states (--oxidation): '{symbol}' is not an element.")
        if symbol in states:
            _fail(f"oxidation states (--oxidation): {symbol} is given twice.")
        states[symbol] = charge
    if not states:
        _fail("oxidation states (--oxidation) are EL=Q pairs such as Sr=2 Ti=4 O=-2.")
    return states


def _check_oxidation_states(states: dict, structure) -> dict[str, float] | None:
    """The given states, one per element of ``structure``, in its element order;
    ``None`` when every state is 0 (no ions, so no dipole to test)."""
    elements = list(dict.fromkeys(site.specie.symbol for site in structure))
    given = {str(symbol): value for symbol, value in dict(states).items()}
    for symbol, value in given.items():
        if not (isinstance(value, Real) and math.isfinite(value)):
            _fail(f"the oxidation state of {symbol} must be a number (got {value}).")
    foreign = [symbol for symbol in given if symbol not in elements]
    if foreign:
        _fail(f"oxidation states (--oxidation) for {', '.join(foreign)}, which the structure "
              f"lacks (it has {', '.join(elements)}).")
    missing = [symbol for symbol in elements if symbol not in given]
    if missing:
        _fail(f"no oxidation state for {', '.join(missing)} (--oxidation needs one per element: "
              f"{' '.join(f'{e}=Q' for e in elements)}).")
    ordered = {symbol: float(given[symbol]) for symbol in elements}
    counts = Counter(site.specie.symbol for site in structure)
    units = math.gcd(*counts.values())
    charge = sum(ordered[symbol] * n for symbol, n in counts.items()) / units
    if abs(charge) > _CHARGE_TOL:
        _fail(f"oxidation states {format_oxidation_states(ordered)} (--oxidation) leave "
              f"{_composition_formula(structure)} charged ({format_charge(charge)} e per formula "
              "unit).")
    if not any(ordered.values()):
        return None
    return ordered


def resolve_oxidation_states(structure, oxidation_states=None):
    """The states used for the dipole and where they came from.

    Returns:
        ``(states, source)``: the given states (checked against the
        structure: one per element, balancing the bulk) and ``"given"``,
        pymatgen's guess and ``"guessed"``, or ``(None, None)`` when there is
        no guess.  Given states that are all 0 come back as ``(None,
        "given")``: no ions, so the polarity is unknown.

    Raises:
        SystemExit: Given states that miss an element, name one the
            structure does not have, or leave the bulk charged
            (``ValueError`` through :mod:`crystod.slab`).
    """
    if oxidation_states is not None:
        return _check_oxidation_states(oxidation_states, structure), "given"
    states = guess_oxidation_states(structure)
    return (states, "guessed") if states else (None, None)


def format_charge(value: float) -> str:
    """``"+4"``, ``"-1"``, ``"0"``, ``"+2.67"``: a charge in units of e."""
    if abs(value) <= _CHARGE_TOL:
        return "0"
    if abs(value - round(value)) < 1e-6:
        return f"{int(round(value)):+d}"
    return f"{value:+.3g}"


def format_oxidation_states(states: dict[str, float] | None) -> str:
    """``"Sr+2 Ti+4 O-2"`` (``"none"`` for no states)."""
    if not states:
        return "none"
    return " ".join(f"{symbol}{format_charge(value)}" if value else f"{symbol}0"
                    for symbol, value in states.items())


# --------------------------------------------------------------------- the slabs
@dataclass
class SlabRecord:
    """One slab: a termination of one Miller index, with its measurements.

    Attributes:
        miller_index: ``(h, k, l)`` of the surface (pymatgen's, divided by
            any common factor).
        termination: 1, 2, ... in pymatgen's order within the index.
        shift: pymatgen's termination shift, fractional ``c`` of the
            oriented unit cell.
        slab: The ``pymatgen.core.surface.Slab``.
        n_atoms: Atoms in the slab cell.
        thickness: Extent of the atoms along the surface normal (Angstrom).
        vacuum: Height of the cell along the normal minus the thickness.
        area: In-plane area of the cell (Angstrom^2).
        formula: Composition of the slab cell (``Sr5Ti5O15``), elements in
            the order of ``species_order``.
        polar: ``Slab.is_polar`` of the slab decorated with the oxidation
            states; ``None`` when there are none.
        symmetric: ``Slab.is_symmetric`` at the symmetry tolerance.
        bulk_formula: Reduced formula of the bulk, elements in the order of
            ``species_order`` and never grouped (for the file names).
        species_order: Element order of the bulk structure (for writing).
        charge: Net formal charge of the slab cell (e) from the oxidation
            states, ``None`` without them; nonzero only for a
            non-stoichiometric (symmetrized) slab.
        symmetrized: The slab comes from a ``symmetrize=True`` run (its
            label and file name use ``s1, s2, ...`` instead of ``t1, t2``).
    """

    miller_index: tuple[int, int, int]
    termination: int
    shift: float
    slab: object
    n_atoms: int
    thickness: float
    vacuum: float
    area: float
    formula: str
    polar: bool | None
    symmetric: bool
    bulk_formula: str
    species_order: tuple[str, ...]
    charge: float | None = None
    symmetrized: bool = False

    @property
    def label(self) -> str:
        """``"(1 1 1) t1"`` (``"(1 0 0) s1"`` for a symmetrized slab)."""
        kind = "s" if self.symmetrized else "t"
        return f"{format_miller_index(self.miller_index)} {kind}{self.termination}"

    @property
    def charged(self) -> bool:
        """The slab carries a net formal charge (see :attr:`charge`)."""
        return self.charge is not None and abs(self.charge) > _CHARGE_TOL


class SlabSet(list):
    """The :class:`SlabRecord` list of :func:`generate_slabs`, with its context.

    Attributes:
        miller_indices: The indices enumerated, in order.
        oxidation_states: The states the polarity was computed with, or None.
        oxidation_source: ``"guessed"``, ``"given"`` or None.
        notes: What the command prints as NOTE lines (an index without a
            symmetric termination, an index given twice, ...).
    """

    def __init__(self, records=(), *, miller_indices=(), oxidation_states=None,
                 oxidation_source=None, notes=()):
        super().__init__(records)
        self.miller_indices = list(miller_indices)
        self.oxidation_states = oxidation_states
        self.oxidation_source = oxidation_source
        self.notes = list(notes)


def _is_single_index(value) -> bool:
    """One index (``"1 0 0"``, ``(1, 0, 0)``, ``["1", "0", "0"]``) rather than a list of them."""
    if isinstance(value, str):
        return True
    try:
        parts = list(value)
    except TypeError:
        return False
    if all(isinstance(part, Integral) and not isinstance(part, bool) for part in parts):
        return True
    return all(isinstance(part, str) and len(_index_tokens(part)) == 1 for part in parts)


def _reduced(index: tuple[int, ...]) -> tuple[int, ...]:
    divisor = math.gcd(*index)
    return tuple(int(i) // divisor for i in index)


def _normal_extent(slab) -> tuple[float, float, np.ndarray]:
    """``(thickness, height, frac_c)``: the extent of the atoms along the surface
    normal, the height of the cell along it, and the fractional ``c`` of each
    atom made contiguous (the bottom face first).

    The fractional ``c`` coordinates are taken modulo 1 and the widest empty
    gap is the vacuum, so a slab stored across the cell boundary is measured
    whole; since ``a`` and ``b`` lie in the surface, the height of an atom
    along the normal is its fractional ``c`` times the cell height.
    """
    height = abs(float(np.dot(slab.lattice.matrix[2], slab.normal)))
    frac = np.mod(slab.frac_coords[:, 2], 1.0)
    order = np.sort(frac)
    gaps = np.diff(np.concatenate([order, [order[0] + 1.0]]))
    widest = int(np.argmax(gaps))
    bottom = order[(widest + 1) % len(order)]
    contiguous = bottom + np.mod(frac - bottom, 1.0)
    return float((1.0 - gaps[widest]) * height), height, contiguous


def _is_polar(slab, states: dict[str, float], contiguous: np.ndarray) -> bool:
    """``Slab.is_polar`` of the slab decorated with ``states``.

    pymatgen takes the dipole from the Cartesian positions as stored, so the
    copy that is decorated has its atoms moved by whole cells to the
    contiguous positions first (a centered slab needs no move).
    """
    decorated = slab.copy()
    for site, jump in enumerate(np.rint(contiguous - decorated.frac_coords[:, 2])):
        if jump:
            decorated.translate_sites([site], [0, 0, int(jump)], frac_coords=True,
                                      to_unit_cell=False)
    decorated.add_oxidation_state_by_element(states)
    return bool(decorated.is_polar())


def _is_symmetric(slab, tolerance: float) -> bool:
    try:
        return bool(slab.is_symmetric(symprec=tolerance))
    except Exception:  # spglib found no symmetry at this tolerance
        return False


def _formula(counts: Counter, order: tuple[str, ...]) -> str:
    symbols = [s for s in order if s in counts] + sorted(s for s in counts if s not in order)
    return "".join(f"{s}{counts[s] if counts[s] != 1 else ''}" for s in symbols)


def _generator_class():
    """pymatgen's ``SlabGenerator`` with the faces of ``symmetrize=True``
    compared at the tolerance of the run (pymatgen's own test is fixed at 0.1)."""
    from pymatgen.core.surface import SlabGenerator

    class _SlabGenerator(SlabGenerator):
        symprec = DEFAULT_TOLERANCE
        gave_up = 0

        def nonstoichiometric_symmetrized_slab(self, init_slab):
            # pymatgen's SlabGenerator.nonstoichiometric_symmetrized_slab, step for
            # step, with Slab.is_symmetric at self.symprec: atoms are removed from
            # the top (then, in a copy, the bottom) until the faces are equivalent,
            # or given up when the slab has no more atoms than the bulk cell
            if _is_symmetric(init_slab, self.symprec):
                return [init_slab]
            symmetrized = []
            for surface in ("top", "bottom"):
                slab = init_slab.copy()
                slab.energy = init_slab.energy
                while True:
                    z_coords = [site[2] for site in slab.frac_coords]
                    extreme = max(z_coords) if surface == "top" else min(z_coords)
                    slab.remove_sites([z_coords.index(extreme)])
                    if len(slab) <= len(self.parent):
                        self.gave_up += 1
                        break
                    if _is_symmetric(slab, self.symprec):
                        symmetrized.append(slab)
                        break
            return symmetrized

    return _SlabGenerator


def _reduce_in_plane(slab):
    """``--lll-reduce``: the slab with the shortest in-plane cell ``a, b``
    (Lagrange-Gauss reduction) and ``c`` minus the in-plane lattice vector
    closest to its in-plane part, so that ``a`` and ``b`` stay in the surface.

    pymatgen's own ``lll_reduce`` LLL-reduces the whole slab cell, which can
    mix ``c`` into ``a`` or ``b`` and so tilt the cell out of the surface.
    The fractional ``c`` of every atom is unchanged.
    """
    from pymatgen.core import Lattice
    from pymatgen.core.surface import Slab

    a, b, c = (np.array(vector, dtype=float) for vector in slab.lattice.matrix)
    ta, tb = np.array([1, 0, 0]), np.array([0, 1, 0])
    for _ in range(1000):
        if b @ b < a @ a - 1e-12:
            a, b, ta, tb = b, a, tb, ta
        mu = int(round(float(a @ b) / float(a @ a)))
        if mu == 0:
            break
        b, tb = b - mu * a, tb - mu * ta
    if np.dot(np.cross(a, b), c) < 0:  # keep the handedness of the cell
        b, tb = -b, -tb
    normal = np.cross(a, b) / np.linalg.norm(np.cross(a, b))
    in_plane = c - float(c @ normal) * normal
    gram = np.array([[a @ a, a @ b], [a @ b, b @ b]])
    x, y = np.linalg.solve(gram, [in_plane @ a, in_plane @ b])
    best = None
    for i, j in itertools.product((-1, 0, 1), repeat=2):
        p, q = int(round(x)) + i, int(round(y)) + j
        rest = c - p * a - q * b
        if best is None or rest @ rest < best[0] - 1e-12:
            best = (rest @ rest, p, q, rest)
    _, p, q, c = best
    transformation = np.array([ta, tb, np.array([0, 0, 1]) - p * ta - q * tb])
    lattice = Lattice(np.array([a, b, c]))
    frac = lattice.get_fractional_coords(slab.cart_coords)
    frac[:, :2] = np.mod(np.round(frac[:, :2], 12), 1.0)
    return Slab(lattice, slab.species_and_occu, frac, slab.miller_index, slab.oriented_unit_cell,
                slab.shift, np.dot(transformation, slab.scale_factor), reorient_lattice=True,
                site_properties=slab.site_properties, energy=slab.energy)


def generate_slabs(
    structure,
    miller_index=None,
    *,
    max_index: int | None = None,
    slab_size: float = DEFAULT_SLAB_SIZE,
    vacuum_size: float = DEFAULT_VACUUM_SIZE,
    symmetrize: bool = False,
    oxidation_states: dict[str, float] | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    center: bool = True,
    primitive: bool = True,
    lll_reduce: bool = False,
    in_unit_planes: bool = False,
    max_normal_search: int | None = None,
    orthogonal_c: bool = False,
) -> SlabSet:
    """Every distinct termination of the given surfaces, measured.

    This is ``crystod-slab`` without the files.  For each index,
    ``SlabGenerator(structure, hkl, slab_size, vacuum_size, ...)
    .get_slabs(symmetrize=symmetrize, tol=0.1, ftol=0.1)``; the slabs are
    numbered ``t1, t2, ...`` per index in pymatgen's order (``s1, s2, ...``
    with ``symmetrize``).

    Args:
        structure: The bulk ``Structure`` (see :func:`load_structure`);
            Miller indices refer to its axes.
        miller_index: One index (``(1, 0, 0)``, ``"1 0 0"``, a four-index
            ``(1, 0, -1, 0)``) or a list of them.  Give this or ``max_index``.
        max_index: Every index with ``|h|, |k|, |l| <= max_index`` distinct
            under the Laue group of the crystal (:func:`distinct_miller_indices`).
        slab_size: Minimum slab height in Angstrom (in ``d_hkl`` lattice
            planes with ``in_unit_planes``): whole oriented cells are stacked
            until they reach it, and the atoms span that height less the gap
            at the cut.
        vacuum_size: Minimum vacuum height, likewise in whole cells.
        symmetrize: Symmetric slabs: atoms are removed from one face of every
            asymmetric slab until the faces are equivalent at ``tolerance``
            (non-stoichiometric, possibly charged: :attr:`SlabRecord.charge`).
        oxidation_states: ``{"Sr": 2, "Ti": 4, "O": -2}`` for the dipole; by
            default pymatgen's guess, and no polarity (``polar = None``) when
            there is none.  Given states must balance the bulk.
        tolerance: Symmetry tolerance in Angstrom of the Laue group of
            ``max_index``, of ``Slab.is_symmetric`` and of the symmetrization.
        center: Center the slab in the cell (pymatgen ``center_slab``).
        primitive: Reduce the in-plane cell to a primitive one when pymatgen
            can (it keeps the oriented cell's ``a``, ``b`` otherwise).
        lll_reduce: Reduce the in-plane cell to its shortest ``a``, ``b`` and
            make ``c`` as perpendicular as whole in-plane vectors allow
            (``a`` and ``b`` stay in the surface).
        in_unit_planes: ``slab_size`` and ``vacuum_size`` count ``d_hkl``
            lattice planes of the cell (one can hold several atomic layers).
        max_normal_search: pymatgen ``max_normal_search`` (a ``c`` vector as
            perpendicular to the surface as possible).
        orthogonal_c: Make ``c`` perpendicular to the surface
            (``Slab.get_orthogonal_c_slab``) before the measurements.

    Returns:
        The :class:`SlabSet` (a list of :class:`SlabRecord`).

    Raises:
        SystemExit: Both or neither of ``miller_index`` and ``max_index``, a
            bad index, non-positive sizes or tolerance, oxidation states that
            do not fit the structure, or a slab pymatgen cannot build
            (``ValueError`` through :mod:`crystod.slab`).

    Example:
        >>> from crystod import slab
        >>> from crystod.examples import example_path
        >>> s = slab.load_structure(example_path("221_PPOSCAR_SrTiO3"))
        >>> records = slab.generate_slabs(s, max_index=1)
        >>> [(r.miller_index, r.n_atoms, r.polar) for r in records]
        [((1, 1, 1), 25, True), ((1, 1, 0), 20, True), ((1, 0, 0), 15, False)]
    """
    if (miller_index is None) == (max_index is None):
        _fail("give either miller_index or max_index (exactly one of them).")
    slab_size = _check_size(slab_size, "slab height")
    vacuum_size = _check_size(vacuum_size, "vacuum height")
    tolerance = _check_tolerance(tolerance)
    if max_normal_search is not None and (
            isinstance(max_normal_search, bool) or not isinstance(max_normal_search, Integral)
            or max_normal_search < 1):
        _fail(f"max_normal_search must be a positive integer (got {max_normal_search}).")
    states, source = resolve_oxidation_states(structure, oxidation_states)

    notes: list[str] = []
    if max_index is not None:
        indices = distinct_miller_indices(structure, max_index, tolerance)
    else:
        given = [miller_index] if _is_single_index(miller_index) else list(miller_index)
        if not given:
            _fail("no Miller index given.")
        indices = []
        for index in given:
            hkl = parse_miller_index(index)
            reduced = _reduced(hkl)
            if reduced != hkl:
                notes.append(f"{format_miller_index(hkl)} is the plane family "
                             f"{format_miller_index(reduced)}")
            if reduced in indices:
                notes.append(f"{format_miller_index(reduced)} is given more than once; "
                             "its slabs are listed once")
                continue
            indices.append(reduced)

    order = tuple(dict.fromkeys(site.specie.symbol for site in structure))
    bulk_formula = _composition_formula(structure, order)
    generator_class = _generator_class()
    rotations = _rotations(structure, tolerance) if symmetrize else []
    records: list[SlabRecord] = []
    for hkl in indices:
        label = format_miller_index(hkl)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                generator = generator_class(
                    structure, hkl, slab_size, vacuum_size, lll_reduce=False,
                    center_slab=center, in_unit_planes=in_unit_planes, primitive=primitive,
                    max_normal_search=max_normal_search)
                generator.symprec = tolerance
                slabs = generator.get_slabs(symmetrize=symmetrize, tol=_SLAB_TOL,
                                            ftol=_SLAB_FTOL)
            except Exception as exc:  # anything pymatgen raises for this surface
                _fail(f"pymatgen could not build the {label} slab: {_one_line(exc)}.")
            measured = []
            for termination, slab in enumerate(slabs, start=1):
                if lll_reduce:
                    slab = _reduce_in_plane(slab)
                if orthogonal_c:
                    slab = slab.get_orthogonal_c_slab()
                thickness, height, contiguous = _normal_extent(slab)
                polar = None if states is None else _is_polar(slab, states, contiguous)
                counts = Counter(site.specie.symbol for site in slab)
                charge = (None if states is None
                          else float(sum(states[symbol] * n for symbol, n in counts.items())))
                measured.append(SlabRecord(
                    miller_index=tuple(int(i) for i in hkl),
                    termination=termination,
                    shift=float(slab.shift),
                    slab=slab,
                    n_atoms=len(slab),
                    thickness=thickness,
                    vacuum=height - thickness,
                    area=float(slab.surface_area),
                    formula=_formula(counts, order),
                    polar=polar,
                    symmetric=_is_symmetric(slab, tolerance),
                    bulk_formula=bulk_formula,
                    species_order=order,
                    charge=charge,
                    symmetrized=bool(symmetrize),
                ))
        records.extend(measured)
        messages = list(dict.fromkeys(
            " ".join(str(w.message).split()) for w in caught
            if issubclass(w.category, UserWarning)
            and "pymatgen" in (w.filename or "")))
        gave_up = generator.gave_up or any(m.startswith("Too many sites removed")
                                            for m in messages)
        messages = [m for m in messages if not m.startswith("Too many sites removed")]
        if symmetrize and not measured:
            if _turns_over(hkl, rotations):
                hint = "a larger --slab-size may give one"
            else:
                hint = ("none may exist: no operation of the bulk turns this normal over, so "
                        "a thicker slab will not help")
            notes.append(f"no symmetric termination found for {label} (removing atoms from one "
                         f"face did not make the faces equivalent before the slab had no more "
                         f"atoms than the bulk cell; {hint})")
        elif not measured:
            notes.append(f"pymatgen returned no slab for {label}")
        elif symmetrize and gave_up:
            notes.append(f"{label}: a termination could not be symmetrized (removing atoms from "
                         "one face did not make the faces equivalent before the slab had no "
                         "more atoms than the bulk cell) and is left out")
        notes.extend(f"{label}: pymatgen: {m}" for m in messages)

    return SlabSet(records, miller_indices=indices, oxidation_states=states,
                   oxidation_source=source, notes=notes)


# ------------------------------------------------------------------- the table
def slab_status(record: SlabRecord, max_atoms: int | None = None, no_polar: bool = False,
                no_files: bool = False) -> str:
    """What ``crystod-slab`` does with the slab, in the words of its table.

    ``"skipped: 25 > 16 atoms"`` (more atoms than ``max_atoms``), then
    ``"skipped: polar"`` (``no_polar`` and a polar slab; a slab of unknown
    polarity is kept), otherwise ``"written"``, or ``"not written
    (--no-files)"`` with ``no_files``.
    """
    if max_atoms is not None and record.n_atoms > max_atoms:
        return f"skipped: {record.n_atoms} > {max_atoms} atoms"
    if no_polar and record.polar:
        return "skipped: polar"
    return "not written (--no-files)" if no_files else "written"


_HEADERS = (
    ("#", "#"), ("h", "h"), ("k", "k"), ("l", "l"), ("shift", "shift"), ("atoms", "atoms"),
    ("thickness (A)", "thickness"), ("vacuum (A)", "vacuum"), ("area (A^2)", "area"),
    ("formula", "formula"), ("charge", "charge"), ("polar", "polar"),
    ("symmetric", "symmetric"), ("status", "status"),
)
_LEFT = {"formula", "polar", "symmetric", "status"}


def _has_charge_column(records) -> bool:
    return any(record.charged for record in records)


def table_legend(records) -> str:
    """:data:`TABLE_LEGEND`, plus :data:`CHARGE_LEGEND` when the table of
    ``records`` has a charge column (a charged, symmetrized slab)."""
    return TABLE_LEGEND + (" " + CHARGE_LEGEND if _has_charge_column(records) else "")


def format_table(records, max_atoms: int | None = None, no_polar: bool = False,
                 no_files: bool = False, width: int = OUTPUT_WIDTH) -> str:
    """The slab table ``crystod-slab`` prints, one header line and one line per slab.

    The columns are ``# h k l shift atoms thickness vacuum area formula
    [charge] polar symmetric status`` (see :func:`table_legend`; the charge
    column only when a slab is charged); the status is :func:`slab_status`.
    The column widths follow the content; when the table would be wider than
    ``width`` columns the units leave the header, then the columns move
    closer together, then the formulas leave the table for lines of their
    own (``#1 Sr5Ti5O15``) below it.

    Returns:
        The table as one string (lines joined by newlines, no trailing one;
        :func:`crystod.search.format_table` returns a list of lines instead).
    """
    def flag(value) -> str:
        return "?" if value is None else ("yes" if value else "no")

    with_charge = _has_charge_column(records)
    rows = []
    for number, record in enumerate(records, start=1):
        h, k, l = record.miller_index
        charge = "?" if record.charge is None else format_charge(record.charge)
        rows.append((str(number), str(h), str(k), str(l), f"{record.shift:.3f}",
                     str(record.n_atoms), f"{record.thickness:.2f}", f"{record.vacuum:.2f}",
                     f"{record.area:.2f}", record.formula, charge, flag(record.polar),
                     flag(record.symmetric),
                     slab_status(record, max_atoms, no_polar, no_files)))

    def render(header_kind: int, gap: int, with_formula: bool) -> list[str]:
        columns = [i for i in range(len(_HEADERS))
                   if (with_formula or _HEADERS[i][0] != "formula")
                   and (with_charge or _HEADERS[i][0] != "charge")]
        titles = [_HEADERS[i][header_kind] for i in columns]
        widths = [max([len(title)] + [len(row[i]) for row in rows])
                  for title, i in zip(titles, columns)]
        widths[0] = max(widths[0], 3)

        def line(cells) -> str:
            parts = []
            for n, (cell, size) in enumerate(zip(cells, widths)):
                name = _HEADERS[columns[n]][0]
                text = cell.ljust(size) if name in _LEFT else cell.rjust(size)
                # h k l sit closer together than the other columns, # stands apart
                if n == 0:
                    sep = ""
                elif name in ("k", "l"):
                    sep = " "
                else:
                    sep = " " * (2 if name == "h" else gap)
                parts.append(sep + text)
            return "".join(parts).rstrip()

        return [line(titles), *(line([row[i] for i in columns]) for row in rows)]

    for header_kind, gap, with_formula in ((0, 2, True), (1, 2, True), (1, 1, True),
                                           (1, 1, False)):
        lines = render(header_kind, gap, with_formula)
        if max(len(text) for text in lines) <= width:
            break
    if not with_formula:
        # "#n formula" stays together on one line (a non-breaking space)
        text = " formulas: " + "  ".join(f"#{row[0]}\xa0{row[9]}" for row in rows)
        wrapped = textwrap.wrap(text, width=width, subsequent_indent="   ",
                                break_on_hyphens=False, break_long_words=False)
        lines.extend(line.replace("\xa0", " ") for line in wrapped)
    return "\n".join(lines)


# ------------------------------------------------------------------ the files
def slab_filename(record: SlabRecord) -> str:
    """``POSCAR_slab_{bulk formula}_{h}{k}{l}_t{n}``: ``POSCAR_slab_SrTiO3_111_t1``.

    A negative index is written as ``-1`` (``POSCAR_slab_NaCl_10-1_t1``); when
    a component is 10 or more the components are separated by ``_``
    (``POSCAR_slab_SrTiO3_1_1_10_t1``); a symmetrized slab has ``s{n}``
    instead of ``t{n}`` (``POSCAR_slab_SrTiO3_100_s1``).  Any character other
    than a letter, a digit, ``.``, ``_`` or ``-`` becomes ``_``.
    """
    h, k, l = record.miller_index
    index = f"{h}{k}{l}" if max(abs(h), abs(k), abs(l)) < 10 else f"{h}_{k}_{l}"
    kind = "s" if record.symmetrized else "t"
    name = f"POSCAR_slab_{record.bulk_formula}_{index}_{kind}{record.termination}"
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def slab_poscar_text(record: SlabRecord, species_order=None) -> str:
    """The POSCAR of a slab: direct coordinates, one block per element.

    The sites are sorted (stably) by element in ``species_order``, by
    default the element order of the bulk structure; the comment line reads
    ``SrTiO3 (1 1 1) slab, termination t1, Sr5Ti5O15, shift 0.250,
    crystod-slab`` (``symmetrized s1`` for a symmetrized slab).
    """
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.vasp.inputs import Poscar

    order = list(species_order or record.species_order)
    slab = record.slab
    order += [s for s in dict.fromkeys(site.specie.symbol for site in slab) if s not in order]
    rank = {symbol: n for n, symbol in enumerate(order)}
    sites = sorted(range(len(slab)), key=lambda i: rank[slab[i].specie.symbol])
    matrix = np.round(slab.lattice.matrix, 12) + 0.0
    coords = np.array(slab.frac_coords[sites], dtype=float)
    # in-plane coordinates into [0, 1); c stays as stored (the slab is contiguous)
    coords[:, :2] = np.mod(np.round(coords[:, :2], 12), 1.0)
    coords = np.round(coords, 12) + 0.0
    structure = Structure(Lattice(matrix), [slab[i].specie.symbol for i in sites], coords)
    kind = "symmetrized s" if record.symmetrized else "termination t"
    comment = (f"{record.bulk_formula} {format_miller_index(record.miller_index)} slab, "
               f"{kind}{record.termination}, {record.formula}, shift {record.shift:.3f}, "
               "crystod-slab")
    return Poscar(structure, comment=comment).get_str(direct=True, significant_figures=12)


def write_slab_poscar(record: SlabRecord, directory=".", species_order=None, *,
                      overwrite: bool = False) -> tuple[str, str]:
    """Write the POSCAR of a slab (:func:`slab_poscar_text`) into ``directory``.

    The directory is created if needed.  As :func:`crystod.search.write_poscar`
    does, an identical file of the same name (:func:`slab_filename`) is kept,
    and one with different content is replaced only with ``overwrite``.

    Returns:
        ``(path, status)`` with status ``"wrote"``, ``"kept"`` (an identical
        file was already there) or ``"replaced"``.

    Raises:
        SystemExit: A different file is in the way and ``overwrite`` is
            False, the directory cannot be created or the file cannot be
            written (``ValueError`` through :mod:`crystod.slab`).
    """
    directory = Path(directory)
    if directory.exists() and not directory.is_dir():
        _fail(f"{directory} exists and is not a directory.")
    path = directory / slab_filename(record)
    text = slab_poscar_text(record, species_order)
    status = "wrote"
    if path.is_dir():
        _fail(f"{path} is a directory.")
    if path.exists():
        try:
            same = path.read_bytes() == text.encode("utf-8")
        except OSError:
            same = False
        if same:
            return str(path), "kept"
        if not overwrite:
            _fail(f"{path} already exists with different content (overwrite=True, or --force, "
                  "replaces it).")
        status = "replaced"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        _fail(f"cannot write {path}: {exc.strerror or exc}.")
    return str(path), status
