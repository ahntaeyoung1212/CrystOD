"""Superlattice reflections of a distorted structure (``crystod-xrd --parent``).

A structural phase transition that multiplies the primitive cell (octahedral
tilts of a perovskite, antiparallel cation shifts, a modulation) makes new,
weak Bragg reflections appear between the reflections of the high-symmetry
parent.  This module computes the powder pattern of the distorted (child)
structure reflection family by reflection family, indexes every family on
the parent lattice, and assigns it:

- integer parent indices (in the parent primitive basis): a *fundamental*
  reflection (the parent Gamma point), printed with the child intensity and
  the intensity of the parent reference structure on the same scale;
- fractional primitive parent indices: a *superlattice* reflection; the
  fractional part is a parent k point, named after the k star of the
  symmetry-mode analysis it belongs to (``R``, ``M``, ``X``, ``SM`` ...),
  and the
  intensity is split among the parent irreps of that star by evaluating the
  structure factor of the parent reference structure plus each irrep's
  projected displacement field alone (``share_X = |F_X|^2 / sum_Y |F_Y|^2``
  over the irreps Y of the star).  A reflection with ``share_X >= 0.95`` is
  *diagnostic* for X: its intensity comes from that irrep alone.

The pairing of the two structures, the k stars and the irrep-projected
displacements come from the symmetry-mode analysis behind ``crystod-group
--supergroup-cif`` (:class:`crystod.symmetry_mode.SymmetryModeAnalysis`),
so the irrep labels are the ISO-IR labels of that analysis.

The structure factor ``F(G) = sum_j f_j(s) exp(2 pi i G . r_j)`` is
evaluated with pymatgen's tables and formulas: the X-ray form factors
``f = Z - 41.78214 s^2 sum_i a_i exp(-b_i s^2)`` (``s = sin(theta) /
lambda``) of :class:`~pymatgen.analysis.diffraction.xrd.XRDCalculator` with
the Lorentz-polarization factor ``(1 + cos^2 2theta) / (sin^2 theta cos
theta)``, or the coherent neutron scattering lengths of
:class:`~pymatgen.analysis.diffraction.neutron.NDCalculator` with the
Lorentz factor ``1 / (sin^2 theta cos theta)``.  No Debye-Waller factor, no
magnetic scattering, no preferred orientation.  Summed over the families at
equal ``2theta`` the intensities are those of ``crystod-xrd -c CHILD``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from .xrd_pattern import (
    DEFAULT_PEAK_PROFILE,
    DEFAULT_RADIATION,
    DEFAULT_TWO_THETA_RANGE,
    DEFAULT_WIDTH,
    DEFAULT_XRAY_TYPE,
    PEAK_PROFILES,
    RADIATIONS,
    XRAY_TYPES,
    radiation_lines,
)

#: A superlattice reflection is *diagnostic* for an irrep whose share of the
#: intensity (among the irreps of its k star) is at least this fraction.
DIAGNOSTIC_SHARE = 0.95

#: An irrep is extinct at first order at a superlattice reflection when the
#: linear term of its structure factor, ``|sum_j f_j exp(2 pi i G.r_j) 2 pi i
#: G.u_j|^2``, is below this fraction of its largest value over the families
#: of the star.
EXTINCT_FIRST_ORDER = 1e-8

#: Families of the child weaker than this percentage of the strongest one
#: are systematic absences (zero up to rounding) and are left out.
ABSENT_INTENSITY = 1e-6

#: Displacement amplitudes (Angstrom) below this are irreps that the child
#: symmetry allows but the structure does not distort along.
_ZERO_AMPLITUDE = 1e-4

_GAMMA = "GM"
_LABELS_PER_IRREP = 5
_STRUCTURE_SUFFIXES = (".cif", ".vasp", ".poscar")


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


def _fraction_label(values) -> str:
    """``"(1 1 3)/2"`` for (1/2, 1/2, 3/2), ``"(1 0 0)"`` for integers."""
    denominator = 1
    for value in values:
        denominator = denominator * value.denominator // np.gcd(denominator, value.denominator)
    numerators = [int(value * denominator) for value in values]
    body = "(" + " ".join(str(n) for n in numerators) + ")"
    return body if denominator == 1 else f"{body}/{denominator}"


def _kvector_label(values) -> str:
    """``"(1/2,1/2,1/2)"``: a k vector as the symmetry-mode table prints it."""
    return "(" + ",".join(str(value) for value in values) + ")"


@dataclass(frozen=True)
class SuperlatticeReflection:
    """One reflection family of the child structure, indexed on the parent.

    Attributes:
        two_theta: Scattering angle ``2theta`` in degrees (the first line of
            the radiation).
        d: Interplanar spacing in Angstrom.
        hkl_child: Miller indices of the family in the conventional cell of
            the child space group (spglib standard setting).
        multiplicity: Number of reciprocal-lattice points in the family (the
            orbit under the child point group plus Friedel partners).
        hkl_parent: The same reflection on the conventional cell of the
            parent, as exact fractions.
        kname: The parent k star the reflection belongs to (``"GM"`` for a
            fundamental reflection), or the k vector itself when it matches
            no star of the analysis.
        kvec: The parent k point, the fractional part of the parent indices
            in the parent primitive reciprocal basis, as fractions in [0, 1).
        intensity: Child intensity, the strongest family of the child = 100.
        intensity_parent: For a fundamental reflection, the parent reference
            structure at the same reflection (same multiplicity, Lorentz
            factor and scale); None for a superlattice reflection, to which
            the parent contributes nothing.
        contributions: For a superlattice reflection, ``(irrep, share)``
            for every irrep of the k star, largest share first (shares sum
            to 1); empty for a fundamental reflection.
        first_order_extinct: The irreps of the star whose first-order
            structure factor vanishes at this reflection (see
            :data:`EXTINCT_FIRST_ORDER`).
        equivalents: The parent indices of every member of the family.
        star: Index of the k star in the star list of the symmetry-mode
            analysis (``analysis.stars``), -1 when the k point matches none.
            Two stars can share a name (two points of one symmetry line);
            the index tells them apart.
        k_label: The k column of the table: :attr:`kname`, followed by the
            k vector when another star of the analysis has the same name
            (``"DT(0,1/3,0)"``).
    """

    two_theta: float
    d: float
    hkl_child: tuple[int, int, int]
    multiplicity: int
    hkl_parent: tuple[Fraction, Fraction, Fraction]
    kname: str
    kvec: tuple[Fraction, Fraction, Fraction]
    intensity: float
    intensity_parent: float | None
    contributions: tuple[tuple[str, float], ...] = ()
    first_order_extinct: tuple[str, ...] = ()
    equivalents: tuple[tuple[Fraction, Fraction, Fraction], ...] = field(default=(), repr=False)
    star: int = -1
    k_label: str = ""

    @property
    def superlattice(self) -> bool:
        """True unless the reflection indexes to the parent Gamma point (the
        parent primitive indices are not all integers; the conventional
        indices of a centred parent can still be, with the parity the
        centring forbids)."""
        return self.kname != _GAMMA

    @property
    def hkl_child_label(self) -> str:
        """``"(1 0 3) x8"``: child indices and multiplicity."""
        h, k, l = self.hkl_child
        return f"({h} {k} {l}) x{self.multiplicity}"

    @property
    def hkl_parent_label(self) -> str:
        """``"(1 1 3)/2"``: parent indices with their common denominator."""
        return _fraction_label(self.hkl_parent)

    @property
    def contributions_label(self) -> str:
        """``"R4+ 97 %, R5+ 3 %"``; irreps that round to 0 % are left out.

        ``"-"`` for a fundamental reflection; ``"higher order"`` for a
        superlattice reflection of a star along which the child is not
        distorted (its intensity comes from higher-order terms of the other
        stars: harmonics and couplings).
        """
        if not self.superlattice:
            return "-"
        parts = [f"{irrep} {round(100 * share):d} %" for irrep, share in self.contributions
                 if round(100 * share) > 0]
        return ", ".join(parts) if parts else "higher order"

    @property
    def diagnostic(self) -> str | None:
        """The irrep this reflection is diagnostic for, or None."""
        for irrep, share in self.contributions:
            if share >= DIAGNOSTIC_SHARE:
                return irrep
        return None


@dataclass(frozen=True)
class KPointSummary:
    """The superlattice reflections of one parent k star.

    Attributes:
        kname: The k star (``"R"``).
        kvec: Its representative k vector (parent primitive reciprocal basis).
        irreps: ``(irrep, amplitude in Angstrom)`` of the irreps of this star
            along which the child is distorted, from the symmetry-mode
            analysis.  A complex-type pair whose two irreps sit on the star
            of k and on the separate star of -k is listed under both stars
            with the one amplitude of the pair (see :attr:`shared`).
        reflections: Indices into :attr:`SuperlatticePattern.reflections` of
            the families at this star, in order of ``2theta``.
        diagnostic: ``(irrep, indices)``: the families diagnostic for each
            irrep (share >= :data:`DIAGNOSTIC_SHARE`).
        extinctions: ``(irrep, rule, indices, n_absent)``: the families at
            which an irrep is extinct at first order (indices into
            :attr:`SuperlatticePattern.reflections`), how many of the
            families absent in the child are too, and ``rule``, a condition
            on the parent indices that some reflection of every such family
            and of no other family of the star fulfils (``"|h| = |k| =
            |l|"``), or ``""``.
        absent: Indices into :attr:`SuperlatticePattern.absent` of the
            families of this star with no intensity in the child.
        star: Index of the star in ``analysis.stars`` (see
            :attr:`SuperlatticeReflection.star`).
        shared: ``(irrep, other star)``: the irreps of :attr:`irreps` that
            are complex-conjugate pairs spanning this star and the separate
            star of -k (a parent with no operation taking k to -k), with
            the other star as ``"GQ (2/3,0,0)"``; their amplitude is that of
            the pair, listed under both stars and counted once.
    """

    kname: str
    kvec: tuple[Fraction, Fraction, Fraction]
    irreps: tuple[tuple[str, float], ...]
    reflections: tuple[int, ...]
    diagnostic: tuple[tuple[str, tuple[int, ...]], ...]
    extinctions: tuple[tuple[str, str, tuple[int, ...], int], ...]
    absent: tuple[int, ...] = ()
    star: int = -1
    shared: tuple[tuple[str, str], ...] = ()


@dataclass
class SuperlatticePattern:
    """The child powder pattern indexed on the parent lattice.

    Attributes:
        parent_file: The parent (high-symmetry) structure file.
        child_file: The child (distorted) structure file.
        parent_formula: Reduced formula of the parent.
        child_formula: Reduced formula of the child.
        parent_space_group: ``"Pm-3m (No. 221)"``.
        child_space_group: ``"I4/mcm (No. 140)"``.
        size: Primitive-cell multiplication of the child.
        cell_relation: The child primitive basis in parent primitive units
            (rows), as ``crystod-group --supergroup-cif`` prints it.
        parent_setting: The matrix ``A`` (rows: the parent primitive basis in
            conventional units) that converts the parent indices,
            ``h_conv = h_prim A^-T``.
        radiation: ``"xray"`` or ``"neutron"``.
        radiation_name: ``"CuKa"``, ``"MoKa1"`` or ``"neutron"``.
        lines: ``((line, wavelength, weight), ...)`` of the radiation; the
            table uses the first line, the drawn pattern all of them.
        two_theta_range: The ``(min, max)`` window in degrees.
        reflections: Every family of the child in the window, in order of
            ``2theta`` (the k-point summary is built from all of them).
        absent: The superlattice families of the child lattice in the
            window whose child intensity is zero (systematic absences of the
            child space group); not printed in the table, used for the
            extinction rules.
        kpoints: One :class:`KPointSummary` per parent k star with
            superlattice reflections in the window, in the order of the
            symmetry-mode analysis.
        child_peaks: ``(2theta, intensity)`` of every reciprocal-lattice
            point of the child for every line (weights applied), the input
            of the drawn pattern.
        parent_peaks: The same for the parent reference structure.
        min_intensity: Families weaker than this percentage are left out of
            the printed and written table (:attr:`listed`), not out of
            :attr:`reflections` or the k-point summary.
    """

    parent_file: str
    child_file: str
    parent_formula: str
    child_formula: str
    parent_space_group: str
    child_space_group: str
    size: int
    cell_relation: tuple[tuple[int, int, int], ...]
    parent_setting: np.ndarray
    radiation: str
    radiation_name: str
    lines: tuple[tuple[str, float, float], ...]
    two_theta_range: tuple[float, float]
    reflections: list[SuperlatticeReflection]
    absent: list[SuperlatticeReflection]
    kpoints: list[KPointSummary]
    child_peaks: tuple[np.ndarray, np.ndarray] = field(repr=False)
    parent_peaks: tuple[np.ndarray, np.ndarray] = field(repr=False)
    min_intensity: float = 0.0

    @property
    def listed(self) -> list[SuperlatticeReflection]:
        """The families of the table: those at or above :attr:`min_intensity`."""
        return [reflection for reflection in self.reflections
                if reflection.intensity >= self.min_intensity]

    @property
    def superlattice(self) -> list[SuperlatticeReflection]:
        """The superlattice reflections only."""
        return [reflection for reflection in self.reflections if reflection.superlattice]

    @property
    def fundamental(self) -> list[SuperlatticeReflection]:
        """The fundamental reflections only."""
        return [reflection for reflection in self.reflections if not reflection.superlattice]

    @property
    def radiation_label(self) -> str:
        """``"CuKa1 = 1.5405929 A"`` or ``"neutron, 1.5 A"``."""
        if self.radiation == "neutron":
            return f"neutron, {self.lines[0][1]:g} A"
        return f"{self.lines[0][0]} = {self.lines[0][1]:.7f} A"


# ---------------------------------------------------------------------------
# structure factors
# ---------------------------------------------------------------------------


def _symbols(numbers) -> list[str]:
    from pymatgen.core.periodic_table import Element

    return [Element.from_Z(int(z)).symbol for z in numbers]


def _scattering_factors(numbers, radiation: str):
    """``s2 -> (n_points, n_atoms)`` scattering factors (pymatgen's tables).

    Raises:
        SystemExit: An element has no tabulated scattering data.
    """
    symbols = _symbols(numbers)
    if radiation == "neutron":
        from pymatgen.analysis.diffraction.neutron import ATOMIC_SCATTERING_LEN

        missing = sorted({s for s in symbols if s not in ATOMIC_SCATTERING_LEN})
        if missing:
            raise SystemExit(f"ERROR: no neutron scattering length for {', '.join(missing)}.")
        lengths = np.array([ATOMIC_SCATTERING_LEN[s] for s in symbols])

        def neutron(s2):
            return np.broadcast_to(lengths, (len(s2), len(lengths)))

        return neutron

    from pymatgen.analysis.diffraction.xrd import ATOMIC_SCATTERING_PARAMS

    missing = sorted({s for s in symbols if s not in ATOMIC_SCATTERING_PARAMS})
    if missing:
        raise SystemExit(f"ERROR: no X-ray scattering coefficients for {', '.join(missing)}.")
    zs = np.array([int(z) for z in numbers], dtype=float)
    coeffs = np.array([ATOMIC_SCATTERING_PARAMS[s] for s in symbols], dtype=float)

    def xray(s2):
        s2 = np.asarray(s2, dtype=float)[:, None, None]
        return zs[None, :] - 41.78214 * s2[:, :, 0] * np.sum(
            coeffs[None, :, :, 0] * np.exp(-coeffs[None, :, :, 1] * s2), axis=2)

    return xray


def _structure_factors(h, g, frac, factors) -> np.ndarray:
    """``F(G)`` at the points ``h`` (fractional, rows) of lengths ``g``."""
    s2 = (np.asarray(g, dtype=float) / 2.0) ** 2
    phases = np.exp(2j * np.pi * (np.asarray(h, dtype=float) @ np.asarray(frac, dtype=float).T))
    return np.sum(factors(s2) * phases, axis=1)


def _lorentz(g, wavelength: float, radiation: str) -> tuple[np.ndarray, np.ndarray]:
    """``(2theta in degrees, Lorentz(-polarization) factor)`` for lengths ``g``."""
    theta = np.arcsin(np.clip(wavelength * np.asarray(g, dtype=float) / 2.0, -1.0, 1.0))
    factor = 1.0 / (np.sin(theta) ** 2 * np.cos(theta))
    if radiation != "neutron":
        factor = factor * (1.0 + np.cos(2.0 * theta) ** 2)
    return np.degrees(2.0 * theta), factor


def _reciprocal_points(lattice, wavelength: float, two_theta_range) -> tuple[np.ndarray, np.ndarray]:
    """Integer ``h`` (rows) and ``|G|`` of the reciprocal points in the window."""
    lattice = np.asarray(lattice, dtype=float)
    reciprocal = np.linalg.inv(lattice).T
    low, high = (2.0 * np.sin(np.radians(t / 2.0)) / wavelength for t in two_theta_range)
    bounds = [int(np.floor(high * np.linalg.norm(lattice[i]))) for i in range(3)]
    axes = [np.arange(-b, b + 1) for b in bounds]
    h = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    g = np.linalg.norm(h @ reciprocal, axis=1)
    keep = (g > 0) & (g <= high) & (g >= low)
    return h[keep], g[keep]


def _point_intensities(lattice, frac, numbers, radiation, wavelength, two_theta_range, cells):
    """``(h, g, 2theta, raw intensity)`` of every reciprocal point in the window.

    The structure factor is divided by ``cells`` (the number of parent
    primitive cells in the cell), so cells of different size share a scale.
    """
    h, g = _reciprocal_points(lattice, wavelength, two_theta_range)
    factors = _scattering_factors(numbers, radiation)
    two_theta, lorentz = _lorentz(g, wavelength, radiation)
    structure_factor = _structure_factors(h, g, frac, factors) / cells
    return h, g, two_theta, np.abs(structure_factor) ** 2 * lorentz


# ---------------------------------------------------------------------------
# structures
# ---------------------------------------------------------------------------


def _read_cell(path: str):
    """``(lattice, fractional positions, atomic numbers)`` of a CIF or POSCAR,
    read the way the symmetry-mode analysis reads it."""
    import warnings

    from .symmetry_mode import _ordered_numbers
    from .vasp_io import poscar_structure

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from pymatgen.core import Structure

        try:
            structure = Structure.from_file(path)
        except Exception:
            try:
                with open(path, encoding="utf-8") as handle:
                    structure = Structure.from_str(handle.read(), fmt="poscar")
            except Exception as exc:
                raise SystemExit(f"ERROR: could not read {path}: {exc}") from None
        structure = poscar_structure(structure, path)
    return (np.asarray(structure.lattice.matrix), np.asarray(structure.frac_coords),
            _ordered_numbers(structure, path))


def _child_primitive(child_file: str, analysis, tolerance: float):
    """The child as given, in the primitive basis the analysis paired.

    spglib's primitive cell without idealization keeps the atoms where the
    file puts them (the analysis works on the idealized copy); the idealized
    cell is the fallback when the two do not describe the same atoms.
    """
    import spglib

    reference = analysis.child_prim
    raw = spglib.standardize_cell(_read_cell(child_file), to_primitive=True,
                                  no_idealize=True, symprec=tolerance)
    if raw is not None and list(raw[2]) == list(reference[2]):
        difference = (np.asarray(raw[1]) - np.asarray(reference[1]) + 0.5) % 1.0 - 0.5
        if np.max(np.abs(difference)) < 0.05:
            return np.asarray(raw[0]), np.asarray(raw[1]), list(raw[2])
    return np.asarray(reference[0]), np.asarray(reference[1]), list(reference[2])


def _formula(numbers) -> str:
    from pymatgen.core.composition import Composition

    counts: dict[str, int] = {}
    for symbol in _symbols(numbers):
        counts[symbol] = counts.get(symbol, 0) + 1
    return Composition(counts).reduced_formula


def _rational_matrix(matrix, max_denominator: int = 12) -> list[list[Fraction]]:
    return [[Fraction(float(v)).limit_denominator(max_denominator) for v in row]
            for row in np.asarray(matrix, dtype=float)]


def _apply(h, matrix) -> tuple[Fraction, ...]:
    """Row vector ``h`` times a matrix of Fractions."""
    return tuple(sum((Fraction(h[i]) * matrix[i][j] for i in range(3)), Fraction(0))
                 for j in range(3))


def _star_arms(analysis) -> dict[tuple[int, ...], int]:
    """``k (units of 1/DEN, mod DEN) -> star index`` for every arm of every
    k star of the analysis (the index into ``analysis.stars``; two stars
    can share a name, never an arm)."""
    from .spacegroup_product import DEN

    algebra = analysis.algebra
    arms: dict[tuple[int, ...], int] = {}
    for index, star in enumerate(analysis.stars):
        if star.get("kind") == "tabulated":
            vectors = np.asarray(algebra.star(star["kname"])[0], dtype=np.int64)
        else:
            canonical = np.asarray(star["canonical"], dtype=np.int64)
            vectors = np.array([canonical @ np.rint(W).astype(np.int64)
                                for W in algebra.rotations])
        for vector in vectors:
            arms.setdefault(tuple(int(v) % DEN for v in vector), index)
    return arms


def _star_kvec(star) -> tuple[Fraction, ...]:
    """The representative k vector of a star of the analysis, as fractions."""
    return tuple(Fraction(float(v)).limit_denominator(24) % 1 for v in star["kvec"])


def _star_labels(analysis) -> list[str]:
    """The k-column label of every star: its name, followed by its k vector
    when another star of the analysis has the same name."""
    names = [star["kname"] for star in analysis.stars]
    return [name + _kvector_label(_star_kvec(star)) if names.count(name) > 1 else name
            for name, star in zip(names, analysis.stars)]


def _star_index(analysis, kname, kvec) -> int:
    """Index in ``analysis.stars`` of the star with this name and
    representative k vector, -1 when there is none."""
    for index, star in enumerate(analysis.stars):
        if star["kname"] == kname and np.allclose(
                np.asarray(star["kvec"], dtype=float), np.asarray(kvec, dtype=float)):
            return index
    return -1


def _mode_star(analysis, mode) -> int:
    """Index in ``analysis.stars`` of the star a mode entry is recorded on."""
    return _star_index(analysis, mode.kname, mode.kvec)


def _mode_stars(analysis, mode) -> tuple[int, ...]:
    """Indices in ``analysis.stars`` of every star a mode entry spans.

    A complex-type pair whose conjugate irrep lives on the separate star of
    -k (a parent without an operation taking k to -k) is recorded once, on
    the star of k, with a field (and amplitude) that includes the -k
    Fourier components; the -k star (``mode.partner_kname``) is a star of
    the pair too.  Stars that match nothing are left out.
    """
    indices = [_mode_star(analysis, mode)]
    partner = getattr(mode, "partner_kname", None)
    if partner is not None:
        indices.append(_star_index(analysis, partner, mode.partner_kvec))
    return tuple(dict.fromkeys(index for index in indices if index >= 0))


# ---------------------------------------------------------------------------
# computation
# ---------------------------------------------------------------------------


def superlattice_reflections(
    parent_file: str,
    child_file: str,
    radiation: str = DEFAULT_RADIATION,
    xray_type: str = DEFAULT_XRAY_TYPE,
    wavelength: float | None = None,
    two_theta_range: tuple[float, float] = DEFAULT_TWO_THETA_RANGE,
    tolerance: float = 0.01,
    *,
    min_intensity: float = 0.0,
) -> SuperlatticePattern:
    """Index the powder pattern of a distorted structure on its parent.

    This is ``crystod-xrd --parent PARENT -c CHILD`` without the files.  The
    two structures are paired by the symmetry-mode analysis; every
    reflection family of the child in the window is indexed on the parent
    lattice, classified as fundamental or superlattice, assigned to its
    parent k star, and (superlattice) split among the irreps of that star.

    Args:
        parent_file: High-symmetry structure (CIF or POSCAR).
        child_file: Distorted structure (CIF or POSCAR) whose primitive cell
            is an integer multiple of the parent's.
        radiation: ``"xray"`` (default) or ``"neutron"``.
        xray_type: X-ray line or doublet (:data:`~crystod.xrd_pattern.XRAY_TYPES`);
            the table uses its first line (Ka1 of a doublet).  Not used for
            neutrons.
        wavelength: Neutron wavelength in Angstrom (required for neutrons,
            refused for X-rays).
        two_theta_range: ``(min, max)`` in degrees.
        tolerance: Symmetry tolerance (Angstrom) of the symmetry-mode
            analysis and of the child point group.
        min_intensity: Families weaker than this percentage of the strongest
            child family are left out of the table
            (:attr:`SuperlatticePattern.listed`); the k-point summary is
            built from every family.

    Returns:
        The :class:`SuperlatticePattern`.

    Raises:
        SystemExit: Bad radiation options or window, unreadable files, a
            child that cannot be mapped onto the parent, or an element
            without scattering data (``ValueError`` when called through
            ``crystod.xrd``).

    Example:
        >>> from crystod import xrd
        >>> pattern = xrd.superlattice_reflections(
        ...     "221_PPOSCAR_SrTiO3.cif", "140_PPOSCAR_SrTiO3.cif")
        >>> first = pattern.superlattice[0]
        >>> first.hkl_parent_label, first.kname, first.contributions_label
        ('(3 1 1)/2', 'R', 'R5- 100 %')
    """
    import spglib

    from .spacegroup_product import DEN
    from .symmetry_mode import SymmetryModeAnalysis

    kind = str(radiation).strip().lower()
    name, lines = radiation_lines(kind, None if kind == "neutron" else xray_type, wavelength)
    low, high = float(two_theta_range[0]), float(two_theta_range[1])
    if not 0.0 <= low < high <= 180.0:
        raise SystemExit(
            f"ERROR: the 2theta range must satisfy 0 <= min < max <= 180 degrees "
            f"(got {two_theta_range[0]} {two_theta_range[1]}).")
    window = (low, high)
    for path in (parent_file, child_file):
        if not os.path.isfile(path):
            raise SystemExit(f"ERROR: structure file not found: {path}")

    analysis = SymmetryModeAnalysis(parent_file, child_file, tolerance)
    size = int(analysis.size)
    S = np.rint(np.asarray(analysis.mapping.S, dtype=float)).astype(np.int64)
    det = int(round(np.linalg.det(S)))
    adjugate = np.rint(np.linalg.inv(S) * det).astype(np.int64)  # inv(S) = adj / det

    lattice, frac, numbers = _child_primitive(child_file, analysis, tolerance)
    line_name, line_wavelength, _ = lines[0]
    h, g, two_theta, raw = _point_intensities(lattice, frac, numbers, kind, line_wavelength,
                                              window, size)
    if len(h) == 0:
        raise SystemExit("ERROR: no reflection of the child in this 2theta window.")

    # parent primitive indices h_child S^-T (exact) and the k point they fold to
    parent_numerators = h @ adjugate.T  # = det * h_parent
    arms = _star_arms(analysis)
    star_labels = _star_labels(analysis)

    def star_of(numerators) -> tuple[int, str, str, tuple[Fraction, ...]]:
        """``(star index, star name, k-column label, k)``."""
        k = tuple(Fraction(int(n), det) % 1 for n in numerators)
        key = tuple(int(value * DEN) % DEN if (value * DEN).denominator == 1 else None
                    for value in k)
        if all(value == 0 for value in k):
            index = arms.get(key, -1)
            return index, _GAMMA, _GAMMA, k
        if None not in key and key in arms:
            index = arms[key]
            return index, analysis.stars[index]["kname"], star_labels[index], k
        return -1, _kvector_label(k), _kvector_label(k), k

    point_star = [star_of(n) for n in parent_numerators]

    # families: orbits under the child point group plus Friedel partners,
    # never across parent k stars
    dataset = spglib.get_symmetry((lattice, frac, numbers), symprec=tolerance)
    rotations = np.unique(np.asarray(dataset["rotations"], dtype=np.int64), axis=0)
    index = {tuple(int(v) for v in row): i for i, row in enumerate(h)}
    assigned = np.full(len(h), -1)
    families: list[list[int]] = []
    for i in np.argsort(g, kind="stable"):
        if assigned[i] >= 0:
            continue
        orbit = set()
        for W in rotations:
            image = h[i] @ W
            for sign in (1, -1):
                j = index.get(tuple(int(v) for v in sign * image))
                if j is not None and assigned[j] < 0 and point_star[j][2] == point_star[i][2]:
                    orbit.add(j)
        orbit.add(int(i))
        for j in orbit:
            assigned[j] = len(families)
        families.append(sorted(orbit))

    family_raw = np.array([raw[members].sum() for members in families])
    scale = 100.0 / float(family_raw.max()) if family_raw.max() > 0 else 0.0

    # parent reference and the irrep-projected structures on the core cell
    parent_factors = _scattering_factors(analysis.parent_numbers, kind)
    core_factors = _scattering_factors(analysis.ref_z, kind)
    _, lorentz = _lorentz(g, line_wavelength, kind)
    h_parent = parent_numerators / det
    inverse_L = np.linalg.inv(analysis.L_parent)
    star_modes: dict[int, list] = {}
    for mode in analysis.modes:
        if mode.amplitude < _ZERO_AMPLITUDE:
            continue
        for index in _mode_stars(analysis, mode):
            star_modes.setdefault(index, []).append(mode)

    # conventional-cell indices of the parent and of the child
    to_parent_conventional = _rational_matrix(np.linalg.inv(analysis.A_setting).T)
    child_conv_lattice = np.asarray(analysis.child_conv[0], dtype=float)
    child_prim_lattice = np.asarray(analysis.child_prim[0], dtype=float)
    to_child_conventional = np.linalg.inv(child_prim_lattice @ np.linalg.inv(child_conv_lattice)).T

    records = []
    first_order_max: dict[tuple[int, str], float] = {}
    for members, total in zip(families, family_raw):
        intensity = float(total) * scale
        star, kname, k_label, kvec = point_star[members[0]]
        absent = intensity < ABSENT_INTENSITY
        if kname == _GAMMA and absent:
            continue
        parent_conv = {j: _apply([Fraction(int(n), det) for n in parent_numerators[j]],
                                 to_parent_conventional) for j in members}
        child_conv = {j: tuple(int(v) for v in np.rint(h[j] @ to_child_conventional))
                      for j in members}
        # the member with the fewest negative indices, child and parent
        # together; of a Friedel pair, the one whose first non-zero child
        # index is positive
        representative = max(members, key=lambda j: (
            sum(v >= 0 for v in child_conv[j]) + sum(v >= 0 for v in parent_conv[j]),
            next((v > 0 for v in child_conv[j] if v != 0), True),
            sum(v >= 0 for v in parent_conv[j]), parent_conv[j], child_conv[j]))
        hkl_child = child_conv[representative]
        intensity_parent = None
        weights, linear = [], []
        if kname == _GAMMA:
            f_parent = _structure_factors(h_parent[members], g[members],
                                          analysis.parent_positions, parent_factors)
            intensity_parent = float(np.sum(np.abs(f_parent) ** 2 * lorentz[members])) * scale
        else:
            hp = h_parent[members]
            phases = core_factors((g[members] / 2.0) ** 2) * np.exp(
                2j * np.pi * (hp @ analysis.ref_frac.T))
            for mode in star_modes.get(star, []):
                u_frac = mode.projected_u @ inverse_L
                f_mode = _structure_factors(hp, g[members], analysis.ref_frac + u_frac,
                                            core_factors) / analysis.core_size
                weights.append((mode.irrep_name, float(np.sum(np.abs(f_mode) ** 2))))
                # linear term of the structure factor in the displacements
                f_linear = np.sum(phases * (2j * np.pi * (hp @ u_frac.T)), axis=1)
                value = float(np.sum(np.abs(f_linear / analysis.core_size) ** 2))
                linear.append((mode.irrep_name, value))
                key = (star, mode.irrep_name)
                first_order_max[key] = max(first_order_max.get(key, 0.0), value)
        records.append(dict(
            two_theta=float(two_theta[representative]), d=float(1.0 / g[representative]),
            hkl_child=hkl_child, multiplicity=len(members),
            hkl_parent=parent_conv[representative], kname=kname, kvec=kvec,
            star=star, k_label=k_label,
            intensity=0.0 if absent else intensity, intensity_parent=intensity_parent,
            weights=weights, linear=linear, absent=absent,
            equivalents=tuple(parent_conv[j] for j in members)))

    reflections: list[SuperlatticeReflection] = []
    absent_families: list[SuperlatticeReflection] = []
    for record in records:
        weights, linear = record.pop("weights"), record.pop("linear")
        norm = sum(weight for _, weight in weights)
        contributions = () if norm <= 0 else tuple(sorted(
            ((irrep, weight / norm) for irrep, weight in weights), key=lambda item: -item[1]))
        extinct = tuple(irrep for irrep, value in linear
                        if value <= EXTINCT_FIRST_ORDER
                        * first_order_max.get((record["star"], irrep), 0.0))
        target = absent_families if record.pop("absent") else reflections
        target.append(SuperlatticeReflection(
            contributions=contributions, first_order_extinct=extinct, **record))
    reflections.sort(key=lambda r: (round(r.two_theta, 6), r.superlattice, r.hkl_child))
    absent_families.sort(key=lambda r: (round(r.two_theta, 6), r.hkl_child))

    kpoints = _summarize_stars(analysis, reflections, absent_families, star_modes)

    # point lists of the drawn patterns, every line of the radiation
    child_x, child_y, parent_x, parent_y = [], [], [], []
    for _, line_lambda, weight in lines:
        _, _, x, y = _point_intensities(lattice, frac, numbers, kind, line_lambda, window, size)
        child_x.append(x)
        child_y.append(y * scale * weight)
        _, _, x, y = _point_intensities(analysis.L_parent, analysis.parent_positions,
                                        analysis.parent_numbers, kind, line_lambda, window, 1)
        parent_x.append(x)
        parent_y.append(y * scale * weight)

    return SuperlatticePattern(
        parent_file=parent_file, child_file=child_file,
        parent_formula=_formula(analysis.parent_numbers), child_formula=_formula(numbers),
        parent_space_group=f"{analysis.parent_symbol} (No. {analysis.parent_number})",
        child_space_group=f"{analysis.child_symbol} (No. {analysis.child_number})",
        size=size, cell_relation=tuple(tuple(int(v) for v in row) for row in S),
        parent_setting=np.asarray(analysis.A_setting, dtype=float),
        radiation=kind, radiation_name=name, lines=lines, two_theta_range=window,
        reflections=reflections, absent=absent_families, kpoints=kpoints,
        child_peaks=(np.concatenate(child_x), np.concatenate(child_y)),
        parent_peaks=(np.concatenate(parent_x), np.concatenate(parent_y)),
        min_intensity=float(min_intensity),
    )


#: Conditions on the parent indices tried as extinction rules, in order.
_RULES = (
    ("|h| = |k| = |l|", lambda a: abs(a[0]) == abs(a[1]) == abs(a[2])),
    ("|h| = |k|", lambda a: abs(a[0]) == abs(a[1])),
    ("|k| = |l|", lambda a: abs(a[1]) == abs(a[2])),
    ("|h| = |l|", lambda a: abs(a[0]) == abs(a[2])),
    ("h = k", lambda a: a[0] == a[1]),
    ("h = -k", lambda a: a[0] == -a[1]),
    ("k = l", lambda a: a[1] == a[2]),
    ("k = -l", lambda a: a[1] == -a[2]),
    ("h = l", lambda a: a[0] == a[2]),
    ("h = -l", lambda a: a[0] == -a[2]),
    ("h = 0", lambda a: a[0] == 0),
    ("k = 0", lambda a: a[1] == 0),
    ("l = 0", lambda a: a[2] == 0),
)


def _rule(extinct, others) -> str:
    """The first condition of :data:`_RULES` that some member of every
    extinct family fulfils and no member of any other family, or ``""``."""
    def holds(reflection, condition) -> bool:
        return any(condition(hkl) for hkl in (reflection.equivalents or (reflection.hkl_parent,)))

    for text, condition in _RULES:
        if all(holds(r, condition) for r in extinct) and not any(
                holds(r, condition) for r in others):
            return text
    return ""


def _summarize_stars(analysis, reflections, absent, star_modes) -> list[KPointSummary]:
    """One :class:`KPointSummary` per k star with superlattice reflections."""
    def star_text(index) -> str:
        star = analysis.stars[index]
        return f"{star['kname']} {_kvector_label(_star_kvec(star))}"

    summaries = []
    for index, star in enumerate(analysis.stars):
        kname = star["kname"]
        if kname == _GAMMA:
            continue
        members = [i for i, r in enumerate(reflections) if r.star == index]
        missing = [i for i, r in enumerate(absent) if r.star == index]
        if not members and not missing:
            continue
        irreps = tuple((mode.irrep_name, float(mode.amplitude))
                       for mode in star_modes.get(index, []))
        diagnostic = []
        extinctions = []
        everything = [reflections[i] for i in members] + [absent[i] for i in missing]
        for irrep, _ in irreps:
            diagnostic.append((irrep, tuple(
                i for i in members
                if dict(reflections[i].contributions).get(irrep, 0.0) >= DIAGNOSTIC_SHARE)))
            extinct = [r for r in everything if irrep in r.first_order_extinct]
            if extinct:
                others = [r for r in everything if irrep not in r.first_order_extinct]
                extinctions.append((
                    irrep, _rule(extinct, others),
                    tuple(i for i in members if irrep in reflections[i].first_order_extinct),
                    sum(1 for i in missing if irrep in absent[i].first_order_extinct)))
        shared = tuple(
            (mode.irrep_name, star_text(other)) for mode in star_modes.get(index, [])
            for other in _mode_stars(analysis, mode) if other != index)
        summaries.append(KPointSummary(
            kname=kname, kvec=_star_kvec(star), irreps=irreps, reflections=tuple(members),
            diagnostic=tuple(diagnostic), extinctions=tuple(extinctions),
            absent=tuple(missing), star=index, shared=shared))
    return summaries


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def _matrix_rows(matrix) -> str:
    return ", ".join("(" + ", ".join(str(v) for v in row) + ")" for row in matrix)


def _setting_note(pattern: SuperlatticePattern) -> str:
    """How the parent indices relate to the parent cell."""
    A = pattern.parent_setting
    symbol = pattern.parent_space_group.split(" (")[0]
    if np.allclose(A, np.eye(3)):
        return f"parent indices: the cell of {symbol} (primitive = conventional)"
    rows = _matrix_rows(_rational_matrix(A))
    return (f"parent indices: conventional cell of {symbol}, h_conv = h_prim A^-T with A = "
            f"{rows}\n(rows: the parent primitive basis in conventional units)")


def _reflection_list(pattern, indices, limit: int = 8, irrep: str | None = None) -> str:
    """``"(3 1 1)/2 at 37.85, ..."``: the strongest ``limit`` families.

    With ``irrep``, a family at which that irrep is extinct at first order
    is marked ``(higher order)``.
    """
    def item(i) -> str:
        r = pattern.reflections[i]
        mark = " (higher order)" if irrep is not None and irrep in r.first_order_extinct else ""
        return f"{r.hkl_parent_label} at {r.two_theta:.2f}{mark}"

    shown = list(indices)
    if len(shown) > limit:
        shown = sorted(sorted(indices, key=lambda i: -pattern.reflections[i].intensity)[:limit])
    items = [item(i) for i in shown]
    if len(indices) > limit:
        items.append(f"and {len(indices) - limit} more (see the table)")
    return ", ".join(items)


def table_lines(pattern: SuperlatticePattern) -> list[str]:
    """The ``* Reflections *`` table (the :attr:`~SuperlatticePattern.listed`
    families), one string per line."""
    listed = pattern.listed
    child_width = max([16] + [len(r.hkl_child_label) for r in listed])
    parent_width = max([14] + [len(r.hkl_parent_label) for r in listed])
    k_width = max([4] + [len(r.k_label or r.kname) for r in listed])
    lines = [f"{'2theta':>9} {'d (A)':>9}  {'(h k l)_child':<{child_width}} "
             f"{'(h k l)_parent':<{parent_width}} {'k':<{k_width}} {'I_child':>9} "
             f"{'I_parent':>9}  contributions"]
    for r in listed:
        parent = "-" if r.intensity_parent is None else f"{r.intensity_parent:.3f}"
        lines.append(
            f"{r.two_theta:>9.4f} {r.d:>9.5f}  {r.hkl_child_label:<{child_width}} "
            f"{r.hkl_parent_label:<{parent_width}} {r.k_label or r.kname:<{k_width}} "
            f"{r.intensity:>9.3f} {parent:>9}  {r.contributions_label}")
    return lines


def format_superlattice_pattern(pattern: SuperlatticePattern, files=()) -> str:
    """The report ``crystod-xrd --parent`` prints.

    Args:
        pattern: From :func:`superlattice_reflections`.
        files: Paths of the files written (listed in ``* Output files *``).

    Returns:
        The report, ``* ... *`` blocks separated by blank lines.
    """
    out = ["", "* Structures *",
           f"parent: {os.path.basename(pattern.parent_file)}  {pattern.parent_formula}, "
           f"{pattern.parent_space_group}",
           f"child : {os.path.basename(pattern.child_file)}  {pattern.child_formula}, "
           f"{pattern.child_space_group}",
           "child primitive basis in parent primitive units (rows): "
           + _matrix_rows(pattern.cell_relation),
           f"primitive cell multiplication: {pattern.size}",
           _setting_note(pattern),
           f"child indices: conventional cell of {pattern.child_space_group.split(' (')[0]}"]

    out += ["", "* Radiation *"]
    if pattern.radiation == "neutron":
        out.append(f"{pattern.radiation_label} (nuclear scattering only; no magnetic "
                   "scattering, no Debye-Waller factor)")
    else:
        out.append(f"{pattern.radiation_name}: {pattern.radiation_label} (no Debye-Waller factor)")
        if len(pattern.lines) > 1:
            out.append("the table lists the " + pattern.lines[0][0] + " positions; the drawn "
                       "pattern adds " + ", ".join(line for line, _, _ in pattern.lines[1:])
                       + " at its weight")

    low, high = pattern.two_theta_range
    listed = pattern.listed
    n_super = sum(1 for r in listed if r.superlattice)
    out += ["", f"* Reflections ({len(listed)} families, {n_super} superlattice) "
            f"in {low:g} - {high:g} deg *"]
    out.extend(table_lines(pattern))
    out.append("(I_child: strongest child family = 100; I_parent: the parent reference "
               "structure on the same scale;\n contributions: share of |F|^2 among the irreps "
               "of the k star, each with its own displacements alone)")
    hidden = len(pattern.reflections) - len(listed)
    if hidden:
        out.append(f"{hidden} families below --min-intensity {pattern.min_intensity:g} are not "
                   "listed in the table; the k-point summary below includes them")

    out += ["", "* Superlattice reflections by k point *"]
    if not pattern.kpoints:
        out.append("no superlattice reflections (the child keeps the parent lattice)")
    for summary in pattern.kpoints:
        irreps = ", ".join(f"{irrep} ({amplitude:.4f} A)" for irrep, amplitude in summary.irreps)
        absent = (f" (+ {len(summary.absent)} absent in the child)" if summary.absent else "")
        out.append(f"{summary.kname} {_kvector_label(summary.kvec)}: "
                   f"{len(summary.reflections)} families{absent}; irreps of the star: "
                   f"{irreps or '-'}")
        if not summary.irreps:
            out.append("  no irrep of this star distorts the child; the intensity comes from "
                       "higher-order terms of the other stars (harmonics, couplings)")
        elif len(summary.irreps) == 1:
            out.append(f"  all families from {summary.irreps[0][0]} (the only irrep of the star)")
        else:
            for irrep, indices in summary.diagnostic:
                text = _reflection_list(pattern, indices, irrep=irrep) if indices else "none"
                out.append(f"  diagnostic for {irrep}: {text}")
        for irrep, rule, indices, n_absent in summary.extinctions:
            where = f"for {rule}" if rule else "at"
            text = _reflection_list(pattern, indices) if indices else ""
            if n_absent:
                text += ("; " if text else "") + f"{n_absent} families absent in the child"
            out.append(f"  {irrep} extinct at first order {where}: {text}")
    if pattern.kpoints:
        out.append(f"(diagnostic: share >= {DIAGNOSTIC_SHARE:.0%}; extinct at first order: the "
                   "term of the irrep's structure factor\n linear in its displacements vanishes; "
                   "positions in degrees 2theta)")
        if any("(higher order)" in line for line in out if line.startswith("  diagnostic for")):
            out.append("((higher order): a diagnostic family at which its irrep is extinct at "
                       "first order;\n its intensity comes from higher powers of the "
                       "displacements of that irrep)")
        # complex pairs spanning a star and its separate -k star, grouped
        # by the two stars (in the order the summary first meets them)
        spans: dict[frozenset, tuple[tuple[str, str], list[str]]] = {}
        for summary in pattern.kpoints:
            here = f"{summary.kname} {_kvector_label(summary.kvec)}"
            for irrep, other in summary.shared:
                _, irreps = spans.setdefault(frozenset((here, other)), ((here, other), []))
                if irrep not in irreps:
                    irreps.append(irrep)
        for (first, second), irreps in spans.values():
            what = ("one complex-conjugate pair" if len(irreps) == 1
                    else "complex-conjugate pairs")
            each = "" if len(irreps) == 1 else "each "
            out.append(f"({', '.join(irreps)}: {what} over the star {first} and its -k star "
                       f"{second};\n {each}listed under both stars with the one amplitude of "
                       "the pair)")

    if files:
        out += ["", "* Output files *"]
        out.extend(f"  {path}" for path in files)
    out += ["", "Pairing and irreps: the symmetry-mode analysis of crystod-group "
            "--supergroup-cif;", "structure factors: pymatgen scattering tables (S. P. Ong et "
            "al., Comput. Mater. Sci. 68, 314 (2013)).", ""]
    return "\n".join(out)


def write_superlattice_table(pattern: SuperlatticePattern, path: str) -> str:
    """Write the reflection table as comma-separated text.

    Columns: ``two_theta, d, h_child, k_child, l_child, multiplicity,
    h_parent, k_parent, l_parent, k point, I_child, I_parent,
    contributions``; the parent indices are fractions (``1/2``), the k
    point is :attr:`SuperlatticeReflection.k_label` with spaces for its
    commas (``DT(0 1/3 0)``), ``I_parent`` is empty for a superlattice
    reflection, the contributions are ``irrep share`` pairs separated by
    ``;`` (empty for a fundamental reflection and for a star along which the
    child is not distorted).  Only the :attr:`~SuperlatticePattern.listed`
    families are written.

    Args:
        pattern: From :func:`superlattice_reflections`.
        path: Output file name.

    Returns:
        ``path``.
    """
    lines = [
        f"# reflections of {os.path.basename(pattern.child_file)} ({pattern.child_formula}, "
        f"{pattern.child_space_group}) indexed on {os.path.basename(pattern.parent_file)} "
        f"({pattern.parent_formula}, {pattern.parent_space_group})",
        f"# radiation: {pattern.radiation_label}; 2theta range: {pattern.two_theta_range[0]:g} - "
        f"{pattern.two_theta_range[1]:g} deg; I_child: strongest child family = 100",
        "# two_theta (deg), d (A), h_child, k_child, l_child, multiplicity, h_parent, k_parent, "
        "l_parent, k point, I_child, I_parent, contributions",
    ]
    for r in pattern.listed:
        parent = "" if r.intensity_parent is None else f"{r.intensity_parent:.4f}"
        contributions = "; ".join(f"{irrep} {share:.4f}" for irrep, share in r.contributions)
        k_label = (r.k_label or r.kname).replace(",", " ")
        lines.append(
            f"{r.two_theta:.4f},{r.d:.6f},{r.hkl_child[0]},{r.hkl_child[1]},{r.hkl_child[2]},"
            f"{r.multiplicity},{','.join(str(v) for v in r.hkl_parent)},{k_label},"
            f"{r.intensity:.4f},{parent},{contributions}")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def _broaden(x, positions, heights, profile: str, width: float):
    if len(positions) == 0:
        return np.zeros_like(x)
    delta = x[:, None] - np.asarray(positions)[None, :]
    if profile == "gaussian":
        shapes = np.exp(-delta**2 / (2.0 * width**2)) / (width * np.sqrt(2.0 * np.pi))
    else:
        shapes = (width / np.pi) / (delta**2 + width**2)
    return shapes @ np.asarray(heights)


def plot_superlattice_pattern(
    pattern: SuperlatticePattern,
    path: str,
    profile: str = DEFAULT_PEAK_PROFILE,
    width: float = DEFAULT_WIDTH,
    show: bool = False,
) -> str:
    """Draw the parent and child patterns with the superlattice reflections marked.

    The child pattern (black) is drawn over the parent reference pattern
    (grey), both broadened like ``crystod-xrd`` patterns; tick marks below
    the curves mark the fundamental reflections (grey) and the superlattice
    reflections coloured by k star, and diagnostic reflections carry the
    name of their irrep.

    Args:
        pattern: From :func:`superlattice_reflections`.
        path: Output file; the extension selects the format.
        profile: ``"lorentzian"`` or ``"gaussian"``.
        width: Profile width in degrees.
        show: Open an interactive matplotlib window as well.

    Returns:
        ``path``.
    """
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    low, high = pattern.two_theta_range
    x = np.linspace(low, high, 5000)
    child = _broaden(x, *pattern.child_peaks, profile, width)
    parent = _broaden(x, *pattern.parent_peaks, profile, width)
    top = float(max(child.max(), parent.max(), 1e-12))

    figure, axis = plt.subplots(figsize=(7.5, 4.6))
    axis.plot(x, parent, color="0.6", linewidth=0.8,
              label=f"parent {pattern.parent_formula} {pattern.parent_space_group.split(' (')[0]}")
    axis.plot(x, child, color="black", linewidth=0.8,
              label=f"child {pattern.child_formula} {pattern.child_space_group.split(' (')[0]}")
    fundamental = [r.two_theta for r in pattern.fundamental]
    axis.vlines(fundamental, -top * 0.12, -top * 0.06, color="0.6", linewidth=0.6)
    colors = plt.get_cmap("tab10")
    # labels closer than this (degrees) would overlap; a second row takes
    # them, and a label of the same irrep next to one already drawn is dropped
    spacing = (high - low) / 75.0
    rows: list[list[float]] = [[], []]
    drawn: dict[str, list[float]] = {}
    for n, summary in enumerate(pattern.kpoints):
        color = colors(n % 10)
        positions = [pattern.reflections[i].two_theta for i in summary.reflections]
        axis.vlines(positions, -top * 0.22, -top * 0.14, color=color, linewidth=0.9,
                    label=f"{summary.kname} {_kvector_label(summary.kvec)}")
        if len(summary.irreps) < 2:
            continue  # every reflection of a one-irrep star belongs to that irrep
        for irrep, indices in summary.diagnostic:
            # the strongest diagnostic reflections of each irrep
            for i in sorted(indices, key=lambda i: -pattern.reflections[i].intensity)[
                    :_LABELS_PER_IRREP]:
                x0 = pattern.reflections[i].two_theta
                if any(abs(x0 - x) < spacing for x in drawn.get(irrep, [])):
                    continue
                row = next((r for r, used in enumerate(rows)
                            if all(abs(x0 - x) >= spacing for x in used)), None)
                if row is None:
                    continue
                rows[row].append(x0)
                drawn.setdefault(irrep, []).append(x0)
                axis.text(x0, -top * (0.23 + 0.13 * row), irrep, rotation=90,
                          fontsize=6, color=color, ha="center", va="top")
    axis.set_xlim(low, high)
    axis.set_ylim(-top * (0.49 if rows[1] else 0.36), top * 1.08)
    axis.set_xlabel(r"2$\theta$ (deg.)", fontsize=12)
    axis.set_ylabel("Intensity", fontsize=12)
    axis.tick_params(labelleft=False, labelsize=12)
    radiation = (f"neutron {pattern.lines[0][1]:g} A" if pattern.radiation == "neutron"
                 else pattern.radiation_name)
    axis.set_title(f"{pattern.child_formula}: superlattice reflections on the parent lattice, "
                   f"{radiation}", fontsize=10)
    axis.legend(fontsize=8, loc="upper right")
    figure.tight_layout()
    figure.savefig(path)
    if show:
        plt.show()
    plt.close(figure)
    return path


def _structure_name(path: str) -> str:
    name = os.path.basename(path)
    stem, suffix = os.path.splitext(name)
    return stem if suffix.lower() in _STRUCTURE_SUFFIXES else name


def default_prefix(pattern: SuperlatticePattern) -> str:
    """``XRD_<child>_vs_<parent>_<xraytype>`` or ``ND_<child>_vs_<parent>_<wavelength>A``."""
    child = _structure_name(pattern.child_file)
    parent = _structure_name(pattern.parent_file)
    if pattern.radiation == "neutron":
        return f"ND_{child}_vs_{parent}_{pattern.lines[0][1]:g}A"
    return f"XRD_{child}_vs_{parent}_{pattern.radiation_name}"


def build_parser():
    """The argument parser of the implementation entry point.

    ``crystod-xrd --parent`` (``crystod.cli.xrd``) is the documented front
    end; this parser is what it dispatches to.
    """
    from argparse import ArgumentParser, RawTextHelpFormatter

    parser = ArgumentParser(
        prog="python -m crystod.xrd_superlattice",
        description="Superlattice reflections of a child structure on the parent lattice "
                    "(crystod-xrd --parent).",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("--parent", required=True, help="parent structure (CIF or POSCAR)")
    parser.add_argument("--child", required=True, help="child structure (CIF or POSCAR)")
    parser.add_argument("--radiation", default=DEFAULT_RADIATION,
                        help="radiation: " + ", ".join(RADIATIONS))
    parser.add_argument("--xraytype", default=None,
                        help="X-ray line or doublet: " + ", ".join(XRAY_TYPES))
    parser.add_argument("--wavelength", type=float, default=None,
                        help="neutron wavelength in Angstrom")
    parser.add_argument("--peak-profile", default=DEFAULT_PEAK_PROFILE,
                        help="broadening profile: " + ", ".join(PEAK_PROFILES))
    parser.add_argument("--two-theta", nargs=2, type=float,
                        default=list(DEFAULT_TWO_THETA_RANGE), metavar=("MIN", "MAX"))
    parser.add_argument("--width", type=float, default=DEFAULT_WIDTH)
    parser.add_argument("--min-intensity", type=float, default=0.0)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--output", default=None, metavar="PREFIX")
    parser.add_argument("--show", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point behind ``crystod-xrd --parent``."""
    args = build_parser().parse_args(argv)
    profile = args.peak_profile.strip().lower()
    if profile not in PEAK_PROFILES:
        raise SystemExit(f"ERROR: unknown peak profile '{args.peak_profile}'; choose one of "
                         f"{', '.join(PEAK_PROFILES)}.")
    if args.width <= 0:
        raise SystemExit("ERROR: the peak width must be positive.")
    radiation_lines(args.radiation, args.xraytype, args.wavelength)
    pattern = superlattice_reflections(
        args.parent, args.child, radiation=args.radiation,
        xray_type=args.xraytype or DEFAULT_XRAY_TYPE, wavelength=args.wavelength,
        two_theta_range=tuple(args.two_theta), tolerance=args.tolerance,
        min_intensity=args.min_intensity)
    prefix = args.output or default_prefix(pattern)
    table = write_superlattice_table(pattern, prefix + ".txt")
    figure = plot_superlattice_pattern(pattern, prefix + ".pdf", profile, args.width,
                                       show=args.show)
    files = [f"Reflection table written to: {table}",
             f"Patterns ({profile} profile, width {args.width:g} deg) written to: {figure}"]
    print(format_superlattice_pattern(pattern, files))


if __name__ == "__main__":
    main()
