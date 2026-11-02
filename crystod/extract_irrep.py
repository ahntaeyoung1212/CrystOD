"""Band irreps of one finished VASP run, straight from its WAVECAR.

``crystod --extract-irrep [PATH]`` labels the electronic bands of a single
VASP run with the irreducible representations of the little groups of its k
points -- no crystal-orbital diagram, no sublattice runs, no PROCAR.  The
plane-wave coefficients of the requested k points are read from the WAVECAR
(:class:`crystod.vasp_wavecar.Wavecar`, one k point at a time, so a WAVECAR
of any size is labelled in seconds) and decomposed by
:func:`crystod.wavecar_irreps.irreps_at_kpoint`, the labelling engine of
``crystod --diagram --vasp --vasp-engine overlap``: ISO-IR labels for scalar
runs (``GM5+``, ``X3-``, ``R4-``), the double-valued ``-K<n><p>`` names
(``-R6+``) for spin-orbit (vasp_ncl) runs, in the frame of the run's own cell.

What is analysed
----------------
* The structure is the run's POSCAR (``-c`` names another file; it must be
  the same cell, since the coefficients refer to the run's own origin).
* k points: by default every special point of the space group (ISO-IR,
  :func:`crystod.crystal_orbital_overlap.special_kpoints_of_run`) that the
  WAVECAR holds, on any arm of its star and modulo reciprocal-lattice
  vectors; a point that occurs several times (a band path through Gamma
  twice) is analysed at its first occurrence and the other indices are
  listed.  ``kpoints=[...]`` restricts the analysis to labels or fractional
  coordinates; ``kpoints="all"`` (``--all-kpoints``) labels every k point
  of the file, lines and general points included (named by the ISO-IR k
  label of the little group: ``DT``, ``SM``, ``GP`` ...).
* Energies are relative to the valence-band maximum (``zero="vbm"``: the
  highest eigenvalue with occupation above 1/2 over every k point and spin
  channel of the file), to the Fermi level (``"efermi"``, from OUTCAR, else
  vasprun.xml, DOSCAR or the WAVECAR header), or raw.
* Spin: an ``ISPIN = 2`` run gives one table per spin channel; spin-orbit
  runs are recognised from the WAVECAR (spinor records).

Levels
------
Bands are grouped into levels by ``degeneracy_tol`` (default 1 meV, the
overlap engine's) and assessed by the labelling engine: a level whose
multiplicities are not integral absorbs the next levels within 0.3 eV.  The
engine then splits a complete level of several irreps into its finest
complete blocks whenever the bands happen to be pure; ``--extract-irrep``
puts the blocks of one degeneracy window back into one ``A/B`` level
(:func:`_regroup`), so that the degeneracy column is the degeneracy of the
run.  Such a level is classified by Wigner's test of time reversal at k
(:func:`_time_reversal`): a time-reversal pair (``-L4/-L5`` of zincblende
with spin-orbit coupling, ``DT3/DT4`` along Gamma-X of zincblende without
it), one irrep doubled by time reversal (the Kramers pairs ``-GP2/-GP2`` of
a centrosymmetric spin-orbit run at a general point) or an accidental
degeneracy (with the irrep of each band).  What stays unresolved is
reported, not hidden -- in the note column: an incomplete multiplet at the
top of the band list, a time-reversal partner missing from its level, a
numerically split multiplet that was merged, symmetry leakage, and the norm
share of a band whose plane-wave images fell outside the G sphere.

Sublattice runs (``crystod --diagram --vasp-setup``) are labelled too: each
Va point-charge species is a species of its own, as in the overlap engine.

Example (from the top of the source tree):
    >>> from crystod.salc import extract_band_irreps
    >>> result = extract_band_irreps("example/03_hybridization/vasp_wavecar/SrTiO3")
    >>> [(lv.bands, lv.irrep, lv.mulliken) for lv in result.kpoint("GM").levels[7:9]]
    [([17, 18, 19], 'GM5+', 'T2g'), ([20, 21], 'GM3+', 'Eg')]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import warnings
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from .wavecar_irreps import DEGENERACY_SEED_EV

__all__ = [
    "DEFAULT_DEGENERACY_TOL",
    "ZERO_MODES",
    "SPIN_CHOICES",
    "BandLevel",
    "KPointIrreps",
    "BandEdges",
    "BandIrreps",
    "extract_band_irreps",
    "format_band_irreps",
    "write_band_irreps_yaml",
    "main",
]

#: Default seed window (eV) of the level clustering: the overlap engine's.
DEFAULT_DEGENERACY_TOL = DEGENERACY_SEED_EV

#: Energy zeros of ``--zero`` (``vbm`` is the default).
ZERO_MODES = ("vbm", "efermi", "raw")

#: Spin channels of ``--spin`` (``both`` is the default).
SPIN_CHOICES = ("up", "down", "both")

_PURPOSE = "crystod --extract-irrep"
_GAMMA_ALIASES = {"GAMMA": "GM", "G": "GM", "Γ": "GM"}
# notes: symmetry leakage and plane-wave norm lost under the little group
# are reported above these shares
_LEAKAGE_NOTE = 1.0e-3
_LOST_NOTE = 1.0e-3
# a band is occupied above this occupation (the --vasp-zero vbm rule)
_OCCUPIED = 0.5
# eigenvalues this close (eV) to the band edge count as the edge when its k
# point is named (the copies of a special point on a band path)
_EDGE_TIE = 1.0e-4
# occupations strictly between these mark a partially filled band
_PARTIAL = (0.01, 0.99)
_ZERO_TEXT = {"vbm": "E - E_VBM", "efermi": "E - E_F", "raw": "E (raw)"}


# ===================================================================== data
@dataclass
class BandLevel:
    """One level (a set of degenerate bands) at one k point.

    Attributes:
        level: Level number at this k point, 1-based, in energy order over
            all bands of the file (a ``window`` hides levels but keeps the
            numbers).
        bands: Band indices, 1-based.
        energy: Mean energy in eV relative to the energy zero.
        energy_raw: Mean eigenvalue in eV (the run's own scale).
        degeneracy: Number of bands.
        irrep: ISO-IR label (``GM5+``), ``A/B`` for degenerate irreps, or
            ``""`` when the level is not a complete multiplet.
        mulliken: Mulliken symbol(s) at Gamma (``T2g``, ``T1u/T2u``), else
            ``None``.
        note: What the labelling engine flagged (empty when nothing).
        occupation: Mean occupation of the bands (0..1).
        irreps: ``[(name, count), ...]`` of a complete level.
        multiplicities: Multiplicities of the irreps (values above 1e-3),
            the evidence for an unassigned level.
        spread: Energy spread of the bands in eV.
        leakage: Largest norm share a symmetry operation moves out of the
            level.
        lost: Largest norm share of a band whose plane-wave images fell
            outside the G sphere.
        parts: ``[(bands, irrep), ...]`` when the labelling engine resolved
            the bands of a level of several irreps one by one (pure states of
            a time-reversal pair, an accidental degeneracy inside the
            degeneracy window); else empty.
    """

    level: int
    bands: list
    energy: float
    energy_raw: float
    degeneracy: int
    irrep: str
    mulliken: str | None
    note: str
    occupation: float
    irreps: list = field(default_factory=list)
    multiplicities: dict = field(default_factory=dict)
    spread: float = 0.0
    leakage: float = 0.0
    lost: float = 0.0
    parts: list = field(default_factory=list)

    @property
    def label(self) -> str:
        """The irrep column: ``GM4- (T1u)``, ``R4+``, or ``-`` (unassigned)."""
        if not self.irrep:
            return "-"
        return f"{self.irrep} ({self.mulliken})" if self.mulliken else self.irrep

    @property
    def occupied(self) -> bool:
        """Whether the level is occupied (mean occupation above 1/2)."""
        return self.occupation > _OCCUPIED


@dataclass
class KPointIrreps:
    """The levels of one k point (and one spin channel).

    Attributes:
        label: Special-point name (``GM``), else the ISO-IR k label of the
            little group (``DT``, ``GP``), else ``k``.
        kpoint: Fractional k point as stored in the WAVECAR.
        index: 1-based index in the WAVECAR's k list.
        little_group: Space-group type of the little group, ``P4/mmm (123)``.
        k_label: ISO-IR k label of the irreps.
        special: Whether k lies in the star of a special point.
        copies: Other 1-based indices of the same special point.
        spin: ``"up"`` / ``"down"`` for an ``ISPIN = 2`` run, else ``None``.
        levels: The :class:`BandLevel` list, in energy order.
    """

    label: str
    kpoint: list
    index: int
    little_group: str
    k_label: str
    special: bool
    copies: list
    spin: str | None
    levels: list

    @property
    def title(self) -> str:
        """``GM [0, 0, 0]`` (and ``, spin up`` for a spin channel)."""
        text = f"{self.label} {_fraction_vector(self.kpoint)}"
        return text + (f", spin {self.spin}" if self.spin else "")


@dataclass
class BandEdges:
    """Highest occupied and lowest empty level of one k point and spin.

    Attributes:
        label: The k point's label.
        index: Its 1-based index in the WAVECAR.
        spin: Spin channel or ``None``.
        vbm: The highest occupied :class:`BandLevel`, or ``None``.
        cbm: The lowest empty :class:`BandLevel`, or ``None``.
        direct_gap: ``cbm.energy - vbm.energy`` in eV, or ``None``.
    """

    label: str
    index: int
    spin: str | None
    vbm: BandLevel | None
    cbm: BandLevel | None
    direct_gap: float | None


@dataclass
class BandIrreps:
    """Everything ``crystod --extract-irrep`` prints and writes.

    Attributes:
        directory: The run directory.
        wavecar: The WAVECAR file.
        structure: The structure file the labels refer to.
        formula: Chemical formula of the cell (a Va point-charge species of a
            sublattice run in parentheses: ``SrTi(Va2-)3``).
        n_atoms: Atoms in the cell (Va point charges included).
        nbands: Bands per k point.
        nkpts: k points in the file.
        ispin: 1 or 2.
        spinor: Spin-orbit (vasp_ncl) run.
        encut: Plane-wave cutoff in eV (WAVECAR header).
        precision: ``complex64`` or ``complex128`` coefficients.
        efermi: Fermi level in eV.
        efermi_source: Where it was read (``OUTCAR``, ``vasprun.xml``,
            ``DOSCAR`` or ``WAVECAR header``).
        vbm: Valence-band maximum in eV (raw), highest occupied eigenvalue
            over every k point and spin channel of the file.
        vbm_at: ``(k index, band, spin)`` of the VBM (1-based index and
            band, spin channel 0 or 1).
        vbm_kpoint: Fractional k point of the VBM.
        vbm_label: Its special-point name, or ``""``.
        cbm: Lowest empty eigenvalue in eV (raw), same range.
        cbm_at: ``(k index, band, spin)`` of the CBM.
        cbm_kpoint: Fractional k point of the CBM.
        cbm_label: Its special-point name, or ``""``.
        metallic: Partially occupied bands exist (no gap is quoted).
        zero: ``"vbm"``, ``"efermi"`` or ``"raw"``.
        zero_energy: The eV subtracted from the raw eigenvalues.
        space_group: Hermann-Mauguin symbol.
        space_group_number: International number.
        symprec: Symmetry tolerance.
        degeneracy_tol: Seed window of the level clustering in eV.
        window: ``(emin, emax)`` eV relative to the zero, or ``None``.
        selection: ``"special"``, ``"requested"`` or ``"all"``.
        kpoints: The :class:`KPointIrreps` list (k outer, spin inner).
        missing: Special points ``(name, k)`` absent from the file
            (default selection only).
        edges: The :class:`BandEdges` of every analysed k point and spin.
    """

    directory: str
    wavecar: str
    structure: str
    formula: str
    n_atoms: int
    nbands: int
    nkpts: int
    ispin: int
    spinor: bool
    encut: float
    precision: str
    efermi: float
    efermi_source: str
    vbm: float | None
    vbm_at: tuple | None
    vbm_kpoint: list | None
    vbm_label: str
    cbm: float | None
    cbm_at: tuple | None
    cbm_kpoint: list | None
    cbm_label: str
    metallic: bool
    zero: str
    zero_energy: float
    space_group: str
    space_group_number: int
    symprec: float
    degeneracy_tol: float
    window: tuple | None
    selection: str
    kpoints: list
    missing: list
    edges: list

    @property
    def gap(self) -> float | None:
        """CBM - VBM over the whole file in eV (``None`` for a metal)."""
        if self.metallic or self.vbm is None or self.cbm is None:
            return None
        return self.cbm - self.vbm

    def kpoint(self, label: str, spin: str | None = None) -> KPointIrreps:
        """The analysed k point named ``label`` (first match).

        Raises:
            KeyError: No such k point.
        """
        for entry in self.kpoints:
            if entry.label == label and (spin is None or entry.spin == spin):
                return entry
        raise KeyError(label)

    def to_dict(self) -> dict:
        """The content of the YAML file (``-o``) as plain Python data.

        Floats are rounded to six decimals, so that ``yaml.safe_load`` of the
        written file equals this dictionary.
        """
        return _result_dict(self)


# ================================================================= helpers
def _fraction_text(value: float) -> str:
    """``1/2`` for 0.5; six decimals when no small fraction fits."""
    fraction = Fraction(float(value)).limit_denominator(100)
    if abs(float(fraction) - float(value)) < 1e-6:
        return str(fraction)
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def _fraction_vector(k) -> str:
    """``[0, 1/2, 0]``."""
    return "[" + ", ".join(_fraction_text(x) for x in k) + "]"


def _stored_vector(k) -> str:
    """``[ 0.0000000,  0.5000000,  0.0000000]``: k as stored in the WAVECAR."""
    return "[" + ", ".join(f"{float(x) + 0.0:10.7f}" for x in k) + "]"


def _band_range(bands) -> str:
    """``7-9`` (1-based, consecutive), or ``7``."""
    bands = list(bands)
    if len(bands) == 1:
        return str(bands[0])
    if bands == list(range(bands[0], bands[-1] + 1)):
        return f"{bands[0]}-{bands[-1]}"
    return ",".join(str(b) for b in bands)


def _round(value, digits: int = 6):
    if value is None:
        return None
    return round(float(value), digits) + 0.0


def _formula(symbols) -> str:
    """``SrTiO3``; a Va point-charge species in parentheses: ``SrTi(Va2-)3``."""
    counts: dict = {}
    for symbol in symbols:
        counts[symbol] = counts.get(symbol, 0) + 1
    return "".join(f"{f'({s})' if s.startswith('Va') and len(s) > 2 else s}"
                   f"{n if n > 1 else ''}" for s, n in counts.items())


def _species_numbers(symbols) -> list:
    ids: dict = {}
    return [ids.setdefault(symbol, len(ids) + 1) for symbol in symbols]


def _resolve_run(path: str) -> tuple:
    """``(run directory, WAVECAR path)`` of ``PATH`` (file or directory).

    Raises:
        SystemExit: No WAVECAR at ``PATH``.
    """
    path = os.path.expanduser(str(path))
    if os.path.isdir(path):
        wavecar = os.path.join(path, "WAVECAR")
        if not os.path.isfile(wavecar):
            raise SystemExit(f"ERROR: no WAVECAR in {path} (crystod --extract-irrep reads "
                             "the wavefunctions: rerun with LWAVE = .TRUE., or give the "
                             "WAVECAR file itself).")
        return os.path.abspath(path), os.path.abspath(wavecar)
    if os.path.isfile(path):
        return os.path.dirname(os.path.abspath(path)), os.path.abspath(path)
    raise SystemExit(f"ERROR: no WAVECAR at {path} (give a WAVECAR file or the run "
                     "directory that holds one).")


def _read_efermi(directory: str, header_value: float) -> tuple:
    """``(E_fermi, source)``: OUTCAR, else vasprun.xml, DOSCAR, the WAVECAR."""
    from .vasp_io import read_outcar, resolve_vasp_file

    outcar = resolve_vasp_file(os.path.join(directory, "OUTCAR"))
    if outcar:
        try:
            value = read_outcar(outcar).get("efermi")
        except (OSError, ValueError, UnicodeDecodeError):
            value = None
        if value is not None:
            return float(value), "OUTCAR"
    vasprun = resolve_vasp_file(os.path.join(directory, "vasprun.xml"))
    if vasprun:
        from .vasp_io import open_vasp_file

        pattern = re.compile(r'<i name="efermi">\s*([-+0-9.eEdD]+)\s*</i>')
        value = None
        try:
            with open_vasp_file(vasprun) as handle:
                for line in handle:
                    match = pattern.search(line)
                    if match:
                        value = float(match.group(1).replace("D", "E").replace("d", "e"))
        except (OSError, ValueError, UnicodeDecodeError):
            value = None
        if value is not None:
            return value, "vasprun.xml"
    doscar = os.path.join(directory, "DOSCAR")
    if os.path.isfile(doscar):
        try:
            with open(doscar) as handle:
                lines = [handle.readline() for _ in range(6)]
            return float(lines[5].split()[3]), "DOSCAR"
        except (OSError, ValueError, IndexError, UnicodeDecodeError):
            pass
    return float(header_value), "WAVECAR header"


def _read_structure(directory: str, cell_path: str | None, wav):
    """The structure the labels refer to, checked against the WAVECAR's cell.

    Returns:
        ``(VaspStructure, path)``.

    Raises:
        SystemExit: No POSCAR, a cell that is not the WAVECAR's, or a ``-c``
            file in another setting than the run's POSCAR.
    """
    from .crystal_orbital_overlap import _same_setting
    from .vasp_wavecar import read_run_structure

    own = os.path.join(directory, "POSCAR")
    if cell_path is None:
        if not os.path.isfile(own):
            raise SystemExit(f"ERROR: no POSCAR in {directory}: the irreps need the "
                             "run's structure (give it with -c FILE; CONTCAR is not used).")
        path = own
    else:
        if not os.path.isfile(cell_path):
            raise SystemExit(f"ERROR: no structure file {cell_path} (-c).")
        path = cell_path
    structure = read_run_structure(path)
    lattice = np.asarray(structure.lattice, float)
    if not np.allclose(lattice, np.asarray(wav.lattice, float), atol=1e-4):
        raise SystemExit(
            f"ERROR: the cell of {path} is not the cell of {wav.path} (lattice vectors "
            "differ): the labels need the structure the run was done with.")
    if cell_path is not None and os.path.isfile(own) \
            and os.path.abspath(own) != os.path.abspath(cell_path):
        if not _same_setting(_phonopy_cell(structure), _phonopy_cell(read_run_structure(own))):
            raise SystemExit(
                f"ERROR: -c {cell_path} and {own} are not the same cell and origin: the "
                "plane-wave coefficients refer to the run's own setting, so the labels can "
                "only be taken in it (leave -c out).")
    return structure, path


# stand-in elements of the Va point-charge species in the symmetry cell
_VA_STAND_INS = ("He", "Ne", "Ar", "Kr", "Xe", "Rn", "Og", "Ts", "Lv", "Mc", "Fl", "Nh")


def _symmetry_symbols(symbols) -> list:
    """Chemical symbols for the symmetry cell: a stand-in element per Va species.

    A CrystOD point-charge species (``Va2-``, ``Va4+``: the sublattice runs of
    ``crystod --diagram --vasp-setup``) is a species of its own, as in the
    labelling engine (:func:`_species_numbers`), but phonopy and spglib need
    chemical symbols: each Va species gets a distinct element that the cell
    does not contain.  Only the partition into species matters for the
    symmetry, so the space group, the special points, the ISO-IR frame and
    the Mulliken symbols are those of the run.

    Raises:
        SystemExit: More Va species than stand-ins (never in practice).
    """
    from .vasp_io import parse_va_species

    symbols = list(symbols)
    free = [element for element in _VA_STAND_INS if element not in symbols]
    stand_in: dict = {}
    for symbol in symbols:
        if parse_va_species(symbol) is not None and symbol not in stand_in:
            if not free:
                raise SystemExit("ERROR: too many Va point-charge species in the POSCAR.")
            stand_in[symbol] = free.pop(0)
    return [stand_in.get(symbol, symbol) for symbol in symbols]


def _phonopy_cell(structure):
    from phonopy.structure.atoms import PhonopyAtoms

    return PhonopyAtoms(symbols=_symmetry_symbols(structure.symbols),
                        cell=np.asarray(structure.lattice, float),
                        scaled_positions=np.asarray(structure.positions, float))


def _space_group(cell, symprec: float) -> tuple:
    """``(symbol, number)`` of the cell (spglib)."""
    import spglib

    from .runtime_compat import SymmetryDatasetAdapter, get_scaled_positions

    spg = (np.asarray(cell.cell, float), np.asarray(get_scaled_positions(cell), float),
           np.asarray(cell.numbers, int))
    dataset = spglib.get_symmetry_dataset(spg, symprec=symprec)
    if dataset is None:
        raise SystemExit("ERROR: spglib finds no symmetry for the run's structure "
                         "(try a larger --tolerance).")
    dataset = SymmetryDatasetAdapter(dataset)
    return str(dataset["international"]), int(dataset["number"])


def _little_group_symbol(group) -> str:
    """``P4/mmm (123)``: the space-group type of the little-group operations."""
    try:
        from spglib import get_spacegroup_type_from_symmetry

        found = get_spacegroup_type_from_symmetry(
            np.asarray(group.rotations, dtype="intc"),
            np.asarray(group.translations, dtype="double"))
        symbol = getattr(found, "international_short", None) or found["international_short"]
        number = getattr(found, "number", None) or found["number"]
        return f"{symbol} ({number})"
    except Exception:  # an annotation only: never fail the run for it
        return f"order {group.order}"


def _parse_kpoint_request(tokens) -> list:
    """``["GM", "0", "1/2", "0", "R"]`` -> ``["GM", (0, 0.5, 0), "R"]``.

    Labels and fractional coordinates may be mixed; a coordinate is three
    numbers (fractions allowed), or one ``k1,k2,k3`` token.

    Raises:
        SystemExit: Incomplete coordinates.
    """
    def number(text):
        try:
            return float(Fraction(str(text).strip()))
        except (ValueError, ZeroDivisionError):
            return None

    out = []
    items = list(tokens)
    i = 0
    while i < len(items):
        item = items[i]
        if not isinstance(item, str):
            values = [float(x) for x in np.asarray(item, float).reshape(-1)]
            if len(values) != 3:
                raise SystemExit(f"ERROR: --kpoint {item!r}: a k point has three "
                                 "fractional coordinates.")
            out.append(tuple(values))
            i += 1
            continue
        parts = [p for p in re.split(r"[,\s]+", item.strip()) if p]
        if len(parts) == 3 and all(number(p) is not None for p in parts):
            out.append(tuple(number(p) for p in parts))
            i += 1
            continue
        if number(item) is not None:
            values = [number(t) for t in items[i:i + 3]]
            if len(values) < 3 or any(v is None for v in values):
                raise SystemExit(
                    f"ERROR: --kpoint {' '.join(map(str, items[i:i + 3]))}: give three "
                    "fractional coordinates (e.g. 0 1/2 0) or a special-point label.")
            out.append(tuple(values))
            i += 3
            continue
        out.append(item.strip())
        i += 1
    return out


def _star_arms(k, rotations) -> list:
    """The arms ``k R`` of the star of ``k`` (modulo reciprocal-lattice vectors)."""
    arms = [np.asarray(k, float)]
    for rotation in rotations:
        arm = arms[0] @ np.asarray(rotation, float)
        if not any(np.all(np.abs((arm - a) - np.rint(arm - a)) < 1e-5) for a in arms):
            arms.append(arm)
    return arms


def _in_star(kpoints, k, rotations) -> list:
    """0-based indices of every k point of the list in the star of ``k``."""
    kpoints = np.asarray(kpoints, float)
    hits = set()
    for arm in _star_arms(k, rotations):
        difference = kpoints - arm
        hits.update(int(i) for i in np.where(
            np.all(np.abs(difference - np.rint(difference)) < 1e-5, axis=1))[0])
    return sorted(hits)


def _select(klist, special, rotations, request) -> tuple:
    """The k points to analyse.

    Returns:
        ``(targets, missing, selection)``: ``targets`` is a list of
        ``(label or None, index, special, copies)`` (0-based indices; label
        ``None`` = to be named from the little group), ``missing`` the special
        points ``(name, k)`` absent from the file.

    Raises:
        SystemExit: An unknown label, a requested point not in the file, or
            no special point in the file at all.
    """
    from .crystal_orbital_overlap import find_in_star

    def star_name(index):
        for name, k in special:
            if index in _in_star(klist, k, rotations):
                return name
        return None

    def available():
        names = []
        for name, k in special:
            index, _ = find_in_star(klist, k, rotations)
            if index is not None:
                names.append(f"{name} (k#{index + 1})")
        return ", ".join(names) or "none"

    if request == "all":
        return ([(star_name(i), i, star_name(i) is not None, []) for i in range(len(klist))],
                [], "all")
    if request is None:
        targets, missing = [], []
        for name, k in special:
            index, _ = find_in_star(klist, k, rotations)
            if index is None:
                missing.append((name, list(k)))
                continue
            copies = [i for i in _in_star(klist, k, rotations) if i != index]
            targets.append((name, index, True, copies))
        if not targets:
            raise SystemExit(
                "ERROR: none of the special points of the space group ("
                + ", ".join(f"{name} {_fraction_vector(k)}" for name, k in special)
                + f") is in the k list of the WAVECAR ({len(klist)} k points); label every "
                "k point with --all-kpoints, or pick points with --kpoint K1 K2 ...")
        return targets, missing, "special"
    targets = []
    names = {name.upper(): (name, k) for name, k in special}
    for item in _parse_kpoint_request(request):
        if isinstance(item, str):
            key = _GAMMA_ALIASES.get(item.upper(), item.upper())
            if key not in names:
                raise SystemExit(
                    f"ERROR: --kpoint {item}: not a special point of this space group "
                    f"(its special points: {', '.join(n for n, _ in special)}; fractional "
                    "coordinates are accepted too, or --all-kpoints for every k point).")
            name, k = names[key]
            index, _ = find_in_star(klist, k, rotations)
            if index is None:
                raise SystemExit(
                    f"ERROR: --kpoint {item}: {name} {_fraction_vector(k)} is not in the k "
                    f"list of the WAVECAR (special points in the file: {available()}).")
            copies = [i for i in _in_star(klist, k, rotations) if i != index]
        else:
            k = np.asarray(item, float)
            difference = np.asarray(klist, float) - k
            hit = np.where(np.all(np.abs(difference - np.rint(difference)) < 1e-5,
                                  axis=1))[0]
            if len(hit):
                index = int(hit[0])
            else:
                index, _ = find_in_star(klist, k, rotations)
            if index is None:
                raise SystemExit(
                    f"ERROR: --kpoint {_fraction_vector(k)}: no k point of the WAVECAR "
                    "equals it or another arm of its star (special points in the file: "
                    f"{available()}).")
            name = star_name(index)
            copies = []
        if any(index == t[1] for t in targets):
            continue
        targets.append((name, index, name is not None, copies))
    return targets, [], "requested"


def _mulliken(cell, symprec: float) -> dict:
    """Mulliken symbols of the Gamma irreps ({} when they cannot be found)."""
    try:
        from .phonon_activity import mulliken_symbols

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(io.StringIO()):
                return mulliken_symbols(cell, symprec=symprec)
    except Exception:  # an annotation only
        return {}


def _time_reversal(group, lattice, positions, numbers, symprec: float):
    """What time reversal does to the small irreps at k (Wigner's test).

    Time reversal T, combined with every operation g0 = {R0|v0} of the space
    group that takes k to -k, is an antiunitary symmetry of the states at k.
    It maps a small irrep D onto the irrep with the characters
    conj(chi_D(g0^-1 h g0)) (the conjugated element taken back to the coset
    representative of its rotation with the Bloch phase of the lattice
    vector left over, and for spinors the sign of
    U(g0)^-1 U(h) U(g0) against the SU(2) matrix of that representative), and
    the Herring sum (1/|G_k|) sum_g0 chi_D((T g0)^2), with T^2 = -1 for
    spinors, sorts D into the three cases of Wigner: ``"a"`` (+1, D stays
    alone), ``"b"`` (-1, two copies of D are degenerate: a Kramers doublet
    of one irrep) and ``"c"`` (0, D is degenerate with its partner D' != D:
    a time-reversal pair, such as -L4/-L5 of the zincblende structure with
    spin-orbit coupling).

    Returns:
        ``{name: (case, partner name)}``; ``{}`` when no operation takes k to
        -k (time reversal relates k to another k point only), ``None`` when
        the test cannot be made (it never fails on the groups tested; the
        note then makes no claim).
    """
    from .wavecar_irreps import _space_group as space_group_operations

    try:
        _, rotations, translations = space_group_operations(lattice, positions, numbers,
                                                            symprec)
        k = np.asarray(group.kpoint, float)

        def wrapped(x):
            return np.abs(x - np.rint(x))

        reversing = [i for i, R in enumerate(rotations) if np.all(wrapped(k @ R + k) < 1e-6)]
        if not reversing:
            return {}
        spinor = group.spin_rotations is not None
        if spinor:
            from spgrep.spinor import get_spinor_unitary_rotation
        index = {np.asarray(R, np.int64).tobytes(): j for j, R in enumerate(group.rotations)}
        characters = np.asarray(group.characters, complex)

        def representative(R, t, U):
            """``(j, factor)``: the element {R|t} (U) is factor x element j."""
            j = index[np.asarray(np.rint(R), np.int64).tobytes()]
            d = np.asarray(t, float) - np.asarray(group.translations[j], float)
            if np.any(wrapped(d) > 1e-5):
                raise ValueError("not a little-group element")
            factor = np.exp(-2j * np.pi * float(np.dot(k, np.rint(d))))
            if spinor:
                if np.allclose(U, group.spin_rotations[j], atol=1e-6):
                    pass
                elif np.allclose(U, -group.spin_rotations[j], atol=1e-6):
                    factor = -factor
                else:
                    raise ValueError("SU(2) matrices do not match")
            return j, factor

        def unitary(R):
            return get_spinor_unitary_rotation(np.asarray(lattice, float), R) if spinor else None

        herring = np.zeros(len(characters), complex)
        for i in reversing:
            R0, v0 = rotations[i], translations[i]
            U0 = unitary(R0)
            j, factor = representative(R0 @ R0, R0 @ v0 + v0,
                                       U0 @ U0 if spinor else None)
            herring += factor * characters[:, j]
        herring *= (-1.0 if spinor else 1.0) / group.order
        R0, v0 = rotations[reversing[0]], translations[reversing[0]]
        R0_inverse = np.rint(np.linalg.inv(R0)).astype(np.int64)
        U0 = unitary(R0)
        image = np.zeros_like(characters)
        for i, (S, w) in enumerate(zip(group.rotations, group.translations)):
            U = np.linalg.inv(U0) @ group.spin_rotations[i] @ U0 if spinor else None
            j, factor = representative(R0_inverse @ S @ R0, R0_inverse @ (S @ v0 + w - v0), U)
            image[:, i] = np.conj(factor * characters[:, j])
        out = {}
        for a, name in enumerate(group.names):
            hits = [b for b in range(len(characters))
                    if np.allclose(characters[b], image[a], atol=1e-5)]
            if len(hits) != 1:
                return None
            partner = hits[0]
            if partner != a:
                case = "c"
            elif herring[a].real < -0.5:
                case = "b"
            else:
                case = "a"
            out[name] = (case, group.names[partner])
        return out
    except (KeyError, ValueError, IndexError, np.linalg.LinAlgError):
        return None


def _regroup(levels, energies, tolerance: float, group) -> list:
    """The engine's levels regrouped by the degeneracy window.

    :func:`crystod.wavecar_irreps.irreps_at_kpoint` splits a complete level
    of several irreps into its finest complete blocks whenever the bands
    happen to be pure (a time-reversal pair returned as two pure states,
    two irreps 0.5 meV apart).  The rule of ``--extract-irrep`` is the
    degeneracy window: bands seeded into one level by ``tolerance`` (the
    engine's own seeding, ``_seed_levels``) stay one level.  Blocks that
    share a seed are merged back into one ``A/B`` level, and the blocks are
    kept in ``parts`` (``[(bands, label), ...]``) for the note.  The
    overlap engine's levels are not touched (it calls the engine directly).
    """
    from .wavecar_irreps import _level_name, _seed_levels

    seed_of = {band: i for i, seed in enumerate(_seed_levels(energies, tolerance))
               for band in seed}
    groups: list = []
    for level in levels:
        touched = {seed_of[band] for band in level["bands"]}
        if groups and touched & groups[-1][1]:
            groups[-1][0].append(level)
            groups[-1][1].update(touched)
        else:
            groups.append(([level], set(touched)))
    out = []
    for parts, _ in groups:
        if len(parts) == 1:
            out.append(dict(parts[0], parts=[]))
            continue
        bands = [band for part in parts for band in part["bands"]]
        integral = all(part["integral"] for part in parts)
        counts: dict = {}
        multiplicities: dict = {}
        for part in parts:
            for name, count in part["irreps"]:
                counts[name] = counts.get(name, 0) + int(count)
            for name, value in part["multiplicities"].items():
                multiplicities[name] = round(multiplicities.get(name, 0.0) + value, 4)
        values = np.asarray(energies, float)[bands]
        out.append({
            "bands": bands,
            "energy": float(np.mean(values)),
            "spread": float(values[-1] - values[0]),
            "dim": len(bands),
            "label": (_level_name(group.names, [counts.get(n, 0) for n in group.names])
                      if integral else ""),
            "irreps": [(n, counts[n]) for n in group.names if counts.get(n)] if integral else [],
            "multiplicities": multiplicities,
            "leakage": max(part["leakage"] for part in parts),
            "integral": integral,
            "lost": max(part["lost"] for part in parts),
            "parts": [(list(part["bands"]), part["label"]) for part in parts],
        })
    return out


def _degeneracy(irreps, reversal) -> tuple:
    """``(kind, missing)`` of a complete level of several irreps.

    ``kind``: ``"time-reversal pair"`` (one irrep and its partner, Wigner's
    case c), ``"doubled by time reversal"`` (two copies of a case-b irrep),
    ``"accidental"`` (more than one such unit), or ``"unknown"`` when the
    time-reversal test could not be made.  ``missing``: ``[(irrep,
    partner)]`` whose time-reversal partner is not in the level.
    """
    if reversal is None:
        return "unknown", []
    counts = dict(irreps)
    units, missing, seen = 0, [], set()
    for name, count in irreps:
        if name in seen:
            continue
        case, partner = reversal.get(name, ("a", name))
        seen.update((name, partner))
        if case == "c":
            other = counts.get(partner, 0)
            units += max(count, other)
            if count != other:
                missing.append((name, partner) if count > other else (partner, name))
        elif case == "b":
            units += (count + 1) // 2
            if count % 2:
                missing.append((name, name))
        else:
            units += count
    if sum(counts.values()) > 1 and units == 1 and not missing:
        case = reversal.get(irreps[0][0], ("a",))[0]
        return ("time-reversal pair" if case == "c" else "doubled by time reversal"), []
    return "accidental", missing


def _note(level: dict, n_bands: int, tolerance: float, reversal) -> str:
    """The note column of one level (the labelling engine's assessment).

    ``reversal`` is :func:`_time_reversal` of the k point.
    """
    notes = []
    if not level["integral"]:
        parts = sorted(level["multiplicities"].items(), key=lambda item: -abs(item[1]))[:3]
        evidence = ", ".join(f"{name} {value:.2f}" for name, value in parts) or "none"
        if n_bands - 1 in level["bands"]:
            notes.append(f"incomplete: the band list ends inside this multiplet "
                         f"(multiplicities {evidence})")
        else:
            notes.append(f"not assigned: multiplicities {evidence}, leakage "
                         f"{level['leakage']:.2f}")
    else:
        # alphabetical, as in the A/B label
        names = sorted(name for name, count in level["irreps"] for _ in range(count))
        kind, missing = _degeneracy(level["irreps"], reversal)
        if len(names) > 1:
            text = {"time-reversal pair": "time-reversal pair",
                    "doubled by time reversal": "doubled by time reversal",
                    "accidental": "accidental degeneracy",
                    "unknown": "degenerate irreps"}[kind] + " (" + " + ".join(names) + ")"
            if kind != "time-reversal pair" and kind != "doubled by time reversal" \
                    and level.get("parts"):
                text += "; by band: " + ", ".join(
                    f"{_band_range([b + 1 for b in bands])} {label or '-'}"
                    for bands, label in level["parts"])
                if level["spread"] >= 5e-5:
                    text += f" ({1000 * level['spread']:.1f} meV apart)"
            notes.append(text)
        for name, partner in missing:
            if n_bands - 1 in level["bands"]:
                notes.append(f"incomplete: the band list ends inside a time-reversal "
                             f"{'pair' if name != partner else 'doublet'} ({name} needs "
                             f"{partner})")
            else:
                notes.append(f"time-reversal partner {partner} of {name} not in this "
                             "level (broken time reversal?)")
        if level["spread"] > tolerance:
            notes.append(f"split by {1000 * level['spread']:.1f} meV (merged)")
        if level["leakage"] > _LEAKAGE_NOTE:
            notes.append(f"leakage {level['leakage']:.3f}")
    if level["lost"] > _LOST_NOTE:
        notes.append(f"{100 * level['lost']:.1f}% of a band's norm outside the G sphere "
                     "under symmetry")
    return "; ".join(notes)


def _level_mulliken(names, mulliken) -> str | None:
    if not mulliken:
        return None
    symbols = [mulliken.get(name) for name in names]
    if not symbols or any(symbol is None for symbol in symbols):
        return None
    return "/".join(symbols)


def _special_name(klist, where, special, rotations) -> str:
    """The special-point name of the k point of ``where`` (1-based), or ``""``."""
    if where is None:
        return ""
    for name, k in special:
        if where[0] - 1 in _in_star(klist, k, rotations):
            return name
    return ""


def _edges(entry: KPointIrreps) -> BandEdges:
    occupied = [lv for lv in entry.levels if lv.occupied]
    empty = [lv for lv in entry.levels if not lv.occupied]
    vbm = max(occupied, key=lambda lv: lv.energy) if occupied else None
    cbm = min(empty, key=lambda lv: lv.energy) if empty else None
    gap = cbm.energy - vbm.energy if vbm is not None and cbm is not None else None
    return BandEdges(entry.label, entry.index, entry.spin, vbm, cbm, gap)


# ================================================================== engine
def extract_band_irreps(path: str = ".", kpoints=None, zero: str = "vbm", window=None,
                        spin: str = "both", symprec: float = 1e-5,
                        degeneracy_tol: float | None = None, *,
                        cell: str | None = None) -> BandIrreps:
    """Irreps of the bands of one VASP run, from its WAVECAR.

    The Python form of ``crystod --extract-irrep [PATH]``.

    Args:
        path: The WAVECAR file or the run directory that holds it.
        kpoints: ``None`` (every special point of the space group the file
            holds), ``"all"`` (every k point of the file), or a list of
            special-point labels (``"GM"``, ``"X"``) and fractional
            coordinates (``[0, 0.5, 0]`` or ``"0 1/2 0"``) in the run's
            reciprocal basis.
        zero: Energy zero: ``"vbm"``, ``"efermi"`` or ``"raw"``.
        window: ``(emin, emax)`` in eV relative to the zero: only the levels
            inside are kept (the level numbers stay those of the full list).
        spin: ``"up"``, ``"down"`` or ``"both"`` (``ISPIN = 2`` runs).
        symprec: Symmetry tolerance of spglib.
        degeneracy_tol: Seed window of the level clustering in eV (default
            :data:`DEFAULT_DEGENERACY_TOL`, 1 meV).
        cell: Structure file in place of the run's POSCAR (the same cell and
            origin).

    Returns:
        A :class:`BandIrreps`.

    Raises:
        SystemExit: Every user error, as one ``ERROR:`` line (the
            ``crystod.salc`` namespace raises ``ValueError`` instead).
    """
    from .crystal_orbital_overlap import special_kpoints_of_run
    from .vasp_wavecar import Wavecar
    from .wavecar_irreps import irreps_at_kpoint, little_group_data

    if zero not in ZERO_MODES:
        raise SystemExit(f"ERROR: --zero {zero}: choose one of {', '.join(ZERO_MODES)}.")
    if spin not in SPIN_CHOICES:
        raise SystemExit(f"ERROR: --spin {spin}: choose one of {', '.join(SPIN_CHOICES)}.")
    if window is not None:
        window = (float(window[0]), float(window[1]))
        if not window[0] < window[1]:
            raise SystemExit("ERROR: --window takes EMIN < EMAX (eV relative to the zero).")
    tolerance = DEFAULT_DEGENERACY_TOL if degeneracy_tol is None else float(degeneracy_tol)
    if not tolerance > 0:
        raise SystemExit("ERROR: --degeneracy-tol takes a positive energy in eV.")
    if isinstance(kpoints, str):
        kpoints = "all" if kpoints.lower() == "all" else [kpoints]
    elif kpoints is not None:
        kpoints = list(kpoints)
        if len(kpoints) == 3 and all(isinstance(x, (int, float, np.integer, np.floating))
                                     for x in kpoints):
            kpoints = [kpoints]          # one coordinate triple, [0, 0.5, 0]
        if not kpoints:
            kpoints = None

    directory, wavecar_path = _resolve_run(path)
    try:
        wav = Wavecar(wavecar_path, allow_spin_polarized=True, purpose=_PURPOSE)
    except (ValueError, IndexError, OSError) as error:
        raise SystemExit(f"ERROR: {wavecar_path}: unreadable WAVECAR ({error}).") from None
    with wav:
        try:
            klist = wav.kpoints()
            spinor = wav.spinor
            headers = [[wav.header(ik, isp) for ik in range(wav.nk)]
                       for isp in range(wav.nspin)]
        except (ValueError, IndexError, OSError) as error:
            raise SystemExit(f"ERROR: {wavecar_path}: unreadable WAVECAR "
                             f"({error}).") from None
        if wav.nspin == 1 and spin != "both":
            raise SystemExit(f"ERROR: --spin {spin}: the run is not spin-polarized "
                             "(ISPIN = 1).")
        structure, structure_path = _read_structure(directory, cell, wav)
        phonopy_cell = _phonopy_cell(structure)
        lattice = np.asarray(structure.lattice, float)
        positions = np.asarray(structure.positions, float)
        numbers = _species_numbers(structure.symbols)
        sg_symbol, sg_number = _space_group(phonopy_cell, symprec)
        with contextlib.redirect_stdout(io.StringIO()):
            special, rotations = special_kpoints_of_run(phonopy_cell, symprec)

        # band edges over the whole file (every k point, every spin channel);
        # the location is the first k point within _EDGE_TIE of the extremum,
        # so that a special point listed twice (a band path) is named by the
        # copy that is analysed
        eps_all = np.array([[h[2] for h in channel] for channel in headers])
        occ_all = np.array([[h[3] for h in channel] for channel in headers])
        occupied = occ_all > _OCCUPIED
        partial = bool(np.any((occ_all > _PARTIAL[0]) & (occ_all < _PARTIAL[1])))

        def extremum(mask, sign):
            if not np.any(mask):
                return None, None
            values = np.where(mask, sign * eps_all, -np.inf)
            best = float(np.max(values))
            near = mask & (values >= best - _EDGE_TIE)
            isp, ik = min((ik, isp) for isp, ik in zip(*np.where(np.any(near, axis=2))))[::-1]
            band = int(np.argmax(np.where(near[isp, ik], values[isp, ik], -np.inf)))
            return sign * best, (int(ik) + 1, band + 1, int(isp))

        vbm, vbm_at = extremum(occupied, 1.0)
        cbm, cbm_at = extremum(~occupied, -1.0)
        metallic = partial or (vbm is not None and cbm is not None and cbm < vbm)
        efermi, efermi_source = _read_efermi(directory, wav.efermi)
        if zero == "vbm" and vbm is None:
            raise SystemExit("ERROR: no occupied band in the WAVECAR: --zero vbm needs one "
                             "(use --zero efermi or raw).")
        zero_energy = {"vbm": vbm, "efermi": efermi, "raw": 0.0}[zero]

        targets, missing, selection = _select(klist, special, rotations, kpoints)
        channels = list(range(wav.nspin))
        if wav.nspin == 2 and spin != "both":
            channels = [0 if spin == "up" else 1]
        spin_names = ["up", "down"] if wav.nspin == 2 else [None]
        mulliken = None
        entries = []
        for name, index, is_special, copies in targets:
            for isp in channels:
                try:
                    gvecs, coeffs, eps, occ, kvec = wav.bands(index, isp)
                    group = little_group_data(lattice, positions, numbers, kvec,
                                              spinor=spinor, symprec=symprec)
                    levels = _regroup(
                        irreps_at_kpoint(lattice, positions, numbers, kvec, gvecs, coeffs,
                                         eps, spinor=spinor, degeneracy_tol=tolerance,
                                         symprec=symprec),
                        eps, tolerance, group)
                except ValueError as error:
                    raise SystemExit(f"ERROR: irreps at k#{index + 1} "
                                     f"{_fraction_vector(klist[index])}: {error}.") from None
                reversal = _time_reversal(group, lattice, positions, numbers, symprec)
                label = name or group.k_label or "k"
                gamma = np.allclose(group.kpoint, np.rint(group.kpoint), atol=1e-8)
                if gamma and not spinor and mulliken is None:
                    mulliken = _mulliken(phonopy_cell, symprec)
                rows = []
                for number, level in enumerate(levels, start=1):
                    bands = [b + 1 for b in level["bands"]]
                    names = [n for n, c in level["irreps"] for _ in range(c)]
                    energy_raw = float(level["energy"])
                    rows.append(BandLevel(
                        level=number, bands=bands, energy=energy_raw - zero_energy,
                        energy_raw=energy_raw, degeneracy=int(level["dim"]),
                        irrep=level["label"],
                        mulliken=(_level_mulliken(sorted(names), mulliken)
                                  if gamma and level["label"] and not spinor else None),
                        note=_note(level, len(eps), tolerance, reversal),
                        occupation=float(np.mean(occ[level["bands"]])),
                        irreps=[(n, int(c)) for n, c in level["irreps"]],
                        multiplicities=dict(level["multiplicities"]),
                        spread=float(level["spread"]), leakage=float(level["leakage"]),
                        lost=float(level["lost"]),
                        parts=[([b + 1 for b in part], part_label)
                               for part, part_label in level["parts"]]))
                entries.append(KPointIrreps(
                    label=label, kpoint=[float(x) for x in klist[index]], index=index + 1,
                    little_group=_little_group_symbol(group), k_label=group.k_label,
                    special=bool(is_special), copies=[c + 1 for c in copies],
                    spin=spin_names[isp] if wav.nspin == 2 else None, levels=rows))
        edges = [_edges(entry) for entry in entries]
        if window is not None:
            for entry in entries:
                entry.levels = [lv for lv in entry.levels
                                if window[0] <= lv.energy <= window[1]]
        result = BandIrreps(
            directory=directory, wavecar=wavecar_path, structure=os.path.abspath(structure_path),
            formula=_formula(structure.symbols), n_atoms=len(structure.symbols),
            nbands=wav.nb, nkpts=wav.nk, ispin=wav.nspin, spinor=bool(spinor),
            encut=float(wav.encut), precision=np.dtype(wav.cdtype).name,
            efermi=float(efermi), efermi_source=efermi_source,
            vbm=vbm, vbm_at=vbm_at,
            vbm_kpoint=None if vbm_at is None else [float(x) for x in klist[vbm_at[0] - 1]],
            vbm_label=_special_name(klist, vbm_at, special, rotations),
            cbm=cbm, cbm_at=cbm_at,
            cbm_kpoint=None if cbm_at is None else [float(x) for x in klist[cbm_at[0] - 1]],
            cbm_label=_special_name(klist, cbm_at, special, rotations),
            metallic=bool(metallic),
            zero=zero, zero_energy=float(zero_energy), space_group=sg_symbol,
            space_group_number=sg_number, symprec=float(symprec),
            degeneracy_tol=tolerance, window=window, selection=selection,
            kpoints=entries, missing=missing, edges=edges)
    return result


# ================================================================== report
def _where(result: BandIrreps, index: int, label: str, kpoint) -> str:
    """``k#206 R``; ``k#9 T`` when analysed; else ``k#9 [1/2, 1/2, 1/4]``."""
    if not label:
        label = next((entry.label for entry in result.kpoints if entry.index == index), "")
    return f"k#{index} {label}" if label else f"k#{index} {_fraction_vector(kpoint)}"


def _energy(value: float) -> float:
    """Rounded to the printed 4 decimals, without a negative zero."""
    return round(float(value), 4) + 0.0


def format_band_irreps(result: BandIrreps) -> list:
    """The terminal report of :func:`extract_band_irreps`, line by line.

    The ``* Title *`` blocks ``VASP run``, ``Space group``, ``Special k
    points in the WAVECAR`` (``k points in the WAVECAR`` with
    ``--all-kpoints``), one ``Band irreps at ...`` table per k point (and
    spin channel), and ``Band edges``.
    """
    out = []
    say = out.append
    zero_text = _ZERO_TEXT[result.zero]
    size = os.path.getsize(result.wavecar) if os.path.isfile(result.wavecar) else 0
    say("")
    say("* VASP run *")
    say(f"  directory    {result.directory}")
    say(f"  WAVECAR      {result.wavecar} ({size / 1e6:.1f} MB, {result.precision})")
    sites = (f"{result.n_atoms} sites, Va point charges included" if "(Va" in result.formula
             else f"{result.n_atoms} atoms")
    say(f"  structure    {result.structure} ({result.formula}, {sites}; "
        "the irrep labels refer to this cell)")
    say(f"  NBANDS {result.nbands}, NKPTS {result.nkpts}, ISPIN {result.ispin}, "
        f"spin-orbit {'yes (double-valued irreps)' if result.spinor else 'no'}, "
        f"ENCUT {result.encut:g} eV")
    say(f"  E_fermi      {result.efermi:.4f} eV ({result.efermi_source})")
    if result.vbm is not None:
        ik, band, isp = result.vbm_at
        where = f"{_where(result, ik, result.vbm_label, result.vbm_kpoint)}, band {band}"
        if result.ispin == 2:
            where += f", spin {('up', 'down')[isp]}"
        say(f"  VBM          {result.vbm:.4f} eV (highest occupied eigenvalue over all "
            f"{result.nkpts} k points: {where})")
    else:
        say("  VBM          none (no occupied band)")
    say(f"  energy zero  {result.zero}: the energies below are {zero_text}"
        + (f" ({result.zero_energy:.4f} eV subtracted)" if result.zero != "raw" else ""))
    if result.window is not None:
        say(f"  window       {result.window[0]:g} .. {result.window[1]:g} eV ({zero_text})")

    say("")
    say("* Space group *")
    say(f" {result.space_group} ({result.space_group_number})")

    say("")
    say("* k points in the WAVECAR *" if result.selection == "all"
        else "* Special k points in the WAVECAR *")
    seen = set()
    width = max([len(e.label) for e in result.kpoints] + [2])
    for entry in result.kpoints:
        if entry.index in seen:
            continue
        seen.add(entry.index)
        line = (f"  {entry.label:<{width}}  {_stored_vector(entry.kpoint)}  "
                f"k#{entry.index:<5d} {entry.little_group}")
        if entry.copies:
            line += "   also " + ", ".join(f"k#{c}" for c in entry.copies)
        say(line)
    if result.missing:
        say("  not in the file: " + ", ".join(
            f"{name} {_fraction_vector(k)}" for name, k in result.missing)
            + " (add them as zero-weight k points to label them)")

    for entry in result.kpoints:
        say("")
        say(f"* Band irreps at {entry.title} *")
        irrep_width = max([len(lv.label) for lv in entry.levels] + [5])
        band_width = max([len(_band_range(lv.bands)) for lv in entry.levels] + [5])
        energy_head = f"{zero_text} (eV)"
        say(f"level  {'bands':<{band_width}}  {energy_head:>14}  deg  "
            f"{'irrep':<{irrep_width}}  note")
        if not entry.levels:
            say("  (no level in the window)")
        for lv in entry.levels:
            say(f"{lv.level:5d}  {_band_range(lv.bands):<{band_width}}  {_energy(lv.energy):14.4f}  "
                f"{lv.degeneracy:3d}  {lv.label:<{irrep_width}}  {lv.note}".rstrip())

    say("")
    say("* Band edges *")
    rows = [("k point", "", "VBM (eV)", "irrep", "CBM (eV)", "irrep", "direct gap")]
    for edge in result.edges:
        rows.append((
            edge.label + (f" ({edge.spin})" if edge.spin else ""), f"k#{edge.index}",
            f"{_energy(edge.vbm.energy):.4f}" if edge.vbm is not None else "--",
            edge.vbm.label if edge.vbm is not None else "-",
            f"{_energy(edge.cbm.energy):.4f}" if edge.cbm is not None else "--",
            edge.cbm.label if edge.cbm is not None else "-",
            f"{_energy(edge.direct_gap):.4f}" if edge.direct_gap is not None else "--"))
    widths = [max(len(row[i]) for row in rows) for i in range(7)]
    for row in rows:
        say(f"{row[0]:<{widths[0]}}  {row[1]:<{widths[1]}}  {row[2]:>{widths[2]}}  "
            f"{row[3]:<{widths[3]}}  {row[4]:>{widths[4]}}  {row[5]:<{widths[5]}}  "
            f"{row[6]:>{widths[6]}}".rstrip())
    if result.metallic:
        say(f"  metallic: partially occupied bands (occupations between "
            f"{_PARTIAL[0]} and {_PARTIAL[1]}); no gap is quoted")
    elif result.gap is not None:
        vk, vb, vs = result.vbm_at
        ck, cb, cs = result.cbm_at
        spin_v = f", spin {('up', 'down')[vs]}" if result.ispin == 2 else ""
        spin_c = f", spin {('up', 'down')[cs]}" if result.ispin == 2 else ""
        kind = "direct" if vk == ck else "indirect"
        say(f"  gap {result.gap:.4f} eV ({kind}) over all {result.nkpts} k points of the "
            f"file: VBM at {_where(result, vk, result.vbm_label, result.vbm_kpoint)} (band "
            f"{vb}{spin_v}), CBM at {_where(result, ck, result.cbm_label, result.cbm_kpoint)} (band "
            f"{cb}{spin_c})")
    return out


# ==================================================================== YAML
def _level_dict(level: BandLevel) -> dict:
    return {
        "level": level.level,
        "bands": list(level.bands),
        "energy": _round(level.energy),
        "degeneracy": level.degeneracy,
        "irrep": level.irrep or None,
        "mulliken": level.mulliken,
        "occupation": _round(level.occupation, 4),
        "note": level.note,
    }


def _edge_dict(level: BandLevel | None) -> dict | None:
    if level is None:
        return None
    return {"level": level.level, "bands": list(level.bands),
            "energy": _round(level.energy), "irrep": level.irrep or None,
            "mulliken": level.mulliken}


def _result_dict(result: BandIrreps) -> dict:
    from . import __version__

    spin_names = ("up", "down")
    data = {
        "crystod_version": __version__,
        "mode": "extract-irrep",
        "run": {
            "directory": result.directory,
            "wavecar": result.wavecar,
            "structure": result.structure,
            "formula": result.formula,
            "nbands": result.nbands,
            "nkpts": result.nkpts,
            "ispin": result.ispin,
            "spin_orbit": result.spinor,
            "encut": _round(result.encut),
            "efermi": _round(result.efermi),
            "efermi_source": result.efermi_source,
            "vbm": _round(result.vbm),
            "cbm": _round(result.cbm),
        },
        "space_group": result.space_group,
        "space_group_number": result.space_group_number,
        "symprec": result.symprec,
        "energy_zero": result.zero,
        "zero_energy": _round(result.zero_energy),
        "degeneracy_tol": result.degeneracy_tol,
        "window": None if result.window is None else [_round(x) for x in result.window],
        "kpoint_selection": result.selection,
        "kpoints": [],
        "missing_special_points": [
            {"k_label": name, "k_position": [_round(x) for x in k]}
            for name, k in result.missing],
        "irreps": [],
        "band_edges": [],
        "gap": {
            "metallic": result.metallic,
            "gap": _round(result.gap),
            "vbm": None if result.vbm_at is None else {
                "energy": _round(result.vbm - result.zero_energy),
                "k_index": result.vbm_at[0],
                "k_position": [_round(x, 7) for x in result.vbm_kpoint],
                "k_label": result.vbm_label or None,
                "band": result.vbm_at[1],
                "spin": spin_names[result.vbm_at[2]] if result.ispin == 2 else None},
            "cbm": None if result.cbm_at is None else {
                "energy": _round(result.cbm - result.zero_energy),
                "k_index": result.cbm_at[0],
                "k_position": [_round(x, 7) for x in result.cbm_kpoint],
                "k_label": result.cbm_label or None,
                "band": result.cbm_at[1],
                "spin": spin_names[result.cbm_at[2]] if result.ispin == 2 else None},
        },
    }
    seen = set()
    for entry in result.kpoints:
        if entry.index in seen:
            continue
        seen.add(entry.index)
        data["kpoints"].append({
            "k_label": entry.label,
            "k_position": [_round(x, 7) for x in entry.kpoint],
            "k_index": entry.index,
            "little_group": entry.little_group,
            "special": entry.special,
            "copies": list(entry.copies),
        })
    for entry in result.kpoints:
        data["irreps"].append({
            "k_label": entry.label,
            "k_position": [_round(x, 7) for x in entry.kpoint],
            "k_index": entry.index,
            "spin": entry.spin,
            "levels": [_level_dict(level) for level in entry.levels],
        })
    for edge in result.edges:
        data["band_edges"].append({
            "k_label": edge.label,
            "k_index": edge.index,
            "spin": edge.spin,
            "vbm": _edge_dict(edge.vbm),
            "cbm": _edge_dict(edge.cbm),
            "direct_gap": _round(edge.direct_gap),
        })
    return data


def _yaml_scalar(value) -> str:
    """One YAML scalar that ``yaml.safe_load`` reads back as ``value``."""
    import json

    import yaml

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        text = repr(float(value))
        if "inf" in text or "nan" in text:
            return {"inf": ".inf", "-inf": "-.inf"}.get(text, ".nan")
        if "e" in text:
            # YAML 1.1 reads an exponent as a float only after a dot (1.0e-05)
            mantissa, exponent = text.split("e")
            if "." not in mantissa:
                mantissa += ".0"
            sign = "-" if exponent.startswith("-") else "+"
            text = f"{mantissa}e{sign}{exponent.lstrip('+-').zfill(2)}"
        return text
    text = str(value)
    try:
        if text and yaml.safe_load(f"key: {text}") == {"key": text}:
            return text
    except yaml.YAMLError:
        pass
    return json.dumps(text)


def _yaml_lines(data, indent: int = 0) -> list:
    """Block-style YAML of nested dicts and lists, scalar lists in flow style."""
    pad = " " * indent
    lines = []
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                if value:
                    lines.append(f"{pad}{key}:")
                    lines.extend(_yaml_lines(value, indent + 2))
                else:
                    lines.append(f"{pad}{key}: {{}}")
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                lines.append(f"{pad}{key}:")
                lines.extend(_yaml_lines(value, indent))
            elif isinstance(value, list):
                lines.append(f"{pad}{key}: [" + ", ".join(_yaml_scalar(v) for v in value)
                             + "]")
            else:
                lines.append(f"{pad}{key}: {_yaml_scalar(value)}")
    elif isinstance(data, list):
        for item in data:
            body = _yaml_lines(item, indent + 2)
            if body:
                body[0] = f"{pad}- " + body[0][indent + 2:]
            lines.extend(body)
    return lines


def write_band_irreps_yaml(result: BandIrreps, path: str) -> str:
    """Write the YAML file of ``-o FILE`` (the layout of :meth:`BandIrreps.to_dict`).

    The layout follows ``phonon_irreps.yaml``: run metadata first, then the
    analysed k points, then per k point (and spin channel) the list of levels
    with ``bands`` (1-based), ``energy`` (eV relative to the zero),
    ``degeneracy``, ``irrep`` (``null`` when unassigned), ``mulliken``
    (Gamma), ``occupation`` and ``note``, then the band edges.

    Returns:
        ``path``.
    """
    lines = ["# crystod --extract-irrep: irreducible representations of the bands "
             "of a VASP run (energies in eV)"]
    data = result.to_dict()
    blocks = [("crystod_version", "mode", "run", "space_group", "space_group_number",
               "symprec", "energy_zero", "zero_energy", "degeneracy_tol", "window",
               "kpoint_selection"), ("kpoints", "missing_special_points"), ("irreps",),
              ("band_edges", "gap")]
    for keys in blocks:
        lines.extend(_yaml_lines({key: data[key] for key in keys}))
        lines.append("")
    with open(path, "w") as handle:
        handle.write("\n".join(lines))
    return path


# ===================================================================== CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crystod --extract-irrep",
        description="Irreducible representations of the electronic bands of one finished "
                    "VASP run, from its WAVECAR: every special k point of the space group "
                    "in the file (or --kpoint / --all-kpoints), levels sorted by energy.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", default=".",
                        help="WAVECAR file or run directory (default: .)")
    parser.add_argument("-c", "--cell", "--poscar", dest="cell", default=None,
                        metavar="FILE",
                        help="structure file in place of the run's POSCAR (same cell)")
    parser.add_argument("--kpoint", nargs="+", default=None, metavar="K",
                        help="special-point labels (GM X M R ...) or fractional "
                             "coordinates (0 1/2 0)")
    parser.add_argument("--all-kpoints", action="store_true",
                        help="label every k point of the file")
    parser.add_argument("--zero", choices=ZERO_MODES, default="vbm",
                        help="energy zero (default vbm)")
    parser.add_argument("--window", type=float, nargs=2, default=None,
                        metavar=("EMIN", "EMAX"),
                        help="list only the levels in this range (eV, relative to the zero)")
    parser.add_argument("--spin", choices=SPIN_CHOICES, default="both",
                        help="spin channel of an ISPIN = 2 run (default both)")
    parser.add_argument("--degeneracy-tol", type=float, default=None, metavar="EV",
                        help=f"bands within EV of the first band of a level form one level "
                             f"(default {DEFAULT_DEGENERACY_TOL:g})")
    parser.add_argument("--tolerance", type=float, default=1e-5,
                        help="symmetry tolerance (default 1e-5)")
    parser.add_argument("-o", "--output", default=None, metavar="FILE",
                        help="write the results as YAML")
    return parser


def main(argv=None) -> int:
    """``crystod --extract-irrep`` (also ``python -m crystod.extract_irrep``).

    Args:
        argv: Argument list (default: ``sys.argv[1:]``).

    Returns:
        Exit status 0; every user error exits with one ``ERROR:`` line.
    """
    import sys

    from .cli.common import mark_negative_fractions

    argv = mark_negative_fractions(list(sys.argv[1:] if argv is None else argv))
    args = build_parser().parse_args(argv)
    if args.all_kpoints and args.kpoint:
        raise SystemExit("ERROR: --kpoint and --all-kpoints exclude each other.")
    result = extract_band_irreps(
        args.path, kpoints="all" if args.all_kpoints else args.kpoint, zero=args.zero,
        window=args.window, spin=args.spin, symprec=args.tolerance,
        degeneracy_tol=args.degeneracy_tol, cell=args.cell)
    for line in format_band_irreps(result):
        print(line)
    if args.output:
        existed = os.path.isfile(args.output)
        write_band_irreps_yaml(result, args.output)
        print("")
        print("* Output files *")
        print(f"  Band irreps {'OVERWRITTEN' if existed else 'written'}: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
