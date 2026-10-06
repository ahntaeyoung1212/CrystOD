"""Readers and writers for the VASP files the crystal-orbital engine needs.

``crystod --diagram --vasp`` builds its three diagram columns from ordinary
VASP output: the eigenvalues and occupations of ``EIGENVAL``/``PROCAR``, the
complex ``LORBIT = 12`` spherical-harmonic projections of ``PROCAR``, and a
handful of facts from ``OUTCAR`` (electron count, Fermi level, functional,
the PAW valence configuration and the CrystOD point-charge block).  The
sublattice runs name their switched-off atoms ``Va2-``/``Va2+``/``Va4+``
(classical point charges, see ``vasp.6.5.1.crystod``), which neither pymatgen
nor phonopy can parse, so the POSCAR reader here is deliberately
self-contained and depends on nothing but numpy.

How the numbers of a POSCAR become a cell is decided in ONE place,
:func:`poscar_geometry`, which follows VASP itself: the scale line (one
factor, a negative volume, or three factors) applies to the lattice vectors
and to Cartesian coordinates alike.  :func:`read_poscar` reads through it, and
so do the two bridges to the libraries the rest of CrystOD reads structures
with -- :func:`read_poscar_cell` (phonopy) and :func:`poscar_structure`
(pymatgen), which keep the library's own result wherever it is VASP's and
correct it where it is not.

Every reader accepts a gzipped file under the same name (``PROCAR.gz``), so an
archived run needs no unpacking.

Nothing in this module needs, reads or reproduces VASP source code; it only
parses the text files a finished run leaves behind.
"""

from __future__ import annotations

import gzip
import os
import re
from dataclasses import dataclass, field

import numpy as np

#: ``Va<charge><sign>`` (and the swapped ``Va<sign><charge>``) point-charge
#: species of the CrystOD-patched VASP.  A bare ``Va`` is NOT a point charge
#: (it would be ambiguous with a real element symbol), and the match is case
#: sensitive on ``Va``.
VA_SPECIES = re.compile(r"^Va(?:(\d+(?:\.\d+)?)([+-])|([+-])(\d+(?:\.\d+)?))$")

#: Real-orbital component order of a ``PROCAR`` lm block: plain ``m = -l..l``
#: for every shell (``py pz px`` for p, ``dxy dyz dz2 dxz dx2-y2`` for d).
VASP_M_ORDER = {l: list(range(-l, l + 1)) for l in range(4)}

#: The same order crystod uses everywhere else (``visualize_basis``'s
#: ``ORBITAL_COMPONENT_NAMES``): ``p_x p_y p_z`` for p, and the f order of
#: ``complex_to_real_transform_orbital``.
CRYSTOD_M_ORDER = {
    0: [0],
    1: [1, -1, 0],
    2: [-2, -1, 0, 1, 2],
    3: [3, -3, 2, -2, 1, -1, 0],
}

_NUMBER = re.compile(r"[-+]?\d*\.\d+(?:[eEdD][-+]?\d+)?|[-+]?\d+")


def resolve_vasp_file(path: str) -> str:
    """The file itself, or its ``.gz`` sibling, or ``""``.

    A band-structure ``PROCAR`` is the largest file of a VASP run by far, so
    runs are often archived gzipped.  Every reader of this module accepts
    either spelling; ``PROCAR`` wins when both exist.

    Args:
        path: Path of the plain (uncompressed) file.

    Returns:
        The path that exists, or an empty string when neither does.
    """
    if os.path.isfile(path):
        return path
    return path + ".gz" if os.path.isfile(path + ".gz") else ""


def open_vasp_file(path: str):
    """Open ``path`` (or its ``.gz`` sibling) as a text stream."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def _numbers(text: str) -> list[float]:
    """Every number in ``text``, robust against missing separators.

    VASP writes ``0.50000000-0.50000000`` when a coordinate is negative, so
    splitting on whitespace is not enough.
    """
    return [float(token.replace("D", "E").replace("d", "e"))
            for token in _NUMBER.findall(text)]


def parse_va_species(name: str) -> float | None:
    """Signed charge of a ``Va<q><sign>`` point-charge species.

    Args:
        name: A POSCAR species name such as ``"Va2-"``, ``"Va+4"`` or ``"O"``.

    Returns:
        The signed charge (``-2.0`` for ``Va2-``), or ``None`` when the name
        is an ordinary chemical species.
    """
    match = VA_SPECIES.match(name.strip())
    if not match:
        return None
    if match.group(1) is not None:
        magnitude, sign = match.group(1), match.group(2)
    else:
        sign, magnitude = match.group(3), match.group(4)
    return float(magnitude) * (1.0 if sign == "+" else -1.0)


def va_species_name(charge: float) -> str:
    """POSCAR species name of a point charge: ``+2 -> "Va2+"``."""
    return f"Va{abs(charge):g}{'+' if charge > 0 else '-'}"


@dataclass
class VaspStructure:
    """A POSCAR/CONTCAR, point-charge species included.

    Attributes:
        comment: The first line of the file.
        lattice: Lattice vectors as rows, in Angstrom (the scale is applied).
        species: Species names in POSCAR order, e.g. ``["Sr", "Ti", "Va2-"]``.
        counts: Number of atoms per species.
        symbols: One species name per atom.
        positions: Fractional coordinates, shape ``(n_atoms, 3)``.
        charges: Point-charge value per atom, ``None`` for a real atom.
        path: Where the file was read from.
    """

    comment: str
    lattice: np.ndarray
    species: list[str]
    counts: list[int]
    symbols: list[str]
    positions: np.ndarray
    charges: list[float | None]
    path: str = ""

    @property
    def is_point_charge(self) -> list[bool]:
        return [charge is not None for charge in self.charges]

    @property
    def real_indices(self) -> list[int]:
        return [index for index, charge in enumerate(self.charges) if charge is None]

    @property
    def real_elements(self) -> list[str]:
        """The non-Va species names, in POSCAR order, without repetitions."""
        seen: list[str] = []
        for name, charge in zip(self.species, self._species_charges()):
            if charge is None and name not in seen:
                seen.append(name)
        return seen

    def _species_charges(self) -> list[float | None]:
        return [parse_va_species(name) for name in self.species]


@dataclass
class PoscarGeometry:
    """The cell of a POSCAR, resolved as VASP itself resolves it.

    Attributes:
        lattice: Lattice vectors as rows, in Angstrom (the scale is applied).
        positions: Fractional coordinates, shape ``(n_atoms, 3)``.
        counts: Number of atoms per species.
        cartesian: Whether the file lists Cartesian coordinates.
        species_line: Index of the species-name line; ``None`` for the VASP 4
            layout, which has none.
    """

    lattice: np.ndarray
    positions: np.ndarray
    counts: list[int]
    cartesian: bool
    species_line: int | None

    def agrees_with(self, lattice, positions, tolerance: float = 1e-10) -> bool:
        """Whether the cell another parser returned is this one, up to rounding."""
        lattice = np.asarray(lattice, dtype=float)
        positions = np.asarray(positions, dtype=float)
        return bool(
            lattice.shape == self.lattice.shape
            and positions.shape == self.positions.shape
            and np.allclose(lattice, self.lattice, rtol=tolerance, atol=tolerance)
            and np.allclose(positions, self.positions, rtol=0.0, atol=tolerance))


def poscar_geometry(lines: list[str], path: str = "the POSCAR") -> PoscarGeometry:
    """Lattice and fractional coordinates of a POSCAR, as VASP reads them.

    This is the ONE rule every POSCAR reader of CrystOD follows --
    :func:`read_poscar` takes its numbers from here, and what phonopy and
    pymatgen return is checked against it (:func:`read_poscar_cell`,
    :func:`poscar_structure`) -- so that a file means the same crystal in
    every command: the crystal VASP computes.  VASP's reading, as the POSCAR
    page of the VASP wiki specifies it:

    * line 2 holds ONE factor ``s``, or THREE factors ``sx sy sz``, one per
      Cartesian axis.  A negative ``s`` is the cell VOLUME: the factor is then
      ``(|s| / |det A|) ** (1/3)``, ``A`` being the lattice as written;
    * the factor multiplies the lattice vectors AND Cartesian coordinates
      (their ``x`` components by ``s * sx``, and so on).  Direct coordinates
      are fractions of the lattice vectors and are left alone;
    * the coordinates are Cartesian when the first letter of the mode line,
      after an optional ``Selective dynamics`` line, is one of ``C c K k``.

    The VASP 5 layout (species names on line 6) and the VASP 4 layout (atom
    counts on line 6) are both accepted.

    Args:
        lines: The lines of the file.
        path: The file's name, for the messages.

    Returns:
        The :class:`PoscarGeometry`.

    Raises:
        ValueError: The lines are not a POSCAR; the message says what is
            wrong.
    """
    if len(lines) < 8:
        raise ValueError(f"{path} is too short to be a POSCAR")
    try:
        given = []
        for token in lines[1].split():
            try:
                given.append(float(token))
            except ValueError:
                break
        rows = np.array([[float(x) for x in lines[index].split()[:3]]
                         for index in (2, 3, 4)], dtype=float)
        if rows.shape != (3, 3):
            raise ValueError("a lattice vector has fewer than three components")
        if len(given) == 1:
            scale = given[0]
            if scale < 0:
                if np.linalg.det(rows) == 0:
                    raise ValueError("the lattice vectors span no volume")
                scale = (abs(scale) / abs(np.linalg.det(rows))) ** (1.0 / 3.0)
            factors = np.array([scale, scale, scale], dtype=float)
        elif len(given) == 3:
            factors = np.array(given, dtype=float)
        else:
            raise ValueError("line 2 must hold one scale factor, or three")
        if not np.all(np.isfinite(factors)) or np.any(factors == 0.0):
            raise ValueError("a scale factor is zero or not a number")
    except (IndexError, ValueError) as error:
        raise ValueError(f"cannot read the cell of {path} ({error})") from None
    # one factor per Cartesian component, i.e. per COLUMN of the row vectors
    lattice = rows * factors

    def integers(text):
        try:
            return [int(token) for token in text.split()]
        except ValueError:
            return None

    def skip_blank(index):
        # an empty line before the mode line is passed over
        while index < len(lines) and not lines[index].strip():
            index += 1
        return index

    species_line = None
    index = 5
    counts = integers(lines[index])
    if counts is None:
        species_line = index
        index += 1
        counts = integers(lines[index])
    if not counts or min(counts) < 0 or sum(counts) == 0:
        raise ValueError(f"cannot read the atom counts of {path}")
    index = skip_blank(index + 1)
    if index < len(lines) and lines[index].strip()[:1] in ("s", "S"):
        index = skip_blank(index + 1)
    cartesian = index < len(lines) and lines[index].strip()[:1] in ("c", "C", "k", "K")
    index += 1
    total = sum(counts)
    if len(lines) < index + total:
        raise ValueError(f"{path} has fewer coordinate lines than atoms")
    try:
        numbers = np.array([[float(x) for x in lines[index + row].split()[:3]]
                            for row in range(total)], dtype=float)
        if numbers.shape != (total, 3):
            raise ValueError("a coordinate line has fewer than three numbers")
        positions = (numbers * factors) @ np.linalg.inv(lattice) if cartesian else numbers
    except (ValueError, np.linalg.LinAlgError) as error:
        raise ValueError(f"cannot read the coordinates of {path} ({error})") from None
    return PoscarGeometry(lattice=lattice, positions=positions, counts=counts,
                          cartesian=cartesian, species_line=species_line)


def _plain_poscar_text(lines: list[str], geometry: PoscarGeometry) -> str:
    """The POSCAR again, in the one spelling every parser reads as VASP does.

    Scale ``1.0`` and direct coordinates.  The numbers are written with
    ``repr``, so parsing the text gives the arrays of :func:`poscar_geometry`
    back bit for bit; the comment, species and counts lines, from which a
    parser takes the chemical symbols, are the file's own.
    """
    def line(row):
        return "  " + "  ".join(repr(float(value)) for value in row)

    header_end = 6 if geometry.species_line is None else 7
    out = [lines[0], "1.0"]
    out.extend(line(row) for row in geometry.lattice)
    out.extend(lines[5:header_end])
    out.append("Direct")
    out.extend(line(row) for row in geometry.positions)
    return "\n".join(out) + "\n"


def read_poscar(path: str) -> VaspStructure:
    """Read a POSCAR/CONTCAR that may carry ``Va`` point-charge species.

    The lattice and the coordinates are those of :func:`poscar_geometry`,
    VASP's own reading of the scale line and of Cartesian coordinates.

    Args:
        path: Path of the file.

    Returns:
        The :class:`VaspStructure`.

    Raises:
        SystemExit: The file is missing or malformed.
    """
    resolved = resolve_vasp_file(path)
    if not resolved:
        raise SystemExit(f"ERROR: no POSCAR at {path}.")
    with open_vasp_file(resolved) as handle:
        lines = handle.read().splitlines()
    if len(lines) < 8:
        raise SystemExit(f"ERROR: {path} is too short to be a POSCAR.")
    try:
        geometry = poscar_geometry(lines, path)
    except ValueError as error:
        raise SystemExit(f"ERROR: {error}.") from None
    species = lines[5].split()
    if not species or species[0][0].isdigit():
        raise SystemExit(
            f"ERROR: {path} has no species-name line (VASP 4 format); "
            "the --vasp engine needs the VASP 5 POSCAR with species names.")
    try:
        counts = [int(value) for value in lines[6].split()]
    except ValueError:
        raise SystemExit(f"ERROR: cannot read the atom counts of {path}.") from None
    if len(counts) != len(species):
        raise SystemExit(
            f"ERROR: {path} lists {len(species)} species but {len(counts)} counts.")
    symbols: list[str] = []
    charges: list[float | None] = []
    for name, count in zip(species, counts):
        charge = parse_va_species(name)
        symbols.extend([name] * count)
        charges.extend([charge] * count)
    return VaspStructure(comment=lines[0].rstrip(), lattice=geometry.lattice,
                         species=species, counts=counts, symbols=symbols,
                         positions=geometry.positions, charges=charges, path=path)


def read_poscar_cell(path: str):
    """The crystal of a POSCAR as ``PhonopyAtoms``, read as VASP reads it.

    phonopy parses the file as it always did, so the chemical symbols and the
    atom order are its own.  Its cell is kept wherever it agrees with
    :func:`poscar_geometry` -- every POSCAR in direct coordinates with a
    positive scale factor, for which the object returned IS phonopy's -- and
    is VASP's reading otherwise: phonopy 4.3.0 does not apply the scale
    factor to Cartesian coordinates, takes a negative one (the cell volume)
    for a factor, and cannot read three of them.

    Args:
        path: Path of the POSCAR.

    Returns:
        The ``PhonopyAtoms``.

    Raises:
        FileNotFoundError: There is no such file (worded as ``phonopy.load``
            words a missing unit cell).
    """
    from phonopy.interface.vasp import read_vasp, read_vasp_from_strings

    if not os.path.isfile(path):
        raise FileNotFoundError("'%s' could not be found." % path)
    try:
        with open(path) as handle:
            lines = handle.read().splitlines()
        geometry = poscar_geometry(lines, path)
    except (OSError, ValueError):
        # nothing the rule can read: phonopy's own answer, or its own error
        return read_vasp(path)
    try:
        cell = read_vasp(path)
    except Exception:  # noqa: BLE001 - e.g. float("2.0 0.5 1.25") for three factors
        cell = None
    if cell is not None and geometry.agrees_with(cell.cell, cell.scaled_positions):
        return cell
    return read_vasp_from_strings(_plain_poscar_text(lines, geometry))


def poscar_structure(structure, path: str):
    """A pymatgen ``Structure`` parsed from a POSCAR, in VASP's reading of it.

    pymatgen applies the scale factor to Cartesian coordinates, but as the
    number written on line 2: with a negative one (the cell volume) every atom
    is misplaced (pymatgen-core 2026.5.18).  Wherever the parsed cell differs
    from :func:`poscar_geometry` it is replaced; direct coordinates never
    differ.

    Args:
        structure: What pymatgen parsed from ``path``.
        path: The file.  One that is not a POSCAR (a CIF, ...) leaves the
            structure as it is.

    Returns:
        ``structure`` itself, or a ``Structure`` with the same species and
        site properties on VASP's lattice and coordinates.
    """
    try:
        with open_vasp_file(path) as handle:
            geometry = poscar_geometry(handle.read().splitlines(), path)
    except (OSError, EOFError, ValueError):
        return structure
    if (len(structure) != len(geometry.positions)
            or geometry.agrees_with(structure.lattice.matrix, structure.frac_coords)):
        return structure
    from pymatgen.core import Lattice, Structure

    return Structure(Lattice(geometry.lattice), [site.species for site in structure],
                     geometry.positions, site_properties=structure.site_properties)


def format_poscar(comment: str, lattice, species: list[str], counts: list[int],
                  positions) -> str:
    """POSCAR text (VASP 5, direct coordinates) for the given cell."""
    def line(row):
        return "   " + "   ".join(f"{value:.14f}" for value in row)

    out = [comment, "1.0"]
    out.extend(line(row) for row in np.asarray(lattice, dtype=float))
    out.append("   " + "   ".join(species))
    out.append(" ".join(str(int(value)) for value in counts))
    out.append("Direct")
    out.extend(line(row) for row in np.asarray(positions, dtype=float))
    return "\n".join(out) + "\n"


@dataclass
class ProcarData:
    """The contents of one ``PROCAR`` written with ``LORBIT = 12``.

    Attributes:
        kpoints: Fractional k points in the reciprocal basis of the run's
            own cell, shape ``(n_k, 3)``.
        weights: Their k-point weights (zero for band-structure points).
        energies: Eigenvalues in eV, shape ``(n_k, n_bands)``.
        occupations: Occupations, same shape.
        phases: Complex projections ``<Y_lm at ion | psi_nk>``, as a mapping
            ``{k index: array of shape (n_bands, n_ions, n_lm)}`` holding
            ONLY the requested k points.  A band-structure PROCAR carries
            hundreds to thousands of k points, so a dense
            ``(n_k, n_bands, n_ions, n_lm)`` array would cost gigabytes for
            the handful of special points the diagram actually reads.
        lm: The lm component names of the file's header row.
        n_ions: Number of ions.
        n_bands: Number of bands.
        n_kpoints: Number of k points.
        loaded: Indices of the k points whose projections were parsed (the
            keys of :attr:`phases`).
    """

    kpoints: np.ndarray
    weights: np.ndarray
    energies: np.ndarray
    occupations: np.ndarray
    phases: dict
    lm: list[str]
    n_ions: int
    n_bands: int
    n_kpoints: int
    loaded: set = field(default_factory=set)
    path: str = ""


def _matches_any(kpoint, wanted, tolerance) -> bool:
    difference = wanted - np.asarray(kpoint, dtype=float)
    return bool(np.any(np.all(np.abs(difference - np.rint(difference)) < tolerance, axis=1)))


def read_procar(path: str, kpoints=None, tolerance: float = 1e-5) -> ProcarData:
    """Read a ``PROCAR``; keep the projections of the requested k points.

    The eigenvalues, occupations, k points and weights are read for every k
    point (they are cheap); the complex projection tables, which dominate the
    file size, are stored only for the k points listed in ``kpoints`` (modulo
    a reciprocal-lattice vector).

    Args:
        path: Path of the ``PROCAR``.
        kpoints: Fractional k points to keep the projections of, or ``None``
            for none of them.
        tolerance: Matching tolerance of a k point.

    Returns:
        The :class:`ProcarData`.

    Raises:
        SystemExit: The file is missing, is not a ``LORBIT = 12`` PROCAR, or
            cannot be parsed.
    """
    resolved = resolve_vasp_file(path)
    if not resolved:
        raise SystemExit(f"ERROR: no PROCAR at {path}.")
    with open_vasp_file(resolved) as handle:
        lines = handle.readlines()
    if len(lines) < 2 or "phase" not in lines[0]:
        raise SystemExit(
            f"ERROR: {path} carries no phases; the --vasp engine needs "
            "LORBIT = 12 (PROCAR lm decomposed + phase).")
    header = re.search(
        r"# of k-points:\s*(\d+)\s*# of bands:\s*(\d+)\s*# of ions:\s*(\d+)", lines[1])
    if not header:
        raise SystemExit(f"ERROR: cannot read the header of {path}.")
    n_k, n_bands, n_ions = (int(header.group(index)) for index in (1, 2, 3))
    kpoint_array = np.zeros((n_k, 3))
    weights = np.zeros(n_k)
    energies = np.zeros((n_k, n_bands))
    occupations = np.zeros((n_k, n_bands))
    # {k index: (n_bands, n_ions, n_lm)} -- only the requested k points, so a
    # 1000-k-point band run costs the few megabytes of the special points
    # instead of the gigabyte a dense array over every k point would take
    phases: dict = {}
    lm_names: list[str] = []
    wanted = None if kpoints is None else np.asarray(kpoints, dtype=float)
    loaded: set = set()
    index, total, k_index, keep = 2, len(lines), -1, False
    while index < total:
        line = lines[index]
        stripped = line.lstrip()
        if stripped.startswith("k-point"):
            head, _, body = line.partition(":")
            k_index = int(_numbers(head)[0]) - 1
            values = _numbers(body)
            kpoint_array[k_index] = values[:3]
            weights[k_index] = values[3] if len(values) > 3 else 0.0
            keep = wanted is not None and _matches_any(
                kpoint_array[k_index], wanted, tolerance)
            index += 1
            continue
        if stripped.startswith("band "):
            values = _numbers(line)
            band = int(values[0]) - 1
            energies[k_index, band] = values[1]
            occupations[k_index, band] = values[2]
            if not keep:
                index += 1
                continue
            cursor = index + 1
            while cursor < total and not lines[cursor].lstrip().startswith("ion"):
                cursor += 1
            if cursor >= total:
                break
            if not lm_names:
                lm_names = lines[cursor].split()[1:-1]
            n_lm = len(lines[cursor].split()) - 2
            cursor += 1 + n_ions
            if cursor < total and lines[cursor].lstrip().startswith("tot"):
                cursor += 1
            while cursor < total and not lines[cursor].lstrip().startswith("ion"):
                cursor += 1
            if cursor >= total:
                break
            table = phases.get(k_index)
            if table is None:
                table = np.zeros((n_bands, n_ions, n_lm), dtype=complex)
                phases[k_index] = table
            for ion in range(n_ions):
                values = _numbers(lines[cursor + 1 + ion])[1:]
                if len(values) < 2 * n_lm:
                    raise SystemExit(
                        f"ERROR: truncated phase block in {path} "
                        f"(k point {k_index + 1}, band {band + 1}).")
                block = np.asarray(values[:2 * n_lm]).reshape(n_lm, 2)
                table[band, ion] = block[:, 0] + 1j * block[:, 1]
            loaded.add(k_index)
            index = cursor + 1 + n_ions
            continue
        index += 1
    return ProcarData(kpoints=kpoint_array, weights=weights, energies=energies,
                      occupations=occupations, phases=phases, lm=lm_names,
                      n_ions=n_ions, n_bands=n_bands, n_kpoints=n_k,
                      loaded=loaded, path=path)


def read_eigenval(path: str):
    """Read ``EIGENVAL``: ``(kpoints, weights, energies, occupations)``."""
    resolved = resolve_vasp_file(path)
    if not resolved:
        raise SystemExit(f"ERROR: no EIGENVAL at {path}.")
    with open_vasp_file(resolved) as handle:
        lines = handle.read().splitlines()
    n_spin = int(lines[0].split()[3])
    if n_spin != 1:
        raise SystemExit(
            f"ERROR: {path} is spin polarized (ISPIN = 2); the --vasp "
            "crystal-orbital engine handles non-spin-polarized runs only.")
    _, n_k, n_bands = (int(value) for value in lines[5].split()[:3])
    kpoints = np.zeros((n_k, 3))
    weights = np.zeros(n_k)
    energies = np.zeros((n_k, n_bands))
    occupations = np.zeros((n_k, n_bands))
    cursor = 6
    for index in range(n_k):
        while not lines[cursor].strip():
            cursor += 1
        values = _numbers(lines[cursor])
        kpoints[index] = values[:3]
        weights[index] = values[3]
        cursor += 1
        for band in range(n_bands):
            row = _numbers(lines[cursor])
            energies[index, band] = row[1]
            occupations[index, band] = row[2]
            cursor += 1
    return kpoints, weights, energies, occupations


@dataclass
class PotcarSpecies:
    """What ``OUTCAR`` reports about one PAW dataset.

    Attributes:
        title: The ``TITEL`` line, e.g. ``"PAW_PBE Sr_sv 07Sep2000"``.
        element: The chemical symbol taken from ``VRHFIN``.
        valence_text: The ``VRHFIN`` configuration string, e.g. ``"4s4p5s"``.
        zval: Number of valence electrons.
        lexch: The exchange-correlation tag of the dataset (``"PE"``, ...).
        configuration: ``[(n, l, energy_eV, occupation), ...]`` of the
            ``Atomic configuration`` table.
    """

    title: str = ""
    element: str = ""
    valence_text: str = ""
    zval: float = 0.0
    lexch: str = ""
    configuration: list = field(default_factory=list)

    def valence_shells(self) -> dict:
        """``{l: [n, ...]}`` of the shells the dataset treats as valence.

        The ``Atomic configuration`` entries are sorted by energy and taken
        from the top until their occupations account for ``ZVAL``.  When that
        table is absent -- an OUTCAR excerpt need not carry it -- the
        ``VRHFIN`` configuration string is used instead, but only when it
        names the shells with their principal quantum number
        (``Sr: 4s4p5s``).  Many datasets write occupations there instead
        (``O: s2p4``, ``Ti: d3 s1``), which says nothing about ``n``; those
        return nothing and leave the caller on its own valence table.
        """
        if self.configuration and self.zval > 0:
            entries = sorted(self.configuration, key=lambda row: -row[2])
            total = 0.0
            shells: dict[int, set] = {}
            for n, l, _energy, occupation in entries:
                if total >= self.zval - 1e-6:
                    break
                shells.setdefault(l, set()).add(n)
                total += occupation
            if abs(total - self.zval) <= 0.5:
                return {l: sorted(values) for l, values in shells.items()}
        return self._vrhfin_shells()

    def _vrhfin_shells(self) -> dict:
        """``{l: [n, ...]}`` from a ``VRHFIN`` string such as ``4s4p5s``."""
        text = self.valence_text.strip()
        if not text:
            return {}
        pairs = re.findall(r"(\d)\s*([spdf])", text)
        if not pairs or len(pairs) != len(re.findall(r"[spdf]", text)):
            return {}
        shells: dict[int, set] = {}
        for n, letter in pairs:
            shells.setdefault("spdf".index(letter), set()).add(int(n))
        return {l: sorted(values) for l, values in shells.items()}


def read_outcar(path: str) -> dict:
    """The handful of ``OUTCAR`` facts the crystal-orbital engine uses.

    Args:
        path: Path of the ``OUTCAR``.

    Returns:
        ``{"nelect", "efermi", "ispin", "nbands", "encut", "functional",
        "gga", "metagga", "species" (list of :class:`PotcarSpecies`), "va"
        (list of point-charge dicts), "lorbit", "noncollinear"}``; missing
        entries are ``None`` or empty (``noncollinear`` is ``True`` for a
        spinor run, ``LNONCOLLINEAR``/``LSORBIT``, else ``False``).
        ``"functional"`` is the exchange-correlation tag
        the eigenvalues belong to: the ``METAGGA`` tag of a meta-GGA run
        (``"LAK"``, ``"SCAN"``, ``"R2SCAN"``, upper case), otherwise the
        ``GGA`` tag (``"PE"``; see :data:`GGA_NAMES`).

    Raises:
        SystemExit: The file is missing.
    """
    resolved = resolve_vasp_file(path)
    if not resolved:
        raise SystemExit(f"ERROR: no OUTCAR at {path}.")
    facts: dict = {"nelect": None, "efermi": None, "ispin": None, "nbands": None,
                   "encut": None, "functional": None, "gga": None,
                   "metagga": None, "species": [], "va": [], "lorbit": None,
                   "noncollinear": False}
    species: list[PotcarSpecies] = []
    current: PotcarSpecies | None = None
    in_configuration = False
    in_va_table = False
    seen_titles: set = set()

    def flush() -> None:
        """Register the dataset just read, unless an equal one is known.

        A dataset is closed by the next ``VRHFIN`` line or by the end of the
        file, not by the end of its ``Atomic configuration`` table: an OUTCAR
        excerpt may carry the ``VRHFIN``/``TITEL`` lines without the table.
        VASP prints the block once per species and repeats it in the run
        summary, hence the de-duplication.
        """
        if current is None:
            return
        key = current.title or current.element
        if key and key not in seen_titles:
            seen_titles.add(key)
            species.append(current)

    with open_vasp_file(resolved) as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("VRHFIN") and "=" in stripped:
                flush()
                text = stripped.split("=", 1)[1]
                element, _, valence = text.partition(":")
                current = PotcarSpecies(element=element.strip(),
                                        valence_text=valence.strip())
                in_configuration = False
                continue
            if current is not None and stripped.startswith("TITEL"):
                current.title = stripped.split("=", 1)[1].strip()
                continue
            if current is not None and "ZVAL" in stripped and "mass and valenz" in stripped:
                match = re.search(r"ZVAL\s*=\s*([-\d.]+)", stripped)
                if match:
                    current.zval = float(match.group(1))
                continue
            if current is not None and stripped.startswith("LEXCH"):
                current.lexch = stripped.split("=", 1)[1].strip()
                continue
            if current is not None and stripped.startswith("Atomic configuration"):
                in_configuration = True
                continue
            if in_configuration:
                values = _numbers(stripped)
                if len(values) == 5 and stripped.split()[0].isdigit():
                    current.configuration.append(
                        (int(values[0]), int(values[1]), values[3], values[4]))
                    continue
                if stripped.startswith("Description") or stripped.startswith("local"):
                    in_configuration = False
                    continue
                continue
            if stripped.startswith("V_loc(r)") or stripped.startswith("typ  name"):
                in_va_table = stripped.startswith("typ  name")
                continue
            if in_va_table:
                # columns: typ name ions q sigma A b PSCORE V(0); the species
                # NAME carries digits ("Va2-"), so the row is split on
                # whitespace rather than scanned for numbers
                parts = stripped.split()
                if len(parts) >= 9 and parts[0].isdigit():
                    facts["va"].append({
                        "name": parts[1], "ions": int(parts[2]),
                        "charge": float(parts[3]), "sigma": float(parts[4]),
                        "wall": float(parts[5]), "rwall": float(parts[6])})
                else:
                    in_va_table = False
                continue
            if facts["nelect"] is None and "NELECT" in stripped:
                match = re.search(r"NELECT\s*=\s*([-\d.]+)", stripped)
                if match:
                    facts["nelect"] = float(match.group(1))
                continue
            if "E-fermi" in stripped:
                match = re.search(r"E-fermi\s*:\s*([-\d.]+)", stripped)
                if match:
                    facts["efermi"] = float(match.group(1))
                continue
            if facts["ispin"] is None and "ISPIN" in stripped:
                match = re.search(r"ISPIN\s*=\s*(\d+)", stripped)
                if match:
                    facts["ispin"] = int(match.group(1))
                continue
            # a spinor (vasp_ncl) run: "LNONCOLLINEAR = T" in the parameter
            # list, or the "LSORBIT = .TRUE." of the INCAR echo
            if (stripped.startswith(("LNONCOLLINEAR", "LSORBIT"))
                    and "=" in stripped):
                value = stripped.split("=", 1)[1].split()
                if value and value[0].strip(".").upper().startswith("T"):
                    facts["noncollinear"] = True
                continue
            if facts["nbands"] is None and "NBANDS=" in stripped:
                match = re.search(r"NBANDS=\s*(\d+)", stripped)
                if match:
                    facts["nbands"] = int(match.group(1))
                continue
            if facts["encut"] is None and "ENCUT" in stripped and "=" in stripped:
                match = re.search(r"ENCUT\s*=\s*([\d.]+)", stripped)
                if match:
                    facts["encut"] = float(match.group(1))
                continue
            if facts["lorbit"] is None and "LORBIT" in stripped:
                match = re.search(r"LORBIT\s*=\s*(\d+)", stripped)
                if match:
                    facts["lorbit"] = int(match.group(1))
                continue
            if facts["gga"] is None and "GGA     =" in line:
                facts["gga"] = stripped.split("=", 1)[1].split()[0]
                continue
            # VASP 6 writes "METAGGA = LAK    functional components" (and
            # echoes the INCAR line) and then no GGA line at all; VASP 5
            # writes "METAGGA=      F    non-selfconsistent MetaGGA calc."
            # into EVERY run, a flag rather than a functional
            if (facts["metagga"] is None and stripped.startswith("METAGGA")
                    and "=" in stripped):
                tag = re.split(r"[\s;!#]+",
                               stripped.split("=", 1)[1].strip())[0].upper()
                if tag not in _METAGGA_UNSET:
                    facts["metagga"] = tag
                continue
    flush()
    facts["species"] = species
    # the eigenvalues of a meta-GGA run are meta-GGA eigenvalues whatever
    # the GGA line says, so METAGGA wins and GGA is the fallback
    facts["functional"] = facts["metagga"] or facts["gga"]
    return facts


#: ``METAGGA`` values that do not name a functional: VASP 5's flag of a
#: non-self-consistent meta-GGA energy evaluated on GGA eigenvalues, and the
#: spellings of "not set".
_METAGGA_UNSET = {"", "F", "T", "FALSE", "TRUE", ".FALSE.", ".TRUE.", "--",
                  "NONE"}


#: What the ``GGA`` tag of ``OUTCAR`` means, for the method chip.
GGA_NAMES = {"PE": "PBE", "PS": "PBEsol", "RP": "revPBE", "91": "PW91",
             "AM": "AM05", "B3": "B3LYP", "--": "LDA", "CA": "LDA"}


def split_potcar(text: str) -> list:
    """Split a concatenated ``POTCAR`` into its datasets.

    Args:
        text: The whole file.

    Returns:
        ``[(element, block_text), ...]`` in file order; ``element`` is taken
        from the block's ``VRHFIN`` line (``"Sr_sv"`` -> ``"Sr"``).
    """
    blocks = []
    current: list[str] = []
    for line in text.splitlines(keepends=True):
        current.append(line)
        if "End of Dataset" in line:
            blocks.append("".join(current))
            current = []
    if current and "".join(current).strip():
        blocks.append("".join(current))
    out = []
    for block in blocks:
        match = re.search(r"VRHFIN\s*=\s*([A-Za-z]+)\s*:", block)
        element = match.group(1) if match else ""
        out.append((element, block))
    return out


def potcar_block_facts(block: str) -> tuple:
    """``(titel, zval)`` of one PAW dataset, from its ``POTCAR`` text.

    ``--vasp-setup`` prints them so that a user sees WHICH dataset was picked
    (``Sc`` and ``Sc_sv`` differ by eight valence electrons) and so that a
    sublattice whose formal charge would strip every valence electron can be
    refused instead of silently running with ``NELECT = 0``.

    Args:
        block: One dataset of a ``POTCAR``.

    Returns:
        ``(TITEL string, ZVAL)``; empty string and ``0.0`` when absent.
    """
    title = re.search(r"TITEL\s*=\s*(.+)", block)
    zval = re.search(r"ZVAL\s*=\s*([-+0-9.eEdD]+)", block)
    return (title.group(1).strip() if title else "",
            float(zval.group(1).replace("D", "E")) if zval else 0.0)


def reorder_to_crystod(l: int) -> np.ndarray:
    """``Q`` with ``Q[crystod_row, vasp_row] = 1`` for the real shell ``l``."""
    size = 2 * l + 1
    matrix = np.zeros((size, size))
    vasp = VASP_M_ORDER[l]
    for row, m in enumerate(CRYSTOD_M_ORDER[l]):
        matrix[row, vasp.index(m)] = 1.0
    return matrix


def lm_block_layout(lm_names: list[str]) -> list:
    """``[(l, first_column), ...]`` of a PROCAR lm header row.

    Args:
        lm_names: The header row between ``ion`` and ``tot``, e.g.
            ``["s", "py", "pz", "px", "dxy", ...]``.

    Returns:
        One entry per shell present, in file order.

    Raises:
        SystemExit: The header is not a whole number of complete shells.
    """
    layout = []
    column = 0
    for l in range(4):
        size = 2 * l + 1
        if column + size > len(lm_names):
            break
        layout.append((l, column))
        column += size
        if column == len(lm_names):
            return layout
    raise SystemExit(
        f"ERROR: cannot interpret the PROCAR lm header {' '.join(lm_names)} "
        "as complete s/p/d/f shells.")


# --------------------------------------------------------------- run settings

#: Relative tolerance on the Cartesian lattice matrices of a run's POSCAR and
#: of the analysis cell.  Two POSCARs of the same relaxed structure written by
#: different tools differ in the last printed digits (the SrTiO3 example:
#: 8e-6), while a genuinely different cell differs far more.
LATTICE_MATCH_RTOL = 1e-3

#: Tolerance on a fractional coordinate when a run's atom is matched with an
#: atom of the analysis cell.
POSITION_MATCH_TOL = 1e-3


class StructureMappingError(Exception):
    """A run's POSCAR is not the analysis cell in another setting.

    Attributes:
        reason: One line naming WHAT differs, ready to be quoted in the
            engine's error message.
        kind: ``"count"``, ``"lattice"``, ``"basis"``, ``"composition"`` or
            ``"positions"`` -- which check failed.
    """

    def __init__(self, reason: str, kind: str = "positions"):
        super().__init__(reason)
        self.reason = reason
        self.kind = kind


@dataclass
class RunMapping:
    """How one VASP run's POSCAR sits on the analysis cell of ``-c``.

    The run is the analysis cell with its origin moved: atom ``a`` of the
    analysis cell is ion ``ions[a]`` of the run, and

    ``y[ions[a]] = x[a] + shift + wraps[a]``

    holds exactly in fractional coordinates (``x`` the analysis cell, ``y``
    the run's POSCAR, ``wraps`` integral).

    Attributes:
        ions: ``ions[a]`` -- the run's 0-based ion index of analysis atom a.
        shift: The one global origin shift, fractional, in ``[0, 1)``.
        wraps: ``(n_atoms, 3)`` integer lattice-vector wraps.
        lattice_residual: Relative difference of the two lattice matrices.
        identity: ``True`` when the run is the analysis cell atom for atom.
    """

    ions: list
    shift: np.ndarray
    wraps: np.ndarray
    lattice_residual: float = 0.0

    @property
    def identity(self) -> bool:
        return (list(self.ions) == list(range(len(self.ions)))
                and not np.any(self.wraps)
                and not np.any(np.abs(self.shift) > POSITION_MATCH_TOL))

    def phases(self, kpoint) -> np.ndarray:
        """The per-atom Bloch factor the transport needs, one per analysis atom.

        The PROCAR projection vector is in VASP's PERIODIC Bloch gauge -- the
        same gauge crystod's site-permutation representation is built in (see
        the engine report, section 2: no gauge conjugation).  A periodic-gauge
        coefficient belongs to an ATOM, not to one image of it, so the
        lattice-vector wraps of :attr:`wraps` cancel against the lattice sum
        of the basis function and the global origin shift contributes the ONE
        factor ``exp(-2 pi i k.shift)`` to every atom alike -- a band phase,
        invisible to irreps, purities, shares and energies.  Transport is
        therefore the bare reordering, and this returns ones; the routine
        exists so that the convention has one place and one docstring, and so
        that ``CRYSTOD_VASP_WRAP_PHASE=atomic`` can put the ATOMIC-gauge
        factor ``exp(-2 pi i k.wraps)`` back for the numerical test that
        decided between the two.

        Args:
            kpoint: The k point in the fractional reciprocal basis of the cell.

        Returns:
            Complex array of shape ``(n_atoms,)``.
        """
        if os.environ.get("CRYSTOD_VASP_WRAP_PHASE") == "atomic":
            return np.exp(-2j * np.pi * (self.wraps @ np.asarray(kpoint, float)))
        return np.ones(len(self.ions), dtype=complex)

    def describe(self, symbols, run_symbols, name="the run") -> str:
        """One line stating the mapping, empty when it is the identity."""
        if self.identity:
            return ""
        order = " ".join(f"{run_symbols[ion]}#{ion + 1}" for ion in self.ions)
        return (f"{name} = the -c cell shifted by {format_shift(self.shift)}; "
                f"atom order {' '.join(symbols)} -> {order}")


def format_shift(shift) -> str:
    """``(1/2, 1/2, 1/2)`` for a fractional translation."""
    from fractions import Fraction

    parts = []
    for value in np.asarray(shift, dtype=float):
        fraction = Fraction(float(value)).limit_denominator(24)
        parts.append(str(fraction) if abs(float(fraction) - value) < 1e-6
                     else f"{value:.6f}")
    return "(" + ", ".join(parts) + ")"


def map_run_onto_cell(lattice, positions, symbols, structure, *, prefer=None,
                      tolerance=POSITION_MATCH_TOL,
                      lattice_rtol=LATTICE_MATCH_RTOL) -> RunMapping:
    """Map a run's POSCAR onto the analysis cell derived from ``-c``.

    The two are allowed to differ by ONE global origin shift, a permutation of
    the atoms and lattice-vector wraps -- the freedom a VASP run has that the
    ``-c`` file does not know about (SrTiO3 written with Ti at the origin
    against a ``-c`` file with Sr at the origin).  Nothing else: the lattice
    matrices must agree entry by entry within ``lattice_rtol``, so a rotated
    or differently chosen basis, a supercell and a different crystal are all
    rejected, each with its own reason.

    A run's ``Va`` point-charge ion stands for a removed atom and therefore
    matches an analysis atom of ANY element; a real ion must carry that atom's
    element.

    Args:
        lattice: ``(3, 3)`` Cartesian lattice of the analysis cell, rows.
        positions: ``(n, 3)`` fractional coordinates of the analysis cell.
        symbols: One chemical symbol per analysis atom.
        structure: The run's :class:`VaspStructure`.
        prefer: A shift to try first (the crystal run's, so that the three
            runs of one calculation are described in the same way).
        tolerance: Fractional-coordinate tolerance.
        lattice_rtol: Relative tolerance of the lattice matrices.

    Returns:
        The :class:`RunMapping`.

    Raises:
        StructureMappingError: They are not the same cell in two settings.
    """
    lattice = np.asarray(lattice, dtype=float)
    positions = np.asarray(positions, dtype=float)
    run_lattice = np.asarray(structure.lattice, dtype=float)
    run_positions = np.asarray(structure.positions, dtype=float)
    if len(structure.symbols) != len(symbols):
        raise StructureMappingError(
            f"the run has {len(structure.symbols)} atoms and the -c primitive "
            f"cell {len(symbols)}", "count")
    scale = np.linalg.norm(lattice)
    residual = float(np.linalg.norm(run_lattice - lattice) / scale)
    if residual > lattice_rtol:
        metric = np.linalg.norm(run_lattice @ run_lattice.T
                                - lattice @ lattice.T) / scale ** 2
        if metric <= lattice_rtol:
            raise StructureMappingError(
                "the run's lattice has the same lengths and angles but a "
                f"rotated basis (relative difference {residual:.2e})", "basis")
        raise StructureMappingError(
            "the run's lattice differs from the -c primitive cell by a "
            f"relative {residual:.2e} (volumes "
            f"{abs(np.linalg.det(run_lattice)):.3f} and "
            f"{abs(np.linalg.det(lattice)):.3f} A^3)", "lattice")
    real = [name for name, charge in zip(structure.symbols, structure.charges)
            if charge is None]
    for name in dict.fromkeys(real):
        if real.count(name) > list(symbols).count(name):
            raise StructureMappingError(
                f"the run has {real.count(name)} real {name} atoms and the -c "
                f"primitive cell {list(symbols).count(name)}", "composition")

    def compatible(atom, ion):
        return (structure.charges[ion] is not None
                or structure.symbols[ion] == symbols[atom])

    def assign(shift):
        """``ions`` for this shift, or ``None``."""
        ions = []
        for atom, position in enumerate(positions):
            difference = run_positions - (position + shift)
            difference -= np.rint(difference)
            hits = np.where(np.all(np.abs(difference) < tolerance, axis=1))[0]
            if len(hits) != 1 or not compatible(atom, int(hits[0])):
                return None
            ions.append(int(hits[0]))
        return ions if len(set(ions)) == len(ions) else None

    candidates = []
    if prefer is not None:
        candidates.append(np.asarray(prefer, dtype=float))
    for ion in range(len(structure.symbols)):
        if compatible(0, ion):
            candidates.append(run_positions[ion] - positions[0])
    seen: list = []
    for shift in candidates:
        shift = shift - np.floor(shift + tolerance)
        if any(np.allclose(shift, other, atol=tolerance) for other in seen):
            continue
        seen.append(shift)
        ions = assign(shift)
        if ions is None:
            continue
        wraps = np.rint(run_positions[ions] - positions - shift).astype(int)
        return RunMapping(ions=ions, shift=shift, wraps=wraps,
                          lattice_residual=residual)
    raise StructureMappingError(
        "no origin shift maps the run's atoms onto the -c primitive cell "
        "(same lattice, but the sites or the elements on them differ)",
        "positions")
