"""Isotropy-subgroup tables of a parent space group and the reverse lookup.

``crystod-group --parent SG --irrep IR`` answers "which subgroups does this
irrep give?".  This module answers the reverse question: given the parent G
and a subgroup type H, which irreps (and which order-parameter directions)
give an isotropy subgroup of type H?  It is the offline counterpart of the
reverse search of the ISOSUBGROUP tool of the ISOTROPY Software Suite
(https://iso.byu.edu).

How it works:

- ``isotropy_table(parent)`` enumerates, once per parent, every stratum of
  every irrep at every special k point of the ISO-IR tables (the irreps of
  ``crystod-group --parent SG --kpoint K`` for every K), with the direction
  pattern, the subgroup type, the cell size, the index, the conventional
  basis and origin of the subgroup in the parent setting, a generic
  order-parameter vector and the projector onto the stratum.
- The table is written to ``<cache>/isotropy/<number>_<version>.json.gz``,
  where ``<cache>`` is ``$CRYSTOD_CACHE_DIR`` or ``~/.cache/crystod`` and
  ``<version>`` is ``crystod.__version__`` (the labels and settings follow
  the conventions of the installed version, so no data is shipped with the
  package).  Later lookups read the file instead of recomputing.
- ``find_isotropy_irreps(parent, child)`` filters the table by subgroup type
  (by number; in a parent with improper operations the enantiomorphic
  partner type of ``ENANTIOMORPHIC_PAIRS`` counts as a match, since an
  improper operation of the parent maps the order parameter to another one
  of the same stratum whose stabilizer is the partner; in a Sohncke parent
  the two types come from different strata and are kept apart), and
  optionally by cell size, index and k point.
- A complex-type pair whose partner sits at the -k star (``K1KA1`` of P3)
  is listed once, at the first of the two stars; ``--kpoint`` with the
  other star selects the same rows.

Several rows of the same type are several embeddings of H in G (different
orientations or origins of the subgroup cell); their conventional basis and
origin tell them apart.  Only single-irrep order parameters at the special
k points are tabulated.  A subgroup that needs two coupled irreps (Pnma
with cell size 4 from the R4+ + M3+ tilts of Pm-3m) is found by
``find_coupled_isotropy_irreps`` (``--child H --coupled``): it searches the
pairs of irreps, first two irreps at the same k point, then pairs at
different k points, and builds the coupled strata of a pair by intersecting
the isotropy subgroups of the single strata of the table (every coupled
isotropy subgroup is ``S1 & S2'`` with ``S1`` an isotropy subgroup of the
first irrep and ``S2'`` any conjugate of one of the second).  Only irreps
with a single stratum whose point group has an order divisible by that of
H (and, with ``--size`` and ``--index``, whose size and index divide the
requested ones) can take part, which prunes most pairs.

If you use this feature, please cite: H. T. Stokes, S. van Orden and
B. J. Campbell, "Tool for Generating Isotropy Subgroups of Crystallographic
Space Groups", J. Appl. Cryst. 49, 1849-1853 (2016).
"""

from __future__ import annotations

import gzip
import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np

CACHE_ENV = "CRYSTOD_CACHE_DIR"
_FORMAT = 2
# sources whose changes can change the table (labels, settings, strata)
_FINGERPRINT_FILES = ("isotropy_subgroup.py", "isotropy_table.py", "isoir.py",
                      "spacegroup_product.py", "CIR_data.txt.gz")
# point groups without improper operations (the 65 Sohncke space groups)
_SOHNCKE_POINT_GROUPS = frozenset(
    {"1", "2", "222", "4", "422", "3", "32", "6", "622", "23", "432"}
)


# ---------------------------------------------------------------- data model


@dataclass(frozen=True)
class IsotropyStratum:
    """One stratum (order-parameter direction type) of one irrep.

    A row of the ``crystod-group --parent SG --kpoint K`` table with the
    setting and the order parameter behind it.

    Attributes:
        label: ISO-IR label of the irrep, e.g. ``"R4+"`` (the pair label
            such as ``"GM2+GM3+"`` for a complex-type pair).
        kname: Name of its special k point, e.g. ``"R"``.
        dimension: Dimension of the order parameter of the irrep.
        direction: Direction pattern, e.g. ``"(0,0,a)"`` (``;`` between
            star arms).
        n_free: Number of free order-parameter parameters.
        number: Space-group number of the isotropy subgroup.
        symbol: Its international short symbol.
        size: Primitive-cell multiplication relative to the parent.
        index: Index of the subgroup in the parent.
        basis: Rows of the conventional cell of the subgroup in parent
            conventional units (as ``--order-parameter`` prints them), or
            ``None`` when the setting could not be standardized.
        origin: Origin of that cell in parent conventional coordinates, or
            ``None``.
        vector: A generic order-parameter vector of the stratum (its
            stabilizer is the isotropy subgroup).
        projector: The orthogonal projector onto the stratum subspace.
    """

    label: str
    kname: str
    dimension: int
    direction: str
    n_free: int
    number: int
    symbol: str
    size: int
    index: int
    basis: np.ndarray | None = field(default=None, compare=False)
    origin: np.ndarray | None = field(default=None, compare=False)
    vector: np.ndarray | None = field(default=None, compare=False)
    projector: np.ndarray | None = field(default=None, compare=False)


@dataclass(frozen=True)
class IsotropyMatch:
    """One answer of the reverse lookup: an irrep stratum giving type H.

    Attributes:
        label: ISO-IR label of the irrep, e.g. ``"R4+"``.
        kname: Its special k point, e.g. ``"R"``.
        direction: Order-parameter direction, e.g. ``"(0,0,a)"``.
        number: Space-group number of the isotropy subgroup (H, or its
            enantiomorphic partner).
        symbol: International short symbol of the subgroup.
        size: Primitive-cell multiplication relative to the parent.
        index: Index of the subgroup in the parent.
        basis: Conventional basis of the subgroup (rows, parent conventional
            units), or ``None``.
        origin: Origin of the subgroup cell (parent conventional
            coordinates), or ``None``.
    """

    label: str
    kname: str
    direction: str
    number: int
    symbol: str
    size: int
    index: int
    basis: np.ndarray | None = field(default=None, compare=False)
    origin: np.ndarray | None = field(default=None, compare=False)

    def __str__(self) -> str:
        return (
            f"{self.label}{self.direction} ({self.kname}) -> {self.symbol} "
            f"(No. {self.number}), size {self.size}, index {self.index}"
        )


@dataclass
class IsotropyTable:
    """Every single-irrep stratum of a parent space group.

    Returned by :func:`isotropy_table`.

    Attributes:
        space_group: International short symbol of the parent.
        number: Its number.
        version: The CrystOD version that built the table.
        kpoints: Maps every special k-point name, in table order, to a dict
            with ``coordinates`` (the tabulated arm, primitive reciprocal
            basis), ``arms`` (all arms), ``n_arms`` and ``listed_at`` (the
            k points under which its irreps are listed: itself, or the
            first star of a pair that spans both the k and the -k star).
        irreps: ``(label, kname, dimension)`` of every analyzed irrep, in
            table order.
        strata: Every :class:`IsotropyStratum`, irrep by irrep, each irrep
            in the order of the ``--parent`` table.
        errors: Maps the label of an irrep that could not be enumerated to
            the reason.
        sohncke: ``True`` when the parent has no improper operation (then
            an enantiomorphic partner type is not a match).
        fingerprint: Hash of the sources the table was built with; a
            cached table with another fingerprint is rebuilt.
        cache_path: The cache file read or written, or ``None``.
        from_cache: ``True`` when the table was read from ``cache_path``.
    """

    space_group: str
    number: int
    version: str
    kpoints: dict
    irreps: list
    strata: list
    errors: dict
    sohncke: bool = False
    fingerprint: str = ""
    cache_path: str | None = None
    from_cache: bool = False

    def select(self, numbers=None, size=None, index=None, kpoints=None) -> list:
        """The strata passing every given filter.

        Args:
            numbers: Subgroup numbers to keep (any iterable), or ``None``.
            size: Cell size to keep, or ``None``.
            index: Index to keep, or ``None``.
            kpoints: k-point names to keep, or ``None``.

        Returns:
            The matching :class:`IsotropyStratum` records, in table order.
        """
        numbers = None if numbers is None else set(int(n) for n in numbers)
        knames = None if kpoints is None else set(kpoints)
        return [
            s for s in self.strata
            if (numbers is None or s.number in numbers)
            and (size is None or s.size == size)
            and (index is None or s.index == index)
            and (knames is None or s.kname in knames)
        ]

    def to_dict(self) -> dict:
        """The JSON form written to the cache file."""

        def array(value, digits):
            if value is None:
                return None
            return (np.round(np.asarray(value, dtype=float), digits) + 0.0).tolist()

        return {
            "format": _FORMAT,
            "crystod_version": self.version,
            "fingerprint": self.fingerprint,
            "sohncke": self.sohncke,
            "space_group": self.space_group,
            "number": self.number,
            "kpoints": self.kpoints,
            "irreps": [list(entry) for entry in self.irreps],
            "errors": self.errors,
            "strata": [
                {
                    "label": s.label, "kname": s.kname, "dimension": s.dimension,
                    "direction": s.direction, "n_free": s.n_free,
                    "number": s.number, "symbol": s.symbol,
                    "size": s.size, "index": s.index,
                    "basis": array(s.basis, 6), "origin": array(s.origin, 6),
                    "vector": array(s.vector, 10),
                    "projector": array(s.projector, 10),
                }
                for s in self.strata
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> IsotropyTable:
        """Inverse of :meth:`to_dict`.

        Raises:
            ValueError: The data is not a table of the current format or was
                built from other sources.
        """
        if data.get("format") != _FORMAT:
            raise ValueError("unknown isotropy-table format")
        if data.get("fingerprint") != _fingerprint():
            raise ValueError("isotropy table built from other sources")

        def array(value):
            return None if value is None else np.array(value, dtype=float)

        strata = [
            IsotropyStratum(
                label=row["label"], kname=row["kname"], dimension=int(row["dimension"]),
                direction=row["direction"], n_free=int(row["n_free"]),
                number=int(row["number"]), symbol=row["symbol"],
                size=int(row["size"]), index=int(row["index"]),
                basis=array(row["basis"]), origin=array(row["origin"]),
                vector=array(row["vector"]), projector=array(row["projector"]),
            )
            for row in data["strata"]
        ]
        return cls(
            space_group=data["space_group"], number=int(data["number"]),
            version=data["crystod_version"], kpoints=data["kpoints"],
            irreps=[tuple(entry) for entry in data["irreps"]],
            strata=strata, errors=dict(data["errors"]),
            sohncke=bool(data["sohncke"]), fingerprint=data["fingerprint"],
        )


# ---------------------------------------------------------------- building


def cache_directory() -> str:
    """The directory of the isotropy-table cache files.

    Returns:
        ``$CRYSTOD_CACHE_DIR/isotropy`` when the variable is set, otherwise
        ``~/.cache/crystod/isotropy``.
    """
    root = os.environ.get(CACHE_ENV) or os.path.join(
        os.path.expanduser("~"), ".cache", "crystod"
    )
    return os.path.join(root, "isotropy")


def cache_path(number: int) -> str:
    """The cache file of one parent space group (by number)."""
    from . import __version__

    return os.path.join(cache_directory(), f"{int(number)}_{__version__}.json.gz")


_FINGERPRINT: list = []


def _fingerprint() -> str:
    """Short hash of the sources behind the table (computed once).

    Keys the cache beyond the version string, so that a change of the
    label, setting or enumeration code within one version (an editable
    install, an unreleased version) rebuilds the cached tables.
    """
    if not _FINGERPRINT:
        import hashlib

        digest = hashlib.sha1()
        directory = os.path.dirname(os.path.abspath(__file__))
        for name in _FINGERPRINT_FILES:
            digest.update(name.encode())
            try:
                with open(os.path.join(directory, name), "rb") as handle:
                    digest.update(handle.read())
            except OSError:
                digest.update(b"missing")
        _FINGERPRINT.append(digest.hexdigest()[:16])
    return _FINGERPRINT[0]


def is_sohncke(parent) -> bool:
    """``True`` when the space group has no improper operation.

    In such a (Sohncke) parent the two members of an enantiomorphic pair of
    subgroup types are not conjugate and come from different strata.
    """
    return str(_resolve(parent).pointgroup_international) in _SOHNCKE_POINT_GROUPS


def _resolve(space_group):
    """spglib type info of a space group given by symbol or number."""
    from .spacegroup_product import _resolve_space_group

    if isinstance(space_group, (int, np.integer)):
        space_group = str(int(space_group))
    return _resolve_space_group(str(space_group))


def _progress_to_stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def build_isotropy_table(parent, progress=_progress_to_stderr) -> IsotropyTable:
    """Enumerate every stratum of every special-point irrep (no cache).

    Args:
        parent: Symbol or number of the parent space group.
        progress: Called with one line per k point before it is enumerated
            (``"building the isotropy table of Fm-3m: W (4 of 4) ..."``);
            ``None`` for silence.

    Returns:
        The :class:`IsotropyTable` (``cache_path`` unset).

    Raises:
        SystemExit: Unknown space group.
    """
    from . import __version__
    from .isotropy_subgroup import _direction_records, kpoint_irrep_tables
    from .spacegroup_product import DEN, SpaceGroupIrrepAlgebra

    algebra = SpaceGroupIrrepAlgebra(str(parent))
    symbol = str(algebra.sg_type.international_short)
    knames = list(algebra.k_by_kname)  # the tabulated special points only
    kpoints: dict = {}
    irreps: list = []
    strata: list = []
    errors: dict = {}
    first_kname: dict = {}  # irrep label -> the k point it is listed at
    for position, kname in enumerate(knames, start=1):
        if progress is not None:
            progress(f"building the isotropy table of {symbol}: {kname} "
                     f"({position} of {len(knames)}) ...")
        arms, _ = algebra.star(kname)
        kpoints[kname] = {
            "coordinates": [int(v) / DEN for v in algebra.k_by_kname[kname]],
            "arms": [[int(v) / DEN for v in arm] for arm in arms],
            "n_arms": len(arms),
            "listed_at": [],
        }

        def rows_of(analyzer, kname=kname):
            representation = analyzer.representation
            label = representation.label
            rows = []
            for index, n_free, full_label, info, size, extra in _direction_records(analyzer):
                _, _, _, B, rotations, translations, lattice = extra["subgroup"]
                try:
                    setting = analyzer.conventional_setting(
                        B, rotations, translations, lattice, info
                    )
                except Exception:
                    setting = None
                basis, origin = (
                    (setting[0] + 0.0, setting[1] + 0.0) if setting is not None
                    else (None, None)
                )
                rows.append(IsotropyStratum(
                    label=label, kname=kname, dimension=int(representation.dimension),
                    direction=full_label[len(label):], n_free=int(n_free),
                    number=int(info.number), symbol=str(info.international_short),
                    size=int(size), index=int(index), basis=basis, origin=origin,
                    vector=np.asarray(extra["generic"], dtype=float),
                    projector=np.real(np.asarray(extra["projector"])),
                ))
            return int(representation.dimension), rows

        tables, failed = kpoint_irrep_tables(algebra, kname, rows_of)
        listed_at = kpoints[kname]["listed_at"]
        for label, (dimension, rows) in tables.items():
            # a pair spanning the k and the -k star (H1HA1 at H and at HA)
            # is one order parameter: keep it at the first star only
            home = first_kname.setdefault(label, kname)
            if home not in listed_at:
                listed_at.append(home)
            if home != kname:
                continue
            irreps.append((label, kname, dimension))
            strata.extend(rows)
        for label, reason in failed.items():
            home = first_kname.setdefault(label, kname)
            if home not in listed_at:
                listed_at.append(home)
            errors.setdefault(label, reason)
        if not listed_at:
            listed_at.append(kname)
    return IsotropyTable(
        space_group=symbol, number=int(algebra.sg_type.number), version=__version__,
        kpoints=kpoints, irreps=irreps, strata=strata, errors=errors,
        sohncke=str(algebra.sg_type.pointgroup_international) in _SOHNCKE_POINT_GROUPS,
        fingerprint=_fingerprint(),
    )


def _read_cache(path: str) -> IsotropyTable | None:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            table = IsotropyTable.from_dict(json.load(handle))
    except (OSError, ValueError, KeyError, TypeError, EOFError):
        return None  # missing or unreadable: rebuild
    table.cache_path = path
    table.from_cache = True
    return table


def _write_cache(table: IsotropyTable, path: str) -> bool:
    directory = os.path.dirname(path)
    temporary = f"{path}.{os.getpid()}.tmp"
    try:
        os.makedirs(directory, exist_ok=True)
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            json.dump(table.to_dict(), handle, separators=(",", ":"))
        os.replace(temporary, path)
    except OSError as exc:
        try:
            os.remove(temporary)
        except OSError:
            pass
        print(f"WARNING: the isotropy table could not be cached at {path} ({exc}).",
              file=sys.stderr)
        return False
    return True


def _isotropy_table(parent, *, use_cache=True, write_cache=True,
                    progress=_progress_to_stderr) -> IsotropyTable:
    info = _resolve(parent)
    path = cache_path(info.number)
    if use_cache:
        table = _read_cache(path)
        if table is not None:
            return table
    table = build_isotropy_table(str(info.number), progress=progress)
    if write_cache and _write_cache(table, path):
        table.cache_path = path
    return table


def isotropy_table(parent, *, use_cache: bool = True, write_cache: bool = True,
                   progress: bool = True) -> IsotropyTable:
    """The isotropy table of a parent space group, cached on disk.

    Every stratum of every irrep at every special k point of the ISO-IR
    tables, as ``crystod-group --parent SG --kpoint K`` lists them for all
    K, with the conventional settings of the subgroups.  The first call for
    a parent builds the table (seconds for most groups, minutes for the
    large stars of the face-centred cubic groups) and writes it to
    :func:`cache_path`; later calls read the file.

    Args:
        parent: Symbol (``"Pm-3m"``) or number (``221``) of the parent.
        use_cache: Read the cache file when it exists (``False`` rebuilds
            the table; the file is still written).
        write_cache: Write the table to the cache file after building it.
        progress: Print one line per k point to stderr while building.

    Returns:
        An :class:`IsotropyTable`.

    Raises:
        ValueError: Unknown space group.

    Example:
        >>> from crystod import group
        >>> table = group.isotropy_table("Pm-3m")
        >>> [(s.label, s.direction, s.symbol) for s in table.select([167], size=2)]
        [('R4+', '(a,a,a)', 'R-3c'), ('R5-', '(a,a,a)', 'R-3c')]
    """
    try:
        return _isotropy_table(
            parent, use_cache=use_cache, write_cache=write_cache,
            progress=_progress_to_stderr if progress else None,
        )
    except SystemExit as exc:
        message = " ".join(str(exc).split())
        raise ValueError(message.removeprefix("ERROR: ") or "invalid input") from None


# ---------------------------------------------------------------- lookup


def resolve_child(child):
    """Number and symbol of the subgroup type given by symbol or number.

    Raises:
        SystemExit: Not a space-group symbol or number.
    """
    text = str(child).strip()
    if text.isdigit() and not 1 <= int(text) <= 230:
        raise SystemExit(f'ERROR: "{child}" is not a space-group number (1-230).')
    info = _resolve(text)
    return int(info.number), str(info.international_short)


def child_numbers(number: int, include_partner: bool = True) -> list[int]:
    """The subgroup numbers that count as type ``number``.

    Args:
        number: Space-group number of the subgroup type H.
        include_partner: Also count the enantiomorphic partner type of H
            (right for a parent with improper operations, wrong for a
            Sohncke parent, see :func:`is_sohncke`).

    Returns:
        ``[number]``, followed by the partner number when it is counted.
    """
    from .isotropy_subgroup import ENANTIOMORPHIC_PAIRS

    partner = ENANTIOMORPHIC_PAIRS.get(int(number)) if include_partner else None
    return [int(number)] + ([partner] if partner is not None else [])


def resolve_kpoints(table: IsotropyTable, tokens) -> list[str]:
    """k-point names of the table from ``--kpoint`` values.

    Args:
        table: The :class:`IsotropyTable`.
        tokens: k-point names (case ignored, ``G``/``GAMMA`` for ``GM``),
            or three coordinates of one special point in the primitive
            basis (any arm, modulo reciprocal lattice vectors).

    Returns:
        The names under which the selected irreps are listed, in the
        order given (the -k star of a pair spanning both stars maps to the
        first star).

    Raises:
        SystemExit: A name or a point that is not a special point of the
            table.
    """
    from fractions import Fraction

    from .isotropy_subgroup import _GAMMA_ALIASES

    tokens = [piece for token in tokens for piece in str(token).replace(",", " ").split()]
    available = ", ".join(table.kpoints)
    numeric = all(token[:1].isdigit() or token[:1] in "+-." for token in tokens)
    if numeric and len(tokens) == 3:
        try:
            k = np.array([float(Fraction(token)) for token in tokens])
        except (ValueError, ZeroDivisionError):
            raise SystemExit(f"ERROR: invalid k point {' '.join(tokens)}.") from None
        for kname, entry in table.kpoints.items():
            for arm in entry["arms"]:
                d = k - np.asarray(arm)
                if np.allclose(d, np.rint(d), atol=5e-4):
                    return list(entry.get("listed_at") or [kname])
        raise SystemExit(
            f"ERROR: k = ({', '.join(tokens)}) is not a special k point of "
            f"{table.space_group} (No. {table.number}); special points: {available}."
        )
    by_upper = {kname.upper(): kname for kname in table.kpoints}
    names = []
    for token in tokens:
        requested = token.upper()
        if requested not in by_upper and requested in _GAMMA_ALIASES:
            requested = "GM"
        if requested not in by_upper:
            raise SystemExit(
                f'ERROR: "{token}" is not a special k point of {table.space_group} '
                f"(No. {table.number}); special points: {available}."
            )
        kname = by_upper[requested]
        for name in table.kpoints[kname].get("listed_at") or [kname]:
            if name not in names:
                names.append(name)
    return names


def _match(stratum: IsotropyStratum) -> IsotropyMatch:
    return IsotropyMatch(
        label=stratum.label, kname=stratum.kname, direction=stratum.direction,
        number=stratum.number, symbol=stratum.symbol, size=stratum.size,
        index=stratum.index, basis=stratum.basis, origin=stratum.origin,
    )


def find_isotropy_irreps(parent, child, size=None, index=None, kpoints=None,
                         *, coupled: bool = False, use_cache: bool = True) -> list:
    """Irreps and directions whose isotropy subgroup is of a given type.

    The reverse lookup of ``crystod-group --parent G --child H``: every
    stratum of every special-point irrep of G whose isotropy subgroup is of
    type H, read from the cached :func:`isotropy_table`.  When G has an
    improper operation, a stratum listed with the enantiomorphic partner
    type of H is a match too (an improper operation of G maps its order
    parameter to one of the same stratum with a stabilizer of type H); in a
    Sohncke parent the two types are different strata and only H counts.

    Args:
        parent: Symbol or number of the parent space group G.
        child: Symbol (``"I4/mcm"``) or number (``140``) of the subgroup
            type H.
        size: Keep only this primitive-cell multiplication.
        index: Keep only this index [G:H].
        kpoints: Keep only these special k points (names, or one point as
            three coordinates).
        coupled: Also search the pairs of irreps (``--coupled``) and append
            the :class:`CoupledIsotropyMatch` records of
            :func:`find_coupled_isotropy_irreps` after the single-irrep ones.
        use_cache: Read the cached table when it exists.

    Returns:
        A list of :class:`IsotropyMatch`, in table order (k point, irrep,
        then the order of the ``--parent`` table); several rows of one type
        are distinct embeddings (conventional basis and origin). Empty when
        no single irrep gives type H: the subgroup may need coupled irreps
        (``coupled=True``).

    Raises:
        ValueError: Unknown parent or child, or a k point that is not a
            special point of the parent.

    Example:
        >>> from crystod import group
        >>> for match in group.find_isotropy_irreps("Pm-3m", "R-3c", size=2):
        ...     print(match)
        R4+(a,a,a) (R) -> R-3c (No. 167), size 2, index 8
        R5-(a,a,a) (R) -> R-3c (No. 167), size 2, index 8
    """
    try:
        number, _ = resolve_child(child)
        table = _isotropy_table(parent, use_cache=use_cache, progress=None)
        knames = None if kpoints is None else resolve_kpoints(
            table, [kpoints] if isinstance(kpoints, str) else kpoints
        )
    except SystemExit as exc:
        message = " ".join(str(exc).split())
        raise ValueError(message.removeprefix("ERROR: ") or "invalid input") from None
    matches = [
        _match(s)
        for s in table.select(child_numbers(number, not table.sohncke),
                              size=size, index=index, kpoints=knames)
    ]
    if coupled:
        matches += coupled_search(table, number, size=size, index=index,
                                  knames=knames).matches
    return matches


# ---------------------------------------------------------------- coupled lookup

# order of every crystallographic point group (spglib's international symbols)
_POINT_GROUP_ORDER = {
    "1": 1, "-1": 2, "2": 2, "m": 2, "2/m": 4, "222": 4, "mm2": 4, "mmm": 8,
    "4": 4, "-4": 4, "4/m": 8, "422": 8, "4mm": 8, "-42m": 8, "-4m2": 8,
    "4/mmm": 16, "3": 3, "-3": 6, "32": 6, "3m": 6, "-3m": 12, "6": 6, "-6": 6,
    "6/m": 12, "622": 12, "6mm": 12, "-6m2": 12, "-62m": 12, "6/mmm": 24,
    "23": 12, "m-3": 24, "432": 24, "-43m": 24, "m-3m": 48,
}
_PG_ORDER_BY_NUMBER: dict = {}


def point_group_order(number: int) -> int:
    """Order of the point group of the space-group type ``number``."""
    if not _PG_ORDER_BY_NUMBER:
        import spglib

        from .runtime_compat import get_spacegroup_type

        for hall_number in range(1, 531):
            info = get_spacegroup_type(spglib.get_spacegroup_type(hall_number))
            _PG_ORDER_BY_NUMBER.setdefault(
                int(info.number), _POINT_GROUP_ORDER[str(info.pointgroup_international)]
            )
    return _PG_ORDER_BY_NUMBER[int(number)]


@dataclass(frozen=True)
class CoupledIsotropyMatch:
    """One answer of the coupled reverse lookup: two irreps giving type H.

    A coupled stratum of the direct sum of two irreps (both amplitudes
    nonzero) whose isotropy subgroup has type H, where neither irrep alone
    (with the other one as a secondary order parameter) has that isotropy
    subgroup.

    Attributes:
        label1: ISO-IR label of the first irrep (table order), e.g. ``"R4+"``.
        kname1: Its special k point.
        direction1: Its part of the direction, e.g. ``"(0,a,a)"``.
        label2: Label of the second irrep, e.g. ``"M3+"``.
        kname2: Its special k point.
        direction2: Its part of the direction, e.g. ``"(d;0;0)"`` (the
            letters continue after those of the first irrep, as in
            ``crystod-group --parent SG --irrep IR1 IR2``).
        number: Space-group number of the isotropy subgroup (H, or its
            enantiomorphic partner).
        symbol: Its international short symbol.
        size: Primitive-cell multiplication relative to the parent.
        index: Index of the subgroup in the parent.
        basis: Conventional basis of the subgroup (rows, parent conventional
            units), or ``None``.
        origin: Origin of the subgroup cell (parent conventional
            coordinates), or ``None``.
        vector: A generic order-parameter vector of the stratum (the first
            irrep's components, then the second's).
        projector: The orthogonal projector onto the stratum subspace in
            the direct sum.
    """

    label1: str
    kname1: str
    direction1: str
    label2: str
    kname2: str
    direction2: str
    number: int
    symbol: str
    size: int
    index: int
    basis: np.ndarray | None = field(default=None, compare=False)
    origin: np.ndarray | None = field(default=None, compare=False)
    vector: np.ndarray | None = field(default=None, compare=False)
    projector: np.ndarray | None = field(default=None, compare=False)

    @property
    def direction(self) -> str:
        """The coupled direction, e.g. ``"R4+(0,a,a) M3+(d;0;0)"``."""
        return f"{self.label1}{self.direction1} {self.label2}{self.direction2}"

    @property
    def same_k(self) -> bool:
        """``True`` when both irreps belong to the same special k point."""
        return self.kname1 == self.kname2

    def __str__(self) -> str:
        return (
            f"{self.direction} -> {self.symbol} (No. {self.number}), "
            f"size {self.size}, index {self.index}"
        )


@dataclass
class CoupledSearch:
    """The result of a coupled reverse lookup with its search statistics.

    Attributes:
        matches: The :class:`CoupledIsotropyMatch` records, sorted by index,
            then size, then the order of the irrep pairs.
        n_irreps: Number of irreps that passed the point-group filter.
        n_pairs_same_k: Number of irrep pairs searched at the same k point.
        n_pairs_cross_k: Number of irrep pairs searched at different k points.
        errors: Maps an irrep pair (``"X1 + X2"``) that could not be
            analyzed to the reason.
        data: Per match (same order), the objects behind it: the
            ``algebra``, the coupled ``representation``, the stabilizer
            ``members`` and the sublattice basis ``B`` (for the secondary
            order parameters).
    """

    matches: list
    n_irreps: int = 0
    n_pairs_same_k: int = 0
    n_pairs_cross_k: int = 0
    errors: dict = field(default_factory=dict)
    data: list = field(default_factory=list, repr=False)


def _fixing_mask(blocks: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Mask of the matrices of ``blocks`` (``(G, n, n)``) fixing every
    column of ``basis`` (tolerance 1e-6, as ``_RepCache.fixes``)."""
    D = np.matmul(blocks, basis) - basis[None]
    return np.max(np.abs(D), axis=(1, 2)) < 1e-6


def _orbit(cache, projector: np.ndarray) -> dict:
    """``_RepCache.orbit`` with the keys of all images rounded at once (the
    same keys as ``_projector_key``)."""
    images = np.matmul(np.matmul(cache.Ed, projector), cache.Ed.transpose(0, 2, 1))
    rounded = np.round(images, 6) + 0.0
    orbit: dict = {}
    for image, key_array in zip(images, rounded):
        orbit.setdefault(key_array.tobytes(), image)
    return orbit


_OWN_MASKS: dict = {}


def _stratum_masks(representation, strata, with_orbit: bool) -> np.ndarray:
    """Distinct stabilizer masks, over the elements of ``representation``
    itself, of the given strata (``with_orbit``: of every orbit image).

    Kept per representation and stratum list: every pair that contains the
    irrep needs them.
    """
    from .isotropy_subgroup import _orth_basis, _rep_cache

    cache = _rep_cache(representation)
    key = (id(cache), tuple(id(s) for s in strata), with_orbit)
    entry = _OWN_MASKS.get(key)
    if entry is not None and entry[0] is cache:
        return entry[1]
    bases = []
    for stratum in strata:
        if with_orbit:
            bases.extend(_orth_basis(image)
                         for image in _orbit(cache, stratum.projector).values())
        else:
            bases.append(_orth_basis(stratum.projector))
    masks: dict = {}
    by_rank: dict = {}
    for basis in bases:
        by_rank.setdefault(basis.shape[1], []).append(basis)
    for group in by_rank.values():
        for start in range(0, len(group), 32):
            B = np.array(group[start:start + 32])  # (m, n, r)
            D = np.matmul(cache.E[None], B[:, None]) - B[:, None]
            for mask in np.max(np.abs(D), axis=(2, 3)) < 1e-6:
                masks.setdefault(np.packbits(mask).tobytes(), mask)
    result = np.array(list(masks.values())) if masks else np.zeros((0, len(cache.E)), bool)
    if len(_OWN_MASKS) > 512:
        _OWN_MASKS.clear()
    _OWN_MASKS[key] = (cache, result)
    return result


def _element_map(representation, coupled) -> np.ndarray:
    """For every element of ``coupled``, the index of the element of
    ``representation`` (one of its parts) with the same matrix block."""
    lookup = {
        (int(i), tuple(int(x) for x in t)): g
        for g, (i, t, _) in enumerate(representation.elements)
    }
    grid = int(representation.grid_n)
    return np.array([
        lookup[(int(i), tuple(int(x) % grid for x in t))] for i, t, _ in coupled.elements
    ])


def _split_coupled_label(label: str, label1: str, label2: str) -> tuple[str, str]:
    """``"R4+(0,a,a) M3+(d;0;0)"`` -> ``("(0,a,a)", "(d;0;0)")``."""
    first, second = label.split(" ", 1)
    return first[len(label1):], second[len(label2):]


def coupled_search(table: IsotropyTable, number: int, size=None, index=None,
                   knames=None, progress=None, keep_data: bool = False) -> CoupledSearch:
    """Search the irrep pairs of a parent for coupled strata of type H.

    The engine of ``--child H --coupled`` and of
    :func:`find_coupled_isotropy_irreps`.  Every isotropy subgroup of the
    direct sum ``D1 + D2`` with both parts nonzero is ``S1 & S2'``, with
    ``S1`` the isotropy subgroup of a single stratum of ``D1`` (a
    representative of the table) and ``S2'`` any conjugate of one of
    ``D2``; its isotropy subspace is ``Fix_1 + Fix_2`` (group-average
    projectors).  The strata are the parent orbits of these subspaces, with
    the representative and the direction label that ``crystod-group
    --parent G --irrep D1 D2`` prints.

    Args:
        table: The :class:`IsotropyTable` of the parent (the single strata
            of every irrep are the candidates).
        number: Space-group number of the subgroup type H (the
            enantiomorphic partner counts in a parent with improper
            operations, see :func:`child_numbers`).
        size: Keep only this cell size.
        index: Keep only this index.
        knames: Use only the irreps of these k points (names of the table),
            or ``None`` for every k point.
        progress: Called with one line per pair of k points once the search
            has run for more than three seconds; ``None`` for silence.
        keep_data: Keep the representation, the stabilizer and the
            sublattice of every match in ``data``.

    Returns:
        A :class:`CoupledSearch`.
    """
    import time

    from .isotropy_subgroup import _failure_reason, _label_rank, kpoint_irrep_tables
    from .spacegroup_product import SpaceGroupIrrepAlgebra

    start = time.time()
    numbers = set(child_numbers(number, not table.sohncke))
    target = point_group_order(number)

    def candidate(stratum) -> bool:
        return (point_group_order(stratum.number) % target == 0
                and (size is None or size % stratum.size == 0)
                and (index is None or index % stratum.index == 0))

    strata_by_label: dict = {}
    for stratum in table.strata:
        if candidate(stratum):
            strata_by_label.setdefault(stratum.label, []).append(stratum)
    allowed = None if knames is None else set(knames)
    irreps = [
        (label, kname) for label, kname, _ in table.irreps
        if label in strata_by_label and (allowed is None or kname in allowed)
    ]
    result = CoupledSearch(matches=[], n_irreps=len(irreps))
    if len(irreps) < 2:
        return result

    knames_order = list(dict.fromkeys(kname for _, kname in irreps))
    groups = [(k, k) for k in knames_order] + [
        (k1, k2) for i, k1 in enumerate(knames_order) for k2 in knames_order[i + 1:]
    ]
    algebra = SpaceGroupIrrepAlgebra(str(table.number))
    identity = algebra._rotation_index[algebra._key(np.eye(3))]
    representations: dict = {}

    def representation_of(label: str, kname: str):
        if kname not in representations:
            tables, _ = kpoint_irrep_tables(
                algebra, kname, lambda analyzer: analyzer.representation
            )
            representations[kname] = tables
        return representations[kname].get(label)

    found = []
    position = 0
    for group_number, (k1, k2) in enumerate(groups, start=1):
        if progress is not None and time.time() - start > 3.0:
            progress(f"searching the coupled irrep pairs of {table.space_group}: "
                     f"{k1} + {k2} ({group_number} of {len(groups)}) ...")
        labels1 = [label for label, kname in irreps if kname == k1]
        labels2 = [label for label, kname in irreps if kname == k2]
        if k1 == k2:
            pairs = [(a, b) for i, a in enumerate(labels1) for b in labels1[i + 1:]]
            result.n_pairs_same_k += len(pairs)
        else:
            pairs = [(a, b) for a in labels1 for b in labels2]
            result.n_pairs_cross_k += len(pairs)
        for label1, label2 in pairs:
            position += 1
            try:
                rep1 = representation_of(label1, k1)
                rep2 = representation_of(label2, k2)
                if rep1 is None or rep2 is None:
                    raise RuntimeError("representation not available")
                for match, extra in _coupled_pair(
                    algebra, identity, rep1, rep2, k1, k2,
                    strata_by_label[label1], strata_by_label[label2],
                    numbers, target, size, index, keep_data,
                ):
                    found.append((match.index, match.size, position,
                                  _label_rank(match.direction), match, extra))
            except (SystemExit, Exception) as exc:  # noqa: BLE001 - reported
                result.errors[f"{label1} + {label2}"] = _failure_reason(exc)
    found.sort(key=lambda item: item[:4])
    result.matches = [item[4] for item in found]
    result.data = [item[5] for item in found] if keep_data else []
    return result


def _coupled_pair(algebra, identity, rep1, rep2, k1, k2, strata1, strata2, numbers,
                  target, size, index, keep_data):
    """The coupled strata of type H of one irrep pair (generator of
    ``(CoupledIsotropyMatch, data)``)."""
    from .isotropy_subgroup import (
        CoupledRepresentation,
        IsotropyAnalyzer,
        _label_rank,
        _orth_basis,
        _projector_key,
        _rep_cache,
    )

    coupled = CoupledRepresentation.from_parts(algebra, [rep1, rep2])
    cache = _rep_cache(coupled)
    n1 = rep1.dimension
    E1 = cache.E[:, :n1, :n1]
    E2 = cache.E[:, n1:, n1:]
    own1 = _stratum_masks(rep1, strata1, with_orbit=False)
    own2 = _stratum_masks(rep2, strata2, with_orbit=True)
    if not len(own1) or not len(own2):
        return
    masks1 = own1[:, _element_map(rep1, coupled)]
    masks2 = own2[:, _element_map(rep2, coupled)]
    elements = coupled.elements
    grid_volume = int(coupled.grid_n) ** 3
    pure = np.array([i == identity for i, _, _ in elements])
    rotation_index = np.array([i for i, _, _ in elements])
    one_hot = np.zeros((len(elements), algebra.n_ops), dtype=np.float32)
    one_hot[np.arange(len(elements)), rotation_index] = 1.0
    # the isotropy subgroup of Fix(h) contains h: an h with more rotations
    # than H (or a smaller cell than --size) cannot give H
    intersections: dict = {}
    for mask1 in masks1:
        H = masks2 & mask1[None, :]
        n_rot = np.count_nonzero(H.astype(np.float32) @ one_hot, axis=1)
        keep = n_rot <= target
        if size is not None:
            keep &= np.count_nonzero(H & pure[None, :], axis=1) <= grid_volume // size
        for h, packed in zip(H[keep], np.packbits(H[keep], axis=1)):
            intersections.setdefault(packed.tobytes(), h)
    used: set = set()
    analyzer = None
    for h in intersections.values():
        basis1 = _orth_basis(E1[h].mean(axis=0))
        basis2 = _orth_basis(E2[h].mean(axis=0))
        if basis1.shape[1] == 0 or basis2.shape[1] == 0:
            continue
        W = np.zeros((coupled.dimension, basis1.shape[1] + basis2.shape[1]))
        W[:n1, :basis1.shape[1]] = basis1
        W[n1:, basis1.shape[1]:] = basis2
        projector = W @ W.T
        key = _projector_key(projector)
        if key in used:
            continue
        # every test below is a property of the conjugacy class, so a
        # rejected subspace needs no orbit
        full = _fixing_mask(cache.E, W)
        # one irrep alone already has this isotropy subgroup (the other one
        # is a secondary order parameter of it): not a coupled answer
        count = int(full.sum())
        if (int(_fixing_mask(E1, basis1).sum()) == count
                or int(_fixing_mask(E2, basis2).sum()) == count):
            continue
        # cheap type filters before the spglib identification
        n_point = len(set(rotation_index[full].tolist()))
        if n_point != target:
            continue
        n_pure = int(np.sum(full & pure))
        cell = grid_volume // max(n_pure, 1)
        if size is not None and cell != size:
            continue
        if index is not None and algebra.n_ops * cell // n_point != index:
            continue
        orbit = _orbit(cache, projector)
        used.update(orbit)
        if analyzer is None:
            analyzer = IsotropyAnalyzer.from_representation(algebra, coupled)
        # the type is a property of the class: identify it before choosing
        # the representative (the label search over the orbit is the
        # expensive part)
        info = analyzer.subgroup_of(
            [(elements[g][0], elements[g][1]) for g in np.nonzero(full)[0]]
        )[0]
        if int(info.number) not in numbers:
            continue
        best = min(orbit.values(),
                   key=lambda Q: _label_rank(analyzer._direction_pattern(Q)[0]))
        label, generic = analyzer.direction_label(best)
        members = [
            (elements[g][0], elements[g][1])
            for g in np.nonzero(cache.vector_stabilizer_mask(generic))[0]
        ]
        info, cell_size, subgroup_index, B, rotations, translations, lattice = (
            analyzer.subgroup_of(members)
        )
        if int(info.number) not in numbers:
            continue
        if size is not None and cell_size != size:
            continue
        if index is not None and subgroup_index != index:
            continue
        try:
            setting = analyzer.conventional_setting(
                B, rotations, translations, lattice, info
            )
        except Exception:  # noqa: BLE001 - the setting is optional
            setting = None
        direction1, direction2 = _split_coupled_label(label, rep1.label, rep2.label)
        match = CoupledIsotropyMatch(
            label1=rep1.label, kname1=k1, direction1=direction1,
            label2=rep2.label, kname2=k2, direction2=direction2,
            number=int(info.number), symbol=str(info.international_short),
            size=int(cell_size), index=int(subgroup_index),
            basis=None if setting is None else setting[0] + 0.0,
            origin=None if setting is None else setting[1] + 0.0,
            vector=np.asarray(generic, dtype=float), projector=np.real(best),
        )
        extra = ({"algebra": algebra, "representation": coupled,
                  "members": members, "B": B} if keep_data else None)
        yield match, extra


def find_coupled_isotropy_irreps(parent, child, size=None, index=None, kpoints=None,
                                 *, use_cache: bool = True) -> list:
    """Irrep pairs and directions whose coupled isotropy subgroup has type H.

    The coupled reverse lookup of ``crystod-group --parent G --child H
    --coupled``: every stratum of the direct sum of two special-point
    irreps of G, with both amplitudes nonzero, whose isotropy subgroup has
    type H while neither irrep alone has it.  The pairs are searched at the
    same k point first, then at different k points; the single strata of
    the cached :func:`isotropy_table` are the candidates.

    Args:
        parent: Symbol or number of the parent space group G.
        child: Symbol or number of the subgroup type H.
        size: Keep only this primitive-cell multiplication.
        index: Keep only this index [G:H].
        kpoints: Use only the irreps of these special k points.
        use_cache: Read the cached table when it exists.

    Returns:
        A list of :class:`CoupledIsotropyMatch`, sorted by index, then by
        size, then in the order of the irrep pairs.

    Raises:
        ValueError: Unknown parent or child, or a k point that is not a
            special point of the parent.

    Example:
        >>> from crystod import group
        >>> for match in group.find_coupled_isotropy_irreps("Pm-3m", "Pnma", size=4):
        ...     print(match)
        R4+(0,a,a) M3+(d;0;0) -> Pnma (No. 62), size 4, index 24
    """
    try:
        number, _ = resolve_child(child)
        table = _isotropy_table(parent, use_cache=use_cache, progress=None)
        knames = None if kpoints is None else resolve_kpoints(
            table, [kpoints] if isinstance(kpoints, str) else kpoints
        )
    except SystemExit as exc:
        message = " ".join(str(exc).split())
        raise ValueError(message.removeprefix("ERROR: ") or "invalid input") from None
    return coupled_search(table, number, size=size, index=index, knames=knames).matches


# ---------------------------------------------------------------- report


def _setting_text(values) -> str:
    from .isotropy_subgroup import _format_setting_value

    if values is None:
        return "-"
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return "(" + ",".join(_format_setting_value(x) for x in array) + ")"
    return ",".join(_setting_text(row) for row in array)


def _origin_text(values) -> str:
    """The origin reduced into [0,1) modulo parent lattice translations.

    Integer conventional translations are lattice translations of every
    parent, so the reduced origin gives a G-conjugate member of the same
    stratum and makes rows comparable by eye.
    """
    if values is None:
        return "-"
    reduced = np.mod(np.round(np.asarray(values, dtype=float), 6), 1.0)
    reduced[np.isclose(reduced, 1.0, atol=1e-6)] = 0.0
    return _setting_text(reduced + 0.0)


def _home_relative(path: str) -> str:
    home = os.path.expanduser("~")
    if path == home or path.startswith(home + os.sep):
        return "~" + path[len(home):]
    return path


def _filter_text(size, index, knames) -> str:
    parts = []
    if size is not None:
        parts.append(f"size {size}")
    if index is not None:
        parts.append(f"index {index}")
    if knames:
        parts.append(("k point " if len(knames) == 1 else "k points ") + ", ".join(knames))
    return ", ".join(parts)


def _table_lines(header: list, cells: list) -> list:
    """Left-aligned columns separated by two spaces (header first)."""
    widths = [max(len(header[c]), *(len(row[c]) for row in cells))
              for c in range(len(header))]
    return ["  ".join(text.ljust(width) for text, width in zip(row, widths)).rstrip()
            for row in [header] + cells]


def _partner_note(number: int, symbol: str, partner_rows: list, listed: str) -> list:
    import textwrap

    from .isotropy_subgroup import ENANTIOMORPHIC_PAIRS

    partner = ENANTIOMORPHIC_PAIRS.get(number)
    partner_symbol = partner_rows[0].symbol
    return [line.replace("\u00a0", " ") for line in textwrap.wrap(
        f"note: {number} <-> {partner} ({symbol} <-> {partner_symbol}) are "
        "enantiomorphic partner types: an improper operation of the parent maps "
        "the order parameter of such a stratum to another one of the same "
        f"stratum whose stabilizer is the partner. The rows {listed} are listed with a "
        f"stabilizer of type {partner_symbol} (No.\u00a0{partner}).",
        width=78, break_on_hyphens=False,
    )]


def format_coupled_block(search: CoupledSearch, number: int, symbol: str,
                         filters: str = "") -> str:
    """The ``* Coupled isotropy subgroups of type H *`` block as text.

    Args:
        search: The :class:`CoupledSearch` of :func:`coupled_search`.
        number: Number of the subgroup type H.
        symbol: Its symbol.
        filters: The filter text (``"size 4"``), or ``""``.

    Returns:
        The block (no leading or trailing blank line).
    """
    n_pairs = search.n_pairs_same_k + search.n_pairs_cross_k
    lines = [
        f"* Coupled isotropy subgroups of type {symbol} *",
        f"{search.n_irreps} irreps pass the point-group filter; {n_pairs} pairs searched",
        f"({search.n_pairs_same_k} at the same k point, "
        f"{search.n_pairs_cross_k} at different k points)",
    ]
    if not search.matches:
        where = f" with {filters}" if filters else ""
        lines.append(f"no coupled stratum of type {symbol} (No. {number}){where}")
    else:
        header = ["irrep1", "dir1", "irrep2", "dir2", "size", "index",
                  "conventional basis", "origin"]
        cells = [
            [m.label1, m.direction1, m.label2, m.direction2, str(m.size), str(m.index),
             _setting_text(m.basis), _origin_text(m.origin)]
            for m in search.matches
        ]
        lines.extend(_table_lines(header, cells))
        lines.append("every row needs both irreps: neither irrep alone has this "
                     "isotropy subgroup")
        partner_rows = [m for m in search.matches if m.number != number]
        if partner_rows:
            listed = ", ".join(m.direction.replace(" ", "\u00a0") for m in partner_rows)
            lines.append("")
            lines.extend(_partner_note(number, symbol, partner_rows, listed))
    for pair, reason in search.errors.items():
        lines.append(f"note: {pair}: not analyzed ({reason})")
    return "\n".join(lines)


def format_child_report(table: IsotropyTable, number: int, symbol: str,
                        size=None, index=None, knames=None, coupled=None,
                        secondary=None) -> str:
    """The ``crystod-group --parent G --child H`` report as text.

    Args:
        table: The :class:`IsotropyTable` of the parent.
        number: Number of the subgroup type H.
        symbol: Its symbol.
        size: The ``--size`` filter, or ``None``.
        index: The ``--index`` filter, or ``None``.
        knames: The resolved ``--kpoint`` names, or ``None``.
        coupled: The :class:`CoupledSearch` of ``--coupled``, or ``None``.
        secondary: Text blocks of ``--secondary`` (one per coupled match),
            or ``None``.

    Returns:
        The report, ``* ... *`` blocks separated by blank lines.
    """
    numbers = child_numbers(number, not table.sohncke)
    rows = table.select(numbers, size=size, index=index, kpoints=knames)
    lines = ["", "* Supergroup *", f"{table.space_group} (No. {table.number})",
             "", "* Subgroup *", f"{symbol} (No. {number})", "",
             "* Isotropy table *"]
    knames_all = ", ".join(table.kpoints)
    lines.append(f"{len(table.irreps)} irreps at the special k points {knames_all}; "
                 f"{len(table.strata)} strata")
    if table.cache_path is not None:
        state = "read" if table.from_cache else "written"
        lines.append(f"cache: {_home_relative(table.cache_path)} ({state})")
    filters = _filter_text(size, index, knames)
    if filters:
        lines.append(f"selected: {filters}")
    lines.append("")
    lines.append(f"* Isotropy subgroups of type {symbol} *")
    if not rows:
        where = f" with {filters}" if filters else ""
        lines.append(f"no single-irrep stratum of type {symbol} (No. {number}){where};")
        if coupled is None:
            lines.append("the subgroup may need two coupled irreps (--coupled)")
        else:
            lines.append("the subgroup may need two coupled irreps (see below)")
    else:
        header = ["irrep", "k", "direction", "size", "index", "conventional basis", "origin"]
        cells = [
            [s.label, s.kname, s.direction, str(s.size), str(s.index),
             _setting_text(s.basis), _origin_text(s.origin)]
            for s in rows
        ]
        lines.extend(_table_lines(header, cells))
    partner_rows = [s for s in rows if s.number != number]
    if partner_rows:
        listed = ", ".join(f"{s.label}{s.direction}\u00a0at\u00a0{s.kname}"
                           for s in partner_rows)
        lines.append("")
        lines.extend(_partner_note(number, symbol, partner_rows, listed))
    if coupled is not None:
        lines.append("")
        lines.append(format_coupled_block(coupled, number, symbol, filters))
    for block in secondary or []:
        lines.append("")
        lines.append(block)
    if table.errors:
        lines.append("")
        for label, reason in table.errors.items():
            lines.append(f"note: {label}: not enumerated ({reason})")
    lines.append("")
    lines.append("Conventions and validation: ISOSUBGROUP (https://iso.byu.edu):")
    lines.append('H. T. Stokes, S. van Orden and B. J. Campbell, "Tool for Generating')
    lines.append('Isotropy Subgroups of Crystallographic Space Groups",')
    lines.append("J. Appl. Cryst. 49, 1849-1853 (2016).")
    return "\n".join(lines)


def _secondary_blocks(search: CoupledSearch, max_degree: int) -> list:
    """The ``--secondary`` block of every coupled match."""
    import warnings

    from .secondary_order_parameter import analyze_secondary, format_secondary

    blocks = []
    for match, data in zip(search.matches, search.data):
        header = (f"H = {match.symbol} ({match.number}), index {match.index}: "
                  f"{match.direction}")
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", UserWarning)
                result = analyze_secondary(
                    data["algebra"], data["representation"], data["members"],
                    data["B"], max_degree,
                )
            block = format_secondary(result, header, max_degree)
            for warning in caught:
                block += f"\nWARNING: {warning.message}"
        except (ValueError, RuntimeError) as exc:
            block = (f"* Secondary order parameters *\n{header}\n"
                     f"not analyzed ({exc})")
        blocks.append(block)
    return blocks


def main(argv: list[str] | None = None) -> None:
    """``crystod-group --parent G --child H`` (dispatched by cli/group.py)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Reverse lookup: the irreps whose isotropy subgroup has a given type."
    )
    parser.add_argument("--parent", required=True)
    parser.add_argument("--child", required=True)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--kpoint", nargs="+", default=None)
    parser.add_argument("--no-cache", dest="no_cache", action="store_true")
    parser.add_argument("--coupled", action="store_true")
    parser.add_argument("--secondary", action="store_true")
    parser.add_argument("--degree", type=int, default=None)
    if argv is None:
        argv = sys.argv[1:]
    args = parser.parse_args(list(argv))
    if args.secondary and not args.coupled:
        parser.error("--secondary with --child needs --coupled (the secondary order "
                     "parameters of every coupled answer).")
    if args.degree is not None and not args.secondary:
        parser.error("--degree is only used with --invariants or --secondary.")
    degree = 4 if args.degree is None else args.degree
    if args.secondary and not 2 <= degree <= 12:
        parser.error("--secondary needs --degree between 2 and 12.")
    number, symbol = resolve_child(args.child)
    parent_info = _resolve(args.parent)
    table = _isotropy_table(str(parent_info.number), use_cache=not args.no_cache)
    knames = resolve_kpoints(table, args.kpoint) if args.kpoint else None
    coupled = secondary = None
    if args.coupled:
        coupled = coupled_search(
            table, number, size=args.size, index=args.index, knames=knames,
            progress=_progress_to_stderr, keep_data=args.secondary,
        )
        if args.secondary:
            secondary = _secondary_blocks(coupled, degree)
    print(format_child_report(table, number, symbol, args.size, args.index, knames,
                              coupled=coupled, secondary=secondary))
