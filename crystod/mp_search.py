"""Materials Project search and POSCAR download (``crystod-search``).

A query is read the way the search box of the Materials Project website
reads it:

- ``SrTiO3`` -- a formula: every polymorph of that composition;
- ``ABO3`` -- an anonymous formula (every letter stands for any element);
- ``Sr-Ti-O`` -- a chemical system: the compounds of exactly these elements
  (``Sr-*`` adds a wildcard element);
- ``Sr,Ti,O`` -- elements: every material containing at least these;
- ``mp-5229`` -- material IDs (comma-separated for several).

The matches are listed like the website's table (formula, space group,
material ID, band gap, energy above the hull, number of sites), a star after
the ID marking the experimentally observed materials, and the structures
are downloaded as POSCAR files -- by default the standardized conventional
cell (``POSCAR_...``), the standardized primitive cell on request
(``PPOSCAR_...``), both found with the tolerances the Materials Project
itself uses for its symmetry (``symprec = 0.1``, 5 degrees), so the file
carries the space group of the table.

The client talks to the REST API of the Materials Project
(``https://api.materialsproject.org``, the ``materials/summary`` endpoint)
with ``requests``, the same requests pymatgen's own basic ``MPRester``
sends; pymatgen turns the downloaded structures into ``Structure`` objects,
standardizes them and writes the POSCAR files.  The API key is read from
the ``MP_API_KEY`` environment variable or from ``PMG_MAPI_KEY`` in the
pymatgen settings file (``~/.config/.pmgrc.yaml`` or ``~/.pmgrc.yaml``);
``MP_API_ENDPOINT`` points the client at another server.

Material IDs are shown the way the website shows them.  The API returns
every ID in the alphabetical form of the current database (``mp-aaaaahtd``
is ``mp-5229`` written in base 26); IDs up to ``mp-3347529`` are converted
back to the familiar numbers, later ones stay alphabetical (``mp-aaaieiuj``),
as on the website.  Both spellings are accepted as input.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field, replace
from itertools import combinations
from typing import Any

#: The public REST API of the Materials Project.
DEFAULT_ENDPOINT = "https://api.materialsproject.org"
#: Page of one material on the Materials Project website.
MATERIAL_URL = "https://next-gen.materialsproject.org/materials/{material_id}"
#: Where a (free) API key is issued.
API_KEY_URL = "https://next-gen.materialsproject.org/api"

#: Largest integer ID still written in the legacy numeric form (``mp-5229``);
#: larger ones are shown alphabetically (``mp-aaaieiuj``).  The cut point of
#: ``emmet.core.mpid.AlphaID``, which the website follows.
LEGACY_ID_CUTOFF = 3347529
#: Length of the alphabetical IDs of the API (``mp-aaaaahtd``).
ALPHA_ID_LENGTH = 8

#: Sort orders of ``--sort``: name -> (field the server sorts on, description).
#: The server cannot sort on the space-group number (it silently ignores a
#: nested field such as ``symmetry.number``), so ``spg`` sorts locally.  The
#: default, the energy above the hull, is the order of the website's table.
SORT_KEYS: dict[str, tuple[str | None, str]] = {
    "ehull": ("energy_above_hull", "energy above hull"),
    "gap": ("band_gap", "band gap"),
    "sites": ("nsites", "number of sites"),
    "id": ("material_id", "material ID"),
    "formula": ("formula_pretty", "formula"),
    "spg": (None, "space-group number"),
}
DEFAULT_SORT = "ehull"

#: Cells ``--cell`` can write: the standardized conventional cell (default)
#: or the standardized primitive cell.
CELL_CHOICES: tuple[str, ...] = ("conventional", "primitive")
DEFAULT_CELL = "conventional"
#: File-name prefix of each cell: ``POSCAR_SrTiO3_Pm-3m_mp-5229`` holds the
#: conventional cell, ``PPOSCAR_...`` the primitive one (the P of phonopy's
#: PPOSCAR and of CrystOD's ``221_PPOSCAR_*`` examples); with
#: ``--directory`` the prefix is the file name.
POSCAR_PREFIXES: dict[str, str] = {
    "conventional": "POSCAR",
    "primitive": "PPOSCAR",
}
#: Symmetry tolerance (Angstrom) of the standardization; the Materials
#: Project determines its space groups with the same value.
DEFAULT_TOLERANCE = 0.1
#: Angle tolerance (degrees) of the standardization, the Materials Project's.
ANGLE_TOLERANCE = 5.0

#: Default cap on the number of listed materials (one page of the API).
DEFAULT_MAX_RESULTS = 1000
#: Most matches fetched to list more than one page, or to sort a list longer
#: than one page by space group (the pages are fetched in ID order and sorted
#: locally, since pages sorted by a key with ties can overlap on the server).
MAX_FULL_FETCH = 20000
_PAGE_SIZE = 1000  # the largest _limit the summary endpoint accepts
_ID_CHUNK = 100  # material IDs per structure request (keeps the URL short)
_TIMEOUT = 60  # seconds per request

_TABLE_FIELDS = (
    "material_id", "formula_pretty", "symmetry", "band_gap",
    "energy_above_hull", "nsites", "theoretical",
)
_STRUCTURE_FIELDS = (*_TABLE_FIELDS, "structure", "deprecated")

MP_CITATION = (
    "Data: the Materials Project, A. Jain et al., APL Mater. 1, 011002 (2013). "
    "https://doi.org/10.1063/1.4812323"
)

ELEMENTS = frozenset("""
H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu
Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba
La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb
Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs
Mt Ds Rg Cn Nh Fl Mc Lv Ts Og
""".split())

_ID_RE = re.compile(r"^mp-([0-9]+|[a-z]+)$")
# IDs with another prefix (mvc-123, ...): the summary endpoint drops the prefix
# and would answer with the mp- material of the same number
_OTHER_ID_RE = re.compile(r"^[a-z]+-[0-9]+$")


class MaterialsProjectError(RuntimeError):
    """The Materials Project could not answer: no or bad API key, no
    network, or an error reported by the server."""


# ------------------------------------------------------------------ IDs
def _alpha_to_int(letters: str) -> int:
    value = 0
    for letter in letters:
        value = value * 26 + (ord(letter) - ord("a"))
    return value


def _int_to_alpha(value: int) -> str:
    letters = ""
    while value:
        value, digit = divmod(value, 26)
        letters = chr(ord("a") + digit) + letters
    return letters.rjust(ALPHA_ID_LENGTH, "a")


def normalize_material_id(text: str) -> str:
    """The website spelling of a Materials Project ID.

    ``mp-aaaaahtd``, ``MP-5229`` and ``mp-5229`` all give ``mp-5229``; an ID
    past :data:`LEGACY_ID_CUTOFF` is given in the 8-letter alphabetical form
    (``mp-3732049`` gives ``mp-aaaieiuj``).

    Raises:
        SystemExit: ``text`` is not a Materials Project ID (``ValueError``
            when called through :mod:`crystod.search`).
    """
    match = _ID_RE.match(str(text).strip().lower())
    if not match:
        raise SystemExit(
            f"ERROR: '{text}' is not a Materials Project ID "
            "(mp-5229, mp-aaaieiuj, ...)."
        )
    (body,) = match.groups()
    value = int(body) if body.isdigit() else _alpha_to_int(body)
    if value <= LEGACY_ID_CUTOFF:
        return f"mp-{value}"
    return f"mp-{_int_to_alpha(value)}"


def _id_number(material_id: str) -> int:
    """The integer behind an ID (``mp-5229`` and ``mp-aaaaahtd`` give 5229)."""
    body = material_id.split("-", 1)[1]
    return int(body) if body.isdigit() else _alpha_to_int(body)


# ------------------------------------------------------------------ records
@dataclass(frozen=True)
class Material:
    """One material of the Materials Project, as a row of the search table.

    Attributes:
        material_id: The ID as the website writes it (``mp-5229``).
        formula: The reduced formula (``SrTiO3``).
        space_group: The Hermann-Mauguin symbol (``Pm-3m``, ``P6_3/mmc``).
        space_group_number: 1-230.
        crystal_system: ``Cubic``, ``Tetragonal``, ...
        band_gap: The computed band gap in eV (``None`` when not computed).
        energy_above_hull: The energy above the convex hull in eV/atom.
        nsites: Number of sites of the stored (computed) cell.
        experimental: True when the structure matches an experimentally
            observed one (an ICSD or other experimental entry); these are
            starred in the table.
        structure: The downloaded pymatgen ``Structure`` -- set by
            :func:`fetch_materials` (in the cell it was asked for), ``None``
            in search results.
        cell: The cell ``structure`` is in (see :data:`CELL_CHOICES`).
        deprecated: The Materials Project has deprecated this entry.
        cell_space_group_number: The space group spglib finds for
            ``structure`` -- set by :func:`fetch_materials`; it differs from
            ``space_group_number`` only when the standardization at the
            chosen tolerance lands on another group.
    """

    material_id: str
    formula: str
    space_group: str
    space_group_number: int
    crystal_system: str
    band_gap: float | None
    energy_above_hull: float | None
    nsites: int
    experimental: bool
    structure: Any = field(default=None, compare=False, repr=False)
    cell: str | None = None
    deprecated: bool = False
    cell_space_group_number: int | None = None

    @property
    def url(self) -> str:
        """The page of the material on the Materials Project website."""
        return MATERIAL_URL.format(material_id=self.material_id)

    @property
    def label(self) -> str:
        """The ID with the star of an experimentally observed material."""
        return f"{self.material_id} *" if self.experimental else self.material_id

    @classmethod
    def from_document(cls, doc: dict) -> Material:
        """A record from one document of the ``materials/summary`` endpoint."""
        symmetry = doc.get("symmetry") or {}
        return cls(
            material_id=normalize_material_id(doc["material_id"]),
            formula=doc.get("formula_pretty") or "?",
            space_group=symmetry.get("symbol") or "?",
            space_group_number=int(symmetry.get("number") or 0),
            crystal_system=symmetry.get("crystal_system") or "?",
            band_gap=doc.get("band_gap"),
            energy_above_hull=doc.get("energy_above_hull"),
            nsites=int(doc.get("nsites") or 0),
            experimental=doc.get("theoretical") is False,
            deprecated=bool(doc.get("deprecated")),
        )


TABLE_HEADER = (
    "Formula", "Space group", "Material ID",
    "Band Gap (eV)", "Energy Above Hull (eV/atom)", "Sites",
)


def format_table(materials: list[Material]) -> list[str]:
    """The search table as text lines, one header line and one line per material.

    Formula, space group and ID are left-aligned, the numbers right-aligned;
    an experimentally observed material has ``" *"`` (a space and a star)
    after its ID.
    """
    def number(value: float | None) -> str:
        return "-" if value is None else f"{value:.3f}"

    rows = [
        (m.formula, m.space_group, m.label, number(m.band_gap),
         number(m.energy_above_hull), str(m.nsites))
        for m in materials
    ]
    widths = [max([len(title)] + [len(row[i]) for row in rows])
              for i, title in enumerate(TABLE_HEADER)]

    def line(cells) -> str:
        parts = [cell.ljust(width) if i < 3 else cell.rjust(width)
                 for i, (cell, width) in enumerate(zip(cells, widths))]
        return "  ".join(parts).rstrip()

    return [line(TABLE_HEADER), *(line(row) for row in rows)]


@dataclass
class SearchResult:
    """The answer to one search.

    Attributes:
        query: The query as given.
        description: How it was read (``chemical system Sr-Ti-O``, ...).
        materials: The matches, in the order of ``sort``.
        total: How many materials match; more than ``len(materials)`` when
            the list was cut at ``max_results``.
        sort: The key of :data:`SORT_KEYS` the list is sorted by.
        filters: One phrase per filter that was applied.
    """

    query: str
    description: str
    materials: list[Material]
    total: int
    sort: str = DEFAULT_SORT
    filters: tuple[str, ...] = ()

    @property
    def truncated(self) -> bool:
        """True when more materials match than were listed."""
        return self.total > len(self.materials)

    def table_lines(self) -> list[str]:
        """The table of :func:`format_table` for these materials."""
        return format_table(self.materials)


# ------------------------------------------------------------------ queries
def _split(text: str, separator: str) -> list[str]:
    return [token.strip() for token in text.split(separator)]


def _check_elements(symbols: list[str], query: str, wildcard: bool = False) -> None:
    bad = [s for s in symbols if s not in ELEMENTS and not (wildcard and s == "*")]
    if bad:
        raise SystemExit(
            f"ERROR: {', '.join(repr(s) for s in bad)} in '{query}' "
            f"{'is not an element symbol' if len(bad) == 1 else 'are not element symbols'} "
            "(symbols start with a capital letter: Sr, Ti, O)."
        )
    repeated = sorted({s for s in symbols if s != "*" and symbols.count(s) > 1})
    if repeated:
        raise SystemExit(f"ERROR: '{query}' names {', '.join(repeated)} more than once.")


def _read_formula(formula: str, query: str) -> str:
    """``"formula"`` or ``"anonymous"``; exits on an unreadable formula."""
    if "." in formula:
        raise SystemExit(
            f"ERROR: '{formula}' contains '.': the Materials Project matches integer "
            "formulas only; write a hydrate as one formula (CuSO9H10, not CuSO4.5H2O)."
        )
    if not formula or not re.fullmatch(r"[A-Za-z0-9()*]+", formula):
        raise SystemExit(f"ERROR: cannot read '{formula}' in '{query}' as a formula.")
    leftover = re.sub(r"[A-Z][a-z]*", "", formula)
    if re.search(r"[a-z]", leftover):
        raise SystemExit(
            f"ERROR: cannot read '{formula}' as a formula: element symbols start "
            "with a capital letter (SrTiO3, not srtio3)."
        )
    symbols = re.findall(r"[A-Z][a-z]*", formula)
    if not symbols:
        raise SystemExit(f"ERROR: '{formula}' names no element.")
    if all(s in ELEMENTS for s in symbols):
        return "formula"
    if all(len(s) == 1 for s in symbols):
        return "anonymous"  # ABO3: every letter is a placeholder, as on the website
    bad = [s for s in symbols if s not in ELEMENTS]
    raise SystemExit(
        f"ERROR: {', '.join(repr(s) for s in bad)} in '{query}' "
        f"{'is not an element symbol' if len(bad) == 1 else 'are not element symbols'} "
        "(symbols start with a capital letter: Sr, Ti, O)."
    )


def _subsystems(elements: list[str]) -> str:
    """``Sr-Ti-O`` -> every chemical system made of a non-empty subset."""
    unique = sorted(set(elements))
    return ",".join("-".join(combo) for size in range(1, len(unique) + 1)
                    for combo in combinations(unique, size))


def interpret_query(query: str, subsystems: bool = False) -> tuple[str, dict[str, str]]:
    """Read a query the way the Materials Project website does.

    Returns:
        ``(description, criteria)``: a phrase for the header of the table
        and the filter parameters of the ``materials/summary`` endpoint.

    Raises:
        SystemExit: The query cannot be read, or ``subsystems`` is set
            for a query that is not a single chemical system (``ValueError``
            through :mod:`crystod.search`).
    """
    query = query.strip()
    if not query:
        raise SystemExit("ERROR: empty query.")
    tokens = [token for token in _split(query, ",") if token]
    if not tokens:
        raise SystemExit(f"ERROR: cannot read '{query}' as a query.")
    not_single_system = "ERROR: --subsystems needs a chemical system such as Sr-Ti-O."

    if all(_ID_RE.match(t.lower()) for t in tokens):
        ids = list(dict.fromkeys(normalize_material_id(t) for t in tokens))
        if subsystems:
            raise SystemExit(not_single_system)
        return f"material ID{'s' if len(ids) > 1 else ''} {', '.join(ids)}", {
            "material_ids": ",".join(ids)}
    other = next((t for t in tokens if _OTHER_ID_RE.match(t.lower())), None)
    if other:
        raise SystemExit(
            f"ERROR: '{other}' is not served by the current Materials Project API, "
            "which knows mp- IDs only (mp-5229, mp-aaaieiuj)."
        )

    if any("-" in t for t in tokens):
        systems = [_split(t, "-") for t in tokens]
        for system in systems:
            _check_elements(system, query, wildcard=True)
        cleaned = ",".join("-".join(system) for system in systems)
        if subsystems:
            if len(systems) > 1 or "*" in systems[0]:
                raise SystemExit(
                    "ERROR: --subsystems needs a single chemical system without "
                    "a wildcard, such as Sr-Ti-O."
                )
            return (f"chemical system {cleaned} and all its subsystems",
                    {"chemsys": _subsystems(systems[0])})
        scope = ("each * stands for one more element" if "*" in cleaned
                 else "compounds of exactly these elements")
        return (f"chemical system{'s' if len(systems) > 1 else ''} {cleaned} "
                f"({scope})"), {"chemsys": cleaned}

    if subsystems:
        raise SystemExit(not_single_system)

    if len(tokens) > 1 and all(t in ELEMENTS for t in tokens):
        _check_elements(tokens, query)
        names = ", ".join(tokens[:-1]) + " and " + tokens[-1]
        return f"materials containing {names}", {"elements": ",".join(tokens)}

    kinds = {_read_formula(t, query) for t in tokens}
    cleaned = ",".join(tokens)
    if kinds == {"anonymous"}:
        return (f"anonymous formula{'s' if len(tokens) > 1 else ''} {cleaned} "
                "(each letter stands for any element)"), {"formula": cleaned}
    if "anonymous" in kinds:
        raise SystemExit("ERROR: do not mix anonymous formulas and formulas in one query.")
    return f"formula{'s' if len(tokens) > 1 else ''} {cleaned}", {"formula": cleaned}


def resolve_space_group(text: str) -> tuple[int, str]:
    """``(number, symbol)`` of a space group given by number or short symbol.

    The symbol is the Hermann-Mauguin short symbol as spglib and the
    Materials Project write it (``P6_3/mmc``); the underscore of a screw
    axis may be left out (``P63/mmc``).

    Raises:
        SystemExit: Not a space group (``ValueError`` through the API).
    """
    import spglib

    table: dict[str, tuple[int, str]] = {}
    for hall in range(1, 531):
        entry = spglib.get_spacegroup_type(hall)
        short = getattr(entry, "international_short", None) or entry["international_short"]
        number = getattr(entry, "number", None) or entry["number"]
        table.setdefault(short, (number, short))
    text = str(text).strip()
    if text.isdigit():
        number = int(text)
        if not 1 <= number <= 230:
            raise SystemExit(f"ERROR: space-group number {number} is not in 1-230.")
        symbol = next(symbol for n, symbol in table.values() if n == number)
        return number, symbol
    if text in table:
        return table[text]
    for key in (lambda s: s.replace("_", ""), lambda s: s.replace("_", "").lower()):
        matches = {value for symbol, value in table.items() if key(symbol) == key(text)}
        if len(matches) == 1:
            return matches.pop()
    raise SystemExit(
        f"ERROR: unknown space group '{text}' (give the number, or the short "
        "symbol such as Pm-3m or P6_3/mmc)."
    )


def _finite(value, minimum: float = 0.0) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value >= minimum


def build_criteria(
    *,
    experimental: bool = False,
    stable: bool = False,
    max_energy_above_hull: float | None = None,
    band_gap: tuple[float, float] | None = None,
    nsites: tuple[int, int] | None = None,
    space_group: str | int | None = None,
    exclude_elements: list[str] | tuple[str, ...] | None = None,
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Filter parameters of the summary endpoint and a phrase for each.

    The arguments are those of :func:`search_materials`.

    Raises:
        SystemExit: A bound that is negative, not finite or reversed, an
            unknown space group or element (``ValueError`` through the API).
    """
    criteria: dict[str, str] = {}
    phrases: list[str] = []
    if experimental:
        criteria["theoretical"] = "false"
        phrases.append("experimentally observed only")
    if stable:
        criteria["is_stable"] = "true"
        phrases.append("on the convex hull only")
    if max_energy_above_hull is not None:
        if not _finite(max_energy_above_hull):
            raise SystemExit("ERROR: the energy above hull bound (--ehull) must be a number >= 0.")
        criteria["energy_above_hull_max"] = f"{max_energy_above_hull:g}"
        phrases.append(f"energy above hull <= {max_energy_above_hull:g} eV/atom")
    if band_gap is not None:
        low, high = band_gap
        if not (_finite(low) and _finite(high) and low <= high):
            raise SystemExit("ERROR: the band-gap range (--band-gap MIN MAX) needs 0 <= MIN <= MAX.")
        criteria["band_gap_min"], criteria["band_gap_max"] = f"{low:g}", f"{high:g}"
        phrases.append(f"band gap {low:g}-{high:g} eV")
    if nsites is not None:
        low, high = nsites
        if not (_finite(low, 1) and _finite(high, 1) and low <= high):
            raise SystemExit("ERROR: the site range (--sites MIN MAX) needs 1 <= MIN <= MAX.")
        criteria["nsites_min"], criteria["nsites_max"] = str(int(low)), str(int(high))
        phrases.append(f"{int(low)}-{int(high)} sites")
    if space_group is not None:
        number, symbol = resolve_space_group(str(space_group))
        criteria["spacegroup_number"] = str(number)
        phrases.append(f"space group {symbol} (No. {number})")
    if exclude_elements is not None:
        if isinstance(exclude_elements, str):
            exclude_elements = [exclude_elements]
        symbols = [s for item in exclude_elements for s in _split(item, ",") if s]
        if not symbols:
            raise SystemExit("ERROR: the excluded elements (--exclude) name no element.")
        _check_elements(symbols, "--exclude")
        criteria["exclude_elements"] = ",".join(symbols)
        phrases.append(f"without {', '.join(symbols)}")
    return criteria, tuple(phrases)


# ------------------------------------------------------------------ HTTP
def find_api_key(explicit: str | None = None) -> str:
    """The Materials Project API key: ``explicit``, else ``MP_API_KEY``,
    else ``PMG_MAPI_KEY`` (environment or pymatgen settings file).

    Raises:
        MaterialsProjectError: No key, or a key of the retired legacy API
            (16 characters; the current API issues 32-character keys).
    """
    source = "the api_key argument"
    key = explicit
    if not key:
        key, source = os.environ.get("MP_API_KEY"), "MP_API_KEY"
    if not key:
        key, source = os.environ.get("PMG_MAPI_KEY"), "PMG_MAPI_KEY"
    if not key:
        from pymatgen.core import SETTINGS

        key, source = SETTINGS.get("PMG_MAPI_KEY"), "PMG_MAPI_KEY of the pymatgen settings file"
    if not key:
        raise MaterialsProjectError(
            "no Materials Project API key found. Get one (free) at "
            f"{API_KEY_URL} and either export MP_API_KEY=<your key> or run "
            "'pmg config --add PMG_MAPI_KEY <your key>'."
        )
    key = str(key).strip()
    if len(key) != 32:
        raise MaterialsProjectError(
            f"the API key in {source} has {len(key)} characters; the current "
            "Materials Project API needs the 32-character key shown at "
            f"{API_KEY_URL} (16-character keys belong to the retired legacy API)."
        )
    return key


def _endpoint() -> str:
    endpoint = os.environ.get("MP_API_ENDPOINT")
    if not endpoint:
        from pymatgen.core import SETTINGS

        endpoint = SETTINGS.get("PMG_MAPI_ENDPOINT") or DEFAULT_ENDPOINT
    return str(endpoint).rstrip("/")


class _Client:
    """A ``requests`` session on the ``materials/summary`` endpoint."""

    def __init__(self, key: str | None = None) -> None:
        import platform
        import sys

        import requests

        from . import __version__

        self._requests = requests
        self.endpoint = _endpoint()
        self.session = requests.Session()
        self.session.headers["x-api-key"] = find_api_key(key)
        self.session.headers["user-agent"] = (
            f"CrystOD/{__version__} (Python/{sys.version.split()[0]} "
            f"{platform.system()}/{platform.release()})"
        )

    def __enter__(self) -> _Client:
        return self

    def __exit__(self, *exc_info) -> None:
        self.session.close()

    def summary(self, params: dict[str, str]) -> tuple[list[dict], int]:
        """One page of summary documents and the total number of matches."""
        url = f"{self.endpoint}/materials/summary/"
        try:
            response = self.session.get(url, params=params, timeout=_TIMEOUT)
        except self._requests.RequestException as exc:
            raise MaterialsProjectError(
                f"could not reach the Materials Project API at {self.endpoint} "
                f"({type(exc).__name__}). Check the network connection."
            ) from None
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        if response.status_code == 401:
            reason = payload.get("message") or "invalid API key"
            raise MaterialsProjectError(
                f"the Materials Project rejected the API key ({reason}). The key "
                f"is shown at {API_KEY_URL}."
            )
        if response.status_code == 429:
            raise MaterialsProjectError(
                "the Materials Project API is rate-limiting this client; wait a "
                "minute and try again."
            )
        if response.status_code != 200 or "data" not in payload:
            detail = payload.get("detail") or payload.get("message") or response.text[:200]
            if isinstance(detail, list):  # FastAPI validation errors
                detail = "; ".join(str(item.get("msg", item)) if isinstance(item, dict)
                                   else str(item) for item in detail)
            raise MaterialsProjectError(
                f"the Materials Project API answered HTTP {response.status_code}: {detail}"
            )
        data = payload["data"]
        if not isinstance(data, list) or not all(isinstance(doc, dict) for doc in data):
            raise MaterialsProjectError(
                "unexpected answer from the Materials Project API (no list of documents)."
            )
        try:
            total = int((payload.get("meta") or {}).get("total_doc"))
        except (TypeError, ValueError):
            total = len(data)
        return data, total

    def page(self, criteria: dict[str, str], fields, *, sort: str | None = None,
             limit: int = _PAGE_SIZE, skip: int = 0) -> tuple[list[dict], int]:
        """One page of at most ``limit`` (<= 1000) documents."""
        params = {**criteria, "_fields": ",".join(fields),
                  "_limit": str(limit), "_skip": str(skip)}
        if sort:
            params["_sort_fields"] = sort
        return self.summary(params)

    def everything(self, criteria: dict[str, str], fields, total: int) -> list[dict]:
        """Every matching document, paged in material-ID order.

        The ID is unique, so the pages neither overlap nor leave gaps; a sort
        on a key with ties (energy above hull, band gap, ...) would not
        guarantee that, because the server may order equal keys differently
        from one page to the next.
        """
        documents: dict[str, dict] = {}
        skip = 0
        while skip < total:
            page, total = self.page(criteria, fields, sort="material_id", skip=skip)
            if not page:
                break
            for doc in page:
                documents.setdefault(str(doc.get("material_id")), doc)
            skip += len(page)
        return list(documents.values())


def _sort_key(sort: str):
    """Key function ordering materials by ``sort``, missing values last.

    Python's sort is stable, so ties keep the order the server returned
    them in (the website's order for a list of one page; ID order for a
    list fetched in several pages).
    """
    attribute = {"ehull": "energy_above_hull", "gap": "band_gap", "sites": "nsites",
                 "formula": "formula", "spg": "space_group_number"}.get(sort)

    def key(material: Material):
        if attribute is None:  # "id"
            return (False, _id_number(material.material_id))
        value = getattr(material, attribute)
        return (value is None, 0 if value is None else value)

    return key


def _materials(documents: list[dict]) -> list[Material]:
    try:
        return [Material.from_document(doc) for doc in documents]
    except (KeyError, TypeError, ValueError, AttributeError, SystemExit) as exc:
        raise MaterialsProjectError(
            f"unexpected document from the Materials Project API ({type(exc).__name__}: {exc})"
        ) from None


# ------------------------------------------------------------------ search
def search_materials(
    query: str,
    *,
    experimental: bool = False,
    stable: bool = False,
    max_energy_above_hull: float | None = None,
    band_gap: tuple[float, float] | None = None,
    nsites: tuple[int, int] | None = None,
    space_group: str | int | None = None,
    exclude_elements: list[str] | tuple[str, ...] | None = None,
    subsystems: bool = False,
    sort: str = DEFAULT_SORT,
    max_results: int = DEFAULT_MAX_RESULTS,
    api_key: str | None = None,
) -> SearchResult:
    """Search the Materials Project the way its website does.

    Args:
        query: ``SrTiO3`` (formula), ``ABO3`` (anonymous formula),
            ``Sr-Ti-O`` (chemical system), ``Sr,Ti,O`` (elements) or
            ``mp-5229`` (IDs); see :func:`interpret_query`.
        experimental: Only experimentally observed materials.
        stable: Only materials on the convex hull.
        max_energy_above_hull: Upper bound in eV/atom.
        band_gap: ``(min, max)`` in eV.
        nsites: ``(min, max)`` number of sites.
        space_group: Number or short symbol.
        exclude_elements: Element symbols the materials must not contain.
        subsystems: With a chemical system, also every subsystem
            (``Sr-Ti-O`` adds Sr, Ti, O, Sr-Ti, Sr-O and Ti-O).
        sort: A key of :data:`SORT_KEYS` (default: energy above hull).
        max_results: Cap on the number of materials listed.  Up to 1000 of
            a longer list cost one request; more (or a list longer than 1000
            sorted by space group, which the server cannot sort on) need
            every match fetched, which is refused beyond
            :data:`MAX_FULL_FETCH` matches.
        api_key: The API key; by default read as :func:`find_api_key` describes.

    Raises:
        SystemExit: Unreadable query or filter, or a list too long to fetch
            (``ValueError`` through :mod:`crystod.search`).
        MaterialsProjectError: No API key, no network, or a server error.
    """
    if sort not in SORT_KEYS:
        raise SystemExit(f"ERROR: unknown sort key '{sort}' (choose from {', '.join(SORT_KEYS)}).")
    if not isinstance(max_results, int) or max_results < 1:
        raise SystemExit("ERROR: the number of listed materials (--max) must be at least 1.")
    description, criteria = interpret_query(query, subsystems=subsystems)
    filters, phrases = build_criteria(
        experimental=experimental, stable=stable,
        max_energy_above_hull=max_energy_above_hull, band_gap=band_gap,
        nsites=nsites, space_group=space_group, exclude_elements=exclude_elements,
    )
    criteria = {**criteria, **filters, "deprecated": "false"}
    server_sort = SORT_KEYS[sort][0]
    with _Client(api_key) as client:
        # one page, sorted by the server when it can: when every match fits,
        # or when the first page already holds the materials to list
        documents, total = client.page(criteria, _TABLE_FIELDS,
                                       sort=server_sort or "material_id")
        complete = total <= len(documents)
        if not complete and not (server_sort and max_results <= len(documents)):
            if total > MAX_FULL_FETCH:
                reason = ("to sort them by space-group number" if not server_sort
                          else f"to list more than {len(documents)} of them")
                raise SystemExit(
                    f"ERROR: {total} materials match, too many to fetch all of them "
                    f"{reason} (the limit is {MAX_FULL_FETCH}); narrow the query "
                    "with filters, or list at most 1000 with another --sort."
                )
            documents = client.everything(criteria, _TABLE_FIELDS, total)
    materials = sorted(_materials(documents), key=_sort_key(sort))
    return SearchResult(
        query=query,
        description=description,
        materials=materials[:max_results],
        total=max(total, len(documents)),
        sort=sort,
        filters=phrases,
    )


# ------------------------------------------------------------------ structures
def _check_tolerance(tolerance) -> None:
    # spglib crashes the interpreter (segmentation fault) on a negative or NaN
    # symprec, so the value is checked before it gets there
    if not (isinstance(tolerance, (int, float)) and math.isfinite(tolerance) and tolerance > 0):
        raise SystemExit(
            f"ERROR: the symmetry tolerance must be a positive number of Angstrom (got {tolerance})."
        )


def standardize_cell(structure, cell: str = DEFAULT_CELL,
                     tolerance: float = DEFAULT_TOLERANCE):
    """``structure`` in the requested cell.

    ``"primitive"`` and ``"conventional"`` are the standardized cells of
    spglib (``standardize_cell`` with and without ``to_primitive``, atoms
    moved onto their ideal positions), the setting CrystOD's own analyses
    standardize to.  The angle tolerance is 5 degrees, the value the Materials Project uses with its
    ``symprec = 0.1``.

    Raises:
        SystemExit: Unknown cell, a tolerance that is not a positive number,
            or a cell spglib cannot standardize (``ValueError`` through
            :mod:`crystod.search`).
    """
    if cell not in CELL_CHOICES:
        raise SystemExit(f"ERROR: unknown cell '{cell}' (choose from {', '.join(CELL_CHOICES)}).")
    _check_tolerance(tolerance)
    import spglib
    from pymatgen.core import Structure

    kinds = list(dict.fromkeys(site.species for site in structure))
    numbers = [kinds.index(site.species) + 1 for site in structure]
    standardized = spglib.standardize_cell(
        (structure.lattice.matrix, structure.frac_coords, numbers),
        to_primitive=(cell == "primitive"), no_idealize=False, symprec=tolerance,
        angle_tolerance=ANGLE_TOLERANCE,
    )
    if standardized is None:
        raise SystemExit(f"ERROR: spglib could not standardize the cell at tolerance {tolerance:g}.")
    lattice, positions, numbers = standardized
    return Structure(lattice, [kinds[n - 1] for n in numbers], positions)


def _space_group_number(structure, symprec: float) -> int | None:
    """The space-group number spglib finds for ``structure`` (None if none)."""
    import spglib

    kinds = list(dict.fromkeys(site.species for site in structure))
    numbers = [kinds.index(site.species) + 1 for site in structure]
    dataset = spglib.get_symmetry_dataset(
        (structure.lattice.matrix, structure.frac_coords, numbers),
        symprec=symprec, angle_tolerance=ANGLE_TOLERANCE,
    )
    if dataset is None:
        return None
    return getattr(dataset, "number", None) or dataset["number"]


def _tidy(structure):
    """The structure with lattice and coordinates rounded to 1e-12 (no -0.0)."""
    import numpy as np
    from pymatgen.core import Lattice, Structure

    matrix = np.round(structure.lattice.matrix, 12) + 0.0
    coords = np.round(structure.frac_coords, 12) + 0.0
    coords[coords == 1.0] = 0.0
    return Structure(Lattice(matrix), structure.species, coords)


def fetch_materials(
    material_ids: str | list[str] | tuple[str, ...],
    *,
    cell: str = DEFAULT_CELL,
    tolerance: float = DEFAULT_TOLERANCE,
    api_key: str | None = None,
) -> list[Material]:
    """Download the structures of the given materials.

    Args:
        material_ids: One ID or a list (``mp-5229``, ``mp-aaaaahtd``, ...).
        cell: ``"conventional"`` (default) or ``"primitive"`` (see
            :func:`standardize_cell`).
        tolerance: Symmetry tolerance of the standardization in Angstrom.
        api_key: The API key; by default read as :func:`find_api_key` describes.

    Returns:
        One :class:`Material` per distinct material, in the order first
        given (a repeated ID, or the two spellings ``mp-5229`` and
        ``mp-aaaaahtd`` of one ID, give one entry), with ``structure`` and
        ``cell_space_group_number`` set.

    Raises:
        SystemExit: A malformed ID, an ID the Materials Project does not
            have, or a tolerance that is not a positive number
            (``ValueError`` through :mod:`crystod.search`).
        MaterialsProjectError: No API key, no network, or a server error.
    """
    if isinstance(material_ids, str):
        material_ids = [material_ids]
    material_ids = [str(i).strip() for i in material_ids]
    spelled = {normalize_material_id(i): i for i in reversed(material_ids)}
    wanted = list(dict.fromkeys(normalize_material_id(i) for i in material_ids))
    if not wanted:
        raise SystemExit("ERROR: no material ID given.")
    if cell not in CELL_CHOICES:
        raise SystemExit(f"ERROR: unknown cell '{cell}' (choose from {', '.join(CELL_CHOICES)}).")
    _check_tolerance(tolerance)
    documents: dict[str, dict] = {}
    with _Client(api_key) as client:
        for start in range(0, len(wanted), _ID_CHUNK):
            chunk = wanted[start:start + _ID_CHUNK]
            page, _ = client.page({"material_ids": ",".join(chunk)}, _STRUCTURE_FIELDS,
                                  limit=len(chunk))
            for doc in page:
                try:
                    documents[normalize_material_id(doc["material_id"])] = doc
                except (KeyError, TypeError, SystemExit):
                    raise MaterialsProjectError(
                        "unexpected document from the Materials Project API (no material ID)."
                    ) from None
    missing = [spelled[i] for i in wanted if i not in documents]
    if missing:
        raise SystemExit(
            f"ERROR: the Materials Project has no material {', '.join(missing)} "
            "(check the ID on the website; deprecated entries are not served)."
        )

    from pymatgen.core import Structure

    materials = []
    for material_id in wanted:
        doc = documents[material_id]
        (record,) = _materials([doc])
        try:
            stored = Structure.from_dict(doc["structure"])
        except Exception as exc:  # anything pymatgen raises on a malformed dict
            raise MaterialsProjectError(
                f"the Materials Project sent no readable structure for {material_id} "
                f"({type(exc).__name__})."
            ) from None
        structure = standardize_cell(stored, cell, tolerance)
        # the idealized standardized cell has its symmetry exactly
        found = _space_group_number(structure, 1e-3)
        materials.append(replace(record, structure=structure, cell=cell,
                                 cell_space_group_number=found))
    return materials


def _material_stem(material: Material) -> str:
    """``{formula}_{space group}_{ID}`` with ``/`` and ``_`` dropped from the symbol."""
    symbol = material.space_group.replace("/", "").replace("_", "")
    return f"{material.formula}_{symbol}_{material.material_id}"


def poscar_filename(material: Material, directory: bool = False) -> str:
    """The default file name of a fetched material's POSCAR.

    ``{prefix}_{formula}_{space group}_{ID}`` with the prefix of
    :data:`POSCAR_PREFIXES` for the cell: ``POSCAR_Sr2TiO4_I4mmm_mp-5532``
    (conventional cell), ``PPOSCAR_Sr2TiO4_I4mmm_mp-5532`` (primitive
    cell).  The ``/`` and ``_`` of
    the space-group symbol are dropped (``P6_3/mmc`` -> ``P63mmc``).  With
    ``directory``, ``{formula}_{space group}_{ID}/{prefix}``
    (``Sr2TiO4_I4mmm_mp-5532/POSCAR``).
    """
    prefix = POSCAR_PREFIXES.get(material.cell or DEFAULT_CELL, "POSCAR")
    stem = _material_stem(material)
    return os.path.join(stem, prefix) if directory else f"{prefix}_{stem}"


def poscar_text(material: Material) -> str:
    """The POSCAR of a fetched material; the comment line names its source."""
    from pymatgen.io.vasp import Poscar

    if material.structure is None:
        raise SystemExit(f"ERROR: {material.material_id} has no structure; use fetch_materials().")
    cell = {"primitive": "primitive cell"}.get(material.cell, "conventional cell")
    comment = (f"{material.formula} {material.space_group} {material.material_id} "
               f"(Materials Project, {cell})")
    return Poscar(_tidy(material.structure), comment=comment,
                  sort_structure=True).get_str(significant_figures=12)


def write_poscar(material: Material, path: str | None = None, *,
                 directory: bool = False, overwrite: bool = False,
                 rename_hint: bool = True) -> tuple[str, str]:
    """Write the POSCAR of a fetched material.

    Args:
        material: A material from :func:`fetch_materials`.
        path: File name; default :func:`poscar_filename` (``POSCAR_...``
            for the conventional cell, ``PPOSCAR_...`` for the primitive
            one).
        directory: Write ``{formula}_{space group}_{ID}/POSCAR`` (``PPOSCAR``
            for the primitive cell) instead.
        overwrite: Replace an existing file with different content.
        rename_hint: Suggest ``-o`` in the error about a file in the way
            (the command line turns it off where ``-o`` is not allowed).

    Returns:
        ``(path, status)`` with status ``"wrote"``, ``"kept"`` (an identical
        file was already there) or ``"replaced"``.

    Raises:
        SystemExit: A different file is in the way and ``overwrite`` is
            False, the path is a directory, or the file cannot be written
            (``ValueError`` through :mod:`crystod.search`).
    """
    text = poscar_text(material)
    if path is None:
        path = poscar_filename(material, directory=directory)
    if not os.path.basename(path) or os.path.isdir(path):
        raise SystemExit(f"ERROR: {path} is a directory; give a file name.")
    parent = os.path.dirname(path)
    if parent and os.path.exists(parent) and not os.path.isdir(parent):
        raise SystemExit(f"ERROR: {parent} exists and is not a directory.")
    status = "wrote"
    if os.path.exists(path):
        try:
            with open(path, "rb") as handle:
                same = handle.read() == text.encode("utf-8")
        except OSError:
            same = False
        if same:
            return path, "kept"
        if not overwrite:
            rename = ", choose another name with -o," if rename_hint else ""
            raise SystemExit(
                f"ERROR: {path} already exists with different content; remove it{rename} "
                "or pass --force."
            )
        status = "replaced"
    try:
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
    except OSError as exc:
        raise SystemExit(f"ERROR: cannot write {path}: {exc.strerror or exc}.") from None
    return path, status
