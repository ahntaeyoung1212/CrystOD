"""ISO-IR (ISOTROPY Software Suite) irreducible-representation tables.

Parses the ISO-IR data files of Stokes & Campbell (2011 version)

    CIR_data.txt : complex irreducible representations
    PIR_data.txt : physically irreducible representations

and evaluates small-representation (little-group) characters at arbitrary
k points, including non-special k vectors (symmetry lines, planes and the
general point) that carry free parameters alpha/beta/gamma.

The tables store, for every irrep, the FULL space-group representation
matrices (all arms of the star) for the coset representatives in the
standard conventional setting used by ISOTROPY (orthorhombic axes abc,
monoclinic axes a(b)c cell choice 1, origin choice 2, hexagonal axes).
The small representation at one arm is the diagonal block of that arm,
multiplied by the translation phase exp(+2*pi*i k.t) [ISO-IR convention;
note spgrep uses exp(-2*pi*i k.t), so spgrep characters are matched
against the COMPLEX CONJUGATE of the ISO-IR characters].

Every irrep label CrystOD prints comes from these tables (special points,
lines, planes and the general point).  The labels follow the Miller-Love /
ISOTROPY convention (e.g. T1..T5, DT5, LD3, GP1); a k point whose star is
the -k partner of a tabulated one (PA of I-4; the star at the negative
parameter of a polar line) is named as the ISOTROPY software names it, with
the complex conjugate irreps (PA1 = conj P1, LE1 = conj LD1; see
``minus_k_type``).

Data location: the gzip-compressed table ``CIR_data.txt.gz`` is bundled
inside the crystod package directory itself.  The lookup order is the
environment variable ``CRYSTOD_ISOIR_PATH`` first, then the package
directory, then ``<repository root>/ISOTROPY`` (the original ISO-IR
download layout with ``CIR_data/CIR_data.txt``).

File format (from CIR_data.f / PIR_data.f):
  header line:
      irnum sgnum "sgsymbol" "irlabel" irdim irtype kcount pmkcount opcount
  k vectors (CIR: kcount arms, PIR: pmkcount arms), 16 ints per arm,
  column-major kvec(4,4):
      col 1     = (x, y, z, denominator) constant part
      cols 2..4 = alpha/beta/gamma coefficient columns (x, y, z, denom)
  per operator (opcount of them):
      16 ints: 4x4 augmented operator matrix, ROW-major, common
               denominator at [3][3]
      [only if k is non-special] 4 ints: IR-translation (x, y, z, denom)
      irdim^2 IR-matrix entries, row-major
          CIR: complex tokens "(re,im)"
          PIR: bare real tokens
"""
from __future__ import annotations

import gzip
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

_HEADER_RE = re.compile(
    r'^\s*(\d+)\s+(\d+)\s+"([^"]*)"\s+"([^"]*)"'
    r"\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$"
)

# centering translations (conventional basis) by first letter of the HM symbol
_CENTERING_TRANSLATIONS = {
    "P": [],
    "A": [(0.0, 0.5, 0.5)],
    "B": [(0.5, 0.0, 0.5)],
    "C": [(0.5, 0.5, 0.0)],
    "F": [(0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)],
    "I": [(0.5, 0.5, 0.5)],
    "R": [(2 / 3, 1 / 3, 1 / 3), (1 / 3, 2 / 3, 2 / 3)],
}

_KTYPE_RE = re.compile(r"^([A-Z]+)")


# In a group without inversion the star of -k can differ from the star of k.
# ISO-IR tabulates one of the two; the ISOTROPY software names the other by
# its own letters in the physically irreducible labels (LD1LE1, DT6DU6,
# GP1GQ1, P1PA1, P1PC1 in its data file data_little.txt): the next letter
# for LD, DT, SM and GP, the suffix A otherwise, and the suffix C for the
# types below.  Over all 230 groups these names exist exactly for the k
# types whose -k lies outside the star (4141 types checked).
_MINUS_K_LETTERS = {"LD": "LE", "DT": "DU", "SM": "SN", "GP": "GQ"}
_MINUS_K_EXCEPTIONS = {
    143: {"P": "PC", "B": "BC", "C": "CC", "D": "DC", "E": "EC"},
    144: {"P": "PC", "B": "BC", "C": "CC", "D": "DC", "E": "EC"},
    145: {"P": "PC", "B": "BC", "C": "CC", "D": "DC", "E": "EC"},
    156: {"D": "DC"},
    157: {"P": "PC", "C": "CC"},
    158: {"D": "DC"},
    159: {"P": "PC", "C": "CC"},
    174: {"B": "BC", "E": "EC"},
}


def minus_k_type(ktype: str, sgnum: Optional[int] = None) -> str:
    """ISOTROPY's name of the -k partner of a k-vector type (``P`` ->
    ``PA``, ``LD`` -> ``LE``, ``P`` -> ``PC`` in P3).

    Meaningful only for a type whose -k lies outside its star; for another
    type the name returned may be a tabulated type of its own (``ZA`` of
    P23), and the callers never ask for it."""
    if sgnum is not None:
        special = _MINUS_K_EXCEPTIONS.get(int(sgnum), {}).get(ktype)
        if special is not None:
            return special
    return _MINUS_K_LETTERS.get(ktype, ktype + "A")


def minus_k_label(label: str, sgnum: Optional[int] = None) -> str:
    """Name of the -k partner of a k-vector type or irrep: ISOTROPY's
    partner letters on the type (``P1`` -> ``PA1``, ``DT6`` -> ``DU6``).
    The partner irrep with that name is the complex conjugate of the
    tabulated one."""
    return _KTYPE_RE.sub(lambda match: minus_k_type(match.group(1), sgnum), label)


def minus_k_base(label: str, sgnum: int) -> Optional[str]:
    """The tabulated label whose -k partner ``label`` names (``LE1`` ->
    ``LD1``), or None when ``label`` is not such a name in this group."""
    match = _KTYPE_RE.match(label)
    if match is None:
        return None
    prefix = match.group(1)
    types = {ir.ktype for ir in load_isoir_irreps(int(sgnum), "cir")}
    if prefix in types:
        return None
    for ktype in sorted(types):
        if minus_k_type(ktype, sgnum) == prefix:
            return ktype + label[len(prefix):]
    return None


def _parameter_key(params) -> tuple:
    """Order of line or plane parameters: the canonical value is the
    smallest, the positive one first among values of equal size."""
    q = np.round(np.asarray(params, dtype=float), 7) + 0.0
    return (round(float(q @ q), 12), tuple(-q))


@dataclass
class IsoIrrep:
    """One irrep block of an ISO-IR data file."""

    irnum: int
    sgnum: int
    sgsymbol: str
    label: str
    dim: int
    irtype: int
    kcount: int
    pmkcount: int
    opcount: int
    kvecs: np.ndarray = field(repr=False, default=None)  # (narms, 4, 4) int
    special: bool = True
    rotations: np.ndarray = field(repr=False, default=None)  # (nop, 3, 3) int
    translations: np.ndarray = field(repr=False, default=None)  # (nop, 3)
    irtrans: np.ndarray = field(repr=False, default=None)  # (nop, 3)
    matrices: np.ndarray = field(repr=False, default=None)  # (nop, dim, dim)

    @property
    def centering(self) -> str:
        return self.sgsymbol[0]

    @property
    def narms(self) -> int:
        return len(self.kvecs)

    @property
    def small_dim(self) -> int:
        return self.dim // self.narms

    @property
    def ktype(self) -> str:
        """k-vector type label, e.g. 'T' for 'T5', 'GM' for 'GM1+'."""
        return _KTYPE_RE.match(self.label).group(1)

    @property
    def num_free_params(self) -> int:
        used = 0
        for p in range(3):
            col = self.kvecs[0][p + 1]
            if col[3] != 0 and any(col[j] != 0 for j in range(3)):
                used += 1
        return used

    def arm_k(self, arm: int, params) -> np.ndarray:
        """k vector of an arm (conventional reciprocal basis) at parameters."""
        kv = self.kvecs[arm]
        k = np.array([kv[0][j] / kv[0][3] for j in range(3)], dtype=float)
        for p in range(3):  # alpha, beta, gamma
            col = kv[p + 1]
            if col[3] != 0:
                k = k + params[p] * np.array(
                    [col[j] / col[3] for j in range(3)], dtype=float
                )
        return k

    def _arm_parametrization(self, arm: int):
        """``(const, parameter indices, A, pinv(A))`` of an arm, with
        ``arm_k(arm, params) = const + A @ params[indices]``; ``A`` is None
        for a special point."""
        cache = self.__dict__.setdefault("_arm_cache", {})
        if arm not in cache:
            kv = self.kvecs[arm]
            const = np.array([kv[0][j] / kv[0][3] for j in range(3)], dtype=float)
            index, columns = [], []
            for p in range(3):
                c = kv[p + 1]
                if c[3] != 0 and any(c[j] != 0 for j in range(3)):
                    index.append(p)
                    columns.append([c[j] / c[3] for j in range(3)])
            if columns:
                A = np.array(columns, dtype=float).T
                cache[arm] = (const, index, A, np.linalg.pinv(A))
            else:
                cache[arm] = (const, index, None, None)
        return cache[arm]

    def match_k(self, k, atol: float = 1e-6) -> Optional[tuple[int, np.ndarray]]:
        """Find (arm, params) with arm_k(arm, params) == k modulo the
        reciprocal lattice of the (possibly centered) crystal lattice.

        `k` is in conventional fractional reciprocal coordinates.

        On a line or plane the same k is reached with several parameter
        values: through another arm (``(0,-1/2,0)`` is the arm ``(0,a,0)`` at
        ``a = -1/2`` and the arm ``(0,-a,0)`` at ``a = 1/2``) and through a
        reciprocal lattice vector.  The irrep an ISO-IR label names depends
        on the parameter value (DT3 at ``-a`` is DT4 at ``a`` in F-43m), so
        one value has to be singled out.  The parameters returned are the
        smallest ones, and among those of equal size the positive ones
        (the largest in lexicographic order).  The choice is a property of
        the star: every arm of one star gets the same parameters, and k and
        k + G do as well, so that a label does not depend on which arm or
        which copy of k the caller works with.
        """
        k = np.asarray(k, dtype=float)
        best = None
        for arm in range(self.narms):
            const, index, A, A_pinv = self._arm_parametrization(arm)
            d = k - const
            if A is None:
                n = np.rint(d)
                if np.all(np.abs(d - n) <= atol) and _is_reciprocal_lattice_vector(
                    n, self.centering
                ):
                    return arm, np.zeros(3)
                continue
            # A p = d + G for a reciprocal lattice vector G near -d
            offsets = _reciprocal_offsets_near(self.centering, -np.rint(d))
            rhs = d + offsets
            sols = rhs @ A_pinv.T
            hits = np.all(np.abs(sols @ A.T - rhs) <= atol, axis=1)
            for sol in sols[hits]:
                params = np.zeros(3)
                params[index] = sol
                key = _parameter_key(params)
                if best is None or key < best[0]:
                    best = (key, arm, params)
        if best is None:
            return None
        return best[1], best[2]

    def find_operator(self, rotation) -> Optional[int]:
        index = self.__dict__.get("_operator_index")
        if index is None:
            index = {}
            for i in range(self.opcount):
                index.setdefault(
                    tuple(int(v) for v in np.asarray(self.rotations[i]).ravel()), i
                )
            self.__dict__["_operator_index"] = index
        rotation = np.asarray(rotation)
        if rotation.shape != (3, 3):
            return None
        try:
            return index.get(tuple(int(v) for v in rotation.ravel()))
        except (TypeError, ValueError):
            return None

    def small_character(self, rotation, translation, arm: int, params) -> complex:
        """Character of the small representation at the given arm/params for
        the conventional-setting operator {rotation|translation}, in the
        ISO-IR phase convention exp(+2*pi*i k.t).
        """
        i = self.find_operator(rotation)
        if i is None:
            raise LookupError(f"operator not found in ISO-IR {self.label}")
        dt = np.asarray(translation, dtype=float) - self.translations[i]
        if not _is_lattice_translation(dt, self.centering):
            raise LookupError(
                f"translation mismatch for ISO-IR {self.label}: {dt}"
            )
        tph = dt + (self.irtrans[i] if not self.special else 0.0)
        kk = self.arm_k(arm, params)
        phase = np.exp(2j * np.pi * np.dot(kk, tph))
        nb = self.small_dim
        block = self.matrices[i][
            arm * nb : (arm + 1) * nb, arm * nb : (arm + 1) * nb
        ]
        return phase * np.trace(block)

    def in_little_group(self, rotation, arm: int, params) -> bool:
        """Is {rotation|*} in the little group of the arm's k?  (R^-T k = k
        modulo the reciprocal lattice of the centered crystal lattice.)
        """
        kk = self.arm_k(arm, params)
        kp = np.linalg.inv(np.asarray(rotation, dtype=float).T) @ kk
        d = kp - kk
        di = np.rint(d)
        if not np.allclose(d, di, atol=1e-6):
            return False
        return _is_reciprocal_lattice_vector(di, self.centering)


_INTEGER_BOX = np.array(
    [(gx, gy, gz) for gx in range(-3, 4) for gy in range(-3, 4) for gz in range(-3, 4)],
    dtype=float,
)


def _reciprocal_offsets_near(centering: str, base) -> np.ndarray:
    """Reciprocal lattice vectors of the centered lattice (conventional
    reciprocal basis) within three units of the integer vector ``base``."""
    offsets = np.asarray(base, dtype=float) + _INTEGER_BOX
    keep = np.ones(len(offsets), dtype=bool)
    for t in _CENTERING_TRANSLATIONS[centering]:
        s = offsets @ np.asarray(t, dtype=float)
        keep &= np.abs(s - np.rint(s)) < 1e-9
    return offsets[keep]


def _is_reciprocal_lattice_vector(G, centering: str) -> bool:
    """Is the integer vector G (conventional reciprocal basis) a reciprocal
    lattice vector of the centered crystal lattice?  True iff G.t is an
    integer for every centering translation t.
    """
    for t in _CENTERING_TRANSLATIONS[centering]:
        s = float(np.dot(G, t))
        if not np.isclose(s - round(s), 0.0, atol=1e-9):
            return False
    return True


def _is_lattice_translation(t, centering: str, atol: float = 1e-6) -> bool:
    """Is t (conventional basis) a translation of the centered lattice?"""
    for c in [np.zeros(3)] + [np.array(v) for v in _CENTERING_TRANSLATIONS[centering]]:
        d = np.asarray(t) - c
        if np.allclose(d - np.rint(d), 0, atol=atol):
            return True
    return False


_COMPLEX_TOKEN_RE = re.compile(r"\(([^,]+),([^)]+)\)")


def _parse_complex_token(token: str) -> complex:
    m = _COMPLEX_TOKEN_RE.fullmatch(token)
    if not m:
        raise ValueError(f"bad complex token in ISO-IR data: {token!r}")
    return complex(float(m.group(1)), float(m.group(2)))


class _TokenStream:
    """Whitespace-token stream over the lines following a header line."""

    def __init__(self, line_iter):
        self._lines = line_iter
        self._buf: list[str] = []
        self._pos = 0

    def next_tokens(self, n: int) -> list[str]:
        out: list[str] = []
        while len(out) < n:
            if self._pos >= len(self._buf):
                self._buf = next(self._lines).split()
                self._pos = 0
                continue
            out.append(self._buf[self._pos])
            self._pos += 1
        return out


def _isoir_data_file(data_dir: Path, kind: str) -> Optional[Path]:
    """Path of the (possibly gzip-compressed) data file of one kind.

    Two layouts are accepted: the flat package layout (``CIR_data.txt.gz``
    directly inside ``data_dir``, as bundled with the crystod package) and
    the original ISO-IR distribution layout (``CIR_data/CIR_data.txt``).
    """
    name = f"{kind.upper()}_data.txt"
    for base in (data_dir / name, data_dir / f"{kind.upper()}_data" / name):
        for path in (base, base.parent / (base.name + ".gz")):
            if path.is_file():
                return path
    return None


def find_isoir_data_dir() -> Optional[Path]:
    """Locate the ISO-IR data directory.

    Search order: the ``CRYSTOD_ISOIR_PATH`` environment variable, the
    crystod package directory itself (bundled ``CIR_data.txt.gz``), then
    ``<repository root>/ISOTROPY`` (original ISO-IR download layout).
    """
    env = os.environ.get("CRYSTOD_ISOIR_PATH")
    candidates = []
    if env:
        candidates.append(Path(env))
    package_dir = Path(__file__).resolve().parent
    candidates.append(package_dir)
    candidates.append(package_dir.parent / "ISOTROPY")
    for cand in candidates:
        if _isoir_data_file(cand, "cir") is not None:
            return cand
    return None


_CACHE: dict[tuple[str, int, str], list[IsoIrrep]] = {}


def load_isoir_irreps(sgnum: int, kind: str = "cir",
                      data_dir: Optional[Path] = None) -> list[IsoIrrep]:
    """Parse all irreps of one space group from an ISO-IR data file.

    Direct access to the tables behind every ISO-IR label that CrystOD
    prints: for each irrep the full space-group matrices over the coset
    representatives of the ISOTROPY standard setting, the (parametrized) k
    vectors of every star arm, and the operator translations.  Results are
    cached per (file, space group).

    Args:
        sgnum: Space-group number (1-230).
        kind: ``"cir"`` for the complex irreps (``CIR_data``, bundled with
            the package) or ``"pir"`` for the physically irreducible ones
            (``PIR_data``, not bundled).
        data_dir: Directory holding the data file; by default the lookup
            order of ``find_isoir_data_dir`` (``CRYSTOD_ISOIR_PATH``, the
            package directory, ``<repository root>/ISOTROPY``).

    Returns:
        One ``IsoIrrep`` record per irrep, in table order, with ``label``
        (``"GM4-"``, ``"DT5"``, ...), ``dim`` (full dimension), ``narms``,
        ``small_dim``, ``ktype``, ``num_free_params``, the arrays
        ``kvecs``, ``rotations``, ``translations``, ``irtrans`` and
        ``matrices``, and the methods ``arm_k``, ``match_k``,
        ``small_character`` and ``in_little_group``.

    Raises:
        ValueError: ``kind`` is neither ``"cir"`` nor ``"pir"``, or the file
            is malformed.
        FileNotFoundError: No data directory or data file found.

    Example:
        >>> from crystod import group
        >>> irreps = group.load_isoir_irreps(221)
        >>> len(irreps)
        72
        >>> [ir.label for ir in irreps if ir.ktype == "R"]
        ['R1+', 'R2+', 'R3+', 'R4+', 'R5+', 'R1-', 'R2-', 'R3-', 'R4-', 'R5-']
    """
    if kind not in ("cir", "pir"):
        raise ValueError(f"kind must be 'cir' or 'pir', got {kind!r}")
    if data_dir is None:
        data_dir = find_isoir_data_dir()
    if data_dir is None:
        raise FileNotFoundError(
            "ISO-IR data directory not found (set CRYSTOD_ISOIR_PATH or place "
            "the ISOTROPY directory next to the crystod package)"
        )
    path = _isoir_data_file(data_dir, kind)
    if path is None:
        raise FileNotFoundError(
            f"ISO-IR {kind.upper()} data file not found under {data_dir}"
        )
    key = (str(path), sgnum, kind)
    if key in _CACHE:
        return _CACHE[key]

    opener = gzip.open if path.suffix == ".gz" else open
    irreps: list[IsoIrrep] = []
    with opener(path, "rt") as f:
        lines = iter(f)
        for line in lines:
            if '"' not in line:
                continue
            m = _HEADER_RE.match(line)
            if not m:
                continue
            sg = int(m.group(2))
            if sg > sgnum:
                break
            if sg != sgnum:
                continue
            irnum = int(m.group(1))
            sgsym = m.group(3).strip()
            irlabel = m.group(4).strip()
            dim = int(m.group(5))
            irtype = int(m.group(6))
            kcount = int(m.group(7))
            pmkcount = int(m.group(8))
            nop = int(m.group(9))
            # CIR stores the full star of k, PIR only the star of +/-k
            narms = kcount if kind == "cir" else pmkcount
            ts = _TokenStream(lines)
            kints = [int(t) for t in ts.next_tokens(16 * narms)]
            kvecs = np.array(kints, dtype=int).reshape(narms, 4, 4)
            special = True
            for arm in range(narms):
                for col in (1, 2, 3):
                    if any(kvecs[arm][col][j] != 0 for j in range(3)):
                        special = False
            rotations = np.zeros((nop, 3, 3), dtype=int)
            translations = np.zeros((nop, 3), dtype=float)
            irtrans = np.zeros((nop, 3), dtype=float)
            matrices = np.zeros((nop, dim, dim), dtype=complex)
            for i in range(nop):
                a = np.array(
                    [int(t) for t in ts.next_tokens(16)], dtype=int
                ).reshape(4, 4)
                denom = a[3][3]
                rotations[i] = a[:3, :3] // denom
                if not np.array_equal(rotations[i] * denom, a[:3, :3]):
                    raise ValueError(
                        f"non-integer rotation in ISO-IR irrep {irnum}"
                    )
                translations[i] = a[:3, 3] / denom
                if not special:
                    v = [int(t) for t in ts.next_tokens(4)]
                    irtrans[i] = np.array(v[:3], dtype=float) / v[3]
                tokens = ts.next_tokens(dim * dim)
                if kind == "cir":
                    values = [_parse_complex_token(t) for t in tokens]
                else:
                    values = [float(t) for t in tokens]
                matrices[i] = np.array(values, dtype=complex).reshape(dim, dim)
            irreps.append(
                IsoIrrep(
                    irnum, sg, sgsym, irlabel, dim, irtype, kcount,
                    pmkcount, nop, kvecs, special, rotations, translations,
                    irtrans, matrices,
                )
            )
    _CACHE[key] = irreps
    return irreps


def isoir_available() -> bool:
    return find_isoir_data_dir() is not None


_ISO_HALL_CACHE: dict[int, int] = {}


def iso_hall_number(sgnum: int) -> int:
    """spglib Hall number of the ISO-IR standard setting of a space group.

    ISOTROPY's preferences: origin choice 2, orthorhombic axes abc,
    monoclinic axes a(b)c cell choice 1, hexagonal axes.  In terms of the
    spglib `choice` strings this means, in order of preference:
    '2' (origin choice 2), '' (unique standard setting), 'b1' (monoclinic),
    'H' (rhombohedral on hexagonal axes), '1' (origin choice 1 only).
    """
    if not _ISO_HALL_CACHE:
        import spglib

        by_sg: dict[int, list[tuple[int, str]]] = {}
        for hall in range(1, 531):
            t = spglib.get_spacegroup_type(hall)
            number = t['number'] if isinstance(t, dict) else t.number
            choice = t['choice'] if isinstance(t, dict) else t.choice
            by_sg.setdefault(number, []).append((hall, choice))
        for number, entries in by_sg.items():
            chosen = entries[0][0]
            for preferred in ("2", "", "b1", "H", "1"):
                hits = [h for h, c in entries if c == preferred]
                if hits:
                    chosen = hits[0]
                    break
            _ISO_HALL_CACHE[number] = chosen
    return _ISO_HALL_CACHE[sgnum]


class IsoIRLabeler:
    """Label spgrep small representations with ISO-IR (Miller-Love) labels.

    The labeling engine shared by every CrystOD command (crystal orbitals,
    phonons, spin bases, ``crystod-group --table --sg``): spgrep computes
    the small irreps of the little group of k in the primitive basis of the
    user's cell, and this class matches their characters against the
    ISO-IR tables in the ISOTROPY standard setting (origin choice 2,
    orthorhombic axes abc, monoclinic axes a(b)c cell choice 1, hexagonal
    axes), at tabulated k points as well as on symmetry lines, planes and
    the general point.  Because ISO-IR uses the phase convention
    ``exp(+2 pi i k.t)`` and spgrep ``exp(-2 pi i k.t)``, spgrep characters
    are compared with the complex conjugate of the ISO-IR characters.

    Args:
        sgnum: Space-group number (1-230).
        transformation_matrix: spglib-style transformation ``P`` into the
            ISO-IR setting (``x_conventional = P x_primitive +
            origin_shift``); give it together with ``origin_shift`` when
            ``cell`` is omitted.
        origin_shift: The origin shift of that transformation.
        cell: Alternatively, the primitive cell ``(lattice,
            scaled_positions, numbers)`` whose operations feed spgrep; the
            transformation is then computed with spglib for the Hall number
            of the ISO-IR setting.
        symprec: Symmetry tolerance for spglib when ``cell`` is given.

    Attributes:
        sgnum: The space-group number.
        P: The transformation matrix into the ISO-IR setting; ``Pinv`` its
            inverse and ``origin_shift`` the accompanying shift.
        irreps: The ``IsoIrrep`` records of the space group, from
            ``load_isoir_irreps``.

    Raises:
        ValueError: spglib could not standardize ``cell`` to the ISO-IR
            setting.
        FileNotFoundError: The ISO-IR data file is not available.

    Example:
        >>> import numpy as np
        >>> from crystod import group
        >>> labeler = group.IsoIRLabeler(221, transformation_matrix=np.eye(3),
        ...                              origin_shift=np.zeros(3))
        >>> labeler.kpoint_name([0.5, 0.5, 0.4]), labeler.kpoint_name([0, 0, 0])
        ('T', 'GM')
    """

    def __init__(self, sgnum: int, transformation_matrix=None,
                 origin_shift=None, cell=None, symprec: float = 1e-5):
        self.sgnum = sgnum
        if cell is not None:
            import spglib

            try:
                dataset = spglib.get_symmetry_dataset(
                    cell, symprec=symprec, hall_number=iso_hall_number(sgnum)
                )
            except Exception as exc:  # spglib raises its own SpglibError family
                raise ValueError(
                    "spglib standardization to the ISO-IR setting failed"
                ) from exc
            if dataset is None:
                raise ValueError(
                    "spglib standardization to the ISO-IR setting failed"
                )
            number = (
                dataset['number'] if isinstance(dataset, dict)
                else dataset.number
            )
            if number != sgnum:
                raise ValueError(
                    "spglib standardization to the ISO-IR setting failed"
                )
            transformation_matrix = (
                dataset['transformation_matrix'] if isinstance(dataset, dict)
                else dataset.transformation_matrix
            )
            origin_shift = (
                dataset['origin_shift'] if isinstance(dataset, dict)
                else dataset.origin_shift
            )
        self.P = np.asarray(transformation_matrix, dtype=float)
        self.Pinv = np.linalg.inv(self.P)
        self.origin_shift = np.asarray(origin_shift, dtype=float)
        self.irreps = load_isoir_irreps(sgnum, "cir")

    # -- setting conversion --------------------------------------------------
    def conventional_k(self, k_primitive) -> np.ndarray:
        """k vector in the ISO-IR conventional reciprocal basis.

        Args:
            k_primitive: k vector in the primitive reciprocal basis.

        Returns:
            ``k_primitive @ P^-1`` as a float array.
        """
        return np.asarray(k_primitive, dtype=float) @ self.Pinv

    def conventional_operations(self, rotations, translations):
        """Map primitive-basis operations into the ISO-IR conventional setting.

        Args:
            rotations: Integer rotation matrices in the primitive basis.
            translations: Their fractional translations.

        Returns:
            ``[(R_c, t_c), ...]`` with ``R_c = P R P^-1`` and
            ``t_c = P t + (1 - R_c) origin_shift``.
        """
        conv = []
        for R_p, t_p in zip(rotations, translations):
            R_c = np.rint(self.P @ R_p @ self.Pinv).astype(int)
            t_c = self.P @ np.asarray(t_p, dtype=float) + (
                np.eye(3) - R_c
            ) @ self.origin_shift
            conv.append((R_c, t_c))
        return conv

    def kpoint_name(self, k_primitive, minus_k: bool = True) -> Optional[str]:
        """Most specific ISO-IR k-vector type label containing a k point.

        Args:
            k_primitive: k vector in the primitive reciprocal basis.
            minus_k: Name a point whose -k partner is tabulated more
                specifically by the partner's name (``"PA"``, see
                ``minus_k_type``).

        Returns:
            The type label with the fewest free parameters, e.g. ``"T"``
            for ``(1/2, 1/2, 0.4)`` in Pm-3m; ``None`` when no entry
            matches.
        """
        k_conv = self.conventional_k(k_primitive)
        best = self._most_specific_type(k_conv)
        if minus_k:
            # a point of a -k star that is not tabulated (PA of I-4) lies on
            # a tabulated line; the point itself is the partner of the
            # tabulated one
            partner = self._most_specific_type(-k_conv)
            if partner is not None and (best is None or partner[0] < best[0]):
                return minus_k_type(partner[1], self.sgnum)
        return best[1] if best is not None else None

    def _most_specific_type(self, k_conv):
        """``(number of free parameters, k-type)`` of the most specific
        tabulated k-vector type containing k (the partner name on the -k
        side of a line or plane, see ``_minus_side``), or None."""
        families = self._matched_families(k_conv)
        if not families:
            return None
        ktype, family = families[0]
        if self._minus_side(ktype, family, k_conv) is not None:
            ktype = minus_k_type(ktype, self.sgnum)
        return family[0][0].num_free_params, ktype

    def _minus_side(self, ktype, family, k_conv):
        """The family at -k when k is the -k partner of the star there.

        On a line, a plane or the general point whose -k lies outside the
        star (a polar line such as LD of P4), k and -k are both on the
        tabulated type, at parameters p and -p.  The star with the
        canonical parameter (``_parameter_key``: the positive one) keeps the
        tabulated name; the other one is its -k partner, named as ISOTROPY
        names it (LE for LD, DU for DT; ``minus_k_type``) with the complex
        conjugates of the tabulated irreps.  Returns None when k is on the
        tabulated side or -k is in the star of k.
        """
        if family[0][0].num_free_params == 0:
            return None
        minus = dict(self._matched_families(-np.asarray(k_conv, dtype=float)))
        minus_family = minus.get(ktype)
        if minus_family is None or len(minus_family) != len(family):
            return None
        if not _parameter_key(minus_family[0][2]) < _parameter_key(family[0][2]):
            return None
        return minus_family

    # -- labeling ------------------------------------------------------------
    def label_characters(
        self,
        k_primitive,
        little_rotations,
        little_translations,
        spgrep_characters,
        atol: float = 1e-5,
        minus_k: bool = True,
    ) -> Optional[tuple[dict[int, str], str]]:
        """Match spgrep small-irrep characters against the ISO-IR tables.

        Args:
            k_primitive: k vector in the primitive reciprocal basis.
            little_rotations: Rotations of the little group of k in the
                primitive basis.
            little_translations: Their fractional translations.
            spgrep_characters: One character vector per spgrep irrep,
                aligned with the little-group operations and computed with
                the spgrep phase convention ``exp(-2 pi i k.t)``.
            atol: Tolerance of the character comparison.
            minus_k: When k has no assignment, label it through -k: the
                small irreps at k are the complex conjugates of those at
                -k, and only one star of such a pair is tabulated (P of
                I-4, not PA).  The labels then carry ISOTROPY's partner
                name (``"PA1"`` is the conjugate of ``"P1"``).  The -k
                partner of a line, plane or general point is named that
                way in any case (``"LE1"`` for the conjugate of ``"LD1"``;
                ``_minus_side``).

        Returns:
            ``({spgrep irrep index: ISO-IR label}, k-type label)``, e.g.
            ``({0: "R1+", ...}, "R")``, or ``None`` when no consistent
            assignment exists.
        """
        k_conv = self.conventional_k(k_primitive)
        conv_ops = self.conventional_operations(
            little_rotations, little_translations
        )
        conjugates = [np.conj(chi) for chi in spgrep_characters]
        for ktype, family in self._matched_families(k_conv):
            minus_family = self._minus_side(ktype, family, k_conv)
            if minus_family is not None:
                # the -k partner of the star at -k (LE of LD)
                result = self._try_family(minus_family, conv_ops, conjugates, atol)
                if result is not None:
                    return (
                        {m: minus_k_label(name, self.sgnum)
                         for m, name in result.items()},
                        minus_k_type(ktype, self.sgnum),
                    )
                continue
            result = self._try_family(family, conv_ops, spgrep_characters, atol)
            if result is not None:
                return result, ktype
        if minus_k:
            for ktype, family in self._matched_families(-k_conv):
                result = self._try_family(family, conv_ops, conjugates, atol)
                if result is not None:
                    return (
                        {m: minus_k_label(name, self.sgnum)
                         for m, name in result.items()},
                        minus_k_type(ktype, self.sgnum),
                    )
        return None

    def decompose_characters(
        self,
        k_primitive,
        little_rotations,
        little_translations,
        reducible_characters,
        atol: float = 1e-3,
    ) -> Optional[tuple[list[tuple[str, int, int]], str]]:
        """Decompose a reducible character vector into ISO-IR irreps.

        Used for phonopy band sets, whose characters can be reducible under
        accidental degeneracy.  As in ``label_characters``, the -k partner
        of a tabulated star gets ISOTROPY's partner names (PA1, LE1).

        Args:
            k_primitive: k vector in the primitive reciprocal basis.
            little_rotations: Rotations of the little group of k in the
                primitive basis.
            little_translations: Their fractional translations.
            reducible_characters: The character vector, aligned with the
                little-group operations, in the spgrep/phonopy phase
                convention ``exp(-2 pi i k.t)``.
            atol: Tolerance of the multiplicity check.

        Returns:
            ``([(label, multiplicity, small_dim), ...], k-type label)``, or
            ``None`` when no consistent decomposition exists.
        """
        results = self.decompose_characters_many(
            k_primitive, little_rotations, little_translations,
            [reducible_characters], atol=atol,
        )
        return results[0]

    def decompose_characters_many(
        self,
        k_primitive,
        little_rotations,
        little_translations,
        character_vectors,
        atol: float = 1e-3,
    ) -> list[Optional[tuple[list[tuple[str, int, int]], str]]]:
        """Decompose several reducible character vectors at one k point.

        Same as ``decompose_characters``, but the candidate-family search
        (the expensive part) is done once and shared by all vectors; use
        this for phonopy band sets, which all live at the same q.

        Args:
            k_primitive: k vector in the primitive reciprocal basis.
            little_rotations: Rotations of the little group of k in the
                primitive basis.
            little_translations: Their fractional translations.
            character_vectors: The reducible character vectors, each
                aligned with the little-group operations (phase convention
                ``exp(-2 pi i k.t)``).
            atol: Tolerance of the multiplicity check.

        Returns:
            One entry per input vector, each
            ``([(label, multiplicity, small_dim), ...], k-type label)`` or
            ``None`` when no consistent decomposition exists.
        """
        n_vectors = len(character_vectors)
        k_conv = self.conventional_k(k_primitive)
        conv_ops = self.conventional_operations(
            little_rotations, little_translations
        )
        e_index = None
        for j, (R_c, t_c) in enumerate(conv_ops):
            if np.array_equal(R_c, np.eye(3, dtype=int)) and np.allclose(
                np.asarray(t_c) - np.rint(t_c), 0, atol=1e-6
            ):
                e_index = j
                break
        if e_index is None:
            return [None] * n_vectors
        # k itself (the -k partner of the star at -k on a line, see
        # label_characters), then -k with the conjugate characters
        candidates = []
        for ktype, family in self._matched_families(k_conv):
            minus_family = self._minus_side(ktype, family, k_conv)
            if minus_family is None:
                candidates.append((ktype, family, False))
            else:
                candidates.append((ktype, minus_family, True))
        candidates += [
            (ktype, family, True) for ktype, family in self._matched_families(-k_conv)
        ]
        for ktype, family, conjugate in candidates:
            iso_chars = self._family_characters_checked(family, conv_ops)
            if iso_chars is None:
                continue
            name = minus_k_type(ktype, self.sgnum) if conjugate else ktype
            results: list[Optional[tuple[list[tuple[str, int, int]], str]]] = []
            for reducible in character_vectors:
                red = np.asarray(reducible, dtype=complex)
                counts = self._decompose_against(
                    family, iso_chars, np.conj(red) if conjugate else red,
                    e_index, len(conv_ops), atol,
                )
                if counts is not None and conjugate:
                    counts = [
                        (minus_k_label(label, self.sgnum), n, dim)
                        for label, n, dim in counts
                    ]
                results.append((counts, name) if counts is not None else None)
            # the family validity checks are vector-independent, so
            # per-vector failures here are numerical; keep them as None
            if any(result is not None for result in results):
                return results
        return [None] * n_vectors

    @staticmethod
    def _decompose_against(family, iso_chars, red, e_index, n_ops, atol):
        """Multiplicities of one reducible character vector in a family, or
        None when they are not consistent non-negative integers.

        The multiplicity of the irrep with characters conj(chi_iso) in red is
        n = (1/|G|) sum_g conj(conj(chi_iso(g))) red(g).
        """
        total_dim = int(round(red[e_index].real))
        counts: list[tuple[str, int, int]] = []
        dim_sum = 0
        for (ir, _, _), chi in zip(family, iso_chars):
            n = complex(np.dot(chi, red)) / n_ops
            ni = int(round(n.real))
            if abs(n - ni) > atol or ni < 0:
                return None
            if ni:
                counts.append((ir.label, ni, ir.small_dim))
                dim_sum += ni * ir.small_dim
        if dim_sum != total_dim:
            return None
        return counts

    def _matched_families(self, k_conv):
        """Candidate irreps whose star contains k, grouped by k-type and
        sorted most specific k type (fewest free parameters) first."""
        # the search depends on the tables and on k only, and every band of
        # every atom asks for the same few k points
        key = (self.sgnum,) + tuple(np.round(np.asarray(k_conv, dtype=float), 10))
        if key not in _FAMILY_CACHE:
            if len(_FAMILY_CACHE) >= 4096:
                _FAMILY_CACHE.clear()
            families: dict[str, list[tuple[IsoIrrep, int, np.ndarray]]] = {}
            by_type: dict = {}  # the irreps of one k type share their k vectors
            for ir in self.irreps:
                if ir.ktype not in by_type:
                    by_type[ir.ktype] = (ir.kvecs, ir.match_k(k_conv))
                kvecs, matched = by_type[ir.ktype]
                if kvecs is not ir.kvecs and not np.array_equal(kvecs, ir.kvecs):
                    matched = ir.match_k(k_conv)
                if matched is not None:
                    families.setdefault(ir.ktype, []).append(
                        (ir, matched[0], matched[1])
                    )
            _FAMILY_CACHE[key] = sorted(
                families.items(), key=lambda item: item[1][0][0].num_free_params
            )
        return _FAMILY_CACHE[key]

    def _family_characters_checked(self, family, conv_ops):
        """ISO-IR character vectors of a family, or None when the family does
        not describe the little group of these operations."""
        # all little-group operations must belong to the family's little group
        for ir, arm, params in family:
            for R_c, _ in conv_ops:
                if not ir.in_little_group(R_c, arm, params):
                    return None
        # the family must exhaust the little group
        if sum(ir.small_dim**2 for ir, _, _ in family) != len(conv_ops):
            return None
        try:
            return self._family_characters(family, conv_ops)
        except LookupError:
            # setting mismatch (operator or translation not found): the
            # transformation into the ISO-IR setting is wrong -- fail safe
            return None

    def _try_family(self, family, conv_ops, spgrep_characters, atol):
        iso_chars = self._family_characters_checked(family, conv_ops)
        if iso_chars is None:
            return None

        label_map: dict[int, str] = {}
        used: set[str] = set()
        for m, chi in enumerate(spgrep_characters):
            hits = [
                ir.label
                for (ir, _, _), chi_iso in zip(family, iso_chars)
                if ir.label not in used
                and np.allclose(chi, np.conj(chi_iso), atol=atol)
            ]
            if len(hits) != 1:
                return None
            label_map[m] = hits[0]
            used.add(hits[0])
        return label_map

    @staticmethod
    def _family_characters(family, conv_ops):
        iso_chars = []
        for ir, arm, params in family:
            iso_chars.append(
                np.array(
                    [
                        ir.small_character(R_c, t_c, arm, params)
                        for R_c, t_c in conv_ops
                    ]
                )
            )
        return iso_chars


# --------------------------------------------------------------- shared helpers

_LABELER_CACHE: dict = {}
_FAMILY_CACHE: dict = {}
_NORMALIZER_CACHE: dict = {}
_OPERATIONS_CACHE: dict = {}


# ------------------------------------------------- the ISO-IR frame of a cell
#
# An irrep label is only defined relative to a description of the crystal in
# the ISO-IR standard setting.  The setting leaves a choice: translating the
# origin by an element of the Euclidean normalizer (Si on Wyckoff 8a or 8b of
# Fd-3m, Sr or Ti at the origin of Pm-3m) is again a standard setting, and it
# permutes labels at zone-boundary points (L1+ <-> L2-, R4+ <-> R5-).  spglib
# returns one of these frames, and which one depends on the cell it is given,
# so two commands working on different cells of one structure (phonopy's
# primitive cell, the spglib-standardized primitive cell) would name one
# irrep differently.  The frame is therefore fixed by these rules:
#
# 1. a cell that is already in the ISO-IR setting keeps its own axes and
#    origin (the user's choice of setting is respected, including the sense
#    of a polar axis); so does an n1 x n2 x n3 supercell of such a cell
#    of any size (a non-diagonal supercell falls to rule 2);
# 2. otherwise the frame is a property of the structure, not of the cell it
#    came from.  The candidates are spglib's frame and its images under the
#    Euclidean normalizer: a proper rotation W of the lattice that maps the
#    group onto itself (the sixfold axis for P-6m2, the twofold axis that
#    reverses the polar axis of P4), followed by a normalizer translation.
#    Frames that keep the cell's own origin are preferred when there are
#    any; among the candidates the one with the lexicographically smallest
#    list of atomic positions is taken (coordinates compared by their
#    cluster rank at 0.71, 0.071 and 0.0071 times the symmetry tolerance,
#    _smallest_structure).  Along a polar axis the origin is free and carries
#    no information; it is put on an atom of the species with the smallest
#    atomic number, again by the smallest list;
# 3. a working cell derived from an input cell (primitive reduction,
#    standardization) inherits the frame of the input cell through the affine
#    map between the two.


def _dataset_item(dataset, name):
    return dataset[name] if isinstance(dataset, dict) else getattr(dataset, name)


def _iso_table_entry(sgnum: int) -> IsoIrrep:
    """The ISO-IR block listing every coset representative of the group."""
    return max(load_isoir_irreps(sgnum, "cir"), key=lambda ir: ir.opcount)


def _frame_is_valid(entry: IsoIrrep, rotations, translations, P, origin,
                    atol: float) -> bool:
    """Do the operations of a cell become the tabulated ISO-IR operations
    under x_iso = P x + origin?"""
    P = np.asarray(P, dtype=float)
    try:
        P_inv = np.linalg.inv(P)
    except np.linalg.LinAlgError:
        return False
    found = set()
    for R, t in zip(rotations, translations):
        R_c = P @ np.asarray(R, dtype=float) @ P_inv
        R_int = np.rint(R_c)
        if not np.allclose(R_c, R_int, atol=1e-6):
            return False
        j = entry.find_operator(R_int.astype(int))
        if j is None:
            return False
        t_c = P @ np.asarray(t, dtype=float) + (np.eye(3) - R_int) @ origin
        if not _is_lattice_translation(
            t_c - entry.translations[j], entry.centering, atol
        ):
            return False
        found.add(j)
    return len(found) == entry.opcount


def _polar_axes(entry: IsoIrrep) -> np.ndarray:
    """Conventional axes along which the origin is free: e_i is kept by
    every rotation of the group (z of P4, x and z of Pm, all of P1)."""
    identity = np.eye(3, dtype=int)
    return np.array([
        all(np.array_equal(np.asarray(R)[:, i], identity[:, i])
            for R in entry.rotations)
        for i in range(3)
    ])


def _normalizer_translations(sgnum: int, W=None) -> np.ndarray:
    """Translations w (on the 1/24 grid, zero along the polar axes) for which
    x -> W x + w maps the ISO-IR setting of a space group onto itself.

    ``W`` is a rotation in the conventional basis (default: the identity,
    which gives the origin shifts of the setting).  The result is empty when
    W is not the linear part of a normalizer element."""
    W = np.eye(3) if W is None else np.asarray(W, dtype=float)
    key = (int(sgnum), np.round(W, 6).tobytes())
    if key not in _NORMALIZER_CACHE:
        entry = _iso_table_entry(sgnum)
        polar = _polar_axes(entry)
        axes = [np.zeros(1) if polar[i] else np.arange(24) / 24.0 for i in range(3)]
        grid = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
        centerings = [np.zeros(3)] + [
            np.array(v) for v in _CENTERING_TRANSLATIONS[entry.centering]
        ]
        W_inv = np.linalg.inv(W)
        keep = np.ones(len(grid), dtype=bool)
        for R, tau in zip(entry.rotations, entry.translations):
            # {W|w} {R|tau} {W|w}^-1 = {W R W^-1 | W tau + (1 - W R W^-1) w}
            image_R = W @ np.asarray(R, dtype=float) @ W_inv
            image_int = np.rint(image_R)
            j = (
                entry.find_operator(image_int.astype(int))
                if np.allclose(image_R, image_int, atol=1e-6) else None
            )
            if j is None:
                keep[:] = False
                break
            image = (
                W @ np.asarray(tau, dtype=float)
                + grid @ (np.eye(3) - image_int).T
                - entry.translations[j]
            )
            lattice = np.zeros(len(grid), dtype=bool)
            for c in centerings:
                d = image - c
                lattice |= np.all(np.abs(d - np.rint(d)) < 1e-9, axis=1)
            keep &= lattice
        _NORMALIZER_CACHE[key] = grid[keep]
    return _NORMALIZER_CACHE[key]


def _lattice_rotations(P, lattice, centering: str, symprec: float) -> list:
    """Proper rotations of the crystal lattice, in the conventional basis of
    the frame ``x_iso = P x + o`` of a cell with the given lattice."""
    import spglib

    conventional = np.linalg.inv(np.asarray(P, dtype=float)).T @ np.asarray(
        lattice, dtype=float
    )
    points = [np.zeros(3)] + [np.array(v) for v in _CENTERING_TRANSLATIONS[centering]]
    symmetry = spglib.get_symmetry(
        (conventional, points, [1] * len(points)), symprec=symprec
    )
    if symmetry is None:
        return [np.eye(3)]
    seen, result = set(), []
    for R in _dataset_item(symmetry, "rotations"):
        R = np.asarray(R, dtype=int)
        if round(float(np.linalg.det(R))) != 1 or R.tobytes() in seen:
            continue
        seen.add(R.tobytes())
        result.append(R.astype(float))
    return result or [np.eye(3)]


def _smallest_structure(frames, positions, numbers, centering: str,
                        tolerance: float) -> int:
    """Index of the frame in which the sorted list of atomic positions of the
    conventional cell is the smallest (atoms of the species with the
    smallest type number on the origin, then on low fractions; the type
    numbers are atomic numbers in the CrystOD commands).

    Coordinates closer than the tolerance count as equal: a coordinate is
    replaced by the rank of its cluster among the coordinates of all frames,
    so that the comparison does not depend on how a value such as 0.07465
    happens to round.  Frames that tie are compared again at one tenth and
    one hundredth of the level; frames that still tie describe the
    structure identically within that tolerance.  The levels are 0.71 times
    powers of ten of the tolerance: coordinates typed with a few decimals
    differ by multiples of 1e-5, 1e-6, ..., and a level equal to such a gap
    would leave the decision to floating-point noise."""
    centerings = [np.zeros(3)] + [
        np.array(v) for v in _CENTERING_TRANSLATIONS[centering]
    ]
    positions = np.asarray(positions, dtype=float)
    species = np.tile(np.asarray(numbers, dtype=np.int64), len(centerings))
    coordinates = []
    for P, origin in frames:
        x = positions @ np.asarray(P, dtype=float).T + origin
        x = np.concatenate([x + c for c in centerings])
        # the wrap at 1 uses the first level too: a gap of exactly one
        # decimal unit between two atoms must not straddle it
        coordinates.append(x - np.floor(x + 0.71 * tolerance))
    alive = list(range(len(frames)))
    for level in (0.71 * tolerance, 0.071 * tolerance, 0.0071 * tolerance):
        if len(alive) == 1:
            break
        values = np.sort(np.concatenate([coordinates[i].ravel() for i in alive]))
        rank = np.concatenate(
            [[0], np.cumsum(np.diff(values) > level * (1.0 + 1e-9))]
        )
        keys = []
        for i in alive:
            index = rank[np.searchsorted(values, coordinates[i])]
            rows = np.unique(np.column_stack([species, index]), axis=0)
            keys.append(rows.ravel().tolist())
        smallest = min(keys)
        alive = [i for i, key in zip(alive, keys) if key == smallest]
    return alive[0]


def _cell_operations(sgnum: int, cell, symprec: float):
    """``(rotations, translations, P, origin)``: every operation of the
    crystal in the basis of the given cell, and spglib's frame
    ``x_iso = P x + origin`` of the cell; None when spglib and the ISO-IR
    table do not describe the same group.

    spglib lists only the operations that keep the lattice of the cell, so a
    supercell loses some (a 2x1x1 cell of a cubic crystal keeps 16 of the 48
    rotations) and gains pure translations.  The list returned here is the
    ISO-IR table carried back through spglib's frame instead; a rotation
    need not be an integer matrix in the basis of a supercell.  The lattice
    translations of the crystal come last, as operations with the identity
    rotation, so that a frame is valid only when it also has the lattice of
    the crystal."""
    key = (int(sgnum), float(symprec)) + _cell_key(cell)
    if key not in _OPERATIONS_CACHE:
        if len(_OPERATIONS_CACHE) >= 64:
            _OPERATIONS_CACHE.clear()
        _OPERATIONS_CACHE[key] = _cell_operations_uncached(sgnum, cell, symprec)
    return _OPERATIONS_CACHE[key]


def _cell_operations_uncached(sgnum: int, cell, symprec: float):
    import spglib

    try:
        dataset = spglib.get_symmetry_dataset(
            cell, symprec=symprec, hall_number=iso_hall_number(sgnum)
        )
        symmetry = spglib.get_symmetry(cell, symprec=symprec)
    except Exception:  # spglib raises its own SpglibError family
        return None
    if (
        dataset is None or symmetry is None
        or _dataset_item(dataset, "number") != sgnum
    ):
        return None
    P = np.asarray(_dataset_item(dataset, "transformation_matrix"), dtype=float)
    origin = np.asarray(_dataset_item(dataset, "origin_shift"), dtype=float)
    entry = _iso_table_entry(sgnum)
    atol = max(1e-4, 10.0 * float(symprec))
    P_inv = np.linalg.inv(P)
    # the operations spglib lists for the cell must be tabulated ones
    for R, t in zip(
        _dataset_item(symmetry, "rotations"), _dataset_item(symmetry, "translations")
    ):
        R_c = P @ np.asarray(R, dtype=float) @ P_inv
        R_int = np.rint(R_c)
        j = (
            entry.find_operator(R_int.astype(int))
            if np.allclose(R_c, R_int, atol=1e-6) else None
        )
        if j is None:
            return None
        t_c = P @ np.asarray(t, dtype=float) + (np.eye(3) - R_int) @ origin
        if not _is_lattice_translation(
            t_c - entry.translations[j], entry.centering, atol
        ):
            return None
    rotations, translations = [], []
    for R, t in zip(entry.rotations, entry.translations):
        R = np.asarray(R, dtype=float)
        rotations.append(P_inv @ R @ P)
        translations.append(P_inv @ (R @ origin + np.asarray(t, dtype=float) - origin))
    lattice_vectors = list(np.eye(3)) + [
        np.array(v) for v in _CENTERING_TRANSLATIONS[entry.centering]
    ]
    for v in lattice_vectors:
        rotations.append(np.eye(3))
        translations.append(P_inv @ v)
    return rotations, translations, P, origin


def _frame_matches(entry: IsoIrrep, operations, P, origin, atol: float) -> bool:
    """Is ``x_iso = P x + origin`` an ISO-IR frame of the crystal: the
    tabulated operations on the lattice of the crystal (not a sublattice or
    a superlattice of it)?"""
    rotations, translations, P_spglib, _ = operations
    try:
        volume = abs(np.linalg.det(np.asarray(P, dtype=float) @ np.linalg.inv(P_spglib)))
    except np.linalg.LinAlgError:
        return False
    if abs(volume - 1.0) > 1e-6:
        return False
    return _frame_is_valid(entry, rotations, translations, P, origin, atol)


def canonical_iso_frame(sgnum: int, cell, symprec: float = 1e-5):
    """The ISO-IR frame ``(P, origin_shift)`` of a cell, by rules 1 and 2.

    Args:
        sgnum: Space-group number (1-230).
        cell: spglib tuple ``(lattice, scaled_positions, numbers)``.
        symprec: Symmetry tolerance for spglib.

    Returns:
        ``(P, origin_shift)`` with ``x_iso = P x + origin_shift``, or
        ``None`` when spglib or the ISO-IR data cannot provide a frame.
    """
    lattice, positions, numbers = cell
    operations = _cell_operations(sgnum, cell, symprec)
    if operations is None:
        return None
    P, origin = operations[2], operations[3]
    entry = _iso_table_entry(sgnum)
    atol = max(1e-4, 10.0 * float(symprec))
    zero = np.zeros(3)

    # rule 1: the cell's own axes and origin, for a conventional cell, its
    # primitive cell in the standard centring, or an n1 x n2 x n3 supercell
    # of either.  n_i is read from the lattice: the i-th cell vector is n_i
    # times a primitive lattice vector (the edges of both unit cells are
    # primitive).
    units = [np.eye(3)]
    if entry.centering != "P":
        from phonopy.structure.cells import get_primitive_matrix_by_centring

        units.append(
            np.array(get_primitive_matrix_by_centring(entry.centering), dtype=float)
        )
    content = np.linalg.inv(units[-1]) @ P  # columns: cell vectors, primitive basis
    if np.allclose(content, np.rint(content), atol=1e-5):
        multiples = np.gcd.reduce(np.abs(np.rint(content).astype(np.int64)), axis=0)
        if np.all(multiples >= 1):
            for unit in units:
                P0 = unit @ np.diag(multiples.astype(float))
                if _frame_matches(entry, operations, P0, zero, atol):
                    return P0, zero

    # rule 2: the images of spglib's frame under the Euclidean normalizer
    lattice_rotations = _lattice_rotations(P, lattice, entry.centering, symprec)
    candidates = [
        (W @ P, zero) for W in lattice_rotations
        if _frame_matches(entry, operations, W @ P, zero, atol)
    ]
    if not candidates:
        # no frame keeps the cell's own origin; a rotation of the group
        # itself gives an equivalent frame, so one W per coset is enough
        group = {
            tuple(int(v) for v in np.asarray(R).ravel()) for R in entry.rotations
        }
        cosets = []
        for W in lattice_rotations:
            if any(
                tuple(int(v) for v in np.rint(W @ np.linalg.inv(V)).ravel()) in group
                for V in cosets
            ):
                continue
            shifts = _normalizer_translations(sgnum, W)
            if len(shifts) == 0:
                continue
            cosets.append(W)
            candidates.extend((W @ P, W @ origin + shift) for shift in shifts)
    polar = _polar_axes(entry)
    if polar.any():
        # the free origin along the polar axes: on an atom of the species
        # with the smallest type number (the atomic number in the CrystOD
        # commands; any of its copies under the centring translations)
        numbers = np.asarray(numbers)
        lightest = np.asarray(positions, dtype=float)[numbers == numbers.min()]
        centerings = [np.zeros(3)] + [
            np.array(v) for v in _CENTERING_TRANSLATIONS[entry.centering]
        ]
        anchored = []
        for P_c, origin_c in candidates:
            seen = set()  # the atoms of a supercell repeat the same positions
            for x in lightest @ P_c.T + origin_c:
                for c in centerings:
                    offset = (x + c)[polar]
                    reduced = np.round(offset - np.floor(offset + 1e-7), 6)
                    key = tuple(np.where(reduced >= 1.0, 0.0, reduced))
                    if key in seen:
                        continue
                    seen.add(key)
                    shifted = np.array(origin_c, dtype=float)
                    shifted[polar] -= offset
                    anchored.append((P_c, shifted))
        candidates = anchored
    if candidates:
        P_c, origin_c = candidates[_smallest_structure(
            candidates, positions, numbers, entry.centering, float(symprec)
        )]
        if _frame_matches(entry, operations, P_c, origin_c, atol):
            return P_c, origin_c - np.floor(origin_c + 1e-9)
    return P, origin


def _affine_map_between_cells(input_cell, work_cell, symprec: float):
    """``(A, b)`` with ``x_input = A x_work + b`` for two cells of one
    structure, or None.

    Two relations are recognized: the working cell shares the Cartesian
    frame and origin of the input cell (phonopy's primitive cell), or it is
    the spglib standardization of the input cell (idealized frame, origin
    shift).  The map is accepted only when it sends every atom of the working
    cell onto an atom of the same species of the input cell."""
    import spglib

    L_in, X_in, Z_in = (np.asarray(a) for a in input_cell)
    L_w, X_w, Z_w = (np.asarray(a) for a in work_cell)
    L_in = L_in.astype(float)
    L_w = L_w.astype(float)
    tolerance = max(1e-3, 10.0 * float(symprec))  # Angstrom

    def maps_onto(A, b) -> bool:
        if not np.isfinite(A).all() or abs(np.linalg.det(A)) < 1e-8:
            return False
        for x, z in zip(X_w, Z_w):
            image = A @ np.asarray(x, dtype=float) + b
            same = X_in[Z_in == z]
            if len(same) == 0:
                return False
            d = image - same
            d = d - np.rint(d)
            if np.min(np.linalg.norm(d @ L_in, axis=1)) > tolerance:
                return False
        return True

    candidates = [((L_w @ np.linalg.inv(L_in)).T, np.zeros(3))]
    try:
        dataset = spglib.get_symmetry_dataset(input_cell, symprec=symprec)
        T = np.asarray(_dataset_item(dataset, "transformation_matrix"), dtype=float)
        shift = np.asarray(_dataset_item(dataset, "origin_shift"), dtype=float)
        L_std = np.asarray(_dataset_item(dataset, "std_lattice"), dtype=float)
        T_inv = np.linalg.inv(T)
        candidates.append((T_inv @ (L_w @ np.linalg.inv(L_std)).T, -T_inv @ shift))
    except Exception:
        pass
    for A, b in candidates:
        if maps_onto(A, b):
            return A, b
    return None


def iso_frame_for_derived_cell(sgnum: int, input_cell, work_cell,
                               symprec: float = 1e-5):
    """ISO-IR frame of a working cell inherited from its input cell (rule 3).

    Returns ``(P, origin_shift)`` for the working cell, or ``None`` when the
    relation between the two cells cannot be established or the inherited
    frame does not reproduce the tabulated operations.
    """
    frame = canonical_iso_frame(sgnum, input_cell, symprec)
    relation = _affine_map_between_cells(input_cell, work_cell, symprec)
    if frame is None or relation is None:
        return None
    P_in, origin_in = frame
    A, b = relation
    P = P_in @ A
    origin = P_in @ b + origin_in
    operations = _cell_operations(sgnum, work_cell, symprec)
    if operations is None or not _frame_matches(
        _iso_table_entry(sgnum), operations, P, origin,
        max(1e-4, 10.0 * float(symprec)),
    ):
        return None
    return P, origin


def _cell_key(cell) -> tuple:
    lattice, positions, numbers = cell
    return (
        np.asarray(lattice, dtype=float).round(10).tobytes(),
        np.asarray(positions, dtype=float).round(10).tobytes(),
        np.asarray(numbers, dtype=int).tobytes(),
    )


def get_cached_labeler(sgnum: int, cell, symprec: float = 1e-5, input_cell=None):
    """Cached IsoIRLabeler for a primitive cell, or None when unavailable.

    ``cell`` is a spglib tuple (lattice, scaled_positions, numbers) of the
    primitive cell whose symmetry operations feed spgrep.  ``input_cell`` is
    the cell the user gave when ``cell`` was derived from it (primitive
    reduction or standardization): the labels then refer to the ISO-IR frame
    of the input cell, so every command names an irrep of one structure the
    same way whatever cell it works on.  Returns None when the ISO-IR data
    files are missing or standardization fails, so callers can fall through
    to their existing generic labels.
    """
    key = (int(sgnum), float(symprec)) + _cell_key(cell)
    if input_cell is not None:
        key += _cell_key(input_cell)
    if key not in _LABELER_CACHE:
        try:
            frame = None
            if input_cell is not None:
                frame = iso_frame_for_derived_cell(sgnum, input_cell, cell, symprec)
            if frame is None:
                frame = canonical_iso_frame(sgnum, cell, symprec)
            if frame is None:
                labeler = IsoIRLabeler(sgnum, cell=cell, symprec=symprec)
            else:
                labeler = IsoIRLabeler(
                    sgnum, transformation_matrix=frame[0], origin_shift=frame[1]
                )
            _LABELER_CACHE[key] = labeler
        except Exception:
            _LABELER_CACHE[key] = None
    return _LABELER_CACHE[key]


def get_isoir_label_map(
    sgnum: int,
    cell,
    symprec: float,
    kpoint,
    little_rotations,
    little_translations,
    spgrep_characters,
    input_cell=None,
    minus_k: bool = True,
) -> Optional[tuple[dict[int, str], str]]:
    """Label spgrep small irreps at a k point with ISO-IR labels.

    One-stop entry point shared by every labeling path of CrystOD (crystal
    orbitals, orbital hybridization, spin bases, phonons): builds (and
    caches) the ``IsoIRLabeler`` of the cell and calls its
    ``label_characters``.  Never raises: any failure yields ``None`` so
    that callers can fall back to generic labels.

    Args:
        sgnum: Space-group number (1-230).
        cell: Primitive cell as an spglib tuple ``(lattice,
            scaled_positions, numbers)`` whose operations feed spgrep.
        symprec: Symmetry tolerance for spglib.
        kpoint: k vector in the primitive reciprocal basis.
        little_rotations: Rotations of the little group of k in the
            primitive basis.
        little_translations: Their fractional translations.
        spgrep_characters: One character vector per spgrep irrep, aligned
            with the little-group operations (phase convention
            ``exp(-2 pi i k.t)``).
        input_cell: The cell the user gave, when ``cell`` was derived from
            it; the labels then refer to the ISO-IR frame of that cell (see
            ``get_cached_labeler``).
        minus_k: Label a k point without a tabulated star through its -k
            partner, with ISOTROPY's partner names (see
            ``IsoIRLabeler.label_characters``).

    Returns:
        ``({spgrep irrep index: label}, k-type label)`` such as
        ``({0: "Q1"}, "Q")``, or ``None`` when the ISO-IR data are
        unavailable or no consistent assignment exists.
    """
    labeler = get_cached_labeler(sgnum, cell, symprec, input_cell)
    if labeler is None:
        return None
    try:
        return labeler.label_characters(
            kpoint, little_rotations, little_translations, spgrep_characters,
            minus_k=minus_k,
        )
    except Exception:
        return None


def get_isoir_kpoint_name(sgnum: int, cell, symprec: float, kpoint,
                          input_cell=None, minus_k: bool = True) -> Optional[str]:
    """Most specific ISO-IR k-vector type label for a k point, or None.

    With ``minus_k``, a point whose -k partner is the tabulated one is named
    by ISOTROPY's partner name (``"PA"``; ``minus_k_type``)."""
    labeler = get_cached_labeler(sgnum, cell, symprec, input_cell)
    if labeler is None:
        return None
    try:
        return labeler.kpoint_name(kpoint, minus_k=minus_k)
    except Exception:
        return None


def special_points_in_frame(names, kpoints, labeler, canonical=None):
    """The tabulated special points of a list, named in the frame of the
    labels.

    The lists of special points are built from the tables through spglib's
    standardized basis, while the labels refer to the ISO-IR frame of the
    input cell (``get_cached_labeler``).  The two differ by an element of
    the normalizer when the input is not in the ISO-IR setting, and that
    element can turn the tabulated point H into the -k partner of H or into
    another tabulated point.  This returns the same list with every point
    named by the labeller, and with a point that is the 'A' partner of a
    tabulated type replaced by the tabulated point itself (-k), so that the
    names in the list are the letters of the labels printed at each point.

    Args:
        names: k-point names of the list.
        kpoints: Their coordinates in the primitive reciprocal basis.
        labeler: The ``IsoIRLabeler`` of the cell, or ``None``.
        canonical: Function applied to the coordinates of a replaced point
            (the snapping convention of the caller); default: none.

    Returns:
        ``(names, kpoints)``; the input lists themselves when there is no
        labeller or the renaming is not one-to-one.
    """
    if labeler is None:
        return names, kpoints
    tabulated = list(names)
    new_names, new_points = [], []
    for name, k in zip(names, kpoints):
        try:
            found = labeler.kpoint_name(k)
        except Exception:
            found = None
        point = k
        if found is None:
            found = name
        elif found not in tabulated:
            partners = [
                t for t in tabulated if minus_k_label(t, labeler.sgnum) == found
            ]
            if not partners:
                return names, kpoints
            found = partners[0]
            point = [-float(v) + 0.0 for v in k]
            if canonical is not None:
                point = canonical(point)
        new_names.append(found)
        new_points.append(point)
    if len(set(new_names)) != len(new_names):
        return names, kpoints
    return new_names, new_points


def get_isoir_band_decomposition(
    sgnum: int,
    cell,
    symprec: float,
    kpoint,
    little_rotations,
    little_translations,
    reducible_characters,
    input_cell=None,
) -> Optional[tuple[list[tuple[str, int, int]], str]]:
    """Decompose one reducible character vector into ISO-IR irreps.

    The labelling route of phonopy band sets (whose characters can be
    reducible under accidental degeneracy); a k point tabulated only through
    its -k partner gets 'A' labels, as in ``label_characters``.  Returns
    ([(label, multiplicity, dim), ...], k-type label) or None.  Never raises.
    """
    labeler = get_cached_labeler(sgnum, cell, symprec, input_cell)
    if labeler is None:
        return None
    try:
        return labeler.decompose_characters(
            kpoint, little_rotations, little_translations, reducible_characters
        )
    except Exception:
        return None


def get_isoir_band_decompositions(
    sgnum: int,
    cell,
    symprec: float,
    kpoint,
    little_rotations,
    little_translations,
    character_vectors,
    input_cell=None,
) -> list[Optional[tuple[list[tuple[str, int, int]], str]]]:
    """Decompose several reducible character vectors at one k point.

    Batch version of get_isoir_band_decomposition: the candidate-family
    search is done once for all vectors (phonopy band sets share the q);
    'A' labels at a k point tabulated only through -k as there.  Returns
    one entry per vector.  Never raises.
    """
    labeler = get_cached_labeler(sgnum, cell, symprec, input_cell)
    if labeler is None:
        return [None] * len(character_vectors)
    try:
        return labeler.decompose_characters_many(
            kpoint, little_rotations, little_translations, character_vectors
        )
    except Exception:
        return [None] * len(character_vectors)
