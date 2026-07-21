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

Used as a labeling fallback for k points that are absent from the
Bilbao-convention `irreptables` character tables (which contain only the
maximal k points).  The resulting labels follow the Miller-Love /
ISOTROPY convention (e.g. T1..T5, DT5, LD3, GP1).

Data location: the directory holding ``CIR_data/CIR_data.txt`` is looked
up from the environment variable ``CRYSTOD_ISOIR_PATH`` first, then from
``<repository root>/ISOTROPY`` next to this package.

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

    def match_k(self, k, atol: float = 1e-6) -> Optional[tuple[int, np.ndarray]]:
        """Find (arm, params) with arm_k(arm, params) == k modulo the
        reciprocal lattice of the (possibly centered) crystal lattice.

        `k` is in conventional fractional reciprocal coordinates.  Offsets
        G are tried in order of increasing norm so that, when possible,
        the parameters describe `k` itself rather than a translated copy
        (the small-irrep phases are only correct for the untranslated
        parametrization in non-symmorphic groups).
        """
        k = np.asarray(k, dtype=float)
        offsets = sorted(
            (
                (gx, gy, gz)
                for gx in range(-2, 3)
                for gy in range(-2, 3)
                for gz in range(-2, 3)
            ),
            key=lambda g: abs(g[0]) + abs(g[1]) + abs(g[2]),
        )
        for arm in range(self.narms):
            kv = self.kvecs[arm]
            const = np.array([kv[0][j] / kv[0][3] for j in range(3)])
            cols = []
            for p in range(3):
                c = kv[p + 1]
                if c[3] != 0 and any(c[j] != 0 for j in range(3)):
                    cols.append((p, np.array([c[j] / c[3] for j in range(3)])))
            A = np.column_stack([c for _, c in cols]) if cols else None
            for gx, gy, gz in offsets:
                G = np.array([gx, gy, gz], dtype=float)
                if not _is_reciprocal_lattice_vector(G, self.centering):
                    continue
                rhs = k + G - const
                if A is None:
                    if np.allclose(rhs, 0, atol=atol):
                        return arm, np.zeros(3)
                    continue
                sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
                if np.allclose(A @ sol - rhs, 0, atol=atol):
                    params = np.zeros(3)
                    for (p, _), v in zip(cols, sol):
                        params[p] = v
                    return arm, params
        return None

    def find_operator(self, rotation) -> Optional[int]:
        for i in range(self.opcount):
            if np.array_equal(self.rotations[i], rotation):
                return i
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
    """Path of the (possibly gzip-compressed) data file of one kind."""
    base = data_dir / f"{kind.upper()}_data" / f"{kind.upper()}_data.txt"
    for path in (base, base.parent / (base.name + ".gz")):
        if path.is_file():
            return path
    return None


def find_isoir_data_dir() -> Optional[Path]:
    """Locate the ISO-IR data directory (containing CIR_data/, PIR_data/)."""
    env = os.environ.get("CRYSTOD_ISOIR_PATH")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.append(Path(__file__).resolve().parent.parent / "ISOTROPY")
    for cand in candidates:
        if _isoir_data_file(cand, "cir") is not None:
            return cand
    return None


_CACHE: dict[tuple[str, int, str], list[IsoIrrep]] = {}


def load_isoir_irreps(sgnum: int, kind: str = "cir",
                      data_dir: Optional[Path] = None) -> list[IsoIrrep]:
    """Parse all irreps of one space group from an ISO-IR data file.

    kind: 'cir' (complex irreps) or 'pir' (physically irreducible irreps).
    Results are cached per (file, space group).
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

    Parameters
    ----------
    sgnum:
        Space group number (1-230).
    cell:
        Primitive cell (lattice, scaled_positions, numbers) whose symmetry
        operations are used by spgrep.  The transformation into the ISO-IR
        standard setting (origin choice 2 etc.) is computed with spglib
        using the Hall number of that setting.
    transformation_matrix, origin_shift:
        Alternatively, an explicit spglib-style transformation into the
        ISO-IR setting (x_conventional = P x_primitive + origin_shift).
    """

    def __init__(self, sgnum: int, transformation_matrix=None,
                 origin_shift=None, cell=None, symprec: float = 1e-5):
        self.sgnum = sgnum
        if cell is not None:
            import spglib

            dataset = spglib.get_symmetry_dataset(
                cell, symprec=symprec, hall_number=iso_hall_number(sgnum)
            )
            number = (
                dataset['number'] if isinstance(dataset, dict)
                else dataset.number
            )
            if dataset is None or number != sgnum:
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
        return np.asarray(k_primitive, dtype=float) @ self.Pinv

    def conventional_operations(self, rotations, translations):
        """Map primitive-basis operations into the conventional setting."""
        conv = []
        for R_p, t_p in zip(rotations, translations):
            R_c = np.rint(self.P @ R_p @ self.Pinv).astype(int)
            t_c = self.P @ np.asarray(t_p, dtype=float) + (
                np.eye(3) - R_c
            ) @ self.origin_shift
            conv.append((R_c, t_c))
        return conv

    def kpoint_name(self, k_primitive) -> Optional[str]:
        """Most specific ISO-IR k-vector type label containing this k
        (fewest free parameters), e.g. 'T' for (1/2, 1/2, 0.4) in Pm-3m.
        """
        k_conv = self.conventional_k(k_primitive)
        best = None
        for ir in self.irreps:
            if ir.match_k(k_conv) is not None:
                if best is None or ir.num_free_params < best.num_free_params:
                    best = ir
        return best.ktype if best is not None else None

    # -- labeling ------------------------------------------------------------
    def label_characters(
        self,
        k_primitive,
        little_rotations,
        little_translations,
        spgrep_characters,
        atol: float = 1e-5,
    ) -> Optional[tuple[dict[int, str], str]]:
        """Match spgrep small-irrep characters against ISO-IR.

        spgrep_characters: list over spgrep irreps of character vectors
        aligned with (little_rotations, little_translations), computed with
        the spgrep phase convention exp(-2*pi*i k.t).  They are compared
        with the complex conjugate of the ISO-IR characters.

        Returns ({spgrep irrep index: ISO-IR label}, k-type label) or None.
        """
        k_conv = self.conventional_k(k_primitive)
        conv_ops = self.conventional_operations(
            little_rotations, little_translations
        )

        # candidate irreps whose star contains this k, grouped by k-type,
        # most specific k type (fewest free parameters) first
        families: dict[str, list[tuple[IsoIrrep, int, np.ndarray]]] = {}
        for ir in self.irreps:
            matched = ir.match_k(k_conv)
            if matched is not None:
                families.setdefault(ir.ktype, []).append(
                    (ir, matched[0], matched[1])
                )
        for ktype in sorted(
            families, key=lambda t: families[t][0][0].num_free_params
        ):
            result = self._try_family(
                families[ktype], conv_ops, spgrep_characters, atol
            )
            if result is not None:
                return result, ktype
        return None

    def _try_family(self, family, conv_ops, spgrep_characters, atol):
        ref = family[0][0]
        # all little-group operations must belong to the family's little group
        for ir, arm, params in family:
            for R_c, _ in conv_ops:
                if not ir.in_little_group(R_c, arm, params):
                    return None
        # the family must exhaust the little group
        if sum(ir.small_dim**2 for ir, _, _ in family) != len(conv_ops):
            return None

        try:
            iso_chars = self._family_characters(family, conv_ops)
        except LookupError:
            # setting mismatch (operator or translation not found): the
            # transformation into the ISO-IR setting is wrong -- fail safe
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
