"""Magnetic space groups, symmetry-allowed responses and collinear classes.

This module adds three analyses to the spin bases of ``crystod-mag``:

- :func:`magnetic_space_group`: the magnetic space group (MSG) of a spin
  arrangement from spglib (``get_magnetic_symmetry_dataset`` with axial
  moments), with its BNS number and symbol, its OG number, its type (I-IV),
  the magnetic point group (MPG) derived from the BNS symbol and checked
  against the operations, and the operations themselves;
- :func:`macroscopic_responses`: the macroscopic responses the magnetic point
  group allows by symmetry: a net magnetization (time-odd axial vector,
  Jahn ``aeV``), the linear magnetoelectric tensor (``aeV2``) and the
  piezomagnetic tensor (``aeV[V2]``), through
  :func:`crystod.tensor_form.tensor_form_of_operations` with the
  time-reversal flag of every operation;
- :func:`classify_collinear`: the nonrelativistic class of a collinear
  arrangement from the sign pattern of its moments alone: ferromagnet,
  ferrimagnet, conventional antiferromagnet (a translation or the inversion
  exchanges the two sublattices), altermagnet (only rotations, proper or
  improper, exchange them), with the d-, g- or i-wave type, the lowest-order
  form of the spin splitting allowed by symmetry and the spin-degenerate
  planes and lines of an
  altermagnet (L. Smejkal, J. Sinova and T. Jungwirth, Phys. Rev. X 12,
  031042 (2022)).

The magnetic space group uses the moment directions (relativistic: spin and
lattice rotate together); the collinear class uses only the signs of the
moments along their common axis (spin and lattice decoupled). The two can
therefore differ in how they treat two arrangements: the moments along z
and along x of a rutile altermagnet have different magnetic space groups
and the same class.

Example:
    >>> import numpy as np
    >>> from crystod import mag
    >>> lattice = np.eye(3) * 2.87                      # bcc Fe cell, AFM pair
    >>> positions = [[0, 0, 0], [0.5, 0.5, 0.5]]
    >>> group = mag.magnetic_space_group(lattice, positions, [26, 26],
    ...                                  [[0, 0, 1], [0, 0, -1]])
    >>> group.bns_symbol, group.bns_number, group.type_name
    ('P_I4/mnc', '128.410', 'IV')
    >>> group.magnetic_point_group
    "4/mmm1'"
"""

from __future__ import annotations

import contextlib
import functools
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

#: Roman numerals of the magnetic space-group types.
MSG_TYPE_NAMES = {1: "I", 2: "II", 3: "III", 4: "IV"}

#: Reference of the altermagnet classification.
ALTERMAGNET_REFERENCE = (
    "L. Smejkal, J. Sinova and T. Jungwirth, Phys. Rev. X 12, 031042 (2022)"
)

_WAVE_NAMES = {2: "d-wave", 4: "g-wave", 6: "i-wave"}

CLASS_FERROMAGNET = "ferromagnet"
CLASS_FERRIMAGNET = "ferrimagnet"
CLASS_COMPENSATED = "compensated, sublattices not related by symmetry"
CLASS_AFM_TRANSLATION = (
    "antiferromagnet (conventional: a translation exchanges the sublattices, T tau)"
)
CLASS_AFM_INVERSION = (
    "antiferromagnet (conventional: inversion exchanges the sublattices, P T)"
)
CLASS_ALTERMAGNET = "altermagnet"
CLASS_NONCOLLINEAR = "noncollinear (not classified)"

_SHORT_CLASS = {
    CLASS_AFM_TRANSLATION: "antiferromagnet (T tau)",
    CLASS_AFM_INVERSION: "antiferromagnet (P T)",
    CLASS_COMPENSATED: "compensated (unrelated sublattices)",
    CLASS_NONCOLLINEAR: "noncollinear",
}


# ------------------------------------------------------------ dataclasses


@dataclass(frozen=True, eq=False)
class MagneticGroupInfo:
    """The magnetic space group of a spin arrangement.

    Attributes:
        uni_number: spglib's serial number of the MSG (1-1651).
        bns_number: BNS number, ``"136.499"``.
        bns_symbol: BNS symbol from pymatgen, ``"P4_2'/mnm'"`` (a type IV
            symbol carries the lattice subscript, ``"P_I4/mnc"``).
        og_number: Opechowski-Guccione number, ``"136.5.1161"``.
        msg_type: 1-4 (type I: no operation with time reversal; II: grey
            group; III: half of the operations with time reversal, no pure
            antitranslation; IV: black-white lattice).
        parent_number: The space-group number of the type that spglib
            reports (for types I, II and III the space group of the
            unprimed and primed operations together; for type IV the family
            space group of the BNS symbol).
        magnetic_point_group: The MPG symbol derived from the BNS symbol;
            followed by ``"?"`` when it disagrees with the operations.
            ``uni_number``, ``bns_number``, ``bns_symbol``, ``og_number``,
            ``msg_type``, ``parent_number`` and ``magnetic_point_group`` are
            None when spglib finds the operations but cannot identify the
            group (:attr:`identified` False).
        point_group_checked: True when the derived symbol generates the same
            rotations, with the same time-reversal flags, as the operations.
        n_operations: Number of operations in the cell given (centring and
            magnetic-cell translations counted).
        n_primed: Number of those operations combined with time reversal.
        rotations: Cartesian rotation matrices, ``(n_operations, 3, 3)``, in
            the frame of the lattice given.
        translations: Fractional translations in the cell given.
        time_reversals: Boolean time-reversal flag of every operation.
    """

    uni_number: int | None
    bns_number: str | None
    bns_symbol: str | None
    og_number: str | None
    msg_type: int | None
    parent_number: int | None
    magnetic_point_group: str | None
    point_group_checked: bool
    n_operations: int
    n_primed: int
    rotations: np.ndarray
    translations: np.ndarray
    time_reversals: np.ndarray

    @property
    def identified(self) -> bool:
        """False when spglib returned the operations but could not name the
        group (the identification fields are then None)."""
        return self.uni_number is not None

    @property
    def type_name(self) -> str:
        """The type as a Roman numeral (``"III"``), ``"-"`` when the group
        is not identified."""
        if self.msg_type is None:
            return "-"
        return MSG_TYPE_NAMES.get(self.msg_type, str(self.msg_type))

    @property
    def n_point_operations(self) -> int:
        """Order of the magnetic point group: the distinct (rotation,
        time-reversal) pairs of the operations."""
        return len(_unique_operations(self.rotations, self.time_reversals)[1])

    @property
    def n_point_primed(self) -> int:
        """Primed elements of the magnetic point group."""
        return int(sum(_unique_operations(self.rotations, self.time_reversals)[1]))


@dataclass(frozen=True, eq=False)
class MagneticResponses:
    """Macroscopic responses allowed by a magnetic point group.

    Attributes:
        magnetization: :class:`crystod.tensor_form.TensorForm` of the net
            magnetization (Jahn ``aeV``).
        magnetoelectric: Form of the linear magnetoelectric tensor
            ``P_i = alpha_ij H_j`` (Jahn ``aeV2``).
        piezomagnetic: Form of the piezomagnetic tensor
            ``M_i = Lambda_ijk sigma_jk`` (Jahn ``aeV[V2]``).
        magnetization_direction: ``"none"``, ``"along z"``,
            ``"in the xy plane"``, ``"any direction"`` (directions in the
            Cartesian axes of the cell given).
    """

    magnetization: object
    magnetoelectric: object
    piezomagnetic: object
    magnetization_direction: str

    @property
    def n_magnetization(self) -> int:
        """Independent components of the net magnetization."""
        return self.magnetization.n_independent

    @property
    def n_magnetoelectric(self) -> int:
        """Independent components of the linear magnetoelectric tensor."""
        return self.magnetoelectric.n_independent

    @property
    def n_piezomagnetic(self) -> int:
        """Independent components of the piezomagnetic tensor."""
        return self.piezomagnetic.n_independent

    def summary(self) -> str:
        """One-line summary: ``"net magnetization along z; linear
        magnetoelectric effect none; piezomagnetism 2 components"``."""
        return (f"net magnetization {self.magnetization_direction}; "
                f"linear magnetoelectric effect {_count_text(self.n_magnetoelectric)}; "
                f"piezomagnetism {_count_text(self.n_piezomagnetic)}")


@dataclass(frozen=True, eq=False)
class CollinearClass:
    """Nonrelativistic class of a collinear spin arrangement.

    Attributes:
        name: One of ``"ferromagnet"``, ``"ferrimagnet"``,
            ``"compensated, sublattices not related by symmetry"``,
            ``"antiferromagnet (conventional: a translation exchanges the
            sublattices, T tau)"``, ``"antiferromagnet (conventional:
            inversion exchanges the sublattices, P T)"``, ``"altermagnet"``
            or ``"noncollinear (not classified)"``.
        wave: ``"d-wave"``, ``"g-wave"`` or ``"i-wave"`` for an altermagnet,
            else None.
        order: The even degree ``l`` of the wave (2, 4, 6) or None.
        spin_splitting: The lowest-order polynomials in ``(k_x, k_y, k_z)``
            that symmetry allows for the spin splitting of an altermagnet,
            in the Cartesian axes of the cell given (``("k_x k_y",)``).
        degenerate_planes: Normals of the spin-degenerate planes of an
            altermagnet as Cartesian directions (``("[100]", "[010]")``).
        degenerate_lines: Spin-degenerate lines that do not lie in one of
            the planes, as Cartesian directions.
        axis: The common spin axis (unit Cartesian vector) or None.
        n_preserving: Number of space-group operations (in the magnetic
            cell) that keep the sign pattern.
        n_swapping: Number of operations that reverse it (exchange the
            sublattices).
        net_moment: The net moment over the magnetic cell.
    """

    name: str
    wave: str | None = None
    order: int | None = None
    spin_splitting: tuple = ()
    degenerate_planes: tuple = ()
    degenerate_lines: tuple = ()
    axis: np.ndarray | None = None
    n_preserving: int = 0
    n_swapping: int = 0
    net_moment: np.ndarray | None = None

    @property
    def short_name(self) -> str:
        """Short form for tables: ``"altermagnet (d-wave)"``,
        ``"antiferromagnet (P T)"``, ``"ferromagnet"``."""
        if self.name == CLASS_ALTERMAGNET and self.wave:
            return f"altermagnet ({self.wave})"
        return _SHORT_CLASS.get(self.name, self.name)

    def text(self) -> str:
        """The description printed after ``collinear order:``.

        Example: ``"altermagnet (d-wave; spin splitting allowed by symmetry
        ~ k_x k_y; spin-degenerate planes perpendicular to [100], [010])"``.
        """
        if self.name != CLASS_ALTERMAGNET:
            return self.name
        parts = [self.wave or "wave type not found"]
        if self.spin_splitting:
            parts.append("spin splitting allowed by symmetry ~ "
                         + ", ".join(self.spin_splitting))
        if self.degenerate_planes:
            parts.append("spin-degenerate planes perpendicular to "
                         + ", ".join(self.degenerate_planes))
        if self.degenerate_lines:
            parts.append("spin-degenerate lines along " + ", ".join(self.degenerate_lines))
        return f"altermagnet ({'; '.join(parts)})"


# ------------------------------------------------------------ small helpers


def _count_text(count: int) -> str:
    if count == 0:
        return "none"
    return f"{count} component" + ("s" if count != 1 else "")


def _cartesian(rotation, lattice) -> np.ndarray:
    """Cartesian matrix of an integer rotation acting on fractional
    coordinates of the lattice whose rows are the lattice vectors."""
    lattice_t = np.transpose(np.asarray(lattice, dtype=float))
    return lattice_t @ np.asarray(rotation, dtype=float) @ np.linalg.inv(lattice_t)


def _direction_text(vector) -> str:
    """Cartesian direction as ``[100]``, ``[1-10]`` or ``[0.577 0.577 0.577]``."""
    vector = np.asarray(vector, dtype=float)
    vector = vector / np.max(np.abs(vector))
    for value in vector:          # first nonzero component positive
        if abs(value) > 1e-8:
            if value < 0:
                vector = -vector
            break
    vector[np.abs(vector) < 1e-8] = 0.0
    fractions = [Fraction(float(v)).limit_denominator(12) for v in vector]
    if all(abs(float(f) - v) < 1e-6 for f, v in zip(fractions, vector)):
        denominator = math.lcm(*[f.denominator for f in fractions])
        integers = [int(f * denominator) for f in fractions]
        if all(abs(i) < 10 for i in integers):
            return "[" + "".join(str(i) for i in integers) + "]"
        return "[" + " ".join(str(i) for i in integers) + "]"
    # hexagonal axes: components that are rational multiples of 1 or sqrt(3)
    root3 = math.sqrt(3.0)
    smallest = min(abs(v) for v in vector if abs(v) > 1e-8)
    terms = []
    for value in vector / smallest:
        plain = Fraction(float(value)).limit_denominator(12)
        surd = Fraction(float(value / root3)).limit_denominator(12)
        if abs(float(plain) - value) < 1e-6:
            terms.append((plain, ""))
        elif abs(float(surd) * root3 - value) < 1e-6:
            terms.append((surd, "sqrt3"))
        else:
            terms = []
            break
    if terms:
        denominator = math.lcm(*[f.denominator for f, _ in terms])
        texts = []
        for fraction, unit in terms:
            number = int(fraction * denominator)
            if unit and abs(number) == 1:
                texts.append(("-" if number < 0 else "") + unit)
            else:
                texts.append(f"{number}{unit}")
        return "[" + " ".join(texts) + "]"
    unit = vector / np.linalg.norm(vector)
    return "[" + " ".join(f"{v:.3f}" for v in unit) + "]"


def _axis_name(vector) -> str | None:
    vector = np.asarray(vector, dtype=float)
    vector = vector / np.linalg.norm(vector)
    for index, name in enumerate("xyz"):
        if abs(abs(vector[index]) - 1.0) < 1e-6:
            return name
    return None


def _unique_operations(rotations, time_reversals) -> tuple[np.ndarray, list[bool]]:
    """Distinct (Cartesian rotation, time-reversal flag) pairs."""
    seen: dict = {}
    for rotation, flag in zip(np.asarray(rotations, dtype=float), time_reversals):
        key = (tuple(np.round(rotation, 6).ravel()), bool(flag))
        seen.setdefault(key, rotation)
    keys = list(seen)
    return np.array([seen[k] for k in keys]), [k[1] for k in keys]


# ------------------------------------------------------------ magnetic point group


def _strip_lattice(symbol: str) -> str:
    body = symbol[1:]
    if body.startswith("_"):
        body = body[2:]           # BNS lattice subscripts are one letter
    return body


def magnetic_point_group_symbol(bns_symbol: str, msg_type: int) -> str:
    """Magnetic point-group symbol derived from a BNS space-group symbol.

    The lattice letter and its subscript are dropped, glide letters
    (``a, b, c, n, d, e``) become ``m``, screw axes become plain rotations
    (``4_2 -> 4``) and every prime is kept; a type II or type IV group gives
    the non-magnetic point group followed by ``1'``.

    Args:
        bns_symbol: BNS symbol in pymatgen's spelling (``"P4_2'/mnm'"``,
            ``"P_I4/mnc"``, ``"Pm-3m1'"``).
        msg_type: The type (1-4).

    Returns:
        The MPG symbol, e.g. ``"4'/mmm'"`` or ``"4/mmm1'"``.

    Example:
        >>> from crystod.magnetic_group import magnetic_point_group_symbol
        >>> magnetic_point_group_symbol("Pn'ma'", 3)
        "m'mm'"
    """
    body = _strip_lattice(bns_symbol)
    body = re.sub(r"([1-6])_[1-5]", r"\1", body)
    body = re.sub(r"[abcnde]", "m", body)
    if msg_type == 2 and body.endswith("1'"):
        body = body[:-2]
    if msg_type in (2, 4):
        body = body.replace("'", "") + "1'"
    return body


_TOKEN = re.compile(r"-?[1-6]'?(?:/m'?)?|m'?")


def _rotation_matrix(axis, order: int) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    theta = 2.0 * np.pi / order
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * (k @ k)


def _mirror(normal) -> np.ndarray:
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    return np.eye(3) - 2.0 * np.outer(normal, normal)


def _symbol_generators(symbol: str) -> list[tuple[np.ndarray, bool]] | None:
    """Generators (Cartesian matrix, primed) of an MPG symbol, or None when
    the symbol cannot be parsed."""
    body, grey = symbol, False
    if body.endswith("1'") and len(body) > 2 and body[-3] != "-":
        # a trailing 1' marks a grey group (types II and IV); -1' is not one
        body, grey = body[:-2], True
    tokens = []
    position = 0
    while position < len(body):
        match = _TOKEN.match(body, position)
        if match is None:
            return None
        tokens.append(match.group(0))
        position = match.end()
    if not tokens:
        return None

    def rotation_order(token: str) -> int:
        digits = re.match(r"-?([1-6])", token)
        return int(digits.group(1)) if digits else 0

    z, x, y = (0, 0, 1), (1, 0, 0), (0, 1, 0)
    first = rotation_order(tokens[0])
    if len(tokens) >= 2 and rotation_order(tokens[1]) == 3:
        axes = [z, (1, 1, 1), (1, -1, 0)][: len(tokens)]   # cubic
    elif first in (3, 6):
        axes = [z, x, y][: len(tokens)]                    # hexagonal axes
    elif first == 4:
        axes = [z, x, (1, 1, 0)][: len(tokens)]
    elif len(tokens) == 1:
        axes = [z]
    elif len(tokens) == 3:
        axes = [x, y, z]
    else:
        return None
    generators = []
    for token, axis in zip(tokens, axes):
        if token.startswith("m"):
            generators.append((_mirror(axis), token.endswith("'")))
            continue
        rotation_part, _, mirror_part = token.partition("/")
        order = rotation_order(rotation_part)
        matrix = _rotation_matrix(axis, order)
        if rotation_part.startswith("-"):
            matrix = -matrix
        generators.append((matrix, rotation_part.endswith("'")))
        if mirror_part:
            generators.append((_mirror(axis), mirror_part.endswith("'")))
    if grey:
        generators.append((np.eye(3), True))
    return generators


def _closure(generators) -> list[tuple[np.ndarray, bool]] | None:
    elements = {(tuple(np.round(np.eye(3), 6).ravel()), False): np.eye(3)}
    frontier = [(np.eye(3), False)]
    while frontier:
        new = []
        for matrix, flag in frontier:
            for generator, gflag in generators:
                product = matrix @ generator
                pflag = flag ^ gflag
                key = (tuple(np.round(product, 6).ravel()), pflag)
                if key not in elements:
                    elements[key] = product
                    new.append((product, pflag))
                    if len(elements) > 96:
                        return None
        frontier = new
    return [(matrix, key[1]) for key, matrix in elements.items()]


def _signature(pairs) -> Counter:
    """Frame-independent multiset of (det, trace, primed) over (R, flag)."""
    return Counter(
        (int(round(float(np.linalg.det(r)))), int(round(float(np.trace(r)))), bool(f))
        for r, f in pairs
    )


def check_point_group_symbol(symbol: str, rotations, time_reversals) -> bool:
    """Whether an MPG symbol matches the rotation parts of the operations.

    The group generated from the symbol (standard axes per crystal system)
    and the distinct (rotation, time-reversal) pairs of the operations must
    have the same order and the same multiset of (det, trace, primed); a
    group in which a rotation occurs with both flags (a grey group) needs
    the trailing ``1'``.

    Args:
        symbol: MPG symbol (``"4'/mmm'"``).
        rotations: Rotation matrices of the operations (integer or
            Cartesian: determinant and trace do not depend on the frame).
        time_reversals: Time-reversal flags of the operations.

    Returns:
        True when they match.
    """
    generators = _symbol_generators(symbol)
    if generators is None:
        return False
    group = _closure(generators)
    if group is None:
        return False
    rotations = np.asarray(rotations, dtype=float)
    seen: dict = {}
    for rotation, flag in zip(rotations, time_reversals):
        seen.setdefault((tuple(np.round(rotation, 6).ravel()), bool(flag)), rotation)
    pairs = [(rotation, key[1]) for key, rotation in seen.items()]
    return _signature(group) == _signature(pairs)


@functools.lru_cache(maxsize=None)
def bns_symbol(bns_number: str) -> str:
    """BNS symbol of a BNS number from pymatgen's table.

    Args:
        bns_number: ``"136.499"``.

    Returns:
        ``"P4_2'/mnm'"``.
    """
    from pymatgen.symmetry.maggroups import MagneticSpaceGroup

    number, minor = (int(part) for part in bns_number.split("."))
    return str(MagneticSpaceGroup([number, minor]).sg_symbol)


def magnetic_space_group(lattice, positions, numbers, magmoms, symprec: float = 1e-5,
                         mag_symprec: float | None = None) -> MagneticGroupInfo:
    """The magnetic space group of a spin arrangement (spglib).

    Args:
        lattice: Lattice vectors as rows, ``(3, 3)`` in Angstrom.
        positions: Fractional coordinates, ``(n_atoms, 3)``.
        numbers: Atomic numbers (or any type labels), ``(n_atoms,)``.
        magmoms: Moments as Cartesian axial vectors, ``(n_atoms, 3)``
            (zero on the non-magnetic atoms).
        symprec: Distance tolerance of spglib.
        mag_symprec: Tolerance on the moments (spglib's ``mag_symprec``;
            None uses ``symprec``).

    Returns:
        A :class:`MagneticGroupInfo`.

    Raises:
        SystemExit: spglib finds no magnetic symmetry operations at all
            (``ValueError`` through ``crystod.mag``).

    Note:
        spglib's identification of the group sometimes fails for one
        choice of the cell and succeeds for an equivalent one (the bcc
        H-point arrangement with the moments along x). The cell is then
        re-expressed by its Niggli-reduced, its Delaunay-reduced and a few
        sheared bases (the same lattice and the same atoms) until one is
        identified. When none is, the operations of
        ``spglib.get_magnetic_symmetry`` are returned with the
        identification fields set to None (:attr:`MagneticGroupInfo.identified`
        False), so that :func:`macroscopic_responses` still applies.

    Example:
        >>> import numpy as np
        >>> from crystod import mag
        >>> group = mag.magnetic_space_group(np.eye(3) * 2.87,
        ...     [[0, 0, 0], [0.5, 0.5, 0.5]], [26, 26], [[0, 0, 1], [0, 0, -1]])
        >>> group.uni_number, group.msg_type
        (1070, 4)
    """
    import spglib

    lattice = np.asarray(lattice, dtype=float)
    positions = np.asarray(positions, dtype=float)
    magmoms = np.asarray(magmoms, dtype=float)
    numbers = list(numbers)
    tolerance = -1.0 if mag_symprec is None else float(mag_symprec)
    if not (np.all(np.isfinite(magmoms)) and np.all(np.isfinite(positions))):
        raise SystemExit("ERROR: the spin arrangement holds non-finite values; "
                         "no magnetic space group.")

    dataset, used_lattice = None, lattice
    with _quiet_spglib():
        for candidate in _equivalent_lattices(lattice):
            moved = positions @ lattice @ np.linalg.inv(candidate)
            cell = (candidate, moved - np.floor(moved), numbers, magmoms)
            try:
                dataset = spglib.get_magnetic_symmetry_dataset(
                    cell, is_axial=True, symprec=symprec, mag_symprec=tolerance)
            except Exception:  # noqa: BLE001 - spglib reports a failed search as its own error type
                dataset = None
            if dataset is not None:
                used_lattice = candidate
                break
        if dataset is None:
            try:
                operations = spglib.get_magnetic_symmetry(
                    (lattice, positions, numbers, magmoms), is_axial=True,
                    symprec=symprec, mag_symprec=tolerance)
            except Exception:  # noqa: BLE001
                operations = None
    if dataset is None:
        if operations is None or len(operations["rotations"]) == 0:
            raise SystemExit("ERROR: spglib found no magnetic space group for this spin "
                             "arrangement; try another --tolerance.")
        flags = np.asarray(operations["time_reversals"], dtype=bool)
        return MagneticGroupInfo(
            uni_number=None, bns_number=None, bns_symbol=None, og_number=None,
            msg_type=None, parent_number=None, magnetic_point_group=None,
            point_group_checked=False,
            n_operations=len(flags),
            n_primed=int(flags.sum()),
            rotations=np.array([_cartesian(r, lattice) for r in operations["rotations"]]),
            translations=np.asarray(operations["translations"], dtype=float),
            time_reversals=flags,
        )
    group_type = spglib.get_magnetic_spacegroup_type(dataset.uni_number)
    symbol = bns_symbol(group_type.bns_number)
    integer_rotations = np.asarray(dataset.rotations)
    flags = np.asarray(dataset.time_reversals, dtype=bool)
    point_group = magnetic_point_group_symbol(symbol, int(group_type.type))
    checked = check_point_group_symbol(point_group, integer_rotations, flags)
    if not checked:
        point_group += "?"
    # translations back in the fractional coordinates of the cell given
    translations = np.asarray(dataset.translations, dtype=float) @ used_lattice @ np.linalg.inv(lattice)
    return MagneticGroupInfo(
        uni_number=int(dataset.uni_number),
        bns_number=str(group_type.bns_number),
        bns_symbol=symbol,
        og_number=str(group_type.og_number),
        msg_type=int(group_type.type),
        parent_number=int(group_type.number),
        magnetic_point_group=point_group,
        point_group_checked=checked,
        n_operations=len(integer_rotations),
        n_primed=int(flags.sum()),
        rotations=np.array([_cartesian(r, used_lattice) for r in integer_rotations]),
        translations=translations,
        time_reversals=flags,
    )


@contextlib.contextmanager
def _quiet_spglib():
    """Silence spglib's C-level warnings (``spglib: Failed to match with UNI
    number!``) for the duration of the block; the previous setting of
    ``SPGLIB_WARNING`` is restored."""
    previous = os.environ.get("SPGLIB_WARNING")
    os.environ["SPGLIB_WARNING"] = "OFF"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("SPGLIB_WARNING", None)
        else:
            os.environ["SPGLIB_WARNING"] = previous


_SHEARS = tuple(
    np.array(matrix, dtype=int) for matrix in (
        [[1, 1, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 1], [0, 0, 1]],
        [[1, 0, 1], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [1, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 1, 1]], [[1, 0, 0], [0, 1, 0], [1, 0, 1]],
    )
)


def _equivalent_lattices(lattice: np.ndarray):
    """The lattice given, then its Niggli- and Delaunay-reduced bases and a
    few sheared bases: the same lattice (unimodular changes of basis)."""
    import spglib

    yield lattice
    seen = [lattice]
    reduced = []
    for reduce in (spglib.niggli_reduce, spglib.delaunay_reduce):
        try:
            candidate = reduce(lattice)
        except Exception:  # noqa: BLE001
            candidate = None
        if candidate is not None:
            reduced.append(np.asarray(candidate, dtype=float))
    for candidate in reduced + [shear @ lattice for shear in _SHEARS]:
        change = candidate @ np.linalg.inv(lattice)
        if not (np.allclose(change, np.rint(change), atol=1e-6)
                and abs(abs(np.linalg.det(change)) - 1.0) < 1e-6):
            continue
        if any(np.allclose(candidate, other, atol=1e-8) for other in seen):
            continue
        seen.append(candidate)
        yield candidate


def format_magnetic_group(group: MagneticGroupInfo) -> str:
    """``"P4_2'/mnm' (BNS 136.499, type III), 16 operations, 8 with time
    reversal"``; ``"not identified by spglib (128 operations, 64 with time
    reversal)"`` when spglib cannot name the group."""
    if not group.identified:
        return (f"not identified by spglib ({group.n_operations} operations, "
                f"{group.n_primed} with time reversal)")
    return (f"{group.bns_symbol} (BNS {group.bns_number}, type {group.type_name}), "
            f"{group.n_operations} operations, {group.n_primed} with time reversal")


def format_magnetic_point_group(group: MagneticGroupInfo) -> str:
    """``"4'/mmm'"``; ``"not identified (32 elements, 16 with time
    reversal)"`` when spglib cannot name the group."""
    if not group.identified:
        return (f"not identified ({group.n_point_operations} elements, "
                f"{group.n_point_primed} with time reversal)")
    return str(group.magnetic_point_group)


# ------------------------------------------------------------ responses


def _magnetization_direction(form) -> str:
    tensors = np.asarray(form.tensors, dtype=float).reshape(-1, 3)
    count = len(tensors)
    if count == 0:
        return "none"
    if count >= 3:
        return "any direction"
    if count == 1:
        vector = tensors[0]
        name = _axis_name(vector)
        return f"along {name}" if name else f"along {_direction_text(vector)}"
    normal = np.cross(tensors[0], tensors[1])
    name = _axis_name(normal)
    if name is not None:
        plane = {"x": "yz", "y": "xz", "z": "xy"}[name]
        return f"in the {plane} plane"
    return f"in the plane perpendicular to {_direction_text(normal)}"


def macroscopic_responses(group_or_rotations, time_reversals=None) -> MagneticResponses:
    """Responses that a magnetic point group allows by symmetry.

    Args:
        group_or_rotations: A :class:`MagneticGroupInfo`, or Cartesian
            rotation matrices ``(n, 3, 3)`` of the operations.
        time_reversals: The time-reversal flags (needed with rotations).

    Returns:
        A :class:`MagneticResponses` with the forms of the net
        magnetization (``aeV``), the linear magnetoelectric tensor
        (``aeV2``) and the piezomagnetic tensor (``aeV[V2]``).

    Raises:
        ValueError: Rotations given without flags.

    Example:
        >>> responses = mag.macroscopic_responses(group)   # see magnetic_space_group
        >>> responses.summary()
        'net magnetization none; linear magnetoelectric effect none; piezomagnetism none'
    """
    from .tensor_form import tensor_form_of_operations

    if isinstance(group_or_rotations, MagneticGroupInfo):
        rotations = group_or_rotations.rotations
        time_reversals = group_or_rotations.time_reversals
    else:
        rotations = group_or_rotations
        if time_reversals is None:
            raise ValueError("time_reversals are needed with explicit rotations")
    unique, flags = _unique_operations(rotations, time_reversals)
    magnetization = tensor_form_of_operations(unique, "aeV", time_reversal=flags)
    magnetoelectric = tensor_form_of_operations(unique, "aeV2", time_reversal=flags)
    piezomagnetic = tensor_form_of_operations(unique, "aeV[V2]", time_reversal=flags)
    return MagneticResponses(
        magnetization=magnetization,
        magnetoelectric=magnetoelectric,
        piezomagnetic=piezomagnetic,
        magnetization_direction=_magnetization_direction(magnetization),
    )


def _matrix_lines(form, letter: str) -> list[str]:
    from .tensor_form import _table

    n_columns = len(form.matrix[0])
    rows = [[letter] + [str(c + 1) for c in range(n_columns)]]
    rows += [[str(r + 1)] + [entry.replace("T", letter) for entry in row]
             for r, row in enumerate(form.matrix)]
    return ["    " + line for line in _table(rows)]


def format_magnetic_responses(responses: MagneticResponses, forms: bool = True) -> list[str]:
    """Lines describing the responses, in the wording "allowed by symmetry".

    Args:
        responses: A :class:`MagneticResponses`.
        forms: Also give the matrix forms of the allowed tensors (the
            magnetoelectric tensor as ``alpha``, the piezomagnetic one in
            the 3x6 matrix notation as ``L``).

    Returns:
        Lines indented by two spaces.
    """
    lines = []
    if responses.n_magnetization:
        lines.append(f"  net magnetization: allowed by symmetry "
                     f"{responses.magnetization_direction}")
    else:
        lines.append("  net magnetization: forbidden by symmetry")
    for name, form, letter in (
        ("linear magnetoelectric effect", responses.magnetoelectric, "alpha"),
        ("piezomagnetism", responses.piezomagnetic, "L"),
    ):
        count = form.n_independent
        if not count:
            lines.append(f"  {name}: forbidden by symmetry")
            continue
        lines.append(f"  {name}: allowed by symmetry, {count} independent "
                     f"component{'s' if count != 1 else ''}")
        if forms:
            lines += _matrix_lines(form, letter)
    return lines


# ------------------------------------------------------------ collinear classes


def magnetic_cell_translations(cell_matrix) -> list[np.ndarray]:
    """Lattice translations of the primitive cell inside a magnetic cell.

    Args:
        cell_matrix: Integer ``(3, 3)`` matrix whose rows are the magnetic
            cell vectors in units of the primitive lattice vectors.

    Returns:
        The translations ``T`` (primitive fractional coordinates, one per
        primitive cell of the magnetic cell) in the order ``crystod-mag``
        lays out the spin field ``field[cell, site]``.
    """
    cell_matrix = np.asarray(cell_matrix, dtype=float)
    inverse_cell = np.linalg.inv(cell_matrix)
    corner_shifts = [np.zeros(3)]
    for axis in range(3):
        corner_shifts = corner_shifts + [shift + cell_matrix[axis] for shift in corner_shifts]
    corner_array = np.array(corner_shifts, dtype=float)
    t_low = np.floor(corner_array.min(axis=0)).astype(int) - 1
    t_high = np.ceil(corner_array.max(axis=0)).astype(int) + 1
    coset_eps = 1e-8
    translations = []
    for t1 in range(t_low[0], t_high[0] + 1):
        for t2 in range(t_low[1], t_high[1] + 1):
            for t3 in range(t_low[2], t_high[2] + 1):
                translation = np.array([t1, t2, t3], dtype=float)
                frac = translation @ inverse_cell
                if np.all(frac > -coset_eps) and np.all(frac < 1 - coset_eps):
                    translations.append(translation)
    return translations


def _chi_l(rotation: np.ndarray, degree: int) -> float:
    proper = rotation * float(np.sign(np.linalg.det(rotation)))
    cos_theta = np.clip((np.trace(proper) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        return float(2 * degree + 1)
    return float(np.sin((degree + 0.5) * theta) / np.sin(theta / 2.0))


def _monomials(degree: int) -> list[tuple[int, int, int]]:
    return sorted(
        ((a, b, degree - a - b) for a in range(degree + 1) for b in range(degree + 1 - a)),
        reverse=True,
    )


def _evaluate(points: np.ndarray, exponents) -> np.ndarray:
    return np.array([np.prod(points ** np.array(e), axis=1) for e in exponents]).T


def _monomial_text(exponent) -> str:
    parts = []
    for name, power in zip(("k_x", "k_y", "k_z"), exponent):
        if power == 1:
            parts.append(name)
        elif power > 1:
            parts.append(f"{name}^{power}")
    return " ".join(parts)


def _surd_coefficients(values) -> list[tuple[int, str]] | None:
    """Integer multiples of 1 and sqrt(3) proportional to ``values``
    (hexagonal axes), e.g. ``[(3, ""), (-2, "sqrt3"), (-3, "")]``; the
    scaling with the fewest sqrt3 terms is chosen. None when the values are
    not of that form."""
    root3 = math.sqrt(3.0)
    smallest = min(abs(v) for v in values if abs(v) > 1e-12)
    best = None
    for factor in (1.0, root3, 1.0 / root3):
        terms = []
        for value in np.asarray(values, dtype=float) / smallest * factor:
            plain = Fraction(float(value)).limit_denominator(12)
            surd = Fraction(float(value / root3)).limit_denominator(12)
            if abs(float(plain) - value) < 1e-6:
                terms.append((plain, ""))
            elif abs(float(surd) * root3 - value) < 1e-6:
                terms.append((surd, "sqrt3"))
            else:
                terms = None
                break
        if terms is None:
            continue
        denominator = math.lcm(*[f.denominator for f, _ in terms])
        integers = [int(f * denominator) for f, _ in terms]
        divisor = math.gcd(*integers)
        candidate = [(i // divisor, unit) for i, (_, unit) in zip(integers, terms)]
        key = (sum(1 for _, unit in candidate if unit), max(abs(i) for i, _ in candidate))
        if best is None or key < best[0]:
            best = (key, candidate)
    return None if best is None else best[1]


def _polynomial_text(coefficients, exponents) -> str:
    terms = [(c, e) for c, e in zip(coefficients, exponents) if abs(c) > 1e-9]
    if not terms:
        return "0"
    exact = []
    for value, _ in terms:
        fraction = Fraction(float(value)).limit_denominator(1000)
        exact.append(fraction if abs(float(fraction) - value) < 1e-8 else None)
    surds = None
    if all(f is not None for f in exact):
        denominator = math.lcm(*[f.denominator for f in exact])
        integers = [int(f * denominator) for f in exact]
        divisor = math.gcd(*integers)
        values: list = [i // divisor for i in integers]
    else:
        surds = _surd_coefficients([v for v, _ in terms])
        if surds is not None:
            values = [number for number, _ in surds]
        else:
            scale = max(abs(v) for v, _ in terms)
            values = [v / scale for v, _ in terms]
    if values[0] < 0:
        values = [-v for v in values]
    common = tuple(min(e[i] for _, e in terms) for i in range(3))
    reduced = [tuple(p - c for p, c in zip(e, common)) for _, e in terms]
    pieces = []
    for index, (value, exponent) in enumerate(zip(values, reduced)):
        magnitude = abs(value)
        body = _monomial_text(exponent)
        unit = surds[index][1] if surds is not None else ""
        if unit:
            coefficient = unit if magnitude == 1 else f"{magnitude}{unit}"
        elif isinstance(magnitude, int):
            coefficient = "" if magnitude == 1 and body else str(magnitude)
        else:
            coefficient = f"{magnitude:.4f}"
        text = " ".join(part for part in (coefficient, body) if part)
        if not pieces:
            pieces.append(("-" if value < 0 else "") + text)
        else:
            pieces.append(("- " if value < 0 else "+ ") + text)
    inner = " ".join(pieces)
    prefix = _monomial_text(common)
    if len(terms) == 1:
        return prefix
    return f"{prefix} ({inner})" if prefix else inner


def spin_splitting_forms(rotations, characters, degree: int) -> tuple[str, ...]:
    """Polynomials ``p`` of one degree in ``(k_x, k_y, k_z)`` with
    ``p(R k) = chi(R) p(k)`` for every rotation ``R``.

    Args:
        rotations: Cartesian rotation matrices of a point group.
        characters: The one-dimensional character (+1 or -1) of each.
        degree: The polynomial degree.

    Returns:
        The polynomials as text (``"k_x k_y"``), one per basis vector of
        the reduced row-echelon basis.
    """
    from .invariants import _rref

    exponents = _monomials(degree)
    rng = np.random.default_rng(20221)
    points = rng.normal(size=(3 * len(exponents) + 6, 3))
    blocks = []
    base = _evaluate(points, exponents)
    for rotation, character in zip(rotations, characters):
        moved = _evaluate(points @ np.asarray(rotation, dtype=float).T, exponents)
        blocks.append(moved - character * base)
    matrix = np.vstack(blocks)
    _, singular, vt = np.linalg.svd(matrix)
    tolerance = 1e-8 * max(1.0, float(singular[0]))
    rank = int(np.sum(singular > tolerance))
    null = vt[rank:]
    if len(null) == 0:
        return ()
    reduced, _ = _rref(null)
    return tuple(_polynomial_text(row, exponents) for row in reduced)


def _rotation_axis(rotation: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eig(rotation)
    index = int(np.argmin(np.abs(values - 1.0)))
    axis = np.real(vectors[:, index])
    return axis / np.linalg.norm(axis)


def _degenerate_sets(swapping_rotations) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Normals of the spin-degenerate planes and directions of the lines.

    A sublattice-exchanging rotation part ``R`` keeps the two spin channels
    degenerate at every k with ``R k = k`` and, because the nonrelativistic
    bands satisfy ``E(k) = E(-k)``, at every k with ``R k = -k``: a mirror
    or a two-fold rotation gives a plane, a proper rotation (or the proper
    part of an improper one) gives its axis.
    """
    planes: list[np.ndarray] = []
    lines: list[np.ndarray] = []

    def add(collection, vector):
        vector = vector / np.linalg.norm(vector)
        if not any(abs(abs(np.dot(vector, other)) - 1.0) < 1e-6 for other in collection):
            collection.append(vector)

    for rotation in swapping_rotations:
        proper = rotation * float(np.sign(np.linalg.det(rotation)))
        if np.allclose(proper, np.eye(3), atol=1e-6):
            continue
        axis = _rotation_axis(proper)
        if abs(np.trace(proper) + 1.0) < 1e-6:          # two-fold proper part
            add(planes, axis)
        add(lines, axis)
    lines = [line for line in lines
             if not any(abs(np.dot(line, normal)) < 1e-6 for normal in planes)]
    return planes, lines


def classify_collinear(rotations, translations, positions, site_indices, cell_matrix,
                       field, lattice, *, symprec: float | None = None) -> CollinearClass:
    """Nonrelativistic class of one collinear spin arrangement.

    The arrangement is reduced to its sign pattern ``s = +1 / -1 / 0`` along
    the common spin axis (with the magnitudes); the space-group operations
    of the non-magnetic crystal, combined with every primitive translation
    inside the magnetic cell, are sorted into those that keep the pattern
    (``H``) and those that reverse every sign (``A``, the operations that
    exchange the two sublattices). Net moment nonzero: ferromagnet or
    ferrimagnet; ``A`` empty: compensated arrangement whose sublattices are
    not related by symmetry; ``A`` holding a pure translation or the
    inversion: conventional antiferromagnet; otherwise an altermagnet,
    whose wave type is the lowest even ``l`` at which the character
    ``chi_A`` (+1 on the rotation parts of ``H``, -1 on those of ``A``)
    occurs in the angular-momentum representation ``D_l`` (2: d-wave,
    4: g-wave, 6: i-wave), after Smejkal, Sinova and Jungwirth, Phys. Rev.
    X 12, 031042 (2022).

    Args:
        rotations: Integer rotations of the space group of the primitive
            cell (``SymmetryOnlyVibrations.rotations``).
        translations: Their fractional translations.
        positions: Fractional positions of the primitive cell.
        site_indices: Indices of the N magnetic sites.
        cell_matrix: Integer ``(3, 3)`` magnetic cell in primitive units
            (rows).
        field: The real spin field ``(n_cells, N, 3)`` over the magnetic
            cell, cells in the order of :func:`magnetic_cell_translations`.
        lattice: Primitive lattice vectors as rows (for the Cartesian
            rotations and directions).
        symprec: Distance tolerance (Angstrom) of the position matching:
            an image of a site matches the nearest site within it (the
            tolerance the operations were found with, ``--tolerance`` of
            ``crystod-mag``). Sites within 1e-5 in every fractional
            coordinate always match; None (default) matches those only.

    Returns:
        A :class:`CollinearClass`.

    Example:
        >>> # bcc cell with a +z / -z pair: the body-centring translation
        >>> # exchanges the two sublattices
        >>> import numpy as np, spglib
        >>> from crystod import mag
        >>> lattice = np.eye(3) * 2.87
        >>> positions = [[0, 0, 0], [0.5, 0.5, 0.5]]
        >>> sym = spglib.get_symmetry_dataset((lattice, positions, [26, 26]))
        >>> field = np.array([[[0, 0, 1], [0, 0, -1]]], dtype=float)
        >>> klass = mag.classify_collinear(sym.rotations, sym.translations, positions,
        ...                                [0, 1], np.eye(3, dtype=int), field, lattice)
        >>> klass.name
        'antiferromagnet (conventional: a translation exchanges the sublattices, T tau)'
    """
    field = np.asarray(field, dtype=float)
    n_cells, n_sites, _ = field.shape
    flat = field.reshape(-1, 3)
    norms = np.linalg.norm(flat, axis=1)
    scale = max(float(norms.max()), 1e-300)
    nonzero = norms > 1e-8 * scale
    net = flat.sum(axis=0)
    if not nonzero.any():
        return CollinearClass(name=CLASS_NONCOLLINEAR, net_moment=net)
    axis = flat[np.argmax(nonzero)] / norms[np.argmax(nonzero)]
    for vector, norm in zip(flat[nonzero], norms[nonzero]):
        if np.linalg.norm(np.cross(axis, vector / norm)) > 1e-6:
            return CollinearClass(name=CLASS_NONCOLLINEAR, net_moment=net)
    for value in axis:           # canonical axis: first nonzero component positive
        if abs(value) > 1e-8:
            if value < 0:
                axis = -axis
            break
    projected = flat @ axis / scale
    projected[~nonzero] = 0.0

    if np.linalg.norm(net) > 1e-6 * scale:
        signs = np.sign(projected[nonzero])
        name = CLASS_FERROMAGNET if np.all(signs == signs[0]) else CLASS_FERRIMAGNET
        return CollinearClass(name=name, axis=axis, net_moment=net)

    positions = np.asarray(positions, dtype=float)
    site_positions = positions[list(site_indices)]
    cell_translations = np.array(magnetic_cell_translations(cell_matrix))
    inverse_cell = np.linalg.inv(np.asarray(cell_matrix, dtype=float))
    # (cell, site) -> fractional position
    sites = (cell_translations[:, None, :] + site_positions[None, :, :]).reshape(-1, 3)

    metric_lattice = np.asarray(lattice, dtype=float)
    distance_tolerance = -1.0 if symprec is None else float(symprec)

    def locate(points: np.ndarray) -> np.ndarray | None:
        """Index (cell * N + site) of every point, modulo the magnetic cell."""
        result = np.empty(len(points), dtype=int)
        for p, point in enumerate(points):
            diff = point[None, :] - site_positions
            residual = diff - np.rint(diff)
            match = np.where(np.all(np.abs(residual) < 1e-5, axis=1))[0]
            if len(match) == 0 and distance_tolerance >= 0.0:
                # off-ideal positions: the nearest site within the tolerance
                distances = np.linalg.norm(residual @ metric_lattice, axis=1)
                nearest = int(np.argmin(distances))
                if distances[nearest] <= distance_tolerance:
                    match = np.array([nearest])
            if len(match) == 0:
                return None
            site = int(match[0])
            shift = np.rint(diff[site])
            relative = (shift[None, :] - cell_translations) @ inverse_cell
            cell = np.where(np.all(np.abs(relative - np.rint(relative)) < 1e-6, axis=1))[0]
            if len(cell) == 0:
                return None
            result[p] = int(cell[0]) * n_sites + site
        return result

    lattice = np.asarray(lattice, dtype=float)
    preserving, swapping = [], []
    for rotation, translation in zip(np.asarray(rotations), np.asarray(translations)):
        moved = sites @ np.asarray(rotation, dtype=float).T + translation
        for shift in cell_translations:
            image = locate(moved + shift)
            if image is None:
                continue
            target = projected[image]
            if np.allclose(target, projected, atol=1e-6):
                preserving.append((rotation, shift + translation))
            elif np.allclose(target, -projected, atol=1e-6):
                swapping.append((rotation, shift + translation))

    common = dict(axis=axis, n_preserving=len(preserving), n_swapping=len(swapping),
                  net_moment=net)
    if not swapping:
        return CollinearClass(name=CLASS_COMPENSATED, **common)
    identity = np.eye(3, dtype=int)
    if any(np.array_equal(np.asarray(r), identity) for r, _ in swapping):
        return CollinearClass(name=CLASS_AFM_TRANSLATION, **common)
    if any(np.array_equal(np.asarray(r), -identity) for r, _ in swapping):
        return CollinearClass(name=CLASS_AFM_INVERSION, **common)

    def distinct(operations):
        found: dict = {}
        for rotation, _ in operations:
            found.setdefault(tuple(np.asarray(rotation).ravel()), rotation)
        return [_cartesian(r, lattice) for r in found.values()]

    kept = distinct(preserving)
    exchanged = distinct(swapping)
    for a in kept:
        if any(np.allclose(a, b, atol=1e-6) for b in exchanged):
            raise RuntimeError("a rotation part both keeps and reverses the sign pattern")
    group = kept + exchanged
    characters = [1.0] * len(kept) + [-1.0] * len(exchanged)
    order = None
    for degree in (2, 4, 6):
        multiplicity = sum(c * _chi_l(r, degree) for r, c in zip(group, characters)) / len(group)
        if multiplicity > 0.5:
            order = degree
            break
    forms = spin_splitting_forms(group, characters, order) if order else ()
    planes, lines = _degenerate_sets(exchanged)
    return CollinearClass(
        name=CLASS_ALTERMAGNET,
        wave=_WAVE_NAMES.get(order),
        order=order,
        spin_splitting=forms,
        degenerate_planes=tuple(sorted(_direction_text(v) for v in planes)[::-1]),
        degenerate_lines=tuple(sorted(_direction_text(v) for v in lines)[::-1]),
        **common,
    )
