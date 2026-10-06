"""Symmetry-allowed forms of property tensors (crystod-group --tensor).

Neumann's principle: a property tensor of a crystal is invariant under every
operation of its point group.  For a tensor of rank ``r`` the operation
``R`` acts on the components as ``T -> (det R)^e tau R x R x ... x R T``,
with ``e = 1`` for an axial tensor and the time-reversal parity ``tau`` for
an operation combined with time reversal.  The allowed tensors are the
invariant vectors of that representation, restricted to the subspace fixed
by the intrinsic (index-permutation) symmetry of the tensor.  This module
computes them as the degree-1 invariants of the representation
(:func:`crystod.invariants.invariants_of_matrices`, checked against the
Molien series), picks the independent components by reduced row-echelon
form in the matrix-notation order of Nye (``11, 22, 33, 23, 13, 12``) and
prints the matrix form, the offline counterpart of the TENSOR program of the
Bilbao Crystallographic Server and of the tables of J. F. Nye, *Physical
Properties of Crystals* (Oxford, 1957).

Tensor kinds are given by name (``dielectric``, ``pyroelectric``,
``piezoelectric``, ``elastic``, ``compliance``, ``gyration``, ``raman``) or
by Jahn symbol:

- ``V`` -- polar vector; ``V2`` -- second rank without index symmetry;
  ``V[V2]`` -- third rank symmetric in the last two indices;
- ``[X n]`` / ``{X n}`` -- symmetric / antisymmetric n-th power of ``X``
  (``[V2]``, ``[[V2]2]``, ``{V2}``, ``[V3]``);
- prefix ``e`` -- axial (one extra factor ``det R``), prefix ``a`` --
  odd under time reversal.

Axes.  Cartesian axes follow the crystal-physics convention: ``x, y, z``
along ``a, b, c`` for orthogonal lattices; ``x || a, z || c`` for the
hexagonal and trigonal groups (hexagonal axes); ``y || b`` (unique axis)
and ``z || c`` for monoclinic groups.

Time reversal.  A point group given by its symbol is the point group of a
non-magnetic crystal, whose full symmetry also contains time reversal (the
grey group ``G1'``).  A time-odd (``a``) tensor therefore vanishes
identically for every such group; the magnetic point groups, in which a
time-odd tensor can survive, are reached through
:func:`tensor_form_of_operations` with per-operation time-reversal flags.

Example:
    >>> from crystod import group
    >>> form = group.tensor_form("4mm", "piezoelectric")
    >>> form.independent
    ('d15', 'd31', 'd33')
    >>> form.relations
    ('d24 = d15', 'd32 = d31')
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

# ------------------------------------------------------------- point groups

_SCHOENFLIES = {
    "1": "C1", "-1": "Ci", "2": "C2", "m": "Cs", "2/m": "C2h", "222": "D2",
    "mm2": "C2v", "mmm": "D2h", "4": "C4", "-4": "S4", "4/m": "C4h",
    "422": "D4", "4mm": "C4v", "-42m": "D2d", "4/mmm": "D4h", "3": "C3",
    "-3": "C3i", "32": "D3", "3m": "C3v", "-3m": "D3d", "6": "C6",
    "-6": "C3h", "6/m": "C6h", "622": "D6", "6mm": "C6v", "-6m2": "D3h",
    "6/mmm": "D6h", "23": "T", "m-3": "Th", "432": "O", "-43m": "Td",
    "m-3m": "Oh",
}

_SYSTEMS = (
    ("triclinic", ("1", "-1")),
    ("monoclinic", ("2", "m", "2/m")),
    ("orthorhombic", ("222", "mm2", "mmm")),
    ("tetragonal", ("4", "-4", "4/m", "422", "4mm", "-42m", "4/mmm")),
    ("trigonal", ("3", "-3", "32", "3m", "-3m")),
    ("hexagonal", ("6", "-6", "6/m", "622", "6mm", "-6m2", "6/mmm")),
    ("cubic", ("23", "m-3", "432", "-43m", "m-3m")),
)
CRYSTAL_SYSTEM = {pg: system for system, groups in _SYSTEMS for pg in groups}

LAUE_CLASS = {
    "1": "-1", "-1": "-1", "2": "2/m", "m": "2/m", "2/m": "2/m",
    "222": "mmm", "mm2": "mmm", "mmm": "mmm", "4": "4/m", "-4": "4/m",
    "4/m": "4/m", "422": "4/mmm", "4mm": "4/mmm", "-42m": "4/mmm",
    "4/mmm": "4/mmm", "3": "-3", "-3": "-3", "32": "-3m", "3m": "-3m",
    "-3m": "-3m", "6": "6/m", "-6": "6/m", "6/m": "6/m", "622": "6/mmm",
    "6mm": "6/mmm", "-6m2": "6/mmm", "6/mmm": "6/mmm", "23": "m-3",
    "m-3": "m-3", "432": "m-3m", "-43m": "m-3m", "m-3m": "m-3m",
}

#: The 32 crystallographic point groups in the order of the International Tables.
POINT_GROUPS = tuple(pg for _, groups in _SYSTEMS for pg in groups)

# columns: a, b, c of the hexagonal lattice in Cartesian axes (x || a, z || c)
_HEXAGONAL_BASIS = np.array([[1.0, -0.5, 0.0], [0.0, np.sqrt(3.0) / 2.0, 0.0],
                             [0.0, 0.0, 1.0]])

_AXES_TEXT = {
    "triclinic": "z || c, x in the ac plane, y = z x x",
    "monoclinic": "y || b (unique axis), z || c, x = y x z",
    "orthorhombic": "x || a, y || b, z || c",
    "tetragonal": "x || a, y || b, z || c",
    "trigonal": "x || a, z || c (hexagonal axes), y = z x x",
    "hexagonal": "x || a, z || c, y = z x x",
    "cubic": "x || a, y || b, z || c",
}


@dataclass(frozen=True, eq=False)
class PointGroupSetting:
    """A crystallographic point group as Cartesian operations.

    Attributes:
        symbol: Hermann-Mauguin symbol (phonopy/spglib spelling, ``"4mm"``).
        schoenflies: Schoenflies symbol (``"C4v"``).
        crystal_system: ``"tetragonal"`` etc.
        rotations: Orthogonal Cartesian matrices, shape ``(n_ops, 3, 3)``.
        lattice_basis: Columns ``a, b, c`` of the lattice the directions are
            quoted in (the identity, or the hexagonal basis).
        source: What the group was taken from (``"point group 4mm"``,
            ``"space group P4mm (No. 99)"``, ``"structure POSCAR"``).
        axes: The axis convention as text.
        input_axes: For a structure, the tensor axes x, y, z as rows in the
            Cartesian frame of the input file; None otherwise.
        space_group: Space-group symbol and number for ``--sg`` and ``-c``
            (``"P4mm (No. 99)"``); empty for a bare point group.
        table_rotations: For a bare point group, the integer matrices of the
            phonopy character table (class order) behind ``rotations``;
            None otherwise.
        class_index: For a bare point group, the class number of every
            operation in the phonopy character table.
    """

    symbol: str
    schoenflies: str
    crystal_system: str
    rotations: np.ndarray
    lattice_basis: np.ndarray
    source: str
    axes: str
    input_axes: np.ndarray | None = None
    space_group: str = ""
    table_rotations: np.ndarray | None = field(default=None, repr=False)
    class_index: tuple = field(default=(), repr=False)

    @property
    def order(self) -> int:
        """Number of operations."""
        return len(self.rotations)

    @property
    def centrosymmetric(self) -> bool:
        """True when the inversion is an operation."""
        return any(np.allclose(r, -np.eye(3)) for r in self.rotations)

    @property
    def polar(self) -> bool:
        """True when a polar vector is invariant (the 10 polar classes)."""
        total = np.sum(self.rotations, axis=0) / self.order
        return bool(np.linalg.norm(total) > 1e-8)

    @property
    def chiral(self) -> bool:
        """True when every operation is proper (the 11 enantiomorphic classes)."""
        return all(np.linalg.det(r) > 0 for r in self.rotations)


def _cartesian_rotations(integer_rotations) -> tuple[np.ndarray, np.ndarray]:
    """Integer rotations in a lattice basis -> orthogonal Cartesian matrices
    and the lattice basis used (hexagonal when the integer matrices are not
    orthogonal, the identity otherwise; as in
    ``basis_function._cartesianize_rotations``)."""
    rotations = np.asarray(integer_rotations, dtype=float)
    hexagonal = any(not np.allclose(r.T @ r, np.eye(3)) for r in rotations)
    basis = _HEXAGONAL_BASIS if hexagonal else np.eye(3)
    inverse = np.linalg.inv(basis)
    cartesian = np.array([basis @ r @ inverse for r in rotations])
    cartesian[np.abs(cartesian) < 1e-12] = 0.0
    return cartesian, basis


def _unique_rotations(rotations) -> np.ndarray:
    seen, unique = set(), []
    for rotation in np.asarray(rotations):
        key = tuple(np.rint(rotation).astype(int).ravel())
        if key not in seen:
            seen.add(key)
            unique.append(np.rint(rotation).astype(int))
    return np.array(unique)


def _character_table(point_group: str) -> dict:
    from .decompose_irrep import get_character_table

    return get_character_table(point_group)


def _check_symbol(symbol: str) -> str:
    text = "".join(str(symbol).split())
    if text not in _SCHOENFLIES:
        raise SystemExit(
            f'ERROR: "{symbol}" is not a crystallographic point group.\n'
            f"Choose from: {', '.join(POINT_GROUPS)}"
        )
    return text


def point_group_setting(point_group: str | None = None, *, space_group=None,
                        cell=None, symprec: float = 1e-5) -> PointGroupSetting:
    """The Cartesian operations of a point group, of a space group's point
    group, or of a crystal structure's point group.

    Exactly one source is given.  A point-group symbol takes the operations
    of phonopy's character table (the first setting where there are two,
    e.g. ``-42m`` with the two-fold axes along x and y); a space group the
    rotations of its default setting (spglib's first Hall number, the
    setting of the International Tables); a structure the rotations of
    spglib's standardized conventional cell.

    Args:
        point_group: Hermann-Mauguin symbol (``"4mm"``, ``"-43m"``).
        space_group: Space-group symbol or number (``"P4mm"``, ``99``).
        cell: A POSCAR path or a ``PhonopyAtoms``.
        symprec: Symmetry tolerance of the structure route (Angstrom).

    Returns:
        A :class:`PointGroupSetting`.

    Raises:
        SystemExit: Unknown symbol, or not exactly one source given
            (``ValueError`` through ``crystod.group``).
        ValueError: The structure cannot be read.
    """
    given = [value is not None for value in (point_group, space_group, cell)]
    if sum(given) != 1:
        raise SystemExit("ERROR: give exactly one of a point group, a space group "
                         "or a structure.")
    if point_group is not None:
        symbol = _check_symbol(point_group)
        table = _character_table(symbol)
        integer, classes = [], []
        names = table["rotation_list"]
        for index, name in enumerate((names,) if isinstance(names, str) else names):
            for rotation in table["mapping_table"][name]:
                integer.append(np.array(rotation, dtype=int))
                classes.append(index)
        cartesian, basis = _cartesian_rotations(integer)
        system = CRYSTAL_SYSTEM[symbol]
        return PointGroupSetting(
            symbol=symbol, schoenflies=_SCHOENFLIES[symbol], crystal_system=system,
            rotations=cartesian, lattice_basis=basis, source=f"point group {symbol}",
            axes=_AXES_TEXT[system], table_rotations=np.array(integer),
            class_index=tuple(classes),
        )
    if space_group is not None:
        import spglib

        from .runtime_compat import get_spacegroup_type
        from .spacegroup_product import _resolve_space_group

        info = _resolve_space_group(str(space_group))
        hall = int(info.hall_number)
        integer = _unique_rotations(spglib.get_symmetry_from_database(hall)["rotations"])
        sg_type = get_spacegroup_type(spglib.get_spacegroup_type(hall))
        symbol = _check_symbol(sg_type.pointgroup_international)
        cartesian, basis = _cartesian_rotations(integer)
        system = CRYSTAL_SYSTEM[symbol]
        name = f"{sg_type.international_short} (No. {sg_type.number})"
        return PointGroupSetting(
            symbol=symbol, schoenflies=_SCHOENFLIES[symbol], crystal_system=system,
            rotations=cartesian, lattice_basis=basis, source=f"space group {name}",
            axes=_AXES_TEXT[system] + " of the conventional cell", space_group=name,
        )
    return _structure_setting(cell, symprec)


def _structure_setting(cell, symprec: float) -> PointGroupSetting:
    import spglib

    from .phonon_activity import _read_cell

    atoms = _read_cell(cell)
    lattice = np.asarray(atoms.cell, dtype=float)
    spg_cell = (lattice, np.asarray(atoms.scaled_positions, dtype=float),
                np.asarray(atoms.numbers, dtype=int))
    dataset = spglib.get_symmetry_dataset(spg_cell, symprec=symprec)
    if dataset is None:
        raise ValueError(f"spglib found no symmetry for {cell} (symprec {symprec})")
    transformation = np.asarray(dataset.transformation_matrix, dtype=float)
    inverse = np.linalg.inv(transformation)
    standard = [transformation @ np.asarray(w, dtype=float) @ inverse
                for w in dataset.rotations]
    if not np.allclose(standard, np.rint(standard), atol=1e-6):
        raise ValueError("the standardized rotations are not integer matrices")
    integer = _unique_rotations(standard)
    cartesian, basis = _cartesian_rotations(integer)
    symbol = _check_symbol(dataset.pointgroup)
    system = CRYSTAL_SYSTEM[symbol]
    # the standard basis vectors in the Cartesian frame of the input file
    vectors = lattice.T @ inverse          # columns a_s, b_s, c_s
    z = vectors[:, 2] / np.linalg.norm(vectors[:, 2])
    x = vectors[:, 0] - (vectors[:, 0] @ z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    name = f"{dataset.international} (No. {dataset.number})"
    label = cell if isinstance(cell, str) else "the structure"
    return PointGroupSetting(
        symbol=symbol, schoenflies=_SCHOENFLIES[symbol], crystal_system=system,
        rotations=cartesian, lattice_basis=basis,
        source=f"structure {label} (symprec {symprec:g})",
        axes=_AXES_TEXT[system] + " of the standard conventional cell",
        input_axes=np.array([x, y, z]), space_group=name,
    )


# ------------------------------------------------------------ Jahn symbols


@dataclass(frozen=True)
class _Kind:
    jahn: str
    letter: str
    voigt: str          # "", "strain" (Nye d) or "compliance" (Nye s)
    description: str


KINDS = {
    "dielectric": _Kind(
        "[V2]", "eps", "",
        "dielectric permittivity eps_ij, D_i = eps_ij E_j (the same form holds "
        "for every symmetric polar second-rank property: susceptibility, "
        "conductivity, thermal expansion)"),
    "pyroelectric": _Kind(
        "V", "p", "", "pyroelectric coefficients p_i, dP_i = p_i dT"),
    "piezoelectric": _Kind(
        "V[V2]", "d", "strain",
        "piezoelectric strain coefficients d_ijk, P_i = d_ijk sigma_jk"),
    "elastic": _Kind(
        "[[V2]2]", "c", "",
        "elastic stiffness c_ijkl, sigma_ij = c_ijkl e_kl"),
    "compliance": _Kind(
        "[[V2]2]", "s", "compliance",
        "elastic compliance s_ijkl, e_ij = s_ijkl sigma_kl"),
    "gyration": _Kind(
        "e[V2]", "g", "",
        "optical-activity gyration tensor g_ij (symmetric axial second rank; "
        "Nye, ch. XIV)"),
    "raman": _Kind(
        "[V2]", "a", "",
        "Raman tensors: symmetric second-rank polarizability derivatives, "
        "one set per Raman-active irrep"),
}

_INDEX_LETTERS = "ijklmn"
_MAX_RANK = 6


@dataclass(frozen=True)
class JahnSymbol:
    """A parsed Jahn symbol.

    Attributes:
        symbol: The symbol as given (prefixes included).
        rank: Number of vector indices.
        axial: True for an ``e`` prefix (extra factor ``det R``).
        time_odd: True for an ``a`` prefix (odd under time reversal).
        permutations: The intrinsic index-permutation group as
            ``(permutation, sign)`` pairs (``sign = -1`` for antisymmetry).
        intrinsic: The intrinsic symmetry as text, one entry per bracket.
    """

    symbol: str
    rank: int
    axial: bool
    time_odd: bool
    permutations: tuple
    intrinsic: tuple


def _shift(generators, offset: int, rank: int):
    shifted = []
    for perm, sign in generators:
        full = list(range(rank))
        for position, target in enumerate(perm):
            full[offset + position] = offset + target
        shifted.append((tuple(full), sign))
    return shifted


class _Factor:
    """Rank, permutation generators and symmetry texts of a sub-symbol."""

    def __init__(self, rank: int, generators=(), texts=()):
        self.rank = rank
        self.generators = list(generators)
        self.texts = list(texts)          # (kind, blocks of positions)

    @staticmethod
    def concat(factors):
        rank = sum(f.rank for f in factors)
        out = _Factor(rank)
        offset = 0
        for f in factors:
            out.generators += _shift(f.generators, offset, rank)
            out.texts += [(kind, [[p + offset for p in block] for block in blocks])
                          for kind, blocks in f.texts]
            offset += f.rank
        return out

    @staticmethod
    def power(base, n: int, sign: int):
        copies = _Factor.concat([base] * n)
        r = base.rank
        for k in range(n - 1):
            perm = list(range(copies.rank))
            for p in range(r):
                perm[k * r + p], perm[(k + 1) * r + p] = (k + 1) * r + p, k * r + p
            copies.generators.append((tuple(perm), sign))
        blocks = [list(range(k * r, (k + 1) * r)) for k in range(n)]
        copies.texts.append(("symmetric" if sign > 0 else "antisymmetric", blocks))
        return copies


def _matching(text: str, start: int) -> int:
    opening = text[start]
    closing = "]" if opening == "[" else "}"
    depth = 0
    for position in range(start, len(text)):
        if text[position] in "[{":
            depth += 1
        elif text[position] in "]}":
            depth -= 1
            if depth == 0:
                if text[position] != closing:
                    break
                return position
    raise ValueError(f"unbalanced brackets in {text!r}")


def _parse_product(text: str, whole: str) -> _Factor:
    if not text:
        raise ValueError(f"empty factor in the Jahn symbol {whole!r}")
    factors = []
    position = 0
    while position < len(text):
        char = text[position]
        if char == "V":
            factor = _Factor(1)
            position += 1
        elif char in "[{":
            end = _matching(text, position)
            inner = text[position + 1:end]
            closing = text[end]
            match = re.fullmatch(r"(.+?)(\d+)", inner)
            if not match or int(match.group(2)) < 2:
                raise ValueError(
                    f"{char}...{closing} in {whole!r} needs a power "
                    "of 2 or more before the closing bracket, e.g. [V2] or [[V2]2]")
            base = _parse_product(match.group(1), whole)
            factor = _Factor.power(base, int(match.group(2)), 1 if char == "[" else -1)
            position = end + 1
        else:
            raise ValueError(
                f"unexpected {char!r} in the Jahn symbol {whole!r} (use V, [ ], {{ }}, "
                "digits, and the prefixes e (axial) and a (time-odd))")
        digits = re.match(r"\d+", text[position:])
        if digits:
            count = int(digits.group(0))
            if count < 1:
                raise ValueError(f"power 0 in the Jahn symbol {whole!r}")
            factor = _Factor.concat([factor] * count)
            position += len(digits.group(0))
        factors.append(factor)
    return _Factor.concat(factors)


def _closure(generators, rank: int) -> tuple:
    identity = (tuple(range(rank)), 1)
    group = {identity}
    frontier = [identity]
    while frontier:
        new = []
        for perm, sign in frontier:
            for gperm, gsign in generators:
                composed = (tuple(perm[g] for g in gperm), sign * gsign)
                if composed not in group:
                    group.add(composed)
                    new.append(composed)
        frontier = new
    return tuple(sorted(group))


def _letters(positions) -> str:
    return "".join(_INDEX_LETTERS[p] for p in positions)


def _intrinsic_text(kind: str, blocks) -> str:
    if all(len(block) == 1 for block in blocks):
        return f"{kind} in ({_letters([b[0] for b in blocks])})"
    names = ", ".join(f"({_letters(block)})" for block in blocks)
    verb = "symmetric" if kind == "symmetric" else "antisymmetric"
    return f"{verb} under the exchange of the index groups {names}"


def parse_jahn(symbol: str) -> JahnSymbol:
    """Parse a Jahn symbol such as ``V[V2]``, ``[[V2]2]`` or ``ae[V2]``.

    Args:
        symbol: The Jahn symbol: optional prefixes ``e`` (axial) and ``a``
            (time-odd), then a product of ``V``, ``[X n]`` (symmetric power),
            ``{X n}`` (antisymmetric power) and plain powers ``X n``.

    Returns:
        A :class:`JahnSymbol`.

    Raises:
        ValueError: The symbol cannot be parsed, or its rank exceeds 6.

    Example:
        >>> from crystod.tensor_form import parse_jahn
        >>> jahn = parse_jahn("[[V2]2]")
        >>> jahn.rank, len(jahn.permutations), jahn.intrinsic
        (4, 8, ('symmetric in (ij)', 'symmetric in (kl)', 'symmetric under the exchange of the index groups (ij), (kl)'))
    """
    text = "".join(str(symbol).split())
    match = re.match(r"[ae]*", text)
    prefixes = match.group(0)
    if len(set(prefixes)) != len(prefixes):
        raise ValueError(f"repeated prefix in the Jahn symbol {symbol!r}")
    body = text[len(prefixes):]
    factor = _parse_product(body, text)
    if factor.rank > _MAX_RANK:
        raise ValueError(f"the Jahn symbol {symbol!r} has rank {factor.rank}; "
                         f"at most {_MAX_RANK} is supported")
    texts = tuple(_intrinsic_text(kind, blocks) for kind, blocks in factor.texts)
    return JahnSymbol(
        symbol=text, rank=factor.rank, axial="e" in prefixes,
        time_odd="a" in prefixes,
        permutations=_closure(factor.generators, factor.rank), intrinsic=texts,
    )


# ------------------------------------------------------------ tensor algebra

_VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def _rotate(tensor: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    out = tensor
    for axis in range(tensor.ndim):
        out = np.moveaxis(np.tensordot(rotation, out, axes=([1], [axis])), 0, axis)
    return out


def _symmetrizer(jahn: JahnSymbol) -> np.ndarray:
    """Orthogonal projector onto the tensors with the intrinsic symmetry."""
    size = 3 ** jahn.rank
    index = np.arange(size).reshape([3] * jahn.rank)
    projector = np.zeros((size, size))
    for perm, sign in jahn.permutations:
        moved = index.transpose(perm).ravel()
        projector[np.arange(size), moved] += sign
    return projector / len(jahn.permutations)


def _layout(jahn: JahnSymbol) -> str:
    perms = {p for p, s in jahn.permutations if s > 0}
    if jahn.rank == 1:
        return "vector"
    if jahn.rank == 2:
        return "matrix"
    if jahn.rank == 3 and (0, 2, 1) in perms:
        return "3x6"
    if jahn.rank == 4 and (1, 0, 2, 3) in perms and (0, 1, 3, 2) in perms:
        return "6x6"
    return "list"


def _cells(layout: str, rank: int):
    """Display cells as (row, column, index tuple); row/column 0-based."""
    if layout == "vector":
        return [(0, i, (i,)) for i in range(3)]
    if layout == "matrix":
        return [(i, j, (i, j)) for i in range(3) for j in range(3)]
    if layout == "3x6":
        return [(i, lam, (i,) + _VOIGT_PAIRS[lam]) for i in range(3) for lam in range(6)]
    if layout == "6x6":
        return [(lam, mu, _VOIGT_PAIRS[lam] + _VOIGT_PAIRS[mu])
                for lam in range(6) for mu in range(6)]
    return [(n, 0, t) for n, t in enumerate(itertools.product(range(3), repeat=rank))]


def _cell_factor(voigt: str, layout: str, row: int, column: int) -> int:
    if voigt == "strain" and layout == "3x6":
        return 2 if column >= 3 else 1
    if voigt == "compliance" and layout == "6x6":
        return (2 if row >= 3 else 1) * (2 if column >= 3 else 1)
    return 1


def _shear_count(layout: str, row: int, column: int) -> int:
    """Number of shear (off-diagonal) index pairs of a display cell."""
    if layout == "matrix":
        return int(row != column)
    if layout == "3x6":
        return int(column >= 3)
    if layout == "6x6":
        return int(row >= 3) + int(column >= 3)
    return 0


def _cell_name(letter: str, layout: str, row: int, column: int, indices) -> str:
    if layout == "vector":
        return f"{letter}{column + 1}"
    if layout == "list":
        return letter + "".join(str(i + 1) for i in indices)
    return f"{letter}{row + 1}{column + 1}"


def _orbit(indices, permutations):
    """{tuple: sign relative to ``indices``} of the orbit, or None when the
    component vanishes by antisymmetry."""
    orbit: dict = {}
    for perm, sign in permutations:
        image = tuple(indices[p] for p in perm)
        if image in orbit and orbit[image] != sign:
            return None
        orbit[image] = sign
    return orbit


# ------------------------------------------------------------ exact numbers


def _exact(value: float):
    """(Fraction, radicand) with value = Fraction * sqrt(radicand), or None."""
    for radicand in (1, 3, 2, 6):
        scaled = value / math.sqrt(radicand)
        frac = Fraction(scaled).limit_denominator(1000)
        if abs(scaled - float(frac)) < 1e-9:
            return frac, radicand
    return None


def _term_text(coefficient, name: str) -> str:
    """|coefficient| * name, coefficient as (Fraction, radicand) or float."""
    if isinstance(coefficient, float):
        return f"{abs(coefficient):.6g}{name}"
    frac, radicand = coefficient
    frac = abs(frac)
    root = "" if radicand == 1 else f"sqrt({radicand})"
    numerator = "" if frac.numerator == 1 and root else (
        "" if frac.numerator == 1 else str(frac.numerator))
    text = f"{numerator}{root}{name}"
    if frac.denominator != 1:
        text += f"/{frac.denominator}"
    return text


def _sign(coefficient) -> int:
    value = coefficient if isinstance(coefficient, float) else float(coefficient[0])
    return -1 if value < 0 else 1


def combination_text(terms) -> str:
    """A linear combination of named components as compact text.

    Args:
        terms: ``(value, name)`` pairs (floats; zero values are skipped).

    Returns:
        ``"0"``, ``"d15"``, ``"-2d22"``, ``"(c11-c12)/2"`` or
        ``"2(s11-s12)"``: a common rational factor is taken out of a sum
        with rational coefficients.

    Example:
        >>> from crystod.tensor_form import combination_text
        >>> combination_text([(0.5, "c11"), (-0.5, "c12")])
        '(c11-c12)/2'
        >>> combination_text([(2.0, "s11"), (-2.0, "s12")])
        '2(s11-s12)'
    """
    exact = []
    for value, name in terms:
        if abs(value) < 1e-9:
            continue
        found = _exact(float(value))
        exact.append((found if found is not None else float(value), name))
    if not exact:
        return "0"
    if len(exact) == 1:
        coefficient, name = exact[0]
        return ("-" if _sign(coefficient) < 0 else "") + _term_text(coefficient, name)
    rational = all(not isinstance(c, float) and c[1] == 1 for c, _ in exact)
    if rational:
        fracs = [c[0] for c, _ in exact]
        numerator = math.gcd(*[abs(f.numerator) for f in fracs])
        denominator = math.lcm(*[f.denominator for f in fracs])
        factor = Fraction(numerator, denominator) * (1 if fracs[0] > 0 else -1)
        inner = ""
        for position, ((_, name), frac) in enumerate(zip(exact, fracs)):
            k = frac / factor
            magnitude = "" if abs(k) == 1 else str(abs(k))
            sign = "-" if k < 0 else ("+" if position else "")
            inner += f"{sign}{magnitude}{name}"
        if factor == 1:
            return inner
        if factor == -1:
            return f"-({inner})"
        prefix = "-" if factor < 0 else ""
        magnitude = abs(factor)
        head = "" if magnitude.numerator == 1 else str(magnitude.numerator)
        tail = "" if magnitude.denominator == 1 else f"/{magnitude.denominator}"
        return f"{prefix}{head}({inner}){tail}"
    text = ""
    for position, (coefficient, name) in enumerate(exact):
        sign = "-" if _sign(coefficient) < 0 else ("+" if position else "")
        text += sign + _term_text(coefficient, name)
    return text


# ------------------------------------------------------------ tensor forms


@dataclass(frozen=True, eq=False)
class TensorForm:
    """The symmetry-allowed form of a property tensor in one point group.

    Attributes:
        kind: The kind as given (``"piezoelectric"``) or the Jahn symbol.
        jahn: The parsed :class:`JahnSymbol`.
        description: What the tensor is.
        letter: Component letter (``"d"``; ``"T"`` for a bare Jahn symbol).
        voigt: Matrix-notation factors: ``""`` (none), ``"strain"`` (Nye's
            ``d_iL = 2 d_ijk`` for L = 4-6) or ``"compliance"`` (Nye's
            ``s_LM`` with factors 2 and 4).
        layout: ``"vector"``, ``"matrix"`` (3x3), ``"3x6"``, ``"6x6"`` or
            ``"list"`` (components listed one by one).
        setting: The :class:`PointGroupSetting`.
        components: Names of all components that do not vanish by intrinsic
            symmetry, in matrix-notation order.
        basis: Array ``(n_independent, n_components)``: the reduced
            row-echelon basis of the allowed tensors in matrix notation.
        independent: Names of the independent components (the pivots).
        relations: ``"d24 = d15"``: every other nonzero component in terms
            of the independent ones.
        matrix: The matrix form, rows of entry texts (one row for a vector;
            for the ``list`` layout, ``(component, entry)`` pairs of the
            nonzero components).
        tensors: The allowed tensors as full Cartesian arrays of shape
            ``(n_independent, 3, ..., 3)``: tensor ``k`` is the one with
            independent component ``k`` equal to 1 and the others 0.
        forbidden_by_time_reversal: True for a time-odd tensor in a
            non-magnetic (grey) point group.
    """

    kind: str
    jahn: JahnSymbol
    description: str
    letter: str
    voigt: str
    layout: str
    setting: PointGroupSetting
    components: tuple
    basis: np.ndarray
    independent: tuple
    relations: tuple
    matrix: tuple
    tensors: np.ndarray
    forbidden_by_time_reversal: bool = False

    @property
    def n_independent(self) -> int:
        """Number of independent components."""
        return len(self.independent)


def _resolve_kind(kind: str):
    name = str(kind).strip()
    known = KINDS.get(name.lower())
    if known is not None:
        return name.lower(), known
    try:
        parse_jahn(name)
    except ValueError as exc:
        raise SystemExit(
            f"ERROR: --tensor {name}: {exc}.\n"
            f"Give one of {', '.join(KINDS)} or a Jahn symbol such as V, [V2], "
            "V[V2], [[V2]2], eV, e[V2] (e = axial, a = time-reversal odd)."
        ) from None
    symbol = "".join(name.split())
    same = [key for key, value in KINDS.items() if value.jahn == symbol and key != "raman"]
    description = "tensor given by its Jahn symbol"
    if same:
        description += f" (the symmetry type of: {', '.join(same)})"
    return symbol, _Kind(symbol, "T", "", description)


def tensor_form_of_operations(rotations, kind: str, *, time_reversal=None,
                              setting: PointGroupSetting | None = None) -> TensorForm:
    """The allowed form of a tensor under explicit Cartesian operations.

    Args:
        rotations: Orthogonal Cartesian matrices of a finite group,
            ``(n_ops, 3, 3)``.
        kind: Tensor name or Jahn symbol (see the module docstring).
        time_reversal: Optional booleans, one per rotation: True for an
            operation combined with time reversal (a magnetic point group).
            None means the grey group of a non-magnetic crystal, in which a
            time-odd tensor vanishes.
        setting: The :class:`PointGroupSetting` to record (built from the
            rotations alone when omitted).

    Returns:
        A :class:`TensorForm`.

    Raises:
        SystemExit: Unknown kind (``ValueError`` through ``crystod.group``).
        RuntimeError: The invariant count disagrees with the Molien series.
    """
    from .invariants import _rref, invariants_of_matrices

    name, spec = _resolve_kind(kind)
    jahn = parse_jahn(spec.jahn)
    rotations = np.asarray(rotations, dtype=float)
    if setting is None:
        setting = PointGroupSetting(
            symbol="?", schoenflies="?", crystal_system="", rotations=rotations,
            lattice_basis=np.eye(3), source="explicit operations", axes="")
    layout = _layout(jahn)
    rank = jahn.rank

    # orthonormal basis of the tensors with the intrinsic symmetry
    projector = _symmetrizer(jahn)
    values, vectors = np.linalg.eigh(projector)
    intrinsic = vectors[:, values > 0.5]                # (3^r, m)
    m = intrinsic.shape[1]
    forbidden = False
    if time_reversal is None:
        flags = [False] * len(rotations)
        if jahn.time_odd:
            # grey group: time reversal itself acts as -1 on the tensor
            forbidden = True
    else:
        flags = [bool(f) for f in time_reversal]
    matrices = []
    shaped = intrinsic.T.reshape([m] + [3] * rank)
    for rotation, primed in zip(rotations, flags if m else []):
        scale = 1.0
        if jahn.axial:
            scale *= float(np.sign(np.linalg.det(rotation)))
        if jahn.time_odd and primed:
            scale = -scale
        moved = np.array([_rotate(t, rotation) for t in shaped]).reshape(m, -1)
        matrices.append(scale * (intrinsic.T @ moved.T))   # D'(g) = Q^T D(g) Q
    if forbidden:
        matrices += [-matrix for matrix in matrices]
    invariant = []
    if m:
        result = invariants_of_matrices(matrices, [m], 1)
        invariant = [poly.coefficients for poly in result.polynomials[1]]
        if len(invariant) != result.counts[1]:
            raise RuntimeError("invariant count disagrees with the Molien series")
    full = [intrinsic @ np.asarray(c, dtype=float) for c in invariant]

    # components (orbits of index tuples) in display order
    cells = _cells(layout, rank)
    components, comp_of_cell, seen = [], {}, {}
    for row, column, indices in cells:
        orbit = _orbit(indices, jahn.permutations)
        if orbit is None:
            comp_of_cell[(row, column)] = None
            continue
        key = min(orbit)
        if key not in seen:
            seen[key] = len(components)
            components.append({
                "name": _cell_name(spec.letter, layout, row, column, indices),
                "indices": indices,
                "factor": _cell_factor(spec.voigt, layout, row, column),
                "shear": _shear_count(layout, row, column),
            })
        reference = components[seen[key]]
        # T[cell] = relative * T[reference cell] by the intrinsic symmetry
        relative = _orbit(reference["indices"], jahn.permutations)[indices]
        ratio = _cell_factor(spec.voigt, layout, row, column) / reference["factor"]
        comp_of_cell[(row, column)] = (seen[key], relative * ratio)
    names = [c["name"] for c in components]
    values_matrix = np.array([
        [c["factor"] * tensor.reshape([3] * rank)[c["indices"]] for c in components]
        for tensor in full
    ]).reshape(len(full), len(components))
    # pivots: normal components before shear ones, and among them the
    # longitudinal ones (all tensor indices equal) first, then the
    # matrix-notation order. This is Nye's choice: c11, c12 independent and
    # c66 = (c11-c12)/2; d22 with d21 = -d22 and d16 = -2d22 in 3m.
    order = sorted(range(len(names)), key=lambda c: (
        components[c]["shear"], len(set(components[c]["indices"])) > 1, c))
    rows = np.zeros((len(full), len(names)))
    pivots: list[int] = []
    if len(full):
        reduced, reduced_pivots = _rref(values_matrix[:, order])
        rows[:, order] = reduced
        pivots = [order[p] for p in reduced_pivots]
        by_position = np.argsort(pivots)
        rows = rows[by_position]
        pivots = [pivots[i] for i in by_position]
    rows[np.abs(rows) < 1e-10] = 0.0
    independent = tuple(names[p] for p in pivots)

    def expression(component: int) -> str:
        return combination_text([(rows[j, component], independent[j])
                                 for j in range(len(pivots))])

    relations = tuple(f"{names[c]} = {expression(c)}" for c in range(len(names))
                      if c not in pivots and np.any(np.abs(rows[:, c]) > 1e-10))

    def entry(row: int, column: int) -> str:
        found = comp_of_cell.get((row, column))
        if found is None:
            return "0"
        component, factor = found
        return combination_text([(factor * rows[j, component], independent[j])
                                 for j in range(len(pivots))])

    if layout == "list":
        matrix = tuple((_cell_name(spec.letter, layout, r, c, t), entry(r, c))
                       for r, c, t in cells if entry(r, c) != "0")
    else:
        n_rows = 1 + max(r for r, _, _ in cells)
        n_columns = 1 + max(c for _, c, _ in cells)
        matrix = tuple(tuple(entry(r, c) for c in range(n_columns)) for r in range(n_rows))

    # the tensor with independent component k = 1, the others 0
    tensors = np.zeros((len(pivots),) + (3,) * rank)
    if len(full):
        solve = np.linalg.lstsq(values_matrix[:, pivots].T, np.eye(len(pivots)),
                                rcond=None)[0]                 # (n_basis, k)
        stacked = np.array(full)                               # (n_basis, 3^r)
        tensors = (solve.T @ stacked).reshape((len(pivots),) + (3,) * rank)
        tensors[np.abs(tensors) < 1e-12] = 0.0
    return TensorForm(
        kind=name, jahn=jahn, description=spec.description, letter=spec.letter,
        voigt=spec.voigt, layout=layout, setting=setting, components=tuple(names),
        basis=rows, independent=independent, relations=relations, matrix=matrix,
        tensors=tensors, forbidden_by_time_reversal=forbidden,
    )


def tensor_form(point_group: str | None = None, kind: str = "dielectric", *,
                space_group=None, cell=None, symprec: float = 1e-5) -> TensorForm:
    """The symmetry-allowed form of a property tensor (Neumann's principle).

    Args:
        point_group: Hermann-Mauguin point-group symbol (``"4mm"``); or give
            ``space_group`` or ``cell`` instead.
        kind: ``dielectric``, ``pyroelectric``, ``piezoelectric``,
            ``elastic``, ``compliance``, ``gyration`` or a Jahn symbol
            (``"V[V2]"``, ``"e[V2]"``, ``"[[V2]2]"``; prefix ``a`` for a
            time-odd tensor, which vanishes in every non-magnetic point
            group).  ``raman`` gives the symmetric second-rank form; the
            per-irrep Raman tensors are :func:`raman_forms`.
        space_group: Space-group symbol or number; its point group in the
            default setting.
        cell: POSCAR path or ``PhonopyAtoms``; its point group in the
            standard conventional setting (the axes of
            :attr:`PointGroupSetting.input_axes`).
        symprec: Symmetry tolerance of the structure route.

    Returns:
        A :class:`TensorForm` (``independent``, ``relations``, ``matrix``).

    Raises:
        SystemExit: Unknown point group or kind, or not exactly one group
            source (``ValueError`` through ``crystod.group``).

    Example:
        >>> from crystod import group
        >>> form = group.tensor_form("6/mmm", "elastic")
        >>> form.independent, form.relations[-1]
        (('c11', 'c12', 'c13', 'c33', 'c44'), 'c66 = (c11-c12)/2')
    """
    setting = point_group_setting(point_group, space_group=space_group, cell=cell,
                                  symprec=symprec)
    return tensor_form_of_operations(setting.rotations, kind, setting=setting)


# ------------------------------------------------------------ Raman forms


def _table_alignment(setting: PointGroupSetting):
    """Orthogonal U with {U R U^T} = the phonopy table operations, and the
    table class of every operation of ``setting``."""
    table_setting = point_group_setting(setting.symbol)
    table = _keyed({tuple(np.round(r, 8).ravel()): c
                    for r, c in zip(table_setting.rotations, table_setting.class_index)})
    permutations = [np.eye(3)[list(p)] for p in itertools.permutations(range(3))]
    for degrees in range(0, 360, 15):
        angle = math.radians(degrees)
        rz = np.array([[math.cos(angle), -math.sin(angle), 0.0],
                       [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]])
        for permutation in permutations:
            u = rz @ permutation
            classes = []
            for rotation in setting.rotations:
                key = tuple(np.round(u @ rotation @ u.T + 0.0, 8).ravel())
                key = tuple(0.0 if abs(v) < 1e-8 else v for v in key)
                if key not in table:
                    break
                classes.append(table[key])
            else:
                if len(classes) == len(table):
                    return u, classes
    raise RuntimeError(f"cannot align the operations with the {setting.symbol} table")


def _keyed(table: dict) -> dict:
    return {tuple(0.0 if abs(v) < 1e-8 else v for v in key): c for key, c in table.items()}


def raman_forms(point_group: str | None = None, *, space_group=None,
                setting: PointGroupSetting | None = None) -> list:
    """Raman tensors of every Raman-active irrep of a point group.

    The point-group route of ``crystod-phonon --raman-tensor``: the same
    projection (:func:`crystod.phonon_activity.raman_tensor_basis`, Loudon's
    forms with shared constants for a repeated degenerate irrep) on the
    operations of the point group, with the Mulliken labels of phonopy's
    character table, for every irrep (not only those of the phonons of a
    structure).

    Args:
        point_group: Hermann-Mauguin symbol; or ``space_group`` (its
            point group in the default setting); or a ready ``setting``.
        space_group: Space-group symbol or number.
        setting: A :class:`PointGroupSetting`.

    Returns:
        :class:`crystod.phonon_activity.RamanTensors` records in the order of
        the character table, labelled with Mulliken symbols, in the Cartesian
        axes of the setting.

    Example:
        >>> from crystod.tensor_form import raman_forms
        >>> [(r.label, len(r.tensors)) for r in raman_forms("m-3m")]
        [('A1g', 1), ('Eg', 2), ('T2g', 3)]
    """
    from .phonon_activity import _GammaIrreps, _raman_records

    if setting is None:
        setting = point_group_setting(point_group, space_group=space_group)
    table = _character_table(setting.symbol)
    if setting.class_index:
        classes = list(setting.class_index)
    else:
        _, classes = _table_alignment(setting)
    labels = list(table["character_table"])
    # phonopy writes the table of point group 1 with bare values ("A": 1)
    characters = np.array([[np.atleast_1d(table["character_table"][label])[c]
                            for c in classes] for label in labels], dtype=complex)
    identity = int(np.flatnonzero([np.allclose(r, np.eye(3)) for r in setting.rotations])[0])
    gamma = _GammaIrreps(
        rotations=setting.rotations, cartesian=setting.rotations, labels=tuple(labels),
        characters=characters,
        dimensions=tuple(int(round(chi[identity].real)) for chi in characters),
        counts=(1,) * len(labels), vibrations=None,
    )
    records = _raman_records(gamma, only_present=False)
    order = {label: n for n, label in enumerate(labels)}
    return sorted(records, key=lambda record: order.get(record.label, len(order)))


# ------------------------------------------------------------ printing


def _direction(vector, basis) -> tuple[str, tuple]:
    """A Cartesian direction as the lattice direction ``[uvw]`` (with the
    axis name when it is x, y or z) and its sort key (x, y, z first, then
    the shortest lattice directions)."""
    lattice = np.linalg.solve(basis, vector)
    lattice = lattice / np.max(np.abs(lattice))
    fracs = [Fraction(float(v)).limit_denominator(12) for v in lattice]
    scale = math.lcm(*[f.denominator for f in fracs])
    ints = [int(f * scale) for f in fracs]
    common = math.gcd(*[abs(i) for i in ints]) or 1
    ints = [i // common for i in ints]
    if next(i for i in ints if i) < 0:
        ints = [-i for i in ints]
    text = "[" + "".join(str(i) for i in ints) + "]"
    unit = vector / np.linalg.norm(vector)
    named = [axis for axis in range(3) if abs(abs(unit[axis]) - 1.0) < 1e-8]
    if named:
        text += f" ({'xyz'[named[0]]})"
        return text, (0, named[0])
    return text, (1, sum(1 for i in ints if i), sum(abs(i) for i in ints),
                  tuple(-i for i in ints))


def _directions_text(keys, basis) -> str:
    found = sorted((_direction(np.array(key), basis) for key in keys),
                   key=lambda item: item[1])
    return ", ".join(text for text, _ in found)


def symmetry_elements(setting: PointGroupSetting) -> list[str]:
    """The rotation axes and mirror normals of a point group as text lines.

    Args:
        setting: A :class:`PointGroupSetting`.

    Returns:
        Lines such as ``"4 || [001] (z)"`` (the highest rotation or
        rotoinversion on each axis) and ``"m perpendicular to [100] (x), [010] (y)"``.
    """
    axes: dict = {}
    mirrors = []

    def canonical(vector):
        vector = vector / np.linalg.norm(vector)
        first = next(v for v in vector if abs(v) > 1e-8)
        return tuple(np.round(vector * np.sign(first), 8) + 0.0)

    for rotation in setting.rotations:
        det = float(np.linalg.det(rotation))
        proper = rotation * det
        if np.allclose(proper, np.eye(3)):
            continue
        angle = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(proper) - 1) / 2))))
        n = int(round(360.0 / angle))
        values, vectors = np.linalg.eig(proper)
        axis = np.real(vectors[:, int(np.argmin(np.abs(values - 1.0)))])
        key = canonical(axis)
        entry = axes.setdefault(key, [1, 0])
        if det > 0:
            entry[0] = max(entry[0], n)
        elif n == 2:
            mirrors.append(key)
        else:
            entry[1] = max(entry[1], n)
    groups: dict = {}
    for key, (n_proper, n_improper) in axes.items():
        if n_improper > n_proper or (n_improper == n_proper == 3):
            token = f"-{n_improper}"
        elif n_proper > 1:
            token = str(n_proper)
        else:
            continue
        groups.setdefault(token, []).append(key)
    order = ["6", "-6", "4", "-4", "3", "-3", "2"]
    lines = []
    for token in sorted(groups, key=order.index):
        lines.append(f"{token} || {_directions_text(groups[token], setting.lattice_basis)}")
    if mirrors:
        lines.append("m perpendicular to "
                     + _directions_text(set(mirrors), setting.lattice_basis))
    if setting.centrosymmetric:
        lines.append("-1 (inversion)")
    return lines


def _vector_text(vector) -> str:
    return "(" + ", ".join(f"{round(float(v), 4) + 0.0:.4f}" for v in vector) + ")"


def format_point_group(setting: PointGroupSetting) -> list[str]:
    """The ``* Point group *`` block.

    Args:
        setting: A :class:`PointGroupSetting`.

    Returns:
        The block lines (title first).
    """
    if setting.centrosymmetric:
        nature = "centrosymmetric"
    else:
        traits = [t for t, on in (("polar", setting.polar), ("chiral", setting.chiral)) if on]
        nature = ", ".join(["non-centrosymmetric"] + traits)
    head = f"{setting.symbol} ({setting.schoenflies})"
    if setting.space_group:
        head += f", point group of {setting.space_group}"
    lines = ["* Point group *",
             f"  {head}, order {setting.order}, {setting.crystal_system}, "
             f"Laue class {LAUE_CLASS[setting.symbol]}, {nature}",
             f"  source: {setting.source}",
             f"  axes: {setting.axes}"]
    if setting.input_axes is not None:
        lines.append("  tensor axes in the Cartesian frame of the input file: "
                     + ", ".join(f"{name} = {_vector_text(v)}"
                                 for name, v in zip("xyz", setting.input_axes)))
    lines += [f"  {line}" for line in symmetry_elements(setting)]
    return lines


_MATRIX_NOTATION = {
    "strain": "matrix notation (Nye): d_iL, L = 1..6 for jk = 11, 22, 33, 23, 13, 12; "
              "d_iL = d_ijk for L = 1-3 and 2 d_ijk for L = 4-6",
    "compliance": "matrix notation (Nye): s_LM, L, M = 1..6 for ij, kl = 11, 22, 33, "
                  "23, 13, 12; s_LM = s_ijkl, 2 s_ijkl or 4 s_ijkl when none, one or "
                  "both of L, M are 4-6",
}


def _tensor_lines(form: TensorForm) -> list[str]:
    jahn = form.jahn
    nature = (f"{'axial' if jahn.axial else 'polar'}, time-reversal "
              f"{'odd' if jahn.time_odd else 'even'}")
    lines = ["* Tensor *",
             f"  kind: {form.description}",
             f"  Jahn symbol: {jahn.symbol} ({nature})",
             f"  rank: {jahn.rank}",
             "  intrinsic symmetry: " + ("; ".join(jahn.intrinsic) or "none")]
    if form.voigt in _MATRIX_NOTATION:
        lines.append(f"  {_MATRIX_NOTATION[form.voigt]}")
    elif form.layout in ("3x6", "6x6"):
        lines.append("  matrix notation: L = 1..6 for the index pairs 11, 22, 33, 23, "
                     "13, 12 (tensor components, no factors)")
    return lines


def _table(rows: list[list[str]]) -> list[str]:
    widths = [max(len(row[c]) for row in rows) for c in range(len(rows[0]))]
    return ["  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip()
            for row in rows]


def format_tensor_form(form: TensorForm) -> str:
    """The report ``crystod-group --tensor`` prints.

    Args:
        form: A :class:`TensorForm`.

    Returns:
        The blocks ``* Point group *``, ``* Tensor *``, ``* Independent
        components *``, ``* Matrix form *`` and ``* Relations *``.
    """
    lines = format_point_group(form.setting) + [""] + _tensor_lines(form)
    lines += ["", "* Independent components *",
              f"  {form.n_independent}" + (f": {', '.join(form.independent)}"
                                           if form.independent else "")]
    lines += ["", "* Matrix form *"]
    if form.layout == "list":
        if form.matrix:
            lines += _table([["component", "value"]] + [list(pair) for pair in form.matrix])
        else:
            lines.append("every component is zero")
    elif form.layout == "vector":
        lines += _table([[form.letter, "1", "2", "3"], [""] + list(form.matrix[0])])
    else:
        n_columns = len(form.matrix[0])
        rows = [[form.letter] + [str(c + 1) for c in range(n_columns)]]
        rows += [[str(r + 1)] + list(row) for r, row in enumerate(form.matrix)]
        lines += _table(rows)
    lines += ["", "* Relations *"]
    if form.forbidden_by_time_reversal:
        lines.append(f"  time-reversal odd: forbidden in the non-magnetic point group "
                     f"{form.setting.symbol}1' (time reversal is a symmetry); this mode "
                     "takes non-magnetic point groups only")
    elif not form.independent:
        lines.append("  every component is zero")
    else:
        lines += [f"  {relation}" for relation in form.relations]
        if not form.relations:
            lines.append("  none: the nonzero components are independent")
        n_zero = sum(1 for c in range(len(form.components))
                     if not np.any(np.abs(form.basis[:, c]) > 1e-10))
        if n_zero:
            lines.append("  every other component is zero")
    return "\n".join(lines)


def format_raman_forms(records, setting: PointGroupSetting, *, axes_line=None,
                       mulliken=None) -> str:
    """The ``* Raman tensors *`` block.

    Args:
        records: :class:`crystod.phonon_activity.RamanTensors` records
            (:func:`raman_forms`, or ``gamma_raman_tensors`` of a structure).
        setting: The point group the tensors refer to.
        axes_line: The axes the tensors are written in (default: those of
            the setting).
        mulliken: Optional ``{ISO-IR label: Mulliken symbol}`` map.

    Returns:
        The block as text.
    """
    from .phonon_activity import format_raman_tensors

    lines = ["* Raman tensors *",
             f"  {axes_line or 'Cartesian axes of the point group above'}"]
    lines += format_raman_tensors(records, mulliken=mulliken)[1:]
    return "\n".join(lines)


def _raman_report(setting: PointGroupSetting, cell, symprec: float) -> str:
    form = tensor_form_of_operations(setting.rotations, "raman", setting=setting)
    head = format_point_group(setting) + [""] + _tensor_lines(form)
    if cell is not None:
        from .phonon_activity import gamma_raman_tensors, mulliken_symbols

        records = gamma_raman_tensors(cell, symprec=symprec)
        block = format_raman_forms(
            records, setting, mulliken=mulliken_symbols(cell, symprec=symprec),
            axes_line="Cartesian axes of the input cell (not the standard axes above); "
                      "the Gamma phonon irreps of the structure, ISO-IR [Mulliken]")
    else:
        axes_line = ("Cartesian axes of the point group above; every Raman-active "
                     "irrep, Mulliken labels of the phonopy character table")
        if not setting.class_index:
            u, _ = _table_alignment(setting)
            if not np.allclose(u, np.eye(3)):
                axes_line += (f" (whose {setting.symbol} setting differs from this one "
                              "by a change of axes: the labels are carried over)")
        block = format_raman_forms(raman_forms(setting=setting), setting,
                                   axes_line=axes_line)
    return "\n".join(head) + "\n\n" + block


def build_parser():
    from argparse import ArgumentParser

    parser = ArgumentParser(prog="crystod-group --tensor",
                            description="Symmetry-allowed form of a property tensor.")
    parser.add_argument("--kind", required=True)
    parser.add_argument("--point-group", dest="point_group", default=None)
    parser.add_argument("--space-group", dest="space_group", default=None)
    parser.add_argument("--cell", default=None)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Command-line entry (dispatched from ``crystod-group --tensor``)."""
    args = build_parser().parse_args(argv)
    kind, _ = _resolve_kind(args.kind)
    try:
        setting = point_group_setting(args.point_group, space_group=args.space_group,
                                      cell=args.cell, symprec=args.tolerance)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"ERROR: {exc}") from None
    if kind == "raman":
        print(_raman_report(setting, args.cell, args.tolerance))
        return
    form = tensor_form_of_operations(setting.rotations, kind, setting=setting)
    print(format_tensor_form(form))


if __name__ == "__main__":
    main()
