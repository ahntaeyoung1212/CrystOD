"""Parent-irrep labels of unfolded phonon levels (``crystod-phonon --unfold``).

The unfolded vectors u_J(k) of :mod:`crystod.phonon_unfold` (child mode J
projected onto the Bloch states of the parent at the parent k) are decomposed
into the irreps of the little group of k of the parent with CrystOD's own
machinery: :class:`crystod.vibration_modes.SymmetryOnlyVibrations` on the
parent primitive cell as it is (``standardize=False``, ISO-IR frame of the
parent file) gives the spgrep small irreps and the displacement
representation Gamma(g) of the little co-group, ``get_irrep_labels`` their
ISO-IR names, and :func:`crystod.phonon_activity.mulliken_symbols` the
Mulliken symbols at GM.  The labels are those of ``crystod-phonon --irreps``
on the parent POSCAR (with Ti at the origin of CaTiO3 the tilts are R4+ and
M3+).

For every irrep i present in the displacement representation the isotypic
projector is ``Pi_i = (d_i / |G0|) sum_g conj(chi_i(g)) Gamma(g)`` (checked:
complete, idempotent, Hermitian, mutually orthogonal, commuting with the
masses).  A pair of complex irreps that time reversal makes degenerate
(Wigner's case (c), :func:`crystod.extract_irrep._time_reversal`) is one unit
with the projector ``Pi_a + Pi_b`` and the name ``S1+S2``.  The weight of unit
u in child mode J is ``P_J^(u)(k) = || Pi_u R u_J(k) ||^2`` (R: the rigid
rotation from the child to the parent Cartesian frame); ``sum_u P_J^(u) =
P_J``.

Labels belong to degenerate groups g of child modes (``|df| <= 1e-4`` THz): with
``P_g = sum_{J in g} P_J`` and ``f_u = sum_{J in g} P_J^(u) / P_g`` the group
gets the unit with ``f_u >= 0.5``; otherwise the smallest set of units that
the child symmetry cannot separate with a summed fraction >= 0.5, written
``Z1/Z3``; otherwise ``mixed``; ``-`` when ``P_g`` is below the weight cut.
Units u and v cannot be separated when one child level can hold weight in
both: an irrep of H(k) (the child operations that fix k) in u and one in v
occur together in one irrep of the child little group of K = M k, or in its
time-reversal partner.  By Mackey's theorem this is a non-zero multiplicity
``<Res_H Ind_H^G(K) u, v>_H`` (plus the same with the induced representation
conjugated by a child operation that takes K to -K): the child operations of
the little group of K that take k to another arm folding onto K, and time
reversal, join the partners of a degenerate child level that carry different
irreps of H(k) (DT1 and DT5 at k = (0, 1/4, 0) of Pnma CaTiO3, which folds
onto the child Y point).

Along a straight run of path points with the same little group the ISO-IR
names are computed at the first point and copied to the others by matching
the characters stripped of their Bloch phase (``chi(g) exp(+2 pi i k.t_g)``,
constant along a line), and checked again at the last point of the run.

Conventions: those of :mod:`crystod.phonon_unfold` (rows = lattice vectors,
``A_child = M @ A_parent``, ``k_child = M @ k_parent``, THz, imaginary
negative).  CPU only (numpy): the projectors are at most a few tens square.
"""

from __future__ import annotations

import re
import time
import warnings
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from .spglib_compat import ensure_spglib_compat

ensure_spglib_compat()

from .phonon_unfold import (  # noqa: E402
    CHUNK,
    DEGENERACY_TOLERANCE,
    FRACTION_MIN,
    PROJECTOR_TOLERANCE,
    SYMPREC,
    WEIGHT_MIN,
    _cell_tuple,
    _check_child_cell,
    _operations,
    _run_qpoints,
    _vec,
    degenerate_groups,
    rotate_unfolded_vectors,
)

_LABEL_RE = re.compile(r"^(?P<name>.*?)\((?P<dim>\d+)\)$")
_STRIPPED_TOLERANCE = 1e-6  # characters of one irrep at two points of a straight run


def split_label(label: str):
    """``'R4+(3)'`` -> ``('R4+', 3)``; a label without the dimension suffix -> ``(label, None)``."""
    m = _LABEL_RE.match(label.strip())
    if m is None:
        return label.strip(), None
    return m.group("name"), int(m.group("dim"))


def kname_of(name: str):
    """k-vector type of an ISO-IR irrep name: ``'GM4-'`` -> ``'GM'``, ``'irrep_2'`` -> None."""
    m = re.match(r"^([A-Z]+)\d", name)
    return m.group(1) if m else None


def irrep_sort_key(name: str):
    """Natural order of the irreps of one k type: index, then '+' before '-' (GM1+, GM1-, ...)."""
    m = re.match(r"^([A-Z]+)(\d+)([+-]?)", name)
    if m is None:
        return (1, name, 0, 0)
    return (0, m.group(1), int(m.group(2)), {"+": 0, "": 1, "-": 2}[m.group(3)])


@dataclass
class IrrepProjectorSet:
    """The isotypic projectors of the parent displacement representation at one k.

    Attributes:
        k: (3,) parent fractional k.
        little_ops: (``|G0|``,) indices of the little co-group operations in the
            operations of :class:`ParentIrrepBasis`.
        order: ``|G0|``.
        irreps: The spgrep small irreps at k (all of them).
        characters: (n_irrep, ``|G0|``) their characters.
        stripped: (n_irrep, ``|G0|``) characters times ``exp(+2 pi i k.t_g)``.
        present: Indices of the irreps that occur in the displacement representation.
        projectors: (n_present, 3 n, 3 n) orthogonal projectors (parent atom
            order, Cartesian components of the parent frame, tau phase).
        dims, multiplicities: (n_present,) d_i and n_i (trace = d_i n_i).
        checks: Largest deviations of the projector identities.
        names: ISO-IR names of all irreps (no ``(dim)``); None before labelling.
        kname: ISO-IR k-vector type of k.
        labelled_directly: The names come from ``get_irrep_labels`` at this k
            (not copied along a run).
        units: Names of the units (an irrep, or a time-reversal pair ``S1+S2``),
            natural order; ``unit_members`` their positions in ``present``;
            ``unit_characters`` (n_units, ``|G0|``); ``unit_traces`` (n_units,).
        tr_pairs: The time-reversal pairs that form units.
        tr_failed: Wigner's test could not be made at k.
        classes: Units the child symmetry cannot separate (tuples of unit
            positions, classes of more than one unit), set by
            :func:`_inseparable_classes`; ``class_deviation`` the largest
            distance of the multiplicities behind them from integers.
    """

    k: np.ndarray
    little_ops: np.ndarray
    order: int
    irreps: list
    characters: np.ndarray
    stripped: np.ndarray
    present: np.ndarray
    projectors: np.ndarray
    dims: np.ndarray
    multiplicities: np.ndarray
    checks: dict
    names: list | None = None
    kname: str | None = None
    labelled_directly: bool = False
    units: list | None = None
    unit_members: list | None = None
    unit_characters: np.ndarray | None = None
    unit_traces: np.ndarray | None = None
    tr_pairs: list = field(default_factory=list)
    tr_failed: bool = False
    classes: list | None = None
    class_deviation: float = 0.0

    @property
    def present_names(self) -> list:
        return [self.names[i] for i in self.present]


class ParentIrrepBasis:
    """CrystOD irrep projectors of the parent primitive cell, cached per k.

    Args:
        cell: The parent primitive cell of the unfolding (PhonopyAtoms; it
            must be primitive as given).
        input_cell: The cell of the parent file, whose ISO-IR frame names the
            irreps (default ``cell``).
        symprec: spglib tolerance.

    Attributes:
        vib: The :class:`crystod.vibration_modes.SymmetryOnlyVibrations`.
        rotations, translations: Space-group operations of ``cell`` (spglib order).
        mulliken: ``{GM irrep: Mulliken symbol}``; empty when unavailable.
        notes: NOTE lines (Mulliken symbols missing, warnings of the tables).
        child_ops: Indices of the operations that are child operations (set by
            :func:`irrep_resolve`).
        child_affine: The child operations in the parent setting with their
            conjugation table (:func:`_child_affine_operations`, set by
            :func:`irrep_resolve`).
    """

    def __init__(self, cell, input_cell=None, symprec: float = SYMPREC):
        from .phonon_activity import mulliken_symbols
        from .vibration_modes import SymmetryOnlyVibrations

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.vib = SymmetryOnlyVibrations(
                cell, symprec=symprec, standardize=False, input_cell=input_cell, quiet=True
            )
        primitive = self.vib.primitive_cell
        if (
            len(primitive) != len(cell)
            or np.abs(np.array(primitive.cell) - np.array(cell.cell)).max() > 1e-10
            or np.abs(np.array(primitive.scaled_positions) - np.array(cell.scaled_positions)).max()
            > 1e-10
        ):
            raise SystemExit(
                "ERROR: internal error: the irrep analysis changed the parent primitive cell."
            )
        self.cell = cell
        self.input_cell = cell if input_cell is None else input_cell
        self.lattice, self.frac, self.numbers = _cell_tuple(cell)
        self.masses = np.array(cell.masses, dtype=float)
        self.n_atoms = len(cell)
        self.symprec = symprec
        self.rotations = np.asarray(self.vib.rotations, dtype=np.int64)
        self.translations = np.asarray(self.vib.translations, dtype=float)
        self.space_group_number = int(self.vib.spglib_dataset["number"])
        self.notes: list = []
        self.child_ops = None
        self.child_affine = None
        self._sets: dict = {}
        self.label_time = 0.0
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.mulliken = mulliken_symbols(self.vib)
            for warning in caught:
                if not issubclass(warning.category, DeprecationWarning):
                    self.notes.append("NOTE: " + " ".join(str(warning.message).split()))
        except Exception as exc:  # the brackets are an annotation: say why they are missing
            self.mulliken = {}
            self.notes.append(
                "NOTE: no Mulliken symbols: " + (" ".join(str(exc).split()) or type(exc).__name__)
            )

    # -- projectors --------------------------------------------------------
    def build(self, k):
        """``(projector set at k (cached), displacement representation (|G0|, 3n, 3n))``."""
        from .runtime_compat import get_character

        k = np.asarray(k, dtype=float).reshape(3)
        key = tuple(np.round(k, 10) + 0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            irreps, rep, ops = self.vib.get_vibration_rep([float(v) for v in k])
        ops = np.asarray(ops, dtype=np.int64)
        pset = self._sets.get(key)
        if pset is not None:
            return pset, rep
        order = len(rep)
        n3 = rep.shape[1]
        characters = np.array([np.asarray(get_character(ir), dtype=complex) for ir in irreps])
        stripped = characters * np.exp(2j * np.pi * (self.translations[ops] @ k))[None, :]
        present, projectors, dims, mults = [], [], [], []
        for i, ir in enumerate(irreps):
            d = int(ir.shape[1])
            P = (d / order) * np.einsum("g,gij->ij", characters[i].conj(), rep)
            trace = float(np.real(np.trace(P)))
            if trace < 0.5:  # irrep absent from the displacement representation
                continue
            n_i = trace / d
            if abs(n_i - round(n_i)) > 1e-6:
                raise SystemExit(
                    f"ERROR: internal error: non-integer multiplicity {n_i:.6f} of an irrep at k = "
                    f"{_vec(k)}."
                )
            present.append(i)
            projectors.append(P)
            dims.append(d)
            mults.append(int(round(n_i)))
        projectors = np.array(projectors)
        mass3 = np.repeat(self.masses, 3)
        cross = 0.0
        for a in range(len(projectors)):
            for b in range(a + 1, len(projectors)):
                cross = max(cross, float(np.abs(projectors[a] @ projectors[b]).max()))
        checks = {
            "completeness": float(np.abs(projectors.sum(axis=0) - np.eye(n3)).max()),
            "idempotency": float(max(np.abs(P @ P - P).max() for P in projectors)),
            "hermiticity": float(max(np.abs(P - P.conj().T).max() for P in projectors)),
            "orthogonality": cross,
            "mass_commutator": float(
                max(np.abs(P * mass3[None, :] - mass3[:, None] * P).max() for P in projectors)
            ),
        }
        worst = max(checks, key=checks.get)
        if checks[worst] > PROJECTOR_TOLERANCE:
            raise SystemExit(
                f"ERROR: internal error: the irrep projectors at k = {_vec(k)} fail the {worst} "
                f"check ({checks[worst]:.1e})."
            )
        pset = IrrepProjectorSet(
            k=k.copy(),
            little_ops=ops,
            order=order,
            irreps=list(irreps),
            characters=characters,
            stripped=stripped,
            present=np.array(present, dtype=np.int64),
            projectors=projectors,
            dims=np.array(dims, dtype=np.int64),
            multiplicities=np.array(mults, dtype=np.int64),
            checks=checks,
        )
        self._sets[key] = pset
        return pset, rep

    def projector_set(self, k) -> IrrepProjectorSet:
        """The projector set at k, labelled and with its time-reversal units."""
        pset, _rep = self.build(k)
        if pset.names is None:
            self.label_directly(pset)
        if pset.units is None:
            self.make_units(pset)
        return pset

    # -- names ---------------------------------------------------------------
    def direct_labels(self, pset: IrrepProjectorSet):
        """``(names of all irreps, k type)`` from ``get_irrep_labels`` at the point of ``pset``."""
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            labels = self.vib.get_irrep_labels(
                [float(v) for v in pset.k], pset.irreps, pset.little_ops
            )
        names = [split_label(label)[0] for label in labels]
        kname = (
            getattr(self.vib, "isoir_qpoint_name", None)
            if getattr(self.vib, "labels_from_isoir", False)
            else None
        )
        knames = {kname_of(names[i]) for i in pset.present} - {None}
        if kname is None or kname not in knames:
            kname = sorted(knames)[0] if knames else "k"
        self.label_time += time.perf_counter() - t0
        return names, kname

    def label_directly(self, pset: IrrepProjectorSet) -> None:
        pset.names, pset.kname = self.direct_labels(pset)
        pset.labelled_directly = True
        pset.units = None

    def make_units(self, pset: IrrepProjectorSet) -> None:
        """Group the present irreps into units; time-reversal pairs (Wigner's case (c)) are one."""
        from .extract_irrep import _time_reversal

        group = SimpleNamespace(
            kpoint=pset.k,
            rotations=self.rotations[pset.little_ops],
            translations=self.translations[pset.little_ops],
            characters=pset.characters,
            names=list(pset.names),
            order=pset.order,
            spin_rotations=None,
        )
        reversal = _time_reversal(group, self.lattice, self.frac, self.numbers, self.symprec)
        present_names = pset.present_names
        position = {name: a for a, name in enumerate(present_names)}
        units, members, pairs, done = [], [], [], set()
        for a, name in enumerate(present_names):
            if a in done:
                continue
            partner = None
            if reversal and name in reversal and reversal[name][0] == "c":
                other = reversal[name][1]
                if other != name and other in position and position[other] not in done:
                    partner = other
            if partner is None:
                units.append(name)
                members.append((a,))
                done.add(a)
            else:
                pair = tuple(sorted((name, partner), key=irrep_sort_key))
                units.append("+".join(pair))
                members.append(tuple(position[x] for x in pair))
                pairs.append(pair)
                done |= {a, position[partner]}
        order = sorted(range(len(units)), key=lambda i: irrep_sort_key(units[i]))
        pset.units = [units[i] for i in order]
        pset.unit_members = [members[i] for i in order]
        chars = pset.characters[pset.present]
        pset.unit_characters = np.array([chars[list(m)].sum(axis=0) for m in pset.unit_members])
        pset.unit_traces = np.array(
            [int(sum(pset.dims[j] * pset.multiplicities[j] for j in m)) for m in pset.unit_members]
        )
        pset.tr_pairs = pairs
        pset.tr_failed = reversal is None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def irrep_weights(vectors, projectors) -> np.ndarray:
    """``||Pi_i v||^2`` for the rows v of ``vectors`` (n_vec, 3n): shape (n_vec, n_projector)."""
    v = np.atleast_2d(np.asarray(vectors, dtype=complex))
    pv = np.einsum("iab,nb->nia", np.asarray(projectors), v)
    return np.sum(np.abs(pv) ** 2, axis=-1)


def _unit_weights(pset: IrrepProjectorSet, irrep_w) -> np.ndarray:
    """Weights per unit (sum over the members of each unit), shape (n_vec, n_units)."""
    return np.stack([irrep_w[:, list(m)].sum(axis=1) for m in pset.unit_members], axis=1)


def child_operations_in_parent(mapping, basis: ParentIrrepBasis, symprec: float = SYMPREC):
    """The space-group operations of the child, written in the parent setting.

    A child operation (W_c, w_c) becomes, with ``x_parent = M.T x_child + p``,
    ``W_p = M.T W_c inv(M.T)`` and ``w_p = M.T w_c + p - W_p p``; each must be
    an operation of the parent (same rotation, translation equal modulo the
    parent lattice).

    Returns:
        ``(sorted indices of those parent operations, number of child
        operations, largest translation mismatch)``.
    """
    rotations, translations = _operations(mapping.child_cell, symprec)
    MT = mapping.M.T.astype(float)
    MT_inv = np.linalg.inv(MT)
    p = np.asarray(mapping.origin_shift, dtype=float)
    found, worst = set(), 0.0
    for Wc, wc in zip(rotations, translations):
        Wp = MT @ Wc @ MT_inv
        Wi = np.rint(Wp).astype(np.int64)
        best = None
        if np.abs(Wp - Wi).max() < 1e-8:
            wp = MT @ wc + p - Wi @ p
            for j in np.where(np.all(basis.rotations == Wi[None], axis=(1, 2)))[0]:
                dt = basis.translations[j] - wp
                dev = float(np.abs(dt - np.rint(dt)).max())
                if dev < 1e-3 and (best is None or dev < best[1]):
                    best = (int(j), dev)
        if best is None:
            raise SystemExit(
                "ERROR: internal error: a child operation is not a parent operation in this "
                "setting."
            )
        found.add(best[0])
        worst = max(worst, best[1])
    return sorted(found), len(rotations), worst


def _child_positions(pset: IrrepProjectorSet, child_ops) -> list:
    """Positions (in the little co-group of k) of the child operations that belong to it: H(k)."""
    position = {int(j): a for a, j in enumerate(pset.little_ops)}
    return [position[j] for j in child_ops if j in position]


def _child_affine_operations(mapping, basis: ParentIrrepBasis, symprec: float = SYMPREC):
    """The child operations in the parent setting, with their conjugation table.

    A child operation (W_c, w_c) is ``W = M.T W_c inv(M.T)``, ``w = M.T w_c + p
    - W p`` in parent fractional coordinates (the coset representatives of the
    child modulo its own lattice).  For every pair (x, h) the table holds the
    child coset of ``y = x^-1 h x`` and the parent lattice vector ``t`` with
    ``y = (W_j, w_j + t)``, (W_j, w_j) the parent operation of that coset, so
    that the character of y at k is ``chi(j) exp(-2 pi i k.t)`` (the
    convention of the spgrep small representations: ``D((E, t) g) = exp(-2 pi
    i k.t) D(g)``).

    Returns:
        A namespace with ``rotations`` (n_c, 3, 3) int64, ``translations``
        (n_c, 3), ``parent_index`` (n_c,) (the parent operation of each
        child operation), ``lattice`` (n_c, 3) (``w - w_j``, integers),
        ``conj_index`` (n_c, n_c) and ``conj_lattice`` (n_c, n_c, 3) (``[x, h]``
        for ``x^-1 h x``).
    """
    rotations, translations = _operations(mapping.child_cell, symprec)
    MT = mapping.M.T.astype(float)
    MT_inv = np.linalg.inv(MT)
    p = np.asarray(mapping.origin_shift, dtype=float)
    W = np.array([np.rint(MT @ Wc @ MT_inv) for Wc in rotations], dtype=np.int64)
    w = np.array([MT @ wc + p - Wp @ p for Wp, wc in zip(W, translations)], dtype=float)
    parent_index = np.empty(len(W), dtype=np.int64)
    for c, Wp in enumerate(W):
        match = np.flatnonzero(np.all(basis.rotations == Wp[None], axis=(1, 2)))
        if len(match) != 1:
            raise SystemExit(
                "ERROR: internal error: a child operation is not a parent operation in this "
                "setting."
            )
        parent_index[c] = match[0]
    lattice = w - basis.translations[parent_index]
    by_rotation = {W[c].tobytes(): c for c in range(len(W))}
    n = len(W)
    conj_index = np.empty((n, n), dtype=np.int64)
    conj_lattice = np.empty((n, n, 3), dtype=float)
    worst = float(np.abs(lattice - np.rint(lattice)).max()) if n else 0.0
    for x in range(n):
        Wx_inv = np.rint(np.linalg.inv(W[x])).astype(np.int64)
        for h in range(n):
            Wy = Wx_inv @ W[h] @ W[x]
            wy = Wx_inv @ (W[h] @ w[x] + w[h] - w[x])
            c = by_rotation.get(Wy.tobytes())
            if c is None:
                raise SystemExit(
                    "ERROR: internal error: the child operations do not close under conjugation."
                )
            t = wy - basis.translations[parent_index[c]]
            worst = max(worst, float(np.abs(t - np.rint(t)).max()))
            conj_index[x, h] = c
            conj_lattice[x, h] = np.rint(t)
    if worst > 1e-3:
        raise SystemExit(
            "ERROR: internal error: a child operation is not a parent operation in this setting "
            f"(translation off by {worst:.1e})."
        )
    return SimpleNamespace(
        rotations=W,
        translations=w,
        parent_index=parent_index,
        lattice=np.rint(lattice),
        conj_index=conj_index,
        conj_lattice=conj_lattice,
    )


def _inseparable_classes(pset: IrrepProjectorSet, basis: ParentIrrepBasis, mapping) -> list:
    """Units the child symmetry cannot separate; classes of more than one unit.

    Units u and v are linked when an irrep of H(k) (the child operations that
    fix k) in u and one in v occur together in one irrep of the child little
    group G(K) of K = M k, or in its time-reversal partner, i.e. when one
    child level can hold weight in both.  With Mackey's theorem the test is
    ``<Res_H Ind_H^G u, v>_H + <Res_H (Ind_H^G u)', v>_H > 0``, where the
    prime is the conjugation by an antiunitary operation (time reversal times
    a child operation A that takes K to -K): ``chi'(h) = conj(chi(A^-1 h A))``.
    Both multiplicities are non-negative integers (checked: the largest
    distance from integers is kept in ``pset.class_deviation``).  Without the
    induction (u and v sharing an irrep of H(k) only) the partners of a
    degenerate child level that carry different irreps of H(k) would be
    called separable.  Classes are the connected components of the links.

    Returns a list of tuples of unit positions (natural order); cached on ``pset``.
    """
    if pset.classes is not None:
        return pset.classes
    if basis.child_affine is None:
        basis.child_affine = _child_affine_operations(mapping, basis, basis.symprec)
    ops = basis.child_affine
    n_units = len(pset.units)
    if n_units < 2:
        pset.classes, pset.class_deviation = [], 0.0
        return pset.classes
    k = np.asarray(pset.k, dtype=float)
    M = np.asarray(mapping.M, dtype=float)
    rotated = np.einsum("cji,j->ci", ops.rotations, k)  # W^T k

    def integer(v):
        return np.all(np.abs(v - np.rint(v)) < 1e-8, axis=-1)

    in_H = integer(rotated - k)
    H = np.flatnonzero(in_H)
    G = np.flatnonzero(integer((rotated - k) @ M.T))
    A = np.flatnonzero(integer((rotated + k) @ M.T))
    position = np.full(len(basis.rotations), -1, dtype=np.int64)
    position[np.asarray(pset.little_ops, dtype=np.int64)] = np.arange(len(pset.little_ops))
    chars = np.asarray(pset.unit_characters, dtype=complex)

    def characters(cosets, lattice):
        """Characters (n_units, *shape) of the elements (coset, parent lattice translation)."""
        cols = position[ops.parent_index[cosets]]
        phase = np.exp(-2j * np.pi * (lattice @ k))
        return chars[:, np.where(cols >= 0, cols, 0)] * np.where(cols >= 0, phase, 0.0)

    order = len(H)
    chi_H = characters(H, ops.lattice[H])  # (n_units, |H|)

    def induced(X):
        """(1/|H|) sum_{x in X, x^-1 h x in H} chi_u(x^-1 h x) for every h in H: (n_units, |H|)."""
        if len(X) == 0:
            return np.zeros((n_units, order), dtype=complex)
        cosets = ops.conj_index[np.ix_(X, H)]  # (|X|, |H|)
        lattice = ops.conj_lattice[np.ix_(X, H)]  # (|X|, |H|, 3)
        values = characters(cosets, lattice) * in_H[cosets][None]
        return values.sum(axis=1) / order

    L = induced(G) @ chi_H.conj().T / order + induced(A).conj() @ chi_H.conj().T / order
    deviation = float(max(np.abs(L.imag).max(), np.abs(L.real - np.rint(L.real)).max()))
    link = np.rint(L.real) >= 1
    link = link | link.T
    classes, seen = [], set()
    for i in range(n_units):
        if i in seen:
            continue
        stack, component = [i], set()
        while stack:
            a = stack.pop()
            if a in component:
                continue
            component.add(a)
            stack.extend(int(b) for b in np.flatnonzero(link[a]) if b not in component)
        seen |= component
        if len(component) > 1:
            classes.append(tuple(sorted(component, key=lambda u: irrep_sort_key(pset.units[u]))))
    pset.classes, pset.class_deviation = classes, deviation
    return classes


def _unit_member_names(unit: str, pset: IrrepProjectorSet) -> tuple:
    index = pset.units.index(unit)
    names = pset.present_names
    return tuple(names[j] for j in pset.unit_members[index])


def _group_labels(
    pset: IrrepProjectorSet, classes, unit_w, freqs, P, weight_min: float, tol: float
):
    """The label rule on the degenerate groups of child modes at one k.

    Fractions are compared after rounding to 1e-9, and ties go by the natural
    order of the irreps (:func:`irrep_sort_key`; ``pset.units`` is in that
    order): of two irreps with 50 % each the first is the label (the fractions
    column lists both), of two sets with the same number of irreps and the same
    share the one whose sorted names come first.  The rule therefore does not
    depend on round-off.

    Returns per mode: label (str), members (tuple of irrep names), fraction;
    and the list of groups ``(indices, label, members, fraction, P_g, f (n_units,))``.
    """
    n_band = len(freqs)
    labels = ["-"] * n_band
    members = [()] * n_band
    fractions = np.zeros(n_band)
    groups = []
    for g in degenerate_groups(freqs, tol):
        Pg = float(P[g].sum())
        f = unit_w[g].sum(axis=0) / Pg if Pg > 0 else np.zeros(len(pset.units))
        fr = np.round(f, 9)  # the fractions the rule compares
        if Pg < weight_min:
            label, mem, frac = "-", (), 0.0
        else:
            best = float(fr.max())
            if best >= FRACTION_MIN:
                u = int(np.flatnonzero(fr == best)[0])  # ties: natural order
                label, chosen = pset.units[u], [u]
            else:
                candidates = []
                for cls in classes:
                    ranked = sorted(cls, key=lambda u: (-fr[u], irrep_sort_key(pset.units[u])))
                    chosen, total = [], 0.0
                    for u in ranked:
                        chosen.append(u)
                        total = round(total + float(fr[u]), 9)
                        if total >= FRACTION_MIN:
                            break
                    if total >= FRACTION_MIN:
                        names = sorted((pset.units[u] for u in chosen), key=irrep_sort_key)
                        candidates.append(
                            (len(chosen), -total, [irrep_sort_key(n) for n in names], chosen)
                        )
                if candidates:
                    chosen = min(candidates, key=lambda c: c[:3])[3]
                    label = "/".join(sorted((pset.units[u] for u in chosen), key=irrep_sort_key))
                else:
                    label, chosen = "mixed", []
            mem = tuple(
                name
                for u in sorted(chosen, key=lambda u: irrep_sort_key(pset.units[u]))
                for name in _unit_member_names(pset.units[u], pset)
            )
            frac = float(sum(f[u] for u in chosen)) if chosen else 0.0
        for J in g:
            labels[J], members[J], fractions[J] = label, mem, frac
        groups.append((g, label, mem, frac, Pg, f))
    return labels, members, fractions, groups


def _selection_rules(rep_H, chi_H, vectors, freqs, P, irrep_w, weight_min: float, tol: float):
    """Selection rules of the child symmetry H on the degenerate groups of child modes at one k.

    The unfolded vectors of a group g (canonical: mutually orthogonal, with
    norms ``P_J``) span a subspace that H maps onto itself; its character
    ``chi_g(h) = Tr(Q^H Gamma(h) Q)`` (Q an orthonormal basis of the vectors
    with ``P_J > 1e-10``) contains the irreps of H the group carries, and the
    group's weight can lie only in parent irreps that share one of them.  For
    a non-degenerate mode ``chi_g`` is the one-dimensional character of H on
    its vector.  ``rep_H`` and ``chi_H`` are taken at the same parent coset
    representatives, so the lattice-translation phases cancel.  Returns
    ``(largest weight fraction of a group in forbidden irreps, largest
    residual ||(1 - Q Q^H) Gamma(h) Q|| of the subspaces, groups checked)``.
    """
    if len(rep_H) == 0:
        return 0.0, 0.0, 0
    forbidden = residual = 0.0
    checked = 0
    for g in degenerate_groups(freqs, tol):
        Pg = float(P[g].sum())
        if Pg < weight_min:
            continue
        keep = [int(J) for J in g if P[J] > 1e-10]
        Q, _r = np.linalg.qr(np.asarray(vectors[keep]).T)
        image = np.einsum("hab,bj->haj", rep_H, Q)
        block = np.einsum("ai,haj->hij", Q.conj(), image)
        residual = max(
            residual,
            float(np.abs(image - np.einsum("ai,hij->haj", Q, block)).max()),
        )
        chi_g = np.einsum("hii->h", block)
        m = np.real(chi_H @ chi_g.conj()) / len(rep_H)
        bad = m < 0.5
        forbidden = max(forbidden, float(irrep_w[g][:, bad].sum() / Pg))
        checked += 1
    return forbidden, residual, checked


def _match_stripped(reference, stripped):
    """Permutation ``perm`` with ``stripped[i] == reference[perm[i]]`` (one to one), or None."""
    if len(reference) != len(stripped):
        return None
    perm, used = [], set()
    for row in stripped:
        hits = [
            j
            for j in range(len(reference))
            if j not in used and np.abs(row - reference[j]).max() <= _STRIPPED_TOLERANCE
        ]
        if len(hits) != 1:
            return None
        perm.append(hits[0])
        used.add(hits[0])
    return perm


def _label_points(basis: ParentIrrepBasis, psets: list, path) -> dict:
    """Name the irreps of every projector set; along straight runs of the path copy the names.

    A run is a maximal set of consecutive interior points of one leg with the
    same little-group operations (points of one open line segment with the same
    little group are of one k type).  The end points of the legs (the vertices)
    never join a run: two different special points can share their little
    group and their phase-stripped characters (GM and R of a symmorphic cubic
    group), so they are always labelled directly.  The first point of a run is
    labelled directly, the others by matching phase-stripped characters; the
    last point of a run of three or more points is labelled directly as a
    check.  A failed match or a check that disagrees labels the whole run point
    by point (counted as a mismatch).
    """
    stats = {"runs": 0, "labelled_directly": 0, "copied": 0, "mismatches": 0}
    if path is None:
        for pset in psets:
            if pset.names is None:
                basis.label_directly(pset)
    else:
        runs = []
        segment = np.asarray(path.segment_index)
        n_k = len(psets)
        for ik, pset in enumerate(psets):
            end = (
                ik == 0
                or segment[ik] != segment[ik - 1]
                or ik == n_k - 1
                or segment[ik] != segment[ik + 1]
            )
            key = (int(segment[ik]), tuple(int(j) for j in pset.little_ops), ik if end else -1)
            if runs and runs[-1][0] == key and not end:
                runs[-1][1].append(ik)
            else:
                runs.append((key, [ik]))
        for _key, members in runs:
            first = psets[members[0]]
            if first.names is None:
                basis.label_directly(first)
            todo = [psets[ik] for ik in members[1:] if psets[ik].names is None]
            if not todo:
                continue
            stats["runs"] += 1
            consistent = True
            for pset in todo:
                perm = _match_stripped(first.stripped, pset.stripped)
                if perm is None:
                    consistent = False
                    break
                pset.names = [first.names[j] for j in perm]
                pset.kname = first.kname
                pset.labelled_directly = False
            last = psets[members[-1]]
            if consistent and len(members) >= 3 and not last.labelled_directly:
                names, _kname = basis.direct_labels(last)
                if names == last.names:
                    last.labelled_directly = True
                else:
                    consistent = False
            if not consistent:
                stats["mismatches"] += 1
                for ik in members:
                    if not psets[ik].labelled_directly:
                        basis.label_directly(psets[ik])
    unique = {id(p): p for p in psets}.values()
    stats["labelled_directly"] = sum(1 for p in unique if p.labelled_directly)
    stats["copied"] = sum(1 for p in unique if not p.labelled_directly)
    return stats


def _ordered_units(psets: list) -> list:
    """Every unit met, grouped by k type in order of appearance, natural order inside a type."""
    by_type: dict = {}
    for pset in psets:
        bucket = by_type.setdefault(pset.kname, [])
        for unit in pset.units:
            if unit not in bucket:
                bucket.append(unit)
    out = []
    for units in by_type.values():
        out.extend(sorted(units, key=irrep_sort_key))
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# the result
# ---------------------------------------------------------------------------
@dataclass
class UnfoldedIrreps:
    """Parent-irrep resolution of unfolded bands (:func:`irrep_resolve`).

    Attributes:
        units: Every irrep unit met along the k points (``'GM4-'``,
            ``'S1+S2'``, ...), grouped by k type, natural order inside a type.
        kname: ISO-IR k type of every k (``'GM'``, ``'DT'``, ``'X'``, ...).
        weights: (n_k, n_band, n_units) P_J^(u)(k) per mode (canonical basis);
            0 where a unit does not occur.
        group_label: (n_k, n_band) str, the label of the mode's degenerate
            group (``'R4+'``, ``'Z1/Z3'``, ``'S1+S2'``, ``'mixed'``, ``'-'``).
        group_members: ``[k][mode]`` -> tuple of the irrep names of the label
            (``('U2+', 'U3+')``, ``('Z1', 'Z3')``; ``()`` for mixed and -).
        group_fraction: (n_k, n_band) summed share of P_g held by the label's
            units (0 for mixed and -).
        inseparable: ``[k]`` -> tuples of units the child symmetry cannot separate.
        tr_units: ``[k]`` -> the time-reversal pairs present at k.
        parent_labels: (n_k, 3 n_parent) str, unit of every parent band (with
            parent phonons), else None; ``parent_frequencies`` likewise (THz).
        mulliken: ``{parent GM unit: Mulliken symbol}``.
        checks: Largest deviations: projector identities ('completeness',
            'idempotency', 'hermiticity', 'orthogonality', 'mass_commutator'),
            'sum_over_irreps' (sum_u P_J^(u) - P_J), 'per_irrep_sum_rule'
            (sum_J P_J^(u) - trace), 'parent_purity' (distance of the unit
            weights of every parent level from whole numbers of bands),
            'forbidden_weight' (largest share of a degenerate group in parent
            irreps its irreps of H(k) forbid), 'selection_residual' (largest
            residual of the H(k)-invariance of the groups' subspaces),
            'selection_groups' (groups checked), 'class_multiplicity'
            (distance from integers of the multiplicities behind the
            inseparable sets), 'translation_mismatch'.
        cache: ``{'runs', 'labelled_directly', 'copied', 'mismatches'}`` of the line-label cache.
        basis: The :class:`ParentIrrepBasis` (reused by :func:`crystod.phonon_unfold.gamma_origin`).
        notes: NOTE lines (time-reversal test failed, Mulliken symbols, checks above 1e-6).
        timings: CPU wall times of the steps.
    """

    units: list
    kname: list
    weights: np.ndarray
    group_label: np.ndarray
    group_members: list
    group_fraction: np.ndarray
    inseparable: list
    tr_units: list
    parent_labels: np.ndarray | None
    parent_frequencies: np.ndarray | None
    mulliken: dict
    checks: dict
    cache: dict
    basis: object
    notes: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)


def _parent_eigen(parent_phonon, mapping, kpoints):
    """Parent frequencies (n_k, 3n) and eigenvectors as rows (n_k, 3n bands, 3n) at the parent k."""
    _check_parent_cell(parent_phonon, mapping)
    freqs, vecs = [], []
    for start in range(0, len(kpoints), CHUNK):
        result = _run_qpoints(parent_phonon, kpoints[start : start + CHUNK], with_eigenvectors=True)
        freqs.append(np.array(result.frequencies, dtype=float))
        vecs.append(np.transpose(np.array(result.eigenvectors), (0, 2, 1)))
    return np.concatenate(freqs), np.concatenate(vecs)


def _check_parent_cell(parent_phonon, mapping) -> None:
    try:
        _check_child_cell(parent_phonon.primitive, mapping.parent_cell)
    except ValueError:
        raise ValueError(
            "the primitive cell of parent_phonon is not the parent cell of the mapping"
        ) from None


def irrep_resolve(
    bands,
    mapping,
    *,
    path=None,
    parent_phonon=None,
    weight_min: float = WEIGHT_MIN,
    degeneracy_tolerance: float = DEGENERACY_TOLERANCE,
    symprec: float = SYMPREC,
) -> UnfoldedIrreps:
    """Resolve unfolded bands by the irreps of the parent; label every degenerate group.

    Args:
        bands: :class:`crystod.phonon_unfold.UnfoldedBands`.
        mapping: The :class:`crystod.phonon_unfold.ParentMapping`.
        path: The :class:`crystod.phonon_unfold.ParentPath` of ``bands`` (its
            straight runs reuse the ISO-IR names); None labels every k directly.
        parent_phonon: ``phonopy.Phonopy`` of the parent (primitive cell =
            ``mapping.parent_cell``) for the parent levels, or None.
        weight_min: Groups with ``P_g`` below it get the label ``'-'``.
        degeneracy_tolerance: THz, degenerate groups of child (and parent) modes.
        symprec: spglib tolerance of the symmetry analysis.

    Returns:
        :class:`UnfoldedIrreps`.

    Raises:
        SystemExit: A projector identity fails (an internal error).
    """
    timings = {}
    t0 = time.perf_counter()
    basis = ParentIrrepBasis(
        mapping.parent_cell, input_cell=mapping.parent_unitcell, symprec=symprec
    )
    child_ops, _n_child_ops, mismatch = child_operations_in_parent(mapping, basis, symprec)
    basis.child_ops = child_ops
    basis.child_affine = _child_affine_operations(mapping, basis, symprec)
    timings["basis_s"] = time.perf_counter() - t0
    kpoints = np.atleast_2d(np.asarray(bands.k_parent, dtype=float))
    n_k, n_band = bands.frequencies.shape
    if path is not None and (
        len(path.kpoints) != n_k or np.abs(path.kpoints - kpoints).max() > 1e-12
    ):
        raise ValueError("path does not hold the k points of bands")
    vectors = rotate_unfolded_vectors(bands.vectors, mapping.frame_rotation)
    freqs, P = bands.frequencies, bands.weights_canonical
    parent_freqs = parent_vecs = None
    if parent_phonon is not None:
        parent_freqs, parent_vecs = _parent_eigen(parent_phonon, mapping, kpoints)
    # pass 1: projectors, irrep weights, selection rules
    t1 = time.perf_counter()
    psets, w_irrep, w_parent = [], [], []
    forbidden = residual = 0.0
    n_checked = 0
    for ik in range(n_k):
        pset, rep = basis.build(kpoints[ik])
        psets.append(pset)
        w = irrep_weights(vectors[ik], pset.projectors)
        w_irrep.append(w)
        if parent_vecs is not None:
            w_parent.append(irrep_weights(parent_vecs[ik], pset.projectors))
        H = _child_positions(pset, child_ops)
        f_dev, e_dev, n = _selection_rules(
            rep[H],
            pset.characters[pset.present][:, H],
            vectors[ik],
            freqs[ik],
            P[ik],
            w,
            weight_min,
            degeneracy_tolerance,
        )
        forbidden, residual, n_checked = max(forbidden, f_dev), max(residual, e_dev), n_checked + n
    timings["projectors_s"] = time.perf_counter() - t1
    # pass 2: ISO-IR names (line cache), time-reversal units
    t2 = time.perf_counter()
    basis.label_time = 0.0
    cache = _label_points(basis, psets, path)
    for pset in {id(p): p for p in psets}.values():
        if pset.units is None:
            basis.make_units(pset)
    timings["labels_s"] = time.perf_counter() - t2
    # pass 3: unit weights, inseparable classes, group labels
    t3 = time.perf_counter()
    units = _ordered_units(psets)
    index = {u: i for i, u in enumerate(units)}
    weights = np.zeros((n_k, n_band, len(units)))
    group_label = np.empty((n_k, n_band), dtype=object)
    group_members = []
    group_fraction = np.zeros((n_k, n_band))
    inseparable, tr_units = [], []
    per_irrep = sum_over = class_dev = 0.0
    notes = list(basis.notes)
    tr_failed = []
    for ik in range(n_k):
        pset = psets[ik]
        unit_w = _unit_weights(pset, w_irrep[ik])
        weights[ik][:, [index[u] for u in pset.units]] = unit_w
        per_irrep = max(
            per_irrep,
            float(np.abs(w_irrep[ik].sum(axis=0) - pset.dims * pset.multiplicities).max()),
        )
        sum_over = max(
            sum_over,
            float(np.abs(w_irrep[ik].sum(axis=1) - np.sum(np.abs(vectors[ik]) ** 2, axis=1)).max()),
        )
        classes = _inseparable_classes(pset, basis, mapping)
        class_dev = max(class_dev, pset.class_deviation)
        inseparable.append([tuple(pset.units[u] for u in c) for c in classes])
        tr_units.append(list(pset.tr_pairs))
        if pset.tr_failed and not any(np.allclose(pset.k, k_) for k_ in tr_failed):
            tr_failed.append(pset.k)
        labels, members, fractions, _groups = _group_labels(
            pset, classes, unit_w, freqs[ik], P[ik], weight_min, degeneracy_tolerance
        )
        group_label[ik] = labels
        group_members.append(members)
        group_fraction[ik] = fractions
    for k_ in tr_failed:
        notes.append(
            f"NOTE: Wigner's time-reversal test failed at k = {_vec(k_)}; levels there are "
            "labelled per irrep."
        )
    # pass 4: parent levels
    parent_labels = None
    purity = 0.0
    if parent_vecs is not None:
        parent_labels = np.empty(parent_freqs.shape, dtype=object)
        for ik in range(n_k):
            pset = psets[ik]
            unit_w = _unit_weights(pset, w_parent[ik])
            parent_labels[ik] = [pset.units[u] for u in np.argmax(unit_w, axis=1)]
            # every unit holds a whole number of bands of each level (also where two
            # irreps happen to be degenerate within the tolerance)
            for level in degenerate_groups(parent_freqs[ik], degeneracy_tolerance):
                total = unit_w[level].sum(axis=0)
                purity = max(purity, float(np.abs(total - np.rint(total)).max()))
        parent_labels = parent_labels.astype(str)
    timings["weights_and_labels_s"] = time.perf_counter() - t3
    unique = list({id(p): p for p in psets}.values())
    checks = {key: max(p.checks[key] for p in unique) for key in unique[0].checks} if unique else {}
    checks.update(
        {
            "sum_over_irreps": sum_over,
            "per_irrep_sum_rule": per_irrep,
            "parent_purity": purity,
            "forbidden_weight": forbidden,
            "selection_residual": residual,
            "selection_groups": float(n_checked),
            "class_multiplicity": class_dev,
            "translation_mismatch": mismatch,
        }
    )
    if forbidden > 1e-6:
        notes.append(
            f"NOTE: weight forbidden by the child symmetry reaches {forbidden:.1e} (child levels "
            "in parent irreps that share no irrep of the child operations fixing k with them)."
        )
    if class_dev > 1e-6:
        notes.append(
            f"NOTE: the multiplicities behind the sets the child symmetry cannot separate are off "
            f"integers by {class_dev:.1e}; the sets may be wrong."
        )
    if purity > 1e-6:
        notes.append(
            f"NOTE: a parent level does not hold whole bands of its irrep units ({purity:.1e}); "
            "the parent labels there are those of the largest unit."
        )
    if cache["mismatches"]:
        runs = "run" if cache["mismatches"] == 1 else "runs"
        notes.append(
            f"NOTE: {cache['mismatches']} {runs} of labels recomputed point by point (the copied "
            "labels differed at its last point)."
        )
    timings["total_s"] = time.perf_counter() - t0
    return UnfoldedIrreps(
        units=units,
        kname=[p.kname for p in psets],
        weights=weights,
        group_label=group_label.astype(str),
        group_members=group_members,
        group_fraction=group_fraction,
        inseparable=inseparable,
        tr_units=tr_units,
        parent_labels=parent_labels,
        parent_frequencies=parent_freqs,
        mulliken=dict(basis.mulliken),
        checks=checks,
        cache=cache,
        basis=basis,
        notes=notes,
        timings=timings,
    )


def resolve_point(
    basis: ParentIrrepBasis,
    mapping,
    k,
    vectors,
    freqs,
    weights,
    *,
    weight_min: float = WEIGHT_MIN,
    degeneracy_tolerance: float = DEGENERACY_TOLERANCE,
):
    """Irrep weights and group labels of given unfolded vectors at one parent k.

    Args:
        basis: The :class:`ParentIrrepBasis` (with ``child_ops`` set).
        mapping: The :class:`crystod.phonon_unfold.ParentMapping`.
        k: (3,) parent fractional k.
        vectors: (n_band, 3 n_parent) unfolded vectors (child frame, parent origin).
        freqs, weights: (n_band,) frequencies (THz) and weights P_J(k).

    Returns:
        A namespace with ``pset``, ``unit_weights`` (n_band, n_units),
        ``labels``, ``members``, ``fractions`` per mode, ``groups`` and ``classes``.
    """
    pset = basis.projector_set(k)
    if basis.child_ops is None:
        basis.child_ops = child_operations_in_parent(mapping, basis)[0]
    u = rotate_unfolded_vectors(np.asarray(vectors), mapping.frame_rotation)
    unit_w = _unit_weights(pset, irrep_weights(u, pset.projectors))
    classes = _inseparable_classes(pset, basis, mapping)
    labels, members, fractions, groups = _group_labels(
        pset,
        classes,
        unit_w,
        np.asarray(freqs),
        np.asarray(weights),
        weight_min,
        degeneracy_tolerance,
    )
    return SimpleNamespace(
        pset=pset,
        unit_weights=unit_w,
        labels=labels,
        members=members,
        fractions=fractions,
        groups=groups,
        classes=classes,
    )


def format_label(label: str, members: tuple, kname: str, mulliken: dict) -> str:
    """A group label for the terminal: at GM with the Mulliken symbols of its members in brackets.

    ``'GM4- [T1u]'``, ``'GM4-/GM5- [T1u/T2u]'``; a time-reversal unit
    ``'GM3+GM4 [E]'`` (repeated symbols are written once); away from GM, and
    for ``mixed`` and ``-``, the label unchanged.
    """
    if kname != "GM" or label in ("mixed", "-") or not mulliken or not members:
        return label
    parts = []
    for unit in label.split("/"):
        if unit in members:
            names = [unit]
        else:  # a time-reversal unit 'a+b' of two members
            names = next(
                ([a, b] for a in members for b in members if a != b and f"{a}+{b}" == unit), []
            )
        symbols = [mulliken.get(n) for n in names]
        if not names or any(s is None for s in symbols):
            return label
        parts.append("+".join(dict.fromkeys(symbols)))
    return f"{label} [{'/'.join(parts)}]"
