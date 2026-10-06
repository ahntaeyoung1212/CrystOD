"""Electric-dipole selection rules between crystal-orbital levels at one k point.

The crystal-orbital diagrams of ``crystod --diagram`` (extended-Hueckel,
PySCF and VASP engines) label every level with the ISO-IR name of a small
irrep of the little group G_k.  An electric-dipole (optical) transition
between two levels of the same k (a vertical transition) is
symmetry-allowed for the polarization component e_a (a = x, y, z) only if
the matrix element <f| p_a |i> survives the group average, i.e. if
Gamma_f* x Gamma_V x Gamma_i contains the identity of G_k with an
invariant that has a nonzero a-component (Gamma_V: the polar-vector
representation of the momentum or position operator).

The test is done with the representation matrices themselves, not with
characters alone, so that it resolves the Cartesian component: the
operator

    P = (1/|G_k/T|) sum_g  R(g) (x) conj(D_f(g)) (x) D_i(g)

over the coset representatives g = {R|t} of G_k with respect to the
lattice translations T is the projector onto the invariant tensors of
V (x) Gamma_f* (x) Gamma_i (the Bloch phases exp(-i k.t) of the two small
irreps cancel, so the summand is a true representation of the factor
group).  The rows of P that belong to the component a are zero exactly
when every invariant tensor has a vanishing a-component, i.e. when
a-polarized light cannot drive the transition.  trace(P) is the number of
independent invariants, (1/|G_k/T|) sum_g chi_V(g) conj(chi_f(g))
chi_i(g).

The small irreps are spgrep's (``get_spacegroup_irreps_from_primitive_symmetry``)
named by the existing ISO-IR labeller (``SymmetryOnlyVibrations.get_irrep_labels``,
the frame rules of ``crystod.isoir``); this module never relabels.  The
operations are those of the builder's spglib-standardized primitive cell,
but the polarization components refer to the Cartesian axes x, y, z of
the INPUT cell (the user's POSCAR, ``builder.input_cell``): the Cartesian
rotations are carried over with spglib's ``std_rotation_matrix``
(:func:`input_frame_rotation`), the frame of the Raman tensors of
``crystod-phonon --raman-tensor``.  An allowed polarization subspace that
is not spanned by coordinate axes is reported as vectors in the input
frame (``(1 0 0), (0 1 1)``) instead of axis letters: in an input cell
rotated against the standard setting, and also in a standard setting
along a k arm that is not an axis (``(sqrt(3) 1 0)`` at M of a hexagonal
cell).  The k coordinates are those of the builder's primitive
reciprocal basis.

Spinor (spin-orbit, double-group) levels are not handled.

Example:
    >>> from phonopy.interface.calculator import read_crystal_structure
    >>> from crystod.examples import example_path
    >>> from crystod.visualize_basis import SymmetryAdaptedOrbitalBasis
    >>> cell, _ = read_crystal_structure(
    ...     str(example_path("221_PPOSCAR_SrTiO3")), interface_mode="vasp")
    >>> builder = SymmetryAdaptedOrbitalBasis(cell=cell)
    >>> table = little_group_dipole_table(builder, [0, 0, 0])
    >>> table.components("GM5-", "GM5+"), table.components("GM4-", "GM4-")
    (('x', 'y', 'z'), ())

The diagram-level entry point is :func:`dipole_selection_rules` (one
record per k point of a ``--diagram`` engine object).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CARTESIAN_AXES = ("x", "y", "z")

# levels closer than this (eV) count as one degenerate band edge (two
# complex-conjugate irreps that time reversal makes degenerate are split
# into two level records by the engines)
EDGE_TOLERANCE = 1e-3

_PROJECTOR_TOL = 1e-6


def _bare(label: str) -> str:
    """``"GM4-(3)"`` -> ``"GM4-"`` (the engines store the bare name)."""
    return str(label).split("(")[0].strip()


def _is_rigid_map(frame, primitive, input_cell, tol: float = 1e-3) -> bool:
    """Whether ``x -> frame @ x + t`` carries ``primitive`` onto ``input_cell``.

    Both the lattice and the atoms are tested: every lattice vector of the
    input cell must be an integer combination of the rotated primitive
    vectors (the input cell may be a supercell), and for one translation
    ``t`` every rotated primitive atom must land on an input atom of the
    same species modulo the input lattice.  A point-group comparison alone
    cannot tell apart two frames that differ by an element of the
    normalizer (cubic GaP and its 90-degree rotation about x); the atoms
    can.
    """
    lattice_p = np.asarray(primitive.cell, dtype=float) @ frame.T
    lattice_in = np.asarray(input_cell.cell, dtype=float)
    supercell = lattice_in @ np.linalg.inv(lattice_p)
    if not np.allclose(supercell, np.round(supercell), atol=tol):
        return False
    positions_p = np.asarray(primitive.scaled_positions, dtype=float) @ lattice_p
    numbers_p = np.asarray(primitive.numbers)
    fractional_in = np.asarray(input_cell.scaled_positions, dtype=float)
    numbers_in = np.asarray(input_cell.numbers)
    inverse_in = np.linalg.inv(lattice_in)
    length = max(1.0, float(np.max(np.linalg.norm(lattice_in, axis=1))))
    for anchor in np.where(numbers_in == numbers_p[0])[0]:
        shift = fractional_in[anchor] @ lattice_in - positions_p[0]
        mapped = (positions_p + shift) @ inverse_in
        ok = True
        for atom, number in zip(mapped, numbers_p):
            delta = fractional_in - atom
            delta -= np.round(delta)
            distance = np.linalg.norm(delta @ lattice_in, axis=1)
            if not np.any((distance < tol * length) & (numbers_in == number)):
                ok = False
                break
        if ok:
            return True
    return False


def input_frame_rotation(builder) -> np.ndarray:
    """Rotation from the builder's Cartesian frame to that of its input cell.

    The builder (:class:`~crystod.visualize_basis.SymmetryAdaptedOrbitalBasis`)
    usually works on the spglib-standardized primitive cell, whose Cartesian
    axes spglib rotates into the standard orientation: a vector ``v_std`` of
    that frame is ``Q @ v_std`` in the axes of the input cell, with
    ``Q = std_rotation_matrix^T`` of the input cell's symmetry dataset (the
    same rotation that keeps the Raman tensors of ``crystod-phonon`` in the
    input axes).  Two candidates are tried, the identity and that ``Q``,
    the identity first unless the builder's cell is spglib's standardized
    primitive cell of the input; a candidate is accepted only when it maps
    the builder's cell rigidly (lattice and atoms) onto the input cell.  A
    builder made with ``standardize=False`` from a cell in the input axes
    therefore gets the identity.

    Args:
        builder: A ``SymmetryOnlyVibrations`` or
            ``SymmetryAdaptedOrbitalBasis`` (``input_cell``,
            ``primitive_cell``, ``symprec``).

    Returns:
        The orthogonal 3 x 3 matrix ``Q``; a Cartesian rotation ``W`` of the
        builder becomes ``Q W Q^T`` in the input axes.

    Raises:
        ValueError: Neither candidate maps the builder's cell onto the
            input cell (the two cells are not the same crystal in either
            orientation).
    """
    import spglib

    from .runtime_compat import SymmetryDatasetAdapter

    input_cell = getattr(builder, "input_cell", None)
    primitive = builder.primitive_cell
    if input_cell is None or input_cell is primitive:
        return np.eye(3)
    dataset = SymmetryDatasetAdapter(
        spglib.get_symmetry_dataset(input_cell.totuple(), symprec=builder.symprec))
    standard = np.asarray(dataset["std_rotation_matrix"], dtype=float).T
    standardized = spglib.standardize_cell(
        input_cell.totuple(), to_primitive=True, symprec=builder.symprec)
    is_standardized = (
        standardized is not None
        and np.allclose(np.asarray(standardized[0], dtype=float),
                        np.asarray(primitive.cell, dtype=float), atol=1e-6))
    candidates = ([standard, np.eye(3)] if is_standardized
                  else [np.eye(3), standard])
    tol = max(1e-3, 10 * float(builder.symprec))
    for frame in candidates:
        if _is_rigid_map(frame, primitive, input_cell, tol):
            return frame
    raise ValueError("the builder's cell maps onto the input cell neither in "
                     "its own axes nor through spglib's std_rotation_matrix")


def _span(vectors, tol: float = _PROJECTOR_TOL) -> np.ndarray:
    """Orthonormal real basis (rows) of the span of real 3-vectors."""
    vectors = np.asarray(vectors, dtype=float).reshape(-1, 3)
    if not len(vectors):
        return np.zeros((0, 3))
    _, singular, vh = np.linalg.svd(vectors, full_matrices=False)
    scale = max(1.0, float(singular[0])) if len(singular) else 1.0
    return vh[: int(np.sum(singular > tol * scale))]


def polarization_subspace(final_irrep, initial_irrep, cartesian_rotations,
                          tol: float = _PROJECTOR_TOL) -> tuple[np.ndarray, int]:
    """Real polarization subspace of a dipole transition between two irreps.

    The invariant tensors T of V x Gamma_f* x Gamma_i give the matrix
    elements ``<f_m| e.r |i_n> ~ sum_a e_a T[a, m, n]``; light polarized
    along a real unit vector ``e`` drives the transition unless ``e`` is
    orthogonal to every column ``T[:, m, n]`` (real and imaginary parts).
    The subspace spanned by those columns is returned.

    Args:
        final_irrep: Representation matrices of the final level's small
            irrep, shape ``(n_ops, d_f, d_f)``, one per coset representative.
        initial_irrep: Those of the initial level, ``(n_ops, d_i, d_i)``, for
            the same operations in the same order.
        cartesian_rotations: The rotation parts of the same operations as
            Cartesian 3 x 3 matrices, ``(n_ops, 3, 3)``, in the frame the
            polarization refers to.
        tol: Relative singular-value threshold.

    Returns:
        ``(basis, multiplicity)``: an orthonormal basis of the allowed
        polarization subspace as rows, shape ``(r, 3)`` (``r = 0``: the
        transition is dipole-forbidden), and the number of independent
        invariants of V x Gamma_f* x Gamma_i.
    """
    final = np.asarray(final_irrep, dtype=complex)
    initial = np.asarray(initial_irrep, dtype=complex)
    rotations = np.real(np.asarray(cartesian_rotations, dtype=complex))
    n_ops = len(rotations)
    d_f, d_i = final.shape[1], initial.shape[1]
    size = 3 * d_f * d_i
    projector = np.zeros((size, size), dtype=complex)
    for g in range(n_ops):
        projector += np.kron(rotations[g], np.kron(np.conj(final[g]), initial[g]))
    projector /= n_ops
    multiplicity = int(round(float(np.real(np.trace(projector)))))
    if multiplicity <= 0:
        return np.zeros((0, 3)), 0
    # columns of P span the invariants; row index a * block + (m, n)
    columns = projector.reshape(3, d_f * d_i * size)
    stacked = np.concatenate([np.real(columns), np.imag(columns)], axis=1).T
    return _span(stacked, tol), multiplicity


def _rational(value: float, max_denominator: int = 12):
    """``(numerator, denominator)`` of a small-denominator fraction, or None."""
    for denominator in range(1, max_denominator + 1):
        numerator = value * denominator
        if abs(numerator - round(numerator)) < 1e-5:
            return int(round(numerator)), denominator
    return None


def _surd_vector(vector):
    """Integer form of a direction with entries in Q and Q*sqrt(r).

    Returns ``(coefficients, kinds, root)``: entry ``j`` is
    ``coefficients[j]`` (``kinds[j] == 0``) or ``coefficients[j] *
    sqrt(root)`` (``kinds[j] == 1``), with coprime integers; ``root == 1``
    for a rational direction.  None when no root 1, 2, 3 or 6 fits.
    """
    from math import lcm

    nonzero = [j for j in range(3) if abs(vector[j]) > 1e-6]
    for root in (1, 2, 3, 6):
        best = None
        for pivot in nonzero:
            scaled = vector / vector[pivot]
            parts = []
            for value in scaled:
                found = _rational(value)
                kind = 0
                if found is None and root != 1:
                    found, kind = _rational(value / np.sqrt(root)), 1
                if found is None:
                    break
                parts.append((found, kind))
            else:
                common = lcm(*(denominator for (_, denominator), _ in parts))
                coefficients = [numerator * common // denominator
                                for (numerator, denominator), _ in parts]
                divisor = int(np.gcd.reduce(
                    [abs(c) for c in coefficients if c])) or 1
                coefficients = [c // divisor for c in coefficients]
                first = next(c for c in coefficients if c)
                if first < 0:
                    coefficients = [-c for c in coefficients]
                # the smallest integers over the choices of pivot:
                # (sqrt(3) 1 0) rather than (3 sqrt(3) 0)
                size = sum(abs(c) for c in coefficients)
                if best is None or size < best[0]:
                    best = (size, coefficients, [kind for _, kind in parts])
        if best is not None:
            return best[1], best[2], root
    return None


def _nice_vector(vector) -> str:
    """``(1 0 0)``, ``(0 1 -1)``, ``(sqrt(3) 1 0)``.

    Small integers when the ratios of the entries are rational, integers
    times one square root (2, 3 or 6) when they are of that form (the k
    arms of hexagonal cells), else the entries scaled to a largest
    magnitude of 1 with three decimals.
    """
    vector = np.asarray(vector, dtype=float)
    vector = vector / vector[np.argmax(np.abs(vector) > 1e-6)]
    surd = _surd_vector(vector)
    if surd is not None:
        coefficients, kinds, root = surd
        entries = []
        for coefficient, kind in zip(coefficients, kinds):
            if not kind or not coefficient:
                entries.append(str(coefficient))
                continue
            sign = "-" if coefficient < 0 else ""
            magnitude = abs(coefficient)
            entries.append(sign + ("" if magnitude == 1 else str(magnitude))
                           + f"sqrt({root})")
        return "(" + " ".join(entries) + ")"
    vector = vector / np.max(np.abs(vector))
    return "(" + " ".join(f"{v:.3f}".rstrip("0").rstrip(".") if abs(v) > 1e-6 else "0"
                          for v in vector) + ")"


def polarization_labels(basis, tol: float = 1e-5) -> tuple[str, ...]:
    """Report labels of a polarization subspace.

    Args:
        basis: Orthonormal rows of the subspace, shape ``(r, 3)``.
        tol: Tolerance of the axis test.

    Returns:
        The axis letters (a subset of ``("x", "y", "z")``) when the subspace
        is spanned by coordinate axes, else a basis of the subspace as
        vectors in reduced row-echelon form (``("(1 0 0)", "(0 1 1)")``);
        an empty tuple for the zero subspace.
    """
    basis = np.asarray(basis, dtype=float).reshape(-1, 3)
    if not len(basis):
        return ()
    projector = basis.T @ basis
    diagonal = np.diag(projector)
    if (np.allclose(projector, np.diag(diagonal), atol=tol)
            and np.all((np.abs(diagonal) < tol) | (np.abs(diagonal - 1) < tol))):
        return tuple(axis for a, axis in enumerate(CARTESIAN_AXES)
                     if diagonal[a] > 0.5)
    # reduced row-echelon basis of the row space
    rows = basis.copy()
    pivot_row = 0
    for column in range(3):
        if pivot_row == len(rows):
            break
        pivot = pivot_row + int(np.argmax(np.abs(rows[pivot_row:, column])))
        if abs(rows[pivot, column]) < tol:
            continue
        rows[[pivot_row, pivot]] = rows[[pivot, pivot_row]]
        rows[pivot_row] /= rows[pivot_row, column]
        for other in range(len(rows)):
            if other != pivot_row:
                rows[other] -= rows[other, column] * rows[pivot_row]
        pivot_row += 1
    return tuple(_nice_vector(row) for row in rows[:pivot_row])


def axis_components(basis, tol: float = _PROJECTOR_TOL) -> tuple[str, ...]:
    """Coordinate axes along which polarized light drives the transition.

    Args:
        basis: Orthonormal rows of the polarization subspace.
        tol: Norm below which an axis counts as orthogonal to the subspace.

    Returns:
        The letters of the axes that are not orthogonal to the subspace.
    """
    basis = np.asarray(basis, dtype=float).reshape(-1, 3)
    return tuple(axis for a, axis in enumerate(CARTESIAN_AXES)
                 if len(basis) and np.linalg.norm(basis[:, a]) > tol)


def dipole_components(final_irrep, initial_irrep, cartesian_rotations,
                      tol: float = _PROJECTOR_TOL) -> tuple[tuple[str, ...], int]:
    """Allowed polarizations of a dipole transition between two irreps.

    Args:
        final_irrep: Representation matrices of the final level's small
            irrep, shape ``(n_ops, d_f, d_f)``, one per coset representative.
        initial_irrep: Those of the initial level, ``(n_ops, d_i, d_i)``, for
            the same operations in the same order.
        cartesian_rotations: The rotation parts of the same operations as
            Cartesian 3 x 3 matrices, ``(n_ops, 3, 3)``.
        tol: Relative threshold of :func:`polarization_subspace`.

    Returns:
        ``(components, multiplicity)``: the :func:`polarization_labels` of
        the allowed subspace (axis letters such as ``("x", "y")``, or
        vectors when the subspace is not spanned by axes; empty when the
        transition is dipole-forbidden) and the number of independent
        invariants of V x Gamma_f* x Gamma_i.
    """
    basis, multiplicity = polarization_subspace(
        final_irrep, initial_irrep, cartesian_rotations, tol)
    return polarization_labels(basis), multiplicity


@dataclass
class LittleGroupDipoleTable:
    """Dipole selection rules between every pair of small irreps at one k.

    Attributes:
        kpoint: The k point, in the primitive reciprocal basis of the builder.
        labels: Bare ISO-IR names of the small irreps, in spgrep order.
        dimensions: ``{label: dimension}``.
        pairs: ``{(label_a, label_b): components}`` for every unordered pair
            (``label_a <= label_b``): the :func:`polarization_labels` of the
            allowed subspace, an empty tuple when forbidden.  The rule is
            symmetric in initial and final level.
        subspaces: ``{pair: basis}``, the orthonormal rows of each pair's
            polarization subspace.
        axes: ``"input"`` (the Cartesian axes of the builder's input cell)
            or ``"standardized"`` (those of the standardized primitive cell).
        frame_rotation: The :func:`input_frame_rotation` applied (the
            identity for ``axes="standardized"``).
    """

    kpoint: list
    labels: list
    dimensions: dict
    pairs: dict = field(default_factory=dict)
    subspaces: dict = field(default_factory=dict)
    axes: str = "input"
    frame_rotation: np.ndarray = field(default_factory=lambda: np.eye(3))

    def components(self, initial: str, final: str) -> tuple[str, ...] | None:
        """Allowed polarizations for ``initial -> final``.

        Args:
            initial: Bare irrep name of the initial level, e.g. ``"GM5-"``.
            final: Bare irrep name of the final level.

        Returns:
            The allowed components (empty tuple: forbidden), or ``None``
            when either name is not a small irrep of this table.
        """
        key = tuple(sorted((_bare(initial), _bare(final))))
        return self.pairs.get(key)

    def subspace(self, initial: str, final: str) -> np.ndarray | None:
        """Orthonormal basis rows of the allowed polarization subspace.

        Args:
            initial: Bare irrep name of the initial level.
            final: Bare irrep name of the final level.

        Returns:
            Shape ``(r, 3)`` (``r = 0``: forbidden), or ``None`` when either
            name is not a small irrep of this table.
        """
        key = tuple(sorted((_bare(initial), _bare(final))))
        return self.subspaces.get(key)


def little_group_dipole_table(builder, kpoint,
                              axes: str = "input") -> LittleGroupDipoleTable:
    """Dipole selection rules of every irrep pair of the little group at ``k``.

    Args:
        builder: A :class:`~crystod.visualize_basis.SymmetryAdaptedOrbitalBasis`
            (or any ``SymmetryOnlyVibrations``): its ``rotations``,
            ``translations``, ``rotations_cartesian`` and
            ``get_irrep_labels`` define the operations and the ISO-IR labels.
        kpoint: Three primitive reciprocal coordinates of ``builder``'s cell.
        axes: ``"input"`` (default): polarizations in the Cartesian axes of
            the builder's input cell (:func:`input_frame_rotation`);
            ``"standardized"``: in those of the standardized primitive cell.

    Returns:
        The :class:`LittleGroupDipoleTable` of that k point.

    Raises:
        ValueError: ``axes`` is neither ``"input"`` nor ``"standardized"``.
    """
    from .runtime_compat import get_spacegroup_irreps_from_primitive_symmetry

    if axes not in ("input", "standardized"):
        raise ValueError(f"axes must be 'input' or 'standardized', not {axes!r}")
    irreps, mapping = get_spacegroup_irreps_from_primitive_symmetry(
        rotations=builder.rotations,
        translations=builder.translations,
        kpoint=kpoint,
    )
    labels = [_bare(label) for label in
              builder.get_irrep_labels(kpoint, irreps, mapping)]
    frame = input_frame_rotation(builder) if axes == "input" else np.eye(3)
    rotations = np.array([frame @ rotation @ frame.T for rotation in
                          np.real(np.asarray(builder.rotations_cartesian)[mapping])])
    table = LittleGroupDipoleTable(
        kpoint=[float(value) for value in kpoint],
        labels=labels,
        dimensions={label: int(irrep.shape[1])
                    for label, irrep in zip(labels, irreps)},
        axes=axes,
        frame_rotation=frame,
    )
    for a, (label_a, irrep_a) in enumerate(zip(labels, irreps)):
        for label_b, irrep_b in list(zip(labels, irreps))[a:]:
            key = tuple(sorted((label_a, label_b)))
            if key in table.pairs:
                # two irreps sharing one label (an unlabelled fallback):
                # keep the first, the label cannot tell them apart
                continue
            basis, _ = polarization_subspace(irrep_b, irrep_a, rotations)
            table.subspaces[key] = basis
            table.pairs[key] = polarization_labels(basis)
    return table


@dataclass
class DipoleTransition:
    """One transition of the band-edge report.

    Attributes:
        initial: Labels of the initial level(s) (a degenerate set).
        final: Labels of the final level(s).
        initial_irreps: Their bare irrep names.
        final_irreps: Their bare irrep names.
        initial_energy: Energy of the initial level(s) in eV.
        final_energy: Energy of the final level(s) in eV.
        components: Allowed polarizations (:func:`polarization_labels`;
            empty: forbidden), or ``None`` when an irrep is not in the
            little-group table.
        subspace: Orthonormal rows of the allowed polarization subspace
            (``None`` with ``components``).
    """

    initial: list
    final: list
    initial_irreps: list
    final_irreps: list
    initial_energy: float
    final_energy: float
    components: tuple | None
    subspace: np.ndarray | None = None

    @property
    def allowed(self) -> bool:
        """Dipole-allowed for at least one polarization."""
        return bool(self.components)

    @property
    def verdict(self) -> str:
        """``"allowed (x, y, z)"``, ``"allowed (1 0 0), (0 1 1)"``,
        ``"forbidden"`` or ``"undetermined ..."``."""
        return verdict_text(self.components)


def verdict_text(components) -> str:
    """Report wording of a component tuple.

    Args:
        components: Allowed polarizations (axis letters or vectors), an
            empty tuple (forbidden) or ``None``.

    Returns:
        ``"allowed (x, y)"``, ``"allowed (1 0 0), (0 1 1)"`` (directions in
        the input axes that are not coordinate axes), ``"forbidden"`` or
        ``"undetermined (irrep not in the little-group table)"``.
    """
    if components is None:
        return "undetermined (irrep not in the little-group table)"
    if not components:
        return "forbidden"
    if all(component in CARTESIAN_AXES for component in components):
        return "allowed (" + ", ".join(components) + ")"
    return "allowed " + ", ".join(components)


@dataclass
class DipoleSelectionRules:
    """Dipole selection rules of one k point of a crystal-orbital diagram.

    Attributes:
        name: The k-point label, e.g. ``"GM"``.
        kpoint: Its primitive reciprocal coordinates.
        table: The :class:`LittleGroupDipoleTable`, or ``None`` when the
            rules were not evaluated (see ``note``).
        band_edge: VBM -> CBM transition (``None`` without an occupied and
            an empty crystal level).
        first_allowed: The allowed occupied -> empty transition of smallest
            energy, reported when the band edge is forbidden.
        note: Why the rules were not evaluated (spinor levels), else ``""``.
    """

    name: str
    kpoint: list
    table: LittleGroupDipoleTable | None
    band_edge: DipoleTransition | None = None
    first_allowed: DipoleTransition | None = None
    note: str = ""

    def pair_table(self, irreps) -> dict:
        """Allowed pairs among ``irreps``, as page data.

        Args:
            irreps: The bare irrep names of the levels on the page.

        Returns:
            ``{"ir": [evaluated names], "ok": {"A|B": "x, z"}}`` (sorted
            pair keys, forbidden pairs omitted; the value is the
            polarization text of the terminal report, the
            :func:`polarization_labels` joined by ``", "``: axis letters
            such as ``"x, z"`` or vectors such as ``"(0 1 -1)"``), or ``{}``
            without a table.
        """
        if self.table is None:
            return {}
        known = sorted({_bare(name) for name in irreps
                        if _bare(name) in self.table.dimensions})
        allowed = {}
        for a, name_a in enumerate(known):
            for name_b in known[a:]:
                basis = self.table.subspaces.get((name_a, name_b))
                if basis is not None and len(basis):
                    allowed[f"{name_a}|{name_b}"] = ", ".join(polarization_labels(basis))
        return {"ir": known, "ok": allowed}


def _edge_set(levels, top: bool):
    """The degenerate set of levels at the top (or bottom) of ``levels``."""
    if not levels:
        return []
    pick = max if top else min
    edge = pick(level.energy for level in levels)
    return sorted((level for level in levels
                   if abs(level.energy - edge) <= EDGE_TOLERANCE),
                  key=lambda level: level.label)


def _transition(table, initial_levels, final_levels) -> DipoleTransition:
    """Union of the allowed polarizations over the members of two level sets."""
    vectors: list | None = []
    for initial in initial_levels:
        for final in final_levels:
            basis = table.subspace(initial.irrep, final.irrep)
            if basis is None:
                vectors = None
                break
            vectors.extend(basis)
        if vectors is None:
            break
    subspace = None if vectors is None else _span(vectors)
    return DipoleTransition(
        initial=[level.label for level in initial_levels],
        final=[level.label for level in final_levels],
        initial_irreps=[level.irrep for level in initial_levels],
        final_irreps=[level.irrep for level in final_levels],
        initial_energy=float(np.mean([lv.energy for lv in initial_levels])),
        final_energy=float(np.mean([lv.energy for lv in final_levels])),
        components=None if subspace is None else polarization_labels(subspace),
        subspace=subspace,
    )


def band_edge_selection_rules(builder, name: str, kpoint, crystal_levels,
                              spinor: bool = False,
                              axes: str = "input") -> DipoleSelectionRules:
    """Band-edge dipole selection rules of one k point, from its levels.

    The VBM is the highest crystal level holding electrons, the CBM the
    lowest empty one (the HOMO/LUMO markers of the page); degenerate levels
    within 1 meV form one edge.  When the band edge is forbidden, the
    allowed occupied -> empty pair of smallest energy difference is
    reported as well.  :func:`dipole_selection_rules` is the same for the
    k points of a diagram object.

    Args:
        builder: The diagram's :class:`SymmetryAdaptedOrbitalBasis`
            (``diagram.builder`` of every engine).
        name: The k-point label.
        kpoint: Its primitive reciprocal coordinates in ``builder``'s cell.
        crystal_levels: The crystal-column levels (``levels["mo"]``), records
            with ``label``, ``irrep``, ``energy`` and ``electrons``.
        spinor: Spinor (double-group) levels: the rules are not evaluated.
        axes: Frame of the polarizations, as in
            :func:`little_group_dipole_table` (default: the input cell).

    Returns:
        A :class:`DipoleSelectionRules`.

    Example:
        >>> from phonopy.interface.calculator import read_crystal_structure
        >>> from crystod import salc
        >>> from crystod.examples import example_path
        >>> cell, _ = read_crystal_structure(
        ...     str(example_path("221_PPOSCAR_SrTiO3")), interface_mode="vasp")
        >>> diagram = salc.CrystalOrbitalDiagram(cell, ["SrTi"], ["O3"])
        >>> levels, _ = diagram.solve_at([0, 0, 0])
        >>> rules = salc.band_edge_selection_rules(
        ...     diagram.builder, "GM", [0, 0, 0], levels["mo"])
        >>> rules.band_edge.initial, rules.band_edge.final, rules.band_edge.verdict
        (['GM5- #1'], ['GM5+ #1'], 'allowed (x, y, z)')
    """
    kpoint = [float(value) for value in kpoint]
    if spinor:
        return DipoleSelectionRules(
            name, kpoint, None,
            note="not evaluated for spinor (double-group) levels")
    table = little_group_dipole_table(builder, kpoint, axes=axes)
    rules = DipoleSelectionRules(name, kpoint, table)
    occupied = [lv for lv in crystal_levels if lv.electrons]
    empty = [lv for lv in crystal_levels if not lv.electrons]
    if not occupied or not empty:
        rules.note = ("no " + ("occupied" if not occupied else "empty")
                      + " crystal level at this k point")
        return rules
    vbm, cbm = _edge_set(occupied, top=True), _edge_set(empty, top=False)
    rules.band_edge = _transition(table, vbm, cbm)
    if rules.band_edge.components == ():
        best = None
        for initial in occupied:
            for final in empty:
                found = table.components(initial.irrep, final.irrep)
                if not found:
                    continue
                gap = final.energy - initial.energy
                # smallest energy; ties broken by the higher initial level
                rank = (round(gap, 6), -initial.energy)
                if best is None or rank < best[0]:
                    best = (rank, initial, final)
        if best is not None:
            rules.first_allowed = _transition(table, [best[1]], [best[2]])
    return rules


def kpoint_selection_rules(diagram, name: str, kpoint,
                           levels) -> DipoleSelectionRules:
    """Rules of one solved k point of a diagram, cached on the diagram.

    The terminal block and the HTML page of every engine call this, so they
    show the same result; the cache is ``diagram.dipole_rules``.

    Args:
        diagram: A ``--diagram`` engine object (``builder``, optionally
            ``spinor``).
        name: The k-point label.
        kpoint: Its primitive reciprocal coordinates in the builder's cell.
        levels: ``{column: [DiagramLevel]}`` of that k point (the first
            value of ``diagram.solve_at``).

    Returns:
        A :class:`DipoleSelectionRules`.
    """
    cache = getattr(diagram, "dipole_rules", None)
    if cache is None:
        cache = {}
        diagram.dipole_rules = cache
    key = (name, tuple(round(float(value), 8) for value in kpoint))
    if key not in cache:
        cache[key] = band_edge_selection_rules(
            diagram.builder, name, kpoint, levels.get("mo", []),
            spinor=bool(getattr(diagram, "spinor", False)))
    return cache[key]


def dipole_selection_rules(diagram, kpoint=None) -> list[DipoleSelectionRules]:
    """Band-edge dipole selection rules of a crystal-orbital diagram.

    The rules of the ``* Dipole selection rules at <k> *`` blocks of
    ``crystod --diagram``, for the k points of a
    :class:`CrystalOrbitalDiagram`, :class:`PySCFCrystalOrbitalDiagram` or
    ``VASPCrystalOrbitalDiagram``: each k point is solved with
    ``diagram.solve_at`` unless the engine report already did (the rules
    are cached on the diagram).  An object without ``special_kpoints`` and
    ``solve_at`` (the page object of the VASP overlap engine) gives the
    rules its report cached.  Polarizations refer to the Cartesian axes of
    the input cell; the k coordinates are those of the builder's
    (standardized) primitive reciprocal basis, as in the ``* k point *``
    blocks of the report.

    Args:
        diagram: The diagram object.
        kpoint: ``None`` (default) for every special k point of the diagram
            (``diagram.special_kpoints()``), a special-point name such as
            ``"GM"``, or three primitive reciprocal coordinates.

    Returns:
        One :class:`DipoleSelectionRules` per k point, in the order of
        ``diagram.special_kpoints()``.

    Raises:
        ValueError: ``kpoint`` names no special point of the space group
            (or, for a page object, no k point its report evaluated).

    Example:
        >>> from phonopy.interface.calculator import read_crystal_structure
        >>> from crystod import salc
        >>> from crystod.examples import example_path
        >>> cell, _ = read_crystal_structure(
        ...     str(example_path("221_PPOSCAR_SrTiO3")), interface_mode="vasp")
        >>> diagram = salc.CrystalOrbitalDiagram(cell, ["SrTi"], ["O3"])
        >>> rules = salc.dipole_selection_rules(diagram, "GM")[0]
        >>> rules.band_edge.verdict
        'allowed (x, y, z)'
    """
    if not (hasattr(diagram, "special_kpoints") and hasattr(diagram, "solve_at")):
        return _cached_selection_rules(diagram, kpoint)
    special = list(diagram.special_kpoints())
    if kpoint is None:
        points = special
    elif isinstance(kpoint, str):
        points = [(name, k) for name, k in special if name == kpoint]
        if not points:
            raise ValueError(
                f"k point {kpoint!r} is not a special point of this space group "
                f"(available: {', '.join(name for name, _ in special)})")
    else:
        coordinates = [float(value) for value in kpoint]
        names = [name for name, k in special
                 if np.allclose(np.asarray(k, dtype=float), coordinates, atol=1e-6)]
        points = [(names[0] if names else "k", coordinates)]
    results = []
    for name, k in points:
        cache = getattr(diagram, "dipole_rules", None) or {}
        key = (name, tuple(round(float(value), 8) for value in k))
        if key in cache:
            results.append(cache[key])
            continue
        levels, _ = diagram.solve_at(k)
        results.append(kpoint_selection_rules(diagram, name, k, levels))
    return results


def _cached_selection_rules(diagram, kpoint) -> list[DipoleSelectionRules]:
    """The rules an engine report cached on a page object without solver."""
    cached = list((getattr(diagram, "dipole_rules", None) or {}).values())
    if kpoint is None:
        return cached
    if isinstance(kpoint, str):
        found = [rules for rules in cached if rules.name == kpoint]
    else:
        coordinates = np.asarray([float(value) for value in kpoint])
        found = [rules for rules in cached
                 if np.allclose(rules.kpoint, coordinates, atol=1e-6)]
    if not found:
        raise ValueError(
            f"k point {kpoint!r} was not evaluated by the report of this diagram "
            f"(available: {', '.join(rules.name for rules in cached) or 'none'})")
    return found


AXES_NOTE = ("(vertical transitions in the little group of k; polarizations "
             "in the Cartesian axes x, y, z of the input cell)")


def format_dipole_selection_rules(rules: DipoleSelectionRules,
                                  k_text: str = "",
                                  nested: bool = False) -> list[str]:
    """Report block of :func:`dipole_selection_rules`.

    Args:
        rules: The rules of one k point.
        k_text: The k coordinates as printed next to the label, e.g.
            ``"(1/2,1/2,1/2)"``.
        nested: Indent the block as a part of an engine report (``" * ... *"``
            and three-space content, as the ``* k point <k> *`` blocks of
            ``crystod --diagram``); the default is a standalone block
            (``"* ... *"`` at column 0, two-space content).

    Returns:
        The lines of the ``* Dipole selection rules at ... *`` block, the
        leading blank line included.
    """
    head, pad = (" ", "   ") if nested else ("", "  ")
    title = f"{rules.name} {k_text}".strip()
    lines = ["", f"{head}* Dipole selection rules at {title} *"]
    if rules.table is None or rules.band_edge is None:
        lines.append(f"{pad}{rules.note}")
        return lines

    def side(labels, energy):
        return f"{' + '.join(labels)} ({energy:.2f} eV)"

    edge = rules.band_edge
    lines.append(f"{pad}VBM {side(edge.initial, edge.initial_energy)} -> "
                 f"CBM {side(edge.final, edge.final_energy)}: {edge.verdict}")
    if edge.components == ():
        first = rules.first_allowed
        if first is None:
            lines.append(f"{pad}no dipole-allowed transition between the levels "
                         "of this k point")
        else:
            lines.append(
                f"{pad}first allowed: {side(first.initial, first.initial_energy)}"
                f" -> {side(first.final, first.final_energy)}, "
                f"dE = {first.final_energy - first.initial_energy:.2f} eV: "
                f"{first.verdict}")
    lines.append(pad + AXES_NOTE)
    return lines
