"""Symmetry-mode analysis of a group-subgroup structure pair
(crystod-group --supergroup-cif/--subgroup-cif).

Given a high-symmetry (parent) structure and a low-symmetry (distorted)
structure of the same compound, the displacive distortion is decomposed into
symmetry-adapted modes of the parent space group: the output lists, for each
parent irrep, the k-vector, the order-parameter direction, the isotropy
subgroup, the number of independent modes, and the mode amplitude in
Angstrom -- the offline counterpart of AMPLIMODES of the Bilbao
Crystallographic Server.  If you use this feature, please cite:
D. Orobengoa, C. Capillas, M. I. Aroyo and J. M. Perez-Mato, "AMPLIMODES:
symmetry-mode analysis on the Bilbao Crystallographic Server",
J. Appl. Cryst. 42, 820-833 (2009).

Method: both structures are spglib-standardized; the parent is converted to
the primitive setting of the bundled ISO-IR tables (CIR_data) used by
crystod's space-group irrep machinery (an origin-shift fit absorbs any
difference between the spglib and ISO-IR origin conventions, verified by
invariance of the structure under every tabulated operation).  The subgroup translation lattice and the origin shift
are found by strain-tolerant lattice matching plus atom pairing (the
assignment minimizing the total distortion is chosen; among the
symmetry-equivalent sublattice settings a symmetric parent leaves
degenerate, the one whose child axes are rotated least against the parent
axes -- both as oriented in the input files -- so that displacements and
mode VESTA files follow the axes of the subgroup structure), the subgroup elements
are identified as the parent operations that leave the distorted structure
invariant, and the displacement field is projected onto every parent irrep
at the k points folding to the subgroup Gamma point with the full induced
irrep matrices (the same machinery as --parent).  Amplitudes follow the
AMPLIMODES convention: A = sqrt(sum |u_atom|^2) over the primitive cell of
the distorted structure, with Cartesian displacements measured in the
strain-free parent-derived reference lattice.  A completeness check
(sum of all projectors = identity on the displacement space) closes every
run.
"""

from __future__ import annotations

import argparse
import os
from fractions import Fraction
from itertools import product

import numpy as np

from .isotropy_subgroup import (
    ComputedInducedRepresentation,
    InducedRepresentation,
    IsotropyAnalyzer,
    _nullspace,
    _projector,
)
from .spacegroup_product import DEN, SpaceGroupIrrepAlgebra

# An operation whose worst atom mismatch stays under this (Angstrom) leaves
# the idealized structure invariant to machine precision -- not "small", but
# exactly satisfied.  It separates operations that genuinely survive from
# ones broken by a real, however tiny, displacement (a pseudo-symmetric
# child can break its own operations by as little as ~0.001 A).
_EXACT_MISMATCH = 1e-8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crystod-group --supergroup-cif",
        description=(
            "Symmetry-mode (AMPLIMODES-style) analysis of a supergroup-"
            "subgroup structure pair."
        ),
    )
    parser.add_argument(
        "--supergroup-cif",
        dest="parent",
        required=True,
        metavar="FILE",
        help="High-symmetry (parent) structure file (CIF or POSCAR).",
    )
    parser.add_argument(
        "--subgroup-cif",
        dest="child",
        required=True,
        metavar="FILE",
        help="Low-symmetry (distorted) structure file (CIF or POSCAR).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Symmetry-detection tolerance (symprec) in Angstrom (default: 0.01).",
    )
    parser.add_argument(
        "--conventional",
        action="store_true",
        help="Write the per-irrep mode VESTA files in the parent conventional "
        "basis instead of the invariant-core (primitive-derived) cell "
        "(file names get a _conv suffix).",
    )
    return parser


# ---------------------------------------------------------------------------
# structure loading and setting conversion
# ---------------------------------------------------------------------------


def _load_standardized(path: str, tolerance: float):
    """(conventional cell, primitive cell, dataset-number) via spglib."""
    import spglib

    if not os.path.isfile(path):
        raise SystemExit(f"ERROR: structure file not found: {path}")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from pymatgen.core import Structure

        try:
            structure = Structure.from_file(path)
        except Exception:
            try:
                structure = Structure.from_str(open(path).read(), fmt="poscar")
            except Exception as exc:
                raise SystemExit(f"ERROR: could not read {path}: {exc}")
    cell = (
        np.asarray(structure.lattice.matrix),
        np.asarray(structure.frac_coords),
        [site.specie.Z for site in structure],
    )
    conventional = spglib.standardize_cell(
        cell, to_primitive=False, no_idealize=False, symprec=tolerance
    )
    primitive = spglib.standardize_cell(
        cell, to_primitive=True, no_idealize=False, symprec=tolerance
    )
    if conventional is None or primitive is None:
        raise SystemExit(f"ERROR: spglib could not standardize {path}.")
    dataset = spglib.get_symmetry_dataset(conventional, symprec=1e-5)
    # the idealized cells are rotated into spglib's canonical Cartesian
    # orientation, which forgets how the input structure was oriented; the
    # rotation back to the input frame is recovered from the same cells
    # standardized WITHOUT idealization
    rotation = None
    try:
        raw_conventional = spglib.standardize_cell(
            cell, to_primitive=False, no_idealize=True, symprec=tolerance
        )
        raw_primitive = spglib.standardize_cell(
            cell, to_primitive=True, no_idealize=True, symprec=tolerance
        )
        rotation = _idealization_rotation(
            conventional, primitive, raw_conventional, raw_primitive
        )
    except Exception:
        rotation = None
    return conventional, primitive, dataset, rotation


def _idealization_rotation(conventional, primitive, raw_conventional,
                           raw_primitive):
    """Rotation Q (row convention, L_input = L_idealized @ Q) that takes
    spglib's idealized standardized lattices back into the Cartesian frame
    of the input file, or None when the non-idealized cells cannot be
    trusted to be the same cells merely un-rotated (different atom count or
    ordering, a non-orthogonal or improper map, or different rotations for
    the conventional and the primitive cell)."""
    if raw_conventional is None or raw_primitive is None:
        return None
    if len(raw_primitive[2]) != len(primitive[2]) or list(
            raw_primitive[2]) != list(primitive[2]):
        return None
    # same basis and origin: the (non-idealized) positions must agree with
    # the idealized ones atom by atom
    difference = (np.asarray(raw_primitive[1]) - np.asarray(primitive[1])
                  + 0.5) % 1.0 - 0.5
    if np.max(np.abs(difference)) > 0.05:
        return None
    rotations = []
    for ideal, raw in ((conventional, raw_conventional),
                       (primitive, raw_primitive)):
        try:
            Q = np.linalg.solve(np.asarray(ideal[0], dtype=float),
                                np.asarray(raw[0], dtype=float))
        except np.linalg.LinAlgError:
            return None
        if (not np.allclose(Q @ Q.T, np.eye(3), atol=0.05)
                or np.linalg.det(Q) <= 0):
            return None
        rotations.append(Q)
    if not np.allclose(rotations[0], rotations[1], atol=0.05):
        return None
    return rotations[1]


def _dataset_field(dataset, name):
    if isinstance(dataset, dict):
        return dataset.get(name)
    return getattr(dataset, name, None)


def _fit_origin_shift(algebra, spg_rotations, spg_translations):
    """delta with x_isoir = x_spg + delta: (I - W) delta = v_isoir - v_spg,
    modulo the full conventional translation lattice (centring included).

    Matches the ISO-IR conventional operations of the algebra against the
    spglib conventional operations of the standardized structure.  For
    centred lattices the spglib conventional dataset carries every centring
    copy of each operation, so the translation comparison must be made
    modulo the centring translations."""
    cdml = {}
    for sym in algebra.table.symmetries:
        cdml[np.rint(np.asarray(sym.R, dtype=float)).astype(np.int64).tobytes()] = (
            np.asarray(sym.t, dtype=float) % 1.0
        )
    # centring translations in conventional fractional units.  The phonopy
    # primitive matrix is column-convention (columns = primitive vectors in
    # conventional units), so centring vectors are integer combinations of
    # the COLUMNS.  Rows give the same mod-1 group for I/F (symmetric) and
    # even for A/C centring, but NOT for rhombohedral R: rows generate
    # (1/3,1/3,1/3) instead of the obverse (2/3,1/3,1/3) -- an origin fit
    # against spglib's obverse copies then finds no solution at all.
    M = algebra.primitive_matrix
    centring = []
    seen = set()
    for z in product(range(3), repeat=3):
        vector = np.round(M @ np.array(z, dtype=float), 6) % 1.0
        key = tuple(vector)
        if key not in seen:
            seen.add(key)
            centring.append(np.array(key))
    pairs = []
    for W, v in zip(spg_rotations, spg_translations):
        key = np.rint(W).astype(np.int64).tobytes()
        if key not in cdml:
            raise SystemExit(
                "ERROR: the parent operations do not match the ISO-IR setting "
                "(different conventional cell); please report this case."
            )
        pairs.append((np.asarray(W, dtype=float), cdml[key] - np.asarray(v) % 1.0))
    grid = np.arange(24) / 24.0
    candidates = np.array(list(product(grid, grid, grid)))
    for W, diff in pairs:
        residual = candidates @ (np.eye(3) - W).T - diff
        keep = np.zeros(len(candidates), dtype=bool)
        for c in centring:
            wrapped = (residual - c + 0.5) % 1.0 - 0.5
            keep |= np.all(np.abs(wrapped) < 1e-6, axis=1)
        candidates = candidates[keep]
        if len(candidates) == 0:
            raise SystemExit(
                "ERROR: could not fit the ISO-IR origin shift; please report "
                "this case."
            )
    return candidates[0]


def _to_algebra_primitive(algebra, conventional, delta):
    """Convert the spglib conventional structure to the algebra's primitive
    setting; returns (L_prim rows, fractional positions, Z numbers, A) with
    the row/column convention A resolved by an invariance check."""
    lattice_conv, positions_conv, numbers_conv = conventional
    M = algebra.primitive_matrix
    shifted = (np.asarray(positions_conv) + delta) % 1.0

    for A in (M, M.T):
        # consistent row convention: L_prim = A L_conv (rows = primitive
        # vectors in conventional units) requires x_p = x_c A^-1 for the SAME
        # Cartesian point.  Using A_inv.T here (the column convention) breaks
        # the geometry whenever A is not symmetric -- C-centred groups! -- yet
        # the damaged structure can still pass the group-invariance check
        # below, because invariance is necessary but not sufficient.
        A_inv = np.linalg.inv(A)
        prim_positions = (shifted @ A_inv) % 1.0  # row: x_p = x_c A^-1
        lattice_prim = A @ np.asarray(lattice_conv)
        merged, merged_z = _merge_duplicates(prim_positions, numbers_conv)
        if _invariant_under_algebra(algebra, merged, merged_z):
            return lattice_prim, merged, merged_z, A
    raise SystemExit(
        "ERROR: could not express the parent structure in the ISO-IR primitive "
        "setting; please report this case."
    )


def _merge_duplicates(positions, numbers, tol=1e-4):
    """Remove duplicate sites (conventional cell contains centring copies)."""
    kept, kept_z = [], []
    for x, z in zip(positions, numbers):
        duplicate = False
        for y, zz in zip(kept, kept_z):
            d = (np.asarray(x) - np.asarray(y) + 0.5) % 1.0 - 0.5
            if zz == z and np.all(np.abs(d) < tol):
                duplicate = True
                break
        if not duplicate:
            kept.append(np.asarray(x))
            kept_z.append(int(z))
    return np.array(kept), kept_z


def _invariant_under_algebra(algebra, positions, numbers, tol=1e-4) -> bool:
    for i in range(algebra.n_ops):
        W = algebra.rotations[i]
        v = np.array(algebra.translations[i], dtype=float) / DEN
        for x, z in zip(positions, numbers):
            image = (W @ x + v) % 1.0
            hit = False
            for y, zz in zip(positions, numbers):
                d = (image - y + 0.5) % 1.0 - 0.5
                if zz == z and np.all(np.abs(d) < tol):
                    hit = True
                    break
            if not hit:
                return False
    return True


# ---------------------------------------------------------------------------
# sublattice and atom-mapping search
# ---------------------------------------------------------------------------


def _sublattice_candidates(L_parent, L_child, n, strain_tol=0.20):
    """Integer matrices S (rows: child primitive basis in parent primitive
    units, det = n) whose principal strains against the child metric stay
    within the tolerance.  n comes from the exact atom-count ratio, so large
    volume strains (strongly tilted structures) are handled correctly."""
    G_child = L_child @ L_child.T
    lengths = np.sqrt(np.diag(G_child))
    max_coeff = int(np.ceil((1 + strain_tol) * np.max(lengths) / np.min(
        np.linalg.norm(L_parent, axis=1)))) + 1
    rng = range(-max_coeff, max_coeff + 1)
    vectors = [np.array(v) for v in product(rng, rng, rng) if any(v)]
    norms = {i: [] for i in range(3)}
    for v in vectors:
        length = np.linalg.norm(v @ L_parent)
        for i in range(3):
            if abs(length - lengths[i]) < strain_tol * lengths[i]:
                norms[i].append(v)
    G_parent = L_parent @ L_parent.T
    candidates = []
    for v1 in norms[0]:
        for v2 in norms[1]:
            for v3 in norms[2]:
                S = np.array([v1, v2, v3])
                if round(np.linalg.det(S)) != n:
                    continue
                G = S @ G_parent @ S.T
                # principal strains: sqrt(eig(G^-1 G_child)) - 1
                try:
                    eigenvalues = np.linalg.eigvals(np.linalg.solve(G, G_child))
                except np.linalg.LinAlgError:
                    continue
                if np.any(eigenvalues.real <= 0):
                    continue
                strains = np.sqrt(np.abs(eigenvalues.real)) - 1.0
                if np.max(np.abs(strains)) < strain_tol:
                    candidates.append(S)
    if not candidates:
        raise SystemExit(
            "ERROR: no integer sublattice of the parent matches the child "
            f"lattice within {strain_tol:.0%} principal strain; is the child "
            "really a distorted version of the parent structure?"
        )
    return candidates


def _invariant_core(S: np.ndarray, rotations) -> np.ndarray:
    """Largest sublattice of the row lattice of S invariant under all the
    (integer, primitive-basis) point operations: {t : W t in T_H for all W}.

    Since n Z^3 (n = det S) is contained in it, it is found by enumerating
    the residues modulo n and taking the Hermite normal form of the
    generators."""
    from sympy import Matrix
    from sympy.matrices.normalforms import hermite_normal_form

    n = abs(int(round(np.linalg.det(S))))
    S_inv = np.linalg.inv(S)
    residues = []
    for t in product(range(n), repeat=3):
        vector = np.array(t, dtype=float)
        ok = True
        for W in rotations:
            frac = ((W @ vector) @ S_inv + 0.5) % 1.0 - 0.5
            if not np.all(np.abs(frac) < 1e-6):
                ok = False
                break
        if ok:
            residues.append(np.array(t, dtype=np.int64))
    generators = residues + [n * e for e in np.eye(3, dtype=np.int64)]
    H = np.array(
        hermite_normal_form(Matrix(np.array(generators, dtype=np.int64).T))
    ).astype(np.int64)
    return H.T  # rows = invariant sublattice basis


def _translation_reps(S: np.ndarray) -> list[np.ndarray]:
    """Representatives of Z^3 / (rows of S) (parent lattice mod sublattice)."""
    n = abs(int(round(np.linalg.det(S))))
    S_inv = np.linalg.inv(S)
    reps, seen = [], set()
    bound = int(np.max(np.abs(S))) + 1
    candidates = sorted(
        product(range(-bound, bound + 1), repeat=3),
        key=lambda t: (sum(abs(v) for v in t),
                       sum(1 for v in t if v < 0), t),
    )
    for t in candidates:
        frac = (np.array(t, dtype=float) @ S_inv + 1e-9) % 1.0
        key = tuple(np.round(frac, 6))
        if key not in seen:
            seen.add(key)
            reps.append(np.array(t, dtype=np.int64))
        if len(reps) == n:
            break
    if len(reps) != n:
        raise SystemExit("ERROR: broken sublattice bookkeeping.")
    return reps


class MappingResult:
    def __init__(self, S, p, ref_frac, ref_orbit, child_frac, child_z, u_frac):
        self.S = S                  # child primitive basis in parent prim units
        self.p = p                  # origin shift (parent primitive fractional)
        self.ref_frac = ref_frac    # reference atoms, parent primitive frac
        self.ref_orbit = ref_orbit  # parent orbit id per reference atom
        self.child_frac = child_frac  # paired child atoms, parent primitive frac
        self.child_z = child_z
        self.u_frac = u_frac        # displacements (parent primitive frac)


def _match_atoms(parent_positions, parent_numbers, parent_orbits,
                 child_positions, child_numbers, S, L_parent):
    """Pair the child atoms with parent atoms in the S-cell; returns the
    best MappingResult for this S (or None)."""
    reps = _translation_reps(S)
    ref_frac, ref_z, ref_orbit = [], [], []
    for x, z, orbit in zip(parent_positions, parent_numbers, parent_orbits):
        for t in reps:
            ref_frac.append(np.asarray(x, dtype=float) + t)
            ref_z.append(int(z))
            ref_orbit.append(orbit)
    ref_frac = np.array(ref_frac)
    if len(ref_frac) != len(child_positions):
        return None

    S_inv = np.linalg.inv(S)
    child_par = np.asarray(child_positions) @ S  # child frac -> parent frac

    def pair_with(p):
        """Greedy nearest-unused pairing for an origin shift p."""
        used = set()
        pairs = []
        total = 0.0
        for y, zc in zip(child_par, child_numbers):
            y_shift = y + p
            best_j, best_d, best_u = None, None, None
            for j, (xr, zr) in enumerate(zip(ref_frac, ref_z)):
                if zr != zc or j in used:
                    continue
                d_frac = y_shift - xr
                d_frac = d_frac - np.rint(d_frac @ S_inv) @ S
                dist = np.linalg.norm(d_frac @ L_parent)
                if best_d is None or dist < best_d:
                    best_j, best_d, best_u = j, dist, d_frac
            if best_j is None or best_d > 1.8:
                return None
            used.add(best_j)
            pairs.append((best_j, best_u))
            total += best_d**2
        return total, pairs

    # anchor candidates: the first child atom of EVERY species onto every
    # same-species reference atom (the anchor species may itself be
    # displaced, so the origin is refined continuously afterwards)
    anchor_children = {}
    for index, z in enumerate(child_numbers):
        anchor_children.setdefault(z, index)
    best = None
    for z_anchor, child_index in anchor_children.items():
        for x, z in zip(ref_frac, ref_z):
            if z != z_anchor:
                continue
            p = x - child_par[child_index]
            result = pair_with(p)
            if result is None:
                continue
            # continuous origin refinement: subtract the mean displacement
            # (for non-polar subgroups it is ~0; for polar ones this is the
            # AMPLIMODES minimum-distortion origin) and re-pair once
            for _ in range(2):
                mean = np.mean([u for _, u in result[1]], axis=0)
                if np.linalg.norm(mean @ L_parent) < 1e-8:
                    break
                p = p - mean
                refined = pair_with(p)
                if refined is None:
                    break
                result = refined
            if result is None:
                continue
            total, pairs = result
            if best is not None and total >= best[0]:
                continue
            u = np.zeros((len(ref_frac), 3))
            child_sorted = np.zeros((len(ref_frac), 3))
            child_z_sorted = [0] * len(ref_frac)
            for i_child, (j_ref, d_frac) in enumerate(pairs):
                u[j_ref] = d_frac
                child_sorted[j_ref] = ref_frac[j_ref] + d_frac
                child_z_sorted[j_ref] = child_numbers[i_child]
            best = (
                total,
                MappingResult(S, p, ref_frac, ref_orbit, child_sorted,
                              child_z_sorted, u),
            )
    return best


def _select_setting(matches, L_parent_input, L_child_input):
    """The atom mapping to use among the candidate sublattice bases.

    The total distortion decides first: the assignment minimizing it.  A
    symmetric parent leaves that minimum degenerate -- every S W (W a
    point operation of the parent) pairs the atoms equally well and merely
    presents the SAME distortion in a rotated setting, up to 24 of them for
    a cubic parent, and the enumeration order used to pick one at random
    (a polarization along c of a P4mm file came out along -b of the cubic
    parent).  Among the tied settings the one whose child basis is rotated
    least against the parent basis, both taken as the input files orient
    them, is chosen, so that the displacement table and the mode VESTA
    files follow the axes of the subgroup structure the user supplied.
    Without an orientation reference the first candidate is kept.

    Args:
        matches: ``(total squared distortion, MappingResult)`` per
            candidate basis.
        L_parent_input: Parent primitive lattice (rows) in the Cartesian
            frame of the parent input file, or None.
        L_child_input: Child primitive lattice (rows) in the Cartesian
            frame of the child input file, or None.

    Returns:
        ``(mapping, rotation, n_tied)``: the chosen mapping, the residual
        rotation (degrees) between the child and the parent axes of the
        input files for that setting (None without a reference), and the
        number of settings that tied on the distortion.
    """
    best_total = min(total for total, _ in matches)
    threshold = best_total + max(1e-9, 1e-6 * best_total)
    tied = [mapping for total, mapping in matches if total <= threshold]
    if len(tied) == 1 or L_parent_input is None or L_child_input is None:
        return tied[0], None, len(tied)
    scored = []
    for index, mapping in enumerate(tied):
        S = np.asarray(mapping.S, dtype=float)
        # deformation gradient F taking the S cell of the parent onto the
        # child cell, both in the input frames: L_child = (S L_parent) F^T;
        # its polar (rotation) factor is the rigid rotation between the
        # two settings, the strain being the symmetric factor
        try:
            F_T = np.linalg.solve(S @ L_parent_input, L_child_input)
        except np.linalg.LinAlgError:
            continue
        U, _, Vt = np.linalg.svd(F_T)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
        cosine = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
        angle = float(np.degrees(np.arccos(cosine)))
        S_int = np.rint(S).astype(np.int64)
        # deterministic tie-break for settings rotated equally (a 45 deg
        # sqrt2 x sqrt2 cell has two): the basis closest to the identity
        key = (round(angle, 6), -int(np.trace(S_int)),
               -int(np.sum(S_int >= 0)), tuple((-S_int).flatten()), index)
        scored.append((key, mapping, angle))
    if not scored:
        return tied[0], None, len(tied)
    scored.sort(key=lambda item: item[0])
    return scored[0][1], scored[0][2], len(tied)


# ---------------------------------------------------------------------------
# the analysis
# ---------------------------------------------------------------------------


class SymmetryModeAnalysis:
    """Symmetry-mode decomposition of a parent-child structure pair.

    The analysis behind ``crystod-group --supergroup-cif PARENT
    --subgroup-cif CHILD``: the displacive distortion between a
    high-symmetry (parent) structure and a low-symmetry (child) structure
    of the same compound is decomposed into symmetry-adapted modes of the
    parent space group, giving for every parent irrep the k vector, the
    order-parameter direction, the isotropy subgroup, the number of
    independent modes and the amplitude in Angstrom (AMPLIMODES
    convention).  All the work happens in the constructor; the results are
    read from the attributes.

    Args:
        parent_file: High-symmetry structure file (CIF or POSCAR).
        child_file: Low-symmetry (distorted) structure file (CIF or
            POSCAR); its primitive cell must be an integer multiple of the
            parent's.
        tolerance: Symmetry-detection tolerance (symprec, Angstrom) for
            both structures; the command's default is ``0.01``.

    Attributes:
        parent_number: Space-group number of the parent (``parent_symbol``
            its Hermann-Mauguin symbol); ``child_number`` and
            ``child_symbol`` likewise for the child.
        algebra: The ``SpaceGroupIrrepAlgebra`` of the parent.
        L_parent: Parent primitive lattice vectors (rows, Angstrom) of the
            strain-free reference in which every displacement is measured.
        parent_positions: Parent atoms in the primitive setting of the
            ISO-IR tables (fractional); ``parent_numbers`` their atomic
            numbers.
        size: Primitive-cell multiplication of the child relative to the
            parent.
        mapping: The atom pairing: ``mapping.S`` (child primitive basis in
            parent primitive units, rows), ``mapping.p`` (origin shift),
            ``mapping.ref_frac`` (reference atoms), ``mapping.child_z``
            (atomic numbers) and ``mapping.u_frac`` (displacements, parent
            primitive fractional).  Among the symmetry-equivalent sublattice
            settings of a symmetric parent, ``S`` is the one whose child
            axes are rotated least against the parent axes as the input
            files orient them (``setting_rotation``, the residual rotation
            in degrees, None when the input orientation could not be
            recovered; ``equivalent_settings``, how many settings tied).
        core_size: Multiplication of the invariant-core analysis cell; its
            atoms are ``ref_frac`` (``n_atoms`` of them) with atomic
            numbers ``ref_z`` and Cartesian displacements ``u_cart``.
        subgroup_members: The ``(i, t)`` parent operations that leave the
            distorted structure invariant.
        stars: The parent k stars folding to the child Gamma point, one
            dict per star with ``kname``, ``kvec`` (primitive basis) and
            ``kind`` (``"tabulated"`` or ``"computed"``).
        modes: One entry per parent irrep with a nonzero number of modes,
            carrying ``kname``, ``kvec``, ``irrep_name``, ``dim`` (number
            of independent modes), ``amplitude`` (Angstrom),
            ``projected_u`` (the irrep-projected displacement field on the
            core cell, shape ``(n_atoms, 3)``) and ``label_info()``, which
            returns ``(direction label, subgroup info, index)``.
        total_distortion: Total distortion amplitude (Angstrom), normalized
            within the primitive cell of the distorted structure.
        parent_formula: Reduced chemical formula of the parent (the command
            names its ``sym_mode_<formula>`` table after it).

    Raises:
        SystemExit: The child cell is not an integer multiple of the parent
            cell, the child cannot be mapped onto the parent, or an
            internal consistency check (mode completeness) fails.

    Example:
        >>> from crystod import group
        >>> analysis = group.SymmetryModeAnalysis("221.cif", "140.cif", 0.01)
        >>> for mode in analysis.modes:
        ...     label, info, index = mode.label_info()
        ...     print(mode.irrep_name, label, info.international_short,
        ...           mode.dim, round(mode.amplitude, 4))
    """

    def __init__(self, parent_file: str, child_file: str, tolerance: float):
        parent_conv, parent_prim, parent_ds, parent_rotation = (
            _load_standardized(parent_file, tolerance)
        )
        child_conv, child_prim, child_ds, child_rotation = _load_standardized(
            child_file, tolerance
        )
        self.parent_number = int(_dataset_field(parent_ds, "number"))
        self.child_number = int(_dataset_field(child_ds, "number"))
        self.parent_symbol = str(
            _dataset_field(parent_ds, "international")).replace("_", "")
        self.child_symbol = str(
            _dataset_field(child_ds, "international")).replace("_", "")
        self.tolerance = tolerance
        self.parent_conv = parent_conv
        self.child_conv = child_conv
        self.child_prim = child_prim

        self.algebra = SpaceGroupIrrepAlgebra(str(self.parent_number))
        delta = _fit_origin_shift(
            self.algebra,
            _dataset_field(parent_ds, "rotations"),
            _dataset_field(parent_ds, "translations"),
        )
        (self.L_parent, self.parent_positions, self.parent_numbers,
         self.A_setting) = _to_algebra_primitive(self.algebra, parent_conv, delta)
        self.delta = delta
        # the parent basis as the parent input file orients it (orientation
        # reference for the choice among equivalent sublattice settings)
        self.L_parent_input = (
            None if parent_rotation is None else self.L_parent @ parent_rotation
        )

        # parent orbits (element + Wyckoff) in the primitive setting
        self.parent_orbits = self._parent_orbits()

        # chemically ordered reduced formula of the parent (file names of the
        # saved decomposition table); never fatal
        try:
            from .poscar2cif import chemical_formula_parts, format_chemical_formula

            self.parent_formula = format_chemical_formula(
                chemical_formula_parts(
                    parent_conv[2], _dataset_field(parent_ds, "equivalent_atoms")
                )
            )
        except Exception:
            from collections import Counter

            from pymatgen.core.periodic_table import Element

            counts = Counter(
                Element.from_Z(int(z)).symbol for z in parent_conv[2]
            )
            divisor = 0
            for value in counts.values():
                divisor = int(np.gcd(divisor, value))
            self.parent_formula = "".join(
                f"{s}{n // divisor}" if n // divisor != 1 else s
                for s, n in sorted(counts.items())
            )

        # child primitive structure, fractional in its own basis
        L_child, child_positions, child_numbers = child_prim
        if len(child_numbers) % len(self.parent_numbers) != 0:
            raise SystemExit(
                f"ERROR: the child primitive cell ({len(child_numbers)} atoms) "
                "is not an integer multiple of the parent primitive cell "
                f"({len(self.parent_numbers)} atoms)."
            )
        self.size = len(child_numbers) // len(self.parent_numbers)
        self.L_child_input = (
            None if child_rotation is None else L_child @ child_rotation
        )
        candidates = _sublattice_candidates(self.L_parent, L_child, self.size)
        matches = []
        for S in candidates:
            result = _match_atoms(
                self.parent_positions, self.parent_numbers, self.parent_orbits,
                child_positions, child_numbers, S, self.L_parent,
            )
            if result is not None:
                matches.append(result)
        if not matches:
            raise SystemExit(
                "ERROR: could not map the child structure onto the parent "
                "(is it really a distorted version of the parent structure?)."
            )
        (self.mapping, self.setting_rotation,
         self.equivalent_settings) = _select_setting(
            matches, self.L_parent_input, self.L_child_input
        )

        # analysis cell: the largest G-invariant sublattice of T_H, so that
        # the displacement space carries a full representation of the parent
        # group (complete stars); amplitudes are rescaled back to the
        # primitive cell of the distorted structure at the end
        self.S_core = _invariant_core(self.mapping.S, self.algebra.rotations)
        self.core_size = abs(int(round(np.linalg.det(self.S_core))))
        self.n_parent = len(self.parent_numbers)
        self._build_core_cell()

        self.op_tables = self._op_tables()
        self.subgroup_members = self._find_subgroup_members()
        self._remove_acoustic_offset()
        self.stars = self._folding_stars()
        self.modes = self._decompose()

    def _remove_acoustic_offset(self):
        """Continuous origin refinement: subtract the uniform-translation
        component allowed by the polar directions of the subgroup, so that
        the global distortion is minimal (the AMPLIMODES origin convention).
        For a non-polar subgroup nothing changes."""
        LT = self.L_parent.T
        rows = []
        seen = set()
        for i, _ in self.subgroup_members:
            if i in seen:
                continue
            seen.add(i)
            W = self.algebra.rotations[i]
            R = LT @ W @ np.linalg.inv(LT)
            rows.append(R - np.eye(3))
        free = _nullspace(np.vstack(rows))
        self.polar_directions = free.shape[1]
        if free.shape[1] == 0:
            return
        # the atom-pairing stage already refines the origin continuously, so
        # the residual projected here is usually ~0; the flag above still
        # records that the subgroup is polar (free origin) for the report
        mean = self.u_cart.mean(axis=0)
        shift = free @ (free.T @ mean)
        if np.linalg.norm(shift) < 1e-10:
            return
        shift_frac = shift @ np.linalg.inv(self.L_parent)
        self.core_u_frac = self.core_u_frac - shift_frac
        self.u_cart = self.core_u_frac @ self.L_parent
        self.core_child_frac = self.ref_frac + self.core_u_frac
        self.mapping.u_frac = self.mapping.u_frac - shift_frac
        self.mapping.child_frac = self.mapping.child_frac - shift_frac

    def _build_core_cell(self):
        """Reference atoms and displacements on the invariant-core cell."""
        mapping = self.mapping
        S_H_inv = np.linalg.inv(mapping.S)
        reps_core = _translation_reps(self.S_core)
        # atom ordering (parent atom p major, core translation s minor) and
        # this translation list are what the Fourier star blocks index by
        self.reps_core = reps_core
        ref_frac, ref_z, ref_orbit, u_frac = [], [], [], []
        for x, z, orbit in zip(self.parent_positions, self.parent_numbers,
                               self.parent_orbits):
            for t in reps_core:
                position = np.asarray(x, dtype=float) + t
                # displacement of the T_H-equivalent atom of the mapping
                found = None
                for j, xr in enumerate(mapping.ref_frac):
                    d = position - xr
                    if np.all(np.abs((d @ S_H_inv + 0.5) % 1.0 - 0.5) < 1e-6):
                        found = j
                        break
                if found is None:
                    raise SystemExit(
                        "ERROR: broken invariant-core bookkeeping."
                    )
                ref_frac.append(position)
                ref_z.append(int(z))
                ref_orbit.append(orbit)
                u_frac.append(mapping.u_frac[found])
        self.ref_frac = np.array(ref_frac)
        self.ref_z = ref_z
        self.ref_orbit = ref_orbit
        self.core_u_frac = np.array(u_frac)
        self.n_atoms = len(ref_frac)
        self.u_cart = self.core_u_frac @ self.L_parent
        # child structure on the core cell (for the subgroup search)
        self.core_child_frac = self.ref_frac + self.core_u_frac

    # -- parent orbits
    def _parent_orbits(self):
        algebra = self.algebra
        n = len(self.parent_positions)
        orbit = list(range(n))
        for i in range(algebra.n_ops):
            W = algebra.rotations[i]
            v = np.array(algebra.translations[i], dtype=float) / DEN
            for j in range(n):
                image = (W @ self.parent_positions[j] + v) % 1.0
                for m in range(n):
                    d = (image - self.parent_positions[m] + 0.5) % 1.0 - 0.5
                    if (self.parent_numbers[j] == self.parent_numbers[m]
                            and np.all(np.abs(d) < 1e-4)):
                        root = min(orbit[j], orbit[m])
                        orbit[j] = orbit[m] = root
                        break
        return orbit

    # -- parent-cell operation tables (image atom, integer offset, Cartesian R)
    def _op_tables(self):
        """Per parent operation i: (img, tau, R) with W x_p + v = x_img[p] +
        tau[p] (tau integer) and R the Cartesian rotation.  This is the whole
        geometric content of the displacement representation: the action on
        any supercell follows by translation bookkeeping."""
        algebra = self.algebra
        LT = self.L_parent.T
        tables = []
        for i in range(algebra.n_ops):
            W = algebra.rotations[i]
            v = np.array(algebra.translations[i], dtype=float) / DEN
            R = LT @ W @ np.linalg.inv(LT)
            img = np.zeros(self.n_parent, dtype=np.int64)
            tau = np.zeros((self.n_parent, 3), dtype=np.int64)
            for p in range(self.n_parent):
                image = W @ self.parent_positions[p] + v
                target = None
                for m in range(self.n_parent):
                    d = image - self.parent_positions[m]
                    if (self.parent_numbers[p] == self.parent_numbers[m]
                            and np.allclose(d, np.rint(d), atol=1e-4)):
                        target, offset = m, np.rint(d).astype(np.int64)
                        break
                if target is None:
                    raise SystemExit(
                        "ERROR: broken parent-operation bookkeeping."
                    )
                img[p] = target
                tau[p] = offset
            tables.append((img, tau, R))
        return tables

    # -- subgroup elements: parent operations preserving the child structure
    def _find_subgroup_members(self):
        """Parent operations that survive in the child, selected adaptively.

        Every (operation, translation) candidate gets a mismatch distance:
        the worst atom-to-nearest-partner distance of the transformed child
        structure.  Genuine members of H sit at the numerical-noise level of
        the standardized child, while broken operations sit at the scale of
        the symmetry-BREAKING part of the distortion — which can be far
        smaller than any fixed cutoff (pseudo-symmetric structures whose
        breaking component is ~0.001 A while the fully symmetric component
        is large) or far larger (strong tilts).  A fixed threshold therefore
        cannot work; instead, the child's own space group fixes how many
        members MUST survive, and the threshold is placed at that point of
        the sorted mismatch spectrum after checking the gap is clean.

        The search runs on the child (T_H) cell; translations are the
        representatives of Z^3 / T_H."""
        import spglib

        algebra = self.algebra
        S = self.mapping.S
        S_inv = np.linalg.inv(S)
        reps = _translation_reps(S)
        positions = self.mapping.ref_frac + self.mapping.u_frac
        numbers = self.mapping.child_z
        candidates = []
        for i in range(algebra.n_ops):
            W = algebra.rotations[i]
            v = np.array(algebra.translations[i], dtype=float) / DEN
            for t in reps:
                worst = 0.0
                for x, z in zip(positions, numbers):
                    image = W @ x + v + t
                    best = None
                    for y, zz in zip(positions, numbers):
                        if zz != z:
                            continue
                        d = image - y
                        d = d - np.rint(d @ S_inv) @ S
                        dist = np.linalg.norm(d @ self.L_parent)
                        if best is None or dist < best:
                            best = dist
                    worst = max(worst, best)
                candidates.append(((i, np.asarray(t, dtype=np.int64)), worst))
        candidates.sort(key=lambda entry: entry[1])

        # expected factor-group order of the child on its own (T_H) cell --
        # a LOWER bound only.  Displacements are measured in the strain-free
        # parent-derived reference lattice (the AMPLIMODES convention), so
        # an operation broken by the child's METRIC alone, with no atomic
        # counterpart, still leaves the displacement field invariant and
        # belongs in H.  That is a whole class of transitions: a purely
        # ferroelastic one, where the distortion is the fully symmetric
        # irrep plus a spontaneous strain (La3Ni2O7 I4/mmm -> Fmmm keeps
        # every one of the 16 parent operations exactly while spglib reads
        # the child as Fmmm, 8).  Whenever at least `expected` operations
        # sit at the numerical-noise level, they all survive.
        child_ops = spglib.get_symmetry(self.child_prim, symprec=1e-5)
        expected = len(child_ops["rotations"])
        if expected >= len(candidates):
            return [entry[0] for entry in candidates]
        exact = sum(1 for _, worst in candidates if worst < _EXACT_MISMATCH)
        if exact >= expected:
            self.strain_only_operations = exact - expected
            return [entry[0] for entry in candidates[:exact]]
        low = candidates[expected - 1][1]
        high = candidates[expected][1]
        if high < 2.0 * low + 1e-6:
            raise SystemExit(
                "ERROR: cannot separate the surviving from the broken parent "
                f"operations (mismatch gap {low:.2e} .. {high:.2e} A); the "
                "child symmetry is ambiguous at this precision -- try a "
                "different --tolerance."
            )
        return [entry[0] for entry in candidates[:expected]]

    # -- k stars folding to the child Gamma point
    def _folding_stars(self):
        """One record per distinct parent k star folding to the child Gamma.

        The folding points are the reciprocal-lattice points of T_H inside
        the parent zone (`size` of them).  Points on tabulated special
        stars keep their table entry; the others (symmetry lines, planes
        or general points -- e.g. the SM point (1/4,1/4,0) of the Pbam
        antiferroelectric of PbZrO3) are handled through spgrep small
        irreps at the exact k, named with the ISO-IR line machinery."""
        algebra = self.algebra
        S_H = np.asarray(self.mapping.S, dtype=np.int64)
        S_inv_T = np.linalg.inv(S_H).T
        points = []
        for z in _translation_reps(S_H.T):
            k = (np.asarray(z, dtype=float) @ S_inv_T) % 1.0
            k_int = k * DEN
            if not np.allclose(k_int, np.rint(k_int), atol=1e-6):
                raise SystemExit(
                    "ERROR: a k point folding to the child Gamma point "
                    f"({np.round(k, 6)}) is not on the 1/{DEN} grid of the "
                    "irrep tables; this cell multiplication is not "
                    "supported yet."
                )
            points.append(np.rint(k_int).astype(np.int64) % DEN)
        if len(points) != self.size:
            raise SystemExit("ERROR: broken folding-point enumeration.")

        # tabulated stars, by arm membership
        tabulated = {}
        for kname in algebra.k_by_kname:
            arms, _ = algebra.star(kname)
            for arm in arms:
                tabulated[tuple(int(v) % DEN for v in arm)] = kname

        stars = []
        assigned = set()
        for point in points:
            key = tuple(int(v) for v in point)
            if key in assigned:
                continue
            kname = tabulated.get(key)
            if kname is not None:
                arms, _ = algebra.star(kname)
                record = {
                    "kind": "tabulated",
                    "kname": kname,
                    "kvec": np.array(algebra.k_by_kname[kname], dtype=float)
                    / DEN,
                }
            else:
                arms, _ = algebra._star_of_vector(point)
                canonical = np.array(
                    min(tuple(int(v) for v in arm) for arm in arms),
                    dtype=np.int64,
                )
                point_name, names, source = algebra._line_names(canonical)
                record = {
                    "kind": "computed",
                    "kname": point_name,
                    "canonical": canonical,
                    "names": names,
                    "kvec": algebra.isoir_display_arm(canonical) / DEN,
                }
            arm_keys = {tuple(int(v) % DEN for v in arm) for arm in arms}
            assigned |= arm_keys
            record["sort_key"] = min(
                tuple(np.round((np.array(k, dtype=float) / DEN) % 1.0, 6))
                for k in arm_keys
            )
            stars.append(record)
        return sorted(stars, key=lambda record: record["sort_key"])

    # -- Fourier star blocks of the displacement representation
    #
    # Basis of one star block: (arm a, parent atom p, Cartesian mu) -> the
    # plane-wave displacement  e^{+2 pi i q_a.t/DEN} u_p / sqrt(n_t)  on the
    # sublattice copies of atom p over the invariant-core cell.  In this
    # basis every group element is block-sparse over the arms, the sizes are
    # set by the star (3 n_parent m), and the core cell never appears as a
    # matrix dimension -- the old dense route built 3N x 3N displacement
    # matrices on the core cell, which is unusable already at the
    # 4x4x4-fold invariant core of the Pbam antiferroelectric (N = 320).

    def _star_dhat(self, arms, arm_index, i):
        """V^dagger D(i, t=0) V on the star block space."""
        img, tau, R = self.op_tables[i]
        m = len(arms)
        np3 = 3 * self.n_parent
        matrix = np.zeros((m * np3, m * np3), dtype=np.complex128)
        W = self.algebra.rotations[i]
        for a in range(m):
            q_b = tuple(int(v) % DEN for v in (np.asarray(arms[a]) @ W))
            b = arm_index[q_b]
            phases = np.exp(-2j * np.pi * (tau @ np.asarray(arms[a])) / DEN)
            for p in range(self.n_parent):
                row = a * np3 + 3 * img[p]
                col = b * np3 + 3 * p
                matrix[row : row + 3, col : col + 3] = phases[p] * R
        return matrix

    def _star_translation_phases(self, arms, t):
        """Diagonal of V^dagger T(t) V (per-arm phases, repeated 3 n_p)."""
        phases = np.exp(-2j * np.pi * (np.asarray(arms) @ np.asarray(t)) / DEN)
        return np.repeat(phases, 3 * self.n_parent)

    def _star_uhat(self, arms):
        """Fourier components of the displacement field over the core cell."""
        n_t = len(self.reps_core)
        u = self.u_cart.reshape(self.n_parent, n_t, 3)
        t_matrix = np.asarray(self.reps_core, dtype=float)  # (n_t, 3)
        phases = np.exp(
            -2j * np.pi * (np.asarray(arms) @ t_matrix.T) / DEN
        ) / np.sqrt(n_t)  # (m, n_t)
        # uhat[(a, p, mu)] = sum_s conj(f_a(s)) u[p, s, mu]
        uhat = np.einsum("as,psm->apm", phases, u)
        return uhat.reshape(-1)

    def _star_to_core(self, arms, block_vector):
        """Real-space (core cell) displacement field of a star-block vector."""
        n_t = len(self.reps_core)
        t_matrix = np.asarray(self.reps_core, dtype=float)
        phases = np.exp(
            2j * np.pi * (np.asarray(arms) @ t_matrix.T) / DEN
        ) / np.sqrt(n_t)  # (m, n_t)
        blocks = block_vector.reshape(len(arms), self.n_parent, 3)
        field = np.einsum("as,apm->psm", phases, blocks)
        if np.max(np.abs(field.imag)) > 1e-6:
            raise SystemExit(
                "ERROR: non-real projected displacement field (internal bug)."
            )
        result = field.real.reshape(self.n_atoms, 3)
        result[np.abs(result) < 1e-12] = 0.0  # no signed-zero noise in output
        return result

    def _star_p_hat_H(self, arms, arm_index, dhats):
        """Subgroup projector V^dagger P_H V on the star block space.

        H = child operations x T_H translations; the T_H average is the
        diagonal 0/1 projector onto the arms in the reciprocal lattice of
        T_H, the rest is the average over the child-cell members."""
        S_H = np.asarray(self.mapping.S)
        keep = np.array(
            [np.all((np.asarray(arm) @ S_H.T) % DEN == 0) for arm in arms],
            dtype=float,
        )
        pi = np.repeat(keep, 3 * self.n_parent)
        size = len(arms) * 3 * self.n_parent
        P_H = np.zeros((size, size), dtype=np.complex128)
        for i, t in self.subgroup_members:
            phases = self._star_translation_phases(arms, t)
            P_H += (phases * pi)[:, None] * dhats[i]
        return P_H / len(self.subgroup_members)

    # -- the mode decomposition
    def _decompose(self):
        algebra = self.algebra
        # amplitudes: AMPLIMODES normalizes within the primitive cell of the
        # distorted structure (T_H); the core cell repeats it core/size times
        rescale = np.sqrt(self.size / self.core_size)

        u = self.u_cart.reshape(-1)
        total = np.linalg.norm(u)

        modes = []
        residual = u.copy()
        for star in self.stars:
            representations = self._star_representations(star)
            if not representations:
                continue
            arms = representations[0][1].arms
            for _, representation in representations[1:]:
                if not np.array_equal(representation.arms, arms):
                    raise SystemExit(
                        "ERROR: inconsistent star-arm ordering (internal bug)."
                    )
            arm_index = {
                tuple(int(v) % DEN for v in arm): a
                for a, arm in enumerate(arms)
            }
            negatives = [
                arm_index.get(tuple(int(v) % DEN for v in (-np.asarray(arm))))
                for arm in arms
            ]
            self_conjugate = all(n is not None for n in negatives)
            dhats = [
                self._star_dhat(arms, arm_index, i)
                for i in range(algebra.n_ops)
            ]
            P_H = self._star_p_hat_H(arms, arm_index, dhats)
            uhat = self._star_uhat(arms)

            np3 = 3 * self.n_parent
            completeness = np.zeros_like(P_H)
            for irrep_name, representation in representations:
                d_small = representation.dim_small
                d_tau = representation.dimension
                # complex isotypic projector P_tau = (d/n_ops) sum_i
                # delta_a(i)* [row-arm-a blocks of D(i)]
                P_c = np.zeros_like(P_H)
                for i in range(algebra.n_ops):
                    diag = np.diagonal(representation.blocks[i])
                    delta = np.conj(
                        np.add.reduceat(diag, np.arange(0, d_tau, d_small))
                    )
                    P_c += np.repeat(delta, np3)[:, None] * dhats[i]
                P_c *= d_tau / algebra.n_ops
                completeness += P_c
                # the real projector of the old dense route: Re chi* over a
                # real displacement space = (P_tau + conj(P_tau)) / 2, with
                # conj(P_tau) living on the -k arms
                if self_conjugate:
                    swapped = np.zeros_like(P_c)
                    for a in range(len(arms)):
                        for b in range(len(arms)):
                            swapped[
                                a * np3 : (a + 1) * np3, b * np3 : (b + 1) * np3
                            ] = np.conj(
                                P_c[
                                    negatives[a] * np3 : (negatives[a] + 1) * np3,
                                    negatives[b] * np3 : (negatives[b] + 1) * np3,
                                ]
                            )
                    P = 0.5 * (P_c + swapped)
                else:
                    P = 0.5 * P_c
                dim = int(round(float(np.trace(P @ P_H).real)))
                if dim <= 0:
                    continue
                projected_hat = P @ uhat
                projected = self._star_to_core(arms, projected_hat)
                residual = residual - projected.reshape(-1)
                amplitude = float(np.linalg.norm(projected_hat)) * rescale
                modes.append(
                    _ModeEntry(self, star["kname"], star["kvec"], irrep_name,
                               representation, dim, amplitude, projected)
                )
            if not np.allclose(
                completeness, np.eye(completeness.shape[0]), atol=1e-6
            ):
                raise SystemExit(
                    f"ERROR: mode-projector completeness check failed at the "
                    f"{star['kname']} star; please report this case."
                )
        if np.linalg.norm(residual) > 1e-3 * max(1.0, total):
            raise SystemExit(
                "ERROR: the distortion is not fully captured by the listed "
                "modes; please report this case."
            )
        self.total_distortion = total * rescale
        return modes

    def _star_representations(self, star):
        """[(irrep name, induced representation)] of one folding star."""
        algebra = self.algebra
        if star["kind"] == "tabulated":
            return [
                (irrep.name, InducedRepresentation(algebra, irrep.name))
                for irrep in algebra.irreps_by_kname[star["kname"]]
            ]
        canonical = star["canonical"]
        smalls = algebra.computed_irreps_at(canonical)
        names = star["names"]
        if names is None:
            names = [
                f"{star['kname']}.{index + 1}" for index in range(len(smalls))
            ]
            print(
                f"NOTE: the small irreps at the non-tabulated point "
                f"{star['kname']} could not be matched to ISO-IR labels; "
                "positional names are used."
            )
        # conjugate partners (for the paired label of doubled irreps)
        partners = []
        for index, small in enumerate(smalls):
            partner = None
            for other, candidate in enumerate(smalls):
                if other == index:
                    continue
                if set(candidate["chi"]) == set(small["chi"]) and all(
                    abs(candidate["chi"][op] - np.conj(small["chi"][op]))
                    < 1e-6
                    for op in small["chi"]
                ):
                    partner = names[other]
                    break
            partners.append(partner)
        # preferred basis: the bundled ISO-IR matrices (tabulated arm order,
        # deterministic across spgrep versions).  All-or-nothing per star so
        # every representation shares one arm ordering.
        if star["names"] is not None:
            try:
                return [
                    (
                        names[index],
                        ComputedInducedRepresentation.from_isoir(
                            algebra, canonical, small, names[index],
                            star["kname"], partners[index],
                        ),
                    )
                    for index, small in enumerate(smalls)
                ]
            except (LookupError, FileNotFoundError, ValueError):
                pass
        return [
            (
                names[index],
                ComputedInducedRepresentation(
                    algebra, canonical, small, names[index], star["kname"],
                    partners[index],
                ),
            )
            for index, small in enumerate(smalls)
        ]


class _ModeEntry:
    def __init__(self, analysis, kname, kvec, irrep_name, representation,
                 dim, amplitude, projected_u):
        self.analysis = analysis
        self.kname = kname
        self.kvec = np.asarray(kvec, dtype=float)
        self.irrep_name = irrep_name
        self.representation = representation
        self.dim = dim
        self.amplitude = amplitude
        # irrep-projected displacement field on the core cell, (n_atoms, 3)
        self.projected_u = projected_u
        self._label_info = None

    def label_info(self):
        """(direction label, subgroup info, index) via the isotropy machinery."""
        if self._label_info is not None:
            return self._label_info
        analyzer = IsotropyAnalyzer.from_representation(
            self.analysis.algebra, self.representation
        )

        # H on the representation's translation grid: the child-cell members
        # extended by the T_H lattice modulo the grid (T_H translations act
        # nontrivially on star arms outside the reciprocal lattice of T_H)
        N = self.representation.grid_n
        S_H = np.asarray(self.analysis.mapping.S, dtype=np.int64)
        tau_set = {
            tuple((np.asarray(z, dtype=np.int64) @ S_H) % N)
            for z in product(range(N), repeat=3)
        }
        members = []
        seen = set()
        for i, t in self.analysis.subgroup_members:
            for tau in tau_set:
                shifted = tuple(
                    (np.asarray(t, dtype=np.int64) + np.asarray(tau)) % N
                )
                if (i, shifted) not in seen:
                    seen.add((i, shifted))
                    members.append((i, np.asarray(shifted, dtype=np.int64)))
        fixed = analyzer.fixed_space(members)
        projector = _projector(fixed)
        label, _ = analyzer.direction_label(projector)
        stabilizer = analyzer.stabilizer_of(projector)
        info, size, index, *_ = analyzer.subgroup_of(stabilizer)
        self._label_info = (label, info, index)
        return self._label_info


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def _format_fraction(value: float) -> str:
    fraction = Fraction(value).limit_denominator(24)
    return str(fraction)


def _kvector_string(kvec) -> str:
    return "(" + ",".join(
        _format_fraction(v % 1.0) for v in np.asarray(kvec, dtype=float)
    ) + ")"


def _element_symbol(z: int) -> str:
    from pymatgen.core.periodic_table import Element

    return Element.from_Z(int(z)).symbol


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    analysis = SymmetryModeAnalysis(args.parent, args.child, args.tolerance)

    print("\n* Supergroup (parent) structure *")
    print(f"{analysis.parent_symbol} (No. {analysis.parent_number})")
    print("\n* Subgroup (distorted) structure *")
    print(f"{analysis.child_symbol} (No. {analysis.child_number})")
    if getattr(analysis, "strain_only_operations", 0) > 0:
        print(
            f"note: {analysis.strain_only_operations} parent operations that "
            "the child's own metric breaks leave its\natomic pattern exactly "
            "invariant -- that part of the symmetry lowering is a "
            "spontaneous\nstrain, which the displacive decomposition below "
            "does not carry (mode amplitudes are\nmeasured in the strain-free "
            "parent-derived reference lattice)."
        )

    mapping = analysis.mapping
    print("\n* Cell relation *")
    print("child primitive basis in parent primitive units (rows):")
    for row in mapping.S:
        print("  (" + ", ".join(str(int(v)) for v in row) + ")")
    print(f"origin shift (parent primitive fractional): ("
          + ", ".join(_format_fraction(v % 1.0) for v in mapping.p) + ")")
    print(f"primitive cell multiplication: {analysis.size}")
    rotation = getattr(analysis, "setting_rotation", None)
    if rotation is not None and rotation > 0.5:
        print(
            f"(setting: of {analysis.equivalent_settings} equivalent "
            "sublattice bases the one closest to the orientation of the\n "
            f"input files; the child axes are rotated {rotation:.1f} deg "
            "against the parent axes)"
        )

    print("\n* Atom pairings and displacements (parent primitive setting) *")
    print(f"{'atom':<5} {'reference':<28} {'displacement (frac)':<28} |u| (A)")
    u_cart_cell = mapping.u_frac @ analysis.L_parent
    max_u = 0.0
    for j in range(len(mapping.ref_frac)):
        z = mapping.child_z[j]
        norm = np.linalg.norm(u_cart_cell[j])
        max_u = max(max_u, norm)
        ref = mapping.ref_frac[j]
        line = (
            f"{_element_symbol(z):<5} "
            f"({ref[0]:.5f},{ref[1]:.5f},{ref[2]:.5f})".ljust(29)
            + f"({mapping.u_frac[j][0]:+.5f},{mapping.u_frac[j][1]:+.5f},"
              f"{mapping.u_frac[j][2]:+.5f})".ljust(29)
            + f"{norm:.4f}"
        )
        print(line)
    print(f"\nmaximum atomic displacement: {max_u:.4f} A")
    print(f"total distortion amplitude : {analysis.total_distortion:.4f} A")
    print("(normalized within the primitive cell of the distorted structure)")
    if getattr(analysis, "polar_directions", 0) > 0:
        print(
            "note: the subgroup is polar; the free origin is placed at the "
            "minimum of the total\ndistortion (AMPLIMODES convention, no "
            "acoustic translation in the modes) -- programs\npinning the "
            "origin differently (e.g. ISODISTORT) report different "
            "amplitudes for the\npolar irreps."
        )

    table_lines = ["* Symmetry-mode decomposition *"]
    table_lines.append(
        f"{'k-vector':<16} {'irrep':<7} {'direction':<12} "
        f"{'isotropy subgroup':<19} {'dim':<4} amplitude (A)"
    )
    for mode in analysis.modes:
        label, info, index = mode.label_info()
        subgroup = f"{info.number} {info.international_short}"
        table_lines.append(
            f"{_kvector_string(mode.kvec):<16} "
            f"{mode.irrep_name:<7} {label:<12} {subgroup:<19} "
            f"{mode.dim:<4} {mode.amplitude:.4f}"
        )
    if any(star["kind"] == "computed" for star in analysis.stars):
        table_lines.append(
            "(non-special k points: the order-parameter components are "
            "expressed in the bundled\n ISO-IR matrix basis, whose phase "
            "gauge may differ from the ISOTROPY web tables;\n the isotropy "
            "subgroup, dimension and amplitude are gauge-independent)"
        )
    print("\n" + "\n".join(table_lines))

    # the table again as a text file, named by the parent composition
    table_path = f"sym_mode_{analysis.parent_formula}"
    with open(table_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(table_lines) + "\n")
    print(f"\nDecomposition table saved to {table_path}")

    print("\n* Normalized mode components (parent primitive fractional, per 1 A) *")
    for mode in analysis.modes:
        if mode.amplitude < 1e-4:
            print(f"{mode.irrep_name}: amplitude 0 (allowed but not activated)")
            continue
        direction = mode.projected_u.reshape(-1)
        # normalize to 1 A within the primitive cell of the distorted structure
        direction = direction / np.linalg.norm(direction) * np.sqrt(
            analysis.core_size / analysis.size
        )
        frac = (direction.reshape(-1, 3) @ np.linalg.inv(analysis.L_parent))
        parts = []
        for j in range(analysis.n_atoms):
            if np.linalg.norm(direction.reshape(-1, 3)[j]) > 1e-6:
                parts.append(
                    f"{_element_symbol(analysis.ref_z[j])}"
                    f"({analysis.ref_frac[j][0]:.3f},"
                    f"{analysis.ref_frac[j][1]:.3f},"
                    f"{analysis.ref_frac[j][2]:.3f}): "
                    f"({frac[j][0]:+.4f},{frac[j][1]:+.4f},{frac[j][2]:+.4f})"
                )
        print(f"{mode.irrep_name}:")
        for part in parts:
            print(f"  {part}")

    _export_mode_vesta_files(analysis, args.parent, conventional=args.conventional)

    print("\nConventions and validation: AMPLIMODES (Bilbao Crystallographic "
          "Server):")
    print('D. Orobengoa, C. Capillas, M. I. Aroyo and J. M. Perez-Mato,')
    print('"AMPLIMODES: symmetry-mode analysis on the Bilbao Crystallographic')
    print('Server", J. Appl. Cryst. 42, 820-833 (2009).')
    print()


_VESTA_ARROW_LENGTH = 1.5  # A, largest arrow per file (same as --vector)


def _conventional_display_cell(analysis):
    """Display supercell for --conventional VESTA output.

    Returns (D, translations): D is the smallest diagonal multiple of the
    parent conventional cell (rows in parent primitive units) whose lattice
    is a sublattice of the invariant-core lattice, so every mode pattern is
    periodic over the display cell; translations are the core-lattice
    representatives that tile the display cell.
    """
    from math import gcd

    from .phonon_vector import get_conventional_matrix

    conventional = get_conventional_matrix(analysis.parent_symbol[0])
    S_core = np.asarray(analysis.S_core)
    S_core_inv = np.linalg.inv(S_core)
    sizes = []
    for row in conventional:
        multiple = 1
        for component in np.asarray(row, dtype=float) @ S_core_inv:
            denominator = Fraction(float(component)).limit_denominator(48).denominator
            multiple = multiple * denominator // gcd(multiple, denominator)
        sizes.append(multiple)
    D = np.diag(sizes) @ conventional
    D_inv = np.linalg.inv(D)
    n_copies = int(round(abs(np.linalg.det(D)) / abs(np.linalg.det(S_core))))
    bound = int(np.ceil(np.abs(np.asarray(D, dtype=float) @ S_core_inv).sum())) + 1
    translations = []
    seen = set()
    for m in product(range(-bound, bound + 1), repeat=3):
        t = np.asarray(m) @ S_core
        frac = np.mod(np.round(t @ D_inv, 8), 1.0)
        key = tuple(np.round(frac, 6))
        if key in seen:
            continue
        seen.add(key)
        translations.append(t)
        if len(translations) == n_copies:
            break
    return D, translations


def _export_mode_vesta_files(analysis, parent_path: str,
                             conventional: bool = False) -> None:
    """Write one VESTA file per activated irrep showing its displacement
    pattern: arrows of the irrep-projected distortion on the parent-derived
    reference structure, in the invariant-core cell (default) or in the
    parent conventional basis (--conventional, _conv suffix).
    """
    from .phonon_vector import write_vesta_with_arrows

    parent_base = os.path.basename(parent_path)
    if parent_base.lower().endswith(".cif"):
        parent_base = parent_base[: -len(".cif")]

    if conventional:
        D, translations = _conventional_display_cell(analysis)
        lattice = D @ analysis.L_parent
        D_inv = np.linalg.inv(D)
        positions = []
        atom_source = []
        for j in range(analysis.n_atoms):
            for t in translations:
                positions.append(
                    np.mod((analysis.ref_frac[j] + t) @ D_inv, 1.0)
                )
                atom_source.append(j)
        scaled_positions = np.array(positions)
        suffix = "_conv"
        cell_note = "parent conventional basis"
    else:
        lattice = analysis.S_core @ analysis.L_parent
        scaled_positions = np.mod(
            analysis.ref_frac @ np.linalg.inv(analysis.S_core), 1.0
        )
        atom_source = list(range(analysis.n_atoms))
        suffix = ""
        cell_note = "invariant-core cell"
    symbols = [_element_symbol(analysis.ref_z[j]) for j in atom_source]

    written = []
    for mode in analysis.modes:
        if mode.amplitude < 1e-4:
            continue
        arrows_core = mode.projected_u
        peak = float(np.max(np.linalg.norm(arrows_core, axis=1)))
        if peak < 1e-10:
            continue
        arrows = arrows_core[atom_source] * (_VESTA_ARROW_LENGTH / peak)
        filename = f"{parent_base}_{mode.irrep_name}{suffix}.vesta"
        write_vesta_with_arrows(
            filepath=filename,
            lattice=lattice,
            scaled_positions=scaled_positions,
            symbols=symbols,
            arrows_cartesian=arrows,
            title=(
                f"{parent_base} {mode.irrep_name} mode "
                f"(amplitude {mode.amplitude:.4f} A)"
            ),
        )
        written.append((filename, mode.amplitude))

    if not written:
        return
    print(f"\n* Mode displacement VESTA files ({cell_note}) *")
    if conventional:
        print("display cell in parent primitive units (rows):")
        for row in D:
            print("  (" + ", ".join(str(int(v)) for v in row) + ")")
    for filename, amplitude in written:
        print(f"  {filename}  (amplitude {amplitude:.4f} A)")
    print(f"Arrows are scaled so the largest displacement is "
          f"{_VESTA_ARROW_LENGTH} A per file; adjust in VESTA via "
          "Edit > Vectors if needed.")


if __name__ == "__main__":
    main()
