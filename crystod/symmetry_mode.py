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
irrep matrices (the same machinery as --parent).  When the minimum-
distortion mapping is not a group-subgroup setting (some operation of the
child is not an exact parent operation in it, or the pairing breaks one),
the mapping is redone among the settings in which every child operation is
a parent operation, with the origin solved exactly from the translation
parts and a symmetry-consistent pairing.  A complex-type irrep and its
conjugate enter as one physically irreducible pair (label GM3+GM4+, the
ISODISTORT convention).  Amplitudes follow the
AMPLIMODES convention: A = sqrt(sum |u_atom|^2) over the primitive cell of
the distorted structure, with Cartesian displacements measured in the
strain-free parent-derived reference lattice.  A completeness check
(sum of all projectors = identity on the displacement space) closes every
run.
"""

from __future__ import annotations

import argparse
import os
import textwrap
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
    files = parser.add_mutually_exclusive_group()
    files.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        metavar="DIR",
        help="Directory for the decomposition table (sym_mode_<formula>) and "
        "the per-irrep VESTA files, created if missing (default: the current "
        "directory).",
    )
    files.add_argument(
        "--no-files",
        dest="no_files",
        action="store_true",
        help="Print the analysis only; write no table or VESTA files.",
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
        # a POSCAR means the cell VASP reads from it (scale line, Cartesian
        # coordinates); a CIF is left as pymatgen read it
        from .vasp_io import poscar_structure

        structure = poscar_structure(structure, path)
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
    third = np.array(norms[2], dtype=np.int64).reshape(-1, 3)
    candidates = []
    for v1 in norms[0]:
        for v2 in norms[1]:
            # det [v1; v2; v3] = (v1 x v2) . v3, exact in integers, for all
            # v3 at once (the loop over v3 dominated the run time of large
            # cell multiplications); same candidates in the same order
            for index in np.nonzero(third @ np.cross(v1, v2) == n)[0]:
                S = np.array([v1, v2, third[index]])
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
            # AMPLIMODES minimum-distortion origin) and re-pair once; the
            # origin moves only together with an accepted re-pairing, so the
            # stored origin is the one the stored positions were paired at
            for _ in range(2):
                mean = np.mean([u for _, u in result[1]], axis=0)
                if np.linalg.norm(mean @ L_parent) < 1e-8:
                    break
                refined = pair_with(p - mean)
                if refined is None:
                    break
                p = p - mean
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
# group-subgroup consistent mapping
# ---------------------------------------------------------------------------
#
# The minimum-distortion mapping above chooses the sublattice basis S and
# the origin p by the distortion alone.  Nothing makes the child's own
# operations parent operations under that (S, p): a strained twin setting
# or a shifted origin can pair the atoms marginally better (a pseudo-cubic
# parent offers many), and then only part of the child group survives as
# exact parent operations.  When that happens the mapping is redone in the
# settings where it does (SymmetryModeAnalysis._consistent_mapping).

# largest displacement accepted for one atom of a pairing (Angstrom)
_MAX_PAIR_DISTANCE = 1.8

# a displacement field counts as invariant under an operation when
# |u(g x) - R u(x)| stays under this (Angstrom): the child is idealized by
# spglib, so a symmetric pairing is symmetric to machine precision, while a
# pairing that breaks the symmetry does so by an interatomic distance
_FIELD_INVARIANT = 1e-6


class _MappingRejected(Exception):
    """The minimum-distortion mapping is not a group-subgroup setting; the
    message says why (it goes into the NOTE of the redone mapping)."""


def _smith_diagonal(A):
    """(U, D, V), integer, with U A V = D diagonal and U, V unimodular.

    The divisibility chain of the Smith normal form is not needed here (only
    the solution set of a congruence system is), so the reduction stops as
    soon as the matrix is diagonal; the nonzero entries come first."""
    D = [[int(v) for v in row] for row in np.asarray(A)]
    m, n = len(D), len(D[0])
    U = [[int(i == j) for j in range(m)] for i in range(m)]
    V = [[int(i == j) for j in range(n)] for i in range(n)]
    for t in range(min(m, n)):
        while True:
            entries = [(abs(D[i][j]), i, j) for i in range(t, m)
                       for j in range(t, n) if D[i][j]]
            if not entries:
                return np.array(U), np.array(D), np.array(V)
            _, i0, j0 = min(entries)
            D[t], D[i0] = D[i0], D[t]
            U[t], U[i0] = U[i0], U[t]
            for row in D:
                row[t], row[j0] = row[j0], row[t]
            for row in V:
                row[t], row[j0] = row[j0], row[t]
            pivot = D[t][t]
            clean = True
            for i in range(t + 1, m):
                q = D[i][t] // pivot
                if q:
                    D[i] = [a - q * b for a, b in zip(D[i], D[t])]
                    U[i] = [a - q * b for a, b in zip(U[i], U[t])]
                clean = clean and D[i][t] == 0
            for j in range(t + 1, n):
                q = D[t][j] // pivot
                if q:
                    for row in D:
                        row[j] -= q * row[t]
                    for row in V:
                        row[j] -= q * row[t]
                clean = clean and D[t][j] == 0
            if clean:
                break
    return np.array(U), np.array(D), np.array(V)


def _origin_solutions(rotations, rhs):
    """Every origin x (mod Z^3) with (I - W_k) x = rhs_k (mod Z^3) for all k.

    Returns ``(solutions, free, residual)``: the particular solutions (one
    per coset of the discrete solution set, polar components zero), the
    free (polar) directions as integer columns, and the largest violation
    of the compatibility conditions (nonzero when the translation parts of
    the child operations cannot be matched by any origin)."""
    A = np.vstack([np.eye(3, dtype=np.int64) - np.asarray(W, dtype=np.int64)
                   for W in rotations])
    c = np.concatenate([np.asarray(r, dtype=float) for r in rhs])
    U, D, V = _smith_diagonal(A)
    Uc = U.astype(float) @ c
    diagonal = [int(D[k][k]) for k in range(3)]
    rank = sum(1 for d in diagonal if d != 0)
    tail = Uc[rank:]
    residual = float(np.max(np.abs(tail - np.rint(tail)))) if len(tail) else 0.0
    choices = [
        [(Uc[k] + j) / diagonal[k] for j in range(abs(diagonal[k]))]
        for k in range(rank)
    ]
    solutions = []
    for combo in product(*choices):
        y = np.zeros(3)
        y[:rank] = combo
        solutions.append((V.astype(float) @ y) % 1.0)
    return solutions, V[:, rank:].astype(float), residual


def _conjugated_child_operations(algebra, S, child_ops):
    """The child operations carried into the parent primitive setting by the
    sublattice basis S (child fractional x -> parent fractional S^T x):
    ``[(i, W, w0)]`` with W = S^T R S^-T the rotation of parent operation i
    and w0 = S^T tau the translation part before the origin shift, or None
    when some child rotation is not a parent rotation (S does not carry the
    child point group)."""
    S = np.asarray(S, dtype=float)
    S_inv = np.linalg.inv(S)
    index = {
        np.asarray(W, dtype=np.int64).tobytes(): i
        for i, W in enumerate(algebra.rotations)
    }
    result = []
    for R, tau in zip(child_ops["rotations"], child_ops["translations"]):
        W = S.T @ np.asarray(R, dtype=float) @ S_inv.T
        W_int = np.rint(W).astype(np.int64)
        if not np.allclose(W, W_int, atol=1e-6):
            return None
        i = index.get(W_int.tobytes())
        if i is None:
            return None
        result.append((i, W_int.astype(float), S.T @ np.asarray(tau, dtype=float)))
    return result


def _reference_cell(parent_positions, parent_numbers, parent_orbits, S):
    """Parent atoms repeated over Z^3 / (rows of S): the strain-free
    reference of the child cell (positions in parent primitive fractional
    coordinates, atomic numbers, parent orbit ids)."""
    ref_frac, ref_z, ref_orbit = [], [], []
    for x, z, orbit in zip(parent_positions, parent_numbers, parent_orbits):
        for t in _translation_reps(S):
            ref_frac.append(np.asarray(x, dtype=float) + t)
            ref_z.append(int(z))
            ref_orbit.append(orbit)
    return np.array(ref_frac), np.array(ref_z), ref_orbit


def _operation_permutations(points, numbers, operations, S, tol):
    """Per operation (W, w): the index of the image W x_j + w of every point
    (same species, modulo the rows of S), as an array of shape (n_ops, n);
    None when some image is not one of the points."""
    S = np.asarray(S, dtype=float)
    S_inv = np.linalg.inv(S)
    same = numbers[:, None] == numbers[None, :]
    permutations = []
    for W, w in operations:
        image = points @ np.asarray(W, dtype=float).T + w
        d = image[:, None, :] - points[None, :, :]
        d = d - np.rint(d @ S_inv) @ S
        hit = np.all(np.abs(d) < tol, axis=2) & same
        if not np.all(hit.sum(axis=1) == 1):
            return None
        permutations.append(np.argmax(hit, axis=1))
    return np.array(permutations)


def _minimum_image(d, S, L_parent):
    """Shortest lattice images (rows of S) of the fractional difference
    vectors d (..., 3) in the parent metric: (vectors, lengths in A)."""
    S = np.asarray(S, dtype=float)
    d = d - np.rint(d @ np.linalg.inv(S)) @ S
    shifts = np.array(list(product((-1, 0, 1), repeat=3)), dtype=float) @ S
    trial = d[..., None, :] + shifts
    lengths = np.linalg.norm(trial @ L_parent, axis=-1)
    best = np.argmin(lengths, axis=-1)
    vectors = np.take_along_axis(trial, best[..., None, None], axis=-2)[..., 0, :]
    return vectors, np.take_along_axis(lengths, best[..., None], axis=-1)[..., 0]


def _pair_by_assignment(ref_frac, ref_z, child_par, child_z, S, L_parent):
    """Minimum total squared distortion pairing (optimal assignment per
    species, minimum image over the rows of S): ``(total, u_frac)`` with
    u_frac indexed like the reference, or None when some atom moves farther
    than _MAX_PAIR_DISTANCE."""
    from scipy.optimize import linear_sum_assignment

    u = np.zeros((len(ref_frac), 3))
    total = 0.0
    for z in np.unique(ref_z):
        rows = np.where(child_z == z)[0]
        cols = np.where(ref_z == z)[0]
        if len(rows) != len(cols):
            return None
        vectors, lengths = _minimum_image(
            child_par[rows][:, None, :] - ref_frac[cols][None, :, :], S, L_parent
        )
        r_index, c_index = linear_sum_assignment(lengths**2)
        if np.max(lengths[r_index, c_index]) > _MAX_PAIR_DISTANCE:
            return None
        u[cols[c_index]] = vectors[r_index, c_index]
        total += float(np.sum(lengths[r_index, c_index] ** 2))
    return total, u


def _pair_equivariantly(ref_frac, ref_z, child_par, child_z, S, L_parent,
                        rotations, ref_perm, child_perm):
    """Pairing that commutes with the child operations, orbit by orbit: an
    orbit representative of the reference takes the nearest unused child
    atom with the same stabilizer, and the rest of the orbit follows by
    symmetry, so the displacement field is invariant by construction.  A
    child atom sitting half a lattice vector from its site is rejected (its
    site symmetry would not fix the displacement itself, only modulo the
    lattice).  Returns ``(total, u_frac)`` or None."""
    n = len(ref_frac)
    representatives = []
    seen = set()
    for j in range(n):
        if j in seen:
            continue
        seen |= set(int(v) for v in ref_perm[:, j])
        representatives.append(j)
    ref_stab = [frozenset(np.nonzero(ref_perm[:, j] == j)[0]) for j in range(n)]
    child_stab = [frozenset(np.nonzero(child_perm[:, c] == c)[0])
                  for c in range(n)]
    options = []
    for j in representatives:
        candidates = [c for c in range(n)
                      if child_z[c] == ref_z[j] and child_stab[c] == ref_stab[j]]
        if not candidates:
            return None
        vectors, lengths = _minimum_image(
            child_par[candidates] - ref_frac[j], S, L_parent
        )
        for c, u, length in zip(candidates, vectors, lengths):
            if length > _MAX_PAIR_DISTANCE:
                continue
            if any(not np.allclose(rotations[k] @ u, u, atol=1e-6)
                   for k in ref_stab[j]):
                continue
            options.append((float(length), j, c, u))
    options.sort(key=lambda item: item[0])
    u_frac = np.zeros((n, 3))
    paired = np.full(n, -1)
    used = np.zeros(n, dtype=bool)
    done = set()
    for _, j, c, u in options:
        if j in done:
            continue
        targets_ref = ref_perm[:, j]
        targets_child = child_perm[:, c]
        if np.any(used[targets_child]):
            continue
        for k, W in enumerate(rotations):
            u_frac[targets_ref[k]] = W @ u
            paired[targets_ref[k]] = targets_child[k]
        used[targets_child] = True
        done.add(j)
    if len(done) != len(representatives) or np.any(paired < 0):
        return None
    return float(np.sum(np.linalg.norm(u_frac @ L_parent, axis=1) ** 2)), u_frac


def _field_violation(u_frac, rotations, ref_perm, L_parent):
    """Largest |u(g x) - R_g u(x)| (A) over the reference atoms of the
    child cell and the operations (rotations + atom permutations)."""
    worst = 0.0
    for W, perm in zip(rotations, ref_perm):
        difference = u_frac[perm] - u_frac @ np.asarray(W, dtype=float).T
        worst = max(worst, float(np.max(np.linalg.norm(
            difference @ L_parent, axis=1))))
    return worst


def _consistent_pairing(ref_frac, ref_z, child_cell, child_z, S, L_parent,
                        base, polar, rotations, ref_perm, child_perm):
    """Best symmetry-consistent pairing for one origin class: ``(total,
    u_frac, origin)`` or None.

    ``child_cell`` are the child atoms in parent primitive fractional
    coordinates before the origin shift, ``base`` the exact origin of the
    class and ``polar`` the Cartesian projector onto its free (polar)
    directions (None for a non-polar child).  At every trial origin the
    optimal assignment is kept when its field is invariant under the child
    operations, otherwise the orbit-wise pairing is used.  Along polar
    directions the trial origins put the first child atom of every species
    level with each reference atom of its species (through the lattice
    image whose non-polar remainder is shortest: in a skew basis a lattice
    vector is not orthogonal to the polar directions), and each is refined
    twice by the polar part of the mean displacement (the AMPLIMODES
    minimum-distortion origin)."""

    def pair(origin):
        child_par = child_cell + origin
        result = _pair_by_assignment(ref_frac, ref_z, child_par, child_z, S,
                                     L_parent)
        if result is not None and _field_violation(
                result[1], rotations, ref_perm, L_parent) <= _FIELD_INVARIANT:
            return result
        return _pair_equivariantly(ref_frac, ref_z, child_par, child_z, S,
                                   L_parent, rotations, ref_perm, child_perm)

    if polar is None:
        result = pair(base)
        return None if result is None else (result[0], result[1], base)

    L_inv = np.linalg.inv(L_parent)
    S_inv = np.linalg.inv(S)
    images = np.array(list(product((-1, 0, 1), repeat=3)), dtype=float) @ S

    def polar_part(delta):
        return (polar @ (delta @ L_parent)) @ L_inv

    starts = []
    seen = set()
    anchors = {}
    for index, z in enumerate(child_z):
        anchors.setdefault(int(z), index)
    for z, anchor in anchors.items():
        for x in ref_frac[ref_z == z]:
            delta = x - (child_cell[anchor] + base)
            delta = delta - np.rint(delta @ S_inv) @ S
            cartesian = (delta + images) @ L_parent
            along = cartesian @ polar
            k = int(np.argmin(np.linalg.norm(cartesian - along, axis=1)))
            start = base + along[k] @ L_inv
            key = tuple(np.round(start % 1.0, 6) % 1.0)
            if key not in seen:
                seen.add(key)
                starts.append(start)
    best = None
    for origin in starts:
        result = pair(origin)
        for step in range(3):
            if result is None:
                break
            if best is None or result[0] < best[0] - 1e-12:
                best = (result[0], result[1], origin)
            if step == 2:
                break
            shift = polar_part(np.mean(result[1], axis=0))
            if np.linalg.norm(shift @ L_parent) < 1e-10:
                break
            refined = pair(origin - shift)
            if refined is None:
                break
            origin = origin - shift
            result = refined
    return best


def _wyckoff_conflict(ref_z, child_z, ref_perm, child_perm):
    """(operation position, Z, reference sites fixed, child atoms fixed) for
    the first operation that fixes a different number of sites of one
    species in the reference and in the child, or None.  Any such conflict
    rules out a pairing that commutes with the operations."""
    n = len(ref_z)
    for k in range(len(ref_perm)):
        ref_fixed = ref_perm[k] == np.arange(n)
        child_fixed = child_perm[k] == np.arange(n)
        for z in sorted(set(int(v) for v in ref_z)):
            n_ref = int(np.sum(ref_fixed & (ref_z == z)))
            n_child = int(np.sum(child_fixed & (child_z == z)))
            if n_ref != n_child:
                return k, z, n_ref, n_child
    return None


def _point_operation_name(W) -> str:
    """Kind of a point operation from its (integer) matrix: '2-fold
    rotation', 'mirror', 'inversion', ..."""
    W = np.asarray(W, dtype=float)
    det = int(round(np.linalg.det(W)))
    trace = int(round(np.trace(W)))
    if det > 0:
        return {3: "identity", -1: "2-fold rotation", 0: "3-fold rotation",
                1: "4-fold rotation", 2: "6-fold rotation"}.get(trace, "rotation")
    return {-3: "inversion", 1: "mirror", 0: "-3 rotoinversion",
            -1: "-4 rotoinversion", -2: "-6 rotoinversion"}.get(
                trace, "rotoinversion")


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
        mapping_note: None, or the NOTE (printed by the command with the
            cell relation) that the minimum-distortion mapping was not a
            group-subgroup setting and the atoms were re-paired in a
            setting in which every child operation is a parent operation.
        core_size: Multiplication of the invariant-core analysis cell; its
            atoms are ``ref_frac`` (``n_atoms`` of them) with atomic
            numbers ``ref_z`` and Cartesian displacements ``u_cart``.
        subgroup_members: The ``(i, t)`` parent operations that leave the
            distorted structure invariant.
        stars: The parent k stars folding to the child Gamma point, one
            dict per star with ``kname``, ``kvec`` (primitive basis) and
            ``kind`` (``"tabulated"`` or ``"computed"``).
        modes: One entry per parent irrep with a nonzero number of modes
            (a complex-type irrep and its conjugate form one entry, the
            physically irreducible pair, ``irrep_name`` joining both labels
            as in ``GM3+GM4+``; a pseudoreal irrep keeps its single name),
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
            cell, the child cannot be mapped onto the parent (no sublattice
            setting carries the child point group, the translation parts
            of the child operations fit no origin, a Wyckoff conflict, or no
            symmetry-consistent pairing within 1.8 A), or an internal
            consistency check (mode completeness) fails.

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
        self._sublattice_bases = candidates
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
        self.mapping_note = None
        try:
            self.subgroup_members = self._find_subgroup_members()
        except _MappingRejected as rejected:
            self.subgroup_members = self._consistent_mapping(str(rejected))
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
        # the child atoms moved with the origin: keep the printed origin
        # shift that of the stored positions (a polar shift commutes with
        # every member, so the members stay as they are)
        self.mapping.p = np.asarray(self.mapping.p, dtype=float) - shift_frac

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
        """Parent operations that survive in the child.

        Every (operation, translation) candidate gets a mismatch distance:
        the worst atom-to-nearest-partner distance of the transformed child
        structure.  The child is idealized by spglib, so in a group-subgroup
        setting its own operations are parent operations that leave the
        mapped structure invariant to machine precision (_EXACT_MISMATCH),
        however small the symmetry-breaking part of the distortion is
        (pseudo-symmetric structures break their other operations by as
        little as ~0.001 A, strong tilts by far more): the members are the
        exact operations, and the child's own space group fixes how many
        there must be at least.

        The minimum-distortion mapping is accepted only when it passes three
        checks: at least that many exact operations, every child operation
        (carried over by the sublattice basis and the origin) among them,
        and a displacement field invariant under them to machine precision
        (_FIELD_INVARIANT; an atom pairing can break the symmetry of an
        invariant structure by exchanging atoms).  None of them depends on
        --tolerance.  Otherwise _MappingRejected is raised and the mapping
        is redone in a group-subgroup setting (_consistent_mapping).

        The search runs on the child (T_H) cell; translations are the
        representatives of Z^3 / T_H."""
        import spglib

        candidates = self._operation_mismatches()

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
        self._child_ops = spglib.get_symmetry(self.child_prim, symprec=1e-5)
        expected = len(self._child_ops["rotations"])
        exact = [entry[0] for entry in candidates if entry[1] < _EXACT_MISMATCH]
        if len(exact) < min(expected, len(candidates)):
            verb = "is a parent operation" if len(exact) == 1 else (
                "are parent operations")
            raise _MappingRejected(
                f"only {len(exact)} of the {expected} operations of the child "
                f"{verb} in it; the next one is broken by "
                f"{candidates[len(exact)][1]:.2e} A"
            )
        # the count alone is not enough: exact strain-only operations could
        # make it up while a child operation is missing (a twin setting)
        conjugated = _conjugated_child_operations(
            self.algebra, self.mapping.S, self._child_ops
        )
        if conjugated is None:
            raise _MappingRejected(
                "its sublattice basis does not carry the point group of the "
                "child onto parent rotations"
            )
        own = self._child_operation_members(conjugated, self.mapping)
        if own is None:
            raise _MappingRejected(
                "the translation parts of the child operations are not those "
                "of parent operations at its origin"
            )
        listed = {(i, tuple(int(v) for v in t)) for i, t in exact}
        for (i, t), (_, W, _) in zip(own, conjugated):
            if (i, tuple(int(v) for v in t)) not in listed:
                kind = _point_operation_name(W)
                article = "an" if kind[0] in "aeiou" else "a"
                raise _MappingRejected(
                    f"{article} {kind} of the child is not an exact parent "
                    "operation in it"
                )
        # a symmetric pairing of the (idealized) child is symmetric to
        # machine precision; one that exchanges atoms breaks the symmetry
        # by about an interatomic distance, however weak the distortion
        violation = self._member_field_violation(exact)
        if violation > _FIELD_INVARIANT:
            raise _MappingRejected(
                "its atom pairing breaks the child symmetry by "
                f"{violation:.2e} A"
            )
        if expected < len(candidates):
            self.strain_only_operations = len(exact) - expected
        return exact

    def _child_operation_members(self, conjugated, mapping):
        """The child operations ``[(i, W, w0)]`` (_conjugated_child_operations)
        as members (i, t) on the T_H cell of ``mapping`` -- t is the lattice
        part w0 + (I - W) p - v_i, as a representative of Z^3 / T_H -- or
        None when some of these is not a lattice vector (the origin p does
        not make that child operation a parent operation)."""
        identity = np.eye(3)
        S_inv = np.linalg.inv(np.asarray(mapping.S, dtype=float))
        cosets = {
            tuple(np.round((np.asarray(t, dtype=float) @ S_inv) % 1.0, 6) % 1.0): t
            for t in _translation_reps(mapping.S)
        }
        p = np.asarray(mapping.p, dtype=float)
        members = []
        for i, W, w0 in conjugated:
            T = (w0 + (identity - W) @ p
                 - np.array(self.algebra.translations[i], dtype=float) / DEN)
            T_int = np.rint(T)
            key = tuple(np.round((T_int @ S_inv) % 1.0, 6) % 1.0)
            if not np.allclose(T, T_int, atol=1e-5) or key not in cosets:
                return None
            members.append((i, np.asarray(cosets[key], dtype=np.int64)))
        return members

    def _operation_mismatches(self):
        """Sorted ``((i, t), worst)`` over the parent operations (i, t) on the
        child (T_H) cell: the worst atom-to-nearest-partner distance (A) of
        the transformed child structure of the current mapping."""
        algebra = self.algebra
        S = np.asarray(self.mapping.S, dtype=float)
        S_inv = np.linalg.inv(S)
        reps = _translation_reps(self.mapping.S)
        positions = self.mapping.ref_frac + self.mapping.u_frac
        numbers = np.asarray(self.mapping.child_z)
        other_species = numbers[:, None] != numbers[None, :]
        candidates = []
        for i in range(algebra.n_ops):
            W = np.asarray(algebra.rotations[i], dtype=float)
            v = np.array(algebra.translations[i], dtype=float) / DEN
            moved = positions @ W.T + v
            for t in reps:
                d = (moved + t)[:, None, :] - positions[None, :, :]
                d = d - np.rint(d @ S_inv) @ S
                distances = np.linalg.norm(d @ self.L_parent, axis=-1)
                distances[other_species] = np.inf
                worst = float(np.max(np.min(distances, axis=1)))
                candidates.append(((i, np.asarray(t, dtype=np.int64)), worst))
        candidates.sort(key=lambda entry: entry[1])
        return candidates

    def _member_field_violation(self, members):
        """Largest |u(g x) - R_g u(x)| (A) over the core-cell atoms and the
        (i, t) members: zero when the displacement field is invariant under
        them, an interatomic distance when the atom pairing breaks them."""
        if not members:
            return 0.0
        n_t = len(self.reps_core)
        S_core_inv = np.linalg.inv(np.asarray(self.S_core, dtype=float))
        # integer labels of the core-lattice cosets: x S_core^-1 has the
        # denominator det S_core = n_t
        def labels(vectors):
            k = np.rint(vectors @ S_core_inv * n_t).astype(np.int64) % n_t
            return (k[..., 0] * n_t + k[..., 1]) * n_t + k[..., 2]

        reps = np.asarray(self.reps_core, dtype=float)
        rep_labels = labels(reps)
        order = np.argsort(rep_labels)
        sorted_labels = rep_labels[order]
        u = self.u_cart.reshape(self.n_parent, n_t, 3)
        worst = 0.0
        for i, t in members:
            img, tau, R = self.op_tables[i]
            W = np.asarray(self.algebra.rotations[i], dtype=float)
            shift = (tau[:, None, :] + (reps @ W.T)[None, :, :]
                     + np.asarray(t, dtype=float))
            wanted = labels(shift)
            position = np.minimum(np.searchsorted(sorted_labels, wanted), n_t - 1)
            if not np.array_equal(sorted_labels[position], wanted):
                return np.inf
            target = order[position]
            difference = u[img[:, None], target] - u @ R.T
            worst = max(worst, float(np.max(np.linalg.norm(difference, axis=-1))))
        return worst

    # -- the group-subgroup consistent mapping
    def _consistent_mapping(self, reason):
        """Redo the atom mapping in a group-subgroup setting; returns the
        subgroup members.

        The child operations (spglib, on the idealized child cell) are
        carried into the parent primitive setting through every candidate
        sublattice basis S; a basis qualifies when each child rotation
        becomes a parent rotation, W = S^T R S^-T.  The origin is solved
        exactly from the translation parts, (I - W) x = v_W - S^T tau
        (mod Z^3), one solution per coset of the solution set (Smith normal
        form); the polar directions stay free and are fixed by putting the
        first child atom of every species level with each reference atom of
        its species, then refined to the minimum-distortion origin.  At
        each origin the atoms are paired by the optimal assignment, which is
        accepted only when its displacement field is invariant under the
        child operations; otherwise they are paired orbit by orbit
        (_pair_equivariantly).  The least total distortion wins (ties: the
        orientation rule of _select_setting).  The members are the child
        operations themselves, plus every further parent operation that
        leaves both the mapped structure and the displacement field
        invariant (the strain-only part of a ferroelastic symmetry
        lowering); a final check requires the structure and the field to be
        invariant under every member to machine precision
        (_FIELD_INVARIANT)."""
        algebra = self.algebra
        child_ops = self._child_ops
        expected = len(child_ops["rotations"])
        _, child_positions, child_numbers = self.child_prim
        child_positions = np.asarray(child_positions, dtype=float)
        child_z = np.array([int(z) for z in child_numbers])
        identity = np.eye(3)
        found = []  # the best mapping per sublattice basis
        conflicts = []
        n_embedding = n_compatible = n_unpaired = 0
        for S_int in self._sublattice_bases:
            conjugated = _conjugated_child_operations(algebra, S_int, child_ops)
            if conjugated is None:
                continue
            n_embedding += 1
            rotations = [W for _, W, _ in conjugated]
            origins, free, residual = _origin_solutions(
                rotations,
                [np.array(algebra.translations[i], dtype=float) / DEN - w0
                 for i, _, w0 in conjugated],
            )
            if residual > 1e-5:
                continue
            n_compatible += 1
            S = np.asarray(S_int, dtype=float)
            ref_frac, ref_z, ref_orbit = _reference_cell(
                self.parent_positions, self.parent_numbers, self.parent_orbits, S
            )
            child_cell = child_positions @ S
            polar = None
            if free.shape[1]:
                Q, _ = np.linalg.qr(self.L_parent.T @ free)
                polar = Q @ Q.T  # Cartesian projector onto the free directions
            best_here = None
            for base in origins:
                operations = [(W, w0 + (identity - W) @ base)
                              for _, W, w0 in conjugated]
                ref_perm = _operation_permutations(ref_frac, ref_z, operations,
                                                   S, 1e-6)
                child_perm = _operation_permutations(child_cell + base, child_z,
                                                     operations, S, 1e-4)
                if ref_perm is None or child_perm is None:
                    n_unpaired += 1
                    continue
                conflict = _wyckoff_conflict(ref_z, child_z, ref_perm, child_perm)
                if conflict is not None:
                    k, z, n_ref, n_child = conflict
                    conflicts.append((rotations[k], z, n_ref, n_child))
                    continue
                best = _consistent_pairing(
                    ref_frac, ref_z, child_cell, child_z, S, self.L_parent,
                    base, polar, rotations, ref_perm, child_perm,
                )
                if best is None:
                    n_unpaired += 1
                    continue
                if best_here is None or best[0] < best_here[0] - 1e-12:
                    best_here = best
            if best_here is not None:
                total, u, origin = best_here
                # integer components exactly integer (printed modulo 1)
                origin = np.where(np.abs(origin - np.rint(origin)) < 1e-9,
                                  np.rint(origin), origin)
                found.append((
                    total,
                    MappingResult(S_int, origin, ref_frac, list(ref_orbit),
                                  ref_frac + u, [int(z) for z in ref_z], u),
                    conjugated,
                ))
        if not found:
            raise SystemExit(self._mapping_failure(
                reason, n_embedding, n_compatible, n_unpaired, conflicts
            ))

        mapping, rotation, n_tied = _select_setting(
            [(total, candidate) for total, candidate, _ in found],
            self.L_parent_input, self.L_child_input,
        )
        conjugated = next(c for _, candidate, c in found if candidate is mapping)
        # the child operations as (i, t) members on the T_H cell
        members = self._child_operation_members(conjugated, mapping)
        if members is None:
            raise SystemExit(
                "ERROR: broken subgroup-member bookkeeping in the "
                "group-subgroup consistent mapping (internal bug); please "
                "report this case."
            )

        previous = self.mapping
        self.mapping = mapping
        self.setting_rotation = rotation
        self.equivalent_settings = n_tied
        self.S_core = _invariant_core(mapping.S, algebra.rotations)
        self.core_size = abs(int(round(np.linalg.det(self.S_core))))
        self._build_core_cell()

        # parent operations beyond the child's own that leave the mapped
        # structure and the displacement field exactly invariant: the
        # strain-only part of the symmetry lowering (cf. the legacy search)
        mismatches = self._operation_mismatches()
        worst = {(i, tuple(int(v) for v in t)): value for (i, t), value in mismatches}
        listed = {(i, tuple(int(v) for v in t)) for i, t in members}
        for (i, t), value in mismatches:
            if value >= _EXACT_MISMATCH:
                break
            if (i, tuple(int(v) for v in t)) in listed:
                continue
            if self._member_field_violation([(i, t)]) <= _FIELD_INVARIANT:
                members.append((i, t))
        # the child operations are exact by construction (exact origin,
        # symmetric pairing): an internal-consistency check, independent of
        # --tolerance
        mismatch = max(worst[(i, tuple(int(v) for v in t))] for i, t in members)
        field = self._member_field_violation(members)
        if mismatch > _FIELD_INVARIANT or field > _FIELD_INVARIANT:
            raise SystemExit(
                "ERROR: the group-subgroup consistent atom mapping leaves a "
                f"symmetry violation of {max(mismatch, field):.2e} A (structure "
                f"{mismatch:.2e} A, displacement field {field:.2e} A) "
                "(internal inconsistency); please report this case."
            )
        if len(members) > expected:
            self.strain_only_operations = len(members) - expected

        u_old = float(np.max(np.linalg.norm(previous.u_frac @ self.L_parent, axis=1)))
        u_new = float(np.max(np.linalg.norm(mapping.u_frac @ self.L_parent, axis=1)))
        self.mapping_note = (
            "NOTE: the minimum-distortion atom mapping is not a group-subgroup "
            f"setting ({reason}); the atoms were re-paired in a setting in "
            "which every operation of the child space group "
            f"{self.child_symbol} is a parent operation (maximum displacement "
            f"{u_old:.4f} A -> {u_new:.4f} A)."
        )
        return members

    def _mapping_failure(self, reason, n_embedding, n_compatible, n_unpaired,
                         conflicts):
        """Error message when no group-subgroup consistent mapping exists."""
        child = f"{self.child_symbol} (No. {self.child_number})"
        parent = f"{self.parent_symbol} (No. {self.parent_number})"
        settings = (
            f"{len(self._sublattice_bases)} sublattice setting(s) of the "
            f"{self.size}-fold primitive cell within 20% principal strain"
        )
        if n_embedding == 0:
            message = (
                f"ERROR: the child space group {child} is not a subgroup of "
                f"the parent {parent} in any of the {settings}: no setting "
                "carries every point operation of the child onto a parent "
                "operation.  Check that both files describe the same "
                "structure type and the cell multiplication (the space groups "
                "are those spglib finds at --tolerance "
                f"{self.tolerance} A)."
            )
        elif n_compatible == 0:
            message = (
                f"ERROR: the child space group {child} is not a subgroup of "
                f"the parent {parent}: its point group embeds in "
                f"{n_embedding} of the {settings}, but in none of them does "
                "an origin shift turn the translation parts of the child "
                "operations (screw axes, glide planes, centring) into those "
                "of parent operations, so the parent is not a supergroup of "
                "the child.  Check the parent structure."
            )
        elif conflicts and n_unpaired == 0:
            W, z, n_ref, n_child = conflicts[0]
            element = _element_symbol(z)
            message = (
                "ERROR: the child is not a displacive distortion of this "
                f"parent: in each of the {n_compatible} sublattice setting(s) "
                f"in which {child} is a subgroup of {parent}, the Wyckoff "
                "splitting disagrees -- e.g. a parent "
                f"{_point_operation_name(W)} kept by the child fixes "
                f"{n_ref} {element} site(s) of the reference structure but "
                f"{n_child} {element} atom(s) of the child, so no atom "
                "pairing can respect the child symmetry (a different "
                "structure type or polymorph, or a wrong parent)."
            )
        else:
            message = (
                f"ERROR: the child space group {child} is a subgroup of the "
                f"parent {parent} in {n_compatible} sublattice setting(s), "
                "but no atom pairing consistent with it keeps every atom "
                f"within {_MAX_PAIR_DISTANCE} A of its parent site (at every "
                "origin that keeps the child operations parent operations): "
                "a different polymorph or stacking, a large rigid shift, or "
                "a wrong parent."
            )
        return (f"{message}\n(The minimum-distortion atom mapping is not a "
                f"group-subgroup setting: {reason}.)")

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

    def _star_to_core(self, arms, block_vector, add_conjugate=False):
        """Real-space (core cell) displacement field of a star-block vector.

        With ``add_conjugate`` the vector is the D part of a D + D* pair on
        a star without its -k arms; the D* part lives on the -k star and is
        the complex conjugate field, so the real field is twice the real
        part."""
        n_t = len(self.reps_core)
        t_matrix = np.asarray(self.reps_core, dtype=float)
        phases = np.exp(
            2j * np.pi * (np.asarray(arms) @ t_matrix.T) / DEN
        ) / np.sqrt(n_t)  # (m, n_t)
        blocks = block_vector.reshape(len(arms), self.n_parent, 3)
        field = np.einsum("as,apm->psm", phases, blocks)
        if add_conjugate:
            field = 2.0 * field.real + 0j
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
        # complex-type irreps enter as their physically irreducible pair
        # D + D* (one line, label P1P2 / H1HA1); the conjugate characters
        # of every pair listed so far identify the partner when it comes
        # up -- on the same star, or on the -k star of a star without its
        # -k arms -- independently of the labels
        listed_pairs = []
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
                # dimension of the complex induced irrep whose matrices are
                # `blocks` (n_arms x dim_small); `dimension` is twice that
                # for a doubled (complex- or pseudoreal-type) irrep
                d_tau = representation.blocks[0].shape[0]
                # complex isotypic projector P_tau = (d/n_ops) sum_i
                # delta_a(i)* [row-arm-a blocks of D(i)]
                P_c = np.zeros_like(P_H)
                characters = np.zeros((algebra.n_ops, len(arms)),
                                      dtype=np.complex128)
                for i in range(algebra.n_ops):
                    diag = np.diagonal(representation.blocks[i])
                    characters[i] = np.add.reduceat(
                        diag, np.arange(0, d_tau, d_small)
                    )
                    delta = np.conj(characters[i])
                    P_c += np.repeat(delta, np3)[:, None] * dhats[i]
                P_c *= d_tau / algebra.n_ops
                completeness += P_c
                pair = (representation.doubled
                        and getattr(representation, "fs_type", "") == "complex")
                if pair:
                    # chi(g_i + t) = sum_a characters[i, a] e^{SIGMA 2 pi i
                    # q_a.t}: the partner D* has the conjugate entries on
                    # the arms -q_a
                    own = {
                        tuple(int(v) % DEN for v in arm): characters[:, a]
                        for a, arm in enumerate(arms)
                    }
                    found = None
                    for record in listed_pairs:
                        conjugate = record["conjugate"]
                        if set(conjugate) == set(own) and all(
                            np.allclose(own[key], conjugate[key], atol=1e-6)
                            for key in own
                        ):
                            found = record
                            break
                    if found is not None:
                        # already listed through its partner; the label is
                        # the pair the characters identify (it replaces the
                        # one from conjugate_partner(), which it normally
                        # equals)
                        entry = found["entry"]
                        if entry is not None:
                            entry.irrep_name = "".join(
                                sorted([found["name"], irrep_name])
                            )
                        continue
                    record = {
                        "name": irrep_name,
                        "entry": None,
                        "conjugate": {
                            tuple(int(v) % DEN for v in -np.asarray(arm)):
                                np.conj(characters[:, a])
                            for a, arm in enumerate(arms)
                        },
                    }
                    listed_pairs.append(record)
                # the real projector of the old dense route: Re chi* over a
                # real displacement space = (P_tau + conj(P_tau)) / 2, with
                # conj(P_tau) living on the -k arms; for a complex-type
                # irrep the physically irreducible D + D* takes the sum
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
                    P = (P_c + swapped) if pair else 0.5 * (P_c + swapped)
                    dim = int(round(float(np.trace(P @ P_H).real)))
                    if dim <= 0:
                        continue
                    projected_hat = P @ uhat
                    projected = self._star_to_core(arms, projected_hat)
                    amplitude = float(np.linalg.norm(projected_hat)) * rescale
                else:
                    # a star without its -k arms carries complex-type irreps
                    # only; D* (and the conjugate Fourier components of the
                    # real field) live on the -k star
                    if not pair:
                        raise SystemExit(
                            "ERROR: real-type irrep on a star without its -k "
                            "arms (internal bug)."
                        )
                    dim = int(round(2.0 * float(np.trace(P_c @ P_H).real)))
                    if dim <= 0:
                        continue
                    projected = self._star_to_core(
                        arms, P_c @ uhat, add_conjugate=True
                    )
                    amplitude = float(np.linalg.norm(projected)) * rescale
                residual = residual - projected.reshape(-1)
                label = irrep_name
                if pair:
                    partner = representation.conjugate_partner()
                    if partner is not None:
                        label = "".join(sorted([irrep_name, partner]))
                entry = _ModeEntry(self, star["kname"], star["kvec"], label,
                                   representation, dim, amplitude, projected)
                if pair:
                    record["entry"] = entry
                modes.append(entry)
            if not np.allclose(
                completeness, np.eye(completeness.shape[0]), atol=1e-6
            ):
                raise SystemExit(
                    f"ERROR: mode-projector completeness check failed at the "
                    f"{star['kname']} star; please report this case."
                )
        if np.linalg.norm(residual) > 1e-3 * max(1.0, total):
            # the mapping stage guarantees a displacement field invariant
            # under the subgroup, which the modes listed (dim > 0) capture
            # completely; a residual here is an internal inconsistency
            raise SystemExit(
                "ERROR: the distortion is not fully captured by the listed "
                "modes (internal inconsistency of the mode projectors); "
                "please report this case."
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
        # every representation shares one arm ordering.  A star whose ISO-IR
        # matrices cannot be realified (SystemExit from _realify) also falls
        # back on the spgrep basis instead of stopping the analysis.
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
            except (LookupError, FileNotFoundError, ValueError, SystemExit):
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
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output_dir is not None and os.path.exists(args.output_dir) \
            and not os.path.isdir(args.output_dir):
        parser.error(f"--output-dir {args.output_dir}: exists and is not a directory.")
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
    if getattr(analysis, "mapping_note", None):
        print(textwrap.fill(analysis.mapping_note, width=79))
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
    print("\n* Distortion amplitude *")
    print(f"maximum atomic displacement: {max_u:.4f} A")
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
    rows = []
    for mode in analysis.modes:
        label, info, index = mode.label_info()
        rows.append((mode, label, f"{info.number} {info.international_short}"))
    # the pair labels of complex-type irreps (GM3+GM4+, K2KA2) and their
    # directions are longer than the default columns
    name_width = max([7] + [len(mode.irrep_name) for mode, _, _ in rows])
    label_width = max([12] + [len(label) for _, label, _ in rows])
    table_lines.append(
        f"{'k-vector':<16} {'irrep':<{name_width}} {'direction':<{label_width}} "
        f"{'isotropy subgroup':<19} {'dim':<4} amplitude (A)"
    )
    for mode, label, subgroup in rows:
        table_lines.append(
            f"{_kvector_string(mode.kvec):<16} "
            f"{mode.irrep_name:<{name_width}} {label:<{label_width}} {subgroup:<19} "
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
    output_lines: list[str] = []
    if not args.no_files:
        table_path = _output_path(args.output_dir, f"sym_mode_{analysis.parent_formula}")
        with open(table_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(table_lines) + "\n")
        output_lines.append(f"Decomposition table saved to {table_path}")

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

    if not args.no_files:
        output_lines.extend(
            _export_mode_vesta_files(analysis, args.parent, conventional=args.conventional,
                                     output_dir=args.output_dir)
        )

    print("\nConventions and validation: AMPLIMODES (Bilbao Crystallographic "
          "Server):")
    print('D. Orobengoa, C. Capillas, M. I. Aroyo and J. M. Perez-Mato,')
    print('"AMPLIMODES: symmetry-mode analysis on the Bilbao Crystallographic')
    print('Server", J. Appl. Cryst. 42, 820-833 (2009).')
    if output_lines:
        print("\n* Output files *")
        for line in output_lines:
            print(f"  {line}")
    print()


def _output_path(output_dir: str | None, filename: str) -> str:
    """Where an output file goes: the bare name in the current directory
    (the default), or inside --output-dir, which is created if missing."""
    if output_dir is None:
        return filename
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, filename)


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
                             conventional: bool = False,
                             output_dir: str | None = None) -> list[str]:
    """Write one VESTA file per activated irrep showing its displacement
    pattern: arrows of the irrep-projected distortion on the parent-derived
    reference structure, in the invariant-core cell (default) or in the
    parent conventional basis (--conventional, _conv suffix), into the
    current directory or ``output_dir`` (created if missing).

    Returns:
        The lines describing the written files for the ``* Output files *``
        block (empty when no irrep is activated).
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
        filename = _output_path(output_dir, f"{parent_base}_{mode.irrep_name}{suffix}.vesta")
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
        return []
    lines = [f"Mode displacement VESTA files ({cell_note}):"]
    if conventional:
        lines.append("  display cell in parent primitive units (rows):")
        for row in D:
            lines.append("    (" + ", ".join(str(int(v)) for v in row) + ")")
    for filename, amplitude in written:
        lines.append(f"  {filename}  (amplitude {amplitude:.4f} A)")
    lines.append(f"Arrows are scaled so the largest displacement is "
                 f"{_VESTA_ARROW_LENGTH} A per file; adjust in VESTA via "
                 "Edit > Vectors if needed.")
    return lines


if __name__ == "__main__":
    main()
