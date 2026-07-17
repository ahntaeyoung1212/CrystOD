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
the CDML (irreptables) primitive setting of crystod's space-group irrep
machinery (an origin-shift fit absorbs any difference between the spglib and
CDML origin conventions, verified by invariance of the structure under every
tabulated operation).  The subgroup translation lattice and the origin shift
are found by strain-tolerant lattice matching plus atom pairing (the
assignment minimizing the total distortion is chosen), the subgroup elements
are identified as the parent operations that leave the distorted structure
invariant, and the displacement field is projected onto every parent irrep
at the k points folding to the subgroup Gamma point with the full induced
irrep matrices (the same machinery as --supergroup).  Amplitudes follow the
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
    InducedRepresentation,
    IsotropyAnalyzer,
    _nullspace,
    _projector,
)
from .spacegroup_product import DEN, SpaceGroupIrrepAlgebra


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
    return conventional, primitive, dataset


def _dataset_field(dataset, name):
    if isinstance(dataset, dict):
        return dataset.get(name)
    return getattr(dataset, name, None)


def _fit_origin_shift(algebra, spg_rotations, spg_translations):
    """delta with x_cdml = x_spg + delta: (I - W) delta = v_cdml - v_spg,
    modulo the full conventional translation lattice (centring included).

    Matches the CDML conventional operations of the algebra against the
    spglib conventional operations of the standardized structure.  For
    centred lattices the spglib conventional dataset carries every centring
    copy of each operation, so the translation comparison must be made
    modulo the centring translations."""
    cdml = {}
    for sym in algebra.table.symmetries:
        cdml[np.rint(np.asarray(sym.R, dtype=float)).astype(np.int64).tobytes()] = (
            np.asarray(sym.t, dtype=float) % 1.0
        )
    # centring translations in conventional fractional units
    M = algebra.primitive_matrix
    centring = []
    seen = set()
    for z in product(range(3), repeat=3):
        vector = (np.array(z, dtype=float) @ M) % 1.0
        key = tuple(np.round(vector, 6))
        if key not in seen:
            seen.add(key)
            centring.append(np.array(key))
    pairs = []
    for W, v in zip(spg_rotations, spg_translations):
        key = np.rint(W).astype(np.int64).tobytes()
        if key not in cdml:
            raise SystemExit(
                "ERROR: the parent operations do not match the CDML setting "
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
                "ERROR: could not fit the CDML origin shift; please report "
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
        A_inv = np.linalg.inv(A)
        prim_positions = (shifted @ A_inv.T) % 1.0  # column: x_p = A^-1 x_c
        lattice_prim = A @ np.asarray(lattice_conv)
        merged, merged_z = _merge_duplicates(prim_positions, numbers_conv)
        if _invariant_under_algebra(algebra, merged, merged_z):
            return lattice_prim, merged, merged_z, A
    raise SystemExit(
        "ERROR: could not express the parent structure in the CDML primitive "
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


# ---------------------------------------------------------------------------
# the analysis
# ---------------------------------------------------------------------------


class SymmetryModeAnalysis:
    def __init__(self, parent_file: str, child_file: str, tolerance: float):
        parent_conv, parent_prim, parent_ds = _load_standardized(
            parent_file, tolerance
        )
        child_conv, child_prim, child_ds = _load_standardized(child_file, tolerance)
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

        # parent orbits (element + Wyckoff) in the primitive setting
        self.parent_orbits = self._parent_orbits()

        # child primitive structure, fractional in its own basis
        L_child, child_positions, child_numbers = child_prim
        if len(child_numbers) % len(self.parent_numbers) != 0:
            raise SystemExit(
                f"ERROR: the child primitive cell ({len(child_numbers)} atoms) "
                "is not an integer multiple of the parent primitive cell "
                f"({len(self.parent_numbers)} atoms)."
            )
        self.size = len(child_numbers) // len(self.parent_numbers)
        candidates = _sublattice_candidates(self.L_parent, L_child, self.size)
        best = None
        for S in candidates:
            result = _match_atoms(
                self.parent_positions, self.parent_numbers, self.parent_orbits,
                child_positions, child_numbers, S, self.L_parent,
            )
            if result is not None and (best is None or result[0] < best[0]):
                best = result
        if best is None:
            raise SystemExit(
                "ERROR: could not map the child structure onto the parent "
                "(is it really a distorted version of the parent structure?)."
            )
        self.mapping = best[1]

        # analysis cell: the largest G-invariant sublattice of T_H, so that
        # the displacement space carries a full representation of the parent
        # group (complete stars); amplitudes are rescaled back to the
        # primitive cell of the distorted structure at the end
        self.S_core = _invariant_core(self.mapping.S, self.algebra.rotations)
        self.core_size = abs(int(round(np.linalg.det(self.S_core))))
        self._build_core_cell()

        self.subgroup_members = self._find_subgroup_members()
        self._remove_acoustic_offset()
        self.k_folding = self._folding_kpoints()
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
        if free.shape[1] == 0:
            return
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

    # -- subgroup elements: parent operations preserving the child structure
    def _find_subgroup_members(self):
        algebra = self.algebra
        S = self.S_core
        S_inv = np.linalg.inv(S)
        reps = _translation_reps(S)
        positions = self.core_child_frac
        numbers = self.ref_z
        members = []
        for i in range(algebra.n_ops):
            W = algebra.rotations[i]
            v = np.array(algebra.translations[i], dtype=float) / DEN
            for t in reps:
                good = True
                for x, z in zip(positions, numbers):
                    image = W @ x + v + t
                    hit = False
                    for y, zz in zip(positions, numbers):
                        d = image - y
                        d = d - np.rint(d @ S_inv) @ S
                        if zz == z and np.linalg.norm(d @ self.L_parent) < 0.05:
                            hit = True
                            break
                    if not hit:
                        good = False
                        break
                if good:
                    members.append((i, np.asarray(t, dtype=np.int64)))
        return members

    # -- k points folding to the analysis-cell Gamma point
    def _folding_kpoints(self):
        algebra = self.algebra
        S = self.S_core
        n = self.core_size
        S_inv_T = np.linalg.inv(S).T
        points = set()
        bound = int(np.max(np.abs(S))) + 1
        for z in product(range(-bound, bound + 1), repeat=3):
            k = (np.array(z, dtype=float) @ S_inv_T) % 1.0
            points.add(tuple(np.round(k, 6) % 1.0))
            if len(points) > 4 * n:
                break
        # match every folding point to the star that contains it (any arm)
        matched: dict[tuple, str] = {}
        for kname in algebra.k_by_kname:
            arms, _ = algebra.star(kname)
            for arm in arms:
                key = tuple(np.round((np.array(arm, dtype=float) / DEN) % 1.0, 6))
                if key in points and key not in matched:
                    matched[key] = kname
        missing = [p for p in points if p not in matched]
        if missing:
            raise SystemExit(
                "ERROR: some folding k-points are not tabulated special "
                f"points of {self.parent_symbol}: {sorted(missing)}; this "
                "group-subgroup index is not supported yet."
            )
        # one entry per distinct star
        stars: dict[str, tuple] = {}
        for key, kname in sorted(matched.items()):
            stars.setdefault(kname, key)
        return {key: kname for kname, key in stars.items()}

    # -- displacement representation of one parent element on the core cell
    def _displacement_matrix(self, i, t):
        algebra = self.algebra
        S_inv = np.linalg.inv(self.S_core)
        W = algebra.rotations[i]
        v = np.array(algebra.translations[i], dtype=float) / DEN
        # Cartesian rotation: r = L^T x (columns) -> R = L^T W (L^T)^-1
        LT = self.L_parent.T
        R = LT @ W @ np.linalg.inv(LT)
        n = self.n_atoms
        matrix = np.zeros((3 * n, 3 * n))
        ref = self.ref_frac
        for j in range(n):
            image = W @ ref[j] + v + t
            target = None
            for m in range(n):
                d = image - ref[m]
                d = d - np.rint(d @ S_inv) @ self.S_core
                if (np.linalg.norm(d @ self.L_parent) < 1e-3
                        and self.ref_z[j] == self.ref_z[m]):
                    target = m
                    break
            if target is None:
                raise SystemExit("ERROR: broken displacement-representation "
                                 "bookkeeping.")
            matrix[3 * target:3 * target + 3, 3 * j:3 * j + 3] = R
        return matrix

    # -- the mode decomposition
    def _decompose(self):
        algebra = self.algebra
        reps = _translation_reps(self.S_core)
        n_F = algebra.n_ops * len(reps)
        # amplitudes: AMPLIMODES normalizes within the primitive cell of the
        # distorted structure (T_H); the core cell repeats it core/size times
        rescale = np.sqrt(self.size / self.core_size)

        # displacement matrices of the whole factor group (cached)
        disp = {}
        for i in range(algebra.n_ops):
            for t in reps:
                disp[(i, tuple(t))] = self._displacement_matrix(i, t)

        # subgroup projector (H-invariant displacements)
        P_H = np.zeros((3 * self.n_atoms, 3 * self.n_atoms))
        for i, t in self.subgroup_members:
            P_H += disp[(i, tuple(t))]
        P_H /= len(self.subgroup_members)

        u = self.u_cart.reshape(-1)
        total = np.linalg.norm(u)

        modes = []
        completeness = np.zeros_like(P_H)
        for k_tuple, kname in sorted(self.k_folding.items()):
            for irrep in self.algebra.irreps_by_kname[kname]:
                try:
                    representation = InducedRepresentation(algebra, irrep.name)
                except SystemExit:
                    raise
                # character of (i, t): trace of T(t) B_i
                d_tau = representation.dimension
                P = np.zeros_like(P_H)
                for i in range(algebra.n_ops):
                    diag = np.diagonal(representation.blocks[i])
                    for t in reps:
                        chi = np.sum(
                            representation.translation_phases(t) * diag
                        )
                        P += np.real(np.conj(chi)) * disp[(i, tuple(t))]
                P *= d_tau / n_F
                completeness += P
                dim = int(round(np.trace(P @ P_H)))
                if dim <= 0:
                    continue
                amplitude = float(np.linalg.norm(P @ u)) * rescale
                modes.append(
                    _ModeEntry(self, kname, irrep.name, representation, P, P_H,
                               dim, amplitude)
                )
        if not np.allclose(completeness, np.eye(3 * self.n_atoms), atol=1e-4):
            raise SystemExit(
                "ERROR: mode-projector completeness check failed; please "
                "report this case."
            )
        residual = u.copy()
        for mode in modes:
            residual = residual - mode.projector @ u
        if np.linalg.norm(residual) > 1e-3 * max(1.0, total):
            raise SystemExit(
                "ERROR: the distortion is not fully captured by the listed "
                "modes; please report this case."
            )
        self.total_distortion = total * rescale
        return modes


class _ModeEntry:
    def __init__(self, analysis, kname, irrep_name, representation, projector,
                 P_H, dim, amplitude):
        self.analysis = analysis
        self.kname = kname
        self.irrep_name = irrep_name
        self.representation = representation
        self.projector = projector
        self.dim = dim
        self.amplitude = amplitude
        self._label_info = None

    def label_info(self):
        """(direction label, subgroup info, index) via the isotropy machinery."""
        if self._label_info is not None:
            return self._label_info
        analyzer = IsotropyAnalyzer.__new__(IsotropyAnalyzer)
        analyzer.algebra = self.analysis.algebra
        analyzer.representation = self.representation
        analyzer.elements = self.representation.image_elements()

        # H members reduced to the representation's translation grid
        grid = {tuple(t) for _, t, _ in analyzer.elements}
        N = max(max(t) for t in grid) + 1 if grid else 1
        members = []
        for i, t in self.analysis.subgroup_members:
            members.append((i, np.asarray(t, dtype=np.int64) % N))
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


def _kvector_string(algebra, kname) -> str:
    k = np.array(algebra.k_by_kname[kname], dtype=float) / DEN
    return "(" + ",".join(_format_fraction(v % 1.0) for v in k) + ")"


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

    mapping = analysis.mapping
    print("\n* Cell relation *")
    print("child primitive basis in parent primitive units (rows):")
    for row in mapping.S:
        print("  (" + ", ".join(str(int(v)) for v in row) + ")")
    print(f"origin shift (parent primitive fractional): ("
          + ", ".join(_format_fraction(v % 1.0) for v in mapping.p) + ")")
    print(f"primitive cell multiplication: {analysis.size}")

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

    print("\n* Symmetry-mode decomposition *")
    header = (f"{'k-vector':<16} {'irrep':<7} {'direction':<12} "
              f"{'isotropy subgroup':<19} {'dim':<4} amplitude (A)")
    print(header)
    for mode in analysis.modes:
        label, info, index = mode.label_info()
        subgroup = f"{info.number} {info.international_short}"
        print(
            f"{_kvector_string(analysis.algebra, mode.kname):<16} "
            f"{mode.irrep_name:<7} {label:<12} {subgroup:<19} "
            f"{mode.dim:<4} {mode.amplitude:.4f}"
        )

    print("\n* Normalized mode components (parent primitive fractional, per 1 A) *")
    for mode in analysis.modes:
        if mode.amplitude < 1e-4:
            print(f"{mode.irrep_name}: amplitude 0 (allowed but not activated)")
            continue
        direction = (mode.projector @ analysis.u_cart.reshape(-1))
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

    print("\nConventions and validation: AMPLIMODES (Bilbao Crystallographic "
          "Server):")
    print('D. Orobengoa, C. Capillas, M. I. Aroyo and J. M. Perez-Mato,')
    print('"AMPLIMODES: symmetry-mode analysis on the Bilbao Crystallographic')
    print('Server", J. Appl. Cryst. 42, 820-833 (2009).')
    print()


if __name__ == "__main__":
    main()
