"""Isotropy subgroups of space-group irreps (crystod-group --supergroup).

Given a space group G and one of its irreps (CDML label), a distortion that
transforms as that irrep reduces the symmetry to the isotropy subgroup

    H(eta) = { g in G : D(g) eta = eta }

where D is the full (induced) representation and eta the order parameter.
Distinct order-parameter directions (a,0,0), (a,a,0), ... give distinct
isotropy subgroups; this module enumerates all of them (the strata of the
representation), or resolves a user-given direction, and identifies each
subgroup with spglib (symbol, number, cell size, index, and the basis /
origin of its conventional cell in the parent convention).

This is the offline counterpart of the ISOSUBGROUP tool of the ISOTROPY
Software Suite (https://iso.byu.edu), and is validated against its output.
If you use this feature, please cite: H. T. Stokes, S. van Orden and
B. J. Campbell, "Tool for Generating Isotropy Subgroups of Crystallographic
Space Groups", J. Appl. Cryst. 49, 1849-1853 (2016).

The order-parameter components refer to the real irrep basis produced by
spgrep; for multi-arm stars the components are grouped arm by arm. The basis
may differ from ISOTROPY's by an orthogonal change (direction labels can be
permuted/rotated relative to the ISOSUBGROUP listing), but the resulting
subgroups are convention-independent.
"""

from __future__ import annotations

import argparse
from fractions import Fraction

import numpy as np

from .spacegroup_product import DEN, SIGMA, SpaceGroupIrrepAlgebra

_PARAMETER_NAMES = "abcdefgh"


# --------------------------------------------------------------- representation


class InducedRepresentation:
    """Full (induced) irrep matrices of a space group, as explicit blocks.

    Basis index = (arm a, small-irrep row p). Elements are parametrized as
    (coset representative i, lattice translation t):

        D(g_i + t) = T(t) B_i,   T(t) = diag_a exp(SIGMA*2j*pi q_a.t) (x) 1_d
    """

    def __init__(self, algebra: SpaceGroupIrrepAlgebra, irrep_label: str):
        self.algebra = algebra
        self.irrep = algebra.find_irrep(irrep_label)
        kpname = self.irrep.kpname
        if kpname not in algebra.k_by_kname:
            raise SystemExit(f"ERROR: unknown k point for irrep {irrep_label}.")
        self.k = algebra.k_by_kname[kpname]
        self.arms, self.representatives = algebra.star(kpname)
        self.n_arms = len(self.arms)

        small_matrices, little = self._small_matrices()
        self.little = little
        self.dim_small = small_matrices[next(iter(small_matrices))].shape[0]
        self.dimension = self.n_arms * self.dim_small

        self.blocks = self._induce(small_matrices)
        self._verify()
        # materialize all distinct elements (coset rep i, lattice translation t)
        self.elements = [
            (i, t, self.translation_phases(t)[:, None] * self.blocks[i])
            for i in range(algebra.n_ops)
            for t in self._translation_grid()
        ]
        self._realify()

    # -- small irrep matrices matched to the CDML label
    def _small_matrices(self):
        algebra, irrep = self.algebra, self.irrep
        from spgrep.core import get_spacegroup_irreps_from_primitive_symmetry

        try:
            irreps, mapping = get_spacegroup_irreps_from_primitive_symmetry(
                rotations=algebra.rotations,
                translations=np.array(algebra.translations, dtype=float) / DEN,
                kpoint=np.array(self.k, dtype=float) / DEN,
            )
        except Exception as exc:
            raise SystemExit(
                f"ERROR: spgrep could not compute the small irreps at "
                f"{irrep.kpname}: {exc}"
            ) from exc
        mapping = [int(m) for m in np.asarray(mapping).ravel()]

        # reference characters of the requested irrep (spgrep-refined)
        table = {int(key) - 1: complex(v) for key, v in irrep.characters.items()}
        refined = algebra._refine_small_characters(np.asarray(self.k), table)
        if refined is None:
            raise SystemExit(
                f"ERROR: the tabulated characters of {irrep.name} are not those "
                "of a single allowed small irrep; isotropy-subgroup analysis is "
                "not available for this entry."
            )
        matches = []
        for matrices in irreps:
            matrices = np.asarray(matrices)
            chi = {op: complex(np.trace(matrices[j])) for j, op in enumerate(mapping)}
            if set(chi) == set(refined) and all(
                abs(chi[op] - refined[op]) < 1e-3 for op in refined
            ):
                matches.append(matrices)
        if len(matches) != 1:
            raise SystemExit(
                f"ERROR: could not match {irrep.name} to a unique computed small "
                f"irrep ({len(matches)} candidates); possibly a paired (physically "
                "combined) irrep, which is not supported yet."
            )
        matrices = matches[0]
        small = {op: np.asarray(matrices[j]) for j, op in enumerate(mapping)}
        # when 2k = 0 (mod reciprocal lattice) the translation phases are
        # real, so a real small irrep gives a real induced rep in the natural
        # arm-blocked basis (nice, arm-grouped order-parameter components)
        if np.all((2 * np.asarray(self.k)) % DEN == 0):
            small = _realify_matrix_set(small) or small
        return small, sorted(mapping)

    def _induce(self, small: dict) -> list[np.ndarray]:
        algebra = self.algebra
        d, m = self.dim_small, self.n_arms
        arm_index = {tuple(arm): a for a, arm in enumerate(self.arms)}
        blocks = []
        for i in range(algebra.n_ops):
            matrix = np.zeros((m * d, m * d), dtype=np.complex128)
            for b, s_b in enumerate(self.representatives):
                # row arm: q_a = q_b . W_i^{-1}
                q_a = tuple((self.arms[b] @ algebra.inverse_rotations[i]) % DEN)
                a = arm_index[q_a]
                s_a = self.representatives[a]
                W_sa_inv = algebra.inverse_rotations[s_a]
                v_sa_inv = -W_sa_inv @ algebra.translations[s_a]
                # h = s_a^{-1} (g_i s_b)
                W_gs = algebra.rotations[i] @ algebra.rotations[s_b]
                v_gs = algebra.rotations[i] @ algebra.translations[s_b] + algebra.translations[i]
                W_h = W_sa_inv @ W_gs
                tau_h = W_sa_inv @ v_gs + v_sa_inv
                mindex = algebra._rotation_index[algebra._key(W_h)]
                if mindex not in small:
                    raise SystemExit("ERROR: broken induction bookkeeping.")
                t_extra = tau_h - algebra.translations[mindex]
                if np.any(t_extra % DEN != 0):
                    raise SystemExit("ERROR: non-lattice residue in induction.")
                phase = np.exp(
                    SIGMA * 2j * np.pi * float(self.k @ (t_extra // DEN)) / DEN
                )
                matrix[a * d : (a + 1) * d, b * d : (b + 1) * d] = phase * small[mindex]
            blocks.append(matrix)
        return blocks

    def translation_phases(self, t: np.ndarray) -> np.ndarray:
        """Diagonal of T(t) (t in integer lattice units)."""
        phases = np.exp(
            SIGMA * 2j * np.pi * (self.arms @ np.asarray(t, dtype=np.int64)) / DEN
        )
        return np.repeat(phases, self.dim_small)

    def _realify(self) -> None:
        """Transform to a real matrix form (possible when all characters are
        real, i.e. real-type irreps; the practically relevant case)."""
        if all(np.allclose(matrix.imag, 0, atol=1e-8) for _, _, matrix in self.elements):
            self.elements = [
                (i, t, matrix.real.copy()) for i, t, matrix in self.elements
            ]
            return
        # real (orthogonal) form via the antilinear real structure J v = S v*:
        # S = sum_g D(g)* A D(g)^dagger intertwines D -> D*; for a real-type
        # irrep S S* = c > 0 and J^2 = 1 after normalization, and the fixed
        # points of J span a real basis in which every D(g) is real.
        rng = np.random.default_rng(7)
        n = self.dimension
        A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        S = np.zeros((n, n), dtype=np.complex128)
        for _, _, D in self.elements:
            S += np.conj(D) @ A @ D.conj().T
        c_matrix = S @ np.conj(S)
        c = c_matrix[0, 0]
        if not np.allclose(c_matrix, c * np.eye(n), atol=1e-6 * max(1, abs(c))) or c.real <= 0:
            raise SystemExit(
                f"ERROR: {self.irrep.name} is a complex- or pseudoreal-type "
                "irrep; the physically irreducible (doubled) real form is not "
                "supported yet."
            )
        S = S / np.sqrt(c.real)
        # real basis: orthonormalize J-fixed vectors v + S v*
        basis: list[np.ndarray] = []
        trial = 0
        while len(basis) < n and trial < 20 * n:
            trial += 1
            v = rng.normal(size=n) + 1j * rng.normal(size=n)
            w = v + S @ np.conj(v)
            for prior in basis:
                w = w - prior * np.real(np.vdot(prior, w))
            norm = np.linalg.norm(w)
            if norm > 1e-6:
                basis.append(w / norm)
        if len(basis) < n:
            raise SystemExit(f"ERROR: could not realify {self.irrep.name}.")
        T = np.column_stack(basis)
        T_inv = np.linalg.inv(T)
        new_elements = []
        for i, t, matrix in self.elements:
            transformed = T_inv @ matrix @ T
            if not np.allclose(transformed.imag, 0, atol=1e-6):
                raise SystemExit(
                    f"ERROR: could not realify {self.irrep.name}; complex-type "
                    "irreps are not supported yet."
                )
            new_elements.append((i, t, transformed.real.copy()))
        self.elements = new_elements

    def _translation_grid(self) -> list[np.ndarray]:
        denominators = [int(DEN // np.gcd(int(v), DEN)) if v else 1 for v in self.k]
        N = max(denominators + [1])
        return [
            np.array([t1, t2, t3], dtype=np.int64)
            for t1 in range(N)
            for t2 in range(N)
            for t3 in range(N)
        ]

    def _verify(self) -> None:
        """Traces must reproduce the validated induced characters."""
        arms, C = self.algebra.induced_characters(self.irrep)
        for i in (0, min(3, self.algebra.n_ops - 1), self.algebra.n_ops - 1):
            expected = np.sum(C[i])
            actual = np.trace(self.blocks[i])
            if abs(expected - actual) > 1e-6:
                raise SystemExit(
                    "ERROR: induced-matrix construction disagrees with the "
                    "validated induced characters (internal bug)."
                )

    # -- group elements of the image, with bookkeeping
    def image_elements(self):
        """All (coset rep i, lattice translation t, real matrix)."""
        return self.elements


# ------------------------------------------------------------------ stabilizers


def _nullspace(matrix: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    _, sing, Vh = np.linalg.svd(matrix)
    rank = int(np.sum(sing > tol)) if len(sing) else 0
    return Vh[rank:].T.conj()


def _projector(basis: np.ndarray) -> np.ndarray:
    if basis.shape[1] == 0:
        return np.zeros((basis.shape[0], basis.shape[0]))
    Q, _ = np.linalg.qr(basis)
    return Q @ Q.T.conj()


class IsotropyAnalyzer:
    def __init__(self, space_group: str, irrep_label: str):
        self.algebra = SpaceGroupIrrepAlgebra(space_group)
        self.representation = InducedRepresentation(self.algebra, irrep_label)
        self.elements = self.representation.image_elements()

    # -- stabilizer of a direction (subspace)
    def stabilizer_of(self, projector: np.ndarray):
        """(i, t) elements acting as the identity on the subspace."""
        return [
            (i, t)
            for i, t, matrix in self.elements
            if np.allclose(matrix @ projector, projector, atol=1e-6)
        ]

    def fixed_space(self, members) -> np.ndarray:
        """Common fixed subspace of the given (i, t) elements (as basis)."""
        n = self.representation.dimension
        stack = []
        member_keys = {(i, tuple(t)) for i, t in members}
        for i, t, matrix in self.elements:
            if (i, tuple(t)) in member_keys:
                stack.append(matrix - np.eye(n))
        if not stack:
            return np.eye(n)
        return _nullspace(np.vstack(stack))

    # -- enumerate strata (order-parameter direction types)
    def enumerate_directions(self):
        n = self.representation.dimension
        seen: dict[bytes, np.ndarray] = {}

        def add(basis: np.ndarray):
            if basis.shape[1] == 0:
                return None
            projector = _projector(basis)
            key = _projector_key(projector)
            if key not in seen:
                seen[key] = projector
                return projector
            return None

        # seed: fixed spaces of every group element, and the full space
        seeds = []
        for _, _, matrix in self.elements:
            basis = _nullspace(matrix - np.eye(n))
            if add(basis) is not None:
                seeds.append(_projector(basis))
        add(np.eye(n))

        # closure under pairwise intersection
        frontier = list(seen.values())
        while frontier:
            new = []
            for P in frontier:
                for Q in list(seen.values()):
                    intersection = _nullspace(
                        np.vstack([P - np.eye(n), Q - np.eye(n)])
                    )
                    result = add(intersection)
                    if result is not None:
                        new.append(result)
            frontier = new

        # keep isotropy subspaces: V == Fix(Stab(V)); dedupe by group orbit
        strata = []
        used = set()
        for key, projector in seen.items():
            if key in used:
                continue
            members = self.stabilizer_of(projector)
            fixed = self.fixed_space(members)
            if not np.allclose(_projector(fixed), projector, atol=1e-6):
                continue
            # orbit dedup; keep the orbit member with the prettiest label
            orbit_keys = set()
            orbit_projectors = []
            for _, _, matrix in self.elements:
                image = np.real_if_close(matrix @ projector @ matrix.T.conj())
                image_key = _projector_key(image)
                if image_key not in orbit_keys:
                    orbit_keys.add(image_key)
                    orbit_projectors.append(image)
            if orbit_keys & used:
                continue
            used |= orbit_keys
            best = min(
                orbit_projectors,
                key=lambda P: _label_rank(self.direction_label(P)[0]),
            )
            strata.append((best, self.stabilizer_of(best)))
        return strata

    # -- subgroup identification
    def subgroup_of(self, members):
        """Space-group type of the isotropy subgroup given its (i, t) members."""
        import spglib
        from sympy import Matrix
        from sympy.matrices.normalforms import hermite_normal_form

        algebra = self.algebra
        # translation lattice: t with T(t) acting as identity on eta happens
        # exactly for members with i == identity
        identity_index = algebra._rotation_index[algebra._key(np.eye(3))]
        pure = [t for i, t in members if i == identity_index and not np.any(
            algebra.translations[identity_index])] or [np.zeros(3, dtype=np.int64)]
        grid_n = max(int(DEN // np.gcd(int(v), DEN)) if v else 1 for v in self.representation.k)
        generators = [t for t in pure] + [grid_n * e for e in np.eye(3, dtype=np.int64)]
        H = np.array(
            hermite_normal_form(Matrix(np.array(generators, dtype=np.int64).T))
        ).astype(np.int64)
        B = H.T  # rows = sublattice basis in parent primitive units
        size = abs(int(round(np.linalg.det(B))))

        # one representative (W, v + t) per coset-rep index
        chosen: dict[int, np.ndarray] = {}
        for i, t in members:
            if i not in chosen:
                chosen[i] = np.asarray(t, dtype=np.int64)
        # ops in the sublattice basis (column-vector convention)
        B_inv_T = np.linalg.inv(B.T)
        rotations, translations = [], []
        for i, t in chosen.items():
            W = B_inv_T @ algebra.rotations[i] @ B.T
            W_int = np.rint(W).astype(np.int64)
            if not np.allclose(W, W_int, atol=1e-8):
                raise SystemExit("ERROR: subgroup operation is incompatible with its lattice.")
            v = B_inv_T @ (np.array(algebra.translations[i], dtype=float) / DEN + t)
            rotations.append(W_int)
            translations.append(np.mod(v, 1.0))

        lattice_parent = self._invariant_lattice()
        lattice = B @ lattice_parent
        try:
            info = spglib.get_spacegroup_type_from_symmetry(
                np.array(rotations), np.array(translations), lattice=lattice, symprec=1e-5
            )
        except Exception as exc:
            raise SystemExit(f"ERROR: spglib could not identify the subgroup: {exc}")
        if info is None:
            raise SystemExit("ERROR: spglib could not identify the subgroup.")
        from .runtime_compat import get_spacegroup_type

        info = get_spacegroup_type(info)
        n_point = len({algebra._key(np.rint(r).astype(np.int64)) for r in rotations})
        index = algebra.n_ops * size // n_point
        return info, size, index, B, rotations, translations, lattice

    def conventional_setting(self, B, rotations, translations, lattice, info):
        """Child conventional basis and origin, in parent-conventional units.

        Built from a generic-orbit structure with exactly the subgroup
        symmetry, standardized by spglib.
        """
        import spglib

        positions = []
        numbers = []
        for species, x0 in enumerate(
            (np.array([0.1234, 0.2345, 0.3178]), np.array([0.4321, 0.0567, 0.1873]))
        ):
            orbit = []
            for W, v in zip(rotations, translations):
                x = np.mod(W @ x0 + v, 1.0)
                if not any(np.allclose(x, p, atol=1e-6) for p in orbit):
                    orbit.append(x)
            positions.extend(orbit)
            numbers.extend([species + 1] * len(orbit))
        dataset = spglib.get_symmetry_dataset(
            (lattice, np.array(positions), numbers), symprec=1e-4
        )

        def field(name):  # spglib < 2.4 returns a dict, >= 2.4 an object
            if dataset is None:
                return None
            if isinstance(dataset, dict):
                return dataset.get(name)
            return getattr(dataset, name, None)

        if dataset is None or field("number") != info.number:
            return None
        P = np.array(field("transformation_matrix"), dtype=float)
        shift = np.array(field("origin_shift"), dtype=float)
        # child conventional lattice rows in cartesian: L_c = (P^-1)^T L_input
        L_child_conv = np.linalg.inv(P).T @ lattice
        # parent conventional lattice rows: A_p = M^T A_c (phonopy convention)
        M = self.algebra.primitive_matrix
        L_parent_prim = self._invariant_lattice()
        L_parent_conv = np.linalg.inv(M).T @ L_parent_prim
        basis = L_child_conv @ np.linalg.inv(L_parent_conv)
        # child origin: x_std = P x + p -> the child cell origin (x_std = 0)
        # sits at x = -P^-1 p (input = subgroup-primitive coords)
        origin_sub = -np.linalg.inv(P) @ shift
        origin_cart = origin_sub @ lattice
        origin = origin_cart @ np.linalg.inv(L_parent_conv)
        return np.round(basis, 6), np.round(origin, 6)

    def _invariant_lattice(self) -> np.ndarray:
        """A parent primitive lattice (rows) with the full point symmetry."""
        g0 = np.diag([1.0, 1.07, 1.13])
        g = np.zeros((3, 3))
        for W in self.algebra.rotations:
            g += W.T @ g0 @ W
        g /= self.algebra.n_ops
        return np.linalg.cholesky(g).T

    # -- direction formatting / parsing
    def direction_label(self, projector: np.ndarray) -> tuple[str, np.ndarray]:
        """Pretty parameter pattern of a stratum and a generic representative."""
        basis = _orth_basis(projector)
        n_free = basis.shape[1]
        # RREF + integer prettification (same style as the molecular SALCs)
        from .molecular_salc import _pretty_coefficients, _rref_orthogonal

        rows = _rref_orthogonal([basis[:, j] for j in range(n_free)])
        generic = np.zeros(self.representation.dimension)
        magnitudes = [1.0, 0.6180339887, 0.4142135624, 0.2928932188,
                      0.2360679775, 0.1926, 0.1573, 0.1235]
        for j, row in enumerate(rows):
            generic = generic + magnitudes[j] * np.asarray(row)
        pretty_rows = []
        for row in rows:
            coefficients, _ = _pretty_coefficients(np.asarray(row))
            pretty_rows.append(np.asarray(coefficients, dtype=float))
        components = []
        for slot in range(self.representation.dimension):
            terms = []
            for j, row in enumerate(pretty_rows):
                value = row[slot]
                if abs(value) < 1e-8:
                    continue
                terms.append(_format_coefficient(value) + _PARAMETER_NAMES[j])
            components.append("+".join(terms).replace("+-", "-") if terms else "0")
        return "(" + ",".join(components) + ")", generic

    def resolve_direction(self, tokens: list[str]) -> np.ndarray:
        """Build a representative order parameter from CLI tokens like
        0 0 a  /  a a 0  /  a b 0  /  numeric values."""
        n = self.representation.dimension
        if len(tokens) != n:
            raise SystemExit(
                f"ERROR: --order-parameter needs {n} components for "
                f"{self.representation.irrep.name} (dim {n})."
            )
        values = np.zeros(n)
        symbol_values: dict[str, float] = {}
        magnitudes = [1.0, 0.6180339887, 0.4142135624, 0.2928932188]
        for slot, token in enumerate(tokens):
            token = token.strip()
            sign = 1.0
            if token.startswith("-"):
                sign, token = -1.0, token[1:]
            if token in ("0", "0.0", ""):
                continue
            try:
                values[slot] = sign * float(Fraction(token))
                continue
            except ValueError:
                pass
            if token not in symbol_values:
                symbol_values[token] = magnitudes[len(symbol_values) % len(magnitudes)]
            values[slot] = sign * symbol_values[token]
        if not np.any(values):
            raise SystemExit("ERROR: the order parameter must not be zero.")
        return values


def _realify_matrix_set(matrices: dict) -> dict | None:
    """Similarity-transform a set of unitary matrices to real form, when a
    real form exists (real-type rep); returns None otherwise."""
    keys = list(matrices)
    if all(np.allclose(np.asarray(matrices[key]).imag, 0, atol=1e-8) for key in keys):
        return {key: np.asarray(matrices[key]).real.copy() for key in keys}
    rng = np.random.default_rng(7)
    n = np.asarray(matrices[keys[0]]).shape[0]
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    S = np.zeros((n, n), dtype=np.complex128)
    for key in keys:
        D = np.asarray(matrices[key])
        S += np.conj(D) @ A @ D.conj().T
    c_matrix = S @ np.conj(S)
    c = c_matrix[0, 0]
    if not np.allclose(c_matrix, c * np.eye(n), atol=1e-6 * max(1, abs(c))) or c.real <= 0:
        return None
    S = S / np.sqrt(c.real)
    basis: list[np.ndarray] = []
    trial = 0
    while len(basis) < n and trial < 20 * n:
        trial += 1
        v = rng.normal(size=n) + 1j * rng.normal(size=n)
        w = v + S @ np.conj(v)
        for prior in basis:
            w = w - prior * np.real(np.vdot(prior, w))
        norm = np.linalg.norm(w)
        if norm > 1e-6:
            basis.append(w / norm)
    if len(basis) < n:
        return None
    T = np.column_stack(basis)
    T_inv = np.linalg.inv(T)
    result = {}
    for key in keys:
        transformed = T_inv @ np.asarray(matrices[key]) @ T
        if not np.allclose(transformed.imag, 0, atol=1e-6):
            return None
        result[key] = transformed.real.copy()
    return result


def _orth_basis(projector: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((projector + projector.T.conj()) / 2)
    return np.real_if_close(vectors[:, values > 0.5])


def _projector_key(projector: np.ndarray) -> bytes:
    rounded = np.round(np.real(projector), 6) + 0.0  # normalize -0.0
    return rounded.tobytes()


def _label_rank(label: str) -> tuple:
    """Ordering that prefers simple direction labels ((a,a,0) over (a,-b,b))."""
    return (label.count("."), label.count("-"), len(label), label)


def _format_setting_value(value: float) -> str:
    fraction = Fraction(float(value)).limit_denominator(12)
    if abs(float(fraction) - float(value)) < 1e-4:
        if fraction.denominator == 1:
            return str(fraction.numerator)
        return f"{fraction.numerator}/{fraction.denominator}"
    return f"{float(value):.4g}"


def _format_coefficient(value: float) -> str:
    if abs(value - 1.0) < 1e-6:
        return ""
    if abs(value + 1.0) < 1e-6:
        return "-"
    return f"{value:.3g}"


# ---------------------------------------------------------------------- report


def format_subgroup_line(analyzer, label, info, size, index) -> str:
    return (
        f"{label:<18} {info.number:>4} {info.international_short:<10} "
        f"size {size}  index {index}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Isotropy subgroups of a space-group irrep."
    )
    parser.add_argument("--supergroup", required=True, help='e.g. "Pm-3m" or 221.')
    parser.add_argument("--irrep", required=True, help="CDML irrep label, e.g. GM4-.")
    parser.add_argument(
        "--order-parameter",
        nargs="+",
        default=None,
        help='components, e.g. "0 0 a" or "a a 0" (symbols = free parameters).',
    )
    args = parser.parse_args(argv)

    import spglib

    spglib_version = tuple(int(x) for x in spglib.__version__.split(".")[:2])
    if spglib_version < (2, 4):
        print(
            "WARNING: spglib >= 2.4 is recommended for reliable subgroup "
            f"identification (found {spglib.__version__}).",
        )

    analyzer = IsotropyAnalyzer(args.supergroup, args.irrep)
    representation = analyzer.representation
    algebra = analyzer.algebra
    irrep_name = representation.irrep.name

    print()
    print("* Supergroup *")
    print(f"{algebra.sg_type.international_short} (No. {algebra.sg_type.number})")
    print()
    print("* Irrep *")
    star_note = (
        f" (star of {representation.n_arms} arm(s) x small dim "
        f"{representation.dim_small})"
        if representation.n_arms > 1
        else ""
    )
    print(f"{irrep_name}: order parameter dimension {representation.dimension}{star_note}")
    print()

    if args.order_parameter:
        eta = analyzer.resolve_direction(args.order_parameter)
        projector = _projector(eta[:, None])
        members = analyzer.stabilizer_of(projector)
        # the direction may be non-generic in its own fixed space; use the
        # exact stabilizer of eta itself
        members = [
            (i, t)
            for i, t, matrix in analyzer.elements
            if np.allclose(matrix @ eta, eta, atol=1e-6)
        ]
        info, size, index, B, rotations, translations, lattice = analyzer.subgroup_of(members)
        direction = "(" + ",".join(args.order_parameter) + ")"
        print("* Isotropy subgroup *")
        print(f"{irrep_name} {direction} -> {info.international_short} (No. {info.number})")
        print(f"cell size {size}, index {index}")
        basis_rows = ", ".join("(" + ",".join(str(int(x)) for x in row) + ")" for row in B)
        print(f"sublattice basis (parent primitive units): {basis_rows}")
        setting = analyzer.conventional_setting(B, rotations, translations, lattice, info)
        if setting is not None:
            basis, origin = setting
            rows = ", ".join(
                "(" + ",".join(_format_setting_value(x) for x in row) + ")" for row in basis
            )
            origin_text = "(" + ",".join(_format_setting_value(x) for x in origin) + ")"
            print(f"conventional basis (parent conventional units): {rows}")
            print(f"origin: {origin_text}")
    else:
        strata = analyzer.enumerate_directions()
        results = []
        for projector, members in strata:
            label, generic = analyzer.direction_label(projector)
            exact_members = [
                (i, t)
                for i, t, matrix in analyzer.elements
                if np.allclose(matrix @ generic, generic, atol=1e-6)
            ]
            info, size, index, B, *_ = analyzer.subgroup_of(exact_members)
            n_free = _orth_basis(projector).shape[1]
            results.append((index, n_free, label, info, size))
        results.sort(key=lambda r: (r[1], r[0], r[3].number))
        print("* Order parameter directions and isotropy subgroups *")
        print(f"{'direction':<20} {'subgroup':<18} {'size':<5} {'index':<5}")
        for index, n_free, label, info, size in results:
            subgroup = f"{info.number} {info.international_short}"
            print(f"{label:<20} {subgroup:<18} {size:<5} {index:<5}")
    print()
    print("Conventions and validation: ISOSUBGROUP (https://iso.byu.edu):")
    print('H. T. Stokes, S. van Orden and B. J. Campbell, "Tool for Generating')
    print('Isotropy Subgroups of Crystallographic Space Groups",')
    print("J. Appl. Cryst. 49, 1849-1853 (2016).")


if __name__ == "__main__":
    main()
