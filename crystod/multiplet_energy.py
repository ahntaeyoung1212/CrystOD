"""Coulomb multiplet energies in Racah/Slater parameters (crystod-group --multiplet).

For an electron configuration over irrep shells of one atomic l shell, the
diagonal energies <psi(term)|H_ee|psi(term)> of the electron-electron
(Coulomb) interaction are exact linear combinations of the Slater-Condon
parameters F^k (k = 0, 2, ..., 2l).  For d shells they are expressed in the
Racah parameters A, B, C (F^0 = A + 7C/5, F^2 = 49B + 7C, F^4 = 63C/5), e.g.
(t2g)^3 of m-3m:

    ^4A2g: 3A - 15B      ^2Eg: 3A - 6B + 3C
    ^2T1g: 3A - 6B + 3C  ^2T2g: 3A + 5C

(Tanabe-Sugano strong-field energies).  The pipeline is exact and general:

1. two-electron integrals <m1 m2|1/r12|m3 m4> in the complex Y_lm basis from
   Gaunt coefficients (sympy, exact), transformed to the real-orbital basis;
2. the shell orbitals are identified as the irrep subspaces of the
   (2l+1)-dimensional real-orbital representation (projection with
   wigner_D_real over the point-group elements);
3. the Coulomb Hamiltonian is built over the Slater determinants of the
   configuration (Slater-Condon rules) and each term is isolated with a total-
   spin projector (polynomial in S^2) times the point-group character
   projector (the group action on determinants);
4. a term appearing once has an exact linear energy; a term appearing m times
   mixes within the configuration (configuration interaction): for m = 2 the
   two eigenvalues are printed in closed form (mean +- sqrt of a quadratic
   form), for larger m numerically at C/B = 4.5.

Every run is closed by the trace identity sum (2S+1) dim(Gamma) E = tr(H_ee)
over the configuration.  The one-electron (crystal-field) part is an additive
constant within a configuration and is omitted.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from math import comb

import numpy as np

from .ligand_field import ORBITAL_AZIMUTHAL_NUMBER
from .molecular_salc import _table_operations_cartesian
from .operations import wigner_D_real

# Condon-Shortley reduced denominators F_k = F^k / DENOM[l][k/2] make the
# term-energy coefficients small integers; for l = 2 the Racah combination
# A = F_0 - 49 F_4, B = F_2 - 5 F_4, C = 35 F_4 is used instead.
_REDUCED_DENOMS = {
    0: [Fraction(1)],
    1: [Fraction(1), Fraction(25)],
    2: [Fraction(1), Fraction(49), Fraction(441)],
    3: [Fraction(1), Fraction(225), Fraction(1089), Fraction(184041, 25)],
}
_PARAM_NAMES = {
    0: ["F0"],
    1: ["F0", "F2"],
    2: ["A", "B", "C"],
    3: ["F0", "F2", "F4", "F6"],
}
_SNAP_TOL = 1e-6
_TYPICAL_C_OVER_B = Fraction(9, 2)


# ---------------------------------------------------------------------------
# exact two-electron integrals of one l shell
# ---------------------------------------------------------------------------


def _complex_two_electron(l: int):
    """<m1 m2|1/r12|m3 m4> = sum_k c^k(m1,m3) c^k(m4,m2) F^k as exact sympy
    coefficient vectors over k = 0, 2, ..., 2l."""
    import sympy
    from sympy.physics.wigner import gaunt

    k_values = list(range(0, 2 * l + 1, 2))

    def c_coefficient(k, m, mp):
        # c^k(l m; l m') = sqrt(4 pi / (2k+1)) int Y*_{lm} Y_{k,m-m'} Y_{lm'}
        return (
            sympy.sqrt(4 * sympy.pi / (2 * k + 1))
            * sympy.Integer(-1) ** m
            * gaunt(l, k, l, -m, m - mp, mp)
        )

    integrals = {}
    m_range = range(-l, l + 1)
    for m1 in m_range:
        for m2 in m_range:
            for m3 in m_range:
                m4 = m1 + m2 - m3
                if m4 < -l or m4 > l:
                    continue
                coeffs = [
                    sympy.simplify(c_coefficient(k, m1, m3) * c_coefficient(k, m4, m2))
                    for k in k_values
                ]
                if any(value != 0 for value in coeffs):
                    integrals[(m1, m2, m3, m4)] = coeffs
    return integrals, k_values


def _sympy_real_transform(l: int):
    """Exact sympy version of complex_to_real_transform_orbital."""
    import sympy

    size = 2 * l + 1
    inv_sqrt2 = 1 / sympy.sqrt(2)
    rows = {}
    for m in range(-l, l + 1):
        row = [sympy.Integer(0)] * size
        if m < 0:
            row[l + m] = sympy.I * inv_sqrt2
            row[l - m] = -((-1) ** abs(m)) * sympy.I * inv_sqrt2
        elif m == 0:
            row[l] = sympy.Integer(1)
        else:
            row[l - m] = inv_sqrt2
            row[l + m] = ((-1) ** m) * inv_sqrt2
        rows[m] = row

    if l == 1:
        order = [1, -1, 0]  # px, py, pz
    elif l == 2:
        order = [-2, -1, 0, 1, 2]  # dxy, dyz, dz2, dxz, dx2-y2
    elif l == 3:
        order = [3, -3, 2, -2, 1, -1, 0]
    else:
        order = list(range(-l, l + 1))
    return [rows[m] for m in order]


def real_two_electron_integrals(l: int) -> dict[tuple[int, int, int, int], tuple]:
    """<ab|1/r12|cd> in the real-orbital basis as coefficient tuples over the
    printed parameters (Racah A, B, C for l = 2, reduced F_k otherwise).
    Coefficients are high-precision floats: the tensor entries of f shells
    are not all rational (sqrt-valued off-diagonal integrals); the final term
    energies are snapped to exact fractions downstream."""
    import sympy

    complex_integrals, k_values = _complex_two_electron(l)
    transform = _sympy_real_transform(l)
    size = 2 * l + 1
    n_params = len(k_values)

    # nonzero complex components of each real orbital
    components = [
        [(m - l, value) for m, value in enumerate(row) if value != 0]
        for row in transform
    ]

    raw: dict[tuple[int, int, int, int], list] = {}
    for a in range(size):
        for b in range(size):
            for c in range(size):
                for d in range(size):
                    total = [sympy.Integer(0)] * n_params
                    for m1, t1 in components[a]:
                        for m2, t2 in components[b]:
                            for m3, t3 in components[c]:
                                for m4, t4 in components[d]:
                                    entry = complex_integrals.get((m1, m2, m3, m4))
                                    if entry is None:
                                        continue
                                    weight = (
                                        sympy.conjugate(t1)
                                        * sympy.conjugate(t2)
                                        * t3
                                        * t4
                                    )
                                    for p in range(n_params):
                                        total[p] += weight * entry[p]
                    values = []
                    for v in total:
                        # the integral is real, but its F^k coefficients need
                        # not be rational (f shells have sqrt-valued entries)
                        # -> evaluate at high precision and keep as float
                        real_part, imag_part = sympy.expand(v).evalf(40).as_real_imag()
                        if abs(float(imag_part)) > 1e-20:
                            raise SystemExit(
                                "ERROR: two-electron integral came out complex; "
                                "please report this case."
                            )
                        values.append(float(real_part))
                    if any(abs(v) > 1e-14 for v in values):
                        raw[(a, b, c, d)] = values

    # F^k -> printed parameters (exact linear substitution)
    if l == 2:
        # F^0 = A + 7C/5, F^2 = 49B + 7C, F^4 = 63C/5
        conversion = [
            [Fraction(1), Fraction(0), Fraction(0)],
            [Fraction(0), Fraction(49), Fraction(0)],
            [Fraction(0), Fraction(0), Fraction(0)],
        ]
        conversion[0][2] = Fraction(7, 5)
        conversion[1][2] = Fraction(7)
        conversion[2][2] = Fraction(63, 5)
    else:
        denoms = _REDUCED_DENOMS[l]
        conversion = [
            [denoms[i] if i == j else Fraction(0) for j in range(n_params)]
            for i in range(n_params)
        ]

    integrals = {}
    for key, values in raw.items():
        converted = tuple(
            sum(values[k] * float(conversion[k][p]) for k in range(n_params))
            for p in range(n_params)
        )
        integrals[key] = converted
    return integrals


# ---------------------------------------------------------------------------
# shell subspaces of the l shell
# ---------------------------------------------------------------------------


def _shell_bases(character_table: dict, l: int, shell_irreps: list[str]):
    """Orthonormal basis (columns) of each shell's irrep subspace within the
    (2l+1)-dimensional real-orbital space, plus the Cartesian group action."""
    operations, class_labels = _table_operations_cartesian(character_table)
    class_names = list(character_table["rotation_list"])
    class_index = {name: i for i, name in enumerate(class_names)}
    d_matrices = [wigner_D_real(l, op) for op in operations]
    order = len(operations)

    bases = []
    for irrep in shell_irreps:
        characters = np.asarray(
            character_table["character_table"][irrep], dtype=float
        )
        dim = int(round(characters[class_index["E"]]))
        projector = np.zeros((2 * l + 1, 2 * l + 1))
        for label, dmat in zip(class_labels, d_matrices):
            projector += characters[class_index[label]] * dmat
        projector *= dim / order
        eigenvalues, eigenvectors = np.linalg.eigh((projector + projector.T) / 2)
        kept = eigenvectors[:, eigenvalues > 0.5]
        if kept.shape[1] != dim:
            raise SystemExit(
                f"ERROR: the {irrep} subspace of the l={l} shell has "
                f"multiplicity {kept.shape[1] // max(dim, 1)} in this point "
                "group; the shell orbitals are not defined by symmetry alone, "
                "so multiplet energies are unavailable here."
            )
        bases.append(kept)
    return bases, operations, class_labels, d_matrices, class_index


# ---------------------------------------------------------------------------
# Slater determinants and the Coulomb Hamiltonian
# ---------------------------------------------------------------------------


class _DeterminantSpace:
    """Slater determinants of a fixed shell-occupation configuration.

    Spin-orbital index = 2*orbital + spin (spin 0 = up, 1 = down) over the
    concatenated shell orbitals."""

    def __init__(self, shell_dims: list[int], occupations: list[int]):
        self.n_orbitals = sum(shell_dims)
        offsets = np.cumsum([0] + shell_dims)
        shell_choices = []
        for dim, n in zip(shell_dims, occupations):
            offset = offsets[len(shell_choices)]
            spin_orbitals = [
                2 * (offset + orb) + spin for orb in range(dim) for spin in (0, 1)
            ]
            shell_choices.append(
                [tuple(sorted(chosen)) for chosen in combinations(spin_orbitals, n)]
            )
        dets = [()]
        for choices in shell_choices:
            dets = [prev + choice for prev in dets for choice in choices]
        self.dets = [tuple(sorted(det)) for det in dets]
        self.index = {det: i for i, det in enumerate(self.dets)}

    def sz2(self, det) -> int:
        """2 Sz of a determinant."""
        return sum(1 if so % 2 == 0 else -1 for so in det)

    def sector(self, sz2: int) -> list[int]:
        return [i for i, det in enumerate(self.dets) if self.sz2(det) == sz2]


def _apply_one_body(det: tuple, create: int, annihilate: int):
    """a+_create a_annihilate |det> -> (sign, new det) or None."""
    if annihilate not in det:
        return None
    position = det.index(annihilate)
    sign = (-1) ** position
    removed = det[:position] + det[position + 1 :]
    if create in removed:
        return None
    insert_at = sum(1 for so in removed if so < create)
    sign *= (-1) ** insert_at
    return sign, removed[:insert_at] + (create,) + removed[insert_at:]


def _coulomb_hamiltonians(space, sector, integrals_by_param):
    """One numpy matrix per parameter on the determinant sector
    (Slater-Condon rules; spin-conserving <ij|kl> delta factors)."""
    n_params = len(integrals_by_param)
    size = len(sector)
    matrices = [np.zeros((size, size)) for _ in range(n_params)]
    dets = [space.dets[i] for i in sector]
    position = {space.dets[i]: row for row, i in enumerate(sector)}

    def v(p, i, j, k, l_):
        """<ij|kl> over spin-orbitals for parameter p (0 unless spins match)."""
        if i % 2 != k % 2 or j % 2 != l_ % 2:
            return 0.0
        return integrals_by_param[p].get((i // 2, j // 2, k // 2, l_ // 2), 0.0)

    for row, det in enumerate(dets):
        occupied = list(det)
        # diagonal
        for p in range(n_params):
            total = 0.0
            for a, b in combinations(occupied, 2):
                total += v(p, a, b, a, b) - v(p, a, b, b, a)
            matrices[p][row, row] = total
        # single and double excitations within the same shell-occupation space
        for i_pos, i in enumerate(occupied):
            for a in range(2 * space.n_orbitals):
                if a in det:
                    continue
                one = _apply_one_body(det, a, i)
                if one is None:
                    continue
                sign_a, det_a = one
                col = position.get(det_a)
                if col is not None and col > row:
                    for p in range(n_params):
                        total = 0.0
                        for j in occupied:
                            if j == i:
                                continue
                            total += v(p, a, j, i, j) - v(p, a, j, j, i)
                        value = sign_a * total
                        matrices[p][row, col] += value
                        matrices[p][col, row] += value
                for j_pos in range(i_pos + 1, len(occupied)):
                    j = occupied[j_pos]
                    for b in range(a + 1, 2 * space.n_orbitals):
                        if b in det:
                            continue
                        two = _apply_one_body(det_a, b, j)
                        if two is None:
                            continue
                        sign_b, det2 = two
                        col = position.get(det2)
                        if col is None or col <= row:
                            continue
                        for p in range(n_params):
                            value = sign_a * sign_b * (
                                v(p, a, b, i, j) - v(p, a, b, j, i)
                            )
                            matrices[p][row, col] += value
                            matrices[p][col, row] += value
    return matrices


def _s2_matrix(space, sector):
    """S^2 on the determinant sector (units hbar = 1)."""
    size = len(sector)
    matrix = np.zeros((size, size))
    position = {space.dets[i]: row for row, i in enumerate(sector)}
    for row, det_index in enumerate(sector):
        det = space.dets[det_index]
        sz = space.sz2(det) / 2.0
        matrix[row, row] += sz * sz + sz
        # S- S+ : S+ = sum_p a+_{p up} a_{p down}
        for orb in range(space.n_orbitals):
            up, down = 2 * orb, 2 * orb + 1
            plus = _apply_one_body(det, up, down)
            if plus is None:
                continue
            sign_p, det_p = plus
            for orb2 in range(space.n_orbitals):
                up2, down2 = 2 * orb2, 2 * orb2 + 1
                minus = _apply_one_body(det_p, down2, up2)
                if minus is None:
                    continue
                sign_m, det_m = minus
                col = position.get(det_m)
                if col is not None:
                    matrix[row, col] += sign_p * sign_m
    return matrix


def _group_action(space, sector, orbital_matrices):
    """Determinant-space matrices of the group elements on a sector.

    orbital_matrices: per group element, the M x M orthogonal matrix on the
    concatenated shell orbitals (block diagonal over shells)."""
    dets = [space.dets[i] for i in sector]
    size = len(dets)
    matrices = []
    for u in orbital_matrices:
        result = np.zeros((size, size))
        for col, det in enumerate(dets):
            occ = list(det)
            n = len(occ)
            # column vectors of transformed spin-orbitals in the full basis
            for row, target in enumerate(dets):
                sub = np.zeros((n, n))
                for jj, so_t in enumerate(target):
                    for kk, so_c in enumerate(occ):
                        if so_t % 2 != so_c % 2:
                            continue
                        sub[jj, kk] = u[so_t // 2, so_c // 2]
                value = np.linalg.det(sub) if n else 1.0
                if abs(value) > 1e-12:
                    result[row, col] = value
        matrices.append(result)
    return matrices


# ---------------------------------------------------------------------------
# term energies
# ---------------------------------------------------------------------------


def _snap(value: float) -> Fraction:
    fraction = Fraction(value).limit_denominator(720)
    if abs(float(fraction) - value) > _SNAP_TOL:
        raise SystemExit(
            "ERROR: multiplet-energy coefficient failed to snap to an exact "
            f"fraction ({value}); please report this case."
        )
    return fraction


def format_linear(coeffs, params) -> str:
    parts = []
    for coeff, param in zip(coeffs, params):
        if coeff == 0:
            continue
        magnitude = abs(coeff)
        if magnitude.denominator == 1:
            body = param if magnitude == 1 else f"{magnitude.numerator}{param}"
        else:
            body = f"({magnitude}){param}"
        if not parts:
            parts.append(body if coeff > 0 else f"-{body}")
        else:
            parts.append(f"+ {body}" if coeff > 0 else f"- {body}")
    return " ".join(parts) if parts else "0"


def _format_quadratic(matrix, params) -> str:
    parts = []
    n = len(params)
    for i in range(n):
        for j in range(i, n):
            coeff = matrix[i][j] if i == j else matrix[i][j] + matrix[j][i]
            if coeff == 0:
                continue
            name = f"{params[i]}^2" if i == j else f"{params[i]}{params[j]}"
            magnitude = abs(coeff)
            if magnitude.denominator == 1:
                body = name if magnitude == 1 else f"{magnitude.numerator}{name}"
            else:
                body = f"({magnitude}){name}"
            if not parts:
                parts.append(body if coeff > 0 else f"-{body}")
            else:
                parts.append(f"+ {body}" if coeff > 0 else f"- {body}")
    return " ".join(parts) if parts else "0"


def _single_square_root(matrix, params) -> str | None:
    """(1/2)sqrt(Q) as k sqrt(m) PARAM when Q = c PARAM^2 (one variable only)."""
    n = len(params)
    nonzero = [
        (i, j)
        for i in range(n)
        for j in range(n)
        if matrix[i][j] != 0
    ]
    if len(nonzero) != 1 or nonzero[0][0] != nonzero[0][1]:
        return None
    index = nonzero[0][0]
    coeff = matrix[index][index]
    if coeff <= 0:
        return None

    def split_square(value: int) -> tuple[int, int]:
        square, rest, factor = 1, value, 2
        while factor * factor <= rest:
            while rest % (factor * factor) == 0:
                rest //= factor * factor
                square *= factor
            factor += 1
        return square, rest

    sq_num, rad_num = split_square(coeff.numerator)
    sq_den, rad_den = split_square(coeff.denominator)
    rational = Fraction(sq_num, 2 * sq_den * rad_den)
    radicand = rad_num * rad_den
    if radicand == 1:
        prefix = "" if rational == 1 else (
            str(rational.numerator)
            if rational.denominator == 1
            else f"({rational})"
        )
        return f"{prefix}{params[index]}"
    prefix = "" if rational == 1 else (
        str(rational.numerator) if rational.denominator == 1 else f"({rational})"
    )
    return f"{prefix}sqrt({radicand}){params[index]}"


class TermEnergy:
    """Energy of one term: exact linear coefficients when unique in the
    configuration, or a CI block (m >= 2)."""

    def __init__(self, spin, irrep, dim, multiplicity, params):
        self.spin = spin
        self.irrep = irrep
        self.dim = dim
        self.multiplicity = multiplicity
        self.params = params
        self.linear = None  # tuple of Fractions (m == 1)
        self.ci_mean = None  # tuple of Fractions: (E1+E2)/2 (m == 2)
        self.ci_quadratic = None  # matrix of Fractions: (E1-E2)^2 (m == 2)
        self.numeric = None  # list of floats at the reference point
        self.reference_note = "the reference parameter point"

    def reference_values(self, reference) -> list[float]:
        """Numeric energies at the reference parameter values."""
        if self.linear is not None:
            return [float(sum(c * r for c, r in zip(self.linear, reference)))]
        return list(self.numeric)

    def describe(self) -> list[str]:
        if self.linear is not None:
            return [format_linear(self.linear, self.params)]
        if self.ci_mean is not None:
            mean = format_linear(self.ci_mean, self.params)
            simple = _single_square_root(self.ci_quadratic, self.params)
            if simple is not None:
                return [f"{mean} +- {simple}   (x2, configuration mixing)"]
            disc = _format_quadratic(self.ci_quadratic, self.params)
            return [f"{mean} +- (1/2)sqrt({disc})   (x2, configuration mixing)"]
        values = ", ".join(f"{value:+.3f}" for value in sorted(self.numeric))
        return [
            f"[{values}]  (x{self.multiplicity}, configuration mixing; numeric)"
        ]


def compute_term_energies(
    character_table: dict,
    l: int,
    shells: list[tuple[str, int]],
    terms: list[tuple[Fraction, str, int]],
):
    """Coulomb energies of every term of the configuration.

    Returns (params, list of TermEnergy ordered like `terms`)."""
    params = _PARAM_NAMES[l]
    n_params = len(params)
    shell_irreps = [name for name, _ in shells]
    occupations = [count for _, count in shells]

    bases, operations, class_labels, d_matrices, class_index = _shell_bases(
        character_table, l, shell_irreps
    )
    basis = np.hstack(bases)  # (2l+1) x M
    shell_dims = [b.shape[1] for b in bases]

    exact = real_two_electron_integrals(l)
    # transform the 4-index tensor into the shell-orbital basis, per parameter
    size = 2 * l + 1
    integrals_by_param = []
    for p in range(n_params):
        tensor = np.zeros((size, size, size, size))
        for (a, b, c, d), coeffs in exact.items():
            tensor[a, b, c, d] = float(coeffs[p])
        transformed = np.einsum(
            "abcd,ap,bq,cr,ds->pqrs", tensor, basis, basis, basis, basis
        )
        entries = {}
        m = basis.shape[1]
        for a in range(m):
            for b in range(m):
                for c in range(m):
                    for d in range(m):
                        value = transformed[a, b, c, d]
                        if abs(value) > 1e-12:
                            entries[(a, b, c, d)] = value
        integrals_by_param.append(entries)

    # orbital-space group action in the shell basis (block over shells)
    orbital_matrices = []
    for dmat in d_matrices:
        blocks = [b.T @ dmat @ b for b in bases]
        m = basis.shape[1]
        u = np.zeros((m, m))
        offset = 0
        for block in blocks:
            k = block.shape[0]
            u[offset : offset + k, offset : offset + k] = block
            offset += k
        orbital_matrices.append(u)

    space = _DeterminantSpace(shell_dims, occupations)
    n_electrons = sum(occupations)
    order = len(operations)

    # reference point for numeric CI blocks and the ground-state selection:
    # typical physical parameter ratios of the shell
    reference = [0.0] * n_params
    if l == 2:
        reference[1] = 1.0
        reference[2] = float(_TYPICAL_C_OVER_B)
        reference_note = "C/B = 4.5, typical for 3d ions"
    elif l == 3:
        # hydrogenic 4f ratios of the reduced parameters:
        # F4/F2 = 0.138, F6/F2 = 0.0151
        reference[1] = 1.0
        reference[2] = 0.138
        reference[3] = 0.0151
        reference_note = "hydrogenic 4f ratios F4/F2 = 0.138, F6/F2 = 0.0151"
    elif l == 1:
        reference[1] = 1.0
        reference_note = "F2 = 1"
    else:
        reference[0] = 1.0
        reference_note = "F0 = 1"

    results = []
    sector_cache: dict[int, dict] = {}
    for spin, irrep, multiplicity in terms:
        sz2 = int(2 * spin)
        if sz2 not in sector_cache:
            sector = space.sector(sz2)
            h_matrices = _coulomb_hamiltonians(space, sector, integrals_by_param)
            s2 = _s2_matrix(space, sector)
            group = _group_action(space, sector, orbital_matrices)
            sector_cache[sz2] = {
                "sector": sector,
                "h": h_matrices,
                "s2": s2,
                "group": group,
            }
        cache = sector_cache[sz2]
        sector, h_matrices, s2 = cache["sector"], cache["h"], cache["s2"]

        # spin projector: polynomial in S^2 annihilating the other S values
        target = float(spin * (spin + 1))
        projector = np.eye(len(sector))
        s_value = Fraction(sz2, 2)
        possible = []
        s_iter = s_value
        while s_iter <= Fraction(n_electrons, 2):
            possible.append(float(s_iter * (s_iter + 1)))
            s_iter += 1
        for other in possible:
            if abs(other - target) < 1e-9:
                continue
            projector = projector @ (s2 - other * np.eye(len(sector))) / (
                target - other
            )

        # point-group character projector on the determinant sector
        characters = np.asarray(
            character_table["character_table"][irrep], dtype=float
        )
        dim = int(round(characters[class_index["E"]]))
        pg_projector = np.zeros((len(sector), len(sector)))
        for label, gmat in zip(class_labels, cache["group"]):
            pg_projector += characters[class_index[label]] * gmat
        pg_projector *= dim / order

        combined = pg_projector @ projector
        combined = (combined + combined.T) / 2
        eigenvalues, eigenvectors = np.linalg.eigh(combined)
        kept = eigenvectors[:, eigenvalues > 0.5]
        expected = multiplicity * dim
        if kept.shape[1] != expected:
            raise SystemExit(
                f"ERROR: projector rank {kept.shape[1]} != {expected} for "
                f"term ({spin}, {irrep}); please report this case."
            )

        blocks = [kept.T @ h @ kept for h in h_matrices]
        entry = TermEnergy(spin, irrep, dim, multiplicity, params)
        entry.reference_note = reference_note
        if multiplicity == 1:
            entry.linear = tuple(
                _snap(np.trace(block) / dim) for block in blocks
            )
        else:
            trace_lin = [_snap(np.trace(block) / dim) for block in blocks]
            reference_block = sum(
                r * block for r, block in zip(reference, blocks)
            )
            block_eigen = np.linalg.eigvalsh(reference_block)
            grouped = sorted(block_eigen)
            entry.numeric = [
                float(np.mean(grouped[i * dim : (i + 1) * dim]))
                for i in range(multiplicity)
            ]
            if multiplicity == 2:
                entry.ci_mean = tuple(value / 2 for value in trace_lin)
                # (E1 - E2)^2 = 2 tr(Hb^2)/dim - (tr Hb / dim)^2
                quad = [
                    [Fraction(0)] * n_params for _ in range(n_params)
                ]
                for p in range(n_params):
                    for q in range(n_params):
                        second = _snap(
                            np.trace(blocks[p] @ blocks[q]) / dim
                        )
                        quad[p][q] = 2 * second - trace_lin[p] * trace_lin[q]
                entry.ci_quadratic = quad
        results.append(entry)

    _verify_trace(space, integrals_by_param, results, reference, n_params)
    return params, results, reference, reference_note


def _verify_trace(space, integrals_by_param, results, reference, n_params):
    """Trace identity: sum over terms (2S+1) dim E = tr(H_ee) per parameter."""
    for p in range(n_params):
        total = 0.0
        for det in space.dets:
            for a, b in combinations(det, 2):
                if a % 2 == b % 2:
                    total += integrals_by_param[p].get(
                        (a // 2, b // 2, a // 2, b // 2), 0.0
                    ) - integrals_by_param[p].get(
                        (a // 2, b // 2, b // 2, a // 2), 0.0
                    )
                else:
                    total += integrals_by_param[p].get(
                        (a // 2, b // 2, a // 2, b // 2), 0.0
                    )
        term_sum = 0.0
        for entry in results:
            weight = int(2 * entry.spin + 1) * entry.dim
            if entry.linear is not None:
                term_sum += weight * float(entry.linear[p])
            elif entry.ci_mean is not None:
                term_sum += weight * 2 * float(entry.ci_mean[p])
            else:
                unit = [1.0 if q == p else 0.0 for q in range(n_params)]
                # numeric-only blocks: skip exact per-parameter check
                term_sum = None
                break
        if term_sum is None:
            break
        if abs(term_sum - total) > 1e-6:
            raise SystemExit(
                "ERROR: multiplet-energy trace identity failed "
                f"(parameter {p}: {term_sum} vs {total}); please report this case."
            )


# ---------------------------------------------------------------------------
# ground state
# ---------------------------------------------------------------------------


def ground_state(results, reference):
    """(list of lowest TermEnergy, unconditional) at the reference point;
    unconditional means provably lowest for any positive parameters."""
    minima = [min(entry.reference_values(reference)) for entry in results]
    lowest = min(minima)
    winners = [
        entry
        for entry, value in zip(results, minima)
        if value < lowest + 1e-9
    ]
    unconditional = False
    if len(winners) == 1 and winners[0].linear is not None:
        winner = winners[0]
        unconditional = all(
            entry is winner
            or (
                entry.linear is not None
                and all(
                    entry.linear[p] >= winner.linear[p]
                    for p in range(1, len(winner.params))
                )
            )
            for entry in results
        )
    return winners, unconditional


# ---------------------------------------------------------------------------
# coupled-parent CI matrices (for comparison with the textbook tables)
# ---------------------------------------------------------------------------


def _partial_s2_matrix(space, sector, orbital_subset):
    """S^2 of the electrons in a subset of the shell orbitals."""
    subset = set(orbital_subset)
    size = len(sector)
    matrix = np.zeros((size, size))
    position = {space.dets[i]: row for row, i in enumerate(sector)}
    for row, det_index in enumerate(sector):
        det = space.dets[det_index]
        sz = sum(
            (0.5 if so % 2 == 0 else -0.5) for so in det if so // 2 in subset
        )
        matrix[row, row] += sz * sz + sz
        for orb in subset:
            plus = _apply_one_body(det, 2 * orb, 2 * orb + 1)
            if plus is None:
                continue
            sign_p, det_p = plus
            for orb2 in subset:
                minus = _apply_one_body(det_p, 2 * orb2 + 1, 2 * orb2)
                if minus is None:
                    continue
                sign_m, det_m = minus
                col = position.get(det_m)
                if col is not None:
                    matrix[row, col] += sign_p * sign_m
    return matrix


def coupled_parent_matrices(
    character_table: dict,
    l: int,
    shells: list[tuple[str, int]],
    shell_term_lists: list[list],
    terms: list[tuple[Fraction, str, int]],
):
    """CI matrices of the doubly-occurring terms of a two-shell configuration
    in the coupled-parent basis |shell1^n1(S1 Gamma1) shell2^n2(S2 Gamma2)>,
    i.e. the representation used by the Tanabe-Sugano/Griffith strong-field
    tables. Returns {term index: (parent labels, diag1, diag2, offdiag)} with
    the entries as per-parameter Fraction tuples (off-diagonal up to a global
    sign, which is a basis convention)."""
    if len(shells) != 2:
        return {}
    if not any(multiplicity == 2 for _, _, multiplicity in terms):
        return {}

    shell_irreps = [name for name, _ in shells]
    occupations = [count for _, count in shells]
    bases, operations, class_labels, d_matrices, class_index = _shell_bases(
        character_table, l, shell_irreps
    )
    basis = np.hstack(bases)
    shell_dims = [b.shape[1] for b in bases]
    params = _PARAM_NAMES[l]
    n_params = len(params)

    exact = real_two_electron_integrals(l)
    size = 2 * l + 1
    integrals_by_param = []
    for p in range(n_params):
        tensor = np.zeros((size, size, size, size))
        for (a, b, c, d), coeffs in exact.items():
            tensor[a, b, c, d] = float(coeffs[p])
        transformed = np.einsum(
            "abcd,ap,bq,cr,ds->pqrs", tensor, basis, basis, basis, basis
        )
        entries = {}
        m = basis.shape[1]
        for a in range(m):
            for b in range(m):
                for c in range(m):
                    for d in range(m):
                        value = transformed[a, b, c, d]
                        if abs(value) > 1e-12:
                            entries[(a, b, c, d)] = value
        integrals_by_param.append(entries)

    def block_matrices(shell_index):
        matrices = []
        for dmat in d_matrices:
            m = basis.shape[1]
            u = np.eye(m)
            offset = sum(shell_dims[:shell_index])
            k = shell_dims[shell_index]
            u[offset:offset + k, offset:offset + k] = (
                bases[shell_index].T @ dmat @ bases[shell_index]
            )
            matrices.append(u)
        return matrices

    space = _DeterminantSpace(shell_dims, occupations)
    n_electrons = sum(occupations)
    order = len(operations)
    shell_orbitals = [
        list(range(shell_dims[0])),
        list(range(shell_dims[0], shell_dims[0] + shell_dims[1])),
    ]

    results = {}
    sector_cache: dict[int, dict] = {}
    for index, (spin, irrep, multiplicity) in enumerate(terms):
        if multiplicity != 2:
            continue
        sz2 = int(2 * spin)
        if sz2 not in sector_cache:
            sector = space.sector(sz2)
            cache = {
                "sector": sector,
                "h": _coulomb_hamiltonians(space, sector, integrals_by_param),
                "s2": _s2_matrix(space, sector),
            }
            full_mats = []
            for dmat in d_matrices:
                m = basis.shape[1]
                u = np.zeros((m, m))
                offset = 0
                for b in bases:
                    k = b.shape[1]
                    u[offset:offset + k, offset:offset + k] = b.T @ dmat @ b
                    offset += k
                full_mats.append(u)
            cache["group"] = _group_action(space, sector, full_mats)
            cache["g1"] = _group_action(space, sector, block_matrices(0))
            cache["g2"] = _group_action(space, sector, block_matrices(1))
            cache["s2a"] = _partial_s2_matrix(space, sector, shell_orbitals[0])
            cache["s2b"] = _partial_s2_matrix(space, sector, shell_orbitals[1])
            sector_cache[sz2] = cache
        cache = sector_cache[sz2]
        sector = cache["sector"]
        n = len(sector)

        def spin_projector(s2_matrix, target_spin, max_spin):
            projector = np.eye(n)
            target = float(target_spin * (target_spin + 1))
            s_iter = Fraction(0) if int(2 * max_spin) % 2 == 0 else Fraction(1, 2)
            while s_iter <= max_spin:
                other = float(s_iter * (s_iter + 1))
                if abs(other - target) > 1e-9:
                    projector = projector @ (
                        s2_matrix - other * np.eye(n)
                    ) / (target - other)
                s_iter += 1
            return projector

        def pg_projector(group_matrices, target_irrep):
            characters = np.asarray(
                character_table["character_table"][target_irrep], dtype=float
            )
            dim = int(round(characters[class_index["E"]]))
            projector = np.zeros((n, n))
            for label, gmat in zip(class_labels, group_matrices):
                projector += characters[class_index[label]] * gmat
            return projector * dim / order

        # total-term projector and its range
        total = pg_projector(cache["group"], irrep) @ spin_projector(
            cache["s2"], spin, Fraction(n_electrons, 2)
        )
        total = (total + total.T) / 2
        eigenvalues, eigenvectors = np.linalg.eigh(total)
        kept = eigenvectors[:, eigenvalues > 0.5]
        characters = np.asarray(
            character_table["character_table"][irrep], dtype=float
        )
        dim = int(round(characters[class_index["E"]]))
        if kept.shape[1] != 2 * dim:
            continue

        blocks = [kept.T @ h @ kept for h in cache["h"]]

        # parent subspaces inside the kept space
        parents = []
        for s1, g1, _ in shell_term_lists[0]:
            for s2m, g2m, _ in shell_term_lists[1]:
                q = (
                    pg_projector(cache["g1"], g1)
                    @ spin_projector(cache["s2a"], s1,
                                     Fraction(occupations[0], 2))
                    @ pg_projector(cache["g2"], g2m)
                    @ spin_projector(cache["s2b"], s2m,
                                     Fraction(occupations[1], 2))
                )
                m_q = kept.T @ ((q + q.T) / 2) @ kept
                q_eigenvalues, q_eigenvectors = np.linalg.eigh(m_q)
                columns = q_eigenvectors[:, q_eigenvalues > 0.5]
                if columns.shape[1] == dim:
                    parents.append(((s1, g1, s2m, g2m), columns))
        if len(parents) != 2:
            continue

        (label1, c1), (label2, c2) = parents
        diag1 = tuple(
            _snap(float(np.trace(c1.T @ block @ c1)) / dim) for block in blocks
        )
        diag2 = tuple(
            _snap(float(np.trace(c2.T @ block @ c2)) / dim) for block in blocks
        )
        # off-diagonal linear form up to a global sign:
        # tr(M_p M_q^T)/dim = c_p c_q for M_p = c1^T H_p c2
        cross = [c1.T @ block @ c2 for block in blocks]
        gram = np.array([
            [float(np.trace(cross[p] @ cross[q].T)) / dim
             for q in range(n_params)]
            for p in range(n_params)
        ])
        pivot = int(np.argmax(np.abs(np.diag(gram))))
        if gram[pivot, pivot] < 1e-12:
            offdiag = tuple(Fraction(0) for _ in range(n_params))
        else:
            lead = float(np.sqrt(gram[pivot, pivot]))
            offdiag = tuple(
                _snap(gram[pivot, q] / lead) for q in range(n_params)
            )
        results[index] = ((label1, label2), diag1, diag2, offdiag)
    return results
