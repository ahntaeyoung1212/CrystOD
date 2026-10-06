"""Invariant polynomials of space-group irreps (crystod-group --invariants).

The Landau free energy of a distortion that transforms as a space-group irrep
D is a polynomial in the order-parameter components that is invariant under
every operation of the parent group, ``f(D(g) Q) = f(Q)``.  This module
lists a basis of those invariant polynomials degree by degree, the offline
counterpart of the INVARIANTS program of the ISOTROPY Software Suite
(https://iso.byu.edu; D. M. Hatch and H. T. Stokes, J. Appl. Cryst. 36,
951-952 (2003)).

How it works:

- The representation is given as a list of real orthogonal matrices, one per
  element of the finite group ``G / T_N`` (``InducedRepresentation.elements``
  for a space-group irrep), together with the dimensions of its blocks
  ("parts": one part for a single irrep, several for a direct sum).
- For every (multi)degree the invariance condition is linear in the
  coefficients of the monomial basis.  It is evaluated at random points on
  the unit sphere, one generator of the group at a time, and solved as a
  nullspace (SVD, threshold relative to the largest singular value); the
  symmetric power of D is never built explicitly.
- The dimension found is checked against the Molien series, computed from
  the power sums ``tr D(g)^k`` with Newton's identities, averaged over all
  elements.
- The basis is brought to reduced row-echelon form in graded lexicographic
  monomial order (``Q1^n`` first), the coefficients are rationalized where
  possible (fractions, square roots of 2 and 3), and the invariants that are
  products of lower-degree invariants are separated from the new ones.

The order-parameter components refer to the real irrep basis of
``InducedRepresentation`` (the basis of the direction patterns printed by
``crystod-group --parent``).  ISOTROPY uses a different real basis, related
by an orthogonal transformation, so its polynomials can look different while
spanning the same space; the numbers of invariants agree.

Example:
    >>> from crystod import group
    >>> basis = group.invariant_polynomials("Pm-3m", ["R4+"], degree=4)
    >>> basis.counts
    {1: 0, 2: 1, 3: 0, 4: 2}
    >>> [term.expression for term in basis.polynomials[2]]
    ['Q1^2 + Q2^2 + Q3^2']
"""

from __future__ import annotations

import itertools
import math
import re
import warnings
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

# warn above this many monomials in one (multi)degree (SVD cost grows as n^3)
MONOMIAL_WARNING = 5000
# refuse above this many: the dense point matrix alone would need gigabytes
MONOMIAL_LIMIT = 20000

# an exact coefficient printed as a 6-decimal float (no exact form found)
_DECIMAL = re.compile(r"-?\d+\.\d+")

INVARIANTS_CITATION = (
    "Conventions and validation: ISOTROPY INVARIANTS (https://iso.byu.edu):\n"
    "D. M. Hatch and H. T. Stokes, J. Appl. Cryst. 36, 951-952 (2003)."
)


# ------------------------------------------------------------------ data types


@dataclass
class InvariantPolynomial:
    """One invariant polynomial of the canonical basis.

    Attributes:
        name: ``I<degree>_<n>`` for a new invariant of one irrep,
            ``I(<multidegree>)_<n>`` for a direct sum (``I(2,0)_1``), or the
            product written in the names of lower-degree invariants
            (``I2_1^2``).
        degree: Total degree.
        multidegree: Degree in the variables of every part.
        exponents: Monomial exponents, shape ``(n_monomials, n_variables)``,
            in graded lexicographic order (``Q1^n`` first).
        coefficients: Coefficient of every monomial (floats, full
            precision).
        exact: The coefficients as rationalized strings (``"3/2"``,
            ``"sqrt(3)/2"``, ``"-1 + sqrt(3)"``), or a 6-decimal float when
            no exact form fits.  The decimals are display values: they
            occur when the real basis of the irrep (the arbitrary real form
            chosen by ``InducedRepresentation``, e.g. at the W points of
            Fm-3m) makes the coefficients irrational, so they depend on that
            basis and are not exact; ``sympy()`` uses ``coefficients`` for
            them.
        expression: The polynomial as text, e.g. ``"Q1^2 + Q2^2 + Q3^2"``.
    """

    name: str
    degree: int
    multidegree: tuple
    exponents: np.ndarray
    coefficients: np.ndarray
    exact: list = field(default_factory=list)
    expression: str = ""

    def sympy(self, variables: list[str]):
        """The polynomial as a sympy expression.

        Exact coefficients are used where they exist; a coefficient shown
        as a 6-decimal float enters with its full floating-point value.

        Args:
            variables: Variable names, e.g. ``InvariantBasis.variables``
                (names such as ``R4+_1`` are turned into valid symbols by
                ``sympy.Symbol``).

        Returns:
            A ``sympy.Expr``.
        """
        import sympy

        symbols = [sympy.Symbol(name) for name in variables]
        total = 0
        for row, text, value in zip(self.exponents, self.exact, self.coefficients):
            if text == "0":
                continue
            if _DECIMAL.fullmatch(text):
                term = sympy.Float(float(value), 17)
            else:
                term = sympy.sympify(text.replace("^", "**"))
            for symbol, power in zip(symbols, row):
                term *= symbol ** int(power)
            total += term
        return sympy.expand(total)


@dataclass
class InvariantBasis:
    """Invariant polynomials of a representation, degree by degree.

    Attributes:
        variables: Names of the order-parameter components (``Q1``, ...
            for one irrep; ``<label>_<n>`` for a direct sum).
        parts: Dimension of every block of the representation.
        degrees: The total degrees computed (1 .. the requested degree).
        counts: Number of linearly independent invariants of every total
            degree (the Molien count, equal to the nullspace dimension).
        polynomials: For every total degree, the new invariants (not
            products of lower-degree ones) as ``InvariantPolynomial``.
        products: For every total degree, the independent products of
            lower-degree invariants that complete the basis.
        labels: Irrep label of every part (empty for a bare matrix group).
        arm_chunks: Components per star arm of every part (for the
            direction-pattern letters).
        multidegree_counts: Molien count of every multidegree.
    """

    variables: list
    parts: list
    degrees: list
    counts: dict
    polynomials: dict
    products: dict
    labels: list = field(default_factory=list)
    arm_chunks: list = field(default_factory=list)
    multidegree_counts: dict = field(default_factory=dict)


@dataclass
class LandauLifshitz:
    """Landau and Lifshitz conditions of an irrep.

    Attributes:
        n_cubic: Number of cubic invariants (Landau condition: zero).
        n_lifshitz: Multiplicity of the identity in the antisymmetric square
            of D times the polar-vector representation (Lifshitz condition:
            zero).
        continuous_allowed: ``True`` when both conditions hold, so a
            continuous (second-order) transition is allowed by symmetry.
    """

    n_cubic: int
    n_lifshitz: int
    continuous_allowed: bool


@dataclass
class CouplingTerm:
    """Lowest-order coupling term of a direct sum of irreps.

    A coupling term is an invariant whose multidegree has every entry at
    least 1, i.e. one that involves the components of every part.

    Attributes:
        multidegree: Degree in the variables of every part, e.g.
            ``(1, 1, 1)`` for a trilinear term.
        polynomials: Every invariant of that multidegree (products of
            lower-degree invariants first, then the new ones), as
            ``InvariantPolynomial`` records; their number is the Molien
            count of the multidegree.
        trilinear: ``True`` when the multidegree is ``(1, 1, ..., 1)``
            (for three parts, the hybrid-improper term ``Q1 Q2 P``).
    """

    multidegree: tuple
    polynomials: list
    trilinear: bool = False


# ------------------------------------------------------------- group helpers


def _key(matrix: np.ndarray) -> bytes:
    return (np.round(np.asarray(matrix, dtype=float), 5) + 0.0).tobytes()


def _unique_with_counts(matrices) -> list[tuple[np.ndarray, int]]:
    unique: dict[bytes, list] = {}
    for matrix in matrices:
        key = _key(matrix)
        if key in unique:
            unique[key][1] += 1
        else:
            unique[key] = [np.asarray(matrix, dtype=float), 1]
    return [(m, c) for m, c in unique.values()]


def _generating_subset(candidates, group_keys: set) -> list[np.ndarray]:
    """A subset of ``candidates`` that generates the matrix group whose
    elements have the keys ``group_keys`` (greedy closure).  Falls back to
    all distinct candidates when they do not generate the whole group."""
    chosen: list[np.ndarray] = []
    n = np.asarray(candidates[0]).shape[0] if candidates else 0
    closure = {_key(np.eye(n)): np.eye(n)}
    for candidate in candidates:
        if len(closure) >= len(group_keys):
            break
        if _key(candidate) in closure:
            continue
        chosen.append(np.asarray(candidate, dtype=float))
        frontier = list(closure.values())
        while frontier:
            new = []
            for element in frontier:
                for generator in chosen:
                    product = generator @ element
                    key = _key(product)
                    if key not in closure:
                        closure[key] = product
                        new.append(product)
            frontier = new
            if len(closure) > 4 * len(group_keys) + 8:
                break  # not the expected group: keep every candidate
    if len(closure) != len(group_keys) or set(closure) != group_keys:
        distinct = {}
        for candidate in candidates:
            distinct.setdefault(_key(candidate), np.asarray(candidate, dtype=float))
        return list(distinct.values())
    return chosen


# --------------------------------------------------------------- Molien series


def _complete_homogeneous(matrix: np.ndarray, degree: int) -> np.ndarray:
    """h_0..h_degree of the eigenvalues of ``matrix`` (Newton's identities).

    The power sums are the traces ``tr M^k``; no eigendecomposition is
    needed (LAPACK's eigenvalue solver can fail to converge on the
    permutation-like matrices of large induced irreps)."""
    p = [float(matrix.shape[0])]
    power = np.eye(matrix.shape[0])
    for _ in range(degree):
        power = power @ matrix
        p.append(float(np.trace(power)))
    h = [1.0]
    for n in range(1, degree + 1):
        h.append(sum(p[k] * h[n - k] for k in range(1, n + 1)) / n)
    return np.array(h)


def _round_count(value: complex, what: str) -> int:
    count = int(round(value.real))
    if abs(value - count) > 1e-6:
        raise RuntimeError(
            f"the Molien average for {what} is not an integer ({value:.8f})."
        )
    return count


def molien_counts(matrices, parts, degrees) -> dict:
    """Number of linearly independent invariants of given (multi)degrees.

    Args:
        matrices: The real matrices of every group element (one per element
            of the finite group; repeated matrices count with multiplicity).
        parts: Block dimensions of the (block-diagonal) matrices.
        degrees: Total degrees (ints; summed over all multidegrees) and/or
            multidegrees (tuples with one entry per part).

    Returns:
        A dict from every requested degree to the number of invariants.

    Raises:
        RuntimeError: The group average is not an integer (the matrices do
            not form a group, or the parts are wrong).

    Example:
        >>> import numpy as np
        >>> mirror = [np.eye(1), -np.eye(1)]
        >>> molien_counts(mirror, [1], [1, 2, 3, 4])
        {1: 0, 2: 1, 3: 0, 4: 1}
    """
    parts = [int(p) for p in parts]
    degrees = list(degrees)
    max_degree = 0
    for degree in degrees:
        max_degree = max(max_degree, sum(degree) if isinstance(degree, tuple) else int(degree))
    offsets = np.cumsum([0] + parts)
    unique = _unique_with_counts(matrices)
    order = sum(count for _, count in unique)
    # h_n of every part (and of the whole matrix) for every unique matrix
    per_part, whole = [], []
    for matrix, count in unique:
        per_part.append([
            _complete_homogeneous(
                matrix[offsets[j]:offsets[j + 1], offsets[j]:offsets[j + 1]],
                max_degree,
            )
            for j in range(len(parts))
        ])
        whole.append(_complete_homogeneous(matrix, max_degree))
    weights = np.array([count for _, count in unique], dtype=float)
    result = {}
    for degree in degrees:
        if isinstance(degree, tuple):
            values = np.array([
                np.prod([h[j][n] for j, n in enumerate(degree)]) for h in per_part
            ])
        else:
            values = np.array([h[int(degree)] for h in whole])
        result[degree] = _round_count(np.sum(weights * values) / order, f"degree {degree}")
    return result


# ------------------------------------------------------------------ monomials


def _compositions(total: int, length: int):
    """Exponent tuples of the given total, lexicographically descending."""
    if length == 1:
        yield (total,)
        return
    for first in range(total, -1, -1):
        for rest in _compositions(total - first, length - 1):
            yield (first,) + rest


def monomial_exponents(parts, multidegree) -> np.ndarray:
    """Monomial basis of a multidegree in graded lexicographic order.

    Args:
        parts: Block dimensions.
        multidegree: Degree in the variables of every part.

    Returns:
        Integer array ``(n_monomials, sum(parts))``; the first row is
        ``Q1^n`` (of the first part with nonzero degree).
    """
    per_part = [list(_compositions(int(n), int(d))) for d, n in zip(parts, multidegree)]
    rows = [sum(combo, ()) for combo in itertools.product(*per_part)]
    rows.sort(reverse=True)
    return np.array(rows, dtype=np.int64).reshape(len(rows), sum(int(p) for p in parts))


def _evaluate(points: np.ndarray, exponents: np.ndarray) -> np.ndarray:
    """Monomial values, shape ``(n_points, n_monomials)``."""
    max_power = int(exponents.max()) if exponents.size else 0
    powers = np.ones((max_power + 1,) + points.shape)
    for k in range(1, max_power + 1):
        powers[k] = powers[k - 1] * points
    values = np.ones((exponents.shape[0], points.shape[0]))
    for v in range(points.shape[1]):
        column = exponents[:, v]
        if np.any(column):
            values *= powers[column, :, v]
    return values.T


def _nullspace(matrix: np.ndarray, reference: float) -> np.ndarray:
    """Orthonormal nullspace basis (columns).

    The singular-value threshold is relative: ``1e-8`` times the largest
    singular value, or times ``reference`` (the scale of the monomial
    values) when the matrix vanishes numerically (a generator that fixes
    every monomial of the degree)."""
    if matrix.shape[1] == 0:
        return np.zeros((0, 0))
    _, sing, Vh = np.linalg.svd(matrix, full_matrices=True)
    largest = float(sing[0]) if sing.size else 0.0
    rank = int(np.sum(sing > 1e-8 * max(largest, reference)))
    return Vh[rank:].T


def _invariant_space(generators, exponents, n_vars, expected, rng_seed) -> np.ndarray:
    """Nullspace of the invariance conditions (columns = invariants)."""
    n_mono = exponents.shape[0]
    n_pts = 2 * n_mono + 10
    found = None
    for attempt in range(4):
        rng = np.random.default_rng(rng_seed + attempt)
        points = rng.normal(size=(n_pts, n_vars))
        points /= np.linalg.norm(points, axis=1)[:, None]
        base = _evaluate(points, exponents)
        reference = float(np.linalg.norm(base, 2))
        B = np.eye(n_mono)
        for generator in generators:
            moved = _evaluate(points @ np.asarray(generator).T, exponents)
            A = (moved - base) @ B
            B = B @ _nullspace(A, reference)
            if B.shape[1] == 0:
                break
        found = B.shape[1]
        if found == expected:
            # orthonormalize the columns (products of orthonormal bases are
            # orthonormal, but keep it exact)
            if found:
                B, _ = np.linalg.qr(B)
            return B
        n_pts *= 2
    raise RuntimeError(
        f"the invariant nullspace has dimension {found}, but the Molien series "
        f"gives {expected} (monomials {n_mono})."
    )


# ------------------------------------------------------- canonical form, text


def _rref(rows: np.ndarray, tol: float = 1e-7) -> tuple[np.ndarray, list[int]]:
    """Reduced row-echelon form (partial pivoting along the columns)."""
    A = np.array(rows, dtype=float, copy=True)
    if A.size == 0:
        return A.reshape(0, rows.shape[1] if rows.ndim == 2 else 0), []
    scale = max(1.0, float(np.max(np.abs(A))))
    pivots: list[int] = []
    r = 0
    n_rows, n_cols = A.shape
    for c in range(n_cols):
        if r >= n_rows:
            break
        p = r + int(np.argmax(np.abs(A[r:, c])))
        if abs(A[p, c]) <= tol * scale:
            continue
        A[[r, p]] = A[[p, r]]
        A[r] /= A[r, c]
        for other in range(n_rows):
            if other != r and A[other, c] != 0:
                A[other] -= A[other, c] * A[r]
        A[r, np.abs(A[r]) < 1e-12] = 0.0
        pivots.append(c)
        r += 1
    return A[:r], pivots


# Tolerance of the exact forms.  The nullspace coefficients are accurate to
# ~1e-13; a looser tolerance (1e-6) would let limit_denominator(1000) "find"
# a fraction for most irrational numbers (the fractions with denominators up
# to 1000 are spaced ~3e-6 apart), e.g. 818/259 for 3.15830076.
_EXACT_TOLERANCE = 1e-9


def _rationalize(value: float) -> str:
    """Exact text of a coefficient: a fraction, a fraction times sqrt(2),
    sqrt(3) or sqrt(6), a short surd found by ``sympy.nsimplify``, or a
    6-decimal float."""
    if abs(value) < 1e-9:
        return "0"
    frac = Fraction(value).limit_denominator(1000)
    if abs(value - float(frac)) < _EXACT_TOLERANCE:
        return str(frac)
    for radicand in (2, 3, 6):
        root = float(np.sqrt(radicand))
        frac = Fraction(value / root).limit_denominator(1000)
        if abs(value - float(frac) * root) < _EXACT_TOLERANCE:
            sign = "-" if frac < 0 else ""
            frac = abs(frac)
            text = f"sqrt({radicand})"
            if frac.numerator != 1:
                text = f"{frac.numerator}*{text}"
            if frac.denominator != 1:
                text = f"{text}/{frac.denominator}"
            return sign + text
    import sympy

    try:
        guess = sympy.nsimplify(
            value, [sympy.sqrt(2), sympy.sqrt(3)], tolerance=_EXACT_TOLERANCE
        )
        if (
            not guess.is_Float
            and abs(float(guess) - value) < _EXACT_TOLERANCE
            and sympy.count_ops(guess) <= 6
        ):
            return str(guess).replace("**", "^")
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    return f"{value:.6f}"


def _monomial_text(row, variables) -> str:
    factors = []
    for name, power in zip(variables, row):
        if power == 1:
            factors.append(name)
        elif power > 1:
            factors.append(f"{name}^{int(power)}")
    return "*".join(factors) if factors else "1"


def _signed_magnitude(coefficient: str) -> tuple[bool, str]:
    """Sign and printed magnitude of an exact coefficient.

    A compound coefficient (a sum such as ``-1 + sqrt(3)``) is one signed
    unit: its sign is the sign of its value, and the magnitude is the
    parenthesized expression, never the text with the first ``-`` cut off.
    """
    negative = coefficient.startswith("-")
    magnitude = coefficient[1:] if negative else coefficient
    if any(op in magnitude for op in "+-"):
        import sympy

        expr = sympy.sympify(coefficient.replace("^", "**"))
        negative = bool(float(expr) < 0)
        magnitude = f"({_positive_first(sympy.expand(-expr if negative else expr))})"
    return negative, magnitude


def _positive_first(expr) -> str:
    """A sum as text with its positive terms first, rational terms first
    within each sign (``2*sqrt(3) - 3``, not sympy's ``-3 + 2*sqrt(3)``;
    ``6 + 2*sqrt(3)``); ``^`` powers."""
    import sympy

    terms = sorted(
        sympy.Add.make_args(expr),
        key=lambda term: (term.could_extract_minus_sign(), not term.is_Number, str(term)),
    )
    text = ""
    for term in terms:
        negative = term.could_extract_minus_sign()
        body = str(-term if negative else term).replace("**", "^")
        if not text:
            text = f"-{body}" if negative else body
        else:
            text += f" - {body}" if negative else f" + {body}"
    return text


def _polynomial_text(exact: list[str], exponents: np.ndarray, variables) -> str:
    """The polynomial ``sum exact[j] * monomial[j]`` as text (``^`` powers,
    ``*`` products, compound coefficients in parentheses)."""
    text = ""
    for coefficient, row in zip(exact, exponents):
        if coefficient == "0":
            continue
        negative, magnitude = _signed_magnitude(coefficient)
        monomial = _monomial_text(row, variables)
        term = monomial if magnitude == "1" else f"{magnitude}*{monomial}"
        if not text:
            text = f"-{term}" if negative else term
        else:
            text += f" - {term}" if negative else f" + {term}"
    return text or "0"


def _multiply(a: dict, b: dict) -> dict:
    out: dict = {}
    for ea, ca in a.items():
        for eb, cb in b.items():
            key = tuple(x + y for x, y in zip(ea, eb))
            out[key] = out.get(key, 0.0) + ca * cb
    return out


def _as_dict(poly: InvariantPolynomial) -> dict:
    return {
        tuple(int(x) for x in row): float(c)
        for row, c in zip(poly.exponents, poly.coefficients)
        if c != 0
    }


def _product_name(factors: list[InvariantPolynomial]) -> str:
    names, counts = [], {}
    for factor in factors:
        if factor.name not in counts:
            names.append(factor.name)
        counts[factor.name] = counts.get(factor.name, 0) + 1
    return "*".join(name if counts[name] == 1 else f"{name}^{counts[name]}" for name in names)


def _multidegrees(n_parts: int, total: int):
    return list(_compositions(total, n_parts))


def monomial_count(parts, degree: int) -> int:
    """Largest number of monomials in one multidegree of total ``degree``.

    This is the size of the largest linear problem the engine solves for
    that degree (the monomials of one part number
    ``comb(d + n - 1, n)``).

    Args:
        parts: Block dimensions.
        degree: Total degree.

    Returns:
        The largest monomial count over the multidegrees of total
        ``degree``.

    Example:
        >>> monomial_count([12], 6)
        12376
    """
    parts = [int(p) for p in parts]
    return max(
        math.prod(math.comb(d + n - 1, n) for d, n in zip(parts, multidegree))
        for multidegree in _multidegrees(len(parts), int(degree))
    )


def check_monomial_count(parts, degree: int) -> int:
    """Refuse a degree whose linear problems are too large.

    Args:
        parts: Block dimensions.
        degree: Highest total degree requested.

    Returns:
        The largest monomial count (see ``monomial_count``).

    Raises:
        ValueError: More than ``MONOMIAL_LIMIT`` monomials in one
            multidegree (the message names the count and the degree).
    """
    count = max(monomial_count(parts, n) for n in range(1, int(degree) + 1))
    if count > MONOMIAL_LIMIT:
        raise ValueError(
            f"degree {degree} needs {count} monomials of {sum(parts)} components "
            f"in one (multi)degree, more than the limit of {MONOMIAL_LIMIT}; "
            "lower the degree."
        )
    return count


def _invariant_name(multidegree: tuple, serial: int) -> str:
    if len(multidegree) == 1:
        return f"I{multidegree[0]}_{serial}"
    return f"I({','.join(str(n) for n in multidegree)})_{serial}"


def invariant_space(matrices, parts, multidegree, generators, rng_seed: int = 0):
    """Invariants of one multidegree as coefficient vectors.

    The low-level step of ``invariants_of_matrices``, for callers that need
    a single multidegree (coupling terms, secondary order parameters).

    Args:
        matrices: Real matrices of every group element (with multiplicity).
        parts: Block dimensions.
        multidegree: Degree in the variables of every part.
        generators: Matrices that generate the group (used as given).
        rng_seed: Seed of the random evaluation points.

    Returns:
        ``(exponents, space)``: the monomial exponents of the multidegree
        and an orthonormal basis of the invariant coefficient vectors as
        columns (``(n_monomials, count)``; ``count`` is the Molien count).

    Raises:
        ValueError: Too many monomials (``MONOMIAL_LIMIT``).
        RuntimeError: The nullspace disagrees with the Molien series.
    """
    parts = [int(p) for p in parts]
    multidegree = tuple(int(n) for n in multidegree)
    exponents = monomial_exponents(parts, multidegree)
    if exponents.shape[0] > MONOMIAL_LIMIT:
        raise ValueError(
            f"multidegree {multidegree} needs {exponents.shape[0]} monomials, "
            f"more than the limit of {MONOMIAL_LIMIT}; lower the degree."
        )
    expected = molien_counts(matrices, parts, [multidegree])[multidegree]
    if expected == 0:
        return exponents, np.zeros((exponents.shape[0], 0))
    space = _invariant_space(
        [np.asarray(g, dtype=float) for g in generators],
        exponents, sum(parts), expected, rng_seed,
    )
    return exponents, space


# ------------------------------------------------------------------- engine


def invariants_of_matrices(
    matrices,
    parts,
    degree: int,
    generators=None,
    rng_seed: int = 0,
    variables: list[str] | None = None,
) -> InvariantBasis:
    """Invariant polynomials of a finite real matrix group.

    Args:
        matrices: Real orthogonal matrices, one per group element (they
            must form a group; repeated matrices are allowed).
        parts: Block dimensions of the block-diagonal matrices (``[d]`` for
            an irreducible representation).
        degree: Highest total degree to compute.
        generators: Matrices generating the group; by default a generating
            subset of ``matrices`` is chosen.
        rng_seed: Seed of the random evaluation points.
        variables: Variable names (default ``Q1``, ``Q2``, ...).

    Returns:
        An ``InvariantBasis`` with the degrees 1 .. ``degree``.

    Raises:
        RuntimeError: The nullspace dimension disagrees with the Molien
            series even after enlarging the point set (the generators do
            not generate the group, or the problem is ill conditioned).
        ValueError: ``degree`` is smaller than 1, the parts do not add up
            to the matrix size, or one multidegree has more than
            ``MONOMIAL_LIMIT`` monomials (checked before any work).

    Warns:
        UserWarning: More than ``MONOMIAL_WARNING`` monomials in one
            multidegree (the computation may take long).

    Example:
        >>> import numpy as np
        >>> c4 = np.array([[0.0, -1.0], [1.0, 0.0]])
        >>> group = [np.linalg.matrix_power(c4, k) for k in range(4)]
        >>> basis = invariants_of_matrices(group, [2], 4)
        >>> basis.counts
        {1: 0, 2: 1, 3: 0, 4: 3}
    """
    parts = [int(p) for p in parts]
    n_vars = sum(parts)
    if degree < 1:
        raise ValueError("the degree must be at least 1.")
    largest = check_monomial_count(parts, degree)
    if largest > MONOMIAL_WARNING:
        warnings.warn(
            f"{largest} monomials in one (multi)degree at degree <= {degree}; "
            "the computation may take long.",
            stacklevel=2,
        )
    matrices = [np.asarray(m, dtype=float) for m in matrices]
    if not matrices or matrices[0].shape != (n_vars, n_vars):
        raise ValueError("the parts do not add up to the matrix dimension.")
    if variables is None:
        variables = [f"Q{n + 1}" for n in range(n_vars)]
    group_keys = {_key(m) for m in matrices}
    if generators is None:
        generators = _generating_subset(matrices, group_keys)
    else:
        generators = _generating_subset([np.asarray(g, dtype=float) for g in generators],
                                        group_keys)

    degrees = list(range(1, degree + 1))
    all_multidegrees = [m for n in degrees for m in _multidegrees(len(parts), n)]
    multidegree_counts = molien_counts(matrices, parts, all_multidegrees)

    counts: dict[int, int] = {}
    polynomials: dict[int, list] = {}
    products: dict[int, list] = {}
    new_by_multidegree: dict[tuple, list] = {}
    for n in degrees:
        counts[n] = 0
        polynomials[n] = []
        products[n] = []
        serial = 0
        for multidegree in _multidegrees(len(parts), n):
            expected = multidegree_counts[multidegree]
            counts[n] += expected
            if expected == 0:
                new_by_multidegree[multidegree] = []
                continue
            if len(parts) > 1:
                serial = 0  # direct sums: I(2,0)_1, numbered per multidegree
            exponents = monomial_exponents(parts, multidegree)
            space = _invariant_space(generators, exponents, n_vars, expected, rng_seed)
            column = {tuple(int(x) for x in row): j for j, row in enumerate(exponents)}

            # products of lower-degree new invariants with this multidegree
            product_rows, product_names = [], []
            for factors in _factorizations(multidegree, new_by_multidegree):
                poly = _as_dict(factors[0])
                for factor in factors[1:]:
                    poly = _multiply(poly, _as_dict(factor))
                vector = np.zeros(exponents.shape[0])
                for key, value in poly.items():
                    vector[column[key]] += value
                product_rows.append(vector)
                product_names.append(_product_name(factors))
            accepted, accepted_rows = [], []
            basis_q = np.zeros((exponents.shape[0], 0))
            for name, vector in zip(product_names, product_rows):
                residual = vector - basis_q @ (basis_q.T @ vector)
                if np.linalg.norm(residual) > 1e-6 * max(1.0, np.linalg.norm(vector)):
                    basis_q = np.column_stack([basis_q, residual / np.linalg.norm(residual)])
                    accepted.append(name)
                    accepted_rows.append(vector)
            # the new invariants: the invariant space reduced modulo the products
            rows, _ = _rref(space.T)
            if accepted_rows:
                p_rows, p_pivots = _rref(np.array(accepted_rows))
                for r in range(rows.shape[0]):
                    for p_row, pivot in zip(p_rows, p_pivots):
                        rows[r] -= rows[r, pivot] * p_row
                rows, _ = _rref(rows)
            n_new = expected - len(accepted)
            if rows.shape[0] != n_new:
                raise RuntimeError(
                    f"degree {n}: {expected} invariants, {len(accepted)} products, "
                    f"but {rows.shape[0]} new ones after the reduction."
                )
            for name, vector in zip(accepted, accepted_rows):
                exact = [_rationalize(v) for v in vector]
                products[n].append(InvariantPolynomial(
                    name=name, degree=n, multidegree=multidegree,
                    exponents=exponents, coefficients=vector, exact=exact,
                    expression=_polynomial_text(exact, exponents, variables),
                ))
            new = []
            for row in rows:
                serial += 1
                exact = [_rationalize(v) for v in row]
                new.append(InvariantPolynomial(
                    name=_invariant_name(multidegree, serial), degree=n,
                    multidegree=multidegree,
                    exponents=exponents, coefficients=row, exact=exact,
                    expression=_polynomial_text(exact, exponents, variables),
                ))
            new_by_multidegree[multidegree] = new
            polynomials[n].extend(new)
    return InvariantBasis(
        variables=list(variables),
        parts=parts,
        degrees=degrees,
        counts=counts,
        polynomials=polynomials,
        products=products,
        multidegree_counts=multidegree_counts,
    )


def _factorizations(multidegree: tuple, new_by_multidegree: dict):
    """Multisets of (at least two) lower-degree new invariants whose
    multidegrees add up to ``multidegree``, in the order of the generator
    list (degree, then serial number)."""
    generators = [
        poly
        for md in sorted(new_by_multidegree, key=lambda m: (sum(m), tuple(-x for x in m)))
        for poly in new_by_multidegree[md]
    ]
    target = tuple(multidegree)
    found = []

    def extend(start: int, remaining: tuple, chosen: list):
        if not any(remaining):
            if len(chosen) >= 2:
                found.append(list(chosen))
            return
        for index in range(start, len(generators)):
            poly = generators[index]
            md = poly.multidegree
            if all(r >= m for r, m in zip(remaining, md)) and sum(md) < sum(target):
                chosen.append(poly)
                extend(index, tuple(r - m for r, m in zip(remaining, md)), chosen)
                chosen.pop()

    extend(0, target, [])
    return found


# ------------------------------------------------------------ space groups


def _representation(space_group, irreps):
    from .isotropy_subgroup import CoupledRepresentation, InducedRepresentation
    from .spacegroup_product import SpaceGroupIrrepAlgebra

    if isinstance(irreps, str):
        irreps = [irreps]
    irreps = list(irreps)
    if not irreps:
        raise SystemExit("ERROR: give at least one irrep label.")
    algebra = SpaceGroupIrrepAlgebra(str(space_group))
    if len(irreps) == 1:
        representation = InducedRepresentation(algebra, irreps[0])
        parts = [representation]
    else:
        representation = CoupledRepresentation(algebra, irreps)
        parts = representation.parts
    return algebra, representation, parts


def _generator_candidates(algebra, representation) -> list[np.ndarray]:
    """The generators (id, e_j) and (i, 0) of G / T_N, translations first."""
    identity = algebra._rotation_index[algebra._key(np.eye(3))]
    lookup = {
        (i, tuple(int(x) for x in t)): matrix for i, t, matrix in representation.elements
    }
    candidates = []
    if representation.grid_n > 1:
        for e in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            candidates.append(lookup[(identity, e)])
    candidates += [matrix for i, t, matrix in representation.elements if not np.any(t)]
    return candidates


def invariant_polynomials(space_group, irreps, degree: int = 4) -> InvariantBasis:
    """Invariant polynomials of one space-group irrep or of a direct sum.

    Args:
        space_group: Space-group number or international symbol
            (``"Pm-3m"``, ``221``).
        irreps: ISO-IR label(s): one label (``"R4+"``) or a list; several
            labels give the invariants of the direct sum, with variables
            ``<label>_<n>``.
        degree: Highest total degree (default 4).

    Returns:
        An ``InvariantBasis``.  For one irrep the variables ``Q1 .. Qd``
        follow the real basis of ``InducedRepresentation`` (arm by arm;
        doubled complex-type irreps: real rows, then imaginary rows of each
        arm), the order of the direction patterns of ``--parent``.

    Raises:
        SystemExit: Unknown space group or irrep (``ValueError`` through
            ``crystod.group``).
        RuntimeError: The nullspace disagrees with the Molien series.

    Example:
        >>> from crystod import group
        >>> basis = group.invariant_polynomials("Pm-3m", "M1+", degree=3)
        >>> basis.polynomials[3][0].expression
        'Q1*Q2*Q3'
    """
    algebra, representation, parts = _representation(space_group, irreps)
    return invariants_of_representation(algebra, representation, degree)


def invariants_of_representation(algebra, representation, degree: int = 4) -> InvariantBasis:
    """Invariant polynomials of an already-built space-group representation.

    Args:
        algebra: The ``SpaceGroupIrrepAlgebra`` the representation was built
            from.
        representation: An ``InducedRepresentation`` or a
            ``CoupledRepresentation``.
        degree: Highest total degree.

    Returns:
        An ``InvariantBasis`` (see ``invariant_polynomials``).

    Raises:
        RuntimeError: The nullspace disagrees with the Molien series.
    """
    parts = getattr(representation, "parts", None) or [representation]
    if len(parts) == 1:
        variables = [f"Q{n + 1}" for n in range(representation.dimension)]
    else:
        variables = [
            f"{part.label}_{n + 1}" for part in parts for n in range(part.dimension)
        ]
    matrices = [matrix for _, _, matrix in representation.elements]
    basis = invariants_of_matrices(
        matrices,
        [part.dimension for part in parts],
        degree,
        generators=_generator_candidates(algebra, representation),
        variables=variables,
    )
    basis.labels = [part.label for part in parts]
    basis.arm_chunks = [list(part.arm_chunks) for part in parts]
    return basis


def landau_lifshitz_of_elements(elements, rotations) -> LandauLifshitz:
    """Landau and Lifshitz counts from the group elements.

    Args:
        elements: ``(i, t, matrix)`` triples of the (physically irreducible,
            real) representation, one per element of ``G / T_N``.
        rotations: Rotation matrices of the coset representatives
            (``algebra.rotations``); only their traces are used.

    Returns:
        A ``LandauLifshitz`` record.
    """
    traces_w = [float(np.trace(rotations[i])) for i in range(len(rotations))]
    cubic_sum = 0.0 + 0j
    lifshitz_sum = 0.0
    seen: dict[tuple, tuple] = {}
    for i, _, matrix in elements:
        key = (int(i), _key(matrix))
        if key not in seen:
            chi = float(np.trace(matrix))
            chi2 = float(np.trace(matrix @ matrix))
            h3 = _complete_homogeneous(matrix, 3)[3]
            seen[key] = (h3, (chi * chi - chi2) / 2.0)
        h3, antisym = seen[key]
        cubic_sum += h3
        lifshitz_sum += antisym * traces_w[int(i)]
    order = len(elements)
    n_cubic = _round_count(cubic_sum / order, "the cubic invariants")
    n_lifshitz = _round_count(complex(lifshitz_sum / order), "the Lifshitz invariants")
    return LandauLifshitz(n_cubic, n_lifshitz, n_cubic == 0 and n_lifshitz == 0)


def landau_lifshitz(space_group, irrep: str) -> LandauLifshitz:
    """Landau (no cubic invariant) and Lifshitz conditions of an irrep.

    Args:
        space_group: Space-group number or international symbol.
        irrep: ISO-IR irrep label (the physically irreducible, possibly
            doubled real representation is used).

    Returns:
        A ``LandauLifshitz`` record; ``continuous_allowed`` is ``True`` when
        neither a cubic nor a Lifshitz invariant exists.

    Example:
        >>> from crystod import group
        >>> group.landau_lifshitz("Pm-3m", "R4+")
        LandauLifshitz(n_cubic=0, n_lifshitz=0, continuous_allowed=True)
    """
    algebra, representation, _ = _representation(space_group, [irrep])
    return landau_lifshitz_of_elements(representation.elements, algebra.rotations)


def lowest_coupling_terms(basis: InvariantBasis) -> list[CouplingTerm]:
    """Coupling terms of the lowest total degree of a direct sum.

    Args:
        basis: An ``InvariantBasis`` of a direct sum (two or more parts).

    Returns:
        One ``CouplingTerm`` per multidegree with every entry >= 1 and a
        nonzero count at the lowest such total degree, in the printed order
        (descending lexicographic); an empty list when there is none up to
        the computed degree or the basis has a single part.
    """
    if len(basis.parts) < 2:
        return []
    trilinear = (1,) * len(basis.parts)
    for n in basis.degrees:
        found = []
        for multidegree in _multidegrees(len(basis.parts), n):
            if min(multidegree) < 1 or basis.multidegree_counts.get(multidegree, 0) == 0:
                continue
            polys = [
                poly
                for poly in basis.products[n] + basis.polynomials[n]
                if tuple(poly.multidegree) == multidegree
            ]
            found.append(CouplingTerm(multidegree, polys, multidegree == trilinear))
        if found:
            return found
    return []


def coupling_terms(space_group, irreps, max_degree: int = 4) -> CouplingTerm | None:
    """Lowest-order coupling term of a direct sum of irreps.

    The invariants of the direct sum are computed up to total degree
    ``max_degree`` and the first multidegree with every entry at least 1
    (lowest total degree, then descending lexicographic order) is
    returned.  For the hybrid improper mechanism, give two order
    parameters and a polar Gamma irrep: a trilinear term ``Q1 Q2 P`` has
    the multidegree ``(1, 1, 1)``.

    Args:
        space_group: Space-group number or international symbol.
        irreps: Two or more ISO-IR labels, e.g. ``["X2+", "X3-", "GM5-"]``.
        max_degree: Highest total degree searched (default 4).

    Returns:
        A ``CouplingTerm``, or ``None`` when no coupling term exists up to
        ``max_degree``.

    Raises:
        SystemExit: Fewer than two labels, or an unknown space group or
            irrep (``ValueError`` through ``crystod.group``).

    Example:
        >>> from crystod import group
        >>> term = group.coupling_terms("I4/mmm", ["X2+", "X3-", "GM5-"], 3)
        >>> term.multidegree, len(term.polynomials), term.trilinear
        ((1, 1, 1), 1, True)
    """
    if isinstance(irreps, str) or len(list(irreps)) < 2:
        raise SystemExit("ERROR: a coupling term needs at least two irreps.")
    basis = invariant_polynomials(space_group, list(irreps), max_degree)
    terms = lowest_coupling_terms(basis)
    return terms[0] if terms else None


def gamma_tensor_multiplicities(elements, rotations) -> tuple[int, int]:
    """Multiplicities of a Gamma irrep in the polar vector and in the strain.

    The polar-vector representation has the character ``tr W``, the strain
    (symmetric second-rank tensor) ``((tr W)^2 + tr W^2) / 2``.  A doubled
    (physically irreducible) form counts once per real copy.

    Args:
        elements: ``(i, t, matrix)`` triples of a Gamma-point
            representation (one matrix per coset representative ``i``; the
            translations act trivially).
        rotations: Rotation matrices of the coset representatives
            (``algebra.rotations``).

    Returns:
        ``(n_polar, n_strain)``.
    """
    by_i: dict[int, np.ndarray] = {}
    for i, _, matrix in elements:
        by_i.setdefault(int(i), matrix)
    n_ops = len(rotations)
    chi = np.array([np.trace(by_i[i]) for i in range(n_ops)])
    traces = np.array([np.trace(w) for w in rotations], dtype=float)
    traces2 = np.array([np.trace(w @ w) for w in rotations], dtype=float)
    norm = float(chi @ chi) / n_ops  # 1 for a real irrep, 2 for a doubled one
    polar = float(chi @ traces) / n_ops / norm
    strain = float(chi @ ((traces ** 2 + traces2) / 2.0)) / n_ops / norm
    return int(round(polar)), int(round(strain))


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def landau_lifshitz_lines(record: LandauLifshitz) -> list[str]:
    """The Landau, Lifshitz and verdict lines of the ``* Irrep *`` block.

    Args:
        record: A ``LandauLifshitz`` record.

    Returns:
        Three lines, e.g. ``Landau condition: satisfied (no cubic
        invariant)``, ``Lifshitz condition: satisfied (no Lifshitz
        invariant)`` and ``-> a continuous transition is allowed``.
    """
    landau = (
        "satisfied (no cubic invariant)" if record.n_cubic == 0
        else f"violated ({_plural(record.n_cubic, 'cubic invariant')})"
    )
    lifshitz = (
        "satisfied (no Lifshitz invariant)" if record.n_lifshitz == 0
        else f"violated ({_plural(record.n_lifshitz, 'Lifshitz invariant')})"
    )
    reasons = [
        name for name, count in (("Landau", record.n_cubic), ("Lifshitz", record.n_lifshitz))
        if count
    ]
    verdict = (
        "-> a continuous transition is allowed" if not reasons
        else f"-> a continuous transition is forbidden ({' and '.join(reasons)})"
    )
    return [f"Landau condition: {landau}", f"Lifshitz condition: {lifshitz}", verdict]


def landau_lifshitz_note(label: str, record: LandauLifshitz) -> str:
    """The one-line form used under the ``--kpoint`` table (after ``note: ``).

    Args:
        label: Irrep label.
        record: Its ``LandauLifshitz`` record.

    Returns:
        E.g. ``R4+: Landau condition satisfied, Lifshitz condition satisfied
        (continuous transition allowed)``.
    """
    landau = (
        "Landau condition satisfied" if record.n_cubic == 0
        else f"Landau condition violated ({_plural(record.n_cubic, 'cubic invariant')})"
    )
    lifshitz = (
        "Lifshitz condition satisfied" if record.n_lifshitz == 0
        else "Lifshitz condition violated "
        f"({_plural(record.n_lifshitz, 'Lifshitz invariant')})"
    )
    verdict = "allowed" if record.continuous_allowed else "forbidden"
    return f"{label}: {landau}, {lifshitz} (continuous transition {verdict})"


# ------------------------------------------------- restriction to a direction


def _component_expressions(tokens: list[str]):
    """sympy expressions of ``--order-parameter`` tokens (``a``, ``-a``,
    ``2a``, ``1/2a``, ``0``, ``1/2``) and the parameter symbols in order of
    appearance."""
    import sympy

    from .isotropy_subgroup import _SCALED_PARAMETER

    expressions, symbols = [], {}
    for token in tokens:
        token = token.strip()
        sign = 1
        body = token
        if body.startswith("-"):
            sign, body = -1, body[1:]
        if body in ("", "0", "0.0"):
            expressions.append(sympy.Integer(0))
            continue
        try:
            expressions.append(sign * sympy.Rational(str(Fraction(body))))
            continue
        except ValueError:
            pass
        factor = sympy.Integer(1)
        scaled = _SCALED_PARAMETER.fullmatch(body)
        if scaled is not None:
            factor = sympy.Rational(str(Fraction(scaled.group(1))))
            body = scaled.group(2)
        if body not in symbols:
            symbols[body] = sympy.Symbol(body)
        expressions.append(sign * factor * symbols[body])
    return expressions, list(symbols.values())


def restricted_invariants(basis: InvariantBasis, tokens: list[str]) -> dict:
    """The invariants restricted to an order-parameter direction.

    Every invariant (products of lower-degree invariants included) is
    evaluated on the direction pattern, e.g. ``(a, a, 0)``; per degree the
    restricted polynomials that vanish or are linearly dependent are
    dropped, and the rest is brought to reduced row-echelon form in the
    monomials of the parameters (so ``(a, b, 0)`` of R4+ gives
    ``a^4 + b^4`` and ``a^2*b^2`` at degree 4).

    Args:
        basis: An ``InvariantBasis`` (one irrep or a direct sum).
        tokens: One ``--order-parameter`` token per component, in the order
            of ``basis.variables`` (letters are parameters; ``-a``, ``2a``,
            ``1/2a`` and numbers are accepted).

    Returns:
        A dict from every degree with a surviving invariant to the list of
        independent restricted polynomials as text (``"a^2 + b^2"``).

    Raises:
        SystemExit: The number of tokens differs from the number of
            components.

    Example:
        >>> from crystod import group
        >>> basis = group.invariant_polynomials("Pm-3m", "R4+", 4)
        >>> restricted_invariants(basis, ["a", "a", "a"])
        {2: ['a^2'], 4: ['a^4']}
    """
    import sympy

    if len(tokens) != len(basis.variables):
        raise SystemExit(
            f"ERROR: the direction needs {len(basis.variables)} components "
            f"(got {len(tokens)})."
        )
    expressions, parameters = _component_expressions(tokens)
    names = [str(p) for p in parameters]
    variables = [sympy.Symbol(name) for name in basis.variables]
    substitution = dict(zip(variables, expressions))
    result: dict[int, list[str]] = {}
    for n in basis.degrees:
        rows: list[dict] = []
        for poly in basis.products[n] + basis.polynomials[n]:
            restricted = sympy.expand(poly.sympy(basis.variables).subs(substitution))
            if restricted == 0:
                continue
            if parameters:
                terms = sympy.Poly(restricted, *parameters).terms()
            else:
                terms = [((), restricted)]
            rows.append({tuple(int(e) for e in mono): float(coeff) for mono, coeff in terms})
        if not rows:
            continue
        monomials = sorted({m for row in rows for m in row}, reverse=True)
        matrix = np.array([[row.get(m, 0.0) for m in monomials] for row in rows])
        reduced, _ = _rref(matrix)
        texts = []
        exponents = np.array(monomials, dtype=np.int64).reshape(len(monomials), len(names))
        for row in reduced:
            if np.max(np.abs(row)) < 1e-9:
                continue
            exact = [_rationalize(v) for v in row]
            texts.append(_polynomial_text(exact, exponents, names))
        if texts:
            result[n] = texts
    return result


# ------------------------------------------------------------------ printing


def _components_line(basis: InvariantBasis) -> str:
    from .isotropy_subgroup import _PARAMETER_NAMES

    variables = ", ".join(basis.variables)
    if not basis.arm_chunks or len(basis.variables) > len(_PARAMETER_NAMES):
        return f"Order-parameter components: ({variables})"
    letters, start = [], 0
    for chunks in basis.arm_chunks:
        arms = []
        for size in chunks:
            arms.append(", ".join(_PARAMETER_NAMES[start:start + size]))
            start += size
        letters.append("; ".join(arms))
    return (
        f"Order-parameter components: ({variables}), in the order of the "
        f"direction patterns ({' | '.join(letters)})"
    )


def _count_line(head: str, count: int, n_products: int) -> str:
    if count == 0:
        return f"{head}: none"
    noun = _plural(count, "invariant")
    if n_products:
        return f"{head}: {noun} ({n_products} from lower degrees, {count - n_products} new)"
    return f"{head}: {noun}"


def format_invariants(basis: InvariantBasis) -> str:
    """The ``--invariants`` block as text.

    One irrep is listed by total degree (``degree 2: 1 invariant``); a
    direct sum by multidegree, in the order of total degree and then
    descending lexicographic order (``degree (2, 0): 1 invariant``,
    ``degree (1, 1): none``, ...), followed by the lowest-order coupling
    term (a multidegree with every entry >= 1).

    Args:
        basis: The result of ``invariant_polynomials`` or
            ``invariants_of_matrices``.

    Returns:
        The text block, starting with the ``* Invariant polynomials *``
        header and ending with the Molien-check line (no trailing newline).
    """
    top = max(basis.degrees)
    lines = [f"* Invariant polynomials (degree <= {top}) *", _components_line(basis)]
    multi = len(basis.parts) > 1
    for n in basis.degrees:
        if not multi:
            count = basis.counts[n]
            if n == 1 and count == 0:
                continue
            lines.append(_count_line(f"degree {n}", count, len(basis.products[n])))
            for product in basis.products[n]:
                lines.append(f"  {product.name}")
            for poly in basis.polynomials[n]:
                lines.append(f"  {poly.name} = {poly.expression}")
            continue
        for multidegree in _multidegrees(len(basis.parts), n):
            count = basis.multidegree_counts.get(multidegree, 0)
            if n == 1 and count == 0:
                continue
            products = [p for p in basis.products[n] if tuple(p.multidegree) == multidegree]
            news = [p for p in basis.polynomials[n] if tuple(p.multidegree) == multidegree]
            head = f"degree ({', '.join(str(x) for x in multidegree)})"
            lines.append(_count_line(head, count, len(products)))
            for product in products:
                lines.append(f"  {product.name}")
            for poly in news:
                lines.append(f"  {poly.name} = {poly.expression}")
    if multi:
        terms = lowest_coupling_terms(basis)
        if terms:
            described = [
                f"degree ({', '.join(str(x) for x in term.multidegree)}), "
                f"{_plural(len(term.polynomials), 'invariant')}"
                for term in terms
            ]
            text = f"Lowest-order coupling term: {described[0]}"
            if len(described) > 1:
                text += f" (also {'; '.join(described[1:])})"
            lines.append(text)
        else:
            lines.append(f"No coupling term up to total degree {top}")
    lines.append("Numbers of invariants checked against the Molien series.")
    return "\n".join(lines)


def format_restricted(restricted: dict, header: str) -> str:
    """The restricted free energy block printed after the invariants.

    Args:
        restricted: The result of ``restricted_invariants``.
        header: The direction and subgroup, e.g. ``(a,0,0) [I4/mcm (140)]``.

    Returns:
        ``Restricted to <header>:`` followed by one ``degree n: ...`` line
        per surviving degree (independent polynomials separated by ``; ``).
    """
    lines = [f"Restricted to {header}:"]
    if not restricted:
        lines.append("  no invariant survives on this direction")
    for n, texts in restricted.items():
        lines.append(f"degree {n}: {'; '.join(texts)}")
    return "\n".join(lines)
