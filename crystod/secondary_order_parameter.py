"""Secondary order parameters of an isotropy subgroup (crystod-group --secondary).

When an order parameter of irrep D condenses along a direction, the
symmetry drops to the isotropy subgroup H.  Every other parent irrep D'
that has a nonzero vector fixed by H is allowed to appear as well: it is a
secondary order parameter, induced through a coupling term ``Q^m eta``
(Q the primary order parameter, eta the secondary one) of the Landau free
energy.  This module lists them, the offline counterpart of the secondary
order parameters shown by ISOTROPY (https://iso.byu.edu).

How it works:

- H is the stabilizer of the direction in the representation of the
  primary irrep(s): the ``(i, t)`` members on the translation grid of
  that representation, with the sublattice basis ``B`` of its translations.
- Candidates are the irreps of the Gamma point and of every tabulated
  special k point that has a star arm in the reciprocal lattice of the
  sublattice (``(arm @ B.T) % DEN == 0``); for the other k points the
  translations of H already average every component to zero.  Points of
  that reciprocal lattice on symmetry lines or planes are not tabulated
  and are reported as not covered.
- For every candidate the members of H are lifted to its translation grid
  (through the least common multiple of the two grids) and the number of
  free parameters is ``n_free = (1/|H|) sum_h tr D'(h)``, checked against
  the dimension of the fixed space.  The direction of the secondary order
  parameter is that fixed space, written as an ISOTROPY-style pattern.
- The coupling is the smallest power m of the primary order parameter in
  an invariant of multidegree ``(m, 1)`` (one per primary irrep for
  coupled order parameters) that does not vanish when the primary
  components lie in the fixed space of H (the stratum of the direction,
  which equals the direction for a generic one) and the secondary ones in
  their fixed space; it is computed with ``crystod.invariants``.
- The type is ``primary`` for the given irreps, ``polar`` for a Gamma irrep
  other than the identity contained in the polar-vector representation,
  ``strain`` for the identity irrep and for one in the
  symmetric square of the vector representation, and ``other`` otherwise.

Example:
    >>> from crystod import group
    >>> rows = group.secondary_order_parameters("Pm-3m", "R4+", "a 0 0")
    >>> [(row.label, row.n_free, row.kind) for row in rows][:2]
    [('GM1+', 1, 'strain'), ('GM3+', 1, 'strain')]
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from .spacegroup_product import DEN


@dataclass
class SecondaryOrderParameter:
    """One irrep allowed by the isotropy subgroup of a direction.

    Attributes:
        label: ISO-IR label (pair label such as ``R4R5`` for a doubled
            irrep).
        kname: Name of its k point (``GM``, ``R``, ...).
        k: The tabulated k vector of the star, primitive basis, as
            ``Fraction`` values.
        dimension: Dimension of the (physically irreducible) irrep.
        n_free: Number of free parameters of the irrep fixed by the
            subgroup (dimension of its fixed space).
        direction_pattern: The fixed space as an ISOTROPY-style direction
            pattern, e.g. ``(a,0)``.
        coupling: The lowest coupling term to the primary order parameter
            as text: ``Q^2 eta`` for one primary irrep, ``Q1^1 Q2^1 eta``
            for two (several lowest multidegrees are separated by ``, ``),
            ``> N`` when there is none up to total degree N,
            ``> N (monomial limit)`` when the search stopped at degree N + 1
            because its linear problem exceeds ``MONOMIAL_LIMIT``, and an
            empty string for a primary irrep.  "Does not vanish" is tested
            on the fixed space of the subgroup in the primary irreps (the
            stratum of the direction; for a non-generic direction such as
            Fm-3m W1 ``(a;0;0;0;0;0)`` it is larger than the direction).
        coupling_degree: Total degree of that term (``m + 1``), or ``None``.
        coupling_multidegrees: The multidegrees of the lowest coupling
            terms (primary degrees, then 1 for the secondary irrep).
        kind: ``"primary"``, ``"polar"`` (a non-identity Gamma irrep in the
            polar-vector representation), ``"strain"`` (the identity irrep,
            or a Gamma irrep in the symmetric square of the vector
            representation) or ``"other"``.
    """

    label: str
    kname: str
    k: tuple
    dimension: int
    n_free: int
    direction_pattern: str
    coupling: str
    coupling_degree: int | None
    coupling_multidegrees: list = field(default_factory=list)
    kind: str = "other"


@dataclass
class _Secondary:
    rows: list
    uncovered: list
    failed: dict


# ------------------------------------------------------------------ helpers


def _lift(members, n_from: int, n_to: int) -> list[tuple[int, np.ndarray]]:
    """The members ``(i, t mod n_from)`` of H on the grid ``n_to``, through
    the common grid lcm(n_from, n_to); every image element appears equally
    often, so averages over the list are group averages."""
    common = int(np.lcm(n_from, n_to))
    steps = common // n_from
    lifted = []
    for i, t in members:
        t = np.asarray(t, dtype=np.int64)
        for z in itertools.product(range(steps), repeat=3):
            lifted.append((int(i), (t + n_from * np.asarray(z, dtype=np.int64)) % n_to))
    return lifted


def _lookup(representation) -> dict:
    return {
        (int(i), tuple(int(x) for x in t)): matrix
        for i, t, matrix in representation.elements
    }


def _sublattice_reciprocal(B: np.ndarray) -> tuple[list[np.ndarray], int]:
    """The reciprocal-lattice points of the sublattice ``B`` (rows) modulo
    the parent reciprocal lattice, as integer numerators over ``det``."""
    B = np.asarray(B, dtype=np.int64)
    det = abs(int(round(np.linalg.det(B))))
    adjugate = np.rint(np.linalg.inv(B) * det).astype(np.int64)
    generators = [adjugate[:, j] % det for j in range(3)]
    seen = {(0, 0, 0)}
    frontier = [np.zeros(3, dtype=np.int64)]
    while frontier:
        new = []
        for k in frontier:
            for g in generators:
                image = (k + g) % det
                key = tuple(int(x) for x in image)
                if key not in seen:
                    seen.add(key)
                    new.append(image)
        frontier = new
    return [np.array(k, dtype=np.int64) for k in sorted(seen)], det


def _k_text(numerators, denominator: int) -> str:
    return "(" + ",".join(
        str(Fraction(int(v), denominator)) for v in np.asarray(numerators) % denominator
    ) + ")"


def _uncovered_points(algebra, B) -> list[str]:
    """The reciprocal-lattice points of the sublattice that are not on a
    tabulated special star (one representative per star)."""
    points, det = _sublattice_reciprocal(B)
    tabulated = set()
    for kname in algebra.k_by_kname:
        arms, _ = algebra.star(kname)
        tabulated |= {tuple(int(x) for x in np.asarray(arm) % DEN) for arm in arms}
    texts, done = [], set()
    for k in points:
        key = tuple(int(x) for x in k)
        if key in done:
            continue
        orbit = {
            tuple(int(x) for x in (k @ np.asarray(W, dtype=np.int64)) % det)
            for W in algebra.rotations
        }
        done |= orbit
        if np.all((k * DEN) % det == 0) and tuple(
            int(x) for x in (k * DEN // det) % DEN
        ) in tabulated:
            continue
        texts.append(_k_text(k, det))
    return texts


def _kind(algebra, representation, primary: bool) -> str:
    from .invariants import gamma_tensor_multiplicities

    if primary:
        return "primary"
    if np.any(np.asarray(representation.k) % DEN):
        return "other"
    if representation.dimension == 1 and all(
        abs(float(np.trace(matrix)) - 1.0) < 1e-6 for _, _, matrix in representation.elements
    ):
        return "strain"  # the identity irrep: a volume strain, never polar
    n_polar, n_strain = gamma_tensor_multiplicities(
        representation.elements, algebra.rotations
    )
    if n_polar:
        return "polar"
    if n_strain:
        return "strain"
    return "other"


def _coupling_text(multidegrees: list[tuple], n_primary: int) -> str:
    texts = []
    for multidegree in multidegrees:
        if n_primary == 1:
            texts.append(f"Q^{multidegree[0]} eta")
            continue
        factors = [
            f"Q{j + 1}^{power}" for j, power in enumerate(multidegree[:-1]) if power
        ]
        texts.append(" ".join(factors + ["eta"]))
    return ", ".join(texts)


def _lowest_coupling(algebra, primary_parts, candidate, primary_basis,
                     secondary_basis, max_degree: int, rng_seed: int = 7):
    """Smallest primary degree m of a non-vanishing ``(m.., 1)`` invariant.

    The invariants of a multidegree are evaluated on random points of
    ``Fix_H(primary) x Fix_H(candidate)`` (the columns of
    ``primary_basis`` and ``secondary_basis``).

    Returns ``(multidegrees, m, limit)``: ``([], None, None)`` when none
    exists up to total degree ``max_degree``; ``([], None, n)`` when the
    search stopped at total degree ``n + 1`` because a multidegree has more
    than ``MONOMIAL_LIMIT`` monomials (none exists up to total degree
    ``n``).  A multidegree with more than ``MONOMIAL_WARNING`` monomials
    raises a ``UserWarning`` before its linear problem is solved."""
    import math
    import warnings

    from .invariants import (
        MONOMIAL_LIMIT,
        MONOMIAL_WARNING,
        _compositions,
        _evaluate,
        _generator_candidates,
        _key,
        invariant_space,
        molien_counts,
    )
    from .isotropy_subgroup import CoupledRepresentation

    combined = CoupledRepresentation.from_parts(algebra, list(primary_parts) + [candidate])
    matrices = [matrix for _, _, matrix in combined.elements]
    distinct: dict[bytes, np.ndarray] = {}
    for matrix in _generator_candidates(algebra, combined):
        distinct.setdefault(_key(matrix), matrix)
    generators = list(distinct.values())
    dims = [part.dimension for part in primary_parts] + [candidate.dimension]
    n_primary = len(primary_parts)

    rng = np.random.default_rng(rng_seed)
    n_points = 8
    points = np.hstack([
        rng.normal(size=(n_points, primary_basis.shape[1])) @ primary_basis.T,
        rng.normal(size=(n_points, secondary_basis.shape[1])) @ secondary_basis.T,
    ])
    wanted = [
        tuple(primary) + (1,)
        for m in range(1, max_degree)
        for primary in _compositions(m, n_primary)
    ]
    if not wanted:
        return [], None, None
    counts = molien_counts(matrices, dims, wanted)
    for m in range(1, max_degree):
        found = []
        for primary in _compositions(m, n_primary):
            multidegree = tuple(primary) + (1,)
            if counts[multidegree] == 0:
                continue
            size = math.prod(math.comb(d + n - 1, n) for d, n in zip(dims, multidegree))
            if size > MONOMIAL_LIMIT:
                return [], None, m
            if size > MONOMIAL_WARNING:
                warnings.warn(
                    f"{candidate.label}: coupling multidegree {multidegree} has {size} "
                    "monomials; this may take long (lower it with --degree).",
                    UserWarning, stacklevel=2,
                )
            exponents, space = invariant_space(matrices, dims, multidegree, generators)
            values = _evaluate(points, exponents)
            restricted = values @ space
            scale = max(1.0, float(np.max(np.abs(values))))
            if np.max(np.abs(restricted)) > 1e-8 * scale:
                found.append(multidegree)
        if found:
            return found, m, None
    return [], None, None


# ------------------------------------------------------------------ analysis


def analyze_secondary(algebra, representation, members, B, max_degree: int = 4) -> _Secondary:
    """Secondary order parameters of a resolved isotropy subgroup.

    The engine behind ``--secondary`` and ``secondary_order_parameters``,
    for callers that already hold the stabilizer (the CLI).

    Args:
        algebra: The ``SpaceGroupIrrepAlgebra`` of the parent group.
        representation: The ``InducedRepresentation`` or
            ``CoupledRepresentation`` of the primary order parameter.
        members: The ``(i, t)`` elements of the isotropy subgroup on the
            grid of ``representation`` (the exact stabilizer of the
            direction).
        B: Sublattice basis of the subgroup (rows, parent primitive units),
            from ``IsotropyAnalyzer.subgroup_of``.
        max_degree: Highest total degree of the coupling terms searched.

    Returns:
        A record with ``rows`` (``SecondaryOrderParameter`` list, Gamma
        first, then the other k points in table order), ``uncovered`` (the
        reciprocal-lattice points of the sublattice off the tabulated
        special points, as text) and ``failed`` (label -> reason for irreps
        whose representation could not be built).

    Raises:
        RuntimeError: An internal consistency check failed (the trace
            average differs from the fixed-space dimension, or the primary
            irreps do not reproduce the fixed space of the direction).
    """
    from .isotropy_subgroup import (
        CoupledRepresentation,
        InducedRepresentation,
        IsotropyAnalyzer,
        _failure_reason,
        _projector,
    )

    coupled = isinstance(representation, CoupledRepresentation)
    primary_parts = list(representation.parts) if coupled else [representation]
    primary_by_label = {part.label: part for part in primary_parts}
    grid = representation.grid_n
    analyzer = IsotropyAnalyzer.from_representation(algebra, representation)
    primary_basis = analyzer.fixed_space(members)
    if primary_basis.shape[1] == 0:
        raise RuntimeError("the direction has no fixed vector (internal error).")

    B = np.asarray(B, dtype=np.int64)
    knames = sorted(
        algebra.k_by_kname,
        key=lambda name: (bool(np.any(np.asarray(algebra.k_by_kname[name]) % DEN)),),
    )
    rows: list[SecondaryOrderParameter] = []
    failed: dict[str, str] = {}
    listed_labels: set[str] = set()
    for kname in knames:
        arms, _ = algebra.star(kname)
        if not any(np.all((np.asarray(arm) @ B.T) % DEN == 0) for arm in arms):
            continue
        characters: list[np.ndarray] = []
        for irrep in algebra.irreps_by_kname[kname]:
            try:
                candidate = InducedRepresentation(algebra, irrep.name)
            except (SystemExit, Exception) as exc:  # noqa: BLE001 - reported
                failed[irrep.name] = _failure_reason(exc)
                continue
            character = np.array([np.trace(m) for _, _, m in candidate.elements])
            if candidate.doubled and any(
                np.allclose(character, other, atol=1e-6) for other in characters
            ):
                continue  # the conjugate partner of a listed pair
            characters.append(character)
            if candidate.label in listed_labels:
                continue  # a -k pair already listed at its partner point
            listed_labels.add(candidate.label)
            is_primary = candidate.label in primary_by_label
            if is_primary:
                candidate = primary_by_label[candidate.label]  # the user's basis

            lifted = _lift(members, grid, candidate.grid_n)
            lookup = _lookup(candidate)
            average = sum(
                float(np.trace(lookup[(i, tuple(int(x) for x in t))])) for i, t in lifted
            ) / len(lifted)
            n_free = int(round(average))
            if abs(average - n_free) > 1e-6:
                raise RuntimeError(
                    f"{candidate.label}: the fixed-space average {average:.6f} is "
                    "not an integer (internal error)."
                )
            if n_free == 0:
                continue
            sub = IsotropyAnalyzer.from_representation(algebra, candidate)
            fixed = sub.fixed_space(lifted)
            if fixed.shape[1] != n_free:
                raise RuntimeError(
                    f"{candidate.label}: n_free {n_free} differs from the fixed-space "
                    f"dimension {fixed.shape[1]} (internal error)."
                )
            pattern, _ = sub._direction_pattern(_projector(fixed))
            if is_primary:
                coupling, degree, multidegrees = "", None, []
            else:
                multidegrees, m, limit = _lowest_coupling(
                    algebra, primary_parts, candidate, primary_basis, fixed, max_degree
                )
                if limit is not None:
                    coupling, degree = f"> {limit} (monomial limit)", None
                elif m is None:
                    coupling, degree = f"> {max_degree}", None
                else:
                    coupling = _coupling_text(multidegrees, len(primary_parts))
                    degree = m + 1
            k = tuple(Fraction(int(v), DEN) for v in np.asarray(algebra.k_by_kname[kname]))
            rows.append(SecondaryOrderParameter(
                label=candidate.label,
                kname=kname,
                k=k,
                dimension=int(candidate.dimension),
                n_free=n_free,
                direction_pattern=pattern,
                coupling=coupling,
                coupling_degree=degree,
                coupling_multidegrees=list(multidegrees),
                kind=_kind(algebra, candidate, is_primary),
            ))
    primary_total = sum(row.n_free for row in rows if row.kind == "primary")
    if primary_total != primary_basis.shape[1]:
        raise RuntimeError(
            f"the primary irreps fix {primary_total} parameters, the direction "
            f"{primary_basis.shape[1]} (internal error)."
        )
    return _Secondary(rows, _uncovered_points(algebra, B), failed)


def secondary_order_parameters(space_group, irreps, direction, max_degree: int = 4):
    """Secondary order parameters of an order-parameter direction.

    Args:
        space_group: Parent space-group number or international symbol.
        irreps: One ISO-IR label (``"R4+"``) or a list for coupled order
            parameters (``["R4+", "M3+"]``).
        direction: The order-parameter components, as ``--order-parameter``
            takes them: a string (``"a 0 0"``, ``"a;0;0"``) or a list of
            tokens, one per component (all primary irreps, in order).
        max_degree: Highest total degree of the coupling terms searched
            (default 4).

    Returns:
        A list of ``SecondaryOrderParameter`` records (the primary irreps
        included, with ``kind == "primary"``), Gamma first, then the other
        tabulated special k points.

    Raises:
        SystemExit: Unknown space group or irrep, or a direction with the
            wrong number of components (``ValueError`` through
            ``crystod.group``).

    Example:
        >>> from crystod import group
        >>> rows = group.secondary_order_parameters(
        ...     "I4/mmm", ["X2+", "X3-"], "0 a 0 c")
        >>> [(r.label, r.coupling) for r in rows if r.kind == "polar"]
        [('GM5-', 'Q1^1 Q2^1 eta')]
    """
    from .isotropy_subgroup import IsotropyAnalyzer, _order_parameter_tokens

    tokens = _order_parameter_tokens(
        [direction] if isinstance(direction, str) else [str(x) for x in direction]
    )
    analyzer = IsotropyAnalyzer(str(space_group), irreps)
    eta = analyzer.resolve_direction(tokens)
    members = [
        (i, t) for i, t, matrix in analyzer.elements
        if np.allclose(matrix @ eta, eta, atol=1e-6)
    ]
    _, _, _, B, *_ = analyzer.subgroup_of(members)
    return analyze_secondary(
        analyzer.algebra, analyzer.representation, members, B, max_degree
    ).rows


def format_secondary(result: _Secondary, header: str, max_degree: int) -> str:
    """The ``* Secondary order parameters *`` block as text.

    Args:
        result: The record returned by ``analyze_secondary``.
        header: The subgroup line, e.g. ``H = I4/mcm (140), index 6``.
        max_degree: The highest total degree searched (for the notes).

    Returns:
        The text block (no trailing newline).
    """
    table = [("irrep", "k", "dim", "n_free", "direction", "coupling", "type")]
    for row in result.rows:
        k_text = "(" + ",".join(str(v) for v in row.k) + ")"
        table.append((
            row.label, k_text, str(row.dimension), str(row.n_free),
            row.direction_pattern, row.coupling or "-", row.kind,
        ))
    widths = [max(len(r[c]) for r in table) for c in range(len(table[0]))]
    lines = ["* Secondary order parameters *", header]
    for r in table:
        lines.append("  ".join(cell.ljust(width) for cell, width in zip(r, widths)).rstrip())
    lines.append(
        "coupling: lowest invariant Q^m eta (Q primary, eta this irrep) that does "
        f"not vanish on the fixed space of H, total degree <= {max_degree}"
    )
    if result.uncovered:
        lines.append(
            "not covered (k points of the subgroup lattice on symmetry lines or "
            f"planes, not tabulated): {', '.join(result.uncovered)}"
        )
    for label, reason in result.failed.items():
        lines.append(f"note: {label}: not analyzed ({reason})")
    return "\n".join(lines)
