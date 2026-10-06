"""Correlation of irreps (crystod-group --correlate).

Three kinds of restriction of a representation to a subgroup are computed:

- Point-group correlation tables (``--correlate --pg G --subgroup H``):
  every irrep of the point group G restricted to each inequivalent
  orientation of a subgroup type H (m-3m -> 4/mmm: T2g -> B2g + Eg).

- Subduction to an isotropy subgroup (``--correlate --parent SG --irrep IR
  --order-parameter ...``): which irreps of the Gamma point of the isotropy
  subgroup H a parent irrep becomes.  This is how a zone-boundary soft mode
  of the parent turns into Raman-active modes of the low-symmetry phase:
  R4+ of Pm-3m (SrTiO3) becomes A1g + Eg of I4/mcm.
- Compatibility relations along a symmetry line (``--correlate --sg SG
  --kpoint K0 K1 [--line L]``): how the small irreps at two special points
  split or join on the line between them (GM4- -> DT1 + DT5 in Pm-3m),
  the offline counterpart of COMPATIBILITY RELATIONS of the Bilbao
  Crystallographic Server.

How the subduction works:

- H is the stabilizer of the order-parameter direction in the
  representation of the primary irrep(s), with the sublattice basis ``B``
  of its translations (``IsotropyAnalyzer.subgroup_of``).
- For every irrep D of the list, the vectors invariant under the
  translations of H span its Gamma sector for H: the projector ``P`` is
  the average of ``D(1|t)`` over the lattice translations ``t`` of H, and
  ``dimension - rank P`` components belong to non-Gamma k points of H.
- H acts on that sector through its point group; the character
  ``tr P D(h)`` on one coset representative h per rotation is decomposed
  into the Gamma irreps of H with the ISO-IR labeler of H, built on a
  generic-orbit structure of H in the sublattice basis
  (``crystod.isoir.get_cached_labeler``), so the labels refer to the
  ISO-IR frame of that structure (printed as a basis and an origin in
  parent conventional units: the labels of H depend on the choice among
  normalizer-equivalent settings).
- Checks: the multiplicity of the identity irrep equals the number of
  components fixed by H (``n_free`` of the direction for a primary irrep),
  and the dimensions add up to ``rank P``.

How the compatibility relations work:

- The line joining the two points is the segment from the tabulated k0 to
  the nearest arm of the star of k1 (the one with the largest little
  group first); ``kdelta`` is the first point of the 1/24 grid on the
  segment whose little group is the little group of the line.
- The small characters at both end points (spgrep phase convention,
  ``SpaceGroupIrrepAlgebra.induced_characters``) are restricted to that
  little group and multiplied by ``exp(SIGMA 2 pi i (kdelta - k_end).v)``
  for the translation ``v`` of every coset representative, which turns
  them into characters with the factor system of kdelta; the result is
  decomposed into the small irreps at kdelta with the ISO-IR labeler of
  the space group (``IsoIRLabeler.decompose_characters``).
- Check: the dimensions of the line irreps add up to the dimension of the
  small irrep at the end point.

How the correlation tables work:

- The subgroups of G are generated from phonopy's table of G (setting 0)
  by adjoining one element at a time to the cyclic subgroups, identified
  with ``spglib.get_pointgroup`` and grouped into conjugacy classes under
  G: one column per class (m-3m has three classes of mm2: the twofold axis
  along z with the axial or the diagonal mirrors, and along [110]).
- The elements of a representative are carried onto the elements of
  phonopy's table of H: unchanged when they match one of its settings (the
  two settings of -42m, 3m, -3m and -6m2 are both tried, and the header
  names the setting), else with spglib's transformation to the standard
  orientation or a conjugation solved for the generators, whichever puts
  the principal axis of the table first (z before x before diagonals).
  The classes of H that fix the
  orientation (``C2' = 2_x, 2_y``) are printed, since B1/B2/B3 depend on
  them.
- The restricted characters are reduced with the norm of each row of the
  table of H (a complex-conjugate pair merged into one real E counts
  once); the multiplicities must be integers and the dimensions must add
  up to that of the irrep of G (checked for every row).

Example:
    >>> from crystod import group
    >>> group.correlation_table("m-3m", "4/mmm").rows["Eg"]
    [[('A1g', 1), ('B1g', 1)]]
    >>> [(s.label, s.gamma_part) for s in group.subduce_to_child(
    ...     "Pm-3m", "R4+", "a a a")]
    [('R4+', [('GM1+', 1), ('GM3+', 1)])]
"""

from __future__ import annotations

import argparse
import functools
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

# the generic points of the orbit structure of a subgroup (the ones of
# IsotropyAnalyzer._identify_type, so that spglib sees the same structure)
_GENERIC_POINTS = (
    np.array([0.1234, 0.2345, 0.3178]),
    np.array([0.4321, 0.0567, 0.1873]),
)


@dataclass
class Subduction:
    """One parent irrep restricted to the Gamma point of an isotropy subgroup.

    Attributes:
        label: ISO-IR label of the parent irrep (pair label such as ``P1P2``
            for a doubled irrep).
        kname: Name of its k point (``GM``, ``R``, ...).
        dimension: Dimension of the (physically irreducible) parent irrep.
        gamma_part: The Gamma irreps of the subgroup H in its Gamma sector,
            as ``(ISO-IR label, multiplicity)`` pairs in table order; a
            complex irrep and its conjugate partner that occur equally often
            form one physically irreducible term with the ISOTROPY pair
            label (``GM2+GM3+``).
        nongamma_components: Number of components at non-Gamma k points of
            H (``dimension`` minus the size of the Gamma sector).
        child_number: Space-group number of H.
        child_symbol: International symbol of H.
        basis: Rows of the conventional basis of H in the ISO-IR frame the
            labels refer to, in parent conventional units.
        origin: Origin of that frame in parent conventional units.
        mulliken: Mulliken symbols of the labels of ``gamma_part``
            (phonopy's point-group tables, axes of spglib's conventional
            setting of H); a label without a symbol is missing.
        primary: ``True`` for an irrep of the order parameter itself.
        n_free: Multiplicity of the identity irrep of H, i.e. the number of
            components fixed by H (the free parameters of the direction for
            a primary irrep).
        gamma_irreps: ``gamma_part`` with complex pairs left as two
            separate ISO-IR irreps.
    """

    label: str
    kname: str
    dimension: int
    gamma_part: list
    nongamma_components: int
    child_number: int
    child_symbol: str
    basis: np.ndarray | None = None
    origin: np.ndarray | None = None
    mulliken: dict = field(default_factory=dict)
    primary: bool = False
    n_free: int = 0
    gamma_irreps: list = field(default_factory=list)


@dataclass
class Compatibility:
    """A small irrep at a special point restricted to a symmetry line.

    Attributes:
        k0_label: Name of the special point (``GM``, ``X``, ...).
        k0_irrep: ISO-IR label of the small irrep at that point.
        line_label: ISO-IR name of the line (``DT``, ``SM``, ...).
        line_irreps: The small irreps on the line, as ``(ISO-IR label,
            multiplicity)`` pairs.
        dimension: Dimension of the small irrep at the point.
        k_point: The arm of the point the line starts from, primitive basis,
            as ``Fraction`` values.
        k_delta: The point of the line at which the line irreps are
            labeled, primitive basis, as ``Fraction`` values.
        fixed_dimension: Dimension of the k subspace fixed by the little
            co-group of ``k_delta``: 1 for a symmetry line, 2 for a symmetry
            plane, 3 for a general point (possible only when two points are
            given that no line joins).
    """

    k0_label: str
    k0_irrep: str
    line_label: str
    line_irreps: list
    dimension: int = 0
    k_point: tuple = ()
    k_delta: tuple = ()
    fixed_dimension: int = 1


@dataclass
class _SubductionReport:
    """Everything the ``--correlate --parent`` report prints."""

    algebra: object
    header: str
    info: object
    size: int
    index: int
    B: np.ndarray
    setting: tuple | None
    rows: list


@dataclass
class _Segment:
    """A symmetry line between two special points."""

    name0: str
    name1: str
    k0: np.ndarray
    k1: np.ndarray
    kdelta: np.ndarray
    ops: list
    line: str
    length: float
    fixed_dimension: int = 1


# ------------------------------------------------------------------ helpers


def _orbit_cell(rotations, translations, lattice):
    """Generic-orbit structure of a subgroup in its sublattice basis (the
    structure ``IsotropyAnalyzer._identify_type`` identifies)."""
    positions, numbers = [], []
    for species, x0 in enumerate(_GENERIC_POINTS):
        orbit = []
        for W, v in zip(rotations, translations):
            x = np.mod(np.asarray(W) @ x0 + np.asarray(v), 1.0)
            if not any(np.allclose(x, p, atol=1e-6) for p in orbit):
                orbit.append(x)
        positions.extend(orbit)
        numbers.extend([species + 1] * len(orbit))
    return np.asarray(lattice, dtype=float), np.array(positions), numbers


def _lookup(representation) -> dict:
    return {
        (int(i), tuple(int(x) for x in t)): matrix
        for i, t, matrix in representation.elements
    }


def _sublattice_translations(B: np.ndarray, period: int) -> list[np.ndarray]:
    """The lattice translations of the sublattice ``B`` (rows, parent
    primitive units) modulo ``period`` (``period`` times every parent
    lattice vector must lie in the sublattice)."""
    B_inv = np.linalg.inv(np.asarray(B, dtype=float))
    found = []
    for t in np.ndindex(period, period, period):
        c = np.asarray(t, dtype=float) @ B_inv
        if np.allclose(c, np.rint(c), atol=1e-8):
            found.append(np.asarray(t, dtype=np.int64))
    return found


def _frame_setting(labeler, lattice, B, primitive_matrix):
    """Basis rows and origin (parent conventional units) of the ISO-IR frame
    ``x_iso = P x + origin_shift`` of the subgroup, as
    ``IsotropyAnalyzer.conventional_setting`` computes it for spglib's."""
    P = np.asarray(labeler.P, dtype=float)
    shift = np.asarray(labeler.origin_shift, dtype=float)
    lattice = np.asarray(lattice, dtype=float)
    L_parent_prim = np.linalg.inv(np.asarray(B, dtype=float)) @ lattice
    L_parent_conv = np.linalg.inv(np.asarray(primitive_matrix, dtype=float)).T @ L_parent_prim
    L_child = np.linalg.inv(P).T @ lattice
    basis = L_child @ np.linalg.inv(L_parent_conv)
    # reduce modulo the translations of H (cell coordinates of H), not modulo
    # the parent lattice: a parent translation outside T_H need not normalize
    # H, and would describe the frame of a conjugate subgroup (another domain)
    origin_cell = -np.linalg.inv(P) @ shift
    origin_cell = origin_cell - np.floor(origin_cell + 1e-9)
    origin = origin_cell @ lattice @ np.linalg.inv(L_parent_conv)
    return np.round(basis, 6) + 0.0, np.round(origin, 6) + 0.0


def _gamma_family(labeler, rotations, translations):
    """ISO-IR labels and characters of the Gamma irreps of the subgroup on
    its operations, or ``None``.

    Coupled to private parts of ``crystod.isoir.IsoIRLabeler``
    (``conventional_k``, ``conventional_operations``, ``_matched_families``,
    ``_family_characters_checked``); the section 16 tests guard the result.
    """
    k_conv = labeler.conventional_k([0.0, 0.0, 0.0])
    conv_ops = labeler.conventional_operations(rotations, translations)
    for _, family in labeler._matched_families(k_conv):
        chars = labeler._family_characters_checked(family, conv_ops)
        if chars is not None:
            return tuple(ir.label for ir, _, _ in family), np.asarray(chars, dtype=complex)
    return None


def _complex_partners(family) -> dict:
    """``{label: partner}`` for every complex-type Gamma irrep of the family
    and its complex-conjugate partner (empty for ``None``)."""
    if family is None:
        return {}
    labels, chars = family
    partners = {}
    for a, chi in enumerate(chars):
        if np.abs(chi.imag).max() <= 1e-6:
            continue
        for b, other in enumerate(chars):
            if b != a and np.allclose(other, chi.conj(), atol=1e-6):
                partners[labels[a]] = labels[b]
                break
    return partners


def _pair_label(name: str, partner: str) -> str:
    """The ISOTROPY pair label of a complex irrep and its conjugate
    (``GM2+GM3+``), as ``InducedRepresentation.label`` and
    ``crystod.phonon_activity`` write it."""
    from .phonon_activity import _natural_key

    return "".join(sorted((name, partner), key=_natural_key))


def _merge_pairs(gamma_part, partners) -> list:
    """Join each complex-conjugate pair of equal multiplicity into one pair
    term (the physically irreducible representation), in place of its first
    member."""
    counts = dict(gamma_part)
    merged, used = [], set()
    for name, n in gamma_part:
        if name in used:
            continue
        partner = partners.get(name)
        if partner is not None and counts.get(partner) == n:
            merged.append((_pair_label(name, partner), n))
            used |= {name, partner}
        else:
            merged.append((name, n))
            used.add(name)
    return merged


def _gamma_mulliken(family, cell, rotations) -> dict:
    """Mulliken symbols of the Gamma irreps of the subgroup, keyed by ISO-IR
    label and by pair label of complex pairs (phonopy's tables, as
    ``crystod.phonon.mulliken_symbols``); an empty dict when they cannot be
    assigned.

    Coupled to the private ``crystod.phonon_activity._mulliken_of_table``
    (fed a duck-typed table); the section 16 tests assert bracketed symbols.
    """
    import types
    import warnings

    import spglib

    from .phonon_activity import _mulliken_of_table

    if family is None:
        return {}
    try:
        labels, chars = family
        dataset = spglib.get_symmetry_dataset(cell, symprec=1e-4)
        if dataset is None:
            return {}

        def item(name):
            return dataset[name] if isinstance(dataset, dict) else getattr(dataset, name)

        table = types.SimpleNamespace(
            vibrations=types.SimpleNamespace(spglib_dataset={
                "pointgroup": item("pointgroup"),
                "transformation_matrix": item("transformation_matrix"),
            }),
            rotations=np.asarray(rotations),
            labels=labels,
            characters=np.array([np.conj(chi) for chi in chars]),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _mulliken_of_table(table)
    except Exception:
        return {}


def _format_vector(values) -> str:
    from .isotropy_subgroup import _format_setting_value

    return "(" + ",".join(_format_setting_value(x) for x in values) + ")"


def _k_fractions(k_int) -> tuple:
    from .spacegroup_product import DEN

    return tuple(Fraction(int(v), DEN) for v in np.asarray(k_int))


def _k_text(values) -> str:
    return "(" + ", ".join(str(v) for v in values) + ")"


def _direction_tokens(direction) -> list[str]:
    from .isotropy_subgroup import _order_parameter_tokens

    if isinstance(direction, str):
        return _order_parameter_tokens([direction])
    return _order_parameter_tokens([str(x) for x in direction])


# ------------------------------------------------------- (c) subduction to H


def _subduction_report(parent, irreps, direction, irrep_list=None) -> _SubductionReport:
    """Run the subduction and keep what the report prints."""
    from .isotropy_subgroup import (
        CoupledRepresentation,
        InducedRepresentation,
        IsotropyAnalyzer,
        _arm_join,
        _resolve_stratum,
    )
    from .isoir import get_cached_labeler

    if isinstance(irreps, str):
        irreps = [irreps]
    tokens = _direction_tokens(direction)
    analyzer = IsotropyAnalyzer(str(parent), list(irreps))
    algebra = analyzer.algebra
    representation = analyzer.representation
    coupled = isinstance(representation, CoupledRepresentation)
    parts = list(representation.parts) if coupled else [representation]
    members, info, size, index, B, rotations, translations, lattice = (
        _resolve_stratum(analyzer, tokens)
    )
    if coupled:
        chunks, start = [], 0
        for part in parts:
            piece = tokens[start : start + part.dimension]
            chunks.append(f"{part.label}({_arm_join(piece, part.arm_chunks)})")
            start += part.dimension
        header = " ".join(chunks)
    else:
        header = f"{representation.label}({_arm_join(tokens, representation.arm_chunks)})"

    # per primary part: its number of components fixed by H
    fixed = analyzer.fixed_space(members)
    bounds = np.cumsum([0] + [part.dimension for part in parts])
    n_free_of = {}
    for j, part in enumerate(parts):
        block = fixed[bounds[j] : bounds[j + 1], :]
        n_free_of[part.label] = int(np.linalg.matrix_rank(block, tol=1e-6)) if block.size else 0
        n_free_of.setdefault(part.irrep.name, n_free_of[part.label])

    # one coset representative per rotation of H, in the order of
    # subgroup_of (aligned with its rotations and translations)
    chosen: dict[int, np.ndarray] = {}
    for i, t in members:
        if i not in chosen:
            chosen[i] = np.asarray(t, dtype=np.int64)
    if len(chosen) != len(rotations):
        raise RuntimeError("internal error: coset representatives of H do not "
                           "match its operations.")
    identity = algebra._rotation_index[algebra._key(np.eye(3))]

    cell = _orbit_cell(rotations, translations, lattice)
    labeler = get_cached_labeler(int(info.number), cell)
    if labeler is None:
        raise SystemExit(
            f"ERROR: no ISO-IR frame for the isotropy subgroup "
            f"{info.international_short} (No. {info.number}); its Gamma irreps "
            "cannot be labeled."
        )
    setting = _frame_setting(labeler, lattice, B, algebra.primitive_matrix)
    try:
        family = _gamma_family(labeler, rotations, translations)
    except Exception:
        family = None
    partners = _complex_partners(family)
    mulliken = _gamma_mulliken(family, cell, rotations)

    by_label = {part.label: part for part in parts}
    by_label.update({part.irrep.name: part for part in parts})
    labels = list(irrep_list) if irrep_list else [part.label for part in parts]
    rows = []
    done = set()
    grid_n = int(representation.grid_n)
    for label in labels:
        if label in done:
            continue
        done.add(label)
        rep = by_label.get(label)
        primary = rep is not None
        if rep is None:
            rep = InducedRepresentation(algebra, label)
        lookup = _lookup(rep)
        period = int(rep.grid_n)
        translations_H = _sublattice_translations(B, int(np.lcm(grid_n, period)))
        P = sum(
            lookup[(identity, tuple(int(x) for x in t % period))] for t in translations_H
        ) / len(translations_H)
        rank = int(round(float(np.trace(P).real)))
        characters = np.array([
            np.trace(P @ lookup[(i, tuple(int(x) for x in t % period))])
            for i, t in chosen.items()
        ])
        trivial = complex(np.mean(characters))
        n_free = int(round(trivial.real))
        if abs(trivial - n_free) > 1e-6:
            raise RuntimeError(f"internal error: non-integral identity multiplicity "
                               f"in the subduction of {rep.label}.")
        gamma_irreps: list[tuple[str, int]] = []
        if rank:
            result = labeler.decompose_characters(
                [0.0, 0.0, 0.0], rotations, translations, characters
            )
            if result is None:
                raise RuntimeError(
                    f"internal error: the Gamma sector of {rep.label} does not "
                    f"decompose into Gamma irreps of {info.international_short}."
                )
            gamma_irreps = [(name, int(n)) for name, n, _ in result[0]]
            total = sum(n * dim for _, n, dim in result[0])
            if total != rank:
                raise RuntimeError(
                    f"internal error: the Gamma irreps of {rep.label} add up to "
                    f"{total} components, its Gamma sector has {rank}."
                )
            identity_count = sum(n for name, n in gamma_irreps if name in ("GM1", "GM1+"))
            if identity_count != n_free:
                raise RuntimeError(
                    f"internal error: {rep.label} contains GM1 {identity_count} "
                    f"times, but {n_free} components are fixed by H."
                )
        if primary and n_free != n_free_of[label]:
            raise RuntimeError(
                f"internal error: {rep.label} contains the identity irrep of H "
                f"{n_free} times, but the direction has {n_free_of[label]} free "
                "parameters in it."
            )
        rows.append(Subduction(
            label=rep.label,
            kname=rep.irrep.kpname,
            dimension=int(rep.dimension),
            gamma_part=_merge_pairs(gamma_irreps, partners),
            nongamma_components=int(rep.dimension) - rank,
            child_number=int(info.number),
            child_symbol=info.international_short,
            basis=setting[0],
            origin=setting[1],
            mulliken={name: mulliken[name] for name, _ in
                      _merge_pairs(gamma_irreps, partners) + gamma_irreps
                      if name in mulliken},
            primary=primary,
            n_free=n_free,
            gamma_irreps=gamma_irreps,
        ))
    conventional = analyzer.conventional_setting(B, rotations, translations, lattice, info)
    return _SubductionReport(algebra, header, info, size, index, B, conventional, rows)


def subduce_to_child(parent, irreps, direction, irrep_list=None) -> list[Subduction]:
    """Parent irreps restricted to the Gamma point of an isotropy subgroup.

    The computation behind ``crystod-group --correlate --parent SG --irrep IR
    --order-parameter ...``.

    Args:
        parent: Parent space-group number or international symbol.
        irreps: One ISO-IR label (``"R4+"``) or a list for coupled order
            parameters (``["R4+", "M3+"]``): the irreps whose direction
            defines the isotropy subgroup H.
        direction: The order-parameter components, as ``--order-parameter``
            takes them: a string (``"a 0 0"``, ``"a;0;0"``) or a list of
            tokens, one per component (all primary irreps, in order).
        irrep_list: The parent irreps to subduce (default: the primary
            irreps).  Any tabulated irrep of the parent may be given.

    Returns:
        One ``Subduction`` per irrep of ``irrep_list``, in its order.

    Raises:
        SystemExit: Unknown space group or irrep, or a direction with the
            wrong number of components (``ValueError`` through
            ``crystod.group``).
        RuntimeError: An internal consistency check failed (the identity
            multiplicity differs from the number of fixed components, or
            the dimensions do not add up).

    Example:
        >>> from crystod import group
        >>> rows = group.subduce_to_child("Pm-3m", ["R4+", "M3+"], "0 a a d 0 0")
        >>> [(r.label, r.gamma_part, r.nongamma_components) for r in rows]
        [('R4+', [('GM1+', 1), ('GM2+', 1), ('GM4+', 1)], 0), ('M3+', [('GM1+', 1)], 2)]
    """
    return _subduction_report(parent, irreps, direction, irrep_list).rows


def _term_text(name: str, count: int, mulliken: dict) -> str:
    text = f"{count} {name}" if count > 1 else name
    if name in mulliken:
        text += f" [{mulliken[name]}]"
    return text


def format_subduction(rows: list[Subduction]) -> str:
    """The ``* Subduction to the Gamma point of H *`` block as text.

    Args:
        rows: The records returned by ``subduce_to_child`` (one subgroup).

    Returns:
        The text block (no trailing newline): the subgroup and the ISO-IR
        frame its labels refer to, then one line per parent irrep, e.g.
        ``R4+ (R, dim 3) -> GM1+ [A1g] + GM5+ [Eg]``.
    """
    lines = ["* Subduction to the Gamma point of H *"]
    if not rows:
        return "\n".join(lines)
    first = rows[0]
    lines.append(f"H = {first.child_symbol} (No. {first.child_number}); Gamma "
                 "labels in the ISO-IR frame of H:")
    if first.basis is not None:
        basis = ", ".join(_format_vector(row) for row in first.basis)
        lines.append(f"basis (parent conventional units): {basis}")
        lines.append(f"origin: {_format_vector(first.origin)}")
    for row in rows:
        terms = [_term_text(name, n, row.mulliken) for name, n in row.gamma_part]
        if row.nongamma_components:
            terms.append(f"{row.nongamma_components} component"
                         f"{'s' if row.nongamma_components > 1 else ''} at non-Gamma "
                         "k of H")
        lines.append(f"  {row.label} ({row.kname}, dim {row.dimension}) -> "
                     + " + ".join(terms))
    if any(row.primary for row in rows):
        lines.append(
            "note: GM1 of H occurs in a primary irrep once per free parameter of "
            "the direction"
        )
    return "\n".join(lines)


def format_isotropy_block(report: _SubductionReport) -> str:
    """The ``* Supergroup *`` and ``* Isotropy subgroup *`` blocks, with the
    lines ``crystod-group --parent --order-parameter`` prints."""
    from .isotropy_subgroup import _format_setting_value

    algebra, info = report.algebra, report.info
    lines = [
        "* Supergroup *",
        f"{algebra.sg_type.international_short} (No. {algebra.sg_type.number})",
        "",
        "* Isotropy subgroup *",
        f"{report.header} -> {info.international_short} (No. {info.number})",
        f"cell size {report.size}, index {report.index}",
        "sublattice basis (parent primitive units): "
        + ", ".join("(" + ",".join(str(int(x)) for x in row) + ")" for row in report.B),
    ]
    if report.setting is not None:
        basis, origin = report.setting
        rows = ", ".join(
            "(" + ",".join(_format_setting_value(x) for x in row) + ")" for row in basis
        )
        lines.append(f"conventional basis (parent conventional units): {rows}")
        lines.append(
            "origin: (" + ",".join(_format_setting_value(x) for x in origin) + ")"
        )
    return "\n".join(lines)


# ------------------------------------------- (b) compatibility along a line


def _reciprocal_metric(algebra) -> np.ndarray:
    """Metric of the reciprocal lattice of a lattice with the full point
    symmetry (the invariant lattice of the isotropy machinery)."""
    g0 = np.diag([1.0, 1.07, 1.13])
    g = sum(W.T @ g0 @ W for W in np.asarray(algebra.rotations, dtype=float))
    return np.linalg.inv(g / algebra.n_ops)


def _labeler_context(algebra):
    from .isoir import get_cached_labeler

    inputs = algebra._isoir_labeler_inputs()
    if inputs is None:
        raise SystemExit("ERROR: the ISO-IR labeler is not available for space group "
                         f"{algebra.sg_type.number}.")
    cell, rotations, translations = inputs
    labeler = get_cached_labeler(algebra.sg_type.number, cell, 1e-5)
    if labeler is None:
        raise SystemExit("ERROR: the ISO-IR labeler is not available for space group "
                         f"{algebra.sg_type.number}.")
    return labeler, rotations, translations


def _segments(algebra, labeler, name0: str, name1: str) -> list[_Segment]:
    """Every symmetry line from the tabulated k0 to an arm (+ G) of the star
    of k1, with its first interior grid point and little group."""
    from .spacegroup_product import DEN

    metric = _reciprocal_metric(algebra)
    M_inv = np.linalg.inv(algebra.primitive_matrix)
    tabulated = set()
    for kname in algebra.k_by_kname:
        arms, _ = algebra.star(kname)
        tabulated |= {tuple(int(x) for x in np.asarray(arm) % DEN) for arm in arms}
    k0 = np.asarray(algebra.k_by_kname[name0], dtype=np.int64)
    little0 = set(algebra.little_group(k0))
    arms1, _ = algebra.star(name1)
    found: dict[tuple, _Segment] = {}
    names: dict[tuple, str | None] = {}  # one k-type lookup per star (slow)
    for arm in arms1:
        for shift in np.ndindex(3, 3, 3):
            k1 = np.asarray(arm, dtype=np.int64) + DEN * (np.asarray(shift) - 1)
            d = k1 - k0
            if not d.any():
                continue
            ops = [
                i for i in sorted(little0)
                if np.array_equal(d @ algebra.inverse_rotations[i], d)
            ]
            steps = int(np.gcd.reduce(np.abs(d)))
            unit = d // steps
            for j in range(1, steps):
                kdelta = k0 + unit * j
                if tuple(int(x) for x in kdelta % DEN) in tabulated:
                    continue
                if sorted(algebra.little_group(kdelta)) != ops:
                    continue
                star_arms, _ = algebra._star_of_vector(kdelta % DEN)
                star = min(tuple(int(x) for x in arm) for arm in star_arms)
                if star not in names:
                    names[star] = labeler.kpoint_name((kdelta / DEN) @ M_inv)
                name = names[star]
                key = (name, tuple(int(x) for x in k1))
                # dimension of the k subspace fixed by the little co-group:
                # 1 for a symmetry line, 2 for a plane, 3 for a general point
                Q = np.mean([np.asarray(algebra.inverse_rotations[i], dtype=float)
                             for i in ops], axis=0)
                found[key] = _Segment(
                    name0, name1, k0, k1, kdelta, ops, name or "?",
                    float(np.sqrt(d @ metric @ d)) / DEN,
                    int(np.linalg.matrix_rank(Q, tol=1e-6)),
                )
                break
    return list(found.values())


def _segment_rank(segment: _Segment, tabulated_k1) -> tuple:
    k1 = segment.k1
    return (
        -len(segment.ops),
        round(segment.length, 6),
        0 if np.array_equal(k1, tabulated_k1) else 1,
        int(np.sum(k1 < 0)),
        tuple(int(-x) for x in k1),
    )


def _choose_segments(algebra, labeler, name0, names1, line, strict):
    chosen = []
    available = {}
    for name1 in names1:
        segments = _segments(algebra, labeler, name0, name1)
        available[name1] = sorted({s.line for s in segments})
        if line is not None:
            segments = [s for s in segments if s.line.upper() == line.upper()]
        elif not strict:
            segments = [s for s in segments if s.fixed_dimension == 1]
        if not segments:
            continue
        tabulated = np.asarray(algebra.k_by_kname[name1], dtype=np.int64)
        chosen.append(min(segments, key=lambda s: _segment_rank(s, tabulated)))
    return chosen, available


def _relations_of_segment(algebra, labeler, conv_R, conv_t, segment) -> list[Compatibility]:
    from .spacegroup_product import DEN, SIGMA

    k_conv = (segment.kdelta / DEN) @ np.linalg.inv(algebra.primitive_matrix)
    little_R = [conv_R[i] for i in segment.ops]
    little_t = [conv_t[i] for i in segment.ops]
    records = []
    for name, k_end in ((segment.name0, segment.k0), (segment.name1, segment.k1)):
        delta = segment.kdelta - k_end
        phases = np.array([
            np.exp(SIGMA * 2j * np.pi * float(delta @ algebra.translations[i]) / DEN**2)
            for i in segment.ops
        ])
        for irrep in algebra.irreps_by_kname[name]:
            try:
                arms, C = algebra.induced_characters(irrep)
            except SystemExit as exc:
                raise SystemExit(
                    f"ERROR: the small characters of {irrep.name} are not available "
                    f"({str(exc).removeprefix('ERROR: ')})"
                ) from None
            matches = [a for a, arm in enumerate(arms)
                       if np.all((np.asarray(arm) - k_end) % DEN == 0)]
            if not matches:
                raise RuntimeError(f"internal error: {segment.name1} arm not in the star.")
            a = matches[0]
            identity = algebra._rotation_index[algebra._key(np.eye(3))]
            dimension = int(round(C[identity, a].real))
            characters = np.array([C[i, a] for i in segment.ops]) * phases
            result = labeler.decompose_characters(k_conv, little_R, little_t, characters)
            if result is None:
                raise RuntimeError(
                    f"internal error: {irrep.name} does not decompose into small "
                    f"irreps of the line {segment.line}."
                )
            total = sum(n * dim for _, n, dim in result[0])
            if total != dimension:
                raise RuntimeError(
                    f"internal error: the line irreps of {irrep.name} add up to "
                    f"{total} dimensions, the irrep has {dimension}."
                )
            records.append(Compatibility(
                k0_label=name,
                k0_irrep=irrep.name,
                line_label=result[1],
                line_irreps=[(label, int(n)) for label, n, _ in result[0]],
                dimension=dimension,
                k_point=_k_fractions(k_end),
                k_delta=_k_fractions(segment.kdelta),
                fixed_dimension=segment.fixed_dimension,
            ))
    return records


def compatibility_relations(sg, k0, k1=None, line=None) -> list[Compatibility]:
    """Compatibility relations between special points along symmetry lines.

    The computation behind ``crystod-group --correlate --sg SG --kpoint K0
    K1 [--line L]``.

    Args:
        sg: Space-group number or international symbol.
        k0: The first special point: ISO-IR name (``"GM"``, ``"X"``; ``G``
            and ``GAMMA`` mean ``GM``) or three primitive coordinates.
        k1: The second special point, in the same forms; ``None`` takes
            every other tabulated special point joined to k0 by a symmetry
            line.
        line: ISO-IR name of the line (``"DT"``) when several lines join the
            two points; ``None`` takes the line with the largest little
            group (then the shortest).

    Returns:
        ``Compatibility`` records: for every line, the small irreps at k0
        and then those at the other end, in table order.

    Raises:
        SystemExit: Unknown space group or k point, the same point twice,
            or no symmetry line (of the requested name) between the points
            (``ValueError`` through ``crystod.group``).
        RuntimeError: An internal consistency check failed (a restriction
            that does not decompose, or dimensions that do not add up).

    Example:
        >>> from crystod import group
        >>> rows = group.compatibility_relations("Pm-3m", "GM", "X")
        >>> [(r.k0_irrep, r.line_irreps) for r in rows if r.k0_irrep == "GM4-"]
        [('GM4-', [('DT1', 1), ('DT5', 1)])]
    """
    from .isotropy_subgroup import resolve_special_kpoint
    from .spacegroup_product import SpaceGroupIrrepAlgebra

    algebra = SpaceGroupIrrepAlgebra(str(sg))
    name0 = resolve_special_kpoint(algebra, k0)
    labeler, conv_R, conv_t = _labeler_context(algebra)
    group_text = (f"{algebra.sg_type.international_short} "
                  f"(No. {algebra.sg_type.number})")
    if k1 is not None:
        name1 = resolve_special_kpoint(algebra, k1)
        if name1 == name0:
            raise SystemExit(f"ERROR: give two different special points (both are {name0}).")
        names1 = [name1]
    else:
        names1 = [name for name in algebra.k_by_kname if name != name0]
    segments, available = _choose_segments(
        algebra, labeler, name0, names1, line, strict=k1 is not None
    )
    if not segments:
        if k1 is not None:
            lines = ", ".join(available[names1[0]]) or "none"
            wanted = f"line named {line}" if line is not None else "symmetry line"
            raise SystemExit(
                f"ERROR: no {wanted} joins {name0} and {names1[0]} in {group_text}; "
                f"the segments between them lie on: {lines}."
            )
        wanted = f" named {line}" if line is not None else ""
        raise SystemExit(f"ERROR: no symmetry line{wanted} starts at {name0} in {group_text}.")
    records = []
    for segment in segments:
        records.extend(_relations_of_segment(algebra, labeler, conv_R, conv_t, segment))
    return records


def format_compatibility(records: list[Compatibility]) -> str:
    """The ``* Compatibility relations *`` block as text.

    Args:
        records: The records returned by ``compatibility_relations``.

    Returns:
        The text block (no trailing newline): per line a header naming the
        line, its end points and kdelta, then one line per small irrep of
        each end point, e.g. ``GM4- -> DT1 + DT5``.
    """
    lines = ["* Compatibility relations *"]
    groups: dict[tuple, list[Compatibility]] = {}
    for record in records:
        groups.setdefault(
            (record.line_label, record.k_delta, record.fixed_dimension), []
        ).append(record)
    kinds = {1: "line", 2: "plane", 3: "general segment"}
    for (line_label, k_delta, fixed), group in groups.items():
        ends: dict[str, tuple] = {}
        for record in group:
            ends.setdefault(record.k0_label, record.k_point)
        between = " and ".join(f"{name} {_k_text(k)}" for name, k in ends.items())
        lines.append(f"{kinds.get(fixed, 'line')} {line_label} between {between}, labeled at "
                     f"kdelta = {_k_text(k_delta)} (primitive basis)")
        width = max(len(record.k0_irrep) for record in group)
        for record in group:
            terms = " + ".join(
                f"{n} {label}" if n > 1 else label for label, n in record.line_irreps
            )
            lines.append(f"  {record.k0_irrep:<{width}} (dim {record.dimension}) -> {terms}")
    return "\n".join(lines)


# ------------------------------------- (a) point-group correlation tables

# the two settings phonopy tabulates for these point groups, by the full
# symbol of the setting (character_table[pg][0] and [1])
_SETTING_NAMES = {
    "-42m": ("-42m", "-4m2"),
    "3m": ("3m1", "31m"),
    "-3m": ("-3m1", "-31m"),
    "-6m2": ("-6m2", "-62m"),
}

# the class of phonopy's table of H whose axis is printed as the axis of an
# orientation (the z axis of the table); None for 1, -1 and the cubic groups
_PRINCIPAL_CLASS = {
    "2": "C2", "m": "sgh", "2/m": "C2", "222": "C2", "mm2": "C2", "mmm": "C2",
    "4": "C4", "-4": "S4", "4/m": "C4", "422": "C4", "4mm": "C4", "-42m": "S4",
    "4/mmm": "C4", "3": "C3", "-3": "S6", "32": "C3", "3m": "C3", "-3m": "S6",
    "6": "C6", "-6": "S3", "6/m": "C6", "622": "C6", "6mm": "C6", "-6m2": "S3",
    "6/mmm": "C6",
}


@dataclass
class CorrelationTable:
    """The correlation table of a point group G and one subgroup type H.

    Attributes:
        parent: The point group G (phonopy's symbol, e.g. ``"m-3m"``).
        subgroup: The subgroup type H (``"4/mmm"``).
        orientations: One ``(description, rotations)`` pair per conjugacy
            class of subgroups of type H in G (the inequivalent
            orientations): the column header (``"4/mmm (4 || z)"``) and the
            rotation matrices of the representative subgroup, in the axes
            of the table of G, as an ``(|H|, 3, 3)`` integer array.
        rows: ``{irrep of G: decompositions}`` in the order of the table of
            G, with one decomposition per orientation: a list of
            ``(Mulliken symbol of H, multiplicity)`` pairs in the order of
            the table of H.  A complex-conjugate pair that phonopy's table
            merges into one real ``E`` keeps that symbol.
        conjugates: Number of subgroups conjugate to each representative.
        settings: The setting of phonopy's table of H each orientation is
            labeled with (``"-4m2"``; ``""`` for the groups with one
            setting).
        axes: Per orientation, the classes of H that fix its orientation
            about the principal axis, with their elements in the axes of
            G (``"C2' = 2_x, 2_y; C2'' = 2_[110], 2_[1-10]"``); the
            symbols B1/B2/B3 refer to them.
        parent_order: Order of G.
        subgroup_order: Order of H.
        hexagonal_axes: ``True`` when the table of G uses hexagonal axes
            (trigonal and hexagonal G): directions are then ``[uvw]`` in
            those axes, otherwise ``x``/``y``/``z`` along the cell axes.
    """

    parent: str
    subgroup: str
    orientations: list
    rows: dict
    conjugates: list = field(default_factory=list)
    settings: list = field(default_factory=list)
    axes: list = field(default_factory=list)
    parent_order: int = 0
    subgroup_order: int = 0
    hexagonal_axes: bool = False


def _point_group_table(pg: str, setting: int = 0) -> dict:
    """phonopy's character table of a point group (one setting)."""
    from phonopy.phonon.character_table import character_table

    if pg not in character_table:
        available = ", ".join(character_table)
        raise SystemExit(f'ERROR: "{pg}" is not in the point groups.\nChoose from: {available}')
    return character_table[pg][setting]


def _table_classes(table: dict) -> list[str]:
    names = table["rotation_list"]
    return [names] if isinstance(names, str) else list(names)


def _table_operations(table: dict) -> list[tuple[np.ndarray, str]]:
    """``(rotation, class name)`` for every element of a table, class by class."""
    return [
        (np.array(rotation, dtype=int), name)
        for name in _table_classes(table)
        for rotation in table["mapping_table"][name]
    ]


def _table_characters(table: dict) -> dict:
    """``{irrep: {class: character}}`` of a table (scalars of point group 1
    made vectors)."""
    classes = _table_classes(table)
    return {
        irrep: dict(zip(classes, np.atleast_1d(np.asarray(values, dtype=complex))))
        for irrep, values in table["character_table"].items()
    }


def _closure(seed, table_product, identity: int) -> frozenset:
    group = set(seed) | {identity}
    while True:
        grown = group | {int(table_product[a, b]) for a in group for b in group}
        if grown == group:
            return frozenset(group)
        group = grown


def _all_subgroups(table_product, identity: int) -> set:
    """Every subgroup of a finite group given by its multiplication table,
    grown from the cyclic subgroups by adjoining one element at a time."""
    n = len(table_product)
    layer = {_closure([a], table_product, identity) for a in range(n)}
    found = set(layer)
    while layer:
        larger = set()
        for group in layer:
            for a in range(n):
                if a not in group:
                    joined = _closure(group | {a}, table_product, identity)
                    if joined not in found:
                        found.add(joined)
                        larger.add(joined)
        layer = larger
    return found


def _integer_direction(vector) -> np.ndarray:
    """The shortest integer vector along ``vector``, first nonzero entry > 0."""
    v = np.asarray(vector, dtype=float)
    v = v / np.abs(v).max()
    for scale in range(1, 13):
        w = v * scale
        if np.allclose(w, np.rint(w), atol=1e-6):
            w = np.rint(w).astype(int)
            break
    else:  # pragma: no cover - lattice directions of point-group axes are short
        raise RuntimeError("internal error: an axis is not a lattice direction.")
    w = w // np.gcd.reduce(np.abs(w))
    if w[np.flatnonzero(w)[0]] < 0:
        w = -w
    return w


def _element_kind(R) -> tuple[str, np.ndarray | None]:
    """Hermann-Mauguin symbol (``4``, ``-4``, ``m``, ...) and axis of a
    rotation in lattice axes (for a mirror the normal: the direction it
    reverses); the axis is ``None`` for 1 and -1."""
    R = np.asarray(R, dtype=float)
    det = int(round(np.linalg.det(R)))
    proper = R * det
    order = {3: 1, -1: 2, 0: 3, 1: 4, 2: 6}[int(round(np.trace(proper)))]
    if det > 0:
        symbol = str(order)
    else:
        symbol = {1: "-1", 2: "m", 3: "-3", 4: "-4", 6: "-6"}[order]
    if order == 1:
        return symbol, None
    w, v = np.linalg.eig(proper)
    axis = v[:, int(np.argmin(np.abs(w - 1.0)))].real
    return symbol, _integer_direction(axis)


def _direction_text(axis, orthogonal: bool) -> str:
    """``x``/``y``/``z`` for the axes of an orthogonal table, else ``[uvw]``."""
    if orthogonal and np.count_nonzero(axis) == 1:
        return "xyz"[int(np.flatnonzero(axis)[0])]
    return "[" + "".join(str(int(x)) for x in axis) + "]"


def _element_text(R, orthogonal: bool) -> str:
    symbol, axis = _element_kind(R)
    if axis is None:
        return symbol
    direction = _direction_text(axis, orthogonal)
    return f"{symbol}_{direction}"


def _axis_key(axis) -> tuple:
    """Order of preference of axes: z, x, y, then face and body diagonals."""
    if axis is None:
        return (0,)
    a = np.abs(axis)
    nonzero = int(np.count_nonzero(axis))
    return (
        nonzero,
        -int(a[2]) if nonzero == 1 else 0,
        int(a[2] != 0) if nonzero == 2 else 0,
        tuple(-int(x) for x in a),
        tuple(-int(x) for x in axis),
    )


def _generators(rotations: list[np.ndarray]) -> list[np.ndarray]:
    """A small generating set of a matrix group (greedy, in list order)."""
    keys = {R.tobytes() for R in rotations}
    span = {np.eye(3, dtype=int).tobytes()}
    generators: list[np.ndarray] = []
    for R in rotations:
        if R.tobytes() in span:
            continue
        generators.append(R)
        members = [np.frombuffer(k, dtype=int).reshape(3, 3) for k in span] + [R]
        while True:
            grown = {(A @ B).tobytes() for A in members for B in members} | {
                M.tobytes() for M in members
            }
            if len(grown) == len(members):
                break
            members = [np.frombuffer(k, dtype=int).reshape(3, 3) for k in grown]
        span = {M.tobytes() for M in members}
        if span == keys:
            break
    return generators


def _conjugated_classes(rotations, X, table_ops) -> list[str] | None:
    """Class names of ``X^-1 R X`` in a table, or ``None`` when some image
    is not an element of it."""
    lookup = {R.tobytes(): name for R, name in table_ops}
    X_inv = np.linalg.inv(X)
    classes = []
    for R in rotations:
        image = X_inv @ R @ X
        if not np.allclose(image, np.rint(image), atol=1e-6):
            return None
        name = lookup.get(np.rint(image).astype(int).tobytes())
        if name is None:
            return None
        classes.append(name)
    if len(set(lookup)) != len(rotations):
        return None
    return classes


def _solve_conjugation(rotations, table_ops) -> list[list[str]]:
    """Class names of the elements of a matrix group under every isomorphism
    ``R -> X^-1 R X`` onto the elements of a table (generators mapped to
    elements of the same kind, ``R X = X T`` solved for X), one list per
    distinct assignment."""
    import itertools

    generators = _generators(rotations)

    def kind(R):
        return (round(float(np.linalg.det(R))), int(round(float(np.trace(R)))))

    candidates = [[T for T, _ in table_ops if kind(T) == kind(g)] for g in generators]
    eye = np.eye(3)
    weights = np.array([1.0, 0.37, 0.61, 0.83, 0.29, 0.47, 0.71, 0.19, 0.53])
    found: list[list[str]] = []
    for images in itertools.product(*candidates):
        system = np.vstack([
            np.kron(eye, g) - np.kron(np.asarray(t, dtype=float).T, eye)
            for g, t in zip(generators, images)
        ])
        _, s, vh = np.linalg.svd(system)
        null = vh[np.sum(s > 1e-8):]
        if not len(null):
            continue
        X = (weights[: len(null)] @ null).reshape(3, 3, order="F")
        if abs(np.linalg.det(X)) < 1e-8:
            continue
        classes = _conjugated_classes(rotations, X, table_ops)
        if classes is not None and classes not in found:
            found.append(classes)
    return found


def _assignment_key(rotations, classes, subgroup: str, setting: int) -> tuple:
    """Preference among class assignments: the principal axis first (z
    before x before diagonals), then the axes of the other classes in the
    order of the table."""
    members: dict[str, list] = {}
    for R, name in zip(rotations, classes):
        members.setdefault(name, []).append(_axis_key(_element_kind(R)[1]))
    principal = _PRINCIPAL_CLASS.get(subgroup)
    head = min(members[principal]) if principal in members else (0,)
    names = _table_classes(_point_group_table(subgroup, setting))
    return (head, tuple(min(members.get(name, [(0,)])) for name in names))


def _embedding(rotations, subgroup: str) -> tuple[int, list[str]] | None:
    """The setting of phonopy's table of H and the class of every element
    of a subgroup of type H given in the axes of G: the axes of G as they
    are; otherwise spglib's transformation to the standard orientation and
    the conjugations found for the generators (needed for a subgroup that
    spglib leaves in a non-standard orientation, such as mm2 with the
    twofold axis along x), of which the one whose principal axis comes
    first (z before x before diagonals, ``_assignment_key``) is kept, a tie
    going to spglib's (so the diagonal mmm of 4/mmm keeps C2z = 2_z)."""
    import spglib

    settings = range(len(_all_tables(subgroup)))
    tables = [_table_operations(_point_group_table(subgroup, s)) for s in settings]
    for s in settings:
        classes = _conjugated_classes(rotations, np.eye(3), tables[s])
        if classes is not None:
            return s, classes
    # candidates: spglib's transformation first, then the solver's
    # assignments in order of preference; the principal axis decides (z
    # before x before diagonals) and a tie keeps the earlier candidate
    candidates: list[tuple[int, list[str]]] = []
    _, _, P = spglib.get_pointgroup(np.asarray(rotations, dtype="intc"))
    P = np.asarray(P, dtype=float)
    if abs(np.linalg.det(P)) > 1e-8:
        for s in settings:
            classes = _conjugated_classes(rotations, P, tables[s])
            if classes is not None:
                candidates.append((s, classes))
                break
    for s in settings:
        found = _solve_conjugation(rotations, tables[s])
        if found:
            found.sort(key=lambda c: _assignment_key(rotations, c, subgroup, s))
            candidates += [(s, c) for c in found]
            break
    if not candidates:
        return None
    return min(candidates, key=lambda sc: _assignment_key(rotations, sc[1], subgroup, sc[0])[0])


def _all_tables(pg: str) -> list:
    from phonopy.phonon.character_table import character_table

    _point_group_table(pg)
    return character_table[pg]


@dataclass
class _Census:
    """The subgroups of a point group, by type and conjugacy class."""

    ops: list
    orthogonal: bool
    by_type: dict


@functools.lru_cache(maxsize=None)
def _subgroup_census(pg: str) -> _Census:
    """Every subgroup of the point group (phonopy's table, setting 0),
    identified with ``spglib.get_pointgroup`` and grouped into conjugacy
    classes under the group: ``{type: [[subgroup, ...] per class]}`` with
    the types in the order of phonopy's tables, each subgroup a frozenset
    of indices into ``ops``."""
    import spglib
    from phonopy.phonon.character_table import character_table

    ops = _table_operations(_point_group_table(pg))
    n = len(ops)
    index = {R.tobytes(): i for i, (R, _) in enumerate(ops)}
    product = np.array([[index[(ops[a][0] @ ops[b][0]).tobytes()] for b in range(n)]
                        for a in range(n)])
    identity = index[np.eye(3, dtype=int).tobytes()]
    inverse = [int(np.flatnonzero(product[a] == identity)[0]) for a in range(n)]
    found: dict[str, dict[tuple, list]] = {}
    for group in sorted(_all_subgroups(product, identity), key=sorted):
        rotations = np.asarray([ops[i][0] for i in sorted(group)], dtype="intc")
        kind = spglib.get_pointgroup(rotations)[0].strip()
        canonical = min(
            tuple(sorted(int(product[product[g, i], inverse[g]]) for i in group))
            for g in range(n)
        )
        found.setdefault(kind, {}).setdefault(canonical, []).append(group)
    by_type = {kind: list(found[kind].values()) for kind in character_table if kind in found}
    orthogonal = all(np.array_equal(R.T @ R, np.eye(3, dtype=int)) for R, _ in ops)
    return _Census(ops, orthogonal, by_type)


@dataclass
class _Orientation:
    rotations: list
    setting: int
    classes: list
    conjugates: int
    principal: tuple
    identity_axes: bool


def _orientation_of(group, ops, subgroup) -> _Orientation | None:
    rotations = [ops[i][0] for i in sorted(group)]
    found = _embedding(rotations, subgroup)
    if found is None:
        return None
    setting, classes = found
    principal_class = _PRINCIPAL_CLASS.get(subgroup)
    principal = (None, None)
    if principal_class is not None:
        R = next(R for R, name in zip(rotations, classes) if name == principal_class)
        principal = _element_kind(R)
    table = _table_operations(_point_group_table(subgroup, setting))
    identity_axes = _conjugated_classes(rotations, np.eye(3), table) is not None
    return _Orientation(rotations, setting, classes, 0, principal, identity_axes)


def _secondary_classes(orientation: _Orientation, subgroup: str,
                       orthogonal: bool) -> list[tuple[str, list[str]]]:
    """The classes of H that fix its orientation about the principal axis,
    as ``(class name, element symbols)``: those of twofold rotations off the
    principal axis, else those of mirrors whose normal is off it
    (``("C2'", ["2_x", "2_y"])``)."""
    if subgroup not in _PRINCIPAL_CLASS:
        return []
    axis = orientation.principal[1]
    groups: dict[str, list] = {}
    for R, name in zip(orientation.rotations, orientation.classes):
        groups.setdefault(name, []).append(R)
    table = _point_group_table(subgroup, orientation.setting)

    def pick(symbol):
        chosen = []
        for name in _table_classes(table):
            members = groups.get(name, [])
            kinds = [_element_kind(R) for R in members]
            if members and all(k[0] == symbol for k in kinds) and not any(
                axis is not None and np.array_equal(k[1], axis) for k in kinds
            ):
                ordered = sorted(members, key=lambda R: _axis_key(_element_kind(R)[1]))
                chosen.append((name, [_element_text(R, orthogonal) for R in ordered]))
        return chosen

    return pick("2") or pick("m")


def correlation_table(pg: str, subgroup: str) -> CorrelationTable:
    """The correlation table of a point group and a subgroup type.

    The computation behind ``crystod-group --correlate --pg G --subgroup H``:
    every subgroup of G (from phonopy's table, order at most 48) is generated, the
    ones of type H (``spglib.get_pointgroup``) are grouped into conjugacy
    classes under G, and the irreps of G are restricted to one
    representative of each class.  The restricted characters are reduced
    with phonopy's table of H after the elements are carried onto the
    elements of that table (in the axes of G when they match one of its
    settings, else with spglib's transformation to the standard
    orientation, else by a conjugation found for the generators).

    Args:
        pg: The point group G, as phonopy names it (``"m-3m"``, ``"6/mmm"``).
        subgroup: The subgroup type H (``"4/mmm"``, ``"3m"``).

    Returns:
        A ``CorrelationTable`` with one orientation per conjugacy class of
        subgroups of type H, ordered by the principal axis (z first).

    Raises:
        SystemExit: Unknown point group, or H not a subgroup of G
            (``ValueError`` through ``crystod.group``).
        RuntimeError: A restriction does not decompose into the irreps of H
            or the dimensions do not add up (internal check).

    Example:
        >>> from crystod import group
        >>> table = group.correlation_table("m-3m", "4/mmm")
        >>> [header for header, _ in table.orientations]
        ['4/mmm (4 || z)']
        >>> table.rows["T2g"]
        [[('B2g', 1), ('Eg', 1)]]
    """
    parent_table = _point_group_table(pg)
    _point_group_table(subgroup)
    census = _subgroup_census(pg)
    ops, n, orthogonal = census.ops, len(census.ops), census.orthogonal
    classes = census.by_type.get(subgroup)
    if not classes:
        raise SystemExit(f"ERROR: {subgroup} is not a subgroup of {pg} (its subgroup "
                         f"types: {', '.join(census.by_type)}).")

    orientations: list[_Orientation] = []
    for members in classes:
        candidates = [o for o in (_orientation_of(g, ops, subgroup) for g in members) if o]
        if not candidates:
            raise RuntimeError(f"internal error: a subgroup of type {subgroup} of {pg} "
                               "does not match phonopy's table of it.")
        best = min(candidates, key=lambda o: (
            _axis_key(o.principal[1]), 0 if o.identity_axes else 1,
            _assignment_key(o.rotations, o.classes, subgroup, o.setting)))
        best.conjugates = len(members)
        orientations.append(best)
    orientations.sort(key=lambda o: (
        _axis_key(o.principal[1]), 0 if o.identity_axes else 1, o.setting,
        _assignment_key(o.rotations, o.classes, subgroup, o.setting)))

    parent_chars = _table_characters(parent_table)
    parent_class = {R.tobytes(): name for R, name in ops}
    results: dict[str, list] = {irrep: [] for irrep in parent_chars}
    for orientation in orientations:
        sub_chars = _table_characters(_point_group_table(subgroup, orientation.setting))
        dims = {irrep: int(round(chars[_table_classes(
            _point_group_table(subgroup, orientation.setting))[0]].real))
            for irrep, chars in sub_chars.items()}
        for irrep, chars in parent_chars.items():
            restricted = np.array([chars[parent_class[R.tobytes()]]
                                   for R in orientation.rotations])
            terms = []
            total = 0
            for name, sub in sub_chars.items():
                chi = np.array([sub[c] for c in orientation.classes])
                value = complex(np.vdot(chi, restricted) / np.vdot(chi, chi))
                count = int(round(value.real))
                if abs(value - count) > 1e-6:
                    raise RuntimeError(f"internal error: {irrep} of {pg} restricted to "
                                       f"{subgroup} gives a non-integral multiplicity.")
                if count:
                    terms.append((name, count))
                    total += count * dims[name]
            dimension = int(round(chars[_table_classes(parent_table)[0]].real))
            if total != dimension:
                raise RuntimeError(f"internal error: {irrep} of {pg} (dim {dimension}) "
                                   f"restricted to {subgroup} adds up to {total}.")
            results[irrep].append(terms)

    headers = []
    for orientation in orientations:
        symbol, axis = orientation.principal
        parts = []
        if axis is not None:
            direction = _direction_text(axis, orthogonal)
            parts.append(f"{symbol} || {direction}" if symbol != "m"
                         else f"m_{direction}")
        headers.append(parts)
    # the setting describes the orientation only in the axes of G (a
    # transformed table is just phonopy's index): name it there, or when
    # two columns differ by nothing else
    for j, orientation in enumerate(orientations):
        if subgroup in _SETTING_NAMES and (orientation.identity_axes or any(
                headers[k][:1] == headers[j][:1] and o.setting != orientation.setting
                for k, o in enumerate(orientations))):
            headers[j] = headers[j] + [
                f"{_SETTING_NAMES[subgroup][orientation.setting]} setting"]
    secondary = [_secondary_classes(o, subgroup, orthogonal) for o in orientations]
    axes = ["; ".join(f"{name} = {', '.join(elements)}" for name, elements in found)
            for found in secondary]
    texts = [f"{subgroup} ({', '.join(parts)})" if parts else subgroup for parts in headers]
    for j, text in enumerate(texts):
        # columns with the same principal axis: name the first element of
        # the first class that fixes the orientation
        if texts.count(text) > 1 and secondary[j]:
            headers[j] = headers[j][:1] + [secondary[j][0][1][0]] + headers[j][1:]
    texts = [f"{subgroup} ({', '.join(parts)})" if parts else subgroup for parts in headers]

    return CorrelationTable(
        parent=pg,
        subgroup=subgroup,
        orientations=[(text, np.asarray(o.rotations, dtype=int))
                      for text, o in zip(texts, orientations)],
        rows=results,
        conjugates=[o.conjugates for o in orientations],
        settings=[_SETTING_NAMES[subgroup][o.setting] if subgroup in _SETTING_NAMES else ""
                  for o in orientations],
        axes=axes,
        parent_order=n,
        subgroup_order=len(orientations[0].rotations),
        hexagonal_axes=not orthogonal,
    )


def _decomposition_text(terms) -> str:
    return " + ".join(f"{n} {name}" if n > 1 else name for name, n in terms)


def format_correlation_table(table: CorrelationTable) -> str:
    """The ``* Point group *``, ``* Subgroup *`` and ``* Correlation table *``
    blocks of ``--correlate --pg G --subgroup H`` as text.

    Args:
        table: The record returned by ``correlation_table``.

    Returns:
        The text (no leading blank line, no trailing newline): G and its
        order, H with its orientations (number of conjugate subgroups and
        the classes that fix the orientation), the table with one row per
        irrep of G and one column per orientation, and a note on the axes.
    """
    count = len(table.orientations)
    index = table.parent_order // table.subgroup_order
    lines = [
        "* Point group *",
        f"{table.parent} (order {table.parent_order})",
        "",
        "* Subgroup *",
        f"{table.subgroup} (order {table.subgroup_order}, index {index}): {count} "
        f"inequivalent orientation{'s' if count > 1 else ''}",
    ]
    for (header, _), conjugates, axes in zip(table.orientations, table.conjugates, table.axes):
        text = f"  {header}: {conjugates} conjugate subgroup{'s' if conjugates > 1 else ''}"
        if axes:
            text += f"; {axes}"
        lines.append(text)
    grid = [["irrep"] + [header for header, _ in table.orientations]]
    for irrep, decompositions in table.rows.items():
        grid.append([irrep] + [_decomposition_text(terms) for terms in decompositions])
    widths = [max(len(row[j]) for row in grid) for j in range(len(grid[0]))]
    lines += ["", "* Correlation table *"]
    for row in grid:
        lines.append("  ".join(cell.ljust(width) for cell, width in
                               zip(row, widths)).rstrip())
    import textwrap

    frame = (f"directions [uvw] are in the hexagonal axes a, b, c of the table of "
             f"{table.parent}" if table.hexagonal_axes else
             f"x, y, z are the axes [100], [010], [001] of the table of {table.parent}")
    note = (f"note: {frame}; 2_d is the twofold rotation about d, m_d the mirror "
            "normal to d; the symbols of H are those of phonopy's table of H")
    if any(table.axes):
        note += (" with its classes as listed under *\x00Subgroup\x00* (B1/B2/B3 "
                 "follow them)")
    note += "."
    lines += [""] + [line.replace("\x00", " ") for line in
                     textwrap.wrap(note, width=78, subsequent_indent="      ")]
    return "\n".join(lines)


# ------------------------------------------------------------------ command


def main(argv: list[str] | None = None) -> None:
    """Command-line entry behind ``crystod-group --correlate``."""
    from .isotropy_subgroup import _protect_component_values

    parser = argparse.ArgumentParser(
        description="Correlation of irreps: point-group correlation tables, "
        "subduction to an isotropy subgroup, compatibility relations along a line."
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--pg", help="point group (correlation table).")
    target.add_argument("--parent", help="parent space group (subduction).")
    target.add_argument("--sg", help="space group (compatibility relations).")
    parser.add_argument("--subgroup", default=None)
    parser.add_argument("--irrep", nargs="+", default=None)
    parser.add_argument("--order-parameter", nargs="+", default=None)
    parser.add_argument("--irrep-list", nargs="+", default=None)
    parser.add_argument("--kpoint", nargs="+", default=None)
    parser.add_argument("--line", default=None)
    if argv is None:
        import sys

        argv = sys.argv[1:]
    args = parser.parse_args(_protect_component_values(list(argv)))

    try:
        if args.pg:
            if not args.subgroup:
                parser.error("--pg needs --subgroup.")
            table = correlation_table(args.pg, args.subgroup)
            print()
            print(format_correlation_table(table))
            return
        if args.parent:
            if not args.irrep or not args.order_parameter:
                parser.error("--parent needs --irrep and --order-parameter.")
            report = _subduction_report(
                args.parent, args.irrep, [value.strip() for value in args.order_parameter],
                args.irrep_list,
            )
            print()
            print(format_isotropy_block(report))
            print()
            print(format_subduction(report.rows))
            return
        if not args.kpoint or len(args.kpoint) > 2:
            parser.error("--sg needs --kpoint with one or two k-point names.")
        k1 = args.kpoint[1] if len(args.kpoint) == 2 else None
        records = compatibility_relations(args.sg, args.kpoint[0], k1, args.line)
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from None
    from .spacegroup_product import SpaceGroupIrrepAlgebra

    algebra = SpaceGroupIrrepAlgebra(args.sg)
    print()
    print("* Space group *")
    print(f"{algebra.sg_type.international_short} (No. {algebra.sg_type.number})")
    print()
    print(format_compatibility(records))


if __name__ == "__main__":
    main()
