"""Group-theory tools (the ``crystod-group`` domain).

Isotropy subgroups and their reverse lookup, the group-subgroup graph,
space-group and point-group irrep products, reducible-representation and
ligand-field decompositions, character tables, the Landau invariants of
space-group irreps, correlation tables, compatibility relations and
subductions, and property-tensor forms, all on ``crystod.group``.
"""

from __future__ import annotations

import contextlib
import re

from ..utils import (
    ORBITALS,
    blocking_tool,
    fenced,
    format_character,
    format_kpoint,
    is_point_group,
    markdown_table,
    parse_irrep_names,
    parse_isoir_labels,
    parse_kpoint,
    parse_orbital,
    parse_order_parameter,
    parse_point_group,
    parse_space_group,
    resolve_kpoint_label,
    special_kpoints,
)


# a bare k-point name of the ISO-IR tables (GM, R, X, PA, ...) in place of an
# irrep label asks for every irrep of that k point
_KPOINT_NAME = re.compile(r"[A-Z]{1,2}A?|GAMMA", re.IGNORECASE)


def _class_multiplicities(character_table: dict) -> list[int]:
    import numpy as np

    return [
        int(np.asarray(character_table["mapping_table"][name]).shape[0])
        for name in character_table["rotation_list"]
    ]


def _decomposition_line(counts: dict[str, int]) -> str:
    terms = [f"{count}({irrep})" for irrep, count in counts.items() if count > 0]
    return " + ".join(terms) if terms else "0 (the characters contain no irrep)"


def _raw_multiplicities(character_table: dict, characters: list[float]) -> dict[str, float]:
    """The unrounded multiplicities, to detect characters of no representation.

    The reduction formula of ``crystod.group.decompose_representation``:
    sum_C |C| chi_i(C) chi(C) divided by the norm sum_C |C| |chi_i(C)|^2 of
    each irrep, which is |G| for an ordinary irrep and 2|G| for an entry that
    the table stores as a real pair of complex-conjugate irreps (E of 4), so
    that such a pair counts once per real copy.
    """
    import numpy as np

    multiplicity = np.array(_class_multiplicities(character_table), dtype=float)
    reducible = np.array(characters, dtype=float)
    raw = {}
    for irrep, values in character_table["character_table"].items():
        chi = np.atleast_1d(np.real(np.asarray(values, dtype=complex)))
        norm = float(np.sum(multiplicity * chi * chi))
        raw[irrep] = float(np.sum(multiplicity * chi * reducible)) / norm
    return raw


@blocking_tool
def crystod_isotropy_subgroups(
    parent: str,
    irrep: str,
    order_parameter: str | None = None,
) -> str:
    """Which space groups result when a distortion of a given irrep condenses in a parent space group (the isotropy subgroups), e.g. which subgroups the R4+ octahedral-rotation mode of Pm-3m leads to.

    Answers "the soft mode at R has irrep R4+ (or R5-, M3+, GM4-, ...); what
    low-symmetry structures can it produce?": every order-parameter direction
    is enumerated with the subgroup it condenses into, its space-group number,
    the primitive-cell multiplication (size) and the index in the parent.
    This is ``crystod-group --parent PARENT --irrep IRREP``. The whole
    table is always returned (the CrystOD API has no maximum-index cut-off,
    so there is no max_index argument; the index is the primitive-cell
    multiplication times the point-group index, so it exceeds the order of
    the parent point group for irreps at k != 0). When the irrep is not
    known yet, give the bare k-point name instead of a label to get the
    subgroups of every irrep of that k point in one table
    (``crystod-group --parent PARENT --kpoint K``).

    Args:
        parent: Parent space group as an international symbol (``Pm-3m``,
            ``P6_3/mmc``, ``I4/mmm``) or number (``221``).
        irrep: Irrep label in ISO-IR notation: k-point name, index and parity
            sign, e.g. ``R4+``, ``GM4-``, ``M3+``, ``X5-`` (which labels a space
            group has: ``crystod_character_table``). Two labels separated by a
            space (``X3- X2+``) enumerate the subgroups of the coupled order
            parameter. A bare k-point name (``GM``, ``R``, ``X``, ``M``, ...;
            which names a space group has: ``crystod_special_kpoints``) lists
            the subgroups of every irrep of that k point.
        order_parameter: Optional direction of the order parameter with letters
            for free components and numbers for fixed ones, e.g. ``"0 0 a"`` or
            ``"a a 0"``; then only that direction is resolved and the
            conventional basis and origin of the subgroup in the parent setting
            are reported as well. Needs one irrep label (or coupled labels),
            not a k-point name.

    Returns:
        Markdown: a table with one row per order-parameter direction
        (direction, subgroup symbol, number, size, index, free parameters);
        with a k-point name, one row per irrep and direction.
    """
    from crystod import group

    space_group = parse_space_group(parent)
    if isinstance(irrep, str) and _KPOINT_NAME.fullmatch(irrep.strip()):
        return _kpoint_isotropy_subgroups(space_group, irrep.strip(), order_parameter)
    labels = parse_isoir_labels(irrep)
    direction = parse_order_parameter(order_parameter)
    sg_type = special_kpoints(space_group)[0]

    subgroups = group.isotropy_subgroups(
        space_group,
        labels if len(labels) > 1 else labels[0],
        direction,
        with_settings=direction is not None,
    )

    irrep_text = "+".join(labels)
    parent_text = f"{sg_type.international_short} (No. {sg_type.number})"
    lines = [f"## Isotropy subgroups of {irrep_text} in {parent_text}", ""]
    if direction is not None:
        lines[0] = f"## Isotropy subgroup of {irrep_text}({','.join(direction)}) in {parent_text}"
    rows = [
        (sub.direction, sub.symbol, sub.number, sub.size, sub.index, sub.n_free)
        for sub in subgroups
    ]
    lines.append(markdown_table(
        ("Direction", "Subgroup", "No.", "Size", "Index", "Free parameters"), rows
    ))
    lines.append("")
    lines.append(
        f"{len(subgroups)} isotropy subgroup(s). Size: primitive-cell multiplication "
        "relative to the parent; index: index of the subgroup in the parent; free "
        "parameters: independent order-parameter components (1 = a single "
        "amplitude fixes the structure)."
    )
    for sub in subgroups:
        if sub.basis is None or sub.origin is None:
            continue
        basis = "; ".join(
            "(" + ", ".join(f"{float(v) + 0.0:g}" for v in row) + ")" for row in sub.basis
        )
        origin = "(" + ", ".join(f"{float(v) + 0.0:g}" for v in sub.origin) + ")"
        lines.append("")
        lines.append(
            f"Conventional cell of {sub.symbol} in the parent setting: basis rows "
            f"{basis}; origin {origin} (parent conventional units)."
        )
    return "\n".join(lines)


def _kpoint_isotropy_subgroups(space_group: str, kpoint: str, order_parameter) -> str:
    """The subgroups of every irrep of one k point (``--parent SG --kpoint K``)."""
    from crystod import group

    if parse_order_parameter(order_parameter) is not None:
        raise ValueError(
            f'"{kpoint}" is a k-point name, which lists every irrep of that k '
            "point; order_parameter needs one irrep label such as "
            f"{kpoint.upper()}1 or R4+ (call again with only the k-point name "
            "to see the labels)."
        )
    enumerate_kpoint = getattr(group, "isotropy_subgroups_at_kpoint", None)
    if enumerate_kpoint is None:
        raise ValueError(
            "listing every irrep of a k point needs CrystOD >= 0.4.3; give one "
            f"irrep label of {kpoint.upper()} instead (crystod_character_table "
            "lists them)."
        )
    table = enumerate_kpoint(space_group, kpoint, with_settings=False)
    parent_text = f"{table.space_group} (No. {table.space_group_number})"
    lines = [
        f"## Isotropy subgroups of every irrep at {table.kpoint} in {parent_text}",
        "",
    ]
    rows = [
        (label, sub.direction, sub.symbol, sub.number, sub.size, sub.index, sub.n_free)
        for label, subgroups in table.items()
        for sub in subgroups
    ]
    lines.append(markdown_table(
        ("Irrep", "Direction", "Subgroup", "No.", "Size", "Index", "Free parameters"),
        rows,
    ))
    lines.append("")
    lines.append(
        f"{len(rows)} isotropy subgroup(s) of {len(table)} irrep(s) at "
        f"{table.kpoint} = {format_kpoint(table.coordinates)} (primitive basis, "
        f"star of {table.n_arms} arm(s)). Size: primitive-cell multiplication "
        "relative to the parent; index: index of the subgroup in the parent; free "
        "parameters: independent order-parameter components (1 = a single "
        "amplitude fixes the structure). Call again with one irrep label and an "
        "order_parameter for the conventional cell of a subgroup."
    )
    for label, reason in table.errors.items():
        lines.append("")
        lines.append(f"Not enumerated: {label} ({reason}).")
    return "\n".join(lines)


@blocking_tool
def crystod_irrep_product(space_group: str, irreps: list[str]) -> str:
    """Decompose the direct product of two or more irreps into irreps, e.g. R4- x R5+ of Pm-3m or T2g x T2g of the point group m-3m (selection rules, coupling terms in Landau expansions).

    For a space group the factors are full space-group irreps in ISO-IR
    notation and the product is decomposed over all k points it reaches
    (``crystod-group --product IRREP... --sg SG``, cross-validated against the
    Bilbao DIRPRO tables). For a point group the factors are point-group irreps
    (``T2g``, ``Eg``, ``A1``, ...) and the product is decomposed with the
    character table (``crystod-group --product IRREP... --pg PG``).

    Args:
        space_group: Space group as an international symbol (``Pm-3m``) or
            number (``221``). A point-group symbol in Hermann-Mauguin notation
            (``m-3m``, ``4/mmm``, ``3m``) is accepted too and selects the
            point-group product; a bare number always means a space group.
        irreps: Two or more irrep labels, e.g. ``["R4-", "R5+"]`` for a space
            group or ``["T2g", "T2g", "T1u"]`` for a point group.

    Returns:
        Markdown: the decomposition line in bold, followed by the CrystOD
        report (k points and star sizes, dimension check) for a space group or
        the multiplicity list for a point group.
    """
    from crystod import group

    names = parse_irrep_names(irreps)
    text = str(space_group).strip()

    if is_point_group(text) and not text.isdigit():
        point_group = parse_point_group(text)
        character_table = group.get_character_table(point_group)
        product = group.direct_product_character(character_table, point_group, names)
        counts = group.decompose_representation(character_table, product)
        formula = " x ".join(names)
        result = " + ".join(
            (f"{count}" if count > 1 else "") + irrep for irrep, count in counts.items() if count > 0
        )
        lines = [
            f"## Direct product in point group {point_group}",
            "",
            f"**{formula} = {result}**",
            "",
            markdown_table(
                ("Irrep", "Multiplicity"),
                [(irrep, count) for irrep, count in counts.items() if count > 0],
            ),
            "",
            "Characters of the product: "
            + ", ".join(
                f"{cls}: {format_character(value)}"
                for cls, value in zip(character_table["rotation_list"], product)
            )
            + ".",
        ]
        return "\n".join(lines)

    sg = parse_space_group(text)
    labels = parse_isoir_labels(names)
    algebra = group.SpaceGroupIrrepAlgebra(sg)
    report = group.format_product_report(algebra, labels)

    product_line = ""
    report_lines = report.splitlines()
    for index, line in enumerate(report_lines):
        if line.startswith("* Direct product") and index + 1 < len(report_lines):
            product_line = report_lines[index + 1].strip()
            break
    parent = f"{algebra.sg_type.international_short} (No. {algebra.sg_type.number})"
    lines = [f"## Direct product of space-group irreps in {parent}", ""]
    if product_line:
        lines += [f"**{product_line}**", ""]
    lines.append(fenced(report))
    return "\n".join(lines)


@blocking_tool
def crystod_decompose_representation(point_group: str, characters: list[float]) -> str:
    """Reduce a reducible representation of a point group into irreps from its characters (one per class), e.g. characters 3, 0, 1 in 3m give A1 + E.

    This is the textbook reduction formula n_i = (1/|G|) sum_C m_C chi_i(C)*
    chi(C), as ``crystod-group --decompose --pg PG --characters ...``; an E
    that the table stores as a real pair of complex-conjugate irreps (E of
    4, 3, 6, -4, 23, ...) is divided by its norm 2|G|, so the pair counts
    once per doubly degenerate level. The
    class order is that of CrystOD's character table for the group, which
    the result lists; call ``crystod_character_table`` first if unsure.

    Args:
        point_group: Point group in Hermann-Mauguin notation, e.g. ``m-3m``
            (Oh), ``4/mmm`` (D4h), ``3m`` (C3v), ``-43m`` (Td), ``mmm`` (D2h).
        characters: The character of the reducible representation for each
            class, in the class order of the character table (E first), e.g.
            ``[3, 0, 1]`` for 3m (classes E, 2C3, 3sigma_v).

    Returns:
        Markdown: the classes with their multiplicities and the given
        characters, then the decomposition ``Gamma = n1(irrep1) + ...``.
    """
    from crystod import group

    pg = parse_point_group(point_group)
    character_table = group.get_character_table(pg)
    classes = list(character_table["rotation_list"])
    multiplicities = _class_multiplicities(character_table)
    if not isinstance(characters, (list, tuple)) or len(characters) != len(classes):
        given = 0 if not isinstance(characters, (list, tuple)) else len(characters)
        class_text = ", ".join(f"{m}{c}" for m, c in zip(multiplicities, classes))
        raise ValueError(
            f"point group {pg} has {len(classes)} classes ({class_text}); give exactly "
            f"one character per class in that order ({given} given)."
        )
    values = [float(v) for v in characters]
    counts = group.decompose(values, character_table, multiplicities)
    raw = _raw_multiplicities(character_table, values)

    lines = [
        f"## Reduction of a representation of {pg}",
        "",
        markdown_table(
            ("Class", "Operations", "Character"),
            [(cls, mult, format_character(value)) for cls, mult, value in zip(classes, multiplicities, values)],
        ),
        "",
        f"**Gamma = {_decomposition_line(counts)}**",
    ]
    if any(abs(value - round(value)) > 1e-6 for value in raw.values()):
        lines += [
            "",
            "WARNING: the reduction formula gives non-integer multiplicities "
            + ", ".join(f"{irrep}: {value:.3f}" for irrep, value in raw.items() if abs(value - round(value)) > 1e-6)
            + "; the characters are not those of a representation of "
            f"{pg} in this class order (check the order against crystod_character_table).",
        ]
    return "\n".join(lines)


@blocking_tool
def crystod_ligand_field(point_group: str, orbital: str) -> str:
    """How an atomic orbital shell (s, p, d, f, ...) splits in a crystal/ligand field of a given point group, e.g. d in m-3m (octahedral) gives Eg + T2g.

    The characters of the (2l+1)-dimensional orbital representation are
    computed from the rotation angles of each class and reduced with the
    character table (``crystod-group --ligand-field ORBITAL --pg PG``).

    Args:
        point_group: Point group of the site in Hermann-Mauguin notation, e.g.
            ``m-3m`` (octahedral Oh), ``-43m`` (tetrahedral Td), ``4/mmm``
            (D4h), ``3m`` (C3v).
        orbital: The shell letter: ``s``, ``p``, ``d``, ``f`` (also g, h, i).

    Returns:
        Markdown: the orbital characters per class and the splitting
        ``d in m-3m = 1(Eg) + 1(T2g)``.
    """
    from crystod import group

    pg = parse_point_group(point_group)
    shell = parse_orbital(orbital, ORBITALS)
    character_table = group.get_character_table(pg)
    orbital_characters = group.get_orbital_characters(shell, character_table)
    classes = list(character_table["rotation_list"])
    multiplicities = _class_multiplicities(character_table)
    values = [float(orbital_characters[cls]) for cls in classes]
    counts = group.decompose(values, character_table, multiplicities)

    lines = [
        f"## Ligand-field splitting of the {shell} shell in {pg}",
        "",
        markdown_table(
            ("Class", "Operations", "Character of the " + shell + " shell"),
            [(cls, mult, format_character(value)) for cls, mult, value in zip(classes, multiplicities, values)],
        ),
        "",
        f"**{shell} in {pg} = {_decomposition_line(counts)}**",
        "",
        f"The {shell} shell (l = {ORBITALS.index(shell)}, {2 * ORBITALS.index(shell) + 1} orbitals) "
        "splits into the listed irreps; the irrep dimension is the degeneracy of each level"
        + (
            "; in this point group 1(E) is one doubly degenerate level (a "
            "time-reversal pair of complex-conjugate one-dimensional irreps, which "
            "the character table lists as one E)."
            if pg.replace(" ", "") in _CONJUGATE_PAIR_GROUPS else "."
        ),
    ]
    return "\n".join(lines)


# point groups whose character tables carry complex-conjugate pairs of
# one-dimensional irreps as one real two-dimensional E; the multiplicity of
# such an E counts the pair once (one doubly degenerate level per copy)
_CONJUGATE_PAIR_GROUPS = {"3", "-3", "4", "-4", "4/m", "6", "-6", "6/m", "23", "m-3"}


@blocking_tool
def crystod_character_table(group: str, kpoint: str | list[float] | None = None) -> str:
    """Character table of a point group (m-3m, 3m, ...) or of the little group of a k point of a space group (Pm-3m at R, ...), with ISO-IR irrep labels for space groups.

    Without ``kpoint`` and with a point-group symbol the table is the
    point-group character table (``crystod-group --table --pg PG``). With a
    space group the table is that of the little group of ``kpoint``, labeled
    with the ISO-IR irreps (``R1+ ... R5-`` at R of Pm-3m) at tabulated k
    points and at symmetry lines/planes (``crystod-group --table --sg SG
    --kpoint ...``); it also lists which irrep labels exist at that k point.

    Args:
        group: Point group in Hermann-Mauguin notation (``m-3m``, ``4/mmm``,
            ``3m``, ``-43m``) or space group as a symbol (``Pm-3m``) or number
            (``221``). A bare number that is also a point-group name
            (``222``, ``23``, ``32``, ``1``...) is read as the point group
            unless ``kpoint`` is given.
        kpoint: For a space group: a k-point label (``GM``, ``X``, ``M``,
            ``R``, ...) or three fractional coordinates in the primitive
            reciprocal basis (``"1/2 1/2 1/2"`` or ``[0.5, 0.5, 0.5]``).
            Omitted for a space group, the Gamma point is used.

    Returns:
        Markdown: a table of characters per class (point group) or the
        CrystOD little-group table with one row per ISO-IR irrep.
    """
    from crystod import group as crystod_group

    text = str(group).strip()
    label, coordinates = parse_kpoint(kpoint)

    if label is None and coordinates is None and is_point_group(text):
        pg = parse_point_group(text)
        character_table = crystod_group.get_character_table(pg)
        classes = list(character_table["rotation_list"])
        multiplicities = _class_multiplicities(character_table)
        headers = ["Irrep"] + [f"{m}{cls}" for m, cls in zip(multiplicities, classes)]
        rows = [
            [irrep] + [format_character(v) for v in values]
            for irrep, values in character_table["character_table"].items()
        ]
        return "\n".join([
            f"## Character table of point group {pg}",
            "",
            markdown_table(headers, rows),
            "",
            f"Order {sum(multiplicities)}; column headers give the number of operations "
            "in each class. The irreps are labeled in Mulliken notation; the same "
            "labels are used by crystod_ligand_field and crystod_decompose_representation.",
        ])

    sg = parse_space_group(text)
    sg_type, names, primitive, _ = special_kpoints(sg)
    note = ""
    if label is not None:
        coordinates = resolve_kpoint_label(sg, label)
    elif coordinates is None:
        coordinates = [0.0, 0.0, 0.0]
        label = "GM"
        note = "No k point was given, so the table is that of the Gamma point."
    table = crystod_group.format_spacegroup_table(sg, coordinates)
    parent = f"{sg_type.international_short} (No. {sg_type.number})"
    where = f"{label} {format_kpoint(coordinates)}" if label else format_kpoint(coordinates)
    lines = [f"## Character table of the little group of k = {where} in {parent}", ""]
    if note:
        lines += [note, ""]
    lines.append(fenced(table))
    lines += [
        "",
        "Rows are the ISO-IR irreps of the little group of k (the labels used for "
        "phonon modes, crystal orbitals and isotropy subgroups); columns are the "
        "symmetry operations of the little group in the primitive basis. Tabulated "
        f"k points of this space group: "
        + ", ".join(f"{n} {format_kpoint(k)}" for n, k in zip(names, primitive))
        + ".",
    ]
    return "\n".join(lines)


# the degrees --invariants accepts (crystod-group --degree)
_MAX_DEGREE = 12


def _parse_degree(value: object) -> int:
    """The highest total degree of the invariants: an integer from 1 to 12."""
    if isinstance(value, bool):
        value = None
    try:
        degree = int(str(value).strip())
    except (TypeError, ValueError):
        degree = 0
    if not 1 <= degree <= _MAX_DEGREE:
        raise ValueError(
            f"degree must be an integer from 1 to {_MAX_DEGREE} (the highest total "
            f'degree of the invariant polynomials, e.g. 4); got "{value}".'
        )
    return degree


def _direction_text(tokens: list[str], arm_chunks: list[int]) -> str:
    """``(a,0;0,b)``: the components of one irrep, arms separated by ``;``."""
    arms, start = [], 0
    for size in arm_chunks or [len(tokens)]:
        arms.append(",".join(tokens[start : start + size]))
        start += size
    return "(" + ";".join(arms) + ")"


def _coupling_lines(basis, top: int) -> list[str]:
    from crystod.invariants import lowest_coupling_terms

    terms = lowest_coupling_terms(basis)
    if not terms:
        return [
            f"No coupling term (an invariant in the components of every irrep) up to "
            f"total degree {top}; raise degree to search further."
        ]
    lines = []
    for term in terms:
        multidegree = "(" + ", ".join(str(x) for x in term.multidegree) + ")"
        kind = " (trilinear)" if term.trilinear else ""
        count = len(term.polynomials)
        noun = "invariant" if count == 1 else "invariants"
        lines.append(f"- degree {multidegree}{kind}: {count} {noun}")
        for poly in term.polynomials:
            text = f"{poly.name} = {poly.expression}" if poly.expression else poly.name
            lines.append(f"  - `{text}`")
    return lines


def _secondary_rows(rows) -> list[tuple]:
    table = []
    for row in rows:
        k_text = "(" + ",".join(str(v) for v in row.k) + ")"
        table.append((
            row.label, k_text, row.dimension, row.n_free, row.direction_pattern,
            row.coupling or "-", row.kind,
        ))
    return table


@blocking_tool
def crystod_invariants(
    space_group: str,
    irreps: list[str] | str,
    degree: int = 4,
    direction: str | None = None,
) -> str:
    """Landau free-energy invariants of a space-group irrep or of a direct sum of irreps: the invariant polynomials degree by degree, the Landau and Lifshitz conditions for a continuous transition, the lowest-order coupling term (e.g. the trilinear X2+ X3- GM5- term of hybrid improper ferroelectrics in I4/mmm) and, for a direction, the secondary order parameters.

    The offline counterpart of the ISOTROPY INVARIANTS program
    (``crystod-group --parent SG --irrep IR [IR2 ...] --invariants``). The
    polynomials in the order-parameter components that every operation of
    the parent group leaves unchanged are listed up to total ``degree``,
    separated into products of lower-degree invariants and new ones
    (``I<degree>_<n>``), with exact coefficients where they exist; their
    numbers are checked against the Molien series. The components ``Q1,
    Q2, ...`` (``<label>_<n>`` for a direct sum) follow the real basis of
    the direction patterns of ``crystod_isotropy_subgroups``. With
    ``direction`` the free energy is also restricted to that order-parameter
    direction and the secondary order parameters of its isotropy subgroup
    are listed (``--order-parameter DIR --secondary``).

    Args:
        space_group: Parent space group as an international symbol
            (``Pm-3m``, ``I4/mmm``) or number (``221``).
        irreps: One ISO-IR irrep label (``R4+``) or several for a direct sum
            (``["X2+", "X3-", "GM5-"]`` or ``"X2+ X3- GM5-"``); which labels a
            space group has: ``crystod_character_table``.
        degree: Highest total degree of the invariants and of the coupling
            terms searched, 1 to 12 (default 4). High degrees of
            many-component irreps are refused when one linear problem would
            exceed the monomial limit.
        direction: Optional order-parameter direction, as
            ``crystod_isotropy_subgroups`` takes it: letters for free
            components and numbers for fixed ones, one per component of all
            the irreps in order (``"a 0 0"`` for R4+, ``"0 a 0 b"`` for
            X2+ X3- of I4/mmm; ``,`` and ``;`` separate components like
            spaces). Needs ``degree`` >= 2.

    Returns:
        Markdown: the parent and the irreps (dimension, Landau and Lifshitz
        lines), the invariant polynomials in a code block, the lowest-order
        coupling term of a direct sum and, with ``direction``, the restricted
        free energy and the secondary-order-parameter table (irrep, k, dim,
        n_free, direction, coupling, type).
    """
    import warnings

    from crystod import group
    from crystod.invariants import (
        INVARIANTS_CITATION,
        format_invariants,
        landau_lifshitz_lines,
        restricted_invariants,
    )

    space_group_text = parse_space_group(space_group)
    labels = parse_isoir_labels(irreps)
    top = _parse_degree(degree)
    tokens = parse_order_parameter(direction)
    if tokens is not None and top < 2:
        raise ValueError(
            "a direction needs degree 2 or higher (the coupling terms of the "
            "secondary order parameters have total degree >= 2)."
        )
    sg_type = special_kpoints(space_group_text)[0]
    parent_text = f"{sg_type.international_short} (No. {sg_type.number})"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        basis = group.invariant_polynomials(
            space_group_text, labels if len(labels) > 1 else labels[0], top
        )
    coupled = len(basis.parts) > 1
    irrep_text = " + ".join(labels)

    lines = [f"## Invariant polynomials of {irrep_text} in {parent_text}", ""]
    lines.append("### Irreps" if coupled else "### Irrep")
    lines.append("")
    for label, dimension in zip(labels, basis.parts):
        record = group.landau_lifshitz(space_group_text, label)
        landau, lifshitz, verdict = landau_lifshitz_lines(record)
        lines.append(
            f"- {label}: order-parameter dimension {dimension}; {landau}; {lifshitz}; "
            f"{verdict.removeprefix('-> ')}."
        )
    if coupled:
        lines.append(
            f"- Coupled order parameter dimension {sum(basis.parts)} "
            f"({' + '.join(str(d) for d in basis.parts)})."
        )
    lines += ["", f"### Invariant polynomials (degree <= {top})", "", fenced(format_invariants(basis))]

    if coupled:
        lines += ["", "### Lowest-order coupling term", ""]
        lines += _coupling_lines(basis, top)

    if tokens is not None:
        if len(tokens) != len(basis.variables):
            raise ValueError(
                f"the direction needs {len(basis.variables)} components, one per "
                f"order-parameter component of {irrep_text} in order ({len(tokens)} given)."
            )
        subgroup = group.isotropy_subgroups(
            space_group_text, labels if len(labels) > 1 else labels[0], tokens,
            with_settings=False,
        )[0]
        pieces, start = [], 0
        for label, dimension, chunks in zip(labels, basis.parts, basis.arm_chunks):
            pieces.append(f"{label}{_direction_text(tokens[start : start + dimension], chunks)}")
            start += dimension
        where = " ".join(pieces)
        lines += [
            "",
            f"### Restricted to {where}",
            "",
            f"Isotropy subgroup {subgroup.symbol} (No. {subgroup.number}), size "
            f"{subgroup.size}, index {subgroup.index}.",
            "",
        ]
        restricted = restricted_invariants(basis, tokens)
        if not restricted:
            lines.append("No invariant survives on this direction.")
        for n, texts in restricted.items():
            lines.append(f"- degree {n}: `{'; '.join(texts)}`")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            rows = group.secondary_order_parameters(
                space_group_text, labels if len(labels) > 1 else labels[0], tokens, top
            )
        lines += [
            "",
            f"### Secondary order parameters of {subgroup.symbol} (No. {subgroup.number})",
            "",
            markdown_table(
                ("Irrep", "k", "Dim", "n_free", "Direction", "Coupling", "Type"),
                _secondary_rows(rows),
            ),
            "",
            "Every parent irrep with a component the isotropy subgroup leaves free "
            "(n_free > 0), at Gamma and at the tabulated special k points of the "
            "subgroup lattice. Coupling: the lowest invariant Q^m eta (Q the primary "
            "order parameter, eta this irrep) that does not vanish on the fixed space "
            f"of the subgroup, searched up to total degree {top}; type: primary, polar "
            "(in the vector representation), strain (identity or in the symmetric "
            "square of the vector representation) or other.",
        ]
        notes = []
        for warning in caught:
            message = " ".join(str(warning.message).split())
            if issubclass(warning.category, UserWarning) and message not in notes:
                notes.append(message)
        for message in notes:
            lines += ["", f"Note: {message}"]

    lines += ["", " ".join(INVARIANTS_CITATION.split())]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# reverse lookup, group-subgroup graph, correlation, property tensors
# --------------------------------------------------------------------------

_ISOSUBGROUP_CITATION = (
    "Conventions and validation: ISOSUBGROUP (https://iso.byu.edu): H. T. Stokes, "
    "S. van Orden and B. J. Campbell, \"Tool for Generating Isotropy Subgroups of "
    "Crystallographic Space Groups\", J. Appl. Cryst. 49, 1849-1853 (2016)."
)


def _needs_043(feature: str) -> ValueError:
    return ValueError(f"{feature} needs CrystOD >= 0.4.3; upgrade the CrystOD that "
                      "crystod-mcp is installed with.")


@contextlib.contextmanager
def _as_value_error():
    """Turn the ``SystemExit`` of CrystOD's implementation modules into a
    ``ValueError`` (asyncio re-raises a ``SystemExit`` of a worker thread out
    of the event loop instead of returning it to the tool wrapper)."""
    try:
        yield
    except SystemExit as exc:
        message = " ".join(str(exc).split()).removeprefix("ERROR: ")
        raise ValueError(message or "invalid input") from None


def _count(number: int, singular: str, plural: str) -> str:
    return f"{number} {singular if number == 1 else plural}"


def _optional_positive_int(value: object, name: str) -> int | None:
    """``None``, or a positive integer (``size``, ``index``)."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    number = None
    if not isinstance(value, bool):
        try:
            number = float(str(value).strip())
        except ValueError:
            number = None
    if number is None or number != number or number < 1 or number != int(number):
        raise ValueError(f'{name} must be a positive integer such as 2 or 4; got "{value}".')
    return int(number)


def _fraction_text(value: object) -> str:
    from fractions import Fraction

    fraction = Fraction(float(value)).limit_denominator(48)
    if fraction.denominator == 1:
        return str(fraction.numerator)
    return f"{fraction.numerator}/{fraction.denominator}"


def _setting_vector(values) -> str:
    """``(1,-1,0)``: one basis row or an origin, fractions exact."""
    return "(" + ",".join(_fraction_text(v) for v in values) + ")"


def _setting_rows(rows) -> str:
    if rows is None:
        return "-"
    return ",".join(_setting_vector(row) for row in rows)


def _reduced_origin(values) -> str:
    """The origin reduced into [0,1) by a parent lattice translation."""
    import numpy as np

    if values is None:
        return "-"
    reduced = np.mod(np.round(np.asarray(values, dtype=float), 6), 1.0)
    reduced[np.isclose(reduced, 1.0, atol=1e-6)] = 0.0
    return _setting_vector(reduced + 0.0)


def _kpoint_tokens(value: object) -> list[str]:
    if isinstance(value, (list, tuple)):
        raw = " ".join(str(item) for item in value)
    else:
        raw = "" if value is None else str(value)
    return [token for token in re.split(r"[\s,;]+", raw.strip()) if token]


def _partner_sentence(number: int, symbol: str, partner_rows: list) -> str:
    from crystod.isotropy_subgroup import ENANTIOMORPHIC_PAIRS

    partner = ENANTIOMORPHIC_PAIRS.get(number)
    return (
        f"{number} and {partner} ({symbol} and {partner_rows[0].symbol}) are "
        "enantiomorphic partner types: an improper operation of the parent maps "
        "the order parameter of such a stratum to another one of the same stratum "
        "whose stabilizer is the partner, so the rows listed with subgroup "
        f"{partner_rows[0].symbol} count as type {symbol}."
    )


@blocking_tool
def crystod_find_isotropy_irreps(
    parent: str,
    child: str,
    size: int | None = None,
    index: int | None = None,
    kpoints: str | list[str] | None = None,
    coupled: bool = False,
) -> str:
    """Reverse lookup of isotropy subgroups: which irreps and order-parameter directions of a parent space group give an isotropy subgroup of a given type, e.g. which distortions of Pm-3m lead to I4/mcm, or which pair of coupled irreps gives the Pnma tilt system.

    The inverse of ``crystod_isotropy_subgroups`` (``crystod-group --parent
    G --child H [--size N] [--index N] [--kpoint K ...] [--coupled]``): every
    stratum of every irrep at the special k points of the ISO-IR tables
    whose isotropy subgroup has type H, with the k point, direction, cell
    size, index and the conventional basis and origin of the subgroup. In a
    parent with improper operations the enantiomorphic partner type of H
    (P4_122 for P4_322) counts as a match. With ``coupled`` the pairs of
    irreps are searched too (same k point first, then different k points)
    for coupled strata that need both irreps: Pm-3m, Pnma, size 4 gives
    ``R4+ (0,a,a)`` + ``M3+ (d;0;0)``.

    The lookup reads the isotropy table of the parent (every stratum of
    every special-point irrep). The first call on a parent builds that table
    and caches it under ``$CRYSTOD_CACHE_DIR/isotropy`` (default
    ``~/.cache/crystod``): a few seconds for most parents (Pm-3m about 5 s),
    about a minute for the face-centred cubic groups with their large W
    stars; later calls on the same parent read the file and take about a
    second. The coupled search of all pairs of Pm-3m takes about 10 s.

    Args:
        parent: Parent space group G as an international symbol (``Pm-3m``,
            ``I4/mmm``) or number (``221``).
        child: Subgroup type H as an international symbol (``I4/mcm``,
            ``Pnma``, ``R-3c``) or number (``140``).
        size: Optional primitive-cell multiplication to keep (``2``, ``4``).
        index: Optional index [G:H] to keep (``6``, ``24``).
        kpoints: Optional special k points to keep: names (``"R"``, ``"R M"``
            or ``["R", "M"]``; which names a space group has:
            ``crystod_special_kpoints``) or three coordinates of one point.
        coupled: Also search the pairs of coupled irreps (default false).

    Returns:
        Markdown: the isotropy-table summary, a table of the single-irrep
        strata of type H (irrep, k, direction, subgroup, size, index,
        conventional basis, origin) and, with ``coupled``, a table of the
        coupled strata (both irreps and their directions).
    """
    space_group = parse_space_group(parent)
    child_text = "" if child is None else str(child).strip()
    if not child_text:
        raise ValueError(
            "give the subgroup type H as an international symbol such as I4/mcm or "
            "Pnma, or as its space-group number."
        )
    size_value = _optional_positive_int(size, "size")
    index_value = _optional_positive_int(index, "index")
    tokens = _kpoint_tokens(kpoints)
    try:
        from crystod import isotropy_table as lookup
    except ImportError:
        raise _needs_043("the reverse lookup of isotropy subgroups") from None

    with _as_value_error():
        number, symbol = lookup.resolve_child(child_text)
        table = lookup.isotropy_table(space_group, progress=False)
        knames = lookup.resolve_kpoints(table, tokens) if tokens else None
    numbers = lookup.child_numbers(number, not table.sohncke)
    rows = table.select(numbers, size=size_value, index=index_value, kpoints=knames)

    filters = []
    if size_value is not None:
        filters.append(f"size {size_value}")
    if index_value is not None:
        filters.append(f"index {index_value}")
    if knames:
        filters.append(("k point " if len(knames) == 1 else "k points ") + ", ".join(knames))
    where = f" with {', '.join(filters)}" if filters else ""

    parent_text = f"{table.space_group} (No. {table.number})"
    state = "read from the cache" if table.from_cache else "built on this call"
    if not table.from_cache and table.cache_path is not None:
        state += " and cached for later calls"
    lines = [
        f"## Irreps of {parent_text} with isotropy subgroups of type {symbol} (No. {number})",
        "",
        f"Isotropy table of {table.space_group}: {len(table.irreps)} irreps at the "
        f"special k points {', '.join(table.kpoints)}; {len(table.strata)} strata "
        f"({state}).",
    ]
    if filters:
        lines.append(f"Selected: {', '.join(filters)}.")
    lines += ["", "### Single irreps", ""]
    if rows:
        lines.append(markdown_table(
            ("Irrep", "k", "Direction", "Subgroup", "Size", "Index",
             "Conventional basis", "Origin"),
            [(s.label, s.kname, s.direction, f"{s.symbol} ({s.number})", s.size, s.index,
              _setting_rows(s.basis), _reduced_origin(s.origin)) for s in rows],
        ))
        lines += [
            "",
            f"{_count(len(rows), 'stratum', 'strata')}. Every row is one embedding of the subgroup "
            "(several rows of one type differ in the orientation or origin of the "
            "subgroup cell); conventional basis: rows in parent conventional units; "
            "origin: in parent conventional coordinates, reduced into [0,1). "
            "crystod_isotropy_subgroups with the irrep and the direction gives the "
            "same subgroup.",
        ]
        partner_rows = [s for s in rows if s.number != number]
        if partner_rows:
            lines += ["", "Note: " + _partner_sentence(number, symbol, partner_rows)]
    else:
        lines.append(
            f"No single-irrep stratum of type {symbol} (No. {number}){where}; the "
            "subgroup may need two coupled irreps"
            + ("." if coupled else " (call again with coupled=true).")
        )

    if coupled:
        with _as_value_error():
            search = lookup.coupled_search(table, number, size=size_value,
                                           index=index_value, knames=knames)
        n_pairs = search.n_pairs_same_k + search.n_pairs_cross_k
        lines += [
            "",
            "### Coupled pairs of irreps",
            "",
            f"{search.n_irreps} irreps pass the point-group filter; {n_pairs} pairs "
            f"searched ({search.n_pairs_same_k} at the same k point, "
            f"{search.n_pairs_cross_k} at different k points).",
            "",
        ]
        if search.matches:
            lines.append(markdown_table(
                ("Irrep 1", "Direction 1", "Irrep 2", "Direction 2", "Subgroup", "Size",
                 "Index", "Conventional basis", "Origin"),
                [(m.label1, m.direction1, m.label2, m.direction2,
                  f"{m.symbol} ({m.number})", m.size, m.index,
                  _setting_rows(m.basis), _reduced_origin(m.origin))
                 for m in search.matches],
            ))
            lines += [
                "",
                f"{_count(len(search.matches), 'coupled stratum', 'coupled strata')}. Every row needs both "
                "irreps: neither irrep alone has this isotropy subgroup. The letters "
                "of the second direction continue after those of the first, as "
                "crystod_isotropy_subgroups prints a coupled direction.",
            ]
            partner_rows = [m for m in search.matches if m.number != number]
            if partner_rows:
                lines += ["", "Note: " + _partner_sentence(number, symbol, partner_rows)]
        else:
            lines.append(f"No coupled stratum of type {symbol} (No. {number}){where}.")
        for pair, reason in search.errors.items():
            lines += ["", f"Not analyzed: {pair} ({reason})."]
    for label, reason in table.errors.items():
        lines += ["", f"Not enumerated: {label} ({reason})."]
    lines += ["", _ISOSUBGROUP_CITATION]
    return "\n".join(lines)


@blocking_tool
def crystod_subgroup_graph(parent: str, irreps: list[str] | str) -> str:
    """Group-subgroup graph of the isotropy subgroups of one irrep or of a direct sum of irreps: every stratum layered by index, with the maximal-subgroup lines (solid, or dashed when a continuous transition from the parent is forbidden), e.g. the perovskite tilt systems of Pm-3m R4+ + M3+.

    The text form of ``crystod-group --parent SG --irrep IR [IR2 ...]
    --graph`` (no HTML or DOT file is written). The nodes are the parent,
    every stratum of the representation (for several irreps also those in
    which only some irreps condense) and the kernel (the generic order
    parameter); a line joins a group to each of its maximal subgroups among
    the nodes, up to conjugacy in the parent. A line from the parent is
    dashed when the irrep that alone gives the stratum violates the Landau
    condition (a cubic invariant) or the Lifshitz condition, or when the
    stratum needs several irreps at once; lines between strata are solid
    (their continuity is not tested). ``--parent Pm-3m --irrep R4+ M3+``
    contains the 15 tilt systems of Howard and Stokes.

    Args:
        parent: Parent space group as an international symbol (``Pm-3m``,
            ``I4/mmm``) or number (``221``).
        irreps: One ISO-IR irrep label (``R4+``) or several for a direct sum
            (``["R4+", "M3+"]`` or ``"R4+ M3+"``); which labels a space group
            has: ``crystod_character_table``.

    Returns:
        Markdown: the irreps with their Landau and Lifshitz lines, the nodes
        by index layer, a node table (subgroup, size, index, direction, free
        parameters, the nodes it is maximal in, the line from the parent)
        and the edge list (from, to, solid or dashed, index ratio).
    """
    from crystod import group

    space_group = parse_space_group(parent)
    labels = parse_isoir_labels(irreps)
    build = getattr(group, "subgroup_graph", None)
    if build is None:
        raise _needs_043("the group-subgroup graph")
    from crystod.invariants import landau_lifshitz_lines

    graph = build(space_group, labels if len(labels) > 1 else labels[0])
    parent_text = f"{graph.space_group} (No. {graph.number})"
    irrep_text = " + ".join(graph.irreps)
    lines = [f"## Group-subgroup graph of {irrep_text} in {parent_text}", ""]
    lines += ["### Irreps" if len(graph.irreps) > 1 else "### Irrep", ""]
    for label, dimension in zip(graph.irreps, graph.dimensions):
        landau, lifshitz, verdict = landau_lifshitz_lines(graph.landau[label])
        lines.append(f"- {label}: order-parameter dimension {dimension}; {landau}; "
                     f"{lifshitz}; {verdict.removeprefix('-> ')}.")

    n_strata = sum(node.kind != "parent" for node in graph.nodes)
    n_dashed = sum(edge.dashed for edge in graph.edges)
    lines += [
        "",
        "### Nodes by index",
        "",
        f"{len(graph.nodes)} node(s) (the parent and {n_strata} strata), "
        f"{len(graph.edges)} edge(s) ({n_dashed} dashed).",
        "",
    ]
    for index, layer in graph.layers():
        lines.append(f"- index {index}: " + ", ".join(node.label for node in layer))

    rows = []
    for node in graph.nodes:
        above = graph.above(node.id)
        from_parent = [edge for edge in above if graph.nodes[edge.parent].kind == "parent"]
        line_text = "-"
        if from_parent:
            line_text = "dashed" if any(edge.dashed for edge in from_parent) else "solid"
        kind = {"parent": "parent", "kernel": "kernel"}.get(node.kind, "stratum")
        rows.append((
            node.id + 1, node.label, kind, node.size, node.index, node.direction or "-",
            node.n_free, ",".join(str(edge.parent + 1) for edge in above) or "-", line_text,
        ))
    lines += [
        "",
        "### Nodes",
        "",
        markdown_table(
            ("Node", "Subgroup", "Kind", "Size", "Index", "Direction", "Free parameters",
             "Maximal in", "From parent"),
            rows,
        ),
        "",
        "Maximal in: the nodes in which this subgroup is maximal (among the nodes); "
        "from parent: the line from the parent, dashed when the Landau or the "
        "Lifshitz condition of its irrep fails (no continuous transition).",
        "",
        "### Edges",
        "",
        markdown_table(
            ("From", "To", "Line", "Index ratio"),
            [(f"{edge.parent + 1} {graph.nodes[edge.parent].label}",
              f"{edge.child + 1} {graph.nodes[edge.child].label}",
              "dashed" if edge.dashed else "solid", edge.index_ratio)
             for edge in graph.edges],
        ),
        "",
        "Every edge joins a group to a maximal subgroup among the nodes, up to "
        "conjugacy in the parent (transitively reduced). crystod-group --parent "
        f"{graph.space_group} --irrep {' '.join(graph.irreps)} --graph draws the same "
        "graph as an HTML page.",
        "",
        _ISOSUBGROUP_CITATION,
    ]
    if graph.number == 221 and set(graph.irreps) <= {"R4+", "M3+"}:
        from crystod.subgroup_graph import HOWARD_STOKES_CITATION

        lines.append("Octahedral tilt systems of the perovskites: " + HOWARD_STOKES_CITATION)
    return "\n".join(lines)


def _decomposition_terms(terms) -> str:
    return " + ".join(f"{count} {name}" if count > 1 else name for name, count in terms) or "0"


def _correlation_markdown(point_group: str, subgroup: str) -> str:
    from crystod import group

    table = group.correlation_table(point_group, subgroup)
    count = len(table.orientations)
    index = table.parent_order // table.subgroup_order
    lines = [
        f"## Correlation table {table.parent} -> {table.subgroup}",
        "",
        f"{table.parent} (order {table.parent_order}) -> {table.subgroup} (order "
        f"{table.subgroup_order}, index {index}): {count} inequivalent "
        f"orientation{'s' if count > 1 else ''}.",
        "",
    ]
    for (header, _), conjugates, axes in zip(table.orientations, table.conjugates, table.axes):
        text = f"- {header}: {conjugates} conjugate subgroup{'s' if conjugates > 1 else ''}"
        lines.append(text + (f"; {axes}" if axes else ""))
    lines += [
        "",
        markdown_table(
            # "4 || z": a bare | would end the Markdown cell
            [f"Irrep of {table.parent}"]
            + [header.replace("|", "\\|") for header, _ in table.orientations],
            [[irrep] + [_decomposition_terms(terms) for terms in decompositions]
             for irrep, decompositions in table.rows.items()],
        ),
        "",
    ]
    frame = (f"directions [uvw] are in the hexagonal axes a, b, c of the table of "
             f"{table.parent}" if table.hexagonal_axes else
             f"x, y, z are the axes [100], [010], [001] of the table of {table.parent}")
    lines.append(
        f"One column per inequivalent orientation of {table.subgroup}; {frame}; 2_d is "
        "the twofold rotation about d, m_d the mirror normal to d; the symbols of "
        f"{table.subgroup} are those of its character table (crystod_character_table) "
        "with the classes listed above, which B1/B2/B3 follow."
    )
    return "\n".join(lines)


def _compatibility_points(value: object) -> list:
    """One or two special points: names, or three coordinates each."""
    if isinstance(value, (list, tuple)) and value and all(
        isinstance(item, str) and len(_kpoint_tokens(item)) == 3 for item in value
    ):
        return [_kpoint_tokens(item) for item in value]
    tokens = _kpoint_tokens(value)
    numeric = bool(tokens) and all(re.fullmatch(r"[-+0-9./]+", token) for token in tokens)
    if numeric and len(tokens) in (3, 6):
        return [tokens[:3]] + ([tokens[3:]] if len(tokens) == 6 else [])
    if not tokens or len(tokens) > 2 or numeric:
        raise ValueError(
            'kpoints must name one or two special points, e.g. "GM X" (the line '
            'between them) or "GM" (every line from Gamma); crystod_special_kpoints '
            "lists the names."
        )
    return tokens


def _compatibility_markdown(space_group: str, kpoints, line) -> str:
    from crystod import group

    points = _compatibility_points(kpoints)
    line_name = None if line is None or not str(line).strip() else str(line).strip().upper()
    sg_type = special_kpoints(space_group)[0]
    records = group.compatibility_relations(
        space_group, points[0], points[1] if len(points) == 2 else None, line_name,
    )
    lines = [
        f"## Compatibility relations in {sg_type.international_short} (No. {sg_type.number})",
    ]
    groups: dict = {}
    for record in records:
        groups.setdefault(
            (record.line_label, record.k_delta, record.fixed_dimension), []
        ).append(record)
    kinds = {1: "Line", 2: "Plane", 3: "General segment"}
    for (line_label, k_delta, fixed), members in groups.items():
        ends: dict = {}
        for record in members:
            ends.setdefault(record.k0_label, record.k_point)
        between = " and ".join(f"{name} {format_kpoint(k)}" for name, k in ends.items())
        lines += [
            "",
            f"### {kinds.get(fixed, 'Line')} {line_label} between {between}",
            "",
            f"Line irreps labeled at kdelta = {format_kpoint(k_delta)} (primitive basis).",
            "",
            markdown_table(
                ("Irrep", "Point", "Dim", f"On {line_label}"),
                [(record.k0_irrep, record.k0_label, record.dimension,
                  _decomposition_terms(record.line_irreps)) for record in members],
            ),
        ]
    lines += [
        "",
        "Every small irrep of the end points is restricted to the little group of the "
        "line point kdelta (translation phases corrected for the wave-vector "
        "difference) and decomposed into the ISO-IR irreps of the line; the "
        "dimensions add up for every row.",
    ]
    return "\n".join(lines)


def _subduction_markdown(parent: str, irreps, direction, irrep_list) -> str:
    from crystod import group

    labels = parse_isoir_labels(irreps)
    tokens = parse_order_parameter(direction)
    if tokens is None:
        raise ValueError(
            'the subduction needs direction, the order-parameter components of the '
            'irreps in order, e.g. "a 0 0" for R4+ or "0 a a d 0 0" for R4+ M3+.'
        )
    targets = None if irrep_list is None else parse_isoir_labels(irrep_list)
    sg_type = special_kpoints(parent)[0]
    try:
        rows = group.subduce_to_child(parent, labels if len(labels) > 1 else labels[0],
                                      tokens, targets)
    except ValueError as exc:
        raise ValueError(str(exc).replace("--order-parameter", "direction")) from None
    first = rows[0]
    parent_text = f"{sg_type.international_short} (No. {sg_type.number})"
    where = f"{' + '.join(labels)} along ({','.join(tokens)})"
    lines = [
        f"## Subduction to the Gamma point of {first.child_symbol} "
        f"(No. {first.child_number}), {where} in {parent_text}",
        "",
        f"H = {first.child_symbol} (No. {first.child_number}), the isotropy subgroup of "
        f"{where}. The Gamma labels of H are ISO-IR labels in the frame of H with "
        f"basis rows {_setting_rows(first.basis)} and origin "
        f"{_setting_vector(first.origin) if first.origin is not None else '-'} (parent "
        "conventional units); Mulliken symbols in brackets.",
        "",
    ]
    table_rows = []
    for row in rows:
        terms = []
        for name, count in row.gamma_part:
            text = f"{count} {name}" if count > 1 else name
            if name in row.mulliken:
                text += f" [{row.mulliken[name]}]"
            terms.append(text)
        table_rows.append((
            row.label, row.kname, row.dimension, " + ".join(terms) or "-",
            row.nongamma_components, "yes" if row.primary else "no",
        ))
    lines.append(markdown_table(
        ("Parent irrep", "k", "Dim", "Gamma irreps of H", "Non-Gamma components", "Primary"),
        table_rows,
    ))
    lines += [
        "",
        "The Gamma sector of every parent irrep (the part invariant under the "
        "translations of H) is decomposed into Gamma irreps of H; the rest lies at "
        "non-Gamma k points of H. GM1 of H occurs in a primary irrep once per free "
        "parameter of the direction (checked).",
    ]
    return "\n".join(lines)


@blocking_tool
def crystod_correlate(
    point_group: str | None = None,
    subgroup: str | None = None,
    space_group: str | None = None,
    kpoints: str | list[str] | None = None,
    line: str | None = None,
    parent: str | None = None,
    irreps: list[str] | str | None = None,
    direction: str | None = None,
    irrep_list: list[str] | str | None = None,
) -> str:
    """Correlation of irreps to a subgroup, in one of three forms: a point-group correlation table (m-3m -> 4/mmm: T2g -> B2g + Eg), the compatibility relations along a symmetry line (Pm-3m GM4- -> DT1 + DT5), or the subduction of parent irreps to the Gamma point of an isotropy subgroup (R4+ of Pm-3m becomes A1g + Eg of I4/mcm).

    The three forms of ``crystod-group --correlate``; give the arguments of
    exactly one:

    (a) ``point_group`` and ``subgroup`` -- the correlation table
        (``--correlate --pg G --subgroup H``): every irrep of G restricted to
        each inequivalent orientation of the subgroup type H, one column per
        orientation (m-3m has three orientations of mm2).
    (b) ``space_group`` and ``kpoints`` (optional ``line``) -- the
        compatibility relations (``--correlate --sg SG --kpoint K0 K1
        [--line L]``): how the small irreps at two special points split or
        join on the symmetry line between them; with one point, every line
        from that point.
    (c) ``parent``, ``irreps`` and ``direction`` (optional ``irrep_list``)
        -- the subduction (``--correlate --parent SG --irrep IR
        --order-parameter ...``): which Gamma irreps of the isotropy
        subgroup H of that direction a parent irrep becomes, e.g. which
        Raman-active modes a zone-boundary soft mode turns into.

    Args:
        point_group: (a) The point group G in Hermann-Mauguin notation
            (``m-3m``, ``6/mmm``, ``-43m``).
        subgroup: (a) The subgroup type H (``4/mmm``, ``mm2``, ``3m``).
        space_group: (b) Space group as a symbol (``Pm-3m``) or number.
        kpoints: (b) One or two special-point names (``"GM X"`` or
            ``["GM", "X"]``; ``"GM"`` alone for every line from Gamma), or
            three coordinates per point.
        line: (b) Optional ISO-IR name of the line (``DT``) when several
            lines join the two points.
        parent: (c) Parent space group as a symbol (``Pm-3m``) or number.
        irreps: (c) The ISO-IR label(s) of the order parameter (``R4+``, or
            ``["R4+", "M3+"]`` for a coupled one).
        direction: (c) The order-parameter components of those irreps in
            order, letters for free and numbers for fixed ones (``"a 0 0"``,
            ``"0 a a d 0 0"``).
        irrep_list: (c) Optional parent irreps to subduce (any tabulated
            irrep, e.g. ``["R4+", "R5+", "M3+", "GM4-"]``); by default the
            ``irreps`` themselves.

    Returns:
        Markdown: (a) the orientations of H and a table with one row per
        irrep of G; (b) per line a table of the end-point irreps and their
        line irreps; (c) the subgroup H with its ISO-IR frame and a table of
        the Gamma irreps of H (with Mulliken symbols) of every parent irrep.
    """
    def given(value) -> bool:
        if value is None:
            return False
        if isinstance(value, (list, tuple)):
            return bool(value)
        return bool(str(value).strip())

    forms = {
        "a": (given(point_group), ("subgroup",), (subgroup,)),
        "b": (given(space_group), ("kpoints", "line"), (kpoints, line)),
        "c": (given(parent), ("irreps", "direction", "irrep_list"),
              (irreps, direction, irrep_list)),
    }
    selected = [name for name, (on, _, _) in forms.items() if on]
    if len(selected) != 1:
        raise ValueError(
            "give exactly one of point_group (with subgroup) for a correlation table, "
            "space_group (with kpoints) for compatibility relations, or parent (with "
            "irreps and direction) for the subduction to the Gamma point of an "
            "isotropy subgroup."
        )
    form = selected[0]
    stray = [
        name
        for other, (_, names, values) in forms.items() if other != form
        for name, value in zip(names, values) if given(value)
    ]
    leading = {"a": "point_group", "b": "space_group", "c": "parent"}[form]
    if stray:
        verb = "belongs" if len(stray) == 1 else "belong"
        raise ValueError(
            f"{', '.join(stray)} {verb} to another form of the correlation than the "
            f"one {leading} selects; give the arguments of exactly one form."
        )
    if form == "a":
        if not given(subgroup):
            raise ValueError(
                "the correlation table needs subgroup, the subgroup type H in "
                "Hermann-Mauguin notation, e.g. 4/mmm for point_group m-3m."
            )
        return _correlation_markdown(parse_point_group(point_group), parse_point_group(subgroup))
    if form == "b":
        if not given(kpoints):
            raise ValueError(
                'the compatibility relations need kpoints, one or two special-point '
                'names such as "GM X" (crystod_special_kpoints lists them).'
            )
        return _compatibility_markdown(parse_space_group(space_group), kpoints, line)
    if not given(irreps):
        raise ValueError(
            "the subduction needs irreps, the ISO-IR label(s) of the order parameter "
            "such as R4+, together with direction."
        )
    return _subduction_markdown(parse_space_group(parent), irreps, direction, irrep_list)


def _tensor_kind(kind: object) -> str:
    from crystod.tensor_form import KINDS, parse_jahn

    text = "" if kind is None else str(kind).strip()
    names = ", ".join(KINDS)
    if not text:
        raise ValueError(
            f"give the tensor kind: one of {names}, or a Jahn symbol such as V[V2], "
            "[[V2]2] or e[V2]."
        )
    if text.lower() in KINDS:
        return text.lower()
    try:
        parse_jahn(text)
    except ValueError as exc:
        reason = " ".join(str(exc).split())
        raise ValueError(
            f'"{text}" is not a tensor kind or a Jahn symbol ({reason}); give one of '
            f"{names}, or a Jahn symbol such as V[V2], [[V2]2] or e[V2] (prefix e: "
            "axial, a: time-reversal odd)."
        ) from None
    return text


def _raman_block(tensors, setting, cell) -> str:
    """The ``* Raman tensors *`` block: every Raman-active irrep of the point
    group, or the Raman-active Gamma phonon irreps of a structure."""
    if cell is None:
        return tensors.format_raman_forms(
            tensors.raman_forms(setting=setting), setting,
            axes_line="Cartesian axes of the point group above; every Raman-active "
                      "irrep, Mulliken labels of the point-group character table",
        )
    from crystod.phonon_activity import gamma_raman_tensors, mulliken_symbols

    return tensors.format_raman_forms(
        gamma_raman_tensors(cell), setting, mulliken=mulliken_symbols(cell),
        axes_line="Cartesian axes of the POSCAR (not the standard axes above); "
                  "the Gamma phonon irreps of the structure, ISO-IR [Mulliken]",
    )


def _matrix_markdown(form) -> str:
    if form.layout == "list":
        if not form.matrix:
            return "Every component is zero."
        return markdown_table(("Component", "Value"), [list(pair) for pair in form.matrix])
    if form.layout == "vector":
        return markdown_table((form.letter, "1", "2", "3"), [[""] + list(form.matrix[0])])
    n_columns = len(form.matrix[0])
    return markdown_table(
        [form.letter] + [str(c + 1) for c in range(n_columns)],
        [[str(r + 1)] + list(row) for r, row in enumerate(form.matrix)],
    )


@blocking_tool
def crystod_tensor_form(
    kind: str,
    point_group: str | None = None,
    space_group: str | None = None,
    poscar: str | None = None,
) -> str:
    """Symmetry-allowed form of a property tensor (Neumann's principle) in a point group, a space group's point group or a structure's point group: the independent components, the matrix form in Nye's notation and the relations, e.g. the piezoelectric d_iL of 4mm (d15, d31, d33) or the Raman tensor of every Raman-active irrep.

    ``crystod-group --tensor KIND --pg PG | --sg SG | -c POSCAR``, the
    offline counterpart of the TENSOR program of the Bilbao Crystallographic
    Server and of Nye's tables. Give exactly one of ``point_group``,
    ``space_group`` and ``poscar``. Axes: x, y, z along a, b, c for the
    orthogonal lattices, x || a and z || c for the trigonal and hexagonal
    groups, y || b (unique axis) for the monoclinic ones; a space group
    takes the point group of its default setting, a structure that of
    spglib's standard conventional cell (the tensor axes in the frame of the
    POSCAR are listed). A time-odd tensor (Jahn prefix ``a``) vanishes in
    every non-magnetic point group, as the result states.

    Args:
        kind: ``dielectric``, ``pyroelectric``, ``piezoelectric``,
            ``elastic``, ``compliance``, ``gyration``, ``raman`` (the Raman
            tensor of every Raman-active irrep), or a Jahn symbol such as
            ``V[V2]``, ``[[V2]2]``, ``e[V2]`` (``e`` axial, ``a`` time-odd; up
            to rank 6).
        point_group: Point group in Hermann-Mauguin notation (``4mm``,
            ``6/mmm``, ``-43m``).
        space_group: Space group as a symbol (``P6_3mc``) or number (``186``);
            its point group in the default setting.
        poscar: The full text of a VASP POSCAR; the point group of the
            structure. With ``raman``, the Raman-active Gamma phonon irreps
            of the structure with ISO-IR and Mulliken labels, in the axes of
            the POSCAR.

    Returns:
        Markdown: the point group (order, crystal system, Laue class, polar
        or chiral, axes, symmetry elements), the tensor (Jahn symbol, rank,
        intrinsic symmetry), the number and names of the independent
        components, the matrix form and the relations; for ``raman``, the
        Raman tensors of every Raman-active irrep.
    """
    from ..utils import read_poscar_cell, temporary_files, validate_poscar

    try:
        from crystod import tensor_form as tensors
    except ImportError:
        raise _needs_043("the property-tensor forms") from None

    name = _tensor_kind(kind)
    sources = [value is not None and str(value).strip() != ""
               for value in (point_group, space_group, poscar)]
    if sum(sources) != 1:
        raise ValueError(
            "give exactly one of point_group (e.g. 4mm), space_group (e.g. P6_3mc) or "
            "poscar (the full text of a POSCAR) for the point group of the tensor."
        )
    cell = None
    if point_group is not None and str(point_group).strip():
        source = {"point_group": parse_point_group(point_group)}
    elif space_group is not None and str(space_group).strip():
        source = {"space_group": parse_space_group(space_group)}
    else:
        text = validate_poscar(poscar)
        with temporary_files({"POSCAR": text}) as directory:
            try:
                cell = read_poscar_cell(directory / "POSCAR")
            except Exception as exc:  # noqa: BLE001 - the reader reports many shapes of bad input
                raise ValueError(
                    f"the POSCAR text could not be read ({' '.join(str(exc).split())}); "
                    "pass the full text of a VASP POSCAR in the VASP 5 format."
                ) from None
        source = {"cell": cell}
    with _as_value_error():
        setting = tensors.point_group_setting(**source)
        form = tensors.tensor_form_of_operations(setting.rotations, name, setting=setting)
    block = tensors.format_point_group(setting)
    jahn = form.jahn
    nature = (f"{'axial' if jahn.axial else 'polar'}, time-reversal "
              f"{'odd' if jahn.time_odd else 'even'}")
    if name == "raman":
        title = "Raman tensors"
    elif name in tensors.KINDS:
        title = f"{name.capitalize()} tensor"
    else:
        title = f"Tensor {jahn.symbol}"
    lines = [
        f"## {title} in point group {setting.symbol} ({setting.schoenflies})",
        "",
        "### Point group",
        "",
    ]
    lines += [f"- {entry.strip().replace('structure the structure', 'the POSCAR structure')}"
              for entry in block[1:]]
    lines += [
        "",
        "### Tensor",
        "",
        f"- kind: {form.description}",
        f"- Jahn symbol: {jahn.symbol} ({nature}), rank {jahn.rank}",
        "- intrinsic symmetry: " + ("; ".join(jahn.intrinsic) or "none"),
    ]
    notation = tensors._MATRIX_NOTATION.get(form.voigt) if hasattr(tensors, "_MATRIX_NOTATION") else None
    if notation:
        lines.append(f"- {notation}")
    elif form.layout in ("3x6", "6x6"):
        lines.append("- matrix notation: L = 1..6 for the index pairs 11, 22, 33, 23, 13, "
                     "12 (tensor components, no factors)")

    if name == "raman":
        with _as_value_error():
            raman = _raman_block(tensors, setting, cell)
        raman_lines = raman.splitlines()
        lines += ["", "### Raman tensors", "", raman_lines[1].strip() + ".", ""]
        body = [entry.strip() for entry in raman_lines[2:] if entry.strip()]
        lines.append(fenced("\n".join(body)) if body else "No Raman-active irrep.")
        return "\n".join(lines)

    count = form.n_independent
    noun = "component" if count == 1 else "components"
    listed = f": {', '.join(form.independent)}" if form.independent else ""
    lines += ["", f"**{count} independent {noun}{listed}**", "", "### Matrix form", ""]
    lines.append(_matrix_markdown(form))
    lines += ["", "### Relations", ""]
    if form.forbidden_by_time_reversal:
        lines.append(
            f"Time-reversal odd: forbidden in the non-magnetic point group "
            f"{setting.symbol}1' (time reversal is a symmetry of a non-magnetic crystal)."
        )
    elif not form.independent:
        lines.append("Every component is zero.")
    else:
        lines += [f"- {relation}" for relation in form.relations]
        if not form.relations:
            lines.append("- none: the nonzero components are independent")
        import numpy as np

        if any(not np.any(np.abs(form.basis[:, c]) > 1e-10)
               for c in range(len(form.components))):
            lines.append("- every other component is zero")
    return "\n".join(lines)


__all__ = [
    "crystod_isotropy_subgroups",
    "crystod_irrep_product",
    "crystod_decompose_representation",
    "crystod_ligand_field",
    "crystod_character_table",
    "crystod_invariants",
    "crystod_find_isotropy_irreps",
    "crystod_subgroup_graph",
    "crystod_correlate",
    "crystod_tensor_form",
]
