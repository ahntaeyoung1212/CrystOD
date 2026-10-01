"""Group-theory tools (the ``crystod-group`` domain).

Isotropy subgroups, space-group and point-group irrep products, reducible
representation and ligand-field decompositions, and character tables, all
on ``crystod.group``.
"""

from __future__ import annotations

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
    """The unrounded (1/|G|) sum_C m_C chi_irrep(C) chi(C), to detect bad input."""
    import numpy as np

    multiplicity = np.array(_class_multiplicities(character_table), dtype=float)
    reducible = np.array(characters, dtype=float)
    order = float(multiplicity.sum())
    raw = {}
    for irrep, values in character_table["character_table"].items():
        raw[irrep] = float(np.sum(multiplicity * np.real(np.asarray(values)) * reducible)) / order
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
    chi(C), as ``crystod-group --decompose --pg PG --characters ...``. The
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
            " -- except that in this point group an E irrep stands for a pair of "
            "complex-conjugate one-dimensional irreps, so 2(E) is one doubly "
            "degenerate level."
            if pg.replace(" ", "") in _CONJUGATE_PAIR_GROUPS else "."
        ),
    ]
    return "\n".join(lines)


# point groups whose character tables carry complex-conjugate pairs of
# one-dimensional irreps (shown as a doubled E); the multiplicity of such
# an E counts both partners
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


__all__ = [
    "crystod_isotropy_subgroups",
    "crystod_irrep_product",
    "crystod_decompose_representation",
    "crystod_ligand_field",
    "crystod_character_table",
]
