"""Brillouin-zone tool (the ``crystod-bz`` domain): special k points."""

from __future__ import annotations

from ..utils import (
    blocking_tool,
    format_kpoint,
    markdown_table,
    parse_space_group,
    space_group_of_poscar,
    special_kpoints,
    validate_poscar,
)


@blocking_tool
def crystod_special_kpoints(
    space_group: str | None = None,
    poscar: str | None = None,
) -> str:
    """List the special (high-symmetry) k points of a space group with their labels and coordinates, e.g. GM, X, M, R of Pm-3m, from the space group itself or from a POSCAR.

    The labels and coordinates are the ISO-IR convention that all other
    CrystOD tools use for irrep names (``crystod-bz --show-kpoint --sg SG``):
    primitive reciprocal basis, and for centred lattices (F, I, C, A, B, R)
    the conventional basis as well, where the two differ. Give either the
    space group or a POSCAR (whose space group is detected with spglib).

    Args:
        space_group: Space group as an international symbol (``Pm-3m``,
            ``Fm-3m``, ``Pnma``) or number (``221``).
        poscar: Full text of a VASP POSCAR instead of the space group.

    Returns:
        Markdown: a table of k-point labels with primitive (and, for centred
        lattices, conventional) fractional coordinates.
    """
    has_sg = space_group is not None and str(space_group).strip() != ""
    has_poscar = poscar is not None and str(poscar).strip() != ""
    if has_sg == has_poscar:
        raise ValueError(
            "give exactly one of space_group (a symbol such as Pm-3m or a number) "
            "or poscar (the full POSCAR text)."
        )
    detected = ""
    if has_poscar:
        number, symbol = space_group_of_poscar(validate_poscar(poscar))
        sg = str(number)
        detected = f"Space group detected from the POSCAR with spglib: {symbol} (No. {number})."
    else:
        sg = parse_space_group(space_group)

    sg_type, names, primitive, conventional = special_kpoints(sg)
    headers = ["Label", "Primitive reciprocal basis"]
    rows: list[list[str]] = [[name, format_kpoint(k)] for name, k in zip(names, primitive)]
    if conventional is not None:
        headers.append("Conventional reciprocal basis")
        for row, k in zip(rows, conventional):
            row.append(format_kpoint(k))

    lines = [
        f"## Special k points of {sg_type.international_short} (No. {sg_type.number})",
        "",
    ]
    if detected:
        lines += [detected, ""]
    lines.append(markdown_table(headers, rows))
    lines.append("")
    if conventional is None:
        lines.append(
            "Primitive lattice: the primitive and conventional reciprocal bases coincide."
        )
    else:
        lines.append(
            "Centred lattice: the primitive coordinates are the ones CrystOD's irrep "
            "labels refer to; the conventional ones are given for comparison with "
            "band-structure plots in the conventional setting."
        )
    lines.append(
        "The irreps at each k point: crystod_character_table(group, kpoint); the "
        "coordinates are accepted as the kpoint/qpoint of the other tools."
    )
    return "\n".join(lines)


__all__ = ["crystod_special_kpoints"]
