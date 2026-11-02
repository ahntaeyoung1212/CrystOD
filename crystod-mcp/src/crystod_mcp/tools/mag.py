"""Magnetism tool (the ``crystod-mag`` domain).

The tool reads the POSCAR text from a temporary directory and calls
``crystod.mag.analyze_spin_basis``, the data function behind
``crystod-mag --qpoint``: the symmetry-adapted spin bases of one element at
one q point, each exported component with its magnetic space group, magnetic
point group, the responses that group allows and its collinear class. The
Markdown is built from the returned data, not from the terminal output.
"""

from __future__ import annotations

from fractions import Fraction

from ..utils import (
    blocking_tool,
    markdown_table,
    parse_element,
    parse_kpoint,
    read_poscar_cell,
    temporary_files,
    validate_poscar,
)

_HEADERS = ("Basis", "Component", "MSG (BNS)", "Type", "MPG", "Net magnetization", "Class")


@blocking_tool
def crystod_mag_classify(
    poscar: str,
    element: str,
    qpoint: str = "GM",
    classify_only: bool = True,
) -> str:
    """Classify the symmetry-adapted spin arrangements of a magnetic element: for every spin basis vector at a q point, its magnetic space group (BNS symbol, number and type I-IV), magnetic point group, the net magnetization symmetry allows, and its collinear class (ferromagnet, conventional antiferromagnet, or altermagnet with its d/g/i-wave type, e.g. the rutile-type d-wave altermagnet P4_2'/mnm').

    The analysis of ``crystod-mag --qpoint ... --classify-only``: the spins
    of the selected element are decomposed into ISO-IR irreps of the space
    group at q, ferromagnetic and antiferromagnetic combinations are
    separated at Gamma (at Gamma and the other time-reversal-invariant q
    points a complex irrep and its conjugate form one real space under
    their pair label, such as GM3+GM4+ or Z3+Z4+), and every exported
    component (x, y, z, 111 or comp<N>) is analysed. The magnetic space group follows the moment
    directions (spglib); the collinear class follows only the signs of the
    moments along their common axis (nonrelativistic; L. Smejkal, J. Sinova
    and T. Jungwirth, Phys. Rev. X 12, 031042 (2022)). Symmetry decides what
    is allowed, not what happens.

    Args:
        poscar: Full text of the VASP POSCAR of the crystal (VASP 5 format
            with the element line); the primitive cell of the input is used
            as it is.
        element: The magnetic element whose sites carry the spins, e.g.
            ``Ni``.
        qpoint: A k-point label (``GM``, ``X``, ``M``, ``R``, ``L``, ...) or
            three fractional coordinates in the primitive reciprocal basis
            (``"1/2 1/2 1/2"``); default ``GM``.
        classify_only: Return the structure, the decomposition and the
            summary table only (default true); false adds, per component,
            the spins of the sites, the responses allowed by symmetry and the
            full collinear description.

    Returns:
        Markdown: the space group and magnetic sites, the decomposition of
        the spin representation, and the table (basis, component, MSG with
        BNS number, type, MPG, net magnetization allowed by symmetry,
        class).
    """
    from crystod import mag as crystod_mag
    from crystod.magnetic_group import format_magnetic_group, format_magnetic_point_group

    poscar_text = validate_poscar(poscar)
    symbol = parse_element(element)
    label, coordinates = parse_kpoint(qpoint)
    if label is None and coordinates is None:
        label = "GM"
    tokens = [label] if label else [
        str(Fraction(float(value)).limit_denominator(24)) for value in coordinates
    ]

    with temporary_files({"POSCAR": poscar_text}) as directory:
        try:
            cell = read_poscar_cell(directory / "POSCAR")
        except Exception as exc:  # noqa: BLE001 - the readers report bad input in many ways
            reason = " ".join(str(exc).split()) or type(exc).__name__
            raise ValueError(
                f"the POSCAR text could not be read ({reason}); pass the full text of "
                "a VASP POSCAR in the VASP 5 format."
            ) from None
    analysis = crystod_mag.analyze_spin_basis(cell, symbol, tokens)

    decomposition = " + ".join(
        f"{count} x {name}" if count > 1 else name
        for name, count in analysis.decomposition.items()
    )
    sites = ", ".join(
        f"{symbol}{k + 1} ({', '.join(f'{v:.6g}' for v in position)})"
        for k, position in enumerate(analysis.site_positions)
    )
    q_text = ", ".join(f"{Fraction(float(v)).limit_denominator(24)}" for v in analysis.qpoint)
    lines = [
        f"## Magnetic groups of the {symbol} spin bases of {analysis.space_group} "
        f"(No. {analysis.space_group_number}) at {analysis.qpoint_label} ({q_text})",
        "",
        f"Magnetic sites: {symbol} x {analysis.n_sites}: {sites}.",
        "",
        f"Spin (axial-vector) representation, {3 * analysis.n_sites} dimensions: "
        f"{decomposition}.",
        "",
    ]
    if not analysis.is_gamma:
        sizes = "x".join(str(int(v)) for v in analysis.supercell_sizes)
        lines += [
            f"Magnetic cell: {sizes} supercell of the primitive cell; the magnetic space "
            "group refers to the real spin field of that cell (global phase fixed as the "
            "VESTA export of crystod-mag).",
            "",
        ]
    rows = crystod_mag.magnetic_summary_rows(analysis)
    lines.append(markdown_table(_HEADERS, rows))
    lines.append("")
    if not classify_only:
        for component in analysis.components:
            if component.field is None or component.collinear is None:
                continue
            lines.append(f"### {component.header}, component "
                         f"{component.file_tag or component.tag}")
            lines.append("")
            for k, spin in enumerate(component.field[0]):
                lines.append(f"- {symbol}{k + 1}: S = [{spin[0]:.4f}, {spin[1]:.4f}, {spin[2]:.4f}]")
            group = component.group
            if group is None:
                lines.append("- magnetic space group: not identified by spglib")
            else:
                lines.append(f"- magnetic space group: {format_magnetic_group(group)}")
                lines.append(f"- magnetic point group: {format_magnetic_point_group(group)}")
            if component.responses is not None:
                lines.append(f"- allowed by symmetry: {component.responses.summary()}")
            lines.append(f"- collinear order: {component.collinear.text()}")
            lines.append("")
    note = (
        "MSG: magnetic space group from the moment directions (BNS number; type I-IV). "
        "MPG: magnetic point group. MSG and MPG symbols are in the standard BNS setting. "
        "Net magnetization: the direction allowed by symmetry, in the Cartesian axes of "
        "the input cell. Component: the tag of the VESTA file name of crystod-mag "
        "(comp<N>: the N-th basis vector of its space). Class: from the signs of the "
        "moments along their common axis only (nonrelativistic; L. Smejkal, J. Sinova "
        "and T. Jungwirth, Phys. Rev. X 12, 031042 (2022)); noncollinear components are "
        "not classified."
    )
    if any(c.collinear is not None and c.group is not None and not c.group.identified
           for c in analysis.components):
        note += (" MSG '-': spglib returned the operations but did not identify the group; "
                 "the net magnetization comes from the operations.")
    if analysis.pair_labels:
        note += (f" {', '.join(analysis.pair_labels)}: a complex irrep and its conjugate, "
                 "merged into one real space (the physically irreducible representation).")
    lines.append(note)
    return "\n".join(lines)


__all__ = ["crystod_mag_classify"]
