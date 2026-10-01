"""Crystal-orbital SALC tool (the flagship ``crystod`` command).

The printed report of ``crystod -c POSCAR --element EL --orbital ORB`` is the
deliverable, so the command runs in a subprocess on a temporary POSCAR and
its standard output is returned.
"""

from __future__ import annotations

from ..utils import (
    ORBITALS,
    blocking_tool,
    fenced,
    parse_element,
    parse_kpoint,
    parse_orbital,
    resolve_kpoint_label,
    run_cli,
    space_group_of_poscar,
    validate_poscar,
)


@blocking_tool
def crystod_crystal_orbital_irreps(
    poscar: str,
    element: str,
    orbital: str,
    kpoint: str | list[float] | None = None,
) -> str:
    """Irreps of the crystal orbitals (Bloch SALCs) built from one element's orbital shell in a crystal, e.g. Sc 3d of cubic ScF3 at Gamma gives GM3+ (eg) + GM5+ (t2g).

    The site symmetry, Wyckoff position and, per k point, the little group and
    the decomposition of the SALC representation into ISO-IR irreps are
    reported as printed by ``crystod -c POSCAR --element EL --orbital ORB``.
    Without a k point every tabulated special point of the space group is
    analyzed. The irrep labels are the ones a band structure at that k point
    carries and the ones ``crystod_character_table`` lists.

    Args:
        poscar: Full text of the VASP POSCAR (VASP 5 format with the element
            line); it is converted to the primitive cell.
        element: Element whose sites carry the orbitals, e.g. ``Sc``, ``Ti``,
            ``O``.
        orbital: The shell letter: ``s``, ``p``, ``d``, ``f`` (also g, h, i).
        kpoint: A k-point label of the space group (``GM``, ``X``, ``M``,
            ``R``, ...) or three fractional coordinates in the primitive
            reciprocal basis (``"1/2 1/2 1/2"``). Omitted, all special points.

    Returns:
        Markdown with the CrystOD report: space group, Wyckoff letter and site
        symmetry of the element, and for each k point the little group and the
        irrep decomposition (irrep dimensions in parentheses).
    """
    poscar_text = validate_poscar(poscar)
    symbol = parse_element(element)
    shell = parse_orbital(orbital, ORBITALS)
    label, coordinates = parse_kpoint(kpoint)

    argv = ["-c", "POSCAR", "--element", symbol, "--orbital", shell]
    heading = f"## Crystal-orbital irreps of {symbol} {shell}"
    if label is not None:
        number, _ = space_group_of_poscar(poscar_text)
        coordinates = resolve_kpoint_label(str(number), label)
        heading += f" at {label}"
    if coordinates is not None:
        argv += ["--kpoint", *(repr(float(v)) for v in coordinates)]
    else:
        heading += " at every special k point"

    report = run_cli("crystod.cli.main", argv, {"POSCAR": poscar_text})
    return "\n".join([heading, "", fenced(report)])


__all__ = ["crystod_crystal_orbital_irreps"]
