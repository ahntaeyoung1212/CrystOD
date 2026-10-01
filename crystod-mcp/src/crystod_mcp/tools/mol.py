"""Molecular tool (the ``crystod-mol`` domain): point group and molecular SALCs.

The printed report of ``crystod-mol`` is the deliverable, so the command runs
in a subprocess on a temporary XYZ file and its standard output is returned.
"""

from __future__ import annotations

from ..utils import blocking_tool, fenced, parse_element, parse_orbital, run_cli, validate_xyz


@blocking_tool
def crystod_molecular_symmetry(
    xyz: str,
    element: str | None = None,
    orbital: str | None = None,
) -> str:
    """Point group of a molecule from its XYZ coordinates (e.g. CH4 is Td), and optionally the molecular SALCs of one element's orbital shell (e.g. the four H 1s orbitals of CH4 span A1 + T2).

    Without ``element``/``orbital`` the point group (Schoenflies and
    Hermann-Mauguin symbols) and its symmetry operations by class are
    reported (``crystod-mol --symmetry``). With both, the site-permutation
    representation of that element's atoms is multiplied by the orbital
    characters, decomposed into point-group irreps, and the explicit SALCs
    are printed (``crystod-mol --element EL --orbital ORB``); this needs one
    of the 32 crystallographic point groups (linear molecules are named but
    not decomposed).

    Args:
        xyz: Full text of an XYZ file: number of atoms, comment line, then
            one ``Symbol x y z`` line per atom in Angstrom.
        element: Element whose atoms carry the orbitals for the SALC analysis,
            e.g. ``H``.
        orbital: Orbital shell for the SALC analysis: ``s``, ``p``, ``d`` or
            ``f``. Give ``element`` and ``orbital`` together or neither.

    Returns:
        Markdown with the CrystOD report: molecule formula, point group and
        operations; with element/orbital also the reducible representation per
        class, its decomposition ``Gamma = 1(A1) + 1(T2)`` and the SALCs.
    """
    xyz_text = validate_xyz(xyz)
    if (element is None) != (orbital is None):
        raise ValueError(
            "give both element and orbital for the SALC analysis (e.g. element='H', "
            "orbital='s'), or neither for the point group alone."
        )
    if element is None:
        argv = ["--symmetry", "--xyz", "molecule.xyz"]
        heading = "## Molecular point group"
    else:
        symbol = parse_element(element)
        shell = parse_orbital(orbital, ("s", "p", "d", "f"))
        argv = ["--xyz", "molecule.xyz", "--element", symbol, "--orbital", shell]
        heading = f"## Molecular point group and {symbol} {shell} SALCs"

    report = run_cli("crystod.cli.mol", argv, {"molecule.xyz": xyz_text})
    return "\n".join([heading, "", fenced(report)])


__all__ = ["crystod_molecular_symmetry"]
