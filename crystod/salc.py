"""Public crystal-orbital SALC API of CrystOD (the flagship ``crystod`` command).

    from crystod import salc

    # SALC irrep analysis at a k point (crystod)
    co = salc.CrystalOrbital(...)

    # crystal-orbital diagrams (crystod --diagram, extended Hueckel / PySCF)
    diagram = salc.CrystalOrbitalDiagram(...)
    diagram = salc.PySCFCrystalOrbitalDiagram(...)

    # symmetry-adapted orbital bases + HTML viewer (crystod --visualize)
    basis = salc.SymmetryAdaptedOrbitalBasis(...)

    # star of k (crystod --star-of-k)
    salc.compute_star(...)

Attributes resolve lazily (PEP 562): importing this module is instant and
pulls in phonopy/spgrep/pyscf only on first use.
"""

from __future__ import annotations

from ._api import lazy_namespace

_EXPORTS = {
    # SALC irreps at k (crystod)
    "CrystalOrbital": ("crystal_orbital_spgrep", "CrystalOrbital"),
    # crystal-orbital diagrams (crystod --diagram)
    "CrystalOrbitalDiagram": ("crystal_orbital_diagram", "CrystalOrbitalDiagram"),
    "assign_bond_characters": ("crystal_orbital_diagram", "assign_bond_characters"),
    "PySCFCrystalOrbitalDiagram": ("crystal_orbital_pyscf", "PySCFCrystalOrbitalDiagram"),
    # symmetry-adapted orbital bases (crystod --visualize)
    "SymmetryAdaptedOrbitalBasis": ("visualize_basis", "SymmetryAdaptedOrbitalBasis"),
    # star of k (crystod --star-of-k)
    "compute_star": ("star_of_k", "compute_star"),
    "format_star_lines": ("star_of_k", "format_star_lines"),
    "resolve_kpoint_input": ("star_of_k", "resolve_kpoint_input"),
}

__getattr__, __dir__, __all__ = lazy_namespace(globals(), _EXPORTS)
