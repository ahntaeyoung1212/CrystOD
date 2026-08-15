"""Public molecular API of CrystOD (the ``crystod-mol`` domain).

    from crystod import mol

    molecule = mol.load_molecule("NH3.xyz")
    ops, point_group = mol.get_symmetry(molecule)
    salcs = mol.project_salcs(...)

    # MO diagrams (crystod-mol --diagram; extended Hueckel / fragments / PySCF)
    diagram = mol.MODiagram(...)
    diagram = mol.EhtFragmentDiagram(...)
    diagram = mol.PyscfDiagram(...)

Attributes resolve lazily (PEP 562); pymatgen/pyscf load only on first use.
"""

from __future__ import annotations

from ._api import lazy_namespace

_EXPORTS = {
    # molecular point groups and SALCs (crystod-mol)
    "load_molecule": ("molecular_salc", "load_molecule"),
    "get_symmetry": ("molecular_salc", "get_symmetry"),
    "get_permutation_matrices": ("molecular_salc", "get_permutation_matrices"),
    "project_salcs": ("molecular_salc", "project_salcs"),
    "format_salc": ("molecular_salc", "format_salc"),
    "SCHOENFLIES_TO_HM": ("molecular_salc", "SCHOENFLIES_TO_HM"),
    # MO diagrams (crystod-mol --diagram)
    "MODiagram": ("mo_diagram", "MODiagram"),
    "AtomicOrbital": ("mo_diagram", "AtomicOrbital"),
    "build_basis": ("mo_diagram", "build_basis"),
    "overlap_matrix": ("mo_diagram", "overlap_matrix"),
    "hamiltonian_matrix": ("mo_diagram", "hamiltonian_matrix"),
    "EHT_PARAMETERS": ("mo_diagram", "EHT_PARAMETERS"),
    "EhtFragmentDiagram": ("mo_diagram_fragment", "EhtFragmentDiagram"),
    "PyscfDiagram": ("mo_diagram_pyscf", "PyscfDiagram"),
}

__getattr__, __dir__, __all__ = lazy_namespace(globals(), _EXPORTS)
