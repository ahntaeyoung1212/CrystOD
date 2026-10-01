"""Molecular symmetry, molecular SALCs and MO diagrams (``crystod-mol``).

The molecular domain of CrystOD, the Python form of the ``crystod-mol``
command and the molecular counterpart of the crystalline SALC analysis: the
point group of a molecule given as an XYZ file is detected, the
symmetry-adapted linear combinations (SALCs) of the atomic orbitals on a set
of equivalent sites are projected out per irrep with the same point-group
character tables as ``crystod-group``, and molecular-orbital diagrams are
drawn as interactive HTML pages, either semi-quantitatively from symmetry
and overlap (symmetry-adapted extended Hueckel) or quantitatively from PySCF
calculations.

Point group and SALCs (``crystod-mol --symmetry``, ``--element/--orbital``):

- ``load_molecule`` reads an XYZ file into a centered pymatgen ``Molecule``.
- ``get_symmetry`` returns the Schoenflies symbol and the symmetry operations
  (pymatgen's ``PointGroupAnalyzer``).
- ``get_permutation_matrices`` builds the site-permutation matrix of every
  operation on a set of sites.
- ``project_salcs`` projects the explicit SALCs of one orbital shell out of
  the permutation x orbital representation, irrep by irrep.
- ``format_salc`` writes one SALC vector as a readable linear combination.
- ``SCHOENFLIES_TO_HM`` maps the Schoenflies symbols of the 32
  crystallographic point groups to their Hermann-Mauguin symbols.

MO diagrams (``crystod-mol --diagram``):

- ``MODiagram`` draws the diagram of a single-center molecule (central atom
  plus ligands: NH3, CH4, SF6, ...) from symmetry and overlap, in the columns
  ligand AOs, ligand SALCs, MOs and central-atom AOs.
- ``EhtFragmentDiagram`` draws the three-column diagram of a molecule split
  into two arbitrary fragments by formula (``--ao-left/--ao-right``), in the
  same extended-Hueckel AO space.
- ``PyscfDiagram`` makes the fragment diagram quantitative with three PySCF
  SCF calculations in one AO space (``--pyscf``; needs the ``[quantum]``
  extra).
- ``AtomicOrbital``, ``build_basis``, ``overlap_matrix``,
  ``hamiltonian_matrix`` and ``EHT_PARAMETERS`` are the extended-Hueckel
  building blocks: the single-zeta STO valence basis, its exact two-center
  overlap matrix and the Wolfsberg-Helmholz Hamiltonian.

Attributes resolve lazily (PEP 562): importing this module is instant,
pymatgen is loaded by the first call and PySCF only by ``PyscfDiagram``. Bad
input makes the functions raise ``ValueError`` (the command-line form of the
same code stops with an ``ERROR:`` line); the diagram classes raise
``SystemExit`` like the command.
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
