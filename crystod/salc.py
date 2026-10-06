"""Crystal-orbital SALC analysis: the Python face of the ``crystod`` command.

Everything the main command does with a structure file is available here
with a ``phonopy.structure.atoms.PhonopyAtoms`` cell in place of the
``-c POSCAR`` argument: the irreducible representations of the crystal
orbitals -- the symmetry-adapted linear combinations (SALCs) of one atomic
shell's Bloch sums -- at any k point, the SALC coefficient bases behind
the interactive 3D viewer, the crystal-orbital diagrams built from
extended-Hueckel or periodic PySCF overlaps, and the star of a k point.
The classes take the same inputs as the command-line flags they mirror
and return the numbers the command prints.

SALC irreps at a k point (``crystod -c POSCAR --element EL --orbital ORB``)
    :class:`CrystalOrbital`
        Irrep decomposition of one element's shell at a k point, or at
        every special point, with ISO-IR labels; ``--spinor`` is the
        ``spior`` flag of the constructor.

Crystal-orbital diagrams (``crystod --diagram``)
    :class:`CrystalOrbitalDiagram`
        The symmetry + extended-Hueckel engine: fragment sublattices from
        ``--co-left``/``--co-right``, full core + valence basis, one
        ``solve_at`` per special k point.
    :class:`PySCFCrystalOrbitalDiagram`
        The quantitative engine of ``--pyscf``: three periodic PySCF
        calculations sharing one AO space, deep-level column alignment
        (needs ``pip install "CrystOD[quantum]"``).
    :func:`assign_bond_characters`
        The COOP bonding/antibonding/nonbonding classification both
        engines apply to their levels.
    :func:`dipole_selection_rules`
        ``dipole_selection_rules(diagram)``: the band-edge electric-dipole
        selection rules of every k point of a diagram object of any engine
        (Gamma_f x Gamma_V x Gamma_i per polarization, in the Cartesian
        axes of the input cell), the blocks the engines print;
        :func:`band_edge_selection_rules` does one k point from its solved
        levels, :func:`little_group_dipole_table` gives the rule for every
        irrep pair of the little group, and
        :func:`format_dipole_selection_rules` the report block.

Symmetry-adapted orbital bases (``crystod --visualize``)
    :class:`SymmetryAdaptedOrbitalBasis`
        Explicit SALC coefficient vectors of one shell at a k point, from
        the projected representation matrices; the data of the SALC viewer.

Star of k (``crystod --star-of-k``)
    :func:`compute_star`
        The arms of the star of a k point and the operations reaching each.
    :func:`format_star_lines`
        The report lines of those arms.
    :func:`resolve_kpoint_input`
        A ``--kpoint`` argument (label or coordinates) as label plus
        coordinates.

Attributes resolve lazily (PEP 562): ``import crystod.salc`` is instant;
the implementation modules, and with them phonopy and spgrep (and pyscf
for the PySCF diagram), are imported the first time an attribute is used.
Bad input that the command line reports as ``ERROR: ...`` and exits on is
raised as ``ValueError`` from the functions of this namespace; the classes
raise ``SystemExit`` from their constructors and methods, as the
implementation modules do.
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
    "dipole_selection_rules": ("selection_rules", "dipole_selection_rules"),
    "band_edge_selection_rules": ("selection_rules", "band_edge_selection_rules"),
    "little_group_dipole_table": ("selection_rules", "little_group_dipole_table"),
    "format_dipole_selection_rules": ("selection_rules", "format_dipole_selection_rules"),
    # symmetry-adapted orbital bases (crystod --visualize)
    "SymmetryAdaptedOrbitalBasis": ("visualize_basis", "SymmetryAdaptedOrbitalBasis"),
    # star of k (crystod --star-of-k)
    "compute_star": ("star_of_k", "compute_star"),
    "format_star_lines": ("star_of_k", "format_star_lines"),
    "resolve_kpoint_input": ("star_of_k", "resolve_kpoint_input"),
}

__getattr__, __dir__, __all__ = lazy_namespace(globals(), _EXPORTS)
