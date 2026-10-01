"""crystod-mcp: the CrystOD symmetry analyses as Model Context Protocol tools.

The server exposes ten text-in / text-out tools built on the public Python
API of CrystOD (``crystod.group``, ``crystod.phonon``, ``crystod.bz``,
``crystod.mol`` and the ``crystod`` command line): isotropy subgroups,
space-group irrep products, reducible-representation and ligand-field
decompositions, character tables, phonon irrep labels, imaginary-mode
subgroups, crystal-orbital (SALC) irreps, special k points and molecular
point groups. Start it with ``crystod-mcp`` or ``python -m crystod_mcp``.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
