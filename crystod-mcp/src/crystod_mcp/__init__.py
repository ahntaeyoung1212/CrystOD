"""crystod-mcp: the CrystOD symmetry analyses as Model Context Protocol tools.

The server exposes eighteen text-in / text-out tools built on the public Python
API of CrystOD (``crystod.group``, ``crystod.phonon``, ``crystod.bz``,
``crystod.mol``, ``crystod.slab`` and the ``crystod`` command line): isotropy subgroups and
their reverse lookup, group-subgroup graphs, space-group irrep products,
reducible-representation and ligand-field decompositions, character tables,
Landau invariants and secondary order parameters, correlation tables,
compatibility relations and subductions, property-tensor forms, phonon irrep
labels, imaginary-mode subgroups, Gamma-point IR/Raman activity,
crystal-orbital (SALC) irreps, special k points, molecular point groups and
surface slabs. Start it with ``crystod-mcp`` or ``python -m crystod_mcp``.
"""

__version__ = "0.3.0"

__all__ = ["__version__"]
