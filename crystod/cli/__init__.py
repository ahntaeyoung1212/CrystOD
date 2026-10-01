"""CrystOD command-line interfaces.

This package hosts the main command and one module per sectioned command
(phonopy style):

- ``main``    -- ``crystod``: crystal-orbital SALC analysis (the flagship;
  no mode flag needed), --visualize, --star-of-k. A flag belonging to a
  sectioned command is answered with the command that does the job.
- ``bz``      -- ``crystod-bz``: Brillouin-zone plotting.
- ``md``      -- ``crystod-md``: MD-trajectory analyses (ADPs as CIF, summary).
- ``mag``     -- ``crystod-mag``: magnetism analyses (symmetry-adapted spin bases).
- ``phonon``  -- ``crystod-phonon``: phonon analyses (irreps, fatband, lt, vector,
  modulation, vibration).
- ``group``   -- ``crystod-group``: point/space-group representation-theory
  calculator (product, table, decompose, ligand-field, basis, generate-basis,
  coset).
- ``mol``     -- ``crystod-mol``: molecular point groups, SALCs and MO diagrams.
- ``xrd``     -- ``crystod-xrd``: powder X-ray diffraction patterns.
- ``search``  -- ``crystod-search``: Materials Project search and POSCAR download.
"""

from .main import build_parser, main

__all__ = ["build_parser", "main"]
