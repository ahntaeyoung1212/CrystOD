"""Brillouin zones, k paths and special k points (the ``crystod-bz`` domain).

``crystod.bz`` is the Python form of the ``crystod-bz`` command. It builds the
first Brillouin zone of a lattice as a polyhedron, generates the seekpath
high-symmetry k path or parses a manual one, folds the reciprocal lattice of
a supercell into the unit-cell zone (``--trans-mat``), and lists the ISO-IR
special k points of a space group (``--show-kpoint``). The plotting helpers
return plain Plotly ``scatter3d`` trace dictionaries, so the geometry can be
rendered by any Plotly front end or read as numbers. Reciprocal lattices are
``(3, 3)`` arrays whose rows are ``b1``, ``b2``, ``b3``; the command uses
``inv(lattice).T``, without the factor of 2 pi.

**Brillouin-zone geometry and k paths** (``crystod-bz -c POSCAR``)

- ``get_brillouin_zone_3d``: vertices, edges and facets of the first
  Brillouin zone by Voronoi decomposition of the reciprocal lattice.
- ``get_seekpath_kpath``: the seekpath path of a cell as coordinate segments,
  labels, standardized primitive lattice and space group.
- ``parse_manual_band``: a ``--band`` string as ``(N, 3)`` segments.
- ``prettify_label``: ``GAMMA``, ``X_1`` and the like as plot labels.
- ``build_bz_traces``: the Plotly traces of the zone with its k path.

**Supercell folding** (``crystod-bz --trans-mat``)

- ``parse_transformation_matrix``: a ``--trans-mat`` string as a ``(3, 3)``
  matrix.
- ``get_folded_gamma_points``: the ``|det T|`` unit-cell q points that fold
  onto the supercell Gamma point.
- ``build_supercell_bz_traces``: the Plotly traces of the unit-cell zone
  tiled with the supercell zone at those points.

**Special k-point tables** (``crystod-bz --show-kpoint``)

- ``get_special_kpoints``: the ISO-IR special k points of a space group in
  the primitive (and, for centred lattices, conventional) reciprocal basis.

A typical session::

    import numpy as np
    from phonopy.interface.vasp import read_vasp
    from crystod import bz
    from crystod.examples import example_path

    cell = read_vasp(example_path("221_PPOSCAR_ScF3"))
    segments, labels, lattice, symbol, number = bz.get_seekpath_kpath(cell, 1e-5)
    rec_lat = np.linalg.inv(lattice).T
    vertices, ridges, facets = bz.get_brillouin_zone_3d(rec_lat)
    traces = bz.build_bz_traces(rec_lat, segments, labels)

    sg_type, names, primitive, conventional = bz.get_special_kpoints("Fm-3m")

Attributes resolve lazily (PEP 562): ``import crystod.bz`` costs nothing beyond
NumPy, and scipy, seekpath, phonopy and the ISO-IR tables are loaded by the
first call that needs them. Bad input raises ``ValueError`` through this
namespace (the implementation modules raise ``SystemExit``, which suits the
command line).
"""

from __future__ import annotations

from ._api import lazy_namespace

_EXPORTS = {
    # Brillouin-zone geometry and k paths (crystod-bz)
    "get_brillouin_zone_3d": ("brillouin_zone", "get_brillouin_zone_3d"),
    "get_seekpath_kpath": ("brillouin_zone", "get_seekpath_kpath"),
    "build_bz_traces": ("brillouin_zone", "build_bz_traces"),
    "parse_manual_band": ("brillouin_zone", "parse_manual_band"),
    "prettify_label": ("brillouin_zone", "prettify_label"),
    # supercell folding (crystod-bz --trans-mat)
    "parse_transformation_matrix": ("bz_supercell", "parse_transformation_matrix"),
    "get_folded_gamma_points": ("bz_supercell", "get_folded_gamma_points"),
    "build_supercell_bz_traces": ("bz_supercell", "build_supercell_bz_traces"),
    # special k-point tables (crystod-bz --show-kpoint)
    "get_special_kpoints": ("show_kpoints", "get_special_kpoints"),
}

__getattr__, __dir__, __all__ = lazy_namespace(globals(), _EXPORTS)
