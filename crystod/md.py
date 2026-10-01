"""Public MD-trajectory API of CrystOD (the ``crystod-md`` domain).

This module mirrors ``crystod-md --adp``: the building blocks that turn a
molecular-dynamics ``XDATCAR`` trajectory into a time-averaged structure with
symmetry-constrained anisotropic displacement parameters (ADPs, the ``U_ij``
tensors of the ``_atom_site_aniso_U_*`` loop of a CIF file). The command
itself (``crystod.xdatcar_adp.main``) chains these functions with the folding
of the MD supercell onto one unit cell, the spglib symmetry search of the
averaged structure and the CIF writer; the API exposes the reusable pieces,
so the same site-symmetry constraints can be applied to displacement
statistics obtained elsewhere.

Trajectory input:

- ``read_xdatcar`` -- read an ``XDATCAR`` (fixed-cell, or NpT with repeated
  headers) into the element list, the per-frame lattices and the fractional
  coordinates of every frame.

Site-symmetry constraints on ``U_ij`` (``crystod-md --adp``):

- ``get_site_symmetry_operations`` -- the space-group rotations that leave a
  site fixed;
- ``build_symmetry_projector`` -- the 6x6 projector onto the ``U`` tensors
  invariant under those rotations;
- ``apply_symmetry_constraints`` -- symmetrize a 3x3 ``U`` tensor with such a
  projector;
- ``get_constraint_description`` -- spell a projector out as
  ``U11=U22, U12=0, ...`` for reports.

The functions chain in that order::

    import spglib
    from crystod import md

    symbols, lattices, frames = md.read_xdatcar("XDATCAR")
    symmetry = spglib.get_symmetry(cell)        # the time-averaged unit cell
    site_ops = md.get_site_symmetry_operations(
        site, symmetry["rotations"], symmetry["translations"])
    projector = md.build_symmetry_projector(site_ops)
    u_ij = md.apply_symmetry_constraints(u_raw, projector)

Attributes resolve lazily (PEP 562): importing this module is instant, and the
implementation module ``crystod.xdatcar_adp`` (numpy and spglib) is loaded on
first use. Functions reached through this namespace report bad input as
``ValueError`` instead of the ``SystemExit`` the implementation raises for the
command line.
"""

from __future__ import annotations

from ._api import lazy_namespace

_EXPORTS = {
    # ADPs from an MD trajectory (crystod-md --adp)
    "read_xdatcar": ("xdatcar_adp", "read_xdatcar"),
    "build_symmetry_projector": ("xdatcar_adp", "build_symmetry_projector"),
    "apply_symmetry_constraints": ("xdatcar_adp", "apply_symmetry_constraints"),
    "get_constraint_description": ("xdatcar_adp", "get_constraint_description"),
    "get_site_symmetry_operations": ("xdatcar_adp", "get_site_symmetry_operations"),
}

__getattr__, __dir__, __all__ = lazy_namespace(globals(), _EXPORTS)
