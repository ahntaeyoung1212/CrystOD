"""Public MD-trajectory API of CrystOD (the ``crystod-md`` domain).

This module mirrors ``crystod-md --adp`` and ``crystod-md --project``. For
``--adp`` it holds the building blocks that turn a
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

Mode projection (``crystod-md --project``):

- ``project_trajectory`` -- match the trajectory atoms to the ideal supercell
  of the parent primitive cell and project the displacements of every frame
  onto its symmetry-adapted patterns (the bases of ``crystod-phonon
  --vibration``): ``Q(t)`` per irrep, mode space, component and q point, the
  statistics of each series and the sum rule; returns a
  ``TrajectoryProjection``. With ``phonon=`` (a phonopy object with force
  constants, ``crystod-md --project --yaml``) the basis is the phonon
  eigenvectors at each q instead: one mass-weighted normal coordinate
  ``a(t)`` per mode (the modes of ``crystod-phonon --modulation``), with the
  effective frequency ``sqrt(k_B T/<a^2>)/(2 pi)`` when ``temperature=`` is
  given; ``star=True`` projects every arm of the star of each requested q;
- ``format_projection`` -- the terminal report of ``--project`` for such a
  result;
- ``commensurate_qpoints`` -- the q points a diagonal supercell resolves.

``Q`` is the displacement norm per parent primitive cell carried by a
pattern (at a time-reversal invariant q, the amplitude of ``crystod-phonon
--modulation``; the mass-weighted centre-of-mass drift of every frame is
removed first)::

    symbols, lattices, frames = md.read_xdatcar("XDATCAR")
    result = md.project_trajectory("221_PPOSCAR_ScF3", (4, 4, 4), symbols,
                                   lattices[1000:], frames[1000:],
                                   qpoints=["R", "M"], irreps=["R4+", "M3+"])
    print(md.format_projection(result))

    import phonopy
    phonon = phonopy.load("phonopy_params.yaml", is_nac=False)
    modes = md.project_trajectory(None, (4, 4, 4), symbols, lattices[1000:],
                                  frames[1000:], qpoints=["R", "M", "GM"],
                                  phonon=phonon, temperature=300)
    print(modes.projections[0].columns[0],            # "1_R4+"
          modes.projections[0].effective_frequencies[0])

The functions chain in that order::

    import spglib
    from crystod import md

    symbols, lattices, frames = md.read_xdatcar("XDATCAR")
    symmetry = spglib.get_symmetry(cell)        # the time-averaged unit cell
    site_ops = md.get_site_symmetry_operations(   # tolerance in Angstrom
        site, symmetry["rotations"], symmetry["translations"], lattice=cell[0])
    projector = md.build_symmetry_projector(site_ops, lattice=cell[0])
    u_ij = md.apply_symmetry_constraints(u_raw, projector)   # CIF U^ij

Attributes resolve lazily (PEP 562): importing this module is instant, and the
implementation modules ``crystod.xdatcar_adp`` (numpy and spglib) and
``crystod.md_projection`` are loaded on first use. Functions reached through this namespace report bad input as
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
    # mode projection of an MD trajectory (crystod-md --project)
    "project_trajectory": ("md_projection", "project_trajectory"),
    "format_projection": ("md_projection", "format_projection"),
    "commensurate_qpoints": ("md_projection", "commensurate_qpoints"),
    "TrajectoryProjection": ("md_projection", "TrajectoryProjection"),
}

__getattr__, __dir__, __all__ = lazy_namespace(globals(), _EXPORTS)
