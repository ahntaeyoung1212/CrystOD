"""Public magnetism API of CrystOD: the ``crystod-mag`` domain.

This module covers the symmetry analysis of spin arrangements. The spins on
the sites of one element are treated as axial-vector degrees of freedom, the
3N-dimensional spin space is decomposed into irreps of the space group at a q
point, and the resulting symmetry-adapted spin bases (cluster multipoles /
symmetry-adapted multipole moments, SAMM, after M.-T. Suzuki et al., Phys.
Rev. B 95, 094406 (2017) and Phys. Rev. B 99, 174407 (2019)) are classified
as ferromagnetic (cluster dipole) or antiferromagnetic (octupole and higher).
It mirrors the ``crystod-mag`` command, whose spin listings, VESTA exports and
noncollinear magnetization input (VASP ``MAGMOM``, Quantum ESPRESSO
``starting_magnetization``/``angle1``/``angle2``) are built from these
functions; the implementation lives in ``crystod.spin_basis``.

**Spin representation** (``crystod-mag --qpoint``, and the survey over all
special k points when ``--qpoint`` is omitted)

- ``get_spin_representation``: the little-group irreps at q and the
  axial-vector representation on the selected sites, ready for
  ``spgrep.representation.project_to_irrep``.

**Multipole classification** (``crystod-mag`` at q = 0)

- ``get_multipole_rank_lists``: for every irrep, the ranks of the magnetic
  multipoles (dipole, octupole, ...) in which it appears.
- ``separate_ferro_combination``: splits the net-moment combination (FM,
  cluster dipole) off the projected spaces of one irrep; the rest are AFM.
- ``MULTIPOLE_NAMES``: rank-to-name table (``1: "dipole"``,
  ``3: "octupole"``, ...).

A session mirrors the command::

    from phonopy.interface.vasp import read_vasp
    from spgrep.representation import project_to_irrep
    from crystod import mag, phonon

    vib = phonon.SymmetryOnlyVibrations(read_vasp("POSCAR"), standardize=False)
    sites = [i for i, s in enumerate(vib.primitive_cell.symbols) if s == "Ni"]
    q = [0, 0, 0]
    irreps, spin_rep, mapping = mag.get_spin_representation(vib, sites, q)
    labels = vib.get_irrep_labels(q, irreps, mapping)
    ranks = mag.get_multipole_rank_lists(vib, vib.rotations[mapping], irreps)
    for irrep, label, rank_list in zip(irreps, labels, ranks):
        spaces = [s.real for s in project_to_irrep(spin_rep, irrep)]
        if spaces:
            ferro, afm = mag.separate_ferro_combination(spaces, spin_rep, len(sites))

Attributes resolve lazily (PEP 562): importing this module is instant, and
phonopy/spgrep are loaded on the first attribute access. Bad input raises
``ValueError`` here, where the implementation module raises ``SystemExit``.
"""

from __future__ import annotations

from ._api import lazy_namespace

_EXPORTS = {
    # symmetry-adapted spin bases (crystod-mag)
    "get_spin_representation": ("spin_basis", "get_spin_representation"),
    "get_multipole_rank_lists": ("spin_basis", "get_multipole_rank_lists"),
    "separate_ferro_combination": ("spin_basis", "separate_ferro_combination"),
    "MULTIPOLE_NAMES": ("spin_basis", "MULTIPOLE_NAMES"),
}

__getattr__, __dir__, __all__ = lazy_namespace(globals(), _EXPORTS)
