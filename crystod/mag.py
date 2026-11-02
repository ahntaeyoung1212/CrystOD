"""Public magnetism API of CrystOD: the ``crystod-mag`` domain.

This module covers the symmetry analysis of spin arrangements. The spins on
the sites of one element are treated as axial-vector degrees of freedom, the
3N-dimensional spin space is decomposed into irreps of the space group at a q
point, and the resulting symmetry-adapted spin bases (cluster multipoles /
symmetry-adapted multipole moments, SAMM, after M.-T. Suzuki et al., Phys.
Rev. B 95, 094406 (2017) and Phys. Rev. B 99, 174407 (2019)) are classified
as ferromagnetic (cluster dipole) or antiferromagnetic (net moment zero). At
q = 0 an AFM basis is named after the next rank at which its irrep occurs in
the magnetic multipole representations: octupole, 32pole, ... for an irrep
even under the inversion, quadrupole, hexadecapole, ... for an odd one (the
magnetic multipole of rank p has parity (-1)**(p + 1); Cr2O3, Cr: GM1-
quadrupole, GM2- hexadecapole, GM1+ octupole).
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
  multipoles (dipole, quadrupole, octupole, ...) in which it appears.
- ``separate_ferro_combination``: splits the net-moment combination (FM,
  cluster dipole) off the projected spaces of one irrep; the rest are AFM.
  ``analyze_spin_basis`` falls back on the real form of the irrep when the
  copies spgrep gives (in a complex basis for the E irreps of trigonal and
  hexagonal crystals) do not separate cleanly, and turns the partners of all
  copies onto directions that symmetry fixes.
- ``MULTIPOLE_NAMES``: rank-to-name table (``1: "dipole"``,
  ``3: "octupole"``, ...).

**Magnetic groups and collinear classes** (``crystod-mag --qpoint``, every
exported component; ``--classify-only`` for the table alone)

- ``analyze_spin_basis``: the whole q-point analysis as data
  (``SpinBasisAnalysis``: decomposition, bases, the real spin field of every
  component and its magnetic group, responses and class);
  ``magnetic_summary_rows`` gives the rows of the summary table and
  ``magnetic_summary_notes`` the notes under it.
- ``magnetic_space_group``: magnetic space group of a spin arrangement
  (spglib): BNS number and symbol, OG number, type I-IV, the magnetic point
  group derived from the BNS symbol and checked against the operations
  (``MagneticGroupInfo``).
- ``macroscopic_responses``: net magnetization, linear magnetoelectric and
  piezomagnetic tensors allowed by the magnetic point group
  (``MagneticResponses``); ``format_magnetic_responses`` prints them.
- ``classify_collinear``: ferromagnet, ferrimagnet, conventional
  antiferromagnet or altermagnet (d/g/i-wave, spin-splitting form,
  spin-degenerate planes) from the sign pattern of a collinear arrangement
  (``CollinearClass``), after L. Smejkal, J. Sinova and T. Jungwirth, Phys.
  Rev. X 12, 031042 (2022).
- ``magnetic_point_group_symbol``: the MPG symbol of a BNS symbol.

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
    # magnetic space groups, responses and collinear classes (crystod-mag)
    "analyze_spin_basis": ("spin_basis", "analyze_spin_basis"),
    "magnetic_summary_rows": ("spin_basis", "magnetic_summary_rows"),
    "magnetic_summary_notes": ("spin_basis", "magnetic_summary_notes"),
    "SpinBasisAnalysis": ("spin_basis", "SpinBasisAnalysis"),
    "SpinBasisComponent": ("spin_basis", "SpinBasisComponent"),
    "magnetic_space_group": ("magnetic_group", "magnetic_space_group"),
    "macroscopic_responses": ("magnetic_group", "macroscopic_responses"),
    "format_magnetic_responses": ("magnetic_group", "format_magnetic_responses"),
    "classify_collinear": ("magnetic_group", "classify_collinear"),
    "magnetic_point_group_symbol": ("magnetic_group", "magnetic_point_group_symbol"),
    "MagneticGroupInfo": ("magnetic_group", "MagneticGroupInfo"),
    "MagneticResponses": ("magnetic_group", "MagneticResponses"),
    "CollinearClass": ("magnetic_group", "CollinearClass"),
}

__getattr__, __dir__, __all__ = lazy_namespace(globals(), _EXPORTS)
