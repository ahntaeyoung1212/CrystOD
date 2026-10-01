"""Phonon analyses of CrystOD: the ``crystod-phonon`` command as a library.

This domain covers what ``crystod-phonon`` does from phonopy force data (a
unit cell with ``FORCE_SETS`` or ``FORCE_CONSTANTS``, or a
``phonopy_params.yaml``): labeling phonon modes with ISO-IR space-group
irreps, following imaginary modes to the isotropy subgroups they can condense
into, exporting eigenvectors for VESTA, resolving the longitudinal/transverse
character of bands, and freezing modes into modulated structures. The
symmetry-only vibration bases need no force data at all. Where the command
takes files, the functions take a live ``phonopy.Phonopy`` object; build it
with ``primitive_matrix="auto"``, so that a zone-boundary instability appears
at its own q point instead of being folded onto the supercell Gamma point.

Irrep labeling (``crystod-phonon --irreps``):

- :func:`get_irrep_labels` -- irrep labels, band indices and frequencies of
  the modes at a tabulated q point, from phonopy characters and the ISO-IR
  tables;
- :func:`get_irt_special_points` -- the special q points of the ISO-IR tables
  in the primitive basis;
- :func:`find_star_representative` -- map any arm of a star onto its
  tabulated arm.

Mode labeling and isotropy subgroups (``crystod-phonon --subgroup``; the
structure-search API):

- :func:`label_phonon_modes` -- the levels at one q point as
  :class:`PhononMode` records (1-based band indices, THz frequency, ISO-IR
  labels);
- :func:`imaginary_mode_subgroups` -- the isotropy subgroups of every
  imaginary level at one q point, as :class:`ImaginaryModeResult` records;
- :func:`scan_imaginary_modes` -- the same over every q point the supercell
  resolves, most unstable first;
- :func:`commensurate_qpoints` -- that set of q points;
- :func:`isotropy_subgroups` -- the subgroups of a space-group irrep as
  :class:`IsotropySubgroup` records, no phonopy object needed (the API form
  of ``crystod-group --parent``);
- :func:`isotropy_subgroups_at_kpoint` -- the same for every irrep of one
  special k point (``crystod-group --parent SG --kpoint K``).

Eigenvectors and VESTA export (``crystod-phonon --vector``):

- :func:`build_symmetry_adapted_modes` -- eigenvectors of the dynamical
  matrix whose degenerate partners transform with the irrep matrices (real
  patterns along directions symmetry fixes at time-reversal-invariant q);
- :func:`resolve_qpoint` -- ``--qpoint`` tokens (a label or coordinates) to a
  label and primitive-basis coordinates;
- :func:`get_commensurate_supercell_matrix` -- the smallest supercell on
  which a mode at q is periodic;
- :func:`write_vesta_with_arrows` -- a complete VESTA file with displacement
  arrows.

Longitudinal/transverse character (``crystod-phonon --lt``):

- :func:`get_longitudinal_ratio` -- the longitudinal character of every
  (q point, band) pair: 1 = longitudinal, 0 = transverse.

Modulated structures (``crystod-phonon --modulation``):

- :class:`SymmetryAdaptedModulation` -- the symmetry-adapted modes at one q
  point and the structures obtained by freezing them in (harmonic
  eigen-displacements on phonopy's primitive cell as it is; the amplitude is
  the displacement norm of one primitive cell where 2q is a reciprocal
  lattice vector);
- :class:`ModulationTerm` -- the modes and amplitudes of one q point in a
  combined modulation.

Symmetry-only vibration bases (``crystod-phonon --vibration``):

- :class:`SymmetryOnlyVibrations` -- irrep-projected displacement bases from
  the crystal structure alone.

Attributes resolve lazily (PEP 562): importing this module is instant and
pulls in phonopy/spgrep only on first use. The implementation lives in
``crystod.phonon_irreps``, ``crystod.phonon_subgroups``,
``crystod.phonon_vector``, ``crystod.phonon_lt``, ``crystod.modulation`` and
``crystod.vibration_modes``, whose import paths keep working. Bad input
raises ``ValueError`` from the functions of this namespace, where the
implementation modules exit the process (``SystemExit``) as a command line
wants it.

Example::

    import phonopy
    from crystod import phonon
    from crystod.examples import example_path

    ph = phonopy.load(
        unitcell_filename=example_path("221_PPOSCAR_SrTiO3"),
        force_sets_filename=example_path("FORCE_SETS_SrTiO3"),
        supercell_matrix=[4, 4, 4], primitive_matrix="auto")

    modes = phonon.label_phonon_modes(ph, [0.5, 0.5, 0.5])   # R point
    results = phonon.imaginary_mode_subgroups(ph, [0.5, 0.5, 0.5])
    for result in results:
        print(result.mode)                                    # R5- soft mode
        for sub in result.subgroups:
            print("   ", sub.label, "->", sub.symbol)         # I4/mcm, R-3c, ...
"""

from __future__ import annotations

from ._api import lazy_namespace

_EXPORTS = {
    # irrep labeling (crystod-phonon --irreps)
    "get_irrep_labels": ("phonon_irreps", "get_irrep_labels"),
    "get_irt_special_points": ("phonon_irreps", "get_irt_special_points"),
    "find_star_representative": ("phonon_irreps", "find_star_representative"),
    # high-level labeling / subgroup API (macer-style structure searches)
    "label_phonon_modes": ("phonon_subgroups", "label_phonon_modes"),
    "imaginary_mode_subgroups": ("phonon_subgroups", "imaginary_mode_subgroups"),
    "scan_imaginary_modes": ("phonon_subgroups", "scan_imaginary_modes"),
    "commensurate_qpoints": ("phonon_subgroups", "commensurate_qpoints"),
    "isotropy_subgroups": ("phonon_subgroups", "isotropy_subgroups"),
    "isotropy_subgroups_at_kpoint": ("phonon_subgroups", "isotropy_subgroups_at_kpoint"),
    "PhononMode": ("phonon_subgroups", "PhononMode"),
    "ImaginaryModeResult": ("phonon_subgroups", "ImaginaryModeResult"),
    "IsotropySubgroup": ("phonon_subgroups", "IsotropySubgroup"),
    # eigenvectors / VESTA export (crystod-phonon --vector)
    "resolve_qpoint": ("phonon_vector", "resolve_qpoint"),
    "build_symmetry_adapted_modes": ("phonon_vector", "build_symmetry_adapted_modes"),
    "get_commensurate_supercell_matrix": ("phonon_vector", "get_commensurate_supercell_matrix"),
    "write_vesta_with_arrows": ("phonon_vector", "write_vesta_with_arrows"),
    # longitudinal/transverse character (crystod-phonon --lt)
    "get_longitudinal_ratio": ("phonon_lt", "get_longitudinal_ratio"),
    # modulated structures (crystod-phonon --modulation)
    "ModulationTerm": ("modulation", "ModulationTerm"),
    "SymmetryAdaptedModulation": ("modulation", "SymmetryAdaptedModulation"),
    # symmetry-only vibration bases (crystod-phonon --vibration)
    "SymmetryOnlyVibrations": ("vibration_modes", "SymmetryOnlyVibrations"),
}

__getattr__, __dir__, __all__ = lazy_namespace(globals(), _EXPORTS)
