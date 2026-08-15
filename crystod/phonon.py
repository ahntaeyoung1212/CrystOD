"""Public phonon API of CrystOD (the ``crystod-phonon`` domain).

    from crystod import phonon

    # ISO-IR irrep labels of the modes of a phonopy object at one q point
    modes = phonon.label_phonon_modes(ph, [0.5, 0.5, 0.5])

    # imaginary modes -> isotropy subgroups (structure-search workflows)
    results = phonon.imaginary_mode_subgroups(ph)

    # symmetry-adapted eigenvector bases / modulated structures
    phonon.build_symmetry_adapted_modes(...)
    phonon.SymmetryAdaptedModulation(...)

Attributes resolve lazily (PEP 562): importing this module is instant and
pulls in phonopy/spgrep only on first use.  The implementation lives in
``crystod.phonon_irreps``, ``crystod.phonon_vector``, ``crystod.phonon_lt``,
``crystod.modulation``, ``crystod.vibration_modes``, and
``crystod.phonon_subgroups``.
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
