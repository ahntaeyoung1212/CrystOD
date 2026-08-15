"""Public group-theory API of CrystOD (the ``crystod-group`` domain).

    from crystod import group

    # space-group irrep algebra (products, characters; crystod-group --product)
    algebra = group.SpaceGroupIrrepAlgebra("Pm-3m")

    # isotropy subgroups of an irrep (crystod-group --supergroup)
    subs = group.isotropy_subgroups("Pm-3m", "R4+")
    analyzer = group.IsotropyAnalyzer("Pm-3m", "R4+")   # full machinery

    # symmetry-mode (AMPLIMODES-style) analysis (crystod-group --supergroup-cif)
    analysis = group.SymmetryModeAnalysis(...)

    # point-group tools (crystod-group --decompose/--ligand-field/--multiplet)
    group.decompose(...); group.get_orbital_characters(...); group.shell_terms(...)

Attributes resolve lazily (PEP 562): importing this module is instant and
pulls in phonopy/spgrep only on first use.
"""

from __future__ import annotations

from ._api import lazy_namespace

_EXPORTS = {
    # space-group irrep algebra (crystod-group --product / --table)
    "SpaceGroupIrrepAlgebra": ("spacegroup_product", "SpaceGroupIrrepAlgebra"),
    "format_product_report": ("spacegroup_product", "format_product_report"),
    # isotropy subgroups (crystod-group --supergroup)
    "IsotropyAnalyzer": ("isotropy_subgroup", "IsotropyAnalyzer"),
    "InducedRepresentation": ("isotropy_subgroup", "InducedRepresentation"),
    "CoupledRepresentation": ("isotropy_subgroup", "CoupledRepresentation"),
    "isotropy_subgroups": ("phonon_subgroups", "isotropy_subgroups"),
    "IsotropySubgroup": ("phonon_subgroups", "IsotropySubgroup"),
    # symmetry-mode analysis (crystod-group --supergroup-cif)
    "SymmetryModeAnalysis": ("symmetry_mode", "SymmetryModeAnalysis"),
    # point-group reduction (crystod-group --decompose)
    "get_character_table": ("decompose_irrep", "get_character_table"),
    "decompose": ("decompose_irrep", "decompose"),
    # point-group direct products (crystod-group --product)
    "direct_product_character": ("direct_product", "direct_product_character"),
    "decompose_representation": ("direct_product", "decompose_representation"),
    # ligand-field splitting (crystod-group --ligand-field)
    "get_orbital_characters": ("ligand_field", "get_orbital_characters"),
    # spin multiplets (crystod-group --multiplet)
    "shell_terms": ("multiplet", "shell_terms"),
    "couple_shells": ("multiplet", "couple_shells"),
    "parse_config": ("multiplet", "parse_config"),
    "hund_candidates": ("multiplet", "hund_candidates"),
    "compute_term_energies": ("multiplet_energy", "compute_term_energies"),
    "ground_state": ("multiplet_energy", "ground_state"),
    # polynomial basis functions (crystod-group --basis / --table)
    "format_irrep_table": ("basis_function", "format_irrep_table"),
    "format_spacegroup_table": ("basis_function", "format_spacegroup_table"),
    # POSCAR <-> Bilbao-style CIF (crystod-group --poscar2cif / --cif2poscar)
    "bilbao_cif_lines": ("poscar2cif", "bilbao_cif_lines"),
    "poscar_lines": ("poscar2cif", "poscar_lines"),
    # star of k (crystod --star-of-k; also exposed in crystod.salc)
    "compute_star": ("star_of_k", "compute_star"),
    "format_star_lines": ("star_of_k", "format_star_lines"),
    "resolve_kpoint_input": ("star_of_k", "resolve_kpoint_input"),
    # ISO-IR (Stokes-Campbell) table access
    "IsoIRLabeler": ("isoir", "IsoIRLabeler"),
    "get_isoir_label_map": ("isoir", "get_isoir_label_map"),
    "load_isoir_irreps": ("isoir", "load_isoir_irreps"),
}

__getattr__, __dir__, __all__ = lazy_namespace(globals(), _EXPORTS)
