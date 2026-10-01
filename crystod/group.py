"""Group-theory API of CrystOD: the ``crystod-group`` command as Python objects.

``crystod.group`` covers the representation theory that the ``crystod-group``
command exposes: direct products and character tables of point and space
groups, the reduction of reducible representations, ligand-field splittings
and multi-electron terms, isotropy subgroups and symmetry-mode analysis, and
the ISO-IR (ISOTROPY) irrep tables behind every label CrystOD prints.  Every
name below is the implementation object of the corresponding command mode,
so the vocabulary (irrep labels, order-parameter directions, k-point names)
is the same as in the printed output.

Usage::

    from crystod import group

    subs = group.isotropy_subgroups("Pm-3m", "R4+")
    algebra = group.SpaceGroupIrrepAlgebra("Pm-3m")
    print(group.format_product_report(algebra, ["R4-", "R5+"]))

Space-group irreps (``crystod-group --product --sg``):

- ``SpaceGroupIrrepAlgebra`` -- space-group operations and ISO-IR irreps in
  the primitive basis; stars, induced characters, direct products.
- ``format_product_report`` -- the ``--product`` report as text.

Isotropy subgroups (``crystod-group --parent``):

- ``isotropy_subgroups`` -- subgroups of an irrep (or of coupled irreps) as
  ``IsotropySubgroup`` records: the data-level entry point.
- ``isotropy_subgroups_at_kpoint`` -- the same for every irrep of one
  special k point (``--parent SG --kpoint K``), as a
  ``KpointIsotropySubgroups`` mapping from irrep label to records.
- ``IsotropyAnalyzer`` -- the full machinery: directions, stabilizers,
  subgroup identification, conventional settings.
- ``InducedRepresentation``, ``CoupledRepresentation`` -- the real matrices
  of the order parameter of one irrep or of several coupled irreps.

Symmetry-mode analysis (``crystod-group --supergroup-cif``):

- ``SymmetryModeAnalysis`` -- AMPLIMODES-style decomposition of the
  distortion between a parent and a child structure into parent irreps.

Point-group tools (``--table``, ``--decompose``, ``--product --pg``,
``--ligand-field``):

- ``get_character_table`` -- phonopy character table of a point group;
  ``format_irrep_table`` renders it.
- ``decompose``, ``decompose_representation`` -- reduce a character vector
  into irreps; ``direct_product_character`` multiplies irreps.
- ``get_orbital_characters`` -- the (2l+1)-dimensional orbital representation
  (ligand-field splitting).
- ``format_spacegroup_table`` -- the character table of the little group of
  k with ISO-IR labels (``--table --sg --kpoint``).

Multiplets (``crystod-group --multiplet``):

- ``parse_config``, ``shell_terms``, ``couple_shells``, ``hund_candidates``
  -- term symbols of an electron configuration over irrep shells.
- ``compute_term_energies``, ``ground_state`` -- exact Coulomb multiplet
  energies in Racah / Slater parameters and the resulting ground term.

Structure files (``crystod-group --poscar2cif``, ``--cif2poscar``):

- ``bilbao_cif_lines``, ``poscar_lines`` -- Bilbao-style CIF and POSCAR
  writers.

Star of k (``crystod --star-of-k``; also in ``crystod.salc``):

- ``compute_star``, ``format_star_lines``, ``resolve_kpoint_input``.

ISO-IR tables:

- ``IsoIRLabeler``, ``get_isoir_label_map`` -- label spgrep small irreps with
  ISO-IR (Miller-Love) labels; ``load_isoir_irreps`` reads the tables.

Attributes resolve lazily (PEP 562): ``import crystod.group`` is instant and
pulls in phonopy, spgrep and spglib only when a name is first used.
Functions reached through this namespace report bad input as
``ValueError``; the classes are the implementation classes themselves and
raise ``SystemExit`` as the command line does.
"""

from __future__ import annotations

from ._api import lazy_namespace

_EXPORTS = {
    # space-group irrep algebra (crystod-group --product / --table)
    "SpaceGroupIrrepAlgebra": ("spacegroup_product", "SpaceGroupIrrepAlgebra"),
    "format_product_report": ("spacegroup_product", "format_product_report"),
    # isotropy subgroups (crystod-group --parent)
    "IsotropyAnalyzer": ("isotropy_subgroup", "IsotropyAnalyzer"),
    "InducedRepresentation": ("isotropy_subgroup", "InducedRepresentation"),
    "CoupledRepresentation": ("isotropy_subgroup", "CoupledRepresentation"),
    "isotropy_subgroups": ("phonon_subgroups", "isotropy_subgroups"),
    "isotropy_subgroups_at_kpoint": ("phonon_subgroups", "isotropy_subgroups_at_kpoint"),
    "IsotropySubgroup": ("phonon_subgroups", "IsotropySubgroup"),
    "KpointIsotropySubgroups": ("phonon_subgroups", "KpointIsotropySubgroups"),
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
