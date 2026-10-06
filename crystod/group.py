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
- ``find_isotropy_irreps`` -- the reverse lookup (``--parent G --child H``):
  every special-point irrep stratum whose isotropy subgroup has type H, as
  ``IsotropyMatch`` records.
- ``find_coupled_isotropy_irreps`` -- the coupled reverse lookup
  (``--child H --coupled``): the pairs of irreps and directions whose
  coupled isotropy subgroup has type H, as ``CoupledIsotropyMatch`` records.
- ``isotropy_table`` -- every stratum of every special-point irrep of a
  parent with its setting, as an ``IsotropyTable`` (cached on disk below
  ``$CRYSTOD_CACHE_DIR`` or ``~/.cache/crystod``).
- ``subgroup_graph`` -- the group-subgroup graph of the isotropy subgroups
  of one irrep or of a direct sum (``--parent SG --irrep IR [IR2 ...]
  --graph``), as a ``SubgroupGraph`` of ``GraphNode`` and ``GraphEdge``
  records.

Invariant polynomials (``crystod-group --parent SG --irrep IR --invariants``):

- ``invariant_polynomials`` -- the invariant polynomials (Landau
  free-energy terms) of an irrep degree by degree, as an
  ``InvariantBasis``, checked against the Molien series.
- ``landau_lifshitz`` -- the Landau (no cubic invariant) and Lifshitz
  conditions of an irrep, as a ``LandauLifshitz`` record.
- ``coupling_terms`` -- the lowest-order coupling term of a direct sum of
  irreps (e.g. the trilinear ``Q1 Q2 P`` of hybrid improper
  ferroelectrics), as a ``CouplingTerm``.
- ``secondary_order_parameters`` -- the irreps with a nonzero component
  fixed by the isotropy subgroup of a given order-parameter direction
  (``--secondary``), as ``SecondaryOrderParameter`` records.

Correlation of irreps (``crystod-group --correlate``):

- ``correlation_table`` -- the irreps of a point group restricted to every
  inequivalent orientation of a subgroup type (``--correlate --pg G
  --subgroup H``), as a ``CorrelationTable``; ``format_correlation_table``
  renders it.

- ``subduce_to_child`` -- parent irreps restricted to the Gamma point of the
  isotropy subgroup of a direction (``--correlate --parent``), as
  ``Subduction`` records; ``format_subduction`` renders them.
- ``compatibility_relations`` -- the small irreps of two special points
  restricted to the symmetry line joining them (``--correlate --sg``), as
  ``Compatibility`` records; ``format_compatibility`` renders them.

Symmetry-mode analysis (``crystod-group --supergroup-cif``):

- ``SymmetryModeAnalysis`` -- AMPLIMODES-style decomposition of the
  distortion between a parent and a child structure into parent irreps.

Property tensors (``crystod-group --tensor``):

- ``tensor_form`` -- the symmetry-allowed form of a property tensor
  (Neumann's principle) in a point group, a space group's point group or a
  structure's point group, by name (``dielectric``, ``piezoelectric``,
  ``elastic``, ...) or Jahn symbol, as a ``TensorForm`` (independent
  components, matrix form, relations); ``format_tensor_form`` renders it.
- ``raman_forms`` -- the Raman tensors of every Raman-active irrep of a
  point group (Mulliken labels).

Point-group tools (``--table``, ``--decompose``, ``--product --pg``,
``--ligand-field``):

- ``get_character_table`` -- phonopy character table of a point group;
  ``format_irrep_table`` renders it.
- ``decompose``, ``decompose_representation`` -- reduce a character vector
  into irreps; ``direct_product_character`` multiplies irreps.
- ``symmetric_square``, ``antisymmetric_square`` -- the squares
  ``[IR x IR]`` and ``{IR x IR}`` (``--product IR IR --symmetric``);
  ``jahn_teller_modes`` -- Jahn-Teller active modes and symmetry-allowed
  pseudo-Jahn-Teller coupling modes as a ``JahnTellerModes`` record
  (``--jahn-teller``).  The space-group squares are
  ``SpaceGroupIrrepAlgebra.decompose_square``.
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
    # reverse lookup (crystod-group --parent G --child H)
    "find_isotropy_irreps": ("isotropy_table", "find_isotropy_irreps"),
    "IsotropyMatch": ("isotropy_table", "IsotropyMatch"),
    "isotropy_table": ("isotropy_table", "isotropy_table"),
    "IsotropyTable": ("isotropy_table", "IsotropyTable"),
    "find_coupled_isotropy_irreps": ("isotropy_table", "find_coupled_isotropy_irreps"),
    "CoupledIsotropyMatch": ("isotropy_table", "CoupledIsotropyMatch"),
    # group-subgroup graph (crystod-group --parent SG --irrep IR ... --graph)
    "subgroup_graph": ("subgroup_graph", "subgroup_graph"),
    "SubgroupGraph": ("subgroup_graph", "SubgroupGraph"),
    "GraphNode": ("subgroup_graph", "GraphNode"),
    "GraphEdge": ("subgroup_graph", "GraphEdge"),
    # invariant polynomials (crystod-group --parent --irrep --invariants)
    "invariant_polynomials": ("invariants", "invariant_polynomials"),
    "InvariantBasis": ("invariants", "InvariantBasis"),
    "landau_lifshitz": ("invariants", "landau_lifshitz"),
    "LandauLifshitz": ("invariants", "LandauLifshitz"),
    "coupling_terms": ("invariants", "coupling_terms"),
    "CouplingTerm": ("invariants", "CouplingTerm"),
    # secondary order parameters (crystod-group --parent ... --secondary)
    "secondary_order_parameters": (
        "secondary_order_parameter", "secondary_order_parameters"),
    "SecondaryOrderParameter": ("secondary_order_parameter", "SecondaryOrderParameter"),
    # correlation of irreps (crystod-group --correlate)
    "subduce_to_child": ("correlation", "subduce_to_child"),
    "Subduction": ("correlation", "Subduction"),
    "compatibility_relations": ("correlation", "compatibility_relations"),
    "Compatibility": ("correlation", "Compatibility"),
    "format_subduction": ("correlation", "format_subduction"),
    "format_compatibility": ("correlation", "format_compatibility"),
    "correlation_table": ("correlation", "correlation_table"),
    "CorrelationTable": ("correlation", "CorrelationTable"),
    "format_correlation_table": ("correlation", "format_correlation_table"),
    # symmetry-mode analysis (crystod-group --supergroup-cif)
    "SymmetryModeAnalysis": ("symmetry_mode", "SymmetryModeAnalysis"),
    # property tensors (crystod-group --tensor)
    "tensor_form": ("tensor_form", "tensor_form"),
    "TensorForm": ("tensor_form", "TensorForm"),
    "format_tensor_form": ("tensor_form", "format_tensor_form"),
    "raman_forms": ("tensor_form", "raman_forms"),
    # point-group reduction (crystod-group --decompose)
    "get_character_table": ("decompose_irrep", "get_character_table"),
    "decompose": ("decompose_irrep", "decompose"),
    # point-group direct products (crystod-group --product)
    "direct_product_character": ("direct_product", "direct_product_character"),
    "decompose_representation": ("direct_product", "decompose_representation"),
    # symmetrized squares and Jahn-Teller modes (--product --symmetric, --jahn-teller)
    "symmetric_square": ("direct_product", "symmetric_square"),
    "antisymmetric_square": ("direct_product", "antisymmetric_square"),
    "jahn_teller_modes": ("direct_product", "jahn_teller_modes"),
    "JahnTellerModes": ("direct_product", "JahnTellerModes"),
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
