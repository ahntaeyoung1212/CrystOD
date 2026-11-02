"""The eighteen crystod-mcp tools, in the order of the tool table of the README.

Each tool is an ``async`` function whose synchronous body runs under the
timeout of :func:`crystod_mcp.utils.blocking_tool`; ``TOOLS`` is what
``crystod_mcp.server.create_server`` registers.
"""

from __future__ import annotations

from .bz import crystod_special_kpoints
from .group import (
    crystod_character_table,
    crystod_correlate,
    crystod_decompose_representation,
    crystod_find_isotropy_irreps,
    crystod_invariants,
    crystod_irrep_product,
    crystod_isotropy_subgroups,
    crystod_ligand_field,
    crystod_subgroup_graph,
    crystod_tensor_form,
)
from .mag import crystod_mag_classify
from .mol import crystod_molecular_symmetry
from .phonon import (
    crystod_imaginary_mode_subgroups,
    crystod_phonon_activity,
    crystod_phonon_irreps,
)
from .salc import crystod_crystal_orbital_irreps
from .slab import crystod_slab

TOOLS = (
    # group
    crystod_isotropy_subgroups,
    crystod_irrep_product,
    crystod_decompose_representation,
    crystod_ligand_field,
    crystod_character_table,
    crystod_invariants,
    crystod_find_isotropy_irreps,
    crystod_subgroup_graph,
    crystod_correlate,
    crystod_tensor_form,
    # phonon
    crystod_phonon_irreps,
    crystod_imaginary_mode_subgroups,
    crystod_phonon_activity,
    # magnetism
    crystod_mag_classify,
    # SALC
    crystod_crystal_orbital_irreps,
    # BZ
    crystod_special_kpoints,
    # molecules
    crystod_molecular_symmetry,
    # surfaces
    crystod_slab,
)

TOOL_NAMES = tuple(tool.__name__ for tool in TOOLS)

__all__ = ["TOOLS", "TOOL_NAMES", *TOOL_NAMES]
