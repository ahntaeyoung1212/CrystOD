"""Public MD-trajectory API of CrystOD (the ``crystod-md`` domain).

    from crystod import md

    frames = md.read_xdatcar("XDATCAR")
    projector = md.build_symmetry_projector(...)   # site-symmetry ADP constraints
    constrained = md.apply_symmetry_constraints(...)

Attributes resolve lazily (PEP 562); spglib loads only on first use.
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
