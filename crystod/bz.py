"""Public Brillouin-zone API of CrystOD (the ``crystod-bz`` domain).

    from crystod import bz

    vertices = bz.get_brillouin_zone_3d(reciprocal_lattice)
    kpath = bz.get_seekpath_kpath(...)
    gammas = bz.get_folded_gamma_points(...)      # supercell folding
    names, coords = bz.get_special_kpoints(...)   # --show-kpoint tables

Attributes resolve lazily (PEP 562); seekpath/scipy load only on first use.
"""

from __future__ import annotations

from ._api import lazy_namespace

_EXPORTS = {
    # Brillouin-zone geometry and k paths (crystod-bz)
    "get_brillouin_zone_3d": ("brillouin_zone", "get_brillouin_zone_3d"),
    "get_seekpath_kpath": ("brillouin_zone", "get_seekpath_kpath"),
    "build_bz_traces": ("brillouin_zone", "build_bz_traces"),
    "parse_manual_band": ("brillouin_zone", "parse_manual_band"),
    "prettify_label": ("brillouin_zone", "prettify_label"),
    # supercell folding (crystod-bz --trans-mat)
    "parse_transformation_matrix": ("bz_supercell", "parse_transformation_matrix"),
    "get_folded_gamma_points": ("bz_supercell", "get_folded_gamma_points"),
    "build_supercell_bz_traces": ("bz_supercell", "build_supercell_bz_traces"),
    # special k-point tables (crystod-bz --show-kpoint)
    "get_special_kpoints": ("show_kpoints", "get_special_kpoints"),
}

__getattr__, __dir__, __all__ = lazy_namespace(globals(), _EXPORTS)
