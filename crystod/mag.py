"""Public magnetism API of CrystOD (the ``crystod-mag`` domain).

    from crystod import mag

    # symmetry-adapted spin bases (cluster multipoles / SAMM)
    rep = mag.get_spin_representation(...)
    ranks = mag.get_multipole_rank_lists(...)

Attributes resolve lazily (PEP 562); phonopy/spgrep load only on first use.
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
