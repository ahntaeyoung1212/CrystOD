"""crystod package.

Command-line tools (``crystod``, ``crystod-group``, ``crystod-bz``,
``crystod-phonon``, ``crystod-mag``, ``crystod-md``, ``crystod-mol``,
``crystod-xrd``, ``crystod-search``) live in ``crystod.cli``; the Python API mirrors the same
domains:

    import crystod

    crystod.salc      # crystal-orbital SALC analysis (the main command)
    crystod.group     # irrep algebra, isotropy subgroups, symmetry modes
    crystod.phonon    # phonon irreps, eigenvectors, modulation, subgroups
    crystod.bz        # Brillouin-zone plots and special k points
    crystod.mag       # symmetry-adapted spin bases
    crystod.md        # MD-trajectory analyses (ADPs)
    crystod.mol       # molecular SALCs and MO diagrams
    crystod.xrd       # powder X-ray diffraction patterns
    crystod.search    # Materials Project search and POSCAR download

The most common entry points are also importable from the top level::

    from crystod import isotropy_subgroups, label_phonon_modes

Domain modules and the symbols below resolve lazily (PEP 562), so plain
``import crystod`` stays light; the implementation modules
(``crystod.phonon_irreps``, ``crystod.isotropy_subgroup``, ...) remain
importable directly, exactly as before.
"""

from .operations import (
    characterize_rotation,
    complex_to_real_transform,
    complex_to_real_transform_orbital,
    get_seitz_symbol,
    rotation_matrix_to_euler_zyz,
    wigner_D_matrix,
    wigner_D_real,
)

__all__ = [
    "__version__",
    "characterize_rotation",
    "complex_to_real_transform",
    "complex_to_real_transform_orbital",
    "get_seitz_symbol",
    "rotation_matrix_to_euler_zyz",
    "wigner_D_matrix",
    "wigner_D_real",
    # API domains (lazy)
    "salc",
    "group",
    "phonon",
    "bz",
    "mag",
    "md",
    "mol",
    "xrd",
    "search",
    # flagship API symbols (lazy)
    "isotropy_subgroups",
    "isotropy_subgroups_at_kpoint",
    "IsotropySubgroup",
    "label_phonon_modes",
    "imaginary_mode_subgroups",
    "scan_imaginary_modes",
    "PhononMode",
    "ImaginaryModeResult",
    "IsotropyAnalyzer",
    "SpaceGroupIrrepAlgebra",
]

__version__ = "0.4.3"

_API_DOMAINS = ("salc", "group", "phonon", "bz", "mag", "md", "mol", "xrd", "search")

_LAZY_SYMBOLS = {
    "isotropy_subgroups": ("phonon_subgroups", "isotropy_subgroups"),
    "isotropy_subgroups_at_kpoint": ("phonon_subgroups", "isotropy_subgroups_at_kpoint"),
    "IsotropySubgroup": ("phonon_subgroups", "IsotropySubgroup"),
    "label_phonon_modes": ("phonon_subgroups", "label_phonon_modes"),
    "imaginary_mode_subgroups": ("phonon_subgroups", "imaginary_mode_subgroups"),
    "scan_imaginary_modes": ("phonon_subgroups", "scan_imaginary_modes"),
    "PhononMode": ("phonon_subgroups", "PhononMode"),
    "ImaginaryModeResult": ("phonon_subgroups", "ImaginaryModeResult"),
    "IsotropyAnalyzer": ("isotropy_subgroup", "IsotropyAnalyzer"),
    "SpaceGroupIrrepAlgebra": ("spacegroup_product", "SpaceGroupIrrepAlgebra"),
}


def __getattr__(name):
    import importlib

    if name in _API_DOMAINS:
        return importlib.import_module(f".{name}", __name__)
    if name in _LAZY_SYMBOLS:
        module_name, attribute = _LAZY_SYMBOLS[name]
        module = importlib.import_module(f".{module_name}", __name__)
        value = getattr(module, attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
