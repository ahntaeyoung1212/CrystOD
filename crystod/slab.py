"""Public surface-slab API of CrystOD (the ``crystod-slab`` domain).

Cut surface slabs from a bulk crystal structure, the Python form of
``crystod-slab``.  For each Miller index every distinct termination is built
with pymatgen's ``SlabGenerator`` and measured: number of atoms, thickness
along the surface normal, vacuum, in-plane area, formula, whether the slab is
polar (a nonzero dipole per area from the oxidation states,
``Slab.is_polar``) and whether its two faces are equivalent
(``Slab.is_symmetric``).  The slabs are written as VASP POSCAR files.

**The bulk structure**

- :func:`load_structure` -- read a POSCAR (CIF works too), optionally in the
  standardized conventional cell; :func:`conventional_cell` -- the
  standardization on its own.
- :func:`describe_cell` -- space group and the comparison with the
  conventional standard cell, as a :class:`CellInfo`.

**Miller indices**

- :func:`parse_miller_index` -- ``"1 0 0"``, ``"1,0,0"`` or a four-index
  ``(h k i l)`` of hexagonal axes, as ``(h, k, l)``.
- :func:`read_index_list` -- an index list file, one index per line.
- :func:`distinct_miller_indices` -- the indices up to a maximum that are
  distinct under the Laue group of the crystal (in the axes of the cell).
- :func:`format_miller_index` -- ``"(1 0 -1)"``.

**Slabs**

- :func:`generate_slabs` -- every termination of the given surfaces (or of
  every distinct surface up to ``max_index``), as a :class:`SlabSet` of
  :class:`SlabRecord`.
- :func:`guess_oxidation_states`, :func:`parse_oxidation_states`,
  :func:`resolve_oxidation_states`, :func:`format_oxidation_states`,
  :func:`format_charge` -- the oxidation states of the polarity test and
  the net charge of a (symmetrized) slab.
- :func:`format_table`, :func:`slab_status` -- the table ``crystod-slab``
  prints (one string; :func:`crystod.search.format_table` returns a list of
  lines) and the status column; :func:`table_legend`, :data:`TABLE_LEGEND`,
  :data:`CHARGE_LEGEND` -- what its columns mean.

**Files**

- :func:`write_slab_poscar`, :func:`slab_poscar_text`,
  :func:`slab_filename` -- the POSCAR files (``POSCAR_slab_SrTiO3_111_t1``,
  ``..._100_s1`` for a symmetrized slab); like
  :func:`crystod.search.write_poscar`, ``write_slab_poscar`` returns
  ``(path, status)`` and keeps an identical file, replacing a different one
  only with ``overwrite=True``.

**Constants**

- :data:`SLAB_CITATION_LINES` -- the papers to cite;
  :data:`DEFAULT_SLAB_SIZE`, :data:`DEFAULT_VACUUM_SIZE`,
  :data:`DEFAULT_TOLERANCE` -- the defaults of the command.

Usage::

    from crystod import slab

    structure = slab.load_structure("221_PPOSCAR_SrTiO3")
    records = slab.generate_slabs(structure, max_index=1)
    for record in records:
        print(record.label, record.n_atoms, record.polar, record.symmetric)
    print(slab.format_table(records))
    slab.write_slab_poscar(records[0])        # ('POSCAR_slab_SrTiO3_111_t1', 'wrote')
    records = slab.generate_slabs(structure, miller_index=(1, 0, 0), symmetrize=True)

Attributes resolve lazily (PEP 562): importing this module is instant and
pymatgen loads only on first use.  Functions report bad input as
``ValueError`` through this namespace (the implementation module
:mod:`crystod.surface_slab` raises ``SystemExit``, as the command line wants).
"""

from __future__ import annotations

from ._api import lazy_namespace

_EXPORTS = {
    # the bulk structure
    "load_structure": ("surface_slab", "load_structure"),
    "conventional_cell": ("surface_slab", "conventional_cell"),
    "describe_cell": ("surface_slab", "describe_cell"),
    "CellInfo": ("surface_slab", "CellInfo"),
    # Miller indices
    "parse_miller_index": ("surface_slab", "parse_miller_index"),
    "read_index_list": ("surface_slab", "read_index_list"),
    "distinct_miller_indices": ("surface_slab", "distinct_miller_indices"),
    "format_miller_index": ("surface_slab", "format_miller_index"),
    # slabs
    "generate_slabs": ("surface_slab", "generate_slabs"),
    "SlabRecord": ("surface_slab", "SlabRecord"),
    "SlabSet": ("surface_slab", "SlabSet"),
    "guess_oxidation_states": ("surface_slab", "guess_oxidation_states"),
    "parse_oxidation_states": ("surface_slab", "parse_oxidation_states"),
    "resolve_oxidation_states": ("surface_slab", "resolve_oxidation_states"),
    "format_oxidation_states": ("surface_slab", "format_oxidation_states"),
    "format_charge": ("surface_slab", "format_charge"),
    "format_table": ("surface_slab", "format_table"),
    "slab_status": ("surface_slab", "slab_status"),
    "table_legend": ("surface_slab", "table_legend"),
    "TABLE_LEGEND": ("surface_slab", "TABLE_LEGEND"),
    "CHARGE_LEGEND": ("surface_slab", "CHARGE_LEGEND"),
    # files
    "write_slab_poscar": ("surface_slab", "write_slab_poscar"),
    "slab_poscar_text": ("surface_slab", "slab_poscar_text"),
    "slab_filename": ("surface_slab", "slab_filename"),
    # constants
    "SLAB_CITATION_LINES": ("surface_slab", "SLAB_CITATION_LINES"),
    "DEFAULT_SLAB_SIZE": ("surface_slab", "DEFAULT_SLAB_SIZE"),
    "DEFAULT_VACUUM_SIZE": ("surface_slab", "DEFAULT_VACUUM_SIZE"),
    "DEFAULT_TOLERANCE": ("surface_slab", "DEFAULT_TOLERANCE"),
}

__getattr__, __dir__, __all__ = lazy_namespace(globals(), _EXPORTS)
