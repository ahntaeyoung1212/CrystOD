"""Public Materials Project API of CrystOD (the ``crystod-search`` domain).

Search the Materials Project the way its website does and download the
structures, the Python form of ``crystod-search``.  A query is a formula
(``SrTiO3``), an anonymous formula (``ABO3``), a chemical system
(``Sr-Ti-O``), a list of elements (``Sr,Ti,O``) or material IDs
(``mp-5229``); the answer lists formula, space group, ID, band gap, energy
above the hull and number of sites, with the experimentally observed
materials flagged.  An API key of the Materials Project is needed (free,
from https://next-gen.materialsproject.org/api), read from the
``MP_API_KEY`` environment variable or from ``PMG_MAPI_KEY`` in the pymatgen
settings file, or passed as ``api_key=``.

**Searching**

- :func:`search_materials` -- a query plus filters (experimentally observed,
  on the hull, energy above hull, band gap, number of sites, space group,
  excluded elements), as a :class:`SearchResult` of :class:`Material`
  records.
- :func:`format_table` -- the table ``crystod-search`` prints.
- :func:`interpret_query`, :func:`resolve_space_group`,
  :func:`normalize_material_id` -- how queries, space groups and IDs are
  read (``mp-aaaaahtd`` is ``mp-5229``).

**Downloading structures**

- :func:`fetch_materials` -- the structures of given IDs as pymatgen
  ``Structure`` objects, in the standardized conventional cell (default)
  or the standardized primitive cell.
- :func:`standardize_cell` -- the cell conversion on its own.
- :func:`write_poscar`, :func:`poscar_text`, :func:`poscar_filename` --
  the POSCAR files ``crystod-search --get`` writes.

**Tables**

- :data:`SORT_KEYS`, :data:`CELL_CHOICES` -- the names ``--sort`` and
  ``--cell`` accept; :data:`POSCAR_PREFIXES` -- the file-name prefix of
  each cell (``POSCAR``, ``PPOSCAR``).

Usage::

    from crystod import search

    result = search.search_materials("Sr-Ti-O", experimental=True)
    for material in result.materials:
        print(material.label, material.formula, material.space_group)
    (sr2tio4,) = search.fetch_materials("mp-5532")
    search.write_poscar(sr2tio4)  # POSCAR_Sr2TiO4_I4mmm_mp-5532 (conventional)
    (sr2tio4,) = search.fetch_materials("mp-5532", cell="primitive")
    search.write_poscar(sr2tio4)  # PPOSCAR_Sr2TiO4_I4mmm_mp-5532

Attributes resolve lazily (PEP 562): importing this module is instant and
requests and pymatgen load only on first use.  Functions report bad input
as ``ValueError`` through this namespace (the implementation module raises
``SystemExit``, as the command line wants); a missing API key, a network
failure or an error of the server raise :class:`MaterialsProjectError`.
"""

from __future__ import annotations

from ._api import lazy_namespace

_EXPORTS = {
    # searching (crystod-search QUERY)
    "search_materials": ("mp_search", "search_materials"),
    "SearchResult": ("mp_search", "SearchResult"),
    "Material": ("mp_search", "Material"),
    "format_table": ("mp_search", "format_table"),
    "interpret_query": ("mp_search", "interpret_query"),
    "resolve_space_group": ("mp_search", "resolve_space_group"),
    "normalize_material_id": ("mp_search", "normalize_material_id"),
    "MaterialsProjectError": ("mp_search", "MaterialsProjectError"),
    # downloading structures (crystod-search --get)
    "fetch_materials": ("mp_search", "fetch_materials"),
    "standardize_cell": ("mp_search", "standardize_cell"),
    "write_poscar": ("mp_search", "write_poscar"),
    "poscar_text": ("mp_search", "poscar_text"),
    "poscar_filename": ("mp_search", "poscar_filename"),
    # tables
    "SORT_KEYS": ("mp_search", "SORT_KEYS"),
    "CELL_CHOICES": ("mp_search", "CELL_CHOICES"),
    "POSCAR_PREFIXES": ("mp_search", "POSCAR_PREFIXES"),
}

__getattr__, __dir__, __all__ = lazy_namespace(globals(), _EXPORTS)
