"""Public powder-diffraction API of CrystOD (the ``crystod-xrd`` domain).

Powder X-ray and neutron diffraction patterns of a crystal structure, the
Python form of ``crystod-xrd``: the Bragg peaks of a pymatgen ``Structure``
for a chosen radiation (a K-alpha doublet such as ``CuKa`` with the Ka1 +
Ka2 lines in the 2:1 ratio, a single line such as ``MoKa1``, or neutrons of
a given wavelength, nuclear scattering only), their broadening into a
continuous pattern, the text-table and figure writers the command uses, and
the superlattice reflections of a distorted structure indexed on its parent
lattice and assigned to parent k points and irreps.

**Computing patterns**

- :func:`load_structure` -- a POSCAR as a pymatgen ``Structure``.
- :func:`compute_xrd_pattern` -- the Bragg peaks for one radiation, as an
  :class:`XRDPattern` of :class:`Peak` records (``h k l``, multiplicity,
  ``d``, ``2theta``, intensity, line); ``radiation="neutron"`` with a
  ``wavelength`` gives the neutron pattern.
- :func:`radiation_lines` -- the lines and wavelengths of a radiation.

**Superlattice reflections (crystod-xrd --parent)**

- :func:`superlattice_reflections` -- the reflection families of a child
  structure indexed on the parent lattice (fundamental or superlattice), the
  parent k point of each and the share of each parent irrep of that star,
  as a :class:`SuperlatticePattern` of :class:`SuperlatticeReflection`
  records with one :class:`KPointSummary` per k star (diagnostic
  reflections, first-order extinctions).
- :func:`format_superlattice_pattern`, :func:`write_superlattice_table`,
  :func:`plot_superlattice_pattern` -- the report, the comma-separated table
  and the overlaid parent/child patterns the command writes.

**Broadening and output**

- :func:`smear_pattern` -- the peaks broadened with a Lorentzian or Gaussian
  profile of unit area, as ``(two_theta, intensity)`` arrays.
- :func:`write_peak_table` -- the comma-separated table ``crystod-xrd`` writes.
- :func:`plot_xrd_pattern` -- the PDF (or PNG, SVG) with tick marks at the
  peak positions.

**Tables**

- :data:`WAVELENGTHS` -- the line wavelengths (RIETAN-FP manual).
- :data:`KALPHA_DOUBLETS` -- the doublet compositions.
- :data:`XRAY_TYPES`, :data:`PEAK_PROFILES`, :data:`RADIATIONS` -- the names
  ``--xraytype``, ``--peak-profile`` and ``--radiation`` accept.

Usage::

    from crystod import xrd
    from crystod.examples import example_path

    structure = xrd.load_structure(str(example_path("221_PPOSCAR_ScF3")))
    pattern = xrd.compute_xrd_pattern(structure, "CuKa", (10, 120))
    for peak in pattern.peaks[:3]:
        print(peak.hkl, round(peak.two_theta, 3), round(peak.intensity, 2), peak.line)
    two_theta, intensity = xrd.smear_pattern(pattern, "lorentzian", width=0.1)

    neutron = xrd.compute_xrd_pattern(structure, radiation="neutron", wavelength=1.5)
    superlattice = xrd.superlattice_reflections("221_PPOSCAR_SrTiO3.cif",
                                                "140_PPOSCAR_SrTiO3.cif")
    for reflection in superlattice.superlattice[:3]:
        print(reflection.hkl_parent_label, reflection.kname,
              reflection.contributions_label)

Attributes resolve lazily (PEP 562): importing this module is instant and
pymatgen and matplotlib load only on first use.  Functions report bad input
as ``ValueError`` through this namespace (the implementation module raises
``SystemExit``, as the command line wants).
"""

from __future__ import annotations

from ._api import lazy_namespace

_EXPORTS = {
    # patterns (crystod-xrd)
    "load_structure": ("xrd_pattern", "load_structure"),
    "compute_xrd_pattern": ("xrd_pattern", "compute_xrd_pattern"),
    "XRDPattern": ("xrd_pattern", "XRDPattern"),
    "Peak": ("xrd_pattern", "Peak"),
    "radiation_lines": ("xrd_pattern", "radiation_lines"),
    # superlattice reflections (crystod-xrd --parent)
    "superlattice_reflections": ("xrd_superlattice", "superlattice_reflections"),
    "SuperlatticePattern": ("xrd_superlattice", "SuperlatticePattern"),
    "SuperlatticeReflection": ("xrd_superlattice", "SuperlatticeReflection"),
    "KPointSummary": ("xrd_superlattice", "KPointSummary"),
    "format_superlattice_pattern": ("xrd_superlattice", "format_superlattice_pattern"),
    "write_superlattice_table": ("xrd_superlattice", "write_superlattice_table"),
    "plot_superlattice_pattern": ("xrd_superlattice", "plot_superlattice_pattern"),
    # broadening and output
    "smear_pattern": ("xrd_pattern", "smear_pattern"),
    "write_peak_table": ("xrd_pattern", "write_peak_table"),
    "plot_xrd_pattern": ("xrd_pattern", "plot_xrd_pattern"),
    # tables
    "WAVELENGTHS": ("xrd_pattern", "WAVELENGTHS"),
    "KALPHA_DOUBLETS": ("xrd_pattern", "KALPHA_DOUBLETS"),
    "XRAY_TYPES": ("xrd_pattern", "XRAY_TYPES"),
    "PEAK_PROFILES": ("xrd_pattern", "PEAK_PROFILES"),
    "RADIATIONS": ("xrd_pattern", "RADIATIONS"),
}

__getattr__, __dir__, __all__ = lazy_namespace(globals(), _EXPORTS)
