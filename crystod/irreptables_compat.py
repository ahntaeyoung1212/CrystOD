"""ISO-IR-backed drop-in replacement for the former ``irreptables`` package.

Historically CrystOD read the BILBAO-derived character tables of the
``irreptables`` python package (files ``irreps-SG=*.dat``).  Those tables
follow C. J. Bradley & A. P. Cracknell (1972) label conventions, while the
ISO-IR tables bundled with CrystOD (``CIR_data.txt.gz``, parsed by
:mod:`crystod.isoir`) follow the CDML convention of A. P. Cracknell,
B. L. Davies, S. C. Miller & W. F. Love (1979) as distributed by
H. T. Stokes & B. J. Campbell (ISOTROPY Software Suite).

This module now builds the same ``(IrrepTable, Irrep)`` interface entirely
from the ISO-IR data, removing the external dependency:

* ``IrrepTable(sgnum, spinor)`` exposes ``.symmetries`` (the coset
  representatives of the space group in the conventional ISO-IR setting:
  origin choice 2, monoclinic b-unique cell choice 1, hexagonal axes) and
  ``.irreps`` (one small irrep per maximal, parameter-free k point).
* ``Irrep.characters`` is a dict ``{1-based index into table.symmetries:
  character}`` restricted to the little group of ``Irrep.k``, exactly like
  the old interface.  The characters are the raw ISO-IR small-representation
  traces (phase convention ``exp(+2*pi*i k.t)``) — verified to coincide with
  the old BILBAO table characters for all 230 space groups, so every
  existing matching path keeps working unchanged.
* ``Irrep.k`` is the representative arm of the star in conventional
  reciprocal coordinates (identical to the old tables for all 230 groups).
* Labels are ISO-IR labels; at maximal k points they coincide with the old
  BILBAO labels for every space group (verified programmatically), so only
  non-maximal k naming (handled elsewhere via :mod:`crystod.isoir`) and the
  operator listing order differ from the old package.

Double-valued (spinor) representations are NOT contained in the ISO-IR CIR
data: ``IrrepTable(sgnum, spinor=True)`` returns a table whose ``.irreps``
is empty (the ``.symmetries`` are still available for operator mapping);
callers fall back to generic ``irrep_N`` labels.
"""

from __future__ import annotations

import numpy as np

from .isoir import load_isoir_irreps

_ZERO3 = np.zeros(3)


class IsoSymop:
    """One conventional-setting symmetry operation (``SymopTable`` stand-in).

    Attributes match the old ``irreptables.SymopTable`` surface used by
    CrystOD: ``R`` (integer rotation, conventional basis), ``t`` (fractional
    translation), ``S`` (SU(2) part, identity — single-valued tables) and
    ``time_reversal`` (always False, unitary operations only).
    """

    __slots__ = ("R", "t", "S", "time_reversal")

    def __init__(self, rotation, translation):
        self.R = np.asarray(rotation, dtype=int)
        self.t = np.asarray(translation, dtype=float)
        self.S = np.eye(2)
        self.time_reversal = False


class IsoTableIrrep:
    """One small irrep at a maximal k point (``irreptables.Irrep`` stand-in).

    ``characters`` maps the 1-based index of an operation in
    ``IsoIrrepTable.symmetries`` to the small-representation trace; only
    little-group operations of ``k`` are present, mirroring the old tables.
    """

    __slots__ = ("k", "kpname", "name", "dim", "nsym", "reality", "characters")

    def __init__(self, k, kpname, name, dim, characters):
        self.k = np.asarray(k, dtype=float)
        self.kpname = kpname
        self.name = name
        self.dim = int(dim)
        self.characters = characters
        self.nsym = len(characters)
        values = np.array(list(characters.values()), dtype=complex)
        self.reality = bool(np.max(np.abs(values.imag), initial=0.0) < 1e-8)

    def show(self):
        print(self.kpname, self.name, self.dim, self.reality)


class IsoIrrepTable:
    """Character table of one space group built from ISO-IR CIR data.

    Drop-in for ``irreptables.IrrepTable``: same attribute surface
    (``number``, ``name``, ``spinor``, ``nsym``, ``symmetries``, ``irreps``)
    and the same constructor signature ``IrrepTable(SGnumber, spinor)``.
    Instances are cached per ``(space group, spinor)``; treat them as
    read-only.
    """

    def __new__(cls, SGnumber, spinor, *args, **kwargs):
        key = (int(SGnumber), bool(spinor))
        cached = _TABLE_CACHE.get(key)
        if cached is not None:
            return cached
        table = super().__new__(cls)
        table._build(int(SGnumber), bool(spinor))
        _TABLE_CACHE[key] = table
        return table

    def __init__(self, SGnumber, spinor, *args, **kwargs):
        # construction happens in _build (via __new__) so cache hits skip it
        pass

    def _build(self, sgnum: int, spinor: bool) -> None:
        self.number = sgnum
        self.number_str = str(sgnum)
        self.spinor = spinor

        iso_irreps = load_isoir_irreps(sgnum, "cir")
        reference = iso_irreps[0]
        self.name = reference.sgsymbol
        self.nsym = reference.opcount
        self.symmetries = [
            IsoSymop(reference.rotations[i], reference.translations[i])
            for i in range(reference.opcount)
        ]

        self.irreps = []
        if spinor:
            # ISO-IR CIR data contain single-valued irreps only; callers
            # detect the empty list and fall back to generic labels.
            return

        little_by_ktype: dict[str, list[int]] = {}
        for ir in iso_irreps:
            if not ir.special:
                # symmetry lines/planes and the general point stay the
                # domain of crystod.isoir (IsoIRLabeler)
                continue
            # the representative arm of a special-k irrep is arm 0 with no
            # free parameters; all irreps of one k type share that arm
            if ir.ktype not in little_by_ktype:
                little_by_ktype[ir.ktype] = [
                    i
                    for i in range(ir.opcount)
                    if ir.in_little_group(reference.rotations[i], 0, _ZERO3)
                ]
            characters = {}
            for i in little_by_ktype[ir.ktype]:
                characters[i + 1] = complex(
                    ir.small_character(
                        reference.rotations[i],
                        reference.translations[i],
                        0,
                        _ZERO3,
                    )
                )
            self.irreps.append(
                IsoTableIrrep(
                    ir.arm_k(0, _ZERO3),
                    ir.ktype,
                    ir.label,
                    ir.small_dim,
                    characters,
                )
            )

    def show(self):
        for i, s in enumerate(self.symmetries):
            print(i + 1, "\n", s.R, "\n", s.t, "\n")
        for irr in self.irreps:
            irr.show()


_TABLE_CACHE: dict[tuple[int, bool], IsoIrrepTable] = {}


def load_irreptables():
    """Return the ``(IrrepTable, Irrep)`` pair backed by the ISO-IR tables.

    Kept as the single entry point used across CrystOD so that the swap from
    the external ``irreptables`` package to the bundled ISO-IR data did not
    require touching the importing modules.
    """
    return IsoIrrepTable, IsoTableIrrep
