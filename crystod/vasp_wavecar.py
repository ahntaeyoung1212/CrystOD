"""All-electron overlaps between the Bloch states of VASP runs (WAVECAR + POTCAR).

Stage 1 of the wavefunction-overlap engine of ``crystod --diagram --vasp``
(:mod:`crystod.crystal_orbital_overlap` is stage 2).  The engine compares the
Bloch states of three ordinary VASP runs that share one cell, one ``ENCUT``,
one ``PREC`` and one k-point list -- the crystal and two sublattice
("fragment") runs in which the removed sublattice is replaced by classical
point charges (POSCAR species ``Va2-``, ``Va2+``, ``Va4+``, ... of the
CrystOD-patched VASP).  Because the runs share the plane-wave basis at every
k, a sublattice state ``phi_fk`` can be expanded directly in the crystal
states ``psi_nk``; the analysis needs no energy alignment between the runs at
all, only the overlaps

    A_fn = <phi_fk | psi_nk>          (all-electron, PAW)

and the crystal eigenvalues.  This module reads the files, computes those
overlaps (and the fragment-fragment overlaps, PAW-sphere weights and the
orthonormality tests) at the requested k points, and stores them in an
:class:`OverlapCache` -- a ``.npz`` file a few MB in size from which the
analysis can be repeated in seconds without the GB-sized WAVECARs.

PAW overlaps
------------
The conventions are VASP's (``nonl.F`` SPHER/PROJ1 and ``radial.F``
RAD_CHECK_QPAW, re-derived here; no VASP source is used).  The projection of a
pseudo wavefunction on the projector ``p~_i`` of atom ``a`` at ``R_a`` is

    beta_{a,i}[psi] = <p~_i^a | psi~>
                    = Omega^-1/2 sum_G C(G) exp(i (k+G).R_a) i^l Y_lm(k+G) P_i(|k+G|)

with ``P_i`` the reciprocal-space projector table of the POTCAR (cubic
Lagrange interpolation exactly as VASP with ``NLSPLINE = .FALSE.``) and real,
normalised spherical harmonics ``Y_lm`` (any orthonormal real basis works:
the one-centre matrices below are diagonal in ``(l, m)``).  For two
wavefunctions A and B the all-electron overlap is the plane-wave overlap plus
one term per PAW sphere, which depends on whether the site carries PAW data
in the two runs:

    both runs (an atom in both):   conj(beta^A) Q^a beta^B,
                                   Q_ij = <phi_i|phi_j> - <phi~_i|phi~_j>
    run B only ("cross sphere":    sum_j <phi~^A | phi_j - phi~_j> beta^B_j
    A has a point charge at a,
    so A is smooth there)
    neither run:                   no correction.

The cross-sphere term is mandatory: without it the captured weight of a
fragment state reaches 1.03 and its spectral energy moves by up to 0.36 eV
(SrTiO3).  It is evaluated in one of three ways (``mode``):

* ``"bessel"`` (the reference) -- exactly, from the plane waves of A, with the
  radial transform ``D_j(q) = 4 pi int (u_j^AE - u_j^PS) j_l(qr) r dr`` on the
  logarithmic PAW grid;
* ``"projector"`` -- ``sum_ij conj(beta^A_i) M_ij beta^B_j`` with
  ``M_ij = <phi~_i|phi_j> - <phi~_i|phi~_j>``, i.e. A locally expanded in
  pseudo partial waves (a diagnostic of the partial-wave completeness);
* ``"none"`` -- no correction (a diagnostic of how much the term matters).

The PAW data of every site are taken from the crystal's POTCAR; the species
and positions of every run from its own POSCAR (``Va`` names included).

Sphere weights and tests
------------------------
* PAW-sphere weights per (site, l) are the PROCAR quantity
  ``sum_ij conj(beta_i) QTOT_ij beta_j`` with ``QTOT_ij = int_0^RDEP u_i^AE
  u_j^AE dr`` (``sphpro.F``); at the sites where a run has a point charge the
  smooth density inside the crystal's sphere, ``TPS_ij`` built from the pseudo
  partial waves, flags states that live on the point charges.
* T1 / T2: the all-electron overlap of the crystal (fragment) states with
  themselves must be the unit matrix; it is, to 2-7e-8 on the LAK runs, which
  validates every piece above.  The partial-wave completeness of the crystal
  states is reported per site and l.

Spinors
-------
A ``vasp_ncl`` WAVECAR (``LSORBIT``) is recognised from its plane-wave count:
every band record holds both spinor components, spin-up block first, i.e.
twice the scalar number of coefficients.  Overlaps, projections and sphere
weights are then sums over the two components (Q, M and D_j are spin
independent; the spin-orbit part of the PAW Hamiltonian never enters because
the energies come from the eigenvalues).  Collinear spin-polarized
(``ISPIN = 2``) and gamma-only (``vasp_gam``) WAVECARs are refused with an
ERROR.

The cache
---------
:class:`OverlapCache` holds, per requested k point, everything stage 2 needs:
the eigenvalues and occupations of every run, ``A`` for every fragment run and
cross-sphere mode, the fragment-fragment overlaps, sphere weights and
point-charge-sphere weights, the test results, and per-band irrep labels when
a label provider was given; plus run metadata (POSCAR species, positions and
lattice, NELECT, NBANDS, ENCUT, spinor flag, the whole-k-list VBM/HOMO) and
per-species PAW metadata (TITEL, VRHFIN, ZVAL, derived Q/M matrices and the
valence shells -- derived numbers only, never PAW data themselves).  It is a
compressed ``.npz`` with the metadata as one JSON string (schema
``crystod-overlap-cache/1``) and loads without pickle.  A cache can be
trimmed to the rows of selected fragment states (the regression fixtures of
``example/03_hybridization/vasp_overlap`` are such caches).

``read_bands`` exposes the raw plane-wave data of one k point (G vectors in
WAVECAR order, coefficients, eigenvalues, occupations) for the plane-wave
irrep module.
"""

from __future__ import annotations

import datetime
import itertools
import json
import math
import os
import re
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.special import spherical_jn

from .vasp_io import VaspStructure, open_vasp_file, parse_va_species, read_poscar, \
    resolve_vasp_file

# ---------------------------------------------------------------- constants
# VASP's own values (constant.F): energies and lengths below follow VASP
# exactly so that the plane-wave sphere |k+G|^2 hbar^2/2m < ENCUT selects the
# same G vectors as the run did.
AUTOA = 0.529177249
RYTOEV = 13.605826
HSQDTM = RYTOEV * AUTOA * AUTOA      # hbar^2 / 2 m_e in eV A^2
# Table sizes of a POTCAR dataset (pseudo_struct.F).
NPSNL = 100
NPSRNL = 100
NPSPTS = 1000

#: The three evaluations of the cross-sphere term (see the module docstring).
CROSS_SPHERE_MODES = ("bessel", "projector", "none")

#: Schema string of :class:`OverlapCache` files.
CACHE_SCHEMA = "crystod-overlap-cache/1"

_SHELL_LETTERS = "spdf"


# =============================================================== POSCAR
def _clean_species(name: str) -> str:
    """Element symbol of a POSCAR species token (``Sr_sv`` -> ``Sr``).

    ``Va`` point-charge names are kept as they are.
    """
    if parse_va_species(name) is not None:
        return name
    return name.split("/")[0].split("_")[0]


def read_run_structure(path: str) -> VaspStructure:
    """Read the POSCAR of a run, ``Va`` point-charge species included.

    A thin layer over :func:`crystod.vasp_io.read_poscar`, which reads the
    cell as VASP does (:func:`crystod.vasp_io.poscar_geometry`): species
    tokens such as ``Sr_sv`` are reduced to the element symbol (``Va`` names
    are kept).

    Args:
        path: Path of the POSCAR.

    Returns:
        A :class:`crystod.vasp_io.VaspStructure` with cleaned species names.

    Raises:
        SystemExit: The file is missing or malformed.
    """
    raw = read_poscar(path)
    species = [_clean_species(name) for name in raw.species]
    symbols = [_clean_species(name) for name in raw.symbols]
    return VaspStructure(comment=raw.comment, lattice=raw.lattice, species=species,
                         counts=list(raw.counts), symbols=symbols,
                         positions=raw.positions, charges=list(raw.charges),
                         path=raw.path)


# =============================================================== POTCAR
class _Cursor:
    """Line cursor over one POTCAR dataset (Fortran list-directed reads)."""

    def __init__(self, lines):
        self.lines = lines
        self.i = 0

    def line(self) -> str:
        text = self.lines[self.i]
        self.i += 1
        return text

    def floats(self, n: int) -> np.ndarray:
        out: list[float] = []
        while len(out) < n:
            out.extend(float(t.replace("D", "E").replace("d", "e"))
                       for t in self.line().split())
        # a Fortran list-directed read discards the rest of the last line
        return np.array(out[:n])


def simpson_weights(r: np.ndarray, nmax: int | None = None) -> np.ndarray:
    """VASP's Simpson weights on the logarithmic radial grid (``SET_SIMP``).

    Args:
        r: The radial grid ``r_k = r_1 exp((k - 1) h)``.
        nmax: Number of points to integrate over (default: all).

    Returns:
        Weights ``w_k`` with ``int f(r) dr = sum_k w_k f(r_k)``.
    """
    n = len(r) if nmax is None else nmax
    h = math.log(r[-1] / r[0]) / (len(r) - 1)
    si = np.zeros(len(r))
    # Fortran: DO K = NMAX, 3, -2 (1-based)
    for k in range(n, 2, -2):
        k0 = k - 1
        si[k0] += r[k0] * h / 3.0
        si[k0 - 1] = 4.0 * r[k0 - 1] * h / 3.0
        si[k0 - 2] = r[k0 - 2] * h / 3.0
    return si


class PawDataset:
    """The parts of one POTCAR dataset that the PAW overlaps need.

    Attributes:
        name: First line of the dataset (``"PAW_PBE Sr_sv 07Sep2000"``).
        title: The ``TITEL`` string.
        element: Chemical symbol from ``VRHFIN``.
        vrhfin: The ``VRHFIN`` configuration text (``"4s4p5s"``, ``"d3 s1"``,
            or empty, as for ``Pb``).
        zval: Number of valence electrons.
        rdep: Radius of the radial grids in Angstrom (PROCAR spheres).
        lps: ``l`` of every projector channel.
        configuration: ``[(n, l, j, E, occ), ...]`` of the ``Atomic
            configuration`` table of the PSCTR header.
        Q, M, Dd, QTOT, TPS: one-centre matrices per channel pair (see
            :meth:`finalize`); ``*full`` versions per projector component.
    """

    def __init__(self):
        self.name = ""
        self.title = ""
        self.element = ""
        self.vrhfin = ""
        self.zval = 0.0
        self.rdep = 0.0
        self.psmaxn = 0.0
        self.lps: list = []
        self.pspnl: list = []
        self.dion: list = []
        self.configuration: list = []
        self.r = None
        self.wae = None            # (nmax, nch)  r * phi_AE
        self.wps = None            # (nmax, nch)  r * phi_PS
        self.qpaw_file = None
        self.psdmax = 0.0

    def finalize(self) -> PawDataset:
        """Derive the one-centre matrices from the partial waves.

        * ``Q_ij = int (u_i^AE u_j^AE - u_i^PS u_j^PS) dr`` (RAD_CHECK_QPAW,
          L = 0), compared with the augmentation charges of the file;
        * ``M_ij = <phi~_i | phi_j - phi~_j>`` (projector cross-sphere mode);
        * ``Dd_ij = <phi_i - phi~_i | phi_j - phi~_j>`` (exact self overlap);
        * ``QTOT_ij = int_0^RDEP u_i^AE u_j^AE dr`` (PROCAR sphere weights);
        * ``TPS_ij`` the same with the pseudo partial waves (smooth density
          inside a crystal sphere at a site where a run has a point charge).

        Returns:
            The dataset itself.
        """
        self.lps = np.array(self.lps, int)
        nch = len(self.lps)
        same_l = self.lps[:, None] == self.lps[None, :]
        si = simpson_weights(self.r)
        self.si = si
        wae, wps = self.wae, self.wps
        dw = wae - wps
        self.Q = np.where(same_l, np.einsum("k,ki,kj->ij", si, wae, wae)
                          - np.einsum("k,ki,kj->ij", si, wps, wps), 0.0)
        self.M = np.where(same_l, np.einsum("k,ki,kj->ij", si, wps, dw), 0.0)
        self.Dd = np.where(same_l, np.einsum("k,ki,kj->ij", si, dw, dw), 0.0)
        irmax = len(self.r)
        if self.rdep > 0:
            for k in range(len(self.r) - 1):
                if self.r[k] - self.rdep > -5e-3:
                    irmax = k + 1
                    break
        # simpson_weights() takes h from the truncated grid -- identical on a
        # logarithmic grid
        si_t = simpson_weights(self.r[:irmax])
        si_t = np.concatenate([si_t, np.zeros(len(self.r) - irmax)])
        self.QTOT = np.where(same_l, np.einsum("k,ki,kj->ij", si_t, wae, wae), 0.0)
        self.TPS = np.where(same_l, np.einsum("k,ki,kj->ij", si_t, wps, wps), 0.0)
        if self.qpaw_file is not None:
            qf = np.where(same_l, self.qpaw_file, 0.0)
            self.qpaw_check = float(np.max(np.abs(qf - self.Q))) if nch else 0.0
        else:
            self.qpaw_check = None
        # (channel, l, m) of every projector component
        idx = []
        for ch, l in enumerate(self.lps):
            for m in range(2 * l + 1):
                idx.append((ch, int(l), m))
        self.lm_index = idx
        self.nproj = len(idx)
        ch_of = np.array([c for c, _, _ in idx], int)
        l_of = np.array([l for _, l, _ in idx], int)
        m_of = np.array([m for _, _, m in idx], int)
        same = (l_of[:, None] == l_of[None, :]) & (m_of[:, None] == m_of[None, :])
        self.ch_of, self.l_of, self.m_of = ch_of, l_of, m_of
        self.Qfull = np.where(same, self.Q[ch_of][:, ch_of], 0.0)
        self.Mfull = np.where(same, self.M[ch_of][:, ch_of], 0.0)
        self.Ddfull = np.where(same, self.Dd[ch_of][:, ch_of], 0.0)
        self.QTOTfull = np.where(same, self.QTOT[ch_of][:, ch_of], 0.0)
        self.TPSfull = np.where(same, self.TPS[ch_of][:, ch_of], 0.0)
        return self

    # -- reciprocal-space form factors ---------------------------------------
    def projector_ff(self, ch: int, q: np.ndarray) -> np.ndarray:
        """Reciprocal-space projector ``P_ch(q)`` interpolated as VASP does.

        Cubic Lagrange interpolation of the POTCAR table (``nonl.F`` SPHER
        with ``NLSPLINE = .FALSE.``); zero beyond the table.

        Args:
            ch: Projector channel.
            q: ``|k + G|`` in 1/Angstrom.

        Returns:
            ``P_ch(q)``, same shape as ``q``.
        """
        tab = self.pspnl[ch]
        argsc = NPSNL / self.psmaxn
        arg = q * argsc + 1.0
        naddr = np.floor(arg).astype(int)
        out = np.zeros_like(q)
        ok = naddr < NPSNL - 2
        na = naddr[ok]
        rem = arg[ok] - na
        v1, v2, v3, v4 = tab[na - 1], tab[na], tab[na + 1], tab[na + 2]
        t0 = v2
        t1 = ((6 * v3) - (2 * v1) - (3 * v2) - v4) / 6.0
        t2 = (v1 + v3 - (2 * v2)) / 2.0
        t3 = (v4 - v1 + (3 * (v2 - v3))) / 6.0
        out[ok] = t0 + rem * (t1 + rem * (t2 + rem * t3))
        return out

    def radial_ff(self, ch: int, q: np.ndarray, which: str = "dphi") -> np.ndarray:
        """Bessel transform ``4 pi int u(r) j_l(qr) r dr`` on the PAW grid.

        Args:
            ch: Partial-wave channel.
            q: ``|k + G|`` in 1/Angstrom.
            which: ``"dphi"`` (``u_AE - u_PS``), ``"ps"`` or ``"ae"``.

        Returns:
            The transform, same shape as ``q``.

        Raises:
            ValueError: Unknown ``which``.
        """
        l = int(self.lps[ch])
        if which == "dphi":
            u = self.wae[:, ch] - self.wps[:, ch]
        elif which == "ps":
            u = self.wps[:, ch]
        elif which == "ae":
            u = self.wae[:, ch]
        else:
            raise ValueError(which)
        qr = np.outer(q, self.r)
        jl = spherical_jn(l, qr)
        return 4.0 * np.pi * (jl @ (self.si * u * self.r))

    def duality_check(self, nq: int = 4000) -> np.ndarray:
        """``(1/8 pi^3) int q^2 P_i(q) U~_j(q) dq`` -- delta_ij for same l.

        A self-test of the reciprocal projector tables against the pseudo
        partial waves (``<p~_i|phi~_j> = delta_ij``).

        Args:
            nq: Number of quadrature points.

        Returns:
            The ``(nch, nch)`` matrix.
        """
        q = np.linspace(0.0, self.psmaxn * (NPSNL - 3) / NPSNL, nq)
        nch = len(self.lps)
        res = np.zeros((nch, nch))
        for i in range(nch):
            pi_ = self.projector_ff(i, q)
            for j in range(nch):
                if self.lps[i] != self.lps[j]:
                    continue
                uj = self.radial_ff(j, q, "ps")
                res[i, j] = np.trapezoid(q * q * pi_ * uj, q) / (8 * np.pi ** 3)
        return res


def _parse_psctr(lines, ds: PawDataset) -> None:
    """Header facts of one dataset (TITEL, RDEP, VRHFIN, ZVAL, configuration)."""
    for text in lines:
        m = re.search(r"TITEL\s*=\s*(.*)", text)
        if m:
            ds.title = m.group(1).strip()
        m = re.search(r"\bRDEP\s*=\s*([-+0-9.EeDd]+)", text)
        if m:
            ds.rdep = float(m.group(1).replace("D", "E").replace("d", "e")) * AUTOA
        m = re.search(r"VRHFIN\s*=\s*([A-Za-z]+)\s*:?(.*)", text)
        if m:
            ds.element = m.group(1)
            ds.vrhfin = m.group(2).strip()
        m = re.search(r"ZVAL\s*=\s*([-+0-9.Ee]+)", text)
        if m:
            ds.zval = float(m.group(1))
    for pos, text in enumerate(lines):
        if "Atomic configuration" not in text:
            continue
        try:
            count = int(lines[pos + 1].split()[0])
        except (IndexError, ValueError):
            break
        rows = []
        for row in lines[pos + 3:pos + 3 + count]:
            parts = row.split()
            if len(parts) < 5:
                break
            try:
                rows.append((int(parts[0]), int(parts[1]), float(parts[2]),
                             float(parts[3]), float(parts[4])))
            except ValueError:
                break
        ds.configuration = rows
        break


def _parse_dataset(lines) -> PawDataset:
    """One POTCAR dataset -> :class:`PawDataset` (PAW datasets only)."""
    cur = _Cursor(lines)
    ds = PawDataset()
    ds.name = cur.line().strip()
    ds.zval = float(cur.line().split()[0])
    sel = cur.line()
    if "PSCTR" not in sel:
        raise SystemExit("ERROR: POTCAR without PSCTR header is not supported.")
    psctr = []
    while True:
        text = cur.line()
        if "END of PSCTR" in text:
            break
        psctr.append(text)
    _parse_psctr(psctr, ds)
    if not ds.element:
        ds.element = ds.name.split()[1].split("_")[0]
    cur.line()                                  # local part
    cur.line()                                  # PSGMAX
    cur.floats(NPSPTS)
    sel = cur.line().strip()
    if sel[:1] == "g":
        cur.line()
        sel = cur.line().strip()
    for tag in ("c", "k", "K"):
        if sel[:1] == tag:
            cur.floats(NPSPTS)
            sel = cur.line().strip()
    cur.floats(NPSPTS)                          # atomic pseudo charge density
    ds.psmaxn = float(cur.line().split()[0])
    sel = ""
    if ds.psmaxn > 0:
        while True:
            sel = cur.line().strip()
            if sel[:1] in ("D", "A", "P", "E"):
                break
            toks = cur.line().split()
            l, nlpro = int(toks[0]), int(toks[1])
            ds.dion.append(cur.floats(nlpro * nlpro).reshape(nlpro, nlpro))
            for _ in range(nlpro):
                cur.line()                      # Reciprocal Space Part
                tab = cur.floats(NPSNL)
                ext = np.zeros(NPSNL + 1)
                ext[1:] = tab
                ext[0] = tab[1] if l % 2 == 0 else -tab[1]
                ds.pspnl.append(ext)
                ds.lps.append(l)
                cur.line()                      # Real Space Part
                cur.floats(NPSRNL)
    if sel[:1] != "P":
        raise SystemExit(f"ERROR: {ds.title}: not a PAW dataset; the overlap engine "
                         "needs PAW POTCARs.")
    nch = len(ds.lps)
    toks = cur.line().split()
    nmax, ds.psdmax = int(toks[0]), float(toks[1])
    cur.line()                                  # format
    cur.line()                                  # augmentation charges
    ds.qpaw_file = cur.floats(nch * nch).reshape(nch, nch).T
    sel = cur.line().strip()
    if sel[:1] == "t":
        cur.floats(nch * nch)
        cur.line()
    cur.floats(nch * nch)                       # QATO
    blocks = []
    while cur.i < len(cur.lines):
        hdr = cur.line().strip()
        if not hdr:
            continue
        if hdr.startswith("End of Dataset"):
            break
        blocks.append((hdr, cur.floats(nmax)))
    pse, ae = [], []
    for hdr, arr in blocks:
        h = hdr.lower()
        if h.startswith("grid"):
            ds.r = arr
        elif h.startswith("pseudo wavefunction"):
            pse.append(arr)
        elif h.startswith("ae wavefunction"):
            ae.append(arr)
    if ds.r is None or len(pse) != nch or len(ae) != nch:
        raise SystemExit(f"ERROR: {ds.title}: PAW radial sets incomplete "
                         f"({len(pse)} ps / {len(ae)} ae waves for {nch} channels).")
    ds.wps = np.array(pse).T
    ds.wae = np.array(ae).T
    return ds.finalize()


def read_potcar(path: str) -> list:
    """Read the PAW datasets of a (concatenated) POTCAR.

    Args:
        path: Path of the POTCAR.

    Returns:
        One :class:`PawDataset` per dataset, in file order.

    Raises:
        SystemExit: The file is missing or not a PAW POTCAR.
    """
    resolved = resolve_vasp_file(path)
    if not resolved:
        raise SystemExit(f"ERROR: no POTCAR at {path} (the overlap engine needs the PAW "
                         "datasets of the crystal run).")
    with open_vasp_file(resolved) as fh:
        text = fh.read()
    parts = re.split(r"^\s*End of Dataset\s*$", text, flags=re.M)
    out = []
    for part in parts:
        lines = part.split("\n")
        while lines and not lines[0].strip():
            lines.pop(0)
        if len(lines) < 10:
            continue
        out.append(_parse_dataset(lines))
    return out


# =============================================================== periodic table
_PT = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu "
       "Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba "
       "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi "
       "Po At Rn").split()


def period_group(element: str) -> tuple:
    """``(period, IUPAC group, Z)`` of an element; lanthanides count as group 3.

    Args:
        element: Chemical symbol (H to Rn).

    Returns:
        The tuple.

    Raises:
        ValueError: Unknown symbol.
    """
    z = _PT.index(element) + 1
    starts = [0, 2, 10, 18, 36, 54, 86]
    p = next(i for i in range(1, 7) if z <= starts[i]) if z <= 86 else 6
    pos = z - starts[p - 1]
    if p == 1:
        g = 1 if pos == 1 else 18
    elif p in (2, 3):
        g = pos if pos <= 2 else pos + 10
    elif p in (4, 5):
        g = pos
    else:
        g = pos if pos <= 2 else (3 if pos <= 17 else pos - 14)
    return p, g, z


def lowest_n(element: str, l: int, vrhfin: str = "", zval: float = 0.0) -> int:
    """Lowest principal quantum number of the ``(element, l)`` valence channel.

    Explicit ``nl`` tokens of ``VRHFIN`` (``"4s4p5s"``) win; otherwise the
    period rule: s, p -> period; d -> period - 1 when that d shell is empty or
    in the valence, else period; f -> 4 up to Lu, then 5.

    Args:
        element: Chemical symbol.
        l: Angular momentum.
        vrhfin: The dataset's ``VRHFIN`` configuration text.
        zval: The dataset's number of valence electrons.

    Returns:
        ``n``.
    """
    cfg = vrhfin or ""
    if cfg[:1].isdigit():
        found = [int(n) for n, ll in re.findall(r"(\d)([spdf])", cfg)
                 if _SHELL_LETTERS.index(ll) == l]
        if found:
            return min(found)
    p, g, z = period_group(element)
    if l <= 1:
        return max(p, l + 1)
    if l == 2:
        if g <= 12 or zval > (g - 10) + 0.5:
            return max(p - 1, 3)
        return max(p, 3)
    return 4 if z <= 71 else 5


def valence_shells(element: str, vrhfin: str = "", zval: float = 0.0,
                   configuration=None) -> list:
    """Valence shells ``[(n, l), ...]`` of a PAW dataset, in ``VRHFIN`` order.

    ``VRHFIN`` names them either with principal quantum numbers
    (``"4s4p5s"``, ``"3d3 4s1"``) or by letter and occupation (``"s2p4"``,
    ``"d3 s1"``; ``n`` from :func:`lowest_n`).  Some datasets leave it empty
    (``Pb``); then the ``Atomic configuration`` table decides: its occupied
    entries, highest in energy first, until their occupations account for
    ``ZVAL`` (``Pb``: 6p2 6s2), listed in ``(n, l)`` order.

    Args:
        element: Chemical symbol.
        vrhfin: ``VRHFIN`` configuration text.
        zval: Number of valence electrons.
        configuration: ``[(n, l, j, E, occ), ...]`` from the POTCAR header.

    Returns:
        The shells, without repetitions.
    """
    cfg = (vrhfin or "").strip()
    out: list = []
    if cfg[:1].isdigit():
        for n, letter in re.findall(r"(\d)([spdf])", cfg):
            shell = (int(n), _SHELL_LETTERS.index(letter))
            if shell not in out:
                out.append(shell)
    elif cfg:
        for letter in re.findall(r"([spdf])", cfg):
            l = _SHELL_LETTERS.index(letter)
            shell = (lowest_n(element, l, cfg, zval), l)
            if shell not in out:
                out.append(shell)
    if out:
        return out
    if configuration and zval > 0:
        total, picked = 0.0, []
        for n, l, _j, _e, occ in sorted(configuration, key=lambda row: -row[3]):
            if total >= zval - 1e-6:
                break
            if occ <= 0:
                continue
            picked.append((int(n), int(l)))
            total += occ
        if abs(total - zval) <= 1e-3:
            for shell in sorted(set(picked)):
                out.append(shell)
            return out
    # last resort: the textbook valence of the group
    p, g, _ = period_group(element)
    out = [(lowest_n(element, 0, "", zval), 0)]
    if 3 <= g <= 12:
        out.append((lowest_n(element, 2, "", zval), 2))
    elif g >= 13:
        out.append((lowest_n(element, 1, "", zval), 1))
    return out


# =============================================================== WAVECAR
class Wavecar:
    """Random-access reader of a ``vasp_std`` or ``vasp_ncl`` WAVECAR.

    For spinor (``vasp_ncl``) files every band record holds both components,
    spin-up block first, i.e. ``2 n_G`` coefficients; :attr:`spinor` tells
    which case it is.  Use as a context manager or call :meth:`close`.

    Raises:
        SystemExit: Missing file, unknown precision tag, a spin-polarized
            (``ISPIN = 2``) or a gamma-only (``vasp_gam``) WAVECAR.
    """

    def __init__(self, path: str):
        if not os.path.isfile(path):
            raise SystemExit(f"ERROR: no WAVECAR at {path} (the overlap engine reads "
                             "the wavefunctions: rerun with LWAVE = .TRUE.).")
        self.path = path
        self.fh = open(path, "rb")
        rec0 = np.fromfile(self.fh, np.float64, 3)
        self.recl = int(rec0[0])
        self.nspin = int(rec0[1])
        self.rtag = int(rec0[2])
        if self.rtag in (45200, 53300):
            self.cdtype = np.complex64
        elif self.rtag in (45210, 53310):
            self.cdtype = np.complex128
        else:
            self.close()
            raise SystemExit(f"ERROR: {path}: unknown WAVECAR precision tag {self.rtag}.")
        if self.nspin != 1:
            self.close()
            raise SystemExit(
                f"ERROR: {path} is spin-polarized (ISPIN = {self.nspin}); the overlap "
                "engine supports ISPIN = 1 (vasp_std) and non-collinear (vasp_ncl) runs.")
        self.fh.seek(self.recl)
        h = np.fromfile(self.fh, np.float64, 13)
        self.nk, self.nb = int(h[0]), int(h[1])
        self.encut = float(h[2])
        self.lattice = h[3:12].reshape(3, 3)
        self.efermi = float(h[12])
        recl8 = self.recl // 8
        self.nhdr = (4 + 3 * self.nb + recl8 - 1) // recl8
        self.rec_per_k = self.nhdr + self.nb
        self.recip = np.linalg.inv(self.lattice).T          # rows b_i (no 2 pi)
        self._hdr_cache: dict = {}
        self._g_cache: dict = {}
        self._spinor = None

    def close(self) -> None:
        """Close the file handle."""
        if self.fh is not None and not self.fh.closed:
            self.fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def spinor(self) -> bool:
        """True for a ``vasp_ncl`` WAVECAR (decided at the first k point)."""
        if self._spinor is None:
            self.gvectors(0)
        return self._spinor

    def _krec(self, ik: int, isp: int = 0) -> int:
        return 2 + (isp * self.nk + ik) * self.rec_per_k

    def header(self, ik: int, isp: int = 0) -> tuple:
        """``(n_coefficients, k_frac, eigenvalues, occupations)`` of k point ``ik``.

        Args:
            ik: 0-based k index.
            isp: Spin index (always 0 here).

        Returns:
            The tuple; ``n_coefficients`` counts both spinor components.
        """
        key = (ik, isp)
        if key not in self._hdr_cache:
            self.fh.seek(self._krec(ik, isp) * self.recl)
            d = np.fromfile(self.fh, np.float64, 4 + 3 * self.nb)
            enocc = d[4:].reshape(self.nb, 3)
            self._hdr_cache[key] = (int(d[0]), d[1:4].copy(), enocc[:, 0].copy(),
                                    enocc[:, 2].copy())
        return self._hdr_cache[key]

    def kpoints(self) -> np.ndarray:
        """Fractional k points of the file, shape ``(nk, 3)``."""
        return np.array([self.header(ik)[1] for ik in range(self.nk)])

    def coeffs(self, ik: int, isp: int = 0) -> np.ndarray:
        """Plane-wave coefficients ``(nbands, n_coefficients)`` (complex128)."""
        npl = self.header(ik, isp)[0]
        out = np.empty((self.nb, npl), np.complex128)
        base = self._krec(ik, isp) + self.nhdr
        for ib in range(self.nb):
            self.fh.seek((base + ib) * self.recl)
            out[ib] = np.fromfile(self.fh, self.cdtype, npl)
        return out

    def gvectors(self, ik: int) -> np.ndarray:
        """Integer G vectors in the WAVECAR order (z outer, y, x inner).

        The sphere ``hbar^2 |k+G|^2 / 2m < ENCUT`` is regenerated in VASP's
        FFT order; its size decides between a scalar and a spinor file.

        Args:
            ik: 0-based k index.

        Returns:
            ``(n_G, 3)`` integer array.

        Raises:
            SystemExit: The count matches neither (gamma-only WAVECAR, or a
                file whose cell or ENCUT differ from its header).
        """
        if ik in self._g_cache:
            return self._g_cache[ik]
        kvec = self.header(ik)[1]
        npl = self.header(ik)[0]
        gmax = math.sqrt(self.encut / HSQDTM) / (2 * np.pi)       # |k+G| / 2 pi [1/A]
        nmax = [int(math.ceil(gmax * np.linalg.norm(self.lattice[i]))) + 2 for i in range(3)]

        def order(n):
            return np.array(list(range(0, n + 1)) + list(range(-n, 0)))

        g3, g2, g1 = np.meshgrid(order(nmax[2]), order(nmax[1]), order(nmax[0]),
                                 indexing="ij")
        g = np.stack([g1.ravel(), g2.ravel(), g3.ravel()], axis=1)
        kc = (g + kvec) @ self.recip * 2 * np.pi
        e = HSQDTM * np.einsum("ij,ij->i", kc, kc)
        sel = g[e < self.encut]
        if len(sel) == npl:
            spin = False
        elif 2 * len(sel) == npl:
            spin = True
        elif 2 * npl - 1 == len(sel):
            raise SystemExit(
                f"ERROR: {self.path} is a gamma-only WAVECAR (vasp_gam: {npl} of "
                f"{len(sel)} plane waves at k point {ik + 1}); the overlap engine needs "
                "vasp_std or vasp_ncl runs.")
        else:
            raise SystemExit(
                f"ERROR: {self.path}: generated {len(sel)} plane waves at k point "
                f"{ik + 1}, the WAVECAR has {npl} (gamma-only WAVECAR?).")
        if self._spinor is None:
            self._spinor = spin
        elif self._spinor != spin:
            raise SystemExit(f"ERROR: {self.path}: mixed spinor / scalar records.")
        self._g_cache[ik] = sel
        return sel


def _wavecar_path(path: str) -> str:
    return os.path.join(path, "WAVECAR") if os.path.isdir(path) else path


def read_bands(path: str, k_index: int) -> tuple:
    """Raw plane-wave data of one k point of a WAVECAR.

    Args:
        path: The WAVECAR file, or a run directory that holds one.
        k_index: 0-based index into the WAVECAR's k-point list.

    Returns:
        ``(gvecs, coeffs, eigenvalues, occupations, k_frac)``: ``gvecs`` the
        integer G vectors ``(npw, 3)`` in WAVECAR order, ``coeffs`` complex
        ``(nbands, nspinor, npw)`` (``nspinor`` = 2 for ``vasp_ncl``, spin-up
        component first), ``eigenvalues`` in eV and ``occupations`` (0..1 per
        band) of length ``nbands``, ``k_frac`` the fractional k point.

    Raises:
        SystemExit: Missing, spin-polarized or gamma-only WAVECAR, or a k
            index outside the file.
    """
    with Wavecar(_wavecar_path(path)) as wav:
        if not 0 <= k_index < wav.nk:
            raise SystemExit(f"ERROR: {path}: k index {k_index} outside 0..{wav.nk - 1}.")
        gvecs = wav.gvectors(k_index)
        _, kvec, eps, occ = wav.header(k_index)
        coeffs = wav.coeffs(k_index)
        nspinor = 2 if wav.spinor else 1
        coeffs = coeffs.reshape(wav.nb, nspinor, len(gvecs))
        return gvecs.copy(), coeffs, eps.copy(), occ.copy(), kvec.copy()


# =============================================================== Y_lm
def real_ylm(l: int, u: np.ndarray) -> np.ndarray:
    """Real, normalised spherical harmonics ``(2l + 1, N)`` of unit vectors.

    The order is VASP's (``m = -l..l``: ``py pz px`` for p).  Any orthonormal
    real basis gives the same overlaps, since the one-centre matrices are
    diagonal in ``(l, m)``.

    Args:
        l: 0 to 3.
        u: Unit vectors ``(N, 3)``.

    Returns:
        The harmonics.

    Raises:
        ValueError: ``l > 3``.
    """
    x, y, z = u[:, 0], u[:, 1], u[:, 2]
    pi = np.pi
    if l == 0:
        return np.full((1, len(x)), 0.5 * math.sqrt(1 / pi))
    if l == 1:
        c = math.sqrt(3 / (4 * pi))
        return np.array([c * y, c * z, c * x])
    if l == 2:
        c1 = 0.5 * math.sqrt(15 / pi)
        return np.array([c1 * x * y, c1 * y * z, 0.25 * math.sqrt(5 / pi) * (3 * z * z - 1),
                         c1 * x * z, 0.25 * math.sqrt(15 / pi) * (x * x - y * y)])
    if l == 3:
        return np.array([
            0.25 * math.sqrt(35 / (2 * pi)) * y * (3 * x * x - y * y),
            0.5 * math.sqrt(105 / pi) * x * y * z,
            0.25 * math.sqrt(21 / (2 * pi)) * y * (5 * z * z - 1),
            0.25 * math.sqrt(7 / pi) * z * (5 * z * z - 3),
            0.25 * math.sqrt(21 / (2 * pi)) * x * (5 * z * z - 1),
            0.25 * math.sqrt(105 / pi) * z * (x * x - y * y),
            0.25 * math.sqrt(35 / (2 * pi)) * x * (x * x - 3 * y * y)])
    raise ValueError("real_ylm: l > 3 is not implemented")


# =============================================================== runs
class VaspRun:
    """One VASP run (crystal or fragment) at the level the overlaps need.

    Attributes:
        path: Absolute run directory.
        role: ``"crystal"`` or ``"fragment"``.
        structure: The POSCAR (:func:`read_run_structure`).
        wavecar: The open :class:`Wavecar`.
        potcar_titles: ``TITEL`` strings of the run's own POTCAR (may be empty).
        own: Per site, True for a real atom, False for a ``Va`` point charge.
    """

    def __init__(self, path: str, role: str):
        self.path = os.path.abspath(path)
        self.role = role
        self.structure = read_run_structure(os.path.join(path, "POSCAR"))
        self.wavecar = Wavecar(os.path.join(path, "WAVECAR"))
        self.potcar_titles: list = []
        potcar = resolve_vasp_file(os.path.join(path, "POTCAR"))
        if potcar:
            with open_vasp_file(potcar) as fh:
                self.potcar_titles = [m.group(1).strip() for m in
                                      re.finditer(r"TITEL\s*=\s*(.*)", fh.read())]
        self.own = [charge is None for charge in self.structure.charges]

    @property
    def site_species(self) -> list:
        return list(self.structure.symbols)

    def homo_vbm(self) -> float:
        """Highest occupied eigenvalue (occupation > 0.5) over the whole k list."""
        best = -np.inf
        for ik in range(self.wavecar.nk):
            _, _, e, occ = self.wavecar.header(ik)
            if np.any(occ > 0.5):
                best = max(best, float(np.max(e[occ > 0.5])))
        return best

    @property
    def spinor(self) -> bool:
        return self.wavecar.spinor

    @property
    def electrons_per_band(self) -> float:
        """2 for a scalar (ISPIN = 1) run, 1 for a spinor run."""
        return 1.0 if self.wavecar.spinor else 2.0

    def nelect(self) -> float:
        """Number of electrons from the occupations of the first k point."""
        _, _, _, occ = self.wavecar.header(0)
        if self.wavecar.spinor:
            return float(np.sum(occ))
        return float(2 * np.sum(occ))

    def close(self) -> None:
        self.wavecar.close()


class PawSites:
    """PAW data per site of the crystal (from the crystal's POTCAR).

    Raises:
        SystemExit: The POTCAR does not match the crystal's real species.
    """

    def __init__(self, crystal: VaspRun, potcar_path: str, report=print):
        self.datasets = read_potcar(potcar_path)
        st = crystal.structure
        real_species = [s for s in st.species if parse_va_species(s) is None]
        if len(real_species) != len(self.datasets):
            raise SystemExit(f"ERROR: the crystal POSCAR has {len(real_species)} real "
                             f"species, the POTCAR {len(self.datasets)} datasets.")
        self.by_species = dict(zip(real_species, self.datasets))
        for sp, ds in self.by_species.items():
            if ds.element and ds.element != sp:
                report(f"WARNING: POSCAR species {sp} vs POTCAR element {ds.element}")
        self.site_ds = [self.by_species.get(s) for s in st.symbols]
        self.frac = st.positions
        self.lmax = max(int(max(ds.lps)) for ds in self.datasets)


class KPointProjections:
    """Plane-wave data of all runs at one k point, with the PAW projections.

    Holds the coefficients, eigenvalues and occupations of every run, the
    projector matrices of every PAW site and the projections
    ``beta = <p~|psi~>`` (and ``gamma = <phi - phi~|psi~>`` for the exact
    cross-sphere term), per spinor component for ``vasp_ncl`` runs.

    Args:
        runs: ``[crystal, fragment, ...]`` (:class:`VaspRun`).
        ik: 0-based k index (the same in every run).
        paw: :class:`PawSites` of the crystal.
        need_bessel: Also build the Bessel-transform matrices (needed for
            ``mode="bessel"`` and the exact self overlap).

    Raises:
        SystemExit: The runs do not share the plane-wave set at this k, or
            spinor and scalar runs are mixed.
    """

    def __init__(self, runs, ik: int, paw: PawSites, need_bessel: bool = True):
        self.ik = ik
        self.runs = runs
        w0 = runs[0].wavecar
        npl, kvec, _, _ = w0.header(ik)
        for r in runs[1:]:
            n2, k2, _, _ = r.wavecar.header(ik)
            if n2 != npl or np.max(np.abs(k2 - kvec)) > 1e-6:
                raise SystemExit(f"ERROR: k point {ik + 1}: the plane-wave sets differ "
                                 f"between {runs[0].path} and {r.path}.")
        self.kvec = kvec
        self.g = w0.gvectors(ik)
        for r in runs[1:]:
            if not np.array_equal(r.wavecar.gvectors(ik), self.g):
                raise SystemExit("ERROR: the G-vector lists of the runs differ.")
        self.C = [r.wavecar.coeffs(ik) for r in runs]
        self.eps = [r.wavecar.header(ik)[2] for r in runs]
        self.occ = [r.wavecar.header(ik)[3] for r in runs]
        omega = abs(np.linalg.det(w0.lattice))
        kc = (self.g + kvec) @ w0.recip * 2 * np.pi
        qlen = np.sqrt(np.einsum("ij,ij->i", kc, kc))
        qsafe = np.maximum(qlen, 1e-10)
        u = kc / qsafe[:, None]
        ylm = {l: real_ylm(l, u) for l in range(paw.lmax + 1)}
        # the radial form factors depend on the dataset and channel only, not
        # on the site: computed once per dataset (same numbers, fewer calls)
        ff_proj: dict = {}
        ff_bes: dict = {}
        self.P, self.Pd = [], []          # per site: projector / Delta-phi matrices
        for a, ds in enumerate(paw.site_ds):
            if ds is None:
                self.P.append(None)
                self.Pd.append(None)
                continue
            phase = np.exp(2j * np.pi * ((self.g + kvec) @ paw.frac[a])) / math.sqrt(omega)
            rows, rowsd = [], []
            for ch, l in enumerate(ds.lps):
                il = (1j) ** int(l)
                key = (id(ds), ch)
                if key not in ff_proj:
                    ff_proj[key] = ds.projector_ff(ch, qlen)
                f = ff_proj[key]
                rows.append(il * f[None, :] * ylm[int(l)])
                if need_bessel:
                    if key not in ff_bes:
                        ff_bes[key] = ds.radial_ff(ch, qlen, "dphi")
                    fd = ff_bes[key]
                    rowsd.append(il * fd[None, :] * ylm[int(l)])
            self.P.append(np.vstack(rows) * phase[None, :])
            self.Pd.append(np.vstack(rowsd) * phase[None, :] if need_bessel else None)
        # projections beta (nproj, nb) and gamma = <Delta phi | psi~>; for spinor
        # runs per component (Q, M, Delta are spin diagonal)
        spins = {r.wavecar.spinor for r in runs}
        if len(spins) != 1:
            raise SystemExit("ERROR: spinor (vasp_ncl) and scalar runs cannot be mixed.")
        self.ns = 2 if spins.pop() else 1
        self.paw = paw
        if self.ns == 1:
            self.B = [[None if P is None else P @ C.T for P in self.P] for C in self.C]
            self.G = [[None if (Pd is None) else Pd @ C.T for Pd in self.Pd] for C in self.C]
            self.comp = [(self.C, self.B, self.G)]
        else:
            ng = len(self.g)
            self.comp = []
            for sc in range(2):
                Cs = [C[:, sc * ng:(sc + 1) * ng] for C in self.C]
                Bs = [[None if P is None else P @ C.T for P in self.P] for C in Cs]
                Gs = [[None if (Pd is None) else Pd @ C.T for Pd in self.Pd] for C in Cs]
                self.comp.append((Cs, Bs, Gs))
            self.B = self.G = None

    def _parts(self, Cl, Bl, Gl, ia, ib, mode):
        A, B = self.runs[ia], self.runs[ib]
        S_pw = np.conj(Cl[ia]) @ Cl[ib].T
        S_q = np.zeros_like(S_pw)
        S_x = np.zeros_like(S_pw)
        for a, ds in enumerate(self.paw.site_ds):
            if ds is None:
                continue
            oa, ob = A.own[a], B.own[a]
            ba, bb = Bl[ia][a], Bl[ib][a]
            if oa and ob:
                S_q += np.conj(ba).T @ ds.Qfull @ bb
            elif oa and not ob:          # B smooth at a
                if mode == "bessel":
                    S_x += np.conj(ba).T @ Gl[ib][a]
                elif mode == "projector":
                    S_x += np.conj(ba).T @ ds.Mfull.T @ bb
            elif ob and not oa:          # A smooth at a
                if mode == "bessel":
                    S_x += np.conj(Gl[ia][a]).T @ bb
                elif mode == "projector":
                    S_x += np.conj(ba).T @ ds.Mfull @ bb
        return S_pw, S_q, S_x

    def overlap(self, ia: int, ib: int, mode: str = "bessel", terms: bool = False):
        """All-electron overlap ``<phi^A_m | phi^B_n>`` between all bands.

        Args:
            ia: Run index of the bra (0 = crystal).
            ib: Run index of the ket.
            mode: Cross-sphere evaluation (:data:`CROSS_SPHERE_MODES`).
            terms: Also return the plane-wave, one-centre and cross parts.

        Returns:
            ``(nb_A, nb_B)`` complex matrix (summed over the spinor
            components), or ``(S, S_pw, S_q, S_x)`` with ``terms=True``.

        Raises:
            ValueError: Unknown mode.
        """
        if mode not in CROSS_SPHERE_MODES:
            raise ValueError(f"unknown cross-sphere mode {mode!r}")
        S_pw, S_q, S_x = self._parts(*self.comp[0], ia, ib, mode)
        for extra in self.comp[1:]:
            p_pw, p_q, p_x = self._parts(*extra, ia, ib, mode)
            S_pw, S_q, S_x = S_pw + p_pw, S_q + p_q, S_x + p_x
        if terms:
            return S_pw + S_q + S_x, S_pw, S_q, S_x
        return S_pw + S_q + S_x

    def overlap_exact_self(self, ia: int) -> np.ndarray:
        """Self overlap without the partial-wave completeness of psi~ itself.

        ``<psi~|psi~> + <psi~|Dpsi> + <Dpsi|psi~> + <Dpsi|Dpsi>`` with the
        exact Bessel-transform projections.
        """
        S = None
        for Cl, Bl, Gl in self.comp:
            Sc = np.conj(Cl[ia]) @ Cl[ia].T
            for a, ds in enumerate(self.paw.site_ds):
                if ds is None or not self.runs[ia].own[a]:
                    continue
                b, g = Bl[ia][a], Gl[ia][a]
                Sc += np.conj(g).T @ b + np.conj(b).T @ g + np.conj(b).T @ ds.Ddfull @ b
            S = Sc if S is None else S + Sc
        return S

    def sphere_weights(self, ia: int) -> dict:
        """PROCAR-type all-electron sphere weights of run ``ia``.

        Returns:
            ``{(site, l): weights (nb,)}`` over the run's real atoms, in site
            order and ``l`` order.
        """
        out: dict = {}
        for Cl, Bl, Gl in self.comp:
            for a, ds in enumerate(self.paw.site_ds):
                if ds is None or not self.runs[ia].own[a]:
                    continue
                b = Bl[ia][a]
                v = np.real(np.einsum("in,ij,jn->ijn", np.conj(b), ds.QTOTfull, b))
                for l in sorted(set(ds.lps)):
                    mask = ds.l_of == l
                    w = v[mask][:, mask].sum(axis=(0, 1))
                    out[(a, int(l))] = w if (a, int(l)) not in out else out[(a, int(l))] + w
        return out

    def pointcharge_weights(self, ia: int) -> np.ndarray:
        """Smooth density of run ``ia`` inside the crystal's spheres at its Va sites.

        A pseudo-partial-wave estimate (``TPS``); large for the diffuse states
        that live on the point charges of a sublattice run.
        """
        w = np.zeros(len(self.eps[ia]))
        for Cl, Bl, Gl in self.comp:
            for a, ds in enumerate(self.paw.site_ds):
                if ds is None or self.runs[ia].own[a]:
                    continue
                b = Bl[ia][a]
                w += np.real(np.einsum("in,ij,jn->n", np.conj(b), ds.TPSfull, b))
        return w

    def completeness_rel_err(self, ia: int, a: int, ds: PawDataset, msk) -> float:
        """``|| <Dphi_j|psi~> - sum_i M_ij beta_i || / || <Dphi_j|psi~> ||`` per (site, l)."""
        if self.ns == 1:
            g_ex = self.G[ia][a]
            g_pw = ds.Mfull.T @ self.B[ia][a]
            den = np.linalg.norm(g_ex[msk])
            return float(np.linalg.norm(g_ex[msk] - g_pw[msk]) / den) if den else 0.0
        g_ex = np.hstack([Gl[ia][a][msk] for Cl, Bl, Gl in self.comp])
        g_pw = np.hstack([(ds.Mfull.T @ Bl[ia][a])[msk] for Cl, Bl, Gl in self.comp])
        den = np.linalg.norm(g_ex)
        return float(np.linalg.norm(g_ex - g_pw) / den) if den else 0.0

    def orthonormality_tests(self) -> dict:
        """The T1 (crystal) and T2 (fragments) all-electron orthonormality tests.

        Returns:
            The ``tests`` block of one k point (without the analysis-dependent
            entries: fragment columns, Kramers check, Bessel inequality).
        """
        crys = 0
        nb_c = len(self.eps[crys])
        tests: dict = {}
        S, Spw, _, _ = self.overlap(crys, crys, terms=True)
        dev = np.abs(S - np.eye(nb_c))
        tests["T1_crystal_max_dev"] = float(dev.max())
        tests["T1_crystal_max_dev_diag"] = float(np.max(np.abs(np.diag(S).real - 1)))
        tests["T1_crystal_max_dev_offdiag"] = float(np.max(dev - np.diag(np.diag(dev))))
        tests["T1_crystal_pseudo_norm_range"] = [float(np.min(np.diag(Spw).real)),
                                                 float(np.max(np.diag(Spw).real))]
        tests["T1_crystal_without_Q_max_dev"] = float(np.max(np.abs(Spw - np.eye(nb_c))))
        Sx = self.overlap_exact_self(crys)
        tests["T1_crystal_max_dev_exact_sphere_form"] = float(np.max(np.abs(Sx - np.eye(nb_c))))
        frag_tests = []
        for ir in range(1, len(self.runs)):
            nb = len(self.eps[ir])
            S, Spw, _, _ = self.overlap(ir, ir, terms=True)
            frag_tests.append({
                "run": self.runs[ir].path,
                "T2_max_dev": float(np.max(np.abs(S - np.eye(nb)))),
                "pseudo_norm_range": [float(np.min(np.diag(Spw).real)),
                                      float(np.max(np.diag(Spw).real))],
                "without_Q_max_dev": float(np.max(np.abs(Spw - np.eye(nb))))})
        tests["T2_fragments"] = frag_tests
        compl = {}
        for a, ds in enumerate(self.paw.site_ds):
            if ds is None:
                continue
            for l in sorted(set(int(x) for x in ds.lps)):
                msk = ds.l_of == l
                key = f"{self.runs[crys].site_species[a]}{a + 1} {_SHELL_LETTERS[l]}"
                compl[key] = self.completeness_rel_err(crys, a, ds, msk)
        tests["crystal_partial_wave_completeness_rel_err"] = compl
        return tests


# =============================================================== k points
def find_kpoint(kpoints: np.ndarray, candidates, tol: float = 1e-5) -> tuple:
    """Index of a k point in a WAVECAR k list.

    The candidates are tried in order, each with all sign patterns of its
    coordinates (``(0, 1/2, 0)`` also finds ``(0, -1/2, 0)``); when none
    matches, a candidate is accepted up to a reciprocal-lattice vector.

    Args:
        kpoints: ``(nk, 3)`` fractional k points of the file.
        candidates: One fractional k point or a list of equivalent ones.
        tol: Coordinate tolerance.

    Returns:
        ``(index, k_frac)`` with the 0-based index, or ``(None, None)``.
    """
    cands = np.atleast_2d(np.asarray(candidates, float))
    kpoints = np.asarray(kpoints, float)
    for c in cands:
        for sgn in itertools.product((1, -1), repeat=3):
            cc = c * np.array(sgn)
            hit = np.where(np.all(np.abs(kpoints - cc) < tol, axis=1))[0]
            if len(hit):
                return int(hit[0]), [float(x) for x in kpoints[hit[0]]]
    for c in cands:
        d = kpoints - c
        hit = np.where(np.all(np.abs(d - np.round(d)) < tol, axis=1))[0]
        if len(hit):
            return int(hit[0]), [float(x) for x in kpoints[hit[0]]]
    return None, None


def _frac_token(text: str) -> float:
    text = text.strip()
    if "/" in text:
        num, den = text.split("/")
        return float(num) / float(den)
    return float(text)


def parse_kpoint_tokens(tokens) -> list:
    """``["GM=0,0,0", "X=0,1/2,0"]`` -> ``[("GM", (0, 0, 0)), ("X", (0, 0.5, 0))]``.

    Args:
        tokens: ``NAME=k1,k2,k3`` strings (fractions such as ``1/2`` allowed),
            or the single token ``all``.

    Returns:
        The k-point list, or ``"all"``.

    Raises:
        SystemExit: A token without coordinates.
    """
    if [t.lower() for t in tokens] == ["all"]:
        return "all"
    out = []
    for token in tokens:
        if "=" not in token:
            raise SystemExit(f"ERROR: k point {token!r}: give NAME=k1,k2,k3 "
                             "(fractional, e.g. X=0,1/2,0) or 'all'.")
        name, value = token.split("=", 1)
        coords = tuple(_frac_token(x) for x in value.split(","))
        if len(coords) != 3:
            raise SystemExit(f"ERROR: k point {token!r} needs three coordinates.")
        out.append((name.strip(), coords))
    return out


# =============================================================== cache
@dataclass
class KPointOverlaps:
    """Everything the analysis needs at one k point.

    Run index 0 is the crystal, 1.. the fragments (in the order given).

    Attributes:
        name: k-point name (``"GM"``).
        frac: Fractional k point of the WAVECAR.
        index: 0-based index in the WAVECAR k list.
        n_planewaves: Number of G vectors (per spinor component).
        eigenvalues: Per run, eV, raw scale of that run.
        occupations: Per run, VASP occupations (0..1 per band).
        overlaps: ``{mode: [A_f (rows_f, nb_crystal) complex, ...]}`` per
            fragment, ``A[f, n] = <phi_f|psi_n>``.
        rows: Per fragment, the band indices (0-based) of the rows of ``A``.
        fragment_overlaps: ``{mode: {(i, j): S (rows_i, rows_j)}}`` for every
            fragment pair ``i < j`` (run indices).
        sphere_keys: Per run, the ``(site, l)`` pairs of its sphere weights.
        sphere_weights: Per run, ``(n_keys, nb)`` PAW-sphere weights.
        pointcharge_weights: Per run, ``(nb,)`` smooth weight in the spheres
            of its point-charge sites.
        tests: T1/T2 tests and the partial-wave completeness.
        labels: Per run, one irrep label per band (``None`` = unknown), or
            ``None`` when no labels are available for that run.
    """

    name: str
    frac: list
    index: int
    n_planewaves: int
    eigenvalues: list
    occupations: list
    overlaps: dict
    rows: list
    fragment_overlaps: dict
    sphere_keys: list
    sphere_weights: list
    pointcharge_weights: list
    tests: dict = field(default_factory=dict)
    labels: list = field(default_factory=list)

    def row_index(self, fragment: int) -> dict:
        """``{band: row}`` of fragment ``fragment`` (1-based run index)."""
        return {int(b): p for p, b in enumerate(self.rows[fragment - 1])}


class OverlapCache:
    """Overlap data of a crystal and its sublattice runs at selected k points.

    Attributes:
        meta: JSON-serialisable metadata (schema, runs, PAW species, ...).
        kpoints: The :class:`KPointOverlaps` in analysis order.
    """

    def __init__(self, meta: dict, kpoints: list):
        self.meta = meta
        self.kpoints = kpoints

    # -- access ---------------------------------------------------------------
    @property
    def runs(self) -> list:
        return self.meta["runs"]

    @property
    def spinor(self) -> bool:
        return bool(self.meta["runs"][0]["spinor"])

    @property
    def modes(self) -> list:
        return list(self.meta["modes"])

    @property
    def trimmed(self) -> bool:
        return bool(self.meta.get("trimmed"))

    def kpoint(self, name: str) -> KPointOverlaps:
        """The k point called ``name``.

        Raises:
            KeyError: Not in the cache.
        """
        for kp in self.kpoints:
            if kp.name == name:
                return kp
        raise KeyError(name)

    def attach_labels(self, label_provider) -> None:
        """Store per-band irrep labels from ``label_provider``.

        Args:
            label_provider: ``f(run_index, k_name, k_frac) -> list[str] | None``
                (one label per band; ``None`` for no labels).
        """
        for kp in self.kpoints:
            labels = []
            for ir in range(len(self.runs)):
                lab = label_provider(ir, kp.name, list(kp.frac))
                labels.append(None if lab is None else [None if x is None else str(x)
                                                        for x in lab])
            kp.labels = labels

    # -- trimming -------------------------------------------------------------
    def trim(self, keep_rows: dict, kpoints=None, modes=None) -> OverlapCache:
        """A copy restricted to some fragment states, k points and modes.

        Eigenvalues, occupations, sphere weights and labels of every band are
        kept (the shell names and the active space depend on all fragment
        levels); only the rows of ``A`` and of the fragment-fragment overlaps
        are cut.

        Args:
            keep_rows: ``{k_name: [bands of fragment 1, bands of fragment 2,
                ...]}`` (0-based band indices).
            kpoints: Names of the k points to keep (default: all).
            modes: Cross-sphere modes to keep (default: all).

        Returns:
            The trimmed :class:`OverlapCache` (``meta["trimmed"] = True``).
        """
        keep_k = list(kpoints) if kpoints is not None else [kp.name for kp in self.kpoints]
        keep_m = list(modes) if modes is not None else self.modes
        out = []
        for kp in self.kpoints:
            if kp.name not in keep_k:
                continue
            want = [sorted(int(b) for b in bands) for bands in keep_rows[kp.name]]
            pos = []
            for f, bands in enumerate(want):
                idx = kp.row_index(f + 1)
                missing = [b for b in bands if b not in idx]
                if missing:
                    raise ValueError(f"{kp.name}: fragment {f + 1} rows {missing} "
                                     "are not in the cache")
                pos.append([idx[b] for b in bands])
            overlaps = {m: [kp.overlaps[m][f][pos[f]] for f in range(len(want))]
                        for m in keep_m}
            pairs = {}
            for m in keep_m:
                pairs[m] = {(i, j): S[np.ix_(pos[i - 1], pos[j - 1])]
                            for (i, j), S in kp.fragment_overlaps.get(m, {}).items()}
            out.append(KPointOverlaps(
                name=kp.name, frac=list(kp.frac), index=kp.index,
                n_planewaves=kp.n_planewaves, eigenvalues=kp.eigenvalues,
                occupations=kp.occupations, overlaps=overlaps,
                rows=[np.array(b, int) for b in want], fragment_overlaps=pairs,
                sphere_keys=kp.sphere_keys, sphere_weights=kp.sphere_weights,
                pointcharge_weights=kp.pointcharge_weights, tests=kp.tests,
                labels=kp.labels))
        meta = json.loads(json.dumps(self.meta))
        meta["trimmed"] = True
        meta["modes"] = keep_m
        return OverlapCache(meta, out)

    # -- persistence ----------------------------------------------------------
    def save(self, path: str) -> str:
        """Write the cache as a compressed ``.npz`` (no pickle).

        Args:
            path: Output file; ``.npz`` is appended by numpy when missing.

        Returns:
            The path written.
        """
        arrays: dict = {}
        kmeta = []
        for i, kp in enumerate(self.kpoints):
            p = f"k{i}__"
            nrun = len(kp.eigenvalues)
            for r in range(nrun):
                arrays[f"{p}eps__{r}"] = np.asarray(kp.eigenvalues[r], float)
                arrays[f"{p}occ__{r}"] = np.asarray(kp.occupations[r], float)
                arrays[f"{p}sw__{r}"] = np.asarray(kp.sphere_weights[r], float)
                arrays[f"{p}pcw__{r}"] = np.asarray(kp.pointcharge_weights[r], float)
            for m, mats in kp.overlaps.items():
                for f, A in enumerate(mats):
                    arrays[f"{p}A__{m}__{f + 1}"] = np.asarray(A, complex)
            for m, pairs in kp.fragment_overlaps.items():
                for (a, b), S in pairs.items():
                    arrays[f"{p}S__{m}__{a}__{b}"] = np.asarray(S, complex)
            for f, rows in enumerate(kp.rows):
                arrays[f"{p}rows__{f + 1}"] = np.asarray(rows, np.int32)
            kmeta.append({
                "name": kp.name, "frac": [float(x) for x in kp.frac], "index": int(kp.index),
                "n_planewaves": int(kp.n_planewaves),
                "sphere_keys": [[[int(a), int(l)] for a, l in keys]
                                for keys in kp.sphere_keys],
                "pairs": sorted({f"{a},{b}" for pairs in kp.fragment_overlaps.values()
                                 for (a, b) in pairs}),
                "tests": kp.tests, "labels": kp.labels})
        meta = {k: v for k, v in self.meta.items() if k != "source"}   # set by load()
        meta["schema"] = CACHE_SCHEMA
        meta["kpoints"] = kmeta
        arrays["meta"] = np.array(json.dumps(meta))
        np.savez_compressed(path, **arrays)
        return path if path.endswith(".npz") else path + ".npz"

    @classmethod
    def load(cls, path: str) -> OverlapCache:
        """Read a cache written by :meth:`save`.

        Args:
            path: The ``.npz`` file.

        Returns:
            The :class:`OverlapCache`.

        Raises:
            SystemExit: Missing file or wrong schema.
        """
        if not os.path.isfile(path):
            raise SystemExit(f"ERROR: no overlap cache at {path}.")
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            if meta.get("schema") != CACHE_SCHEMA:
                raise SystemExit(f"ERROR: {path}: schema {meta.get('schema')!r}, "
                                 f"expected {CACHE_SCHEMA!r}.")
            kpoints = []
            nrun = len(meta["runs"])
            for i, km in enumerate(meta["kpoints"]):
                p = f"k{i}__"
                overlaps = {m: [data[f"{p}A__{m}__{f}"] for f in range(1, nrun)]
                            for m in meta["modes"]}
                pairs: dict = {}
                for m in meta["modes"]:
                    pairs[m] = {}
                    for key in km.get("pairs", []):
                        a, b = (int(x) for x in key.split(","))
                        name = f"{p}S__{m}__{a}__{b}"
                        if name in data:
                            pairs[m][(a, b)] = data[name]
                kpoints.append(KPointOverlaps(
                    name=km["name"], frac=km["frac"], index=km["index"],
                    n_planewaves=km["n_planewaves"],
                    eigenvalues=[data[f"{p}eps__{r}"] for r in range(nrun)],
                    occupations=[data[f"{p}occ__{r}"] for r in range(nrun)],
                    overlaps=overlaps,
                    rows=[data[f"{p}rows__{f}"].astype(int) for f in range(1, nrun)],
                    fragment_overlaps=pairs,
                    sphere_keys=[[(int(a), int(l)) for a, l in keys]
                                 for keys in km["sphere_keys"]],
                    sphere_weights=[data[f"{p}sw__{r}"] for r in range(nrun)],
                    pointcharge_weights=[data[f"{p}pcw__{r}"] for r in range(nrun)],
                    tests=km.get("tests") or {}, labels=km.get("labels") or []))
        meta = {k: v for k, v in meta.items() if k != "kpoints"}
        meta["source"] = os.path.abspath(path)
        return cls(meta, kpoints)


def _check_runs(runs) -> None:
    """The fragment runs must share cell, sites, k list and ENCUT with the crystal."""
    crystal = runs[0]
    cst = crystal.structure
    for r in runs[1:]:
        st = r.structure
        if len(st.symbols) != len(cst.symbols) or \
                np.max(np.abs(st.lattice - cst.lattice)) > 1e-6:
            raise SystemExit(f"ERROR: {r.path}: cell or number of sites differ from the "
                             "crystal run.")
        dd = st.positions - cst.positions
        dd -= np.round(dd)
        if np.max(np.abs(dd)) > 1e-5:
            raise SystemExit(f"ERROR: {r.path}: site positions differ from the crystal run.")
        for a, (sf, sc) in enumerate(zip(st.symbols, cst.symbols)):
            if parse_va_species(sf) is None and sf != sc:
                raise SystemExit(f"ERROR: {r.path}: site {a + 1} is {sf} but {sc} in the "
                                 "crystal run.")
        if r.wavecar.nk != crystal.wavecar.nk or \
                abs(r.wavecar.encut - crystal.wavecar.encut) > 1e-6 or \
                np.max(np.abs(r.wavecar.kpoints() - crystal.wavecar.kpoints())) > 1e-6:
            raise SystemExit(f"ERROR: {r.path}: k list or ENCUT differ from the crystal run.")


def _formula(species, counts) -> str:
    return "".join(f"{sp}{n if n > 1 else ''}" for sp, n in zip(species, counts))


def _run_meta(run: VaspRun) -> dict:
    st = run.structure
    real = [sp for sp in st.species if parse_va_species(sp) is None]
    own_counts = [sum(1 for x, o in zip(st.symbols, run.own) if o and x == sp)
                  for sp in real]
    return {
        "path": run.path, "role": run.role, "species": list(st.species),
        "counts": [int(n) for n in st.counts], "site_species": list(st.symbols),
        "positions": np.asarray(st.positions, float).tolist(),
        "lattice": np.asarray(st.lattice, float).tolist(),
        "own": [bool(x) for x in run.own],
        "charges": [None if c is None else float(c) for c in st.charges],
        "formula": _formula(real, own_counts),
        "nbands": int(run.wavecar.nb), "nelect": run.nelect(),
        "encut": float(run.wavecar.encut), "nk": int(run.wavecar.nk),
        "spinor": bool(run.spinor), "electrons_per_band": run.electrons_per_band,
        "homo_raw": run.homo_vbm(), "efermi_header": float(run.wavecar.efermi),
        "potcar_titles": list(run.potcar_titles)}


def _paw_meta(paw: PawSites) -> dict:
    out = {}
    for sp, ds in paw.by_species.items():
        out[sp] = {
            "title": ds.title, "element": ds.element, "vrhfin": ds.vrhfin,
            "zval": float(ds.zval), "lps": [int(l) for l in ds.lps], "rdep_A": ds.rdep,
            "qpaw_file_vs_computed_max": ds.qpaw_check,
            "Q": ds.Q.round(6).tolist(), "M": ds.M.round(6).tolist(),
            "duality_check_q_space": ds.duality_check().round(4).tolist(),
            "valence_shells": [[int(n), int(l)] for n, l in
                               valence_shells(sp, ds.vrhfin, ds.zval, ds.configuration)]}
    return out


def build_overlap_cache(crystal_dir: str, fragment_dirs, kpoints, *,
                        modes=CROSS_SPHERE_MODES, potcar: str | None = None,
                        label_provider=None, report=print) -> OverlapCache:
    """Read the runs and compute the overlap data at the requested k points.

    Args:
        crystal_dir: Run directory of the crystal (POSCAR, POTCAR, WAVECAR).
        fragment_dirs: Run directories of the sublattice runs (POSCAR with Va
            species, WAVECAR; a POTCAR, when present, is checked against the
            crystal's).
        kpoints: ``[(name, frac), ...]`` where ``frac`` is one fractional k
            point or a list of equivalent candidates, or ``"all"`` (every k
            point of the WAVECAR, named ``k1``, ``k2``, ...).
        modes: Cross-sphere modes to store (``"bessel"`` is the reference).
        potcar: POTCAR with the PAW data (default: the crystal's).
        label_provider: Optional ``f(run_index, k_name, k_frac) -> list[str]``.
        report: Print function for progress lines (``None`` = silent).

    Returns:
        The :class:`OverlapCache`.

    Raises:
        SystemExit: Inconsistent runs, unsupported WAVECARs, no k point found.
    """
    say = report or (lambda *a, **k: None)
    modes = [m for m in CROSS_SPHERE_MODES if m in set(modes)]
    if "bessel" not in modes:
        modes = ["bessel"] + modes
    t0 = time.time()
    crystal = VaspRun(crystal_dir, "crystal")
    fragments = [VaspRun(f, "fragment") for f in fragment_dirs]
    runs = [crystal] + fragments
    try:
        _check_runs(runs)
        paw = PawSites(crystal, potcar or os.path.join(crystal_dir, "POTCAR"), report=say)
        titles = [ds.title for ds in paw.datasets]
        for r in fragments:
            for t in r.potcar_titles:
                if t not in titles:
                    raise SystemExit(f"ERROR: {r.path}: POTCAR dataset '{t}' is not in "
                                     "the crystal POTCAR.")
        kpts = crystal.wavecar.kpoints()
        if isinstance(kpoints, str) and kpoints.lower() == "all":
            targets = [(f"k{ik + 1}", ik) for ik in range(len(kpts))]
        else:
            targets = []
            for name, cands in kpoints:
                ik, _ = find_kpoint(kpts, cands)
                if ik is None:
                    say(f"WARNING: k point {name} {list(np.ravel(cands))} is not in the "
                        "k list of the WAVECAR; skipped")
                    continue
                targets.append((name, ik))
        if not targets:
            raise SystemExit("ERROR: none of the requested k points is in the WAVECAR.")
        meta = {
            "schema": CACHE_SCHEMA,
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "modes": modes, "trimmed": False,
            "runs": [_run_meta(r) for r in runs],
            "vbm_raw": crystal.homo_vbm(),
            "paw": _paw_meta(paw),
            "potcar": os.path.abspath(potcar or os.path.join(crystal_dir, "POTCAR"))}
        try:
            from . import __version__
            meta["crystod_version"] = __version__
        except ImportError:                      # pragma: no cover
            meta["crystod_version"] = None
        out = []
        for name, ik in targets:
            t1 = time.time()
            kd = KPointProjections(runs, ik, paw, need_bessel=True)
            tests = kd.orthonormality_tests()
            overlaps = {m: [kd.overlap(ir, 0, mode=m) for ir in range(1, len(runs))]
                        for m in modes}
            pairs = {m: {(i, j): kd.overlap(i, j, mode=m)
                         for i in range(1, len(runs)) for j in range(i + 1, len(runs))}
                     for m in modes}
            keys, weights = [], []
            for ir in range(len(runs)):
                sw = kd.sphere_weights(ir)
                keys.append(list(sw.keys()))
                weights.append(np.array([sw[key] for key in sw]) if sw else
                               np.zeros((0, len(kd.eps[ir]))))
            out.append(KPointOverlaps(
                name=name, frac=[float(x) for x in kd.kvec], index=ik,
                n_planewaves=int(len(kd.g)),
                eigenvalues=[np.array(e) for e in kd.eps],
                occupations=[np.array(o) for o in kd.occ],
                overlaps=overlaps,
                rows=[np.arange(len(kd.eps[ir])) for ir in range(1, len(runs))],
                fragment_overlaps=pairs, sphere_keys=keys, sphere_weights=weights,
                pointcharge_weights=[kd.pointcharge_weights(ir) for ir in range(len(runs))],
                tests=tests, labels=[]))
            say(f"[{name}] k#{ik + 1} {[round(float(x), 6) for x in kd.kvec]}  "
                f"T1={tests['T1_crystal_max_dev']:.2e}  "
                f"T2={[float('%.2e' % t['T2_max_dev']) for t in tests['T2_fragments']]}"
                f"  ({time.time() - t1:.2f} s)")
            del kd
        cache = OverlapCache(meta, out)
        if label_provider is not None:
            cache.attach_labels(label_provider)
        cache.meta["seconds_stage1"] = round(time.time() - t0, 2)
        return cache
    finally:
        for r in runs:
            r.close()
