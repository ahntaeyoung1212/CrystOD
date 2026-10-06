"""Crystal-orbital diagrams from finished VASP runs (crystod --diagram --vasp).

The quantitative engine that reads plane-wave DFT instead of running it.  It
needs three ordinary VASP calculations that share one cell:

    column          real atoms                 switched-off sublattice
    left fragment   the --co-left sublattice   Va point charges
    right fragment  the --co-right sublattice  Va point charges
    crystal         everything                 --

The sublattice runs use the CrystOD-patched VASP, where a POSCAR species named
``Va<q><sign>`` is a classical (Gaussian-smeared, wall-regularized) point
charge with no electrons and no projectors, so the removed sublattice is
present electrostatically and absent chemically -- the plane-wave counterpart
of the point-charge lattice of ``--pyscf``.  ``--vasp-setup`` writes the inputs
of those runs; ``--vasp`` reads them back.

What is taken from the runs, and how:

* **Levels**: the eigenvalues and occupations of ``EIGENVAL``/``PROCAR`` at the
  tabulated special k points of the space group.
* **Symmetry**: ``LORBIT = 12`` gives the complex projections
  ``<Y_lm at atom | psi_nk>``.  They are treated as a vector in an AO-like
  ``(atom, l, m)`` basis and classified with exactly the representation
  matrices crystod builds for a genuine AO basis (site permutation x real
  Wigner matrices, conjugated into the atomic Bloch gauge that VASP's
  projectors use), so the irrep labels are the same objects the extended-
  Hueckel and PySCF engines print.
* **Character**: per level, the shares by (element, l) of the projected weight.
* **Composition**: a crystal level's share on a fragment is its projected
  weight on that fragment's atoms, distributed over the fragment levels of the
  same irrep in proportion to the gauge-invariant projection overlap.
* **Alignment**: each of the three runs pins its own G = 0 average potential to
  zero, so the columns are put on one scale by the same XPS-style deep-level
  alignment the ``--pyscf`` engine uses (``align_fragment_columns``).
* **Bond character**: a plane-wave calculation has no overlap matrix, so the
  COOP sign of the other engines is unavailable; bonding/antibonding is read
  from the aligned energies against the composition-weighted parent energy.

Spin-polarized runs are not supported (a clean ERROR), nor are spin-orbit
(non-collinear) runs, which the WAVECAR-overlap engine reads instead
(``--vasp-engine overlap``, :mod:`crystod.crystal_orbital_overlap`: no energy
alignment at all, frozen-ion parents and a sublattice-orbital COHP from the
all-electron overlaps of the sublattice and crystal Bloch states).
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass

import numpy as np

from .crystal_orbital_diagram import (
    CrystalOrbitalDiagram,
    DiagramLevel,
    _format_kpoint,
    print_dipole_rules,
    write_crystal_diagram_html,
)
from .operations import wigner_D_real
from .runtime_compat import get_character, get_chemical_symbols, get_scaled_positions
from .visualize_basis import SymmetryAdaptedOrbitalBasis

# A level is kept only when its mean projected weight per degenerate partner,
# summed over the column's real atoms and all lm channels, reaches this value.
# The PAW spherical projections of a PROCAR do NOT sum to one: a genuine
# atom-centred band of SrTiO3 collects 0.4-1.0, while the free-electron-like
# bands that live between the atoms (and the states that pile up on the Va
# point-charge sites of a fragment run) collect 0.0-0.16.  This is the
# plane-wave analogue of the PySCF engine's GHOST_FRACTION_THRESHOLD.
#
# The two populations are well separated AT GAMMA (0.27 against 0.50 in the
# SrTiO3 anion run), but NOT away from it: in the cation run at X the nearest
# pair around the cut is 0.279 (dropped) against 0.315 (kept), a 13 % margin,
# and at M 0.275 against 0.347.  Which of the high-lying, weakly projected
# fragment levels appear is therefore sensitive to this number -- they all lie
# above the default drawing window, so nothing drawn moves, but a user who
# opens the window should test the sensitivity with
# ``--vasp-projection-floor``.  The engine prints the weights that bracket the
# cut once per run.
PROJECTION_FLOOR = 0.30

# Seed window (eV) for clustering degenerate bands, widened until the irrep
# multiplicities of a group come out integral -- the PySCF engine's rule.
DEGENERACY_SEED_EV = 1.0e-3
DEGENERACY_MAX_WINDOW_EV = 0.30
_MULTIPLICITY_TOL = 0.05

# The runtime self-check: a labelled level whose best irrep carries less than
# this much of its projection is reported as a symmetry warning.
IRREP_PURITY_WARN = 0.90

# Default crystal energy window, relative to the valence-band maximum.
DEFAULT_WINDOW = (None, 10.0)

# A crystal level with at least this share on ONE fragment has no partner to
# bond with and is drawn nonbonding.
NONBONDING_SHARE = 0.95

# Energy difference (eV) below which a crystal level counts as degenerate with
# its composition-weighted parents, i.e. nonbonding.  It is of the order of the
# residual uncertainty of the column alignment itself (the trace diagnostic
# printed by the report measures that uncertainty for the run at hand).
BOND_ENERGY_TOL = 0.30

# Per-(element, shell) shares below this are not listed in the tooltip, the
# panel bars or the level table.
SHARE_FLOOR = 5.0e-4

# Connector weights below this are dropped before the lines are drawn and
# before the parent energy is formed.
CONNECTOR_FLOOR = 1.0e-3

# Column-scale test (see _column_scale_report).  For every fragment level that
# has a crystal counterpart of at least this absolute composition, the residual
# delta = E_crystal - E_fragment is collected per (element, shell).  One rigid
# shift can describe a column only if those deltas agree; when the spread over
# the shells of one column exceeds COLUMN_SCALE_TOL the column is NOT on one
# scale with the crystal and the bond-character colouring that would rest on it
# is switched off.
COLUMN_SCALE_PURITY = 0.80
COLUMN_SCALE_TOL = 1.0

# ------------------------------------------------- site-resolved alignment
#
# A fragment run and the crystal run differ by ONE constant only if every atom
# of the fragment feels the same change of site potential.  On SrTiO3 they do
# not: the Ti-O covalent back-donation charges the Ti site in the crystal and
# raises EVERY Ti level by ~11 eV through the intra-atomic Coulomb repulsion,
# which the bare Ti(4+) + point-charge fragment lacks, while the ionic Sr site
# barely moves.  Shells of the SAME atom move together; different SITES do not.
# The default alignment therefore fits one shift per (column, ELEMENT).
#
# A fragment level is an anchor candidate when it has a crystal counterpart of
# one of two kinds, and its dominant element carries at least
# ANCHOR_DOMINANT_SHARE of its projected weight (otherwise the anchor would not
# belong to one site):
#
#   * a SYMMETRY-FORBIDDEN probe -- the fragment level's irrep at that k does
#     not occur among the other sublattice's projected manifolds at all, so no
#     state of the other sublattice may mix with it.  Such a level is strictly
#     nonbonding: after a correct alignment it MUST coincide with the crystal
#     level of that irrep it overlaps most.  This is an exact constraint, not
#     an approximate one, which is why it is weighted ANCHOR_WEIGHT_FORBIDDEN
#     times a pure counterpart;
#   * a PURE counterpart -- a crystal level carrying at least
#     COLUMN_SCALE_PURITY of its own projected weight along the fragment level
#     (the XPS-style anchors the rigid alignment already used).  These still
#     carry the bonding shift of the pair they belong to, and because the
#     element's shift is fitted TO them their residuals average to zero: an
#     element whose anchors are all of this kind cannot show a net shift, only
#     the scatter around it.  Hence the strong preference for the first kind.
#
# Weighting the two kinds against each other is not enough, and the review
# measured why.  With delta_Ti the purity- and class-weighted mean of 2
# forbidden probes (+15.177, +15.332) and 6 pure counterparts (+17.09 .. +18.12)
# the fit came out at +16.131 -- and the 2.9 eV spread of the pure list is NOT
# model error, it is the bonding shift itself (the eg members sit 1.9 eV higher
# because they ARE the antibonding partners).  The scale was therefore fitted
# to the very shifts the diagram is then asked to read, and the two EXACT
# constraints missed their crystal partners by +0.954 and +0.799 eV although
# they agree with each other to 0.155 eV.  So: when an element has at least one
# symmetry-forbidden probe, its delta is fitted to THOSE ALONE (purity-weighted)
# and the pure counterparts are reported as what they are -- the measured
# bonding shift of each pair.  An element with no forbidden probe keeps the
# combined fit, with the zero-sum caveat below printed; CaF2 Ca is that case.
ANCHOR_DOMINANT_SHARE = 0.70
ANCHOR_WEIGHT_FORBIDDEN = 4.0

# The anchor pool is restricted to the levels the DEFAULT window would draw
# (crystal VBM + this, both columns; the fragment column placed by the robust
# rigid shift first).  Two reasons: the PROCAR basis has one radial channel per
# (atom, l), so a high-lying "Sr 5s" fragment level projects perfectly onto the
# Sr 4s semicore BAND 45 eV below it and would otherwise enter the fit as a
# -42 eV anchor; and the alignment must describe the diagram that is drawn.
# The bound is deliberately the DEFAULT window and never --vasp-window, so that
# cutting the drawing cannot move the energy scale (see apply_window).
ANCHOR_WINDOW_EV = 10.0

# Per-element scale test.  The premise of the site-resolved alignment is that
# the shells of one atom move together, so the test is the spread of the
# element's per-SHELL mean deltas.  Above this the element is not described by
# one shift and the bond character of every crystal level drawing on it is left
# unclassified.
ELEMENT_SCALE_TOL = 1.0

# A fragment level counts as a site that may VETO the classification of a
# crystal level -- because its own element failed the scale test, or because it
# has no single site -- only from this connector weight upwards.  Below it a
# connector is an orthogonality tail of the radially blind projection basis (a
# valence band always keeps one or two percent inside the semicore channel of
# the same l), not a real parentage, and a 2 % tail must not cost a level its
# colour.
#
# The floor is deliberately NOT applied to the PARENT ENERGY any more.  Used
# there it produced a bookkeeping contradiction: at W in CaF2 the three levels
# of irrep W2 are one Ca 3p / F 2p / Ca 3d problem, and dropping the 4.9 %
# Ca 3p admixture of the middle one left it with a single parent, renormalized
# to 1.0, that it sat 0.43 eV above -- so BOTH the middle and the upper level
# came out antibonding and none bonding.  The parent average now runs over ALL
# parents, with the deep semicore shells excluded instead (see
# :meth:`VASPCrystalOrbitalDiagram._semicore_shells`), which is what
# ``--valence-only`` does for the extended-Hueckel and PySCF engines and is the
# physically meaningful version of the same cut.
PARENT_WEIGHT_FLOOR = 0.05

# A bond character may only be read from a parent set that actually holds the
# level.  Below this coverage -- the sum of the connector weights, i.e. the
# fraction of the crystal level's projected character that HAS a fragment level
# on the page -- the sign of E - E_parents is the sign of whatever minority
# channel happened to survive the window, not of the level's own bonding.
# SrTiO3 R4- #2 (+6.95 eV, 79.3 % Sr 4d / 10.7 % O 2s / 10.0 % Ti 4p) came out
# antibonding and red because its only on-page parent was O 2s R4- 23.4 eV
# below it, holding 10.7 % of the level; X5- #3 and M1+ #4 did the same at
# 0.22 and 0.18 coverage.  Such a level is now left UNCLASSIFIED.
PARENT_COVERAGE_FLOOR = 0.50

# A fragment ``(element, shell)`` block whose highest OCCUPIED level sits at
# least this far below the crystal VBM, on the aligned scale, is an
# energetically inert semicore shell.  Its admixture in a valence level is the
# on-site orthogonality tail of the PAW projection, not a bonding partner, so
# it is left out of the parent energy.  Same number and same criterion as
# ``--valence-only`` in ``visualize_pyscf`` and ``visualize_eht``.
SEMICORE_DEPTH_EV = 12.0


@dataclass
class VaspSpec:
    """One ``(element, shell)`` block of the PROCAR ``(atom, l, m)`` basis.

    Duck-typed replacement for ``SublatticeSpec``: the diagram code only ever
    reads ``element``, ``shell``, ``l``, ``sites``, ``column``, ``offset`` and
    ``n_ao`` off it.
    """

    element: str
    shell: str
    l: int
    sites: list
    column: str
    offset: int = 0

    @property
    def n_ao(self) -> int:
        return len(self.sites) * (2 * self.l + 1)


def _level_shell(level) -> str:
    """``"Sr 4d"``: the ``(element, n l)`` manifold a level's own label quotes.

    The PROCAR itself resolves only ``(atom, l)``, but the shell namer of
    :meth:`VASPCrystalOrbitalDiagram._shell_namer` walks each ``(element, l)``
    channel in energy order and hands out the PAW valence ``n`` of that
    channel, so every share row and every fragment label IS ``n``-resolved.
    This is the key the connector hand-off, the parent energy and the semicore
    cut all use, so that a crystal level's ``Sr 5s`` character can never be
    handed to the ``Sr 4s`` semicore level 39 eV below it.
    """
    parts = str(getattr(level, "base_label", None) or level.label or "").split()
    return " ".join(parts[:2]) if len(parts) >= 2 else ""


def _share_channels(level) -> dict:
    """``{"Sr 5s": weight, ...}`` of a level, from its displayed share rows.

    The rows are ``"<element> <n l> <irrep>"`` with one row per
    ``(element, l)`` channel, so the keys are unique and the weights are the
    same normalized shares :meth:`_shares` produced.
    """
    channels: dict = {}
    for text, value in getattr(level, "display_composition", []) or []:
        key = " ".join(str(text).split()[:2])
        if key:
            channels[key] = channels.get(key, 0.0) + float(value)
    return channels


def _element_shares(level) -> dict:
    """``{element: share of the level's projected weight}`` (sums to ~1).

    :attr:`DiagramLevel.shares` is resolved by ``(element, l)`` channel; the
    site-resolved alignment asks which ATOM a level lives on, so the l
    channels of one element are added up.
    """
    totals: dict = {}
    for value, element, _l in getattr(level, "shares", []) or []:
        totals[element] = totals.get(element, 0.0) + float(value)
    return totals


def _dominant_element(level):
    """``(element, share)`` of the atom a level mostly lives on, or ``(None, 0)``."""
    totals = _element_shares(level)
    if not totals:
        return (None, 0.0)
    norm = sum(totals.values()) or 1.0
    element = max(totals, key=lambda key: totals[key])
    return (element, totals[element] / norm)


def _bond_class(energy, parent):
    """``"bonding"`` / ``"nonbonding"`` / ``"antibonding"`` of one level."""
    if parent is None:
        return None
    if energy < parent - BOND_ENERGY_TOL:
        return "bonding"
    if energy > parent + BOND_ENERGY_TOL:
        return "antibonding"
    return "nonbonding"


def _fit_site_delta(entries):
    """``(delta, fit)`` of one ``(column, element)`` from its anchor rows.

    ``entries`` are the tuples :meth:`VASPCrystalOrbitalDiagram.site_anchors`
    returns.  A symmetry-forbidden probe is an EXACT constraint -- nothing of
    the other sublattice may mix with that fragment level, so after the shift
    it must coincide with its crystal partner -- while a pure counterpart
    still carries the bonding shift of the pair it belongs to.  Mixing the two
    fits the scale to the bonding shifts, so as soon as ONE forbidden probe
    exists the delta is the purity-weighted mean of the probes alone
    (``fit = "forbidden"``).  Only an element without any probe falls back to
    the purity- and class-weighted mean of everything (``fit = "all"``), whose
    residuals average to zero by construction.
    """
    entries = list(entries)
    probes = [entry for entry in entries if entry[1]]
    if probes:
        return (_weighted_mean([entry[0] for entry in probes],
                               [entry[2] for entry in probes]), "forbidden")
    return (_weighted_mean(
        [entry[0] for entry in entries],
        [entry[2] * (ANCHOR_WEIGHT_FORBIDDEN if entry[1] else 1.0)
         for entry in entries]), "all")


def _weighted_mean(values, weights) -> float:
    """Weighted mean, falling back to the plain mean when every weight is 0."""
    total = float(np.sum(weights))
    if total <= 0.0:
        return float(np.mean(values)) if len(values) else float("nan")
    return float(np.dot(values, weights) / total)


def _spread(values) -> float:
    """``max - min`` of a sequence (0.0 for fewer than two entries)."""
    values = list(values)
    return float(max(values) - min(values)) if len(values) > 1 else 0.0


def _suffix_duplicate_labels(levels) -> None:
    """Number the fragment levels that share a base label (``Sr 4d X4-#1``).

    Called once in :meth:`VASPCrystalOrbitalDiagram.solve_at` and AGAIN after
    :meth:`VASPCrystalOrbitalDiagram.apply_window`, from the untouched
    ``base_label``: a window that removes the ``#2`` twin must not leave a
    dangling ``#1`` on the page.
    """
    counts: dict = {}
    for level in levels:
        base = getattr(level, "base_label", level.label)
        counts[base] = counts.get(base, 0) + 1
    occurrence: dict = {}
    for level in levels:
        base = getattr(level, "base_label", level.label)
        if counts[base] > 1:
            occurrence[base] = occurrence.get(base, 0) + 1
            level.label = f"{base}#{occurrence[base]}"
        else:
            level.label = base


def _canonical(space, tolerance=1e-9):
    """Orthonormal basis of the column span of ``space`` (S = 1)."""
    gram = space.conj().T @ space
    values, vectors = np.linalg.eigh(gram)
    scale = float(values.max()) if values.size else 0.0
    keep = values > tolerance * max(scale, 1e-30)
    if not np.any(keep):
        return np.zeros((space.shape[0], 0), dtype=complex)
    return space @ (vectors[:, keep] / np.sqrt(values[keep]))


def _multiplicities(space, representation, irreps):
    """Irrep content of the span of ``space`` (may be fractional)."""
    orthonormal = _canonical(space)
    if orthonormal.shape[1] == 0:
        return [0.0] * len(irreps), orthonormal
    characters = np.array([
        np.trace(orthonormal.conj().T @ D @ orthonormal) for D in representation
    ])
    order = len(representation)
    return [
        float(np.real(np.sum(characters * np.conj(
            np.array(get_character(irrep), dtype=complex))) / order))
        for irrep in irreps
    ], orthonormal


class VASPCrystalOrbitalDiagram(CrystalOrbitalDiagram):
    """Crystal-orbital diagram engine reading three finished VASP runs.

    Args:
        cell: The crystal structure as ``phonopy.structure.atoms.PhonopyAtoms``
            (reduced to the spglib primitive cell).
        left_tokens: ``--co-left`` formula tokens, e.g. ``["SrTi"]``.
        right_tokens: ``--co-right`` formula tokens, e.g. ``["O3"]``.
        directories: ``{"mo": dir, "left": dir, "right": dir}`` -- the VASP
            directories of the crystal and of the two sublattice runs.
        symprec: Symmetry tolerance handed to spglib.
        electrons: Electrons per cell of the crystal column; the default is
            the run's own ``NELECT``.
        oxidation: ``{element: formal charge}``; the default is pymatgen's
            guess.  Only the report and the chips use it here -- the point
            charges themselves are already baked into the VASP runs.
        window: ``(emin, emax)`` of the crystal column, relative to the
            valence-band maximum; ``None`` entries mean "no bound" / +10 eV.
        conventional: Draw the hover sketches in the conventional cell.
        degeneracy_tol: Seed window (eV) of the degeneracy clustering.
        align_mode: ``"site"`` (the default: one shift per fragment ELEMENT,
            see :meth:`site_anchor_report`) or ``"rigid"`` (one shift per
            fragment column, the XPS-style anchor of the PySCF engine).
        zero: Energy zero of the page and of the tables -- ``"vbm"``
            (default, ``E - E_VBM``), ``"efermi"`` or ``"raw"``.

    Attributes:
        runs: ``{column: {"structure", "procar", "outcar", "dir", "ions"}}``.
        specs: One :class:`VaspSpec` per (element, shell) block;
            ``side_specs[column]`` those of one fragment.
        n_ao: Size of the ``(atom, l, m)`` projection basis.

    Raises:
        SystemExit: A run is missing, spin polarized, does not match the
            primitive cell, or does not cover the special k points.
    """

    def __init__(self, cell, left_tokens, right_tokens, *, directories,
                 symprec=1e-5, electrons=None, oxidation=None, window=None,
                 conventional=False, degeneracy_tol=None,
                 projection_floor=None, anchor=None, align_mode="site",
                 zero="vbm"):
        from .crystal_orbital_pyscf import PySCFCrystalOrbitalDiagram

        self.conventional = bool(conventional)
        self.projection_floor = (PROJECTION_FLOOR if projection_floor is None
                                 else float(projection_floor))
        self.align_mode = str(align_mode)
        self.zero_mode = str(zero)
        # {column: (element, shell)} forced anchors of --vasp-anchor; the
        # column is resolved from the element once the fragments are assigned
        self.anchor_request = list(anchor or [])
        self.anchor_override: dict = {}
        # --vasp-anchor pins ONE shell of ONE column, which is a statement
        # about a rigid column shift; in site mode it would only perturb the
        # anchor window and silently move the per-element fits, so it selects
        # the rigid alignment instead of quietly doing something else
        self.anchor_note = ""
        if self.anchor_request and self.align_mode == "site":
            self.align_mode = "rigid"
            self.anchor_note = (
                "   (--vasp-anchor pins one shell of a whole column, which is "
                "the RIGID alignment;\n"
                "    --vasp-align rigid is used instead of the site-resolved "
                "default)")
        # {column: bool} -- filled by _column_scale_report after the alignment
        self.column_scale_ok = {"left": True, "right": True}
        # {column: {element: bool}} -- filled by site_anchor_report
        self.element_scale_ok: dict = {"left": {}, "right": {}}
        self.site_report: dict = {}
        self.rigid_shifts: dict = {}
        self.column_scale_notes: list = []
        self.floor_margin: list = []
        self.builder = SymmetryAdaptedOrbitalBasis(cell=cell, symprec=symprec)
        primitive = self.builder.primitive_cell
        self.symbols = get_chemical_symbols(primitive)
        self.positions = np.array(get_scaled_positions(primitive))
        self.lattice = np.array(primitive.cell)
        self.cartesian = self.positions @ self.lattice
        self.symprec = float(symprec)
        self.sketch_specs = None
        self.sketch_tokens = None
        self.degeneracy_tol = (DEGENERACY_SEED_EV if degeneracy_tol is None
                               else float(degeneracy_tol))
        self.degeneracy_window = DEGENERACY_MAX_WINDOW_EV

        PySCFCrystalOrbitalDiagram._assign_fragments(self, left_tokens, right_tokens)
        PySCFCrystalOrbitalDiagram._resolve_oxidation(self, oxidation)
        self._resolve_anchor_override()

        self.directories = dict(directories)
        # {column: RunMapping} -- how each run's POSCAR sits on the -c cell;
        # the crystal run is mapped first and its origin shift is then the
        # preferred one, so that the three runs of one calculation are
        # described in the same setting even where a fragment run's Va sites
        # would allow another
        self.mappings: dict = {}
        self.preferred_shift = None
        self._load_runs()
        self._build_specs()

        low, high = window or DEFAULT_WINDOW
        self.window = (low, DEFAULT_WINDOW[1] if high is None else float(high))

        self.side_electrons = {
            column: int(round(self.runs[column]["outcar"]["nelect"] or 0.0))
            for column in ("left", "right")
        }
        crystal_electrons = int(round(self.runs["mo"]["outcar"]["nelect"] or 0.0))
        self.crystal_electrons = crystal_electrons
        self.electrons = float(crystal_electrons if electrons is None else electrons)

        self.vbm = self._valence_band_maximum()
        self.last_purity_warnings: list = []

        functional = self.runs["mo"]["outcar"].get("functional") or ""
        from .vasp_io import GGA_NAMES

        # the METAGGA tag of a meta-GGA run (LAK, SCAN, R2SCAN) is printed as
        # it is; GGA_NAMES only spells out the GGA tags (PE -> PBE)
        self.functional = GGA_NAMES.get(functional, functional or "PAW")
        encut = self.runs["mo"]["outcar"].get("encut")
        self.method_chip = (f"VASP PAW/{self.functional}"
                            + (f", ENCUT {encut:g} eV" if encut else ""))
        self.basis_chip = "(PAW spheres, LORBIT=12 projections)"
        self.embedding_chip = self._embedding_chip()
        self.ao_foot = ""
        self.bond_foot = (
            " Crystal-orbital line colors: "
            "<span style=\"color:#1565c0\">bonding</span> / "
            "<span style=\"color:#333\">nonbonding</span> / "
            "<span style=\"color:#d32f2f\">antibonding</span>.  A plane-wave "
            "calculation has no overlap matrix, so the COOP population of the "
            "other engines does not exist here and the character is read from "
            "the ALIGNED energies instead: a crystal level carrying "
            f"{100 * NONBONDING_SHARE:.0f}% or more of its projected weight on "
            "ONE sublattice has no partner and is nonbonding; otherwise it is "
            "bonding when it lies below, and antibonding when it lies above, "
            "the composition-weighted energy of the fragment levels it is made "
            f"of (deadband {BOND_ENERGY_TOL:g} eV).  MEASURED_RESIDUAL"
            "Click a level for its fragment shares, "
            "its connector weights and the reason for its character.  A level "
            "whose parents sit on an ELEMENT whose own anchors disagree by "
            f"more than {ELEMENT_SCALE_TOL:g} eV keeps the NEUTRAL grey "
            "stroke: one shift does not describe that site, so the sign of "
            "E &minus; E<sub>parents</sub> carries no chemistry there.  A "
            "level is also left grey when less than "
            f"{100 * PARENT_COVERAGE_FLOOR:.0f}% of its projected character "
            "has a fragment level on the page, or when the fragment levels "
            "the window hides would change its character: the colour must "
            "come from the level itself, not from what happens to be drawn.  "
            "With "
            "BARE point charges on the removed sublattice that happens on the "
            "site whose shells disagree most -- a sigma-smeared point charge "
            "does not reproduce the screening the real anion valence shell "
            "gives a cation site.  --vasp-align rigid falls back to one shift "
            "per column, --vasp-anchor EL nl re-anchors a column on a chosen "
            "shell, --no-align shows the three raw scales."
        )
        self.foot_intro = (
            "Crystal-orbital diagram (COD, --vasp): every column is a finished "
            "plane-wave DFT calculation.  The fragment columns are the two "
            "sublattices computed on their own, with the removed sublattice "
            "replaced by classical Va point charges (no electrons, no "
            "projectors), so their Bloch states are the electronic states "
            "before chemical bond formation; states sharing an irrep of the "
            "little group at k mix into bonding/antibonding crystal orbitals "
            "(center).  Energies: "
            f"VASP {self.functional} eigenvalues at the tabulated special "
            "k points.  Each run pins its own G = 0 average potential to "
            "zero, so the CRYSTAL column is the reference (shift 0) and each "
            "fragment column is brought onto it by a SITE-RESOLVED shift: one "
            "constant per (column, element), fitted to the symmetry-forbidden "
            "probes (fragment levels whose irrep has no partner in the other "
            "sublattice, which must therefore coincide with their crystal "
            "counterpart) and to the pure XPS-style counterparts.  Shells of "
            "one atom move together, different SITES do not -- a cation that "
            "receives covalent back-donation in the crystal has all its levels "
            "raised by the intra-atomic Coulomb repulsion the bare-point-charge "
            "fragment lacks.  Irreps, characters and compositions come "
            "from the complex LORBIT = 12 spherical-harmonic projections "
            "inside the PAW spheres, which do not sum to 1: the percentages "
            "are shares of a level's TOTAL projected weight, and bands whose "
            "projected weight is below "
            f"{PROJECTION_FLOOR:g} (free-electron-like states between the "
            "atoms, and states piling up on the point-charge sites) are not "
            "drawn.  NOTE the parent&rarr;crystal offsets also carry the "
            "point-charge-model-vs-crystal environment difference."
        )
        self.axis_title = {"vbm": "E - E_VBM (eV)",
                           "efermi": "E - E_F (eV)"}.get(self.zero_mode,
                                                         "E (eV)")
        self.last_coupling = []

    # ------------------------------------------------------------- run set-up

    def _resolve_anchor_override(self) -> None:
        """Turn ``--vasp-anchor EL nl`` into ``{column: (element, shell)}``.

        The column is the one the element belongs to, so ``--vasp-anchor Ti
        3d`` pins the cation column on the Ti 3d manifold instead of letting
        the deepest-inert-level rule pick the Sr 4s semicore.
        """
        for text in self.anchor_request:
            parts = text.replace("=", " ").split()
            if len(parts) != 2:
                raise SystemExit(
                    f"ERROR: --vasp-anchor takes 'ELEMENT SHELL' pairs "
                    f"(e.g. 'Ti 3d'), not {text!r}.")
            element, shell = parts[0], parts[1]
            column = self.element_column.get(element)
            if column not in ("left", "right"):
                raise SystemExit(
                    f"ERROR: --vasp-anchor names {element}, which is not an "
                    "element of the --co-left / --co-right sublattices.")
            self.anchor_override[column] = (element, shell)

    def _embedding_chip(self) -> str:
        blocks = []
        for column in ("left", "right"):
            for entry in self.runs[column]["outcar"].get("va", []):
                blocks.append(entry)
        if not blocks:
            return "point charges: (none found in OUTCAR)"
        names = " ".join(dict.fromkeys(entry["name"] for entry in blocks))
        sigma = blocks[0]["sigma"]
        rwall = blocks[0]["rwall"]
        return (f"Va point charges {names} (sigma {sigma:g} A, "
                f"wall width {rwall:g} A)")

    def _load_runs(self) -> None:
        from .vasp_io import lm_block_layout, read_outcar, read_poscar, read_procar

        self.kpoint_names = self.special_kpoints()
        self.runs: dict = {}
        for column in ("mo", "left", "right"):
            directory = self.directories[column]
            structure = read_poscar(os.path.join(directory, "POSCAR"))
            outcar = read_outcar(os.path.join(directory, "OUTCAR"))
            if outcar.get("ispin") not in (None, 1):
                raise SystemExit(
                    f"ERROR: {directory} is spin polarized (ISPIN = "
                    f"{outcar['ispin']}); the --vasp crystal-orbital engine "
                    "handles non-spin-polarized runs only.")
            if outcar.get("noncollinear"):
                # a spinor PROCAR carries one Kramers partner per band and
                # magnetization blocks the projection reader does not split:
                # every degeneracy and electron count would come out wrong
                raise SystemExit(
                    f"ERROR: {directory} is a spin-orbit (non-collinear, vasp_ncl) "
                    "run; the anchor engine reads scalar runs only.  The overlap "
                    "engine reads spinor runs: add --vasp-engine overlap.")
            mapping = self._match_atoms(structure, directory)
            self.mappings[column] = mapping
            ions = list(mapping.ions)
            kpoints = [self._to_run_basis(structure, kpoint)
                       for _, kpoint in self.kpoint_names]
            procar = read_procar(os.path.join(directory, "PROCAR"), kpoints=kpoints)
            self.runs[column] = {"dir": directory, "structure": structure,
                                 "outcar": outcar, "procar": procar, "ions": ions,
                                 "mapping": mapping}
        layout = lm_block_layout(self.runs["mo"]["procar"].lm)
        for column in ("left", "right"):
            if lm_block_layout(self.runs[column]["procar"].lm) != layout:
                raise SystemExit(
                    f"ERROR: {self.directories[column]} and "
                    f"{self.directories['mo']} have different PROCAR lm "
                    "channels; the three runs must use the same LORBIT "
                    "setting and comparable POTCARs.")
        self.lm_layout = layout
        self.lm_names = list(self.runs["mo"]["procar"].lm)
        self.n_lm = sum(2 * l + 1 for l, _ in layout)
        self.n_ao = len(self.symbols) * self.n_lm
        # the real (non-Va) atoms of each fragment run must be exactly that
        # fragment's sublattice
        for column in ("left", "right"):
            structure = self.runs[column]["structure"]
            real = {self.runs[column]["ions"].index(ion)
                    for ion in range(len(structure.symbols))
                    if structure.charges[ion] is None}
            expected = set(self.side_atoms[column])
            if real != expected:
                got = "".join(sorted(structure.real_elements))
                raise SystemExit(
                    f"ERROR: the real atoms of {self.directories[column]} "
                    f"({got}) are not the {column} sublattice "
                    f"{self.formula[column]}; check --vasp-left/--vasp-right.")

    def _match_atoms(self, structure, directory):
        """The :class:`~.vasp_io.RunMapping` of one run onto the ``-c`` setting.

        A run may sit on ANOTHER ORIGIN than the ``-c`` file (SrTiO3 written
        with Ti at the origin against a ``-c`` file with Sr there): one global
        shift, a permutation of the atoms and lattice-vector wraps are
        accepted and transported; anything else is a
        :class:`~.vasp_io.StructureMappingError` the caller turns into the
        fallback or into the one-line ERROR.
        """
        from .vasp_io import StructureMappingError, map_run_onto_cell

        try:
            mapping = map_run_onto_cell(
                self.lattice, self.positions, self.symbols, structure,
                prefer=self.preferred_shift)
        except StructureMappingError as error:
            raise StructureMappingError(
                f"{os.path.join(directory, 'POSCAR')} is not the -c cell in "
                f"another setting: {error.reason}", error.kind) from None
        if self.preferred_shift is None:
            self.preferred_shift = mapping.shift
        return mapping

    def setting_lines(self) -> list:
        """One line per run whose setting is not the ``-c`` setting."""
        lines = []
        for column in ("mo", "left", "right"):
            mapping = self.mappings.get(column)
            if mapping is None:
                continue
            run = self.runs[column]
            text = mapping.describe(
                self.symbols, run["structure"].symbols,
                os.path.join(run["dir"], "POSCAR"))
            if text:
                lines.append(text)
        return lines

    def _to_run_basis(self, structure, kpoint):
        """A k point of the analysis cell in the reciprocal basis of a run.

        The mapping has already established that the run's lattice matrix IS
        the analysis one (within ``LATTICE_MATCH_RTOL``) -- only the origin
        may differ, and an origin shift leaves the reciprocal basis alone.  So
        the fractional k point carries over unchanged, and it does so exactly:
        converting through Cartesian space instead would smear the tabulated
        0.5 by the relative lattice-constant difference of the two files
        (4e-6 for the SrTiO3 example) and, at the ``LATTICE_MATCH_RTOL``
        limit, past the tolerance ``read_procar`` matches k points with.
        """
        del structure
        return np.asarray(kpoint, dtype=float)

    # --------------------------------------------------------- the AO-like basis

    def _valence_shells(self, column) -> dict:
        """``{element: {l: [n, ...]}}`` from the run's POTCAR valence data."""
        from .mo_diagram import EHT_PARAMETERS

        shells: dict = {}
        by_element = {entry.element: entry
                      for entry in self.runs[column]["outcar"]["species"]}
        elements = dict.fromkeys(self.symbols[i] for i in self.side_atoms[column])
        for element in elements:
            found = {}
            entry = by_element.get(element)
            if entry is not None:
                found = entry.valence_shells()
            if not found:
                found = {}
                for shell, n, l, _zeta, _h in EHT_PARAMETERS.get(element, []):
                    found.setdefault(l, []).append(n)
                    del shell
            highest = max((max(values) for values in found.values()), default=0)
            for l, _column in self.lm_layout:
                if l not in found:
                    # aufbau: the d shell that goes with an n s valence shell
                    # is (n-1)d, the f shell (n-2)f -- so Sr, whose valence s
                    # is 5s, gets 4d rather than 5d.  s and p keep the
                    # element's own n.
                    found[l] = [max(l + 1, highest - max(l - 1, 0))]
            shells[element] = {l: sorted(values) for l, values in found.items()}
        return shells

    def _build_specs(self) -> None:
        self.valence_shells = {}
        for column in ("left", "right"):
            self.valence_shells.update(self._valence_shells(column))
        self.specs: list = []
        self.side_specs = {"left": [], "right": []}
        for column in ("left", "right"):
            for element in dict.fromkeys(self.symbols[i]
                                         for i in self.side_atoms[column]):
                sites = [i for i in self.side_atoms[column]
                         if self.symbols[i] == element]
                for l, _offset in self.lm_layout:
                    for n in self.valence_shells[element][l]:
                        spec = VaspSpec(element=element, shell=f"{n}{'spdf'[l]}",
                                        l=l, sites=sites, column=column)
                        self.specs.append(spec)
                        self.side_specs[column].append(spec)

    def _rows(self, atoms, l=None):
        """Row indices of the ``(atom, l, m)`` basis for the given atoms."""
        rows = []
        for atom in atoms:
            for shell_l, offset in self.lm_layout:
                if l is not None and shell_l != l:
                    continue
                base = atom * self.n_lm + offset
                rows.extend(range(base, base + 2 * shell_l + 1))
        return np.array(rows, dtype=int)

    # ------------------------------------------------------------- the levels

    def _projections(self, column, kpoint):
        """``(vectors, energies, occupations)`` of one run at one k point.

        ``vectors`` has shape ``(n_ao, n_bands)``: the PROCAR projections
        embedded into the shared ``(atom, l, m)`` basis of the primitive cell
        and permuted from VASP's ``m = -l..l`` order into crystod's.
        """
        from .vasp_io import reorder_to_crystod

        run = self.runs[column]
        procar = run["procar"]
        target = self._to_run_basis(run["structure"], kpoint)
        index = None
        for candidate in sorted(procar.loaded):
            difference = procar.kpoints[candidate] - target
            if np.all(np.abs(difference - np.rint(difference)) < 1e-5):
                index = candidate
                break
        if index is None:
            raise SystemExit(
                f"ERROR: {run['dir']} has no PROCAR entry for the special "
                f"k point {np.round(kpoint, 6).tolist()} (in the run's basis "
                f"{np.round(target, 6).tolist()}); add it as a zero-weight "
                "line to KPOINTS, or point --vasp-crystal/--vasp-left/"
                "--vasp-right at the directory that has it.")
        phases = procar.phases[index]                       # (nb, n_ion, n_lm)
        n_bands = phases.shape[0]
        vectors = np.zeros((self.n_ao, n_bands), dtype=complex)
        # the transport of the projection vector from the run's setting into
        # the -c setting: reorder the atoms and apply the mapping's per-atom
        # Bloch factor (see RunMapping.phases -- it is 1 in VASP's periodic
        # gauge, which is the gauge the PROCAR is in)
        bloch = run["mapping"].phases(target)
        for atom, ion in enumerate(run["ions"]):
            for l, offset in self.lm_layout:
                size = 2 * l + 1
                block = phases[:, ion, offset:offset + size]     # (nb, size)
                base = atom * self.n_lm + offset
                vectors[base:base + size, :] = (
                    bloch[atom] * (reorder_to_crystod(l) @ block.T))
        return vectors, procar.energies[index], procar.occupations[index]

    def little_group_data(self, kpoint):
        """Irreps, labels and the ``(atom, l, m)`` representation at ``k``.

        Same construction as the extended-Hueckel and PySCF engines:
        ``D[(a', l, m'), (a, l, m)] = P[a', a] W^l[m', m]`` with crystod's
        Bloch-phased site permutation ``P`` and the real Wigner matrices, and
        NO further gauge conjugation: the PROCAR coefficient vector is already
        in the same Bloch gauge as ``P``.

        The engine used to conjugate this with
        ``Lambda_a = exp(2 pi i k . tau_a)``, by analogy with the PySCF engine,
        whose AO coefficients really are in the atomic gauge.  That analogy is
        wrong for VASP, and it is a no-op for every site on half-integer
        coordinates, which is why SrTiO3 and ScF3 could not see it.  Fluorite
        CaF2, whose F sites sit on quarter coordinates of the primitive basis,
        does: with the conjugation 21 of 85 labelled levels fall below irrep
        purity 0.90 (minimum 0.000 at W, 12 failures at L), the fragment
        manifolds no longer reproduce the SALC decomposition, both sites fail
        the per-element scale test and the page carries no bonding and no
        antibonding level at all.  Without it every one of the 80 labelled
        levels has purity 1.000 at GM, L, X and W, the manifolds match the
        group-theory decomposition exactly, and the three half-integer-site
        systems reproduce their previous numbers to the last printed digit.
        See section 2 of ``crystod_dev/reports/crystod_vasp_engine.md``.
        """
        from .runtime_compat import get_spacegroup_irreps_from_primitive_symmetry

        irreps, mapping = get_spacegroup_irreps_from_primitive_symmetry(
            rotations=self.builder.rotations,
            translations=self.builder.translations,
            kpoint=kpoint,
        )
        labels = self.builder.get_irrep_labels(kpoint, irreps, mapping)
        permutations = self.builder.get_permutation_reps_at_k(
            little_rotations=self.builder.rotations[mapping],
            little_translations=self.builder.translations[mapping],
            kpoint=kpoint,
        )
        representation = []
        for op_index, op in enumerate(mapping):
            rotation = np.real(self.builder.rotations_cartesian[op])
            wigners = {l: wigner_D_real(l, rotation) for l, _ in self.lm_layout}
            matrix = np.zeros((self.n_ao, self.n_ao), dtype=complex)
            for source in range(len(self.symbols)):
                for image in range(len(self.symbols)):
                    phase = permutations[op_index][image, source]
                    if abs(phase) < 1e-12:
                        continue
                    for l, offset in self.lm_layout:
                        size = 2 * l + 1
                        row = image * self.n_lm + offset
                        column = source * self.n_lm + offset
                        matrix[row:row + size, column:column + size] = (
                            phase * wigners[l])
            representation.append(matrix)
        return irreps, mapping, labels, representation

    def site_symmetry_irreps(self, kpoint, representation, irreps, labels):
        """Irrep content of every ``(element, l)`` block at ``k``.

        The pure group-theory SALC decomposition ``crystod --element EL
        --orbital ORB`` prints, recomputed from the representation used for
        the labelling -- the validation target of the PROCAR classification.
        """
        order = len(representation)
        content: dict = {}
        for spec in self.specs:
            key = (spec.element, spec.shell)
            if key in content:
                continue
            indices = self._rows(spec.sites, spec.l)
            characters = np.array([np.trace(D[np.ix_(indices, indices)])
                                   for D in representation])
            parts = []
            for irrep, label in zip(irreps, labels):
                chi = np.array(get_character(irrep), dtype=complex)
                count = int(round(float(np.real(
                    np.sum(characters * np.conj(chi)) / order))))
                if count > 0:
                    name = label.split("(")[0]
                    parts.append(name if count == 1 else f"{count}{name}")
            content[key] = parts
        return content

    def _adaptive_groups(self, energies, vectors, representation, irreps):
        """Cluster bands into complete multiplets (the PySCF engine's rule)."""
        seeds = []
        start = 0
        for index in range(1, len(energies) + 1):
            if (index == len(energies)
                    or energies[index] - energies[start] > self.degeneracy_tol):
                seeds.append((float(np.mean(energies[start:index])),
                              vectors[:, start:index], start, index))
                start = index
        groups = []
        index = 0
        while index < len(seeds):
            start_energy = seeds[index][0]
            space = seeds[index][1]
            bands = [seeds[index][2], seeds[index][3]]
            weights = [space.shape[1]]
            centres = [seeds[index][0]]
            last = index
            while not self._complete(space, representation, irreps):
                if (last + 1 >= len(seeds)
                        or seeds[last + 1][0] - start_energy > self.degeneracy_window):
                    break
                last += 1
                space = np.hstack([space, seeds[last][1]])
                weights.append(seeds[last][1].shape[1])
                centres.append(seeds[last][0])
                bands[1] = seeds[last][3]
            groups.append((float(np.average(centres, weights=weights)), space,
                           bands[0], bands[1]))
            index = last + 1
        return groups

    def _complete(self, space, representation, irreps) -> bool:
        values, orthonormal = _multiplicities(space, representation, irreps)
        if orthonormal.shape[1] == 0:
            return True
        if not any(value > 0.5 for value in values):
            return False
        return all(abs(value - round(value)) < _MULTIPLICITY_TOL for value in values)

    def _split_by_irrep(self, space, representation, irreps, labels):
        """``[(label, subspace, purity), ...]`` of one degenerate group."""
        values, orthonormal = _multiplicities(space, representation, irreps)
        if orthonormal.shape[1] == 0:
            return []
        order = len(representation)
        rank = orthonormal.shape[1]
        components = []
        for irrep, label, multiplicity in zip(irreps, labels, values):
            count = int(round(multiplicity))
            if count <= 0:
                continue
            dimension = int(np.asarray(irrep).shape[1])
            chi = np.array(get_character(irrep), dtype=complex)
            projector = np.zeros((self.n_ao, self.n_ao), dtype=complex)
            for index, D in enumerate(representation):
                projector = projector + np.conj(chi[index]) * D
            projector *= dimension / order
            subspace = _canonical(projector @ orthonormal)
            if subspace.shape[1] > count * dimension:
                subspace = subspace[:, :count * dimension]
            if subspace.shape[1] == 0:
                continue
            purity = float(multiplicity * dimension / rank)
            components.append((label.split("(")[0], subspace, purity))
        if not components:
            components = [("?", orthonormal, 0.0)]
        return components

    def _shares(self, space):
        """``[(weight, element, l), ...]`` of a level; the weights sum to 1.

        Every ``(element, l)`` channel is listed, however small, so that the
        shell namer's running count is exact; the display rows drop the ones
        below :data:`SHARE_FLOOR`.
        """
        total = float(np.sum(np.abs(space) ** 2))
        if total <= 0.0:
            return [], 0.0
        shares = []
        for element in dict.fromkeys(self.symbols):
            atoms = [i for i, symbol in enumerate(self.symbols) if symbol == element]
            for l, _offset in self.lm_layout:
                rows = self._rows(atoms, l)
                shares.append((float(np.sum(np.abs(space[rows]) ** 2)) / total,
                               element, l))
        shares.sort(key=lambda item: -item[0])
        return shares, total

    def _shell_namer(self):
        """A stateful ``(element, l, weight) -> shell`` namer for one column.

        Within a column the states of one ``(element, l)`` manifold are met in
        energy order, so the first ``n_sites (2l+1)`` of them belong to the
        deepest valence n of that l, the next block to the one above it -- the
        rule that separates a semicore Sr 4s from the Sr 5s above it.  Once the
        POTCAR's valence n of that l are used up, n keeps counting upwards
        (Ti 4p, then 5p, ...): those manifolds are the plane-wave continuum
        rather than atomic shells, and the aufbau continuation at least names
        them distinctly instead of repeating the last valence shell.
        """
        counters: dict = {}

        def name(element, l, amount):
            shells = self.valence_shells[element][l]
            sites = sum(1 for symbol in self.symbols if symbol == element)
            capacity = float(sites * (2 * l + 1))
            used = counters.get((element, l), 0.0)
            index = int(used / capacity + 1e-9)
            counters[(element, l)] = used + amount
            if index < len(shells):
                return f"{shells[index]}{'spdf'[l]}"
            return f"{shells[-1] + index - len(shells) + 1}{'spdf'[l]}"

        return name

    def solve_at(self, kpoint):
        """Levels of the three columns at one k point, from the PROCAR data.

        Args:
            kpoint: Three primitive reciprocal coordinates.

        Returns:
            ``(levels, labels)`` with the columns ``"left"``, ``"mo"`` and
            ``"right"``; energies on the raw per-run references until
            :meth:`align_fragment_columns` shifts them.  EVERY level above the
            projection floor is solved here, whatever ``--vasp-window`` says:
            the window is a view on the finished, aligned diagram and is
            applied by :meth:`apply_window`, never before the alignment has
            chosen its anchors.
        """
        irreps, _mapping, labels, representation = self.little_group_data(kpoint)
        levels = {"left": [], "mo": [], "right": []}
        self.last_purity_warnings = []
        for column in ("left", "right", "mo"):
            vectors, energies, occupations = self._projections(column, kpoint)
            atoms = (list(range(len(self.symbols))) if column == "mo"
                     else self.side_atoms[column])
            rows = self._rows(atoms)
            namer = self._shell_namer()
            counts: dict = {}
            raw_levels = []
            for energy, space, first, last in self._adaptive_groups(
                    energies, vectors, representation, irreps):
                partners = max(last - first, 1)
                weight = float(np.sum(np.abs(space[rows]) ** 2)) / partners
                self.floor_margin.append((weight, column))
                if weight < self.projection_floor:
                    continue
                electrons = int(round(float(np.sum(occupations[first:last]))))
                for name, subspace, purity in self._split_by_irrep(
                        space, representation, irreps, labels):
                    raw_levels.append((energy, name, subspace, purity, weight,
                                       electrons, partners))
            for (energy, name, subspace, purity, weight, electrons,
                 partners) in raw_levels:
                shares, total = self._shares(subspace)
                degeneracy = subspace.shape[1]
                share_electrons = int(round(electrons * degeneracy / partners))
                # one namer per column, walked in energy order, so the display
                # rows and the level label always quote the same shell
                # How much of an (element, l) channel a level uses up.  A
                # FRAGMENT level is a sublattice state of one manifold -- the
                # one its label quotes -- so it consumes a whole slot there
                # (and only its share of the channels it merely leaks into),
                # which is what makes a fragment manifold close exactly when
                # its states have been passed.  A CRYSTAL band is a mixture
                # with no manifold of its own and consumes its share of each.
                rows = [
                    (f"{element} "
                     f"{namer(element, l, degeneracy if (index == 0 and column != 'mo') else value * degeneracy)} "
                     f"{name}", value)
                    for index, (value, element, l) in enumerate(shares)]
                rows = [row for row in rows if row[1] > SHARE_FLOOR]
                if column == "mo":
                    counts[name] = counts.get(name, 0) + 1
                    label = f"{name} #{counts[name]}"
                else:
                    label = (f"{rows[0][0]}" if rows else f"? ? {name}")
                level = DiagramLevel(
                    level_id=f"{column}{len(levels[column])}",
                    column=column,
                    energy=float(energy),
                    degeneracy=degeneracy,
                    irrep=name,
                    label=label,
                    electrons=share_electrons,
                    vectors=subspace,
                )
                level.irrep_purity = purity
                level.projected_weight = weight
                level.total_weight = total
                level.raw_energy = float(energy)
                level.shares = shares
                level.base_label = label
                level.display_composition = rows
                populations = ", ".join(f"{text} {100 * value:.1f}%"
                                        for text, value in rows)
                level.detail = (
                    "PROCAR projections (shares of the level's total projected "
                    f"weight, {weight:.2f} per partner): {populations}")
                levels[column].append(level)
                if purity < IRREP_PURITY_WARN:
                    self.last_purity_warnings.append(
                        (column, label, float(energy), purity))
            if column != "mo":
                _suffix_duplicate_labels(levels[column])
        self._assign_compositions(levels)
        return levels, labels

    def _side_weights(self, level):
        """``{column: normalized projected weight}`` of a crystal level."""
        total = float(np.sum(np.abs(level.vectors) ** 2))
        weights = {}
        for column in ("left", "right"):
            rows = self._rows(self.side_atoms[column])
            weights[column] = (float(np.sum(np.abs(level.vectors[rows]) ** 2)) / total
                               if total > 0 else 0.0)
        norm = sum(weights.values()) or 1.0
        return {column: value / norm for column, value in weights.items()}

    def _assign_compositions(self, levels) -> None:
        """Connector weights of every crystal level (see the module docstring).

        The share of a crystal level on a fragment is its normalized projected
        weight on that fragment's atoms; inside the fragment it is distributed
        over the fragment levels of the SAME irrep in proportion to the
        gauge-invariant projection overlap ``sum_ij |<c_F,j|c_L,i>|^2`` of the
        normalized projection vectors restricted to the fragment's atoms.
        ``absolute_composition`` keeps the ABSOLUTE version of that overlap --
        the fraction of the crystal level's own projected weight that the
        fragment level accounts for, so a deep semicore band that reappears
        unchanged comes out at ~1 and a level with a few percent of that
        character cannot masquerade as a pure counterpart.  That is what the
        alignment anchors are selected on.

        **Channel attribution.**  The Loewdin weights alone are radially blind:
        the PROCAR has one channel per ``(atom, l)``, so the Sr 4s semicore
        band of SrTiO3 keeps a 17 % residual onto the Sr 5s / Ti 4s fragment
        level 35 eV above it although it carries 0.0 % Ti s character.  The
        connector weights are therefore built CHANNEL BY CHANNEL: the share
        the crystal level carries in one ``(element, n l)`` channel is handed
        to the fragment levels whose own manifold that is, split among them by
        their Loewdin weights.  A connector into a channel the crystal level
        has no character in is exactly zero, and a channel whose fragment
        levels are not drawn (outside the energy window, or below the
        projection floor) simply loses its weight instead of donating it to an
        unrelated shell -- distributing the whole side weight over whatever
        survived used to hand the Ti 4p parentage of the O 2p valence band to
        the Sr 4p semicore 12 eV below it as soon as the Ti levels left the
        window.  The connector weights therefore need NOT sum to 1; what is
        missing is the parentage of levels the page does not show.

        The key is ``(element, n, l)`` and not ``(element, l)``: keyed on the
        l channel alone the hand-off is still right ACROSS l but wrong across
        n within one l, and SrTiO3 ``GM1+ #3`` (51.3 % Sr 5s, 43.0 % O 2s) got
        its whole Sr s share handed to the Sr 4s semicore level 39 eV below
        it, for a parent energy of -26.2 eV on a level at +6.2 eV.  The shell
        namer already resolves n (see :func:`_level_shell`), so the share rows
        the page prints and the hand-off now use one and the same label.
        """
        groups: dict = {}
        for column in ("left", "right"):
            for fragment in levels[column]:
                groups.setdefault((column, fragment.irrep), []).append(fragment)
        for crystal in levels["mo"]:
            crystal.side_weights = self._side_weights(crystal)
            norm = float(np.sum(np.abs(crystal.vectors) ** 2)) or 1.0
            channel = _share_channels(crystal)
            composition = []
            absolute = []
            for column in ("left", "right"):
                rows = self._rows(self.side_atoms[column])
                block = crystal.vectors[rows]
                fragments = groups.get((column, crystal.irrep), [])
                weights = self._lowdin_weights(block, fragments, rows)
                by_channel: dict = {}
                for fragment, value in zip(fragments, weights):
                    # the RAW (double-counting) projection serves the alignment
                    # anchors, exactly as in the PySCF engine: it says how much
                    # of the crystal level lies along THIS fragment state, and
                    # a Loewdin weight -- which mixes the non-orthogonal
                    # fragment states -- would understate a deep band that
                    # reappears unchanged (Sr 4s of SrTiO3: 0.63 instead of 1.0)
                    other = _canonical(fragment.vectors[rows])
                    if other.shape[1]:
                        raw = float(np.sum(np.abs(other.conj().T @ block) ** 2)) / norm
                        if raw > 1e-6:
                            absolute.append((fragment.level_id, raw))
                    by_channel.setdefault(
                        _level_shell(fragment), []).append(
                            (fragment, max(float(value), 0.0)))
                for key, items in by_channel.items():
                    available = channel.get(key, 0.0)
                    total = sum(value for _, value in items)
                    if available <= 0.0 or total <= 0.0:
                        continue
                    for fragment, value in items:
                        if value > 0.0:
                            composition.append((fragment.level_id,
                                                available * value / total))
            crystal.composition = [(level_id, value)
                                   for level_id, value in composition
                                   if value >= CONNECTOR_FLOOR]
            crystal.absolute_composition = absolute
            crystal.composition_completeness = sum(
                value for _, value in absolute)

    def _lowdin_weights(self, block, fragments, rows):
        """Projection of a crystal level onto Loewdin-orthogonalized fragments.

        The PROCAR basis has ONE radial channel per ``(atom, l)``, so two
        fragment levels of the same shell letter -- a Sr 4s semicore band and
        the Sr 5s band above it -- are far from orthogonal in it, and plain
        projections would give the deep band a large connector into the shallow
        one.  Symmetric (Loewdin) orthogonalization of the fragment-level set,
        exactly what ``assign_fragment_compositions`` does for the AO engines,
        removes that double counting.
        """
        if not fragments or block.shape[1] == 0:
            return [0.0] * len(fragments)
        basis = np.hstack([fragment.vectors[rows] for fragment in fragments])
        gram = basis.conj().T @ basis
        values, vectors = np.linalg.eigh(gram)
        scale = float(values.max()) if values.size else 0.0
        keep = values > 1e-8 * max(scale, 1e-30)
        if not np.any(keep):
            return [0.0] * len(fragments)
        inverse_half = ((vectors[:, keep] / np.sqrt(values[keep]))
                        @ vectors[:, keep].conj().T)
        tilde = inverse_half @ (basis.conj().T @ block)
        weights = []
        start = 0
        for fragment in fragments:
            count = fragment.vectors.shape[1]
            weights.append(float(np.sum(np.abs(tilde[start:start + count]) ** 2)))
            start += count
        return weights

    # ------------------------------------------------------------- alignment

    def zero_offset(self) -> float:
        """The constant that puts the page's energy zero where ``--vasp-zero`` says.

        ``vbm`` (the default) reports ``E - E_VBM`` with the valence-band
        maximum taken as the highest OCCUPIED eigenvalue of the crystal run
        over all its k points (not only the special ones); ``efermi`` uses the
        crystal run's Fermi level; ``raw`` leaves the crystal run's own G = 0
        reference alone.
        """
        if self.zero_mode == "raw":
            return 0.0
        if self.zero_mode == "efermi":
            return -float(self.runs["mo"]["outcar"].get("efermi") or 0.0)
        return -float(self.vbm)

    def rigid_alignment(self, records):
        """One shift per fragment COLUMN, referenced to the crystal column.

        The XPS-style anchor rule of the PySCF engine (reused verbatim: it
        needs nothing but ``absolute_composition``, which
        :meth:`_assign_compositions` fills), re-referenced afterwards so that
        the CRYSTAL column keeps shift 0 and each fragment column moves by its
        own ``delta = E_crystal - E_fragment``.  ``--vasp-anchor EL nl`` is
        honoured through the shared ``anchor_override`` hook.

        The shared rule pairs every fragment level with its inert crystal
        counterpart one-to-one and in energy order
        (``crystal_orbital_pyscf._alignment_counterparts``).  That is what
        the radially blind PROCAR basis needs: without it the Sr 4s R2-
        semicore level of a METAGGA = LAK SrTiO3 run was paired with the
        empty Sr 5s-like level R2- #2, 46 eV above its own band, because both
        overlaps are 1.000 and the empty one came out larger by 4e-16 (the
        PBE run has the same tie and kept the right level only because its
        two overlaps are bit-identical).  The result is not only the
        ``--vasp-align rigid`` shift: it
        places the fragment levels against the anchor-window cut in the FIRST
        pass of :meth:`site_anchor_report`, and it is the shift a column falls
        back to when none of its elements finds an anchor, so the default
        site-resolved alignment depends on it wherever that cut or that
        fallback decides (``--kpoint R`` on that run lost every Sr and Ti
        anchor to it, moved the whole cation column up by the +50.0 eV and
        so left it off the page).

        Returns:
            ``(deltas, anchors)``: ``deltas`` maps ``"left"``/``"right"`` to
            the fragment shift in eV, ``anchors`` is the shared routine's
            per-column anchor record.  ``(None, anchors)`` when no chemically
            inert fragment level exists.  Nothing is applied here.
        """
        from .crystal_orbital_pyscf import PySCFCrystalOrbitalDiagram

        shifts, anchors = PySCFCrystalOrbitalDiagram.align_fragment_columns(
            self, records)
        if shifts is None:
            return None, anchors
        # the shared routine SHIFTS the levels and anchors on a fragment; undo
        # both, so that the crystal column is the reference and every fragment
        # level carries its own raw energy plus its column delta
        for record in records:
            for column in ("left", "right", "mo"):
                for level in record["levels"][column]:
                    level.energy = level.raw_energy
        reference = shifts["reference"]
        delta_ref = -shifts["mo"]
        deltas = {reference: delta_ref,
                  {"left": "right", "right": "left"}[reference]:
                      shifts[{"left": "right", "right": "left"}[reference]]
                      + delta_ref}
        return deltas, anchors

    def site_anchors(self, records, provisional=None):
        """Per-``(column, element)`` anchors of the site-resolved alignment.

        See the module-level notes on :data:`ANCHOR_WEIGHT_FORBIDDEN`.  For
        every fragment level inside the anchor window (:data:`ANCHOR_WINDOW_EV`
        above the crystal VBM, the fragment column placed by the rigid shift)
        whose dominant element carries at least :data:`ANCHOR_DOMINANT_SHARE`
        of its projected weight, the crystal level of the SAME irrep with the
        largest overlap is looked up and the pair is kept as an anchor when

        * the fragment level's irrep does not occur among the other
          sublattice's projected manifolds at that k (a symmetry-forbidden
          probe: it cannot mix, so it must coincide), or
        * that overlap reaches :data:`COLUMN_SCALE_PURITY` (a pure
          counterpart).

        The anchor list is window-independent in the sense that matters: it is
        cut at the DEFAULT window, never at ``--vasp-window``.

        The pool is cut PER ELEMENT, with that element's own provisional
        shift.  Cutting it with one rigid shift per column is wrong as soon as
        the two sites of a column are far apart, and it fails silently: in
        ``example/SrTiO3_Pm-3m_Tisv`` the rigid pre-shift re-anchors from
        Sr 4s (+4.4 eV) to Ti 3s (+14.4 eV), which lifts the empty diffuse
        Sr 4d fragment manifold above the cut.  Sr then keeps 8 of its 13
        anchors, every 4d anchor among the five it loses, and the engine
        reports "shell spread 0.20 eV -> shells agree" for a site whose true
        4s-to-4d spread, measurable in the same run's own level table, is
        1.56 eV.  :meth:`site_anchor_report` therefore calls this twice: once
        with the rigid shift to get a provisional per-element delta, then
        again with those deltas.

        Args:
            records: The per-k-point records, with RAW energies.
            provisional: ``{column: {element: shift}}`` used to place each
                fragment level against the cut.  ``None`` (the first pass)
                falls back to the rigid column shift of
                :attr:`rigid_shifts` for every element.

        Returns:
            ``{column: [(delta, forbidden, purity, label, k_name,
            crystal_label, element, shell), ...]}``.  Also fills
            :attr:`anchor_pool_shells` with the ``(element, shell)`` blocks
            that reached the pool, whether or not they yielded an anchor.
        """
        top = self.vbm + ANCHOR_WINDOW_EV
        found: dict = {"left": [], "right": []}
        pool_shells: dict = {"left": {}, "right": {}}
        for column in ("left", "right"):
            other = {"left": "right", "right": "left"}[column]
            rigid = float(self.rigid_shifts.get(column, 0.0))
            guesses = (provisional or {}).get(column, {})

            def placed(level, guesses=guesses, rigid=rigid):
                element, share = _dominant_element(level)
                if element is not None and share >= ANCHOR_DOMINANT_SHARE:
                    return level.raw_energy + float(guesses.get(element, rigid))
                return level.raw_energy + rigid

            for record in records:
                fragments = {
                    level.level_id: level
                    for level in record["levels"][column]
                    if placed(level) <= top
                }
                for level in fragments.values():
                    element, share = _dominant_element(level)
                    if element is None or share < ANCHOR_DOMINANT_SHARE:
                        continue
                    pool_shells[column].setdefault(element, set()).add(
                        " ".join(level.label.split()[:2]))
                partner_irreps = {level.irrep
                                  for level in record["levels"][other]}
                best: dict = {}
                for crystal in record["levels"]["mo"]:
                    if crystal.raw_energy > top:
                        continue
                    for level_id, weight in getattr(
                            crystal, "absolute_composition", []):
                        if level_id not in fragments:
                            continue
                        if weight > best.get(level_id, (0.0, None))[0]:
                            best[level_id] = (weight, crystal)
                # One crystal band cannot be the counterpart of TWO orthogonal
                # fragment states of the same irrep at the same k: the PROCAR
                # basis is radially blind, so a high-lying "Sc 4s" fragment
                # level projects onto the Sc 3s semicore BAND with weight
                # 0.999, exactly like the Sc 3s fragment level itself, and
                # contributes a -50 eV anchor.  The default-window cut was the
                # only guard, and it is not one: it moves with the element's
                # own provisional shift.  Keep the claimant with the larger
                # overlap and, at equal overlap, the one whose delta is the
                # smaller -- the artificial pair is tens of eV off, never a
                # few.
                claims: dict = {}
                for level_id, (weight, crystal) in best.items():
                    claims.setdefault(crystal.level_id, []).append(
                        (level_id, weight, crystal))
                kept: dict = {}
                for claimants in claims.values():
                    winner = min(
                        claimants,
                        key=lambda item: (-round(item[1], 2),
                                          abs(item[2].raw_energy
                                              - fragments[item[0]].raw_energy)))
                    kept[winner[0]] = (winner[1], winner[2])
                for level_id, (weight, crystal) in kept.items():
                    fragment = fragments[level_id]
                    element, share = _dominant_element(fragment)
                    if element is None or share < ANCHOR_DOMINANT_SHARE:
                        continue
                    forbidden = fragment.irrep not in partner_irreps
                    if not forbidden and weight < COLUMN_SCALE_PURITY:
                        continue
                    found[column].append((
                        float(crystal.raw_energy - fragment.raw_energy),
                        forbidden, float(weight), fragment.label,
                        record["name"], crystal.label, element,
                        " ".join(fragment.label.split()[:2])))
        self.anchor_pool_shells = pool_shells
        return found

    def site_anchor_report(self, records):
        """Fit one shift per ``(column, element)`` and judge each element.

        The fit is the EXACT constraints alone wherever they exist: an element
        with at least one symmetry-forbidden probe takes the purity-weighted
        mean of its probes (:func:`_fit_site_delta`), and the residuals of its
        pure counterparts are reported as the measured bonding shift of each
        pair instead of being folded into the scale.  An element with no probe
        keeps the combined purity- and :data:`ANCHOR_WEIGHT_FORBIDDEN`-weighted
        mean and is flagged with the zero-sum caveat.

        The element passes the scale test when the spread of its
        per-SHELL mean deltas stays inside :data:`ELEMENT_SCALE_TOL`: the
        premise of the whole construction is that the shells of one atom move
        together, so that spread is exactly the premise's error bar.

        An element of the column that has no anchor at all inherits the
        column's weighted mean and is flagged.

        The fit is made TWICE.  The first pass places the fragment levels
        against the anchor-window cut with the rigid column shift, which is
        the only shift available before any element has a delta; the second
        pass repeats it with each element's own provisional delta, so that a
        column whose two sites are far apart cannot lose a whole shell of one
        site to a cut made for the other (see :meth:`site_anchors`).  Any
        ``(element, shell)`` block that reached the pool but contributed no
        anchor is recorded in ``"shells_without_anchor"`` and warned about:
        the per-shell spread is the scale test, and a shell that is drawn but
        never measured cannot enter it.

        Returns:
            ``{column: {"elements": {element: {...}}, "mean": float,
            "missing": [element, ...]}}``.  Also fills
            :attr:`element_scale_ok`.
        """
        anchors = self.site_anchors(records)
        first: dict = {}
        for column in ("left", "right"):
            grouped: dict = {}
            for entry in anchors[column]:
                grouped.setdefault(entry[6], []).append(entry)
            first[column] = {element: _fit_site_delta(entries)[0]
                             for element, entries in grouped.items()}
        anchors = self.site_anchors(records, provisional=first)
        pool_shells = getattr(self, "anchor_pool_shells", {})
        report: dict = {}
        for column in ("left", "right"):
            rows = anchors[column]
            weights = [entry[2] * (ANCHOR_WEIGHT_FORBIDDEN if entry[1] else 1.0)
                       for entry in rows]
            column_mean = (_weighted_mean([entry[0] for entry in rows], weights)
                           if rows else float(self.rigid_shifts.get(column, 0.0)))
            elements: dict = {}
            for entry in rows:
                elements.setdefault(entry[6], []).append(entry)
            resolved: dict = {}
            for element, entries in elements.items():
                delta, fit = _fit_site_delta(entries)
                shells: dict = {}
                for entry in entries:
                    shells.setdefault(entry[7], []).append(entry)
                shell_rows = []
                for shell, group in sorted(shells.items()):
                    shell_rows.append((
                        shell,
                        _weighted_mean(
                            [item[0] for item in group],
                            [item[2] * (ANCHOR_WEIGHT_FORBIDDEN if item[1] else 1.0)
                             for item in group]),
                        _spread(item[0] for item in group), len(group),
                        any(item[1] for item in group)))
                spread = _spread(row[1] for row in shell_rows)
                probes = [entry for entry in entries if entry[1]]
                pure = [entry for entry in entries if not entry[1]]
                resolved[element] = {
                    "delta": float(delta),
                    "fit": fit,
                    "anchors": sorted(entries, key=lambda e: (e[7], e[4])),
                    "shells": shell_rows,
                    "spread": float(spread),
                    "spread_all": _spread(entry[0] for entry in entries),
                    "n": len(entries),
                    "n_forbidden": len(probes),
                    "forbidden_mean": (
                        _weighted_mean([entry[0] for entry in probes],
                                       [entry[2] for entry in probes])
                        if probes else None),
                    "forbidden_spread": _spread(entry[0] for entry in probes),
                    # how far the EXACT constraints still miss after the fit
                    "forbidden_residual": (
                        max(abs(entry[0] - float(delta)) for entry in probes)
                        if probes else 0.0),
                    # what is LEFT after the exact constraints have set the
                    # scale: the measured bonding shift of each pure pair
                    "pure_residuals": sorted(
                        (float(entry[0] - delta), entry[4], entry[3],
                         entry[5], entry[7]) for entry in pure),
                    "ok": bool(spread <= ELEMENT_SCALE_TOL),
                    "fallback": False,
                    "shells_without_anchor": sorted(
                        pool_shells.get(column, {}).get(element, set())
                        - set(shells)),
                }
            missing = []
            for element in dict.fromkeys(
                    self.symbols[index] for index in self.side_atoms[column]):
                if element not in resolved:
                    resolved[element] = {
                        "delta": float(column_mean), "anchors": [],
                        "fit": "none", "pure_residuals": [],
                        "forbidden_residual": 0.0,
                        "shells": [], "spread": 0.0, "spread_all": 0.0,
                        "n": 0, "n_forbidden": 0, "forbidden_mean": None,
                        "forbidden_spread": 0.0, "ok": False, "fallback": True,
                        "shells_without_anchor": sorted(
                            pool_shells.get(column, {}).get(element, set())),
                    }
                    missing.append(element)
            report[column] = {"elements": resolved, "mean": float(column_mean),
                              "missing": missing}
            self.element_scale_ok[column] = {
                element: info["ok"] for element, info in resolved.items()}
        self.site_report = report
        return report

    def _level_site_shift(self, level, column):
        """``(shift, element, note)`` of one fragment level in site mode.

        A level whose dominant element carries at least
        :data:`ANCHOR_DOMINANT_SHARE` of its projected weight takes that
        element's shift.  A level spread over several sites has no site of its
        own and takes the share-weighted mean of their shifts, with a note
        saying so.
        """
        info = self.site_report[column]
        deltas = {element: entry["delta"]
                  for element, entry in info["elements"].items()}
        element, share = _dominant_element(level)
        if element is not None and share >= ANCHOR_DOMINANT_SHARE:
            if element in deltas:
                return deltas[element], element, ""
        totals = _element_shares(level)
        norm = sum(totals.values())
        if totals and norm > 0 and any(key in deltas for key in totals):
            shift = sum(deltas.get(key, info["mean"]) * value
                        for key, value in totals.items()) / norm
            parts = "/".join(
                f"{key} {100 * value / norm:.0f}%"
                for key, value in sorted(totals.items(),
                                         key=lambda kv: -kv[1])[:3])
            return (shift, None,
                    f"no single site carries {100 * ANCHOR_DOMINANT_SHARE:.0f}% "
                    f"of this level ({parts}): it was shifted by the "
                    f"share-weighted mean of the site shifts, {shift:+.3f} eV")
        return (info["mean"], None,
                "no site could be resolved for this level: it was shifted by "
                f"the column mean, {info['mean']:+.3f} eV")

    def align_fragment_columns(self, records):
        """Put the three columns on the crystal column's energy scale.

        The crystal run is the REFERENCE (shift 0); each fragment level is
        raised by ``delta = E_crystal - E_fragment`` measured on its own site
        (``--vasp-align site``, the default, :meth:`site_anchor_report`) or on
        its whole column (``--vasp-align rigid``, :meth:`rigid_alignment`).
        The page's energy zero is then set by ``--vasp-zero``
        (:meth:`zero_offset`), which moves all three columns together and
        therefore changes no energy DIFFERENCE.

        Args:
            records: The per-k-point list built by :func:`report_and_write`.
                The level energies are shifted in place.

        Returns:
            ``(shifts, anchors)``.  ``shifts`` carries ``"mode"``, the applied
            ``"left"``/``"right"``/``"mo"`` column constants (in site mode the
            weighted column means, for display only), ``"zero"``,
            ``"reference"`` (always ``"mo"``), ``"rigid"`` and, in site mode,
            ``"elements"``.  ``anchors`` is the rigid anchor record, which the
            report prints as a diagnostic in both modes.  ``(None, anchors)``
            when no anchor exists at all; nothing is shifted then.
        """
        deltas, anchors = self.rigid_alignment(records)
        if deltas is None:
            return None, anchors
        self.rigid_shifts = dict(deltas)
        offset = self.zero_offset()
        shifts = {"mo": offset, "reference": "mo", "zero": offset,
                  "zero_mode": self.zero_mode, "mode": self.align_mode,
                  "rigid": dict(deltas)}
        if self.align_mode == "rigid":
            shifts["left"] = deltas["left"] + offset
            shifts["right"] = deltas["right"] + offset
            for record in records:
                for column in ("left", "right"):
                    for level in record["levels"][column]:
                        level.energy = level.raw_energy + deltas[column]
                        level.site_shift = deltas[column]
                        level.site_element = None
                        level.site_note = ""
        else:
            report = self.site_anchor_report(records)
            shifts["elements"] = {
                column: {element: entry["delta"]
                         for element, entry in report[column]["elements"].items()}
                for column in ("left", "right")}
            shifts["left"] = report["left"]["mean"] + offset
            shifts["right"] = report["right"]["mean"] + offset
            for record in records:
                for column in ("left", "right"):
                    for level in record["levels"][column]:
                        shift, element, note = self._level_site_shift(
                            level, column)
                        level.energy = level.raw_energy + shift
                        level.site_shift = shift
                        level.site_element = element
                        level.site_note = note
                        if note:
                            level.detail += f"\n{note}"
        for record in records:
            for level in record["levels"]["mo"]:
                level.energy = level.raw_energy
                level.site_shift = 0.0
                level.site_element = None
                level.site_note = ""
            for column in ("left", "right", "mo"):
                for level in record["levels"][column]:
                    level.energy += offset
                    level.site_shift += offset
        return shifts, anchors

    def trace_diagnostic(self, records, shifts):
        """Residual of the occupied-band sum rule, using no projection at all.

        Bond formation mixes the occupied fragment states among themselves, and
        a unitary mixing leaves the TRACE of the occupied block unchanged, so
        to first order the sum of the crystal's occupied one-electron energies
        equals the sum of the two fragments' occupied energies on the same
        scale.  With the CRYSTAL column as the reference, the residual of that
        sum rule is an independent measure of how far the fragment columns
        still are from the crystal's scale: it is the constant both of them
        would have to move by, TOGETHER, to satisfy it.  It uses no projection
        and no anchor, so it cross-checks the whole alignment; what is left in
        it is genuine charge transfer and relaxation.

        The residual is invariant under a rigid move of all three columns
        (the crystal appears once, the fragments once each, and their electron
        counts add up), so ``--vasp-zero`` cannot change it.

        Args:
            records: The per-k-point records of :func:`report_and_write`, after
                the alignment has been applied.
            shifts: The shift dictionary of :meth:`align_fragment_columns`.

        Returns:
            ``{"trace_shift", "electrons"}``, or ``None`` when the occupied
            electron counts of the crystal and of the two fragments do not
            match.
        """
        if not shifts:
            return None
        totals = {column: [0.0, 0] for column in ("mo", "left", "right")}
        for record in records:
            for column in ("mo", "left", "right"):
                for level in record["levels"][column]:
                    if level.electrons <= 0:
                        continue
                    totals[column][0] += level.energy * level.electrons
                    totals[column][1] += level.electrons
        fragment_electrons = totals["left"][1] + totals["right"][1]
        if fragment_electrons == 0 or totals["mo"][1] != fragment_electrons:
            return None
        trace_shift = ((totals["mo"][0] - totals["left"][0] - totals["right"][0])
                       / fragment_electrons)
        return {"trace_shift": float(trace_shift),
                "electrons": totals["mo"][1]}

    def manifold_residuals(self, records):
        """Measured alignment residual, per COMPLETE ``(k, irrep)`` manifold.

        Mixing the two fragment blocks of one irrep into the crystal block of
        the same irrep is a unitary transformation of a Hermitian matrix, so
        the centre of gravity of that manifold is invariant: where the crystal
        and the fragments carry the SAME number of states of an irrep at a k
        point (degeneracies counted), the two mean energies must agree once
        the scales are right.  The residual is therefore a direct, model-free
        measure of the alignment error of THIS system -- no projection, no
        anchor, no free parameter -- and it is what the bond deadband should
        be read against.

        It is measured on the drawn levels, because that is the diagram the
        colours belong to; a manifold the window has cut is incomplete and is
        skipped.  The number matters: :data:`BOND_ENERGY_TOL` describes itself
        as "the order of the residual alignment uncertainty", which holds for
        CaF2 (0.19 eV mean) but understates SrTiO3 (0.47 eV mean, 1.12 eV max)
        by some 60 %, and the error is systematic -- negative on the
        cation-d-only irreps, positive on the mixed p-d ones -- so it does not
        average away.  The report, the footer and the JSON now quote it.

        Returns:
            ``{"n", "mean", "max", "worst", "rows"}``, or ``None`` when no
            complete manifold exists.
        """
        rows = []
        for record in records:
            irreps = {level.irrep for column in ("left", "mo", "right")
                      for level in record["levels"][column]}
            for irrep in sorted(irreps):
                crystal = [level for level in record["levels"]["mo"]
                           if level.irrep == irrep]
                fragment = [level for column in ("left", "right")
                            for level in record["levels"][column]
                            if level.irrep == irrep]
                n_crystal = sum(level.degeneracy for level in crystal)
                n_fragment = sum(level.degeneracy for level in fragment)
                if n_crystal == 0 or n_crystal != n_fragment:
                    continue
                mean_crystal = sum(level.degeneracy * level.energy
                                   for level in crystal) / n_crystal
                mean_fragment = sum(level.degeneracy * level.energy
                                    for level in fragment) / n_fragment
                rows.append((record["name"], irrep, n_crystal,
                             float(mean_crystal - mean_fragment)))
        if not rows:
            return None
        values = [abs(row[3]) for row in rows]
        worst = max(rows, key=lambda row: abs(row[3]))
        return {"n": len(rows), "mean": float(sum(values) / len(values)),
                "max": float(max(values)), "worst": worst, "rows": rows}

    def apply_window(self, records, fragments=True) -> None:
        """Drop the levels outside the energy window, on the ALIGNED scale.

        The window is a VIEW on the finished diagram, never an input to the
        physics: it is applied HERE, after :meth:`align_fragment_columns` has
        chosen its anchors on the complete level set, so that cutting the
        drawing at ``--vasp-window`` cannot move the energy scale itself.  (It
        did until this was fixed: a window that hid the crystal counterpart of
        the anchor left the alignment resting on a 19 %-pure level and shifted
        the whole crystal column by 43 eV.)

        The bound is the crystal column's (``--vasp-window``, relative to the
        valence-band maximum); the fragment columns are cut at the same
        energies so that the three columns show one common range, and the
        connector weights of a crystal level are renormalized over the
        fragment levels that survive.

        Args:
            records: The per-k-point records of :func:`report_and_write`.
            fragments: Cut the fragment columns too.  ``False`` with
                ``--no-align``, where the fragment scales are the runs' own
                and the crystal window means nothing on them.
        """
        low, high = self.window
        shifted_vbm = self.vbm + getattr(self, "crystal_shift", 0.0)
        top = shifted_vbm + high
        bottom = None if low is None else shifted_vbm + low

        def inside(level):
            if level.energy > top:
                return False
            return bottom is None or level.energy >= bottom

        columns = ("left", "right") if fragments else ()
        for record in records:
            # what the window takes off the TOP of a fragment column, kept for
            # the per-k report: an empty, diffuse cation d manifold lands
            # within a few tenths of an eV of the default VBM + 10 eV cut (on
            # CaF2 at GM, 0.05 eV above it), and losing it silently leaves the
            # LUMO of the page drawn with no parent and no connector while the
            # other k points look perfectly healthy
            lost = []
            for column in columns:
                for level in record["levels"][column]:
                    if level.energy > top:
                        lost.append((float(level.energy), level.label))
            record["above_window"] = sorted(lost, reverse=True)
            record["levels"]["mo"] = [level for level in record["levels"]["mo"]
                                      if inside(level)]
            for column in columns:
                record["levels"][column] = [level for level
                                            in record["levels"][column]
                                            if inside(level)]
                # the occurrence suffixes were handed out before the window
                # took levels away; a lone "#1" must lose its number
                _suffix_duplicate_labels(record["levels"][column])
            # Recompute the connectors over the fragment levels that survive
            # instead of renormalizing the old ones.  The PROCAR basis has one
            # radial channel per (atom, l), so an undrawn "O 3p" level sits in
            # the same channel as the O 2p band and the Loewdin step splits the
            # oxygen parentage evenly between them; simply rescaling what is
            # left then reports 51 % where the level's own O p share is 64 %.
            self._assign_compositions(record["levels"])

    # ------------------------------------------------- column-scale diagnostic

    def column_scale_report(self, records):
        """Is each fragment column really on ONE scale with the crystal?

        The alignment applies ONE rigid shift per column, which is only
        legitimate if the fragment run and the crystal run differ by one
        constant.  For every fragment level that has a crystal counterpart of
        at least :data:`COLUMN_SCALE_PURITY` absolute composition, the residual
        ``delta = E_crystal - E_fragment`` is collected per ``(element,
        shell)``.  A column whose shells disagree by more than
        :data:`COLUMN_SCALE_TOL` is NOT on one scale, and the bond character
        that would be read off its energies is meaningless.

        The cleanest probes are the SYMMETRY-FORBIDDEN pairs: a fragment level
        whose irrep does not occur in the other fragment at that k point cannot
        mix with the other sublattice at all, so after a correct alignment its
        crystal counterpart must sit at the same energy.  They are marked in
        the result.

        On SrTiO3 this is what exposes the physics limit of the bare point
        charges: the Sr 4s / 4p semicore of the cation run agrees with the
        crystal to 0.3 eV, while the EMPTY Ti 3d shell of the same run is
        11-13 eV too deep, because a sigma = 0.5 A point charge does not
        reproduce the screening a real O(2-) valence shell gives the cation
        site.  No rigid shift can fix both.

        Args:
            records: The per-k-point records, AFTER the alignment.

        Returns:
            ``{column: {"groups": [(element_shell, mean, spread, n, forbidden),
            ...], "spread": float, "ok": bool}}``.
        """
        result: dict = {}
        for column in ("left", "right"):
            other = {"left": "right", "right": "left"}[column]
            groups: dict = {}
            for record in records:
                fragment_levels = {level.level_id: level
                                   for level in record["levels"][column]}
                partner_irreps = {level.irrep
                                  for level in record["levels"][other]}
                best: dict = {}
                for crystal in record["levels"]["mo"]:
                    for level_id, weight in getattr(
                            crystal, "absolute_composition", []):
                        if not level_id.startswith(column):
                            continue
                        if weight > best.get(level_id, (0.0, None))[0]:
                            best[level_id] = (weight, crystal)
                for level_id, (weight, crystal) in best.items():
                    if weight < COLUMN_SCALE_PURITY:
                        continue
                    fragment = fragment_levels.get(level_id)
                    if fragment is None:
                        continue
                    key = " ".join(fragment.label.split()[:2])
                    groups.setdefault(key, []).append(
                        (crystal.energy - fragment.energy,
                         fragment.irrep not in partner_irreps,
                         record["name"], fragment.label))
            rows = []
            for key, values in sorted(groups.items()):
                deltas = [value[0] for value in values]
                rows.append((key, float(np.mean(deltas)),
                             float(max(deltas) - min(deltas)), len(deltas),
                             any(value[1] for value in values),
                             [value for value in values if value[1]]))
            means = [row[1] for row in rows]
            spread = float(max(means) - min(means)) if len(means) > 1 else 0.0
            result[column] = {"groups": rows, "spread": spread,
                              "ok": spread <= COLUMN_SCALE_TOL}
            self.column_scale_ok[column] = result[column]["ok"]
        return result

    # --------------------------------------------------------- bond character

    def _semicore_shells(self, records) -> set:
        """``{"Ca 3p", "F 2s", ...}``: the shells left out of the parent energy.

        A fragment ``(element, shell)`` block whose highest OCCUPIED level lies
        more than :data:`SEMICORE_DEPTH_EV` below the crystal VBM on the
        aligned scale.  Same criterion as ``--valence-only`` for the other two
        engines: what such a shell contributes to a valence level is the
        on-site orthogonality tail of the radially blind PROCAR projection, not
        a bonding partner, and counting it pulls the parent energy of that
        level down by more than the whole bond deadband.
        """
        vbm = max((level.energy for record in records
                   for level in record["levels"]["mo"] if level.electrons > 0),
                  default=None)
        if vbm is None:
            return set()
        top: dict = {}
        for record in records:
            for column in ("left", "right"):
                for level in record["levels"][column]:
                    if level.electrons <= 0:
                        continue
                    key = " ".join(level.label.split()[:2])
                    top[key] = max(top.get(key, -1e30), level.energy)
        return {key for key, high in top.items()
                if high < vbm - SEMICORE_DEPTH_EV}

    def record_full_parents(self, records) -> None:
        """Parent energy of every crystal level over the UNWINDOWED fragment set.

        Called once, after the alignment and BEFORE :meth:`apply_window`, so
        that each crystal level remembers what its parent energy would be if
        every fragment level it has weight on were drawn.  The drawn parent
        energy is the one the page can justify; this one says whether the
        window changed the answer.

        It exists because ``delta_Ti = +16 eV`` lifts the Ti 4s and Ti 4p
        fragment manifolds ~20 eV above the VBM, so the cation partner of the
        O 2p and O 2s valence bands leaves the page: SrTiO3 ``GM4- #2``
        (67 % O 2p, 31 % Ti 4p) is the BOTTOM of the O 2p valence band and was
        drawn antibonding red against a parent set that had lost its cation
        partner.  Where the two answers differ,
        :meth:`assign_bond_characters` refuses to colour the level at all.
        """
        semicore = self._semicore_shells(records)
        for record in records:
            fragments = {level.level_id: level
                         for column in ("left", "right")
                         for level in record["levels"][column]}
            for crystal in record["levels"]["mo"]:
                everything = [(fragments[i], w) for i, w in crystal.composition
                              if i in fragments]
                pairs = [(level, weight) for level, weight in everything
                         if _level_shell(level) not in semicore]
                whole = sum(weight for _, weight in everything)
                if sum(weight for _, weight in pairs) < 0.2 * whole:
                    pairs = everything
                total = sum(weight for _, weight in pairs)
                crystal.full_parent_energy = (
                    sum(level.energy * weight for level, weight in pairs) / total
                    if total > 0 else None)
                crystal.full_parent_coverage = float(
                    sum(weight for _, weight in crystal.composition))

    def assign_bond_characters(self, records) -> None:
        """Bonding / antibonding / nonbonding of every crystal level.

        There is no overlap matrix in a plane-wave calculation, so the COOP
        population of the other engines cannot be formed.  The classification
        is made from the ALIGNED energies and the composition instead: a
        crystal level with an appreciable share on both fragments is bonding
        when it lies below its composition-weighted parent energy and
        antibonding when it lies above; a level with at least
        ``NONBONDING_SHARE`` of its projected weight on one fragment has no
        partner to mix with and is nonbonding.

        The classification is only as good as the energy scale it reads.  A
        crystal level with a parent on a SITE whose own anchors disagree by
        more than :data:`ELEMENT_SCALE_TOL` (``--vasp-align site``), or on a
        COLUMN the rigid scale test failed (``--vasp-align rigid``), or any
        level at all when ``--no-align`` left the three runs on their own
        G = 0 references, is left UNCLASSIFIED: ``bond_character`` stays
        ``None``, the page draws it with the neutral stroke, and the tooltip
        says why.  Colouring an occupied Ti 3d - O 2p bonding band red because
        its Ti 3d parent sits 11 eV too deep would be worse than not colouring
        it.

        A level carrying ``NONBONDING_SHARE`` of its weight on ONE fragment is
        classified nonbonding BEFORE that test: it has no partner to mix with,
        which is a statement about the projection, not about the energy scale.

        The parent energy averages over ALL parents of the level, with the
        deep semicore shells of :meth:`_semicore_shells` left out (unless the
        level has no other parent at all).  See the note on
        :data:`PARENT_WEIGHT_FLOOR`, which now governs only which sites may
        veto a classification.
        """
        suppressed = {column for column in ("left", "right")
                      if not self.column_scale_ok.get(column, True)}
        site_mode = (getattr(self, "align_mode", "site") == "site"
                     and bool(self.site_report))
        semicore = self._semicore_shells(records)
        for record in records:
            fragments = {level.level_id: level
                         for column in ("left", "right")
                         for level in record["levels"][column]}
            for crystal in record["levels"]["mo"]:
                shares = getattr(crystal, "side_weights", {"left": 0.0, "right": 0.0})
                parts = [(column, value) for column, value in shares.items()]
                dominant = max(parts, key=lambda item: item[1]) if parts else ("", 0.0)
                everything = [(fragments[i], w) for i, w in crystal.composition
                              if i in fragments]
                pairs = [(level, w) for level, w in everything
                         if " ".join(level.label.split()[:2]) not in semicore]
                whole = sum(weight for _, weight in everything)
                if sum(weight for _, weight in pairs) < 0.2 * whole:
                    # nothing but semicore parents: the level IS a semicore
                    # band, and its own shell is the only scale it has
                    pairs = everything
                total = sum(weight for _, weight in pairs)
                parent = (sum(level.energy * weight for level, weight in pairs) / total
                          if total > 0 else None)
                crystal.parent_energy = parent
                covered = sum(weight for _, weight in crystal.composition)
                crystal.parent_coverage = float(covered)
                if site_mode:
                    # the site-resolved scale is judged per ELEMENT, so only
                    # the sites this level actually draws on can spoil it --
                    # and "draws on" means its SHARES, not only the parent
                    # links that happened to survive the window.  Read off the
                    # links alone, the veto misses exactly the levels that
                    # need it: SrTiO3 R4- #2 is 79.3 % Sr 4d, but no Sr 4d R4-
                    # fragment level is inside the window, so there was no Sr
                    # link to veto with and the level kept a colour read from
                    # its 10.7 % O 2s parent.
                    bad = set()
                    element_shares = _element_shares(crystal)
                    norm = sum(element_shares.values()) or 1.0
                    for element, value in element_shares.items():
                        column = self.element_column.get(element)
                        if column not in ("left", "right"):
                            continue
                        if value / norm < PARENT_WEIGHT_FLOOR:
                            continue
                        if not self.element_scale_ok.get(column, {}).get(
                                element, False):
                            bad.add(element)
                    for level_id, weight in crystal.composition:
                        if weight < PARENT_WEIGHT_FLOOR or level_id not in fragments:
                            continue
                        column = fragments[level_id].column
                        element, share = _dominant_element(fragments[level_id])
                        if element is None or share < ANCHOR_DOMINANT_SHARE:
                            bad.add(f"{column} (mixed-site parent "
                                    f"{fragments[level_id].label})")
                        elif not self.element_scale_ok.get(column, {}).get(
                                element, False):
                            bad.add(element)
                    tainted = sorted(bad)
                    trouble = (
                        "not classified: "
                        + " and ".join(tainted)
                        + (" do not have" if len(tainted) > 1 else " does not have")
                        + " one energy scale with the crystal (the site-resolved "
                          "anchors of that site disagree by more than "
                          f"{ELEMENT_SCALE_TOL:g} eV), so the sign of "
                          "E - E_parents carries no chemistry")
                else:
                    tainted = sorted(
                        column for column in suppressed
                        if any(level_id.startswith(column)
                               for level_id, _w in crystal.composition))
                    trouble = (
                        "not classified: the "
                        + " and ".join(tainted)
                        + (" columns are" if len(tainted) > 1 else " column is")
                        + " not on one energy scale with the crystal (see the "
                          "column-scale diagnostic), so the sign of "
                          "E - E_parents carries no chemistry")
                full = getattr(crystal, "full_parent_energy", None)
                crystal.unclassified_kind = ""
                drawn_class = _bond_class(crystal.energy, parent)
                full_class = _bond_class(crystal.energy, full)
                if dominant[1] >= NONBONDING_SHARE:
                    crystal.bond_character = "nonbonding"
                    reason = (f"{100 * dominant[1]:.0f}% of the projected weight "
                              f"on the {dominant[0]} fragment alone")
                elif tainted:
                    crystal.bond_character = None
                    crystal.unclassified_kind = "site scale"
                    reason = trouble
                elif parent is None:
                    crystal.bond_character = "nonbonding"
                    reason = ("no fragment level of its irrep is drawn, so it "
                              "has no parent to be shifted against")
                elif covered < PARENT_COVERAGE_FLOOR:
                    crystal.bond_character = None
                    crystal.unclassified_kind = "parent coverage"
                    reason = (
                        "not classified: only "
                        f"{100 * covered:.0f}% of this level's projected "
                        "character has a fragment level on the page (floor "
                        f"{100 * PARENT_COVERAGE_FLOOR:.0f}%), so the sign of "
                        "E - E_parents would be read off a minority channel "
                        "rather than off the level itself")
                elif full is not None and full_class != drawn_class:
                    crystal.bond_character = None
                    crystal.unclassified_kind = "parents off the window"
                    reason = (
                        "not classified: "
                        f"{100 * (1.0 - covered):.0f}% of this level's "
                        "character belongs to fragment levels the window does "
                        "not show, and with them the parent energy is "
                        f"{full:.2f} eV instead of {parent:.2f} eV, i.e. "
                        f"{full_class} instead of {drawn_class} -- the colour "
                        "would say more about the window than about the bond")
                elif drawn_class == "bonding":
                    crystal.bond_character = "bonding"
                    reason = (f"{crystal.energy - parent:+.2f} eV against its "
                              f"parents ({parent:.2f} eV)")
                elif drawn_class == "antibonding":
                    crystal.bond_character = "antibonding"
                    reason = (f"{crystal.energy - parent:+.2f} eV against its "
                              f"parents ({parent:.2f} eV)")
                else:
                    crystal.bond_character = "nonbonding"
                    reason = (f"within {BOND_ENERGY_TOL:g} eV of its parents "
                              f"({parent:.2f} eV)")
                margin = (None if parent is None
                          else abs(crystal.energy - parent))
                noise = getattr(self, "manifold_residual", None)
                inside = bool(
                    crystal.bond_character in ("bonding", "antibonding")
                    and margin is not None and noise
                    and margin < float(noise["mean"]))
                crystal.inside_residual = inside
                links = "  |  ".join(
                    f"{fragments[i].label} {100 * w:.1f}%"
                    for i, w in sorted(crystal.composition,
                                       key=lambda kv: -kv[1])
                    if i in fragments)
                # the connector weights are channel shares, so what is missing
                # from 100 % is the parentage of fragment levels the page does
                # not show (outside the window, or below the projection floor)
                missing = ("" if covered >= 0.98 else
                           f"\n{100 * (1.0 - covered):.1f}% of this level's "
                           "projected character belongs to fragment levels "
                           "that are not drawn (outside the energy window, or "
                           "below the projection floor); the parent energy "
                           "uses only the parents shown")
                crystal.detail += (
                    f"\nshares: left {100 * shares.get('left', 0.0):.1f}% / "
                    f"right {100 * shares.get('right', 0.0):.1f}%"
                    + (f"\nmade of: {links}" if links else "")
                    + missing
                    + f"\n{crystal.bond_character or 'bond character'}: {reason}"
                    + ("" if not inside else
                       f"\nNOTE: |E - E_parents| = {margin:.2f} eV is below the "
                       f"measured alignment residual of this system "
                       f"({noise['mean']:.2f} eV mean, {noise['max']:.2f} eV "
                       "max over the complete (k, irrep) manifolds), so this "
                       "colour is inside the noise"))

    # ------------------------------------------------------------- utilities

    def _valence_band_maximum(self) -> float:
        """Highest occupied eigenvalue of the crystal run, over all its k."""
        procar = self.runs["mo"]["procar"]
        occupied = procar.energies[procar.occupations > 0.5]
        if occupied.size == 0:
            return 0.0
        return float(np.max(occupied))

    def sketch_partners(self, level, kpoint, sites):
        """Real wave-function amplitudes of a level on the supercell atoms.

        The PROCAR projection vector already lives in the ``(atom, l, m)``
        real-harmonic basis the sketch draws, so the lobe amplitudes are the
        projections themselves (no radial calibration is needed: the
        projections are the population amplitudes).  ``f`` components are not
        drawn, as in the other engines.

        Args:
            level: A :class:`DiagramLevel` from :meth:`solve_at`.
            kpoint: The k point the level was solved at.
            sites: The ``(atom index, translation)`` list of
                :meth:`supercell_for`.

        Returns:
            One list per degenerate partner, each holding one
            ``[atom, s, px, py, pz, dxy, dyz, dz2, dxz, dx2-y2]`` entry per
            drawn supercell atom, or ``None``.
        """
        from .visualize_basis import realify_basis_space

        if level.vectors is None or level.vectors.shape[1] == 0:
            return None
        rows, _ = realify_basis_space(level.vectors.T)
        width = 9
        slot_of = {0: 0, 1: 1, 2: 4}
        column = getattr(level, "column", "mo")
        atoms = (list(range(len(self.symbols))) if column == "mo"
                 else self.side_atoms[column])
        partners = []
        for vector in np.asarray(rows):
            grid_re = np.zeros((len(sites), width))
            grid_im = np.zeros_like(grid_re)
            for row_index, (atom, translation) in enumerate(sites):
                if atom not in atoms:
                    continue
                phase = np.exp(2j * np.pi * float(np.dot(kpoint, translation)))
                for l, offset in self.lm_layout:
                    if l not in slot_of:
                        continue
                    size = 2 * l + 1
                    base = atom * self.n_lm + offset
                    values = np.asarray(vector[base:base + size]) * phase
                    slot = slot_of[l]
                    grid_re[row_index, slot:slot + size] += values.real
                    grid_im[row_index, slot:slot + size] += values.imag
            grid = (grid_re if np.linalg.norm(grid_re) >= np.linalg.norm(grid_im)
                    else grid_im)
            peak = np.max(np.abs(grid)) or 1.0
            entries = []
            for row_index in range(grid.shape[0]):
                values = grid[row_index] / peak
                if np.max(np.abs(values)) >= 0.04:
                    entries.append([row_index]
                                   + [round(float(x), 3) for x in values])
            partners.append(entries)
        return [entry for entry in partners if entry] or None


# --------------------------------------------------------------------- set-up


#: Width (A) of the smeared Coulomb potential of a Va point charge,
#: ``VACSIGMA``.  A 72-point scan over ``(sigma, VACRWALL, wall factor)`` on
#: SrTiO3 and ScF3 found sigma nearly inert at fixed wall -- points with the
#: same height and width but different sigma agree to 0.05 eV rms -- while
#: sigma 0.3 is measurably more prone to collapsing into the bare ``-q/r``
#: well.  0.5 A it is.
VACSIGMA_DEFAULT = 0.5

#: Wall height ``VACWALL`` in units of the smeared Coulomb depth
#: ``|V(0)| = q e^2 sqrt(2/pi) / sigma``, i.e. ``VACWALL = factor * |V(0)|``.
#: The module shipped with 1.25; the calibration below picked twice that.
VACWALL_FACTOR_DEFAULT = 2.5

#: Wall width (A) of the repulsive regularization around a POSITIVE Va point
#: charge, ``VACRWALL``.
#:
#: These three numbers are the recommendation of the 72-point scan, and they
#: are also the defaults of ``src/vacancy_charge.F`` in the patched VASP, so a
#: run made by hand and a run set up by ``--vasp-setup`` agree.  The scan is
#: organized by the single coordinate ``r10 = b sqrt(2 ln(A / 10 eV))``, the
#: radius at which the wall still repels by 10 eV: below ``r10 = 0.7 A`` the
#: anion sublattice collapses into the bare ``-q/r`` well (bands below the
#: projection floor, negative gaps, metallic points), above ``r10 = 1.4 A`` the
#: wall expels the anion valence shell (an O 2p manifold three times too wide,
#: alignment shifts running to +47 eV).  ``(0.5, 2.5, 0.45)`` gives
#: ``r10 = 0.99 A`` for ``q = +2``, dead centre of the ``0.85-1.25 A`` plateau,
#: and 1.07 / 1.13 A for ``q = +3`` and ``q = +4``: because the height scales
#: with ``q``, ONE setting serves all three charges.  It is the joint minimum
#: of the anchor residual rms on both test systems (SrTiO3 0.407 eV, ScF3
#: 0.304 eV) and simultaneously the point where the anion p-manifold width
#: matches the crystal best (-0.13 and +0.06 eV, against +0.49 and +0.40 eV at
#: the values shipped before), and it transfers to CaF2 -- a third space group
#: and a tetrahedral anion cage -- without retuning (rms 0.289 eV).
#:
#: What the wall chooses is the anion column's POSITION: over the whole scan
#: the alignment shift ranges from +1.8 to +46.7 eV while the residual rms on
#: the plateau moves only between 0.40 and 0.52 eV.  The engine therefore
#: warns when the two sublattice runs it reads were not computed with the same
#: VACRWALL / VACSIGMA.
VACRWALL_DEFAULT = 0.45

#: ``FELECT`` of ``src/constant.inc``: ``e^2`` in eV A.
_FELECT = 2.0 * 13.605826 * 0.529177249


def _wall_height(charge, sigma=VACSIGMA_DEFAULT,
                 factor=VACWALL_FACTOR_DEFAULT) -> float:
    """``VACWALL`` (eV) of one Va species, the same formula VASP defaults to.

    ``factor * q * e^2 * sqrt(2/pi) / sigma`` for ``q > 0`` and 0 otherwise:
    the wall only regularizes an ATTRACTIVE bare point charge, and a negative
    Va repels the electrons by itself.  Writing one value per Va species keeps
    the ``q`` scaling automatic, which is what makes a single ``(sigma,
    factor, VACRWALL)`` serve ``Va2+``, ``Va3+`` and ``Va4+`` alike.
    """
    if charge <= 0.0:
        return 0.0
    return float(factor) * float(charge) * _FELECT * math.sqrt(2.0 / math.pi) \
        / float(sigma)

_INCAR_TEMPLATE = """SYSTEM = {system}
PREC = Accurate
ENCUT = 550
LREAL = .FALSE.
ISMEAR = 0
SIGMA = 0.05
EDIFF = 1E-6
NELM = 200
IBRION = -1
NSW = 0
LORBIT = 12
LASPH = .TRUE.
GGA = PE
LWAVE = {lwave}
LCHARG = .FALSE.
NBANDS = {nbands}
{va}"""


#: INCAR tags that must NOT be carried over from the crystal run.  The
#: electron count is derived by VASP from the signed Va charges; each
#: sublattice run is self-contained (no CHGCAR/WAVECAR to restart from); and
#: the parallel layout of the crystal run does not fit the command this module
#: prints -- a copied ``KPAR = 12`` with ``-np 8`` is not a valid VASP setup,
#: and a copied ``NPAR`` is silently overwritten by VASP anyway.
_DROPPED_INCAR_TAGS = ("ICHARG", "ISTART", "NELECT", "NPAR", "NCORE", "KPAR")


def _va_incar_lines(charges, *, sigma=VACSIGMA_DEFAULT,
                    rwall=VACRWALL_DEFAULT,
                    factor=VACWALL_FACTOR_DEFAULT) -> dict:
    """The ``VACSIGMA`` / ``VACRWALL`` / ``VACWALL`` tags of one sublattice run.

    ``charges`` are the signed Va charges in POSCAR SPECIES order, which is
    the order the patched VASP reads these lists in.  ``VACWALL`` is written
    out per species rather than left to VASP's default so that the INCAR
    records the calibration explicitly and a run stays reproducible if the
    default ever moves again; the values are the default formula, so the two
    agree today.
    """
    if not charges:
        return {}
    heights = [_wall_height(charge, sigma, factor) for charge in charges]
    return {"VACSIGMA": f"{sigma:g}", "VACRWALL": f"{rwall:g}",
            "VACWALL": " ".join(f"{value:.6f}" for value in heights)}


def _strip_incar(text, nbands, va=None, lwave=False) -> str:
    """The crystal INCAR, made fit for a sublattice run.

    ``NELECT`` is dropped (VASP derives it from the signed Va charges),
    ``ICHARG``/``ISTART`` are dropped (each run is self-contained),
    ``NPAR``/``NCORE``/``KPAR`` are dropped (the crystal run's parallel layout
    need not fit the rank count of the printed command), ``LORBIT`` is forced
    to 12, the three ``Va`` tags to the calibrated wall of
    :func:`_va_incar_lines`, ``LWAVE``/``LCHARG`` to ``.FALSE.`` (a
    single-shot run has nothing to restart) -- ``LWAVE = .TRUE.`` with
    ``lwave``, for the overlap engine, which reads the wavefunctions -- and
    the ionic loop is switched off: the three runs must share one geometry,
    and a relaxation would move the real atoms away from it.
    """
    forced = {"LORBIT": "12", "IBRION": "-1", "NSW": "0",
              "NBANDS": str(nbands),
              "LWAVE": ".TRUE." if lwave else ".FALSE.", "LCHARG": ".FALSE."}
    forced.update(va or {})
    out = []
    seen: set = set()
    for line in text.splitlines():
        body = line.split("#")[0].split("!")[0]
        keep = []
        for tag in (part.strip() for part in body.split(";")):
            if not tag:
                continue
            name = tag.split("=")[0].strip().upper()
            if name in _DROPPED_INCAR_TAGS:
                continue
            if name in forced:
                seen.add(name)
                keep.append(f"{name} = {forced[name]}")
                continue
            keep.append(tag)
        if keep:
            out.append(" ; ".join(keep))
    for name, value in forced.items():
        if name not in seen:
            out.append(f"{name} = {value}")
    return "\n".join(out) + "\n"


def _kpoints_text(lattice, positions, numbers, special, mesh) -> str:
    """Symmetry-reduced Gamma-centred mesh plus zero-weight special points.

    A special point that the irreducible mesh already carries (modulo a
    reciprocal-lattice vector) is NOT repeated as a zero-weight line: listing
    Gamma twice makes VASP diagonalize it twice and leaves two PROCAR entries
    for one k point.  The header comment names the ones the mesh covers.
    """
    import spglib

    result = spglib.get_ir_reciprocal_mesh(
        mesh, (lattice, positions, numbers), is_shift=[0, 0, 0])
    if result is None:
        raise SystemExit("ERROR: spglib could not reduce the k mesh.")
    mapping, grid = result
    counts: dict = {}
    for value in mapping:
        counts[int(value)] = counts.get(int(value), 0) + 1
    unique = sorted(counts)
    mesh_points = [np.asarray(grid[index], dtype=float)
                   / np.asarray(mesh, dtype=float) for index in unique]

    def already_in_mesh(point):
        for other in mesh_points:
            difference = np.asarray(point, dtype=float) - other
            if np.all(np.abs(difference - np.rint(difference)) < 1e-6):
                return True
        return False

    extra = [(name, point) for name, point in special
             if not already_in_mesh(point)]
    covered = [name for name, point in special if already_in_mesh(point)]
    head = (f"CrystOD --vasp-setup: {mesh[0]}x{mesh[1]}x{mesh[2]} irreducible "
            f"mesh ({len(unique)} points) + {len(extra)} zero-weight special "
            f"point{'s' if len(extra) != 1 else ''}")
    if covered:
        head += f" ({', '.join(covered)} already in the mesh)"
    lines = [head, f"{len(unique) + len(extra)}", "Reciprocal lattice"]
    for point, index in zip(mesh_points, unique):
        lines.append("".join(f"{value:>20.14f}" for value in point)
                     + f"{counts[index]:>14d}")
    for name, point in extra:
        lines.append("".join(f"{value:>20.14f}" for value in point)
                     + f"{0:>14d}  ! {name}")
    return "\n".join(lines) + "\n"


#: A directory holding one of these is a FINISHED run: --vasp-setup refuses
#: to rewrite its inputs without --force, because the INCAR it would write is
#: not the INCAR that was run (IBRION, NSW, NBANDS, the Va wall tags).
FINISHED_RUN_FILES = ("OUTCAR", "PROCAR", "vasprun.xml")


def _finished_run_files(directory) -> list:
    """The files that say ``directory`` already holds a finished calculation."""
    from .vasp_io import resolve_vasp_file

    return [name for name in FINISHED_RUN_FILES
            if resolve_vasp_file(os.path.join(directory, name))]


def _setup_setting(shell, crystal_dir, report):
    """The setting the written POSCARs must follow: the CRYSTAL RUN's.

    ``--vasp-setup`` resolves the species, the fragments and the formal
    charges from the ``-c`` file, whose primitive cell may sit on another
    origin, carry another atom order and differ in the lattice constant by a
    rounding (the user's SrTiO3: Sr at the origin against Ti at the origin,
    3.9080205 against 3.9079888 A).  The sublattice runs must nevertheless be
    the crystal run's own cell with one sublattice replaced by point charges
    -- otherwise the three runs are three slightly different crystals, and
    the mapping of the analysis path silently repairs it.  So the ``-c``
    assignment is carried back through the same
    :class:`~.vasp_io.RunMapping` the analysis path uses.

    Args:
        shell: The ``-c`` shell object (``symbols``, ``positions``,
            ``lattice``).
        crystal_dir: ``<ROOT>/BAND``.
        report: Where the setting line goes.

    Returns:
        ``(lattice, positions, order)``: the lattice matrix to write, the
        positions of the analysis atoms in that setting (row ``a`` is
        analysis atom ``a``) and the analysis atoms sorted into the crystal
        run's ion order.  Without a crystal POSCAR this is the ``-c`` setting
        itself.

    Raises:
        SystemExit: There is a crystal POSCAR and it is not the ``-c`` cell in
            another setting.
    """
    from .vasp_io import (
        StructureMappingError,
        map_run_onto_cell,
        read_poscar,
        resolve_vasp_file,
    )

    identity = (np.asarray(shell.lattice, dtype=float),
                np.asarray(shell.positions, dtype=float),
                list(range(len(shell.symbols))))
    path = resolve_vasp_file(os.path.join(crystal_dir, "POSCAR"))
    if not path:
        return identity
    structure = read_poscar(path)
    try:
        mapping = map_run_onto_cell(shell.lattice, shell.positions,
                                    shell.symbols, structure)
    except StructureMappingError as error:
        raise SystemExit(
            f"ERROR: {path} is not the -c cell in another setting: "
            f"{error.reason}; write the inputs in the crystal run's own "
            f"setting with -c {path}.") from None
    ions = list(mapping.ions)
    positions = np.asarray(structure.positions, dtype=float)[ions]
    order = sorted(range(len(ions)), key=lambda atom: ions[atom])
    line = mapping.describe(shell.symbols, structure.symbols, path)
    report(f" setting       : {line}" if line
           else f" setting       : {path} (= the -c setting)")
    return np.asarray(structure.lattice, dtype=float), positions, order


def write_vasp_inputs(diagram_cell, *, left, right, root, symprec, oxidation,
                      potcar_dir=None, potcar_map=None, mesh=None,
                      rwall=VACRWALL_DEFAULT, sigma=VACSIGMA_DEFAULT,
                      wall_factor=VACWALL_FACTOR_DEFAULT, binary=None,
                      force=False, engine="anchor", frozen_window=None,
                      report=print) -> None:
    """Write the inputs of the two sublattice runs (and of the crystal run).

    The POSCARs keep the crystal's cell and atom order; the atoms of the
    removed sublattice are renamed to ``Va<q><sign>`` point charges built from
    the oxidation states crystod resolves.  The POTCAR is assembled from the
    crystal run's own POTCAR by splitting it on ``End of Dataset`` and keeping
    the blocks of the real species; the INCAR is copied from the crystal run
    (``LORBIT = 12``, ``VACRWALL``, no ``NELECT``, no ``ICHARG``/``ISTART``,
    no ``NPAR``/``NCORE``/``KPAR``, ``LWAVE = LCHARG = .FALSE.``) or written
    from a template; the KPOINTS file is one explicit list of the
    symmetry-reduced Gamma-centred mesh with weights followed by the
    zero-weight special points the mesh does not already carry, so ONE run per
    directory gives both the SCF and the special-point eigenvalues.

    Args:
        diagram_cell: The ``PhonopyAtoms`` crystal structure.
        left: ``--co-left`` tokens.
        right: ``--co-right`` tokens.
        root: The directory that holds (or will hold) ``BAND``,
            ``BAND_sublattice1`` and ``BAND_sublattice2``.
        symprec: Symmetry tolerance.
        oxidation: ``{element: charge}`` or ``None`` for pymatgen's guess.
        potcar_dir: Where to look for ``<El>*/POTCAR`` when the crystal run has
            no POTCAR; ``None`` falls back to ``PMG_VASP_PSP_DIR``.
        potcar_map: ``{element: dataset directory name}``, e.g.
            ``{"Sc": "Sc_sv"}``; without it the semicore variant is preferred
            whenever the formal charge would strip a whole valence shell.
        mesh: The Gamma-centred mesh; ``None`` uses
            ``n_i = max(1, round(24 / |a_i|))``.
        rwall: ``VACRWALL`` (A) written into the INCAR.
        sigma: ``VACSIGMA`` (A) written into the INCAR.
        wall_factor: ``VACWALL`` in units of the smeared Coulomb depth
            ``q e^2 sqrt(2/pi) / sigma``; one height per Va species is
            written, so the ``q`` scaling stays automatic.
        binary: Absolute path of the CrystOD-patched ``vasp_std`` for the
            printed commands; ``None`` looks at ``CRYSTOD_VASP`` and ``PATH``.
        engine: ``"overlap"`` prepares the runs for the WAVECAR-overlap
            engine: ``LWAVE = .TRUE.``, the sublattice runs get at least the
            crystal run's NBANDS, and the crystal run is checked
            (:func:`_overlap_band_check`).  ``"anchor"`` leaves everything
            as it was.
        frozen_window: The overlap engine's frozen window (eV above the
            VBM, default 14) the crystal bands must reach.

    Returns:
        ``None``.  The run commands are printed (unless a POTCAR is missing);
        VASP is not started.

    Raises:
        SystemExit: A sublattice would be left with no electrons at all.
    """
    from .crystal_orbital_pyscf import PySCFCrystalOrbitalDiagram
    from .vasp_io import (
        format_poscar,
        parse_va_species,
        potcar_block_facts,
        split_potcar,
        va_species_name,
    )

    binary, binary_note = _vasp_binary(binary)

    class _Shell(CrystalOrbitalDiagram):
        def __init__(self):
            self.builder = SymmetryAdaptedOrbitalBasis(cell=diagram_cell,
                                                       symprec=symprec)
            primitive = self.builder.primitive_cell
            self.symbols = get_chemical_symbols(primitive)
            self.positions = np.array(get_scaled_positions(primitive))
            self.lattice = np.array(primitive.cell)
            PySCFCrystalOrbitalDiagram._assign_fragments(self, left, right)
            PySCFCrystalOrbitalDiagram._resolve_oxidation(self, oxidation)

    shell = _Shell()
    special = shell.special_kpoints()
    if mesh is None:
        lengths = np.linalg.norm(shell.lattice, axis=1)
        mesh = [max(1, int(round(24.0 / length))) for length in lengths]
    numbers = []
    order: dict = {}
    for symbol in shell.symbols:
        numbers.append(order.setdefault(symbol, len(order) + 1))
    kpoints_text = _kpoints_text(shell.lattice, shell.positions, numbers,
                                 special, mesh)

    crystal_dir = os.path.join(root, "BAND")
    crystal_potcar = os.path.join(crystal_dir, "POTCAR")
    blocks: dict = {}
    if os.path.isfile(crystal_potcar):
        for element, block in split_potcar(open(crystal_potcar).read()):
            blocks.setdefault(element, block)
        report(f" POTCAR source : {crystal_potcar} "
               f"({', '.join(blocks)} blocks)")
    else:
        directory = potcar_dir or _pmg_psp_dir()
        if directory:
            for element in dict.fromkeys(shell.symbols):
                path = _choose_potcar(directory, element,
                                      shell.oxidation.get(element, 0.0),
                                      potcar_map or {})
                if path:
                    blocks[element] = open(path).read()
            report(f" POTCAR source : {directory}")
        if not blocks:
            report(" POTCAR source : NONE -- no BAND/POTCAR and no "
                   "--potcar-dir / PMG_VASP_PSP_DIR; write the POTCAR files "
                   "yourself (one block per real species, in POSCAR order)")
    # which dataset was picked, and what it leaves the sublattice to compute:
    # the difference between Sc (ZVAL 3) and Sc_sv (ZVAL 11) is a run with 0
    # electrons and a run with 8
    zval: dict = {}
    for element, block in blocks.items():
        title, value = potcar_block_facts(block)
        zval[element] = value
        report(f"   {element:<3} {title or '(no TITEL)':<28} ZVAL {value:g}"
               f"  -> {value - shell.oxidation.get(element, 0.0):g} e- "
               "per atom after the formal charge")

    nbands = _suggest_nbands(shell)
    incar_source = os.path.join(crystal_dir, "INCAR")
    incar_text = (open(incar_source).read() if os.path.isfile(incar_source)
                  else None)
    overlap = engine == "overlap"
    if overlap:
        crystal_bands = _overlap_band_check(
            crystal_dir, 14.0 if frozen_window is None else float(frozen_window),
            report)
        if crystal_bands:
            nbands = max(nbands, crystal_bands)
    commands = []
    not_ready: list = []
    targets = [("left", os.path.join(root, "BAND_sublattice1")),
               ("right", os.path.join(root, "BAND_sublattice2"))]
    # the crystal run's own inputs are written only when there are none AND
    # the run has not already been done (a finished run needs no setup, and
    # its files are the record of what was run)
    if not os.path.isfile(os.path.join(crystal_dir, "INCAR")) \
            and not _finished_run_files(crystal_dir):
        targets.insert(0, (None, crystal_dir))
    # the inputs of a FINISHED run are the record of what was actually run:
    # the INCAR written here is not that INCAR (IBRION, NSW, NBANDS, the Va
    # wall tags), so rewriting one destroys the record.  --force says so.
    if not force:
        finished = [(directory, _finished_run_files(directory))
                    for _column, directory in targets]
        finished = [(directory, names) for directory, names in finished if names]
        if finished:
            raise SystemExit(
                "ERROR: "
                + "; ".join(f"{directory} already holds a finished run "
                            f"({', '.join(names)}) -- POSCAR, KPOINTS, INCAR "
                            "and POTCAR would be replaced"
                            for directory, names in finished)
                + ".  Give --force to overwrite, or --vasp-setup a ROOT whose "
                  "sublattice directories are empty.")
    lattice, positions, atom_order = _setup_setting(shell, crystal_dir, report)
    for column, directory in targets:
        os.makedirs(directory, exist_ok=True)
        species, counts = [], []
        names: dict = {}
        for atom in atom_order:
            symbol = shell.symbols[atom]
            if column is None or shell.atom_column[atom] == column:
                names[atom] = symbol
            else:
                names[atom] = va_species_name(shell.oxidation[symbol])
        for atom in atom_order:
            if names[atom] not in species:
                species.append(names[atom])
                counts.append(0)
            counts[species.index(names[atom])] += 1
        if column is None:
            real = list(species)
        else:
            real = [name for name in species
                    if name in shell.symbols and shell.element_column.get(name) == column]
        order_index = []
        for name in species:
            order_index.extend([atom for atom in atom_order
                                if names[atom] == name])
        comment = " ".join(
            f"{name}{count}" for name, count in zip(species, counts))
        text = format_poscar(comment, lattice, species, counts,
                             positions[order_index])
        open(os.path.join(directory, "POSCAR"), "w").write(text)
        open(os.path.join(directory, "KPOINTS"), "w").write(kpoints_text)
        va_charges = [charge for charge in
                      (parse_va_species(name) for name in species)
                      if charge is not None]
        va_tags = _va_incar_lines(va_charges, sigma=sigma, rwall=rwall,
                                  factor=wall_factor)
        if incar_text is not None:
            open(os.path.join(directory, "INCAR"), "w").write(
                _strip_incar(incar_text, nbands, va_tags, lwave=overlap))
        else:
            open(os.path.join(directory, "INCAR"), "w").write(
                _INCAR_TEMPLATE.format(
                    system=comment, nbands=nbands,
                    lwave=".TRUE." if overlap else ".FALSE.",
                    va="".join(f"{name} = {value}\n"
                               for name, value in va_tags.items())))
        missing = [name for name in real if name not in blocks]
        if blocks and not missing:
            open(os.path.join(directory, "POTCAR"), "w").write(
                "".join(blocks[name] for name in real))
        elif missing:
            report(f" {directory}: POTCAR NOT written, missing "
                   f"{', '.join(missing)}")
            not_ready.append(directory)
        elif not blocks:
            not_ready.append(directory)
        report(f" wrote {directory}/POSCAR ({' '.join(species)}), KPOINTS, INCAR"
               + (", POTCAR" if blocks and not missing else ""))
        if column is not None and blocks and not missing:
            # VASP derives NELECT from ZVAL and the signed Va charges; a
            # dataset without semicore can leave the sublattice with nothing
            # to compute (Sc with ZVAL 3 and Sc(3+): zero electrons)
            electrons = sum(
                zval.get(name, 0.0) - shell.oxidation.get(name, 0.0)
                for atom, name in enumerate(shell.symbols)
                if shell.atom_column[atom] == column)
            report(f"   {column:<5} sublattice NELECT = {electrons:g}")
            if electrons <= 0:
                raise SystemExit(
                    f"ERROR: the {column} sublattice would run with NELECT = "
                    f"{electrons:g}: the PAW datasets chosen for "
                    f"{', '.join(real)} have too few valence electrons for "
                    "the formal charges. Pick a semicore dataset with "
                    "--potcar-map (e.g. --potcar-map Sc=Sc_sv).")
        commands.append(
            f"cd {directory} && OMP_NUM_THREADS=1 mpirun -np 8 {binary}")
    if not_ready:
        report("\n NOT ready to run: no POTCAR in "
               + ", ".join(not_ready)
               + ".\n Supply the PAW datasets with --potcar-dir DIR (and "
                 "--potcar-map EL=NAME to choose a variant), or copy the "
                 "POTCAR files in by hand; then re-run --vasp-setup.")
        return
    report(f"\n Run them ({binary_note}):")
    for command in commands:
        report(f"   {command}")
    report("\n Then draw the diagram with the same command and --vasp "
           f"{root}" + (" (the overlap engine reads the three WAVECARs)"
                        if overlap else ""))


def _choose_potcar(directory, element, charge, potcar_map):
    """Path of the PAW dataset to use for ``element``.

    A plain glob takes the first match alphabetically, which is exactly the
    wrong one where it matters: ``Sc`` sorts before ``Sc_sv``, and ``Sc`` has
    ZVAL 3, so a Sc(3+) sublattice would be run with zero electrons.  The
    semicore/valence variants (``_sv``, ``_pv``) are therefore preferred
    whenever the formal charge would take at least as many electrons as the
    plain dataset has, and ``--potcar-map EL=NAME`` overrides the choice
    outright.

    Args:
        directory: The PAW library (``<DIR>/<name>/POTCAR``).
        element: Chemical symbol.
        charge: The formal charge crystod resolved for it.
        potcar_map: ``{element: dataset name}`` from ``--potcar-map``.

    Returns:
        The path, or ``None`` when nothing matches.
    """
    import glob

    from .vasp_io import potcar_block_facts

    forced = potcar_map.get(element)
    if forced:
        path = os.path.join(directory, forced, "POTCAR")
        if not os.path.isfile(path):
            raise SystemExit(
                f"ERROR: --potcar-map names {forced} for {element}, but "
                f"{path} does not exist.")
        return path
    matches = sorted(glob.glob(os.path.join(directory, f"{element}*", "POTCAR")))
    matches = [path for path in matches
               if re.fullmatch(rf"{re.escape(element)}(_[a-z]+|\d?)",
                               os.path.basename(os.path.dirname(path)))]
    if not matches:
        return None
    best = matches[0]
    _title, best_zval = potcar_block_facts(open(best).read())
    if best_zval - float(charge) > 0:
        return best
    # the plain dataset has no electrons left after the formal charge: take
    # the richest variant the library offers instead
    richest, richest_zval = best, best_zval
    for path in matches[1:]:
        _title, value = potcar_block_facts(open(path).read())
        if value > richest_zval:
            richest, richest_zval = path, value
    return richest


def _vasp_binary(binary):
    """``(command, note)`` for the printed run lines.

    A bare ``vasp_std`` is worse than useless on a machine whose only VASP on
    ``PATH`` is a stock build: that binary cannot read the ``Va2-`` species and
    would either fail or, worse, mis-parse the POSCAR.  The absolute path of
    the patched build is used when ``--vasp-bin`` or ``CRYSTOD_VASP`` names
    it; otherwise the note says plainly what the binary has to be.
    """
    import shutil

    candidate = binary or os.environ.get("CRYSTOD_VASP")
    if candidate:
        return candidate, f"the CrystOD-patched build at {candidate}"
    found = shutil.which("vasp_std")
    if found:
        return found, (f"{found} -- it MUST be the CrystOD-patched build; a "
                       "stock vasp_std cannot read the Va point-charge "
                       "species")
    return "vasp_std", (
        "vasp_std is not on PATH here: replace it with the absolute path of "
        "the CrystOD-patched build, or set CRYSTOD_VASP / pass --vasp-bin.  "
        "A stock VASP cannot read the Va point-charge species")


def _pmg_psp_dir():
    path = os.path.expanduser("~/.pmgrc.yaml")
    if not os.path.isfile(path):
        return None
    for line in open(path):
        match = re.match(r"\s*PMG_VASP_PSP_DIR\s*:\s*(.+?)\s*$", line)
        if match:
            return os.path.expanduser(match.group(1).strip('"\''))
    return None


#: The overlap engine's outer bands: the crystal bands must reach this far
#: above the frozen window (VBM + 14 eV), so that the disentanglement has
#: bands to draw the empty active shells from (the published CsPbI3 + SOC run,
#: NBANDS 128, reaches VBM + 17.5 eV).
OVERLAP_BAND_MARGIN_EV = 3.0


def _incar_tag(text, name):
    """The value of one INCAR tag (``None`` when it is not set)."""
    for line in (text or "").splitlines():
        body = line.split("#")[0].split("!")[0]
        for tag in body.split(";"):
            if "=" in tag and tag.split("=")[0].strip().upper() == name:
                return tag.split("=", 1)[1].strip()
    return None


def _overlap_band_check(crystal_dir, window, report) -> int | None:
    """What the overlap engine needs of the crystal run, checked where possible.

    The engine reads the crystal's WAVECAR (``LWAVE = .TRUE.``) and keeps
    its levels up to VBM + ``window`` exactly, drawing the empty active shells
    from the bands above; so the crystal bands must reach VBM + ``window`` +
    :data:`OVERLAP_BAND_MARGIN_EV` at every k point.  A finished crystal run
    (``EIGENVAL``) is checked and a larger NBANDS suggested when its bands
    stop short; an unfinished one is checked for ``LWAVE``.

    Args:
        crystal_dir: ``<ROOT>/BAND``.
        window: The frozen window (eV above the VBM).
        report: Print function.

    Returns:
        The crystal run's NBANDS (finished run, or the INCAR tag), else
        ``None``; the sublattice runs are given at least as many.
    """
    from .vasp_io import read_eigenval, resolve_vasp_file

    target = window + OVERLAP_BAND_MARGIN_EV
    report(f" overlap engine: LWAVE = .TRUE. in every run; the crystal bands must "
           f"reach VBM + {target:g} eV (frozen window {window:g} eV + "
           f"{OVERLAP_BAND_MARGIN_EV:g} eV of outer bands) at every k point")
    incar = os.path.join(crystal_dir, "INCAR")
    incar_text = open(incar).read() if os.path.isfile(incar) else ""
    tagged = _incar_tag(incar_text, "NBANDS")
    eigenval = resolve_vasp_file(os.path.join(crystal_dir, "EIGENVAL"))
    if not eigenval:
        lwave = (_incar_tag(incar_text, "LWAVE") or "").upper().strip(".")
        if incar_text and not lwave.startswith("T"):
            report(f"   NOTE: {incar} does not set LWAVE = .TRUE.: add it before "
                   "running the crystal (its WAVECAR is read)")
        report("   the band reach of the crystal run is checked here once it has run")
        return int(tagged) if tagged and tagged.isdigit() else None
    _kpoints, _weights, energies, occupations = read_eigenval(eigenval)
    nbands = energies.shape[1]
    occupied = energies[occupations > 0.5]
    if not occupied.size:
        return nbands
    vbm = float(occupied.max())
    reach = float(energies[:, -1].min()) - vbm
    if not os.path.isfile(os.path.join(crystal_dir, "WAVECAR")):
        report(f"   NOTE: {crystal_dir} holds no WAVECAR: repeat the crystal run with "
               "LWAVE = .TRUE.")
    if reach >= target:
        report(f"   crystal run: NBANDS {nbands} reaches VBM + {reach:.1f} eV -- enough")
        return nbands
    bottom = float(energies.min()) - vbm
    # free-electron-like counting, N(E) ~ (E - E_bottom)^(3/2), rounded up to 8
    estimate = nbands * ((target - bottom) / max(reach - bottom, 1e-6)) ** 1.5
    suggested = int(8 * math.ceil(estimate / 8))
    report(f"   WARNING: the crystal run's highest band reaches only VBM + {reach:.1f} eV "
           f"(NBANDS {nbands}); repeat it with NBANDS = {suggested} or more "
           "(LWAVE = .TRUE.), and give the sublattice runs as many")
    return max(nbands, suggested)


def _suggest_nbands(shell) -> int:
    """Enough bands for the empty cation states of the biggest fragment."""
    from .mo_diagram import EHT_PARAMETERS

    electrons = 0
    virtual = 0
    for symbol in shell.symbols:
        electrons += 8
        virtual += sum(2 * l + 1 for _s, _n, l, _z, _h in EHT_PARAMETERS.get(symbol, []))
    return int(max(16, round(electrons / 2 + virtual + 8)))


# --------------------------------------------------------------------- report


#: What ``--vasp`` accepts, quoted by every error that rejects its paths.
VASP_PATH_FORMS = ("--vasp takes no path (the current directory as ROOT), one "
                   "ROOT holding BAND and BAND_sublattice1/2, or the three "
                   "run directories in any order")


def _resolve_directories(paths, overrides, report=print) -> dict:
    """Locate the crystal and sublattice runs of ``--vasp``.

    ``paths`` is what ``--vasp`` was given: nothing (the current directory is
    the ROOT), one ROOT, or the three run directories in any order.  With
    three, the crystal run is the one whose POSCAR carries no ``Va``
    point-charge species and the other two are sorted into the left and right
    columns by their real elements, exactly as under a ROOT.

    Each column can still be overridden on its own with ``--vasp-crystal`` /
    ``--vasp-left`` / ``--vasp-right``: the auto-detection then has to supply
    only the columns that were not given, and a directory named by an override
    is taken out of the auto-detection set.
    """
    import glob

    # a single ROOT may be given as a bare string (the older signature of this
    # function, which report_and_write still accepts as root=...)
    paths = [paths] if isinstance(paths, str) else list(paths or [])
    if len(paths) == 3:
        return _resolve_triple(paths, overrides, report)
    if len(paths) > 1:
        raise SystemExit(
            f"ERROR: {VASP_PATH_FORMS}; got {len(paths)} "
            f"({', '.join(paths)}).")
    root = paths[0] if paths else "."
    directories = {}
    crystal = overrides.get("mo") or os.path.join(root, "BAND")
    if not os.path.isdir(crystal):
        raise SystemExit(
            f"ERROR: no crystal run at {crystal} (expected <ROOT>/BAND, or "
            "give --vasp-crystal DIR).")
    directories["mo"] = _with_band_subdirectory(crystal)
    explicit = {column: overrides.get(column) for column in ("left", "right")}
    for column, path in explicit.items():
        if path and not os.path.isdir(path):
            raise SystemExit(
                f"ERROR: --vasp-{column} points at {path}, which is not a "
                "directory.")
    report(f" crystal run  : {directories['mo']}")
    if all(explicit.values()):
        for column in ("left", "right"):
            directories[column] = _with_band_subdirectory(explicit[column])
            report(f" {column:<5} run  : {directories[column]} (--vasp-{column})")
        return directories
    # only BAND_sublattice<N>, not BAND_sublattice<N>_<anything>: an archived
    # run parked beside the current one (BAND_sublattice2_wall06, the data of
    # an earlier point-charge wall) must not be counted as a third column
    candidates = sorted(
        path for path in glob.glob(os.path.join(root, "BAND_sublattice*"))
        if os.path.isdir(path)
        and re.fullmatch(r"BAND_sublattice\d+", os.path.basename(path)))
    given = set()
    for path in [*explicit.values(), crystal]:
        given |= _directory_aliases(path)
    candidates = [path for path in candidates
                  if not _directory_aliases(path) & given]
    wanted = [column for column in ("left", "right") if not explicit[column]]
    if len(candidates) != len(wanted):
        raise SystemExit(
            f"ERROR: cannot resolve the {' and '.join(wanted)} column"
            f"{'s' if len(wanted) > 1 else ''}: {len(candidates)} "
            f"<ROOT>/BAND_sublattice<N> director"
            f"{'ies' if len(candidates) != 1 else 'y'} left to choose from "
            f"under {root} after the overrides, {len(wanted)} needed; give "
            + " ".join(f"--vasp-{column} DIR" for column in wanted) + ".")
    return {"mo": directories["mo"], "candidates": candidates,
            "explicit": explicit}


def _directory_aliases(path) -> set:
    """Every spelling of one run directory, so overrides can be matched.

    ``DIR`` and ``DIR/band`` are the same run, and a column may be named in
    either spelling (``--vasp`` gives ``DIR``, ``--vasp-left`` may give
    ``DIR/band``).  Comparing one spelling only let an overridden directory
    stay in the auto-detection set and then collide with its own override.
    """
    if not path:
        return set()
    absolute = os.path.abspath(path)
    aliases = {absolute, os.path.abspath(_with_band_subdirectory(path))}
    if os.path.basename(absolute) == "band":
        aliases.add(os.path.dirname(absolute))
    return aliases


def _resolve_triple(paths, overrides, report=print) -> dict:
    """``--vasp DIR1 DIR2 DIR3``: the three runs, given in any order.

    The crystal run is the one whose POSCAR has no ``Va`` point-charge
    species; the other two are handed to :func:`_sort_sublattices`, which
    reads their real elements.  So the caller does not have to remember which
    sublattice ``--vasp-setup`` called ``1`` and which ``2``, and the
    directories need not be named ``BAND*`` or live under one root.
    """
    from .vasp_io import read_poscar

    resolved = []
    for path in paths:
        if not os.path.isdir(path):
            raise SystemExit(
                f"ERROR: --vasp names {path}, which is not a directory "
                f"({VASP_PATH_FORMS}).")
        resolved.append(_with_band_subdirectory(path))
    if len(set(os.path.abspath(path) for path in resolved)) != 3:
        raise SystemExit(
            "ERROR: --vasp was given the same run directory twice "
            f"({', '.join(paths)}).")
    explicit = {column: overrides.get(column) for column in ("left", "right")}
    for column, path in explicit.items():
        if path and not os.path.isdir(path):
            raise SystemExit(
                f"ERROR: --vasp-{column} points at {path}, which is not a "
                "directory.")
    crystal = overrides.get("mo")
    if crystal is None:
        pure = [path for path in resolved
                if not any(read_poscar(os.path.join(path, "POSCAR")).is_point_charge)]
        if len(pure) != 1:
            raise SystemExit(
                f"ERROR: the three --vasp directories hold {len(pure)} runs "
                "without Va point-charge species; exactly one of them is the "
                "crystal run (or give --vasp-crystal DIR).")
        crystal = pure[0]
    directories = {"mo": _with_band_subdirectory(crystal)}
    report(f" crystal run  : {directories['mo']}")
    given = set()
    for path in [*explicit.values(), directories["mo"]]:
        given |= _directory_aliases(path)
    candidates = [path for path in resolved
                  if not _directory_aliases(path) & given]
    return {"mo": directories["mo"], "candidates": candidates,
            "explicit": explicit}


def _with_band_subdirectory(directory) -> str:
    """``DIR/band`` when it holds a PROCAR (or ``PROCAR.gz``), else ``DIR``."""
    from .vasp_io import resolve_vasp_file

    nested = os.path.join(directory, "band")
    if resolve_vasp_file(os.path.join(nested, "PROCAR")):
        return nested
    return directory


def _sort_sublattices(resolved, element_column, report=print) -> dict:
    """Decide which sublattice directory is the left and which the right."""
    from .vasp_io import read_poscar

    if "candidates" not in resolved:
        return resolved
    directories = {"mo": resolved["mo"]}
    explicit = resolved["explicit"]
    remaining = list(resolved["candidates"])
    for column in ("left", "right"):
        if explicit[column]:
            directories[column] = _with_band_subdirectory(explicit[column])
    for path in remaining:
        # the POSCAR is read from the directory the run data will be read
        # from, so that a DIR holding nothing but its band/ subdirectory is
        # sorted as well as one that also kept its SCF files
        nested = _with_band_subdirectory(path)
        structure = read_poscar(os.path.join(nested, "POSCAR"))
        columns = {element_column.get(element) for element in structure.real_elements}
        columns.discard(None)
        if len(columns) != 1:
            raise SystemExit(
                f"ERROR: the real elements of {nested}/POSCAR "
                f"({', '.join(structure.real_elements)}) do not form one of the "
                "two --co-left/--co-right sublattices.")
        column = columns.pop()
        if column in directories:
            raise SystemExit(
                f"ERROR: {path} and {directories[column]} both look like the "
                f"{column} sublattice; give --vasp-left DIR --vasp-right DIR.")
        directories[column] = nested
    missing = [column for column in ("left", "right")
               if column not in directories]
    if missing:
        raise SystemExit(
            f"ERROR: could not resolve the {' and '.join(missing)} column"
            f"{'s' if len(missing) > 1 else ''}; give "
            + " ".join(f"--vasp-{column} DIR" for column in missing) + ".")
    for column in ("left", "right"):
        report(f" {column:<5} run  : {directories[column]}"
               + (f" (--vasp-{column})" if explicit[column] else ""))
    return directories


def _alignment_purity() -> float:
    from .crystal_orbital_pyscf import ALIGNMENT_PURITY

    return ALIGNMENT_PURITY


def _anchor_extent(info) -> str:
    """``"4 anchor levels at 4 k points"`` for an anchor record.

    ``n_k`` counts (k point, level) PAIRS: one shell can contribute two levels
    at the same k point, and then part of the quoted spread is a level
    splitting rather than a k dispersion.  Say both numbers.
    """
    levels = info.get("n_levels", info.get("n_k", 0))
    kpoints = info.get("n_kpoints", levels)
    return (f"{levels} anchor level{'s' if levels != 1 else ''} at "
            f"{kpoints} k point{'s' if kpoints != 1 else ''}")


def _report_point_charge_settings(diagram) -> None:
    """Print the Va point-charge parameters, and warn when they disagree.

    The sublattice eigenvalues depend on the wall width: on SrTiO3 the whole
    anion column moves by 0.8 eV on average between VACRWALL 0.5 and 0.6 and
    four O 2p level pairs change order, which is far more than the bond
    deadband.  Two runs made with different settings are not two halves of one
    diagram.
    """
    settings = {}
    for column in ("left", "right"):
        entries = diagram.runs[column]["outcar"].get("va", [])
        if not entries:
            continue
        settings[column] = (entries[0].get("sigma"), entries[0].get("rwall"))
        names = " ".join(dict.fromkeys(entry["name"] for entry in entries))
        print(f" {column:<5} point charges {names}: VACSIGMA "
              f"{entries[0].get('sigma', 0.0):g} A, VACRWALL "
              f"{entries[0].get('rwall', 0.0):g} A")
    values = set(settings.values())
    if len(values) > 1:
        print(" *** WARNING: the two sublattice runs were computed with "
              "DIFFERENT Va point-charge settings")
        print("     (VACSIGMA/VACRWALL above). They are not two halves of one "
              "diagram: on SrTiO3 a")
        print("     0.1 A change of the wall width moves a sublattice column "
              "by ~0.8 eV and reorders")
        print("     O 2p levels. Recompute both with the same settings. ***")
    elif values:
        sigma, rwall = next(iter(values))
        if rwall is not None and abs(rwall - VACRWALL_DEFAULT) > 1e-6:
            print(f"   (note: these runs use VACRWALL {rwall:g} A, while "
                  f"--vasp-setup writes {VACRWALL_DEFAULT:g} A; the fragment "
                  "energies are sensitive to it)")
        del sigma


def _report_column_scale(diagram, records, site=False) -> None:
    """Print the per-(element, shell) residuals of each column's shift.

    In ``site`` mode this is a RESIDUAL check on the drawn levels -- the
    per-element verdict has already been given by
    :func:`_report_site_alignment` -- so the column-wide WARNING, which asks
    whether ONE shift describes the column, is not repeated.
    """
    scale = diagram.column_scale_report(records)
    print("\n * Column scale check (residual E_crystal - E_fragment after the "
          "shift) *")
    for column in ("left", "right"):
        info = scale[column]
        if not info["groups"]:
            print(f"   {column:<5}: no fragment level has a crystal "
                  f"counterpart above {COLUMN_SCALE_PURITY:.2f}; not checked")
            continue
        parts = "  ".join(
            f"{key} {mean:+.2f}" + ("*" if forbidden else "")
            for key, mean, _spread, _n, forbidden, _probes in info["groups"])
        print(f"   {column:<5}: {parts}   (eV; * = a symmetry-forbidden "
              "probe is among them)")
        probes = [probe for row in info["groups"] for probe in row[5]]
        if probes:
            print("          symmetry-forbidden pairs (the fragment irrep has "
                  "no partner in the other")
            print("          sublattice at that k, so crystal and fragment "
                  "MUST coincide):")
            for delta, _forbidden, k_name, label in sorted(
                    probes, key=lambda item: -abs(item[0]))[:6]:
                print(f"            {k_name:<4} {label:<16} {delta:+7.2f} eV")
        if site:
            print(f"          spread over the shells {info['spread']:.2f} eV "
                  "(one shift per SITE was applied, not per column)")
        elif info["ok"]:
            print(f"          spread over the shells {info['spread']:.2f} eV "
                  f"<= {COLUMN_SCALE_TOL:g} eV: one rigid shift describes this "
                  "column")
        else:
            print(f"\n   *** WARNING: the {column} column is NOT on one energy "
                  f"scale with the crystal:")
            print(f"       its shells disagree by {info['spread']:.2f} eV, far "
                  f"more than the {COLUMN_SCALE_TOL:g} eV one rigid shift can "
                  "absorb.")
            print("       Cause: the removed sublattice is represented by BARE "
                  "point charges, which do not")
            print("       screen the remaining site the way the real valence "
                  "shell does -- on a cation")
            print("       sublattice the EMPTY d states come out ~10 eV too "
                  "deep while the semicore agrees.")
            print("       Consequence: every crystal level that draws on this "
                  "column is drawn WITHOUT a")
            print("       bonding/antibonding colour, and its parent energy in "
                  "the level table is not")
            print("       a pre-bonding energy. --vasp-anchor EL nl re-anchors "
                  "the column on another shell. ***")


def _site_lines(diagram, report) -> list:
    """The site-resolved anchor tables, as report lines (also reused by the JSON)."""
    lines = []
    for column in ("left", "right"):
        info = report[column]
        lines.append(f"   {column:<5} {diagram.formula[column]}"
                     f"   (column mean {info['mean']:+.3f} eV)")
        for element, entry in sorted(info["elements"].items()):
            if entry["fallback"]:
                lines.append(
                    f"     {element:<3} delta = {entry['delta']:+8.3f} eV   "
                    "NO ANCHOR: the column mean was used")
                continue
            verdict = ("shells agree" if entry["ok"]
                       else f"NOT one scale (> {ELEMENT_SCALE_TOL:g} eV)")
            fitted = ("the symmetry-forbidden probes ALONE"
                      if entry.get("fit") == "forbidden"
                      else "all anchors (NO symmetry-forbidden probe)")
            lines.append(
                f"     {element:<3} delta = {entry['delta']:+8.3f} eV   "
                f"{entry['n']} anchors ({entry['n_forbidden']} "
                f"symmetry-forbidden), shell spread {entry['spread']:.2f} eV "
                f"-> {verdict}")
            lines.append(f"        fitted to {fitted}")
            for shell, mean, spread, count, forbidden in entry["shells"]:
                lines.append(
                    f"        {shell:<8} {mean:+8.3f}  n={count}  "
                    f"spread {spread:5.2f}" + ("  *" if forbidden else ""))
            for shell in entry.get("shells_without_anchor", []):
                lines.append(
                    f"        {shell:<8} {'--':>8}  WARNING: this shell is on "
                    "the page but gave no anchor,")
                lines.append(
                    "                   so it does not enter the shell "
                    "spread above")
            if entry["forbidden_mean"] is not None:
                lines.append(
                    f"        symmetry-forbidden probes only: "
                    f"{entry['forbidden_mean']:+.3f} eV "
                    f"(n={entry['n_forbidden']}, spread "
                    f"{entry['forbidden_spread']:.2f} eV); after the shift "
                    f"they close to {entry['forbidden_residual']:.3f} eV")
            else:
                lines.append(
                    "        NO symmetry-forbidden probe for this element: "
                    "the fit rests on pure counterparts alone,")
                lines.append(
                    "          whose residuals average to zero BY "
                    "CONSTRUCTION -- this site can show scatter but no")
                lines.append(
                    "          net shift, so its delta carries the mean "
                    "bonding shift of its own anchors.")
            for (delta, forbidden, purity, label, k_name, crystal_label,
                 _element, _shell) in entry["anchors"]:
                residual = delta - entry["delta"]
                lines.append(
                    f"          {k_name:<4} {label:<16} "
                    f"{'forbidden' if forbidden else 'pure     '} "
                    f"w={purity:5.3f} -> {crystal_label:<12} "
                    f"{delta:+8.3f} eV   "
                    + (f"closes to {residual:+6.3f}" if forbidden
                       else f"bonding shift {residual:+6.3f}"))
            if entry.get("pure_residuals"):
                values = [row[0] for row in entry["pure_residuals"]]
                lines.append(
                    "        measured bonding shift of the pure counterparts "
                    f"(E_crystal - E_fragment - delta): mean "
                    f"{sum(values) / len(values):+.3f} eV, range "
                    f"{min(values):+.3f} .. {max(values):+.3f} eV over "
                    f"{len(values)} pair(s) -- this is chemistry, not scale, "
                    "and is now left ON the page")
        resolved = {element: entry["delta"]
                    for element, entry in info["elements"].items()
                    if not entry["fallback"]}
        if len(resolved) > 1:
            high = max(resolved, key=lambda key: resolved[key])
            low = min(resolved, key=lambda key: resolved[key])
            lines.append(
                f"     {high} - {low}: {resolved[high] - resolved[low]:+.2f} eV "
                "-- the site-potential change between the point-charge model "
                "and the")
            lines.append(
                "          self-consistent crystal: the site that receives "
                "covalent back-donation in the")
            lines.append(
                "          crystal has ALL its levels raised by the "
                "intra-atomic Coulomb repulsion the")
            lines.append("          bare point-charge fragment lacks.")
    return lines


def _report_site_alignment(diagram, shifts, anchors) -> None:
    """Print the site-resolved alignment: per-element deltas, anchors, verdicts."""
    report = diagram.site_report
    zero = {"vbm": f"E - E_VBM  (VBM {diagram.vbm:.3f} eV, highest occupied "
                   "crystal eigenvalue over all k of the crystal run)",
            "efermi": "E - E_F  (Fermi level of the crystal run, "
                      f"{diagram.runs['mo']['outcar'].get('efermi') or 0.0:.3f} eV)",
            "raw": "the crystal run's own G = 0 reference (--vasp-zero raw)",
            }.get(diagram.zero_mode, "")
    print("\n * Site-resolved alignment (reference = the crystal column, "
          "shift 0) *")
    print(f"   energy zero: {zero}")
    print("   one shift per (column, element): shells of one atom move "
          "together, different SITES do not.")
    print("   Anchors are symmetry-forbidden probes (a fragment irrep with no "
          "partner in the other")
    print("   sublattice at that k, so crystal and fragment MUST coincide; "
          f"weight x{ANCHOR_WEIGHT_FORBIDDEN:g}) and pure")
    print(f"   counterparts (absolute composition >= "
          f"{COLUMN_SCALE_PURITY:.2f}), inside "
          f"VBM + {ANCHOR_WINDOW_EV:g} eV.")
    for line in _site_lines(diagram, report):
        print(line)
    missing = [(column, element) for column in ("left", "right")
               for element in report[column]["missing"]]
    if missing:
        print("\n   *** WARNING: no anchor was found for "
              + ", ".join(f"{element} ({column})" for column, element in missing)
              + ".")
        print("       Those levels inherit their column's mean shift, which "
              "assumes the site behaves")
        print("       like the column average. Widen --vasp-window's data "
              "(the anchor pool is the")
        print(f"       default VBM + {ANCHOR_WINDOW_EV:g} eV window) or "
              "re-run with --vasp-align rigid. ***")
    failed = [(column, element)
              for column in ("left", "right")
              for element, entry in sorted(report[column]["elements"].items())
              if not entry["ok"]]
    if failed:
        print("\n   *** WARNING: "
              + ", ".join(f"{element} ({column})" for column, element in failed)
              + " is not described by ONE shift:")
        print("       its shells disagree by more than "
              f"{ELEMENT_SCALE_TOL:g} eV. Every crystal level with a parent on "
              "that site is")
        print("       drawn WITHOUT a bonding/antibonding colour. ***")
    else:
        print(f"\n   every site's shells agree within "
              f"{ELEMENT_SCALE_TOL:g} eV: bond character is classified.")
    # the previous single-shift-per-column alignment, kept as a diagnostic
    rigid = shifts.get("rigid") or {}
    print("\n   diagnostic, the RIGID alignment (--vasp-align rigid): "
          f"left {rigid.get('left', 0.0):+.3f} eV | "
          f"right {rigid.get('right', 0.0):+.3f} eV")
    for column in ("left", "right"):
        info = anchors.get(column)
        if info:
            print(f"     {column:<5} anchored on {info['label']} "
                  f"({info['fragment_energy']:.2f} eV, counterpart purity "
                  f"{100 * info['purity']:.0f}%, spread {info['spread']:.3f} eV "
                  f"over {_anchor_extent(info)})")


def _alignment_chips(diagram, shifts) -> list:
    """The two header chips that say how the columns were put on one scale."""
    if shifts is None:
        return ["no alignment (--no-align): three raw G = 0 references"]
    zero = {"vbm": "E &minus; E<sub>VBM</sub>",
            "efermi": "E &minus; E<sub>F</sub>",
            "raw": "raw (crystal G = 0)"}.get(diagram.zero_mode, "raw")
    if shifts["mode"] != "site":
        return [f"rigid alignment: left {shifts['rigid']['left']:+.2f} / "
                f"right {shifts['rigid']['right']:+.2f} eV", zero]
    parts = []
    for column in ("left", "right"):
        for element, entry in sorted(
                diagram.site_report[column]["elements"].items()):
            parts.append(f"{element} {entry['delta']:+.2f}")
    return ["site-resolved alignment: " + ", ".join(parts) + " eV", zero]


def _setting_chip(diagram) -> str:
    """The header chip that says which setting the irrep labels refer to."""
    from .vasp_io import format_shift

    if getattr(diagram, "setting_note", ""):
        return ("setting: the crystal run's own (the -c cell is another "
                "basis of the same crystal)")
    moved = [mapping for mapping in diagram.mappings.values()
             if not mapping.identity]
    if not moved:
        return ""
    source = os.path.basename(getattr(diagram, "cell_path", None) or "") or "-c"
    shifts = {format_shift(mapping.shift) for mapping in moved}
    if len(shifts) == 1 and len(moved) == len(diagram.mappings):
        return (f"setting: {source}; the runs are that cell shifted by "
                f"{shifts.pop()}")
    return (f"setting: {source}; the runs are mapped onto that cell "
            "(see the level table)")


def _report_projection_floor(diagram) -> None:
    """Show how close the projection-weight cut is to the nearest levels."""
    below = sorted((value for value, _c in diagram.floor_margin
                    if value < diagram.projection_floor), reverse=True)[:3]
    above = sorted(value for value, _c in diagram.floor_margin
                   if value >= diagram.projection_floor)[:3]
    if not below and not above:
        return
    print(f"\n * Projection floor {diagram.projection_floor:g} "
          "(--vasp-projection-floor) *")
    print("   dropped just below: "
          + (", ".join(f"{value:.3f}" for value in below) or "-")
          + "   kept just above: "
          + (", ".join(f"{value:.3f}" for value in above) or "-"))
    if below and above and above[0] < 1.5 * below[0]:
        print("   the two populations are NOT well separated here; which "
              "weakly projected")
        print("   levels appear depends on this threshold -- vary it to test "
              "the sensitivity")


def _report_above_window(diagram, records) -> None:
    """Name the fragment levels the drawn window cuts off the top.

    The window is a view, not physics (the anchors are always chosen inside
    the default one), but an empty, diffuse cation d manifold lands within a
    few tenths of an eV of the default ``VBM + 10 eV`` cut and can fall out at
    ONE k point only -- on CaF2 the Ca 3d manifolds are at +10.05 and
    +10.35 eV at GM and at 8.2-9.9 eV at L, X and W, so the cation column
    vanishes at GM alone and the LUMO is drawn parentless while the other
    three pages look correct.  That is worth a line.
    """
    rows = [(record["name"], record.get("above_window") or [])
            for record in records]
    if not any(entries for _name, entries in rows):
        return
    high = max(entry[0] for _name, entries in rows for entry in entries)
    print("\n * Fragment levels above the drawn window *")
    for name, entries in rows:
        if not entries:
            continue
        shown = ", ".join(f"{label} ({energy:+.2f})"
                          for energy, label in entries[:4])
        more = f" ... (+{len(entries) - 4} more)" if len(entries) > 4 else ""
        print(f"   {name:<4} {len(entries):>3} level(s) lie above the window: "
              f"{shown}{more}")
    print(f"   The highest is at {high:+.2f} eV; --vasp-window "
          f"{diagram.window[0] if diagram.window[0] is not None else -42:g} "
          f"{high - diagram.vbm - getattr(diagram, 'crystal_shift', 0.0) + 1:.0f}"
          " would show them all.")
    print("   They are NOT in the anchor pool either way: the alignment is "
          "always fitted inside")
    print(f"   the default VBM + {DEFAULT_WINDOW[1]:g} eV window, so widening "
          "the drawing moves no energy.")


def _residual_sentence(residual) -> str:
    """One sentence quoting the measured alignment residual of this system."""
    if not residual:
        return ""
    k_name, irrep, _n, value = residual["worst"]
    return (
        f"Measured alignment residual of THIS system: {residual['mean']:.2f} eV "
        f"mean and {residual['max']:.2f} eV max over {residual['n']} complete "
        f"(k, irrep) manifolds (worst {k_name} {irrep}, {value:+.2f} eV) -- the "
        "centre-of-gravity sum rule, which uses no projection and no anchor. "
        "Read the deadband against THAT number: a colour whose "
        "|E &minus; E<sub>parents</sub>| is smaller than the mean residual is "
        "inside the noise, and the level's own panel says so.  ")


def _report_manifold_residual(diagram) -> None:
    """Print the measured, model-free alignment residual of the run at hand."""
    residual = getattr(diagram, "manifold_residual", None)
    if not residual:
        return
    k_name, irrep, _n, value = residual["worst"]
    print("\n * Measured alignment residual (centre of gravity of every "
          "complete (k, irrep) manifold) *")
    print(f"   {residual['n']} complete manifolds: mean |residual| "
          f"{residual['mean']:.3f} eV, max {residual['max']:.3f} eV "
          f"({k_name} {irrep}, {value:+.3f} eV)")
    print("   Uses no projection and no anchor: where crystal and fragments "
          "carry the same number of")
    print("   states of one irrep, their centres of gravity must coincide. "
          "This is the error bar the")
    print(f"   {BOND_ENERGY_TOL:g} eV bond deadband should be read against.")
    if residual["mean"] > BOND_ENERGY_TOL:
        print(f"   NOTE: it is LARGER than the deadband, so a colour with "
              f"|E - E_parents| < {residual['mean']:.2f} eV is inside the "
              "noise; those levels are")
        print("   flagged in the click panel, the level table and the JSON.")
    worst_rows = sorted(residual["rows"], key=lambda row: -abs(row[3]))[:5]
    for k_name, irrep, count, value in worst_rows:
        print(f"     {k_name:<4} {irrep:<6} n={count:<3} {value:+7.3f} eV")


def _cell_formula(symbols) -> str:
    """``"SrTiO3"`` from the primitive cell's symbols, in order of appearance."""
    counts: dict = {}
    for symbol in symbols:
        counts[symbol] = counts.get(symbol, 0) + 1
    return "".join(f"{symbol}{count if count > 1 else ''}"
                   for symbol, count in counts.items())


def _same_crystal(cell, other, symprec) -> bool:
    """Do two structures describe the same crystal?

    Composition (per primitive cell), space-group number and primitive volume:
    enough to tell "the same crystal written on another origin, which the
    atom-by-atom mapping merely failed to see" from "a different compound",
    without pulling pymatgen in for a full structure match.
    """
    builders = []
    for structure in (cell, other):
        builder = SymmetryAdaptedOrbitalBasis(cell=structure, symprec=symprec)
        builders.append(builder)
    left, right = builders
    if (sorted(get_chemical_symbols(left.primitive_cell))
            != sorted(get_chemical_symbols(right.primitive_cell))):
        return False
    if (left.spglib_dataset["number"] != right.spglib_dataset["number"]):
        return False
    volumes = [abs(np.linalg.det(np.array(builder.primitive_cell.cell)))
               for builder in builders]
    return abs(volumes[0] - volumes[1]) <= 0.01 * max(volumes)


def report_and_write(cell, *, left, right, symprec, electrons, kpoint_filter,
                     output_path, structure_label, oxidation=None, root=".",
                     overrides=None, window=None, align=True,
                     conventional=False, degeneracy_tol=None,
                     projection_floor=None, anchor=None, align_mode="site",
                     zero="vbm", resolved=None, cell_path=None):
    """Terminal report, level table and HTML for the VASP crystal-orbital diagram."""
    from .star_of_k import read_poscar_or_exit
    from .vasp_io import StructureMappingError

    if resolved is None:
        print("\n * VASP runs *")
        resolved = _resolve_directories(root, overrides or {})
    diagram = None
    # the fragment assignment is needed to sort the sublattice directories, so
    # build a throwaway assignment first
    from .crystal_orbital_pyscf import PySCFCrystalOrbitalDiagram

    class _Assign:
        pass

    probe = _Assign()
    probe.symbols = get_chemical_symbols(
        SymmetryAdaptedOrbitalBasis(cell=cell, symprec=symprec).primitive_cell)
    PySCFCrystalOrbitalDiagram._assign_fragments(probe, left, right)
    directories = _sort_sublattices(resolved, probe.element_column)

    def build(structure):
        return VASPCrystalOrbitalDiagram(
            structure, left, right, directories=directories, symprec=symprec,
            electrons=electrons, oxidation=oxidation, window=window,
            conventional=conventional, degeneracy_tol=degeneracy_tol,
            projection_floor=projection_floor, anchor=anchor,
            align_mode=align_mode, zero=zero)

    crystal_poscar = os.path.join(directories["mo"], "POSCAR")
    setting_note = ""
    try:
        diagram = build(cell)
    except StructureMappingError as error:
        # the run is not the -c cell in another setting.  When it is the same
        # CRYSTAL all the same (another basis, another primitive cell), the
        # analysis can still be done in the crystal run's OWN setting -- the
        # labels then refer to that setting, which is said here, in the
        # tables and on the page.  When it is not, there is nothing to fall
        # back to and the run stops with what differs.
        own = read_poscar_or_exit(crystal_poscar)
        second = None
        if _same_crystal(cell, own, symprec):
            try:
                diagram = build(own)
            except StructureMappingError as retry:
                second = retry
        if diagram is None:
            if second is not None:
                raise SystemExit(
                    f"ERROR: {error.reason}, and the crystal run's own "
                    f"setting does not work either ({second.reason}).") from None
            raise SystemExit(
                f"ERROR: {error.reason}; run it in the crystal run's own "
                f"setting with -c {crystal_poscar}.") from None
        setting_note = (
            f"the -c cell and {crystal_poscar} are the same crystal in "
            f"DIFFERENT BASES ({error.reason}), so the analysis is done in "
            "the crystal run's own setting and every irrep label refers to "
            "it, not to the -c file")
        print(f"\n * Setting *\n   {setting_note}")

    dataset = diagram.builder.spglib_dataset
    diagram.setting_note = setting_note
    diagram.cell_path = cell_path
    print("\n * Space group *")
    print(f" {dataset['international']} ({dataset['number']})\n")
    setting_lines = diagram.setting_lines()
    if setting_lines:
        print(" * Setting *")
        for line in setting_lines:
            print(f" {line}")
        print(f" irrep labels refer to the {cell_path or '-c'} setting\n")
    if output_path is None or structure_label is None:
        formula = _cell_formula(diagram.symbols)
        if structure_label is None:
            structure_label = f"{formula}, {dataset['international']}"
        if output_path is None:
            output_path = (f"CrystOD_{formula}_"
                           f"{dataset['international'].replace('/', '')}"
                           "_vasp.html")

    print(" * Columns *")
    for column in ("left", "right"):
        outcar = diagram.runs[column]["outcar"]
        charges = ", ".join(
            f"{entry['name']} x{entry['ions']}" for entry in outcar.get("va", []))
        print(f" {column:<5} {diagram.formula[column]:<8} charge "
              f"{diagram.side_charge[column]:+g}, "
              f"{diagram.side_electrons[column]} electrons (NELECT), "
              f"point charges {charges or 'none'}")
    print(f" crystal {diagram.formula['left'] + diagram.formula['right']:<7} "
          f"{diagram.crystal_electrons} electrons "
          f"= {diagram.side_electrons['left']} + "
          f"{diagram.side_electrons['right']}")
    outcar = diagram.runs["mo"]["outcar"]
    print(f" functional {diagram.functional}, ENCUT "
          f"{outcar.get('encut') or 0:g} eV, "
          f"E-fermi {outcar.get('efermi') or 0:.3f} eV (crystal run), "
          f"VBM {diagram.vbm:.3f} eV")
    print(" projection basis: (atom, l, m) from the LORBIT=12 PROCAR, "
          f"{diagram.n_ao} components ({' '.join(diagram.lm_names)})")
    print(" every run pins its own G=0 average potential to zero, so the raw\n"
          " columns are offset by one constant each -- removed below by\n"
          " deep-level alignment")
    _report_point_charge_settings(diagram)

    kpoints = diagram.special_kpoints()
    if kpoint_filter is not None:
        available = [name for name, _ in kpoints]
        kpoints = [(name, k) for name, k in kpoints if name == kpoint_filter]
        if not kpoints:
            raise SystemExit(
                f"ERROR: k point '{kpoint_filter}' is not a special point of this "
                f"space group (available: {', '.join(available)}).")

    records = []
    warnings = []
    for name, kpoint in kpoints:
        levels, _labels = diagram.solve_at(kpoint)
        irreps, _mapping, labels, representation = diagram.little_group_data(kpoint)
        warnings.extend((name, *entry) for entry in diagram.last_purity_warnings)
        records.append({
            "name": name, "kpoint": kpoint, "levels": levels,
            "content": diagram.site_symmetry_irreps(
                kpoint, representation, irreps, labels),
        })

    shifts, anchors = (None, {})
    if align:
        shifts, anchors = diagram.align_fragment_columns(records)
        if shifts is None:
            print("\n * Deep-level alignment (pre-bonding reference) *")
            print("   no chemically inert fragment level found; columns left "
                  "on their raw G=0 references")
        elif shifts["mode"] == "site":
            _report_site_alignment(diagram, shifts, anchors)
        else:
            print("\n * Rigid alignment (--vasp-align rigid, reference = the "
                  "crystal column) *")
            if diagram.anchor_note:
                print(diagram.anchor_note)
            print(f"   energy zero: --vasp-zero {diagram.zero_mode} "
                  f"({shifts['zero']:+.3f} eV on every column)")
            for column in ("left", "right"):
                info = anchors.get(column)
                if info:
                    print(f"   {column:<5} delta = "
                          f"{shifts['rigid'][column]:+.3f} eV, anchored on "
                          f"{info['label']} ({info['fragment_energy']:.2f} eV, "
                          f"counterpart purity {100 * info['purity']:.0f}%, "
                          f"spread {info['spread']:.3f} eV over "
                          f"{_anchor_extent(info)})")
            print(f"   shifts : left {shifts['left']:+.3f} eV | "
                  f"crystal {shifts['mo']:+.3f} eV | "
                  f"right {shifts['right']:+.3f} eV")
        if shifts is not None:
            # ALIGNMENT_PURITY is a requirement, not a preference: an anchor
            # that only reaches it through the "or pairs" fallback carries the
            # chemistry of a mixed level into the column offset, and nothing
            # downstream can tell.  Say so loudly rather than in a parenthesis.
            weak = [column for column in ("left", "right")
                    if anchors.get(column) and anchors[column]["fallback"]]
            if weak:
                print("\n   *** WARNING: no fragment level of the "
                      + " and ".join(weak) + " column reaches the "
                      f"{100 * _alignment_purity():.0f}% purity the rigid "
                      "anchor rule requires.")
                for column in weak:
                    info = anchors[column]
                    print(f"       {column}: best candidate {info['label']} at "
                          f"{100 * info['purity']:.0f}% -- that column's offset "
                          "inherits the chemistry of a mixed level.")
                print("       Re-run with --vasp-anchor EL nl to pin the "
                      "column on a shell you trust, or with --no-align to see "
                      "the raw per-run scales. ***")
            trace = diagram.trace_diagnostic(records, shifts)
            if trace is None:
                print("   trace diagnostic: not available (the occupied "
                      "electron counts of the crystal and of the two "
                      "fragments do not match)")
            else:
                print(f"   trace diagnostic: the occupied-band sum rule "
                      f"({trace['electrons']} electrons over the "
                      f"{len(records)} k points) is off by "
                      f"{trace['trace_shift']:+.3f} eV per electron, i.e. both "
                      "fragment columns\n"
                      "     together would move by that much -- an independent "
                      "estimate that uses no projection")
            diagram.crystal_shift = shifts["mo"]
            # remember, while the complete level set is still there, what each
            # crystal level's parent energy would be with nothing hidden
            diagram.record_full_parents(records)
    else:
        print("\n * Deep-level alignment disabled (--no-align): each run keeps "
              "its own G=0 reference *")
    # The window is a VIEW, applied only now that the alignment has chosen its
    # anchors on the complete level set.  On the raw references the fragment
    # scales mean nothing, so --no-align cuts only the crystal column.
    diagram.apply_window(records, fragments=shifts is not None)
    # Is each fragment column really on ONE scale with the crystal?  Asked of
    # the levels the page will SHOW, so that the verdict describes the drawing.
    if shifts is not None:
        _report_column_scale(diagram, records,
                             site=shifts["mode"] == "site")
    else:
        diagram.column_scale_ok = {"left": False, "right": False}
        print("   (bond character is not classified while the three columns "
              "sit on their own energy references)")
    diagram.manifold_residual = (diagram.manifold_residuals(records)
                                 if shifts is not None else None)
    _report_manifold_residual(diagram)
    diagram.assign_bond_characters(records)
    _report_projection_floor(diagram)
    _report_above_window(diagram, records)

    # only the levels that survived the window are on the page, so only their
    # purity is worth warning about
    warnings = [entry for entry in warnings
                if entry[2] in {level.label for record in records
                                if record["name"] == entry[0]
                                for level in record["levels"][entry[1]]}]
    if warnings:
        print("\n * Irrep purity warnings *")
        for k_name, column, label, energy, purity in warnings[:20]:
            print(f"   {k_name} {column:<5} {label:<18} {energy:9.3f} eV  "
                  f"best irrep weight {purity:.3f} < {IRREP_PURITY_WARN}")
    else:
        print(f"\n * Irrep purity: every labelled level is above "
              f"{IRREP_PURITY_WARN} *")

    for record in records:
        name, kpoint, levels = record["name"], record["kpoint"], record["levels"]
        print(f"\n * k point {name} {_format_kpoint(kpoint)} *")
        print("   site-symmetry induced representations (SALC decomposition):")
        for (element, shell), parts in record["content"].items():
            if parts:
                print(f"     {element} {shell:<4} = {' + '.join(parts)}")
        for column in ("left", "right"):
            parts = ", ".join(
                f"{level.label} ({level.energy:.2f})"
                for level in sorted(levels[column], key=lambda lv: lv.energy))
            print(f"   {diagram.formula[column]:<10}: {parts}")
        print("   crystal   :")
        for level in sorted(levels["mo"], key=lambda lv: lv.energy):
            composition = "  ".join(
                f"{label} {100 * weight:.1f}%"
                for label, weight in getattr(level, "display_composition", []))
            occupancy = f"{level.electrons}e" if level.electrons else "  "
            print(f"     {level.label:<10} {level.energy:9.2f} eV  "
                  f"x{level.degeneracy}  {occupancy:<4} {composition}")
        print_dipole_rules(diagram, name, kpoint, levels)

    tally: dict = {}
    for record in records:
        for level in record["levels"]["mo"]:
            key = getattr(level, "bond_character", None) or "not classified"
            tally[key] = tally.get(key, 0) + 1
    print("\n * Bond character of the drawn crystal levels *")
    print("   " + ", ".join(f"{count} {key}" for key, count
                            in sorted(tally.items(), key=lambda kv: -kv[1])))
    kinds: dict = {}
    noisy = 0
    for record in records:
        for level in record["levels"]["mo"]:
            if getattr(level, "bond_character", None) is None:
                kind = getattr(level, "unclassified_kind", "") or "other"
                kinds[kind] = kinds.get(kind, 0) + 1
            if getattr(level, "inside_residual", False):
                noisy += 1
    if kinds:
        print("   why the neutral ones are neutral: "
              + ", ".join(f"{count} {key}" for key, count
                          in sorted(kinds.items(), key=lambda kv: -kv[1])))
    if noisy:
        print(f"   {noisy} coloured level(s) sit closer to their parents than "
              "the measured alignment residual of this system,")
        print("   i.e. inside the noise; each says so in its own panel, in "
              "the level table and in the JSON.")

    stem = output_path[:-len(".html")] if output_path.endswith(".html") \
        else output_path
    table_path = stem + "_levels.txt"
    json_path = stem + "_levels.json"
    existed = os.path.isfile(output_path)
    _write_level_table(diagram, records, table_path, shifts)
    _write_level_json(diagram, records, json_path, shifts)

    diagram.bond_foot = diagram.bond_foot.replace(
        "MEASURED_RESIDUAL",
        _residual_sentence(getattr(diagram, "manifold_residual", None)))
    for line in diagram.setting_lines():
        diagram.foot_intro += (
            f"  SETTING: {line}; the irrep labels refer to the "
            f"{cell_path or '-c'} setting.")
    if setting_note:
        diagram.foot_intro += f"  SETTING: {setting_note}."
    diagram.extra_chips = _alignment_chips(diagram, shifts)
    setting_chip = _setting_chip(diagram)
    if setting_chip:
        diagram.extra_chips.append(setting_chip)
    entries = [(record["name"], record["kpoint"], record["levels"])
               for record in records]
    write_crystal_diagram_html(diagram, entries, output_path, structure_label)
    print(f"\nCrystal-orbital diagram {'OVERWRITTEN' if existed else 'written'}"
          f": {output_path}")
    print(f"Level table {'overwritten' if existed else 'written'}: {table_path}")
    print(f"Level JSON  {'overwritten' if existed else 'written'}: {json_path}")
    # the output name comes from the CRYSTAL directory alone, so two runs that
    # differ only in --vasp-left/--vasp-right land on the same file
    if any((overrides or {}).get(column) for column in ("mo", "left", "right")):
        print(" (the name is taken from the crystal run, so this run with "
              "--vasp-crystal/-left/-right replaced\n"
              "  whatever the default run had written here; use --output NAME "
              "to keep both)")
    return diagram


def _setting_table_lines(diagram) -> list:
    """``# setting`` header lines, one per run that is not in the -c setting."""
    lines = []
    note = getattr(diagram, "setting_note", "")
    source = getattr(diagram, "cell_path", None) or "-c"
    if note:
        lines.append(f"# setting  {note}\n")
        return lines
    for line in diagram.setting_lines():
        lines.append(f"# setting  {line}\n")
    if lines:
        lines.append(f"#          irrep labels refer to the {source} "
                     "setting\n")
    return lines


def _setting_json(diagram) -> dict:
    """The mapping of the three runs onto the analysis setting, for the JSON."""
    from .vasp_io import format_shift

    runs = {}
    for column in ("mo", "left", "right"):
        mapping = diagram.mappings.get(column)
        if mapping is None:
            continue
        runs[column] = {
            "directory": diagram.runs[column]["dir"],
            "identity": bool(mapping.identity),
            "shift": [round(float(value), 8) for value in mapping.shift],
            "shift_label": format_shift(mapping.shift),
            "ions": [int(ion) + 1 for ion in mapping.ions],
            "wraps": mapping.wraps.tolist(),
            "lattice_residual": round(float(mapping.lattice_residual), 10),
        }
    return {
        "cell": getattr(diagram, "cell_path", None),
        "note": getattr(diagram, "setting_note", "") or None,
        "identity": all(entry["identity"] for entry in runs.values()),
        "runs": runs,
    }


def _write_level_table(diagram, records, path, shifts) -> None:
    """Every number the diagram shows, as one plain-text table."""
    with open(path, "w") as handle:
        handle.write(
            "# CrystOD --diagram --vasp: all levels of the three columns\n"
            "# E_raw    the run's own eigenvalue (its own G=0 reference)\n"
            "# E_align  on the page's scale: the crystal column is the\n"
            "#          reference, each fragment level is raised by its own\n"
            "#          site shift, and --vasp-zero moves all three together\n"
            "# shift    E_align - E_raw of that level (0 for a crystal level\n"
            "#          apart from the --vasp-zero offset)\n"
            "# purity   weight of the assigned irrep in the level's "
            "projection\n"
            "# w_proj   projected weight per degenerate partner (PAW spheres; "
            f"levels below {PROJECTION_FLOOR:g} are not drawn)\n"
            "# shares   per (element, shell) share of the level's total "
            "projected weight\n"
            "# links    connector weights into the fragment levels\n")
        for line in _setting_table_lines(diagram):
            handle.write(line)
        if shifts:
            handle.write(
                f"# mode     {shifts['mode']} alignment, reference column "
                f"{shifts['reference']}, energy zero "
                f"--vasp-zero {shifts['zero_mode']} "
                f"({shifts['zero']:+.4f} eV)\n"
                f"# columns  left {shifts['left']:+.4f} | crystal "
                f"{shifts['mo']:+.4f} | right {shifts['right']:+.4f} eV"
                + ("  (weighted means; the applied shifts are per element "
                   "below)\n" if shifts["mode"] == "site" else "\n"))
            handle.write(f"# rigid    left {shifts['rigid']['left']:+.4f} | "
                         f"right {shifts['rigid']['right']:+.4f} eV "
                         "(the one-shift-per-column diagnostic)\n")
        residual = getattr(diagram, "manifold_residual", None)
        if residual:
            k_name, irrep, _n, value = residual["worst"]
            handle.write(
                f"# residual measured alignment residual of this system: "
                f"{residual['mean']:.4f} eV mean, {residual['max']:.4f} eV max "
                f"({k_name} {irrep})\n"
                f"#          over {residual['n']} complete (k, irrep) "
                "manifolds (centre-of-gravity sum rule, no projection);\n"
                f"#          the bond deadband is {BOND_ENERGY_TOL:g} eV, so "
                "read the colours against the larger of the two\n")
        if shifts and shifts["mode"] == "site" and diagram.site_report:
            handle.write("#\n# Site-resolved anchor table "
                         "(delta = E_crystal - E_fragment, raw scales)\n")
            for line in _site_lines(diagram, diagram.site_report):
                handle.write(f"# {line.rstrip()}\n")
        handle.write(
            "#\n# k      column  level       irrep   deg  purity  E_raw      "
            "E_align    shift      occ  w_proj  label\n")
        names = {}
        for record in records:
            for column in ("left", "right"):
                for level in record["levels"][column]:
                    names[(record["name"], level.level_id)] = level.label
        for record in records:
            handle.write(f"\n# k point {record['name']} "
                         f"{_format_kpoint(record['kpoint'])}\n")
            for column in ("left", "mo", "right"):
                for level in sorted(record["levels"][column],
                                    key=lambda lv: lv.energy):
                    handle.write(
                        f"{record['name']:<7} {column:<7} {level.level_id:<11} "
                        f"{level.irrep:<7} {level.degeneracy:<4} "
                        f"{getattr(level, 'irrep_purity', 0.0):6.3f}  "
                        f"{level.raw_energy:10.4f} {level.energy:10.4f} "
                        f"{getattr(level, 'site_shift', 0.0):+10.4f} "
                        f"{level.electrons:<4} "
                        f"{getattr(level, 'projected_weight', 0.0):6.3f}  "
                        f"{level.label}\n")
                    if column != "mo":
                        element = getattr(level, "site_element", None)
                        handle.write(
                            f"{'':>8}site shift: {getattr(level, 'site_shift', 0.0):+.4f} eV"
                            + (f" (site {element})" if element else "")
                            + "\n")
                        note = getattr(level, "site_note", "")
                        if note:
                            handle.write(f"{'':>8}note: {note}\n")
                    shares = "  ".join(
                        f"{label} {100 * weight:.1f}%" for label, weight
                        in getattr(level, "display_composition", []))
                    if shares:
                        handle.write(f"{'':>8}shares: {shares}\n")
                    if column == "mo":
                        side = getattr(level, "side_weights", {})
                        if side:
                            handle.write(
                                f"{'':>8}fragment shares: left "
                                f"{100 * side.get('left', 0.0):.1f}%  right "
                                f"{100 * side.get('right', 0.0):.1f}%  "
                                f"({getattr(level, 'bond_character', None) or 'not classified'})\n")
                        links = "  ".join(
                            f"{names.get((record['name'], i), i)} "
                            f"{100 * w:.1f}%"
                            for i, w in sorted(level.composition,
                                               key=lambda kv: -kv[1]))
                        if links:
                            handle.write(f"{'':>8}links: {links}\n")
                        covered = getattr(level, "parent_coverage", 1.0)
                        if covered < 0.98:
                            handle.write(
                                f"{'':>8}not drawn: "
                                f"{100 * (1.0 - covered):.1f}% of this level's "
                                "character has no fragment level on the page\n")
                        full = getattr(level, "full_parent_energy", None)
                        parent = getattr(level, "parent_energy", None)
                        if (full is not None and parent is not None
                                and abs(full - parent) > 1e-6):
                            handle.write(
                                f"{'':>8}parent energy: {parent:.4f} eV over "
                                f"the drawn parents, {full:.4f} eV over all of "
                                "them\n")
                        if getattr(level, "inside_residual", False):
                            handle.write(
                                f"{'':>8}inside the noise: |E - E_parents| is "
                                "below the measured alignment residual of this "
                                "system\n")


def _write_level_json(diagram, records, path, shifts) -> None:
    """The same content as the level table, machine-readable.

    One JSON object: the run metadata, the alignment (mode, energy zero, the
    per-element site shifts with every anchor behind them, the rigid
    diagnostic) and one entry per level of every column at every k point,
    carrying its raw and aligned energy, the shift applied to it, its irrep,
    purity, shares, fragment shares, connectors and bond character.
    """
    import json

    def anchor_json(entry):
        return {
            "delta": round(entry["delta"], 6),
            "fit": entry.get("fit", "all"),
            "forbidden_residual": round(entry.get("forbidden_residual", 0.0), 6),
            "pure_residuals": [
                {"residual": round(value, 6), "kpoint": k_name,
                 "fragment": label, "crystal": crystal_label, "shell": shell}
                for value, k_name, label, crystal_label, shell
                in entry.get("pure_residuals", [])],
            "n": entry["n"], "n_forbidden": entry["n_forbidden"],
            "shell_spread": round(entry["spread"], 6),
            "anchor_spread": round(entry["spread_all"], 6),
            "forbidden_mean": (None if entry["forbidden_mean"] is None
                               else round(entry["forbidden_mean"], 6)),
            "forbidden_spread": round(entry["forbidden_spread"], 6),
            "ok": entry["ok"], "no_anchor": entry["fallback"],
            "shells_without_anchor": list(
                entry.get("shells_without_anchor", [])),
            "shells": [{"shell": shell, "delta": round(mean, 6),
                        "spread": round(spread, 6), "n": count,
                        "has_forbidden": bool(forbidden)}
                       for shell, mean, spread, count, forbidden
                       in entry["shells"]],
            "anchors": [{"delta": round(delta, 6), "forbidden": bool(forbidden),
                         "purity": round(purity, 6), "fragment": label,
                         "kpoint": k_name, "crystal": crystal_label,
                         "element": element, "shell": shell}
                        for (delta, forbidden, purity, label, k_name,
                             crystal_label, element, shell)
                        in entry["anchors"]],
        }

    document = {
        "engine": "crystod --diagram --vasp",
        "structure": {
            "formula": diagram.formula["left"] + diagram.formula["right"],
            "space_group": diagram.builder.spglib_dataset["international"],
            "space_group_number":
                int(diagram.builder.spglib_dataset["number"]),
        },
        "method": {
            "functional": diagram.functional,
            "encut": diagram.runs["mo"]["outcar"].get("encut"),
            "efermi": diagram.runs["mo"]["outcar"].get("efermi"),
            "vbm": round(diagram.vbm, 6),
            "electrons": diagram.crystal_electrons,
            "projection_floor": diagram.projection_floor,
        },
        "columns": {column: diagram.formula[column]
                    for column in ("left", "right")},
        "setting": _setting_json(diagram),
        "alignment": None,
        "kpoints": [],
    }
    if shifts:
        document["alignment"] = {
            "mode": shifts["mode"],
            "reference": shifts["reference"],
            "zero_mode": shifts["zero_mode"],
            "zero_offset": round(shifts["zero"], 6),
            "column_shift": {key: round(shifts[key], 6)
                             for key in ("left", "mo", "right")},
            "rigid": {key: round(value, 6)
                      for key, value in shifts["rigid"].items()},
            "element_scale_tol": ELEMENT_SCALE_TOL,
            "bond_deadband": BOND_ENERGY_TOL,
            "parent_coverage_floor": PARENT_COVERAGE_FLOOR,
            "measured_residual": (
                None if not getattr(diagram, "manifold_residual", None) else {
                    "n_manifolds": diagram.manifold_residual["n"],
                    "mean_abs": round(diagram.manifold_residual["mean"], 6),
                    "max_abs": round(diagram.manifold_residual["max"], 6),
                    "manifolds": [
                        {"kpoint": k_name, "irrep": irrep, "n": count,
                         "residual": round(value, 6)}
                        for k_name, irrep, count, value
                        in diagram.manifold_residual["rows"]],
                }),
            "sites": ({column: {element: anchor_json(entry)
                                for element, entry
                                in diagram.site_report[column]["elements"].items()}
                       for column in ("left", "right")}
                      if shifts["mode"] == "site" and diagram.site_report
                      else None),
        }
    for record in records:
        names = {level.level_id: level.label
                 for column in ("left", "right")
                 for level in record["levels"][column]}
        entry = {"name": record["name"],
                 "kpoint": [float(value) for value in record["kpoint"]],
                 "levels": []}
        for column in ("left", "mo", "right"):
            for level in sorted(record["levels"][column],
                                key=lambda lv: lv.energy):
                item = {
                    "column": column, "id": level.level_id,
                    "label": level.label, "irrep": level.irrep,
                    "degeneracy": level.degeneracy,
                    "irrep_purity": round(
                        float(getattr(level, "irrep_purity", 0.0)), 6),
                    "energy_raw": round(float(level.raw_energy), 6),
                    "energy": round(float(level.energy), 6),
                    "shift": round(float(getattr(level, "site_shift", 0.0)), 6),
                    "electrons": int(level.electrons),
                    "projected_weight": round(
                        float(getattr(level, "projected_weight", 0.0)), 6),
                    "shares": [[label, round(float(weight), 6)] for label, weight
                               in getattr(level, "display_composition", [])],
                }
                if column == "mo":
                    item["fragment_shares"] = {
                        key: round(float(value), 6) for key, value
                        in getattr(level, "side_weights", {}).items()}
                    item["bond_character"] = getattr(
                        level, "bond_character", None)
                    parent = getattr(level, "parent_energy", None)
                    item["parent_energy"] = (None if parent is None
                                             else round(float(parent), 6))
                    item["parent_coverage"] = round(
                        float(getattr(level, "parent_coverage", 1.0)), 6)
                    full = getattr(level, "full_parent_energy", None)
                    item["parent_energy_unwindowed"] = (
                        None if full is None else round(float(full), 6))
                    item["inside_residual"] = bool(
                        getattr(level, "inside_residual", False))
                    item["unclassified_kind"] = (
                        getattr(level, "unclassified_kind", "") or None)
                    item["links"] = [[names.get(i, i), round(float(w), 6)]
                                     for i, w in sorted(level.composition,
                                                        key=lambda kv: -kv[1])]
                else:
                    item["site"] = getattr(level, "site_element", None)
                    if getattr(level, "site_note", ""):
                        item["site_note"] = level.site_note
                entry["levels"].append(item)
        document["kpoints"].append(entry)
    with open(path, "w") as handle:
        json.dump(document, handle, indent=1)
        handle.write("\n")


def main(argv=None) -> None:
    import argparse
    from pathlib import Path

    from .crystal_orbital_diagram import parse_oxidation_tokens
    from .star_of_k import read_poscar_or_exit

    parser = argparse.ArgumentParser(
        description="Crystal-orbital diagram from three finished VASP runs.")
    parser.add_argument("--poscar", default=None)
    parser.add_argument("--co-left", nargs="+", required=True, metavar="FORMULA")
    parser.add_argument("--co-right", nargs="+", required=True, metavar="FORMULA")
    parser.add_argument("--oxidation", nargs="+", default=None, metavar="EL=Q")
    parser.add_argument("--kpoint", default=None)
    parser.add_argument("--electrons", type=float, default=None)
    parser.add_argument("--vasp", nargs="*", default=None, metavar="PATH")
    parser.add_argument("--vasp-setup", nargs="?", const=".", default=None,
                        metavar="ROOT")
    parser.add_argument("--vasp-crystal", default=None, metavar="DIR")
    parser.add_argument("--vasp-left", default=None, metavar="DIR")
    parser.add_argument("--vasp-right", default=None, metavar="DIR")
    parser.add_argument("--vasp-window", type=float, nargs=2, default=None,
                        metavar=("EMIN", "EMAX"))
    parser.add_argument("--vasp-mesh", type=int, nargs=3, default=None,
                        metavar=("N1", "N2", "N3"))
    parser.add_argument("--potcar-dir", default=None, metavar="DIR")
    parser.add_argument("--potcar-map", nargs="+", default=None, metavar="EL=NAME")
    parser.add_argument("--vasp-bin", default=None, metavar="PATH")
    parser.add_argument("--vasp-rwall", type=float, default=VACRWALL_DEFAULT,
                        metavar="A")
    parser.add_argument("--vasp-sigma", type=float, default=VACSIGMA_DEFAULT,
                        metavar="A")
    parser.add_argument("--vasp-wall-factor", type=float,
                        default=VACWALL_FACTOR_DEFAULT, metavar="F")
    parser.add_argument("--vasp-anchor", nargs=2, action="append", default=None,
                        metavar=("EL", "SHELL"))
    parser.add_argument("--vasp-projection-floor", type=float, default=None,
                        metavar="W")
    parser.add_argument("--vasp-align", choices=("site", "rigid"),
                        default="site")
    parser.add_argument("--vasp-zero", choices=("vbm", "efermi", "raw"),
                        default="vbm")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-align", action="store_true")
    parser.add_argument("--degeneracy-tol", type=float, default=None)
    parser.add_argument("--conventional", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    # the WAVECAR-overlap engine (crystod.crystal_orbital_overlap)
    parser.add_argument("--vasp-engine", choices=("anchor", "overlap"),
                        default="anchor")
    parser.add_argument("--vasp-shells", nargs="+", default=None, metavar="SHELL")
    parser.add_argument("--vasp-frozen-window", type=float, default=None,
                        metavar="EV")
    parser.add_argument("--vasp-frozen-qmin", type=float, default=None, metavar="Q")
    parser.add_argument("--vasp-cache", default=None, metavar="FILE")
    parser.add_argument("--vasp-scalar-reference", default=None, metavar="JSON")
    parser.add_argument("--vasp-irrep-json", default=None, metavar="PATTERN")
    parser.add_argument("--vasp-cross-sphere", default="bessel",
                        choices=("bessel", "projector", "none"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    oxidation = (parse_oxidation_tokens(args.oxidation) if args.oxidation else None)
    if args.vasp_setup is None and args.vasp_engine == "overlap":
        from .crystal_orbital_overlap import run_vasp_diagram

        run_vasp_diagram(
            args.co_left, args.co_right, vasp_paths=args.vasp,
            overrides={"mo": args.vasp_crystal, "left": args.vasp_left,
                       "right": args.vasp_right},
            cell_path=args.poscar, kpoint=args.kpoint, output=args.output,
            symprec=args.tolerance, shells=args.vasp_shells,
            frozen_window=args.vasp_frozen_window,
            frozen_qmin=args.vasp_frozen_qmin, cache_path=args.vasp_cache,
            scalar_reference=args.vasp_scalar_reference,
            irrep_json=args.vasp_irrep_json,
            window=(tuple(args.vasp_window) if args.vasp_window else None),
            cross_sphere=args.vasp_cross_sphere)
        return
    if args.vasp_setup is not None:
        # -c is optional here for the same reason it is optional with --vasp:
        # the crystal run's POSCAR IS the structure, and it is also the
        # setting the written POSCARs follow
        cell_path = args.poscar
        if cell_path is None:
            cell_path = os.path.join(args.vasp_setup, "BAND", "POSCAR")
            if not os.path.isfile(cell_path):
                raise SystemExit(
                    f"ERROR: no crystal POSCAR at {cell_path}; give the "
                    "structure with -c FILE.")
        cell = read_poscar_or_exit(cell_path)
        print("\n * CrystOD --vasp-setup: inputs of the sublattice runs *")
        write_vasp_inputs(cell, left=args.co_left, right=args.co_right,
                          root=args.vasp_setup, symprec=args.tolerance,
                          oxidation=oxidation, potcar_dir=args.potcar_dir,
                          potcar_map=_parse_potcar_map(args.potcar_map),
                          mesh=args.vasp_mesh, rwall=args.vasp_rwall,
                          sigma=args.vasp_sigma,
                          wall_factor=args.vasp_wall_factor,
                          binary=args.vasp_bin, force=args.force,
                          engine=args.vasp_engine,
                          frozen_window=args.vasp_frozen_window)
        return
    overrides = {"mo": args.vasp_crystal, "left": args.vasp_left,
                 "right": args.vasp_right}
    print("\n * VASP runs *")
    resolved = _resolve_directories(args.vasp, overrides)
    # -c is optional with --vasp: without it the structure IS the crystal
    # run's POSCAR, so a finished calculation needs nothing but its
    # directories.  The name of the page must then not be the directory's
    # ("BAND" says nothing); it is the formula and the space group, and so it
    # is for a -c file that lives INSIDE one of the run directories.
    cell_path = args.poscar
    run_directories = {os.path.abspath(path)
                       for key, path in resolved.items()
                       if key in ("mo", "left", "right") and path}
    run_directories.update(os.path.abspath(path)
                           for path in resolved.get("candidates", []))
    run_directories.update(os.path.abspath(path)
                           for path in (resolved.get("explicit") or {}).values()
                           if path)
    # DIR/band was resolved out of DIR, and a -c file beside the SCF files of
    # that run is just as much "inside a run directory" as one beside its
    # band/ PROCAR
    run_directories.update(os.path.dirname(path) for path in list(run_directories)
                           if os.path.basename(path) == "band")
    if cell_path is None:
        cell_path = os.path.join(resolved["mo"], "POSCAR")
        named = False
    else:
        named = (os.path.abspath(os.path.dirname(os.path.abspath(cell_path)))
                 not in run_directories)
    cell = read_poscar_or_exit(cell_path)
    stem = Path(cell_path).name
    for extension in (".vasp", ".poscar"):
        if stem.lower().endswith(extension):
            stem = stem[: -len(extension)]
    if stem.upper() in ("POSCAR", "CONTCAR"):
        stem = Path(cell_path).resolve().parent.name
    output_path = args.output or (f"CrystOD_{stem}_vasp.html" if named else None)
    report_and_write(
        cell, left=args.co_left, right=args.co_right, symprec=args.tolerance,
        electrons=args.electrons, kpoint_filter=args.kpoint,
        output_path=output_path,
        structure_label=stem if named else None, oxidation=oxidation,
        resolved=resolved, cell_path=cell_path,
        overrides=overrides,
        window=(tuple(args.vasp_window) if args.vasp_window else None),
        align=not args.no_align, conventional=args.conventional,
        degeneracy_tol=args.degeneracy_tol,
        projection_floor=args.vasp_projection_floor,
        anchor=[" ".join(pair) for pair in (args.vasp_anchor or [])],
        align_mode=args.vasp_align, zero=args.vasp_zero)


def _parse_potcar_map(tokens):
    """``["Sc=Sc_sv", "F=F"] -> {"Sc": "Sc_sv", "F": "F"}``."""
    mapping: dict = {}
    for token in tokens or []:
        for piece in token.split(","):
            if not piece.strip():
                continue
            if "=" not in piece:
                raise SystemExit(
                    f"ERROR: --potcar-map takes EL=NAME pairs, not {piece!r}.")
            element, name = piece.split("=", 1)
            mapping[element.strip()] = name.strip()
    return mapping


if __name__ == "__main__":
    main()
