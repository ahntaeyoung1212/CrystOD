"""Crystal-orbital diagrams from VASP wavefunctions: the overlap engine.

The second engine of ``crystod --diagram --vasp``.  The anchor engine
(:mod:`crystod.crystal_orbital_vasp`) aligns the energy scales of the crystal
and of the two sublattice runs with one shift per site, fitted to the levels
that symmetry forbids to mix.  Where a site has no such level -- the B-site s
and p shells of SrGeO3 (Ge 4s/4p) and CsPbI3 (Pb 6s/6p) at Gamma, X, M and R
-- the fit absorbs the very bonding shift the diagram is meant to show.  This
engine needs no alignment at all: it uses only the WAVEFUNCTIONS of the
sublattice runs and the EIGENVALUES of the crystal.

Stage 1 (:mod:`crystod.vasp_wavecar`) computes the all-electron PAW overlaps
``A_fn = <phi_f|psi_n>`` between the sublattice ("fragment") Bloch states and
the crystal Bloch states at the requested k points and stores them in an
:class:`~crystod.vasp_wavecar.OverlapCache`.  This module turns a cache into
the diagram data; every energy is relative to the crystal VBM.

Spectral on-site energies (diagnostic level)
--------------------------------------------
With the crystal eigenvalues ``eps_n`` the fragment state has the captured
weight ``W_f = sum_n |A_fn|^2`` and the spectral on-site energy
``d_f = sum_n |A_fn|^2 eps_n / W_f = <phi_f|H_crystal|phi_f>`` -- exact for a
complete band sum.  A point-charge sublattice orbital is not an eigenstate of
the crystal's atomic potentials; the part of it that relaxes projects onto
crystal states far above the gap, so ``d_f`` drifts with NBANDS.  All sums
therefore run over a FROZEN WINDOW, the crystal levels up to VBM + 14 eV
(whole multiplets).  Degenerate levels are always summed over their partners
(tolerance 1e-3 eV), so every number is gauge invariant.

Active space
------------
A minimal valence set of sublattice orbitals, as COHP codes use a minimal
basis.  Default rule: per element the shells of the POTCAR valence
configuration (``VRHFIN``; ``"Sr: 4s4p5s"``, ``"Ti: d3 s1"`` with the
principal numbers from the period, the atomic-configuration table when
``VRHFIN`` is empty, as for Pb) plus ONE standard empty shell -- groups 1-2:
the (n-1)d shell (n = outermost s; np in periods 2-3), groups 3-12: the np
shell, p block: nothing (np when the valence has no p).  SrTiO3: Sr 4s 4p 5s
4d, Ti 3d 4s 4p, O 2s 2p; SrGeO3: Ge 4s 4p; CsPbI3: Cs 5s 5p 6s 5d, Pb 6s 6p,
I 5s 5p -- the sets of the published analysis.  Diffuse higher manifolds
(Pb 7s, Pb 6d, Cs 6p, Ge 4d, Sr 5p) in a symmetric Loewdin basis produce
spurious populations and COHPs; they are excluded by the rule.  Fragment
levels are named by their dominant PAW-sphere (element, l) character with the
principal number counted from the POTCAR valence in energy order, every
``N_atoms (2l+1)`` states closing one manifold.  States that live on the
point charges are never active.  ``active_shells="auto-full"`` restores the
automatic rule (occupied levels plus the lowest empty manifold of every
cation (element, l)) for sensitivity tests.

Model space (frozen window + disentanglement)
---------------------------------------------
The crystal levels of the frozen window are kept exactly, except window levels
the active shells do not span (projection ``Q_nn`` on the active span below
``frozen_qmin`` = 0.5: interstitial and Rydberg-like states of open
structures such as CsPbI3), which enter only through the disentanglement.  If
there are more active states than frozen levels, the missing dimensions are
the leading eigenvectors of ``P_nf Q P_nf`` over the bands above the window
(Souza-Marzari-Vanderbilt projection-only disentanglement, no iteration); the
outer block is diagonalised into "effective outer levels".  With
``A[f, m] = <phi_f|m>`` in the model space, ``G = A A^H`` and
``Hp = A (eps - E_VBM) A^H``.

Three pictures (one model space)
--------------------------------
* (A) **lowdin** -- symmetric Loewdin, ``C' = G^-1/2 A``,
  ``H' = C' eps C'^H``: populations ``|C'_fn|^2``, the sublattice-orbital
  COHP ``COHP_n(f,g) = 2 Re[conj(C'_fn) H'_fg C'_gn]`` (negative = bonding;
  LOBSTER's convention) summed inter-sublattice, intra-sublattice and per
  shell pair (``"Ge 4s | O 2s"``), and the bond verdict from the
  inter-sublattice COHP (+-0.05 eV).  Paper convention: colours, populations
  and connector weights.
* (B) **ionic** -- frozen-ion reference: the occupied fragment orbitals of
  both sublattices Loewdin-orthonormalised among themselves (their span is
  the antisymmetrised frozen-ion state), the empty ones projected off that
  span and then orthonormalised.  Parent levels = eigenvalues of the
  sublattice blocks of ``H'' = C eps C^H``; the four-step ledger of every
  active level: bare ``d_f`` (potential shift) -> parent (Pauli shift) ->
  level of ``H''`` restricted to parents of the same occupation
  (filled-filled = closed-shell mixing; empty-empty for an empty parent) ->
  crystal level (covalent = occupied-empty, donor-acceptor mixing); the
  covalency count ``N_cov = sum_{n occ} sum_{f empty} occ_n |C_fn|^2``.
* (C) **bare** -- Ritz levels of the projected, non-orthogonal states,
  ``Hp_XX v = e G_XX v`` per sublattice, with Loewdin populations.

Checks reported per k point: unitarity of the orthonormalisation, eigenvalues
of the picture Hamiltonian against the model levels, the sum rule on-site +
COHP = ``(eps_n - E_VBM) x completeness`` per band, and the nonbonding
identities (an irrep carried by one sublattice only has parent = crystal
level).  The ionic parents are re-evaluated with frozen windows of 10 and 18
eV and without the cross-sphere term (sensitivity).

Spinors
-------
For ``vasp_ncl`` runs one band holds one electron, shell capacities are
``N_at (2l+1) x 2``, Kramers degeneracy is checked per run, and a parent that
inherits an accidental-degeneracy label (``-R11/-R8``) is named by its
dimension.  ``scalar_reference`` (a scalar results JSON of the same compound)
adds, at every k point, the scalar levels of the compatible single-valued
irreps next to every SOC level: a scalar level of irrep Gamma goes with the
double-valued irreps of Gamma x D(1/2)
(:func:`crystod.wavecar_irreps.spin_orbit_compatibility`, in the names the
SOC labels carry).

Irrep labels
------------
Labels come from a ``label_provider(run_index, k_name, k_frac)`` that returns
one label per band (run 0 = crystal); a level takes the common label of its
bands.  Convention of the labels: the label of the degenerate subspace a band
belongs to, ``"A/B"`` for an accidental degeneracy and a ``"?"`` suffix for
a fractional multiplicity (an incomplete multiplet at the top of the band
list), ``None`` when unknown.  ``crystod --diagram --vasp --vasp-engine
overlap`` (:func:`run_vasp_diagram`) uses :func:`wavecar_label_provider`:
the irreps of every run computed from its plane-wave coefficients
(:mod:`crystod.wavecar_irreps`), CrystOD's ISO-IR labels for scalar runs and
``-K<n><p>`` names (``-R6+``) for spinor runs, or IrRep's Bilbao names
mapped by band index when IrRep JSONs are given.

Output
------
:func:`write_json` writes the results with the structure and key names of the
paper's ``spectral_onsite.py`` v4.1 JSON (``kpoints[*]`` with the crystal and
fragment level lists and ``phase3``), so that the paper's ``plot_cod.py``
reads it unchanged, plus a top-level ``"crystod"`` block (version, options,
active shells).  :func:`write_report` writes a Markdown-compatible plain-text
report with a "Key results" block.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field, fields

import numpy as np

from .vasp_wavecar import (
    CROSS_SPHERE_MODES,
    OverlapCache,
    build_overlap_cache,
    find_kpoint,
    lowest_n,
    parse_kpoint_tokens,
    period_group,
)

_LNAME = "spdf"


# =============================================================== options
@dataclass
class OverlapOptions:
    """Analysis settings (the defaults are those of the published analysis).

    Attributes:
        emax: Frozen window, crystal levels up to VBM + emax (eV; whole
            multiplets); ``None`` = all bands.
        cross_sphere: Cross-sphere mode of the overlaps used
            (``"bessel"``, ``"projector"`` or ``"none"``).
        frozen_qmin: Window levels whose projection on the active span is
            below this are not frozen but left to the disentanglement.
        active_shells: Active shells (``["Ti 3d", "O 2p"]``) or ``None`` for
            the automatic rule.
        degen_tol: Degeneracy tolerance (eV).
        wmin_active: Automatic rule: minimum W over all computed bands.
        anion_empty: Automatic rule: also accept empty anion levels.
        reg_tol: G eigenvalues below this mark linear dependence.
        no_prune: Keep linearly dependent active states (regularise G).
        cohp_tol: Bond verdict threshold on the inter-sublattice COHP (eV).
        completeness_min: Crystal levels with a smaller Loewdin completeness
            are flagged "not drawn".
        shell_n: Overrides of the lowest principal number, ``"Sr:s=4"``.
        outer_skip: Exclude the top N bands from the outer window (default:
            the top level group and incomplete multiplets).
        sensitivity_windows: Frozen windows for the parent sensitivity
            (``None`` switches the sensitivity off).
        window_scan: Windows for the W/d_f scan of every fragment level.
        main_threshold: Partners listed above this |a|^2.
        pop_window: Fragment levels up to this above their HOMO count as
            "drawn" in the crystal populations.
        offdiag_threshold: List fragment-basis couplings above this (eV).
        symprec: Symmetry tolerance of the spin-orbit compatibility tables
            of the scalar bridge.
    """

    emax: float | None = 14.0
    cross_sphere: str = "bessel"
    frozen_qmin: float = 0.5
    active_shells: list | None = None
    degen_tol: float = 1e-3
    wmin_active: float = 0.5
    anion_empty: bool = False
    reg_tol: float = 1e-3
    no_prune: bool = False
    cohp_tol: float = 0.05
    completeness_min: float = 0.5
    shell_n: list = field(default_factory=list)
    outer_skip: int | None = None
    sensitivity_windows: list | None = field(default_factory=lambda: [10, 18])
    window_scan: list = field(default_factory=lambda: [6, 8, 10, 12, 14, 16, 20])
    main_threshold: float = 0.005
    pop_window: float = 12.0
    offdiag_threshold: float = 0.01
    symprec: float = 1e-5


_OPTION_NAMES = {f.name for f in fields(OverlapOptions)}


def _engine_name() -> str:
    try:
        from . import __version__
    except ImportError:                          # pragma: no cover
        __version__ = "unknown"
    # the prefix keeps the paper's plot_cod.py on its spectral_onsite adapter
    return f"spectral_onsite (crystod {__version__}, crystal_orbital_overlap)"


def _version() -> str:
    try:
        from . import __version__
        return __version__
    except ImportError:                          # pragma: no cover
        return "unknown"


# =============================================================== small helpers
def group_levels(eps, tol: float) -> list:
    """Consecutive bands closer than ``tol`` (eV) form one level group.

    Args:
        eps: Eigenvalues in ascending order.
        tol: Degeneracy tolerance.

    Returns:
        ``[[band, ...], ...]`` (0-based).
    """
    groups, cur = [], [0]
    for i in range(1, len(eps)):
        if eps[i] - eps[i - 1] < tol:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    return groups


def shell_summary(sw: dict, site_species, groups) -> list:
    """PAW-sphere weights per ``"El l"`` for every level group (per partner)."""
    res = []
    for g in groups:
        acc: dict = {}
        for (a, l), v in sw.items():
            key = f"{site_species[a]} {_LNAME[l]}"
            acc[key] = acc.get(key, 0.0) + float(np.mean(v[g]))
        res.append(dict(sorted(acc.items(), key=lambda kv: -kv[1])))
    return res


def block_norms(X, groups_a, groups_b) -> np.ndarray:
    """Gauge-invariant size of the blocks of ``X`` between level groups.

    ``||X_FF'||_Frobenius / sqrt(min(deg F, deg F'))`` -- ``|x|`` for a
    scalar block and ``|h|`` for a block ``h U`` with ``U`` unitary, as
    Wigner-Eckart requires for two levels of the same irrep.
    """
    return np.array([[np.linalg.norm(X[np.ix_(ga, gb)]) / math.sqrt(min(len(ga), len(gb)))
                      for gb in groups_b] for ga in groups_a])


def _isqrt_h(M, tol):
    """Hermitian ``M^-1/2`` on the eigenvalues > tol (pseudo-inverse beyond)."""
    lam, U = np.linalg.eigh(M)
    keep = lam > tol
    Uk = U[:, keep]
    return (Uk / np.sqrt(lam[keep])) @ Uk.conj().T, lam, int((~keep).sum())


def _pair_key(sa, ra, sb, rb, cation_runs):
    """``"cation shell | anion shell"`` (inter) or sorted ``"a | b"`` (intra)."""
    if ra != rb:
        if ra in cation_runs or (rb not in cation_runs and ra < rb):
            return f"{sa} | {sb}", "inter"
        return f"{sb} | {sa}", "inter"
    a, b = sorted([sa, sb])
    return f"{a} | {b}", "intra"


def _nanmax(values):
    clean = [v for v in values if v is not None and not (isinstance(v, float) and
                                                         math.isnan(v))]
    return max(clean) if clean else float("nan")


# =============================================================== irrep labels
def _group_label(band_labels):
    """Label of a level group from the labels of its bands (None if any unknown)."""
    if any(lab is None for lab in band_labels):
        return None
    uniq = list(dict.fromkeys(band_labels))
    if len(uniq) == 1:
        return uniq[0]
    parts = sorted({p for lab in uniq for p in lab.split("/")})
    return "/".join(parts)


def _labels_of(kp, run_index):
    if not kp.labels or len(kp.labels) <= run_index:
        return None
    return kp.labels[run_index]


def irrep_group_labels(band_labels, groups):
    """Level-group labels from per-band labels (``None`` when there are none)."""
    if band_labels is None:
        return None
    out = []
    for g in groups:
        if g[-1] >= len(band_labels):
            out.append(None)
            continue
        out.append(_group_label([band_labels[b] for b in g]))
    return out


def little_group_dimensions(cache, kp, symprec: float = 1e-5) -> dict:
    """``{irrep: dimension}`` of the small irreps at one k point, from symmetry.

    The irreps of the little group of the cache's crystal cell
    (:func:`crystod.wavecar_irreps.little_group_data`; double-valued for a
    spinor cache) in the names the plane-wave labels carry, plus their Bilbao
    names where the cache holds the CrystOD-to-Bilbao map of IrRep-labelled
    runs.  Exact where :func:`irrep_dimensions` can be fooled: a Kramers pair
    split by a few meV is two level groups of one band each, both carrying
    the pair's two-dimensional label.

    Returns:
        The dimensions, or ``{}`` when the symmetry data cannot be built.
    """
    from .wavecar_irreps import little_group_data

    run = cache.runs[0]
    try:
        group = little_group_data(np.asarray(run["lattice"], float),
                                  np.asarray(run["positions"], float),
                                  _species_numbers(run["site_species"]), list(kp.frac),
                                  spinor=bool(run["spinor"]), symprec=symprec)
    except (ValueError, KeyError):
        return {}
    dims = {str(name): int(dim) for name, dim in zip(group.names, group.dims)}
    name_map = (cache.meta.get("irrep_name_maps") or {}).get(kp.name) or {}
    dims.update({name_map[name]: dim for name, dim in list(dims.items())
                 if name in name_map})
    return dims


def irrep_dimensions(kp, tol: float) -> dict:
    """``{irrep: dimension}`` at one k point, learnt from the labelled levels.

    A level whose bands all carry one plain label (no ``/``, no ``?``) has
    the dimension of that irrep; the smallest such level counts.  The
    analysis lets :func:`little_group_dimensions` override these.
    """
    dims: dict = {}
    for r, eps in enumerate(kp.eigenvalues):
        labels = _labels_of(kp, r)
        if labels is None:
            continue
        groups = group_levels(eps, tol)
        for g, lab in zip(groups, irrep_group_labels(labels, groups)):
            if not lab or "/" in lab or "?" in lab:
                continue
            dims[lab] = min(dims.get(lab, len(g)), len(g))
    return dims


# =============================================================== active space
def _shell_name(element: str, n: int, l: int) -> str:
    return f"{element} {n}{_LNAME[l]}"


def default_active_shells(cache: OverlapCache) -> list:
    """The default minimal valence active space of a cache.

    Per element of the crystal (POSCAR order): the POTCAR valence shells
    (``VRHFIN``, see :func:`crystod.vasp_wavecar.valence_shells`) plus one
    standard empty shell -- groups 1-2: (n-1)d with n the outermost s shell
    (np when n - 1 < 3), groups 3-12 (lanthanides included): np, p block:
    nothing, or np when the valence has no p shell.

    Args:
        cache: The overlap cache (its ``paw`` metadata).

    Returns:
        Shell names such as ``["Sr 4s", "Sr 4p", "Sr 5s", "Sr 4d", ...]``.
    """
    crystal = cache.runs[0]
    paw = cache.meta["paw"]
    out = []
    for el in crystal["species"]:
        if el not in paw:
            continue
        shells = [(int(n), int(l)) for n, l in paw[el]["valence_shells"]]
        names = [_shell_name(el, n, l) for n, l in shells]
        s_n = [n for n, l in shells if l == 0]
        try:
            _, group, _ = period_group(el)
        except ValueError:
            group = 0
        extra = None
        if s_n:
            n = max(s_n)
            if group in (1, 2):
                extra = (n - 1, 2) if n - 1 >= 3 else (n, 1)
            elif 3 <= group <= 12:
                extra = (n, 1)
            elif not any(l == 1 for _, l in shells):
                extra = (n, 1)
        if extra is not None and _shell_name(el, *extra) not in names:
            names.append(_shell_name(el, *extra))
        out.extend(names)
    return out


def parse_shell_overrides(items) -> dict:
    """``["Sr:s=4"]`` -> ``{("Sr", "s"): 4}``.

    Raises:
        SystemExit: Malformed item.
    """
    out = {}
    for it in items or []:
        m = re.fullmatch(r"\s*([A-Z][a-z]?)\s*:\s*([spdf])\s*=\s*(\d+)\s*", it)
        if not m:
            raise SystemExit(f"ERROR: shell override expects El:l=n (e.g. Sr:s=4), got {it!r}.")
        out[(m.group(1), m.group(2))] = int(m.group(3))
    return out


def assign_shell_names(flev, run, paw, overrides) -> None:
    """Shell name with principal quantum number for every level of one run.

    The dominant (element, l) PAW-sphere character decides the channel; the
    states of one channel are counted in energy order, every
    ``N_atoms(element) (2l + 1)`` states (x 2 for spinors) closing one
    manifold (n, n + 1, ...).  A mixed empty state goes to its second channel
    when the dominant one has already filled its lowest empty manifold and
    the second has not (Ti 4s / Sr 5s mixtures in GM1+).

    Args:
        flev: Level dicts of the run (``shells``, ``occupation``,
            ``degeneracy``); updated in place.
        run: Run metadata of the cache.
        paw: PAW metadata per species of the cache.
        overrides: ``{(element, letter): n}``.
    """
    nat: dict = {}
    for sp, own in zip(run["site_species"], run["own"]):
        if own:
            nat[sp] = nat.get(sp, 0) + 1
    count: dict = {}
    first_empty: dict = {}
    spinor = bool(run["spinor"])
    ef = float(run["electrons_per_band"])

    def cap_of(key):
        el, lch = key.split()
        return nat[el] * (2 * _LNAME.index(lch) + 1) * (2 if spinor else 1)
    for L in flev:
        L["occupied"] = bool(L["occupation"] >= 0.5 * ef * L["degeneracy"] - 1e-6)
        L["_ef"] = ef                 # electrons per state (not written)
        sh = {k: v for k, v in L["shells"].items() if k.split()[0] in nat}
        if not sh:
            L.update(shell=None, shell_el=None, shell_l=None, manifold=None)
            continue
        ranked = sorted(sh, key=sh.get, reverse=True)
        key = ranked[0]
        if not L["occupied"] and len(ranked) > 1:
            k2 = ranked[1]
            full = (count.get(key, 0) // cap_of(key)) > first_empty.get(key, 10 ** 6)
            free2 = (count.get(k2, 0) // cap_of(k2)) <= first_empty.get(
                k2, count.get(k2, 0) // cap_of(k2))
            if full and free2 and sh[k2] >= 0.5 * sh[key]:
                key = k2
        el, lch = key.split()
        l = _LNAME.index(lch)
        cap = cap_of(key)
        before = count.get(key, 0)
        count[key] = before + L["degeneracy"]
        idx = before // cap
        if not L["occupied"]:
            first_empty.setdefault(key, idx)
        ds = paw.get(el)
        n0 = overrides.get((el, lch)) or lowest_n(el, l, ds["vrhfin"] if ds else "",
                                                  ds["zval"] if ds else 0.0)
        L.update(shell=f"{el} {n0 + idx}{lch}", shell_el=el, shell_l=lch, manifold=idx)


def select_active(flev, role: str, opts: OverlapOptions) -> None:
    """Active-space rule; sets ``L["active"]`` and ``L["reject"]``."""
    first_empty: dict = {}
    for L in flev:
        if L.get("shell") and not L["occupied"]:
            key = (L["shell_el"], L["shell_l"])
            first_empty[key] = min(first_empty.get(key, 10 ** 6), L["manifold"])
    for L in flev:
        why = []
        if opts.active_shells:
            if L.get("shell") not in opts.active_shells:
                why.append("shell not in --active-shells")
        else:
            wv = L.get("W_all", L["W"])
            if wv < opts.wmin_active:
                why.append(f"W(all bands) {wv:.2f} < {opts.wmin_active:g}")
            if not L.get("shell"):
                why.append("no atomic character")
            if role == "anion" and not L["occupied"] and not opts.anion_empty:
                why.append("empty level of the anion run")
            if role == "cation" and not L["occupied"] and L.get("shell") and \
                    L["manifold"] > first_empty.get((L["shell_el"], L["shell_l"]), 0):
                why.append("higher manifold")
        if L["pointcharge_site_state"]:
            why.append("point-charge-site state")
        L["active"] = not why
        L["reject"] = "; ".join(why) or None


def active_rule_text(opts: OverlapOptions) -> str:
    """One-sentence statement of the active-space rule (JSON and report)."""
    tail = ("; no pruning (missing dimensions come from the SMV disentanglement); only "
            "as a last resort, while lambda_min(G) <= reg_tol in the model space, the "
            "lowest-W level group in the null space of G is removed"
            if not opts.no_prune else "")
    if opts.active_shells:
        return ("active = levels whose shell is in " + ", ".join(opts.active_shells) +
                " AND not a point-charge-site state" + tail)
    return (f"active = W_f(all computed bands) >= {opts.wmin_active:g} AND not a "
            "point-charge-site state (own PAW-sphere weight < 0.3 and below its smooth "
            "weight in the point-charge spheres) AND atomic character; anion run: "
            "occupied levels " + ("and empty ones" if opts.anion_empty else "only") +
            "; cation run: occupied levels and the lowest empty manifold of every "
            "(element, l)" + tail)


# =============================================================== model space
def build_model_space(A_full, eps, frozen, tol=1e-8, outer=None):
    """Souza-Marzari-Vanderbilt projection-only disentanglement (no iteration).

    Args:
        A_full: ``A[f, n] = <phi_f|psi_n>`` over ALL computed bands.
        eps: Crystal eigenvalues (relative to the VBM).
        frozen: Bool mask of the frozen window (kept exactly).
        tol: Eigenvalue floor of the Loewdin inverse square root.
        outer: Bool mask of the bands the outer space may use.

    Returns:
        ``(A_m, e_m, info, eff)``: ``A_m[f, m] = <phi_f|m>`` in the model
        basis (frozen bands first, then the effective outer levels), their
        energies, the bookkeeping dict, and ``(NF, X, et, q_eff)`` of the
        effective levels (``None`` when there are none).
    """
    if outer is None:
        outer = np.ones(len(eps), bool)
    F = np.where(frozen)[0]
    NF = np.where(~frozen & outer)[0]
    n_act = A_full.shape[0]
    n_extra = min(max(0, n_act - len(F)), len(NF))
    info = {"n_frozen": int(len(F)), "n_active": int(n_act), "n_extra": int(n_extra),
            "frozen_bands": [int(b + 1) for b in F]}
    if n_extra == 0:
        info.update(pqp_leading=[], pqp_next=[], effective=[])
        return A_full[:, F], eps[F].copy(), info, None
    ob = np.where(frozen | outer)[0]
    Ao = A_full[:, ob]
    Gm, lam, _ = _isqrt_h(Ao @ Ao.conj().T, tol)
    U0 = np.zeros((len(eps), n_act), complex)
    U0[ob] = Ao.conj().T @ Gm                      # band space, Loewdin fragment states
    Unf = U0[NF]
    mu, Y = np.linalg.eigh(Unf @ Unf.conj().T)     # P_nf Q P_nf
    order = np.argsort(-mu)
    mu, Y = mu[order], Y[:, order]
    V = Y[:, :n_extra]
    et, Wt = np.linalg.eigh(V.conj().T @ (eps[NF][:, None] * V))
    X = V @ Wt                                     # effective levels in band space
    A_m = np.hstack([A_full[:, F], A_full[:, NF] @ X])
    e_m = np.concatenate([eps[F], et])
    q_eff = np.real(np.einsum("ij,ik,kj->j", np.conj(X), Unf @ Unf.conj().T, X))
    info.update(pqp_leading=[float(x) for x in mu[:n_extra]],
                pqp_next=[float(x) for x in mu[n_extra:n_extra + 3]],
                G_all_min_eig=float(lam.min()))
    return A_m, e_m, info, (NF, X, et, q_eff)


def _q_filter(A_full, frozen0, outer, groups_c, clev, qmin):
    """Frozen mask restricted to the window levels with ``Q_nn >= qmin``.

    ``Q_nn`` is the projection of a crystal level on the span of the active
    fragment states (over the outer window).  While there are more frozen
    bands than active states, the frozen level with the smallest ``Q`` goes
    to the disentanglement as well.
    """
    frozen = frozen0.copy()
    excluded = []
    if qmin <= 0:
        return frozen, excluded
    nb = len(frozen0)
    ob = np.where(frozen0 | outer)[0]
    Ao = A_full[:, ob]
    Gq, _, _ = _isqrt_h(Ao @ Ao.conj().T, 1e-8)
    Uq = Ao.conj().T @ Gq
    qnn = np.zeros(nb)
    qnn[ob] = np.real(np.einsum("ij,ij->i", Uq, np.conj(Uq)))
    qg = {}
    for j, gc in enumerate(groups_c):
        if frozen0[gc[0]]:
            qj = float(np.mean(qnn[gc]))
            qg[j] = qj
            if qj < qmin:
                frozen[gc] = False
                excluded.append({"id": clev[j]["id"], "label": clev[j].get("label"),
                                 "e_rel_vbm": clev[j]["energy_rel_vbm"],
                                 "degeneracy": len(gc), "q": round(qj, 4),
                                 "reason": f"Q < {qmin:g}"})
    n_act = A_full.shape[0]
    while int(frozen.sum()) > n_act:
        j = min((j for j in qg if frozen[groups_c[j][0]]), key=lambda j: qg[j])
        frozen[groups_c[j]] = False
        excluded.append({"id": clev[j]["id"], "label": clev[j].get("label"),
                         "e_rel_vbm": clev[j]["energy_rel_vbm"],
                         "degeneracy": len(groups_c[j]), "q": round(qg[j], 4),
                         "reason": "smallest Q while n_frozen > n_active"})
    return frozen, excluded


# =============================================================== pictures
def _picture_parents(Hb, idx_by_run, act_lid, act_shell, flevs_lookup, tol, rot=None,
                     energies=None):
    """Parent levels of a picture: eigen-decomposition of the sublattice blocks.

    For the bare picture the Ritz energies and rotations are given instead.

    Returns:
        ``{run: (idx, Y, parent levels)}``.
    """
    out = {}
    for ir, idx in idx_by_run.items():
        if len(idx) == 0:
            continue
        if rot is None:
            e, Y = np.linalg.eigh(Hb[np.ix_(idx, idx)])
        else:
            e, Y = energies[ir], rot[ir]
        comp = np.abs(Y) ** 2
        levs = []
        for jg, rg in enumerate(group_levels(e, tol)):
            share: dict = {}
            for p, q in enumerate(idx):
                lid = act_lid[q]
                share[lid] = share.get(lid, 0.0) + float(comp[p, rg].sum()) / len(rg)
            top = max(share, key=share.get)
            L = flevs_lookup[top]
            irr = L.get("irrep")
            pure = sum(v for k, v in share.items() if flevs_lookup[k].get("irrep") == irr)
            occ = sum(float(comp[p, rg].sum()) * (flevs_lookup[act_lid[q]].get("_ef", 2.0)
                                                 if flevs_lookup[act_lid[q]]["occupied"]
                                                 else 0.0) for p, q in enumerate(idx))
            levs.append({"id": None, "ritz_index": [int(x) for x in rg],
                         "e_rel_vbm": float(np.mean(e[rg])), "degeneracy": len(rg),
                         "irrep": irr, "irrep_purity": float(pure), "label": L.get("label"),
                         "shell": L.get("shell"), "electrons": float(occ),
                         "composition": sorted([[k, round(v, 5)] for k, v in share.items()
                                                if v > 1e-3], key=lambda x: -x[1])})
        out[ir] = (idx, Y, levs)
    return out


def _resolve_by_dimension(par, dims) -> None:
    """Name a parent that inherits an accidental-degeneracy label by its dimension."""
    if not dims:
        return
    for ir, (idx, Y, levs) in par.items():
        for R in levs:
            irr = R.get("irrep")
            if not irr or "/" not in irr:
                continue
            cand = sorted({p for p in irr.split("/") if dims.get(p) == R["degeneracy"]})
            if len(cand) == 1:
                if R.get("label"):
                    R["label"] = R["label"].replace(irr, cand[0])
                R["irrep"] = cand[0]


def _level_analysis(C, H, e_m, occ_m, eff_m, groups_m, meta, parents, opts):
    """Per model level and per level group: populations, COHP, sum rule, verdict."""
    n_act = C.shape[0]
    run = meta["run"]
    occ = meta["occupied"]
    shells = meta["shell"]
    cat_names = ("occ-occ", "occ-emp", "emp-emp")
    keys, kinds = [], []
    K = -np.ones((n_act, n_act), int)
    CAT = np.zeros((n_act, n_act), int)
    for a in range(n_act):
        for b in range(a + 1, n_act):
            key, kind = _pair_key(shells[a], run[a], shells[b], run[b], meta["cation_runs"])
            if key not in keys:
                keys.append(key)
                kinds.append(kind)
            K[a, b] = keys.index(key)
            CAT[a, b] = int(not occ[a]) + int(not occ[b])
    up = K >= 0
    is_inter = (run[:, None] != run[None, :])
    sh_u = sorted(set(shells))
    sh_idx = np.array([sh_u.index(s) for s in shells])
    bands = []
    for m in range(len(e_m)):
        c = C[:, m]
        T = np.conj(c)[:, None] * H * c[None, :]
        pop = np.abs(c) ** 2
        v = 2 * np.real(T)
        pv = np.bincount(K[up], weights=v[up], minlength=len(keys))
        inter_mask = up & is_inter
        intra_mask = up & ~is_inter
        split = {cat_names[t]: float(v[inter_mask & (CAT == t)].sum()) for t in range(3)}
        rp = {}
        for ir, (idx, Y, levs) in parents.items():
            q = np.abs(Y.conj().T @ c[idx]) ** 2
            for R in levs:
                rp[R["id"]] = float(q[R["ritz_index"]].sum())
        fp: dict = {}
        for p in range(n_act):
            fp[meta["lid"][p]] = fp.get(meta["lid"][p], 0.0) + float(pop[p])
        compl = float(pop.sum())
        onsite = float(np.real(np.trace(T)))
        inter = float(v[inter_mask].sum())
        intra = float(v[intra_mask].sum())
        bands.append({
            "model_index": m, "e_rel_vbm": float(e_m[m]), "occupation": float(occ_m[m]),
            "effective": bool(eff_m[m]), "completeness": compl, "onsite": onsite,
            "cohp_inter": inter, "cohp_intra": intra, "cohp_inter_split": split,
            "sumrule_residual": onsite + inter + intra - float(e_m[m]) * compl,
            "pairs": {keys[t]: round(float(x), 5) for t, x in enumerate(pv) if abs(x) > 1e-4},
            "pop_shell": {sh_u[s]: round(float(x), 5) for s, x in
                          enumerate(np.bincount(sh_idx, weights=pop, minlength=len(sh_u)))
                          if x > 1e-4},
            "pop_level": fp, "pop_parent": rp})
    levels = []
    for gi, g in enumerate(groups_m):
        mem = [bands[m] for m in g]
        dN = len(g)

        def avg(key, mem=mem, dN=dN):
            return sum(x[key] for x in mem) / dN

        def avgd(key, mem=mem, dN=dN):
            acc: dict = {}
            for x in mem:
                for k2, val in x[key].items():
                    acc[k2] = acc.get(k2, 0.0) + val / dN
            return acc
        inter = avg("cohp_inter")
        levels.append({
            "group": gi, "e_rel_vbm": avg("e_rel_vbm"), "degeneracy": dN,
            "occupation": sum(x["occupation"] for x in mem), "effective": mem[0]["effective"],
            "completeness": avg("completeness"), "onsite": avg("onsite"),
            "cohp_inter": inter, "cohp_intra": avg("cohp_intra"),
            "cohp_inter_split": avgd("cohp_inter_split"),
            "sumrule_residual": avg("sumrule_residual"),
            "bond": ("bonding" if inter < -opts.cohp_tol else
                     "antibonding" if inter > opts.cohp_tol else "nonbonding"),
            "cohp_pairs": dict(sorted(((k2, round(val, 5)) for k2, val in
                                       avgd("pairs").items() if abs(val) > 5e-4),
                                      key=lambda x: -abs(x[1]))),
            "pop_shell": dict(sorted(((k2, round(val, 5)) for k2, val in
                                      avgd("pop_shell").items()), key=lambda x: -x[1])),
            "pop_level": dict(sorted(((k2, round(val, 5)) for k2, val in
                                      avgd("pop_level").items() if val > 1e-3),
                                     key=lambda x: -x[1])),
            "pop_parent": dict(sorted(((k2, round(val, 5)) for k2, val in
                                       avgd("pop_parent").items() if val > 1e-3),
                                      key=lambda x: -x[1])),
            "drawn": bool(avg("completeness") >= opts.completeness_min)})
    compact = [{k: v for k, v in b.items() if k not in ("pop_level", "pop_parent")}
               for b in bands]
    return levels, compact, bands


def _ionic_coefficients(A_m, occ_rows, tol):
    """Frozen-ion orthogonalisation.

    Occupied rows Loewdin among themselves, empty rows projected onto the
    complement of the occupied span, then Loewdin.
    """
    O = np.where(occ_rows)[0]
    Vr = np.where(~occ_rows)[0]
    C = np.zeros_like(A_m)
    info = {}
    if len(O):
        Gi, lam, nd = _isqrt_h(A_m[O] @ A_m[O].conj().T, tol)
        C[O] = Gi @ A_m[O]
        info["occupied_min_eig"] = float(lam.min())
        info["occupied_dropped"] = nd
        P = C[O].conj().T @ C[O]
    else:
        P = np.zeros((A_m.shape[1], A_m.shape[1]))
    if len(Vr):
        Av = A_m[Vr] - A_m[Vr] @ P
        Gi, lam, nd = _isqrt_h(Av @ Av.conj().T, tol)
        C[Vr] = Gi @ Av
        info["empty_min_eig"] = float(lam.min())
        info["empty_dropped"] = nd
    return C, info


def _same_occupation_levels(H, par, occ_rows, tol):
    """Two-step split of the mixing shift of the parents of a picture.

    In the parent basis H has the parent energies on the diagonal and only
    inter-sublattice couplings.  Step 1 diagonalises H within the occupied
    parents and within the empty parents separately (filled-filled =
    closed-shell, and empty-empty mixing); step 2, the rest up to the crystal
    level, is the occupied-empty (covalent, donor-acceptor) mixing.

    Returns:
        ``{parent id: {"class": "filled" | "empty", "e_block": energy}}``.
    """
    owner, filled = [], []
    vecs = []
    n_act = H.shape[0]
    for ir, (idx, Y, levs) in par.items():
        for R in levs:
            for r in R["ritz_index"]:
                v = np.zeros(n_act, complex)
                v[idx] = Y[:, r]
                vecs.append(v)
                owner.append(R["id"])
            w = float(np.sum(np.abs(Y[np.ix_(np.where(occ_rows[idx])[0], R["ritz_index"])]) ** 2)
                      ) / len(R["ritz_index"])
            filled += [w >= 0.5] * len(R["ritz_index"])
    P = np.array(vecs).T
    Hp = P.conj().T @ H @ P
    filled = np.array(filled)
    out = {}
    for cls, mask in (("filled", filled), ("empty", ~filled)):
        sel = np.where(mask)[0]
        if not len(sel):
            continue
        e, V = np.linalg.eigh(Hp[np.ix_(sel, sel)])
        grp = group_levels(e, tol)
        by_parent: dict = {}
        for pos, c in enumerate(sel):
            by_parent.setdefault(owner[c], []).append(pos)
        for pid, rows in by_parent.items():
            w = [float(np.sum(np.abs(V[np.ix_(rows, g)]) ** 2)) for g in grp]
            g = grp[int(np.argmax(w))]
            out[pid] = {"class": cls, "e_block": float(np.mean(e[g]))}
    return out


def phase3_analysis(ctx, eps_raw, occ_raw, groups_c, clev, win, flevs, Amats, S12, Amats_none,
                    opts, irrep_dims=None):
    """Disentangled model space and the ionic, lowdin and bare pictures at one k.

    Args:
        ctx: Analysis context (runs, roles, columns, VBM, spinor flag).
        eps_raw: Crystal eigenvalues (raw scale).
        occ_raw: Crystal occupations (0..1 per band).
        groups_c: Crystal level groups.
        clev: Crystal level dicts.
        win: Frozen-window flag per crystal level group.
        flevs: ``{run: fragment level dicts}``.
        Amats: ``{run: A (nb_f, nb_c)}`` of the main cross-sphere mode.
        S12: Fragment-fragment overlap (two fragments) or ``None``.
        Amats_none: ``{run: A}`` without the cross-sphere term, or ``None``.
        opts: :class:`OverlapOptions`.
        irrep_dims: ``{irrep: dimension}`` at this k.

    Returns:
        The ``phase3`` block.

    Raises:
        SystemExit: An active level has no overlap rows in a trimmed cache.
    """
    vbm = ctx["vbm"]
    cols = ctx["cols"]
    eps = eps_raw - vbm
    nb = len(eps)
    frozen0 = np.zeros(nb, bool)
    for j, gc in enumerate(groups_c):
        frozen0[gc] = win[j]
    occ_c = occ_raw * ctx["ef"]
    cation_runs = {ir for ir, r in ctx["roles"].items() if r == "cation"}
    lookup = {L["id"]: L for flev in flevs.values() for L in flev}
    out = {"rule": active_rule_text(opts), "accepted": [], "rejected": []}
    # outer window: all computed bands except the top level groups that may be
    # incomplete multiplets (the top group, and from the top down every group
    # whose irrep is unknown or fractional, '?'); outer_skip N drops the top N
    outer = np.ones(nb, bool)
    if opts.outer_skip is not None:
        if opts.outer_skip > 0:
            outer[nb - opts.outer_skip:] = False
    elif any(clev[j].get("irrep") for j in range(len(groups_c))):
        for j in range(len(groups_c) - 1, -1, -1):
            lab = clev[j].get("irrep")
            if j == len(groups_c) - 1 or lab is None or "?" in lab:
                if not win[j]:
                    outer[groups_c[j]] = False
                continue
            break
    else:                       # no irreps (general k): drop the top 1 eV of the bands
        etop = eps[groups_c[-1]].max()
        for j, gc in enumerate(groups_c):
            if not win[j] and (j == len(groups_c) - 1 or eps[gc].min() > etop - 1.0):
                outer[gc] = False
    out["outer_window_bands"] = [1, int(np.where(outer)[0].max() + 1)]
    # ---------------------------------------------- active space (+ last-resort pruning)
    pruned = []
    while True:
        act = [(ir, i, b - 1) for ir, flev in flevs.items()
               for i, L in enumerate(flev) if L["active"] for b in L["bands"]]
        if not act:
            out["n_active"] = 0
            return out
        A_full = np.array([Amats[ir][b, :] for ir, i, b in act])
        if not np.all(np.isfinite(A_full)):
            bad = sorted({flevs[ir][i]["id"] for (ir, i, b), row in zip(act, A_full)
                          if not np.all(np.isfinite(row))})
            raise SystemExit("ERROR: the overlap cache has no rows for the active levels "
                             f"{', '.join(bad)} (trimmed for another active space).")
        frozen, excluded = _q_filter(A_full, frozen0, outer, groups_c, clev, opts.frozen_qmin)
        A_m, e_m, minfo, eff = build_model_space(A_full, eps, frozen, outer=outer)
        G = A_m @ A_m.conj().T
        lam, U = np.linalg.eigh(G)
        if lam[0] > opts.reg_tol or opts.no_prune:
            break
        null = np.abs(U[:, lam <= opts.reg_tol]) ** 2
        wgt: dict = {}
        for p, (ir, i, b) in enumerate(act):
            wgt[(ir, i)] = wgt.get((ir, i), 0.0) + float(null[p].sum())
        cand = [key for key, w in wgt.items() if w > 0.05]
        if not cand:
            break
        ir, i = min(cand, key=lambda key: (flevs[key[0]][key[1]]["W_all"], -wgt[key]))
        L = flevs[ir][i]
        L["active"] = False
        L["reject"] = (f"last resort: linearly dependent in the model space "
                       f"(lambda_min(G) = {lam[0]:.1e})")
        pruned.append(L["id"])
    out["pruned_last_resort"] = pruned
    for ir, flev in flevs.items():
        for L in flev:
            rec = {"id": L["id"], "column": cols[ir], "label": L.get("label"),
                   "shell": L.get("shell"), "irrep": L.get("irrep"),
                   "degeneracy": L["degeneracy"], "occupied": L["occupied"],
                   "energy_raw": L["energy_raw"], "W_all": L["W_all"], "W_window": L["W"]}
            if L["active"]:
                out["accepted"].append(rec)
            elif L["W_all"] >= 0.2 or L["W"] >= 0.2:
                rec["reason"] = L["reject"]
                out["rejected"].append(rec)
    n_act = len(act)
    out["n_active"] = n_act
    if opts.active_shells:      # states found per listed shell vs N_at (2l+1) [x 2 spinor]
        found: dict = {}
        for ir, flev in flevs.items():
            for L in flev:
                if L["active"]:
                    found[L["shell"]] = found.get(L["shell"], 0) + L["degeneracy"]
        cnt = {}
        for sh in opts.active_shells:
            el, nl = sh.split()
            nat = sum(1 for r in ctx["runs"][1:] for sp, own in zip(r["site_species"], r["own"])
                      if own and sp == el)
            cnt[sh] = [found.get(sh, 0),
                       nat * (2 * _LNAME.index(nl[-1]) + 1) * (2 if ctx["spinor"] else 1)]
        out["active_shell_counts"] = cnt
    # ---------------------------------------------- model space
    nF = int(frozen.sum())
    occ_m = np.concatenate([occ_c[frozen], np.zeros(len(e_m) - nF)])
    eff_m = np.concatenate([np.zeros(nF, bool), np.ones(len(e_m) - nF, bool)])
    frozen_groups = [j for j in range(len(groups_c)) if frozen[groups_c[j][0]]]
    pos = {b: p for p, b in enumerate(np.where(frozen)[0])}
    groups_m = [[pos[b] for b in groups_c[j]] for j in frozen_groups]
    eff_levels = []
    if eff is not None:
        NF, X, et, q_eff = eff
        crys_irr = {}
        for j, gc in enumerate(groups_c):
            for b in gc:
                crys_irr[b] = clev[j].get("irrep")
        for t, rg in enumerate(group_levels(et, opts.degen_tol)):
            w = np.sum(np.abs(X[:, rg]) ** 2, axis=1) / len(rg)
            top = np.argsort(-w)[:4]
            irw: dict = {}
            for i_nf, wv in zip(NF, w):
                key = crys_irr.get(i_nf) or "?"
                irw[key] = irw.get(key, 0.0) + float(wv)
            irr = max(irw, key=irw.get)
            eff_levels.append({
                "id": f"e{t}", "e_rel_vbm": float(np.mean(et[rg])), "degeneracy": len(rg),
                "irrep": irr, "irrep_purity": float(irw[irr]),
                "pqp": float(np.mean(q_eff[rg])),
                "bands": [[int(NF[i] + 1), round(float(w[i]), 4)] for i in top if w[i] > 0.01]})
            groups_m.append([nF + r for r in rg])
    minfo["effective"] = eff_levels
    minfo["G_min_eig"] = float(lam.min())
    if excluded:
        minfo["window_levels_not_frozen"] = excluded
    out["model_space"] = minfo
    mlev = []
    for gi, g in enumerate(groups_m):
        if gi < len(frozen_groups):
            j = frozen_groups[gi]
            mlev.append({"id": clev[j]["id"], "label": clev[j].get("label"),
                         "irrep": clev[j].get("irrep"), "effective": False})
        else:
            E = eff_levels[gi - len(frozen_groups)]
            mlev.append({"id": E["id"], "label": f"{E['irrep']} (eff)",
                         "irrep": E["irrep"], "effective": True})
    # ---------------------------------------------- metadata of the active rows
    meta = {"run": np.array([ir for ir, i, b in act]),
            "lid": [flevs[ir][i]["id"] for ir, i, b in act],
            "shell": [flevs[ir][i].get("shell") or "?" for ir, i, b in act],
            "occupied": np.array([flevs[ir][i]["occupied"] for ir, i, b in act]),
            "cation_runs": cation_runs}
    idx_by_run = {ir: np.where(meta["run"] == ir)[0] for ir in flevs}
    Hp = (A_m * e_m) @ A_m.conj().T
    # bare: diagonal d_f in the model space
    dmod = np.real(np.diag(Hp)) / np.real(np.diag(G))
    for L in lookup.values():
        rows = [p for p in range(n_act) if meta["lid"][p] == L["id"]]
        L["d_model"] = float(np.mean(dmod[rows])) if rows else None
    pictures = {}
    # ---- lowdin
    Gm, lamG, nd = _isqrt_h(G, opts.reg_tol)
    Cl = Gm @ A_m
    Hl = Cl @ (e_m[:, None] * Cl.conj().T)
    # ---- bare (Ritz in the projected, non-orthogonal basis; Loewdin populations)
    ritz_e, ritz_rot = {}, {}
    for ir, idx in idx_by_run.items():
        if len(idx) == 0:
            continue
        lx, Ux = np.linalg.eigh(G[np.ix_(idx, idx)])
        kx = lx > opts.reg_tol
        Xc = Ux[:, kx] / np.sqrt(lx[kx])
        e, Yr = np.linalg.eigh(Xc.conj().T @ Hp[np.ix_(idx, idx)] @ Xc)
        ritz_e[ir] = e
        ritz_rot[ir] = Ux[:, kx] @ Yr
    # ---- ionic
    Ci, iinfo = _ionic_coefficients(A_m, meta["occupied"], opts.reg_tol)
    Hi = Ci @ (e_m[:, None] * Ci.conj().T)
    defs = {"ionic": (Ci, Hi, None), "lowdin": (Cl, Hl, None),
            "bare": (Cl, Hl, (ritz_e, ritz_rot))}
    par_ionic = None
    for name, (Cx, Hx, rr) in defs.items():
        if rr is None:
            par = _picture_parents(Hx, idx_by_run, meta["lid"], meta["shell"], lookup,
                                   opts.degen_tol)
        else:
            par = _picture_parents(None, idx_by_run, meta["lid"], meta["shell"], lookup,
                                   opts.degen_tol, rot=rr[1], energies=rr[0])
        _resolve_by_dimension(par, irrep_dims)
        for ir, (idx, Y, levs) in par.items():
            for jg, R in enumerate(levs):
                R["id"] = f"{name[0].upper()}{cols[ir]}{jg}"
                R["column"] = cols[ir]
        if name == "ionic":
            par_ionic = par
        levels, compact, bands = _level_analysis(Cx, Hx, e_m, occ_m, eff_m, groups_m, meta,
                                                 par, opts)
        for L2, M in zip(levels, mlev):
            L2.update(M)
        chk = {}
        if Cx.shape[0] == Cx.shape[1]:
            chk["unitarity_max_dev"] = float(np.max(np.abs(Cx @ Cx.conj().T -
                                                           np.eye(Cx.shape[0]))))
        ev = np.sort(np.linalg.eigvalsh(Hx))
        chk["H_eigs_vs_model_levels_max_dev"] = (
            float(np.max(np.abs(ev - np.sort(e_m)))) if len(ev) == len(e_m) else None)
        chk["max_abs_sumrule_residual"] = float(max(abs(b["sumrule_residual"]) for b in bands))
        # nonbonding identities: irreps carried by one sublattice
        nbchk = []
        irr_rows: dict = {}
        for p in range(n_act):
            lab = lookup[meta["lid"][p]].get("irrep")
            for part in (lab.split("/") if lab else [None]):
                irr_rows.setdefault(part, set()).add(meta["run"][p])
        if None not in irr_rows:
            par_levels = [R for (_, _, levs) in par.values() for R in levs]
            for irr, owners in sorted(irr_rows.items()):
                if len(owners) != 1 or "/" in irr or "?" in irr:
                    continue
                pe = sorted(R["e_rel_vbm"] for R in par_levels if R["irrep"] == irr)
                ce = sorted(L2["e_rel_vbm"] for L2 in levels if L2["irrep"] == irr)
                devs = [min(ce, key=lambda x, e=e: abs(x - e)) - e for e in pe] if ce else []
                nbchk.append({"irrep": irr, "column": cols[next(iter(owners))],
                              "parents": pe, "crystal": ce, "n_parents": len(pe),
                              "n_crystal": len(ce),
                              "max_abs_dev": float(max(abs(x) for x in devs)) if devs else None})
        chk["nonbonding_identities"] = nbchk
        pic = {"parents": {cols[ir]: levs for ir, (_, _, levs) in par.items()},
               "levels": levels, "bands": compact, "checks": chk}
        if name == "ionic":
            pic["orthogonalisation"] = iinfo
            # covalency: electrons in formally empty fragment shells
            cov: dict = {}
            for m in range(len(e_m)):
                if occ_m[m] < 0.5:
                    continue
                for p in range(n_act):
                    if not meta["occupied"][p]:
                        cov[meta["shell"][p]] = cov.get(meta["shell"][p], 0.0) + \
                            float(occ_m[m]) * float(np.abs(Cx[p, m]) ** 2)
            pic["covalency_electrons"] = {k: round(v, 4) for k, v in
                                          sorted(cov.items(), key=lambda x: -x[1])}
            pic["covalency_total"] = round(sum(cov.values()), 4)
        if name == "lowdin":
            pic["G_min_eig"] = float(lamG.min())
            pic["n_dropped"] = nd
        pictures[name] = pic
        # fragment level -> parent of this picture
        for ir, (idx, Y, levs) in par.items():
            for L in flevs[ir]:
                if L["active"]:
                    best = max(levs, key=lambda R, L=L: dict(R["composition"]).get(L["id"], 0.0))
                    L.setdefault("parent", {})[name] = {"id": best["id"],
                                                         "e_rel_vbm": best["e_rel_vbm"]}
    out["pictures"] = pictures
    # ---------------------------------------------- mixing shift: filled-filled / covalent
    blk = _same_occupation_levels(Hi, par_ionic, meta["occupied"], opts.degen_tol)
    ion_par = {R["id"]: R for (_, _, levs) in par_ionic.values() for R in levs}
    for L2 in pictures["ionic"]["levels"]:
        if not L2["pop_parent"]:
            continue
        pid = next(iter(L2["pop_parent"]))
        if pid not in blk:
            continue
        R, b = ion_par[pid], blk[pid]
        L2["parent_shift"] = {
            "parent": pid, "parent_label": R["label"], "parent_e": R["e_rel_vbm"],
            "parent_share": L2["pop_parent"][pid], "class": b["class"],
            "block_e": b["e_block"],
            ("filled_filled_mixing" if b["class"] == "filled" else "empty_empty_mixing"):
                b["e_block"] - R["e_rel_vbm"],
            "covalent_shift": L2["e_rel_vbm"] - b["e_block"]}
    # ---------------------------------------------- ledger (ionic picture)
    led = []
    ion_levels = pictures["ionic"]["levels"]
    hion = np.real(np.diag(Hi))
    for ir, flev in flevs.items():
        for L in flev:
            if not L["active"]:
                continue
            carry = sorted(((L2["pop_level"].get(L["id"], 0.0) * L2["degeneracy"] /
                             L["degeneracy"], L2) for L2 in ion_levels), key=lambda x: -x[0])
            main = carry[0][1]
            par_e = L["parent"]["ionic"]["e_rel_vbm"]
            b = blk.get(L["parent"]["ionic"]["id"], {"class": "filled", "e_block": par_e})
            rows = [p for p in range(n_act) if meta["lid"][p] == L["id"]]
            h_on = float(np.mean(hion[rows]))
            led.append({
                "id": L["id"], "column": cols[ir], "label": L.get("label"),
                "occupied": L["occupied"], "degeneracy": L["degeneracy"],
                "E_raw": L["energy_raw"], "d_window": L["d_rel_vbm"], "d_model": L["d_model"],
                "ritz": L["parent"]["bare"]["e_rel_vbm"],
                "lowdin_parent": L["parent"]["lowdin"]["e_rel_vbm"], "ionic_parent": par_e,
                "ionic_onsite": h_on,
                "carried_by": [[x[1]["id"], x[1]["label"], round(x[1]["e_rel_vbm"], 4),
                                round(x[0], 4)] for x in carry[:3] if x[0] > 0.01],
                "potential_shift": L["d_model"] - L["energy_raw"] + vbm,
                "pauli_shift": par_e - L["d_model"],
                "pauli_orthogonalisation": h_on - L["d_model"],
                "intra_sublattice_mixing": par_e - h_on,
                "mixing_shift": main["e_rel_vbm"] - par_e,
                "parent_class": b["class"], "block_level": b["e_block"],
                ("filled_filled_mixing" if b["class"] == "filled" else "empty_empty_mixing"):
                    b["e_block"] - par_e,
                "covalent_shift": main["e_rel_vbm"] - b["e_block"]})
    out["ledger"] = led
    # ---------------------------------------------- inter-sublattice pairs
    pairs = []
    if S12 is not None and len(flevs) == 2:
        r1, r2 = sorted(flevs)
        rows_of: dict = {}
        for p, lid in enumerate(meta["lid"]):
            rows_of.setdefault(lid, []).append(p)
        for L1 in [L for L in flevs[r1] if L["active"]]:
            for L2 in [L for L in flevs[r2] if L["active"]]:
                g1 = [b - 1 for b in L1["bands"]]
                g2 = [b - 1 for b in L2["bands"]]
                dm = math.sqrt(min(len(g1), len(g2)))
                s = float(np.linalg.norm(S12[np.ix_(g1, g2)]) / dm)
                p1, p2 = rows_of[L1["id"]], rows_of[L2["id"]]
                rec = {"pair": [L1["id"], L2["id"]], "labels": [L1.get("label"), L2.get("label")],
                       "S_exact": s,
                       "H_bare": float(np.linalg.norm(Hp[np.ix_(p1, p2)]) / dm),
                       "H_lowdin": float(np.linalg.norm(Hl[np.ix_(p1, p2)]) / dm),
                       "H_ionic": float(np.linalg.norm(Hi[np.ix_(p1, p2)]) / dm)}
                if s > 0.02 or rec["H_ionic"] > 0.05:
                    pairs.append(rec)
    out["active_pairs"] = sorted(pairs, key=lambda x: -x["S_exact"])
    # ---------------------------------------------- sensitivity of the ionic parents
    if opts.sensitivity_windows is not None:
        variants = {}
        for em in opts.sensitivity_windows:
            fz = np.zeros(nb, bool)
            egc = np.array([np.mean(eps_raw[g]) for g in groups_c])
            for j, gc in enumerate(groups_c):
                fz[gc] = egc[j] <= vbm + em
            variants[f"window {em:g} eV"] = (np.array([Amats[ir][b, :] for ir, i, b in act]),
                                             fz)
        if Amats_none is not None:
            variants["no cross-sphere"] = (np.array([Amats_none[ir][b, :]
                                                     for ir, i, b in act]), frozen0)
        sens: dict = {}
        occ_rows = meta["occupied"]
        for tag, (A_alt, frozen_alt) in variants.items():
            frozen_alt, _ = _q_filter(A_alt, frozen_alt, outer, groups_c, clev,
                                      opts.frozen_qmin)
            Am2, em2, _, _ = build_model_space(A_alt, eps, frozen_alt, outer=outer)
            C2, _ = _ionic_coefficients(Am2, occ_rows, opts.reg_tol)
            H2 = C2 @ (em2[:, None] * C2.conj().T)
            par2 = _picture_parents(H2, idx_by_run, meta["lid"], meta["shell"], lookup,
                                    opts.degen_tol)
            _resolve_by_dimension(par2, irrep_dims)
            sens[tag] = {}
            for ir, (_, _, levs) in par2.items():
                for R in levs:
                    key, n = f"{cols[ir]}:{R['label']}", 1
                    # same label twice (no label): spinor runs keep both; the scalar
                    # path keeps the published keys (last one wins)
                    while ctx["spinor"] and key in sens[tag]:
                        n += 1
                        key = f"{cols[ir]}:{R['label']} ({n})"
                    sens[tag][key] = round(R["e_rel_vbm"], 4)
        out["sensitivity_ionic_parents"] = sens
    return out


# =============================================================== one k point
def _padded(nb_rows, ncols, rows, block):
    """Full-size matrix with the stored rows and NaN elsewhere (trimmed caches)."""
    if len(rows) == nb_rows and np.array_equal(rows, np.arange(nb_rows)):
        return np.asarray(block)
    full = np.full((nb_rows, ncols), np.nan + 1j * np.nan, complex)
    full[np.asarray(rows, int)] = block
    return full


def _padded_pair(nb_a, nb_b, rows_a, rows_b, block):
    if len(rows_a) == nb_a and len(rows_b) == nb_b:
        return np.asarray(block)
    full = np.full((nb_a, nb_b), np.nan + 1j * np.nan, complex)
    full[np.ix_(np.asarray(rows_a, int), np.asarray(rows_b, int))] = block
    return full


def _kramers(ctx, eps_all, tol, emax):
    """Kramers check of every run: odd level groups and pair splittings."""
    kram = []
    for ir, run in enumerate(ctx["runs"]):
        e = eps_all[ir]
        grp = group_levels(e, tol)
        # window: up to VBM (crystal) / HOMO (fragment) + emax and below the top
        # 5 % of the bands (not converged in a VASP run; reported separately)
        top = min((ctx["vbm"] if ir == 0 else ctx["homos"][ir]) + (emax if emax else 1e9),
                  float(e[int(0.95 * len(e))]) - 1e-6)
        inw = [g for g in grp if np.mean(e[g]) <= top]
        odd_w = [[int(b + 1) for b in g] for g in inw if len(g) % 2]
        nw = sum(len(g) for g in inw)
        pair = np.abs(e[1:nw:2] - e[0:nw:2]) if not odd_w else np.zeros(0)
        kram.append({"run": run["path"], "window_top_eV": float(top),
                     "odd_groups_in_window": odd_w,
                     "max_pair_splitting_in_window_eV": float(pair.max()) if len(pair)
                     else None,
                     "odd_groups_above_window": [[int(b + 1) for b in g] for g in grp
                                                 if len(g) % 2 and np.mean(e[g]) > top]})
    return kram


def _analyse_kpoint(kp, ctx, opts):
    """Crystal and fragment levels plus the phase-3 analysis at one k point."""
    t0 = time.time()
    tol = opts.degen_tol
    runs = ctx["runs"]
    vbm, homos = ctx["vbm"], ctx["homos"]
    columns = ctx["columns"]
    modes = list(kp.overlaps)
    main_mode = opts.cross_sphere
    eps_c = np.asarray(kp.eigenvalues[0], float)
    occ_c = np.asarray(kp.occupations[0], float)
    nb_c = len(eps_c)
    groups_c = group_levels(eps_c, tol)
    egc = np.array([np.mean(eps_c[g]) for g in groups_c])
    win = np.ones(len(groups_c), bool) if opts.emax is None else (egc <= vbm + opts.emax)
    out = {"name": kp.name, "frac": [float(x) for x in kp.frac], "index": int(kp.index) + 1,
           "n_planewaves": int(kp.n_planewaves)}
    # --------------------------------------------------- tests
    tests = copy.deepcopy(kp.tests)
    t2 = []
    for ir, rec in enumerate(tests.get("T2_fragments", []), start=1):
        new = {"run": rec.get("run"), "column": columns[ir - 1]}
        new.update({k: v for k, v in rec.items() if k != "run"})
        t2.append(new)
    tests["T2_fragments"] = t2
    if ctx["spinor"]:
        tests["kramers"] = _kramers(ctx, [np.asarray(e, float) for e in kp.eigenvalues],
                                    tol, opts.emax)
    out["tests"] = tests

    def sphere_dict(r):
        return {tuple(key): np.asarray(kp.sphere_weights[r][i])
                for i, key in enumerate(kp.sphere_keys[r])}
    # --------------------------------------------------- crystal column
    sw_c = sphere_dict(0)
    shells_c = shell_summary(sw_c, runs[0]["site_species"], groups_c)
    clev = []
    for ig, g in enumerate(groups_c):
        clev.append({
            "id": f"c{ig}", "bands": [b + 1 for b in g], "energy_raw": float(egc[ig]),
            "energy_rel_vbm": float(egc[ig] - vbm), "degeneracy": len(g),
            "occupation": float(ctx["ef"] * np.sum(occ_c[g])),
            "label": None, "irrep": None,
            "shells": {k: round(v, 4) for k, v in shells_c[ig].items() if v > 0.005},
            "in_window": bool(win[ig]),
            "truncated_top": bool(g[-1] == nb_c - 1)})
    ir_c = irrep_group_labels(_labels_of(kp, 0), groups_c)
    if ir_c:
        seen: dict = {}
        for L, lab in zip(clev, ir_c):
            L["irrep_crystod"] = L["irrep"]
            if lab:
                L["irrep"] = lab
                if "/" not in lab and "?" not in lab:
                    seen[lab] = seen.get(lab, 0) + 1
                    L["label"] = f"{lab} #{seen[lab]}"
                else:
                    L["label"] = lab
    out["crystal"] = {"run": runs[0]["path"], "levels": clev}
    # --------------------------------------------------- fragment columns
    frags = []
    Amats_all, Amats_none, flevs_all, groups_all = {}, {}, {}, {}
    full_rows = {}
    for ir in range(1, len(runs)):
        col = columns[ir - 1]
        eps_f = np.asarray(kp.eigenvalues[ir], float)
        occ_f = np.asarray(kp.occupations[ir], float)
        nb_f = len(eps_f)
        rows = np.asarray(kp.rows[ir - 1], int)
        full_rows[ir] = len(rows) == nb_f
        groups_f = group_levels(eps_f, tol)
        groups_all[ir] = groups_f
        egf = np.array([np.mean(eps_f[g]) for g in groups_f])
        A = {m: _padded(nb_f, nb_c, rows, kp.overlaps[m][ir - 1]) for m in modes}
        Amain = A[main_mode]
        # summed |a|^2 between level groups (gauge invariant)
        mats = {m: np.array([[float((np.abs(A[m][np.ix_(gf, gc)]) ** 2).sum())
                              for gc in groups_c] for gf in groups_f]) for m in modes}
        P = mats[main_mode]
        sw_f = sphere_dict(ir)
        shells_f = shell_summary(sw_f, runs[ir]["site_species"], groups_f)
        fw = np.asarray(kp.pointcharge_weights[ir], float)
        ef_f = float(runs[ir]["electrons_per_band"])
        flev = []
        for ig, g in enumerate(groups_f):
            dF = len(g)
            row = np.where(win, P[ig], 0.0)
            Wsum = row.sum()
            W = Wsum / dF
            d = float(np.dot(row, egc) / Wsum) if Wsum > 0 else float("nan")
            spread = float(math.sqrt(max(np.dot(row, (egc - d) ** 2) / Wsum, 0.0))) \
                if Wsum > 0 else float("nan")
            order = np.argsort(-row)
            main = [[clev[j]["id"], round(float(row[j] / dF), 5)] for j in order
                    if row[j] / dF > opts.main_threshold]
            alt = {}
            for m in CROSS_SPHERE_MODES:
                if m not in mats:
                    continue
                rm = np.where(win, mats[m][ig], 0.0)
                alt[m] = {"W": float(rm.sum() / dF),
                          "d_raw": float(np.dot(rm, egc) / rm.sum()) if rm.sum() > 0 else None}
            scan = {}
            for em in opts.window_scan:
                wm = egc <= vbm + em
                rm = np.where(wm, P[ig], 0.0)
                scan[f"{em:g}"] = {"W": float(rm.sum() / dF),
                                   "d_raw": float(np.dot(rm, egc) / rm.sum())
                                   if rm.sum() > 0 else None}
            own_w = float(sum(shells_f[ig].values()))
            for_w = float(np.mean(fw[g]))
            flev.append({
                "id": f"{col}{ig}", "bands": [b + 1 for b in g],
                "energy_raw": float(egf[ig]), "degeneracy": dF,
                "occupation": float(ef_f * np.sum(occ_f[g])),
                "label": None, "irrep": None, "label_crystod": None, "irrep_crystod": None,
                "shells": {k: round(v, 4) for k, v in shells_f[ig].items() if v > 0.005},
                "own_sphere_weight": own_w, "pointcharge_sphere_weight": for_w,
                "pointcharge_site_state": bool(own_w < 0.3 and for_w > own_w),
                "W": float(W), "W_low": bool(W < 0.5),
                "W_all": float(P[ig].sum() / dF), "d_raw": d, "d_rel_vbm": d - vbm,
                "delta": d - float(egf[ig]), "spread": spread,
                "max_single_share": float(row.max() / Wsum) if Wsum > 0 else None,
                "above_homo": float(egf[ig] - homos[ir]),
                "main_crystal_levels": main, "alt": alt, "window_scan": scan,
                "truncated_top": bool(g[-1] == len(eps_f) - 1)})
        # shells with principal quantum numbers, irreps, labels, active space
        assign_shell_names(flev, runs[ir], ctx["paw"], ctx["shell_over"])
        ir_f = irrep_group_labels(_labels_of(kp, ir), groups_f)
        if ir_f:
            for L, lab in zip(flev, ir_f):
                if lab:
                    L["irrep"] = lab
        twins: dict = {}
        for L in flev:
            if L.get("shell") and L.get("irrep"):
                twins.setdefault((L["shell"], L["irrep"]), []).append(L)
        for (shell, irr), group in twins.items():
            for t, L in enumerate(group):
                L["label"] = f"{shell} {irr}" + (f"#{t + 1}" if len(group) > 1 else "")
        for L in flev:
            if not L.get("label") and L.get("shell"):
                L["label"] = L["shell"]
        select_active(flev, ctx["roles"].get(ir, "cation"), opts)
        Amats_all[ir] = Amain
        Amats_none[ir] = A.get("none")
        flevs_all[ir] = flev
        # fragment-basis crystal Hamiltonian and truncated overlap
        if full_rows[ir]:
            bandwin = np.zeros(nb_c, bool)
            for j, gc in enumerate(groups_c):
                bandwin[gc] = win[j]
            Aw = Amain[:, bandwin]
            Hb = np.conj(Aw) @ np.diag(eps_c[bandwin]) @ Aw.T
            Sb = np.conj(Aw) @ Aw.T
            Hn = block_norms(Hb, groups_f, groups_f)
            Sn = block_norms(Sb, groups_f, groups_f)
            offd = []
            for i1 in range(len(groups_f)):
                for i2 in range(i1 + 1, len(groups_f)):
                    h = Hn[i1, i2]
                    if h > opts.offdiag_threshold:
                        d1, d2 = flev[i1]["d_raw"], flev[i2]["d_raw"]
                        sep = abs(d1 - d2)
                        offd.append({"pair": [flev[i1]["id"], flev[i2]["id"]], "h": float(h),
                                     "s_truncated": float(Sn[i1, i2]), "d_sep": float(sep),
                                     "ratio": float(h / sep) if sep > 0 else None,
                                     "second_order_shift": float(h * h / sep)
                                     if sep > 0 else None})
            h_blocks, s_blocks = Hn.round(5).tolist(), Sn.round(5).tolist()
        else:
            h_blocks = s_blocks = None
            offd = []
        frags.append({"run": runs[ir]["path"], "column": col, "levels": flev,
                      "a2": [[float(x) for x in r] for r in P],
                      "a2_other_modes": {m: [[float(x) for x in r] for r in mats[m]]
                                         for m in CROSS_SPHERE_MODES
                                         if m != main_mode and m in mats},
                      "H_blocks": h_blocks, "S_blocks": s_blocks,
                      "H_offdiag": sorted(offd, key=lambda x: -x["h"])})
        if ir - 1 < len(tests["T2_fragments"]):
            tests["T2_fragments"][ir - 1]["bessel_inequality_max_W"] = float(
                _nanmax([L["alt"][main_mode]["W"] for L in flev]))
        # crystal populations per crystal partner; 'drawn' = the fragment levels a
        # diagram would show (within pop_window of the fragment HOMO), without the
        # diffuse states that live on the point-charge sites
        drawn = np.array([L["above_homo"] <= opts.pop_window
                          and not L["pointcharge_site_state"] for L in flev])
        for L, dr in zip(flev, drawn):
            L["drawn"] = bool(dr)
        Pz = np.where(np.isnan(P), 0.0, P)
        for jg, gc in enumerate(groups_c):
            pops = Pz[:, jg] / len(gc)
            clev[jg].setdefault("population", {})[col] = float(pops.sum())
            clev[jg].setdefault("population_drawn", {})[col] = float(pops[drawn].sum())
            clev[jg].setdefault("fragment_levels", {})[col] = [
                [flev[i]["id"], round(float(pops[i]), 5)] for i in np.argsort(-pops)
                if pops[i] > opts.main_threshold]
    out["fragments"] = frags
    # --------------------------------------------------- fragment-fragment
    S12 = None
    if len(runs) == 3:
        stored = kp.fragment_overlaps.get(main_mode, {}).get((1, 2))
        if stored is not None:
            nb1, nb2 = len(kp.eigenvalues[1]), len(kp.eigenvalues[2])
            S12 = _padded_pair(nb1, nb2, kp.rows[0], kp.rows[1], stored)
            if full_rows[1] and full_rows[2]:
                A1, gr1 = Amats_all[1], groups_all[1]
                A2, gr2 = Amats_all[2], groups_all[2]
                H12 = np.conj(A1) @ np.diag(eps_c) @ A2.T
                S12t = np.conj(A1) @ A2.T
                out["fragment_fragment"] = {
                    "rows": [L["id"] for L in frags[0]["levels"]],
                    "cols": [L["id"] for L in frags[1]["levels"]],
                    "S_exact": block_norms(S12, gr1, gr2).round(5).tolist(),
                    "S_spectral": block_norms(S12t, gr1, gr2).round(5).tolist(),
                    "H_spectral": block_norms(H12, gr1, gr2).round(5).tolist()}
    out["label_disagreements"] = []
    cols = {ir: columns[ir - 1] for ir in range(1, len(runs))}
    pctx = dict(ctx)
    pctx["cols"] = cols
    out["phase3"] = phase3_analysis(
        pctx, eps_c, occ_c, groups_c, clev, win, flevs_all, Amats_all, S12,
        Amats_none if all(v is not None for v in Amats_none.values()) else None, opts,
        irrep_dims={**irrep_dimensions(kp, tol),
                    **(ctx.get("group_dims") or {}).get(kp.name, {})})
    out["seconds"] = round(time.time() - t0, 2)
    return out


# =============================================================== SOC bridge
def spin_orbit_table(cache, kp, symprec: float = 1e-5):
    """``{single-valued irrep: [double-valued irreps]}`` of Gamma x D(1/2) at one k.

    :func:`crystod.wavecar_irreps.spin_orbit_compatibility` on the crystal
    cell of the cache, in the names the spinor levels carry: CrystOD's
    ``-K<n><p>`` names, or the Bilbao names of IrRep where the labels came
    from IrRep's JSON (the CrystOD-to-Bilbao map the label provider measured,
    ``cache.meta["irrep_name_maps"]``).

    Args:
        cache: The spin-orbit :class:`~crystod.vasp_wavecar.OverlapCache`.
        kp: One of its :class:`~crystod.vasp_wavecar.KPointOverlaps`.
        symprec: Symmetry tolerance.

    Returns:
        The table, or ``None`` when the symmetry data cannot be built.
    """
    from .wavecar_irreps import spin_orbit_compatibility

    run = cache.runs[0]
    name_map = (cache.meta.get("irrep_name_maps") or {}).get(kp.name)
    try:
        return spin_orbit_compatibility(
            np.asarray(run["lattice"], float), np.asarray(run["positions"], float),
            _species_numbers(run["site_species"]), list(kp.frac), symprec=symprec,
            name_map=name_map or None)
    except ValueError:
        return None


def scalar_bridge(ksoc, kscal, dvbm=None, compatibility=None):
    """Scalar-relativistic levels next to every SOC crystal level and parent.

    Crystal levels: the compatible scalar levels ranked by the parent-shell
    population they share with the SOC level (sum over shells of the smaller
    population); those sharing >= 0.25 are kept (else the best one).  A
    scalar level of irrep Gamma is compatible with a SOC level whose irrep is
    in Gamma x D(1/2) (``compatibility``, :func:`spin_orbit_table`).

    Args:
        ksoc: SOC k-point record.
        kscal: Scalar k-point record of the same k.
        dvbm: VBM(SOC) - VBM(scalar) on the absolute eigenvalue scales.
        compatibility: ``{single-valued irrep: [double-valued irreps]}`` at
            this k, in the names of the SOC labels.

    Returns:
        ``{"crystal": [...], "parents": [...]}`` or ``None``.
    """
    p3s, p3r = ksoc.get("phase3"), kscal.get("phase3")
    if not p3s or not p3r or compatibility is None:
        return None
    ion_s, ion_r = p3s["pictures"]["ionic"], p3r["pictures"]["ionic"]
    par_s = {R["id"]: R for levs in ion_s["parents"].values() for R in levs}

    def compatible(irr_double, L):
        comp = compatibility.get(L.get("irrep"))
        return bool(comp and irr_double and
                    any(part in comp for part in irr_double.split("/")))

    def matches(irr_double, items, shell=None):
        return [[L.get("label"), round(L["e_rel_vbm"], 4), L.get("bond")] for L in items
                if compatible(irr_double, L) and (shell is None or L.get("shell") == shell)]

    def ranked(L):
        ps = L.get("pop_shell") or {}
        cand = []
        for S in ion_r["levels"]:
            if compatible(L.get("irrep"), S):
                pr = S.get("pop_shell") or {}
                share = sum(min(v, pr.get(s, 0.0)) for s, v in ps.items())
                cand.append([S.get("label"), round(S["e_rel_vbm"], 4), S.get("bond"),
                             round(share, 3)])
        cand.sort(key=lambda x: -x[3])
        return [x for x in cand if x[3] >= 0.25] or [x for x in cand[:1] if x[3] >= 0.05]
    crystal = []
    for L in ion_s["levels"]:
        sl = ranked(L)
        rec = {
            "id": L["id"], "label": L.get("label"), "irrep": L.get("irrep"),
            "e_rel_vbm": L["e_rel_vbm"], "degeneracy": L["degeneracy"],
            "occupation": L["occupation"], "bond": L["bond"],
            "cohp_inter": L["cohp_inter"], "cohp_inter_split": L["cohp_inter_split"],
            "cohp_pairs": dict(list(L["cohp_pairs"].items())[:6]),
            "pop_shell": dict(list((L.get("pop_shell") or {}).items())[:4]),
            "parents": [[par_s[i]["label"], round(v, 4)] for i, v in
                        list(L["pop_parent"].items())[:4] if i in par_s],
            "scalar_levels": sl}
        if sl and sl[0][3] >= 0.25:
            rec["shift_rel_vbm_eV"] = round(L["e_rel_vbm"] - sl[0][1], 4)
            if dvbm is not None:
                rec["shift_absolute_eV"] = round(L["e_rel_vbm"] - sl[0][1] + dvbm, 4)
        crystal.append(rec)
    parents = []
    scal_par = [R for levs in ion_r["parents"].values() for R in levs]
    for R in par_s.values():
        parents.append({"id": R["id"], "label": R["label"], "irrep": R["irrep"],
                        "e_rel_vbm": R["e_rel_vbm"], "degeneracy": R["degeneracy"],
                        "scalar_parents": matches(R.get("irrep"), scal_par, R.get("shell"))})
    return {"crystal": crystal, "parents": parents}


# =============================================================== driver
def _roles(runs) -> dict:
    """Role of every fragment run from the sign of its point charges."""
    roles = {}
    for i, r in enumerate(runs[1:], start=1):
        q = [c for c, own in zip(r["charges"], r["own"]) if not own and c is not None]
        roles[i] = ("cation" if q and all(x < 0 for x in q) else
                    "anion" if q and all(x > 0 for x in q) else "mixed")
    return roles


def analyse_cache(cache, *, columns=None, active_shells=None, frozen_window: float | None = 14.0,
                  frozen_qmin: float = 0.5, cross_sphere: str = "bessel",
                  scalar_reference=None, label_provider=None, compound: str | None = None,
                  report=print, **options) -> dict:
    """Stage 2: the full analysis from an overlap cache.

    Args:
        cache: :class:`~crystod.vasp_wavecar.OverlapCache` or the path of one.
        columns: Column name of every fragment run (default: the names stored
            in the cache, else ``frag1``, ``frag2``, ...).
        active_shells: ``None`` = the default minimal valence rule
            (:func:`default_active_shells`), ``"auto-full"`` = the automatic
            rule (full caches only), or explicit shells (``["Ti 3d", ...]``).
        frozen_window: Frozen window above the VBM (eV); ``None`` = all bands.
        frozen_qmin: Window levels with less projection on the active span
            are left to the disentanglement.
        cross_sphere: Cross-sphere mode of the overlaps (must be in the cache).
        scalar_reference: SOC runs: scalar results (dict or JSON path) of
            the same compound for the scalar bridge.
        label_provider: Optional ``f(run_index, k_name, k_frac) -> labels``;
            replaces the labels stored in the cache.
        compound: Name for the report title.
        report: Print function (``None`` = silent).
        **options: Further :class:`OverlapOptions` fields (``degen_tol``,
            ``cohp_tol``, ``sensitivity_windows``, ...).

    Returns:
        The results dict (see :func:`write_json`).

    Raises:
        SystemExit: Unknown option, a mode missing in the cache, the automatic
            rule on a trimmed cache.
    """
    say = report or (lambda *a, **k: None)
    t0 = time.time()
    if isinstance(cache, (str, os.PathLike)):
        cache = OverlapCache.load(str(cache))
    unknown = set(options) - _OPTION_NAMES
    if unknown:
        raise SystemExit(f"ERROR: unknown analysis option(s): {', '.join(sorted(unknown))}.")
    opts = OverlapOptions(emax=frozen_window, frozen_qmin=frozen_qmin,
                          cross_sphere=cross_sphere, **options)
    if opts.emax is not None and opts.emax < 0:
        opts.emax = None
    if cross_sphere not in cache.modes:
        raise SystemExit(f"ERROR: cross-sphere mode {cross_sphere!r} is not in the cache "
                         f"(has {', '.join(cache.modes)}).")
    if label_provider is not None:
        cache.attach_labels(label_provider)
        if hasattr(label_provider, "name_maps"):
            # the Bilbao name map belongs to the labels just attached
            cache.meta.pop("irrep_name_maps", None)
            if label_provider.name_maps:
                cache.meta["irrep_name_maps"] = label_provider.name_maps
    runs = cache.runs
    nfrag = len(runs) - 1
    if columns is None:
        columns = cache.meta.get("columns") or [None] * nfrag
    columns = [c or f"frag{i + 1}" for i, c in enumerate(list(columns))]
    if len(columns) != nfrag:
        raise SystemExit(f"ERROR: {len(columns)} column names for {nfrag} fragment runs.")
    if active_shells is None:
        shells, rule = default_active_shells(cache), "default"
    elif isinstance(active_shells, str) and active_shells == "auto-full":
        if cache.trimmed:
            raise SystemExit("ERROR: the automatic active-space rule needs the overlaps of "
                             "every fragment band; this cache is trimmed.")
        shells, rule = None, "auto-full"
    else:
        shells = [active_shells] if isinstance(active_shells, str) else list(active_shells)
        shells, rule = [" ".join(s.split()) for s in shells], "explicit"
    opts.active_shells = shells
    if shells:
        by_el: dict = {}
        for s in shells:
            by_el.setdefault(s.split()[0], []).append(s.split()[1])
        how = "POTCAR valence + standard empty shell" if rule == "default" else rule
        say(f"active shells ({how}): "
            + "; ".join(f"{el} {' '.join(v)}" for el, v in by_el.items()))
    else:
        say("active shells: automatic rule (occupied levels + the lowest empty manifold of "
            "every cation (element, l))")
    crystal = runs[0]
    spinor = bool(crystal["spinor"])
    vbm = float(cache.meta["vbm_raw"])
    homos = {i: float(r["homo_raw"]) for i, r in enumerate(runs)}
    roles = _roles(runs)
    ctx = {"runs": runs, "columns": columns, "roles": roles, "vbm": vbm, "homos": homos,
           "spinor": spinor, "ef": float(crystal["electrons_per_band"]),
           "paw": cache.meta["paw"], "shell_over": parse_shell_overrides(opts.shell_n),
           "group_dims": {kp.name: little_group_dimensions(cache, kp, opts.symprec)
                          for kp in cache.kpoints}}
    result = {
        "engine": _engine_name(),
        "method": {
            "formula": "a_fn = <phi_f|psi_n> (all-electron PAW); W_f = sum_n |a_fn|^2; "
                       "d_f = sum_n |a_fn|^2 eps_n / W_f = <phi_f|H_crystal|phi_f> "
                       "(within the crystal band window)",
            "cross_sphere": opts.cross_sphere,
            "emax_rel_vbm": opts.emax,
            "degeneracy_tol_eV": opts.degen_tol,
            "notes": "Level groups are degenerate multiplets (tolerance degeneracy_tol_eV). "
                     "'a2'[F][N] = sum over f in F and n in N of |a_fn|^2 (divide by deg F "
                     "for the weight per fragment partner, by deg N for the population per "
                     "crystal partner). 'W', 'main_crystal_levels' are per fragment partner; "
                     "crystal 'population' and 'fragment_levels' per crystal partner. "
                     "'H_blocks'/'S_blocks' = ||X_FF'||_F/sqrt(min deg) of the fragment-basis "
                     "crystal Hamiltonian / overlap from the truncated spectral sums; "
                     "'fragment_fragment' = same between the two fragments, 'S_exact' from the "
                     "direct PAW overlap of the two fragment runs. delta = d_raw - energy_raw."},
        "crystal": {"run": crystal["path"], "species": crystal["species"],
                    "formula": "".join(f"{sp}{n if n > 1 else ''}" for sp, n in
                                       zip(crystal["species"], crystal["counts"])),
                    "nbands": crystal["nbands"], "nelect": crystal["nelect"],
                    "vbm_raw": vbm, "encut": crystal["encut"], "nk": crystal["nk"]},
        "fragment_runs": [{"run": r["path"], "species": r["species"], "formula": r["formula"],
                           "role": roles.get(i + 1), "column": columns[i],
                           "own_sites": [a + 1 for a, o in enumerate(r["own"]) if o],
                           "pointcharge_sites": [a + 1 for a, o in enumerate(r["own"]) if not o],
                           "nbands": r["nbands"], "nelect": r["nelect"],
                           "homo_raw": homos[i + 1]} for i, r in enumerate(runs[1:])],
        "paw": {sp: {k: v for k, v in d.items()
                     if k in ("title", "lps", "rdep_A", "qpaw_file_vs_computed_max", "Q", "M",
                              "duality_check_q_space")}
                for sp, d in cache.meta["paw"].items()},
        "kpoints": [],
    }
    if spinor:
        result["crystal"]["spinor"] = True
    labelled = sorted({r for kp in cache.kpoints for r, lab in enumerate(kp.labels or [])
                       if lab is not None})
    result["phase2_settings"] = {
        "active_rule": active_rule_text(opts),
        "roles": {columns[i - 1]: r for i, r in roles.items()},
        "reg_tol": opts.reg_tol, "cohp_tol": opts.cohp_tol,
        "completeness_min": opts.completeness_min,
        "irrep_runs": [runs[i]["path"] for i in labelled],
        "conventions": "A[f,n] = <phi_f|psi_n> over the window; G = A A^H; "
                       "Hp = A (eps - E_VBM) A^H; c'_n = G^-1/2 A[:,n]; "
                       "H' = G^-1/2 Hp G^-1/2; COHP_n(f,g) = 2 Re[conj(c'_nf) H'_fg c'_ng] "
                       "(negative = bonding); Ritz: Hp_XX v = e G_XX v per sublattice; "
                       "Loewdin populations mapped on Ritz levels by the block-Loewdin "
                       "rotation G_XX^1/2 V_X (sum-preserving)."}
    for kp in cache.kpoints:
        kres = _analyse_kpoint(kp, ctx, opts)
        result["kpoints"].append(kres)
        t = kres["tests"]
        say(f"[{kp.name}] k#{kp.index + 1}  T1={t.get('T1_crystal_max_dev', float('nan')):.2e}"
            f"  T2={[float('%.2e' % x['T2_max_dev']) for x in t.get('T2_fragments', [])]}"
            f"  active {kres['phase3'].get('n_active', 0)}  ({kres['seconds']} s)")
    if scalar_reference is not None:
        if isinstance(scalar_reference, (str, os.PathLike)):
            with open(scalar_reference) as fh:
                ref = json.load(fh)
            ref_path = os.path.abspath(str(scalar_reference))
        else:
            ref, ref_path = scalar_reference, None
        ref_k = {kp["name"]: kp for kp in ref.get("kpoints", [])}
        try:
            dvbm = float(result["crystal"]["vbm_raw"]) - float(ref["crystal"]["vbm_raw"])
        except (KeyError, TypeError, ValueError):
            dvbm = None
        for kp, kres in zip(cache.kpoints, result["kpoints"]):
            if kres["name"] in ref_k and kres.get("phase3"):
                kres["phase3"]["scalar_bridge"] = scalar_bridge(
                    kres, ref_k[kres["name"]], dvbm,
                    compatibility=spin_orbit_table(cache, kp, opts.symprec))
        result["scalar_reference"] = ref_path
        result["vbm_soc_minus_scalar_eV"] = dvbm
    result["seconds"] = round(time.time() - t0, 1)
    opt_dict = asdict(opts)
    result["crystod"] = {
        "version": _version(), "module": "crystod.crystal_orbital_overlap",
        "compound": compound,
        "active_shell_rule": rule, "active_shells": shells,
        "options": opt_dict, "columns": columns,
        "cache": {"source": cache.meta.get("source"), "schema": cache.meta.get("schema"),
                  "trimmed": cache.trimmed, "modes": cache.modes,
                  "created": cache.meta.get("created"),
                  "stage1_seconds": cache.meta.get("seconds_stage1")},
        "potcar": {sp: {"title": d.get("title"), "vrhfin": d.get("vrhfin"),
                        "zval": d.get("zval"), "valence_shells": d.get("valence_shells")}
                   for sp, d in cache.meta["paw"].items()}}
    return result


def _wavecar_stamps(run_dirs) -> list:
    """``[size, mtime]`` of the WAVECAR of every run (stale-cache detection)."""
    stamps = []
    for directory in run_dirs:
        path = os.path.join(directory, "WAVECAR")
        stamps.append([os.path.getsize(path), int(os.path.getmtime(path))]
                      if os.path.isfile(path) else None)
    return stamps


def _cache_verdict(cache, crystal_dir, fragment_dirs, kpoints, cross_sphere) -> tuple:
    """Can a stored cache serve a request?

    Returns:
        ``("covers", cache)`` with the cache restricted to the requested k
        points (in the requested order); ``("foreign", reason)`` for a cache
        of other runs or a trimmed one, which must not be overwritten;
        ``("partial", reason)`` when the runs are the same but a k point or
        the cross-sphere mode is missing, or a WAVECAR changed since.
    """
    paths = [os.path.abspath(p) for p in [crystal_dir, *fragment_dirs]]
    if cache.trimmed:
        return "foreign", "it is a trimmed cache (a regression fixture)"
    if [r["path"] for r in cache.runs] != paths:
        return "foreign", "it holds the overlaps of other runs (" + ", ".join(
            r["path"] for r in cache.runs) + ")"
    stamps = cache.meta.get("wavecar_stamps")
    if stamps and stamps != _wavecar_stamps(paths):
        return "partial", "a WAVECAR changed since it was written"
    if cross_sphere not in cache.modes:
        return "partial", f"it has no {cross_sphere!r} overlaps"
    if isinstance(kpoints, str):
        if kpoints.lower() == "all" and len(cache.kpoints) == cache.runs[0]["nk"]:
            return "covers", cache
        return "partial", "it does not hold every k point"
    have = {kp.name: kp for kp in cache.kpoints}
    chosen = []
    for name, cands in kpoints:
        kp = have.get(name)
        if kp is None or find_kpoint(np.array([kp.frac]), cands)[0] is None:
            return "partial", f"it has no k point {name}"
        chosen.append(kp)
    if [kp.name for kp in chosen] == [kp.name for kp in cache.kpoints]:
        return "covers", cache
    return "covers", OverlapCache(cache.meta, chosen)


def run_overlap_analysis(crystal_dir, fragment_dirs, *, columns, kpoints, active_shells=None,
                         frozen_window: float | None = 14.0, frozen_qmin: float = 0.5,
                         cross_sphere: str = "bessel", cache_path: str | None = None,
                         label_provider=None, scalar_reference=None, potcar: str | None = None,
                         modes=CROSS_SPHERE_MODES, compound: str | None = None,
                         report=print, **options) -> dict:
    """Both stages: overlaps from the VASP runs, then the analysis.

    Args:
        crystal_dir: Crystal run directory (POSCAR, POTCAR, WAVECAR).
        fragment_dirs: Sublattice run directories (Va point charges).
        columns: Column name of every fragment run (``["left", "right"]``).
        kpoints: ``[(name, frac), ...]`` (``frac`` one fractional k point or a
            list of equivalent candidates) or ``"all"``.
        active_shells: See :func:`analyse_cache`.
        frozen_window: Frozen window above the VBM (eV).
        frozen_qmin: See :func:`analyse_cache`.
        cross_sphere: Cross-sphere mode used by the analysis.
        cache_path: ``.npz`` cache: reused (with the labels it holds) when it
            covers the request -- same runs, unchanged WAVECARs, every
            requested k point (a cache of more k points serves a subset);
            rebuilt and rewritten when the runs are the same but it does not
            cover the request; never overwritten when it belongs to other
            runs or is trimmed (an ERROR).
        label_provider: ``f(run_index, k_name, k_frac) -> list[str]`` (one
            irrep label per band; run 0 = crystal); its ``name_maps``
            attribute, when present, is stored in the cache for the scalar
            bridge (:func:`spin_orbit_table`).
        scalar_reference: See :func:`analyse_cache`.
        potcar: POTCAR with the PAW data (default: the crystal's).
        modes: Cross-sphere modes stored in a new cache.
        compound: Name for the report title.
        report: Print function (``None`` = silent).
        **options: Further :class:`OverlapOptions` fields.

    Returns:
        The results dict.

    Raises:
        SystemExit: The cache file belongs to other runs, or the stage-1
            errors of :func:`crystod.vasp_wavecar.build_overlap_cache`.
    """
    say = report or (lambda *a, **k: None)
    cache = None
    if cache_path and not str(cache_path).endswith(".npz"):
        cache_path = str(cache_path) + ".npz"     # numpy appends it when saving
    if cache_path and os.path.isfile(cache_path):
        verdict, found = _cache_verdict(OverlapCache.load(cache_path), crystal_dir,
                                        fragment_dirs, kpoints, cross_sphere)
        if verdict == "foreign":
            raise SystemExit(f"ERROR: --vasp-cache {cache_path}: {found}; give another "
                             "file name (it is not overwritten).")
        if verdict == "covers":
            cache = found
            source = getattr(label_provider, "source", None)
            if (label_provider is not None and any(kp.labels for kp in cache.kpoints)
                    and source == cache.meta.get("label_source")):
                label_provider = None        # the labels it was built with
            say(f"overlap cache: {cache_path} (reused" + ("; the irrep labels are "
                "recomputed for the labels asked for" if label_provider is not None
                else "; no WAVECAR is read") + ")")
            if label_provider is not None:
                cache.meta["label_source"] = source
        else:
            say(f"overlap cache: {cache_path} is rebuilt ({found})")
    if cache is None:
        cache = build_overlap_cache(crystal_dir, fragment_dirs, kpoints, modes=modes,
                                    potcar=potcar, label_provider=label_provider,
                                    report=report)
        cache.meta["columns"] = list(columns)
        cache.meta["wavecar_stamps"] = _wavecar_stamps([crystal_dir, *fragment_dirs])
        cache.meta["label_source"] = getattr(label_provider, "source", None)
        name_maps = getattr(label_provider, "name_maps", None)
        if name_maps:
            cache.meta["irrep_name_maps"] = name_maps
        if cache_path:
            cache.save(cache_path)
            cache.meta["source"] = os.path.abspath(cache_path)
            say(f"overlap cache written: {cache_path}")
        label_provider = None                # already attached
    return analyse_cache(cache, columns=columns, active_shells=active_shells,
                         frozen_window=frozen_window, frozen_qmin=frozen_qmin,
                         cross_sphere=cross_sphere, scalar_reference=scalar_reference,
                         label_provider=label_provider, compound=compound, report=report,
                         **options)


# =============================================================== JSON
def _clean(o):
    """Strict JSON: NaN/inf -> null, keys starting with '_' dropped."""
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        return _clean(o.item())
    return o


def write_json(results: dict, path: str) -> str:
    """Write the results as strict JSON (``spectral_onsite`` v4.1 layout).

    Args:
        results: From :func:`analyse_cache` or :func:`run_overlap_analysis`.
        path: Output file.

    Returns:
        The path written.
    """
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(_clean(results), fh, indent=1)
    return path


# =============================================================== report
def _f(x, fmt="+.3f"):
    return "-" if x is None else format(x, fmt)


def _md_cell(text):
    """Table-cell text: shell pairs 'Ti 3d | O 2p' keep their pipe as an escaped one."""
    return str(text).replace("|", "\\|")


def _mix_cell(Ld):
    """Ledger cell: filled-filled (occupied parent) or empty-empty (empty parent) mixing."""
    if "filled_filled_mixing" in Ld:
        return f"{Ld['filled_filled_mixing']:+.3f}"
    if "empty_empty_mixing" in Ld:
        return f"({Ld['empty_empty_mixing']:+.3f})"
    return "-"


def _edge_levels(p3):
    """Indices of the highest occupied and lowest empty frozen level of the model space."""
    lv = p3["pictures"]["ionic"]["levels"]
    occ = [i for i, L in enumerate(lv) if not L["effective"] and L["occupation"] > 1e-3]
    emp = [i for i, L in enumerate(lv) if not L["effective"] and L["occupation"] <= 1e-3]
    v = max(occ, key=lambda i: lv[i]["e_rel_vbm"]) if occ else None
    c = min(emp, key=lambda i: lv[i]["e_rel_vbm"]) if emp else None
    return v, c


def _inter_pairs(p3, pairs):
    """Inter-sublattice shell pairs ('cation shell | anion shell') of a cohp_pairs dict."""
    col: dict = {}
    for r in p3.get("accepted", []):
        col.setdefault(r.get("shell"), r.get("column"))
    out = {}
    for k, v in pairs.items():
        a, _, b = k.partition(" | ")
        if col.get(a) is not None and col.get(b) is not None and col[a] != col[b]:
            out[k] = v
    return out


def _level_summary(p3, i, ledger_by_parent, top=3):
    """One Key-results row: (A) Loewdin populations and COHP, (B) parent and ledger."""
    ion = p3["pictures"]["ionic"]["levels"][i]
    low = p3["pictures"]["lowdin"]["levels"][i]
    pops = ", ".join(f"{k} {v:.2f}" for k, v in list(low["pop_shell"].items())[:top] if v >= 0.01)
    prs = ", ".join(f"{_md_cell(k)} {v:+.2f}" for k, v in
                    list(_inter_pairs(p3, low["cohp_pairs"]).items())[:2] if abs(v) >= 0.01)
    ps = ion.get("parent_shift")
    if ps:
        Ld = ledger_by_parent.get(ps["parent"])
        mix = ps.get("filled_filled_mixing", ps.get("empty_empty_mixing", 0.0))
        led = (f"{ps['parent_label']} {ps['parent_e']:+.2f} ({ps['parent_share']:.2f})" +
               (f"; d_f {Ld['d_model']:+.2f}, Pauli {Ld['pauli_shift']:+.2f}" if Ld else "") +
               f"; {'filled-filled' if ps['class'] == 'filled' else 'empty-empty'} "
               f"{mix:+.2f}; covalent {ps['covalent_shift']:+.2f}")
    else:
        led = "-"
    return (f"{ion['label'] or ion['id']} | {ion['e_rel_vbm']:+.3f} | {pops} | "
            f"{low['cohp_inter']:+.2f} ({prs or '-'}) | {low['bond']} | {led} | "
            f"{ion['cohp_inter_split'].get('occ-emp', 0.0):+.2f} |")


def _key_results(res):
    """Key-results block: band edges of every k point and the covalency count."""
    out = ["\n## Key results (auto-generated)\n",
           "Band edges of every analysed k point (highest occupied / lowest empty level of "
           "the model space).  (A) symmetric Loewdin: shell populations, inter-sublattice COHP "
           "per state with the two main inter-sublattice shell pairs, bond verdict; (B) "
           "frozen-ion: main parent "
           "(share) with d_f, Pauli, filled-filled (or empty-empty) and covalent shifts, and "
           "the occupied-empty (donor-acceptor) part of the ionic COHP.  eV vs the VBM.\n",
           "| k | edge | level | E | populations (A) | COHP (A) total (main pairs) | verdict (A) "
           "| parent (B): E (share); ledger | donor-acceptor COHP (B) |",
           "|---|---|---|---|---|---|---|---|---|"]
    cov = []
    for kp in res["kpoints"]:
        p3 = kp.get("phase3")
        if not p3 or not p3.get("n_active"):
            continue
        led_by_frag = {Ld["id"]: Ld for Ld in p3["ledger"]}
        lbp = {R["id"]: led_by_frag.get(R["composition"][0][0])
               for levs in p3["pictures"]["ionic"]["parents"].values() for R in levs
               if R["composition"]}
        v, c = _edge_levels(p3)
        for tag, i in (("VBM(k)", v), ("CBM(k)", c)):
            if i is not None:
                out.append(f"| {kp['name']} | {tag} | " + _level_summary(p3, i, lbp))
        ion = p3["pictures"]["ionic"]
        main = ", ".join(f"{k} {x:.2f}" for k, x in
                         list(ion.get("covalency_electrons", {}).items())[:2] if x >= 0.005)
        cov.append(f"{kp['name']} {ion.get('covalency_total', 0):.3f} e" +
                   (f" ({main})" if main else ""))
    out.append("\nCovalency count (B; electrons in formally empty shells per cell): " +
               "; ".join(cov) + ".\n")
    return out


def _window_text(res, zero="VBM", fmt=""):
    """``"VBM + 14.0 eV"``, or ``"the top of the computed bands"`` without a window."""
    emax = res["method"]["emax_rel_vbm"]
    return "the top of the computed bands" if emax is None else f"{zero} + {emax:{fmt}} eV"


def format_report(res: dict, compound: str | None = None, top_pairs: int = 4,
                  top_pops: int = 4, key_parents=None) -> str:
    """Markdown-compatible plain-text report of an analysis.

    Settings and checks, the Key-results block, the sensitivity of the ionic
    parents, then per k point the effective outer levels, the ledger of every
    active fragment level, the parents of the three pictures and the crystal
    levels of the ionic and symmetric-Loewdin pictures (and, for SOC runs
    with a scalar reference, the scalar bridge).

    Args:
        res: The results dict.
        compound: Title (default: the compound stored in the results, else
            the formula).
        top_pairs: Shell pairs listed per crystal level.
        top_pops: Populations listed per crystal level.
        key_parents: Restrict the sensitivity table to parents whose key
            contains one of these strings.

    Returns:
        The report text.
    """
    st = res.get("phase2_settings", {})
    vbm = res["crystal"]["vbm_raw"]
    name = compound or (res.get("crystod") or {}).get("compound") or \
        res["crystal"].get("formula", "crystal")
    cr = res.get("crystod") or {}
    out = [f"# {name}: spectral on-site energies, frozen-ion parents and level-resolved COHP\n",
           f"Engine: {res['engine']}.  Crystal: `{res['crystal']['run']}` (NBANDS "
           f"{res['crystal']['nbands']}, VBM {vbm:.4f} eV raw).  Fragments: " +
           "; ".join(f"`{f['run']}` ({f.get('role')}, column {f['column']})"
                     for f in res["fragment_runs"]) + ".\n",
           f"Frozen window: crystal levels up to {_window_text(res)} "
           f"(whole multiplets), reproduced exactly; outer window: all {res['crystal']['nbands']}"
           f" bands (SMV projection-only disentanglement).  Cross-sphere: "
           f"{res['method']['cross_sphere']}.  Energies in eV relative to the crystal VBM "
           "unless marked raw.\n",
           f"Active-space rule ({cr.get('active_shell_rule', 'explicit')}): "
           f"{st.get('active_rule')}.\n",
           "Pictures: **ionic** (default for diagrams) = frozen-ion orthogonalisation: "
           "occupied fragment states of both runs Loewdin among themselves, empty states "
           "projected off the occupied span and then Loewdin; parents = eigenvalues of the "
           "sublattice blocks of H'' = C eps C^H.  **lowdin** = symmetric Loewdin of all active "
           "states; parents = eigenvalues of the blocks of H'.  **bare** = Ritz levels of the "
           "projected (non-orthogonal) states, Hp_XX v = e G_XX v, with Loewdin populations "
           "and COHP.  COHP_n(f,g) = 2 Re[conj(c_nf) H_fg c_ng], negative = bonding; the "
           "inter-sublattice COHP is split into occupied-occupied (closed shell), "
           "occupied-empty (donor-acceptor) and empty-empty parts.  Ledger: potential shift "
           "= d_f - E_raw (raw scales), Pauli shift = ionic parent - d_f (= orthogonalisation "
           "H''_ff - d_f + intra-sublattice block mixing parent - H''_ff); the mixing shift "
           "parent -> crystal level (the level carrying most of the fragment level) is split "
           "in two steps: filled-filled (closed-shell) mixing = level of H'' restricted to "
           "the occupied parents - parent (empty-empty mixing for an empty parent), then "
           "covalent (occupied-empty, donor-acceptor) mixing = crystal level - that level.  "
           "Paper convention: (A) symmetric Loewdin = populations, sublattice-orbital COHP "
           "and bond character; (B) frozen-ion = parent levels and ledger.\n",
           f"Irrep labels: {', '.join(st.get('irrep_runs', [])) or 'none'}.\n"]
    if (cr.get("cache") or {}).get("trimmed"):
        out.append("Overlap cache trimmed to the active fragment states: the per-level "
                   "diagnostics of the other fragment levels, the crystal populations and "
                   "the fragment-fragment block norms are incomplete or absent.\n")
    out += _key_results(res)
    # ------------------------------------------------------------- checks
    out.append("\n## Checks\n")
    out.append("| k | frozen bands | active | extra dims | P_nf Q P_nf kept (min) / next (max) | "
               "min eig G (model) | last-resort pruning | ionic: unitarity, eig(H'') vs model "
               "| max \\|sum rule\\| (ionic) | nonbonding identities, ionic (irrep n_par/n_cry "
               "max\\|dev\\|) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for kp in res["kpoints"]:
        p3 = kp.get("phase3")
        if not p3 or not p3.get("n_active"):
            continue
        ms = p3["model_space"]
        ion = p3["pictures"]["ionic"]["checks"]
        nb = "; ".join(f"{c['irrep']} {c['n_parents']}/{c['n_crystal']} "
                       f"{_f(c['max_abs_dev'], '.0e')}" for c in ion["nonbonding_identities"])
        kept = (f"{min(ms['pqp_leading']):.3f} / {max(ms['pqp_next'] or [0]):.3f}"
                if ms["pqp_leading"] else "-")
        out.append(f"| {kp['name']} | {ms['n_frozen']} | {ms['n_active']} | {ms['n_extra']} | "
                   f"{kept} | {ms['G_min_eig']:.3f} | "
                   f"{', '.join(p3.get('pruned_last_resort', [])) or 'none'} | "
                   f"{_f(ion.get('unitarity_max_dev'), '.0e')}, "
                   f"{_f(ion.get('H_eigs_vs_model_levels_max_dev'), '.0e')} | "
                   f"{ion['max_abs_sumrule_residual']:.0e} | {nb} |")
    sc_lines = []
    for kp in res["kpoints"]:
        cnt = (kp.get("phase3") or {}).get("active_shell_counts")
        if cnt:
            short = [f"{s_} {a}/{b}" for s_, (a, b) in cnt.items() if a != b]
            sc_lines.append(f"{kp['name']}: " + (", ".join(short) if short else "complete"))
    if sc_lines:
        out.append("\nActive shells (states found / expected N_at (2l+1)" +
                   (" x 2" if res["crystal"].get("spinor") else "") + "; a missing state is a "
                   "fragment level rejected as a point-charge-site state): " +
                   "; ".join(sc_lines) + ".\n")
    lo = []
    for kp in res["kpoints"]:
        ms = (kp.get("phase3") or {}).get("model_space") or {}
        x = ms.get("window_levels_not_frozen")
        if x:
            lo.append(f"{kp['name']}: " + ", ".join(
                f"{y['label'] or y['id']} {y['e_rel_vbm']:+.2f} (Q {y['q']:.2f})" for y in x))
    if lo:
        out.append(f"Crystal levels below {_window_text(res, fmt='g')} left "
                   "out of the frozen set (not spanned by the active shells, Q_nn < "
                   "frozen_qmin; they enter only through the disentanglement): " +
                   "; ".join(lo) + ".\n")
    if res["crystal"].get("spinor"):
        out.append("\nSpinor runs (vasp_ncl): all-electron orthonormality of the crystal (T1) "
                   "and fragment (T2) spinors, and Kramers degeneracy (level groups of odd "
                   "dimension; largest splitting of a Kramers pair) up to "
                   f"{_window_text(res, 'VBM/HOMO')} and below the top 5 % of the bands; "
                   "odd groups above that window are unconverged top bands.\n")
        out.append("| k | T1 max dev | T2 max dev | odd groups in window (crystal, fragments) "
                   "| max Kramers splitting (eV) | odd groups above window |")
        out.append("|---|---|---|---|---|---|")
        for kp in res["kpoints"]:
            t = kp["tests"]
            kr = t.get("kramers", [])
            out.append(f"| {kp['name']} | {t['T1_crystal_max_dev']:.1e} | " +
                       ", ".join(f"{x['T2_max_dev']:.1e}" for x in t["T2_fragments"]) + " | " +
                       ", ".join(str(len(x["odd_groups_in_window"])) for x in kr) + " | " +
                       ", ".join(_f(x["max_pair_splitting_in_window_eV"], ".1e") for x in kr) +
                       " | " + ", ".join(str(len(x["odd_groups_above_window"])) for x in kr) +
                       " |")
    # ------------------------------------------------------------- sensitivity
    out.append("\n## Sensitivity of the ionic parents (eV; main = window "
               f"{res['method']['emax_rel_vbm']:g} eV with cross-sphere)\n"
               if res["method"]["emax_rel_vbm"] is not None else
               "\n## Sensitivity of the ionic parents (eV)\n")
    for kp in res["kpoints"]:
        p3 = kp.get("phase3")
        if not p3 or "sensitivity_ionic_parents" not in p3:
            continue
        main = {f"{R['column']}:{R['label']}": R["e_rel_vbm"]
                for levs in p3["pictures"]["ionic"]["parents"].values() for R in levs}
        tags = list(p3["sensitivity_ionic_parents"])
        out.append(f"\n{kp['name']}:\n")
        out.append("| parent | main | " + " | ".join(f"{t} (diff)" for t in tags) + " |")
        out.append("|---|---|" + "---|" * len(tags))
        for key, e0 in main.items():
            if key_parents and not any(kk in key for kk in key_parents):
                continue
            cells = []
            for t in tags:
                e1 = p3["sensitivity_ionic_parents"][t].get(key)
                cells.append("-" if e1 is None else f"{e1 - e0:+.3f}")
            out.append(f"| {key} | {e0:+.3f} | " + " | ".join(cells) + " |")
    # ------------------------------------------------------------- per k
    for kp in res["kpoints"]:
        p3 = kp.get("phase3")
        if not p3 or not p3.get("n_active"):
            continue
        out.append(f"\n## {kp['name']} (k = {kp['frac']}, k-point #{kp['index']})\n")
        ms = p3["model_space"]
        if ms.get("window_levels_not_frozen"):
            out.append("Window levels without fragment character (Q_nn < frozen_qmin; not "
                       "frozen, not drawn): " + "; ".join(
                           f"{x['label'] or x['id']} {x['e_rel_vbm']:+.2f} (q {x['q']:.2f})"
                           for x in ms["window_levels_not_frozen"]) + "\n")
        if ms["effective"]:
            out.append("Effective outer levels (disentangled from the bands above the frozen "
                       "window; pqp = <x|Q|x>, the fragment character of the level)\n")
            out.append("| id | e | deg | irrep (purity) | pqp | crystal bands (weight) |")
            out.append("|---|---|---|---|---|---|")
            for E in ms["effective"]:
                out.append(f"| {E['id']} | {E['e_rel_vbm']:+.3f} | {E['degeneracy']} | "
                           f"{E['irrep']} ({E['irrep_purity']:.2f}) | {E['pqp']:.3f} | " +
                           ", ".join(f"{b}: {w:.2f}" for b, w in E["bands"]) + " |")
        out.append("\nLedger of the active fragment levels (E_raw on the fragment's own scale; "
                   "all other energies vs the crystal VBM)\n")
        out.append("| col | level | occ | E_raw | d_f (window) | d_f (model) | Ritz | "
                   "Loewdin parent | H''_ff | ionic parent | carried by (crystal level: share) "
                   "| potential | Pauli (orthog. + intra) | filled-filled (empty-empty) mixing "
                   "| covalent (occ-emp) |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for Ld in p3["ledger"]:
            carry = "; ".join(f"{lab or i} {e:+.2f}: {w:.2f}" for i, lab, e, w in Ld["carried_by"])
            out.append(f"| {Ld['column']} | {Ld['label']} | {'occ' if Ld['occupied'] else 'emp'} | "
                       f"{Ld['E_raw']:.3f} | {_f(Ld['d_window'])} | {_f(Ld['d_model'])} | "
                       f"{_f(Ld['ritz'])} | {_f(Ld['lowdin_parent'])} | {_f(Ld['ionic_onsite'])} | "
                       f"{_f(Ld['ionic_parent'])} | {carry} | {Ld['potential_shift']:+.3f} | "
                       f"{Ld['pauli_shift']:+.3f} ({Ld['pauli_orthogonalisation']:+.2f} "
                       f"{Ld['intra_sublattice_mixing']:+.2f}) | {_mix_cell(Ld)} | "
                       f"{Ld['covalent_shift']:+.3f} |")
        flab = {L["id"]: L.get("label") for fr in kp["fragments"] for L in fr["levels"]}
        out.append("\nParents of the three pictures (composition over fragment levels)\n")
        out.append("| picture | col | parent | label | deg | e | composition |")
        out.append("|---|---|---|---|---|---|---|")
        for pic in ("ionic", "lowdin", "bare"):
            for col, levs in p3["pictures"][pic]["parents"].items():
                for R in levs:
                    comp = "; ".join(f"{flab.get(i) or i}: {v:.3f}"
                                     for i, v in R["composition"][:3])
                    out.append(f"| {pic} | {col} | {R['id']} | {R['label']} | {R['degeneracy']} | "
                               f"{R['e_rel_vbm']:+.3f} | {comp} |")
        ion = p3["pictures"]["ionic"]
        low = {L2["id"]: L2 for L2 in p3["pictures"]["lowdin"]["levels"]}
        par_lab = {R["id"]: R["label"] for levs in ion["parents"].values() for R in levs}
        out.append("\nCrystal levels of the model space, ionic picture (per crystal partner; "
                   "COHP inter split as occupied-occupied / occupied-empty / empty-empty); "
                   "Loewdin and bare verdicts alongside\n")
        out.append("| level | label | deg | occ | E | compl. | populations (ionic parents) | "
                   "COHP inter | oo / oe / ee | main shell-pair COHP | intra | ionic | lowdin |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for C in ion["levels"]:
            pops = "; ".join(f"{par_lab.get(i, i)}: {v:.3f}"
                             for i, v in list(C["pop_parent"].items())[:top_pops] if v >= 0.01)
            sp = C["cohp_inter_split"]
            prs = "; ".join(f"{_md_cell(k)}: {v:+.3f}"
                            for k, v in list(C["cohp_pairs"].items())[:top_pairs])
            lv = low.get(C["id"], {})
            eff = " (eff.)" if C["effective"] and "(eff)" not in str(C["label"]) else ""
            out.append(f"| {C['id']} | {C['label'] or '-'}{eff} | "
                       f"{C['degeneracy']} | {C['occupation']:.0f} | {C['e_rel_vbm']:+.3f} | "
                       f"{C['completeness']:.3f} | {pops} | {C['cohp_inter']:+.3f} | "
                       f"{sp.get('occ-occ', 0):+.2f} / {sp.get('occ-emp', 0):+.2f} / "
                       f"{sp.get('emp-emp', 0):+.2f} | {prs} | {C['cohp_intra']:+.3f} | "
                       f"{C['bond']}{'' if C['drawn'] else ' (not drawn)'} | "
                       f"{lv.get('bond', '-')} |")
        lowp = p3["pictures"]["lowdin"]
        out.append("\nCrystal levels, symmetric Loewdin picture (A): populations by shell, "
                   "sublattice-orbital COHP (per state) and bond verdict\n")
        out.append("| level | label | deg | occ | E | populations (shells) | COHP inter | "
                   "main shell-pair COHP | verdict |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for C in lowp["levels"]:
            pops = "; ".join(f"{k}: {v:.3f}" for k, v in list(C["pop_shell"].items())[:top_pops]
                             if v >= 0.01)
            prs = "; ".join(f"{_md_cell(k)}: {v:+.3f}"
                            for k, v in list(C["cohp_pairs"].items())[:top_pairs])
            out.append(f"| {C['id']} | {C['label'] or '-'} | {C['degeneracy']} | "
                       f"{C['occupation']:.0f} | {C['e_rel_vbm']:+.3f} | {pops} | "
                       f"{C['cohp_inter']:+.3f} | {prs} | {C['bond']} |")
        cov = ion.get("covalency_electrons", {})
        out.append(f"\nCovalency (ionic picture): {ion.get('covalency_total', 0):.3f} electrons in "
                   "formally empty fragment shells per cell at this k: " +
                   ", ".join(f"{k} {v:.3f}" for k, v in cov.items() if v >= 0.005) + ".\n")
        if p3.get("active_pairs"):
            out.append("Active inter-sublattice level pairs: direct PAW overlap S and the "
                       "coupling in the three pictures (block norms / sqrt(deg))\n")
            out.append("| pair | S | H bare | H lowdin | H ionic |")
            out.append("|---|---|---|---|---|")
            for P in p3["active_pairs"][:20]:
                out.append(f"| {P['labels'][0]} / {P['labels'][1]} | {P['S_exact']:.3f} | "
                           f"{P['H_bare']:.3f} | {P['H_lowdin']:.3f} | {P['H_ionic']:.3f} |")
        br = p3.get("scalar_bridge")
        if br:
            dvb = res.get("vbm_soc_minus_scalar_eV")
            out.append("\nScalar-relativistic bridge (ionic picture): every SOC crystal level with "
                       "its ionic parents, COHP and the scalar levels of the compatible "
                       "single-group irreps (R1+ -> -R6, R3+ -> -R10, R4+ -> -R6 + -R10, "
                       "R5+ -> -R7 + -R10, R4- -> -R8 + -R11), ranked by the parent-shell "
                       "population they share with the SOC level (share >= 0.25 kept). "
                       "Shift = E(SOC) - E(first scalar level), both relative to their own "
                       "VBM" + (f"; VBM(SOC) - VBM(scalar) = {dvb:+.3f} eV on the absolute "
                                "eigenvalue scale" if dvb is not None else "") + ".\n")
            out.append("| SOC level | irrep | deg | occ | E | ionic parents (pop.) | COHP inter "
                       "| main shell-pair COHP | scalar levels (E, bond, shared) | shift |")
            out.append("|---|---|---|---|---|---|---|---|---|---|")
            for B in br["crystal"]:
                pars = "; ".join(f"{lab}: {v:.3f}" for lab, v in B["parents"] if v >= 0.01)
                prs = "; ".join(f"{_md_cell(k)}: {v:+.3f}"
                                for k, v in list(B["cohp_pairs"].items())[:3])
                sc = "; ".join(f"{lab} {e:+.3f} ({b}, {s:.2f})" for lab, e, b, s in
                               B["scalar_levels"])
                sh = (f"{B['shift_rel_vbm_eV']:+.3f}" if "shift_rel_vbm_eV" in B else "")
                out.append(f"| {B['label']} | {B['irrep']} | {B['degeneracy']} | "
                           f"{B['occupation']:.0f} | {B['e_rel_vbm']:+.3f} | {pars} | "
                           f"{B['cohp_inter']:+.3f} | {prs} | {sc} | {sh} |")
            out.append("\n| SOC parent | E | scalar parents of the same shell (E) |")
            out.append("|---|---|---|")
            for P in br["parents"]:
                sc = "; ".join(f"{lab} {e:+.3f}" for lab, e, b in P["scalar_parents"])
                out.append(f"| {P['label']} | {P['e_rel_vbm']:+.3f} | {sc} |")
        rej = [r for r in p3["rejected"] if (r.get("W_all") or 0) >= 0.3]
        if rej:
            out.append("\nRejected fragment levels with W(all) >= 0.3: " +
                       "; ".join(f"{r['label'] or r['id']} ({r['reason']})" for r in rej) + ".\n")
    return "\n".join(out) + "\n"


_LEDGER_KEYS = ("id", "column", "label", "occupied", "degeneracy", "E_raw", "d_window",
                "d_model", "ritz", "lowdin_parent", "ionic_parent", "ionic_onsite",
                "potential_shift", "pauli_shift", "pauli_orthogonalisation",
                "intra_sublattice_mixing", "mixing_shift", "parent_class", "block_level",
                "filled_filled_mixing", "empty_empty_mixing", "covalent_shift", "carried_by")


def key_numbers(results: dict, kpoints=None, pair_floor: float = 0.05) -> dict:
    """Compact digest of the key results (regression fixtures, summaries).

    Per k point: the crystal levels of the frozen window with their irreps,
    the ionic (frozen-ion) parents, the band edges with their (A) Loewdin
    populations, inter-sublattice COHP, shell-pair COHPs and verdict and their
    (B) parent and ledger, every model level with its COHP, main shell pairs
    (``|COHP| >= pair_floor``) and the verdicts of both pictures, the ledger
    rows and the covalency count.

    Args:
        results: From :func:`analyse_cache` (or the JSON read back).
        kpoints: Names of the k points to include (default: all).
        pair_floor: Smallest |COHP| of a listed shell pair (eV).

    Returns:
        A JSON-serialisable dict.
    """
    out = {"compound": results["crystal"].get("formula"),
           "spinor": bool(results["crystal"].get("spinor", False)),
           "vbm_raw": results["crystal"]["vbm_raw"],
           "active_rule": (results.get("phase2_settings") or {}).get("active_rule"),
           "kpoints": {}}
    for kp in results["kpoints"]:
        if kpoints and kp["name"] not in kpoints:
            continue
        p3 = kp.get("phase3") or {}
        if not p3.get("n_active"):
            continue
        ion, low = p3["pictures"]["ionic"], p3["pictures"]["lowdin"]
        crystal = [[L["id"], L.get("label"), L.get("irrep"), L["degeneracy"],
                    L["energy_rel_vbm"], L["occupation"]]
                   for L in kp["crystal"]["levels"] if L["in_window"]]
        parents = {col: [[R["id"], R.get("label"), R.get("irrep"), R["degeneracy"],
                          R["e_rel_vbm"], R["electrons"]] for R in levs]
                   for col, levs in ion["parents"].items()}
        v, c = _edge_levels(p3)
        edges = {}
        for tag, i in (("VBM", v), ("CBM", c)):
            if i is None:
                continue
            Li, Ll = ion["levels"][i], low["levels"][i]
            edges[tag] = {"id": Ll["id"], "label": Ll.get("label"), "e_rel_vbm": Ll["e_rel_vbm"],
                          "lowdin_pop_shell": Ll["pop_shell"],
                          "lowdin_cohp_inter": Ll["cohp_inter"], "lowdin_bond": Ll["bond"],
                          "lowdin_cohp_pairs": _inter_pairs(p3, Ll["cohp_pairs"]),
                          "ionic_parent_shift": Li.get("parent_shift"),
                          "ionic_cohp_inter_split": Li["cohp_inter_split"]}
        levels = [[L["id"], L.get("label"), L["e_rel_vbm"], L["degeneracy"], L["cohp_inter"],
                   L["bond"], ion["levels"][j]["bond"],
                   {k: x for k, x in L["cohp_pairs"].items() if abs(x) >= pair_floor}]
                  for j, L in enumerate(low["levels"])]
        ledger = [{k: Ld[k] for k in _LEDGER_KEYS if k in Ld} for Ld in p3["ledger"]]
        ms = p3["model_space"]
        out["kpoints"][kp["name"]] = {
            "frac": kp["frac"], "n_active": p3["n_active"],
            "model_space": [ms["n_frozen"], ms["n_active"], ms["n_extra"]],
            "crystal_levels": crystal, "ionic_parents": parents, "band_edges": edges,
            "levels": levels, "ledger": ledger,
            "covalency_total": ion.get("covalency_total"),
            "covalency_electrons": ion.get("covalency_electrons")}
    return out


def write_report(results: dict, path: str, compound: str | None = None) -> str:
    """Write :func:`format_report` to ``path``.

    Returns:
        The path written.
    """
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(format_report(results, compound=compound))
    return path


# =============================================================== crystod --diagram --vasp
# ``--vasp-shells`` token: element, optional separator, principal quantum
# number and l letter (``Sr-4s``, ``Ti_3d``, ``O2p``)
_SHELL_TOKEN = re.compile(r"([A-Z][a-z]?)[-_:]?(\d+)([spdf])")

#: File names of IrRep's JSON inside an IrRep output directory (IrRep 2.x
#: writes the ``-json_file`` name as given, with or without the extension).
IRREP_JSON_NAMES = ("irrep-output.json", "irrep-output")


def parse_shell_tokens(tokens):
    """``--vasp-shells`` tokens -> the ``active_shells`` of :func:`analyse_cache`.

    Args:
        tokens: ``["Sr-4s", "Ti-3d", "O-2p"]`` (``-``, ``_`` or no separator;
            commas also split), or the single token ``auto`` (the default
            minimal valence rule) or ``auto-full`` (the automatic rule).

    Returns:
        ``["Sr 4s", "Ti 3d", "O 2p"]``, ``None`` for ``auto`` (and for no
        tokens), or ``"auto-full"``.

    Raises:
        SystemExit: A malformed token, an impossible shell (``O-1p``), a
            shell given twice, or ``auto`` / ``auto-full`` next to shells.

    Example:
        >>> parse_shell_tokens(["Sr-4s", "Ti_3d", "O2p"])
        ['Sr 4s', 'Ti 3d', 'O 2p']
        >>> parse_shell_tokens(["auto"]) is None
        True
    """
    words = [word.strip() for token in tokens or [] for word in str(token).split(",")
             if word.strip()]
    if not words:
        return None
    for keyword in ("auto", "auto-full"):
        if keyword in [word.lower() for word in words]:
            if len(words) != 1:
                raise SystemExit(f"ERROR: --vasp-shells {keyword} stands alone; it cannot be "
                                 "combined with a list of shells.")
            return None if keyword == "auto" else "auto-full"
    shells = []
    for word in words:
        match = _SHELL_TOKEN.fullmatch(word)
        if not match or int(match.group(2)) <= _LNAME.index(match.group(3)):
            raise SystemExit(
                "ERROR: --vasp-shells takes element-shell tokens such as Sr-4s Ti-3d O-2p "
                f"(or auto / auto-full), not {word!r}.")
        element, n, letter = match.groups()
        name = _shell_name(element, int(n), _LNAME.index(letter))
        if name in shells:
            raise SystemExit(f"ERROR: --vasp-shells lists {element}-{int(n)}{letter} twice.")
        shells.append(name)
    return shells


def _run_name(directory: str) -> str:
    """Name of a run directory (``BAND``; ``DIR/band`` is named after DIR)."""
    path = os.path.normpath(os.path.abspath(directory))
    if os.path.basename(path) == "band":
        path = os.path.dirname(path)
    return os.path.basename(path)


def irrep_json_for_runs(pattern: str, run_dirs) -> list:
    """IrRep's JSON of every run from ``--vasp-irrep-json PATTERN``.

    Args:
        pattern: A path with a ``{run}`` placeholder for the name of the run
            directory (``.../irrep_{run}/irrep-output.json``), or a directory
            that holds ``<run name>/irrep-output.json`` (or ``irrep-output``).
            A placeholder path that names a directory is searched for those
            two file names as well.
        run_dirs: The run directories (crystal first).

    Returns:
        One path per run, ``None`` for a run without a JSON.
    """
    out = []
    for directory in run_dirs:
        name = _run_name(directory)
        candidate = (pattern.replace("{run}", name) if "{run}" in pattern
                     else os.path.join(pattern, name))
        found = None
        if os.path.isfile(candidate):
            found = candidate
        elif os.path.isdir(candidate):
            found = next((os.path.join(candidate, file_name) for file_name in IRREP_JSON_NAMES
                          if os.path.isfile(os.path.join(candidate, file_name))), None)
        out.append(found)
    return out


def _species_numbers(symbols) -> list:
    """Species ids in order of appearance (a ``Va`` point charge is a species)."""
    ids: dict = {}
    return [ids.setdefault(symbol, len(ids) + 1) for symbol in symbols]


def wavecar_label_provider(run_dirs, *, symprec: float = 1e-5, degeneracy_tol=None,
                           irrep_json=None):
    """The production label provider: irreps from the plane waves of the runs.

    For run ``i`` (0 = the crystal) at a k point it reads that k point's
    plane-wave coefficients from the run's WAVECAR and names the levels with
    :func:`crystod.wavecar_irreps.label_bands`: CrystOD's ISO-IR labels for
    scalar runs, the double-valued ``-K<n><p>`` names (``-R6+``) for spinor
    runs.  The sublattice runs are named with the symmetry of the CRYSTAL
    cell (``symmetry_cell``), so that the three columns share one frame.

    Args:
        run_dirs: The crystal run, then the sublattice runs.
        symprec: Symmetry tolerance of spglib.
        degeneracy_tol: Seed window of the level clustering (eV); ``None``
            is the module default (1 meV).
        irrep_json: Optional IrRep JSON per run (``None`` entries allowed):
            those runs get IrRep's (Bilbao) names instead, mapped by band
            index (:func:`crystod.wavecar_irreps.bcs_labels`).

    Returns:
        ``provider(run_index, k_name, k_frac) -> [label or None per band]``;
        ``provider.name_maps`` collects, per k-point name, the
        CrystOD-to-Bilbao correspondence of the double-valued names measured
        on the runs with an IrRep JSON (the scalar bridge names its
        compatibility lists with it).
    """
    from .vasp_wavecar import Wavecar, read_bands, read_run_structure
    from .wavecar_irreps import (
        bcs_labels,
        irrep_json_name_map,
        irreps_at_kpoint,
        labels_from_levels,
        read_irrep_json,
    )

    cells = []
    for directory in run_dirs:
        structure = read_run_structure(os.path.join(directory, "POSCAR"))
        cells.append((np.asarray(structure.lattice, float),
                      np.asarray(structure.positions, float),
                      _species_numbers(structure.symbols)))
    jsons = [read_irrep_json(path) if path else None
             for path in (irrep_json or [None] * len(run_dirs))]
    name_maps: dict = {}

    def provider(run_index, k_name, k_frac):
        path = os.path.join(run_dirs[run_index], "WAVECAR")
        with Wavecar(path) as wav:
            index, _ = find_kpoint(wav.kpoints(), [k_frac])
        if index is None:
            return None
        gvecs, coeffs, eigenvalues, _, kvec = read_bands(path, index)
        lattice, positions, numbers = cells[run_index]
        try:
            levels = irreps_at_kpoint(lattice, positions, numbers, kvec, gvecs, coeffs,
                                      eigenvalues, spinor=coeffs.shape[1] == 2,
                                      degeneracy_tol=degeneracy_tol, symprec=symprec,
                                      symmetry_cell=cells[0] if run_index else None)
        except ValueError as error:
            raise SystemExit(f"ERROR: irreps of {run_dirs[run_index]} at {k_name}: "
                             f"{error}.") from None
        data = jsons[run_index]
        if data is not None:
            names = bcs_labels(levels, data, kvec)
            if names is not None:
                mapping, _ = irrep_json_name_map(levels, data, kvec)
                known = name_maps.setdefault(k_name, {})
                known.update({key: value for key, value in mapping.items()
                              if key not in known})
                for level, name in zip(levels, names):
                    if name:
                        level["label"] = name
        return [label or None for label in labels_from_levels(levels, len(eigenvalues))]
    provider.name_maps = name_maps
    # what the labels are, so that a cache relabels when another kind is asked for
    provider.source = {"module": "crystod.wavecar_irreps",
                       "irrep_json": [os.path.abspath(path) if path else None
                                      for path in (irrep_json or [None] * len(run_dirs))]}
    return provider


def special_kpoints_of_run(cell, symprec: float = 1e-5) -> tuple:
    """The special points ``crystod --diagram`` draws, in the run cell's basis.

    The list is that of every ``--diagram`` engine
    (:meth:`crystod.crystal_orbital_diagram.CrystalOrbitalDiagram.special_kpoints`:
    the tabulated points of the space group in table order, with their ISO-IR
    names in the frame of the labels), converted from spglib's standardized
    primitive basis into the reciprocal basis of ``cell`` -- the basis of the
    WAVECAR k points and of the plane-wave irrep labels.

    Args:
        cell: The crystal run's primitive cell (``PhonopyAtoms``).
        symprec: Symmetry tolerance.

    Returns:
        ``([(name, k), ...], rotations)``: the points and the point-group
        rotations of ``cell`` (fractional, for the stars of k).
    """
    import spglib

    from .crystal_orbital_diagram import CrystalOrbitalDiagram
    from .runtime_compat import SymmetryDatasetAdapter, get_scaled_positions
    from .visualize_basis import SymmetryAdaptedOrbitalBasis

    class _Shell:
        """The one attribute ``special_kpoints`` reads."""

    shell = _Shell()
    shell.builder = SymmetryAdaptedOrbitalBasis(cell=cell, symprec=symprec)
    special = CrystalOrbitalDiagram.special_kpoints(shell)
    lattice = np.asarray(cell.cell, float)
    spg = (lattice, np.asarray(get_scaled_positions(cell), float),
           np.asarray(cell.numbers, int))
    # k_run = k_std M^T with L_run = M L_std (rows): the integer basis change
    # between spglib's standardized primitive cell (not idealized, so not
    # rotated) and the run's own primitive cell
    transform = np.eye(3)
    standard = spglib.standardize_cell(spg, to_primitive=True, no_idealize=True,
                                       symprec=symprec)
    if standard is not None:
        matrix = lattice @ np.linalg.inv(np.asarray(standard[0], float))
        if np.max(np.abs(matrix - np.rint(matrix))) < 1e-3:
            transform = np.rint(matrix)
    points = [(name, [round(float(x), 10) + 0.0 for x in np.asarray(k, float) @ transform.T])
              for name, k in special]
    dataset = SymmetryDatasetAdapter(spglib.get_symmetry_dataset(spg, symprec=symprec))
    return points, np.asarray(dataset.rotations)


def find_in_star(kpoints, k, rotations, tol: float = 1e-5) -> tuple:
    """The first k point of a list in the star of ``k``.

    ``k`` itself (modulo a reciprocal-lattice vector) is tried first, then
    the other arms ``k R`` of the point group -- so a run on a full mesh
    (spin-orbit runs with ``ISYM = -1``) finds X at (1/2, 0, 0) when it does
    not list (0, 1/2, 0).  ``-k`` is not tried: it need not be in the star.

    Args:
        kpoints: ``(nk, 3)`` fractional k points (the WAVECAR's).
        k: The k point.
        rotations: Point-group rotations (fractional, real space).
        tol: Coordinate tolerance.

    Returns:
        ``(index, k_found)`` (0-based index), or ``(None, None)``.
    """
    kpoints = np.asarray(kpoints, float)
    arms = [np.asarray(k, float)]
    for rotation in rotations:
        arm = arms[0] @ np.asarray(rotation, float)
        if not any(np.all(np.abs((arm - a) - np.rint(arm - a)) < tol) for a in arms):
            arms.append(arm)
    for arm in arms:
        difference = kpoints - arm
        hit = np.where(np.all(np.abs(difference - np.rint(difference)) < tol, axis=1))[0]
        if len(hit):
            return int(hit[0]), [float(x) for x in kpoints[hit[0]]]
    return None, None


def _check_primitive(cell, symprec: float, where: str) -> None:
    """The overlap engine reads the crystal in a primitive cell.

    Raises:
        SystemExit: ``cell`` holds more atoms than its primitive cell.
    """
    import spglib

    from .runtime_compat import get_scaled_positions

    spg = (np.asarray(cell.cell, float), np.asarray(get_scaled_positions(cell), float),
           np.asarray(cell.numbers, int))
    primitive = spglib.find_primitive(spg, symprec=symprec)
    if primitive is not None and len(primitive[2]) < len(spg[2]):
        raise SystemExit(
            f"ERROR: {where} is not a primitive cell ({len(spg[2])} atoms, the primitive "
            f"cell has {len(primitive[2])}); the overlap engine needs the three runs in the "
            "primitive cell (a supercell folds several k points onto one).")


def _same_setting(cell, other, tol: float = 1e-4) -> bool:
    """Same lattice matrix and the same atoms (species, fractional positions)."""
    from .runtime_compat import get_chemical_symbols, get_scaled_positions

    if not np.allclose(np.asarray(cell.cell, float), np.asarray(other.cell, float), atol=tol):
        return False
    first = list(zip(get_chemical_symbols(cell), np.asarray(get_scaled_positions(cell))))
    second = list(zip(get_chemical_symbols(other), np.asarray(get_scaled_positions(other))))
    if len(first) != len(second):
        return False
    for symbol, x in first:
        if not any(symbol == s and np.all(np.abs((x - y) - np.rint(x - y)) < tol)
                   for s, y in second):
            return False
    return True


def _cell_of_cache(cache):
    """The crystal cell recorded in a cache (``PhonopyAtoms``)."""
    from phonopy.structure.atoms import PhonopyAtoms

    run = cache.runs[0]
    return PhonopyAtoms(symbols=list(run["site_species"]),
                        cell=np.asarray(run["lattice"], float),
                        scaled_positions=np.asarray(run["positions"], float))


def _cell_of_run(path: str):
    """The crystal run's POSCAR as ``PhonopyAtoms``, read as stage 1 reads it.

    :func:`crystod.vasp_wavecar.read_run_structure` (VASP's reading: the
    scale factor applies to Cartesian positions too, and ``Sr_sv``-style
    species names are reduced to the element), so that the frame of the k
    points, of the irrep labels and of the page is the frame of the overlaps.
    """
    from phonopy.structure.atoms import PhonopyAtoms

    from .vasp_wavecar import read_run_structure

    structure = read_run_structure(path)
    return PhonopyAtoms(symbols=list(structure.symbols),
                        cell=np.asarray(structure.lattice, float),
                        scaled_positions=np.asarray(structure.positions, float))


def _fragment_columns(cell, left, right) -> dict:
    """``{element: "left" | "right"}`` from ``--co-left`` / ``--co-right``."""
    from .crystal_orbital_pyscf import PySCFCrystalOrbitalDiagram
    from .runtime_compat import get_chemical_symbols

    class _Probe:
        """The attributes ``_assign_fragments`` reads and writes."""

    probe = _Probe()
    probe.symbols = list(get_chemical_symbols(cell))
    PySCFCrystalOrbitalDiagram._assign_fragments(probe, left, right)
    return probe.element_column


def _cache_columns(cache, element_column) -> list:
    """Column (``left``/``right``) of every sublattice run of a cache."""
    columns = []
    for run in cache.runs[1:]:
        elements = {species for species, own in zip(run["site_species"], run["own"]) if own}
        found = {element_column.get(element) for element in elements} - {None}
        if len(found) != 1:
            raise SystemExit(
                f"ERROR: the real elements of the sublattice run {run['path']} "
                f"({', '.join(sorted(elements))}) do not form one of the two "
                "--co-left/--co-right sublattices.")
        columns.append(found.pop())
    if sorted(columns) != ["left", "right"]:
        raise SystemExit("ERROR: the sublattice runs of the overlap cache do not give one "
                         "--co-left and one --co-right column.")
    return columns


def _wavecar_directory(path: str) -> str:
    """The directory of a run whose WAVECAR the engine reads.

    ``DIR/band`` (where ``--vasp`` looks for a PROCAR) falls back to ``DIR``
    when only that holds the WAVECAR.

    Raises:
        SystemExit: No WAVECAR.
    """
    if os.path.isfile(os.path.join(path, "WAVECAR")):
        return path
    parent = os.path.dirname(os.path.normpath(os.path.abspath(path)))
    if (os.path.basename(os.path.normpath(path)) == "band"
            and os.path.isfile(os.path.join(parent, "WAVECAR"))):
        return parent
    raise SystemExit(
        f"ERROR: no WAVECAR in {path}: the overlap engine reads the wavefunctions of all "
        "three runs; repeat the run with LWAVE = .TRUE. (crystod --vasp-setup ... "
        "--vasp-engine overlap writes it).")


def _output_stem(cell, cell_path, run_dirs, symprec: float) -> tuple:
    """``(stem, named)`` of the output files, as the anchor engine names them.

    A ``-c`` file outside the run directories names the outputs after
    itself (``POSCAR``/``CONTCAR`` after their directory); otherwise the
    name is the formula and the space group of the crystal.
    """
    import spglib

    from .crystal_orbital_vasp import _cell_formula
    from .runtime_compat import SymmetryDatasetAdapter, get_chemical_symbols, \
        get_scaled_positions

    if cell_path is not None:
        inside = {os.path.abspath(path) for path in run_dirs}
        inside |= {os.path.dirname(path) for path in list(inside)
                   if os.path.basename(path) == "band"}
        if os.path.dirname(os.path.abspath(cell_path)) not in inside:
            stem = os.path.basename(cell_path)
            for extension in (".vasp", ".poscar"):
                if stem.lower().endswith(extension):
                    stem = stem[: -len(extension)]
            if stem.upper() in ("POSCAR", "CONTCAR"):
                stem = os.path.basename(os.path.dirname(os.path.abspath(cell_path)))
            return stem, True
    spg = (np.asarray(cell.cell, float), np.asarray(get_scaled_positions(cell), float),
           np.asarray(cell.numbers, int))
    dataset = SymmetryDatasetAdapter(spglib.get_symmetry_dataset(spg, symprec=symprec))
    formula = _cell_formula(get_chemical_symbols(cell))
    return f"{formula}_{dataset['international'].replace('/', '')}", False


def _key_results_lines(text: str) -> list:
    """The Key-results block of :func:`format_report`, line by line."""
    head = "## Key results (auto-generated)"
    if head not in text:
        return []
    block = text.split(head, 1)[1].split("\n## ", 1)[0]
    return [line for line in block.strip().splitlines()]


def run_vasp_diagram(left, right, *, vasp_paths=(), overrides=None, cell_path=None,
                     kpoint=None, output=None, symprec: float = 1e-5, shells=None,
                     frozen_window=None, frozen_qmin=None, cache_path=None,
                     scalar_reference=None, irrep_json=None, window=None,
                     cross_sphere: str = "bessel", label_provider=None,
                     report=print) -> dict:
    """``crystod --diagram --vasp --vasp-engine overlap``.

    Resolves the three runs exactly as the anchor engine does (``--vasp``
    with no path, one ROOT or the three run directories; ``--vasp-crystal``
    / ``--vasp-left`` / ``--vasp-right``), finds the special points of the
    space group in the crystal run's k list, computes the overlaps from the
    WAVECARs (stage 1, or reuses ``cache_path``), runs the analysis (stage 2)
    and writes ``CrystOD_<stem>_vasp_overlap.{html,json,txt}`` (``_soc``
    appended for spinor runs) into the current directory.  With a cache file
    that exists and no run directories (no ``--vasp`` path, no overrides, no
    ``./BAND``), everything comes from the cache alone.

    Args:
        left: ``--co-left`` tokens.
        right: ``--co-right`` tokens.
        vasp_paths: The paths given to ``--vasp`` (none, one ROOT, or three).
        overrides: ``{"mo": dir, "left": dir, "right": dir}`` of
            ``--vasp-crystal`` / ``--vasp-left`` / ``--vasp-right``.
        cell_path: ``-c`` file: names the outputs and must be the runs'
            crystal; the analysis is always done in the crystal run's own
            cell, the frame of the WAVECAR and of the labels.
        kpoint: One special-point name (``--kpoint``); ``None`` = all.
        output: HTML path (``--output``); the JSON and the report take its
            stem.
        symprec: Symmetry tolerance.
        shells: ``--vasp-shells`` tokens (:func:`parse_shell_tokens`).
        frozen_window: Frozen window above the VBM (eV; default 14).
        frozen_qmin: Projection threshold of the frozen levels (default 0.5).
        cache_path: ``--vasp-cache`` file (``.npz``).
        scalar_reference: Spinor runs: the scalar results JSON of the same
            compound (the scalar bridge).
        irrep_json: ``--vasp-irrep-json`` pattern (IrRep's Bilbao names).
        window: ``(low, high)`` eV vs the VBM: levels outside are left off the
            page (``None`` keeps every level).
        cross_sphere: Cross-sphere mode of the analysis (diagnostic).
        label_provider: Replaces the plane-wave irrep labels (tests).
        report: Print function.

    Returns:
        ``{"results", "html", "json", "txt"}``.

    Raises:
        SystemExit: Every user error, with one ERROR line.
    """
    from ._overlap_page import write_overlap_diagram_html
    from .crystal_orbital_vasp import _resolve_directories, _same_crystal, _sort_sublattices
    from .runtime_compat import get_chemical_symbols
    from .star_of_k import read_poscar_or_exit
    from .vasp_wavecar import Wavecar

    say = report or (lambda *a, **k: None)

    def indented(text="", *args, **kwargs):
        """The progress lines of the two stages, indented as the report's."""
        say(f"   {text}")

    overrides = {key: value for key, value in (overrides or {}).items() if value}
    active = parse_shell_tokens(shells)
    emax = 14.0 if frozen_window is None else float(frozen_window)
    qmin = 0.5 if frozen_qmin is None else float(frozen_qmin)
    if not emax > 0.0:
        raise SystemExit("ERROR: --vasp-frozen-window takes a positive energy in eV (the "
                         "frozen window is VBM + EV).")
    if not 0.0 < qmin <= 1.0:
        raise SystemExit("ERROR: --vasp-frozen-qmin takes a projection between 0 and 1.")
    if window is not None and window[0] is not None and window[1] is not None \
            and window[0] >= window[1]:
        raise SystemExit("ERROR: --vasp-window takes EMIN < EMAX.")
    cache_file = None
    if cache_path:
        cache_file = str(cache_path) if str(cache_path).endswith(".npz") \
            else f"{cache_path}.npz"
    paths = list(vasp_paths or [])
    no_runs = not paths and not overrides and not os.path.isdir("BAND")
    cache_only = cache_file is not None and os.path.isfile(cache_file) and no_runs
    if cache_file is not None and not os.path.isfile(cache_file) and no_runs:
        raise SystemExit(f"ERROR: no overlap cache at {cache_file}, and no run directory "
                         "to compute it from (give --vasp ROOT, or run in the ROOT that "
                         "holds BAND).")

    say("\n * CrystOD --diagram --vasp, overlap engine *")
    cache = None
    run_dirs: list = []
    if cache_only:
        say(f" no run directories: the overlap cache {cache_file} alone (no WAVECAR is read)")
        cache = OverlapCache.load(cache_file)
        cell = _cell_of_cache(cache)
        where = f"the crystal of {cache_file}"
    else:
        say("\n * VASP runs *")
        resolved = _resolve_directories(paths, overrides, say)
        where = os.path.join(resolved["mo"], "POSCAR")
        cell = _cell_of_run(where)
    _check_primitive(cell, symprec, where)
    element_column = _fragment_columns(cell, left, right)
    if not cache_only:
        directories = _sort_sublattices(resolved, element_column, say)
        run_dirs = [_wavecar_directory(directories[column])
                    for column in ("mo", "left", "right")]
    if cell_path is not None:
        given = read_poscar_or_exit(cell_path)
        if not _same_crystal(given, cell, symprec):
            raise SystemExit(f"ERROR: the -c cell {cell_path} is not the crystal of "
                             f"{where}; leave -c out, or give the crystal run's POSCAR.")
        if not _same_setting(given, cell):
            say(f"\n * Setting *\n   the -c cell and {where} are the same crystal in "
                "different settings: the overlap engine works in the crystal run's own "
                "cell (the frame of the WAVECAR), and every irrep label refers to it")
    stem, named = _output_stem(cell, cell_path,
                               run_dirs or [r["path"] for r in cache.runs], symprec)
    symbols = list(get_chemical_symbols(cell))
    formula = "".join(f"{s}{symbols.count(s) if symbols.count(s) > 1 else ''}"
                      for s in dict.fromkeys(symbols))

    # ---------------------------------------------------------- k points
    if cache_only:
        names = [kp.name for kp in cache.kpoints]
        if kpoint is not None:
            if kpoint not in names:
                raise SystemExit(f"ERROR: k point '{kpoint}' is not in the overlap cache "
                                 f"(available: {', '.join(names)}).")
            cache = OverlapCache(cache.meta, [kp for kp in cache.kpoints if kp.name == kpoint])
        spinor = cache.spinor
        say("\n * k points (from the cache) *")
        for kp in cache.kpoints:
            say(f"   {kp.name:<3} {_kfrac_text(kp.frac):<16} k#{kp.index + 1}")
        targets = None
    else:
        special, rotations = special_kpoints_of_run(cell, symprec)
        if kpoint is not None:
            available = [name for name, _ in special]
            special = [(name, k) for name, k in special if name == kpoint]
            if not special:
                raise SystemExit(
                    f"ERROR: k point '{kpoint}' is not a special point of this space "
                    f"group (available: {', '.join(available)}).")
        with Wavecar(os.path.join(run_dirs[0], "WAVECAR")) as wav:
            klist = wav.kpoints()
            spinor = wav.spinor
        targets, missing = [], []
        say("\n * k points (special points of the space group in the crystal run's k "
            "list) *")
        for name, k in special:
            index, found = find_in_star(klist, k, rotations)
            if index is None:
                missing.append(f"{name} {_kfrac_text(k)}")
                continue
            targets.append((name, [found]))
            say(f"   {name:<3} {_kfrac_text(found):<16} k#{index + 1}")
        if missing:
            say(f"   not in the k list (skipped): {', '.join(missing)} -- add them as "
                "zero-weight k points (crystod --vasp-setup writes them)")
        if not targets:
            raise SystemExit(f"ERROR: none of the special points of the space group is in "
                             f"the k list of {os.path.join(run_dirs[0], 'WAVECAR')}.")

    # ---------------------------------------------------------- spin-orbit
    reference = None
    if scalar_reference is not None:
        if not spinor:
            raise SystemExit("ERROR: --vasp-scalar-reference is for spin-orbit (vasp_ncl) "
                             "runs; these runs are scalar.")
        from ._overlap_page import load_results

        if not os.path.isfile(scalar_reference):
            raise SystemExit(f"ERROR: no scalar overlap results at {scalar_reference}.")
        reference = load_results(scalar_reference)
        crystal = reference.get("crystal") or {}
        if not reference.get("kpoints") or crystal.get("spinor"):
            raise SystemExit(f"ERROR: {scalar_reference} is not the JSON of a scalar "
                             "overlap-engine run (crystod --diagram --vasp --vasp-engine "
                             "overlap on the runs without spin-orbit coupling).")
    if spinor:
        say("\n * Spin-orbit coupling: spinor (vasp_ncl) runs, one electron per band, "
            "double-valued irreps *")
        if reference is None:
            say("   (no --vasp-scalar-reference: the scalar bridge is left out)")
        else:
            say(f"   scalar bridge from {scalar_reference}")

    # ---------------------------------------------------------- labels
    provider = label_provider
    if irrep_json is not None and cache_only:
        raise SystemExit("ERROR: --vasp-irrep-json needs the WAVECARs (the levels are "
                         "grouped from the plane waves); a cache keeps the labels it was "
                         "built with.")
    if provider is None and not cache_only:
        jsons = irrep_json_for_runs(irrep_json, run_dirs) if irrep_json else None
        provider = wavecar_label_provider(run_dirs, symprec=symprec, irrep_json=jsons)
        say("\n * Irrep labels *")
        say("   from the plane-wave coefficients of each run (crystod.wavecar_irreps): "
            + ("double-valued irreps of the spinor states, -K<n><p> names (-R6+ = "
               "R-bar 6+)" if spinor else "CrystOD's ISO-IR labels")
            + "; the sublattice runs in the crystal's frame")
        if jsons is not None:
            for directory, path in zip(run_dirs, jsons):
                say(f"   {_run_name(directory):<18} "
                    + (f"Bilbao names from {path}" if path else
                       "no IrRep JSON found: CrystOD names"))

    # ---------------------------------------------------------- stages 1 and 2
    compound = formula + (" (SOC)" if spinor else "")
    if cache_only:
        columns = _cache_columns(cache, element_column)
        say("\n * Analysis (stage 2) *")
        results = analyse_cache(cache, columns=columns, active_shells=active,
                                frozen_window=emax, frozen_qmin=qmin,
                                cross_sphere=cross_sphere, scalar_reference=reference,
                                compound=compound, report=indented, symprec=symprec)
    else:
        say("\n * All-electron overlaps (stage 1) and analysis (stage 2) *")
        results = run_overlap_analysis(run_dirs[0], run_dirs[1:], columns=["left", "right"],
                                       kpoints=targets, active_shells=active,
                                       frozen_window=emax, frozen_qmin=qmin,
                                       cross_sphere=cross_sphere, cache_path=cache_file,
                                       label_provider=provider, scalar_reference=reference,
                                       compound=compound, report=indented, symprec=symprec)
    if isinstance(active, list):
        found = {level.get("shell") for kp in results["kpoints"]
                 for fragment in kp.get("fragments") or [] for level in fragment["levels"]}
        unknown = [shell for shell in active if shell not in found]
        if unknown:
            raise SystemExit(
                "ERROR: --vasp-shells "
                + " ".join(shell.replace(" ", "-") for shell in unknown)
                + ": no sublattice level of that shell at any k point (shells found: "
                + ", ".join(sorted(s.replace(" ", "-") for s in found if s)) + ").")

    # ---------------------------------------------------------- outputs
    if output:
        html_path = output
        base = output[: -len(".html")] if output.endswith(".html") else output
    else:
        base = f"CrystOD_{stem}_vasp_overlap" + ("_soc" if spinor else "")
        html_path = base + ".html"
    json_path, txt_path = base + ".json", base + ".txt"
    existed = os.path.isfile(html_path)
    write_json(results, json_path)
    text = format_report(results, compound=compound)
    with open(txt_path, "w") as handle:
        handle.write(text)
    page, entries = write_overlap_diagram_html(
        _clean(results), cell, left, right, html_path,
        structure_label=(stem + (" (SOC)" if spinor else "") if named else None),
        window=window, symprec=symprec)
    say("\n * Key results (eV vs the crystal VBM; the report has the full tables) *")
    for line in _key_results_lines(text):
        say(f"   {line}")
    from .crystal_orbital_diagram import dipole_rules_lines

    for name, kpoint, levels in entries:
        for line in dipole_rules_lines(page, name, kpoint, levels):
            say(line)
    say(f"\nCrystal-orbital diagram {'OVERWRITTEN' if existed else 'written'}: {html_path}")
    say(f"Results JSON written: {json_path}")
    say(f"Report written: {txt_path}")
    return {"results": results, "html": html_path, "json": json_path, "txt": txt_path}


def _kfrac_text(k) -> str:
    """``(0,1/2,0)``: a fractional k point with small denominators."""
    from fractions import Fraction

    return "(" + ",".join(str(Fraction(float(x)).limit_denominator(12)) for x in k) + ")"


# =============================================================== CLI
def main(argv=None) -> int:
    """``python -m crystod.crystal_orbital_overlap``: both stages from the command line.

    Args:
        argv: Argument list (default: ``sys.argv[1:]``).

    Returns:
        Exit status 0.
    """
    p = argparse.ArgumentParser(
        prog="python -m crystod.crystal_orbital_overlap",
        description="Crystal-orbital diagram data from the wavefunctions of a crystal run "
                    "and its sublattice (Va point-charge) runs: all-electron PAW overlaps, "
                    "frozen-ion parents, sublattice-orbital COHP.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crystal", help="crystal run directory (POSCAR, POTCAR, WAVECAR)")
    p.add_argument("--fragment", action="append", default=[],
                   help="sublattice run directory; give once per sublattice")
    p.add_argument("--columns", nargs="+", default=None,
                   help="column name of every sublattice run (default: left right)")
    p.add_argument("--kpoints", nargs="+", default=None,
                   help="NAME=k1,k2,k3 (fractional, 1/2 allowed) or 'all'")
    p.add_argument("--cache", help="overlap cache (.npz): reused when it covers the request, "
                                   "else built and written")
    p.add_argument("--from-cache", help="analyse this overlap cache only (no WAVECAR access)")
    p.add_argument("--potcar", help="POTCAR with the PAW data (default: the crystal's)")
    p.add_argument("--active-shells", nargs="+", default=None,
                   help='active shells ("Ti 3d" "O 2p" ...), or auto-full; default: POTCAR '
                        "valence + one standard empty shell per element")
    p.add_argument("--frozen-window", type=float, default=14.0,
                   help="frozen window above the VBM in eV (default 14; negative = all bands)")
    p.add_argument("--frozen-qmin", type=float, default=0.5,
                   help="window levels with less projection on the active span are left to "
                        "the disentanglement (default 0.5)")
    p.add_argument("--cross-sphere", choices=CROSS_SPHERE_MODES, default="bessel")
    p.add_argument("--scalar-reference", help="spinor runs: results JSON of the scalar analysis")
    p.add_argument("--json",
                   help="output JSON (default CrystOD_<crystal dir>_vasp_overlap.json)")
    p.add_argument("--report",
                   help="output report (default CrystOD_<crystal dir>_vasp_overlap.txt)")
    p.add_argument("--compound", help="compound name for the report title")
    p.add_argument("--degen-tol", type=float, default=1e-3)
    p.add_argument("--cohp-tol", type=float, default=0.05)
    p.add_argument("--reg-tol", type=float, default=1e-3)
    p.add_argument("--sensitivity-windows", type=float, nargs="*", default=[10, 18])
    p.add_argument("--shell-n", nargs="+", default=[],
                   help="lowest principal quantum number overrides, e.g. Sr:s=4")
    args = p.parse_args(argv)
    active = args.active_shells
    if active and len(active) == 1 and active[0] == "auto-full":
        active = "auto-full"
    options = dict(degen_tol=args.degen_tol, cohp_tol=args.cohp_tol, reg_tol=args.reg_tol,
                   sensitivity_windows=args.sensitivity_windows, shell_n=args.shell_n)
    window = None if args.frozen_window < 0 else args.frozen_window
    if args.from_cache:
        res = analyse_cache(args.from_cache, columns=args.columns, active_shells=active,
                            frozen_window=window, frozen_qmin=args.frozen_qmin,
                            cross_sphere=args.cross_sphere,
                            scalar_reference=args.scalar_reference, compound=args.compound,
                            **options)
        stem = os.path.splitext(os.path.basename(args.from_cache))[0]
    else:
        if not args.crystal or not args.fragment or not args.kpoints:
            p.error("--crystal, --fragment (one per sublattice) and --kpoints are required "
                    "unless --from-cache is given")
        kpoints = parse_kpoint_tokens(args.kpoints)
        columns = args.columns or (["left", "right"] if len(args.fragment) == 2 else
                                   [f"frag{i + 1}" for i in range(len(args.fragment))])
        res = run_overlap_analysis(args.crystal, args.fragment, columns=columns,
                                   kpoints=kpoints, active_shells=active,
                                   frozen_window=window, frozen_qmin=args.frozen_qmin,
                                   cross_sphere=args.cross_sphere, cache_path=args.cache,
                                   label_provider=wavecar_label_provider(
                                       [args.crystal, *args.fragment]),
                                   scalar_reference=args.scalar_reference,
                                   potcar=args.potcar, compound=args.compound, **options)
        stem = os.path.basename(os.path.abspath(args.crystal).rstrip("/"))
    json_path = args.json or f"CrystOD_{stem}_vasp_overlap.json"
    report_path = args.report or f"CrystOD_{stem}_vasp_overlap.txt"
    write_json(res, json_path)
    write_report(res, report_path, compound=args.compound)
    print(f"wrote {json_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
