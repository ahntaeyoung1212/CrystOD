"""Interactive page of the WAVECAR-overlap crystal-orbital diagram.

The overlap engine of ``crystod --diagram --vasp`` (``--vasp-engine
overlap``, :mod:`crystod.crystal_orbital_overlap`) needs no energy
alignment: it evaluates the crystal Kohn-Sham Hamiltonian in the basis of
the sublattice orbitals -- the Bloch states of the two sublattice runs,
projected on the crystal Bloch states with all-electron PAW overlaps -- so
every energy it returns is on the crystal's own scale, relative to the
crystal VBM.  Its results hold, per special k point, the crystal levels of
the frozen window (VBM + 14 eV, reproduced exactly), the effective outer
levels of the disentangled model space, and three pictures of the
sublattice orbitals -- ``"ionic"`` (the frozen-ion reference: occupied
sublattice orbitals orthonormalized first, empty ones projected off them),
``"lowdin"`` (symmetric Loewdin) and ``"bare"`` (Ritz levels of the
projected orbitals) -- each with parent levels, populations, a
sublattice-orbital COHP resolved by shell pair and a bond verdict; the
frozen-ion picture adds the energy ledger of every sublattice orbital and
the covalency count.

This module draws those results as the interactive crystal-orbital-diagram
page every ``--diagram`` engine writes through
:func:`crystod.crystal_orbital_diagram.write_crystal_diagram_html`, in the
"hybrid" convention of the quantitative-COD paper:

* fragment columns: the parent levels of ``parent_picture`` (default the
  frozen-ion reference), labelled ``"Ti 3d GM5+"``;
* crystal column: the frozen-window levels plus the effective outer levels,
  drawn dotted (``"GM5+ #1"``, ``"GM1+ (eff)"``);
* level colour: sign of the inter-sublattice COHP of ``colour_picture``
  (default symmetric Loewdin; below the analysis tolerance, 0.05 eV, the
  level is nonbonding), in the bond colours of the other engines;
* connectors: the populations of ``connector_picture`` (default Loewdin)
  on the drawn parents -- the picture's own parent populations when it is
  the parent picture, otherwise its populations on the sublattice orbitals
  distributed over the parents by non-negative least squares per
  (sublattice, irrep) block (:func:`map_populations`; the results store no
  phases);
* tooltip and level panel: populations by shell, COHP by shell pair, the
  frozen-ion parent shift and donor-acceptor COHP of a crystal level, and
  for a parent the ledger bare d_f -> Pauli shift -> closed-shell
  (filled-filled) or empty-empty mixing -> covalent shift; the covalency
  count of every k point is a header chip.

Spinor (spin-orbit) results carry double-group irreps: CrystOD's ``-R6+``
(:mod:`crystod.wavecar_irreps`) is drawn R-bar 6+ and IrRep's ``-R8`` R-bar 8
(combining overline over the k letters), and a Kramers-degenerate set of
spinor states is ONE level of degeneracy 2 or 4 (one bar per Kramers pair,
two electrons each).

The results carry no orbital coefficients, so the page has no hover
wave-function sketch: every level's ``vectors`` is ``None``, which the
writer turns into a sketch-less level panel.
"""

from __future__ import annotations

import gzip
import json
import os
import re

import numpy as np

from .crystal_orbital_diagram import (
    CrystalOrbitalDiagram,
    DiagramLevel,
    SublatticeSpec,
    parse_fragment_formula,
    write_crystal_diagram_html,
)
from .runtime_compat import get_chemical_symbols, get_scaled_positions

PICTURES = ("ionic", "lowdin", "bare")

# the pictures in the plain-text tooltips and in the HTML chips/footer
PICTURE_NAMES = {"ionic": "frozen-ion", "lowdin": "symmetric Loewdin",
                 "bare": "bare (Ritz)"}
_PICTURE_HTML = {"ionic": "frozen-ion", "lowdin": "symmetric L&ouml;wdin",
                 "bare": "bare (Ritz)"}

# populations below this are not kept as connectors (the floor of the paper
# script); the writer itself draws a connector from 2 % up
POPULATION_FLOOR = 0.005
# shell populations / sublattice-orbital shares below this are left out of
# the level panel bars
SHARE_FLOOR = 0.01
# COHP shell pairs smaller than this (eV) are left out of the tooltip
PAIR_FLOOR = 0.005
# population of a crystal level on sublattice orbitals that no drawn parent
# holds, above which the tooltip says so
LOST_NOTE = 0.01
# level ids of the effective outer levels start with this prefix; the page
# style below draws them dotted
EFFECTIVE_PREFIX = "eff_"

_OVERLINE = "\u0305"   # combining overline
_DOUBLE_VALUED = re.compile(r"(?<![A-Za-z0-9])-([A-Z]+)(?=\d)")
_BAR_SUFFIX = re.compile(r"\b([A-Z]+)(\d+)_?bar\b")

_BOND_NAMES = {"b": "bonding", "bonding": "bonding",
               "n": "nonbonding", "nonbonding": "nonbonding",
               "a": "antibonding", "antibonding": "antibonding"}

# Extra page style: the effective outer levels are drawn dotted and lighter,
# as in the paper's figures, and the level-panel heading leaves Helvetica
# Neue, whose bold face cannot place a combining overline (R-bar 6 lost its
# bar there in Chrome; Helvetica and Arial bold keep it); the writer adds
# ``extra_css`` to the page style.
_PAGE_CSS = (
    '#gLvl g.lvl[data-id^="' + EFFECTIVE_PREFIX + '"] .seg'
    "{stroke-dasharray:2.4 2.6;stroke-opacity:.55}"
    "#panel h2{font-family:Helvetica,Arial,sans-serif}")


def overbar_label(text) -> str:
    r"""Double-valued irrep names drawn with an overbar.

    CrystOD (:mod:`crystod.wavecar_irreps`) and IrRep (Bilbao) name a
    double-valued irrep with a leading minus, ``-R6+`` and ``-R8`` (or
    ``R8bar``); the page draws R-bar 6+ and R-bar 8, with a combining
    overline on every letter of the k-point name and the index and parity
    after it.  Single-valued names (``GM4-``, ``O 2p GM4-#1``) come back
    unchanged.

    Args:
        text: A level label or an irrep name.

    Returns:
        The text with every double-valued irrep name overbarred.

    Example:
        >>> overbar_label("I 5s -R11/-R8") == "I 5s R\u030511/R\u03058"
        True
        >>> overbar_label("-R8-/-R6-") == "R\u03058-/R\u03056-"
        True
        >>> overbar_label("O 2p GM4-#1")
        'O 2p GM4-#1'
    """
    def bar(letters: str) -> str:
        return "".join(letter + _OVERLINE for letter in letters)

    text = _DOUBLE_VALUED.sub(lambda match: bar(match.group(1)), str(text))
    return _BAR_SUFFIX.sub(lambda match: bar(match.group(1)) + match.group(2),
                           text)


def _irrep_clean(text) -> str:
    """Irrep name without the ``?`` of an uncertain IrRep assignment."""
    return str(text or "").replace("?", "")


def load_results(path) -> dict:
    """Read an overlap-engine results file (``.json`` or ``.json.gz``).

    Args:
        path: The JSON file the engine wrote (the paper's
            ``*_onsite.json`` files have the same layout).

    Returns:
        The results dict.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        return json.load(handle)


def map_populations(populations: dict, parents: list) -> tuple[dict, float]:
    """Populations of one crystal level on sublattice orbitals -> on parents.

    The connector weights of the hybrid diagram: ``populations`` belong to
    one picture (e.g. symmetric Loewdin), ``parents`` to another (e.g. the
    frozen-ion reference).  Each parent carries its composition over the
    sublattice orbitals, ``share(P, f) = |U_fP|^2`` of the unitary rotation
    inside its block (same sublattice, same irrep), so that ``sum_P
    share(P, f) = 1``.  The results store no phases, so per block the
    non-negative weights ``w_P`` solving ``sum_P share(P, f) w_P = pop(f)``
    in the least-squares sense are taken and rescaled to the block's
    population.  For single-orbital parents this is the identity, and a
    crystal level equal to one parent (a symmetry-nonbonding level) gets
    weight 1 on it exactly, without the spurious cross weights of the
    direct ``sum_f pop(f) share(P, f)`` when parents mix sublattice orbitals
    of one sublattice (Ti 3d / Sr 4d, Ti 4s / Sr 5s).  The same mapping as
    ``_map_populations`` of the paper's ``plot_cod.py``.

    Args:
        populations: ``{sublattice-orbital id: population}`` of one crystal
            level in one picture (its ``pop_level``).
        parents: The parent records of the drawn picture, each with
            ``id``, ``column``, ``irrep`` and ``composition``
            (``[[sublattice-orbital id, share], ...]``).

    Returns:
        ``({parent id: population}, lost)``: the populations on the parents
        and the population on sublattice orbitals that no parent holds.
    """
    from scipy.optimize import nnls

    blocks: dict = {}
    for parent in parents:
        key = (parent.get("column"), _irrep_clean(parent.get("irrep")))
        blocks.setdefault(key, []).append(parent)
    weights: dict = {}
    covered: set = set()
    for members in blocks.values():
        orbitals = sorted({orbital for parent in members
                           for orbital, _ in (parent.get("composition") or [])})
        covered.update(orbitals)
        target = np.array([float(populations.get(orbital, 0.0))
                           for orbital in orbitals])
        total = float(target.sum())
        if not orbitals or total <= 0.0:
            continue
        matrix = np.zeros((len(orbitals), len(members)))
        for column, parent in enumerate(members):
            for orbital, share in parent.get("composition") or []:
                matrix[orbitals.index(orbital), column] += float(share)
        solution = nnls(matrix, target)[0]
        if solution.sum() <= 0.0:
            continue
        solution *= total / solution.sum()
        for parent, value in zip(members, solution, strict=True):
            if value > 0.0:
                weights[parent["id"]] = (weights.get(parent["id"], 0.0)
                                         + float(value))
    lost = sum(float(value) for orbital, value in populations.items()
               if orbital not in covered)
    return weights, lost


def _formula_elements(text) -> set:
    return set(re.findall(r"[A-Z][a-z]?", str(text or "")))


def _column_map(results: dict, left_tokens, right_tokens) -> dict:
    """``{results column: page column}`` from the sublattice formulas.

    The page columns follow ``--co-left`` / ``--co-right``, whatever column
    the engine filed a sublattice run under.
    """
    wanted = {
        "left": {element for element, _ in
                 parse_fragment_formula(left_tokens, "--co-left")},
        "right": {element for element, _ in
                  parse_fragment_formula(right_tokens, "--co-right")},
    }
    mapping: dict = {}
    found = []
    for run in results.get("fragment_runs") or []:
        elements = _formula_elements(run.get("formula"))
        if not elements:
            elements = {species for species in run.get("species") or []
                        if not str(species).startswith("Va")}
        found.append("".join(sorted(elements)))
        for column, members in wanted.items():
            if elements == members:
                mapping[run.get("column")] = column
    if sorted(mapping.values()) != ["left", "right"]:
        raise SystemExit(
            f"ERROR: --co-left {''.join(left_tokens)} / --co-right "
            f"{''.join(right_tokens)} do not match the sublattice runs of the "
            f"overlap results ({' and '.join(found) or 'none found'}).")
    return mapping


def _shell_key(shell: str) -> tuple:
    match = re.fullmatch(r"(\d+)([spdf])", shell)
    return ((int(match.group(1)), "spdf".index(match.group(2)))
            if match else (99, 9))


def _top(values: dict, count: int, floor: float) -> list:
    """The ``count`` largest-|value| entries of ``values`` above ``floor``."""
    ranked = sorted(values.items(), key=lambda item: -abs(float(item[1])))
    return [(key, float(value)) for key, value in ranked
            if abs(float(value)) >= floor][:count]


def _signed(value, digits: int = 2) -> str:
    """``+1.23`` / ``-0.45``, without the ``-0.00`` of a tiny negative."""
    return f"{round(float(value), digits) + 0.0:+.{digits}f}"


class OverlapDiagramPage:
    """The overlap-engine results as a diagram object of the page writer.

    :func:`~crystod.crystal_orbital_diagram.write_crystal_diagram_html`
    reads a diagram object (formula, chips, footer texts, the hover-sketch
    supercell) and one ``{column: [DiagramLevel]}`` record per k point;
    this adapter builds both from the results.  Nothing is recomputed
    except the population mapping of :func:`map_populations`.

    Args:
        results: The overlap-engine results (the dict the engine returns
            and writes as JSON).
        cell: The crystal run's own structure (``PhonopyAtoms``), the frame
            of the k-point coordinates of ``results``.
        left_tokens: ``--co-left`` formula tokens, e.g. ``["SrTi"]``.
        right_tokens: ``--co-right`` formula tokens, e.g. ``["O3"]``.
        parent_picture: Picture whose parent levels fill the fragment
            columns: ``"ionic"`` (default), ``"lowdin"`` or ``"bare"``.
        colour_picture: Picture whose inter-sublattice COHP colours the
            crystal levels (default ``"lowdin"``).
        connector_picture: Picture whose populations give the connectors
            (default ``"lowdin"``).
        symprec: Symmetry tolerance of the space-group chip.

    Attributes:
        pictures: ``{"parent": ..., "colour": ..., "connector": ...}``.
        column_of: ``{results column: page column}``.
        spinor: The results are spinor (spin-orbit) results.
        side_specs: One :class:`SublatticeSpec` per active (element, shell)
            of each fragment column (no exponents or on-site energies).
        extra_chips: Header chips set by :meth:`build_entries` (window,
            pictures, covalency count per k point, energy zero).

    Raises:
        SystemExit: Unknown picture names, tokens that do not match the
            sublattice runs, or a cell that is not the results' crystal.
    """

    # the writer's hover-sketch geometry: the k-commensurate supercell of the
    # crystal cell, built exactly as the other engines build it (the
    # geometry is embedded but unused while no level carries coefficients)
    supercell_for = CrystalOrbitalDiagram.supercell_for

    def __init__(self, results: dict, cell, left_tokens, right_tokens, *,
                 parent_picture: str = "ionic", colour_picture: str = "lowdin",
                 connector_picture: str = "lowdin", symprec: float = 1e-5):
        from .visualize_basis import SymmetryAdaptedOrbitalBasis

        for role, name in (("parent", parent_picture),
                           ("colour", colour_picture),
                           ("connector", connector_picture)):
            if name not in PICTURES:
                raise SystemExit(
                    f"ERROR: unknown {role} picture {name!r} (choose from "
                    f"{', '.join(PICTURES)}).")
        self.results = results
        self.pictures = {"parent": parent_picture, "colour": colour_picture,
                         "connector": connector_picture}
        # the run's own cell is the frame of the k points: no standardization
        self.builder = SymmetryAdaptedOrbitalBasis(cell=cell, symprec=symprec,
                                                   standardize=False)
        self.conventional = False
        self.symbols = list(get_chemical_symbols(cell))
        self.positions = np.array(get_scaled_positions(cell), dtype=float)
        self.lattice = np.array(cell.cell, dtype=float)
        self.formula = {"left": "".join(left_tokens),
                        "right": "".join(right_tokens)}
        self.column_of = _column_map(results, left_tokens, right_tokens)
        crystal = results.get("crystal") or {}
        expected = _formula_elements(crystal.get("formula"))
        if expected and expected != set(self.symbols):
            raise SystemExit(
                f"ERROR: the cell ({''.join(dict.fromkeys(self.symbols))}) is "
                f"not the crystal of the overlap results "
                f"({crystal.get('formula')}).")
        self.crystal_formula = (crystal.get("formula")
                                or "".join(dict.fromkeys(self.symbols)))
        self.spinor = bool(crystal.get("spinor"))
        self.electrons = float(crystal.get("nelect") or 0.0)
        settings = results.get("phase2_settings") or {}
        self.cohp_tol = float(settings.get("cohp_tol", 0.05))
        self.window_top = float(
            (results.get("method") or {}).get("emax_rel_vbm", 14.0))
        self.side_specs = self._active_specs(left_tokens, right_tokens)
        self.extra_chips: list = []
        self._page_texts(crystal)

    # ---------------------------------------------------------- page texts

    def _active_specs(self, left_tokens, right_tokens) -> dict:
        """Active (element, shell) blocks of each fragment column."""
        order = {column: [element for element, _ in
                          parse_fragment_formula(tokens, "--co-" + column)]
                 for column, tokens in (("left", left_tokens),
                                        ("right", right_tokens))}
        found: dict = {"left": set(), "right": set()}
        for record in self.results.get("kpoints") or []:
            for orbital in (record.get("phase3") or {}).get("accepted") or []:
                column = self.column_of.get(orbital.get("column"))
                parts = str(orbital.get("shell") or "").split()
                if column and len(parts) == 2 and re.fullmatch(r"\d+[spdf]",
                                                               parts[1]):
                    found[column].add((parts[0], parts[1]))
        specs: dict = {"left": [], "right": []}
        for column in ("left", "right"):
            ranked = sorted(found[column], key=lambda item: (
                order[column].index(item[0]) if item[0] in order[column]
                else 99, _shell_key(item[1])))
            for element, shell in ranked:
                n_shell, l_shell = _shell_key(shell)
                sites = [index for index, symbol in enumerate(self.symbols)
                         if symbol == element]
                specs[column].append(SublatticeSpec(
                    element, shell[-1], shell, n_shell, l_shell, None,
                    float("nan"), sites, column))
        return specs

    def _page_texts(self, crystal: dict) -> None:
        """Chips and footer texts the writer reads from the diagram object."""
        parts = []
        for column in ("left", "right"):
            by_element: dict = {}
            for spec in self.side_specs[column]:
                by_element.setdefault(spec.element, []).append(spec.shell)
            parts.append(", ".join(f"{element} {' '.join(shells)}"
                                   for element, shells in by_element.items()))
        self.basis_chip = ("(sublattice orbitals: " + " | ".join(parts) + ")"
                           if any(parts) else "(sublattice orbitals)")
        charges = [str(species) for run in self.results.get("fragment_runs") or []
                   for species in run.get("species") or []
                   if str(species).startswith("Va")]
        self.embedding_chip = ("sublattice runs: Va point charges "
                               + " ".join(dict.fromkeys(charges))
                               if charges else "sublattice runs")
        functional = crystal.get("functional")
        encut = crystal.get("encut")
        self.method_chip = ("VASP PAW" + (f"/{functional}" if functional else "")
                            + (f", ENCUT {float(encut):g} eV" if encut else "")
                            + ": all-electron WAVECAR overlaps, no alignment")
        self.axis_title = "E - E_VBM (eV)"
        self.ao_foot = ""
        parent, colour, connector = (self.pictures["parent"],
                                     self.pictures["colour"],
                                     self.pictures["connector"])
        parent_text = {
            "ionic": "the parent levels of the frozen-ion reference "
                     "(occupied sublattice orbitals of both sublattices "
                     "orthonormalized among themselves, empty ones projected "
                     "off them; eigenvalues of each sublattice block of the "
                     "crystal Hamiltonian)",
            "lowdin": "the parent levels of the symmetric-L&ouml;wdin picture "
                      "(eigenvalues of each sublattice block of the crystal "
                      "Hamiltonian in the symmetrically orthonormalized "
                      "sublattice orbitals)",
            "bare": "the Ritz levels of the projected, non-orthogonal "
                    "sublattice orbitals of each sublattice (bare picture)",
        }[parent]
        if connector == parent:
            connector_text = (f"the {_PICTURE_HTML[connector]} populations of "
                              "each crystal level on the drawn parents")
        else:
            connector_text = (
                f"the {_PICTURE_HTML[connector]} populations of each crystal "
                f"level, distributed over the {_PICTURE_HTML[parent]} parents "
                "of the same sublattice and irrep by non-negative least "
                "squares on the parents' sublattice-orbital compositions")
        self.foot_intro = (
            "Crystal-orbital diagram (COD, --vasp, overlap engine): no energy "
            "alignment.  The crystal Kohn-Sham Hamiltonian is written in the "
            "basis of the sublattice orbitals -- the Bloch states of the two "
            "sublattice runs (removed sublattice = Va point charges), "
            "projected on the crystal Bloch states with all-electron PAW "
            "overlaps -- so the three columns share the crystal's energy "
            "scale, E &minus; E<sub>VBM</sub>.  Centre: the DFT levels up to "
            f"E<sub>VBM</sub> + {self.window_top:g} eV (frozen window, "
            "reproduced exactly) and, dotted, the effective outer levels of "
            "the disentangled model space (projected from the bands above "
            f"the window).  Sides: {parent_text}, labelled by shell and "
            f"irrep.  Connectors: {connector_text} (drawn from 2%).  Click a "
            "level for its populations and its COHP by shell pair, a parent "
            "for its energy ledger: bare d<sub>f</sub> = &lang;&phi;|H|&phi;"
            "&rang; &rarr; Pauli shift (orthogonalization to the occupied "
            "orbitals of the other sublattice) &rarr; closed-shell "
            "(filled-filled) or empty-empty mixing &rarr; covalent shift "
            "(occupied-empty mixing) &rarr; crystal level.")
        self.bond_foot = (
            " Crystal-orbital line colors: "
            "<span style=\"color:#1565c0\">bonding</span> / "
            "<span style=\"color:#333\">nonbonding</span> / "
            "<span style=\"color:#d32f2f\">antibonding</span>, from the sign "
            f"of the inter-sublattice COHP of the {_PICTURE_HTML[colour]} "
            "picture, COHP<sub>n</sub>(f,g) = 2 Re[c<sub>nf</sub><sup>*</sup> "
            "H<sub>fg</sub> c<sub>ng</sub>] summed over the pairs of "
            "sublattice orbitals on different sublattices (eV per cell and "
            "per state; negative = bonding; below "
            f"{self.cohp_tol:g} eV in magnitude nonbonding; value and "
            "shell-pair split in the level's tooltip)."
            + (" Spin-orbit coupling: one bar per Kramers pair (two spinor "
               "states), two electrons each; double-valued irreps carry an "
               "overbar (-R6+ is R&#773;6+, Koster-style numbering; "
               "IrRep's Bilbao -R8 is R&#773;8)."
               if self.spinor else ""))
        # the writer's sketch sentence, the page style and one note per k
        self.sketch_foot = (" No wave-function sketch: the overlap engine "
                            "stores no orbital coefficients.")
        self.extra_css = _PAGE_CSS
        self.k_notes: dict = {}

    def sketch_partners(self, level, kpoint, sites):
        """Hover-sketch amplitudes: none, the results carry no coefficients.

        Args:
            level: A :class:`DiagramLevel` of this page.
            kpoint: Its k point.
            sites: The supercell sites of :meth:`supercell_for`.

        Returns:
            ``None`` (the writer never asks: every level's ``vectors`` is
            ``None``).
        """
        return None

    # ------------------------------------------------------------- levels

    def levels_at(self, record: dict) -> dict:
        """The three diagram columns of one k point.

        Args:
            record: One entry of ``results["kpoints"]``.

        Returns:
            ``{"left": [...], "mo": [...], "right": [...]}`` of
            :class:`DiagramLevel`; crystal levels the analysis marked as not
            drawn (completeness below its threshold) are left out.

        Raises:
            SystemExit: A picture of the page is missing at this k point.
        """
        name = record.get("name", "?")
        phase3 = record.get("phase3") or {}
        pictures = phase3.get("pictures") or {}
        for picture in dict.fromkeys(self.pictures.values()):
            if picture not in pictures:
                raise SystemExit(
                    f"ERROR: the overlap results have no {picture!r} picture "
                    f"at k point {name} (found: "
                    f"{', '.join(pictures) or 'none'}).")
        by_id = {picture: {level["id"]: level for level in data.get("levels") or []}
                 for picture, data in pictures.items()}
        parent_picture = pictures[self.pictures["parent"]]
        parents = [parent for members in (parent_picture.get("parents") or {}).values()
                   for parent in members]
        orbital_labels = {level["id"]: level.get("label") or level["id"]
                          for fragment in record.get("fragments") or []
                          for level in fragment.get("levels") or []}
        shell_column = {}
        for orbital in phase3.get("accepted") or []:
            shell_column.setdefault(orbital.get("shell"), orbital.get("column"))
        for parent in parents:
            shell_column.setdefault(parent.get("shell"), parent.get("column"))
        ionic = pictures.get("ionic") or {}
        context = {
            "name": name,
            "by_id": by_id,
            "parents": parents,
            "parent_labels": {parent["id"]: overbar_label(parent.get("label")
                                                          or parent["id"])
                              for parent in parents},
            "orbital_labels": orbital_labels,
            "shell_column": shell_column,
            "ledger": {entry["id"]: entry for entry in phase3.get("ledger") or []},
            "effective": {entry["id"]: entry for entry in
                          (phase3.get("model_space") or {}).get("effective") or []},
            "covalency": ionic.get("covalency_electrons") or {},
            "covalency_total": ionic.get("covalency_total"),
        }
        levels: dict = {"left": [], "mo": [], "right": []}
        for parent in parents:
            column = self.column_of.get(parent.get("column"))
            if column is None:
                continue
            levels[column].append(self._parent_level(parent, column, context))
        for crystal in parent_picture.get("levels") or []:
            if crystal.get("drawn", True):
                levels["mo"].append(self._crystal_level(crystal, context))
        self._k_summary(levels["mo"], context)
        return levels

    def _parent_level(self, parent: dict, column: str, context: dict):
        """One fragment-column level: a parent of the parent picture."""
        level = DiagramLevel(
            level_id=str(parent["id"]), column=column,
            energy=float(parent["e_rel_vbm"]),
            degeneracy=int(parent["degeneracy"]),
            irrep=overbar_label(_irrep_clean(parent.get("irrep"))),
            label=context["parent_labels"][parent["id"]],
            electrons=int(round(float(parent.get("electrons") or 0.0))))
        self._kramers_bars(level)
        composition = sorted(((orbital, float(share)) for orbital, share
                              in parent.get("composition") or []),
                             key=lambda item: -item[1])
        level.display_composition = [
            (overbar_label(context["orbital_labels"].get(orbital, orbital)), share)
            for orbital, share in composition if share >= SHARE_FLOOR]
        picture = self.pictures["parent"]
        lines = [{
            "ionic": "frozen-ion parent: eigenvalue of the sublattice block "
                     "of the crystal Hamiltonian in the frozen-ion basis",
            "lowdin": "symmetric-Loewdin parent: eigenvalue of the sublattice "
                      "block of the crystal Hamiltonian in the Loewdin basis",
            "bare": "bare (Ritz) parent: Ritz level of the projected "
                    "sublattice orbitals of this sublattice",
        }[picture]]
        # a parent mixing several sublattice orbitals says so in the tooltip
        # (the panel shows the same shares as bars)
        if len(level.display_composition) > 1:
            lines.append("sublattice orbitals: " + ", ".join(
                f"{label} {100 * share:.1f}%"
                for label, share in level.display_composition))
        entry = context["ledger"].get(composition[0][0]) if composition else None
        if entry is not None:
            lines += self._ledger_lines(entry, level, context)
        shell = parent.get("shell")
        covalency = context["covalency"]
        if level.electrons == 0 and shell in covalency:
            total = context["covalency_total"]
            lines.append(
                f"covalency at {context['name']}: the formally empty {shell} "
                f"shell holds {float(covalency[shell]):.3f} e per cell (all its "
                "parents, frozen-ion picture)"
                + (f"; {float(total):.3f} e in all formally empty shells"
                   if total is not None else ""))
        level.detail = "\n".join(lines)
        return level

    def _ledger_lines(self, entry: dict, level, context: dict) -> list:
        """The frozen-ion energy ledger of a parent's main sublattice orbital."""
        orbital = overbar_label(entry.get("label") or entry["id"])
        parent = float(entry["ionic_parent"])
        filled = entry.get("parent_class", "filled") == "filled"
        mixing = float(entry.get("filled_filled_mixing",
                                 entry.get("empty_empty_mixing", 0.0)))
        block = float(entry.get("block_level", parent + mixing))
        split = ""
        if ("pauli_orthogonalisation" in entry
                and "intra_sublattice_mixing" in entry):
            split = (" (orthogonalization "
                     f"{_signed(entry['pauli_orthogonalisation'])}, "
                     "intra-sublattice mixing "
                     f"{_signed(entry['intra_sublattice_mixing'])})")
        lines = [f"frozen-ion ledger of the sublattice orbital {orbital} "
                 "(eV vs the VBM):",
                 f"bare d_f = <phi|H|phi>: {_signed(entry['d_model'])}",
                 f"Pauli shift {_signed(entry['pauli_shift'])}{split} -> "
                 f"frozen-ion parent {_signed(parent)}",
                 f"{'closed-shell (filled-filled)' if filled else 'empty-empty'}"
                 f" mixing {_signed(mixing)} -> {_signed(block)}"]
        carried = entry.get("carried_by") or []
        covalent = _signed(entry.get("covalent_shift", 0.0))
        if carried:
            _, label, energy, share = carried[0]
            lines.append(f"covalent shift {covalent} -> crystal level "
                         f"{overbar_label(label)} {_signed(energy)} (holds "
                         f"{100 * float(share):.0f}% of the orbital)")
        else:
            lines.append(f"covalent shift {covalent}")
        if self.pictures["parent"] == "ionic" and abs(parent - level.energy) > 1e-3:
            lines.append(f"(the orbital's frozen-ion parent lies at "
                         f"{_signed(parent)}: this level shares the orbital "
                         "with it)")
        return lines

    def _kramers_bars(self, level) -> None:
        """Spinor levels: one bar per Kramers pair, holding two electrons.

        Without ``bars`` the writer draws one bar per degenerate partner,
        i.e. per spinor state; the paper draws one per Kramers pair.
        """
        if self.spinor:
            level.bars = max(1, level.degeneracy // 2)

    def _crystal_level(self, crystal: dict, context: dict):
        """One crystal-column level: a frozen-window or effective level."""
        identifier = crystal["id"]
        colour = context["by_id"][self.pictures["colour"]].get(identifier, crystal)
        connector = context["by_id"][self.pictures["connector"]].get(identifier,
                                                                      crystal)
        effective = bool(crystal.get("effective"))
        label_text = str(crystal.get("label") or "")
        irrep = overbar_label(_irrep_clean(crystal.get("irrep")
                                           or (label_text.split() or [""])[0]))
        label = overbar_label(label_text) if label_text else (
            irrep + (" (eff)" if effective else ""))
        level = DiagramLevel(
            level_id=(EFFECTIVE_PREFIX if effective else "") + str(identifier),
            column="mo", energy=float(crystal["e_rel_vbm"]),
            degeneracy=int(crystal["degeneracy"]), irrep=irrep, label=label,
            electrons=int(round(float(crystal.get("occupation") or 0.0))))
        self._kramers_bars(level)
        level.effective = effective
        level.source_id = identifier
        # connectors: the connector picture's populations on the drawn parents
        if self.pictures["connector"] == self.pictures["parent"]:
            populations, lost = dict(connector.get("pop_parent") or {}), 0.0
        else:
            populations, lost = map_populations(connector.get("pop_level") or {},
                                                context["parents"])
        level.composition = sorted(
            ((parent, float(weight)) for parent, weight in populations.items()
             if float(weight) >= POPULATION_FLOOR), key=lambda item: -item[1])
        # panel bars: populations by shell, as shares of the level's span
        completeness = float(connector.get("completeness") or 0.0)
        shells = connector.get("pop_shell") or {}
        level.display_composition = sorted(
            ((shell, float(value) / completeness) for shell, value in shells.items()
             if completeness > 0 and float(value) / completeness >= SHARE_FLOOR),
            key=lambda item: -item[1])
        bond = _BOND_NAMES.get(str(colour.get("bond") or "").lower())
        if bond:
            level.bond_character = bond
        level.cohp_inter = colour.get("cohp_inter")

        lines = []
        labels = context["parent_labels"]
        made_of = [f"{labels.get(parent, parent)} {100 * weight:.1f}%"
                   for parent, weight in level.composition[:4] if weight >= 0.02]
        if made_of:
            # the panel lists the connectors itself; this line is for the
            # hover tooltip only (the writer drops "made of:" from the panel)
            lines.append("made of: " + "  |  ".join(made_of))
        if effective:
            info = context["effective"].get(identifier) or {}
            bands = ", ".join(f"{band} ({float(weight):.2f})"
                              for band, weight in (info.get("bands") or [])[:4])
            lines.append(
                "effective outer level (dotted): disentangled from the "
                "crystal bands above the frozen window (E_VBM + "
                f"{self.window_top:g} eV)"
                + (f"; sublattice character <x|Q|x> {float(info['pqp']):.3f}"
                   if "pqp" in info else "")
                + (f"; crystal bands {bands}" if bands else ""))
        shares = _top(shells, 5, SHARE_FLOOR)
        if shares:
            lines.append(
                f"populations ({PICTURE_NAMES[self.pictures['connector']]}): "
                + ", ".join(f"{shell} {value:.3f}" for shell, value in shares)
                + (f" (completeness {completeness:.3f})"
                   if completeness < 0.995 else ""))
        lines += self._cohp_lines(colour, bond, context)
        ionic = context["by_id"].get("ionic", {}).get(identifier)
        if ionic is not None:
            shift = ionic.get("parent_shift")
            if shift:
                mixing = float(shift.get("filled_filled_mixing",
                                         shift.get("empty_empty_mixing", 0.0)))
                kind = ("closed-shell (filled-filled)"
                        if shift.get("class") == "filled" else "empty-empty")
                lines.append(
                    "frozen-ion parent "
                    f"{overbar_label(shift.get('parent_label', ''))} "
                    f"{_signed(shift['parent_e'])} eV (population "
                    f"{float(shift['parent_share']):.2f}): {kind} mixing "
                    f"{_signed(mixing)}, covalent shift "
                    f"{_signed(shift['covalent_shift'])} eV")
            occupied_empty = (ionic.get("cohp_inter_split") or {}).get("occ-emp")
            if occupied_empty is not None:
                lines.append("donor-acceptor COHP (frozen-ion basis, "
                             f"occupied-empty): {_signed(occupied_empty, 3)} eV")
        others = []
        for picture in ("lowdin", "ionic"):
            other = context["by_id"].get(picture, {}).get(identifier)
            if picture == self.pictures["colour"] or other is None:
                continue
            if other.get("cohp_inter") is not None:
                others.append(f"{PICTURE_NAMES[picture]} {other.get('bond')} "
                              f"({_signed(other['cohp_inter'], 3)})")
        if others:
            lines.append("verdict in the other convention: " + "; ".join(others))
        if lost > LOST_NOTE:
            lines.append(
                f"NOTE: {lost:.3f} of the "
                f"{PICTURE_NAMES[self.pictures['connector']]} population sits "
                "on sublattice orbitals that no "
                f"{PICTURE_NAMES[self.pictures['parent']]} parent holds")
        level.detail = "\n".join(lines)
        return level

    def _cohp_lines(self, colour: dict, bond, context: dict) -> list:
        """Inter-sublattice COHP of the colour picture, total and by pair."""
        cohp = colour.get("cohp_inter")
        if cohp is None:
            return []
        lines = [f"{bond or 'not classified'}: inter-sublattice COHP "
                 f"({PICTURE_NAMES[self.pictures['colour']]}) "
                 f"{_signed(cohp, 3)} eV (negative = bonding; below "
                 f"{self.cohp_tol:g} eV in magnitude nonbonding)"]
        split = colour.get("cohp_inter_split") or {}
        if split:
            lines.append("  occupied-occupied / occupied-empty / empty-empty: "
                         + " / ".join(_signed(split.get(key, 0.0), 3)
                                      for key in ("occ-occ", "occ-emp",
                                                  "emp-emp")) + " eV")
        inter, intra = {}, {}
        for pair, value in (colour.get("cohp_pairs") or {}).items():
            first, _, second = str(pair).partition(" | ")
            columns = (context["shell_column"].get(first),
                       context["shell_column"].get(second))
            same = None not in columns and columns[0] == columns[1]
            (intra if same else inter)[pair] = value
        if _top(inter, 4, PAIR_FLOOR):
            lines.append("  shell pairs: " + ", ".join(
                f"{pair} {_signed(value, 3)}"
                for pair, value in _top(inter, 4, PAIR_FLOOR)))
        if _top(intra, 2, PAIR_FLOOR):
            lines.append("  intra-sublattice: " + ", ".join(
                f"{pair} {_signed(value, 3)}"
                for pair, value in _top(intra, 2, PAIR_FLOOR)))
        return lines

    def _k_summary(self, crystal_levels: list, context: dict) -> None:
        """The k-point summary: covalency count and effective levels.

        Stored as ``k_notes[name]`` (HTML), which the writer shows under the
        k-point buttons.
        """
        name = context["name"]
        total = context["covalency_total"]
        effective = sum(1 for level in crystal_levels
                        if getattr(level, "effective", False))
        main = ", ".join(f"{shell} {float(value):.3f}" for shell, value
                         in _top(context["covalency"], 3, 0.0005))
        covalency = ("" if total is None else
                     f"covalency count {float(total):.3f} e per cell in "
                     "formally empty shells" + (f" ({main})" if main else ""))
        note = covalency + (
            f"{'; ' if covalency else ''}{effective} effective outer "
            f"level{'s' if effective != 1 else ''} above the frozen window"
            if effective else "")
        if note:
            self.k_notes[name] = f"{name} (frozen-ion picture): {note}"

    # ------------------------------------------------------------ k points

    def build_entries(self, kpoints=None, window=None) -> list:
        """The ``(name, kpoint, levels)`` entries of the writer.

        Args:
            kpoints: k-point names to draw, in this order (a name or a
                list); ``None`` draws every k point of the results.
            window: ``(low, high)`` in eV relative to the VBM; levels outside
                are left off the page (``None`` entries are open bounds).
                ``None`` keeps every level of the model space.

        Returns:
            One ``(name, kpoint, {column: [DiagramLevel]})`` per k point.

        Raises:
            SystemExit: An unknown k-point name, or nothing left to draw.
        """
        records = list(self.results.get("kpoints") or [])
        available = [str(record.get("name")) for record in records]
        if kpoints is not None:
            wanted = [kpoints] if isinstance(kpoints, str) else list(kpoints)
            unknown = [name for name in wanted if name not in available]
            if unknown:
                raise SystemExit(
                    f"ERROR: k point(s) {', '.join(unknown)} not in the overlap "
                    f"results (available: {', '.join(available) or 'none'}).")
            records = [records[available.index(name)] for name in wanted]
        low, high = window if window is not None else (None, None)
        entries = []
        covalency = []
        for record in records:
            levels = self.levels_at(record)
            if window is not None:
                for column in levels:
                    levels[column] = [
                        level for level in levels[column]
                        if (low is None or level.energy >= low)
                        and (high is None or level.energy <= high)]
            if not levels["mo"]:
                continue
            kpoint = [round(float(value), 8)
                      for value in record.get("frac") or (0.0, 0.0, 0.0)]
            entries.append((str(record.get("name")), kpoint, levels))
            total = ((record.get("phase3") or {}).get("pictures") or {}).get(
                "ionic", {}).get("covalency_total")
            if total is not None:
                covalency.append(f"{record.get('name')} {float(total):.3f}")
        if not entries:
            raise SystemExit("ERROR: no k point of the overlap results has a "
                             "crystal level to draw"
                             + (" inside the window." if window else "."))
        has_effective = any(getattr(level, "effective", False)
                            for _, _, levels in entries for level in levels["mo"])
        self.extra_chips = [
            f"frozen window E<sub>VBM</sub> + {self.window_top:g} eV"
            + ("; effective outer levels dotted" if has_effective else ""),
            f"parents: {_PICTURE_HTML[self.pictures['parent']]} | colours: "
            f"{_PICTURE_HTML[self.pictures['colour']]} COHP | connectors: "
            f"{_PICTURE_HTML[self.pictures['connector']]} populations",
        ]
        if covalency:
            self.extra_chips.append("covalency count (frozen-ion, e / cell): "
                                    + " | ".join(covalency))
        self.extra_chips.append("E &minus; E<sub>VBM</sub>")
        if self.spinor:
            self.extra_chips.append("spin-orbit coupling: spinor states, "
                                    "double-group irreps")
        return entries

    def default_label(self) -> str:
        """``"SrTiO3, Pm-3m"`` (``" (SOC)"`` appended for spinor results)."""
        label = (f"{self.crystal_formula}, "
                 f"{self.builder.spglib_dataset['international']}")
        return label + (" (SOC)" if self.spinor else "")


def write_overlap_diagram_html(results, cell, left_tokens, right_tokens,
                               output_path, *, kpoints=None,
                               structure_label=None, window=None,
                               parent_picture="ionic", colour_picture="lowdin",
                               connector_picture="lowdin",
                               symprec=1e-5):
    """Write the interactive crystal-orbital diagram of overlap-engine results.

    The page family of every ``crystod --diagram`` engine (one diagram per k
    point, k buttons, energy-window controls, level panel), drawn in the
    hybrid convention described in the module docstring.

    Args:
        results: The overlap-engine results dict, or the path of the JSON
            (``.json`` / ``.json.gz``) it was written to.
        cell: The crystal run's own structure (``PhonopyAtoms``), e.g. from
            ``crystod.star_of_k.read_poscar_or_exit(".../BAND/POSCAR")``.
        left_tokens: ``--co-left`` formula tokens, e.g. ``["SrTi"]``.
        right_tokens: ``--co-right`` formula tokens, e.g. ``["O3"]``.
        output_path: The HTML file to write.
        kpoints: k-point names to draw, in this order; ``None`` draws all.
        structure_label: Page title and first chip; default
            ``"SrTiO3, Pm-3m"`` (``" (SOC)"`` appended for spinor results).
        window: ``(low, high)`` in eV relative to the VBM: levels outside are
            left off the page; ``None`` keeps the whole model space (the
            interactive view opens on the frontier states either way).
        parent_picture: Parents of the fragment columns: ``"ionic"``
            (frozen-ion reference, default), ``"lowdin"`` or ``"bare"``.
        colour_picture: Picture whose inter-sublattice COHP colours the
            crystal levels (default ``"lowdin"``).
        connector_picture: Picture whose populations give the connectors
            (default ``"lowdin"``).
        symprec: Symmetry tolerance of the space-group chip.

    Returns:
        ``(page, entries)``: the :class:`OverlapDiagramPage` and its
        ``(name, kpoint, levels)`` entries, as drawn (the terminal report
        prints their dipole selection rules from them).

    Raises:
        SystemExit: Tokens that do not match the sublattice runs, a cell
            that is not the results' crystal, unknown pictures or k points.

    Example:
        >>> from crystod.star_of_k import read_poscar_or_exit
        >>> cell = read_poscar_or_exit("BAND/POSCAR")       # doctest: +SKIP
        >>> write_overlap_diagram_html(                      # doctest: +SKIP
        ...     "SrTiO3_onsite.json", cell, ["SrTi"], ["O3"],
        ...     "CrystOD_SrTiO3_vasp_overlap.html")
    """
    if isinstance(results, (str, os.PathLike)):
        results = load_results(results)
    page = OverlapDiagramPage(results, cell, left_tokens, right_tokens,
                              parent_picture=parent_picture,
                              colour_picture=colour_picture,
                              connector_picture=connector_picture,
                              symprec=symprec)
    entries = page.build_entries(kpoints=kpoints, window=window)
    write_crystal_diagram_html(page, entries, output_path,
                               structure_label or page.default_label())
    return page, entries
