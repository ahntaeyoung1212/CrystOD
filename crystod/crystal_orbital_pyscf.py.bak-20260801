r"""Quantitative crystal-orbital diagrams via PySCF PBC
(``crystod --diagram --pyscf``).

The crystalline counterpart of ``crystod-mol --diagram --pyscf``.  The
symmetry + extended-Hueckel diagram of :mod:`crystod.crystal_orbital_diagram`
is replaced by three periodic self-consistent-field calculations that share
one atomic-orbital space:

======================= ============================== ==================
calculation             real atoms                     point charges
======================= ============================== ==================
left fragment           the ``--co-left`` sublattice   the right sublattice
right fragment          the ``--co-right`` sublattice  the left sublattice
crystal                 everything                     none
======================= ============================== ==================

The removed sublattice stays in the basis as **ghost atoms** (basis functions
without a nucleus), so all three calculations span the same AO space and the
crystal orbitals can be projected exactly onto the fragment Bloch orbitals --
the counterpoise-consistent construction of the molecular version, carried
over to a periodic system.  In addition the removed sublattice acts on the
fragment through its **formal-charge point lattice**, evaluated as a
jellium-referenced Ewald/FFT potential, so each fragment is the electronic
state of one sublattice in the Madelung field of the other: the state before
chemical bond formation.

Charges
-------
Both the ions that remain and the point charges that replace the removed
sublattice carry the formal oxidation states (``--oxidation`` to override).
For ScF3 that makes the left fragment Sc(3+) with three F(-1) point charges
and the right fragment 3 F(-1) with one Sc(3+) point charge, so

* every cell is **electrically neutral**, which removes the monopole
  divergence a charged fragment array would introduce (the residual
  constant offset between the calculations is handled by the deep-level
  alignment below);
* every electron count is even, so a restricted (KRKS/KRHF) treatment is
  consistent;
* the fragment electron counts add up to the electron count of the crystal.

k points
--------
A crystal-orbital diagram needs the crystal orbitals at the high-symmetry
points where the irreducible representations are tabulated -- not a full band
path.  The diagram is therefore built at the special points of the space
group (the same list as ``crystod-bz --show-kpoint``), and the SCF runs on a
small regular mesh whose default is taken from the lattice constants,
n_i = round(8 Angstrom / |a_i|) (``--kmesh`` to override).

Energy reference (deep-level alignment)
---------------------------------------
The eigenvalues of a periodic calculation have no absolute zero: each of the
three calculations pins the G = 0 (cell-averaged) Coulomb potential to zero,
and although the neutral cells remove the monopole divergence, the *value* of
the average potential still depends on the second moment of the cell's charge
density -- which changes when an ion is replaced by a bare point charge.  The
result is one rigid, k-independent offset per calculation (for ScF3 the Sc
column sits ~2 eV and the F3 column ~5 eV below the crystal column), exactly
the reference problem familiar from band-offset calculations.

The columns are therefore aligned the XPS way: a **chemically inert deep
level** -- the deepest fragment level that reappears in the crystal almost
unchanged (counterpart purity >= 80%) -- must have the same energy before and
after bond formation.  The energy zero is kept at that anchor's *pre-bonding*
(fragment) value, i.e. the crystal column and the other fragment column are
shifted onto the frame of the deepest unhybridized sublattice level; the
constants are printed, and ``--no-align`` disables the whole step.

Irreducible representations
---------------------------
Every level -- fragment and crystal -- is labelled with the little-group irrep
of its Bloch state, using crystod's own machinery: the site-permutation
representation at k combined with the real-orbital Wigner matrices, projected
with the spgrep characters.  The representation is verified against PySCF's
own overlap matrix (D+ S D = S) at every k point before it is used.  Levels
of the same irrep are connected in the diagram, levels of different irreps are
not; the strength of each connection is the projection of the crystal orbital
onto the fragment orbital, and the fragment-fragment coupling matrix element
<phi_left| F(k) |phi_right> of the converged crystal Fock operator is reported
alongside, since two levels of the same irrep only mix appreciably when that
matrix element is large compared with their energy separation.

Reference: Y. Mochizuki, M. Nishibori and T. Fukushima, "Crystal Orbital
Diagram of Perovskites: A Revisit from Symmetry-Adapted Linear Combination"
(in preparation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .crystal_orbital_diagram import (
    CrystalOrbitalDiagram,
    DiagramLevel,
    _composition_string,
    _format_kpoint,
    assign_bond_characters,
    assign_fragment_compositions,
    parse_fragment_formula,
    write_crystal_diagram_html,
)
from .operations import wigner_D_real
from .runtime_compat import get_chemical_symbols, get_scaled_positions
from .visualize_basis import SymmetryAdaptedOrbitalBasis

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.52917721092

# Default target for the automatic SCF mesh: n_i = round(K / |a_i|) with the
# lattice constant in Angstrom.  K = 8 reproduces 2x2x2 for the ~4 Angstrom
# perovskite-like cells this module is written for.
KMESH_TARGET_ANGSTROM = 8.0

# A fragment level whose Mulliken population sits mostly on the ghost basis of
# the *other* sublattice is a counterpoise artifact, not a state of the
# fragment, and is dropped (same rule as crystod-mol --pyscf).
GHOST_FRACTION_THRESHOLD = 0.35

# --onsite block diagonalization: sublattice Bloch combinations whose overlap
# eigenvalue falls below this are dropped (canonical orthogonalization), the
# same near-dependence guard the SCF itself applies to the full basis
ONSITE_OVERLAP_FLOOR = 1.0e-7

# Seed window in eV for clustering degenerate levels.  The uniform FFT grid on
# which the Coulomb, the exchange-correlation and the point-charge potentials
# are evaluated does not respect the point group exactly, so levels that are
# degenerate by symmetry come out split -- by ~5e-4 eV for the occupied states
# of ScF3, but by tens of meV for the empty states of ZrO2 at a coarse cutoff.
# No fixed window can cover that range without merging genuinely distinct
# levels, so this is only the starting guess: groups are then merged until the
# irrep multiplicities come out integral (see _adaptive_groups), which is the
# statement that the group is a whole number of complete multiplets.
DEGENERACY_SEED_EV = 1.0e-3

# A group is never widened beyond this, so two levels that are really distinct
# can never be merged just because the projection is noisy.
DEGENERACY_MAX_WINDOW_EV = 0.30

# How far a multiplicity may sit from an integer and still count as one.
_MULTIPLICITY_TOL = 0.05

# A fragment level counts as chemically inert -- usable as an alignment anchor
# -- when some crystal level consists of it to at least this fraction.
ALIGNMENT_PURITY = 0.80

# crystod orders the real orbitals as in complex_to_real_transform_orbital;
# PySCF uses m = -l..l except for p, which it orders x, y, z.  The two agree
# for s, p and d and differ for f and above, so the AO rotation is permuted
# into PySCF's order.
_CRYSTOD_M_ORDER = {
    0: [0],
    1: [1, -1, 0],
    2: [-2, -1, 0, 1, 2],
    3: [3, -3, 2, -2, 1, -1, 0],
}


def _import_pyscf():
    try:
        from pyscf.pbc import dft, gto, scf, tools  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "ERROR: --pyscf requires the pyscf package (pip install pyscf)."
        ) from exc
    # some conda builds of pyscf default to a SINGLE OpenMP thread unless
    # OMP_NUM_THREADS is exported; that turns a minutes-long diagram into an
    # hours-long one.  Use every core unless the user chose otherwise.
    import os

    from pyscf import lib

    if not os.environ.get("OMP_NUM_THREADS"):
        lib.num_threads(os.cpu_count())


def _pyscf_m_order(l: int) -> list[int]:
    return [1, -1, 0] if l == 1 else list(range(-l, l + 1))


def _reorder_to_pyscf(l: int) -> np.ndarray:
    """Q with Q[pyscf_row, crystod_row] = 1 for the real orbitals of shell l."""
    crystod_order = _CRYSTOD_M_ORDER.get(l, list(range(-l, l + 1)))
    pyscf_order = _pyscf_m_order(l)
    matrix = np.zeros((2 * l + 1, 2 * l + 1))
    for column, m in enumerate(crystod_order):
        matrix[pyscf_order.index(m), column] = 1.0
    return matrix


def wigner_pyscf(l: int, rotation: np.ndarray) -> np.ndarray:
    """Real-orbital rotation matrix in PySCF's AO ordering."""
    transform = _reorder_to_pyscf(l)
    return transform @ wigner_D_real(l, rotation) @ transform.T


def default_kmesh(lattice: np.ndarray) -> list[int]:
    """n_i = round(8 Angstrom / |a_i|), at least 1 -- the user-facing rule of
    thumb that a ~4 Angstrom cell wants a 2x2x2 mesh."""
    lengths = np.linalg.norm(np.asarray(lattice, dtype=float), axis=1)
    return [max(1, int(round(KMESH_TARGET_ANGSTROM / length))) for length in lengths]


@dataclass
class AOBlock:
    """One (element, shell) block of the PySCF AO space, for level labels."""

    element: str
    shell: str
    l: int
    offset: int
    n_ao: int
    sites: list[int]
    column: str
    radial: float = 1.0    # radial amplitude at the sketch radius r0
    # amplitudes at SKETCH_RADII, for the multi-radius sign of the viewer:
    # a single probe radius can sit right on the orthogonalization node of a
    # semicore-carrying channel (Sc s of the valence R1+ of ScF3: -0.0013 at
    # 2 bohr), where the drawn sign would be decided by noise
    radial_profile: tuple = ()


class PySCFCrystalOrbitalDiagram(CrystalOrbitalDiagram):
    """Crystal-orbital diagram from three periodic PySCF calculations.

    Subclasses the extended-Hueckel engine only to reuse its k-point list,
    degenerate-group clustering, irrep projection, level filling and supercell
    helper; the Hamiltonian, the overlap and the orbitals all come from PySCF.
    """

    def __init__(self, cell, left_tokens, right_tokens, *, symprec=1e-5,
                 basis="gth-dzvp-molopt-sr", pseudo="gth-pbe", xc="pbe",
                 kmesh=None, ke_cutoff=200.0, oxidation=None, electrons=None,
                 sigma=0.0, degeneracy_tol=None, no_ghost=False, symmetrize=True,
                 max_l=None, projection="lowdin", chk=None, onsite=False,
                 conv_tol=1e-8, max_cycle=100, max_memory=4000.0, verbose=0):
        _import_pyscf()

        self.builder = SymmetryAdaptedOrbitalBasis(cell=cell, symprec=symprec)
        primitive = self.builder.primitive_cell
        self.symbols = get_chemical_symbols(primitive)
        self.positions = np.array(get_scaled_positions(primitive))
        self.lattice = np.array(primitive.cell)
        self.cartesian = self.positions @ self.lattice

        self.basis_name = basis
        self.pseudo_name = pseudo
        self.xc = xc
        self.ke_cutoff = ke_cutoff
        if ke_cutoff and ke_cutoff < 80.0:
            # measured on ScF3 (dzvp, 2x2x2): below ~80 Hartree the FFT grid
            # cannot resolve the compact Gaussians (semicore shells) and the
            # SCF does not converge at all -- the grid here is the DENSITY
            # grid, the analogue of VASP's augmentation grid (~4x ENCUT in
            # energy), not of ENCUT itself.
            print(f"WARNING: --ke-cutoff {ke_cutoff:g} Hartree is below the "
                  "~80 Hartree the GTH Gaussian basis needs; expect the SCF "
                  "to fail. 100-200 Hartree is the working range.")
        self.sigma = float(sigma)
        self.no_ghost = bool(no_ghost)
        self.symmetrize = bool(symmetrize)
        self.max_l = max_l
        if projection not in ("lowdin", "mulliken"):
            raise SystemExit(
                "ERROR: --projection must be 'lowdin' or 'mulliken', "
                f"not {projection!r}.")
        self.projection = projection
        # display label for the population rows ("Loewdin: ..."/"Mulliken: ...")
        self.projection_label = ("Loewdin" if projection == "lowdin"
                                 else "Mulliken")
        self.chk_path = chk
        self.degeneracy_tol = (DEGENERACY_SEED_EV if degeneracy_tol is None
                               else float(degeneracy_tol))
        self.degeneracy_window = DEGENERACY_MAX_WINDOW_EV
        self.retry_sigma = 0.2
        self.smeared: set[str] = set()
        self.conv_tol = conv_tol
        self.max_cycle = max_cycle
        self.max_memory = max_memory
        self.pyscf_verbose = verbose
        self.sketch_specs = None      # sketches always use all AO components
        self.sketch_tokens = None
        self.method_chip = f"PySCF {xc.upper()}/{basis}"
        # --onsite: single-Hamiltonian mode.  Only the crystal SCF runs; the
        # fragment columns are the sublattice BLOCKS of its converged Fock,
        # F[rows,rows] c = E S[rows,rows] c -- the pre-bonding sublattice
        # states with the left-right mixing switched off.  One operator for
        # every column, so no reference alignment is needed and a level's
        # rise/drop against its parents is purely the orbital interaction.
        self.onsite = bool(onsite)
        self.scf_columns = ("mo",) if self.onsite else ("mo", "left", "right")
        if self.onsite and self.no_ghost:
            print("NOTE: --onsite ignores --no-ghost (no fragment SCF runs; "
                  "the columns are crystal-Fock blocks).")
            # canonicalize: the flag is numerically inert here, and keeping
            # it would poison the chk parameter check for no reason
            self.no_ghost = False
        self.basis_chip = "(GTH valence basis)"
        if self.onsite:
            self.embedding_chip = "one Hamiltonian: crystal-Fock blocks"
            self.foot_intro = (
                "Crystal-orbital diagram (COD, --onsite): every column comes "
                "from the ONE converged crystal Fock operator F(k).  The "
                "fragment columns are the per-(element, shell) ON-SITE "
                "multiplets -- F(k) diagonalized within each shell's own "
                "symmetry-adapted Bloch orbitals, the tight-binding on-site "
                "energies &lt;&phi;|F|&phi;&gt; of the actual AO shells, one "
                "level per induced irrep and no cross-shell mixing -- the "
                "center its full eigenstates; states sharing an irrep of the "
                "little group at k mix into bonding/antibonding crystal "
                "orbitals.  No fragment SCF and no reference alignment: a "
                "crystal level's drop/rise against its parents is the "
                "orbital interaction (level repulsion/hybridization) with "
                "everything else, cross-shell and left&ndash;right alike.  "
                "Column occupations (electron arrows) are the formal ionic "
                "counts from the oxidation states -- a display convention, "
                "not an output of the Fock operator.")
        else:
            self.embedding_chip = ""
            self.foot_intro = (
                "Crystal-orbital diagram (COD): the fragment-sublattice "
                "Bloch orbitals (columns) are the electronic states before "
                "chemical bond formation -- each fragment is its own "
                "periodic DFT calculation (formal-charge ions + ghost basis "
                "+ point-charge lattice of the removed sublattice); states "
                "sharing an irrep of the little group at k mix into "
                "bonding/antibonding crystal orbitals (center).  Energies: "
                f"PySCF {xc.upper()} eigenvalues, the three calculations put "
                "on one scale by deep-level (XPS-style) alignment.  NOTE the "
                "parent&rarr;crystal vertical offsets also carry the "
                "point-charge-model-vs-crystal environment difference "
                "(site-dependent, up to ~1.5 eV) -- read bonding from the "
                "line colors (COOP), not from the offsets; --onsite removes "
                "this by drawing every column from the crystal Fock itself.")

        self._assign_fragments(left_tokens, right_tokens)
        self._resolve_oxidation(oxidation)

        self.kmesh = list(kmesh) if kmesh else default_kmesh(self.lattice)
        if any(n < 1 for n in self.kmesh):
            raise SystemExit("ERROR: every --kmesh entry must be at least 1.")

        # the crystal cell first: with --no-ghost the fragment cells span only
        # their own sublattice's AOs and are embedded into the crystal's AO
        # space through the atom slices of the crystal cell
        self.cells = {"mo": self._make_cell(None)}
        self.n_ao = int(self.cells["mo"].nao_nr())
        slices = self.cells["mo"].aoslice_by_atom()
        self.embed_rows = {
            column: np.concatenate([
                np.arange(int(slices[atom, 2]), int(slices[atom, 3]))
                for atom in self.side_atoms[column]
            ])
            for column in ("left", "right")
        }
        self.cells["left"] = self._make_cell("left")
        self.cells["right"] = self._make_cell("right")
        self._check_shared_ao_space()
        self._build_ao_blocks()

        self.side_electrons = {
            column: int(self.cells[column].nelectron) for column in ("left", "right")
        }
        crystal_electrons = int(self.cells["mo"].nelectron)
        if electrons is None:
            electrons = crystal_electrons
        self.electrons = float(electrons)
        self.crystal_electrons = crystal_electrons

        self.mean_field = {}
        self.density_matrix = {}
        self.scf_energy = {}
        self._pc_potential = {}

    # ------------------------------------------------------------ fragments

    def _assign_fragments(self, left_tokens, right_tokens) -> None:
        formulas = {
            "left": parse_fragment_formula(left_tokens, "--co-left"),
            "right": parse_fragment_formula(right_tokens, "--co-right"),
        }
        composition: dict[str, int] = {}
        for symbol in self.symbols:
            composition[symbol] = composition.get(symbol, 0) + 1
        comp_str = _composition_string(self.symbols)
        flags = {"left": "--co-left", "right": "--co-right"}

        assigned: dict[str, str] = {}
        for column in ("left", "right"):
            for element, count in formulas[column]:
                if element in assigned:
                    raise SystemExit(
                        f"ERROR: element {element} is listed more than once "
                        "across --co-left/--co-right."
                    )
                if element not in composition:
                    raise SystemExit(
                        f"ERROR: element {element} is not in the structure "
                        f"(primitive-cell composition: {comp_str})."
                    )
                if count is not None and count != composition[element]:
                    raise SystemExit(
                        f"ERROR: {flags[column]} lists {element}{count} but the "
                        f"primitive cell has {composition[element]} {element} "
                        f"atom(s) (composition: {comp_str})."
                    )
                assigned[element] = column
        missing = [element for element in composition if element not in assigned]
        if missing:
            raise SystemExit(
                f"ERROR: element(s) {', '.join(missing)} not assigned to "
                f"--co-left/--co-right (primitive-cell composition: {comp_str}; "
                "every atom must belong to one fragment)."
            )
        self.element_column = assigned
        self.atom_column = [assigned[symbol] for symbol in self.symbols]
        self.side_atoms = {
            column: [index for index, side in enumerate(self.atom_column) if side == column]
            for column in ("left", "right")
        }
        self.formula = {
            column: _composition_string([self.symbols[i] for i in self.side_atoms[column]])
            for column in ("left", "right")
        }

    def _resolve_oxidation(self, oxidation) -> None:
        composition: dict[str, int] = {}
        for symbol in self.symbols:
            composition[symbol] = composition.get(symbol, 0) + 1
        if oxidation is None:
            from pymatgen.core import Composition

            guesses = Composition(_composition_string(self.symbols)).oxi_state_guesses()
            if not guesses:
                raise SystemExit(
                    "ERROR: could not guess the oxidation states of "
                    f"{_composition_string(self.symbols)}; pass them explicitly, "
                    "e.g. --oxidation Sc=+3 F=-1."
                )
            oxidation = {element: float(q) for element, q in guesses[0].items()}
        missing = [element for element in composition if element not in oxidation]
        if missing:
            raise SystemExit(f"ERROR: --oxidation misses element(s) {', '.join(missing)}.")
        net = sum(oxidation[element] * count for element, count in composition.items())
        if abs(net) > 1e-6:
            raise SystemExit(
                f"ERROR: oxidation states are not charge-neutral (net {net:+g} per cell)."
            )
        self.oxidation = oxidation
        self.side_charge = {
            column: sum(self.oxidation[self.symbols[i]] for i in self.side_atoms[column])
            for column in ("left", "right")
        }

    # ---------------------------------------------------------- pyscf cells

    def _make_cell(self, column):
        """Cell in which only ``column``'s atoms are real; None = the crystal.

        By default every atom keeps its basis functions (ghosts elsewhere), so
        all three cells span one AO space (counterpoise-consistent).  With
        ``no_ghost`` the fragment cells contain no trace of the removed
        sublattice at all -- its basis functions are excluded from the
        variational space, which is the hard constraint that no fragment
        wave function can sit on the removed atoms.
        """
        from pyscf.pbc import gto

        atoms, basis, pseudo = [], {}, {}
        for index, symbol in enumerate(self.symbols):
            real = column is None or self.atom_column[index] == column
            if not real and self.no_ghost:
                continue
            tag = f"{symbol}{index}" if real else f"ghost-{symbol}{index}"
            shells = gto.basis.load(self.basis_name, symbol)
            if self.max_l is not None:
                shells = [shell for shell in shells if shell[0] <= self.max_l]
                if not shells:
                    raise SystemExit(
                        f"ERROR: --max-l {self.max_l} removes every shell of "
                        f"{symbol}.")
            basis[tag] = shells
            if real:
                pseudo[tag] = self.pseudo_name
            atoms.append((tag, tuple(self.cartesian[index])))

        cell = gto.Cell()
        cell.a = self.lattice
        cell.atom = atoms
        cell.unit = "Angstrom"
        cell.basis = basis
        cell.pseudo = pseudo
        # formal-charge ions: the fragment carries the total oxidation state of
        # its own sublattice, so together with the point charges of the removed
        # one the cell is neutral
        cell.charge = 0 if column is None else int(round(self.side_charge[column]))
        cell.ke_cutoff = self.ke_cutoff
        cell.max_memory = self.max_memory
        cell.verbose = self.pyscf_verbose
        # required by make_kpts(space_group_symmetry=True): the SCF then only
        # solves the irreducible wedge of the mesh (2x2x2 of Pm-3m: 4 of 8)
        cell.space_group_symmetry = True
        cell.symmorphic = False
        cell.build()
        if cell.nelectron % 2:
            raise SystemExit(
                f"ERROR: the {'crystal' if column is None else column + ' fragment'} has "
                f"{cell.nelectron} electrons, which this restricted driver cannot treat.\n"
                "       Check --oxidation; the formal charges must make every fragment "
                "closed-shell."
            )
        return cell

    def _check_shared_ao_space(self) -> None:
        for column in ("left", "right"):
            expected = (len(self.embed_rows[column]) if self.no_ghost
                        else self.n_ao)
            actual = int(self.cells[column].nao_nr())
            if actual != expected:
                raise SystemExit(
                    f"ERROR: the {column} fragment spans {actual} AOs, expected "
                    f"{expected}; this is a bug, please report it."
                )

    def _embed(self, column, coefficients):
        """Zero-pad small-basis fragment eigenvectors into the shared AO space
        (identity without --no-ghost)."""
        if column == "mo" or not self.no_ghost:
            return coefficients
        full = np.zeros((self.n_ao, coefficients.shape[1]),
                        dtype=coefficients.dtype)
        full[self.embed_rows[column]] = coefficients
        return full

    def _build_ao_blocks(self) -> None:
        """(element, shell) blocks of the PySCF AO space, in AO order."""
        cell = self.cells["mo"]
        slices = cell.aoslice_by_atom()
        raw_labels = [label.split() for label in cell.ao_labels()]
        blocks: list[AOBlock] = []
        index = 0
        sketch_r0 = 2.0  # bohr, the representative bonding-region radius
        sketch_radii = (1.5, 2.0, 2.5, 3.0)
        for shell in range(cell.nbas):
            atom = int(cell.bas_atom(shell))
            l = int(cell.bas_angular(shell))
            exponents = np.asarray(cell.bas_exp(shell), dtype=float)
            contractions = np.asarray(cell.bas_ctr_coeff(shell), dtype=float)
            # radial amplitude of each contracted function at r0, for the
            # wave-function sketch: same-l shells of one atom accumulate with
            # these weights, so the drawn lobe signs are those of the real
            # wave function there (a bare coefficient of one shell would be
            # wrong for semicore levels, whose orthogonalization tails invert)
            radial_amplitudes = (
                sketch_r0 ** l
                * (contractions * np.exp(-exponents * sketch_r0 ** 2)[:, None]).sum(axis=0)
            )
            profiles = [
                radius ** l
                * (contractions * np.exp(-exponents * radius ** 2)[:, None]).sum(axis=0)
                for radius in sketch_radii
            ]
            for contraction in range(cell.bas_nctr(shell)):
                # the AO label carries the shell name, e.g. '0 Sc 3dxy' -> '3d'
                name = raw_labels[index][2]
                shell_name = name[: len(name) - len(name.lstrip("0123456789"))] + "spdfgh"[l]
                blocks.append(AOBlock(
                    element=self.symbols[atom], shell=shell_name, l=l,
                    offset=index, n_ao=2 * l + 1, sites=[atom],
                    column=self.atom_column[atom],
                    radial=float(radial_amplitudes[contraction]),
                    radial_profile=tuple(
                        float(profile[contraction]) for profile in profiles),
                ))
                index += 2 * l + 1
        self.ao_blocks = blocks
        self.ao_offset_in_atom = {
            block.offset: block.offset - int(slices[block.sites[0], 2]) for block in blocks
        }
        # merged (element, shell) specs for the level labels and the report
        merged: dict[tuple[str, str], AOBlock] = {}
        self.side_specs = {"left": [], "right": []}
        for block in blocks:
            key = (block.element, block.shell)
            if key in merged:
                merged[key].sites.append(block.sites[0])
                continue
            spec = AOBlock(block.element, block.shell, block.l, block.offset,
                           block.n_ao, list(block.sites), block.column)
            merged[key] = spec
        self.specs = list(merged.values())
        for spec in self.specs:
            self.side_specs[spec.column].append(spec)
        # per-(element, shell) AO index lists, for Mulliken weights
        self.spec_indices = {}
        for (element, shell), spec in merged.items():
            indices: list[int] = []
            for block in blocks:
                if block.element == element and block.shell == shell:
                    indices.extend(range(block.offset, block.offset + block.n_ao))
            self.spec_indices[id(spec)] = np.array(indices, dtype=int)

    # --------------------------------------------------- point-charge field

    def _point_charge_potential(self, cell, column):
        """Grid values of -sum_i q_i/|r - R_i| for the removed sublattice.

        Jellium-referenced (the G = 0 term is dropped), which is the same zero
        of potential PySCF uses for the electrons and the nuclei, so all three
        calculations share one energy reference.  Validated against
        :func:`crystod.point_charge_field.ewald_site_potential`.
        """
        from pyscf.pbc import tools

        other = {"left": "right", "right": "left"}[column]
        sites = self.side_atoms[other]
        if not sites:
            return None
        positions = np.array([self.cartesian[i] for i in sites]) / BOHR_TO_ANGSTROM
        charges = np.array([self.oxidation[self.symbols[i]] for i in sites], dtype=float)

        mesh = cell.mesh
        Gv = cell.get_Gv(mesh)
        coulG = tools.get_coulG(cell, mesh=mesh, Gv=Gv)
        structure = np.exp(-1j * np.einsum("gx,ix->gi", Gv, positions))
        # the electron charge is -1, hence the minus sign on the charges
        potential_G = (structure @ (-charges)) * coulG
        return tools.ifft(potential_G, mesh).real

    def _point_charge_matrix(self, cell, column, kpts):
        """<phi_mu k| V_pointcharge |phi_nu k> for every k of ``kpts``.

        The potential itself depends only on the cell and is cached; only the
        AO quadrature is repeated, so this is cheap to call again for the
        band k points of the diagram.
        """
        from pyscf.pbc.dft import numint

        if column not in self._pc_potential:
            self._pc_potential[column] = self._point_charge_potential(cell, column)
        potential = self._pc_potential[column]
        if potential is None:
            return None
        coords = cell.get_uniform_grids(cell.mesh)
        weight = cell.vol / len(coords)
        matrices = []
        for ao in numint.KNumInt().eval_ao(cell, coords, np.asarray(kpts).reshape(-1, 3)):
            matrices.append(np.einsum("gi,g,gj->ij", ao.conj(), potential * weight, ao))
        return np.asarray(matrices)

    def _hcore_with_point_charges(self, cell, column, bare_get_hcore):
        """``get_hcore`` replacement that follows whatever k points it is asked
        for -- the SCF mesh during the SCF, the special point during the band
        step -- instead of freezing the mesh-sized matrix."""
        cache: dict[bytes, np.ndarray] = {}

        def get_hcore(cell_arg=None, kpts=None):
            target = cell_arg if cell_arg is not None else cell
            bare = np.asarray(bare_get_hcore(target, kpts))
            if kpts is None:
                requested = target.make_kpts([1, 1, 1])
            elif hasattr(kpts, "kpts_ibz"):
                # symmetry-adapted SCF: the Hamiltonian lives on the wedge
                requested = np.asarray(kpts.kpts_ibz)
            else:
                requested = np.asarray(kpts)
            requested = requested.reshape(-1, 3)
            key = np.ascontiguousarray(requested).tobytes()
            if key not in cache:
                cache[key] = self._point_charge_matrix(target, column, requested)
            correction = cache[key]
            return bare if correction is None else bare + correction

        return get_hcore

    # ------------------------------------------------------------------ SCF

    def _make_mean_field(self, cell, kpts, column, sigma=0.0, level_shift=0.0):
        """KRKS/KRHF with the point-charge field of the removed sublattice."""
        from pyscf.pbc import dft, scf

        if self.xc.lower() in {"hf", "hartree-fock"}:
            mean_field = scf.KRHF(cell, kpts)
            hybrid = True
        else:
            mean_field = dft.KRKS(cell, kpts)
            mean_field.xc = self.xc
            hybrid = abs(mean_field._numint.hybrid_coeff(self.xc)) > 1e-10
        if hybrid and not getattr(self, "_hybrid_warned", False):
            self._hybrid_warned = True
            print("NOTE: exact exchange is evaluated on the FFT grid here; its grid error\n"
                  "      is much larger than the semilocal one (raw degeneracy splittings\n"
                  "      of a few 0.1 eV at ke_cutoff 80). The diagram re-diagonalizes the\n"
                  "      group-averaged Fock, so the levels stay exactly symmetric, but for\n"
                  "      hybrid energetics prefer --ke-cutoff 150 or more.")
        mean_field.conv_tol = self.conv_tol
        mean_field.max_cycle = self.max_cycle
        mean_field.max_memory = self.max_memory
        # MINAO/atom guesses assume all-electron shell structures and crash for
        # compact GTH sets; hcore is the safe choice here
        mean_field.init_guess = "hcore"

        if column != "mo":
            mean_field.get_hcore = self._hcore_with_point_charges(
                cell, column, type(mean_field).get_hcore.__get__(mean_field)
            )
        if level_shift > 0:
            # classic remedy for oscillating charged cells: bias the virtuals
            # during the SCF (removed again before any eigenvalue is reported)
            mean_field.level_shift = level_shift
        if sigma > 0:
            from pyscf.pbc.scf import addons

            mean_field = addons.smearing_(mean_field, sigma=sigma / HARTREE_TO_EV,
                                          method="fermi")
        return mean_field

    def _convergence_ladder(self):
        """Fallback (smearing eV, max cycles, level shift Hartree), in order."""
        cycles = 3 * self.max_cycle
        base = self.sigma if self.sigma > 0 else 0.2
        return [
            (base, cycles, 0.0),
            # diffuse cation shells in a highly charged fragment cell make the
            # aufbau occupations oscillate; a virtual-level shift stabilizes it
            (base, cycles, 0.3),
            (5.0 * base, cycles, 0.3),
        ]

    def run(self, report=print) -> None:
        """The three self-consistent calculations on the regular mesh."""
        import os

        if self.chk_path and os.path.exists(self.chk_path):
            self._load_chk(report)
            return

        # The crystal is solved first: its converged density restricted to one
        # sublattice's AO block (ghost rows/columns zeroed) is the best
        # available guess for that fragment -- it differs from the pre-bonding
        # state only by the bonding redistribution.  The hcore guess that
        # replaces it for the crystal itself is fine there, but for a highly
        # charged cation fragment with diffuse shells (SrTi^6+ with dzvp) it
        # starts so far away that the SCF never finds its way back.
        slices = self.cells["mo"].aoslice_by_atom()
        for column in self.scf_columns:
            cell = self.cells[column]
            # symmetry-reduced SCF mesh: only the irreducible wedge is solved
            # (2x2x2 in Pm-3m: 4 of 8 k points), then to_khf() expands the
            # result back to the full mesh for the band step and the guesses
            try:
                kpts = cell.make_kpts(self.kmesh, space_group_symmetry=True,
                                      time_reversal_symmetry=True)
                reduced = kpts.nkpts_ibz < kpts.nkpts
            except Exception:
                kpts = cell.make_kpts(self.kmesh)
                reduced = False
            if not reduced and not isinstance(kpts, np.ndarray):
                kpts = cell.make_kpts(self.kmesh)
            mean_field = self._make_mean_field(cell, kpts, column, sigma=self.sigma)

            guess = None
            if column != "mo" and "mo" in self.density_matrix:
                if self.no_ghost:
                    # the fragment basis IS the sublattice block: take the
                    # crystal density restricted to it
                    rows = self.embed_rows[column]
                    guess = np.asarray(self.density_matrix["mo"])[
                        :, rows[:, None], rows[None, :]]
                else:
                    mask = np.zeros(self.n_ao)
                    for atom in self.side_atoms[column]:
                        mask[int(slices[atom, 2]):int(slices[atom, 3])] = 1.0
                    guess = (np.asarray(self.density_matrix["mo"])
                             * mask[None, :, None] * mask[None, None, :])
                if reduced:
                    # the stored crystal density lives on the full mesh; the
                    # symmetry-adapted SCF wants it on the irreducible wedge
                    guess = guess[np.asarray(kpts.ibz2bz)]
            energy = mean_field.kernel(dm0=guess)
            # A degenerate level straddling the Fermi energy makes the aufbau
            # occupation jump between iterations (typical on a coarse mesh and
            # for the highly charged cation fragments); fractional occupations
            # and more cycles fix it.  Escalate, and say which rung was used --
            # smearing changes the level occupancies in the diagram.
            for sigma, cycles, shift in self._convergence_ladder():
                if mean_field.converged:
                    break
                report(f"   {column:<5} not converged; retrying with "
                       f"{sigma} eV Fermi smearing"
                       + (f" and a {shift} Hartree virtual-level shift" if shift else "")
                       + f" ({cycles} cycles)")
                mean_field = self._make_mean_field(cell, kpts, column,
                                                   sigma=sigma, level_shift=shift)
                mean_field.max_cycle = cycles
                energy = mean_field.kernel(dm0=guess)
                if sigma > 0:
                    self.smeared.add(column)
            # the shift must not appear in any reported eigenvalue: get_bands
            # rebuilds the Fock operator from the converged density, so with the
            # attribute cleared the band energies are shift-free
            mean_field.level_shift = 0.0
            if not mean_field.converged:
                raise SystemExit(
                    f"ERROR: the {column} SCF did not converge, even with Fermi "
                    "smearing. Try a different --kmesh, a larger --max-cycle or "
                    "--sigma, or a different --basis."
                )
            if reduced:
                # expand the irreducible wedge back to the full mesh, exactly
                # like the VASP-style band scripts; everything downstream
                # (get_bands, the masked guesses) then sees a plain KRKS.
                # to_khf() builds a fresh object, so the point-charge hcore
                # override must be re-attached for the fragments.
                mean_field = mean_field.to_khf()
                if column != "mo":
                    mean_field.get_hcore = self._hcore_with_point_charges(
                        cell, column,
                        type(mean_field).get_hcore.__get__(mean_field),
                    )
            self.mean_field[column] = mean_field
            self.density_matrix[column] = mean_field.make_rdm1()
            self.scf_energy[column] = float(energy)
            report(f"   {column:<5} {self.formula.get(column, 'crystal'):<8} "
                   f"E = {energy:16.8f} Hartree   ({cell.nelectron} electrons, "
                   f"charge {cell.charge:+d})")

        if self.chk_path:
            self._save_chk(report)

    # WAVECAR-style restart: the converged density matrices are all that the
    # band step needs (get_bands rebuilds the Fock operator from them), so
    # saving the three of them plus the defining parameters skips the SCFs
    # entirely on the next run with the same --chk file.
    def _chk_params(self) -> dict:
        return {
            "version": 1,
            "basis": self.basis_name, "pseudo": self.pseudo_name,
            "xc": self.xc, "kmesh": list(self.kmesh),
            "ke_cutoff": float(self.ke_cutoff), "max_l": self.max_l,
            "no_ghost": self.no_ghost, "sigma": self.sigma,
            "electrons": float(self.electrons),
            "oxidation": {el: float(q) for el, q in self.oxidation.items()},
            "left": self.formula["left"], "right": self.formula["right"],
            "symbols": list(self.symbols),
        }

    def _save_chk(self, report) -> None:
        import json

        with open(self.chk_path, "wb") as handle:
            # a file handle keeps the exact name (np.savez would append .npz)
            # --onsite runs (and therefore saves) only the crystal SCF; the
            # energy_columns array records which densities the file holds
            np.savez_compressed(
                handle,
                params=json.dumps(self._chk_params(), sort_keys=True),
                positions=self.positions, lattice=self.lattice,
                smeared=np.array(sorted(self.smeared)),
                energy_columns=np.array(list(self.scf_columns)),
                energies=np.array([self.scf_energy[c]
                                   for c in self.scf_columns]),
                **{f"dm_{column}": np.asarray(self.density_matrix[column])
                   for column in self.scf_columns},
            )
        report(f"   SCF saved to {self.chk_path} (reuse with --chk; delete "
               "the file to force a fresh SCF)")

    def _load_chk(self, report) -> None:
        import json

        data = np.load(self.chk_path, allow_pickle=False)
        saved = json.loads(str(data["params"]))
        current = self._chk_params()
        # --onsite reads only the crystal density, which no_ghost never
        # touches (it shapes the fragment SCFs) -- a full-run chk written
        # with either setting is equally valid here
        ignored = {"no_ghost"} if self.onsite else set()
        mismatched = [key for key in current
                      if key not in ignored and saved.get(key) != current[key]]
        if (not np.allclose(np.asarray(data["positions"]), self.positions,
                            atol=1e-6)
                or not np.allclose(np.asarray(data["lattice"]), self.lattice,
                                   atol=1e-6)):
            mismatched.append("structure")
        if mismatched:
            raise SystemExit(
                f"ERROR: {self.chk_path} was written with different "
                f"parameters ({', '.join(sorted(mismatched))}); delete the "
                "file or rerun with the matching options.\n"
                f"       (crystod --chk-info {self.chk_path} shows the "
                "stored conditions and a ready-to-paste option string)")
        self.smeared = set(str(s) for s in data["smeared"])
        energies = np.asarray(data["energies"], dtype=float)
        # pre-onsite checkpoints carry no energy_columns record; they always
        # hold all three calculations in this fixed order
        stored = ([str(c) for c in data["energy_columns"]]
                  if "energy_columns" in data.files
                  else ["mo", "left", "right"])
        for column in self.scf_columns:
            if f"dm_{column}" not in data.files:
                raise SystemExit(
                    f"ERROR: {self.chk_path} was written with --onsite and "
                    "holds only the crystal density; rerun without --chk (or "
                    "with a checkpoint from a full three-SCF run) to build "
                    f"the {column} fragment column.")
            cell = self.cells[column]
            # a mean-field object is still needed for get_bands, but with the
            # density matrix passed explicitly it is never iterated
            kpts = cell.make_kpts(self.kmesh)
            self.mean_field[column] = self._make_mean_field(
                cell, kpts, column, sigma=self.sigma)
            self.density_matrix[column] = np.asarray(data[f"dm_{column}"])
            energy = float(energies[stored.index(column)])
            self.scf_energy[column] = energy
            report(f"   {column:<5} {self.formula.get(column, 'crystal'):<8} "
                   f"E = {energy:16.8f} Hartree   "
                   f"({cell.nelectron} electrons, charge {cell.charge:+d})  "
                   f"[read from {self.chk_path}]")
        # only name the calculations this run actually uses (--onsite reads
        # the crystal density alone; a smeared fragment SCF never enters it)
        relevant = self.smeared & set(self.scf_columns)
        if relevant:
            report(f"   ({', '.join(sorted(relevant))} carried Fermi "
                   "smearing when the file was written)")

    def prepare_bands(self, kpoints) -> None:
        """Diagonalise every calculation at every diagram k point in one call.

        ``get_bands`` rebuilds the density on the FFT grid each time it is
        called, so asking for all the special points at once instead of one per
        k point removes that cost from all but the first.
        """
        self._band_cache = {}
        keys = [self._kpoint_key(kpoint) for kpoint in kpoints]
        for column in self.scf_columns:
            cell = self.cells[column]
            kpts_band = cell.get_abs_kpts(np.array(kpoints, dtype=float))
            energies, coefficients = self.mean_field[column].get_bands(
                kpts_band, cell=cell, dm_kpts=self.density_matrix[column]
            )
            for key, energy, coefficient in zip(keys, energies, coefficients):
                self._band_cache[(column, key)] = (
                    np.asarray(energy).real * HARTREE_TO_EV,
                    self._embed(column, np.asarray(coefficient)),
                )

    @staticmethod
    def _kpoint_key(kpoint):
        return tuple(np.round(np.asarray(kpoint, dtype=float), 9))

    def _bands_at(self, column, kpoint):
        """(energies in eV, coefficients) of one calculation at one k point."""
        key = (column, self._kpoint_key(kpoint))
        cached = getattr(self, "_band_cache", {}).get(key)
        if cached is not None:
            return cached
        cell = self.cells[column]
        kpts_band = cell.get_abs_kpts(np.array([kpoint], dtype=float))
        energies, coefficients = self.mean_field[column].get_bands(
            kpts_band, cell=cell, dm_kpts=self.density_matrix[column]
        )
        return (np.asarray(energies[0]).real * HARTREE_TO_EV,
                self._embed(column, np.asarray(coefficients[0])))

    def _population_shares(self, vectors, degeneracy, S, S_half, specs=None):
        """Sorted per-(element, shell) AO populations of one level space in
        the selected projection (the displayed composition measure).
        ``specs`` restricts the listed shells (fragment columns list their
        own sublattice only)."""
        if self.projection == "mulliken":
            gross = (vectors.conj() * (S @ vectors)
                     ).real.sum(axis=1) / degeneracy
        else:
            gross = (np.abs(S_half @ vectors) ** 2).sum(axis=1) / degeneracy
        shares = []
        for spec in (self.specs if specs is None else specs):
            value = float(gross[self.spec_indices[id(spec)]].sum())
            if abs(value) >= 0.001:
                shares.append((value, f"{spec.element} {spec.shell}"))
        shares.sort(key=lambda item: -item[0])
        return shares

    def _block_bands(self, column, S):
        """Per-shell on-site multiplets of the crystal Fock (--onsite).

        For every (element, shell) of the sublattice, diagonalizes (F, S)
        restricted to THAT SHELL's own Bloch AOs -- the tight-binding
        on-site energies <phi|F|phi> of the actual shells, one level per
        induced irrep, matching the site-symmetry table exactly.  No
        cross-shell mixing on purpose: diagonalizing a whole sublattice
        block instead is variationally unstable with this diffuse basis --
        the raw AO block lets diffuse cation functions fall into the
        removed side's potential wells (a "Ti 3d GM3+" at O-2p depth with
        28% Loewdin weight on O: a disguised anion state, caught by the
        user on SrTiO3), while Loewdin- or projection-orthogonalized
        blocks load strongly overlapped shells with huge orthogonalization
        penalties (the O 2s on-site swung from -6.9 to +0.5 to +12.6 eV
        across those three conventions; per-shell Rayleigh quotients have
        no such freedom).  Shell AO sets map onto themselves under the
        space group, so exact multiplets of the group-averaged Fock carry
        over (raw FFT-grid splittings under --no-symmetrize).
        """
        fock = self.fock["mo"]
        chunks = []
        dropped = 0
        for spec in self.side_specs[column]:
            rows = np.asarray(self.spec_indices[id(spec)])
            overlap = S[np.ix_(rows, rows)]
            values, vectors = np.linalg.eigh(overlap)
            keep = values > ONSITE_OVERLAP_FLOOR
            dropped += int(values.size - int(keep.sum()))
            basis = vectors[:, keep] / np.sqrt(values[keep])
            energies, mixing = np.linalg.eigh(
                basis.conj().T @ fock[np.ix_(rows, rows)] @ basis)
            states = basis @ mixing
            full = np.zeros((self.n_ao, states.shape[1]), dtype=complex)
            full[rows] = states
            chunks.append((energies.real, full))
        energies = np.concatenate([e for e, _ in chunks])
        coefficients = np.hstack([c for _, c in chunks])
        order = np.argsort(energies)
        return energies[order], coefficients[:, order], dropped

    def overlap_at(self, kpoint) -> np.ndarray:
        cell = self.cells["mo"]
        kpts_band = cell.get_abs_kpts(np.array([kpoint], dtype=float))
        return np.asarray(cell.pbc_intor("int1e_ovlp", hermi=1, kpts=kpts_band))[0]

    @staticmethod
    def _sqrt_overlap(overlap) -> np.ndarray:
        """S^(1/2), for Loewdin populations (|coefficient|^2 in the
        symmetrically orthogonalized basis -- the orthonormal set closest to
        the atomic orbitals, i.e. the site-bound attribution)."""
        values, vectors = np.linalg.eigh(overlap)
        return (vectors * np.sqrt(np.clip(values, 0.0, None))) @ vectors.conj().T

    # -------------------------------------------------------------- symmetry

    def little_group_data(self, kpoint):
        """spgrep irreps, physical labels, and the AO representation at k.

        Same construction as the extended-Hueckel engine, but written directly
        in PySCF's AO ordering: D[(a', shell, m'), (a, shell, m)] =
        P[a', a] W^l[m', m].
        """
        from spgrep.core import get_spacegroup_irreps_from_primitive_symmetry

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
        # crystod builds the permutation representation in the *periodic* gauge,
        # chi_mu k = sum_T exp(i k (T + tau_mu)) phi_mu(r - tau_mu - T), while
        # PySCF puts the phase on the lattice translation alone (the atomic
        # gauge).  The two bases differ by Lambda_mu = exp(2 pi i k . tau_mu),
        # so the representation must be conjugated with it.  The difference is
        # invisible whenever k . tau is a multiple of 1/2 for every site (ScF3,
        # SrTiO3) and essential when it is not (the 1/4, 3/4 sites of fluorite).
        slices = self.cells["mo"].aoslice_by_atom()
        site_phase = np.ones(self.n_ao, dtype=complex)
        for atom in range(len(self.symbols)):
            value = np.exp(2j * np.pi * float(np.dot(kpoint, self.positions[atom])))
            site_phase[int(slices[atom, 2]):int(slices[atom, 3])] = value

        blocks = self.ao_blocks
        representation = []
        for op_index, op in enumerate(mapping):
            rotation = np.real(self.builder.rotations_cartesian[op])
            permutation = permutations[op_index]
            wigners: dict[int, np.ndarray] = {}
            matrix = np.zeros((self.n_ao, self.n_ao), dtype=complex)
            for block in blocks:
                if block.l not in wigners:
                    wigners[block.l] = wigner_pyscf(block.l, rotation)
                wigner = wigners[block.l]
                atom = block.sites[0]
                for image in blocks:
                    if image.l != block.l:
                        continue
                    if self.ao_offset_in_atom[image.offset] != self.ao_offset_in_atom[block.offset]:
                        continue
                    phase = permutation[image.sites[0], atom]
                    if abs(phase) < 1e-12:
                        continue
                    matrix[image.offset:image.offset + image.n_ao,
                           block.offset:block.offset + block.n_ao] = phase * wigner
            representation.append(site_phase[:, None] * matrix / site_phase[None, :])
        return irreps, mapping, labels, representation

    def site_symmetry_irreps(self, kpoint, representation, irreps, labels):
        """Irrep content of every (element, shell) block at k -- the
        site-symmetry induced representation of ``crystod --element/--orbital``,
        recomputed here from the very representation used for the labelling."""
        from .runtime_compat import get_character

        order = len(representation)
        content = {}
        for spec in self.specs:
            indices = self.spec_indices[id(spec)]
            characters = np.array([
                np.trace(D[np.ix_(indices, indices)]) for D in representation
            ])
            parts = []
            for irrep, label in zip(irreps, labels):
                chi = np.array(get_character(irrep), dtype=complex)
                multiplicity = float(np.real(np.sum(characters * np.conj(chi)) / order))
                count = int(round(multiplicity))
                if count > 0:
                    name = label.split("(")[0]
                    parts.append(name if count == 1 else f"{count}{name}")
            content[(spec.element, spec.shell)] = parts
        return content

    # --------------------------------------------------------------- solving

    def _group_levels(self, energies, vectors):
        """Cluster eigenvalues into degenerate groups (eV, grid-noise aware)."""
        groups = []
        start = 0
        for index in range(1, len(energies) + 1):
            if (index == len(energies)
                    or energies[index] - energies[start] > self.degeneracy_tol):
                groups.append((float(np.mean(energies[start:index])),
                               vectors[:, start:index]))
                start = index
        return groups

    def _multiplicities(self, space, overlap, representation, irreps):
        """Irrep content of an eigenspace (may be fractional if it is not a
        whole number of complete multiplets)."""
        from .runtime_compat import get_character

        gram = space.conj().T @ overlap @ space
        values, basis = np.linalg.eigh(gram)
        keep = values > 1e-10
        orthonormal = space @ (basis[:, keep] / np.sqrt(values[keep]))
        characters = np.array([
            np.trace(orthonormal.conj().T @ overlap @ D @ orthonormal)
            for D in representation
        ])
        order = len(representation)
        return [
            float(np.real(np.sum(characters * np.conj(
                np.array(get_character(irrep), dtype=complex))) / order))
            for irrep in irreps
        ]

    def _is_complete_multiplet(self, space, overlap, representation, irreps) -> bool:
        multiplicities = self._multiplicities(space, overlap, representation, irreps)
        if not any(value > 0.5 for value in multiplicities):
            return False
        return all(abs(value - round(value)) < _MULTIPLICITY_TOL
                   for value in multiplicities)

    def _adaptive_groups(self, energies, vectors, overlap, representation, irreps):
        """Cluster levels into complete multiplets.

        The seed window only separates obviously distinct levels; adjacent
        seeds are then merged until the irrep multiplicities of the group are
        integral, which is exactly the condition that no multiplet is cut in
        half.  Grid noise splits symmetry-degenerate levels by anything from
        1e-4 to several 1e-2 eV depending on the system and the cutoff, so this
        is far more robust than any fixed tolerance -- and the merging can
        never run away, because it stops at DEGENERACY_MAX_WINDOW_EV.
        """
        seeds = self._group_levels(energies, vectors)
        groups = []
        index = 0
        while index < len(seeds):
            start_energy = seeds[index][0]
            space = seeds[index][1]
            weights = [space.shape[1]]
            centres = [seeds[index][0]]
            last = index
            while not self._is_complete_multiplet(space, overlap, representation, irreps):
                if (last + 1 >= len(seeds)
                        or seeds[last + 1][0] - start_energy > self.degeneracy_window):
                    break
                last += 1
                space = np.hstack([space, seeds[last][1]])
                weights.append(seeds[last][1].shape[1])
                centres.append(seeds[last][0])
            energy = float(np.average(centres, weights=weights))
            groups.append((energy, space))
            index = last + 1
        return groups

    def align_fragment_columns(self, records):
        """Deep-level (XPS-style) alignment of the three energy columns.

        Each calculation carries its own G = 0 average-potential reference, so
        the raw columns are offset by one rigid constant each.  For every
        fragment column the deepest *chemically inert* level is located -- the
        deepest fragment level some crystal level consists of to at least
        ALIGNMENT_PURITY -- and its fragment -> crystal energy difference,
        averaged over the k points where the pair exists, is that column's
        offset.  The zero is then put at the deeper of the two anchors in its
        PRE-BONDING (fragment) value: that fragment column stays, the crystal
        column moves by -delta_ref, the other fragment column by
        delta_other - delta_ref.

        ``records`` is the per-k-point list built by report_and_write; the
        level energies are shifted in place.  Returns (shifts, anchors).
        """
        anchors = {}
        for column in ("left", "right"):
            pairs = []  # (fragment_energy, delta, fragment_label, k_name, purity)
            for record in records:
                fragment_levels = {
                    level.level_id: level for level in record["levels"][column]
                }
                best: dict[str, tuple[float, object]] = {}
                for crystal in record["levels"]["mo"]:
                    # absolute projections: the renormalized composition can
                    # show ~100% for a level whose true overlap with the
                    # retained fragment states is tiny
                    weights = getattr(crystal, "absolute_composition",
                                      crystal.composition)
                    for level_id, weight in weights:
                        if not level_id.startswith(column):
                            continue
                        if weight > best.get(level_id, (0.0, None))[0]:
                            best[level_id] = (weight, crystal)
                for level_id, (purity, crystal) in best.items():
                    fragment = fragment_levels[level_id]
                    pairs.append((fragment.energy, crystal.energy - fragment.energy,
                                  fragment.label, record["name"], purity))
            inert = [pair for pair in pairs if pair[4] >= ALIGNMENT_PURITY]
            pool = inert or pairs
            if not pool:
                anchors[column] = None
                continue
            deepest = min(pool, key=lambda pair: pair[0])
            key = tuple(deepest[2].split()[:2])  # (element, shell), any k
            # the same shell across the k points -- but only levels of the same
            # band (within 1 eV of the anchor), since one shell can span several
            # irreps of quite different chemistry at a single k point
            same = [pair for pair in pool
                    if tuple(pair[2].split()[:2]) == key
                    and abs(pair[0] - deepest[0]) < 1.0]
            # the reference offset is one constant, but the anchor shell may
            # hybridize more at some k than at others; trust only the pairs
            # within 2% of the best purity, where the chemistry is smallest
            best_purity = max(pair[4] for pair in same)
            trusted = [pair for pair in same if pair[4] >= best_purity - 0.02]
            values = [pair[1] for pair in trusted]
            purest = max(trusted, key=lambda pair: pair[4])
            anchors[column] = {
                "label": purest[2],
                "fragment_energy": purest[0],
                "delta": float(np.mean(values)),
                "spread": float(np.max(values) - np.min(values)) if len(values) > 1 else 0.0,
                "n_k": len(trusted),
                "purity": best_purity,
                "fallback": not inert,
            }

        available = [c for c in ("left", "right") if anchors.get(c)]
        if not available:
            return None, anchors
        reference = min(available, key=lambda c: anchors[c]["fragment_energy"])
        delta_ref = anchors[reference]["delta"]
        shifts = {reference: 0.0, "mo": -delta_ref}
        other = {"left": "right", "right": "left"}[reference]
        if anchors.get(other):
            shifts[other] = anchors[other]["delta"] - delta_ref
        else:
            shifts[other] = -delta_ref  # no anchor: move with the crystal
        shifts["reference"] = reference

        for record in records:
            for column in ("left", "right", "mo"):
                for level in record["levels"][column]:
                    level.energy += shifts[column]
        return shifts, anchors

    # ---------------------------------------------------- wave-function sketch

    def sketch_partners(self, level, kpoint, cells):
        """Per-partner lobes on the supercell atoms, from the PySCF AO
        coefficients -- same entry format as the extended-Hueckel sketch:
        [atom, s, px, py, pz, dxy, dyz, dz2, dxz, dx2-y2].

        The PySCF eigenvectors are in the atomic Bloch gauge, so the
        cell-to-cell phase is exp(2 pi i k . T) without the site offset.
        Same-l shells of one atom accumulate with their contracted-GTO radial
        amplitude at r0 = 2 bohr (see _build_ao_blocks), which fixes the lobe
        signs and orientation; the lobe SIZE is then rescaled to the Loewdin
        population of that (atom, l) channel.  Raw r0 amplitudes would
        misstate the sizes: a diffuse gth-dzvp Sc 4p is ~5x an F 2p at r0, so
        a 7%-population Sc admixture used to draw at 83% of the largest F
        lobe.  f shells and higher are omitted from the drawing.
        """
        from .visualize_basis import realify_basis_space

        rows, _ = realify_basis_space(level.vectors.T)
        rows = np.asarray(rows)
        if self.projection == "mulliken":
            overlap = getattr(level, "overlap", None)
            if overlap is None:
                overlap = self.overlap_at(kpoint)
        else:
            overlap = getattr(level, "sqrt_overlap", None)
            if overlap is None:
                overlap = self._sqrt_overlap(self.overlap_at(kpoint))
        n_prim = len(self.symbols)
        width = 9
        slot_of = {0: 0, 1: 1, 2: 4}
        angular = {0: 0.28209479, 1: 0.48860251, 2: 0.63078313}
        # a fragment level is drawn from its own sublattice's components only:
        # amplitude on the ghost basis of the removed sublattice is the
        # variational tail toward the point charges (symmetry-allowed BSSE
        # borrowing), and drawing it as an atom-centred orbital on the empty
        # site misreads it; the ghost fraction is reported numerically instead
        column = getattr(level, "column", "mo")
        blocks = [b for b in self.ao_blocks
                  if column == "mo" or b.column == column]
        raw_partners = []
        channel_pop = np.zeros((n_prim, width))    # multiplet-summed Loewdin
        channel_amp2 = np.zeros((n_prim, width))   # multiplet-summed |amp|^2
        for vector in rows:
            v = np.asarray(vector, dtype=complex)
            # per-AO population of this partner in the selected projection
            # (--projection): Loewdin |S^(1/2)c|^2 -- squared coefficients in
            # the symmetrically orthogonalized basis, non-negative and summing
            # to 1 -- or Mulliken gross populations Re[c* (S c)], whose
            # overlap cross terms can go negative on diffuse empty levels
            if self.projection == "mulliken":
                gross = (v.conj() * (overlap @ v)).real
            else:
                gross = np.abs(overlap @ v) ** 2
            amp_re = np.zeros((len(cells) * n_prim, width))
            amp_im = np.zeros_like(amp_re)
            for block in blocks:
                if block.l not in slot_of:
                    continue
                slot = slot_of[block.l]
                site = block.sites[0]
                channel_pop[site, slot] += float(
                    gross[block.offset:block.offset + block.n_ao].sum())
                coefficients = np.asarray(
                    vector[block.offset:block.offset + block.n_ao]
                ) * (angular[block.l] * block.radial)
                for cell_index, cell_vector in enumerate(cells):
                    phase = np.exp(2j * np.pi * float(np.dot(kpoint, cell_vector)))
                    values = coefficients * phase
                    row_index = cell_index * n_prim + site
                    amp_re[row_index, slot:slot + block.n_ao] += values.real
                    amp_im[row_index, slot:slot + block.n_ao] += values.imag
            for site in range(n_prim):
                for l, slot in slot_of.items():
                    n_m = 2 * l + 1
                    channel = (amp_re[site, slot:slot + n_m]
                               + 1j * amp_im[site, slot:slot + n_m])
                    channel_amp2[site, slot] += float(np.linalg.norm(channel)) ** 2
            raw_partners.append((amp_re, amp_im))
        # one calibration factor per (atom, l) channel, from the MULTIPLET
        # sums: within a degenerate multiplet the population/amplitude^2 ratio
        # of a channel is partner-independent by symmetry, so a shared factor
        # keeps symmetry-equivalent atoms exactly equal and preserves the
        # sigma/pi contrast between partners, while lobe areas become
        # proportional to the Loewdin electron weight on the site instead of
        # the diffuse-function amplitude at r0 (a gth Sc 4p is ~5x an F 2p
        # there; per-partner rescaling instead broke the F-site equivalence
        # slightly through the RREF canonicalization below)
        col_factor = np.zeros((n_prim, width))
        for site in range(n_prim):
            for l, slot in slot_of.items():
                n_m = 2 * l + 1
                amp2 = channel_amp2[site, slot]
                if amp2 > 1e-24:
                    col_factor[site, slot:slot + n_m] = np.sqrt(
                        max(channel_pop[site, slot], 0.0) / amp2)
        tiled = np.tile(col_factor, (len(cells), 1))
        partner_rows = []
        for amp_re, amp_im in raw_partners:
            amp_re = amp_re * tiled
            amp_im = amp_im * tiled
            choice = (amp_re if np.linalg.norm(amp_re) >= np.linalg.norm(amp_im)
                      else amp_im)
            partner_rows.append(choice.reshape(-1))
        rows_arr = np.array(partner_rows)
        if len(partner_rows) > 1:
            from .molecular_salc import _rref_orthogonal

            work = rows_arr.copy()
            peak = np.max(np.abs(work)) or 1.0
            work[:, np.max(np.abs(work), axis=0) < 0.05 * peak] = 0.0
            canonical = _rref_orthogonal(list(work))
            if len(canonical) == len(partner_rows):
                rows_arr = np.array(canonical, dtype=float)
        partners = []
        for row in rows_arr:
            grid = row.reshape(len(cells) * n_prim, width)
            peak = np.max(np.abs(grid)) or 1.0
            entries = []
            for atom_index in range(grid.shape[0]):
                values = grid[atom_index] / peak
                if np.max(np.abs(values)) >= 0.04:
                    entries.append(
                        [atom_index] + [round(float(x), 3) for x in values]
                    )
            partners.append(entries)
        return [p for p in partners if p] or None

    def _symmetrize_fock(self, fock, overlap, representation):
        """Group-average an operator so its eigenspaces carry complete irreps
        EXACTLY.  The raw SCF has no point-group constraint -- the FFT grid
        (and, much worse, grid-evaluated exact exchange of hybrid functionals)
        breaks degeneracies by up to a few 0.1 eV -- so the displayed levels
        are re-diagonalized from F_avg = (1/|G|) sum_g D(g)+ F D(g), which is
        invariant under the verified AO representation by construction.
        Returns (energies, coefficients, max |F_avg - F|)."""
        average = sum(D.conj().T @ fock @ D for D in representation)
        average = average / len(representation)
        average = 0.5 * (average + average.conj().T)
        deviation = float(np.max(np.abs(average - fock)))
        s_values, s_vectors = np.linalg.eigh(overlap)
        keep = s_values > 1e-9 * float(s_values.max())
        basis = s_vectors[:, keep] / np.sqrt(s_values[keep])
        reduced = basis.conj().T @ average @ basis
        reduced = 0.5 * (reduced + reduced.conj().T)
        energies, rotation = np.linalg.eigh(reduced)
        return energies, basis @ rotation, deviation

    def _symmetrized_bands(self, column, energies, coefficients, overlap,
                           representation):
        """Re-derive one calculation's levels from the group-averaged Fock.

        With --no-ghost the fragment spans only its own AO block, so the
        averaging runs in that subspace (the space-group operations never mix
        the sublattices, hence D is block-diagonal in them).
        """
        if self.no_ghost and column != "mo":
            rows = self.embed_rows[column]
            sub = np.ix_(rows, rows)
            small_overlap = overlap[sub]
            small_representation = [D[sub] for D in representation]
            small_c = coefficients[rows]
            fock = (small_overlap @ small_c @ np.diag(energies)
                    @ small_c.conj().T @ small_overlap)
            e, c, deviation = self._symmetrize_fock(
                fock, small_overlap, small_representation)
            return e, self._embed(column, c), deviation
        fock = (overlap @ coefficients @ np.diag(energies)
                @ coefficients.conj().T @ overlap)
        return self._symmetrize_fock(fock, overlap, representation)

    def _ghost_fraction(self, column, space, overlap) -> float:
        """Mean Mulliken population of a level space on the ghost basis of the
        *other* sublattice -- a counterpoise artifact when it dominates."""
        other = {"left": "right", "right": "left"}[column]
        slices = self.cells["mo"].aoslice_by_atom()
        rows: list[int] = []
        for atom in self.side_atoms[other]:
            rows.extend(range(int(slices[atom, 2]), int(slices[atom, 3])))
        if not rows:
            return 0.0
        rows = np.array(rows, dtype=int)
        gross = (space.conj() * (overlap @ space)).real
        total = np.clip(gross.sum(axis=0), 1e-12, None)
        return float(np.mean(gross[rows, :].sum(axis=0) / total))

    def _dominant_spec(self, space, S, column):
        """(element, shell) block with the largest Mulliken population."""
        weighted = S @ space
        best, best_weight = None, -np.inf
        for spec in self.side_specs[column]:
            indices = self.spec_indices[id(spec)]
            weight = float(np.real(np.sum(np.conj(space[indices]) * weighted[indices])))
            if weight > best_weight:
                best, best_weight = spec, weight
        return best

    def solve_at(self, kpoint):
        """All fragment and crystal levels at one k point."""
        S = self.overlap_at(kpoint)
        S_half = self._sqrt_overlap(S)
        irreps, mapping, labels, representation = self.little_group_data(kpoint)

        worst = max(float(np.max(np.abs(D.conj().T @ S @ D - S))) for D in representation)
        if worst > 1e-6:
            raise SystemExit(
                "ERROR: the AO representation does not leave the PySCF overlap invariant "
                f"(residual {worst:.2e}); please report this case."
            )
        self.last_gauge_residual = worst
        self.last_dropped = 0

        def strip(label):
            return label.split("(")[0]

        levels = {"left": [], "mo": [], "right": []}
        self.fock = {}
        self.last_symbreak = {}
        # --onsite solves the crystal first: the fragment columns are the
        # sublattice blocks of its Fock operator and need it in hand
        order = (("mo", "left", "right") if self.onsite
                 else ("left", "right", "mo"))
        for column in order:
            if self.onsite and column != "mo":
                energies, coefficients, dropped = self._block_bands(column, S)
                self.last_dropped += dropped
            else:
                energies, coefficients = self._bands_at(column, kpoint)
                if self.symmetrize:
                    energies, coefficients, deviation = self._symmetrized_bands(
                        column, energies, coefficients, S, representation)
                    self.last_symbreak[column] = deviation
            if column == "mo":
                # F(k) = S C E C+ S, exact for the eigenvectors of F with metric S
                self.fock["mo"] = (
                    S @ coefficients @ np.diag(energies) @ coefficients.conj().T @ S
                )
            counts: dict[str, int] = {}
            for energy, group in self._adaptive_groups(
                    energies, coefficients, S, representation, irreps):
                if (column != "mo"
                        and self._ghost_fraction(column, group, S)
                        > GHOST_FRACTION_THRESHOLD):
                    continue
                for irrep_label, space in self._irrep_split(
                    group, S, representation, irreps, labels
                ):
                    name = strip(irrep_label)
                    detail = ""
                    ghost_fraction = 0.0
                    if column == "mo":
                        # "GM4- #2" = second GM4- multiplet from the bottom;
                        # "GM4-(2)" read like a degeneracy count
                        counts[name] = counts.get(name, 0) + 1
                        label = f"{name} #{counts[name]}"
                    else:
                        spec = self._dominant_spec(space, S, column)
                        label = f"{spec.element} {spec.shell} {name}"
                        if not self.no_ghost:
                            ghost_fraction = self._ghost_fraction(column, space, S)
                            if ghost_fraction >= 0.005:
                                detail = ("weight on the removed sublattice's "
                                          "(ghost) basis: "
                                          f"{100 * ghost_fraction:.0f}% "
                                          "(variational tail toward the point "
                                          "charges; not drawn in the sketch)")
                    level = DiagramLevel(
                        level_id=f"{column}{len(levels[column])}",
                        column=column,
                        energy=float(energy),
                        degeneracy=space.shape[1],
                        irrep=name,
                        label=label,
                        vectors=space,
                        detail=detail,
                    )
                    level.ghost_fraction = ghost_fraction
                    # shared references, for the population-scaled sketches
                    level.overlap = S
                    level.sqrt_overlap = S_half
                    levels[column].append(level)
            if column != "mo":
                # two fragment levels can share (element, shell, irrep) -- e.g.
                # the two F 2p GM4- combinations; number them so the crystal
                # compositions stay readable
                seen: dict[str, int] = {}
                for level in levels[column]:
                    seen[level.label] = seen.get(level.label, 0) + 1
                repeated = {label for label, n in seen.items() if n > 1}
                occurrence: dict[str, int] = {}
                for level in levels[column]:
                    if level.label in repeated:
                        occurrence[level.label] = occurrence.get(level.label, 0) + 1
                        level.label = f"{level.label}#{occurrence[level.label]}"

        # fragment projection (level.composition): crystal levels in the
        # Loewdin-orthogonalized fragment-level basis.  Internal only -- it
        # positions the connector lines and feeds the alignment anchors
        # (absolute_composition keeps the unnormalized weights) but is NOT
        # displayed as percentages: the fragment eigenstates mix AO shells
        # among themselves (the "Sc 3d R5+" fragment level carries 4d AO
        # character), so its weights disagree with the AO populations below
        # and showing both confused more than it explained.
        assign_fragment_compositions(levels, S)
        for crystal in levels["mo"]:
            # the ONE displayed composition (panel bars, hover tooltip and
            # the terminal all quote this list): per-(element, shell) AO
            # populations of the PySCF eigenvector in the selected projection
            # (--projection lowdin/mulliken; Loewdin sums to exactly 100%
            # with non-negative entries) -- the same partial-charge measure
            # as `crystod --dos --pyscf`.  Symmetry does the orbital
            # selection: a state of irrep G only picks up the G-adapted
            # combination of each shell (forbidden shells project to ~0), so
            # every entry is labeled with the crystal irrep.
            shares = self._population_shares(
                crystal.vectors, crystal.degeneracy, S, S_half)
            crystal.display_composition = [
                (f"{name} {crystal.irrep}", value) for value, name in shares
            ]
            populations = ", ".join(f"{name} {100 * value:.1f}%"
                                    for value, name in shares)
            crystal.detail = f"{self.projection_label}: {populations}"
        # the fragment/onsite columns carry the same displayed composition,
        # but list ONLY their own sublattice's shells -- a fragment level is
        # a sublattice state, and per-shell entries of the other element
        # read as contamination ("O states inside a Ti-only level").  The
        # weight that does sit beyond the own shells (the counterpoise
        # ghost tail in the fragment-SCF mode, and in every mode the
        # Loewdin attribution of the overlap density) is aggregated into
        # one closing note instead.
        for column in ("left", "right"):
            own = self.side_specs[column]
            for level in levels[column]:
                shares = self._population_shares(
                    level.vectors, level.degeneracy, S, S_half, specs=own)
                level.display_composition = [
                    (f"{name} {level.irrep}", value) for value, name in shares
                ]
                row = self.projection_label + ": " + ", ".join(
                    f"{name} {100 * value:.1f}%" for value, name in shares)
                remainder = 1.0 - sum(value for value, _ in shares)
                if remainder >= 0.005:
                    if self.onsite:
                        row += (f" (+{100 * remainder:.1f}% "
                                f"{self.projection_label}-attributed to the "
                                "other sublattice's basis: overlap density; "
                                "the state has no coefficients there)")
                    else:
                        row += (f" (+{100 * remainder:.1f}% on the removed "
                                "sublattice's basis: ghost tail + "
                                f"{self.projection_label} attribution)")
                level.detail = (row if not level.detail
                                else f"{row}\n{level.detail}")

        # left-right coupling <phi_left| F(k) |phi_right> of the crystal Fock
        # operator: same irrep and a large matrix element compared with the
        # level separation is what actually splits bonding from antibonding
        self.last_coupling = []
        fock = self.fock["mo"]
        for left in levels["left"]:
            for right in levels["right"]:
                if left.irrep != right.irrep:
                    continue
                # the fragment states are NOT mutually orthogonal, so the bare
                # matrix element h = <phi_L|F|phi_R> carries an
                # overlap-times-mean-energy part s*(e_L+e_R)/2 that mimics a
                # coupling even between non-interacting states (diffuse F 3s/3d
                # fragment virtuals at +50 eV showed |h| of 5-15 eV against
                # semicore levels this way).  The reported strength is the
                # first-order Loewdin-orthogonalized coupling
                #   H~ = h - s (e_L + e_R)/2
                # with e_X = <phi_X|F|phi_X> the crystal-Fock expectations --
                # invariant under the G=0 reference (F -> F + V S shifts h by
                # V s and both e_X by V).
                overlap_block = left.vectors.conj().T @ S @ right.vectors
                raw_block = left.vectors.conj().T @ fock @ right.vectors
                mean_energy = 0.5 * (
                    float(np.trace(left.vectors.conj().T @ fock
                                   @ left.vectors).real) / left.degeneracy
                    + float(np.trace(right.vectors.conj().T @ fock
                                     @ right.vectors).real) / right.degeneracy)
                block = raw_block - mean_energy * overlap_block
                strength = float(np.sqrt(np.sum(np.abs(block) ** 2) / left.degeneracy))
                raw_strength = float(np.sqrt(
                    np.sum(np.abs(raw_block) ** 2) / left.degeneracy))
                overlap_norm = float(np.sqrt(
                    np.sum(np.abs(overlap_block) ** 2) / left.degeneracy))
                gap = abs(left.energy - right.energy)
                # two-level mixing: tan(2 theta) = 2|H| / dE, so the minority
                # weight of each mixed orbital is sin^2(theta)
                angle = 0.5 * np.arctan2(2.0 * strength, gap)
                self.last_coupling.append(
                    (left, right, strength, gap, float(np.sin(angle) ** 2),
                     overlap_norm, raw_strength)
                )
        self.last_coupling.sort(key=lambda item: -item[4])

        # COOP bonding character of every crystal level (see
        # assign_bond_characters); the occupations must be filled first
        self._fill(levels["mo"], self.electrons)
        for column in ("left", "right"):
            self._fill(levels[column], self.side_electrons[column])
        spec_ranges = {(spec.element, spec.shell): self.spec_indices[id(spec)]
                       for spec in self.specs}
        assign_bond_characters(
            levels, S, self.embed_rows["left"], self.embed_rows["right"],
            spec_ranges, sqrt_overlap=S_half, hamiltonian=fock,
        )

        return levels, labels


# --------------------------------------------------------------------- report


def describe_chk(path: str) -> None:
    """Print the calculation conditions stored in a --chk checkpoint.

    The file is a compressed npz (binary for size and load speed); this is
    the human-readable window into it: the defining parameters, what it
    holds, and a ready-to-paste option string that reproduces them
    (``crystod --chk-info FILE``).
    """
    import json
    import os

    if not os.path.exists(path):
        raise SystemExit(f"ERROR: {path} does not exist.")
    try:
        data = np.load(path, allow_pickle=False)
        params = json.loads(str(data["params"]))
    except Exception as error:
        raise SystemExit(
            f"ERROR: {path} is not a CrystOD --chk checkpoint ({error}).")

    stored = ([str(c) for c in data["energy_columns"]]
              if "energy_columns" in data.files
              else ["mo", "left", "right"])
    energies = np.asarray(data["energies"], dtype=float)
    lattice = np.asarray(data["lattice"], dtype=float)
    lengths = np.linalg.norm(lattice, axis=1)
    n_kpts, n_ao = np.asarray(data["dm_mo"]).shape[:2]
    smeared = sorted(str(s) for s in data["smeared"])
    kind = ("full three-SCF run"
            if all(f"dm_{c}" in data.files for c in ("mo", "left", "right"))
            else "crystal density only (--onsite run)")
    oxidation = " ".join(f"{el}={q:+g}"
                         for el, q in sorted(params["oxidation"].items()))

    size = os.path.getsize(path) / 1e6
    print(f" * {path} -- CrystOD SCF checkpoint "
          f"(WAVECAR-style restart, {size:.1f} MB) *")
    print(f"   structure : {params['left']} + {params['right']}, "
          f"{len(params['symbols'])} atoms/cell, "
          f"a = {lengths[0]:.4f} / {lengths[1]:.4f} / {lengths[2]:.4f} A")
    print(f"   method    : {params['xc'].upper()} / {params['basis']} / "
          f"{params['pseudo']}, ke_cutoff {params['ke_cutoff']:g} Ha, "
          f"k-mesh {'x'.join(map(str, params['kmesh']))}"
          + (f", max_l {params['max_l']}"
             if params.get("max_l") is not None else "")
          + (f", smearing sigma {params['sigma']:g} eV"
             if params.get("sigma") else ""))
    print(f"   electrons : {params['electrons']:g} per cell "
          f"(oxidation {oxidation})")
    print(f"   fragments : left {params['left']} | right {params['right']}"
          + (", own-sublattice basis (--no-ghost)"
             if params.get("no_ghost") else ", counterpoise ghosts"))
    print(f"   contents  : {kind}; densities on {n_kpts} k points x "
          f"{n_ao} AOs")
    print("   energies  : " + "  |  ".join(
        f"{column} {energies[index]:.8f} Ha"
        for index, column in enumerate(stored)))
    if smeared:
        print(f"   smearing  : {', '.join(smeared)} carried Fermi smearing "
              "when written")
    reuse = (f"--co-left {params['left']} --co-right {params['right']} "
             f"--xc {params['xc']} --basis {params['basis']} "
             f"--pseudo {params['pseudo']} "
             f"--kmesh {' '.join(map(str, params['kmesh']))} "
             f"--ke-cutoff {params['ke_cutoff']:g}"
             + (f" --max-l {params['max_l']}"
                if params.get("max_l") is not None else "")
             + (" --no-ghost" if params.get("no_ghost") else "")
             + (f" --sigma {params['sigma']:g}"
                if params.get("sigma") else "")
             + f" --oxidation {oxidation}"
             # a crystal-only checkpoint can only feed --onsite runs
             + ("" if kind.startswith("full") else " --onsite"))
    print(f"   reuse with: {reuse} --chk {path}")


def report_and_write(cell, *, left, right, symprec, electrons, kpoint_filter,
                     output_path, structure_label, oxidation=None, basis=None,
                     pseudo=None, xc="pbe", kmesh=None, ke_cutoff=200.0,
                     sigma=0.0, degeneracy_tol=None, align=True, no_ghost=False,
                     symmetrize=True, max_l=None, projection="lowdin",
                     chk=None, onsite=False, verbose=0):
    """Terminal report + HTML for the PySCF crystal-orbital diagram."""
    diagram = PySCFCrystalOrbitalDiagram(
        cell, left, right, symprec=symprec, electrons=electrons,
        oxidation=oxidation, basis=basis or "gth-dzvp-molopt-sr",
        pseudo=pseudo or "gth-pbe", xc=xc, kmesh=kmesh, ke_cutoff=ke_cutoff, sigma=sigma,
        degeneracy_tol=degeneracy_tol, no_ghost=no_ghost, symmetrize=symmetrize,
        max_l=max_l, projection=projection, chk=chk, onsite=onsite,
        verbose=verbose,
    )
    dataset = diagram.builder.spglib_dataset
    print("\n * Space group *")
    print(f" {dataset['international']} ({dataset['number']})\n")

    # ---- stage 1: the valence orbitals the pseudopotential leaves ----------
    print(" * Valence basis (pseudopotential valence shells + polarization) *")
    for column in ("left", "right"):
        for element in dict.fromkeys(
            spec.element for spec in diagram.side_specs[column]
        ):
            shells = [spec.shell for spec in diagram.side_specs[column]
                      if spec.element == element]
            sites = len({site for spec in diagram.side_specs[column]
                         if spec.element == element for site in spec.sites})
            print(f" {element:<3} {' '.join(shells):<32} x{sites} site(s)")
    print(f" basis {diagram.basis_name} / pseudo {diagram.pseudo_name} / "
          f"functional {diagram.xc.upper()}")
    print(f" AO space shared by all three calculations: {diagram.n_ao} orbitals\n")

    # ---- the fragments ----------------------------------------------------
    if diagram.onsite:
        print(" * Fragment columns (--onsite): sublattice blocks of the "
              "crystal Fock *")
        for column in ("left", "right"):
            print(f" {column:<5} {diagram.formula[column]:<8} "
                  f"{diagram.side_electrons[column]} electrons (formal "
                  "count), levels = per-shell on-site multiplets "
                  "<phi|F(k)|phi> of its own shells")
        print(f" crystal {diagram.formula['left'] + diagram.formula['right']:<7} "
              f"{diagram.crystal_electrons} electrons "
              f"= {diagram.side_electrons['left']} + "
              f"{diagram.side_electrons['right']}")
        print(" ONE Hamiltonian for every column (no fragment SCF, no point\n"
              " charges, no reference alignment): a crystal level's rise or\n"
              " drop against its parents is purely the left-right orbital\n"
              " interaction\n")
    else:
        print(" * Fragments (formal-charge ions + ghost basis of the removed "
              "sublattice) *")
        other = {"left": "right", "right": "left"}
        for column in ("left", "right"):
            felt = " + ".join(
                f"{element}^{diagram.oxidation[element]:+g}"
                for element in dict.fromkeys(
                    diagram.symbols[i]
                    for i in diagram.side_atoms[other[column]]
                )
            )
            print(f" {column:<5} {diagram.formula[column]:<8} charge "
                  f"{diagram.side_charge[column]:+g}, "
                  f"{diagram.side_electrons[column]} electrons, in the {felt} "
                  "point-charge lattice")
        print(f" crystal {diagram.formula['left'] + diagram.formula['right']:<7} "
              f"{diagram.crystal_electrons} electrons "
              f"= {diagram.side_electrons['left']} + "
              f"{diagram.side_electrons['right']}")
        print(" every cell is neutral (no monopole divergence), but each calculation\n"
              " still pins its own G=0 average potential to zero, so the raw columns\n"
              " are offset by one constant each -- removed below by deep-level alignment")
    if diagram.onsite:
        pass
    elif diagram.no_ghost:
        print(" fragment basis: OWN sublattice only (--no-ghost) -- the removed\n"
              " sublattice's functions are excluded from the variational space,\n"
              " so no fragment wave function can sit on the removed atoms\n")
    else:
        print(" fragment basis: counterpoise (ghost functions of the removed\n"
              " sublattice kept); sketches draw own-sublattice components only,\n"
              " the ghost weight of each level is reported numerically\n")

    from pyscf import lib as _pyscf_lib

    print(f" * Self-consistent calculations ({'x'.join(map(str, diagram.kmesh))} "
          f"k-mesh, ke_cutoff {diagram.ke_cutoff} Hartree, "
          f"{_pyscf_lib.num_threads()} OpenMP threads) *")
    diagram.run()

    kpoints = diagram.special_kpoints()
    if kpoint_filter is not None:
        available = [name for name, _ in kpoints]
        kpoints = [(name, k) for name, k in kpoints if name == kpoint_filter]
        if not kpoints:
            raise SystemExit(
                f"ERROR: k point '{kpoint_filter}' is not a special point of this "
                f"space group (available: {', '.join(available)})."
            )

    diagram.prepare_bands([kpoint for _, kpoint in kpoints])

    # ---- solve every k point first; printing follows the alignment ---------
    records = []
    for name, kpoint in kpoints:
        levels, _ = diagram.solve_at(kpoint)
        irreps, mapping, labels, representation = diagram.little_group_data(kpoint)
        records.append({
            "name": name,
            "kpoint": kpoint,
            "levels": levels,
            "coupling": list(diagram.last_coupling),
            "residual": diagram.last_gauge_residual,
            "symbreak": dict(getattr(diagram, "last_symbreak", {})),
            "content": diagram.site_symmetry_irreps(
                kpoint, representation, irreps, labels),
        })

    # ---- deep-level alignment ----------------------------------------------
    if diagram.onsite:
        align = False
        print("\n * Single-Hamiltonian mode (--onsite): every column is "
              "measured on the\n   crystal Fock's own scale -- no alignment "
              "step needed *")
    if diagram.onsite:
        pass
    elif align:
        shifts, anchors = diagram.align_fragment_columns(records)
        print("\n * Deep-level alignment (pre-bonding reference) *")
        if shifts is None:
            print("   no chemically inert fragment level found; columns left "
                  "on their raw G=0 references")
        else:
            reference = shifts["reference"]
            anchor = anchors[reference]
            print(f"   anchor : {anchor['label']} of the {reference} fragment "
                  f"({anchor['fragment_energy']:.2f} eV, counterpart purity "
                  f"{100 * anchor['purity']:.0f}%, "
                  f"k-spread {anchor['spread']:.3f} eV over {anchor['n_k']} k)")
            print(f"   shifts : left {shifts['left']:+.3f} eV | "
                  f"crystal {shifts['mo']:+.3f} eV | "
                  f"right {shifts['right']:+.3f} eV")
            other = {"left": "right", "right": "left"}[reference]
            if anchors.get(other):
                info = anchors[other]
                print(f"   ({other} anchored via {info['label']}, purity "
                      f"{100 * info['purity']:.0f}%, k-spread {info['spread']:.3f} eV"
                      + ("; WARNING: purity below "
                         f"{100 * ALIGNMENT_PURITY:.0f}%, the anchor mixes and the "
                         "column offset inherits that error" if info["fallback"] else "")
                      + ")")
            print("   each calculation pins its own average (G=0) potential to zero;"
                  " the columns\n   are shifted so this inert deep level keeps its"
                  " pre-bonding energy -- XPS-style")
    else:
        print("\n * Deep-level alignment disabled (--no-align): each column keeps "
              "its own G=0 reference *")

    # ---- per-k-point report -------------------------------------------------
    for record in records:
        name, kpoint, levels = record["name"], record["kpoint"], record["levels"]
        print(f"\n * k point {name} {_format_kpoint(kpoint)} *")
        print(f"   (AO representation verified against the PySCF overlap: "
              f"max |D+SD - S| = {record['residual']:.1e})")
        if record.get("symbreak"):
            parts = ", ".join(f"{col} {1000 * dev:.0f} meV"
                              for col, dev in record["symbreak"].items())
            print("   raw-SCF point-group breaking, removed by Fock "
                  f"group-averaging: {parts}")

        # ---- stage 2: site-symmetry induced irreps -----------------------
        print("   site-symmetry induced representations:")
        for (element, shell), parts in record["content"].items():
            if parts:
                print(f"     {element} {shell:<4} = {' + '.join(parts)}")

        for column in ("left", "right"):
            parts = ", ".join(
                f"{lv.label} ({lv.energy:.2f}"
                + (f", ghost {100 * getattr(lv, 'ghost_fraction', 0.0):.0f}%"
                   if getattr(lv, "ghost_fraction", 0.0) >= 0.05 else "")
                + ")"
                for lv in sorted(levels[column], key=lambda lv: lv.energy)
            )
            print(f"   {diagram.formula[column]:<10}: {parts}")

        print("   crystal   :")
        for lv in sorted(levels["mo"], key=lambda lv: lv.energy):
            # the same AO-population list as the HTML panel and tooltip
            composition = "  ".join(
                f"{label} {100 * w:.1f}%"
                for label, w in getattr(lv, "display_composition", [])
            )
            occupancy = f"{lv.electrons}e" if lv.electrons else "  "
            print(f"     {lv.label:<10} {lv.energy:9.2f} eV  x{lv.degeneracy}"
                  f"  {occupancy:<4} {composition}")

        if record["coupling"]:
            # dE and the mixing fraction on the ALIGNED scale -- the raw
            # separations carried the per-calculation reference offsets
            rescored = []
            for (lv_left, lv_right, strength, _gap, _mix,
                 overlap_norm, raw_strength) in record["coupling"]:
                gap = abs(lv_left.energy - lv_right.energy)
                mixing = float(np.sin(0.5 * np.arctan2(2.0 * strength, gap)) ** 2)
                rescored.append((lv_left, lv_right, strength, gap, mixing,
                                 overlap_norm, raw_strength))
            rescored.sort(key=lambda item: -item[4])
            record["coupling_aligned"] = rescored
            print("   same-irrep couplings, Loewdin-corrected "
                  "|H~| = |<L|F|R> - S (e_L+e_R)/2| "
                  "(mix = sin^2 of the two-level mixing angle):")
            for (lv_left, lv_right, strength, gap, mixing,
                 overlap_norm, _raw) in rescored[:8]:
                print(f"     {lv_left.label:<16} x {lv_right.label:<16} "
                      f"|H~| = {strength:6.2f} eV   |S| = {overlap_norm:5.3f}"
                      f"   dE = {gap:7.2f} eV   mix = {100 * mixing:4.1f}%")

    # the terminal shows only the top-8 couplings per k point; the full list
    # is a result worth keeping, so it is written next to the HTML
    coupling_path = output_path
    if coupling_path.endswith(".html"):
        coupling_path = coupling_path[: -len(".html")]
    coupling_path += "_coupling.txt"
    with open(coupling_path, "w") as handle:
        scale_note = ("energies on the crystal Fock's own scale (--onsite)"
                      if diagram.onsite else "energies on the aligned scale")
        handle.write(
            "# same-irrep couplings between the fragment levels, from the\n"
            f"# converged crystal Fock operator F(k); {scale_note}\n"
            "# |S|    = |<phi_L| S |phi_R>|: the fragment states are NOT\n"
            "#          mutually orthogonal\n"
            "# |H|raw = |<phi_L| F |phi_R>|: carries an unphysical\n"
            "#          overlap-times-mean-energy part |S|*(e_L+e_R)/2\n"
            "# |H~|   = |<phi_L|F|phi_R> - S (e_L+e_R)/2|, e_X = <phi_X|F|phi_X>:\n"
            "#          first-order Loewdin-orthogonalized coupling, the\n"
            "#          resonance integral to reason with\n"
            "# mix = sin^2(theta) with tan(2 theta) = 2|H~| / dE "
            "(two-level mixing fraction)\n"
            f"# columns: irrep  left_level  E_left(eV)  right_level  "
            "E_right(eV)  |S|  |H|raw(eV)  |H~|(eV)  dE(eV)  mix(%)\n")
        for record in records:
            handle.write(f"\n# k point {record['name']} "
                         f"{_format_kpoint(record['kpoint'])}\n")
            for (lv_left, lv_right, strength, gap, mixing,
                 overlap_norm, raw_strength) in record.get(
                    "coupling_aligned", []):
                handle.write(
                    f"{lv_left.irrep:<6} {lv_left.label:<18} "
                    f"{lv_left.energy:9.3f}  {lv_right.label:<18} "
                    f"{lv_right.energy:9.3f}  {overlap_norm:6.3f} "
                    f"{raw_strength:8.3f} {strength:8.3f} {gap:8.3f} "
                    f"{100 * mixing:7.2f}\n")

    entries = [(r["name"], r["kpoint"], r["levels"]) for r in records]
    write_crystal_diagram_html(diagram, entries, output_path, structure_label)
    print(f"\nCrystal-orbital diagram written to {output_path}")
    print(f"Same-irrep couplings written to {coupling_path}")


def main(argv: list[str] | None = None) -> None:
    import argparse
    from pathlib import Path

    from .crystal_orbital_diagram import parse_oxidation_tokens

    parser = argparse.ArgumentParser(
        description="Crystal-orbital diagram from three periodic PySCF calculations."
    )
    parser.add_argument("--poscar", default="POSCAR")
    parser.add_argument("--co-left", nargs="+", required=True, metavar="FORMULA")
    parser.add_argument("--co-right", nargs="+", required=True, metavar="FORMULA")
    parser.add_argument("--oxidation", nargs="+", default=None, metavar="EL=Q")
    parser.add_argument("--kpoint", default=None,
                        help="restrict to one special k point label (e.g. GM)")
    parser.add_argument("--electrons", type=float, default=None)
    parser.add_argument("--basis", default="gth-dzvp-molopt-sr")
    parser.add_argument("--pseudo", default="gth-pbe")
    parser.add_argument("--xc", default="pbe")
    parser.add_argument("--kmesh", type=int, nargs=3, default=None,
                        metavar=("N1", "N2", "N3"))
    parser.add_argument("--ke-cutoff", type=float, default=200.0)
    parser.add_argument("--sigma", type=float, default=0.0,
                        help="Fermi smearing width in eV (0 = integer occupations)")
    parser.add_argument("--no-symmetrize", action="store_true",
                        help="do not re-diagonalize the group-averaged Fock (debug: shows "
                        "the raw grid-broken degeneracies)")
    parser.add_argument("--max-l", type=int, default=None,
                        help="drop basis shells with l above this from every element "
                        "(e.g. 2 removes the f polarization functions)")
    parser.add_argument("--no-ghost", action="store_true",
                        help="exclude the removed sublattice's basis functions from the "
                        "fragment calculations entirely (hard constraint: no fragment "
                        "wave function on the removed atoms; loses counterpoise "
                        "consistency)")
    parser.add_argument("--no-align", action="store_true",
                        help="keep each calculation's own G=0 reference instead of "
                        "the deep-level (XPS-style) column alignment")
    parser.add_argument("--degeneracy-tol", type=float, default=None,
                        help="seed window in eV for clustering degenerate levels "
                        f"(default {DEGENERACY_SEED_EV}; groups are then merged "
                        "until the irrep multiplicities are integral)")
    parser.add_argument("--projection", choices=("lowdin", "mulliken"),
                        default="lowdin",
                        help="population measure for the sketch lobe sizes and "
                        "the per-(element, shell) rows: Loewdin |S^(1/2)c|^2 "
                        "(default; non-negative, sums to 100%%) or Mulliken "
                        "gross populations Re[c*(Sc)]")
    parser.add_argument("--chk", default=None, metavar="FILE",
                        help="WAVECAR-style restart file: written after the "
                        "SCFs if missing, read (skipping all three SCFs) if "
                        "present; parameters are verified before reuse")
    parser.add_argument("--onsite", action="store_true",
                        help="single-Hamiltonian mode: only the crystal SCF "
                        "runs, and the fragment columns are the per-shell "
                        "on-site multiplets <phi|F|phi> of the crystal Fock "
                        "(tight-binding on-site energies, one level per "
                        "induced irrep) -- no point charges, no reference "
                        "alignment; a full-run --chk is reused, only the "
                        "crystal density is read")
    parser.add_argument("--output", default=None)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args(argv)

    from phonopy.interface.calculator import read_crystal_structure

    cell, _ = read_crystal_structure(args.poscar, interface_mode="vasp")
    stem = Path(args.poscar).name
    for extension in (".vasp", ".poscar"):
        if stem.lower().endswith(extension):
            stem = stem[: -len(extension)]
    report_and_write(
        cell,
        left=args.co_left,
        right=args.co_right,
        symprec=args.tolerance,
        electrons=args.electrons,
        kpoint_filter=args.kpoint,
        output_path=args.output or f"CrystOD_{stem}_pyscf.html",
        structure_label=stem,
        oxidation=(parse_oxidation_tokens(args.oxidation) if args.oxidation else None),
        basis=args.basis,
        pseudo=args.pseudo,
        xc=args.xc,
        kmesh=args.kmesh,
        ke_cutoff=args.ke_cutoff,
        sigma=args.sigma,
        degeneracy_tol=args.degeneracy_tol,
        align=not args.no_align,
        no_ghost=args.no_ghost,
        onsite=args.onsite,
        symmetrize=not args.no_symmetrize,
        max_l=args.max_l,
        projection=args.projection,
        chk=args.chk,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
