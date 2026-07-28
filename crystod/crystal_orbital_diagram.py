"""Crystal-orbital diagrams from symmetry + extended-Hueckel overlap
(crystod --diagram).

The crystalline counterpart of the molecular-orbital diagram of
``crystod-mol --diagram --ao-left ... --ao-right ...``: the two fragment
sublattices given by ``--co-left``/``--co-right`` (e.g. the SrTi cation
framework and the O3 anion framework of SrTiO3) are treated with their
full-electron basis -- every core and valence shell of every atom
(WIEN2k-style; core shells with Slater-rule exponents and the archived
neutral-atom PySCF Hartree-Fock levels of reference/atomic_level_*,
collected into crystod/atomic_levels.py; shells frozen into the def2
effective core potential beyond Kr are omitted) -- and each fragment
feels the removed
sublattice as a point-charge lattice with the formal oxidation states
(the Madelung ligand field; see crystod.point_charge_field), so their
Bloch states are the complete electronic states before chemical bond
formation.  At every high-symmetry k point,

1. the Bloch orbitals of each fragment sublattice are symmetry-adapted
   (the crystal-orbital irreps of ``crystod --atomic-orbital``, i.e. the
   site-symmetry induced representations),
2. all inter- and intra-sublattice overlap integrals are evaluated exactly
   as Bloch lattice sums of single/double-zeta STO overlaps,
3. the generalized eigenvalue problem with the Wolfsberg-Helmholz
   Hamiltonian H_ij = K S_ij (H_ii + H_jj)/2 is solved -- fragment
   orbitals sharing an irrep of the little group mix into bonding and
   antibonding crystal orbitals, orbitals whose irrep finds no partner
   remain nonbonding -- exactly the construction rule of the crystal
   orbital diagram (COD),
4. the result is written as an interactive HTML diagram with one energy
   diagram per k point (fragment | crystal | fragment columns; the energy
   window opens on -20 .. 10 eV, "Show all energy levels" reveals the
   deep shells).

Every level carries a hover wave-function sketch: the Re[psi] amplitudes
of all its atomic-orbital components, rendered on the k-commensurate
supercell (same-l shells accumulate with their radial weights, so the
drawn lobe signs are those of the real wave function in the bonding
region).

Reference: Y. Mochizuki, M. Nishibori and T. Fukushima, "Crystal Orbital
Diagram of Perovskites: A Revisit from Symmetry-Adapted Linear
Combination" (in preparation).
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field

import numpy as np

from .atomic_levels import ATOMIC_LEVELS
from .mo_diagram import (
    ANGSTROM_TO_BOHR,
    CORE_SHELLS,
    EHT_PARAMETERS,
    WOLFSBERG_HELMHOLZ_K,
    AtomicOrbital,
    make_aligned_cache,
    pair_overlap,
    render_diagram_page,
    svg_sub_digits,
)
from .point_charge_field import (
    COULOMB_EV_ANGSTROM,
    ewald_site_potential,
    point_charge_block,
    radial_overlap,
    slater_zeta,
)
from .runtime_compat import get_character, get_chemical_symbols, get_scaled_positions
from .spglib_compat import ensure_spglib_compat

ensure_spglib_compat()

from phonopy.structure.cells import get_primitive_matrix_by_centring
from spgrep.core import get_spacegroup_irreps_from_primitive_symmetry

from .irreptables_compat import load_irreptables
from .operations import wigner_D_real, snap_qpoint
from .visualize_basis import SymmetryAdaptedOrbitalBasis

IrrepTable, _Irrep = load_irreptables()

_DEGENERACY_TOL = 1e-5

# default view of the interactive energy window: +-8 eV around the HOMO/LUMO
# midpoint (the VBM/CBM region one usually inspects first); the "Show all
# energy levels" button reveals everything outside it.  The fixed window below
# is only the fallback when no HOMO/LUMO pair exists.
_VIEW_HALF_WINDOW = 8.0
_VIEW_E_MIN = -20.0
_VIEW_E_MAX = 10.0

# Bloch combinations whose overlap eigenvalue falls below this floor are
# removed by canonical orthogonalization.  Diffuse cation valence shells
# (e.g. Sr 5s/5p, Ti 4s/4p) overlap so strongly in a dense sublattice that
# some Bloch combinations become nearly expressible by the rest of the
# basis; for those the extended-Hueckel H is no longer consistent and the
# energies diverge as (1-K) H_ii / eigenvalue (the well-known EHT overlap
# catastrophe), polluting even the occupied manifold.  Dropping the
# near-dependent combinations is the standard remedy and leaves the
# physical states untouched (measured on ScF3/SrTiO3: catastrophic modes
# all have eigenvalue <= 0.19, physical ones >= 0.26).
_OVERLAP_FLOOR = 0.2


@dataclass
class SublatticeSpec:
    """One (element, shell) block of a fragment sublattice."""

    element: str
    letter: str            # s / p / d
    shell: str             # e.g. "3d"
    n: int
    l: int
    zeta: object           # scalar or [(zeta, coeff), ...]
    h_ii: float
    sites: list[int]
    column: str            # "left" / "right"
    offset: int = 0        # first AO index of this spec

    @property
    def n_ao(self) -> int:
        return len(self.sites) * (2 * self.l + 1)


@dataclass
class DiagramLevel:
    level_id: str
    column: str            # "left" / "mo" / "right"
    energy: float
    degeneracy: int
    irrep: str             # bare irrep name, e.g. GM4-
    label: str             # display label
    electrons: int = 0
    vectors: np.ndarray | None = None    # (n_ao, degeneracy), S-orthonormal
    composition: list = field(default_factory=list)   # [(level_id, weight)]
    detail: str = ""


def parse_fragment_formula(tokens: list[str], flag: str) -> list[tuple[str, int | None]]:
    """Parse formula tokens (SrTi, O3, ...) into (element, count?) pairs."""
    pairs = []
    for token in tokens:
        if not re.fullmatch(r"(?:[A-Z][a-z]?\d*)+", token):
            raise SystemExit(
                f"ERROR: invalid {flag} formula '{token}' (expected element "
                "symbols with optional counts, e.g. SrTi or O3)."
            )
        for element, count in re.findall(r"([A-Z][a-z]?)(\d*)", token):
            pairs.append((element, int(count) if count else None))
    return pairs


def parse_oxidation_tokens(tokens: list[str]) -> dict[str, float]:
    """Parse El=Q tokens (Sr=+2, O=-2, Ti=4) into {element: charge}."""
    oxidation = {}
    for token in tokens:
        match = re.fullmatch(r"([A-Z][a-z]?)=([+-]?\d+(?:\.\d+)?)", token)
        if not match:
            raise SystemExit(
                f"ERROR: invalid --oxidation token '{token}' "
                "(expected e.g. Sr=+2 Ti=+4 O=-2)."
            )
        oxidation[match.group(1)] = float(match.group(2))
    return oxidation


def parse_sketch_tokens(tokens: list[str]) -> list[tuple[str, str]]:
    """Parse El-shell tokens (Ti-3d, Ti-d, O_2p) into (element, shell)."""
    wanted = []
    for token in tokens:
        parts = re.split(r"[-_]", token)
        if len(parts) != 2 or not re.fullmatch(r"\d?[spd]", parts[1]):
            raise SystemExit(
                f"ERROR: invalid --atomic-orbital token '{token}' "
                "(expected e.g. Ti-3d, Ti-d or O_2p)."
            )
        wanted.append((parts[0], parts[1]))
    return wanted


def _composition_string(symbols: list[str]) -> str:
    counts: dict[str, int] = {}
    for symbol in symbols:
        counts[symbol] = counts.get(symbol, 0) + 1
    return "".join(
        f"{element}{count if count > 1 else ''}" for element, count in counts.items()
    )


class CrystalOrbitalDiagram:
    """Symmetry + extended-Hueckel crystal-orbital diagram engine."""

    def __init__(self, cell, left_tokens: list[str], right_tokens: list[str],
                 symprec: float = 1e-5, electrons: float | None = None,
                 sketch_tokens: list[str] | None = None,
                 oxidation: dict[str, float] | None = None):
        self.builder = SymmetryAdaptedOrbitalBasis(cell=cell, symprec=symprec)
        primitive = self.builder.primitive_cell
        self.symbols = get_chemical_symbols(primitive)
        self.positions = np.array(get_scaled_positions(primitive))
        self.lattice = np.array(primitive.cell)   # rows, Angstrom

        formulas = {
            "left": parse_fragment_formula(left_tokens, "--co-left"),
            "right": parse_fragment_formula(right_tokens, "--co-right"),
        }
        self.formula = {
            "left": "".join(left_tokens),
            "right": "".join(right_tokens),
        }
        composition: dict[str, int] = {}
        for symbol in self.symbols:
            composition[symbol] = composition.get(symbol, 0) + 1
        comp_str = _composition_string(self.symbols)

        flags = {"left": "--co-left", "right": "--co-right"}
        assigned: dict[str, str] = {}
        for side in ("left", "right"):
            for element, count in formulas[side]:
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
                        f"ERROR: {flags[side]} lists {element}{count} but the "
                        f"primitive cell has {composition[element]} "
                        f"{element} atom(s) (composition: {comp_str})."
                    )
                if element not in EHT_PARAMETERS:
                    raise SystemExit(
                        f"ERROR: no extended-Hueckel parameters for element "
                        f"{element} (supported: {', '.join(EHT_PARAMETERS)})."
                    )
                assigned[element] = side
        missing = [el for el in composition if el not in assigned]
        if missing:
            raise SystemExit(
                f"ERROR: element(s) {', '.join(missing)} not assigned to "
                f"--co-left/--co-right (primitive-cell composition: "
                f"{comp_str}; every atom must belong to one fragment)."
            )

        # formal oxidation states of the ions: the removed sublattice enters
        # each fragment as a point-charge lattice with these charges
        if oxidation is None:
            from pymatgen.core import Composition

            guesses = Composition(comp_str).oxi_state_guesses()
            if not guesses:
                raise SystemExit(
                    "ERROR: could not guess the oxidation states of "
                    f"{comp_str}; pass them explicitly, e.g. "
                    "--oxidation Sr=+2 Ti=+4 O=-2."
                )
            oxidation = {el: float(q) for el, q in guesses[0].items()}
        missing_ox = [el for el in composition if el not in oxidation]
        if missing_ox:
            raise SystemExit(
                f"ERROR: --oxidation misses element(s) "
                f"{', '.join(missing_ox)}."
            )
        net = sum(oxidation[el] * composition[el] for el in composition)
        if abs(net) > 1e-6:
            raise SystemExit(
                f"ERROR: oxidation states are not charge-neutral "
                f"(net {net:+g} per cell)."
            )
        self.oxidation = oxidation

        # full-electron basis (core + valence shells of every atom),
        # fragment-major so each fragment is one contiguous AO block; core
        # shells get Slater-rule exponents and the archived PySCF
        # neutral-atom Hartree-Fock levels (reference/atomic_level_*),
        # valence shells the extended-Hueckel parameters.  Shells frozen
        # into the def2 effective core potential (beyond Kr) do not exist
        # in the atomic data and are omitted -- the pseudopotential
        # picture.
        self.specs: list[SublatticeSpec] = []
        self.side_specs = {"left": [], "right": []}
        offset = 0
        for side in ("left", "right"):
            for element, _count in formulas[side]:
                if element not in ATOMIC_LEVELS:
                    raise SystemExit(
                        f"ERROR: no archived atomic levels for {element}; "
                        "run script/generate_atomic_levels.py and "
                        "script/collect_atomic_levels.py."
                    )
                sites = [
                    index for index, symbol in enumerate(self.symbols)
                    if symbol == element
                ]
                levels = ATOMIC_LEVELS[element]["levels"]
                shells = [
                    (shell, int(shell[0]), "spdf".index(shell[-1]),
                     slater_zeta(element, shell), levels[shell])
                    for shell in CORE_SHELLS[element] if shell in levels
                ] + list(EHT_PARAMETERS[element])
                for shell, n, l, zeta, h_ii in shells:
                    spec = SublatticeSpec(element, shell[-1], shell, n, l,
                                          zeta, h_ii, sites, side, offset)
                    self.specs.append(spec)
                    self.side_specs[side].append(spec)
                    offset += spec.n_ao
        self.n_ao = offset
        self.side_slice = {}
        start = 0
        for side in ("left", "right"):
            width = sum(spec.n_ao for spec in self.side_specs[side])
            self.side_slice[side] = slice(start, start + width)
            start += width

        # AO objects in the representation ordering (spec-major, site-major)
        self.orbitals: list[AtomicOrbital] = []
        for spec in self.specs:
            for site in spec.sites:
                for m in range(2 * spec.l + 1):
                    self.orbitals.append(AtomicOrbital(
                        site, spec.element, spec.shell, spec.n, spec.l, m,
                        spec.zeta, spec.h_ii,
                    ))

        self.side_electrons = {
            side: sum(
                ATOMIC_LEVELS[element]["electrons"] * composition[element]
                for element, _count in formulas[side]
            )
            for side in ("left", "right")
        }
        if electrons is None:
            electrons = sum(self.side_electrons.values())
        self.electrons = float(electrons)

        # --atomic-orbital: the sketch filter (indices into self.specs)
        self.sketch_specs: frozenset[int] | None = None
        self.sketch_tokens = sketch_tokens
        if sketch_tokens is not None:
            available = []
            for spec in self.specs:
                token = f"{spec.element}-{spec.shell}"
                if token not in available:
                    available.append(token)
            selected = set()
            for element, shell in parse_sketch_tokens(sketch_tokens):
                hits = [
                    index for index, spec in enumerate(self.specs)
                    if spec.element == element
                    and (spec.shell == shell
                         or (len(shell) == 1 and spec.letter == shell))
                ]
                if not hits:
                    raise SystemExit(
                        f"ERROR: --atomic-orbital {element}-{shell} matches no "
                        f"basis orbital (available: {', '.join(available)})."
                    )
                selected.update(hits)
            self.sketch_specs = frozenset(selected)

        self._aligned = make_aligned_cache()
        self._cutoffs = self._pair_cutoffs()
        self._images = self._lattice_images(max(self._cutoffs.values()))
        self._build_ligand_field()

    # -------------------------------------------------- point-charge field

    def _build_ligand_field(self, r_cut_angstrom: float = 7.0):
        """Same-site matrices of the removed-sublattice point-charge field.

        Every atom feels the complementary sublattice as a lattice of point
        charges with the formal oxidation states: charges within
        r_cut_angstrom enter as exact <phi_i|q/|r-R||phi_j> STO integrals
        (monopole shift + multipole ligand-field splitting), the long-range
        rest as the Ewald site potential (neutralizing-background
        convention, as in charged periodic DFT cells).  The blocks are
        added identically to the fragment and the crystal Hamiltonians, so
        the three diagram columns share one energy reference."""
        complementary = {"left": "right", "right": "left"}
        side_of_atom = {}
        for side in ("left", "right"):
            for spec in self.side_specs[side]:
                for site in spec.sites:
                    side_of_atom[site] = side
        charge_lattice = {
            side: [
                (self.oxidation[self.symbols[site]], self.positions[site])
                for site in range(len(self.symbols))
                if side_of_atom[site] == side
            ]
            for side in ("left", "right")
        }
        bounds = []
        volume = abs(np.linalg.det(self.lattice))
        for i in range(3):
            j, k = (i + 1) % 3, (i + 2) % 3
            perpendicular = volume / np.linalg.norm(
                np.cross(self.lattice[j], self.lattice[k])
            )
            bounds.append(int(np.ceil(r_cut_angstrom / perpendicular)) + 1)
        images = [
            np.array([n1, n2, n3])
            for n1 in range(-bounds[0], bounds[0] + 1)
            for n2 in range(-bounds[1], bounds[1] + 1)
            for n3 in range(-bounds[2], bounds[2] + 1)
        ]

        self.h_raw = np.array([o.h_ii for o in self.orbitals], dtype=float)
        self.h_bar = self.h_raw.copy()
        self.v_onsite = np.zeros((self.n_ao, self.n_ao))
        self.site_potential = np.zeros(len(self.symbols))
        for atom in range(len(self.symbols)):
            charges = charge_lattice[complementary[side_of_atom[atom]]]
            near = []
            near_monopole = 0.0
            for q, frac in charges:
                for image in images:
                    vector = (frac + image - self.positions[atom]) @ self.lattice
                    distance = float(np.linalg.norm(vector))
                    if distance <= r_cut_angstrom:
                        near.append((q, vector * ANGSTROM_TO_BOHR))
                        near_monopole += q / distance
            v_ewald = ewald_site_potential(
                self.lattice, charges, self.positions[atom]
            )
            self.site_potential[atom] = -COULOMB_EV_ANGSTROM * v_ewald
            e_far = -COULOMB_EV_ANGSTROM * (v_ewald - near_monopole)
            indices = [
                index for index, orbital in enumerate(self.orbitals)
                if orbital.atom == atom
            ]
            orbitals = [self.orbitals[index] for index in indices]
            block = point_charge_block(orbitals, near)
            # The EHT basis treats same-site shells of one l (2p/3p/4p,
            # ...) as orthonormal, but the raw STO radials are not: the
            # point-charge matrix must be expressed in the Loewdin-
            # orthogonalized on-site basis, so that a constant potential
            # maps exactly onto the identity (and the far-field term is
            # exactly diagonal).
            s_site = np.eye(len(indices))
            for a_local, oa in enumerate(orbitals):
                for b_local in range(a_local + 1, len(indices)):
                    ob = orbitals[b_local]
                    if (oa.l, oa.m) == (ob.l, ob.m) and oa.shell != ob.shell:
                        s_site[a_local, b_local] = s_site[b_local, a_local] \
                            = radial_overlap((oa.n, oa.zeta), (ob.n, ob.zeta))
            s_values, s_vectors = np.linalg.eigh(s_site)
            o_half = (s_vectors / np.sqrt(s_values)) @ s_vectors.T
            block = o_half @ block @ o_half
            block += np.eye(len(indices)) * e_far
            # The monopole (Madelung) part of the site shift is omitted: it
            # depends on the neutralizing-background convention of the
            # charged sublattice array (it flips the SrTiO3 band ordering),
            # and it largely cancels against the intra-atomic charging
            # energy not present in the extended-Hueckel VSIPs -- the
            # standard argument why neutral-atom VSIPs work in ionic
            # crystals.  What remains is background-independent and
            # absolutely convergent: the anisotropic multipole ligand field
            # (t2g/eg splittings, ...) and the near-shell penetration
            # corrections.  The omitted jellium-referenced monopole is kept
            # in self.site_potential for the report.
            block -= np.eye(len(indices)) * self.site_potential[atom]
            self.v_onsite[np.ix_(indices, indices)] = block
            # shell-averaged (rotation-invariant) shift for the W-H h_bar
            shells = {}
            for local, orbital in enumerate(orbitals):
                shells.setdefault(orbital.shell, []).append(local)
            for members in shells.values():
                average = float(np.mean([block[m, m] for m in members]))
                for m in members:
                    self.h_bar[indices[m]] += average

    # ------------------------------------------------------------ Bloch sums

    def _pair_cutoffs(self, tol: float = 2e-5, r_max: float = 44.0):
        """Per shell-pair lattice-sum cutoff from the actual STO tails.

        Diffuse cation shells (e.g. Sr 5s, Ti 4p) still overlap at 30+ bohr;
        a fixed short cutoff truncates their Bloch sums so badly that S_k
        loses positive semidefiniteness.  The sigma overlap along z is
        probed outward until it falls below tol."""
        sigma_m = {0: 0, 1: 2, 2: 2}   # s / pz / dz2: the slowest-decaying
        kinds = []
        seen = set()
        for spec in self.specs:
            key = (spec.element, spec.shell)
            if key not in seen:
                seen.add(key)
                kinds.append(spec)
        cutoffs = {}
        for index, a_spec in enumerate(kinds):
            for b_spec in kinds[index:]:
                key_a = (a_spec.element, a_spec.shell)
                key_b = (b_spec.element, b_spec.shell)
                if a_spec.l > 2 or b_spec.l > 2:
                    # f shells appear only as ultra-compact cores (Pb/Bi
                    # 4f); their inter-site overlap is neglected
                    cutoffs[key_a, key_b] = cutoffs[key_b, key_a] = 6.0
                    continue
                a = AtomicOrbital(0, a_spec.element, a_spec.shell, a_spec.n,
                                  a_spec.l, sigma_m[a_spec.l], a_spec.zeta,
                                  a_spec.h_ii)
                b = AtomicOrbital(0, b_spec.element, b_spec.shell, b_spec.n,
                                  b_spec.l, sigma_m[b_spec.l], b_spec.zeta,
                                  b_spec.h_ii)
                r = 6.0
                while r < r_max and abs(pair_overlap(
                    a, b, np.array([0.0, 0.0, r]), self._aligned
                )) > tol:
                    r += 2.0
                cutoffs[key_a, key_b] = r
                cutoffs[key_b, key_a] = r
        return cutoffs

    def _lattice_images(self, cutoff_bohr: float):
        """Integer lattice translations n with any-atom pair within cutoff."""
        lattice_bohr = self.lattice * ANGSTROM_TO_BOHR
        volume = abs(np.linalg.det(lattice_bohr))
        bounds = []
        for i in range(3):
            j, k = (i + 1) % 3, (i + 2) % 3
            perpendicular = volume / np.linalg.norm(
                np.cross(lattice_bohr[j], lattice_bohr[k])
            )
            # +1: margin for the in-cell offset x_j - x_i
            bounds.append(int(np.ceil(cutoff_bohr / perpendicular)) + 1)
        return np.array([
            [n1, n2, n3]
            for n1 in range(-bounds[0], bounds[0] + 1)
            for n2 in range(-bounds[1], bounds[1] + 1)
            for n3 in range(-bounds[2], bounds[2] + 1)
        ])

    def bloch_overlap(self, kpoint) -> np.ndarray:
        """Bloch overlap matrix in the atom gauge:
        S_k(i, j) = sum_n exp(2 pi i k.(n + x_j - x_i)) s(i at 0, j at n)."""
        k = np.asarray(kpoint, dtype=float)
        S = np.zeros((self.n_ao, self.n_ao), dtype=complex)
        lattice_bohr = self.lattice * ANGSTROM_TO_BOHR
        for i in range(self.n_ao):
            for j in range(i, self.n_ao):
                a, b = self.orbitals[i], self.orbitals[j]
                cutoff = self._cutoffs[
                    (a.element, a.shell), (b.element, b.shell)
                ]
                offset = self.positions[b.atom] - self.positions[a.atom]
                fractionals = offset + self._images
                distances = np.linalg.norm(fractionals @ lattice_bohr, axis=1)
                total = 0.0 + 0.0j
                for image_index in np.nonzero(distances <= cutoff)[0]:
                    fractional = fractionals[image_index]
                    if distances[image_index] < 1e-9:
                        overlap = 1.0 if (a.shell, a.m) == (b.shell, b.m) else 0.0
                    elif a.l > 2 or b.l > 2:
                        overlap = 0.0   # compact f cores: no inter-site overlap
                    else:
                        overlap = pair_overlap(
                            a, b, fractional @ lattice_bohr, self._aligned
                        )
                    if overlap == 0.0:
                        continue
                    phase = np.exp(2j * np.pi * float(k @ fractional))
                    total += phase * overlap
                S[i, j] = total
                S[j, i] = np.conj(total)
        return S

    def hamiltonian(self, S: np.ndarray) -> np.ndarray:
        """Wolfsberg-Helmholz Hamiltonian over the Bloch overlaps.

        The W-H prefactor uses the shell-averaged shifted energies h_bar
        (rotation-invariant, so the symmetry of H stays exact); the on-site
        blocks carry the bare atomic energies plus the full anisotropic
        point-charge ligand-field matrices (v_onsite).  The diagonal of S_k
        is 1 + the same-orbital neighbour-cell Bloch sum, so only the
        on-site R = 0 term is the bare atomic energy: the diagonal
        correction (1 - K) restores h + K h_bar (S_kk - 1) + V_ii."""
        H = 0.5 * WOLFSBERG_HELMHOLZ_K * (
            self.h_bar[:, None] + self.h_bar[None, :]
        ) * S
        H += self.v_onsite
        H[np.diag_indices(self.n_ao)] += (
            self.h_raw - WOLFSBERG_HELMHOLZ_K * self.h_bar
        )
        return H

    # -------------------------------------------------------- representation

    def little_group_data(self, kpoint):
        """spgrep irreps, physical labels, and the combined orbital rep."""
        irreps, mapping = get_spacegroup_irreps_from_primitive_symmetry(
            rotations=self.builder.rotations,
            translations=self.builder.translations,
            kpoint=kpoint,
        )
        labels = self.builder.get_irrep_labels(kpoint, irreps, mapping)
        little_rotations = self.builder.rotations[mapping]
        little_translations = self.builder.translations[mapping]
        permutations = self.builder.get_permutation_reps_at_k(
            little_rotations=little_rotations,
            little_translations=little_translations,
            kpoint=kpoint,
        )
        representation = []
        for op_index, op in enumerate(mapping):
            blocks = []
            wigners = {}
            for spec in self.specs:
                if spec.l not in wigners:
                    wigners[spec.l] = wigner_D_real(
                        spec.l,
                        np.real(self.builder.rotations_cartesian[op]),
                    )
                grid = np.ix_(spec.sites, spec.sites)
                blocks.append(np.kron(
                    permutations[op_index][grid], wigners[spec.l]
                ))
            matrix = np.zeros((self.n_ao, self.n_ao), dtype=complex)
            row = 0
            for block in blocks:
                size = block.shape[0]
                matrix[row:row + size, row:row + size] = block
                row += size
            representation.append(matrix)
        return irreps, mapping, labels, representation

    # -------------------------------------------------------------- solving

    @staticmethod
    def _generalized_eigh(H: np.ndarray, S: np.ndarray):
        """Canonically orthogonalized generalized eigenproblem (complex
        Hermitian).  Near-dependent Bloch combinations (overlap eigenvalue
        below _OVERLAP_FLOOR, see there) are dropped; the number of dropped
        combinations is returned."""
        s_values, s_vectors = np.linalg.eigh(S)
        keep = s_values > _OVERLAP_FLOOR
        X = s_vectors[:, keep] / np.sqrt(s_values[keep])
        H_orth = X.conj().T @ H @ X
        energies, coefficients = np.linalg.eigh(H_orth)
        return energies, X @ coefficients, int(np.sum(~keep))

    def _group_levels(self, energies, vectors):
        """Cluster eigenvalues into degenerate groups."""
        groups = []
        start = 0
        for i in range(1, len(energies) + 1):
            if i == len(energies) or energies[i] - energies[start] > max(
                _DEGENERACY_TOL, 1e-6 * max(1.0, abs(energies[start]))
            ):
                groups.append((float(np.mean(energies[start:i])),
                               vectors[:, start:i]))
                start = i
        return groups

    def _irrep_split(self, vectors, S, representation, irreps, labels,
                     subspace=None):
        """Split an S-orthonormal degenerate group into its irrep components.

        Robust against accidental degeneracies (several irreps at one
        energy, e.g. nearly uncoupled sublattice shells): the group space is
        decomposed with the character projectors and one (label, vectors)
        entry is returned per contributing irrep."""
        if subspace is not None:
            embedded = np.zeros((self.n_ao, vectors.shape[1]), dtype=complex)
            embedded[subspace] = vectors
            vectors = embedded
        gram = vectors.conj().T @ S @ vectors
        values, basis = np.linalg.eigh(gram)
        V = vectors @ (basis / np.sqrt(values))
        characters = np.array([
            np.trace(V.conj().T @ S @ D @ V) for D in representation
        ])
        order = len(representation)
        components = []
        for irrep, label in zip(irreps, labels):
            chi = np.array(get_character(irrep), dtype=complex)
            multiplicity = float(np.real(
                np.sum(characters * np.conj(chi)) / order
            ))
            count = int(round(multiplicity))
            if count <= 0:
                continue
            dimension = irrep.shape[1]
            projector = np.zeros((self.n_ao, self.n_ao), dtype=complex)
            for g, D in enumerate(representation):
                projector += np.conj(chi[g]) * D
            projector *= dimension / order
            projected = projector @ V
            # S-orthonormal basis of the projected span
            gram_p = projected.conj().T @ S @ projected
            p_values, p_basis = np.linalg.eigh(gram_p)
            keep = p_values > 1e-6
            space = projected @ (p_basis[:, keep] / np.sqrt(p_values[keep]))
            if space.shape[1] != count * dimension:
                # numerical safety: fall back to the expected count
                space = space[:, : count * dimension]
            components.append((label, space))
        if not components:
            components.append(("?", V))
        return components

    def _dominant_spec(self, space, S, column) -> SublatticeSpec:
        """Fragment (element, shell) block with the largest Mulliken gross
        population of an S-orthonormal level space (for the level label)."""
        Sv = S @ space
        weights = []
        for spec in self.side_specs[column]:
            rows = slice(spec.offset, spec.offset + spec.n_ao)
            weights.append(float(np.real(
                np.sum(np.conj(space[rows]) * Sv[rows])
            )))
        return self.side_specs[column][int(np.argmax(weights))]

    def solve_at(self, kpoint):
        """All fragment and crystal levels at one k point."""
        S = self.bloch_overlap(kpoint)
        H = self.hamiltonian(S)
        irreps, mapping, labels, representation = self.little_group_data(kpoint)

        # gauge self-check: the representation must leave S invariant
        worst = max(
            float(np.max(np.abs(D.conj().T @ S @ D - S)))
            for D in representation
        )
        if worst > 1e-5:
            raise SystemExit(
                "ERROR: Bloch-overlap gauge inconsistency "
                f"(residual {worst:.2e}); please report this case."
            )
        # ... and H (checks the point-charge ligand-field blocks against
        # the site-symmetry representation, i.e. the real-harmonics
        # conventions)
        h_scale = float(np.max(np.abs(H)))
        worst_h = max(
            float(np.max(np.abs(D.conj().T @ H @ D - H)))
            for D in representation
        ) / max(h_scale, 1.0)
        if worst_h > 1e-6:
            raise SystemExit(
                "ERROR: Hamiltonian symmetry inconsistency "
                f"(relative residual {worst_h:.2e}); please report this case."
            )

        def strip(label):
            return label.split("(")[0]

        levels = {"left": [], "mo": [], "right": []}
        self.last_dropped = 0
        # fragment (sublattice) levels: the full valence problem of one side
        for column in ("left", "right"):
            block = self.side_slice[column]
            indices = np.arange(block.start, block.stop)
            energies, vectors, dropped = self._generalized_eigh(
                H[block, block], S[block, block]
            )
            self.last_dropped += dropped
            for energy, group in self._group_levels(energies, vectors):
                for irrep_label, space in self._irrep_split(
                    group, S, representation, irreps, labels, subspace=indices
                ):
                    name = strip(irrep_label)
                    spec = self._dominant_spec(space, S, column)
                    levels[column].append(DiagramLevel(
                        level_id=f"{column}{len(levels[column])}",
                        column=column,
                        energy=float(energy),
                        degeneracy=space.shape[1],
                        irrep=name,
                        label=f"{spec.element} {spec.shell} {name}",
                        vectors=space,
                    ))
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

        # crystal levels
        energies, vectors, dropped = self._generalized_eigh(H, S)
        self.last_dropped += dropped
        counts: dict[str, int] = {}
        for energy, group in self._group_levels(energies, vectors):
            for irrep_label, space in self._irrep_split(
                group, S, representation, irreps, labels
            ):
                name = strip(irrep_label)
                counts[name] = counts.get(name, 0) + 1
                occurrence = counts[name]
                levels["mo"].append(DiagramLevel(
                    level_id=f"mo{len(levels['mo'])}",
                    column="mo",
                    energy=float(energy),
                    degeneracy=space.shape[1],
                    irrep=name,
                    # "GM4- #2" = second GM4- multiplet from the bottom, the
                    # same #N numbering as the fragment columns; "GM4-(2)"
                    # read like a degeneracy count
                    label=f"{name} #{occurrence}",
                    vectors=space,
                ))

        # compositions: crystal level onto fragment levels (S metric)
        for crystal in levels["mo"]:
            weights = []
            for column in ("left", "right"):
                for fragment in levels[column]:
                    overlap = fragment.vectors.conj().T @ S @ crystal.vectors
                    weight = float(np.sum(np.abs(overlap) ** 2))
                    weight /= crystal.degeneracy
                    if weight > 1e-6:
                        weights.append((fragment.level_id, weight))
            total = sum(w for _, w in weights) or 1.0
            crystal.composition = [(i, w / total) for i, w in weights]

        # electron filling (aufbau, 2 electrons per orbital)
        self._fill(levels["mo"], self.electrons)
        for column in ("left", "right"):
            self._fill(levels[column], self.side_electrons[column])
        return levels, labels

    @staticmethod
    def _fill(column_levels, electrons):
        remaining = float(electrons)
        for level in sorted(column_levels, key=lambda lv: lv.energy):
            capacity = 2 * level.degeneracy
            take = int(round(min(remaining, capacity)))
            level.electrons = max(take, 0)
            remaining -= take
            if remaining <= 0:
                break

    # ---------------------------------------------------- wave-function sketch

    def supercell_for(self, kpoint):
        """k-commensurate supercell: (cells, symbols, cartesian positions)."""
        from fractions import Fraction

        repetitions = [
            Fraction(float(value)).limit_denominator(12).denominator
            for value in kpoint
        ]
        cells = [
            np.array([n1, n2, n3])
            for n1 in range(repetitions[0])
            for n2 in range(repetitions[1])
            for n3 in range(repetitions[2])
        ]
        symbols, cartesian = [], []
        for cell_vector in cells:
            for index, symbol in enumerate(self.symbols):
                symbols.append(symbol)
                cartesian.append((self.positions[index] + cell_vector)
                                 @ self.lattice)
        return cells, symbols, np.array(cartesian)

    def sketch_partners(self, level: DiagramLevel, kpoint, cells):
        """Per-partner real wave-function amplitudes on the supercell atoms,
        as sketch entries [atom, s, px, py, pz, dxy, dyz, dz2, dxz, dx2-y2].

        Only the --atomic-orbital components (self.sketch_specs) are drawn.
        The amplitudes are Re[psi] (or Im[psi] when Re vanishes) of the
        Bloch crystal orbital, so the sign alternation between the cells of
        the k-commensurate supercell is displayed faithfully; degenerate
        partners are realified and RREF-canonicalized like the molecular
        sketch.

        Same-l shells of one atom (e.g. Sc 2p/3p/4p) share their slots and
        accumulate, each weighted by its STO radial amplitude at a
        representative bonding-region radius (r0 = 2 bohr), so the drawn
        lobe signs are the signs of the real wave function there.  (A bare
        coefficient of one shell is wrong: a semicore level like Sc 3p
        would be drawn from the tiny orthogonalization tail of the 4p
        shell, whose sign is inverted -- the crystal analogue of the
        contracted-GTO compression in the molecular PySCF sketch.)"""
        from .point_charge_field import _primitives
        from .visualize_basis import realify_basis_space

        rows, _ = realify_basis_space(level.vectors.T)
        rows = np.asarray(rows)
        n_prim = len(self.symbols)
        width = 9
        slot_of = {0: 0, 1: 1, 2: 4}
        r0 = 2.0  # bohr
        angular = {0: 0.28209479, 1: 0.48860251, 2: 0.63078313}
        radial_weight = [
            angular[spec.l] * sum(
                c * r0 ** (n - 1) * np.exp(-z * r0)
                for c, n, z in _primitives(spec.n, spec.zeta)
            ) if spec.l in slot_of else 0.0
            for spec in self.specs
        ]
        partner_rows = []
        for vector in rows:
            amp_re = np.zeros((len(cells) * n_prim, width))
            amp_im = np.zeros_like(amp_re)
            for spec_index, spec in enumerate(self.specs):
                if (self.sketch_specs is not None
                        and spec_index not in self.sketch_specs):
                    continue
                if spec.l not in slot_of:
                    continue
                slot = slot_of[spec.l]
                w = 2 * spec.l + 1
                for site_pos, site in enumerate(spec.sites):
                    start = spec.offset + site_pos * w
                    block = vector[start:start + w]
                    for c_index, cell_vector in enumerate(cells):
                        phase = np.exp(2j * np.pi * float(np.dot(
                            kpoint, cell_vector + self.positions[site]
                        )))
                        values = (np.asarray(block) * phase
                                  * radial_weight[spec_index])
                        row_index = c_index * n_prim + site
                        amp_re[row_index, slot:slot + w] += values.real
                        amp_im[row_index, slot:slot + w] += values.imag
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

    # ------------------------------------------------------------- k points

    def special_kpoints(self):
        """(name, kpoint) list of the tabulated special points."""
        table = IrrepTable(self.builder.spglib_dataset["number"], spinor=False)
        primitive_matrix = get_primitive_matrix_by_centring(
            self.builder.spglib_dataset["international"][0]
        )
        names, kpoints = [], []
        for irrep in table.irreps:
            kpoint = snap_qpoint(np.array(irrep.k) @ primitive_matrix)
            if kpoint not in kpoints:
                kpoints.append(kpoint)
                names.append(irrep.kpname)
        return list(zip(names, kpoints))


# --------------------------------------------------------------------- output


def _format_kpoint(kpoint) -> str:
    from fractions import Fraction

    parts = []
    for value in kpoint:
        fraction = Fraction(float(value)).limit_denominator(12)
        parts.append(str(fraction))
    return "(" + ",".join(parts) + ")"


def _detail_html(level: DiagramLevel, names: dict) -> str:
    # consumed as the SVG <title> textContent (the native hover tooltip),
    # which renders newlines but shows HTML tags literally
    rows = [
        f"E = {level.energy:.2f} eV",
        f"irrep: {level.irrep} (degeneracy {level.degeneracy})",
        f"electrons: {level.electrons}",
    ]
    if level.detail:
        rows.append(level.detail)
    return "\n".join(rows)


def write_crystal_diagram_html(diagram: CrystalOrbitalDiagram,
                               k_entries: list, output_path: str,
                               structure_label: str) -> None:
    """One interactive page with one energy diagram per k point."""
    columns = {"left": 200, "mo": 480, "right": 760}
    half = {"left": 34, "mo": 34, "right": 34}
    order = ["left", "mo", "right"]
    side = {"left": -1, "mo": 1, "right": 1}
    headers = {
        "left": svg_sub_digits(diagram.formula["left"]),
        "mo": "crystal orbitals",
        "right": svg_sub_digits(diagram.formula["right"]),
    }

    from .mo_diagram import diagram_geometry

    variants = []
    for name, kpoint, levels in k_entries:
        cells, super_symbols, super_positions = diagram.supercell_for(kpoint)
        geometry = diagram_geometry(super_symbols, super_positions)
        names = {
            level.level_id: level.label
            for column in ("left", "right") for level in levels[column]
        }
        levels_json = []
        for column in order:
            for level in levels[column]:
                levels_json.append({
                    "id": level.level_id,
                    "col": level.column,
                    "e": round(level.energy, 4),
                    "deg": level.degeneracy,
                    "label": level.label,
                    "el": level.electrons,
                    "occ": level.electrons > 0,
                    "links": [
                        [i, round(w, 4)]
                        for i, w in level.composition if w >= 0.02
                    ],
                    "comp": [
                        [names[i], round(100 * w, 1)]
                        for i, w in sorted(level.composition,
                                           key=lambda kv: -kv[1])
                        if w > 0.005
                    ],
                    "detail": _detail_html(level, names),
                    "orb": diagram.sketch_partners(level, kpoint, cells),
                })
        occupied = [lv for lv in levels["mo"] if lv.electrons > 0]
        empty = [lv for lv in levels["mo"] if lv.electrons == 0]
        homo_level = max(occupied, key=lambda lv: lv.energy) if occupied else None
        lumo_level = min(empty, key=lambda lv: lv.energy) if empty else None
        homo = homo_level.level_id if homo_level else None
        lumo = lumo_level.level_id if lumo_level else None
        energies = [lv.energy for column in order for lv in levels[column]]
        e_min, e_max = min(energies), max(energies)
        padding = 0.08 * (e_max - e_min) or 1.0
        # the interactive view opens on the frontier states: +-8 eV around the
        # HOMO/LUMO midpoint ("Show all energy levels" reveals the deep shells
        # outside it); without a HOMO/LUMO pair, fall back to a fixed window
        if homo_level is not None and lumo_level is not None:
            center = 0.5 * (homo_level.energy + lumo_level.energy)
            view_lo, view_hi = center - _VIEW_HALF_WINDOW, center + _VIEW_HALF_WINDOW
        else:
            view_lo = max(e_min - padding, _VIEW_E_MIN)
            view_hi = min(e_max + padding, _VIEW_E_MAX)
        if view_hi - view_lo < 1.0:
            view_lo, view_hi = e_min - padding, e_max + padding
        variants.append({
            "key": f"{name} {_format_kpoint(kpoint)}",
            "levels": levels_json,
            "homo": homo,
            "lumo": lumo,
            "eMin": round(view_lo, 2),
            "eMax": round(view_hi, 2),
            "geom": geometry,
        })

    first = variants[0]
    fragment_names = (f"{svg_sub_digits(diagram.formula['left'])} + "
                      f"{svg_sub_digits(diagram.formula['right'])}")
    chips = [
        structure_label,
        f"{diagram.builder.spglib_dataset['international']} "
        f"(No. {diagram.builder.spglib_dataset['number']})",
        f"{fragment_names} (full-electron basis)",
        f"{int(diagram.electrons)} electrons / cell",
        "point charges: " + " ".join(
            f"{element}{diagram.oxidation[element]:+g}"
            for element in dict.fromkeys(diagram.symbols)
        ),
        getattr(diagram, "method_chip", "extended H&uuml;ckel + SALC"),
    ]
    sketch_foot = (
        " Click or hover a level for its composition and the real-space "
        "wave function Re[&psi;] of all its atomic-orbital components drawn "
        "on the k-commensurate supercell (drag to rotate; degenerate "
        "partners switchable)."
    )
    render_diagram_page(
        output_path,
        title=f"Crystal orbital diagram: {structure_label}",
        heading_html=(
            f"Crystal-orbital diagram: {structure_label} "
            f"<span style=\"color:#90a4ae;font-size:14px\">{fragment_names}, "
            "per k point</span>"
        ),
        chips=chips,
        columns=columns, half=half, order=order, side=side, headers=headers,
        levels_json=first["levels"],
        homo_id=first["homo"],
        lumo_id=first["lumo"],
        e_min=first["eMin"], e_max=first["eMax"],
        foot_html=(
            "Crystal-orbital diagram (COD): the fragment-sublattice Bloch "
            "orbitals (columns, full-electron basis: every core and valence "
            "shell) are the electronic states before chemical bond "
            "formation, in the point-charge ligand field of the removed "
            "sublattice (formal oxidation states; exact multipole + "
            "penetration terms, background-dependent monopole omitted); "
            "states sharing an irrep of the little group at k mix into "
            "bonding/antibonding crystal orbitals (center), states without "
            "a partner remain nonbonding. Energies: symmetry-adapted "
            "extended H&uuml;ckel (VSIP/core-level diagonal + "
            "Wolfsberg-Helmholz off-diagonal over exact Bloch STO overlap "
            "sums). The energy window opens on -20 .. 10 eV; use \"Show all "
            "energy levels\" for the core shells. Switch the k point with "
            "the buttons above." + sketch_foot
        ),
        geometry=variants[0]["geom"],
        variants=variants,
    )


def report_and_write(cell, *, left, right, symprec, electrons,
                     kpoint_filter, output_path, structure_label,
                     oxidation=None):
    """Terminal report + HTML for the crystal-orbital diagram."""
    diagram = CrystalOrbitalDiagram(
        cell, left, right, symprec=symprec, electrons=electrons,
        oxidation=oxidation,
    )
    dataset = diagram.builder.spglib_dataset
    print("\n * Space group *")
    print(f" {dataset['international']} ({dataset['number']})\n")
    print(" * Fragments (full-electron basis: core + valence shells;"
          " atomic levels from reference/atomic_level_*) *")
    for column in ("left", "right"):
        by_element: dict[str, list] = {}
        for spec in diagram.side_specs[column]:
            by_element.setdefault(spec.element, []).append(spec)
        parts = " | ".join(
            f"{element} " + " ".join(spec.shell for spec in specs)
            + f" x{len(specs[0].sites)} site(s)"
            + (f" [ECP-{ATOMIC_LEVELS[element]['ecp_core']} core frozen]"
               if ATOMIC_LEVELS[element]["ecp_core"] else "")
            for element, specs in by_element.items()
        )
        print(f" {column:<5} {diagram.formula[column]:<6}: {parts}, "
              f"{diagram.side_electrons[column]} electrons")
    print(f" electrons per cell in the diagram: {int(diagram.electrons)}"
          + (" (all electrons of the neutral atoms; override with"
             " --electrons)" if electrons is None else ""))
    print(" * Ligand-field point charges (removed sublattice) *")
    shifts: dict[str, list] = {}
    for atom, symbol in enumerate(diagram.symbols):
        shifts.setdefault(symbol, []).append(diagram.site_potential[atom])
    other = {"left": "right", "right": "left"}
    for column in ("left", "right"):
        felt = " + ".join(
            f"{element}^{diagram.oxidation[element]:+g}"
            for element in dict.fromkeys(
                spec.element for spec in diagram.side_specs[other[column]]
            )
        )
        own = ", ".join(
            f"{element} {np.mean(shifts[element]):+.2f} eV"
            for element in dict.fromkeys(
                spec.element for spec in diagram.side_specs[column]
            )
        )
        print(f" {column:<5} {diagram.formula[column]:<6} feels the {felt} "
              f"lattice (multipole ligand field + penetration; "
              f"jellium-referenced monopole {own} omitted)")
    print(" hover wave-function sketches: all atomic-orbital components "
          "of every level\n")

    entries = []
    kpoints = diagram.special_kpoints()
    if kpoint_filter is not None:
        available = [name for name, _ in kpoints]
        kpoints = [
            (name, kpoint) for name, kpoint in kpoints
            if name == kpoint_filter
        ]
        if not kpoints:
            raise SystemExit(
                f"ERROR: k point '{kpoint_filter}' is not a special point of "
                f"this space group (available: {', '.join(available)})."
            )
    for name, kpoint in kpoints:
        levels, _ = diagram.solve_at(kpoint)
        entries.append((name, kpoint, levels))
        print(f" * k point {name} {_format_kpoint(kpoint)} *")
        if diagram.last_dropped:
            print(f"   ({diagram.last_dropped} near-dependent diffuse Bloch "
                  "combination(s) removed by canonical orthogonalization; "
                  f"overlap floor {_OVERLAP_FLOOR})")
        for column in ("left", "right"):
            parts = ", ".join(
                f"{lv.label} ({lv.energy:.2f})"
                for lv in sorted(levels[column], key=lambda lv: lv.energy)
            )
            print(f"   {diagram.formula[column]:<10}: {parts}")
        print("   crystal   :")
        names = {
            lv.level_id: lv.label
            for column in ("left", "right") for lv in levels[column]
        }
        for lv in sorted(levels["mo"], key=lambda lv: lv.energy):
            composition = "  ".join(
                f"{names[i]} {100 * w:.0f}%"
                for i, w in sorted(lv.composition, key=lambda kv: -kv[1])
                if w >= 0.01
            )
            occupancy = f"{lv.electrons}e" if lv.electrons else "  "
            print(f"     {lv.label:<10} {lv.energy:9.2f} eV  x{lv.degeneracy}"
                  f"  {occupancy:<4} {composition}")
        print("")

    write_crystal_diagram_html(diagram, entries, output_path, structure_label)
    print(f"Crystal-orbital diagram written to {output_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Crystal-orbital diagram from symmetry + extended-Hueckel "
        "overlap."
    )
    parser.add_argument("--poscar", default="POSCAR")
    parser.add_argument("--co-left", nargs="+", required=True,
                        metavar="FORMULA",
                        help="left fragment sublattice, e.g. SrTi")
    parser.add_argument("--co-right", nargs="+", required=True,
                        metavar="FORMULA",
                        help="right fragment sublattice, e.g. O3")
    parser.add_argument("--oxidation", nargs="+", default=None,
                        metavar="EL=Q",
                        help="formal oxidation states for the removed-"
                        "sublattice point charges, e.g. Sr=+2 Ti=+4 O=-2 "
                        "(default: guessed with pymatgen)")
    parser.add_argument("--kpoint", default=None,
                        help="restrict to one special k point label (e.g. GM)")
    parser.add_argument("--electrons", type=float, default=None,
                        help="electrons per primitive cell "
                        "(default: neutral-atom valence counts)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args(argv)

    from phonopy.interface.calculator import read_crystal_structure

    cell, _ = read_crystal_structure(args.poscar, interface_mode="vasp")
    from pathlib import Path

    stem = Path(args.poscar).name
    for extension in (".vasp", ".poscar"):
        if stem.lower().endswith(extension):
            stem = stem[: -len(extension)]
    output_path = args.output or f"CrystOD_{stem}.html"
    report_and_write(
        cell,
        left=args.co_left,
        right=args.co_right,
        symprec=args.tolerance,
        electrons=args.electrons,
        kpoint_filter=args.kpoint,
        output_path=output_path,
        structure_label=stem,
        oxidation=(parse_oxidation_tokens(args.oxidation)
                   if args.oxidation else None),
    )


if __name__ == "__main__":
    main()
