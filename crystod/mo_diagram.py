"""Semi-quantitative molecular-orbital diagrams (crystod-mol --diagram).

Builds the qualitative MO diagram of a single-center molecule (one central
atom + surrounding ligands, e.g. NH3, CH4, SF6) from symmetry and overlap
alone -- no self-consistent quantum chemistry:

1. The molecular point group is detected and the ligand atomic orbitals are
   symmetry-adapted (SALCs) per irrep, reusing :mod:`crystod.molecular_salc`.
2. Every valence atomic orbital is a single-zeta Slater-type orbital (STO);
   the exponents and diagonal energies H_ii (valence-state ionization
   energies) are the standard extended-Hueckel parameters.
3. All two-center STO overlap integrals are evaluated exactly (numerical
   Gauss-Laguerre x Gauss-Legendre quadrature in prolate-spheroidal
   coordinates, machine precision) -- no neglect of ligand-ligand overlap.
4. Within each irrep block the generalized eigenvalue problem
   H C = S C E with the Wolfsberg-Helmholz off-diagonals
   H_ij = K S_ij (H_ii + H_jj)/2 (K = 1.75) is solved, so a large overlap
   integral directly produces a large bonding/antibonding splitting.
5. The result is written as an interactive HTML/SVG diagram (four columns:
   isolated ligand AOs | ligand-group SALCs | molecular orbitals | central
   atom AOs) with dashed correlation lines, electron filling and per-level
   composition.

References:
  M. Wolfsberg and L. Helmholz, J. Chem. Phys. 20, 837 (1952).
  R. Hoffmann, J. Chem. Phys. 39, 1397 (1963).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field

import numpy as np

from .decompose_irrep import get_character_table
from .molecular_salc import (
    _hm_symbol,
    _match_operations,
    _table_operations_cartesian,
    format_salc,
    get_permutation_matrices,
    get_symmetry,
    load_molecule,
    project_salcs,
)

WOLFSBERG_HELMHOLZ_K = 1.75

# Standard extended-Hueckel valence parameters (Hoffmann and successors):
# element -> list of (shell label, n, l, Slater exponent zeta, H_ii in eV).
EHT_PARAMETERS = {
    "H":  [("1s", 1, 0, 1.300, -13.6)],
    "Li": [("2s", 2, 0, 0.650, -5.4), ("2p", 2, 1, 0.650, -3.5)],
    "Be": [("2s", 2, 0, 0.975, -10.0), ("2p", 2, 1, 0.975, -6.0)],
    "B":  [("2s", 2, 0, 1.300, -15.2), ("2p", 2, 1, 1.300, -8.5)],
    "C":  [("2s", 2, 0, 1.625, -21.4), ("2p", 2, 1, 1.625, -11.4)],
    "N":  [("2s", 2, 0, 1.950, -26.0), ("2p", 2, 1, 1.950, -13.4)],
    "O":  [("2s", 2, 0, 2.275, -32.3), ("2p", 2, 1, 2.275, -14.8)],
    "F":  [("2s", 2, 0, 2.425, -40.0), ("2p", 2, 1, 2.425, -18.1)],
    "Na": [("3s", 3, 0, 0.733, -5.1), ("3p", 3, 1, 0.733, -3.0)],
    "Mg": [("3s", 3, 0, 1.100, -9.0), ("3p", 3, 1, 1.100, -4.5)],
    "Al": [("3s", 3, 0, 1.167, -12.3), ("3p", 3, 1, 1.167, -6.5)],
    "Si": [("3s", 3, 0, 1.383, -17.3), ("3p", 3, 1, 1.383, -9.2)],
    "P":  [("3s", 3, 0, 1.750, -18.6), ("3p", 3, 1, 1.300, -14.0)],
    "S":  [("3s", 3, 0, 2.122, -20.0), ("3p", 3, 1, 1.827, -13.3)],
    "Cl": [("3s", 3, 0, 2.183, -26.3), ("3p", 3, 1, 1.733, -14.2)],
}

VALENCE_ELECTRONS = {
    "H": 1, "Li": 1, "Be": 2, "B": 3, "C": 4, "N": 5, "O": 6, "F": 7,
    "Na": 1, "Mg": 2, "Al": 3, "Si": 4, "P": 5, "S": 6, "Cl": 7,
}

# Core shells (not treated explicitly, but counted in the MO numbering so
# that the labels match the photoelectron-spectroscopy convention,
# e.g. CH4: 1a1 = C 1s core, valence = 2a1 + 1t2).
CORE_SHELLS = {
    "H": [], "Li": ["1s"], "Be": ["1s"], "B": ["1s"], "C": ["1s"],
    "N": ["1s"], "O": ["1s"], "F": ["1s"],
    "Na": ["1s", "2s", "2p"], "Mg": ["1s", "2s", "2p"],
    "Al": ["1s", "2s", "2p"], "Si": ["1s", "2s", "2p"],
    "P": ["1s", "2s", "2p"], "S": ["1s", "2s", "2p"], "Cl": ["1s", "2s", "2p"],
}

P_LABELS = ["px", "py", "pz"]  # real-orbital order used by wigner_D_real(1)

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903


# ----------------------------------------------------- STO overlap integrals


def _slater_norm(n: int, zeta: float) -> float:
    """Radial normalization of N r^(n-1) e^(-zeta r)."""
    from math import factorial, sqrt

    return (2.0 * zeta) ** (n + 0.5) / sqrt(float(factorial(2 * n)))


_QUAD_CACHE: dict = {}


def _quadrature(n_lag: int = 40, n_leg: int = 48):
    key = (n_lag, n_leg)
    if key not in _QUAD_CACHE:
        t, wt = np.polynomial.laguerre.laggauss(n_lag)
        v, wv = np.polynomial.legendre.leggauss(n_leg)
        _QUAD_CACHE[key] = (t[:, None], wt[:, None], v[None, :], wv[None, :])
    return _QUAD_CACHE[key]


def sto_overlap_aligned(n1, l1, z1, n2, l2, z2, R, m) -> float:
    """Overlap of two real STOs with atom 1 at the origin and atom 2 at
    (0, 0, R), both orbitals quantized along the global z axis.

    ``m = 0``: sigma overlap (s or p_z); ``m = 1``: pi overlap (p_x | p_x).
    Distances in Bohr. Exact within quadrature (machine precision).
    """
    if m == 1 and (l1 == 0 or l2 == 0):
        return 0.0
    t, wt, v, wv = _quadrature()
    alpha = 0.5 * R * (z1 + z2)
    beta = 0.5 * R * (z1 - z2)
    mu = 1.0 + t / alpha  # Gauss-Laguerre: absorbs e^(-alpha(mu-1))
    nu = v
    r1 = 0.5 * R * (mu + nu)
    r2 = 0.5 * R * (mu - nu)
    integrand = (
        r1 ** (n1 - 1)
        * r2 ** (n2 - 1)
        * np.exp(-beta * nu)
        * (R ** 3 / 8.0)
        * (mu ** 2 - nu ** 2)
        / alpha
    )
    cos1 = (1.0 + mu * nu) / (mu + nu)
    cos2 = (mu * nu - 1.0) / (mu - nu)
    c_p = np.sqrt(3.0 / (4.0 * np.pi))
    c_s = np.sqrt(1.0 / (4.0 * np.pi))
    if m == 0:
        ang1 = c_p * cos1 if l1 == 1 else c_s
        ang2 = c_p * cos2 if l2 == 1 else c_s
        phi = 2.0 * np.pi
    else:
        sin1 = np.sqrt(np.clip(1.0 - cos1 ** 2, 0.0, None))
        sin2 = np.sqrt(np.clip(1.0 - cos2 ** 2, 0.0, None))
        ang1, ang2 = c_p * sin1, c_p * sin2
        phi = np.pi
    value = float(np.sum(wt * wv * integrand * ang1 * ang2))
    return _slater_norm(n1, z1) * _slater_norm(n2, z2) * phi * np.exp(-alpha) * value


# ------------------------------------------------------------ AO basis and S


@dataclass
class AtomicOrbital:
    atom: int          # site index in the molecule
    element: str
    shell: str         # e.g. "2s", "2p"
    n: int
    l: int
    m: int             # index into P_LABELS for l = 1, 0 for l = 0
    zeta: float
    h_ii: float        # VSIP / diagonal Hamiltonian (eV)

    @property
    def orbital_label(self) -> str:
        return self.shell if self.l == 0 else self.shell[:-1] + P_LABELS[self.m]


def build_basis(symbols: list[str], site_indices: list[int]) -> list[AtomicOrbital]:
    orbitals = []
    for i in site_indices:
        element = symbols[i]
        if element not in EHT_PARAMETERS:
            raise SystemExit(
                f"ERROR: no extended-Hueckel parameters for element {element} "
                f"(supported: {', '.join(EHT_PARAMETERS)})."
            )
        for shell, n, l, zeta, h_ii in EHT_PARAMETERS[element]:
            for m in range(2 * l + 1):
                orbitals.append(AtomicOrbital(i, element, shell, n, l, m, zeta, h_ii))
    return orbitals


def overlap_matrix(orbitals: list[AtomicOrbital], coordinates: np.ndarray) -> np.ndarray:
    """AO overlap matrix (Slater-Koster assembly of the aligned integrals)."""
    n_ao = len(orbitals)
    S = np.eye(n_ao)
    pair_cache: dict = {}

    def aligned(a: AtomicOrbital, b: AtomicOrbital, R: float, m: int) -> float:
        key = (a.element, a.shell, b.element, b.shell, round(R, 10), m)
        if key not in pair_cache:
            pair_cache[key] = sto_overlap_aligned(
                a.n, a.l, a.zeta, b.n, b.l, b.zeta, R, m
            )
        return pair_cache[key]

    for i in range(n_ao):
        for j in range(i + 1, n_ao):
            a, b = orbitals[i], orbitals[j]
            if a.atom == b.atom:
                S[i, j] = S[j, i] = 0.0 if (a.shell, a.m) != (b.shell, b.m) else 1.0
                continue
            vector = (coordinates[b.atom] - coordinates[a.atom]) * ANGSTROM_TO_BOHR
            R = float(np.linalg.norm(vector))
            direction = vector / R
            if a.l == 0 and b.l == 0:
                value = aligned(a, b, R, 0)
            elif a.l == 0 and b.l == 1:
                value = direction[b.m] * aligned(a, b, R, 0)
            elif a.l == 1 and b.l == 0:
                value = direction[a.m] * aligned(a, b, R, 0)
            else:
                sigma = aligned(a, b, R, 0)
                pi = aligned(a, b, R, 1)
                delta = 1.0 if a.m == b.m else 0.0
                value = (
                    direction[a.m] * direction[b.m] * sigma
                    + (delta - direction[a.m] * direction[b.m]) * pi
                )
            S[i, j] = S[j, i] = value
    return S


def hamiltonian_matrix(orbitals: list[AtomicOrbital], S: np.ndarray) -> np.ndarray:
    h = np.array([orbital.h_ii for orbital in orbitals])
    H = 0.5 * WOLFSBERG_HELMHOLZ_K * (h[:, None] + h[None, :]) * S
    np.fill_diagonal(H, h)
    return H


# ------------------------------------------------------------ symmetry setup


@dataclass
class FragmentShell:
    """One (fragment, shell) block: SALCs of one orbital shell on one set of
    symmetry-equivalent sites."""

    name: str                    # e.g. "3H (1s)" or "N 2p"
    element: str
    shell: str
    l: int
    h_ii: float
    sites: list[int]
    is_center: bool
    salcs: dict = field(default_factory=dict)   # irrep -> list of AO-space vectors
    site_labels: list[str] = field(default_factory=list)


def _ao_space_vector(vector, shell: FragmentShell, ao_index, n_ao: int) -> np.ndarray:
    """Map a (site x orbital) SALC vector onto the full AO basis."""
    result = np.zeros(n_ao)
    width = 2 * shell.l + 1
    for k, site in enumerate(shell.sites):
        for m in range(width):
            result[ao_index[(site, shell.shell, m)]] = vector[k * width + m]
    return result


def _generalized_eigh(H: np.ndarray, S: np.ndarray):
    """Solve H C = S C E for a (small) positive-definite S."""
    values, vectors = np.linalg.eigh(S)
    if values.min() < 1e-10:
        raise SystemExit("ERROR: linearly dependent symmetry-adapted basis.")
    X = vectors @ np.diag(values ** -0.5) @ vectors.T
    energies, transformed = np.linalg.eigh(X @ H @ X)
    return energies, X @ transformed


def _group_degenerate(energies, tol: float = 1e-4):
    """Indices of degenerate groups (list of lists), energies ascending."""
    groups: list[list[int]] = []
    for i in np.argsort(energies):
        if groups and abs(energies[groups[-1][0]] - energies[i]) < tol:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    return groups


# ---------------------------------------------------------------- MO levels


@dataclass
class Level:
    """One horizontal level of the diagram (possibly degenerate)."""

    column: str          # "ligand-ao" | "salc" | "mo" | "center-ao"
    energy: float        # eV
    degeneracy: int      # number of orbitals drawn side by side
    irrep: str | None    # Mulliken symbol (table capitalization) or None
    label: str           # display label, e.g. "2a1", "a1", "2p"
    electrons: int = 0
    composition: list = field(default_factory=list)  # (level_id, weight)
    level_id: str = ""
    detail: str = ""     # tooltip text
    vectors: list = field(default_factory=list)  # AO-space partner vectors


def lowercase_irrep(name: str) -> str:
    return name[0].lower() + name[1:] if name else name


class MODiagram:
    """Symmetry + overlap MO diagram of a single-center molecule."""

    def __init__(self, xyz_path: str, tolerance: float = 0.3,
                 center_element: str | None = None):
        self.xyz_path = xyz_path
        self.tolerance = tolerance
        self.molecule = load_molecule(xyz_path)
        self.formula = self.molecule.composition.reduced_formula
        schoenflies, operations = get_symmetry(self.molecule, tolerance)
        self.schoenflies = schoenflies
        self.hm = _hm_symbol(schoenflies)
        if self.hm is None:
            raise SystemExit(
                "ERROR: the MO diagram supports the 32 crystallographic point "
                f"groups; this molecule's group is {schoenflies}."
            )
        self.character_table = get_character_table(self.hm)
        table_ops, table_classes = _table_operations_cartesian(self.character_table)
        alignment, matched = _match_operations(operations, table_ops, table_classes)
        # standard point-group frame (as --align): exact table operations
        self.operations = [table_ops[i] for i in matched]
        self.operation_classes = [table_classes[i] for i in matched]
        self.symbols = [site.specie.symbol for site in self.molecule]
        self.coordinates = (
            np.array([site.coords for site in self.molecule]) @ alignment.T
        )
        self._symmetrize_coordinates()
        self._identify_fragments(center_element)
        # conventional formula: central atom first (CH4, NH3, SF6, ...);
        # the group-16 hydrides are conventionally written hydrogen-first
        self.formula = self.symbols[self.center] + "".join(
            f"{element}{len(sites)}" if len(sites) > 1 else element
            for element, sites in sorted(self.ligand_sites.items())
        )
        self.formula = {"OH2": "H2O", "SH2": "H2S", "SeH2": "H2Se"}.get(
            self.formula, self.formula
        )
        self._build_matrices()
        self._project_fragment_salcs()
        self._count_core_shells()
        self._solve()

    def _symmetrize_coordinates(self) -> None:
        """Average the (numerically noisy) input geometry over the group orbit
        so that symmetry-required degeneracies are exact."""
        permutations = get_permutation_matrices(
            self.operations, self.coordinates, self.tolerance
        )
        symmetrized = np.zeros_like(self.coordinates)
        for rotation, permutation in zip(self.operations, permutations):
            # P[i, j] = 1 when g maps site j onto site i, so g^-1 x_i -> x_j
            symmetrized += (permutation.T @ self.coordinates) @ rotation
        self.coordinates = symmetrized / len(self.operations)

    # -------------------------------------------------- fragment recognition

    def _identify_fragments(self, center_element: str | None) -> None:
        distances = np.linalg.norm(self.coordinates, axis=1)
        if center_element is not None:
            candidates = [
                i for i, s in enumerate(self.symbols) if s == center_element
            ]
            if len(candidates) != 1:
                raise SystemExit(
                    f"ERROR: --center {center_element} must select exactly one "
                    f"atom (found {len(candidates)})."
                )
            self.center = candidates[0]
        else:
            self.center = int(np.argmin(distances))
            others = np.delete(distances, self.center)
            if len(others) and distances[self.center] > 0.5 * others.min():
                raise SystemExit(
                    "ERROR: could not identify a unique central atom; specify "
                    "it with --center EL (single-center molecules only)."
                )
        self.ligand_sites: dict[str, list[int]] = {}
        for i, symbol in enumerate(self.symbols):
            if i != self.center:
                self.ligand_sites.setdefault(symbol, []).append(i)
        if not self.ligand_sites:
            raise SystemExit("ERROR: the molecule has no ligand atoms.")

    # ------------------------------------------------------- matrices, SALCs

    def _build_matrices(self) -> None:
        self.orbitals = build_basis(self.symbols, list(range(len(self.symbols))))
        self.ao_index = {
            (ao.atom, ao.shell, ao.m): i for i, ao in enumerate(self.orbitals)
        }
        self.S = overlap_matrix(self.orbitals, self.coordinates)
        self.H = hamiltonian_matrix(self.orbitals, self.S)
        self.n_electrons = sum(VALENCE_ELECTRONS[s] for s in self.symbols)

    def _project_fragment_salcs(self) -> None:
        n_ao = len(self.orbitals)
        self.fragment_shells: list[FragmentShell] = []
        center_symbol = self.symbols[self.center]

        def make(element, sites, is_center):
            coordinates = self.coordinates[sites]
            permutations = get_permutation_matrices(
                self.operations, coordinates, self.tolerance
            )
            count = len(sites)
            for shell, _n, l, _zeta, h_ii in EHT_PARAMETERS[element]:
                if is_center:
                    name = f"{element} {shell}"
                else:
                    name = f"{count}{element} {shell}" if count > 1 else f"{element} {shell}"
                fragment = FragmentShell(
                    name, element, shell, l, h_ii, list(sites), is_center
                )
                fragment.site_labels = [
                    f"{element}{k + 1}" for k in range(count)
                ] if not is_center else [element]
                raw = project_salcs(
                    self.operations, self.operation_classes, permutations,
                    l, self.character_table,
                )
                fragment.salcs = {
                    irrep: [
                        _ao_space_vector(v, fragment, self.ao_index, n_ao)
                        for v in vectors
                    ]
                    for irrep, vectors in raw.items()
                }
                self.fragment_shells.append(fragment)

        for element, sites in sorted(self.ligand_sites.items()):
            make(element, sites, False)
        make(center_symbol, [self.center], True)

    def _count_core_shells(self) -> None:
        """Number of core levels per irrep (for the MO numbering only).

        A core shell of angular momentum l decomposes into the same irreps as
        the corresponding valence shell of the same fragment, so the valence
        SALC structure is reused for the counting.
        """
        rotation_list = list(self.character_table["rotation_list"])
        e_index = rotation_list.index("E")
        dimension = {
            irrep: int(round(float(np.real(np.atleast_1d(chars)[e_index]))))
            for irrep, chars in self.character_table["character_table"].items()
        }
        self.core_counts: dict[str, int] = {}
        self.core_summary: list[str] = []
        by_key = {(f.element, f.is_center, f.l): f for f in self.fragment_shells}
        counted = set()
        for fragment in self.fragment_shells:
            if (fragment.element, fragment.is_center) in counted:
                continue
            counted.add((fragment.element, fragment.is_center))
            for core_shell in CORE_SHELLS[fragment.element]:
                l = 0 if core_shell.endswith("s") else 1
                proxy = by_key.get((fragment.element, fragment.is_center, l))
                if proxy is None:
                    continue
                irreps = []
                for irrep, vectors in proxy.salcs.items():
                    count = len(vectors) // dimension[irrep]
                    self.core_counts[irrep] = self.core_counts.get(irrep, 0) + count
                    irreps.extend([lowercase_irrep(irrep)] * count)
                origin = (
                    f"{fragment.element} {core_shell}" if fragment.is_center
                    else f"{len(fragment.sites)}{fragment.element} {core_shell}"
                )
                self.core_summary.append(f"{origin} -> {' + '.join(sorted(irreps))}")

    # ------------------------------------------------------------ the solve

    def _solve(self) -> None:
        """Per-irrep generalized eigenproblems; build all diagram levels."""
        self.levels: dict[str, Level] = {}
        self.mo_levels: list[Level] = []
        self.salc_levels: list[Level] = []
        self.center_levels: list[Level] = []
        self.ligand_ao_levels: list[Level] = []
        self.irrep_blocks: list[dict] = []  # per-irrep report data

        # leftmost / rightmost columns: isolated atomic levels
        for fragment in self.fragment_shells:
            if fragment.is_center:
                continue
            key = ("ligand-ao", fragment.element, fragment.shell)
            if key not in self.levels:
                level = Level(
                    "ligand-ao", fragment.h_ii, 2 * fragment.l + 1, None,
                    fragment.name, level_id=f"lig_{fragment.element}_{fragment.shell}",
                    detail=f"isolated {fragment.element} {fragment.shell} "
                           f"(H_ii = {fragment.h_ii:.1f} eV)",
                )
                self.levels[key] = level
                self.ligand_ao_levels.append(level)
        for fragment in self.fragment_shells:
            if not fragment.is_center:
                continue
            irreps = sorted(fragment.salcs)
            annotation = " + ".join(lowercase_irrep(i) for i in irreps)
            level = Level(
                "center-ao", fragment.h_ii, 2 * fragment.l + 1, None,
                f"{fragment.shell} ({annotation})",
                level_id=f"cen_{fragment.shell}",
                detail=f"{fragment.element} {fragment.shell} "
                       f"(H_ii = {fragment.h_ii:.1f} eV), irreps: {annotation}",
            )
            for m in range(2 * fragment.l + 1):
                unit = np.zeros(len(self.orbitals))
                unit[self.ao_index[(self.center, fragment.shell, m)]] = 1.0
                level.vectors.append(unit)
            self.levels[("center-ao", fragment.shell)] = level
            self.center_levels.append(level)

        irrep_names = list(self.character_table["character_table"])
        mo_records = []  # (energy, irrep, degeneracy, composition, columns)
        for irrep in irrep_names:
            ligand_columns, ligand_owner = [], []
            center_columns, center_owner = [], []
            for fragment in self.fragment_shells:
                for vector in fragment.salcs.get(irrep, []):
                    if fragment.is_center:
                        center_columns.append(vector)
                        center_owner.append(fragment)
                    else:
                        ligand_columns.append(vector)
                        ligand_owner.append(fragment)
            if not ligand_columns and not center_columns:
                continue

            block = {"irrep": irrep, "salc_levels": [], "center_shells": [],
                     "overlaps": [], "mo_groups": []}

            # ligand-group SALC levels: ligand-only eigenproblem
            basis_columns, basis_level_ids = [], []
            if ligand_columns:
                Bl = np.array(ligand_columns).T
                Sl = Bl.T @ self.S @ Bl
                Hl = Bl.T @ self.H @ Bl
                energies_l, C_l = _generalized_eigh(Hl, Sl)
                Ll = Bl @ C_l  # S-orthonormal ligand levels in AO space
                for group in _group_degenerate(energies_l):
                    energy = float(energies_l[group[0]])
                    # shell composition of this ligand level (Mulliken)
                    weights: dict[str, float] = {}
                    for column in group:
                        c = C_l[:, column]
                        mulliken = c * (Sl @ c)
                        for w, fragment in zip(mulliken, ligand_owner):
                            weights[fragment.name] = weights.get(fragment.name, 0.0) + float(w)
                    total = sum(weights.values())
                    shares = {k: v / total for k, v in weights.items()}
                    dominant = max(shares, key=shares.get)
                    level_id = f"salc_{irrep}_{len([l for l in self.salc_levels if l.irrep == irrep])}"
                    detail = ", ".join(
                        f"{100 * share:.0f}% {name}"
                        for name, share in sorted(shares.items(), key=lambda kv: -kv[1])
                        if share > 0.005
                    )
                    level = Level(
                        "salc", energy, len(group), irrep,
                        lowercase_irrep(irrep), level_id=level_id,
                        detail=f"ligand SALC {lowercase_irrep(irrep)}: {detail}",
                    )
                    # correlation to the isolated-AO column
                    for name, share in shares.items():
                        fragment = next(f for f in self.fragment_shells if f.name == name)
                        level.composition.append(
                            (f"lig_{fragment.element}_{fragment.shell}", share)
                        )
                    level.vectors = [Ll[:, c] for c in group]
                    self.salc_levels.append(level)
                    block["salc_levels"].append(
                        (level, level.vectors)
                    )
                    for column in group:
                        basis_columns.append(Ll[:, column])
                        basis_level_ids.append(level.level_id)

            for vector, fragment in zip(center_columns, center_owner):
                basis_columns.append(vector)
                basis_level_ids.append(f"cen_{fragment.shell}")
            center_shells_here = []
            for fragment in center_owner:
                if fragment.shell not in center_shells_here:
                    center_shells_here.append(fragment.shell)
            block["center_shells"] = center_shells_here

            # inter-fragment overlap report (SALC level | central shell)
            for level, level_columns in block["salc_levels"]:
                for shell in center_shells_here:
                    columns = [
                        v for v, f in zip(center_columns, center_owner)
                        if f.shell == shell
                    ]
                    M = np.array(
                        [[u @ self.S @ v for v in columns] for u in level_columns]
                    )
                    value = float(np.linalg.svd(M, compute_uv=False)[0]) if M.size else 0.0
                    block["overlaps"].append((level, shell, value))

            # full block: ligand levels + central AOs
            B = np.array(basis_columns).T
            Sb = B.T @ self.S @ B
            Hb = B.T @ self.H @ B
            energies, C = _generalized_eigh(Hb, Sb)
            for group in _group_degenerate(energies):
                energy = float(energies[group[0]])
                weights: dict[str, float] = {}
                for column in group:
                    c = C[:, column]
                    mulliken = c * (Sb @ c)
                    for w, level_id in zip(mulliken, basis_level_ids):
                        weights[level_id] = weights.get(level_id, 0.0) + float(w)
                total = sum(weights.values())
                composition = [
                    (level_id, w / total) for level_id, w in weights.items()
                ]
                vectors = [B @ C[:, k] for k in group]
                mo_records.append((energy, irrep, len(group), composition, vectors))
                block["mo_groups"].append((energy, len(group), composition))
            self.irrep_blocks.append(block)

        # number the ligand SALC levels per irrep when there is more than one
        from collections import Counter
        salc_irrep_counts = Counter(level.irrep for level in self.salc_levels)
        salc_counter: dict[str, int] = {}
        for level in sorted(self.salc_levels, key=lambda l: l.energy):
            if salc_irrep_counts[level.irrep] > 1:
                salc_counter[level.irrep] = salc_counter.get(level.irrep, 0) + 1
                level.label = f"{salc_counter[level.irrep]}{lowercase_irrep(level.irrep)}"

        # electron filling and MO labels (numbering counts the core shells)
        mo_records.sort(key=lambda record: record[0])
        remaining = self.n_electrons
        irrep_counter: dict[str, int] = dict(self.core_counts)
        for energy, irrep, degeneracy, composition, vectors in mo_records:
            irrep_counter[irrep] = irrep_counter.get(irrep, 0) + 1
            label = f"{irrep_counter[irrep]}{lowercase_irrep(irrep)}"
            electrons = int(min(remaining, 2 * degeneracy))
            remaining -= electrons
            level = Level(
                "mo", energy, degeneracy, irrep, label,
                electrons=electrons,
                composition=composition,
                level_id=f"mo_{len(self.mo_levels)}",
                vectors=vectors,
            )
            parts = ", ".join(
                f"{100 * w:.0f}% {self._level_name(i)}"
                for i, w in sorted(composition, key=lambda kv: -kv[1])
                if w > 0.005
            )
            level.detail = (
                f"{label}: E = {energy:.2f} eV, {electrons} e-  |  {parts}"
            )
            self.mo_levels.append(level)
            self.levels[("mo", level.level_id)] = level

        occupied = [l for l in self.mo_levels if l.electrons > 0]
        empty = [l for l in self.mo_levels if l.electrons == 0]
        self.homo = occupied[-1] if occupied else None
        self.lumo = empty[0] if empty else None

    def _level_name(self, level_id: str) -> str:
        for level in (self.salc_levels + self.center_levels + self.ligand_ao_levels):
            if level.level_id == level_id:
                if level.column == "salc":
                    return f"SALC {level.label}"
                if level.column == "center-ao":
                    return f"{self.symbols[self.center]} {level.label.split()[0]}"
                return level.label
        return level_id

    # ------------------------------------------------------------- reporting

    def print_report(self) -> None:
        table = self.character_table
        print("\n* Molecule *")
        print(f"{self.xyz_path} ({self.formula}, {len(self.symbols)} atoms)")
        print("\n* Point group *")
        print(f"{self.schoenflies} (Hermann-Mauguin: {self.hm})")
        center_symbol = self.symbols[self.center]
        ligands = ", ".join(
            f"{len(sites)} {element}" for element, sites in sorted(self.ligand_sites.items())
        )
        print("\n* Fragments *")
        print(f"central atom: {center_symbol}; ligands: {ligands}")

        print("\n* Valence AO parameters (single-zeta STO, extended Hueckel) *")
        seen = set()
        for element in [center_symbol] + sorted(self.ligand_sites):
            if element in seen:
                continue
            seen.add(element)
            for shell, n, l, zeta, h_ii in EHT_PARAMETERS[element]:
                print(f"{element} {shell}:  zeta = {zeta:.3f} / bohr,  H_ii = {h_ii:.1f} eV")

        print("\n* Ligand SALCs (standard point-group axes) *")
        for fragment in self.fragment_shells:
            if fragment.is_center:
                continue
            width = 2 * fragment.l + 1
            orbital_names = [fragment.shell] if fragment.l == 0 else [
                fragment.shell[:-1] + p for p in P_LABELS
            ]
            term_labels = [
                f"{orbital_names[m]}({label})"
                for label in fragment.site_labels
                for m in range(width)
            ]
            print(f"-- {fragment.name} --")
            for irrep in table["character_table"]:
                if irrep not in fragment.salcs:
                    continue
                site_major = []
                for vector in fragment.salcs[irrep]:
                    compact = np.zeros(len(term_labels))
                    for k, site in enumerate(fragment.sites):
                        for m in range(width):
                            compact[k * width + m] = vector[
                                self.ao_index[(site, fragment.shell, m)]
                            ]
                    site_major.append(compact)
                formatted = ", ".join(format_salc(v, term_labels) for v in site_major)
                print(f"{irrep}: [{formatted}]")

        print("\n* Ligand SALC | central AO overlap integrals *")
        any_overlap = False
        for block in self.irrep_blocks:
            for level, shell, value in block["overlaps"]:
                any_overlap = True
                print(
                    f"{block['irrep']:>4}:  < {level.label} (E = {level.energy:7.2f} eV)"
                    f" | {center_symbol} {shell} >  S = {value:.4f}"
                )
        if not any_overlap:
            print("(none: no irrep is shared by the ligand SALCs and the central AOs)")

        print("\n* Molecular orbitals (Wolfsberg-Helmholz, K = 1.75) *")
        print(f"{'MO':>6} {'E (eV)':>9} {'occ':>4}  composition")
        for level in reversed(self.mo_levels):
            parts = ", ".join(
                f"{100 * w:.0f}% {self._level_name(i)}"
                for i, w in sorted(level.composition, key=lambda kv: -kv[1])
                if w > 0.005
            )
            degeneracy = f" x{level.degeneracy}" if level.degeneracy > 1 else ""
            print(f"{level.label + degeneracy:>6} {level.energy:9.2f} {level.electrons:>4}  {parts}")

        print(f"\n* Electron filling ({self.n_electrons} valence electrons) *")
        configuration = " ".join(
            f"({level.label})^{level.electrons}"
            for level in self.mo_levels if level.electrons > 0
        )
        print(configuration)
        if self.core_summary:
            print("(MO numbering counts the core shells, not shown: "
                  + "; ".join(self.core_summary) + ")")
        if self.homo and self.lumo:
            print(
                f"HOMO = {self.homo.label} ({self.homo.energy:.2f} eV), "
                f"LUMO = {self.lumo.label} ({self.lumo.energy:.2f} eV), "
                f"gap = {self.lumo.energy - self.homo.energy:.2f} eV"
            )

        print("\nMethod: symmetry-adapted extended Hueckel over single-zeta STOs")
        print("(exact two-center overlaps; energies are semi-quantitative).")
        print("M. Wolfsberg and L. Helmholz, J. Chem. Phys. 20, 837 (1952);")
        print("R. Hoffmann, J. Chem. Phys. 39, 1397 (1963).")

    # ---------------------------------------------------------- HTML diagram

    def write_html(self, output_path: str) -> None:
        write_diagram_html(self, output_path)


# ------------------------------------------------------- orbital sketches

COVALENT_RADII = {
    "H": 0.31, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71,
    "O": 0.66, "F": 0.57, "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11,
    "P": 1.07, "S": 1.05, "Cl": 1.02,
}


def diagram_geometry(symbols: list[str], coordinates: np.ndarray) -> dict:
    """Atoms (with VESTA colors) and bonds for the in-panel orbital sketch."""
    colors_path = os.path.join(os.path.dirname(__file__), "vesta_element_rgb.json")
    try:
        with open(colors_path) as handle:
            rgb = json.load(handle)
    except OSError:
        rgb = {}
    center = np.asarray(coordinates, dtype=float)
    center = center - center.mean(axis=0)
    atoms = []
    for symbol, position in zip(symbols, center):
        color = rgb.get(symbol)
        color_hex = (
            "#{:02x}{:02x}{:02x}".format(*color) if color else "#9e9e9e"
        )
        atoms.append([symbol, round(float(position[0]), 4),
                      round(float(position[1]), 4), round(float(position[2]), 4),
                      color_hex])
    bonds = []
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            cutoff = 1.25 * (COVALENT_RADII.get(symbols[i], 0.8)
                             + COVALENT_RADII.get(symbols[j], 0.8))
            if np.linalg.norm(center[i] - center[j]) < cutoff:
                bonds.append([i, j])
    radius = float(np.max(np.linalg.norm(center, axis=1))) if len(center) else 1.0
    return {"atoms": atoms, "bonds": bonds, "radius": max(radius, 0.8)}


def canonical_sketch_partners(per_atom_list: list[dict]) -> list[dict]:
    """Canonicalize the compressed partner amplitudes of a degenerate level.

    The partners of an exactly degenerate level are defined only up to an
    orthogonal mixture; the SCF (or eigh) returns an arbitrary one. Applying
    the same RREF + Gram-Schmidt canonicalization as the SALC viewer makes
    the displayed partners match the visualized SALCs
    (crystod-mol --visualize)."""
    if len(per_atom_list) < 2:
        return per_atom_list
    from .molecular_salc import _rref_orthogonal

    atoms = sorted(set().union(*[set(d) for d in per_atom_list]))
    rows = np.array([
        np.concatenate([
            np.asarray(d.get(a, [0.0, 0.0, 0.0, 0.0]), dtype=float) for a in atoms
        ])
        for d in per_atom_list
    ])
    # drop negligible feature columns (e.g. tiny polarization tails) so the
    # RREF pivots land on the same dominant components as the SALC viewer
    peak = np.max(np.abs(rows)) or 1.0
    rows[:, np.max(np.abs(rows), axis=0) < 0.05 * peak] = 0.0
    canonical = _rref_orthogonal(list(rows))
    if len(canonical) != len(rows):
        return per_atom_list
    result = []
    for row in canonical:
        per_atom = {}
        for k, a in enumerate(atoms):
            values = row[4 * k:4 * k + 4]
            if np.max(np.abs(values)) > 1e-8:
                per_atom[a] = [float(v) for v in values]
        result.append(per_atom)
    return result


def _sketch_entries(per_atom: dict) -> list:
    """Compress {atom: [s, px, py, pz]} to sketch entries, normalized so the
    largest component is 1; atoms with negligible amplitude are dropped."""
    if not per_atom:
        return []
    peak = max(
        max(abs(value) for value in values) for values in per_atom.values()
    )
    if peak < 1e-8:
        return []
    entries = []
    for atom, values in sorted(per_atom.items()):
        scaled = [round(float(value) / peak, 3) for value in values]
        if max(abs(value) for value in scaled) >= 0.04:
            entries.append([atom] + scaled)
    return entries


# --------------------------------------------------------------- SVG diagram


_DIAGRAM_SCRIPT = r"""
const CFG = __CONFIG__;
const LEVELS = __LEVELS__;
const GEOM = __GEOM__;
const byId = {};
LEVELS.forEach(l => byId[l.id] = l);
let eMin = CFG.eMin, eMax = CFG.eMax;
const NS = 'http://www.w3.org/2000/svg';
const gGrid = document.getElementById('gGrid');
const gCon = document.getElementById('gCon');
const gLvl = document.getElementById('gLvl');
const svg = document.getElementById('diagram');

function el(name, attrs, text) {
  const node = document.createElementNS(NS, name);
  for (const k in attrs) node.setAttribute(k, attrs[k]);
  if (text !== undefined) node.textContent = text;
  return node;
}

// Mulliken label with subscripts, e.g. 2a1g -> 2a_(1g); also 1σg -> 1σ_(g)
function mulliken(node, text) {
  const re = /[σπδφ]/.test(text)
    ? /(\d*)([σπδφ])([gu]?[+-]?)/g
    : /\b(\d*)([abet])([123]?[gu]?'{0,2})\b/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) node.appendChild(document.createTextNode(text.slice(last, m.index)));
    node.appendChild(document.createTextNode(m[1] + m[2]));
    if (m[3]) {
      const sub = el('tspan', {'baseline-shift': 'sub', 'font-size': '9'});
      sub.textContent = m[3];
      node.appendChild(sub);
    }
    last = m.index + m[0].length;
  }
  if (last < text.length) node.appendChild(document.createTextNode(text.slice(last)));
}

function yOf(E) { return CFG.top + (eMax - E) / (eMax - eMin) * CFG.H; }

function segments(level) {
  const x = CFG.columns[level.col], h = CFG.half[level.col], d = level.deg;
  const seg = Math.min(2 * h, (2 * h + 8) / d - 5);
  const total = d * seg + (d - 1) * 5;
  const out = [];
  for (let k = 0; k < d; k++) {
    const x1 = x - total / 2 + k * (seg + 5);
    out.push([x1, x1 + seg]);
  }
  return out;
}

// push overlapping labels apart, keeping clusters centered
function nudge(values, gap) {
  const order = values.map((v, i) => i).sort((a, b) => values[a] - values[b]);
  const out = values.slice();
  for (let k = 1; k < order.length; k++) {
    const p = order[k - 1], c = order[k];
    if (out[c] < out[p] + gap) out[c] = out[p] + gap;
  }
  let cluster = order.length ? [order[0]] : [];
  const clusters = [];
  for (let k = 1; k < order.length; k++) {
    const p = order[k - 1], c = order[k];
    if (out[c] - out[p] < gap + 1e-6) cluster.push(c);
    else { clusters.push(cluster); cluster = [c]; }
  }
  if (cluster.length) clusters.push(cluster);
  clusters.forEach(members => {
    const off = members.reduce((s, i) => s + out[i] - values[i], 0) / members.length;
    members.forEach(i => out[i] -= off);
  });
  return out;
}

function tickStep(span) {
  for (const s of [0.5, 1, 2, 5, 10, 20, 50, 100])
    if (span / s <= 12) return s;
  return 200;
}

let sketchLevel = null, sketchPartner = 0;
let rotYaw = -0.6, rotPitch = 0.35;

function show(id) {
  const d = byId[id];
  if (!d) return;
  const head = document.createElement('h2');
  mullikenHtml(head, d.label);
  let sub = 'E = ' + d.e.toFixed(2) + ' eV';
  if (d.deg > 1) sub += '  (×' + d.deg + ')';
  if (d.el !== null) sub += '  •  ' + d.el + ' e−';
  let html = '<div class="sub">' + sub + '</div>';
  for (const [name, pct] of d.comp) {
    html += '<div class="cname">' + name + ' — ' + pct + '%</div>';
    html += '<div class="bar" style="width:' + Math.max(3, 2.1 * pct) + 'px"></div>';
  }
  if (d.orb && GEOM) {
    if (sketchLevel !== id) sketchPartner = 0;
    html += '<div id="onav"></div>';
    html += '<svg id="oview" viewBox="0 0 222 190" width="222" height="190"></svg>';
    html += '<div class="ohint">orbital sketch — drag to rotate</div>';
  }
  const body = document.getElementById('pbody');
  body.innerHTML = html;
  body.insertBefore(head, body.firstChild);
  if (d.orb && GEOM) {
    sketchLevel = id;
    setupSketch(d);
  }
}

function setupSketch(d) {
  const nav = document.getElementById('onav');
  nav.textContent = '';  // rebuildable: clear before adding partner buttons
  if (d.orb.length > 1) {
    for (let k = 0; k < d.orb.length; k++) {
      const b = document.createElement('button');
      b.textContent = k + 1;
      b.className = 'obtn' + (k === sketchPartner ? ' sel' : '');
      b.addEventListener('click', () => { sketchPartner = k; setupSketch(d); });
      nav.appendChild(b);
    }
    const note = document.createElement('span');
    note.className = 'ohint';
    note.textContent = ' degenerate partner';
    nav.appendChild(note);
  }
  const view = document.getElementById('oview');
  if (!view.dataset.bound) {  // attach the drag handlers only once
    view.dataset.bound = '1';
    let dragging = null;
    view.addEventListener('pointerdown', e => {
      dragging = [e.clientX, e.clientY];
      view.setPointerCapture(e.pointerId);
      e.stopPropagation();
    });
    view.addEventListener('pointermove', e => {
      if (!dragging) return;
      rotYaw += (e.clientX - dragging[0]) * 0.012;
      rotPitch += (e.clientY - dragging[1]) * 0.012;
      dragging = [e.clientX, e.clientY];
      drawSketch(byId[sketchLevel]);
    });
    view.addEventListener('pointerup', () => dragging = null);
  }
  drawSketch(d);
}

function drawSketch(d) {
  const view = document.getElementById('oview');
  if (!view) return;
  const cy2 = Math.cos(rotYaw), sy2 = Math.sin(rotYaw);
  const cp = Math.cos(rotPitch), sp = Math.sin(rotPitch);
  function rot(p) {
    const x1 = cy2 * p[0] + sy2 * p[2];
    const z1 = -sy2 * p[0] + cy2 * p[2];
    return [x1, cp * p[1] - sp * z1, sp * p[1] + cp * z1];
  }
  const scale = 78 / (GEOM.radius + 0.7);
  const cx = 111, cyc = 92;
  const pts = GEOM.atoms.map(a => rot([a[1], a[2], a[3]]));
  const prims = [];
  GEOM.bonds.forEach(([i, j]) => {
    prims.push([0.5 * (pts[i][2] + pts[j][2]) - 50,
      '<line x1="' + (cx + scale * pts[i][0]).toFixed(1) + '" y1="' + (cyc - scale * pts[i][1]).toFixed(1) +
      '" x2="' + (cx + scale * pts[j][0]).toFixed(1) + '" y2="' + (cyc - scale * pts[j][1]).toFixed(1) +
      '" stroke="#b0bec5" stroke-width="2.4"/>']);
  });
  GEOM.atoms.forEach((a, i) => {
    const r = a[0] === 'H' ? 3.2 : 5;
    prims.push([pts[i][2],
      '<circle cx="' + (cx + scale * pts[i][0]).toFixed(1) + '" cy="' + (cyc - scale * pts[i][1]).toFixed(1) +
      '" r="' + r + '" fill="' + a[4] + '" stroke="#546e7a" stroke-width="0.8"/>']);
  });
  const POS = '#e8c400', NEG = '#19b8d8';
  (d.orb[sketchPartner] || []).forEach(entry => {
    const i = entry[0], s = entry[1], p = [entry[2], entry[3], entry[4]];
    const ax = cx + scale * pts[i][0], ay = cyc - scale * pts[i][1];
    if (Math.abs(s) >= 0.04) {
      prims.push([pts[i][2] + 0.02,
        '<circle cx="' + ax.toFixed(1) + '" cy="' + ay.toFixed(1) +
        '" r="' + (3 + 13 * Math.abs(s)).toFixed(1) + '" fill="' + (s > 0 ? POS : NEG) +
        '" fill-opacity="0.55" stroke="' + (s > 0 ? POS : NEG) + '" stroke-opacity="0.9"/>']);
    }
    const pn = Math.hypot(p[0], p[1], p[2]);
    if (pn >= 0.04) {
      const v = rot(p);
      const L = 5 + 15 * pn;
      const sn = Math.hypot(v[0], v[1]) || 1e-6;
      const ux = v[0] / sn, uy = -v[1] / sn;
      const off = L * 0.62 * Math.min(1, sn / pn);
      const rl = (2.5 + L * 0.42).toFixed(1);
      prims.push([pts[i][2] + 0.02 + 0.3 * v[2] / (pn || 1),
        '<circle cx="' + (ax + off * ux).toFixed(1) + '" cy="' + (ay + off * uy).toFixed(1) +
        '" r="' + rl + '" fill="' + POS + '" fill-opacity="0.55" stroke="' + POS + '" stroke-opacity="0.9"/>']);
      prims.push([pts[i][2] + 0.02 - 0.3 * v[2] / (pn || 1),
        '<circle cx="' + (ax - off * ux).toFixed(1) + '" cy="' + (ay - off * uy).toFixed(1) +
        '" r="' + rl + '" fill="' + NEG + '" fill-opacity="0.55" stroke="' + NEG + '" stroke-opacity="0.9"/>']);
    }
  });
  prims.sort((a, b) => a[0] - b[0]);
  view.innerHTML = prims.map(p => p[1]).join('');
}

// HTML (panel) version of the subscript renderer
function mullikenHtml(node, text) {
  const re = /[σπδφ]/.test(text)
    ? /(\d*)([σπδφ])([gu]?[+-]?)/g
    : /\b(\d*)([abet])([123]?[gu]?'{0,2})\b/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    node.appendChild(document.createTextNode(text.slice(last, m.index) + m[1] + m[2]));
    if (m[3]) {
      const sub = document.createElement('sub');
      sub.textContent = m[3];
      node.appendChild(sub);
    }
    last = m.index + m[0].length;
  }
  node.appendChild(document.createTextNode(text.slice(last)));
}

function highlight(id, on) {
  gCon.querySelectorAll('.con').forEach(c => {
    const linked = c.classList.contains('l_' + id);
    c.classList.toggle('hi', on && linked);
    c.classList.toggle('dim', on && !linked);
  });
}

function render() {
  gGrid.textContent = '';
  gCon.textContent = '';
  gLvl.textContent = '';
  document.getElementById('emin').value = eMin.toFixed(1);
  document.getElementById('emax').value = eMax.toFixed(1);

  // gridlines and ticks
  const step = tickStep(eMax - eMin);
  for (let v = Math.ceil(eMin / step) * step; v <= eMax + 1e-9; v += step) {
    const y = yOf(v);
    gGrid.appendChild(el('line', {x1: CFG.left - 13, y1: y, x2: CFG.left - 3, y2: y, 'class': 'axis'}));
    gGrid.appendChild(el('text', {x: CFG.left - 17, y: y + 4, 'class': 'tick', 'text-anchor': 'end'},
                        String(Math.round(v * 100) / 100)));
    gGrid.appendChild(el('line', {x1: CFG.left + 2, y1: y, x2: CFG.width - 20, y2: y, 'class': 'grid'}));
  }

  const visible = LEVELS.filter(l => {
    const y = yOf(l.e);
    return y >= CFG.top - 6 && y <= CFG.top + CFG.H + 6;
  });
  const visIds = new Set(visible.map(l => l.id));

  // connectors (below the levels)
  visible.forEach(level => {
    const x = CFG.columns[level.col], y = yOf(level.e);
    (level.links || []).forEach(([sid, w]) => {
      const src = byId[sid];
      if (!src || !visIds.has(sid)) return;
      const sx = CFG.columns[src.col], sy = yOf(src.e);
      let x1, x2;
      if (sx < x) { x1 = sx + CFG.half[src.col]; x2 = x - CFG.half[level.col]; }
      else { x1 = sx - CFG.half[src.col]; x2 = x + CFG.half[level.col]; }
      const line = el('line', {x1: x1, y1: sy, x2: x2, y2: y,
                               'class': 'con l_' + level.id + ' l_' + sid});
      line.style.opacity = (0.2 + 0.55 * Math.min(1, w)).toFixed(2);
      gCon.appendChild(line);
    });
  });

  // levels with nudged labels per column
  CFG.order.forEach(col => {
    const levels = visible.filter(l => l.col === col);
    const rawY = levels.map(l => yOf(l.e));
    const labY = nudge(rawY, 13);
    const side = CFG.side[col];
    levels.forEach((level, i) => {
      const x = CFG.columns[col], y = rawY[i], h = CFG.half[col];
      const g = el('g', {'class': 'lvl', tabindex: 0});
      g.dataset.id = level.id;
      const css = level.el !== null ? (level.occ ? 'occ' : 'virt') : 'frag';
      segments(level).forEach(([x1, x2], k) => {
        g.appendChild(el('line', {x1: x1, y1: y, x2: x2, y2: y, 'class': 'seg ' + css}));
        if (level.el) {
          const d = level.deg;
          const ups = Math.min(level.el, d) > k ? 1 : 0;
          const downs = (level.el - Math.min(level.el, d)) > k ? 1 : 0;
          const arrows = '↑'.repeat(ups) + '↓'.repeat(downs);
          if (arrows) g.appendChild(el('text', {x: (x1 + x2) / 2, y: y - 3,
                                                'class': 'el', 'text-anchor': 'middle'}, arrows));
        }
      });
      const lx = x + side * (h + 9), ly = labY[i];
      if (Math.abs(ly - y) > 6)
        g.appendChild(el('line', {x1: x + side * h, y1: y, x2: lx - side * 2, y2: ly, 'class': 'lead'}));
      const text = el('text', {x: lx, y: ly + 4, 'class': 'lab',
                               'text-anchor': side > 0 ? 'start' : 'end'});
      mulliken(text, level.label);
      g.appendChild(text);
      const title = el('title', {});
      title.textContent = level.detail;
      g.appendChild(title);
      if (level.id === CFG.homo || level.id === CFG.lumo)
        g.appendChild(el('text', {x: lx + 40, y: ly + 4, 'class': 'hl'},
                         level.id === CFG.homo ? 'HOMO' : 'LUMO'));
      g.addEventListener('mouseenter', () => { highlight(level.id, true); show(level.id); });
      g.addEventListener('mouseleave', () => highlight(level.id, false));
      g.addEventListener('click', () => show(level.id));
      gLvl.appendChild(g);
    });
  });
}

function setRange(lo, hi) {
  if (!(isFinite(lo) && isFinite(hi)) || hi - lo < 0.5) return;
  eMin = lo; eMax = hi; render();
}

document.getElementById('emin').addEventListener('change', e =>
  setRange(parseFloat(e.target.value), eMax));
document.getElementById('emax').addEventListener('change', e =>
  setRange(eMin, parseFloat(e.target.value)));
document.getElementById('ereset').addEventListener('click', () =>
  setRange(CFG.eMin, CFG.eMax));

// Ctrl/Cmd + wheel (or trackpad pinch): zoom the energy axis around the cursor
svg.addEventListener('wheel', e => {
  if (!(e.ctrlKey || e.metaKey)) return;  // plain scroll keeps scrolling the page
  e.preventDefault();
  const box = svg.getBoundingClientRect();
  const my = (e.clientY - box.top) * (CFG.height / box.height);
  const Ec = eMax - (my - CFG.top) / CFG.H * (eMax - eMin);
  const f = e.deltaY > 0 ? 1.15 : 1 / 1.15;
  const lo = Ec + (eMin - Ec) * f, hi = Ec + (eMax - Ec) * f;
  if (hi - lo >= 0.5 && hi - lo <= 2000) setRange(lo, hi);
}, {passive: false});

// drag: pan the energy window
let dragY = null, dragMin = 0, dragMax = 0;
svg.addEventListener('pointerdown', e => {
  dragY = e.clientY; dragMin = eMin; dragMax = eMax;
  svg.setPointerCapture(e.pointerId);
});
svg.addEventListener('pointermove', e => {
  if (dragY === null) return;
  const box = svg.getBoundingClientRect();
  const dE = (e.clientY - dragY) * (CFG.height / box.height) / CFG.H * (dragMax - dragMin);
  eMin = dragMin + dE; eMax = dragMax + dE; render();
});
svg.addEventListener('pointerup', () => dragY = null);
svg.addEventListener('pointercancel', () => dragY = null);

render();
"""


def svg_sub_digits(text: str) -> str:
    """Digits after a letter as SVG subscripts (H4 -> H_4) for column headers."""
    import re

    return re.sub(
        r"(?<=[A-Za-z])(\d+)",
        r'<tspan baseline-shift="sub" font-size="10">\1</tspan>',
        text,
    )


def render_diagram_page(
    output_path: str,
    *,
    title: str,
    heading_html: str,
    chips: list[str],
    columns: dict[str, float],
    half: dict[str, float],
    order: list[str],
    side: dict[str, int],
    headers: dict[str, str],
    levels_json: list[dict],
    homo_id: str | None,
    lumo_id: str | None,
    e_min: float,
    e_max: float,
    foot_html: str,
    geometry: dict | None = None,
) -> None:
    """Write the standalone interactive MO-diagram page.

    Fully generic in the column layout: ``order`` lists the column keys left
    to right, ``columns``/``half`` give their x positions and level half
    widths, ``side`` the label side (+1 right, -1 left), ``headers`` the
    column titles (SVG-ready). The level records carry everything else; the
    embedded script renders the plot client-side so the energy window can be
    changed interactively (input boxes, Ctrl/Cmd + scroll zoom, drag pan).
    """
    width, height = 960, 780
    top, left = 84, 74
    plot_height = height - top - 36

    config = {
        "width": width, "height": height, "top": top, "left": left,
        "H": plot_height, "columns": columns, "half": half,
        "order": order, "side": side,
        "eMin": round(e_min, 2), "eMax": round(e_max, 2),
        "homo": homo_id, "lumo": lumo_id,
    }
    header_svg = "".join(
        f'<text x="{columns[column]}" y="{top - 40}" class="header" '
        f'text-anchor="middle">{text}</text>'
        for column, text in headers.items()
    )
    chip_html = "".join(f'<span class="chip">{chip}</span>' for chip in chips if chip)
    script = (
        _DIAGRAM_SCRIPT
        .replace("__CONFIG__", json.dumps(config))
        .replace("__LEVELS__", json.dumps(levels_json, ensure_ascii=False))
        .replace("__GEOM__", json.dumps(geometry))
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
 body {{ font-family: 'Helvetica Neue', Arial, sans-serif; margin: 0; background: #fafafa; color: #222; }}
 #page {{ max-width: 1280px; margin: 0 auto; padding: 14px 18px; }}
 h1 {{ font-size: 19px; margin: 4px 0 6px; font-weight: 600; }}
 .chip {{ display: inline-block; background: #eceff1; border-radius: 4px; padding: 2px 9px;
          margin: 0 6px 6px 0; font-size: 12.5px; color: #37474f; }}
 #flex {{ display: flex; gap: 14px; align-items: flex-start; flex-wrap: wrap; }}
 #controls {{ font-size: 12.5px; color: #37474f; margin: 2px 0 8px; }}
 #controls input {{ width: 62px; font-size: 12.5px; padding: 1px 4px;
                    border: 1px solid #b0bec5; border-radius: 3px; }}
 #controls button {{ font-size: 12px; padding: 2px 10px; margin-left: 6px;
                     border: 1px solid #b0bec5; border-radius: 3px;
                     background: #eceff1; cursor: pointer; }}
 #controls .hint {{ color: #90a4ae; margin-left: 10px; }}
 svg {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 6px;
        touch-action: none; cursor: ns-resize; }}
 #panel {{ width: 250px; background: #fff; border: 1px solid #e0e0e0; border-radius: 6px;
           padding: 12px 14px; font-size: 13px; min-height: 120px; }}
 #panel h2 {{ font-size: 14px; margin: 0 0 4px; }}
 #panel .sub {{ color: #666; font-size: 12px; margin-bottom: 8px; }}
 #oview {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 5px;
           margin-top: 8px; cursor: grab; touch-action: none; display: block; }}
 .ohint {{ color: #90a4ae; font-size: 11px; margin-top: 2px; }}
 .obtn {{ font-size: 11px; padding: 1px 8px; margin: 6px 4px 0 0;
          border: 1px solid #b0bec5; border-radius: 3px; background: #eceff1;
          cursor: pointer; }}
 .obtn.sel {{ background: #1565c0; color: #fff; border-color: #1565c0; }}
 .bar {{ height: 8px; background: #90caf9; border-radius: 3px; margin: 1px 0 5px; }}
 .cname {{ font-size: 12.2px; }}
 .axis {{ stroke: #555; stroke-width: 1; }}
 .grid {{ stroke: #000; stroke-opacity: 0.045; }}
 .tick {{ font-size: 11px; fill: #555; }}
 .axistitle {{ font-size: 12.5px; fill: #333; }}
 .header {{ font-size: 13.5px; font-weight: 600; fill: #263238; }}
 .seg {{ stroke-width: 2.4; }}
 .seg.frag {{ stroke: #607d8b; }}
 .seg.occ {{ stroke: #1565c0; }}
 .seg.virt {{ stroke: #b0bec5; }}
 .con {{ stroke: #888; stroke-width: 1; stroke-dasharray: 5 4; }}
 .con.hi {{ stroke: #e65100; stroke-width: 1.6; opacity: 0.95 !important; }}
 .con.dim {{ opacity: 0.06 !important; }}
 .lab {{ font-size: 12.5px; fill: #222; }}
 .lead {{ stroke: #bbb; stroke-width: 0.7; }}
 .el {{ font-size: 10.5px; fill: #1565c0; }}
 .hl {{ font-size: 11px; fill: #e65100; font-weight: 600; }}
 .lvl {{ cursor: pointer; outline: none; }}
 .lvl:hover .seg {{ stroke-width: 4; }}
 #foot {{ color: #777; font-size: 11.5px; margin-top: 8px; }}
</style>
</head>
<body>
<div id="page">
<h1>{heading_html}</h1>
<div>{chip_html}</div>
<div id="controls">
 Energy window (eV): <input id="emin" type="number" step="1"> &ndash;
 <input id="emax" type="number" step="1">
 <button id="ereset">reset</button>
 <span class="hint">Ctrl/&#8984; + scroll (or pinch) to zoom, drag to pan</span>
</div>
<div id="flex">
<svg id="diagram" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
<defs><clipPath id="plotclip">
<rect x="{left - 14}" y="{top - 10}" width="{width - left - 6}" height="{plot_height + 20}"/>
</clipPath></defs>
{header_svg}
<line x1="{left - 8}" y1="{top - 14}" x2="{left - 8}" y2="{top + plot_height + 6}" class="axis"/>
<text x="{left - 44}" y="{top + plot_height / 2:.1f}" class="axistitle"
 transform="rotate(-90 {left - 44} {top + plot_height / 2:.1f})"
 text-anchor="middle">E (eV)</text>
<g id="gGrid"></g>
<g id="gCon" clip-path="url(#plotclip)"></g>
<g id="gLvl" clip-path="url(#plotclip)"></g>
</svg>
<div id="panel"><h2>Level details</h2>
<div class="sub">hover or click a level</div><div id="pbody"></div></div>
</div>
<div id="foot">{foot_html}</div>
</div>
<script>
{script}
</script>
</body>
</html>
"""
    with open(output_path, "w") as handle:
        handle.write(html)


def write_diagram_html(diagram: MODiagram, output_path: str) -> None:
    import re

    columns = {"ligand-ao": 165, "salc": 360, "mo": 600, "center-ao": 830}
    half = {"ligand-ao": 30, "salc": 30, "mo": 34, "center-ao": 30}
    order = ["ligand-ao", "salc", "mo", "center-ao"]
    side = {"ligand-ao": -1, "salc": -1, "mo": 1, "center-ao": 1}

    all_levels = (
        diagram.ligand_ao_levels + diagram.salc_levels
        + diagram.mo_levels + diagram.center_levels
    )
    energies = [level.energy for level in all_levels]
    e_min, e_max = min(energies), max(energies)
    padding = 0.06 * (e_max - e_min) or 1.0

    def per_atom_components(vector):
        components: dict[int, list[float]] = {}
        for index, orbital in enumerate(diagram.orbitals):
            if abs(vector[index]) < 1e-10:
                continue
            values = components.setdefault(orbital.atom, [0.0, 0.0, 0.0, 0.0])
            if orbital.l == 0:
                values[0] += float(vector[index])
            else:
                values[1 + orbital.m] += float(vector[index])
        return components

    levels_json = []
    for level in all_levels:
        partners = canonical_sketch_partners(
            [per_atom_components(vector) for vector in level.vectors]
        )
        orb = [_sketch_entries(per_atom) for per_atom in partners] or None
        levels_json.append({
            "id": level.level_id,
            "col": level.column,
            "e": round(level.energy, 4),
            "deg": level.degeneracy,
            "label": level.label,
            "el": level.electrons if level.column == "mo" else None,
            "occ": bool(level.column == "mo" and level.electrons > 0),
            "links": [
                [source_id, round(weight, 4)]
                for source_id, weight in level.composition if weight >= 0.02
            ],
            "comp": [
                [diagram._level_name(source_id), round(100 * weight, 1)]
                for source_id, weight in sorted(level.composition, key=lambda kv: -kv[1])
                if weight > 0.005
            ],
            "detail": level.detail,
            "orb": orb,
        })

    center_symbol = diagram.symbols[diagram.center]
    ligand_parts = [
        (f"{len(sites)}{element}" if len(sites) > 1 else element)
        for element, sites in sorted(diagram.ligand_sites.items())
    ]
    salc_text = "".join(
        f"{element}{len(sites)}" if len(sites) > 1 else element
        for element, sites in sorted(diagram.ligand_sites.items())
    )
    headers = {
        "ligand-ao": f"{' + '.join(ligand_parts)} AOs",
        "salc": f"{svg_sub_digits(salc_text)} SALCs",
        "mo": f"{svg_sub_digits(diagram.formula)} MOs",
        "center-ao": f"{center_symbol} AOs",
    }

    formula_html = re.sub(r"(\d+)", r"<sub>\1</sub>", diagram.formula)
    gap = ""
    if diagram.homo and diagram.lumo:
        gap = (
            f"HOMO&ndash;LUMO gap {diagram.lumo.energy - diagram.homo.energy:.2f} eV"
        )
    chips = [
        formula_html,
        f"{diagram.schoenflies} ({diagram.hm})",
        f"{diagram.n_electrons} valence electrons",
        gap,
        "extended H&uuml;ckel / STO overlaps",
    ]

    render_diagram_page(
        output_path,
        title=f"MO diagram: {diagram.formula} ({diagram.schoenflies})",
        heading_html=f"Molecular-orbital diagram: {formula_html}",
        chips=chips,
        columns=columns, half=half, order=order, side=side, headers=headers,
        levels_json=levels_json,
        homo_id=diagram.homo.level_id if diagram.homo else None,
        lumo_id=diagram.lumo.level_id if diagram.lumo else None,
        e_min=e_min - padding, e_max=e_max + padding,
        foot_html=(
            "Semi-quantitative diagram from symmetry + overlap only "
            "(symmetry-adapted extended H&uuml;ckel, Wolfsberg&ndash;Helmholz "
            "K = 1.75, single-&zeta; STOs, exact two-center overlap integrals). "
            "Generated by CrystOD (crystod-mol --diagram)."
        ),
        geometry=diagram_geometry(diagram.symbols, diagram.coordinates),
    )


# ----------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Semi-quantitative molecular-orbital diagram from symmetry "
                    "and overlap (symmetry-adapted extended Hueckel)."
    )
    parser.add_argument("--xyz", required=True, metavar="FILE",
                        help="Molecule file in XYZ format.")
    parser.add_argument("--center", default=None, metavar="EL",
                        help="Element of the central atom (default: the atom "
                             "closest to the molecular center).")
    parser.add_argument("--tolerance", type=float, default=0.3,
                        help="Distance tolerance (Angstrom) for the symmetry "
                             "detection (default: 0.3).")
    parser.add_argument("--output", default=None, metavar="FILE",
                        help="Output HTML path "
                             "(default: MolOD_{molecule}.html, or "
                             "MolOD_{molecule}_pyscf.html with --pyscf).")
    parser.add_argument("--pyscf", action="store_true",
                        help="Quantitative diagram from three PySCF SCF "
                             "calculations (molecule + two fragments in the "
                             "full molecular basis).")
    parser.add_argument("--basis", default="def2-svp", type=str.lower,
                        help="PySCF basis set (default: def2-svp).")
    parser.add_argument("--theory", default="scf", choices=["scf", "dft"],
                        help="PySCF level of theory (default: scf = Hartree-Fock).")
    parser.add_argument("--xc", default="b3lyp", type=str.lower,
                        help="Exchange-correlation functional for --theory dft "
                             "(default: b3lyp).")
    parser.add_argument("--charge", type=int, default=0,
                        help="Total charge of the molecule (--pyscf; default 0).")
    parser.add_argument("--spin", type=int, default=None,
                        help="Molecular spin 2S (--pyscf; default: 0 or 1 by "
                             "electron parity, e.g. use --spin 2 for triplet O2).")
    parser.add_argument("--ao-left", default=None, metavar="FORMULA",
                        help="Left-fragment formula for --pyscf, e.g. H4 or O "
                             "(default: the ligand atoms).")
    parser.add_argument("--ao-right", default=None, metavar="FORMULA",
                        help="Right-fragment formula for --pyscf, e.g. CO or O "
                             "(default: the central atom).")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.pyscf:
        from .mo_diagram_pyscf import run_pyscf_diagram

        run_pyscf_diagram(args)
        return
    if args.ao_left or args.ao_right:
        parser.error("--ao-left/--ao-right require --pyscf.")
    diagram = MODiagram(args.xyz, args.tolerance, args.center)
    diagram.print_report()
    stem = os.path.splitext(os.path.basename(args.xyz))[0]
    output_path = args.output or f"MolOD_{stem}.html"
    diagram.write_html(output_path)
    print(f"\nMO diagram written to {output_path}")
    print()


if __name__ == "__main__":
    main()
