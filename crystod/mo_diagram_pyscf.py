"""Quantitative molecular-orbital diagrams via PySCF
(crystod-mol --diagram --pyscf).

The symmetry-only diagram of :mod:`crystod.mo_diagram` sketches the MO
diagram from SALCs + STO overlaps; this module makes it quantitative with
three self-consistent-field calculations sharing one AO space:

- the full molecule,
- the "left" fragment  (the other atoms replaced by ghost atoms),
- the "right" fragment (idem),

all at the same geometry and basis, so the fragment levels are the
pre-bonding energy states (counterpoise-consistent: every calculation uses
the full molecular basis) and the molecular MOs can be projected exactly
onto the fragment MOs for the correlation lines and composition panel.

By default the fragments are the ligand cage (left) and the central atom
(right), as in the symmetry-only mode; ``--ao-left``/``--ao-right`` select
any partition by chemical formula (CH3OH: ``--ao-left H4 --ao-right CO``;
O2: ``--ao-left O --ao-right O``).

Irrep labels use crystod's own point-group machinery (characters of the MO
under the exact character-table operations, evaluated on the AO
representation), so they agree with the symmetry-only diagram and with
crystod-group; linear molecules fall back to PySCF's Dooh/Coov labels
rendered as sigma/pi/delta.

References:
  Q. Sun et al., PySCF, WIREs Comput. Mol. Sci. 8, e1340 (2018);
  Q. Sun et al., J. Chem. Phys. 153, 024109 (2020).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np

from .decompose_irrep import get_character_table
from .mo_diagram import (
    CORE_SHELLS,
    _sketch_entries,
    canonical_sketch_partners,
    diagram_geometry,
    lowercase_irrep,
    render_diagram_page,
    svg_sub_digits,
)
from .molecular_salc import (
    _hm_symbol,
    _match_operations,
    _table_operations_cartesian,
    get_symmetry,
    load_molecule,
)

HARTREE_TO_EV = 27.211386245988

CORE_ELECTRON_COUNT = {
    element: sum(2 if shell.endswith("s") else 6 for shell in shells)
    for element, shells in CORE_SHELLS.items()
}

_GREEK = {"A": "σ", "E1": "π", "E2": "δ", "E3": "φ"}

# Fragment levels whose Mulliken population sits mostly on the GHOST basis
# functions (counterpoise/BSSE artifact levels, not states of the fragment)
# are dropped from the diagram and the compositions below this threshold.
GHOST_FRACTION_THRESHOLD = 0.35


def _import_pyscf():
    try:
        from pyscf import dft, gto, scf  # noqa: F401
        import pyscf
    except ImportError as exc:
        raise SystemExit(
            "ERROR: --pyscf requires the pyscf package (pip install pyscf)."
        ) from exc
    return pyscf


def parse_fragment_spec(spec: str) -> dict[str, int]:
    """'H4' -> {'H': 4}; 'CO' -> {'C': 1, 'O': 1}; 'CH3' -> {'C': 1, 'H': 3}."""
    counts: dict[str, int] = {}
    position = 0
    for match in re.finditer(r"([A-Z][a-z]?)(\d*)", spec):
        if match.start() != position or not match.group(1):
            break
        counts[match.group(1)] = counts.get(match.group(1), 0) + int(match.group(2) or 1)
        position = match.end()
    if position != len(spec) or not counts:
        raise SystemExit(
            f"ERROR: could not parse the fragment formula '{spec}' "
            "(expected element symbols with optional counts, e.g. H4, CO, CH3)."
        )
    return counts


# ------------------------------------------------------- AO representation


def _euler_zyz(rotation: np.ndarray) -> tuple[float, float, float]:
    """z-y-z Euler angles of a proper rotation (active convention,
    R = Rz(alpha) Ry(beta) Rz(gamma))."""
    beta = float(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0)))
    if rotation[2, 2] > 1.0 - 1e-10:
        return float(np.arctan2(rotation[1, 0], rotation[0, 0])), 0.0, 0.0
    if rotation[2, 2] < -1.0 + 1e-10:
        return float(np.arctan2(-rotation[1, 0], -rotation[0, 0])), np.pi, 0.0
    alpha = float(np.arctan2(rotation[1, 2], rotation[0, 2]))
    gamma = float(np.arctan2(rotation[2, 1], -rotation[2, 0]))
    return alpha, beta, gamma


def _real_sph_rotation(l: int, rotation: np.ndarray) -> np.ndarray:
    """Rotation matrix on PySCF's real spherical harmonics of momentum l.

    Improper operations pick up the parity (-1)^l. The convention (Euler
    extraction + pyscf Dmatrix orientation) is validated on every call
    against the l = 1 case, where the p_x/p_y/p_z representation must equal
    the rotation matrix itself.
    """
    from pyscf.symm import Dmatrix

    det = 1.0 if np.linalg.det(rotation) > 0 else -1.0
    proper = det * rotation
    alpha, beta, gamma = _euler_zyz(proper)
    D1 = Dmatrix.Dmatrix(1, alpha, beta, gamma, reorder_p=True)
    transpose = np.abs(D1.T - proper).max() < np.abs(D1 - proper).max()
    if min(np.abs(D1 - proper).max(), np.abs(D1.T - proper).max()) > 1e-8:
        raise SystemExit("ERROR: pyscf Dmatrix convention check failed.")
    D = Dmatrix.Dmatrix(l, alpha, beta, gamma, reorder_p=True)
    if transpose:
        D = D.T
    return D * (det ** l)


def _atom_permutation(coordinates, tags, rotation, tolerance=1e-3):
    """perm[j] = i when the operation maps center j onto center i (same tag),
    or None when the operation does not preserve the tagged centers."""
    mapped = coordinates @ np.asarray(rotation).T
    permutation = []
    for j, position in enumerate(mapped):
        distances = np.linalg.norm(coordinates - position, axis=1)
        i = int(np.argmin(distances))
        if distances[i] > tolerance or tags[i] != tags[j] or i in permutation:
            return None
        permutation.append(i)
    return permutation


def _ao_representation(mol, rotation, permutation) -> np.ndarray:
    """Matrix of the symmetry operation on the (spherical) AO basis."""
    n_ao = mol.nao_nr()
    ao_loc = mol.ao_loc_nr()
    gamma = np.zeros((n_ao, n_ao))
    blocks: dict[int, np.ndarray] = {}
    for shell in range(mol.nbas):
        l = mol.bas_angular(shell)
        if l not in blocks:
            blocks[l] = _real_sph_rotation(l, rotation)
        source_atom = mol.bas_atom(shell)
        target_atom = permutation[source_atom]
        # the matching shell on the target atom: same position in the
        # per-atom shell list (all atoms of one tag share the basis layout)
        source_shells = [s for s in range(mol.nbas) if mol.bas_atom(s) == source_atom]
        target_shells = [s for s in range(mol.nbas) if mol.bas_atom(s) == target_atom]
        target_shell = target_shells[source_shells.index(shell)]
        p0, p1 = ao_loc[shell], ao_loc[shell + 1]
        q0 = ao_loc[target_shell]
        n_contracted = mol.bas_nctr(shell)
        width = 2 * l + 1
        for c in range(n_contracted):
            gamma[q0 + c * width:q0 + (c + 1) * width,
                  p0 + c * width:p0 + (c + 1) * width] = blocks[l]
    return gamma


# ------------------------------------------------------------------ levels


@dataclass
class PyscfLevel:
    column: str
    energy: float               # eV
    degeneracy: int
    label: str
    electrons: int
    irrep: str | None
    level_id: str
    orbital_indices: list[int]
    composition: list = field(default_factory=list)
    detail: str = ""
    real_fraction: float = 1.0  # Mulliken population share on the real atoms


class PyscfDiagram:
    """MO diagram from three PySCF SCF calculations in one AO space."""

    def __init__(self, xyz_path, tolerance=0.3, center_element=None,
                 left_spec=None, right_spec=None, basis="def2-svp",
                 theory="scf", xc="b3lyp", charge=0, spin=None):
        _import_pyscf()
        self.xyz_path = xyz_path
        self.basis = basis
        self.theory = theory
        self.xc = xc
        self.charge = charge
        molecule = load_molecule(xyz_path)
        self.symbols = [site.specie.symbol for site in molecule]
        self.coordinates = np.array([site.coords for site in molecule])

        # ---- symmetry: align to the standard point-group frame when possible
        schoenflies, operations = get_symmetry(molecule, tolerance)
        self.schoenflies = schoenflies
        self.hm = _hm_symbol(schoenflies)
        self.linear = "*" in schoenflies
        self.character_table = None
        self.operations = []
        self.operation_classes = []
        if self.hm is not None:
            self.character_table = get_character_table(self.hm)
            table_ops, table_classes = _table_operations_cartesian(self.character_table)
            alignment, matched = _match_operations(operations, table_ops, table_classes)
            self.operations = [table_ops[i] for i in matched]
            self.operation_classes = [table_classes[i] for i in matched]
            self.coordinates = self.coordinates @ alignment.T
            self._symmetrize_coordinates(tolerance)
        elif self.linear:
            # molecular axis along z, so pyscf's symmetry frame matches ours
            centered = self.coordinates - self.coordinates.mean(axis=0)
            _, _, Vt = np.linalg.svd(centered)
            axis = Vt[0] / np.linalg.norm(Vt[0])
            z = np.array([0.0, 0.0, 1.0])
            v = np.cross(axis, z)
            if np.linalg.norm(v) < 1e-12:
                rotation = np.eye(3) if axis[2] > 0 else np.diag([1.0, -1.0, -1.0])
            else:
                v = v / np.linalg.norm(v)
                angle = np.arccos(np.clip(axis @ z, -1.0, 1.0))
                K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                rotation = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
            self.coordinates = centered @ rotation.T

        # ---- fragments
        self._identify_fragments(center_element, left_spec, right_spec)
        self.formula = self._conventional_formula()

        # ---- the three SCF calculations
        self.n_electrons = (
            sum(_atomic_number(s) for s in self.symbols) - self.charge
        )
        if spin is None:
            spin = self.n_electrons % 2
        if (self.n_electrons - spin) % 2:
            raise SystemExit(
                f"ERROR: spin {spin} (2S) is inconsistent with "
                f"{self.n_electrons} electrons."
            )
        self.spin = spin
        self.calculations = {
            "mo": self._run_scf(sorted(self.left + self.right), self.charge, self.spin),
            "left": self._run_scf(self.left, 0, self._fragment_spin(self.left),
                                  fractional=True),
            "right": self._run_scf(self.right, 0, self._fragment_spin(self.right),
                                   fractional=True),
        }
        self._build_levels()
        self._link_levels()

    # ---------------------------------------------------------------- setup

    def _symmetrize_coordinates(self, tolerance):
        from .molecular_salc import get_permutation_matrices

        permutations = get_permutation_matrices(
            self.operations, self.coordinates, tolerance
        )
        symmetrized = np.zeros_like(self.coordinates)
        for rotation, permutation in zip(self.operations, permutations):
            symmetrized += (permutation.T @ self.coordinates) @ rotation
        self.coordinates = symmetrized / len(self.operations)

    def _identify_fragments(self, center_element, left_spec, right_spec):
        if (left_spec is None) != (right_spec is None):
            raise SystemExit("ERROR: give both --ao-left and --ao-right (or neither).")
        if left_spec is not None:
            left_counts = parse_fragment_spec(left_spec)
            right_counts = parse_fragment_spec(right_spec)
            totals: dict[str, int] = {}
            for symbol in self.symbols:
                totals[symbol] = totals.get(symbol, 0) + 1
            combined = {
                element: left_counts.get(element, 0) + right_counts.get(element, 0)
                for element in set(left_counts) | set(right_counts)
            }
            if combined != totals:
                molecule_formula = " ".join(f"{k}{v}" for k, v in sorted(totals.items()))
                raise SystemExit(
                    f"ERROR: --ao-left {left_spec} + --ao-right {right_spec} does "
                    f"not partition the molecule ({molecule_formula})."
                )
            remaining = dict(left_counts)
            self.left, self.right = [], []
            for i, symbol in enumerate(self.symbols):
                if remaining.get(symbol, 0) > 0:
                    self.left.append(i)
                    remaining[symbol] -= 1
                else:
                    self.right.append(i)
            self.left_name = _format_formula(left_counts)
            self.right_name = _format_formula(right_counts)
            self.center = None
            return
        # default: ligand cage (left) | central atom (right)
        distances = np.linalg.norm(
            self.coordinates - self.coordinates.mean(axis=0), axis=1
        )
        if center_element is not None:
            candidates = [i for i, s in enumerate(self.symbols) if s == center_element]
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
                    "ERROR: could not identify a unique central atom; use "
                    "--center EL, or partition explicitly with "
                    "--ao-left/--ao-right."
                )
        self.left = [i for i in range(len(self.symbols)) if i != self.center]
        self.right = [self.center]
        left_counts: dict[str, int] = {}
        for i in self.left:
            left_counts[self.symbols[i]] = left_counts.get(self.symbols[i], 0) + 1
        self.left_name = _format_formula(left_counts)
        self.right_name = self.symbols[self.center]

    def _conventional_formula(self):
        counts: dict[str, int] = {}
        for symbol in self.symbols:
            counts[symbol] = counts.get(symbol, 0) + 1
        if self.center is not None:
            ligand = {k: v for k, v in counts.items()}
            ligand[self.symbols[self.center]] -= 1
            if ligand[self.symbols[self.center]] == 0:
                del ligand[self.symbols[self.center]]
            formula = self.symbols[self.center] + _format_formula(ligand)
            return {"OH2": "H2O", "SH2": "H2S", "SeH2": "H2Se"}.get(formula, formula)
        return _format_formula(counts, hill=True)

    def _fragment_spin(self, sites):
        # fragments are spin-averaged pre-bonding references: lowest spin
        # consistent with the electron count, plus fractional occupation of
        # the degenerate frontier shell (see _run_scf)
        electrons = sum(_atomic_number(self.symbols[i]) for i in sites)
        return electrons % 2

    def _run_scf(self, real_sites, charge, spin, fractional=False):
        """One SCF in the full molecular basis (removed atoms as ghosts).

        Fragments use fractional occupations of degenerate frontier levels
        (``scf.addons.frac_occ``): a partially filled degenerate shell (C 2p,
        the t2 shell of an H4 cage, ...) would otherwise break the point-group
        symmetry, whereas the pre-bonding reference states should keep it.
        """
        from pyscf import dft, gto, scf

        real = set(real_sites)
        atom = []
        for i, (symbol, position) in enumerate(zip(self.symbols, self.coordinates)):
            name = symbol if i in real else f"ghost-{symbol}"
            atom.append(f"{name} {position[0]:.10f} {position[1]:.10f} {position[2]:.10f}")
        try:
            mol = gto.M(
                atom="; ".join(atom), basis=self.basis, charge=charge, spin=spin,
                unit="Angstrom", verbose=0, symmetry=self.linear,
            )
        except Exception:
            mol = gto.M(
                atom="; ".join(atom), basis=self.basis, charge=charge, spin=spin,
                unit="Angstrom", verbose=0,
            )
        def make_mf():
            if self.theory == "dft":
                base = dft.RKS(mol) if spin == 0 else dft.ROKS(mol)
                base.xc = self.xc
            else:
                base = scf.RHF(mol) if spin == 0 else scf.ROHF(mol)
            base.verbose = 0
            base.max_cycle = 200
            return base

        mf = make_mf()
        if fractional:
            mf = scf.addons.frac_occ(mf)
        mf.kernel()
        if fractional and not mf.converged:
            # the default degeneracy detection of frac_occ can miss part of
            # the frontier shell (symmetry-broken occupations never become
            # stationary); retry with a wide degeneracy window
            mf = scf.addons.frac_occ(make_mf(), tol=0.1)
            mf.kernel()
        return {
            "mol": mol, "mf": mf, "spin": spin, "charge": charge,
            "converged": bool(mf.converged),
            "energy": float(mf.e_tot),
            "mo_energy": np.asarray(mf.mo_energy, dtype=float) * HARTREE_TO_EV,
            "mo_occ": np.asarray(mf.mo_occ, dtype=float),
            "mo_coeff": np.asarray(mf.mo_coeff, dtype=float),
            "real_sites": sorted(real),
        }

    # ------------------------------------------------------------ labeling

    def _valid_operations(self, calc):
        """Table operations that map real atoms onto real atoms (and ghosts
        onto ghosts) of the same element, with the atom permutations."""
        real = set(calc["real_sites"])
        tags = [
            (symbol, i in real) for i, symbol in enumerate(self.symbols)
        ]
        valid = []
        for rotation, class_name in zip(self.operations, self.operation_classes):
            permutation = _atom_permutation(self.coordinates, tags, rotation)
            if permutation is not None:
                valid.append((rotation, class_name, permutation))
        return valid

    def _irrep_labels(self, calc, groups):
        """Irrep label per degenerate group (list parallel to groups), or
        None when no labeling is possible."""
        if self.character_table is not None:
            valid = self._valid_operations(calc)
            if len(valid) == len(self.operations):
                return self._labels_from_characters(calc, groups, valid)
            return None
        if self.linear:
            return self._labels_from_pyscf_symmetry(calc, groups)
        return None

    def _labels_from_characters(self, calc, groups, valid):
        mol = calc["mol"]
        S = mol.intor("int1e_ovlp")
        C = calc["mo_coeff"]
        rotation_list = list(self.character_table["rotation_list"])
        table = self.character_table["character_table"]
        # one representative per class is enough (characters are class functions)
        representatives = {}
        for rotation, class_name, permutation in valid:
            representatives.setdefault(class_name, (rotation, permutation))
        gammas = {
            class_name: _ao_representation(mol, rotation, permutation)
            for class_name, (rotation, permutation) in representatives.items()
        }
        labels = []
        for group in groups:
            vectors = C[:, group]
            characters = []
            for class_name in rotation_list:
                gamma = gammas[class_name]
                characters.append(
                    float(np.trace(vectors.T @ S @ gamma @ vectors))
                )
            best, best_error = None, 1e9
            for irrep, chars in table.items():
                row = [float(np.real(c)) for c in np.atleast_1d(chars)]
                error = max(abs(a - b) for a, b in zip(characters, row))
                if error < best_error:
                    best, best_error = irrep, error
            labels.append(best if best_error < 0.05 else None)
        return labels

    def _labels_from_pyscf_symmetry(self, calc, groups):
        """sigma/pi/delta labels for linear molecules via PySCF's Dooh/Coov.

        The geometry was pre-aligned to the z axis, so the symmetric mol of
        the calculation itself carries the symmetry-adapted basis and no
        reorientation of the MO coefficients is needed."""
        from pyscf import symm

        mol = calc["mol"]
        if getattr(mol, "symm_orb", None) is None:
            return None
        try:
            names = symm.label_orb_symm(
                mol, mol.irrep_name, mol.symm_orb, calc["mo_coeff"], check=False
            )
        except Exception:
            return None
        labels = []
        for group in groups:
            raw = str(names[group[0]])
            match = re.match(r"([AE])(\d?)([gu]?)", raw)
            if not match:
                labels.append(None)
                continue
            letter = match.group(1) + (match.group(2) if match.group(1) == "E" else "")
            greek = _GREEK.get("A" if letter.startswith("A") else letter)
            labels.append((greek or raw) + match.group(3))
        return labels

    # ------------------------------------------------------------- levels

    def _build_levels(self):
        self.levels: dict[str, list[PyscfLevel]] = {}
        for column in ("left", "mo", "right"):
            calc = self.calculations[column]
            energies = calc["mo_energy"]
            occupations = calc["mo_occ"]
            groups = _group_degenerate(energies, tol=2e-3)
            labels = self._irrep_labels(calc, groups)
            if labels is not None:
                # an accidental near-degeneracy can merge different irreps
                # into one unlabeled group: re-split those finely and relabel
                refined = []
                for group, label in zip(groups, labels):
                    if label is None and len(group) > 1:
                        subgroups = _group_degenerate(energies[group], tol=1e-6)
                        refined.extend([[group[i] for i in sub] for sub in subgroups])
                    else:
                        refined.append(group)
                if len(refined) != len(groups):
                    groups = sorted(refined, key=lambda g: float(energies[g[0]]))
                    labels = self._irrep_labels(calc, groups)
            counters: dict[str, int] = {}
            mol = calc["mol"]
            S_ao = mol.intor("int1e_ovlp")
            C_ao = calc["mo_coeff"]
            real = set(calc["real_sites"])
            ao_atom = np.zeros(mol.nao_nr(), dtype=int)
            ao_loc = mol.ao_loc_nr()
            for b in range(mol.nbas):
                ao_atom[ao_loc[b]:ao_loc[b + 1]] = mol.bas_atom(b)
            real_mask = np.array([atom in real for atom in ao_atom])
            column_levels = []
            for g_index, group in enumerate(groups):
                energy = float(np.mean(energies[group]))
                electrons = int(round(float(np.sum(occupations[group]))))
                irrep = labels[g_index] if labels else None
                if irrep is not None:
                    counters[irrep] = counters.get(irrep, 0) + 1
                    label = f"{counters[irrep]}{lowercase_irrep(irrep)}"
                else:
                    label = f"{g_index + 1}"
                fraction = 1.0
                if column != "mo":
                    populations = [
                        float(np.sum((C_ao[:, k] * (S_ao @ C_ao[:, k]))[real_mask]))
                        for k in group
                    ]
                    fraction = min(1.0, max(0.0, float(np.mean(populations))))
                level = PyscfLevel(
                    column=column, energy=energy, degeneracy=len(group),
                    label=label, electrons=electrons, irrep=irrep,
                    level_id=f"{column}_{g_index}", orbital_indices=list(group),
                    real_fraction=fraction,
                )
                column_levels.append(level)
            self.levels[column] = column_levels

        occupied = [l for l in self.levels["mo"] if l.electrons > 0]
        empty = [l for l in self.levels["mo"] if l.electrons == 0]
        self.homo = occupied[-1] if occupied else None
        self.lumo = empty[0] if empty else None

    def _link_levels(self):
        """Project the molecule MOs onto the fragment MOs (same AO space)."""
        S = self.calculations["mo"]["mol"].intor("int1e_ovlp")
        C_mo = self.calculations["mo"]["mo_coeff"]
        projections = {}
        for column in ("left", "right"):
            projections[column] = self.calculations[column]["mo_coeff"].T @ S @ C_mo
        for level in self.levels["mo"]:
            weights = []
            for column in ("left", "right"):
                P = projections[column]
                for fragment_level in self.levels[column]:
                    if fragment_level.real_fraction < GHOST_FRACTION_THRESHOLD:
                        continue  # counterpoise/BSSE artifact level
                    weight = float(np.sum(
                        P[np.ix_(fragment_level.orbital_indices,
                                 level.orbital_indices)] ** 2
                    )) * fragment_level.real_fraction
                    if weight > 1e-6:
                        weights.append((fragment_level, weight))
            total = sum(w for _, w in weights) or 1.0
            level.composition = [
                (fragment_level.level_id, weight / total)
                for fragment_level, weight in weights
            ]
        # tooltips
        names = {
            level.level_id: f"{self._display_name(level.column)} {level.label}"
            for column_levels in self.levels.values() for level in column_levels
        }
        for column_levels in self.levels.values():
            for level in column_levels:
                parts = ", ".join(
                    f"{100 * w:.0f}% {names[i]}"
                    for i, w in sorted(level.composition, key=lambda kv: -kv[1])
                    if w > 0.02
                )
                level.detail = (
                    f"{level.label}: E = {level.energy:.2f} eV, "
                    f"{level.electrons} e-" + (f"  |  {parts}" if parts else "")
                )

    def _column_name(self, column):
        return {"left": self.left_name, "mo": self.formula,
                "right": self.right_name}[column]

    def _display_name(self, column):
        """Column name for composition lists; disambiguates identical
        fragment formulas (O2 from O + O -> 'O(L)' / 'O(R)')."""
        name = self._column_name(column)
        if column != "mo" and self.left_name == self.right_name:
            return f"{name}({'L' if column == 'left' else 'R'})"
        return name

    def _sketch_partners(self, column, level):
        """Per-atom s/p amplitudes of every partner orbital of a level.

        Contracted s and p shells are summed per atom (the radial parts are
        all positive at large r, so the sum approximates the far-field
        lobes; d and higher shells are omitted). Only the REAL atoms of the
        calculation are drawn: the fragment orbitals also carry small tails
        on the ghost basis functions (counterpoise polarization), which
        would obscure the pre-bonding SALC picture. Degenerate partners are
        canonicalized to match the SALC viewer."""
        calc = self.calculations[column]
        mol = calc["mol"]
        C = calc["mo_coeff"]
        real = set(calc["real_sites"])
        ao_loc = mol.ao_loc_nr()
        partners = []
        for k in level.orbital_indices:
            vector = C[:, k]
            per_atom: dict[int, list[float]] = {}
            for shell in range(mol.nbas):
                l = mol.bas_angular(shell)
                if l > 1:
                    continue
                atom = mol.bas_atom(shell)
                if atom not in real:
                    continue
                values = per_atom.setdefault(atom, [0.0, 0.0, 0.0, 0.0])
                p0 = ao_loc[shell]
                width = 2 * l + 1
                for c in range(mol.bas_nctr(shell)):
                    block = vector[p0 + c * width:p0 + (c + 1) * width]
                    if l == 0:
                        values[0] += float(block[0])
                    else:  # pyscf p order: (px, py, pz)
                        for m in range(3):
                            values[1 + m] += float(block[m])
            partners.append(per_atom)
        partners = canonical_sketch_partners(partners)
        return [_sketch_entries(per_atom) for per_atom in partners]

    def _core_count(self, column):
        calc = self.calculations[column]
        return sum(
            CORE_ELECTRON_COUNT.get(self.symbols[i], 0) for i in calc["real_sites"]
        ) // 2

    # ------------------------------------------------------------- report

    def print_report(self):
        print("\n* Molecule *")
        print(f"{self.xyz_path} ({self.formula}, {len(self.symbols)} atoms)")
        print("\n* Point group *")
        if self.hm is not None:
            print(f"{self.schoenflies} (Hermann-Mauguin: {self.hm})")
        else:
            print(f"{self.schoenflies}")
        print("\n* Fragments (pre-bonding states; removed atoms kept as ghost basis) *")
        left_atoms = ", ".join(self.symbols[i] for i in self.left)
        right_atoms = ", ".join(self.symbols[i] for i in self.right)
        print(f"left : {self.left_name} ({left_atoms}), "
              f"spin 2S = {self.calculations['left']['spin']}")
        print(f"right: {self.right_name} ({right_atoms}), "
              f"spin 2S = {self.calculations['right']['spin']}")
        print("(fragments are spin/spherically averaged: fractional "
              "occupation of degenerate frontier shells)")

        method = ("Hartree-Fock method" if self.theory == "scf"
                  else f"DFT method ({self.xc.upper()} functional)")
        print(f"\n* PySCF calculations ({method} / {self.basis} basis) *")
        for column in ("left", "right", "mo"):
            calc = self.calculations[column]
            name = self._column_name(column)
            flavor = ("R" if calc["spin"] == 0 else "RO") + (
                "KS" if self.theory == "dft" else "HF"
            )
            convergence = "converged" if calc["converged"] else "NOT CONVERGED"
            print(f"{name:>8} ({flavor}): E_tot = {calc['energy']:.6f} Ha ({convergence})")
        interaction = (
            self.calculations["mo"]["energy"]
            - self.calculations["left"]["energy"]
            - self.calculations["right"]["energy"]
        )
        print(f"interaction energy E(mol) - E(left) - E(right) = "
              f"{interaction:.6f} Ha = {interaction * HARTREE_TO_EV:.3f} eV")
        print("(counterpoise-consistent: all three calculations use the full "
              "molecular basis)")

        print("\n* Molecular orbitals *")
        print(f"{'MO':>6} {'E (eV)':>10} {'occ':>4}  composition")
        core = self._core_count("mo")
        shown = [
            level for level in self.levels["mo"]
            if min(level.orbital_indices) >= core
        ]
        names = {
            level.level_id: f"{self._display_name(level.column)} {level.label}"
            for column in ("left", "right") for level in self.levels[column]
        }
        for level in reversed(shown):
            if level.energy > (self.lumo.energy + 12 if self.lumo else 1e9):
                continue
            parts = ", ".join(
                f"{100 * w:.0f}% {names[i]}"
                for i, w in sorted(level.composition, key=lambda kv: -kv[1])
                if w > 0.02
            )
            degeneracy = f" x{level.degeneracy}" if level.degeneracy > 1 else ""
            print(f"{level.label + degeneracy:>6} {level.energy:10.2f} "
                  f"{level.electrons:>4}  {parts}")
        if core:
            print(f"(+ {core} core level(s) below; visible in the HTML by "
                  "panning the energy window)")
        ghost_hidden = sum(
            1 for column in ("left", "right")
            for level in self.levels[column]
            if level.real_fraction < GHOST_FRACTION_THRESHOLD
        )
        if ghost_hidden:
            print(f"({ghost_hidden} fragment level(s) dominated by the ghost "
                  "basis (BSSE artifacts) are omitted from the diagram and "
                  "the compositions)")

        print(f"\n* Electron filling ({self.n_electrons} electrons, "
              f"molecule spin 2S = {self.spin}) *")
        configuration = " ".join(
            f"({level.label})^{level.electrons}"
            for level in self.levels["mo"] if level.electrons > 0
        )
        print(configuration)
        if self.homo and self.lumo:
            print(f"HOMO = {self.homo.label} ({self.homo.energy:.2f} eV), "
                  f"LUMO = {self.lumo.label} ({self.lumo.energy:.2f} eV), "
                  f"gap = {self.lumo.energy - self.homo.energy:.2f} eV")

        print("\nMethod: fragment-resolved SCF MO diagram (PySCF).")
        print("Q. Sun et al., WIREs Comput. Mol. Sci. 8, e1340 (2018);")
        print("Q. Sun et al., J. Chem. Phys. 153, 024109 (2020).")

    # --------------------------------------------------------------- HTML

    def write_html(self, output_path):
        columns = {"left": 200, "mo": 480, "right": 760}
        half = {"left": 34, "mo": 34, "right": 34}
        order = ["left", "mo", "right"]
        side = {"left": -1, "mo": 1, "right": 1}
        headers = {
            column: f"{svg_sub_digits(self._display_name(column))} MOs"
            for column in order
        }

        names = {
            other.level_id: f"{self._display_name(other.column)} {other.label}"
            for levels in self.levels.values() for other in levels
        }
        ghost_hidden = sum(
            1 for column_levels in self.levels.values()
            for level in column_levels
            if level.real_fraction < GHOST_FRACTION_THRESHOLD
        )
        levels_json = []
        for column_levels in self.levels.values():
            for level in column_levels:
                if level.real_fraction < GHOST_FRACTION_THRESHOLD:
                    continue
                levels_json.append({
                    "id": level.level_id,
                    "col": level.column,
                    "e": round(level.energy, 4),
                    "deg": level.degeneracy,
                    "label": level.label,
                    "el": level.electrons,
                    "occ": level.electrons > 0,
                    "links": [
                        [i, round(w, 4)] for i, w in level.composition if w >= 0.02
                    ],
                    "comp": [
                        [names[i], round(100 * w, 1)]
                        for i, w in sorted(level.composition, key=lambda kv: -kv[1])
                        if w > 0.005
                    ],
                    "detail": level.detail,
                    "orb": self._sketch_partners(level.column, level) or None,
                })

        # default window: valence occupied .. a little above the LUMO
        floors = []
        for column in ("left", "mo", "right"):
            core = self._core_count(column)
            energies = sorted(
                level.energy for level in self.levels[column]
                if level.real_fraction >= GHOST_FRACTION_THRESHOLD
            )
            if core < len(energies):
                floors.append(energies[core] if core else energies[0])
        e_min = min(floors) - 3.0 if floors else -30.0
        top_candidates = [level.energy for level in self.levels["mo"]]
        e_max = (self.lumo.energy + 8.0) if self.lumo else max(top_candidates) + 2.0
        e_max = min(e_max, max(top_candidates) + 2.0)

        method = ("Hartree-Fock method" if self.theory == "scf"
                  else f"DFT method ({self.xc.upper()})")
        formula_html = re.sub(r"(\d+)", r"<sub>\1</sub>", self.formula)
        gap = ""
        if self.homo and self.lumo:
            gap = (f"HOMO&ndash;LUMO gap "
                   f"{self.lumo.energy - self.homo.energy:.2f} eV")
        chips = [
            formula_html,
            f"{self.schoenflies}" + (f" ({self.hm})" if self.hm else ""),
            f"{self.left_name} + {self.right_name}",
            f"{method} / {self.basis} basis (PySCF)",
            gap,
        ]
        render_diagram_page(
            output_path,
            title=f"MO diagram: {self.formula} (PySCF)",
            heading_html=f"Molecular-orbital diagram: {formula_html} "
                         f"<span style=\"color:#90a4ae;font-size:14px\">"
                         f"{self.left_name} + {self.right_name}, PySCF</span>",
            chips=chips,
            columns=columns, half=half, order=order, side=side, headers=headers,
            levels_json=levels_json,
            homo_id=self.homo.level_id if self.homo else None,
            lumo_id=self.lumo.level_id if self.lumo else None,
            e_min=e_min, e_max=e_max,
            geometry=diagram_geometry(self.symbols, self.coordinates),
            foot_html=(
                "Quantitative MO diagram from three SCF calculations in one "
                "AO space (fragments with ghost basis on the removed atoms, "
                "counterpoise-consistent; molecule MOs projected exactly onto "
                "the fragment MOs, weighted by each fragment level's "
                "real-atom Mulliken population)."
                + (f" {ghost_hidden} fragment level(s) dominated by the ghost "
                   "basis (BSSE artifacts, real-atom population &lt; 35%) are "
                   "not drawn." if ghost_hidden else "")
                + " PySCF: Q. Sun et al., WIREs Comput. Mol. "
                "Sci. 8, e1340 (2018). Generated by CrystOD "
                "(crystod-mol --diagram --pyscf)."
            ),
        )


def _group_degenerate(energies, tol=2e-3):
    groups: list[list[int]] = []
    for i in np.argsort(energies):
        if groups and abs(energies[groups[-1][0]] - energies[i]) < tol:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    return groups


def _format_formula(counts: dict[str, int], hill: bool = False) -> str:
    keys = sorted(counts)
    if hill and "C" in counts:
        keys = ["C"] + (["H"] if "H" in counts else []) + sorted(
            k for k in counts if k not in ("C", "H")
        )
    return "".join(f"{k}{counts[k]}" if counts[k] > 1 else k for k in keys)


def _atomic_number(symbol: str) -> int:
    from pyscf.data.elements import charge

    return int(charge(symbol))


def run_pyscf_diagram(args) -> None:
    diagram = PyscfDiagram(
        args.xyz, args.tolerance, args.center,
        left_spec=args.ao_left, right_spec=args.ao_right,
        basis=args.basis, theory=args.theory, xc=args.xc,
        charge=args.charge, spin=args.spin,
    )
    diagram.print_report()
    stem = os.path.splitext(os.path.basename(args.xyz))[0]
    output_path = args.output or f"MolOD_{stem}_pyscf.html"
    diagram.write_html(output_path)
    print(f"\nMO diagram written to {output_path}")
    print()
