"""Two-fragment extended-Hueckel MO diagrams (crystod-mol --diagram
--ao-left/--ao-right without --pyscf).

Semi-quantitative sibling of :mod:`crystod.mo_diagram_pyscf`: the molecule is
split into two arbitrary submolecules by chemical formula, and all three
columns (left | molecule | right) are solved in ONE atomic-orbital space --
the single-zeta STO extended-Hueckel basis of the whole molecule, with
Wolfsberg-Helmholz off-diagonals over exact two-center overlap integrals.
A fragment's pre-bonding levels are the generalized eigenstates of its own
(H, S) sub-block: in extended Hueckel the sub-block of the molecular matrices
IS the isolated fragment (H_ij depends only on the two orbitals), so no
counterpoise/ghost machinery is needed and every projection uses the one
shared overlap matrix.

Irrep labels follow the PySCF engine's "solve first, label by characters"
strategy (the fragments need not be invariant under the full molecular point
group; labels are omitted when they are not), and the MO numbering counts the
omitted core shells like the single-center EHT diagram, so e.g. the first
drawn a1g of C6 is 2a1g on top of the 1a1g core.

References:
  M. Wolfsberg and L. Helmholz, J. Chem. Phys. 20, 837 (1952).
  R. Hoffmann, J. Chem. Phys. 39, 1397 (1963).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np

from .decompose_irrep import get_character_table
from .mo_diagram import (
    CORE_SHELLS,
    VALENCE_ELECTRONS,
    AtomicOrbital,
    _group_degenerate,
    _sketch_entries,
    build_basis,
    canonical_sketch_partners,
    diagram_geometry,
    element_color,
    hamiltonian_matrix,
    lowercase_irrep,
    overlap_matrix,
    render_diagram_page,
    svg_sub_digits,
)
from .mo_diagram_pyscf import (
    _atom_permutation,
    _format_formula,
    parse_fragment_spec,
)
from .molecular_salc import (
    _hm_symbol,
    _match_operations,
    _table_operations_cartesian,
    get_symmetry,
    load_molecule,
)
from .operations import wigner_D_real


@dataclass
class FragmentLevel:
    column: str
    energy: float                # eV
    degeneracy: int
    label: str
    electrons: int
    irrep: str | None
    level_id: str
    vectors: np.ndarray          # (n_ao, degeneracy), full AO space
    composition: list = field(default_factory=list)
    detail: str = ""


class EhtFragmentDiagram:
    """MO diagram of an arbitrary two-fragment split in one EHT AO space."""

    def __init__(self, xyz_path, tolerance=0.3, left_spec=None,
                 right_spec=None):
        self.xyz_path = xyz_path
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
            table_ops, table_classes = _table_operations_cartesian(
                self.character_table
            )
            alignment, matched = _match_operations(
                operations, table_ops, table_classes
            )
            self.operations = [table_ops[i] for i in matched]
            self.operation_classes = [table_classes[i] for i in matched]
            self.coordinates = self.coordinates @ alignment.T
            self._symmetrize_coordinates(tolerance)

        # ---- fragments (explicit split only; the default center/ligand
        # architecture is the classic 4-column MODiagram)
        self._identify_fragments(left_spec, right_spec)
        self.formula = _format_formula(
            {s: self.symbols.count(s) for s in set(self.symbols)}, hill=True
        )

        # ---- one EHT AO space for all three columns
        self.orbitals = build_basis(self.symbols, list(range(len(self.symbols))))
        self.S = overlap_matrix(self.orbitals, self.coordinates)
        self.H = hamiltonian_matrix(self.orbitals, self.S)
        self.rows = {
            "mo": np.arange(len(self.orbitals)),
            "left": np.array([i for i, ao in enumerate(self.orbitals)
                              if ao.atom in set(self.left)], dtype=int),
            "right": np.array([i for i, ao in enumerate(self.orbitals)
                               if ao.atom in set(self.right)], dtype=int),
        }
        self.electron_counts = {
            "mo": sum(VALENCE_ELECTRONS[s] for s in self.symbols),
            "left": sum(VALENCE_ELECTRONS[self.symbols[i]] for i in self.left),
            "right": sum(VALENCE_ELECTRONS[self.symbols[i]] for i in self.right),
        }

        self._build_levels()
        self._link_levels()
        self._assign_bond_characters()

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

    def _identify_fragments(self, left_spec, right_spec):
        if left_spec is None or right_spec is None:
            raise SystemExit("ERROR: give both --ao-left and --ao-right.")
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
            molecule_formula = " ".join(
                f"{k}{v}" for k, v in sorted(totals.items())
            )
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

    # ------------------------------------------------------------- symmetry

    def _valid_operations(self, column):
        """Table operations that map the column's atoms onto themselves
        (and the other fragment onto itself), with the atom permutations."""
        inside = set(self.left if column == "left" else
                     self.right if column == "right" else
                     range(len(self.symbols)))
        tags = [(symbol, i in inside) for i, symbol in enumerate(self.symbols)]
        valid = []
        for rotation, class_name in zip(self.operations, self.operation_classes):
            permutation = _atom_permutation(self.coordinates, tags, rotation)
            if permutation is not None:
                valid.append((rotation, class_name, permutation))
        return valid

    def _ao_representation(self, rotation, permutation):
        """Matrix of the symmetry operation on the EHT AO basis (real
        orbitals in the wigner_D_real component order)."""
        n_ao = len(self.orbitals)
        gamma = np.zeros((n_ao, n_ao))
        blocks: dict[int, np.ndarray] = {}
        index = {(ao.atom, ao.shell, ao.m): i
                 for i, ao in enumerate(self.orbitals)}
        for i, ao in enumerate(self.orbitals):
            if ao.l not in blocks:
                blocks[ao.l] = wigner_D_real(ao.l, rotation)
            target_atom = permutation[ao.atom]
            D = blocks[ao.l]
            for m in range(2 * ao.l + 1):
                gamma[index[(target_atom, ao.shell, m)], i] = D[m, ao.m]
        return gamma

    def _irrep_labels(self, column, vectors_by_group):
        """Irrep label per degenerate group, or None when the fragment is
        not invariant under the full point group (or no table exists)."""
        if self.character_table is None:
            return None
        valid = self._valid_operations(column)
        if len(valid) != len(self.operations):
            return None
        rotation_list = list(self.character_table["rotation_list"])
        table = self.character_table["character_table"]
        representatives = {}
        for rotation, class_name, permutation in valid:
            representatives.setdefault(class_name, (rotation, permutation))
        gammas = {
            class_name: self._ao_representation(rotation, permutation)
            for class_name, (rotation, permutation) in representatives.items()
        }
        labels = []
        for vectors in vectors_by_group:
            characters = [
                float(np.trace(vectors.T @ self.S @ gammas[c] @ vectors))
                for c in rotation_list
            ]
            best, best_error = None, 1e9
            for irrep, chars in table.items():
                row = [float(np.real(c)) for c in np.atleast_1d(chars)]
                error = max(abs(a - b) for a, b in zip(characters, row))
                if error < best_error:
                    best, best_error = irrep, error
            labels.append(best if best_error < 0.05 else None)
        return labels

    def _split_group_by_irrep(self, column, vectors):
        """Decompose an exactly degenerate unlabeled group with the irrep
        projectors P = (d/|G|) sum chi(g) Gamma(g).

        A diatomic fragment inside a low-symmetry molecule keeps its own
        higher symmetry (e.g. the CO pi pair of CH3OH under Cs): the pair is
        exactly degenerate, chi(E) = 2 matches no Cs irrep, and no energy
        tolerance can split it -- but the projectors can (a' + a'').
        Returns [(irrep, vectors)] or None when the decomposition fails."""
        if self.character_table is None:
            return None
        valid = self._valid_operations(column)
        if len(valid) != len(self.operations):
            return None
        rotation_list = list(self.character_table["rotation_list"])
        table = self.character_table["character_table"]
        e_index = rotation_list.index("E")
        gammas = [self._ao_representation(rotation, permutation)
                  for rotation, _, permutation in valid]
        pieces = []
        for irrep, chars in table.items():
            row = {c: float(np.real(v)) for c, v in
                   zip(rotation_list, np.atleast_1d(chars))}
            dimension = row["E"] if "E" in row else float(
                np.real(np.atleast_1d(chars)[e_index])
            )
            projector = sum(
                row[class_name] * gamma
                for (_, class_name, _), gamma in zip(valid, gammas)
            ) * (dimension / len(valid))
            projected = projector @ vectors
            gram = projected.T @ self.S @ projected
            values, mixing = np.linalg.eigh(gram)
            keep = values > 1e-6
            if np.any(keep):
                pieces.append((irrep,
                               projected @ mixing[:, keep]
                               / np.sqrt(values[keep])))
        if sum(piece.shape[1] for _, piece in pieces) != vectors.shape[1]:
            return None
        return pieces

    def _core_counts(self, column):
        """Number of omitted core levels per irrep of this column (the EHT
        basis is valence-only; cores only shift the MO numbering, matching
        the all-electron PySCF labels)."""
        if self.character_table is None:
            return {}
        valid = self._valid_operations(column)
        if len(valid) != len(self.operations):
            return {}
        sites = (self.left if column == "left" else
                 self.right if column == "right" else
                 list(range(len(self.symbols))))
        rotation_list = list(self.character_table["rotation_list"])
        table = self.character_table["character_table"]
        class_size: dict[str, int] = {}
        for name in self.operation_classes:
            class_size[name] = class_size.get(name, 0) + 1
        order = len(self.operations)
        counts: dict[str, int] = {}
        for element in sorted({self.symbols[i] for i in sites}):
            element_sites = {i for i in sites if self.symbols[i] == element}
            for core_shell in CORE_SHELLS.get(element, []):
                l = {"s": 0, "p": 1, "d": 2, "f": 3}[core_shell[-1]]
                # class characters of (site permutation) x (orbital rotation)
                characters = {}
                for rotation, class_name, permutation in valid:
                    if class_name in characters:
                        continue
                    fixed = sum(1 for i in element_sites if permutation[i] == i)
                    characters[class_name] = (
                        fixed * float(np.trace(wigner_D_real(l, rotation)))
                    )
                for irrep, chars in table.items():
                    row = [float(np.real(c)) for c in np.atleast_1d(chars)]
                    # <chi, chi> = 2 for the physically-irreducible merged
                    # conjugate pairs of the C3/C4/C6/S4/S6/C3h/C4h/C6h/T/Th
                    # tables (one chi(E) = 2 row per pair): the multiplicity
                    # of the PAIR is the raw projection divided by that norm
                    row_norm = sum(
                        class_size[c] * row[k] ** 2
                        for k, c in enumerate(rotation_list)
                    ) / order
                    n = sum(
                        class_size[c] * characters[c] * row[k]
                        for k, c in enumerate(rotation_list)
                    ) / order / row_norm
                    n = int(round(n))
                    if n:
                        counts[irrep] = counts.get(irrep, 0) + n
        return counts

    # --------------------------------------------------------------- levels

    def _solve_column(self, column):
        """Generalized eigenstates of the column's (H, S) sub-block,
        embedded back into the full AO space (canonical orthogonalization)."""
        rows = self.rows[column]
        H = self.H[np.ix_(rows, rows)]
        S = self.S[np.ix_(rows, rows)]
        values, vectors = np.linalg.eigh(S)
        keep = values > 1e-8
        if not np.all(keep):
            print(f"NOTE: dropped {int((~keep).sum())} near-dependent basis "
                  f"combination(s) in the {column} block.")
        X = vectors[:, keep] / np.sqrt(values[keep])
        energies, transformed = np.linalg.eigh(X.T @ H @ X)
        C = np.zeros((len(self.orbitals), len(energies)))
        C[rows] = X @ transformed
        return energies, C

    def _build_levels(self):
        self.levels: dict[str, list[FragmentLevel]] = {}
        for column in ("left", "mo", "right"):
            energies, C = self._solve_column(column)
            groups = _group_degenerate(energies, tol=1e-4)
            labels = self._irrep_labels(
                column, [C[:, group] for group in groups]
            )
            if labels is not None:
                # an accidental near-degeneracy can merge different irreps
                # into one unlabeled group: re-split those finely and relabel
                refined = []
                for group, label in zip(groups, labels):
                    if label is None and len(group) > 1:
                        subgroups = _group_degenerate(energies[group], tol=1e-8)
                        refined.extend([[group[i] for i in sub]
                                        for sub in subgroups])
                    else:
                        refined.append(group)
                if len(refined) != len(groups):
                    groups = sorted(refined, key=lambda g: float(energies[g[0]]))
                    labels = self._irrep_labels(
                        column, [C[:, group] for group in groups]
                    )
            # (energy, vectors, irrep) entries; exactly degenerate unlabeled
            # groups (a higher-symmetry fragment inside a lower-symmetry
            # molecule) are decomposed with the irrep projectors
            entries = []
            for g_index, group in enumerate(groups):
                irrep = labels[g_index] if labels else None
                vectors = C[:, group]
                energy = float(np.mean(energies[group]))
                if labels is not None and irrep is None and len(group) > 1:
                    split = self._split_group_by_irrep(column, vectors)
                    if split is not None:
                        entries.extend(
                            (energy, piece, piece_irrep)
                            for piece_irrep, piece in split
                        )
                        continue
                entries.append((energy, vectors, irrep))
            counters = dict(self._core_counts(column))
            column_levels = []
            electrons_left = self.electron_counts[column]
            for g_index, (energy, vectors, irrep) in enumerate(entries):
                if irrep is not None:
                    counters[irrep] = counters.get(irrep, 0) + 1
                    label = f"{counters[irrep]}{lowercase_irrep(irrep)}"
                else:
                    label = f"{g_index + 1}"
                electrons = int(min(electrons_left, 2 * vectors.shape[1]))
                electrons_left -= electrons
                column_levels.append(FragmentLevel(
                    column=column, energy=energy,
                    degeneracy=vectors.shape[1], label=label,
                    electrons=electrons, irrep=irrep,
                    level_id=f"{column}_{g_index}", vectors=vectors,
                ))
            self.levels[column] = column_levels

        occupied = [l for l in self.levels["mo"] if l.electrons > 0]
        empty = [l for l in self.levels["mo"] if l.electrons == 0]
        self.homo = occupied[-1] if occupied else None
        self.lumo = empty[0] if empty else None

    def _link_levels(self):
        """Project the molecule MOs onto the fragment MOs (same AO space)."""
        for level in self.levels["mo"]:
            weights = []
            for column in ("left", "right"):
                for fragment_level in self.levels[column]:
                    P = fragment_level.vectors.T @ self.S @ level.vectors
                    weight = float(np.sum(P ** 2))
                    if weight > 1e-6:
                        weights.append((fragment_level, weight))
            total = sum(w for _, w in weights) or 1.0
            level.composition = [
                (fragment_level.level_id, weight / total)
                for fragment_level, weight in weights
            ]
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

    def _assign_bond_characters(self):
        """COOP bonding character via the shared crystal-engine classifier
        (crystal_orbital_diagram.assign_bond_characters); fragment levels are
        also tagged with their dominant (element, shell) for the VESTA line
        colors, like the PySCF engine."""
        from types import SimpleNamespace

        from .crystal_orbital_diagram import assign_bond_characters

        values, vectors = np.linalg.eigh(self.S)
        sqrt_overlap = (
            vectors * np.sqrt(np.clip(values, 0.0, None))
        ) @ vectors.T
        spec_lists: dict[tuple[str, str], list[int]] = {}
        for i, ao in enumerate(self.orbitals):
            spec_lists.setdefault((ao.element, ao.shell), []).append(i)
        spec_ranges = {
            key: np.array(indices, dtype=int)
            for key, indices in spec_lists.items()
        }
        proxy_levels: dict[str, list] = {}
        for column in ("left", "right"):
            row_set = set(self.rows[column].tolist())
            side_ranges = {
                key: np.array([i for i in indices if i in row_set], dtype=int)
                for key, indices in spec_ranges.items()
                if any(i in row_set for i in indices)
            }
            proxies = []
            for level in self.levels[column]:
                gross = (np.abs(sqrt_overlap @ level.vectors) ** 2).sum(axis=1)
                dominant = max(
                    side_ranges,
                    key=lambda key: float(gross[side_ranges[key]].sum()),
                )
                level.dominant_spec = dominant
                proxies.append(SimpleNamespace(
                    label=f"{dominant[0]} {dominant[1]} {level.label}",
                    energy=level.energy,
                    electrons=level.electrons,
                ))
            proxy_levels[column] = proxies
        proxy_levels["mo"] = [
            SimpleNamespace(
                vectors=level.vectors, degeneracy=level.degeneracy,
                electrons=level.electrons, energy=level.energy,
                label=level.label, detail="",
            )
            for level in self.levels["mo"]
        ]
        assign_bond_characters(
            proxy_levels, self.S, self.rows["left"], self.rows["right"],
            spec_ranges, sqrt_overlap=sqrt_overlap,
        )
        for level, proxy in zip(self.levels["mo"], proxy_levels["mo"]):
            level.bond_character = proxy.bond_character
            level.overlap_population = proxy.overlap_population
            level.detail += proxy.detail

    # --------------------------------------------------------------- output

    def _column_name(self, column):
        return {"left": self.left_name, "mo": self.formula,
                "right": self.right_name}[column]

    def _display_name(self, column):
        name = self._column_name(column)
        if column != "mo" and self.left_name == self.right_name:
            return f"{name}({'L' if column == 'left' else 'R'})"
        return name

    def _sketch_partners(self, level):
        """Per-atom orbital amplitudes of every partner (minimal STO basis:
        the bare coefficients are the drawn lobes, like the single-center
        EHT diagram; 9 slots when the basis carries d orbitals)."""
        has_d = any(ao.l == 2 for ao in self.orbitals)
        width = 9 if has_d else 4
        slot_of = {0: 0, 1: 1, 2: 4}
        partners = []
        for k in range(level.vectors.shape[1]):
            vector = level.vectors[:, k]
            per_atom: dict[int, list[float]] = {}
            for i, ao in enumerate(self.orbitals):
                if ao.l > 2 or abs(vector[i]) < 1e-10:
                    continue
                values = per_atom.setdefault(ao.atom, [0.0] * width)
                values[slot_of[ao.l] + ao.m] += float(vector[i])
            partners.append(per_atom)
        if not has_d:
            partners = canonical_sketch_partners(partners)
        return [_sketch_entries(per_atom) for per_atom in partners]

    def print_report(self):
        print("\n* Molecule *")
        print(f"{self.xyz_path} ({self.formula}, {len(self.symbols)} atoms)")
        print("\n* Point group *")
        print(self.schoenflies + (f" ({self.hm})" if self.hm else ""))
        print(f"\n* Fragments ({self.left_name} | {self.right_name}) *")
        for name, sites in (("left", self.left), ("right", self.right)):
            atoms = " ".join(f"{self.symbols[i]}{i}" for i in sites)
            print(f"{self._display_name(name):>8s}: {atoms} "
                  f"({self.electron_counts[name]} valence e-)")
        print("\n* Molecular orbitals (extended Hueckel, Wolfsberg-Helmholz "
              "K = 1.75) *")
        names = {
            level.level_id: f"{self._display_name(level.column)} {level.label}"
            for column_levels in self.levels.values() for level in column_levels
        }
        print(f"{'MO':>7s} {'E (eV)':>10s} {'occ':>4s}  composition")
        ceiling = self.lumo.energy + 12.0 if self.lumo else np.inf
        for level in reversed(self.levels["mo"]):
            if level.energy > ceiling:
                continue
            parts = ", ".join(
                f"{100 * w:.0f}% {names[i]}"
                for i, w in sorted(level.composition, key=lambda kv: -kv[1])
                if w > 0.02
            )
            tag = f"{level.label} x{level.degeneracy}" if level.degeneracy > 1 \
                else level.label
            print(f"{tag:>7s} {level.energy:10.2f} {level.electrons:4d}  {parts}")
        filling = " ".join(
            f"({level.label})^{level.electrons}"
            for level in self.levels["mo"] if level.electrons
        )
        print(f"\n* Electron filling ({self.electron_counts['mo']} valence "
              "electrons) *")
        print(filling)
        if self.homo and self.lumo:
            print(f"HOMO = {self.homo.label} ({self.homo.energy:.2f} eV), "
                  f"LUMO = {self.lumo.label} ({self.lumo.energy:.2f} eV), "
                  f"gap = {self.lumo.energy - self.homo.energy:.2f} eV")
        print("\nMethod: two-fragment symmetry + overlap MO diagram "
              "(extended Hueckel)")

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
        bond_letter = {"bonding": "b", "nonbonding": "n", "antibonding": "a"}
        levels_json = []
        for column_levels in self.levels.values():
            for level in column_levels:
                character = getattr(level, "bond_character", None)
                spec = getattr(level, "dominant_spec", None)
                elc = (element_color(spec[0])
                       if level.column != "mo" and spec else None)
                levels_json.append({
                    **({"elc": elc} if elc else {}),
                    "id": level.level_id,
                    "col": level.column,
                    "e": round(level.energy, 4),
                    "deg": level.degeneracy,
                    **({"bond": bond_letter[character]} if character else {}),
                    "label": level.label,
                    "el": level.electrons,
                    "occ": level.electrons > 0,
                    "links": [
                        [i, round(w, 4)] for i, w in level.composition
                        if w >= 0.02
                    ],
                    "comp": [
                        [names[i], round(100 * w, 1)]
                        for i, w in sorted(level.composition,
                                           key=lambda kv: -kv[1])
                        if w > 0.005
                    ],
                    "detail": level.detail,
                    "orb": self._sketch_partners(level) or None,
                })
        all_energies = [
            level.energy for column_levels in self.levels.values()
            for level in column_levels
        ]
        e_min = min(all_energies) - 3.0
        e_max = (self.lumo.energy + 8.0) if self.lumo else max(all_energies)
        e_max = min(e_max, max(all_energies) + 2.0)
        formula_html = re.sub(r"(\d+)", r"<sub>\1</sub>", self.formula)
        gap = ""
        if self.homo and self.lumo:
            gap = (f"HOMO&ndash;LUMO gap "
                   f"{self.lumo.energy - self.homo.energy:.2f} eV")
        chips = [
            formula_html,
            f"{self.schoenflies}" + (f" ({self.hm})" if self.hm else ""),
            f"{self.left_name} + {self.right_name}",
            "extended H&uuml;ckel / STO overlaps",
            gap,
        ]
        render_diagram_page(
            output_path,
            title=f"MO diagram: {self.formula} ({self.left_name} + "
                  f"{self.right_name})",
            heading_html=f"Molecular-orbital diagram: {formula_html} "
                         f"<span style=\"color:#90a4ae;font-size:14px\">"
                         f"{self.left_name} + {self.right_name}, "
                         f"extended H&uuml;ckel</span>",
            chips=chips,
            columns=columns, half=half, order=order, side=side,
            headers=headers, levels_json=levels_json,
            homo_id=self.homo.level_id if self.homo else None,
            lumo_id=self.lumo.level_id if self.lumo else None,
            e_min=e_min, e_max=e_max,
            geometry=diagram_geometry(self.symbols, self.coordinates),
            foot_html=(
                "Semi-quantitative two-fragment diagram from symmetry + "
                "overlap only (extended H&uuml;ckel, Wolfsberg&ndash;Helmholz "
                "K = 1.75, single-&zeta; STOs, exact two-center overlap "
                "integrals; all three columns solved in the one molecular AO "
                "space, fragment levels from the fragment's own (H, S) "
                "sub-block, molecule MOs projected onto them through the "
                "shared overlap matrix). Valence shells only &mdash; core "
                "shells enter the MO numbering, not the diagram."
                + (" MO line colors: "
                   "<span style=\"color:#1565c0\">bonding</span> / "
                   "<span style=\"color:#333\">nonbonding</span> / "
                   "<span style=\"color:#d32f2f\">antibonding</span>, from "
                   "the left&ndash;right overlap population 2 Re "
                   "c<sub>L</sub>&#8224;S c<sub>R</sub> of each state "
                   "(COOP-style; value in the level's tooltip)."
                   if any(getattr(level, "bond_character", None)
                          for level in self.levels["mo"]) else "")
                + " Generated by CrystOD (crystod-mol --diagram "
                "--ao-left/--ao-right)."
            ),
        )


def run_fragment_diagram(args) -> None:
    diagram = EhtFragmentDiagram(
        args.xyz, args.tolerance,
        left_spec=args.ao_left, right_spec=args.ao_right,
    )
    diagram.print_report()
    stem = os.path.splitext(os.path.basename(args.xyz))[0]
    output_path = args.output or f"MolOD_{stem}.html"
    diagram.write_html(output_path)
    print(f"\nMO diagram written to {output_path}")
