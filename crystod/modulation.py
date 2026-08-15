"""
Symmetry-adapted phonon modulation workflow for crystod.
"""

from __future__ import annotations

from dataclasses import dataclass
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, RawDescriptionHelpFormatter, RawTextHelpFormatter
from fractions import Fraction
from pathlib import Path
import re

import numpy as np
import phonopy
import spglib
from ase import Atoms
from ase.io import write as ase_write
from numpy.typing import NDArray

from .operations import parse_qpoint_token
from .runtime_compat import (
    SymmetryDatasetAdapter,
    get_chemical_symbols,
    get_little_group,
    get_scaled_positions,
)
from .spglib_compat import ensure_spglib_compat

ensure_spglib_compat()

from phonopy.structure.atoms import PhonopyAtoms
from spgrep.core import get_spacegroup_irreps_from_primitive_symmetry
from spgrep.representation import project_to_irrep


class MyHelpFormatter(
    RawTextHelpFormatter,
    RawDescriptionHelpFormatter,
    ArgumentDefaultsHelpFormatter,
):
    pass


desc = """
Generate modulated crystal structures from symmetry-adapted phonon modes.

# Command Example:
crystod-phonon --modulation --yaml phonopy_params.yaml --qpoint 0.5 0.5 0.5                      (list modes and star of q only)
crystod-phonon --modulation --yaml phonopy_params.yaml --qpoint 0.5 0.5 0.5 --mode 1 2 3 --amplitude 0.3
crystod-phonon --modulation --yaml phonopy_params.yaml --qpoint1 0 0.5 0.5 --mode1 1 --qpoint2 0.5 0 0.5 --mode2 1 --output POSCAR_combined
"""


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=desc, formatter_class=MyHelpFormatter)
    parser.add_argument(
        "--yaml",
        dest="yaml_path",
        default=None,
        help="Path to phonopy_params.yaml(.xz) (default: phonopy_params.yaml "
        "when no structure file is given).",
    )
    parser.add_argument(
        "--poscar",
        "-c",
        "--cell",
        dest="cell",
        default=None,
        help="Unit-cell file, used with FORCE_SETS (or FORCE_CONSTANTS with "
        "--readfc) instead of a phonopy yaml.",
    )
    parser.add_argument(
        "--dim",
        default=None,
        help='Supercell of the force calculation, e.g. "4 4 4". Inferred from '
        "phonopy_disp.yaml or from the force file when omitted.",
    )
    parser.add_argument(
        "--readfc",
        action="store_true",
        help="Read FORCE_CONSTANTS instead of FORCE_SETS.",
    )
    parser.add_argument(
        "--qpoint",
        nargs=3,
        type=parse_qpoint_token,
        help="Target q-point in primitive reciprocal coordinates "
        "(fractions such as 1/3 are allowed).",
    )
    parser.add_argument(
        "--mode",
        nargs="+",
        type=int,
        help="Mode number(s) to apply, 1-based as in the printed mode table.\n"
        "If omitted, only the mode table and the star of q are printed.",
    )
    parser.add_argument(
        "--amplitude",
        nargs="+",
        type=float,
        default=[0.3],
        help="Modulation amplitude(s) in Angstroms.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output POSCAR path (default: MPOSCAR_{q}_{mode}_{irrep}_{subgroup}).",
    )
    parser.add_argument(
        "--tolerance",
        "--symprec",
        dest="symprec",
        type=float,
        # sentinel: the mode construction and the space group of the generated
        # structure want different defaults (1e-5 / 0.1), but an explicit
        # --tolerance has to reach BOTH -- it used to reach only the first
        default=None,
        help="Symmetry tolerance (default: 1e-5 for the mode construction, "
        "0.1 for the space group of the generated structure).",
    )
    parser.add_argument(
        "--keep-q-coords",
        dest="keep_q_coords",
        action="store_true",
        help="Name output files of a non-special q with its coordinates "
        "(q_<coords>) instead of the ISO-IR k-vector-type label, so scans "
        "along one symmetry line do not overwrite each other.",
    )
    return parser


def _find_intertwiner(
    rep: NDArray[np.complex128],
    space_s: NDArray[np.complex128],
    space_r: NDArray[np.complex128],
) -> NDArray[np.complex128] | None:
    """Unitary aligning the irrep basis of space_s to that of space_r.

    Both spaces must carry equivalent irreps of the same (projective)
    representation ``rep``; returns None when they are inequivalent.
    """
    dim = space_s.shape[0]
    d_s = np.array([space_s @ G @ space_s.conj().T for G in rep])
    d_r = np.array([space_r @ G @ space_r.conj().T for G in rep])
    for seed_index in range(dim * dim):
        seed = np.zeros((dim, dim))
        seed[seed_index // dim, seed_index % dim] = 1.0
        averaged = sum(a @ seed @ b.conj().T for a, b in zip(d_s, d_r)) / len(rep)
        if np.linalg.norm(averaged) > 1e-6:
            u, _, vh = np.linalg.svd(averaged)
            return u @ vh
    return None


# ---------------------------------------------------------------------------
# phonopy input: a phonopy yaml, or a unit cell + FORCE_SETS/FORCE_CONSTANTS
# ---------------------------------------------------------------------------

DEFAULT_PARAMS_YAML = "phonopy_params.yaml"

# where the supercell shape may be recorded, best source first
_SUPERCELL_YAML_CANDIDATES = (
    "phonopy_disp.yaml",
    "phonopy_disp.yaml.xz",
    "phonopy_params.yaml",
    "phonopy_params.yaml.xz",
)

# "the file states a supercell this workflow cannot use" -- distinct from "the
# file says nothing about the supercell", because guessing over a stated
# non-diagonal supercell would produce a wrong answer that nothing downstream
# can detect: the guess has the right atom count by construction
_DECLARED_UNUSABLE = object()


def _open_text(path: Path):
    """Open a phonopy file, transparently handling the .xz form."""
    if str(path).endswith(".xz"):
        import lzma

        return lzma.open(path, "rt")
    return open(path, "r")


def _diagonal_supercell_from_yaml(path: Path) -> list[int] | None:
    """Read ``supercell_matrix`` out of a phonopy yaml, header only.

    phonopy_params.yaml carries the whole force-constant matrix (hundreds of
    kB), so the file is scanned line by line and the scan stops as soon as the
    three rows have been read. Returns None when the file has no such block or
    the supercell is not diagonal.
    """
    rows: list[list[int]] = []
    inside = False
    try:
        with _open_text(path) as handle:
            for line in handle:
                stripped = line.strip()
                if not inside:
                    if stripped.startswith("supercell_matrix:"):
                        inside = True
                    continue
                match = re.match(r"^-\s*\[([-\d\s,]+)\]$", stripped)
                if not match:
                    break
                try:
                    rows.append([int(v) for v in match.group(1).replace(",", " ").split()])
                except ValueError:
                    return _DECLARED_UNUSABLE
                if len(rows) == 3:
                    break
    # a candidate file the user never named: any read failure (missing,
    # unreadable, a .xz that is not one, a stray binary) means "says nothing"
    except Exception:
        return None
    if not rows:
        return None  # no supercell_matrix block at all
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        return _DECLARED_UNUSABLE
    if any(rows[i][j] for i in range(3) for j in range(3) if i != j):
        return _DECLARED_UNUSABLE  # non-diagonal: not supported by this workflow
    diagonal = [rows[0][0], rows[1][1], rows[2][2]]
    return diagonal if all(value > 0 for value in diagonal) else _DECLARED_UNUSABLE


def _supercell_atom_count(force_path: Path, readfc: bool) -> int | None:
    """Supercell atom count from the header of FORCE_SETS/FORCE_CONSTANTS."""
    try:
        with open(force_path) as handle:
            for line in handle:
                tokens = line.split()
                if not tokens:
                    continue
                try:
                    numbers = [int(token) for token in tokens]
                except ValueError:
                    return None
                # FORCE_SETS opens with the supercell atom count; FORCE_CONSTANTS
                # opens with "n_satom" (full) or "n_patom n_satom" (compact).
                return max(numbers) if readfc else numbers[0]
    except OSError:
        return None
    return None


def _infer_diagonal_supercell(lattice, multiplicity: int) -> list[int]:
    """Most isotropic diagonal supercell of the requested volume multiplicity.

    A force file records only how many atoms its supercell holds, never the
    shape, so the shape has to be guessed when nothing else states it. Two
    rules pick it: axes of equal length keep equal multipliers (a supercell
    that breaks the lattice's own axis equivalence is not one anybody builds),
    and among what is left the most nearly cubic supercell wins. ``--dim``
    overrides the guess and the choice is always printed.
    """
    lengths = np.linalg.norm(np.asarray(lattice, dtype=float), axis=1)
    equivalent = [
        [j for j in range(3) if abs(lengths[j] - lengths[i]) <= 1e-4 * lengths[i]]
        for i in range(3)
    ]

    def search(respect_equivalence: bool) -> list[int] | None:
        best: list[int] | None = None
        best_score: float | None = None
        for n1 in range(1, multiplicity + 1):
            if multiplicity % n1:
                continue
            rest = multiplicity // n1
            for n2 in range(1, rest + 1):
                if rest % n2:
                    continue
                n3 = rest // n2
                counts = [n1, n2, n3]
                if respect_equivalence and any(
                    counts[j] != counts[i] for i in range(3) for j in equivalent[i]
                ):
                    continue
                edges = np.array(counts, dtype=float) * lengths
                score = float(edges.max() / edges.min())
                if best_score is None or score < best_score - 1e-12:
                    best, best_score = counts, score
        return best

    return search(True) or search(False)


def load_phonon(
    yaml_path: str | None = None,
    cell_path: str | None = None,
    dim: str | list[int] | None = None,
    readfc: bool = False,
) -> tuple[object, str, str]:
    """Build the phonopy object of a modulation run.

    Either ``yaml_path`` (a phonopy_params.yaml) or ``cell_path`` (a unit cell
    next to FORCE_SETS/FORCE_CONSTANTS) is used. Returns the object, a short
    label naming the input files, and a note describing where the supercell
    came from -- which the caller prints, so that an inferred supercell is
    never silent.
    """
    if yaml_path is not None:
        path = Path(yaml_path)
        if not path.exists():
            raise SystemExit(f"ERROR: '{path}' does not exist.")
        return phonopy.load(str(path)), str(path), ""

    if cell_path is None:
        raise SystemExit("ERROR: either a phonopy yaml or a unit-cell file is required.")
    cell = Path(cell_path)
    if not cell.exists():
        raise SystemExit(f"ERROR: '{cell}' does not exist.")

    force_name = "FORCE_CONSTANTS" if readfc else "FORCE_SETS"
    force_path = Path(force_name)
    if not force_path.exists() and cell.parent != Path("."):
        force_path = cell.parent / force_name
    if not force_path.exists():
        raise SystemExit(
            f"ERROR: {force_name} not found next to '{cell}' or in the current "
            "directory. --modulation needs FORCE_SETS (or FORCE_CONSTANTS with "
            "--readfc), or a phonopy yaml given with --yaml."
        )

    if dim:
        tokens = dim.split() if isinstance(dim, str) else [str(value) for value in dim]
        try:
            diagonal = [int(token) for token in tokens]
        except ValueError:
            raise SystemExit(f'ERROR: --dim requires integers, got: {" ".join(tokens)}')
        if len(diagonal) != 3 or any(value <= 0 for value in diagonal):
            raise SystemExit("ERROR: --dim requires three positive integers.")
        note = ""  # explicit: nothing to report back
    else:
        diagonal = None
        note = ""
        for candidate in _SUPERCELL_YAML_CANDIDATES:
            found = _diagonal_supercell_from_yaml(Path(candidate))
            if found is _DECLARED_UNUSABLE:
                raise SystemExit(
                    f"ERROR: '{candidate}' states a supercell_matrix that is not a "
                    "positive diagonal matrix; this workflow supports diagonal "
                    "supercells only. Inferring one instead would silently give the "
                    "wrong answer, since any guess with the right atom count loads "
                    "without complaint. Use --yaml phonopy_params.yaml, or give the "
                    'diagonal supercell with --dim "n n n".'
                )
            if found:
                shape = "x".join(str(value) for value in found)
                diagonal = found
                note = f"Supercell {shape} read from {candidate}."
                break
        if diagonal is None:
            from phonopy.interface.vasp import read_vasp

            unitcell = read_vasp(str(cell))
            n_unit = len(unitcell.scaled_positions)
            n_super = _supercell_atom_count(force_path, readfc)
            if not n_super or n_unit <= 0 or n_super % n_unit:
                raise SystemExit(
                    f"ERROR: cannot infer the supercell of '{force_path}' from "
                    f"'{cell}'; give it explicitly, e.g. --dim \"2 2 2\"."
                )
            diagonal = _infer_diagonal_supercell(unitcell.cell, n_super // n_unit)
            shape = "x".join(str(value) for value in diagonal)
            note = (
                f"Supercell {shape} inferred from the {n_super} atoms of "
                f"{force_path.name}; pass --dim if that is not the supercell "
                "of your force calculation."
            )

    try:
        phonon = phonopy.load(
            supercell_matrix=diagonal,
            # "auto" as in --irreps/--fatband/--vector/--subgroup: a conventional
            # centred cell handed to -c means the primitive-cell phonons, and
            # --subgroup --modulate prints --modulation commands that have to
            # reproduce its own structures exactly
            primitive_matrix="auto",
            unitcell_filename=str(cell),
            force_sets_filename=None if readfc else str(force_path),
            force_constants_filename=str(force_path) if readfc else None,
        )
    except (ValueError, RuntimeError) as exc:
        # phonopy reports every one of these as a bare traceback. Do not assert
        # a single diagnosis: an inconsistent unit cell and a truncated force
        # file both land here, and blaming the supercell then sends the user
        # after the one thing that is right.
        text = " ".join(str(exc).split())
        if isinstance(exc, RecursionError):
            detail = (
                f"phonopy could not parse '{force_path}' ({text}); the file is "
                "most likely truncated or malformed."
            )
        else:
            shape = "x".join(str(value) for value in diagonal)
            where = f" ({note.rstrip('.')})" if note else ""
            detail = (
                f"phonopy could not build force constants from '{cell}' + "
                f"'{force_path}' with supercell {shape}{where}: {text}\n"
                "       Check that the unit cell is the one the forces were "
                'calculated for, and that the supercell is right (--dim "n n n").'
            )
        raise SystemExit(f"ERROR: {detail}") from None
    primitive_note = (
        f"Primitive cell: {len(phonon.primitive)} atoms of the "
        f"{len(phonon.unitcell)}-atom input cell (primitive_matrix auto)."
    )
    note = f"{note} {primitive_note}" if note else primitive_note
    return phonon, f"{cell} + {force_path.name}", note


def _irrep_filename_tag(labels: list[str]) -> str:
    """Compact irrep tag for file names: drop the dimension suffix "(n)" and
    join distinct labels with '+' (e.g. "GM1(1), GM5(2)" -> "GM1+GM5")."""
    cleaned: list[str] = []
    for label in labels:
        for part in label.split(","):
            part = re.sub(r"\(\d+\)", "", part).strip()
            if part and part != "-" and part not in cleaned:
                cleaned.append(part)
    return "+".join(cleaned)


@dataclass
class ModulationTerm:
    qpoint: list[float]
    mode_indices: list[int]
    amplitudes: list[float]


@dataclass
class PreparedModulationTerm:
    modulation: SymmetryAdaptedModulation
    mode_indices: list[int]
    amplitudes: list[float]


class _CoreRepresentation:
    def __init__(self, cell: PhonopyAtoms, symprec: float = 1e-5):
        self.input_cell = cell
        primitive_lattice, primitive_pos, primitive_numbers = spglib.standardize_cell(
            cell.totuple(),
            to_primitive=True,
            symprec=symprec,
        )
        self.primitive_cell = PhonopyAtoms(
            numbers=primitive_numbers,
            scaled_positions=primitive_pos,
            cell=primitive_lattice,
        )
        print("\n ### Inputed cell was converted into primitive cell. ###")

        dataset = SymmetryDatasetAdapter(
            spglib.get_symmetry_dataset(self.primitive_cell.totuple(), symprec=symprec)
        )
        self.spglib_dataset = dataset
        self.rotations = dataset.rotations
        self.translations = dataset.translations

    def get_modified_permutation_rep(
        self,
        rotation: NDArray[np.int_],
        translation: NDArray[np.float64],
        kpoint: list[float],
    ) -> NDArray[np.complex128]:
        positions = get_scaled_positions(self.primitive_cell)
        num_atom = len(positions)
        matrix = np.zeros((num_atom, num_atom), dtype=complex)
        for i, pos_in in enumerate(positions):
            pos_rot = np.dot(rotation, pos_in) + translation
            for j, pos_out in enumerate(positions):
                diff = pos_rot - pos_out
                if (abs(diff - np.rint(diff)) < 1e-5).all():
                    phase_factor = np.dot(
                        kpoint,
                        np.dot(np.linalg.inv(rotation), pos_out - translation) - pos_out,
                    )
                    matrix[j, i] = np.exp(2j * np.pi * phase_factor)
        return matrix

    def get_permutation_reps_at_k(
        self,
        little_rotations: NDArray[np.int_],
        little_translations: NDArray[np.float64],
        kpoint: list[float],
    ) -> NDArray[np.complex128]:
        return np.array(
            [
                self.get_modified_permutation_rep(rotation, translation, kpoint)
                for rotation, translation in zip(little_rotations, little_translations)
            ],
            dtype=np.complex128,
        )

    def get_little_group(self, kpoint: list[float]):
        return get_little_group(
            rotations=self.rotations,
            translations=self.translations,
            kpoint=kpoint,
        )


class _Vibrations(_CoreRepresentation):
    def __init__(self, cell: PhonopyAtoms, symprec: float = 1e-5):
        super().__init__(cell=cell, symprec=symprec)
        lattice_t = np.transpose(self.primitive_cell.cell)
        lattice_t_inv = np.linalg.inv(lattice_t)
        self.rotations_cartesian = np.array(
            [lattice_t @ rotation @ lattice_t_inv for rotation in self.rotations],
            dtype=np.complex128,
        )

    def get_vibration_rep(self, kpoint: list[float]):
        irreps, mapping_little_group = get_spacegroup_irreps_from_primitive_symmetry(
            rotations=self.rotations,
            translations=self.translations,
            kpoint=kpoint,
        )
        little_rotations = self.rotations[mapping_little_group]
        little_translations = self.translations[mapping_little_group]
        permutation_matrices = self.get_permutation_reps_at_k(
            little_rotations=little_rotations,
            little_translations=little_translations,
            kpoint=kpoint,
        )
        cartesian_rep = self.rotations_cartesian[mapping_little_group]
        vibration_rep = np.array(
            [
                np.kron(permutation_matrix, cartesian_rotation)
                for permutation_matrix, cartesian_rotation in zip(permutation_matrices, cartesian_rep)
            ],
            dtype=np.complex128,
        )
        return irreps, vibration_rep

    def get_vibration_basis(self, irreps, vibration_rep) -> list[NDArray[np.complex128]]:
        basis_vectors: list[NDArray[np.complex128]] = []
        for irrep in irreps:
            basis_vectors.extend(project_to_irrep(vibration_rep, irrep))
        return basis_vectors


class SymmetryAdaptedModulation:
    def __init__(self, yaml_path: str | None = None, qpoint: list[float] | None = None,
                 symprec: float = 1e-5, keep_q_coords: bool = False,
                 phonon=None) -> None:
        if qpoint is None:
            raise ValueError("qpoint is required.")
        self.qpoint = np.array(qpoint, dtype=float)
        self.symprec = symprec
        self.keep_q_coords = keep_q_coords
        # A prebuilt phonopy object lets the same force data drive several q
        # points without reloading it, and lets the caller build it from a unit
        # cell + FORCE_SETS instead of a phonopy yaml.
        if phonon is None:
            if yaml_path is None:
                raise ValueError("either yaml_path or phonon is required.")
            phonon = phonopy.load(yaml_path)
        self.phonon = phonon
        dynamical_matrix = self.phonon.dynamical_matrix
        if dynamical_matrix is None:
            raise ValueError(
                "the phonopy object carries no force constants "
                "(FORCE_SETS/FORCE_CONSTANTS missing?)."
            )

        primitive = self.phonon.primitive
        primitive_atoms = PhonopyAtoms(
            numbers=primitive.numbers,
            scaled_positions=primitive.scaled_positions,
            cell=primitive.cell,
        )

        self.vibrations = _Vibrations(cell=primitive_atoms, symprec=symprec)
        irreps, vibration_rep = self.vibrations.get_vibration_rep(qpoint)
        vibration_basis = self.vibrations.get_vibration_basis(irreps, vibration_rep)
        self.irreps = irreps
        self.vibration_basis = vibration_basis

        dynamical_matrix.run(qpoint)
        raw_matrix = dynamical_matrix.dynamical_matrix.copy()
        primitive_positions = self.vibrations.primitive_cell.scaled_positions
        self.n_atoms = len(primitive_positions)

        phase = np.exp(2j * np.pi * np.dot(primitive_positions, qpoint))
        modified_matrix = np.zeros_like(raw_matrix)
        for i in range(self.n_atoms):
            for j in range(self.n_atoms):
                modified_matrix[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = (
                    raw_matrix[3 * i : 3 * i + 3, 3 * j : 3 * j + 3]
                    * np.conj(phase[i])
                    * phase[j]
                )

        spaces = vibration_basis
        dims = [space.shape[0] for space in spaces]
        if sum(dims) != modified_matrix.shape[0]:
            raise RuntimeError("Irrep projection does not span the full vibration space.")
        offsets = np.cumsum([0] + dims)
        n_spaces = len(spaces)
        stacked = np.vstack(spaces)
        block_matrix = stacked @ modified_matrix @ stacked.conj().T

        # Spaces carrying equivalent irreps couple through the dynamical
        # matrix; diagonalizing each projected block on its own would drop
        # that coupling and give wrong frequencies whenever an irrep occurs
        # more than once at q. Group coupled spaces into clusters and
        # diagonalize per cluster (same construction as crystod-phonon --vector).
        coupled = np.zeros((n_spaces, n_spaces), dtype=bool)
        for s in range(n_spaces):
            for t in range(n_spaces):
                sub = block_matrix[offsets[s] : offsets[s + 1], offsets[t] : offsets[t + 1]]
                coupled[s, t] = bool(np.abs(sub).max() > 1e-6)
        clusters: list[list[int]] = []
        seen: set[int] = set()
        for s in range(n_spaces):
            if s in seen:
                continue
            stack, cluster = [s], []
            while stack:
                u = stack.pop()
                if u in seen:
                    continue
                seen.add(u)
                cluster.append(u)
                stack.extend(v for v in range(n_spaces) if coupled[u, v] and v not in seen)
            clusters.append(sorted(cluster))

        self.mode_info: list[dict[str, float | int]] = []
        self.mode_vectors: list[NDArray[np.complex128]] = []
        for cluster in clusters:
            dim = dims[cluster[0]]
            if any(dims[index] != dim for index in cluster):
                raise RuntimeError("Coupled irrep spaces with different dimensions.")
            multiplicity = len(cluster)

            aligned = [spaces[cluster[0]]]
            for index in cluster[1:]:
                intertwiner = _find_intertwiner(vibration_rep, spaces[index], spaces[cluster[0]])
                if intertwiner is None:
                    raise RuntimeError("Coupled irrep spaces are not equivalent.")
                aligned.append(intertwiner.conj().T @ spaces[index])

            # After alignment every coupling block is a scalar multiple of the
            # identity (Schur), so the cluster reduces to one multiplicity-sized
            # Hermitian matrix shared by all irrep components.
            coupling = np.zeros((multiplicity, multiplicity), dtype=complex)
            for a in range(multiplicity):
                for b in range(multiplicity):
                    sub = aligned[a] @ modified_matrix @ aligned[b].conj().T
                    if np.abs(sub - np.eye(dim) * np.trace(sub) / dim).max() > 1e-6:
                        raise RuntimeError("Coupling between irrep spaces is not scalar.")
                    coupling[a, b] = np.trace(sub) / dim
            eigenvalues, eigenvectors = np.linalg.eigh(coupling)
            eigenvalues = eigenvalues.real

            # Preserve the symmetry-adapted basis when the cluster is
            # numerically degenerate. Re-diagonalizing an exactly degenerate
            # cluster can pick an arbitrary rotated basis and lower the
            # apparent symmetry of an individual mode.
            if multiplicity > 1 and np.allclose(eigenvalues, eigenvalues.mean(), atol=1e-10, rtol=1e-8):
                eigenvalues = np.full(multiplicity, eigenvalues.mean())
                eigenvectors = np.eye(multiplicity, dtype=complex)

            for w in range(multiplicity):
                eigval = float(eigenvalues[w])
                frequency = np.sign(eigval) * np.sqrt(np.abs(eigval)) * 15.633302
                for component in range(dim):
                    # Rows of the projected basis are bras; keep this module's
                    # convention that the displacement is Re(vector * e^{2 pi i q.R}),
                    # so the bra-space combination uses conjugated coefficients.
                    mode_vector = np.zeros(3 * self.n_atoms, dtype=complex)
                    for a in range(multiplicity):
                        mode_vector += np.conj(eigenvectors[a, w]) * aligned[a][component]
                    self.mode_info.append(
                        {
                            "frequency_THz": frequency,
                            "degeneracy": dim,
                        }
                    )
                    self.mode_vectors.append(mode_vector)

        sort_indices = np.argsort([float(info["frequency_THz"]) for info in self.mode_info], kind="stable")
        self.mode_info = [self.mode_info[index] for index in sort_indices]
        self.mode_vectors = [self.mode_vectors[index] for index in sort_indices]
        self._mode_labels: list[str] | None = None
        self._q_label: str | None = None

        # Verify against the plain phonopy spectrum before trusting the result.
        reference = np.sort(np.linalg.eigvalsh(modified_matrix).real)
        reference = np.sign(reference) * np.sqrt(np.abs(reference)) * 15.633302
        frequencies = [float(info["frequency_THz"]) for info in self.mode_info]
        if not np.allclose(frequencies, reference, atol=1e-3):
            raise RuntimeError("Symmetry-adapted frequencies do not match the phonopy spectrum.")
        for info, mode_vector in zip(self.mode_info, self.mode_vectors):
            eigenvalue = np.sign(info["frequency_THz"]) * (info["frequency_THz"] / 15.633302) ** 2
            ket = np.conj(mode_vector)
            if np.linalg.norm(modified_matrix @ ket - eigenvalue * ket) > 1e-6:
                raise RuntimeError("A symmetry-adapted mode is not an eigenvector of the dynamical matrix.")

    @property
    def n_modes(self) -> int:
        return len(self.mode_info)

    def get_mode_labels(self) -> list[str]:
        """Per-mode irrep labels (e.g. 'X3-(1)'); '-' when labeling is unavailable.

        Uses the ISO-IR-table-based labeling of crystod-phonon --vector/--irreps.
        The label of band i applies to mode i because the symmetry-adapted
        frequencies are verified to match the plain phonopy spectrum.
        """
        if self._mode_labels is None:
            labels = ["-"] * self.n_modes
            try:
                from .phonon_vector import _get_mode_labels
                from .runtime_compat import get_symmetry_dataset

                dataset = get_symmetry_dataset(self.phonon.symmetry)
                _, labels = _get_mode_labels(
                    [float(value) for value in self.qpoint],
                    self.phonon,
                    dataset,
                    degeneracy_tolerance=1e-3,
                )
            except Exception:
                pass
            self._mode_labels = labels
        return self._mode_labels

    def get_q_label(self) -> str:
        """Short q label for file names: the ISO-IR name (e.g. 'X') when q lies
        in the star of a tabulated special point, else 'q_<coordinates>'."""
        if self._q_label is None:
            label = "q" + "".join(f"_{value:g}" for value in self.qpoint).replace("/", "o")
            try:
                from phonopy.structure.cells import get_primitive_matrix_by_centring

                from .irreptables_compat import load_irreptables
                from .phonon_irreps import find_star_representative, get_irt_special_points
                from .runtime_compat import get_symmetry_dataset

                irrep_table_cls, _ = load_irreptables()
                dataset = get_symmetry_dataset(self.phonon.primitive_symmetry)
                irt_table = irrep_table_cls(dataset["number"], spinor=False)
                prim_mat = get_primitive_matrix_by_centring(dataset["international"][0])
                q_names, q_list = get_irt_special_points(irt_table, prim_mat)
                representative = find_star_representative(
                    self.qpoint, dataset["rotations"], q_names, q_list
                )
                if representative is not None:
                    label = representative[0]
                elif not self.keep_q_coords:
                    # non-special q: fall back to the ISO-IR k-vector type
                    from .isoir import get_isoir_kpoint_name

                    primitive = self.phonon.primitive
                    isoir_name = get_isoir_kpoint_name(
                        dataset["number"],
                        (primitive.cell, primitive.scaled_positions, primitive.numbers),
                        self.phonon.primitive_symmetry.tolerance,
                        self.qpoint,
                    )
                    if isoir_name is not None:
                        label = isoir_name
            except Exception:
                pass
            self._q_label = label
        return self._q_label

    def print_mode_info(self) -> None:
        labels = self.get_mode_labels()
        print(f"Phonon modes at q = {self.qpoint}")
        print(f"{'Mode':>5s}  {'Freq (THz)':>12s}  {'Irrep':>12s}  {'Degeneracy':>11s}")
        print("-" * 50)
        for mode_index, info in enumerate(self.mode_info):
            print(
                f"{mode_index + 1:5d}  {float(info['frequency_THz']):12.4f}  "
                f"{labels[mode_index]:>12s}  {int(info['degeneracy']):11d}"
            )

    @staticmethod
    def get_commensurate_supercell_sizes(qpoint: list[float] | NDArray[np.float64]) -> NDArray[np.int_]:
        sizes = []
        for component in qpoint:
            if abs(component) < 1e-10:
                sizes.append(1)
            else:
                sizes.append(Fraction(float(component)).limit_denominator(12).denominator)
        return np.array(sizes, dtype=int)

    def _get_commensurate_supercell_matrix(self) -> NDArray[np.int_]:
        return np.diag(self.get_commensurate_supercell_sizes(self.qpoint)).astype(int)

    def get_modulated_structure(
        self,
        mode_indices: list[int],
        amplitudes: list[float],
    ) -> Atoms:
        for mode_index in mode_indices:
            if mode_index < 0 or mode_index >= self.n_modes:
                raise SystemExit(
                    f"ERROR: mode number {mode_index + 1} is out of range "
                    f"[1, {self.n_modes}] (numbering is 1-based)."
                )

        supercell_matrix = self._get_commensurate_supercell_matrix()
        n1, n2, n3 = np.diag(supercell_matrix)

        primitive = self.vibrations.primitive_cell
        lattice = primitive.cell
        frac_pos = primitive.scaled_positions
        symbols_prim = get_chemical_symbols(primitive)

        positions = []
        displacements_total = []
        all_symbols = []

        for i1 in range(n1):
            for i2 in range(n2):
                for i3 in range(n3):
                    translation_frac = np.array([i1, i2, i3])
                    phase = np.exp(2j * np.pi * np.dot(self.qpoint, translation_frac))
                    for atom_index in range(self.n_atoms):
                        pos_frac = frac_pos[atom_index] + translation_frac
                        positions.append(pos_frac @ lattice)

                        displacement = np.zeros(3)
                        for mode_index, amplitude in zip(mode_indices, amplitudes):
                            mode_vector = self.mode_vectors[mode_index].reshape(self.n_atoms, 3)
                            displacement += np.real(mode_vector[atom_index] * phase) * amplitude

                        displacements_total.append(displacement)
                        all_symbols.append(symbols_prim[atom_index])

        positions = np.array(positions)
        displacements_total = np.array(displacements_total)
        modulated_positions = positions + displacements_total

        supercell_lattice = lattice.copy()
        supercell_lattice[0] *= n1
        supercell_lattice[1] *= n2
        supercell_lattice[2] *= n3

        unique_symbols = []
        for symbol in symbols_prim:
            if symbol not in unique_symbols:
                unique_symbols.append(symbol)

        sorted_indices = []
        for symbol in unique_symbols:
            for atom_index, atom_symbol in enumerate(all_symbols):
                if atom_symbol == symbol:
                    sorted_indices.append(atom_index)

        sorted_symbols = [all_symbols[index] for index in sorted_indices]
        sorted_positions = modulated_positions[sorted_indices]
        return Atoms(sorted_symbols, sorted_positions, cell=supercell_lattice, pbc=True)

    @staticmethod
    def analyze_symmetry(atoms: Atoms, symprec: float = 0.1) -> dict[str, str | int]:
        cell = (
            atoms.cell.array,
            atoms.get_scaled_positions(),
            atoms.numbers,
        )
        dataset = SymmetryDatasetAdapter(spglib.get_symmetry_dataset(cell, symprec=symprec))
        info = {
            "international": dataset.international,
            "number": dataset.number,
            "hall": dataset.hall,
        }
        print(f"Space group: {info['international']} (#{info['number']})")
        print(f"Hall symbol: {info['hall']}")
        return info


def _normalize_amplitudes(mode_indices: list[int], amplitudes: list[float]) -> list[float]:
    if len(amplitudes) == 1:
        return amplitudes * len(mode_indices)
    if len(amplitudes) != len(mode_indices):
        raise ValueError(
            f"Number of amplitudes ({len(amplitudes)}) must match number of modes ({len(mode_indices)}) "
            "or be a single value."
        )
    return amplitudes


def _parse_numbered_modulation_terms(extra_argv: list[str]) -> list[ModulationTerm]:
    if not extra_argv:
        return []

    grouped: dict[int, dict[str, list[str]]] = {}
    index = 0
    pattern = re.compile(r"^--(qpoint|mode|amplitude)(\d+)$")

    while index < len(extra_argv):
        token = extra_argv[index]
        match = pattern.fullmatch(token)
        if not match:
            raise ValueError(f"Unrecognized modulation argument: {token}")

        key, suffix = match.group(1), int(match.group(2))
        index += 1
        values: list[str] = []
        while index < len(extra_argv) and not extra_argv[index].startswith("--"):
            values.append(extra_argv[index])
            index += 1

        if not values:
            raise ValueError(f"{token} requires value(s).")
        grouped.setdefault(suffix, {})[key] = values

    terms: list[ModulationTerm] = []
    for suffix in sorted(grouped):
        entry = grouped[suffix]
        if "qpoint" not in entry:
            raise ValueError(f"--qpoint{suffix} is required when using numbered modulation arguments.")
        if "mode" not in entry:
            raise ValueError(f"--mode{suffix} is required when using numbered modulation arguments.")
        if len(entry["qpoint"]) != 3:
            raise ValueError(f"--qpoint{suffix} requires exactly three coordinates.")

        qpoint = [parse_qpoint_token(value) for value in entry["qpoint"]]
        mode_indices = [int(value) - 1 for value in entry["mode"]]
        raw_amplitudes = [float(value) for value in entry.get("amplitude", ["0.3"])]
        amplitudes = _normalize_amplitudes(mode_indices, raw_amplitudes)
        terms.append(
            ModulationTerm(
                qpoint=qpoint,
                mode_indices=mode_indices,
                amplitudes=amplitudes,
            )
        )

    return terms


def _build_combined_modulated_structure(terms: list[PreparedModulationTerm]) -> Atoms:
    if not terms:
        raise ValueError("At least one modulation term is required.")

    reference = terms[0].modulation.vibrations.primitive_cell
    lattice = reference.cell
    frac_pos = reference.scaled_positions
    symbols_prim = get_chemical_symbols(reference)
    n_atoms = len(frac_pos)

    for term in terms[1:]:
        primitive = term.modulation.vibrations.primitive_cell
        if not np.allclose(primitive.cell, lattice):
            raise ValueError("All modulation terms must share the same primitive lattice.")
        if not np.allclose(primitive.scaled_positions, frac_pos):
            raise ValueError("All modulation terms must share the same primitive positions.")
        if get_chemical_symbols(primitive) != symbols_prim:
            raise ValueError("All modulation terms must share the same primitive species ordering.")

    supercell_sizes = np.ones(3, dtype=int)
    for term in terms:
        supercell_sizes = np.lcm(
            supercell_sizes,
            SymmetryAdaptedModulation.get_commensurate_supercell_sizes(term.modulation.qpoint),
        )
    n1, n2, n3 = supercell_sizes

    prepared_terms: list[tuple[np.ndarray, list[np.ndarray], list[float]]] = []
    for term in terms:
        reshaped_vectors = [term.modulation.mode_vectors[index].reshape(n_atoms, 3) for index in term.mode_indices]
        prepared_terms.append((term.modulation.qpoint, reshaped_vectors, term.amplitudes))

    positions = []
    displacements_total = []
    all_symbols = []

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                translation_frac = np.array([i1, i2, i3], dtype=float)
                for atom_index in range(n_atoms):
                    pos_frac = frac_pos[atom_index] + translation_frac
                    positions.append(pos_frac @ lattice)

                    displacement = np.zeros(3)
                    for qpoint, mode_vectors, amplitudes in prepared_terms:
                        phase = np.exp(2j * np.pi * np.dot(qpoint, translation_frac))
                        for mode_vector, amplitude in zip(mode_vectors, amplitudes):
                            displacement += np.real(mode_vector[atom_index] * phase) * amplitude

                    displacements_total.append(displacement)
                    all_symbols.append(symbols_prim[atom_index])

    positions = np.array(positions)
    displacements_total = np.array(displacements_total)
    modulated_positions = positions + displacements_total

    supercell_lattice = lattice.copy()
    supercell_lattice[0] *= n1
    supercell_lattice[1] *= n2
    supercell_lattice[2] *= n3

    unique_symbols: list[str] = []
    for symbol in symbols_prim:
        if symbol not in unique_symbols:
            unique_symbols.append(symbol)

    sorted_indices = []
    for symbol in unique_symbols:
        for atom_index, atom_symbol in enumerate(all_symbols):
            if atom_symbol == symbol:
                sorted_indices.append(atom_index)

    sorted_symbols = [all_symbols[index] for index in sorted_indices]
    sorted_positions = modulated_positions[sorted_indices]
    return Atoms(sorted_symbols, sorted_positions, cell=supercell_lattice, pbc=True)


def _default_output_name(prepared_terms: list[PreparedModulationTerm], spacegroup: str) -> str:
    """Auto output name: MPOSCAR_{q}_{mode}_{irrep}_{subgroup}, one
    {q}_{mode}_{irrep} group per modulation term."""
    parts = ["MPOSCAR"]
    for term in prepared_terms:
        labels = term.modulation.get_mode_labels()
        unique_labels: list[str] = []
        for index in term.mode_indices:
            if labels[index] not in unique_labels:
                unique_labels.append(labels[index])
        parts.append(term.modulation.get_q_label())
        parts.append("mode" + "+".join(str(index + 1) for index in term.mode_indices))
        irrep_tag = _irrep_filename_tag(unique_labels)
        if irrep_tag:
            parts.append(irrep_tag)
    parts.append(spacegroup.replace("/", "").replace(" ", ""))
    return "_".join(parts)


# ---------------------------------------------------------------------------
# order-parameter directions -> modulated structures (crystod-phonon --subgroup
# --modulate). Which combination of the degenerate modes realizes a given
# direction depends on a basis convention that is not fixed anywhere, so the
# mapping is established the other way round: candidate combinations are
# generated, the space group of each generated structure is measured with
# spglib, and the measurement is matched against the enumerated table. Nothing
# is ever labeled by assumption -- a direction that no candidate reproduces is
# reported as not generated.
# ---------------------------------------------------------------------------

# distinct, deliberately non-commensurate amplitudes for the free parameters of
# a direction: equal ratios could realize a higher-symmetry direction by accident
_PARAMETER_VALUES = (1.0, 0.5478, 0.2971, 0.1607, 0.0871, 0.0472)


@dataclass(frozen=True)
class GeneratedDirection:
    """One order-parameter direction and the structure that realizes it."""

    label: str
    number: int
    symbol: str
    size: int
    index: int
    qpoints: tuple[tuple[float, ...], ...]
    modes: tuple[tuple[int, ...], ...]
    amplitudes: tuple[tuple[float, ...], ...]
    path: str | None = None


def _canonical_labelings(n_nonzero: int) -> list[tuple[int, ...]]:
    """Set partitions of n slots, as labels numbered by first occurrence.

    (1,1,1) -- all three slots share one free parameter -- comes before
    (1,1,2) and (1,2,3), so equal-amplitude (higher-symmetry) directions are
    tried first.
    """
    labelings: list[tuple[int, ...]] = []

    def walk(position: int, labels: list[int], used: int) -> None:
        if position == n_nonzero:
            labelings.append(tuple(labels))
            return
        for label in range(1, min(used + 1, len(_PARAMETER_VALUES)) + 1):
            walk(position + 1, labels + [label], max(used, label))

    walk(0, [], 0)
    return labelings


def _coefficient_patterns(n_slots: int, max_patterns: int = 2048) -> list[tuple[int, ...]]:
    """Every way of assigning n slots to "zero" or to a free parameter.

    Parameters are numbered by first occurrence, so this enumerates the set
    partitions of the slots with one distinguished zero block -- exactly the
    shapes an ISOTROPY order-parameter direction can take, and every placement
    of them. Patterns with the most zeros come first, so the high-symmetry
    directions are reached with the fewest trials; the cap keeps a large star
    (many arms x a degenerate level) from enumerating combinatorially.
    """
    from itertools import combinations

    patterns: list[tuple[int, ...]] = []
    for n_nonzero in range(1, n_slots + 1):
        labelings = _canonical_labelings(n_nonzero)
        for positions in combinations(range(n_slots), n_nonzero):
            for labels in labelings:
                assignment = [0] * n_slots
                for position, label in zip(positions, labels):
                    assignment[position] = label
                patterns.append(tuple(assignment))
                if len(patterns) >= max_patterns:
                    return patterns
    return patterns


def _primitive_signature(cell, symprec: float) -> tuple[int, int]:
    """(atoms in the spglib primitive cell, order of the point group)."""
    primitive = spglib.find_primitive(cell, symprec=symprec)
    if primitive is None:
        raise ValueError("spglib could not reduce the cell to a primitive one.")
    n_atoms = len(primitive[2])
    symmetry = spglib.get_symmetry(primitive, symprec=symprec)
    return n_atoms, len(symmetry["rotations"])


def classify_distorted_structure(
    atoms: Atoms, parent_cell, symprec: float = 1e-5
) -> tuple[int, str, int, int]:
    """Space group of a distorted structure as the isotropy table states it.

    Returns (number, symbol, size, index), where ``size`` is the primitive-cell
    multiplication against the parent and ``index`` is [G:H] -- the same two
    quantities ``crystod-group --supergroup`` prints, computed here from the
    structure itself so that a generated structure can be matched against an
    enumerated order-parameter direction.
    """
    child_cell = (atoms.cell[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    dataset = SymmetryDatasetAdapter(spglib.get_symmetry_dataset(child_cell, symprec=symprec))
    n_parent, ops_parent = _primitive_signature(parent_cell, symprec)
    n_child, ops_child = _primitive_signature(child_cell, symprec)
    if n_parent <= 0 or n_child % n_parent:
        raise ValueError("the distorted cell is not a supercell of the parent.")
    size = n_child // n_parent
    index = ops_parent * size // ops_child
    return int(dataset["number"]), str(dataset["international"]), size, index


def _direction_shape(label: str) -> tuple[int, tuple[int, ...]]:
    """Equality pattern of an order-parameter direction.

    ``R5-(0,a,b)`` -> (1 zero, blocks (1, 1)); ``R5-(a,a,b)`` -> (0 zeros,
    blocks (2, 1)). Two directions that condense into the same space group with
    the same cell size and index are told apart by this, which is what decides
    which of them a generated structure is labeled with.
    """
    match = re.match(r"^.*?\(([^)]*)\)\s*$", label.strip())
    if not match:
        return (0, ())
    zeros = 0
    blocks: dict[str, int] = {}
    anonymous = 0
    for token in match.group(1).replace(";", ",").split(","):
        token = token.strip().lstrip("-")
        if token in ("", "0", "0.0"):
            zeros += 1
            continue
        letters = "".join(ch for ch in token if ch.isalpha())
        if letters:
            blocks[letters] = blocks.get(letters, 0) + 1
        else:
            anonymous += 1  # a bare number: its own one-element block
    sizes = sorted(list(blocks.values()) + [1] * anonymous, reverse=True)
    return (zeros, tuple(sizes))


def _pattern_shape(pattern: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
    """The same signature for a candidate coefficient pattern."""
    zeros = sum(1 for value in pattern if not value)
    counts: dict[int, int] = {}
    for value in pattern:
        if value:
            counts[value] = counts.get(value, 0) + 1
    return (zeros, tuple(sorted(counts.values(), reverse=True)))


def _conventional_metric(cell, symprec: float) -> tuple[float, ...] | None:
    """Lengths and angles of the spglib conventional cell, rounded.

    Two structures that are domains of one subgroup share this; two different
    strata that happen to share (space group, size, index) do not, so it is
    what keeps a domain of an already-generated direction from being written
    out under a second direction's name.
    """
    standardized = spglib.standardize_cell(cell, symprec=symprec)
    if standardized is None:
        return None
    lattice = np.asarray(standardized[0], dtype=float)
    lengths = np.linalg.norm(lattice, axis=1)
    angles = [
        float(np.degrees(np.arccos(np.clip(
            np.dot(lattice[i], lattice[j]) / (lengths[i] * lengths[j]), -1.0, 1.0))))
        for i, j in ((1, 2), (0, 2), (0, 1))
    ]
    return tuple(np.round(np.concatenate([np.sort(lengths), np.sort(angles)]), 4))


def _direction_file_tag(label: str) -> str:
    """File-name form of a direction label: R5-(0,0,a) -> R5-_0-0-a."""
    match = re.match(r"^(.*?)\(([^)]*)\)\s*$", label.strip())
    if not match:
        return re.sub(r"[^\w+-]", "", label)
    irrep, direction = match.groups()
    direction = direction.replace(";", "_").replace(",", "-").replace(" ", "")
    return f"{irrep}_{direction}"


def generate_direction_structures(
    phonon,
    qpoints,
    mode_indices,
    targets,
    *,
    amplitude: float = 0.3,
    symprec: float = 1e-5,
    keep_q_coords: bool = False,
    prefix: str = "MPOSCAR",
    max_trials: int = 600,
    write: bool = True,
) -> tuple[list[GeneratedDirection], list[dict]]:
    """Realize each enumerated order-parameter direction as a structure.

    ``qpoints`` are the arms of the star of q (one modulation term each),
    ``mode_indices`` the 0-based modes of the degenerate level, and ``targets``
    the enumerated directions as dicts with ``label``/``number``/``symbol``/
    ``size``/``index``. Returns the directions that were realized and the ones
    that were not.
    """
    import contextlib
    import io

    arms = [list(map(float, q)) for q in qpoints]
    # one modulation per arm; their constructors narrate the cell conversion,
    # which would repeat once per arm in the middle of the subgroup report
    with contextlib.redirect_stdout(io.StringIO()):
        modulations = [
            SymmetryAdaptedModulation(
                phonon=phonon, qpoint=arm, symprec=symprec, keep_q_coords=keep_q_coords
            )
            for arm in arms
        ]
    parent = modulations[0].vibrations.primitive_cell
    parent_cell = (
        parent.cell,
        parent.scaled_positions,
        parent.numbers,
    )

    # Several enumerated directions can share (space group, cell size, index) --
    # R5+ of Pm-3m puts (0,a,b) and (a,a,b) both at C2/m, size 2, index 24. Keep
    # a queue per key instead of one entry, or the later rows would be dropped
    # without ever being generated or reported.
    wanted: dict[tuple[int, int, int], list[dict]] = {}
    for target in targets:
        key = (int(target["number"]), int(target["size"]), int(target["index"]))
        wanted.setdefault(key, []).append(
            {**target, "_shape": _direction_shape(str(target["label"]))}
        )
    accepted_metrics: dict[tuple[int, int, int], list[tuple[float, ...]]] = {}

    found: list[GeneratedDirection] = []
    n_modes = len(mode_indices)
    trials = 0
    for pattern in _coefficient_patterns(len(arms) * n_modes):
        if not wanted or trials >= max_trials:
            break
        coefficients = [_PARAMETER_VALUES[value - 1] if value else 0.0 for value in pattern]
        terms: list[PreparedModulationTerm] = []
        for arm_index, modulation in enumerate(modulations):
            chunk = coefficients[arm_index * n_modes : (arm_index + 1) * n_modes]
            selected = [
                (mode_indices[position], amplitude * coefficient)
                for position, coefficient in enumerate(chunk)
                if coefficient
            ]
            if selected:
                terms.append(
                    PreparedModulationTerm(
                        modulation=modulation,
                        mode_indices=[index for index, _ in selected],
                        amplitudes=[value for _, value in selected],
                    )
                )
        if not terms:
            continue
        trials += 1
        try:
            if len(terms) == 1:
                atoms = terms[0].modulation.get_modulated_structure(
                    mode_indices=terms[0].mode_indices, amplitudes=terms[0].amplitudes
                )
            else:
                atoms = _build_combined_modulated_structure(terms)
            key = classify_distorted_structure(atoms, parent_cell, symprec=symprec)
        except (ValueError, RuntimeError):
            continue
        key_triple = (key[0], key[2], key[3])
        queue = wanted.get(key_triple)
        if not queue:
            continue
        if len(queue) > 1 or accepted_metrics.get(key_triple):
            # this key holds (or held) more than one direction: make sure the
            # candidate is a genuinely different structure and not a domain of
            # one already written under a sibling direction's name
            try:
                metric = _conventional_metric(
                    (atoms.cell[:], atoms.get_scaled_positions(),
                     atoms.get_atomic_numbers()),
                    symprec,
                )
            except (ValueError, RuntimeError):
                metric = None
            if metric is not None:
                if metric in accepted_metrics.get(key_triple, []):
                    continue
                accepted_metrics.setdefault(key_triple, []).append(metric)
        # among the directions sharing this key, take the one whose pattern of
        # zeros and equal components matches the candidate's
        shape = _pattern_shape(pattern)
        position = next(
            (i for i, entry in enumerate(queue) if entry["_shape"] == shape), 0
        )
        target = queue.pop(position)
        if not queue:
            del wanted[key_triple]
        path = None
        if write:
            path = f"{prefix}_{_direction_file_tag(str(target['label']))}_" + str(
                target["symbol"]
            ).replace("/", "").replace(" ", "")
            ase_write(path, atoms, format="vasp", direct=True)
        found.append(
            GeneratedDirection(
                label=str(target["label"]),
                number=key[0],
                symbol=key[1],
                size=key[2],
                index=key[3],
                qpoints=tuple(tuple(term.modulation.qpoint.tolist()) for term in terms),
                modes=tuple(tuple(index + 1 for index in term.mode_indices) for term in terms),
                amplitudes=tuple(tuple(term.amplitudes) for term in terms),
                path=path,
            )
        )
    missing = [
        {name: value for name, value in entry.items() if name != "_shape"}
        for queue in wanted.values()
        for entry in queue
    ]
    return found, missing


def _load_modulation_with_report(
    phonon,
    source_label: str,
    qpoint: list[float],
    symprec: float,
    keep_q_coords: bool = False,
) -> SymmetryAdaptedModulation:
    """Build the modulation at q and print the mode table and the star of q."""
    print(f"Loading '{source_label}' at q = {qpoint}...")
    modulation = SymmetryAdaptedModulation(
        phonon=phonon,
        qpoint=qpoint,
        symprec=symprec,
        keep_q_coords=keep_q_coords,
    )
    print()
    modulation.print_mode_info()

    from .star_of_k import print_star_of_k

    print("\nStar of q (arms related by the space-group rotations):")
    print_star_of_k(
        rotations=modulation.vibrations.rotations,
        translations=modulation.vibrations.translations,
        kpoint=[float(value) for value in qpoint],
        indent="  ",
    )
    return modulation


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args, extra_argv = parser.parse_known_args(argv)

    # --yaml wins when given; a structure file selects the FORCE_SETS route;
    # with neither, the documented phonopy_params.yaml default applies (and the
    # structure route is the fallback when that file is absent).
    if args.yaml_path is not None:
        phonon, source_label, source_note = load_phonon(yaml_path=args.yaml_path)
    elif args.cell is not None:
        phonon, source_label, source_note = load_phonon(
            cell_path=args.cell, dim=args.dim, readfc=args.readfc
        )
    elif args.dim or args.readfc:
        # --dim/--readfc only mean anything on the structure route; honouring the
        # yaml here would silently ignore them
        phonon, source_label, source_note = load_phonon(
            cell_path="POSCAR", dim=args.dim, readfc=args.readfc
        )
    elif Path(DEFAULT_PARAMS_YAML).exists():
        phonon, source_label, source_note = load_phonon(yaml_path=DEFAULT_PARAMS_YAML)
    else:
        phonon, source_label, source_note = load_phonon(
            cell_path="POSCAR", dim=args.dim, readfc=args.readfc
        )
    if source_note:
        print(source_note)

    # one explicit --tolerance drives both the symmetry-adapted mode
    # construction and the space group reported for the generated structure
    symprec = 1e-5 if args.symprec is None else args.symprec
    display_symprec = 0.1 if args.symprec is None else args.symprec

    try:
        numbered_terms = _parse_numbered_modulation_terms(extra_argv)
    except ValueError as exc:
        parser.error(str(exc))

    if numbered_terms:
        if args.qpoint is not None or args.mode is not None or args.amplitude != [0.3]:
            parser.error(
                "Use either --qpoint/--mode/--amplitude or numbered sets such as "
                "--qpoint1/--mode1/--amplitude1, but not both."
            )
        terms = numbered_terms
    else:
        if args.qpoint is None:
            parser.error("--modulation requires --qpoint, or numbered arguments such as --qpoint1.")
        if args.mode is None:
            # Preview: show the mode table and the star of q so that a mode
            # can be chosen, without generating a modulated structure.
            _load_modulation_with_report(
                phonon, source_label, args.qpoint, symprec, args.keep_q_coords
            )
            print(
                "\nNo --mode given. Choose mode number(s) from the table above and rerun with"
                "\n--mode (and optionally --amplitude) to generate a modulated structure."
            )
            return
        mode_indices = [value - 1 for value in args.mode]
        terms = [
            ModulationTerm(
                qpoint=args.qpoint,
                mode_indices=mode_indices,
                amplitudes=_normalize_amplitudes(mode_indices, args.amplitude),
            )
        ]

    modulation_cache: dict[tuple[float, float, float], SymmetryAdaptedModulation] = {}
    prepared_terms: list[PreparedModulationTerm] = []

    for term_index, term in enumerate(terms, start=1):
        qpoint_key = tuple(float(value) for value in term.qpoint)
        modulation = modulation_cache.get(qpoint_key)
        if modulation is None:
            modulation = _load_modulation_with_report(
                phonon, source_label, term.qpoint, symprec, args.keep_q_coords
            )
            modulation_cache[qpoint_key] = modulation
            if term_index != len(terms):
                print()

        for mode_index in term.mode_indices:
            if mode_index < 0 or mode_index >= modulation.n_modes:
                raise SystemExit(
                    f"ERROR: mode number {mode_index + 1} is out of range "
                    f"[1, {modulation.n_modes}] (numbering is 1-based)."
                )

        prepared_terms.append(
            PreparedModulationTerm(
                modulation=modulation,
                mode_indices=term.mode_indices,
                amplitudes=term.amplitudes,
            )
        )

    print("\nGenerating modulated structure...")
    if len(prepared_terms) == 1:
        print(f"  q-point: {prepared_terms[0].modulation.qpoint.tolist()}")
        print(f"  Modes: {[index + 1 for index in prepared_terms[0].mode_indices]}")
        print(f"  Amplitudes (A): {prepared_terms[0].amplitudes}")
    else:
        for term_index, term in enumerate(prepared_terms, start=1):
            print(
                f"  Term {term_index}: q = {term.modulation.qpoint.tolist()}, "
                f"modes = {[index + 1 for index in term.mode_indices]}, "
                f"amplitudes (A) = {term.amplitudes}"
            )

    if len(prepared_terms) == 1:
        term = prepared_terms[0]
        atoms = term.modulation.get_modulated_structure(
            mode_indices=term.mode_indices,
            amplitudes=term.amplitudes,
        )
    else:
        atoms = _build_combined_modulated_structure(prepared_terms)

    print("\nSymmetry of the generated structure:")
    symmetry = SymmetryAdaptedModulation.analyze_symmetry(atoms, symprec=display_symprec)

    output_path = args.output or _default_output_name(prepared_terms, str(symmetry["international"]))
    ase_write(output_path, atoms, format="vasp", direct=True)
    print(f"\nModulated structure written to: {output_path}")


if __name__ == "__main__":
    main()
