"""
Symmetry-only vibration basis workflow for crystod.
"""

from __future__ import annotations

import re
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, RawDescriptionHelpFormatter, RawTextHelpFormatter
from fractions import Fraction
from pathlib import Path

import numpy as np
import spglib
from ase import Atoms
from ase.io import write as ase_write
from numpy.typing import NDArray

from phonopy.structure.cells import get_primitive_matrix_by_centring

from .irreptables_compat import load_irreptables
from .operations import conjugated_little_group_map, find_star_arm, parse_qpoint_token, snap_qpoint
from .runtime_compat import (
    SymmetryDatasetAdapter,
    get_character,
    get_chemical_symbols,
    get_little_group,
    get_scaled_positions,
)
from .spglib_compat import ensure_spglib_compat

ensure_spglib_compat()

from phonopy.interface.calculator import read_crystal_structure
from phonopy.structure.atoms import PhonopyAtoms
from .runtime_compat import get_spacegroup_irreps_from_primitive_symmetry
from spgrep.representation import project_to_irrep

IrrepTable, Irrep = load_irreptables()


def _snap_if_rational(vector) -> list[float]:
    """``snap_qpoint`` for the coordinates within 1e-6 of a simple fraction,
    the value itself for the others: a seekpath point whose coordinates
    depend on the lattice parameters is not shifted along its line."""
    return [parse_qpoint_token(float(value)) for value in vector]


class MyHelpFormatter(
    RawTextHelpFormatter,
    RawDescriptionHelpFormatter,
    ArgumentDefaultsHelpFormatter,
):
    pass


desc = """
Construct symmetry-allowed vibration basis vectors without phonon force data.

# Command Examples:
crystod-phonon --vibration -c example/test_POSCARs/221_PPOSCAR_ScF3 --qpoint 0.5 0.5 0.5
crystod-phonon --vibration -c example/test_POSCARs/221_PPOSCAR_ScF3 --qpoint R --mode-index 3 --component-index 1 --output POSCAR_vibration
"""


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=desc, formatter_class=MyHelpFormatter)
    parser.add_argument(
        "--poscar",
        default="POSCAR",
        help="POSCAR path.",
    )
    parser.add_argument(
        "--qpoint",
        nargs="+",
        default=None,
        help="Either an ISO-IR special-point name such as GM/X/M/R or three primitive\n"
        "reciprocal coordinates.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Symmetry tolerance.",
    )
    parser.add_argument(
        "--list-qpoints",
        action="store_true",
        help="Only list the ISO-IR special q points (primitive coordinates) and exit.",
    )
    parser.add_argument(
        "--raman-tensor",
        dest="raman_tensor",
        action="store_true",
        help="At Gamma, print the symmetry-allowed Raman tensors of the Raman-active\n"
        "irreps (Cartesian axes of the input cell).",
    )
    parser.add_argument(
        "--mode-index",
        type=int,
        default=None,
        help="Irrep-grouped mode-space number to inspect (1-based).",
    )
    parser.add_argument(
        "--component-index",
        type=int,
        default=1,
        help="Component number inside the selected degenerate mode space (1-based).",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.3,
        help="Amplitude used when writing a displaced structure.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output POSCAR path for the selected mode/component.",
    )
    parser.add_argument(
        "--export-npz",
        default=None,
        help="Optional .npz path to save positions, displacements, symbols, and lattice.",
    )
    return parser


class _CoreRepresentation:
    def __init__(
        self,
        cell: PhonopyAtoms,
        symprec: float = 1e-5,
        standardize: bool = True,
        input_cell: PhonopyAtoms | None = None,
    ):
        # the cell as given: the ISO-IR frame of the irrep labels is taken
        # from it, not from the standardized cell derived below
        self.input_cell = cell if input_cell is None else input_cell
        if standardize:
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
        else:
            # Keep the input cell as-is (it must already be primitive). This
            # preserves the caller's atom positions so that phase conventions
            # stay consistent with an externally built dynamical matrix.
            self.primitive_cell = cell

        dataset = SymmetryDatasetAdapter(
            spglib.get_symmetry_dataset(self.primitive_cell.totuple(), symprec=symprec)
        )
        self.spglib_dataset = dataset
        self.rotations = dataset.rotations
        self.translations = dataset.translations
        self.symprec = symprec

    def _get_isoir_label_list(
        self,
        qpoint: list[float],
        irreps,
        mapping_little_group,
    ) -> list[str] | None:
        """ISO-IR (Miller-Love) labels for spgrep irreps at a non-tabulated q,
        or None when the ISO-IR data are unavailable or matching fails.
        """
        from .isoir import get_isoir_label_map

        self.labels_from_isoir = False
        matched = get_isoir_label_map(
            self.spglib_dataset["number"],
            self.primitive_cell.totuple(),
            self.symprec,
            qpoint,
            [self.rotations[index] for index in mapping_little_group],
            [self.translations[index] for index in mapping_little_group],
            [get_character(irrep) for irrep in irreps],
            input_cell=self.input_cell.totuple(),
        )
        if matched is None:
            return None
        label_map, qpoint_name = matched
        self.labels_from_isoir = True
        self.isoir_qpoint_name = qpoint_name
        return [
            f"{label_map[index]}({irrep.shape[1]})"
            for index, irrep in enumerate(irreps)
        ]

    def _get_isoir_labeler(self):
        """The ISO-IR labeller of the labels (that of
        :meth:`_get_isoir_label_list`), or None."""
        from .isoir import get_cached_labeler

        return get_cached_labeler(
            self.spglib_dataset["number"], self.primitive_cell.totuple(),
            self.symprec, input_cell=self.input_cell.totuple(),
        )

    def _special_points_in_label_frame(self, names, points):
        """A list of tabulated special points named in the frame of the
        labels (``crystod.isoir.special_points_in_frame``): the list a
        survey loops over and shows."""
        from .isoir import special_points_in_frame

        return special_points_in_frame(
            names, points, self._get_isoir_labeler(), canonical=snap_qpoint
        )

    def _spglib_frame_labeler(self):
        """An ISO-IR labeller in spglib's own frame of the primitive cell,
        the frame seekpath and the special-point tables name their points
        in, or None."""
        if not hasattr(self, "_spglib_labeler"):
            from .isoir import IsoIRLabeler

            try:
                self._spglib_labeler = IsoIRLabeler(
                    self.spglib_dataset["number"], cell=self.primitive_cell.totuple(),
                    symprec=self.symprec,
                )
            except Exception:
                self._spglib_labeler = None
        return self._spglib_labeler

    def _renamed_by_frame(self, qpoint) -> str | None:
        """The ISO-IR name of q in the frame of the labels when spglib's
        frame names q differently (an input outside the ISO-IR setting,
        where the point seekpath calls P can be the PA of the labels), else
        None."""
        labeler, spglib_frame = self._get_isoir_labeler(), self._spglib_frame_labeler()
        if labeler is None or spglib_frame is None:
            return None
        try:
            name = labeler.kpoint_name(qpoint)
            if name is not None and name != spglib_frame.kpoint_name(qpoint):
                return name
        except Exception:
            pass
        return None

    def get_modified_permutation_rep(
        self,
        rotation: NDArray[np.int_],
        translation: NDArray[np.float64],
        kpoint: list[float],
    ) -> NDArray[np.complex128]:
        """Atom-permutation matrix of one operation at q, with Bloch phases."""
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
        """Permutation matrices of the little-group operations at q."""
        return np.array(
            [
                self.get_modified_permutation_rep(rotation, translation, kpoint)
                for rotation, translation in zip(little_rotations, little_translations)
            ],
            dtype=np.complex128,
        )

    def get_little_group(self, kpoint: list[float]):
        """Little-group operations of q (see ``runtime_compat.get_little_group``)."""
        return get_little_group(
            rotations=self.rotations,
            translations=self.translations,
            kpoint=kpoint,
        )


class SymmetryOnlyVibrations(_CoreRepresentation):
    """Symmetry-allowed vibration bases of a crystal, without force data.

    The engine of ``crystod-phonon --vibration``: at a q point, the
    displacement representation of the little group of q (the permutation
    representation of the atoms times the Cartesian rotation, with Bloch
    phases) is projected onto the spgrep irreps, giving one basis of
    symmetry-adapted displacement patterns per irrep occurrence. The spaces
    are labeled with ISO-IR irrep names, and any component can be written out
    as a displaced structure on the commensurate supercell: the partner of
    :meth:`get_symmetry_adapted_spaces` -- fixed by the conventions of
    ``--modulation``, real with the Bloch phase of every atom at a
    time-reversal-invariant q, so that it freezes into an isotropy subgroup
    of its irrep -- a unit-norm symmetry-adapted displacement pattern, not a
    normal mode. :func:`crystod.symmetry_adapted_modes.solve_symmetry_adapted_modes`
    (behind ``crystod-phonon --modulation`` and ``--vector``) uses the same
    basis, on phonopy's primitive cell as it is (``standardize=False``), to
    block-diagonalize a dynamical matrix.

    Args:
        cell: The crystal structure as a ``phonopy.structure.atoms.PhonopyAtoms``
            object, e.g. from ``phonopy.interface.calculator.read_crystal_structure``.
        symprec: Symmetry tolerance of the spglib analysis.
        standardize: Reduce ``cell`` to the spglib primitive cell first (the
            default; a note is printed). ``False`` keeps the input cell as
            it is, which must then already be primitive; this preserves the
            caller's atom positions so that phase conventions stay consistent
            with an externally built dynamical matrix. Either way the
            analysis, the Bloch phases and the written supercells all use
            ``primitive_cell``.
        input_cell: The cell whose ISO-IR frame names the irreps, when it is
            not ``cell`` itself (e.g. the unit cell a phonopy primitive cell
            was built from); default ``cell``.

    Attributes:
        primitive_cell: The primitive cell the analysis runs on.
        spglib_dataset: Its spglib symmetry dataset (``["number"]``,
            ``["international"]``, ...).
        rotations: Rotation parts of the space-group operations in the
            primitive basis, shape ``(n_ops, 3, 3)``.
        translations: The corresponding translation parts, shape
            ``(n_ops, 3)``.
        rotations_cartesian: The rotations in Cartesian coordinates.
        symprec: The symmetry tolerance.
        labels_from_isoir: ``True`` when the last :meth:`get_irrep_labels`
            call took its labels from the general ISO-IR k-vector lookup
            rather than from the special-point table.

    Example:
        List the vibration spaces of cubic ScF3 at the R point::

            from phonopy.interface.calculator import read_crystal_structure
            from crystod import phonon
            from crystod.examples import example_path

            cell, _ = read_crystal_structure(
                example_path("221_PPOSCAR_ScF3"), interface_mode="vasp")
            vibrations = phonon.SymmetryOnlyVibrations(cell)
            label, qpoint = vibrations.resolve_qpoint(["R"])
            irreps, spaces, labels = vibrations.describe_mode_spaces(qpoint)
            labels                      # ['R1+(1)', 'R3+(2)', 'R4+(3)', ...]
            [space.shape for space in spaces]   # [(1, 12), (2, 12), (3, 12), ...]
    """

    def __init__(
        self,
        cell: PhonopyAtoms,
        symprec: float = 1e-5,
        standardize: bool = True,
        input_cell: PhonopyAtoms | None = None,
    ):
        super().__init__(
            cell=cell, symprec=symprec, standardize=standardize, input_cell=input_cell
        )
        lattice_t = np.transpose(self.primitive_cell.cell)
        lattice_t_inv = np.linalg.inv(lattice_t)
        self.rotations_cartesian = np.array(
            [lattice_t @ rotation @ lattice_t_inv for rotation in self.rotations],
            dtype=np.complex128,
        )

    def get_high_symmetry_qpoints(self) -> dict[str, list[float]]:
        """High-symmetry q points of the primitive cell, from seekpath.

        ``crystod-phonon --vibration --list-qpoints`` prints this map. The
        coordinates are expressed in the reciprocal basis of *this* object's
        (spglib) primitive cell: seekpath's own primitive cell can differ
        from it by an integer change of basis (base-centred monoclinic cells,
        for instance), and the tabulated coordinates are transformed
        accordingly. A warning is issued only when the two cells are not
        related by such a change of basis, in which case the coordinates are
        returned as seekpath gives them.

        seekpath names its points in spglib's standardized frame, while the
        irrep labels refer to the ISO-IR frame of the input cell; for an
        input outside the ISO-IR setting the two differ by an element of the
        normalizer, which can turn seekpath's P into the PA of the labels.
        A point whose ISO-IR type differs between the two frames is
        therefore moved to the point that carries its type in the frame of
        the labels (-k for the -k partner of a tabulated type, else its
        image under the change of frame), so that ``--qpoint P`` is the
        point the labels call P. The names stay seekpath's: for a triclinic
        cell seekpath's letters X, Y, Z, R, T, U, V denote other points than
        the ISO-IR names of the labels (seekpath's Z of
        ``1_PPOSCAR_RbBe2F5`` carries ``X1``).

        Returns:
            Dict mapping seekpath labels (``"GAMMA"``, ``"R"``, ``"X"``, ...)
            to fractional coordinates in the primitive reciprocal basis.
        """
        import seekpath
        import warnings

        structure = (
            self.primitive_cell.cell,
            self.primitive_cell.scaled_positions,
            self.primitive_cell.numbers,
        )
        path_data = seekpath.get_path(structure, symprec=1e-5)
        point_coords = path_data["point_coords"]
        own_lattice = np.asarray(self.primitive_cell.cell, dtype=float)
        seekpath_lattice = np.asarray(path_data["primitive_lattice"], dtype=float)
        if not np.allclose(own_lattice, seekpath_lattice, atol=1e-4):
            # rows of the seekpath cell as integer combinations of our rows:
            # P_seek = M P_own; a k point with fractional coordinates k_seek
            # in seekpath's reciprocal basis is k_own = k_seek inv(M)^T here
            change = seekpath_lattice @ np.linalg.inv(own_lattice)
            rounded = np.rint(change)
            if (np.allclose(change, rounded, atol=1e-6)
                    and abs(round(np.linalg.det(rounded))) == 1):
                to_own = np.linalg.inv(rounded).T
                point_coords = {
                    label: (np.asarray(coords, dtype=float) @ to_own).tolist()
                    for label, coords in point_coords.items()
                }
            else:
                warnings.warn(
                    "The primitive cell from seekpath does not match the spglib "
                    "primitive cell. The q-point coordinates might need a basis "
                    "transformation.",
                    stacklevel=2,
                )
        # the labels whose point carries seekpath's ISO-IR type in the frame
        # of the labels (all of them unless a move below fails): the ones
        # resolve_qpoint may name coordinates by
        self._labels_in_frame = set(point_coords)
        labeler, spglib_frame = self._get_isoir_labeler(), self._spglib_frame_labeler()
        if labeler is None or spglib_frame is None:
            return point_coords
        # k_label = k_spglib Pinv_spglib P_label carries a point onto the one
        # with the same ISO-IR conventional coordinates in the label frame
        change = spglib_frame.Pinv @ labeler.P
        in_frame = {}
        for label, coords in point_coords.items():
            in_frame[label] = coords
            try:
                wanted = spglib_frame.kpoint_name(coords)
                if wanted is None or labeler.kpoint_name(coords) == wanted:
                    continue
                self._labels_in_frame.discard(label)
                vector = np.asarray(coords, dtype=float)
                for candidate in (_snap_if_rational(-vector), _snap_if_rational(vector @ change)):
                    if labeler.kpoint_name(candidate) == wanted:
                        in_frame[label] = candidate
                        self._labels_in_frame.add(label)
                        break
            except Exception:
                continue
        return in_frame

    def resolve_qpoint(self, raw_qpoint: list[str]) -> tuple[str, list[float]]:
        """Resolve ``--qpoint`` tokens into a label and coordinates.

        One token is a seekpath label (``GM``, ``G`` and the Greek capital
        gamma are accepted for ``GAMMA``) of
        :meth:`get_high_symmetry_qpoints`; three tokens are coordinates in
        the primitive reciprocal basis, fractions such as ``1/3`` allowed.
        Coordinates of a point of :meth:`get_high_symmetry_qpoints` are
        labeled with its name there. Other coordinates get, where the frame
        of the labels names q differently from spglib's frame (an input
        outside the ISO-IR setting), the ISO-IR name in the frame of the
        labels (``PA`` above PA labels); else the name of the star arm the
        space-group rotations map them onto, else the ISO-IR k-vector type
        of q, else ``"custom"``.

        Args:
            raw_qpoint: The tokens, one label or three coordinate strings.

        Returns:
            ``(label, qpoint)`` with ``qpoint`` a list of three floats.

        Raises:
            ValueError: For an unknown label, or a token count other than one
                or three.
        """
        qpoint_map = self.get_high_symmetry_qpoints()
        alias_map = {
            "GM": "GAMMA",
            "G": "GAMMA",
            "Γ": "GAMMA",
        }

        if len(raw_qpoint) == 1:
            requested = raw_qpoint[0].strip().upper()
            # the aliases stand in for GAMMA only where the cell has no
            # special point of that name (body-centred tetragonal cells have
            # a genuine G)
            if requested not in qpoint_map:
                requested = alias_map.get(requested, requested)
            if requested in qpoint_map:
                return requested, list(qpoint_map[requested])
            available = ", ".join(sorted(qpoint_map))
            raise ValueError(
                f"Unknown q-point label '{raw_qpoint[0]}'. Available labels: {available}"
            )

        if len(raw_qpoint) != 3:
            raise ValueError("--qpoint must be either one label or three coordinates.")

        qpoint = [parse_qpoint_token(value) for value in raw_qpoint]
        # a listed point is named as the list names it: its point carries
        # the ISO-IR type of seekpath's name in the frame of the labels
        in_frame = getattr(self, "_labels_in_frame", set(qpoint_map))
        matched_label = None
        for label, coords in qpoint_map.items():
            if label in in_frame and np.allclose(qpoint, coords, atol=1e-8):
                matched_label = label
                break
        # else the frame of the labels first where spglib's frame, in which
        # seekpath names its points, would give q another name
        if matched_label is None:
            matched_label = self._renamed_by_frame(qpoint)
        if matched_label is None:
            for label, coords in qpoint_map.items():
                if np.allclose(qpoint, coords, atol=1e-8):
                    matched_label = label
                    break
        if matched_label is None:
            # q may be a non-tabulated arm of a special-point star: label it
            # with the name of the arm the space-group rotations map it onto.
            arm = find_star_arm(qpoint, self.rotations, list(qpoint_map.values()))
            if arm is not None:
                for label, coords in qpoint_map.items():
                    if np.allclose(arm[1], coords, atol=1e-8):
                        matched_label = label
                        break
        if matched_label is None:
            # q outside the special-point list (a line, plane or generic q,
            # or a -k star such as PA): the ISO-IR k-vector type label
            from .isoir import get_isoir_kpoint_name

            matched_label = get_isoir_kpoint_name(
                self.spglib_dataset["number"], self.primitive_cell.totuple(),
                self.symprec, qpoint, input_cell=self.input_cell.totuple(),
            )
        return matched_label or "custom", qpoint

    def get_special_qpoints(self) -> dict[str, list[float]]:
        """The ISO-IR special q points of the space group, in the frame of the labels.

        ``crystod-phonon --vibration`` lists these points (``--list-qpoints``)
        and resolves ``--qpoint NAME`` through them: one entry per special
        point of the ISO-IR tables, named as the irrep labels at it are named
        (the frame of the labels, ``crystod.isoir.special_points_in_frame``;
        a point tabulated only through its -k partner is listed at the
        tabulated point), the set of points ``crystod-phonon --irreps``
        writes under ``special_points:`` for the same structure. The -k
        partners (``PA``) and seekpath's extra points (``H_2``) are not
        listed. The command prints them in the order of seekpath's path
        (``GM, A, K, H, M, L`` for P6_3mc); this method keeps the table order.

        Returns:
            Dict mapping ISO-IR names (``"GM"``, ``"A"``, ``"K"``, ...), in the
            order of the tables, to fractional coordinates in the reciprocal
            basis of :attr:`primitive_cell`; empty when the tables do not
            cover the space group.
        """
        cached = getattr(self, "_special_qpoints", None)
        if cached is None:
            from .phonon_irreps import get_irt_special_points

            cached = {}
            try:
                table = IrrepTable(self.spglib_dataset["number"], spinor=False)
                primitive_matrix = get_primitive_matrix_by_centring(
                    self.spglib_dataset["international"][0]
                )
                names, points = self._special_points_in_label_frame(
                    *get_irt_special_points(table, primitive_matrix)
                )
                cached = {
                    name: [float(value) + 0.0 for value in point]
                    for name, point in zip(names, points)
                }
            except Exception:
                cached = {}
            self._special_qpoints = cached
        return {name: list(point) for name, point in cached.items()}

    def name_qpoint(self, qpoint) -> str:
        """The name ``crystod-phonon --vibration`` gives to q point coordinates.

        The ISO-IR name of a point of :meth:`get_special_qpoints`; for another
        arm of its star (or ``q + G``) the name of the arm the space-group
        rotations map q onto; else the ISO-IR k-vector type of q in the frame
        of the labels (``T`` for a symmetry line, ``PA`` for the -k partner
        of a tabulated point, ``GP`` for a general point); else ``"custom"``.

        Args:
            qpoint: Fractional coordinates in the primitive reciprocal basis.

        Returns:
            The name.
        """
        points = self.get_special_qpoints()
        for name, coords in points.items():
            if np.allclose(qpoint, coords, atol=1e-8):
                return name
        arm = find_star_arm(qpoint, self.rotations, list(points.values()))
        if arm is not None:
            for name, coords in points.items():
                if np.allclose(arm[1], coords, atol=1e-8):
                    return name
        from .isoir import get_isoir_kpoint_name

        try:
            name = get_isoir_kpoint_name(
                self.spglib_dataset["number"], self.primitive_cell.totuple(),
                self.symprec, list(qpoint), input_cell=self.input_cell.totuple(),
            )
        except Exception:
            name = None
        return name or "custom"

    def resolve_special_qpoint(self, raw_qpoint: list[str]) -> tuple[str, list[float]]:
        """Resolve ``--qpoint`` tokens of ``crystod-phonon --vibration``.

        One token is an ISO-IR name of :meth:`get_special_qpoints` (any
        case); ``GAMMA``, ``G`` and the Greek capital gamma stand for ``GM``,
        and a seekpath name of :meth:`get_high_symmetry_qpoints` that is not
        an ISO-IR name of the list (``H_2``) is accepted as an alias of its
        point. Three tokens are coordinates in the primitive reciprocal basis
        (fractions such as ``1/3`` allowed), named by :meth:`name_qpoint`.
        :meth:`resolve_qpoint` keeps the seekpath names for the other
        commands.

        Args:
            raw_qpoint: The tokens, one name or three coordinate strings.

        Returns:
            ``(name, qpoint)`` with ``qpoint`` a list of three floats.

        Raises:
            ValueError: For an unknown name, or a token count other than one
                or three.
        """
        if len(raw_qpoint) == 1:
            token = raw_qpoint[0].strip()
            requested = token.upper()
            points = self.get_special_qpoints()
            if requested in points:
                return requested, list(points[requested])
            if requested in ("GAMMA", "G", "Γ") and "GM" in points:
                return "GM", list(points["GM"])
            try:
                seekpath_points = self.get_high_symmetry_qpoints()
            except Exception:
                seekpath_points = {}
            if requested in seekpath_points:
                qpoint = [float(value) for value in seekpath_points[requested]]
                return self.name_qpoint(qpoint), qpoint
            available = ", ".join(points) or ", ".join(seekpath_points)
            raise ValueError(
                f"Unknown q-point label '{token}'. Available labels: {available}"
            )
        if len(raw_qpoint) != 3:
            raise ValueError("--qpoint must be either one label or three coordinates.")
        qpoint = [parse_qpoint_token(value) for value in raw_qpoint]
        return self.name_qpoint(qpoint), qpoint

    def get_vibration_rep(self, kpoint: list[float]):
        """Displacement representation of the little group of q.

        Args:
            kpoint: Fractional coordinates of q in the primitive reciprocal
                basis.

        Returns:
            ``(irreps, vibration_rep, mapping_little_group)``: the spgrep
            irreps of the little group of q; the representation matrices of
            the little-group operations on the ``3 * n_atoms`` displacement
            space, shape ``(n_little, 3 * n_atoms, 3 * n_atoms)``; and the
            indices of the little-group operations within ``rotations``.
        """
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
        return irreps, vibration_rep, mapping_little_group

    def get_vibration_basis(
        self,
        irreps,
        vibration_rep,
        irrep_labels: list[str] | None = None,
    ) -> tuple[list[NDArray[np.complex128]], list[str]]:
        """Project the displacement representation onto each irrep.

        Args:
            irreps: The spgrep irreps from :meth:`get_vibration_rep`.
            vibration_rep: The representation matrices from
                :meth:`get_vibration_rep`.
            irrep_labels: One label per irrep, e.g. from
                :meth:`get_irrep_labels`; generic ``irrep_N(dim)`` labels are
                used when omitted.

        Returns:
            ``(basis_vectors, basis_labels)``: one ``(dim, 3 * n_atoms)`` array
            per occurrence of an irrep in the displacement representation, and
            the label of each space. The rows are the symmetry-adapted
            displacement patterns as kets: for one array ``B`` and the irrep
            matrices ``d``, ``vibration_rep[g] @ B.T == B.T @ d[g]``. They are
            in the atom-position phase convention of phonopy's eigenvectors
            (Bloch factor ``exp(2 pi i q . x_j)`` of each atom not included).
            Repeated occurrences of one irrep transform with the same
            matrices but are not orthogonal to each other.
        """
        basis_vectors: list[NDArray[np.complex128]] = []
        basis_labels: list[str] = []
        fallback_labels = irrep_labels or [f"irrep_{index + 1}({irrep.shape[1]})" for index, irrep in enumerate(irreps)]
        for irrep, irrep_label in zip(irreps, fallback_labels):
            projected_spaces = project_to_irrep(vibration_rep, irrep)
            basis_vectors.extend(projected_spaces)
            basis_labels.extend([irrep_label] * len(projected_spaces))
        return basis_vectors, basis_labels

    def _get_irt_irreps_at_q(self, qpoint: list[float], irt_table, prim_mat) -> list[Irrep]:
        irreps_at_q = []
        prim_inv = np.linalg.inv(prim_mat)
        conventional_q = np.array(qpoint) @ prim_inv
        for irrep_at_q in irt_table.irreps:
            if np.allclose(irrep_at_q.k, conventional_q):
                irreps_at_q.append(irrep_at_q)
        return irreps_at_q

    def _get_mapping_to_irt(
        self,
        irt_little_rotations: NDArray[np.int_],
        found_little_rotations: NDArray[np.int_],
        prim_mat: NDArray[np.float64],
    ) -> list[int]:
        conventional_little_rotations = prim_mat @ found_little_rotations @ np.linalg.inv(prim_mat)
        mapping_to_irt = []
        for irt_rotation in irt_little_rotations:
            for index, rotation in enumerate(conventional_little_rotations):
                if np.allclose(irt_rotation, rotation):
                    mapping_to_irt.append(index)
                    break
        return mapping_to_irt

    def get_irrep_labels(
        self,
        qpoint: list[float],
        irreps,
        mapping_little_group: NDArray[np.int_],
    ) -> list[str]:
        """ISO-IR labels of the spgrep irreps at q.

        The ISO-IR labeller (``crystod.isoir``) names the spgrep irreps at
        any q -- a special point, any arm of its star or copy q + G, a
        symmetry line, plane or generic q (Miller-Love labels) -- in the
        ISO-IR frame of the input cell; a q point tabulated only through its
        -k partner gets the 'A' names of the conjugate irreps (``PA1``).
        Only where the labeller gives nothing are the characters compared
        directly with the special-point table (at a tabulated point, or by
        conjugation onto the tabulated arm for another arm of its star). An
        irrep neither route names keeps its generic ``irrep_N(dim)`` label.

        Args:
            qpoint: Fractional coordinates of q in the primitive reciprocal
                basis.
            irreps: The spgrep irreps from :meth:`get_vibration_rep`.
            mapping_little_group: The little-group indices from
                :meth:`get_vibration_rep`.

        Returns:
            One label per irrep, e.g. ``"R4+(3)"`` (the irrep name with its
            dimension).
        """
        generic_labels = [f"irrep_{index + 1}({irrep.shape[1]})" for index, irrep in enumerate(irreps)]
        # The ISO-IR labeller is the label authority at every q: it compares
        # in the ISO-IR setting, at the exact translations and with the
        # conjugate phase convention.  The direct special-point comparison
        # further down ignores all three and names a physically different
        # irrep at some points (H/K of the hexagonal groups, P of I4/mcm,
        # Y/T of Ccce, ...); it is kept only as a fallback.
        isoir_labels = self._get_isoir_label_list(qpoint, irreps, mapping_little_group)
        try:
            irt_table = IrrepTable(self.spglib_dataset["number"], spinor=False)
        except Exception:
            return isoir_labels or generic_labels

        prim_mat = get_primitive_matrix_by_centring(self.spglib_dataset["international"][0])
        irt_irreps = self._get_irt_irreps_at_q(qpoint, irt_table, prim_mat)
        conjugated = None
        if not irt_irreps:
            # q may be a non-tabulated arm of a special-point star: map it onto
            # the tabulated arm and transport the characters by conjugation.
            special_points: list[list[float]] = []
            for irrep_at_q in irt_table.irreps:
                primitive_q = snap_qpoint(np.array(irrep_at_q.k) @ prim_mat)
                if primitive_q not in special_points:
                    special_points.append(primitive_q)
            arm = find_star_arm(qpoint, self.rotations, special_points)
            if arm is not None:
                candidate_irreps = self._get_irt_irreps_at_q(arm[1], irt_table, prim_mat)
                transported = conjugated_little_group_map(
                    self.rotations, self.translations, arm[0], arm[1], mapping_little_group
                )
                if candidate_irreps and transported is not None:
                    irt_irreps = candidate_irreps
                    conjugated = transported
        if not irt_irreps:
            # Not tabulated as a special point (e.g. a symmetry line/plane or
            # generic q): only the general ISO-IR (ISOTROPY) k-vector lookup
            # covers it.  Labels then follow the Miller-Love convention.
            return isoir_labels or generic_labels
        if isoir_labels is not None:
            # a special point (or another arm of its star): same labeller,
            # but the flag keeps its meaning "q is not a tabulated point"
            self.labels_from_isoir = False
            return isoir_labels

        irt_little_rotations = np.array(
            [irt_table.symmetries[index - 1].R for index in irt_irreps[0].characters.keys()]
        )
        found_little_rotations = self.rotations[mapping_little_group]
        if conjugated is None:
            mapping_to_irt = self._get_mapping_to_irt(irt_little_rotations, found_little_rotations, prim_mat)
            character_phases = np.ones(len(irt_little_rotations), dtype=complex)
        else:
            # mapping_to_irt[m] = position (within the little group of q) of the
            # operation h whose conjugate g^-1 h g is the m-th tabulated
            # operation; the transported character picks up the Bloch phase.
            conj_indices, conj_phases = conjugated
            prim_mat_inv = np.linalg.inv(prim_mat)
            mapping_to_irt = []
            phases: list[complex] = []
            for irt_rotation in irt_little_rotations:
                rotation_prim = np.rint(prim_mat_inv @ irt_rotation @ prim_mat).astype(int)
                table_op_index = None
                for j, rotation in enumerate(self.rotations):
                    if (rotation == rotation_prim).all():
                        table_op_index = j
                        break
                if table_op_index is None or table_op_index not in conj_indices:
                    break
                position = conj_indices.index(table_op_index)
                mapping_to_irt.append(position)
                phases.append(conj_phases[position])
            character_phases = np.array(phases, dtype=complex)
        if len(mapping_to_irt) != len(irt_little_rotations):
            return generic_labels

        resolved_labels: list[str] = []
        used_irt_labels: set[str] = set()
        for generic_label, irrep in zip(generic_labels, irreps):
            spgrep_character = np.array(get_character(irrep), dtype=complex)[mapping_to_irt]
            best_label = generic_label
            best_overlap = -1.0
            for irt_irrep in irt_irreps:
                irt_label = f"{irt_irrep.name}({irt_irrep.dim})"
                irt_character = (
                    np.array(list(irt_irrep.characters.values()), dtype=complex) * character_phases
                )
                overlap = np.abs(
                    np.dot(spgrep_character, np.conjugate(irt_character)) / irt_irrep.nsym
                )
                if overlap > best_overlap:
                    best_label = irt_label
                    best_overlap = float(overlap)
            if best_overlap < 0.9:
                best_label = generic_label
            elif best_label in used_irt_labels:
                best_label = f"{best_label} [{generic_label}]"
            used_irt_labels.add(best_label)
            resolved_labels.append(best_label)
        return resolved_labels

    def describe_mode_spaces(
        self,
        qpoint: list[float],
    ) -> tuple[object, list[NDArray[np.complex128]], list[str]]:
        """Irreps, projected vibration spaces and their labels at q.

        The one-call form of :meth:`get_vibration_rep`,
        :meth:`get_irrep_labels` and :meth:`get_vibration_basis`, as
        ``crystod-phonon --vibration`` prints them (one "Mode Space" line per
        irrep occurrence, with its label and dimension).

        Args:
            qpoint: Fractional coordinates of q in the primitive reciprocal
                basis.

        Returns:
            ``(irreps, basis_spaces, basis_space_labels)``: the spgrep irreps,
            one ``(dim, 3 * n_atoms)`` array per irrep occurrence, and the
            ISO-IR label of each space. The mode-space numbers of the command
            are 1-based positions in ``basis_spaces``. These are spgrep's raw
            projected spaces; the partners the command writes out are those
            of :meth:`get_symmetry_adapted_spaces`.
        """
        irreps, vibration_rep, mapping_little_group = self.get_vibration_rep(qpoint)
        irrep_labels = self.get_irrep_labels(qpoint, irreps, mapping_little_group)
        basis_spaces, basis_space_labels = self.get_vibration_basis(irreps, vibration_rep, irrep_labels)
        return irreps, basis_spaces, basis_space_labels

    def get_symmetry_adapted_spaces(self, qpoint: list[float]) -> list[NDArray[np.complex128]]:
        """Symmetry-adapted partners of every mode space, as ``--vibration`` freezes them.

        The spaces of :meth:`describe_mode_spaces` (same order, same
        dimensions) with their partners fixed by the conventions of
        ``crystod-phonon --modulation``
        (:func:`crystod.symmetry_adapted_modes.solve_symmetry_adapted_spaces`,
        the mode solver with no dynamical matrix): the spaces of a repeated
        irrep are an orthonormal basis of its isotypic space fixed by the
        displacements (the atom orbit each lives on, then how the atoms move
        along the crystal axes, then the products of the displacements of
        neighbouring atoms), not the copies spgrep's projection happens to
        return; at a time-reversal-invariant q (``2q`` a reciprocal lattice
        vector) every partner is, with the Bloch factor
        ``exp(2 pi i q . x_j)`` of each atom, a real displacement pattern
        along a direction symmetry operations fix up to sign -- a single
        partner freezes into an isotropy subgroup of its irrep, whatever the
        origin; at any other q each space is a complex Bloch wave whose
        global phase is a convention. A complex irrep and its conjugate,
        which time reversal joins into one real space at such a q, share its
        real partners: the first half belongs to the irrep whose label sorts
        first. At a time-reversal-invariant q the k-th space of a given label
        thus holds the same partners however q is written (q, q + G or -q);
        the position of a label in the list follows spgrep's irrep order at
        the q given and can differ between those spellings. (At any other q
        the relative phases of the partners of a larger irrep follow the
        irrep matrices spgrep builds at the q given.) The rows are unit-norm
        symmetry-adapted displacement patterns, not normal modes: there are
        no force constants to select a combination of the spaces of a
        repeated irrep, and no masses.

        Args:
            qpoint: Fractional coordinates of q in the primitive reciprocal
                basis.

        Returns:
            One ``(dim, 3 * n_atoms)`` array per mode space, rows in the
            atom-position phase convention of :meth:`get_vibration_basis`
            (the Bloch factor of each atom not included), ready for
            :meth:`get_supercell_displacements`.
        """
        from .symmetry_adapted_modes import solve_symmetry_adapted_spaces

        irreps, _, mapping_little_group = self.get_vibration_rep(qpoint)
        labels = self.get_irrep_labels(qpoint, irreps, mapping_little_group)
        # natural order of the labels (T2 before T10), independent of spgrep's
        # irrep order at the q given
        keys = [
            tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", label))
            for label in labels
        ]
        phase = np.repeat(
            np.exp(2j * np.pi * (np.asarray(self.primitive_cell.scaled_positions, dtype=float)
                                 @ np.asarray(qpoint, dtype=float))),
            3,
        )
        return [
            rows * phase.conj()[None, :]
            for rows in solve_symmetry_adapted_spaces(self, qpoint, irrep_keys=keys)
        ]

    def get_supercell_size(self, qpoint: list[float]) -> tuple[int, int, int]:
        """Supercell multiplicities along a, b, c commensurate with q.

        Args:
            qpoint: Fractional coordinates of q in the primitive reciprocal
                basis.

        Returns:
            ``(n1, n2, n3)``: 1 for a zero component, else the denominator of
            the component (limited to 6).
        """
        sizes = []
        for component in qpoint:
            if abs(component) < 1e-10:
                sizes.append(1)
            else:
                sizes.append(Fraction(float(component)).limit_denominator(6).denominator)
        return tuple(sizes)

    def get_supercell_displacements(
        self,
        qpoint: list[float],
        mode_vector: NDArray[np.complex128],
        supercell_size: tuple[int, int, int],
    ):
        """Displacement pattern of one basis vector on a supercell.

        Atom j of the primitive cell at lattice translation R is displaced by
        ``Re(mode_j * exp(2 pi i q . (R + x_j)))`` (unit amplitude), with
        ``x_j`` the scaled position of atom j in ``primitive_cell``, the cell
        the supercell is built from: the Bloch wave the ket ``mode``
        describes in the atom-position phase convention. For a row of
        :meth:`get_symmetry_adapted_spaces` at a time-reversal-invariant q
        the displacement of every primitive cell has unit norm.

        Args:
            qpoint: Fractional coordinates of q in the primitive reciprocal
                basis.
            mode_vector: One row of :meth:`get_symmetry_adapted_spaces` (what
                ``--vibration`` writes) or of a projected space of
                :meth:`describe_mode_spaces` (whose global phase, and at a
                time-reversal-invariant q its partner basis, spgrep leaves
                arbitrary), ``3 * n_atoms`` complex components.
            supercell_size: ``(n1, n2, n3)`` multiplicities, e.g. from
                :meth:`get_supercell_size`.

        Returns:
            ``(positions, displacements, symbols, supercell_lattice)``:
            Cartesian positions and displacements of the supercell atoms,
            shape ``(n1 * n2 * n3 * n_atoms, 3)``, their chemical symbols, and
            the supercell lattice vectors as rows.
        """
        primitive = self.primitive_cell
        n_atoms = len(primitive.scaled_positions)
        lattice = primitive.cell
        frac_pos = primitive.scaled_positions
        symbols_prim = get_chemical_symbols(primitive)
        mode = mode_vector.reshape((-1, 3))
        n1, n2, n3 = supercell_size

        all_positions = []
        all_displacements = []
        all_symbols = []

        for i1 in range(n1):
            for i2 in range(n2):
                for i3 in range(n3):
                    translation_frac = np.array([i1, i2, i3])
                    for atom_index in range(n_atoms):
                        pos_frac = frac_pos[atom_index] + translation_frac
                        pos_cart = pos_frac @ lattice
                        all_positions.append(pos_cart)
                        # the Bloch factor of the atom's own position, not
                        # only of its cell
                        phase = np.exp(2j * np.pi * np.dot(qpoint, pos_frac))
                        displacement = np.real(mode[atom_index] * phase)
                        all_displacements.append(displacement)
                        all_symbols.append(symbols_prim[atom_index])

        supercell_lattice = lattice.copy()
        supercell_lattice[0] *= n1
        supercell_lattice[1] *= n2
        supercell_lattice[2] *= n3
        return (
            np.array(all_positions),
            np.array(all_displacements),
            all_symbols,
            supercell_lattice,
        )

    def write_displaced_structure(
        self,
        positions: NDArray[np.float64],
        displacements: NDArray[np.float64],
        symbols: list[str],
        supercell_lattice: NDArray[np.float64],
        amplitude: float,
        output_path: str,
    ) -> None:
        """Write ``positions + amplitude * displacements`` as a POSCAR.

        Args:
            positions: Cartesian positions from
                :meth:`get_supercell_displacements`.
            displacements: The unit-amplitude displacements from the same
                call.
            symbols: Chemical symbols of the atoms.
            supercell_lattice: Supercell lattice vectors as rows.
            amplitude: Displacement amplitude in Angstroms.
            output_path: Output file path (VASP format, direct coordinates).
        """
        atoms = Atoms(
            symbols=symbols,
            positions=positions + amplitude * displacements,
            cell=supercell_lattice,
            pbc=True,
        )
        ase_write(output_path, atoms, format="vasp", direct=True)


def format_fraction_vector(qpoint) -> str:
    """Coordinates as ``(0, 0, 1/2)``: each component as a fraction of
    denominator at most 24 where it is one (``1/3``, ``-1/2``), else with four
    decimals.

    Args:
        qpoint: Three fractional coordinates.

    Returns:
        The text, e.g. ``"(1/3, 1/3, 0)"``.
    """
    parts = []
    for value in qpoint:
        value = float(value)
        fraction = Fraction(value).limit_denominator(24)
        if abs(float(fraction) - value) < 1e-6:
            parts.append(str(fraction))
        else:
            parts.append(f"{value + 0.0:.4f}")
    return "(" + ", ".join(parts) + ")"


def _in_seekpath_order(
    vibrations: "SymmetryOnlyVibrations", qpoints: dict[str, list[float]]
) -> dict[str, list[float]]:
    """The special q points in the order seekpath lists its points.

    The set, names and coordinates are those of
    :meth:`SymmetryOnlyVibrations.get_special_qpoints`; only the display
    order follows seekpath (``GM, A, K, H, M, L`` for P6_3mc). A point that
    matches no seekpath point (mod G) keeps its table position after the
    matched ones; the table order is kept when seekpath fails.
    """
    import warnings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            seek_points = list(vibrations.get_high_symmetry_qpoints().values())
    except Exception:
        return qpoints

    def rank(item: tuple[int, tuple[str, list[float]]]) -> tuple[int, int]:
        table_index, (_, coords) = item
        for seek_index, seek in enumerate(seek_points):
            delta = np.asarray(coords, dtype=float) - np.asarray(seek, dtype=float)
            if np.allclose(delta, np.rint(delta), atol=1e-6):
                return seek_index, table_index
        return len(seek_points), table_index

    ordered = sorted(enumerate(qpoints.items()), key=rank)
    return {label: coords for _, (label, coords) in ordered}


def _print_qpoint_block(title: str, qpoints: dict[str, list[float]]) -> None:
    print(f"\n* {title} *")
    for label, coords in qpoints.items():
        print(f"  {label:8s} {format_fraction_vector(coords)}")


def _print_mode_spaces(
    basis_spaces: list[NDArray[np.complex128]],
    irrep_labels: list[str],
    activities: list | None = None,
    mulliken: dict[str, str] | None = None,
) -> None:
    from .phonon_activity import _with_mulliken

    print("\n* Irrep-grouped vibration spaces *")
    for mode_index, (space, irrep_label) in enumerate(zip(basis_spaces, irrep_labels), start=1):
        dim = space.shape[0]
        activity = ""
        if activities is not None:
            activity = f", activity = {'+'.join(activities[mode_index - 1].activity)}"
        print(
            f"  Mode Space {mode_index:2d}: irrep = {_with_mulliken(irrep_label, mulliken)}, "
            f"dimension = {dim}, component numbers = 1..{dim}{activity}"
        )
    if activities is not None:
        from .phonon_activity import _activity_summary_parts

        print("\n* Gamma-point activity *")
        for part in _activity_summary_parts(activities, True, mulliken):
            print(part)
        print("  (acoustic: one set per occurrence of the irrep in the vector representation;")
        print("   without force constants the mode spaces are symmetry-adapted patterns,")
        print("   not normal modes)")


def _print_gamma_extras(vibrations, cell, irrep_labels, args, mulliken=None) -> None:
    """The Wyckoff-orbit breakdown at Gamma and, with ``--raman-tensor``, the
    Raman tensors in the axes of the input cell."""
    from .phonon_activity import (
        _gamma_irreps,
        _input_axes_vibrations,
        _raman_records,
        format_raman_tensors,
        format_wyckoff_orbits,
        wyckoff_orbit_decomposition,
    )

    lines = format_wyckoff_orbits(
        wyckoff_orbit_decomposition(vibrations), irrep_labels, mulliken=mulliken
    )
    print("\n* Wyckoff-orbit breakdown (Gamma) *")
    print("\n".join(lines[1:]))
    if args.raman_tensor:
        table = _gamma_irreps(_input_axes_vibrations(cell, args.tolerance))
        lines = format_raman_tensors(_raman_records(table), mulliken=mulliken)
        print("\n* Raman tensors (Cartesian axes of the input cell) *")
        print("\n".join(lines[1:]))


def _gamma_space_activities(vibrations, qpoint, basis_spaces, irrep_labels) -> list | None:
    """The activity of every mode space when q is Gamma, else None; None too
    when the spaces cannot be matched one to one (never expected)."""
    from .phonon_activity import _is_gamma, mode_space_activities

    if not _is_gamma(qpoint):
        return None
    activities = mode_space_activities(vibrations, qpoint)
    expected = [(re.sub(r"\(\d+\)$", "", label), space.shape[0])
                for label, space in zip(irrep_labels, basis_spaces)]
    found = [(record.labels[0], record.dimension) for record in activities]
    return activities if found == expected else None


def _gamma_mulliken(vibrations, qpoint) -> dict[str, str]:
    """The Mulliken symbols of the Gamma irreps when q is Gamma, else {}."""
    from .phonon_activity import _is_gamma, mulliken_symbols

    if not _is_gamma(qpoint):
        return {}
    try:
        return mulliken_symbols(vibrations)
    except Exception as exc:  # the bracket is an annotation; never fail on it
        import warnings

        warnings.warn(f"no Mulliken symbols: {exc}", stacklevel=2)
        return {}


def main(argv: list[str] | None = None) -> None:
    from .star_of_k import read_poscar_or_exit

    args = build_parser().parse_args(argv)
    cell = read_poscar_or_exit(args.poscar)
    vibrations = SymmetryOnlyVibrations(cell=cell, symprec=args.tolerance)

    _print_qpoint_block(
        "Q points (primitive)",
        _in_seekpath_order(vibrations, vibrations.get_special_qpoints()),
    )
    if args.list_qpoints:
        return

    if not args.qpoint:
        raise ValueError("--qpoint is required unless --list-qpoints is used.")

    try:
        qpoint_label, qpoint = vibrations.resolve_special_qpoint(args.qpoint)
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from None
    if args.raman_tensor and not np.allclose(qpoint, np.rint(qpoint), atol=1e-8):
        raise SystemExit(
            f"ERROR: --raman-tensor is defined at Gamma only (--qpoint GM); got "
            f"{qpoint_label} = {format_fraction_vector(qpoint)}."
        )
    _print_qpoint_block("Selected Q point", {qpoint_label: qpoint})

    irreps, basis_spaces, irrep_labels = vibrations.describe_mode_spaces(qpoint)
    activities = _gamma_space_activities(vibrations, qpoint, basis_spaces, irrep_labels)
    mulliken = _gamma_mulliken(vibrations, qpoint)
    _print_mode_spaces(basis_spaces, irrep_labels, activities, mulliken)
    if activities is not None:
        _print_gamma_extras(vibrations, cell, irrep_labels, args, mulliken)

    if args.mode_index is None:
        return

    if args.mode_index < 1 or args.mode_index > len(basis_spaces):
        raise SystemExit(
            f"ERROR: mode-space number {args.mode_index} is out of range "
            f"[1, {len(basis_spaces)}] (numbering is 1-based)."
        )

    selected_space = basis_spaces[args.mode_index - 1]
    if args.component_index < 1 or args.component_index > selected_space.shape[0]:
        raise SystemExit(
            f"ERROR: component number {args.component_index} is out of range "
            f"[1, {selected_space.shape[0]}] (numbering is 1-based)."
        )

    # the symmetry-adapted partner, not the raw projected row: its Bloch phase
    # and, at a time-reversal-invariant q, its partner basis are fixed so that
    # the component freezes into an isotropy subgroup of its irrep
    adapted_spaces = vibrations.get_symmetry_adapted_spaces(qpoint)
    if [space.shape for space in adapted_spaces] != [space.shape for space in basis_spaces]:
        raise RuntimeError("The symmetry-adapted spaces do not match the projected spaces.")
    mode_vector = adapted_spaces[args.mode_index - 1][args.component_index - 1]
    supercell_size = vibrations.get_supercell_size(qpoint)
    print("\n* Selected basis vector *")
    print(f"Selected mode space: {args.mode_index}")
    print(f"Selected irrep     : {irrep_labels[args.mode_index - 1]}")
    print(f"Selected component : {args.component_index}")
    print(f"Commensurate supercell size: {supercell_size}")

    positions, displacements, symbols, supercell_lattice = vibrations.get_supercell_displacements(
        qpoint=qpoint,
        mode_vector=mode_vector,
        supercell_size=supercell_size,
    )
    norms = np.linalg.norm(displacements, axis=1)
    print(f"Supercell atom count: {len(symbols)}")
    print(f"Displacement norm range (unit amplitude): min={norms.min():.6f}, max={norms.max():.6f}")
    print("First 5 displacement vectors:")
    for index in range(min(5, len(displacements))):
        print(
            f"  {index:2d} {symbols[index]:2s} "
            f"pos={np.round(positions[index], 6).tolist()} "
            f"disp={np.round(displacements[index], 6).tolist()}"
        )

    if args.export_npz:
        extra = {}
        if activities is not None:
            extra["activities"] = np.array(
                ["+".join(record.activity) for record in activities], dtype=object
            )
        np.savez(
            args.export_npz,
            **extra,
            positions=positions,
            displacements=displacements,
            symbols=np.array(symbols, dtype=object),
            supercell_lattice=supercell_lattice,
            qpoint=np.array(qpoint, dtype=float),
            qpoint_label=np.array(qpoint_label, dtype=object),
            mode_index=np.array(args.mode_index),
            component_index=np.array(args.component_index),
            irrep_labels=np.array(irrep_labels, dtype=object),
            selected_irrep_label=np.array(irrep_labels[args.mode_index - 1], dtype=object),
            mode_space_dimensions=np.array([space.shape[0] for space in basis_spaces], dtype=int),
            selected_mode_dimension=np.array(selected_space.shape[0], dtype=int),
            amplitude=np.array(args.amplitude, dtype=float),
        )
        print(f"Saved mode data to: {args.export_npz}")

    if args.output:
        vibrations.write_displaced_structure(
            positions=positions,
            displacements=displacements,
            symbols=symbols,
            supercell_lattice=supercell_lattice,
            amplitude=args.amplitude,
            output_path=args.output,
        )
        print(f"Saved displaced structure to: {args.output}")


if __name__ == "__main__":
    main()
