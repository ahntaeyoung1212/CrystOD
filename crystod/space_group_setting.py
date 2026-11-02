"""Axis settings of a space group (crystod-group --cif-axis / --spacegroup-set).

A CIF or POSCAR is usually written in the default setting of its space
group, but every space group with a choice of axes, unique axis, cell choice,
origin or rhombohedral/hexagonal axes has several settings: the Hall-symbol
entries of the International Tables (spglib's 530 Hall numbers).  For Cmcm
(No. 63) these are Cmcm (a, b, c), Ccmm (b, a, -c), Amma (c, a, b),
Amam (-c, b, a), Bbmm (b, c, a) and Bmmb (a, -c, b), the list VESTA shows
under "Setting".

``--cif-axis`` lists every setting of the space group of a structure file
with the lattice parameters the file's own cell has in that setting, and
marks the setting the file is in; ``--cif2poscar --spacegroup-set SETTING``
writes the structure as the conventional cell of a chosen setting (see
:func:`crystod.poscar2cif.cif2poscar_main`).

Convention: fractional coordinates transform as ``x_setting = P x_input + p``
(column form), so the lattice vectors of the setting, in the input's
Cartesian frame, are the rows of ``inv(P).T @ L_input``.  The first setting
is spglib's standardization of the input (the input itself when it is in one
of the settings); the monoclinic and orthorhombic settings follow from it
with the matrices of the International Tables (axis permutations and cell
choices), the others are spglib's standardization into each Hall setting.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

_ORTHORHOMBIC = range(16, 75)
_MONOCLINIC = range(3, 16)
_PERMUTATION = re.compile(r"(-?[abc])(-?[abc])(-?[abc])")
_UNIQUE_AXIS = re.compile(r"(-?)([abc])([123]?)")


@dataclass
class SpaceGroupSetting:
    """One setting (Hall-symbol entry) of a space group, seen from a structure.

    Attributes:
        index: 1-based position in the list of settings of the group
            (Hall-number order: the order of VESTA and the International
            Tables).
        hall_number: spglib Hall number (1-530).
        symbol: Hermann-Mauguin symbol of the setting, e.g. ``Amam`` or
            ``P2_1/n``.
        full_symbol: Full Hermann-Mauguin symbol, e.g. ``A 2_1/m 2/a 2/m``.
        choice: spglib's choice string, e.g. ``-cba``, ``b2``, ``2``,
            ``R`` (empty for the single or default setting of a group).
        axes_text: The choice in words, e.g. ``(-c, b, a)`` or
            ``(unique axis b, cell choice 2)``.
        transformation: ``P`` of ``x_setting = P x_input + p``, shape
            ``(3, 3)``.
        origin_shift: ``p``, shape ``(3,)``, wrapped into ``[0, 1)``.
        lattice: Lattice vectors of the setting as rows, in the input's
            Cartesian frame and not idealized (Angstrom).
        parameters: ``(a, b, c, alpha, beta, gamma)`` of ``lattice``
            (Angstrom, degrees).
        is_input: True when the input cell is the conventional cell of this
            setting (same symmetry operations, origin included; two rows
            of No. 68 share one Hall symbol and are marked together).
        shared_symbol: True when another setting of the group has the same
            ``symbol`` (it is then told apart by ``choice``).
        std_lattice: The idealized conventional lattice of the setting
            (rows; a along x, b in the xy plane): spglib's conventional cell
            carried into the setting by ``P`` and ``p``.
        std_positions: Fractional coordinates in ``std_lattice``.
        std_numbers: Atomic numbers of ``std_positions``.
    """

    index: int
    hall_number: int
    symbol: str
    full_symbol: str
    choice: str
    axes_text: str
    transformation: np.ndarray
    origin_shift: np.ndarray
    lattice: np.ndarray
    parameters: tuple
    is_input: bool
    shared_symbol: bool
    std_lattice: np.ndarray
    std_positions: np.ndarray
    std_numbers: np.ndarray

    @property
    def label(self) -> str:
        """Symbol and axes, e.g. ``Amam (-c, b, a)``."""
        return f"{self.symbol} {self.axes_text}"

    @property
    def tag(self) -> str:
        """File-name tag: the symbol without ``/`` and ``_``, plus
        ``_<choice>`` when the symbol is shared (``Amam``, ``P21n_b2``,
        ``Fd-3m_2``, ``R-3c_R``)."""
        tag = self.symbol.replace("/", "").replace("_", "")
        if self.shared_symbol and self.choice:
            tag += f"_{self.choice}"
        return tag


@dataclass
class SettingTable:
    """Every setting of the space group of a structure.

    Attributes:
        number: Space-group number.
        symbol: Hermann-Mauguin symbol of the default setting (spglib's
            symbol of the input).
        settings: The ``SpaceGroupSetting`` records in Hall-number order.
        input_index: 1-based index of the (first) setting the input cell
            is in, or None (a primitive cell, a supercell or a shifted
            origin).
        tolerance: Symmetry tolerance (symprec, Angstrom).
        input_parameters: ``(a, b, c, alpha, beta, gamma)`` of the input
            cell.
    """

    number: int
    symbol: str
    settings: list
    input_index: int | None
    tolerance: float
    input_parameters: tuple


def _field(dataset, name):
    """A field of a spglib dataset (a dict before spglib 2.4, an object after)."""
    if isinstance(dataset, dict):
        return dataset.get(name)
    return getattr(dataset, name, None)


def _cell_parameters(lattice) -> tuple:
    """(a, b, c, alpha, beta, gamma) of row lattice vectors."""
    lattice = np.asarray(lattice, dtype=float)
    lengths = np.linalg.norm(lattice, axis=1)

    def angle(i, j):
        cosine = np.dot(lattice[i], lattice[j]) / (lengths[i] * lengths[j])
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    return (float(lengths[0]), float(lengths[1]), float(lengths[2]),
            angle(1, 2), angle(0, 2), angle(0, 1))


def read_structure_cell(path: str):
    """The cell of a CIF or POSCAR file as spglib reads it.

    A POSCAR is read as VASP reads it (scale line, Cartesian coordinates;
    :func:`crystod.vasp_io.poscar_structure`); a CIF as pymatgen reads it.

    Args:
        path: CIF or POSCAR file.

    Returns:
        ``(lattice, frac, numbers)``: lattice vectors as rows (Angstrom),
        fractional coordinates and atomic numbers.

    Raises:
        ValueError: The file is missing, unreadable or disordered.

    Example:
        >>> from crystod.space_group_setting import read_structure_cell
        >>> lattice, frac, numbers = read_structure_cell(
        ...     "example/test_POSCARs/225_PPOSCAR_NaCl")  # doctest: +SKIP
    """
    if not os.path.isfile(path):
        raise ValueError(f"structure file not found: {path}")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from pymatgen.core import Structure
        except ImportError as exc:
            raise ValueError("reading a CIF or POSCAR requires pymatgen "
                             "(pip install pymatgen)") from exc

        try:
            structure = Structure.from_file(path)
        except Exception:
            try:
                with open(path) as handle:
                    structure = Structure.from_str(handle.read(), fmt="poscar")
            except Exception as exc:
                raise ValueError(f"could not read {path}: {exc}") from exc
        # a POSCAR means the cell VASP reads from it; a CIF stays as read
        from .vasp_io import poscar_structure

        structure = poscar_structure(structure, path)
    if not structure.is_ordered:
        raise ValueError(f"{path} has partially occupied sites; "
                         "an ordered structure is needed")
    return (np.asarray(structure.lattice.matrix, dtype=float),
            np.asarray(structure.frac_coords, dtype=float),
            [int(site.specie.Z) for site in structure])


def _setting_symbol(number: int, full_symbol: str, short_symbol: str) -> str:
    """Hermann-Mauguin symbol of one setting.

    spglib's short symbol is the same for every monoclinic setting (all nine
    of No. 14 read P2_1/c), so monoclinic symbols drop the ``1`` axes of the
    full symbol (``P 1 2_1/n 1`` -> ``P2_1/n``).  Orthorhombic symbols take
    the part after ``/`` of each full-symbol axis (``A 2_1/m 2/a 2/m`` ->
    ``Amam``); this agrees with spglib's short symbol except Hall 331, whose
    short symbol is the obsolete ``Bbcb`` of the setting ``B 2/b 2/e 2/b``.
    """
    parts = full_symbol.split()
    if number in _MONOCLINIC:
        return parts[0] + "".join(part for part in parts[1:] if part != "1")
    if number in _ORTHORHOMBIC:
        return parts[0] + "".join(part.split("/")[-1] for part in parts[1:])
    return short_symbol


def _axes_text(choice: str) -> str:
    """spglib's choice string in words."""
    if choice in ("", "abc"):
        return "(a, b, c)"
    if choice == "H":
        return "(hexagonal axes)"
    if choice == "R":
        return "(rhombohedral axes)"
    if choice in ("1", "2"):
        return f"(origin choice {choice})"
    match = _PERMUTATION.fullmatch(choice)
    if match:
        return "(" + ", ".join(match.groups()) + ")"
    if choice[:1] in ("1", "2"):
        match = _PERMUTATION.fullmatch(choice[1:])
        if match:
            return "(" + ", ".join(match.groups()) + f"; origin choice {choice[0]})"
    match = _UNIQUE_AXIS.fullmatch(choice)
    if match:
        sign, axis, cell_choice = match.groups()
        text = f"(unique axis {sign}{axis}"
        if cell_choice:
            text += f", cell choice {cell_choice}"
        return text + ")"
    return f"({choice})"


def _same_operations(rotations, translations, reference, tol=1e-3) -> bool:
    """True when two sets of symmetry operations agree (translations mod 1)."""
    ref_rotations = np.asarray(reference["rotations"])
    ref_translations = np.asarray(reference["translations"], dtype=float)
    if len(rotations) != len(ref_rotations):
        return False
    unmatched = list(range(len(ref_rotations)))
    for rotation, translation in zip(rotations, translations):
        for k in unmatched:
            if np.array_equal(rotation, ref_rotations[k]):
                diff = np.asarray(translation, dtype=float) - ref_translations[k]
                if np.all(np.abs(diff - np.rint(diff)) < tol):
                    unmatched.remove(k)
                    break
        else:
            return False
    return not unmatched


_UNIT = {"a": (1, 0, 0), "b": (0, 1, 0), "c": (0, 0, 1)}
# ITA cell choices 2 and 3 of unique axis b, in the basis of cell choice 1:
# (a2, c2) = (-a1 - c1, a1), (a3, c3) = (c1, -a1 - c1)
_CELL_CHOICE = {
    "1": np.eye(3),
    "2": np.array([[-1.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    "3": np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, -1.0]]),
}
# the axis permutation that takes unique axis b to each unique axis
_UNIQUE_PERMUTATION = {"b": "abc", "-b": "-cba", "c": "cab", "-c": "a-cb",
                       "a": "bca", "-a": "ba-c"}
_OFFSET_GRID = np.array([(i, j, k) for i in range(24) for j in range(24)
                         for k in range(24)], dtype=float) / 24.0


def _permutation_basis(choice: str) -> np.ndarray:
    """Rows: the axes of a permutation choice (``-cba``) in the old axes."""
    rows = []
    for token in _PERMUTATION.fullmatch(choice).groups():
        sign = -1.0 if token.startswith("-") else 1.0
        rows.append([sign * v for v in _UNIT[token[-1]]])
    return np.array(rows)


def _nominal_basis(number: int, choice: str):
    """The ITA basis of a setting in the axes of the first setting, or None.

    Returns the rows ``(a_j, b_j, c_j)`` in terms of the first setting's
    ``(a_1, b_1, c_1)`` for the monoclinic and orthorhombic groups (the axis
    permutation of the choice string, and the cell-choice matrix of the
    International Tables); None for the other groups, whose settings differ
    only by the origin or by hexagonal/rhombohedral axes.
    """
    if number in _MONOCLINIC:
        match = re.fullmatch(r"(-?[abc])([123]?)", choice)
        if not match:
            return None
        axis, cell_choice = match.groups()
        return (_permutation_basis(_UNIQUE_PERMUTATION[axis])
                @ _CELL_CHOICE[cell_choice or "1"])
    if number in _ORTHORHOMBIC:
        permutation = choice.lstrip("12")
        if permutation == "":
            return np.eye(3)
        if _PERMUTATION.fullmatch(permutation):
            return _permutation_basis(permutation)
    return None


def _origin_offset(coordinates, reference, target, tol=1e-3):
    """The origin ``t`` that makes ``x' = N x + t`` carry ``reference`` onto
    ``target`` (symmetry operations of two Hall settings), or None.

    ``t`` is searched on a grid of 1/24; the shortest one is returned.
    """
    inverse = np.linalg.inv(coordinates)
    rotations = [np.rint(coordinates @ r @ inverse).astype(int)
                 for r in np.asarray(reference["rotations"])]
    translations = [coordinates @ t
                    for t in np.asarray(reference["translations"], dtype=float)]
    target_rotations = np.asarray(target["rotations"])
    target_translations = np.asarray(target["translations"], dtype=float)
    if (sorted(tuple(r.ravel()) for r in rotations)
            != sorted(tuple(r.ravel()) for r in target_rotations)):
        return None
    grid = _OFFSET_GRID
    keep = np.ones(len(grid), dtype=bool)
    for rotation, translation in zip(rotations, translations):
        allowed = target_translations[[np.array_equal(rotation, r)
                                       for r in target_rotations]]
        residual = translation + grid - grid @ rotation.T
        diff = residual[:, None, :] - allowed[None, :, :]
        keep &= np.any(np.all(np.abs(diff - np.rint(diff)) < tol, axis=2), axis=1)
        if not keep.any():
            return None
    candidates = grid[keep]
    wrapped = candidates - np.rint(candidates)
    return candidates[int(np.argmin(np.linalg.norm(wrapped, axis=1)))]


def _wrap(values, tol=1e-6) -> np.ndarray:
    """Fractional values into [0, 1)."""
    values = np.asarray(values, dtype=float)
    values = values - np.floor(values + tol)
    values[np.abs(values) < tol] = 0.0
    return values


def _canonical_lattice(lattice) -> np.ndarray:
    """The same lattice rotated so that a lies along x and b in the xy plane."""
    q, r = np.linalg.qr(np.asarray(lattice, dtype=float).T)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    rotated = (r * signs[:, None]).T
    rotated[np.abs(rotated) < 1e-10] = 0.0
    return rotated


def list_space_group_settings(cell, symprec: float = 0.01) -> SettingTable:
    """Every setting of the space group of a cell, with its lattice parameters.

    The first setting is spglib's standardization of the cell (the cell
    itself when it is the conventional cell of one of the settings).  The
    other settings of a monoclinic or orthorhombic group are reached from it
    with the matrices of the International Tables: the axis permutation of
    the choice string (``-cba``: a' = -c, b' = b, c' = a) and the cell-choice
    matrices (cell choice 2: a' = -a - c, c' = a; cell choice 3: a' = c,
    c' = -a - c), so that the listed lengths follow the axes text.  The
    settings of the other groups (origin choices, hexagonal/rhombohedral
    axes) are spglib's standardization into each Hall setting.

    Args:
        cell: ``(lattice, frac, numbers)`` as spglib takes it (lattice
            vectors as rows, Angstrom).
        symprec: Symmetry tolerance (Angstrom).

    Returns:
        A ``SettingTable``: one ``SpaceGroupSetting`` per Hall number of the
        space group, in Hall-number order.

    Raises:
        ValueError: spglib finds no symmetry at this tolerance, or the
            tolerance is not positive.

    Example:
        >>> from crystod import group
        >>> table = group.list_space_group_settings(
        ...     ([[4, 0, 0], [0, 5, 0], [0, 0, 6]], [[0, 0, 0]], [1]))
        >>> table.number, len(table.settings), table.settings[0].label
        (47, 1, 'Pmmm (a, b, c)')
    """
    import spglib

    if not symprec > 0:
        raise ValueError(f"the tolerance must be positive (got {symprec})")
    lattice = np.asarray(cell[0], dtype=float)
    cell = (lattice, np.asarray(cell[1], dtype=float), list(cell[2]))
    try:
        dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
    except Exception:  # spglib >= 2.5 can raise instead of returning None
        dataset = None
    if dataset is None:
        raise ValueError("spglib could not determine the symmetry of the "
                         f"structure (tolerance {symprec}); try another --tolerance")
    number = int(_field(dataset, "number"))
    rotations = np.asarray(_field(dataset, "rotations"))
    translations = np.asarray(_field(dataset, "translations"), dtype=float)

    halls = [h for h in range(1, 531)
             if int(spglib.get_spacegroup_type(h).number) == number]
    raw = []
    for hall in halls:
        sg_type = spglib.get_spacegroup_type(hall)
        full = str(sg_type.international_full)
        raw.append((hall, _setting_symbol(number, full, str(sg_type.international_short)),
                    full, str(sg_type.choice)))
    symbols = [entry[1] for entry in raw]

    # spglib's standardization of the cell into every Hall setting
    spglib_rows = []
    for hall, symbol, _full, _choice in raw:
        try:
            setting_data = spglib.get_symmetry_dataset(cell, symprec=symprec,
                                                       hall_number=hall)
        except Exception:
            setting_data = None
        if setting_data is None:
            raise ValueError(f"spglib gives no cell in the setting {symbol} "
                             f"(Hall number {hall}); try another --tolerance")
        transformation = np.asarray(_field(setting_data, "transformation_matrix"),
                                    dtype=float)
        origin_shift = np.asarray(_field(setting_data, "origin_shift"), dtype=float)
        identity = (np.allclose(transformation, np.eye(3), atol=1e-6)
                    and np.allclose(origin_shift - np.rint(origin_shift), 0.0, atol=1e-6))
        database = spglib.get_symmetry_from_database(hall)
        # Ccce, Aeaa and Bbeb list one Hall symbol twice in origin choice 1
        # (1 = 1ba-c, ...): both rows are the input's setting then
        is_input = identity or _same_operations(rotations, translations, database)
        spglib_rows.append((setting_data, transformation, origin_shift, database,
                            is_input))
    marked = [row[4] for row in spglib_rows]
    input_index = marked.index(True) + 1 if any(marked) else None

    # the ITA frames x_j = N_j x_1 + t_j of the monoclinic/orthorhombic settings
    frames = []
    for hall, _symbol, _full, choice in raw:
        basis = _nominal_basis(number, choice)
        frame = None
        if basis is not None:
            coordinates = np.linalg.inv(basis).T
            offset = _origin_offset(coordinates, spglib_rows[0][3],
                                    spglib.get_symmetry_from_database(hall))
            if offset is not None:
                frame = (coordinates, offset)
        frames.append(frame)

    # the first setting's frame: the input's own when it is in setting k
    first_data, first_p, first_shift = spglib_rows[0][:3]
    reference_p, reference_shift = first_p, first_shift
    if input_index is not None and frames[input_index - 1] is not None:
        coordinates, offset = frames[input_index - 1]
        reference_p = np.linalg.inv(coordinates)
        reference_shift = -reference_p @ offset
    # spglib's first-setting cell -> the first-setting frame chosen above
    to_reference = np.rint(reference_p @ np.linalg.inv(first_p))
    reference_offset = reference_shift - to_reference @ first_shift
    first_lattice = np.asarray(_field(first_data, "std_lattice"), dtype=float)
    first_positions = np.asarray(_field(first_data, "std_positions"), dtype=float)
    first_numbers = np.asarray(_field(first_data, "std_types"), dtype=int)

    settings = []
    for index, ((hall, symbol, full, choice), spglib_row, frame) in enumerate(
            zip(raw, spglib_rows, frames), start=1):
        setting_data, transformation, origin_shift, _database, is_input = spglib_row
        if frame is not None:
            coordinates, offset = frame
            transformation = coordinates @ reference_p
            origin_shift = coordinates @ reference_shift + offset
            cell_p = coordinates @ to_reference
            cell_shift = coordinates @ reference_offset + offset
            if (np.allclose(cell_p, np.eye(3))
                    and np.allclose(cell_shift - np.rint(cell_shift), 0.0)):
                std_lattice, std_positions = first_lattice, first_positions
            else:
                std_lattice = _canonical_lattice(np.linalg.inv(cell_p).T @ first_lattice)
                std_positions = _wrap(first_positions @ cell_p.T + cell_shift)
            std_numbers = first_numbers
        else:
            if is_input:
                transformation, origin_shift = np.eye(3), np.zeros(3)
            std_lattice = np.asarray(_field(setting_data, "std_lattice"), dtype=float)
            std_positions = np.asarray(_field(setting_data, "std_positions"), dtype=float)
            std_numbers = np.asarray(_field(setting_data, "std_types"), dtype=int)
        transformation = np.where(np.abs(transformation) < 1e-12, 0.0, transformation)
        origin_shift = _wrap(origin_shift, tol=1e-8)
        setting_lattice = np.linalg.inv(transformation).T @ lattice
        settings.append(SpaceGroupSetting(
            index=index,
            hall_number=hall,
            symbol=symbol,
            full_symbol=full,
            choice=choice,
            axes_text=_axes_text(choice),
            transformation=transformation,
            origin_shift=origin_shift,
            lattice=setting_lattice,
            parameters=_cell_parameters(setting_lattice),
            is_input=is_input,
            shared_symbol=symbols.count(symbol) > 1,
            std_lattice=std_lattice,
            std_positions=std_positions,
            std_numbers=std_numbers,
        ))
    return SettingTable(
        number=number,
        symbol=str(_field(dataset, "international")),
        settings=settings,
        input_index=input_index,
        tolerance=symprec,
        input_parameters=_cell_parameters(lattice),
    )


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _bare(text: str) -> str:
    return _compact(text).replace("/", "").replace("_", "")


def _listing(settings) -> str:
    return "; ".join(f"No. {s.index} {s.label}"
                     + (f" [choice {s.choice}]" if s.choice else "")
                     for s in settings)


def resolve_setting(table: SettingTable, spec) -> SpaceGroupSetting:
    """The setting a user names.

    Args:
        table: The settings of the group (:func:`list_space_group_settings`).
        spec: The 1-based No. of the setting, its choice string (``-cba``,
            ``b2``, ``2``, ``R``, ``2ba-c``), its symbol (spaces ignored, also
            without ``/`` and ``_``: ``Amam``, ``A m a m``, ``P21n``), its
            full symbol, or its file-name tag (``P21n_b2``).

    Returns:
        The matching ``SpaceGroupSetting``.

    Raises:
        ValueError: The symbol is shared by several settings (the message
            lists them: give the No. or the choice), or nothing matches (the
            message lists every setting).

    Example:
        >>> from crystod import group
        >>> table = group.list_space_group_settings(
        ...     ([[4, 0, 0], [0, 5, 0], [0, 0, 6]], [[0, 0, 0]], [1]))
        >>> group.resolve_setting(table, "1").symbol
        'Pmmm'
    """
    text = str(spec).strip()
    settings = table.settings
    if text.isdigit() and 1 <= int(text) <= len(settings):
        return settings[int(text) - 1]
    by_choice = [s for s in settings if s.choice and s.choice == text]
    if len(by_choice) == 1:
        return by_choice[0]
    by_tag = [s for s in settings if s.tag == text]
    if len(by_tag) == 1:
        return by_tag[0]
    compact = _compact(text)
    matches = [s for s in settings
               if compact in (_compact(s.symbol), _compact(s.full_symbol))]
    if not matches:
        bare = _bare(text)
        matches = [s for s in settings
                   if bare in (_bare(s.symbol), _bare(s.full_symbol))]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"the setting {text} of {table.symbol} (No. {table.number}) is not "
            f"unique ({_listing(matches)}); give the No. or the choice")
    raise ValueError(
        f"no setting {text} of {table.symbol} (No. {table.number}); the "
        f"settings are: {_listing(settings)}")


def transform_to_setting(cell, spec, symprec: float = 0.01):
    """The conventional cell of a structure in a chosen setting.

    Args:
        cell: ``(lattice, frac, numbers)`` as spglib takes it.
        spec: The setting, as :func:`resolve_setting` takes it.
        symprec: Symmetry tolerance (Angstrom).

    Returns:
        ``(setting, std_lattice, std_positions, std_numbers)``: the
        ``SpaceGroupSetting`` and the idealized conventional cell of it
        (the cell ``--cif2poscar --spacegroup-set`` writes).

    Raises:
        ValueError: As :func:`list_space_group_settings` and
            :func:`resolve_setting`.

    Example:
        >>> from crystod import group
        >>> setting, lattice, positions, numbers = group.transform_to_setting(
        ...     ([[4, 0, 0], [0, 5, 0], [0, 0, 6]], [[0, 0, 0]], [1]), "Pmmm")
        >>> len(numbers)
        1
    """
    table = list_space_group_settings(cell, symprec=symprec)
    setting = resolve_setting(table, spec)
    return setting, setting.std_lattice, setting.std_positions, setting.std_numbers


def _rational(value: float) -> str:
    """A number as a short fraction when it is one (0, -1, 1/2, 3/8)."""
    fraction = Fraction(float(value)).limit_denominator(48)
    if abs(float(fraction) - value) < 1e-6:
        return str(fraction)
    return f"{value:.6f}"


def _parameter_text(parameters) -> str:
    a, b, c, alpha, beta, gamma = parameters
    return (f"a = {a:.5f}, b = {b:.5f}, c = {c:.5f} A, alpha = {alpha:.4f}, "
            f"beta = {beta:.4f}, gamma = {gamma:.4f} deg")


def format_transform(setting: SpaceGroupSetting) -> str:
    """``x_<tag> = P x_input + p`` with the rows of P and p as text."""
    rows = ", ".join("(" + ", ".join(_rational(v) for v in row) + ")"
                     for row in setting.transformation)
    shift = "(" + ", ".join(_rational(v) for v in setting.origin_shift) + ")"
    return f"x_{setting.tag} = P x_input + p, P rows {rows}; p = {shift}"


def format_setting_table(table: SettingTable, input_name: str) -> list[str]:
    """The ``--cif-axis`` report as lines (``* Structure *`` and
    ``* CIF axis settings *`` blocks).

    Args:
        table: From :func:`list_space_group_settings`.
        input_name: The input file name to print.

    Returns:
        The lines, without terminators.
    """
    lines = [
        "",
        "* Structure *",
        f"input      : {input_name}",
        f"space group: {table.symbol} (No. {table.number}), tolerance {table.tolerance}",
        f"input cell : {_parameter_text(table.input_parameters)}",
        "",
        "* CIF axis settings *",
    ]
    width = max(len("setting"), *(len(s.label) for s in table.settings))
    lines.append(f"{'No.':<4} {'setting':<{width}} {'a (A)':>10} {'b (A)':>11} "
                 f"{'c (A)':>11} {'alpha':>9} {'beta':>9} {'gamma':>9}")
    for setting in table.settings:
        a, b, c, alpha, beta, gamma = setting.parameters
        line = (f"{setting.index:<4} {setting.label:<{width}} {a:10.5f} {b:11.5f} "
                f"{c:11.5f} {alpha:9.4f} {beta:9.4f} {gamma:9.4f}")
        if setting.is_input:
            line += "   <- input"
        lines.append(line)
    if table.input_index is None:
        lines.append("input cell : not the conventional cell of a listed setting "
                     "(a primitive cell, a supercell or a shifted origin)")
    lines.append("(the Hall-symbol settings of spglib, in the order of VESTA and "
                 "the International Tables;")
    lines.append(" lengths and angles of the input cell in the axes of each setting, "
                 "x_setting = P x_input + p;")
    lines.append(" write one with --cif2poscar -c FILE --spacegroup-set "
                 "<symbol | No. | choice>)")
    return lines


def format_setting_structure(table: SettingTable, setting: SpaceGroupSetting,
                             input_name: str, n_atoms: int) -> list[str]:
    """The ``* Structure *`` block of ``--cif2poscar --spacegroup-set``.

    Args:
        table: From :func:`list_space_group_settings`.
        setting: The chosen setting.
        input_name: The input file name to print.
        n_atoms: Number of atoms in the written cell.

    Returns:
        The lines, without terminators.
    """
    choice = f"{setting.label}, No. {setting.index} of {len(table.settings)}"
    return [
        "",
        "* Structure *",
        f"input      : {input_name}",
        f"space group: {table.symbol} (No. {table.number}), tolerance {table.tolerance}",
        f"setting    : {choice}",
        f"transform  : {format_transform(setting)}",
        f"conventional cell ({setting.symbol} setting), {n_atoms} atoms: "
        + _parameter_text(_cell_parameters(setting.std_lattice)),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crystod-group --cif-axis",
        description="List the axis settings of the space group of a CIF or "
        "POSCAR with the lattice parameters of the file in each setting.",
    )
    parser.add_argument("--cell", "-c", "--cif", "--poscar", dest="cell",
                        required=True, metavar="FILE",
                        help="Input CIF or POSCAR file.")
    parser.add_argument("--tolerance", type=float, default=0.01,
                        help="Symmetry tolerance (symprec) in Angstrom "
                        "(default: 0.01).")
    return parser


def main(argv: list[str] | None = None) -> None:
    """``crystod-group --cif-axis -c FILE``: print the settings table."""
    args = build_parser().parse_args(argv)
    if not args.tolerance > 0:
        raise SystemExit(f"ERROR: --tolerance must be positive (got {args.tolerance})")
    try:
        cell = read_structure_cell(args.cell)
        table = list_space_group_settings(cell, symprec=args.tolerance)
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}")
    print("\n".join(format_setting_table(table, args.cell)))
