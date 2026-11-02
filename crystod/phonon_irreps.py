"""
__author__ = "Hiroki Koiso, Yasuhide Mochizuki"
__copyright__ = "Copyright 2026, Mochizuki group"
__version__ = "1.0"
__maintainer__ = "Hiroki Koiso, Yasuhide Mochizuki"
__email__ = "mochizuki@rs.tus.ac.jp"
__status__ = "Development"
__released_date__ = "November 2, 2024"
__last_update__= "June 29, 2026"
"""

from __future__ import annotations

from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    RawDescriptionHelpFormatter,
    RawTextHelpFormatter,
)

import re
import warnings

import numpy as np
from numpy.typing import NDArray

from .spglib_compat import ensure_spglib_compat

ensure_spglib_compat()

from phonopy import load
from phonopy.structure.cells import get_primitive_matrix_by_centring

from .irreptables_compat import load_irreptables
from .operations import parse_qpoint_token, snap_qpoint
from .phonon_activity import _activity_summary_parts, format_activity_summary
from .runtime_compat import get_symmetry_dataset
from .vasp_io import read_poscar_cell

IrrepTable, Irrep = load_irreptables()


class MyHelpFormatter(
    RawTextHelpFormatter,
    RawDescriptionHelpFormatter,
    ArgumentDefaultsHelpFormatter,
):
    pass


desc = """
This program identify the ISO-IR (ISOTROPY, Miller-Love) labels for the phonon irreducible representations.
POSCAR and FORCE_STES must exist in the directory where this code runs.

# Command Example:
python3 phonon_irreps.py --dim "2 2 2"
"""


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=desc, formatter_class=MyHelpFormatter)
    parser.add_argument(
        "--dim",
        "-dim",
        dest="dim",
        required=True,
        type=str,
        help="Supercell dimension.",
    )
    parser.add_argument(
        "--poscar",
        "-poscar",
        dest="poscar",
        type=str,
        default="POSCAR",
        help="POSCAR.",
    )
    parser.add_argument(
        "--readfc",
        "-readfc",
        dest="readfc",
        action="store_true",
        help="Read FORCE_CONSTANS.",
    )
    parser.add_argument(
        "--tolerance",
        "-tol",
        dest="tol",
        type=float,
        default=1e-3,
        help="Degeneracy tolerance.",
    )
    parser.add_argument(
        "--all-irreps",
        dest="all_irreps",
        action="store_true",
        help="Additionally label the phonon irreps at the midpoints of the\n"
        "seekpath k-path segments (the symmetry lines DT, Z, SM, ...;\n"
        "ISO-IR labels). Slower than the default special-points-only survey.",
    )
    parser.add_argument(
        "--nac",
        dest="nac",
        action="store_true",
        help="Read ./BORN (without it a BORN file is not read) and report the mode\n"
        "effective charges, the dielectric contribution of every IR set at Gamma,\n"
        "eps_0, the acoustic sum rule of the Born charges and the LST check.",
    )
    parser.add_argument(
        "--raman-tensor",
        dest="raman_tensor",
        action="store_true",
        help="Print the symmetry-allowed Raman tensors of the Raman-active Gamma\n"
        "irreps (Cartesian axes of the input cell).",
    )
    parser.add_argument(
        "--output",
        "-o",
        dest="output",
        default=None,
        help="Path of the YAML file (default: phonon_irreps.yaml, or\n"
        "phonon_irreps_all.yaml with --all-irreps). Next to it the phonon\n"
        "dispersion along the k path with the ISO-IR label of every level at\n"
        "its special points is drawn: the YAML name with .pdf.",
    )
    parser.add_argument(
        "--band",
        dest="band",
        default=None,
        help="Manual k path of the dispersion figure (primitive reciprocal basis), e.g.\n"
        '"0 0 0  0 1/2 0  1/2 1/2 0, 1/2 1/2 0  1/2 1/2 1/2" (default: seekpath).',
    )
    parser.add_argument(
        "--band-labels",
        dest="band_labels",
        default=None,
        help="Names of the vertices of --band (default: their ISO-IR k-vector types).",
    )
    parser.add_argument(
        "--npoints",
        dest="npoints",
        type=int,
        default=None,
        help="q points per leg of the k path of the dispersion figure (default 51).",
    )
    parser.add_argument(
        "--threshold",
        dest="threshold",
        type=float,
        default=-0.1,
        help="Frequency (THz) below which the dispersion figure draws a level in red\n"
        "(imaginary).",
    )
    return parser


def _born_missing(born_dirs) -> str:
    """The error of ``--nac`` without a BORN file, naming the directories
    that were searched."""
    import os

    places: list[str] = []
    for directory in born_dirs:
        if os.path.abspath(str(directory)) == os.getcwd():
            place = "the current directory"
        else:
            place = os.path.normpath(str(directory))
        if place not in places:
            places.append(place)
    return (
        "ERROR: --nac requires a BORN file (Born effective charges and dielectric\n"
        f"       tensor) in {' or '.join(places)}, e.g. generated with phonopy-vasp-born."
    )


def nac_load_options(nac: bool, *, yaml: bool = False, born_dirs=(".",)) -> dict:
    """Keyword arguments of ``phonopy.load`` that make the NAC choice explicit.

    ``phonopy.load`` applies the non-analytical term correction by default
    and silently reads ``./BORN`` (or the NAC parameters of a phonopy yaml).
    Every ``crystod-phonon`` mode that loads force data passes these options
    instead, so that NAC is used only with ``--nac``.

    Args:
        nac: ``True`` when ``--nac`` was given.
        yaml: The force data come from a phonopy yaml; with ``nac`` its own
            NAC parameters take priority over ``./BORN`` (phonopy's order),
            and :func:`check_nac_loaded` reports when neither exists.
        born_dirs: Directories searched for ``BORN``, in order (structure
            route only).

    Returns:
        ``{"is_nac": False}`` without ``nac``; else ``{"is_nac": True}`` plus
        ``born_filename`` on the structure route.

    Raises:
        SystemExit: ``nac`` on the structure route and no ``BORN`` file found.
    """
    import os

    if not nac:
        return {"is_nac": False}
    if yaml:
        return {"is_nac": True}
    for directory in born_dirs:
        candidate = os.path.normpath(os.path.join(str(directory), "BORN"))
        if os.path.isfile(candidate):
            return {"is_nac": True, "born_filename": candidate}
    raise SystemExit(_born_missing(born_dirs))


def check_nac_loaded(phonon, nac: bool, source: str) -> str | None:
    """The one-line NAC note of a run, or an error when ``--nac`` found nothing.

    Args:
        phonon: The loaded ``phonopy.Phonopy`` object.
        nac: ``True`` when ``--nac`` was given.
        source: Where the NAC parameters were looked for, for the messages
            (``"BORN"``, or ``"phonopy_params.yaml or BORN"``).

    Returns:
        ``"NAC: Born effective charges and dielectric tensor read from ..."``
        with ``--nac``, ``None`` without.

    Raises:
        SystemExit: ``--nac`` was given and no NAC parameters were found.
    """
    if not nac:
        return None
    if getattr(phonon, "nac_params", None) is None:
        raise SystemExit(
            f"ERROR: --nac found no NAC parameters in {source} (Born effective "
            "charges and dielectric tensor; e.g. a BORN file from phonopy-vasp-born)."
        )
    return f"NAC: Born effective charges and dielectric tensor read from {source}."


def yaml_nac_source(yaml_path: str) -> str:
    """Where ``phonopy.load(yaml_path, is_nac=True)`` takes NAC from: the
    yaml's own ``nac_params`` when it holds them (phonopy's priority), else
    ``BORN`` of the working directory; for the note of
    :func:`check_nac_loaded`."""
    import os

    if not os.path.isfile("BORN"):
        return str(yaml_path)
    try:
        from phonopy.interface.phonopy_yaml import PhonopyYaml

        reader = PhonopyYaml()
        reader.read(yaml_path)
        return str(yaml_path) if reader.nac_params is not None else "BORN"
    except Exception:
        return f"{yaml_path} or BORN"


def format_qpoint(q, decimals: int = 6) -> list[float]:
    """Format a q-point for yaml output using plain Python floats."""
    return [float(np.round(float(value), decimals)) for value in q]


def get_irt_special_points(irt_table, prim_mat) -> tuple[list[str], list[list[float]]]:
    """Unique special q points of the ISO-IR tables, in the primitive basis.

    The ISO-IR tables list their irreps per special k point in the
    conventional reciprocal basis; this collects the distinct points, converts
    them to the primitive basis of the phonopy object, and returns them in
    table order. ``crystod-phonon --irreps`` surveys exactly these points.
    Coordinates are snapped to exact fractions (1/3 stays 1/3, not 0.333333):
    decimal-rounded values break the little-group detection and the ISO-IR
    table lookups downstream.

    Args:
        irt_table: ISO-IR irrep table of the space group, i.e.
            ``IrrepTable(number, spinor=False)`` with ``IrrepTable`` from
            ``crystod.irreptables_compat.load_irreptables()``.
        prim_mat: Conventional-to-primitive matrix of the centring, e.g.
            ``phonopy.structure.cells.get_primitive_matrix_by_centring("P")``.

    Returns:
        ``(q_names, q_list)``: the k-point labels (``"GM"``, ``"R"``, ...) and
        their fractional coordinates in the primitive reciprocal basis, in the
        order of the tables (Gamma first).

    Example:
        >>> from phonopy.structure.cells import get_primitive_matrix_by_centring
        >>> from crystod import phonon
        >>> from crystod.irreptables_compat import load_irreptables
        >>> IrrepTable, _ = load_irreptables()
        >>> table = IrrepTable(221, spinor=False)          # Pm-3m
        >>> prim_mat = get_primitive_matrix_by_centring("P")
        >>> names, points = phonon.get_irt_special_points(table, prim_mat)
        >>> names
        ['GM', 'R', 'X', 'M']
        >>> points[1]
        [0.5, 0.5, 0.5]
    """
    q_list = []
    q_names = []
    for irrep in irt_table.irreps:
        q_primitive = snap_qpoint(np.dot(irrep.k, prim_mat))
        if q_primitive not in q_list:
            q_list.append(q_primitive)
            q_names.append(irrep.kpname)
    return q_names, q_list


def find_star_representative(
    qpoint: list[float] | NDArray[np.float64],
    rotations: NDArray[np.int_],
    q_names: list[str],
    q_list: list[list[float]],
) -> tuple[str, list[float]] | None:
    """Map a q point onto the tabulated arm of its star.

    The ISO-IR tables list only one representative arm per special point
    (e.g. only (1/2, 1/2, 0) for the three M arms of Pm-3m), so a direct
    coordinate lookup fails for the other arms. Some space-group rotation R
    sends q onto a tabulated point when ``q @ R`` equals it modulo
    reciprocal-lattice translations; the spectra of star arms coincide band by
    band, so the labels read at the representative apply to the modes at q.
    This is how ``crystod-phonon --irreps``/``--vector`` and
    :func:`label_phonon_modes` label a non-representative arm.

    Args:
        qpoint: Fractional coordinates of q in the primitive reciprocal basis.
        rotations: Rotation parts of the space-group operations, shape
            ``(n_ops, 3, 3)``, in the same (primitive) basis as ``qpoint`` and
            the tabulated points.
        q_names: Labels of the tabulated special points.
        q_list: Their coordinates, as returned by
            :func:`get_irt_special_points`.

    Returns:
        ``(label, representative_q)`` for the first tabulated point some
        rotation maps q onto, or ``None`` when q lies in no tabulated star.

    Example:
        >>> from crystod import phonon
        >>> from crystod.runtime_compat import get_symmetry_dataset
        >>> names = ["GM", "R", "X", "M"]
        >>> points = [[0, 0, 0], [0.5, 0.5, 0.5], [0, 0.5, 0], [0.5, 0.5, 0]]
        >>> # ph: a phonopy.Phonopy object of cubic SrTiO3 (Pm-3m)
        >>> rotations = get_symmetry_dataset(ph.primitive_symmetry)["rotations"]
        >>> phonon.find_star_representative([0.5, 0, 0], rotations, names, points)
        ('X', [0, 0.5, 0])
    """
    qpoint = np.asarray(qpoint, dtype=float)
    for name, q_special in zip(q_names, q_list):
        target = np.asarray(q_special, dtype=float)
        for rotation in rotations:
            diff = qpoint @ rotation - target
            if (np.abs(diff - np.rint(diff)) < 1e-8).all():
                return name, list(q_special)
    return None


def get_irt_irreps_at_q(
    q: list[float], irt_table, prim_mat, warn: bool = True
) -> list[Irrep]:
    """Get irreps at the q-point from the ISO-IR tables."""
    irreps_at_q = []
    prim_inv = np.linalg.inv(prim_mat)
    conventional_q = np.array(q) @ prim_inv
    for irrep_at_q in irt_table.irreps:
        if np.allclose(irrep_at_q.k, conventional_q):
            irreps_at_q.append(irrep_at_q)
    if not irreps_at_q and warn:
        warnings.warn(f"No irreps at {q} in the ISO-IR tables!", stacklevel=2)
    return irreps_at_q


def get_mapping_to_irt(
    irt_little_r: NDArray[np.int_],
    found_little_r: NDArray[np.int_],
    prim_mat,
) -> list[int]:
    """Get mapping from phonopy little-group rotations to the ISO-IR table order."""
    conv_little_r = prim_mat @ found_little_r @ np.linalg.inv(prim_mat)
    mapping_to_irt = []
    for irt_r in irt_little_r:
        for i, r in enumerate(conv_little_r):
            if np.allclose(irt_r, r):
                mapping_to_irt.append(i)
                break
    return mapping_to_irt


def _input_cell(phonon):
    """The unit cell the phonopy object was built from, as an spglib tuple.

    The ISO-IR frame of the labels is taken from this cell (the one the user
    gave), so that every command names an irrep of one structure the same
    way whether it works on phonopy's primitive cell or on another cell."""
    unitcell = phonon.unitcell
    return (unitcell.cell, unitcell.scaled_positions, unitcell.numbers)


def _isoir_labeler(phonon):
    """The ISO-IR labeller of the phonon labels (that of
    :func:`_get_isoir_band_labels`): phonopy's primitive cell in the ISO-IR
    frame of the input cell; None when unavailable."""
    from .isoir import get_cached_labeler

    primitive = phonon.primitive
    dataset = get_symmetry_dataset(phonon.primitive_symmetry)
    return get_cached_labeler(
        dataset["number"],
        (primitive.cell, primitive.scaled_positions, primitive.numbers),
        phonon.primitive_symmetry.tolerance,
        input_cell=_input_cell(phonon),
    )


def _special_points_in_label_frame(
    phonon, irt_table, prim_mat
) -> tuple[list[str], list[list[float]]]:
    """The special q points of :func:`get_irt_special_points`, named in the
    frame of the labels.

    The list a command shows, surveys and resolves ``--qpoint NAME``
    through: every point named by the labeller of the phonon labels, a point
    that is the -k partner of a tabulated type replaced by -k
    (``crystod.isoir.special_points_in_frame``).  The table's own list stays
    the one to map q onto the tabulated arm for the table-character
    fallback of :func:`get_irrep_labels`.
    """
    from .isoir import special_points_in_frame

    q_names, q_list = get_irt_special_points(irt_table, prim_mat)
    try:
        labeler = _isoir_labeler(phonon)
    except Exception:
        labeler = None
    return special_points_in_frame(q_names, q_list, labeler, canonical=snap_qpoint)


def _get_isoir_band_labels(
    q: list[float],
    phonon,
    phonon_irreps,
) -> list[list[str] | None] | None:
    """ISO-IR (Miller-Love) labels per degenerate band set, or None.

    The primary labelling route at every q (special points, lines, planes,
    the general point): the phonopy band-set characters are decomposed
    against the ISO-IR small irreps (they can be reducible under accidental
    degeneracy), in the ISO-IR frame of the input cell (frame rules in
    ``crystod.isoir``).  A q point tabulated only through its -k partner
    gets the 'A' names of the conjugate irreps (``PA1``).  A set that cannot
    be decomposed is ``None``; :func:`get_irrep_labels` then falls back on
    the direct character overlap at a tabulated special point.
    """
    from .isoir import get_isoir_band_decompositions

    primitive = phonon.primitive
    cell = (primitive.cell, primitive.scaled_positions, primitive.numbers)
    dataset = get_symmetry_dataset(phonon.primitive_symmetry)
    rotations = getattr(phonon_irreps, "_rotations_at_q")
    translations = getattr(phonon_irreps, "_translations_at_q")
    decompositions = get_isoir_band_decompositions(
        dataset["number"],
        cell,
        phonon.primitive_symmetry.tolerance,
        q,
        rotations,
        translations,
        list(phonon_irreps.characters),
        input_cell=_input_cell(phonon),
    )
    labels: list[list[str] | None] = [
        None if decomposed is None
        else [f"{label}({dim})" for label, _, dim in decomposed[0]]
        for decomposed in decompositions
    ]
    return labels if any(label is not None for label in labels) else None


def get_irrep_labels(
    q: list[float],
    phonon,
    irt_table,
    prim_mat,
    degeneracy_tolerance: float,
) -> tuple[list[list[str] | None], list[list[int]], NDArray[np.float64]]:
    """Irrep labels, band indices, and frequencies of the phonon modes at q.

    The labeling step of ``crystod-phonon --irreps``: phonopy's character
    analysis (``Phonopy.set_irreps``) groups the bands at q into degenerate
    sets and computes their characters, and each set is decomposed against
    the ISO-IR (ISOTROPY) small irreps of q by the ISO-IR labeller, with
    Miller-Love labels, at a special point as on a symmetry line or plane or
    at a generic q, in the ISO-IR frame of the input cell; a q point
    tabulated only through its -k partner gets the 'A' names (``PA1``).
    Only at a tabulated special point, a set the labeller cannot decompose
    falls back on the direct character overlap with the special-point table
    (a set is labeled when the overlap exceeds 0.9). For a
    non-representative arm of a star, map q onto the tabulated arm with
    :func:`find_star_representative` first; :func:`label_phonon_modes` does
    both steps in one call.

    Args:
        q: Fractional coordinates of q in the primitive reciprocal basis, as
            tabulated (the representative arm of its star).
        phonon: A ``phonopy.Phonopy`` object with force constants, built with
            ``primitive_matrix="auto"``.
        irt_table: ISO-IR irrep table of the space group (see
            :func:`get_irt_special_points`).
        prim_mat: Conventional-to-primitive matrix of the centring.
        degeneracy_tolerance: Frequency tolerance (THz) within which bands
            count as degenerate; ``--tolerance`` of ``crystod-phonon --irreps``
            (default 1e-3).

    Returns:
        ``(labels, band_indices, frequencies)``: one entry of ``labels`` per
        degenerate set, each a list of ``"R4+(3)"``-style labels (the irrep
        name with its dimension) or ``None`` when no tabulated irrep matched;
        ``band_indices`` the 0-based band indices of each set; and
        ``frequencies`` the THz frequencies of all bands at q.

    Raises:
        ValueError: If the q point can be labeled from neither table.

    Example:
        >>> from phonopy.structure.cells import get_primitive_matrix_by_centring
        >>> from crystod import phonon
        >>> from crystod.irreptables_compat import load_irreptables
        >>> IrrepTable, _ = load_irreptables()
        >>> table = IrrepTable(221, spinor=False)          # ph: cubic SrTiO3
        >>> prim_mat = get_primitive_matrix_by_centring("P")
        >>> labels, bands, freqs = phonon.get_irrep_labels(
        ...     [0.5, 0.5, 0.5], ph, table, prim_mat, 1e-3)
        >>> labels[0], bands[0], round(float(freqs[0]), 4)
        (['R5-(3)'], [0, 1, 2], -1.0867)
    """
    phonon.set_irreps(q=np.array(q), degeneracy_tolerance=degeneracy_tolerance)
    phonon_irreps = phonon.irreps
    irt_irreps = get_irt_irreps_at_q(np.array(q), irt_table, prim_mat, warn=False)

    if not irt_irreps:
        # Not in the special-point table (e.g. a symmetry line/plane or
        # generic q): decompose against the full ISO-IR (ISOTROPY) tables,
        # which cover every
        # k-vector type.  Labels then follow the Miller-Love convention.
        isoir_labels = _get_isoir_band_labels(q, phonon, phonon_irreps)
        if isoir_labels is not None:
            band_indices = phonon_irreps.band_indices
            frequencies = getattr(phonon_irreps, "frequencies", None)
            if frequencies is None:
                frequencies = getattr(phonon_irreps, "_freqs")
            return isoir_labels, band_indices, frequencies
        # only warn when the ISO-IR labeller could not label the q point
        warnings.warn(f"No irreps at {q} in the ISO-IR tables!", stacklevel=2)
        raise ValueError(f"no irrep labels available at {q}")

    # A tabulated special point: the ISO-IR labeller is still the label
    # authority.  It compares in the ISO-IR setting, at the exact translations
    # and with the conjugate phase convention; the direct character overlap
    # below ignores all three and names a physically different irrep at some
    # points (H/K of the hexagonal groups, P of I4/mcm, Y/T of Ccce, ...).  It
    # is kept as a fallback for band sets the labeller cannot decompose.
    isoir_labels = _get_isoir_band_labels(q, phonon, phonon_irreps)

    irt_little_r = [irt_table.symmetries[i - 1].R for i in irt_irreps[0].characters.keys()]
    phonon_little_r = getattr(phonon_irreps, "_rotations_at_q")
    mapping_to_irt = get_mapping_to_irt(irt_little_r, phonon_little_r, prim_mat)

    band_indices = phonon_irreps.band_indices
    frequencies = getattr(phonon_irreps, "frequencies", None)
    if frequencies is None:
        # phonopy >= 2.21 dropped the public property; fall back to the internal array.
        frequencies = getattr(phonon_irreps, "_freqs")
    phonon_irreps_characters = phonon_irreps.characters

    labels: list[list[str] | None] = []
    for set_index, phonon_irrep_charac in enumerate(phonon_irreps_characters):
        if isoir_labels is not None and isoir_labels[set_index] is not None:
            labels.append(isoir_labels[set_index])
            continue
        found = False
        label = []
        for irt_irrep in irt_irreps:
            irt_irrep_character = np.array(list(irt_irrep.characters.values()))
            overlap = np.dot(
                phonon_irrep_charac[mapping_to_irt],
                np.conjugate(irt_irrep_character),
            ) / irt_irrep.nsym
            if overlap > 0.9:
                label.append(f"{irt_irrep.name}({irt_irrep.dim})")
                found = True
        labels.append(label if found else None)

    assert len(labels) == len(band_indices)
    return labels, band_indices, frequencies


def _path_points_renamed_by_frame(phonon, sgnum, cell, coords) -> dict[str, str]:
    """``{seekpath name: ISO-IR name}`` of the seekpath points to which the
    frame of the labels gives another ISO-IR type than spglib's frame of
    ``cell``, in which seekpath names its points; empty when the two frames
    agree (an input in spglib's setting) or a labeller is unavailable."""
    from .isoir import IsoIRLabeler

    try:
        labeler = _isoir_labeler(phonon)
        spglib_frame = IsoIRLabeler(
            sgnum, cell=cell, symprec=phonon.primitive_symmetry.tolerance
        )
    except Exception:
        return {}
    if labeler is None:
        return {}
    renamed = {}
    for name, point in coords.items():
        # exact fractions for the lookups, lattice-dependent values as they are
        k = [parse_qpoint_token(float(v)) for v in point]
        try:
            wanted = labeler.kpoint_name(k)
            if wanted is not None and wanted != spglib_frame.kpoint_name(k):
                renamed[name] = wanted
        except Exception:
            continue
    return renamed


def _seekpath_path_midpoints(phonon) -> tuple[str | None, list[tuple[str, str, list[float]]]]:
    """Midpoints of the seekpath k-path segments, labeled via ISO-IR.

    For every segment of the automatic seekpath k-path (e.g. GM-X of Pm-3m)
    the midpoint of the two endpoint coordinates is computed (a point on the
    connecting symmetry line, e.g. DT (0, 1/4, 0)) and labeled with its
    ISO-IR k-vector-type letter.  Returns (path string, [(label, segment,
    midpoint), ...]); midpoints are skipped with a warning when the seekpath
    primitive cell does not match the phonopy primitive cell.

    The endpoints keep seekpath's names, except that an endpoint to which
    the frame of the labels gives another ISO-IR type than spglib's frame,
    in which seekpath names its points (an input outside the ISO-IR
    setting), is shown with its ISO-IR name in the frame of the labels:
    seekpath's P of a shifted I-4 cell is the PA of the labels.
    """
    import seekpath

    primitive = phonon.primitive
    cell = (primitive.cell, primitive.scaled_positions, primitive.numbers)
    path_data = seekpath.get_path(cell, symprec=1e-5)
    if not np.allclose(primitive.cell, path_data["primitive_lattice"], atol=1e-4):
        warnings.warn(
            "The seekpath primitive cell does not match the phonopy primitive "
            "cell; k-path midpoints are skipped.",
            stacklevel=2,
        )
        return None, []

    coords = path_data["point_coords"]
    segments = path_data["path"]
    dataset = get_symmetry_dataset(phonon.primitive_symmetry)
    renamed = _path_points_renamed_by_frame(phonon, dataset["number"], cell, coords)

    def display(name: str) -> str:
        if name in renamed:
            return renamed[name]
        return "GM" if name == "GAMMA" else name

    # compress consecutive segments into a path string like GM-X-M-GM-R-X | R-M
    parts: list[list[str]] = []
    for start, end in segments:
        if parts and parts[-1][-1] == start:
            parts[-1].append(end)
        else:
            parts.append([start, end])
    path_string = " | ".join("-".join(display(n) for n in part) for part in parts)
    legs = [(display(start), display(end), coords[start], coords[end]) for start, end in segments]
    return path_string, _label_segment_midpoints(phonon, legs)


def _label_segment_midpoints(phonon, legs) -> list[tuple[str, str, list[float]]]:
    """Midpoints of the legs of a k path, labeled via ISO-IR.

    Args:
        phonon: The ``phonopy.Phonopy`` object (its primitive cell and symmetry).
        legs: ``[(start_name, end_name, k_start, k_end), ...]``, the
            coordinates in the primitive reciprocal basis of ``phonon``.

    Returns:
        ``[(label, segment, midpoint), ...]`` in the order of the legs: the
        midpoint snapped to simple fractions (:func:`snap_qpoint`), its
        ISO-IR k-vector-type letter (``q_<coords>`` when it has none) and the
        segment ``"start-end"``; a midpoint equal to an earlier one is left
        out.  :func:`_seekpath_path_midpoints` (``--all-irreps``) and the
        segment midpoints of ``crystod-phonon --visualize --all-irreps`` use it.
    """
    from .isoir import get_isoir_kpoint_name

    primitive = phonon.primitive
    cell = (primitive.cell, primitive.scaled_positions, primitive.numbers)
    number = get_symmetry_dataset(phonon.primitive_symmetry)["number"]
    midpoints: list[tuple[str, str, list[float]]] = []
    seen: set[tuple[float, ...]] = set()
    for start, end, k_start, k_end in legs:
        midpoint = snap_qpoint(
            (np.asarray(k_start, dtype=float) + np.asarray(k_end, dtype=float)) / 2.0
        )
        key = tuple(np.round(midpoint, 8))
        if key in seen:
            continue
        seen.add(key)
        label = get_isoir_kpoint_name(
            number, cell, phonon.primitive_symmetry.tolerance, midpoint,
            input_cell=_input_cell(phonon),
        )
        if label is None:
            label = "q" + "".join(f"_{value:g}" for value in midpoint)
        midpoints.append((label, f"{start}-{end}", list(midpoint)))
    return midpoints


def _is_gamma_point(q) -> bool:
    return bool(np.allclose(q, np.rint(np.asarray(q, dtype=float)), atol=1e-8))


def _yaml_vector(values) -> str:
    # + 0.0 turns a rounded -0.0 into 0.0
    return "[" + ", ".join(f"{round(float(value), 6) + 0.0:.6f}" for value in values) + "]"


def _yaml_matrix(rows) -> str:
    """Nested one-line list of a matrix (six decimals), for the yaml."""
    return "[" + ", ".join(_yaml_vector(row) for row in np.asarray(rows)) + "]"


def _yaml_qpoint(q) -> str:
    """A q point as a one-line yaml list of floats (:func:`format_qpoint`);
    a value Python would print in exponent form gets six decimals, since
    YAML 1.1 reads ``1e-06`` as a string."""
    parts = []
    for value in format_qpoint(q):
        text = repr(value)
        parts.append(f"{value:.6f}" if "e" in text or "E" in text else text)
    return "[" + ", ".join(parts) + "]"


def _yaml_text(text) -> str:
    """A string as a yaml scalar: plain when ``yaml.safe_load`` reads it back
    as the same string (``GM``, ``P4mm``, ``A1 + E``), else double-quoted
    (a JSON string is a valid YAML double-quoted scalar)."""
    import json

    import yaml

    text = str(text)
    try:
        if text and yaml.safe_load(f"key: {text}") == {"key": text}:
            return text
    except yaml.YAMLError:
        pass
    return json.dumps(text)


def _yaml_flow_texts(items, *, quote: bool = False) -> str:
    """A one-line yaml list of strings: single-quoted with ``quote``
    (``['GM4-(3)']``), else plain where a flow sequence allows it
    (``[IR, Raman]``)."""
    import json

    import yaml

    items = [str(item) for item in items]
    if quote:
        return "[" + ", ".join("'" + item.replace("'", "''") + "'" for item in items) + "]"
    plain = "[" + ", ".join(items) + "]"
    try:
        if yaml.safe_load(f"key: {plain}") == {"key": items}:
            return plain
    except yaml.YAMLError:
        pass
    return "[" + ", ".join(json.dumps(item) for item in items) + "]"


def _write_band_sets(fp, band_indices, freqs, labels, *, mulliken=None,
                     activities=None, dielectric=None) -> None:
    """The ``band_sets:`` list of one ``irreps`` entry: one mapping per
    degenerate set, band indices 1-based."""
    if not len(band_indices):
        fp.write("  band_sets: []\n")
        return
    fp.write("  band_sets:\n")
    for i, index in enumerate(band_indices):
        bands = ", ".join(str(int(idx) + 1) for idx in index)
        fp.write(f"  - band_indices: [{bands}]\n")
        fp.write(f"    irrep_label: {_yaml_flow_texts(labels[i], quote=True)}\n")
        symbols = _set_mulliken(labels[i], mulliken) if mulliken else None
        if symbols:
            fp.write(f"    mulliken: {_yaml_text(symbols)}\n")
        fp.write("    frequency: %14.10f\n" % (freqs[index[0]]))
        if activities is not None:
            fp.write(f"    activity: {_yaml_flow_texts(activities[i].activity)}\n")
        if dielectric is not None:
            item = dielectric.sets[i]
            fp.write(f"    mode_effective_charge: {_yaml_matrix(item.charges)}\n")
            fp.write(
                "    dielectric_contribution: "
                f"{_yaml_vector(np.diag(item.contribution))}\n"
            )


def read_phonon_irreps(path) -> dict:
    """Read a ``phonon_irreps.yaml`` or ``phonon_irreps_all.yaml`` file.

    The file is the one ``crystod-phonon --irreps`` (``--all-irreps``) writes;
    this is ``yaml.safe_load`` of it, checked for the ``irreps`` list, so any
    YAML library reads the same data.

    Args:
        path: Path of the file.

    Returns:
        dict: The file as a plain dictionary, with these keys:

        - ``space_group`` (str): Hermann-Mauguin symbol of the space group.
        - ``nac`` (bool): whether the NAC correction (``--nac``) was applied.
        - ``dielectric_electronic``, ``dielectric_static`` (3x3 nested lists;
          only with ``--nac``): eps_inf and eps_0.
        - ``special_points``: list of ``{"q_label": str, "q_position":
          [float, float, float]}`` (primitive reciprocal basis).
        - ``k_path`` (str) and ``path_midpoints`` (list of ``{"q_label",
          "segment", "q_position"}``; only in ``phonon_irreps_all.yaml``):
          the seekpath k-path and the midpoints of its segments.
        - ``irreps``: list with one entry per q point, ``{"q_label",
          "q_position", "band_sets"}`` plus ``"segment"`` at a k-path
          midpoint and ``"activity_summary"`` (str) at Gamma.
          ``band_sets`` lists the degenerate sets, each a dictionary with
          ``band_indices`` (1-based ints), ``irrep_label`` (list of ISO-IR
          labels with their dimension, ``"GM4-(3)"``; a set holding several
          irreps lists them all), ``frequency`` (THz, negative for an
          imaginary mode) and, at Gamma, ``mulliken`` (str, ``"A1 + E"``
          for a set holding several irreps), ``activity`` (list of
          ``"IR"``, ``"Raman"``, ``"silent"``, ``"acoustic"``) and, with
          ``--nac``, ``mode_effective_charge`` (3x3) and
          ``dielectric_contribution`` (diagonal, 3 floats). ``band_sets`` is
          empty at a k-path midpoint whose labeling failed.

    Raises:
        FileNotFoundError: The file does not exist.
        ValueError: The file is not YAML, has no ``irreps`` list, or has an
            ``irreps`` entry without a ``band_sets`` list; a file whose band
            sets sit directly under ``q_label`` is not YAML and is rejected
            here (write it again with ``crystod-phonon --irreps``).

    Example::

        from crystod import phonon

        data = phonon.read_phonon_irreps("phonon_irreps.yaml")
        for point in data["irreps"]:
            for band_set in point["band_sets"]:
                print(point["q_label"], band_set["band_indices"],
                      band_set["irrep_label"], band_set["frequency"])
    """
    import yaml

    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(
            f"{path} is not a YAML file; write it again with crystod-phonon --irreps "
            f"({str(exc).splitlines()[0]})"
        ) from None
    if not isinstance(data, dict) or not isinstance(data.get("irreps"), list):
        raise ValueError(f"{path} has no 'irreps' list; it is not a phonon_irreps.yaml")
    for point in data["irreps"]:
        if not isinstance(point, dict) or not isinstance(point.get("band_sets"), list):
            raise ValueError(
                f"{path}: an 'irreps' entry has no 'band_sets' list; write the file "
                "again with crystod-phonon --irreps"
            )
    return data


def _gamma_activities(phonon, q, labels):
    """The activity of every degenerate set when ``q`` is Gamma (the irreps
    of ``phonon`` were just set there by :func:`get_irrep_labels`), else
    None."""
    from .phonon_activity import activities_from_phonopy_irreps

    if not np.allclose(q, np.rint(np.asarray(q, dtype=float)), atol=1e-8):
        return None
    return activities_from_phonopy_irreps(phonon, labels)


def _gamma_mulliken(phonon) -> dict[str, str]:
    """The Mulliken symbols of the Gamma irreps of the phonopy object
    (``crystod.phonon_activity.mulliken_symbols``), or {} when they cannot
    be determined (the symbols are an annotation; a warning says why)."""
    from .phonon_activity import mulliken_symbols

    try:
        return mulliken_symbols(phonon)
    except Exception as exc:
        warnings.warn(f"no Mulliken symbols: {exc}", stacklevel=2)
        return {}


def _set_mulliken(set_labels, mulliken) -> str | None:
    """The ``mulliken:`` value of one degenerate set (``"A1"``, or
    ``"A1 + E"`` for a set holding several irreps), or None when a label of
    the set has no symbol."""
    if not set_labels or not mulliken:
        return None
    symbols = [mulliken.get(re.sub(r"\(\d+\)$", "", label)) for label in set_labels]
    if any(symbol is None for symbol in symbols):
        return None
    return " + ".join(symbols)


def _check_path_options(args) -> None:
    """Stop on a malformed ``--band``/``--band-labels``/``--npoints`` before
    the survey: they shape the dispersion figure, whose own failures are only
    reported (the YAML is the result of the run)."""
    if args.npoints is not None and args.npoints < 2:
        raise SystemExit("ERROR: --npoints must be at least 2.")
    if args.band_labels and not args.band:
        raise SystemExit("ERROR: --band-labels requires --band.")
    if not args.band:
        return
    from .brillouin_zone import parse_manual_band

    try:
        segments = parse_manual_band(args.band)
    except ValueError as exc:
        raise SystemExit(f"ERROR: --band: {exc}") from None
    if args.band_labels:
        expected = sum(len(segment) for segment in segments)
        given = len(args.band_labels.split())
        if given != expected:
            raise SystemExit(
                f"ERROR: --band-labels needs {expected} labels for this --band path, "
                f"but {given} were given."
            )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _check_path_options(args)

    supercell_mat = [float(n) for n in args.dim.split()]
    if args.readfc:
        force_stes = None
        force_constans = "./FORCE_CONSTANTS"
    else:
        force_stes = "./FORCE_SETS"
        force_constans = None

    phonon = load(
        supercell_matrix=supercell_mat,
        primitive_matrix="auto",
        unitcell=read_poscar_cell(args.poscar),
        force_sets_filename=force_stes,
        force_constants_filename=force_constans,
        **nac_load_options(args.nac),
    )
    nac_note = check_nac_loaded(phonon, args.nac, "BORN")
    if nac_note:
        print(nac_note)

    dataset = get_symmetry_dataset(phonon.symmetry)
    irt_table = IrrepTable(dataset["number"], spinor=False)
    prim_mat = get_primitive_matrix_by_centring(dataset["international"][0])

    q_names, q_list = _special_points_in_label_frame(phonon, irt_table, prim_mat)
    # Gamma first: its activities (and, with --nac, the dielectric response)
    # feed the yaml header and the terminal report
    gamma = None
    for q in q_list:
        if _is_gamma_point(q):
            labels, band_indices, freqs = get_irrep_labels(
                q=q, phonon=phonon, irt_table=irt_table, prim_mat=prim_mat,
                degeneracy_tolerance=args.tol,
            )
            activities = _gamma_activities(phonon, q, labels)
            dielectric = None
            if args.nac and activities is not None:
                from .phonon_activity import dielectric_response

                dielectric = dielectric_response(phonon, activities)
            gamma = (labels, band_indices, freqs, activities, dielectric)
            break
    mulliken = _gamma_mulliken(phonon) if gamma is not None else {}
    path_string, path_midpoints = None, []
    if args.all_irreps:
        try:
            path_string, path_midpoints = _seekpath_path_midpoints(phonon)
        except Exception:
            pass
    yaml_name = args.output or (
        "phonon_irreps_all.yaml" if args.all_irreps else "phonon_irreps.yaml"
    )
    gamma_summary = None
    terminal_summary = None
    with open(yaml_name, "w") as fp:
        fp.write(f"space_group: {_yaml_text(dataset['international'])}\n")
        fp.write(f"nac: {'true' if args.nac else 'false'}\n")
        dielectric = gamma[4] if gamma is not None else None
        if dielectric is not None:
            fp.write(f"dielectric_electronic: {_yaml_matrix(dielectric.eps_inf)}\n")
            fp.write(f"dielectric_static: {_yaml_matrix(dielectric.eps_static)}\n")
        fp.write("special_points:\n" if q_names else "special_points: []\n")
        for qname, q in zip(q_names, q_list):
            fp.write(f"- q_label: {_yaml_text(qname)}\n")
            fp.write(f"  q_position: {_yaml_qpoint(q)}\n")
        fp.write("\n")
        if path_midpoints:
            fp.write(f"k_path: {_yaml_text(path_string)}  # seekpath\n")
            fp.write("path_midpoints:  # midpoints of the k-path segments, ISO-IR k-vector types\n")
            for label, segment, midpoint in path_midpoints:
                fp.write(f"- q_label: {_yaml_text(label)}\n")
                fp.write(f"  segment: {_yaml_text(segment)}\n")
                fp.write(f"  q_position: {_yaml_qpoint(midpoint)}\n")
            fp.write("\n")
        fp.write("irreps:\n" if q_names or path_midpoints else "irreps: []\n")
        for qname, q in zip(q_names, q_list):
            fp.write(f"- q_label: {_yaml_text(qname)}\n")
            fp.write(f"  q_position: {_yaml_qpoint(q)}\n")
            dielectric = None
            if gamma is not None and _is_gamma_point(q):
                labels, band_indices, freqs, activities, dielectric = gamma
            else:
                labels, band_indices, freqs = get_irrep_labels(
                    q=q,
                    phonon=phonon,
                    irt_table=irt_table,
                    prim_mat=prim_mat,
                    degeneracy_tolerance=args.tol,
                )
                activities = _gamma_activities(phonon, q, labels)
            if activities is not None:
                gamma_summary = format_activity_summary(activities)
                terminal_summary = _activity_summary_parts(activities, False, mulliken)
                fp.write(f"  activity_summary: {_yaml_text(gamma_summary)}\n")
            _write_band_sets(
                fp, band_indices, freqs, labels,
                mulliken=mulliken if _is_gamma_point(q) else None,
                activities=activities, dielectric=dielectric,
            )
            fp.write("\n")
        for label, segment, midpoint in path_midpoints:
            fp.write(f"- q_label: {_yaml_text(label)}\n")
            fp.write(f"  segment: {_yaml_text(segment)}\n")
            fp.write(f"  q_position: {_yaml_qpoint(midpoint)}\n")
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="No irreps at")
                    labels_mid, band_indices_mid, freqs_mid = get_irrep_labels(
                        q=midpoint,
                        phonon=phonon,
                        irt_table=irt_table,
                        prim_mat=prim_mat,
                        degeneracy_tolerance=args.tol,
                    )
            except Exception:
                fp.write("  band_sets: []  # irrep labeling failed at this q point\n\n")
                continue
            _write_band_sets(fp, band_indices_mid, freqs_mid, labels_mid)
            fp.write("\n")
    # the dispersion with these labels, next to the YAML; the YAML is the
    # result of the run, so a failure of the figure is reported, not raised
    import os

    figure_name = os.path.splitext(yaml_name)[0] + ".pdf"
    dispersion, figure_error = None, None
    try:
        from .phonon_irreps_plot import plot_irreps_dispersion

        dispersion = plot_irreps_dispersion(
            phonon, irt_table, prim_mat, figure_name,
            space_group=dataset["international"], nac=args.nac, threshold=args.threshold,
            degeneracy_tolerance=args.tol, band=args.band, band_labels=args.band_labels,
            npoints=args.npoints,
        )
    except Exception as exc:
        figure_error = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
    # terminal report: titled blocks, the written file last
    if terminal_summary is not None:
        print("\n* Gamma-point activity *")
        for part in terminal_summary:
            print(part)
    if gamma is not None and gamma[4] is not None:
        from .phonon_activity import format_dielectric_table

        _print_titled_lines(format_dielectric_table(gamma[4]))
    if args.raman_tensor:
        from .phonon_activity import format_raman_tensors, gamma_raman_tensors

        _print_titled_lines(format_raman_tensors(gamma_raman_tensors(phonon), mulliken=mulliken))
    print("\n* Band path *")
    if dispersion is not None:
        source = "--band" if args.band else "seekpath"
        print(f"  k path ({source}): {dispersion.path_string}  ({dispersion.n_qpoints} q points)")
        print("  Labeled points: " + ", ".join(point.name for point in dispersion.points))
        for note in dispersion.notes:
            print(f"  NOTE: {note}")
    else:
        print(f"  NOTE: the dispersion figure was not drawn ({figure_error})")
    print("\n* Output files *")
    print(f"  Phonon irreps written to: {yaml_name}")
    if dispersion is not None:
        print(f"  Phonon dispersion with irrep labels written to: {figure_name}")


def _print_titled_lines(lines: list[str]) -> None:
    """Print a ``format_*`` block whose first line is its title (``"Title:"``)
    as a ``* Title *`` block: a blank line, the header, then the other lines."""
    print(f"\n* {lines[0].rstrip(':')} *")
    for line in lines[1:]:
        print(line)


if __name__ == "__main__":
    main()
