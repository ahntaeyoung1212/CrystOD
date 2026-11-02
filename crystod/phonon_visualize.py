"""
Phonon mode viewer: ``crystod-phonon --visualize``.

The phonon counterpart of the SALC viewer of ``crystod --visualize``.  From
POSCAR + FORCE_SETS (or FORCE_CONSTANTS) it

1. runs phonopy along the k path (the seekpath path of the primitive cell, or
   ``--band``) with eigenvectors and writes phonopy's ``band.yaml`` under the
   name ``band_vector.yaml``, the vertices named by their ISO-IR k-vector types
   (``GM``, ``X``, ``M``, ``R``) as the irreps figure ticks them;
2. writes one standalone HTML page (``phonon_visualize.html``): the dispersion
   along the path with a marker on every degenerate level of the labeled
   special points, and a 3D scene with the atomic displacement pattern of the
   selected mode, every mode of every special point listed with its ISO-IR
   label (and, at Gamma, its Mulliken symbol and IR/Raman activity), arrows
   and an optional animation of the vibration.  With ``--all-irreps``
   (``phonon_visualize_all.html``) the page also shows the midpoint of every
   segment of the path (the symmetry lines DT, Z, SM, ... of
   ``crystod-phonon --irreps --all-irreps``), each with its line irreps and
   its own display cell.

The bands, the path and the labeled points are those of the irreps figure
(:func:`crystod.phonon_irreps_plot.compute_dispersion`), the labels those of
``phonon_irreps.yaml`` (:func:`crystod.phonon_irreps.get_irrep_labels`), the
mode vectors the symmetry-adapted eigenvectors of ``crystod-phonon --vector``
(:func:`crystod.phonon_vector.build_symmetry_adapted_modes`: the partners of
a degenerate level transform with the irrep matrices).

Conventions: frequencies in THz, imaginary ones negative (phonopy); q in the
primitive reciprocal basis of phonopy's primitive cell.  The displacement of
primitive atom ``a`` in the cell at lattice translation ``T`` (primitive
basis) is drawn as ``Re[u_a exp(i(2 pi q.T - phi))]`` with the atomic-gauge
vector ``u_a = e_a exp(2 pi i q.tau_a) / sqrt(m_a)`` (``e_a`` the
mass-weighted phonopy eigenvector, ``tau_a`` the fractional position of the
atom), the global phase of :func:`crystod.phonon_vector.get_supercell_displacement_field`
(maximal real part, phonopy's MODULATION choice) and normalized so that the
largest displacement in the display cell at ``phi = 0`` is 1 (the amplitude
slider gives it in Angstrom); ``phi`` is the animation phase.

The page layout (left control/mode sidebar + central viewport) is modeled
after the phonon website by Henrique Miranda
(https://henriquemiranda.github.io/phononwebsite/, BSD-3-Clause); the
rendering uses plotly; atom colors are VESTA's.
"""

from __future__ import annotations

import json
import re
import warnings
from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    RawDescriptionHelpFormatter,
    RawTextHelpFormatter,
)
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray

from .spglib_compat import ensure_spglib_compat

ensure_spglib_compat()

#: Default file names: the band structure keeps its fixed name, ``-o`` renames
#: the page only.
BAND_VECTOR_YAML = "band_vector.yaml"
DEFAULT_OUTPUT = "phonon_visualize.html"
#: The page of ``--all-irreps`` (the pair of ``phonon_irreps_all.yaml``).
DEFAULT_OUTPUT_ALL = "phonon_visualize_all.html"
#: Default largest displacement of a pattern on the page: this fraction of the
#: nearest-neighbor distance, rounded to 0.05 A (0.5 A for ScF3 and SrTiO3).
AMPLITUDE_FRACTION = 0.25
#: Length of an arrow head as a fraction of the arrow (plotly cone, ``raw`` size).
HEAD_FRACTION = 0.5
#: Band line and imaginary-level colors of the irreps figure.
CURVE = "#184f95"
IMAGINARY = "#e34948"
#: Initial camera of the 3D scene: off the body diagonal, so that the atoms of
#: a cubic cell do not hide each other, and closer than the SALC viewer's.
CAMERA_EYE = {"x": 1.55, "y": 1.0, "z": 0.8}
#: Arrow color of the displacement pattern: near-black, which no VESTA atom
#: color is close to (red arrows vanish on the red O atoms of oxides).
ARROW = "#222222"
#: Displacements below this fraction of the largest one get no arrow.
ARROW_CUTOFF = 0.02
#: A display cell with more atoms than this gets a NOTE (the page may be slow);
#: a segment midpoint of ``--all-irreps`` whose cell would be larger is skipped.
LARGE_CELL_ATOMS = 1000
#: Largest denominator of the q coordinates read as exact fractions (that of
#: :func:`crystod.operations.snap_qpoint`): the display cell and the fractions
#: of the page and the terminal.
Q_DENOMINATOR = 48
#: Decimals of the shipped data: positions (A), mode vectors, bands.
_POSITION_DECIMALS = 4
_VECTOR_DECIMALS = 7
_BAND_DECIMALS = 4
_DISTANCE_DECIMALS = 5
_GREEK = {"Gamma": "Γ", "Delta": "Δ", "Sigma": "Σ", "Lambda": "Λ"}
_GAMMA_ALIASES = {"G", "GM", "GAMMA", "Γ"}


class MyHelpFormatter(
    RawTextHelpFormatter,
    RawDescriptionHelpFormatter,
    ArgumentDefaultsHelpFormatter,
):
    pass


desc = """
Phonon mode viewer: one standalone HTML page with the dispersion along the k
path and the displacement pattern of every mode at its special points, each
labeled with its ISO-IR irrep (Mulliken symbol and IR/Raman activity at
Gamma); also writes phonopy's band.yaml with eigenvectors as band_vector.yaml.
POSCAR and FORCE_SETS (or FORCE_CONSTANTS with --readfc) must exist in the
directory where this code runs, as in --irreps mode.

# Command Examples:
crystod-phonon --visualize --dim "4 4 4" -c 221_PPOSCAR_ScF3
crystod-phonon --visualize --dim "4 4 4" -c 221_PPOSCAR_ScF3 --bond Sc F 2.3
crystod-phonon --visualize --dim "4 4 4" -c 221_PPOSCAR_ScF3 --qpoint R -o ScF3_R.html
crystod-phonon --visualize --dim "4 4 4" -c 227_PPOSCAR_Si --readfc --conventional
crystod-phonon --visualize --dim "4 4 4" -c 221_PPOSCAR_SrTiO3 --bond Ti O 2.1 --all-irreps

The page opens on the first q point and its first mode; the URL hash
#q=R&mode=4&view=a opens the q point R with band 4 selected, viewed down the
a axis (orthographic; view=b, view=c likewise). With --all-irreps the page
(phonon_visualize_all.html) also shows the midpoints of the path segments, as
crystod-phonon --irreps --all-irreps labels them: #q=DT opens the first
midpoint named DT, #q=DT(GM-X) the midpoint of GM-X, #q=7 the seventh q button.
"""


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


@dataclass
class ViewerMode:
    """One phonon mode (band) at a q point of the viewer.

    Attributes:
        band: 1-based band index (frequency order).
        frequency: Frequency in THz (negative: imaginary).
        irrep: Text of the degenerate set the band belongs to, as
            ``phonon_irreps.yaml`` labels it (``GM4-(3)``; ``X3-`` for a
            one-dimensional irrep; ``GM1 + GM5(2)`` for a set holding several
            irreps; ``?`` when unlabeled).
        component: 1..dimension inside the degenerate set.
        dimension: Number of bands of the set.
        set_index: 0-based index of the set at this q point.
        mulliken: Mulliken symbol of the set at Gamma (``T1u``), else None.
        activity: Activity of the set at Gamma (``IR``, ``Raman``,
            ``IR+Raman``, ``silent``, ``acoustic``), else None.
        u: Atomic-gauge complex displacement vectors, shape
            ``(n_primitive_atoms, 3)``, normalized so that the largest
            displacement of the display cell at phase 0 is 1.
        scale: The factor removed by that normalization (the largest real
            displacement of the field of
            :func:`crystod.phonon_vector.get_supercell_displacement_field`).
        vector: The mass-weighted mode vector the pattern comes from (phonopy
            convention, ``3 * n_atoms`` complex components).
    """

    band: int
    frequency: float
    irrep: str
    component: int
    dimension: int
    set_index: int
    mulliken: str | None
    activity: str | None
    u: NDArray[np.complex128]
    scale: float
    vector: NDArray[np.complex128] = field(repr=False)


@dataclass
class ViewerPoint:
    """One q point of the viewer: its display cell, atoms, bonds and modes.

    Attributes:
        name: ISO-IR name of the point (``GM``, ``X``; ``U|K`` for the two
            sides of a break that carry the same levels; ``DT`` for a segment
            midpoint, its ISO-IR k-vector type).
        q: Fractional coordinates, primitive reciprocal basis.
        x: Position on the dispersion (path distance; the middle of its leg
            for a segment midpoint), or None when the point is not on the
            path (``--qpoint`` elsewhere).
        cell_matrix: Rows are the display-cell vectors in the primitive basis
            (commensurate with q: each axis times the denominator of q,
            :func:`_display_cell_matrix`).
        cell_description: ``primitive, 2 x 2 x 2 cells`` (as the SALC viewer).
        lattice: The display-cell lattice vectors as rows (Angstrom).
        symbols, positions, atom_index, translations: The drawn atoms
            (boundary completion and bonded images included): element,
            Cartesian position (A), primitive atom index ``a`` and
            primitive-basis lattice translation ``T`` (ints).
        bonds: ``[(name, [(slot_i, slot_j), ...]), ...]`` per ``--bond``.
        polyhedra: ``[(element, center_slot, [(slot_i, slot_j, slot_k), ...]),
            ...]``: the triangles of the convex hull of each coordination
            polyhedron at equilibrium (kept while the atoms move).
        modes: The :class:`ViewerMode` list, in band order.
        notes: Notes of this point (fallbacks).
        segment: The segment of the path whose midpoint this point is
            (``GM-X``; ``--all-irreps``), None at a special point.
    """

    name: str
    q: list
    x: float | None
    cell_matrix: NDArray[np.int_]
    cell_description: str
    lattice: NDArray[np.float64]
    symbols: list
    positions: NDArray[np.float64]
    atom_index: list
    translations: NDArray[np.int_]
    bonds: list
    polyhedra: list
    modes: list
    notes: list = field(default_factory=list)
    segment: str | None = None

    @property
    def is_gamma(self) -> bool:
        return bool(np.allclose(self.q, np.rint(np.asarray(self.q, dtype=float)), atol=1e-8))

    @property
    def is_midpoint(self) -> bool:
        """Whether the point is a segment midpoint of ``--all-irreps``."""
        return self.segment is not None


@dataclass
class PhononViewer:
    """Everything the phonon mode viewer page shows.

    Attributes:
        formula: Reduced formula (``ScF3``).
        space_group: Hermann-Mauguin symbol; ``number`` its number.
        supercell: The force-calculation supercell (``4 x 4 x 4``).
        force_data: ``FORCE_SETS`` or ``FORCE_CONSTANTS`` (or None).
        nac: Whether the NAC correction is applied.
        path_source: ``seekpath`` or ``--band``.
        dispersion: The :class:`crystod.phonon_irreps_plot.Dispersion` of the path.
        points: The :class:`ViewerPoint` list, in the order of the selector.
        threshold: Frequency (THz) below which a level counts as imaginary.
        bonds: The ``(el1, el2, max_length)`` bond specifications.
        primitive_lattice: Rows, Angstrom; ``conventional_lattice`` the
            conventional cell with ``--conventional`` (else None).
        amplitude: Default largest displacement on the page (Angstrom,
            :func:`default_amplitude`).
        notes: Notes of the run (path, NAC, segment midpoints not drawn).
        all_irreps: Whether the segment midpoints are shown too
            (``--all-irreps``; they follow the special points in ``points``).
    """

    formula: str
    space_group: str
    number: int
    supercell: str
    force_data: str | None
    nac: bool
    path_source: str
    dispersion: object
    points: list
    threshold: float
    bonds: list
    primitive_lattice: NDArray[np.float64]
    conventional_lattice: NDArray[np.float64] | None
    amplitude: float = 0.5
    notes: list = field(default_factory=list)
    all_irreps: bool = False


def default_amplitude(phonon) -> float:
    """The page's default largest displacement: :data:`AMPLITUDE_FRACTION`
    of the nearest-neighbor distance of the primitive cell, rounded to
    0.05 A and kept within 0.2 to 1.0 A (0.5 A for ScF3 and SrTiO3, whose
    shortest bonds are about 2 A)."""
    primitive = phonon.primitive
    lattice = np.array(primitive.cell, dtype=float)
    cart = np.array(primitive.scaled_positions, dtype=float) @ lattice
    offsets = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
                       dtype=float) @ lattice
    images = (cart[None, :, None, :] + offsets[None, None, :, :]) - cart[:, None, None, :]
    distances = np.linalg.norm(images, axis=-1).ravel()
    distances = distances[distances > 1e-3]
    nearest = float(distances.min()) if distances.size else 2.0
    return float(min(1.0, max(0.2, round(AMPLITUDE_FRACTION * nearest / 0.05) * 0.05)))


# --------------------------------------------------------------------------
# text helpers
# --------------------------------------------------------------------------


def _fraction(value: float) -> str:
    """``1/2``, ``-1/3``, ``3/16``, ``0`` (denominators up to
    :data:`Q_DENOMINATOR`), else ``%g``."""
    frac = Fraction(float(value)).limit_denominator(Q_DENOMINATOR)
    if abs(float(frac) - float(value)) > 1e-6:
        return f"{float(value):g}"
    if frac.denominator == 1:
        return str(frac.numerator)
    return f"{frac.numerator}/{frac.denominator}"


def format_q_fraction(q) -> str:
    """``(0, 1/2, 0)``: q as fractions, the form of the viewer's q selector."""
    return "(" + ", ".join(_fraction(v) for v in np.asarray(q, dtype=float) + 0.0) + ")"


def _tick_html(text: str) -> str:
    """The HTML of a tick label of the irreps figure (mathtext ``$\\Gamma$``,
    ``X$_{1}$``) for the plotly chart (Greek capitals, ``<sub>``)."""
    text = re.sub(r"\$\\(Gamma|Delta|Sigma|Lambda)\$", lambda m: _GREEK[m.group(1)], text)
    text = re.sub(r"\$_\{(.*?)\}\$", r"<sub>\1</sub>", text)
    return text.replace("$", "")


def _supercell_text(phonon) -> str:
    """``4 x 4 x 4`` for a diagonal supercell matrix, else its rows."""
    matrix = np.rint(np.asarray(phonon.supercell_matrix)).astype(int)
    if np.count_nonzero(matrix - np.diag(np.diag(matrix))) == 0:
        return " x ".join(str(int(v)) for v in np.diag(matrix))
    return " / ".join(" ".join(str(int(v)) for v in row) for row in matrix)


def _clean(value: float, decimals: int) -> float:
    """``round(value, decimals)`` without a negative zero."""
    return round(float(value), decimals) + 0.0


# --------------------------------------------------------------------------
# band path and band_vector.yaml
# --------------------------------------------------------------------------


def _band_path(phonon, band, band_labels, npoints, tolerance):
    """``(qpoints, connections, labels, segments, names, notes)`` of the path
    of :func:`crystod.phonon_irreps_plot.compute_dispersion`: the same
    vertices (``_path``) and the same q points per leg, the vertex names
    flattened in phonopy's ``labels`` order (the start of every continuous
    part, then every end point)."""
    from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

    from .phonon_irreps_plot import _path

    segments, names, notes = _path(phonon, band, band_labels, tolerance)
    band_paths = [np.array(segment, dtype=float) for segment in segments]
    qpoints, connections = get_band_qpoints_and_path_connections(
        band_paths, npoints=npoints or 51
    )
    labels = [str(name) for part in names for name in part]
    return qpoints, connections, labels, segments, names, notes


def write_band_vector_yaml(
    phonon,
    *,
    band: str | None = None,
    band_labels: str | None = None,
    npoints: int | None = None,
    tolerance: float = 1e-5,
    filename: str = BAND_VECTOR_YAML,
) -> int:
    """Write phonopy's ``band.yaml`` with eigenvectors along the viewer's path.

    The path and the q points are those of the irreps figure and the viewer
    (:func:`crystod.phonon_irreps_plot.compute_dispersion`); the vertices are
    labeled with their ISO-IR names (``GM``, ``X``) in phonopy's ``labels``
    list.  The file is phonopy's own format (``nqpoint``, ``npath``,
    ``segment_nqpoint``, ``labels``, ``reciprocal_lattice``, the primitive
    cell, then ``phonon:`` with one entry per q point: ``q-position``,
    ``distance`` and ``band:``, each band with its ``frequency`` (THz) and
    ``eigenvector``: one list per atom of three ``[re, im]`` pairs, the
    mass-weighted eigenvector in phonopy's atom-position phase convention),
    so phonopy's and other band.yaml readers read it.

    Args:
        phonon: ``phonopy.Phonopy`` object with force constants.
        band, band_labels, npoints, tolerance: The path, as for
            :func:`crystod.phonon_irreps_plot.compute_dispersion`.
        filename: Output path.

    Returns:
        The number of q points written (``nqpoint``).
    """
    qpoints, connections, labels, _, _, _ = _band_path(
        phonon, band, band_labels, npoints, tolerance
    )
    phonon.run_band_structure(
        qpoints, path_connections=connections, labels=labels,
        with_eigenvectors=True, is_band_connection=False,
    )
    phonon.write_yaml_band_structure(filename=filename)
    return int(sum(len(q) for q in qpoints))


# --------------------------------------------------------------------------
# modes, labels and the display cell of one q point
# --------------------------------------------------------------------------


def _plain_modes(phonon, qpoint):
    """Plain phonopy eigenvectors at q, as ``crystod-phonon --vector`` falls
    back on them: ``[(frequency_THz, vector), ...]``."""
    from .phonon_vector import FREQUENCY_CONVERSION_THZ

    dynamical_matrix = phonon.dynamical_matrix
    dynamical_matrix.run(qpoint)
    eigenvalues, vectors = np.linalg.eigh(dynamical_matrix.dynamical_matrix)
    eigenvalues = eigenvalues.real
    frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * FREQUENCY_CONVERSION_THZ
    return [(float(f), vectors[:, i]) for i, f in enumerate(frequencies)]


def _degenerate_sets(frequencies, tolerance):
    sets: list[list[int]] = []
    for index, frequency in enumerate(frequencies):
        if sets and abs(frequency - frequencies[sets[-1][0]]) < tolerance:
            sets[-1].append(index)
        else:
            sets.append([index])
    return sets


def _set_labels(phonon, qpoint, labelling, degeneracy_tolerance):
    """``(texts, band_sets, mulliken_texts, activity_texts, note)`` at q:
    one entry per degenerate set, labeled at the tabulated arm of the star of
    q (as ``crystod-phonon --vector`` and the irreps figure do); the Gamma
    entries are None elsewhere."""
    from .phonon_irreps import (
        _gamma_activities,
        _set_mulliken,
        find_star_representative,
        get_irrep_labels,
    )
    from .phonon_irreps_plot import _set_text

    irt_table, prim_mat, q_names, q_list, rotations, mulliken = labelling
    label_q = list(qpoint)
    representative = find_star_representative(qpoint, rotations, q_names, q_list)
    if representative is not None:
        label_q = list(representative[1])
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="No irreps at")
            labels, band_indices, frequencies = get_irrep_labels(
                q=label_q, phonon=phonon, irt_table=irt_table, prim_mat=prim_mat,
                degeneracy_tolerance=degeneracy_tolerance,
            )
    except Exception as exc:
        phonon.run_qpoints([qpoint])
        from .runtime_compat import get_qpoints_result

        frequencies = get_qpoints_result(phonon).frequencies[0]
        sets = _degenerate_sets(list(frequencies), degeneracy_tolerance)
        return (["?"] * len(sets), sets, [None] * len(sets), [None] * len(sets),
                f"no ISO-IR labels ({exc}); the sets are shown as '?'")
    band_sets = [[int(i) for i in indices] for indices in band_indices]
    texts = [_set_text(set_labels) for set_labels in labels]
    activities = _gamma_activities(phonon, label_q, labels)
    if activities is None:
        return texts, band_sets, [None] * len(texts), [None] * len(texts), None
    mulliken_texts = [_set_mulliken(set_labels, mulliken) for set_labels in labels]
    activity_texts = ["+".join(record.activity) for record in activities]
    return texts, band_sets, mulliken_texts, activity_texts, None


def _hull_faces(points: NDArray[np.float64]) -> list[tuple[int, int, int]]:
    """Triangles of the convex hull of a coordination polyhedron (indices into
    ``points``); a flat polygon (square-planar coordination) is fanned from
    its first vertex in angular order."""
    from scipy.spatial import ConvexHull, QhullError

    try:
        return [tuple(int(i) for i in simplex) for simplex in ConvexHull(points).simplices]
    except (QhullError, ValueError):
        center = points.mean(axis=0)
        _, _, vt = np.linalg.svd(points - center)
        local = (points - center) @ vt[:2].T
        order = list(np.argsort(np.arctan2(local[:, 1], local[:, 0])))
        return [(int(order[0]), int(order[i]), int(order[i + 1])) for i in range(1, len(order) - 1)]


def _display_atoms(phonon, cell_matrix, bonds):
    """The drawn atoms of one display cell and their bonds and polyhedra.

    The atoms of the SALC viewer (``_display_cell_entries``: boundary
    completion), plus, with ``bonds``, the periodic images that complete the
    coordination polyhedra (``_bond_pairs``, as in VESTA), each with its own
    primitive atom index and lattice translation.

    Returns:
        ``(symbols, positions, atom_index, translations, bond_list,
        polyhedra)``; see :class:`ViewerPoint`.
    """
    from .visualize_basis import _bond_pairs, _display_cell_entries

    primitive = phonon.primitive
    lattice = np.array(primitive.cell, dtype=float)
    frac = np.array(primitive.scaled_positions, dtype=float)
    prim_symbols = list(primitive.symbols)
    cell_matrix = np.asarray(cell_matrix, dtype=int)
    display_lattice = np.array(cell_matrix, dtype=float) @ lattice

    entries = _display_cell_entries(frac, cell_matrix)
    atom_index = [int(a) for a, _ in entries]
    translations = [np.rint(t).astype(int) for _, t in entries]
    positions = [(frac[a] + t) @ lattice for a, t in entries]
    symbols = [prim_symbols[a] for a in atom_index]

    bond_list, polyhedra = [], []
    if bonds:
        base_positions = np.array(positions)
        base_symbols = list(symbols)
        # an atom is (a, T): a bonded image already drawn (a boundary
        # duplicate, or the image of another bond) is not drawn twice
        slots = {(a, tuple(int(v) for v in t)): slot
                 for slot, (a, t) in enumerate(zip(atom_index, translations))}

        def slot_of(b, offset, position):
            if not any(offset):
                return b
            translation = translations[b] + np.asarray(offset, dtype=int) @ cell_matrix
            key = (atom_index[b], tuple(int(v) for v in translation))
            if key not in slots:
                slots[key] = len(positions)
                positions.append(np.array(position, dtype=float))
                atom_index.append(atom_index[b])
                translations.append(translation)
                symbols.append(symbols[b])
            return slots[key]

        for el1, el2, max_length in bonds:
            pairs, centers = [], {}
            for a, b, offset, position_b in _bond_pairs(
                base_positions, base_symbols, display_lattice, el1, el2, max_length
            ):
                slot = slot_of(b, offset, position_b)
                pairs.append((a, slot))
                centers.setdefault(a, []).append(slot)
            if pairs:
                bond_list.append((f"{el1}-{el2} bonds", pairs))
            for center, vertices in centers.items():
                if len(vertices) >= 4:  # a convex hull needs at least four ligands
                    faces = _hull_faces(np.array([positions[v] for v in vertices]))
                    if faces:
                        polyhedra.append(
                            (el1, center, [tuple(vertices[i] for i in face) for face in faces])
                        )
    return (
        symbols, np.array(positions, dtype=float), atom_index,
        np.array(translations, dtype=int).reshape(-1, 3), bond_list, polyhedra,
    )


def _atomic_gauge_vectors(phonon, qpoint, cell_matrix, vector):
    """``(u, scale)``: the atomic-gauge displacement vectors of the primitive
    atoms (``(n_atoms, 3)`` complex) carrying the global phase of
    :func:`crystod.phonon_vector.get_supercell_displacement_field` on the
    display cell, normalized by ``scale``, the largest real displacement of
    that field.  ``u`` is read off the field itself, ``u_a = u(l, a)
    exp(-2 pi i q.T_l)``, and checked to be the same in every cell."""
    from .phonon_vector import get_supercell_displacement_field

    supercell, displacements = get_supercell_displacement_field(
        phonon, qpoint, cell_matrix, vector
    )
    matrix = np.asarray(cell_matrix, dtype=int)
    frac = np.array(phonon.primitive.scaled_positions, dtype=float)
    s2uu = np.array([supercell.u2u_map[x] for x in supercell.s2u_map], dtype=int)
    x_primitive = np.array(supercell.scaled_positions, dtype=float) @ matrix
    translations = np.rint(x_primitive - frac[s2uu])
    back = np.exp(-2j * np.pi * translations @ np.asarray(qpoint, dtype=float))
    per_cell = displacements * back[:, None]
    u = np.zeros((len(frac), 3), dtype=complex)
    for atom in range(len(frac)):
        u[atom] = per_cell[np.flatnonzero(s2uu == atom)[0]]
    spread = float(np.max(np.abs(per_cell - u[s2uu]))) if len(s2uu) else 0.0
    size = float(np.max(np.abs(per_cell))) or 1.0
    if spread > 1e-8 * size:
        raise RuntimeError(
            f"the displacement field is not a Bloch wave on the display cell ({spread:.1e})"
        )
    scale = float(np.max(np.linalg.norm(displacements.real, axis=1)))
    if scale < 1e-12:
        # the real part of the pattern vanishes (cannot happen with the
        # maximal-real-part phase unless the vector is zero)
        scale = float(np.max(np.linalg.norm(np.abs(displacements), axis=1))) or 1.0
    return u / scale, scale


def _display_cell_matrix(qpoint, base_matrix) -> NDArray[np.int_]:
    """The display cell of q: the smallest diagonal multiple of ``base_matrix``
    commensurate with q, from the exact fractions of q.

    :func:`crystod.phonon_vector.get_commensurate_supercell_matrix` reads each
    component as a fraction with a denominator up to 12, which gives a cell
    that is not commensurate with a segment midpoint such as (3/16, 3/16,
    3/8) (the K-GM midpoint of an fcc path); here the components are the
    fractions of :func:`crystod.operations.snap_qpoint` (denominators up to
    :data:`Q_DENOMINATOR`), multiplied into the base-cell reciprocal basis
    exactly.  A q that is no such fraction falls back on
    ``get_commensurate_supercell_matrix``.

    Args:
        qpoint: Fractional coordinates of q in the primitive reciprocal basis.
        base_matrix: Rows are the base-cell vectors in the primitive basis.

    Returns:
        Integer matrix ``S`` (rows: the display-cell vectors in the primitive
        basis) with ``q . S_row`` an integer for every row.
    """
    from .phonon_vector import get_commensurate_supercell_matrix

    fractions = []
    for value in np.asarray(qpoint, dtype=float):
        frac = Fraction(float(value)).limit_denominator(Q_DENOMINATOR)
        if abs(float(frac) - float(value)) > 1e-9:
            return np.array(get_commensurate_supercell_matrix(qpoint, base_matrix), dtype=int)
        fractions.append(frac)
    base = np.rint(np.asarray(base_matrix, dtype=float)).astype(int)
    sizes = [
        sum((int(base[row, col]) * fractions[col] for col in range(3)), Fraction(0)).denominator
        for row in range(3)
    ]
    return np.diag(sizes) @ base


def _cell_atom_count(phonon, cell_matrix) -> int:
    """Atoms of the display cell (primitive cells times primitive atoms),
    without the boundary completion of the drawing."""
    cells = int(round(abs(np.linalg.det(np.asarray(cell_matrix, dtype=float)))))
    return cells * len(phonon.primitive)


def _viewer_point(phonon, name, qpoint, x, labelling, degeneracy_tolerance,
                  base_matrix, conventional_centring, bonds, segment=None) -> ViewerPoint:
    from .phonon_vector import build_symmetry_adapted_modes

    notes: list[str] = []
    qpoint = [float(v) for v in qpoint]
    try:
        modes = build_symmetry_adapted_modes(phonon, qpoint)
    except Exception as exc:
        notes.append(
            f"symmetry-adapted mode construction failed ({exc}); plain phonopy "
            "eigenvectors are drawn, which may look tilted within degenerate subspaces"
        )
        modes = _plain_modes(phonon, qpoint)
    texts, band_sets, mulliken, activity, note = _set_labels(
        phonon, qpoint, labelling, degeneracy_tolerance
    )
    if note:
        notes.append(note)

    cell_matrix = _display_cell_matrix(qpoint, base_matrix)
    if conventional_centring is not None:
        multiples = np.rint(
            np.diag(cell_matrix @ np.linalg.inv(np.array(base_matrix, dtype=float)))
        ).astype(int)
        description = (
            f"conventional ({conventional_centring} centring), "
            f"{multiples[0]} x {multiples[1]} x {multiples[2]} cells"
        )
    else:
        n1, n2, n3 = np.diag(cell_matrix)
        description = f"primitive, {n1} x {n2} x {n3} cells"
    lattice = np.array(cell_matrix, dtype=float) @ np.array(phonon.primitive.cell, dtype=float)
    symbols, positions, atom_index, translations, bond_list, polyhedra = _display_atoms(
        phonon, cell_matrix, bonds
    )
    if len(symbols) > LARGE_CELL_ATOMS:
        notes.append(
            f"the display cell of q = {format_q_fraction(qpoint)} holds {len(symbols)} atoms; "
            "the page may be slow"
        )

    set_of_band = {}
    for set_index, indices in enumerate(band_sets):
        for component, band_index in enumerate(indices):
            set_of_band[band_index] = (set_index, component + 1, len(indices))
    viewer_modes = []
    for band_index, (frequency, vector) in enumerate(modes):
        set_index, component, dimension = set_of_band.get(band_index, (len(band_sets), 1, 1))
        u, scale = _atomic_gauge_vectors(phonon, qpoint, cell_matrix, vector)
        has_set = set_index < len(band_sets)
        viewer_modes.append(
            ViewerMode(
                band=band_index + 1,
                frequency=float(frequency),
                irrep=texts[set_index] if has_set else "?",
                component=component,
                dimension=dimension,
                set_index=set_index,
                mulliken=mulliken[set_index] if has_set else None,
                activity=activity[set_index] if has_set else None,
                u=u,
                scale=scale,
                vector=np.asarray(vector, dtype=complex),
            )
        )
    return ViewerPoint(
        name=name, q=qpoint, x=x, cell_matrix=cell_matrix, cell_description=description,
        lattice=lattice, symbols=symbols, positions=positions, atom_index=atom_index,
        translations=translations, bonds=bond_list, polyhedra=polyhedra,
        modes=viewer_modes, notes=notes, segment=segment,
    )


def _same_q(q1, q2) -> bool:
    diff = np.asarray(q1, dtype=float) - np.asarray(q2, dtype=float)
    return bool(np.allclose(diff, np.rint(diff), atol=1e-6))


def _selected_points(phonon, dispersion, segments, names, qpoint, dataset):
    """``[(name, q, x), ...]``: the labeled special points of the path in the
    order of their first appearance on it, or the one ``--qpoint`` point."""
    from .phonon_irreps import _special_points_in_label_frame
    from .phonon_vector import resolve_qpoint
    from .runtime_compat import get_symmetry_dataset

    vertices = [(n, k) for part, ns in zip(segments, names) for n, k in zip(ns, part)]
    if qpoint is None:
        def first(point):
            for index, (_, k) in enumerate(vertices):
                if _same_q(k, point.q):
                    return index
            return len(vertices)

        ordered = sorted(dispersion.points, key=first)
        return [(point.name, list(point.q), point.x) for point in ordered]

    from phonopy.structure.cells import get_primitive_matrix_by_centring

    from .phonon_vector import IrrepTable

    try:
        irt_table = IrrepTable(dataset["number"], spinor=False)
        prim_mat = get_primitive_matrix_by_centring(dataset["international"][0])
        q_names, q_list = _special_points_in_label_frame(phonon, irt_table, prim_mat)
    except Exception:
        q_names, q_list = [], []
    rotations = get_symmetry_dataset(phonon.primitive_symmetry)["rotations"]
    primitive, unitcell = phonon.primitive, phonon.unitcell
    isoir_context = (
        dataset["number"],
        (primitive.cell, primitive.scaled_positions, primitive.numbers),
        phonon.primitive_symmetry.tolerance,
        (unitcell.cell, unitcell.scaled_positions, unitcell.numbers),
    )
    if len(qpoint) == 1:
        requested = qpoint[0].strip().upper()
        requested = "GM" if requested in _GAMMA_ALIASES else requested
        # a label names the vertex of the path called so (its marker and its
        # modes then agree; K of a merged U|K is the K vertex of the path, at
        # the marker of U|K), else the tabulated point
        for point in dispersion.points:
            parts = [part.upper() for part in point.name.split("|")]
            if parts[0] == requested:
                return [(point.name, list(point.q), point.x)]
            if requested in parts:
                for name, k in vertices:
                    if str(name).upper() == requested:
                        return [(str(name), [float(v) for v in k], point.x)]
    try:
        name, q = resolve_qpoint(
            qpoint, q_names, q_list, rotations=rotations, isoir_context=isoir_context
        )
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from None
    for point in dispersion.points:
        if _same_q(point.q, q):
            return [(point.name, list(q), point.x)]
    return [(name, list(q), None)]


def _segment_midpoints(phonon, dispersion, segments, names, band, taken, base_matrix):
    """``(midpoints, notes)``: the segment midpoints of ``--all-irreps``.

    ``midpoints`` is ``[(label, q, x, segment), ...]`` in path order: for the
    seekpath path the midpoints of ``crystod-phonon --irreps --all-irreps``
    (:func:`crystod.phonon_irreps._seekpath_path_midpoints`, the same labels
    and segment names), for a ``--band`` path those of its own legs
    (:func:`crystod.phonon_irreps._label_segment_midpoints`); ``x`` is the
    middle of the leg on the dispersion.  A midpoint equal (modulo a
    reciprocal lattice vector) to a q of ``taken`` (the special points, then
    the midpoints already met) is left out; one whose display cell would
    hold more than :data:`LARGE_CELL_ATOMS` atoms is left out with a note.
    """
    from .phonon_irreps import _label_segment_midpoints, _seekpath_path_midpoints
    from .operations import snap_qpoint

    legs = [
        (str(ns[i]) or "?", str(ns[i + 1]) or "?", part[i], part[i + 1])
        for part, ns in zip(segments, names) for i in range(len(ns) - 1)
    ]
    notes: list[str] = []
    try:
        if band:
            found = _label_segment_midpoints(phonon, legs)
        else:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _, found = _seekpath_path_midpoints(phonon)
            if not found and any("does not match the phonopy primitive" in str(w.message)
                                 for w in caught):
                notes.append(
                    "the segment midpoints are not drawn: the seekpath primitive cell differs "
                    "from the primitive cell of phonopy (as in --irreps --all-irreps; give "
                    "--band for a path in the primitive cell of the input)"
                )
    except Exception as exc:
        found = []
        notes.append(f"the segment midpoints are not drawn ({type(exc).__name__}: {exc})")
    leg_midpoints = [
        snap_qpoint((np.asarray(k1, dtype=float) + np.asarray(k2, dtype=float)) / 2.0)
        for _, _, k1, k2 in legs
    ]
    taken = [list(q) for q in taken]
    midpoints = []
    for label, segment, q in found:
        q = [float(v) for v in q]
        if any(_same_q(q, other) for other in taken):
            continue
        x = None
        for index, k in enumerate(leg_midpoints):
            if index < len(dispersion.distances) and np.allclose(k, q, atol=1e-8):
                leg = np.asarray(dispersion.distances[index], dtype=float)
                x = float(leg[0] + leg[-1]) / 2.0
                break
        n_atoms = _cell_atom_count(phonon, _display_cell_matrix(q, base_matrix))
        taken.append(q)  # one NOTE for a skipped point, however often the path meets it
        if n_atoms > LARGE_CELL_ATOMS:
            coordinates = " ".join(_fraction(v) for v in np.asarray(q, dtype=float) + 0.0)
            notes.append(
                f"the midpoint {label} {format_q_fraction(q)} of {segment} is not drawn: its "
                f"display cell would hold {n_atoms} atoms (give --qpoint {coordinates} to "
                "draw it)"
            )
            continue
        midpoints.append((label, q, x, segment))
    return midpoints, notes


def phonon_viewer_data(
    phonon,
    *,
    irt_table=None,
    prim_mat=None,
    degeneracy_tolerance: float = 1e-3,
    band: str | None = None,
    band_labels: str | None = None,
    npoints: int | None = None,
    qpoint: list[str] | None = None,
    conventional: bool = False,
    bonds: list[tuple[str, str, float]] | None = None,
    threshold: float = -0.1,
    force_data: str | None = None,
    tolerance: float = 1e-5,
    all_irreps: bool = False,
) -> PhononViewer:
    """Compute everything the phonon mode viewer shows.

    Args:
        phonon: ``phonopy.Phonopy`` object with force constants, built with
            ``primitive_matrix="auto"`` (as ``crystod-phonon --irreps``).
        irt_table, prim_mat: ISO-IR table of the space group and centring
            matrix (default: from the symmetry of ``phonon``).
        degeneracy_tolerance: THz within which bands form one set.
        band, band_labels, npoints, tolerance: The path (``--band``,
            ``--band-labels``, ``--npoints``; default the seekpath path).
        qpoint: ``--qpoint`` tokens (one label or three coordinates) to show
            that point only; default every labeled special point of the path.
        conventional: Display the conventional cell (times the multiples
            commensurate with q) instead of the primitive cell.
        bonds: ``[(el1, el2, max_length_A), ...]``: bonds and the VESTA-style
            coordination polyhedra around ``el1``.
        threshold: Frequency (THz) below which a level is drawn as imaginary.
        force_data: Name of the force data shown on the page.
        all_irreps: Also show the midpoint of every segment of the path, as
            ``crystod-phonon --irreps --all-irreps`` labels it (the symmetry
            lines DT, Z, SM, ...; ``--band``: the midpoints of its legs),
            after the special points; the points get ``segment`` set.  A
            midpoint equal to a special point or to an earlier midpoint is
            left out, and so is one whose display cell would hold more than
            :data:`LARGE_CELL_ATOMS` atoms (with a note; ``qpoint`` draws it).

    Returns:
        The :class:`PhononViewer`; :func:`write_phonon_viewer_html` writes it.

    Raises:
        ValueError: ``all_irreps`` together with ``qpoint``, or an unknown
            ``qpoint`` label.
    """
    if all_irreps and qpoint is not None:
        raise ValueError("all_irreps is not used with qpoint (qpoint shows one point).")
    from phonopy.structure.cells import get_primitive_matrix_by_centring

    from .phonon_irreps import _gamma_mulliken, get_irt_special_points
    from .phonon_irreps_plot import compute_dispersion
    from .phonon_vector import IrrepTable, get_conventional_matrix, reduced_formula
    from .runtime_compat import get_symmetry_dataset

    dataset = get_symmetry_dataset(phonon.symmetry)
    if irt_table is None:
        irt_table = IrrepTable(dataset["number"], spinor=False)
    if prim_mat is None:
        prim_mat = get_primitive_matrix_by_centring(dataset["international"][0])
    dispersion = compute_dispersion(
        phonon, irt_table, prim_mat, degeneracy_tolerance=degeneracy_tolerance,
        band=band, band_labels=band_labels, npoints=npoints, tolerance=tolerance,
    )
    nac = getattr(phonon, "nac_params", None) is not None
    notes = list(dispersion.notes)
    if nac:
        notes.append(
            "at GM the labels and patterns are those of the TO modes (phonopy applies no q "
            "direction at GM itself); the LO branches end there without labels"
        )
    _, _, _, segments, names, _ = _band_path(phonon, band, band_labels, npoints, tolerance)
    selected = _selected_points(phonon, dispersion, segments, names, qpoint, dataset)

    lattice = np.array(phonon.primitive.cell, dtype=float)
    centring = None
    base_matrix = np.eye(3, dtype=int)
    conventional_lattice = None
    if conventional:
        centring = dataset["international"][0]
        base_matrix = get_conventional_matrix(centring)
        conventional_lattice = np.array(base_matrix, dtype=float) @ lattice

    entries = [(name, q, x, None) for name, q, x in selected]
    if all_irreps:
        midpoints, midpoint_notes = _segment_midpoints(
            phonon, dispersion, segments, names, band, [q for _, q, _ in selected], base_matrix
        )
        entries += midpoints
        notes += midpoint_notes

    q_names, q_list = get_irt_special_points(irt_table, prim_mat)
    rotations = get_symmetry_dataset(phonon.primitive_symmetry)["rotations"]
    has_gamma = any(np.allclose(q, np.rint(q), atol=1e-8) for _, q, _ in selected)
    mulliken = _gamma_mulliken(phonon) if has_gamma else {}
    labelling = (irt_table, prim_mat, q_names, q_list, rotations, mulliken)

    points = [
        _viewer_point(phonon, name, q, x, labelling, degeneracy_tolerance,
                      base_matrix, centring, bonds or [], segment=segment)
        for name, q, x, segment in entries
    ]
    return PhononViewer(
        formula=reduced_formula(list(phonon.primitive.symbols)),
        space_group=str(dataset["international"]),
        number=int(dataset["number"]),
        supercell=_supercell_text(phonon),
        force_data=force_data,
        nac=nac,
        path_source="--band" if band else "seekpath",
        dispersion=dispersion,
        points=points,
        threshold=float(threshold),
        bonds=list(bonds or []),
        primitive_lattice=lattice,
        conventional_lattice=conventional_lattice,
        amplitude=round(default_amplitude(phonon), 2),
        notes=notes,
        all_irreps=bool(all_irreps),
    )


# --------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------


def _round_list(values, decimals):
    return [_clean(v, decimals) for v in np.asarray(values, dtype=float).ravel()]


def _polyhedra_payload(polyhedra) -> list[dict]:
    """The polyhedra of one q point, one mesh per element: ``{element,
    faces: [[i, j, k], ...]}`` with the drawn-atom slots of the triangles."""
    merged: dict[str, list] = {}
    for element, _, faces in polyhedra:
        merged.setdefault(element, []).extend([[int(v) for v in face] for face in faces])
    return [{"element": element, "faces": faces} for element, faces in merged.items()]


def _point_payload(point: ViewerPoint, label: str | None = None) -> dict:
    """The data of one q point as shipped to the page: the page rebuilds the
    displacement of drawn atom ``j`` as ``Re[u_{a_j} exp(2 pi i q.T_j)]``.
    A segment midpoint carries its ``segment`` and, when another q point of
    the page has the same name, the button ``label`` ``DT (GM-X)``."""
    from .visualize_basis import _lattice_edge_traces

    modes = []
    for mode in point.modes:
        entry = {
            "band": mode.band,
            "freq": _clean(mode.frequency, _BAND_DECIMALS),
            "irrep": mode.irrep,
            "comp": mode.component,
            "dim": mode.dimension,
            "set": mode.set_index,
            "u": [
                [[_clean(c.real, _VECTOR_DECIMALS), _clean(c.imag, _VECTOR_DECIMALS)] for c in atom]
                for atom in mode.u
            ],
        }
        if mode.mulliken:
            entry["mulliken"] = mode.mulliken
        if mode.activity:
            entry["activity"] = mode.activity
        modes.append(entry)
    edges = _lattice_edge_traces(point.lattice, (1, 1, 1))
    for axis in "xyz":
        edges[axis] = [None if v is None else _clean(v, _POSITION_DECIMALS) for v in edges[axis]]
    extra = {}
    if point.segment is not None:
        extra["segment"] = point.segment
    if label is not None:
        extra["label"] = label
    return {
        "name": point.name,
        **extra,
        "q": [float(v) + 0.0 for v in point.q],
        "qText": format_q_fraction(point.q),
        "x": None if point.x is None else _clean(point.x, _DISTANCE_DECIMALS),
        "gamma": point.is_gamma,
        "cell": point.cell_description,
        "lattice": [_round_list(row, _POSITION_DECIMALS) for row in point.lattice],
        "edges": edges,
        "atoms": {
            "s": list(point.symbols),
            "p": [_round_list(p, _POSITION_DECIMALS) for p in point.positions],
            "a": [int(a) for a in point.atom_index],
            "T": [[int(t) for t in row] for row in point.translations],
        },
        "bonds": [{"name": name, "pairs": [[int(i), int(j)] for i, j in pairs]}
                  for name, pairs in point.bonds],
        "polyhedra": _polyhedra_payload(point.polyhedra),
        "modes": modes,
        "notes": list(point.notes),
    }


def _payload(data: PhononViewer) -> dict:
    """Everything the page script reads (``var DATA``)."""
    from .phonon_vector import VESTA_ELEMENTS, DEFAULT_ELEMENT
    from .visualize_basis import _load_atom_colors

    dispersion = data.dispersion
    colors = _load_atom_colors()
    elements = sorted({s for point in data.points for s in point.symbols})
    return {
        "threshold": data.threshold,
        "amplitude": data.amplitude,
        "arrowCutoff": ARROW_CUTOFF,
        "colors": {s: colors.get(s, "#cccccc") for s in elements},
        "radii": {s: VESTA_ELEMENTS.get(s, DEFAULT_ELEMENT)[0] for s in elements},
        "band": {
            "distances": [_round_list(d, _DISTANCE_DECIMALS) for d in dispersion.distances],
            "frequencies": [
                [_round_list(row, _BAND_DECIMALS) for row in np.asarray(f, dtype=float)]
                for f in dispersion.frequencies
            ],
            "ticks": [_clean(t, _DISTANCE_DECIMALS) for t in dispersion.ticks],
            "tickText": [_tick_html(t) for t in dispersion.tick_labels],
        },
        "points": [
            _point_payload(point, label)
            for point, label in zip(data.points, _button_labels(data.points))
        ],
    }


def _button_labels(points) -> list:
    """The q-button text of each point that needs more than its name: a
    segment midpoint whose name another point of the page carries too (two
    arms of one line type) is ``DT (GM-X)``; None for every other point."""
    counts: dict[str, int] = {}
    for point in points:
        counts[point.name] = counts.get(point.name, 0) + 1
    return [
        f"{point.name} ({point.segment})"
        if point.segment is not None and counts[point.name] > 1 else None
        for point in points
    ]


def _info_rows(data: PhononViewer) -> str:
    bond_summary = "; ".join(
        f"{el1}&ndash;{el2} &le; {length:g} &Aring;" for el1, el2, length in data.bonds
    )
    rows = (
        ("Compound", data.formula),
        ("Space group", f"{data.space_group} (#{data.number})"),
        ("Supercell", data.supercell),
        ("Force data", data.force_data or ""),
        ("NAC", "yes" if data.nac else "no"),
        (f"k path ({data.path_source})", data.dispersion.path_string),
        ("q points", str(data.dispersion.n_qpoints)),
        ("Bonds", bond_summary),
    )
    return "".join(f"<tr><td>{name}</td><td>{value}</td></tr>" for name, value in rows if value)


_CSS = """\
  * { box-sizing: border-box; }
  body { margin: 0; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
         color: #333; background: #fff; height: 100vh; display: flex; flex-direction: column; }
  #topbar { background: #2c3e50; color: #ecf0f1; padding: 8px 16px;
            display: flex; justify-content: space-between; align-items: baseline; gap: 16px; }
  #topbar .brand { font-size: 18px; font-weight: bold; white-space: nowrap; }
  #topbar .brand small { font-weight: normal; opacity: 0.8; margin-left: 8px; }
  #topbar .page-title { font-size: 13px; opacity: 0.9; text-align: right; }
  #container { flex: 1; display: flex; min-height: 0; }
  #sidebar { width: 360px; min-width: 360px; overflow-y: auto; background: #f7f7f7;
             border-right: 1px solid #ddd; padding: 12px 16px; font-size: 13px; }
  #sidebar h2 { font-size: 13px; text-transform: uppercase; letter-spacing: 0.05em;
                color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 3px;
                margin: 18px 0 8px; }
  #sidebar h2:first-child { margin-top: 4px; }
  table.info { width: 100%; border-collapse: collapse; }
  table.info td { padding: 2px 4px; vertical-align: top; }
  table.info td:first-child { color: #777; width: 36%; }
  #qpoints { display: flex; flex-wrap: wrap; gap: 4px; }
  #qpoints .qgroup { flex-basis: 100%; color: #777; font-size: 11px; margin-top: 4px; }
  button.qbtn, button.vbtn { font: inherit; font-size: 12px; padding: 3px 7px;
                             border: 1px solid #c8d2dc;
                background: #fff; border-radius: 3px; cursor: pointer; color: #2c3e50; }
  button.qbtn:hover, button.vbtn:hover { background: #eaf1f8; }
  button.qbtn.active { background: #2c3e50; border-color: #2c3e50; color: #fff; }
  table#mode-table { width: 100%; border-collapse: collapse; background: #fff;
                     border: 1px solid #e0e0e0; font-size: 12px; }
  table#mode-table th { background: #2c3e50; color: #fff; font-weight: normal;
                        padding: 4px 5px; text-align: left; white-space: nowrap; }
  table#mode-table td { padding: 3px 5px; border-top: 1px solid #eee; cursor: pointer;
                        white-space: nowrap; }
  table#mode-table td.num { text-align: right; font-variant-numeric: tabular-nums; }
  table#mode-table th.num { text-align: right; }
  table#mode-table td.irrep { font-family: Menlo, Consolas, monospace; font-size: 11px; }
  table#mode-table tr.set-start td { border-top: 1px solid #b9c4cf; }
  tr.mode-row:hover { background: #eaf1f8; }
  tr.mode-row.active { background: #d5e5f5; font-weight: bold; }
  tr.mode-row.imaginary td { color: #e34948; }
  .control { margin: 6px 0; display: flex; align-items: center; gap: 8px; }
  .control label { flex: 1; }
  .control input[type=range] { width: 130px; }
  .control .value { width: 52px; text-align: right; font-variant-numeric: tabular-nums; }
  .note { color: #777; font-size: 12px; margin: 4px 0; }
  .credit { margin-top: 24px; padding-top: 8px; border-top: 1px solid #ddd;
            font-size: 11px; color: #888; }
  .credit a { color: #2c6aa0; }
  #main { flex: 1; display: flex; flex-direction: column; min-width: 0; }
  #mode-title { padding: 8px 14px; font-size: 15px; border-bottom: 1px solid #eee;
                background: #fff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  #mode-title .irrep { font-family: Menlo, Consolas, monospace; color: #2c6aa0; }
  #mode-title .imaginary { color: #e34948; }
  #band { flex: 0 0 38%; min-height: 0; border-bottom: 1px solid #eee; }
  #plot { flex: 1; min-height: 0; }
"""


_SCRIPT = r"""
var CURVE = '__CURVE__', IMAGINARY = '__IMAGINARY__', ARROW = '__ARROW__';
var CAMERA_EYE = __CAMERA_EYE__;
var plotDiv = document.getElementById('plot');
var bandDiv = document.getElementById('band');
var state = {point: 0, mode: 0, phase: 0, animating: false, last: null, uirevision: 0,
             view: null, aspect: null};
var ORTHO_ZOOM = 1.9, ORTHO_FILL = 0.9;
var scene = null;  // trace layout and the field of the current mode
function isChecked(id) {
  var element = document.getElementById(id);
  return element ? element.checked : true;
}
function amplitude() { return Number(document.getElementById('amplitude').value); }
function isImaginary(f) { return f < DATA.threshold; }
function fmt(f) { return (Math.abs(f) < 5e-5 ? 0 : f).toFixed(4); }

// ---- the displacement field: Re[u_a exp(i(2 pi q.T - phi))] -------------
function modeField(point, mode) {
  var q = point.q, atoms = point.atoms, re = [], im = [];
  for (var j = 0; j < atoms.a.length; j++) {
    var u = mode.u[atoms.a[j]], T = atoms.T[j];
    var theta = 2 * Math.PI * (q[0] * T[0] + q[1] * T[1] + q[2] * T[2]);
    var c = Math.cos(theta), s = Math.sin(theta);
    re.push([u[0][0] * c - u[0][1] * s, u[1][0] * c - u[1][1] * s, u[2][0] * c - u[2][1] * s]);
    im.push([u[0][0] * s + u[0][1] * c, u[1][0] * s + u[1][1] * c, u[2][0] * s + u[2][1] * c]);
  }
  return {re: re, im: im};
}
function atomPositions(phase) {
  // equilibrium positions, displaced by the pattern at the animation phase
  var point = DATA.points[state.point], p = point.atoms.p;
  if (!state.animating) { return p; }
  var A = amplitude(), c = Math.cos(phase), s = Math.sin(phase), out = [];
  for (var j = 0; j < p.length; j++) {
    var re = scene.field.re[j], im = scene.field.im[j];
    out.push([p[j][0] + A * (re[0] * c + im[0] * s),
              p[j][1] + A * (re[1] * c + im[1] * s),
              p[j][2] + A * (re[2] * c + im[2] * s)]);
  }
  return out;
}

// ---- 3D traces -----------------------------------------------------------
function groupTrace(point, group, positions) {
  var xs = [], ys = [], zs = [];
  group.slots.forEach(function (j) {
    xs.push(positions[j][0]); ys.push(positions[j][1]); zs.push(positions[j][2]);
  });
  return {x: xs, y: ys, z: zs};
}
function arrowData(positions) {
  // shafts from the (moving) atom to atom + pattern, heads as cones at the tip
  var A = amplitude(), re = scene.field.re;
  var sx = [], sy = [], sz = [], cx = [], cy = [], cz = [], cu = [], cv = [], cw = [];
  for (var j = 0; j < positions.length; j++) {
    var d = re[j], n = Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    if (n < DATA.arrowCutoff) { continue; }
    var p = positions[j];
    var tip = [p[0] + A * d[0], p[1] + A * d[1], p[2] + A * d[2]];
    var f = 1 - 0.9 * DATA.headFraction;  // the shaft ends inside the head
    sx.push(p[0], p[0] + f * A * d[0], null); sy.push(p[1], p[1] + f * A * d[1], null);
    sz.push(p[2], p[2] + f * A * d[2], null);
    cx.push(tip[0]); cy.push(tip[1]); cz.push(tip[2]);
    cu.push(A * d[0]); cv.push(A * d[1]); cw.push(A * d[2]);
  }
  return {shaft: {x: sx, y: sy, z: sz}, cone: {x: cx, y: cy, z: cz, u: cu, v: cv, w: cw}};
}
function bondData(bond, positions) {
  var xs = [], ys = [], zs = [];
  bond.pairs.forEach(function (pair) {
    var a = positions[pair[0]], b = positions[pair[1]];
    xs.push(a[0], b[0], null); ys.push(a[1], b[1], null); zs.push(a[2], b[2], null);
  });
  return {x: xs, y: ys, z: zs};
}
function polyData(positions) {
  // every drawn atom is a vertex; the faces index them
  return {x: positions.map(function (p) { return p[0]; }),
          y: positions.map(function (p) { return p[1]; }),
          z: positions.map(function (p) { return p[2]; })};
}
function groupsOf(point) {
  // one trace per element, in alphabetical order (as the SALC viewer)
  var symbols = Array.from(new Set(point.atoms.s)).sort();
  return symbols.map(function (symbol) {
    var slots = [];
    point.atoms.s.forEach(function (s, j) { if (s === symbol) { slots.push(j); } });
    return {symbol: symbol, slots: slots};
  });
}
function sceneTraces(point) {
  // atom markers have a fixed size in pixels: shrink them with the extent of
  // the display cell so that the default view keeps the proportions
  var extent = Math.max.apply(null, point.lattice.map(function (v) {
    return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  }));
  scene.markerScale = Math.max(0.55, Math.min(1.0, 6.0 / extent));
  var traces = [], positions = atomPositions(state.phase);
  var cell = Object.assign({}, point.edges);
  traces.push(cell);
  DATA.axes.forEach(function (trace) { traces.push(trace); });
  var cellEnd = traces.length;
  var groups = groupsOf(point);
  groups.forEach(function (group) {
    var xyz = groupTrace(point, group, positions);
    var radius = DATA.radii[group.symbol] || 0.8;
    traces.push({type: 'scatter3d', mode: 'markers', x: xyz.x, y: xyz.y, z: xyz.z,
      marker: {size: scene.markerScale * (1.6 + 2.2 * radius),
               color: DATA.colors[group.symbol] || '#cccccc',
               line: {color: '#333333', width: 1}},
      name: group.symbol,
      text: group.slots.map(function (j) {
        return group.symbol + (point.atoms.a[j] + 1)
          + '  T = (' + point.atoms.T[j].join(', ') + ')';
      }),
      hoverinfo: 'text'});
  });
  var atomsEnd = traces.length;
  var arrows = arrowData(positions);
  traces.push({type: 'scatter3d', mode: 'lines',
    x: arrows.shaft.x, y: arrows.shaft.y, z: arrows.shaft.z,
    line: {color: ARROW, width: 6}, hoverinfo: 'skip', showlegend: false});
  traces.push({type: 'cone', x: arrows.cone.x, y: arrows.cone.y, z: arrows.cone.z,
    u: arrows.cone.u, v: arrows.cone.v, w: arrows.cone.w,
    anchor: 'tip', sizemode: 'raw', sizeref: DATA.headFraction,
    colorscale: [[0, ARROW], [1, ARROW]], showscale: false,
    lighting: {ambient: 0.75, diffuse: 0.5, specular: 0.2},
    hoverinfo: 'skip', showlegend: false});
  var arrowsEnd = traces.length;
  point.bonds.forEach(function (bond) {
    var xyz = bondData(bond, positions);
    traces.push({type: 'scatter3d', mode: 'lines', x: xyz.x, y: xyz.y, z: xyz.z,
      line: {color: '#7a7a7a', width: 5}, name: bond.name, hoverinfo: 'skip'});
  });
  var bondsEnd = traces.length;
  point.polyhedra.forEach(function (poly) {
    var xyz = polyData(positions);
    traces.push({type: 'mesh3d', x: xyz.x, y: xyz.y, z: xyz.z,
      i: poly.faces.map(function (f) { return f[0]; }),
      j: poly.faces.map(function (f) { return f[1]; }),
      k: poly.faces.map(function (f) { return f[2]; }),
      opacity: 0.35, color: DATA.colors[poly.element] || '#cccccc', flatshading: true,
      name: poly.element + ' polyhedra', showlegend: true, hoverinfo: 'skip'});
  });
  scene.groups = groups;
  scene.cellEnd = cellEnd; scene.atomsEnd = atomsEnd; scene.arrowsEnd = arrowsEnd;
  scene.bondsEnd = bondsEnd; scene.n = traces.length;
  traces.forEach(function (trace, index) { trace.visible = traceVisible(index); });
  return traces;
}
function traceVisible(index) {
  if (index < scene.cellEnd) { return isChecked('show-cell'); }
  if (index < scene.atomsEnd) { return isChecked('show-atoms'); }
  if (index < scene.arrowsEnd) { return isChecked('show-arrows'); }
  if (index < scene.bondsEnd) { return isChecked('show-bonds'); }
  return isChecked('show-poly');
}
function applyVisibility() {
  var indices = [], visible = [];
  for (var i = 0; i < scene.n; i++) { indices.push(i); visible.push(traceVisible(i)); }
  Plotly.restyle(plotDiv, {visible: visible}, indices);
}
function updateScene() {
  // move atoms, arrows, bonds and polyhedra to the current phase: one restyle
  // of their coordinates, one of the cone vectors
  var point = DATA.points[state.point], positions = atomPositions(state.phase);
  var update = {x: [], y: [], z: []}, indices = [];
  var add = function (xyz, index) {
    update.x.push(xyz.x); update.y.push(xyz.y); update.z.push(xyz.z); indices.push(index);
  };
  scene.groups.forEach(function (group, k) {
    add(groupTrace(point, group, positions), scene.cellEnd + k);
  });
  var arrows = arrowData(positions);
  add(arrows.shaft, scene.atomsEnd);
  add(arrows.cone, scene.atomsEnd + 1);
  point.bonds.forEach(function (bond, k) { add(bondData(bond, positions), scene.arrowsEnd + k); });
  point.polyhedra.forEach(function (poly, k) { add(polyData(positions), scene.bondsEnd + k); });
  Plotly.restyle(plotDiv, update, indices);
  Plotly.restyle(plotDiv, {u: [arrows.cone.u], v: [arrows.cone.v], w: [arrows.cone.w]},
                 [scene.atomsEnd + 1]);
}

// ---- dispersion chart ----------------------------------------------------
function bandTraces() {
  var band = DATA.band, xs = [], ys = [], rx = [], ry = [];
  var nBands = band.frequencies[0][0].length;
  for (var b = 0; b < nBands; b++) {
    for (var leg = 0; leg < band.distances.length; leg++) {
      var d = band.distances[leg], f = band.frequencies[leg];
      for (var i = 0; i < d.length; i++) {
        xs.push(d[i]); ys.push(f[i][b]);
        rx.push(d[i]); ry.push(isImaginary(f[i][b]) ? f[i][b] : null);
      }
      xs.push(null); ys.push(null); rx.push(null); ry.push(null);
    }
  }
  var mx = [], my = [], mc = [], mt = [], md = [];
  DATA.points.forEach(function (point, pi) {
    if (point.x === null) { return; }
    var seen = {};
    point.modes.forEach(function (mode, mi) {
      if (seen[mode.set]) { return; }
      seen[mode.set] = true;
      mx.push(point.x); my.push(mode.freq);
      mc.push(isImaginary(mode.freq) ? IMAGINARY : CURVE);
      mt.push(pointLabel(point) + '  ' + mode.irrep + '<br>' + fmt(mode.freq) + ' THz');
      md.push([pi, mi]);
    });
  });
  return [
    {type: 'scatter', mode: 'lines', x: xs, y: ys, line: {color: CURVE, width: 1.3},
     hoverinfo: 'skip', showlegend: false},
    {type: 'scatter', mode: 'lines', x: rx, y: ry, line: {color: IMAGINARY, width: 1.6},
     hoverinfo: 'skip', showlegend: false},
    {type: 'scatter', mode: 'markers', x: mx, y: my, text: mt, customdata: md,
     marker: {size: 8, color: mc, line: {color: '#ffffff', width: 1}},
     hovertemplate: '%{text}<extra></extra>', showlegend: false},
    {type: 'scatter', mode: 'markers', x: [], y: [],
     marker: {size: 15, color: 'rgba(0,0,0,0)', line: {color: '#2c3e50', width: 2.5}},
     hoverinfo: 'skip', showlegend: false}
  ];
}
function bandLayout() {
  var band = DATA.band, last = band.distances[band.distances.length - 1];
  var xmax = last[last.length - 1];
  var shapes = band.ticks.slice(1, -1).map(function (t) {
    return {type: 'line', xref: 'x', yref: 'paper', x0: t, x1: t, y0: 0, y1: 1,
            line: {color: '#c8c8c8', width: 1}, layer: 'below'};
  });
  shapes.push({type: 'line', xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: 0, y1: 0,
               line: {color: '#777777', width: 1, dash: 'dash'}, layer: 'below'});
  return {
    margin: {l: 58, r: 14, t: 10, b: 28},
    paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff',
    font: {family: "'Helvetica Neue', Helvetica, Arial, sans-serif", size: 12, color: '#333'},
    xaxis: {range: [0, xmax], tickvals: band.ticks, ticktext: band.tickText, showgrid: false,
            zeroline: false, showline: true, linecolor: '#444', mirror: true, ticks: '',
            tickfont: {size: 13}, fixedrange: true},
    yaxis: {title: {text: 'Frequency (THz)', standoff: 6}, zeroline: false, showgrid: false,
            showline: true, linecolor: '#444', mirror: true, ticks: 'outside', fixedrange: true},
    hovermode: 'closest', shapes: shapes, dragmode: false
  };
}
function markSelected() {
  var point = DATA.points[state.point], mode = point.modes[state.mode];
  var x = point.x === null ? [] : [point.x], y = point.x === null ? [] : [mode.freq];
  Plotly.restyle(bandDiv, {x: [x], y: [y]}, [3]);
}

// ---- sidebar -------------------------------------------------------------
function pointLabel(point) { return point.label || point.name; }
function buildQButtons() {
  // the special points, then (--all-irreps) the segment midpoints
  var html = '', midpoints = false;
  DATA.points.forEach(function (point, i) {
    if (point.segment && !midpoints) {
      midpoints = true;
      html += '<div class="qgroup">Segment midpoints</div>';
    }
    html += '<button class="qbtn" data-index="' + i + '" onclick="selectPoint(' + i + ', 0)"'
      + (point.segment ? ' title="midpoint of ' + point.segment + '"' : '') + '>'
      + pointLabel(point) + ' ' + point.qText + '</button>';
  });
  document.getElementById('qpoints').innerHTML = html;
}
function buildModeTable(point) {
  var gamma = point.gamma;
  var head = '<tr><th class="num">Band</th><th class="num">Freq (THz)</th><th>Irrep</th>'
    + '<th class="num">Comp.</th>'
    + (gamma ? '<th>Mulliken</th><th>Activity</th>' : '') + '</tr>';
  var rows = point.modes.map(function (mode, i) {
    var classes = 'mode-row' + (mode.comp === 1 ? ' set-start' : '')
      + (isImaginary(mode.freq) ? ' imaginary' : '');
    return '<tr class="' + classes + '" onclick="selectMode(' + i + ')">'
      + '<td class="num">' + mode.band + '</td><td class="num">' + fmt(mode.freq) + '</td>'
      + '<td class="irrep">' + mode.irrep + '</td><td class="num">' + mode.comp + '</td>'
      + (gamma ? '<td>' + (mode.mulliken || '') + '</td><td>' + (mode.activity || '') + '</td>'
               : '')
      + '</tr>';
  }).join('');
  document.getElementById('mode-table').innerHTML = head + rows;
  document.getElementById('cell-line').innerHTML =
    (point.segment ? 'Segment: ' + point.segment + ' (midpoint)<br>' : '')
    + 'Display cell: ' + point.cell;
  document.getElementById('point-notes').innerHTML = point.notes.map(function (note) {
    return '<div class="note">NOTE: ' + note + '</div>';
  }).join('');
}
function updateTitle() {
  var point = DATA.points[state.point], mode = point.modes[state.mode];
  var extra = point.gamma && (mode.mulliken || mode.activity)
    ? '  &middot;  ' + [mode.mulliken, mode.activity].filter(Boolean).join(', ') : '';
  document.getElementById('mode-title').innerHTML =
    '<b>' + point.name + '</b> ' + point.qText
    + (point.segment ? ', midpoint of ' + point.segment : '') + '  &middot;  band ' + mode.band
    + '  <span class="irrep">' + mode.irrep + '</span> comp ' + mode.comp
    + '  &middot;  <span class="' + (isImaginary(mode.freq) ? 'imaginary' : '') + '">'
    + fmt(mode.freq) + ' THz</span>' + extra;
}
function highlight() {
  document.querySelectorAll('#qpoints button.qbtn').forEach(function (button, i) {
    button.classList.toggle('active', i === state.point);
  });
  document.querySelectorAll('tr.mode-row').forEach(function (row, i) {
    row.classList.toggle('active', i === state.mode);
  });
}

// ---- selection -----------------------------------------------------------
function currentCamera() {
  var layoutScene = plotDiv._fullLayout && plotDiv._fullLayout.scene;
  return (layoutScene && layoutScene.camera && layoutScene.camera.eye)
    ? layoutScene.camera
    : {eye: CAMERA_EYE, up: {x: 0, y: 0, z: 1}};
}
function selectPoint(index, modeIndex) {
  var changed = index !== state.point || scene === null;
  state.point = index;
  var point = DATA.points[index];
  state.mode = Math.max(0, Math.min(modeIndex || 0, point.modes.length - 1));
  if (changed) {
    var camera = scene === null ? null : currentCamera();
    scene = {field: modeField(point, point.modes[state.mode])};
    buildModeTable(point);
    var traces = sceneTraces(point);
    var layout = JSON.parse(JSON.stringify(LAYOUT));
    layout.uirevision = state.uirevision;
    if (camera && !state.view) {
      layout.scene.camera = {eye: camera.eye, up: camera.up, center: camera.center};
    }
    if (state.view) {
      // stay orthographic down the axis of the new cell: every change of the
      // projection makes plotly build a new WebGL context, and the browser
      // drops the oldest one after about 16
      layout.scene.camera = axisCamera(state.view, point.lattice);
    }
    state.aspect = null;
    Plotly.react(plotDiv, traces, layout, {responsive: true, displaylogo: false}).then(function () {
      // an axis view is set up again on the new display cell
      if (state.view) { return viewAlong(state.view); }
      syncCompass(currentCamera());
    });
  } else {
    scene.field = modeField(point, point.modes[state.mode]);
    updateScene();
  }
  highlight();
  updateTitle();
  markSelected();
}
function selectMode(index) { selectPoint(state.point, index); }
function onAmplitude(value) {
  document.getElementById('amplitude-value').innerHTML = Number(value).toFixed(2) + ' &Aring;';
  updateScene();
}
function onSpeed(value) {
  document.getElementById('speed-value').innerHTML = Number(value).toFixed(1) + ' Hz';
}
function toggleAnimation(on) {
  state.animating = on;
  state.phase = 0;
  state.last = null;
  if (on) { requestAnimationFrame(step); } else { updateScene(); }
}
function step(time) {
  if (!state.animating) { return; }
  if (state.last !== null) {
    var speed = Number(document.getElementById('speed').value);
    state.phase = (state.phase + 2 * Math.PI * speed * (time - state.last) / 1000) % (2 * Math.PI);
  }
  state.last = time;
  updateScene();
  requestAnimationFrame(step);
}
function syncCompass(camera) {
  // update ONLY the compass scene through plotly's internal setViewport:
  // calling Plotly.relayout during a drag would redraw the main scene
  // from the (stale) stored camera and cancel the rotation in progress.
  var eye = camera.eye;
  var radius = Math.sqrt(eye.x * eye.x + eye.y * eye.y + eye.z * eye.z) || 1;
  var scale = 2.0 / radius;  // compass orientation follows the view, size stays fixed
  var sceneLayout = plotDiv._fullLayout && plotDiv._fullLayout.scene2;
  var sceneObject = sceneLayout && sceneLayout._scene;
  if (!sceneObject) { return; }
  var scaledEye = {x: eye.x * scale, y: eye.y * scale, z: eye.z * scale};
  var upVector = camera.up
    ? {x: camera.up.x, y: camera.up.y, z: camera.up.z}
    : {x: 0, y: 0, z: 1};
  sceneLayout.camera.eye = scaledEye;
  sceneLayout.camera.up = upVector;
  sceneLayout.camera.center = {x: 0, y: 0, z: 0};
  sceneLayout.camera.projection = sceneLayout.camera.projection || {type: 'perspective'};
  // persist into the user layout too: on drag end plotly rebuilds the
  // compass scene from plotDiv.layout, which would reset the camera
  if (plotDiv.layout && plotDiv.layout.scene2) {
    plotDiv.layout.scene2.camera = {eye: scaledEye, up: upVector, center: {x: 0, y: 0, z: 0}};
  }
  try { sceneObject.setViewport(sceneLayout); } catch (error) { /* keep the drag alive */ }
}
function axisCamera(axis, lattice) {
  // the orthographic camera down the a, b or c axis of a cell (up: c, c, b)
  var k = {a: 0, b: 1, c: 2}[axis];
  var unit = function (v) {
    var n = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    return {x: v[0] / n, y: v[1] / n, z: v[2] / n};
  };
  var d = unit(lattice[k]), up = unit(lattice[k === 2 ? 1 : 2]);
  return {eye: {x: 2 * d.x, y: 2 * d.y, z: 2 * d.z}, up: up, center: {x: 0, y: 0, z: 0},
          projection: {type: 'orthographic'}};
}
function orthoZoom(camera, lattice, base) {
  // the zoom of an axis view: ORTHO_ZOOM, or less when the cell projected on
  // the screen would fill more than ORTHO_FILL of the scene (an orthographic
  // scene shows about 2 zoomed units vertically); a cube-shaped cell keeps
  // ORTHO_ZOOM, an elongated one (1 x 4 x 1 cells of a line midpoint seen
  // down a) is zoomed out until it fits
  var sceneLayout = plotDiv._fullLayout.scene, scale = 0;
  ['x', 'y', 'z'].forEach(function (key) {
    var range = sceneLayout[key + 'axis'].range;
    scale = Math.max(scale, base[key] / Math.max(range[1] - range[0], 1e-9));
  });
  var unit = function (v) {
    var n = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) || 1;
    return [v[0] / n, v[1] / n, v[2] / n];
  };
  var d = unit([camera.eye.x, camera.eye.y, camera.eye.z]);
  var u = [camera.up.x, camera.up.y, camera.up.z];
  var ud = u[0] * d[0] + u[1] * d[1] + u[2] * d[2];
  u = unit([u[0] - ud * d[0], u[1] - ud * d[1], u[2] - ud * d[2]]);
  var r = [d[1] * u[2] - d[2] * u[1], d[2] * u[0] - d[0] * u[2], d[0] * u[1] - d[1] * u[0]];
  var lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
  for (var corner = 0; corner < 8; corner++) {
    var f = [corner & 1, (corner >> 1) & 1, (corner >> 2) & 1], p = [0, 0, 0];
    for (var c = 0; c < 3; c++) {
      p[c] = f[0] * lattice[0][c] + f[1] * lattice[1][c] + f[2] * lattice[2][c];
    }
    var across = p[0] * r[0] + p[1] * r[1] + p[2] * r[2];
    var upward = p[0] * u[0] + p[1] * u[1] + p[2] * u[2];
    lo[0] = Math.min(lo[0], across); hi[0] = Math.max(hi[0], across);
    lo[1] = Math.min(lo[1], upward); hi[1] = Math.max(hi[1], upward);
  }
  var width = (hi[0] - lo[0]) * scale, height = (hi[1] - lo[1]) * scale;
  var ratio = plotDiv.clientWidth / Math.max(plotDiv.clientHeight, 1);
  return Math.min(ORTHO_ZOOM, 2 * ORTHO_FILL / Math.max(height, 1e-9),
                  2 * ORTHO_FILL * ratio / Math.max(width, 1e-9));
}
function viewAlong(axis) {
  // look down the a, b or c axis of the display cell (up: c, c, b) in
  // orthographic projection, so that the atoms of a column coincide, or
  // return to the initial perspective view
  var camera = {eye: CAMERA_EYE, up: {x: 0, y: 0, z: 1}, center: {x: 0, y: 0, z: 0},
                projection: {type: 'perspective'}};
  var k = {a: 0, b: 1, c: 2}[axis];
  var layout = JSON.parse(JSON.stringify(LAYOUT));
  state.view = k === undefined ? null : axis;
  if (k !== undefined) {
    camera = axisCamera(axis, DATA.points[state.point].lattice);
    // plotly zooms an orthographic scene through its aspect ratio
    var aspect = plotDiv._fullLayout.scene.aspectratio;
    if (plotDiv._fullLayout.scene.aspectmode !== 'manual') { state.aspect = aspect; }
    var base = state.aspect || aspect;
    var zoom = orthoZoom(camera, DATA.points[state.point].lattice, base);
    layout.scene.aspectmode = 'manual';
    layout.scene.aspectratio = {x: zoom * base.x, y: zoom * base.y, z: zoom * base.z};
  }
  // a new projection needs a full redraw: react with a new uirevision
  state.uirevision += 1;
  layout.scene.camera = camera;
  layout.uirevision = state.uirevision;
  return Plotly.react(plotDiv, plotDiv.data, layout).then(function () { syncCompass(camera); });
}
function fromHash() {
  // #q=R&mode=4&view=a : the first q point named R (G, GM, GAMMA for Gamma),
  // band 4, viewed down the a axis; #q=DT(GM-X) the midpoint of the segment
  // GM-X, #q=7 the seventh q button
  var params = {};
  location.hash.replace(/^#/, '').split('&').forEach(function (part) {
    var kv = part.split('=');
    if (kv.length === 2) { params[kv[0].toLowerCase()] = decodeURIComponent(kv[1]); }
  });
  var pointIndex = 0, modeIndex = 0;
  if (params.q) {
    var wanted = params.q.toUpperCase().replace(/\s+/g, '');
    if (['G', 'GAMMA', 'Γ'].indexOf(wanted) >= 0) { wanted = 'GM'; }
    var found = -1;
    if (/^[0-9]+$/.test(wanted)) {
      var n = parseInt(wanted, 10);
      if (n >= 1 && n <= DATA.points.length) { found = n - 1; }
    }
    DATA.points.forEach(function (point, i) {
      if (found >= 0) { return; }
      var qualified = point.segment ? (point.name + '(' + point.segment + ')').toUpperCase() : '';
      if (point.name.toUpperCase().split('|').indexOf(wanted) >= 0 || wanted === qualified) {
        found = i;
      }
    });
    if (found >= 0) { pointIndex = found; }
  }
  if (params.mode) {
    var band = parseInt(params.mode, 10);
    DATA.points[pointIndex].modes.forEach(function (mode, i) {
      if (mode.band === band) { modeIndex = i; }
    });
  }
  return [pointIndex, modeIndex, params.view];
}

buildQButtons();
Plotly.newPlot(bandDiv, bandTraces(), bandLayout(), {responsive: true, displaylogo: false,
  displayModeBar: false}).then(function () {
  bandDiv.on('plotly_click', function (event) {
    var hit = event.points && event.points.filter(function (p) { return p.customdata; })[0];
    if (hit) { selectPoint(hit.customdata[0], hit.customdata[1]); }
  });
  var start = fromHash();
  selectPoint(start[0], start[1]);
  if (start[2]) { setTimeout(function () { viewAlong(start[2]); }, 0); }
  plotDiv.on('plotly_relayouting', function (event) {
    if (event['scene.camera']) { syncCompass(event['scene.camera']); }
  });
  plotDiv.on('plotly_relayout', function (event) {
    if (event['scene.camera']) { syncCompass(event['scene.camera']); }
  });
});
window.addEventListener('hashchange', function () {
  var start = fromHash();
  selectPoint(start[0], start[1]);
  if (start[2]) { viewAlong(start[2]); }
});
document.addEventListener('keydown', function (event) {
  if (event.target && /INPUT|SELECT|TEXTAREA/.test(event.target.tagName)) { return; }
  var n = DATA.points[state.point].modes.length;
  if (event.key === 'ArrowDown') {
    selectMode(Math.min(state.mode + 1, n - 1)); event.preventDefault();
  }
  if (event.key === 'ArrowUp') { selectMode(Math.max(state.mode - 1, 0)); event.preventDefault(); }
});
"""


def _scene_layout(conventional: bool) -> dict:
    """The 3D scene and the compass scene2 of the SALC viewer."""
    return {
        "scene": {
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "aspectmode": "data",
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "bgcolor": "#ffffff",
            "camera": {"eye": dict(CAMERA_EYE)},
        },
        "scene2": {
            "domain": (
                {"x": [0.02, 0.26], "y": [0.02, 0.34]}
                if conventional
                else {"x": [0.03, 0.19], "y": [0.02, 0.22]}
            ),
            "aspectmode": "cube",
            "xaxis": {"visible": False, "range": [-2.2, 2.2]},
            "yaxis": {"visible": False, "range": [-2.2, 2.2]},
            "zaxis": {"visible": False, "range": [-2.2, 2.2]},
            "bgcolor": "rgba(0,0,0,0)",
            "dragmode": False,
        },
        "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
        "paper_bgcolor": "#ffffff",
        "legend": {"x": 0.99, "y": 0.95, "xanchor": "right"},
        "showlegend": True,
        "uirevision": 0,
    }


def write_phonon_viewer_html(data: PhononViewer, output: str | None = None) -> None:
    """Write the standalone phonon mode viewer page.

    The page ships, per q point, the drawn atoms (element, Cartesian
    position, primitive atom index ``a``, lattice translation ``T``), the
    bonds and polyhedra as atom pairs and vertex lists, and per mode the
    atomic-gauge vectors ``u_a`` (:class:`ViewerMode`); the script rebuilds
    the displacement of atom ``(a, T)`` as ``Re[u_a exp(i(2 pi q.T - phi))]``
    times the amplitude (Angstrom), ``phi`` the animation phase.  The
    dispersion ships the bands per leg, the ticks and, per labeled level,
    the mode it selects.

    The page opens on the first q point and its first mode; the URL hash
    ``#q=R&mode=4&view=a`` selects the first q point named ``R`` (``G``,
    ``GM`` or ``GAMMA`` for Gamma), the mode of band 4 and the orthographic
    view down the a axis of the display cell (``b``, ``c``); ``#q=DT(GM-X)``
    selects the midpoint of the segment GM-X and ``#q=7`` the seventh q
    button.  The arrow keys step through the modes.

    Args:
        data: The :class:`PhononViewer` of :func:`phonon_viewer_data`.
        output: Path of the HTML file (default ``phonon_visualize.html``,
            ``phonon_visualize_all.html`` when ``data.all_irreps``).
    """
    if output is None:
        output = DEFAULT_OUTPUT_ALL if data.all_irreps else DEFAULT_OUTPUT
    from .visualize_basis import _axis_traces, _crystod_citation_html

    payload = _payload(data)
    payload["axes"] = _axis_traces(
        data.primitive_lattice, "abc", conventional_lattice=data.conventional_lattice
    )
    payload["headFraction"] = HEAD_FRACTION
    conventional = data.conventional_lattice is not None
    has_bonds = any(point.bonds for point in data.points)
    has_poly = any(point.polyhedra for point in data.points)
    bond_controls = ""
    if has_bonds:
        bond_controls += (
            "  <div class=\"control\"><label><input type=\"checkbox\" id=\"show-bonds\" checked\n"
            "           onchange=\"applyVisibility()\"/> show bonds</label></div>\n"
        )
    if has_poly:
        bond_controls += (
            "  <div class=\"control\"><label><input type=\"checkbox\" id=\"show-poly\" checked\n"
            "           onchange=\"applyVisibility()\"/> show polyhedra</label></div>\n"
        )
    run_notes = "".join(f"<div class=\"note\">NOTE: {note}</div>" for note in data.notes)
    title = f"{data.formula} phonon modes"
    page_title = (
        f"{data.formula} ({data.space_group}, #{data.number}): phonon modes along "
        f"{data.dispersion.path_string}" + (" (NAC)" if data.nac else "")
    )
    script = (
        _SCRIPT.replace("__CURVE__", CURVE)
        .replace("__IMAGINARY__", IMAGINARY)
        .replace("__ARROW__", ARROW)
        .replace("__CAMERA_EYE__", json.dumps(CAMERA_EYE))
    )
    html = (
        "<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"utf-8\"/>\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>\n"
        f"<title>{title}</title>\n"
        "<script src=\"https://cdn.plot.ly/plotly-2.32.0.min.js\"></script>\n"
        f"<style>\n{_CSS}</style>\n</head>\n<body>\n"
        "<div id=\"topbar\">\n"
        "  <div class=\"brand\">CrystOD<small>Phonon mode viewer</small></div>\n"
        f"  <div class=\"page-title\">{page_title}</div>\n"
        "</div>\n"
        "<div id=\"container\">\n"
        "<div id=\"sidebar\">\n"
        "  <h2>Structure</h2>\n"
        f"  <table class=\"info\">{_info_rows(data)}</table>\n"
        f"  {run_notes}\n"
        "  <h2>q point</h2>\n"
        "  <div id=\"qpoints\"></div>\n"
        "  <h2>Phonon modes (click to show)</h2>\n"
        "  <table id=\"mode-table\"></table>\n"
        "  <div id=\"point-notes\"></div>\n"
        "  <h2>Display</h2>\n"
        "  <div class=\"control\"><label>Amplitude</label>\n"
        "    <input type=\"range\" id=\"amplitude\" min=\"0.05\" max=\"1.5\" step=\"0.05\"\n"
        f"           value=\"{data.amplitude:g}\" oninput=\"onAmplitude(this.value)\"/>\n"
        f"    <span class=\"value\" id=\"amplitude-value\">{data.amplitude:.2f} &Aring;"
        "</span></div>\n"
        "  <div class=\"control\"><label><input type=\"checkbox\" id=\"animate\"\n"
        "           onchange=\"toggleAnimation(this.checked)\"/> animate</label>\n"
        "    <input type=\"range\" id=\"speed\" min=\"0.1\" max=\"2\" step=\"0.1\" value=\"0.5\"\n"
        "           oninput=\"onSpeed(this.value)\"/>\n"
        "    <span class=\"value\" id=\"speed-value\">0.5 Hz</span></div>\n"
        "  <div class=\"control\"><label><input type=\"checkbox\" id=\"show-arrows\" checked\n"
        "           onchange=\"applyVisibility()\"/> show arrows</label></div>\n"
        "  <div class=\"control\"><label><input type=\"checkbox\" id=\"show-cell\" checked\n"
        "           onchange=\"applyVisibility()\"/> show cell edges &amp; abc axes</label></div>\n"
        "  <div class=\"control\"><label><input type=\"checkbox\" id=\"show-atoms\" checked\n"
        "           onchange=\"applyVisibility()\"/> show atoms</label></div>\n"
        + bond_controls +
        "  <div class=\"control\"><label>View along</label>\n"
        "    <button class=\"vbtn\" onclick=\"viewAlong('a')\">a</button>\n"
        "    <button class=\"vbtn\" onclick=\"viewAlong('b')\">b</button>\n"
        "    <button class=\"vbtn\" onclick=\"viewAlong('c')\">c</button>\n"
        "    <button class=\"vbtn\" onclick=\"viewAlong('reset')\">reset</button></div>\n"
        "  <div class=\"note\" id=\"cell-line\"></div>\n"
        "  <div class=\"credit\">Viewer layout inspired by the\n"
        "    <a href=\"https://henriquemiranda.github.io/phononwebsite/\"\n"
        "       target=\"_blank\">phonon website</a>\n"
        "    by Henrique Miranda (BSD-3-Clause). 3D rendering and charts by plotly;\n"
        "    atom colors from VESTA. Irrep labels: ISO-IR (H. T. Stokes, B. J. Campbell\n"
        "    and R. Cordes, Acta Cryst. A69, 388 (2013)).<br><br>\n"
        f"    {_crystod_citation_html()}</div>\n"
        "</div>\n"
        "<div id=\"main\">\n"
        "  <div id=\"mode-title\"></div>\n"
        "  <div id=\"band\"></div>\n"
        "  <div id=\"plot\"></div>\n"
        "</div>\n"
        "</div>\n"
        "<script>\n"
        f"var DATA = {json.dumps(payload, separators=(',', ':'))};\n"
        f"var LAYOUT = {json.dumps(_scene_layout(conventional), separators=(',', ':'))};\n"
        f"{script}"
        "</script>\n</body>\n</html>\n"
    )
    with open(output, "w", encoding="utf-8") as handle:
        handle.write(html)


# --------------------------------------------------------------------------
# self-check of the shipped data
# --------------------------------------------------------------------------


def _check_reconstruction(phonon, data: PhononViewer, tol: float = 1e-6) -> tuple[float, int]:
    """Check the page's displacement fields against phonopy's modulation.

    For every q point and mode, the field the page draws (the shipped,
    rounded ``u`` and atom list, rebuilt with the page formula
    ``Re[u_a exp(2 pi i q.T)]``) must equal the real part of
    :func:`crystod.phonon_vector.get_supercell_displacement_field` on the
    display cell up to one overall real scale, atom by atom (the drawn atoms
    are matched to the supercell atoms by position modulo the display
    lattice, so the boundary duplicates and bonded images are checked with
    the phase of their own translation), and every supercell atom must be
    drawn.

    Args:
        phonon: The ``phonopy.Phonopy`` object ``data`` was computed from.
        data: The :class:`PhononViewer`.
        tol: Largest allowed deviation, relative to the largest displacement.

    Returns:
        ``(worst relative deviation, number of modes checked)``.

    Raises:
        AssertionError: A field deviates by more than ``tol``, or the drawn
            atoms do not cover the display cell.
    """
    from .phonon_vector import get_supercell_displacement_field

    payload = json.loads(json.dumps(_payload(data)))
    worst, count = 0.0, 0
    for point, shipped in zip(data.points, payload["points"]):
        q = np.array(shipped["q"], dtype=float)
        atom_index = np.array(shipped["atoms"]["a"], dtype=int)
        translations = np.array(shipped["atoms"]["T"], dtype=float)
        positions = np.array(shipped["atoms"]["p"], dtype=float)
        phases = np.exp(2j * np.pi * translations @ q)
        match = None
        for mode, shipped_mode in zip(point.modes, shipped["modes"]):
            u = np.array(shipped_mode["u"], dtype=float)
            u = u[..., 0] + 1j * u[..., 1]
            drawn = np.real(u[atom_index] * phases[:, None])
            supercell, field_ = get_supercell_displacement_field(
                phonon, point.q, point.cell_matrix, mode.vector
            )
            reference = field_.real
            if match is None:
                lattice = np.array(supercell.cell, dtype=float)
                frac_drawn = positions @ np.linalg.inv(lattice)
                frac_super = np.array(supercell.scaled_positions, dtype=float)
                diff = frac_drawn[:, None, :] - frac_super[None, :, :]
                diff -= np.rint(diff)
                distance = np.linalg.norm(diff @ lattice, axis=2)
                match = np.argmin(distance, axis=1)
                if np.max(distance[np.arange(len(match)), match]) > 2e-3:
                    raise AssertionError(
                        f"{point.name}: a drawn atom is not an atom of the display cell"
                    )
                if len(set(match.tolist())) != len(frac_super):
                    raise AssertionError(
                        f"{point.name}: the drawn atoms do not cover the display cell"
                    )
                s2uu = [supercell.u2u_map[x] for x in supercell.s2u_map]
                if any(s2uu[m] != a for m, a in zip(match, atom_index)):
                    raise AssertionError(
                        f"{point.name}: drawn atom matched to another primitive atom"
                    )
            target = reference[match]
            size = float(np.max(np.linalg.norm(reference, axis=1)))
            scale = float(np.sum(drawn * target) / max(np.sum(drawn * drawn), 1e-300))
            deviation = float(np.max(np.abs(scale * drawn - target))) / max(size, 1e-300)
            if deviation > tol:
                raise AssertionError(
                    f"{point.name} band {mode.band}: drawn field deviates by {deviation:.2e} "
                    "from phonopy's modulation"
                )
            worst = max(worst, deviation)
            count += 1
    return worst, count


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=desc, formatter_class=MyHelpFormatter)
    parser.add_argument(
        "--dim", required=True, type=str,
        help="Supercell dimension used for the force calculation.",
    )
    parser.add_argument("--poscar", type=str, default="POSCAR", help="POSCAR path.")
    parser.add_argument(
        "--readfc", action="store_true", help="Read FORCE_CONSTANTS instead of FORCE_SETS.",
    )
    parser.add_argument(
        "--nac", dest="nac", action="store_true",
        help="Apply the non-analytical term correction with ./BORN (without it\n"
        "a BORN file is not read).",
    )
    parser.add_argument(
        "--tolerance", "-tol", dest="tol", type=float, default=1e-3,
        help="Degeneracy tolerance (THz) of the irrep labels.",
    )
    parser.add_argument(
        "--qpoint", nargs="+", default=None,
        help="Show one q point only: a special-point label such as R or three\n"
        "primitive reciprocal coordinates (default: every labeled special point\n"
        "of the path).",
    )
    parser.add_argument(
        "--band", default=None,
        help="Manual k path (primitive reciprocal basis), e.g.\n"
        '"0 0 0  0 1/2 0  1/2 1/2 0, 1/2 1/2 0  1/2 1/2 1/2" (default: seekpath).',
    )
    parser.add_argument(
        "--band-labels", dest="band_labels", default=None,
        help="Names of the vertices of --band (default: their ISO-IR k-vector types).",
    )
    parser.add_argument(
        "--npoints", type=int, default=None, help="q points per leg of the path (default 51).",
    )
    parser.add_argument(
        "--threshold", type=float, default=-0.1,
        help="Frequency (THz) below which a level is drawn in red (imaginary).",
    )
    parser.add_argument(
        "--conventional", action="store_true",
        help="Display the conventional cell instead of the primitive cell.",
    )
    parser.add_argument(
        "--bond", nargs=3, action="append", default=None, metavar=("EL1", "EL2", "MAX"),
        help="Draw bonds between EL1 and EL2 atoms up to MAX Angstroms, plus the\n"
        "VESTA-style coordination polyhedra around the EL1 atoms (repeatable),\n"
        "e.g. --bond Sc F 2.3.",
    )
    parser.add_argument(
        "--all-irreps", dest="all_irreps", action="store_true",
        help="Also show the modes at the midpoints of the path segments (the symmetry\n"
        "lines DT, Z, SM, ... with their ISO-IR line irreps), as crystod-phonon\n"
        "--irreps --all-irreps labels them; the page is then\n"
        f"{DEFAULT_OUTPUT_ALL}. Not with --qpoint.",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help=f"Path of the HTML page (default {DEFAULT_OUTPUT}, with --all-irreps\n"
        f"{DEFAULT_OUTPUT_ALL}; {BAND_VECTOR_YAML} keeps its name).",
    )
    return parser


def _mode_table_lines(point: ViewerPoint) -> list[str]:
    """The terminal table of the modes of one q point (unindented)."""
    width = max([5] + [len(mode.irrep) for mode in point.modes])
    gamma = point.is_gamma
    header = f"{'Band':>4s}  {'Freq (THz)':>10s}  {'Irrep':<{width}s}  Comp."
    if gamma:
        mulliken_width = max([8] + [len(mode.mulliken or "") for mode in point.modes])
        header += f"  {'Mulliken':<{mulliken_width}s}  Activity"
    lines = [header]
    for mode in point.modes:
        line = (
            f"{mode.band:4d}  {_clean(mode.frequency, 4):10.4f}  "
            f"{mode.irrep:<{width}s}  {mode.component:5d}"
        )
        if gamma:
            line += f"  {mode.mulliken or '-':<{mulliken_width}s}  {mode.activity or '-'}"
        lines.append(line.rstrip())
    return lines


def main(argv: list[str] | None = None) -> None:
    from phonopy import load

    from .phonon_irreps import check_nac_loaded, nac_load_options
    from .vasp_io import read_poscar_cell

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.all_irreps and args.qpoint:
        parser.error("--all-irreps is not used with --qpoint (--qpoint shows one point).")
    output = args.output or (DEFAULT_OUTPUT_ALL if args.all_irreps else DEFAULT_OUTPUT)
    bonds: list[tuple[str, str, float]] = []
    for el1, el2, max_text in args.bond or []:
        try:
            bonds.append((el1, el2, float(max_text)))
        except ValueError:
            raise SystemExit(
                f"ERROR: --bond expects a numeric maximum length in Angstroms, got '{max_text}'."
            ) from None

    supercell_matrix = [float(n) for n in args.dim.split()]
    if args.readfc:
        force_sets, force_constants, force_data = None, "./FORCE_CONSTANTS", "FORCE_CONSTANTS"
    else:
        force_sets, force_constants, force_data = "./FORCE_SETS", None, "FORCE_SETS"
    phonon = load(
        supercell_matrix=supercell_matrix,
        primitive_matrix="auto",
        unitcell=read_poscar_cell(args.poscar),
        force_sets_filename=force_sets,
        force_constants_filename=force_constants,
        **nac_load_options(args.nac),
    )
    nac_note = check_nac_loaded(phonon, args.nac, "BORN")

    data = phonon_viewer_data(
        phonon,
        degeneracy_tolerance=args.tol,
        band=args.band,
        band_labels=args.band_labels,
        npoints=args.npoints,
        qpoint=args.qpoint,
        conventional=args.conventional,
        bonds=bonds,
        threshold=args.threshold,
        force_data=force_data,
        all_irreps=args.all_irreps,
    )
    n_qpoints = write_band_vector_yaml(
        phonon, band=args.band, band_labels=args.band_labels, npoints=args.npoints,
        filename=BAND_VECTOR_YAML,
    )
    if n_qpoints != data.dispersion.n_qpoints:  # the two runs share the path
        raise RuntimeError("band_vector.yaml and the dispersion disagree on the path")
    write_phonon_viewer_html(data, output)

    print("\n* Structure *")
    print(f"  Space group: {data.space_group} (#{data.number})")
    print(f"  Formula: {data.formula}")
    print(f"  Supercell: {data.supercell} ({force_data})")
    if nac_note:
        print(f"  {nac_note}")
    print("\n* Band path *")
    print(f"  k path ({data.path_source}): {data.dispersion.path_string}  "
          f"({data.dispersion.n_qpoints} q points)")
    print("  Labeled points: " + ", ".join(point.name for point in data.dispersion.points))
    if data.all_irreps:
        midpoints = [f"{point.name} ({point.segment})" for point in data.points
                     if point.is_midpoint]
        print("  Segment midpoints: " + (", ".join(midpoints) or "none"))
    for note in data.notes:
        print(f"  NOTE: {note}")
    for point in data.points:
        print(f"\n* Phonon modes at q = {point.name} {format_q_fraction(point.q)} *")
        for line in _mode_table_lines(point):
            print(line)
        if point.is_midpoint:
            print(f"  Segment: {point.segment}")
        print(f"  Display cell: {point.cell_description}")
        if point.x is None:
            print("  NOTE: this q point is not a labeled vertex of the path "
                  "(no marker on the dispersion)")
        for note in point.notes:
            print(f"  NOTE: {note}")
    print("\n* Output files *")
    print(f"  Phonon band structure with eigenvectors written to: {BAND_VECTOR_YAML}")
    print(f"  Phonon mode viewer written to: {output}")


if __name__ == "__main__":
    main()
