"""Phonon band unfolding onto the Brillouin zone of a parent structure.

The engine of ``crystod-phonon --unfold``.  The phonons of a distorted,
low-symmetry structure (the *child*, e.g. CaTiO3 Pnma) are projected onto the
Bloch states of the translation group of its high-symmetry *parent* (e.g.
CaTiO3 Pm-3m) and drawn along a path of the parent Brillouin zone: child mode
J, computed at the child wave vector K = M k, appears at the parent k with the
weight P_J(k) of its projection (Allen, Berlijn, Casavant and Soler,
Phys. Rev. B 87, 085322 (2013)).  The transformation between the two cells
and the origin shift are those of ``crystod-group --supergroup-cif PARENT
--subgroup-cif CHILD`` (:class:`crystod.symmetry_mode.SymmetryModeAnalysis`);
the unfolded levels are labelled with the irreps of the parent
(:mod:`crystod.phonon_unfold_irreps`) and drawn by
:mod:`crystod.phonon_unfold_plot`.

Conventions (every function, file and printed line of the feature)
------------------------------------------------------------------
* Lattice matrices hold the lattice vectors as rows (Angstrom), ``A[i] = a_i``.
* ``A_child = M @ A_parent`` with M an integer matrix, ``N = det M > 0``;
  ``A_parent`` is the parent primitive cell the unfolding uses.  phonopy's
  supercell matrix of the child relative to the parent is ``M.T``.
* Pseudo-parent cell ``A_pp = inv(M) @ A_child`` (child Cartesian frame).
  Parent k points are fractional coordinates of the reciprocal basis of the
  parent primitive cell; the same numbers refer to the reciprocal basis of
  ``A_pp``.
* Fractional k as columns: ``k_child = M @ k_parent``; row-stacked arrays:
  ``K = k @ M.T``.
* Positions: ``x_parent = M.T @ x_child + p`` (columns); p, the origin shift,
  is the child origin in parent fractional coordinates.  crystod-group prints
  the same setting as (P, p), ``x_parent = P x_child + p``, for the cells of
  the two input files (``P = M.T`` when the files are the unfolding cells).
* Frequencies in THz; imaginary frequencies are negative numbers; mode
  numbers are printed 1-based, array indices are 0-based.
* phonopy eigenvectors: mass-weighted, unit norm, atom-position (tau) phase
  convention, columns are bands.  The unfolded amplitude (ideal-site gauge,
  parent origin) of child mode J at the parent k is::

      u_J(k; kappa, alpha) = N^(-1/2) sum_{s: kappa(s) = kappa}
          exp(-2 pi i G.x0_s) exp(2 pi i K.d_s) e_J(K; s, alpha) exp(-2 pi i k.p)

  with ``M k = K + G``, x0_s the ideal site of child atom s and
  ``d_s = x_s - x0_s`` its displacement (minimum image); the weight is
  ``P_J(k) = sum_{kappa, alpha} |u_J|^2`` and ``sum_J P_J(k) = 3 n_parent``.
  The factor ``exp(2 pi i K.d_s)`` turns phonopy's actual-position phase into
  the ideal-site phase; without it the weights of a distorted child are wrong
  by up to 0.14.  phonopy's ``Unfolding`` keeps the actual-position phase and
  is not used.

Everything runs on the CPU (numpy/LAPACK): the matrices are at most a few
hundred square, and the checks of 1e-8 and below need float64.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import tempfile
import textwrap
import time
import warnings
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product

import numpy as np

from .spglib_compat import ensure_spglib_compat

ensure_spglib_compat()

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
DEGENERACY_TOLERANCE = 1e-4  # THz, single-linkage degenerate groups
WEIGHT_MIN = 0.02  # default of --weight-min: groups below are not drawn, label '-'
LEVEL_TABLE_MIN = 0.10  # groups printed in '* Unfolded irreps at special points *'
FRACTION_MIN = 0.5  # label rule: share of P_g a label must hold
SYMPREC = 1e-5  # spglib/phonopy tolerance on the phonon cells
MAPPING_TOLERANCE = 0.01  # default of --tolerance (SymmetryModeAnalysis, as crystod-group)
MAPPING_TOLERANCE_MAX = 0.5  # Angstrom; --tolerance must lie in (0, this]
MAX_DISPLACEMENT = 1.8  # Angstrom, largest displacement of an accepted atom map
NPOINTS = 51  # default of --npoints (points per path leg, inclusive)
SUM_RULE_TOLERANCE = 1e-6  # |sum_J P_J(k) - 3 n_parent| above this: internal error
PROJECTOR_TOLERANCE = 1e-8  # projector identities above this: internal error
ORBIT_SPECTRUM_TOLERANCE = (1e-4, 1e-4)  # THz, weight: arms of one class; above: NOTE
CHUNK = 512  # k points per phonopy run_qpoints call

TWO_PI_I = 2j * np.pi
_POINT_GROUP_WARNING = ".*Point group symmetries of supercell and primitivecell are different"
_IMAGES = np.array(list(product((-1, 0, 1), repeat=3)), dtype=float)

CONVENTIONS = (
    "rows = lattice vectors (Angstrom); A_child = M @ A_parent, N = det M, A_parent = the parent "
    "primitive cell of the unfolding; A_pp = inv(M) @ A_child (pseudo-parent, child Cartesian "
    "frame); k_parent fractional in the reciprocal basis of the parent primitive cell (= that of "
    "A_pp); K_child = k_parent @ M.T (k_child = M k_parent); x_parent = M.T x_child + "
    "origin_shift (columns); setting_P/setting_p = the same setting as crystod-group prints it "
    "for the input files (columns of P = child basis in parent units, x_parent = P x_child + p); "
    "frequencies THz, imaginary negative; mode index 0-based in arrays; weights = sum |u|^2 of "
    "the unfolded vectors u_J(k; kappa, alpha) = N^-1/2 sum_s exp(-2 pi i G.x0_s) exp(2 pi i "
    "K.d_s) e_J(K; s, alpha) exp(-2 pi i k.p) (phonopy mass-weighted eigenvectors, ideal-site "
    "gauge, parent origin, alpha Cartesian in the CHILD frame; v_parent = frame_rotation @ "
    "v_child); weights_canonical, vectors_canonical: degenerate child modes (|df| <= 1e-4 THz) "
    "rotated so that their unfolded vectors are orthogonal (basis independent; group sums "
    "unchanged; inside a set of two levels closer than 1e-4 THz the pairing of a per-mode weight "
    "with a frequency is arbitrary); per level (every mode carries its level's value): "
    "group_weight = P_g of the mode's degenerate group, group_label, group_fraction, "
    "gamma_group_label, gamma_child_irrep; distances 1/Angstrom without "
    "2 pi in the reciprocal metric of A_pp; arm_*: every arm of the stars of the path vertices; "
    "davg_*: domain average, one row per class of equivalent arms of the star of a path k "
    "(davg_weights = canonical weight of the class representative x multiplicity / n_arms); "
    "irrep_weights[k, J, u] = ||Pi_u(k) R u_J(k)||^2 with the CrystOD projectors of the parent; "
    "group_label = label of the degenerate group (>= 50 % of P_g in one irrep unit, else the "
    "smallest set the child symmetry cannot separate written A/B, else 'mixed'; '-' below the "
    "weight cut); gamma_*: child Gamma modes unfolded onto the parent k folding onto the child "
    "Gamma point (gamma_weights canonical at each of these k, as weights_canonical; the weights "
    "of a level add up to its number of modes over them)"
)


# ---------------------------------------------------------------------------
# small text and phonopy helpers
# ---------------------------------------------------------------------------
def _frac_text(value, max_denominator: int = 24) -> str:
    """``1/2``, ``-1``, ``0``; a value that is no such fraction with four decimals."""
    value = float(value)
    fraction = Fraction(value).limit_denominator(max_denominator)
    if abs(float(fraction) - value) < 1e-6:
        return str(fraction)
    return f"{value + 0.0:.4f}"


def _vec(values) -> str:
    """Fractional coordinates as ``(1/2, 0, 1/2)`` (vibration_modes.format_fraction_vector)."""
    from .vibration_modes import format_fraction_vector

    return format_fraction_vector(
        [float(v) + 0.0 for v in np.asarray(values, dtype=float).reshape(-1)]
    )


def _matrix_text(M) -> str:
    """An integer matrix row-wise, rows separated by two spaces: ``1 0 1  0 2 0  -1 0 1``."""
    rows = np.rint(np.asarray(M, dtype=float)).astype(int).reshape(3, 3)
    return "  ".join(" ".join(str(int(v)) for v in row) for row in rows)


def _thz(value: float, width: int = 9) -> str:
    """A frequency with four decimals; -0.0000 is printed as 0.0000."""
    return f"{round(float(value), 4) + 0.0:{width}.4f}"


def _domains(n: int) -> str:
    return f"{n} orientation domain{'s' if n != 1 else ''}"


def _note(text: str) -> list:
    """A NOTE under a block: ``'  NOTE: ...'`` wrapped to 100 columns, continuation lines
    aligned with the text after ``NOTE:``."""
    return textwrap.wrap(
        text,
        width=100,
        initial_indent="  NOTE: ",
        subsequent_indent=" " * 8,
        break_on_hyphens=False,
    )


def _kv(key: str, text: str) -> str:
    """One summary line of a block: the key padded to 15 characters."""
    return f"{key:<15}{text}" if len(key) < 15 else f"{key} {text}"


def _display_path(path: str) -> str:
    """A path as the user reads it: relative to the working directory when it lies below it."""
    relative = os.path.relpath(path)
    return path if relative.startswith("..") else relative


def _cell_tuple(cell):
    """``(lattice, fractional positions, atomic numbers)`` of a PhonopyAtoms."""
    return (
        np.array(cell.cell, dtype=float),
        np.array(cell.scaled_positions, dtype=float),
        np.array(cell.numbers, dtype=int),
    )


def _dataset_value(dataset, name):
    return getattr(dataset, name) if hasattr(dataset, name) else dataset[name]


def _space_group(cell, symprec: float = SYMPREC):
    """``(international symbol, number)`` of a cell (spglib)."""
    import spglib

    dataset = spglib.get_symmetry_dataset(_cell_tuple(cell), symprec=symprec)
    if dataset is None:
        return "P1", 1
    return str(_dataset_value(dataset, "international")), int(_dataset_value(dataset, "number"))


def _run_qpoints(phonon, qpoints, **kwargs):
    """``phonon.run_qpoints(qpoints, **kwargs)`` and its result object for every phonopy version.

    phonopy 4.3 returns the QpointsPhonon object, phonopy 2.x returns None and
    keeps it in the ``qpoints`` property (crystod.runtime_compat.get_qpoints_result).
    """
    result = phonon.run_qpoints(np.asarray(qpoints, dtype=float), **kwargs)
    if result is not None and hasattr(result, "frequencies"):
        return result
    from .runtime_compat import get_qpoints_result

    return get_qpoints_result(phonon)


@contextlib.contextmanager
def _empty_working_directory():
    """Run the block inside a new, empty temporary directory (removed afterwards).

    phonopy.load reads ``./FORCE_CONSTANTS`` (and ``./force_constants.hdf5``) of
    the working directory whenever it finds one and then ignores the
    FORCE_SETS it was given; with two structures from two directories that
    would silently mix the force data.  Every load of ``--unfold`` therefore
    gives phonopy absolute paths and an empty working directory.
    """
    previous = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="crystod_unfold_") as empty:
        os.chdir(empty)
        try:
            yield empty
        finally:
            os.chdir(previous)


# ---------------------------------------------------------------------------
# lattice and k-point helpers
# ---------------------------------------------------------------------------
def _as_supercell_matrix(M) -> np.ndarray:
    """M as an int64 (3, 3) array; integer entries and det M > 0 are required."""
    Mf = np.array(M, dtype=float)
    if Mf.size == 9:
        Mf = Mf.reshape(3, 3)
    if Mf.shape != (3, 3):
        raise SystemExit(f"ERROR: M must be a 3x3 matrix, got {Mf.size} values.")
    Mi = np.rint(Mf).astype(np.int64)
    if np.abs(Mf - Mi).max() > 1e-6:
        raise SystemExit("ERROR: M must be an integer matrix (A_child = M A_parent).")
    det = int(round(np.linalg.det(Mi)))
    if det <= 0:
        raise SystemExit(
            "ERROR: det M must be positive (A_child = M A_parent, a right-handed child cell), got "
            f"{det}."
        )
    return Mi


def _reciprocal_lattice(A) -> np.ndarray:
    """Reciprocal lattice vectors as rows, without 2 pi: ``B = inv(A).T``."""
    return np.linalg.inv(np.asarray(A, dtype=float)).T


def _snap_rational(values, max_denominator: int = 48, tol: float = 1e-8) -> np.ndarray:
    """Entries that are rational numbers (denominator <= 48) snapped to exact floats."""
    array = np.array(values, dtype=float)
    flat = array.reshape(-1)
    for i, value in enumerate(flat):
        fraction = Fraction(float(value)).limit_denominator(max_denominator)
        if abs(float(fraction) - value) < tol:
            flat[i] = float(fraction)
    return flat.reshape(array.shape) + 0.0


def _reduce_half(k) -> np.ndarray:
    """Fractional k reduced into (-1/2, 1/2] (X is (0, 1/2, 0), not (0, -1/2, 0))."""
    k = np.asarray(k, dtype=float)
    out = k - np.ceil(k - 0.5 - 1e-9)
    out[np.abs(out) < 1e-12] = 0.0
    return out + 0.0


def _reduce_unit(k) -> np.ndarray:
    """Fractional k reduced into [0, 1)."""
    k = np.asarray(k, dtype=float)
    out = k - np.floor(k + 1e-9)
    out[np.abs(out) < 1e-12] = 0.0
    return out + 0.0


def _same_modulo_lattice(a, b, tol: float = 1e-8) -> bool:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return bool(np.abs(d - np.rint(d)).max() < tol)


def _commensurate_points(M) -> np.ndarray:
    """The N parent k points g (parent fractional, rows, in [0, 1)^3) with M @ g integer."""
    Mi = _as_supercell_matrix(M)
    N = int(round(np.linalg.det(Mi)))
    Minv = np.linalg.inv(Mi.astype(float))
    # Z^3 / M Z^3 has N cosets and N Z^3 lies in M Z^3: n in {0..N-1}^3 meets every coset
    n = np.array(list(product(range(N), repeat=3)), dtype=float)
    g = n @ Minv.T
    g -= np.floor(g + 1e-9)
    g[np.abs(g) < 1e-12] = 0.0
    kept = {}
    for v in g:
        kept.setdefault(tuple(np.round(v, 9)), v)
    points = np.array([kept[key] for key in sorted(kept)])
    if len(points) != N:
        raise SystemExit(
            f"ERROR: internal error: {len(points)} commensurate points, expected N = {N}."
        )
    return points


def _parent_kpoints_folding_to_gamma(M) -> np.ndarray:
    """The N parent k points (reduced into (-1/2, 1/2]^3) with M k integer.

    Sorted by the number of non-zero components, then by descending absolute
    values (GM, X, M, R for a cubic parent).
    """
    points = [_snap_rational(_reduce_half(g)) for g in _commensurate_points(M)]
    points.sort(key=lambda k: (int(np.sum(np.abs(k) > 1e-9)), tuple(-np.abs(k))))
    return np.array(points, dtype=float)


# ---------------------------------------------------------------------------
# degenerate modes and Cartesian frames
# ---------------------------------------------------------------------------
def degenerate_groups(freqs_1d, tol: float = DEGENERACY_TOLERANCE) -> list:
    """Index sets of degenerate modes at one k (single linkage on the sorted
    frequencies: consecutive values closer than ``tol`` THz belong together).

    Returns:
        A list of int arrays, the groups in ascending frequency.
    """
    f = np.asarray(freqs_1d, dtype=float)
    order = np.argsort(f, kind="stable")
    cuts = np.where(np.diff(f[order]) > tol)[0] + 1
    return [np.sort(g) for g in np.split(order, cuts)]


def degenerate_group_ids(freqs, tol: float = DEGENERACY_TOLERANCE) -> np.ndarray:
    """Group index of every mode, (n_k, n_band) (0, 1, ... in ascending frequency at each k)."""
    f = np.atleast_2d(np.asarray(freqs, dtype=float))
    ids = np.zeros(f.shape, dtype=np.int64)
    for ik in range(f.shape[0]):
        for ig, g in enumerate(degenerate_groups(f[ik], tol)):
            ids[ik, g] = ig
    return ids


def _sum_weights_by_group(weights, group_ids) -> np.ndarray:
    """Summed weight of every group, returned per mode (each mode carries its group's total)."""
    w = np.atleast_2d(np.asarray(weights, dtype=float))
    gid = np.atleast_2d(group_ids)
    out = np.zeros_like(w)
    for ik in range(w.shape[0]):
        total = np.bincount(gid[ik], weights=w[ik])
        out[ik] = total[gid[ik]]
    return out


def canonical_degenerate_rotation(freqs, uvec, tol: float = DEGENERACY_TOLERANCE):
    """Rotate every set of degenerate child modes so that their unfolded vectors are orthogonal.

    For a set g (``|df| <= tol`` THz, single linkage) at one k the overlap
    ``Q_ab = <u_a|u_b>`` is Hermitian with eigenvalues in [0, 1];
    ``Q = W diag(lam) W^H`` (lam descending) gives the rotated modes
    ``E'[:, g] = E[:, g] @ W`` with unfolded vectors ``u'_g = W.T @ u_g`` and
    weights lam.  The result does not depend on the basis the eigensolver chose
    inside the set; the group sums are unchanged.

    The rotated modes take the places of the set's members in index order (the
    largest weight at the first member, the lowest frequency).  Within an
    exactly degenerate level that is no choice; when single linkage joins two
    levels closer than ``tol`` the rotated modes mix them, and the pairing of a
    per-mode weight with a member's frequency is arbitrary (a weight of 1 can
    sit at the frequency of the other level, less than ``tol`` away): use the
    group values (``P_g``, ``group_weight``), not the per-mode weights, inside
    such a set.

    Args:
        freqs: (n_k, n_band) THz.
        uvec: (n_k, n_band, 3 n_parent) unfolded vectors.

    Returns:
        ``(uvec_rot, weights_rot, rotations)``; ``rotations[ik]`` lists
        ``(indices, W)`` for the sets of more than one mode.
    """
    f = np.atleast_2d(np.asarray(freqs, dtype=float))
    u = np.array(uvec, dtype=complex, copy=True)
    rotations = []
    for ik in range(f.shape[0]):
        rot_k = []
        for g in degenerate_groups(f[ik], tol):
            if len(g) < 2:
                continue
            V = u[ik, g, :]
            Q = V.conj() @ V.T
            _lam, W = np.linalg.eigh(0.5 * (Q + Q.conj().T))
            W = W[:, ::-1]
            u[ik, g, :] = W.T @ V
            rot_k.append((g, W))
        rotations.append(rot_k)
    weights = np.sum(np.abs(u) ** 2, axis=-1)
    return u, weights, rotations


def frame_rotation(A_pp, A_parent) -> np.ndarray:
    """Proper rotation that carries Cartesian vectors from the child frame to the parent frame.

    ``v_parent = R @ v_child``.  Polar decomposition (column form)
    ``a_pp_i = Q U a_parent_i`` with Q a proper rotation and U symmetric
    positive (the strain); ``R = Q.T``.  The strain is not removed: for a
    strained pseudo-parent this is the best rigid rotation, exact when A_pp is
    a rotated copy of A_parent.
    """
    A_pp = np.asarray(A_pp, dtype=float)
    A_par = np.asarray(A_parent, dtype=float)
    F = np.linalg.solve(A_par, A_pp)  # A_pp = A_parent @ F (rows)
    W, _s, Vt = np.linalg.svd(F.T)  # F.T = Q U
    Q = W @ Vt
    if np.linalg.det(Q) < 0:
        raise SystemExit(
            "ERROR: internal error: the pseudo-parent cell and the parent cell have opposite "
            "handedness."
        )
    return Q.T


def rotate_unfolded_vectors(uvec, rotation) -> np.ndarray:
    """Apply a Cartesian rotation to every (kappa, alpha) triplet of the last axis."""
    u = np.asarray(uvec)
    shape = u.shape
    return np.einsum(
        "ab,...pb->...pa", np.asarray(rotation), u.reshape(shape[:-1] + (shape[-1] // 3, 3))
    ).reshape(shape)


# ---------------------------------------------------------------------------
# the unfolding projection
# ---------------------------------------------------------------------------
def unfold_eigenvectors(
    eigvecs,
    K_used,
    k_parent,
    M,
    kappa,
    x_actual,
    x_ideal,
    *,
    gauge: str = "ideal",
    origin_shift=None,
):
    """Unfold given child eigenvectors onto parent k points (the numpy core of :func:`unfold`).

    Args:
        eigvecs: (n_k, 3 n_child, n_band) complex, phonopy layout (columns are
            bands), computed at the child wave vectors ``K_used``.
        K_used: (n_k, 3) child fractional wave vectors of ``eigvecs``,
            congruent to ``M @ k_parent`` modulo integers.
        k_parent: (n_k, 3) parent fractional k.
        M: ``A_child = M @ A_parent``.
        kappa: (n_child,) parent atom of every child atom.
        x_actual: (n_child, 3) child fractional positions of phonopy's cell.
        x_ideal: (n_child, 3) ideal positions (child fractional; the lattice
            image next to each actual atom is used).
        gauge: ``"ideal"`` (exact projection on the parent translations) or
            ``"actual"`` (diagnostic: phonopy's actual-position phase kept).
        origin_shift: p of ``x_parent = M.T x_child + p``; the global phase
            ``exp(-2 pi i k.p)`` refers the vectors to the parent origin.

    Returns:
        ``(uvec (n_k, n_band, 3 n_parent), weights (n_k, n_band), G (n_k, 3) int)``.
    """
    Mi = _as_supercell_matrix(M)
    N = int(round(np.linalg.det(Mi)))
    E = np.asarray(eigvecs)
    k_par = np.atleast_2d(np.asarray(k_parent, dtype=float))
    K = np.atleast_2d(np.asarray(K_used, dtype=float))
    kappa = np.asarray(kappa, dtype=np.int64)
    n_k, n3, n_band = E.shape
    n_child = n3 // 3
    n_parent = n_child // N
    k_c = k_par @ Mi.T.astype(float)
    Gf = k_c - K
    G = np.rint(Gf)
    if np.abs(Gf - G).max() > 1e-8:
        raise ValueError("K_used is not congruent to M @ k_parent modulo integers")
    x = np.asarray(x_actual, dtype=float)
    d = x - np.asarray(x_ideal, dtype=float)
    d -= np.rint(d)
    x0 = x - d
    if gauge == "ideal":
        phase = np.exp(TWO_PI_I * (K @ d.T)) * np.exp(-TWO_PI_I * (G @ x0.T))
    elif gauge == "actual":
        phase = np.exp(-TWO_PI_I * (G @ x.T))
    else:
        raise ValueError("gauge must be 'ideal' or 'actual'")
    if origin_shift is not None:
        phase = phase * np.exp(-TWO_PI_I * (k_par @ np.asarray(origin_shift, dtype=float)))[:, None]
    C = np.zeros((n_parent, n_child))
    C[kappa, np.arange(n_child)] = 1.0
    Ek = E.reshape(n_k, n_child, 3, n_band) * phase[:, :, None, None]
    u = np.einsum("ps,ksab->kpab", C, Ek) / np.sqrt(N)
    uvec = np.ascontiguousarray(u.reshape(n_k, 3 * n_parent, n_band).transpose(0, 2, 1))
    weights = np.sum(np.abs(uvec) ** 2, axis=-1)
    return uvec, weights, G.astype(np.int64)


# ---------------------------------------------------------------------------
# atom map helpers
# ---------------------------------------------------------------------------
def _min_image(dfrac, A):
    """Minimum-image Cartesian vectors of fractional differences (..., 3) in the cell A (rows).

    Returns ``(cartesian (..., 3), integer shift (..., 3))`` with
    ``cartesian = (dfrac + shift) @ A``.
    """
    base = -np.rint(dfrac)
    d0 = dfrac + base
    candidates = d0[..., None, :] + _IMAGES
    cart = candidates @ A
    norms = np.linalg.norm(cart, axis=-1)
    index = np.argmin(norms, axis=-1)
    selected = np.take_along_axis(cart, index[..., None, None], axis=-2)[..., 0, :]
    shift = base + _IMAGES[index]
    return selected, np.rint(shift).astype(np.int64)


def _coset_translations(M) -> np.ndarray:
    """The N parent lattice vectors t (integer, parent fractional) with inv(M.T) @ t in [0, 1)^3.

    They label the N copies of the parent cell inside the child cell; sorted by
    their child fractional coordinates.
    """
    M = np.asarray(M, dtype=np.int64)
    n = int(round(abs(np.linalg.det(M))))
    MT_inv = np.linalg.inv(M.T.astype(float))
    bound = int(np.abs(M).sum(axis=0).max()) + 1
    found = {}
    for t in product(range(-bound, bound + 1), repeat=3):
        y = MT_inv @ np.array(t, dtype=float)
        if np.all(y > -1e-9) and np.all(y < 1.0 - 1e-9):
            found.setdefault(tuple(np.round(y, 8)), np.array(t, dtype=np.int64))
    if len(found) != n:
        raise SystemExit(
            f"ERROR: internal error: {len(found)} parent translations in the child cell, expected "
            f"N = {n}."
        )
    return np.array([found[key] for key in sorted(found)], dtype=np.int64)


def _translation_permutations(ideal_frac_child, species, M, tol: float = 1e-6) -> np.ndarray:
    """``perm[m, i] = j``: ideal site of child atom i moved by parent translation m is that of j.

    Raises SystemExit when a parent translation does not permute the ideal
    sites (the ideal structure would not be a parent superstructure).
    """
    translations = _coset_translations(M)
    MT_inv = np.linalg.inv(np.asarray(M, dtype=float).T)
    x = np.asarray(ideal_frac_child, dtype=float)
    sp = np.array(species)
    perm = np.zeros((len(translations), len(x)), dtype=np.int64)
    for m, t in enumerate(translations):
        y = x + MT_inv @ t
        d = y[:, None, :] - x[None, :, :]
        d -= np.rint(d)
        hit = (np.abs(d).max(axis=-1) < tol) & (sp[:, None] == sp[None, :])
        if not np.all(hit.sum(axis=1) == 1):
            raise SystemExit(
                f"ERROR: internal error: the parent translation {tuple(int(v) for v in t)} does "
                "not permute the ideal sites of the child."
            )
        perm[m] = np.argmax(hit, axis=1)
    return perm


def _lattice_parameters(A):
    """``(lengths a, b, c in Angstrom, angles alpha, beta, gamma in degrees)``."""
    A = np.asarray(A, dtype=float)
    lengths = np.linalg.norm(A, axis=1)

    def angle(i, j):
        c = A[i] @ A[j] / (lengths[i] * lengths[j])
        return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))

    return lengths, np.array([angle(1, 2), angle(0, 2), angle(0, 1)])


def _strain_and_rotation(A_pp, A_parent) -> dict:
    """Strain and rigid rotation of the pseudo-parent cell against the parent cell.

    ``A_pp = A_parent @ F`` (rows); in column form ``a_pp = F.T a_parent =
    Q U a_parent`` (polar decomposition, Q proper rotation, U symmetric).  The
    principal strains are eig(U) - 1; the rotation angle compares the two
    cells as oriented in their own files.
    """
    A_pp = np.asarray(A_pp, dtype=float)
    A_parent = np.asarray(A_parent, dtype=float)
    F = np.linalg.solve(A_parent, A_pp)
    W, s, Vt = np.linalg.svd(F.T)
    Q = W @ Vt
    if np.linalg.det(Q) < 0:
        W[:, -1] *= -1
        s[-1] *= -1
        Q = W @ Vt
    U = Vt.T @ np.diag(s) @ Vt
    principal = np.sort(np.linalg.eigvalsh(0.5 * (U + U.T)) - 1.0)
    angle = float(np.degrees(np.arccos(np.clip((np.trace(Q) - 1.0) / 2.0, -1.0, 1.0))))
    l_pp, ang_pp = _lattice_parameters(A_pp)
    l_p, ang_p = _lattice_parameters(A_parent)
    return {
        "lengths_pp": l_pp.tolist(),
        "angles_pp": ang_pp.tolist(),
        "parent_lengths": l_p.tolist(),
        "parent_angles": ang_p.tolist(),
        "length_strain_percent": ((l_pp / l_p - 1.0) * 100.0).tolist(),
        "angle_deviation_deg": (ang_pp - ang_p).tolist(),
        "principal_strains_percent": (principal * 100.0).tolist(),
        "volume_strain_percent": float(
            (abs(np.linalg.det(A_pp)) / abs(np.linalg.det(A_parent)) - 1.0) * 100.0
        ),
        "rotation_deg": angle,
    }


def _validate_atom_map(
    M, kappa, x_actual, x_ideal, A_child, symbols, parent_frac, origin_shift, tol: float = 1e-6
) -> dict:
    """Check a child -> parent atom map; return the bookkeeping arrays.

    Every kappa = 0..n_parent-1 owns N = det M child atoms of one species; the
    ideal sites of one kappa differ by parent lattice vectors and lie in N
    different cosets; all ideal sites are parent sites for one common origin
    shift (x_parent = M.T x0 + p), which must be ``origin_shift`` modulo 1.
    Raises SystemExit on a violation.
    """
    Mi = _as_supercell_matrix(M)
    N = int(round(np.linalg.det(Mi)))
    kappa = np.asarray(kappa, dtype=np.int64)
    x = np.asarray(x_actual, dtype=float)
    n_child = len(x)
    n_parent = n_child // N
    fail = "ERROR: internal error: the child -> parent atom map is not valid: "
    if sorted(set(kappa.tolist())) != list(range(n_parent)):
        raise SystemExit(fail + f"kappa does not take every value 0..{n_parent - 1}.")
    d = x - np.asarray(x_ideal, dtype=float)
    d -= np.rint(d)
    x0 = x - d
    MT = Mi.T.astype(float)
    members = np.zeros((n_parent, N), dtype=np.int64)
    for kap in range(n_parent):
        s_k = np.where(kappa == kap)[0]
        if len(s_k) != N:
            raise SystemExit(
                fail + f"parent atom {kap + 1} has {len(s_k)} child atoms, expected N = {N}."
            )
        if len({symbols[s] for s in s_k}) != 1:
            raise SystemExit(
                fail + f"parent atom {kap + 1} is mapped to child atoms of different species."
            )
        dx = x0[s_k] - x0[s_k[0]]
        t = dx @ MT.T
        if np.abs(t - np.rint(t)).max() > tol:
            raise SystemExit(
                fail + f"the ideal sites of parent atom {kap + 1} are not related by parent "
                "lattice vectors."
            )
        dd = dx[:, None, :] - dx[None, :, :]
        same = np.abs(dd - np.rint(dd)).max(axis=-1) < tol
        if same.sum() != N:
            raise SystemExit(
                fail + f"the ideal sites of parent atom {kap + 1} are not N distinct cosets."
            )
        members[kap] = s_k
    tau = np.asarray(parent_frac, dtype=float)
    p_all = tau[kappa] - x0 @ MT.T
    dp = p_all - p_all[0]
    if np.abs(dp - np.rint(dp)).max() > tol:
        raise SystemExit(fail + "the ideal sites are not parent sites for one common origin shift.")
    dp0 = np.asarray(origin_shift, dtype=float) - p_all[0]
    if np.abs(dp0 - np.rint(dp0)).max() > tol:
        raise SystemExit(fail + "the ideal sites do not match the origin shift of the setting.")
    return {"N": N, "n_parent": n_parent, "child_atoms_of_kappa": members}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
@dataclass
class UnfoldInputs:
    """The child and the parent as :func:`load_unfold_inputs` reads them.

    Attributes:
        child_poscar, parent_poscar: The structure files as given.
        child_force_file: Absolute path of the child's FORCE_SETS (or
            FORCE_CONSTANTS) that was read.
        parent_force_file: The same for the parent; None without ``parent_dim``.
        child_dim, parent_dim: Diagonal supercells of the force calculations
            (``parent_dim`` None without parent force data).
        child_readfc, parent_readfc: FORCE_CONSTANTS instead of FORCE_SETS.
        child_unitcell, parent_unitcell: The cells of the files (PhonopyAtoms,
            read with :func:`crystod.vasp_io.read_poscar_cell`).
        child_primitive_matrix, parent_primitive_matrix: phonopy primitive
            matrices (``A_prim = P.T @ A_unit``); the identity when a cell is
            used as given.
        child_cell: The child unfolding cell (``child_phonon.primitive``).
        parent_cell: The parent primitive cell of the unfolding.
        child_phonon: ``phonopy.Phonopy`` with the child force constants.
        parent_phonon: The same for the parent, or None without ``parent_dim``.
    """

    child_poscar: str
    parent_poscar: str
    child_force_file: str
    parent_force_file: str | None
    child_dim: tuple
    parent_dim: tuple | None
    child_readfc: bool
    parent_readfc: bool
    child_unitcell: object
    parent_unitcell: object
    child_primitive_matrix: np.ndarray
    parent_primitive_matrix: np.ndarray
    child_cell: object
    parent_cell: object
    child_phonon: object
    parent_phonon: object | None


def _dim_tuple(dim, name: str) -> tuple:
    """Three positive integers from ``"2 2 2"``, ``[2, 2, 2]`` or ``(2, 2, 2)``."""
    tokens = dim.split() if isinstance(dim, str) else list(np.asarray(dim).reshape(-1))
    try:
        values = tuple(int(str(v)) if isinstance(v, str) else int(v) for v in tokens)
    except (TypeError, ValueError):
        raise SystemExit(f"ERROR: {name} needs three positive integers, got {dim!r}.") from None
    if len(values) != 3 or any(v <= 0 for v in values):
        raise SystemExit(f"ERROR: {name} needs three positive integers, got {dim!r}.")
    return values


def _read_unitcell(path: str):
    """The cell of a POSCAR (crystod.vasp_io.read_poscar_cell); a clean ERROR otherwise."""
    from .vasp_io import read_poscar_cell

    if not os.path.isfile(path):
        raise SystemExit(f"ERROR: '{path}' does not exist.")
    try:
        return read_poscar_cell(path)
    except (OSError, ValueError, IndexError, KeyError) as exc:
        raise SystemExit(f"ERROR: could not read '{path}': {' '.join(str(exc).split())}") from None


def _unfolding_primitive_matrix(unitcell, role: str) -> np.ndarray:
    """phonopy primitive matrix of the unfolding cell.

    Parent: phonopy's ``"auto"`` (as ``--irreps`` and ``--fatband``).  Child:
    the cell as given when it is primitive (phonopy's guessed primitive matrix
    has ``|det| = 1``, even if it would permute the axes), else the guessed matrix.
    """
    from phonopy.structure.cells import guess_primitive_matrix

    matrix = np.asarray(guess_primitive_matrix(unitcell, symprec=SYMPREC), dtype=float)
    if role == "child" and abs(abs(np.linalg.det(matrix)) - 1.0) < 1e-8:
        return np.eye(3)
    return matrix


def _primitive_cell(unitcell, matrix):
    """The primitive cell of phonopy for ``unitcell`` and a primitive matrix (no force data)."""
    from phonopy import Phonopy

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="primitive_matrix defaulted to 'auto'")
        warnings.filterwarnings("ignore", message=_POINT_GROUP_WARNING)
        return Phonopy(
            unitcell,
            supercell_matrix=np.eye(3, dtype=int),
            primitive_matrix=matrix,
            symprec=SYMPREC,
            log_level=0,
        ).primitive


def _force_file(poscar: str, readfc: bool, role: str) -> str:
    """Absolute path of the force file of one structure.

    Next to its POSCAR: ``NAME_<POSCAR file name>`` (the name macer gives the
    force data of each structure, e.g. FORCE_SETS_POSCAR_child), then ``NAME``;
    the child also in the working directory, in the same order.  NAME is
    FORCE_SETS, or FORCE_CONSTANTS with ``readfc``.  The per-structure name
    lets a child and a parent POSCAR share one directory.
    """
    name = "FORCE_CONSTANTS" if readfc else "FORCE_SETS"
    own = f"{name}_{os.path.basename(poscar)}"
    directories = [os.path.dirname(os.path.abspath(poscar))]
    if role == "child" and os.path.abspath(os.curdir) not in directories:
        directories.append(os.path.abspath(os.curdir))
    for directory in directories:
        for candidate in (own, name):
            path = os.path.join(directory, candidate)
            if os.path.isfile(path):
                return path
    flag = "--child-readfc" if role == "child" else "--parent-readfc"
    other = f"FORCE_SETS without {flag}" if readfc else f"FORCE_CONSTANTS with {flag}"
    if role == "child":
        raise SystemExit(
            f"ERROR: {name} not found next to '{poscar}' (as {own} or {name}) or in the current "
            f"directory (the child's force data; {other})."
        )
    raise SystemExit(
        f"ERROR: {name} not found next to '{poscar}' (as {own} or {name}; the parent's force "
        f"data, read with --parent-dim; {other})."
    )


def _force_file_atoms(force_file: str, readfc: bool):
    """Number of supercell atoms a force file is for, from its header: the first line of a
    FORCE_SETS (``natom``) or of a FORCE_CONSTANTS (``n_rows n_atoms``); None when the file does
    not say it (FORCE_SETS of phonopy's type 2, one line per atom and displacement)."""
    try:
        with open(force_file) as handle:
            for line in handle:
                tokens = line.split()
                if tokens:
                    break
            else:
                return None
    except (OSError, UnicodeDecodeError):
        return None
    try:
        if readfc and len(tokens) == 2:
            return int(tokens[1])
        if not readfc and len(tokens) == 1:
            return int(tokens[0])
    except ValueError:
        return None
    return None


def _check_force_file_size(unitcell, dim, force_file: str, readfc: bool, role: str) -> None:
    """One-line ERROR when the force file is for another supercell than DIM gives (before phonopy
    reads it, which fails with an index error for a smaller DIM)."""
    natom = _force_file_atoms(force_file, readfc)
    n_cell = len(unitcell)
    expected = int(np.prod(dim)) * n_cell
    if natom is None or natom == expected:
        return
    option = "--child-dim" if role == "child" else "--parent-dim"
    given = " ".join(str(v) for v in dim)
    if natom % n_cell == 0:
        cells = natom // n_cell
        side = round(cells ** (1.0 / 3.0))
        hint = f"{cells} cells of the POSCAR" + (
            f", e.g. {option} {side} {side} {side}" if side**3 == cells else ""
        )
    else:
        hint = f"not a multiple of the {n_cell} atoms of the POSCAR"
    raise SystemExit(
        f"ERROR: {option} {given} gives a supercell of {expected} atoms, but the {role} force "
        f"data '{_display_path(force_file)}' are for {natom} atoms ({hint})."
    )


def _load_phonon(unitcell, dim, matrix, force_file: str, readfc: bool, role: str):
    """phonopy.load of one structure inside an empty working directory (see above)."""
    import phonopy

    _check_force_file_size(unitcell, dim, force_file, readfc, role)
    kwargs = dict(
        supercell_matrix=np.diag(dim),
        primitive_matrix=matrix,
        unitcell=unitcell,
        is_nac=False,
        is_compact_fc=False,
        log_level=0,
    )
    if readfc:
        kwargs["force_constants_filename"] = force_file
    else:
        kwargs["force_sets_filename"] = force_file
    shown = _display_path(force_file)
    try:
        with _empty_working_directory(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="primitive_matrix defaulted to 'auto'")
            # a child cell that is an exact supercell of a smaller cell (unfolded on that cell,
            # as the setup block says) has a supercell of lower point group than its primitive
            warnings.filterwarnings("ignore", message=_POINT_GROUP_WARNING)
            phonon = phonopy.load(**kwargs)
    except RecursionError:  # phonopy reads past the end of a truncated FORCE_SETS
        raise SystemExit(
            f"ERROR: could not read the {role} force data '{shown}': the file ends early or is "
            "not in phonopy's format."
        ) from None
    except (
        ValueError,
        RuntimeError,
        IndexError,
        KeyError,
        OSError,
        TypeError,
        AssertionError,
    ) as exc:
        message = " ".join(str(exc).split()) or type(exc).__name__
        raise SystemExit(
            f"ERROR: could not read the {role} force data '{shown}': {message}"
        ) from None
    fc = phonon.force_constants
    n_super = len(phonon.supercell)
    if fc is None:
        raise SystemExit(
            f"ERROR: could not read the {role} force data '{shown}': no force constants."
        )
    if np.asarray(fc).shape[:2] != (n_super, n_super):
        raise SystemExit(
            f"ERROR: the {role} force data '{shown}' are for {np.asarray(fc).shape[1]} atoms, but "
            f"the supercell {' '.join(str(v) for v in dim)} of the {role} has {n_super} atoms."
        )
    return phonon


def load_unfold_inputs(
    child_poscar: str,
    child_dim,
    parent_poscar: str,
    parent_dim=None,
    *,
    child_readfc: bool = False,
    parent_readfc: bool = False,
) -> UnfoldInputs:
    """Read the child and the parent: structures, primitive cells and force constants.

    The child's force file (FORCE_SETS, or FORCE_CONSTANTS with
    ``child_readfc``) is looked up next to ``child_poscar``, then in the
    working directory; the parent's (only with ``parent_dim``) next to
    ``parent_poscar`` only.  In each place the per-structure name
    ``FORCE_SETS_<POSCAR file name>`` (macer's name, e.g.
    FORCE_SETS_POSCAR_child) comes before ``FORCE_SETS``, so that the two
    POSCARs can share one directory.  Every phonopy.load runs inside an empty temporary
    working directory with absolute paths (phonopy would otherwise read a
    stray ./FORCE_CONSTANTS instead of the FORCE_SETS it is given).

    Args:
        child_poscar: The child (low-symmetry) structure.
        child_dim: Diagonal supercell of the child force calculation.
        parent_poscar: The parent (high-symmetry) structure.
        parent_dim: Diagonal supercell of the parent force calculation, or
            None (no parent force data: no parent bands).
        child_readfc, parent_readfc: Read FORCE_CONSTANTS instead of FORCE_SETS.

    Returns:
        :class:`UnfoldInputs`.

    Raises:
        SystemExit: A file is missing or unreadable, or child and parent would
            read the same force file (``ValueError`` through ``crystod.phonon``).
    """
    child_unitcell = _read_unitcell(child_poscar)
    parent_unitcell = _read_unitcell(parent_poscar)
    child_dim = _dim_tuple(child_dim, "--child-dim")
    parent_dim = None if parent_dim is None else _dim_tuple(parent_dim, "--parent-dim")
    child_force = _force_file(child_poscar, child_readfc, "child")
    parent_force = None
    if parent_dim is not None:
        parent_force = _force_file(parent_poscar, parent_readfc, "parent")
        if os.path.realpath(parent_force) == os.path.realpath(child_force):
            name = "FORCE_CONSTANTS" if child_readfc else "FORCE_SETS"
            shown = _display_path(child_force)
            child_name = os.path.basename(child_poscar)
            parent_name = os.path.basename(parent_poscar)
            if os.path.realpath(child_poscar) == os.path.realpath(parent_poscar):
                raise SystemExit(
                    f"ERROR: the child and the parent are the same file '{child_poscar}' and would "
                    f"both read '{shown}'; to unfold a structure onto itself, give the parent as a "
                    f"copy in another directory with its own {name} (e.g. --parent-poscar "
                    f"same/{parent_name})."
                )
            if child_name == parent_name:
                raise SystemExit(
                    f"ERROR: the child and the parent would both read '{shown}'; give each its "
                    f"own force file: a {name} next to each POSCAR, in two directories (e.g. "
                    f"child/{child_name} and parent/{parent_name})."
                )
            raise SystemExit(
                f"ERROR: the child and the parent would both read '{shown}'; give each its own "
                f"force file: {name}_{child_name} and {name}_{parent_name} next to the POSCARs, "
                f"or the parent POSCAR with its own {name} in another directory (e.g. "
                "--parent-poscar parent/POSCAR)."
            )
    child_matrix = _unfolding_primitive_matrix(child_unitcell, "child")
    parent_matrix = _unfolding_primitive_matrix(parent_unitcell, "parent")
    child_phonon = _load_phonon(
        child_unitcell, child_dim, child_matrix, child_force, child_readfc, "child"
    )
    parent_phonon = None
    if parent_dim is not None:
        parent_phonon = _load_phonon(
            parent_unitcell, parent_dim, parent_matrix, parent_force, parent_readfc, "parent"
        )
        parent_cell = parent_phonon.primitive
    else:
        parent_cell = _primitive_cell(parent_unitcell, parent_matrix)
    return UnfoldInputs(
        child_poscar=child_poscar,
        parent_poscar=parent_poscar,
        child_force_file=child_force,
        parent_force_file=parent_force,
        child_dim=child_dim,
        parent_dim=parent_dim,
        child_readfc=bool(child_readfc),
        parent_readfc=bool(parent_readfc),
        child_unitcell=child_unitcell,
        parent_unitcell=parent_unitcell,
        child_primitive_matrix=child_matrix,
        parent_primitive_matrix=parent_matrix,
        child_cell=child_phonon.primitive,
        parent_cell=parent_cell,
        child_phonon=child_phonon,
        parent_phonon=parent_phonon,
    )


# ---------------------------------------------------------------------------
# mapping: setting, origin shift, atom map
# ---------------------------------------------------------------------------
@dataclass
class ParentMapping:
    """The child -> parent setting and atom map of the unfolding.

    Attributes:
        M: (3, 3) int64, ``A_child = M @ A_parent`` (unfolding cells).
        origin_shift: (3,) p in [0, 1), parent primitive fractional
            (``x_parent = M.T x_child + p``).
        N: det M.
        kappa: (n_child,) parent primitive atom of every child atom.
        ideal_frac_child: (n_child, 3) ideal site next to each atom (child fractional).
        displacement_cart: (n_child, 3) Angstrom, child Cartesian frame.
        max_displacement, max_displacement_species, rms_displacement: In the
            child lattice (Angstrom).
        max_displacement_parent_lattice: The same fractional displacements in
            the strain-free parent lattice (the measure of crystod-group).
        A_child, A_parent, A_pp: Lattices (rows); ``A_pp = inv(M) @ A_child``.
        strain: Lengths, angles, strains, volume strain and rotation angle of
            the pseudo-parent cell against the parent cell.
        frame_rotation: (3, 3) proper rotation, ``v_parent = R @ v_child``.
        child_cell, parent_cell: The unfolding cells (PhonopyAtoms).
        child_unitcell, parent_unitcell: The cells of the files (frames of the
            ISO-IR names).
        child_space_group, parent_space_group: ``'Pnma (No. 62)'`` (spglib at
            1e-5 on the unfolding cells); ``child_symbol``, ``child_number``,
            ``parent_symbol``, ``parent_number`` the parts.
        folding_kpoints: (N, 3) parent k with M k integer, in (-1/2, 1/2]; GM first.
        folding_names: Their ISO-IR names in the parent (``GM``, ``X``, ...).
        setting_P, setting_p: The setting used, as crystod-group prints it for
            the input files (columns of P: the child basis in parent units).
        crystod_group_P, crystod_group_p: The setting crystod-group finds.
        equivalent_settings: Every M valid for ``--trans-mat`` (``M0 @ W.T``
            over the proper rotations W of the parent), the crystod-group
            setting M0 first.
        parent_operation: ``(W, w)`` relating a ``--trans-mat`` setting to the
            crystod-group one, else None.
        source: ``'crystod-group'``, ``'--trans-mat'`` or ``'given'``.
        sma_modes: ``[(k name, irrep, amplitude in Angstrom), ...]`` of
            crystod-group's decomposition of the distortion.
        sma_max_displacement: crystod-group's largest atomic displacement.
        sma_child_space_group: The child space group at the mapping tolerance.
        notes: NOTE lines about the mapping (printed under the setup block).
    """

    M: np.ndarray
    origin_shift: np.ndarray
    N: int
    kappa: np.ndarray
    ideal_frac_child: np.ndarray
    displacement_cart: np.ndarray
    max_displacement: float
    max_displacement_species: str
    rms_displacement: float
    max_displacement_parent_lattice: float
    A_child: np.ndarray
    A_parent: np.ndarray
    A_pp: np.ndarray
    strain: dict
    frame_rotation: np.ndarray
    child_cell: object
    parent_cell: object
    child_unitcell: object
    parent_unitcell: object
    child_space_group: str
    parent_space_group: str
    child_symbol: str
    child_number: int
    parent_symbol: str
    parent_number: int
    folding_kpoints: np.ndarray
    folding_names: list
    setting_P: np.ndarray | None = None
    setting_p: np.ndarray | None = None
    crystod_group_P: np.ndarray | None = None
    crystod_group_p: np.ndarray | None = None
    equivalent_settings: list = field(default_factory=list)
    parent_operation: tuple | None = None
    source: str = "given"
    sma_modes: list = field(default_factory=list)
    sma_max_displacement: float | None = None
    sma_child_space_group: str | None = None
    notes: list = field(default_factory=list)


_NAME_CACHE: dict = {}


def _cell_key(cell) -> tuple:
    lattice, positions, numbers = _cell_tuple(cell)
    return (
        tuple(np.round(lattice, 8).ravel()),
        tuple(np.round(positions, 8).ravel()),
        tuple(numbers),
    )


def _isoir_name(sgnum: int, cell, input_cell, k):
    """ISO-IR k-vector type of k (primitive basis of ``cell``), or None."""
    from .isoir import get_isoir_kpoint_name

    key = (int(sgnum), _cell_key(cell), _cell_key(input_cell), tuple(np.round(_reduce_unit(k), 9)))
    if key not in _NAME_CACHE:
        if len(_NAME_CACHE) > 4096:
            _NAME_CACHE.clear()
        try:
            name = get_isoir_kpoint_name(
                int(sgnum),
                _cell_tuple(cell),
                SYMPREC,
                [float(v) for v in _reduce_half(k)],
                input_cell=_cell_tuple(input_cell),
            )
        except Exception:  # the names are annotations; a missing one prints '?'
            name = None
        _NAME_CACHE[key] = name
    return _NAME_CACHE[key]


def _parent_point_name(mapping: ParentMapping, k) -> str | None:
    """ISO-IR name of a parent k point (parent primitive basis)."""
    return _isoir_name(mapping.parent_number, mapping.parent_cell, mapping.parent_unitcell, k)


def _child_point_name(mapping: ParentMapping, K) -> str:
    """ISO-IR name of the child point K (child unfolding-cell basis); '?' when unknown."""
    name = _isoir_name(mapping.child_number, mapping.child_cell, mapping.child_unitcell, K)
    return name or "?"


def parent_mapping_from_setting(
    child_cell,
    parent_cell,
    M,
    origin_shift,
    *,
    child_unitcell=None,
    parent_unitcell=None,
    max_displacement: float = MAX_DISPLACEMENT,
) -> ParentMapping:
    """The atom map of the child onto the parent for a given setting (M, p).

    No symmetry-mode analysis: the setting is taken as given.  The ideal sites
    are the N copies of the parent atoms inside the child cell,
    ``inv(M.T) @ (x_parent[kappa] + t - p)``; the child atoms of each species
    are paired with them by the minimum total squared displacement (Hungarian
    algorithm, minimum image in the child lattice); the pairing is validated
    (every parent atom owns N child atoms of its species in N distinct
    cosets, one common origin shift, every parent translation permutes the
    ideal sites).

    Args:
        child_cell: The child unfolding cell (PhonopyAtoms).
        parent_cell: The parent primitive cell (PhonopyAtoms).
        M: ``A_child = M @ A_parent`` (integer, det > 0).
        origin_shift: p of ``x_parent = M.T x_child + p``.
        child_unitcell, parent_unitcell: Cells whose frames name the k points
            (default: the unfolding cells).
        max_displacement: Largest accepted displacement (Angstrom).

    Returns:
        :class:`ParentMapping` with ``source = 'given'``.

    Raises:
        SystemExit: Atom counts that do not fit, or a displacement above
            ``max_displacement`` (``ValueError`` through ``crystod.phonon``).
    """
    from scipy.optimize import linear_sum_assignment

    Mi = _as_supercell_matrix(M)
    N = int(round(np.linalg.det(Mi)))
    A_c = np.array(child_cell.cell, dtype=float)
    A_p = np.array(parent_cell.cell, dtype=float)
    x_c = np.array(child_cell.scaled_positions, dtype=float)
    x_p = np.array(parent_cell.scaled_positions, dtype=float)
    sym_c = np.array(list(child_cell.symbols))
    sym_p = np.array(list(parent_cell.symbols))
    n_c, n_p = len(x_c), len(x_p)
    if n_c != N * n_p:
        raise SystemExit(
            f"ERROR: the child cell has {n_c} atoms, N x n_parent = {N} x {n_p} = {N * n_p}."
        )
    species = list(dict.fromkeys(list(sym_p) + list(sym_c)))
    for s in species:
        count_c = int(np.sum(sym_c == s))
        count_p = int(np.sum(sym_p == s))
        if count_c != N * count_p:
            raise SystemExit(
                f"ERROR: species {s}: {count_c} child atoms, but N x {count_p} = {N * count_p} "
                "expected."
            )
    p = _snap_rational(_reduce_unit(np.asarray(origin_shift, dtype=float).reshape(3)))
    translations = _coset_translations(Mi)
    MT = Mi.T.astype(float)
    MT_inv = np.linalg.inv(MT)
    site_kappa = np.repeat(np.arange(n_p), N)
    site_t = np.tile(translations, (n_p, 1))
    site_x = (x_p[site_kappa] + site_t - p) @ MT_inv.T
    kappa = np.full(n_c, -1, dtype=np.int64)
    ideal = np.zeros_like(x_c)
    for s in species:
        ic = np.where(sym_c == s)[0]
        js = np.where(sym_p[site_kappa] == s)[0]
        d = x_c[ic][:, None, :] - site_x[js][None, :, :]
        cart, shift = _min_image(d, A_c)
        rows, cols = linear_sum_assignment(np.sum(cart**2, axis=-1))
        kappa[ic[rows]] = site_kappa[js[cols]]
        ideal[ic[rows]] = x_c[ic[rows]] - (d[rows, cols] + shift[rows, cols])
    u_frac = x_c - ideal
    disp = u_frac @ A_c
    norms = np.linalg.norm(disp, axis=1)
    worst = int(np.argmax(norms))
    if norms[worst] > max_displacement:
        raise SystemExit(
            f"ERROR: the largest displacement {norms[worst]:.2f} A ({sym_c[worst]}) exceeds "
            f"{max_displacement:g} A: the cells do not match in this setting."
        )
    _validate_atom_map(Mi, kappa, x_c, ideal, A_c, list(sym_c), x_p, p)
    _translation_permutations(ideal, list(sym_c), Mi)
    A_pp = np.linalg.solve(Mi.astype(float), A_c)
    parent_norms = np.linalg.norm((u_frac @ Mi.astype(float)) @ A_p, axis=1)
    child_symbol, child_number = _space_group(child_cell)
    parent_symbol, parent_number = _space_group(parent_cell)
    mapping = ParentMapping(
        M=Mi,
        origin_shift=p,
        N=N,
        kappa=kappa,
        ideal_frac_child=ideal,
        displacement_cart=disp,
        max_displacement=float(norms[worst]),
        max_displacement_species=str(sym_c[worst]),
        rms_displacement=float(np.sqrt(np.mean(norms**2))),
        max_displacement_parent_lattice=float(parent_norms.max()),
        A_child=A_c,
        A_parent=A_p,
        A_pp=A_pp,
        strain=_strain_and_rotation(A_pp, A_p),
        frame_rotation=frame_rotation(A_pp, A_p),
        child_cell=child_cell,
        parent_cell=parent_cell,
        child_unitcell=child_cell if child_unitcell is None else child_unitcell,
        parent_unitcell=parent_cell if parent_unitcell is None else parent_unitcell,
        child_space_group=f"{child_symbol} (No. {child_number})",
        parent_space_group=f"{parent_symbol} (No. {parent_number})",
        child_symbol=child_symbol,
        child_number=child_number,
        parent_symbol=parent_symbol,
        parent_number=parent_number,
        folding_kpoints=_parent_kpoints_folding_to_gamma(Mi),
        folding_names=[],
    )
    mapping.folding_names = [_parent_point_name(mapping, k) or "k" for k in mapping.folding_kpoints]
    return mapping


def _operations(cell, symprec: float = SYMPREC):
    """``(rotations, translations)`` of a cell (spglib), fractional."""
    import spglib

    symmetry = spglib.get_symmetry(_cell_tuple(cell), symprec=symprec)
    return (
        np.asarray(symmetry["rotations"], dtype=np.int64),
        np.asarray(symmetry["translations"], dtype=float),
    )


def find_parent_mapping(
    child_poscar: str,
    parent_poscar: str,
    *,
    child_cell=None,
    parent_cell=None,
    trans_mat=None,
    tolerance: float = MAPPING_TOLERANCE,
    max_displacement: float = MAX_DISPLACEMENT,
) -> ParentMapping:
    """The setting of crystod-group, converted to the unfolding cells, with the atom map.

    ``crystod.symmetry_mode.SymmetryModeAnalysis(parent, child, tolerance)``
    (the analysis of ``crystod-group --supergroup-cif PARENT --subgroup-cif
    CHILD``) gives the transformation (P, p) between the two input files;
    with the phonopy primitive matrices Pc, Pp of the unfolding cells
    (``A_prim = P.T @ A_file``) it becomes ``M0 = Pc.T @ P.T @ inv(Pp.T)`` and
    ``p0 = inv(Pp) @ p``.  ``trans_mat`` chooses another of the equivalent
    settings: it must be ``M0 @ W.T`` for a rotation W of an operation (W, w)
    of the parent, and the origin becomes ``W @ p0 + w``.

    Args:
        child_poscar, parent_poscar: The two structure files.
        child_cell, parent_cell: The unfolding cells (default: the cells of the
            files reduced as :func:`load_unfold_inputs` does).
        trans_mat: None, or nine integers row-wise, ``A_child = T @ A_parent``
            (the ``crystod-bz --trans-mat`` convention) relative to the
            unfolding cells.
        tolerance: Symmetry tolerance of the analysis (crystod-group's 0.01), in
            (0, 0.5] Angstrom.
        max_displacement: Largest accepted displacement (Angstrom).

    Returns:
        :class:`ParentMapping`.

    Raises:
        SystemExit: crystod-group cannot relate the structures, or
            ``trans_mat`` is not an equivalent setting (``ValueError`` through
            ``crystod.phonon``).
    """
    from .symmetry_mode import SymmetryModeAnalysis

    tolerance = float(tolerance)
    if not 0.0 < tolerance <= MAPPING_TOLERANCE_MAX:  # spglib fails at 0, crashes at -1 and 2
        raise SystemExit(
            f"ERROR: the tolerance of the setting search must lie in (0, "
            f"{MAPPING_TOLERANCE_MAX:g}] Angstrom (default {MAPPING_TOLERANCE:g}), got "
            f"{tolerance:g}."
        )
    child_unitcell = _read_unitcell(child_poscar)
    parent_unitcell = _read_unitcell(parent_poscar)
    if child_cell is None:
        child_cell = _primitive_cell(
            child_unitcell, _unfolding_primitive_matrix(child_unitcell, "child")
        )
    if parent_cell is None:
        parent_cell = _primitive_cell(
            parent_unitcell, _unfolding_primitive_matrix(parent_unitcell, "parent")
        )
    Pc = np.linalg.solve(
        np.array(child_unitcell.cell, dtype=float).T, np.array(child_cell.cell, dtype=float).T
    )
    Pp = np.linalg.solve(
        np.array(parent_unitcell.cell, dtype=float).T, np.array(parent_cell.cell, dtype=float).T
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.simplefilter("ignore", UserWarning)
            analysis = SymmetryModeAnalysis(parent_poscar, child_poscar, tolerance)
    except SystemExit as exc:
        message = " ".join(str(exc).split()).removeprefix("ERROR: ")
        raise SystemExit(
            f"ERROR: crystod-group could not relate the two structures: {message}"
        ) from None
    st = analysis.setting_transformation(frame="input")
    if st.frame != "input":
        raise SystemExit(
            "ERROR: spglib gave no setting transformation for one of the input files; the "
            "transformation cannot be expressed in the input cells (give the structures in a "
            "setting spglib recognises)."
        )
    M_file = np.asarray(st.matrix, dtype=float).T
    M0f = Pc.T @ M_file @ np.linalg.inv(Pp.T)
    M0 = np.rint(M0f).astype(np.int64)
    if np.abs(M0f - M0).max() > 1e-6 or int(round(np.linalg.det(M0))) <= 0:
        raise SystemExit(
            "ERROR: the crystod-group setting is not an integer matrix between the unfolding "
            f"cells (M = {np.round(M0f, 4).tolist()})."
        )
    p0 = _snap_rational(_reduce_unit(np.linalg.inv(Pp) @ np.asarray(st.origin, dtype=float)))
    rotations, translations = _operations(parent_cell)
    equivalent = [M0]
    for W in rotations:
        candidate = M0 @ W.T
        if int(round(np.linalg.det(candidate))) > 0 and not any(
            np.array_equal(candidate, e) for e in equivalent
        ):
            equivalent.append(candidate)
    M, p, operation, source = M0, p0, None, "crystod-group"
    if trans_mat is not None:
        T = np.rint(np.asarray(trans_mat, dtype=float)).astype(np.int64).reshape(3, 3)
        X = np.linalg.solve(M0.astype(float), T.astype(float))
        hit = None
        if np.abs(X - np.rint(X)).max() < 1e-6:
            W = np.rint(X).astype(np.int64).T
            for Wp, wp in zip(rotations, translations):
                if np.array_equal(Wp, W):
                    hit = (W, np.asarray(wp, dtype=float))
                    break
        if hit is None or int(round(np.linalg.det(T))) <= 0:
            lines = [
                f"ERROR: --trans-mat {_matrix_text(T)} is not one of the {len(equivalent)} "
                "equivalent settings of this child (M0 @ W.T for the proper rotations W of the "
                f"parent); the crystod-group setting is {_matrix_text(M0)}.",
                "valid settings (--trans-mat):",
            ]
            lines += [f"  {_matrix_text(e)}" for e in equivalent]
            raise SystemExit("\n".join(lines))
        W, w = hit
        M, p, operation, source = (
            T,
            _snap_rational(_reduce_unit(W @ p0 + w)),
            (W, _snap_rational(w)),
            "--trans-mat",
        )
    mapping = parent_mapping_from_setting(
        child_cell,
        parent_cell,
        M,
        p,
        child_unitcell=child_unitcell,
        parent_unitcell=parent_unitcell,
        max_displacement=max_displacement,
    )
    mapping.source = source
    mapping.parent_operation = operation
    mapping.crystod_group_P = np.asarray(st.matrix, dtype=float)
    mapping.crystod_group_p = np.asarray(st.origin, dtype=float)
    if operation is None:
        # exactly what crystod-group prints (a centred parent: p could differ by a centring vector)
        mapping.setting_P = mapping.crystod_group_P.copy()
        mapping.setting_p = mapping.crystod_group_p.copy()
    else:
        mapping.setting_P = _snap_rational((np.linalg.inv(Pc.T) @ M.astype(float) @ Pp.T).T)
        mapping.setting_p = _snap_rational(_reduce_unit(Pp @ mapping.origin_shift))
    mapping.equivalent_settings = equivalent
    mapping.sma_modes = [
        (str(m.kname), str(m.irrep_name), float(m.amplitude)) for m in analysis.modes
    ]
    mapping.sma_max_displacement = float(np.linalg.norm(analysis.u_cart, axis=1).max())
    mapping.sma_child_space_group = f"{analysis.child_symbol} (No. {analysis.child_number})"
    if getattr(analysis, "mapping_note", None):  # crystod-group prints it with the cell relation
        note = " ".join(str(analysis.mapping_note).split())
        mapping.notes.append(note if note.startswith("NOTE") else f"NOTE: {note}")
    if abs(mapping.max_displacement_parent_lattice - mapping.sma_max_displacement) > 1e-3:
        mapping.notes.append(
            "NOTE: the largest displacement in the strain-free parent lattice, "
            f"{mapping.max_displacement_parent_lattice:.4f} A, differs from crystod-group's "
            f"{mapping.sma_max_displacement:.4f} A (crystod-group measures on the symmetrized "
            "child)."
        )
    if analysis.child_number != mapping.child_number:
        mapping.notes.append(
            f"NOTE: the child cell is {mapping.sma_child_space_group} at the mapping tolerance "
            f"{tolerance:g} but {mapping.child_space_group} at {SYMPREC:g}; the arm classes and "
            f"the irrep separation use {mapping.child_symbol}."
        )
    return mapping


# ---------------------------------------------------------------------------
# the parent band path
# ---------------------------------------------------------------------------
@dataclass
class ParentPath:
    """A band path of the parent zone, as :func:`make_parent_path` builds it.

    Attributes:
        kpoints: (n_k, 3) parent primitive fractional k.
        segment_index: (n_k,) leg of every point.
        distances: (n_k,) cumulative path length, 1/Angstrom without 2 pi, in
            the reciprocal metric of A_pp; no jump at a break (phonopy).
        legs: ``[(k_start, k_end), ...]``; ``leg_labels`` their labels.
        connections: phonopy path_connections (leg i continues into leg i + 1).
        tick_positions, tick_labels: One tick per vertex; ``'X|R'`` at a break.
        vertices: ``[(label, k, K = M k, child point name), ...]``, one per leg
            end point in path order (both sides of a break appear).
        path_text: ``'GM-X-M-GM-R-X|R-M'``.
        source: ``'seekpath'`` or ``'--band'``.
        npoints: Points per leg (inclusive end points).
        note: The NOTE about a basis conversion of the seekpath path, or None.
    """

    kpoints: np.ndarray
    segment_index: np.ndarray
    distances: np.ndarray
    legs: list
    leg_labels: list
    connections: list
    tick_positions: np.ndarray
    tick_labels: list
    vertices: list
    path_text: str
    source: str
    npoints: int
    note: str | None = None


def _tick_child_points(path: ParentPath) -> list:
    """Child point name of every tick of a path (``'GM|U'`` at a break between two points)."""
    names, i = [], 0
    for label in path.tick_labels:
        points = []
        for _part in str(label).split("|"):
            points.append(path.vertices[i][3] if i < len(path.vertices) else "?")
            i += 1
        names.append(points[0] if len(set(points)) == 1 else "|".join(points))
    return names


def _manual_segments(band: str, band_labels: str | None = None) -> list:
    """The segments of a ``--band`` path (parent primitive coordinates), checked with its labels."""
    from .brillouin_zone import parse_manual_band

    try:
        segments = [np.asarray(s, dtype=float) for s in parse_manual_band(band)]
    except (ValueError, ZeroDivisionError) as exc:
        raise SystemExit(f"ERROR: --band: {' '.join(str(exc).split())}") from None
    if band_labels:
        expected = sum(len(s) for s in segments)
        given = len(band_labels.split())
        if given != expected:
            raise SystemExit(
                f"ERROR: --band-labels needs {expected} labels for this --band path, but "
                f"{given} were given."
            )
    return segments


def _clean_seekpath_label(label: str) -> str:
    return "GM" if label in ("GAMMA", "G") else label.replace("_", "")


def make_parent_path(
    mapping: ParentMapping,
    *,
    band: str | None = None,
    band_labels: str | None = None,
    npoints: int = NPOINTS,
    symprec: float = SYMPREC,
) -> ParentPath:
    """The band path of the parent zone along which the child phonons are unfolded.

    Default: seekpath's path of the parent primitive cell (the ``--fatband``
    path), in the parent primitive basis (converted when seekpath's
    standardized primitive cell is another basis).  ``band``: a manual path in
    the ``--band`` format, parent primitive fractional coordinates.  The
    vertex labels are the ISO-IR names of the vertices in the parent unless
    ``band_labels`` gives one label per vertex.

    Args:
        mapping: The :class:`ParentMapping` (parent cell, M, A_pp).
        band: Manual path, e.g. ``"0 0 0  0 1/2 0  1/2 1/2 0, 1/2 1/2 0  1/2 1/2 1/2"``.
        band_labels: Labels of every vertex of ``band``, e.g. ``"GM X M M R"``.
        npoints: Points per leg (inclusive end points).
        symprec: seekpath tolerance.

    Returns:
        :class:`ParentPath`.
    """
    from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

    if int(npoints) < 2:
        raise SystemExit("ERROR: --npoints must be at least 2.")
    note = None
    if band:
        segments = _manual_segments(band, band_labels)
        source = "--band"
        if band_labels:
            flat = band_labels.split()
            labels, cursor = [], 0
            for s in segments:
                labels.append(flat[cursor : cursor + len(s)])
                cursor += len(s)
        else:
            labels = [[_parent_point_name(mapping, k) or "?" for k in s] for s in segments]
    else:
        if band_labels:
            raise SystemExit("ERROR: --band-labels requires --band.")
        from .brillouin_zone import get_seekpath_kpath

        segments, label_segments, lattice, _symbol, _number = get_seekpath_kpath(
            mapping.parent_cell, symprec
        )
        segments = [np.asarray(s, dtype=float) for s in segments]
        if not np.allclose(np.asarray(lattice, dtype=float), mapping.A_parent, atol=1e-4):
            import seekpath

            coords = seekpath.get_path_orig_cell(
                tuple(x.tolist() for x in _cell_tuple(mapping.parent_cell)), symprec=symprec
            )["point_coords"]
            segments = [
                np.array([_snap_rational(coords[name]) for name in names])
                for names in label_segments
            ]
            note = (
                "NOTE: the seekpath coordinates were converted from the seekpath standardized "
                "cell to the parent primitive cell."
            )
        labels = [
            [
                _parent_point_name(mapping, k) or _clean_seekpath_label(name)
                for k, name in zip(s, names)
            ]
            for s, names in zip(segments, label_segments)
        ]
        source = "seekpath"
    qpoints, connections = get_band_qpoints_and_path_connections(segments, npoints=int(npoints))
    qpoints = [np.asarray(q, dtype=float) for q in qpoints]
    legs, leg_labels = [], []
    for s, names in zip(segments, labels):
        for i in range(len(s) - 1):
            legs.append((s[i].copy(), s[i + 1].copy()))
            leg_labels.append((names[i], names[i + 1]))
    B = _reciprocal_lattice(mapping.A_pp)
    distances, start = [], 0.0
    for q in qpoints:
        steps = np.linalg.norm(np.diff(q, axis=0) @ B, axis=1)
        distances.append(start + np.concatenate([[0.0], np.cumsum(steps)]))
        start = float(distances[-1][-1])
    ticks = [float(distances[0][0])] + [float(d[-1]) for d in distances]
    tick_labels = [leg_labels[0][0]]
    for i, connected in enumerate(connections):
        end = leg_labels[i][1]
        if connected or i == len(connections) - 1:
            tick_labels.append(end)
        else:
            following = leg_labels[i + 1][0]
            tick_labels.append(end if end == following else f"{end}|{following}")
    M = mapping.M.astype(float)
    vertices = []
    for s, names in zip(segments, labels):
        for k, name in zip(s, names):
            K = M @ k
            vertices.append((name, k.copy(), K, _child_point_name(mapping, K)))
    return ParentPath(
        kpoints=np.concatenate(qpoints),
        segment_index=np.repeat(np.arange(len(qpoints)), [len(q) for q in qpoints]),
        distances=np.concatenate(distances),
        legs=legs,
        leg_labels=leg_labels,
        connections=list(connections),
        tick_positions=np.array(ticks),
        tick_labels=tick_labels,
        vertices=vertices,
        path_text="|".join("-".join(names) for names in labels),
        source=source,
        npoints=int(npoints),
        note=note,
    )


# ---------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------
@dataclass
class UnfoldedBands:
    """Child phonons unfolded onto parent k points (:func:`unfold`).

    Attributes:
        k_parent: (n_k, 3) parent fractional k.
        K_child: (n_k, 3) ``k_parent @ M.T`` (what phonopy was given; unwrapped).
        frequencies: (n_k, n_band) THz, ascending, n_band = 3 n_child.
        weights: (n_k, n_band) P_J(k) in phonopy's eigenvector basis.
        weights_canonical: (n_k, n_band) after the canonical rotation of
            degenerate sets (basis independent).
        vectors: (n_k, n_band, 3 n_parent) complex canonical u_J(k), child
            Cartesian frame, parent origin, index 3 kappa + alpha.
        degenerate_group: (n_k, n_band) group index (0, 1, ... ascending) at each k.
        group_weight: (n_k, n_band) P_g of the mode's degenerate group.
        sum_rule_max_deviation: max over k of ``|sum_J P_J(k) - 3 n_parent|``.
        timings: ``{'eigen_s', 'unfold_s', 'canonical_s'}`` (CPU wall time).
    """

    k_parent: np.ndarray
    K_child: np.ndarray
    frequencies: np.ndarray
    weights: np.ndarray
    weights_canonical: np.ndarray
    vectors: np.ndarray
    degenerate_group: np.ndarray
    group_weight: np.ndarray
    sum_rule_max_deviation: float
    timings: dict


def _check_child_cell(cell, reference) -> None:
    """The phonopy cell of the child must be the cell of the mapping."""
    same = (
        len(cell) == len(reference)
        and list(cell.symbols) == list(reference.symbols)
        and np.abs(np.array(cell.cell) - np.array(reference.cell)).max() < 1e-8
    )
    if same:
        d = np.array(cell.scaled_positions) - np.array(reference.scaled_positions)
        same = bool(np.abs(d - np.rint(d)).max() < 1e-8)
    if not same:
        raise ValueError("the primitive cell of child_phonon is not the child cell of the mapping")


def unfold(
    child_phonon,
    mapping: ParentMapping,
    kpoints,
    *,
    degeneracy_tolerance: float = DEGENERACY_TOLERANCE,
    gauge: str = "ideal",
    method: str = "unwrapped",
) -> UnfoldedBands:
    """Unfold the child phonons onto parent k points.

    Args:
        child_phonon: ``phonopy.Phonopy`` whose primitive cell is
            ``mapping.child_cell`` (e.g. ``load_unfold_inputs(...).child_phonon``).
        mapping: The :class:`ParentMapping`.
        kpoints: (n_k, 3) parent primitive fractional k.
        degeneracy_tolerance: THz, degenerate child modes (canonical rotation, groups).
        gauge: ``"ideal"`` (default) or ``"actual"`` (diagnostic, without the
            ideal-site phase factor).
        method: ``"unwrapped"`` (K = M k, default) or ``"wrapped"`` (K reduced
            into [-1/2, 1/2] with the explicit G phase; identical results).

    Returns:
        :class:`UnfoldedBands`.

    Raises:
        SystemExit: The sum rule is violated (an internal error).
    """
    _check_child_cell(child_phonon.primitive, mapping.child_cell)
    k = np.atleast_2d(np.asarray(kpoints, dtype=float))
    M = mapping.M
    K_unwrapped = k @ M.T.astype(float)
    if method == "unwrapped":
        K = K_unwrapped.copy()
    elif method == "wrapped":
        K = K_unwrapped - np.rint(K_unwrapped)
    else:
        raise ValueError("method must be 'unwrapped' or 'wrapped'")
    x = np.array(mapping.child_cell.scaled_positions, dtype=float)
    n_k = len(k)
    n_band = 3 * len(x)
    n_parent = len(mapping.parent_cell)
    freqs = np.zeros((n_k, n_band))
    weights = np.zeros((n_k, n_band))
    uvec = np.zeros((n_k, n_band, 3 * n_parent), dtype=complex)
    t_eigen = t_unfold = 0.0
    for start in range(0, n_k, CHUNK):
        part = slice(start, min(start + CHUNK, n_k))
        t0 = time.perf_counter()
        result = _run_qpoints(child_phonon, K[part], with_eigenvectors=True)
        f = np.array(result.frequencies, dtype=float)
        E = np.array(result.eigenvectors)
        t1 = time.perf_counter()
        u, w, _G = unfold_eigenvectors(
            E,
            K[part],
            k[part],
            M,
            mapping.kappa,
            x,
            mapping.ideal_frac_child,
            gauge=gauge,
            origin_shift=mapping.origin_shift,
        )
        freqs[part], weights[part], uvec[part] = f, w, u
        t_eigen += t1 - t0
        t_unfold += time.perf_counter() - t1
    t2 = time.perf_counter()
    vectors, weights_c, _rotations = canonical_degenerate_rotation(
        freqs, uvec, degeneracy_tolerance
    )
    groups = degenerate_group_ids(freqs, degeneracy_tolerance)
    group_weight = _sum_weights_by_group(weights_c, groups)
    t_canonical = time.perf_counter() - t2
    deviation = np.abs(weights.sum(axis=1) - 3 * n_parent)
    worst = int(np.argmax(deviation)) if n_k else 0
    if n_k and deviation[worst] > SUM_RULE_TOLERANCE:
        raise SystemExit(
            f"ERROR: internal error: the unfolding sum rule is violated by {deviation[worst]:.1e} "
            f"at k = {_vec(k[worst])}."
        )
    return UnfoldedBands(
        k_parent=k,
        K_child=K_unwrapped,
        frequencies=freqs,
        weights=weights,
        weights_canonical=weights_c,
        vectors=vectors,
        degenerate_group=groups,
        group_weight=group_weight,
        sum_rule_max_deviation=float(deviation.max()) if n_k else 0.0,
        timings={"eigen_s": t_eigen, "unfold_s": t_unfold, "canonical_s": t_canonical},
    )


# ---------------------------------------------------------------------------
# star arms and the domain average
# ---------------------------------------------------------------------------
def point_rotations(cell, symprec: float = SYMPREC) -> np.ndarray:
    """Unique point-group rotations of a cell (integer, fractional, acting on positions as W x)."""
    rotations, _translations = _operations(cell, symprec)
    return np.unique(rotations.reshape(-1, 9), axis=0).reshape(-1, 3, 3)


def child_rotations_in_parent(child_rotations, M) -> np.ndarray:
    """Child point operations in the parent fractional basis, ``W_p = M.T W_c inv(M.T)``."""
    MT = np.asarray(M, dtype=float).T
    Wp = np.einsum(
        "ab,gbc,cd->gad", MT, np.asarray(child_rotations, dtype=float), np.linalg.inv(MT)
    )
    Wi = np.rint(Wp).astype(np.int64)
    if np.abs(Wp - Wi).max() > 1e-8:
        raise SystemExit(
            "ERROR: internal error: a child point operation is not a parent operation in this "
            "setting."
        )
    return Wi


def star_arms(k_parent, rotations, tol: float = 1e-8):
    """Arms of the star of a parent k point.

    Args:
        k_parent: (3,) parent fractional k.
        rotations: (n_g, 3, 3) integer point-group rotations of the parent
            (fractional, acting on positions); fractional k transforms with
            ``W.T`` over the whole group.
        tol: Tolerance of the comparison modulo reciprocal lattice vectors.

    Returns:
        ``(arms, order of the little co-group)``: arms (n_arm, 3) reduced into
        (-1/2, 1/2], k itself first, the others in lexicographically
        descending order (X: (0,1/2,0), (1/2,0,0), (0,0,1/2)).
    """
    k = np.asarray(k_parent, dtype=float).reshape(3)
    W = np.asarray(rotations)
    images = np.einsum("gji,j->gi", W.astype(float), k)
    d = images[:, None, :] - images[None, :, :]
    same = np.abs(d - np.rint(d)).max(axis=-1) < 10 * tol
    kept, taken = [], np.zeros(len(W), dtype=bool)
    for g in range(len(W)):
        if not taken[g]:
            kept.append(g)
            taken |= same[g]
    dk = images - k[None, :]
    stabilizer = int(np.sum(np.abs(dk - np.rint(dk)).max(axis=1) < 10 * tol))
    arms = _reduce_half(images[kept])
    if len(arms) * stabilizer != len(W):
        raise SystemExit(
            f"ERROR: internal error: the star of {_vec(k)} has {len(arms)} arms and a little "
            f"co-group of {stabilizer} operations, not {len(W)} in all."
        )
    own = _reduce_half(k)
    d_own = arms - own[None, :]
    is_own = np.abs(d_own - np.rint(d_own)).max(axis=1) < 10 * tol
    others = arms[~is_own]
    if len(others):
        others = others[np.lexsort(-others.T[::-1])]
    return np.vstack([own[None, :], others]), stabilizer


def arm_orbits(
    arms, child_rotations_parent, *, time_reversal: bool = True, tol: float = 1e-8
) -> np.ndarray:
    """Classes of equivalent arms: orbits of the child point group (and time reversal).

    The child rotations (parent basis) act on k as ``W.T k`` modulo
    reciprocal lattice vectors; with ``time_reversal`` also ``k -> -k``
    (real force constants: P_J(-k) = P_J(k)).  Arms of one class give the
    same unfolded spectrum.

    Returns:
        (n_arm,) class index (0, 1, ... in order of the first arm of each class).
    """
    arms = np.atleast_2d(np.asarray(arms, dtype=float))
    W = np.asarray(child_rotations_parent, dtype=float)
    if time_reversal:
        W = np.concatenate([W, -W])
    classes = -np.ones(len(arms), dtype=np.int64)
    c = 0
    for i in range(len(arms)):
        if classes[i] >= 0:
            continue
        images = np.einsum("gji,j->gi", W, arms[i])
        d = images[:, None, :] - arms[None, :, :]
        hit = np.any(np.abs(d - np.rint(d)).max(axis=-1) < 10 * tol, axis=0)
        classes[(classes < 0) & hit] = c
        classes[i] = c
        c += 1
    return classes


def _rotation_sets(mapping: ParentMapping, parent_rotations=None, child_rotations_parent=None):
    if parent_rotations is None:
        parent_rotations = point_rotations(mapping.parent_cell)
    if child_rotations_parent is None:
        child_rotations_parent = child_rotations_in_parent(
            point_rotations(mapping.child_cell), mapping.M
        )
    known = {tuple(np.asarray(W).ravel()) for W in parent_rotations}
    if not all(tuple(np.asarray(W).ravel()) in known for W in child_rotations_parent):
        raise SystemExit(
            "ERROR: internal error: a child point operation is not a parent operation in this "
            "setting."
        )
    return np.asarray(parent_rotations), np.asarray(child_rotations_parent)


@dataclass
class StarArms:
    """Every arm of the stars of the path vertices (:func:`vertex_star_arms`).

    Attributes:
        star: Parent ISO-IR name of the star of each arm (``'GM'``, ``'X'``, ...).
        k_parent: (n_arms, 3), the path's arm first inside each star.
        K_child_reduced: (n_arms, 3) M k modulo 1, in [0, 1).
        child_point: ISO-IR name of K in the child (``'GM'``, ``'U'``, ...).
        condensed: Bool, M k integer (the arm folds onto the child Gamma point).
        on_path: Bool, the arm is a point of the band path.
        orbit: Class index inside the star (printed a, b, ...).
        frequencies, weights_canonical: (n_arms, n_band) unfolded spectra.
    """

    star: list
    k_parent: np.ndarray
    K_child_reduced: np.ndarray
    child_point: list
    condensed: np.ndarray
    on_path: np.ndarray
    orbit: np.ndarray
    frequencies: np.ndarray
    weights_canonical: np.ndarray


def vertex_star_arms(
    child_phonon,
    mapping: ParentMapping,
    path: ParentPath,
    *,
    parent_rotations=None,
    child_rotations_parent=None,
) -> StarArms:
    """Unfold every arm of the stars of the path vertices (arms block, lower panel of page 1)."""
    parent_rotations, child_rotations_parent = _rotation_sets(
        mapping, parent_rotations, child_rotations_parent
    )
    stars = []
    for label, k, _K, _point in path.vertices:
        if any(any(_same_modulo_lattice(k, a) for a in arms) for _name, arms in stars):
            continue
        arms, _order = star_arms(k, parent_rotations)
        stars.append((_parent_point_name(mapping, k) or label, arms))
    names, ks, orbit = [], [], []
    for name, arms in stars:
        names += [name] * len(arms)
        ks.append(arms)
        orbit.append(arm_orbits(arms, child_rotations_parent))
    ks = np.vstack(ks)
    bands = unfold(child_phonon, mapping, ks)
    K = ks @ mapping.M.T.astype(float)
    condensed = np.all(np.abs(K - np.rint(K)) < 1e-9, axis=1)
    on_path = np.array([any(_same_modulo_lattice(a, q) for q in path.kpoints) for a in ks])
    return StarArms(
        star=names,
        k_parent=ks,
        K_child_reduced=_reduce_unit(K),
        child_point=[_child_point_name(mapping, Ki) for Ki in K],
        condensed=condensed,
        on_path=on_path,
        orbit=np.concatenate(orbit),
        frequencies=bands.frequencies,
        weights_canonical=bands.weights_canonical,
    )


@dataclass
class DomainAverage:
    """The unfolded spectrum averaged over the orientation domains (:func:`domain_average`).

    Attributes:
        path_index: (n_rows,) path k of each class row.
        k_parent: (n_rows, 3) representative arm (the first arm of the class).
        multiplicity: (n_rows,) m, arms in the class.
        n_arms: (n_k,) n, arms of the star of each path k.
        frequencies: (n_rows, n_band).
        weights_canonical: (n_rows, n_band) of the representative.
        weights: (n_rows, n_band) ``weights_canonical * m / n``.
        n_domains: ``|parent point group| / |child point group|``.
        sum_rule_max_deviation: max over the path k of ``|sum of its rows - 3 n_parent|``.
        orbit_spectral_max_deviation: (THz, weight), largest difference between
            the arms of one class (all arms are unfolded).
        n_arms_unfolded: Arms unfolded in all.
    """

    path_index: np.ndarray
    k_parent: np.ndarray
    multiplicity: np.ndarray
    n_arms: np.ndarray
    frequencies: np.ndarray
    weights_canonical: np.ndarray
    weights: np.ndarray
    n_domains: int
    sum_rule_max_deviation: float
    orbit_spectral_max_deviation: tuple
    n_arms_unfolded: int


def domain_average(
    child_phonon,
    mapping: ParentMapping,
    kpoints,
    *,
    parent_rotations=None,
    child_rotations_parent=None,
    degeneracy_tolerance: float = DEGENERACY_TOLERANCE,
) -> DomainAverage:
    """Average the unfolded spectrum over every arm of the star of each k.

    The average over the ``|G_parent| / |G_child|`` orientation domains of the
    child (a twinned crystal): all arms of the star of each k are unfolded;
    arms related by a child operation or by time reversal form a class
    (:func:`arm_orbits`) and the first arm of each class carries the weight
    ``m / n`` of its m arms out of n.  The arms of one class are all unfolded
    and compared (``orbit_spectral_max_deviation``).
    """
    parent_rotations, child_rotations_parent = _rotation_sets(
        mapping, parent_rotations, child_rotations_parent
    )
    kpoints = np.atleast_2d(np.asarray(kpoints, dtype=float))
    arms_list = [star_arms(k, parent_rotations)[0] for k in kpoints]
    n_arms = np.array([len(a) for a in arms_list], dtype=np.int64)
    rows_k = np.vstack(arms_list)
    bands = unfold(child_phonon, mapping, rows_k, degeneracy_tolerance=degeneracy_tolerance)
    freqs, w_c = bands.frequencies, bands.weights_canonical
    rep, mult, pidx = [], [], []
    dev_f = dev_w = 0.0
    start = 0
    for ik in range(len(kpoints)):
        idx = np.arange(start, start + n_arms[ik])
        start += int(n_arms[ik])
        classes = arm_orbits(rows_k[idx], child_rotations_parent)
        for c in range(int(classes.max()) + 1):
            members = idx[classes == c]
            r = int(members[0])
            rep.append(r)
            mult.append(len(members))
            pidx.append(ik)
            for m in members[1:]:
                dev_f = max(dev_f, float(np.abs(freqs[m] - freqs[r]).max()))
                dev_w = max(dev_w, float(np.abs(w_c[m] - w_c[r]).max()))
    rep = np.array(rep, dtype=np.int64)
    mult = np.array(mult, dtype=np.int64)
    pidx = np.array(pidx, dtype=np.int64)
    weights = w_c[rep] * (mult / n_arms[pidx])[:, None]
    total = np.bincount(pidx, weights=weights.sum(axis=1), minlength=len(kpoints))
    n_parent = len(mapping.parent_cell)
    return DomainAverage(
        path_index=pidx,
        k_parent=rows_k[rep],
        multiplicity=mult,
        n_arms=n_arms,
        frequencies=freqs[rep],
        weights_canonical=w_c[rep],
        weights=weights,
        n_domains=int(len(parent_rotations) // len(child_rotations_parent)),
        sum_rule_max_deviation=float(np.abs(total - 3 * n_parent).max()),
        orbit_spectral_max_deviation=(dev_f, dev_w),
        n_arms_unfolded=int(len(rows_k)),
    )


# ---------------------------------------------------------------------------
# the child Gamma point: where its modes come from
# ---------------------------------------------------------------------------
@dataclass
class GammaOrigin:
    """The child modes at K = 0 unfolded onto the parent k folding onto it (:func:`gamma_origin`).

    Attributes:
        k_fold: (N, 3) = ``mapping.folding_kpoints``; ``k_names`` their names.
        on_path: (N,) bool, the folding k is a point of the band path.
        frequencies: (n_band,) child frequencies at K = 0 (THz).
        groups: Degenerate groups (index arrays) of the child Gamma modes.
        weights: (N, n_band) P_J(k) at each folding k, canonical at each k (the modes of a
            degenerate level rotated as for ``weights_canonical`` of the path, independently
            at each k): per level, ``sum_k P_g(k)`` is the number of modes of the level; per
            mode, only the weights at one k are basis independent.
        group_labels, group_fractions, group_members: ``[N][n_groups]`` label
            (U1 rule), its share and its irrep names at each folding k; None
            without irreps.
        child_irreps: Per group, ``'<labels> [<Mulliken>] <activity>'`` of
            ``label_phonon_modes`` on the child; ``child_short`` the Mulliken
            symbols (or labels) alone.
        origin: Per group, ``(k name, label or '', P_g)`` of the k with the largest P_g.
        soft_levels: With parent phonons, every parent level below -0.1 THz at
            a folding k: dicts with ``k_name``, ``k``, ``unit``, ``frequency``,
            ``n``, ``contents`` (``[(group index, c_g, child irrep)]``, c_g >=
            0.02, descending) and ``mean_frequency`` (content-weighted).
        checks: ``'sum_over_k'`` (max of ``|sum_k P_J(k) - 1|`` per mode before the
            rotation and ``|sum_k P_g(k) - n_g|`` per level), ``'sum_rule_k'``
            (``|sum_J P_J(k) - 3 n_parent|``).
        notes: NOTE lines (child labels or Mulliken symbols missing).
    """

    k_fold: np.ndarray
    k_names: list
    on_path: np.ndarray
    frequencies: np.ndarray
    groups: list
    weights: np.ndarray
    group_labels: list | None
    group_fractions: list | None
    group_members: list | None
    child_irreps: list
    child_short: list
    origin: list
    soft_levels: list
    checks: dict
    notes: list = field(default_factory=list)


def _child_gamma_irreps(child_phonon, freqs, groups, tol: float, notes: list):
    """Labels of the child's own Gamma levels per group (label_phonon_modes, Mulliken, activity)."""
    from .phonon_activity import mulliken_symbols
    from .phonon_subgroups import label_phonon_modes

    full = ["?"] * len(groups)
    short = ["?"] * len(groups)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            modes = label_phonon_modes(child_phonon, (0.0, 0.0, 0.0), degeneracy_tolerance=tol)
            try:
                mulliken = mulliken_symbols(child_phonon)
            except Exception as exc:  # an annotation: say why it is missing
                mulliken = {}
                notes.append(
                    "NOTE: no Mulliken symbols of the child: "
                    + (" ".join(str(exc).split()) or type(exc).__name__)
                )
    except RuntimeError as exc:
        notes.append("NOTE: no labels of the child modes at GM: " + " ".join(str(exc).split()))
        return full, short
    for warning in caught:
        if not issubclass(warning.category, DeprecationWarning):
            notes.append("NOTE: " + " ".join(str(warning.message).split()))
    level_of = {}
    deviation = 0.0
    for mode in modes:
        for b in mode.band_indices:
            level_of[b - 1] = mode
            deviation = max(deviation, abs(float(freqs[b - 1]) - mode.frequency))
    if deviation > 1e-6:
        notes.append(
            "NOTE: the labelled child levels at GM differ from the unfolded ones by "
            f"{deviation:.1e} THz."
        )
    for ig, g in enumerate(groups):
        levels = list(dict.fromkeys(level_of[J] for J in g if J in level_of))
        parts, shorts = [], []
        for mode in levels:
            label = "+".join(mode.labels) if mode.labels else "?"
            symbol = (
                "+".join(mulliken.get(x, "?") for x in mode.labels)
                if mulliken and mode.labels
                else ""
            )
            activity = "/".join(mode.activity)
            parts.append(
                " ".join(x for x in (label, f"[{symbol}]" if symbol else "", activity) if x)
            )
            shorts.append(symbol if symbol and "?" not in symbol else label)
        if parts:
            full[ig] = " / ".join(parts)
            short[ig] = "/".join(shorts)
    return full, short


def gamma_origin(
    child_phonon,
    mapping: ParentMapping,
    *,
    irreps=None,
    parent_phonon=None,
    path=None,
    weight_min: float = WEIGHT_MIN,
    degeneracy_tolerance: float = DEGENERACY_TOLERANCE,
) -> GammaOrigin:
    """Where the child Gamma modes come from: their weights at the parent k folding onto K = 0.

    The child eigenvectors are computed once at K = 0 and unfolded onto each
    of the N parent k with M k integer (explicit G phase), so mode J is the
    same vector at every k and ``sum_k P_J(k) = 1``; then, at each k, the
    modes of a degenerate level are rotated canonically (as along the path),
    which keeps the level sums ``sum_k P_g(k) = n_g`` and makes the weights
    basis independent.  With ``irreps`` (from
    :func:`irrep_resolve`) every child level gets its parent label at each k;
    with ``parent_phonon`` the unstable parent levels are followed into the
    child: ``c_g(S) = sum_{J in g} sum_{s in S} |<e_s(k)|u_J(k)>|^2``.

    Args:
        child_phonon, mapping: As for :func:`unfold`.
        irreps: :class:`UnfoldedIrreps` (its basis is reused), or None.
        parent_phonon: ``phonopy.Phonopy`` of the parent, or None.
        path: The :class:`ParentPath` (marks the folding k on the path), or None.
        weight_min: Labels below it are ``'-'``.
        degeneracy_tolerance: THz, degenerate child levels.

    Returns:
        :class:`GammaOrigin`.
    """
    _check_child_cell(child_phonon.primitive, mapping.child_cell)
    k_fold = np.asarray(mapping.folding_kpoints, dtype=float)
    N = len(k_fold)
    result = _run_qpoints(child_phonon, np.zeros((1, 3)), with_eigenvectors=True)
    freqs = np.array(result.frequencies, dtype=float)[0]
    E = np.array(result.eigenvectors)[0]
    x = np.array(mapping.child_cell.scaled_positions, dtype=float)
    uvec, weights, _G = unfold_eigenvectors(
        np.repeat(E[None], N, axis=0),
        np.zeros((N, 3)),
        k_fold,
        mapping.M,
        mapping.kappa,
        x,
        mapping.ideal_frac_child,
        gauge="ideal",
        origin_shift=mapping.origin_shift,
    )
    groups = degenerate_groups(freqs, degeneracy_tolerance)
    # sums over k per mode, before the rotation below (one vector per mode at every k)
    per_mode_sum = float(np.abs(weights.sum(axis=0) - 1.0).max())
    # the canonical rotation of the path (canonical_degenerate_rotation) at each folding k, so that
    # the weight of one mode of a degenerate child level does not depend on the basis the
    # eigensolver chose inside the level; the group sums are unchanged, and after the rotation
    # mode J is a different vector at each k (its weights no longer add up to 1 over the k, the
    # weights of a level add up to its number of modes)
    uvec, weights, _rotations = canonical_degenerate_rotation(
        np.repeat(freqs[None], N, axis=0), uvec, degeneracy_tolerance
    )
    gP = np.array([[float(weights[i][g].sum()) for g in groups] for i in range(N)])
    on_path = np.array(
        [path is not None and any(_same_modulo_lattice(k, q) for q in path.kpoints) for k in k_fold]
    )
    notes: list = []
    child_full, child_short = _child_gamma_irreps(
        child_phonon, freqs, groups, degeneracy_tolerance, notes
    )
    group_labels = group_fractions = group_members = None
    resolved = []
    if irreps is not None:
        from .phonon_unfold_irreps import resolve_point

        group_labels, group_fractions, group_members = [], [], []
        for i in range(N):
            r = resolve_point(
                irreps.basis,
                mapping,
                k_fold[i],
                uvec[i],
                freqs,
                weights[i],
                weight_min=weight_min,
                degeneracy_tolerance=degeneracy_tolerance,
            )
            resolved.append(r)
            group_labels.append([r.labels[g[0]] for g in groups])
            group_fractions.append([float(r.fractions[g[0]]) for g in groups])
            group_members.append([r.members[g[0]] for g in groups])
    origin = []
    for ig in range(len(groups)):
        io = int(np.argmax(gP[:, ig]))
        label = group_labels[io][ig] if group_labels is not None else ""
        origin.append((mapping.folding_names[io], label, float(gP[io, ig])))
    soft = []
    if parent_phonon is not None:
        from .phonon_unfold_irreps import _parent_eigen, _unit_weights, irrep_weights

        pf, pv = _parent_eigen(parent_phonon, mapping, k_fold)
        u_par = rotate_unfolded_vectors(uvec, mapping.frame_rotation)
        overlap = (
            np.abs(np.einsum("kba,kja->kjb", pv.conj(), u_par)) ** 2
        )  # (N, n_band, n_parent bands)
        for i in range(N):
            if resolved:
                pset = resolved[i].pset
                unit_w = _unit_weights(pset, irrep_weights(pv[i], pset.projectors))
                units = [pset.units[u] for u in np.argmax(unit_w, axis=1)]
            else:
                units = [""] * len(pf[i])
            order = np.argsort(pf[i], kind="stable")
            used = set()
            for b in order:
                if b in used or pf[i][b] >= -0.1:
                    continue
                level = [
                    j
                    for j in order
                    if j not in used and abs(pf[i][j] - pf[i][b]) <= 1e-3 and units[j] == units[b]
                ]
                used.update(level)
                c = overlap[i][:, level].sum(axis=1)
                cg = np.array([float(c[g].sum()) for g in groups])
                chosen = [ig for ig in np.argsort(-cg, kind="stable") if cg[ig] >= 0.02]
                soft.append(
                    {
                        "k_name": mapping.folding_names[i],
                        "k": k_fold[i],
                        "unit": units[b],
                        "frequency": float(np.mean(pf[i][level])),
                        "n": len(level),
                        "contents": [(int(ig), float(cg[ig]), child_short[ig]) for ig in chosen],
                        "mean_frequency": float(np.sum(c * freqs) / np.sum(c))
                        if np.sum(c) > 0
                        else 0.0,
                    }
                )
    n_parent = len(mapping.parent_cell)
    checks = {
        "sum_over_k": max(
            per_mode_sum,
            max(abs(float(gP[:, ig].sum()) - len(g)) for ig, g in enumerate(groups)),
        ),
        "sum_rule_k": float(np.abs(weights.sum(axis=1) - 3 * n_parent).max()),
    }
    return GammaOrigin(
        k_fold=k_fold,
        k_names=list(mapping.folding_names),
        on_path=on_path,
        frequencies=freqs,
        groups=groups,
        weights=weights,
        group_labels=group_labels,
        group_fractions=group_fractions,
        group_members=group_members,
        child_irreps=child_full,
        child_short=child_short,
        origin=origin,
        soft_levels=soft,
        checks=checks,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# the irreps (implemented in crystod.phonon_unfold_irreps; re-exported here)
# ---------------------------------------------------------------------------
_IRREP_EXPORTS = (
    "UnfoldedIrreps",
    "irrep_resolve",
    "format_label",
    "ParentIrrepBasis",
    "IrrepProjectorSet",
    "irrep_weights",
    "child_operations_in_parent",
)


def __getattr__(name: str):
    if name in _IRREP_EXPORTS:
        from . import phonon_unfold_irreps

        return getattr(phonon_unfold_irreps, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# output files
# ---------------------------------------------------------------------------
@dataclass
class UnfoldRun:
    """Everything one ``crystod-phonon --unfold`` run computed (returned by :func:`run`)."""

    inputs: UnfoldInputs
    mapping: ParentMapping
    path: ParentPath
    bands: UnfoldedBands
    arms: StarArms
    domain_average: DomainAverage
    irreps: object | None
    gamma: GammaOrigin
    parent_frequencies: np.ndarray | None
    weight_min: float = WEIGHT_MIN
    files: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    checks: dict = field(default_factory=dict)
    command: str = "crystod-phonon --unfold"


def write_dat(
    path,
    mapping: ParentMapping,
    path_obj: ParentPath,
    bands: UnfoldedBands,
    davg: DomainAverage,
    irreps=None,
    *,
    child_text: str = "child",
    parent_text: str = "parent",
    weight_min: float = WEIGHT_MIN,
) -> None:
    """The ``.dat`` table: dataset 0 (one domain), dataset 1 (domain average); gnuplot index 0/1."""
    n_k, n_band = bands.frequencies.shape
    n3 = 3 * len(mapping.parent_cell)
    freqs = np.round(bands.frequencies, 6) + 0.0  # no -0.000000
    davg_freqs = np.round(davg.frequencies, 6) + 0.0
    lines = [
        "# crystod-phonon --unfold: phonon bands of a child unfolded onto the Brillouin zone of "
        "its parent",
        f"# child  {child_text} ({mapping.child_space_group}, {len(mapping.child_cell)} atoms); "
        f"parent {parent_text} ({mapping.parent_space_group}, {len(mapping.parent_cell)} atoms)",
        f"# M = {_matrix_text(mapping.M)} (A_child = M A_parent, k_child = M k_parent), N = "
        f"{mapping.N}, origin shift p = {_vec(mapping.origin_shift)}",
        f"# path {path_obj.path_text}, {n_k} parent k, fractional in the parent primitive "
        "reciprocal basis;",
        "# distance: 1/A without 2 pi in the reciprocal metric of A_pp = inv(M) A_child",
        "# frequency: THz (imaginary as negative); mode: child mode number at K = M k (1-based, "
        "ascending frequency)",
        "# weight: canonical unfolding weight in [0, 1] (degenerate child modes, |df| <= 1e-4 "
        "THz, rotated so that the",
        "#   weights do not depend on the eigenvector basis); at every k the weights of a dataset "
        f"add up to 3 n_parent = {n3}",
        "# group_weight: P_g of the mode's degenerate group; label: the group's parent irrep (>= "
        "50 % of P_g, else the",
        "#   smallest set the child symmetry cannot separate, written A/B, else mixed); '-' below "
        f"{weight_min:g}; fraction: share",
        "#   of P_g held by the label's irreps (0 for mixed and -)"
        + ("" if irreps is not None else "; no labels (--no-irreps)"),
        "#",
        "# dataset 0: one domain, the path arm k itself",
        "# k_index distance kx ky kz mode frequency weight group_weight label fraction",
    ]
    for ik in range(n_k):
        kx, ky, kz = path_obj.kpoints[ik]
        d = path_obj.distances[ik]
        for J in range(n_band):
            label = irreps.group_label[ik, J] if irreps is not None else "-"
            fraction = irreps.group_fraction[ik, J] if irreps is not None else 0.0
            lines.append(
                "%5d %10.6f %9.6f %9.6f %9.6f %4d %12.6f %10.8f %10.8f %s %6.4f"
                % (
                    ik,
                    d,
                    kx,
                    ky,
                    kz,
                    J + 1,
                    freqs[ik, J],
                    bands.weights_canonical[ik, J],
                    bands.group_weight[ik, J],
                    label,
                    fraction,
                )
            )
        lines.append("")
    lines += [
        "",  # with the blank line after the last k: two blank lines, gnuplot's dataset break
        f"# dataset 1: average over the {_domains(davg.n_domains)} (all arms of the star of k, one "
        "row per class of equivalent",
        "#   arms and mode; weight = canonical weight of the class representative x multiplicity "
        "/ n_arms)",
        "# k_index distance class multiplicity n_arms akx aky akz mode frequency weight",
    ]
    rows_of = {}
    for r, ik in enumerate(davg.path_index):
        rows_of.setdefault(int(ik), []).append(r)
    for ik in range(n_k):
        d = path_obj.distances[ik]
        for c, r in enumerate(rows_of.get(ik, [])):
            ax, ay, az = davg.k_parent[r]
            for J in range(n_band):
                lines.append(
                    "%5d %10.6f %4d %3d %3d %9.6f %9.6f %9.6f %4d %12.6f %10.8f"
                    % (
                        ik,
                        d,
                        c + 1,
                        davg.multiplicity[r],
                        davg.n_arms[ik],
                        ax,
                        ay,
                        az,
                        J + 1,
                        davg_freqs[r, J],
                        davg.weights[r, J],
                    )
                )
        lines.append("")
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def _str_array(values) -> np.ndarray:
    values = list(values)
    return np.array([str(v) for v in values], dtype=str) if values else np.array([], dtype="<U1")


def write_npz(path, run: UnfoldRun) -> str:
    """All arrays of a run (``np.savez_compressed``, no object arrays); returns the file written."""
    import crystod

    m, p, b, a, dv, g = run.mapping, run.path, run.bands, run.arms, run.domain_average, run.gamma
    data = {
        "conventions": np.array(CONVENTIONS),
        "crystod_version": np.array(str(getattr(crystod, "__version__", ""))),
        "command": np.array(run.command),
        "child_poscar": np.array(run.inputs.child_poscar),
        "parent_poscar": np.array(run.inputs.parent_poscar),
        "child_force_file": np.array(run.inputs.child_force_file),
        "parent_force_file": np.array(run.inputs.parent_force_file or ""),
        "child_dim": np.array(run.inputs.child_dim, dtype=np.int64),
        "parent_dim": np.array(run.inputs.parent_dim or (0, 0, 0), dtype=np.int64),
        "M": m.M.astype(np.int64),
        "N": np.array(m.N, dtype=np.int64),
        "origin_shift": m.origin_shift,
        "setting_P": np.asarray(m.setting_P if m.setting_P is not None else m.M.T, dtype=float),
        "setting_p": np.asarray(
            m.setting_p if m.setting_p is not None else m.origin_shift, dtype=float
        ),
        "equivalent_settings": np.array(m.equivalent_settings or [m.M], dtype=np.int64),
        "A_child": m.A_child,
        "A_parent": m.A_parent,
        "A_pp": m.A_pp,
        "frame_rotation": m.frame_rotation,
        "child_symbols": _str_array(m.child_cell.symbols),
        "parent_symbols": _str_array(m.parent_cell.symbols),
        "child_frac": np.array(m.child_cell.scaled_positions, dtype=float),
        "parent_frac": np.array(m.parent_cell.scaled_positions, dtype=float),
        "ideal_frac_child": m.ideal_frac_child,
        "displacement_cart": m.displacement_cart,
        "kappa": m.kappa.astype(np.int64),
        "k_parent": b.k_parent,
        "K_child": b.K_child,
        "segment_index": p.segment_index.astype(np.int64),
        "distances": p.distances,
        "tick_positions": p.tick_positions,
        "tick_labels": _str_array(p.tick_labels),
        "vertex_child_points": _str_array(_tick_child_points(p)),
        "path_vertex_labels": _str_array(v[0] for v in p.vertices),
        "path_vertex_k_parent": np.array([v[1] for v in p.vertices], dtype=float),
        "path_vertex_child_points": _str_array(v[3] for v in p.vertices),
        "frequencies": b.frequencies,
        "weights": b.weights,
        "weights_canonical": b.weights_canonical,
        "group_weight": b.group_weight,
        "degenerate_group": b.degenerate_group.astype(np.int64),
        "vectors_canonical": b.vectors.astype(np.complex128),
        "arm_star": _str_array(a.star),
        "arm_child_point": _str_array(a.child_point),
        "arm_k_parent": a.k_parent,
        "arm_K_child_reduced": a.K_child_reduced,
        "arm_condensed": a.condensed.astype(bool),
        "arm_on_path": a.on_path.astype(bool),
        "arm_class": a.orbit.astype(np.int64),
        "arm_frequencies": a.frequencies,
        "arm_weights_canonical": a.weights_canonical,
        "davg_path_index": dv.path_index.astype(np.int64),
        "davg_multiplicity": dv.multiplicity.astype(np.int64),
        "davg_k_parent": dv.k_parent,
        "davg_n_arms": dv.n_arms.astype(np.int64),
        "davg_frequencies": dv.frequencies,
        "davg_weights": dv.weights,
        "n_domains": np.array(dv.n_domains),
        "gamma_k_parent": g.k_fold,
        "gamma_k_names": _str_array(g.k_names),
        "gamma_frequencies": g.frequencies,
        "gamma_weights": g.weights,
        "gamma_child_irrep": _str_array(
            g.child_irreps[_group_of(g.groups, J)] for J in range(len(g.frequencies))
        ),
        "checks_keys": _str_array(run.checks.keys()),
        "checks_values": np.array(list(run.checks.values()), dtype=float),
        "timings_keys": _str_array(run.timings.keys()),
        "timings_s": np.array(list(run.timings.values()), dtype=float),
    }
    if run.parent_frequencies is not None:
        data["parent_frequencies"] = run.parent_frequencies
    ir = run.irreps
    if ir is not None:
        data.update(
            {
                "irrep_units": _str_array(ir.units),
                "irrep_kname": _str_array(ir.kname),
                "irrep_weights": ir.weights,
                "group_label": np.asarray(ir.group_label, dtype=str),
                "group_fraction": ir.group_fraction,
            }
        )
        if ir.parent_labels is not None:
            data["parent_labels"] = np.asarray(ir.parent_labels, dtype=str)
        if g.group_labels is not None:
            data["gamma_group_label"] = np.array(
                [
                    [g.group_labels[i][_group_of(g.groups, J)] for J in range(len(g.frequencies))]
                    for i in range(len(g.k_fold))
                ],
                dtype=str,
            )
    name = _npz_name(str(path))
    np.savez_compressed(name, **data)
    return name


def _group_of(groups, J) -> int:
    for ig, g in enumerate(groups):
        if J in g:
            return ig
    return -1


def plot(pdf_path, run: UnfoldRun, *, weight_min: float = WEIGHT_MIN) -> int:
    """Write the PDF (page 1: bands and star arms; page 2: irreps). Returns the number of pages."""
    from .phonon_unfold_plot import plot_unfold

    return plot_unfold(pdf_path, run, weight_min=weight_min)


# ---------------------------------------------------------------------------
# terminal blocks
# ---------------------------------------------------------------------------
def _matrix_fraction_text(matrix) -> str:
    return "  ".join(
        " ".join(_frac_text(v) for v in row) for row in np.asarray(matrix, dtype=float)
    )


def _cell_line(poscar: str, unitcell, cell, matrix, space_group: str) -> str:
    text = f"{_display_path(poscar)}: {space_group}, {len(unitcell)} atoms, "
    if np.allclose(matrix, np.eye(3)):
        return text + "cell used as given"
    return text + (
        f"conventional cell; unfolded on its primitive cell of {len(cell)} atoms (phonopy "
        f"primitive matrix {_matrix_fraction_text(matrix)})"
    )


def _print_setup(inputs: UnfoldInputs, mapping: ParentMapping) -> None:
    from .symmetry_mode import SettingTransformation

    reduced = not (
        np.allclose(inputs.child_primitive_matrix, np.eye(3))
        and np.allclose(inputs.parent_primitive_matrix, np.eye(3))
    )
    child_shown = _display_path(inputs.child_poscar)
    parent_shown = _display_path(inputs.parent_poscar)
    lines = [
        "",
        "* Unfolding setup *",
        _kv(
            "child",
            _cell_line(
                inputs.child_poscar,
                inputs.child_unitcell,
                inputs.child_cell,
                inputs.child_primitive_matrix,
                mapping.child_space_group,
            ),
        ),
        _kv(
            "child forces",
            f"{_display_path(inputs.child_force_file)}, supercell "
            f"{' '.join(str(v) for v in inputs.child_dim)}",
        ),
        _kv(
            "parent",
            _cell_line(
                inputs.parent_poscar,
                inputs.parent_unitcell,
                inputs.parent_cell,
                inputs.parent_primitive_matrix,
                mapping.parent_space_group,
            ),
        ),
    ]
    if inputs.parent_force_file is None:
        lines.append(_kv("parent forces", "- (no --parent-dim: no parent bands)"))
    else:
        lines.append(
            _kv(
                "parent forces",
                f"{_display_path(inputs.parent_force_file)}, supercell "
                f"{' '.join(str(v) for v in inputs.parent_dim)}",
            )
        )
    n_eq = len(mapping.equivalent_settings)
    if mapping.source == "--trans-mat":
        W, w = mapping.parent_operation
        M0 = mapping.equivalent_settings[0]
        lines.append(
            _kv(
                "setting",
                f"--trans-mat {_matrix_text(mapping.M)}: the crystod-group setting "
                f"{_matrix_text(M0)}",
            )
        )
        lines.append(
            _kv(
                "",
                f"transformed by the parent operation W = {_matrix_text(W)}, w = {_vec(w)} ({n_eq} "
                "equivalent settings)",
            )
        )
        heading = "parent -> child of this setting in crystod-group's convention (columns of P:"
    else:
        lines.append(
            _kv(
                "setting",
                f"crystod-group --supergroup-cif {parent_shown} --subgroup-cif {child_shown}",
            )
        )
        lines.append(_kv("", f"(1 of {n_eq} equivalent settings)"))
        heading = "parent -> child as crystod-group prints it (columns of P:"
    lines.append(_kv("transformation", heading + " the child basis in parent units,"))
    lines.append(
        _kv(
            "",
            "(a', b', c') = (a, b, c) P; last column: the origin shift p, x_parent = P x_child + "
            "p):",
        )
    )
    setting = SettingTransformation(mapping.setting_P, mapping.setting_p, "input")
    lines += setting.lines(parent_shown, child_shown)[2:5]
    key = "M (rows, parent primitive cell -> child cell)" if reduced else "M (rows)"
    lines.append(
        _kv(
            key,
            f"{_matrix_text(mapping.M)}  (A_child = M A_parent, k_child = M k_parent; the form "
            "--trans-mat takes)",
        )
    )
    lines.append(_kv("origin shift", f"p = {_vec(mapping.origin_shift)} (parent fractional)"))
    lines.append(
        _kv(
            "cells",
            f"N = {mapping.N} parent cell{'s' if mapping.N != 1 else ''} in the child cell; "
            f"n_child = {len(mapping.child_cell)}, n_parent = {len(mapping.parent_cell)} atoms "
            f"({' '.join(mapping.parent_cell.symbols)})",
        )
    )
    st = mapping.strain
    lines.append(
        _kv(
            "pseudo-parent",
            "A_pp = inv(M) A_child: a = "
            + " ".join(f"{v:.4f}" for v in st["lengths_pp"])
            + " A, angles "
            + " ".join(f"{v:.3f}" for v in st["angles_pp"])
            + " deg",
        )
    )
    lengths = st["parent_lengths"]
    parent_a = (
        f"a = {lengths[0]:.4f} A"
        if max(lengths) - min(lengths) < 1e-4
        else "a = " + " ".join(f"{v:.4f}" for v in lengths) + " A"
    )
    lines.append(
        _kv(
            "strain",
            "lengths "
            + " ".join(f"{v:.2f}" for v in st["length_strain_percent"])
            + f" %, volume {st['volume_strain_percent']:.2f} % against the parent cell "
            + f"({parent_a}); "
            f"rotation {st['rotation_deg']:.1f} deg",
        )
    )
    lines.append(
        _kv(
            "displacement",
            f"max {mapping.max_displacement:.4f} A ({mapping.max_displacement_species}), rms "
            f"{mapping.rms_displacement:.4f} A in the child lattice; "
            f"{mapping.max_displacement_parent_lattice:.4f} A in the strain-free parent lattice",
        )
    )
    if mapping.sma_max_displacement is not None:
        lines.append(_kv("", f"(crystod-group: {mapping.sma_max_displacement:.4f} A)"))
    if mapping.sma_modes:
        lines.append(
            _kv(
                "condensed",
                ", ".join(f"{irrep} {amp:.4f}" for _k, irrep, amp in mapping.sma_modes)
                + " A (crystod-group mode amplitudes)",
            )
        )
    lines.append(
        _kv(
            "folding",
            "parent k folding onto the child Gamma point: "
            + ", ".join(
                f"{name} {_vec(k)}"
                for name, k in zip(mapping.folding_names, mapping.folding_kpoints)
            ),
        )
    )
    lines += [f"  {note}" for note in mapping.notes]
    print("\n".join(lines))


def _print_path(path: ParentPath) -> None:
    source = (
        "seekpath, parent primitive basis"
        if path.source == "seekpath"
        else "manual --band, parent primitive basis"
    )
    lines = [
        "",
        "* Unfolding path *",
        _kv(
            "path",
            f"{path.path_text} ({source}), {path.npoints} points per leg, {len(path.kpoints)} k",
        ),
        _kv("distances", "1/A without 2 pi, reciprocal metric of the pseudo-parent cell A_pp"),
        f"{'vertex':<8}{'k (parent)':<21}{'K = M k (child)':<22}child point",
    ]
    for label, k, K, point in path.vertices:
        lines.append(f"{label:<8}{_vec(k):<21}{_vec(K):<22}{point}")
    if path.note:
        lines.append(f"  {path.note}")
    print("\n".join(lines))


def _print_arms(
    mapping: ParentMapping, arms: StarArms, davg: DomainAverage, n_child_ops: int, n_parent_ops: int
) -> None:
    lines = [
        "",
        "* Star arms of the path vertices *",
        _kv(
            "point groups",
            f"child {n_child_ops} operations ({mapping.child_symbol}), parent {n_parent_ops} "
            f"({mapping.parent_symbol}): {_domains(davg.n_domains)}",
        ),
        f"{'star':<6}{'arm k (parent)':<23}{'K = M k mod 1':<21}"
        f"{'child point':<13}{'to child GM':<13}{'path':<6}class",
    ]
    for i in range(len(arms.star)):
        letter = (
            chr(ord("a") + int(arms.orbit[i])) if arms.orbit[i] < 26 else str(int(arms.orbit[i]))
        )
        lines.append(
            f"{arms.star[i]:<6}{_vec(arms.k_parent[i]):<23}{_vec(arms.K_child_reduced[i]):<21}"
            f"{arms.child_point[i]:<13}{'yes' if arms.condensed[i] else '-':<13}"
            f"{'path' if arms.on_path[i] else '-':<6}{letter}"
        )
    lines.append(
        "  (arms with the same class letter are related by a child operation or by time reversal "
        "and give the same"
    )
    lines.append(
        "   unfolded spectrum; 'to child GM' = the arm folds onto the child Gamma point, where a "
        "condensed distortion lives)"
    )
    for i in range(len(arms.star)):
        if arms.condensed[i] and not arms.on_path[i]:
            lines += _note(
                f"the condensed arm {arms.star[i]} {_vec(arms.k_parent[i])} is not on the band "
                "path; it is shown in the lower panel of page 1 and in the child GM mode origin."
            )
    dev_f, dev_w = davg.orbit_spectral_max_deviation
    if dev_f > ORBIT_SPECTRUM_TOLERANCE[0] or dev_w > ORBIT_SPECTRUM_TOLERANCE[1]:
        lines += _note(
            f"arms of one class differ by {dev_f:.1e} THz and {dev_w:.1e} in weight (force "
            "constants not fully symmetric); the domain average uses the first arm of each class."
        )
    print("\n".join(lines))


def _levels_text(
    units, freqs, mulliken: dict, kname: str, tol: float = DEGENERACY_TOLERANCE
) -> str:
    """Parent levels ``'GM4- [T1u] -1.4465 x3, ...'`` from per-band units and frequencies."""
    from .phonon_unfold_irreps import format_label

    order = np.argsort(freqs, kind="stable")
    out, i = [], 0
    while i < len(order):
        j = i
        while (
            j + 1 < len(order)
            and abs(freqs[order[j + 1]] - freqs[order[j]]) <= tol
            and units[order[j + 1]] == units[order[i]]
        ):
            j += 1
        unit = units[order[i]]
        shown = format_label(unit, _unit_members_of(unit), kname, mulliken)
        n = j - i + 1
        out.append(f"{shown} {_thz(freqs[order[i]], 0)}" + (f" x{n}" if n > 1 else ""))
        i = j + 1
    return ", ".join(out)


def _unit_members_of(unit: str) -> tuple:
    """Irrep names of a unit name.

    ``'R4+'`` -> ``('R4+',)``, ``'U2++U3+'`` -> ``('U2+', 'U3+')``, ``'S1+S2'`` -> ``('S1', 'S2')``.
    """
    import re

    name = re.compile(r"^[A-Z]+\d+[+-]?$")
    if name.match(unit):
        return (unit,)
    for i, character in enumerate(unit):
        if character == "+" and name.match(unit[:i]) and name.match(unit[i + 1 :]):
            return (unit[:i], unit[i + 1 :])
    return (unit,)


def _fractions_text(f, units, limit: int = 4) -> str:
    """``'R4+ 0.931, R5+ 0.069'``: the units with f >= 0.01, largest first (fractions equal to
    1e-9 in the natural order of the units, as the label rule takes them), at most ``limit``."""
    order = [u for u in np.argsort(-np.round(f, 9), kind="stable") if f[u] >= 0.01][:limit]
    return ", ".join(f"{units[u]} {f[u]:.3f}" for u in order)


def _print_irreps(path: ParentPath, bands: UnfoldedBands, irreps, mapping: ParentMapping) -> None:
    from .phonon_unfold_irreps import (
        _group_labels,
        _inseparable_classes,
        _unit_weights,
        format_label,
        irrep_weights,
    )

    lines = ["", "* Unfolded irreps at special points *"]
    seen = []
    basis = irreps.basis
    for label, k, _K, point in path.vertices:
        if any(np.allclose(k, s) for s in seen):
            continue
        seen.append(k)
        ik = int(np.flatnonzero(np.all(np.abs(path.kpoints - k) < 1e-12, axis=1))[0])
        kname = irreps.kname[ik]
        pset = basis.projector_set(k)
        lines.append(f"{label} {_vec(k)} -> child {point}")
        if irreps.parent_labels is not None:
            lines.append(
                "  parent levels: "
                + _levels_text(
                    list(irreps.parent_labels[ik]),
                    irreps.parent_frequencies[ik],
                    irreps.mulliken,
                    kname,
                )
            )
        classes = irreps.inseparable[ik]
        lines.append(
            "  not separated by the child symmetry: "
            + (
                ", ".join("{" + ", ".join(c) + "}" for c in classes)
                if classes
                else "none (every level is one irrep)"
            )
        )
        lines.append(f"  {'freq(THz)':>9} {'n':>3} {'P_g':>6}  {'label':<18} fractions")
        u = rotate_unfolded_vectors(bands.vectors[ik], mapping.frame_rotation)
        unit_w = _unit_weights(pset, irrep_weights(u, pset.projectors))
        cls = _inseparable_classes(pset, basis, mapping)
        _labels, _members, _fractions, groups = _group_labels(
            pset,
            cls,
            unit_w,
            bands.frequencies[ik],
            bands.weights_canonical[ik],
            0.0,
            DEGENERACY_TOLERANCE,
        )
        for g, _lab, _mem, _frac, Pg, f in groups:
            if Pg < LEVEL_TABLE_MIN:
                continue
            J = int(g[0])
            shown = format_label(
                irreps.group_label[ik, J], irreps.group_members[ik][J], kname, irreps.mulliken
            )
            lines.append(
                f"  {_thz(bands.frequencies[ik, g].mean())} {len(g):3d} {Pg:6.3f}  {shown:<18} "
                f"{_fractions_text(f, pset.units)}"
            )
    lines += textwrap.wrap(
        f"rows: the degenerate groups of child modes (|df| <= 1e-4 THz) with P_g >= "
        f"{LEVEL_TABLE_MIN:.2f}; the .dat file has all groups. The label is the irrep holding at "
        "least 50 % of P_g, else the smallest set of irreps that the child symmetry cannot "
        "separate and that together hold at least 50 % (written A/B), else mixed; two irreps "
        "that time reversal joins count as one (written S1+S2).)",
        width=100,
        initial_indent="  (",
        subsequent_indent="   ",
        break_on_hyphens=False,
    )
    lines += [f"  {note}" for note in irreps.notes]
    print("\n".join(lines))


def _print_gamma(
    gamma: GammaOrigin, mapping: ParentMapping, irreps, parent: bool, weight_min: float
) -> None:
    from .phonon_unfold_irreps import format_label

    n_band = len(gamma.frequencies)
    names = list(gamma.k_names)
    headers = [
        name if names.count(name) == 1 else name + "(" + ",".join(_frac_text(v) for v in k) + ")"
        for name, k in zip(names, gamma.k_fold)
    ]
    described = ", ".join(
        f"{name} {_vec(k)}" + ("" if on else " (not on the path)")
        for name, k, on in zip(names, gamma.k_fold, gamma.on_path)
    )
    mulliken = irreps.mulliken if irreps is not None else {}

    def shown(i, ig):  # the label at folding k i, with the Mulliken symbols at GM
        label = gamma.group_labels[i][ig]
        return format_label(label, tuple(gamma.group_members[i][ig]), names[i], mulliken)

    rows = []
    for ig, g in enumerate(gamma.groups):
        weights = [float(gamma.weights[i][g].sum()) for i in range(len(gamma.k_fold))]
        if gamma.group_labels is not None:
            cells = [
                f"{w:5.3f} " + (shown(i, ig) if w >= weight_min else "-")
                for i, w in enumerate(weights)
            ]
            name, _label, Pg = gamma.origin[ig]
            origin = f"{name} {shown(int(np.argmax(weights)), ig)} {Pg:.2f}"
        else:
            cells = [f"{w:5.3f}" for w in weights]
            origin = gamma.origin[ig][0]
        modes = ",".join(str(int(J) + 1) for J in g)
        frequency = gamma.frequencies[g].mean()
        rows.append((modes, frequency, len(g), gamma.child_irreps[ig], cells, origin))
    # columns as wide as their longest entry (at least 10, 26 and 14 characters)
    w_modes = max([10] + [len(r[0]) + 2 for r in rows])
    w_irrep = max([26] + [len(r[3]) + 2 for r in rows])
    w_cells = [
        max([14, len(h) + 2] + [len(r[4][i]) + 2 for r in rows]) for i, h in enumerate(headers)
    ]
    lines = [
        "",
        "* Child GM mode origin *",
        f"child GM (K = 0), {n_band} modes, from the parent k folding onto it: {described}",
        f"{'modes':<{w_modes}}{'freq(THz)':>9}{'n':>3}  {'child irrep':<{w_irrep}}"
        + "".join(f"{h:<{w}}" for h, w in zip(headers, w_cells))
        + "origin",
    ]
    for modes, frequency, n, irrep, cells, origin in rows:
        lines.append(
            (
                f"{modes:<{w_modes}}{_thz(frequency)}{n:3d}  {irrep:<{w_irrep}}"
                + "".join(f"{c:<{w}}" for c, w in zip(cells, w_cells))
                + origin
            ).rstrip()
        )
    lines.append(
        "  (columns: weight P_g(k) of the child level at each folding k and its label there ('-' "
        f"below {weight_min:g}); the"
    )
    lines.append("   weights of a level add up to its number of modes)")
    if parent:
        lines.append(
            "soft parent levels and the child GM levels they go into (content c_g >= 0.02):"
        )
        if not gamma.soft_levels:
            lines.append("  (none: no parent level below -0.1 THz at the folding k)")
        for level in gamma.soft_levels:
            kname = "GM" if np.allclose(level["k"], 0.0) else level["k_name"]
            unit = level["unit"]
            shown = (
                format_label(unit, _unit_members_of(unit), kname, irreps.mulliken)
                if (irreps and unit)
                else unit
            )
            head = (
                f"  {level['k_name']} {_vec(level['k'])}"
                + (f" {shown}" if shown else "")
                + f" {_thz(level['frequency'], 0)}"
                + (f" x{level['n']}" if level["n"] > 1 else "")
                + ": "
            )
            parts = [
                f"{float(gamma.frequencies[gamma.groups[ig]].mean()):.3f} {short} {c:.2f}"
                for ig, c, short in level["contents"][:8]
            ]
            if len(level["contents"]) > 8:
                parts.append("...")
            lines.append(head + ", ".join(parts) + f"; mean {level['mean_frequency']:.2f} THz")
    lines += [f"  {note}" for note in gamma.notes]
    print("\n".join(lines))


def _group_counts(bands: UnfoldedBands, irreps, weight_min: float):
    """Groups with P_g >= weight_min over the path: (one unit, sets A/B, mixed)."""
    single = sets = mixed = 0
    n_k = bands.frequencies.shape[0]
    for ik in range(n_k):
        for ig in np.unique(bands.degenerate_group[ik]):
            J = int(np.flatnonzero(bands.degenerate_group[ik] == ig)[0])
            if bands.group_weight[ik, J] < weight_min:
                continue
            label = irreps.group_label[ik, J]
            if label == "mixed":
                mixed += 1
            elif "/" in label:
                sets += 1
            elif label != "-":
                single += 1
    return single, sets, mixed


def _summary_checks(run: UnfoldRun) -> dict:
    b, dv, g, ir = run.bands, run.domain_average, run.gamma, run.irreps
    checks = {
        "sum_rule_path": b.sum_rule_max_deviation,
        "sum_rule_domain_average": dv.sum_rule_max_deviation,
        "arm_class_frequency": dv.orbit_spectral_max_deviation[0],
        "arm_class_weight": dv.orbit_spectral_max_deviation[1],
        "gamma_sum_over_k": g.checks["sum_over_k"],
        "gamma_sum_rule_k": g.checks["sum_rule_k"],
        "max_displacement": run.mapping.max_displacement,
        "max_displacement_parent_lattice": run.mapping.max_displacement_parent_lattice,
    }
    if run.mapping.sma_max_displacement is not None:
        checks["crystod_group_max_displacement"] = run.mapping.sma_max_displacement
    if ir is not None:
        checks.update({f"irreps_{key}": float(value) for key, value in ir.checks.items()})
        checks.update({f"labels_{key}": float(value) for key, value in ir.cache.items()})
    return checks


def _print_summary(run: UnfoldRun, pages: int) -> None:
    b, dv, ir, m = run.bands, run.domain_average, run.irreps, run.mapping
    n_k, n_band = b.frequencies.shape
    n3 = 3 * len(m.parent_cell)
    lines = [
        "",
        "* Unfolding summary *",
        _kv(
            "k points",
            f"{n_k} on the path; {dv.n_arms_unfolded} arms of their stars unfolded for the domain "
            f"average ({_domains(dv.n_domains)})",
        ),
        _kv(
            "modes",
            f"{n_band} child modes at every k; {n_k * n_band} (k, mode) pairs, "
            f"{int(np.sum(b.weights_canonical >= run.weight_min))} with weight >= "
            f"{run.weight_min:g}",
        ),
        _kv(
            "sum rule",
            f"max |sum_J P_J(k) - {n3}| = {b.sum_rule_max_deviation:.1e} on the path, "
            f"{dv.sum_rule_max_deviation:.1e} for the domain average",
        ),
        _kv(
            "arm classes",
            f"arms of one class agree to {dv.orbit_spectral_max_deviation[0]:.1e} THz and "
            f"{dv.orbit_spectral_max_deviation[1]:.1e} in weight (child operations and time "
            "reversal)",
        ),
    ]
    if ir is not None:
        c = ir.checks
        projectors = max(c["completeness"], c["idempotency"], c["hermiticity"])
        text = (
            f"projectors {projectors:.1e} (completeness, idempotency, hermiticity); per-irrep sum "
            f"rule {c['per_irrep_sum_rule']:.1e};"
        )
        lines.append(_kv("irreps", text))
        second = (
            f"parent levels pure to {c['parent_purity']:.1e}; "
            if ir.parent_labels is not None
            else ""
        ) + f"weight forbidden by the child symmetry {c['forbidden_weight']:.1e}"
        lines.append(_kv("", second))
        single, sets, mixed = _group_counts(b, ir, run.weight_min)
        cache = ir.cache
        lines.append(
            _kv(
                "labels",
                f"ISO-IR labels computed at {cache['labelled_directly']} k and copied along "
                f"{cache['runs']} straight runs ({cache['mismatches']} mismatches); groups with",
            )
        )
        lines.append(
            _kv(
                "",
                f"P_g >= {run.weight_min:g}: {single} one irrep, {sets} sets (A/B), {mixed} mixed",
            )
        )
    t = run.timings
    parts = [
        ("mapping", "mapping"),
        ("phonopy", "phonopy"),
        ("unfolding", "unfolding"),
        ("arms", "star arms and domain average"),
        ("irreps", "irreps"),
        ("gamma", "child GM origin"),
        ("figures", "figures"),
        ("total", "total"),
    ]
    shown = [
        f"{name} {t[key]:.2f} s" if t[key] < 0.1 else f"{name} {t[key]:.1f} s"
        for key, name in parts
        if key in t
    ]
    lines.append(_kv("timings (CPU)", ", ".join(shown[:5]) + ("," if len(shown) > 5 else "")))
    if len(shown) > 5:
        lines.append(_kv("", ", ".join(shown[5:])))
    print("\n".join(lines))


def _print_files(files: dict, pages: int) -> None:
    lines = [
        "",
        "* Output files *",
        f"  Unfolded bands ({pages} page{'s' if pages != 1 else ''}) written to: {files['pdf']}",
        f"  Weights and labels written to: {files['dat']}",
    ]
    if files.get("npz"):
        lines.append(f"  All arrays written to: {files['npz']}")
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# command line (the dispatch target of crystod-phonon --unfold)
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """The parser of the dispatched arguments (``crystod-phonon --unfold`` builds them)."""
    parser = argparse.ArgumentParser(
        prog="crystod-phonon --unfold",
        description="Unfold the phonon bands of a distorted child structure onto the Brillouin "
        "zone of its parent.",
    )
    parser.add_argument("--child-poscar", dest="child_poscar", required=True, metavar="FILE")
    parser.add_argument("--child-dim", dest="child_dim", required=True, metavar="'N N N'")
    parser.add_argument("--child-readfc", dest="child_readfc", action="store_true")
    parser.add_argument("--parent-poscar", dest="parent_poscar", required=True, metavar="FILE")
    parser.add_argument("--parent-dim", dest="parent_dim", default=None, metavar="'N N N'")
    parser.add_argument("--parent-readfc", dest="parent_readfc", action="store_true")
    parser.add_argument("--trans-mat", dest="trans_mat", default=None, metavar="'T11 ... T33'")
    parser.add_argument("--band", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--npoints", type=int, default=NPOINTS)
    parser.add_argument("--no-irreps", dest="no_irreps", action="store_true")
    parser.add_argument("--weight-min", dest="weight_min", type=float, default=WEIGHT_MIN)
    parser.add_argument("--output", default="phonon_unfold")
    parser.add_argument("--export-npz", dest="export_npz", default=None)
    parser.add_argument("--tolerance", type=float, default=MAPPING_TOLERANCE)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse the dispatched arguments and run (see :func:`run`)."""
    args = build_parser().parse_args(argv)
    args.command_argv = list(argv) if argv is not None else None
    run(args)


def _output_prefix(output: str | None) -> str:
    """``-o PREFIX`` without a trailing ``.pdf``/``.dat``; a directory (``sub/``, ``.``) or an
    empty name gets the default name inside it (``sub/phonon_unfold``), never a hidden file."""
    prefix = output or "phonon_unfold"
    for suffix in (".pdf", ".dat"):
        if prefix.endswith(suffix):
            prefix = prefix[: -len(suffix)]
    if os.path.basename(prefix) in ("", ".", ".."):
        prefix = os.path.join(prefix, "phonon_unfold")
    return prefix


def _npz_name(path: str) -> str:
    """The file ``--export-npz FILE`` writes: FILE, with ``.npz`` appended when missing; a
    directory (``sub/``) or an empty name gets ``phonon_unfold.npz`` inside it."""
    stem = path[: -len(".npz")] if path.endswith(".npz") else path
    if os.path.basename(stem) in ("", ".", ".."):
        stem = os.path.join(stem, "phonon_unfold")
    return f"{stem}.npz"


def _command_text(argv) -> str:
    import shlex

    return "crystod-phonon --unfold" + (
        "" if not argv else " " + " ".join(shlex.quote(str(a)) for a in argv)
    )


def run(args: argparse.Namespace) -> UnfoldRun:
    """The whole ``--unfold`` pipeline; prints the blocks and writes the files.

    Steps: read the structures and force data (:func:`load_unfold_inputs`),
    the setting of crystod-group (:func:`find_parent_mapping`), the parent path
    (:func:`make_parent_path`), the unfolding (:func:`unfold`), the star arms
    and the domain average, the parent irreps (:func:`irrep_resolve`, unless
    ``no_irreps``), the child Gamma origin (:func:`gamma_origin`), then the
    ``.dat``, the PDF and, if asked, the ``.npz``.
    """
    t_start = time.perf_counter()
    timings = {}
    weight_min = float(args.weight_min if args.weight_min is not None else WEIGHT_MIN)
    if args.band:  # a bad path stops the run before anything is computed
        _manual_segments(args.band, args.label)
    npz = _npz_name(args.export_npz) if args.export_npz else None
    for target, option in ((_output_prefix(args.output), "-o"), (npz, "--export-npz")):
        folder = os.path.dirname(target) if target else ""
        if folder and not os.path.isdir(folder):
            raise SystemExit(f"ERROR: the directory '{folder}' of {option} does not exist.")
    t0 = time.perf_counter()
    inputs = load_unfold_inputs(
        args.child_poscar,
        args.child_dim,
        args.parent_poscar,
        args.parent_dim,
        child_readfc=args.child_readfc,
        parent_readfc=args.parent_readfc,
    )
    timings["phonopy"] = time.perf_counter() - t0
    trans_mat = None
    if args.trans_mat:
        trans_mat = [int(Fraction(token)) for token in str(args.trans_mat).split()]
    t0 = time.perf_counter()
    tolerance = MAPPING_TOLERANCE if args.tolerance is None else float(args.tolerance)
    mapping = find_parent_mapping(
        args.child_poscar,
        args.parent_poscar,
        child_cell=inputs.child_cell,
        parent_cell=inputs.parent_cell,
        trans_mat=trans_mat,
        tolerance=tolerance,
    )
    timings["mapping"] = time.perf_counter() - t0
    _print_setup(inputs, mapping)
    path = make_parent_path(
        mapping,
        band=args.band,
        band_labels=args.label,
        npoints=NPOINTS if args.npoints is None else args.npoints,
    )
    _print_path(path)
    t0 = time.perf_counter()
    bands = unfold(inputs.child_phonon, mapping, path.kpoints)
    parent_frequencies = None
    if inputs.parent_phonon is not None:
        parent_frequencies = np.concatenate(
            [
                np.array(
                    _run_qpoints(inputs.parent_phonon, path.kpoints[s : s + CHUNK]).frequencies,
                    dtype=float,
                )
                for s in range(0, len(path.kpoints), CHUNK)
            ]
        )
    timings["unfolding"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    parent_rotations = point_rotations(mapping.parent_cell)
    child_rotations = child_rotations_in_parent(point_rotations(mapping.child_cell), mapping.M)
    arms = vertex_star_arms(
        inputs.child_phonon,
        mapping,
        path,
        parent_rotations=parent_rotations,
        child_rotations_parent=child_rotations,
    )
    davg = domain_average(
        inputs.child_phonon,
        mapping,
        path.kpoints,
        parent_rotations=parent_rotations,
        child_rotations_parent=child_rotations,
    )
    timings["arms"] = time.perf_counter() - t0
    _print_arms(mapping, arms, davg, len(child_rotations), len(parent_rotations))
    irreps = None
    if not args.no_irreps:
        from .phonon_unfold_irreps import irrep_resolve

        t0 = time.perf_counter()
        irreps = irrep_resolve(
            bands, mapping, path=path, parent_phonon=inputs.parent_phonon, weight_min=weight_min
        )
        timings["irreps"] = time.perf_counter() - t0
        _print_irreps(path, bands, irreps, mapping)
    t0 = time.perf_counter()
    gamma = gamma_origin(
        inputs.child_phonon,
        mapping,
        irreps=irreps,
        parent_phonon=inputs.parent_phonon,
        path=path,
        weight_min=weight_min,
    )
    timings["gamma"] = time.perf_counter() - t0
    _print_gamma(gamma, mapping, irreps, inputs.parent_phonon is not None, weight_min)
    prefix = _output_prefix(args.output)
    files = {"pdf": f"{prefix}.pdf", "dat": f"{prefix}.dat"}
    result = UnfoldRun(
        inputs=inputs,
        mapping=mapping,
        path=path,
        bands=bands,
        arms=arms,
        domain_average=davg,
        irreps=irreps,
        gamma=gamma,
        parent_frequencies=parent_frequencies,
        weight_min=weight_min,
        files=files,
        timings=timings,
        command=_command_text(getattr(args, "command_argv", None)),
    )
    t0 = time.perf_counter()
    try:
        write_dat(
            files["dat"],
            mapping,
            path,
            bands,
            davg,
            irreps,
            child_text=_display_path(inputs.child_poscar),
            parent_text=_display_path(inputs.parent_poscar),
            weight_min=weight_min,
        )
        pages = plot(files["pdf"], result, weight_min=weight_min)
        timings["figures"] = time.perf_counter() - t0
        result.checks = _summary_checks(result)
        timings["total"] = time.perf_counter() - t_start
        if args.export_npz:
            files["npz"] = write_npz(args.export_npz, result)
            timings["total"] = time.perf_counter() - t_start
    except OSError as exc:
        raise SystemExit(f"ERROR: could not write the output files: {exc}") from None
    _print_summary(result, pages)
    _print_files(files, pages)
    return result
