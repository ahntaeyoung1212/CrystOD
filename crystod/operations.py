"""Shared symmetry-operation helpers and Wigner-D utilities."""

from __future__ import annotations

from math import acos, atan2, factorial, sqrt

import numpy as np
from numpy.typing import NDArray


def characterize_rotation(rotation: NDArray[np.int_]) -> tuple[bool, int]:
    """Return whether a rotation is proper and its crystallographic order."""
    r_det = int(round(float(np.linalg.det(rotation))))
    if abs(r_det) != 1:
        raise ValueError("|det(R)| != 1. This matrix is not a crystallographic rotation.")

    is_proper = r_det == 1
    no_inv_r = rotation if is_proper else -rotation

    identity = np.eye(3, 3, dtype=int)
    rot_order = 0
    test_r = no_inv_r
    for _ in range(7):
        rot_order += 1
        if np.array_equal(test_r, identity):
            break
        test_r = no_inv_r @ test_r

    if rot_order not in [1, 2, 3, 4, 6]:
        raise ValueError("This rotation matrix is not a symmetry operation of a space group.")
    return is_proper, rot_order


def get_seitz_symbol(rotation: NDArray[np.int_], trans_mat: NDArray[np.float64]) -> str:
    """Label the rotation part of a symmetry operation using Seitz-like notation."""
    r_trans = trans_mat @ rotation @ np.linalg.inv(trans_mat)
    r = np.rint(r_trans).astype(np.int_)
    is_proper, rot_order = characterize_rotation(r)
    if rot_order == 1:
        return "1" if is_proper else "-1"

    no_inv_r = r if is_proper else -r

    eig_vals, eig_vecs = np.linalg.eig(no_inv_r)
    eig_vecs = eig_vecs.astype(np.complex128)
    direction = None
    for idx, value in enumerate(eig_vals):
        if abs(value - 1.0) < 1e-8:
            vector = eig_vecs[:, idx]
            uniques = np.unique(vector).real
            scale = uniques[0]
            if abs(scale) < 1e-8:
                if len(uniques) < 2:
                    scale = 1.0
                else:
                    scale = uniques[1]
            direction = np.rint(vector.real / scale).astype(np.int_)
            break

    if direction is None:
        raise ValueError("Failed to determine the characteristic direction of the rotation.")

    if rot_order == 2:
        prefix = "2_" if is_proper else "m_"
        return prefix + "".join(str(component) for component in direction)

    rot_axis = np.array(
        [
            no_inv_r[2, 1] - no_inv_r[1, 2],
            no_inv_r[0, 2] - no_inv_r[2, 0],
            no_inv_r[1, 0] - no_inv_r[0, 1],
        ]
    )
    rot_sign = "+" if np.dot(rot_axis, direction) > 0 else "-"
    prefix = f"{rot_order}^{rot_sign}_" if is_proper else f"-{rot_order}^{rot_sign}_"
    return prefix + "".join(str(component) for component in direction)


def rotation_matrix_to_euler_zyz(
    rotation: NDArray[np.float64], tol: float = 1e-7
) -> tuple[float, float, float]:
    """Convert an SO(3) rotation matrix to ZYZ Euler angles (R = Rz(a) Ry(b) Rz(g))."""
    beta = acos(np.clip(rotation[2, 2], -1.0, 1.0))
    if beta < tol:
        return 0.0, 0.0, atan2(rotation[1, 0], rotation[0, 0])
    if abs(beta - np.pi) < tol:
        return 0.0, np.pi, atan2(rotation[1, 0], -rotation[0, 0])
    alpha = atan2(rotation[1, 2], rotation[0, 2])
    gamma = atan2(rotation[2, 1], -rotation[2, 0])
    return alpha, beta, gamma


def wigner_d_small_matrix(l: int, beta: float) -> NDArray[np.float64]:
    """Return the small Wigner d matrix d^l(beta) with rows/columns ordered m = -l..l.

    Uses the explicit Wigner sum formula; d[i, j] = d^l_{m' m}(beta) with
    m' = -l + i and m = -l + j.
    """
    size = 2 * l + 1
    d_matrix = np.zeros((size, size))
    cos_half = np.cos(beta / 2.0)
    sin_half = np.sin(beta / 2.0)
    for i, m_prime in enumerate(range(-l, l + 1)):
        for j, m in enumerate(range(-l, l + 1)):
            k_min = max(0, m - m_prime)
            k_max = min(l + m, l - m_prime)
            value = 0.0
            for k in range(k_min, k_max + 1):
                numerator = sqrt(
                    factorial(l + m)
                    * factorial(l - m)
                    * factorial(l + m_prime)
                    * factorial(l - m_prime)
                )
                denominator = (
                    factorial(l + m - k)
                    * factorial(k)
                    * factorial(m_prime - m + k)
                    * factorial(l - m_prime - k)
                )
                value += (
                    (-1.0) ** (m_prime - m + k)
                    * (numerator / denominator)
                    * cos_half ** (2 * l + m - m_prime - 2 * k)
                    * sin_half ** (m_prime - m + 2 * k)
                )
            d_matrix[i, j] = value
    return d_matrix


def complex_to_real_transform(l: int) -> NDArray[np.complex128]:
    """Return the complex-to-real spherical-harmonics transformation matrix.

    Rows are ordered m = -l..l (sin-type for m < 0, Y_l0, cos-type for m > 0);
    columns are complex Y_lm ordered m = -l..l with the Condon-Shortley phase:
    real_row = transform @ [Y_{-l}, ..., Y_{l}].
    """
    size = 2 * l + 1
    transform = np.zeros((size, size), dtype=complex)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    for row, m in enumerate(range(-l, l + 1)):
        if m < 0:
            # sin-type: i (Y_{-|m|} - (-1)^{|m|} Y_{|m|}) / sqrt(2)
            transform[row, l + m] = 1j * inv_sqrt2
            transform[row, l - m] = -((-1.0) ** abs(m)) * 1j * inv_sqrt2
        elif m == 0:
            transform[row, l] = 1.0
        else:
            # cos-type: (Y_{-m} + (-1)^m Y_{m}) / sqrt(2)
            transform[row, l - m] = inv_sqrt2
            transform[row, l + m] = ((-1.0) ** m) * inv_sqrt2
    return transform


def complex_to_real_transform_orbital(l: int) -> NDArray[np.complex128]:
    """Return the complex-to-real transformation matrix in common orbital order.

    Orbital orderings (columns are Y_lm with m = -l..l):
      l=1: p_x, p_y, p_z
      l=2: d_xy, d_yz, d_z2, d_xz, d_x2-y2
      l=3: f_x(x2-3y2), f_y(3x2-y2), f_z(x2-y2), f_xyz, f_xz2, f_yz2, f_z3
    """
    generic = complex_to_real_transform(l)

    def sin_row(m: int) -> NDArray[np.complex128]:
        return generic[l - abs(m)]

    def cos_row(m: int) -> NDArray[np.complex128]:
        return generic[l + abs(m)]

    if l == 1:
        return np.stack([cos_row(1), sin_row(1), generic[l]])
    if l == 2:
        return np.stack([sin_row(2), sin_row(1), generic[l], cos_row(1), cos_row(2)])
    if l == 3:
        return np.stack(
            [cos_row(3), sin_row(3), cos_row(2), sin_row(2), cos_row(1), sin_row(1), generic[l]]
        )
    return generic


def wigner_D_matrix(l: int, rotation: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Return the complex Wigner D matrix for an SO(3) rotation.

    Rows/columns are ordered m = -l..l. The convention is
    Y_lm(R^{-1} r) = sum_{m'} D^l_{m' m}(R) Y_lm'(r).
    """
    alpha, beta, gamma = rotation_matrix_to_euler_zyz(np.asarray(rotation, dtype=float))
    d_matrix = wigner_d_small_matrix(l, beta)
    m_values = np.arange(-l, l + 1, dtype=float)
    left = np.diag(np.exp(-1j * m_values * alpha))
    right = np.diag(np.exp(-1j * m_values * gamma))
    return left @ d_matrix @ right


def wigner_D_real(l: int, rotation: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the real-orbital-basis representation matrix of an O(3) operation.

    The matrix M satisfies f_j(R^{-1} r) = sum_i M[i, j] f_i(r) where f_i are
    the real orbitals ordered as in ``complex_to_real_transform_orbital``.
    Improper operations pick up the parity factor (-1)^l.
    """
    rotation = np.asarray(rotation, dtype=float)
    det_sign = 1.0 if np.linalg.det(rotation) > 0 else -1.0
    d_complex = wigner_D_matrix(l, det_sign * rotation)
    transform = complex_to_real_transform_orbital(l)
    d_real = (transform @ d_complex.T @ np.linalg.inv(transform)).T * (det_sign**l)
    return np.real(np.real_if_close(d_real))


def find_star_arm(
    kpoint: list[float] | NDArray[np.float64],
    rotations: NDArray[np.int_],
    special_points: list[list[float]],
) -> tuple[int, list[float]] | None:
    """Map k onto the tabulated arm of its star.

    irreptables lists only one representative arm per special point (e.g. only
    (1/2, 1/2, 0) for the three M arms of Pm-3m), so a direct coordinate lookup
    fails for the other arms. Returns (g_index, k_rep) where rotation g sends k
    onto the tabulated point k_rep (k' = k R, modulo reciprocal-lattice
    translations); None when k is not in any tabulated star. ``rotations`` must
    be in the same (primitive) basis as k and the tabulated points.
    """
    kpoint = np.asarray(kpoint, dtype=float)
    for k_rep in special_points:
        target = np.asarray(k_rep, dtype=float)
        for g_index, rotation in enumerate(rotations):
            diff = kpoint @ rotation - target
            if (np.abs(diff - np.rint(diff)) < 1e-6).all():
                return g_index, list(k_rep)
    return None


def conjugated_little_group_map(
    rotations: NDArray[np.int_],
    translations: NDArray[np.float64],
    g_index: int,
    k_rep: list[float] | NDArray[np.float64],
    little_indices: NDArray[np.int_] | list[int],
) -> tuple[list[int], NDArray[np.complex128]] | None:
    """Transport little-group operations of k onto those of k_rep = k g.

    Conjugation h -> g^-1 h g is an isomorphism of the little group of k onto
    that of k_rep, so the small-irrep characters at k are those at k_rep
    evaluated at the conjugated operations: chi_k(h) = chi_rep(g^-1 h g). The
    conjugated operation matches a listed operation up to a lattice translation
    Delta, which contributes the Bloch phase e^{-2 pi i k_rep . Delta}.

    ``little_indices`` selects the little-group operations of k within
    ``rotations``/``translations``. Returns (indices, phases) with one entry
    per selected operation: the index of g^-1 h g within ``rotations`` and its
    phase factor; None when the conjugation cannot be resolved.
    """
    rotation_g = rotations[g_index]
    translation_g = translations[g_index]
    rotation_g_inv = np.rint(np.linalg.inv(rotation_g)).astype(int)
    k_rep = np.asarray(k_rep, dtype=float)

    indices: list[int] = []
    phases: list[complex] = []
    for idx in little_indices:
        rotation_h = rotations[idx]
        translation_h = translations[idx]
        rotation_conj = rotation_g_inv @ rotation_h @ rotation_g
        translation_conj = rotation_g_inv @ (rotation_h @ translation_g + translation_h - translation_g)

        conj_index = None
        for j, rotation in enumerate(rotations):
            if (rotation == rotation_conj).all():
                conj_index = j
                break
        if conj_index is None:
            return None
        delta = translation_conj - translations[conj_index]
        delta_int = np.rint(delta)
        if not np.allclose(delta, delta_int, atol=1e-5):
            return None
        indices.append(conj_index)
        phases.append(np.exp(-2j * np.pi * float(k_rep @ delta_int)))
    return indices, np.array(phases, dtype=complex)
