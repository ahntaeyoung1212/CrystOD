"""Shared symmetry-operation helpers and Wigner-D utilities."""

from __future__ import annotations

from math import acos, atan2

import numpy as np
from numpy.typing import NDArray
from sympy import N
from sympy.physics.wigner import wigner_d_small


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


def rotation_matrix_to_euler_zyz(rotation: NDArray[np.float64]) -> tuple[float, float, float]:
    """Convert an SO(3) rotation matrix to ZYZ Euler angles."""
    beta = acos(np.clip(rotation[2, 2], -1.0, 1.0))
    if np.isclose(beta, 0.0):
        alpha = 0.0
        gamma = atan2(rotation[1, 0], rotation[0, 0])
    elif np.isclose(beta, np.pi):
        alpha = 0.0
        gamma = -atan2(rotation[1, 0], rotation[0, 0])
    else:
        alpha = atan2(rotation[1, 2], rotation[0, 2])
        gamma = atan2(rotation[2, 1], -rotation[2, 0])
    return alpha, beta, gamma


def complex_to_real_transform(l: int) -> NDArray[np.complex128]:
    """Return the complex-to-real spherical-harmonics transformation matrix."""
    size = 2 * l + 1
    transform = np.zeros((size, size), dtype=complex)
    m_values = np.arange(-l, l + 1)
    for row, m in enumerate(m_values):
        if m < 0:
            transform[row, l + m] = 1j / np.sqrt(2)
            transform[row, l - m] = -((-1) ** abs(m)) * 1j / np.sqrt(2)
        elif m == 0:
            transform[row, l] = 1
        else:
            transform[row, l - m] = 1 / np.sqrt(2)
            transform[row, l + m] = ((-1) ** abs(m)) / np.sqrt(2)
    return transform


def complex_to_real_transform_orbital(l: int) -> NDArray[np.complex128]:
    """Return the complex-to-real transformation matrix in common orbital order."""
    size = 2 * l + 1
    transform = np.zeros((size, size), dtype=complex)

    if l == 1:
        transform[0, 0] = 1 / np.sqrt(2)
        transform[0, 2] = -1 / np.sqrt(2)
        transform[1, 0] = -1j / np.sqrt(2)
        transform[1, 2] = -1j / np.sqrt(2)
        transform[2, 1] = 1
        return transform

    if l == 2:
        transform[0, 0] = 1j / np.sqrt(2)
        transform[0, 4] = -1j / np.sqrt(2)
        transform[1, 1] = 1j / np.sqrt(2)
        transform[1, 3] = 1j / np.sqrt(2)
        transform[2, 2] = 1
        transform[3, 1] = 1 / np.sqrt(2)
        transform[3, 3] = -1 / np.sqrt(2)
        transform[4, 0] = 1 / np.sqrt(2)
        transform[4, 4] = 1 / np.sqrt(2)
        return transform

    if l == 3:
        transform[0, 0] = 1 / np.sqrt(2)
        transform[0, 6] = -1 / np.sqrt(2)
        transform[1, 0] = 1j / np.sqrt(2)
        transform[1, 6] = 1j / np.sqrt(2)
        transform[2, 1] = 1 / np.sqrt(2)
        transform[2, 5] = 1 / np.sqrt(2)
        transform[3, 1] = 1j / np.sqrt(2)
        transform[3, 5] = -1j / np.sqrt(2)
        transform[4, 2] = 1 / np.sqrt(2)
        transform[4, 4] = -1 / np.sqrt(2)
        transform[5, 2] = 1j / np.sqrt(2)
        transform[5, 4] = 1j / np.sqrt(2)
        transform[6, 3] = 1
        return transform

    return complex_to_real_transform(l)


def wigner_D_matrix(l: int, rotation: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Return the complex Wigner D matrix for an SO(3) rotation."""
    alpha, beta, gamma = rotation_matrix_to_euler_zyz(rotation)
    m_values = np.arange(l, -l - 1, -1, dtype=float)
    if np.isclose(beta, 0.0):
        d_numeric = np.eye(2 * l + 1, dtype=complex)
    else:
        d_numeric = np.array(wigner_d_small(l, beta).applyfunc(N), dtype=complex)
    left = np.diag(np.exp(1j * m_values * alpha))
    right = np.diag(np.exp(1j * m_values * gamma))
    return left @ d_numeric @ right


def wigner_D_real(l: int, rotation: NDArray[np.float64]) -> NDArray[np.float64 | np.complex128]:
    """Return the real-basis Wigner D matrix in the orbital ordering used by matsym."""
    det = float(np.linalg.det(rotation))
    so3_rotation = det * rotation
    d_complex = wigner_D_matrix(l, so3_rotation)
    transform = complex_to_real_transform_orbital(l)
    d_real = (transform @ d_complex @ np.linalg.inv(transform)) * det
    return np.real_if_close(d_real)
