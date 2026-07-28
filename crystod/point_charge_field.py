"""Point-charge (Madelung) ligand field and core-shell parameters for the
crystal-orbital diagrams (crystod --diagram).

The fragment sublattices of the COD must feel the electrostatic field of
the removed sublattice: the removed atoms are represented by point charges
at their crystal sites with their formal oxidation states (e.g. the Sc
fragment of ScF3 in the field of a F lattice with Q = -1, the F3 fragment
in the field of a Sc lattice with Q = +3).  This module provides

- Slater-rule exponents and neutral-atom configurations for the core
  shells (the valence shells keep the extended-Hueckel zetas),
- exact same-site matrix elements <phi_i| sum_Q q/|r-R| |phi_j> of the
  point-charge potential over the STO basis (Laplace expansion into real
  spherical harmonics -- the same real-orbital convention as
  ``wigner_D_real`` -- with closed-form radial integrals), so the fragment
  levels carry the true electrostatic ligand-field splittings (t2g/eg,
  ...),
- an Ewald summation of the (generally non-neutral) point-charge lattice
  potential with the standard neutralizing background, used for the
  long-range tail beyond the explicitly integrated near shells.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfc, expn, gammainc, gammaincc, gamma as gamma_fn

try:  # scipy >= 1.15
    from scipy.special import sph_harm_y as _sph_harm_y

    def _sph_harm(m, l, phi, theta):
        return _sph_harm_y(l, m, theta, phi)
except ImportError:  # scipy < 1.17 keeps the classic name
    from scipy.special import sph_harm as _scipy_sph_harm

    def _sph_harm(m, l, phi, theta):
        return _scipy_sph_harm(m, l, phi, theta)

from .operations import complex_to_real_transform_orbital

ANGSTROM_TO_BOHR = 1.8897259886
HARTREE_TO_EV = 27.211386
COULOMB_EV_ANGSTROM = 14.3996

# --------------------------------------------------------- Slater exponents

_SLATER_NSTAR = {1: 1.0, 2: 2.0, 3: 3.0, 4: 3.7, 5: 4.0, 6: 4.2}
_CAPACITY = {"s": 2, "p": 6, "d": 10, "f": 14}


def atomic_configuration(element: str) -> dict[str, int]:
    """Neutral-atom occupation of every explicitly treated shell."""
    from .mo_diagram import CORE_SHELLS, EHT_PARAMETERS, shell_occupation

    occupation = {shell: _CAPACITY[shell[-1]] for shell in CORE_SHELLS[element]}
    for shell, _n, _l, _zeta, _h in EHT_PARAMETERS[element]:
        occupation[shell] = shell_occupation(element, shell)
    return occupation


def slater_zeta(element: str, shell: str) -> float:
    """Slater-rule STO exponent of one shell of a neutral atom."""
    from pymatgen.core.periodic_table import Element

    Z = Element(element).Z
    occupation = atomic_configuration(element)
    n, letter = int(shell[0]), shell[-1]

    def group(shell_name):
        gn, gl = int(shell_name[0]), shell_name[-1]
        return (gn, "sp" if gl in "sp" else gl)

    # Slater group ordering: (1s)(2sp)(3sp)(3d)(4sp)(4d)(4f)(5sp)(5d)...
    def order(g):
        gn, gl = g
        return gn + {"sp": 0.0, "d": 0.5, "f": 0.7}[gl]

    own = group(shell)
    same_group = sum(
        count for other, count in occupation.items() if group(other) == own
    ) - 1
    screen = (0.30 if n == 1 else 0.35) * max(same_group, 0)
    for other, count in occupation.items():
        g = group(other)
        if g == own:
            continue
        if letter in "sp":
            gn = g[0]
            if gn == n - 1:
                screen += 0.85 * count
            elif gn <= n - 2:
                screen += 1.00 * count
        else:  # d / f shells: every inner electron screens fully
            if order(g) < order(own):
                screen += 1.00 * count
    return max((Z - screen) / _SLATER_NSTAR[n], 0.4)


# ------------------------------------------- real spherical harmonics (grid)

# The angular basis is built from complex_to_real_transform_orbital, so the
# component order matches wigner_D_real exactly (l=1: px py pz; l=2: dxy dyz
# dz2 dxz dx2-y2; ...) and the point-charge blocks commute with the
# site-symmetry representations by construction.
_N_THETA, _N_PHI = 16, 40


def _angular_grid():
    x, w = np.polynomial.legendre.leggauss(_N_THETA)
    theta = np.arccos(x)
    phi = 2.0 * np.pi * np.arange(_N_PHI) / _N_PHI
    T, P = np.meshgrid(theta, phi, indexing="ij")
    W = np.repeat(w[:, None], _N_PHI, axis=1) * (2.0 * np.pi / _N_PHI)
    return T.ravel(), P.ravel(), W.ravel()


_GRID_THETA, _GRID_PHI, _GRID_W = _angular_grid()
_Z_GRID_CACHE: dict[int, np.ndarray] = {}
_GAUNT_CACHE: dict = {}


def _real_harmonics_at(l: int, theta, phi) -> np.ndarray:
    """Real harmonics Z_i(theta, phi), i in the wigner_D_real order."""
    transform = complex_to_real_transform_orbital(l)
    ys = np.array([
        _sph_harm(m, l, phi, theta) for m in range(-l, l + 1)
    ])
    values = transform @ ys.reshape(2 * l + 1, -1)
    return np.real(np.real_if_close(values, tol=1e6)).reshape(
        (2 * l + 1,) + np.shape(theta)
    )


def _z_grid(l: int) -> np.ndarray:
    if l not in _Z_GRID_CACHE:
        _Z_GRID_CACHE[l] = _real_harmonics_at(l, _GRID_THETA, _GRID_PHI)
    return _Z_GRID_CACHE[l]


def real_gaunt(la: int, ia: int, lb: int, ib: int, k: int, ik: int) -> float:
    """Integral of Z^(la)_ia Z^(k)_ik Z^(lb)_ib over the sphere (exact for
    band-limited integrands on the Gauss-Legendre x uniform-phi grid)."""
    key = (la, ia, lb, ib, k, ik)
    if key not in _GAUNT_CACHE:
        value = float(np.sum(
            _z_grid(la)[ia] * _z_grid(k)[ik] * _z_grid(lb)[ib] * _GRID_W
        ))
        _GAUNT_CACHE[key] = 0.0 if abs(value) < 1e-12 else value
    return _GAUNT_CACHE[key]


def real_harmonics_direction(l: int, unit_vector) -> np.ndarray:
    """Z_i(l) evaluated along one direction (wigner_D_real order)."""
    x, y, z = unit_vector
    theta = float(np.arccos(np.clip(z, -1.0, 1.0)))
    phi = float(np.arctan2(y, x))
    return _real_harmonics_at(l, np.array([theta]), np.array([phi]))[:, 0]


# ---------------------------------------------------------- radial integrals

def _primitives(n: int, zeta) -> list[tuple[float, int, float]]:
    """(coefficient x norm, n, zeta) list of a possibly contracted STO."""
    from .mo_diagram import _slater_norm

    if isinstance(zeta, (list, tuple)):
        return [(c * _slater_norm(n, z), n, z) for z, c in zeta]
    return [(_slater_norm(n, zeta), n, zeta)]


_RADIAL_CACHE: dict = {}


def radial_overlap(shell_a, shell_b) -> float:
    """integral R_a(r) R_b(r) r^2 dr of two same-l STO radials."""
    (na, za), (nb, zb) = shell_a, shell_b
    total = 0.0
    for ca, pa, sa in _primitives(na, za):
        for cb, pb, sb in _primitives(nb, zb):
            p = pa + pb
            total += ca * cb * gamma_fn(p + 1) / (sa + sb) ** (p + 1)
    return total


def _radial_laplace_primitive(p: int, s: float, k: int, R: float) -> float:
    """integral_0^inf r^p e^{-s r} (r_<^k / r_>^{k+1}) dr with R the charge
    distance (r_< = min(r, R), r_> = max(r, R))."""
    # inner part: R^{-(k+1)} integral_0^R r^{p+k} e^{-sr} dr
    a = p + k + 1
    inner = (gamma_fn(a) / s**a) * float(gammainc(a, s * R)) / R**(k + 1)
    # outer part: R^k integral_R^inf r^{p-k-1} e^{-sr} dr
    q = p - k - 1
    if q >= 0:
        outer = R**k * (gamma_fn(q + 1) / s**(q + 1)) * float(
            gammaincc(q + 1, s * R)
        )
    else:
        m = -q
        outer = R**k * R**(1 - m) * float(expn(m, s * R))
    return inner + outer


def radial_laplace(shell_a, shell_b, k: int, R_bohr: float) -> float:
    """integral R_a(r) R_b(r) (r_<^k/r_>^{k+1}) r^2 dr for two (n, zeta)
    STOs (zeta may be a contracted [(zeta, coeff), ...] list)."""
    (na, za), (nb, zb) = shell_a, shell_b
    key = (na, str(za), nb, str(zb), k, round(R_bohr, 9))
    if key not in _RADIAL_CACHE:
        total = 0.0
        for ca, pa, sa in _primitives(na, za):
            for cb, pb, sb in _primitives(nb, zb):
                total += ca * cb * _radial_laplace_primitive(
                    pa + pb, sa + sb, k, R_bohr
                )
        _RADIAL_CACHE[key] = total
    return _RADIAL_CACHE[key]


# ----------------------------------------------------------- site V blocks

def point_charge_block(orbitals, charges_bohr) -> np.ndarray:
    """Same-site matrix of the electron energy -sum_Q q <i|1/|r-R_Q||j> (eV).

    orbitals: AtomicOrbital list of ONE atom (any shells, wigner_D_real
    component order within each shell); charges_bohr: (q, vector) list with
    vectors from the atom to each point charge in bohr."""
    size = len(orbitals)
    V = np.zeros((size, size))
    directions = []
    for q, vector in charges_bohr:
        R = float(np.linalg.norm(vector))
        directions.append((q, R, np.asarray(vector, dtype=float) / R))
    zk_cache: dict = {}
    for i in range(size):
        for j in range(i, size):
            a, b = orbitals[i], orbitals[j]
            total = 0.0
            for c_index, (q, R, unit) in enumerate(directions):
                for k in range(abs(a.l - b.l), a.l + b.l + 1, 2):
                    if (c_index, k) not in zk_cache:
                        zk_cache[c_index, k] = real_harmonics_direction(k, unit)
                    zk = zk_cache[c_index, k]
                    angular = 0.0
                    for ik in range(2 * k + 1):
                        g = real_gaunt(a.l, a.m, b.l, b.m, k, ik)
                        if g != 0.0:
                            angular += zk[ik] * g
                    if angular == 0.0:
                        continue
                    radial = radial_laplace(
                        (a.n, a.zeta), (b.n, b.zeta), k, R
                    )
                    total += q * (4.0 * np.pi / (2 * k + 1)) * angular * radial
            V[i, j] = V[j, i] = -HARTREE_TO_EV * total
    return V


# ------------------------------------------------------------------- Ewald

def ewald_site_potential(lattice_angstrom, charges, site_frac,
                         eta: float | None = None) -> float:
    """Potential sum_Q q/r (units e/Angstrom) of a point-charge lattice at a
    site, regularized with the standard neutralizing background for
    non-neutral charge arrays.  Multiply by 14.3996 for volts; the electron
    energy shift is -14.3996 x this value in eV.

    charges: list of (q, fractional position); site_frac must not coincide
    with a charge site."""
    lattice = np.asarray(lattice_angstrom, dtype=float)
    volume = abs(np.linalg.det(lattice))
    reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T
    r_cut = 10.0
    if eta is None:
        eta = 3.2 / r_cut  # erfc(eta r_cut) ~ 1e-9
    g_cut = 2.0 * eta * np.sqrt(23.0)

    site = np.asarray(site_frac, dtype=float)
    # real-space part
    perpendicular = [
        volume / np.linalg.norm(np.cross(lattice[(i + 1) % 3],
                                         lattice[(i + 2) % 3]))
        for i in range(3)
    ]
    bounds = [int(np.ceil(r_cut / perpendicular[i])) + 1 for i in range(3)]
    total = 0.0
    for q, frac in charges:
        delta = site - np.asarray(frac, dtype=float)
        for n1 in range(-bounds[0], bounds[0] + 1):
            for n2 in range(-bounds[1], bounds[1] + 1):
                for n3 in range(-bounds[2], bounds[2] + 1):
                    vector = (delta + [n1, n2, n3]) @ lattice
                    r = float(np.linalg.norm(vector))
                    if r < 1e-8 or r > r_cut:
                        continue
                    total += q * float(erfc(eta * r)) / r
    # reciprocal part
    g_perpendicular = [
        np.linalg.norm(reciprocal[i]) for i in range(3)
    ]
    g_bounds = [int(np.ceil(g_cut / g_perpendicular[i])) + 1 for i in range(3)]
    for m1 in range(-g_bounds[0], g_bounds[0] + 1):
        for m2 in range(-g_bounds[1], g_bounds[1] + 1):
            for m3 in range(-g_bounds[2], g_bounds[2] + 1):
                if m1 == m2 == m3 == 0:
                    continue
                G = np.array([m1, m2, m3], dtype=float) @ reciprocal
                g2 = float(G @ G)
                if g2 > g_cut**2:
                    continue
                structure = sum(
                    q * np.cos(float(G @ ((site - np.asarray(frac)) @ lattice)))
                    for q, frac in charges
                )
                total += (4.0 * np.pi / volume) * np.exp(
                    -g2 / (4.0 * eta**2)
                ) / g2 * structure
    # neutralizing background (G = 0 term)
    total -= (np.pi / (eta**2 * volume)) * sum(q for q, _ in charges)
    return float(total)
