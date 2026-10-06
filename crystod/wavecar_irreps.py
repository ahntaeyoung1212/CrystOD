"""Irreducible representations of plane-wave Bloch states (VASP WAVECAR).

The labelling engine of the WAVECAR-overlap route of ``crystod --diagram
--vasp``.  It assigns the irreps of the little group of k to the Bloch states
of a plane-wave run directly from their coefficients, so that neither IrRep
nor any other external program is needed.  Single-valued irreps carry
CrystOD's ISO-IR labels -- the names every other CrystOD command prints
(``GM5+``, ``X3-``, ``R4-``); the states of a spinor run (vasp_ncl, LSORBIT)
are decomposed into spgrep's double-valued irreps and named by the rule given
below.  The core works on arrays only, so any WAVECAR reader can feed it.

Conventions
-----------
Positions are fractional, ``x``, in the cell ``lattice`` (rows a_i in
Angstrom); k and G are fractional in the reciprocal basis b_i
(a_i . b_j = 2 pi delta_ij), so (k + G) . r = 2 pi (k + G) . x.  A Bloch state
of the run is

    psi_nk(x) = sum_G C_n(G) exp(2 pi i (k + G) . x),

with two such components for a spinor (spin up and down along the Cartesian
z axis of ``lattice`` -- VASP's default SAXIS).  An operation g = {R|t} maps
x to R x + t and acts on states as (O_g psi)(x) = U_g psi(g^-1 x), with U_g
the SU(2) matrix of the proper part of R for spinors and U_g = 1 otherwise.
For g in the little group of k, R^-T k = k + G_g, and

    O_g psi(x) = sum_G C(G) exp(-2 pi i (k + G') . t) exp(2 pi i (k + G') . x),
    G' = R^-T G + G_g:

plane wave G' of O_g psi carries C(G) exp(-2 pi i (k + G') . t), with U_g
acting on the spinor index.  A pure translation acts as exp(-2 pi i k . t),
the convention of spgrep's small representations (the ISO-IR tables use the
opposite sign; ``crystod.isoir`` handles the conjugation).  The sphere
|k + G|^2 < ENCUT of the basis is invariant under the little group, so
G -> G' permutes the coefficient list; an image missing from the list (an
equal-length vector rounded the other way at the sphere surface) drops its
coefficient, and the largest share of a band's norm dropped that way is
reported as ``lost``.

Characters
----------
PAW pseudo wave functions psi~ = T^-1 psi transform exactly as the
all-electron ones (T commutes with the space group), but they are orthonormal
in the PAW metric only, not in the plain one.  On the span of the bands B of
one level the representation matrices are therefore D(g) = S^-1 M(g), with
S = <psi~_m|psi~_n> and M(g) = <psi~_m|O_g|psi~_n> from the pseudo
coefficients, and chi(g) = tr D(g).  This is exact without the augmentation
(IrRep divides by the pseudo norms instead, which agrees when the pseudo
overlaps of the partners vanish).

M(g) is formed from the coefficients for a generating set of the little
group only (two or three operations, taken greedily among those of highest
order); the other D(g) follow from products, D(g_a g_b) =
z exp(2 pi i k.n) D(g_a) D(g_b) for g_a g_b = {E|n} g_c and U_a U_b = z U_c,
which is exact on an invariant span.  Invariance is tested, not assumed:
``leakage`` is the largest share of the norm of a state of the level that a
generator carries out of its span (one minus the lowest eigenvalue of
M^H S^-1 M against S), zero for an invariant span and about 1/2 for one
band of a Kramers pair; invariance under generators is invariance under the
group.  A block of bands that leaks more than 1e-3 (the last, least
converged bands of a run; near-degenerate levels mixed by a slightly
asymmetric density) is judged on characters formed from every operation, so
that the outcome does not depend on the generators.  The multiplicities
m_a = |G_k|^-1 sum_g conj(chi_a(g)) chi(g) of spgrep's small irreps chi_a then
decide: a level is complete when its leakage is at most
:data:`LEAKAGE_TOL` and every m_a lies within :data:`MULTIPLICITY_TOL` of a
non-negative integer n_a, the dimensions adding up.  ``quality`` is the
largest deviation |chi(g) - sum_a n_a chi_a(g)| over the little group.  At
128 spinor bands and 2 x 7100 plane waves a k point takes 0.05-0.25 s, at 64
scalar bands and 1700 plane waves 0.01 s (reading excluded; the symmetry
set-up, 0.1-0.5 s, is cached per cell and k).

Levels
------
Bands are seeded into levels by ``degeneracy_tol`` (1 meV from the first band
of the level).  A level that is not complete is merged with the following
ones until it is, as long as their energies stay within 0.3 eV of its own
-- the rule of ``crystod --diagram --pyscf`` and of the PROCAR engine.  It
absorbs numerically split multiplets such as the Kramers pairs of a
point-charge sublattice run with spin-orbit coupling, which a not fully
symmetric density splits by up to a few meV.  A level that never becomes
complete (a multiplet cut by the top of the band list) keeps its seed and an
empty label; its characters are then formed from every operation, so that
its fractional multiplicities can be read.  A complete level that carries
more than one irrep is finally split into the shortest consecutive band
blocks that are complete on their own, so that two levels 0.2 meV apart are
named separately; levels of different irreps that are degenerate and mixed
stay one level named ``A/B``.

Labels
------
Single-valued: spgrep's small irreps of the little group of k in the cell as
given -- not standardized, so the labels refer to the run's own setting and
origin, as IrRep's do with refUC = 1, shiftUC = 0 -- named by
``crystod.isoir.get_isoir_label_map`` in the ISO-IR frame of that cell (a
cell already in the ISO-IR setting keeps its own origin).  A level is named
by its irrep, or by the '/'-joined names of an accidental degeneracy
(alphabetical, each repeated by its multiplicity, ``GM4-/GM5-``), the format
of the paper scripts that read IrRep.  The three runs of a crystal-orbital
diagram (crystal and the two Va point-charge sublattice runs) have one space
group and one setting; ``symmetry_cell`` takes the crystal's cell for the
sublattice runs, so that the labels of all three columns are taken in one
frame by construction.

Double-valued: spgrep's spinor small irreps
(``get_spacegroup_spinor_irreps_from_primitive_symmetry``), built with the
SU(2) matrices U_g that spgrep assigns (angle in [0, pi],
U_g = exp(-i theta/2 n . sigma), U(-R) = U(R), so inversion acts trivially on
spin); the same U_g act on the wave functions, so the two factor systems
agree.  The ISO-IR tables have no double-valued irreps, and CrystOD names
them ``-K<n><p>`` -- the overbar written as a leading '-' (IrRep's spelling
of \\bar{K}), then the CDML/Koster-style index and parity:

* K is the ISO-IR k label of the single-valued irreps at k (GM, X, R, DT ...);
* the fingerprint of a double-valued irrep D is the decomposition of
  D x D(1/2) (D(1/2): chi = tr U_g) into the ISO-IR-labelled single-valued
  irreps at k.  It identifies D, does not depend on the setting beyond the
  ISO-IR frame, and carries D's parity p: all of its labels end in '+' or all
  in '-' (D(1/2) is even); p is empty where the labels carry no parity;
* parity partners (D and D x det, whose fingerprints differ only in the
  suffixes) share the index n.  The partner classes are ordered by
  dimension, then by fingerprint in natural order, and numbered after the
  single-valued irreps: n = N + 1, N + 2, ..., N the largest index of the
  single-valued labels at k.

The fingerprint order puts the spin-1/2 representation itself (the one whose
fingerprint holds the identity irrep K1) first, Koster's convention: for O_h
the names are Gamma_6+-, Gamma_7+- (two-dimensional) and Gamma_8+- (four), so
the valence-band maximum of CsPbI3 is ``-R6+`` and its spin-orbit split-off
conduction-band minimum ``-R6-``, the names of the perovskite literature.
The names are not those of the Bilbao Crystallographic Server, which numbers
double-valued irreps on without a parity suffix and in its own order
(measured on CsPbI3 in Pm-3m: -GM6+ = BCS -GM6, -GM7+ = -GM7, -GM6- = -GM8,
-GM7- = -GM9, -GM8+ = -GM10, -GM8- = -GM11, the same pattern at R and X, but
-M6+ = -M7 and -M6- = -M9 at M).  :func:`spin_orbit_compatibility` gives, for every
single-valued irrep at k, the double-valued irreps of Gamma x D(1/2) in these
names (the scalar bridge of a spin-orbit diagram, any space group).
:func:`bcs_labels` puts IrRep's BCS names
on the levels by band index when an IrRep JSON of the same run is at hand,
and :func:`run_irrep` produces one by running the ``irrep`` program as a
subprocess (the JSON is only read; the GPL ``irrep`` package is never
imported).

Time reversal is not applied: levels are labelled by unitary small irreps,
and two complex-conjugate irreps held together by time reversal show up as
``A/B``.  Only primitive cells are accepted (a supercell WAVECAR folds
several k points of the primitive zone onto one); a collinear spin-polarized
run is two scalar problems, one per spin channel.

Example:
    >>> import numpy as np
    >>> from crystod.wavecar_irreps import irreps_at_kpoint
    >>> lattice = 4.0 * np.eye(3)
    >>> gvecs = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0],
    ...                   [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    >>> coeffs = np.zeros((4, 1, 7), dtype=complex)
    >>> coeffs[0, 0, 0] = 1.0                        # constant: GM1+
    >>> for n, (p, m) in enumerate([(1, 2), (3, 4), (5, 6)], start=1):
    ...     coeffs[n, 0, p], coeffs[n, 0, m] = 0.5j, -0.5j  # sin: GM4-
    >>> levels = irreps_at_kpoint(lattice, [[0, 0, 0]], [1], [0, 0, 0],
    ...                           gvecs, coeffs, [-1.0, 2.0, 2.0, 2.0])
    >>> [(level["label"], level["dim"]) for level in levels]
    [('GM1+', 1), ('GM4-', 3)]
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

__all__ = [
    "DEGENERACY_SEED_EV",
    "DEGENERACY_MAX_WINDOW_EV",
    "MULTIPLICITY_TOL",
    "LEAKAGE_TOL",
    "LittleGroup",
    "little_group_data",
    "spin_orbit_compatibility",
    "irreps_at_kpoint",
    "label_bands",
    "labels_from_levels",
    "read_irrep_json",
    "irrep_json_labels",
    "irrep_json_name_map",
    "bcs_labels",
    "irrep_executable",
    "run_irrep",
]

# Seed window (eV) of the degeneracy clustering: a band within this distance
# of the first band of a level starts out in that level.
DEGENERACY_SEED_EV = 1.0e-3

# A level whose multiplicities are not integral absorbs the following levels
# whose (mean) energies lie at most this far above its own -- the cap of the
# PySCF and PROCAR engines.
DEGENERACY_MAX_WINDOW_EV = 0.30

# Largest deviation of a multiplicity from its integer for a level to count
# as complete.  Converged bands are integral to 1e-4 or better; a part of a
# multiplet, or one band of a Kramers pair, sits near 1/2.
MULTIPLICITY_TOL = 0.10

# Largest share of the norm of a state of a level that a symmetry operation
# may carry out of the level's span (the invariance test, see _assess).
# Measured on the perovskite runs (2105 complete levels, 215 fragments, every
# operation): at most 0.016 for complete levels, the last bands of a run
# included; 0.09 for the very last band of one run and 0.10 for two Kramers
# pairs of opposite parity 1 meV apart that a slightly asymmetric density
# mixes by a few per cent (both labelled by their dominant irrep, as IrRep
# labels them); 0.24 for an unconverged pair near the top of a band list,
# split by 20 meV (left unassigned; IrRep cannot name it either); 0.40 or
# more for one band of a Kramers pair or a part of a multiplet.
# MULTIPLICITY_TOL is the second gate.
LEAKAGE_TOL = 0.15

# A block of bands whose leakage under the generators lies between this and
# LEAKAGE_TOL is judged on characters formed from every operation: the
# products of a generating set amplify the noise of an imperfect multiplet.
_DIRECT_LEAKAGE = 1.0e-3

# A k coordinate within this distance of a fraction with a denominator up to
# _K_DENOMINATOR is snapped onto it (the WAVECAR stores k in double precision
# but a reader may round it).
_K_SNAP = 1.0e-6
_K_DENOMINATOR = 48

_LABEL_RE = re.compile(r"^(?P<bar>-?)(?P<k>[A-Za-z_]*?)(?P<n>\d+)(?P<p>[+-]?)$")

_LITTLE_GROUP_CACHE: dict = {}
_SPACE_GROUP_CACHE: dict = {}


# ------------------------------------------------------------------ symmetry
@dataclass
class LittleGroup:
    """The little group of one k point, its small irreps and their names.

    Attributes:
        sgnum: Space-group number of the cell.
        kpoint: k in the reciprocal basis of the cell, shape ``(3,)``.
        k_label: ISO-IR k label of the irreps (``"GM"``, ``"X"``, ``"DT"``),
            or ``""`` when the ISO-IR labeller has no answer.
        rotations: Integer rotations of the little-group operations in the
            cell's fractional basis (real space), shape ``(n, 3, 3)``.
        translations: Their translations, shape ``(n, 3)``.
        spin_rotations: The SU(2) matrices U_g, shape ``(n, 2, 2)``, for
            double-valued irreps; ``None`` for single-valued ones.
        characters: Characters of the small irreps, shape ``(n_irreps, n)``,
            in spgrep's convention (pure translation: exp(-2 pi i k.t)).
        dims: Dimensions of the irreps.
        names: Names of the irreps (ISO-IR labels, or ``-K<n><p>`` such as
            ``-R6+`` for double-valued irreps; see the module docstring).
        fingerprints: For double-valued irreps, the ISO-IR labels of
            D x D(1/2) in natural order; ``None`` for single-valued ones.
        from_isoir: Whether the single-valued names came from the ISO-IR
            tables (``False``: generic ``irrep_N`` names).
    """

    sgnum: int
    kpoint: np.ndarray
    k_label: str
    rotations: np.ndarray
    translations: np.ndarray
    spin_rotations: np.ndarray | None
    characters: np.ndarray
    dims: np.ndarray
    names: list
    fingerprints: list | None = None
    from_isoir: bool = False
    plans: dict = field(default_factory=dict, repr=False)

    @property
    def spinor(self) -> bool:
        """Whether the irreps are double-valued."""
        return self.spin_rotations is not None

    @property
    def order(self) -> int:
        """Number of little-group operations (coset representatives)."""
        return len(self.rotations)


def _cell_key(lattice, positions, numbers) -> tuple:
    return (np.round(np.asarray(lattice, dtype=float), 8).tobytes(),
            np.round(np.asarray(positions, dtype=float) % 1.0, 8).tobytes(),
            np.asarray(numbers, dtype=np.int64).tobytes())


def _snap_kpoint(kpoint) -> np.ndarray:
    """k with coordinates near simple fractions snapped onto them."""
    k = np.array(kpoint, dtype=float).reshape(3)
    for i, value in enumerate(k):
        fraction = Fraction(float(value)).limit_denominator(_K_DENOMINATOR)
        if abs(float(fraction) - value) < _K_SNAP:
            k[i] = float(fraction)
    return k + 0.0


def _space_group(lattice, positions, numbers, symprec: float):
    """``(number, rotations, translations)`` of a primitive cell (cached).

    Raises:
        ValueError: spglib finds no symmetry, or the cell is not primitive.
    """
    key = (float(symprec),) + _cell_key(lattice, positions, numbers)
    if key not in _SPACE_GROUP_CACHE:
        import spglib

        cell = (np.asarray(lattice, dtype=float), np.asarray(positions, dtype=float),
                np.asarray(numbers, dtype=int))
        dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
        if dataset is None:
            raise ValueError("spglib found no symmetry for the cell")
        rotations = np.asarray(dataset.rotations, dtype=np.int64)
        translations = np.asarray(dataset.translations, dtype=float)
        identity = np.all(rotations == np.eye(3, dtype=np.int64), axis=(1, 2))
        shifts = translations[identity] - np.rint(translations[identity])
        if int(np.sum(identity)) > 1 or np.any(np.abs(shifts) > 1e-6):
            raise ValueError(
                "the cell is not primitive (it has pure-translation symmetries); "
                "irreps of a WAVECAR need the run of a primitive cell")
        if len(_SPACE_GROUP_CACHE) > 64:
            _SPACE_GROUP_CACHE.clear()
        _SPACE_GROUP_CACHE[key] = (int(dataset.number), rotations, translations)
    return _SPACE_GROUP_CACHE[key]


def _check_operations(rotations, translations, positions, numbers, symprec, lattice):
    """Raise unless every operation maps the cell onto itself species by species."""
    positions = np.asarray(positions, dtype=float)
    numbers = np.asarray(numbers)
    lattice = np.asarray(lattice, dtype=float)
    tolerance = max(10.0 * float(symprec), 1e-4)  # Angstrom
    for R, t in zip(rotations, translations):
        images = positions @ np.asarray(R).T + t
        for image, number in zip(images, numbers):
            same = positions[numbers == number]
            d = image - same
            d = (d - np.rint(d)) @ lattice
            if np.min(np.linalg.norm(d, axis=1)) > tolerance:
                raise ValueError(
                    "symmetry_cell has an operation that is not a symmetry of the "
                    "run's own cell (are the Va point charges on the removed sites?)")


def _scalar_irreps(rotations, translations, kpoint):
    """spgrep's single-valued small irreps (CrystOD's robust wrapper)."""
    from .runtime_compat import get_spacegroup_irreps_from_primitive_symmetry

    irreps, mapping = get_spacegroup_irreps_from_primitive_symmetry(
        rotations=rotations, translations=translations, kpoint=kpoint)
    return [np.asarray(irrep) for irrep in irreps], np.asarray(mapping, dtype=int)


def _spinor_irreps(lattice, rotations, translations, kpoint):
    """spgrep's double-valued small irreps ``(irreps, U_g, mapping)``.

    spgrep ends the enumeration with a Frobenius-Schur check over the coset
    representatives, which is not the indicator at 2k != 0 and can reject a
    valid irrep (see ``runtime_compat.get_spacegroup_irreps_from_primitive_
    symmetry``).  On that error the irreps are rebuilt from the same spgrep
    steps without the check and verified to be complete and inequivalent.
    """
    from spgrep.core import get_spacegroup_spinor_irreps_from_primitive_symmetry

    try:
        irreps, _factor, unitary, mapping = (
            get_spacegroup_spinor_irreps_from_primitive_symmetry(
                lattice=lattice, rotations=rotations, translations=translations,
                kpoint=kpoint))
        return ([np.asarray(irrep) for irrep in irreps], np.asarray(unitary),
                np.asarray(mapping, dtype=int))
    except ValueError as exc:
        if "not irreducible" not in str(exc) or np.allclose(
                2 * kpoint, np.rint(2 * kpoint), atol=1e-8):
            raise
        error = exc
    try:
        from spgrep.symmetry.enumerate import (
            enumerate_unitary_irreps_from_solvable_group_chain,
            purify_irrep_value,
        )
        from spgrep.symmetry.group import (
            get_cayley_table,
            get_factor_system_from_little_group,
            get_little_group,
        )
        from spgrep.symmetry.pointgroup import get_pointgroup_chain_generators
    except ImportError:  # older spgrep layout
        from spgrep.group import (
            get_cayley_table,
            get_factor_system_from_little_group,
            get_little_group,
        )
        from spgrep.irreps import (
            enumerate_unitary_irreps_from_solvable_group_chain,
            purify_irrep_value,
        )
        from spgrep.pointgroup import get_pointgroup_chain_generators
    from spgrep.spinor import get_spinor_factor_system

    little_rotations, little_translations, mapping = get_little_group(
        rotations, translations, kpoint)
    spin_factor, unitary = get_spinor_factor_system(lattice, little_rotations)
    factor = spin_factor * get_factor_system_from_little_group(
        little_rotations, little_translations, kpoint)
    cogroup = enumerate_unitary_irreps_from_solvable_group_chain(
        get_cayley_table(little_rotations), factor,
        get_pointgroup_chain_generators(little_rotations))
    phases = np.exp(-2j * np.pi * np.asarray(little_translations, dtype=float) @ kpoint)
    irreps = [np.asarray(purify_irrep_value(rep)) * phases[:, None, None] for rep in cogroup]
    characters = np.array([np.trace(rep, axis1=1, axis2=2) for rep in irreps])
    order = len(little_rotations)
    gram = np.conj(characters) @ characters.T
    if not (np.allclose(gram, order * np.eye(len(irreps)), atol=1e-6)
            and sum(rep.shape[1] ** 2 for rep in irreps) == order):
        raise error
    return irreps, np.asarray(unitary), np.asarray(mapping, dtype=int)


def _label_key(label: str) -> tuple:
    """Natural order of irrep labels: ``GM2+`` < ``GM10+``, '+' before '-'."""
    match = _LABEL_RE.match(label)
    if match is None:
        return (1, label, 0, "")
    return (0, match["k"], int(match["n"]), match["p"])


def _isoir_names(sgnum, cell, symprec, kpoint, rotations, translations, characters):
    """ISO-IR names and k label of single-valued irreps, or generic names."""
    from .isoir import get_isoir_label_map

    matched = get_isoir_label_map(sgnum, cell, symprec, kpoint, list(rotations),
                                  list(translations), list(characters))
    if matched is None:
        return [f"irrep_{i + 1}" for i in range(len(characters))], "", False
    label_map, k_label = matched
    return [label_map[i] for i in range(len(characters))], str(k_label), True


def _double_valued_names(k_label, single_names, single_characters, double_characters,
                         double_dims, spin_traces):
    """Names ``-K<n><parity>`` and fingerprints of the double-valued irreps.

    See the module docstring for the rule.  Parity partners (the same
    fingerprint up to the parity suffixes) share the index ``n``.
    """
    order = double_characters.shape[1]
    product = double_characters * spin_traces[None, :]
    multiplicities = (np.conj(single_characters) @ product.T).real / order
    counts = np.rint(multiplicities).astype(int)
    if np.max(np.abs(multiplicities - counts)) > 1e-3 or np.any(counts < 0):
        raise ValueError("D x D(1/2) does not decompose into single-valued irreps")
    fingerprints, parities, partners = [], [], []
    for d in range(double_characters.shape[0]):
        labels = sorted((name for name, n in zip(single_names, counts[:, d])
                         for _ in range(n)), key=_label_key)
        fingerprints.append(tuple(labels))
        signs = {label[-1] if label[-1] in "+-" else "" for label in labels}
        parity = signs.pop() if len(signs) == 1 else ""
        parities.append(parity)
        # the fingerprint without the parity suffixes: shared by D and D x det
        partners.append(tuple(_label_key(label[:-1] if parity else label)
                              for label in labels))
    indices = [int(m["n"]) for m in map(_LABEL_RE.match, single_names) if m is not None]
    offset = max(indices) if indices else len(single_names)
    classes: dict = {}
    for d in range(len(fingerprints)):
        classes.setdefault((int(double_dims[d]), partners[d]), []).append(d)
    prefix = k_label if k_label else "irrep_"
    names = [""] * len(fingerprints)
    number = offset
    for key in sorted(classes):
        members = sorted(classes[key], key=lambda d: {"+": 0, "-": 1}.get(parities[d], 2))
        if len({parities[d] for d in members}) != len(members) or (
                len(members) > 1 and "" in {parities[d] for d in members}):
            # no clean parity pairing: number the members one by one
            for d in members:
                number += 1
                names[d] = f"-{prefix}{number}{parities[d]}"
            continue
        number += 1
        for d in members:
            names[d] = f"-{prefix}{number}{parities[d]}"
    return names, fingerprints


def little_group_data(lattice, positions, numbers, k_frac, *, spinor: bool = False,
                      symprec: float = 1e-5, symmetry_cell=None) -> LittleGroup:
    """The little group of k, its small irreps and their names (cached).

    Args:
        lattice: Cell vectors as rows, in Angstrom (the run's own cell; it is
            not standardized).
        positions: Fractional atomic positions, shape ``(n_atoms, 3)``.
        numbers: Species ids; point-charge (Va) species are species of their
            own.
        k_frac: k in the reciprocal basis of ``lattice``.
        spinor: Double-valued irreps (spin-orbit coupled spinors).
        symprec: Symmetry tolerance of spglib.
        symmetry_cell: Optional ``(lattice, positions, numbers)`` whose space
            group and ISO-IR frame are used instead of those of the cell
            itself -- the crystal's cell for a sublattice run, so that all
            runs of one diagram are named in one frame.  Its lattice must be
            ``lattice`` and its operations must be symmetries of the cell.

    Returns:
        The :class:`LittleGroup` of k.

    Raises:
        ValueError: The cell is not primitive, ``symmetry_cell`` does not fit
            the cell, or the irreps cannot be constructed.
    """
    lattice = np.asarray(lattice, dtype=float)
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    numbers = np.asarray(numbers, dtype=int).reshape(-1)
    kpoint = _snap_kpoint(k_frac)
    if symmetry_cell is not None:
        frame = (np.asarray(symmetry_cell[0], dtype=float),
                 np.asarray(symmetry_cell[1], dtype=float).reshape(-1, 3),
                 np.asarray(symmetry_cell[2], dtype=int).reshape(-1))
        if not np.allclose(frame[0], lattice, atol=1e-5):
            raise ValueError("symmetry_cell must have the lattice of the run")
    else:
        frame = (lattice, positions, numbers)
    key = (bool(spinor), float(symprec), tuple(np.round(kpoint, 8))) + _cell_key(*frame)
    if symmetry_cell is not None:
        key += _cell_key(lattice, positions, numbers)
    if key in _LITTLE_GROUP_CACHE:
        return _LITTLE_GROUP_CACHE[key]

    sgnum, rotations, translations = _space_group(*frame, symprec)
    if symmetry_cell is not None:
        _check_operations(rotations, translations, positions, numbers, symprec, lattice)
    single, mapping = _scalar_irreps(rotations, translations, kpoint)
    single_characters = np.array([np.trace(rep, axis1=1, axis2=2) for rep in single])
    names, k_label, from_isoir = _isoir_names(
        sgnum, frame, symprec, kpoint, rotations[mapping], translations[mapping],
        single_characters)
    if not spinor:
        group = LittleGroup(
            sgnum=sgnum, kpoint=kpoint, k_label=k_label,
            rotations=rotations[mapping], translations=translations[mapping],
            spin_rotations=None, characters=single_characters,
            dims=np.array([rep.shape[1] for rep in single]), names=names,
            from_isoir=from_isoir)
    else:
        double, unitary, spin_mapping = _spinor_irreps(frame[0], rotations, translations,
                                                       kpoint)
        # the two enumerations list the little group in the same order; align
        # them explicitly all the same
        position = {int(op): i for i, op in enumerate(mapping)}
        if sorted(position) != sorted(int(op) for op in spin_mapping):
            raise ValueError("single- and double-valued little groups differ")
        reorder = np.array([position[int(op)] for op in spin_mapping])
        double_characters = np.array([np.trace(rep, axis1=1, axis2=2) for rep in double])
        dims = np.array([rep.shape[1] for rep in double])
        spin_traces = np.real(np.trace(unitary, axis1=1, axis2=2))
        double_names, fingerprints = _double_valued_names(
            k_label, names, single_characters[:, reorder], double_characters, dims,
            spin_traces)
        group = LittleGroup(
            sgnum=sgnum, kpoint=kpoint, k_label=k_label,
            rotations=rotations[spin_mapping], translations=translations[spin_mapping],
            spin_rotations=unitary, characters=double_characters, dims=dims,
            names=double_names, fingerprints=fingerprints, from_isoir=from_isoir)
    if len(_LITTLE_GROUP_CACHE) > 256:
        _LITTLE_GROUP_CACHE.clear()
    _LITTLE_GROUP_CACHE[key] = group
    return group


def spin_orbit_compatibility(lattice, positions, numbers, k_frac, *, symprec: float = 1e-5,
                             symmetry_cell=None, name_map: dict | None = None) -> dict:
    """The double-valued irreps in (single-valued irrep) x D(1/2) at k.

    The scalar bridge of a spin-orbit diagram: a level of a scalar-relativistic
    run with irrep Gamma splits under spin-orbit coupling into the
    double-valued irreps contained in Gamma x D(1/2).  The characters
    chi(g) = chi_Gamma(g) tr U_g are taken on the double-group elements of
    :func:`little_group_data` with ``spinor=True`` -- the coset
    representatives and the SU(2) matrices U_g of the spinor decomposition --
    and decomposed into the double-valued small irreps, so the names are
    exactly those :func:`label_bands` gives for a spinor run, in any space
    group and at any k (in Pm-3m: R4- -> -R6- + -R8-, the Pb 6p j = 1/2 and
    j = 3/2 levels of CsPbI3).

    Args:
        lattice: Cell vectors as rows, in Angstrom (the run's own setting).
        positions: Fractional atomic positions.
        numbers: Species ids.
        k_frac: k in the reciprocal basis of ``lattice``.
        symprec: Symmetry tolerance of spglib.
        symmetry_cell: Optional cell that fixes the space group and the
            ISO-IR frame (see :func:`little_group_data`); pass the one the
            levels were labelled with.
        name_map: Optional ``{crystod_name: other_name}`` for the
            double-valued names, e.g. the CrystOD-to-BCS correspondence of
            :func:`irrep_json_name_map`; a name the map lacks is kept.

    Returns:
        ``{single-valued label: [double-valued names]}`` with one entry per
        single-valued small irrep at k; each double-valued name appears as
        often as its multiplicity, in natural order, and the dimensions add
        up to twice that of the single-valued irrep
        (``{"R4-": ["-R6-", "-R8-"], ...}``).

    Raises:
        ValueError: The symmetry data cannot be built, or the product does
            not decompose into integral multiplicities (an inconsistent
            factor system; never seen).
    """
    single = little_group_data(lattice, positions, numbers, k_frac, spinor=False,
                               symprec=symprec, symmetry_cell=symmetry_cell)
    double = little_group_data(lattice, positions, numbers, k_frac, spinor=True,
                               symprec=symprec, symmetry_cell=symmetry_cell)

    def key(rotation, translation):
        return (np.asarray(rotation, dtype=np.int64).tobytes(),
                tuple(np.round(np.asarray(translation, dtype=float) % 1.0, 6) % 1.0))

    position = {key(R, t): i for i, (R, t) in enumerate(zip(single.rotations,
                                                             single.translations))}
    try:
        order = [position[key(R, t)] for R, t in zip(double.rotations, double.translations)]
    except KeyError as exc:
        raise ValueError("single- and double-valued little groups differ") from exc
    traces = np.real(np.trace(double.spin_rotations, axis1=1, axis2=2))
    product = single.characters[:, order] * traces[None, :]
    multiplicities = (np.conj(double.characters) @ product.T) / double.order
    counts = np.rint(multiplicities.real).astype(int)
    if (np.max(np.abs(multiplicities - counts)) > 1e-6 or np.any(counts < 0)
            or not np.array_equal(double.dims @ counts, 2 * single.dims)):
        raise ValueError("Gamma x D(1/2) does not decompose into double-valued irreps")
    mapping = name_map or {}
    result = {}
    for s, name in enumerate(single.names):
        parts = [mapping.get(double.names[d], double.names[d])
                 for d in range(len(double.names)) for _ in range(counts[d, s])]
        result[name] = sorted(parts, key=_label_key)
    return result


# ------------------------------------------------------------ product plan
@dataclass
class _Plan:
    """Which matrices come from the plane waves, and how the others follow.

    Attributes:
        operations: Little-group indices whose matrices <m|O_g|n> are formed
            from the coefficients (a generating set, or every operation).
        steps: ``(target, left, right, factor)`` with
            D(target) = factor * D(left) @ D(right), in an order in which the
            right-hand sides are known.
        identity: Index of the identity.
    """

    operations: list
    steps: list
    identity: int


def _product_plan(group: LittleGroup, all_operations: bool = False) -> _Plan:
    """A generating set of the little group and the products giving the rest.

    For g_a g_b = {E|n} g_c (coset representatives) and U_a U_b = z U_c,
    O_a O_b = z exp(-2 pi i k.n) O_c on the Bloch states at k, so on an
    invariant subspace D(c) = z exp(+2 pi i k.n) D(a) D(b).  The generators
    are taken greedily among the operations of highest order; two or three
    suffice for every crystallographic little group.

    Raises:
        ValueError: The operations do not close into a group.
    """
    key = "all" if all_operations else "generators"
    if key in group.plans:
        return group.plans[key]
    rotations = np.asarray(group.rotations, dtype=np.int64)
    position = {rotation.tobytes(): i for i, rotation in enumerate(rotations)}
    identity = position.get(np.eye(3, dtype=np.int64).tobytes())
    if identity is None or len(position) != group.order:
        raise ValueError("the little-group operations are not coset representatives")
    if all_operations:
        plan = _Plan(list(range(group.order)), [], identity)
        group.plans[key] = plan
        return plan

    def product(a, b):
        c = position.get((rotations[a] @ rotations[b]).tobytes())
        if c is None:
            raise ValueError("the little-group rotations do not form a group")
        shift = (rotations[a] @ group.translations[b] + group.translations[a]
                 - group.translations[c])
        lattice_vector = np.rint(shift)
        if np.max(np.abs(shift - lattice_vector)) > 1e-5:
            raise ValueError("the little-group translations do not close")
        factor = np.exp(2j * np.pi * float(group.kpoint @ lattice_vector))
        if group.spin_rotations is not None:
            pair = group.spin_rotations[a] @ group.spin_rotations[b]
            if np.allclose(pair, group.spin_rotations[c], atol=1e-6):
                pass
            elif np.allclose(pair, -group.spin_rotations[c], atol=1e-6):
                factor = -factor
            else:
                raise ValueError("the SU(2) matrices do not close up to sign")
        return c, complex(factor)

    def element_order(a):
        power, count = rotations[a].copy(), 1
        while not np.array_equal(power, np.eye(3, dtype=np.int64)):
            power, count = power @ rotations[a], count + 1
        return count

    candidates = sorted(range(group.order), key=lambda a: (-element_order(a), a))
    generators, steps, reached = [], [], {identity}
    for candidate in candidates:
        if candidate in reached:
            continue
        generators.append(candidate)
        reached, steps, frontier = {identity}, [], [identity]
        while frontier:
            fresh = []
            for a in frontier:
                for g in generators:
                    c, factor = product(a, g)
                    if c not in reached:
                        reached.add(c)
                        steps.append((c, a, g, factor))
                        fresh.append(c)
            frontier = fresh
        if len(reached) == group.order:
            break
    plan = _Plan(generators, steps, identity)
    group.plans[key] = plan
    return plan


# --------------------------------------------------------- plane-wave algebra
def _as_coefficients(coeffs, spinor: bool) -> np.ndarray:
    """Coefficients as ``(n_bands, n_spinor, n_pw)`` (complex64 is kept)."""
    array = np.asarray(coeffs)
    if not np.iscomplexobj(array):
        array = array.astype(np.complex128)
    if array.ndim == 2:
        array = array[:, None, :]
    if array.ndim != 3:
        raise ValueError("coeffs must have the shape (n_bands, n_spinor, n_pw)")
    expected = 2 if spinor else 1
    if array.shape[1] != expected:
        raise ValueError(
            f"coeffs carry {array.shape[1]} spinor component(s) but spinor={spinor}")
    return array


def _plane_wave_maps(gvecs, group: LittleGroup, operations):
    """Image index and phase of every plane wave under the given operations.

    Returns:
        ``(images, phases)``: ``images[j, i]`` is the position of
        G'_i = R_j^-T G_i + G_g in the list (``n_pw``, one past the end, when
        the image is missing) and ``phases[j, i]`` the factor
        exp(-2 pi i (k + G'_i) . t_j); ``phases[j]`` is ``None`` when t_j = 0.

    Raises:
        ValueError: The G list has duplicates or k is not invariant.
    """
    G = np.asarray(gvecs, dtype=np.int64).reshape(-1, 3)
    n_pw = len(G)
    low = G.min(axis=0)
    span = G.max(axis=0) - low + 1
    flat = np.ravel_multi_index((G - low).T, span)
    if len(np.unique(flat)) != n_pw:
        raise ValueError("the G-vector list has duplicates")
    lookup = np.full(int(np.prod(span)), n_pw, dtype=np.int64)
    lookup[flat] = np.arange(n_pw)
    k = group.kpoint
    images = np.empty((len(operations), n_pw), dtype=np.int64)
    phases = []
    for j, op in enumerate(operations):
        inverse = np.rint(np.linalg.inv(group.rotations[op])).astype(np.int64)
        shift = k @ inverse - k                       # G R^-1 = (R^-T G)^T
        if np.max(np.abs(shift - np.rint(shift))) > 1e-5:
            raise ValueError("k is not invariant under its little group")
        Gp = G @ inverse + np.rint(shift).astype(np.int64)
        index = Gp - low
        inside = np.all((index >= 0) & (index < span), axis=1)
        image = np.full(n_pw, n_pw, dtype=np.int64)
        image[inside] = lookup[np.ravel_multi_index(index[inside].T, span)]
        images[j] = image
        t = np.asarray(group.translations[op], dtype=float)
        phases.append(None if not np.any(t) else np.exp(-2j * np.pi * ((Gp + k) @ t)))
    return images, phases


def _block_matrices(coeffs, blocks, images, phases, spin_rotations):
    """``(S, M)`` of each band block: pseudo overlaps and <m|O_g|n>.

    ``M`` has one ``(n, n)`` matrix per entry of ``images`` (the operations
    of the plan); the coefficient arrays are traversed once per operation
    for all blocks together.
    """
    if not blocks:
        return []
    bands = sorted({b for block in blocks for b in block})
    where = {b: i for i, b in enumerate(bands)}
    C = coeffs[bands]
    n_bands, n_spinor, n_pw = C.shape
    # conj(C) with a zero column for missing images, gathered per operation
    padded = np.concatenate([C, np.zeros((n_bands, n_spinor, 1), dtype=C.dtype)],
                            axis=2).conj()
    rows = [np.array([where[b] for b in block]) for block in blocks]
    flat = C.reshape(n_bands, -1)
    overlaps = [(flat[r].conj() @ flat[r].T).astype(np.complex128) for r in rows]
    matrices = [np.empty((len(images), len(r), len(r)), dtype=np.complex128) for r in rows]
    identity = np.eye(2)
    for j in range(len(images)):
        moved = C if phases[j] is None else C * phases[j].astype(C.dtype)
        if spin_rotations is not None and not np.allclose(spin_rotations[j], identity):
            moved = np.matmul(spin_rotations[j].astype(C.dtype), moved)
        # <m|O_g|n> = sum_i conj(C_m(G'_i)) (U C_n(G_i)) phase_i
        target = np.take(padded, images[j], axis=2).reshape(n_bands, -1)
        moved = moved.reshape(n_bands, -1)
        for r, M in zip(rows, matrices):
            M[j] = target[r] @ moved[r].T
    return list(zip(overlaps, matrices))


def _lost_weight(coeffs, images) -> np.ndarray:
    """Per band, the largest norm share whose plane-wave images are missing."""
    n_pw = coeffs.shape[2]
    weights = np.sum(np.abs(coeffs) ** 2, axis=1)
    totals = np.sum(weights, axis=1)
    lost = np.zeros(len(coeffs))
    for image in images:
        missing = image == n_pw
        if np.any(missing):
            lost = np.maximum(lost, np.sum(weights[:, missing], axis=1) / totals)
    return lost


@dataclass
class _Assessment:
    """Decomposition of the representation carried by one block of bands."""

    characters: np.ndarray
    multiplicities: np.ndarray
    counts: np.ndarray
    deviation: float
    quality: float
    leakage: float
    integral: bool
    direct: bool = False


def _assess(S, M, group: LittleGroup, plan: _Plan) -> _Assessment:
    """Characters of a block of bands and their irrep decomposition.

    D(g) = S^-1 M(g) for the operations of the plan, the products of the
    plan for the others, chi(g) = tr D(g).  ``leakage`` is the largest share
    of the norm of a state of the block that an operation of the plan moves
    out of the block (lowest eigenvalue of M^H S^-1 M against S): zero for an
    invariant span, about 1/2 for one band of a Kramers pair.  Invariance
    under a generating set is invariance under the little group, so the
    products are representation matrices whenever the leakage is small.
    """
    n = S.shape[0]
    inverse = np.linalg.inv(S)
    D = [None] * group.order
    for index, op in enumerate(plan.operations):
        D[op] = inverse @ M[index]
    D[plan.identity] = np.eye(n, dtype=np.complex128)
    for target, left, right, factor in plan.steps:
        D[target] = factor * (D[left] @ D[right])
    characters = np.array([np.trace(matrix) for matrix in D])
    values, vectors = np.linalg.eigh(S)
    half = (vectors / np.sqrt(np.maximum(values, 1e-300))) @ vectors.conj().T
    captured = np.einsum("ij,gjk,kl->gil", half,
                         np.conj(np.swapaxes(M, 1, 2)) @ inverse @ M, half)
    leakage = float(max(0.0, 1.0 - np.min(np.linalg.eigvalsh(captured))))
    multiplicities = np.conj(group.characters) @ characters / group.order
    counts = np.clip(np.rint(multiplicities.real), 0, None).astype(int)
    deviation = float(np.max(np.abs(multiplicities - counts)))
    quality = float(np.max(np.abs(characters - counts @ group.characters)))
    integral = (leakage <= LEAKAGE_TOL and deviation <= MULTIPLICITY_TOL
                and int(counts @ group.dims) == n)
    return _Assessment(characters, multiplicities, counts, deviation, quality, leakage,
                       integral)


def _seed_levels(eigenvalues, tolerance: float) -> list:
    """Consecutive bands within ``tolerance`` of the first band of their level."""
    levels = []
    start = 0
    for index in range(1, len(eigenvalues) + 1):
        if (index == len(eigenvalues)
                or eigenvalues[index] - eigenvalues[start] > tolerance):
            levels.append(list(range(start, index)))
            start = index
    return levels


def _finest_split(bands, S, M, evaluate):
    """Shortest consecutive integral blocks ``[(start, stop, assessment)]``, or None."""
    n = S.shape[0]
    parts = []
    start = 0
    while start < n:
        for stop in range(start + 1, n + 1):
            block = slice(start, stop)
            assessment = evaluate(bands[block], S[block, block], M[:, block, block])
            if assessment.integral:
                parts.append((start, stop, assessment))
                break
        else:
            return None
        start = stop
    return parts


def _level_name(names, counts) -> str:
    """``GM5+``, or ``GM4-/GM5-`` for an accidental degeneracy (alphabetical)."""
    parts = []
    for name, count in zip(names, counts):
        parts.extend([name] * int(count))
    return "/".join(sorted(parts))


def irreps_at_kpoint(lattice, positions, numbers, k_frac, gvecs, coeffs, eigenvalues, *,
                     spinor: bool = False, degeneracy_tol: float | None = None,
                     symprec: float = 1e-5, symmetry_cell=None,
                     max_window: float = DEGENERACY_MAX_WINDOW_EV,
                     all_operations: bool = False) -> list:
    """Levels of one k point with their irreps, from plane-wave coefficients.

    Args:
        lattice: Cell vectors as rows, in Angstrom (the run's own setting).
        positions: Fractional atomic positions.
        numbers: Species ids (a Va point-charge species is a species).
        k_frac: k in the reciprocal basis of ``lattice``.
        gvecs: Integer G vectors of the coefficients, shape ``(n_pw, 3)``, in
            the order of ``coeffs`` (any order; the WAVECAR's is fine).
        coeffs: Plane-wave coefficients, shape ``(n_bands, n_spinor, n_pw)``
            (``(n_bands, n_pw)`` for scalar states; complex64 is used as it
            is); spinor component 0 is spin up along Cartesian z.
        eigenvalues: Band energies in eV, ascending, shape ``(n_bands,)``.
        spinor: The states are spinors (vasp_ncl): double-valued irreps.
        degeneracy_tol: Seed window of the level clustering in eV
            (default :data:`DEGENERACY_SEED_EV`).
        symprec: Symmetry tolerance of spglib.
        symmetry_cell: Optional cell that fixes the space group and the
            ISO-IR frame (see :func:`little_group_data`).
        max_window: Cap in eV of the merging of non-integral levels.
        all_operations: Form <m|O_g|n> from the plane waves for every
            little-group operation instead of a generating set (a check;
            three to sixteen times slower).

    Returns:
        One dict per level, in energy order: ``bands`` (0-based band
        indices), ``energy`` (mean, eV), ``spread`` (eV), ``dim``,
        ``label`` (``""`` when the level is not a complete multiplet),
        ``irreps`` (``[(name, count), ...]``), ``characters`` (complex array
        over the operations of :func:`little_group_data`),
        ``multiplicities`` (``{name: value}``, values above 1e-3),
        ``quality`` (largest character deviation from the integral
        decomposition), ``deviation`` (largest multiplicity deviation),
        ``leakage`` (largest norm share an operation moves out of the
        level), ``integral`` and ``lost`` (largest norm share of a band
        whose plane-wave images fell outside the G list).

    Raises:
        ValueError: Inconsistent shapes, a non-primitive cell, or symmetry
            data that cannot be built.
    """
    C = _as_coefficients(coeffs, spinor)
    energies = np.asarray(eigenvalues, dtype=float).reshape(-1)
    G = np.asarray(gvecs, dtype=np.int64).reshape(-1, 3)
    if len(energies) != C.shape[0] or len(G) != C.shape[2]:
        raise ValueError("coeffs, gvecs and eigenvalues do not match in size")
    if np.any(np.diff(energies) < -1e-9):
        raise ValueError("eigenvalues must be in ascending order")
    group = little_group_data(lattice, positions, numbers, k_frac, spinor=spinor,
                              symprec=symprec, symmetry_cell=symmetry_cell)
    plan = _product_plan(group, all_operations)
    images, phases = _plane_wave_maps(G, group, plan.operations)
    spin = (None if group.spin_rotations is None
            else group.spin_rotations[plan.operations])
    lost = _lost_weight(C, images)
    if np.max(lost, initial=0.0) > 1e-2:
        raise ValueError(
            f"the images of the G vectors carry up to {100 * np.max(lost):.0f} % of a "
            "band's norm out of the list: the G list is not closed under the little "
            "group (a gamma-only WAVECAR, or k and G in different bases)")
    tolerance = DEGENERACY_SEED_EV if degeneracy_tol is None else float(degeneracy_tol)

    # the products of a generating set are exact on invariant spans only: a
    # block that leaks noticeably (the last, least converged bands of a run,
    # mixed near-degenerate levels) is judged on characters formed from every
    # operation, so that the outcome does not depend on the generators chosen
    full = _product_plan(group, True)
    direct_maps: list = []
    direct_cache: dict = {}

    def direct(bands):
        key = tuple(bands)
        if key not in direct_cache:
            if not direct_maps:
                direct_maps.extend(_plane_wave_maps(G, group, full.operations))
            S, M = _block_matrices(C, [list(bands)], *direct_maps, group.spin_rotations)[0]
            direct_cache[key] = _assess(S, M, group, full)
            direct_cache[key].direct = True
        return direct_cache[key]

    def evaluate(bands, S, M):
        assessment = _assess(S, M, group, plan)
        assessment.direct = all_operations
        if all_operations or not _DIRECT_LEAKAGE < assessment.leakage <= LEAKAGE_TOL:
            return assessment
        return direct(bands)

    seeds = _seed_levels(energies, tolerance)
    matrices = _block_matrices(C, seeds, images, phases, spin)
    assessments = [evaluate(seed, S, M) for seed, (S, M) in zip(seeds, matrices)]

    def mean(band_list):
        return float(np.mean(energies[band_list]))

    # a non-integral seed may absorb the following seeds within max_window:
    # the matrices of every such window come from one more pass
    windows = {}
    for index, assessment in enumerate(assessments):
        if assessment.integral:
            continue
        last = index
        while (last + 1 < len(seeds)
               and mean(seeds[last + 1]) - mean(seeds[index]) <= max_window):
            last += 1
        if last > index:
            windows[index] = (last, [b for seed in seeds[index:last + 1] for b in seed])
    window_matrices = dict(zip(windows, _block_matrices(
        C, [union for _, union in windows.values()], images, phases, spin)))

    # (bands, S, M, assessment) of every level
    levels = []
    index = 0
    while index < len(seeds):
        found = None
        if not assessments[index].integral and index in windows:
            last, union = windows[index]
            S, M = window_matrices[index]
            size = len(seeds[index])
            for stop in range(index + 1, last + 1):
                size += len(seeds[stop])
                block = slice(0, size)
                assessment = evaluate(union[:size], S[block, block], M[:, block, block])
                if assessment.integral:
                    found = (stop, union[:size], S[block, block], M[:, block, block],
                             assessment)
                    break
        if found is None:
            levels.append((seeds[index], *matrices[index], assessments[index]))
            index += 1
        else:
            levels.append(found[1:])
            index = found[0] + 1

    # an integral level holding several irreps: its finest integral blocks
    refined = []
    for bands, S, M, assessment in levels:
        if assessment.integral and int(assessment.counts.sum()) > 1:
            parts = _finest_split(bands, S, M, evaluate)
            if parts is not None and len(parts) > 1:
                refined.extend((bands[start:stop], part) for start, stop, part in parts)
                continue
        refined.append((bands, assessment))

    # an unassigned level reports the characters of every operation, which
    # mean something for a non-invariant span, unlike the products
    for i, (bands, assessment) in enumerate(refined):
        if not assessment.integral and not assessment.direct:
            fresh = direct(bands)
            refined[i] = (bands, _Assessment(
                fresh.characters, fresh.multiplicities, fresh.counts, fresh.deviation,
                fresh.quality, max(fresh.leakage, assessment.leakage), False, True))

    result = []
    for bands, assessment in refined:
        integral = bool(assessment.integral)
        result.append({
            "bands": [int(b) for b in bands],
            "energy": mean(bands),
            "spread": float(energies[bands[-1]] - energies[bands[0]]),
            "dim": len(bands),
            "label": _level_name(group.names, assessment.counts) if integral else "",
            "irreps": [(name, int(count)) for name, count
                       in zip(group.names, assessment.counts) if count > 0]
            if integral else [],
            "characters": assessment.characters,
            "multiplicities": {
                name: round(float(value.real), 4)
                for name, value in zip(group.names, assessment.multiplicities)
                if abs(value) > 1e-3},
            "quality": assessment.quality,
            "deviation": assessment.deviation,
            "leakage": assessment.leakage,
            "integral": integral,
            "lost": float(np.max(lost[bands])),
        })
    return result


def labels_from_levels(levels, n_bands: int, key: str = "label") -> list:
    """One label per band from a level list (``""`` for bands not covered).

    Args:
        levels: The output of :func:`irreps_at_kpoint` (or any list of dicts
            with ``bands`` and the label ``key``).
        n_bands: Number of bands.
        key: The dict entry to spread over the bands of each level.

    Returns:
        A list of ``n_bands`` strings.
    """
    labels = [""] * int(n_bands)
    for level in levels:
        for band in level["bands"]:
            if 0 <= band < n_bands:
                labels[band] = level.get(key) or ""
    return labels


def label_bands(lattice, positions, numbers, k_frac, gvecs, coeffs, eigenvalues, *,
                spinor: bool = False, degeneracy_tol: float | None = None,
                symprec: float = 1e-5, symmetry_cell=None, irrep_json=None,
                band_offset: int = 0) -> list:
    """One irrep label per band (``""`` where a band has no complete level).

    The arguments are those of :func:`irreps_at_kpoint`.  With
    ``irrep_json`` (the path or the parsed data of IrRep's JSON for the same
    run) the levels carry IrRep's BCS names instead (:func:`bcs_labels`) --
    the route to the ``-R8``-style names of spinor runs; a level that
    neither IrRep nor the name map names keeps its CrystOD name (for a
    double-valued irrep recognisable by its parity suffix, ``-R6+``).

    Args:
        irrep_json: Optional IrRep JSON of the same run.
        band_offset: 0-based index of the first band IrRep analysed.

    Returns:
        A list with one string per band.
    """
    levels = irreps_at_kpoint(lattice, positions, numbers, k_frac, gvecs, coeffs,
                              eigenvalues, spinor=spinor, degeneracy_tol=degeneracy_tol,
                              symprec=symprec, symmetry_cell=symmetry_cell)
    n_bands = len(np.asarray(eigenvalues).reshape(-1))
    if irrep_json is not None:
        names = bcs_labels(levels, irrep_json, k_frac, band_offset=band_offset)
        if names is not None:
            for level, name in zip(levels, names):
                if name:
                    level["label"] = name
    return labels_from_levels(levels, n_bands)


# ------------------------------------------------------------ IrRep's JSON
def read_irrep_json(source) -> list:
    """The k points of an IrRep (2.x) JSON file.

    Only the file is read; the GPL ``irrep`` package is not imported.

    Args:
        source: Path of the JSON (``irrep-output`` or ``irrep-output.json``)
            or its parsed content.

    Returns:
        ``[(k, [(dimension, {name: multiplicity}), ...]), ...]`` -- one
        entry per k point, one ``(dimension, multiplicities)`` per IrRep
        level from the first band IrRep analysed, multiplicities as the real
        parts IrRep prints (``-R8`` is BCS's \\bar{R}_8).
    """
    data = source
    if not isinstance(source, (dict, list)):
        with open(source) as handle:
            data = json.load(handle)
    if isinstance(data, list):
        return data
    out = []
    for block in data.get("characters and irreps", []):
        for point in block["subspace"]["k points"]:
            k = np.asarray(point["k"]["data"] if isinstance(point["k"], dict)
                           else point["k"], dtype=float)
            dims = point["dimensions"]
            dims = dims["data"] if isinstance(dims, dict) else dims
            levels = []
            for dim, irreps in zip(dims, point["irreps"]):
                values: dict = {}
                for name, value in (irreps or {}).items():
                    real = value[0] if isinstance(value, (list, tuple)) else float(value)
                    values[name] = values.get(name, 0.0) + float(real)
                levels.append((int(dim), values))
            out.append((k, levels))
    return out


def _json_level_name(values: dict) -> str:
    """IrRep multiplicities as a label (``?`` marks a non-integral one)."""
    parts = []
    for name, value in sorted(values.items()):
        if value < 0.15:
            continue
        if abs(value - round(value)) > 0.15:
            parts.append(name + "?")
        else:
            parts.extend([name] * int(round(value)))
    return "/".join(parts)


def irrep_json_labels(levels, irrep_json, k_frac, *, band_offset: int = 0,
                      tol: float = 1e-5) -> list | None:
    """IrRep's (BCS) names of the levels of :func:`irreps_at_kpoint`.

    A level is named when the IrRep levels that cover its bands cover exactly
    those bands; their multiplicities are summed (a doublet IrRep splits by
    more than its 1e-4 eV threshold is one irrep here), and a non-integral
    one is marked with ``?``.

    Args:
        levels: The output of :func:`irreps_at_kpoint`.
        irrep_json: Path or parsed content of IrRep's JSON for the same run
            (or the output of :func:`read_irrep_json`).
        k_frac: The k point, in the reciprocal basis of the run.
        band_offset: 0-based index of the first band IrRep analysed.
        tol: Tolerance of the k-point match (modulo reciprocal lattice
            vectors).

    Returns:
        One name per level (``""`` where the groupings differ), or ``None``
        when the JSON has no such k point.
    """
    data = read_irrep_json(irrep_json)
    k = np.asarray(k_frac, dtype=float)
    for kv, json_levels in data:
        d = np.asarray(kv, dtype=float) - k
        if np.max(np.abs(d - np.rint(d))) > tol:
            continue
        owner = []
        for index, (dim, _values) in enumerate(json_levels):
            owner.extend([index] * dim)
        names = []
        for level in levels:
            bands = [b - band_offset for b in level["bands"]]
            if min(bands) < 0 or max(bands) >= len(owner):
                names.append("")
                continue
            covering = sorted({owner[b] for b in bands})
            if sum(json_levels[i][0] for i in covering) != len(bands):
                names.append("")
                continue
            total: dict = {}
            for i in covering:
                for name, value in json_levels[i][1].items():
                    total[name] = total.get(name, 0.0) + value
            names.append(_json_level_name(total))
        return names
    return None


def irrep_json_name_map(levels, irrep_json, k_frac, *, band_offset: int = 0):
    """The correspondence of CrystOD's irrep names and IrRep's at one k point.

    Only levels made of one irrep on both sides enter.

    Args:
        levels: The output of :func:`irreps_at_kpoint`.
        irrep_json: IrRep's JSON for the same run (see :func:`irrep_json_labels`).
        k_frac: The k point.
        band_offset: 0-based index of the first band IrRep analysed.

    Returns:
        ``(mapping, conflicts)``: ``{crystod_name: irrep_name}`` and the list
        of ``(crystod_name, irrep_name)`` pairs that contradict a one-to-one
        correspondence (empty when the two namings agree level by level).
        ``({}, [])`` when the JSON has no such k point.
    """
    names = irrep_json_labels(levels, read_irrep_json(irrep_json), k_frac,
                              band_offset=band_offset)
    if names is None:
        return {}, []
    mapping: dict = {}
    reverse: dict = {}
    conflicts = []
    for level, other in zip(levels, names):
        if len(level["irreps"]) != 1 or level["irreps"][0][1] != 1:
            continue
        if not other or "/" in other or "?" in other:
            continue
        mine = level["irreps"][0][0]
        if mapping.setdefault(mine, other) != other or reverse.setdefault(other, mine) != mine:
            conflicts.append((mine, other))
    return mapping, conflicts


def bcs_labels(levels, irrep_json, k_frac, *, band_offset: int = 0,
               name_map: dict | None = None) -> list | None:
    """BCS names for the levels of :func:`irreps_at_kpoint`, from IrRep's JSON.

    A level gets IrRep's own name where IrRep's levels cover exactly its
    bands (:func:`irrep_json_labels`).  Where the groupings differ -- IrRep
    lumps an accidental degeneracy that CrystOD resolves, or splits a
    multiplet broken by more than its threshold -- the CrystOD label is
    translated with the CrystOD-to-BCS correspondence of this k point
    (:func:`irrep_json_name_map`), so that every labelled level is named in
    one convention.

    Args:
        levels: The output of :func:`irreps_at_kpoint`.
        irrep_json: IrRep's JSON for the same run (path, content or the output
            of :func:`read_irrep_json`).
        k_frac: The k point.
        band_offset: 0-based index of the first band IrRep analysed.
        name_map: Optional ``{crystod_name: bcs_name}`` to use instead of the
            correspondence measured on this run (e.g. one collected over the
            three runs of a diagram).

    Returns:
        One name per level (``""`` where neither route names it, IrRep's
        non-integral ``?`` names dropped), or ``None`` when the JSON has no
        such k point.
    """
    data = read_irrep_json(irrep_json)
    names = irrep_json_labels(levels, data, k_frac, band_offset=band_offset)
    if names is None:
        return None
    if name_map is None:
        name_map, _ = irrep_json_name_map(levels, data, k_frac, band_offset=band_offset)
    out = []
    for level, name in zip(levels, names):
        if name and "?" not in name:
            out.append(name)
            continue
        parts = level["label"].split("/") if level["label"] else []
        if parts and all(part in name_map for part in parts):
            out.append("/".join(sorted(name_map[part] for part in parts)))
        else:
            out.append("")
    return out


def irrep_executable(given: str | None = None) -> str | None:
    """The IrRep executable: ``given``, the one next to the running Python,
    or the first ``irrep`` on PATH (``None`` when there is none)."""
    import os
    import shutil
    import sys

    candidates = [given] if given else []
    candidates += [os.path.join(os.path.dirname(sys.executable), "irrep"),
                   shutil.which("irrep")]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def run_irrep(run_dir, k_indices, k_names, out_dir, *, spinor: bool = False,
              ecut: float | None = None, degen_thresh: float | None = None,
              executable: str | None = None, threads: int = 2) -> str | None:
    """Run IrRep on a VASP run as an external program; return its JSON path.

    IrRep (GPL) is called through a subprocess only, never imported.  The
    labels refer to the POSCAR setting (``-refUC`` = identity, ``-shiftUC``
    = 0), like CrystOD's.  Without ``ecut`` IrRep uses the full plane-wave
    basis of the run: its often quoted ``-Ecut=50`` drops every plane wave
    above 50 eV, and a state 40 eV up that lives in those waves then gets
    an arbitrary label (seen on SrGeO3 at Gamma and X).  Nothing is written
    into ``run_dir``; an existing JSON in ``out_dir`` is reused.

    Args:
        run_dir: VASP directory with ``WAVECAR`` and ``POSCAR``.
        k_indices: 0-based indices of the k points in the WAVECAR.
        k_names: Their names (``"GM"``, ``"X"``, ...), passed to IrRep.
        out_dir: Directory for IrRep's files (created).
        spinor: A vasp_ncl run (IrRep's ``-spinor``).
        ecut: Optional plane-wave cutoff in eV for IrRep's traces.
        degen_thresh: Optional degeneracy threshold in eV (IrRep's default
            is 1e-4 eV).
        executable: Path of the ``irrep`` program (see :func:`irrep_executable`).
        threads: OpenMP threads of the IrRep process.

    Returns:
        The path of the JSON, or ``None`` when IrRep is not available or
        failed (its output is in ``out_dir/irrep.log``).
    """
    import os
    import subprocess

    out_dir = os.path.abspath(out_dir)
    for name in ("irrep-output", "irrep-output.json"):
        found = os.path.join(out_dir, name)
        if os.path.isfile(found) and os.path.getsize(found) > 0:
            return found
    program = irrep_executable(executable)
    if program is None:
        return None
    os.makedirs(out_dir, exist_ok=True)
    command = [program, "-code=vasp",
               f"-fWAV={os.path.join(os.path.abspath(run_dir), 'WAVECAR')}",
               f"-fPOS={os.path.join(os.path.abspath(run_dir), 'POSCAR')}",
               "-kpoints=" + ",".join(str(int(i) + 1) for i in k_indices),
               "-kpnames=" + ",".join(k_names),
               "-refUC=1,0,0,0,1,0,0,0,1", "-shiftUC=0,0,0",
               "-json_file=irrep-output"]
    if spinor:
        command.insert(1, "-spinor")
    if ecut is not None:
        command.append(f"-Ecut={float(ecut):g}")
    if degen_thresh is not None:
        command.append(f"-degenThresh={float(degen_thresh):g}")
    environment = dict(os.environ, OMP_NUM_THREADS=str(int(threads)))
    with open(os.path.join(out_dir, "irrep.log"), "w") as log:
        log.write(" ".join(command) + "\n")
        log.flush()
        subprocess.run(command, cwd=out_dir, stdout=log, stderr=subprocess.STDOUT,
                       env=environment, check=False)
    for name in ("irrep-output", "irrep-output.json"):
        found = os.path.join(out_dir, name)
        if os.path.isfile(found) and os.path.getsize(found) > 0:
            return found
    return None

