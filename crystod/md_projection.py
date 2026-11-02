"""Mode projection of an MD trajectory (``crystod-md --project``).

Every frame of a molecular-dynamics trajectory of a supercell is turned into
a displacement field relative to the ideal supercell of the parent primitive
cell, and that field is projected onto the symmetry-adapted displacement
patterns of the parent (the bases ``crystod-phonon --vibration`` lists, one
per irrep occurrence and q point). The result is one time series ``Q(t)`` per
component, with its statistics and histogram.

Convention of the amplitude ``Q``. A real pattern ``p`` on the supercell of
``N = n1 n2 n3`` parent cells is normalized to unit norm over the whole
supercell, ``p_hat = p / |p|``, and

    Q(t) = p_hat . u(t) / sqrt(N),

with ``u(t)`` the Cartesian displacements of all atoms (Angstrom), from which
the mass-weighted centre-of-mass displacement of each frame is removed. ``Q``
is the displacement norm per parent primitive cell carried by the pattern. At
a time-reversal invariant q it is the amplitude ``crystod-phonon --modulation
--amplitude`` and ``--vibration`` use: a structure frozen with ``--amplitude
0.3`` projects to ``Q = 0.3``. At a q that is not time-reversal invariant
(``2q`` not a reciprocal lattice vector) each complex row ``e`` of a mode
space gives two real patterns, those of ``e`` and ``i e`` (components ``1c``
and ``1s``); q and -q span the same real patterns, so only one of them is
projected, and a modulation frozen with ``--amplitude A`` there gives
``A / sqrt(2)`` as the norm of ``Q`` over its ``c``/``s`` components (the rms
per-cell norm of the frozen wave). The AMPLIMODES amplitude of an irrep in
``crystod-group --supergroup-cif`` is ``sqrt(size)`` times the norm of its
``Q`` over all components and arms of the star (``Q sqrt(size)`` for a single
component), with ``size`` the size of the child primitive cell in parent
cells. The patterns of all commensurate q form an orthonormal basis of the
supercell displacements, hence the sum
rule ``sum_c <Q_c^2> = <|u|^2> / N``.

Phonon-eigenvector basis. With a phonopy object (force constants, ``--yaml``)
the basis at q is the set of symmetry-adapted eigenvectors ``e_nu`` of the
dynamical matrix that ``crystod-phonon --modulation`` and ``--vector`` use
(:func:`crystod.symmetry_adapted_modes.solve_symmetry_adapted_modes`; mode
numbers 1-based as there): unit, mass-weighted, atom-position phase. The real
supercell patterns ``p_nu(j, R) = Re(e_nu,j exp(2 pi i q.(R + x_j)))`` (and
those of ``i e_nu`` at a q that is not time-reversal invariant) are built by
the same code as the symmetry-only patterns, and the trajectory is taken in
mass-weighted coordinates ``w_j(R) = sqrt(m_j) u_j(R)``: the normal
coordinate of the supercell is ``a_nu(t) = p_hat_nu . w(t)`` (Angstrom
sqrt(amu)), ``A_nu = a_nu / sqrt(N)`` per parent cell, and over all
commensurate q ``sum <a_nu^2> = <sum_jR m_j |u_j(R)|^2>``. For a harmonic
mode in equilibrium classical MD at temperature T, ``<a_nu^2> = k_B T /
omega_nu^2``, which defines the effective frequency
``nu_eff = sqrt(k_B T / <a_nu^2>) / (2 pi)``.
"""

from __future__ import annotations

import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field, replace
from itertools import product

import numpy as np
from numpy.typing import NDArray

# a histogram density at Q = 0 below this fraction of its maximum reads as
# two peaks (when both sides of Q = 0 carry weight)
DOUBLE_PEAK_RATIO = 0.5
# fewer frames than this give no shape hint
MIN_FRAMES_FOR_SHAPE = 20
# an atom farther than this fraction of the shortest parent lattice vector
# from every ideal site of its element is not matched
MATCH_FRACTION = 0.25
# the trajectory cell divided by --dim must agree with the parent cell
# within this fraction of the parent lattice-vector length
LATTICE_FRACTION = 0.10
GAUSSIAN_RATIO = math.sqrt(2.0 / math.pi)
# the components of a mode space are pooled only when their rms values agree
# within this factor (else the pooled shape reads "unequal")
POOL_RMS_FACTOR = 3.0
# Boltzmann constant in amu A^2 ps^-2 K^-1 (1.380649e-23 J/K / (1.66053907e-27
# kg x 1e-20 m^2 / 1e-24 s^2)): k_B T / <a^2> is then omega^2 in (rad/ps)^2
KB_AMU_A2_PS2 = 0.8314462618
# the -c cell and the primitive cell of --yaml agree when lattice vectors and
# atom positions differ by less than this (Angstrom) or the symmetry tolerance
CELL_AGREEMENT = 1e-4


@dataclass
class ComponentStatistics:
    """Statistics of one time series ``Q(t)`` (Angstrom).

    Attributes:
        mean: ``<Q>``.
        rms: ``sqrt<Q^2>``.
        mean_abs: ``<|Q|>``.
        ratio: ``<|Q|> / sqrt<Q^2>`` (0.798 for a Gaussian centred at 0,
            1 for two states ``+-Q0``); NaN for a zero series.
        kurtosis: Excess kurtosis about the mean (0 for a Gaussian, -2 for two
            equally populated states, -1.5 for a sinusoid); NaN for a constant
            series.
        center_ratio: Histogram density at ``Q = 0`` divided by its maximum
            (histogram symmetric about 0); NaN when no shape is given.
        shape: ``"single-peaked"``, ``"double-peaked (hint)"``,
            ``"off-center"`` (the weight sits on one side of ``Q = 0``),
            ``"zero"`` or ``"n/a"`` (fewer than 20 frames).
        n_eff: Effective number of independent samples, the number of frames
            divided by the integrated autocorrelation time of ``Q(t)``
            (summed up to the first zero of the autocorrelation function);
            the ratio and the kurtosis scatter by roughly ``1/sqrt(n_eff)``.
            NaN for a constant series; for pooled components, the sum over
            them.
    """

    mean: float
    rms: float
    mean_abs: float
    ratio: float
    kurtosis: float
    center_ratio: float
    shape: str
    n_eff: float = float("nan")


@dataclass
class NormStatistics:
    """One mode space with several real components, taken as a whole.

    In a structure with the full parent symmetry on average, the components
    of one mode space are equivalent and share one distribution; pooling the
    samples of all components gives a better estimate of its shape than any
    single component.

    Attributes:
        irrep: ISO-IR label with dimension, e.g. ``"R4+(3)"``.
        space: Mode-space number (1-based, as ``crystod-phonon --vibration``).
        n_components: Number of real components.
        pooled: :class:`ComponentStatistics` of the samples of all
            components pooled into one distribution. When the component
            rms values differ by more than ``POOL_RMS_FACTOR`` (or one is
            zero) the components are not equivalent, and the pooled ratio,
            kurtosis and ``n_eff`` are ``nan`` with the shape ``"unequal"``.
        values: The norm ``|Q|`` of the component vector per frame, shape
            ``(n_frames,)``.
        mean: ``<|Q|>``.
        rms: ``sqrt<|Q|^2>``.
        ratio: ``<|Q|> / sqrt<|Q|^2>``.
        gaussian_ratio: The same ratio for an isotropic Gaussian in
            ``n_components`` dimensions (0.921 for three).
    """

    irrep: str
    space: int
    n_components: int
    pooled: ComponentStatistics
    values: NDArray[np.float64]
    mean: float
    rms: float
    ratio: float
    gaussian_ratio: float


@dataclass
class QProjection:
    """The projection at one q point.

    Attributes:
        name: ISO-IR name of q (``"R"``, ``"X"``, ``"DT"``, ...).
        tag: File-name qualifier (the name, or the name with coordinates when
            several projected q share it).
        qpoint: Fractional coordinates in the parent reciprocal basis.
        time_reversal_invariant: ``2q`` is a reciprocal lattice vector (one
            real pattern per row, components ``1, 2, ...``; otherwise two,
            ``1c, 1s, ...``).
        irreps: ISO-IR label of each column, e.g. ``"R4+(3)"``.
        spaces: Mode-space number of each column (1-based).
        components: Component of each column within its space (``"1"`` or
            ``"1c"``/``"1s"``).
        columns: Column names ``label_space_component`` (``"R4+(3)_3_1"``).
        values: ``Q(t)``, shape ``(n_frames, n_columns)``.
        statistics: One :class:`ComponentStatistics` per column.
        norms: One :class:`NormStatistics` per mode space with more than one
            real component (pooled statistics and the norm ``|Q|``).
        frequencies: Phonon-eigenvector basis only: the harmonic frequency of
            each column (THz, imaginary as negative); ``None`` otherwise. In
            that basis ``spaces`` holds the number of the first mode of the
            degenerate level of each column, ``components`` the mode number
            (``"4"``, or ``"4c"``/``"4s"``), ``columns`` the names
            ``<mode>_<irrep>`` (``"4_R4+"``, ``"4c_DT1"``) and ``values`` the
            normal coordinates ``a(t)`` of the supercell (A sqrt(amu)).
        effective_frequencies: With a temperature: ``sqrt(k_B T / <a^2>) /
            (2 pi)`` per column (THz; NaN for a zero series); else ``None``.
    """

    name: str
    tag: str
    qpoint: tuple[float, float, float]
    time_reversal_invariant: bool
    irreps: list[str]
    spaces: list[int]
    components: list[str]
    columns: list[str]
    values: NDArray[np.float64]
    statistics: list[ComponentStatistics]
    norms: list[NormStatistics]
    frequencies: list[float] | None = None
    effective_frequencies: list[float] | None = None


@dataclass
class SiteMatching:
    """How the trajectory atoms were matched to the ideal supercell sites.

    Attributes:
        site_of_atom: Ideal-site index of every trajectory atom (sites in the
            order ``(i1, i2, i3, parent atom)`` of
            ``SymmetryOnlyVibrations.get_supercell_displacements``).
        uniform_shift: Mean deviation of the time-averaged positions from
            their sites, Cartesian (Angstrom).
        max_deviation: Largest deviation from a site after removing the
            uniform shift (Angstrom).
        max_deviation_atom: 1-based index (file order) of that atom.
    """

    site_of_atom: NDArray[np.int_]
    uniform_shift: NDArray[np.float64]
    max_deviation: float
    max_deviation_atom: int


@dataclass
class TrajectoryProjection:
    """Result of :func:`project_trajectory`.

    Attributes:
        formula: Reduced formula of the parent (``"ScF3"``).
        space_group: International symbol of the parent.
        space_group_number: Its number.
        dim: Supercell multiplicities ``(n1, n2, n3)``.
        n_atoms: Atoms in the supercell.
        steps: The step number of every frame used.
        lattice_mode: ``"fixed"`` (one cell), ``"frame"`` (each frame's own
            cell) or ``"mean"`` (the time-averaged cell, ``mean_lattice``).
        matching: The :class:`SiteMatching`.
        projections: One :class:`QProjection` per projected q.
        skipped: Notes on requested q points that were not projected (the -q
            partner of a projected q, or no irrep left by the filter) and
            on filter names that match no irrep at the projected q.
        irrep_filter: The irrep names of the filter, or ``None``.
        projected_msd: ``sum_c <Q_c^2>`` over the projected columns (A^2);
            in the phonon-eigenvector basis ``sum <a^2>`` (amu A^2).
        total_msd: ``<|u|^2> / N`` after drift removal (A^2); in the
            phonon-eigenvector basis ``<sum_jR m_j |u_j(R)|^2>`` over the
            supercell (amu A^2).
        drift_rms: Root mean square over the frames of the removed
            centre-of-mass displacement (Angstrom; it includes a static
            uniform offset of the structure from the ideal sites).
        n_q_covered: Commensurate q covered by the projection (a -q
            partner counts with its q).
        n_q_commensurate: All commensurate q of the supercell.
        basis: ``"symmetry"`` (the patterns of ``crystod-phonon
            --vibration``) or ``"eigenvector"`` (phonon eigenvectors of a
            phonopy object).
        temperature: The temperature of the effective frequencies (K), or
            ``None``.
        star: Every arm of the star of each requested q was projected.
    """

    formula: str
    space_group: str
    space_group_number: int
    dim: tuple[int, int, int]
    n_atoms: int
    steps: NDArray[np.int_]
    lattice_mode: str
    matching: SiteMatching
    projections: list[QProjection]
    skipped: list[str] = field(default_factory=list)
    irrep_filter: list[str] | None = None
    projected_msd: float = 0.0
    total_msd: float = 0.0
    drift_rms: float = 0.0
    n_q_covered: int = 0
    n_q_commensurate: int = 0
    basis: str = "symmetry"
    temperature: float | None = None
    star: bool = False

    @property
    def complete(self) -> bool:
        """Every commensurate q and every irrep was projected."""
        return self.n_q_covered == self.n_q_commensurate and self.irrep_filter is None


# ---------------------------------------------------------------- helpers
def _irrep_name(label: str) -> str:
    """``"R4+(3)"`` -> ``"R4+"``."""
    return re.sub(r"\(\d+\)$", "", label)


def _q_key(qpoint) -> tuple[float, ...]:
    """q modulo reciprocal lattice vectors, rounded, for comparisons."""
    reduced = np.mod(np.asarray(qpoint, dtype=float), 1.0)
    reduced = np.where(reduced > 1.0 - 1e-8, 0.0, reduced)
    return tuple(float(value) + 0.0 for value in np.round(reduced, 8))


def _is_time_reversal_invariant(qpoint) -> bool:
    doubled = 2.0 * np.asarray(qpoint, dtype=float)
    return bool(np.allclose(doubled, np.rint(doubled), atol=1e-8))


def commensurate_qpoints(parent_lattice, dim) -> list[tuple[float, float, float]]:
    """All q of the parent cell commensurate with a diagonal supercell.

    The enumeration of :func:`crystod.phonon_subgroups.commensurate_qpoints`
    (``crystod-phonon --subgroup``) applied to the supercell
    ``diag(dim)`` of the parent.

    Args:
        parent_lattice: Parent lattice vectors as rows, shape ``(3, 3)``.
        dim: ``(n1, n2, n3)``.

    Returns:
        ``n1 n2 n3`` q points (fractional, parent reciprocal basis), Gamma
        first, sorted by length.
    """
    from types import SimpleNamespace

    from .phonon_subgroups import commensurate_qpoints as _commensurate

    lattice = np.asarray(parent_lattice, dtype=float)
    holder = SimpleNamespace(
        supercell=SimpleNamespace(cell=np.diag(np.asarray(dim, dtype=float)) @ lattice),
        primitive=SimpleNamespace(cell=lattice),
    )
    return _commensurate(holder)


def parse_qpoint_tokens(tokens) -> list:
    """Split ``--qpoint`` tokens into labels and coordinate triples.

    A token that reads as a number (``0.5``, ``1/2``, ``-1/2``) starts a
    triple of coordinates; any other token is a label (``R``, ``GM``) or the
    word ``all``.

    Args:
        tokens: The tokens as given on the command line.

    Returns:
        A list of strings (labels) and three-element lists of strings.

    Raises:
        ValueError: A coordinate triple is incomplete.
    """
    from fractions import Fraction

    def is_number(text: str) -> bool:
        try:
            Fraction(text.strip())
        except (ValueError, ZeroDivisionError):
            return False
        return True

    items: list = []
    index = 0
    tokens = [str(token) for token in tokens]
    while index < len(tokens):
        token = tokens[index]
        if is_number(token):
            triple = tokens[index : index + 3]
            if len(triple) < 3 or not all(is_number(value) for value in triple):
                raise ValueError(
                    "--qpoint coordinates come in threes (e.g. --qpoint 1/2 1/2 1/2), got: "
                    + " ".join(value.strip() for value in triple)
                )
            items.append([value.strip() for value in triple])
            index += 3
        else:
            items.append(token.strip())
            index += 1
    return items


def _circular_mean_positions(frames: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per-atom, per-axis circular mean of wrapped fractional coordinates."""
    angles = 2.0 * np.pi * frames
    mean_angle = np.arctan2(np.sin(angles).mean(axis=0), np.cos(angles).mean(axis=0))
    return np.mod(mean_angle / (2.0 * np.pi), 1.0)


def _effective_samples(values: NDArray[np.float64]) -> float:
    """Frames divided by the integrated autocorrelation time of a series."""
    n_frames = len(values)
    centred = values - values.mean()
    variance = float(np.mean(centred**2))
    if n_frames < 2 or variance <= 1e-24:
        return float("nan")
    size = 1 << (2 * n_frames - 1).bit_length()
    spectrum = np.fft.rfft(centred, size)
    autocorrelation = np.fft.irfft(spectrum * np.conj(spectrum), size)[:n_frames]
    autocorrelation /= variance * np.arange(n_frames, 0, -1)
    negative = np.nonzero(autocorrelation <= 0.0)[0]
    stop = int(negative[0]) if len(negative) else n_frames
    tau = max(1.0, 1.0 + 2.0 * float(np.sum(autocorrelation[1:stop])))
    return n_frames / tau


def _component_statistics(values: NDArray[np.float64], n_eff: float | None = None,
                          n_frames: int | None = None) -> ComponentStatistics:
    """Statistics of one series (or of pooled series of ``n_frames`` frames)."""
    n_frames = len(values) if n_frames is None else n_frames
    mean = float(np.mean(values))
    rms = float(np.sqrt(np.mean(values**2)))
    mean_abs = float(np.mean(np.abs(values)))
    ratio = mean_abs / rms if rms > 1e-12 else float("nan")
    variance = float(np.var(values))
    if variance > 1e-24 and n_frames > 1:
        kurtosis = float(np.mean((values - mean) ** 4) / variance**2 - 3.0)
    else:
        kurtosis = float("nan")
    center_ratio = float("nan")
    if rms <= 1e-12:
        shape = "zero"
    elif n_frames < MIN_FRAMES_FOR_SHAPE:
        shape = "n/a"
        kurtosis = float("nan")
    else:
        limit = float(np.max(np.abs(values)))
        n_bins = int(np.clip(np.sqrt(n_frames), 11, 41))
        n_bins += 1 - n_bins % 2  # odd: one bin centred on Q = 0
        counts, _ = np.histogram(values, bins=n_bins, range=(-limit, limit))
        center_ratio = float(counts[n_bins // 2] / counts.max())
        if center_ratio >= DOUBLE_PEAK_RATIO:
            shape = "single-peaked"
        else:
            negative = np.mean(values < 0.0)
            if min(negative, 1.0 - negative) >= 0.1:
                shape = "double-peaked (hint)"
            else:
                shape = "off-center"
    if n_eff is None:
        n_eff = _effective_samples(values)
    return ComponentStatistics(mean, rms, mean_abs, ratio, kurtosis, center_ratio, shape, n_eff)


def _gaussian_norm_ratio(n_components: int) -> float:
    """<|x|>/sqrt<|x|^2> of an isotropic Gaussian in n dimensions."""
    n = n_components
    return math.sqrt(2.0 / n) * math.exp(math.lgamma((n + 1) / 2.0) - math.lgamma(n / 2.0))


def _check_parent_primitive(dataset) -> None:
    identity = np.eye(3, dtype=int)
    n_points = int(sum(np.array_equal(rotation, identity) for rotation in dataset.rotations))
    if n_points > 1:
        raise ValueError(
            f"the parent cell holds {n_points} lattice points; --project needs the "
            "primitive cell of the parent structure."
        )


def _star_arms(vibrations, qpoint) -> list[list[float]]:
    """The arms of the star of q (:func:`crystod.star_of_k.compute_star`).

    The first arm is q as given; the others are as
    :func:`crystod.star_of_k.compute_star` returns them (coordinates in
    ``(-1/2, 1/2]``) and, when -k is an arm too, the one of k and -k whose first nonzero
    coordinate is positive comes first (the other is then reported as its
    -q partner, which has the same real patterns).
    """
    from .star_of_k import compute_star

    arms = [np.asarray(arm["kpoint"], dtype=float)
            for arm in compute_star(vibrations.rotations, vibrations.translations, qpoint)]
    keys = [_q_key(arm) for arm in arms]

    def positive(arm) -> bool:
        nonzero = arm[np.abs(arm) > 1e-8]
        return len(nonzero) == 0 or nonzero[0] > 0

    ordered: list[int] = []
    for index, arm in enumerate(arms):
        if index in ordered:
            continue
        partner_key = _q_key(-arm)
        partner = keys.index(partner_key) if partner_key in keys else None
        if partner is None or partner == index or partner in ordered:
            ordered.append(index)
        elif index == 0 or positive(arm):
            ordered += [index, partner]
        else:
            ordered += [partner, index]
    key = _q_key(qpoint)
    return [
        [float(value) for value in qpoint] if keys[index] == key
        else [float(value) + 0.0 for value in arms[index]]
        for index in ordered
    ]


def _resolve_qpoints(vibrations, request, dim, all_q, star: bool = False):
    """The q points to project as ``(name, qpoint)``, plus notes.

    With ``star`` every q of the request (labels, coordinates or the default
    special points) is replaced by all arms of its star; ``all`` already
    holds every arm. An arm that is not commensurate with ``dim`` (possible
    for a ``dim`` of lower symmetry than the parent) is left out with a note.
    """
    from .vibration_modes import format_fraction_vector

    dim_array = np.asarray(dim, dtype=float)
    dim_text = " ".join(str(n) for n in dim)
    notes: list[str] = []

    def commensurate(qpoint) -> bool:
        scaled = np.asarray(qpoint, dtype=float) * dim_array
        return bool(np.allclose(scaled, np.rint(scaled), atol=1e-6))

    def expand(name, qpoint):
        if not star:
            return [(name, qpoint)]
        result = []
        for arm in _star_arms(vibrations, qpoint):
            arm_name = name if _q_key(arm) == _q_key(qpoint) else vibrations.name_qpoint(arm)
            if commensurate(arm):
                result.append((arm_name, arm))
            else:
                notes.append(f"{arm_name} {format_fraction_vector(arm)}: an arm of the star "
                             f"not commensurate with --dim {dim_text}")
        return result

    if request is None:
        from .vibration_modes import _in_seekpath_order

        special = _in_seekpath_order(vibrations, vibrations.get_special_qpoints())
        resolved = [item for name, q in special.items() if commensurate(q)
                    for item in expand(name, [float(v) for v in q])]
        if not resolved:
            raise ValueError(
                f"no tabulated special point is commensurate with --dim {dim_text}; "
                "give --qpoint."
            )
        return resolved, notes
    if isinstance(request, str):
        request = [request]
    resolved = []
    for item in request:
        if isinstance(item, str) and item.strip().lower() == "all":
            seen: set[tuple[float, ...]] = set()
            for q in all_q:
                if _q_key(q) in seen:
                    continue  # the -q partner of a listed q: same real patterns
                seen.update({_q_key(q), _q_key(-np.asarray(q, dtype=float))})
                resolved.append((vibrations.name_qpoint(list(q)), [float(v) for v in q]))
            continue
        tokens = [item] if isinstance(item, str) else [str(value) for value in item]
        name, qpoint = vibrations.resolve_special_qpoint(tokens)
        if not commensurate(qpoint):
            raise ValueError(
                f"q point {name} {format_fraction_vector(qpoint)} is not commensurate "
                f"with --dim {dim_text}."
            )
        resolved.extend(expand(name, [float(value) for value in qpoint]))
    return resolved, notes


def _match_sites(parent_cell, dim, symbols, mean_frac, supercell_lattice):
    """Match every trajectory atom to an ideal site (j, R)."""
    parent_frac = np.asarray(parent_cell.scaled_positions, dtype=float)
    parent_symbols = list(parent_cell.symbols)
    parent_lattice = np.asarray(parent_cell.cell, dtype=float)
    dim_array = np.asarray(dim, dtype=float)
    cells = np.array(list(product(range(dim[0]), range(dim[1]), range(dim[2]))), dtype=float)
    ideal = ((cells[:, None, :] + parent_frac[None, :, :]) / dim_array).reshape(-1, 3)
    site_symbols = parent_symbols * len(cells)
    threshold = MATCH_FRACTION * float(np.min(np.linalg.norm(parent_lattice, axis=1)))

    site_of_atom = np.full(len(symbols), -1, dtype=int)
    deviations = np.zeros((len(symbols), 3))
    for element in sorted(set(symbols)):
        atoms = np.array([i for i, s in enumerate(symbols) if s == element])
        sites = np.array([i for i, s in enumerate(site_symbols) if s == element])
        delta = mean_frac[atoms][:, None, :] - ideal[sites][None, :, :]
        delta -= np.rint(delta)
        cart = (delta * dim_array) @ parent_lattice
        distance = np.linalg.norm(cart, axis=2)
        nearest = np.argmin(distance, axis=1)
        best = distance[np.arange(len(atoms)), nearest]
        worst = int(np.argmax(best))
        if best[worst] > threshold:
            raise ValueError(
                f"atom {atoms[worst] + 1} ({element}) of the trajectory lies "
                f"{best[worst]:.3f} A from the nearest ideal {element} site of the parent "
                f"supercell (limit {threshold:.3f} A = {MATCH_FRACTION} of the shortest parent "
                "lattice vector); check the parent cell, its origin and --dim."
            )
        counts = Counter(nearest.tolist())
        doubled = [site for site, count in counts.items() if count > 1]
        if doubled:
            claim = [int(atoms[i]) + 1 for i in np.where(nearest == doubled[0])[0]]
            raise ValueError(
                f"atoms {', '.join(str(a) for a in claim)} ({element}) of the trajectory "
                "claim the same ideal site of the parent supercell; the time-averaged "
                "structure does not map one-to-one onto the parent."
            )
        site_of_atom[atoms] = sites[nearest]
        deviations[atoms] = cart[np.arange(len(atoms)), nearest]
    uniform = deviations.mean(axis=0)
    remaining = np.linalg.norm(deviations - uniform, axis=1)
    worst_atom = int(np.argmax(remaining))
    # the shift is reported in the trajectory's Cartesian frame
    shift_frac = uniform @ np.linalg.inv(parent_lattice) / dim_array
    uniform_md = shift_frac @ supercell_lattice
    return ideal, SiteMatching(
        site_of_atom=site_of_atom,
        uniform_shift=uniform_md,
        max_deviation=float(remaining[worst_atom]),
        max_deviation_atom=worst_atom + 1,
    )


def _real_patterns(vibrations, qpoint, rows, dim):
    """Unit-norm real supercell patterns of the rows of one mode space."""
    patterns, components = [], []
    invariant = _is_time_reversal_invariant(qpoint)
    for index, row in enumerate(rows, start=1):
        variants = [(str(index), row)] if invariant else [
            (f"{index}c", row), (f"{index}s", 1j * row)
        ]
        for name, vector in variants:
            _, displacement, _, _ = vibrations.get_supercell_displacements(
                list(qpoint), np.asarray(vector), tuple(dim)
            )
            flat = displacement.reshape(-1)
            norm = float(np.linalg.norm(flat))
            if norm < 1e-10:
                raise RuntimeError(f"a zero pattern at q = {qpoint} (row {index}).")
            patterns.append(flat / norm)
            components.append(name)
    return patterns, components


def _parent_from_phonon(phonon):
    """phonopy's primitive cell as a ``PhonopyAtoms`` (symbols, masses)."""
    from phonopy.structure.atoms import PhonopyAtoms

    primitive = phonon.primitive
    return PhonopyAtoms(
        symbols=list(primitive.symbols),
        cell=np.array(primitive.cell, dtype=float),
        scaled_positions=np.array(primitive.scaled_positions, dtype=float),
        masses=np.array(primitive.masses, dtype=float),
    )


def _check_same_cell(cell, reference, tolerance: float) -> None:
    """Raise ``ValueError`` unless ``cell`` is ``reference`` (atoms in any order)."""
    lattice = np.asarray(cell.cell, dtype=float)
    reference_lattice = np.asarray(reference.cell, dtype=float)
    problem = None
    if len(cell.symbols) != len(reference.symbols):
        problem = f"{len(cell.symbols)} atoms against {len(reference.symbols)}"
    elif np.max(np.abs(lattice - reference_lattice)) > tolerance:
        problem = "the lattice vectors differ"
    else:
        positions = np.asarray(reference.scaled_positions, dtype=float)
        for symbol, position in zip(cell.symbols, np.asarray(cell.scaled_positions, dtype=float)):
            delta = position[None, :] - positions
            delta -= np.rint(delta)
            distance = np.linalg.norm(delta @ reference_lattice, axis=1)
            same = [i for i, s in enumerate(reference.symbols) if s == symbol]
            if not same or float(np.min(distance[same])) > tolerance:
                problem = f"no {symbol} atom of the yaml at the -c position {position.round(6).tolist()}"
                break
    if problem is not None:
        raise ValueError(
            f"the -c cell is not the primitive cell of the phonopy yaml ({problem}; "
            f"tolerance {tolerance:g} A); omit -c, the yaml defines the parent cell."
        )


def _phonon_modes(phonon, qpoint, symprec: float):
    """Labels, frequencies, eigenvectors and degeneracies of the modes at q.

    The modes of ``crystod-phonon --modulation`` and ``--vector``
    (:class:`crystod.modulation.SymmetryAdaptedModulation`): sorted by
    frequency, mode i (0-based here) is mode i + 1 of those commands.
    """
    from .modulation import SymmetryAdaptedModulation

    modulation = SymmetryAdaptedModulation(phonon=phonon, qpoint=list(qpoint), symprec=symprec)
    frequencies = [float(info["frequency_THz"]) for info in modulation.mode_info]
    degeneracies = [int(info["degeneracy"]) for info in modulation.mode_info]
    return (list(modulation.get_mode_labels()), frequencies,
            [np.asarray(vector) for vector in modulation.eigenvectors], degeneracies)


def _effective_frequency(mean_square: float, temperature: float) -> float:
    """``sqrt(k_B T / <a^2>) / (2 pi)`` in THz (``<a^2>`` in amu A^2)."""
    if mean_square <= 1e-24:
        return float("nan")
    return math.sqrt(KB_AMU_A2_PS2 * temperature / mean_square) / (2.0 * math.pi)


# ---------------------------------------------------------------- engine
def project_trajectory(
    parent_cell,
    dim,
    symbols,
    lattices,
    frames,
    qpoints=None,
    irreps=None,
    mean_lattice: bool = False,
    steps=None,
    symprec: float = 1e-5,
    phonon=None,
    temperature: float | None = None,
    star: bool = False,
) -> TrajectoryProjection:
    """Project the displacements of an MD trajectory onto parent modes.

    The engine of ``crystod-md --project``. The trajectory atoms are matched
    to the ideal supercell of the parent (site ``(j, R)`` at
    ``(x_j + R) / dim``) by element and the nearest periodic image of their
    time-averaged positions, so the atom order of the trajectory does not
    matter. The displacement of every atom in every frame is taken relative
    to its ideal site (wrapped to the nearest image, converted to Cartesian
    with the frame's own cell or, with ``mean_lattice``, the time-averaged
    cell), and the mass-weighted centre-of-mass displacement of the frame is
    removed (drift). Each frame is then projected onto the unit-norm real patterns of
    the symmetry-adapted spaces of
    ``SymmetryOnlyVibrations.get_symmetry_adapted_spaces`` at every q:
    ``Q = p_hat . u / sqrt(N)``, the displacement norm per parent primitive
    cell (at a time-reversal invariant q, the amplitude of
    ``crystod-phonon --modulation``).

    With ``phonon`` the basis is the phonon eigenvectors instead (see the
    module docstring): the parent cell is ``phonon.primitive``, the
    trajectory is taken in mass-weighted coordinates and every column is the
    normal coordinate ``a(t)`` of one mode (and, with ``temperature``, its
    effective frequency).

    Args:
        parent_cell: The parent primitive cell, a ``PhonopyAtoms`` or the path
            of a POSCAR file. Its Cartesian axes are those of the patterns;
            the trajectory cell must have the same orientation. May be
            ``None`` with ``phonon``; if both are given, the cell must be
            ``phonon.primitive`` (lattice vectors and positions within the
            larger of ``symprec`` and 1e-4 A, atoms in any order).
        dim: Diagonal supercell multiplicities ``(n1, n2, n3)``.
        symbols: Element of every trajectory atom (``read_xdatcar``).
        lattices: One ``(3, 3)`` supercell lattice per frame (rows).
        frames: Fractional coordinates, shape ``(n_frames, n_atoms, 3)``.
        qpoints: ``None`` for every tabulated special point commensurate
            with ``dim`` (in the order ``crystod-phonon --vibration`` lists
            them), ``"all"`` for every commensurate q, or a list of labels
            (``"R"``) and coordinate triples (``[0.5, 0.5, 0]``, strings such
            as ``"1/2"`` allowed); ``"all"`` may also be an item of the list.
        irreps: ISO-IR irrep names to keep (``["R4+", "M3+"]``; the
            dimension suffix is optional); ``None`` keeps all.
        mean_lattice: Use the time-averaged cell for every frame.
        steps: Step number of every frame (default ``0 .. n_frames - 1``).
        symprec: Symmetry tolerance of the parent.
        phonon: A ``phonopy.Phonopy`` object with force constants (e.g.
            ``phonopy.load("phonopy_params.yaml", is_nac=False)``): project on
            its symmetry-adapted eigenvectors at every q (the modes of
            ``crystod-phonon --modulation``).
        temperature: Temperature of the run in K (``phonon`` only): adds the
            effective frequency of every mode.
        star: Replace every requested q (labels, coordinates, or the default
            special points) by all arms of its star.

    Returns:
        A :class:`TrajectoryProjection`.

    Raises:
        ValueError: The composition is not ``n1 n2 n3`` times the parent's,
            the cells disagree, an atom matches no site or two atoms one site,
            the parent is not primitive, a q is unknown or not commensurate,
            the irrep filter keeps nothing, ``parent_cell`` is not the
            primitive cell of ``phonon``, the phonopy object has no force
            constants, or ``temperature`` is given without ``phonon`` or is
            not positive.

    Example:
        >>> from crystod import md
        >>> symbols, lattices, frames = md.read_xdatcar("XDATCAR")
        >>> result = md.project_trajectory(
        ...     "221_PPOSCAR_ScF3", (4, 4, 4), symbols, lattices, frames[1000:],
        ...     qpoints=["R", "M"], irreps=["R4+", "M3+"])
        >>> [p.columns for p in result.projections]
        [['R4+(3)_3_1', 'R4+(3)_3_2', 'R4+(3)_3_3'], ['M3+(1)_2_1']]
    """
    from .phonon_vector import reduced_formula
    from .vibration_modes import SymmetryOnlyVibrations

    if isinstance(parent_cell, (str, os.PathLike)):
        from .star_of_k import read_poscar_or_exit

        parent_cell = read_poscar_or_exit(str(parent_cell))
    if temperature is not None:
        if phonon is None:
            raise ValueError("a temperature needs the phonon-eigenvector basis (force constants).")
        if not temperature > 0:
            raise ValueError(f"the temperature must be positive, got {temperature:g} K.")
    if phonon is not None:
        if phonon.dynamical_matrix is None:
            raise ValueError(
                "the phonopy object carries no force constants (a phonopy_disp.yaml? "
                "the eigenvector basis needs phonopy_params.yaml)."
            )
        yaml_parent = _parent_from_phonon(phonon)
        if parent_cell is not None:
            _check_same_cell(parent_cell, yaml_parent, max(symprec, CELL_AGREEMENT))
        parent_cell = yaml_parent
    elif parent_cell is None:
        raise ValueError("a parent cell (or a phonopy object) is required.")
    dim = tuple(int(n) for n in dim)
    if len(dim) != 3 or any(n <= 0 for n in dim):
        raise ValueError("dim must be three positive integers.")
    n_cells = dim[0] * dim[1] * dim[2]
    frames = np.asarray(frames, dtype=float)
    if frames.ndim != 3 or frames.shape[2] != 3 or len(frames) == 0:
        raise ValueError("frames must have the shape (n_frames, n_atoms, 3).")
    n_frames, n_atoms, _ = frames.shape
    symbols = list(symbols)
    if len(symbols) != n_atoms:
        raise ValueError(f"{len(symbols)} symbols for {n_atoms} atoms per frame.")
    lattices = np.asarray(lattices, dtype=float)
    if lattices.ndim == 2:
        lattices = np.repeat(lattices[None], n_frames, axis=0)
    if lattices.shape != (n_frames, 3, 3):
        raise ValueError("lattices must hold one (3, 3) cell per frame.")
    steps = np.arange(n_frames) if steps is None else np.asarray(steps, dtype=int)
    if steps.shape != (n_frames,):
        raise ValueError(f"{len(steps)} step numbers for {n_frames} frames.")

    parent_symbols = list(parent_cell.symbols)
    expected = Counter({element: count * n_cells for element, count in Counter(parent_symbols).items()})
    found = Counter(symbols)
    if found != expected:
        def text(counter):
            return " ".join(f"{element}{counter[element]}" for element in sorted(counter))

        raise ValueError(
            f"the trajectory holds {text(found)}, but --dim {' '.join(map(str, dim))} times "
            f"the parent cell holds {text(expected)}."
        )

    parent_lattice = np.asarray(parent_cell.cell, dtype=float)
    mean_cell = lattices.mean(axis=0)
    per_parent = mean_cell / np.asarray(dim, dtype=float)[:, None]
    lengths = np.linalg.norm(parent_lattice, axis=1)
    if np.any(np.linalg.norm(per_parent - parent_lattice, axis=1) > LATTICE_FRACTION * lengths):
        def rows(matrix):
            return "; ".join("(" + ", ".join(f"{v:.4f}" for v in row) + ")" for row in matrix)

        raise ValueError(
            "the trajectory cell divided by --dim, "
            f"{rows(per_parent)}, differs from the parent cell, {rows(parent_lattice)}, "
            f"by more than {LATTICE_FRACTION:.0%} of a lattice vector (same orientation needed)."
        )

    vibrations = SymmetryOnlyVibrations(parent_cell, symprec=symprec, standardize=False)
    _check_parent_primitive(vibrations.spglib_dataset)

    # --- site matching and displacements
    mean_frac = _circular_mean_positions(frames)
    ideal, matching = _match_sites(parent_cell, dim, symbols, mean_frac, mean_cell)
    order = np.empty(n_atoms, dtype=int)
    order[matching.site_of_atom] = np.arange(n_atoms)  # site -> atom
    delta = frames[:, order, :] - ideal[None, :, :]
    delta -= np.rint(delta)
    if mean_lattice:
        displacements = delta @ mean_cell
    else:
        displacements = np.einsum("tai,tij->taj", delta, lattices)
    # drift: the mass-weighted centre of mass of every frame (the quantity MD
    # codes conserve); the plain mean would also remove part of polar optical
    # motion, whose atoms move against each other with unequal masses
    site_masses = np.tile(np.asarray(parent_cell.masses, dtype=float), n_cells)
    com = np.einsum("a,taj->tj", site_masses, displacements) / site_masses.sum()
    displacements -= com[:, None, :]
    com_rms = float(np.sqrt(np.mean(np.sum(com**2, axis=1))))
    if phonon is None:
        u = displacements.reshape(n_frames, -1)
        total_msd = float(np.mean(np.sum(u**2, axis=1)) / n_cells)
    else:
        # mass-weighted coordinates w = sqrt(m) u, over the whole supercell
        u = (displacements * np.sqrt(site_masses)[None, :, None]).reshape(n_frames, -1)
        total_msd = float(np.mean(np.sum(u**2, axis=1)))
    if np.ptp(lattices, axis=0).max() < 1e-10:
        lattice_mode = "fixed"
    else:
        lattice_mode = "mean" if mean_lattice else "frame"

    # --- q points
    all_q = commensurate_qpoints(parent_lattice, dim)
    requested, star_notes = _resolve_qpoints(vibrations, qpoints, dim, all_q, star=star)
    filter_names = None
    if irreps is not None:
        filter_names = [_irrep_name(str(label)).upper() for label in irreps]

    names_seen = Counter(name for name, _ in requested)
    projections: list[QProjection] = []
    skipped: list[str] = list(star_notes)
    covered: set[tuple[float, ...]] = set()
    seen_labels: set[str] = set()
    from .vibration_modes import format_fraction_vector

    examined: set[tuple[float, ...]] = set()
    for name, qpoint in requested:
        key = _q_key(qpoint)
        if key in examined:
            continue  # the same q given twice
        examined.add(key)
        if key in covered:
            skipped.append(
                f"{name} {format_fraction_vector(qpoint)}: the -q partner of a projected q "
                "(same real patterns)"
            )
            continue
        columns_irrep, columns_space, columns_comp, patterns = [], [], [], []
        columns_frequency: list[float] = []
        if phonon is None:
            _, _, labels = vibrations.describe_mode_spaces(qpoint)
            adapted = vibrations.get_symmetry_adapted_spaces(qpoint)
            seen_labels.update(_irrep_name(label) for label in labels)
            for space_number, (label, rows) in enumerate(zip(labels, adapted), start=1):
                if filter_names is not None and _irrep_name(label).upper() not in filter_names:
                    continue
                space_patterns, space_components = _real_patterns(vibrations, qpoint, rows, dim)
                patterns.extend(space_patterns)
                columns_irrep.extend([label] * len(space_patterns))
                columns_space.extend([space_number] * len(space_patterns))
                columns_comp.extend(space_components)
        else:
            labels, frequencies, eigenvectors, degeneracies = _phonon_modes(
                phonon, qpoint, symprec)
            seen_labels.update(_irrep_name(label) for label in labels)
            mode = 0
            while mode < len(labels):
                # one degenerate level: its partners are consecutive modes
                level = range(mode, min(mode + max(1, degeneracies[mode]), len(labels)))
                mode = level.stop
                if filter_names is not None and \
                        _irrep_name(labels[level[0]]).upper() not in filter_names:
                    continue
                for index in level:
                    # the eigenvector is mass-weighted: its pattern lives in w = sqrt(m) u
                    mode_patterns, mode_components = _real_patterns(
                        vibrations, qpoint, [eigenvectors[index]], dim)
                    for pattern, component in zip(mode_patterns, mode_components):
                        patterns.append(pattern)
                        columns_irrep.append(labels[index])
                        columns_space.append(level[0] + 1)
                        columns_comp.append(f"{index + 1}{component[1:]}")
                        columns_frequency.append(frequencies[index])
        if not patterns:
            skipped.append(f"{name} {format_fraction_vector(qpoint)}: no irrep of the filter")
            continue
        covered.add(key)
        covered.add(_q_key(-np.asarray(qpoint)))
        basis = np.array(patterns)
        # symmetry basis: Q per parent cell; eigenvector basis: the normal
        # coordinate a of the supercell (u holds sqrt(m) u there)
        values = u @ basis.T / (math.sqrt(n_cells) if phonon is None else 1.0)
        statistics = [_component_statistics(values[:, c]) for c in range(values.shape[1])]
        norms = []
        for space_number in dict.fromkeys(columns_space):
            selected = [c for c, s in enumerate(columns_space) if s == space_number]
            if len(selected) < 2:
                continue
            norm_values = np.linalg.norm(values[:, selected], axis=1)
            norm_mean = float(norm_values.mean())
            norm_rms = float(np.sqrt(np.mean(norm_values**2)))
            n_eff = [statistics[c].n_eff for c in selected]
            pooled = _component_statistics(
                values[:, selected].reshape(-1),
                n_eff=float(np.sum(n_eff)) if np.all(np.isfinite(n_eff)) else float("nan"),
                n_frames=n_frames,
            )
            widths = [statistics[c].rms for c in selected]
            if min(widths) <= 1e-12 or max(widths) > POOL_RMS_FACTOR * min(widths):
                # the components are not equivalent in this run (the parent
                # symmetry is not kept on average, or a weakly damped mode was
                # sampled in few phases), so their pooled shape means nothing
                pooled = replace(pooled, ratio=float("nan"), kurtosis=float("nan"),
                                 center_ratio=float("nan"), shape="unequal",
                                 n_eff=float("nan"))
            norms.append(NormStatistics(
                irrep=columns_irrep[selected[0]], space=space_number,
                n_components=len(selected), pooled=pooled, values=norm_values, mean=norm_mean,
                rms=norm_rms, ratio=norm_mean / norm_rms if norm_rms > 1e-12 else float("nan"),
                gaussian_ratio=_gaussian_norm_ratio(len(selected)),
            ))
        tag = name if names_seen[name] == 1 else name + "".join(
            f"_{value:g}" for value in qpoint
        )
        if phonon is None:
            columns = [f"{i}_{s}_{c}" for i, s, c in
                       zip(columns_irrep, columns_space, columns_comp)]
            frequency_list = effective = None
        else:
            columns = [f"{c}_{_irrep_name(i)}" for i, c in zip(columns_irrep, columns_comp)]
            frequency_list = columns_frequency
            effective = None if temperature is None else [
                _effective_frequency(stats.rms**2, temperature) for stats in statistics]
        projections.append(QProjection(
            name=name, tag=tag, qpoint=tuple(float(v) + 0.0 for v in qpoint),
            time_reversal_invariant=_is_time_reversal_invariant(qpoint),
            irreps=columns_irrep, spaces=columns_space, components=columns_comp,
            columns=columns, values=values, statistics=statistics, norms=norms,
            frequencies=frequency_list, effective_frequencies=effective,
        ))

    if filter_names is not None:
        known = {label.upper() for label in seen_labels}
        unknown = [str(label) for label, name in zip(irreps, filter_names) if name not in known]
        if not projections or len(unknown) == len(filter_names):
            raise ValueError(
                f"no irrep named {', '.join(irreps)} at the projected q points "
                f"(irreps there: {', '.join(sorted(seen_labels))})."
            )
        for label in unknown:
            skipped.append(
                f"irrep {label}: at none of the requested q points "
                f"(irreps there: {', '.join(sorted(seen_labels))})"
            )

    projected_msd = float(sum(np.sum(np.mean(p.values**2, axis=0)) for p in projections))
    n_covered = sum(1 for q in all_q if _q_key(q) in covered)
    dataset = vibrations.spglib_dataset
    return TrajectoryProjection(
        formula=reduced_formula(parent_symbols),
        space_group=str(dataset["international"]),
        space_group_number=int(dataset["number"]),
        dim=dim,
        n_atoms=n_atoms,
        steps=steps,
        lattice_mode=lattice_mode,
        matching=matching,
        projections=projections,
        skipped=skipped,
        irrep_filter=None if irreps is None else [str(label) for label in irreps],
        projected_msd=projected_msd,
        total_msd=total_msd,
        drift_rms=com_rms,
        n_q_covered=n_covered,
        n_q_commensurate=len(all_q),
        basis="symmetry" if phonon is None else "eigenvector",
        temperature=None if temperature is None else float(temperature),
        star=bool(star),
    )


# ---------------------------------------------------------------- output
def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    widths = [max(len(header[c]), *(len(row[c]) for row in rows)) for c in range(len(header))]
    return ["  ".join(t.ljust(w) for t, w in zip(row, widths)).rstrip()
            for row in [header] + rows]


def _number(value: float, digits: int = 5) -> str:
    if not np.isfinite(value):
        return "-"
    if abs(value) < 0.5 * 10.0 ** (-digits):
        value = 0.0  # no "-0.00000"
    return f"{value + 0.0:.{digits}f}"


def _mode_number(component: str) -> int:
    """``"4"``, ``"4c"`` -> 4."""
    return int(re.match(r"\d+", component).group(0))


def _format_eigenvector_blocks(result: TrajectoryProjection) -> list[str]:
    """Mode tables, pooled levels, sum rule and notes of the eigenvector basis."""
    from .vibration_modes import format_fraction_vector

    n_cells = result.dim[0] * result.dim[1] * result.dim[2]
    thermal = result.temperature is not None
    lines: list[str] = []
    unit = "(A sqrt(amu))"
    header = ["mode", "irrep", "nu_harm (THz)", f"<a> {unit}", f"sqrt<a^2> {unit}",
              "sqrt<A^2> per cell", "ratio", "kurtosis", "n_eff", "shape"]
    if thermal:
        header.append("nu_eff (THz)")
    for projection in result.projections:
        rows = []
        for c, stats in enumerate(projection.statistics):
            row = [
                projection.components[c], projection.irreps[c],
                _number(projection.frequencies[c], 4), _number(stats.mean),
                _number(stats.rms), _number(stats.rms / math.sqrt(n_cells)),
                _number(stats.ratio, 3), _number(stats.kurtosis, 2), _number(stats.n_eff, 0),
                stats.shape,
            ]
            if thermal:
                row.append(_number(projection.effective_frequencies[c], 4))
            rows.append(row)
        coords = format_fraction_vector(projection.qpoint)
        lines += ["", f"* Mode projection at {projection.name} {coords}, "
                  "phonon-eigenvector basis *"]
        lines += _table(header, rows)
        if projection.norms:
            norm_rows = []
            for n in projection.norms:
                numbers = [_mode_number(projection.components[c])
                           for c, space in enumerate(projection.spaces) if space == n.space]
                modes = (str(numbers[0]) if min(numbers) == max(numbers)
                         else f"{min(numbers)}-{max(numbers)}")
                row = [modes, n.irrep, str(n.n_components),
                       _number(projection.frequencies[projection.spaces.index(n.space)], 4),
                       _number(n.pooled.rms), _number(n.pooled.ratio, 3),
                       _number(n.pooled.kurtosis, 2), _number(n.pooled.n_eff, 0),
                       n.pooled.shape, _number(n.mean), _number(n.ratio, 3),
                       _number(n.gaussian_ratio, 3)]
                if thermal:
                    row.append(_number(_effective_frequency(n.pooled.rms**2,
                                                            result.temperature), 4))
                norm_rows.append(row)
            norm_header = ["modes", "irrep", "comps", "nu_harm (THz)", f"sqrt<a^2> {unit}",
                           "ratio", "kurtosis", "n_eff", "shape", f"<|a|> {unit}",
                           "|a| ratio", "Gaussian"]
            if thermal:
                norm_header.append("nu_eff (THz)")
            lines += ["", f"* Pooled degenerate levels and norm |a| at {projection.name} "
                      f"{coords} *"]
            lines += _table(norm_header, norm_rows)

    fraction = result.projected_msd / result.total_msd if result.total_msd > 0 else float("nan")
    lines += [
        "", "* Sum rule *",
        f"  {'sum of <a^2> over the projected modes':42s} : {result.projected_msd:.6f} amu A^2",
        f"  {'<sum m |u|^2> (supercell, drift removed)':42s} : {result.total_msd:.6f} amu A^2",
        f"  fraction covered                           : {100.0 * fraction:.2f} %",
        f"  q points covered                           : {result.n_q_covered} of "
        f"{result.n_q_commensurate} commensurate (a -q partner counts with its q)",
    ]
    if result.complete and result.total_msd > 0:
        difference = abs(result.projected_msd - result.total_msd) / result.total_msd
        lines.append(f"  relative difference (complete basis)       : {difference:.1e}")

    lines += [
        "", "* Notes *",
        "  a: normal coordinate of the supercell, a = p_hat . w with w = sqrt(m) u",
        "     (mass-weighted displacements, A sqrt(amu)) and p_hat the unit-norm real",
        "     pattern of the eigenvector; A = a/sqrt(N) per parent cell. Modes are",
        "     numbered as in crystod-phonon --modulation and --vector (1-based, sorted",
        "     by the harmonic frequency nu_harm; imaginary frequencies as negative).",
        "  --modulation --amplitude A0 (v: its pattern e exp(2 pi i q.x)/sqrt(m), unit",
        "     norm per cell) gives A = A0 |sqrt(m) v| at a time-reversal invariant q,",
        "     elsewhere a norm of A0 |sqrt(m) v|/sqrt(2) over the c/s columns of the mode.",
    ]
    if thermal:
        lines += [
            "  nu_eff = sqrt(k_B T/<a^2>)/(2 pi): the frequency of the harmonic oscillator",
            "     with the same <a^2> at T (classical equipartition); meaningful for an",
            "     equilibrated classical run. nu_eff above nu_harm: the mode is stiffer at",
            "     T than in the harmonic model (for a soft mode, anharmonic stabilization).",
        ]
    lines += [
        "  ratio = <|a|>/sqrt<a^2>: 0.798 for a Gaussian about a = 0 (single well,",
        "     displacive), 1 for two states +-a0 (order-disorder).",
        "  shape: histogram density at a = 0 against its maximum (< 0.5 with weight on",
        "     both sides: double-peaked); a hint from the statistics, not a test.",
        "  n_eff: frames / autocorrelation time; ratio and kurtosis scatter ~ 1/sqrt(n_eff).",
        "  Pooled: the modes of one degenerate level as one sample (equivalent in the",
        "     parent symmetry); unequal: their rms values differ by more than a factor",
        f"     {POOL_RMS_FACTOR:g} (no pooled shape); |a| ratio = <|a|>/sqrt<|a|^2>, Gaussian =",
        "     its value for an isotropic Gaussian in that many components.",
        "  Modes 1c/1s: the real patterns of a complex eigenvector e and of i e (q not",
        "     time-reversal invariant; -q gives the same patterns).",
    ]
    return lines


def format_projection(result: TrajectoryProjection, source: dict | None = None,
                      files=()) -> str:
    """The terminal report of ``crystod-md --project``.

    Args:
        result: The :class:`TrajectoryProjection`.
        source: Optional input description for the ``* Input *`` block, keys
            ``parent``, ``trajectory``, ``frames`` (a text line).
        files: Paths of the files written (listed in ``* Output files *``).

    Returns:
        The report, ``* ... *`` blocks separated by blank lines.
    """
    from .vibration_modes import format_fraction_vector

    source = source or {}
    n_cells = result.dim[0] * result.dim[1] * result.dim[2]
    lattice_text = {
        "fixed": "fixed cell",
        "frame": "each frame's own cell (fractional -> Cartesian)",
        "mean": "time-averaged cell for every frame (--mean-lattice)",
    }[result.lattice_mode]
    lines = ["", "* Input *"]
    if "parent" in source:
        lines.append(f"  Parent cell     : {source['parent']}")
    lines.append(f"  Space group     : {result.space_group} (No. {result.space_group_number})")
    lines.append(f"  Supercell (dim) : {' '.join(map(str, result.dim))} "
                 f"({n_cells} parent cells, {result.n_atoms} atoms)")
    if "trajectory" in source:
        lines.append(f"  Trajectory      : {source['trajectory']}")
    lines.append(f"  Frames used     : {source.get('frames', len(result.steps))}")
    lines.append(f"  Lattice         : {lattice_text}")
    if result.basis == "eigenvector":
        if "yaml" in source:
            lines.append(f"  Force constants : {source['yaml']} (no NAC)")
        lines.append("  Basis           : phonon eigenvectors at each q (mass-weighted)")
        if result.temperature is not None:
            lines.append(f"  Temperature     : {result.temperature:g} K (effective frequencies)")
    if result.star:
        lines.append("  Star            : every arm of each requested q (--star)")

    match = result.matching
    shift = ", ".join(f"{value:+.4f}" for value in match.uniform_shift)
    lines += [
        "", "* Site matching *",
        f"  Sites matched   : {result.n_atoms} (one-to-one, by element and nearest image",
        "                    of the time-averaged position)",
        f"  Uniform shift   : ({shift}) A (mean offset of the averaged structure)",
        f"  Largest offset  : {match.max_deviation:.4f} A (atom {match.max_deviation_atom}, "
        "uniform shift removed)",
        "  Drift           : the mass-weighted centre-of-mass displacement is removed",
        f"                    from every frame (rms {result.drift_rms:.4f} A)",
    ]

    lines += ["", "* Q points *"]
    for projection in result.projections:
        lines.append(f"  {projection.name:8s} {format_fraction_vector(projection.qpoint)}")
    for note in result.skipped:
        lines.append(f"  skipped: {note}")

    if result.basis == "eigenvector":
        lines += _format_eigenvector_blocks(result)
        if files:
            lines += ["", "* Output files *"]
            lines += [f"  {path}" for path in files]
        return "\n".join(lines)

    header = ["irrep", "space", "component", "<Q> (A)", "sqrt<Q^2> (A)", "<|Q|> (A)",
              "ratio", "kurtosis", "n_eff", "shape"]
    for projection in result.projections:
        rows = []
        for c, stats in enumerate(projection.statistics):
            rows.append([
                projection.irreps[c], str(projection.spaces[c]), projection.components[c],
                _number(stats.mean), _number(stats.rms), _number(stats.mean_abs),
                _number(stats.ratio, 3), _number(stats.kurtosis, 2), _number(stats.n_eff, 0),
                stats.shape,
            ])
        coords = format_fraction_vector(projection.qpoint)
        lines += ["", f"* Mode projection at {projection.name} {coords} *"]
        lines += _table(header, rows)
        if projection.norms:
            norm_rows = [[n.irrep, str(n.space), str(n.n_components), _number(n.pooled.rms),
                          _number(n.pooled.ratio, 3), _number(n.pooled.kurtosis, 2),
                          _number(n.pooled.n_eff, 0), n.pooled.shape, _number(n.mean),
                          _number(n.ratio, 3), _number(n.gaussian_ratio, 3)]
                         for n in projection.norms]
            lines += ["", f"* Pooled components and norm |Q| at {projection.name} {coords} *"]
            lines += _table(["irrep", "space", "comps", "sqrt<Q^2> (A)", "ratio", "kurtosis",
                             "n_eff", "shape", "<|Q|> (A)", "|Q| ratio", "Gaussian"],
                            norm_rows)

    fraction = result.projected_msd / result.total_msd if result.total_msd > 0 else float("nan")
    lines += [
        "", "* Sum rule *",
        f"  sum of <Q^2> over the projected components : {result.projected_msd:.6f} A^2",
        f"  <|u|^2> / N (all atoms, drift removed)     : {result.total_msd:.6f} A^2",
        f"  fraction covered                           : {100.0 * fraction:.2f} %",
        f"  q points covered                           : {result.n_q_covered} of "
        f"{result.n_q_commensurate} commensurate (a -q partner counts with its q)",
    ]
    if result.complete and result.total_msd > 0:
        difference = abs(result.projected_msd - result.total_msd) / result.total_msd
        lines.append(f"  relative difference (complete basis)       : {difference:.1e}")

    lines += [
        "", "* Notes *",
        "  Q: displacement norm per parent primitive cell carried by the pattern; at a",
        "     time-reversal invariant q it equals the crystod-phonon --modulation amplitude,",
        "     elsewhere --amplitude A gives a norm of A/sqrt(2) over its c/s components.",
        "  AMPLIMODES amplitude of an irrep = sqrt(size) x the norm of its Q over all",
        "     components and arms (Q sqrt(size) for a single component).",
        "  ratio = <|Q|>/sqrt<Q^2>: 0.798 for a Gaussian about Q = 0 (single well,",
        "     displacive), 1 for two states +-Q0 (order-disorder).",
        "  shape: histogram density at Q = 0 against its maximum (< 0.5 with weight on",
        "     both sides: double-peaked); a hint from the statistics, not a test.",
        "  n_eff: frames / autocorrelation time; ratio and kurtosis scatter ~ 1/sqrt(n_eff).",
        "  Pooled: all components of a mode space as one sample (equivalent components",
        "     in the parent symmetry); unequal: component rms values differ by more than",
        f"     a factor {POOL_RMS_FACTOR:g} (no pooled shape); |Q| ratio = <|Q|>/sqrt<|Q|^2>,",
        "     Gaussian = its value for an isotropic Gaussian in that many components.",
        "  Components 1c/1s: the real patterns of a complex row e and of i e (q not",
        "     time-reversal invariant; -q gives the same patterns).",
    ]
    if files:
        lines += ["", "* Output files *"]
        lines += [f"  {path}" for path in files]
    return "\n".join(lines)


def write_csv(projection: QProjection, steps, path: str) -> None:
    """Write ``Q(t)`` of one q point: ``step`` and one column per component."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(",".join(["step", *projection.columns]) + "\n")
        for step, row in zip(steps, projection.values):
            handle.write(f"{int(step)}," + ",".join(f"{round(value, 8) + 0.0:.8f}" for value in row) + "\n")


def write_npz(result: TrajectoryProjection, path: str) -> None:
    """Write the steps, q points, column names and ``Q(t)`` arrays.

    In the phonon-eigenvector basis the arrays are ``a_<q>`` (normal
    coordinates of the supercell, A sqrt(amu)), with ``frequencies_<q>``
    (harmonic, THz), ``effective_frequencies_<q>`` with a temperature, and
    ``basis``.
    """
    data = {
        "steps": np.asarray(result.steps, dtype=int),
        "qpoints": np.array([p.qpoint for p in result.projections], dtype=float),
        "qpoint_names": np.array([p.tag for p in result.projections]),
        "dim": np.array(result.dim, dtype=int),
    }
    eigenvector = result.basis == "eigenvector"
    if eigenvector:
        data["basis"] = np.array(result.basis)
    for projection in result.projections:
        data[f"{'a' if eigenvector else 'Q'}_{projection.tag}"] = projection.values
        data[f"labels_{projection.tag}"] = np.array(projection.columns)
        if eigenvector:
            data[f"frequencies_{projection.tag}"] = np.array(projection.frequencies)
            if projection.effective_frequencies is not None:
                data[f"effective_frequencies_{projection.tag}"] = np.array(
                    projection.effective_frequencies)
    np.savez(path, **data)


_SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                  "#4a3aa7", "#e34948")


def write_histograms(result: TrajectoryProjection, path: str) -> None:
    """Histograms of every component, one panel per irrep occurrence.

    Each component is drawn as a step histogram (probability density) with a
    dashed Gaussian of the same mean and variance as the reference; twelve
    panels per page of a multi-page PDF.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    from .vibration_modes import format_fraction_vector

    eigenvector = result.basis == "eigenvector"
    panels = []
    for projection in result.projections:
        for space in dict.fromkeys(projection.spaces):
            columns = [c for c, s in enumerate(projection.spaces) if s == space]
            panels.append((projection, space, columns))
    per_page = 12
    with PdfPages(path) as pdf:
        for start in range(0, len(panels), per_page):
            page = panels[start : start + per_page]
            n_cols = min(3, len(page))
            n_rows = (len(page) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.7 * n_cols, 2.9 * n_rows + 0.4),
                                     squeeze=False)
            for ax in axes.flat[len(page):]:
                ax.axis("off")
            for ax, (projection, space, columns) in zip(axes.flat, page):
                limit = max(float(np.max(np.abs(projection.values[:, columns]))), 1e-6)
                grid = np.linspace(-limit, limit, 300)
                for slot, c in enumerate(columns):
                    color = _SERIES_COLORS[slot % len(_SERIES_COLORS)]
                    values = projection.values[:, c]
                    n_bins = int(np.clip(np.sqrt(len(values)), 11, 41)) | 1
                    ax.hist(values, bins=n_bins, range=(-limit, limit), density=True,
                            histtype="step", linewidth=1.5, color=color,
                            label=projection.components[c])
                    sigma = float(np.std(values))
                    if sigma > 1e-12:
                        mean = float(np.mean(values))
                        gauss = np.exp(-0.5 * ((grid - mean) / sigma) ** 2) / (
                            sigma * math.sqrt(2 * math.pi))
                        ax.plot(grid, gauss, linestyle="--", linewidth=1.0, color=color)
                if eigenvector:
                    numbers = [_mode_number(projection.components[c]) for c in columns]
                    modes = (f"mode {numbers[0]}" if min(numbers) == max(numbers)
                             else f"modes {min(numbers)}-{max(numbers)}")
                    ax.set_title(f"{projection.name} {format_fraction_vector(projection.qpoint)}  "
                                 f"{projection.irreps[columns[0]]}, {modes}", fontsize=9)
                    ax.set_xlabel("a (A sqrt(amu))", fontsize=8)
                    ax.set_ylabel("density (1/(A sqrt(amu)))", fontsize=8)
                else:
                    ax.set_title(f"{projection.name} {format_fraction_vector(projection.qpoint)}  "
                                 f"{projection.irreps[columns[0]]}, space {space}", fontsize=9)
                    ax.set_xlabel("Q (A)", fontsize=8)
                    ax.set_ylabel("density (1/A)", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(alpha=0.25, linewidth=0.5)
                if len(columns) > 1:
                    ax.legend(fontsize=7, frameon=False,
                              title="mode" if eigenvector else "component", title_fontsize=7)
            title = ("normal coordinates (phonon-eigenvector basis)" if eigenvector
                     else "mode projections")
            fig.suptitle(f"{result.formula} ({result.space_group}): {title}\n"
                         "dashed: Gaussian of the same mean and variance", fontsize=9)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def _configuration_numbers(path: str) -> list[int]:
    """The ``configuration=`` numbers of an XDATCAR, in file order."""
    numbers = []
    pattern = re.compile(r"configuration\s*=\s*(\d+)", re.IGNORECASE)
    with open(path) as handle:
        for line in handle:
            if line.lstrip()[:1] in ("D", "d") and line.lstrip().lower().startswith("direct"):
                found = pattern.search(line)
                numbers.append(int(found.group(1)) if found else -1)
    return numbers


def main(argv: list[str] | None = None) -> None:
    """``crystod-md --project`` (dispatched by ``crystod/cli/md.py``)."""
    import argparse

    parser = argparse.ArgumentParser(description="Mode projection of an MD trajectory.")
    parser.add_argument("--cell", default=None)
    parser.add_argument("--yaml", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--star", action="store_true")
    parser.add_argument("--dim", required=True)
    parser.add_argument("--xdatcar", default="XDATCAR")
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--end-step", type=int, default=None)
    parser.add_argument("--qpoint", nargs="+", default=None)
    parser.add_argument("--irrep", nargs="+", default=None)
    parser.add_argument("--mean-lattice", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--npz", action="store_true")
    parser.add_argument("--no-histogram", action="store_true")
    parser.add_argument("--symprec", type=float, default=1e-5)
    if argv is None:
        argv = sys.argv[1:]
    args = parser.parse_args(list(argv))

    from .star_of_k import read_poscar_or_exit
    from .xdatcar_adp import read_xdatcar

    if args.cell is None and args.yaml is None:
        raise SystemExit("ERROR: --project needs -c/--cell or --yaml.")
    if args.temperature is not None and args.yaml is None:
        raise SystemExit("ERROR: --temperature needs --yaml (the effective frequencies are "
                         "compared with the harmonic phonon frequencies).")
    parent = read_poscar_or_exit(args.cell) if args.cell is not None else None
    phonon = None
    if args.yaml is not None:
        from .modulation import load_phonon

        try:
            phonon, _, _ = load_phonon(yaml_path=args.yaml)
        except SystemExit:
            raise
        except Exception as exc:  # phonopy reports a bad file in many ways
            message = " ".join(str(exc).split())
            raise SystemExit(f"ERROR: phonopy could not read '{args.yaml}': {message}") from None
    dim = tuple(int(value) for value in args.dim.split())
    try:
        symbols, lattices, frames = read_xdatcar(args.xdatcar)
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from None
    n_total = len(frames)
    if args.start_step < 0 or args.start_step >= n_total:
        raise SystemExit(
            f"ERROR: --start-step {args.start_step} is outside the trajectory "
            f"(steps 0 .. {n_total - 1})."
        )
    end = n_total if args.end_step is None else min(args.end_step + 1, n_total)
    if end <= args.start_step:
        raise SystemExit(
            f"ERROR: --end-step {args.end_step} must not be smaller than "
            f"--start-step {args.start_step}."
        )
    configurations = _configuration_numbers(args.xdatcar)[:n_total]
    if len(configurations) != n_total or min(configurations, default=-1) < 0:
        configurations = list(range(n_total))
        config_text = ""
    else:
        config_text = (f"; configuration {configurations[args.start_step]} .. "
                       f"{configurations[end - 1]}")
    steps = np.array(configurations[args.start_step:end], dtype=int)

    qpoints = None
    if args.qpoint:
        try:
            qpoints = parse_qpoint_tokens(args.qpoint)
        except ValueError as exc:
            raise SystemExit(f"ERROR: {exc}") from None
    try:
        result = project_trajectory(
            parent, dim, symbols, lattices[args.start_step:end], frames[args.start_step:end],
            qpoints=qpoints, irreps=args.irrep, mean_lattice=args.mean_lattice,
            steps=steps, symprec=args.symprec, phonon=phonon,
            temperature=args.temperature, star=args.star,
        )
    except (ValueError, RuntimeError) as exc:
        message = " ".join(str(exc).split())
        raise SystemExit(f"ERROR: {message}") from None

    default_prefix = f"MDPROJ_{result.formula}" + ("_phonon" if phonon is not None else "")
    prefix = args.output or default_prefix
    directory = os.path.dirname(prefix)
    if directory:
        os.makedirs(directory, exist_ok=True)
    files = []
    for projection in result.projections:
        csv_path = f"{prefix}_{projection.tag}.csv"
        write_csv(projection, result.steps, csv_path)
        files.append(csv_path)
    if not args.no_histogram:
        pdf_path = f"{prefix}_hist.pdf"
        write_histograms(result, pdf_path)
        files.append(pdf_path)
    if args.npz:
        npz_path = f"{prefix}.npz"
        write_npz(result, npz_path)
        files.append(npz_path)
    if args.yaml is None:
        parent_text = args.cell
    elif args.cell is None:
        parent_text = f"primitive cell of {args.yaml}"
    else:
        parent_text = f"{args.cell} (the primitive cell of {args.yaml})"
    source = {
        "parent": parent_text,
        "trajectory": f"{args.xdatcar} ({n_total} frame{'s' if n_total != 1 else ''})",
        "frames": f"{end - args.start_step} (step {args.start_step} .. {end - 1}{config_text})",
    }
    if args.yaml is not None:
        source["yaml"] = args.yaml
    print(format_projection(result, source, files))
