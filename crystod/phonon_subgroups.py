"""Isotropy subgroups from phonon modes (Python API).

This module is the programmatic bridge between phonon irrep labeling
(``phonon_irreps``) and isotropy-subgroup enumeration
(``isotropy_subgroup``), designed for downstream packages that follow
imaginary phonon modes to lower-symmetry structures (e.g. the
``macer phonopy tree`` structure search):

    from crystod.phonon import scan_imaginary_modes
    results = scan_imaginary_modes(phonon)         # a phonopy.Phonopy object
    for result in results:
        print(result.mode.frequency, result.mode.labels)
        for sub in result.subgroups:
            print(sub.irrep, sub.direction, sub.number, sub.symbol)

The phonopy object should be built with ``primitive_matrix="auto"`` so
that instabilities appear at their true q points of the primitive cell
(e.g. R of a cubic perovskite) instead of being folded onto the supercell
Gamma point, where mode labeling is not possible.

Four layers are provided:

- ``isotropy_subgroups(space_group, irrep)`` -- pure group theory, no
  phonopy object needed: enumerate the isotropy subgroups of a
  space-group irrep (the API form of ``crystod-group --supergroup``).
- ``label_phonon_modes(phonon, qpoint)`` -- label the phonon modes of a
  live phonopy object at one q point with ISO-IR irrep labels,
  encapsulating the table/star-arm boilerplate of ``phonon_irreps``.
- ``imaginary_mode_subgroups(phonon, qpoint)`` -- combine the two: for
  every imaginary (negative-frequency) degenerate level at q, return the
  irrep label and all isotropy subgroups with their order-parameter
  directions, conventional bases, and origins.
- ``scan_imaginary_modes(phonon)`` -- run ``imaginary_mode_subgroups``
  over every q point commensurate with the phonon supercell (the q set a
  supercell calculation can resolve), deduplicating star arms.

All band/mode indices in this module are 1-based, following the CrystOD
convention used everywhere else in the package and its outputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

_IRREP_DIM_SUFFIX = re.compile(r"\((\d+)\)$")


# ---------------------------------------------------------------------------
# isotropy subgroups of a space-group irrep (no phonopy needed)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IsotropySubgroup:
    """One isotropy subgroup of a space-group irrep.

    ``basis`` (rows, in parent *conventional* units) and ``origin`` describe
    the conventional cell of the subgroup in the parent convention, exactly
    as printed by ``crystod-group --supergroup --order-parameter``; they are
    ``None`` when the setting could not be standardized.
    """

    irrep: str                  # e.g. "R4+" ("X3-+X2+" for coupled input)
    direction: str              # order-parameter direction, e.g. "(a,0,0)"
    label: str                  # full label, e.g. "R4+(a,0,0)"
    number: int                 # space-group number of the subgroup
    symbol: str                 # international short symbol, e.g. "I4/mcm"
    size: int                   # primitive-cell multiplication vs the parent
    index: int                  # index of the subgroup in the parent
    n_free: int                 # number of free order-parameter components
    basis: np.ndarray | None = field(default=None, compare=False)
    origin: np.ndarray | None = field(default=None, compare=False)

    def __str__(self) -> str:
        return f"{self.label} -> {self.symbol} (No. {self.number}), size {self.size}, index {self.index}"


def _subgroup_from_members(analyzer, irrep_name, label, direction, n_free,
                           members, with_settings):
    info, size, index, B, rotations, translations, lattice = (
        analyzer.subgroup_of(members)
    )
    basis = origin = None
    if with_settings:
        try:
            setting = analyzer.conventional_setting(
                B, rotations, translations, lattice, info
            )
        except Exception:
            setting = None
        if setting is not None:
            basis, origin = setting
    return IsotropySubgroup(
        irrep=irrep_name,
        direction=direction,
        label=label,
        number=int(info.number),
        symbol=str(info.international_short),
        size=int(size),
        index=int(index),
        n_free=int(n_free),
        basis=basis,
        origin=origin,
    )


def isotropy_subgroups(
    space_group,
    irrep,
    order_parameter=None,
    *,
    with_settings: bool = True,
):
    """Isotropy subgroups of a space-group irrep, as data.

    Programmatic counterpart of ``crystod-group --supergroup``.

    Raises ``ValueError`` for an unknown space group, an irrep that is not
    tabulated for it (the labels of symmetry lines and planes, e.g. ``DT5``,
    have no isotropy subgroups in the tables), or an invalid order parameter.

    Parameters
    ----------
    space_group : str | int
        International short symbol (e.g. ``"Pm-3m"``) or number (e.g. 221).
    irrep : str | list[str]
        ISO-IR irrep label (e.g. ``"R4+"``); a list of labels enumerates
        the subgroups of the coupled order parameters.
    order_parameter : list[str] | None
        If given (e.g. ``["a", "0", "0"]``), resolve only this direction
        and return a single-element list.
    with_settings : bool
        Also compute the conventional ``basis``/``origin`` of each subgroup
        in the parent convention (slightly slower; on by default).

    Returns
    -------
    list[IsotropySubgroup]
        Sorted like the ``--supergroup`` table (free components, index,
        subgroup number).
    """
    # The implementation modules report bad input the way a command line
    # wants it -- by raising SystemExit -- which would tear down a program
    # that merely called this function. Translate it into a normal exception.
    try:
        return _isotropy_subgroups(
            space_group, irrep, order_parameter, with_settings=with_settings
        )
    except SystemExit as exc:
        message = " ".join(str(exc).split())
        raise ValueError(message.removeprefix("ERROR: ") or "invalid input") from None


def _isotropy_subgroups(space_group, irrep, order_parameter, *, with_settings):
    from .isotropy_subgroup import (
        CoupledRepresentation,
        IsotropyAnalyzer,
        _projector,
    )

    if isinstance(space_group, (int, np.integer)):
        space_group = str(int(space_group))
    if not irrep or (not isinstance(irrep, str) and not list(irrep)):
        raise ValueError("at least one irrep label is required")
    analyzer = IsotropyAnalyzer(space_group, irrep)
    representation = analyzer.representation
    coupled = isinstance(representation, CoupledRepresentation)
    irrep_name = representation.name if coupled else representation.label

    if order_parameter is not None:
        components = [str(c) for c in order_parameter]
        _reject_composite_components(components)
        eta = analyzer.resolve_direction(components)
        members = [
            (i, t)
            for i, t, matrix in analyzer.elements
            if np.allclose(matrix @ eta, eta, atol=1e-6)
        ]
        # same direction string as crystod-group --supergroup: components are
        # grouped arm by arm ("," inside an arm, ";" between arms)
        if coupled:
            chunks, start = [], 0
            for part in representation.parts:
                piece = components[start: start + part.dimension]
                chunks.append(f"{part.label}({_arm_join(piece, part.arm_chunks)})")
                start += part.dimension
            direction = " ".join(chunks)
            label = direction
        else:
            direction = "(" + _arm_join(components, representation.arm_chunks) + ")"
            label = f"{irrep_name}{direction}"
        # free parameters are the distinct symbols the direction actually
        # carries, read exactly as resolve_direction reads them
        n_free = len({
            symbol for symbol in map(_free_symbol, components) if symbol
        })
        return [
            _subgroup_from_members(
                analyzer, irrep_name, label, direction,
                n_free, members, with_settings,
            )
        ]

    results = []
    for projector, members in analyzer.enumerate_directions():
        label, generic = analyzer.direction_label(projector)
        direction = label
        if not coupled:
            label = representation.label + label
        else:
            # skip the single-irrep strata of a coupled representation,
            # exactly as crystod-group --supergroup does
            bounds = np.cumsum([0] + list(representation.dims))
            if any(
                np.linalg.norm(generic[bounds[j]: bounds[j + 1]]) < 1e-8
                for j in range(len(representation.dims))
            ):
                continue
        exact_members = [
            (i, t)
            for i, t, matrix in analyzer.elements
            if np.allclose(matrix @ generic, generic, atol=1e-6)
        ]
        n_free = _orth_rank(projector)
        results.append(
            _subgroup_from_members(
                analyzer, irrep_name, label, direction, n_free,
                exact_members, with_settings,
            )
        )
    results.sort(key=lambda s: (s.n_free, s.index, s.number))
    return results


def _arm_join(components, arm_chunks) -> str:
    """Group order-parameter components arm by arm, as the tables print them."""
    arms, start = [], 0
    for size in arm_chunks:
        arms.append(",".join(components[start: start + size]))
        start += size
    return ";".join(arms)


def _reject_composite_components(components) -> None:
    """Refuse order-parameter tokens the direction resolver cannot honour.

    ``resolve_direction`` understands a number or a bare parameter name, and
    silently treats anything else -- ``0.282a``, ``0.888a+0.46b`` -- as a new
    independent parameter, which resolves to a *different*, more generic
    direction and therefore a different (wrong) subgroup. Those composite
    directions do occur in the enumerated tables of hexagonal and trigonal
    irreps; they can be read from :func:`isotropy_subgroups` but not fed back
    into it, so they are rejected here instead of quietly giving a wrong
    answer.
    """
    from fractions import Fraction

    for component in components:
        token = str(component).strip()
        if token.startswith("-"):
            token = token[1:]
        if token in ("", "0", "0.0") or (len(token) == 1 and token.isalpha()):
            continue
        try:
            Fraction(token)
        except ValueError:
            raise SystemExit(
                f"ERROR: order-parameter component {component!r} is not a plain "
                "number or parameter name. Composite directions such as "
                "'0.282a' appear in the enumerated table but cannot be resolved "
                "from components; take that entry from isotropy_subgroups() "
                "without --order-parameter instead."
            ) from None


def _free_symbol(token: str) -> str | None:
    """The free-parameter name a component carries, or None if it is fixed.

    Mirrors ``IsotropyAnalyzer.resolve_direction``: one leading minus sign
    belongs to the amplitude rather than the name, empty and zero components
    are fixed, and anything ``Fraction`` accepts is a number.
    """
    from fractions import Fraction

    token = str(token).strip()
    if token.startswith("-"):
        token = token[1:]
    if token in ("", "0", "0.0"):
        return None
    try:
        Fraction(token)
        return None
    except ValueError:
        return token


def _orth_rank(projector: np.ndarray) -> int:
    return int(round(float(np.real(np.trace(projector)))))


# ---------------------------------------------------------------------------
# phonon mode labeling from a phonopy object
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhononMode:
    """One degenerate phonon level at a q point.

    ``band_indices`` are 1-based (CrystOD convention).  ``labels`` holds the
    ISO-IR irrep label(s) of the level, without the dimension suffix that
    ``phonon_irreps.yaml`` appends (i.e. ``"R4+"``, not ``"R4+(3)"``); it is
    empty when the level could not be labeled.
    """

    band_indices: tuple[int, ...]
    frequency: float            # THz; negative = imaginary
    labels: tuple[str, ...]
    qpoint: tuple[float, float, float]
    qpoint_label: str | None    # tabulated name of the star (e.g. "R"), if any
    representative_q: tuple[float, float, float]

    @property
    def degeneracy(self) -> int:
        return len(self.band_indices)

    @property
    def is_imaginary(self) -> bool:
        return self.frequency < 0.0

    def __str__(self) -> str:
        bands = ",".join(str(b) for b in self.band_indices)
        label = "+".join(self.labels) if self.labels else "?"
        return f"modes {bands}: {self.frequency:.4f} THz  {label}"


def _strip_dim_suffix(label: str) -> str:
    return _IRREP_DIM_SUFFIX.sub("", label)


def label_phonon_modes(
    phonon,
    qpoint=(0.0, 0.0, 0.0),
    *,
    degeneracy_tolerance: float = 1e-4,
):
    """ISO-IR irrep labels of the phonon modes of ``phonon`` at ``qpoint``.

    ``phonon`` is a live ``phonopy.Phonopy`` object with force constants
    available (e.g. from ``phonopy.load``).  The q point is given in
    fractional coordinates of the primitive reciprocal basis; if it is a
    non-representative arm of a star, it is mapped onto the tabulated arm
    automatically (the spectra of star arms coincide band by band).

    Returns a list of :class:`PhononMode`, one per degenerate level,
    ordered by band index.  Raises ``RuntimeError`` if the space-group
    tables cannot label this q point at all.
    """
    from phonopy.structure.cells import get_primitive_matrix_by_centring

    from .irreptables_compat import load_irreptables
    from .phonon_irreps import (
        find_star_representative,
        get_irrep_labels,
        get_irt_special_points,
    )
    from .runtime_compat import get_symmetry_dataset

    IrrepTable, _ = load_irreptables()

    qpoint = tuple(float(x) for x in qpoint)
    dataset = get_symmetry_dataset(phonon.symmetry)
    try:
        irt_table = IrrepTable(dataset["number"], spinor=False)
        prim_mat = get_primitive_matrix_by_centring(dataset["international"][0])

        label_q = list(qpoint)
        qpoint_label = None
        q_names, q_list = get_irt_special_points(irt_table, prim_mat)
        rotations = get_symmetry_dataset(phonon.primitive_symmetry)["rotations"]
        representative = find_star_representative(
            qpoint, rotations, q_names, q_list
        )
        if representative is not None:
            qpoint_label, label_q = representative
        labels, band_indices, frequencies = get_irrep_labels(
            q=label_q,
            phonon=phonon,
            irt_table=irt_table,
            prim_mat=prim_mat,
            degeneracy_tolerance=degeneracy_tolerance,
        )
    except (Exception, SystemExit) as exc:
        # SystemExit: the table machinery reports some failures the way a
        # command line wants them; a library caller gets an exception
        reason = " ".join(str(exc).split()) or type(exc).__name__
        raise RuntimeError(
            f"could not label the phonon modes at q = {list(qpoint)}: {reason}"
        ) from None

    modes = []
    for label, indices in zip(labels, band_indices):
        clean = tuple(_strip_dim_suffix(text) for text in label) if label else ()
        modes.append(
            PhononMode(
                band_indices=tuple(int(i) + 1 for i in indices),
                frequency=float(frequencies[indices[0]]),
                labels=clean,
                qpoint=qpoint,
                qpoint_label=qpoint_label,
                representative_q=tuple(float(x) for x in label_q),
            )
        )
    modes.sort(key=lambda m: m.band_indices[0])
    return modes


# ---------------------------------------------------------------------------
# imaginary modes -> isotropy subgroups (the macer `phonopy tree` use case)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImaginaryModeResult:
    """Isotropy subgroups reachable from one imaginary phonon level."""

    mode: PhononMode
    space_group: str            # parent international short symbol
    space_group_number: int
    subgroups: tuple[IsotropySubgroup, ...]
    errors: dict = field(default_factory=dict, compare=False)
    # errors maps an irrep label to the reason its subgroup enumeration
    # failed (e.g. a k point outside the tabulated set); empty on success


def imaginary_mode_subgroups(
    phonon,
    qpoint=(0.0, 0.0, 0.0),
    *,
    threshold: float = -0.1,
    degeneracy_tolerance: float = 1e-4,
    with_settings: bool = True,
):
    """Isotropy subgroups of every imaginary phonon level at ``qpoint``.

    For each degenerate level with frequency below ``threshold`` (THz;
    the -0.1 default matches the instability criterion of structure-search
    workflows), the level is labeled with its ISO-IR irrep and the
    isotropy subgroups of that irrep are enumerated, covering *all*
    order-parameter directions -- including the ones a single frozen-in
    modulation would miss.

    Returns a list of :class:`ImaginaryModeResult`.  Levels that could not
    be labeled yield a result with empty ``subgroups`` and an entry in
    ``errors``; a q point that cannot be labeled at all raises
    ``RuntimeError`` (from :func:`label_phonon_modes`).
    """
    from .runtime_compat import get_symmetry_dataset

    dataset = get_symmetry_dataset(phonon.symmetry)
    number = int(dataset["number"])
    symbol = str(dataset["international"])

    modes = label_phonon_modes(
        phonon, qpoint, degeneracy_tolerance=degeneracy_tolerance
    )

    # label -> (subgroups, error message); the error is cached too, so a label
    # that fails once still explains itself on every later level that carries it
    cache: dict[str, tuple[list[IsotropySubgroup], str | None]] = {}
    results = []
    for mode in modes:
        if mode.frequency >= threshold:
            continue
        subgroups: list[IsotropySubgroup] = []
        errors: dict[str, str] = {}
        if not mode.labels:
            errors["?"] = "the level could not be labeled with an irrep"
        for label in mode.labels:
            if label not in cache:
                try:
                    cache[label] = (
                        isotropy_subgroups(number, label, with_settings=with_settings),
                        None,
                    )
                except (Exception, SystemExit) as exc:
                    # SystemExit: the subgroup machinery reports some failures
                    # that way; KeyboardInterrupt must still propagate
                    reason = " ".join(str(exc).split()) or type(exc).__name__
                    cache[label] = ([], reason)
            found, reason = cache[label]
            subgroups.extend(found)
            if reason is not None:
                errors[label] = reason
        results.append(
            ImaginaryModeResult(
                mode=mode,
                space_group=symbol,
                space_group_number=number,
                subgroups=tuple(subgroups),
                errors=errors,
            )
        )
    return results


def commensurate_qpoints(phonon):
    """q points of the primitive cell commensurate with the phonon supercell.

    These are exactly the q points the supercell calculation resolves
    (i.e. the ones that fold onto Gamma of the supercell), as fractional
    coordinates in the primitive reciprocal basis; Gamma comes first.
    """
    supercell_in_primitive = (
        np.asarray(phonon.supercell.cell)
        @ np.linalg.inv(np.asarray(phonon.primitive.cell))
    )
    S = np.rint(supercell_in_primitive).astype(np.int64)
    if not np.allclose(supercell_in_primitive, S, atol=1e-6):
        raise ValueError(
            "the phonon supercell is not an integer multiple of the "
            "primitive cell; cannot enumerate commensurate q points"
        )
    n_classes = abs(int(round(np.linalg.det(S))))
    inv_t = np.linalg.inv(S).T
    from itertools import product as _product

    from .operations import snap_qpoint

    qpoints: list[tuple[float, float, float]] = []
    search = max(2, n_classes)
    # the commensurate q of an N-fold supercell have denominator up to N, so
    # the snapping limit has to cover it (the 48 default would rewrite the q
    # of a supercell longer than 48 cells into an unrelated fraction)
    max_denominator = max(48, n_classes)
    for m in _product(range(-search, search + 1), repeat=3):
        frac = (np.array(m, dtype=float) @ inv_t) % 1.0
        frac = np.where(frac > 1.0 - 1e-8, 0.0, frac)
        frac = tuple(float(x) for x in snap_qpoint(frac, max_denominator))
        if frac not in qpoints:
            qpoints.append(frac)
        if len(qpoints) == n_classes:
            break
    if len(qpoints) != n_classes:
        raise ValueError(
            f"found {len(qpoints)} commensurate q points, expected {n_classes}"
        )
    qpoints.sort(key=lambda q: (np.linalg.norm(q), q))
    return qpoints


def _star_key(qpoint, rotations) -> tuple:
    """Canonical member of the star of q, for deduplication.

    Works for any q, tabulated or not: the whole orbit under the primitive
    rotations (k' = k R, modulo reciprocal-lattice translations) is generated
    and its smallest member returned.
    """
    q = np.asarray(qpoint, dtype=float)
    images = set()
    for rotation in rotations:
        image = np.mod(q @ rotation, 1.0)
        image = np.where(image > 1.0 - 1e-6, 0.0, image)
        images.add(tuple(np.round(image, 6) + 0.0))  # +0.0 normalizes -0.0
    return min(images) if images else tuple(np.round(np.mod(q, 1.0), 6))


def scan_imaginary_modes(
    phonon,
    qpoints=None,
    *,
    threshold: float = -0.1,
    degeneracy_tolerance: float = 1e-4,
    with_settings: bool = True,
):
    """Isotropy subgroups of all imaginary modes at the resolvable q points.

    ``qpoints`` defaults to :func:`commensurate_qpoints` of the phonon
    supercell.  Star arms are deduplicated -- all arms of a star carry the
    same levels -- and the results are sorted most-unstable first.  A q
    point whose modes cannot be labeled is skipped with a warning rather
    than aborting the scan.
    """
    import warnings

    from .runtime_compat import get_symmetry_dataset

    if qpoints is None:
        qpoints = commensurate_qpoints(phonon)

    rotations = get_symmetry_dataset(phonon.primitive_symmetry)["rotations"]

    results = []
    seen_stars: set = set()
    for qpoint in qpoints:
        try:
            star = _star_key(qpoint, rotations)
            if star in seen_stars:
                continue
            found = imaginary_mode_subgroups(
                phonon,
                qpoint,
                threshold=threshold,
                degeneracy_tolerance=degeneracy_tolerance,
                with_settings=with_settings,
            )
        except RuntimeError as exc:
            warnings.warn(str(exc), stacklevel=2)
            continue
        seen_stars.add(star)
        results.extend(found)
    results.sort(key=lambda r: r.mode.frequency)
    return results


# ---------------------------------------------------------------------------
# command line (crystod-phonon --subgroup)
# ---------------------------------------------------------------------------

def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Isotropy subgroups reachable from the imaginary phonon modes."
    )
    parser.add_argument("--poscar", default="POSCAR", help="unit-cell file.")
    parser.add_argument("--dim", default=None, help='supercell dimension, e.g. "4 4 4".')
    parser.add_argument(
        "--readfc", action="store_true",
        help="read FORCE_CONSTANTS instead of FORCE_SETS.",
    )
    parser.add_argument(
        "--yaml", default=None,
        help="phonopy_params.yaml holding cell, supercell and force constants.",
    )
    parser.add_argument(
        "--qpoint", nargs="+", default=None,
        help="q point (label such as R, or three coordinates); "
        "without it every q point commensurate with the supercell is scanned.",
    )
    parser.add_argument(
        "--threshold", type=float, default=-0.1,
        help="frequency below which a mode counts as imaginary (THz).",
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-3,
        help="degeneracy tolerance of the irrep labeling (THz), as in --irreps.",
    )
    parser.add_argument(
        "--modulate", action="store_true",
        help="also generate the distorted structure of every order-parameter "
        "direction (the --modulation step, run automatically).",
    )
    parser.add_argument(
        "--amplitude", type=float, default=0.3,
        help="modulation amplitude in Angstroms for --modulate (default: 0.3).",
    )
    return parser


def _generate_structures_for(
    phonon, result, amplitude, symprec=1e-5, source="", seen=None
) -> None:
    """Write one distorted structure per order-parameter direction and report."""
    from .modulation import generate_direction_structures
    from .runtime_compat import get_symmetry_dataset
    from .star_of_k import compute_star

    mode = result.mode
    dataset = get_symmetry_dataset(phonon.primitive_symmetry)
    arms = compute_star(dataset["rotations"], dataset["translations"], list(mode.qpoint))
    targets = [
        {
            "label": sub.label,
            "number": sub.number,
            "symbol": sub.symbol,
            "size": sub.size,
            "index": sub.index,
        }
        for sub in result.subgroups
    ]
    if not targets:
        return

    q_tag = mode.qpoint_label or "q_" + "_".join(f"{x:g}" for x in mode.qpoint)
    prefix = f"MPOSCAR_{q_tag}"
    if seen is not None:
        # two levels at one q can carry the same irrep, and their direction
        # lists are then identical: the second would overwrite the first
        repeats = seen.get((q_tag, "+".join(mode.labels)), 0)
        seen[(q_tag, "+".join(mode.labels))] = repeats + 1
        if repeats:
            prefix = f"{prefix}_mode{mode.band_indices[0]}"
    try:
        generated, missing = generate_direction_structures(
            phonon,
            [arm["kpoint"] for arm in arms],
            [index - 1 for index in mode.band_indices],
            targets,
            amplitude=amplitude,
            symprec=symprec,
            prefix=prefix,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"  note: no structures generated: {' '.join(str(exc).split())}")
        print()
        return

    if generated:
        print(f"* Distorted structures (amplitude {amplitude} A) *")
        for entry in generated:
            reproduce = " ".join(
                f"--qpoint{i} " + " ".join(f"{x:g}" for x in q)
                + f" --mode{i} " + " ".join(str(m) for m in modes)
                # .10g, not .4g: 0.3 * 0.5478 needs five digits, and a rounded
                # amplitude regenerates a *different* structure
                + f" --amplitude{i} " + " ".join(f"{a:.10g}" for a in amps)
                for i, (q, modes, amps) in enumerate(
                    zip(entry.qpoints, entry.modes, entry.amplitudes), start=1
                )
            )
            if len(entry.qpoints) == 1:
                reproduce = reproduce.replace("--qpoint1", "--qpoint").replace(
                    "--mode1", "--mode"
                ).replace("--amplitude1", "--amplitude")
            print(f"{entry.label:<22} {entry.symbol:<10} -> {entry.path}")
            print(f"    crystod-phonon --modulation {source}{reproduce}")
    for target in missing:
        print(f"  note: no candidate reproduced {target['label']} "
              f"({target['symbol']}); generate it with --modulation by hand.")
    print()


def _modulation_source_options(args) -> str:
    """The input selection of this run, as options for --modulation.

    The printed reproduce command has to name the same input the --subgroup run
    read, or pasting it lands on the POSCAR default and fails.
    """
    if args.yaml:
        return f"--yaml {args.yaml} "
    source = f"-c {args.poscar} "
    if args.dim:
        source += f'--dim "{args.dim}" '
    if args.readfc:
        source += "--readfc "
    return source


def main(argv: list[str] | None = None) -> None:
    import sys

    args = build_parser().parse_args(argv)

    import phonopy

    try:
        if args.yaml:
            phonon = phonopy.load(args.yaml, primitive_matrix="auto")
        else:
            if not args.dim:
                raise SystemExit("ERROR: --subgroup requires --dim (or --yaml).")
            phonon = phonopy.load(
                supercell_matrix=[float(n) for n in args.dim.split()],
                primitive_matrix="auto",
                unitcell_filename=args.poscar,
                force_sets_filename=None if args.readfc else "./FORCE_SETS",
                force_constants_filename="./FORCE_CONSTANTS" if args.readfc else None,
            )
    except FileNotFoundError as exc:
        if exc.filename:
            raise SystemExit(f"ERROR: {exc.filename} not found.")
        raise SystemExit(f"ERROR: {' '.join(str(exc).split())}")
    if phonon.force_constants is None:
        raise SystemExit(
            "ERROR: no force constants available; --subgroup needs FORCE_SETS "
            "(or FORCE_CONSTANTS with --readfc, or --yaml phonopy_params.yaml)."
        )

    from .runtime_compat import get_symmetry_dataset

    dataset = get_symmetry_dataset(phonon.symmetry)

    print()
    print("* Parent structure *")
    print(f"{dataset['international']} (No. {dataset['number']})")
    print()

    if args.qpoint:
        from phonopy.structure.cells import get_primitive_matrix_by_centring

        from .irreptables_compat import load_irreptables
        from .phonon_irreps import get_irt_special_points
        from .phonon_vector import resolve_qpoint

        IrrepTable, _ = load_irreptables()
        try:
            irt_table = IrrepTable(dataset["number"], spinor=False)
            prim_mat = get_primitive_matrix_by_centring(dataset["international"][0])
            q_names, q_list = get_irt_special_points(irt_table, prim_mat)
        except Exception:
            q_names, q_list = [], []
        rotations = get_symmetry_dataset(phonon.primitive_symmetry)["rotations"]
        try:
            _, qpoint = resolve_qpoint(args.qpoint, q_names, q_list, rotations)
        except ValueError as exc:
            raise SystemExit(f"ERROR: {exc}")
        try:
            results = imaginary_mode_subgroups(
                phonon, qpoint,
                threshold=args.threshold,
                degeneracy_tolerance=args.tolerance,
            )
        except RuntimeError as exc:
            raise SystemExit(f"ERROR: {' '.join(str(exc).split())}")
    else:
        results = scan_imaginary_modes(
            phonon,
            threshold=args.threshold,
            degeneracy_tolerance=args.tolerance,
        )

    if not results:
        where = "at this q point" if args.qpoint else "at any commensurate q point"
        print(f"No imaginary mode below {args.threshold} THz {where}.")
        return

    generated_levels: dict[tuple[str, str], int] = {}
    for result in results:
        mode = result.mode
        q_text = ", ".join(f"{x:g}" for x in mode.qpoint)
        name = f" ({mode.qpoint_label})" if mode.qpoint_label else ""
        bands = ", ".join(str(b) for b in mode.band_indices)
        irrep = "+".join(mode.labels) if mode.labels else "unlabeled"
        print(f"* Imaginary mode at q = ({q_text}){name} *")
        print(f"mode {bands}: {mode.frequency:.6f} THz, irrep {irrep}"
              f" (degeneracy {mode.degeneracy})")
        for label, reason in result.errors.items():
            print(f"  note: no subgroups for {label}: {reason}")
        if not result.subgroups:
            print()
            continue
        width = max([20] + [len(s.label) + 1 for s in result.subgroups])
        print()
        print(f"{'irrep':<{width}} {'subgroup':<18} {'size':<5} {'index':<5}")
        for sub in result.subgroups:
            subgroup = f"{sub.number} {sub.symbol}"
            print(f"{sub.label:<{width}} {subgroup:<18} {sub.size:<5} {sub.index:<5}")
        print()
        if args.modulate:
            _generate_structures_for(
                phonon, result, args.amplitude,
                source=_modulation_source_options(args), seen=generated_levels,
            )

    if not args.modulate:
        print("The distortion of each order-parameter direction can be generated with")
        print("crystod-phonon --modulation, or by adding --modulate here.")
    print()
    print("Conventions and validation: ISOSUBGROUP (https://iso.byu.edu):")
    print('H. T. Stokes, S. van Orden and B. J. Campbell, "Tool for Generating')
    print('Isotropy Subgroups of Crystallographic Space Groups",')
    print("J. Appl. Cryst. 49, 1849-1853 (2016).")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
