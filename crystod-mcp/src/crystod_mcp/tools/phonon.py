"""Phonon tools (the ``crystod-phonon`` domain).

Both tools rebuild the phonopy object from the POSCAR text, the supercell
dimension and the FORCE_SETS text in a temporary directory (phonopy reads
the unit cell and the force sets side by side), then call
``crystod.phonon.label_phonon_modes`` / ``scan_imaginary_modes``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from ..utils import (
    blocking_tool,
    format_kpoint,
    markdown_table,
    parse_dim,
    parse_kpoint,
    resolve_kpoint_label,
    special_kpoints,
    temporary_files,
    validate_force_sets,
    validate_poscar,
)


def _load_phonopy(directory: Path, dim: list[int]):
    """``phonopy.load`` of POSCAR + FORCE_SETS in ``directory`` (primitive_matrix auto)."""
    import phonopy

    try:
        phonon = phonopy.load(
            supercell_matrix=dim,
            primitive_matrix="auto",
            unitcell_filename=str(directory / "POSCAR"),
            force_sets_filename=str(directory / "FORCE_SETS"),
            log_level=0,
        )
    except Exception as exc:  # noqa: BLE001 - phonopy reports mismatches in many ways
        reason = " ".join(str(exc).split()) or type(exc).__name__
        raise ValueError(
            f"phonopy could not build the force constants from the POSCAR, dim and "
            f"FORCE_SETS given ({reason}); check that dim is the supercell of the "
            "FORCE_SETS file and that both texts are complete."
        ) from None
    if phonon.force_constants is None:
        raise ValueError(
            "phonopy produced no force constants; the FORCE_SETS text must hold the "
            f"displacements and forces of the {dim[0]}x{dim[1]}x{dim[2]} supercell of the POSCAR."
        )
    # phonopy does not always notice a FORCE_SETS that belongs to another
    # cell or supercell; the atom count in its first line has to match
    declared = int((directory / "FORCE_SETS").read_text(encoding="utf-8").split(None, 1)[0])
    if declared != len(phonon.supercell):
        raise ValueError(
            f"FORCE_SETS is for a {declared}-atom supercell, but the POSCAR "
            f"({len(phonon.unitcell)} atoms) times dim {dim[0]}x{dim[1]}x{dim[2]} gives "
            f"{len(phonon.supercell)} atoms: these force sets do not belong to this cell "
            "and supercell."
        )
    return phonon


def _space_group(phonon) -> tuple[int, str]:
    dataset = phonon.symmetry.dataset
    number = dataset["number"] if isinstance(dataset, dict) else dataset.number
    symbol = dataset["international"] if isinstance(dataset, dict) else dataset.international
    return int(number), str(symbol)


def _mode_rows(modes) -> list[tuple]:
    rows = []
    for mode in modes:
        bands = ",".join(str(b) for b in mode.band_indices)
        frequency = f"{mode.frequency:.4f}"
        if mode.is_imaginary:
            frequency += " (imaginary)"
        label = " + ".join(mode.labels) if mode.labels else "(not labeled)"
        rows.append((bands, frequency, label, mode.degeneracy))
    return rows


@blocking_tool
def crystod_phonon_irreps(
    poscar: str,
    dim: list[int],
    force_sets: str,
    qpoint: str | list[float] | None = None,
) -> str:
    """Label the phonon modes of a crystal at a q point with their ISO-IR irreps (e.g. the imaginary R5- and the R4- modes of cubic SrTiO3 at R) from phonopy force data.

    Needs the unit cell (POSCAR text), the supercell dimension of the force
    calculation and the phonopy FORCE_SETS text; the force constants are
    rebuilt with phonopy and each degenerate level is labeled as in
    ``crystod-phonon --irreps``. Frequencies are in THz; a negative value is
    an imaginary (unstable) mode. Follow up with
    ``crystod_isotropy_subgroups`` for the subgroups an imaginary mode leads
    to, or ``crystod_imaginary_mode_subgroups`` to do both in one call.

    Args:
        poscar: Full text of the VASP POSCAR of the unit cell (VASP 5 format
            with the element line).
        dim: Supercell dimension of the force calculation as three positive
            integers, e.g. ``[4, 4, 4]`` (phonopy's ``--dim``).
        force_sets: Full text of the phonopy ``FORCE_SETS`` file computed for
            that supercell.
        qpoint: A k-point label (``GM``, ``X``, ``M``, ``R``, ...) or three
            fractional coordinates in the primitive reciprocal basis
            (``"1/2 1/2 1/2"``). Omitted, every tabulated special point of the
            space group is labeled.

    Returns:
        Markdown: per q point, a table of levels (band indices, frequency in
        THz, irrep label, degeneracy).
    """
    from crystod import phonon as crystod_phonon

    poscar_text = validate_poscar(poscar)
    supercell = parse_dim(dim)
    force_text = validate_force_sets(force_sets)
    label, coordinates = parse_kpoint(qpoint)

    with temporary_files({"POSCAR": poscar_text, "FORCE_SETS": force_text}) as directory:
        phonon = _load_phonopy(directory, supercell)
        number, symbol = _space_group(phonon)
        _, names, primitive, _ = special_kpoints(str(number))
        if label is not None:
            targets = [(label, resolve_kpoint_label(str(number), label))]
        elif coordinates is not None:
            targets = [(None, coordinates)]
        else:
            targets = [(n, [float(v) for v in k]) for n, k in zip(names, primitive)]

        lines = [
            f"## Phonon irreps of {symbol} (No. {number}), "
            f"{supercell[0]}x{supercell[1]}x{supercell[2]} supercell force constants",
            "",
        ]
        for name, q in targets:
            where = f"{name} {format_kpoint(q)}" if name else f"q = {format_kpoint(q)}"
            lines.append(f"### {where}")
            lines.append("")
            try:
                modes = crystod_phonon.label_phonon_modes(phonon, q)
            except RuntimeError as exc:
                lines.append(f"Could not label this q point: {' '.join(str(exc).split())}")
                lines.append("")
                continue
            if modes and modes[0].qpoint_label and not name:
                lines[-2] = f"### q = {format_kpoint(q)} (star of {modes[0].qpoint_label})"
            lines.append(markdown_table(("Bands", "Frequency (THz)", "Irrep", "Degeneracy"), _mode_rows(modes)))
            lines.append("")
    lines.append(
        "Band indices are 1-based; a negative frequency is an imaginary mode. "
        "Irrep labels are the ISO-IR labels of the little group of q in the "
        "primitive basis of the input cell."
    )
    return "\n".join(lines)


@blocking_tool
def crystod_imaginary_mode_subgroups(
    poscar: str,
    dim: list[int],
    force_sets: str,
    threshold: float = -0.1,
) -> str:
    """Find the space groups an unstable crystal can distort into: every imaginary phonon mode at the q points the supercell resolves is labeled with its irrep and the isotropy subgroups of that irrep are enumerated (e.g. the R5- mode of cubic SrTiO3 gives I4/mcm, R-3c, Imma, ...).

    The structure-search workflow of ``crystod-phonon --subgroup``: the q
    points commensurate with the supercell are scanned (star arms
    deduplicated), levels below ``threshold`` THz are labeled, and for each
    label all order-parameter directions are enumerated, including the
    ones a single frozen-in modulation would miss.

    Args:
        poscar: Full text of the VASP POSCAR of the unit cell.
        dim: Supercell dimension of the force calculation as three positive
            integers, e.g. ``[4, 4, 4]``.
        force_sets: Full text of the phonopy ``FORCE_SETS`` file of that
            supercell.
        threshold: Frequency in THz below which a mode counts as imaginary
            (default -0.1, the instability criterion of structure searches).

    Returns:
        Markdown: per imaginary level (q point, bands, frequency, irrep) a
        table of the isotropy subgroups (direction, subgroup, number, size,
        index); a note when no level lies below the threshold.
    """
    from crystod import phonon as crystod_phonon

    poscar_text = validate_poscar(poscar)
    supercell = parse_dim(dim)
    force_text = validate_force_sets(force_sets)
    try:
        cutoff = float(threshold)
    except (TypeError, ValueError):
        raise ValueError("threshold must be a frequency in THz, e.g. -0.1.") from None

    with temporary_files({"POSCAR": poscar_text, "FORCE_SETS": force_text}) as directory:
        phonon = _load_phonopy(directory, supercell)
        number, symbol = _space_group(phonon)
        qpoints = crystod_phonon.commensurate_qpoints(phonon)
        # scan_imaginary_modes reports a q point it cannot label as a
        # UserWarning instead of aborting; collect those (and only those --
        # library DeprecationWarnings are not the model's business)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("ignore")
            warnings.simplefilter("always", category=UserWarning)
            results = crystod_phonon.scan_imaginary_modes(phonon, qpoints, threshold=cutoff)
    skipped = []
    for warning in caught:
        message = " ".join(str(warning.message).split())
        if issubclass(warning.category, UserWarning) and message not in skipped:
            skipped.append(message)

    lines = [
        f"## Imaginary phonon modes of {symbol} (No. {number}) and their isotropy subgroups",
        "",
        f"{len(qpoints)} q points commensurate with the {supercell[0]}x{supercell[1]}x"
        f"{supercell[2]} supercell were scanned; threshold {cutoff:g} THz.",
        "",
    ]
    if not results:
        lines.append(
            f"No phonon level below {cutoff:g} THz at these q points: the structure is "
            "dynamically stable at the resolution of this supercell."
        )
    for result in results:
        mode = result.mode
        where = format_kpoint(mode.qpoint)
        if mode.qpoint_label:
            where = f"{mode.qpoint_label} {where}"
        bands = ",".join(str(b) for b in mode.band_indices)
        label = " + ".join(mode.labels) if mode.labels else "(not labeled)"
        lines.append(
            f"### {where}: bands {bands}, {mode.frequency:.4f} THz, irrep {label} "
            f"(degeneracy {mode.degeneracy})"
        )
        lines.append("")
        if result.subgroups:
            rows = [
                (sub.label, sub.symbol, sub.number, sub.size, sub.index, sub.n_free)
                for sub in result.subgroups
            ]
            lines.append(markdown_table(
                ("Order parameter", "Subgroup", "No.", "Size", "Index", "Free parameters"), rows
            ))
        for irrep, reason in result.errors.items():
            lines.append(f"No subgroups for {irrep}: {reason}")
        lines.append("")
    if skipped:
        lines.append("Skipped q points: " + "; ".join(skipped))
        lines.append("")
    lines.append(
        "Size: primitive-cell multiplication relative to the parent; index: index of "
        "the subgroup in the parent. Levels are sorted most unstable first."
    )
    return "\n".join(lines)


__all__ = ["crystod_phonon_irreps", "crystod_imaginary_mode_subgroups"]
