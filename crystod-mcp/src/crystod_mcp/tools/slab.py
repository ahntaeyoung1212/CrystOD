"""Surface-slab tool (the ``crystod-slab`` domain).

The tool reads the POSCAR text from a temporary directory with
``crystod.slab.load_structure`` and calls ``crystod.slab.generate_slabs``,
the data function behind ``crystod-slab``: every termination of one Miller
index, or of every index up to a maximum that is distinct under the Laue
group of the crystal, measured (atoms, thickness, vacuum, area, polarity,
equivalent faces). The table is the one ``crystod-slab`` prints
(``crystod.slab.format_table``); the slab POSCAR files are written only into
an ``output_dir`` the call names, and never replace a different file.
"""

from __future__ import annotations

import re
import textwrap
from collections import Counter
from numbers import Integral
from pathlib import Path

from ..utils import blocking_tool, fenced, temporary_files, validate_poscar

_WIDTH = 100

# the option names of crystod-slab in the messages and notes of crystod.slab,
# as the arguments of this tool
_OPTION_NAMES = (("--oxidation", "oxidation_states"), ("--slab-size", "slab_size"))


def _wrap(text: str, first: str = "", rest: str = "") -> list[str]:
    return textwrap.wrap(text, width=_WIDTH, initial_indent=first, subsequent_indent=rest,
                         break_on_hyphens=False, break_long_words=False)


def _arguments(text: str) -> str:
    for option, argument in _OPTION_NAMES:
        text = text.replace(option, argument)
    return text


def _index_token_count(miller_index: object) -> int:
    if isinstance(miller_index, str):
        return len([t for t in re.split(r"[\s,]+", miller_index.strip().strip("()[]")) if t])
    try:
        return len(list(miller_index))
    except TypeError:
        return 1


def _output_directory(output_dir: str | None) -> Path | None:
    text = "" if output_dir is None else str(output_dir).strip()
    if not text:
        return None
    directory = Path(text).expanduser()
    if not directory.is_absolute():
        raise ValueError(
            f'output_dir must be an absolute directory such as /home/me/slabs (got "{text}"): '
            "a relative one would depend on where the server was started."
        )
    if directory.exists() and not directory.is_dir():
        raise ValueError(f"output_dir {directory} exists and is not a directory.")
    return directory


def _write_files(slab, written, directory: Path, order: list[str]) -> list[str]:
    """Write the POSCAR of every slab in ``written`` after checking that none of
    them would replace a different file; the lines that list the files."""
    names = Counter(slab.slab_filename(record) for _, record in written)
    twice = [name for name, count in names.items() if count > 1]
    if twice:
        raise ValueError(f"two slabs of this call have the same file name {twice[0]}.")
    conflicts = []
    for _, record in written:
        path = directory / slab.slab_filename(record)
        if path.is_dir():
            raise ValueError(f"{path} is a directory.")
        if path.exists():
            try:
                same = path.read_bytes() == slab.slab_poscar_text(record, order).encode("utf-8")
            except OSError:
                same = False
            if not same:
                conflicts.append(path)
    if conflicts:
        more = f" (and {len(conflicts) - 1} more)" if len(conflicts) > 1 else ""
        raise ValueError(
            f"{conflicts[0]}{more} already exists with different content; this tool never "
            "replaces a file: remove it or give another output_dir."
        )
    lines = [f"Written into {directory}:", ""]
    for number, record in written:
        path, status = slab.write_slab_poscar(record, directory, order)
        note = {"wrote": "", "kept": " (kept: identical file already there)"}.get(status, "")
        lines.append(f"- #{number} {Path(path).name}{note}")
    return lines


@blocking_tool
def crystod_slab(
    poscar: str,
    miller_index: str | list[int] | None = None,
    max_index: int | None = None,
    slab_size: float = 10.0,
    vacuum_size: float = 15.0,
    symmetrize: bool = False,
    no_polar: bool = False,
    max_atoms: int | None = None,
    oxidation_states: dict[str, float] | None = None,
    conventional: bool = False,
    output_dir: str | None = None,
) -> str:
    """Cut surface slabs from a bulk crystal: every termination of one Miller index, or of every index up to max_index that is distinct under the crystal's Laue group, with its atoms, thickness, vacuum, area, polarity (dipole) and face equivalence, e.g. cubic SrTiO3 with max_index 1 gives (1 1 1) and (1 1 0) polar slabs of 25 and 20 atoms and a (1 0 0) nonpolar one of 15.

    The analysis of ``crystod-slab -c POSCAR --index h k l`` or
    ``--max-index N``: for each index pymatgen's ``SlabGenerator`` stacks
    whole oriented cells to the minimum slab height plus the minimum vacuum
    and lists every termination (t1, t2, ...; s1, s2, ... symmetrized).
    ``polar`` is ``Slab.is_polar`` with the oxidation states (pymatgen's
    guess from the bulk composition unless given; ``?`` without any, as for
    a metal), ``symmetric`` means the two faces are equivalent. The POSCAR
    of every slab the table does not skip is written only when
    ``output_dir`` is given; an identical file there is kept and a different
    one is never replaced. Please cite R. Kiribayashi et al., ACS Appl.
    Mater. Interfaces 17, 61872-61887 (2025) for crystod-slab; the result
    ends with the citation lines.

    Args:
        poscar: Full text of the VASP POSCAR of the bulk crystal (VASP 5
            format with the element line); the Miller indices refer to the
            axes of this cell (see ``conventional``).
        miller_index: One Miller index as three integers (``"1 1 0"``,
            ``[1, 1, 0]``) or four of hexagonal axes (``"1 0 -1 0"``, with
            ``i = -(h + k)``). Give this or ``max_index``.
        max_index: Every index with ``|h|, |k|, |l| <= max_index`` that is
            distinct under the Laue group of the crystal, e.g. ``1``.
        slab_size: Minimum slab height in Angstrom (default 10); the atoms
            span that height less the gap at the cut.
        vacuum_size: Minimum vacuum height in Angstrom (default 15).
        symmetrize: Symmetric terminations: atoms are removed from one face
            of every asymmetric slab until the faces are equivalent (the
            stoichiometry changes; a charged slab gets a charge column).
            Implies ``no_polar``.
        no_polar: Skip polar slabs (a slab of unknown polarity is kept).
        max_atoms: Skip slabs with more atoms than this.
        oxidation_states: Oxidation states for the dipole, one per element of
            the structure and balancing the bulk, e.g.
            ``{"Sr": 2, "Ti": 4, "O": -2}``; by default pymatgen's guess.
        conventional: Standardize the bulk to the conventional standard cell
            first, so that the indices are the textbook ones (e.g. for a
            primitive fcc cell).
        output_dir: Absolute directory to write the slab POSCAR files into
            (``POSCAR_slab_SrTiO3_111_t1``, ...), created if needed. Omitted,
            no file is written.

    Returns:
        Markdown: the bulk space group and cell, the settings and indices,
        the ``crystod-slab`` table (``#  h k l  shift  atoms  thickness
        vacuum  area  formula  polar  symmetric  status``) with its legend
        and notes, the files written, and the citation lines.
    """
    try:
        from crystod import slab
    except ImportError:
        raise ValueError(
            "crystod_slab needs CrystOD 0.4.6 or later (crystod.slab); upgrade CrystOD in "
            "the environment of the server."
        ) from None

    poscar_text = validate_poscar(poscar)
    if max_atoms is not None and (
            isinstance(max_atoms, bool) or not isinstance(max_atoms, Integral) or max_atoms < 1):
        raise ValueError(f"max_atoms must be a positive integer (got {max_atoms}).")
    if oxidation_states is not None and not isinstance(oxidation_states, dict):
        raise ValueError(
            'oxidation_states must map each element to its charge, e.g. {"Sr": 2, "Ti": 4, '
            '"O": -2}.'
        )
    directory = _output_directory(output_dir)
    no_polar = bool(no_polar or symmetrize)
    no_files = directory is None

    with temporary_files({"POSCAR": poscar_text}) as tmp:
        try:
            bulk = slab.load_structure(tmp / "POSCAR")
        except ValueError as exc:
            message = str(exc).replace(str(tmp / "POSCAR"), "the POSCAR text")
            raise ValueError(message) from None
    info = slab.describe_cell(bulk)
    structure = slab.conventional_cell(bulk) if conventional else bulk
    used = slab.describe_cell(structure) if conventional else info
    try:
        records = slab.generate_slabs(
            structure, miller_index, max_index=max_index, slab_size=slab_size,
            vacuum_size=vacuum_size, symmetrize=bool(symmetrize),
            oxidation_states=oxidation_states)
    except ValueError as exc:
        raise ValueError(_arguments(str(exc))) from None

    # ----------------------------------------------------------- bulk, settings
    a, b, c, alpha, beta, gamma = used.lattice
    bulk_text = (f"Bulk: {info.formula}, {info.space_group} (No. {info.space_group_number}), "
                 f"{info.n_atoms} atoms")
    if conventional:
        bulk_text += f"; standardized conventional cell: {used.n_atoms} atoms"
    bulk_text += (f"; a, b, c = {a:.4f}, {b:.4f}, {c:.4f} A; alpha, beta, gamma = "
                  f"{round(alpha, 3):g}, {round(beta, 3):g}, {round(gamma, 3):g} deg. "
                  "Miller indices refer to the axes of this cell.")
    if max_index is not None:
        index_text = (f"Miller indices with |h|, |k|, |l| <= {max_index}, distinct under the "
                      f"Laue group {used.laue_group}: ")
    else:
        index_text = "Miller index: "
    index_text += " ".join(slab.format_miller_index(index) for index in records.miller_indices)
    if symmetrize:
        polar_text = "symmetrized terminations (atoms removed from one face of every asymmetric slab)"
    else:
        polar_text = "skipped" if no_polar else "written"
    states, source = records.oxidation_states, records.oxidation_source
    if states is not None:
        states_text = f"{slab.format_oxidation_states(states)} ({source})"
    elif source == "given":
        states_text = "none (polarity unknown: every state given is 0)"
    else:
        states_text = "none (polarity unknown)"
    settings_text = (f"Settings: minimum slab height {slab_size:g} A, minimum vacuum "
                     f"{vacuum_size:g} A; polar slabs: {polar_text}; oxidation states for the "
                     f"dipole: {states_text}; maximum atoms per slab: "
                     f"{'no limit' if max_atoms is None else max_atoms}.")

    notes: list[str] = []
    if not conventional and not info.is_conventional:
        notes.append(f"the cell is not the conventional standard cell ({info.n_conventional} "
                     "atoms); textbook Miller indices need conventional=true.")
    if miller_index is not None and _index_token_count(miller_index) == 4:
        hexagonal = abs(a - b) < 1e-3 * a and abs(gamma - 120.0) < 0.5
        notes.append("(h k i l) is read as (h k l): the redundant i = -(h + k) is dropped.")
        if not hexagonal:
            notes.append(f"four-index (h k i l) input assumes hexagonal axes (a = b, gamma = "
                         f"120 deg); this cell has gamma = {round(gamma, 3):g} deg.")
    notes.extend(_arguments(note) for note in records.notes)
    charged = [str(n) for n, record in enumerate(records, start=1) if record.charged]
    if charged:
        notes.append(f"#{', #'.join(charged)}: non-stoichiometric, with a net formal charge "
                     "(charge column); 'polar: no' means no dipole about the slab centre, not "
                     "a neutral slab.")
    if states is None:
        why = "every oxidation state given is 0" if source == "given" else "no oxidation states"
        kept = "slabs kept; " if no_polar else ""
        notes.append(f"polarity unknown ({why}): {kept}give them with oxidation_states, e.g. "
                     '{"Sr": 2, "Ti": 4, "O": -2}.')

    lines = [f"## Surface slabs of {info.formula} ({info.space_group}, "
             f"No. {info.space_group_number})", ""]
    for text in (bulk_text, index_text, settings_text):
        lines += [*_wrap(text), ""]
    lines.append(fenced(slab.format_table(records, max_atoms, no_polar, no_files)))
    lines.append("")
    if not records:
        lines += ["No slab was generated.", ""]
    lines += [*_wrap(slab.table_legend(records)), ""]
    for note in notes:
        lines += _wrap(note, "- NOTE: ", "  ")
    if notes:
        lines.append("")

    # -------------------------------------------------------------------- files
    lines += ["### Output files", ""]
    statuses = [slab.slab_status(record, max_atoms, no_polar, no_files) for record in records]
    written = [(number, record) for number, (record, status)
               in enumerate(zip(records, statuses), start=1) if status == "written"]
    if no_files:
        lines += _wrap("None: no output_dir was given (the status 'not written (--no-files)'). "
                       "Give output_dir, an absolute directory, to write the POSCAR of every "
                       "slab the table does not skip.")
    elif not records:
        lines.append("None (no slab was generated).")
    elif not written:
        lines.append("None (every slab was skipped).")
    else:
        order = list(dict.fromkeys(site.specie.symbol for site in bulk))
        lines += _write_files(slab, written, directory, order)
    lines += ["", *slab.SLAB_CITATION_LINES]
    return "\n".join(lines)


__all__ = ["crystod_slab"]
