"""crystod-slab: surface slabs cut from a bulk crystal structure.

Every distinct termination of the requested surfaces is built with pymatgen's
SlabGenerator, listed with its number of atoms, thickness, vacuum, area,
formula, polarity and face symmetry, and written as a POSCAR file (backed by
:mod:`crystod.surface_slab`).  The Miller indices come from ``--index``,
``--index-list`` or ``--max-index``; with none of them the command asks for
the indices and the main settings interactively and prints the equivalent
command line.  The functionality of ``script/slabkit.py`` by R. Kiribayashi.
"""

from __future__ import annotations

import os
import re
import shlex
import sys
import textwrap
from argparse import ArgumentParser, RawTextHelpFormatter
from collections import Counter

from ..surface_slab import (
    DEFAULT_SLAB_SIZE,
    DEFAULT_TOLERANCE,
    DEFAULT_VACUUM_SIZE,
    OUTPUT_WIDTH,
    SLAB_CITATION,
    SLAB_CITATION_LINES,
)
from .common import (
    CRYSTOD_CITATION_LINES,
    ExampleRequested,
    add_cell_argument,
    add_example_argument,
    banner,
    run_example,
)

desc = """\
Cut surface slabs from a bulk crystal structure. For each Miller index
(h k l) every distinct termination is built with pymatgen's SlabGenerator
(whole oriented unit cells stacked to the minimum slab height, plus vacuum),
listed with its number of atoms, thickness, vacuum, area and formula, tested
for a dipole (polar: from the oxidation states) and for two equivalent faces
(symmetric), and written as POSCAR_slab_{formula}_{h}{k}{l}_t{n}.

The Miller indices come from --index (repeatable; four indices h k i l for
hexagonal axes), --index-list FILE (one index per line) or --max-index N
(every index with |h|,|k|,|l| <= N that is distinct under the Laue group of
the crystal). With none of them the indices and the main settings are asked
for interactively, and the equivalent command line is printed for the next
time.

Miller indices refer to the axes of the cell as given; --conventional
standardizes it to the conventional cell first (a primitive fcc POSCAR gives
the textbook (1 0 0), (1 1 0), (1 1 1) only with --conventional).

# Command Examples:
crystod-slab -c 221_PPOSCAR_SrTiO3                 (interactive)
crystod-slab -c 221_PPOSCAR_SrTiO3 --max-index 1
crystod-slab -c 221_PPOSCAR_SrTiO3 --index 1 0 0 --symmetrize
crystod-slab -c 221_PPOSCAR_SrTiO3 --max-index 2 --max-atoms 60 --no-polar
crystod-slab -c 225_PPOSCAR_NaCl --conventional --max-index 1 --no-files
crystod-slab -c 186_PPOSCAR_ZnO --index 1 0 -1 0 --index 0 0 0 1
crystod-slab -c POSCAR --index-list indices.txt --slab-size 15 --vacuum-size 20
crystod-slab --example SrTiO3   (bundled input; --example alone lists the names)
"""


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="crystod-slab",
        description=f"{banner()}\n\n{desc}",
        epilog=SLAB_CITATION + "\n\n" + "\n".join(_crystod_citation_lines()),
        formatter_class=RawTextHelpFormatter,
    )
    add_cell_argument(
        parser,
        "\nCIF and the other formats pymatgen reads work as well. Without -c\n"
        "and without ./POSCAR the dialogue asks for the file.",
        default=None,
    )
    add_example_argument(parser, "crystod-slab")

    source = parser.add_argument_group(
        "Miller indices (one source; none of them starts the dialogue)")
    source.add_argument(
        "--index",
        action="append",
        nargs="+",
        default=None,
        metavar="N",
        help="A Miller index h k l, or h k i l of hexagonal axes with\n"
        "i = -(h + k); 1,0,0 is read too. Repeat for several:\n"
        "--index 1 0 0 --index 1 1 0.",
    )
    source.add_argument(
        "--index-list",
        default=None,
        metavar="FILE",
        help="A file of Miller indices, one per line (1 0 0, 1,0,0 or\n"
        "1 0 -1 0); # starts a comment.",
    )
    source.add_argument(
        "--max-index",
        type=int,
        default=None,
        metavar="N",
        help="Every index with |h|,|k|,|l| <= N that is distinct under the\n"
        "Laue group of the crystal (found at --tolerance in the axes of\n"
        "the cell used).",
    )

    slab = parser.add_argument_group("slab construction")
    slab.add_argument(
        "--slab-size",
        type=float,
        default=DEFAULT_SLAB_SIZE,
        metavar="T",
        help=f"Minimum slab height in Angstrom (default: {DEFAULT_SLAB_SIZE:g}; pymatgen\n"
        "min_slab_size): whole oriented cells are stacked until they\n"
        "reach it, and the atoms span that height less the gap at the\n"
        "cut, so the thickness in the table can be smaller.",
    )
    slab.add_argument(
        "--vacuum-size",
        type=float,
        default=DEFAULT_VACUUM_SIZE,
        metavar="V",
        help=f"Minimum vacuum height in Angstrom (default: {DEFAULT_VACUUM_SIZE:g}), in whole\n"
        "oriented cells as well.",
    )
    slab.add_argument(
        "--in-unit-planes",
        action="store_true",
        help="--slab-size and --vacuum-size count d_hkl lattice planes of\n"
        "the cell instead of Angstrom (one plane can hold several atomic\n"
        "layers, e.g. SrO + TiO2 for SrTiO3 (1 0 0)).",
    )
    slab.add_argument(
        "--conventional",
        action="store_true",
        help="Standardize the input to the conventional standard cell first\n"
        "(pymatgen get_conventional_standard_structure); the Miller\n"
        "indices then refer to that cell. Default: the cell as given.",
    )
    slab.add_argument(
        "--symmetrize",
        action="store_true",
        help="Symmetric slabs: atoms are removed from one face of every\n"
        "asymmetric slab until the two faces are equivalent at --tolerance\n"
        "(pymatgen get_slabs(symmetrize=True)); the stoichiometry changes\n"
        "and the slab can carry a net charge (charge column). The files are\n"
        "named ..._s1, _s2 instead of _t1, _t2. Implies --no-polar.",
    )
    slab.add_argument(
        "--no-center",
        action="store_true",
        help="Do not center the slab in the cell (pymatgen center_slab).",
    )
    slab.add_argument(
        "--no-primitive",
        action="store_true",
        help="Keep the in-plane cell as oriented (pymatgen primitive=False).",
    )
    slab.add_argument(
        "--lll-reduce",
        action="store_true",
        help="Reduce the in-plane cell to its shortest a, b and make c as\n"
        "perpendicular to the surface as whole in-plane vectors allow\n"
        "(a and b stay in the surface; pymatgen's own lll_reduce can tilt\n"
        "the cell out of it).",
    )
    slab.add_argument(
        "--orthogonal-c",
        action="store_true",
        help="Make c perpendicular to the surface (Slab.get_orthogonal_c_slab).",
    )
    slab.add_argument(
        "--max-normal-search",
        type=int,
        default=None,
        metavar="N",
        help="Search lattice-vector combinations up to N for a c vector as\n"
        "perpendicular to the surface as possible (pymatgen).",
    )

    selection = parser.add_argument_group("selection and output")
    selection.add_argument(
        "--max-atoms",
        type=int,
        default=None,
        metavar="N",
        help="Slabs with more than N atoms are listed but not written\n"
        "(default: no limit).",
    )
    selection.add_argument(
        "--no-polar",
        action="store_true",
        help="Do not write polar slabs (they are listed as skipped).",
    )
    selection.add_argument(
        "--oxidation",
        nargs="+",
        default=None,
        metavar="EL=Q",
        help="Oxidation states for the dipole (polarity) test, one per\n"
        "element: --oxidation Sr=2 Ti=4 O=-2 (a fraction such as Fe=8/3\n"
        "is read too); they must balance the bulk, and all 0 means no\n"
        "ions (polarity ?). Default: pymatgen's guess from the\n"
        "composition; without one the polarity is ?.",
    )
    selection.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        metavar="T",
        help="Symmetry tolerance in Angstrom of the space group, the Laue\n"
        "group of --max-index, the conventional cell, the symmetric-slab\n"
        f"test and --symmetrize (default: {DEFAULT_TOLERANCE:g}).",
    )
    selection.add_argument(
        "--output-dir",
        default=".",
        metavar="DIR",
        help="Directory of the POSCAR files (created if missing; default:\n"
        "the current directory). An identical file already there is\n"
        "kept; one with different content is refused unless --force.",
    )
    selection.add_argument(
        "--force",
        action="store_true",
        help="Replace existing POSCAR_slab_* files whose content differs.",
    )
    selection.add_argument(
        "--no-files",
        action="store_true",
        help="List the slabs only; write nothing.",
    )
    return parser


# ----------------------------------------------------------------- printing
def _block(title: str) -> None:
    print(f"\n* {title} *")


def _protect(match: re.Match) -> str:
    return match.group(0).replace(" ", "\xa0")


def _wrapped(text: str, indent: str = " ", subsequent: str | None = None) -> list[str]:
    """``text`` wrapped to OUTPUT_WIDTH columns; a short parenthesis (a Miller
    index ``(1 1 0)``, ``(No. 221)``) and a short quotation are never split."""
    text = " ".join(text.split())
    text = re.sub(r"\([^()]{1,24}\)", _protect, text)
    text = re.sub(r"(?<![A-Za-z])'[^']{1,40}'(?![A-Za-z])", _protect, text)
    lines = textwrap.wrap(text, width=OUTPUT_WIDTH, initial_indent=indent,
                          subsequent_indent=indent if subsequent is None else subsequent,
                          break_on_hyphens=False, break_long_words=False)
    return [line.replace("\xa0", " ") for line in lines]


def _note(text: str) -> None:
    for line in _wrapped(text, " NOTE: ", " " * 7):
        print(line)


def _pack(parts: list[str], separator: str = "   ") -> list[str]:
    """Greedy packing of short phrases into lines of at most OUTPUT_WIDTH columns."""
    lines: list[str] = []
    for part in parts:
        if lines and len(lines[-1]) + len(separator) + len(part) <= OUTPUT_WIDTH:
            lines[-1] += separator + part
        else:
            lines.append(" " + part)
    return lines


def _path_lines(label: str, path: str) -> list[str]:
    """``label path`` on one line, or broken after a ``/`` when it is too long."""
    first = f" {label} "
    if len(first) + len(path) <= OUTPUT_WIDTH:
        return [first + path]
    pieces = [piece + "/" for piece in path.split("/")]
    pieces[-1] = pieces[-1][:-1]
    lines, current = [], first
    for piece in pieces:
        if len(current) + len(piece) > OUTPUT_WIDTH and current.strip() != label:
            lines.append(current)
            current = " " * len(first)
        current += piece
    lines.append(current)
    return lines


def _split_word(line: str) -> list[str]:
    """A command line still wider than OUTPUT_WIDTH (one long path) broken after
    a ``/`` with a backslash; the rest of the word starts at column 0, as the
    shell removes a backslash-newline inside an unquoted word."""
    pieces = []
    while len(line) > OUTPUT_WIDTH:
        cut = line.rfind("/", 0, OUTPUT_WIDTH - 1)
        if cut <= 0 or line[:cut].count("'") % 2 or line[cut + 1:cut + 2] in ("", " "):
            break
        pieces.append(line[:cut + 1] + "\\")
        line = line[cut + 1:]
    pieces.append(line)
    return pieces


def _command_lines(tokens: list[str]) -> list[str]:
    """A shell command as lines of at most OUTPUT_WIDTH columns (``\\`` continuations).

    The line breaks fall between options, an option and its values staying
    together (``--index 1 1 0``); a single path too long for a line is broken
    after a ``/``.
    """
    chunks: list[str] = []
    for token in tokens:
        word = shlex.quote(token)
        if chunks and not (token == "-c" or token.startswith("--")):
            chunks[-1] += " " + word
        else:
            chunks.append(word)
    lines, current = [], " " + chunks[0]
    for chunk in chunks[1:]:
        if len(current) + 1 + len(chunk) + 2 > OUTPUT_WIDTH:
            lines.append(current + " \\")
            current = "   " + chunk
        else:
            current += " " + chunk
    lines.append(current)
    return [piece for line in lines for piece in _split_word(line)]


def _crystod_citation_lines() -> list[str]:
    lines = []
    for line in CRYSTOD_CITATION_LINES:
        lines.extend(textwrap.wrap(line, width=OUTPUT_WIDTH, subsequent_indent="  ",
                                   break_on_hyphens=False, break_long_words=False))
    return lines


def _number(value: float) -> str:
    """A number as the command line takes it back exactly: ``10``, ``10.0000004``."""
    value = float(value)
    return str(int(value)) if value.is_integer() and abs(value) < 1e15 else repr(value)


def _numbers(numbers: list[int]) -> str:
    """``#1-#4, #6, #7``: table row numbers, runs of three or more as a range."""
    runs: list[list[int]] = []
    for number in numbers:
        if runs and number == runs[-1][-1] + 1:
            runs[-1].append(number)
        else:
            runs.append([number])
    return ", ".join(f"#{run[0]}-#{run[-1]}" if len(run) > 2
                     else ", ".join(f"#{n}" for n in run) for run in runs)


# --------------------------------------------------------------- the dialogue
def _ask(prompt: str, parse, default: str | None = None):
    """One question of the dialogue: up to three tries, then an ERROR.

    An empty answer takes ``default`` (an ERROR when there is none); the end
    of the input or Ctrl-C is an ERROR.  When the answers come from a pipe
    they are echoed, so that the transcript reads like the interactive
    session.
    """
    for _ in range(3):
        try:
            answer = input(prompt)
        except EOFError:
            print()
            raise SystemExit("ERROR: the input ended before the dialogue was complete "
                             "(crystod-slab --help lists the options).") from None
        except KeyboardInterrupt:
            print()
            raise SystemExit("ERROR: interrupted; nothing was done.") from None
        if not sys.stdin.isatty():
            print(answer)
        answer = answer.strip()
        if not answer:
            if default is None:
                raise SystemExit(f"ERROR: no answer to '{prompt.strip()}' (it has no default).")
            answer = default
        try:
            return parse(answer)
        except ValueError as exc:
            for line in _wrapped(f"not valid: {exc}", "   ", "     "):
                print(line)
    raise SystemExit(f"ERROR: no valid answer to '{prompt.strip()}' after 3 tries.")


def _library(function, *args):
    """Call a surface_slab function and turn its ``ERROR:`` exit into ValueError."""
    try:
        return function(*args)
    except SystemExit as exc:
        raise ValueError(" ".join(str(exc).split()).removeprefix("ERROR: ")) from None


def _choice(text: str) -> int:
    if text not in ("1", "2", "3", "4"):
        raise ValueError("choose 1, 2, 3 or 4")
    return int(text)


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise ValueError(f"'{text}' is not a whole number") from None
    if value < 1:
        raise ValueError("give a positive whole number")
    return value


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"'{text}' is not a number") from None
    if not value > 0 or value == float("inf"):
        raise ValueError("give a positive number")
    return value


def _atom_limit(text: str) -> int | None:
    if text.lower() in ("no limit", "none", "no", "unlimited", "-"):
        return None
    return _positive_int(text)


def _yes_no(text: str) -> bool:
    answer = text.lower()
    if answer in ("y", "yes"):
        return True
    if answer in ("n", "no"):
        return False
    raise ValueError("answer yes or no")


def _dialogue(args, oxidation):
    """Ask for the indices and the main settings; ``None`` when the user quits.

    The answers are stored in ``args``.  Returns ``(source, bulk)``: the
    tokens of the index source on the equivalent command line
    (``["--max-index", "2"]``) and the structure when the dialogue had to
    ask for its file (``None`` otherwise).
    """
    from .. import surface_slab as ss

    _block("Miller indices")
    bulk = None
    if args.cell is None and not os.path.isfile("POSCAR"):
        args.cell, bulk = _ask(
            " Structure file [POSCAR]: ",
            lambda text: (text, _library(ss.load_structure, text, False, args.tolerance)),
            "POSCAR")
        ss.resolve_oxidation_states(bulk, oxidation)
    print(" How should the Miller indices be given?")
    print("  1) one index (h k l)")
    print("  2) a list file (one index per line)")
    print("  3) every symmetrically distinct index up to a maximum")
    print("  4) quit")
    choice = _ask(" Choice [1-4]: ", _choice)
    if choice == 4:
        print(" quit: nothing was done.")
        return None
    if choice == 1:
        tokens = _ask(" Miller index (h k l, or h k i l for hexagonal axes): ",
                      lambda text: (_library(ss.parse_miller_index, text),
                                    [str(int(t)) for t in ss._index_tokens(text)])[1])
        args.index = [tokens]
        source = ["--index", *tokens]
    elif choice == 2:
        path = _ask(" Index list file (one index per line): ",
                    lambda text: (_library(ss._read_index_entries, text), text)[1])
        args.index_list = path
        source = ["--index-list", path]
    else:
        args.max_index = _ask(" Maximum Miller index [1]: ", _positive_int, "1")
        source = ["--max-index", str(args.max_index)]
    unit = "d_hkl lattice planes" if args.in_unit_planes else "Angstrom"
    args.slab_size = _ask(f" Minimum slab height in {unit} [{_number(args.slab_size)}]: ",
                          _positive_float, _number(args.slab_size))
    args.vacuum_size = _ask(
        f" Minimum vacuum height in {unit} [{_number(args.vacuum_size)}]: ",
        _positive_float, _number(args.vacuum_size))
    limit = "no limit" if args.max_atoms is None else str(args.max_atoms)
    args.max_atoms = _ask(f" Maximum number of atoms per slab [{limit}]: ", _atom_limit, limit)
    polar = "no" if (args.no_polar or args.symmetrize) else "yes"
    allow_polar = _ask(f" Allow polar slabs? yes/no [{polar}]: ", _yes_no, polar)
    if allow_polar:
        args.no_polar = args.symmetrize = False
    else:
        args.no_polar = True
        symmetrize = "yes" if args.symmetrize else "no"
        args.symmetrize = _ask(
            f" Symmetrize the slabs instead (atoms removed from one face)? yes/no "
            f"[{symmetrize}]: ", _yes_no, symmetrize)
    return source, bulk


def _equivalent_command(args, source: list[str]) -> list[str]:
    tokens = ["crystod-slab", "-c", args.cell or "POSCAR", *source]
    if args.slab_size != DEFAULT_SLAB_SIZE:
        tokens += ["--slab-size", _number(args.slab_size)]
    if args.vacuum_size != DEFAULT_VACUUM_SIZE:
        tokens += ["--vacuum-size", _number(args.vacuum_size)]
    if args.max_atoms is not None:
        tokens += ["--max-atoms", str(args.max_atoms)]
    if args.symmetrize:
        tokens.append("--symmetrize")
    elif args.no_polar:
        tokens.append("--no-polar")
    if args.oxidation:
        tokens += ["--oxidation", *args.oxidation]
    for flag, given in (("--conventional", args.conventional),
                        ("--in-unit-planes", args.in_unit_planes),
                        ("--no-center", args.no_center), ("--no-primitive", args.no_primitive),
                        ("--lll-reduce", args.lll_reduce), ("--orthogonal-c", args.orthogonal_c)):
        if given:
            tokens.append(flag)
    if args.max_normal_search is not None:
        tokens += ["--max-normal-search", str(args.max_normal_search)]
    if args.tolerance != DEFAULT_TOLERANCE:
        tokens += ["--tolerance", _number(args.tolerance)]
    if args.output_dir != ".":
        tokens += ["--output-dir", args.output_dir]
    if args.force:
        tokens.append("--force")
    if args.no_files:
        tokens.append("--no-files")
    return tokens


# -------------------------------------------------------------------- the run
def _check_output_dir(directory: str) -> None:
    """The output directory exists as a directory, or can be created: its first
    existing ancestor is a directory we may write into."""
    probe = directory
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe.rstrip("/"))
        probe = parent if parent != probe else ""
    probe = probe or "."
    if not os.path.isdir(probe):
        raise SystemExit(f"ERROR: --output-dir {directory}: {probe} is not a directory.")
    if not os.access(probe, os.W_OK | os.X_OK):
        what = "write into" if probe == directory else f"create {directory} in"
        raise SystemExit(f"ERROR: --output-dir {directory}: cannot {what} {probe} "
                         "(permission denied).")


def _check_options(args) -> None:
    from .. import surface_slab as ss

    sources = [flag for flag, given in (("--index", bool(args.index)),
                                        ("--index-list", args.index_list is not None),
                                        ("--max-index", args.max_index is not None)) if given]
    if len(sources) > 1:
        raise SystemExit("ERROR: give one index source, --index, --index-list or --max-index "
                         f"(got {' and '.join(sources)}).")
    if args.max_index is not None:
        ss._check_max_index(args.max_index)
    ss._check_size(args.slab_size, "slab height (--slab-size)")
    ss._check_size(args.vacuum_size, "vacuum height (--vacuum-size)")
    ss._check_tolerance(args.tolerance)
    if args.max_atoms is not None and args.max_atoms < 1:
        raise SystemExit(f"ERROR: --max-atoms must be a positive integer (got {args.max_atoms}).")
    if args.max_normal_search is not None and args.max_normal_search < 1:
        raise SystemExit("ERROR: --max-normal-search must be a positive integer "
                         f"(got {args.max_normal_search}).")
    if not args.no_files:
        _check_output_dir(args.output_dir)


def _parse_index_option(tokens: list[str]) -> tuple[int, int, int]:
    from .. import surface_slab as ss

    count = len(ss._index_tokens(tokens))
    if count > 4:
        raise SystemExit(f"ERROR: --index takes one Miller index; got '{' '.join(tokens)}' "
                         "(for several: --index 1 0 0 --index 1 1 0).")
    return ss.parse_miller_index(tokens)


def _index_entries(args) -> list[tuple[tuple[int, int, int], list[str]]] | None:
    """``(hkl, tokens)`` of every index of --index or --index-list (None for neither)."""
    from .. import surface_slab as ss

    if args.index:
        return [(_parse_index_option(tokens), ss._index_tokens(tokens)) for tokens in args.index]
    if args.index_list is not None:
        return ss._read_index_entries(args.index_list)
    return None


def _shown_entries(entries) -> str:
    from .. import surface_slab as ss

    shown = []
    for index, tokens in entries:
        hkl = ss.format_miller_index(index)
        shown.append(f"{ss.format_miller_index(tokens)} = {hkl}" if len(tokens) == 4 else hkl)
    return " ".join(shown)


def _write_files(records, written, args, order) -> None:
    """Write the POSCAR files of the slabs to be written, after checking that
    none of them would replace a different file (unless --force)."""
    from .. import surface_slab as ss

    names = Counter(ss.slab_filename(record) for _, record in written)
    twice = [name for name, count in names.items() if count > 1]
    if twice:
        raise SystemExit(f"ERROR: two slabs of this run have the same file name {twice[0]}.")
    conflicts = []
    for _, record in written:
        path = ss.slab_filename(record)
        if os.path.normpath(args.output_dir) != ".":
            path = os.path.join(args.output_dir, path)
        if os.path.isdir(path):
            raise SystemExit(f"ERROR: {path} is a directory.")
        if os.path.exists(path):
            try:
                with open(path, "rb") as handle:
                    same = handle.read() == ss.slab_poscar_text(record, order).encode("utf-8")
            except OSError:
                same = False
            if not same:
                conflicts.append(path)
    if conflicts and not args.force:
        more = f" (and {len(conflicts) - 1} more)" if len(conflicts) > 1 else ""
        raise SystemExit(f"ERROR: {conflicts[0]}{more} already exists with different content "
                         "(--force replaces it).")
    for number, record in written:
        path, status = ss.write_slab_poscar(record, args.output_dir, order, overwrite=args.force)
        note = {"wrote": "", "kept": " (kept: identical file already there)",
                "replaced": " (replaced)"}[status]
        print(f" #{number} -> {os.path.basename(path)}{note}")


def run(args) -> None:
    """Carry out a parsed ``crystod-slab`` command line."""
    from .. import surface_slab as ss

    _check_options(args)
    oxidation = ss.parse_oxidation_states(args.oxidation) if args.oxidation else None
    entries = _index_entries(args)
    interactive = entries is None and args.max_index is None
    bulk = None
    if not interactive or args.cell is not None or os.path.isfile("POSCAR"):
        # the structure and the --oxidation states are checked before any question
        bulk = ss.load_structure(args.cell or "POSCAR", tolerance=args.tolerance)
        ss.resolve_oxidation_states(bulk, oxidation)

    if interactive:
        answer = _dialogue(args, oxidation)
        if answer is None:
            return
        source, asked = answer
        bulk = bulk if asked is None else asked
        entries = _index_entries(args)
        _block("Equivalent command line")
        for line in _command_lines(_equivalent_command(args, source)):
            print(line)
    cell = args.cell or "POSCAR"
    no_polar = args.no_polar or args.symmetrize

    # --------------------------------------------------------------- structure
    info = ss.describe_cell(bulk, args.tolerance)
    structure = ss.conventional_cell(bulk, args.tolerance) if args.conventional else bulk
    used = ss.describe_cell(structure, args.tolerance) if args.conventional else info
    states, source = ss.resolve_oxidation_states(structure, oxidation)
    _block("Structure")
    for line in _wrapped(f"{os.path.basename(cell)}: {info.formula}, {info.space_group} "
                         f"(No. {info.space_group_number}), {info.n_atoms} atoms", " ", "   "):
        print(line)
    if args.conventional:
        print(f" standardized conventional cell: {used.n_atoms} atoms (input: {info.n_atoms})")
    a, b, c, alpha, beta, gamma = used.lattice
    print(f" a, b, c = {a:.4f}, {b:.4f}, {c:.4f} A; alpha, beta, gamma = "
          f"{round(alpha, 3):g}, {round(beta, 3):g}, {round(gamma, 3):g} deg")
    print(" Miller indices refer to the axes of this cell.")
    if not args.conventional and not info.is_conventional:
        _note(f"the cell is not the conventional standard cell ({info.n_conventional} atoms); "
              "textbook Miller indices need --conventional.")
    four_index = any(len(tokens) == 4 for _, tokens in (entries or ()))
    hexagonal = abs(a - b) < 1e-3 * a and abs(gamma - 120.0) < 0.5
    if four_index and not hexagonal:
        _note(f"four-index (h k i l) input assumes hexagonal axes (a = b, gamma = 120 deg); "
              f"this cell has gamma = {round(gamma, 3):g} deg.")

    # ---------------------------------------------------------------- settings
    _block("Slab settings")
    unit = "d_hkl lattice planes" if args.in_unit_planes else "A"
    parts = [f"minimum slab height: {_number(args.slab_size)} {unit}",
             f"minimum vacuum: {_number(args.vacuum_size)} {unit}",
             f"centered: {'no' if args.no_center else 'yes'}",
             f"primitive in-plane cell: {'no' if args.no_primitive else 'yes'}"]
    if args.lll_reduce:
        parts.append("reduced in-plane cell (--lll-reduce): yes")
    if args.orthogonal_c:
        parts.append("c perpendicular to the surface: yes")
    if args.max_normal_search is not None:
        parts.append(f"max normal search: {args.max_normal_search}")
    for line in _pack(parts):
        print(line)
    if args.symmetrize:
        print(" polar slabs: symmetrized terminations (atoms removed from one face of every "
              "asymmetric slab)")
    else:
        print(f" polar slabs: {'skipped' if no_polar else 'written'}")
    if states is not None:
        print(f" oxidation states for the dipole: {ss.format_oxidation_states(states)} ({source})")
    elif source == "given":
        print(" oxidation states for the dipole: none (polarity unknown: --oxidation gives "
              "every element 0)")
    else:
        print(" oxidation states for the dipole: none (polarity unknown)")
    limit = "no limit" if args.max_atoms is None else str(args.max_atoms)
    print(f" maximum atoms per slab: {limit}")

    # ---------------------------------------------------------- Miller indices
    _block("Miller indices")
    if args.max_index is not None:
        indices = ss.distinct_miller_indices(structure, args.max_index, args.tolerance)
        text = (f"indices with |h|,|k|,|l| <= {args.max_index}, distinct under the Laue group "
                f"{used.laue_group}: "
                + " ".join(ss.format_miller_index(index) for index in indices))
    else:
        indices = [index for index, _ in entries]
        where = (f"from {os.path.basename(args.index_list)}" if args.index_list is not None
                 else "given")
        text = f"indices {where}: " + _shown_entries(entries)
    for line in _wrapped(text, " ", "   "):
        print(line)
    if four_index:
        print(" (h k i l) is read as (h k l): the redundant i = -(h + k) is dropped")

    # -------------------------------------------------------------------- slabs
    records = ss.generate_slabs(
        structure, indices, slab_size=args.slab_size, vacuum_size=args.vacuum_size,
        symmetrize=args.symmetrize, oxidation_states=oxidation, tolerance=args.tolerance,
        center=not args.no_center, primitive=not args.no_primitive, lll_reduce=args.lll_reduce,
        in_unit_planes=args.in_unit_planes, max_normal_search=args.max_normal_search,
        orthogonal_c=args.orthogonal_c)
    statuses = [ss.slab_status(record, args.max_atoms, no_polar, args.no_files)
                for record in records]
    _block("Slabs")
    print(ss.format_table(records, args.max_atoms, no_polar, args.no_files))
    if not records:
        print(" no slab was generated")
    for line in _wrapped(ss.table_legend(records)):
        print(line)
    for note in records.notes:
        _note(note)
    charged = [number for number, record in enumerate(records, start=1) if record.charged]
    if charged:
        _note(f"{_numbers(charged)}: non-stoichiometric, with a net formal charge (charge "
              "column); 'polar: no' means no dipole about the slab centre, not a neutral slab.")
    if states is None:
        why = ("every oxidation state given is 0" if source == "given"
               else "no oxidation states")
        if no_polar:
            _note(f"polarity unknown ({why}): slabs kept; give the states with "
                  "--oxidation EL=Q ...")
        else:
            _note(f"polarity unknown ({why}); give them with --oxidation EL=Q ...")

    # -------------------------------------------------------------------- files
    _block("Output files")
    written = [(number, record) for number, (record, status)
               in enumerate(zip(records, statuses), start=1) if status == "written"]
    if args.no_files:
        print(" none (--no-files)")
    elif not records:
        print(" none (no slab was generated)")
    elif not written:
        print(" none (every slab was skipped)")
    else:
        if os.path.normpath(args.output_dir) != ".":
            for line in _path_lines("directory:", args.output_dir):
                print(line)
        _write_files(records, written, args, list(dict.fromkeys(
            site.specie.symbol for site in bulk)))

    print()
    for line in SLAB_CITATION_LINES:
        print(line)
    print()
    for line in _crystod_citation_lines():
        print(line)


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except ExampleRequested as request:
        # --example: put the bundled input in place, then run the ordinary
        # command line it stands for (plus whatever else was given)
        main(run_example("crystod-slab", request.name, argv))
        return
    run(args)


if __name__ == "__main__":
    main()
