"""Shared helpers for the sectioned CrystOD commands.

Option conventions (phonopy-aligned):

- ``-c`` / ``--cell`` selects the crystal structure file; ``--poscar`` is
  kept as a hidden alias for backward compatibility.
- ``-o`` / ``--output`` selects the output path.
"""

from __future__ import annotations

from argparse import Action, ArgumentParser
from pathlib import Path

# "CrystOD" rendered in the figlet "standard" font. Embedded as a literal so
# the banner has no runtime dependency on pyfiglet.
_BANNER_ART = r"""
  ____                _    ___  ____
 / ___|_ __ _   _ ___| |_ / _ \|  _ \
| |   | '__| | | / __| __| | | | | | |
| |___| |  | |_| \__ \ |_| |_| | |_| |
 \____|_|   \__, |___/\__|\___/|____/
            |___/
"""


def banner() -> str:
    """CrystOD ASCII-art banner with the subtitle and current version.

    Used as the leading block of the main command's ``--help`` description.
    """
    from .. import __version__

    return f"{_BANNER_ART.strip(chr(10))}\nCrystal Orbital Diagram   version {__version__}"


# The methods paper of CrystOD itself. Dependency citations (PySCF,
# ISOTROPY, AMPLIMODES, ...) are printed by the feature that uses them;
# this one covers CrystOD's own construction.
CRYSTOD_CITATION_LINES = (
    "If you use CrystOD in your research, please cite:",
    "  H. Koiso and Y. Mochizuki et al., Phys. Rev. B 110, 064104 (2024)."
    " https://doi.org/10.1103/PhysRevB.110.064104",
)
CRYSTOD_CITATION = "\n".join(CRYSTOD_CITATION_LINES)
CRYSTOD_CITATION_HTML = (
    "If you use CrystOD in your research, please cite: "
    "H. Koiso and Y. Mochizuki <i>et al.</i>, "
    "<a href=\"https://doi.org/10.1103/PhysRevB.110.064104\" "
    "target=\"_blank\">Phys. Rev. B <b>110</b>, 064104 (2024)</a>."
)


def print_crystod_citation() -> None:
    """Print the CrystOD citation block (closing footer of a run)."""
    print("\n" + CRYSTOD_CITATION)


def require_pyscf_or_exit(feature: str) -> None:
    """Abort with a one-line ``ERROR:`` when a ``--pyscf`` run lacks PySCF.

    PySCF is an optional dependency since v0.4.0 (``pip install
    "CrystOD[quantum]"``). The check runs before any analysis is dispatched,
    so the user sees the remedy instead of a traceback.
    """
    from .._optional import require_pyscf

    try:
        require_pyscf(feature)
    except ImportError as exc:
        raise SystemExit(f"ERROR: {exc}") from None


class ExampleRequested(Exception):
    """Raised by the ``--example`` action while argparse reads the command line.

    The parse stops at that option, so a bare ``crystod-phonon --example``
    needs neither a mode flag nor ``-c``: ``main()`` catches the exception
    and hands the request to :func:`run_example`.

    Attributes:
        name: The example name, or ``"list"`` when none was given.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name


class _ExampleAction(Action):
    """``--example [NAME]``: stop parsing and report the request."""

    def __call__(self, parser, namespace, values, option_string=None):
        raise ExampleRequested(values)


def add_example_argument(parser: ArgumentParser, command: str) -> None:
    """Register ``--example [NAME]`` on the parser of ``command``.

    argparse owns the option (instead of a pre-scan of ``sys.argv`` for the
    literal spelling), so ``--example NAME``, ``--example=NAME`` and the
    unambiguous abbreviations argparse accepts everywhere else all take
    the same path, and the option is documented in ``--help``. The names
    come from :mod:`crystod.examples`, which imports only the standard
    library, so ``--help`` stays light.
    """
    from ..examples import list_examples

    names = ", ".join(example.name for example in list_examples(command))
    parser.add_argument(
        "--example",
        nargs="?",
        const="list",
        default=None,
        metavar="NAME",
        action=_ExampleAction,
        help="Run a bundled example: copy its input files into the current\n"
        "directory, print the equivalent ordinary command line and run it\n"
        "(further options are passed on). --example alone, or --example\n"
        f"list, lists the names: {names}.",
    )


def _without_example_option(argv: list[str], name: str) -> list[str]:
    """``argv`` minus the ``--example [NAME]`` tokens argparse consumed.

    argparse already resolved the option, so it is the first token spelling
    ``--example`` or an (unambiguous) abbreviation of it; the name follows
    as a token of its own unless it came after ``=`` or was left out.
    """
    for index, token in enumerate(argv):
        flag, separator, _ = token.partition("=")
        if len(flag) > 2 and "--example".startswith(flag):
            rest = argv[index + 1:]
            if not separator and rest and rest[0] == name:
                rest = rest[1:]
            return argv[:index] + rest
    return list(argv)


def run_example(command: str, name: str, argv: list[str]) -> list[str]:
    """Answer ``--example`` for ``command``: list the examples, or set one up.

    ``argv`` is the command line that carried the option. With ``name ==
    "list"`` the table of :func:`crystod.examples.format_example_list` is
    printed and the process exits 0. Otherwise the input files of the
    example are copied into the current directory (a file already there is
    kept when it is identical to the bundled one and never overwritten when
    it differs), one line per file and a ``Running:`` line with the
    equivalent ordinary command are printed, and the argv to run is
    returned: the arguments of the example followed by whatever else the
    user gave.

    Raises:
        SystemExit: Code 0 after the listing; an ``ERROR:`` one-liner for an
            unknown name or a differing file in the way.
    """
    import shlex

    from ..examples import copy_example_files, format_example_list, get_example

    if name == "list":
        print(format_example_list(command))
        raise SystemExit(0)
    try:
        example = get_example(command, name)
    except KeyError as exc:
        raise SystemExit(f"ERROR: {exc.args[0]}") from None

    remaining = _without_example_option(argv, name)
    destination = Path.cwd()
    present = {target for _, target in example.files if (destination / target).exists()}
    try:
        paths = copy_example_files(example, destination)
    except FileExistsError as exc:
        raise SystemExit(f"ERROR: {exc}") from None
    for path in paths:
        if path.name in present:
            print(f"Kept {path.name} (identical to the bundled example input)")
        else:
            print(f"Wrote {path.name} (bundled example input)")
    command_line = " ".join([example.command_line,
                             *(shlex.quote(token) for token in remaining)])
    print(f"Running: {command_line}\n")
    return [*example.argv, *remaining]


def add_cell_argument(
    parser: ArgumentParser, help_suffix: str = "", default: str | None = "POSCAR"
) -> None:
    """Add the phonopy-style structure-file option (-c/--cell, alias --poscar).

    ``default=None`` is the sentinel form: the command can then tell "-c was
    given" from "-c was left out" (crystod-phonon --modulation needs that to
    choose between the structure file and phonopy_params.yaml) and has to
    substitute the documented ``POSCAR`` default itself.
    """
    parser.add_argument(
        "-c",
        "--cell",
        "--poscar",
        dest="cell",
        default=default,
        metavar="FILE",
        help=f"Crystal structure file in VASP POSCAR format (default: POSCAR).{help_suffix}",
    )


def add_output_argument(parser: ArgumentParser, help_text: str) -> None:
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        default=None,
        metavar="FILE",
        help=help_text,
    )
