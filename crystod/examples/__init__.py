"""Bundled examples: the ``--example`` inputs and the example directories.

Two kinds of examples ship inside the package, so that they work right
after ``pip install CrystOD``, without cloning the repository.

**1. The inputs of the ``--example`` flags.** Five small files (about
94 kB in total) lie directly in this directory:

======================  =====================================================
file                    origin
======================  =====================================================
``221_PPOSCAR_ScF3``    cubic ScF3, Pm-3m (example/test_POSCARs)
``221_PPOSCAR_SrTiO3``  cubic SrTiO3, Pm-3m (example/test_POSCARs)
``FORCE_SETS_SrTiO3``   phonopy force sets of SrTiO3, 4x4x4 supercell
                        (example/28_phonon_irrep/SrTiO3_Pm-3m/FORCE_SETS)
``XYZ_CH4.xyz``         methane (example/test_XYZs)
``XYZ_NH3.xyz``         ammonia (example/test_XYZs)
======================  =====================================================

``crystod``, ``crystod-phonon``, ``crystod-mol``, ``crystod-bz``,
``crystod-xrd``, ``crystod-search`` and ``crystod-slab`` accept
``--example NAME``: the files of that example (the searches have none) are
copied into the current directory, the equivalent ordinary command line is printed, and that command
runs. ``--example`` without a name (or ``--example list``) lists the names.

The registry below is the single source of truth for the CLI, the
tutorials and the MCP server::

    from crystod.examples import get_example, example_path

    example = get_example("crystod-phonon", "SrTiO3")
    example.argv          # ('--irreps', '--dim', '4 4 4', '-c', '221_PPOSCAR_SrTiO3')
    example_path("221_PPOSCAR_ScF3")   # absolute Path inside the installed package

**2. The example directories.** Copies of selected directories of the
repository's ``example/`` (``01_wigner_d/`` ... ``53_slab/``, each with
its README and the inputs its commands read, at the same relative paths)
and of the shared ``test_POSCARs/`` and ``test_XYZs/``, which the READMEs
reach as ``../test_POSCARs/...``. Every copied file is byte-identical to
the file of the same path under ``example/``; which files ship is chosen in
``example/PACKAGED`` of the repository, and ``script/sync_packaged_examples.py``
copies them here and writes ``index.json`` (the list of directories, their
files, sizes and the other directories each README uses) and ``README``
(the same list as plain text). The README commands write their outputs
next to their inputs, so a directory is copied into a working directory
first and never run inside the installed package (a destination inside
the package is refused): the console script
``crystod-example`` lists the directories and copies them (into
``./example`` by default), the functions below do the same from Python::

    from crystod.examples import list_example_dirs, copy_example_dirs

    [d.name for d in list_example_dirs()][:2]   # ['01_wigner_d', '02_salc']
    copy_example_dirs([29], "work")
    # work/29_phonon_activity/, plus work/28_phonon_irrep/ and
    # work/30_phonon_fatband/ (its README uses their files) and the shared
    # work/test_POSCARs/ and work/test_XYZs/

A directory is named by its number (``17`` or ``"17"``) or its name
(``"17_isotropy_subgroup"``). When ``index.json`` is missing (a source
tree that has not been synchronized), :func:`list_example_dirs` is empty.
"""

from __future__ import annotations

import filecmp
import json
import re
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Example:
    """One runnable example of a command.

    Attributes:
        name: The name given to ``--example`` (``ScF3_d``, ``SrTiO3``, ...).
        command: The console script the example belongs to (``crystod``,
            ``crystod-phonon``, ``crystod-mol``, ``crystod-bz``, ``crystod-xrd``,
            ``crystod-search``, ``crystod-slab``).
        files: ``(bundled name, name in the working directory)`` pairs of the
            input files the example needs; usually both names are equal, the
            exception being the phonopy force sets, which phonopy reads as
            ``FORCE_SETS`` next to the unit cell.
        argv: The ordinary command-line arguments that reproduce the example
            once the files are in the working directory.
        description: One line saying what the example shows.
    """

    name: str
    command: str
    files: tuple[tuple[str, str], ...]
    argv: tuple[str, ...]
    description: str

    @property
    def command_line(self) -> str:
        """The equivalent command as one shell line (quoted where needed)."""
        import shlex

        return " ".join([self.command, *(shlex.quote(token) for token in self.argv)])


_SCF3 = (("221_PPOSCAR_ScF3", "221_PPOSCAR_ScF3"),)
_SRTIO3 = (("221_PPOSCAR_SrTiO3", "221_PPOSCAR_SrTiO3"),)
_SRTIO3_PHONON = (
    ("221_PPOSCAR_SrTiO3", "221_PPOSCAR_SrTiO3"),
    ("FORCE_SETS_SrTiO3", "FORCE_SETS"),
)

EXAMPLES: tuple[Example, ...] = (
    # crystod (the main command)
    Example(
        "ScF3_d", "crystod", _SCF3,
        ("-c", "221_PPOSCAR_ScF3", "--element", "Sc", "--orbital", "d"),
        "Sc 3d crystal-orbital (SALC) irreps of cubic ScF3 at every special k point",
    ),
    Example(
        "SrTiO3_d", "crystod", _SRTIO3,
        ("-c", "221_PPOSCAR_SrTiO3", "--element", "Ti", "--orbital", "d"),
        "Ti 3d crystal-orbital irreps of cubic SrTiO3 (eg/t2g splitting at Gamma)",
    ),
    Example(
        "ScF3_diagram", "crystod", _SCF3,
        ("--diagram", "-c", "221_PPOSCAR_ScF3", "--co-left", "Sc", "--co-right", "F3"),
        "extended-Hueckel crystal-orbital diagram of ScF3, one HTML page per k point",
    ),
    # crystod-phonon
    Example(
        "SrTiO3", "crystod-phonon", _SRTIO3_PHONON,
        ("--irreps", "--dim", "4 4 4", "-c", "221_PPOSCAR_SrTiO3"),
        "ISO-IR irrep labels of the phonon modes of cubic SrTiO3 (writes phonon_irreps.yaml)",
    ),
    Example(
        "SrTiO3_subgroup", "crystod-phonon", _SRTIO3_PHONON,
        ("--subgroup", "--dim", "4 4 4", "-c", "221_PPOSCAR_SrTiO3", "--qpoint", "R"),
        "space groups the imaginary R-point phonon of cubic SrTiO3 can condense into",
    ),
    # crystod-mol
    Example(
        "CH4", "crystod-mol", (("XYZ_CH4.xyz", "XYZ_CH4.xyz"),),
        ("--diagram", "--xyz", "XYZ_CH4.xyz"),
        "MO diagram of methane from symmetry and overlap (writes MolOD_XYZ_CH4.html)",
    ),
    Example(
        "NH3", "crystod-mol", (("XYZ_NH3.xyz", "XYZ_NH3.xyz"),),
        ("--diagram", "--xyz", "XYZ_NH3.xyz"),
        "MO diagram of ammonia (writes MolOD_XYZ_NH3.html)",
    ),
    # crystod-bz
    Example(
        "ScF3", "crystod-bz", _SCF3,
        ("-c", "221_PPOSCAR_ScF3"),
        "3D Brillouin zone of cubic ScF3 with the seekpath k path (writes BZ_221_PPOSCAR_ScF3.html)",
    ),
    # crystod-xrd
    Example(
        "ScF3", "crystod-xrd", _SCF3,
        ("-c", "221_PPOSCAR_ScF3"),
        "powder XRD pattern of cubic ScF3 for Cu K-alpha (writes XRD_221_PPOSCAR_ScF3_CuKa.txt/.pdf)",
    ),
    Example(
        "SrTiO3", "crystod-xrd", _SRTIO3,
        ("-c", "221_PPOSCAR_SrTiO3", "--xraytype", "CuKa1", "--peak-profile", "gaussian"),
        "monochromatic Cu K-alpha1 pattern of cubic SrTiO3 with Gaussian peaks",
    ),
    # crystod-search (no input files; needs a Materials Project API key)
    Example(
        "SrTiO3", "crystod-search", (),
        ("SrTiO3",),
        "every SrTiO3 polymorph of the Materials Project, starred when observed",
    ),
    Example(
        "Sr-Ti-O", "crystod-search", (),
        ("Sr-Ti-O", "--experimental"),
        "the experimentally observed Sr-Ti-O ternary compounds",
    ),
    Example(
        "mp-5229", "crystod-search", (),
        ("--get", "mp-5229"),
        "download cubic SrTiO3 (writes POSCAR_SrTiO3_Pm-3m_mp-5229)",
    ),
    # crystod-slab
    Example(
        "SrTiO3", "crystod-slab", _SRTIO3,
        ("-c", "221_PPOSCAR_SrTiO3", "--max-index", "1"),
        "symmetrically distinct SrTiO3 surfaces up to (1 1 1), every termination with its "
        "polarity",
    ),
    Example(
        "ScF3", "crystod-slab", _SCF3,
        ("-c", "221_PPOSCAR_ScF3", "--index", "1", "0", "0", "--symmetrize"),
        "symmetrized (non-stoichiometric, charged) terminations of ScF3 (1 0 0)",
    ),
)


def example_dir() -> Path:
    """Directory holding the bundled example files."""
    return Path(__file__).resolve().parent


def example_path(filename: str) -> Path:
    """Absolute path of one bundled file (``example_path("221_PPOSCAR_ScF3")``).

    Raises:
        FileNotFoundError: No bundled file of that name.
    """
    path = example_dir() / filename
    if not path.is_file():
        # the --example inputs only: not the generated index of the example
        # directories, nor a stray .DS_Store of a source tree
        bundled = ", ".join(sorted(p.name for p in example_dir().iterdir()
                                   if p.is_file() and not p.name.startswith(("__", "."))
                                   and p.name not in (INDEX_FILE, INDEX_README)))
        raise FileNotFoundError(
            f"no bundled example file {filename!r}; the package ships: {bundled}")
    return path


def list_examples(command: str | None = None) -> list[Example]:
    """The registered examples, optionally those of one command only."""
    return [ex for ex in EXAMPLES if command is None or ex.command == command]


def get_example(command: str, name: str) -> Example:
    """Look up one example by command and name.

    Raises:
        KeyError: The command has no example of that name; the message lists
            the names it does have.
    """
    for example in EXAMPLES:
        if example.command == command and example.name == name:
            return example
    names = ", ".join(ex.name for ex in list_examples(command)) or "(none)"
    raise KeyError(f"{command} has no example named {name!r}; available: {names}")


def format_example_list(command: str) -> str:
    """A ``--example list`` table for one command."""
    examples = list_examples(command)
    width = max((len(ex.name) for ex in examples), default=4)
    lines = [f"Examples bundled with {command} (run one with --example NAME):", ""]
    for ex in examples:
        lines.append(f"  {ex.name:<{width}}  {ex.description}")
        lines.append(f"  {'':<{width}}  = {ex.command_line}")
    lines += ["", "More examples, a directory per feature with its README and inputs: "
              "crystod-example"]
    return "\n".join(lines)


def copy_example_files(example: Example, destination: str | Path = ".") -> list[Path]:
    """Copy the input files of ``example`` into ``destination``.

    A file that already exists there with identical content is left alone;
    one with different content is never overwritten -- every target is
    checked before anything is written, and the run stops with a message
    naming the offending file, so a user's own ``FORCE_SETS`` or ``POSCAR``
    is safe and no half-copied example is left behind.

    Returns:
        The paths written (or found identical), in registry order.

    Raises:
        FileExistsError: A destination file exists with different content.
    """
    destination = Path(destination)
    pending: list[tuple[Path, Path]] = []
    for bundled, target_name in example.files:
        source = example_path(bundled)
        target = destination / target_name
        if target.exists() and not filecmp.cmp(source, target, shallow=False):
            raise FileExistsError(
                f"{target} already exists and differs from the bundled "
                f"example file; move it away or run the example in an "
                f"empty directory."
            )
        pending.append((source, target))
    written: list[Path] = []
    for source, target in pending:
        if not target.exists():
            shutil.copyfile(source, target)
        written.append(target)
    return written


# ---------------------------------------------------------------------------
# The example directories (copies of example/NN_name/ listed in index.json)
# ---------------------------------------------------------------------------

INDEX_FILE = "index.json"
"""The list of the packaged example directories, written by the sync script."""
INDEX_README = "README"
"""The same list as plain text, written by the sync script."""
_INDEX_FORMAT = 1
# the directory names index.json may hold (one plain path component each)
_EXAMPLE_NAME = re.compile(r"[0-9]{2}_[A-Za-z0-9_.+-]+")
_SHARED_NAME = re.compile(r"test_[A-Za-z0-9_]+")
# how many differing files an ExampleConflictError names before "and N more"
_CONFLICTS_NAMED = 10


@dataclass(frozen=True)
class ExampleDirectory:
    """One packaged example directory (or one of the shared input directories).

    Attributes:
        name: The directory name, ``"17_isotropy_subgroup"``
            (``"test_POSCARs"`` for a shared directory).
        number: The number in front of the name (the testsuite section of
            the same feature), ``17``; ``None`` for a shared directory.
        description: One line saying what the example shows (its line in
            ``example/README``).
        requires: The other example directories whose files the README of
            this one uses (``../28_phonon_irrep/...``); they are copied along.
        files: The packaged files, as ``/``-separated paths relative to this
            package directory (= relative to the repository's ``example/``),
            ``"17_isotropy_subgroup/README"``.
        bytes: The total size of ``files``.
    """

    name: str
    number: int | None
    description: str
    requires: tuple[str, ...]
    files: tuple[str, ...]
    bytes: int


class ExampleConflictError(FileExistsError):
    """Files of a destination differ from the packaged example files.

    Raised by :func:`copy_example_dirs` before anything is written.

    Attributes:
        paths: The destination paths in the way: files with a different
            content, or a directory, file or dangling link where a file or a
            directory of the example has to go.
    """

    def __init__(self, paths: Iterable[Path]) -> None:
        self.paths = tuple(paths)
        count = len(self.paths)
        named = ", ".join(str(path) for path in self.paths[:_CONFLICTS_NAMED])
        if count > _CONFLICTS_NAMED:
            named += f" and {count - _CONFLICTS_NAMED} more"
        super().__init__(
            f"{named} already exist{'s' if count == 1 else ''} and "
            f"differ{'s' if count == 1 else ''} from the packaged example; nothing "
            f"was copied (move {'it' if count == 1 else 'them'} away or copy into "
            "another directory)"
        )


def _index_path() -> Path:
    return example_dir() / INDEX_FILE


def _checked_file(index: Path, name: str, path: object) -> str:
    """A file path of the index, refused unless it is a plain relative path."""
    parts = path.split("/") if isinstance(path, str) else [""]
    if (any(part in ("", ".", "..") for part in parts) or "\\" in path
            or parts[0] != name):
        raise ValueError(f"{index} is malformed: {path!r} is not a relative path "
                         f"inside {name}/")
    return path


def _read_index() -> tuple[tuple[ExampleDirectory, ...], tuple[ExampleDirectory, ...]]:
    """The example directories and the shared directories of ``index.json``."""
    index = _index_path()
    if not index.is_file():
        return (), ()
    try:
        data = json.loads(index.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"{index} is not valid JSON ({exc})") from None
    if not isinstance(data, dict) or data.get("format") != _INDEX_FORMAT:
        found = data.get("format") if isinstance(data, dict) else None
        raise ValueError(f"{index} has format {found!r}; this CrystOD reads format "
                         f"{_INDEX_FORMAT} (reinstall CrystOD)")

    def malformed(problem: str) -> ValueError:
        return ValueError(f"{index} is malformed: {problem}")

    def integer(value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    def entry(raw: dict, shared: bool) -> ExampleDirectory:
        name = raw["name"]
        if not isinstance(name, str) or not (_SHARED_NAME if shared
                                             else _EXAMPLE_NAME).fullmatch(name):
            kind = "a shared directory (test_name)" if shared else "an example directory (NN_name)"
            raise malformed(f"{name!r} is not the name of {kind}")
        number = None if shared else raw.get("number")
        if not shared and number is None:
            number = int(name[:2])
        if not shared and (not integer(number) or number != int(name[:2])):
            raise malformed(f"the number of {name} is {number!r}, not {int(name[:2])}")
        size = raw.get("bytes", 0)
        if not integer(size):
            raise malformed(f"the bytes of {name} are {size!r}, not a byte count")
        files = raw.get("files", [])
        requires = [] if shared else raw.get("requires", [])
        if not isinstance(files, list) or not isinstance(requires, list):
            raise malformed(f"the files and requires of {name} must be lists")
        return ExampleDirectory(
            name=name,
            number=number,
            description=str(raw.get("description", "")),
            requires=tuple(requires),
            files=tuple(_checked_file(index, name, path) for path in files),
            bytes=size,
        )

    try:
        examples = sorted((entry(raw, False) for raw in data.get("examples", ())),
                          key=lambda example: (example.number, example.name))
        shared = tuple(entry(raw, True) for raw in data.get("shared", ()))
    except (KeyError, TypeError, AttributeError) as exc:
        raise malformed(f"{type(exc).__name__}: {exc}") from None
    names = [directory.name for directory in (*examples, *shared)]
    twice = sorted({name for name in names if names.count(name) > 1})
    if twice:
        raise malformed(f"{', '.join(twice)} listed more than once")
    for example in examples:
        for name in example.requires:
            if name == example.name or not any(other.name == name for other in examples):
                raise malformed(f"{example.name} requires {name!r}, which is not another "
                                "example directory of the index")
    return tuple(examples), shared


def list_example_dirs() -> tuple[ExampleDirectory, ...]:
    """The packaged example directories, by number.

    Empty when ``index.json`` is missing (a source tree that has not been
    synchronized with ``script/sync_packaged_examples.py``).
    """
    return _read_index()[0]


def list_shared_dirs() -> tuple[ExampleDirectory, ...]:
    """The shared input directories (``test_POSCARs``, ``test_XYZs``).

    :func:`copy_example_dirs` copies them along with any example, because
    the READMEs read them as ``../test_POSCARs/...``.
    """
    return _read_index()[1]


def _no_such_dir(examples: tuple[ExampleDirectory, ...], key: str | int) -> KeyError:
    if not examples:
        return KeyError(
            f"no example directory {key!r}: this installation of CrystOD has none "
            f"({_index_path()} is missing); they are in example/ of the source "
            "repository")
    names = ", ".join(example.name for example in examples)
    return KeyError(f"no packaged example directory {key!r}; available: {names}")


def _find(examples: tuple[ExampleDirectory, ...], key: str | int) -> ExampleDirectory:
    """The example of ``key``: a number (17, "17", "017") or a name."""
    if isinstance(key, bool) or not isinstance(key, (int, str)):
        raise TypeError(f"an example directory is named by a number or a name, "
                        f"not {key!r}")
    text = key if isinstance(key, str) else str(key)
    # "17_isotropy_subgroup/" as the shell completes a directory name
    text = text.strip().rstrip("/")
    if text.isascii() and text.isdigit():
        number = int(text)
        matches = [example for example in examples if example.number == number]
    else:
        matches = [example for example in examples if example.name == text]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(example.name for example in matches)
        raise KeyError(f"example number {key!r} is ambiguous: {names}; give the name")
    raise _no_such_dir(examples, key)


def get_example_dir(key: str | int) -> ExampleDirectory:
    """Look up one packaged example directory by number or name.

    ``get_example_dir(17)``, ``get_example_dir("17")`` and
    ``get_example_dir("17_isotropy_subgroup")`` are the same directory.

    Raises:
        KeyError: No such directory; the message lists the names there are.
    """
    return _find(list_example_dirs(), key)


def example_dir_path(key: str | int) -> Path:
    """Absolute path of one packaged example directory (inside the package).

    Read the files there, but run the README commands in a copy
    (:func:`copy_example_dirs`): they write their outputs next to the inputs.

    Raises:
        KeyError: No such directory.
    """
    return example_dir() / get_example_dir(key).name


def _resolve(examples: tuple[ExampleDirectory, ...], keys, requires: bool
             ) -> tuple[ExampleDirectory, ...]:
    if isinstance(keys, (str, int)):
        keys = [keys]
    resolved: list[ExampleDirectory] = []
    for key in keys:
        example = _find(examples, key)
        if example not in resolved:
            resolved.append(example)
    if requires:
        by_name = {example.name: example for example in examples}
        position = 0
        # breadth first: the requested directories, then the ones they use,
        # then the ones those use, ...
        while position < len(resolved):
            for name in resolved[position].requires:
                if name not in by_name:
                    raise KeyError(f"{resolved[position].name} uses {name}, which is "
                                   "not packaged (reinstall CrystOD)")
                if by_name[name] not in resolved:
                    resolved.append(by_name[name])
            position += 1
    return tuple(resolved)


def resolve_example_dirs(keys, *, requires: bool = True) -> tuple[ExampleDirectory, ...]:
    """The example directories of ``keys`` and, with ``requires``, the ones they use.

    ``keys`` are numbers or names (a single one is accepted as well). Each
    directory appears once: the requested ones in the order given, then the
    directories their READMEs use (transitively).

    Raises:
        KeyError: A key names no packaged directory.
    """
    return _resolve(list_example_dirs(), keys, requires)


def _is_bytecode(path: str) -> bool:
    # pip byte-compiles the shipped .py files when it installs the wheel
    return "__pycache__" in path.split("/") or path.endswith((".pyc", ".pyo"))


def _blocked_by(destination: Path, path: str) -> Path | None:
    """A non-directory in ``destination`` where a parent directory of ``path`` goes."""
    parts = path.split("/")
    for depth in range(1, len(parts)):
        parent = destination.joinpath(*parts[:depth])
        if parent.is_symlink() or parent.exists():
            if not parent.is_dir():
                return parent
    return None


def _copy_plan(dirs: tuple[ExampleDirectory, ...], shared_dirs: tuple[ExampleDirectory, ...],
               destination: Path) -> list[tuple[Path, Path, bool]]:
    """``(source, target, identical)`` of every file to copy, after every check."""
    if not dirs:
        return []
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(f"{destination} exists and is not a directory")
    root = example_dir()
    # the README commands write next to their inputs: never into the package
    package = root.parent
    resolved = destination.resolve()
    if resolved == package or package in resolved.parents:
        raise ValueError(f"{destination} is inside the installed crystod package ({package}); "
                         "copy the examples into a working directory outside it "
                         "(crystod-example ... -o DIR)")
    paths: list[str] = []
    for directory in (*dirs, *shared_dirs):
        for path in directory.files:
            if path not in paths and not _is_bytecode(path):
                paths.append(path)
    missing = [path for path in paths if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(
            f"the installed package lacks {len(missing)} file(s) listed in {INDEX_FILE} "
            f"({', '.join(missing[:3])}{' ...' if len(missing) > 3 else ''}); "
            "reinstall CrystOD")
    plan: list[tuple[Path, Path, bool]] = []
    conflicts: list[Path] = []
    for path in paths:
        source = root / path
        target = destination.joinpath(*path.split("/"))
        blocker = _blocked_by(destination, path)
        if blocker is not None:
            if blocker not in conflicts:
                conflicts.append(blocker)
        elif target.is_dir() or (target.is_symlink() and not target.exists()):
            conflicts.append(target)
        elif target.exists():
            if filecmp.cmp(source, target, shallow=False):
                plan.append((source, target, True))
            else:
                conflicts.append(target)
        else:
            plan.append((source, target, False))
    if conflicts:
        raise ExampleConflictError(conflicts)
    return plan


def copy_example_dirs(keys, destination: str | Path, *, requires: bool = True,
                      shared: bool = True) -> list[Path]:
    """Copy packaged example directories into ``destination``.

    Every file of each directory of ``resolve_example_dirs(keys,
    requires=requires)`` and, with ``shared``, of every shared directory
    (``test_POSCARs``, ``test_XYZs``) goes to ``destination/<the same
    relative path>``: ``copy_example_dirs([17], "example")`` gives
    ``example/17_isotropy_subgroup/README``. A file already there with the
    same content is kept. If any file there differs, nothing is written --
    every target is checked first -- so a user's own files are never
    overwritten and no half-copied example is left behind. Byte-compiled
    files (``__pycache__/``, ``*.pyc``) are never copied.

    Returns:
        The paths of the copied files, written or found identical, in the
        order of the directories; empty when ``keys`` is empty.

    Raises:
        KeyError: A key names no packaged directory.
        ExampleConflictError: Files in ``destination`` differ from the
            packaged ones (a subclass of ``FileExistsError``).
        NotADirectoryError: ``destination`` is a file.
        ValueError: ``destination`` lies inside the installed ``crystod``
            package (``example_dir()`` and below included), or ``index.json``
            is malformed.
    """
    examples, shared_dirs = _read_index()
    dirs = _resolve(examples, keys, requires)
    destination = Path(destination)
    plan = _copy_plan(dirs, shared_dirs if shared else (), destination)
    for source, target, identical in plan:
        if not identical:
            target.parent.mkdir(parents=True, exist_ok=True)
            # a fresh file with the user's default permissions (copy2 would
            # carry over read-only modes of a system-wide installation)
            shutil.copyfile(source, target)
    return [target for _, target, _ in plan]


__all__ = [
    "Example",
    "EXAMPLES",
    "example_dir",
    "example_path",
    "list_examples",
    "get_example",
    "format_example_list",
    "copy_example_files",
    "ExampleDirectory",
    "ExampleConflictError",
    "list_example_dirs",
    "list_shared_dirs",
    "get_example_dir",
    "example_dir_path",
    "resolve_example_dirs",
    "copy_example_dirs",
]
