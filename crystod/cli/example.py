"""crystod-example: list the packaged example directories and copy them out.

The package carries copies of selected directories of the repository's
``example/`` (``NN_name/`` with its README and inputs) together with the
shared ``test_POSCARs/`` and ``test_XYZs/`` (backed by
:mod:`crystod.examples`). Without arguments they are listed; with numbers
or names, or ``--all``, they are copied into a working directory
(``./example`` by default), along with the other directories whose files
their READMEs use, because the README commands write their outputs next to
their inputs.
"""

from __future__ import annotations

import shlex
import sys
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path

from .common import CRYSTOD_CITATION, banner

DEFAULT_DESTINATION = "example"

desc = """\
Copy the example directories packaged with CrystOD into a working directory.

Each directory (NN_name, NN being the section of testsuite.py that tests the
same feature) holds a README with the commands to run and their output, and
the input files those commands read; it is a copy of example/NN_name of the
CrystOD source tree, without the data that is not in the pip package (large
data, test fixtures, unpublished data). The READMEs read the shared
structures as ../test_POSCARs/... and ../test_XYZs/..., so these two
directories are copied along with every example, and so is any example
directory whose files a README uses (29_phonon_activity brings
28_phonon_irrep and 30_phonon_fatband).

A file already in place with the same content is kept; if any file differs,
nothing is copied and the differing files are named.

Without arguments (or with --list) the packaged directories are listed.
The --example option of crystod, crystod-phonon, crystod-mol, crystod-bz,
crystod-xrd and crystod-search is separate: it copies its input files, if
any, into the current directory and runs one command.

# Command Examples:
crystod-example                    (list the packaged directories)
crystod-example 17                 (./example/17_isotropy_subgroup + the shared inputs)
crystod-example 17_isotropy_subgroup 28 -o work
crystod-example 29                 (with 28_phonon_irrep and 30_phonon_fatband)
crystod-example --all -o ~/crystod-examples
"""


def build_parser() -> ArgumentParser:
    from .. import __version__

    parser = ArgumentParser(
        prog="crystod-example",
        description=f"{banner()}\n\n{desc}",
        epilog=CRYSTOD_CITATION,
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"CrystOD {__version__}"
    )
    parser.add_argument(
        "names",
        nargs="*",
        metavar="NAME",
        help="Example directories to copy, by number (17) or name\n"
        "(17_isotropy_subgroup).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the packaged example directories (the default without\n"
        "arguments).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Copy every packaged example directory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        default=None,
        metavar="DIR",
        help=f"Directory to copy into (default: ./{DEFAULT_DESTINATION}); created if\n"
        "needed. The directories land in DIR/NN_name, the shared inputs in\n"
        "DIR/test_POSCARs and DIR/test_XYZs. A DIR inside the installed\n"
        "crystod package is refused.",
    )
    return parser


def _size(count: int) -> str:
    """A byte count as B, kB or MB (decimal units)."""
    if count < 1000:
        return f"{count} B"
    if count < 999_500:
        return f"{count / 1000:.0f} kB"
    return f"{count / 1_000_000:.2f} MB"


def _plural(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


def _key_of(example) -> str:
    """The short form to type for an example: its number as in the name."""
    return example.name.split("_", 1)[0] if example.number is not None else example.name


def _shown(path: Path) -> str:
    """A destination path as typed, with a trailing slash (``example/``)."""
    text = str(path)
    return text if text.endswith("/") else text + "/"


def _list() -> None:
    from ..examples import (
        INDEX_FILE,
        example_dir,
        list_example_dirs,
        list_shared_dirs,
        resolve_example_dirs,
    )

    examples = list_example_dirs()
    shared = list_shared_dirs()
    print("\n* Packaged example directories *")
    if not examples:
        print(f"  none: this installation has no example directories "
              f"({example_dir() / INDEX_FILE} is missing).")
        print("  They are in example/ of the CrystOD source repository.")
        return
    width = max(len(example.name) for example in examples)
    sizes = [_size(example.bytes) for example in examples]
    size_width = max(len(size) for size in sizes)
    for example, size in zip(examples, sizes):
        print(f"  {example.name:<{width}}  {size:>{size_width}}  {example.description}".rstrip())

    along = []
    for example in examples:
        used = resolve_example_dirs([example.name])[1:]
        if used:
            along.append((example.name, ", ".join(other.name for other in used)))
    if along:
        print("\n* Copied along (the README uses files of these directories) *")
        for name, used in along:
            print(f"  {name:<{width}}  -> {used}")

    if shared:
        print("\n* Shared inputs (copied with every example) *")
        shared_width = max(len(directory.name) for directory in shared)
        counts = [_plural(len(directory.files), "file") for directory in shared]
        count_width = max(len(count) for count in counts)
        shared_sizes = [_size(directory.bytes) for directory in shared]
        shared_size_width = max(len(size) for size in shared_sizes)
        for directory, count, size in zip(shared, counts, shared_sizes):
            print(f"  {directory.name:<{shared_width}}  {count:>{count_width}}  "
                  f"{size:>{shared_size_width}}")

    first = examples[0]
    total = sum(example.bytes for example in (*examples, *shared))
    count = len(examples)
    print(f"\n{count} {'directory' if count == 1 else 'directories'}, "
          f"{_size(total)} with the shared inputs.")
    print(f"Copy one:  crystod-example {_key_of(first)}   "
          f"(into ./{DEFAULT_DESTINATION}/{first.name})")
    print("Copy all:  crystod-example --all -o DIR")


def _copy(names: list[str], copy_all: bool, output: str | None) -> None:
    from ..examples import (
        INDEX_FILE,
        ExampleConflictError,
        copy_example_dirs,
        example_dir,
        list_example_dirs,
        list_shared_dirs,
        resolve_example_dirs,
    )

    destination = Path(output if output is not None else DEFAULT_DESTINATION)
    try:
        if copy_all:
            keys = [example.name for example in list_example_dirs()]
            if not keys:
                raise SystemExit(
                    "ERROR: this installation of CrystOD has no example directories "
                    f"({example_dir() / INDEX_FILE} is missing); they are in example/ "
                    "of the source repository")
        else:
            keys = names
        requested = resolve_example_dirs(keys, requires=False)
        dirs = resolve_example_dirs(keys)
        shared = list_shared_dirs()
    except KeyError as exc:
        raise SystemExit(f"ERROR: {exc.args[0]}") from None
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from None

    # which targets are there already (kept when identical; a differing one
    # stops the copy before anything is written)
    candidates = [destination.joinpath(*path.split("/"))
                  for directory in (*dirs, *shared) for path in directory.files]
    present = {path for path in candidates if path.is_file()}
    try:
        paths = copy_example_dirs([directory.name for directory in dirs], destination,
                                  requires=False)
    except ExampleConflictError as exc:
        raise SystemExit(f"ERROR: {exc}") from None
    except KeyError as exc:
        raise SystemExit(f"ERROR: {exc.args[0]}") from None
    except (OSError, ValueError) as exc:
        # ValueError: a destination inside the installed package, a broken index
        raise SystemExit(f"ERROR: {exc}") from None

    added = dirs[len(requested):]
    listed = (*added, *shared) if copy_all else (*dirs, *shared)
    width = max((len(directory.name) for directory in listed), default=0)

    print("\n* Example directories *")
    if copy_all:
        print(f"  all {len(requested)} packaged directories ({requested[0].name} ... "
              f"{requested[-1].name})")
    else:
        for directory in requested:
            print(f"  {directory.name:<{width}}  {directory.description}".rstrip())
    if added:
        print("\n* Copied along (the READMEs above use their files) *")
        for directory in added:
            users = [other.name for other in dirs if directory.name in other.requires]
            print(f"  {directory.name:<{width}}  used by {', '.join(users)}")
    if shared:
        print("\n* Shared inputs *")
        for directory in shared:
            print(f"  {directory.name:<{width}}  {_plural(len(directory.files), 'file')}")

    written = [path for path in paths if path not in present]
    kept = len(paths) - len(written)
    print("\n* Output files *")
    if written:
        size = sum(path.stat().st_size for path in written)
        print(f"  Wrote {_plural(len(written), 'file')} ({_size(size)}) under "
              f"{_shown(destination)}")
    if kept:
        print(f"  Kept {_plural(kept, 'identical file')} already under {_shown(destination)}")

    first = shlex.quote(str(destination / requested[0].name))
    if copy_all:
        print(f"\nNext: cd into a directory under {_shown(destination)} (for example "
              f"{first}) and follow its README.")
    elif len(requested) == 1:
        print(f"\nNext: cd {first} and follow its README.")
    else:
        print(f"\nNext: cd {first} and follow its README; the same for the others.")


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(list(argv))
    if args.list and (args.names or args.all):
        parser.error("--list takes neither example names nor --all")
    if args.all and args.names:
        parser.error("give example names or --all, not both")
    if args.output is not None and not (args.names or args.all):
        parser.error("-o/--output needs example names or --all")

    if args.names or args.all:
        _copy(args.names, args.all, args.output)
    else:
        try:
            _list()
        except KeyError as exc:
            raise SystemExit(f"ERROR: {exc.args[0]}") from None
        except ValueError as exc:
            raise SystemExit(f"ERROR: {exc}") from None


if __name__ == "__main__":
    main()
