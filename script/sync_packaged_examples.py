#!/usr/bin/env python
"""Copy the packaged example directories from example/ into crystod/examples/.

The repository's ``example/`` is the canonical copy of the worked examples.
The part of it that ships with ``pip install CrystOD`` is chosen in the plain
text file ``example/PACKAGED`` (repository only, not shipped), and this script
makes ``crystod/examples/{NN_*,test_*}`` an exact, byte-for-byte copy of that
choice; ``crystod-example`` then copies the packaged directories into a
working directory.  The generated files are committed.

Usage:
    python script/sync_packaged_examples.py           # synchronise
    python script/sync_packaged_examples.py --check   # compare only

The script finds the repository from its own path, so it can be run from any
directory.  It needs ``git`` and a checkout of the repository.

example/PACKAGED
----------------
One entry per line; ``#`` starts a comment (at the start of a line or after a
blank), blank lines are ignored::

    budget 7000000                         # max total bytes of the packaged files (raw)
    requires 29_phonon_activity: 28_phonon_irrep 30_phonon_fatband
    01_wigner_d/README                     # a path relative to example/
    01_wigner_d/*.py                       # a shell glob
    28_phonon_irrep/**/FORCE_SETS          # '**' matches across '/'
    test_POSCARs/221_PPOSCAR_ScF3

* ``budget N``: exactly one; the total size of the packaged files (the
  NN_*/ and test_*/ copies, raw bytes) may not exceed N.
* ``requires A: B C ...``: example A uses files of the examples B, C, ...
  (its README reads ../B/...), so ``crystod-example A`` copies them as well.
  The relation is transitive; index.json stores the closure.  A and every B
  must be packaged examples.
* Anything else is a path or a shell glob relative to ``example/``, matched
  against the full relative path: ``*``, ``?`` and ``[...]`` (``[!...]``
  negates) do not match ``/``; a component ``**`` matches across ``/`` (zero
  or more directories in ``a/**/b``, every file below in ``a/**``); as in the
  shell, no wildcard matches the leading ``.`` of a name.  Patterns are
  expanded against ``git ls-files example`` only, never against the disk, so
  untracked and ignored files are never packaged.  A pattern that matches
  nothing is an error.

The first component of every selected path decides where it belongs: an
example directory ``NN_name`` (two digits and an underscore), or one of the
shared directories ``test_POSCARs`` / ``test_XYZs`` (direct children only;
they are copied along with every example).  A selected file that the
package-data globs of pyproject.toml would leave out of the wheel (a dot
file or anything below a dot directory, anything below ``__pycache__/``,
``*.pyc``, ``*.html``, ``*.pdf``, a ``POTCAR*`` file or anything below a
``POTCAR*`` directory) is an error, because the wheel would then differ from
crystod/examples/.

Descriptions come from the index lines of ``example/README``
(``NN_name<blanks>description``, first match).

What a run does
---------------
1. Parse example/PACKAGED and expand it against ``git ls-files example``.
2. Stop (exit 1, an ``ERROR:`` line per problem, nothing written) if a
   selected file has uncommitted changes (``git status --porcelain``), a
   pattern matches nothing, a ``requires`` names something that is not a
   packaged example, a selected path is not packageable, or the total size
   exceeds the budget.
3. Make ``crystod/examples/{NN_*,test_*}`` exactly the selection: copy every
   missing or different file (shutil.copy2) and delete everything else under
   those directories (untracked and ignored files, .DS_Store, __pycache__,
   symbolic links), removing directories that become empty.  The package's
   own files (``__init__.py``, the five flat ``--example`` inputs) and
   anything else outside NN_*/test_* are never touched.
4. Write ``crystod/examples/index.json`` (read by crystod.examples) and the
   plain-text index ``crystod/examples/README``.
5. Print a ``* Packaged examples *`` block: the files and bytes of every
   directory, the raw total and a zlib (level 6) estimate of the compressed
   size.

``--check`` performs steps 1, 2 and 5, writes nothing, and exits 1 on any
drift: a missing, extra or different file, an extra directory, or a stale
index.json or README.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import zlib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "example"
TARGET_DIR = ROOT / "crystod" / "examples"
PACKAGED_FILE = SOURCE_DIR / "PACKAGED"
SOURCE_INDEX = SOURCE_DIR / "README"
INDEX_JSON = "index.json"
INDEX_README = "README"
INDEX_FORMAT = 1

# The shared directories that pyproject.toml packages ("test_POSCARs/*",
# "test_XYZs/*": direct children only).  Keep the two lists in step.
SHARED_DIRS = ("test_POSCARs", "test_XYZs")
# the example directory names that crystod.examples accepts in index.json (keep in step)
EXAMPLE_NAME = re.compile(r"[0-9]{2}_[A-Za-z0-9_.+-]+")
# Top-level entries of crystod/examples/ owned by this script.  "221_PPOSCAR_*"
# has three digits before the underscore, so it does not match.
MANAGED_NAME = re.compile(r"(?:[0-9]{2}_|test_)")
INDEX_LINE = re.compile(r"([0-9]{2}_\S+|test_\S+)\s+(\S.*?)\s*$")
GIT_REGULAR_MODES = ("100644", "100755")
MAX_LISTED = 20


class SyncError(Exception):
    """Problems that stop the run; every message becomes one ``ERROR:`` line."""

    def __init__(self, messages: list[str]) -> None:
        super().__init__("\n".join(messages))
        self.messages = messages


# --------------------------------------------------------------------------
# example/PACKAGED
# --------------------------------------------------------------------------


@dataclass
class PackagedSpec:
    """The parsed content of example/PACKAGED."""

    budget: int | None = None
    patterns: list[tuple[int, str]] = field(default_factory=list)
    requires: dict[str, list[str]] = field(default_factory=dict)
    requires_lines: list[tuple[int, list[str]]] = field(default_factory=list)


_NOT_HIDDEN = r"(?!\.)"
_ANY_DIRECTORIES = rf"(?:{_NOT_HIDDEN}[^/]*/)*"
_ANY_NAME = rf"{_NOT_HIDDEN}[^/]*"


def _translate_component(component: str) -> str:
    """The regex of one path component of a glob (no ``/`` inside)."""
    out: list[str] = []
    i, n = 0, len(component)
    while i < n:
        char = component[i]
        # as in the shell, a wildcard does not match the leading '.' of a name
        guard = _NOT_HIDDEN if i == 0 else ""
        if char == "*":
            while i + 1 < n and component[i + 1] == "*":
                i += 1  # '**' inside a name is a plain '*'
            out.append(guard + "[^/]*")
        elif char == "?":
            out.append(guard + "[^/]")
        elif char == "[":
            j = i + 1
            if j < n and component[j] == "!":
                j += 1
            if j < n and component[j] == "]":
                j += 1
            while j < n and component[j] != "]":
                j += 1
            if j >= n:
                out.append(re.escape(char))
            else:
                body = component[i + 1:j]
                negate = body.startswith("!")
                if negate:
                    body = body[1:]
                body = "".join("\\" + c if c in "\\[]^&~|" else c for c in body)
                # a negated class must not match '/' either ('[!b]' is '[^/b]')
                out.append(guard + "[" + ("^/" if negate else "") + body + "]")
                i = j
        else:
            out.append(re.escape(char))
        i += 1
    return "".join(out)


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate a shell glob on a '/'-separated relative path into a regex.

    ``*``, ``?`` and ``[...]`` (``[!...]`` negates, as in fnmatch) stay within
    one path component.  A component that is exactly ``**`` matches zero or
    more directories (``a/**/b``) or, as the last component, every file below
    (``a/**``); ``**`` inside a name is a plain ``*``.  As in the shell, no
    wildcard matches the leading ``.`` of a name: dot files are selected only
    when the pattern spells the dot.
    """
    parts = pattern.split("/")
    out: list[str] = []
    for index, part in enumerate(parts):
        last = index == len(parts) - 1
        if part == "**":
            out.append(_ANY_DIRECTORIES + (_ANY_NAME if last else ""))
        else:
            out.append(_translate_component(part) + ("" if last else "/"))
    return re.compile("".join(out) + r"\Z")


def parse_packaged(text: str) -> tuple[PackagedSpec, list[str]]:
    """Parse example/PACKAGED; returns the spec and the error messages."""
    spec = PackagedSpec()
    errors: list[str] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = re.sub(r"(?:^|\s)#.*", "", raw).strip()
        if not line:
            continue
        where = f"example/PACKAGED:{number}"
        tokens = line.split()
        if tokens[0] == "budget":
            if len(tokens) != 2 or not re.fullmatch(r"[0-9]+(?:_[0-9]+)*", tokens[1]):
                errors.append(f"{where}: write 'budget N' with N a number of bytes: {raw.strip()}")
            elif spec.budget is not None:
                errors.append(f"{where}: a second 'budget' line")
            else:
                spec.budget = int(tokens[1])
        elif tokens[0] == "requires":
            head, colon, tail = line[len("requires"):].partition(":")
            name = head.strip()
            needed = tail.split()
            if not colon or not name or " " in name or not needed:
                errors.append(f"{where}: write 'requires A: B C ...': {raw.strip()}")
                continue
            for item in [name, *needed]:
                if not EXAMPLE_NAME.fullmatch(item):
                    errors.append(f"{where}: '{item}' is not an example directory name "
                                  "(NN_name)")
            if name in needed:
                errors.append(f"{where}: {name} requires itself")
            spec.requires_lines.append((number, [name, *needed]))
            known = spec.requires.setdefault(name, [])
            for item in needed:
                if item not in known and item != name:
                    known.append(item)
        elif len(tokens) > 1:
            errors.append(f"{where}: one path or pattern per line (paths have no blanks): "
                          f"{raw.strip()}")
        else:
            pattern = tokens[0]
            parts = pattern.split("/")
            if pattern.startswith("/") or "" in parts or "." in parts or ".." in parts:
                errors.append(f"{where}: '{pattern}' is not a clean path relative to example/")
            else:
                spec.patterns.append((number, pattern))
    if spec.budget is None:
        errors.append("example/PACKAGED: no 'budget N' line (the size limit is mandatory)")
    if not spec.patterns:
        errors.append("example/PACKAGED: no path or pattern selects anything")
    return spec, errors


# --------------------------------------------------------------------------
# git
# --------------------------------------------------------------------------


def git(*args: str) -> bytes:
    """Run git in the repository; raise SyncError when it fails."""
    try:
        done = subprocess.run(["git", "--no-optional-locks", "-C", str(ROOT), *args],
                              capture_output=True, check=False)
    except FileNotFoundError:
        raise SyncError(["git is not installed or not on PATH"]) from None
    if done.returncode != 0:
        message = done.stderr.decode(errors="replace").strip().splitlines()
        raise SyncError([f"git {' '.join(args[:2])} failed in {ROOT}: "
                         + (message[0] if message else f"exit {done.returncode}")])
    return done.stdout


def tracked_examples() -> dict[str, str]:
    """``{path relative to example/: git mode}`` of the tracked files."""
    out = git("ls-files", "-s", "-z", "--", "example")
    tracked: dict[str, str] = {}
    for record in out.split(b"\0"):
        if not record:
            continue
        info, _, path = record.partition(b"\t")
        mode = info.split()[0].decode()
        relative = os.fsdecode(path)
        if relative.startswith("example/"):
            tracked[relative[len("example/"):]] = mode
    return tracked


def status_entries() -> list[tuple[str, str]]:
    """``(XY code, path relative to example/)`` of ``git status`` under example/.

    Both sides of a rename are listed.  Untracked files come with code ``??``.
    """
    prefix = os.fsdecode(git("rev-parse", "--show-prefix").strip())
    out = git("status", "--porcelain=v1", "-z", "--untracked-files=all", "--", "example")
    items = out.split(b"\0")
    entries: list[tuple[str, str]] = []
    index = 0
    while index < len(items):
        item = items[index]
        index += 1
        if not item:
            continue
        code = item[:2].decode()
        paths = [os.fsdecode(item[3:])]
        if "R" in code or "C" in code:
            paths.append(os.fsdecode(items[index]))
            index += 1
        for path in paths:
            if prefix and path.startswith(prefix):
                path = path[len(prefix):]
            if path.startswith("example/"):
                entries.append((code, path[len("example/"):]))
    return entries


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


@dataclass
class Directory:
    """One packaged directory (an example or a shared directory)."""

    name: str
    shared: bool
    description: str = ""
    requires: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    bytes: int = 0
    compressed: int = 0

    @property
    def number(self) -> int | None:
        return None if self.shared else int(self.name[:2])


@dataclass
class Selection:
    """Everything a run needs: the packaged directories and their contents."""

    budget: int
    directories: list[Directory]
    contents: dict[str, bytes]
    untracked_matches: list[str]

    @property
    def examples(self) -> list[Directory]:
        return [d for d in self.directories if not d.shared]

    @property
    def shared(self) -> list[Directory]:
        return [d for d in self.directories if d.shared]

    @property
    def total_bytes(self) -> int:
        return sum(d.bytes for d in self.directories)

    @property
    def total_compressed(self) -> int:
        return sum(d.compressed for d in self.directories)


def _excluded_from_wheel(path: str) -> bool:
    """True when the package-data rules of pyproject.toml would drop ``path``.

    ``path`` is relative to crystod/examples/.  The include globs never match a
    name that starts with '.'.  setuptools applies the exclude globs
    ("**/__pycache__/*", "**/*.pyc", "**/*.html", "**/*.pdf", "**/POTCAR*",
    "**/.DS_Store") with fnmatch to "crystod/examples/<path>", and fnmatch's
    '*' also matches '/': a glob that ends in '*' drops every file below a
    directory whose name fits (POTCAR_notes/README), while one that ends in a
    fixed text tests the end of the path, i.e. the file name only.  testsuite.py
    section 50 holds the same rule (_pkg_wheel_drops) and checks both against
    fnmatch with the globs read from pyproject.toml; keep the three in step.
    """
    parts = path.split("/")
    return (any(part.startswith(".") for part in parts)              # include globs
            or "__pycache__" in parts[:-1]                           # **/__pycache__/*
            or any(part.startswith("POTCAR") for part in parts)      # **/POTCAR*
            or path.endswith((".pyc", ".html", ".pdf", "/.DS_Store")))


def _shorten(items: list[str], limit: int = 8) -> str:
    shown = ", ".join(items[:limit])
    return shown if len(items) <= limit else f"{shown} and {len(items) - limit} more"


def read_descriptions() -> dict[str, str]:
    """``{directory name: description}`` from the index lines of example/README."""
    descriptions: dict[str, str] = {}
    if not SOURCE_INDEX.is_file():
        return descriptions
    for line in SOURCE_INDEX.read_text(encoding="utf-8").splitlines():
        match = INDEX_LINE.match(line)
        if match:
            descriptions.setdefault(match.group(1), match.group(2))
    return descriptions


def select() -> Selection:
    """Parse example/PACKAGED, expand it and validate the result.

    Raises:
        SyncError: With every problem found (nothing has been written).
    """
    if not PACKAGED_FILE.is_file():
        raise SyncError([f"{PACKAGED_FILE} not found"])
    if not (TARGET_DIR / "__init__.py").is_file() or TARGET_DIR.is_symlink():
        raise SyncError([f"{TARGET_DIR} is not the crystod.examples package directory"])
    spec, errors = parse_packaged(PACKAGED_FILE.read_text(encoding="utf-8"))
    tracked = tracked_examples()
    if not tracked:
        errors.append("git ls-files lists no file under example/")

    selected: set[str] = set()
    compiled: list[re.Pattern[str]] = []
    for number, pattern in spec.patterns:
        regex = glob_to_regex(pattern)
        compiled.append(regex)
        matches = [path for path in tracked if regex.match(path)]
        if not matches:
            hint = ""
            if any(path.startswith(pattern.rstrip("/") + "/") for path in tracked):
                hint = f" (it is a directory; write '{pattern.rstrip('/')}/**' for all of it)"
            elif pattern.startswith("example/"):
                hint = " (paths are relative to example/)"
            errors.append(f"example/PACKAGED:{number}: '{pattern}' matches no file tracked "
                          f"by git{hint}")
        selected.update(matches)

    by_directory: dict[str, list[str]] = {}
    misplaced: list[str] = []
    for path in sorted(selected):
        top, _, rest = path.partition("/")
        if tracked[path] not in GIT_REGULAR_MODES:
            errors.append(f"example/{path} is not a regular file in git (mode {tracked[path]})")
        elif not rest:
            misplaced.append(path)
        elif top in SHARED_DIRS and "/" in rest:
            errors.append(f"example/{path}: only the direct children of {top}/ are packaged "
                          f"(pyproject.toml: \"{top}/*\")")
        elif top not in SHARED_DIRS and not EXAMPLE_NAME.fullmatch(top):
            misplaced.append(path)
        elif _excluded_from_wheel(path):
            errors.append(f"example/{path} would be left out of the wheel by the "
                          "package-data rules of pyproject.toml; do not select it")
        else:
            by_directory.setdefault(top, []).append(path)
    if misplaced:
        errors.append("selected files outside an example directory (NN_name/) or a shared "
                      f"directory ({', '.join(d + '/' for d in SHARED_DIRS)}): "
                      + _shorten([f"example/{p}" for p in misplaced]))

    # requires: both sides must be packaged examples
    packaged_examples = {name for name in by_directory if name not in SHARED_DIRS}
    repository_dirs = {path.partition("/")[0] for path in tracked if "/" in path}
    for number, names in spec.requires_lines:
        for item in names:
            if not EXAMPLE_NAME.fullmatch(item) or item in packaged_examples:
                continue
            if item in repository_dirs:
                errors.append(f"example/PACKAGED:{number}: requires names {item}, which is not "
                              "packaged (no line selects a file of it)")
            else:
                errors.append(f"example/PACKAGED:{number}: requires names {item}, which is not "
                              "an example directory of the repository")

    # uncommitted changes in the selection; untracked files a pattern would match
    dirty: list[str] = []
    untracked: list[str] = []
    for code, path in status_entries():
        matched = path in selected or any(regex.match(path) for regex in compiled)
        if not matched:
            continue
        if code == "??":
            untracked.append(path)
        else:
            dirty.append(f"example/{path} ({code.strip()})")
    if dirty:
        errors.append("uncommitted changes in packaged files (commit or restore them first): "
                      + _shorten(sorted(set(dirty)), limit=MAX_LISTED))

    descriptions = read_descriptions()
    for name in sorted(by_directory):
        if name not in descriptions:
            errors.append(f"example/README has no index line '{name}  <description>'")
    if errors:
        raise SyncError(errors)

    closure: dict[str, list[str]] = {}
    for name in packaged_examples:
        seen: set[str] = set()
        stack = list(spec.requires.get(name, []))
        while stack:
            item = stack.pop()
            if item in seen or item == name:
                continue
            seen.add(item)
            stack.extend(spec.requires.get(item, []))
        closure[name] = sorted(seen, key=lambda item: (int(item[:2]), item))

    contents: dict[str, bytes] = {}
    directories: list[Directory] = []
    for name, files in by_directory.items():
        directory = Directory(name=name, shared=name in SHARED_DIRS,
                              description=descriptions[name],
                              requires=closure.get(name, []), files=sorted(files))
        for path in directory.files:
            data = (SOURCE_DIR / path).read_bytes()
            contents[path] = data
            directory.bytes += len(data)
            directory.compressed += len(zlib.compress(data, 6))
        directories.append(directory)
    directories.sort(key=lambda d: (d.shared, d.number or 0, d.name))
    assert spec.budget is not None
    return Selection(budget=spec.budget, directories=directories, contents=contents,
                     untracked_matches=sorted(untracked))


# --------------------------------------------------------------------------
# generated files
# --------------------------------------------------------------------------


def render_index_json(selection: Selection) -> str:
    """The content of crystod/examples/index.json."""
    index = {
        "format": INDEX_FORMAT,
        "examples": [
            {"name": d.name, "number": d.number, "description": d.description,
             "requires": list(d.requires), "files": list(d.files), "bytes": d.bytes}
            for d in selection.examples
        ],
        "shared": [
            {"name": d.name, "files": list(d.files), "bytes": d.bytes}
            for d in selection.shared
        ],
        "total_bytes": selection.total_bytes,
    }
    return json.dumps(index, indent=1, sort_keys=False) + "\n"


def render_readme(selection: Selection) -> str:
    """The content of crystod/examples/README (plain text)."""
    width = max([29, *(len(d.name) + 2 for d in selection.directories)])
    shared = " and ".join(f"{d.name}/" for d in selection.shared)
    along = (f"The shared directories {shared} are copied along with every example, and an "
             "example" if shared else "An example")
    lines = [
        "CrystOD example directories (packaged copy)",
        "",
        "These are the worked examples of CrystOD, installed with the package so that",
        "they are available without cloning the repository.  Do not run them here:",
        "their commands write output files next to the inputs.  Copy them into a",
        "working directory first:",
        "",
        "  crystod-example                  list the packaged examples",
        "  crystod-example NN [NN ...]      copy examples (number or name) into ./example",
        "  crystod-example NN -o DIR        copy into DIR instead",
        "  crystod-example --all -o DIR     copy every packaged example into DIR",
        "",
        *textwrap.wrap(
            f"then cd into DIR/NN_name and follow its README.  {along} that reads files "
            "of other examples brings those along, so the relative paths in the READMEs "
            "(../test_POSCARs/..., ../NN_name/...) work in the copy.",
            width=78, break_on_hyphens=False),
        "",
        *textwrap.wrap(
            "Every file in the example and shared directories listed below is a "
            "byte-for-byte copy of the same path under example/ of the CrystOD source "
            "tree; only this README and index.json are generated (by "
            "script/sync_packaged_examples.py).  Some data of example/ is not in the pip "
            "package: large data (force-constant files, full MD trajectories), generated "
            "HTML and PDF output, the reference data of the test suite, and the data of "
            "work not yet published (unpublished, not distributed).  It stays in the CrystOD "
            "development tree; the test suite and development need a clone of the "
            "repository.", width=78, break_on_hyphens=False),
        "",
        "Examples",
        "--------",
    ]
    lines += [f"{d.name:<{width}}{d.description}" for d in selection.examples]
    with_requires = [d for d in selection.examples if d.requires]
    if with_requires:
        lines += ["", "Examples that read files of other examples (copied together):"]
        req_width = max(len(d.name) for d in with_requires) + 2
        lines += [f"  {d.name:<{req_width}}{' '.join(d.requires)}" for d in with_requires]
    if selection.shared:
        lines += ["", "Shared directories", "------------------"]
        lines += [f"{d.name:<{width}}{d.description}" for d in selection.shared]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# comparison and synchronisation
# --------------------------------------------------------------------------


@dataclass
class Drift:
    """Differences between crystod/examples/{NN_*,test_*} and the selection."""

    missing: list[str] = field(default_factory=list)    # selected, absent
    different: list[str] = field(default_factory=list)  # other bytes, or not a regular file
    mode: list[str] = field(default_factory=list)       # same bytes, other executable bits
    extra_files: list[str] = field(default_factory=list)  # files and symbolic links
    extra_dirs: list[str] = field(default_factory=list)   # deepest first
    stale: list[str] = field(default_factory=list)        # index.json, README
    unchanged: int = 0

    @property
    def clean(self) -> bool:
        return not (self.missing or self.different or self.mode or self.extra_files
                    or self.extra_dirs or self.stale)


def _exec_bits(path: Path) -> int:
    return path.stat().st_mode & 0o111


def compare(selection: Selection, generated: dict[str, str]) -> Drift:
    """Compare crystod/examples/ with the selection (reads only)."""
    drift = Drift()
    wanted = selection.contents
    needed_dirs = {"/".join(path.split("/")[:k])
                   for path in wanted for k in range(1, path.count("/") + 1)}
    seen: set[str] = set()
    dirs: list[str] = []

    def visit_file(relative: str, path: Path) -> None:
        if relative not in wanted:
            drift.extra_files.append(relative)
            return
        seen.add(relative)
        if path.is_symlink() or not path.is_file():
            drift.different.append(relative)
        elif path.read_bytes() != wanted[relative]:
            drift.different.append(relative)
        elif _exec_bits(path) != _exec_bits(SOURCE_DIR / relative):
            drift.mode.append(relative)
        else:
            drift.unchanged += 1

    for entry in sorted(os.scandir(TARGET_DIR), key=lambda e: e.name):
        if not MANAGED_NAME.match(entry.name):
            continue
        if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
            visit_file(entry.name, Path(entry.path))
            continue
        for current, dirnames, filenames in os.walk(entry.path, followlinks=False):
            base = Path(current).relative_to(TARGET_DIR).as_posix()
            dirs.append(base)
            for name in sorted(dirnames):
                if os.path.islink(os.path.join(current, name)):
                    visit_file(f"{base}/{name}", Path(current, name))
            dirnames[:] = sorted(d for d in dirnames
                                 if not os.path.islink(os.path.join(current, d)))
            for name in sorted(filenames):
                visit_file(f"{base}/{name}", Path(current, name))

    drift.missing = sorted(set(wanted) - seen)
    drift.extra_dirs = sorted((d for d in dirs if d not in needed_dirs),
                              key=lambda d: (-d.count("/"), d))
    for name, text in generated.items():
        path = TARGET_DIR / name
        if (path.is_symlink() or not path.is_file()
                or path.read_bytes() != text.encode("utf-8")):
            drift.stale.append(name)
    return drift


def apply(selection: Selection, generated: dict[str, str], drift: Drift) -> None:
    """Make crystod/examples/ match the selection (removals first, then copies)."""
    for relative in drift.extra_files:
        os.unlink(TARGET_DIR / relative)
    for relative in drift.extra_dirs:
        os.rmdir(TARGET_DIR / relative)
    for relative in drift.different + drift.missing:
        target = TARGET_DIR / relative
        if os.path.lexists(target):
            os.unlink(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE_DIR / relative, target)
    for relative in drift.mode:
        shutil.copymode(SOURCE_DIR / relative, TARGET_DIR / relative)
    for name in drift.stale:
        path = TARGET_DIR / name
        if path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        with open(path, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(generated[name])


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------


def format_summary(selection: Selection) -> str:
    """The ``* Packaged examples *`` block."""
    rows = [(d.name + (" (shared)" if d.shared else ""), len(d.files), d.bytes,
             " ".join(d.requires)) for d in selection.directories]
    width = max([len("Directory"), *(len(row[0]) for row in rows), len("Compressed (zlib -6)")])
    total_files = sum(len(d.files) for d in selection.directories)
    lines = ["* Packaged examples *",
             f" {'Directory':<{width}}  {'Files':>5}  {'Bytes':>11}  Requires"]
    for name, files, size, requires in rows:
        lines.append(f" {name:<{width}}  {files:>5}  {size:>11,}  {requires}".rstrip())
    used = 100.0 * selection.total_bytes / selection.budget if selection.budget else float("inf")
    lines += [
        f" {'Total':<{width}}  {total_files:>5}  {selection.total_bytes:>11,}",
        f" {'Compressed (zlib -6)':<{width}}  {'':>5}  {selection.total_compressed:>11,}",
        f" {'Budget':<{width}}  {'':>5}  {selection.budget:>11,}  ({used:.1f} % used)",
    ]
    return "\n".join(lines)


def _listing(label: str, items: list[str], suffix: str = "") -> list[str]:
    lines = [f" {label:<9} {item}{suffix}" for item in items[:MAX_LISTED]]
    if len(items) > MAX_LISTED:
        lines.append(f" {label:<9} ... and {len(items) - MAX_LISTED} more")
    return lines


def format_drift(drift: Drift, title: str, verbs: dict[str, str]) -> str:
    """A block listing the differences (``--check``) or the changes (a sync)."""
    lines = [f"* {title} *"]
    lines += _listing(verbs["missing"], drift.missing)
    lines += _listing(verbs["different"], drift.different)
    lines += _listing(verbs["mode"], drift.mode)
    lines += _listing(verbs["extra"], drift.extra_files)
    lines += _listing(verbs["extra"], drift.extra_dirs, suffix="/")
    lines += _listing(verbs["stale"], drift.stale)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the synchronisation (or the ``--check``); returns the exit code."""
    parser = argparse.ArgumentParser(
        prog="sync_packaged_examples.py",
        description="Make crystod/examples/{NN_*,test_*} an exact copy of the example "
                    "files selected by example/PACKAGED, and write index.json and README.",
    )
    parser.add_argument("--check", action="store_true",
                        help="compare only: write nothing, exit 1 on any difference")
    args = parser.parse_args(argv)

    try:
        selection = select()
    except SyncError as exc:
        for message in exc.messages:
            print(f"ERROR: {message}", file=sys.stderr)
        return 1

    generated = {INDEX_JSON: render_index_json(selection),
                 INDEX_README: render_readme(selection)}
    print(format_summary(selection))
    print()
    if selection.untracked_matches:
        print(f"Note: {len(selection.untracked_matches)} untracked file(s) under example/ "
              "match a pattern of example/PACKAGED and are not packaged (git add and commit "
              "them to ship them): "
              + _shorten([f"example/{p}" for p in selection.untracked_matches], limit=3))
        print()
    if selection.total_bytes > selection.budget:
        print(f"ERROR: the packaged files take {selection.total_bytes:,} bytes, more than the "
              f"budget of {selection.budget:,} in example/PACKAGED"
              + ("" if args.check else "; nothing was written"), file=sys.stderr)
        return 1

    drift = compare(selection, generated)
    total_files = sum(len(d.files) for d in selection.directories)
    if args.check:
        if drift.clean:
            print("* Check *")
            print(f" crystod/examples matches example/PACKAGED ({total_files} files, "
                  "index.json, README)")
            return 0
        print(format_drift(drift, "Differences from example/PACKAGED",
                           {"missing": "missing", "different": "differs", "mode": "mode",
                            "extra": "extra", "stale": "stale"}))
        print()
        print("ERROR: crystod/examples is out of date; run python "
              "script/sync_packaged_examples.py", file=sys.stderr)
        return 1

    if drift.clean:
        print("* Changes *")
        print(f" none: crystod/examples already matches example/PACKAGED ({total_files} files)")
        return 0
    apply(selection, generated, drift)
    print(format_drift(drift, "Changes",
                       {"missing": "copied", "different": "replaced", "mode": "mode",
                        "extra": "removed", "stale": "wrote"}))
    print(f" {drift.unchanged} file(s) already up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
