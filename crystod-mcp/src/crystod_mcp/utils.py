"""Shared helpers of the crystod-mcp tools.

Three concerns live here so that the tool modules stay declarative:

* **timeouts** -- :func:`blocking_tool` runs a synchronous tool body in a
  worker thread under ``asyncio.wait_for`` (60 s by default, overridable by
  the ``CRYSTOD_MCP_TIMEOUT`` environment variable) and turns a timeout, or
  any error CrystOD raises, into a ``ToolError`` whose one-line message the
  model can read;
* **the citation footer** -- :func:`with_citation` appends the ISO-IR and
  CrystOD citation to every result;
* **input validation** -- the ``parse_*`` / ``validate_*`` helpers raise
  ``ValueError`` with a one-sentence remedy before any calculation starts.

Structure text (POSCAR, FORCE_SETS, XYZ) never reaches the tools as a path:
:func:`run_cli` and :func:`temporary_files` write it into a temporary
directory that is removed after the call.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import math
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from collections.abc import Callable, Iterable, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from mcp.server.mcpserver.exceptions import ToolError

DEFAULT_TIMEOUT = 60.0
TIMEOUT_ENV = "CRYSTOD_MCP_TIMEOUT"

CITATION = (
    "Irrep labels follow ISO-IR (H. T. Stokes, B. J. Campbell and R. Cordes, "
    "Acta Cryst. A69, 388 (2013), https://iso.byu.edu). CrystOD: H. Koiso and "
    "Y. Mochizuki et al., Phys. Rev. B 110, 064104 (2024)."
)

# The 32 crystallographic point groups in the Hermann-Mauguin spelling of
# CrystOD's character tables (crystod-group --pg). Used to tell a point group
# from a space group where one parameter accepts both.
POINT_GROUPS: tuple[str, ...] = (
    "1", "-1", "2", "m", "2/m", "222", "mm2", "mmm",
    "4", "-4", "4/m", "422", "4mm", "-42m", "4/mmm",
    "3", "-3", "32", "3m", "-3m",
    "6", "-6", "6/m", "622", "6mm", "-6m2", "6/mmm",
    "23", "m-3", "432", "-43m", "m-3m",
)

# Orbital shells CrystOD's ligand-field / SALC analyses accept (l = 0..6).
ORBITALS: tuple[str, ...] = ("s", "p", "d", "f", "g", "h", "i")

_ISOIR_LABEL = re.compile(r"^[A-Z]{1,2}[0-9]+[+-]?$")
_ELEMENT = re.compile(r"^[A-Z][a-z]?$")
_KPOINT_LABEL = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


# --------------------------------------------------------------------------
# timeouts and the tool wrapper
# --------------------------------------------------------------------------

class ToolTimeout(TimeoutError):
    """A subprocess-based tool hit the time limit (mapped like a thread timeout)."""


def timeout_seconds() -> float:
    """The per-call time limit in seconds: ``CRYSTOD_MCP_TIMEOUT``, else 60.

    An unusable value (not a number, or not positive) falls back to the
    default rather than disabling every tool.
    """
    raw = os.environ.get(TIMEOUT_ENV, "").strip()
    if not raw:
        return DEFAULT_TIMEOUT
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_TIMEOUT
    if not math.isfinite(value) or value <= 0:
        return DEFAULT_TIMEOUT
    return value


def timeout_message(limit: float) -> str:
    return (
        f"the calculation did not finish within {limit:g} s. Set the "
        f"{TIMEOUT_ENV} environment variable of the server to a larger number "
        "of seconds, or ask a smaller question (one q point instead of a scan, "
        "a smaller supercell)."
    )


def with_citation(text: str) -> str:
    """Append the ISO-IR / CrystOD citation footer to a tool result."""
    return text.rstrip() + "\n\n" + CITATION + "\n"


def _one_line(text: object) -> str:
    return " ".join(str(text).split())


def blocking_tool(fn: Callable[..., str]) -> Callable[..., Any]:
    """Turn a synchronous tool body into the async MCP tool.

    The body runs in a worker thread under ``asyncio.wait_for`` with the
    limit of :func:`timeout_seconds`. Its return value gets the citation
    footer; a timeout, a ``ValueError`` from validation, and any error the
    CrystOD machinery raises (including the ``SystemExit`` the command-line
    implementations use for bad input) become a ``ToolError``, which the SDK
    returns to the model as an ``is_error`` result carrying the message. The
    signature and docstring of ``fn`` are preserved (``functools.wraps``),
    so the SDK derives the tool name, description and input schema from the
    body itself.
    """

    @functools.wraps(fn)
    async def tool(*args: Any, **kwargs: Any) -> str:
        limit = timeout_seconds()
        try:
            result = await asyncio.wait_for(asyncio.to_thread(fn, *args, **kwargs), limit)
        except (asyncio.TimeoutError, TimeoutError):
            raise ToolError(timeout_message(limit)) from None
        except ToolError:
            raise
        except SystemExit as exc:
            message = _one_line(exc).removeprefix("ERROR: ") or "invalid input"
            raise ToolError(message) from None
        except Exception as exc:  # noqa: BLE001 - every failure must reach the model as text
            message = _one_line(exc).removeprefix("ERROR: ") or type(exc).__name__
            raise ToolError(message) from None
        return with_citation(result)

    return tool


# --------------------------------------------------------------------------
# temporary input files and CrystOD command lines
# --------------------------------------------------------------------------

@contextlib.contextmanager
def temporary_files(files: dict[str, str]):
    """Write ``{name: text}`` into a fresh temporary directory; yield its path."""
    with tempfile.TemporaryDirectory(prefix="crystod-mcp-") as tmp:
        directory = Path(tmp)
        for name, text in files.items():
            (directory / name).write_text(text, encoding="utf-8")
        yield directory


def run_cli(module: str, argv: Sequence[str], files: dict[str, str]) -> str:
    """Run one CrystOD command line in a temporary directory; return its stdout.

    ``module`` is the CLI module (``crystod.cli.mol``); ``files`` maps file
    names to the text written into the directory before the run (the POSCAR
    or XYZ the command reads, so ``argv`` refers to them by bare name). The
    command runs with the interpreter of this server, so the CrystOD it sees
    is the one crystod-mcp was installed with. The child is killed at the
    time limit; a non-zero exit becomes a ``ValueError`` carrying the
    command's own ``ERROR:`` line.
    """
    limit = timeout_seconds()
    command = [sys.executable, "-c", f"from {module} import main; main()", *argv]
    with temporary_files(files) as directory:
        try:
            completed = subprocess.run(
                command,
                cwd=directory,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=limit,
            )
        except subprocess.TimeoutExpired:
            raise ToolTimeout(timeout_message(limit)) from None
    if completed.returncode != 0:
        raise ValueError(_cli_failure(completed.stderr, completed.stdout))
    return completed.stdout.strip("\n")


def _cli_failure(stderr: str, stdout: str) -> str:
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    flagged = [line for line in lines if line.startswith("ERROR") or ": error:" in line]
    if flagged:
        return _one_line(" ".join(flagged)).removeprefix("ERROR: ")
    if lines:
        return lines[-1]
    tail = [line.strip() for line in stdout.splitlines() if line.strip()]
    return tail[-1] if tail else "the CrystOD command failed without a message"


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------

def _tokens(value: str) -> list[str]:
    return [token for token in re.split(r"[\s,;]+", value.strip()) if token]


def parse_space_group(value: object) -> str:
    """A space group as CrystOD accepts it: international symbol or number 1-230."""
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(
            "give the space group as an international symbol such as Pm-3m or "
            "P6_3/mmc, or as its number from 1 to 230."
        )
    if text.isdigit() and not 1 <= int(text) <= 230:
        raise ValueError(
            f"{text} is not a space-group number; use 1 to 230 or a symbol such as Pm-3m."
        )
    return text


def parse_point_group(value: object) -> str:
    """A point group in Hermann-Mauguin notation (m-3m, 3m, -43m, ...)."""
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(
            "give the point group in Hermann-Mauguin notation, e.g. m-3m, 4/mmm, 3m "
            "or -43m (Td)."
        )
    if text not in POINT_GROUPS:
        raise ValueError(
            f'"{text}" is not one of the 32 crystallographic point groups in '
            f"Hermann-Mauguin notation; choose from: {', '.join(POINT_GROUPS)}."
        )
    return text


def is_point_group(value: object) -> bool:
    """True when ``value`` spells one of the 32 point groups (Hermann-Mauguin)."""
    return str(value).strip() in POINT_GROUPS


def parse_isoir_labels(value: object) -> list[str]:
    """ISO-IR space-group irrep labels, one or several (``R4+``, ``X3- X2+``).

    Labels are upper-cased (``r4+`` -> ``R4+``) and may be separated by
    spaces or commas; several labels mean a coupled order parameter.
    """
    if isinstance(value, (list, tuple)):
        raw = " ".join(str(item) for item in value)
    else:
        raw = "" if value is None else str(value)
    labels = [token.upper() for token in _tokens(raw)]
    if not labels:
        raise ValueError(
            "give the irrep as an ISO-IR label of the space group, e.g. R4+, GM5- "
            "or X3- (k-point name, index, parity sign)."
        )
    for label in labels:
        if not _ISOIR_LABEL.match(label):
            raise ValueError(
                f'"{label}" is not an ISO-IR irrep label; expected the k-point '
                "name followed by the irrep index and, for centrosymmetric "
                "groups, the parity sign, e.g. R4+, GM5-, X3-, DT5, K1."
            )
    return labels


def parse_irrep_names(values: object) -> list[str]:
    """Irrep names for a product: a list, or one string with spaces/commas."""
    if isinstance(values, str):
        items: Iterable[object] = [values]
    elif isinstance(values, (list, tuple)):
        items = values
    else:
        items = [values] if values is not None else []
    names: list[str] = []
    for item in items:
        names.extend(_tokens(str(item)))
    if len(names) < 2:
        raise ValueError(
            "give at least two irrep labels to multiply, e.g. ['R4-', 'R5+'] for a "
            "space group or ['T2g', 'T2g'] for a point group."
        )
    return names


def parse_order_parameter(value: object) -> list[str] | None:
    """An order-parameter direction (``"0 0 a"``, ``"a,a,0"``) as tokens, or None."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        tokens = [str(item).strip() for item in value if str(item).strip()]
    else:
        tokens = _tokens(str(value))
    if not tokens:
        return None
    for token in tokens:
        if not re.fullmatch(r"[A-Za-z]|-?[0-9]+(\.[0-9]+)?", token):
            raise ValueError(
                f'"{token}" is not an order-parameter component; give letters for '
                'free parameters and numbers for fixed ones, e.g. "0 0 a" or "a a 0".'
            )
    return tokens


def parse_kpoint(value: object) -> tuple[str | None, list[float] | None]:
    """A k point as ``(label, None)`` or ``(None, [k1, k2, k3])``; ``(None, None)`` when absent.

    Accepts a label (``"R"``, ``"GM"``), a string of three coordinates with
    fractions (``"1/2 1/2 1/2"``), or a list of three numbers.
    """
    if value is None:
        return None, None
    if isinstance(value, (list, tuple)):
        tokens = [str(item) for item in value]
    else:
        text = str(value).strip()
        if not text:
            return None, None
        tokens = _tokens(text)
        if len(tokens) == 1 and _KPOINT_LABEL.match(tokens[0]):
            return tokens[0].upper(), None
    if len(tokens) != 3:
        raise ValueError(
            'give the k point as a label such as R or GM, or as three fractional '
            'coordinates in the primitive reciprocal basis such as "1/2 1/2 1/2".'
        )
    try:
        coordinates = [float(Fraction(token)) for token in tokens]
    except (ValueError, ZeroDivisionError):
        raise ValueError(
            f'"{" ".join(tokens)}" are not three k-point coordinates; use numbers '
            'or fractions such as "1/2 1/2 0".'
        ) from None
    return None, coordinates


def format_kpoint(coordinates: Sequence[float]) -> str:
    """``(1/2, 1/2, 0)``-style display of fractional coordinates."""
    parts = []
    for value in coordinates:
        fraction = Fraction(float(value)).limit_denominator(24)
        parts.append(str(fraction.numerator) if fraction.denominator == 1 else str(fraction))
    return "(" + ", ".join(parts) + ")"


def parse_dim(value: object) -> list[int]:
    """The supercell dimension of a force calculation: three positive integers."""
    items = list(value) if isinstance(value, (list, tuple)) else _tokens(str(value or ""))
    dim: list[int] = []
    for item in items:
        try:
            number = float(item)
        except (TypeError, ValueError):
            number = float("nan")
        if number != number or number <= 0 or number != int(number):
            dim = []
            break
        dim.append(int(number))
    if len(dim) != 3:
        raise ValueError(
            "dim must be the three diagonal supercell multiples of the force "
            "calculation, e.g. [4, 4, 4] (phonopy's --dim), as positive integers."
        )
    return dim


def _clean_block(text: object, what: str, keep_leading: bool = False) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{what} is empty; pass the full text of the file, not a path.")
    block = textwrap.dedent(text).rstrip("\n")
    if not keep_leading:
        block = block.lstrip("\n")
    return block + "\n"


def _floats(line: str, count: int) -> bool:
    tokens = line.split()
    if len(tokens) < count:
        return False
    try:
        [float(token) for token in tokens[:count]]
    except ValueError:
        return False
    return True


def validate_poscar(text: object) -> str:
    """The full text of a VASP POSCAR (VASP 5 format, with the element line)."""
    remedy = (
        "poscar must be the full text of a VASP POSCAR: a comment line, the "
        "scale factor, three lattice vectors, the element symbols, their "
        "counts, 'Direct' or 'Cartesian', and one coordinate line per atom."
    )
    # the first line of a POSCAR is a free comment and may be empty, while a
    # pasted block often starts with a stray blank line: the text is tried as
    # given first and with its leading blank lines removed second
    candidates = (_clean_block(text, "poscar", keep_leading=True), _clean_block(text, "poscar"))
    for poscar in candidates:
        lines = poscar.splitlines()
        if (len(lines) >= 8 and _floats(lines[1], 1)
                and all(_floats(lines[i], 3) for i in (2, 3, 4))):
            break
    else:
        raise ValueError(remedy)
    symbols = lines[5].split()
    if not symbols or not all(_ELEMENT.match(token) for token in symbols):
        raise ValueError(
            remedy + " The sixth line must hold the element symbols (VASP 5 format)."
        )
    counts = lines[6].split()
    if not counts or not all(token.isdigit() for token in counts) or len(counts) != len(symbols):
        raise ValueError(remedy + " The seventh line must hold one atom count per element.")
    return poscar


def validate_force_sets(text: object) -> str:
    """The full text of a phonopy FORCE_SETS file."""
    force_sets = _clean_block(text, "force_sets")
    head = [line.strip() for line in force_sets.splitlines() if line.strip()][:2]
    if len(head) < 2 or not head[0].isdigit() or not head[1].isdigit():
        raise ValueError(
            "force_sets must be the full text of a phonopy FORCE_SETS file (first "
            "line: number of atoms of the supercell, second line: number of "
            "displacements, then one displacement block per displacement)."
        )
    return force_sets


def validate_xyz(text: object) -> str:
    """The full text of an XYZ file (atom count, comment, one line per atom)."""
    remedy = (
        "xyz must be the full text of an XYZ file: the number of atoms, a comment "
        "line, then one 'Symbol x y z' line per atom (Angstrom)."
    )
    xyz = _clean_block(text, "xyz")
    lines = xyz.splitlines()
    if not lines or not lines[0].strip().isdigit():
        raise ValueError(remedy)
    natoms = int(lines[0])
    atoms = lines[2 : 2 + natoms]
    if natoms < 1 or len(atoms) < natoms or not all(_floats(" ".join(line.split()[1:]), 3) for line in atoms):
        raise ValueError(remedy)
    return xyz


def parse_element(value: object) -> str:
    """An element symbol (``Sc``, ``H``); case is normalized."""
    text = "" if value is None else str(value).strip()
    normalized = text[:1].upper() + text[1:].lower()
    if not _ELEMENT.match(normalized):
        raise ValueError(f'"{text}" is not an element symbol; give one such as Sc, Ti or H.')
    return normalized


def parse_orbital(value: object, allowed: Sequence[str] = ORBITALS) -> str:
    """An orbital shell letter (``s``, ``p``, ``d``, ``f``, ...)."""
    text = "" if value is None else str(value).strip().lower()
    if text not in allowed:
        raise ValueError(
            f'"{value}" is not an orbital shell; give one of {", ".join(allowed)}.'
        )
    return text


# --------------------------------------------------------------------------
# structures and space groups
# --------------------------------------------------------------------------

def _dataset_field(dataset: Any, name: str) -> Any:
    # spglib >= 2.5 returns a SpglibDataset object, older versions a dict
    return dataset[name] if isinstance(dataset, dict) else getattr(dataset, name)


def space_group_of_poscar(poscar: str, symprec: float = 1e-5) -> tuple[int, str]:
    """``(number, international short symbol)`` of the structure in ``poscar``."""
    import spglib
    from phonopy.interface.vasp import read_vasp

    with temporary_files({"POSCAR": poscar}) as directory:
        try:
            cell = read_vasp(str(directory / "POSCAR"))
        except Exception as exc:  # noqa: BLE001 - phonopy's parser reports many shapes of bad input
            raise ValueError(
                f"the POSCAR text could not be read ({_one_line(exc)}); pass the "
                "full text of a VASP POSCAR in the VASP 5 format."
            ) from None
    dataset = spglib.get_symmetry_dataset(
        (cell.cell, cell.scaled_positions, cell.numbers), symprec=symprec
    )
    if dataset is None:
        raise ValueError("spglib could not determine the space group of the POSCAR structure.")
    return int(_dataset_field(dataset, "number")), str(_dataset_field(dataset, "international"))


def special_kpoints(space_group: str):
    """``(sg_type, names, primitive, conventional)`` of a space group (ISO-IR tables)."""
    from crystod import bz

    return bz.get_special_kpoints(space_group)


def resolve_kpoint_label(space_group: str, label: str) -> list[float]:
    """Primitive coordinates of the tabulated k point ``label`` of ``space_group``."""
    _, names, primitive, _ = special_kpoints(space_group)
    for name, coordinates in zip(names, primitive):
        if name == label:
            return [float(v) for v in coordinates]
    raise ValueError(
        f'"{label}" is not a special k point of space group {space_group}; the '
        f"tabulated ones are {', '.join(names)} (or give three coordinates)."
    )


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------

def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    """A GitHub-flavoured Markdown table."""
    lines = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def fenced(text: str) -> str:
    """Wrap a preformatted CrystOD report in a code fence."""
    return "```\n" + text.strip("\n") + "\n```"


def format_character(value: object) -> str:
    """Display of a character-table entry (integers plain, complex as a+bj)."""
    import numpy as np

    scalar = np.asarray(value).item()
    if isinstance(scalar, complex):
        if abs(scalar.imag) < 1e-10:
            scalar = scalar.real
        else:
            return f"{scalar.real:.4g}{scalar.imag:+.4g}j"
    number = float(scalar)
    if abs(number - round(number)) < 1e-10:
        return str(int(round(number)))
    return f"{number:.4g}"


__all__ = [
    "CITATION",
    "DEFAULT_TIMEOUT",
    "ORBITALS",
    "POINT_GROUPS",
    "TIMEOUT_ENV",
    "ToolTimeout",
    "blocking_tool",
    "fenced",
    "format_character",
    "format_kpoint",
    "is_point_group",
    "markdown_table",
    "parse_dim",
    "parse_element",
    "parse_irrep_names",
    "parse_isoir_labels",
    "parse_kpoint",
    "parse_orbital",
    "parse_order_parameter",
    "parse_point_group",
    "parse_space_group",
    "resolve_kpoint_label",
    "run_cli",
    "space_group_of_poscar",
    "special_kpoints",
    "temporary_files",
    "timeout_seconds",
    "validate_force_sets",
    "validate_poscar",
    "validate_xyz",
    "with_citation",
]
