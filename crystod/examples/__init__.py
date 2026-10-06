"""Bundled example inputs: the data behind the ``--example`` flags.

A handful of small input files ship inside the package (about 100 kB in
total) so that the quick start runs right after ``pip install CrystOD``,
without cloning the repository:

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
``crystod-xrd`` and ``crystod-search`` accept
``--example NAME``: the files of that example (the searches have none) are
copied into the current directory, the equivalent ordinary command line is printed, and that command
runs. ``--example`` without a name (or ``--example list``) lists the names.

The registry below is the single source of truth for the CLI, the
tutorials and the MCP server::

    from crystod.examples import get_example, example_path

    example = get_example("crystod-phonon", "SrTiO3")
    example.argv          # ('--irreps', '--dim', '4 4 4', '-c', '221_PPOSCAR_SrTiO3')
    example_path("221_PPOSCAR_ScF3")   # absolute Path inside the installed package
"""

from __future__ import annotations

import filecmp
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Example:
    """One runnable example of a command.

    Attributes:
        name: The name given to ``--example`` (``ScF3_d``, ``SrTiO3``, ...).
        command: The console script the example belongs to (``crystod``,
            ``crystod-phonon``, ``crystod-mol``, ``crystod-bz``, ``crystod-xrd``,
            ``crystod-search``).
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
        bundled = ", ".join(sorted(p.name for p in example_dir().iterdir()
                                   if p.is_file() and not p.name.startswith("__")))
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


__all__ = [
    "Example",
    "EXAMPLES",
    "example_dir",
    "example_path",
    "list_examples",
    "get_example",
    "format_example_list",
    "copy_example_files",
]
