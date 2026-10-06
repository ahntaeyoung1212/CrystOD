"""crystod-search: search the Materials Project and download POSCAR files.

The query is read like the search box of the Materials Project website
(formula, anonymous formula, chemical system, elements or material IDs);
the matches are listed with formula, space group, material ID, band gap,
energy above the hull and number of sites, a star after the ID marking the
experimentally observed materials.  ``--get`` downloads the structures as
POSCAR files (backed by :mod:`crystod.mp_search`).
"""

from __future__ import annotations

import re
from argparse import ArgumentParser, RawTextHelpFormatter

from ..mp_search import (
    CELL_CHOICES,
    DEFAULT_CELL,
    DEFAULT_MAX_RESULTS,
    DEFAULT_SORT,
    DEFAULT_TOLERANCE,
    SORT_KEYS,
    format_table,
)
from .common import (
    CRYSTOD_CITATION,
    ExampleRequested,
    add_example_argument,
    add_output_argument,
    banner,
    run_example,
)

desc = """\
Search the Materials Project and download crystal structures as POSCAR files.

The query is read like the search box of the Materials Project website:
  SrTiO3     formula: every polymorph of this composition
  Sr-Ti-O    chemical system: the compounds of exactly these elements
             (Sr-* : a wildcard element; --subsystems adds Sr, Ti-O, ...)
  Sr,Ti,O    elements: every material containing at least these
  ABO3       anonymous formula: each letter stands for any element
  mp-5229    material ID (several: mp-5229,mp-5532)
The table lists formula, space group, material ID, band gap, energy above
the hull and number of sites, sorted like the website by the energy above
the hull; a star after the ID marks an experimentally observed material.

--get MPID ... downloads those structures; --get after a query downloads
every listed one. POSCAR_{formula}_{space group}_{ID} holds the standardized
conventional cell; --cell primitive writes the standardized primitive cell
as PPOSCAR_... instead.

An API key of the Materials Project is needed (free, from
https://next-gen.materialsproject.org/api): export MP_API_KEY=<key>, or
store it once with  pmg config --add PMG_MAPI_KEY <key>

# Command Examples:
crystod-search SrTiO3
crystod-search Sr-Ti-O --experimental
crystod-search Sr-Ti-O --stable --subsystems
crystod-search --get mp-5532                    (POSCAR_Sr2TiO4_I4mmm_mp-5532)
crystod-search --get mp-5532 --cell primitive   (PPOSCAR_Sr2TiO4_I4mmm_mp-5532)
crystod-search SrTiO3 --experimental --get --directory
crystod-search --example SrTiO3   (--example alone lists the names)
"""


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="crystod-search",
        description=f"{banner()}\n\n{desc}",
        epilog=CRYSTOD_CITATION,
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Formula (SrTiO3), chemical system (Sr-Ti-O), elements\n"
        "(Sr,Ti,O), anonymous formula (ABO3) or material IDs (mp-5229).",
    )
    parser.add_argument(
        "--get",
        nargs="*",
        default=None,
        metavar="MPID",
        help="Download POSCAR files: of the IDs given, or, after a query\n"
        "and without IDs, of every listed material.",
    )
    add_example_argument(parser, "crystod-search")

    search = parser.add_argument_group("search filters")
    search.add_argument(
        "--experimental", "--exp",
        action="store_true",
        help="Only experimentally observed materials (starred ones).",
    )
    search.add_argument(
        "--stable",
        action="store_true",
        help="Only materials on the convex hull (energy above hull 0).",
    )
    search.add_argument(
        "--ehull",
        type=float,
        default=None,
        metavar="MAX",
        help="Only materials at most MAX eV/atom above the hull.",
    )
    search.add_argument(
        "--band-gap",
        nargs=2,
        type=float,
        default=None,
        metavar=("MIN", "MAX"),
        help="Only band gaps from MIN to MAX eV.",
    )
    search.add_argument(
        "--sites",
        nargs=2,
        type=int,
        default=None,
        metavar=("MIN", "MAX"),
        help="Only cells of MIN to MAX sites.",
    )
    search.add_argument(
        "--spg",
        default=None,
        metavar="SPACEGROUP",
        help="Only this space group: number (221) or symbol (Pm-3m, P6_3/mmc).",
    )
    search.add_argument(
        "--exclude",
        nargs="+",
        default=None,
        metavar="EL",
        help="Leave out materials containing these elements.",
    )
    search.add_argument(
        "--subsystems",
        action="store_true",
        help="With a chemical system, also list its subsystems\n"
        "(Sr-Ti-O: Sr, Ti, O, Sr-Ti, Sr-O and Ti-O as well).",
    )
    search.add_argument(
        "--sort",
        choices=tuple(SORT_KEYS),
        default=DEFAULT_SORT,
        help="Sort key (default: ehull, the energy above the hull, as on\n"
        "the website): " + ", ".join(f"{k} = {v[1]}" for k, v in SORT_KEYS.items()) + ".",
    )
    search.add_argument(
        "--max",
        type=int,
        default=DEFAULT_MAX_RESULTS,
        metavar="N",
        help=f"List at most N materials (default: {DEFAULT_MAX_RESULTS}).",
    )

    download = parser.add_argument_group("POSCAR download (--get)")
    download.add_argument(
        "--cell",
        choices=CELL_CHOICES,
        default=DEFAULT_CELL,
        help="Cell of the POSCAR: conventional, the standardized conventional\n"
        "cell (default, POSCAR_...), or primitive, the standardized\n"
        "primitive cell (PPOSCAR_...).",
    )
    download.add_argument(
        "--tolerance",
        type=_positive_float,
        default=DEFAULT_TOLERANCE,
        help="Symmetry tolerance in Angstrom of the standardization (default:\n"
        "0.1, the value the Materials Project assigns its space groups with).",
    )
    add_output_argument(
        download,
        "File name of the POSCAR (one material only). Default:\n"
        "POSCAR_{formula}_{space group}_{ID}, e.g. POSCAR_SrTiO3_Pm-3m_mp-5229\n"
        "(PPOSCAR_... with --cell primitive).",
    )
    download.add_argument(
        "--directory",
        action="store_true",
        help="Write {formula}_{space group}_{ID}/POSCAR instead (PPOSCAR with\n"
        "--cell primitive), one directory per material.",
    )
    download.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing file whose content differs.",
    )
    return parser


def _filters_given(args) -> list[str]:
    given = []
    for option, value in (("--experimental", args.experimental), ("--stable", args.stable),
                          ("--ehull", args.ehull is not None),
                          ("--band-gap", args.band_gap is not None),
                          ("--sites", args.sites is not None), ("--spg", args.spg is not None),
                          ("--exclude", args.exclude is not None),
                          ("--subsystems", args.subsystems)):
        if value:
            given.append(option)
    return given


def _print_table(title: str, materials, lines: list[str]) -> None:
    print(f"\n * Materials Project: {title} *")
    for line in lines:
        print(" " + line)
    if not materials:
        return
    if lines:
        # the query block above carries the filters and the count; the
        # table gets its own block (a bare --get lists it under the title)
        print("\n * Materials *")
    for line in format_table(materials):
        print(" " + line)
    if any(m.experimental for m in materials):
        print("\n * experimentally observed (the structure matches an ICSD or other"
              " experimental entry)")


def _download(materials, args) -> None:
    import shlex

    from ..mp_search import write_poscar

    cell = f"standardized {args.cell} cell, tolerance {args.tolerance:g} A"
    print(f"\n * POSCAR files ({cell}) *")
    written = []
    rename_hint = len(materials) == 1 and not args.directory
    for material in materials:
        path, status = write_poscar(material, args.output, directory=args.directory,
                                    overwrite=args.force, rename_hint=rename_hint)
        verb = {"wrote": "Wrote", "kept": "Kept", "replaced": "Replaced"}[status]
        note = " (identical file already there)" if status == "kept" else ""
        print(f" {verb} {path}{note}: {material.formula}, {material.space_group}, "
              f"{len(material.structure)} atoms")
        found = material.cell_space_group_number
        if found is not None and found != material.space_group_number:
            print(f"   WARNING: this cell has space group No. {found}, not No. "
                  f"{material.space_group_number} ({material.space_group}) as listed; "
                  "try another --tolerance.")
        written.append(path)
    if len(written) == 1:
        print(f"\nUse it with any CrystOD command, e.g.  crystod-xrd -c {shlex.quote(written[0])}")


def _positive_float(text: str) -> float:
    """argparse type of --tolerance: a finite number > 0."""
    import math
    from argparse import ArgumentTypeError

    try:
        value = float(text)
    except ValueError:
        raise ArgumentTypeError(f"invalid number: {text!r}") from None
    if not (math.isfinite(value) and value > 0):
        raise ArgumentTypeError("must be a positive number of Angstrom")
    return value


def _is_material_id(text: str) -> bool:
    from ..mp_search import _ID_RE

    return bool(_ID_RE.match(text.strip().lower()))


def run(args, parser) -> None:
    """Carry out a parsed ``crystod-search`` command line."""
    from ..mp_search import ELEMENTS, MP_CITATION, fetch_materials, normalize_material_id, search_materials

    # --get mp-5229,mp-5532 as well as --get mp-5229 mp-5532
    ids = [token.strip() for value in (args.get or []) for token in value.split(",")
           if token.strip()]
    if args.query is None and args.get is None:
        if args.exclude and args.exclude[-1] not in ELEMENTS:
            parser.error(f"'{args.exclude[-1]}' was read as an --exclude element; put the "
                         f"query first: crystod-search {args.exclude[-1]} --exclude "
                         f"{' '.join(args.exclude[:-1]) or 'EL'}")
        parser.error("give a query (SrTiO3, Sr-Ti-O, Sr,Ti,O, ABO3, mp-5229) "
                     "or --get MPID")
    not_ids = [token for token in ids if not _is_material_id(token)]
    if not_ids:
        token = not_ids[0]
        if re.match(r"^(\d+|mp-.*|[a-z]+-[0-9a-z]*\d[0-9a-z]*)$", token.lower()):
            normalize_material_id(token)  # an ID gone wrong: its own ERROR line
        parser.error(f"--get takes material IDs, not '{token}'; to download what a "
                     f"query lists, put the query first: crystod-search {token} --get")
    if args.query is not None and ids:
        parser.error("give either a query (its listed materials are downloaded by a "
                     "bare --get) or --get MPID ..., not both")
    if args.query is None and not ids:
        parser.error("--get needs material IDs (--get mp-5229), or a query before it "
                     "(crystod-search SrTiO3 --get)")
    filters = _filters_given(args)
    if args.query is None and filters:
        parser.error(f"{', '.join(filters)} {'filters' if len(filters) == 1 else 'filter'} "
                     "a search; --get MPID downloads exactly the IDs given")
    if args.output and args.directory:
        parser.error("-o/--output and --directory exclude each other")

    if ids:
        if args.output and len({normalize_material_id(i) for i in ids}) > 1:
            parser.error("-o/--output names one file; leave it out (or use --directory) "
                         "for several materials")
        materials = fetch_materials(ids, cell=args.cell, tolerance=args.tolerance)
        ids_text = ", ".join(m.material_id for m in materials)
        _print_table(f"material ID{'s' if len(materials) > 1 else ''} {ids_text}",
                     materials, [])
        _download(materials, args)
        print("\n" + MP_CITATION)
        return

    result = search_materials(
        args.query,
        experimental=args.experimental,
        stable=args.stable,
        max_energy_above_hull=args.ehull,
        band_gap=tuple(args.band_gap) if args.band_gap else None,
        nsites=tuple(args.sites) if args.sites else None,
        space_group=args.spg,
        exclude_elements=args.exclude,
        subsystems=args.subsystems,
        sort=args.sort,
        max_results=args.max,
    )
    sort_phrase = SORT_KEYS[result.sort][1]
    lines = [f"filters: {'; '.join(result.filters)}"] if result.filters else []
    count = len(result.materials)
    if not count:
        lines.append("no material matches.")
    elif result.truncated:
        lines.append(f"{count} of {result.total} materials listed, sorted by {sort_phrase}; "
                     "narrow the query or raise --max to see more.")
    else:
        lines.append(f"{count} material{'s' if count != 1 else ''}, sorted by {sort_phrase}")
    if args.get is not None and args.output and count > 1:
        parser.error(f"-o/--output names one file but {count} materials are listed; "
                     "leave it out or use --directory")
    _print_table(result.description, result.materials, lines)

    if args.get is not None and result.materials:
        materials = fetch_materials([m.material_id for m in result.materials],
                                    cell=args.cell, tolerance=args.tolerance)
        _download(materials, args)
    elif result.materials:
        print(f"\nDownload a POSCAR:  crystod-search --get {result.materials[0].material_id}"
              "\n(or add --get to this command line to download every listed material)")
    print("\n" + MP_CITATION)


def main(argv: list[str] | None = None) -> None:
    import sys

    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except ExampleRequested as request:
        # --example: print the equivalent ordinary command line and run it
        # (the search examples have no input files to copy)
        main(run_example("crystod-search", request.name, argv))
        return

    from ..mp_search import MaterialsProjectError

    try:
        run(args, parser)
    except MaterialsProjectError as exc:
        raise SystemExit(f"ERROR: {exc}") from None


if __name__ == "__main__":
    main()
