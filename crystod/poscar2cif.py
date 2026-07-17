"""POSCAR <-> Bilbao-style CIF (crystod-group --poscar2cif / --cif2poscar).

--poscar2cif converts a POSCAR into a CIF laid out like the files produced by
the Bilbao Crystallographic Server (http://www.cryst.ehu.es): conventional
cell with 4-decimal lattice parameters, the space-group number and
Hermann-Mauguin symbol, the full list of conventional-cell symmetry
operations as compact x,y,z strings, and one representative site per Wyckoff
orbit with 5-decimal fractional coordinates -- unlike the pymatgen CifWriter
layout (which quotes the operators, adds formula/volume/Z lines, and lists
sites with a multiplicity column).  The output file is written next to the
input as <POSCAR>.cif.

--cif2poscar is the inverse: it reads any CIF (Bilbao or pymatgen flavour),
expands the symmetry operations, and writes the spglib-standardized
primitive cell as a POSCAR (the working format of the other crystod
commands; --conventional writes the conventional cell instead).  The output
is the input path without the .cif suffix.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crystod-group --poscar2cif",
        description="Convert a POSCAR into a Bilbao-style CIF (<POSCAR>.cif).",
    )
    parser.add_argument(
        "--cell",
        "-c",
        "--poscar",
        dest="cell",
        required=True,
        metavar="POSCAR",
        help="Input POSCAR file.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Symmetry-detection tolerance (symprec) in Angstrom (default: 0.01).",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Output CIF path (default: <POSCAR>.cif next to the input).",
    )
    return parser


def _operation_string(rotation, translation) -> str:
    """Compact Bilbao operator string 'x+1/2,-y,z' from an integer rotation
    matrix and a fractional translation (both in the conventional basis)."""
    from fractions import Fraction

    labels = ("x", "y", "z")
    parts = []
    for i in range(3):
        term = ""
        for j in range(3):
            entry = int(round(rotation[i][j]))
            if entry == 1:
                term += ("+" if term else "") + labels[j]
            elif entry == -1:
                term += "-" + labels[j]
            elif entry != 0:
                raise SystemExit(
                    "ERROR: non-crystallographic rotation entry in the "
                    "standardized setting; please report this case."
                )
        twelfths = translation[i] * 12
        if abs(twelfths - round(twelfths)) > 1e-4:
            raise SystemExit(
                "ERROR: translation is not a multiple of 1/12 in the "
                "standardized setting; please report this case."
            )
        fraction = Fraction(int(round(twelfths)), 12) % 1
        if fraction:
            term += f"+{fraction}"
        parts.append(term)
    return ",".join(parts)


def _wrap_coordinate(value: float) -> float:
    wrapped = round(value % 1.0, 5) % 1.0
    return abs(wrapped)  # -0.0 -> 0.0


def bilbao_cif_lines(structure, tolerance: float, title: str):
    """(lines, info) of the Bilbao-style CIF for a pymatgen Structure.

    The structure is brought to the spglib-standardized (idealized)
    conventional cell, which follows the International Tables (ITA) setting
    and origin -- the same convention as the Bilbao server."""
    import numpy as np
    import spglib
    from pymatgen.core import Lattice
    from pymatgen.core.periodic_table import Element

    cell = (
        np.asarray(structure.lattice.matrix),
        np.asarray(structure.frac_coords),
        [site.specie.Z for site in structure],
    )
    standardized = spglib.standardize_cell(
        cell, to_primitive=False, no_idealize=False, symprec=tolerance
    )
    if standardized is None:
        raise SystemExit(
            "ERROR: spglib could not determine the symmetry of the structure "
            f"(tolerance {tolerance}); try another --tolerance."
        )
    lattice_matrix, positions, atomic_numbers = standardized
    dataset = spglib.get_symmetry_dataset(standardized, symprec=1e-5)

    def field(name):  # spglib < 2.4 returns a dict, >= 2.4 an object
        if isinstance(dataset, dict):
            return dataset.get(name)
        return getattr(dataset, name, None)

    number = int(field("number"))
    symbol = str(field("international")).replace("_", "")
    rotations = np.asarray(field("rotations"))
    translations = np.asarray(field("translations"))
    # Bilbao presentation: proper operations first, then the improper block
    order_key = sorted(
        range(len(rotations)),
        key=lambda i: 0 if np.linalg.det(rotations[i]) > 0 else 1,
    )
    operations = [
        _operation_string(rotations[i], translations[i]) for i in order_key
    ]

    equivalent = np.asarray(field("equivalent_atoms"))
    orbit_representatives = []
    seen = set()
    for index, orbit_id in enumerate(equivalent):
        if orbit_id not in seen:
            seen.add(orbit_id)
            orbit_representatives.append(index)

    now = datetime.now()
    lattice = Lattice(lattice_matrix)

    lines = [
        "# Created by CrystOD (Bilbao Crystallographic Server style)",
        "# https://www.cryst.ehu.es",
        f"# Date: {now.strftime('%d/%m/%Y %H:%M:%S')}",
        "",
        f"# {title} -- non-magnetic",
        "",
        f"data_{title}",
        f"{'_audit_creation_date':<35}{now.strftime('%Y-%m-%d')}",
        f"{'_audit_creation_method':<35}\"CrystOD (Bilbao style)\"",
        f"{'_symmetry_Int_Tables_number':<35}{number}",
        f"{'_symmetry_space_group_name_H-M':<35}\"{symbol}\"",
        f"{'_cell_length_a':<35}{lattice.a:.4f}",
        f"{'_cell_length_b':<35}{lattice.b:.4f}",
        f"{'_cell_length_c':<35}{lattice.c:.4f}",
        f"{'_cell_angle_alpha':<35}{lattice.alpha:.4f}",
        f"{'_cell_angle_beta':<35}{lattice.beta:.4f}",
        f"{'_cell_angle_gamma':<35}{lattice.gamma:.4f}",
        "",
        "loop_",
        "_symmetry_equiv_pos_site_id",
        "_symmetry_equiv_pos_as_xyz",
    ]
    for index, operation in enumerate(operations, start=1):
        lines.append(f"{index:>4}   {operation}")

    lines.extend(
        [
            "",
            "loop_",
            "_atom_site_label",
            "_atom_site_type_symbol",
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z",
            "_atom_site_occupancy",
        ]
    )

    representatives = sorted(
        orbit_representatives,
        key=lambda index: (
            Element.from_Z(int(atomic_numbers[index])).symbol,
            tuple(np.round(positions[index], 5)),
        ),
    )
    counters: dict[str, int] = {}
    for index in representatives:
        element = Element.from_Z(int(atomic_numbers[index])).symbol
        counters[element] = counters.get(element, 0) + 1
        x, y, z = (_wrap_coordinate(value) for value in positions[index])
        lines.append(
            f"{element}{counters[element]} {element} "
            f"{x:.5f} {y:.5f} {z:.5f} 1.0000"
        )
    lines.append("")
    info = {
        "number": number,
        "symbol": symbol,
        "n_operations": len(operations),
        "n_sites": len(representatives),
    }
    return lines, info


def build_cif2poscar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crystod-group --cif2poscar",
        description="Convert a CIF into a POSCAR (spglib-standardized cell).",
    )
    parser.add_argument(
        "--cell",
        "-c",
        "--cif",
        dest="cell",
        required=True,
        metavar="FILE.cif",
        help="Input CIF file.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Symmetry-detection tolerance (symprec) in Angstrom (default: 0.01).",
    )
    parser.add_argument(
        "--conventional",
        action="store_true",
        help="Write the conventional cell instead of the primitive cell.",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Output POSCAR path (default: the input path without .cif).",
    )
    return parser


def poscar_lines(lattice_matrix, positions, atomic_numbers) -> list[str]:
    """POSCAR content in the crystod test-file style (6 decimals, 'direct',
    element tag per coordinate line; species grouped by first appearance)."""
    from pymatgen.core.periodic_table import Element

    symbols = [Element.from_Z(int(z)).symbol for z in atomic_numbers]
    ordered_elements = []
    for symbol in symbols:
        if symbol not in ordered_elements:
            ordered_elements.append(symbol)
    counts = {element: symbols.count(element) for element in ordered_elements}

    lines = [
        " ".join(f"{element}{counts[element]}" for element in ordered_elements),
        "1.0",
    ]
    for row in lattice_matrix:
        lines.append(" ".join(f"{value:.6f}" for value in row))
    lines.append(" ".join(ordered_elements))
    lines.append(" ".join(str(counts[element]) for element in ordered_elements))
    lines.append("direct")
    for element in ordered_elements:
        for symbol, coords in zip(symbols, positions):
            if symbol == element:
                x, y, z = (_wrap_coordinate(value) for value in coords)
                lines.append(f"{x:.6f} {y:.6f} {z:.6f} {element}")
    lines.append("")
    return lines


def cif2poscar_main(argv: list[str] | None = None) -> None:
    args = build_cif2poscar_parser().parse_args(argv)

    if not os.path.isfile(args.cell):
        raise SystemExit(f"ERROR: CIF file not found: {args.cell}")
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from pymatgen.core import Structure

            structure = Structure.from_file(args.cell)
    except ImportError as exc:
        raise SystemExit(
            "ERROR: --cif2poscar requires pymatgen (pip install pymatgen)."
        ) from exc
    except Exception as exc:
        raise SystemExit(f"ERROR: could not read {args.cell} as a CIF: {exc}")

    import numpy as np
    import spglib

    cell = (
        np.asarray(structure.lattice.matrix),
        np.asarray(structure.frac_coords),
        [site.specie.Z for site in structure],
    )
    standardized = spglib.standardize_cell(
        cell,
        to_primitive=not args.conventional,
        no_idealize=False,
        symprec=args.tolerance,
    )
    if standardized is None:
        raise SystemExit(
            "ERROR: spglib could not determine the symmetry of the structure "
            f"(tolerance {args.tolerance}); try another --tolerance."
        )
    lattice_matrix, positions, atomic_numbers = standardized
    dataset = spglib.get_symmetry_dataset(standardized, symprec=1e-5)

    def field(name):  # spglib < 2.4 returns a dict, >= 2.4 an object
        if isinstance(dataset, dict):
            return dataset.get(name)
        return getattr(dataset, name, None)

    if args.output:
        output = args.output
    elif args.cell.endswith(".cif"):
        output = args.cell[: -len(".cif")]
    else:
        output = f"{args.cell}_POSCAR"
    if os.path.exists(output):
        print(f"NOTE: overwriting existing {output}")
    with open(output, "w") as handle:
        handle.write("\n".join(poscar_lines(lattice_matrix, positions, atomic_numbers)))

    cell_kind = "conventional" if args.conventional else "primitive"
    symbol = str(field("international")).replace("_", "")
    print("\n* CIF -> POSCAR *")
    print(f"input      : {args.cell}")
    print(f"output     : {output}")
    print(
        f"space group: {symbol} (No. {int(field('number'))}), "
        f"tolerance {args.tolerance}"
    )
    print(f"{cell_kind} cell, {len(atomic_numbers)} atoms\n")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if not os.path.isfile(args.cell):
        raise SystemExit(f"ERROR: POSCAR file not found: {args.cell}")
    try:
        from pymatgen.core import Structure
    except ImportError as exc:
        raise SystemExit(
            "ERROR: --poscar2cif requires pymatgen (pip install pymatgen)."
        ) from exc

    try:
        structure = Structure.from_file(args.cell)
    except Exception as exc:
        raise SystemExit(f"ERROR: could not read {args.cell} as a structure: {exc}")

    title = os.path.basename(args.cell)
    lines, info = bilbao_cif_lines(structure, args.tolerance, title)

    output = args.output or f"{args.cell}.cif"
    if os.path.exists(output):
        print(f"NOTE: overwriting existing {output}")
    with open(output, "w") as handle:
        handle.write("\n".join(lines))

    print("\n* POSCAR -> Bilbao-style CIF *")
    print(f"input      : {args.cell}")
    print(f"output     : {output}")
    print(
        f"space group: {info['symbol']} (No. {info['number']}), "
        f"tolerance {args.tolerance}"
    )
    print(
        f"{info['n_operations']} symmetry operations, "
        f"{info['n_sites']} independent sites\n"
    )


if __name__ == "__main__":
    main()
