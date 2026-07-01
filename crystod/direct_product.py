from __future__ import annotations

from argparse import ArgumentParser, RawDescriptionHelpFormatter

import numpy as np
from phonopy.phonon.character_table import character_table as all_character_tables


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="crystod --direct-product",
        description=(
            "Calculate direct products among irreducible representations "
            "of a point group."
        ),
        formatter_class=RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  crystod --direct-product --point-group m-3m --irreps T2g T2g T1u\n"
            '  crystod --direct-product --point-group "6/mmm" --irreps E1u E1u'
        ),
    )
    parser.add_argument(
        "--point-group",
        "-pg",
        required=True,
        help="Point group label, e.g. m-3m or 6/mmm.",
    )
    parser.add_argument(
        "--irreps",
        "-irreps",
        nargs="*",
        default=None,
        help="Irrep labels to multiply, e.g. T2g T2g T1u.",
    )
    parser.add_argument(
        "--show-irrep-table",
        action="store_true",
        help="Show the point-group character table.",
    )
    return parser


def _flatten_irreps(raw_irreps: list[str]) -> list[str]:
    irreps: list[str] = []
    for item in raw_irreps:
        irreps.extend(item.split())
    return irreps


def _format_character_value(value) -> str:
    scalar = np.asarray(value).item()
    if isinstance(scalar, complex):
        if abs(scalar.imag) < 1e-10:
            scalar = scalar.real
        else:
            return f"{scalar.real:.4g}{scalar.imag:+.4g}j"
    if abs(float(scalar) - round(float(scalar))) < 1e-10:
        return str(int(round(float(scalar))))
    return f"{float(scalar):.4g}"


def _get_character_table(point_group: str) -> dict:
    try:
        return all_character_tables[point_group][0]
    except KeyError as exc:
        available = ", ".join(all_character_tables.keys())
        raise SystemExit(
            f'ERROR: "{point_group}" is not in the point groups.\n'
            f"Choose from: {available}"
        ) from exc


def format_irrep_table(point_group: str, ct: dict) -> str:
    class_names = list(ct["rotation_list"])
    class_sizes = [
        np.asarray(ct["mapping_table"][class_name]).shape[0]
        for class_name in class_names
    ]
    irrep_names = list(ct["character_table"].keys())

    header = ["irrep"] + [f"{name}({size})" for name, size in zip(class_names, class_sizes)]
    rows = []
    for irrep_name in irrep_names:
        characters = ct["character_table"][irrep_name]
        rows.append(
            [irrep_name] + [_format_character_value(value) for value in characters]
        )

    widths = [len(item) for item in header]
    for row in rows:
        for idx, item in enumerate(row):
            widths[idx] = max(widths[idx], len(item))

    lines = []
    lines.append("  ".join(item.rjust(widths[idx]) for idx, item in enumerate(header)))
    for row in rows:
        lines.append("  ".join(item.rjust(widths[idx]) for idx, item in enumerate(row)))

    return (
        "\n"
        "* Point group *\n"
        f"{point_group}\n\n"
        "* IrRep Table *\n"
        "table:\n"
        + "\n".join(lines)
        + "\n"
    )


def direct_product_character(ct: dict, point_group: str, irreps: list[str]) -> np.ndarray:
    all_irreps = list(ct["character_table"].keys())
    irreps_character = []
    for irrep in irreps:
        if irrep not in all_irreps:
            available = ", ".join(all_irreps)
            raise SystemExit(
                f'ERROR: "{irrep}" is not in irreps of {point_group}.\n'
                f"Choose from: {available}"
            )
        irreps_character.append(np.asarray(ct["character_table"][irrep], dtype=float))

    return np.prod(np.stack(irreps_character, axis=0), axis=0)


def decompose_representation(ct: dict, reducible_character: np.ndarray) -> dict[str, int]:
    multiplicities = np.array(
        [np.asarray(values).shape[0] for values in ct["mapping_table"].values()],
        dtype=float,
    )
    denominator = float(multiplicities.sum())

    results: dict[str, int] = {}
    for irrep, character in ct["character_table"].items():
        char_array = np.asarray(character, dtype=float)
        numerator = np.dot(char_array, multiplicities * reducible_character)
        numerator = np.real_if_close(numerator, tol=1000)
        if isinstance(numerator, np.ndarray):
            numerator = numerator.item()
        if isinstance(numerator, complex):
            numerator = numerator.real
        results[irrep] = round(float(numerator) / denominator)

    return results


def format_result(point_group: str, irreps: list[str], results: dict[str, int]) -> str:
    formula = "*".join(irreps)
    result = " + ".join(f"{value}({key})" for key, value in results.items() if value > 0)
    return (
        "\n"
        "* Point group *\n"
        f"{point_group}\n\n"
        "* Direct product *\n"
        f"{formula}\n\n"
        "* Result *\n"
        f" {result}\n"
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    point_group = args.point_group
    ct = _get_character_table(point_group)
    outputs: list[str] = []

    if args.show_irrep_table:
        outputs.append(format_irrep_table(point_group, ct).strip("\n"))

    irreps = _flatten_irreps(args.irreps or [])
    if irreps:
        reducible_character = direct_product_character(ct, point_group, irreps)
        results = decompose_representation(ct, reducible_character)
        outputs.append(format_result(point_group, irreps, results).strip("\n"))
    elif not args.show_irrep_table:
        parser.error("Specify --irreps and/or --show-irrep-table.")

    print("\n\n".join(outputs))
