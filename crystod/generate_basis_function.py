"""
Automatic generation of symmetry-adapted polynomial basis functions.

For each requested polynomial order (1st, 2nd, 3rd), all monomials of that
degree are classified into the irreducible representations of a point group,
or of the little group of a k point in a space group.
"""

from __future__ import annotations

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, RawDescriptionHelpFormatter, RawTextHelpFormatter
from fractions import Fraction

from .basis_function import (
    _analyze_point_group,
    _analyze_space_group,
    _parse_function,
)


class MyHelpFormatter(
    RawTextHelpFormatter,
    RawDescriptionHelpFormatter,
    ArgumentDefaultsHelpFormatter,
):
    pass


desc = """
Automatically generate 1st-3rd order polynomial basis functions classified by
irreducible representation.

# Command Examples:
crystod-group --generate-basis --point-group m-3m
crystod-group --generate-basis --point-group m-3m --order 2
crystod-group --generate-basis --space-group Pm-3m --kpoint 0 0 0
"""

DEGREE_MONOMIALS: dict[int, list[str]] = {
    1: ["x", "y", "z"],
    2: ["x^2", "y^2", "z^2", "xy", "yz", "zx"],
    3: [
        "x^3",
        "y^3",
        "z^3",
        "x^2y",
        "x^2z",
        "y^2x",
        "y^2z",
        "z^2x",
        "z^2y",
        "xyz",
    ],
}

_ORDER_TITLES = {1: "1st order (linear)", 2: "2nd order (quadratic)", 3: "3rd order (cubic)"}


def _parse_fractional_float(value: str) -> float:
    try:
        return float(Fraction(value))
    except Exception:
        return float(value)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=desc, formatter_class=MyHelpFormatter)
    parser.add_argument(
        "--point-group",
        "-pg",
        default=None,
        help="Point-group label, e.g. m-3m or 4/mmm.",
    )
    parser.add_argument(
        "--space-group",
        "-sg",
        default=None,
        help="Space-group symbol in standard setting, e.g. Pm-3m.",
    )
    parser.add_argument(
        "--kpoint",
        nargs=3,
        type=_parse_fractional_float,
        default=None,
        help="Primitive-basis k point for space-group analysis.",
    )
    parser.add_argument(
        "--order",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        choices=[1, 2, 3],
        help="Polynomial orders to generate.",
    )
    parser.add_argument(
        "--show-irrep-table",
        action="store_true",
        help="Show the irrep character table before the analysis.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if bool(args.point_group) == bool(args.space_group):
        parser.error("Specify exactly one of --point-group or --space-group.")
    if args.space_group and args.kpoint is None:
        parser.error("--space-group requires --kpoint.")

    orders = sorted(set(args.order))
    for position, order in enumerate(orders):
        seed_expressions = [_parse_function(monomial) for monomial in DEGREE_MONOMIALS[order]]
        show_table = args.show_irrep_table and position == 0

        print("=" * 60)
        print(f"  {_ORDER_TITLES[order]} basis functions")
        print("=" * 60)
        if args.space_group:
            print(
                _analyze_space_group(
                    args.space_group,
                    args.kpoint,
                    seed_expressions,
                    show_table,
                )
            )
        else:
            print(_analyze_point_group(args.point_group, seed_expressions, show_table))
        print("")


if __name__ == "__main__":
    main()
