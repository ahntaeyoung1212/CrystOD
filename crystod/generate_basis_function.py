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
    _parse_function,
    _point_group_blocks,
    _render_blocks,
    _space_group_blocks,
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


def format_generated_basis(
    orders: list[int],
    point_group: str | None = None,
    space_group: str | None = None,
    kpoint: list[float] | None = None,
    show_irrep_table: bool = False,
) -> str:
    """Text report of ``crystod-group --generate-basis``.

    The group blocks (``* Point group *``, or ``* Space group *``,
    ``* Little group of k *`` and ``* k-point (primitive) *``, then the
    ``* IrRep Table *`` with ``show_irrep_table``) are printed once; every
    order follows with its own ``* Input basis functions: 1st order
    (linear) *``, ``* Decomposition: ... *`` ... blocks.

    Args:
        orders: Polynomial orders, any of 1, 2, 3.
        point_group: Point-group label (exclusive with ``space_group``).
        space_group: Space-group symbol; needs ``kpoint``.
        kpoint: k point in the primitive basis.
        show_irrep_table: Add the character table of the (little) group.

    Returns:
        The report as one string (leading newline, no trailing newline).
    """
    blocks: list[tuple[str, list[str]]] = []
    for position, order in enumerate(sorted(set(orders))):
        seed_expressions = [_parse_function(monomial) for monomial in DEGREE_MONOMIALS[order]]
        if space_group:
            header_blocks, analysis_blocks = _space_group_blocks(
                space_group, kpoint, seed_expressions, show_irrep_table
            )
        else:
            header_blocks, analysis_blocks = _point_group_blocks(
                point_group, seed_expressions, show_irrep_table
            )
        if position == 0:
            blocks.extend(header_blocks)
        blocks.extend(
            (f"{title}: {_ORDER_TITLES[order]}", lines) for title, lines in analysis_blocks
        )
    return _render_blocks(blocks)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if bool(args.point_group) == bool(args.space_group):
        parser.error("Specify exactly one of --point-group or --space-group.")
    if args.space_group and args.kpoint is None:
        parser.error("--space-group requires --kpoint.")

    print(format_generated_basis(
        args.order,
        point_group=args.point_group,
        space_group=args.space_group,
        kpoint=args.kpoint,
        show_irrep_table=args.show_irrep_table,
    ))

if __name__ == "__main__":
    main()
