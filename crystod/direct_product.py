from __future__ import annotations

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from dataclasses import dataclass, field

import numpy as np
from phonopy.phonon.character_table import character_table as all_character_tables


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="crystod-group --product",
        description=(
            "Calculate direct products among irreducible representations "
            "of a point group."
        ),
        formatter_class=RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  crystod-group --product T2g T2g T1u\n --point-group m-3m"
            '  crystod-group --product E1u E1u --point-group "6/mmm"'
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
    parser.add_argument(
        "--symmetric",
        action="store_true",
        help="Also decompose the symmetric square [IR x IR] (two identical irreps).",
    )
    parser.add_argument(
        "--antisymmetric",
        action="store_true",
        help="Also decompose the antisymmetric square {IR x IR} (two identical irreps).",
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
    """Characters of the direct product of point-group irreps.

    The first step of ``crystod-group --product IRREP... --pg PG``: the
    character of a direct product is the product of the characters, class
    by class.

    Args:
        ct: Character table from ``crystod.group.get_character_table``.
        point_group: Point-group label (used in the error message only).
        irreps: Irrep labels to multiply, e.g. ``["T2g", "T2g", "T1u"]``.

    Returns:
        The product characters, one per class in the order of
        ``ct["rotation_list"]``.

    Raises:
        SystemExit: An irrep label is not in the table; the message lists
            the available labels (``ValueError`` when called through
            ``crystod.group``).

    Example:
        >>> from crystod import group
        >>> ct = group.get_character_table("m-3m")
        >>> chi = group.direct_product_character(ct, "m-3m", ["T2g", "T2g"])
        >>> chi
        array([9., 0., 1., 1., 1., 9., 1., 0., 1., 1.])
        >>> {k: n for k, n in group.decompose_representation(ct, chi).items() if n}
        {'A1g': 1, 'Eg': 1, 'T1g': 1, 'T2g': 1}
    """
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


def _class_names(ct: dict) -> list[str]:
    """Class names in character order (point group 1 stores the string "E")."""
    names = ct["rotation_list"]
    return [names] if isinstance(names, str) else list(names)


def _vector_table(ct: dict) -> dict:
    """``ct`` with every irrep character as a 1-d array (point group 1 stores
    scalars), as ``crystod.multiplet._GroupClasses`` indexes them."""
    table = dict(ct)
    table["character_table"] = {
        name: np.atleast_1d(np.asarray(values)) for name, values in ct["character_table"].items()
    }
    return table


def decompose_representation(ct: dict, reducible_character: np.ndarray) -> dict[str, int]:
    """Reduce a character vector into the irreps of a point group.

    The second step of ``crystod-group --product`` with a point group, and
    the reduction used by ``--basis``: the reduction formula with the class
    sizes taken from the table, divided by the norm of each irrep character,
    so that an entry combining a complex-conjugate pair of irreps (norm 2,
    e.g. E of 4 or Eg of m-3) counts once per real copy.  The
    representation must be real (products and polynomial spaces are).

    Args:
        ct: Character table from ``crystod.group.get_character_table``.
        reducible_character: Characters of the reducible representation,
            one per class in the order of ``ct["rotation_list"]``.

    Returns:
        ``{irrep label: multiplicity}`` over every irrep of the table (zeros
        included).
    """
    # class sizes in the order of rotation_list (the order of the
    # characters); mapping_table may list the classes in another order (-4)
    multiplicities = np.array(
        [np.asarray(ct["mapping_table"][name]).shape[0] for name in _class_names(ct)],
        dtype=float,
    )

    results: dict[str, int] = {}
    for irrep, character in ct["character_table"].items():
        char_array = np.atleast_1d(np.asarray(character, dtype=float))
        numerator = np.dot(char_array, multiplicities * reducible_character)
        numerator = np.real_if_close(numerator, tol=1000)
        if isinstance(numerator, np.ndarray):
            numerator = numerator.item()
        if isinstance(numerator, complex):
            numerator = numerator.real
        # the tables list physically irreducible (real) irreps: a pair of
        # complex-conjugate irreps (E of 3, 4, 6, m-3, ...) appears as one
        # entry whose character has norm 2 instead of 1
        norm = float(np.dot(char_array, multiplicities * char_array))
        results[irrep] = round(float(numerator) / norm)

    return results


def _check_irrep(ct: dict, point_group: str, irrep: str) -> None:
    all_irreps = list(ct["character_table"].keys())
    if irrep not in all_irreps:
        raise SystemExit(
            f'ERROR: "{irrep}" is not in irreps of {point_group}.\n'
            f"Choose from: {', '.join(all_irreps)}"
        )


def _square_characters(ct: dict, point_group: str, irrep: str, lam: tuple[int, ...]) -> np.ndarray:
    """Characters of the Schur functor ``S^lam`` (``(2,)`` or ``(1, 1)``) of
    a point-group irrep, class by class (the Frobenius formula of
    ``crystod.multiplet``, with the power map from the rotation matrices)."""
    from .multiplet import _GroupClasses, _schur_functor_characters

    _check_irrep(ct, point_group, irrep)
    classes = _GroupClasses(_vector_table(ct))
    return _schur_functor_characters(classes, irrep, lam, classes.power_classes(2))


def symmetric_square(ct: dict, point_group: str, irrep: str) -> dict[str, int]:
    """Decomposition of the symmetrized square ``[Gamma x Gamma]``.

    The computation behind ``crystod-group --product IR IR --pg PG
    --symmetric``: the character ``(chi(g)^2 + chi(g^2)) / 2`` of the
    symmetric part of ``Gamma x Gamma`` (the irreps of quadratic forms in
    the components of ``Gamma``, of the Jahn-Teller coupling of a degenerate
    level, and of the orbital part of two-electron spin singlets), reduced
    into the irreps of the point group.

    Args:
        ct: Character table from ``crystod.group.get_character_table``.
        point_group: Point-group label (used in the error message only).
        irrep: Label of the irrep, e.g. ``"T2g"``.

    Returns:
        ``{irrep label: multiplicity}`` over every irrep of the table (zeros
        included), as ``decompose_representation`` returns it.

    Raises:
        SystemExit: The irrep label is not in the table (``ValueError``
            when called through ``crystod.group``).

    Example:
        >>> from crystod import group
        >>> ct = group.get_character_table("m-3m")
        >>> {k: n for k, n in group.symmetric_square(ct, "m-3m", "T2g").items() if n}
        {'A1g': 1, 'Eg': 1, 'T2g': 1}
    """
    return decompose_representation(ct, _square_characters(ct, point_group, irrep, (2,)))


def antisymmetric_square(ct: dict, point_group: str, irrep: str) -> dict[str, int]:
    """Decomposition of the antisymmetrized square ``{Gamma x Gamma}``.

    The computation behind ``crystod-group --product IR IR --pg PG
    --antisymmetric``: the character ``(chi(g)^2 - chi(g^2)) / 2`` of the
    antisymmetric part of ``Gamma x Gamma`` (the orbital part of
    two-electron spin triplets; for a vector irrep it contains the axial
    vector), reduced into the irreps of the point group.

    Args:
        ct: Character table from ``crystod.group.get_character_table``.
        point_group: Point-group label (used in the error message only).
        irrep: Label of the irrep, e.g. ``"T2g"``.

    Returns:
        ``{irrep label: multiplicity}`` over every irrep of the table (zeros
        included).

    Raises:
        SystemExit: The irrep label is not in the table (``ValueError``
            when called through ``crystod.group``).

    Example:
        >>> from crystod import group
        >>> ct = group.get_character_table("m-3m")
        >>> {k: n for k, n in group.antisymmetric_square(ct, "m-3m", "T2g").items() if n}
        {'T1g': 1}
    """
    return decompose_representation(ct, _square_characters(ct, point_group, irrep, (1, 1)))


def _sum_text(results: dict[str, int]) -> str:
    """``A1g + Eg + 2T2g`` (nonzero terms in table order) or ``(none)``."""
    parts = [
        (f"{count}" if count > 1 else "") + name
        for name, count in results.items()
        if count > 0
    ]
    return " + ".join(parts) if parts else "(none)"


def _irrep_dimension(ct: dict, irrep: str) -> int:
    index = _class_names(ct).index("E")
    return int(round(float(np.atleast_1d(np.asarray(ct["character_table"][irrep]))[index])))


def format_square_result(
    ct: dict, irrep: str, kind: str, results: dict[str, int]
) -> str:
    """The ``* Symmetric square *`` / ``* Antisymmetric square *`` block.

    Args:
        ct: Character table (for the irrep dimensions).
        irrep: Label of the squared irrep.
        kind: ``"symmetric"`` or ``"antisymmetric"``.
        results: The decomposition from ``symmetric_square`` or
            ``antisymmetric_square``.

    Returns:
        The block as one string (no trailing newline): the title line, the
        decomposition (``[T2g x T2g] = ...`` for the symmetric and
        ``{T2g x T2g} = ...`` for the antisymmetric square) and the
        dimension check.
    """
    n = _irrep_dimension(ct, irrep)
    if kind == "symmetric":
        title, left, dimension = "Symmetric square", f"[{irrep} x {irrep}]", n * (n + 1) // 2
    else:
        title, left, dimension = "Antisymmetric square", f"{{{irrep} x {irrep}}}", n * (n - 1) // 2
    dims = " + ".join(
        (f"{count}x" if count > 1 else "") + str(_irrep_dimension(ct, name))
        for name, count in results.items()
        if count > 0
    )
    text = (
        f"* {title} *\n"
        f"{left} = {_sum_text(results)}\n"
        f"dimension: {dimension}" + (f" = {dims}" if dims else "")
    )
    total = sum(count * _irrep_dimension(ct, name) for name, count in results.items())
    if total != dimension or any(count < 0 for count in results.values()):
        text += "\nWARNING: dimension mismatch - please report this case."
    return text


@dataclass
class JahnTellerModes:
    """Jahn-Teller modes of a degenerate level, or the symmetry-allowed
    pseudo-Jahn-Teller coupling modes of two levels.

    Attributes:
        point_group: Point-group label.
        irreps: ``(IR,)`` for the Jahn-Teller effect of a degenerate level,
            ``(IR1, IR2)`` for the pseudo-Jahn-Teller mixing of two levels.
        pseudo: ``True`` for two irreps.
        product: Decomposition of ``[IR x IR]`` (Jahn-Teller) or of the
            plain product ``IR1 x IR2`` (pseudo-Jahn-Teller), nonzero terms
            only, in table order.
        identity: Label of the totally symmetric irrep.
        active: ``product`` without the totally symmetric irrep.  For one
            irrep: the Jahn-Teller active modes, which couple linearly to
            the degenerate level and lower its symmetry (the Jahn-Teller
            theorem).  For two irreps: the symmetry-allowed coupling modes
            only (also available as ``allowed``), a necessary condition for
            the pseudo-Jahn-Teller effect; whether the mixing destabilizes
            the structure depends on the energy gap and the vibronic
            coupling strength, which symmetry does not give.
        first_order: For two identical labels, ``[IR x IR]`` without the
            identity irrep (the first-order Jahn-Teller modes of that
            level, empty for a nondegenerate irrep); ``None`` otherwise.
    """

    point_group: str
    irreps: tuple[str, ...]
    pseudo: bool
    product: dict[str, int]
    identity: str
    active: dict[str, int] = field(default_factory=dict)
    first_order: dict[str, int] | None = None

    @property
    def allowed(self) -> dict[str, int]:
        """The symmetry-allowed coupling modes (same as ``active``)."""
        return self.active


def jahn_teller_modes(
    point_group: str, irrep: str, irrep2: str | None = None
) -> JahnTellerModes:
    """Jahn-Teller active modes of a degenerate level, or the symmetry-allowed
    pseudo-Jahn-Teller coupling modes of two levels.

    The computation behind ``crystod-group --jahn-teller IR [IR2] --pg PG``:
    a vibrational mode Q couples linearly to a degenerate level of symmetry
    ``Gamma`` when ``Gamma_Q`` occurs in the symmetric square
    ``[Gamma x Gamma]`` (spinless levels); the totally symmetric part does
    not lower the symmetry, so the Jahn-Teller active modes are
    ``[Gamma x Gamma]`` minus the identity irrep (the Jahn-Teller theorem).
    For two levels a mode Q can mix them when ``Gamma_Q`` occurs in the
    plain product ``Gamma1 x Gamma2``; the non-totally-symmetric irreps of
    the product are the symmetry-allowed pseudo-Jahn-Teller coupling modes.
    This is a necessary condition only: the mixing lowers the curvature of
    the energy along Q to ``K0 - 2|F|^2/Delta`` (vibronic constant F, gap
    Delta), and whether the high-symmetry structure becomes unstable
    depends on F and Delta, which symmetry does not give.  Which of these
    symmetries actually occur among the vibrations depends on the
    structure.

    Args:
        point_group: Point-group label, e.g. ``"m-3m"``.
        irrep: Label of the (degenerate) level, e.g. ``"Eg"``.
        irrep2: Label of a second level for the pseudo-Jahn-Teller product,
            or ``None``.

    Returns:
        A ``JahnTellerModes`` record.

    Raises:
        SystemExit: Unknown point group or irrep label (``ValueError`` when
            called through ``crystod.group``).

    Example:
        >>> from crystod import group
        >>> group.jahn_teller_modes("m-3m", "T1u").active
        {'Eg': 1, 'T2g': 1}
        >>> group.jahn_teller_modes("m-3m", "Eg").active
        {'Eg': 1}
    """
    ct = _get_character_table(point_group)
    identity = next(
        name
        for name, characters in ct["character_table"].items()
        if np.allclose(np.asarray(characters, dtype=float), 1.0)
    )
    if irrep2 is None:
        decomposition = symmetric_square(ct, point_group, irrep)
        irreps: tuple[str, ...] = (irrep,)
    else:
        decomposition = decompose_representation(
            ct, direct_product_character(ct, point_group, [irrep, irrep2])
        )
        irreps = (irrep, irrep2)
    product = {name: count for name, count in decomposition.items() if count > 0}
    active = {name: count for name, count in product.items() if name != identity}
    first_order = None
    if irrep2 is not None and irrep2 == irrep:
        first_order = {
            name: count
            for name, count in symmetric_square(ct, point_group, irrep).items()
            if count > 0 and name != identity
        }
    return JahnTellerModes(
        point_group=point_group,
        irreps=irreps,
        pseudo=irrep2 is not None,
        product=product,
        identity=identity,
        active=active,
        first_order=first_order,
    )


def format_jahn_teller(result: JahnTellerModes) -> str:
    """Text report of ``jahn_teller_modes``, as printed by
    ``crystod-group --jahn-teller``.

    Args:
        result: A ``JahnTellerModes`` record.

    Returns:
        The ``* Point group *`` and ``* Jahn-Teller active modes *`` (or
        ``* Pseudo-Jahn-Teller coupling *``) blocks as one string (no
        trailing newline).
    """
    lines = ["* Point group *", result.point_group, ""]
    active = _sum_text(result.active)
    if result.pseudo:
        first, second = result.irreps
        lines.append("* Pseudo-Jahn-Teller coupling *")
        lines.append(f"{first} x {second} = {_sum_text(result.product)}")
        if result.first_order is not None:
            if result.first_order:
                lines.append(
                    "same degenerate level (first-order Jahn-Teller): "
                    f"[{first} x {first}] - {result.identity} = "
                    f"{_sum_text(result.first_order)}")
            else:
                lines.append("same level: nondegenerate, no first-order "
                             "Jahn-Teller effect")
        prefix = ""
        if result.first_order is not None:
            # the same label twice: the plain product (antisymmetric square
            # included) applies to two different levels of that symmetry
            prefix = f"two different levels of symmetry {first}, "
        if result.active:
            lines.append(f"{prefix}symmetry-allowed coupling modes: {active}")
        else:
            lines.append(f"{prefix}symmetry-allowed coupling modes: none (only the "
                         "totally symmetric irrep)")
        lines.extend([
            "  (necessary condition only: a mode Q can mix the two levels "
            "when Gamma_Q",
            "   occurs in the product; whether the mixing destabilizes the "
            "high-symmetry",
            "   structure depends on the energy gap and the coupling "
            "strength, which",
            "   symmetry does not give)",
        ])
    else:
        irrep = result.irreps[0]
        lines.append("* Jahn-Teller active modes *")
        lines.append(f"[{irrep} x {irrep}] = {_sum_text(result.product)}")
        if result.active:
            lines.append(f"JT-active: {active}")
            lines.append("  (linear vibronic coupling allowed by symmetry; "
                         "the Jahn-Teller theorem)")
        else:
            lines.append("JT-active: none (a nondegenerate level has no "
                         "Jahn-Teller effect)")
    return "\n".join(lines)


def jahn_teller_main(argv: list[str] | None = None) -> None:
    """``crystod-group --jahn-teller IR [IR2] --pg PG``."""
    parser = ArgumentParser(prog="crystod-group --jahn-teller")
    parser.add_argument("--point-group", "-pg", required=True)
    parser.add_argument("--irreps", nargs="+", required=True)
    args = parser.parse_args(argv)
    irreps = _flatten_irreps(args.irreps)
    if len(irreps) > 2:
        parser.error("--jahn-teller takes one irrep (Jahn-Teller) or two "
                     "(pseudo-Jahn-Teller).")
    result = jahn_teller_modes(args.point_group, irreps[0],
                               irreps[1] if len(irreps) == 2 else None)
    print()
    print(format_jahn_teller(result))


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
        text = format_result(point_group, irreps, results).strip("\n")
        if args.show_irrep_table:
            # the irrep table above already carries the * Point group * block
            text = text.split("\n\n", 1)[1]
        if args.symmetric or args.antisymmetric:
            if len(irreps) != 2 or irreps[0] != irreps[1]:
                parser.error("--symmetric/--antisymmetric need two identical "
                             "irreps, e.g. --product T2g T2g.")
            if args.symmetric:
                square = symmetric_square(ct, point_group, irreps[0])
                text += "\n\n" + format_square_result(ct, irreps[0], "symmetric", square)
            if args.antisymmetric:
                square = antisymmetric_square(ct, point_group, irreps[0])
                text += "\n\n" + format_square_result(ct, irreps[0], "antisymmetric", square)
        outputs.append(text)
    elif not args.show_irrep_table:
        parser.error("Specify --irreps and/or --show-irrep-table.")

    print("\n" + "\n\n".join(outputs))
