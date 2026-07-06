from __future__ import annotations

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, RawDescriptionHelpFormatter, RawTextHelpFormatter
import re

import numpy as np
from phonopy.phonon.character_table import character_table as all_character_tables
from phonopy.structure.cells import get_primitive_matrix_by_centring
from spgrep.core import get_spacegroup_irreps_from_primitive_symmetry
from spgrep.representation import get_character
import spglib
from sympy import Matrix, Poly, Rational, expand, nsimplify, simplify, sqrt, symbols
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from .direct_product import _format_character_value, decompose_representation
from .irreptables_compat import load_irreptables
from .runtime_compat import get_spacegroup_type
from .operations import get_seitz_symbol

IrrepTable, Irrep = load_irreptables()


class MyHelpFormatter(
    RawTextHelpFormatter,
    RawDescriptionHelpFormatter,
    ArgumentDefaultsHelpFormatter,
):
    pass


desc = """
Classify polynomial basis functions into irreducible representations of a point group.

# Command Examples:
crystod --basis-function x y z --point-group m-3m
crystod --basis-function x y z --space-group Pm-3m --kpoint 0 0 0
crystod --basis-function xyz --space-group Pm-3m --kpoint 0.5 0.3 0 --show-irrep-table
crystod --basis-function z^2 --point-group m-3m
crystod --basis-function "x(y^2-z^2)" --point-group m-3m
"""

_PARSER_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

_X, _Y, _Z = symbols("x y z")
_BASE_VECTOR = Matrix([_X, _Y, _Z])


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=desc, formatter_class=MyHelpFormatter)
    parser.add_argument(
        "--basis-function",
        nargs="+",
        required=True,
        help="Polynomial basis functions in x, y, z. Quote expressions containing parentheses.",
    )
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
        type=float,
        default=None,
        help="Primitive-basis k-point for space-group analysis.",
    )
    parser.add_argument(
        "--show-irrep-table",
        action="store_true",
        help="Show the point-group character table before the analysis.",
    )
    return parser


def _get_character_table(point_group: str) -> dict:
    try:
        return all_character_tables[point_group][0]
    except KeyError as exc:
        available = ", ".join(all_character_tables.keys())
        raise SystemExit(
            f'ERROR: "{point_group}" is not in the point groups.\n'
            f"Choose from: {available}"
        ) from exc


def _normalize_symbol(text: str) -> str:
    return "".join(text.split()).lower()


def _resolve_space_group_type(space_group_symbol: str) -> dict:
    requested = _normalize_symbol(space_group_symbol)
    matches = []
    for hall_number in range(1, 531):
        info = get_spacegroup_type(spglib.get_spacegroup_type(hall_number))
        candidates = [
            info.international_short,
            info.international,
            info.international_full,
        ]
        if any(_normalize_symbol(candidate) == requested for candidate in candidates):
            matches.append(info)
    if not matches:
        raise SystemExit(f'ERROR: "{space_group_symbol}" is not recognized as a standard space-group symbol.')
    matches.sort(key=lambda item: item.hall_number)
    return matches[0]


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


def _parse_function(raw_expression: str):
    expr = expand(
        parse_expr(
            raw_expression,
            local_dict={"x": _X, "y": _Y, "z": _Z},
            transformations=_PARSER_TRANSFORMS,
        )
    )
    if expr.free_symbols - {_X, _Y, _Z}:
        unknown = ", ".join(sorted(symbol.name for symbol in expr.free_symbols - {_X, _Y, _Z}))
        raise SystemExit(f'ERROR: basis function "{raw_expression}" uses unsupported symbols: {unknown}')
    return expr


def _rotation_classes(ct: dict) -> tuple[list[str], list[np.ndarray], list[int], list[str]]:
    class_names = list(ct["rotation_list"])
    class_sizes: list[int] = []
    all_rotations: list[np.ndarray] = []
    operation_classes: list[str] = []
    for class_name in class_names:
        rotations = [np.array(rotation, dtype=int) for rotation in ct["mapping_table"][class_name]]
        class_sizes.append(len(rotations))
        all_rotations.extend(rotations)
        operation_classes.extend([class_name] * len(rotations))
    return class_names, all_rotations, class_sizes, operation_classes


def _apply_rotation(expr, rotation: np.ndarray):
    rotated_base = Matrix(rotation) * _BASE_VECTOR
    substituted = expr.subs({
        _X: rotated_base[0],
        _Y: rotated_base[1],
        _Z: rotated_base[2],
    }, simultaneous=True)
    return expand(substituted)


def _expr_to_key(expr) -> str:
    poly = Poly(expand(expr), _X, _Y, _Z, domain="EX")
    return tuple(
        (monomial, nsimplify(coeff))
        for monomial, coeff in sorted(poly.as_dict().items())
        if coeff != 0
    )


def _monomial_order(expressions: list) -> list[tuple[int, int, int]]:
    exponents: set[tuple[int, int, int]] = set()
    for expr in expressions:
        poly = Poly(expr, _X, _Y, _Z, domain="EX")
        exponents.update(poly.as_dict().keys())
    return sorted(exponents, key=lambda item: (sum(item), item[0], item[1], item[2]))


def _vectorize(expr, monomials: list[tuple[int, int, int]]) -> Matrix:
    poly = Poly(expr, _X, _Y, _Z, domain="EX")
    coeff_map = poly.as_dict()
    return Matrix([coeff_map.get(monomial, 0) for monomial in monomials])


def _normalize_expr(expr):
    expanded = expand(expr)
    if expanded == 0:
        return expanded
    poly = Poly(expanded, _X, _Y, _Z, domain="EX")
    coeffs = [coeff for _, coeff in sorted(poly.as_dict().items()) if coeff != 0]
    if not coeffs:
        return expanded
    first_coeff = coeffs[0]
    return expand(nsimplify(expanded / first_coeff))


def _cartesianize_rotations(rotations: list[np.ndarray]) -> list[Matrix]:
    if not rotations:
        return []

    hexagonal_basis = Matrix(
        [
            [1, Rational(-1, 2), 0],
            [0, sqrt(3) / 2, 0],
            [0, 0, 1],
        ]
    )
    hexagonal_basis_inv = hexagonal_basis.inv()

    cartesian_rotations: list[Matrix] = []
    for rotation in rotations:
        rotation_matrix = Matrix(rotation)
        if rotation_matrix.T * rotation_matrix == Matrix.eye(3):
            cartesian_rotations.append(rotation_matrix)
            continue

        # Hexagonal/trigonal character-table rotations are expressed in
        # crystallographic a-b-c axes, so convert them to orthonormal Cartesian
        # axes before acting on polynomial basis functions.
        cartesian_rotations.append(
            simplify(hexagonal_basis * rotation_matrix * hexagonal_basis_inv)
        )
    return cartesian_rotations


def _build_closed_subspace(seed_expressions: list, rotations: list[np.ndarray]):
    orbit = list(seed_expressions)
    seen = {_expr_to_key(expr) for expr in orbit}
    changed = True
    while changed:
        changed = False
        current = list(orbit)
        for expr in current:
            for rotation in rotations:
                rotated_expr = _normalize_expr(_apply_rotation(expr, rotation))
                key = _expr_to_key(rotated_expr)
                if key not in seen:
                    orbit.append(rotated_expr)
                    seen.add(key)
                    changed = True

    monomials = _monomial_order(orbit)
    orbit_vectors = [_vectorize(expr, monomials) for expr in orbit]
    orbit_matrix = Matrix.hstack(*orbit_vectors)
    basis_vectors = orbit_matrix.columnspace()

    basis_expressions = []
    for basis_vector in basis_vectors:
        expr = 0
        for coeff, monomial in zip(basis_vector, monomials):
            if coeff == 0:
                continue
            expr += coeff * (_X ** monomial[0]) * (_Y ** monomial[1]) * (_Z ** monomial[2])
        basis_expressions.append(_normalize_expr(expr))

    normalized_basis_vectors = [_vectorize(expr, monomials) for expr in basis_expressions]
    return monomials, normalized_basis_vectors, basis_expressions


def _representation_matrices(
    basis_matrix: Matrix,
    basis_expressions: list,
    monomials: list[tuple[int, int, int]],
    rotations: list[np.ndarray],
) -> list[Matrix]:
    representation_matrices: list[Matrix] = []
    for rotation in rotations:
        columns = []
        for expr in basis_expressions:
            rotated_expr = expand(_apply_rotation(expr, rotation))
            rotated_vector = _vectorize(rotated_expr, monomials)
            solution, params = basis_matrix.gauss_jordan_solve(rotated_vector)
            if params.shape[0] != 0:
                raise RuntimeError("Failed to build a closed representation matrix.")
            columns.append(solution)
        representation_matrices.append(Matrix.hstack(*columns))
    return representation_matrices


def _character_by_class(
    class_names: list[str],
    class_sizes: list[int],
    rep_matrices: list[Matrix],
) -> np.ndarray:
    chars = []
    offset = 0
    for class_name, class_size in zip(class_names, class_sizes):
        class_chars = []
        for index in range(offset, offset + class_size):
            class_chars.append(complex(rep_matrices[index].trace().evalf()))
        offset += class_size
        chars.append(sum(class_chars) / class_size)
    return np.array(chars, dtype=np.complex128)


def _project_irrep_basis(
    basis_expressions: list,
    rep_matrices: list[Matrix],
    operation_classes: list[str],
    ct: dict,
    irrep_name: str,
) -> list:
    irrep_characters = ct["character_table"][irrep_name]
    class_index = {class_name: index for index, class_name in enumerate(ct["rotation_list"])}
    group_order = len(rep_matrices)
    irrep_dimension = int(round(float(np.asarray(irrep_characters)[0].real)))

    if irrep_dimension == 1:
        representative_matrices: dict[str, Matrix] = {}
        for class_name, rep_matrix in zip(operation_classes, rep_matrices):
            representative_matrices.setdefault(class_name, rep_matrix)

        rows = []
        identity = Matrix.eye(rep_matrices[0].shape[0])
        for class_name in ct["rotation_list"]:
            char_value = nsimplify(irrep_characters[class_index[class_name]])
            rows.extend((representative_matrices[class_name] - char_value * identity).tolist())

        functions = []
        for vector in Matrix(rows).nullspace():
            expr = 0
            for coeff, basis_expr in zip(vector, basis_expressions):
                if coeff == 0:
                    continue
                expr += nsimplify(coeff) * basis_expr
            if expr != 0:
                functions.append(_normalize_expr(expr))
        return functions

    projector = Matrix.zeros(*rep_matrices[0].shape)
    for rep_matrix, class_name in zip(rep_matrices, operation_classes):
        char_value = np.asarray(irrep_characters[class_index[class_name]]).item()
        projector += nsimplify(np.conjugate(char_value)) * rep_matrix
    projector *= Rational(irrep_dimension, group_order)

    functions = []
    for vector in projector.columnspace():
        expr = 0
        for coeff, basis_expr in zip(vector, basis_expressions):
            if coeff == 0:
                continue
            expr += nsimplify(coeff) * basis_expr
        if expr != 0:
            functions.append(_normalize_expr(expr))
    return functions


def _decompose_by_operations(
    rep_characters: np.ndarray,
    irrep_characters_map: dict[str, np.ndarray],
) -> dict[str, int]:
    order = len(rep_characters)
    results: dict[str, int] = {}
    for irrep_name, irrep_characters in irrep_characters_map.items():
        irrep_dimension = int(round(float(np.real_if_close(irrep_characters[0]))))
        multiplicity = np.dot(rep_characters, np.conjugate(irrep_characters)) / order
        multiplicity = np.real_if_close(multiplicity, tol=1000)
        if isinstance(multiplicity, np.ndarray):
            multiplicity = multiplicity.item()
        if isinstance(multiplicity, complex):
            multiplicity = multiplicity.real
        results[irrep_name] = max(0, round(float(multiplicity)))
    return results


def _project_spacegroup_irrep_basis(
    basis_expressions: list,
    rep_matrices: list[Matrix],
    irrep_characters: np.ndarray,
    irrep_name: str,
) -> list:
    group_order = len(rep_matrices)
    irrep_dimension = int(round(float(np.real_if_close(irrep_characters[0]))))
    projector = Matrix.zeros(*rep_matrices[0].shape)
    for rep_matrix, character in zip(rep_matrices, irrep_characters):
        projector += nsimplify(np.conjugate(character)) * rep_matrix
    projector *= Rational(irrep_dimension, group_order)

    functions = []
    for vector in projector.columnspace():
        expr = 0
        for coeff, basis_expr in zip(vector, basis_expressions):
            if coeff == 0:
                continue
            expr += nsimplify(coeff) * basis_expr
        if expr != 0:
            functions.append(_normalize_expr(expr))
    return functions


def _format_kpoint(values: list[float]) -> list[float]:
    return [float(np.round(float(value), 6)) for value in values]


def _match_characters(lhs: np.ndarray, rhs: np.ndarray, atol: float = 1e-5) -> bool:
    if np.allclose(lhs, rhs, atol=atol):
        return True
    if np.allclose(lhs, np.conjugate(rhs), atol=atol):
        return True
    return False


def _get_irt_irreps_at_k(
    irt_table,
    primitive_matrix_inv: np.ndarray,
    kpoint: list[float],
):
    conventional_k = np.array(kpoint) @ primitive_matrix_inv
    return [
        irrep_at_k
        for irrep_at_k in irt_table.irreps
        if np.allclose(irrep_at_k.k, conventional_k, atol=1e-6)
    ]


def _resolve_spacegroup_irrep_labels(
    irreps,
    mapping_little_group: np.ndarray,
    irt_irreps: list,
) -> tuple[list[str], dict[str, str], dict[str, np.ndarray]]:
    generic_labels = [f"irrep_{index + 1}({irrep.shape[1]})" for index, irrep in enumerate(irreps)]
    irrep_characters_map = {
        generic_label: np.array(get_character(irrep), dtype=np.complex128)
        for generic_label, irrep in zip(generic_labels, irreps)
    }
    display_label_map = {generic_label: generic_label for generic_label in generic_labels}

    if not irt_irreps:
        return generic_labels, display_label_map, irrep_characters_map

    used_labels: set[str] = set()
    irt_characters = []
    for irt_irrep in irt_irreps:
        label = f"{irt_irrep.name}({irt_irrep.dim})"
        characters = np.array(
            [irt_irrep.characters[index + 1] for index in mapping_little_group],
            dtype=np.complex128,
        )
        irt_characters.append((label, int(irt_irrep.dim), characters))

    for generic_label in generic_labels:
        spgrep_characters = irrep_characters_map[generic_label]
        irrep_dimension = int(generic_label.split("(")[1][:-1])
        for irt_label, irt_dim, characters in irt_characters:
            if irt_label in used_labels or irt_dim != irrep_dimension:
                continue
            if _match_characters(spgrep_characters, characters):
                display_label_map[generic_label] = irt_label
                used_labels.add(irt_label)
                break

    return generic_labels, display_label_map, irrep_characters_map


def _get_little_group_label(
    rotations: np.ndarray,
    translations: np.ndarray,
) -> str:
    little_group = get_spacegroup_type(spglib.get_spacegroup_type_from_symmetry(rotations, translations))
    international_short = little_group.international_short
    number = little_group.number
    return f"{international_short} ({number})"


def _format_spacegroup_irrep_table(
    little_group_label: str,
    operation_labels: list[str],
    generic_labels: list[str],
    display_label_map: dict[str, str],
    irrep_characters_map: dict[str, np.ndarray],
) -> str:
    header = ["irrep"] + operation_labels
    rows = []
    for generic_label in generic_labels:
        resolved_label = display_label_map[generic_label]
        row_label = generic_label if resolved_label == generic_label else f"{generic_label} = {resolved_label}"
        rows.append(
            [row_label]
            + [_format_character_value(value) for value in irrep_characters_map[generic_label]]
        )

    widths = [len(item) for item in header]
    for row in rows:
        for index, item in enumerate(row):
            widths[index] = max(widths[index], len(item))

    lines = []
    lines.append("  ".join(item.rjust(widths[index]) for index, item in enumerate(header)))
    for row in rows:
        lines.append("  ".join(item.rjust(widths[index]) for index, item in enumerate(row)))

    return (
        "\n"
        "* IrRep Table *\n"
        f"little group: {little_group_label}\n"
        "table:\n"
        + "\n".join(lines)
        + "\n"
    )


def _analyze_point_group(
    point_group: str,
    seed_expressions: list,
    show_irrep_table: bool,
) -> str:
    ct = _get_character_table(point_group)
    class_names, rotations, class_sizes, operation_classes = _rotation_classes(ct)
    cartesian_rotations = _cartesianize_rotations(rotations)

    monomials, basis_vectors, basis_expressions = _build_closed_subspace(seed_expressions, cartesian_rotations)
    basis_matrix = Matrix.hstack(*basis_vectors)
    rep_matrices = _representation_matrices(
        basis_matrix=basis_matrix,
        basis_expressions=basis_expressions,
        monomials=monomials,
        rotations=cartesian_rotations,
    )
    reducible_character = _character_by_class(class_names, class_sizes, rep_matrices)
    multiplicities = decompose_representation(ct, reducible_character)

    outputs: list[str] = []
    if show_irrep_table:
        outputs.append(format_irrep_table(point_group, ct).strip("\n"))

    lines = [
        "",
        "* Point group *",
        f"{point_group}",
        "",
        "* Input basis functions *",
        " " + ", ".join(_format_expression(expr) for expr in seed_expressions),
        "",
        "* Closed basis-function space *",
        f" dimension: {len(basis_expressions)}",
        f" basis: {_format_expression_list(basis_expressions)}",
        "",
        "* Reducible characters *",
    ]
    for class_name, character in zip(class_names, reducible_character):
        lines.append(f"  {class_name}: {_format_character_value(character)}")

    decomposition = " + ".join(
        _format_decomposition_term(key, value)
        for key, value in multiplicities.items()
        if value > 0
    )
    lines.extend([
        "",
        "* Decomposition *",
        f" {decomposition}",
        "",
        "* Irreducible representations for basis functions *",
    ])

    for irrep_name, multiplicity in multiplicities.items():
        if multiplicity <= 0:
            continue
        adapted_functions = _project_irrep_basis(
            basis_expressions=basis_expressions,
            rep_matrices=rep_matrices,
            operation_classes=operation_classes,
            ct=ct,
            irrep_name=irrep_name,
        )
        lines.append(f"  {irrep_name}: {_format_expression_list(adapted_functions)}")

    outputs.append("\n".join(lines).strip("\n"))
    return "\n\n".join(outputs)


def _analyze_space_group(
    space_group_symbol: str,
    kpoint: list[float],
    seed_expressions: list,
    show_irrep_table: bool,
) -> str:
    sg_type = _resolve_space_group_type(space_group_symbol)
    irt_table = IrrepTable(sg_type.number, spinor=False)
    primitive_matrix = np.array(
        get_primitive_matrix_by_centring(sg_type.international_short[0]),
        dtype=float,
    )
    primitive_matrix_inv = np.linalg.inv(primitive_matrix)

    conventional_rotations = np.array([sym.R for sym in irt_table.symmetries], dtype=float)
    conventional_translations = np.array([sym.t for sym in irt_table.symmetries], dtype=float)
    primitive_rotations = np.rint(
        np.array([primitive_matrix_inv @ rotation @ primitive_matrix for rotation in conventional_rotations])
    ).astype(int)
    primitive_translations = np.mod(conventional_translations @ primitive_matrix_inv, 1.0)
    primitive_translations[np.isclose(primitive_translations, 1.0, atol=1e-8)] = 0.0

    irreps, mapping_little_group = get_spacegroup_irreps_from_primitive_symmetry(
        rotations=primitive_rotations,
        translations=primitive_translations,
        kpoint=kpoint,
    )
    irt_irreps = _get_irt_irreps_at_k(irt_table, primitive_matrix_inv, kpoint)

    little_conventional_rotations = np.rint(conventional_rotations[mapping_little_group]).astype(int)
    little_primitive_rotations = primitive_rotations[mapping_little_group]
    little_primitive_translations = primitive_translations[mapping_little_group]
    cartesian_little_rotations = _cartesianize_rotations([rotation for rotation in little_conventional_rotations])
    monomials, basis_vectors, basis_expressions = _build_closed_subspace(
        seed_expressions,
        cartesian_little_rotations,
    )
    basis_matrix = Matrix.hstack(*basis_vectors)
    rep_matrices = _representation_matrices(
        basis_matrix=basis_matrix,
        basis_expressions=basis_expressions,
        monomials=monomials,
        rotations=cartesian_little_rotations,
    )
    rep_characters = np.array([complex(matrix.trace().evalf()) for matrix in rep_matrices], dtype=np.complex128)

    generic_labels, display_label_map, irrep_characters_map = _resolve_spacegroup_irrep_labels(
        irreps=irreps,
        mapping_little_group=mapping_little_group,
        irt_irreps=irt_irreps,
    )
    multiplicities = _decompose_by_operations(rep_characters, irrep_characters_map)
    little_group_label = _get_little_group_label(little_primitive_rotations, little_primitive_translations)
    kpoint_label = irt_irreps[0].kpname if irt_irreps else None
    formatted_kpoint = _format_kpoint(kpoint)
    if kpoint_label:
        kpoint_line = f" {kpoint_label} {formatted_kpoint}"
    else:
        kpoint_line = f" {formatted_kpoint}"

    outputs: list[str] = []
    if show_irrep_table:
        operation_labels = [
            get_seitz_symbol(rotation, primitive_matrix)
            for rotation in little_primitive_rotations
        ]
        outputs.append(
            _format_spacegroup_irrep_table(
                little_group_label=little_group_label,
                operation_labels=operation_labels,
                generic_labels=generic_labels,
                display_label_map=display_label_map,
                irrep_characters_map=irrep_characters_map,
            ).strip("\n")
        )

    lines = [
        "",
        "* Space group *",
        f"{sg_type.international_short} ({sg_type.number})",
        "",
        "* Little group of k *",
        f"{little_group_label}",
        "",
        "* k-point (primitive) *",
        kpoint_line,
        "",
        "* Input basis functions *",
        " " + ", ".join(_format_expression(expr) for expr in seed_expressions),
        "",
        "* Closed basis-function space *",
        f" dimension: {len(basis_expressions)}",
        f" basis: {_format_expression_list(basis_expressions)}",
        "",
        "* Little-group characters *",
    ]
    operation_labels = [
        get_seitz_symbol(rotation, primitive_matrix)
        for rotation in little_primitive_rotations
    ]
    for operation_label, character in zip(operation_labels, rep_characters):
        lines.append(f"  {operation_label}: {_format_character_value(character)}")

    decomposition = " + ".join(
        _format_decomposition_term(display_label_map[key], value)
        for key, value in multiplicities.items()
        if value > 0
    )
    lines.extend([
        "",
        "* Decomposition *",
        f" {decomposition}",
        "",
        "* Irreducible representations for basis functions *",
    ])
    for irrep_name in generic_labels:
        multiplicity = multiplicities[irrep_name]
        if multiplicity <= 0:
            continue
        adapted_functions = _project_spacegroup_irrep_basis(
            basis_expressions=basis_expressions,
            rep_matrices=rep_matrices,
            irrep_characters=irrep_characters_map[irrep_name],
            irrep_name=irrep_name,
        )
        lines.append(f"  {display_label_map[irrep_name]}: {_format_expression_list(adapted_functions)}")

    outputs.append("\n".join(lines).strip("\n"))
    return "\n\n".join(outputs)


def _format_expression_list(expressions: list) -> str:
    if not expressions:
        return "[]"
    return "[" + ", ".join(_format_expression(expr) for expr in expressions) + "]"


def _format_decomposition_term(label: str, multiplicity) -> str:
    return f"{float(multiplicity):.1f} [{label}]"


def _format_expression(expr) -> str:
    text = str(expr).replace("**", "^").replace("*", "")
    return re.sub(r"(?<=\d)(?=[A-Za-z])", " ", text)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    seed_expressions = [_parse_function(raw) for raw in args.basis_function]
    if bool(args.point_group) == bool(args.space_group):
        parser.error("Specify exactly one of --point-group or --space-group.")

    if args.space_group:
        if args.kpoint is None:
            parser.error("--space-group requires --kpoint.")
        print(_analyze_space_group(args.space_group, args.kpoint, seed_expressions, args.show_irrep_table))
        return

    print(_analyze_point_group(args.point_group, seed_expressions, args.show_irrep_table))


if __name__ == "__main__":
    main()
