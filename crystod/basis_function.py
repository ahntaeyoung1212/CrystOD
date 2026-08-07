from __future__ import annotations

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, RawDescriptionHelpFormatter, RawTextHelpFormatter
from fractions import Fraction
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

Besides the polar coordinates x, y, z, the axial-vector (pseudovector)
components Rx, Ry, Rz are supported; they transform with det(R) * R, i.e.
like rotations, angular momenta, or magnetic moments (spins).

# Command Examples:
crystod-group --basis x y z --point-group m-3m
crystod-group --basis Rx Ry Rz --point-group m-3m
crystod-group --basis x y z --space-group Pm-3m --kpoint 0 0 0
crystod-group --basis xyz --space-group Pm-3m --kpoint 0.5 0.3 0 --show-irrep-table
crystod-group --basis z^2 --point-group m-3m
crystod-group --basis "x(y^2-z^2)" --point-group m-3m
crystod-group --basis "x*Ry - y*Rx" --point-group m-3m
"""

_PARSER_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

_X, _Y, _Z = symbols("x y z")
_RX, _RY, _RZ = symbols("Rx Ry Rz")
_GENERATORS = (_X, _Y, _Z, _RX, _RY, _RZ)
_BASE_VECTOR = Matrix([_X, _Y, _Z])
_AXIAL_VECTOR = Matrix([_RX, _RY, _RZ])


def _parse_fractional_float(value: str) -> float:
    try:
        return float(Fraction(value))
    except Exception:
        return float(value)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=desc, formatter_class=MyHelpFormatter)
    parser.add_argument(
        "--basis-function",
        nargs="+",
        required=True,
        help="Polynomial basis functions in x, y, z and the axial components Rx, Ry, Rz. "
        "Quote expressions containing parentheses.",
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
        type=_parse_fractional_float,
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
    try:
        number = int(str(space_group_symbol).strip())
    except ValueError:
        number = None
    matches = []
    for hall_number in range(1, 531):
        info = get_spacegroup_type(spglib.get_spacegroup_type(hall_number))
        if number is not None:
            if info.number == number:
                matches.append(info)
            continue
        candidates = [
            info.international_short,
            info.international,
            info.international_full,
        ]
        if any(_normalize_symbol(candidate) == requested for candidate in candidates):
            matches.append(info)
    if not matches:
        raise SystemExit(
            f'ERROR: "{space_group_symbol}" is not recognized as a standard '
            "space-group symbol or number (1-230)."
        )
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
            local_dict={"x": _X, "y": _Y, "z": _Z, "Rx": _RX, "Ry": _RY, "Rz": _RZ},
            transformations=_PARSER_TRANSFORMS,
        )
    )
    if expr.free_symbols - set(_GENERATORS):
        unknown = ", ".join(sorted(symbol.name for symbol in expr.free_symbols - set(_GENERATORS)))
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
    rotation_matrix = Matrix(rotation)
    rotated_base = rotation_matrix * _BASE_VECTOR
    substitutions = {
        _X: rotated_base[0],
        _Y: rotated_base[1],
        _Z: rotated_base[2],
    }
    if expr.free_symbols & {_RX, _RY, _RZ}:
        # axial vectors (rotations / magnetic moments) pick up det(R)
        determinant = nsimplify(rotation_matrix.det())
        rotated_axial = determinant * rotation_matrix * _AXIAL_VECTOR
        substitutions[_RX] = rotated_axial[0]
        substitutions[_RY] = rotated_axial[1]
        substitutions[_RZ] = rotated_axial[2]
    substituted = expr.subs(substitutions, simultaneous=True)
    return expand(substituted)


def _expr_to_key(expr) -> str:
    poly = Poly(expand(expr), *_GENERATORS, domain="EX")
    return tuple(
        (monomial, nsimplify(coeff))
        for monomial, coeff in sorted(poly.as_dict().items())
        if coeff != 0
    )


def _monomial_order(expressions: list) -> list[tuple[int, int, int]]:
    exponents: set[tuple[int, int, int]] = set()
    for expr in expressions:
        poly = Poly(expr, *_GENERATORS, domain="EX")
        exponents.update(poly.as_dict().keys())
    return sorted(exponents, key=lambda item: (sum(item), item))


def _vectorize(expr, monomials: list[tuple[int, int, int]]) -> Matrix:
    poly = Poly(expr, *_GENERATORS, domain="EX")
    coeff_map = poly.as_dict()
    return Matrix([coeff_map.get(monomial, 0) for monomial in monomials])


def _normalize_expr(expr):
    expanded = expand(expr)
    if expanded == 0:
        return expanded
    poly = Poly(expanded, *_GENERATORS, domain="EX")
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
    requires_hexagonal_conversion = any(
        Matrix(rotation).T * Matrix(rotation) != Matrix.eye(3)
        for rotation in rotations
    )

    cartesian_rotations: list[Matrix] = []
    for rotation in rotations:
        rotation_matrix = Matrix(rotation)
        if not requires_hexagonal_conversion:
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
            term = coeff
            for generator, power in zip(_GENERATORS, monomial):
                term *= generator ** power
            expr += term
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
    irrep_characters = np.atleast_1d(np.asarray(ct["character_table"][irrep_name]))
    class_index = {class_name: index for index, class_name in enumerate(ct["rotation_list"])}
    group_order = len(rep_matrices)
    irrep_dimension = int(round(float(irrep_characters[0].real)))

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


def _sympify_character(value, tol: float = 1e-6):
    """Convert a numeric character to an exact sympy value.

    Characters coming from spgrep/the ISO-IR tables are complex floats carrying
    numerical noise (e.g. -2.0000000000000004 + 6.7e-16j). Feeding these to
    nsimplify directly yields astronomically large exact rationals, so the
    real and imaginary parts are chopped and simplified with a tolerance
    first (crystallographic characters are sums of roots of unity).
    """
    from sympy import I, Integer

    value = complex(value)
    real = 0.0 if abs(value.real) < tol else value.real
    imag = 0.0 if abs(value.imag) < tol else value.imag
    result = nsimplify(real, tolerance=tol) if real else Integer(0)
    if imag:
        result += nsimplify(imag, tolerance=tol) * I
    return result


def _project_spacegroup_irrep_basis(
    basis_expressions: list,
    rep_matrices: list[Matrix],
    irrep_characters: np.ndarray,
    irrep_name: str,
) -> list:
    group_order = len(rep_matrices)
    irrep_dimension = int(round(float(np.real_if_close(irrep_characters[0]))))

    if irrep_dimension == 1:
        rows = []
        identity = Matrix.eye(rep_matrices[0].shape[0])
        for rep_matrix, character in zip(rep_matrices, irrep_characters):
            char_value = _sympify_character(character)
            rows.extend((rep_matrix - char_value * identity).tolist())

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
    for rep_matrix, character in zip(rep_matrices, irrep_characters):
        projector += _sympify_character(np.conjugate(character)) * rep_matrix
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


_GENERIC_SITES = ((0.123, 0.456, 0.789), (0.311, 0.641, 0.157))


def _generic_conventional_lattice(number: int) -> np.ndarray:
    """Generic conventional lattice vectors compatible with the crystal
    system of a space group (rows = lattice vectors); generic parameters
    avoid accidental extra symmetry."""
    if number <= 2:  # triclinic
        parameters = (5.1, 6.3, 7.7, 89.2, 95.4, 103.7)
    elif number <= 15:  # monoclinic, unique axis b
        parameters = (5.1, 6.3, 7.7, 90.0, 97.3, 90.0)
    elif number <= 74:  # orthorhombic
        parameters = (5.1, 6.3, 7.7, 90.0, 90.0, 90.0)
    elif number <= 142:  # tetragonal
        parameters = (5.1, 5.1, 7.7, 90.0, 90.0, 90.0)
    elif number <= 194:  # trigonal (hexagonal axes) / hexagonal
        parameters = (5.1, 5.1, 7.7, 90.0, 90.0, 120.0)
    else:  # cubic
        parameters = (5.1, 5.1, 5.1, 90.0, 90.0, 90.0)
    a, b, c, alpha, beta, gamma = parameters
    alpha, beta, gamma = np.radians((alpha, beta, gamma))
    bx, by = b * np.cos(gamma), b * np.sin(gamma)
    cx = c * np.cos(beta)
    cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    cz = np.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    return np.array([[a, 0.0, 0.0], [bx, by, 0.0], [cx, cy, cz]])


def _synthetic_conventional_cell(
    sg_type,
    conventional_rotations: np.ndarray,
    conventional_translations: np.ndarray,
):
    """Generic two-orbit structure carrying exactly the space-group symmetry,
    in the conventional setting of the ISO-IR table operations.

    --basis works from a space-group symbol without a structure file; this
    synthetic cell lets the ISO-IR labeler determine the transformation into
    the ISOTROPY standard setting with spglib, exactly as the structure-based
    commands do."""
    from .isoir import _CENTERING_TRANSLATIONS

    centerings = [np.zeros(3)] + [
        np.array(vector)
        for vector in _CENTERING_TRANSLATIONS.get(
            sg_type.international_short[0], []
        )
    ]
    lattice = _generic_conventional_lattice(sg_type.number)
    positions: list[np.ndarray] = []
    numbers: list[int] = []
    for species, seed in enumerate(_GENERIC_SITES, start=1):
        seen: set = set()
        for rotation, translation in zip(
            conventional_rotations, conventional_translations
        ):
            for centering in centerings:
                site = (
                    np.asarray(rotation, dtype=float) @ np.asarray(seed)
                    + np.asarray(translation, dtype=float)
                    + centering
                ) % 1.0
                key = tuple(np.round(site, 8) % 1.0)
                if key in seen:
                    continue
                seen.add(key)
                positions.append(site)
                numbers.append(species)
    return lattice, np.array(positions), np.array(numbers)


def _resolve_isoir_labels(
    sg_type,
    conventional_rotations: np.ndarray,
    primitive_matrix: np.ndarray,
    kpoint: list[float],
    irreps,
    mapping_little_group: np.ndarray,
    little_primitive_translations: np.ndarray,
    conventional_translations: np.ndarray,
):
    """ISO-IR (Miller-Love) labels for the spgrep irreps at a k point absent
    from the tabulated ISO-IR special points.  Returns ({irrep index: label},
    k-type letter) or None; never raises."""
    try:
        from .isoir import get_isoir_label_map

        cell = _synthetic_conventional_cell(
            sg_type, conventional_rotations, conventional_translations
        )
        primitive_matrix_inv = np.linalg.inv(primitive_matrix)
        conventional_k = np.array(kpoint, dtype=float) @ primitive_matrix_inv
        little_conventional_rotations = np.rint(
            conventional_rotations[mapping_little_group]
        ).astype(int)
        # invert the conventional -> primitive translation conversion used in
        # _analyze_space_group (column convention, t_conv = M t_prim), so each
        # conventional representative denotes exactly the operation whose
        # characters spgrep computed (they may differ from the table entries
        # by centring-lattice translations, which the labeler phase-corrects)
        little_conventional_translations = (
            primitive_matrix @ np.asarray(little_primitive_translations).T
        ).T
        spgrep_characters = [
            np.array(get_character(irrep), dtype=np.complex128)
            for irrep in irreps
        ]
        return get_isoir_label_map(
            sg_type.number,
            cell,
            1e-5,
            conventional_k,
            list(little_conventional_rotations),
            list(little_conventional_translations),
            spgrep_characters,
        )
    except Exception:
        return None


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


def _spacegroup_irrep_context(space_group_symbol: str, kpoint: list[float]):
    """Space-group little-group irreps at k with resolved labels.

    Shared by the --basis/--generate-basis analysis and the --table display:
    builds the group from the ISO-IR table operations, computes the spgrep
    small irreps at k, resolves the labels (ISO-IR special-point tables at
    tabulated k, ISO-IR labeler fallback otherwise), and prepares the
    display strings.
    """
    from types import SimpleNamespace

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
    # column convention throughout (x_conv = M x_prim), matching the rotation
    # conversion above; the former row form (t @ Minv) silently broke group
    # closure for the non-symmetric R centring matrix
    primitive_translations = np.mod(
        (primitive_matrix_inv @ conventional_translations.T).T, 1.0
    )
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

    generic_labels, display_label_map, irrep_characters_map = _resolve_spacegroup_irrep_labels(
        irreps=irreps,
        mapping_little_group=mapping_little_group,
        irt_irreps=irt_irreps,
    )
    isoir_kpoint_name = None
    if not irt_irreps:
        # k point absent from the tabulated ISO-IR special points (symmetry
        # line/plane/general point):
        # fall back to the ISO-IR (ISOTROPY, Miller-Love) tables
        isoir_result = _resolve_isoir_labels(
            sg_type=sg_type,
            conventional_rotations=conventional_rotations,
            primitive_matrix=primitive_matrix,
            kpoint=kpoint,
            irreps=irreps,
            mapping_little_group=mapping_little_group,
            little_primitive_translations=little_primitive_translations,
            conventional_translations=conventional_translations,
        )
        if isoir_result is not None:
            isoir_label_map, isoir_kpoint_name = isoir_result
            for index, generic_label in enumerate(generic_labels):
                if index in isoir_label_map:
                    display_label_map[generic_label] = (
                        f"{isoir_label_map[index]}({irreps[index].shape[1]})"
                    )
    little_group_label = _get_little_group_label(little_primitive_rotations, little_primitive_translations)
    kpoint_label = irt_irreps[0].kpname if irt_irreps else isoir_kpoint_name
    formatted_kpoint = _format_kpoint(kpoint)
    if kpoint_label:
        kpoint_line = f" {kpoint_label} {formatted_kpoint}"
    else:
        kpoint_line = f" {formatted_kpoint}"
    operation_labels = [
        get_seitz_symbol(rotation, primitive_matrix)
        for rotation in little_primitive_rotations
    ]
    return SimpleNamespace(
        sg_type=sg_type,
        irt_table=irt_table,
        primitive_matrix=primitive_matrix,
        primitive_matrix_inv=primitive_matrix_inv,
        conventional_rotations=conventional_rotations,
        conventional_translations=conventional_translations,
        irreps=irreps,
        mapping_little_group=mapping_little_group,
        irt_irreps=irt_irreps,
        little_conventional_rotations=little_conventional_rotations,
        little_primitive_rotations=little_primitive_rotations,
        little_primitive_translations=little_primitive_translations,
        generic_labels=generic_labels,
        display_label_map=display_label_map,
        irrep_characters_map=irrep_characters_map,
        little_group_label=little_group_label,
        kpoint_label=kpoint_label,
        kpoint_line=kpoint_line,
        operation_labels=operation_labels,
    )


def format_spacegroup_table(space_group_symbol: str, kpoint: list[float]) -> str:
    """Character table of the little group of k for a space group.

    The `crystod-group --table --space-group SG --kpoint ...` display: the
    space-group analogue of the point-group character table, with ISO-IR
    (ISOTROPY, Miller-Love) labels both at tabulated k points and at
    symmetry lines/planes/general points.
    """
    context = _spacegroup_irrep_context(space_group_symbol, kpoint)
    header = [
        "",
        "* Space group *",
        f"{context.sg_type.international_short} ({context.sg_type.number})",
        "",
        "* k-point (primitive) *",
        context.kpoint_line,
    ]
    table = _format_spacegroup_irrep_table(
        little_group_label=context.little_group_label,
        operation_labels=context.operation_labels,
        generic_labels=context.generic_labels,
        display_label_map=context.display_label_map,
        irrep_characters_map=context.irrep_characters_map,
    )
    return "\n".join(header) + "\n" + table


def _analyze_space_group(
    space_group_symbol: str,
    kpoint: list[float],
    seed_expressions: list,
    show_irrep_table: bool,
) -> str:
    context = _spacegroup_irrep_context(space_group_symbol, kpoint)
    sg_type = context.sg_type
    irreps = context.irreps
    little_primitive_rotations = context.little_primitive_rotations
    generic_labels = context.generic_labels
    display_label_map = context.display_label_map
    irrep_characters_map = context.irrep_characters_map
    little_group_label = context.little_group_label
    kpoint_line = context.kpoint_line
    primitive_matrix = context.primitive_matrix

    cartesian_little_rotations = _cartesianize_rotations(
        [rotation for rotation in context.little_conventional_rotations]
    )
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
    multiplicities = _decompose_by_operations(rep_characters, irrep_characters_map)

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


def _get_special_kpoints(space_group_symbol: str) -> tuple[list[str], list[list[float]]]:
    """Unique special k-points of the space group from the ISO-IR tables, in
    the primitive basis (same convention as --salc and --phonon-irrep)."""
    sg_type = _resolve_space_group_type(space_group_symbol)
    irt_table = IrrepTable(sg_type.number, spinor=False)
    primitive_matrix = np.array(
        get_primitive_matrix_by_centring(sg_type.international_short[0]),
        dtype=float,
    )
    names: list[str] = []
    kpoints: list[list[float]] = []
    for irrep in irt_table.irreps:
        k_primitive = [float(np.round(float(v), 6)) for v in np.array(irrep.k, dtype=float) @ primitive_matrix]
        if k_primitive not in kpoints:
            kpoints.append(k_primitive)
            names.append(irrep.kpname)
    return names, kpoints


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    seed_expressions = [_parse_function(raw) for raw in args.basis_function]
    if bool(args.point_group) == bool(args.space_group):
        parser.error("Specify exactly one of --point-group or --space-group.")

    if args.space_group:
        if args.kpoint is None:
            # no k-point given: analyze all special k-points of the space group
            names, kpoints = _get_special_kpoints(args.space_group)
            print("No --kpoint given; analyzing all special k points: " + ", ".join(names))
            for name, kpoint in zip(names, kpoints):
                print(_analyze_space_group(args.space_group, kpoint, seed_expressions, args.show_irrep_table))
                print()
            return
        print(_analyze_space_group(args.space_group, args.kpoint, seed_expressions, args.show_irrep_table))
        return

    print(_analyze_point_group(args.point_group, seed_expressions, args.show_irrep_table))


if __name__ == "__main__":
    main()
