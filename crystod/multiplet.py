"""Multi-electron terms of point-group configurations (crystod-group --multiplet).

Given a point group and an electron configuration over its irrep shells
(e.g. (T2g)^2 in m-3m), computes the LS-coupling terms: the allowed
many-electron states with their spin multiplicities 2S+1 and spatial irreps,
respecting the Pauli principle, e.g.

    (T2g)^2 = ^3T1g + ^1A1g + ^1Eg + ^1T2g

Theory: for n equivalent electrons in a shell spanning irrep Gamma, the
totally antisymmetric n-electron states split by permutation symmetry: the
orbital part transforms as the Schur functor S^lambda(Gamma) where lambda is
a partition of n with at most two columns, and the spin part carries the
conjugate partition [n-k, k] (at most two rows for spin-1/2), fixing the
total spin S = (n-2k)/2.  The character of the symmetrized power follows
from the Frobenius formula

    chi_{S^lambda}(g) = sum_{rho |- n} chi^lambda_{S_n}(rho) / z_rho
                        * prod_i chi_Gamma(g^{rho_i}),

with the symmetric-group characters chi^lambda_{S_n} evaluated by the
Murnaghan-Nakayama rule and the class of g^j found from the explicit
rotation matrices of the point group.  Electrons in different (inequivalent)
shells carry no mutual Pauli restriction: their terms couple by the direct
product of the spatial parts and angular-momentum addition of the spins,
S = |S1-S2|, ..., S1+S2.
"""

from __future__ import annotations

import re
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from fractions import Fraction
from math import comb, factorial

import numpy as np

from .decompose_irrep import decompose, get_character_table
from .ligand_field import ORBITAL_AZIMUTHAL_NUMBER, get_orbital_characters


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="crystod-group --multiplet",
        description=(
            "Multi-electron terms (spin multiplicities and spatial irreps) of "
            "an electron configuration over point-group irrep shells."
        ),
        formatter_class=RawDescriptionHelpFormatter,
        epilog=(
            "Examples (T2g2 = T2g^2; the ^-free form needs no quoting in zsh):\n"
            "  crystod-group --multiplet T2g2 --point-group m-3m\n"
            "  crystod-group --multiplet T2g2 Eg1 --point-group m-3m --orbital d\n"
            "  crystod-group --multiplet E2 --point-group 3m"
        ),
    )
    parser.add_argument(
        "--point-group",
        "-pg",
        dest="point_group",
        required=True,
        help="Point group label, e.g. m-3m.",
    )
    parser.add_argument(
        "--config",
        nargs="+",
        required=True,
        metavar="IRREP^N",
        help="Shell occupations, e.g. T2g^2 or the quoting-free T2g2; several "
        "shells: T2g2 Eg1 (bare IRREP means one electron).",
    )
    parser.add_argument(
        "--orbital",
        default=None,
        help="Parent atomic orbital (s/p/d/f/...); checks that every occupied "
        "shell occurs in its ligand-field splitting and prints the splitting.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Write the exact term eigenstates as an interactive HTML page "
        "(orbital box diagrams with up/down arrows and the Slater-determinant "
        "expansion coefficients); requires --orbital.",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Output HTML path for --visualize "
        "(default: Multiplet_{pg}_{config}.html).",
    )
    return parser


# ---------------------------------------------------------------------------
# symmetric-group characters (Murnaghan-Nakayama rule)
# ---------------------------------------------------------------------------


def _partitions(n: int, largest: int | None = None):
    """All partitions of n as weakly decreasing tuples."""
    if largest is None:
        largest = n
    if n == 0:
        yield ()
        return
    for first in range(min(n, largest), 0, -1):
        for rest in _partitions(n - first, first):
            yield (first,) + rest


def _z_order(rho: tuple[int, ...]) -> int:
    """Order of the centralizer of a permutation of cycle type rho."""
    z = 1
    for part in set(rho):
        count = rho.count(part)
        z *= part**count * factorial(count)
    return z


def _sn_character(lam: tuple[int, ...], rho: tuple[int, ...]) -> int:
    """Character chi^lam of S_n at cycle type rho (Murnaghan-Nakayama rule,
    border strips removed on a padded beta-set / abacus)."""
    if not rho:
        return 1 if not lam else 0
    strip, rest = rho[0], rho[1:]
    n_beads = len(lam) + strip
    beta = [
        (lam[i] if i < len(lam) else 0) + (n_beads - 1 - i) for i in range(n_beads)
    ]
    beta_set = set(beta)
    total = 0
    for position in beta:
        moved = position - strip
        if moved < 0 or moved in beta_set:
            continue
        crossed = sum(1 for other in beta if moved < other < position)
        new_beta = sorted(
            [other for other in beta if other != position] + [moved], reverse=True
        )
        new_lam = tuple(
            value
            for value in (new_beta[i] - (n_beads - 1 - i) for i in range(n_beads))
            if value > 0
        )
        total += (-1) ** crossed * _sn_character(new_lam, rest)
    return total


# ---------------------------------------------------------------------------
# point-group classes and characters of powers
# ---------------------------------------------------------------------------


class _GroupClasses:
    """Conjugacy classes of the point group with power-map support."""

    def __init__(self, character_table: dict):
        self.ct = character_table
        self.names = list(character_table["rotation_list"])
        self.matrices = [
            np.asarray(character_table["mapping_table"][name])
            for name in self.names
        ]
        self.sizes = [mats.shape[0] for mats in self.matrices]
        self.order = int(sum(self.sizes))

    def class_of(self, matrix: np.ndarray) -> int:
        for index, mats in enumerate(self.matrices):
            if any(np.allclose(matrix, m, atol=1e-8) for m in mats):
                return index
        raise SystemExit(
            "ERROR: symmetry-operation power fell outside the group "
            "(inconsistent character table)."
        )

    def power_classes(self, max_power: int) -> list[list[int]]:
        """power_classes[c][j-1] = class index of g^j for g in class c."""
        result = []
        for mats in self.matrices:
            g = np.asarray(mats[0], dtype=float)
            powers = []
            p = np.eye(3)
            for _ in range(max_power):
                p = p @ g
                powers.append(self.class_of(p))
            result.append(powers)
        return result

    def irrep_characters(self, irrep: str) -> np.ndarray:
        return np.asarray(self.ct["character_table"][irrep], dtype=float)

    def irrep_dimension(self, irrep: str) -> int:
        return int(round(self.irrep_characters(irrep)[self.names.index("E")]))


def _schur_functor_characters(
    classes: _GroupClasses,
    irrep: str,
    lam: tuple[int, ...],
    power_classes: list[list[int]],
) -> np.ndarray:
    """Characters (per class) of the Schur functor S^lam applied to the irrep."""
    n = sum(lam)
    chi = classes.irrep_characters(irrep)
    values = np.zeros(len(classes.names))
    for rho in _partitions(n):
        coefficient = Fraction(_sn_character(lam, rho), _z_order(rho))
        if coefficient == 0:
            continue
        for c in range(len(classes.names)):
            product = 1.0
            for part in rho:
                product *= chi[power_classes[c][part - 1]]
            values[c] += float(coefficient) * product
    return values


# ---------------------------------------------------------------------------
# terms of one shell and coupling of shells
# ---------------------------------------------------------------------------


def shell_terms(
    classes: _GroupClasses, irrep: str, n_electrons: int
) -> list[tuple[Fraction, str, int]]:
    """Pauli-allowed terms (S, spatial irrep, count) of (irrep)^n_electrons."""
    dim = classes.irrep_dimension(irrep)
    capacity = 2 * dim
    if not 1 <= n_electrons <= capacity:
        raise SystemExit(
            f"ERROR: shell {irrep} (dimension {dim}) holds 1 to {capacity} "
            f"electrons; got {n_electrons}."
        )
    multiplicities = classes.sizes
    power_classes = classes.power_classes(n_electrons)
    terms: list[tuple[Fraction, str, int]] = []
    for k in range(n_electrons // 2 + 1):
        rows = n_electrons - k
        if rows > dim:
            continue  # Schur functor vanishes: too many antisymmetrized rows
        lam = tuple([2] * k + [1] * (n_electrons - 2 * k))
        spin = Fraction(n_electrons - 2 * k, 2)
        characters = _schur_functor_characters(classes, irrep, lam, power_classes)
        counts = decompose(list(characters), classes.ct, multiplicities)
        for name, count in counts.items():
            if count > 0:
                terms.append((spin, name, count))
    return terms


def couple_shells(
    classes: _GroupClasses,
    terms_a: list[tuple[Fraction, str, int]],
    terms_b: list[tuple[Fraction, str, int]],
) -> list[tuple[Fraction, str, int]]:
    """Couple the term sets of two inequivalent shells (products + spin sums)."""
    multiplicities = classes.sizes
    combined: dict[tuple[Fraction, str], int] = {}
    for spin_a, irrep_a, count_a in terms_a:
        for spin_b, irrep_b, count_b in terms_b:
            characters = classes.irrep_characters(irrep_a) * classes.irrep_characters(
                irrep_b
            )
            counts = decompose(list(characters), classes.ct, multiplicities)
            spin = abs(spin_a - spin_b)
            while spin <= spin_a + spin_b:
                for name, count in counts.items():
                    if count > 0:
                        key = (spin, name)
                        combined[key] = combined.get(key, 0) + count_a * count_b * count
                spin += 1
    return [(spin, name, count) for (spin, name), count in combined.items()]


# ---------------------------------------------------------------------------
# configuration parsing and formatting
# ---------------------------------------------------------------------------


def _split_shell_token(token: str, available: list[str]) -> tuple[str, int]:
    """One shell token -> (irrep, n).  Accepted forms: IRREP^N, (IRREP)^N,
    bare IRREP (one electron), and the shell-quoting-free IRREPN (e.g. T2g2,
    since an unquoted ^ is a glob character in zsh)."""
    match = re.fullmatch(r"\(?([A-Za-z][A-Za-z0-9'\"]*?)\)?\^(\d+)", token)
    if match:
        name, count = match.group(1), int(match.group(2))
        if name not in available:
            raise SystemExit(
                f"ERROR: '{name}' is not an irrep of this point group.\n"
                f"Choose from: {', '.join(available)}"
            )
        return name, count
    bare = token.strip("()")
    if bare in available:
        return bare, 1
    splits = [
        (bare[:i], int(bare[i:]))
        for i in range(1, len(bare))
        if bare[:i] in available and bare[i:].isdigit()
    ]
    if len(splits) == 1:
        return splits[0]
    if len(splits) > 1:
        choices = " / ".join(f"{name}^{count}" for name, count in splits)
        raise SystemExit(
            f"ERROR: shell token '{token}' is ambiguous ({choices}); "
            "use the explicit IRREP^N form."
        )
    raise SystemExit(
        f"ERROR: '{token}' is not an irrep of this point group "
        "(shell tokens: IRREP^N or IRREPN, e.g. T2g^2 or T2g2).\n"
        f"Choose from: {', '.join(available)}"
    )


def parse_config(tokens: list[str], classes: _GroupClasses) -> list[tuple[str, int]]:
    """Parse shell tokens IRREP^N / (IRREP)^N / IRREPN / bare IRREP."""
    available = list(classes.ct["character_table"].keys())
    return [_split_shell_token(token, available) for token in tokens]


def _term_symbol(spin: Fraction, irrep: str) -> str:
    return f"^{int(2 * spin + 1)}{irrep}"


def format_terms(
    classes: _GroupClasses, terms: list[tuple[Fraction, str, int]]
) -> str:
    irrep_order = {name: i for i, name in enumerate(classes.ct["character_table"])}
    ordered = sorted(terms, key=lambda t: (-t[0], irrep_order[t[1]]))
    parts = []
    for spin, name, count in ordered:
        symbol = _term_symbol(spin, name)
        parts.append(symbol if count == 1 else f"{count}({symbol})")
    return " + ".join(parts)


def _config_label(shells: list[tuple[str, int]]) -> str:
    return " ".join(f"({name})^{count}" for name, count in shells)


def hund_candidates(
    classes: _GroupClasses, terms: list[tuple[Fraction, str, int]]
) -> list[tuple[Fraction, str, int]]:
    """Hund's-rule ground-state candidates: maximal 2S+1, then maximal
    orbital dimension."""
    max_spin = max(term[0] for term in terms)
    candidates = [term for term in terms if term[0] == max_spin]
    max_dim = max(classes.irrep_dimension(term[1]) for term in candidates)
    return [
        term
        for term in candidates
        if classes.irrep_dimension(term[1]) == max_dim
    ]


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.visualize and not args.orbital:
        raise SystemExit(
            "ERROR: --visualize needs --orbital (s/p/d/f) to identify the "
            "real orbitals of each shell (e.g. --orbital d for T2g/Eg)."
        )
    if args.output and not args.visualize:
        raise SystemExit("ERROR: --output is only used with --visualize.")

    character_table = get_character_table(args.point_group)
    classes = _GroupClasses(character_table)
    shells = parse_config(args.config, classes)

    print(f"\n* Point group *\n{args.point_group}\n")

    if args.orbital:
        orbital = args.orbital.strip().lower()
        if orbital not in ORBITAL_AZIMUTHAL_NUMBER:
            raise SystemExit(
                f"ERROR: orbital '{args.orbital}' is not supported. "
                f"Choose from: {', '.join(ORBITAL_AZIMUTHAL_NUMBER)}"
            )
        orbital_characters = get_orbital_characters(orbital, character_table)
        splitting = decompose(
            list(orbital_characters.values()), character_table, classes.sizes
        )
        split_irreps = [name for name, count in splitting.items() if count > 0]
        print(f"* Ligand-field splitting of the {orbital} orbital *")
        print(
            " + ".join(
                f"{count}({name})" for name, count in splitting.items() if count > 0
            )
            + "\n"
        )
        for name, _ in shells:
            if name not in split_irreps:
                raise SystemExit(
                    f"ERROR: {name} does not occur in the {orbital}-orbital "
                    f"splitting ({' + '.join(split_irreps)})."
                )

    print(f"* Configuration *\n{_config_label(shells)}\n")

    terms: list[tuple[Fraction, str, int]] | None = None
    shell_term_lists = []
    for name, count in shells:
        current = shell_terms(classes, name, count)
        shell_term_lists.append(current)
        terms = current if terms is None else couple_shells(classes, terms, current)

    total_states = 1
    for name, count in shells:
        total_states *= comb(2 * classes.irrep_dimension(name), count)
    term_states = sum(
        int(2 * spin + 1) * classes.irrep_dimension(name) * count
        for spin, name, count in terms
    )

    print("* Term Symbols *")
    print(f"{_config_label(shells)} = {format_terms(classes, terms)}\n")
    print(
        f"check: {term_states} states = "
        + " x ".join(
            f"C({2 * classes.irrep_dimension(name)},{count})" for name, count in shells
        )
        + f" = {total_states}"
    )
    if term_states != total_states:
        print("WARNING: state count mismatch - please report this case.")
    print()

    irrep_order = {name: i for i, name in enumerate(classes.ct["character_table"])}
    ordered_terms = sorted(terms, key=lambda t: (-t[0], irrep_order[t[1]]))

    if args.orbital:
        from .multiplet_energy import (
            compute_term_energies,
            format_linear,
            ground_state,
        )

        l = ORBITAL_AZIMUTHAL_NUMBER[orbital]
        params, energies, reference, reference_note = compute_term_energies(
            classes.ct, l, shells, ordered_terms
        )
        if l == 2:
            header = "Racah parameters A, B, C"
        else:
            header = f"Slater-Condon parameters {', '.join(params)} (reduced)"
        print(f"* Multiplet Energies ({header}; Coulomb part only) *")
        width = max(len(_term_symbol(s, n)) for s, n, _ in ordered_terms)
        for (spin, name, count), entry in zip(ordered_terms, energies):
            symbol = _term_symbol(spin, name).ljust(width)
            for line in entry.describe():
                print(f"{symbol}: {line}")
        if any(entry.numeric is not None for entry in energies):
            print(f"(numeric CI blocks evaluated at {reference_note})")
        print(
            "(the one-electron / crystal-field part is an additive constant "
            "within the configuration)\n"
        )

        if len(shells) == 2 and any(
            entry.multiplicity == 2 for entry in energies
        ):
            from .multiplet_energy import coupled_parent_matrices

            parent_matrices = coupled_parent_matrices(
                classes.ct, l, shells, shell_term_lists, ordered_terms
            )
            if parent_matrices:
                print("* CI matrices in the coupled-parent basis "
                      "(strong-field/Tanabe-Sugano tables) *")
                for index, (labels, diag1, diag2, offdiag) in sorted(
                    parent_matrices.items()
                ):
                    spin, name, _ = ordered_terms[index]
                    symbol = _term_symbol(spin, name)

                    def parent_label(parent):
                        s1, g1, s2, g2 = parent
                        return (
                            f"{shells[0][0]}^{shells[0][1]}"
                            f"({_term_symbol(s1, g1)}) "
                            f"{shells[1][0]}^{shells[1][1]}"
                            f"({_term_symbol(s2, g2)})"
                        )

                    print(f"{symbol}:")
                    print(f"  |1> = {parent_label(labels[0])}")
                    print(f"  |2> = {parent_label(labels[1])}")
                    print(f"  <1|H|1> = {format_linear(diag1, params)}")
                    print(f"  <2|H|2> = {format_linear(diag2, params)}")
                    print(f"  <1|H|2> = +-({format_linear(offdiag, params)})"
                          "  (sign is a basis convention)")
                print()

        winners, unconditional = ground_state(energies, reference)
        print("* Ground-state Term Symbol (within this configuration) *")
        symbols = ", ".join(
            _term_symbol(entry.spin, entry.irrep) for entry in winners
        )
        if unconditional:
            condition = (
                "(lowest for any B > 0, C > 0)"
                if l == 2
                else "(lowest for any positive Slater parameters)"
            )
        else:
            condition = f"(lowest at {reference_note})"
        print(f"{symbols}   {condition}")

        if args.visualize:
            import os
            import re as _re

            from .multiplet_visualize import (
                compute_term_states,
                write_term_state_html,
            )

            shell_info, term_states = compute_term_states(
                classes.ct, l, shells, ordered_terms, reference
            )
            config_tag = "_".join(
                f"{name}{count}" for name, count in shells
            )
            default_name = _re.sub(
                r"[^A-Za-z0-9_.-]", "", f"Multiplet_{args.point_group}_{config_tag}"
            ) + ".html"
            output_path = args.output or default_name
            write_term_state_html(
                output_path,
                args.point_group,
                _config_label(shells),
                shell_info,
                ordered_terms,
                term_states,
                energies=energies,
                ground_symbols={(e.spin, e.irrep) for e in winners},
                reference_note=reference_note,
                l=l,
            )
            print(f"\nTerm-state viewer written to {output_path}")
    else:
        candidates = hund_candidates(classes, ordered_terms)
        print("* Ground-state Term Symbol (Hund's rules) *")
        if len(candidates) == 1:
            print(_term_symbol(candidates[0][0], candidates[0][1]))
        else:
            print(
                "candidates: "
                + ", ".join(_term_symbol(s, n) for s, n, _ in candidates)
                + "  (2S+1 and orbital-dimension tie)"
            )
        print(
            "(exact Coulomb multiplet energies in Racah/Slater parameters: "
            "add --orbital s|p|d|f of the parent shell)"
        )
    print()


if __name__ == "__main__":
    main()
