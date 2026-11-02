"""Cross-check the trilinear column of the HIF survey with the invariant engine.

Usage:
    python script/validate_hif_trilinear.py [--csv HIF_survey/hif_rows.csv]
        [--sample N] [--seed S] [--all]

Every row of hif_rows.csv names a parent space group (parent_number), two
order parameters (irrep1, irrep2), the polar Gamma irrep(s) induced in the
coupled isotropy subgroup (polar_irrep; several are joined with '+') and
whether the survey found a trilinear term Q1 Q2 P (trilinear).  For each
row this script builds the direct sum (irrep1, irrep2, P) for every listed
polar irrep P with crystod.isotropy_subgroup.CoupledRepresentation and asks
crystod.invariants for the number of invariants of multidegree (1, 1, 1)
(the Molien count, confirmed by the nullspace).  The engine says
"trilinear" when that count is positive for at least one P.

The survey (HIF_survey/hif_sweep.py, coupling_tests) decided the column
with its own random-point method at the order-parameter directions of the
row (dir1, dir2), so the script also reports a restricted test: whether the
(1, 1, 1) invariants have a nonzero derivative with respect to P at those
directions.  A row whose count and column disagree is printed with both
results, which tells whether the invariant exists but vanishes on the
direction (a restricted "no") or the two methods really disagree.

Reference: D. M. Hatch and H. T. Stokes, J. Appl. Cryst. 36, 951-952
(2003) (ISOTROPY INVARIANTS); N. A. Benedek and C. J. Fennie, Phys. Rev.
Lett. 106, 107204 (2011) (hybrid improper ferroelectricity).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crystod.invariants import _evaluate, invariant_space, molien_counts  # noqa: E402
from crystod.isotropy_subgroup import (  # noqa: E402
    _SCALED_PARAMETER,
    CoupledRepresentation,
    InducedRepresentation,
    IsotropyAnalyzer,
    _order_parameter_tokens,
)
from crystod.spacegroup_product import SpaceGroupIrrepAlgebra  # noqa: E402

DEFAULT_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "HIF_survey", "hif_rows.csv",
)


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


class _Parent:
    """Physically irreducible representations of one parent, by label."""

    def __init__(self, number: str):
        with _quiet():
            self.algebra = SpaceGroupIrrepAlgebra(number)
        self._by_kname: dict[str, dict] = {}

    def representations(self, kname: str) -> dict:
        """Display label -> representation for every irrep of a k point,
        listed the way crystod-group --parent SG --kpoint K lists them (a
        complex pair once, under its pair label)."""
        if kname not in self._by_kname:
            parts, listed = {}, []
            for irrep in self.algebra.irreps_by_kname[kname]:
                try:
                    with _quiet():
                        rep = InducedRepresentation(self.algebra, irrep.name)
                except (SystemExit, Exception):  # noqa: BLE001 - not tabulated
                    continue
                character = np.array([np.trace(m) for _, _, m in rep.elements])
                if rep.doubled and any(
                    np.allclose(character, other, atol=1e-6) for other in listed
                ):
                    continue
                listed.append(character)
                label = rep.label if rep.label not in parts else irrep.name
                parts[label] = rep
            self._by_kname[kname] = parts
        return self._by_kname[kname]

    def find(self, label: str):
        for kname in self.algebra.irreps_by_kname:
            parts = self.representations(kname)
            if label in parts:
                return parts[label]
        raise KeyError(label)


def split_labels(text: str, known: list[str]) -> list[str]:
    """Split 'GM2+GM3GM4' (labels joined by '+') against the known labels;
    labels may themselves end in '+', so the split is a search."""
    known = sorted(known, key=len, reverse=True)

    def search(position: int):
        if position == len(text):
            return []
        for label in known:
            if text.startswith(label, position):
                end = position + len(label)
                if end == len(text):
                    return [label]
                if text[end] == "+":
                    rest = search(end + 1)
                    if rest is not None:
                        return [label] + rest
        return None

    result = search(0)
    if result is None:
        raise KeyError(f"cannot split polar irreps {text!r}")
    return result


def _direction_vectors(rep1, rep2, algebra, pattern1: str, pattern2: str):
    """Vectors of the two direction patterns, resolved together so that
    different letters get different magnitudes (exact factors through
    resolve_direction; rounded decimal factors such as 0.577a are taken at
    face value)."""
    tokens1 = _order_parameter_tokens([pattern1])
    tokens2 = _order_parameter_tokens([pattern2])
    coupled = CoupledRepresentation.from_parts(algebra, [rep1, rep2])
    analyzer = IsotropyAnalyzer.from_representation(algebra, coupled)
    try:
        with _quiet():
            eta = analyzer.resolve_direction(tokens1 + tokens2)
    except SystemExit:
        symbols: dict[str, float] = {}
        values = []
        rng = np.random.default_rng(3)
        for token in tokens1 + tokens2:
            sign = -1.0 if token.startswith("-") else 1.0
            body = token.lstrip("-")
            if body in ("0", ""):
                values.append(0.0)
                continue
            factor, name = 1.0, body
            scaled = _SCALED_PARAMETER.fullmatch(body)
            if scaled is not None:
                factor, name = float(scaled.group(1)), scaled.group(2)
            symbols.setdefault(name, float(rng.uniform(0.5, 1.5)))
            values.append(sign * factor * symbols[name])
        eta = np.array(values)
    members = [
        (i, t) for i, t, m in analyzer.elements if np.allclose(m @ eta, eta, atol=1e-6)
    ]
    with _quiet():
        info, *_ = analyzer.subgroup_of(members)
    return eta[: rep1.dimension], eta[rep1.dimension :], int(info.number)


def check_row(parents: dict, row: dict) -> dict:
    number = row["parent_number"]
    if number not in parents:
        parents.clear()  # keep memory bounded: rows are grouped by parent
        parents[number] = _Parent(number)
    parent = parents[number]
    algebra = parent.algebra
    rep1, rep2 = parent.find(row["irrep1"]), parent.find(row["irrep2"])
    gamma = parent.representations("GM")
    polar_labels = split_labels(row["polar_irrep"], list(gamma)) if row["polar_irrep"] else []
    eta1, eta2, number12 = _direction_vectors(
        rep1, rep2, algebra, row["dir1"], row["dir2"]
    )
    # the survey's direction patterns refer to the real basis of the
    # representation at the time of the survey; when the basis has changed
    # since, the pattern is another stratum and the restricted test is void
    stale = number12 != int(row["G12_number"])
    count_any, restricted_any, counts = False, False, {}
    for label in polar_labels:
        rep_p = gamma[label]
        coupled = CoupledRepresentation.from_parts(algebra, [rep1, rep2, rep_p])
        matrices = [m for _, _, m in coupled.elements]
        dims = [rep1.dimension, rep2.dimension, rep_p.dimension]
        count = molien_counts(matrices, dims, [(1, 1, 1)])[(1, 1, 1)]
        counts[label] = count
        if count == 0:
            continue
        count_any = True
        generators = [m for i, t, m in coupled.elements if not np.any(t)]
        identity = algebra._rotation_index[algebra._key(np.eye(3))]
        generators += [
            m for i, t, m in coupled.elements
            if i == identity and int(np.sum(t)) == 1 and int(np.max(t)) == 1
        ]
        exponents, space = invariant_space(matrices, dims, (1, 1, 1), generators)
        points = np.array([
            np.concatenate([eta1, eta2, unit]) for unit in np.eye(rep_p.dimension)
        ])
        values = _evaluate(points, exponents) @ space
        if np.max(np.abs(values)) > 1e-8 * max(1.0, float(np.max(np.abs(points)))):
            restricted_any = True
    survey = row["trilinear"].strip() == "True"
    return {
        "survey": survey,
        "count": count_any,
        "restricted": None if stale else restricted_any,
        "counts": counts,
        "subgroup": number12,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--csv", default=DEFAULT_CSV, help="path of hif_rows.csv")
    parser.add_argument("--sample", type=int, default=None,
                        help="check N random rows (default 400 unless --all)")
    parser.add_argument("--seed", type=int, default=1, help="random seed of --sample")
    parser.add_argument("--all", action="store_true", help="check every row")
    args = parser.parse_args(argv)

    with open(args.csv, newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = list(enumerate(rows, start=2))  # line numbers of the CSV file
    if not args.all:
        n = args.sample if args.sample is not None else 400
        indexed = random.Random(args.seed).sample(indexed, min(n, len(indexed)))
    # group by parent so every space group is set up once
    indexed.sort(key=lambda item: (int(item[1]["parent_number"]), item[0]))

    parents: dict = {}
    agree = disagree = restricted_agree = stale = errors = 0
    start = time.time()
    for line, row in indexed:
        tag = (f"line {line}: {row['parent_number']} {row['irrep1']} {row['irrep2']} "
               f"{row['dir1']} {row['dir2']} -> {row['G12_symbol']}, "
               f"P = {row['polar_irrep'] or '-'}")
        try:
            result = check_row(parents, row)
        except (SystemExit, Exception) as exc:  # noqa: BLE001 - reported per row
            errors += 1
            print(f"ERROR {tag}: {type(exc).__name__}: {exc}")
            continue
        restricted = result["restricted"]
        if restricted is None:
            stale = stale + 1
            reason = (
                "rounded decimal factors" if any(c == "." for c in row["dir1"] + row["dir2"])
                else "real basis changed since the survey"
            )
            restricted = (f"n/a (the pattern gives space group {result['subgroup']} "
                          f"now, not {row['G12_number']}: {reason})")
        elif restricted == result["survey"]:
            restricted_agree += 1
        if result["count"] == result["survey"] and (
            result["restricted"] in (None, result["survey"])
        ):
            agree += 1
            if result["restricted"] is None:
                print(f"STALE {tag}: restricted test {restricted}")
            continue
        if result["count"] == result["survey"]:
            agree += 1
        else:
            disagree += 1
        print(
            f"MISMATCH {tag}: survey trilinear={result['survey']}, "
            f"(1,1,1) counts {result['counts']}, restricted to the "
            f"directions: {restricted}"
        )
    total = agree + disagree
    print()
    print(f"rows checked: {total} (+{errors} errors) in {time.time() - start:.1f} s")
    print(f"(1,1,1) count > 0 agrees with the trilinear column: {agree}/{total}")
    print(f"restricted (1,1,1) test agrees with the trilinear column: "
          f"{restricted_agree}/{total - stale} ({stale} rows whose direction "
          "pattern no longer gives the listed subgroup)")
    return 0 if disagree == 0 and errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
