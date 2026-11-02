"""Validate crystod-group --parent against the ISOSUBGROUP tables.

Usage: python script/validate_isosubgroup.py SG_MIN SG_MAX [outfile]

For every SUBGROUP_SG{N}_{K}.txt in SUBGROUP/ with SG_MIN <= N <= SG_MAX,
every listed IR is enumerated with crystod and the multiset of
(subgroup number, size, index) is compared with the reference rows.
Enantiomorphic-partner substitutions (a representative choice within one
stratum orbit) are counted as passes and reported separately; any other
difference, a table that belongs to another irrep label included, is a FAIL.
The reference tables in SUBGROUP/ cover space groups 62-230.

Reference: ISOSUBGROUP (https://iso.byu.edu), H. T. Stokes, S. van Orden
and B. J. Campbell, J. Appl. Cryst. 49, 1849-1853 (2016).
"""

import re
import sys
import glob
import os
import traceback
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crystod.spacegroup_product import SpaceGroupIrrepAlgebra
from crystod.isotropy_subgroup import (
    InducedRepresentation,
    IsotropyAnalyzer,
    _direction_results,
)

# enantiomorphic space-group pairs: the stabilizers of one stratum orbit
# include both types (they are conjugate under the improper operations of the
# parent), so which one is listed is a representative choice
ENANTIOMORPH = {
    76: 78, 78: 76, 91: 95, 95: 91, 92: 96, 96: 92, 144: 145, 145: 144,
    151: 153, 153: 151, 152: 154, 154: 152, 169: 170, 170: 169, 171: 172,
    172: 171, 178: 179, 179: 178, 180: 181, 181: 180, 212: 213, 213: 212,
}


def normalize(counter):
    out = Counter()
    for (num, size, index), mult in counter.items():
        pair = ENANTIOMORPH.get(num)
        if pair is not None:
            num = min(num, pair)
        out[(num, size, index)] += mult
    return out


ROW = re.compile(
    r"^(?P<ir>\S+)\s+\S+\s+\((?P<dir>[^)]+)\)\s+(?P<num>\d+)\s+(?P<sym>\S+)\s+"
    r"(?P<basis>\([^ \t]+\))\s+(?P<origin>\([^ \t]+\))\s+(?P<size>\d+)\s+"
    r"(?P<index>\d+)\s*$"
)
IR_TOKEN = re.compile(r"[A-Z]+\d+[+-]?")


def parse_reference(path):
    sg = None
    rows = []
    for line in open(path):
        line = line.rstrip("\n")
        # scraper artifact: "6 Pm" autocorrected to the time "6:00 PM"
        line = re.sub(r"\b(\d+):00\s+PM\b", r"\1 Pm", line)
        if line.startswith("Parent space group:"):
            sg = int(line.split()[3])
        m = ROW.match(line)
        if m:
            rows.append(
                (
                    m.group("ir"),
                    int(m.group("num")),
                    int(m.group("size")),
                    int(m.group("index")),
                )
            )
    by_ir = {}
    for ir, num, size, index in rows:
        by_ir.setdefault(ir, Counter())[(num, size, index)] += 1
    return sg, by_ir


def crystod_label(algebra, ref_ir):
    """Map an ISOTROPY IR label to a crystod CDML label."""
    try:
        algebra.find_irrep(ref_ir)
        return ref_ir
    except SystemExit:
        pass
    tokens = IR_TOKEN.findall(ref_ir)
    if len(tokens) == 2:
        for token in tokens:
            try:
                algebra.find_irrep(token)
                return token
            except SystemExit:
                continue
    return None


def main():
    sg_min, sg_max = int(sys.argv[1]), int(sys.argv[2])
    out = open(sys.argv[3], "w") if len(sys.argv) > 3 else sys.stdout
    base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SUBGROUP"
    )
    files = sorted(glob.glob(f"{base}/SUBGROUP_SG*_*.txt"))
    algebras = {}
    n_pass = n_fail = n_skip = n_error = n_enant = 0
    for path in files:
        m = re.match(r"SUBGROUP_SG(\d+)_(\w+)\.txt$", os.path.basename(path))
        if m is None:
            continue
        sgnum, kname = int(m.group(1)), m.group(2)
        if not (sg_min <= sgnum <= sg_max):
            continue
        _, by_ir = parse_reference(path)
        if sgnum not in algebras:
            try:
                algebras[sgnum] = SpaceGroupIrrepAlgebra(str(sgnum))
            except SystemExit as exc:
                print(f"SG{sgnum}: algebra ERROR {exc}", file=out, flush=True)
                n_error += len(by_ir)
                continue
        algebra = algebras[sgnum]
        computed = {}
        for ref_ir in sorted(by_ir):
            tag = f"SG{sgnum} {kname} {ref_ir}"
            label = crystod_label(algebra, ref_ir)
            if label is None:
                computed[ref_ir] = ("SKIP", "no crystod irrep")
                continue
            try:
                rep = InducedRepresentation(algebra, label)
                analyzer = IsotropyAnalyzer.from_representation(algebra, rep)
                results = _direction_results(analyzer)
            except SystemExit as exc:
                computed[ref_ir] = ("ERROR", str(exc))
                continue
            except Exception as exc:
                computed[ref_ir] = ("ERROR", f"{type(exc).__name__} {exc}")
                continue
            computed[ref_ir] = Counter(
                (info.number, size, index) for index, _, _, info, size in results
            )
        for ref_ir, expected in sorted(by_ir.items()):
            tag = f"SG{sgnum} {kname} {ref_ir}"
            got = computed[ref_ir]
            if isinstance(got, tuple):
                kind, msg = got
                print(f"{tag}: {kind} {msg}", file=out, flush=True)
                if kind == "SKIP":
                    n_skip += 1
                else:
                    n_error += 1
                continue
            if got == expected:
                print(f"{tag}: PASS ({sum(got.values())} strata)", file=out, flush=True)
                n_pass += 1
                continue
            if normalize(got) == normalize(expected):
                print(f"{tag}: PASS-ENANTIOMORPH ({sum(got.values())} strata)",
                      file=out, flush=True)
                n_pass += 1
                n_enant += 1
                continue
            miss = expected - got
            extra = got - expected
            print(f"{tag}: FAIL missing={dict(miss)} extra={dict(extra)}",
                  file=out, flush=True)
            n_fail += 1
    print(
        f"\nTOTAL: {n_pass} pass ({n_enant} enantiomorph), "
        f"{n_fail} fail, {n_skip} skip, {n_error} error",
        file=out,
        flush=True,
    )


if __name__ == "__main__":
    main()
