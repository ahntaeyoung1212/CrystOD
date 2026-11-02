#!/usr/bin/env python
"""Regenerate the regression fixtures of the VASP overlap engine.

Writes ``example/03_hybridization/vasp_overlap/<case>/`` for the four
published cases (SrTiO3, SrGeO3, CsPbI3 scalar-relativistic, CsPbI3 with
spin-orbit coupling; METAGGA = LAK on the PBEsol structures):

    overlap_cache.npz   trimmed OverlapCache: the rows of the active fragment
                        states of the default active space, all crystal bands,
                        the bessel cross-sphere mode, the fixture's k points
    POSCAR              the crystal structure
    expected.json       key numbers (crystod.crystal_orbital_overlap.key_numbers)
                        of the analysis of the FULL cache
    README              provenance and regeneration command

The analysis of the trimmed cache must reproduce expected.json to 1e-8; the
script refuses to write a fixture otherwise.  With ``--reference-dir`` the key
numbers are also checked against the paper's spectral_onsite.py JSONs.

Irrep labels.  The analysis needs one irrep label per band; they are stored in
the cache.  Choose where they come from:

    --labels-from-fixtures          reuse the labels of the fixtures already in
                                    place (default when they exist)
    --label-provider MOD:FACTORY    FACTORY(run_dirs) -> provider, with
                                    provider(run_index, k_name, k_frac) ->
                                    [label per band]; the fixtures in place
                                    were made with CrystOD's own labels,
                                    crystod.crystal_orbital_overlap:
                                    wavecar_label_provider (the irreps of the
                                    plane waves, crystod.wavecar_irreps)
    --full-cache-dir DIR            reuse full caches DIR/<case>_cache.npz that
                                    already carry labels (no WAVECAR access)

Usage:
    python script/make_overlap_fixtures.py [--cases SrTiO3 ...]
        [--vasp-root ~/vasp.6.5.1.crystod/vasp_test] [--out DIR]
        [--labels-from-fixtures | --label-provider MOD:FACTORY | --full-cache-dir DIR]
        [--reference-dir PAPER/data]

The VASP run directories are only read.  Reading the WAVECARs takes 5-12 s per
case (four k points; the SOC files are 3 GB each).
"""
from __future__ import annotations

import argparse
import datetime
import importlib
import json
import math
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from crystod import __version__  # noqa: E402
from crystod import crystal_orbital_overlap as coo  # noqa: E402
from crystod import vasp_wavecar as vw  # noqa: E402

OUT_DEFAULT = os.path.join(ROOT, "example", "03_hybridization", "vasp_overlap")
VASP_DEFAULT = os.path.expanduser("~/vasp.6.5.1.crystod/vasp_test")
RUNS = ("BAND", "BAND_sublattice1", "BAND_sublattice2")
KPOINTS = [("GM", [(0.0, 0.0, 0.0)]),
           ("X", [(0.0, 0.5, 0.0), (0.5, 0.0, 0.0), (0.0, 0.0, 0.5)]),
           ("M", [(0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)]),
           ("R", [(0.5, 0.5, 0.5)])]
# k points per fixture (size budget 350 kB per case): the SOC fixture keeps R,
# where the accidental-degeneracy label of I 5s (-R11/-R8) is resolved by dimension
CASES = {
    "SrTiO3": {"root": "SrTiO3_Pm-3m_LAK", "keep": ["GM", "X", "M", "R"],
               "reference": "SrTiO3_LAK_onsite.json",
               "title": "SrTiO3, METAGGA = LAK, scalar-relativistic"},
    "SrGeO3": {"root": "SrGeO3_Pm-3m_LAK", "keep": ["GM", "X", "M", "R"],
               "reference": "SrGeO3_LAK_onsite.json",
               "title": "SrGeO3, METAGGA = LAK, scalar-relativistic"},
    "CsPbI3": {"root": "CsPbI3_Pm-3m_LAK", "keep": ["GM", "R"],
               "reference": "CsPbI3_LAK_onsite.json",
               "title": "CsPbI3, METAGGA = LAK, scalar-relativistic"},
    "CsPbI3_SOC": {"root": "CsPbI3_Pm-3m_LAK_SOC", "keep": ["R"],
                   "reference": "CsPbI3_LAK_SOC_onsite.json",
                   "title": "CsPbI3, METAGGA = LAK with spin-orbit coupling (vasp_ncl)"},
}
COLUMNS = ["left", "right"]
TOL = 1e-8


def tree_diff(a, b, path="", tol=TOL, out=None):
    """Differences between two JSON-like trees (numbers within ``tol``)."""
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'expected' if k in a else 'result'}")
            else:
                tree_diff(a[k], b[k], f"{path}/{k}", tol, out)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            tree_diff(x, y, f"{path}[{i}]", tol, out)
    elif isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or \
            isinstance(a, str) or isinstance(b, str):
        if a != b:
            out.append(f"{path}: {a!r} != {b!r}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not (math.isfinite(a) and math.isfinite(b)) or abs(a - b) > tol:
            out.append(f"{path}: {a!r} != {b!r}")
    else:
        out.append(f"{path}: type {type(a).__name__} != {type(b).__name__}")
    return out


def rounded(o, digits=10):
    """Floats rounded to ``digits`` decimals (1e-10 eV is 100x inside the tolerance)."""
    if isinstance(o, float):
        return round(o, digits) if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: rounded(v, digits) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [rounded(v, digits) for v in o]
    return o


def pretty(o, indent=0, width=100):
    """JSON with short records on one line and long containers expanded."""
    flat = json.dumps(o, separators=(", ", ": "))
    if not isinstance(o, (dict, list)) or len(flat) + indent <= width:
        return flat
    pad = " " * (indent + 1)
    if isinstance(o, dict):
        body = ",\n".join(f"{pad}{json.dumps(k)}: {pretty(v, indent + 1, width)}"
                          for k, v in o.items())
        return "{\n" + body + "\n" + " " * indent + "}"
    body = ",\n".join(pad + pretty(v, indent + 1, width) for v in o)
    return "[\n" + body + "\n" + " " * indent + "]"


def load_provider(spec: str, run_dirs):
    module, _, factory = spec.partition(":")
    mod = importlib.import_module(module)
    return getattr(mod, factory or "factory")(run_dirs)


def labels_from(cache_path: str):
    """A provider that replays the labels stored in an existing cache."""
    old = vw.OverlapCache.load(cache_path)
    table = {kp.name: kp.labels for kp in old.kpoints}

    def provider(run_index, k_name, k_frac):
        labels = table.get(k_name)
        if not labels or run_index >= len(labels):
            return None
        return labels[run_index]
    return provider


def active_rows(results, keep):
    """``{k: [bands of fragment 1, bands of fragment 2]}`` (0-based) of the active levels."""
    rows = {}
    for kp in results["kpoints"]:
        if kp["name"] not in keep:
            continue
        per_frag = []
        for frag in kp["fragments"]:
            per_frag.append(sorted(b - 1 for L in frag["levels"] if L.get("active")
                                   for b in L["bands"]))
        rows[kp["name"]] = per_frag
    return rows


README = """{title}
{underline}

Regression fixture of the wavefunction-overlap engine of crystod --diagram
--vasp (crystod/vasp_wavecar.py, crystod/crystal_orbital_overlap.py).

Provenance
  VASP runs (read only): {root}/{{BAND,BAND_sublattice1,BAND_sublattice2}}
    BAND              the crystal ({formula}; NBANDS {nbands}, ENCUT {encut:g} eV,
                      {nk} k points of the band-structure k list)
    BAND_sublattice1  cation sublattice + point charges ({pc1}) on the anion sites
    BAND_sublattice2  anion sublattice + point charges ({pc2}) on the cation sites
  PAW datasets: {potcars}
  Irrep labels: one label per band, stored in the cache ({labels}).
  The full analysis of these runs is identical, number for number, to the
  published spectral_onsite.py v4.1 analysis (data/{reference} of the paper
  workspace), whose labels came from IrRep 2.6.3: the scalar labels agree in
  the frozen window, and IrRep's Bilbao names of double-valued irreps (-R6,
  -R8) are CrystOD's Koster-style -R6+, -R6- here.

Files
  overlap_cache.npz  OverlapCache (schema crystod-overlap-cache/1), trimmed to
                     the rows of the active fragment states ({nrows})
                     of the default active space ({shells}),
                     all crystal bands, cross-sphere mode bessel,
                     k points {kpoints}
  POSCAR             crystal structure
  expected.json      key numbers of the analysis of the FULL cache
                     (crystal_orbital_overlap.key_numbers): crystal levels and
                     irreps, ionic parents, band edges with Loewdin populations
                     and COHP, shell-pair COHPs, verdicts, ledger, N_cov

Check
  from crystod.crystal_orbital_overlap import analyse_cache, key_numbers
  res = analyse_cache("overlap_cache.npz", columns=["left", "right"], report=None)
  key_numbers(res) reproduces expected.json to 1e-8 (eV, weights).

Regenerate (crystod repository root)
  python script/make_overlap_fixtures.py --cases {case} \\
      --label-provider crystod.crystal_orbital_overlap:wavecar_label_provider \\
      --vasp-root {vasp_root}
  Generated {date} with crystod {version}.
"""


def make_case(case, args):
    spec = CASES[case]
    root = os.path.join(args.vasp_root, spec["root"])
    run_dirs = [os.path.join(root, r) for r in RUNS]
    out_dir = os.path.join(args.out, case)
    fixture_cache = os.path.join(out_dir, "overlap_cache.npz")
    label_note = "none"
    cache = None
    if args.full_cache_dir:
        path = os.path.join(args.full_cache_dir, f"{case}_cache.npz")
        if os.path.isfile(path):
            cache = vw.OverlapCache.load(path)
            label_note = f"copied from the full cache {path}"
            print(f"{case}: full cache {path}")
    if cache is None:
        provider = None
        if args.label_provider:
            provider = load_provider(args.label_provider, run_dirs)
            label_note = f"label provider {args.label_provider}"
        elif args.labels_from_fixtures and os.path.isfile(fixture_cache):
            provider = labels_from(fixture_cache)
            label_note = "replayed from the previous fixture"
        else:
            print(f"WARNING: {case}: no irrep labels (use --label-provider or "
                  "--labels-from-fixtures)")
        cache = vw.build_overlap_cache(run_dirs[0], run_dirs[1:], KPOINTS,
                                       modes=("bessel", "none"), label_provider=provider)
    if args.label_note:
        label_note = args.label_note.format(case=case, root=root)
    cache.meta["columns"] = COLUMNS
    full = coo.analyse_cache(cache, columns=COLUMNS, report=None)
    keep = [k for k in (args.keep or spec["keep"]) if any(kp.name == k for kp in cache.kpoints)]
    expected = coo.key_numbers(full, keep)
    trimmed = cache.trim(active_rows(full, keep), kpoints=keep, modes=["bessel"])
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, "overlap_cache.tmp.npz")
    trimmed.save(tmp)
    res_t = coo.analyse_cache(tmp, columns=COLUMNS, report=None)
    got = coo.key_numbers(res_t)
    diffs = tree_diff(expected, got)
    if diffs:
        os.remove(tmp)
        raise SystemExit(f"ERROR: {case}: the trimmed cache does not reproduce the key "
                         "numbers:\n  " + "\n  ".join(diffs[:20]))
    ref_note = "not checked"
    if args.reference_dir:
        ref_path = os.path.join(args.reference_dir, spec["reference"])
        if os.path.isfile(ref_path):
            with open(ref_path) as fh:
                ref = json.load(fh)
            rdiff = [d for d in tree_diff(coo.key_numbers(ref, keep), expected)
                     if not d.startswith("/active_rule")]
            ref_note = (f"identical to {spec['reference']} to {TOL:g}" if not rdiff else
                        f"{len(rdiff)} differences to {spec['reference']}")
            print(f"{case}: reference check: {ref_note}")
            for d in rdiff[:10]:
                print("   ", d)
    os.replace(tmp, fixture_cache)
    runs = cache.runs
    expected["provenance"] = {
        "generated": datetime.date.today().isoformat(), "crystod_version": __version__,
        "generator": "script/make_overlap_fixtures.py",
        "vasp_runs": [r["path"] for r in runs], "labels": label_note,
        "reference_check": ref_note, "tolerance": TOL,
        "options": {"columns": COLUMNS, "active_shells": "default rule",
                    "frozen_window": 14.0, "frozen_qmin": 0.5, "cross_sphere": "bessel"}}
    with open(os.path.join(out_dir, "expected.json"), "w") as fh:
        fh.write(pretty(rounded(coo._clean(expected))) + "\n")
    shutil.copyfile(os.path.join(run_dirs[0], "POSCAR"), os.path.join(out_dir, "POSCAR"))
    shells = coo.default_active_shells(cache)
    pcs = []
    for r in runs[1:]:
        names = sorted({sp for sp, own in zip(r["site_species"], r["own"]) if not own})
        pcs.append(", ".join(names))
    nrows = ", ".join(f"{k} {sum(len(b) for b in rows)}"
                      for k, rows in active_rows(full, keep).items())
    with open(os.path.join(out_dir, "README"), "w") as fh:
        fh.write(README.format(
            title=f"{case}: {spec['title']}", underline="=" * (len(case) + 2 + len(spec["title"])),
            root=root, formula=full["crystal"]["formula"], nbands=runs[0]["nbands"],
            encut=runs[0]["encut"], nk=runs[0]["nk"], pc1=pcs[0], pc2=pcs[1],
            potcars="; ".join(f"{d['title']} (VRHFIN {d['vrhfin'] or '-'})"
                              for d in cache.meta["paw"].values()),
            labels=label_note, reference=spec["reference"],
            nrows=nrows, shells=", ".join(shells),
            kpoints=" ".join(keep), case=case, vasp_root=args.vasp_root,
            date=datetime.date.today().isoformat(), version=__version__))
    sizes = {name: os.path.getsize(os.path.join(out_dir, name))
             for name in sorted(os.listdir(out_dir))}
    print(f"{case}: written {out_dir}: " + ", ".join(f"{n} {s / 1024:.0f} kB"
                                                    for n, s in sizes.items()) +
          f" (total {sum(sizes.values()) / 1024:.0f} kB)")
    return sum(sizes.values())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    p.add_argument("--vasp-root", default=VASP_DEFAULT)
    p.add_argument("--out", default=OUT_DEFAULT)
    p.add_argument("--label-provider", help="MODULE:FACTORY, FACTORY(run_dirs) -> provider")
    p.add_argument("--labels-from-fixtures", action="store_true",
                   help="replay the labels stored in the existing fixture caches")
    p.add_argument("--full-cache-dir", help="reuse DIR/<case>_cache.npz (full caches)")
    p.add_argument("--reference-dir", help="paper data/ directory with the reference JSONs")
    p.add_argument("--label-note", default=None,
                   help="provenance text of the labels for README/expected.json "
                        "({case} and {root} are substituted)")
    p.add_argument("--keep", nargs="+", default=None,
                   help="k points of the fixture (default: per case, see CASES)")
    args = p.parse_args(argv)
    np.set_printoptions(precision=4)
    total = 0
    for case in args.cases:
        total += make_case(case, args)
    print(f"total {total / 1024:.0f} kB")


if __name__ == "__main__":
    main()
