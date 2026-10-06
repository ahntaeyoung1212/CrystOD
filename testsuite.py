#!/usr/bin/env python
"""
CrystOD full test suite (run inside the `crystod` conda env).

Usage:
    conda activate crystod
    cd ~/CrystOD-main
    python testsuite.py            # run everything
    python testsuite.py 10 15      # run only sections 10 and 15

Sections (grouped by command; example/<NN>_* directories share the numbers):
  -- library core --
   1. wigner_D_real regression (pure numpy)
  -- crystod (main command) --
   2. crystod (SALC)            crystal-orbital irreps
   3. crystod --atomic-orbital  hybridization analysis, and the crystal-orbital
                                diagrams of --diagram (extended Hueckel),
                                --diagram --vasp (anchor engine and WAVECAR-
                                overlap engine, offline fixtures;
                                CRYSTOD_VASP_TESTDATA=<dir of the VASP runs>
                                adds one live overlap run) and --diagram
                                --pyscf (skipped without pyscf)
   4. crystod --diagram         dipole selection rules of the crystal-orbital
                                diagrams (band-edge block, HTML pair table and
                                click handler, parity rule; extended Hueckel only)
   5. crystod --star-of-k       star of k
   6. crystod --visualize       SALC coefficients + 3D HTML viewer
   7. crystod main command      extras (aliases/errors/removed flags)
  -- crystod-group --
   8. crystod-group --product   point-group and space-group irrep direct products
   9. crystod-group --product --symmetric/--antisymmetric, --jahn-teller
                                symmetrized squares (point and space groups) and
                                Jahn-Teller active modes
  10. crystod-group --decompose reducible-representation decomposition
  11. crystod-group --ligand-field orbital splitting in a point-group field
  12. crystod-group --basis     polynomial basis classification
  13. crystod-group --tensor    property-tensor forms (Neumann's principle): Nye's
                                counts over the 32 point groups, matrix forms and
                                relations, Raman tensors (structure route == point-
                                group route)
  14. crystod-group --generate-basis automatic polynomial bases
  15. crystod-group --coset     coset decompositions
  16. crystod-group --correlate subduction of parent irreps to the Gamma point
                                of an isotropy subgroup, compatibility relations
                                along symmetry lines, point-group correlation
                                tables
  17. crystod-group --parent    isotropy subgroups of space-group irreps
  18. crystod-group --parent --irrep --invariants  invariant polynomials
                                (Molien check, Landau/Lifshitz lines, direct
                                sums and coupling terms, restricted free
                                energy, --secondary order parameters)
  19. crystod-group --parent --child  reverse lookup over the cached isotropy
                                table (cache, filters, enantiomorph note,
                                fast enumerator == legacy, --coupled pairs of
                                irreps with --secondary, 50 seeded rows of the
                                HIF survey, skipped without HIF_survey/ or
                                CRYSTOD_HIF_ROWS)
  20. crystod-group --parent --irrep --graph  group-subgroup graph of the
                                isotropy subgroups (Howard-Stokes tilt systems,
                                HTML/SVG and DOT files, dashed Landau/Lifshitz
                                lines)
  21. crystod-group --multiplet spin multiplicities of irrep-shell configurations
  22. crystod-group --poscar2cif / --cif2poscar  POSCAR <-> Bilbao-style CIF
  23. crystod-group --supergroup-cif  symmetry-mode (AMPLIMODES-style) analysis
  24. crystod-group             twelve-mode extras
  -- crystod-bz --
  25. crystod-bz                Brillouin-zone plot (seekpath auto k-path)
  26. crystod-bz --trans-mat    unit-cell + supercell Brillouin-zone plot
  27. crystod-bz                sectioned-command extras (--show-kpoint/identity/errors/removed flags)
  -- crystod-phonon --
  28. crystod-phonon --irreps   phonon irrep labeling (phonopy data)
  29. crystod-phonon --irreps / --vibration  IR/Raman/silent/acoustic activity
                                at Gamma, Raman tensors, Wyckoff orbits, dielectric
                                response from BORN (BORN read only with --nac)
  30. crystod-phonon --fatband  element-projected phonon fatbands (phonopy data)
  31. crystod-phonon --lt       longitudinal/transverse-resolved phonon band
  32. crystod-phonon --vector   phonon eigenvector VESTA export (phonopy data)
  33. crystod-phonon --modulation modulated structures (known space groups)
  34. crystod-phonon --vibration symmetry-only vibration bases
  35. crystod-phonon            seven-mode extras, incl. --subgroup (isotropy
                                subgroups of the imaginary modes)
  -- crystod-mag --
  36. crystod-mag               symmetry-adapted spin bases (cluster multipoles / SAMM)
  37. crystod-mag               --format qe / --conventional extras
  -- crystod-md --
  38. crystod-md --adp          ADPs from an MD XDATCAR trajectory
  39. crystod-md                --adp / --summary extras
  -- crystod-mol --
  40. crystod-mol               molecular point groups and molecular SALCs
  41. crystod-mol --diagram     MO diagram from symmetry + overlap (extended Hueckel)
  42. crystod-mol               --align / --show-matrix / --visualize / error extras
  -- Python API --
  43. crystod.salc/group/phonon/bz/mag/md/mol/xrd/search  library API (import
                                hygiene, backward compatibility, API-vs-CLI agreement)
  -- crystod-xrd --
  44. crystod-xrd               powder X-ray diffraction patterns (pymatgen)
  -- crystod-search --
  45. crystod-search            Materials Project search and POSCAR download (offline stub)
                                (CRYSTOD_LIVE_MP=1 adds one live search)
"""

from __future__ import annotations

import datetime
import glob
import gzip
import hashlib
import http.server
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from urllib.parse import parse_qs, urlsplit

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
POSCAR_ScF3 = os.path.join(ROOT, "example", "test_POSCARs", "221_PPOSCAR_ScF3")
POSCAR_SrTiO3 = os.path.join(ROOT, "example", "test_POSCARs", "221_PPOSCAR_SrTiO3")
POSCAR_NaCl = os.path.join(ROOT, "example", "test_POSCARs", "225_PPOSCAR_NaCl")
MODULATION_DIR = os.path.join(ROOT, "example", "33_modulation", "ScF3_Pm-3m")
MODULATION_STO_DIR = os.path.join(ROOT, "example", "33_modulation", "Sr3Ti2O7_I4mmm")
PHONON_IRREP_DIR = os.path.join(ROOT, "example", "28_phonon_irrep", "SrTiO3_Pm-3m")
PHONON_VECTOR_DIR = os.path.join(ROOT, "example", "32_phonon_vector", "Si_Fd-3m")
XDATCAR_ADP_DIR = os.path.join(ROOT, "example", "38_xdatcar2adp", "ScF3_Pm-3m_NpT_300K")
PHONON_FATBAND_DIR = os.path.join(ROOT, "example", "30_phonon_fatband", "ScF3_Pm-3m")
PHONON_LT_DIR = os.path.join(ROOT, "example", "31_phonon_lt", "ScF3_Pm-3m")
XYZ_DIR = os.path.join(ROOT, "example", "test_XYZs")
VASP_FIXTURE = os.path.join(ROOT, "example", "03_hybridization", "vasp_SrTiO3")
VASP_CAF2_FIXTURE = os.path.join(ROOT, "example", "03_hybridization",
                                 "vasp_CaF2")
VASP_LAK_FIXTURE = os.path.join(ROOT, "example", "03_hybridization",
                                "vasp_SrTiO3_LAK")
VASP_OVERLAP_FIXTURE = os.path.join(ROOT, "example", "03_hybridization",
                                    "vasp_overlap")
MP_SEARCH_DIR = os.path.join(ROOT, "example", "45_mp_search")
MP_FIXTURE = os.path.join(MP_SEARCH_DIR, "mp_summary_Sr-Ti-O.json.gz")

PASS = 0
FAIL = 0
TIMEOUT_SECONDS = 900


def report(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")
        if detail:
            for line in detail.splitlines()[-15:]:
                print(f"         | {line}")


def run_module(module: str, args: list[str], cwd: str | None = None,
               env: dict | None = None) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", module] + args,
            capture_output=True,
            text=True,
            cwd=cwd or ROOT,
            env=env,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return -1, f"TIMEOUT after {TIMEOUT_SECONDS} s"
    return proc.returncode, proc.stdout + proc.stderr


def run_python(script: str, cwd: str | None = None,
               env: dict | None = None) -> tuple[int, str]:
    """Run a Python snippet in a fresh interpreter (for the library API)."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=cwd or ROOT,
            env=env,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return -1, f"TIMEOUT after {TIMEOUT_SECONDS} s"
    return proc.returncode, proc.stdout + proc.stderr


def run_cli(args: list[str], cwd: str | None = None) -> tuple[int, str]:
    return run_module("crystod", args, cwd)


def run_bz(args: list[str], cwd: str | None = None) -> tuple[int, str]:
    return run_module("crystod.cli.bz", args, cwd)


def run_md(args: list[str], cwd: str | None = None) -> tuple[int, str]:
    return run_module("crystod.cli.md", args, cwd)


def run_mag(args: list[str], cwd: str | None = None) -> tuple[int, str]:
    return run_module("crystod.cli.mag", args, cwd)


def run_phonon(args: list[str], cwd: str | None = None) -> tuple[int, str]:
    return run_module("crystod.cli.phonon", args, cwd)


def run_group(args: list[str], cwd: str | None = None) -> tuple[int, str]:
    return run_module("crystod.cli.group", args, cwd)


def run_xrd(args: list[str], cwd: str | None = None) -> tuple[int, str]:
    return run_module("crystod.cli.xrd", args, cwd)


def run_search(args: list[str], cwd: str | None = None,
               env: dict | None = None) -> tuple[int, str]:
    return run_module("crystod.cli.search", args, cwd, env)


# ---------------------------------------------------------------- 1. wigner_D_real
def test_01_wigner_d() -> None:
    print("\n[1] wigner_D_real regression")
    from crystod.operations import wigner_D_real

    c = np.cos(2 * np.pi / 3)
    s = np.sin(2 * np.pi / 3)
    c3z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    c4z = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    inv = -np.eye(3)
    mz = np.diag([1.0, 1.0, -1.0])

    ok = all(np.allclose(wigner_D_real(1, op), op, atol=1e-12) for op in (c3z, c4z))
    report("l=1 proper: D == R", ok)

    ok = all(
        np.allclose(wigner_D_real(l, inv), (-1) ** l * np.eye(2 * l + 1), atol=1e-12)
        for l in range(4)
    )
    report("inversion parity (-1)^l", ok)

    ok = np.allclose(wigner_D_real(2, mz), np.diag([1.0, -1, 1, -1, 1]), atol=1e-12)
    report("m_z on d orbitals", ok)

    def quat_to_rot(q):
        q = q / np.linalg.norm(q)
        w, x, y, z = q
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(20):
        a = quat_to_rot(rng.normal(size=4))
        b = quat_to_rot(rng.normal(size=4))
        for l in range(4):
            err = np.abs(
                wigner_D_real(l, a @ b) - wigner_D_real(l, a) @ wigner_D_real(l, b)
            ).max()
            worst = max(worst, err)
    report(f"homomorphism D(AB)=D(A)D(B) (worst {worst:.2e})", worst < 1e-10)

    worst = 0.0
    for op in (c3z, c4z, inv, mz):
        for l in range(4):
            d = wigner_D_real(l, op)
            worst = max(worst, np.abs(d @ d.T - np.eye(2 * l + 1)).max())
    report(f"orthogonality D D^T = 1 (worst {worst:.2e})", worst < 1e-10)


def _write_cell_variant(source: str, destination: str, dim=(1, 1, 1),
                        shift=(0.0, 0.0, 0.0), by_species: bool = False) -> None:
    """Write a POSCAR of the crystal in ``source`` with every atom translated by
    ``shift`` (fractional, of the source cell) and repeated as a diagonal
    ``dim`` supercell (phonopy's atom order; with ``by_species`` the atoms are
    listed cell by cell and then sorted by species, as in a supercell built
    by hand)."""
    from phonopy.interface.calculator import read_crystal_structure
    from phonopy.interface.vasp import write_vasp
    from phonopy.structure.atoms import PhonopyAtoms
    from phonopy.structure.cells import get_supercell

    cell, _ = read_crystal_structure(source, interface_mode="vasp")
    cell.scaled_positions = (cell.scaled_positions + np.asarray(shift)) % 1.0
    if not by_species:
        write_vasp(destination, get_supercell(cell, np.diag(dim)))
        return
    n = np.asarray(dim)
    cells = [(i, j, k) for i in range(n[0]) for j in range(n[1]) for k in range(n[2])]
    positions = np.concatenate([(cell.scaled_positions + t) / n for t in cells]) % 1.0
    numbers = np.concatenate([cell.numbers for _ in cells])
    order = np.argsort(numbers, kind="stable")
    write_vasp(destination, PhonopyAtoms(cell=np.diag(n) @ cell.cell,
                                         scaled_positions=positions[order],
                                         numbers=numbers[order]))


def _salc_points(out: str) -> list[tuple[str, list[str]]]:
    """(k-point name, letters of every label printed there) of each k point of
    a crystod SALC output, the survey and the single-k form alike."""
    found = re.findall(r"k point \(primitive\):\s+(\S+) \[[^\]]*\]\n"
                       r" little group of k\s*:.*\n irreps\s*:(.*)", out)
    found += re.findall(r"\* k point \(primitive\) \* \n (\S+) \[[^\]]*\]\n"
                        r"(?:.*\n)*? \* Atomic Band Irreducible Representations \*\n(.*)", out)
    return [(name, re.findall(r"\[-?([A-Za-z]+)\d*[+-]?\(\d+\)\]", labels))
            for name, labels in found]


# ---------------------------------------------------------------- 2. crystod (SALC)
def test_02_salc() -> None:
    print("\n[2] crystod (SALC: crystal-orbital irreps)")
    code, out = run_cli(
        ["-c", POSCAR_SrTiO3, "--element", "Ti", "--orbital", "d",
         "--kpoint", "0", "0", "0"]
    )
    report("SrTiO3 Ti_d at GM exit 0", code == 0, out)
    report("Ti_d at GM: GM3+ (eg) and GM5+ (t2g)",
           "GM3+" in out and "GM5+" in out, out)

    code, out = run_cli(
        ["-c", POSCAR_ScF3, "--element", "F", "--orbital", "p",
         "--kpoint", "0", "0", "0"]
    )
    report("ScF3 F_p at GM exit 0", code == 0, out)
    report("F_p at GM: 2.0 [GM4-(3)] + 1.0 [GM5-(3)]",
           "2.0 [GM4-(3)]" in out and "1.0 [GM5-(3)]" in out, out)

    code, out = run_cli(
        ["-c", POSCAR_SrTiO3, "--element", "Ti", "--orbital", "d"]
    )
    report("all special k points mode exit 0", code == 0, out)
    report("all special k points mode lists several k points",
           out.count("k point (primitive)") >= 3, out)

    # non-special k: ISO-IR (ISOTROPY, Miller-Love) fallback labels
    code, out = run_cli(
        ["-c", POSCAR_SrTiO3, "--element", "Ti", "--orbital", "d",
         "--kpoint", "0.5", "0.5", "0.4"]
    )
    report("Ti_d at (1/2,1/2,0.4) exit 0", code == 0, out)
    report("T-line k point named via ISO-IR", "T [0.5, 0.5, 0.4]" in out, out)
    report("T-line irreps labeled T1+T3+T4+T5",
           all(f"[{lbl}(" in out for lbl in ("T1", "T3", "T4", "T5"))
           and "irrep_" not in out, out)
    report("no provenance note printed (ISO-IR unified)",
           "tabulated special point" not in out, out)

    poscar_catio3 = os.path.join(ROOT, "example", "test_POSCARs", "62_PPOSCAR_CaTiO3")
    code, out = run_cli(
        ["-c", poscar_catio3, "--element", "Ti", "--orbital", "d",
         "--kpoint", "0.5", "0.5", "0.4"]
    )
    report("Pnma Ti_d on the Q line exit 0", code == 0, out)
    report("Pnma Q-line irrep labeled 10.0 [Q1(2)]", "10.0 [Q1(2)]" in out, out)

    # a k point whose star is not tabulated, only the star of -k (P of I-4 is
    # tabulated at (1/4, 1/4, 1/4)): named with the 'A' suffix and labelled
    # with the complex conjugates of the tabulated irreps (PA1 = conjugate of
    # P1), as crystod-group --product names it; it used to print a line name
    # (Q) and irrep_N
    poscar_alpo4 = os.path.join(ROOT, "example", "test_POSCARs", "82_PPOSCAR_AlPO4")
    code, out = run_cli(["-c", poscar_alpo4, "--element", "Al", "--orbital", "p",
                         "--kpoint", "-0.25", "-0.25", "-0.25"])
    report("I-4 Al_p at -k of P: named PA, labelled PA2 + PA3 + PA4",
           code == 0 and " PA [-0.25, -0.25, -0.25]" in out
           and "1.0 [PA2(1)] + 1.0 [PA3(1)] + 1.0 [PA4(1)]" in out
           and "irrep_" not in out, out)
    code, out = run_cli(["-c", poscar_alpo4, "--element", "Al", "--orbital", "p",
                         "--kpoint", "0.25", "0.25", "0.25"])
    report("I-4 Al_p at P: the tabulated star keeps P2 + P3 + P4",
           code == 0 and " P [0.25, 0.25, 0.25]" in out
           and "1.0 [P2(1)] + 1.0 [P3(1)] + 1.0 [P4(1)]" in out, out)

    # Si (Fd-3m, two origins on the atoms, 8a and 8b): the SALC labels at L
    # and W are those of the frame crystod-phonon --irreps and --vibration
    # use (W1 + 2 W2; the other origin would give W2 + 2 W1)
    poscar_si = os.path.join(ROOT, "example", "test_POSCARs", "227_PPOSCAR_Si")
    code, out_si = run_cli(["-c", poscar_si, "--element", "Si", "--orbital", "p"])
    report("Si_p: L1+ + L2- + L3+ + L3- at L, W1 + 2 W2 at W",
           code == 0
           and "1.0 [L1+(1)] + 1.0 [L2-(1)] + 1.0 [L3+(2)] + 1.0 [L3-(2)]" in out_si
           and "1.0 [W1(2)] + 2.0 [W2(2)]" in out_si, out_si)

    # a diagonal supercell input is labelled like the cell it repeats (frame
    # rule 1 covers n1 x n2 x n3 supercells of a cell in the ISO-IR setting,
    # of any size): the labeller once took the supercell axes for the
    # conventional axes and named R of a 2x2x2 SrTiO3 cell GM3- + GM4-, M and
    # L of a 2x2x1 MoS2 cell SM and R; W1/W2 of a 2x1x1 Si cell were once
    # exchanged; a 2x2x1 AlPO4 cell once printed PA2..PA4 at P and a 1x1x2
    # CuO cell A3 at A and R1+..R4+ at R.  Non-diagonal supercells are rule 2
    # and are not asserted here (sqrt2 x sqrt2 x 1 CuO gives A3, see the
    # manual, "Limits of the frame rules")
    def salc_lines(text: str) -> list[str]:
        return re.findall(r"^ (?:k point \(primitive\)|irreps)\s*:.*$", text, re.M)

    poscar_mos2 = os.path.join(ROOT, "example", "32_phonon_vector", "MoS2_P-6m2",
                               "187_PPOSCAR_MoS2")
    poscar_cuo = os.path.join(ROOT, "example", "test_POSCARs", "131_PPOSCAR_CuO")
    reference = {poscar_si: out_si}
    with tempfile.TemporaryDirectory() as tmp:
        for source, dim, element, orbital, expected in (
                (POSCAR_SrTiO3, (2, 2, 2), "Ti", "d", ["1.0 [R3-(2)] + 1.0 [R4-(3)]"]),
                (POSCAR_SrTiO3, (2, 1, 1), "Ti", "d", ["1.0 [R3-(2)] + 1.0 [R4-(3)]"]),
                (poscar_mos2, (2, 2, 1), "Mo", "d",
                 ["2.0 [M1(1)] + 1.0 [M2(1)] + 1.0 [M3(1)] + 1.0 [M4(1)]",
                  "2.0 [L1(1)] + 1.0 [L2(1)] + 1.0 [L3(1)] + 1.0 [L4(1)]"]),
                (poscar_si, (2, 1, 1), "Si", "p", ["1.0 [W1(2)] + 2.0 [W2(2)]"]),
                (poscar_alpo4, (2, 2, 1), "Al", "p",
                 ["1.0 [P2(1)] + 1.0 [P3(1)] + 1.0 [P4(1)]"]),
                (poscar_cuo, (1, 1, 2), "Cu", "p",
                 ["1.0 [A1(2)] + 1.0 [A2(2)] + 1.0 [A4(2)]",
                  "1.0 [R1-(1)] + 2.0 [R2-(1)] + 1.0 [R3-(1)] + 2.0 [R4-(1)]"])):
            if source not in reference:
                reference[source] = run_cli(
                    ["-c", source, "--element", element, "--orbital", orbital],
                    cwd=tmp)[1]
            size = "x".join(str(n) for n in dim)
            name = f"SPOSCAR_{size}_{os.path.basename(source)}"
            _write_cell_variant(source, os.path.join(tmp, name), dim)
            code, out = run_cli(["-c", name, "--element", element, "--orbital", orbital],
                                cwd=tmp)
            report(f"{size} supercell of {os.path.basename(source)}: {element}_{orbital} "
                   f"labelled like the cell ({expected[0]})",
                   code == 0 and all(text in out for text in expected)
                   and len(salc_lines(out)) >= 8
                   and salc_lines(out) == salc_lines(reference[source]),
                   out)

    # names and labels refer to one frame: the special points are listed,
    # and a name given with --kpoint is resolved, in the frame of the labels,
    # and given coordinates are named by the labeller.  AlPO4 at a generic
    # origin gets the canonical frame, in which the P of spglib's basis is
    # the PA of the labels: it printed ' P [0.25, 0.25, 0.25]' above PA1 +
    # PA2 + PA3, and --kpoint P analyzed that point
    with tempfile.TemporaryDirectory() as tmp:
        _write_cell_variant(poscar_alpo4, os.path.join(tmp, "POSCAR_AlPO4_shifted"),
                            shift=(0.0731, 0.1593, 0.2417))
        code, out = run_cli(["-c", "POSCAR_AlPO4_shifted", "--element", "Al",
                             "--orbital", "p", "--kpoint", "0.25", "0.25", "0.25"], cwd=tmp)
        points = _salc_points(out)
        report("shifted AlPO4 at (1/4, 1/4, 1/4): the name has the letters of the "
               "labels (PA above PA1 + PA2 + PA3)",
               code == 0 and points == [("PA", ["PA", "PA", "PA"])]
               and " PA [0.25, 0.25, 0.25]" in out, out)
        code, out = run_cli(["-c", "POSCAR_AlPO4_shifted", "--element", "Al",
                             "--orbital", "p", "--kpoint", "P"], cwd=tmp)
        report("shifted AlPO4 --kpoint P: P above P1 + P2 + P3 (the -k of spglib's P)",
               code == 0 and _salc_points(out) == [("P", ["P", "P", "P"])]
               and " P [-0.25, -0.25, -0.25]" in out
               and "1.0 [P1(1)] + 1.0 [P2(1)] + 1.0 [P3(1)]" in out, out)
        # the coordinates the list prints can be given back as fractions:
        # argparse took -1/4 for an option ('expected at least one argument')
        code, out = run_cli(["-c", "POSCAR_AlPO4_shifted", "--element", "Al",
                             "--orbital", "p", "--kpoint", "-1/4", "-1/4", "-1/4"], cwd=tmp)
        report("--kpoint -1/4 -1/4 -1/4 (negative fractions): P above P1 + P2 + P3",
               code == 0 and _salc_points(out) == [("P", ["P", "P", "P"])]
               and " P [-0.25, -0.25, -0.25]" in out, out)

    # HgBr (P3): a 2x2x2 supercell with the atoms sorted by species is
    # re-based by spglib to (-b, -a, -c); it keeps the frame of the cell
    # (rule 1), and its survey now lists H and K as the cell does (it listed
    # H and K above HA1 + HA2 + HA3 and KA1 + KA2 + KA3)
    poscar_hgbr = os.path.join(ROOT, "example", "test_POSCARs", "143_PPOSCAR_HgBr")
    with tempfile.TemporaryDirectory() as tmp:
        surveys = []
        for name, dim in (("POSCAR_HgBr", (1, 1, 1)), ("SPOSCAR_HgBr_sorted", (2, 2, 2))):
            _write_cell_variant(poscar_hgbr, os.path.join(tmp, name), dim, by_species=True)
            code, out = run_cli(["-c", name, "--element", "Hg", "--orbital", "p"], cwd=tmp)
            surveys.append((code, _salc_points(out), out))
        report("HgBr cell and species-sorted 2x2x2 supercell: the same names with "
               "the same labels (H with H1 + H2 + H3)",
               surveys[0][0] == 0 and surveys[1][0] == 0 and len(surveys[0][1]) == 6
               and surveys[0][1] == surveys[1][1]
               and ("H", ["H", "H", "H"]) in surveys[1][1]
               and ("K", ["K", "K", "K"]) in surveys[1][1],
               surveys[0][2] + surveys[1][2])

    # A normalizer rotation that exchanges two tabulated special points
    # (X <-> Y of an orthorhombic crystal with a = b) needs an accidental
    # metric; none of example/test_POSCARs has one in a shifted description
    # (a sweep of the 228 structures found only -k partners, and in P1 the
    # renaming that follows the input's own axes).  P1 RbBe2F5 with its axes
    # permuted cyclically is that renaming: the input keeps its own axes
    # (rule 1), spglib reorders them, and the list follows the input
    poscar_rbbe2f5 = os.path.join(ROOT, "example", "test_POSCARs", "1_PPOSCAR_RbBe2F5")
    with tempfile.TemporaryDirectory() as tmp:
        probe = (
            "import numpy as np\n"
            "from phonopy.interface.calculator import read_crystal_structure\n"
            "from phonopy.interface.vasp import write_vasp\n"
            f"cell, _ = read_crystal_structure({poscar_rbbe2f5!r}, interface_mode='vasp')\n"
            "cell.cell = np.roll(cell.cell, 1, axis=0)\n"
            "cell.scaled_positions = np.roll(cell.scaled_positions, 1, axis=1)\n"
            "write_vasp('POSCAR_RbBe2F5_cyclic', cell)\n"
        )
        run_python(probe, cwd=tmp)
        code, out = run_cli(["-c", "POSCAR_RbBe2F5_cyclic", "--element", "Rb",
                             "--orbital", "p"], cwd=tmp)
        points = _salc_points(out)
        report("P1 with cyclically permuted axes: every listed name has the letters "
               "of its labels",
               code == 0 and len(points) == 8
               and all(set(labels) == {name} for name, labels in points), out)

    # the same over twelve structures, each at a generic origin and as a
    # 2x1x1 supercell (rule 2): at every listed point the name has the
    # letters of every label printed there (at the -k partners of P of the
    # body-centred tetragonal and cubic groups and of H and K of the trigonal
    # groups, nine of the twelve printed the tabulated name above 'A' labels)
    cases = [("82_PPOSCAR_AlPO4", "Al"), ("143_PPOSCAR_HgBr", "Hg"),
             ("121_PPOSCAR_Cu2WS4", "W"), ("145_PPOSCAR_BO3", "B"),
             ("150_PPOSCAR_Ce2O3", "Ce"), ("152_PPOSCAR_CdTe", "Cd"),
             ("157_PPOSCAR_Cu2SiS3", "Si"), ("199_PPOSCAR_UCo", "U"),
             ("217_PPOSCAR_GeF4", "Ge"), ("220_PPOSCAR_Ga", "Ga"),
             ("221_PPOSCAR_SrTiO3", "Ti"), ("62_PPOSCAR_CaTiO3", "Ti")]
    with tempfile.TemporaryDirectory() as tmp:
        for source, _ in cases:
            _write_cell_variant(os.path.join(ROOT, "example", "test_POSCARs", source),
                                os.path.join(tmp, f"POSCAR_{source}"), (2, 1, 1),
                                shift=(0.0731, 0.1593, 0.2417))
        probe = (
            "import contextlib, io\n"
            "from crystod.cli.main import main\n"
            f"for source, element in {cases!r}:\n"
            "    buffer = io.StringIO()\n"
            "    with contextlib.redirect_stdout(buffer):\n"
            "        main(['-c', 'POSCAR_' + source, '--element', element, '--orbital', 'p'])\n"
            "    print('=== ' + source)\n"
            "    print(buffer.getvalue())\n"
        )
        code, out = run_python(probe, cwd=tmp)
    blocks = out.split("=== ")[1:]
    wrong = [(block.split()[0], name, labels) for block in blocks
             for name, labels in _salc_points(block)
             if not labels or set(labels) != {name}]
    report("twelve shifted 2x1x1 supercells: the name of every listed point has "
           "the letters of its labels",
           code == 0 and len(blocks) == 12 and "irrep_" not in out
           and all(len(_salc_points(block)) >= 4 for block in blocks)
           and not wrong, f"{wrong}\n{out}")


# ---------------------------------------------------------------- 3. crystod --atomic-orbital
def test_03_hybridization() -> None:
    print("\n[3] crystod --atomic-orbital (hybridization)")
    code, out = run_cli(
        ["-c", POSCAR_SrTiO3, "--atomic-orbital", "Ti_d", "O_p",
         "--kpoint", "0", "0", "0"]
    )
    report("Ti_d O_p at GM exit 0", code == 0, out)
    report("result section present", "* Result *" in out, out)
    report("GM irreps listed", "GM" in out.split("* Result *")[-1], out)

    # hyphen is accepted as the element/orbital separator too (Ti-d == Ti_d)
    code_hyphen, out_hyphen = run_cli(
        ["-c", POSCAR_SrTiO3, "--atomic-orbital", "Ti-d", "O-p",
         "--kpoint", "0", "0", "0"]
    )
    report("Ti-d O-p (hyphen) gives byte-identical output",
           code_hyphen == code and out_hyphen == out, out_hyphen)

    code_mixed, out_mixed = run_cli(
        ["-c", POSCAR_SrTiO3, "--atomic-orbital", "Ti-d", "O_p",
         "--kpoint", "0", "0", "0"]
    )
    report("mixed separators (Ti-d O_p) also identical",
           code_mixed == code and out_mixed == out, out_mixed)

    code, out = run_cli(
        ["-c", POSCAR_SrTiO3, "--atomic-orbital", "Tid", "O_p",
         "--kpoint", "0", "0", "0"]
    )
    report("separator-less token rejected with a clear message",
           code != 0 and "invalid --atomic-orbital token 'Tid'" in out, out)

    # --diagram: crystal-orbital diagram (full-valence fragments per k point)
    with tempfile.TemporaryDirectory() as tmp:
        html_path = os.path.join(tmp, "CrystOD_ScF3.html")
        code, out = run_cli(
            ["-c", POSCAR_ScF3, "--diagram", "--co-left", "Sc",
             "--co-right", "F3", "--output", html_path]
        )
        report("--diagram ScF3 --co-left Sc --co-right F3 exit 0",
               code == 0, out)
        report("CrystOD's own paper cited at the end of the run",
               "If you use CrystOD in your research, please cite:" in out
               and "Phys. Rev. B 110, 064104 (2024)" in out, out)
        report("full-electron basis: core + valence shells, 48 electrons",
               "Sc 1s 2s 2p 3s 3p 4s 4p 3d" in out and "F 1s 2s 2p" in out
               and "electrons per cell in the diagram: 48" in out, out)
        report("ligand-field point charges guessed (Sc +3, F -1)",
               "feels the F^-1 lattice" in out
               and "feels the Sc^+3 lattice" in out, out)
        report("core levels flat at the archived PySCF atomic levels",
               re.search(r"GM1\+ #1\s+-4514\.1\d eV\s+x1\s+2e\s+"
                         r"Sc 1s GM1\+ 100\.0%", out) is not None
               and re.search(r"F 1s GM\S+ (?:100\.0|99\.\d)%", out)
               is not None
               and "atomic levels from reference/atomic_level_*" in out,
               out)
        report("fragment eg above t2g (octahedral F^-1 ligand field)",
               re.search(r"Sc 3d GM5\+ \(-8\.8\d\)", out) is not None
               and re.search(r"Sc 3d GM3\+ \(-8\.2\d\)", out) is not None,
               out)
        report("GM: F-2s/Sc-3d eg sigma bond, t2g GM5+ pure nonbonding",
               re.search(r"GM3\+ #2\s+-39\.\d+ eV\s+x2\s+4e\s+"
                         r"F 2s GM3\+ \d+\.\d%\s+Sc 3d GM3\+ \d+\.\d%", out)
               is not None
               and re.search(r"GM5\+ #\d\s+-8\.\d+ eV\s+x3\s+"
                             r"Sc 3d GM5\+ 100\.0%", out) is not None, out)
        report("R: eg (R3+) splits into bonding/antibonding, R4+ nonbonding",
               re.search(r"R3\+ #1\s+-19\.\d+ eV\s+x2\s+4e\s+"
                         r"F 2p R3\+ \d+\.\d%"
                         r"\s+Sc 3d R3\+ \d+\.\d%", out) is not None
               and re.search(r"R3\+ #2", out) is not None
               and re.search(r"R4\+ #1\s+-17\.\d+ eV\s+x3\s+6e\s+"
                             r"F 2p R4\+ 100\.0%", out) is not None, out)
        report("R: near-dependent diffuse Bloch combination kept as a "
               "first-order Loewdin estimate (~ energy)",
               "near-dependent diffuse Bloch combination(s) below overlap "
               "floor" in out
               and "first-order Loewdin estimates" in out
               and re.search(r"R1\+ #5\s+~-\d+\.\d+ eV\s+x1\s+\s*"
                             r"Sc 4s R1\+ \d+\.\d%", out) is not None, out)

        # the motivating case of the first-order restore: rocksalt AlN --
        # the dense fcc Al sublattice (12 neighbours at 2.86 A) pushes the
        # SINGLE Al 3s X1+ Bloch combination below the overlap floor; it
        # used to be deleted outright, leaving the Al column at X without
        # any 3s level and three aufbau electrons in 3p
        aln_poscar = os.path.join(tmp, "POSCAR_AlN_Fm-3m")
        with open(aln_poscar, "w") as handle:
            handle.write(
                "Al4 N4\n1.0\n"
                "4.0443663777730645 0.0 0.0\n"
                "0.0 4.0443663777730645 0.0\n"
                "0.0 0.0 4.0443663777730645\n"
                "Al N\n4 4\nDirect\n"
                "0.0 0.0 0.0\n0.5 0.5 0.0\n0.5 0.0 0.5\n0.0 0.5 0.5\n"
                "0.0 0.0 0.5\n0.5 0.0 0.0\n0.0 0.5 0.0\n0.5 0.5 0.5\n"
            )
        aln_html = os.path.join(tmp, "CrystOD_AlN.html")
        code, out = run_cli(
            ["-c", aln_poscar, "--diagram", "--co-left", "Al",
             "--co-right", "N", "--kpoint", "X", "--output", aln_html]
        )
        report("--diagram rocksalt AlN at X exit 0", code == 0, out)
        report("X: the single Al 3s X1+ Bloch combination is kept as a "
               "~ estimate (fragment) and X1+ #5 restored (crystal)",
               re.search(r"Al 3s X1\+ \(~-\d+\.\d+\)", out) is not None
               and re.search(r"X1\+ #5\s+~-\d+\.\d+ eV\s+x1\s", out)
               is not None, out)
        with open(aln_html) as handle:
            aln_variants = json.loads(re.search(
                r"const VARIANTS = (\[.*?\]);\n", handle.read(),
                re.S).group(1))
        aln_left = [lv for lv in aln_variants[0]["levels"]
                    if lv["col"] == "left"]
        aln_3s = [lv for lv in aln_left if lv["label"] == "Al 3s X1+"]
        report("X: the Al column holds all 13 electrons again (1 of them "
               "in the dashed estimated 3s level)",
               sum(lv["el"] for lv in aln_left) == 13
               and len(aln_3s) == 1 and aln_3s[0]["el"] == 1
               and aln_3s[0].get("est") == 1
               and "first-order Loewdin estimate" in aln_3s[0]["detail"],
               aln_html)
        # outermost atomic-shell columns (the MolOD "ligand-ao" analogue):
        # per-(element, shell) on-site levels + splitting connector links
        # from every fragment level to its parent shell
        aln_ao = {lv["id"]: lv for lv in aln_variants[0]["levels"]
                  if lv["col"] == "left-ao"}
        ao_3s = [i for i, lv in aln_ao.items() if lv["label"] == "Al 3s"]
        report("X: atomic-shell columns present, Al 3s fragment level "
               "linked to its parent 'Al 3s' on-site level",
               {lv["col"] for lv in aln_variants[0]["levels"]}
               >= {"left-ao", "left", "mo", "right", "right-ao"}
               and len(ao_3s) == 1
               and aln_ao[ao_3s[0]]["el"] is None
               # the near-dependent 3s combination has tiny norm, so its
               # Loewdin attribution partly spreads to the other
               # sublattice; the dominant-parent link is ~0.71
               and any(i == ao_3s[0] and w > 0.5
                       for i, w in aln_3s[0]["links"]), aln_html)
        # rutile TiO2 at GM: three artifacts of one chain, all disclosed.
        # The diffuse Ti 4p STO (zeta 0.675) engulfs the O^-2 cage, so the
        # point-charge penetration term shifts its on-site energy by
        # ~-45 eV (below Ti 3p!); the surviving near-dependent 4p doublet
        # comes back as a ~ estimate near the O 2p band top and breaks the
        # insulating filling (in THIS experimental cell the phantom lands
        # just above the O 2p top and is itself the half-filled HOMO; in
        # the slightly smaller phonopy-relaxed cell it lands inside the
        # band, swallows 4 electrons and half-fills the O 2p top instead)
        # -- and the sketch of the 89.9%-Ti-4s GM1+ level used to be
        # DRAWN as d lobes
        tio2_poscar = os.path.join(ROOT, "example", "test_POSCARs",
                                   "136_PPOSCAR_TiO2")
        tio2_html = os.path.join(tmp, "CrystOD_TiO2.html")
        code, out = run_cli(
            ["-c", tio2_poscar, "--diagram", "--co-left", "Ti2",
             "--co-right", "O4", "--kpoint", "GM", "--output", tio2_html]
        )
        report("--diagram rutile TiO2 at GM exit 0", code == 0, out)
        report("the extended-Hueckel run CAUTIONs about level ordering "
               "and points at --pyscf",
               "CAUTION: extended Hueckel gets the LEVEL ORDER wrong" in out
               and "--pyscf" in out
               and "The level ORDER can be qualitatively wrong"
               in open(tio2_html).read(), out)
        report("on-site decomposition + cage warning for the diffuse "
               "Ti 4s/4p shells",
               re.search(r"Ti 4p -5\d\.\d\d \[VSIP -5\.44, "
                         r"field -4\d\.\d\d\]", out) is not None
               and "WARNING: the Ti 4p point-charge shift (-4" in out
               and "WARNING: the Ti 4s point-charge shift (-1" in out,
               out)
        report("partial aufbau filling at GM is flagged (crystal GM5- "
               "half-filled by the phantom Ti 4p level)",
               re.search(r"WARNING: aufbau leaves the crystal column's "
                         r"GM5- level at ~?-13\.\d\d eV partially filled "
                         r"\(2 of 4 electrons\)", out) is not None,
               out)
        tio2_gm = None
        if os.path.isfile(tio2_html):
            match = re.search(r"const VARIANTS = (\[.*?\]);\n",
                              open(tio2_html).read(), re.S)
            if match:
                tio2_gm = json.loads(match.group(1))[0]
        ti4s = ([lv for lv in tio2_gm["levels"]
                 if lv["col"] == "left"
                 and lv["label"].startswith("Ti 4s GM1+")]
                if tio2_gm else [])
        lobes = (ti4s[0]["orb"][0][0][1:]
                 if ti4s and ti4s[0].get("orb") else [])
        report("the 89.9%-Ti-4s sketch is drawn as an s sphere again "
               "(Loewdin-calibrated lobes; it used to be d-dominated)",
               bool(lobes)
               and abs(lobes[0]) == max(abs(x) for x in lobes)
               and max(abs(x) for x in lobes[4:]) < 0.15
               and any(lv["col"] == "mo" and lv.get("el") == 2
                       and "PARTIALLY filled" in (lv.get("detail") or "")
                       for lv in tio2_gm["levels"]),
               tio2_html)

        report("diagram HTML written with k-point variants",
               os.path.isfile(html_path), out)
        if os.path.isfile(html_path):
            with open(html_path) as handle:
                html = handle.read()
            report("HTML has the four k-point buttons and variants",
                   html.count('class="kbtn') == 4
                   and "const VARIANTS = [" in html
                   and "R (1/2,1/2,1/2)" in html, html_path)
            report("outermost atomic-shell AO columns (Sc AOs / 3F AOs)",
                   '"col": "left-ao"' in html
                   and '"col": "right-ao"' in html
                   and '"label": "3F 2p"' in html
                   and "Sc AOs" in html and "3F AOs" in html, html_path)
            # an embedded copy (iframe in another page) has to be able to open
            # at any arm of the star, and its level panel has to stay beside
            # the diagram in a container narrower than 960 + 250 px
            report("HTML opens on the k point named in the URL (?k=R / #R)",
                   "URLSearchParams(location.search).get('k')" in html
                   and "location.hash.slice(1)" in html
                   and "if (index >= 0) applyVariant(index);" in html, html_path)
            report("level panel stays beside the shrinking diagram",
                   "#flex > #diagram { flex: 1 1 auto; min-width: 0;" in html
                   and "flex: 0 0 250px" in html
                   and "@media (max-width: 720px)" in html, html_path)
            report("levels carry supercell wave-function sketches",
                   html.count('"orb": [[') > 50
                   and html.count('"geom":') == 4
                   and "jacobi3" in html, html_path)
            report("frontier-centered view window (HOMO/LUMO midpoint ±8 eV)",
                   re.search(r'"eMin": -2[01]\.\d+', html) is not None
                   and "Show all energy levels" in html, html_path)

        # multi-shell same-l sketch filter (Sc-p = 2p+3p+4p): the shells
        # must accumulate with their STO radial amplitudes at the probe
        # radius (per-channel best of 1.5/2/2.5/3 bohr since the Loewdin
        # lobe calibration),
        # not overwrite each other (the last shell's raw coefficient used
        # to win).  The semicore Sc-3p/F-2s pair at X is the sharp probe:
        # the bonding X3- level (Sc 3p ~76%) must draw a dominant Sc p
        # lobe in phase with its +y neighbor F 2s, the antibonding X3-
        # level (F 2s ~65%) the opposite phase -- with the overwrite bug
        # the bonding Sc lobe was the inverted 4p tail (|amp| ~ 0.06) and
        # the antibonding Sc lobe vanished below the display threshold
        scp_path = os.path.join(tmp, "scp_X.html")
        code, out = run_cli(
            ["-c", POSCAR_ScF3, "--diagram", "--co-left", "Sc",
             "--co-right", "F3", "--kpoint", "X", "--output", scp_path]
        )
        report("--diagram --kpoint X (single k point) exit 0",
               code == 0, out)
        match = None
        if os.path.isfile(scp_path):
            with open(scp_path) as handle:
                match = re.search(r"const VARIANTS = (\[.*?\]);\n",
                                  handle.read(), re.S)
        phases = {}
        if match:
            for level in json.loads(match.group(1))[0]["levels"]:
                if level.get("col") != "mo" or not level.get("comp"):
                    continue
                leader = level["comp"][0][0]
                if leader not in ("Sc 3p X3-", "F 2s X3-"):
                    continue
                entries = {row[0]: row for row in level["orb"][0]}
                if 0 in entries and 1 in entries:
                    # (Sc py lobe, its product with the F 2s at +y)
                    phases[leader] = (entries[0][3],
                                      entries[0][3] * entries[1][1])
        report("Sc-p sums 2p+3p+4p: semicore X3- lobe is the 3p, not the "
               "raw 4p tail",
               len(phases) == 2 and abs(phases["Sc 3p X3-"][0]) > 0.5
               and abs(phases["F 2s X3-"][0]) > 0.5, str(phases))
        report("X3- bonding/antibonding Sc-F sketch phases",
               len(phases) == 2 and phases["Sc 3p X3-"][1] > 0
               > phases["F 2s X3-"][1], str(phases))

        # SrTiO3, no sketch filter, one k point
        code, out = run_cli(
            ["-c", POSCAR_SrTiO3, "--diagram", "--co-left", "SrTi",
             "--co-right", "O3", "--kpoint", "R",
             "--output", os.path.join(tmp, "sto_R.html")]
        )
        report("--diagram SrTiO3 --co-left SrTi --co-right O3 exit 0",
               code == 0 and "electrons per cell in the diagram: 56" in out,
               out)
        report("Sr deep core ECP-frozen, semicore Sr-4s/4p + O-1s present",
               "[ECP-28 core frozen]" in out and "Sr 3d" not in out
               and re.search(r"Sr 4s R1\+ \(-53\.\d+\)", out) is not None
               and re.search(r"O 1s R5\+ 100\.0%", out) is not None, out)
        report("R: t2g (R4-) pi and eg (R3-) sigma bonding/antibonding",
               re.search(r"R3- #1\s+-17\.\d+ eV\s+x2\s+4e\s+"
                         r"O 2p R3- \d+\.\d%"
                         r"\s+Ti 3d R3- \d+\.\d%", out) is not None
               and re.search(r"R4- #3\s+-8\.\d+ eV\s+x3\s+"
                             r"Ti 3d R4- \d+\.\d%", out) is not None, out)
        report("R: semicore Ti-3p at the PySCF Hartree-Fock level",
               re.search(r"R5\+ #3\s+-49\.\d+ eV\s+x3\s+6e\s+"
                         r"Ti 3p R5\+ \d+\.\d%", out) is not None, out)
        report("sketches always embedded, all components",
               "hover wave-function sketches" in out, out)
        with open(os.path.join(tmp, "sto_R.html")) as handle:
            html = handle.read()
        report("HTML embeds the sketches without any flag",
               '"orb": [[' in html, html_path)

        # electron-count override and explicit oxidation states
        code, out = run_cli(
            ["-c", POSCAR_ScF3, "--diagram", "--co-left", "Sc",
             "--co-right", "F3", "--electrons", "18", "--kpoint", "GM",
             "--oxidation", "Sc=+3", "F=-1",
             "--output", os.path.join(tmp, "ionic.html")]
        )
        report("--electrons 18 and explicit --oxidation accepted",
               code == 0 and "electrons per cell in the diagram: 18" in out
               and "all electrons of the neutral atoms" not in out, out)

        # --conventional: sketches drawn in the conventional cell (Fm-3m NaCl,
        # F centring; at X the conventional cell is already commensurate)
        conv_path = os.path.join(tmp, "nacl_conv.html")
        code, out = run_cli(
            ["-c", POSCAR_NaCl,
             "--diagram", "--co-left", "Na", "--co-right", "Cl",
             "--kpoint", "X", "--conventional", "--output", conv_path]
        )
        conv_html = ""
        if code == 0 and os.path.isfile(conv_path):
            with open(conv_path) as handle:
                conv_html = handle.read()
        report("--diagram --conventional exit 0, F-centring cell in sketch",
               code == 0
               and "conventional cell (F centring), 1 x 1 x 1" in conv_html
               and "drawn on the conventional cell" in conv_html, out)
        conv_match = re.search(r"const VARIANTS = (\[.*?\]);\n", conv_html,
                               re.S)
        conv_variant = (json.loads(conv_match.group(1))[0] if conv_match
                        else {"geom": {}, "levels": []})
        conv_geom = conv_variant["geom"]
        # the frame must BE the cubic conventional cell: three orthogonal
        # equal-length vectors (the default 2x1x2 fcc-primitive supercell
        # is neither, and it also passes any bare atom-count threshold
        # once the boundary replicas are counted)
        cell_rows = np.array(conv_geom.get("cell", [[0, 0, 0]])[1:], float)
        lengths = np.linalg.norm(cell_rows, axis=1) if len(cell_rows) else []
        report("--conventional frame is the cubic conventional cell",
               len(cell_rows) == 3
               and float(np.ptp(lengths)) < 1e-6
               and abs(cell_rows[0] @ cell_rows[1]) < 1e-6
               and abs(cell_rows[0] @ cell_rows[2]) < 1e-6
               and abs(cell_rows[1] @ cell_rows[2]) < 1e-6,
               str(conv_geom.get("cell")))
        # Bloch signs on the four Na sites of the F cell: primitive
        # X = (1/2,0,1/2) is (0,1,0) in conventional reciprocal
        # coordinates, so the corner and the (1/2,0,1/2) face share the
        # sign while the (1/2,1/2,0) and (0,1/2,1/2) faces are inverted
        signs = {}
        if len(cell_rows) == 3:
            origin = np.array(conv_geom["cell"][0], float)
            inverse = np.linalg.inv(cell_rows)
            conv_atoms = conv_geom.get("atoms", [])
            for level in conv_variant["levels"]:
                if (level.get("col") != "left"
                        or level["label"] != "Na 2s X1+"):
                    continue
                for row in level["orb"][0]:
                    atom = conv_atoms[row[0]]
                    if atom[0] != "Na":
                        continue
                    frac = (np.array(atom[1:4]) - origin) @ inverse
                    key = tuple(np.round(frac, 3) % 1.0)
                    signs.setdefault(key, row[1])
        report("--conventional Na 2s Bloch signs follow exp(2pi i k_conv.r)",
               len(signs) == 4
               and signs.get((0.0, 0.0, 0.0), 0)
               * signs.get((0.5, 0.0, 0.5), 0) > 0
               and signs.get((0.0, 0.0, 0.0), 0)
               * signs.get((0.5, 0.5, 0.0), 0) < 0
               and signs.get((0.0, 0.0, 0.0), 0)
               * signs.get((0.0, 0.5, 0.5), 0) < 0, str(signs))

    # errors
    code, out = run_cli(["-c", POSCAR_SrTiO3, "--diagram"])
    report("--diagram without --co-left/--co-right rejected cleanly",
           code != 0 and "requires --co-left and --co-right" in out
           and "Traceback" not in out, out)
    code, out = run_cli(
        ["-c", POSCAR_SrTiO3, "--diagram", "--atomic-orbital", "Ti-d", "O-p"]
    )
    report("--diagram with only --atomic-orbital points to --co-left/right",
           code != 0 and "requires --co-left and --co-right" in out
           and "wave-function sketch" in out, out)
    code, out = run_cli(
        ["-c", POSCAR_SrTiO3, "--diagram", "--co-left", "SrTi",
         "--co-right", "O3", "--kpoint", "0", "0", "0"]
    )
    report("--diagram with coordinate k point rejected (labels only)",
           code != 0 and "special-point label" in out
           and "Traceback" not in out, out)
    code, out = run_cli(
        ["-c", POSCAR_SrTiO3, "--diagram", "--co-left", "SrTi",
         "--co-right", "O3", "--kpoint", "Q"]
    )
    report("unknown k label rejected with the available list",
           code != 0 and "not a special point" in out and "GM" in out, out)
    code, out = run_cli(["-c", POSCAR_SrTiO3, "--conventional"])
    report("--conventional without --visualize/--diagram rejected",
           code != 0 and "--visualize or --diagram" in out, out)
    code, out = run_cli(["-c", "225_PPOSCAR_ZrO2", "--diagram",
                         "--co-left", "Zr", "--co-right", "O2"])
    report("--diagram with missing POSCAR gives clear error (no traceback)",
           code != 0 and "No POSCAR named 225_PPOSCAR_ZrO2!" in out
           and "Traceback" not in out, out)
    code, out = run_cli(
        ["-c", POSCAR_SrTiO3, "--diagram", "--co-left", "SrTi",
         "--co-right", "O2"]
    )
    report("fragment formula count mismatch rejected",
           code != 0 and "primitive cell has 3 O atom(s)" in out, out)
    code, out = run_cli(
        ["-c", POSCAR_SrTiO3, "--diagram", "--co-left", "Sr",
         "--co-right", "O3"]
    )
    report("unassigned element rejected (every atom needs a fragment)",
           code != 0 and "Ti not assigned to --co-left/--co-right" in out, out)
    code, out = run_cli(
        ["-c", POSCAR_SrTiO3, "--diagram", "--co-left", "SrTi",
         "--co-right", "TiO3"]
    )
    report("element on both sides rejected",
           code != 0 and "listed more than once" in out, out)
    code, out = run_cli(
        ["-c", POSCAR_SrTiO3, "--diagram", "--co-left", "SrTi",
         "--co-right", "O3", "--atomic-orbital", "Ti-5d", "--kpoint", "GM"]
    )
    report("--diagram rejects --atomic-orbital (sketches always embedded)",
           code != 0 and "no longer takes --atomic-orbital" in out, out)
    code, out = run_cli(
        ["-c", POSCAR_SrTiO3, "--diagram", "--co-left", "SrTi",
         "--co-right", "O3", "--oxidation", "Sr2", "--kpoint", "GM"]
    )
    report("malformed --oxidation token rejected",
           code != 0 and "invalid --oxidation token" in out, out)
    code, out = run_cli(
        ["-c", POSCAR_SrTiO3, "--diagram", "--co-left", "SrTi",
         "--co-right", "O3", "--oxidation", "Sr=+2", "Ti=+4", "O=-1",
         "--kpoint", "GM"]
    )
    report("charge-non-neutral --oxidation rejected",
           code != 0 and "not charge-neutral" in out, out)
    code, out = run_cli(
        ["-c", POSCAR_SrTiO3, "--oxidation", "Sr=+2", "--element", "O",
         "--orbital", "p"]
    )
    report("--oxidation outside --diagram rejected cleanly",
           code != 0 and "only used with --diagram" in out, out)
    code, out = run_cli(
        ["-c", POSCAR_SrTiO3, "--electrons", "18", "--element", "O",
         "--orbital", "p"]
    )
    report("--electrons outside --diagram rejected cleanly",
           code != 0 and "only used with --diagram" in out, out)
    code, out = run_cli(
        ["-c", POSCAR_SrTiO3, "--co-left", "SrTi", "--element", "O",
         "--orbital", "p"]
    )
    report("--co-left outside --diagram rejected cleanly",
           code != 0 and "only used with --diagram" in out, out)

    _test_03_vasp_engine()
    # the WAVECAR-overlap engine of --vasp (--vasp-engine overlap)
    _test_03_overlap_engine()
    _test_03_wavecar_irreps()
    _test_03_overlap_cli()

    # --------------------------------------------------- --diagram --pyscf
    # the quantitative crystal engine (crystod/crystal_orbital_pyscf.py),
    # which nothing above touches -- everything so far is extended Hueckel.
    # Deliberately small so the block stays around a minute: NaCl has two
    # atoms per cell, --onsite runs ONE crystal SCF (no fragment SCFs), and
    # --kmesh 1 1 1 / --ke-cutoff 80 / --kpoint X keep that SCF short; the
    # --conventional rerun reads the checkpoint and skips the SCF entirely.
    try:
        import pyscf  # noqa: F401
        has_pyscf = True
    except ImportError:
        has_pyscf = False
    if not has_pyscf:
        print("  [SKIP] pyscf not installed: --diagram --pyscf tests skipped.")
        return

    with tempfile.TemporaryDirectory() as tmp:
        chk_path = os.path.join(tmp, "nacl.npz")
        pyscf_path = os.path.join(tmp, "nacl_pyscf.html")
        pyscf_argv = ["-c", POSCAR_NaCl, "--diagram", "--pyscf", "--onsite",
                      "--co-left", "Na", "--co-right", "Cl", "--kpoint", "X",
                      "--kmesh", "1", "1", "1", "--ke-cutoff", "80",
                      "--chk", chk_path]
        code, out = run_cli(pyscf_argv + ["--output", pyscf_path],
                            cwd=tmp)
        report("--diagram --pyscf --onsite NaCl at X exit 0", code == 0, out)
        report("--onsite runs the crystal SCF only, converged",
               re.search(r"^\s+mo\s+crystal\s+E =\s+-6\d\.\d+ Hartree", out,
                         re.M) is not None
               and re.search(r"^\s+(left|right)\s+\S+\s+E =", out, re.M) is None
               and "16 electrons = 8 + 8" in out
               and "NOT CONVERGED" not in out, out)
        report("fragment columns are the on-site multiplets of that Fock",
               "per-shell on-site multiplets <phi|F(k)|phi>" in out
               and "no fragment SCF, no point" in out, out)
        report("AO representation verified against the PySCF overlap at X",
               re.search(r"AO representation verified against the PySCF "
                         r"overlap: max \|D\+SD - S\| = \d", out) is not None
               and "removed by Fock group-averaging" in out, out)
        report("site-symmetry induced representations of every shell",
               "Na 2p   = X3- + X5-" in out and "Cl 3d   = X1+ + X5+" in out,
               out)
        report("crystal levels: Cl-3p valence band bonds with Na 3p at X",
               re.search(r"X3- #2\s+0\.\d+ eV\s+x1\s+2e\s+"
                         r"Cl 3p X3- 6\d\.\d%\s+Na 3p X3- 2\d\.\d%", out)
               is not None
               and re.search(r"X5- #2\s+1\.\d+ eV\s+x2\s+4e\s+"
                             r"Cl 3p X5- 8\d\.\d%", out) is not None, out)
        report("Loewdin-corrected same-irrep coupling table written",
               "Loewdin-corrected |H~|" in out
               and os.path.isfile(os.path.join(tmp,
                                               "nacl_pyscf_coupling.txt")),
               out)
        report("diagram HTML written", os.path.isfile(pyscf_path), out)

        variant = {"geom": {"atoms": []}, "levels": []}
        pyscf_html = ""
        if os.path.isfile(pyscf_path):
            with open(pyscf_path) as handle:
                pyscf_html = handle.read()
            match = re.search(r"const VARIANTS = (\[.*?\]);\n", pyscf_html,
                              re.S)
            if match:
                variant = json.loads(match.group(1))[0]
        report("PySCF levels carry the supercell sketches and Loewdin rows",
               pyscf_html.count('"orb": [[') > 30
               and pyscf_html.count('"geom":') == 1
               and "Loewdin: Cl 3p" in pyscf_html
               and "left-right overlap population" in pyscf_html, pyscf_path)

        # PySCFCrystalOrbitalDiagram.sketch_partners, which only manual
        # inspection used to cover.  Two independent invariants of the
        # drawn lobes, on the Na-Cl bonds of the 2x1x2 display supercell:
        symbols = [atom[0] for atom in variant["geom"]["atoms"]]
        positions = np.array([atom[1:4] for atom in variant["geom"]["atoms"]],
                             float)
        bonds = []
        if len(positions):
            distances = [(i, j, float(np.linalg.norm(positions[j]
                                                     - positions[i])))
                         for i, left in enumerate(symbols) if left == "Na"
                         for j, right in enumerate(symbols) if right == "Cl"]
            shortest = min(d for _, _, d in distances)
            bonds = [(i, j, (positions[j] - positions[i]) / shortest)
                     for i, j, d in distances if abs(d - shortest) < 1e-3]

        def lowdin_p(detail):
            """{'Na': %, 'Cl': %} of the level's p channel, from its tooltip."""
            match = re.search(r"Loewdin: ([^\n]+)", detail)
            weights = {"Na": 0.0, "Cl": 0.0}
            for token in match.group(1).split(",") if match else []:
                element, shell, percent = token.split()
                if shell.endswith("p"):
                    weights[element] += float(percent.rstrip("%"))
            return weights

        sigma_ok, calibration_ok, bonding, antibonding = True, True, 0, 0
        notes = []
        for level in variant["levels"]:
            if level.get("col") != "mo":
                continue
            weights = lowdin_p(level["detail"])
            # both sublattices must carry real p weight for the bond to exist
            if min(weights["Na"], weights["Cl"]) < 10.0:
                continue
            partners = level.get("orb") or []
            if not partners:
                # sketch_partners drops lobes below 4% of the peak and returns
                # None once nothing survives -- a level this populated must
                # still be drawn, so an empty sketch is a failure of both
                sigma_ok = calibration_ok = False
                notes.append(f"{level['label']}: no sketch drawn")
            for partner in partners:
                lobes = {row[0]: np.array(row[1:], float) for row in partner}
                # (i) sigma overlap of the drawn p lobes over the Na-Cl bonds:
                # the two lobes facing each other along the bond have the same
                # sign on a bonding level and opposite signs on an antibonding
                # one, so this must reproduce the level's own bonding letter --
                # which the engine derives independently, from the COOP overlap
                # population 2 Re c_L+ S c_R of the AO coefficients
                sigma = sum(
                    -float(lobes[i][1:4] @ u) * float(lobes[j][1:4] @ u)
                    for i, j, u in bonds if i in lobes and j in lobes)
                # (ii) lobe SIZES are the Loewdin populations, not the raw
                # r0 amplitudes: after the per-(atom, l) calibration a
                # channel's drawn norm is sqrt(population), so the Na/Cl
                # ratio must match the tooltip's populations
                largest = {
                    element: max([float(np.linalg.norm(lobes[row][1:4]))
                                  for row in lobes
                                  if symbols[row] == element] or [0.0])
                    for element in ("Na", "Cl")}
                expected = float(np.sqrt(weights["Na"] / weights["Cl"]))
                # a p-populated sublattice with no drawn lobe at all is the
                # calibration collapsing, not a passing ratio
                ratio = (largest["Na"] / largest["Cl"] if largest["Cl"]
                         else float("inf"))
                if not abs(ratio - expected) <= 0.02:
                    calibration_ok = False
                if (sigma > 0) != (level["bond"] == "b"):
                    sigma_ok = False
                notes.append(f"{level['label']} bond={level['bond']} "
                              f"sigma={sigma:+.3f} ratio={ratio:.3f} "
                              f"sqrt(pop)={expected:.3f}")
            bonding += level["bond"] == "b"
            antibonding += level["bond"] == "a"
        report("sketch sigma phases follow the COOP bonding classification",
               bonding >= 1 and antibonding >= 1 and sigma_ok,
               "\n".join(notes))
        report("sketch lobe sizes are the per-(atom, l) Loewdin populations",
               bool(notes) and calibration_ok, "\n".join(notes))

        # every fragment level is drawn on its own sublattice, with one
        # partner per degenerate component (--onsite builds the columns from
        # single-shell on-site multiplets, so this is the sublattice
        # attribution and the partner count -- the ghost-basis filter of the
        # default three-SCF mode has no ghosts to remove here)
        stray, partner_counts_ok = set(), True
        for level in variant["levels"]:
            column = level.get("col")
            if column not in ("left", "right"):
                continue
            own = "Na" if column == "left" else "Cl"
            partners = level.get("orb") or []
            for partner in partners:
                stray |= {symbols[row[0]] for row in partner} - {own}
            if len(partners) != level["deg"]:
                partner_counts_ok = False
        report("fragment sketches stay on their own sublattice, one drawn "
               "partner per degenerate component",
               not stray and partner_counts_ok,
               f"stray elements: {sorted(stray)}, "
               f"partner counts match degeneracy: {partner_counts_ok}")

        # --conventional reuses the checkpoint, so the second run is the
        # display path only (no SCF)
        conv_path = os.path.join(tmp, "nacl_pyscf_conv.html")
        code, out = run_cli(pyscf_argv + ["--conventional",
                                          "--output", conv_path], cwd=tmp)
        conv_html = ""
        if code == 0 and os.path.isfile(conv_path):
            with open(conv_path) as handle:
                conv_html = handle.read()
        report("--diagram --pyscf --conventional exit 0 on the reused --chk",
               code == 0 and "[read from" in out
               and "conventional cell (F centring), 1 x 1 x 1" in conv_html
               and "drawn on the conventional cell" in conv_html
               and '"orb": [[' in conv_html, out)

        # default (three-SCF) mode: isolated formal-charge-ion columns --
        # one PySCF calculation per element at its formal charge (Na^+1 and
        # Cl^-1 are both closed-shell 8-electron ions here), deep-shell
        # anchored into the aligned frame, with splitting connector links
        # from every fragment level to its parent shell
        ion_path = os.path.join(tmp, "nacl_pyscf_ions.html")
        code, out = run_cli(
            ["-c", POSCAR_NaCl, "--diagram", "--pyscf",
             "--co-left", "Na", "--co-right", "Cl", "--kpoint", "X",
             "--kmesh", "2", "2", "2", "--ke-cutoff", "80", "--max-l", "2",
             "--output", ion_path], cwd=tmp)
        report("--diagram --pyscf (default mode) NaCl at X exit 0",
               code == 0, out)
        # the caution lives in the DEFAULT footer, so the PySCF pages
        # (which override foot_intro) must not inherit it
        report("the PySCF page does not inherit the extended-Hueckel "
               "level-order caution",
               os.path.isfile(pyscf_path)
               and "level ORDER can be qualitatively wrong"
               not in open(pyscf_path).read(), pyscf_path)
        report("isolated formal-charge ions reported and anchored",
               "Isolated formal-charge ions" in out
               and re.search(r"Na\^\+1 \(RKS PBE, 8 electrons\)", out)
               is not None
               and re.search(r"Cl\^-1 \(RKS PBE, 8 electrons\)", out)
               is not None
               and "anchored to its sublattice band center" in out, out)
        ion_html = ""
        if os.path.isfile(ion_path):
            with open(ion_path) as handle:
                ion_html = handle.read()
        ion_variant = None
        match = re.search(r"const VARIANTS = (\[.*?\]);\n", ion_html, re.S)
        if match:
            ion_variant = json.loads(match.group(1))[0]
        ion_ok = False
        if ion_variant:
            ao_ids = {lv["label"]: lv["id"] for lv in ion_variant["levels"]
                      if lv["col"] == "right-ao"}
            cl_3p = [lv for lv in ion_variant["levels"]
                     if lv["col"] == "right"
                     and lv["label"].startswith("Cl 3p")]
            ion_ok = ("Cl 3p" in ao_ids and bool(cl_3p)
                      and any(any(i == ao_ids["Cl 3p"] and w > 0.5
                                  for i, w in lv["links"]) for lv in cl_3p))
        report("PySCF AO columns present with splitting links "
               "(Cl 3p fragment level -> its 'Cl 3p' ion level)",
               "Na AOs" in ion_html and "Cl AOs" in ion_html
               and ion_ok, ion_path)
        # empty ion shells carry basis-honesty caveats: Na^+1 4s is a
        # finite-basis virtual (second EMPTY s state -- the occupied 2s
        # semicore does not consume the slot, so the genuine affinity
        # levels 3s/3p stay UNflagged), every empty Cl^-1 shell is a
        # discretized continuum state (anion), and the virtual note names
        # the diffuse-richer remedy (molopt-family basis in use)
        cav_det = {}
        if ion_variant:
            cav_det = {lv["label"]: lv.get("detail", "")
                       for lv in ion_variant["levels"]
                       if lv["col"] in ("left-ao", "right-ao")}
        report("empty-shell caveats (virtual/continuum) in terminal "
               "and tooltips, with the gth-qzv2p recommendation",
               "finite-basis virtuals, not physical Rydberg levels" in out
               and "note: 4s: finite-basis" in out
               and "discretized continuum" in out
               and "finite-basis VIRTUAL" in cav_det.get("Na 4s", "")
               and "--basis gth-qzv2p" in cav_det.get("Na 4s", "")
               and "finite-basis VIRTUAL" not in cav_det.get("Na 3s", "x")
               and "CONTINUUM" in cav_det.get("Cl 4s", "")
               and "--basis gth-qzv2p" in ion_html, out)

        chk_before = set(glob.glob(os.path.join(ROOT, "CHK_*.chk")))
        # --pyscf caches its three SCFs under the compound's name without
        # being asked (the runs are expensive): CHK_{formula}.chk is written,
        # reused on the next identical run, and -- unlike an explicit --chk
        # file -- simply recomputed and overwritten when the options differ
        auto_dir = os.path.join(tmp, "autochk")
        os.makedirs(auto_dir, exist_ok=True)
        shutil.copy(POSCAR_NaCl, os.path.join(auto_dir, "POSCAR"))
        auto_argv = ["--diagram", "--pyscf", "--co-left", "Na",
                     "--co-right", "Cl", "--kpoint", "X", "--kmesh",
                     "1", "1", "1", "--ke-cutoff", "80", "--max-l", "2"]
        code, out = run_cli(auto_argv, cwd=auto_dir)
        auto_chk = os.path.join(auto_dir, "CHK_NaCl.chk")
        report("--pyscf saves CHK_{formula}.chk without --chk",
               code == 0 and os.path.isfile(auto_chk)
               and "SCF saved to CHK_NaCl.chk" in out
               and "reused automatically" in out, out)
        code, out = run_cli(auto_argv, cwd=auto_dir)
        report("the automatic checkpoint is reused on the next run",
               code == 0 and "[read from CHK_NaCl.chk]" in out, out)
        code, out = run_cli(auto_argv + ["--xc", "pbesol"], cwd=auto_dir)
        report("mismatched options recompute the automatic checkpoint "
               "instead of aborting (an explicit --chk still aborts)",
               code == 0
               and "was written with different parameters (xc)" in out
               and "running the SCFs again and overwriting it" in out
               and "ERROR" not in out, out)
        # the cache lands in the run's own directory, never in the repo:
        # every --pyscf test above runs with cwd=tmp for exactly this reason.
        # Compared as a DELTA -- a developer's own manual run may have left a
        # checkpoint in the repo, which is not this suite's business
        report("the automatic checkpoint stays in the working directory "
               "(this suite writes no CHK_*.chk into the repository)",
               set(glob.glob(os.path.join(ROOT, "CHK_*.chk"))) <= chk_before,
               str(set(glob.glob(os.path.join(ROOT, "CHK_*.chk")))
                   - chk_before))

        # a same-formula polymorph must not wedge the directory: the two AlN
        # structures share CHK_AlN.chk but have 2 and 4 atoms per cell, and
        # np.allclose raises rather than returning False on that mismatch
        poly_dir = os.path.join(tmp, "polymorph")
        os.makedirs(poly_dir, exist_ok=True)
        hybrid = os.path.join(ROOT, "example", "03_hybridization")
        poly_ok = True
        for name in ("POSCAR_AlN_Fm-3m", "POSCAR_AlN_P63mc"):
            if os.path.isfile(os.path.join(hybrid, name)):
                shutil.copy(os.path.join(hybrid, name), poly_dir)
            else:
                poly_ok = False
        if poly_ok:
            code, out = run_cli(
                ["-c", "POSCAR_AlN_Fm-3m", "--diagram", "--pyscf",
                 "--co-left", "Al", "--co-right", "N", "--kpoint", "GM",
                 "--kmesh", "1", "1", "1", "--ke-cutoff", "80",
                 "--max-l", "2"], cwd=poly_dir)
            first = code == 0
            code, out = run_cli(
                ["-c", "POSCAR_AlN_P63mc", "--diagram", "--pyscf",
                 "--co-left", "Al2", "--co-right", "N2", "--kpoint", "GM",
                 "--kmesh", "1", "1", "1", "--ke-cutoff", "80",
                 "--max-l", "2"], cwd=poly_dir)
            report("a same-formula polymorph recomputes the automatic "
                   "checkpoint instead of crashing on the atom count",
                   first and code == 0
                   and "different parameters" in out
                   and "structure" in out
                   and "Traceback" not in out, out)

            # a corrupt checkpoint (interrupted write, foreign file) must
            # not wedge the directory either
            with open(os.path.join(poly_dir, "CHK_AlN.chk"), "wb") as handle:
                handle.write(b"not a checkpoint at all")
            code, out = run_cli(
                ["-c", "POSCAR_AlN_Fm-3m", "--diagram", "--pyscf",
                 "--co-left", "Al", "--co-right", "N", "--kpoint", "GM",
                 "--kmesh", "1", "1", "1", "--ke-cutoff", "80",
                 "--max-l", "2"], cwd=poly_dir)
            report("an unreadable automatic checkpoint is recomputed, not "
                   "a traceback",
                   code == 0
                   and "is not a readable CrystOD checkpoint" in out
                   and "Traceback" not in out, out)

            # --no-chk: no cache read, no cache written
            os.remove(os.path.join(poly_dir, "CHK_AlN.chk"))
            code, out = run_cli(
                ["-c", "POSCAR_AlN_Fm-3m", "--diagram", "--pyscf",
                 "--co-left", "Al", "--co-right", "N", "--kpoint", "GM",
                 "--kmesh", "1", "1", "1", "--ke-cutoff", "80",
                 "--max-l", "2", "--no-chk"], cwd=poly_dir)
            report("--no-chk skips the automatic checkpoint entirely",
                   code == 0 and "SCF saved to" not in out
                   and not os.path.isfile(
                       os.path.join(poly_dir, "CHK_AlN.chk")), out)

        # neutral-atom sublattices (--oxidation Al=0 N=0, the EHT engine's
        # convention): a fragment with an odd electron count is solved with
        # automatic Fermi smearing instead of being rejected (integer
        # aufbau would drop the unpaired electron), and the isolated-atom
        # columns pick the Hund ground spin by an energy scan over the
        # parity-consistent spins -- neutral N is the 4S spin-3 atom, not
        # the spin-1 doublet a parity fallback would give
        aln_poscar = os.path.join(tmp, "POSCAR_AlN_Fm-3m")
        with open(aln_poscar, "w") as handle:
            handle.write(
                "Al4 N4\n1.0\n"
                "4.0443663777730645 0.0 0.0\n"
                "0.0 4.0443663777730645 0.0\n"
                "0.0 0.0 4.0443663777730645\n"
                "Al N\n4 4\nDirect\n"
                "0.0 0.0 0.0\n0.5 0.5 0.0\n0.5 0.0 0.5\n0.0 0.5 0.5\n"
                "0.0 0.0 0.5\n0.5 0.0 0.0\n0.0 0.5 0.0\n0.5 0.5 0.5\n"
            )
        neutral_path = os.path.join(tmp, "aln_pyscf_neutral.html")
        code, out = run_cli(
            ["-c", aln_poscar, "--diagram", "--pyscf",
             "--co-left", "Al", "--co-right", "N",
             "--oxidation", "Al=0", "N=0", "--kpoint", "X",
             "--kmesh", "2", "2", "2", "--ke-cutoff", "80", "--max-l", "2",
             "--output", neutral_path], cwd=tmp)
        report("--diagram --pyscf neutral sublattices (rocksalt AlN, "
               "--oxidation Al=0 N=0) exit 0", code == 0, out)
        neutral_html = ""
        if os.path.isfile(neutral_path):
            with open(neutral_path) as handle:
                neutral_html = handle.read()
        neutral_fill = None
        match = re.search(r"const VARIANTS = (\[.*?\]);\n", neutral_html,
                          re.S)
        if match:
            neutral_fill = {
                column: sum(lv["el"] or 0
                            for lv in json.loads(match.group(1))[0]["levels"]
                            if lv["col"] == column)
                for column in ("left", "right", "mo")
            }
        report("odd fragments auto-smeared, Hund spin-3 N atom, and the "
               "3 + 5 = 8 electron aufbau at X",
               "odd electron count (3): occupations use Fermi smearing"
               in out
               and "odd electron count (5)" in out
               and re.search(r"Al\^\+0 \(UKS PBE, 3 electrons, spin 1",
                             out) is not None
               and re.search(r"N\^\+0 \(UKS PBE, 5 electrons, spin 3",
                             out) is not None
               and "formally NEUTRAL atom" in neutral_html
               and neutral_fill == {"left": 3, "right": 5, "mo": 8}, out)


def _vasp_kpoint_block(text: str, name: str) -> str:
    """The report block of one k point of a ``--diagram --vasp`` run."""
    match = re.search(rf"\n \* k point {re.escape(name)} .*?"
                      r"(?=\n \* k point |\nCrystal-orbital diagram)", text, re.S)
    return match.group(0) if match else ""


def _vasp_salc(block: str, element: str, shell: str) -> list:
    """The tabulated SALC irreps of one ``(element, shell)``, as a sorted list.

    ``"2GM4- + GM5-"`` becomes ``["GM4-", "GM4-", "GM5-"]``.
    """
    match = re.search(rf"^\s+{element} {shell}\s+= (.+)$", block, re.M)
    if not match:
        return []
    irreps = []
    for token in match.group(1).split(" + "):
        count = re.match(r"(\d*)(\S+)", token.strip())
        irreps.extend([count.group(2)] * int(count.group(1) or 1))
    return sorted(irreps)


def _vasp_found(block: str, formula: str, element: str, shell: str) -> list:
    """The irreps a fragment column actually carries in one manifold."""
    match = re.search(rf"^   {formula}\s+: (.+)$", block, re.M)
    if not match:
        return []
    irreps = []
    for entry in match.group(1).split(", "):
        label = entry.split(" (")[0].strip()
        if not label.startswith(f"{element} {shell} "):
            continue
        irreps.append(label.split()[-1].split("#")[0])
    return sorted(irreps)


def _vasp_levels(path: str) -> dict:
    """``{(k, column, level): (irrep, E_raw, E_align, shift, w_proj)}``.

    The level table of ``--diagram --vasp``, keyed so that two runs of the
    same data in two SETTINGS can be compared level by level: everything but
    the irrep label has to agree to the last printed digit.
    """
    levels = {}
    kpoint = None
    for line in open(path):
        if line.startswith("# k point"):
            kpoint = line.split()[3]
            continue
        if line.startswith("#") or line.startswith(" ") or not line.strip():
            continue
        fields = line.split()
        if len(fields) < 12 or kpoint is None:
            continue
        try:
            numbers = [float(fields[index]) for index in (5, 6, 7, 8, 10)]
        except ValueError:
            continue
        levels[(kpoint, fields[1], fields[2])] = (fields[3], *numbers)
    return levels


def _reorder_vasp_run(source: str, destination: str, order: list) -> None:
    """Copy a run directory with its ions permuted in POSCAR and PROCAR.

    ``order`` is the new ion sequence, 0-based in the source's order, and has
    to keep the POSCAR's species blocks intact (for SrTiO3, a permutation of
    the three O ions).  Reordering the ions of a run consistently is an exact
    transformation: the same calculation, written down in another atom order.
    """
    lines = open(os.path.join(source, "POSCAR")).read().splitlines()
    head, positions = lines[:8], lines[8:8 + len(order)]
    with open(os.path.join(destination, "POSCAR"), "w") as handle:
        handle.write("\n".join(head + [positions[index] for index in order])
                     + "\n")
    name = ("PROCAR.gz" if os.path.isfile(os.path.join(source, "PROCAR.gz"))
            else "PROCAR")
    opener = gzip.open if name.endswith(".gz") else open
    text = opener(os.path.join(source, name), "rt").read().splitlines()
    out, index = [], 0
    while index < len(text):
        line = text[index]
        out.append(line)
        index += 1
        if not line.startswith("ion "):
            continue
        block = text[index:index + len(order)]
        if len(block) < len(order) or not all(
                row.split() and row.split()[0].isdigit() for row in block):
            continue
        for new, old in enumerate(order):
            row = block[old]
            out.append(f"{new + 1:5d}" + row[5:])
        index += len(order)
    with opener(os.path.join(destination, name), "wt") as handle:
        handle.write("\n".join(out) + "\n")


def _test_03_vasp_settings(base: list, poscar: str) -> None:
    """``--vasp`` forms, the optional ``-c``, the origin shift and the names.

    The user-reported failure this covers: three runs written with Ti at the
    origin and a ``-c`` file with Sr there.  ``--vasp`` used to take one ROOT
    only (so naming the three directories was an argparse error) and the
    engine demanded atom-by-atom identity with the ``-c`` primitive cell (so
    the other origin was an error too).
    """
    print("  -- --diagram --vasp settings: the forms of --vasp, the optional "
          "-c, the origin shift --")
    shifted = os.path.join(VASP_FIXTURE, "POSCAR-finish")
    runs = [os.path.join(VASP_FIXTURE, name) for name in
            ("BAND_sublattice2", "BAND", "BAND_sublattice1")]

    with tempfile.TemporaryDirectory() as tmp:
        # D -- no -c: the structure is the crystal run's POSCAR and the page
        # is named after the compound, never after the "BAND" directory
        code, out = run_cli(["--diagram", "--co-left", "SrTi", "--co-right",
                             "O3", "--vasp", VASP_FIXTURE], cwd=tmp)
        page = os.path.join(tmp, "CrystOD_SrTiO3_Pm-3m_vasp.html")
        html = open(page).read() if os.path.isfile(page) else ""
        report("--vasp without -c: the crystal run's POSCAR is the structure, "
               "the page is CrystOD_SrTiO3_Pm-3m_vasp.html titled 'SrTiO3, "
               "Pm-3m'",
               code == 0 and os.path.isfile(page)
               and "<title>Crystal orbital diagram: SrTiO3, Pm-3m" in html
               and os.path.isfile(os.path.join(
                   tmp, "CrystOD_SrTiO3_Pm-3m_vasp_levels.txt")), out)
        plain = _vasp_levels(os.path.join(
            tmp, "CrystOD_SrTiO3_Pm-3m_vasp_levels.txt"))

        # D -- a -c file INSIDE a run directory is not a name either
        code, out = run_cli(base + ["--vasp", VASP_FIXTURE], cwd=tmp)
        report("-c inside a run directory is named after the compound too, "
               "and gives the same levels as no -c at all",
               code == 0 and _vasp_levels(os.path.join(
                   tmp, "CrystOD_SrTiO3_Pm-3m_vasp_levels.txt")) == plain, out)

        # D -- an ordinary -c file keeps the convention of the other engines
        code, out = run_cli(["-c", shifted, "--diagram", "--co-left", "SrTi",
                             "--co-right", "O3", "--vasp", VASP_FIXTURE],
                            cwd=tmp)
        table = os.path.join(tmp, "CrystOD_POSCAR-finish_vasp_levels.txt")
        report("an ordinary -c file keeps the CrystOD_{cell}_vasp.html "
               "convention",
               code == 0 and os.path.isfile(table)
               and os.path.isfile(os.path.join(
                   tmp, "CrystOD_POSCAR-finish_vasp.html")), out)

        # B -- the same crystal on the OTHER origin: everything but the labels
        shift_levels = _vasp_levels(table)
        numbers_equal = all(key in shift_levels
                            and plain[key][1:] == shift_levels[key][1:]
                            for key in plain)
        report("a -c file on another origin (Sr at 0, runs with Ti at 0) is "
               "mapped onto: same levels, same energies, same shifts, same "
               "projected weights",
               set(shift_levels) == set(plain) and numbers_equal,
               f"{len(plain)} levels")
        relabel = {}
        for key, entry in plain.items():
            if key in shift_levels:
                relabel.setdefault((key[0], entry[0]), set()).add(
                    shift_levels[key][0])
        one_to_one = all(len(values) == 1 for values in relabel.values())
        at_r = {irrep: values.pop() for (k, irrep), values
                in relabel.items() if k == "R"}
        report("the irrep labels follow the origin, one-to-one per k point "
               "(R: 1+ <-> 2-, 3+ <-> 3-, 4+ <-> 5-)",
               one_to_one and at_r.get("R1+") == "R2-"
               and at_r.get("R2-") == "R1+" and at_r.get("R3+") == "R3-"
               and at_r.get("R4+") == "R5-" and at_r.get("R5+") == "R4-",
               str(at_r))
        # C -- the mapping is stated everywhere it can be read
        json_path = os.path.join(tmp, "CrystOD_POSCAR-finish_vasp_levels.json")
        document = json.load(open(json_path))
        setting = document.get("setting", {})
        page_html = open(os.path.join(
            tmp, "CrystOD_POSCAR-finish_vasp.html")).read()
        report("the mapping is stated in the report, the level table, the "
               "JSON and on the page",
               "the -c cell shifted by (1/2, 1/2, 1/2)" in out
               and "atom order Sr Ti O O O -> Sr#1 Ti#2 O#5 O#3 O#4" in out
               and "irrep labels refer to the " in out
               and "# setting  " in open(table).read()
               and setting.get("identity") is False
               and setting["runs"]["mo"]["shift_label"] == "(1/2, 1/2, 1/2)"
               and setting["runs"]["mo"]["ions"] == [1, 2, 5, 3, 4]
               and setting["runs"]["right"]["wraps"][1] == [-1, -1, -1]
               and "shifted by (1/2, 1/2, 1/2)" in page_html, out)

    # A -- the three run directories, in an order that is neither the
    # alphabetical one nor crystal-first
    with tempfile.TemporaryDirectory() as tmp:
        code, out = run_cli(base + ["--vasp", *runs], cwd=tmp)
        report("--vasp with the three run directories in any order: the "
               "crystal run is the one without Va species",
               code == 0
               and f"crystal run  : {os.path.join(VASP_FIXTURE, 'BAND')}" in out
               and "left  run  : " + os.path.join(
                   VASP_FIXTURE, "BAND_sublattice1", "band") in out
               and "right run  : " + os.path.join(
                   VASP_FIXTURE, "BAND_sublattice2") in out, out)
        triple = _vasp_levels(os.path.join(
            tmp, "CrystOD_SrTiO3_Pm-3m_vasp_levels.txt"))
        report("the three-directory form gives exactly the ROOT form's output",
               triple == plain, f"{len(triple)} levels")
        # an override names the same run in the other spelling (DIR against
        # DIR/band), which must not leave that directory in the search set
        code, out = run_cli(base + ["--vasp", *runs, "--vasp-left",
                                    os.path.join(VASP_FIXTURE,
                                                 "BAND_sublattice1")], cwd=tmp)
        report("--vasp-left overrides one column of the three-directory form",
               code == 0 and "left  run  : " + os.path.join(
                   VASP_FIXTURE, "BAND_sublattice1", "band")
               + " (--vasp-left)" in out
               and _vasp_levels(os.path.join(
                   tmp, "CrystOD_SrTiO3_Pm-3m_vasp_levels.txt")) == plain, out)

    # A -- --vasp with no path at all: the current directory is the ROOT
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "root")
        shutil.copytree(VASP_FIXTURE, root)
        code, out = run_cli(["--diagram", "--co-left", "SrTi", "--co-right",
                             "O3", "--vasp"], cwd=root)
        report("--vasp with no path takes the current directory as the ROOT",
               code == 0 and _vasp_levels(os.path.join(
                   root, "CrystOD_SrTiO3_Pm-3m_vasp_levels.txt")) == plain, out)

    # B -- the same crystal in a basis the mapping cannot use: the analysis
    # falls back to the crystal run's OWN setting and says so.  spglib
    # standardizes a -c file into the basis the runs use, so the branch has to
    # be provoked: the mapping is made to fail once, for the -c cell only.
    with tempfile.TemporaryDirectory() as tmp:
        page = os.path.join(tmp, "fallback.html")
        script = (
            "import crystod.vasp_io as vio\n"
            "from crystod.star_of_k import read_poscar_or_exit\n"
            "from crystod import crystal_orbital_vasp as cov\n"
            "real, seen = vio.map_run_onto_cell, []\n"
            "def flaky(lattice, positions, symbols, structure, **kw):\n"
            "    if not seen and abs(float(positions[0][0])) < 1e-9:\n"
            "        seen.append(1)\n"
            "        raise vio.StructureMappingError(\n"
            "            'the run\\'s lattice has the same lengths and angles "
            "but a rotated basis (relative difference 1.00e-01)', 'basis')\n"
            "    return real(lattice, positions, symbols, structure, **kw)\n"
            "vio.map_run_onto_cell = flaky\n"
            f"cov.report_and_write(read_poscar_or_exit({shifted!r}),\n"
            "    left=['SrTi'], right=['O3'], symprec=1e-5, electrons=None,\n"
            f"    kpoint_filter='R', output_path={page!r},\n"
            f"    structure_label='fallback', root={VASP_FIXTURE!r},\n"
            f"    cell_path={shifted!r})\n")
        code, out = run_python(script)
        fallback = _vasp_levels(os.path.join(tmp, "fallback_levels.txt"))
        at_r = {key: value[0] for key, value in plain.items() if key[0] == "R"}
        # one k point means a narrower drawn window, so only the levels both
        # runs kept are compared -- their labels must be the run's own setting
        shared = [key for key in fallback if key in at_r]
        report("a -c cell in a basis the mapping cannot use falls back to the "
               "crystal run's own setting, says so, and labels in it",
               code == 0 and "same crystal in DIFFERENT BASES" in out
               and "the analysis is done in the crystal run's own setting" in out
               and len(shared) >= 15
               and all(fallback[key][0] == at_r[key] for key in shared), out)

    # B -- the ions of the crystal run permuted (POSCAR and PROCAR together):
    # the same calculation written in another atom order
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "root")
        shutil.copytree(VASP_FIXTURE, root)
        _reorder_vasp_run(os.path.join(VASP_FIXTURE, "BAND"),
                          os.path.join(root, "BAND"), [0, 1, 4, 3, 2])
        page = os.path.join(tmp, "reordered.html")
        # -c is the ORIGINAL atom order, the crystal run the permuted one
        code, out = run_cli(base + ["--vasp", root, "--output", page], cwd=tmp)
        report("the crystal run's ions reordered in POSCAR and PROCAR "
               "together: the engine permutes them back, same levels",
               code == 0 and _vasp_levels(os.path.join(
                   tmp, "reordered_levels.txt")) == plain, out)


#: Other spellings of a POSCAR that VASP reads as the SAME crystal:
#: ``name -> (line 2, the per-axis factors VASP resolves it to, coordinates)``.
#: ``None`` for line 2 is minus the cell volume, the lattice being written
#: 1.6 times too small.
POSCAR_SPELLINGS = {
    "Cartesian, scale 2": ("2.0", (2.0, 2.0, 2.0), "Cartesian"),
    "Cartesian, scale 0.5": ("0.5", (0.5, 0.5, 0.5), "Cartesian"),
    "Cartesian, scale -V": (None, (1.6, 1.6, 1.6), "Cartesian"),
    "Direct, scale -V": (None, (1.6, 1.6, 1.6), "Direct"),
    "Cartesian, three factors": ("2.0 0.5 1.25", (2.0, 0.5, 1.25), "Cartesian"),
}


def _respell_poscar(source: str, destination: str, spelling: str | None,
                    lattice=None, positions=None) -> None:
    """Write the POSCAR ``source`` (VASP 5, direct coordinates, scale 1) in
    another spelling of the same crystal (a key of :data:`POSCAR_SPELLINGS`;
    ``None`` keeps scale 1 and direct coordinates).

    VASP multiplies the lattice vectors AND Cartesian coordinates by the
    factors of line 2, a single negative factor being the cell volume, so the
    numbers stored here are the true ones divided by the factors.  ``lattice``
    and ``positions`` (fractional) replace those of the file: another crystal
    with the same species lines.
    """
    lines = open(source).read().splitlines()
    cell = np.array([[float(x) for x in lines[index].split()[:3]]
                     for index in (2, 3, 4)] if lattice is None else lattice)
    total = sum(int(value) for value in lines[6].split())
    fractional = np.array([[float(x) for x in lines[8 + row].split()[:3]]
                           for row in range(total)] if positions is None
                          else positions)
    line2, factors, mode = (POSCAR_SPELLINGS[spelling] if spelling
                            else ("1.0", (1.0, 1.0, 1.0), "Direct"))
    if line2 is None:
        line2 = repr(-abs(float(np.linalg.det(cell))))
    rows = fractional if mode == "Direct" else (fractional @ cell) / factors

    def numbers(row):
        return "  " + "  ".join(repr(float(value)) for value in row)

    with open(destination, "w") as handle:
        handle.write("\n".join(
            [lines[0], line2, *(numbers(row) for row in cell / np.array(factors)),
             lines[5], lines[6], mode, *(numbers(row) for row in rows)]) + "\n")


def _written_poscar(path: str):
    """``(comment, species, counts and mode; lattice; coordinates)`` of a
    POSCAR as CrystOD writes it (VASP 5, direct coordinates)."""
    lines = open(path).read().splitlines()
    lattice = float(lines[1]) * np.array(
        [[float(x) for x in lines[index].split()[:3]] for index in (2, 3, 4)])
    total = sum(int(value) for value in lines[6].split())
    positions = np.array([[float(x) for x in lines[8 + row].split()[:3]]
                          for row in range(total)])
    header = (lines[0].split(), lines[5].split(), lines[6].split(),
              lines[7].strip())
    return header, lattice, positions


def _test_03_poscar_spellings() -> None:
    """One POSCAR reading rule: a file is the crystal VASP reads from it.

    VASP multiplies the lattice vectors AND Cartesian coordinates by the scale
    line, and a negative scale is the cell volume.  ``crystod.vasp_io`` left
    Cartesian coordinates unscaled; phonopy 4.3.0, which read every ``-c``
    file, does the same, takes a negative scale for a factor and cannot read
    three factors; pymatgen multiplies Cartesian coordinates by the negative
    number.  ``--vasp-setup`` therefore wrote the sublattice POSCARs of a
    Cartesian file with scale 2 with every atom at HALF its coordinates and
    exit 0 (or, when the misplaced copy had another primitive basis, as for
    the cubic fixture, stopped with "rotated basis"), and ``--vasp`` mapped
    the runs the same way.  Every reader now takes the cell from
    ``crystod.vasp_io.poscar_geometry``; below, the fixture rewritten in five
    other spellings has to give what its Direct files give.
    """
    import ast
    import warnings

    from phonopy.interface.vasp import read_vasp
    from pymatgen.io.vasp.inputs import Poscar

    from crystod.star_of_k import read_poscar_or_exit
    from crystod.vasp_io import poscar_structure, read_poscar
    from crystod.vasp_wavecar import read_run_structure

    print("  -- one POSCAR reading rule: the scale line and Cartesian "
          "coordinates as VASP reads them --")
    crystal = os.path.join("BAND", "POSCAR")
    runs = (crystal, os.path.join("BAND_sublattice1", "band", "POSCAR"),
            os.path.join("BAND_sublattice2", "POSCAR"))
    command = ["--diagram", "--co-left", "SrTi", "--co-right", "O3"]
    table = "CrystOD_SrTiO3_Pm-3m_vasp_levels.txt"
    # rows and columns of a cubic cell cannot be told apart: the readers are
    # also given the same three files on a triclinic lattice
    triclinic = [[4.1, 0.3, -0.2], [1.2, 5.3, 0.4], [-0.7, 0.9, 6.2]]

    def by_vasp_io(path):
        structure = read_poscar(path)
        return structure.lattice, structure.positions

    def by_run_structure(path):
        structure = read_run_structure(path)
        return structure.lattice, structure.positions

    def by_phonopy(path):
        cell = read_poscar_or_exit(path)
        return cell.cell, cell.scaled_positions

    def by_pymatgen(path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structure = poscar_structure(
                Poscar.from_file(path, read_velocities=False).structure, path)
        return structure.lattice.matrix, structure.frac_coords

    readers = {
        "crystod.vasp_io.read_poscar": by_vasp_io,
        "crystod.vasp_wavecar.read_run_structure": by_run_structure,
        "crystod.star_of_k.read_poscar_or_exit (phonopy's reader, corrected)":
            by_phonopy,
        "pymatgen's reader through crystod.vasp_io.poscar_structure":
            by_pymatgen,
    }

    def same_poscar(path, reference):
        if not (os.path.isfile(path) and os.path.isfile(reference)):
            return False
        header, lattice, positions = _written_poscar(path)
        ref_header, ref_lattice, ref_positions = _written_poscar(reference)
        return bool(header == ref_header
                    and np.abs(lattice - ref_lattice).max() < 1e-9
                    and np.abs(positions - ref_positions).max() < 1e-9)

    with tempfile.TemporaryDirectory() as tmp:
        # one copy of the fixture per spelling, the three POSCARs rewritten
        roots = {None: os.path.join(tmp, "direct", "root")}
        shutil.copytree(VASP_FIXTURE, roots[None])
        for number, spelling in enumerate(POSCAR_SPELLINGS):
            roots[spelling] = os.path.join(tmp, f"spelling{number}", "root")
            shutil.copytree(VASP_FIXTURE, roots[spelling])
            for run in runs:
                _respell_poscar(os.path.join(VASP_FIXTURE, run),
                                os.path.join(roots[spelling], run), spelling)
        skew = os.path.join(tmp, "triclinic")
        os.makedirs(skew)
        for number, spelling in enumerate([None, *POSCAR_SPELLINGS]):
            for index, run in enumerate(runs):
                _respell_poscar(os.path.join(VASP_FIXTURE, run),
                                os.path.join(skew, f"{number}_{index}"),
                                spelling, lattice=triclinic)

        # ------------------------------------------------------ the readers
        worst = {name: 0.0 for name in readers}
        failures = {name: [] for name in readers}
        for number, spelling in enumerate(POSCAR_SPELLINGS, start=1):
            for index, run in enumerate(runs):
                pairs = ((os.path.join(roots[spelling], run),
                          os.path.join(roots[None], run), "cubic"),
                         (os.path.join(skew, f"{number}_{index}"),
                          os.path.join(skew, f"0_{index}"), "triclinic"))
                for name, read in readers.items():
                    # pymatgen knows neither the Va species of the sublattice
                    # runs nor three scale factors
                    if read is by_pymatgen and (run != crystal
                                                or "three" in spelling):
                        continue
                    for path, reference, lattice_name in pairs:
                        try:
                            lattice, positions = read(path)
                            ref_lattice, ref_positions = read(reference)
                            deviation = float(max(
                                np.abs(lattice - ref_lattice).max(),
                                np.abs(positions - ref_positions).max()))
                        except (Exception, SystemExit) as error:  # noqa: BLE001
                            deviation = float("inf")
                            failures[name].append(
                                f"{spelling}, {lattice_name}, {run}: {error}")
                        else:
                            if deviation >= 1e-9:
                                failures[name].append(
                                    f"{spelling}, {lattice_name}, {run}: "
                                    f"differs by {deviation:.3g}")
                        worst[name] = max(worst[name], deviation)
        for name in readers:
            report(f"{name}: the fixture POSCARs in Cartesian coordinates "
                   "with scale 2, 0.5 and -V (the volume), in direct "
                   "coordinates with scale -V"
                   + ("" if readers[name] is by_pymatgen
                      else " and with three per-axis factors")
                   + " give the lattice and fractional coordinates of the "
                     "Direct files, on the cubic and on a triclinic lattice "
                     f"(worst {worst[name]:.1e})",
                   worst[name] < 1e-9, "\n".join(failures[name]))

        # a file phonopy reads as VASP does is not touched at all
        untouched = []
        for path in [*(os.path.join(VASP_FIXTURE, run) for run in runs),
                     POSCAR_ScF3, POSCAR_SrTiO3, POSCAR_NaCl]:
            ours, theirs = read_poscar_or_exit(path), read_vasp(path)
            if not (np.array_equal(ours.cell, theirs.cell)
                    and np.array_equal(ours.scaled_positions,
                                       theirs.scaled_positions)
                    and list(ours.symbols) == list(theirs.symbols)):
                untouched.append(path)
        report("a POSCAR in direct coordinates with a positive scale is read "
               "exactly as before: read_poscar_or_exit returns phonopy's own "
               "cell, bit for bit", not untouched, "\n".join(untouched))

        # ---------------------- the anchor engine's run mapping, --vasp-setup
        reference_left = os.path.join(VASP_FIXTURE, runs[1])
        reference_right = os.path.join(VASP_FIXTURE, runs[2])
        reference = None
        for spelling, root in roots.items():
            where = os.path.dirname(root)
            code, out = run_cli(command + ["--vasp", "root"], cwd=where)
            levels = (_vasp_levels(os.path.join(where, table))
                      if os.path.isfile(os.path.join(where, table)) else {})
            if spelling is None:
                reference = levels
                report("--diagram --vasp on the Direct fixture, without -c "
                       "(the reference of the spellings below)",
                       code == 0 and len(levels) > 0, out)
            else:
                report(f"--diagram --vasp, the three runs written '{spelling}': "
                       "the runs are mapped onto the cell as for the Direct "
                       "files, same level table",
                       code == 0 and len(levels) > 0 and levels == reference,
                       out)
            if spelling is None:
                continue
            for name in ("BAND_sublattice1", "BAND_sublattice2"):
                shutil.rmtree(os.path.join(root, name))
            code, out = run_cli(command + ["--vasp-setup", "root"], cwd=where)
            report(f"--vasp-setup, the crystal POSCAR written '{spelling}': "
                   "the sublattice POSCARs are those written for the Direct "
                   "file (species, atom order, lattice, coordinates)",
                   code == 0 and "Traceback" not in out
                   and same_poscar(os.path.join(root, "BAND_sublattice1",
                                                "POSCAR"), reference_left)
                   and same_poscar(os.path.join(root, "BAND_sublattice2",
                                                "POSCAR"), reference_right),
                   out)

        # the SILENT case: a crystal without symmetry keeps its primitive
        # basis when its atoms are misplaced, so nothing stopped the run and
        # every atom was written at half its coordinates
        box = [[3.9, 0.0, 0.0], [0.0, 4.3, 0.0], [0.0, 0.0, 4.9]]
        sites = [[0.47, 0.52, 0.55], [0.03, 0.02, 0.06], [0.04, 0.07, 0.46],
                 [0.44, 0.03, 0.08], [0.06, 0.41, 0.02]]
        written = {}
        codes = []
        for spelling in (None, "Cartesian, scale 2"):
            where = os.path.join(tmp, "low_" + ("cartesian" if spelling else "direct"))
            os.makedirs(os.path.join(where, "root", "BAND"))
            _respell_poscar(os.path.join(VASP_FIXTURE, crystal),
                            os.path.join(where, "root", crystal), spelling,
                            lattice=box, positions=sites)
            code, out = run_cli(command + ["--vasp-setup", "root"], cwd=where)
            codes.append(code)
            written[spelling] = [os.path.join(where, "root", name, "POSCAR")
                                 for name in ("BAND", "BAND_sublattice1",
                                              "BAND_sublattice2")]
        placed = (os.path.isfile(written[None][1])
                  and np.abs(_written_poscar(written[None][1])[2]
                             - np.array(sites)).max() < 1e-9)
        report("--vasp-setup on a crystal without symmetry written in "
               "Cartesian coordinates with scale 2: the three POSCARs it "
               "writes hold the atoms where the Direct file has them (every "
               "atom used to be written at half its coordinates, exit 0)",
               codes == [0, 0] and placed
               and all(same_poscar(path, reference) for path, reference
                       in zip(written["Cartesian, scale 2"], written[None])),
               out)

    # ----------------------------------------------- the rule has no bypass
    phonopy_reads, uncorrected = [], []
    for path in sorted(glob.glob(os.path.join(ROOT, "crystod", "**", "*.py"),
                                 recursive=True)):
        if os.path.basename(path) == "vasp_io.py":
            continue
        calls = [node for node in ast.walk(ast.parse(open(path).read()))
                 if isinstance(node, ast.Call)]
        names = [getattr(node.func, "attr", getattr(node.func, "id", ""))
                 for node in calls]
        for node, name in zip(calls, names):
            if (name in ("read_vasp", "read_vasp_from_strings",
                         "read_crystal_structure")
                    or any(keyword.arg == "unitcell_filename"
                           for keyword in node.keywords)):
                phonopy_reads.append(
                    f"{os.path.relpath(path, ROOT)}:{node.lineno}")
        reads = [node.lineno for node, name in zip(calls, names)
                 if name == "from_file"
                 and getattr(node.func.value, "id", "") in ("Structure", "Poscar")]
        if reads and "poscar_structure" not in names:
            uncorrected.append(f"{os.path.relpath(path, ROOT)}:{reads[0]}")
    report("no CrystOD module reads a POSCAR with phonopy's own reader "
           "(read_vasp, read_crystal_structure, phonopy.load(unitcell_filename"
           "=...)): every cell comes through crystod.vasp_io.read_poscar_cell",
           not phonopy_reads, "\n".join(phonopy_reads))
    report("every module that reads a structure file with pymatgen passes it "
           "through crystod.vasp_io.poscar_structure",
           not uncorrected, "\n".join(uncorrected))


def _test_03_vasp_lak() -> None:
    """``--diagram --vasp`` on a meta-GGA run with a long band list.

    ``example/03_hybridization/vasp_SrTiO3_LAK`` is a trimmed copy of three
    real SrTiO3 runs with METAGGA = LAK and NBANDS = 64 (see its README).
    Two defects showed up on it.  The rigid alignment paired the Sr 4s R2-
    anchor with the empty Sr 5s-like level R2- #2, which the radially blind
    PROCAR projection cannot tell from the semicore band (+15.695 eV with a
    45.779 eV spread instead of +4.257 eV); the PBE fixture has the same
    tie and escaped it only because its two overlaps are bit-identical.  And
    the functional, which a meta-GGA OUTCAR names on a METAGGA line only,
    came out as "PAW".  Every number checked is the number the untrimmed
    runs give.
    """
    print("  -- --diagram --vasp SrTiO3 METAGGA = LAK, NBANDS = 64 (radially "
          "blind anchor ties, the meta-GGA label) --")

    # ------------------------------------------------ the functional label
    # VASP 6 names a meta-GGA on "METAGGA = LAK    functional components"
    # and writes no GGA line; VASP 5.4.4 writes " METAGGA = SCAN  LMAXTAU =
    # 6 ..." for a meta-GGA run and "METAGGA=  F  non-selfconsistent MetaGGA
    # calc." (the LMETAGGA flag) into EVERY run
    with tempfile.TemporaryDirectory() as tmp:
        samples = {
            "vasp5_gga": ("   METAGGA=      F    non-selfconsistent MetaGGA "
                          "calc.\n   GGA     =    PS    GGA type\n"),
            "vasp5_scan": (" METAGGA = SCAN     LMAXTAU =  6    LMIXTAU =  F\n"
                           "   METAGGA=      F    non-selfconsistent MetaGGA "
                           "calc.\n   GGA     =    --    GGA type\n"),
            "vasp6_r2scan": ("   METAGGA = r2scan\n"
                             "   METAGGA = R2SCAN    functional components\n"),
        }
        for name, text in samples.items():
            with open(os.path.join(tmp, name), "w") as handle:
                handle.write(text)
        paths = [os.path.join(VASP_LAK_FIXTURE, "BAND", "OUTCAR"),
                 os.path.join(VASP_FIXTURE, "BAND", "OUTCAR")]
        paths += [os.path.join(tmp, name) for name in samples]
        code, out = run_python(
            "from crystod.vasp_io import read_outcar;"
            f"paths = {paths!r};"
            "[print(f['functional'], f['metagga'], f['gga']) "
            "for f in map(read_outcar, paths)]")
    report("OUTCAR functional: METAGGA wins over GGA (LAK, SCAN, R2SCAN), GGA "
           "is the fallback, and VASP 5's 'METAGGA= F' flag names nothing",
           code == 0 and out.splitlines()[:5] == [
               "LAK LAK None", "PE None PE", "PS None PS", "SCAN SCAN --",
               "R2SCAN R2SCAN None"], out)

    # ------------------------------------- the anchor counterpart assignment
    code, out = run_python(
        "from types import SimpleNamespace as L;"
        "from crystod.crystal_orbital_pyscf import "
        "_alignment_counterparts as pair;"
        "frag = lambda i, e: L(level_id=f'left{i}', energy=e);"
        "cry = lambda i, e, w: L(level_id=f'mo{i}', energy=e, "
        "absolute_composition=w, composition=[]);"
        # the radially blind ladder of SrTiO3 R2-: the Sr 4s/5s/6s fragment
        # levels against the semicore band and the two empty s-like crystal
        # levels, every overlap ~1 -- the semicore band a shade LESS pure
        # than the empty ones, which must not decide anything
        "w = lambda a: [('left0', a), ('left1', a), ('left2', a)];"
        "p = pair([frag(0, -38.3), frag(1, 12.4), frag(2, 34.9)], "
        "[cry(0, -34.1, w(0.99)), cry(1, 11.7, w(1.0)), "
        "cry(2, 35.3, w(1.0))]);"
        "print([(k, v[1].level_id) for k, v in p.items()]);"
        # a unique inert counterpart stays the highest-overlap one, and a
        # level with no inert candidate takes its best crystal level that is
        # not already some fragment level's inert counterpart
        "p = pair([frag(0, -20.0), frag(1, -5.0)], "
        "[cry(0, -18.0, [('left0', 0.95), ('left1', 0.60)]), "
        "cry(1, -3.0, [('left0', 0.02), ('left1', 0.55)])]);"
        "print([(k, v[1].level_id, v[0]) for k, v in p.items()])")
    lines = out.splitlines()
    report("anchor counterparts one-to-one in energy order: the radially "
           "blind Sr 4s/5s/6s R2- ladder pairs level by level, whatever the "
           "last digits of the overlaps say",
           code == 0 and lines[:1] == [
               "[('left0', 'mo0'), ('left1', 'mo1'), ('left2', 'mo2')]"], out)
    report("a unique inert counterpart is kept; a level without one takes its "
           "best crystal level that no inert pair has taken",
           code == 0 and lines[1:2] == [
               "[('left0', 'mo0', 0.95), ('left1', 'mo1', 0.55)]"], out)

    base = ["--diagram", "--co-left", "SrTi", "--co-right", "O3",
            "--vasp", VASP_LAK_FIXTURE]
    with tempfile.TemporaryDirectory() as tmp:
        page = os.path.join(tmp, "sto_lak.html")
        code, out = run_cli(base + ["--output", page], cwd=tmp)
        report("--diagram --vasp SrTiO3 LAK exit 0", code == 0, out)
        report("the report names the meta-GGA (LAK, not PAW)",
               "functional LAK, ENCUT 550 eV, E-fermi 2.829 eV (crystal run), "
               "VBM 2.484 eV" in out, out)
        report("rigid diagnostic anchored on the Sr 4s band itself: "
               "+4.257 eV, spread 0.025 eV (was +15.695 eV, spread 45.779 eV)",
               "diagnostic, the RIGID alignment (--vasp-align rigid): "
               "left +4.257 eV | right +6.559 eV" in out
               and "left  anchored on Sr 4s R2- (-35.85 eV, counterpart "
                   "purity 100%, spread 0.025 eV over 4 anchor levels at 4 k "
                   "points)" in out, out)
        report("the default site-resolved alignment is what it was "
               "(Sr +4.154 / Ti +14.088 / O +6.284 eV)",
               "Sr  delta =   +4.154 eV   11 anchors (2 symmetry-forbidden)"
               in out
               and "Ti  delta =  +14.088 eV   9 anchors (2 symmetry-forbidden)"
                   in out
               and "O   delta =   +6.284 eV   14 anchors (2 symmetry-forbidden)"
                   in out
               and "R    Sr 4s R2-        forbidden w=1.000 -> R2- #1         "
                   "+4.268 eV" in out, out)
        html = open(page).read() if os.path.isfile(page) else ""
        table_path = os.path.join(tmp, "sto_lak_levels.txt")
        table = open(table_path).read() if os.path.isfile(table_path) else ""
        json_path = os.path.join(tmp, "sto_lak_levels.json")
        document = (json.load(open(json_path))
                    if os.path.isfile(json_path) else {})
        report("page chip, foot, level table and JSON carry LAK and the "
               "corrected rigid shift",
               "VASP PAW/LAK, ENCUT 550 eV" in html
               and "VASP LAK eigenvalues" in html
               and "PAW/PAW" not in html
               and "# rigid    left +4.2570 | right +6.5594 eV" in table
               and document.get("method", {}).get("functional") == "LAK"
               and abs(document["alignment"]["rigid"]["left"] - 4.257039)
               < 1e-6
               and abs(document["alignment"]["sites"]["left"]["Sr"]["delta"]
                       - 4.153636) < 1e-6, json_path)

        # ONE k point: the tie used to decide the whole rigid shift (+50.022
        # eV), and through the first-pass anchor-window cut it emptied the
        # cation column of the default site-resolved page as well
        page = os.path.join(tmp, "sto_lak_R.html")
        code, out = run_cli(base + ["--kpoint", "R", "--output", page],
                            cwd=tmp)
        report("--kpoint R: the rigid shift is the semicore pair's +4.268 eV "
               "and the cation column keeps its Sr and Ti anchors",
               code == 0
               and "diagnostic, the RIGID alignment (--vasp-align rigid): "
                   "left +4.268 eV | right +6.560 eV" in out
               and "Sr  delta =   +4.268 eV   2 anchors (1 symmetry-forbidden)"
                   in out
               and "Ti  delta =  +16.519 eV   1 anchors (0 symmetry-forbidden)"
                   in out
               and "SrTi      : Sr 4s R2- (-34.07), Sr 4p R5+ (-15.30), "
                   "Ti 3d R5+ (4.74), Ti 3d R3+ (5.17)" in out, out)

        page = os.path.join(tmp, "sto_lak_rigid.html")
        code, out = run_cli(base + ["--vasp-align", "rigid", "--output", page],
                            cwd=tmp)
        report("--vasp-align rigid puts the Sr 4s fragment levels on their own "
               "crystal band (left +4.257 eV)",
               code == 0
               and "left  delta = +4.257 eV, anchored on Sr 4s R2-" in out
               and "shifts : left +1.773 eV | crystal -2.484 eV | "
                   "right +4.075 eV" in out
               and "SrTi      : Sr 4s GM1+ (-34.11)," in out
               and re.search(r"GM1\+ #1\s+-34\.13 eV", out) is not None, out)


def _test_03_vasp_engine() -> None:
    """``crystod --diagram --vasp``: the plane-wave crystal-orbital engine.

    Runs entirely offline from ``example/03_hybridization/vasp_SrTiO3``, a
    trimmed copy of three real SrTiO3 VASP runs (crystal, cation sublattice,
    anion sublattice) restricted to the four special k points the diagram
    uses.  Every number checked below is the number the untrimmed 288-k-point
    runs give.
    """
    print("  -- --diagram --vasp (plane-wave engine, offline fixture) --")
    poscar = os.path.join(VASP_FIXTURE, "BAND", "POSCAR")
    base = ["-c", poscar, "--diagram", "--co-left", "SrTi", "--co-right", "O3"]

    # ---------------------------------------------------------- the readers
    code, out = run_python(
        "from crystod.vasp_io import (parse_va_species, va_species_name, "
        "read_poscar, read_procar, read_eigenval, read_outcar);"
        "import os;"
        f"root = {VASP_FIXTURE!r};"
        "print([parse_va_species(n) for n in "
        "('Va2-', 'Va2+', 'Va4+', 'Va', 'O', 'Na')]);"
        "print([va_species_name(q) for q in (-2.0, 2.0, 4.0)]);"
        "s = read_poscar(os.path.join(root, 'BAND_sublattice1', 'band', 'POSCAR'));"
        "print(s.species, s.counts, s.real_elements, s.charges);"
        "p = read_procar(os.path.join(root, 'BAND', 'PROCAR'), "
        "kpoints=[[0, 0, 0]]);"
        "print(p.n_ions, p.n_bands, p.n_kpoints, p.lm, sorted(p.loaded));"
        "print(p.phases[list(p.loaded)[0]].shape);"
        "k, w, e, o = read_eigenval(os.path.join(root, 'BAND', 'EIGENVAL'));"
        "print(e.shape, round(float(e[0, 0]), 4), round(float(o[0, 0]), 4));"
        "f = read_outcar(os.path.join(root, 'BAND', 'OUTCAR'));"
        "print(f['nelect'], f['ispin'], f['encut'], f['lorbit'], f['functional']);"
        "print([(x.element, x.valence_shells()) for x in f['species']])"
    )
    lines = out.splitlines()
    report("Va point-charge species names parsed (Va2- Va2+ Va4+, not Va/O/Na)",
           code == 0 and lines[0] == "[-2.0, 2.0, 4.0, None, None, None]"
           and lines[1] == "['Va2-', 'Va2+', 'Va4+']", out)
    report("sublattice POSCAR read with its Va species and per-atom charges",
           code == 0
           and lines[2] == "['Sr', 'Ti', 'Va2-'] [1, 1, 3] ['Sr', 'Ti'] "
                           "[None, None, -2.0, -2.0, -2.0]", out)
    # GM is in the fixture twice (once weighted, once on the zero-weight
    # band list), as it is in the untrimmed 288-point run
    report("gzipped PROCAR read, phases kept for the requested k point only",
           code == 0
           and lines[3] == "5 32 8 ['s', 'py', 'pz', 'px', 'dxy', 'dyz', "
                           "'dz2', 'dxz', 'x2-y2'] [0, 3]"
           and lines[4] == "(32, 5, 9)", out)
    report("gzipped EIGENVAL read (same eigenvalues as the PROCAR)",
           code == 0 and lines[5] == "(8, 32) -30.1863 1.0", out)
    report("OUTCAR facts: NELECT 32, ISPIN 1, ENCUT 550, LORBIT 12, GGA PE",
           code == 0 and lines[6] == "32.0 1 550.0 12 PE", out)
    report("PAW valence shells from the dataset lines (Sr 4s/5s, 4p; Ti 3d; O 2s/2p)",
           code == 0
           and lines[7] == "[('Sr', {0: [4, 5], 1: [4]}), ('Ti', {}), "
                           "('O', {})]", out)

    with tempfile.TemporaryDirectory() as tmp:
        page = os.path.join(tmp, "sto_vasp.html")
        table = os.path.join(tmp, "sto_vasp_levels.txt")
        code, out = run_cli(base + ["--vasp", VASP_FIXTURE, "--output", page],
                            cwd=tmp)
        report("--diagram --vasp SrTiO3 exit 0", code == 0, out)
        report("run directories resolved from the POSCAR species, DIR/band "
               "used where it holds the PROCAR",
               "left  run  : " + os.path.join(VASP_FIXTURE,
                                              "BAND_sublattice1", "band") in out
               and "right run  : " + os.path.join(VASP_FIXTURE,
                                                  "BAND_sublattice2") in out, out)
        report("electron counts 32 = 8 + 24 and the Va point charges per column",
               "crystal SrTiO3  32 electrons = 8 + 24" in out
               and "left  SrTi     charge +6, 8 electrons (NELECT), point "
                   "charges Va2- x3" in out
               and "right O3       charge -6, 24 electrons (NELECT), point "
                   "charges Va2+ x1, Va4+ x1" in out, out)
        report("method and basis read from OUTCAR (PBE, 550 eV, LORBIT=12, 45 AOs)",
               "functional PBE, ENCUT 550 eV" in out
               and "E-fermi 2.794 eV (crystal run), VBM 2.604 eV" in out
               and "(atom, l, m) from the LORBIT=12 PROCAR, 45 components" in out,
               out)
        report("VACSIGMA / VACRWALL of both sublattice runs reported",
               out.count("VACSIGMA 0.5 A, VACRWALL 0.45 A") == 2, out)

        # the validation target: the irreps the PROCAR projections are
        # classified into must BE the site-symmetry induced representation
        for name, manifolds in (
            ("GM", (("Ti", "3d"), ("O", "2p"), ("O", "2s"), ("Sr", "4p"))),
            ("R", (("Ti", "3d"), ("O", "2p"), ("O", "2s"), ("Sr", "4p"))),
        ):
            block = _vasp_kpoint_block(out, name)
            for element, shell in manifolds:
                formula = "SrTi" if element in ("Sr", "Ti") else "O3"
                salc = _vasp_salc(block, element, shell)
                found = _vasp_found(block, formula, element, shell)
                report(f"{name}: {element} {shell} PROCAR irreps = SALC "
                       f"decomposition {' + '.join(salc) or '(none)'}",
                       bool(salc) and found == salc,
                       f"SALC {salc} vs PROCAR {found}\n{block}")

        report("every labelled level is a pure irrep (weight > 0.9)",
               "Irrep purity: every labelled level is above 0.9" in out
               and "Irrep purity warnings" not in out, out)
        # ------------------------------------- the site-resolved alignment
        # The default alignment fits ONE shift per (column, element): the
        # shells of one atom move together, different SITES do not.  On
        # SrTiO3 the Ti site is charged by covalent back-donation in the
        # crystal and every Ti level rises ~12 eV more than the ionic Sr's.
        report("site-resolved alignment: crystal column is the reference, "
               "energy zero E - E_VBM",
               "* Site-resolved alignment (reference = the crystal column, "
               "shift 0) *" in out
               and "energy zero: E - E_VBM  (VBM 2.604 eV, highest occupied "
                   "crystal eigenvalue over all k of the crystal run)" in out,
               out)
        report("per-element shifts Sr +3.530 / Ti +15.255 / O +6.091 eV, each "
               "fitted to the symmetry-forbidden probes alone",
               "Sr  delta =   +3.530 eV   14 anchors (5 symmetry-forbidden)"
               in out
               and "Ti  delta =  +15.255 eV   8 anchors (2 symmetry-forbidden)"
                   in out
               and "O   delta =   +6.091 eV   14 anchors "
                   "(2 symmetry-forbidden)" in out
               and out.count("fitted to the symmetry-forbidden probes ALONE")
                   == 3, out)
        report("the Ti - Sr site-potential change is +11.73 eV",
               "Ti - Sr: +11.73 eV" in out, out)
        report("the symmetry-forbidden Ti 3d probes (no O partner at GM/X) "
               "agree to 0.15 eV and carry the fit",
               "GM   Ti 3d GM5+       forbidden w=0.983 -> GM5+ #1       "
               "+15.177 eV" in out
               and "X    Ti 3d X4+        forbidden w=1.000 -> X4+ #1        "
                   "+15.332 eV" in out
               and "symmetry-forbidden probes only: +15.255 eV (n=2, spread "
                   "0.15 eV); after the shift they close to 0.078 eV"
                   in out, out)
        report("per-element scale test: Sr fails (shell spread 1.26 eV), "
               "Ti and O pass",
               "shell spread 1.26 eV -> NOT one scale (> 1 eV)" in out
               and "shell spread 0.00 eV -> shells agree" in out
               and "shell spread 0.12 eV -> shells agree" in out
               and "*** WARNING: Sr (left) is not described by ONE shift:"
                   in out, out)
        report("the rigid one-shift-per-column numbers are still printed as a "
               "diagnostic",
               "diagnostic, the RIGID alignment (--vasp-align rigid): "
               "left +4.388 eV | right +6.400 eV" in out
               and "left  anchored on Sr 4s R2- (-34.53 eV, counterpart "
                   "purity 100%," in out
               and "right anchored on O 2s GM1+ (-21.23 eV, counterpart "
                   "purity 86%," in out, out)
        report("occupied-trace diagnostic reproduces +0.040 eV per electron",
               "(128 electrons over the 4 k points) is off by +0.040 eV per "
               "electron" in out, out)
        report("bond character is classified only where the scale, the parent "
               "coverage and the window all allow it (27 of 62 neutral)",
               "27 not classified, 17 nonbonding, 10 antibonding, 8 bonding"
               in out, out)
        report("the measured alignment residual of the system is printed "
               "beside the deadband it has to be read against",
               "22 complete manifolds: mean |residual| 0.594 eV, max 1.248 eV"
               in out
               and "0.3 eV bond deadband should be read against" in out, out)
        report("crystal compositions at GM on the E - E_VBM scale (O 2p / "
               "Ti 4p valence band, pure Ti 3d conduction band)",
               re.search(r"GM4- #2\s+-2\.83 eV\s+x3\s+6e\s+O 2p GM4- 67\.3%\s+"
                         r"Ti 4p GM4- 30\.6%", out) is not None
               and re.search(r"GM5\+ #1\s+1\.72 eV\s+x3\s+Ti 3d GM5\+ 97\.8%",
                             out) is not None, out)
        report("the Ti 3d fragment levels now sit next to the crystal "
               "t2g / eg, the O 2p fragments next to the valence band",
               "SrTi      : Sr 4s GM1+ (-33.63), Sr 4p GM4- (-15.34), "
               "Ti 3d GM5+ (1.80), Ti 3d GM3+ (2.18)" in out
               and "O3        : O 2s GM1+ (-17.74), O 2s GM3+ (-15.95), "
                   "O 2p GM4-#1 (-3.98), O 2p GM5- (-1.18), "
                   "O 2p GM4-#2 (-0.37)" in out, out)

        html = open(page).read() if os.path.isfile(page) else ""
        report("HTML page written with the VASP method / embedding chips, the "
               "alignment chip and the E - E_VBM axis",
               os.path.isfile(page)
               and "VASP PAW/PBE, ENCUT 550 eV" in html
               and "(PAW spheres, LORBIT=12 projections)" in html
               and "Va point charges Va2- Va2+ Va4+ (sigma 0.5 A, "
                   "wall width 0.45 A)" in html
               and "site-resolved alignment: Sr +3.53, Ti +15.26, O +6.09 eV"
                   in html
               and "E - E_VBM (eV)" in html
               and "Sr 4s GM1+" in html and "O 2p R4+" in html, page)
        report("an unclassified crystal level gets the NEUTRAL stroke, not "
               "the bonding blue of an occupied level",
               ".seg.bond-u" in html and '"bond": "u"' in html, page)
        text = open(table).read() if os.path.isfile(table) else ""
        report("level table written: alignment header, the anchor table, the "
               "per-level shift, raw and aligned energies, shares and links",
               "# mode     site alignment, reference column mo, energy zero "
               "--vasp-zero vbm (-2.6042 eV)" in text
               and "# rigid    left +4.3884 | right +6.3998 eV" in text
               and "#      Ti  delta =  +15.255 eV" in text
               and "# residual measured alignment residual of this system: "
                   "0.5936 eV mean, 1.2479 eV max (M M4+)" in text
               and re.search(r"GM\s+left\s+left0\s+GM1\+\s+1\s+1\.000\s+"
                             r"-34\.5604\s+-33\.6347\s+\+0\.9257\s+2", text)
               is not None
               and "site shift: +0.9257 eV (site Sr)" in text
               and "shares: Sr 4s GM1+ 100.0%" in text
               and "links: O 2s GM1+ 85.9%  Sr 4s GM1+ 0.7%" in text, table)
        json_path = os.path.join(tmp, "sto_vasp_levels.json")
        document = (json.load(open(json_path))
                    if os.path.isfile(json_path) else {})
        sites = (document.get("alignment") or {}).get("sites") or {}
        report("machine-readable level JSON carries the same alignment, "
               "anchors and levels",
               document.get("alignment", {}).get("mode") == "site"
               and document["alignment"]["zero_mode"] == "vbm"
               and abs(sites["left"]["Ti"]["delta"] - 15.255) < 1e-3
               and sites["left"]["Ti"]["n_forbidden"] == 2
               and sites["left"]["Ti"]["fit"] == "forbidden"
               and sites["left"]["Ti"]["forbidden_residual"] < 0.1
               and document["alignment"]["measured_residual"][
                   "n_manifolds"] == 22
               and sites["left"]["Sr"]["ok"] is False
               and sites["right"]["O"]["ok"] is True
               and len(document["kpoints"]) == 4
               and any(level["label"] == "GM5+ #1"
                       and abs(level["energy"] - 1.7224) < 1e-3
                       for level in document["kpoints"][0]["levels"]),
               json_path)

        # --no-align
        raw_page = os.path.join(tmp, "raw.html")
        code, out = run_cli(base + ["--vasp", VASP_FIXTURE, "--no-align",
                                    "--output", raw_page], cwd=tmp)
        report("--no-align keeps the three raw VASP scales and drops the "
               "bond-character colouring",
               code == 0
               and "Deep-level alignment disabled (--no-align)" in out
               and "shifts :" not in out
               and "bond character is not classified while the three columns "
                   "sit on their own energy references" in out
               and "Sr 4s GM1+ (-34.56)" in out
               and "45 not classified" in out, out)

        # --vasp-align rigid: the previous single shift per column, now also
        # referenced to the crystal column
        rigid_page = os.path.join(tmp, "rigid.html")
        code, out = run_cli(base + ["--vasp", VASP_FIXTURE,
                                    "--vasp-align", "rigid",
                                    "--output", rigid_page], cwd=tmp)
        report("--vasp-align rigid: one shift per column, crystal-referenced, "
               "and the cation column still fails the column-scale test",
               code == 0
               and "left  delta = +4.388 eV, anchored on Sr 4s R2-" in out
               and "right delta = +6.400 eV, anchored on O 2s GM1+" in out
               and "shifts : left +1.784 eV | crystal -2.604 eV | "
                   "right +3.796 eV" in out
               and "its shells disagree by 13.53 eV" in out
               and "spread over the shells 0.20 eV <= 1 eV" in out, out)

        # --vasp-anchor (a statement about a rigid column shift)
        anchor_page = os.path.join(tmp, "anchor.html")
        code, out = run_cli(base + ["--vasp", VASP_FIXTURE,
                                    "--vasp-anchor", "Ti", "3d",
                                    "--output", anchor_page], cwd=tmp)
        report("--vasp-anchor Ti 3d selects the rigid alignment and re-anchors "
               "the cation column on Ti 3d (+15.254 eV)",
               code == 0
               and "--vasp-anchor pins one shell of a whole column" in out
               and "left  delta = +15.254 eV, anchored on Ti 3d X4+" in out,
               out)

        # --vasp-zero
        raw_zero_page = os.path.join(tmp, "zero.html")
        code, out = run_cli(base + ["--vasp", VASP_FIXTURE,
                                    "--vasp-zero", "raw",
                                    "--output", raw_zero_page], cwd=tmp)
        report("--vasp-zero raw leaves the crystal run's own G = 0 reference "
               "(every level moves by the same +2.604 eV)",
               code == 0
               and "energy zero: the crystal run's own G = 0 reference "
                   "(--vasp-zero raw)" in out
               and "Ti - Sr: +11.73 eV" in out
               and re.search(r"GM5\+ #1\s+4\.33 eV", out) is not None, out)

        # --kpoint / --vasp-window
        one_page = os.path.join(tmp, "gm.html")
        code, out = run_cli(base + ["--vasp", VASP_FIXTURE, "--kpoint", "GM",
                                    "--output", one_page], cwd=tmp)
        report("--kpoint GM restricts the diagram to one special point",
               code == 0 and out.count("\n * k point ") == 1
               and "* k point GM (0,0,0) *" in out, out)
        window_page = os.path.join(tmp, "window.html")
        code, win = run_cli(base + ["--vasp", VASP_FIXTURE,
                                    "--vasp-window", "-20", "5",
                                    "--output", window_page], cwd=tmp)
        report("--vasp-window cuts the drawn levels but not the alignment "
               "(the anchor pool is the DEFAULT window, never this one)",
               code == 0
               and "Sr  delta =   +3.530 eV" in win
               and "Ti  delta =  +15.255 eV" in win
               and "O   delta =   +6.091 eV" in win
               and 0 < win.count(" eV  x") < out.count(" eV  x") + 42, win)

        # the extended-Hueckel engine on the same POSCAR is untouched
        eht_page = os.path.join(tmp, "eht.html")
        code, out = run_cli(base + ["--kpoint", "GM", "--output", eht_page],
                            cwd=tmp)
        eht_html = open(eht_page).read() if os.path.isfile(eht_page) else ""
        report("the extended-Hueckel engine on the same POSCAR is unchanged",
               code == 0 and "VASP" not in out
               and "left&ndash;right overlap population" in eht_html
               and "Ti 3d GM5+" in eht_html, out)

    # ------------------------------------------------- CaF2: the gauge case
    # Fluorite is the structure that FIXED the PROCAR gauge convention.  Its
    # F sites sit on quarter coordinates of the primitive fcc basis, so the
    # conjugation the engine used to apply, Lambda_a = exp(2 pi i k . tau_a),
    # is not the no-op it is on every half-integer site: with it 21 of the 85
    # labelled levels fall below irrep purity 0.90 (minimum 0.000 at W), the
    # fragment manifolds stop reproducing the SALC decomposition, both sites
    # fail the per-element scale test and the page carries no bonding and no
    # antibonding level at all.  Without it every labelled level is pure and
    # the diagram is the textbook fluorite one.  SrTiO3 cannot see any of
    # this, which is why CaF2 is here.
    print("  -- --diagram --vasp CaF2 (quarter-coordinate sites: the "
          "PROCAR gauge) --")
    caf2_poscar = os.path.join(VASP_CAF2_FIXTURE, "BAND", "POSCAR")
    caf2_base = ["-c", caf2_poscar, "--diagram", "--co-left", "Ca",
                 "--co-right", "F2"]
    with tempfile.TemporaryDirectory() as tmp:
        page = os.path.join(tmp, "caf2_vasp.html")
        code, out = run_cli(caf2_base + ["--vasp", VASP_CAF2_FIXTURE,
                                         "--output", page], cwd=tmp)
        report("--diagram --vasp CaF2 exit 0", code == 0, out)
        report("CaF2 electron counts 24 = 8 + 16 and one Va species per column",
               "crystal CaF2    24 electrons = 8 + 16" in out
               and "left  Ca       charge +2, 8 electrons (NELECT), point "
                   "charges Va1- x2" in out
               and "right F2       charge -2, 16 electrons (NELECT), point "
                   "charges Va2+ x1" in out, out)
        report("the calibrated wall is what these runs used "
               "(VACSIGMA 0.5, VACRWALL 0.45)",
               out.count("VACSIGMA 0.5 A, VACRWALL 0.45 A") == 2, out)
        # the gauge test itself: with the wrong conjugation 12 levels at L,
        # 6 at W and 3 at X drop below 0.90, the minimum being 0.000
        report("every labelled level of CaF2 is a pure irrep at GM, L, X "
               "and W (the PROCAR gauge convention)",
               "Irrep purity: every labelled level is above 0.9" in out
               and "Irrep purity warnings" not in out, out)
        for name, manifolds in (("L", (("F", "2s"), ("F", "2p"),
                                       ("Ca", "3p"))),
                                ("X", (("F", "2s"), ("F", "2p"))),
                                ("W", (("F", "2p"), ("Ca", "3p")))):
            block = _vasp_kpoint_block(out, name)
            for element, shell in manifolds:
                formula = "Ca" if element == "Ca" else "F2"
                salc = _vasp_salc(block, element, shell)
                found = _vasp_found(block, formula, element, shell)
                report(f"CaF2 {name}: {element} {shell} PROCAR irreps = SALC "
                       f"decomposition {' + '.join(salc) or '(none)'}",
                       bool(salc) and found == salc,
                       f"SALC {salc} vs PROCAR {found}\n{block}")
        report("fluorite is the ionic limit: both sites pass the scale test "
               "and move together (Ca +4.54, F +4.43 eV)",
               "Ca  delta =   +4.536 eV   14 anchors (0 symmetry-forbidden), "
               "shell spread 0.47 eV -> shells agree" in out
               and "F   delta =   +4.416 eV   22 anchors "
                   "(3 symmetry-forbidden), shell spread 0.14 eV -> shells "
                   "agree" in out, out)
        report("the CaF2 page is fully classified: 30 nonbonding, "
               "6 antibonding, 5 bonding, none left neutral",
               "30 nonbonding, 6 antibonding, 5 bonding" in out
               and "not classified" not in out, out)
        # D4: the W2 triple (Ca 3p / F 2p / Ca 3d) used to come out with TWO
        # antibonding members and no bonding one, because the 4.9 % Ca 3p
        # admixture of the middle level was below the parent weight floor and
        # the level was then read against its F 2p parent alone
        caf2_json = os.path.join(tmp, "caf2_vasp_levels.json")
        caf2_doc = (json.load(open(caf2_json))
                    if os.path.isfile(caf2_json) else {})
        w2 = {level["label"]: level.get("bond_character")
              for record in caf2_doc.get("kpoints", [])
              if record["name"] == "W"
              for level in record["levels"]
              if level.get("column") == "mo" and level.get("irrep") == "W2"}
        report("the W2 Ca 3p / F 2p / Ca 3d triple is not two antibonding "
               "levels: the deep semicore parent is excluded, not the "
               "small one",
               re.search(r"W2 #2\s+-2\.13 eV\s+x1\s+2e\s+F 2p W2 93\.1%\s+"
                         r"Ca 3p W2 4\.9%", out) is not None
               and w2 == {"W2 #1": "nonbonding", "W2 #2": "nonbonding",
                          "W2 #3": "antibonding"}, f"{w2}\n{out}")
        # D3: the Ca 3d manifolds sit 0.05 eV above the default VBM + 10 eV
        # cut at GM and 0.1-1.8 eV below it at L, X and W, so the cation
        # column vanishes at ONE k point out of four
        report("fragment levels above the drawn window are named per k point",
               "* Fragment levels above the drawn window *" in out
               and re.search(r"GM\s+\d+ level\(s\) lie above the window: "
                             r".*Ca 3d GM3\+ \(\+10\.35\)", out) is not None,
               out)
        html = open(page).read() if os.path.isfile(page) else ""
        report("CaF2 page written with the Fm-3m chips and the alignment chip",
               os.path.isfile(page)
               and "Fm-3m (No. 225)" in html
               and "Va point charges Va1- Va2+ (sigma 0.5 A, "
                   "wall width 0.45 A)" in html
               and "site-resolved alignment: Ca +4.54, F +4.42 eV" in html,
               page)

    # ---------------------- SrTiO3 METAGGA = LAK, NBANDS = 64: the anchor ties
    _test_03_vasp_lak()

    # ------------------------------------------- settings and the --vasp forms
    _test_03_vasp_settings(base, poscar)

    # ------------------- one POSCAR reading rule (scale line, Cartesian form)
    _test_03_poscar_spellings()

    # ------------------------------------------------------------ error paths
    for label, extra, needle in (
        ("unknown --kpoint", ["--kpoint", "Q"],
         "ERROR: k point 'Q' is not a special point of this space group "
         "(available: GM, R, X, M)."),
        ("--vasp-left at a non-directory", ["--vasp-left", "/no/such/dir"],
         "ERROR: --vasp-left points at /no/such/dir, which is not a directory."),
        ("--vasp-anchor on a foreign element", ["--vasp-anchor", "Zr", "4d"],
         "ERROR: --vasp-anchor names Zr, which is not an element"),
    ):
        code, out = run_cli(base + ["--vasp", VASP_FIXTURE] + extra)
        report(f"{label} rejected with one ERROR line",
               code != 0 and needle in out and "Traceback" not in out, out)
    code, out = run_cli(base + ["--vasp", ROOT])
    report("a ROOT without BAND/ rejected with one ERROR line",
           code != 0 and "expected <ROOT>/BAND, or give --vasp-crystal DIR"
           in out and "Traceback" not in out, out)
    forms = ("--vasp takes no path (the current directory as ROOT), one ROOT "
             "holding BAND and BAND_sublattice1/2, or the three run "
             "directories in any order")
    for count, paths in (
        (2, [os.path.join(VASP_FIXTURE, "BAND"),
             os.path.join(VASP_FIXTURE, "BAND_sublattice1")]),
        (4, [os.path.join(VASP_FIXTURE, "BAND"),
             os.path.join(VASP_FIXTURE, "BAND_sublattice1"),
             os.path.join(VASP_FIXTURE, "BAND_sublattice2"), VASP_FIXTURE]),
    ):
        code, out = run_cli(base + ["--vasp", *paths])
        report(f"--vasp with {count} paths rejected with one ERROR line that "
               "shows the accepted forms",
               code != 0 and f"ERROR: {forms}; got {count} " in out
               and "Traceback" not in out, out)
    with tempfile.TemporaryDirectory() as tmp:
        other = os.path.join(tmp, "POSCAR_tetragonal")
        with open(other, "w") as handle:
            handle.write(
                "SrTiO3 tetragonal\n1.0\n"
                "   3.90798883012252   0.00000000000000   0.00000000000000\n"
                "   0.00000000000000   3.90798883012252   0.00000000000000\n"
                "   0.00000000000000   0.00000000000000   4.20000000000000\n"
                "Sr Ti O\n1 1 3\nDirect\n"
                "  0.0 0.0 0.0\n  0.5 0.5 0.52\n  0.5 0.0 0.5\n"
                "  0.5 0.5 0.0\n  0.0 0.5 0.5\n")
        code, out = run_cli(["-c", other, "--diagram", "--co-left", "SrTi",
                             "--co-right", "O3", "--vasp", VASP_FIXTURE],
                            cwd=tmp)
        report("a -c cell that is NOT the runs' crystal is rejected with one "
               "ERROR line naming what differs and suggesting -c BAND/POSCAR",
               code != 0 and "is not the -c cell in another setting: the "
               "run's lattice differs from the -c primitive cell by a "
               "relative" in out
               and "run it in the crystal run's own setting with -c "
               + os.path.join(VASP_FIXTURE, "BAND", "POSCAR") in out
               and "Traceback" not in out, out)
    code, out = run_cli(["-c", poscar, "--diagram", "--co-left", "Sr",
                         "--co-right", "TiO3", "--vasp", VASP_FIXTURE])
    report("sublattice runs that do not match --co-left/--co-right rejected",
           code != 0 and "do not form one of the two --co-left/--co-right "
           "sublattices" in out and "Traceback" not in out, out)
    code, out = run_cli(base + ["--vasp", VASP_FIXTURE, "--pyscf"])
    report("--vasp together with --pyscf rejected",
           code != 0 and "two different quantitative engines" in out
           and "Traceback" not in out, out)
    code, out = run_cli(base + ["--vasp", VASP_FIXTURE,
                                "--vasp-setup", VASP_FIXTURE])
    report("--vasp together with --vasp-setup rejected",
           code != 0 and "--vasp reads finished runs and --vasp-setup writes "
           "their inputs" in out and "Traceback" not in out, out)
    code, out = run_cli(["-c", POSCAR_SrTiO3, "--vasp-window", "-5", "5",
                         "--element", "O", "--orbital", "p"])
    report("--vasp-window outside --diagram rejected cleanly",
           code != 0 and "only used with --diagram" in out, out)

    # ------------------------------------------------------------ --vasp-setup
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "root")
        shutil.copytree(VASP_FIXTURE, root)
        for name in ("BAND_sublattice1", "BAND_sublattice2"):
            for leftover in glob.glob(os.path.join(root, name, "*")):
                if os.path.isfile(leftover):
                    os.remove(leftover)
        code, out = run_cli(
            ["-c", os.path.join(root, "BAND", "POSCAR"), "--diagram",
             "--co-left", "SrTi", "--co-right", "O3", "--vasp-setup", root],
            cwd=tmp)
        report("--vasp-setup exit 0 and says which POTCARs are missing "
               "(none are shipped)",
               code == 0 and "POTCAR NOT written, missing Sr, Ti" in out
               and "NOT ready to run" in out and "Traceback" not in out, out)
        left = os.path.join(root, "BAND_sublattice1", "POSCAR")
        right = os.path.join(root, "BAND_sublattice2", "POSCAR")
        reference_left = os.path.join(VASP_FIXTURE, "BAND_sublattice1",
                                      "band", "POSCAR")
        reference_right = os.path.join(VASP_FIXTURE, "BAND_sublattice2",
                                       "POSCAR")
        report("--vasp-setup regenerates both sublattice POSCARs "
               "(species lines, counts and atom order of the real runs)",
               os.path.isfile(left) and os.path.isfile(right)
               and open(left).read() == open(reference_left).read()
               and open(right).read() == open(reference_right).read(),
               left)
        kpoints = (open(os.path.join(root, "BAND_sublattice1", "KPOINTS")).read()
                   if os.path.isfile(os.path.join(root, "BAND_sublattice1",
                                                  "KPOINTS")) else "")
        report("--vasp-setup KPOINTS: weighted irreducible mesh plus the "
               "zero-weight special points",
               "6x6x6 irreducible mesh (20 points)" in kpoints
               and "Reciprocal lattice" in kpoints
               and re.search(r"0\.5000\d*\s+0\.0000\d*\s+0\.0000\d*\s+3\b",
                             kpoints) is not None
               and re.search(r"\s+0\s+! X", kpoints) is not None, kpoints)
        incar = (open(os.path.join(root, "BAND_sublattice1", "INCAR")).read()
                 if os.path.isfile(os.path.join(root, "BAND_sublattice1",
                                                "INCAR")) else "")
        report("--vasp-setup INCAR: LORBIT = 12, the calibrated Va wall, "
               "no NELECT / ICHARG",
               "LORBIT = 12" in incar and "VACSIGMA = 0.5" in incar
               and "VACRWALL = 0.45" in incar
               and "NELECT" not in incar and "ICHARG" not in incar
               and "ISTART" not in incar, incar)
        incar2 = (open(os.path.join(root, "BAND_sublattice2", "INCAR")).read()
                  if os.path.isfile(os.path.join(root, "BAND_sublattice2",
                                                 "INCAR")) else "")
        # one VACWALL per Va species, in POSCAR species order, so the charge
        # scaling of 2.5 q e^2 sqrt(2/pi)/sigma stays automatic: the cation
        # run carries only negative Va (no wall), the anion run Va2+ and Va4+
        report("--vasp-setup VACWALL: one calibrated height per Va species, "
               "scaling with the charge",
               "VACWALL = 0.000000" in incar
               and "VACWALL = 114.893678 229.787357" in incar2, incar + incar2)

    # the written POSCARs must follow the CRYSTAL RUN's setting, never the -c
    # file's: POSCAR-finish sits on the other origin and on a lattice constant
    # 3.2e-5 A away, and a fragment run in THAT cell is a different crystal
    def _cell_of(path):
        lines = [line.split() for line in open(path).read().splitlines()]
        scale = float(lines[1][0])
        lattice = [[scale * float(value) for value in row[:3]]
                   for row in lines[2:5]]
        total = sum(int(value) for value in lines[6])
        positions = [[float(value) for value in row[:3]]
                     for row in lines[8:8 + total]]
        return lattice, positions

    def _same_cell(one, other):
        return all(abs(a - b) < 1e-12 for first, second in zip(one, other)
                   for row_a, row_b in zip(first, second)
                   for a, b in zip(row_a, row_b))

    for cell_flag in (["-c", os.path.join(VASP_FIXTURE, "POSCAR-finish")], []):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "root")
            shutil.copytree(VASP_FIXTURE, root)
            for name in ("BAND_sublattice1", "BAND_sublattice2"):
                shutil.rmtree(os.path.join(root, name))
            code, out = run_cli(
                [*cell_flag, "--diagram", "--co-left", "SrTi",
                 "--co-right", "O3", "--vasp-setup", root], cwd=tmp)
            reference = _cell_of(os.path.join(root, "BAND", "POSCAR"))
            written = [os.path.join(root, name, "POSCAR") for name
                       in ("BAND_sublattice1", "BAND_sublattice2")]
            same = code == 0 and all(
                os.path.isfile(path) and _same_cell(_cell_of(path), reference)
                for path in written)
            report("--vasp-setup "
                   + ("with a -c cell on ANOTHER ORIGIN" if cell_flag
                      else "WITHOUT -c (the crystal run is the structure)")
                   + " writes the sublattice POSCARs in the crystal run's "
                     "setting (lattice and ion order of BAND/POSCAR)",
                   same, out + ("" if not written
                                else "\n" + open(written[0]).read()
                                if os.path.isfile(written[0]) else ""))
            report("--vasp-setup states the setting the POSCARs follow"
                   + (" and the shift of the -c cell" if cell_flag else ""),
                   " setting       : " in out
                   and (not cell_flag
                        or "shifted by (1/2, 1/2, 1/2)" in out), out)
    # the inputs of a FINISHED run are the record of what was run
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "root")
        shutil.copytree(VASP_FIXTURE, root)
        command = ["-c", os.path.join(root, "BAND", "POSCAR"), "--diagram",
                   "--co-left", "SrTi", "--co-right", "O3",
                   "--vasp-setup", root]
        before = open(os.path.join(root, "BAND_sublattice2", "POSCAR")).read()
        code, out = run_cli(command, cwd=tmp)
        after = open(os.path.join(root, "BAND_sublattice2", "POSCAR")).read()
        report("--vasp-setup over a directory that already holds a finished "
               "run is refused with one ERROR line that names the files",
               code != 0 and "already holds a finished run (OUTCAR" in out
               and "Give --force to overwrite" in out and after == before
               and "Traceback" not in out, out)
        code, out = run_cli([*command, "--force"], cwd=tmp)
        report("--vasp-setup --force overwrites it", code == 0
               and "wrote " in out and "Traceback" not in out, out)
    code, out = run_cli(["-c", POSCAR_SrTiO3, "--diagram", "--co-left", "Sr",
                         "--co-right", "TiO3", "--force"])
    report("--force outside --vasp-setup rejected cleanly",
           code != 0 and "--force is only used with --vasp-setup" in out, out)


# ------------------------------------------------- 3. --vasp overlap engine
# The WAVECAR-overlap engine of crystod --diagram --vasp (--vasp-engine
# overlap): crystod/vasp_wavecar.py (stage 1, all-electron PAW overlaps),
# crystod/crystal_orbital_overlap.py (stage 2 and the command-line runner),
# crystod/wavecar_irreps.py (irreps of plane-wave states) and
# crystod/_overlap_page.py (the HTML page).  Offline: the regression caches of
# example/03_hybridization/vasp_overlap; CRYSTOD_VASP_TESTDATA adds one live
# run from the WAVECARs.
def _overlap_tree_diff(a, b, path="", tol=1e-8, out=None):
    """Differences of two JSON-like trees (numbers within ``tol``)."""
    import math

    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: missing on one side")
            else:
                _overlap_tree_diff(a[k], b[k], f"{path}/{k}", tol, out)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            _overlap_tree_diff(x, y, f"{path}[{i}]", tol, out)
    elif isinstance(a, (bool, str)) or a is None or isinstance(b, (bool, str)) or b is None:
        if a != b:
            out.append(f"{path}: {a!r} != {b!r}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not (math.isfinite(a) and math.isfinite(b)) or abs(a - b) > tol:
            out.append(f"{path}: {a!r} != {b!r}")
    else:
        out.append(f"{path}: type {type(a).__name__} != {type(b).__name__}")
    return out


def _write_synthetic_wavecar(path, lattice, encut, kpts, nb, *, nspin=1, gamma_only=False,
                             spinor=False, seed=0):
    """A small WAVECAR (complex64, rtag 45200) with the reader's own G sphere."""
    import math

    from crystod.vasp_wavecar import HSQDTM
    recip = np.linalg.inv(lattice).T
    gl = []
    for k in kpts:
        gmax = math.sqrt(encut / HSQDTM) / (2 * np.pi)
        nmax = [int(math.ceil(gmax * np.linalg.norm(lattice[i]))) + 2 for i in range(3)]

        def order(n):
            return np.array(list(range(0, n + 1)) + list(range(-n, 0)))
        g3, g2, g1 = np.meshgrid(order(nmax[2]), order(nmax[1]), order(nmax[0]), indexing="ij")
        g = np.stack([g1.ravel(), g2.ravel(), g3.ravel()], axis=1)
        kc = (g + np.asarray(k)) @ recip * 2 * np.pi
        gl.append(g[HSQDTM * np.einsum("ij,ij->i", kc, kc) < encut])
    npl = [len(g) for g in gl]
    if gamma_only:
        npl = [(n + 1) // 2 for n in npl]
    if spinor:
        npl = [2 * n for n in npl]
    recl = 8 * max(max(npl), 4 + 3 * nb, 13)

    def rec(arr):
        return arr.tobytes().ljust(recl, b"\0")
    rng = np.random.default_rng(seed)
    data = []
    with open(path, "wb") as fh:
        fh.write(rec(np.array([recl, nspin, 45200], float)))
        fh.write(rec(np.array([len(kpts), nb, encut, *np.ravel(lattice), 0.0], float)))
        for _ in range(nspin):
            for ik, k in enumerate(kpts):
                eig = np.sort(rng.normal(size=nb))
                occ = np.array([1.0 if b < nb // 2 else 0.0 for b in range(nb)])
                hdr = [npl[ik], *k]
                for e, o in zip(eig, occ):
                    hdr += [e, 0.0, o]
                fh.write(rec(np.array(hdr, float)))
                coeffs = (rng.normal(size=(nb, npl[ik]))
                          + 1j * rng.normal(size=(nb, npl[ik]))).astype(np.complex64)
                for b in range(nb):
                    fh.write(rec(coeffs[b]))
                data.append((gl[ik], coeffs, eig, occ))
    return data


def _synthetic_overlap_cache(seed=1):
    """An OverlapCache with random contents (2 k points, crystal + 2 fragments)."""
    from crystod.vasp_wavecar import KPointOverlaps, OverlapCache
    rng = np.random.default_rng(seed)
    nb = [6, 5, 4]
    kps = []
    for i, name in enumerate(("GM", "R")):
        def cmat(n, m):
            return rng.normal(size=(n, m)) + 1j * rng.normal(size=(n, m))
        kps.append(KPointOverlaps(
            name=name, frac=[0.5 * i] * 3, index=3 * i, n_planewaves=100 + i,
            eigenvalues=[np.sort(rng.normal(size=n)) for n in nb],
            occupations=[rng.random(n) for n in nb],
            overlaps={m: [cmat(nb[1], nb[0]), cmat(nb[2], nb[0])] for m in ("bessel", "none")},
            rows=[np.arange(nb[1]), np.arange(nb[2])],
            fragment_overlaps={m: {(1, 2): cmat(nb[1], nb[2])} for m in ("bessel", "none")},
            sphere_keys=[[(0, 0), (1, 1)], [(0, 0)], [(1, 0), (1, 1)]],
            sphere_weights=[rng.random((2, nb[0])), rng.random((1, nb[1])),
                            rng.random((2, nb[2]))],
            pointcharge_weights=[rng.random(n) for n in nb],
            tests={"T1_crystal_max_dev": 1e-8, "T2_fragments": [{"run": "a"}, {"run": "b"}]},
            labels=[["A"] * nb[0], None, [None] + ["B"] * (nb[2] - 1)]))
    meta = {"modes": ["bessel", "none"], "trimmed": False, "runs": [{"path": "x"}] * 3,
            "paw": {}, "vbm_raw": 1.0}
    return OverlapCache(meta, kps)


def _test_03_overlap_engine() -> None:
    """The WAVECAR-overlap engine: building blocks and the regression fixtures.

    Pure-numpy checks of every building block of crystod/vasp_wavecar.py and
    crystod/crystal_orbital_overlap.py, then the four trimmed caches of
    example/03_hybridization/vasp_overlap (SrTiO3, SrGeO3, CsPbI3, CsPbI3 with
    spin-orbit coupling) against the key numbers of the published analysis
    (1e-8; the labels are CrystOD's own, from the plane waves).
    """
    print("  -- --diagram --vasp overlap engine: building blocks and fixtures --")
    import math

    from crystod import crystal_orbital_overlap as coo
    from crystod import vasp_wavecar as vw

    # ------------------------------------------------------------- Y_lm
    xg, wg = np.polynomial.legendre.leggauss(24)            # cos(theta)
    nphi = 48
    phi = 2 * np.pi * np.arange(nphi) / nphi
    ct, ph = np.meshgrid(xg, phi, indexing="ij")
    st = np.sqrt(1 - ct ** 2)
    u = np.stack([(st * np.cos(ph)).ravel(), (st * np.sin(ph)).ravel(), ct.ravel()], axis=1)
    w = (wg[:, None] * np.full(nphi, 2 * np.pi / nphi)[None, :]).ravel()
    Y = np.vstack([vw.real_ylm(l, u) for l in range(4)])
    gram = (Y * w) @ Y.T
    dev = float(np.abs(gram - np.eye(16)).max())
    report(f"real Y_lm (l = 0..3) orthonormal on the sphere (max dev {dev:.1e})", dev < 1e-12)

    # ------------------------------------------------------------- radial quadrature
    r = 1e-5 * np.exp(np.arange(801) * math.log(20.0 / 1e-5) / 800)
    si = vw.simpson_weights(r)

    def exact(a, b):
        return (-(b * b + 2 * b + 2) * math.exp(-b)) - (-(a * a + 2 * a + 2) * math.exp(-a))
    err = abs(float(si @ (r * r * np.exp(-r))) - exact(r[0], r[-1]))
    nt = 600
    si_t = vw.simpson_weights(r[:nt])
    err_t = abs(float(si_t @ (r[:nt] ** 2 * np.exp(-r[:nt]))) - exact(r[0], r[nt - 1]))
    report(f"log-grid Simpson weights integrate r^2 exp(-r) (err {err:.1e}, truncated "
           f"{err_t:.1e})", err < 1e-9 and err_t < 1e-8)

    # ------------------------------------------------------------- projector interpolation
    ds = vw.PawDataset()
    ds.psmaxn = 12.0
    argsc = vw.NPSNL / ds.psmaxn

    def cubic(q):
        return 0.3 - 1.1 * q + 0.25 * q ** 2 - 0.02 * q ** 3
    ds.pspnl = [cubic((np.arange(vw.NPSNL + 1) - 1) / argsc)]
    q = np.random.default_rng(3).uniform(0.0, ds.psmaxn * (vw.NPSNL - 3) / vw.NPSNL, 500)
    err = float(np.abs(ds.projector_ff(0, q) - cubic(q)).max())
    beyond = ds.projector_ff(0, np.array([ds.psmaxn * 0.995, ds.psmaxn * 2]))
    report(f"cubic Lagrange projector interpolation exact for a cubic (err {err:.1e}), "
           "zero beyond the table", err < 1e-12 and np.all(beyond == 0.0))

    # ------------------------------------------------------------- orthonormalisations
    rng = np.random.default_rng(7)
    n = 8
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    eps = np.sort(rng.normal(size=n)) * 3
    Gm, lam, nd = coo._isqrt_h(A @ A.conj().T, 1e-10)
    C = Gm @ A
    H = C @ (eps[:, None] * C.conj().T)
    dev_u = float(np.abs(C @ C.conj().T - np.eye(n)).max())
    dev_e = float(np.abs(np.linalg.eigvalsh(H) - eps).max())
    report(f"symmetric Loewdin: C C^H = 1 ({dev_u:.1e}), eig(H') = model levels ({dev_e:.1e})",
           dev_u < 1e-12 and dev_e < 1e-12 and nd == 0)
    occ_rows = np.array([True, False, True, True, False, False, True, False])
    Ci, info = coo._ionic_coefficients(A, occ_rows, 1e-10)
    O, Vr = np.where(occ_rows)[0], np.where(~occ_rows)[0]
    Gi, _, _ = coo._isqrt_h(A[O] @ A[O].conj().T, 1e-10)
    dev_u = float(np.abs(Ci @ Ci.conj().T - np.eye(n)).max())
    dev_o = float(np.abs(Ci[O] - Gi @ A[O]).max())
    dev_x = float(np.abs(Ci[Vr] @ Ci[O].conj().T).max())
    P_a = A[O].conj().T @ np.linalg.solve(A[O] @ A[O].conj().T, A[O])
    dev_s = float(np.abs(Ci[O].conj().T @ Ci[O] - P_a).max())
    dev_e = float(np.abs(np.linalg.eigvalsh(Ci @ (eps[:, None] * Ci.conj().T)) - eps).max())
    report(f"frozen-ion orthogonalisation: unitary ({dev_u:.1e}), occupied rows Loewdin "
           f"among themselves ({dev_o:.1e}) with the span of A_O ({dev_s:.1e}), empty rows "
           f"orthogonal to them ({dev_x:.1e}), eig(H'') = model levels ({dev_e:.1e})",
           max(dev_u, dev_o, dev_x, dev_s, dev_e) < 1e-11)

    # ------------------------------------------------------------- SMV model space
    nb_c, n_act = 12, 6
    A_full = rng.normal(size=(n_act, nb_c)) + 1j * rng.normal(size=(n_act, nb_c))
    eps_c = np.sort(rng.normal(size=nb_c)) * 4
    frozen = np.zeros(nb_c, bool)
    frozen[:4] = True
    A_m, e_m, minfo, eff = coo.build_model_space(A_full, eps_c, frozen)
    NF, X, et, q_eff = eff
    ok = (A_m.shape == (n_act, n_act) and minfo["n_extra"] == 2
          and np.array_equal(A_m[:, :4], A_full[:, :4]) and np.array_equal(e_m[:4], eps_c[:4])
          and float(np.abs(X.conj().T @ X - np.eye(2)).max()) < 1e-12
          and eps_c[4] - 1e-12 <= et.min() and et.max() <= eps_c[-1] + 1e-12
          and np.all((q_eff > 0) & (q_eff <= 1 + 1e-12)))
    report("SMV projection-only disentanglement: frozen levels kept exactly, 2 orthonormal "
           "effective levels inside the outer window", ok, str(minfo))

    # ------------------------------------------------------------- COHP sum rule
    m_act = 7
    A2 = rng.normal(size=(m_act, m_act)) + 1j * rng.normal(size=(m_act, m_act))
    e2 = np.sort(rng.normal(size=m_act)) * 2
    Gm, _, _ = coo._isqrt_h(A2 @ A2.conj().T, 1e-10)
    C2 = Gm @ A2
    H2 = C2 @ (e2[:, None] * C2.conj().T)
    meta = {"run": np.array([1, 1, 1, 2, 2, 2, 2]),
            "lid": ["l0", "l0", "l1", "r0", "r0", "r0", "r1"],
            "shell": ["Ti 3d", "Ti 3d", "Ti 4s", "O 2p", "O 2p", "O 2p", "O 2s"],
            "occupied": np.array([False, False, False, True, True, True, True]),
            "cation_runs": {1}}
    levels, compact, bands = coo._level_analysis(
        C2, H2, e2, np.where(e2 < 0, 2.0, 0.0), np.zeros(m_act, bool),
        [[i] for i in range(m_act)], meta, {}, coo.OverlapOptions())
    worst = max(abs(b["sumrule_residual"]) for b in bands)
    compl = max(abs(b["completeness"] - 1) for b in bands)
    split = max(abs(sum(b["cohp_inter_split"].values()) - b["cohp_inter"]) for b in bands)
    keys = {k for L in levels for k in L["cohp_pairs"]}
    pairs_ok = bool(keys) and all(not (k.startswith("O ") and " | Ti " in k) for k in keys) \
        and "Ti 3d | O 2p" in keys
    report(f"COHP sum rule on-site + inter + intra = e x completeness on a random model "
           f"(max {worst:.1e}; completeness {compl:.1e}; oo/oe/ee split {split:.1e}); cation "
           "shell first in the pair names", worst < 1e-12 and compl < 1e-12 and split < 1e-12
           and pairs_ok)

    # ------------------------------------------------------------- WAVECAR reader
    lattice = np.diag([3.0, 3.2, 3.4])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "WAVECAR")
        data = _write_synthetic_wavecar(path, lattice, 60.0, [(0.0, 0.0, 0.0), (0.0, 0.5, 0.25)],
                                        4)
        g, c, e, o, k = vw.read_bands(path, 1)
        ok = (np.array_equal(g, data[1][0]) and c.shape == (4, 1, len(g))
              and np.array_equal(c[:, 0, :], data[1][1].astype(complex))
              and np.allclose(e, data[1][2]) and np.allclose(k, [0.0, 0.5, 0.25]))
        report(f"read_bands: scalar WAVECAR, G sphere in WAVECAR order ({len(g)} plane "
               "waves), coefficients (nbands, 1, npw), eigenvalues, k", ok)
        data = _write_synthetic_wavecar(path, lattice, 60.0, [(0.0, 0.0, 0.0)], 2, spinor=True)
        g, c, e, o, k = vw.read_bands(tmp, 0)
        n_g = len(data[0][0])
        ok = (c.shape == (2, 2, n_g) and np.array_equal(c[:, 1, :], data[0][1][:, n_g:]))
        report("read_bands: spinor WAVECAR (vasp_ncl) split into (nbands, 2, npw), up first", ok)
        _write_synthetic_wavecar(path, lattice, 60.0, [(0.0, 0.0, 0.0)], 2, nspin=2)
        try:
            vw.read_bands(path, 0)
            msg = ""
        except SystemExit as error:
            msg = str(error)
        report("ISPIN = 2 WAVECAR refused with an ERROR", "spin-polarized" in msg, msg)
        _write_synthetic_wavecar(path, lattice, 60.0, [(0.0, 0.0, 0.0)], 2, gamma_only=True)
        try:
            vw.read_bands(path, 0)
            msg = ""
        except SystemExit as error:
            msg = str(error)
        report("gamma-only WAVECAR refused with an ERROR", "gamma-only" in msg, msg)

    # ------------------------------------------------------------- k points
    kl = np.array([[0, 0, 0], [0, -0.5, 0], [0.5, 0.5, 0.5], [1.25, 0, 0]], float)
    found = [vw.find_kpoint(kl, c)[0] for c in
             ([0, 0.5, 0], [[0.5, 0, 0], [0, 0, 0.5], [0, 0.5, 0]], [0.25, 0, 0], [0.3, 0, 0])]
    tokens = vw.parse_kpoint_tokens(["GM=0,0,0", "X=0,1/2,0"])
    cubic_rot = [np.array(m) for m in ([[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                                       [[0, 0, 1], [1, 0, 0], [0, 1, 0]])]
    star = coo.find_in_star(np.array([[0.1, 0.2, 0.3], [0.5, 0.0, 0.0]]), [0, 0.5, 0],
                            cubic_rot)
    report("k-point lookup: sign flips, candidate lists, reciprocal-lattice equivalence, "
           "NAME=k1,k2,k3 tokens; another arm of the star (X found at (1/2, 0, 0))",
           found == [1, 1, 3, None]
           and tokens == [("GM", (0.0, 0.0, 0.0)), ("X", (0.0, 0.5, 0.0))]
           and star == (1, [0.5, 0.0, 0.0]), f"{found} {star}")

    # ------------------------------------------------------------- valence and active space
    pb_cfg = [(5, 2, 2.5, -21.18, 10.0), (6, 0, 0.5, -12.04, 2.0), (6, 1, 1.5, -3.50, 2.0),
              (5, 3, 2.5, -1.0, 0.0)]
    cases = [(("Sr", "4s4p5s", 10.0), [(4, 0), (4, 1), (5, 0)]),
             (("Ti", "d3 s1", 4.0), [(3, 2), (4, 0)]),
             (("Ti", "3d3 4s1", 4.0), [(3, 2), (4, 0)]),
             (("O", "s2p4", 6.0), [(2, 0), (2, 1)]),
             (("Ge", "s2p2", 4.0), [(4, 0), (4, 1)])]
    got = [vw.valence_shells(*args) for args, _ in cases]
    got_pb = vw.valence_shells("Pb", "", 4.0, pb_cfg)
    report("POTCAR valence shells from VRHFIN (4s4p5s, d3 s1, 3d3 4s1, s2p4, s2p2) and from "
           "the atomic configuration when VRHFIN is empty (Pb: 6s 6p)",
           got == [want for _, want in cases] and got_pb == [(6, 0), (6, 1)], str(got + [got_pb]))

    def fake_cache(species, shells):
        meta = {"runs": [{"species": species}],
                "paw": {sp: {"valence_shells": [list(x) for x in sh]}
                        for sp, sh in zip(species, shells)}}
        return vw.OverlapCache(meta, [])
    rules = [
        coo.default_active_shells(fake_cache(["Sr", "Ti", "O"], [[(4, 0), (4, 1), (5, 0)],
                                                                [(3, 2), (4, 0)],
                                                                [(2, 0), (2, 1)]])),
        coo.default_active_shells(fake_cache(["Sr", "Ge", "O"], [[(4, 0), (4, 1), (5, 0)],
                                                                [(4, 0), (4, 1)],
                                                                [(2, 0), (2, 1)]])),
        coo.default_active_shells(fake_cache(["Cs", "Pb", "I"], [[(5, 0), (5, 1), (6, 0)],
                                                                [(6, 0), (6, 1)],
                                                                [(5, 0), (5, 1)]])),
        coo.default_active_shells(fake_cache(["Na", "Cl"], [[(3, 0)], [(3, 0), (3, 1)]]))]
    want = [["Sr 4s", "Sr 4p", "Sr 5s", "Sr 4d", "Ti 3d", "Ti 4s", "Ti 4p", "O 2s", "O 2p"],
            ["Sr 4s", "Sr 4p", "Sr 5s", "Sr 4d", "Ge 4s", "Ge 4p", "O 2s", "O 2p"],
            ["Cs 5s", "Cs 5p", "Cs 6s", "Cs 5d", "Pb 6s", "Pb 6p", "I 5s", "I 5p"],
            ["Na 3s", "Na 3p", "Cl 3s", "Cl 3p"]]
    report("default active space = POTCAR valence + one standard empty shell (SrTiO3, SrGeO3, "
           "CsPbI3 as published; Na: 3p in period 3)", rules == want, str(rules))
    parsed = [coo.parse_shell_tokens(t) for t in (["Sr-4s", "Ti_3d", "O2p"], ["auto"],
                                                  ["auto-full"], None)]
    errors = []
    for bad in (["Ti3x"], ["O-1p"], ["Sr-4s", "Sr-4s"], ["auto", "Ti-3d"]):
        try:
            coo.parse_shell_tokens(bad)
            errors.append("")
        except SystemExit as error:
            errors.append(str(error))
    report("--vasp-shells tokens: Sr-4s Ti_3d O2p -> 'Sr 4s' 'Ti 3d' 'O 2p', auto, auto-full; "
           "malformed, impossible, repeated and mixed tokens refused",
           parsed == [["Sr 4s", "Ti 3d", "O 2p"], None, "auto-full", None]
           and all(e.startswith("ERROR: --vasp-shells") for e in errors), f"{parsed} {errors}")

    # ------------------------------------------------------------- cache round trip
    cache = _synthetic_overlap_cache()
    with tempfile.TemporaryDirectory() as tmp:
        path = cache.save(os.path.join(tmp, "cache.npz"))
        back = vw.OverlapCache.load(path)
        same = back.meta["schema"] == vw.CACHE_SCHEMA and len(back.kpoints) == 2
        for a, b in zip(cache.kpoints, back.kpoints):
            same = same and a.name == b.name and a.index == b.index and a.labels == b.labels
            same = same and all(np.array_equal(x, y) for x, y in
                                zip(a.eigenvalues + a.occupations + a.sphere_weights
                                    + a.pointcharge_weights,
                                    b.eigenvalues + b.occupations + b.sphere_weights
                                    + b.pointcharge_weights))
            same = same and all(np.array_equal(a.overlaps[m][f], b.overlaps[m][f])
                                for m in ("bessel", "none") for f in (0, 1))
            same = same and np.array_equal(a.fragment_overlaps["none"][(1, 2)],
                                           b.fragment_overlaps["none"][(1, 2)])
            same = same and a.sphere_keys == b.sphere_keys and a.tests == b.tests
        trimmed = cache.trim({"R": [[1, 3], [0, 2]]}, kpoints=["R"], modes=["bessel"])
        t_path = trimmed.save(os.path.join(tmp, "trim.npz"))
        tb = vw.OverlapCache.load(t_path)
        kp0, kt = cache.kpoint("R"), tb.kpoint("R")
        trim_ok = (tb.trimmed and tb.modes == ["bessel"] and len(tb.kpoints) == 1
                   and np.array_equal(kt.overlaps["bessel"][0], kp0.overlaps["bessel"][0][[1, 3]])
                   and np.array_equal(kt.fragment_overlaps["bessel"][(1, 2)],
                                      kp0.fragment_overlaps["bessel"][(1, 2)][np.ix_([1, 3],
                                                                                    [0, 2])])
                   and kt.row_index(1) == {1: 0, 3: 1}
                   and np.array_equal(kt.eigenvalues[1], kp0.eigenvalues[1]))
    report("OverlapCache .npz round trip is exact (schema crystod-overlap-cache/1, JSON "
           "metadata, no pickle)", same)
    report("OverlapCache.trim keeps the selected rows, k points and modes and every "
           "eigenvalue", trim_ok)

    # ------------------------------------------------------------- fixtures
    names = sorted(d for d in os.listdir(VASP_OVERLAP_FIXTURE)
                   if os.path.isfile(os.path.join(VASP_OVERLAP_FIXTURE, d, "expected.json")))
    report("four overlap fixtures present (SrTiO3, SrGeO3, CsPbI3, CsPbI3_SOC)",
           names == ["CsPbI3", "CsPbI3_SOC", "SrGeO3", "SrTiO3"], str(names))
    results = {}
    for name in names:
        folder = os.path.join(VASP_OVERLAP_FIXTURE, name)
        with open(os.path.join(folder, "expected.json")) as fh:
            expected = json.load(fh)
        expected.pop("provenance", None)
        res = coo.analyse_cache(os.path.join(folder, "overlap_cache.npz"),
                                columns=["left", "right"], report=None)
        results[name] = res
        diffs = _overlap_tree_diff(expected, coo.key_numbers(res))
        report(f"{name}: key numbers of the trimmed cache = published analysis (1e-8; "
               f"{', '.join(expected['kpoints'])})", not diffs, "\n".join(diffs[:15]))
        checks = [kp["phase3"]["pictures"][pic]["checks"] for kp in res["kpoints"]
                  for pic in ("ionic", "lowdin")]
        unit = max(c.get("unitarity_max_dev") or 0.0 for c in checks)
        eig = max(c["H_eigs_vs_model_levels_max_dev"] for c in checks)
        rule = max(c["max_abs_sumrule_residual"] for c in checks)
        nonb = max([(x["max_abs_dev"] or 0.0) for kp in res["kpoints"]
                    for x in kp["phase3"]["pictures"]["ionic"]["checks"]
                    ["nonbonding_identities"]] or [0.0])
        report(f"{name}: unitarity {unit:.0e}, eig(H) = model levels {eig:.0e}, sum rule "
               f"{rule:.0e}, nonbonding identities (ionic) {nonb:.0e}",
               unit < 1e-12 and eig < 1e-10 and rule < 1e-10 and nonb < 1e-8)

    def edge(name, k, tag):
        p3 = next(kp for kp in results[name]["kpoints"] if kp["name"] == k)["phase3"]
        v, c = coo._edge_levels(p3)
        i = v if tag == "VBM" else c
        return p3["pictures"]["lowdin"]["levels"][i], p3["pictures"]["ionic"]["levels"][i]
    if {"SrTiO3", "SrGeO3", "CsPbI3", "CsPbI3_SOC"} <= set(results):
        low, ion = edge("SrTiO3", "GM", "CBM")
        low_r, _ = edge("SrTiO3", "R", "VBM")
        report("SrTiO3: CBM GM5+ (Ti 3d t2g) and VBM R4+ (O 2p) nonbonding, COHP = 0",
               low["irrep"] == "GM5+" and low_r["irrep"] == "R4+"
               and abs(low["cohp_inter"]) < 1e-9 and abs(low_r["cohp_inter"]) < 1e-9
               and low["bond"] == low_r["bond"] == "nonbonding")
        ledger = {Ld["label"]: Ld for kp in results["SrTiO3"]["kpoints"] if kp["name"] == "GM"
                  for Ld in kp["phase3"]["ledger"]}
        eg, t2g = ledger.get("Ti 3d GM3+", {}), ledger.get("Ti 3d GM5+", {})
        crystal = {Ld["label"]: Ld["carried_by"][0][2] for Ld in (eg, t2g) if Ld}
        split = crystal.get("Ti 3d GM3+", 0.0) - crystal.get("Ti 3d GM5+", 0.0)
        pauli = eg.get("pauli_shift", 0.0) - t2g.get("pauli_shift", 0.0)
        covalent = eg.get("covalent_shift", 0.0) - t2g.get("covalent_shift", 0.0)
        report(f"SrTiO3 GM: eg - t2g = {split:.2f} eV = Pauli {pauli:.2f} + covalent "
               f"{covalent:.2f} (+ bare d_f {eg.get('d_model', 0) - t2g.get('d_model', 0):.2f})",
               abs(split - 2.22) < 0.005 and abs(pauli - 1.91) < 0.005
               and abs(covalent - 0.28) < 0.005, json.dumps([eg, t2g])[:1500])
        low, ion = edge("SrGeO3", "GM", "CBM")
        ps = ion["parent_shift"]
        report("SrGeO3: CBM GM1+ +1.03 eV antibonding, COHP(Ge 4s | O 2s) +5.84 eV, Ge 4s "
               "parent +0.30 eV, covalent shift +0.73 eV",
               low["irrep"] == "GM1+" and low["bond"] == "antibonding"
               and abs(low["cohp_pairs"]["Ge 4s | O 2s"] - 5.84) < 0.005
               and abs(ps["parent_e"] - 0.30) < 0.005 and abs(ps["covalent_shift"] - 0.73) < 0.005
               and abs(low["e_rel_vbm"] - 1.03) < 0.005)
        low, ion = edge("CsPbI3", "R", "CBM")
        led = next(Ld for kp in results["CsPbI3"]["kpoints"] if kp["name"] == "R"
                   for Ld in kp["phase3"]["ledger"] if Ld["label"].startswith("Pb 6p"))
        report("CsPbI3: CBM R4- = Pb 6p with no Pb 6p | I 5p COHP at R; Pauli +2.09 eV >> "
               "covalent +0.01 eV",
               low["irrep"] == "R4-" and "Pb 6p | I 5p" not in low["cohp_pairs"]
               and abs(led["pauli_shift"] - 2.09) < 0.01 and abs(led["covalent_shift"]) < 0.02)
        p3 = next(kp for kp in results["CsPbI3_SOC"]["kpoints"] if kp["name"] == "R")["phase3"]
        i5s = sorted((R["irrep"], R["degeneracy"]) for R in
                     p3["pictures"]["ionic"]["parents"]["right"] if R["shell"] == "I 5s")
        low_v, _ = edge("CsPbI3_SOC", "R", "VBM")
        low_c, _ = edge("CsPbI3_SOC", "R", "CBM")
        report("CsPbI3 + SOC: VBM -R6+, CBM -R6- (+0.61 eV; Bilbao -R6, -R8); the I 5s "
               "parent of the sublattice run (-R6-/-R8-, accidentally degenerate) is named "
               "by its dimension: -R6- (2) and -R8- (4)",
               low_v["irrep"] == "-R6+" and low_c["irrep"] == "-R6-"
               and abs(low_c["e_rel_vbm"] - 0.613) < 0.005
               and i5s == [("-R6-", 2), ("-R8-", 4)], str(i5s))

    # ------------------------------------------------------------- guards and writers
    sto = os.path.join(VASP_OVERLAP_FIXTURE, "SrTiO3", "overlap_cache.npz")
    msgs = []
    for kwargs in ({"active_shells": "auto-full"},
                   {"active_shells": ["Sr 4s", "Sr 4p", "Sr 5s", "Sr 4d", "Sr 5p", "Ti 3d",
                                      "Ti 4s", "Ti 4p", "O 2s", "O 2p"]},
                   {"cross_sphere": "projector"}):
        try:
            coo.analyse_cache(sto, columns=["left", "right"], report=None, **kwargs)
            msgs.append("")
        except SystemExit as error:
            msgs.append(str(error))
    report("trimmed cache: the automatic rule, an active shell without overlap rows and a "
           "mode not stored are refused with an ERROR",
           "trimmed" in msgs[0] and "no rows" in msgs[1] and "not in the cache" in msgs[2],
           "\n".join(msgs))
    if "SrTiO3" in results:
        with tempfile.TemporaryDirectory() as tmp:
            jpath = coo.write_json(results["SrTiO3"], os.path.join(tmp, "sto.json"))
            tpath = coo.write_report(results["SrTiO3"], os.path.join(tmp, "sto.txt"))
            with open(jpath) as fh:
                text = fh.read()
            doc = json.loads(text)
            with open(tpath) as fh:
                rep = fh.read()
        report("JSON strict (no NaN), spectral_onsite layout (engine prefix read by the paper's "
               "plot_cod.py) + crystod block; report with Key results and ledger",
               "NaN" not in text and doc["engine"].startswith("spectral_onsite")
               and doc["crystod"]["active_shell_rule"] == "default"
               and "phase3" in doc["kpoints"][0]
               and "## Key results" in rep and "Ledger of the active fragment levels" in rep)


def _wi_cubic_gamma():
    """Simple cubic cell, the 19 shortest G vectors and nine Gamma states.

    Returns ``(lattice, positions, numbers, G, coeffs, energies)``; the
    states are GM1+ (1), GM4- (3), GM3+ (2), GM5+ (3) at 0, 1, 2, 3 eV.
    """
    import itertools

    lattice = 4.0 * np.eye(3)
    G = np.array([g for g in itertools.product((-1, 0, 1), repeat=3)
                  if sum(abs(x) for x in g) <= 2])
    index = {tuple(g): i for i, g in enumerate(G)}

    def wave(terms):
        c = np.zeros(len(G), dtype=complex)
        for g, value in terms:
            c[index[tuple(g)]] += value
        return c

    unit = np.eye(3, dtype=int)
    states = [wave([((0, 0, 0), 1.0)])]
    # sin(2 pi x_i) = (e^{+} - e^{-}) / 2i
    states += [wave([(unit[i], -0.5j), (-unit[i], 0.5j)]) for i in range(3)]

    def cos(i):
        return [(unit[i], 0.5), (-unit[i], 0.5)]

    states.append(wave(cos(0) + [(g, -v) for g, v in cos(1)]))
    states.append(wave([(g, 2 * v) for g, v in cos(2)] + [(g, -v) for g, v in cos(0)]
                       + [(g, -v) for g, v in cos(1)]))

    def sinsin(i, j):
        # sin a sin b = -(e^{a+b} - e^{a-b} - e^{-a+b} + e^{-a-b}) / 4
        return [(unit[i] + unit[j], -0.25), (unit[i] - unit[j], 0.25),
                (-unit[i] + unit[j], 0.25), (-unit[i] - unit[j], -0.25)]

    states += [wave(sinsin(0, 1)), wave(sinsin(1, 2)), wave(sinsin(2, 0))]
    coeffs = np.array([s / np.linalg.norm(s) for s in states])[:, None, :]
    energies = np.array([0.0, 1, 1, 1, 2, 2, 3, 3, 3])
    return lattice, [[0, 0, 0]], [1], G, coeffs, energies


def _wi_silicon_epm(k_frac, cutoff=21.0):
    """Cohen-Bergstresser silicon at k: cell, positions, G, energies (eV), coeffs."""
    import itertools

    a = 5.43
    lattice = a / 2 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    B = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=float)  # b_i, 2 pi / a
    G = np.array(list(itertools.product(range(-5, 6), repeat=3)))
    q = (G + np.asarray(k_frac, dtype=float)) @ B
    keep = np.sum(q ** 2, axis=1) < cutoff
    G, q = G[keep], q[keep]
    kinetic = (2 * np.pi / (a / 0.52917721)) ** 2 * np.sum(q ** 2, axis=1)  # Ry
    dG = (G[:, None, :] - G[None, :, :]) @ B
    g2 = np.rint(np.sum(dG ** 2, axis=2)).astype(int)
    form = np.zeros(g2.shape)
    for shell, value in ((3, -0.21), (8, 0.04), (11, 0.08)):  # Ry
        form[g2 == shell] = value
    H = np.diag(kinetic) + form * np.cos(2 * np.pi * (dG @ (np.ones(3) / 8.0)))
    energies, vectors = np.linalg.eigh(H)
    positions = np.array([[1, 1, 1], [-1, -1, -1]]) / 8.0
    return lattice, positions, G, 13.605693 * energies, vectors.T


def _wi_spinor_states():
    """Simple cubic Gamma: constant spinors, then p x spin split by L.sigma.

    Returns ``(lattice, positions, numbers, G, coeffs (8, 2, n_pw), energies)``:
    bands 0-1 constant up/down (0 eV), 2-3 j = 1/2 (1 eV), 4-7 j = 3/2 (2 eV).
    """
    import itertools

    lattice, positions, numbers, G, scalar, _ = _wi_cubic_gamma()
    p = scalar[1:4, 0, :]                                  # sin x, sin y, sin z
    eps = np.zeros((3, 3, 3))
    for i, j, k in itertools.permutations(range(3)):
        eps[i, j, k] = np.linalg.det(np.eye(3)[[i, j, k]])
    L = -1j * eps                                          # (L_k)_ij = -i eps_kij
    sigma = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]),
             np.array([[1, 0], [0, -1]])]
    LS = sum(np.kron(L[k], sigma[k]) for k in range(3))   # basis (p_i, s)
    values, vectors = np.linalg.eigh(LS)                   # -2 (j = 1/2) x2, +1 x4
    states = []
    for spin in range(2):
        c = np.zeros((2, len(G)), dtype=complex)
        c[spin] = scalar[0, 0, :]
        states.append(c)
    for v in vectors.T:
        c = np.zeros((2, len(G)), dtype=complex)
        for i in range(3):
            for s in range(2):
                c[s] += v[2 * i + s] * p[i]
        states.append(c / np.linalg.norm(c))
    energies = np.array([0.0, 0.0, 1, 1, 2, 2, 2, 2])
    return lattice, positions, numbers, G, np.array(states), energies, values


def _wi_irrep_json(levels):
    """A minimal IrRep-style JSON with ``levels = [(dim, {name: mult}), ...]`` at Gamma."""
    return {"characters and irreps": [{"subspace": {"k points": [{
        "k": {"data": [0.0, 0.0, 0.0]},
        "dimensions": {"data": [dim for dim, _ in levels]},
        "irreps": [{name: [value, 0.0] for name, value in mult.items()}
                   for _, mult in levels]}]}}]}


def _test_03_wavecar_irreps() -> None:
    """crystod.wavecar_irreps: irreps of plane-wave states (no VASP data)."""
    print("  -- --diagram --vasp overlap engine: irreps of plane-wave states "
          "(wavecar_irreps) --")
    import time

    from crystod import wavecar_irreps as wi

    # 1. symmetry-adapted plane waves at Gamma of Pm-3m
    lattice, positions, numbers, G, C, E = _wi_cubic_gamma()
    levels = wi.irreps_at_kpoint(lattice, positions, numbers, [0, 0, 0], G, C, E)
    got = [(lv["label"], lv["dim"]) for lv in levels]
    report("Pm-3m Gamma: constant, sin, cos and sin-sin waves are GM1+, GM4-, GM3+, GM5+",
           got == [("GM1+", 1), ("GM4-", 3), ("GM3+", 2), ("GM5+", 3)]
           and max(lv["quality"] for lv in levels) < 1e-10, str(got))
    labels = wi.label_bands(lattice, positions, numbers, [0, 0, 0], G, C, E)
    report("label_bands gives one label per band",
           labels == ["GM1+"] + ["GM4-"] * 3 + ["GM3+"] * 2 + ["GM5+"] * 3, str(labels))

    # 2. basis independence and a non-orthonormal (PAW-like) pseudo basis
    rng = np.random.default_rng(7)
    mixed = C.copy()
    for block in ([1, 2, 3], [4, 5], [6, 7, 8]):
        n = len(block)
        Q, _ = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
        A = Q @ np.diag(rng.uniform(0.4, 1.2, n))          # S != 1 inside the level
        mixed[block] = np.einsum("ij,jsg->isg", A.T, C[block])
    levels = wi.irreps_at_kpoint(lattice, positions, numbers, [0, 0, 0], G, mixed, E)
    report("labels survive a random mixing of partners and S != 1 (D = S^-1 M)",
           [lv["label"] for lv in levels] == ["GM1+", "GM4-", "GM3+", "GM5+"]
           and max(lv["quality"] for lv in levels) < 1e-10,
           str([(lv["label"], lv["quality"]) for lv in levels]))

    # 3. level clustering: a split multiplet, accidental degeneracies, a cut multiplet
    split = E.copy()
    split[1:4] = [1.000, 1.005, 1.008]                     # 8 meV spread > 1 meV seed
    levels = wi.irreps_at_kpoint(lattice, positions, numbers, [0, 0, 0], G, C, split)
    report("a GM4- triplet split by 8 meV is merged into one level (multiplicity rule)",
           [(lv["label"], lv["dim"]) for lv in levels][1] == ("GM4-", 3),
           str([(lv["label"], lv["bands"]) for lv in levels]))
    near = E.copy()
    near[4:6] = 1.0002                                     # GM3+ 0.2 meV above GM4-
    levels = wi.irreps_at_kpoint(lattice, positions, numbers, [0, 0, 0], G, C, near)
    report("an accidental near-degeneracy (0.2 meV) is resolved into GM4- and GM3+",
           [(lv["label"], lv["bands"]) for lv in levels][1:3]
           == [("GM4-", [1, 2, 3]), ("GM3+", [4, 5])],
           str([(lv["label"], lv["bands"]) for lv in levels]))
    blend = C.copy()
    Q, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    blend[0:4] = np.einsum("ij,jsg->isg", Q.T, C[0:4])
    exact = E.copy()
    exact[0:4] = 1.0
    levels = wi.irreps_at_kpoint(lattice, positions, numbers, [0, 0, 0], G, blend, exact)
    report("an exact degeneracy of mixed GM1+ and GM4- states is one level 'GM1+/GM4-'",
           (levels[0]["label"], levels[0]["dim"]) == ("GM1+/GM4-", 4),
           str([(lv["label"], lv["bands"]) for lv in levels]))
    levels = wi.irreps_at_kpoint(lattice, positions, numbers, [0, 0, 0], G, C[:8], E[:8])
    last = levels[-1]
    report("a multiplet cut by the top of the band list stays unlabelled (GM5+ 2/3)",
           last["label"] == "" and not last["integral"]
           and abs(last["multiplicities"].get("GM5+", 0) - 2 / 3) < 1e-3,
           str(last["multiplicities"]))

    # 4. silicon, empirical pseudopotential (non-symmorphic Fd-3m)
    expected = {"GM": ([0, 0, 0], ["GM1+", "GM5+", "GM4-", "GM2-"]),
                "X": ([0, 0.5, 0.5], ["X1", "X3", "X1"])}
    for name, (k, names) in expected.items():
        lattice_si, pos_si, G_si, E_si, C_si = _wi_silicon_epm(k)
        start = time.perf_counter()
        levels = wi.irreps_at_kpoint(lattice_si, pos_si, [14, 14], k, G_si,
                                     C_si[:10], E_si[:10])
        seconds = time.perf_counter() - start
        got = [lv["label"] for lv in levels][:len(names)]
        report(f"Si (Cohen-Bergstresser) at {name}: {', '.join(names)}, quality < 1e-8 "
               f"({seconds:.2f} s)",
               got == names and all(lv["quality"] < 1e-8 for lv in levels if lv["integral"]),
               str([(lv["label"], lv["dim"], round(lv["energy"], 3), lv["quality"])
                    for lv in levels]))
    lattice_si, pos_si, G_si, E_si, C_si = _wi_silicon_epm([0, 0, 0])
    gap = E_si[4] - E_si[3]
    report(f"Si EPM direct gap Gamma25' -> Gamma15 = {gap:.2f} eV (Cohen-Bergstresser 3.4)",
           abs(gap - 3.42) < 0.05, f"{gap}")

    # 5. spinors (double-valued irreps)
    lattice, positions, numbers, G, S_states, E_s, ls_values = _wi_spinor_states()
    levels = wi.irreps_at_kpoint(lattice, positions, numbers, [0, 0, 0], G, S_states, E_s,
                                 spinor=True)
    got = [(lv["label"], lv["dim"]) for lv in levels]
    report("spinors: constant -> -GM6+, p j=1/2 -> -GM6-, p j=3/2 -> -GM8- (L.sigma = -2, 1)",
           got == [("-GM6+", 2), ("-GM6-", 2), ("-GM8-", 4)]
           and np.allclose(ls_values, [-2, -2, 1, 1, 1, 1]), str(got))
    group = wi.little_group_data(lattice, positions, numbers, [0, 0, 0], spinor=True)
    fingerprint = group.fingerprints[group.names.index("-GM6+")]
    report("-GM6+ is the spin-1/2 representation: D x D(1/2) = GM1+ + GM4+",
           fingerprint == ("GM1+", "GM4+"), str(fingerprint))
    c4z = next(i for i, R in enumerate(group.rotations)
               if np.array_equal(R, [[0, -1, 0], [1, 0, 0], [0, 0, 1]]))
    images, phases = wi._plane_wave_maps(G, group, [c4z])
    (S_mat, M_mat), = wi._block_matrices(wi._as_coefficients(S_states, True), [[0, 1]],
                                         images, phases, group.spin_rotations[[c4z]])
    D = np.linalg.solve(S_mat, M_mat[0])
    report("4pi periodicity: D(C4z)^4 = -1 on the spin-1/2 level (chi(2 pi) = -dim)",
           np.allclose(np.linalg.matrix_power(D, 4), -np.eye(2), atol=1e-10)
           and np.allclose(np.linalg.matrix_power(D, 8), np.eye(2), atol=1e-10),
           str(np.round(np.linalg.matrix_power(D, 4), 6)))
    kramers = E_s.copy()
    kramers[2:4] = [1.0, 1.0028]                           # 2.8 meV, as in a Va run
    levels = wi.irreps_at_kpoint(lattice, positions, numbers, [0, 0, 0], G, S_states,
                                 kramers, spinor=True)
    report("a Kramers pair split by 2.8 meV is still one level (-GM6-)",
           [(lv["label"], lv["dim"]) for lv in levels][1] == ("-GM6-", 2),
           str([(lv["label"], lv["bands"]) for lv in levels]))
    single = wi.irreps_at_kpoint(lattice, positions, numbers, [0, 0, 0], G, S_states[:3],
                                 E_s[:3], spinor=True)
    report("one band of a Kramers pair alone is not a level (leakage ~ 1/2 or more)",
           single[-1]["label"] == "" and single[-1]["leakage"] > 0.4,
           str([(lv["label"], lv["leakage"]) for lv in single]))

    # 5b. spin-orbit compatibility, Gamma x D(1/2) (the scalar bridge of SOC diagrams)
    lattice, positions, numbers, G, S_states, E_s, _ = _wi_spinor_states()
    spin_levels = wi.irreps_at_kpoint(lattice, positions, numbers, [0, 0, 0], G, S_states,
                                      E_s, spinor=True)
    table = wi.spin_orbit_compatibility(lattice, positions, numbers, [0, 0, 0])
    bcs = {"-GM6+": "-GM6", "-GM7+": "-GM7", "-GM6-": "-GM8", "-GM7-": "-GM9",
           "-GM8+": "-GM10", "-GM8-": "-GM11"}
    mapped = wi.spin_orbit_compatibility(lattice, positions, numbers, [0, 0, 0],
                                         name_map=bcs)
    report("Pm-3m Gamma x D(1/2): GM4- -> -GM6- + -GM8- (the p x spin levels above), "
           "GM1+ -> -GM6+; BCS form GM3+ -> -GM10, GM4+ -> -GM6 + -GM10, "
           "GM5+ -> -GM7 + -GM10, GM4- -> -GM8 + -GM11",
           table["GM4-"] == [lv["label"] for lv in spin_levels][1:] == ["-GM6-", "-GM8-"]
           and table["GM1+"] == [spin_levels[0]["label"]] == ["-GM6+"]
           and mapped["GM1+"] == ["-GM6"]
           and mapped["GM3+"] == ["-GM10"] and mapped["GM4+"] == ["-GM6", "-GM10"]
           and mapped["GM5+"] == ["-GM7", "-GM10"] and mapped["GM4-"] == ["-GM8", "-GM11"],
           f"{table}\n{mapped}")
    a = 5.65
    fcc = a / 2 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    gaas = (fcc, [[0, 0, 0], [0.25, 0.25, 0.25]], [31, 33])
    td = wi.spin_orbit_compatibility(*gaas, [0, 0, 0])
    td_group = wi.little_group_data(*gaas, [0, 0, 0])
    vector = np.array([np.trace(fcc.T @ R @ np.linalg.inv(fcc.T)) for R in td_group.rotations])
    t2 = td_group.names[int(np.argmax(np.abs(np.conj(td_group.characters) @ vector)))]
    report("F-43m (GaAs) Gamma: Koster's Td table GM1 -> -GM6, GM2 -> -GM7, GM3 -> -GM8, "
           "GM4 (T2, x y z) -> -GM7 + -GM8, GM5 (T1) -> -GM6 + -GM8",
           t2 == "GM4" and td == {"GM1": ["-GM6"], "GM2": ["-GM7"], "GM3": ["-GM8"],
                                  "GM4": ["-GM7", "-GM8"], "GM5": ["-GM6", "-GM8"]},
           f"vector = {t2}; {td}")
    sizes = []
    for cell, k in ((tuple(_wi_cubic_gamma()[:3]), [0, 0.5, 0]),
                    (tuple(_wi_cubic_gamma()[:3]), [0.5, 0.5, 0]),
                    ((lattice_si, pos_si, [14, 14]), [0, 0.5, 0.5]),
                    ((lattice_si, pos_si, [14, 14]), [0.5, 0.5, 0.5])):
        compat = wi.spin_orbit_compatibility(*cell, k)
        scalar_group = wi.little_group_data(*cell, k)
        double = wi.little_group_data(*cell, k, spinor=True)
        dims = dict(zip(double.names, double.dims))
        sizes.append(all(sum(int(dims[d]) for d in compat[name]) == 2 * int(dim)
                         for name, dim in zip(scalar_group.names, scalar_group.dims)))
    report("Gamma x D(1/2) at X, M of Pm-3m and X, L of Fd-3m (non-symmorphic): integral, "
           "dimensions add up to 2 dim", all(sizes), str(sizes))

    # 6. IrRep JSON helpers (synthetic JSON; the irrep package is never imported)
    lattice, positions, numbers, G, C, E = _wi_cubic_gamma()
    levels = wi.irreps_at_kpoint(lattice, positions, numbers, [0, 0, 0], G, C, E)
    fake = _wi_irrep_json([(1, {"GM1+": 1.0}), (1, {"GM4-": 0.34}), (2, {"GM4-": 0.66}),
                           (2, {"GM3+": 1.0}), (3, {"GM5+": 1.0})])
    names = wi.irrep_json_labels(levels, json.loads(json.dumps(fake)), [0, 0, 0])
    mapping, conflicts = wi.irrep_json_name_map(levels, fake, [0, 0, 0])
    report("irrep_json_labels sums a multiplet IrRep splits; the name map is one-to-one",
           names == ["GM1+", "GM4-", "GM3+", "GM5+"] and not conflicts
           and mapping == {"GM1+": "GM1+", "GM4-": "GM4-", "GM3+": "GM3+", "GM5+": "GM5+"},
           f"{names} {mapping} {conflicts}")
    lattice, positions, numbers, G, S_states, E_s, _ = _wi_spinor_states()
    levels = wi.irreps_at_kpoint(lattice, positions, numbers, [0, 0, 0], G, S_states, E_s,
                                 spinor=True)
    fake = _wi_irrep_json([(2, {"-GM6": 1.0}), (2, {"-GM8": 1.0}), (4, {"-GM11": 1.0})])
    per_band = wi.label_bands(lattice, positions, numbers, [0, 0, 0], G, S_states, E_s,
                              spinor=True, irrep_json=fake)
    report("label_bands(irrep_json=...) puts IrRep's BCS names on the bands",
           per_band == ["-GM6"] * 2 + ["-GM8"] * 2 + ["-GM11"] * 4, str(per_band))
    lumped = _wi_irrep_json([(2, {"-GM6": 1.0}), (6, {"-GM8": 1.0, "-GM11": 1.0})])
    plain = wi.irrep_json_labels(levels, lumped, [0, 0, 0])
    translated = wi.bcs_labels(levels, lumped, [0, 0, 0],
                               name_map={"-GM6+": "-GM6", "-GM6-": "-GM8", "-GM8-": "-GM11"})
    report("bcs_labels names the parts of a degeneracy IrRep lumps with the name map",
           plain == ["-GM6", "", ""] and translated == ["-GM6", "-GM8", "-GM11"],
           f"{plain} {translated}")

    # 7. errors
    try:
        wi.irreps_at_kpoint(4.0 * np.eye(3), [[0, 0, 0], [0.5, 0.5, 0.5]], [1, 1], [0, 0, 0],
                            G, C, E)
        ok = False
    except ValueError as exc:
        ok = "not primitive" in str(exc)
    report("a non-primitive cell (pure translations) is refused", ok)
    try:
        wi.irreps_at_kpoint(lattice, positions, numbers, [0, 0, 0], G, C, E, spinor=True)
        ok = False
    except ValueError as exc:
        ok = "spinor" in str(exc)
    report("scalar coefficients with spinor=True are refused", ok)
    try:
        wi.irreps_at_kpoint(lattice, [[0, 0, 0], [0.5, 0, 0]], [1, 2], [0, 0, 0], G, C, E,
                            symmetry_cell=(lattice, [[0, 0, 0]], [1]))
        ok = False
    except ValueError as exc:
        ok = "symmetry_cell" in str(exc)
    report("a symmetry_cell whose operations do not fit the run is refused", ok)
    half = [i for i, g in enumerate(G) if tuple(g) >= (0, 0, 0)]
    try:
        wi.irreps_at_kpoint(lattice, positions, numbers, [0, 0, 0], G[half], C[:, :, half], E)
        ok = False
    except ValueError as exc:
        ok = "not closed" in str(exc)
    report("a G list not closed under the little group (gamma-only half sphere) is refused",
           ok)


def _overlap_page_variants(path: str) -> dict:
    """``{k name: variant}`` of the VARIANTS array embedded in a page."""
    if not os.path.isfile(path):
        return {}
    match = re.search(r"const VARIANTS = (\[.*?\]);\n", open(path).read(), re.S)
    if not match:
        return {}
    return {variant["key"].split()[0]: variant
            for variant in json.loads(match.group(1))}


def _test_03_overlap_page(results_json: str, poscar: str,
                          soc_json: str | None = None,
                          soc_poscar: str | None = None) -> None:
    """``crystod._overlap_page``: the HTML page of the overlap engine.

    Writes the SrTiO3 page (and the CsPbI3 spin-orbit page) from finished
    results, without running the analysis, and checks the level data
    embedded in it: the hybrid convention (frozen-ion parents, Loewdin-COHP
    colours, Loewdin populations mapped on the parents), the effective outer
    levels, the ledger in the parent tooltips, the covalency chip and the
    per-k note, the absent sketches, k-point selection and the energy window,
    and the double-valued labels with an overbar.
    """
    print("  -- --diagram --vasp overlap engine: the HTML page --")
    if not (os.path.isfile(results_json) and os.path.isfile(poscar)):
        report("overlap results for the page checks found", False, results_json)
        return
    bar = "̅"
    with tempfile.TemporaryDirectory() as tmp:
        page = os.path.join(tmp, "sto_overlap.html")
        windowed = os.path.join(tmp, "sto_overlap_window.html")
        code, out = run_python(
            "from crystod._overlap_page import (write_overlap_diagram_html, "
            "map_populations);"
            "from crystod.star_of_k import read_poscar_or_exit;"
            f"cell = read_poscar_or_exit({poscar!r});"
            f"write_overlap_diagram_html({results_json!r}, cell, ['SrTi'], "
            f"['O3'], {page!r});"
            f"write_overlap_diagram_html({results_json!r}, cell, ['SrTi'], "
            f"['O3'], {windowed!r}, kpoints=['R', 'GM'], window=(-5, 5));"
            # a crystal level equal to one parent of a mixed 2 x 2 block gets
            # weight 1 on it exactly (no spurious cross weight)
            "parents = [{'id': 'P1', 'column': 'left', 'irrep': 'GM3+', "
            "'composition': [['f1', 0.8], ['f2', 0.2]]}, "
            "{'id': 'P2', 'column': 'left', 'irrep': 'GM3+', "
            "'composition': [['f2', 0.8], ['f1', 0.2]]}];"
            "w, lost = map_populations({'f1': 0.8, 'f2': 0.2, 'g': 0.05}, "
            "parents);"
            "print(round(w.get('P1', 0), 6), w.get('P2', 0) < 1e-9, "
            "round(lost, 6));"
            "from crystod._overlap_page import OverlapDiagramPage, load_results, "
            "overbar_label;"
            "print(overbar_label('-R8-/-R6-') == 'R\\u03058-/R\\u03056-', "
            "overbar_label('Pb 6p -R6+#1') == 'Pb 6p R\\u03056+#1', "
            "overbar_label('O 2p GM4-#1') == 'O 2p GM4-#1');"
            f"r = load_results({results_json!r});"
            "\ntry:\n"
            "    OverlapDiagramPage(r, cell, ['Sr'], ['TiO3'])\n"
            "except SystemExit as error:\n"
            "    print(error)\n")
        lines = out.splitlines()
        report("overlap page written from finished results (exit 0)",
               code == 0 and os.path.isfile(page), out)
        report("connector mapping: a crystal level equal to one parent of a "
               "mixed block gets weight 1 on it; orbitals without a parent "
               "are reported as lost",
               code == 0 and "1.0 True 0.05" in lines, out)
        report("double-valued names drawn with an overbar over the k letters, index "
               "and parity after it (-R8-/-R6-, Pb 6p -R6+#1); single-valued ones "
               "unchanged", code == 0 and "True True True" in lines, out)
        report("--co-left / --co-right that do not match the sublattice runs "
               "stop with an ERROR naming them",
               code == 0 and any("--co-left Sr / --co-right TiO3 do not match "
                                 "the sublattice runs" in line
                                 for line in lines), out)

        variants = _overlap_page_variants(page)
        html = open(page).read() if os.path.isfile(page) else ""
        report("k points of the results in their order, keys as on the other "
               "--diagram pages",
               [variant["key"] for variant in variants.values()]
               == ["GM (0,0,0)", "X (0,1/2,0)", "M (1/2,1/2,0)",
                   "R (1/2,1/2,1/2)"], str(list(variants)))
        gm = {level["id"]: level for level in variants.get("GM", {}).get("levels", [])}
        columns = {column: sum(1 for level in gm.values() if level["col"] == column)
                   for column in ("left", "mo", "right")}
        report("GM: 9 + 5 frozen-ion parents and 12 frozen-window + 2 effective "
               "crystal levels",
               columns == {"left": 9, "mo": 14, "right": 5}, str(columns))

        def level_is(identifier, label, energy, bond=None, electrons=None):
            level = gm.get(identifier) or {}
            return (level.get("label") == label
                    and abs(level.get("e", 1e9) - energy) < 1e-4
                    and (bond is None or level.get("bond") == bond)
                    and (electrons is None or level.get("el") == electrons))

        report("GM crystal levels on E - E_VBM with the Loewdin-COHP colours "
               "(CBM t2g nonbonding, eg sigma* antibonding, O 2p bonding / "
               "antibonding with filled Sr 4p)",
               level_is("c7", "GM5+ #1", 2.2487, "n", 0)
               and level_is("c8", "GM3+ #2", 4.4724, "a", 0)
               and level_is("c4", "GM4- #2", -2.8905, "b", 6)
               and level_is("c6", "GM4- #3", -0.3571, "a", 6)
               and level_is("c5", "GM5- #1", -1.1734, "n", 6)
               and variants["GM"]["homo"] == "c6"
               and variants["GM"]["lumo"] == "c7",
               json.dumps({k: gm.get(k) for k in ("c4", "c5", "c6", "c7", "c8")})[:1500])
        report("GM frozen-ion parents with shell + irrep labels and electrons",
               level_is("Ileft2", "Ti 3d GM5+", 2.2487, None, 0)
               and level_is("Ileft3", "Ti 3d GM3+", 4.1925, None, 0)
               and level_is("Iright2", "O 2p GM4-#1", -3.0722, None, 6)
               and level_is("Ileft1", "Sr 4p GM4-", -15.0742, None, 6),
               json.dumps({k: gm.get(k, {}).get("label") for k in gm}))
        links = {level["id"]: dict(level["links"]) for level in gm.values()
                 if level["col"] == "mo"}
        report("connectors: Loewdin populations mapped on the frozen-ion "
               "parents (nonbonding t2g CBM = its Ti 3d parent exactly; eg "
               "sigma* Ti 3d 95.8% / O 2s 3.5%)",
               links.get("c7") == {"Ileft2": 1.0}
               and abs(links.get("c8", {}).get("Ileft3", 0) - 0.9579) < 1e-3
               and abs(links.get("c8", {}).get("Iright1", 0) - 0.0345) < 1e-3,
               json.dumps({k: links.get(k) for k in ("c7", "c8")}))
        report("effective outer levels: own ids, labels and the dotted style",
               level_is("eff_e0", "GM1+ (eff)", 17.5924, "a", 0)
               and level_is("eff_e1", "GM4- (eff)", 20.3992, "a", 0)
               and '[data-id^="eff_"] .seg{stroke-dasharray' in html
               and "effective outer level (dotted)" in gm.get("eff_e0", {}).get(
                   "detail", ""), gm.get("eff_e0", {}).get("detail", ""))
        detail = gm.get("Ileft3", {}).get("detail", "")
        report("parent tooltip carries the frozen-ion ledger bare d_f -> Pauli "
               "-> empty-empty mixing -> covalent -> crystal level",
               "bare d_f = <phi|H|phi>: +2.61" in detail
               and "Pauli shift +1.58 (orthogonalization +1.60, "
                   "intra-sublattice mixing -0.02) -> frozen-ion parent "
                   "+4.19" in detail
               and "empty-empty mixing +0.00 -> +4.19" in detail
               and "covalent shift +0.28 -> crystal level GM3+ #2 +4.47 "
                   "(holds 98% of the orbital)" in detail
               and "the formally empty Ti 3d shell holds 0.050 e" in detail,
               detail)
        detail = gm.get("c6", {}).get("detail", "")
        note = variants.get("GM", {}).get("note", "")
        report("crystal tooltip: populations, COHP split and shell pairs, "
               "frozen-ion parent shift; the covalency count of the k point in "
               "the note under the k buttons",
               "populations (symmetric Loewdin): O 2p 0.986" in detail
               and "antibonding: inter-sublattice COHP (symmetric Loewdin) "
                   "+0.097 eV" in detail
               and "shell pairs: Sr 4p | O 2p +0.257, Ti 4p | O 2p -0.160"
                   in detail
               and "closed-shell (filled-filled) mixing +0.10, covalent "
                   "shift -0.08 eV" in detail
               and "GM (frozen-ion picture): covalency count 0.103 e per cell"
                   in note and '<div id="knote"></div>' in html,
               note + "\n" + detail)
        report("header chips: sublattice orbitals, pictures, covalency count "
               "per k point, energy zero",
               "SrTi + O<tspan" in html
               and "(sublattice orbitals: Sr 4s 4p 4d 5s, Ti 3d 4s 4p | O 2s "
                   "2p)" in html
               and "parents: frozen-ion | colours: symmetric L&ouml;wdin COHP "
                   "| connectors: symmetric L&ouml;wdin populations" in html
               and "covalency count (frozen-ion, e / cell): GM 0.103 | X "
                   "0.837 | M 1.168 | R 1.339" in html
               and "E - E_VBM (eV)" in html, page)
        report("no hover sketch: the results carry no orbital coefficients, and the "
               "footer says so instead of promising one",
               bool(variants) and all(level.get("orb") is None
                                      for variant in variants.values()
                                      for level in variant["levels"])
               and "No wave-function sketch: the overlap engine stores no orbital "
                   "coefficients." in html
               and "real-space wave function" not in html, page)
        r_levels = {level["id"]: level for level in
                    variants.get("R", {}).get("levels", [])}
        r1 = r_levels.get("c3", {})
        report("R: VBM R4+ nonbonding, R1+ sigma bonding O 2p 79.8% / Ti 4s "
               "20.2%",
               r_levels.get("c6", {}).get("label") == "R4+ #1"
               and r_levels.get("c6", {}).get("bond") == "n"
               and abs(r_levels.get("c6", {}).get("e", 1) - 0.0) < 1e-4
               and r1.get("label") == "R1+ #1" and r1.get("bond") == "b"
               and abs(dict(r1.get("links", [])).get("Iright1", 0) - 0.798) < 1e-3
               and abs(dict(r1.get("links", [])).get("Ileft7", 0) - 0.202) < 1e-3,
               json.dumps(r1)[:1500])
        cut = _overlap_page_variants(windowed)
        cut_gm = cut.get("GM", {}).get("levels", [])
        report("kpoints= picks and orders the k points; window= drops the "
               "levels outside (-5, 5) eV on all three columns",
               list(cut) == ["R", "GM"]
               and sorted(level["label"] for level in cut_gm
                          if level["col"] == "mo")
               == ["GM3+ #2", "GM4- #2", "GM4- #3", "GM5+ #1", "GM5- #1"]
               and sum(1 for level in cut_gm if level["col"] == "left") == 2
               and sum(1 for level in cut_gm if level["col"] == "right") == 3,
               json.dumps([(level["col"], level["label"]) for level in cut_gm]))

        if not (soc_json and soc_poscar and os.path.isfile(soc_json)
                and os.path.isfile(soc_poscar)):
            report("spin-orbit overlap results for the page checks found", False,
                   str(soc_json))
            return
        soc_page = os.path.join(tmp, "cpi_soc_overlap.html")
        code, out = run_python(
            "from crystod._overlap_page import write_overlap_diagram_html;"
            "from crystod.star_of_k import read_poscar_or_exit;"
            f"cell = read_poscar_or_exit({soc_poscar!r});"
            f"write_overlap_diagram_html({soc_json!r}, cell, ['CsPb'], "
            f"['I3'], {soc_page!r}, kpoints=['R'])")
        soc = {level["id"]: level for level in
               _overlap_page_variants(soc_page).get("R", {}).get("levels", [])}
        soc_html = open(soc_page).read() if os.path.isfile(soc_page) else ""
        report("spin-orbit page: double-valued irreps with an overbar, a "
               "Kramers-degenerate set is one level drawn with one bar per Kramers "
               "pair (R-bar 6+ VBM x2, R-bar 6- CBM, R-bar 8 x4)",
               code == 0
               and soc.get("c11", {}).get("label") == f"R{bar}6+ #3"
               and soc.get("c11", {}).get("deg") == 2
               and soc.get("c11", {}).get("bars") == 1
               and soc.get("c11", {}).get("el") == 2
               and soc.get("c11", {}).get("bond") == "a"
               and soc.get("c12", {}).get("label") == f"R{bar}6- #2"
               and abs(soc.get("c12", {}).get("e", 0) - 0.6126) < 1e-4
               and soc.get("c10", {}).get("deg") == 4
               and soc.get("c10", {}).get("bars") == 2
               and soc.get("Ileft3", {}).get("label") == f"Pb 6s R{bar}6+"
               and "spin-orbit coupling: spinor states" in soc_html
               and "CsPbI3, Pm-3m (SOC)" in soc_html,
               out + json.dumps({k: (soc.get(k, {}).get("label"), soc.get(k, {}).get("bars"))
                                 for k in ("c10", "c11", "c12", "Ileft3")}))


def _test_03_overlap_cli() -> None:
    """``crystod --diagram --vasp --vasp-engine overlap`` from the command line.

    The engine on the regression caches alone (``--vasp-cache`` with no run
    directory; no WAVECAR is read): outputs, names, the key numbers of the
    fixtures, ``--kpoint``, spin-orbit runs with the scalar bridge, the HTML
    pages (:func:`_test_03_overlap_page`); then the argument errors,
    ``--vasp-setup --vasp-engine overlap`` and the anchor engine's refusal of
    spin-orbit runs.  ``CRYSTOD_VASP_TESTDATA=<dir of the VASP runs>`` adds
    one live run from the SrTiO3 WAVECARs (``SrTiO3_Pm-3m_LAK``).
    """
    print("  -- --diagram --vasp --vasp-engine overlap (command line) --")
    from crystod import crystal_orbital_overlap as coo

    def fixture(name, file_name="overlap_cache.npz"):
        return os.path.join(VASP_OVERLAP_FIXTURE, name, file_name)

    def expected_numbers(name):
        with open(fixture(name, "expected.json")) as handle:
            expected = json.load(handle)
        expected.pop("provenance", None)
        return expected

    sto = ["--diagram", "--co-left", "SrTi", "--co-right", "O3", "--vasp",
           "--vasp-engine", "overlap"]
    cpi = ["--diagram", "--co-left", "CsPb", "--co-right", "I3", "--vasp",
           "--vasp-engine", "overlap"]
    with tempfile.TemporaryDirectory() as tmp:
        code, out = run_cli(sto + ["--vasp-cache", fixture("SrTiO3")], cwd=tmp)
        stem = os.path.join(tmp, "CrystOD_SrTiO3_Pm-3m_vasp_overlap")
        report("cache-only run (no run directory: the analysis from --vasp-cache alone) "
               "exit 0, CrystOD_SrTiO3_Pm-3m_vasp_overlap.html/.json/.txt written",
               code == 0 and "no WAVECAR is read" in out
               and all(os.path.isfile(stem + extension)
                       for extension in (".html", ".json", ".txt")), out)
        results = (json.load(open(stem + ".json")) if os.path.isfile(stem + ".json")
                   else {"kpoints": [], "crystal": {}})
        diffs = (_overlap_tree_diff(expected_numbers("SrTiO3"), coo.key_numbers(results))
                 if results["kpoints"] else ["no JSON"])
        report("the JSON of the command reproduces the fixture's key numbers (1e-8)",
               not diffs, "\n".join(diffs[:15]))
        report("chosen shells printed; the terminal summary is the report's Key results "
               "(GM CBM GM5+ +2.249 nonbonding, R VBM R4+ nonbonding, covalency count)",
               "active shells (POTCAR valence + standard empty shell): Sr 4s 4p 5s 4d; "
               "Ti 3d 4s 4p; O 2s 2p" in out
               and "* Key results (eV vs the crystal VBM" in out
               and "| GM | CBM(k) | GM5+ #1 | +2.249 | Ti 3d 0.96, Sr 4d 0.04 | +0.00 (-) "
                   "| nonbonding |" in out
               and "| R | VBM(k) | R4+ #1 | -0.000 | O 2p 1.00 | +0.00 (-) | nonbonding |"
                   in out
               and "Covalency count (B; electrons in formally empty shells per cell): "
                   "GM 0.103 e" in out, out)
        text = open(stem + ".txt").read() if os.path.isfile(stem + ".txt") else ""
        report("report: settings, Key results, checks, sensitivity, ledger per k point",
               "## Key results (auto-generated)" in text and "## Checks" in text
               and "Ledger of the active fragment levels" in text, stem + ".txt")
        code, out = run_cli(sto + ["--vasp-cache", fixture("SrTiO3"), "--kpoint", "R",
                                   "--output", "sto_R.html"], cwd=tmp)
        only_r = (json.load(open(os.path.join(tmp, "sto_R.json")))
                  if os.path.isfile(os.path.join(tmp, "sto_R.json")) else {"kpoints": []})
        report("--kpoint R analyses R alone; --output names the page, the JSON and the "
               "report", code == 0 and [kp["name"] for kp in only_r["kpoints"]] == ["R"]
               and os.path.isfile(os.path.join(tmp, "sto_R.html"))
               and os.path.isfile(os.path.join(tmp, "sto_R.txt")), out)
        code, out = run_cli(sto + ["--vasp-cache", fixture("SrTiO3"), "--kpoint", "Q"],
                            cwd=tmp)
        report("an unknown --kpoint is refused with the k points of the cache",
               code != 0 and "ERROR: k point 'Q' is not in the overlap cache (available: "
               "GM, X, M, R)." in out and "Traceback" not in out, out)

        # spin-orbit coupling: the scalar run first, then the SOC run bridged to it
        code, out = run_cli(cpi + ["--vasp-cache", fixture("CsPbI3")], cwd=tmp)
        scalar = os.path.join(tmp, "CrystOD_CsPbI3_Pm-3m_vasp_overlap.json")
        report("CsPbI3 (scalar) cache-only run exit 0", code == 0
               and os.path.isfile(scalar), out)
        code, out = run_cli(cpi + ["--vasp-cache", fixture("CsPbI3_SOC"),
                                   "--vasp-scalar-reference", scalar], cwd=tmp)
        soc_json = os.path.join(tmp, "CrystOD_CsPbI3_Pm-3m_vasp_overlap_soc.json")
        report("CsPbI3 + SOC: spinor runs named *_soc, double-valued -K<n><p> labels, "
               "VBM -R6+ and CBM -R6- (+0.613 eV)",
               code == 0 and os.path.isfile(soc_json)
               and "Spin-orbit coupling: spinor (vasp_ncl) runs" in out
               and "| R | VBM(k) | -R6+ #3 | -0.000 |" in out
               and "| R | CBM(k) | -R6- #2 | +0.613 |" in out, out)
        bridge = {}
        if os.path.isfile(soc_json):
            record = json.load(open(soc_json))["kpoints"][0]
            levels = record["phase3"]["pictures"]["ionic"]["levels"]
            bridge = {level["label"]: entry["scalar_levels"][:1]
                      for level in levels
                      for entry in record["phase3"]["scalar_bridge"]["crystal"]
                      if entry["id"] == level["id"]}
        report("scalar bridge (Gamma x D(1/2)): the SOC VBM -R6+ comes from the scalar "
               "R1+ VBM, the CBM -R6- from the scalar R4- CBM (+1.77 -> +0.61 eV)",
               [row[0] for row in bridge.get("-R6+ #3", [])] == ["R1+ #2"]
               and [row[:2] for row in bridge.get("-R6- #2", [])] == [["R4- #2", 1.7655]],
               json.dumps({k: bridge.get(k) for k in ("-R6+ #3", "-R6- #2")}))
        code, out = run_cli(sto + ["--vasp-cache", fixture("SrTiO3"),
                                   "--vasp-scalar-reference", scalar], cwd=tmp)
        report("--vasp-scalar-reference on scalar runs is refused",
               code != 0 and "--vasp-scalar-reference is for spin-orbit (vasp_ncl) runs"
               in out and "Traceback" not in out, out)
        _test_03_overlap_page(stem + ".json", fixture("SrTiO3", "POSCAR"),
                              soc_json=soc_json, soc_poscar=fixture("CsPbI3_SOC", "POSCAR"))

        # ------------------------------------------------------ argument errors
        for label, args, needle in (
            ("an overlap option without --vasp-engine overlap",
             ["--diagram", "--co-left", "SrTi", "--co-right", "O3", "--vasp",
              "--vasp-shells", "Ti-3d"],
             "--vasp-shells is only used with --vasp-engine overlap."),
            ("--vasp-engine without --vasp",
             ["-c", POSCAR_SrTiO3, "--diagram", "--co-left", "SrTi", "--co-right",
              "O3", "--vasp-engine", "overlap"],
             "--vasp-engine is only used with --vasp (or --vasp-setup)."),
            ("an anchor-engine option with the overlap engine",
             sto + ["--vasp-anchor", "Ti", "3d"],
             "--vasp-anchor is only used with the anchor engine of --vasp"),
            ("a malformed --vasp-shells token",
             sto + ["--vasp-cache", fixture("SrTiO3"), "--vasp-shells", "Ti3x"],
             "ERROR: --vasp-shells takes element-shell tokens such as Sr-4s Ti-3d O-2p "
             "(or auto / auto-full), not 'Ti3x'."),
            ("--vasp-shells auto-full on a trimmed cache",
             sto + ["--vasp-cache", fixture("SrTiO3"), "--vasp-shells", "auto-full"],
             "the automatic active-space rule needs the overlaps of every fragment band"),
            ("runs without WAVECAR",
             ["-c", os.path.join(VASP_FIXTURE, "BAND", "POSCAR"), "--diagram",
              "--co-left", "SrTi", "--co-right", "O3", "--vasp", VASP_FIXTURE,
              "--vasp-engine", "overlap"],
             "repeat the run with LWAVE = .TRUE."),
        ):
            code, out = run_cli(args, cwd=tmp)
            report(f"{label} rejected with one ERROR line",
                   code != 0 and needle in out and "Traceback" not in out, out)
        code, out = run_cli(["-c", POSCAR_SrTiO3, "--vasp-cache", "x.npz", "--element",
                             "O", "--orbital", "p"], cwd=tmp)
        report("--vasp-cache outside --diagram rejected cleanly",
               code != 0 and "--vasp-cache is only used with --diagram." in out, out)

        # ------------------------------------------------------ --vasp-setup
        root = os.path.join(tmp, "root")
        shutil.copytree(VASP_FIXTURE, root)
        for name in ("BAND_sublattice1", "BAND_sublattice2"):
            shutil.rmtree(os.path.join(root, name))
        code, out = run_cli(["--diagram", "--co-left", "SrTi", "--co-right", "O3",
                             "--vasp-setup", root, "--vasp-engine", "overlap"], cwd=tmp)
        incars = [open(os.path.join(root, name, "INCAR")).read()
                  if os.path.isfile(os.path.join(root, name, "INCAR")) else ""
                  for name in ("BAND_sublattice1", "BAND_sublattice2")]
        report("--vasp-setup --vasp-engine overlap: LWAVE = .TRUE. in both sublattice "
               "INCARs, the crystal run's band reach checked (32 bands reach VBM + 14.7 "
               "eV < VBM + 17 eV: a larger NBANDS is suggested)",
               code == 0 and all("LWAVE = .TRUE." in text for text in incars)
               and "overlap engine: LWAVE = .TRUE. in every run" in out
               and "WARNING: the crystal run's highest band reaches only VBM + 14.7 eV "
                   "(NBANDS 32); repeat it with NBANDS = 40 or more" in out
               and "NOTE: " + os.path.join(root, "BAND") + " holds no WAVECAR" in out,
               out + "\n".join(incars))

        # ------------------------------------------- the anchor engine and SOC
        ncl = os.path.join(tmp, "ncl")
        shutil.copytree(VASP_FIXTURE, ncl)
        outcar = os.path.join(ncl, "BAND", "OUTCAR")
        with open(outcar) as handle:
            text = handle.read()
        with open(outcar, "w") as handle:
            handle.write(text.replace(
                "spin polarized calculation?\n",
                "spin polarized calculation?\n"
                "   LNONCOLLINEAR =      T non collinear calculations\n"))
        code, out = run_cli(["-c", os.path.join(ncl, "BAND", "POSCAR"), "--diagram",
                             "--co-left", "SrTi", "--co-right", "O3", "--vasp", ncl],
                            cwd=tmp)
        report("the anchor engine refuses a spin-orbit (LNONCOLLINEAR) run and points "
               "to --vasp-engine overlap",
               code != 0 and "is a spin-orbit (non-collinear, vasp_ncl) run; the anchor "
               "engine reads scalar runs only" in out
               and "add --vasp-engine overlap" in out and "Traceback" not in out, out)

    # ------------------------------------------------- live run (opt-in)
    data = os.environ.get("CRYSTOD_VASP_TESTDATA")
    runs = os.path.join(data or "", "SrTiO3_Pm-3m_LAK")
    if not data or not os.path.isdir(runs):
        print("  [SKIP] live overlap run from the WAVECARs (set CRYSTOD_VASP_TESTDATA to "
              "the directory holding SrTiO3_Pm-3m_LAK/BAND, BAND_sublattice1/2).")
        return
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "sto_cache.npz")
        code, out = run_cli(sto[:-3] + ["--vasp", runs, "--vasp-engine", "overlap",
                                        "--vasp-cache", cache], cwd=tmp)
        stem = os.path.join(tmp, "CrystOD_SrTiO3_Pm-3m_vasp_overlap.json")
        live = json.load(open(stem)) if os.path.isfile(stem) else {"kpoints": []}
        diffs = (_overlap_tree_diff(expected_numbers("SrTiO3"), coo.key_numbers(live))
                 if live["kpoints"] else ["no JSON"])
        report("live: SrTiO3 from the three WAVECARs (CrystOD's own irrep labels) = the "
               "fixture's key numbers (1e-8)",
               code == 0 and os.path.isfile(cache) and not diffs,
               out + "\n" + "\n".join(diffs[:15]))
        os.remove(stem)
        code, out = run_cli(sto + ["--vasp-cache", cache], cwd=tmp)
        again = json.load(open(stem)) if os.path.isfile(stem) else {"kpoints": []}
        report("live: the same analysis from the cache it wrote alone",
               code == 0 and "no WAVECAR is read" in out and not _overlap_tree_diff(
                   coo.key_numbers(live), coo.key_numbers(again)), out)


# ---------------------------------------------------------------- 4. dipole selection rules
SELECTION_RULES_DIR = os.path.join(ROOT, "example", "04_selection_rules")


def test_04_selection_rules() -> None:
    """Dipole selection rules of the crystal-orbital diagrams (crystod --diagram).

    Extended-Hueckel engine only (offline, a few seconds per run): the
    terminal block, the page data (irrep keys, per-k allowed-pair table)
    and the click handler, the band-edge results of SrTiO3 and Cu2O, the
    polarizations in the input-cell axes (rutile in the standard setting,
    with c along x, and rotated 45 degrees about x), and the group theory
    itself through the API (Ag2O, Si, parity rule, the diagram-level
    ``crystod.salc.dipole_selection_rules``).
    """
    print("\n[4] crystod --diagram  dipole selection rules")
    with tempfile.TemporaryDirectory() as tmp:
        html = os.path.join(tmp, "sto.html")
        code, out = run_cli(["-c", POSCAR_SrTiO3, "--diagram", "--co-left", "SrTi",
                             "--co-right", "O3", "--kpoint", "GM", "--output", html])
        report("SrTiO3 GM: exit 0 and the block ' * Dipole selection rules at GM (0,0,0) *' "
               "after a blank line, indented like the ' * k point GM *' block",
               code == 0 and "\n\n * Dipole selection rules at GM (0,0,0) *\n   VBM " in out
               and "\n * k point GM (0,0,0) *\n" in out, out)
        report("SrTiO3 GM: VBM O 2p GM5- -> CBM Ti 3d GM5+ allowed (x, y, z), axes of the "
               "input cell",
               re.search(r"\n   VBM GM5- #1 \(-?\d+\.\d\d eV\) -> CBM GM5\+ #1 "
                         r"\(-?\d+\.\d\d eV\): allowed \(x, y, z\)\n", out) is not None
               and "\n   (vertical transitions in the little group of k; polarizations in "
                   "the Cartesian axes x, y, z of the input cell)\n" in out, out)
        report("SrTiO3 GM: the existing crystal-level lines are unchanged",
               re.search(r"\n     GM5\+ #1 +-12\.38 eV  x3 ", out) is not None, out)
        page = open(html).read() if os.path.isfile(html) else ""
        match = re.search(r"const VARIANTS = (.*?);\n", page)
        variants = json.loads(match.group(1)) if match else []
        dip = variants[0].get("dip", {}) if variants else {}
        report("SrTiO3 page: per-k table of allowed pairs (GM4-|GM5+ 'x, y, z', the "
               "terminal's polarization text; no GM4-|GM4-) and irrep keys on the crystal "
               "levels only",
               dip.get("ok", {}).get("GM4-|GM5+") == "x, y, z"
               and "GM4-|GM4-" not in dip.get("ok", {}) and "GM5+" in dip.get("ir", [])
               and all(("irrep" in level) == (level["col"] == "mo")
                       for level in variants[0]["levels"]), str(dip))
        report("SrTiO3 page: click handler and its footer sentence",
               "function pickLevel(id)" in page and "function dipVerdict(a, b)" in page
               and "Dipole selection rules: click one crystal orbital" in page)

        # SrTiO3 X (0,1/2,0): the component along k is y in this cell
        code, out = run_cli(["-c", POSCAR_SrTiO3, "--diagram", "--co-left", "SrTi",
                             "--co-right", "O3", "--kpoint", "X",
                             "--output", os.path.join(tmp, "x.html")])
        report("SrTiO3 X: band edge forbidden, the first allowed transition follows "
               "(polarization x, z)",
               code == 0 and ": forbidden\n   first allowed: " in out
               and re.search(r"first allowed: .* dE = \d+\.\d\d eV: allowed \(x, z\)\n",
                             out) is not None, out)

        # Cu2O (cuprite, the Ag2O structure type): d -> s even-even forbidden
        cu2o = os.path.join(SELECTION_RULES_DIR, "POSCAR_Cu2O")
        html = os.path.join(tmp, "cu2o.html")
        code, out = run_cli(["-c", cu2o, "--diagram", "--co-left", "Cu", "--co-right", "O",
                             "--kpoint", "GM", "--output", html])
        page = open(html).read() if os.path.isfile(html) else ""
        match = re.search(r"const VARIANTS = (.*?);\n", page)
        dip = json.loads(match.group(1))[0].get("dip", {}) if match else {}
        report("Cu2O GM: block printed; on the page the Cu 3d GM5+ -> Cu 4s GM1+ pair is "
               "forbidden (yellow exciton series), GM5+ -> GM4- allowed",
               code == 0 and "* Dipole selection rules at GM (0,0,0) *" in out
               and {"GM1+", "GM5+", "GM4-"} <= set(dip.get("ir", []))
               and "GM1+|GM5+" not in dip.get("ok", {})
               and dip.get("ok", {}).get("GM4-|GM5+") == "x, y, z", out[-3000:])

        # polarizations in the axes of the INPUT cell: rutile (P4_2/mnm) as given,
        # with c along x, and rotated by 45 degrees about x (the reviewer's probe)
        rutile = os.path.join(ROOT, "example", "test_POSCARs", "136_PPOSCAR_TiO2")
        with open(rutile) as handle:
            rows = handle.read().splitlines()
        lattice = np.array([[float(v) for v in row.split()] for row in rows[2:5]])
        sites = np.array([[float(v) for v in row.split()[:3]] for row in rows[8:14]])

        def write_rutile(name, cell_rows, positions):
            text = "\n".join(rows[:2] + [" ".join(f"{v:.12f}" for v in r) for r in cell_rows]
                             + rows[5:8] + [" ".join(f"{v:.12f}" for v in p)
                                            for p in positions]) + "\n"
            target = os.path.join(tmp, name)
            with open(target, "w") as handle:
                handle.write(text)
            return target

        half = np.sqrt(0.5)
        turn = np.array([[1, 0, 0], [0, half, -half], [0, half, half]])
        probes = {
            "standard": rutile,
            "c along x": write_rutile("POSCAR_c_along_x",
                                      np.diag(np.diag(lattice)[[2, 0, 1]]), sites[:, [2, 0, 1]]),
            "rotated 45 about x": write_rutile("POSCAR_rotx45", lattice @ turn.T, sites),
        }
        verdicts = {}
        page_values = {}
        for label, poscar in probes.items():
            target = os.path.join(tmp, "rutile.html")
            code, out = run_cli(["-c", poscar, "--diagram", "--co-left", "Ti2",
                                 "--co-right", "O4", "--kpoint", "M",
                                 "--output", target])
            found = re.findall(r"\n   VBM .*: (allowed .*|forbidden)\n", out)
            verdicts[label] = (code, found)
            page = open(target).read() if os.path.isfile(target) else ""
            match = re.search(r"const VARIANTS = (.*?);\n", page)
            dip = json.loads(match.group(1))[0].get("dip", {}) if match else {}
            page_values[label] = sorted(set(dip.get("ok", {}).values()))
        report("rutile M band edge (polarization along c): standard 'allowed (z)', c along x "
               "'allowed (x)', rotated 45 degrees about x 'allowed (0 1 -1)' (the c axis "
               "in the input frame)",
               verdicts == {"standard": (0, ["allowed (z)"]),
                            "c along x": (0, ["allowed (x)"]),
                            "rotated 45 about x": (0, ["allowed (0 1 -1)"])}, str(verdicts))
        report("rutile M page: the click-to-click verdict data carries the terminal's "
               "polarization text (letters 'z' / 'x' in the standard and c-along-x cells, "
               "the vector '(0 1 -1)' in the rotated cell, never split into letters), and "
               "dipVerdict shows vectors verbatim",
               "z" in page_values.get("standard", [])
               and "x" in page_values.get("c along x", [])
               and "(0 1 -1)" in page_values.get("rotated 45 about x", [])
               and not any(len(v) > 1 and "," not in v and not v.startswith("(")
                           for values in page_values.values() for v in values)
               and "allowed.charAt(0) === '(' ? 'allowed ' + allowed" in page
               and "split('')" not in page, str(page_values))
        code, out = run_cli(["-c", probes["rotated 45 about x"], "--diagram", "--co-left", "Ti2",
                             "--co-right", "O4", "--kpoint", "GM",
                             "--output", os.path.join(tmp, "rutile.html")])
        report("rutile rotated 45 degrees about x, GM: the in-plane GM5- -> GM2+ edge is "
               "'allowed (1 0 0), (0 1 1)' (a plane not spanned by axes)",
               code == 0 and re.search(r"\n   VBM GM5- #\d+ .* -> CBM GM2\+ #\d+ .*: "
                                       r"allowed \(1 0 0\), \(0 1 1\)\n", out) is not None, out)

    # the group theory through the API: Ag2O / Cu2O, Si, and the parity rule
    script = (
        "from phonopy.interface.calculator import read_crystal_structure\n"
        "import numpy as np\n"
        "from crystod import salc\n"
        "from crystod.runtime_compat import get_spacegroup_irreps_from_primitive_symmetry\n"
        "from crystod.visualize_basis import SymmetryAdaptedOrbitalBasis\n"
        f"root = {os.path.join(ROOT, 'example', 'test_POSCARs')!r}\n"
        "for name in ('224_PPOSCAR_Ag2O', '227_PPOSCAR_Si', '221_PPOSCAR_SrTiO3'):\n"
        "    cell, _ = read_crystal_structure(root + '/' + name, interface_mode='vasp')\n"
        "    builder = SymmetryAdaptedOrbitalBasis(cell=cell)\n"
        "    table = salc.little_group_dipole_table(builder, [0, 0, 0])\n"
        "    print('RULE', name, table.components('GM5+', 'GM1+'),\n"
        "          table.components('GM5+', 'GM4-'), table.components('GM4-', 'GM5-'))\n"
        "    irreps, mapping = get_spacegroup_irreps_from_primitive_symmetry(\n"
        "        rotations=builder.rotations, translations=builder.translations,\n"
        "        kpoint=[0, 0, 0])\n"
        "    labels = [x.split('(')[0] for x in builder.get_irrep_labels([0, 0, 0], irreps, mapping)]\n"
        "    chi = {lab: np.trace(ir, axis1=1, axis2=2) for lab, ir in zip(labels, irreps)}\n"
        "    chi_v = np.trace(np.real(builder.rotations_cartesian[mapping]), axis1=1, axis2=2)\n"
        "    bad = []\n"
        "    for (a, b), comps in table.pairs.items():\n"
        "        n = np.real(np.sum(chi_v * np.conj(chi[b]) * chi[a])) / len(mapping)\n"
        "        if a[-1] == b[-1] and comps:\n"
        "            bad.append((a, b, 'same parity allowed'))\n"
        "        if bool(comps) != (round(n) > 0):\n"
        "            bad.append((a, b, 'character count', n))\n"
        "        if a[-1] != b[-1] and round(n) > 0 and comps != ('x', 'y', 'z'):\n"
        "            bad.append((a, b, 'cubic: all three components', comps))\n"
        "    print('PARITY', name, len(table.pairs), bad)\n"
        "    if name == '224_PPOSCAR_Ag2O':\n"
        "        tx = salc.little_group_dipole_table(builder, [0, 0.5, 0])\n"
        "        print('XRULE', tx.components('X1', 'X1'), tx.components('X1', 'X3'),\n"
        "              tx.components('X3', 'X3'))\n"
    )
    code, out = run_python(script)
    report("Ag2O (Pn-3m) GM: GM5+ -> GM1+ (d -> s, even-even) forbidden, GM5+ -> GM4- "
           "allowed, GM4- -> GM5- (odd-odd) forbidden",
           code == 0 and "RULE 224_PPOSCAR_Ag2O () ('x', 'y', 'z') ()" in out, out)
    report("Si (Fd-3m) GM: GM5+ (Gamma25') -> GM4- (Gamma15) allowed (x, y, z), the E0' "
           "transition",
           "RULE 227_PPOSCAR_Si () ('x', 'y', 'z') ()" in out, out)
    report("parity rule at GM of Ag2O, Si, SrTiO3: same parity always forbidden, opposite "
           "parity allowed exactly when the character count of V x Gf* x Gi is nonzero",
           all(f"PARITY {name} 55 []" in out for name in
               ("224_PPOSCAR_Ag2O", "227_PPOSCAR_Si", "221_PPOSCAR_SrTiO3")), out)
    report("Ag2O X (0,1/2,0), non-symmorphic 2-dim small irreps: X1|X1 along k (y), "
           "X1|X3 perpendicular (x, z), X3|X3 forbidden",
           "XRULE ('y',) ('x', 'z') ()" in out, out)

    # the diagram-level API: crystod.salc.dipole_selection_rules(diagram)
    script = (
        "from phonopy.interface.calculator import read_crystal_structure\n"
        "from crystod import salc\n"
        "from crystod.selection_rules import input_frame_rotation\n"
        f"cell, _ = read_crystal_structure({POSCAR_SrTiO3!r}, interface_mode='vasp')\n"
        "diagram = salc.CrystalOrbitalDiagram(cell, ['SrTi'], ['O3'])\n"
        "rules = salc.dipole_selection_rules(diagram)\n"
        "print('ALL', [(r.name, r.band_edge.verdict) for r in rules])\n"
        "one = salc.dipole_selection_rules(diagram, 'X')[0]\n"
        "print('ONE', one.name, one.first_allowed.verdict, one is rules[2])\n"
        "levels, _ = diagram.solve_at([0, 0, 0])\n"
        "edge = salc.band_edge_selection_rules(diagram.builder, 'GM', [0, 0, 0], levels['mo'])\n"
        "print('EDGE', edge.band_edge.verdict)\n"
        "print('FRAME', input_frame_rotation(diagram.builder).round(6).tolist())\n"
        "print('BLOCK', salc.format_dipole_selection_rules(rules[0], '(0,0,0)')[1:3])\n"
        "try:\n"
        "    salc.dipole_selection_rules(diagram, 'Q')\n"
        "except ValueError as error:\n"
        "    print('ERR', error)\n"
        "from types import SimpleNamespace\n"
        "page = SimpleNamespace(dipole_rules={key: value for key, value in\n"
        "                                     diagram.dipole_rules.items() if key[0] == 'GM'})\n"
        "print('PAGE', [r.name for r in salc.dipole_selection_rules(page)],\n"
        "      salc.dipole_selection_rules(page, 'GM')[0].name)\n"
    )
    code, out = run_python(script)
    report("API: salc.dipole_selection_rules(diagram) gives one record per special k point "
           "(GM, R allowed (x, y, z); X, M forbidden band edge)",
           code == 0 and "ALL [('GM', 'allowed (x, y, z)'), ('R', 'allowed (x, y, z)'), "
           "('X', 'forbidden'), ('M', 'forbidden')]" in out, out)
    report("API: one k point by name reuses the cached record; band_edge_selection_rules "
           "from solved levels; identity frame for a standard cell; unknown name -> ValueError",
           "ONE X allowed (x, z) True" in out and "EDGE allowed (x, y, z)" in out
           and "FRAME [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]" in out
           and "ERR k point 'Q' is not a special point of this space group" in out, out)
    report("API: format_dipole_selection_rules standalone block at column 0, two-space content",
           "BLOCK ['* Dipole selection rules at GM (0,0,0) *', '  VBM GM5- #1" in out, out)
    report("API: an object without special_kpoints/solve_at (overlap-engine page) returns "
           "the cached rules",
           "PAGE ['GM'] GM" in out, out)

    # input frame: a rigid structural map, not a point-group comparison; hexagonal k arms
    script = (
        "import contextlib, io\n"
        "from phonopy.interface.calculator import read_crystal_structure\n"
        "from crystod import salc\n"
        "from crystod.selection_rules import input_frame_rotation\n"
        "from crystod.vibration_modes import SymmetryOnlyVibrations\n"
        "from crystod.visualize_basis import SymmetryAdaptedOrbitalBasis\n"
        f"root = {os.path.join(ROOT, 'example', 'test_POSCARs')!r}\n"
        "cell, _ = read_crystal_structure(root + '/216_PPOSCAR_GaP', interface_mode='vasp')\n"
        "with contextlib.redirect_stdout(io.StringIO()):\n"
        "    own = SymmetryOnlyVibrations(cell=cell, standardize=False, input_cell=cell.copy())\n"
        "    std = SymmetryAdaptedOrbitalBasis(cell=cell)\n"
        "print('GAPOWN', input_frame_rotation(own).round(6).tolist())\n"
        "print('GAPSTD', input_frame_rotation(std).round(6).tolist())\n"
        "a = salc.little_group_dipole_table(own, [0.5, 0, 0.5])\n"
        "b = salc.little_group_dipole_table(own, [0.5, 0, 0.5], axes='standardized')\n"
        "c = salc.little_group_dipole_table(std, [0.5, 0, 0.5])\n"
        "print('GAPX', a.pairs == b.pairs, a.components('X1', 'X3'), c.components('X1', 'X3'))\n"
        "cell, _ = read_crystal_structure(root + '/186_PPOSCAR_ZnO', interface_mode='vasp')\n"
        "with contextlib.redirect_stdout(io.StringIO()):\n"
        "    zno = SymmetryAdaptedOrbitalBasis(cell=cell)\n"
        "m = salc.little_group_dipole_table(zno, [0.5, 0, 0])\n"
        "print('ZNOM', m.components('M1', 'M4'), m.components('M1', 'M3'),\n"
        "      m.components('M1', 'M1'), m.components('M1', 'M2'))\n"
    )
    code, out = run_python(script)
    report("frame: GaP built with standardize=False in its own axes gets the identity "
           "(4_x passes a point-group test but not the atom map), the standardized "
           "builder gets spglib's 4_x",
           code == 0 and "GAPOWN [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]" in out
           and "GAPSTD [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]" in out, out)
    report("frame: GaP X (1/2,0,1/2) of the fcc cell in its own axes: X1|X3 along k (y); "
           "the standardized builder's (1/2,0,1/2) is the arm along z of the input axes",
           "GAPX True ('y',) ('z',)" in out, out)
    report("ZnO (P6_3mc) M (1/2,0,0), standard setting: polarizations along the k arm "
           "(sqrt(3) 1 0) and perpendicular in-plane (1 -sqrt(3) 0) printed as vectors",
           "ZNOM ('(sqrt(3) 1 0)',) ('(1 -sqrt(3) 0)',) ('z',) ()" in out, out)


# ---------------------------------------------------------------- 5. crystod --star-of-k
def test_05_star_of_k() -> None:
    print("\n[5] crystod --star-of-k")
    code, out = run_cli(["--star-of-k", "-c", POSCAR_ScF3, "--kpoint", "0.5", "0.5", "0"])
    report("M point exit 0", code == 0, out)
    report("M point: |star of k| = 3", "|star of k| = 3" in out, out)

    code, out = run_cli(["--star-of-k", "-c", POSCAR_ScF3, "--kpoint", "0.5", "0.5", "0.5"])
    report("R point: |star of k| = 1", code == 0 and "|star of k| = 1" in out, out)

    # (0, 0.5, 0.5) is an M arm; the header should name it M, not "custom"
    code, out = run_cli(["--star-of-k", "-c", POSCAR_ScF3, "--kpoint", "0", "0.5", "0.5"])
    report("non-representative M arm labeled in header",
           code == 0 and "M [0.0, 0.5, 0.5]" in out and "custom" not in out, out)

    code, out = run_cli(["--star-of-k", "-c", "NO_SUCH_POSCAR", "--kpoint", "0", "0", "0"])
    report("missing POSCAR gives clear error (no traceback)",
           code != 0 and "No POSCAR named NO_SUCH_POSCAR!" in out
           and "Traceback" not in out, out)


# ---------------------------------------------------------------- 6. crystod --visualize
def test_06_visualize_basis() -> None:
    print("\n[6] crystod --visualize (SALC viewer)")
    with tempfile.TemporaryDirectory() as tmp_auto:
        code, out = run_cli(
            ["--visualize", "-c", POSCAR_ScF3, "--element", "F", "--orbital", "p",
             "--kpoint", "0", "0", "0"],
            cwd=tmp_auto,
        )
        report("F_p at GM exit 0", code == 0, out)
        report("decomposition matches --salc (2 GM4- + 1 GM5-)",
               "2.0 [GM4-(3)]" in out and "1.0 [GM5-(3)]" in out, out)
        report("SALC coefficients printed", "SALC basis functions" in out, out)
        report("SALC mode spaces numbered from 1",
               "Mode Space 1:" in out and "Mode Space 0:" not in out, out)
        report("HTML auto-written with default name (SALC_F_p_GM.html)",
               os.path.isfile(os.path.join(tmp_auto, "SALC_F_p_GM.html")))

        code, out = run_cli(
            ["--visualize", "-c", POSCAR_ScF3, "--element", "Sc", "--orbital", "d",
             "--kpoint", "0", "0", "0", "--real-coefficient"],
            cwd=tmp_auto,
        )
        report("Sc_d --real-coefficient exit 0", code == 0, out)
    salc_section = out.split("SALC basis functions")[-1]
    report("no imaginary coefficients remain", "j)" not in salc_section, salc_section)
    report("GM3+ realified to d_z2 / d_x2-y2",
           any("d_z2: +1.0000" in line for line in salc_section.splitlines())
           and any("d_x2-y2: +1.0000" in line for line in salc_section.splitlines()),
           salc_section)

    with tempfile.TemporaryDirectory() as tmp:
        html = os.path.join(tmp, "salc.html")
        code, out = run_cli(
            ["--visualize", "-c", POSCAR_ScF3, "--element", "F", "--orbital", "p",
             "--kpoint", "0", "0", "0", "--output", html]
        )
        report("HTML export exit 0", code == 0, out)
        exists = os.path.isfile(html)
        report("HTML file created", exists)
        if exists:
            text = open(html).read()
            report("HTML contains plotly + clickable mode table (Miranda-style viewer)",
                   "plotly" in text and "mode-table" in text and "phononwebsite" in text)
            report("HTML uses VESTA F color", "#b0b9e6" in text, text[:4000])

        html_bond = os.path.join(tmp, "salc_bond.html")
        code, out = run_cli(
            ["-c", POSCAR_ScF3, "--element", "F", "--orbital", "p", "--kpoint", "0", "0", "0",
             "--visualize", "--bond", "Sc", "F", "2.3", "--output", html_bond]
        )
        report("--bond export exit 0", code == 0, out)
        if os.path.isfile(html_bond):
            text = open(html_bond).read()
            report("bonds and polyhedra traces present (VESTA-style)",
                   "Sc-F bonds" in text and "Sc polyhedra" in text and "alphahull" in text,
                   text[:2000])
            report("bond/polyhedra display toggles present",
                   "show-bonds" in text and "show-poly" in text)
        else:
            report("--bond HTML created", False)

        html_m = os.path.join(tmp, "salc_M.html")
        code, out = run_cli(
            ["--visualize", "-c", POSCAR_ScF3, "--element", "F", "--orbital", "p",
             "--kpoint", "0.5", "0.5", "0", "--output", html_m]
        )
        report("k = M (supercell + Bloch phase) exit 0", code == 0, out)
        report("k = M HTML created", os.path.isfile(html_m))

        # extended-Hueckel eigen-levels viewer: --visualize WITHOUT
        # --element/--orbital runs the diagram engine and writes SALC-viewer
        # pages of the levels (the EHT counterpart of --visualize --pyscf)
        eht_html = os.path.join(tmp, "eht_levels_GM.html")
        code, out = run_cli(
            ["--visualize", "-c", POSCAR_ScF3, "--kpoint", "GM",
             "--output", eht_html]
        )
        report("bare --visualize (EHT levels viewer) exit 0",
               code == 0 and "Extended-Hueckel levels" in out
               and "one shared extended-Hueckel Hamiltonian" in out, out)
        eht_modes = []
        if os.path.isfile(eht_html):
            match = re.search(r"var MODES = (\[.*?\]);\n", open(eht_html).read(),
                              re.S)
            eht_modes = json.loads(match.group(1)) if match else []
        by_irrep = {}
        for mode in eht_modes:
            by_irrep.setdefault(mode["irrep"], mode)
        # the level table must be the diagram's crystal column: the empty
        # Sc-3d t2g GM5+ multiplet just above the pure-F-2p GM5- HOMO band
        report("EHT viewer levels match the diagram engine at GM",
               "GM5+ #1" in by_irrep and "GM5- #1" in by_irrep
               and by_irrep["GM5+ #1"]["el"] == 0
               and by_irrep["GM5- #1"]["el"] == 6
               and -10.0 < by_irrep["GM5+ #1"]["energy"] < -7.5
               and sum(1 for m in eht_modes
                       if m["irrep"] == "GM5+ #1") == 3,
               str([(m["irrep"], m["energy"], m["el"]) for m in eht_modes]))
        code, out = run_cli(
            ["--visualize", "-c", POSCAR_ScF3, "--sublattice", "Sc",
             "--kpoint", "GM", "--output", os.path.join(tmp, "eht_sc.html")]
        )
        report("EHT levels viewer --sublattice Sc exit 0",
               code == 0 and "Sc sublattice" in out, out)

        # k = R: the EHT engine's Bloch orbitals carry the site phase
        # (bloch_overlap gauge), the viewer applies exp(2 pi i k.T) per
        # image -- the per-AO gauge transform between the two must leave
        # the drawn field an exact Bloch state: every lattice translation
        # flips the sign at R (e^{2 pi i k.T} = -1)
        eht_r = os.path.join(tmp, "eht_levels_R.html")
        code, out = run_cli(
            ["--visualize", "-c", POSCAR_ScF3, "--kpoint", "R",
             "--output", eht_r]
        )
        report("EHT levels viewer at R exit 0", code == 0, out)
        alternation_ok = alternation_checked = 0
        if os.path.isfile(eht_r):
            page = open(eht_r).read()
            r_modes = json.loads(
                re.search(r"var MODES = (\[.*?\]);\n", page, re.S).group(1))
            r_lobes = json.loads(
                re.search(r"var LOBES = (\[.*?\]);\n", page, re.S).group(1))
            sigma = [m for m in r_modes if m["irrep"] == "R1+ #4"]
            if sigma:
                mode = sigma[0]
                entries = r_lobes[mode["start"]:mode["start"] + mode["count"]]
                sites = {tuple(np.round(x["c"], 2)): np.array(x["p"])
                         for x in entries}
                spacing = 4.07  # ScF3 a in Angstrom
                for center, poly in sites.items():
                    for axis in range(3):
                        neighbor = list(center)
                        neighbor[axis] = round(neighbor[axis] + spacing, 2)
                        partner = sites.get(tuple(neighbor))
                        if partner is None or len(partner) != len(poly):
                            continue
                        alternation_checked += 1
                        cos = float(np.dot(poly, partner) / (
                            np.linalg.norm(poly) * np.linalg.norm(partner)
                            + 1e-30))
                        if cos < -0.99:
                            alternation_ok += 1
        report("R-point lobes alternate as exact Bloch states (gauge fix)",
               alternation_checked >= 50
               and alternation_ok == alternation_checked,
               f"{alternation_ok}/{alternation_checked}")
        # the headline invocation: no --kpoint, one page per special point
        code, out = run_cli(["--visualize", "-c", POSCAR_ScF3], cwd=tmp)
        default_pages = [
            os.path.join(tmp, f"SALC_eht_221_PPOSCAR_ScF3_crystal_{k}.html")
            for k in ("GM", "R", "M", "X")
        ]
        report("bare --visualize writes one page per special k point",
               code == 0 and all(os.path.isfile(p) for p in default_pages),
               out)
        code, out = run_cli(["--visualize", "-c", POSCAR_ScF3,
                             "--chk", "nope.chk"])
        report("EHT levels viewer rejects PySCF-only flags",
               code != 0 and "needs --visualize --pyscf" in out, out)
        code, out = run_cli(["--visualize", "-c", POSCAR_ScF3,
                             "--element", "Sc"])
        report("--element without --orbital still points to the basis viewer",
               code != 0 and "omit BOTH" in out, out)
        code, out = run_cli(["--visualize", "--diagram", "-c", POSCAR_ScF3])
        report("--visualize --diagram rejected as separate runs",
               code != 0 and "separate runs" in out, out)
        code, out = run_cli(["--visualize", "-c", POSCAR_ScF3, "--element",
                             "F", "--orbital", "p", "--kpoint", "0", "0", "0",
                             "--diagonalize"])
        report("basis viewer rejects levels-viewer-only flags",
               code != 0 and "only used with the levels viewer" in out, out)

    # the all-special-points scan names every written page in ONE final block
    with tempfile.TemporaryDirectory() as tmp_scan:
        code, out = run_cli(["--visualize", "-c", POSCAR_ScF3, "--element", "Sc",
                             "--orbital", "d"], cwd=tmp_scan)
        tail = [line.strip() for line in out.split(" * Output files *")[-1].splitlines()
                if line.strip()]
        report("basis-viewer scan: one final '* Output files *' block with 4 pages",
               code == 0 and out.count(" * Output files *") == 1
               and len(tail) == 4
               and all(line.startswith("Saved 3D visualization to: SALC_221_PPOSCAR_ScF3_Sc_d_")
                       for line in tail), out)


# ---------------------------------------------------------------- 7. crystod main command extras
def test_07_main_command() -> None:
    print("\n[7] crystod main command (SALC without mode flag)")

    code, out = run_cli(["-c", POSCAR_SrTiO3, "--element", "Ti", "--orbital", "d",
                         "--kpoint", "0", "0", "0"])
    report("SALC without mode flag exit 0", code == 0, out)
    report("Ti_d at GM: GM3+ (eg) and GM5+ (t2g)",
           "GM3+(2)" in out and "GM5+(3)" in out, out)
    report("no deprecation notice for the new form", "DEPRECATED" not in out, out)

    code, out = run_cli(["--poscar", POSCAR_SrTiO3, "--element", "Ti", "--orbital", "d",
                         "--kpoint", "0", "0", "0"])
    report("--poscar alias accepted", code == 0, out)

    code, out = run_cli(["-c", POSCAR_ScF3, "--element", "F", "--orbital", "p",
                         "--kpoint", "1/2", "1/2", "0"])
    report("fractional --kpoint accepted", code == 0 and "M5+(2)" in out, out)

    # (0.5, 0, 0) is an X arm; irreptables tabulates only (0, 0.5, 0), so the
    # labels must be transported from the representative arm by conjugation
    code, out = run_cli(["-c", POSCAR_ScF3, "--element", "F", "--orbital", "p",
                         "--kpoint", "0.5", "0", "0"])
    report("SALC at non-representative X arm labeled via star mapping",
           code == 0 and "X5-(2)" in out and "irrep_" not in out, out)

    code, out = run_cli(["-c", POSCAR_SrTiO3, "--atomic-orbital", "Ti_d", "O_p",
                         "--kpoint", "0", "0", "0"])
    report("hybridization via --atomic-orbital", code == 0 and "* Result *" in out, out)

    code, out = run_cli(["--star-of-k", "-c", POSCAR_ScF3, "--kpoint", "0.5", "0.5", "0"])
    report("--star-of-k info mode: |star of k| = 3",
           code == 0 and "|star of k| = 3" in out, out)
    report("--star-of-k carries no deprecation notice", "DEPRECATED" not in out, out)

    code, out = run_cli(["--star-of-k", "-c", POSCAR_ScF3, "--kpoint", "M"])
    report("--star-of-k accepts a high-symmetry label (single-token spelling)",
           code == 0 and "|star of k| = 3" in out, out)

    with tempfile.TemporaryDirectory() as tmp:
        html = os.path.join(tmp, "SALC_vis.html")
        code, out = run_cli(["-c", POSCAR_ScF3, "--element", "F", "--orbital", "p",
                             "--kpoint", "0", "0", "0", "--visualize", "--output", html],
                            cwd=tmp)
        report("--visualize exit 0", code == 0, out)
        report("HTML visualization written", os.path.isfile(html))

        code, out = run_cli(["-c", POSCAR_ScF3, "--element", "F", "--orbital", "p",
                             "--kpoint", "0", "0", "0", "--visualize"], cwd=tmp)
        report("--visualize without --output exit 0", code == 0, out)
        report("default name SALC_F_p_GM.html auto-written",
               os.path.isfile(os.path.join(tmp, "SALC_F_p_GM.html")))

    with tempfile.TemporaryDirectory() as tmp:
        si_poscar = os.path.join(PHONON_VECTOR_DIR, "227_PPOSCAR_Si")
        code, out = run_cli(["-c", si_poscar, "--element", "Si", "--orbital", "p",
                             "--kpoint", "0", "0", "0", "--visualize", "--conventional"],
                            cwd=tmp)
        report("--visualize --conventional exit 0", code == 0, out)
        conv_html = os.path.join(tmp, "SALC_Si_p_GM_conv.html")
        report("conventional-cell HTML written with _conv suffix", os.path.isfile(conv_html))
        if os.path.isfile(conv_html):
            report("conventional display cell noted in the sidebar",
                   "conventional (F centring)" in open(conv_html).read())

        # --conventional sidebar: the k point in BOTH bases (--kpoint is
        # always the primitive basis; the conventional re-expression says
        # why the display supercell has its shape -- X of fcc Si:
        # primitive [0,1/2,1/2] = conventional [1,0,0])
        code, out = run_cli(["-c", si_poscar, "--element", "Si", "--orbital", "p",
                             "--kpoint", "0", "0.5", "0.5", "--visualize",
                             "--conventional"], cwd=tmp)
        x_html = os.path.join(tmp, "SALC_Si_p_X_conv.html")
        x_text = open(x_html).read() if os.path.isfile(x_html) else ""
        report("--conventional sidebar shows the k point in both bases "
               "(fcc X: [0,1/2,1/2] prim = [1,0,0] conv)",
               code == 0
               and "X [0.0, 0.5, 0.5] (primitive)" in x_text
               and "X [1.0, 0.0, 0.0] (conventional)" in x_text, out)
        code, out = run_cli(["-c", si_poscar, "--element", "Si", "--orbital", "p",
                             "--kpoint", "0", "0.5", "0.5", "--visualize"], cwd=tmp)
        plain_x = open(os.path.join(tmp, "SALC_Si_p_X.html")).read()
        report("without --conventional the k-point row stays single "
               "(primitive, untagged)",
               code == 0 and "X [0.0, 0.5, 0.5]</td>" in plain_x
               and "(primitive)" not in plain_x
               and "(conventional)" not in plain_x, out)

        code, out = run_cli(["-c", si_poscar, "--element", "Si", "--orbital", "p",
                             "--kpoint", "0", "0", "0", "--conventional"], cwd=tmp)
        report("--conventional without --visualize rejected cleanly",
               code != 0 and "--visualize" in out and "Traceback" not in out, out)

    from crystod import __version__ as crystod_version

    code, out = run_cli(["--version"])
    report("--version prints the package version",
           code == 0 and f"CrystOD {crystod_version}" in out, out)

    # error handling
    code, out = run_cli([])
    report("no-args guidance names the sectioned commands",
           code != 0 and "crystod-phonon" in out and "Traceback" not in out, out)
    code, out = run_cli(["-c", POSCAR_SrTiO3, "--element", "Ti"])
    report("--element without --orbital rejected cleanly",
           code != 0 and "requires both" in out and "Traceback" not in out, out)
    code, out = run_cli(["--star-of-k", "-c", POSCAR_ScF3])
    report("--star-of-k without --kpoint rejected cleanly",
           code != 0 and "requires --kpoint" in out and "Traceback" not in out, out)
    # since v0.4.0 a label is resolved through the structure in SALC mode too
    code, out = run_cli(["-c", POSCAR_SrTiO3, "--element", "Ti", "--orbital", "d",
                         "--kpoint", "GM"])
    report("k-point label GM accepted in SALC mode (resolved to 0 0 0)",
           code == 0 and "GM [0.0, 0.0, 0.0]" in out and "GM3+(2)" in out, out)
    code, out = run_cli(["-c", POSCAR_SrTiO3, "--element", "Ti", "--orbital", "d",
                         "--atomic-orbital", "O_p"])
    report("--element combined with --atomic-orbital rejected cleanly",
           code != 0 and "--atomic-orbital alone" in out and "Traceback" not in out, out)

    # removed flat forms give replacement guidance
    for flag, replacement in (("--salc", "crystod -c"), ("--visualize-basis", "crystod --visualize")):
        code, out = run_cli([flag])
        report(f"removed {flag} flag points to the new main command",
               code != 0 and "is not a crystod option" in out and replacement in out, out)

    # No stale version history anywhere a user or a contributor reads: every
    # shipped doc page (changelog.md excepted -- it records history by
    # definition), the README, and every CLI module, whose docstrings used to
    # annotate each mode with the flat spelling it replaced.
    code, help_out = run_cli(["--help"])
    docs = [path for path in glob.glob(os.path.join(ROOT, "doc", "*.md"))
            if os.path.basename(path) != "changelog.md"]
    sources = glob.glob(os.path.join(ROOT, "crystod", "cli", "*.py"))
    # "the former ``crystod --x``" is archaeology; a bare "the former"
    # ("the former --poscar spelling is kept as an alias") is not
    archaeology = ("v0.3.0", "flat mode", "flat flag", "flat command",
                   "the former ``crystod", "the former `crystod",
                   "(old `", "(old ``")
    stale = {}
    for path in docs + sources + [os.path.join(ROOT, "README.md")]:
        if not os.path.isfile(path):
            continue
        with open(path) as handle:
            text = handle.read()
        # MOVED_MODE_FLAGS itself is the mapping that answers those
        # spellings; it is data, not a notice about a removal
        if os.path.basename(path) == "main.py":
            text = text.split("MOVED_MODE_FLAGS = {")[0] + \
                text.split("}\n\ndesc =")[-1]
        hits = [word for word in archaeology if word in text]
        if hits:
            stale[os.path.relpath(path, ROOT)] = hits
    report("no removed-flat-mode archaeology in --help, the docs or the "
           "CLI docstrings",
           code == 0 and "v0.3.0" not in help_out and not stale,
           f"stale: {stale}\n{help_out}")

    # --help enumerates the choices of the multi-valued options, and the
    # lists match what PySCF actually ships (they are hard-coded so that
    # --help stays import-free: this check is where drift would show up).
    # The comparison itself needs PySCF; the --help text does not, so only
    # the drift check and the coverage guards below are skipped without it.
    try:
        import pyscf  # noqa: F401
        has_pyscf = True
    except ImportError:
        has_pyscf = False

    def _alias(names):
        return {name.replace("-", "").lower() for name in names}

    report("--help lists the --basis/--pseudo/--xc/--orbital choices",
           all(token in help_out for token in (
               "gth-dzvp-molopt-sr", "gth-qzv3p", "gth-cc-qzvp",
               "gth-pade", "gth-hfrev", "gth-hcth120",
               "r2scan", "hse06", "b97-d",
               "s, p, d, f, g")), help_out)
    # --example: bundled inputs (spec 3.2.2). The files land in the working
    # directory, so every run gets a fresh one
    code, out = run_cli(["--example"])
    report("--example alone lists the bundled examples",
           code == 0 and "ScF3_d" in out and "SrTiO3_d" in out
           and "--example NAME" in out, out)
    code, out = run_cli(["--example", "nope"])
    report("--example with an unknown name is refused with the available names",
           code != 0 and "ERROR:" in out and "available:" in out
           and "Traceback" not in out, out)
    with tempfile.TemporaryDirectory() as tmp:
        code, out = run_cli(["--example", "ScF3_d"], cwd=tmp)
        report("--example ScF3_d copies the cell, announces the command and runs it",
               code == 0 and os.path.isfile(os.path.join(tmp, "221_PPOSCAR_ScF3"))
               and "Running: crystod -c 221_PPOSCAR_ScF3 --element Sc --orbital d" in out
               and "GM3+(2)" in out and "GM5+(3)" in out and "R5+(3)" in out, out)
        # the copied file is identical to the bundled one, so a second run in
        # the same directory keeps it; further options are passed on
        code, out = run_cli(["--example", "ScF3_d", "--kpoint", "0", "0", "0"], cwd=tmp)
        report("--example twice in one directory keeps the identical file, extra options pass on",
               code == 0 and "Kept 221_PPOSCAR_ScF3" in out
               and "--orbital d --kpoint 0 0 0" in out and "GM3+(2)" in out
               and "R5+(3)" not in out, out)
    with tempfile.TemporaryDirectory() as tmp:
        fake = os.path.join(tmp, "221_PPOSCAR_ScF3")
        with open(fake, "w") as handle:
            handle.write("not the bundled file\n")
        code, out = run_cli(["--example", "ScF3_d"], cwd=tmp)
        report("--example never overwrites a differing file of the same name",
               code != 0 and "already exists" in out and "Traceback" not in out
               and open(fake).read() == "not the bundled file\n", out)

    # --kpoint accepts a high-symmetry label in the SALC and hybridization
    # modes too (the documented `--atomic-orbital Sc-d F-p --kpoint R` form)
    code_label, out_label = run_cli(["-c", POSCAR_ScF3, "--atomic-orbital", "Sc-d", "F-p",
                                     "--kpoint", "R"])
    code_coord, out_coord = run_cli(["-c", POSCAR_ScF3, "--atomic-orbital", "Sc-d", "F-p",
                                     "--kpoint", "0.5", "0.5", "0.5"])
    report("--kpoint R accepted in hybridization mode", code_label == 0, out_label)
    report("--kpoint R prints the same report as --kpoint 0.5 0.5 0.5",
           code_label == 0 and code_coord == 0 and out_label == out_coord, out_label)
    code, out = run_cli(["-c", POSCAR_SrTiO3, "--element", "Ti", "--orbital", "d",
                         "--kpoint", "X"])
    report("--kpoint X accepted in SALC mode", code == 0 and " X [" in out, out)
    code, out = run_cli(["-c", POSCAR_ScF3, "--element", "Sc", "--orbital", "d",
                         "--kpoint", "Q9"])
    report("unknown k label is rejected with the list of labels",
           code != 0 and "Available labels" in out, out)
    # the labels are the ISO-IR names the survey prints, resolved in the spglib
    # primitive basis -- on a base-centred monoclinic cell seekpath's names and
    # basis differ from them, and the seekpath-based modes transform the basis
    vo2 = os.path.join(ROOT, "example", "test_POSCARs", "12_PPOSCAR_VO2")
    code, out = run_cli(["-c", vo2, "--element", "V", "--orbital", "d", "--kpoint", "Y"])
    report("--kpoint Y on C2/m VO2 is the ISO-IR Y point",
           code == 0 and "Y [-0.5, 0.5, 0.0]" in out, out)
    code, out = run_cli(["--star-of-k", "-c", vo2, "--kpoint", "C"])
    report("--star-of-k --kpoint C on C2/m VO2: seekpath coordinates in the spglib "
           "basis, no basis warning",
           code == 0 and "C [-0.355135, 0.355135, 0.0]" in out
           and "does not match" not in out, out)
    bct = os.path.join(ROOT, "example", "test_POSCARs", "139_PPOSCAR_Sr3Ti2O7")
    code, out = run_module("crystod.cli.phonon", ["--vibration", "-c", bct, "--qpoint", "G"])
    report("--qpoint G on a body-centred tetragonal cell is the Gamma alias (the ISO-IR "
           "list names no point G)",
           code == 0 and "* Selected Q point *\n  GM       (0, 0, 0)" in out, out)

    if not has_pyscf:
        print("  [SKIP] pyscf not installed: the --basis/--pseudo drift "
              "check and the coverage guards are skipped.")
        return
    from crystod.crystal_orbital_pyscf import (GTH_BASIS_SETS,
                                               GTH_PSEUDOPOTENTIALS)
    from pyscf.pbc.gto import basis as pbc_basis
    from pyscf.pbc.gto import pseudo as pbc_pseudo

    report("the documented GTH basis/pseudopotential lists are complete "
           "and correct against PySCF",
           _alias(GTH_BASIS_SETS) == set(pbc_basis.ALIAS)
           and _alias(GTH_PSEUDOPOTENTIALS) == set(pbc_pseudo.ALIAS)
           and all(name in help_out for name in GTH_BASIS_SETS)
           and all(name in help_out for name in GTH_PSEUDOPOTENTIALS),
           f"basis diff: {_alias(GTH_BASIS_SETS) ^ set(pbc_basis.ALIAS)}\n"
           f"pseudo diff: "
           f"{_alias(GTH_PSEUDOPOTENTIALS) ^ set(pbc_pseudo.ALIAS)}")

    # an element the chosen basis does not cover: name the sets that do,
    # instead of PySCF's BasisNotFoundError from inside cell.build()
    code, out = run_cli(["-c", POSCAR_ScF3, "--diagram", "--pyscf",
                         "--co-left", "Sc", "--co-right", "F3",
                         "--basis", "gth-qzv2p", "--kmesh", "1", "1", "1"])
    report("a basis without an entry for one element is refused with the "
           "sets that cover the structure",
           code != 0 and "has no entry for Sc" in out
           and "gth-dzvp-molopt-sr" in out and "Traceback" not in out, out)

    # --xc is checked up front too: libxc raises a bare KeyError for a name
    # it does not know, and PySCF's periodic code has no VV10 nonlocal
    # correlation, so wb97m-v used to die inside get_veff after the cells
    # were built ("KNumInt has no attribute nr_nlc_vxc")
    for functional, marker in (("pz", "is not a functional libxc knows"),
                               ("wb97m-v", "VV10 nonlocal correlation")):
        code, out = run_cli(["-c", POSCAR_ScF3, "--diagram", "--pyscf",
                             "--co-left", "Sc", "--co-right", "F3",
                             "--xc", functional, "--kmesh", "1", "1", "1",
                             "--ke-cutoff", "80"])
        report(f"--xc {functional} is refused with one sentence, not a "
               "libxc/PySCF traceback",
               code != 0 and marker in out and "Traceback" not in out, out)

    # the lanthanides are in NO GTH basis PySCF ships, so --pyscf cannot run
    # on a rare-earth compound at all -- say that, and say it in --help
    code, out = run_cli(["-c", os.path.join(ROOT, "example", "test_POSCARs",
                                            "150_PPOSCAR_Ce2O3"),
                         "--diagram", "--pyscf", "--co-left", "Ce2",
                         "--co-right", "O3", "--kmesh", "1", "1", "1"])
    report("a rare-earth structure says no GTH basis covers it "
           "(--help warns about La-Lu too)",
           code != 0 and "has no entry for Ce" in out
           and "No GTH basis PySCF ships covers" in out
           and "Traceback" not in out
           and "La-Lu" in help_out, out)

    # --chk-info on a checkpoint written by the current code (its params hold
    # no electron count) and on an early-format file that still has one
    try:
        import pyscf  # noqa: F401
        has_pyscf = True
    except ImportError:
        has_pyscf = False
    if not has_pyscf:
        print("  [SKIP] pyscf not installed: --chk-info / --visualize --pyscf checks skipped.")
        return
    with tempfile.TemporaryDirectory() as tmp:
        chk_file = os.path.join(tmp, "nacl.chk")
        code, out = run_cli(["-c", POSCAR_NaCl, "--diagram", "--pyscf", "--onsite",
                             "--co-left", "Na", "--co-right", "Cl", "--kpoint", "X",
                             "--kmesh", "1", "1", "1", "--ke-cutoff", "80",
                             "--chk", chk_file, "--output", os.path.join(tmp, "n.html")],
                            cwd=tmp)
        info_code, info_out = run_cli(["--chk-info", chk_file], cwd=tmp)
        report("--chk-info on a fresh --pyscf checkpoint: exit 0, conditions and the "
               "reuse string, no KeyError traceback",
               code == 0 and info_code == 0 and "oxidation : Cl=-1 Na=+1" in info_out
               and "reuse with: --co-left Na --co-right Cl" in info_out
               and "Traceback" not in info_out, out[-1500:] + info_out)
        old_file = os.path.join(tmp, "old.chk")
        code, out = run_python(
            "import json, numpy as np\n"
            f"data = dict(np.load({chk_file!r}, allow_pickle=False))\n"
            "params = json.loads(str(data['params']))\n"
            "params['electrons'] = 16.0\n"
            "params.pop('conv_tol'); params.pop('max_cycle')\n"
            "data['params'] = json.dumps(params)\n"
            f"with open({old_file!r}, 'wb') as handle:\n"
            "    np.savez_compressed(handle, **data)\n")
        info_code, info_out = run_cli(["--chk-info", old_file], cwd=tmp)
        report("--chk-info on an early-format checkpoint (electron count stored) still "
               "prints it",
               code == 0 and info_code == 0
               and "electrons : 16 per cell (oxidation Cl=-1 Na=+1)" in info_out
               and "Traceback" not in info_out, out + info_out)

        # --visualize --pyscf: the "SCF saved to" notice is listed under
        # "* Output files *", not inside "* PySCF levels *"
        code, out = run_cli(["-c", POSCAR_NaCl, "--visualize", "--pyscf", "--kpoint", "X",
                             "--kmesh", "1", "1", "1", "--ke-cutoff", "80",
                             "--chk", os.path.join(tmp, "vis.chk")], cwd=tmp)
        files_block = out.split("* Output files *")[-1]
        report("--visualize --pyscf prints 'SCF saved to vis.chk' under * Output files *",
               code == 0 and "* Output files *" in out
               and "  SCF saved to " in files_block
               and out.count("SCF saved to") == 1, out[-2000:])



# ---------------------------------------------------------------- 8. crystod-group --product
def test_08_direct_product() -> None:
    print("\n[8] crystod-group --product")
    code, out = run_group(
        ["--point-group", "m-3m", "--product", "T2g", "T2g"]
    )
    report("T2g x T2g in m-3m exit 0", code == 0, out)
    report("T2g x T2g = A1g + Eg + T1g + T2g",
           all(f"({name})" in out for name in ("A1g", "Eg", "T1g", "T2g")), out)

    code, out = run_group(
        ["--point-group", "m-3m", "--product", "T2g", "T2g", "T1u"]
    )
    report("T2g x T2g x T1u exit 0", code == 0, out)
    report("triple product contains A2u (Raman/IR selection logic)", "(A2u)" in out, out)

    code, out = run_group(["--point-group", "m-3m", "--product", "T2g", "T2g",
                           "--show-irrep-table"])
    report("--product --show-irrep-table: blank line first, one * Point group * block",
           code == 0 and out.startswith("\n* Point group *\n")
           and out.count("* Point group *") == 1 and "* IrRep Table *" in out, out)

    code, out = run_group(["--table", "--point-group", "3m"])
    report("character table of 3m exit 0", code == 0, out)
    report("table lists A1 and E", "A1" in out and "E" in out, out)

    # ---- space-group little-group character tables (--table --sg --kpoint)
    code, out = run_group(["--table", "--space-group", "Pm-3m", "--kpoint", "0", "0", "0"])
    report("little-group table at GM of Pm-3m exit 0", code == 0, out)
    report("GM table lists GM1+ and GM5- with Seitz headers",
           "GM1+(1)" in out and "GM5-(3)" in out and "3^+_111" in out, out)

    code, out = run_group(["--table", "--sg", "Pm-3m", "--kpoint", "0", "0", "0.1"])
    report("little-group table on the DT line labeled via ISO-IR",
           code == 0 and "DT [0.0, 0.0, 0.1]" in out
           and "DT1(1)" in out and "DT5(2)" in out
           and "little group: P4mm (99)" in out, out)

    code, out = run_group(["--table", "--sg", "Pm-3m"])
    report("--table with --sg but no --kpoint rejected cleanly",
           code != 0 and "requires --kpoint" in out, out)

    # ---- space-group irrep products (--sg; validated against Bilbao DIRPRO)
    code, out = run_group(["--product", "R4-", "R5+", "--sg", "Pm-3m"])
    report("R4- x R5+ in Pm-3m exit 0", code == 0, out)
    report("R4- x R5+ = GM2- + GM3- + GM4- + GM5-",
           "R4- x R5+ = GM2- + GM3- + GM4- + GM5-" in out, out)
    report("dimension check printed (9 = 9)", "3 x 3 = 9 -> 1 + 2 + 3 + 3 = 9" in out, out)
    report("Bilbao DIRPRO citation printed", "Acta Cryst. A62" in out, out)

    code, out = run_group(["--product", "X5+", "X5+", "--space-group", "Pm-3m"])
    report("X5+ x X5+ multi-arm star product with multiplicities",
           code == 0 and "GM1+ + GM2+ + 2GM3+ + GM4+ + GM5+" in out
           and "2M5+" in out, out)

    code, out = run_group(["--product", "X1-", "W4", "--sg", "Fm-3m"])
    report("X1- x W4 lands on the DT line (ISO-IR labels)",
           code == 0 and "DT1" in out and "DT2" in out and "W1" in out
           and "non-tabulated" in out, out)

    code, out = run_group(["--product", "P1", "PA1", "--sg", "I-43m"])
    report("P1 x PA1 = GM1 (synthesized -k star of a polar group)",
           code == 0 and "P1 x PA1 = GM1" in out, out)

    # the (1/6,1/6,1/2) line of P6_3/mmc is ISO-IR "Q" (CDML called it "S")
    code, out = run_group(["--product", "K5", "M2+", "H1", "--sg", "P6_3/mmc"])
    report("triple space-group product K5 x M2+ x H1 (dims 48 = 48)",
           code == 0 and "2L1 + 2L2 + 2Q1" in out and "= 48" in out, out)

    code, out = run_group(["--product", "H1", "P1", "--sg", "230"])
    report("space group by number; asymmetric P star resolved (SG230)",
           code == 0 and "H1 x P1 = P1 + P2" in out, out)

    # conjugate-family P/PA pair of I-42d (non-self-conjugate small irreps)
    code, out = run_group(["--product", "P1", "X1", "--sg", "122"])
    report("conjugate-family P of I-42d resolved (P1 x X1, SG122)",
           code == 0 and all(f"LD{i}" in out for i in (1, 2, 3, 4))
           and "2 x 4 = 8 -> 2 + 2 + 2 + 2 = 8" in out, out)
    code, out = run_group(["--product", "P1", "PA1", "--sg", "122"])
    report("P1 x PA1 (I-42d) = GM1 + ... (conjugate-pair naming)",
           code == 0 and "P1 x PA1 = GM1 + GM4 + GM5" in out, out)
    code, out = run_group(["--product", "M1", "P1", "--sg", "122"])
    report("M1 x P1 (I-42d) lands on the synthesized PA star",
           code == 0 and "M1 x P1 = PA1" in out and "1 x 2 = 2 -> 2 = 2" in out,
           out)

    # ---- line terms: ISO-IR (ISOTROPY) labels
    code, out = run_group(["--product", "N1", "P1", "--sg", "I4_132"])
    report("N1 x P1 (I4_132) lands on the DT line with ISO-IR labels",
           code == 0 and all(f"DT{i}" in out for i in (1, 2, 3, 4))
           and "[non-tabulated]" in out
           and "6 x 4 = 24 -> 6 + 6 + 6 + 6 = 24" in out, out)

    # the two stars k and -k of a polar line: the one at the positive line
    # parameter keeps the ISO-IR name, the other one carries ISOTROPY's
    # partner name with the conjugate irreps (LE1 = conj LD1; it was LDA1)
    code, out = run_group(["--product", "P1", "X1", "--sg", "I4"])
    report("P1 x X1 (I4) +/-k line stars named LD and LE (ISOTROPY partner)",
           code == 0 and "P1 x X1 = LD1 + LD2 + LE1 + LE2" in out
           and "LD: (1/4, 1/4, 3/4)" in out and "LE: (3/4, 3/4, 1/4)" in out
           and "2 x 2 = 4 -> 1 + 1 + 1 + 1 = 4" in out, out)

    # --product (and --supergroup-cif) name a polar-line star as every other
    # command does: the same names from the product code and the labeller
    probe = (
        "import contextlib, io\n"
        "import numpy as np\n"
        "from crystod.isoir import get_isoir_label_map\n"
        "from crystod.spacegroup_product import DEN, SpaceGroupIrrepAlgebra\n"
        "for sg, k in (('P2_1', (0, 13, 0)), ('Pmc2_1', (0, 0, 13)),\n"
        "              ('P6_3', (0, 0, 14)), ('P6_3', (0, 0, 10))):\n"
        "    with contextlib.redirect_stdout(io.StringIO()):\n"
        "        algebra = SpaceGroupIrrepAlgebra(sg)\n"
        "    k = np.array(k)\n"
        "    point, names, _ = algebra._line_names(k)\n"
        "    cell, rotations, translations = algebra._isoir_labeler_inputs()\n"
        "    little = sorted(algebra.little_group(k))\n"
        "    characters = [np.array([small['chi'][i] for i in little])\n"
        "                  for small in algebra.computed_irreps_at(k)]\n"
        "    labels, ktype = get_isoir_label_map(\n"
        "        algebra.sg_type.number, cell, 1e-5,\n"
        "        (k / DEN) @ np.linalg.inv(algebra.primitive_matrix),\n"
        "        [rotations[i] for i in little], [translations[i] for i in little],\n"
        "        characters)\n"
        "    same = names == [labels[i] for i in range(len(names))]\n"
        "    print('LINE', sg, point, ktype, ','.join(names), same)\n"
    )
    code, out = run_python(probe)
    report("polar lines: --product names = labeller names (P2_1 LE, Pmc2_1 LE)",
           code == 0 and "LINE P2_1 LE LE LE2,LE1 True" in out
           and "LINE Pmc2_1 LE LE LE3,LE4,LE2,LE1 True" in out, out)
    report("polar line of P6_3: DU at (0,0,7/12), DT at (0,0,5/12), as the labeller",
           "LINE P6_3 DU DU DU2,DU1,DU6,DU5,DU4,DU3 True" in out
           and "LINE P6_3 DT DT DT1,DT2" in out, out)

    code, out = run_group(["--product", "R4-", "Q9", "--sg", "Pm-3m"])
    report("unknown space-group irrep label rejected with available list",
           code != 0 and "not tabulated" in out and "Available irreps" in out
           and "Traceback" not in out, out)

    code, out = run_group(["--product", "T2g", "T2g", "--pg", "m-3m", "--sg", "Pm-3m"])
    report("--product with both --pg and --sg rejected cleanly",
           code != 0 and "exactly one" in out and "Traceback" not in out, out)


# ------------------------------------- 9. --product --symmetric/--antisymmetric, --jahn-teller
def test_09_symmetric_square() -> None:
    print("\n[9] crystod-group --product --symmetric/--antisymmetric, --jahn-teller")

    # ---- point groups: T2g^2 = ^3T1g + ^1(A1g + Eg + T2g) of --multiplet
    code, out = run_group(["--product", "T2g", "T2g", "--pg", "m-3m",
                           "--symmetric", "--antisymmetric"])
    report("T2g x T2g --symmetric --antisymmetric exit 0", code == 0, out)
    report("plain --product lines kept (Result block)",
           "* Direct product *\nT2g*T2g" in out
           and "1(A1g) + 1(Eg) + 1(T1g) + 1(T2g)" in out, out)
    report("[T2g x T2g] = A1g + Eg + T2g (dimension 6)",
           "* Symmetric square *\n[T2g x T2g] = A1g + Eg + T2g\n"
           "dimension: 6 = 1 + 2 + 3" in out, out)
    report("{T2g x T2g} = T1g (dimension 3)",
           "* Antisymmetric square *\n{T2g x T2g} = T1g\ndimension: 3 = 3" in out, out)
    report("blocks separated by one blank line",
           "\n\n* Symmetric square *" in out and "\n\n* Antisymmetric square *" in out
           and "\n\n\n" not in out.strip(), out)
    code, out = run_group(["--product", "T2g", "T2g", "--pg", "m-3m", "--antisymmetric"])
    report("--antisymmetric alone prints only the antisymmetric block",
           code == 0 and "{T2g x T2g} = T1g" in out and "Symmetric square" not in out, out)

    # singlet orbital parts of --multiplet = symmetric square, triplets =
    # antisymmetric square (real-type point groups)
    probe = (
        "from crystod import group\n"
        "from crystod.multiplet import _GroupClasses\n"
        "bad = []\n"
        "for pg in ('m-3m', '432', '-43m', '6/mmm', '4/mmm', '-3m', '3m', 'mmm', '-42m'):\n"
        "    ct = group.get_character_table(pg)\n"
        "    classes = _GroupClasses(ct)\n"
        "    for irrep in ct['character_table']:\n"
        "        terms = group.shell_terms(classes, irrep, 2)\n"
        "        singlet = {n: c for s, n, c in terms if s == 0}\n"
        "        triplet = {n: c for s, n, c in terms if s == 1}\n"
        "        sym = {n: c for n, c in group.symmetric_square(ct, pg, irrep).items() if c}\n"
        "        anti = {n: c for n, c in group.antisymmetric_square(ct, pg, irrep).items() if c}\n"
        "        if sym != singlet or anti != triplet:\n"
        "            bad.append((pg, irrep, sym, singlet, anti, triplet))\n"
        "print('MISMATCH', bad)\n"
    )
    code, out = run_python(probe)
    report("squares = singlet/triplet parts of shell_terms (9 point groups, all irreps)",
           code == 0 and "MISMATCH []" in out, out)

    # complex-conjugate pairs (physical E of 3): the product is now reduced
    # with the character norm (E x E = 2A + E, dimension 4)
    code, out = run_group(["--product", "E", "E", "--pg", "3", "--symmetric",
                           "--antisymmetric"])
    report("3: E x E = 2(A) + 1(E); [E x E] = A + E; {E x E} = A",
           code == 0 and "2(A) + 1(E)" in out and "[E x E] = A + E" in out
           and "{E x E} = A\n" in out, out)
    code, out = run_group(["--basis", "x", "y", "z", "--pg", "4"])
    report("--basis x y z in 4 -> A + E (dimension 3)",
           code == 0 and "1.0 [A] + 1.0 [E]" in out, out)
    # -4: the class sizes follow rotation_list (E, S4, C2), not mapping_table
    code, out = run_group(["--product", "E", "E", "--pg", "-4", "--symmetric",
                           "--antisymmetric"])
    report("-4: E x E = 2(A) + 2(B); [E x E] = A + 2B; {E x E} = A",
           code == 0 and " 2(A) + 2(B)\n" in out
           and "[E x E] = A + 2B\ndimension: 3 = 1 + 2x1" in out
           and "{E x E} = A\n" in out and "WARNING" not in out, out)
    code, out = run_group(["--basis", "x", "y", "z", "--pg", "-4"])
    report("--basis x y z in -4 -> B + E",
           code == 0 and "1.0 [B] + 1.0 [E]" in out, out)
    # point group 1: scalar characters
    code, out = run_group(["--product", "A", "A", "--pg", "1", "--symmetric"])
    report("point group 1: [A x A] = A",
           code == 0 and "[A x A] = A\n" in out and "Traceback" not in out, out)
    code, out = run_group(["--jahn-teller", "A", "--pg", "1"])
    report("point group 1: --jahn-teller A -> JT-active: none",
           code == 0 and "JT-active: none" in out and "Traceback" not in out, out)
    code, out = run_group(["--basis", "x", "y", "z", "--pg", "1"])
    report("point group 1: --basis x y z -> 3.0 [A]",
           code == 0 and "3.0 [A]" in out and "Traceback" not in out, out)

    # element-wise reduction (independent of the class bookkeeping) over all
    # 32 point groups: squares agree, multiplicities >= 0, dimensions add up
    probe = (
        "import numpy as np\n"
        "from crystod import group\n"
        "pgs = ('1 -1 2 m 2/m 222 mm2 mmm 4 -4 4/m 422 4mm -42m 4/mmm 3 -3 32 3m -3m '\n"
        "       '6 -6 6/m 622 6mm -6m2 6/mmm 23 m-3 432 -43m m-3m').split()\n"
        "bad, count = [], 0\n"
        "for pg in pgs:\n"
        "    ct = group.get_character_table(pg)\n"
        "    names = ct['rotation_list']\n"
        "    names = [names] if isinstance(names, str) else list(names)\n"
        "    chars = {k: np.atleast_1d(np.asarray(v, dtype=float))\n"
        "             for k, v in ct['character_table'].items()}\n"
        "    elems = [(names.index(c), np.asarray(m)) for c in names\n"
        "             for m in np.asarray(ct['mapping_table'][c])]\n"
        "    def cls(mat):\n"
        "        return next(i for i, m in elems if np.allclose(m, mat))\n"
        "    sq = [cls(m @ m) for _, m in elems]\n"
        "    for irrep, chi in chars.items():\n"
        "        dim = int(round(chi[names.index('E')]))\n"
        "        for kind, sign, fn in (('sym', 1, group.symmetric_square),\n"
        "                               ('anti', -1, group.antisymmetric_square)):\n"
        "            count += 1\n"
        "            vals = [(chi[c] ** 2 + sign * chi[s]) / 2\n"
        "                    for (c, _), s in zip(elems, sq)]\n"
        "            ref = {}\n"
        "            for j, cj in chars.items():\n"
        "                num = sum(cj[c] * v for (c, _), v in zip(elems, vals))\n"
        "                den = sum(cj[c] ** 2 for c, _ in elems)\n"
        "                ref[j] = round(num / den)\n"
        "            got = fn(ct, pg, irrep)\n"
        "            total = sum(n * int(round(chars[j][names.index('E')]))\n"
        "                        for j, n in got.items())\n"
        "            want = dim * (dim + sign) // 2\n"
        "            if got != ref or min(got.values()) < 0 or total != want:\n"
        "                bad.append((pg, irrep, kind, got, ref))\n"
        "print('CHECKED', count, 'BAD', bad)\n"
    )
    code, out = run_python(probe)
    report("element-wise squares over all 32 point groups (no negative terms)",
           code == 0 and "BAD []" in out and "CHECKED 0 " not in out, out)

    # ---- Jahn-Teller active modes
    code, out = run_group(["--jahn-teller", "Eg", "--pg", "m-3m"])
    report("--jahn-teller Eg (m-3m): [Eg x Eg] = A1g + Eg, JT-active: Eg",
           code == 0 and "* Point group *\nm-3m" in out
           and "* Jahn-Teller active modes *\n[Eg x Eg] = A1g + Eg\nJT-active: Eg\n"
           "  (linear vibronic coupling allowed by symmetry; the Jahn-Teller theorem)" in out,
           out)
    code, out = run_group(["--jahn-teller", "T1u", "--pg", "m-3m"])
    report("--jahn-teller T1u (m-3m): JT-active: Eg + T2g",
           code == 0 and "[T1u x T1u] = A1g + Eg + T2g" in out
           and "JT-active: Eg + T2g" in out, out)
    code, out = run_group(["--jahn-teller", "E", "--pg", "-42m"])
    report("--jahn-teller E (-42m): E x (b1 + b2)",
           code == 0 and "JT-active: B1 + B2" in out, out)
    code, out = run_group(["--jahn-teller", "A1g", "--pg", "m-3m"])
    report("nondegenerate level: JT-active: none",
           code == 0 and "JT-active: none" in out, out)
    code, out = run_group(["--jahn-teller", "Eg", "T2g", "--pg", "m-3m"])
    pjt_note = ("  (necessary condition only: a mode Q can mix the two levels when Gamma_Q\n"
            "   occurs in the product; whether the mixing destabilizes the high-symmetry\n"
            "   structure depends on the energy gap and the coupling strength, which\n"
            "   symmetry does not give)")
    report("pseudo-Jahn-Teller Eg x T2g: symmetry-allowed coupling modes + note",
           code == 0 and "* Pseudo-Jahn-Teller coupling *\nEg x T2g = T1g + T2g\n"
           "symmetry-allowed coupling modes: T1g + T2g\n" + pjt_note in out
           and "PJT-active" not in out, out)
    code, out = run_group(["--jahn-teller", "T1u", "T1u", "--pg", "m-3m"])
    report("pseudo-Jahn-Teller T1u x T1u: first-order line, allowed modes, note",
           code == 0 and "T1u x T1u = A1g + Eg + T1g + T2g\n"
           "same degenerate level (first-order Jahn-Teller): "
           "[T1u x T1u] - A1g = Eg + T2g\n"
           "two different levels of symmetry T1u, "
           "symmetry-allowed coupling modes: Eg + T1g + T2g\n" + pjt_note in out, out)
    code, out = run_group(["--jahn-teller", "A1", "E", "--pg", "4mm"])
    report("pseudo-Jahn-Teller A1 x E (4mm) = E",
           code == 0 and "* Point group *\n4mm" in out
           and "A1 x E = E\nsymmetry-allowed coupling modes: E\n" + pjt_note in out
           and "same " not in out, out)
    code, out = run_group(["--jahn-teller", "A1g", "A1g", "--pg", "m-3m"])
    report("pseudo-Jahn-Teller A1g x A1g: nondegenerate, none allowed",
           code == 0 and "same level: nondegenerate, no first-order Jahn-Teller effect\n"
           "two different levels of symmetry A1g, "
           "symmetry-allowed coupling modes: none (only the totally symmetric irrep)"
           in out, out)

    # ---- space groups: squares of the full irreps
    code, out = run_group(["--product", "R4+", "R4+", "--sg", "Pm-3m", "--symmetric",
                           "--antisymmetric"])
    report("Pm-3m R4+ squares exit 0, plain product kept",
           code == 0 and "R4+ x R4+ = GM1+ + GM3+ + GM4+ + GM5+" in out
           and "Acta Cryst. A62" in out, out)
    report("[R4+ x R4+] = GM1+ + GM3+ + GM5+ (2k = 0)",
           "* Symmetric square (full space-group irreps) *\n"
           "[R4+ x R4+] = GM1+ + GM3+ + GM5+\ndimension: 6 = 1 + 2 + 3" in out, out)
    report("{R4+ x R4+} = GM4+",
           "{R4+ x R4+} = GM4+\ndimension: 3 = 3" in out, out)
    report("output opens with a blank line; DIRPRO citation after the square blocks",
           out.startswith("\n* Space group")
           and "Cross-validated" in out and "* Antisymmetric square" in out
           and out.index("Cross-validated") > out.index("* Antisymmetric square"), out)
    code, out = run_group(["--product", "X5+", "X5+", "--sg", "Pm-3m", "--symmetric",
                           "--antisymmetric"])
    report("Pm-3m [X5+ x X5+] has Gamma and M terms (2k = 0, X + X' = M)",
           code == 0
           and "[X5+ x X5+] = GM1+ + GM2+ + 2GM3+ + GM5+ + M1+ + M4+ + M5+" in out
           and "dimension: 21" in out
           and "{X5+ x X5+} = GM4+ + M2+ + M3+ + M5+" in out
           and "dimension: 15" in out, out)
    code, out = run_group(["--product", "P1", "P1", "--sg", "I4/mcm", "--symmetric"])
    report("complex-type P1 of I4/mcm: square of the real form P1P3 (noted)",
           code == 0 and "P1 is of complex type" in out
           and "[P1P3 x P1P3] = GM1+ + " in out and "dimension: 10 = " in out, out)

    # quadratic invariants (--invariants degree 2) = identity in the symmetric
    # square; symmetric + antisymmetric = plain product
    probe = (
        "import collections\n"
        "from crystod import group\n"
        "for sg, label in (('Pm-3m', 'R4+'), ('Pm-3m', 'X5+'), ('Pm-3m', 'M3+'),\n"
        "                  ('Fm-3m', 'W4'), ('I4/mcm', 'P1'), ('P6_3/mmc', 'K5')):\n"
        "    algebra = group.SpaceGroupIrrepAlgebra(sg)\n"
        "    sym = algebra.decompose_square(label, 'symmetric')\n"
        "    anti = algebra.decompose_square(label, 'antisymmetric')\n"
        "    ident = sum(n for _, irrep, n in sym.terms if irrep.name in ('GM1+', 'GM1'))\n"
        "    quadratic = group.invariant_polynomials(sg, label, degree=2).counts[2]\n"
        "    both = collections.Counter()\n"
        "    for square in (sym, anti):\n"
        "        for _, irrep, n in square.terms:\n"
        "            both[irrep.name] += n\n"
        "    _, terms, _ = algebra.decompose_product([label, label])\n"
        "    plain = collections.Counter()\n"
        "    for _, irrep, n in terms:\n"
        "        plain[irrep.name] += n\n"
        "    dims = [sum(n * algebra.full_dimension(irrep) for _, irrep, n in q.terms)\n"
        "            for q in (sym, anti)]\n"
        "    print('SQ', sg, label, ident == quadratic,\n"
        "          sym.doubled or both == plain, dims == [sym.dimension, anti.dimension])\n"
    )
    code, out = run_python(probe)
    report("identity count in [IR x IR] = degree-2 invariant count (6 irreps)",
           code == 0 and out.count("SQ ") == 6
           and all(line.split()[3] == "True" for line in out.splitlines()
                   if line.startswith("SQ ")), out)
    report("[IR x IR] + {IR x IR} = IR x IR; dimensions n(n+-1)/2",
           code == 0 and all(line.split()[4:] == ["True", "True"]
                             for line in out.splitlines() if line.startswith("SQ ")), out)

    # ---- API
    probe = (
        "from crystod import group\n"
        "ct = group.get_character_table('m-3m')\n"
        "print('SYM', {k: n for k, n in group.symmetric_square(ct, 'm-3m', 'T2g').items() if n})\n"
        "result = group.jahn_teller_modes('m-3m', 'T1u')\n"
        "print('JT', type(result).__name__, result.identity, result.active, result.pseudo)\n"
        "pjt = group.jahn_teller_modes('m-3m', 'T1u', 'T1u')\n"
        "print('PJT', pjt.allowed == pjt.active, pjt.first_order, result.first_order)\n"
        "try:\n"
        "    group.jahn_teller_modes('m-3m', 'Q1')\n"
        "except ValueError as exc:\n"
        "    print('VALUEERROR', 'not in irreps' in str(exc))\n"
    )
    code, out = run_python(probe)
    report("API: symmetric_square, jahn_teller_modes (JahnTellerModes), ValueError",
           code == 0 and "SYM {'A1g': 1, 'Eg': 1, 'T2g': 1}" in out
           and "JT JahnTellerModes A1g {'Eg': 1, 'T2g': 1} False" in out
           and "PJT True {'Eg': 1, 'T2g': 1} None" in out
           and "VALUEERROR True" in out, out)

    # ---- error paths
    code, out = run_group(["--product", "T2g", "Eg", "--pg", "m-3m", "--symmetric"])
    report("--symmetric with different irreps rejected cleanly",
           code != 0 and "needs two identical irreps" in out and "Traceback" not in out, out)
    code, out = run_group(["--product", "R4+", "R5+", "--sg", "Pm-3m", "--antisymmetric"])
    report("--antisymmetric with different space-group irreps rejected cleanly",
           code != 0 and "needs two identical irreps (e.g. --product R4+ R4+)" in out
           and "Traceback" not in out, out)
    code, out = run_group(["--table", "--pg", "m-3m", "--symmetric"])
    report("--symmetric without --product rejected cleanly",
           code != 0 and "only used with --product" in out and "Traceback" not in out, out)
    code, out = run_group(["--jahn-teller", "Eg"])
    report("--jahn-teller without --pg rejected cleanly",
           code != 0 and "requires --pg" in out and "Traceback" not in out, out)
    code, out = run_group(["--jahn-teller", "Eg", "--sg", "Pm-3m"])
    report("--jahn-teller with --sg rejected cleanly",
           code != 0 and "point groups only" in out and "Traceback" not in out, out)
    code, out = run_group(["--jahn-teller", "Eg", "T2g", "T1u", "--pg", "m-3m"])
    report("--jahn-teller with three irreps rejected cleanly",
           code != 0 and "one irrep" in out and "Traceback" not in out, out)
    code, out = run_group(["--jahn-teller", "Xg", "--pg", "m-3m"])
    report("--jahn-teller with an unknown irrep rejected with the available list",
           code != 0 and "Choose from" in out and "Traceback" not in out, out)


# ---------------------------------------------------------------- 10. crystod-group --decompose
def test_10_decompose_irrep() -> None:
    print("\n[10] crystod-group --decompose")
    code, out = run_group(
        ["--decompose", "--point-group", "3m", "--characters", "3", "0", "1"]
    )
    report("3m with characters 3 0 1 exit 0", code == 0, out)
    report("3 0 1 in 3m -> A1 + E", "1(A1)" in out and "1(E)" in out and "(A2)" not in out, out)

    code, out = run_group(
        ["--decompose", "--point-group", "m-3m",
         "--characters", "9", "0", "1", "3", "-1", "-3", "0", "5", "1", "3"]
    )
    report("m-3m Gamma-point ScF3-like rep exit 0", code == 0, out)

    code, out = run_group(["--decompose", "--point-group", "xyz", "--characters", "1"])
    report("unknown point group rejected cleanly",
           code != 0 and "not in the available point groups" in out and "Traceback" not in out, out)

    code, out = run_group(["--decompose", "--point-group", "3m", "--characters", "3", "0"])
    report("wrong character count rejected cleanly",
           code != 0 and "3 characters are required" in out and "Traceback" not in out, out)

    code, out = run_group(
        ["--decompose", "--point-group", "-43m", "--characters", "3", "0", "-1", "-1", "1"]
    )
    report("leading-dash point group -43m accepted (space form)",
           code == 0 and "1(T2)" in out, out)

    code, out = run_group(["--ligand-field", "d", "--point-group", "-43m"])
    report("d in -43m -> E + T2 (tetrahedral splitting)",
           code == 0 and "1(E)" in out and "1(T2)" in out, out)

    # real pairs of complex-conjugate irreps (norm 2|G|) and the class order
    code, out = run_group(["--decompose", "--point-group", "4", "--characters", "2", "0", "-2"])
    report("E of 4 (a real pair of complex irreps) counts once: 1(E)",
           code == 0 and "1(E)" in out and "2(E)" not in out, out)

    code, out = run_group(["--decompose", "--point-group", "-4", "--characters", "5", "-1", "1"])
    report("-4 class sizes in the class order (2S4, 1C2): A + 2B + E",
           code == 0 and "2S4: -1" in out and "1C2: 1" in out
           and "1(A) + 2(B) + 1(E)" in out, out)
    report("--decompose characters follow their * Title * line directly",
           "* Reducible representation *\n1E: 5\n" in out, out)

    code, out = run_group(["--decompose", "--point-group", "1", "--characters", "3"])
    report("point group 1 (scalar characters) -> 3(A)",
           code == 0 and "3(A)" in out and "Traceback" not in out, out)

    from phonopy.phonon.character_table import character_table as all_tables

    from crystod.decompose_irrep import decompose, get_character_table, irrep_character_rows

    failures = []
    for point_group in all_tables:
        table = get_character_table(point_group)
        rows = irrep_character_rows(table)
        for name, row in rows.items():
            counts = decompose(list(row), table)
            if counts != {other: int(other == name) for other in rows}:
                failures.append(f"{point_group} {name}: {counts}")
    report("every irrep row of the 32 point groups reduces to itself once",
           len(all_tables) == 32 and not failures, "\n".join(failures))


# ---------------------------------------------------------------- 11. crystod-group --ligand-field
def test_11_ligand_field_split() -> None:
    print("\n[11] crystod-group --ligand-field")
    code, out = run_group(["--ligand-field", "d", "--point-group", "m-3m"])
    report("d in m-3m exit 0", code == 0, out)
    report("d in m-3m -> Eg + T2g",
           "1(Eg)" in out and "1(T2g)" in out and "(A1g)" not in out, out)

    code, out = run_group(["--ligand-field", "d", "--point-group", "4/mmm"])
    report("d in 4/mmm -> A1g + B1g + B2g + Eg",
           all(f"1({name})" in out for name in ("A1g", "B1g", "B2g", "Eg")), out)

    code, out = run_group(["--ligand-field", "f", "--point-group", "4/mmm"])
    report("f in 4/mmm -> A2u + B1u + B2u + 2Eu",
           all(name in out for name in ("1(A2u)", "1(B1u)", "1(B2u)", "2(Eu)")), out)

    code, out = run_group(["--ligand-field", "q", "--point-group", "m-3m"])
    report("unknown orbital rejected cleanly",
           code != 0 and "is not supported" in out and "Traceback" not in out, out)

    code, out = run_group(["--ligand-field", "d", "--point-group", "xyz"])
    report("unknown point group rejected cleanly",
           code != 0 and "not in the available point groups" in out and "Traceback" not in out, out)

    # real pairs of complex-conjugate irreps and the class order of -4
    code, out = run_group(["--ligand-field", "d", "--point-group", "4"])
    report("d in 4 -> A + 2B + E (5 orbitals; E is one real pair)",
           code == 0 and "1(A) + 2(B) + 1(E)" in out, out)

    code, out = run_group(["--ligand-field", "d", "--point-group", "-4"])
    report("d in -4: class sizes on their classes (2S4, 1C2)",
           code == 0 and "2S4: -1" in out and "1C2: 1" in out
           and "1(A) + 2(B) + 1(E)" in out, out)
    report("--ligand-field characters follow their * Title * line directly",
           "in the -4 field *\n1E: 5\n" in out, out)

    from phonopy.phonon.character_table import character_table as all_tables

    from crystod.decompose_irrep import decompose, get_character_table, irrep_character_rows
    from crystod.ligand_field import get_orbital_characters

    def splitting(point_group: str, orbital: str) -> dict[str, int]:
        table = get_character_table(point_group)
        return decompose(list(get_orbital_characters(orbital, table).values()), table)

    failures = []
    for point_group in all_tables:
        rows = irrep_character_rows(get_character_table(point_group))
        for orbital, size in (("s", 1), ("p", 3), ("d", 5), ("f", 7)):
            counts = splitting(point_group, orbital)
            dimension = sum(count * int(round(rows[name][0])) for name, count in counts.items())
            if dimension != size:
                failures.append(f"{orbital} in {point_group}: {counts} (dimension {dimension})")
    report("s/p/d/f splittings add up to 1/3/5/7 in all 32 point groups",
           not failures, "\n".join(failures))

    # the groups with complex-conjugate pairs, against the m-counting of
    # Y_lm ~ exp(i m phi) (the subduction tables of Altmann and Herzig)
    expected = {
        "4": ("A + E", "A + 2B + E", "A + 2B + 2E"),
        "-4": ("B + E", "A + 2B + E", "2A + B + 2E"),
        "4/m": ("Au + Eu", "Ag + 2Bg + Eg", "Au + 2Bu + 2Eu"),
        "3": ("A + E", "A + 2E", "3A + 2E"),
        "-3": ("Au + Eu", "Ag + 2Eg", "3Au + 2Eu"),
        "6": ("A + E1", "A + E1 + E2", "A + 2B + E1 + E2"),
        "-6": ("E' + A''", "A' + E' + E''", "2A' + E' + A'' + E''"),
        "6/m": ("Au + E1u", "Ag + E1g + E2g", "Au + 2Bu + E1u + E2u"),
        "23": ("T", "E + T", "A + 2T"),
        "m-3": ("Tu", "Eg + Tg", "Au + 2Tu"),
    }
    failures = []
    for point_group, results in expected.items():
        for orbital, result in zip("pdf", results):
            counts = splitting(point_group, orbital)
            text = " + ".join(
                (f"{count}" if count > 1 else "") + name for name, count in counts.items() if count
            )
            if text != result:
                failures.append(f"{orbital} in {point_group}: {text} (expected {result})")
    report("p/d/f in the ten complex-pair groups (4, -4, 4/m, 3, -3, 6, -6, 6/m, 23, m-3)",
           not failures, "\n".join(failures))


# ---------------------------------------------------------------- 12. crystod-group --basis
def test_12_basis_function() -> None:
    print("\n[12] crystod-group --basis")
    code, out = run_group(["--basis", "x", "y", "z", "--point-group", "m-3m"])
    report("x y z in m-3m exit 0", code == 0, out)
    report("x y z in m-3m -> T1u", "T1u" in out, out)

    code, out = run_group(
        ["--basis", "x", "y", "z", "--space-group", "Pm-3m",
         "--kpoint", "0", "0", "0"]
    )
    report("x y z at GM in Pm-3m exit 0", code == 0, out)
    report("x y z at GM -> GM4-", "GM4-" in out, out)

    code, out = run_group(
        ["--basis", "x^2-y^2", "2z^2-x^2-y^2", "xy", "yz", "zx",
         "--space-group", "Pm-3m", "--kpoint", "0", "0", "0"]
    )
    report("d-type set at GM exit 0", code == 0, out)
    report("d-type set -> GM3+ and GM5+", "GM3+" in out and "GM5+" in out, out)
    report("no numerical-noise blowup", re.search(r"\d{15,}", out) is None, out)

    code, out = run_group(["--basis", "Rx", "Ry", "Rz", "--point-group", "m-3m"])
    report("axial Rx Ry Rz in m-3m -> T1g (not T1u)",
           code == 0 and "T1g" in out and "T1u" not in out, out)

    code, out = run_group(
        ["--basis", "Rx", "Ry", "Rz", "--space-group", "Pm-3m",
         "--kpoint", "0", "0", "0"]
    )
    report("axial Rx Ry Rz at GM in Pm-3m -> GM4+", code == 0 and "GM4+" in out, out)

    code, out = run_group(["--basis", "Rx", "Ry", "Rz", "--point-group", "6/mmm"])
    report("axial vector in 6/mmm -> A2g + E1g",
           code == 0 and "A2g" in out and "E1g" in out, out)

    # non-special k: labels fall back to the ISO-IR tables
    code, out = run_group(
        ["--basis", "x", "y", "z", "--space-group", "Pm-3m",
         "--kpoint", "0", "0", "0.1"]
    )
    report("x y z on the DT line labeled via ISO-IR",
           code == 0 and "DT1(1)" in out and "DT5(2)" in out
           and "irrep_" not in out, out)
    report("DT-line k point named via ISO-IR", "DT [0.0, 0.0, 0.1]" in out, out)

    code, out = run_group(
        ["--basis", "x", "y", "z", "--space-group", "Fd-3m",
         "--kpoint", "0.1", "0", "0.1"]
    )
    report("F-centred Fd-3m DT line labeled via ISO-IR",
           code == 0 and "DT1(1)" in out and "DT5(2)" in out, out)

    code, out = run_group(["--basis", "x*Ry - y*Rx", "--point-group", "m-3m"])
    report("toroidal x*Ry - y*Rx -> T1u", code == 0 and "T1u" in out, out)

    code, out = run_group(["--basis", "Rx", "Ry", "Rz", "--space-group", "Pm-3m"])
    report("no --kpoint: all special k points analyzed",
           code == 0 and "analyzing all special k points" in out
           and out.count("k-point (primitive)") >= 4, out)
    report("axial decompositions at GM and R are gerade",
           "GM4+" in out and "R4+" in out, out)
    report("no --kpoint: opens with a * Special k points * block, no double blank",
           out.startswith("\n* Special k points *\n") and "\n\n\n" not in out, out)

    code, out = run_group(["--basis", "x", "y", "z", "--point-group", "m-3m",
                           "--show-irrep-table"])
    report("--basis --show-irrep-table: one * Point group * block",
           code == 0 and out.startswith("\n* Point group *")
           and out.count("* Point group *") == 1 and "T1u" in out, out)


# ---------------------------------------------------------------- 13. crystod-group --tensor
def test_13_tensor_form() -> None:
    print("\n[13] crystod-group --tensor")

    # Nye's numbers of independent components over the 32 point groups
    probe = r'''
import numpy as np
from crystod.tensor_form import (POINT_GROUPS, CRYSTAL_SYSTEM, tensor_form,
                                 point_group_setting, _rotate)
dielectric = {"triclinic": 6, "monoclinic": 4, "orthorhombic": 3, "tetragonal": 2,
              "trigonal": 2, "hexagonal": 2, "cubic": 1}
piezo = {"1": 18, "2": 8, "m": 10, "222": 3, "mm2": 5, "4": 4, "-4": 4, "422": 1,
         "4mm": 3, "-42m": 2, "3": 6, "32": 2, "3m": 4, "6": 4, "-6": 2, "622": 1,
         "6mm": 3, "-6m2": 1, "23": 1, "432": 0, "-43m": 1}
elastic = {"1": 21, "-1": 21, "2": 13, "m": 13, "2/m": 13, "222": 9, "mm2": 9,
           "mmm": 9, "4": 7, "-4": 7, "4/m": 7, "422": 6, "4mm": 6, "-42m": 6,
           "4/mmm": 6, "3": 7, "-3": 7, "32": 6, "3m": 6, "-3m": 6}
pyro = {"1": 3, "2": 1, "m": 2, "mm2": 1, "4": 1, "4mm": 1, "3": 1, "3m": 1,
        "6": 1, "6mm": 1}
gyration = {"1": 6, "2": 4, "m": 2, "222": 3, "mm2": 1, "4": 2, "-4": 2, "422": 2,
            "-42m": 1, "3": 2, "32": 2, "6": 2, "622": 2, "23": 1, "432": 1}
bad = []
for pg in POINT_GROUPS:
    system = CRYSTAL_SYSTEM[pg]
    centro = point_group_setting(pg).centrosymmetric
    want = {
        "dielectric": dielectric[system],
        "piezoelectric": 0 if centro else piezo[pg],
        "elastic": elastic.get(pg, 5 if system == "hexagonal" else 3),
        "compliance": elastic.get(pg, 5 if system == "hexagonal" else 3),
        "pyroelectric": pyro.get(pg, 0),
        "gyration": gyration.get(pg, 0),
        "e[V2]": gyration.get(pg, 0),
        "a[V2]": 0,
    }
    for kind, n in want.items():
        form = tensor_form(pg, kind)
        if form.n_independent != n:
            bad.append(f"{pg} {kind}: {form.n_independent} != {n}")
        # every returned tensor is invariant under every operation
        axial = kind in ("gyration", "e[V2]")
        for tensor in form.tensors:
            for rotation in form.setting.rotations:
                sign = np.sign(np.linalg.det(rotation)) if axial else 1.0
                if not np.allclose(sign * _rotate(tensor, rotation), tensor, atol=1e-9):
                    bad.append(f"{pg} {kind}: tensor not invariant")
                    break
print("SWEEP", len(POINT_GROUPS), "BAD", bad)
from crystod.tensor_form import raman_forms
raman_bad = [pg for pg in POINT_GROUPS
             if sum(len(r.tensors) for r in raman_forms(pg)) != 6]
for number in range(1, 231):
    setting = point_group_setting(space_group=str(number))
    if (sum(len(r.tensors) for r in raman_forms(setting=setting)) != 6
            or tensor_form(None, "piezoelectric", space_group=str(number)).n_independent
            != tensor_form(setting.symbol, "piezoelectric").n_independent):
        raman_bad.append(number)
print("RAMAN BAD", raman_bad)
'''
    code, out = run_python(probe)
    report("sweep over the 32 point groups exit 0", code == 0, out)
    report("32 point groups x 8 kinds: Nye's counts (dielectric 6/4/3/2/1, piezo, "
           "elastic 21/13/9/7-6/7-6/5/3, pyro, gyration; a[V2] 0) and invariance",
           "SWEEP 32 BAD []" in out, out)
    report("Raman tensors of every irrep span the 6 symmetric tensors (32 point groups, "
           "230 space groups); --sg piezo count == point-group count",
           "RAMAN BAD []" in out, out)

    code, out = run_group(["--tensor", "piezoelectric", "--pg", "4mm"])
    report("--tensor piezoelectric --pg 4mm exit 0", code == 0, out)
    report("blocks Point group / Tensor / Independent components / Matrix form / "
           "Relations, each after a blank line",
           all(f"\n\n{title}\n" in "\n\n" + out for title in (
               "* Point group *", "* Tensor *", "* Independent components *",
               "* Matrix form *", "* Relations *")), out)
    report("4mm: d15, d31, d33 with d24 = d15 and d32 = d31",
           "  3: d15, d31, d33" in out and "  d24 = d15" in out
           and "  d32 = d31" in out, out)
    report("4mm: unindented 3x6 Nye matrix",
           re.search(r"^3\s+d31\s+d31\s+d33\s+0\s+0\s+0$", out, re.M) is not None
           and re.search(r"^d\s+1\s+2\s+3\s+4\s+5\s+6$", out, re.M) is not None, out)
    report("4mm: setting and Jahn symbol printed",
           "4mm (C4v)" in out and "4 || [001] (z)" in out
           and "Jahn symbol: V[V2] (polar, time-reversal even)" in out
           and "symmetric in (jk)" in out, out)

    code, out = run_group(["--tensor", "elastic", "--pg", "6/mmm"])
    report("hexagonal stiffness: 5 constants, c66 = (c11-c12)/2",
           code == 0 and "  5: c11, c12, c13, c33, c44" in out
           and "  c66 = (c11-c12)/2" in out, out)
    code, out = run_group(["--tensor", "compliance", "--pg", "6/mmm"])
    report("hexagonal compliance: s66 = 2(s11-s12) (Nye's factors)",
           code == 0 and "  s66 = 2(s11-s12)" in out
           and re.search(r"^6(\s+0){5}\s+2\(s11-s12\)$", out, re.M) is not None, out)
    code, out = run_group(["--tensor", "piezoelectric", "--pg", "32"])
    report("32: d11, d14; d12 = -d11, d25 = -d14, d26 = -2d11 (Nye)",
           code == 0 and "  2: d11, d14" in out and "  d26 = -2d11" in out
           and "  d25 = -d14" in out, out)
    code, out = run_group(["--tensor", "piezoelectric", "--pg", "3m"])
    report("3m: d15, d22, d31, d33; d21 = -d22, d16 = -2d22 (Nye)",
           code == 0 and "  4: d15, d22, d31, d33" in out
           and "  d21 = -d22" in out and "  d16 = -2d22" in out, out)
    code, out = run_group(["--tensor", "elastic", "--pg", "-3m"])
    report("trigonal -3m stiffness: c14 with c24 = -c14, c56 = c14",
           code == 0 and "  6: c11, c12, c13, c14, c33, c44" in out
           and "  c24 = -c14" in out and "  c56 = c14" in out, out)
    code, out = run_group(["--tensor", "gyration", "--pg", "-4"])
    report("gyration in -4: g11, g12 with g22 = -g11 (axial)",
           code == 0 and "  2: g11, g12" in out and "  g22 = -g11" in out
           and "(axial, time-reversal even)" in out, out)
    code, out = run_group(["--tensor", "pyroelectric", "--pg", "m"])
    report("pyroelectric in m: p1, p3 (mirror normal to y)",
           code == 0 and "  2: p1, p3" in out and "m perpendicular to [010] (y)" in out, out)
    code, out = run_group(["--tensor", "dielectric", "--pg", "m-3m"])
    report("cubic dielectric: one constant",
           code == 0 and "  1: eps11" in out and "  eps33 = eps11" in out, out)

    # space group and structure routes
    code, out = run_group(["--tensor", "elastic", "--sg", "194"])
    report("--sg 194 -> 6/mmm of P6_3/mmc, 5 constants",
           code == 0 and "point group of P6_3/mmc (No. 194)" in out
           and "  5: c11, c12, c13, c33, c44" in out, out)
    zno = os.path.join(ROOT, "example", "test_POSCARs", "186_PPOSCAR_ZnO")
    code, out = run_group(["--tensor", "piezoelectric", "-c", zno])
    report("-c ZnO: 6mm of P6_3mc, standard axes in the input frame, d15 d31 d33",
           code == 0 and "6mm (C6v), point group of P6_3mc (No. 186)" in out
           and "tensor axes in the Cartesian frame of the input file: "
               "x = (1.0000, 0.0000, 0.0000)" in out
           and "  3: d15, d31, d33" in out, out)
    batio3 = os.path.join(ROOT, "example", "test_POSCARs", "99_PPOSCAR_BaTiO3")
    code, out = run_group(["--tensor", "dielectric", "-c", batio3, "--tolerance", "0.01"])
    report("-c BaTiO3 --tolerance: 4mm, eps11 eps33",
           code == 0 and "(symprec 0.01)" in out and "  2: eps11, eps33" in out, out)

    # Raman tensors
    code, out = run_group(["--tensor", "raman", "--pg", "m-3m"])
    report("raman m-3m exit 0 with a Raman tensors block",
           code == 0 and "\n\n* Raman tensors *\n" in out, out)
    report("m-3m A1g, Eg, T2g as Loudon",
           "A1g (1 tensor):  [[a, 0, 0], [0, a, 0], [0, 0, a]]" in out
           and "Eg (2 tensors):  [[a, 0, 0], [0, a, 0], [0, 0, -2*a]]   "
               "[[a, 0, 0], [0, -a, 0], [0, 0, 0]]" in out
           and "T2g (3 tensors):  [[0, 0, 0], [0, 0, a], [0, a, 0]]   "
               "[[0, 0, a], [0, 0, 0], [a, 0, 0]]   [[0, a, 0], [a, 0, 0], [0, 0, 0]]" in out
           and "T1g" not in out and "u (" not in out, out)
    code, out = run_group(["--tensor", "raman", "--pg", "-3m"])
    report("-3m Eg: Loudon partners with shared constants",
           code == 0 and "Eg (2 partners, constants a, b):  [[a, 0, 0], [0, -a, b], "
           "[0, b, 0]]   [[0, a, b], [a, 0, 0], [b, 0, 0]]" in out, out)
    code, out = run_group(["--tensor", "raman", "--sg", "P-4m2"])
    report("--sg P-4m2: labels carried over from the -42m table (axes rotated)",
           code == 0 and "labels are carried over" in out
           and "B2 (1 tensor):  [[a, 0, 0], [0, -a, 0], [0, 0, 0]]" in out, out)
    code, out = run_group(["--tensor", "raman", "-c", zno])
    report("raman -c ZnO: phonon irreps with ISO-IR [Mulliken] labels",
           code == 0 and "GM5 [E2] (2 tensors)" in out and "GM6 [E1] (2 tensors)" in out
           and "Cartesian axes of the input cell" in out, out)
    probe = r'''
import numpy as np
from crystod.phonon_activity import format_tensor, gamma_raman_tensors, mulliken_symbols
from crystod.tensor_form import raman_forms
def projector(tensors):
    q, _ = np.linalg.qr(np.array([np.ravel(t) for t in tensors]).T)
    return q @ q.T
for name, pg in (("227_PPOSCAR_Si", "m-3m"), ("99_PPOSCAR_BaTiO3", "4mm"),
                 ("186_PPOSCAR_ZnO", "6mm")):
    path = "example/test_POSCARs/" + name
    mulliken = mulliken_symbols(path)
    table = {record.label: record for record in raman_forms(pg)}
    for record in gamma_raman_tensors(path):
        other = table[mulliken[record.label]]
        same_space = np.allclose(projector(record.tensors), projector(other.tensors))
        same_text = ([format_tensor(t) for t in record.tensors]
                     == [format_tensor(t) for t in other.tensors])
        print("ROUTE", pg, record.label, mulliken[record.label], same_space, same_text)
'''
    code, out = run_python(probe)
    rows = [line.split() for line in out.splitlines() if line.startswith("ROUTE")]
    report("structure route == point-group route for Si (m-3m), BaTiO3 (4mm), ZnO (6mm)",
           code == 0 and len(rows) == 7
           and all(row[-2:] == ["True", "True"] for row in rows)
           and {row[1] for row in rows} == {"m-3m", "4mm", "6mm"}, out)

    # time reversal, Jahn symbols
    code, out = run_group(["--tensor", "a[V2]", "--pg", "4mm"])
    report("time-odd a[V2]: forbidden in the grey group 4mm1'",
           code == 0 and "  0\n" in out and "4mm1'" in out
           and "time-reversal odd" in out, out)
    code, out = run_group(["--tensor", "[V2][V2]", "--pg", "6/mmm"])
    report("Jahn [V2][V2] in 6/mmm: T13 != T31, T66 = (T11-T12)/2",
           code == 0 and "  6: T11, T12, T13, T31, T33, T44" in out
           and "  T66 = (T11-T12)/2" in out, out)
    code, out = run_group(["--tensor", "[[V2]2]", "--pg", "m-3m"])
    report("Jahn [[V2]2] equals elastic (3 constants, letter T)",
           code == 0 and "  3: T11, T12, T44" in out, out)
    code, out = run_group(["--tensor", "{V3}", "--pg", "1"])
    report("Jahn {V3}: one component T123 listed with its signs",
           code == 0 and "  1: T123" in out and re.search(r"^T132\s+-T123$", out, re.M)
           is not None, out)

    # API: magnetic operations through the time-reversal flags (the F8 hook)
    probe = r'''
import numpy as np
import crystod
from crystod.group import tensor_form, TensorForm, format_tensor_form
from crystod.tensor_form import tensor_form_of_operations, point_group_setting
setting = point_group_setting("2/m")
flags = [not (np.allclose(r, np.eye(3)) or np.allclose(r, -np.eye(3)))
         for r in setting.rotations]                 # 2'/m'
print("MAG", tensor_form_of_operations(setting.rotations, "aeV",
                                       time_reversal=flags).independent)
print("GREY", tensor_form("2/m", "aeV").n_independent)
form = tensor_form("6mm", "piezoelectric")
print("TYPE", isinstance(form, TensorForm), form.tensors.shape)
print("FMT", "* Matrix form *" in format_tensor_form(form))
for args in (("4mn", "dielectric"), ("4mm", "foo")):
    try:
        tensor_form(*args)
        print("NOT RAISED", args)
    except ValueError:
        print("VALUEERROR", args[0])
'''
    code, out = run_python(probe)
    report("magnetic 2'/m': magnetization in the xz plane (2 components); grey 2/m1': 0",
           code == 0 and "MAG ('T1', 'T3')" in out and "GREY 0" in out, out)
    report("crystod.group exports tensor_form / TensorForm / format_tensor_form",
           "TYPE True (3, 3, 3, 3)" in out and "FMT True" in out, out)
    report("bad point group or kind raises ValueError through crystod.group",
           "VALUEERROR 4mn" in out and "VALUEERROR 4mm" in out
           and "NOT RAISED" not in out, out)

    # errors
    for args, message in (
        (["--tensor", "dielectric"], "--tensor requires exactly one of"),
        (["--tensor", "dielectric", "--pg", "4mm", "-c", zno], "--tensor requires exactly one of"),
        (["--tensor", "dielectric", "--pg", "4mm", "--tolerance", "0.1"],
         "--tolerance is only used with --tensor -c"),
        (["--tensor", "dielectric", "--pg", "4mm", "--kpoint", "0", "0", "0"],
         "--kpoint is not used with --tensor"),
        (["--tensor", "foo", "--pg", "4mm"], "Give one of dielectric"),
        (["--tensor", "[V2", "--pg", "4mm"], "unbalanced brackets"),
        (["--tensor", "dielectric", "--pg", "4mn"], "is not a crystallographic point group"),
        (["--tensor", "dielectric", "-c", "no_such_POSCAR"], "cannot read the structure"),
    ):
        code, out = run_group(args)
        label = " ".join(os.path.basename(arg) for arg in args[1:])
        report(f"error: --tensor {label}",
               code != 0 and message in out and "Traceback" not in out, out)


# ---------------------------------------------------------------- 14. crystod-group --generate-basis
def test_14_generate_basis_function() -> None:
    print("\n[14] crystod-group --generate-basis")
    code, out = run_group(["--generate-basis", "--point-group", "m-3m"])
    report("point-group mode exit 0", code == 0, out)
    report("all three orders printed",
           all(key in out for key in ("1st order", "2nd order", "3rd order")), out)
    report("1st order -> T1u; 3rd order -> A2u",
           "T1u" in out and "A2u" in out, out)
    report("block layout: group block once, order-suffixed titles, no banners",
           out.startswith("\n* Point group *") and out.count("* Point group *") == 1
           and "\n\n* Decomposition: 1st order (linear) *\n" in out
           and "* Decomposition: 3rd order" in out and "====" not in out, out)

    code, out = run_group(
        ["--generate-basis", "--space-group", "Pm-3m", "--kpoint", "0", "0", "0",
         "--order", "2"]
    )
    report("space-group mode (order 2) exit 0", code == 0, out)
    report("only 2nd order printed", "2nd order" in out and "1st order" not in out, out)
    report("2nd order -> GM1+, GM3+, GM5+",
           all(label in out for label in ("GM1+", "GM3+", "GM5+")), out)
    report("no numerical-noise blowup (GM3+ regression)",
           re.search(r"\d{15,}", out) is None, out)
    gm3_lines = [line for line in out.splitlines() if "GM3+(2):" in line and "[" in line]
    no_imaginary = all(
        "Ix" not in line and "I*" not in line and "*I" not in line for line in gm3_lines
    )
    report("GM3+ basis without imaginary residue", bool(gm3_lines) and no_imaginary,
           "\n".join(gm3_lines))

    # non-special k: shares the ISO-IR fallback of --basis
    code, out = run_group(
        ["--generate-basis", "--space-group", "Pm-3m",
         "--kpoint", "0", "0", "0.1", "--order", "1"]
    )
    report("--generate-basis on the DT line labeled via ISO-IR",
           code == 0 and "DT1(1)" in out and "DT5(2)" in out, out)


# ---------------------------------------------------------------- 15. crystod-group --coset
def test_15_show_coset() -> None:
    print("\n[15] crystod-group --coset")
    code, out = run_group(["--coset", "--point-group", "m-3m", "--subgroup", "4/mmm"])
    report("point-group mode exit 0", code == 0, out)
    report("index [G:H] = 3", "index [G:H] = 3" in out, out)
    report("three cosets listed", out.count("coset ") >= 3, out)
    report("output opens with a blank line and '* Groups *' (no leading space)",
           out.startswith("\n* Groups *\n"), out)

    code, out = run_group(["--coset", "--space-group", "Pm-3m", "--kpoint", "0.5", "0.5", "0"])
    report("space-group mode exit 0", code == 0, out)
    report("index [G:G_k] = |star of k| = 3", "= 3" in out and "G_k" in out, out)

    # error paths: a one-line ERROR and a nonzero exit, no traceback
    code, out = run_group(["--coset", "--point-group", "m-3m", "--subgroup", "3m"])
    report("3m in m-3m (3m is tabulated in hexagonal axes): exit != 0, "
           "one-line ERROR, no traceback",
           code != 0 and out.strip().startswith("ERROR: H = 3m (order 6) is not a subgroup")
           and len(out.strip().splitlines()) == 1 and "Traceback" not in out, out)
    code, out = run_group(["--coset", "--point-group", "4/mmm", "--subgroup", "m-3m"])
    report("H larger than G (m-3m in 4/mmm): exit != 0, ERROR, no traceback",
           code != 0 and "ERROR" in out and "Traceback" not in out, out)

    # Group closure of the primitive-basis symmetry operations, all 230 space
    # groups: every product (R1 R2 | R1 t2 + t1) must be an operation of the
    # set modulo Z^3. A row-form translation transform (t @ Minv) breaks it for
    # R-centred groups with fractional translations (161 R3c, 167 R-3c).
    import numpy as np
    from crystod.coset import _space_group_primitive_symmetry

    broken = []
    for number in range(1, 231):
        _, _, rotations, translations = _space_group_primitive_symmetry(str(number))
        rotations = np.asarray(rotations, dtype=int)
        order = len(rotations)
        product_rotations = np.einsum("iab,jbc->ijac", rotations, rotations).reshape(order * order, 9)
        product_translations = (
            np.einsum("iab,jb->ija", rotations, translations) + translations[:, None, :]
        ).reshape(order * order, 3)
        same_rotation = (product_rotations[:, None, :] == rotations.reshape(order, 9)[None]).all(-1)
        delta = product_translations[:, None, :] - translations[None]
        same_translation = (np.abs(delta - np.rint(delta)) < 1e-6).all(-1)
        missing = int((~(same_rotation & same_translation).any(axis=1)).sum())
        if missing:
            broken.append((number, missing, order * order))
    report("primitive-basis operations close as a group for all 230 space groups",
           not broken, f"(space group, products outside the set, products) = {broken}")

    # The printed cosets do not depend on the translations (R-3c, Gamma and L).
    code, out = run_group(["--coset", "--sg", "R-3c", "--kpoint", "0", "0", "0"])
    report("R-3c Gamma: exit 0, |G_k| = 12, one coset",
           code == 0 and "order |G_k| = 12 (|G| = 12)" in out
           and "index [G:G_k] = |star of k| = 1" in out
           and "{ 1, 3^+_001, 3^-_001, 2_100, 2_110, 2_010, -1, -3^+_001, -3^-_001, "
               "m_100, m_110, m_010 }" in out, out)
    code, out = run_group(["--coset", "--sg", "R-3c", "--kpoint", "0.5", "0", "0"])
    report("R-3c L: exit 0, |G_k| = 4, three cosets with their k arms",
           code == 0 and "order |G_k| = 4 (|G| = 12)" in out
           and "index [G:G_k] = |star of k| = 3" in out
           and "coset 1 (representative: 1, k arm = [+0.5, +0, +0]):\n   { 1, 2_010, -1, m_010 }" in out
           and "coset 2 (representative: 3^+_001, k arm = [+0, +0, +0.5]):\n"
               "   { 3^+_001, 2_110, -3^+_001, m_110 }" in out
           and "coset 3 (representative: 3^-_001, k arm = [+0, +0.5, +0]):\n"
               "   { 3^-_001, 2_100, -3^-_001, m_100 }" in out, out)


# ---------------------------------------------------------------- 16. crystod-group --correlate
def test_16_correlate() -> None:
    print("\n[16] crystod-group --correlate (subduction to an isotropy subgroup, "
          "compatibility relations, point-group correlation tables)")

    def lines_of(text: str) -> list[str]:
        return [" ".join(line.split()) for line in text.splitlines()]

    # (c) subduction of parent irreps to the Gamma point of the isotropy
    # subgroup: the R4+ soft mode of SrTiO3 (Pm-3m -> I4/mcm) is A1g + Eg
    code, out = run_group(["--correlate", "--parent", "Pm-3m", "--irrep", "R4+",
                           "--order-parameter", "0", "0", "a",
                           "--irrep-list", "R4+", "R5+", "M3+", "X5+", "GM4-"])
    rows = lines_of(out)
    report("R4+ (0,0,a): exit 0, the three blocks in order",
           code == 0 and out.find("* Supergroup *") < out.find("* Isotropy subgroup *")
           < out.find("* Subduction to the Gamma point of H *")
           and out.find("* Supergroup *") >= 0, out)
    report("R4+ (0,0,a): the isotropy-subgroup lines of --parent --order-parameter",
           "R4+(0,0,a) -> I4/mcm (No. 140)" in out and "cell size 2, index 6" in out
           and "sublattice basis (parent primitive units):" in out, out)
    report("R4+ (0,0,a): the ISO-IR frame of H is printed (basis and origin)",
           "H = I4/mcm (No. 140); Gamma labels in the ISO-IR frame of H:" in out
           and "basis (parent conventional units): (-1,0,1), (1,0,1), (0,2,0)" in rows
           and "origin: (0,0,0)" in rows, out)
    report("R4+ (0,0,a) -> I4/mcm Gamma: GM1+ [A1g] + GM5+ [Eg]",
           "R4+ (R, dim 3) -> GM1+ [A1g] + GM5+ [Eg]" in rows, out)
    report("R5+ -> GM4+ [B2g] + GM5+ [Eg]; GM4- -> GM3- [A2u] + GM5- [Eu]",
           "R5+ (R, dim 3) -> GM4+ [B2g] + GM5+ [Eg]" in rows
           and "GM4- (GM, dim 3) -> GM3- [A2u] + GM5- [Eu]" in rows, out)
    report("M3+ and X5+ lie at non-Gamma k of I4/mcm (no Gamma sector)",
           "M3+ (M, dim 3) -> 3 components at non-Gamma k of H" in rows
           and "X5+ (X, dim 6) -> 6 components at non-Gamma k of H" in rows, out)

    code, out = run_group(["--correlate", "--parent", "Pm-3m", "--irrep", "R4+",
                           "--order-parameter", "a", "a", "a"])
    rows = lines_of(out)
    report("R4+ (a,a,a) -> R-3c Gamma: GM1+ [A1g] + GM3+ [Eg] (default irrep list)",
           code == 0 and "R4+(a,a,a) -> R-3c (No. 167)" in out
           and "R4+ (R, dim 3) -> GM1+ [A1g] + GM3+ [Eg]" in rows
           and sum(row.startswith(("R", "M", "X", "GM")) and "->" in row
                   and "dim" in row for row in rows) == 1, out)

    # Pnma of CaTiO3 (a-a-c+): R4+ (0,a,a) + M3+ (d;0;0); every irrep holds
    # GM1+ once (one free parameter each); only one arm of M3+ is at Gamma
    code, out = run_group(["--correlate", "--parent", "Pm-3m", "--irrep", "R4+", "M3+",
                           "--order-parameter", "0", "a", "a", "d", "0", "0"])
    rows = lines_of(out)
    report("R4+ (0,a,a) + M3+ (d;0;0) -> Pnma: R4+ -> GM1+ [Ag] + 2 B_g irreps",
           code == 0 and "R4+(0,a,a) M3+(d;0;0) -> Pnma (No. 62)" in out
           and "R4+ (R, dim 3) -> GM1+ [Ag] + GM2+ [B1g] + GM4+ [B2g]" in rows, out)
    report("Pnma: M3+ -> GM1+ [Ag] + 2 components at non-Gamma k of H",
           "M3+ (M, dim 3) -> GM1+ [Ag] + 2 components at non-Gamma k of H" in rows, out)

    # zone boundary: X5+ doubles the cell in one direction only
    code, out = run_group(["--correlate", "--parent", "Pm-3m", "--irrep", "X5+",
                           "--order-parameter", "a", "0", "0", "0", "0", "0"])
    report("X5+ (a,0;0,0;0,0) -> Pmma: GM1+ + GM3+ and 4 non-Gamma components",
           code == 0 and "X5+(a,0;0,0;0,0) -> Pmma (No. 51)" in out
           and "X5+ (X, dim 6) -> GM1+ [Ag] + GM3+ [B3g] + 4 components at non-Gamma "
               "k of H" in lines_of(out), out)

    # the checks of the engine, through the API: GM1 multiplicity = n_free of
    # the direction for every primary irrep, the dimensions add up
    code, out = run_python(
        "from crystod import group\n"
        "cases = [('Pm-3m', ['R4+'], '0 0 a'), ('Pm-3m', ['R4+'], 'a a a'),\n"
        "         ('Pm-3m', ['R4+', 'M3+'], '0 a a d 0 0'), ('Pm-3m', ['M3+'], 'a 0 0'),\n"
        "         ('Pm-3m', ['X5+'], 'a 0 0 0 0 0'), ('Fm-3m', ['X3+'], 'a 0 0'),\n"
        "         ('I4/mmm', ['X2+', 'X3-'], '0 a 0 c')]\n"
        "for parent, irreps, direction in cases:\n"
        "    rows = group.subduce_to_child(parent, irreps, direction)\n"
        "    for r in rows:\n"
        "        gm1 = sum(n for name, n in r.gamma_part if name in ('GM1', 'GM1+'))\n"
        "        print('ROW', parent, r.label, r.primary, r.n_free, gm1, r.dimension,\n"
        "              r.nongamma_components, r.child_number)\n"
        "rows = group.subduce_to_child('Pm-3m', 'R4+', 'a 0 0', ['R4+', 'GM1+', 'GM3+'])\n"
        "print('SEC', [(r.label, r.gamma_part, r.primary) for r in rows])\n"
    )
    found = [line.split() for line in out.splitlines() if line.startswith("ROW")]
    report("API subduce_to_child: GM1 multiplicity = n_free = 1 for every primary irrep",
           code == 0 and len(found) == 9
           and all(row[3] == "True" and row[4] == "1" and row[5] == "1" for row in found),
           out)
    # Eg of Oh -> A1g + B2g: the a axis of I4/mcm is a face diagonal of the parent
    report("API: the strains GM1+ and GM3+ of R4+ (a,0,0) become GM1+ and GM1+ + GM4+",
           "SEC [('R4+', [('GM1+', 1), ('GM5+', 1)], True), ('GM1+', [('GM1+', 1)], False), "
           "('GM3+', [('GM1+', 1), ('GM4+', 1)], False)]" in out, out)

    # (b) compatibility relations: Pm-3m Gamma-Delta-X. BSW notation by the
    # characters of the little co-group 4mm: DT1 = Delta1, DT3 = Delta2'
    # (-1 on the fourfold and on m_100/m_001, +1 on m_101/m_-101), DT5 = Delta5
    code, out = run_group(["--correlate", "--sg", "Pm-3m", "--kpoint", "GM", "X"])
    rows = lines_of(out)
    report("Pm-3m GM-X: exit 0, blocks and the line header",
           code == 0 and "* Space group *" in out and "* Compatibility relations *" in out
           and "line DT between GM (0, 0, 0) and X (0, 1/2, 0), labeled at kdelta = "
               "(0, 1/24, 0) (primitive basis)" in out, out)
    report("GM4- (T1u) -> DT1 + DT5 (Delta1 + Delta5)",
           "GM4- (dim 3) -> DT1 + DT5" in rows, out)
    report("GM5+ (T2g) -> DT3 + DT5 (Delta2' + Delta5)",
           "GM5+ (dim 3) -> DT3 + DT5" in rows, out)
    report("GM3+ (Eg) -> DT1 + DT2; X side: X5+ -> DT5, X3- -> DT1",
           "GM3+ (dim 2) -> DT1 + DT2" in rows and "X5+ (dim 2) -> DT5" in rows
           and "X3- (dim 1) -> DT1" in rows, out)
    report("every irrep of both end points listed (10 at GM, 10 at X)",
           sum(row.startswith("GM") and "->" in row for row in rows) == 10
           and sum(row.startswith("X") and "->" in row for row in rows) == 10, out)
    code, out = run_group(["--table", "--sg", "Pm-3m", "--kpoint", "0", "1/4", "0"])
    table = lines_of(out)
    report("DT numbering: DT5 is the 2-dim irrep, DT3 has -1 on 4_010 and m_100, "
           "+1 on m_101",
           "irrep_3(2) = DT5(2) 2 -2 0 0 0 0 0 0" in table
           and "irrep_5(1) = DT3(1) 1 1 -1 -1 -1 -1 1 1" in table
           and "irrep 1 2_010 4^+_010 4^-_010 m_100 m_001 m_101 m_-101" in table, out)
    code, out = run_group(["--correlate", "--sg", "Pm-3m", "--kpoint", "GM", "X",
                           "--line", "DT"])
    report("--line DT gives the same relations", code == 0
           and "GM4- (dim 3) -> DT1 + DT5" in lines_of(out), out)

    # Si (Fd-3m, non-symmorphic): Gamma25' -> Delta2' + Delta5, X1 -> Delta1 + Delta2'
    code, out = run_group(["--correlate", "--sg", "Fd-3m", "--kpoint", "GM", "X"])
    rows = lines_of(out)
    report("Fd-3m GM-X: GM5+ -> DT3 + DT5, GM4- -> DT1 + DT5, X1 -> DT1 + DT3, X4 -> DT5",
           code == 0 and "line DT between GM (0, 0, 0) and X (1/2, 0, 1/2)" in out
           and "GM5+ (dim 3) -> DT3 + DT5" in rows and "GM4- (dim 3) -> DT1 + DT5" in rows
           and "X1 (dim 2) -> DT1 + DT3" in rows and "X4 (dim 2) -> DT5" in rows, out)

    # zone-boundary lines of Pm-3m: X-Z-M and R-S-X
    code, out = run_group(["--correlate", "--sg", "Pm-3m", "--kpoint", "X", "M"])
    rows = lines_of(out)
    report("Pm-3m X-M: line Z; X5+ -> Z2 + Z4, M5- -> Z1 + Z4",
           code == 0 and "line Z between X (0, 1/2, 0) and M (1/2, 1/2, 0)" in out
           and "X5+ (dim 2) -> Z2 + Z4" in rows and "M5- (dim 2) -> Z1 + Z4" in rows, out)
    code, out = run_group(["--correlate", "--sg", "Pm-3m", "--kpoint", "R", "X"])
    rows = lines_of(out)
    report("Pm-3m R-X: line S; R4+ -> S2 + S3 + S4, X5- -> S1 + S2",
           code == 0 and "line S between R (1/2, 1/2, 1/2) and X (0, 1/2, 0)" in out
           and "R4+ (dim 3) -> S2 + S3 + S4" in rows and "X5- (dim 2) -> S1 + S2" in rows,
           out)
    code, out = run_group(["--correlate", "--sg", "Pm-3m", "--kpoint", "GM"])
    report("one k point: every line from GM (LD to R, DT to X, SM to M)",
           code == 0 and "line LD between GM (0, 0, 0) and R (1/2, 1/2, 1/2)" in out
           and "line DT between GM (0, 0, 0) and X (0, 1/2, 0)" in out
           and "line SM between GM (0, 0, 0) and M (1/2, 1/2, 0)" in out, out)

    # sum rule over every pair of special points of a few groups (the engine
    # raises when a restriction does not decompose or the dimensions differ)
    code, out = run_python(
        "from crystod import group\n"
        "bad = 0; count = 0\n"
        "for sg in ('221', '227', '62', '194', '136', '216'):\n"
        "    algebra = group.SpaceGroupIrrepAlgebra(sg)\n"
        "    names = list(algebra.k_by_kname)\n"
        "    for a in range(len(names)):\n"
        "        for b in range(a + 1, len(names)):\n"
        "            for r in group.compatibility_relations(sg, names[a], names[b]):\n"
        "                count += 1\n"
        "                dims = {n.name: int(n.dim) for n in algebra.irreps_by_kname[r.k0_label]}\n"
        "                bad += dims[r.k0_irrep] != r.dimension\n"
        "print('SUMRULE', count, bad)\n"
    )
    match = re.search(r"SUMRULE (\d+) (\d+)", out)
    report("compatibility over every pair of special points of 221/227/62/194/136/216 "
           "(dimensions add up)",
           code == 0 and match is not None and int(match.group(1)) > 300
           and match.group(2) == "0", out)

    # review fixes: the ISO-IR origin is reduced modulo the translations of H
    # (not of the parent), a complex pair of H is one physically irreducible
    # term, one k name skips symmetry planes, a plane segment is headed plane
    code, out = run_group(["--correlate", "--parent", "Pm-3m", "--irrep", "X2+",
                           "--order-parameter", "0;a;a"])
    block = out[out.find("* Subduction to the Gamma point of H *"):]
    report("X2+ (0;a;a) -> P4/mmm: ISO-IR origin (0,0,1), as the conventional origin",
           code == 0 and "origin: (0,0,1)" in lines_of(block)
           and "origin: (0,0,1)" in lines_of(out[:out.find("* Subduction")]), out)
    code, out = run_group(["--correlate", "--parent", "Pm-3m", "--irrep", "GM4+",
                           "--order-parameter", "a", "a", "a"])
    report("GM4+ (a,a,a) -> R-3: GM1+ [Ag] + GM2+GM3+ [Eg] (complex pair merged)",
           code == 0 and "GM4+ (GM, dim 3) -> GM1+ [Ag] + GM2+GM3+ [Eg]" in lines_of(out),
           out)
    code, out = run_group(["--correlate", "--sg", "Fm-3m", "--kpoint", "W"])
    report("one k point (Fm-3m W): lines only, no symmetry plane",
           code == 0 and "line Q between W" in out and "line V between W" in out
           and "between W (1/2, 1/4, 3/4) and GM" not in out
           and not any(line.startswith("plane") for line in out.splitlines()), out)
    code, out = run_group(["--correlate", "--sg", "Pm-3m", "--kpoint", "GM", "X",
                           "--line", "A"])
    report("a segment on a symmetry plane is headed 'plane'",
           code == 0 and "plane A between GM (0, 0, 0) and X" in out, out)
    code, out = run_python(
        "from crystod import correlation\n"
        "report = correlation._subduction_report('Pm-3m', 'R4+', 'a 0 0')\n"
        "print('CORR-BEGIN'); print(correlation.format_isotropy_block(report))\n"
        "print('CORR-END')\n"
    )
    mine = out[out.find("CORR-BEGIN") + 10:out.find("CORR-END")].strip().splitlines()
    code2, ref = run_group(["--parent", "Pm-3m", "--irrep", "R4+",
                            "--order-parameter", "a", "0", "0"])
    ref_lines = [line.rstrip() for line in ref.splitlines()]
    report("the * Supergroup * / * Isotropy subgroup * lines equal those of "
           "--parent --order-parameter",
           code == 0 and code2 == 0 and len(mine) > 5
           and all(line.rstrip() in ref_lines for line in mine), out + ref)

    # error paths
    code, out = run_group(["--correlate"])
    report("--correlate without --parent/--sg rejected cleanly",
           code != 0 and "--correlate requires --parent SG" in out
           and "Traceback" not in out, out)
    code, out = run_group(["--correlate", "--parent", "Pm-3m", "--irrep", "R4+"])
    report("--correlate --parent without --order-parameter rejected cleanly",
           code != 0 and "requires --irrep and --order-parameter" in out
           and "Traceback" not in out, out)
    code, out = run_group(["--correlate", "--sg", "Pm-3m", "--kpoint", "GM", "Q"])
    report("unknown k-point name rejected cleanly (the k points are listed)",
           code != 0 and 'k point "Q" is not tabulated' in out
           and "Available k points" in out and "Traceback" not in out, out)
    code, out = run_group(["--correlate", "--sg", "Pm-3m", "--kpoint", "GM", "X",
                           "--line", "SM"])
    report("a line that does not join the points rejected cleanly",
           code != 0 and "no line named SM joins GM and X" in out
           and "Traceback" not in out, out)
    code, out = run_group(["--correlate", "--sg", "Pm-3m", "--kpoint", "GM", "GM"])
    report("the same point twice rejected cleanly",
           code != 0 and "give two different special points" in out
           and "Traceback" not in out, out)
    code, out = run_group(["--parent", "Pm-3m", "--irrep", "R4+", "--irrep-list", "R5+"])
    report("--irrep-list without --correlate rejected cleanly",
           code != 0 and "only used with --correlate" in out and "Traceback" not in out, out)
    code, out = run_group(["--correlate", "--sg", "Pm-3m", "--kpoint", "GM", "X",
                           "--irrep", "R4+"])
    report("--irrep with --correlate --sg rejected cleanly",
           code != 0 and "only used with --correlate --parent SG" in out
           and "Traceback" not in out, out)
    code, out = run_group(["--correlate", "--parent", "Pm-3m", "--irrep", "R4+",
                           "--order-parameter", "a", "0", "0", "--kpoint", "GM"])
    report("--kpoint with --correlate --parent rejected cleanly (form-specific message)",
           code != 0 and "--kpoint/--line are only used with --correlate --sg SG" in out
           and "Traceback" not in out, out)

    # (a) point-group correlation tables (--correlate --pg G --subgroup H):
    # one column per conjugacy class of subgroups of type H
    def table_rows(text: str) -> dict:
        block = text[text.find("* Correlation table *"):].split("\n\n")[0]
        rows = {}
        for line in block.splitlines()[1:]:
            cells = re.split(r"\s{2,}", line.strip())
            rows[cells[0]] = cells[1:]
        return rows

    code, out = run_group(["--correlate", "--pg", "m-3m", "--subgroup", "4/mmm"])
    rows = table_rows(out)
    report("m-3m -> 4/mmm (4 || z): A1g -> A1g, Eg -> A1g + B1g, T2g -> B2g + Eg, "
           "T1u -> A2u + Eu",
           code == 0 and rows.get("irrep") == ["4/mmm (4 || z)"]
           and rows.get("A1g") == ["A1g"] and rows.get("Eg") == ["A1g + B1g"]
           and rows.get("T2g") == ["B2g + Eg"] and rows.get("T1u") == ["A2u + Eu"], out)
    report("--correlate --pg: * Point group * / * Subgroup * / * Correlation table * "
           "blocks, each after one blank line",
           code == 0 and out.startswith("\n* Point group *\nm-3m (order 48)\n")
           and "\n\n* Subgroup *\n4/mmm (order 16, index 3): 1 inequivalent orientation\n"
           in out
           and "  4/mmm (4 || z): 3 conjugate subgroups; C2' = 2_x, 2_y; "
           "C2'' = 2_[110], 2_[1-10]" in out
           and "\n\n* Correlation table *\nirrep" in out and "\nnote: x, y, z" in out, out)
    code, out = run_group(["--correlate", "--pg", "m-3m", "--subgroup", "-3m"])
    rows = table_rows(out)
    report("m-3m -> -3m (3 || [111]): T2g -> A1g + Eg, T1u -> A2u + Eu",
           code == 0 and rows.get("irrep", [""])[0].startswith("-3m (-3 || [111]")
           and rows.get("T2g") == ["A1g + Eg"] and rows.get("T1u") == ["A2u + Eu"], out)
    code, out = run_group(["--correlate", "--pg", "-43m", "--subgroup", "3m"])
    rows = table_rows(out)
    report("-43m (Td) -> 3m (C3v): T2 -> A1 + E, T1 -> A2 + E",
           code == 0 and rows.get("T2") == ["A1 + E"] and rows.get("T1") == ["A2 + E"], out)
    code, out = run_group(["--correlate", "--pg", "m-3m", "--subgroup", "mm2"])
    rows = table_rows(out)
    report("m-3m -> mm2: three orientations (2 || z with axial or diagonal mirrors, "
           "2 || [110])",
           code == 0 and "3 inequivalent orientations" in out
           and rows.get("irrep") == ["mm2 (2 || z, m_y)", "mm2 (2 || z, m_[110])",
                                     "mm2 (2 || [110])"]
           and rows.get("A2g") == ["A1", "A2", "B2"]
           and "  mm2 (2 || [110]): 6 conjugate subgroups; sgvxz = m_[1-10]; sgvyz = m_z"
           in out, out)
    code, out = run_group(["--correlate", "--pg", "6/mmm", "--subgroup", "mmm"])
    rows = table_rows(out)
    report("6/mmm -> mmm: hexagonal axes, E1g -> B2g + B3g, E2g -> Ag + B1g",
           code == 0 and rows.get("irrep") == ["mmm (2 || [001])"]
           and rows.get("E1g") == ["B2g + B3g"] and rows.get("E2g") == ["Ag + B1g"]
           and "C2x = 2_[100]; C2y = 2_[120]" in out
           and "directions [uvw] are in the hexagonal axes" in out, out)
    code, out = run_group(["--correlate", "--pg", "4/mmm", "--subgroup", "2/m"])
    rows = table_rows(out)
    report("4/mmm -> 2/m: one column per class of twofold axes (z, x, [110])",
           code == 0 and rows.get("irrep") == ["2/m (2 || z)", "2/m (2 || x)",
                                               "2/m (2 || [110])"]
           and rows.get("B1g") == ["Ag", "Ag", "Bg"] and rows.get("B2g") == ["Ag", "Bg", "Ag"]
           and rows.get("Eg") == ["2 Bg", "Ag + Bg", "Ag + Bg"], out)
    code, out = run_group(["--correlate", "--pg", "4/mmm", "--subgroup", "mmm"])
    rows = table_rows(out)
    report("4/mmm -> mmm: the diagonal class keeps C2z (2 || z, 2_[110]) and gives the "
           "D4h -> D2h(C2'') column A2g -> B1g, B1g -> B1g, B2g -> Ag, Eg -> B2g + B3g",
           code == 0 and rows.get("irrep") == ["mmm (2 || z, 2_x)", "mmm (2 || z, 2_[110])"]
           and [rows.get(k, ["", ""])[1] for k in ("A2g", "B1g", "B2g", "Eg")]
           == ["B1g", "B1g", "Ag", "B2g + B3g"]
           and [rows.get(k, ["", ""])[0] for k in ("A2g", "B1g", "B2g", "Eg")]
           == ["B1g", "Ag", "B1g", "B2g + B3g"]
           and "C2x = 2_[110]; C2y = 2_[1-10]" in out, out)
    code, out = run_group(["--correlate", "--pg", "m-3m", "--subgroup", "23"])
    report("m-3m -> 23 and m-3m -> -3m: no setting name off G's axes, and the note "
           "drops the B1/B2/B3 clause when no class is listed",
           code == 0 and "B1/B2/B3" not in out
           and "symbols of\n      H are those of phonopy's table of H." in out
           and table_rows(run_group(["--correlate", "--pg", "m-3m", "--subgroup", "-3m"])[1])
           .get("irrep") == ["-3m (-3 || [111])"], out)
    code, out = run_group(["--correlate", "--pg", "4/mmm", "--subgroup", "-42m"])
    rows = table_rows(out)
    report("4/mmm -> -42m: both phonopy settings named (B1g -> B1 / B2)",
           code == 0 and rows.get("irrep") == ["-42m (-4 || z, -42m setting)",
                                               "-42m (-4 || z, -4m2 setting)"]
           and rows.get("B1g") == ["B1", "B2"], out)
    code, out = run_group(["--correlate", "--pg", "m-3m", "--subgroup", "6"])
    report("a non-subgroup (m-3m -> 6) gives a one-line error",
           code != 0 and "ERROR: 6 is not a subgroup of m-3m (its subgroup types: 1, -1"
           in out and len(out.strip().splitlines()) == 1 and "Traceback" not in out, out)

    # full tables against Altmann and Herzig (Point-Group Theory Tables) /
    # Bilbao CORREL, through the API; then the sum rule over all 32 groups
    code, out = run_python(
        "from crystod import group\n"
        "ref = {\n"
        " '4/mmm': 'A1g|B1g|A1g+B1g|A2g+Eg|B2g+Eg|A1u|B1u|A1u+B1u|A2u+Eu|B2u+Eu',\n"
        " '-3m': 'A1g|A2g|Eg|A2g+Eg|A1g+Eg|A1u|A2u|Eu|A2u+Eu|A1u+Eu',\n"
        " '4mm': 'A1|B1|A1+B1|A2+E|B2+E|A2|B2|A2+B2|A1+E|B1+E',\n"
        " '3m': 'A1|A2|E|A2+E|A1+E|A2|A1|E|A1+E|A2+E',\n"
        " 'mmm': 'Ag|Ag|2Ag|B1g+B2g+B3g|B1g+B2g+B3g|Au|Au|2Au|B1u+B2u+B3u|B1u+B2u+B3u',\n"
        "}\n"
        "bad = []\n"
        "for H, text in ref.items():\n"
        "    table = group.correlation_table('m-3m', H)\n"
        "    for irrep, want in zip(table.rows, text.split('|')):\n"
        "        got = '+'.join((str(n) if n > 1 else '') + s\n"
        "                       for s, n in table.rows[irrep][0])\n"
        "        if got != want: bad.append((H, irrep, got, want))\n"
        "print('REF', len(ref), bad)\n"
        "from phonopy.phonon.character_table import character_table as T\n"
        "from crystod import correlation as c\n"
        "pairs = columns = failed = 0\n"
        "for G in T:\n"
        "    dimG = {k: round(abs(v[0] if hasattr(v, '__len__') else v))\n"
        "            for k, v in T[G][0]['character_table'].items()}\n"
        "    for H in c._subgroup_census(G).by_type:\n"
        "        table = group.correlation_table(G, H)\n"
        "        pairs += 1; columns += len(table.orientations)\n"
        "        for j, setting in enumerate(table.settings):\n"
        "            s = c._SETTING_NAMES[H].index(setting) if setting else 0\n"
        "            dimH = {k: round(abs(v[0] if hasattr(v, '__len__') else v))\n"
        "                    for k, v in T[H][s]['character_table'].items()}\n"
        "            for irrep, decs in table.rows.items():\n"
        "                if sum(n * dimH[x] for x, n in decs[j]) != dimG[irrep]:\n"
        "                    failed += 1\n"
        "census = {g: (sum(len(m) for v in c._subgroup_census(g).by_type.values() for m in v),\n"
        "              sum(len(v) for v in c._subgroup_census(g).by_type.values()))\n"
        "          for g in ('m-3m', '-43m', '4/mmm', 'mmm')}\n"
        "print('PGSWEEP', pairs, columns, failed, census)\n"
    )
    report("m-3m -> 4/mmm, -3m, 4mm, 3m, mmm equal the Altmann-Herzig tables",
           code == 0 and "REF 5 []" in out, out)
    match = re.search(r"PGSWEEP (\d+) (\d+) (\d+) (.*)", out)
    report("sum rule for every irrep of all 32 point groups over every subgroup type "
           "and orientation (222 pairs, 279 columns); Oh has 98 subgroups in 33 classes",
           code == 0 and match is not None and match.group(1) == "222"
           and match.group(2) == "279" and match.group(3) == "0"
           and "'m-3m': (98, 33)" in match.group(4) and "'-43m': (30, 11)" in match.group(4)
           and "'4/mmm': (35, 27)" in match.group(4) and "'mmm': (16, 16)" in match.group(4),
           out)
    code, out = run_python(
        "from crystod import group\n"
        "try:\n"
        "    group.correlation_table('m-3m', '6')\n"
        "except ValueError as exc:\n"
        "    print('VALUEERROR', exc)\n"
    )
    report("API: correlation_table raises ValueError for a non-subgroup",
           code == 0 and "VALUEERROR 6 is not a subgroup of m-3m" in out, out)

    # error paths of the point-group form
    code, out = run_group(["--correlate", "--pg", "m-3m"])
    report("--correlate --pg without --subgroup rejected cleanly",
           code != 0 and "--correlate --pg requires --subgroup H" in out
           and "Traceback" not in out, out)
    code, out = run_group(["--correlate", "--pg", "m-3m", "--subgroup", "4/mmm",
                           "--kpoint", "GM"])
    report("--kpoint with --correlate --pg rejected cleanly",
           code != 0 and "are not used with --correlate --pg G" in out
           and "Traceback" not in out, out)
    code, out = run_group(["--correlate", "--pg", "m-3m", "--subgroup", "4mmm"])
    report("unknown subgroup symbol rejected cleanly (the point groups are listed)",
           code != 0 and '"4mmm" is not in the point groups' in out
           and "Traceback" not in out, out)


# ---------------------------------------------------------------- 24. crystod-group extras
def test_24_group_command() -> None:
    print("\n[24] crystod-group (sectioned command: 12 modes)")

    def run_group(args: list[str], cwd: str | None = None) -> tuple[int, str]:
        return run_module("crystod.cli.group", args, cwd)

    code, out = run_group(["--product", "T2g", "T2g", "--pg", "m-3m"])
    report("--product T2g T2g exit 0", code == 0, out)
    report("T2g x T2g = A1g + Eg + T1g + T2g",
           all(f"({name})" in out for name in ("A1g", "Eg", "T1g", "T2g")), out)

    # --pointgroup/--spacegroup aliases and space-group numbers
    code, out = run_group(["--ligand-field", "d", "--pointgroup", "m-3m"])
    report("--pointgroup alias accepted",
           code == 0 and "1(Eg) + 1(T2g)" in out, out)
    code, out = run_group(["--basis", "x", "y", "z", "--spacegroup", "221",
                           "--kpoint", "0", "0", "0"])
    report("--spacegroup alias + space-group number accepted",
           code == 0 and "GM4-" in out, out)
    code, out = run_group(["--table", "--pointgroup=-43m"])
    report("--pointgroup with leading-dash label accepted",
           code == 0 and "-43m" in out, out)

    # --supergroup is the backward-compatible alias of --parent (the value is
    # the parent group in both spellings)
    code, out = run_group(["--parent", "Pm-3m", "--irrep", "GM4-",
                           "--order-parameter", "0", "0", "a"])
    code_alias, out_alias = run_group(["--supergroup", "Pm-3m", "--irrep", "GM4-",
                                       "--order-parameter", "0", "0", "a"])
    report("--parent accepted",
           code == 0 and "P4mm" in out, out)
    report("--supergroup alias gives the same output as --parent",
           code == code_alias and out == out_alias, out)
    code, out = run_group(["--parent", "Pm-3m"])
    report("--parent without --irrep rejected cleanly",
           code != 0 and "requires --irrep" in out and "Traceback" not in out, out)

    code, out = run_group(["--product", "T2g", "T2g", "T1u", "--pg", "m-3m",
                           "--show-irrep-table"])
    report("triple product with --show-irrep-table", code == 0 and "(A2u)" in out, out)

    code, out = run_group(["--table", "--pg", "3m"])
    report("--table shows character table of 3m",
           code == 0 and "A1" in out and "E" in out, out)

    code, out = run_group(["--decompose", "--pg", "3m", "--characters", "3", "0", "1"])
    report("--decompose 3 0 1 in 3m -> A1 + E",
           code == 0 and "1(A1) + 1(E)" in out, out)

    code, out = run_group(["--ligand-field", "d", "--pg", "m-3m"])
    report("--ligand-field d in m-3m -> Eg + T2g",
           code == 0 and "1(Eg) + 1(T2g)" in out, out)

    code, out = run_group(["--ligand-field", "d", "--pg", "-43m"])
    report("dash point-group value (-43m) accepted",
           code == 0 and "1(E) + 1(T2)" in out, out)

    code, out = run_group(["--basis", "x", "y", "z", "--pg", "m-3m"])
    report("--basis x y z in m-3m -> T1u", code == 0 and "T1u" in out, out)

    code, out = run_group(["--basis", "x", "y", "z", "--sg", "Pm-3m",
                           "--kpoint", "0", "0", "0"])
    report("--basis x y z at GM in Pm-3m -> GM4-", code == 0 and "GM4-" in out, out)

    code, out = run_group(["--generate-basis", "--pg", "m-3m", "--order", "1"])
    report("--generate-basis order 1 -> T1u",
           code == 0 and "1st order" in out and "T1u" in out, out)

    code, out = run_group(["--coset", "--pg", "m-3m", "--subgroup", "4/mmm"])
    report("--coset point-group mode: index [G:H] = 3",
           code == 0 and "index [G:H] = 3" in out, out)

    code, out = run_group(["--coset", "--sg", "Pm-3m", "--kpoint", "0.5", "0.5", "0"])
    report("--coset space-group mode: little co-group at k",
           code == 0 and "G_k" in out, out)

    # irrep labels at a k point given in any of its forms (--table rows)
    def table_rows(text: str) -> list[str]:
        return [" ".join(line.split()) for line in text.splitlines()
                if line.startswith("irrep_")]

    def characters(rows: list[str]) -> dict:
        found = {}
        for row in rows:
            if " = " not in row:
                continue
            label, values = row.split(" = ", 1)[1].split(" ", 1)
            found[label.split("(")[0]] = [complex(value) for value in values.split()]
        return found

    # on a line the ISO-IR label is evaluated at the canonical line parameter
    # (the smallest |parameter|, positive first), the same for every arm of the
    # star and every k + G: (-1/4,-1/4,0) of F-43m and its copy (3/4,3/4,0) lie
    # on the arm (0,0,-a) at a = 1/2, where the characters (1,-1,-1,1) are DT4;
    # both used to be labelled as at +k
    tables = {}
    for kpoint in ("-1/4 -1/4 0", "3/4 3/4 0", "1/4 1/4 0"):
        tables[kpoint] = run_group(["--table", "--sg", "216", "--kpoint", *kpoint.split()])
    minus_rows = table_rows(tables["-1/4 -1/4 0"][1])
    report("--table F-43m DT at (-1/4,-1/4,0): DT4 (1,-1,-1,1) and DT3 (1,-1,1,-1)",
           tables["-1/4 -1/4 0"][0] == 0
           and "irrep_3(1) = DT4(1) 1 -1 -1 1" in minus_rows
           and "irrep_4(1) = DT3(1) 1 -1 1 -1" in minus_rows,
           tables["-1/4 -1/4 0"][1])
    report("--table F-43m DT at (3/4,3/4,0) = (-1/4,-1/4,0) + G: the same rows",
           tables["3/4 3/4 0"][0] == 0
           and table_rows(tables["3/4 3/4 0"][1]) == minus_rows,
           tables["3/4 3/4 0"][1])
    report("--table F-43m DT at (1/4,1/4,0): DT3 (1,-1,-1,1)",
           tables["1/4 1/4 0"][0] == 0
           and "irrep_3(1) = DT3(1) 1 -1 -1 1" in table_rows(tables["1/4 1/4 0"][1]),
           tables["1/4 1/4 0"][1])

    # -k of the P point of I-4 (only P is tabulated): the PA star, whose irreps
    # are the complex conjugates of those at P (the convention of --product,
    # M1 x P1 = PA1); it used to print irrep_N rows
    code, out = run_group(["--table", "--sg", "82", "--kpoint", "-1/4", "-1/4", "-1/4"])
    code_plus, out_plus = run_group(["--table", "--sg", "82", "--kpoint", "1/4", "1/4", "1/4"])
    minus, plus = characters(table_rows(out)), characters(table_rows(out_plus))
    report("--table I-4 at (-1/4,-1/4,-1/4): PA star with rows PA1, PA4, PA2, PA3",
           code == 0 and " PA [-0.25, -0.25, -0.25]" in out
           and list(minus) == ["PA1", "PA4", "PA2", "PA3"], out)
    report("PAn is the complex conjugate of Pn",
           code_plus == 0 and sorted(plus) == ["P1", "P2", "P3", "P4"]
           and all(f"PA{label[1:]}" in minus
                   and np.allclose(minus[f"PA{label[1:]}"], np.conj(values))
                   for label, values in plus.items())
           and not np.allclose(minus["PA4"], plus["P4"]),
           out + out_plus)

    # the -k partner of a line or plane star carries ISOTROPY's name, with
    # the conjugate irreps: DU of DT (P6_3), PC of P (P3, where the suffix
    # is C), GQ of the general point GP
    code, out = run_group(["--table", "--sg", "173", "--kpoint", "0", "0", "7/12"])
    code_plus, out_plus = run_group(["--table", "--sg", "173", "--kpoint", "0", "0", "5/12"])
    minus, plus = characters(table_rows(out)), characters(table_rows(out_plus))
    report("--table P6_3 at (0,0,7/12): the DU star, DUn = conj DTn at (0,0,5/12)",
           code == 0 and code_plus == 0 and " DU [0.0, 0.0, 0.583333]" in out
           and sorted(minus) == [f"DU{n}" for n in range(1, 7)]
           and sorted(plus) == [f"DT{n}" for n in range(1, 7)]
           and all(np.allclose(minus[f"DU{label[2:]}"], np.conj(values))
                   for label, values in plus.items()),
           out + out_plus)
    code, out = run_group(["--table", "--sg", "143", "--kpoint", "2/3", "2/3", "-1/4"])
    code_plus, out_plus = run_group(["--table", "--sg", "143", "--kpoint", "1/3", "1/3", "1/4"])
    minus, plus = characters(table_rows(out)), characters(table_rows(out_plus))
    report("--table P3 at (2/3,2/3,-1/4): the PC star of P (C suffix in P3)",
           code == 0 and " PC [0.666667, 0.666667, -0.25]" in out
           and sorted(minus) == ["PC1", "PC2", "PC3"]
           and all(np.allclose(minus[f"PC{label[1:]}"], np.conj(values))
                   for label, values in plus.items()),
           out + out_plus)
    code, out = run_group(["--table", "--sg", "1", "--kpoint", "-0.1", "-0.2", "-0.3"])
    code_plus, out_plus = run_group(["--table", "--sg", "1", "--kpoint", "0.1", "0.2", "0.3"])
    report("--table P1: GP at (0.1,0.2,0.3), its -k partner GQ at (-0.1,-0.2,-0.3)",
           code == 0 and code_plus == 0 and " GQ [-0.1, -0.2, -0.3]" in out
           and " GP [0.1, 0.2, 0.3]" in out_plus, out + out_plus)
    code, out = run_group(["--table", "--sg", "174", "--kpoint", "-1/4", "-1/8", "0"])
    code_plus, out_plus = run_group(["--table", "--sg", "174", "--kpoint", "1/4", "1/8", "0"])
    report("--table P-6: plane B at (1/4,1/8,0), its -k partner BC (C suffix in P-6)",
           code == 0 and code_plus == 0 and " BC [-0.25, -0.125, 0.0]" in out
           and " B [0.25, 0.125, 0.0]" in out_plus, out + out_plus)
    # the -k partner of the line P of P3 is the line through K' (PC), so
    # both signs of the parameter of P itself keep the name P
    code, out = run_group(["--table", "--sg", "143", "--kpoint", "1/3", "1/3", "-1/4"])
    report("--table P3 at (1/3,1/3,-1/4): still P (the partner PC is another line)",
           code == 0 and " P [0.333333, 0.333333, -0.25]" in out
           and "PC" not in out, out)
    # a centrosymmetric group has no -k partners
    code, out = run_group(["--table", "--sg", "221", "--kpoint", "-0.1", "-0.2", "-0.35"])
    report("--table Pm-3m at a negative general point: GP (no partner with inversion)",
           code == 0 and " GP [-0.1, -0.2, -0.35]" in out and "GQ" not in out, out)

    # the rhombohedral T point is found for every copy of k (T + (1,1,1) used
    # to be printed without a name and with irrep_N rows)
    code, out = run_group(["--table", "--sg", "166", "--kpoint", "3/2", "3/2", "3/2"])
    report("--table R-3m at (3/2,3/2,3/2) = T + (1,1,1): T1+, T1-, ...",
           code == 0 and " T [1.5, 1.5, 1.5]" in out
           and "irrep_1(1) = T1+(1)" in out and "irrep_2(1) = T1-(1)" in out, out)

    # error handling
    code, out = run_group(["--product", "T2g", "T2g"])
    report("--product without --pg/--sg rejected cleanly",
           code != 0 and "exactly one" in out and "Traceback" not in out, out)
    code, out = run_group(["--basis", "x", "--pg", "m-3m", "--sg", "Pm-3m"])
    report("--basis with both --pg and --sg rejected cleanly",
           code != 0 and "exactly one" in out and "Traceback" not in out, out)
    code, out = run_group(["--coset", "--pg", "m-3m"])
    report("--coset without --subgroup rejected cleanly",
           code != 0 and "requires --subgroup" in out and "Traceback" not in out, out)
    code, out = run_group(["--generate-basis", "--sg", "Pm-3m"])
    report("--generate-basis --sg without --kpoint rejected cleanly",
           code != 0 and "requires --kpoint" in out and "Traceback" not in out, out)

    # removed flat flags give replacement guidance
    for flag in ("--direct-product", "--ligand-field-split", "--show-coset",
                 "--basis-function", "--generate-basis-function", "--decompose-irrep"):
        code, out = run_cli([flag])
        report(f"removed {flag} flag points to crystod-group",
               code != 0 and "is not a crystod option" in out and "crystod-group" in out, out)


# ---------------------------------------------------------------- 25. crystod-bz
def test_25_bz() -> None:
    print("\n[25] crystod-bz (Brillouin-zone plot)")
    with tempfile.TemporaryDirectory() as tmp:
        html = os.path.join(tmp, "BZ_ScF3.html")
        code, out = run_bz(["-c", POSCAR_ScF3, "--output", html])
        report("auto k-path exit 0", code == 0, out)
        report("space group detected (Pm-3m #221)", "Pm-3m" in out and "221" in out, out)
        report("seekpath k-path printed",
               "GAMMA" in out and all(label in out for label in ("X", "M", "R")), out)
        exists = os.path.isfile(html)
        report("HTML file created", exists)
        if exists:
            text = open(html).read()
            report("HTML contains plotly + BZ traces",
                   "plotly" in text and "scatter3d" in text and "goldenrod" in text)
            report("Gamma label present", "\\u0393" in text or "\u0393" in text)

        # default output name: BZ_{POSCAR name}.html in cwd
        code, out = run_bz(["-c", POSCAR_ScF3], cwd=tmp)
        default_html = os.path.join(tmp, "BZ_221_PPOSCAR_ScF3.html")
        report("default output name exit 0", code == 0, out)
        report("BZ_221_PPOSCAR_ScF3.html auto-created", os.path.isfile(default_html))

        # manual --band/--label mode
        html_manual = os.path.join(tmp, "BZ_manual.html")
        code, out = run_bz(
            ["-c", POSCAR_ScF3,
             "--band", "0 0 0  0 1/2 0  1/2 1/2 0  0 0 0  1/2 1/2 1/2  0 1/2 0, 1/2 1/2 0  1/2 1/2 1/2",
             "--label", "GM X M GM R X M R",
             "--output", html_manual]
        )
        report("manual --band/--label exit 0", code == 0, out)
        report("manual path: 2 segments", "2 segment" in out, out)
        report("manual HTML created", os.path.isfile(html_manual))

        # error handling: label count mismatch
        code, out = run_bz(
            ["-c", POSCAR_ScF3,
             "--band", "0 0 0  1/2 1/2 1/2", "--label", "GM", "--output", html_manual]
        )
        report("label count mismatch rejected cleanly",
               code != 0 and "ERROR" in out and "Traceback" not in out, out)


# ---------------------------------------------------------------- 26. crystod-bz --trans-mat
def test_26_bz_supercell() -> None:
    print("\n[26] crystod-bz --trans-mat (ScF3, Pm-3m -> transformed lattice)")
    with tempfile.TemporaryDirectory() as tmp:
        code, out = run_bz(
            ["-c", POSCAR_ScF3,
             "--trans-mat", "0 1 2   -1 0 2   1 -1 2",
             "--output", os.path.join(tmp, "BZ_supercell.html")],
            cwd=tmp,
        )
        report("exit code 0", code == 0, out)
        report("volume ratio |det T| = 6 reported", "|det T| = 6" in out, out)
        report("6 folded Gamma points listed",
               "folding onto the supercell Gamma point (6)" in out, out)
        html_path = os.path.join(tmp, "BZ_supercell.html")
        report("HTML written", os.path.isfile(html_path))
        if os.path.isfile(html_path):
            text = open(html_path).read()
            report("HTML contains plotly traces", "Plotly.newPlot" in text and "scatter3d" in text,
                   text[:300])

        code, out = run_bz(
            ["-c", POSCAR_ScF3, "--trans-mat", "1 0 0  0 1 0"],
            cwd=tmp,
        )
        report("wrong matrix size rejected cleanly",
               code != 0 and "requires nine numbers" in out and "Traceback" not in out, out)


# ---------------------------------------------------------------- 27. crystod-bz extras
def test_27_bz_command() -> None:
    print("\n[27] crystod-bz (sectioned command: unit-cell / supercell BZ)")

    def run_bz(args: list[str], cwd: str | None = None) -> tuple[int, str]:
        return run_module("crystod.cli.bz", args, cwd)

    with tempfile.TemporaryDirectory() as tmp:
        html = os.path.join(tmp, "BZ_new.html")
        code, out = run_bz(["-c", POSCAR_ScF3, "--output", html])
        report("-c auto k-path exit 0", code == 0, out)
        report("space group detected (Pm-3m #221)", "Pm-3m" in out and "221" in out, out)
        report("HTML file created", os.path.isfile(html))

        code, out = run_bz(["--poscar", POSCAR_ScF3, "--output", html])
        report("--poscar alias accepted", code == 0, out)

        html_identity = os.path.join(tmp, "BZ_identity.html")
        code, out = run_bz(
            ["-c", POSCAR_ScF3, "--trans-mat", "1 0 0  0 1 0  0 0 1",
             "--output", html_identity]
        )
        report("explicit identity trans-mat -> unit-cell BZ mode",
               code == 0 and "|det T|" not in out, out)
        report("identity HTML created", os.path.isfile(html_identity))

        html_manual = os.path.join(tmp, "BZ_manual.html")
        code, out = run_bz(
            ["-c", POSCAR_ScF3, "--band", "0 0 0  1/2 1/2 1/2",
             "--band-labels", "GM R", "--output", html_manual]
        )
        report("--band/--band-labels exit 0", code == 0, out)
        report("manual HTML created", os.path.isfile(html_manual))

        html_super = os.path.join(tmp, "BZ_super.html")
        code, out = run_bz(
            ["-c", POSCAR_ScF3, "--trans-mat", "0 1 2   -1 0 2   1 -1 2",
             "--output", html_super]
        )
        report("non-identity trans-mat -> supercell BZ mode exit 0", code == 0, out)
        report("volume ratio |det T| = 6 reported", "|det T| = 6" in out, out)
        report("6 folded Gamma points listed",
               "folding onto the supercell Gamma point (6)" in out, out)
        report("supercell HTML created", os.path.isfile(html_super))

        code, out = run_bz(
            ["-c", POSCAR_ScF3, "--trans-mat", "2 0 0  0 2 0  0 0 2",
             "--band", "0 0 0  1/2 1/2 1/2"]
        )
        report("--band with non-identity trans-mat rejected cleanly",
               code != 0 and "unit-cell BZ mode" in out and "Traceback" not in out, out)

        code, out = run_bz(["-c", POSCAR_ScF3, "--trans-mat", "1 0 0  0 1 0"])
        report("wrong matrix size rejected cleanly",
               code != 0 and "nine numbers" in out and "Traceback" not in out, out)

        # removed flat flags give replacement guidance
        for flag in ("--bz", "--bz-supercell"):
            code, out = run_cli([flag])
            report(f"removed {flag} flag points to crystod-bz",
                   code != 0 and "is not a crystod option" in out and "crystod-bz" in out, out)

        # --show-kpoint: special k points of a space group (CDML convention)
        code, out = run_bz(["--show-kpoint", "--space-group", "Pnma"])
        report("--show-kpoint Pnma exit 0", code == 0, out)
        report("Pnma primitive k points listed",
               "Pnma (No. 62)" in out and "* K points (primitive) *" in out
               and "X: (1/2, 0, 0)" in out and "R: (1/2, 1/2, 1/2)" in out, out)
        report("Pnma (P lattice) prints no conventional section",
               "(conventional)" not in out, out)

        code, out = run_bz(["--show-kpoint", "--space-group", "Fm-3m"])
        report("--show-kpoint Fm-3m primitive + conventional",
               code == 0 and "X: (1/2, 0, 1/2)" in out and "W: (1/2, 1/4, 3/4)" in out
               and "* K points (conventional) *" in out and "X: (0, 1, 0)" in out, out)

        # space-group number and the --sg/--spacegroup aliases
        code, out = run_bz(["--show-kpoint", "--sg", "221"])
        report("--show-kpoint --sg 221 (number + alias)",
               code == 0 and "Pm-3m (No. 221)" in out
               and "R: (1/2, 1/2, 1/2)" in out, out)
        code, out = run_bz(["--show-kpoint", "--spacegroup", "Fm-3m"])
        report("--show-kpoint --spacegroup alias",
               code == 0 and "Fm-3m (No. 225)" in out, out)

        code, out = run_bz(["--show-kpoint"])
        report("--show-kpoint without --space-group rejected cleanly",
               code != 0 and "requires --space-group" in out and "Traceback" not in out, out)

        code, out = run_bz(["--space-group", "Pnma"])
        report("--space-group without --show-kpoint rejected cleanly",
               code != 0 and "only available with --show-kpoint" in out
               and "Traceback" not in out, out)

        code, out = run_bz(["--show-kpoint", "--space-group", "NotASpaceGroup"])
        report("unknown space-group symbol rejected cleanly",
               code != 0 and "not recognized" in out and "Traceback" not in out, out)


    # --example: the bundled ScF3 cell, copied into the working directory
    code, out = run_bz(["--example"])
    report("--example alone lists the bundled examples",
           code == 0 and "ScF3" in out and "crystod-bz -c 221_PPOSCAR_ScF3" in out, out)
    with tempfile.TemporaryDirectory() as tmp:
        code, out = run_bz(["--example", "ScF3"], cwd=tmp)
        report("--example ScF3 writes the cell and BZ_221_PPOSCAR_ScF3.html",
               code == 0 and "Running: crystod-bz -c 221_PPOSCAR_ScF3" in out
               and os.path.isfile(os.path.join(tmp, "221_PPOSCAR_ScF3"))
               and os.path.isfile(os.path.join(tmp, "BZ_221_PPOSCAR_ScF3.html")), out)

# ---------------------------------------------------------------- 28. crystod-phonon --irreps
def test_28_phonon_irrep() -> None:
    print("\n[28] crystod-phonon --irreps (SrTiO3, 4x4x4 FORCE_SETS)")
    if not os.path.isdir(PHONON_IRREP_DIR):
        report("example data found", False, PHONON_IRREP_DIR)
        return
    with tempfile.TemporaryDirectory() as tmp:
        # copy inputs so phonon_irreps.yaml in the example folder is not overwritten
        # (--readfc / FORCE_CONSTANTS input is covered by the Si runs in
        # sections 32 and 35; the SrTiO3 example ships FORCE_SETS only)
        for name in ("221_PPOSCAR_SrTiO3", "FORCE_SETS"):
            shutil.copy(os.path.join(PHONON_IRREP_DIR, name), tmp)
        code, out = run_phonon(
            ["--irreps", "--dim", "4 4 4", "-c", "221_PPOSCAR_SrTiO3"],
            cwd=tmp,
        )
        report("exit code 0", code == 0, out)
        yaml_path = os.path.join(tmp, "phonon_irreps.yaml")
        report("phonon_irreps.yaml written", os.path.isfile(yaml_path))
        if os.path.isfile(yaml_path):
            text = open(yaml_path).read()
            report("yaml contains GM point irreps", "GM" in text, text[:500])
            report("yaml contains R point irreps", "R" in text, text[:500])
            report("default survey has no k-path midpoints",
                   "path_midpoints:" not in text and "midpoint of" not in text,
                   text[:800])

        # --all-irreps: additionally label the seekpath k-path midpoints,
        # written to phonon_irreps_all.yaml so both surveys can coexist
        code, out = run_phonon(
            ["--irreps", "--dim", "4 4 4", "-c", "221_PPOSCAR_SrTiO3",
             "--all-irreps"],
            cwd=tmp,
        )
        report("--all-irreps exit code 0", code == 0, out)
        all_yaml_path = os.path.join(tmp, "phonon_irreps_all.yaml")
        report("--all-irreps writes phonon_irreps_all.yaml",
               os.path.isfile(all_yaml_path)
               and "written to: phonon_irreps_all.yaml" in out, out)
        if os.path.isfile(yaml_path):
            report("--all-irreps leaves phonon_irreps.yaml midpoint-free",
                   "midpoint of" not in open(yaml_path).read())
        if os.path.isfile(all_yaml_path):
            text = open(all_yaml_path).read()
            report("--all-irreps yaml contains the seekpath k-path",
                   "k_path:" in text, text[:800])
            report("--all-irreps yaml lists the k-path midpoints (ISO-IR k types)",
                   "path_midpoints:" in text
                   and "DT (midpoint of GM-X)" in text
                   and "T (midpoint of R-M)" in text, text[:1200])
            report("--all-irreps midpoint irreps labeled via ISO-IR",
                   "segment: GM-X" in text and "DT5(2)" in text
                   and "T5(2)" in text, text)

    # Si (Fd-3m, two origins on the atoms): the irreps are named in the frame
    # with Si on 8a, the same frame SALC and --vibration use; the L-W line (Q)
    # of --all-irreps follows it (the shipped survey once labelled the points
    # in one frame and the line in the other: Q2, Q1, Q1, Q2, Q1, Q2)
    si_dir = os.path.join(ROOT, "example", "28_phonon_irrep", "Si_Fd-3m")

    def labels_by_point(text: str) -> dict:
        found = {}
        for block in text.split("\n- q_label: ")[1:]:
            found.setdefault(block.split()[0], re.findall(
                r"irrep_label: \['([^']+)'\]", block))
        return found

    with tempfile.TemporaryDirectory() as tmp:
        for name in ("227_PPOSCAR_Si", "FORCE_SETS"):
            shutil.copy(os.path.join(si_dir, name), tmp)
        code, out = run_phonon(["--irreps", "--dim", "4 4 4", "-c", "227_PPOSCAR_Si"],
                               cwd=tmp)
        yaml_path = os.path.join(tmp, "phonon_irreps.yaml")
        found = labels_by_point(open(yaml_path).read()) if os.path.isfile(yaml_path) else {}
        report("Si --irreps: L3+, L2-, L1+, L3- at L and W2, W1, W2 at W",
               code == 0
               and found.get("L") == ["L3+(2)", "L2-(1)", "L1+(1)", "L3-(2)"]
               and found.get("W") == ["W2(2)", "W1(2)", "W2(2)"], f"{found}\n{out}")
        code, out = run_phonon(["--irreps", "--dim", "4 4 4", "-c", "227_PPOSCAR_Si",
                                "--all-irreps"], cwd=tmp)
        all_yaml_path = os.path.join(tmp, "phonon_irreps_all.yaml")
        found = (labels_by_point(open(all_yaml_path).read())
                 if os.path.isfile(all_yaml_path) else {})
        report("Si --all-irreps: Q1, Q2, Q2, Q1, Q2, Q1 on the L-W line, in the frame "
               "of the L and W labels",
               code == 0
               and found.get("Q") == ["Q1(1)", "Q2(1)", "Q2(1)", "Q1(1)", "Q2(1)", "Q1(1)"]
               and found.get("W") == ["W2(2)", "W1(2)", "W2(2)"], f"{found}\n{out}")

    # the k-path of --all-irreps keeps seekpath's names, but an endpoint that
    # the frame of the labels gives another ISO-IR type than spglib's frame
    # is printed with its ISO-IR name: seekpath's P of AlPO4 at a generic
    # origin is the PA of the labels (the path read GM-X-P-N for it), and the
    # lines through it are the -k partners WA, QA and BA of W, Q and B
    probe = (
        "import warnings\n"
        "warnings.simplefilter('ignore')\n"
        "import numpy as np\n"
        "from phonopy import Phonopy\n"
        "from phonopy.interface.calculator import read_crystal_structure\n"
        "from phonopy.structure.atoms import PhonopyAtoms\n"
        "from crystod.phonon_irreps import _seekpath_path_midpoints\n"
        f"path = {os.path.join(ROOT, 'example', 'test_POSCARs', '82_PPOSCAR_AlPO4')!r}\n"
        "cell, _ = read_crystal_structure(path, interface_mode='vasp')\n"
        "shifted = PhonopyAtoms(cell=cell.cell, numbers=cell.numbers, scaled_positions=(\n"
        "    cell.scaled_positions + [0.0731, 0.1593, 0.2417]) % 1.0)\n"
        "for unit in (cell, shifted):\n"
        "    ph = Phonopy(unit, supercell_matrix=np.eye(3, dtype=int), primitive_matrix='auto')\n"
        "    string, midpoints = _seekpath_path_midpoints(ph)\n"
        "    print('PATH', string)\n"
        "    print('SEG', ' '.join(f'{label}:{segment}' for label, segment, _ in midpoints))\n"
    )
    code, out = run_python(probe)
    paths = re.findall(r"^PATH (.*)$", out, re.M)
    segments = re.findall(r"^SEG (.*)$", out, re.M)
    report("--all-irreps k-path: GM-X-P-N for AlPO4, GM-X-PA-N (WA on X-PA, QA on "
           "PA-N) at a generic origin",
           code == 0 and len(paths) == 2 and len(segments) == 2
           and paths[0].startswith("GM-X-P-N-") and paths[1].startswith("GM-X-PA-N-")
           and "W:X-P Q:P-N B:N-GM" in segments[0]
           and "WA:X-PA QA:PA-N BA:N-GM" in segments[1]
           and paths[0].replace("-P-", "-PA-") == paths[1], out)


# ------------------------------------------------- 29. crystod-phonon spectroscopic activity
def test_29_phonon_activity() -> None:
    print("\n[29] crystod-phonon IR/Raman/silent/acoustic activity at Gamma, Raman tensors, "
          "Wyckoff orbits, BORN dielectric response, explicit --nac")
    irrep_dir = os.path.join(ROOT, "example", "28_phonon_irrep")
    born_path = os.path.join(PHONON_FATBAND_DIR, "BORN")

    def gamma_block(text: str) -> list:
        """(bands, irrep_label, activity) of every degenerate set at GM."""
        block = text.split("\n- q_label: GM\n", 1)[1].split("\n\n", 1)[0]
        return re.findall(
            r"  - # ([\d ]+)\n    irrep_label: \[([^\]]*)\]\n(?:    mulliken: [^\n]+\n)?"
            r"    frequency: +\S+\n    activity: \[([^\]]*)\]", block)

    def mulliken_lines(text: str) -> list:
        """(bands, irrep_label, mulliken) of every degenerate set at GM."""
        block = text.split("\n- q_label: GM\n", 1)[1].split("\n\n", 1)[0]
        return re.findall(r"  - # ([\d ]+)\n    irrep_label: \[([^\]]*)\]\n"
                          r"    mulliken: ([^\n]+)\n", block)

    def strip_activity(text: str) -> str:
        return "".join(line for line in text.splitlines(True)
                       if not re.match(r"\s*(activity|activity_summary|mulliken|nac"
                                       r"|dielectric_electronic"
                                       r"|dielectric_static|mode_effective_charge"
                                       r"|dielectric_contribution):", line))

    def irreps_run(name: str, poscar: str, extra: tuple = (), files: tuple = ()):
        with tempfile.TemporaryDirectory() as tmp:
            for path in (os.path.join(irrep_dir, name, poscar),
                         os.path.join(irrep_dir, name, "FORCE_SETS")) + files:
                shutil.copy(path, tmp)
            code, out = run_phonon(["--irreps", "--dim", "4 4 4", "-c", poscar, *extra],
                                   cwd=tmp)
            yaml_path = os.path.join(tmp, "phonon_irreps.yaml")
            text = open(yaml_path).read() if os.path.isfile(yaml_path) else ""
        return code, out, text

    # 1. SrTiO3 Pm-3m: 4 GM4- (one acoustic) + GM5-; GM4- IR, GM5- silent, no Raman
    code, out, text = irreps_run("SrTiO3_Pm-3m", "221_PPOSCAR_SrTiO3")
    summary = "GM4- x4 (3 IR, 1 acoustic), GM5- x1 (silent)"
    report("SrTiO3 --irreps exit 0, Gamma summary on the terminal with Mulliken symbols",
           code == 0 and "\n* Gamma-point activity *\nGM4- [T1u] x4 (3 IR, 1 acoustic)\n"
           "GM5- [T2u] x1 (silent)\n" in out, out)
    report("SrTiO3 yaml: a mulliken line after irrep_label in every GM set, nowhere else",
           mulliken_lines(text) == [("1 2 3", "'GM4-(3)'", "T1u"), ("4 5 6", "'GM4-(3)'", "T1u"),
                                    ("7 8 9", "'GM4-(3)'", "T1u"),
                                    ("10 11 12", "'GM5-(3)'", "T2u"),
                                    ("13 14 15", "'GM4-(3)'", "T1u")]
           and text.count("    mulliken: ") == 5, text[-1500:])
    report("SrTiO3 Gamma: acoustic GM4-, 3 IR GM4-, silent GM5- (T1u/T2u of m-3m)",
           gamma_block(text) == [("1 2 3", "'GM4-(3)'", "acoustic"),
                                 ("4 5 6", "'GM4-(3)'", "IR"),
                                 ("7 8 9", "'GM4-(3)'", "IR"),
                                 ("10 11 12", "'GM5-(3)'", "silent"),
                                 ("13 14 15", "'GM4-(3)'", "IR")], text[-1500:])
    report("SrTiO3: no Raman-active mode, summary and nac header in the yaml",
           "Raman" not in text and f"  activity_summary: {summary}" in text
           and "\nnac: false\n" in text, text[:600])
    report("activity lines only in the Gamma block (5 sets)",
           text.count("    activity: [") == 5, text)
    shipped = os.path.join(PHONON_IRREP_DIR, "phonon_irreps.yaml")
    report("SrTiO3 yaml equals the shipped example apart from the added lines",
           os.path.isfile(shipped)
           and strip_activity(text) == strip_activity(open(shipped).read()), text[:800])

    # 2. Si Fd-3m: GM5+ Raman only, GM4- acoustic; T2g Raman tensors (Loudon 1964)
    code, out, text = irreps_run("Si_Fd-3m", "227_PPOSCAR_Si", ("--raman-tensor",))
    report("Si Gamma: GM4- acoustic, GM5+ Raman only",
           code == 0
           and gamma_block(text) == [("1 2 3", "'GM4-(3)'", "acoustic"),
                                     ("4 5 6", "'GM5+(3)'", "Raman")]
           and "\n* Gamma-point activity *\nGM4- [T1u] x1 (acoustic)\nGM5+ [T2g] x1 (Raman)\n" in out
           and mulliken_lines(text) == [("1 2 3", "'GM4-(3)'", "T1u"),
                                        ("4 5 6", "'GM5+(3)'", "T2g")],
           f"{text[-800:]}\n{out}")
    report("Si --raman-tensor: GM5+ (T2g) = the yz, xz, xy pairs",
           "\n* Raman tensors (Cartesian axes of the input cell) *\n"
           "  GM5+ [T2g] (3 tensors):  [[0, 0, 0], [0, 0, a], [0, a, 0]]   "
           "[[0, 0, a], [0, 0, 0], [a, 0, 0]]   [[0, a, 0], [a, 0, 0], [0, 0, 0]]" in out, out)

    # 3. BaTiO3 P4mm: 3 A1 + B1 + 4 E optical; A1 (GM1) and E (GM5) IR and Raman,
    # B1 (GM2) Raman only; the acoustic set is GM1 + GM5 (all at 0 THz by
    # translation invariance)
    code, out, text = irreps_run("BaTiO3_P4mm", "99_PPOSCAR_BaTiO3", ("--raman-tensor",))
    report("BaTiO3 --raman-tensor: A1 diag(a,a,0)+diag(0,0,a), B1 diag(a,-a,0), E yz/xz",
           "  GM1 [A1] (2 tensors):  [[a, 0, 0], [0, a, 0], [0, 0, 0]]   "
           "[[0, 0, 0], [0, 0, 0], [0, 0, a]]\n"
           "  GM2 [B1] (1 tensor):  [[a, 0, 0], [0, -a, 0], [0, 0, 0]]\n"
           "  GM5 [E] (2 tensors):  [[0, 0, 0], [0, 0, a], [0, a, 0]]   "
           "[[0, 0, a], [0, 0, 0], [a, 0, 0]]" in out, out)
    sets = gamma_block(text)
    report("BaTiO3 yaml mulliken lines: GM1 = A1, GM2 = B1, GM5 = E, acoustic set A1 + E",
           ("3 4 5", "'GM1(1)', 'GM5(2)'", "A1 + E") in mulliken_lines(text)
           and ("10", "'GM2(1)'", "B1") in mulliken_lines(text)
           and ("1 2", "'GM5(2)'", "E") in mulliken_lines(text)
           and len(mulliken_lines(text)) == len(sets) == 9, text[-1500:])
    report("BaTiO3 Gamma summary: 4 IR+Raman GM5, 3 IR+Raman GM1, GM2 Raman, "
           "acoustic GM1 + GM5",
           code == 0 and "\n* Gamma-point activity *\nGM5 [E] x5 (4 IR+Raman, 1 acoustic)\n"
           "GM1 [A1] x4 (3 IR+Raman, 1 acoustic)\nGM2 [B1] x1 (Raman)\n" in out, out)
    report("BaTiO3: B1 = GM2 Raman only, A1 = GM1 and E = GM5 IR and Raman",
           ("10", "'GM2(1)'", "Raman") in sets
           and ("6", "'GM1(1)'", "IR, Raman") in sets
           and ("7 8", "'GM5(2)'", "IR, Raman") in sets
           and ("3 4 5", "'GM1(1)', 'GM5(2)'", "acoustic") in sets, f"{sets}")

    # NAC: BORN is read only with --nac (ScF3 + the BORN of example 22)
    code_a, out_a, plain = irreps_run("ScF3_Pm-3m", "221_PPOSCAR_ScF3")
    code_b, out_b, with_born = irreps_run("ScF3_Pm-3m", "221_PPOSCAR_ScF3",
                                          files=(born_path,))
    report("--irreps without --nac: a BORN file present changes nothing",
           code_a == 0 and code_b == 0 and plain == with_born and out_a == out_b
           and "NAC" not in out_b, f"{out_a}\n{out_b}")
    code, out, nac = irreps_run("ScF3_Pm-3m", "221_PPOSCAR_ScF3", ("--nac",),
                                files=(born_path,))
    report("--irreps --nac reads BORN (terminal note, nac: true in the yaml)",
           code == 0
           and "NAC: Born effective charges and dielectric tensor read from BORN." in out
           and "\nnac: true\n" in nac
           and strip_activity(nac) == strip_activity(plain), out)
    # BORN quantities: acoustic sum rule, eps_inf, eps_0, Lyddane-Sachs-Teller
    rule = re.search(r"sum_kappa Z\*_kappa: max \|deviation\| = (\S+)", out)
    eps_inf = re.search(r"eps_inf \(diag\): +(\S+) +(\S+) +(\S+)", out)
    eps_0 = re.search(r"eps_0 \(diag\): +(\S+) +(\S+) +(\S+)", out)
    lst = re.search(r"LST check: prod \(nu_LO/nu_TO\)\^2 = (\S+), eps_0/eps_inf = (\S+)", out)
    report("ScF3 --nac: acoustic sum rule |deviation| < 0.05, eps_inf 2.284",
           rule is not None and float(rule.group(1)) < 0.05 and eps_inf is not None
           and all(abs(float(value) - 2.284) < 5e-4 for value in eps_inf.groups()), out)
    report("ScF3 --nac: eps_0 > eps_inf, LST product = eps_0/eps_inf within 1 %",
           eps_0 is not None and eps_inf is not None and lst is not None
           and float(eps_0.group(1)) > float(eps_inf.group(1)) + 1.0
           and abs(float(lst.group(1)) / float(lst.group(2)) - 1.0) < 0.01, out)
    report("ScF3 --nac yaml: dielectric tensors in the header, Z~ and Delta eps per GM set",
           "\ndielectric_electronic: [[2.283939, 0.000000, 0.000000]," in nac
           and "\ndielectric_static: [[" in nac
           and nac.count("    mode_effective_charge: [[") == 4
           and nac.count("    dielectric_contribution: [") == 4, nac[:1500])
    probe = (
        "import warnings\n"
        "warnings.simplefilter('ignore')\n"
        "import numpy as np, phonopy\n"
        "from crystod import phonon\n"
        f"ph = phonopy.load(unitcell_filename={os.path.join(irrep_dir, 'ScF3_Pm-3m', '221_PPOSCAR_ScF3')!r},\n"
        f"    force_sets_filename={os.path.join(irrep_dir, 'ScF3_Pm-3m', 'FORCE_SETS')!r},\n"
        f"    born_filename={born_path!r}, supercell_matrix=[4, 4, 4],\n"
        "    primitive_matrix='auto', is_nac=True)\n"
        "acts = phonon.gamma_mode_activities(ph)\n"
        "ref = phonon.dielectric_response(ph, acts, lst_direction=None)\n"
        "vecs = np.array(ph.irreps.eigenvectors)\n"
        "rng = np.random.default_rng(1)\n"
        "for a in acts:\n"
        "    b = [i - 1 for i in a.band_indices]\n"
        "    z = rng.normal(size=(len(b), len(b))) + 1j * rng.normal(size=(len(b), len(b)))\n"
        "    vecs[:, b] = vecs[:, b] @ np.linalg.qr(z)[0]\n"
        "ph.irreps._eig_vecs = vecs\n"
        "rot = phonon.dielectric_response(ph, acts, lst_direction=None)\n"
        "dz = max(abs(x.charge_norm - y.charge_norm) for x, y in zip(ref.sets, rot.sets))\n"
        "print('INVARIANT', dz < 1e-6, abs(ref.eps_static - rot.eps_static).max() < 1e-6)\n"
        "from dataclasses import replace\n"
        "from crystod.phonon_activity import format_dielectric_table\n"
        "print('\\n'.join(format_dielectric_table(replace(ref, imaginary_bands=(4, 5)))))\n"
    )
    code, out = run_python(probe)
    report("ScF3 |Z~| and eps_0 unchanged by a complex unitary inside each degenerate set",
           code == 0 and "INVARIANT True True" in out, out)
    report("dielectric table: imaginary optical bands with a dipole get a note",
           "  note: 2 imaginary optical band(s) (bands 4 5) contribute with negative sign; "
           "eps_0 is not a static dielectric constant of an unstable structure" in out
           and out.count("  note: ") == 1, out)
    code, out, _ = irreps_run("ScF3_Pm-3m", "221_PPOSCAR_ScF3", ("--nac",))
    report("--irreps --nac without BORN: one-line error",
           code != 0 and "--nac requires a BORN file" in out and "Traceback" not in out, out)
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("221_PPOSCAR_ScF3", "FORCE_SETS"):
            shutil.copy(os.path.join(irrep_dir, "ScF3_Pm-3m", name), tmp)
        vector = ["--vector", "--dim", "4 4 4", "-c", "221_PPOSCAR_ScF3",
                  "--qpoint", "0.1", "0", "0", "--mode", "12"]
        code_a, out_a = run_phonon(vector, cwd=tmp)
        shutil.copy(born_path, tmp)
        code_b, out_b = run_phonon(vector, cwd=tmp)
        code_c, out_c = run_phonon(vector + ["--nac"], cwd=tmp)
        report("--vector near Gamma: BORN ignored without --nac, LO shift with --nac",
               code_a == code_b == code_c == 0 and out_a == out_b
               and "mode 12: DT1(1), 14.98" in out_b and "mode 12: DT1(1), 19.91" in out_c
               and "NAC: Born effective charges" in out_c, f"{out_b}\n{out_c}")
        code, out = run_phonon(["--subgroup", "--dim", "4 4 4", "-c", "221_PPOSCAR_ScF3",
                                "--qpoint", "R", "--nac"], cwd=tmp)
        report("--subgroup accepts --nac (note under the parent structure)",
               code == 0 and "NAC: Born effective charges" in out, out)
        code, out = run_phonon(["--modulation", "-c", "221_PPOSCAR_ScF3", "--dim", "4 4 4",
                                "--qpoint", "1/3", "0", "0", "--nac"], cwd=tmp)
        report("--modulation accepts --nac",
               code == 0 and "NAC: Born effective charges" in out, out)
    with tempfile.TemporaryDirectory() as tmp:
        cell_dir = os.path.join(tmp, "cell")
        os.makedirs(cell_dir)
        for name in ("221_PPOSCAR_ScF3", "FORCE_SETS"):
            shutil.copy(os.path.join(irrep_dir, "ScF3_Pm-3m", name), cell_dir)
        code, out = run_phonon(["--modulation", "-c", os.path.join("cell", "221_PPOSCAR_ScF3"),
                                "--dim", "4 4 4", "--qpoint", "1/3", "0", "0", "--nac"],
                               cwd=tmp)
        report("--modulation --nac without BORN: the error names both searched directories",
               code != 0 and "--nac requires a BORN file" in out
               and "the current directory or cell" in out and "Traceback" not in out, out)
    code, out = run_phonon(["--vibration", "-c", POSCAR_SrTiO3, "--qpoint", "GM", "--nac"])
    report("--vibration rejects --nac", code != 0 and "--nac is not used by --vibration"
           in out and "Traceback" not in out, out)

    # 4. --vibration at Gamma from the structure alone: Bilbao SAM for SrTiO3
    # (4 T1u + T2u: T1u IR, T2u silent, one T1u acoustic)
    with tempfile.TemporaryDirectory() as tmp:
        npz = os.path.join(tmp, "modes.npz")
        code, out = run_phonon(["--vibration", "-c", POSCAR_SrTiO3, "--qpoint", "GM",
                                "--mode-index", "5", "--export-npz", npz, "--raman-tensor"])
        spaces = re.findall(r"Mode Space\s+\d+: irrep = (\S+ \[\S+\]), dimension = \d+, "
                            r"component numbers = 1\.\.\d+, activity = (\S+)", out)
        report("--vibration GM: 4 GM4- [T1u] IR + GM5- [T2u] silent, summary block",
               code == 0
               and spaces == [("GM4-(3) [T1u]", "IR")] * 4 + [("GM5-(3) [T2u]", "silent")]
               and "\n\n* Gamma-point activity *\nGM4- [T1u] x4 (3 IR, 1 acoustic)\n"
               "GM5- [T2u] x1 (silent)\n  (acoustic: " in out, out)
        report("--vibration GM: Wyckoff orbits Sr 1a GM4-, Ti 1b GM4-, O 3c 2 GM4- + GM5-",
               "\n\n* Wyckoff-orbit breakdown (Gamma) *\n"
               "  Wyckoff orbit Sr (1a): GM4- [T1u]\n  Wyckoff orbit Ti (1b): GM4- [T1u]\n"
               "  Wyckoff orbit O (3c): 2 GM4- [T1u] + GM5- [T2u]\n"
               "  sum over the orbits: 4 GM4- + GM5- (equals the mode-space list)" in out, out)
        report("SrTiO3 --raman-tensor: no Raman-active irrep at Gamma",
               "\n\n* Raman tensors (Cartesian axes of the input cell) *\n"
               "  no Raman-active irrep at Gamma\n\n* Selected basis vector *\n"
               "Selected mode space: 5\n" in out, out)
        if os.path.isfile(npz):
            data = np.load(npz, allow_pickle=True)
            report("--export-npz stores the activities at Gamma",
                   "activities" in data.files
                   and list(data["activities"]) == ["IR"] * 4 + ["silent"], str(data.files))
        else:
            report("--export-npz written", False, out)
    code, out = run_phonon(["--vibration", "-c", POSCAR_SrTiO3, "--qpoint", "R"])
    report("--vibration away from Gamma: no activity",
           code == 0 and "Mode Space" in out and "activity" not in out, out)

    # wurtzite ZnO (6mm): Bilbao SAM 2b = A1 + B1 + E1 + E2; Loudon's C6v tensors
    zno = os.path.join(ROOT, "example", "test_POSCARs", "186_PPOSCAR_ZnO")
    code, out = run_phonon(["--vibration", "-c", zno, "--qpoint", "GM", "--raman-tensor"])
    report("ZnO --vibration GM: both 2b orbits GM1 + GM4 + GM5 + GM6 (A1 + B1 + E2 + E1)",
           code == 0 and "  Wyckoff orbit Zn (2b): GM1 [A1] + GM4 [B1] + GM5 [E2] + GM6 [E1]\n"
           "  Wyckoff orbit O (2b): GM1 [A1] + GM4 [B1] + GM5 [E2] + GM6 [E1]\n" in out
           and "(equals the mode-space list)" in out
           and "GM4 [B1] x2 (silent)\nGM5 [E2] x2 (Raman)\n" in out, out)
    report("ZnO --raman-tensor: A1 diag(a,a,0)+diag(0,0,a), E2 xx-yy/xy, E1 yz/xz",
           "  GM1 [A1] (2 tensors):  [[a, 0, 0], [0, a, 0], [0, 0, 0]]   "
           "[[0, 0, 0], [0, 0, 0], [0, 0, a]]\n"
           "  GM5 [E2] (2 tensors):  [[a, 0, 0], [0, -a, 0], [0, 0, 0]]   "
           "[[0, a, 0], [a, 0, 0], [0, 0, 0]]\n"
           "  GM6 [E1] (2 tensors):  [[0, 0, 0], [0, 0, a], [0, a, 0]]   "
           "[[0, 0, a], [0, 0, 0], [a, 0, 0]]" in out, out)

    # Mulliken symbols (phonopy's character tables and axis convention)
    # (i) the force-constant route against phonopy's own ir_labels per set
    probe = (
        "import warnings\n"
        "warnings.simplefilter('ignore')\n"
        "import os, phonopy\n"
        "from crystod import phonon\n"
        f"root = {irrep_dir!r}\n"
        "for name, poscar in (('SrTiO3_Pm-3m', '221_PPOSCAR_SrTiO3'),\n"
        "                     ('ScF3_Pm-3m', '221_PPOSCAR_ScF3'),\n"
        "                     ('Si_Fd-3m', '227_PPOSCAR_Si'), ('BaTiO3_P4mm', '99_PPOSCAR_BaTiO3')):\n"
        "    ph = phonopy.load(unitcell_filename=os.path.join(root, name, poscar),\n"
        "        force_sets_filename=os.path.join(root, name, 'FORCE_SETS'),\n"
        "        supercell_matrix=[4, 4, 4], primitive_matrix='auto', is_nac=False, log_level=0)\n"
        "    modes = phonon.label_phonon_modes(ph, [0, 0, 0])\n"
        "    ph.set_irreps([0, 0, 0], degeneracy_tolerance=1e-3)\n"
        "    ours = phonon.mulliken_symbols(ph)\n"
        "    cell = phonon.mulliken_symbols(os.path.join(root, name, poscar))\n"
        "    pairs = sorted({(mode.labels[0], ours.get(mode.labels[0]), label)\n"
        "                    for mode, label in zip(modes, ph.irreps._ir_labels)\n"
        "                    if len(mode.labels) == 1})\n"
        "    print('FC', name, ours == cell, all(a == b for _, a, b in pairs), pairs)\n"
    )
    code, out = run_python(probe)
    rows = {name: (same, agree, pairs) for name, same, agree, pairs in
            re.findall(r"^FC (\S+) (\S+) (\S+) (.*)$", out, re.M)}
    report("Mulliken symbols agree with phonopy's ir_labels per set and between the "
           "force-constant and structure routes (SrTiO3, ScF3, Si, BaTiO3)",
           code == 0 and len(rows) == 4
           and all(same == agree == "True" for same, agree, _ in rows.values())
           and "('GM4-', 'T1u', 'T1u'), ('GM5-', 'T2u', 'T2u')" in rows["SrTiO3_Pm-3m"][2]
           and "('GM4-', 'T1u', 'T1u'), ('GM5+', 'T2g', 'T2g')" in rows["Si_Fd-3m"][2]
           and "('GM1', 'A1', 'A1'), ('GM2', 'B1', 'B1'), ('GM5', 'E', 'E')"
           in rows["BaTiO3_P4mm"][2], out)
    # (ii) and (iii) the structure route: wurtzite ZnO and YGaO3 (6mm, with the A2
    # and B2 irreps among the phonons); every test structure: every Gamma irrep
    # gets a symbol and the symbols are the Mulliken set of its point group
    probe = (
        "import contextlib, glob, io, os, warnings\n"
        "warnings.simplefilter('ignore')\n"
        "import crystod.vibration_modes\n"
        "warnings.filterwarnings('error', message='no Mulliken')\n"
        "import numpy as np\n"
        "from phonopy.phonon.character_table import character_table\n"
        "from crystod import phonon\n"
        "from crystod.phonon_activity import _gamma_irreps\n"
        "from crystod.vasp_io import read_poscar_cell\n"
        f"root = {os.path.join(ROOT, 'example', 'test_POSCARs')!r}\n"
        "for name in ('186_PPOSCAR_ZnO', '185_PPOSCAR_YGaO3'):\n"
        "    acts = phonon.gamma_mode_activities(os.path.join(root, name))\n"
        "    symbols = phonon.mulliken_symbols(os.path.join(root, name))\n"
        "    print('SIXMM', name, sorted(symbols.items()),\n"
        "          phonon.format_activity_summary(acts, symmetry_only=True, mulliken=symbols))\n"
        "groups, bad = {}, []\n"
        "for path in sorted(glob.glob(os.path.join(root, '*_PPOSCAR_*'))):\n"
        "    if path.endswith(('.cif', '.html', '.txt')):\n"
        "        continue\n"
        "    with contextlib.redirect_stdout(io.StringIO()):\n"
        "        vib = phonon.SymmetryOnlyVibrations(read_poscar_cell(path))\n"
        "    table = _gamma_irreps(vib)\n"
        "    symbols = phonon.mulliken_symbols(vib)\n"
        "    group = str(vib.spglib_dataset['pointgroup'])\n"
        "    by_symbol = {}\n"
        "    for label, chi in zip(table.labels, table.characters):\n"
        "        by_symbol.setdefault(symbols.get(label), []).append(chi)\n"
        "    ok = (None not in by_symbol and set(by_symbol)\n"
        "          == set(character_table[group][0]['character_table'])\n"
        "          and all(len(c) == 1 or (len(c) == 2 and np.allclose(c[0], c[1].conj()))\n"
        "                  for c in by_symbol.values()))\n"
        "    groups.setdefault(group, []).append(ok)\n"
        "    if not ok:\n"
        "        bad.append(os.path.basename(path))\n"
        "print('SWEEP', sum(map(len, groups.values())), len(groups), bad)\n"
    )
    code, out = run_python(probe)
    sweep = re.search(r"^SWEEP (\d+) (\d+) (.*)$", out, re.M)
    report("ZnO: GM1 = A1, GM2 = A2, GM3 = B2, GM4 = B1 (silent), GM5 = E2 (Raman), "
           "GM6 = E1 (IR+Raman); YGaO3 carries A2 and B2",
           code == 0 and "SIXMM 186_PPOSCAR_ZnO [('GM1', 'A1'), ('GM2', 'A2'), ('GM3', 'B2'), "
           "('GM4', 'B1'), ('GM5', 'E2'), ('GM6', 'E1')] GM1 [A1] x2 (1 IR+Raman, 1 acoustic), "
           "GM4 [B1] x2 (silent), GM5 [E2] x2 (Raman), GM6 [E1] x2 (1 IR+Raman, 1 acoustic)" in out
           and "GM2 [A2] x5 (silent)" in out and "GM3 [B2] x10 (silent)" in out, out)
    report("every test structure (all 32 point groups): one Mulliken symbol per Gamma irrep, "
           "the symbol set of the point group, a symbol shared only by a complex pair",
           code == 0 and sweep is not None and int(sweep.group(1)) >= 200
           and sweep.group(2) == "32" and sweep.group(3) == "[]", out)
    # complex pairs: both members and the pair label carry the physically
    # irreducible symbol (E of 3), as phonopy's tables do
    trigonal_c3 = os.path.join(ROOT, "example", "test_POSCARs", "147_PPOSCAR_ZnRe2O8")
    code, out = run_python(
        "import warnings\n"
        "warnings.simplefilter('ignore')\n"
        "from crystod import phonon\n"
        f"print('PAIR', sorted(phonon.mulliken_symbols({os.path.join(ROOT, 'example', 'test_POSCARs', '143_PPOSCAR_HgBr')!r}).items()))\n"
        f"records = phonon.gamma_raman_tensors({trigonal_c3!r})\n"
        f"symbols = phonon.mulliken_symbols({trigonal_c3!r})\n"
        "print('\\n'.join(phonon.format_raman_tensors(records, mulliken=symbols)))\n")
    report("complex pair GM2/GM3 of 3 (P3 HgBr): E for both and for GM2GM3; "
           "ZnRe2O8 (-3) Raman lines GM1+ [Ag] and GM2+GM3+ [Eg]",
           code == 0 and "PAIR [('GM1', 'A'), ('GM2', 'E'), ('GM2GM3', 'E'), ('GM3', 'E')]" in out
           and "  GM1+ [Ag] (" in out and "  GM2+GM3+ [Eg] (2 partners" in out, out)
    # --raman-tensor error paths
    code_a, out_a = run_phonon(["--vibration", "-c", POSCAR_SrTiO3, "--qpoint", "R",
                                "--raman-tensor"])
    code_b, out_b = run_phonon(["--fatband", "--dim", "4 4 4", "--raman-tensor"])
    code_c, out_c = run_phonon(["--vibration", "-c", POSCAR_SrTiO3, "--list-qpoints",
                                "--raman-tensor"])
    report("--raman-tensor rejected away from Gamma and outside --irreps/--vibration",
           code_a != 0 and "--raman-tensor is defined at Gamma only" in out_a
           and code_b != 0 and "--raman-tensor is only used by --irreps and --vibration" in out_b
           and code_c != 0 and "--raman-tensor requires --qpoint GM" in out_c
           and "Traceback" not in out_a + out_b + out_c, f"{out_a}\n{out_b}\n{out_c}")

    # Python API: the same answers, and the classification rules
    trigonal = [os.path.join(ROOT, "example", "test_POSCARs", name)
                for name in ("166_PPOSCAR_Bi", "147_PPOSCAR_ZnRe2O8")]
    probe = (
        "import warnings\n"
        "warnings.simplefilter('ignore')\n"
        "import numpy as np, phonopy\n"
        "from crystod import phonon\n"
        "from crystod.examples import example_path\n"
        "ph = phonopy.load(unitcell_filename=example_path('221_PPOSCAR_SrTiO3'),\n"
        "    force_sets_filename=example_path('FORCE_SETS_SrTiO3'),\n"
        "    supercell_matrix=[4, 4, 4], primitive_matrix='auto', is_nac=False)\n"
        "acts = phonon.gamma_mode_activities(ph)\n"
        "print('PH', phonon.format_activity_summary(acts))\n"
        "print('FIRST', acts[0].labels, acts[0].band_indices, acts[0].activity,\n"
        "      acts[0].n_ir, acts[0].ir_active)\n"
        "cell = phonon.gamma_mode_activities(example_path('221_PPOSCAR_SrTiO3'))\n"
        "print('CELL', phonon.format_activity_summary(cell, symmetry_only=True))\n"
        "modes = phonon.label_phonon_modes(ph, [0, 0, 0])\n"
        "print('MODES', [m.activity for m in modes])\n"
        "print('R', {m.activity for m in phonon.label_phonon_modes(ph, [0.5, 0.5, 0.5])})\n"
        "e, i = np.eye(3, dtype=int), -np.eye(3, dtype=int)\n"
        "print('CI', [a.activity for a in phonon.classify_gamma_irreps([e, i], [[1, 1], [1, -1]])])\n"
        "try:\n"
        "    phonon.classify_gamma_irreps([e, i], [[1, 0]])\n"
        "    print('NOT RAISED')\n"
        "except ValueError as exc:\n"
        "    print('RAISED', 'not a non-negative integer' in str(exc))\n"
        "print('TYPE', type(acts[0]).__name__, phonon.Activity.__module__)\n"
        "wide = phonon.gamma_mode_activities(ph, degeneracy_tolerance=2.7)\n"
        "print('WIDE', len(wide), phonon.format_activity_summary(wide))\n"
        "print('MERGED', wide[0].band_indices[-1], wide[0].labels, wide[0].activity)\n"
        "cubic = [np.diag(d) for d in ([1., 1., 1.], [-1., -1., 1.], [-1., 1., -1.],\n"
        "                                [1., -1., -1.])]   # point group 222\n"
        "print('B1', [t.tolist() for t in phonon.raman_tensor_basis(cubic, [1, 1, -1, -1])])\n"
        f"bi = phonon.gamma_raman_tensors({trigonal[0]!r})\n"
        "print('BI', [(r.label, len(r.tensors), len(r.partners)) for r in bi])\n"
        "print('\\n'.join(phonon.format_raman_tensors(bi)))\n"
        f"c3 = phonon.gamma_raman_tensors({trigonal[1]!r})\n"
        "print('C3', [(r.label, len(r.partners), len(r.partners[0]) if r.partners else 0)\n"
        "             for r in c3])\n"
    )
    code, out = run_python(probe)
    report("Raman tensors of a repeated E (Bi, -3m): 2 partners with shared constants (Loudon)",
           "BI [('GM1+', 2, 0), ('GM3+', 4, 2)]" in out
           and "  GM3+ (2 partners, constants a, b):  [[a, 0, 0], [0, -a, b], [0, b, 0]]   "
           "[[0, a, b], [a, 0, 0], [b, 0, 0]]" in out, out)
    report("Raman tensors of the complex E pair of -3 (ZnRe2O8): 2 partners, 4 constants",
           "C3 [('GM1+', 0, 0), ('GM2+GM3+', 2, 4)]" in out, out)
    report("merged 12-band set (tolerance 2.7) split per irrep: same summary as the default",
           "WIDE 2 GM4- x4 (3 IR, 1 acoustic), GM5- x1 (silent)" in out
           and "MERGED 12 ('GM4-', 'GM5-') ('IR', 'silent', 'acoustic')" in out, out)
    report("raman_tensor_basis: B1 of 222 is the xy pair",
           "B1 [[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]" in out, out)
    report("API gamma_mode_activities (phonopy object) = --irreps summary",
           code == 0 and "PH GM4- x4 (3 IR, 1 acoustic), GM5- x1 (silent)" in out
           and "FIRST ('GM4-',) (1, 2, 3) ('acoustic',) 1 False" in out, out)
    report("API gamma_mode_activities (structure) = --vibration summary",
           "CELL GM4- x4 (3 IR, 1 acoustic), GM5- x1 (silent)" in out, out)
    report("PhononMode.activity at Gamma, empty elsewhere",
           "MODES [('acoustic',), ('IR',), ('IR',), ('silent',), ('IR',)]" in out
           and "R {()}" in out, out)
    report("classify_gamma_irreps: Ag Raman, Au IR (point group -1); bad characters raise",
           "CI [('Raman',), ('IR',)]" in out and "RAISED True" in out
           and "TYPE Activity crystod.phonon_activity" in out, out)


# ---------------------------------------------------------------- 30. crystod-phonon --fatband
def test_30_phonon_fatband() -> None:
    print("\n[30] crystod-phonon --fatband (ScF3, 4x4x4 FORCE_SETS)")
    if not os.path.isdir(PHONON_FATBAND_DIR):
        report("example data found", False, PHONON_FATBAND_DIR)
        return
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("221_PPOSCAR_ScF3", "FORCE_SETS"):
            shutil.copy(os.path.join(PHONON_FATBAND_DIR, name), tmp)

        code, out = run_phonon(
            ["--fatband", "-c", "221_PPOSCAR_ScF3", "--dim", "4", "4", "4",
             "--npoints", "11"],
            cwd=tmp,
        )
        report("exit code 0", code == 0, out)
        report("Pm-3m and seekpath k-path detected",
               "Pm-3m" in out and "k-path (seekpath)" in out, out)
        report("fatband_Sc.pdf written", os.path.isfile(os.path.join(tmp, "fatband_Sc.pdf")))
        report("fatband_F.pdf written", os.path.isfile(os.path.join(tmp, "fatband_F.pdf")))

        code, out = run_phonon(
            ["--fatband", "-c", "221_PPOSCAR_ScF3", "--dim", "4", "4", "4",
             "--npoints", "11", "--element", "F"],
            cwd=tmp,
        )
        report("single-element mode exit 0", code == 0, out)

        code, out = run_phonon(
            ["--fatband", "-c", "221_PPOSCAR_ScF3", "--dim", "4", "4", "4",
             "--npoints", "11", "--element", "Xx"],
            cwd=tmp,
        )
        report("unknown element rejected cleanly",
               code != 0 and "is not in this compound" in out and "Traceback" not in out, out)

        code, out = run_phonon(
            ["--fatband", "-c", "221_PPOSCAR_ScF3", "--dim", "4", "4", "4",
             "--npoints", "11", "--nac", "--element", "F"],
            cwd=tmp,
        )
        report("--nac without BORN rejected cleanly",
               code != 0 and "requires a BORN file" in out and "Traceback" not in out, out)

        born_path = os.path.join(PHONON_FATBAND_DIR, "BORN")
        if os.path.isfile(born_path):
            shutil.copy(born_path, tmp)
            code, out = run_phonon(
                ["--fatband", "-c", "221_PPOSCAR_ScF3", "--dim", "4", "4", "4",
                 "--npoints", "11", "--nac", "--element", "F"],
                cwd=tmp,
            )
            report("--nac with BORN exit 0", code == 0, out)
            report("NAC announced and fatband_nac_F.pdf written",
                   "NAC (LO/TO splitting) enabled" in out
                   and os.path.isfile(os.path.join(tmp, "fatband_nac_F.pdf")), out)
        else:
            report("BORN example found (skipping --nac run)", False, born_path)


# ---------------------------------------------------------------- 31. crystod-phonon --lt
def test_31_phonon_lt() -> None:
    print("\n[31] crystod-phonon --lt (ScF3, 4x4x4 FORCE_SETS)")
    if not os.path.isdir(PHONON_LT_DIR):
        report("example data found", False, PHONON_LT_DIR)
        return
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("221_PPOSCAR_ScF3", "FORCE_SETS", "BORN"):
            shutil.copy(os.path.join(PHONON_LT_DIR, name), tmp)

        code, out = run_phonon(
            ["--lt", "-c", "221_PPOSCAR_ScF3", "--dim", "4", "4", "4",
             "--npoints", "11"],
            cwd=tmp,
        )
        report("exit code 0", code == 0, out)
        report("phonon_band_LT.pdf written",
               os.path.isfile(os.path.join(tmp, "phonon_band_LT.pdf")), out)

        code, out = run_phonon(
            ["--lt", "-c", "221_PPOSCAR_ScF3", "--dim", "4", "4", "4",
             "--npoints", "11", "--nac"],
            cwd=tmp,
        )
        report("--nac exit 0", code == 0, out)
        report("NAC announced and phonon_band_LT_nac.pdf written",
               "NAC (LO/TO splitting) enabled" in out
               and os.path.isfile(os.path.join(tmp, "phonon_band_LT_nac.pdf")), out)

    # longitudinal-ratio sanity: acoustic branches near Gamma along [100]
    from phonopy import load as phonopy_load
    from crystod.phonon_lt import get_longitudinal_ratio
    from crystod.runtime_compat import get_qpoints_result

    phonon = phonopy_load(
        supercell_matrix=[4.0, 4.0, 4.0],
        primitive_matrix="auto",
        unitcell_filename=os.path.join(PHONON_LT_DIR, "221_PPOSCAR_ScF3"),
        force_sets_filename=os.path.join(PHONON_LT_DIR, "FORCE_SETS"),
        is_nac=False,
    )
    q = [0.1, 0.0, 0.0]
    phonon.run_qpoints([q], with_eigenvectors=True)
    qpoints_result = get_qpoints_result(phonon)
    eigvecs = qpoints_result.eigenvectors[0][np.newaxis]
    rec = np.linalg.inv(np.array(phonon.primitive.cell)).T
    ratio = get_longitudinal_ratio(np.array([q]), eigvecs, rec)[0]
    freqs = qpoints_result.frequencies[0]
    acoustic = np.argsort(freqs)[:3]
    report("acoustic set near GM splits into 2 T + 1 L along [100]",
           sorted(np.round(ratio[acoustic], 2))[:2] == [0.0, 0.0]
           and round(max(ratio[acoustic]), 2) > 0.9,
           str(ratio[acoustic]))


# ---------------------------------------------------------------- 32. crystod-phonon --vector
def _load_example_phonon(directory: str, cell: str, dim: list[int]):
    """phonopy object of an example unit cell with its FORCE_SETS (primitive_matrix "auto")."""
    from phonopy import load as phonopy_load

    return phonopy_load(
        supercell_matrix=dim,
        primitive_matrix="auto",
        unitcell_filename=os.path.join(directory, cell),
        force_sets_filename=os.path.join(directory, "FORCE_SETS"),
        log_level=0,
    )


def _phonopy_modulation_overlap(phonon, qpoint: list[float], band: int,
                                amplitude: float = 0.3) -> tuple[float, float]:
    """A mode frozen in by crystod, compared with phonopy's own modulation of it.

    crystod freezes band ``band`` (0-based) at ``qpoint`` with
    ``SymmetryAdaptedModulation``; phonopy's ``run_modulations`` freezes the
    same band on the same supercell of ``phonon.primitive`` (eigenvector over
    sqrt(m) times the Bloch phase exp(2 pi i q.r), global phase making the
    largest component real). Every crystod atom is matched to the nearest
    periodic image of an ideal site of the same species, so neither the atom
    order nor lattice wraps matter, and the rigid translation of each field is
    removed. Meaningful for a non-degenerate level at a time-reversal-invariant
    q, where the frozen-in field is unique up to its sign.

    Returns:
        ``(|cos|, norm)``: the overlap of the two displacement fields, and the
        norm of crystod's displacement per primitive cell in Angstrom.
    """
    from crystod.modulation import SymmetryAdaptedModulation

    modulation = SymmetryAdaptedModulation(phonon=phonon, qpoint=qpoint)
    atoms = modulation.get_modulated_structure([band], [amplitude])
    sizes = SymmetryAdaptedModulation.get_commensurate_supercell_sizes(qpoint)
    result = phonon.run_modulations(sizes, [[qpoint, band, 1.0, 0.0]])
    if result is not None:
        reference, supercell = result.modulations[0], result.supercell
    else:  # older phonopy returns nothing and keeps the result on the object
        modulations, supercell = phonon.get_modulations_and_supercell()
        reference = modulations[0]
    lattice = np.array(supercell.cell)
    sites = np.array(supercell.scaled_positions)
    numbers = np.array(supercell.numbers)
    field = np.zeros_like(sites)
    for position, number in zip(np.linalg.solve(lattice.T, atoms.positions.T).T,
                                atoms.numbers):
        offsets = position - sites
        offsets -= np.rint(offsets)
        distances = np.linalg.norm(offsets @ lattice, axis=1)
        distances[numbers != number] = np.inf
        site = int(np.argmin(distances))
        field[site] = offsets[site] @ lattice
    norm = np.linalg.norm(field) / np.sqrt(np.prod(sizes))
    reference = np.real(reference)
    field -= field.mean(axis=0)
    reference -= reference.mean(axis=0)
    overlap = abs(np.vdot(field, reference)) / (
        np.linalg.norm(field) * np.linalg.norm(reference))
    return float(overlap), float(norm)


def test_32_phonon_vector() -> None:
    print("\n[32] crystod-phonon --vector (Si, 4x4x4 FC)")
    if not os.path.isdir(PHONON_VECTOR_DIR):
        report("example data found", False, PHONON_VECTOR_DIR)
        return
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("227_PPOSCAR_Si", "FORCE_CONSTANTS"):
            shutil.copy(os.path.join(PHONON_VECTOR_DIR, name), tmp)

        code, out = run_phonon(
            ["--vector", "--dim", "4 4 4", "-c", "227_PPOSCAR_Si",
             "--readfc", "--qpoint", "GM"],
            cwd=tmp,
        )
        report("mode table exit 0", code == 0, out)
        report("acoustic modes labeled GM4-", "GM4-" in out, out)
        report("optical modes labeled GM5+", "GM5+" in out, out)
        report("mode table saved as text file",
               os.path.isfile(os.path.join(tmp, "phonon_modes_Si_GM.txt")))
        report("all 6 modes exported by default (1-based names)",
               os.path.isfile(os.path.join(tmp, "POSCAR_Si_GM_mode1_GM4-.vesta"))
               and os.path.isfile(os.path.join(tmp, "POSCAR_Si_GM_mode6_GM5+.vesta")))

        code, out = run_phonon(
            ["--vector", "--dim", "4 4 4", "-c", "227_PPOSCAR_Si",
             "--readfc", "--qpoint", "GM", "--mode", "4"],
            cwd=tmp,
        )
        vesta_path = os.path.join(tmp, "POSCAR_Si_GM_mode4_GM5+.vesta")
        report("GM mode 4 export exit 0", code == 0, out)
        report("VESTA file written with auto name", os.path.isfile(vesta_path))

        # non-special q (DT line): mode labels fall back to the ISO-IR tables,
        # and the q label itself becomes the ISO-IR k-vector type (DT)
        code, out = run_phonon(
            ["--vector", "--dim", "4 4 4", "-c", "227_PPOSCAR_Si",
             "--readfc", "--qpoint", "0.2", "0", "0.2", "--mode", "1"],
            cwd=tmp,
        )
        report("DT-line modes labeled via ISO-IR",
               code == 0 and "DT5(2)" in out and "DT1(1)" in out, out)
        report("DT-line q named via ISO-IR",
               "Selected q-point: DT =" in out, out)
        report("DT-line VESTA file named with ISO-IR q label and irrep tag",
               os.path.isfile(os.path.join(tmp, "POSCAR_Si_DT_mode1_DT5.vesta")))
        report("DT-line mode table named with ISO-IR q label",
               os.path.isfile(os.path.join(tmp, "phonon_modes_Si_DT.txt")))

        # --keep-q-coords: coordinate-based q label, ISO-IR irrep tag kept
        code, out = run_phonon(
            ["--vector", "--dim", "4 4 4", "-c", "227_PPOSCAR_Si",
             "--readfc", "--qpoint", "0.2", "0", "0.2", "--mode", "1",
             "--keep-q-coords"],
            cwd=tmp,
        )
        report("--keep-q-coords keeps the coordinate q label",
               code == 0 and "Selected q-point: q_0.2_0_0.2" in out
               and os.path.isfile(os.path.join(tmp, "POSCAR_Si_q_0.2_0_0.2_mode1_DT5.vesta")), out)
        if os.path.isfile(vesta_path):
            text = open(vesta_path).read()
            report("VESTA file contains arrows (VECTR/VECTT)",
                   "VECTR" in text and "VECTT" in text, text[:500])
            report("VESTA title carries irrep label", "GM5+" in text, text[:500])

        code, out = run_phonon(
            ["--vector", "--dim", "4 4 4", "-c", "227_PPOSCAR_Si",
             "--readfc", "--qpoint", "X", "--mode", "1"],
            cwd=tmp,
        )
        report("X point export exit 0", code == 0, out)
        report("commensurate 2x1x2 supercell built", "2x1x2" in out, out)
        report("X VESTA file written",
               os.path.isfile(os.path.join(tmp, "POSCAR_Si_X_mode1_X4.vesta")))

        code, out = run_phonon(
            ["--vector", "--dim", "4 4 4", "-c", "227_PPOSCAR_Si",
             "--readfc", "--qpoint", "GM", "--mode", "4", "--conventional"],
            cwd=tmp,
        )
        conv_path = os.path.join(tmp, "POSCAR_Si_GM_mode4_GM5+_conv.vesta")
        report("conventional export exit 0", code == 0, out)
        report("conventional VESTA written with _conv suffix", os.path.isfile(conv_path))
        if os.path.isfile(conv_path):
            text = open(conv_path).read()
            report("conventional cubic cell (a = 5.4687)", "5.468728" in text, text[:400])
            arrows = re.findall(
                r"^\s*\d+\s+(-?\d\.\d+)\s+(-?\d\.\d+)\s+(-?\d\.\d+)\s*$",
                text.split("VECTR")[1].split("VECTT")[0], re.M,
            )
            # the partners of the GM5+ triplet are listed along a, b, c
            axis_pure = bool(arrows) and all(
                abs(abs(float(a)) - 1.5) < 1e-4 and abs(float(b)) < 1e-5 and abs(float(c)) < 1e-5
                for a, b, c in arrows
            )
            report("GM mode 4 arrows purely along a in conventional cell", axis_pure)

        code, out = run_phonon(
            ["--vector", "--dim", "4 4 4", "-c", "227_PPOSCAR_Si",
             "--readfc", "--qpoint", "GM", "--mode", "4", "5", "6", "--conventional"],
            cwd=tmp,
        )
        sum_path = os.path.join(tmp, "POSCAR_Si_GM_mode4+5+6_GM5+_conv.vesta")
        report("multi-mode sum export exit 0", code == 0, out)
        report("summed modes written to one file", os.path.isfile(sum_path))
        if os.path.isfile(sum_path):
            text = open(sum_path).read()
            arrows = re.findall(
                r"^\s*\d+\s+(-?\d\.\d+)\s+(-?\d\.\d+)\s+(-?\d\.\d+)\s*$",
                text.split("VECTR")[1].split("VECTT")[0], re.M,
            )
            along_111 = bool(arrows) and all(
                abs(abs(float(a)) - 1.5 / np.sqrt(3.0)) < 1e-4
                and float(a) == float(b) == float(c)
                for a, b, c in arrows
            )
            report("mode 4+5+6 sum points along [111]", along_111)

    # symmetry-adapted directions: degenerate GM optical modes must point
    # along the cubic axes, not arbitrary combinations within the subspace
    from phonopy import load as phonopy_load
    from crystod.phonon_vector import build_symmetry_adapted_modes

    phonon = phonopy_load(
        supercell_matrix=[4.0, 4.0, 4.0],
        primitive_matrix="auto",
        unitcell_filename=os.path.join(PHONON_VECTOR_DIR, "227_PPOSCAR_Si"),
        force_constants_filename=os.path.join(PHONON_VECTOR_DIR, "FORCE_CONSTANTS"),
    )
    modes = build_symmetry_adapted_modes(phonon, [0.0, 0.0, 0.0])
    aligned = True
    for index in (3, 4, 5):
        vector = np.real(modes[index][1]).reshape(-1, 3)[0]
        aligned &= bool(
            (np.sort(np.abs(vector))[:2] < 1e-6).all() and np.abs(vector).max() > 0.1
        )
    report("GM optical eigenvectors axis-aligned (symmetry-adapted)", aligned)
    freqs = [round(mode[0], 4) for mode in modes]
    report("symmetry-adapted frequencies match phonopy",
           freqs == [0.0, 0.0, 0.0, 14.9571, 14.9571, 14.9571],
           str(freqs))

    # the construction has to work at every representative of a q point, not
    # only at the one some table lists: on body-centred Sr3Ti2O7, X written as
    # (0.5, 0.5, 0) or (-0.5, 0.5, 0) (q and q + G), and P, where the projected
    # rows used to be read as bras ("Coupling between irrep spaces is not
    # scalar"; --vector then fell back to plain eigenvectors)
    if not os.path.isdir(MODULATION_STO_DIR):
        report("Sr3Ti2O7 example data found", False, MODULATION_STO_DIR)
        return
    phonon = _load_example_phonon(MODULATION_STO_DIR, "139_PPOSCAR_Sr3Ti2O7", [4, 4, 4])
    for name, spellings in (("X", ([0.5, 0.5, 0.0], [-0.5, 0.5, 0.0])),
                            ("P", ([0.25, 0.25, 0.25], [-0.75, 0.25, 0.25]))):
        spectra = []
        for qpoint in spellings:
            try:
                modes = build_symmetry_adapted_modes(phonon, qpoint)
            except (RuntimeError, ValueError) as exc:
                report(f"Sr3Ti2O7 symmetry-adapted modes at {name} = {qpoint}", False, str(exc))
                continue
            frequencies = np.array([mode[0] for mode in modes])
            result = phonon.run_qpoints([qpoint])
            if result is None:  # older phonopy keeps the result on the object
                result = phonon.qpoints
            reference = np.sort(result.frequencies[0])
            report(f"Sr3Ti2O7 symmetry-adapted modes at {name} = {qpoint} match phonopy",
                   len(modes) == 36 and np.abs(frequencies - reference).max() < 1e-6,
                   str(frequencies - reference))
            spectra.append(frequencies)
        report(f"Sr3Ti2O7 {name}: q and q + G give identical frequencies",
               len(spectra) == 2 and np.abs(spectra[0] - spectra[1]).max() < 1e-8)

    with tempfile.TemporaryDirectory() as tmp:
        for name in ("139_PPOSCAR_Sr3Ti2O7", "FORCE_SETS"):
            shutil.copy(os.path.join(MODULATION_STO_DIR, name), tmp)
        code, out = run_phonon(
            ["--vector", "--dim", "4 4 4", "-c", "139_PPOSCAR_Sr3Ti2O7",
             "--qpoint", "0.25", "0.25", "0.25", "--mode", "1"],
            cwd=tmp,
        )
        report("--vector at Sr3Ti2O7 P keeps the symmetry-adapted modes",
               code == 0 and "falling back" not in out, out)


# ---------------------------------------------------------------- 33. crystod-phonon --modulation
def test_33_modulation() -> None:
    print("\n[33] crystod-phonon --modulation (known space groups from example/33_modulation README)")
    yaml_path = os.path.join(MODULATION_DIR, "phonopy_params.yaml")
    if not os.path.isfile(yaml_path):
        report("phonopy_params.yaml found", False, yaml_path)
        return

    with tempfile.TemporaryDirectory() as tmp:
        # preview mode: no --mode prints the mode table and the star of q only
        code, out = run_phonon(
            ["--modulation", "--yaml", yaml_path, "--qpoint", "0.5", "0.5", "0.5"],
            cwd=tmp,
        )
        report("preview (no --mode) exit 0", code == 0, out)
        report("preview shows mode table", "Phonon modes at q" in out and "Irrep" in out, out)
        report("mode table header uses 'Irrep' (not 'Irrep Block')", "Irrep Block" not in out, out)
        report("mode table shows CDML irrep labels", "R4+(3)" in out, out)
        report("R4+ soft modes are the lowest three", out.count("R4+(3)") == 3, out)
        report("preview shows star of q", "Star of q" in out, out)
        report("preview writes no structure", not os.listdir(tmp), out)

        # star-arm mapping: (0, 0.5, 0.5) is an M arm; irreptables tabulates
        # only (0.5, 0.5, 0), so labeling must map the arm onto that point
        code, out = run_phonon(
            ["--modulation", "--yaml", yaml_path, "--qpoint", "0", "0.5", "0.5"],
            cwd=tmp,
        )
        report("non-representative M arm labeled via star mapping",
               code == 0 and "M3+(1)" in out, out)

        # default output name: MPOSCAR_{q}_{mode}_{irrep}_{subgroup}
        code, out = run_phonon(
            ["--modulation", "--yaml", yaml_path, "--qpoint", "0.5", "0.5", "0.5",
             "--mode", "1", "2", "3", "--amplitude", "0.3"],
            cwd=tmp,
        )
        report("R4+(a,a,a) exit 0", code == 0, out)
        report("R4+(a,a,a) -> R-3c", "R-3c" in out, out)
        report("star of q displayed", "Star of q" in out, out)
        report("default name MPOSCAR_R_mode1+2+3_R4+_R-3c",
               os.path.isfile(os.path.join(tmp, "MPOSCAR_R_mode1+2+3_R4+_R-3c")), out)

        out_poscar = os.path.join(tmp, "POSCAR_I4mcm")
        code, out = run_phonon(
            ["--modulation", "--yaml", yaml_path, "--qpoint", "0.5", "0.5", "0.5",
             "--mode", "1", "--amplitude", "0.3", "--output", out_poscar]
        )
        report("R4+(0,0,a) -> I4/mcm", code == 0 and "I4/mcm" in out.replace("I4mcm", "I4/mcm"), out)

        out_poscar = os.path.join(tmp, "POSCAR_multi_q")
        code, out = run_phonon(
            ["--modulation", "--yaml", yaml_path,
             "--qpoint1", "0", "0.5", "0.5", "--mode1", "1", "--amplitude1", "0.3",
             "--qpoint2", "0.5", "0", "0.5", "--mode2", "1", "--amplitude2", "0.3",
             "--output", out_poscar]
        )
        report("multi-q M3+(a;a;0) exit 0", code == 0, out)
        report("multi-q M3+(a;a;0) -> I4/mmm",
               "I4/mmm" in out.replace("I4mmm", "I4/mmm"), out)
        report("star of q displayed for each q", out.count("Star of q") >= 2, out)

        # R written as (-0.5, 0.5, 0.5), i.e. q + G: the same R4+ triplet (it
        # used to stop with "Coupled irrep spaces with different dimensions")
        code, out = run_phonon(
            ["--modulation", "--yaml", yaml_path, "--qpoint", "-0.5", "0.5", "0.5",
             "--mode", "1", "2", "3", "--amplitude", "0.3", "--tolerance", "1e-5",
             "--output", os.path.join(tmp, "POSCAR_R_q_plus_G")]
        )
        report("R4+(a,a,a) at q + G = (-0.5, 0.5, 0.5) -> R-3c at 1e-5",
               code == 0 and "Space group: R-3c (#167)" in out, out)

        # a q that no supercell of at most 12 cells per axis holds is refused
        # before any mode is computed; the mode table can still be previewed
        code, out = run_phonon(
            ["--modulation", "--yaml", yaml_path, "--qpoint", "0.15", "0", "0",
             "--mode", "1", "--output", os.path.join(tmp, "POSCAR_incommensurate")]
        )
        report("incommensurate q = (0.15, 0, 0) refused cleanly",
               code != 0 and "is not commensurate with a supercell" in out
               and "Phonon modes at q" not in out and "Traceback" not in out
               and not os.path.exists(os.path.join(tmp, "POSCAR_incommensurate")), out)
        code, out = run_phonon(
            ["--modulation", "--yaml", yaml_path, "--qpoint", "0.15", "0", "0"]
        )
        report("incommensurate q still previews its mode table",
               code == 0 and "Phonon modes at q" in out, out)

    # Sr3Ti2O7 (I4/mmm): atoms off the inversion centres, two X arms. The mode
    # construction used to crash at (-0.5, 0.5, 0) ("Coupled irrep spaces are
    # not equivalent") and to freeze a wrong pattern at the equivalent
    # (0.5, 0.5, 0) that read as Cmcm only through the 0.1 A default
    # tolerance (Ama2 at 1e-5); the reported two-arm command crashed too.
    # Expected space groups: crystod-group --parent I4/mmm --irrep X3-.
    if not os.path.isdir(MODULATION_STO_DIR):
        report("Sr3Ti2O7 example data found", False, MODULATION_STO_DIR)
        return
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("139_PPOSCAR_Sr3Ti2O7", "FORCE_SETS", "phonopy_disp.yaml"):
            shutil.copy(os.path.join(MODULATION_STO_DIR, name), tmp)
        strict = ["--modulation", "-c", "139_PPOSCAR_Sr3Ti2O7", "--tolerance", "1e-5"]

        for qpoint, output in ((["-0.5", "0.5", "0"], "ARM_MINUS"),
                               (["0.5", "0.5", "0"], "ARM_PLUS")):
            code, out = run_phonon(
                strict + ["--qpoint", *qpoint, "--mode", "1", "--output", output], cwd=tmp
            )
            report(f"Sr3Ti2O7 X3-(0;a) at q = ({', '.join(qpoint)}) -> Cmcm at 1e-5",
                   code == 0 and "Space group: Cmcm (#63)" in out, out)
        from ase.io import read as _ase_read

        try:
            minus = _ase_read(os.path.join(tmp, "ARM_MINUS"), format="vasp")
            plus = _ase_read(os.path.join(tmp, "ARM_PLUS"), format="vasp")
            difference = max(np.abs(minus.cell[:] - plus.cell[:]).max(),
                             np.abs(minus.positions - plus.positions).max())
            detail = f"largest difference {difference:.2e} A"
        except (OSError, ValueError, IndexError) as exc:
            difference, detail = np.inf, str(exc)
        report("Sr3Ti2O7 X: q and q + G freeze the same structure",
               difference < 1e-8, detail)

        two_arms = ["--qpoint1", "0", "0", "0.5", "--mode1", "1", "--amplitude1", "0.3",
                    "--qpoint2", "-0.5", "0.5", "0", "--mode2", "1"]
        code, out = run_phonon(strict + two_arms + ["--amplitude2", "0.3"], cwd=tmp)
        report("Sr3Ti2O7 two X arms, equal amplitudes: X3-(a;a) -> P4_2/mnm at 1e-5",
               code == 0 and "Space group: P4_2/mnm (#136)" in out, out)
        code, out = run_phonon(strict + two_arms + ["--amplitude2", "0.15"], cwd=tmp)
        report("Sr3Ti2O7 two X arms, unequal amplitudes: X3-(a;b) -> Pnnm at 1e-5",
               code == 0 and "Space group: Pnnm (#58)" in out, out)

    # physics, independent of crystod's conventions: the displacement field
    # frozen in for a non-degenerate mode must be the one phonopy's own
    # modulation builds for that band (eigenvector over sqrt(m) with its Bloch
    # phase), up to sign, and --amplitude must be the displacement norm of one
    # primitive cell. Before the fix the overlap was 0.31 (Sr3Ti2O7 X3- mode 1
    # at (0, 0, 0.5)) and 0.99 (the mixed Sc/F X3- mode 9 of ScF3, where only
    # the missing 1/sqrt(m) shows).
    cases = [("Sr3Ti2O7", MODULATION_STO_DIR, "139_PPOSCAR_Sr3Ti2O7", qpoint, 0)
             for qpoint in ([0.0, 0.0, 0.5], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0])]
    cases.append(("ScF3", MODULATION_DIR, "221_PPOSCAR_ScF3", [0.0, 0.5, 0.0], 8))
    phonons = {}
    for material, directory, cell, qpoint, band in cases:
        if material not in phonons:
            phonons[material] = _load_example_phonon(directory, cell, [4, 4, 4])
        try:
            overlap, norm = _phonopy_modulation_overlap(phonons[material], qpoint, band)
        except (RuntimeError, ValueError) as exc:
            overlap, norm = 0.0, 0.0
            detail = str(exc)
        else:
            detail = f"overlap {overlap:.6f}, norm per primitive cell {norm:.6f} A"
        report(f"{material} q = {qpoint} mode {band + 1}: frozen field is phonopy's "
               "eigen-displacement", overlap > 0.999, detail)
        report(f"{material} q = {qpoint} mode {band + 1}: --amplitude 0.3 is the "
               "displacement norm of one primitive cell", abs(norm - 0.3) < 1e-6, detail)


# ---------------------------------------------------------------- 34. crystod-phonon --vibration
def test_34_vibration() -> None:
    print("\n[34] crystod-phonon --vibration")
    code, out = run_phonon(["--vibration", "-c", POSCAR_ScF3, "--qpoint", "R"])
    report("ScF3 q = R exit 0", code == 0, out)
    report("irrep-grouped mode spaces listed", "Mode Space" in out, out)
    report("mode spaces numbered from 1", "Mode Space  1:" in out and "Mode Space  0:" not in out, out)
    report("ISO-IR special q points listed with fraction coordinates",
           "\n\n* Q points (primitive) *\n  GM       (0, 0, 0)\n  R        (1/2, 1/2, 1/2)\n"
           "  M        (1/2, 1/2, 0)\n  X        (0, 1/2, 0)\n\n* Selected Q point *\n"
           "  R        (1/2, 1/2, 1/2)\n\n* Irrep-grouped vibration spaces *\n" in out
           and "Available high-symmetry" not in out and "X_1" not in out, out)
    report("no mode-space count line and no --mode-index hint; no Mulliken bracket at R",
           "Number of irrep-grouped" not in out and "Use --mode-index" not in out
           and "Gamma-point activity" not in out
           and not re.search(r"irrep = \S+ \[", out), out)

    # the output format of --vibration at Gamma (wurtzite ZnO): blank line +
    # "* ... *" header per block, Mulliken symbols after the Gamma labels, the
    # activity summary one irrep per line
    zno = os.path.join(ROOT, "example", "test_POSCARs", "186_PPOSCAR_ZnO")
    code, out = run_phonon(["--vibration", "-c", zno, "--qpoint", "GM", "--raman-tensor"])
    expected_blocks = (
        "\n\n* Q points (primitive) *\n"
        "  GM       (0, 0, 0)\n  A        (0, 0, 1/2)\n  K        (1/3, 1/3, 0)\n"
        "  H        (1/3, 1/3, 1/2)\n  M        (1/2, 0, 0)\n  L        (1/2, 0, 1/2)\n"
        "\n* Selected Q point *\n  GM       (0, 0, 0)\n"
        "\n* Irrep-grouped vibration spaces *\n"
        "  Mode Space  1: irrep = GM1(1) [A1], dimension = 1, component numbers = 1..1, "
        "activity = IR+Raman\n",
        "  Mode Space  3: irrep = GM4(1) [B1], dimension = 1, component numbers = 1..1, "
        "activity = silent\n",
        "  Mode Space  5: irrep = GM5(2) [E2], dimension = 2, component numbers = 1..2, "
        "activity = Raman\n",
        "  Mode Space  8: irrep = GM6(2) [E1], dimension = 2, component numbers = 1..2, "
        "activity = IR+Raman\n"
        "\n* Gamma-point activity *\n"
        "GM1 [A1] x2 (1 IR+Raman, 1 acoustic)\nGM4 [B1] x2 (silent)\nGM5 [E2] x2 (Raman)\n"
        "GM6 [E1] x2 (1 IR+Raman, 1 acoustic)\n"
        "  (acoustic: one set per occurrence of the irrep in the vector representation;\n"
        "   without force constants the mode spaces are symmetry-adapted patterns,\n"
        "   not normal modes)\n"
        "\n* Wyckoff-orbit breakdown (Gamma) *\n"
        "  Wyckoff orbit Zn (2b): GM1 [A1] + GM4 [B1] + GM5 [E2] + GM6 [E1]\n"
        "  Wyckoff orbit O (2b): GM1 [A1] + GM4 [B1] + GM5 [E2] + GM6 [E1]\n"
        "  sum over the orbits: 2 GM1 + 2 GM4 + 2 GM5 + 2 GM6 (equals the mode-space list)\n"
        "\n* Raman tensors (Cartesian axes of the input cell) *\n"
        "  GM1 [A1] (2 tensors):  [[a, 0, 0], [0, a, 0], [0, 0, 0]]   "
        "[[0, 0, 0], [0, 0, 0], [0, 0, a]]\n"
        "  GM5 [E2] (2 tensors):  [[a, 0, 0], [0, -a, 0], [0, 0, 0]]   "
        "[[0, a, 0], [a, 0, 0], [0, 0, 0]]\n"
        "  GM6 [E1] (2 tensors):  [[0, 0, 0], [0, 0, a], [0, a, 0]]   "
        "[[0, 0, a], [0, 0, 0], [a, 0, 0]]\n",
    )
    positions = [out.find(block) for block in expected_blocks]
    report("ZnO --vibration GM --raman-tensor: the block layout, [A1]/[B1]/[E2]/[E1] after "
           "the labels, one irrep per activity line, no count line, no hint",
           code == 0 and all(p >= 0 for p in positions) and positions == sorted(positions)
           and out.count("Mode Space") == 8 and out.count(" [A1]") == 6
           and "Number of irrep-grouped" not in out and "Use --mode-index" not in out
           and "H_2" not in out and "GAMMA" not in out, out)
    code, out = run_phonon(["--vibration", "-c", zno, "--qpoint", "K"])
    report("ZnO --vibration K: same layout without the Gamma blocks and brackets",
           code == 0 and "\n* Selected Q point *\n  K        (1/3, 1/3, 0)\n\n"
           "* Irrep-grouped vibration spaces *\n" in out
           and "  Mode Space  5: irrep = K3(2), dimension = 2, component numbers = 1..2\n" in out
           and "* Gamma-point activity *" not in out and "Wyckoff" not in out
           and "[" not in out.split("* Irrep-grouped vibration spaces *")[1], out)
    code_l, out_l = run_phonon(["--vibration", "-c", zno, "--list-qpoints"])
    report("--list-qpoints prints the Q-point block only",
           code_l == 0 and out_l.rstrip().endswith("  L        (1/2, 0, 1/2)")
           and "* Q points (primitive) *" in out_l and "Selected" not in out_l, out_l)
    aliases = {}
    for token in ("H_2", "gamma", "G", "Γ", "h", "1/3 1/3 -1/2", "-1/3 2/3 0", "0.123 0 0"):
        code, out = run_phonon(["--vibration", "-c", zno, "--qpoint", *token.split()])
        match = re.search(r"\* Selected Q point \*\n  (\S+) +(\(.*\))\n", out)
        aliases[token] = (code, match.groups() if match else None)
    report("--qpoint: seekpath and Gamma aliases resolve silently to ISO-IR names; "
           "coordinates named by their ISO-IR point, 4 decimals where not a fraction",
           aliases == {"H_2": (0, ("H", "(1/3, 1/3, -1/2)")), "gamma": (0, ("GM", "(0, 0, 0)")),
                       "G": (0, ("GM", "(0, 0, 0)")), "Γ": (0, ("GM", "(0, 0, 0)")),
                       "h": (0, ("H", "(1/3, 1/3, 1/2)")),
                       "1/3 1/3 -1/2": (0, ("H", "(1/3, 1/3, -1/2)")),
                       "-1/3 2/3 0": (0, ("K", "(-1/3, 2/3, 0)")),
                       "0.123 0 0": (0, ("SM", "(0.1230, 0, 0)"))}, f"{aliases}")
    code, out = run_phonon(["--vibration", "-c", zno, "--qpoint", "Q"])
    report("--qpoint with an unknown name: one-line error listing the ISO-IR names",
           code != 0 and "Unknown q-point label 'Q'. Available labels: GM, A, H, K, L, M" in out
           and "Traceback" not in out, out)
    with tempfile.TemporaryDirectory() as tmp:
        code, out = run_phonon(["--vibration", "-c", zno, "--qpoint", "GM", "--mode-index", "3"],
                               cwd=tmp)
    report("--mode-index: the selected basis vector under its own header",
           code == 0 and "\n\n* Selected basis vector *\nSelected mode space: 3\n"
           "Selected irrep     : GM4(1)\nSelected component : 1\n" in out, out)

    # the --vibration list is the special_points list of --irreps for the same
    # structure (phonopy's primitive cell; names, and coordinates where the two
    # primitive cells coincide)
    probe = (
        "import contextlib, io, warnings\n"
        "warnings.simplefilter('ignore')\n"
        "from phonopy import load\n"
        "from phonopy.structure.cells import get_primitive_matrix_by_centring\n"
        "from crystod.irreptables_compat import load_irreptables\n"
        "from crystod.phonon_irreps import _special_points_in_label_frame\n"
        "from crystod.runtime_compat import get_symmetry_dataset\n"
        "from crystod.vasp_io import read_poscar_cell\n"
        "from crystod.phonon import SymmetryOnlyVibrations\n"
        "IrrepTable, _ = load_irreptables()\n"
        f"root = {os.path.join(ROOT, 'example', 'test_POSCARs')!r}\n"
        "for name in ('221_PPOSCAR_SrTiO3', '186_PPOSCAR_ZnO', '227_PPOSCAR_Si',\n"
        "             '139_PPOSCAR_Sr3Ti2O7', '1_PPOSCAR_RbBe2F5', '2_PPOSCAR_PI2',\n"
        "             '12_PPOSCAR_VO2', '14_PPOSCAR_ZrO2', '166_PPOSCAR_Bi'):\n"
        "    cell = read_poscar_cell(root + '/' + name)\n"
        "    ph = load(unitcell=cell, supercell_matrix=[1, 1, 1], primitive_matrix='auto',\n"
        "              produce_fc=False, log_level=0)\n"
        "    dataset = get_symmetry_dataset(ph.symmetry)\n"
        "    table = IrrepTable(dataset['number'], spinor=False)\n"
        "    matrix = get_primitive_matrix_by_centring(dataset['international'][0])\n"
        "    names, points = _special_points_in_label_frame(ph, table, matrix)\n"
        "    with contextlib.redirect_stdout(io.StringIO()):\n"
        "        listed = SymmetryOnlyVibrations(cell).get_special_qpoints()\n"
        "    same = list(listed) == list(names)\n"
        "    coords = all(abs(a - b) < 1e-8 for q, p in zip(listed.values(), points)\n"
        "                 for a, b in zip(q, p))\n"
        "    print('LIST', name, same, coords, ','.join(listed))\n"
    )
    code, out = run_python(probe)
    rows = re.findall(r"^LIST (\S+) (\S+) (\S+) (\S+)$", out, re.M)
    report("--vibration lists the special_points of --irreps (SrTiO3, ZnO, Si, Sr3Ti2O7, "
           "P1, P-1, C2/m, P2_1/c, R-3m)",
           code == 0 and len(rows) == 9 and all(same == "True" for _, same, _, _ in rows)
           and ("186_PPOSCAR_ZnO", "True", "True", "GM,A,H,K,L,M") in rows
           and ("1_PPOSCAR_RbBe2F5", "True", "True", "GM,Z,X,U,V,R,Y,T") in rows, out)

    # (0, 0.5, 0.5) is an M arm; irreptables tabulates only (0.5, 0.5, 0)
    code, out = run_phonon(["--vibration", "-c", POSCAR_ScF3, "--qpoint", "0", "0.5", "0.5"])
    report("non-representative M arm labeled via star mapping",
           code == 0 and "M5+(2)" in out and "irrep_" not in out, out)

    # non-special q (T line): labels fall back to the ISO-IR (ISOTROPY) tables
    code, out = run_phonon(
        ["--vibration", "-c", POSCAR_SrTiO3, "--qpoint", "0.5", "0.5", "0.4"]
    )
    report("T-line q named via ISO-IR",
           code == 0 and "* Selected Q point *\n  T        (1/2, 1/2, 2/5)\n" in out, out)
    report("T-line mode spaces labeled via ISO-IR",
           "T5(2)" in out and "irrep_" not in out, out)

    # -k of the P point of I-4 (only P is tabulated): the 'A' names of the
    # complex-conjugate irreps (PA1 = conjugate of P1), as in SALC and
    # crystod-group; it used to print a line name (Q) and irrep_N
    poscar_alpo4 = os.path.join(ROOT, "example", "test_POSCARs", "82_PPOSCAR_AlPO4")
    code, out = run_phonon(["--vibration", "-c", poscar_alpo4,
                            "--qpoint", "-0.25", "-0.25", "-0.25"])
    report("I-4 AlPO4 at -k of P: q-point PA, mode spaces PA1 .. PA4",
           code == 0 and "* Selected Q point *\n  PA       (-1/4, -1/4, -1/4)\n" in out
           and "  PA " not in out.split("* Selected Q point *")[0]
           and "Mode Space  1: irrep = PA1(1)" in out
           and "Mode Space  5: irrep = PA4(1)" in out and "irrep_" not in out, out)

    def listed(text: str) -> list[str]:
        return re.findall(r"Mode Space\s+\d+: irrep = (\S+?\(\d+\))", text)

    # an input outside the ISO-IR setting: the list, a name given with
    # --qpoint and the name of given coordinates refer to the frame of the
    # labels (AlPO4 at a generic origin: seekpath's P is the PA of the labels,
    # so --qpoint P is the -k of seekpath's point; it printed 'Selected
    # q-point: P' above PA2 for both)
    with tempfile.TemporaryDirectory() as tmp:
        _write_cell_variant(poscar_alpo4, os.path.join(tmp, "POSCAR_AlPO4_shifted"),
                            shift=(0.0731, 0.1593, 0.2417))
        code_p, out_p = run_phonon(["--vibration", "-c", "POSCAR_AlPO4_shifted",
                                    "--qpoint", "P"], cwd=tmp)
        code_k, out_k = run_phonon(["--vibration", "-c", "POSCAR_AlPO4_shifted",
                                    "--qpoint", "0.25", "0.25", "0.25"], cwd=tmp)
    labels_p = re.findall(r"Mode Space\s+\d+: irrep = ([A-Z]+)\d", out_p)
    labels_k = re.findall(r"Mode Space\s+\d+: irrep = ([A-Z]+)\d", out_k)
    report("shifted AlPO4 --vibration --qpoint P: P with P labels; (1/4, 1/4, 1/4): "
           "PA with PA labels",
           code_p == 0 and code_k == 0
           and "* Selected Q point *\n  P        (-1/4, -1/4, -1/4)\n" in out_p
           and "  P        (-1/4, -1/4, -1/4)\n"
           in out_p.split("* Q points (primitive) *")[1].split("* Selected Q point *")[0]
           and len(labels_p) == 18 and set(labels_p) == {"P"}
           and "* Selected Q point *\n  PA       (1/4, 1/4, 1/4)\n" in out_k
           and len(labels_k) == 18 and set(labels_k) == {"PA"}, out_p + out_k)

    # the negative coordinates of the list can be given back as fractions
    # (argparse took -1/4 for an option)
    with tempfile.TemporaryDirectory() as tmp:
        _write_cell_variant(poscar_alpo4, os.path.join(tmp, "POSCAR_AlPO4_shifted"),
                            shift=(0.0731, 0.1593, 0.2417))
        code, out = run_phonon(["--vibration", "-c", "POSCAR_AlPO4_shifted",
                                "--qpoint", "-1/4", "-1/4", "-1/4"], cwd=tmp)
    labels = re.findall(r"Mode Space\s+\d+: irrep = ([A-Z]+)\d", out)
    report("--vibration --qpoint -1/4 -1/4 -1/4 (negative fractions): P with P labels",
           code == 0 and "* Selected Q point *\n  P        (-1/4, -1/4, -1/4)\n" in out
           and len(labels) == 18 and set(labels) == {"P"}, out)

    # the coordinates of a listed point are named as the list names them: a
    # point moved into the frame of the labels was named by its ISO-IR type
    # instead (Ce2O3 at a generic origin: H_2 -> H; RbBe2F5 re-based to
    # (b, c, a+c): Z -> X, X -> Z, V_2 -> T, ...)
    probe = (
        "import contextlib, io\n"
        "import numpy as np\n"
        "from phonopy.interface.calculator import read_crystal_structure\n"
        "from phonopy.structure.atoms import PhonopyAtoms\n"
        "from crystod.phonon import SymmetryOnlyVibrations\n"
        f"root = {os.path.join(ROOT, 'example', 'test_POSCARs')!r}\n"
        "shift = np.array([0.0731, 0.1593, 0.2417])\n"
        "for source, rows in (('150_PPOSCAR_Ce2O3', np.eye(3)),\n"
        "                     ('1_PPOSCAR_RbBe2F5', [[0, 1, 0], [0, 0, 1], [1, 0, 1]])):\n"
        "    cell, _ = read_crystal_structure(root + '/' + source, interface_mode='vasp')\n"
        "    C = np.asarray(rows, dtype=float)\n"
        "    moved = PhonopyAtoms(cell=C @ cell.cell, numbers=cell.numbers, scaled_positions=(\n"
        "        (cell.scaled_positions + shift) @ np.linalg.inv(C)) % 1.0)\n"
        "    with contextlib.redirect_stdout(io.StringIO()):\n"
        "        vibrations = SymmetryOnlyVibrations(moved)\n"
        "        points = vibrations.get_high_symmetry_qpoints()\n"
        "        found = {name: vibrations.resolve_qpoint([repr(float(v)) for v in q])[0]\n"
        "                 for name, q in points.items()}\n"
        "    for name in points:\n"
        "        print('NAME', source, name, found[name])\n"
    )
    code, out = run_python(probe)
    names = re.findall(r"^NAME (\S+) (\S+) (\S+)$", out, re.M)
    report("--vibration --qpoint COORDS of a listed point: the name of the list "
           "(Ce2O3 H_2 at a generic origin, re-based RbBe2F5)",
           code == 0 and len(names) == 16
           and ("150_PPOSCAR_Ce2O3", "H_2", "H_2") in names
           and ("1_PPOSCAR_RbBe2F5", "Z", "Z") in names
           and all(listed == found for _, listed, found in names), out)
    # the same for the ISO-IR list --vibration prints: the coordinates of every
    # listed point, and of its -k, are named as the frame of the labels names them
    probe = probe.replace("points = vibrations.get_high_symmetry_qpoints()",
                          "points = vibrations.get_special_qpoints()").replace(
        "vibrations.resolve_qpoint(", "vibrations.resolve_special_qpoint(")
    code, out = run_python(probe)
    names = re.findall(r"^NAME (\S+) (\S+) (\S+)$", out, re.M)
    report("--vibration --qpoint COORDS of a listed ISO-IR point: its listed name "
           "(Ce2O3 at a generic origin, re-based RbBe2F5)",
           code == 0 and len(names) == 14 and "H_2" not in out
           and all(listed == found for _, listed, found in names), out)

    # a seekpath point moved into the frame of the labels is snapped to
    # simple fractions only where its coordinates are such fractions (a point
    # with lattice-dependent coordinates is not shifted along its line);
    # stub frames move the made-up point Q_0 at zeta to -zeta
    probe = (
        "import contextlib, io\n"
        "from unittest import mock\n"
        "import numpy as np\n"
        "from phonopy.interface.calculator import read_crystal_structure\n"
        "from crystod.phonon import SymmetryOnlyVibrations\n"
        f"cell, _ = read_crystal_structure({poscar_alpo4!r}, interface_mode='vasp')\n"
        "with contextlib.redirect_stdout(io.StringIO()):\n"
        "    vibrations = SymmetryOnlyVibrations(cell)\n"
        "zeta = np.array([0.1234567, 0.5, 0.0])\n"
        "class Frame:\n"
        "    P = Pinv = np.eye(3)\n"
        "    def __init__(self, sign):\n"
        "        self.sign = sign\n"
        "    def kpoint_name(self, k):\n"
        "        if np.allclose(k, self.sign * zeta, atol=1e-12):\n"
        "            return 'Q'\n"
        "        return 'QA' if np.allclose(k, -self.sign * zeta, atol=1e-12) else 'GM'\n"
        "vibrations._get_isoir_labeler = lambda: Frame(-1)\n"
        "vibrations._spglib_frame_labeler = lambda: Frame(1)\n"
        "path = {'point_coords': {'GAMMA': [0.0, 0.0, 0.0], 'Q_0': list(zeta)},\n"
        "        'primitive_lattice': vibrations.primitive_cell.cell, 'path': []}\n"
        "with mock.patch('seekpath.get_path', return_value=path):\n"
        "    print('MOVED', vibrations.get_high_symmetry_qpoints()['Q_0'])\n"
    )
    code, out = run_python(probe)
    report("a moved seekpath point with lattice-dependent coordinates is not snapped",
           code == 0 and "MOVED [-0.1234567, -0.5, 0.0]" in out, out)

    # P4/n in origin choice 1 is not the ISO-IR setting: its frame is the
    # canonical frame of the crystal (spglib's frame composed with the
    # twofold axis along x), in which GM3 and GM4 are those of the raw ISO-IR
    # table; the earlier frame exchanged GM3 and GM4
    poscar_in5 = os.path.join(ROOT, "example", "test_POSCARs", "85_PPOSCAR_IN5")
    code, out = run_phonon(["--vibration", "-c", poscar_in5, "--qpoint", "GM"])
    spaces = listed(out)
    report("P4/n origin choice 1 (IN5) at GM: spaces 10-14 GM4+, 15-19 GM4-, "
           "27-31 GM3+, 32-36 GM3-",
           code == 0 and len(spaces) == 36
           and spaces[9:14] == ["GM4+(1)"] * 5 and spaces[14:19] == ["GM4-(1)"] * 5
           and spaces[26:31] == ["GM3+(1)"] * 5 and spaces[31:36] == ["GM3-(1)"] * 5,
           out)

    # Si (Fd-3m, two origins on the atoms): the frame of crystod-phonon
    # --irreps (Si on 8a), not the other origin (L2-, L1+, L3-, L3+; W1, W1, W2)
    poscar_si = os.path.join(ROOT, "example", "test_POSCARs", "227_PPOSCAR_Si")
    code_l, out_l = run_phonon(["--vibration", "-c", poscar_si, "--qpoint", "L"])
    code_w, out_w = run_phonon(["--vibration", "-c", poscar_si, "--qpoint", "W"])
    report("Si --vibration: L1+, L2-, L3+, L3- at L and W2, W2, W1 at W",
           code_l == 0 and code_w == 0
           and listed(out_l) == ["L1+(1)", "L2-(1)", "L3+(2)", "L3-(2)"]
           and listed(out_w) == ["W2(2)", "W2(2)", "W1(2)"], out_l + out_w)

    # a diagonal supercell input is labelled like the cell it repeats (the
    # labeller once named the R spaces of a 2x2x2 SrTiO3 cell GM2-, GM3-,
    # GM4-, ...)
    code, out_cell = run_phonon(["--vibration", "-c", POSCAR_SrTiO3, "--qpoint", "R"])
    with tempfile.TemporaryDirectory() as tmp:
        for dim in ((2, 2, 2), (2, 1, 1)):
            size = "x".join(str(n) for n in dim)
            name = f"SPOSCAR_{size}_SrTiO3"
            _write_cell_variant(POSCAR_SrTiO3, os.path.join(tmp, name), dim)
            code, out = run_phonon(["--vibration", "-c", name, "--qpoint", "R"], cwd=tmp)
            report(f"{size} supercell of SrTiO3 at R: R2-, R3-, R4-, ... as the cell",
                   code == 0 and listed(out)[:3] == ["R2-(1)", "R3-(2)", "R4-(3)"]
                   and listed(out) == listed(out_cell), out)

    with tempfile.TemporaryDirectory() as tmp:
        out_poscar = os.path.join(tmp, "POSCAR_vibration")
        code, out = run_phonon(
            ["--vibration", "-c", POSCAR_ScF3, "--qpoint", "R",
             "--mode-index", "1", "--component-index", "1", "--output", out_poscar]
        )
        report("mode export exit 0", code == 0, out)
        report("commensurate supercell reported", "supercell size" in out, out)
        report("displaced POSCAR written", os.path.isfile(out_poscar))

    # A written component is the real displacement field of a genuine partner
    # of its irrep space, with the Bloch factor exp(2 pi i q.x_j) of every
    # atom, so it freezes into the isotropy subgroup of its irrep (single arm:
    # crystod-group --parent I4/mmm --irrep X3- lists Cmcm, --parent Cmcm
    # --irrep Y2- lists Pnma, --parent Pm-3m --irrep R4+ lists I4/mcm for
    # (0,0,a)). The per-atom factor used to be missing and the raw projected
    # row was frozen as it came: Sr3Ti2O7 X3- froze into Ama2 at (1/2, 1/2, 0)
    # (4 of 6 spaces) and into Cmce at (-1/2, 1/2, 0), the Cmcm Y2- spaces of
    # SrLi2Nb2O7 into Pna2_1, and ScF3 R4+ into I4/mmm at (-1/2, 1/2, 1/2).
    import spglib
    from ase.io import read as _ase_read
    from ase.io import write as _ase_write

    def spaces_of(out: str, irrep: str) -> list[tuple[int, int]]:
        return [(int(match.group(1)), int(match.group(2))) for match in re.finditer(
            rf"Mode Space\s+(\d+): irrep = {re.escape(irrep)}\(\d+\), dimension = (\d+)", out)]

    def frozen(cell: str, qpoint: list[str], irrep: str, cwd: str,
               tag: str) -> list[tuple[str, str]]:
        """(file, space group at 1e-5) of every component of every space of ``irrep``."""
        code, out = run_phonon(["--vibration", "-c", cell, "--qpoint", *qpoint], cwd=cwd)
        results = []
        for number, dim in spaces_of(out, irrep):
            for component in range(1, dim + 1):
                name = f"{tag}_space{number}_component{component}"
                code, run_out = run_phonon(
                    ["--vibration", "-c", cell, "--qpoint", *qpoint,
                     "--mode-index", str(number), "--component-index", str(component),
                     "--output", name], cwd=cwd)
                if code != 0 or not os.path.isfile(os.path.join(cwd, name)):
                    results.append((name, f"exit {code}: {run_out[-200:]}"))
                    continue
                atoms = _ase_read(os.path.join(cwd, name), format="vasp")
                dataset = spglib.get_symmetry_dataset(
                    (atoms.cell[:], atoms.get_scaled_positions(), atoms.numbers), symprec=1e-5)
                results.append((name, f"{dataset.international} (#{dataset.number})"
                                if dataset is not None else "no symmetry"))
        return results

    def all_are(results: list[tuple[str, str]], expected: str, count: int) -> bool:
        return len(results) == count and all(group == expected for _, group in results)

    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(os.path.join(MODULATION_STO_DIR, "139_PPOSCAR_Sr3Ti2O7"), tmp)
        by_q = {}
        for qpoint, tag in ((["0.5", "0.5", "0"], "PLUS"), (["-0.5", "0.5", "0"], "MINUS")):
            by_q[tag] = frozen("139_PPOSCAR_Sr3Ti2O7", qpoint, "X3-", tmp, tag)
            report(f"Sr3Ti2O7 X3- at q = ({', '.join(qpoint)}): all 6 spaces freeze "
                   "into Cmcm at 1e-5",
                   all_are(by_q[tag], "Cmcm (#63)", 6),
                   "\n".join(f"{name}: {group}" for name, group in by_q[tag]))
        differences = []
        for (plus, _), (minus, _) in zip(by_q["PLUS"], by_q["MINUS"]):
            try:
                first = _ase_read(os.path.join(tmp, plus), format="vasp")
                second = _ase_read(os.path.join(tmp, minus), format="vasp")
                differences.append(max(np.abs(first.cell[:] - second.cell[:]).max(),
                                       np.abs(first.positions - second.positions).max()))
            except (OSError, ValueError, IndexError):
                differences.append(np.inf)
        report("Sr3Ti2O7 X3-: q and q + G write the same structures",
               len(differences) == 6 and max(differences) < 1e-8,
               f"largest difference {max(differences, default=np.inf):.2e} A")

        cif = os.path.join(ROOT, "example", "23_symmetry_mode", "debug1_LSNO",
                           "CONTCAR-POSCAR_SrLi2Nb2O7_Cmcm.cif")
        _ase_write(os.path.join(tmp, "POSCAR_SrLi2Nb2O7_Cmcm"),
                   _ase_read(cif, format="cif"), format="vasp", direct=True)
        results = frozen("POSCAR_SrLi2Nb2O7_Cmcm", ["Y"], "Y2-", tmp, "Y2m")
        report("SrLi2Nb2O7 (Cmcm) Y2-: all 10 spaces freeze into Pnma at 1e-5",
               all_are(results, "Pnma (#62)", 10),
               "\n".join(f"{name}: {group}" for name, group in results))

        shutil.copy(POSCAR_ScF3, tmp)
        results = frozen(os.path.basename(POSCAR_ScF3), ["-0.5", "0.5", "0.5"], "R4+", tmp,
                         "R4p")
        report("ScF3 R4+ at q + G = (-0.5, 0.5, 0.5): every partner freezes into "
               "I4/mcm at 1e-5",
               all_are(results, "I4/mcm (#140)", 3),
               "\n".join(f"{name}: {group}" for name, group in results))

        # the written pattern of an irrep that occurs once is the --modulation mode
        # of that irrep (no mass weighting enters: its space holds one species)
        code, out = run_python(
            "import warnings, contextlib, io\n"
            "warnings.filterwarnings('ignore')\n"
            "import numpy as np, phonopy\n"
            "from phonopy.structure.atoms import PhonopyAtoms\n"
            "from crystod import phonon\n"
            f"ph = phonopy.load({os.path.join(MODULATION_DIR, 'phonopy_params.yaml')!r}, "
            "log_level=0)\n"
            "prim = ph.primitive\n"
            "cell = PhonopyAtoms(numbers=prim.numbers, scaled_positions=prim.scaled_positions, "
            "cell=prim.cell)\n"
            "q = [0.5, 0.5, 0.5]\n"
            "with contextlib.redirect_stdout(io.StringIO()):\n"
            "    vib = phonon.SymmetryOnlyVibrations(cell, standardize=False)\n"
            "    _, spaces, labels = vib.describe_mode_spaces(q)\n"
            "    adapted = vib.get_symmetry_adapted_spaces(q)\n"
            "    mod = phonon.SymmetryAdaptedModulation(phonon=ph, qpoint=q)\n"
            "phase = np.exp(2j * np.pi * np.repeat(prim.scaled_positions @ q, 3))\n"
            "rows = adapted[labels.index('R4+(3)')] * phase\n"
            "modes = np.array(mod.mode_vectors[:3])\n"
            "print('max difference', np.abs(rows - modes).max())\n",
            cwd=tmp,
        )
        match = re.search(r"max difference (\S+)", out)
        report("ScF3 R4+: the written partners are the --modulation modes 1-3",
               code == 0 and match is not None and float(match.group(1)) < 1e-8, out)

        # The mode spaces of a repeated irrep are a basis of its patterns fixed
        # by the structure, not the copies spgrep's projection returns, which
        # depend on how q is written: CaTiO3 (Pnma) mode space 1 (X1 or X2, 13
        # and 17 occurrences) used to write structures 0.3 A apart at 0.5 0 0
        # and 1.5 0 0 under an identical listing, the Cmcm R1 spaces of
        # SrLi2Nb2O7 and the L1 spaces of AlF3 (R-3c) were permuted under q + G,
        # and the complex pair T1/T2 of AlF3 handed its shared real partners to
        # whichever label was listed first. Where symmetry leaves the partner
        # basis open (the real space of the complex pair GM3/GM4 of P-4
        # MnZnGa4Se8, whose atoms move on circles; the pseudo-real R1 of
        # P2_12_12_1 GeF2) the displacement forms were degenerate and round-off
        # turned the basis (by 45 degrees for GM3 at 0 -1 -1).
        catio3 = os.path.join(ROOT, "example", "test_POSCARs", "62_PPOSCAR_CaTiO3")
        shutil.copy(catio3, tmp)
        written_by_q = []
        for qpoint, name in ((["0.5", "0", "0"], "CTO_X"), (["1.5", "0", "0"], "CTO_XG")):
            code, run_out = run_phonon(
                ["--vibration", "-c", "62_PPOSCAR_CaTiO3", "--qpoint", *qpoint,
                 "--mode-index", "1", "--output", name], cwd=tmp)
            listing = re.findall(r"Mode Space\s+\d+: irrep = \S+", run_out)
            written_by_q.append((code, listing, os.path.join(tmp, name)))
        try:
            first = _ase_read(written_by_q[0][2], format="vasp")
            second = _ase_read(written_by_q[1][2], format="vasp")
            difference = max(np.abs(first.cell[:] - second.cell[:]).max(),
                             np.abs(first.positions - second.positions).max())
        except (OSError, ValueError, IndexError):
            difference = np.inf
        report("CaTiO3 X, mode space 1: q and q + G write the same structure",
               all(code == 0 for code, _, _ in written_by_q)
               and written_by_q[0][1] == written_by_q[1][1] and difference < 1e-8,
               f"largest difference {difference:.2e} A")

        code, out = run_python(
            "import warnings, contextlib, io\n"
            "warnings.filterwarnings('ignore')\n"
            "import numpy as np\n"
            "from phonopy.interface.calculator import read_crystal_structure\n"
            "from crystod import phonon\n"
            "def by_label(vib, q):\n"
            "    with contextlib.redirect_stdout(io.StringIO()):\n"
            "        _, _, labels = vib.describe_mode_spaces(q)\n"
            "        adapted = vib.get_symmetry_adapted_spaces(q)\n"
            "    size = vib.get_supercell_size(q)\n"
            "    found = {}\n"
            "    for label, rows in zip(labels, adapted):\n"
            "        found.setdefault(label, []).append(\n"
            "            [vib.get_supercell_displacements(q, row, size)[1] for row in rows])\n"
            "    return found\n"
            "cases = [('62_PPOSCAR_CaTiO3', 'X', [1, 0, 0]),\n"
            "         ('POSCAR_SrLi2Nb2O7_Cmcm', 'R', [1, -1, 1]),\n"
            f"         ({os.path.join(ROOT, 'example', 'test_POSCARs', '167_PPOSCAR_AlF3')!r}, "
            "'L', [1, -1, 1]),\n"
            f"         ({os.path.join(ROOT, 'example', 'test_POSCARs', '167_PPOSCAR_AlF3')!r}, "
            "'T', [1, -1, 1]),\n"
            f"         ({os.path.join(ROOT, 'example', 'test_POSCARs', '81_PPOSCAR_MnZnGa4Se8')!r}, "
            "'GAMMA', [0, -1, -1]),\n"
            f"         ({os.path.join(ROOT, 'example', 'test_POSCARs', '19_PPOSCAR_GeF2')!r}, "
            "'R', [1, 0, 0])]\n"
            "for path, point, shift in cases:\n"
            "    cell, _ = read_crystal_structure(path, interface_mode='vasp')\n"
            "    with contextlib.redirect_stdout(io.StringIO()):\n"
            "        vib = phonon.SymmetryOnlyVibrations(cell)\n"
            "    _, q = vib.resolve_qpoint([point])\n"
            "    reference = by_label(vib, q)\n"
            "    worst, repeated = 0.0, max(len(spaces) for spaces in reference.values())\n"
            "    for other in (list(np.add(q, shift)), list(-np.array(q))):\n"
            "        found = by_label(vib, other)\n"
            "        for label, spaces in reference.items():\n"
            "            for first, second in zip(spaces, found[label]):\n"
            "                for a, b in zip(first, second):\n"
            "                    worst = max(worst, float(np.abs(a - b).max()))\n"
            "    print(f'{point} of {path.split(\"/\")[-1]}: most occurrences {repeated}, "
            "max difference {worst:.2e}')\n",
            cwd=tmp,
        )
        rows = re.findall(r"most occurrences (\d+), max difference (\S+)", out)
        report("q, q + G and -q write the same structure per irrep label, occurrence and "
               "component (CaTiO3 X, SrLi2Nb2O7 R, AlF3 L and T, MnZnGa4Se8 GM, GeF2 R)",
               code == 0 and len(rows) == 6
               and all(int(count) > 1 and float(value) < 1e-8 for count, value in rows), out)


# ---------------------------------------------------------------- 35. crystod-phonon extras
def _modulate_command_pairs(out: str) -> list[tuple[str, str]]:
    """(file, command) pairs printed by ``crystod-phonon --subgroup --modulate``."""
    pairs = []
    lines = out.splitlines()
    for index, line in enumerate(lines[:-1]):
        match = re.match(r"^\S+\s+\S+\s+-> (MPOSCAR_\S+)$", line.strip())
        if match and lines[index + 1].strip().startswith("crystod-phonon"):
            pairs.append((match.group(1), lines[index + 1].strip()))
    return pairs


def _replay_modulate_commands(pairs: list[tuple[str, str]], cwd: str) -> list[str]:
    """Run every printed ``--modulation`` command in ``cwd``.

    Returns the files whose command fails or writes a different structure.
    """
    broken = []
    check_path = os.path.join(cwd, "REPRO_CHECK")
    for name, command in pairs:
        argv = shlex.split(command)[1:] + ["--output", "REPRO_CHECK"]
        code, repro_out = run_phonon(argv, cwd=cwd)
        if code != 0:
            broken.append(f"{name}: exit {code}\n{repro_out[-300:]}")
        elif open(check_path).read() != open(os.path.join(cwd, name)).read():
            broken.append(f"{name}: reproduce command gives a different structure")
    if os.path.isfile(check_path):
        os.remove(check_path)
    return broken


def _measure_modulate_structures(directory: str, parent_name: str,
                                 expected: dict[str, tuple[int, int, int]]) -> list[str]:
    """Re-measure (number, size, index) of each written structure at 1e-5.

    Independent of the code that wrote the files; returns the mismatches.
    """
    from ase.io import read as ase_read

    from crystod.modulation import classify_distorted_structure

    parent = ase_read(os.path.join(directory, parent_name), format="vasp")
    parent_cell = (parent.cell[:], parent.get_scaled_positions(),
                   parent.get_atomic_numbers())
    mismatches = []
    for name, key in expected.items():
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            mismatches.append(f"{name}: not written")
            continue
        atoms = ase_read(path, format="vasp")
        number, _symbol, size, index = classify_distorted_structure(
            atoms, parent_cell, symprec=1e-5
        )
        if (number, size, index) != key:
            mismatches.append(f"{name}: measured {(number, size, index)}, expected {key}")
    return mismatches


def test_35_phonon_command() -> None:
    print("\n[35] crystod-phonon (sectioned command: 7 phonon modes)")

    def run_phonon(args: list[str], cwd: str | None = None) -> tuple[int, str]:
        return run_module("crystod.cli.phonon", args, cwd)

    # --vibration (structure only, fast)
    code, out = run_phonon(["--vibration", "-c", POSCAR_ScF3, "--qpoint", "R"])
    report("--vibration exit 0", code == 0, out)
    report("irrep-grouped mode spaces listed", "Mode Space" in out, out)

    # --irreps with nine-value diagonal dim
    if os.path.isdir(PHONON_IRREP_DIR):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("221_PPOSCAR_SrTiO3", "FORCE_SETS"):
                shutil.copy(os.path.join(PHONON_IRREP_DIR, name), tmp)
            code, out = run_phonon(
                ["--irreps", "--dim", "4", "0", "0", "0", "4", "0", "0", "0", "4",
                 "-c", "221_PPOSCAR_SrTiO3"],
                cwd=tmp,
            )
            report("--irreps (nine-value --dim) exit 0", code == 0, out)
            yaml_path = os.path.join(tmp, "phonon_irreps.yaml")
            report("phonon_irreps.yaml written", os.path.isfile(yaml_path))
            if os.path.isfile(yaml_path):
                report("yaml contains GM point irreps", "GM" in open(yaml_path).read())
    else:
        report("phonon_irrep example data found", False, PHONON_IRREP_DIR)

    # --vector
    if os.path.isdir(PHONON_VECTOR_DIR):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("227_PPOSCAR_Si", "FORCE_CONSTANTS"):
                shutil.copy(os.path.join(PHONON_VECTOR_DIR, name), tmp)
            code, out = run_phonon(
                ["--vector", "--dim", "4 4 4", "-c", "227_PPOSCAR_Si",
                 "--readfc", "--qpoint", "GM", "--mode", "4"],
                cwd=tmp,
            )
            report("--vector exit 0", code == 0, out)
            report("mode irrep labeled", "GM5+(3)" in out, out)
            report("VESTA export written (irrep tag + 1-based number)",
                   os.path.isfile(os.path.join(tmp, "POSCAR_Si_GM_mode4_GM5+.vesta")))
            report("mode table text file written",
                   os.path.isfile(os.path.join(tmp, "phonon_modes_Si_GM.txt")))

            code, out = run_phonon(
                ["--vector", "--dim", "4 4 4", "-c", "227_PPOSCAR_Si",
                 "--readfc", "--qpoint", "GM", "--mode", "0"],
                cwd=tmp,
            )
            report("--mode 0 rejected (numbering is 1-based)",
                   code != 0 and "1-based" in out and "Traceback" not in out, out)

    # --fatband and --lt (small npoints)
    if os.path.isdir(PHONON_FATBAND_DIR):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("221_PPOSCAR_ScF3", "FORCE_SETS"):
                shutil.copy(os.path.join(PHONON_FATBAND_DIR, name), tmp)
            code, out = run_phonon(
                ["--fatband", "--dim", "4 4 4", "-c", "221_PPOSCAR_ScF3",
                 "--npoints", "11", "--element", "Sc"],
                cwd=tmp,
            )
            report("--fatband exit 0", code == 0, out)
            report("fatband_Sc.pdf written",
                   os.path.isfile(os.path.join(tmp, "fatband_Sc.pdf")))
            code, out = run_phonon(
                ["--lt", "--dim", "4 4 4", "-c", "221_PPOSCAR_ScF3", "--npoints", "11"],
                cwd=tmp,
            )
            report("--lt exit 0", code == 0, out)
            report("phonon_band_LT.pdf written",
                   os.path.isfile(os.path.join(tmp, "phonon_band_LT.pdf")))

    # --modulation
    yaml_path = os.path.join(MODULATION_DIR, "phonopy_params.yaml")
    if os.path.isfile(yaml_path):
        with tempfile.TemporaryDirectory() as tmp:
            code, out = run_phonon(
                ["--modulation", "--yaml", yaml_path, "--qpoint", "0.5", "0.5", "0.5",
                 "--mode", "1", "2", "3", "--amplitude", "0.3",
                 "--output", os.path.join(tmp, "POSCAR_mod")],
                cwd=tmp,
            )
            report("--modulation exit 0", code == 0, out)
            report("R4+(a,a,a) -> R-3c", "R-3c" in out, out)

    # --modulation straight from POSCAR + FORCE_SETS: the same structure as the
    # phonopy_params.yaml route, byte for byte, with the supercell taken from
    # phonopy_disp.yaml, inferred from FORCE_SETS, or given as --dim
    if os.path.isdir(MODULATION_DIR):
        modulation_argv = ["--qpoint", "0.5", "0.5", "0.5",
                           "--mode", "1", "2", "3", "--amplitude", "0.3"]
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("221_PPOSCAR_ScF3", "FORCE_SETS", "phonopy_params.yaml",
                         "phonopy_disp.yaml"):
                shutil.copy(os.path.join(MODULATION_DIR, name), tmp)
            code, out = run_phonon(
                ["--modulation", "--yaml", "phonopy_params.yaml", *modulation_argv,
                 "--output", "REF"],
                cwd=tmp,
            )
            report("--modulation --yaml reference run exit 0", code == 0, out)
            reference = open(os.path.join(tmp, "REF")).read() if code == 0 else ""

            code, out = run_phonon(
                ["--modulation", "-c", "221_PPOSCAR_ScF3", *modulation_argv,
                 "--output", "FROM_CELL"],
                cwd=tmp,
            )
            report("--modulation -c (no --yaml, no --dim) exit 0", code == 0, out)
            report("supercell read from phonopy_disp.yaml",
                   "read from phonopy_disp.yaml" in out, out)
            report("-c route reproduces the --yaml structure exactly",
                   code == 0 and open(os.path.join(tmp, "FROM_CELL")).read() == reference,
                   out)

            code, out = run_phonon(
                ["--modulation", "-c", "221_PPOSCAR_ScF3", "--dim", "4 4 4",
                 *modulation_argv, "--output", "FROM_DIM"],
                cwd=tmp,
            )
            report("--modulation -c with explicit --dim identical",
                   code == 0 and open(os.path.join(tmp, "FROM_DIM")).read() == reference,
                   out)

            code, out = run_phonon(
                ["--modulation", "-c", "221_PPOSCAR_ScF3", "--dim", "2 2 2",
                 *modulation_argv, "--output", "BAD"],
                cwd=tmp,
            )
            report("--modulation with a wrong --dim rejected cleanly",
                   code != 0 and "could not build force constants" in out
                   and "supercell 2x2x2" in out and "Traceback" not in out, out)

            code, out = run_phonon(
                ["--modulation", "--yaml", "phonopy_params.yaml",
                 "-c", "221_PPOSCAR_ScF3", *modulation_argv],
                cwd=tmp,
            )
            report("--modulation with both --yaml and -c rejected cleanly",
                   code != 0 and "not both" in out and "Traceback" not in out, out)

        with tempfile.TemporaryDirectory() as tmp:
            # no yaml at all: the supercell has to come out of FORCE_SETS
            for name in ("221_PPOSCAR_ScF3", "FORCE_SETS"):
                shutil.copy(os.path.join(MODULATION_DIR, name), tmp)
            code, out = run_phonon(
                ["--modulation", "-c", "221_PPOSCAR_ScF3", *modulation_argv,
                 "--output", "INFERRED"],
                cwd=tmp,
            )
            report("--modulation infers 4x4x4 from FORCE_SETS",
                   code == 0 and "Supercell 4x4x4 inferred" in out, out)
            report("inferred-supercell structure identical to the --yaml one",
                   code == 0 and open(os.path.join(tmp, "INFERRED")).read() == reference,
                   out)

    # error handling
    code, out = run_phonon(["--irreps", "-c", POSCAR_ScF3])
    report("--irreps without --dim rejected cleanly",
           code != 0 and "requires --dim" in out and "Traceback" not in out, out)
    code, out = run_phonon(["--vector", "--dim", "4 4 4", "-c", POSCAR_ScF3])
    report("--vector without --qpoint rejected cleanly",
           code != 0 and "requires --qpoint" in out and "Traceback" not in out, out)
    code, out = run_phonon(["--vibration", "-c", POSCAR_ScF3])
    report("--vibration without --qpoint rejected cleanly",
           code != 0 and "--list-qpoints" in out and "Traceback" not in out, out)
    code, out = run_phonon(["--vibration", "-c", POSCAR_ScF3, "--qpoint1", "0", "0", "0"])
    report("numbered args outside --modulation rejected cleanly",
           code != 0 and "unrecognized" in out and "Traceback" not in out, out)

    # --subgroup: isotropy subgroups of the imaginary modes
    if os.path.isdir(PHONON_IRREP_DIR):
        code, out = run_phonon(
            ["--subgroup", "--dim", "4 4 4", "-c", "221_PPOSCAR_SrTiO3"],
            cwd=PHONON_IRREP_DIR,
        )
        report("--subgroup (q scan) exit 0", code == 0, out)
        report("--subgroup finds the R-point instability",
               "q = (0.5, 0.5, 0.5) (R)" in out and "-1.086" in out, out)
        report("--subgroup labels the unstable irrep",
               "irrep R5-" in out and "degeneracy 3" in out, out)
        for direction, subgroup in (("R5-(0,0,a)", "140 I4/mcm"),
                                    ("R5-(a,a,a)", "167 R-3c"),
                                    ("R5-(0,a,a)", "74 Imma")):
            report(f"--subgroup lists {direction} -> {subgroup}",
                   direction in out and subgroup in out, out)
        code, out = run_phonon(
            ["--subgroup", "--dim", "4 4 4", "-c", "221_PPOSCAR_SrTiO3",
             "--qpoint", "GM"],
            cwd=PHONON_IRREP_DIR,
        )
        report("--subgroup --qpoint GM reports no instability",
               code == 0 and "No imaginary mode" in out, out)
        code, out = run_phonon(
            ["--subgroup", "--dim", "4 4 4", "-c", "221_PPOSCAR_SrTiO3",
             "--qpoint", "NOSUCHLABEL"],
            cwd=PHONON_IRREP_DIR,
        )
        report("--subgroup unknown q label rejected cleanly",
               code != 0 and "Unknown q-point label" in out
               and "Traceback" not in out, out)

        # --modulate: one distorted structure per order-parameter direction,
        # each verified against the enumerated (number, size, index)
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("221_PPOSCAR_SrTiO3", "FORCE_SETS"):
                shutil.copy(os.path.join(PHONON_IRREP_DIR, name), tmp)
            code, out = run_phonon(
                ["--subgroup", "--dim", "4 4 4", "-c", "221_PPOSCAR_SrTiO3",
                 "--qpoint", "R", "--modulate"],
                cwd=tmp,
            )
            report("--subgroup --modulate exit 0", code == 0, out)
            report("--modulate reports the distorted structures",
                   "Distorted structures" in out, out)
            report("--modulate prints the reproducing --modulation command",
                   "crystod-phonon --modulation -c 221_PPOSCAR_SrTiO3 --dim \"4 4 4\" "
                   "--qpoint 0.5 0.5 0.5 --mode 1 --amplitude 0.3" in out, out)
            # the printed command is a promise: run each one and require it to
            # regenerate byte-identically the file it is printed next to
            pairs = _modulate_command_pairs(out)
            report("--modulate prints one command per written structure",
                   len(pairs) == 6, out)
            broken = _replay_modulate_commands(pairs, tmp)
            report("every printed command regenerates its structure exactly",
                   not broken, "\n".join(broken))
            written = sorted(
                name for name in os.listdir(tmp) if name.startswith("MPOSCAR_")
            )
            report("--modulate writes all six R5- directions",
                   len(written) == 6, "\n".join(written) or out)
            expected = {
                "MPOSCAR_R_R5-_0-0-a_I4mcm": (140, 2, 6),
                "MPOSCAR_R_R5-_a-a-a_R-3c": (167, 2, 8),
                "MPOSCAR_R_R5-_0-a-a_Imma": (74, 2, 12),
                "MPOSCAR_R_R5-_0-a-b_C2m": (12, 2, 24),
                "MPOSCAR_R_R5-_a-a-b_C2c": (15, 2, 24),
                "MPOSCAR_R_R5-_a-b-c_P-1": (2, 2, 48),
            }
            report("--modulate names every direction after its subgroup",
                   set(written) == set(expected), "\n".join(written))
            # re-measure each structure independently of the code that wrote it
            mismatches = _measure_modulate_structures(tmp, "221_PPOSCAR_SrTiO3", expected)
            report("every written structure has the space group it claims",
                   not mismatches, "\n".join(mismatches))

        code, out = run_phonon(
            ["--subgroup", "--dim", "4 4 4", "-c", "221_PPOSCAR_SrTiO3",
             "--qpoint", "R", "--amplitude", "0.3"],
            cwd=PHONON_IRREP_DIR,
        )
        report("--amplitude without --modulate rejected cleanly",
               code != 0 and "only used by --subgroup with --modulate" in out
               and "Traceback" not in out, out)
    else:
        report("phonon_irrep example data found", False, PHONON_IRREP_DIR)

    # --modulate on a two-arm star (X of I4/mmm Sr3Ti2O7): (a;a) and (a;b)
    # freeze both arms together. The second arm used to come out as a wrong
    # pattern, so only (0;a) was written and the other two were reported as
    # "no candidate reproduced".
    if os.path.isdir(MODULATION_STO_DIR):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("139_PPOSCAR_Sr3Ti2O7", "FORCE_SETS"):
                shutil.copy(os.path.join(MODULATION_STO_DIR, name), tmp)
            code, out = run_phonon(
                ["--subgroup", "--dim", "4 4 4", "-c", "139_PPOSCAR_Sr3Ti2O7",
                 "--qpoint", "X", "--threshold", "2.5", "--modulate"],
                cwd=tmp,
            )
            report("Sr3Ti2O7 --subgroup --modulate at X exit 0", code == 0, out)
            report("Sr3Ti2O7 --modulate reproduces every X3- direction (no note)",
                   "no candidate reproduced" not in out, out)
            expected = {
                "MPOSCAR_X_X3-_0_a_Cmcm": (63, 2, 4),
                "MPOSCAR_X_X3-_a_a_P4_2mnm": (136, 4, 4),
                "MPOSCAR_X_X3-_a_b_Pnnm": (58, 4, 8),
            }
            written = sorted(
                name for name in os.listdir(tmp) if name.startswith("MPOSCAR_")
            )
            report("Sr3Ti2O7 --modulate writes the three X3- directions",
                   set(written) == set(expected), "\n".join(written) or out)
            pairs = _modulate_command_pairs(out)
            broken = _replay_modulate_commands(pairs, tmp)
            report("Sr3Ti2O7: every printed command regenerates its structure exactly",
                   len(pairs) == 3 and not broken, "\n".join(broken) or out)
            mismatches = _measure_modulate_structures(tmp, "139_PPOSCAR_Sr3Ti2O7", expected)
            report("Sr3Ti2O7: every written structure has the space group it claims",
                   not mismatches, "\n".join(mismatches))
    else:
        report("Sr3Ti2O7 example data found", False, MODULATION_STO_DIR)

    if os.path.isdir(MODULATION_DIR):
        code, out = run_phonon(
            ["--subgroup", "--yaml", "phonopy_params.yaml"], cwd=MODULATION_DIR
        )
        report("--subgroup --yaml on a stable structure exit 0", code == 0, out)
        report("--subgroup reports no imaginary mode for stable ScF3",
               "No imaginary mode" in out and "Traceback" not in out, out)
    code, out = run_phonon(["--subgroup", "-c", POSCAR_ScF3, "--dim", "4 4 4"])
    report("--subgroup without force data rejected cleanly",
           code != 0 and "FORCE_SETS not found" in out
           and "Traceback" not in out, out)

    if os.path.isdir(MODULATION_DIR):
        # argparse accepts abbreviations, so the --dim/--yaml guard must see
        # them too: it reads the parsed value, not the raw argv spelling
        for spelling in ("--y", "--ya", "--yaml"):
            code, out = run_phonon(
                ["--subgroup", "--dim", "4 4 4", "-c", "221_PPOSCAR_ScF3",
                 spelling, "phonopy_params.yaml"],
                cwd=MODULATION_DIR,
            )
            report(f"--dim with {spelling} rejected as a conflict",
                   code != 0 and "not both" in out and "Traceback" not in out, out)
            code, out = run_phonon(
                ["--subgroup", spelling, "phonopy_params.yaml", "--qpoint", "R"],
                cwd=MODULATION_DIR,
            )
            report(f"{spelling} alone is accepted like --yaml",
                   code == 0 and "No imaginary mode" in out, out)

    # --threshold must not make the pre-existing --t abbreviation ambiguous
    if os.path.isdir(PHONON_IRREP_DIR):
        for spelling in ("--t", "--tol", "--tolerance"):
            code, out = run_phonon(
                ["--irreps", "--dim", "4 4 4", "-c", "221_PPOSCAR_SrTiO3",
                 spelling, "0.001"],
                cwd=PHONON_IRREP_DIR,
            )
            report(f"{spelling} still reaches --tolerance",
                   code == 0 and "ambiguous" not in out, out)

    # removed flat flags give replacement guidance
    for flag in ("--vibration", "--phonon-irrep", "--phonon-fatband", "--phonon-lt",
                 "--phonon-vector", "--modulation"):
        code, out = run_cli([flag])
        report(f"removed {flag} flag points to crystod-phonon",
               code != 0 and "is not a crystod option" in out and "crystod-phonon" in out, out)


    # --example: the bundled SrTiO3 cell + 4x4x4 FORCE_SETS (spec 3.2.2); the
    # required mode group must not get in the way of a bare --example
    code, out = run_phonon(["--example"])
    report("--example alone lists the bundled examples (no mode flag needed)",
           code == 0 and "SrTiO3" in out and "SrTiO3_subgroup" in out
           and "required" not in out, out)
    with tempfile.TemporaryDirectory() as tmp:
        code, out = run_phonon(["--example", "SrTiO3"], cwd=tmp)
        yaml_path = os.path.join(tmp, "phonon_irreps.yaml")
        report("--example SrTiO3 writes FORCE_SETS and phonon_irreps.yaml",
               code == 0
               and "Running: crystod-phonon --irreps --dim '4 4 4' -c 221_PPOSCAR_SrTiO3" in out
               and os.path.isfile(os.path.join(tmp, "FORCE_SETS"))
               and os.path.isfile(yaml_path), out)
        report("bundled SrTiO3 example labels the R-point modes (R5-)",
               os.path.isfile(yaml_path) and "R5-" in open(yaml_path).read(), out)

# ---------------------------------------------------------------- 36. crystod-mag
def test_36_spin_basis() -> None:
    print("\n[36] crystod-mag (AlNi3, Ni 3c cluster, Mn3Ir-type)")
    poscar = os.path.join(ROOT, "example", "test_POSCARs", "221_PPOSCAR_AlNi3")
    if not os.path.isfile(poscar):
        report("example POSCAR found", False, poscar)
        return
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(poscar, tmp)
        code, out = run_mag(
            ["-c", "221_PPOSCAR_AlNi3", "--element", "Ni",
             "--qpoint", "0", "0", "0", "--show-spin-direction"],
            cwd=tmp,
        )
        report("exit code 0", code == 0, out)
        report("decomposition 2 x GM4+ + GM5+",
               "2 x GM4+(3)" in out and "GM5+(3)" in out, out)
        report("FM dipole and AFM octupoles identified",
               "[FM, dipole]" in out and out.count("[AFM, octupole]") >= 2
               and "quadrupole" not in out, out)
        report("AFM net moment vanishes", "All AFM bases satisfy sum_i S_i = 0" in out, out)
        report("MAGMOM line printed for noncollinear input", "MAGMOM =" in out, out)
        for name in ("POSCAR_AlNi3_spin_GM4+_dipole_z.vesta",
                     "POSCAR_AlNi3_spin_GM4+_octupole_111.vesta",
                     "POSCAR_AlNi3_spin_GM5+_octupole_111.vesta"):
            report(f"{name} written", os.path.isfile(os.path.join(tmp, name)))

        code, out = run_mag(
            ["-c", "221_PPOSCAR_AlNi3", "--element", "Ni",
             "--qpoint", "R"],
            cwd=tmp,
        )
        report("R-point exit 0", code == 0, out)
        report("R-point 2x2x2 magnetic supercell and R4+ label",
               "2x2x2" in out and "R4+" in out, out)

        # a negative fraction is a coordinate, not an option (argparse took
        # -1/2 for one: 'expected at least one argument')
        code, out = run_mag(
            ["-c", "221_PPOSCAR_AlNi3", "--element", "Ni",
             "--qpoint", "-1/2", "-1/2", "-1/2"],
            cwd=tmp,
        )
        report("--qpoint -1/2 -1/2 -1/2 (negative fractions): R with R4+",
               code == 0 and "R4+" in out and "Selected q-point: R" in out, out)

        code, out = run_mag(
            ["-c", "221_PPOSCAR_AlNi3", "--element", "Cu",
             "--qpoint", "0", "0", "0"],
            cwd=tmp,
        )
        report("unknown element rejected cleanly",
               code != 0 and "is not in this POSCAR" in out and "Traceback" not in out, out)

        code, out = run_mag(
            ["-c", "221_PPOSCAR_AlNi3", "--element", "Ni"],
            cwd=tmp,
        )
        report("survey mode (no --qpoint) exit 0", code == 0, out)
        report("survey lists all special k points with axial irreps",
               out.count("k point (primitive)") >= 4
               and "2.0 [GM4+(3)]" in out and "R4+" in out, out)

    # 2-dim irreps with circular complex partners (x + iy, x - iy) must export
    # two ORTHOGONAL real components (x and y), not the same file twice
    poscar_327 = os.path.join(ROOT, "example", "36_spin_basis", "La3Ni2O7_I4mmm", "139_PPOSCAR_La3Ni2O7")
    if not os.path.isfile(poscar_327):
        report("La3Ni2O7 example POSCAR found", False, poscar_327)
        return
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(poscar_327, tmp)
        code, out = run_mag(["-c", "139_PPOSCAR_La3Ni2O7", "--element", "Ni",
                             "--qpoint", "0", "0", "0"], cwd=tmp)
        report("GM exit 0 (La3Ni2O7)", code == 0, out)
        x_file = os.path.join(tmp, "POSCAR_La3Ni2O7_spin_GM5+_dipole_x.vesta")
        y_file = os.path.join(tmp, "POSCAR_La3Ni2O7_spin_GM5+_dipole_y.vesta")
        report("2-dim GM5+ exports orthogonal x/y partners (no duplicated _2 file)",
               os.path.isfile(x_file) and os.path.isfile(y_file)
               and not os.path.isfile(os.path.join(tmp, "POSCAR_La3Ni2O7_spin_GM5+_dipole_x_2.vesta")),
               out)
        if os.path.isfile(x_file) and os.path.isfile(y_file):
            report("x and y partner files differ",
                   open(x_file).read() != open(y_file).read(), out)

    # hexagonal K/H points carry 1/3 coordinates: the labels must not fall
    # back to generic irrep_N names (the old 0.333333-rounding problem)
    poscar_hex = os.path.join(ROOT, "example", "36_spin_basis", "LuFeO3_P63cm", "185_PPOSCAR_LuFeO3")
    if not os.path.isfile(poscar_hex):
        report("LuFeO3 example POSCAR found", False, poscar_hex)
        return
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(poscar_hex, tmp)
        code, out = run_mag(["-c", "185_PPOSCAR_LuFeO3", "--element", "Fe"], cwd=tmp)
        report("hexagonal survey labels K and H points (1/3 handled exactly)",
               code == 0 and "K3(2)" in out and "H3(2)" in out and "irrep_" not in out, out)

        code, out = run_mag(["-c", "185_PPOSCAR_LuFeO3", "--element", "Fe",
                             "--qpoint", "1/3", "1/3", "0"], cwd=tmp)
        report("--qpoint accepts fractions (1/3 1/3 0 -> K)",
               code == 0 and "Selected q-point: K" in out and "K3(2)" in out, out)

        code, out = run_mag(["-c", "185_PPOSCAR_LuFeO3", "--element", "Fe",
                             "--qpoint", "0.333333", "0.333333", "0.5"], cwd=tmp)
        report("decimal 0.333333 snapped to 1/3 (-> H)",
               code == 0 and "Selected q-point: H" in out and "H3(2)" in out, out)

    # non-special q: spin-basis labels fall back to the ISO-IR (ISOTROPY) tables
    poscar_327 = os.path.join(ROOT, "example", "test_POSCARs", "139_PPOSCAR_La3Ni2O7")
    if not os.path.isfile(poscar_327):
        report("La3Ni2O7 test POSCAR found", False, poscar_327)
        return
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(poscar_327, tmp)
        code, out = run_mag(
            ["-c", "139_PPOSCAR_La3Ni2O7", "--element", "Ni",
             "--qpoint", "0.5", "0.5", "0.25"],
            cwd=tmp,
        )
        report("La3Ni2O7 low-symmetry q exit 0", code == 0, out)
        report("low-symmetry q named via ISO-IR",
               "Selected q-point: Y =" in out, out)
        report("spin bases labeled via ISO-IR (no generic irrep_N)",
               "Y1(1)" in out and "irrep_" not in out, out)


# ---------------------------------------------------------------- 37. crystod-mag extras
def test_37_mag_command() -> None:
    print("\n[37] crystod-mag (sectioned command: symmetry-adapted spin bases)")
    poscar = os.path.join(ROOT, "example", "test_POSCARs", "221_PPOSCAR_AlNi3")
    if not os.path.isfile(poscar):
        report("example POSCAR found", False, poscar)
        return

    def run_mag(args: list[str], cwd: str | None = None) -> tuple[int, str]:
        return run_module("crystod.cli.mag", args, cwd)

    with tempfile.TemporaryDirectory() as tmp:
        code, out = run_mag(
            ["-c", poscar, "--element", "Ni", "--qpoint", "0", "0", "0"], cwd=tmp
        )
        report("-c exit 0", code == 0, out)
        report("decomposition 2 x GM4+ + GM5+",
               "2 x GM4+(3)" in out and "GM5+(3)" in out, out)
        report("MAGMOM printed by default (no --show-spin-direction)",
               out.count("MAGMOM =") >= 12, out)
        report("VESTA files written",
               os.path.isfile(os.path.join(tmp, "POSCAR_AlNi3_spin_GM4+_octupole_111.vesta")))

        code, out = run_mag(
            ["--poscar", poscar, "--element", "Ni", "--qpoint", "GM",
             "--format", "vasp"],
            cwd=tmp,
        )
        report("--poscar alias + --format vasp accepted",
               code == 0 and "MAGMOM =" in out, out)

        code, out = run_mag(
            ["-c", poscar, "--element", "Ni", "--qpoint", "0", "0", "0",
             "--format", "qe"],
            cwd=tmp,
        )
        report("--format qe exit 0", code == 0, out)
        report("QE noncollinear block printed",
               "noncolin = .true." in out and "starting_magnetization(" in out
               and "angle1(" in out and "angle2(" in out, out)
        report("QE mode prints no MAGMOM line", "MAGMOM =" not in out, out)
        report("120-degree octupole angles (theta 114.09, phi -63.44)",
               "angle1(3) = 114.09" in out and "angle2(3) = -63.43" in out, out)
        report("magnetic element split into QE types by direction",
               "! type 2 (Ni1)" in out and "! type 4 (Ni3)" in out
               and "non-magnetic" in out, out)

        code, out = run_mag(["-c", poscar, "--element", "Ni"], cwd=tmp)
        report("survey mode (no --qpoint) exit 0", code == 0, out)

        ceo2 = os.path.join(ROOT, "example", "06_visualized_basis", "CeO2_Fm-3m", "225_PPOSCAR_CeO2")
        if os.path.isfile(ceo2):
            code, out = run_mag(
                ["-c", ceo2, "--element", "Ce", "--qpoint", "0", "0", "0", "--conventional"],
                cwd=tmp,
            )
            report("--conventional exit 0", code == 0, out)
            report("conventional display cell reported",
                   "conventional (F centring)" in out, out)
            conv_files = [name for name in os.listdir(tmp) if name.endswith("_conv.vesta")]
            report("spin VESTA files written with _conv suffix", len(conv_files) > 0)

        code, out = run_mag(
            ["-c", poscar, "--element", "Ni", "--qpoint", "0", "0", "0",
             "--format", "abinit"],
            cwd=tmp,
        )
        report("unknown --format rejected cleanly",
               code != 0 and "invalid choice" in out and "Traceback" not in out, out)

        code, out = run_cli(["--spin-basis"])
        report("removed --spin-basis flag points to crystod-mag",
               code != 0 and "is not a crystod option" in out and "crystod-mag" in out, out)


# ---------------------------------------------------------------- 38. crystod-md --adp
def test_38_xdatcar2adp() -> None:
    print("\n[38] crystod-md --adp (ScF3 NpT 300K, truncated trajectory)")
    source = os.path.join(XDATCAR_ADP_DIR, "XDATCAR")
    if not os.path.isfile(source):
        report("example data found", False, source)
        return
    with tempfile.TemporaryDirectory() as tmp:
        # 293 frames x 264 lines each (NpT trajectory with repeated headers)
        destination = os.path.join(tmp, "XDATCAR")
        with open(source) as fin, open(destination, "w") as fout:
            for line_number, line in enumerate(fin):
                if line_number >= 77352:
                    break
                fout.write(line)

        code, out = run_md(
            ["--adp", "--dim", "4", "4", "4", "--start-step", "100",
             "--output", "ADP_test.cif"],
            cwd=tmp,
        )
        report("exit code 0", code == 0, out)
        report("Pm-3m detected from time-averaged structure", "Pm-3m" in out, out)
        report("Sc ADP constrained isotropic", "U11=U22, U11=U33" in out, out)
        cif_path = os.path.join(tmp, "ADP_test.cif")
        report("ADP CIF written", os.path.isfile(cif_path))
        if os.path.isfile(cif_path):
            text = open(cif_path).read()
            report("CIF contains aniso U loop and both sites",
                   "_atom_site_aniso_U_11" in text and "Sc0" in text and "F1" in text,
                   text[:600])


# ---------------------------------------------------------------- 39. crystod-md extras
def test_39_md_command() -> None:
    print("\n[39] crystod-md (sectioned command: MD trajectory -> ADPs / summary)")

    def run_md(args: list[str], cwd: str | None = None) -> tuple[int, str]:
        return run_module("crystod.cli.md", args, cwd)

    # --dim normalization (all four accepted input forms + rejections)
    from crystod.cli.md import _parse_dim, build_parser

    md_parser = build_parser()
    ok = True
    for tokens in (["4", "4", "4"], ["4 4 4"],
                   ["4", "0", "0", "0", "4", "0", "0", "0", "4"],
                   ["4 0 0  0 4 0  0 0 4"]):
        ok = ok and _parse_dim(md_parser, tokens) == "4 4 4"
    report('--dim accepts "4 4 4" / 4 4 4 / nine-value diagonal (quoted or not)', ok)

    source = os.path.join(XDATCAR_ADP_DIR, "XDATCAR")
    if not os.path.isfile(source):
        report("example data found", False, source)
        return
    with tempfile.TemporaryDirectory() as tmp:
        # truncated trajectory (293 frames), as in section 38
        destination = os.path.join(tmp, "XDATCAR")
        with open(source) as fin, open(destination, "w") as fout:
            for line_number, line in enumerate(fin):
                if line_number >= 77352:
                    break
                fout.write(line)

        code, out = run_md(
            ["--adp", "--dim", "4", "0", "0", "0", "4", "0", "0", "0", "4",
             "--start-step", "100", "--output", "ADP_new.cif", "--format", "vasp"],
            cwd=tmp,
        )
        report("--adp with nine-value --dim exit 0", code == 0, out)
        report("diagonal extracted (Supercell size [4, 4, 4])",
               "Supercell size : [4, 4, 4]" in out, out)
        report("Pm-3m detected", "Pm-3m" in out, out)
        report("ADP CIF written", os.path.isfile(os.path.join(tmp, "ADP_new.cif")))

        code, out = run_md(
            ["--summary", "--start-step", "100", "--xdatcar", "XDATCAR",
             "--format", "vasp"],
            cwd=tmp,
        )
        report("--summary exit 0", code == 0, out)
        report("lattice statistics printed",
               all(key in out for key in ("a (A)", "gamma (deg)", "V (A^3)", "+/-")), out)
        report("analyzed step range reported", "analyzed steps : 193" in out, out)

        code, out = run_md(
            ["--summary", "--start-step", "100", "--end-step", "199"], cwd=tmp
        )
        report("--end-step honored (100 steps analyzed)",
               code == 0 and "analyzed steps : 100" in out, out)

        code, out = run_md(["--summary", "--dim", "4", "4", "4"], cwd=tmp)
        report("--summary with --dim rejected cleanly",
               code != 0 and "does not use --dim" in out and "Traceback" not in out, out)

        code, out = run_md(["--adp", "--start-step", "100"], cwd=tmp)
        report("--adp without --dim rejected cleanly",
               code != 0 and "requires --dim" in out and "Traceback" not in out, out)

        code, out = run_md(["--adp", "--dim", "4", "4", "4", "--end-step", "200"], cwd=tmp)
        report("--adp with --end-step rejected cleanly",
               code != 0 and "only available with --summary" in out and "Traceback" not in out, out)

        code, out = run_md(["--dim", "4", "4", "4"], cwd=tmp)
        report("missing mode flag rejected cleanly",
               code != 0 and "--adp" in out and "--summary" in out and "Traceback" not in out,
               out)

        code, out = run_md(
            ["--adp", "--dim", "4", "1", "0", "0", "4", "0", "0", "0", "4"], cwd=tmp
        )
        report("non-diagonal matrix rejected cleanly",
               code != 0 and "diagonal" in out and "Traceback" not in out, out)

        code, out = run_md(["--adp", "--dim", "4", "4"], cwd=tmp)
        report("wrong --dim length rejected cleanly",
               code != 0 and "three or nine" in out and "Traceback" not in out, out)

        code, out = run_md(["--adp", "--dim", "4", "4", "4", "--format", "lammps"], cwd=tmp)
        report("--format lammps rejected (not implemented yet)",
               code != 0 and "invalid choice" in out and "Traceback" not in out, out)

        code, out = run_md(
            ["--adp", "--dim", "4", "4", "4", "--xdatcar", "NO_SUCH_FILE"], cwd=tmp
        )
        report("missing trajectory rejected cleanly",
               code != 0 and "not found" in out and "Traceback" not in out, out)

        code, out = run_cli(["--xdatcar2adp"])
        report("removed --xdatcar2adp flag points to crystod-md",
               code != 0 and "is not a crystod option" in out and "crystod-md" in out, out)


# ---------------------------------------------------------------- 40. crystod-mol
def test_40_mol() -> None:
    print("\n[40] crystod-mol (molecular point groups and molecular SALCs)")

    def run_mol(args: list[str], cwd: str | None = None) -> tuple[int, str]:
        return run_module("crystod.cli.mol", args, cwd)

    xyz_o2 = os.path.join(XYZ_DIR, "XYZ_O2.xyz")
    xyz_h2o = os.path.join(XYZ_DIR, "XYZ_H2O.xyz")
    xyz_nh3 = os.path.join(XYZ_DIR, "XYZ_NH3.xyz")
    xyz_ch4 = os.path.join(XYZ_DIR, "XYZ_CH4.xyz")
    for path in (xyz_o2, xyz_h2o, xyz_nh3, xyz_ch4):
        if not os.path.isfile(path):
            report("example data found", False, path)
            return

    # --symmetry: point-group detection
    code, out = run_mol(["--symmetry", "--xyz", xyz_o2])
    report("--symmetry O2 exit 0", code == 0, out)
    report("O2 detected as linear D*h", "D*h" in out and "linear" in out, out)

    code, out = run_mol(["--symmetry", "--xyz", xyz_nh3])
    report("NH3 detected as C3v (3m)", code == 0 and "C3v" in out and "3m" in out, out)
    report("NH3 classes listed (E, 2C3, 3sgv)", "2C3" in out and "3sgv" in out, out)

    code, out = run_mol(["--symmetry", "--xyz", xyz_ch4])
    report("CH4 detected as Td (-43m)", code == 0 and "Td" in out and "-43m" in out, out)

    code, out = run_mol(["--symmetry", "--xyz", xyz_h2o])
    report("H2O detected as C2v (mm2)", code == 0 and "C2v" in out and "mm2" in out, out)

    # SALC mode: character analysis and explicit SALCs
    code, out = run_mol(["--xyz", xyz_nh3, "--element", "H", "--orbital", "s"])
    report("NH3 H s SALC exit 0", code == 0, out)
    report("NH3 H s characters (chi_perm = 3, 0, 1)",
           re.search(r"chi\(perm\):\s+3\s+0\s+1", out) is not None, out)
    report("NH3 H s decomposition A1 + E", "Gamma = 1(A1) + 1(E)" in out, out)
    report("NH3 H s A1 SALC is the in-phase sum",
           "A1: [s(H1) + s(H2) + s(H3)]" in out, out)

    code, out = run_mol(["--xyz", xyz_ch4, "--element", "H", "--orbital", "s"])
    report("CH4 H s decomposition A1 + T2",
           code == 0 and "Gamma = 1(A1) + 1(T2)" in out, out)
    report("CH4 H s characters (chi_perm(8C3) = 1)",
           re.search(r"chi\(perm\):\s+4\s+1\s+0\s+0\s+2", out) is not None, out)

    code, out = run_mol(["--xyz", xyz_h2o, "--element", "H", "--orbital", "s"])
    report("H2O H s decomposition A1 + B1",
           code == 0 and "Gamma = 1(A1) + 1(B1)" in out, out)

    # orbital characters multiply in (perm x p)
    code, out = run_mol(["--xyz", xyz_nh3, "--element", "H", "--orbital", "p"])
    report("NH3 H p decomposition 2A1 + A2 + 3E",
           code == 0 and "Gamma = 2(A1) + 1(A2) + 3(E)" in out, out)

    code, out = run_mol(["--xyz", xyz_nh3, "--element", "N", "--orbital", "p"])
    report("NH3 N p splits into A1 (pz) + E (px, py)",
           code == 0 and "Gamma = 1(A1) + 1(E)" in out
           and "A1: [pz(N1)]" in out and "E: [px(N1), py(N1)]" in out, out)


# ------------------------------------------------- 41. crystod-mol --diagram
def test_41_molod() -> None:
    print("\n[41] crystod-mol --diagram (MO diagram from symmetry + overlap)")

    def run_mol(args: list[str], cwd: str | None = None) -> tuple[int, str]:
        return run_module("crystod.cli.mol", args, cwd)

    molod_dir = os.path.join(ROOT, "example", "41_molod")
    xyz_nh3 = os.path.join(molod_dir, "XYZ_NH3.xyz")
    xyz_ch4 = os.path.join(molod_dir, "XYZ_CH4.xyz")
    xyz_sf6 = os.path.join(molod_dir, "XYZ_SF6.xyz")
    for path in (xyz_nh3, xyz_ch4, xyz_sf6):
        if not os.path.isfile(path):
            report("example data found", False, path)
            return

    with tempfile.TemporaryDirectory() as tmp:
        # NH3: fragments, SALCs, overlaps, textbook MO sequence, HTML default
        code, out = run_mol(["--diagram", "--xyz", xyz_nh3], cwd=tmp)
        report("--diagram NH3 exit 0", code == 0, out)
        report("CrystOD's own paper cited at the end of the run",
               "If you use CrystOD in your research, please cite:" in out
               and "Phys. Rev. B 110, 064104 (2024)" in out, out)
        report("central atom and ligands identified",
               "central atom: N; ligands: 3 H" in out, out)
        report("ligand SALCs printed per irrep",
               "A1: [1s(H1) + 1s(H2) + 1s(H3)]" in out, out)
        report("SALC | central AO overlap integrals printed",
               re.search(r"A1:\s+< a1 \(E =\s+-16\.44 eV\) \| N 2s >\s+S = 0\.7205", out)
               is not None, out)
        report("NH3 filling (2a1)^2 (1e)^4 (3a1)^2 with core numbering",
               "(2a1)^2 (1e)^4 (3a1)^2" in out and "N 1s -> a1" in out, out)
        report("NH3 HOMO is the 3a1 lone pair",
               "HOMO = 3a1" in out and "LUMO = 2e" in out, out)
        report("Wolfsberg-Helmholz / Hoffmann citations printed",
               "J. Chem. Phys. 20, 837 (1952)" in out
               and "J. Chem. Phys. 39, 1397 (1963)" in out, out)
        html_path = os.path.join(tmp, "MolOD_XYZ_NH3.html")
        report("HTML diagram written by default", os.path.isfile(html_path), out)
        if os.path.isfile(html_path):
            with open(html_path) as handle:
                html = handle.read()
            report("diagram page has the four columns and level details",
                   "SALCs" in html and "MOs" in html and "Level details" in html,
                   html_path)
            report("diagram marks HOMO/LUMO and electron arrows",
                   "HOMO" in html and "LUMO" in html and "↑" in html,
                   html_path)
            report("diagram has the adjustable energy window",
                   'id="emin"' in html and 'id="emax"' in html
                   and "Energy window" in html, html_path)
            report("diagram has the in-panel orbital sketch (hover viewer)",
                   "oview" in html and "drawSketch" in html
                   and '"orb":' in html, html_path)

        # CH4: photoelectron-convention labels (matches the textbook diagram)
        code, out = run_mol(["--diagram", "--xyz", xyz_ch4], cwd=tmp)
        report("CH4 filling (2a1)^2 (1t2)^6 (C 1s core counted)",
               code == 0 and "(2a1)^2 (1t2)^6" in out and "C 1s -> a1" in out, out)
        report("CH4 HOMO 1t2, antibonding 2t2/3a1 empty",
               "HOMO = 1t2" in out and "LUMO = 2t2" in out, out)

        # SF6: two ligand shells, --center/--output options
        code, out = run_mol(["--diagram", "--xyz", xyz_sf6, "--center", "S",
                             "--output", "sf6.html"], cwd=tmp)
        report("SF6 --center/--output exit 0",
               code == 0 and os.path.isfile(os.path.join(tmp, "sf6.html")), out)
        report("SF6 F 2p SALCs span a1g+eg+t1g+t2g+t1u+t2u",
               all(f"{name}:" in out for name in ("A1g", "Eg", "T1g", "T2g", "T1u", "T2u")),
               out)
        report("SF6 48-electron filling ends in the nonbonding F 2p block",
               "48 valence electrons" in out and "(1t1g)^6" in out, out)

        # --ao-left/--ao-right without --pyscf: two-fragment extended-Hueckel
        # diagram (arbitrary submolecule split, no central atom needed)
        xyz_c6h6 = os.path.join(molod_dir, "XYZ_C6H6.xyz")
        code, out = run_mol(["--diagram", "--xyz", xyz_c6h6,
                             "--ao-left", "H6", "--ao-right", "C6"], cwd=tmp)
        report("EHT fragment diagram (benzene H6 | C6) exit 0", code == 0, out)
        report("benzene EHT pi frontier labeled 1e1g / 1e2u",
               "HOMO = 1e1g" in out and "LUMO = 1e2u" in out, out)
        report("core-counted numbering starts the sigma stack at 2a1g",
               "(2a1g)^2" in out, out)
        report("fragment EHT HTML written with the plain default name",
               os.path.isfile(os.path.join(tmp, "MolOD_XYZ_C6H6.html")), out)
        # a diatomic fragment keeps its own higher symmetry: the CO pi pair
        # of CH3OH is exactly degenerate under Cs (chi(E) = 2 matches no Cs
        # irrep) and must be split by the irrep projectors, not by energy
        xyz_ch3oh = os.path.join(molod_dir, "XYZ_CH3OH.xyz")
        code, out = run_mol(["--diagram", "--xyz", xyz_ch3oh,
                             "--ao-left", "H4", "--ao-right", "CO"], cwd=tmp)
        report("EHT fragment diagram (CH3OH H4 | CO) exit 0", code == 0, out)
        report("exactly degenerate CO pi pair split into a'/a'' labels",
               "CO 1a''" in out and "CO 2a''" in out, out)
        code, out = run_mol(["--diagram", "--xyz", xyz_ch3oh,
                             "--ao-left", "H3", "--ao-right", "CO"], cwd=tmp)
        report("non-partitioning EHT --ao-left/--ao-right rejected cleanly",
               code != 0 and "does not partition" in out
               and "Traceback" not in out, out)

        # --pyscf: quantitative diagrams (three SCF runs in one AO space)
        try:
            import pyscf  # noqa: F401
            has_pyscf = True
        except ImportError:
            has_pyscf = False
        if not has_pyscf:
            print("  [SKIP] pyscf not installed: --pyscf tests skipped.")
            return

        xyz_h2o = os.path.join(molod_dir, "XYZ_H2O.xyz")
        xyz_o2 = os.path.join(molod_dir, "XYZ_O2.xyz")
        code, out = run_mol(["--diagram", "--xyz", xyz_h2o, "--pyscf",
                             "--basis", "sto-3g"], cwd=tmp)
        report("--pyscf H2O exit 0", code == 0, out)
        report("three SCF calculations reported (H2, O, H2O; all converged)",
               "H2 (RHF)" in out and "O (RHF)" in out and "H2O (RHF)" in out
               and "NOT CONVERGED" not in out, out)
        report("counterpoise-consistent interaction energy printed",
               "interaction energy" in out and "full molecular basis" in out, out)
        report("H2O HOMO is 1b2 with crystod irrep labels",
               "HOMO = 1b2" in out, out)
        pyscf_html_path = os.path.join(tmp, "MolOD_XYZ_H2O_pyscf.html")
        report("pyscf HTML written with its own default name",
               os.path.isfile(pyscf_html_path), out)
        if os.path.isfile(pyscf_html_path):
            with open(pyscf_html_path) as handle:
                pyscf_html = handle.read()
            report("pyscf diagram also carries the orbital sketches",
                   '"orb":' in pyscf_html and "oview" in pyscf_html,
                   pyscf_html_path)
            report("core levels below -40 eV clamp the default window to -40",
                   '"eMin": -40.0' in pyscf_html, pyscf_html_path)
            report("Show-all-energy-levels button present",
                   'id="eshowall"' in pyscf_html
                   and "Show all energy levels" in pyscf_html, pyscf_html_path)

        # bonding/antibonding phase in the sketches (NH3 1e vs 2e): the
        # radial-weighted compression must keep the true wave-function
        # signs -- a bare contracted-coefficient sum inverts the N-2p lobe
        # of the bonding 1e and both sketches come out identical.  The
        # check is the bond-directed product sum_H s_H (p_N . r_NH), which
        # is positive for bonding and negative for antibonding in EVERY
        # real gauge of the degenerate pair (a max-|s| pick is not: the
        # two mirror hydrogens tie with opposite signs in the px-type
        # partner, and which partner comes first is a canonicalization
        # gauge that shifts with the BLAS environment)
        nh3_code, nh3_out = run_mol(["--diagram", "--xyz", xyz_nh3,
                                     "--pyscf"], cwd=tmp)
        report("--pyscf NH3 exit 0", nh3_code == 0, nh3_out)
        nh3_html_path = os.path.join(tmp, "MolOD_XYZ_NH3_pyscf.html")
        if os.path.isfile(nh3_html_path):
            with open(nh3_html_path) as handle:
                nh3_html = handle.read()
            match = re.search(r"LEVELS = (\[\{.*?\}\]);", nh3_html, re.S)
            geom_match = re.search(r"GEOM = (\{.*?\});", nh3_html, re.S)
            levels = json.loads(match.group(1)) if match else []
            geom_atoms = (json.loads(geom_match.group(1))["atoms"]
                          if geom_match else [])
            phases = {}
            for level in levels:
                if level.get("col") == "mo" and level["label"] in ("1e", "2e"):
                    entries = {row[0]: row for row in level["orb"][0]}
                    n_pos = geom_atoms[0][1:4]
                    n_p = entries[0][2:5]
                    total = 0.0
                    for atom, row in entries.items():
                        if atom == 0:
                            continue
                        direction = [geom_atoms[atom][k + 1] - n_pos[k]
                                     for k in range(3)]
                        total += row[1] * sum(
                            p * d for p, d in zip(n_p, direction))
                    phases[level["label"]] = total
            report("NH3 1e bonding / 2e antibonding phases in the sketch",
                   len(phases) == 2 and phases["1e"] > 0 > phases["2e"],
                   str(phases))
        else:
            report("NH3 1e bonding / 2e antibonding phases in the sketch",
                   False, nh3_out)
        report("friendly method/basis wording",
               "Hartree-Fock method / sto-3g basis" in out, out)
        # the three PySCF papers are SOFTWARE citations, so they must be
        # printed under their own "please cite" heading and never read as
        # the source of the fragment/irrep method
        report("PySCF cited as software, separate from the method line",
               "Method: fragment-resolved SCF MO diagram\n" in out
               and "If you use PySCF in your research, please cite:" in out
               and "J. Chem. Phys. 153, 024109 (2020)" in out
               and "WIREs Comput. Mol. Sci. 8, e1340 (2018)" in out
               and "J. Comput. Chem. 36, 1664 (2015)" in out
               and "MO diagram (PySCF)." not in out, out)
        report("CrystOD's own paper cited at the end of the run",
               "If you use CrystOD in your research, please cite:" in out
               and "Phys. Rev. B 110, 064104 (2024)" in out, out)

        # O2: triplet, homonuclear partition, sigma/pi labels
        code, out = run_mol(["--diagram", "--xyz", xyz_o2, "--pyscf",
                             "--basis", "sto-3g", "--spin", "2",
                             "--ao-left", "O", "--ao-right", "O"], cwd=tmp)
        report("--pyscf O2 --ao-left O --ao-right O exit 0", code == 0, out)
        report("O2 triplet filling (1πu)^4 (1πg)^2 with sigma/pi labels",
               "(1πu)^4 (1πg)^2" in out and "HOMO = 1πg" in out, out)
        report("identical fragments disambiguated as O(L)/O(R)",
               "O(L)" in out and "O(R)" in out, out)

        # fragment partition validation
        code, out = run_mol(["--diagram", "--xyz", xyz_ch4, "--pyscf",
                             "--basis", "sto-3g",
                             "--ao-left", "H3", "--ao-right", "CO"], cwd=tmp)
        report("non-partitioning --ao-left/--ao-right rejected cleanly",
               code != 0 and "does not partition" in out and "Traceback" not in out,
               out)


# ---------------------------------------------------------------- 42. crystod-mol extras
def test_42_mol_command() -> None:
    print("\n[42] crystod-mol (extras: --align / --show-matrix / --visualize / errors)")

    def run_mol(args: list[str], cwd: str | None = None) -> tuple[int, str]:
        return run_module("crystod.cli.mol", args, cwd)

    xyz_o2 = os.path.join(XYZ_DIR, "XYZ_O2.xyz")
    xyz_nh3 = os.path.join(XYZ_DIR, "XYZ_NH3.xyz")
    xyz_ch4 = os.path.join(XYZ_DIR, "XYZ_CH4.xyz")

    # --align: textbook axis convention (CH4 in this file is arbitrarily rotated)
    code, out = run_mol(["--xyz", xyz_ch4, "--element", "C", "--orbital", "d", "--align"])
    report("--align exit 0", code == 0, out)
    report("CH4 C d crystal-field splitting E (dz2, dx2-y2) + T2 (dxy, dyz, dxz)",
           "E: [dz2(C1), dx2-y2(C1)]" in out and "T2: [dxy(C1), dyz(C1), dxz(C1)]" in out,
           out)
    report("--align frame is announced", "standard point-group axes" in out, out)

    # --show-matrix: permutation matrices are printed per class
    code, out = run_mol(["--xyz", xyz_nh3, "--element", "H", "--orbital", "s",
                         "--show-matrix"])
    report("--show-matrix prints the site-permutation matrices",
           code == 0 and "Site-permutation matrices" in out, out)

    # --tolerance forwarded (loose tolerance still detects C3v)
    code, out = run_mol(["--symmetry", "--xyz", xyz_nh3, "--tolerance", "0.1"])
    report("--tolerance accepted", code == 0 and "C3v" in out, out)

    # --visualize: standalone HTML viewer (same page as crystod --visualize)
    with tempfile.TemporaryDirectory() as tmp:
        code, out = run_mol(["--xyz", xyz_nh3, "--element", "H", "--orbital", "p",
                             "--visualize", "--bond", "N", "H", "1.2"], cwd=tmp)
        html_path = os.path.join(tmp, "SALC_XYZ_NH3_H_p.html")
        report("--visualize exit 0 and default file name", code == 0 and os.path.isfile(html_path), out)
        if os.path.isfile(html_path):
            with open(html_path) as handle:
                html = handle.read()
            report("viewer shows the point group and decomposition",
                   "C3v (3m)" in html and "2(A1) + 1(A2) + 3(E)" in html, html_path)
            report("viewer has one mode row per SALC (9 for H p)",
                   html.count("mode-row") >= 9, html_path)
            report("viewer draws N-H bonds and xyz compass",
                   "N-H bonds" in html and "show xyz axes" in html, html_path)
            report("molecule viewer hides the vacuum-box cell edges",
                   "show cell edges" not in html, html_path)

        code, out = run_mol(["--xyz", xyz_ch4, "--element", "C", "--orbital", "d",
                             "--align", "--visualize", "--output", "d_salc.html"], cwd=tmp)
        report("--visualize --output custom name",
               code == 0 and os.path.isfile(os.path.join(tmp, "d_salc.html")), out)

    code, out = run_mol(["--xyz", xyz_nh3, "--element", "H", "--orbital", "s",
                         "--output", "x.html"])
    report("--output without --visualize rejected cleanly",
           code != 0 and "only available with --visualize" in out and "Traceback" not in out,
           out)

    code, out = run_mol(["--symmetry", "--xyz", xyz_nh3, "--visualize"])
    report("--visualize with --symmetry rejected cleanly",
           code != 0 and "only available in SALC mode" in out and "Traceback" not in out, out)

    # errors
    code, out = run_mol(["--xyz", xyz_nh3])
    report("missing --element/--orbital rejected cleanly",
           code != 0 and "either use --symmetry" in out and "Traceback" not in out, out)

    code, out = run_mol(["--symmetry", "--xyz", xyz_nh3, "--element", "H"])
    report("--symmetry with --element rejected cleanly",
           code != 0 and "cannot be combined" in out and "Traceback" not in out, out)

    code, out = run_mol(["--xyz", xyz_o2, "--element", "O", "--orbital", "s"])
    report("linear-molecule SALC rejected with guidance",
           code != 0 and "crystallographic point" in out and "Traceback" not in out, out)

    code, out = run_mol(["--xyz", xyz_nh3, "--element", "Fe", "--orbital", "s"])
    report("unknown element rejected cleanly",
           code != 0 and "not in the molecule" in out and "Traceback" not in out, out)

    code, out = run_mol(["--xyz", xyz_nh3, "--element", "H", "--orbital", "q"])
    report("unknown orbital rejected cleanly",
           code != 0 and "not supported" in out and "Traceback" not in out, out)

    code, out = run_mol(["--xyz", os.path.join(XYZ_DIR, "missing.xyz"),
                         "--symmetry"])
    report("missing file rejected cleanly",
           code != 0 and "not found" in out and "Traceback" not in out, out)

    # --diagram argument validation
    code, out = run_mol(["--diagram", "--symmetry", "--xyz", xyz_nh3])
    report("--diagram with --symmetry rejected cleanly",
           code != 0 and "cannot be combined" in out and "Traceback" not in out, out)

    code, out = run_mol(["--diagram", "--xyz", xyz_nh3, "--element", "H"])
    report("--diagram with --element rejected cleanly",
           code != 0 and "cannot be combined" in out and "Traceback" not in out, out)

    code, out = run_mol(["--xyz", xyz_nh3, "--element", "H", "--orbital", "s",
                         "--center", "N"])
    report("--center without --diagram rejected cleanly",
           code != 0 and "only available with --diagram" in out
           and "Traceback" not in out, out)

    code, out = run_mol(["--diagram", "--xyz", xyz_nh3, "--visualize"])
    report("--diagram with --visualize rejected cleanly",
           code != 0 and "not available with --diagram" in out
           and "Traceback" not in out, out)

    code, out = run_mol(["--diagram", "--xyz", xyz_nh3, "--center", "Fe"])
    report("--diagram with absent central element rejected cleanly",
           code != 0 and "exactly one" in out and "Traceback" not in out, out)

    code, out = run_mol(["--diagram", "--xyz", xyz_o2])
    report("--diagram on a linear molecule rejected with guidance",
           code != 0 and "crystallographic point" in out and "Traceback" not in out,
           out)

    code, out = run_mol(["--xyz", xyz_nh3, "--element", "H", "--orbital", "s",
                         "--pyscf"])
    report("--pyscf without --diagram rejected cleanly",
           code != 0 and "only available with --diagram" in out
           and "Traceback" not in out, out)

    code, out = run_mol(["--diagram", "--xyz", xyz_nh3, "--ao-left", "H3"])
    report("--ao-left without --ao-right rejected cleanly",
           code != 0 and "give both --ao-left and --ao-right" in out
           and "Traceback" not in out, out)


    # --example: the bundled methane XYZ (spec 3.2.2); --xyz is required
    # otherwise, so a bare --example has to be answered before that check
    code, out = run_mol(["--example"])
    report("--example alone lists the bundled examples (no --xyz needed)",
           code == 0 and "CH4" in out and "NH3" in out and "required" not in out, out)
    with tempfile.TemporaryDirectory() as tmp:
        code, out = run_mol(["--example", "CH4"], cwd=tmp)
        report("--example CH4 writes XYZ_CH4.xyz and MolOD_XYZ_CH4.html",
               code == 0 and "Running: crystod-mol --diagram --xyz XYZ_CH4.xyz" in out
               and os.path.isfile(os.path.join(tmp, "XYZ_CH4.xyz"))
               and os.path.isfile(os.path.join(tmp, "MolOD_XYZ_CH4.html")), out)

# ------------------------------------------- 17. crystod-group --parent
def test_17_isotropy() -> None:
    print("\n[17] crystod-group --parent (isotropy subgroups)")

    import spglib

    if tuple(int(x) for x in spglib.__version__.split(".")[:2]) < (2, 4):
        print(f"  [SKIP] spglib {spglib.__version__} < 2.4: subgroup identification "
              "is unreliable in old spglib; run this section in the crystod env.")
        return

    # single order-parameter directions (the ISOSUBGROUP reference cases);
    # this first case deliberately uses the --supergroup alias of --parent
    code, out = run_group(["--supergroup", "Pm-3m", "--irrep", "GM4-",
                           "--order-parameter", "0", "0", "a"])
    report("--supergroup alias still accepted: GM4- (0,0,a) -> P4mm",
           code == 0 and "P4mm (No. 99)" in out, out)
    report("index 6 and conventional basis printed",
           "index 6" in out and "conventional basis" in out, out)

    code, out = run_group(["--parent", "Pm-3m", "--irrep", "GM4-",
                           "--order-parameter", "a", "a", "0"])
    report("GM4- (a,a,0) -> Amm2", code == 0 and "Amm2 (No. 38)" in out, out)

    code, out = run_group(["--parent", "Pm-3m", "--irrep", "GM4-",
                           "--order-parameter", "a", "a", "a"])
    report("GM4- (a,a,a) -> R3m", code == 0 and "R3m (No. 160)" in out, out)

    # full enumeration (validated against the ISOSUBGROUP table for Pm-3m GM)
    code, out = run_group(["--parent", "Pm-3m", "--irrep", "GM4-"])
    report("GM4- enumeration exit 0", code == 0, out)
    report("GM4- enumerates P4mm/R3m/Amm2/Pm/Cm/P1",
           all(name in out for name in
               ("99 P4mm", "160 R3m", "38 Amm2", "6 Pm", "8 Cm", "1 P1")), out)

    code, out = run_group(["--parent", "Pm-3m", "--irrep", "GM3+"])
    report("GM3+ -> P4/mmm + Pmmm (indices 3, 6)",
           code == 0 and "123 P4/mmm" in out and "47 Pmmm" in out, out)

    code, out = run_group(["--parent", "Pm-3m", "--irrep", "GM1+"])
    report("GM1+ (identity irrep) keeps the supergroup",
           code == 0 and "221 Pm-3m" in out, out)

    # zone-boundary irreps: cell enlargement (perovskite octahedral tilts)
    code, out = run_group(["--parent", "Pm-3m", "--irrep", "R4+"])
    report("R4+ tilt subgroups I4/mcm + R-3c + Imma (Howard-Stokes)",
           code == 0 and "140 I4/mcm" in out and "167 R-3c" in out
           and "74 Imma" in out, out)
    report("R4+ doubles the cell (size 2)",
           re.search(r"140 I4/mcm\s+2\s+6", out) is not None, out)

    code, out = run_group(["--parent", "Pm-3m", "--irrep", "M3+"])
    report("M3+ tilt subgroups P4/mbm + Im-3 + I4/mmm",
           code == 0 and "127 P4/mbm" in out and "204 Im-3" in out
           and "139 I4/mmm" in out, out)
    report("direction column carries the irrep label (arms ; separated)",
           re.search(r"M3\+\(0;0;a\)\s+127 P4/mbm", out) is not None, out)

    code, out = run_group(["--parent", "Pm-3m", "--irrep", "M3+",
                           "--order-parameter", "a", "a", "a"])
    report("M3+ (a,a,a) -> Im-3 with 2x2x2 cell (size 4)",
           code == 0 and "Im-3 (No. 204)" in out and "cell size 4" in out, out)

    # negative components as the tables print them (M3+(a;-a;0)): argparse
    # took -a for an option; every spelling must give the same subgroup
    code, out = run_group(["--parent", "221", "--irrep", "M3+",
                           "--order-parameter", "a", "-a", "0"])
    report("--order-parameter a -a 0 -> I4/mmm (No. 139)",
           code == 0 and "M3+(a;-a;0) -> I4/mmm (No. 139)" in out
           and "cell size 4, index 12" in out, out)
    spellings = {
        "one string": ["a -a 0"], "semicolons": ["a;-a;0"],
        "commas": ["a,-a,0"], "as printed": ["(a;-a;0)"],
        "leading space": ["a", " -a", "0"], "negative first": ["-b", "b", "0"],
    }
    outcomes = {}
    for name, tokens in spellings.items():
        code, text = run_group(["--parent", "221", "--irrep", "M3+",
                                "--order-parameter", *tokens])
        outcomes[name] = code == 0 and "-> I4/mmm (No. 139)" in text
    report("quoted, ';', ',', '(...)' and leading-space spellings agree",
           all(outcomes.values()), str(outcomes))
    code, out = run_group(["--parent", "221", "--order-parameter", "a", "-a", "0",
                           "--irrep", "M3+"])
    report("an option after the negative components is still an option",
           code == 0 and "M3+(a;-a;0) -> I4/mmm (No. 139)" in out, out)
    code, out = run_module("crystod.isotropy_subgroup",
                           ["--parent", "221", "--irrep", "M3+",
                            "--order-parameter", "a", "-a", "0"])
    report("the isotropy module's own parser takes a -a 0 as well",
           code == 0 and "M3+(a;-a;0) -> I4/mmm (No. 139)" in out, out)
    code, out = run_group(["--parent", "221", "--irrep", "M3+",
                           "--order-parameter", "a", "-0.5a", "0"])
    code2, out2 = run_group(["--parent", "221", "--irrep", "M3+",
                             "--order-parameter", "a", "-1/2a", "0"])
    report("exact factors: a -0.5a 0 = a -1/2a 0 -> Immm (the (a;b;0) stratum)",
           code == 0 and "M3+(a;-0.5a;0) -> Immm (No. 71)" in out
           and code2 == 0 and "M3+(a;-1/2a;0) -> Immm (No. 71)" in out2,
           out + out2)
    code, out = run_group(["--parent", "221", "--irrep", "M3+",
                           "--order-parameter", "a", "0.282a", "0"])
    report("a rounded factor (0.282a) is refused, not read as a new parameter",
           code != 0 and "not an exact fraction" in out
           and "Traceback" not in out, out)

    # complex-type irrep at a non-symmorphic zone-boundary point: the
    # physically irreducible doubled form, paired with the conjugate irrep
    # (validated entry by entry against ISOSUBGROUP SG230 P tables)
    code, out = run_group(["--parent", "Ia-3d", "--irrep", "P2"])
    report("complex P2 -> paired P1P2 doubled form (dim 8)",
           code == 0 and "P1P2: order parameter dimension 8" in out
           and "complex-type irrep -> physically irreducible real form" in out,
           out)
    report("P1P2 subgroups match ISOSUBGROUP (I-4, I222, 2x C2, ...)",
           all(re.search(p, out) is not None for p in
               (r"82 I-4\s+4\s+48", r"23 I222\s+4\s+48",
                r"24 I2_12_12_1\s+4\s+48", r"2 P-1\s+4\s+96",
                r"1 P1\s+4\s+192"))
           and len(re.findall(r"5 C2\s+4\s+96", out)) == 2, out)

    # real-type irrep with complex matrices (translation phases e^(i pi/2)):
    # exercises the antilinear-real-structure realification
    code, out = run_group(["--parent", "Ia-3d", "--irrep", "P3"])
    report("P3 realified (R32/R-3/R3 subgroups as in ISOSUBGROUP)",
           code == 0 and re.search(r"155 R32\s+4\s+32", out) is not None
           and re.search(r"148 R-3\s+4\s+32", out) is not None
           and re.search(r"146 R3\s+4\s+64", out) is not None, out)

    # complex pair with a three-dimensional small irrep at a point with
    # 2k = 0 (R of P-43n): the real-form attempt on the small irrep must
    # turn the pair away instead of normalizing a round-off intertwiner
    # (was LinAlgError: Singular matrix; validated against the ISOSUBGROUP
    # SG218 R table)
    code, out = run_group(["--parent", "218", "--irrep", "R4"])
    report("P-43n R4 -> paired R4R5 doubled form (dim 6), no traceback",
           code == 0 and "Traceback" not in out
           and "R4R5: order parameter dimension 6" in out
           and "complex-type irrep -> physically irreducible real form" in out,
           out)
    r4_rows = [line.split() for line in out.splitlines()
               if line.startswith("R4R5(")]
    report("R4R5 subgroups match ISOSUBGROUP (F222, I-4, R3, C2, P1)",
           all(re.search(p, out) is not None for p in
               (r"22 F222\s+2\s+12", r"82 I-4\s+2\s+12", r"146 R3\s+2\s+16",
                r"5 C2\s+2\s+24", r"1 P1\s+2\s+48"))
           and len(r4_rows) == 5, out)

    code, out = run_group(["--parent", "P-43n", "--irrep", "R5"])
    report("the partner label R5 gives the same R4R5 table",
           code == 0 and len(r4_rows) == 5
           and [line.split() for line in out.splitlines()
                if line.startswith("R4R5(")] == r4_rows, out)

    code, out = run_group(["--parent", "218", "--kpoint", "R"])
    r_rows = [line.split() for line in out.splitlines()
              if re.match(r"R\d\S*\(", line)]
    report("--kpoint R of P-43n: R1R2, R3 and R4R5 rows, no failed irrep",
           code == 0 and "not enumerated" not in out
           and r_rows[:2] == [["R1R2(a,b)", "196", "F23", "2", "4"],
                              ["R3(a,b,c,d)", "22", "F222", "2", "12"]]
           and len(r4_rows) == 5 and r_rows[2:] == r4_rows, out)

    # the crash depended on the rounding of one fixed random intertwiner
    # seed, and for most seeds the old code still answered None (through
    # the last check, on a garbage basis), so 40 seeds are checked for the
    # answer and for where it is taken: None before any basis is built
    # for the complex pairs R1/R4/R5 and the pseudoreal R3, a real form
    # for the real-type M5 (complex matrices as delivered by spgrep)
    probe = (
        "import numpy as np\n"
        "import crystod.isotropy_subgroup as iso\n"
        "original = iso._realify_matrix_set\n"
        "captured = {}\n"
        "def spy(matrices):\n"
        "    captured[label] = {k: np.array(v) for k, v in matrices.items()}\n"
        "    try:\n"
        "        return original(matrices)\n"
        "    except np.linalg.LinAlgError:\n"
        "        return None\n"
        "iso._realify_matrix_set = spy\n"
        "labels = ('R1', 'R3', 'R4', 'R5', 'M5')\n"
        "for label in labels:\n"
        "    iso.InducedRepresentation(iso.SpaceGroupIrrepAlgebra('218'), label)\n"
        "default_rng, column_stack = np.random.default_rng, np.column_stack\n"
        "def counting_stack(vectors):\n"
        "    built.append(seed)\n"
        "    return column_stack(vectors)\n"
        "np.column_stack = counting_stack\n"
        "for label in labels:\n"
        "    matrices = captured[label]\n"
        "    outcomes, built = set(), []\n"
        "    for seed in range(40):\n"
        "        np.random.default_rng = lambda _=None, s=seed: default_rng(s)\n"
        "        try:\n"
        "            result = original(matrices)\n"
        "            outcomes.add('none' if result is None else 'real')\n"
        "        except Exception as exc:\n"
        "            outcomes.add(type(exc).__name__)\n"
        "    np.random.default_rng = default_rng\n"
        "    print('TYPE', label, matrices[next(iter(matrices))].shape[0],\n"
        "          ' '.join(sorted(outcomes)), 'bases', len(built))\n"
    )
    code, out = run_python(probe)
    type_lines = out.splitlines()
    report("small-irrep real form: complex/pseudoreal -> None for every seed",
           code == 0 and all(line in type_lines for line in
                             ("TYPE R1 1 none bases 0",
                              "TYPE R3 2 none bases 0",
                              "TYPE R4 3 none bases 0",
                              "TYPE R5 3 none bases 0")), out)
    report("small-irrep real form: real-type M5 realified for every seed",
           "TYPE M5 2 real bases 40" in type_lines, out)

    # +k/-k pairing (the -k star is not in the star of k): P/PA of I-42d,
    # tabulated in the conjugate gauge
    code, out = run_group(["--parent", "122", "--irrep", "P1"])
    report("I-42d P1 -> P1PA1 pair (I-4 subgroup as in ISOSUBGROUP)",
           code == 0 and "P1PA1: order parameter dimension 4" in out
           and re.search(r"82 I-4\s+4\s+8", out) is not None
           and re.search(r"1 P1\s+4\s+32", out) is not None, out)

    # wrapped-translation gauge at N of I4_132: conj+wrap selection of the
    # spgrep candidate (chiral subgroups P4_122 vs P4_322 distinguish N1/N3)
    code, out = run_group(["--parent", "214", "--irrep", "N1"])
    report("I4_132 N1 (wrapped-gauge N star) -> C222 + P4_122 + R32",
           code == 0 and re.search(r"21 C222\s+2\s+12", out) is not None
           and re.search(r"91 P4_122\s+4\s+12", out) is not None
           and re.search(r"155 R32\s+4\s+16", out) is not None, out)
    report("enantiomorphic-partner note printed (91 <-> 95)",
           "91 <-> 95 are enantiomorphic partner types" in out, out)

    # the labels are those of the ISOTROPY software: N1 of Ia-3d carries
    # the subgroups of SUBGROUP/SUBGROUP_SG230_N.txt (before v0.4.3 they
    # were printed under N2, with a note giving the correspondence)
    code, out = run_group(["--parent", "Ia-3d", "--irrep", "N1"])
    report("Ia-3d N1 -> C222_1, P4_12_12, P-42_1c, R3c as in ISOSUBGROUP",
           code == 0 and re.search(r"20 C222_1\s+2\s+24", out) is not None
           and re.search(r"92 P4_12_12\s+4\s+24", out) is not None
           and re.search(r"114 P-42_1c\s+4\s+24", out) is not None
           and re.search(r"161 R3c\s+4\s+32", out) is not None
           and "ISOTROPY N" not in out, out)

    # mixed translation denominators (H of P3: k = (1/3,1/3,1/2)): the
    # translation grid must be the lcm (6), not the max (3)
    code, out = run_group(["--parent", "P3", "--irrep", "H1"])
    report("P3 H1 -> H1HA1 with 6x cell (lcm translation grid)",
           code == 0 and "H1HA1" in out
           and re.search(r"143 P3\s+6\s+6", out) is not None, out)

    # coupled irreps (I4/mmm X3- + X2+: n=2 Ruddlesden-Popper rotation+tilt;
    # hybrid-improper-ferroelectric ground state Cmc2_1 = A2_1am)
    code, out = run_group(["--parent", "I4/mmm", "--irrep", "X3-", "X2+"])
    report("coupled X3-+X2+ enumeration exit 0", code == 0, out)
    report("coupled header lists both irreps (dim 2+2)",
           "* Coupled irreps *" in out
           and "coupled order parameter dimension 4 (2 + 2)" in out, out)
    report("single-irrep tables printed before the coupled table",
           "(X3- alone) *" in out and "(X2+ alone) *" in out
           and "(coupled) *" in out
           and re.search(r"X3-\(0;a\)\s+63 Cmcm", out) is not None
           and re.search(r"X2\+\(0;c\)\s+64 Cmce", out) is not None
           and re.search(r"X2\+\(c;d\)\s+55 Pbam", out) is not None, out)
    report("zero-chunk directions omitted from the coupled table",
           "(0,0)" not in out, out)
    report("same-arm coupling X3-(0,a) X2+(0,c) -> Cmc2_1 (A2_1am)",
           re.search(r"X3-\(0;a\) X2\+\(0;c\)\s+36 Cmc2_1", out) is not None, out)
    report("cross-arm coupling X3-(0,a) X2+(c,0) -> Pnma",
           re.search(r"X3-\(0;a\) X2\+\(c;0\)\s+62 Pnma", out) is not None, out)
    report("generic coupled direction -> Pm (index 32)",
           re.search(r"X3-\(a;b\) X2\+\(c;d\)\s+6 Pm\s+4\s+32", out) is not None,
           out)

    code, out = run_group(["--parent", "I4/mmm", "--irrep", "X3-", "X2+",
                           "--order-parameter", "0", "a", "0", "c"])
    report("coupled explicit direction resolves to Cmc2_1",
           code == 0 and "X3-(0;a) X2+(0;c) -> Cmc2_1 (No. 36)" in out
           and "cell size 2, index 8" in out, out)

    code, out = run_group(["--parent", "I4/mmm", "--irrep", "X3-", "X2+",
                           "--order-parameter", "0", "a"])
    report("coupled order-parameter length checked against total dim",
           code != 0 and "needs 4 components" in out
           and "X3- + X2+" in out and "Traceback" not in out, out)

    # the representative vector of a stratum is generic as a whole AND in
    # every irrep part: the historical magnitudes give 0.2929 (1,1,1) for
    # the L1- part of L1+(a;a;b) L1-(d;d;e) of P-31m (0.4142/sqrt2 =
    # 0.2929), a part with more symmetry than its subspace; where they are
    # generic they are kept, so no other vector moves
    probe = (
        "import numpy as np\n"
        "from crystod.isotropy_subgroup import (IsotropyAnalyzer, _orth_basis,\n"
        "                                       _LEGACY_MAGNITUDES)\n"
        "from crystod.molecular_salc import _rref_orthogonal\n"
        "an = IsotropyAnalyzer('162', ['L1+', 'L1-'])\n"
        "mats = [m for _, _, m in an.elements]\n"
        "def stab(v, lo, hi):\n"
        "    return {k for k, M in enumerate(mats)\n"
        "            if np.allclose(M[lo:hi, lo:hi] @ v[lo:hi], v[lo:hi], atol=1e-6)}\n"
        "def pointwise(P, lo, hi):\n"
        "    U, s, _ = np.linalg.svd(_orth_basis(P)[lo:hi], full_matrices=False)\n"
        "    Q = U[:, s > 1e-8]\n"
        "    Pj = Q @ Q.T\n"
        "    return {k for k, M in enumerate(mats)\n"
        "            if np.allclose(M[lo:hi, lo:hi] @ Pj, Pj, atol=1e-6)}\n"
        "ok, count, part = True, 0, None\n"
        "for P, _ in an.enumerate_directions():\n"
        "    label, v = an.direction_label(P)\n"
        "    for lo, hi in ((0, 6), (0, 3), (3, 6)):\n"
        "        ok = ok and stab(v, lo, hi) == pointwise(P, lo, hi)\n"
        "    count += 1\n"
        "    if label == 'L1+(a;a;b) L1-(d;d;e)':\n"
        "        part = np.round(v[3:], 6)\n"
        "print('GENERIC', count, ok)\n"
        "print('PART', part[0] == part[1], part[1] != part[2])\n"
        "same = True\n"
        "an = IsotropyAnalyzer('Pm-3m', 'R4+')\n"
        "for P, _ in an.enumerate_directions():\n"
        "    label, v = an.direction_label(P)\n"
        "    rows = _rref_orthogonal(list(_orth_basis(P).T))\n"
        "    legacy = np.asarray(_LEGACY_MAGNITUDES[:len(rows)]) @ np.array(rows)\n"
        "    same = same and np.array_equal(v, legacy)\n"
        "print('LEGACY', same)\n"
    )
    code, out = run_python(probe)
    report("P-31m L1+ L1-: every stratum vector generic as a whole and per irrep",
           code == 0 and re.search(r"GENERIC \d+ True", out) is not None, out)
    report("the L1- part of L1+(a;a;b) L1-(d;d;e) is of (d;d;e) type",
           "PART True True" in out, out)
    report("generic historical vectors are kept (Pm-3m R4+ unchanged)",
           "LEGACY True" in out, out)
    code, out = run_group(["--parent", "162", "--irrep", "L1+", "L1-"])
    report("P-31m L1+ L1- coupled table: the (d;d;e) row is still listed",
           code == 0 and re.search(r"L1\+\(a;a;b\) L1-\(d;d;e\)\s+\d+ \S+", out)
           is not None, out)

    # more than eight free parameters: the parameter values used to repeat
    # after eight letters (i = a, j = b, ...), a less generic direction
    code, out = run_group(["--parent", "219", "--irrep", "L3", "--order-parameter",
                           *"abcdefghijklmnop"])
    report("16 letters for L3 of F-43c -> P1 (index 192), not C2",
           code == 0 and "-> P1 (No. 1)" in out and "cell size 8, index 192" in out,
           out)

    # --kpoint: every irrep of one special k point in a single table
    def table_rows(text: str) -> list[tuple]:
        rows, inside = [], False
        for line in text.splitlines():
            if line.startswith("irrep ") and "subgroup" in line:
                inside = True
            elif inside and not line.strip():
                inside = False
            elif inside:
                rows.append(tuple(line.split()))
        return rows

    code, gm_out = run_group(["--parent", "Pm-3m", "--kpoint", "GM"])
    gm_rows = table_rows(gm_out)
    expected_labels = (
        "GM1+(a) GM2+(a) GM3+(a,0) GM3+(a,b) GM4+(0,0,a) GM4+(a,a,a) "
        "GM4+(0,a,a) GM4+(a,b,c) GM5+(a,a,a) GM5+(0,0,a) GM5+(a,a,b) "
        "GM5+(a,b,c) GM1-(a) GM2-(a) GM3-(a,0) GM3-(0,a) GM3-(a,b) "
        "GM4-(0,0,a) GM4-(a,a,a) GM4-(0,a,a) GM4-(0,a,b) GM4-(a,a,b) "
        "GM4-(a,b,c) GM5-(0,0,a) GM5-(a,a,a) GM5-(0,a,a) GM5-(a,a,b) "
        "GM5-(0,a,b) GM5-(a,b,c)").split()
    report("--kpoint GM exit 0, no traceback",
           code == 0 and "Traceback" not in gm_out, gm_out)
    report("--kpoint GM lists the 29 directions of the ten GM irreps in order",
           [row[0] for row in gm_rows] == expected_labels,
           "\n".join(" ".join(row) for row in gm_rows))
    report("first and last rows: GM1+(a) keeps Pm-3m, GM5-(a,b,c) -> P1 (index 48)",
           len(gm_rows) == 29
           and gm_rows[0] == ("GM1+(a)", "221", "Pm-3m", "1", "1")
           and gm_rows[-1] == ("GM5-(a,b,c)", "1", "P1", "1", "48"), gm_out)
    report("layout: Supergroup and Kpoint blocks, one table, no Irrep block",
           "* Supergroup *\nPm-3m (No. 221)\n\n* Kpoint *\nGM\n" in gm_out
           and gm_out.count("irrep                subgroup") == 1
           and gm_out.count("* Order parameter directions and isotropy subgroups *") == 1
           and "* Irrep *" not in gm_out
           and "Conventions and validation: ISOSUBGROUP" in gm_out, gm_out)

    # the k-point table is the single-irrep tables one after the other
    # (multi-arm stars included); one interpreter for the whole comparison
    probe = (
        "import contextlib, io\n"
        "from crystod.cli.group import main\n"
        "def rows(argv):\n"
        "    buffer = io.StringIO()\n"
        "    with contextlib.redirect_stdout(buffer):\n"
        "        main(argv)\n"
        "    found, inside = [], False\n"
        "    for line in buffer.getvalue().splitlines():\n"
        "        if line.startswith('irrep ') and 'subgroup' in line:\n"
        "            inside = True\n"
        "        elif inside and not line.strip():\n"
        "            inside = False\n"
        "        elif inside:\n"
        "            found.append(tuple(line.split()))\n"
        "    return found\n"
        "for sg, k in (('Pm-3m', 'M'), ('I4/mmm', 'X'), ('Cmcm', 'Y'), ('Pm-3', 'GM'),\n"
        "              ('I4/mcm', 'P')):\n"
        "    table = rows(['--parent', sg, '--kpoint', k])\n"
        "    labels = []\n"
        "    for row in table:\n"
        "        label = row[0].split('(')[0]\n"
        "        if label not in labels:\n"
        "            labels.append(label)\n"
        "    single = []\n"
        "    for label in labels:\n"
        "        # a pair label (GM2+GM3+) is listed once, under its first member\n"
        "        first = label[:len(label) // 2] if label.count(k) == 2 else label\n"
        "        single += rows(['--parent', sg, '--irrep', first])\n"
        "    print('CONCAT', sg, k, len(table), len(labels), table == single)\n"
    )
    code, out = run_python(probe)
    report("--kpoint table == concatenated --irrep tables (Pm-3m M: 3 arms, 80 rows)",
           code == 0 and "CONCAT Pm-3m M 80 10 True" in out, out)
    report("--kpoint table == concatenated --irrep tables (I4/mmm X: 2 arms)",
           "CONCAT I4/mmm X 24 8 True" in out, out)
    report("--kpoint table == concatenated --irrep tables (Cmcm Y)",
           "CONCAT Cmcm Y 8 8 True" in out, out)
    report("complex-conjugate pair listed once (Pm-3 GM: GM2+GM3+, GM2-GM3-)",
           "CONCAT Pm-3 GM 11 6 True" in out, out)
    report("... and at P of I4/mcm (P1P3, P2P4, P5: the tables of P1, P2, P5)",
           "CONCAT I4/mcm P 24 3 True" in out, out)

    # conjugate pairs that the characters of the coset representatives do
    # not decide (the partners differ only on lattice translations): the
    # pair labels are those of the ISOSUBGROUP tables, and every physically
    # irreducible order parameter of the k point is listed exactly once
    probe = (
        "import crystod.isotropy_subgroup as iso\n"
        "from crystod.group import isotropy_subgroups, isotropy_subgroups_at_kpoint\n"
        "def blocks(sg, k):\n"
        "    table = isotropy_subgroups_at_kpoint(sg, k, with_settings=False)\n"
        "    return ' '.join(f'{label}:{len(subs)}' for label, subs in table.items())\n"
        "print('P3K', blocks('P3', 'K'))\n"
        "print('P-6K', blocks('P-6', 'K'))\n"
        "print('I-4P', blocks('I-4', 'P'))\n"
        "print('176H', blocks('P6_3/m', 'H'))\n"
        "print('188H', blocks('P-6c2', 'H'))\n"
        "table = isotropy_subgroups_at_kpoint('I4/mcm', 'P', with_settings=False)\n"
        "print('140P', list(table), [sub.number for sub in table['P2P4']])\n"
        "single = isotropy_subgroups('I4/mcm', 'P2', with_settings=False)\n"
        "print('140P2', sorted({sub.irrep for sub in single}),\n"
        "      [str(sub) for sub in single] == [str(sub) for sub in table['P2P4']])\n"
        "# the pair is recognized by the representation, not by its label: with\n"
        "# the partners mislabelled no order parameter is dropped\n"
        "wrong = {'K2': 'K3', 'K3': 'K2'}\n"
        "original = iso.InducedRepresentation.conjugate_partner\n"
        "iso.InducedRepresentation.conjugate_partner = (\n"
        "    lambda self: wrong.get(self.irrep.name, original(self)))\n"
        "table = isotropy_subgroups_at_kpoint('P3', 'K', with_settings=False)\n"
        "print('MISLABEL', list(table), [len(subs) for subs in table.values()])\n"
        "# three decimals name a special point, two do not\n"
        "print('DECIMAL', isotropy_subgroups_at_kpoint('P3', [0.333, 0.333, 0]).kpoint,\n"
        "      isotropy_subgroups_at_kpoint('P3', ['0.667', '-0.333', '0.5']).kpoint)\n"
        "try:\n"
        "    isotropy_subgroups_at_kpoint('P3', [0.33, 0.33, 0])\n"
        "except ValueError as exc:\n"
        "    print('COARSE', 'is not a special k point' in str(exc))\n"
    )
    code, out = run_python(probe)
    report("P3 K: three pairs with the -K irreps (K1KA1, K2KA2, K3KA3)",
           code == 0 and "P3K K1KA1:1 K2KA2:1 K3KA3:1" in out, out)
    report("P-6 K: six pairs KnKAn, none dropped",
           "P-6K K1KA1:1 K2KA2:1 K3KA3:1 K4KA4:1 K5KA5:1 K6KA6:1" in out, out)
    report("I-4 P: four pairs PnPAn",
           "I-4P P1PA1:1 P2PA2:1 P3PA3:1 P4PA4:1" in out, out)
    report("P6_3/m H and P-6c2 H: each pair once (H3H6 + H4H5, H3H4 + H5H6)",
           "176H H1H2:3 H3H6:3 H4H5:3" in out
           and "188H H1H2:3 H3H4:3 H5H6:3" in out, out)
    report("I4/mcm P: P1P3, P2P4, P5; P2P4 -> Immm, Imma, I-4m2, Imm2",
           "140P ['P1P3', 'P2P4', 'P5'] [71, 74, 119, 44]" in out, out)
    report("--irrep P2 of I4/mcm carries the same label and rows (P2P4)",
           "140P2 ['P2P4'] True" in out, out)
    report("mislabelled partners: no order parameter is dropped from the table",
           "MISLABEL ['K1KA1', 'K2K3', 'K3'] [1, 1, 1]" in out, out)
    report("coordinates: 0.333 is taken for 1/3 (K, HA of P3), 0.33 is not",
           "DECIMAL K HA" in out and "COARSE True" in out, out)

    # ISOTROPY's -k partner names (data_little.txt of the ISOTROPY software:
    # LD1LE1, DT6DU6, GP1GQ1, P1PA1, P1PC1 of P3, B1BC1 of P-6, D1DC1 of
    # P3m1) and their inverse; ZA is a k type of its own in P23
    probe = (
        "import contextlib, io\n"
        "import numpy as np\n"
        "from crystod.isoir import minus_k_base, minus_k_type\n"
        "from crystod.isotropy_subgroup import ComputedInducedRepresentation\n"
        "from crystod.spacegroup_product import SpaceGroupIrrepAlgebra\n"
        "cases = [(4, 'LD'), (173, 'DT'), (75, 'SM'), (1, 'GP'), (82, 'P'),\n"
        "         (143, 'P'), (143, 'K'), (156, 'P'), (156, 'D'), (157, 'C'),\n"
        "         (174, 'B'), (174, 'P'), (119, 'B')]\n"
        "print('TYPES', ' '.join(f'{sg}:{t}>{minus_k_type(t, sg)}' for sg, t in cases))\n"
        "print('BASE', minus_k_base('LE1', 4), minus_k_base('DU6', 173),\n"
        "      minus_k_base('PC2', 143), minus_k_base('LD1', 4), minus_k_base('ZA1', 195))\n"
        "with contextlib.redirect_stdout(io.StringIO()):\n"
        "    algebra = SpaceGroupIrrepAlgebra('P2_1')\n"
        "k = np.array([0, 13, 0])\n"
        "point, names, _ = algebra._line_names(k)\n"
        "routes = []\n"
        "for small, name in zip(algebra.computed_irreps_at(k), names):\n"
        "    try:\n"
        "        ComputedInducedRepresentation.from_isoir(algebra, k, small, name, point)\n"
        "        routes.append(name + ':iso')\n"
        "    except LookupError:\n"
        "        routes.append(name + ':spgrep')\n"
        "print('ROUTE', ' '.join(routes))\n"
    )
    code, out = run_python(probe)
    report("-k partner names as in ISOTROPY (LE, DU, SN, GQ, PA, PC, KA, DC, BC, BA)",
           code == 0 and "TYPES 4:LD>LE 173:DT>DU 75:SM>SN 1:GP>GQ 82:P>PA "
           "143:P>PC 143:K>KA 156:P>PA 156:D>DC 157:C>CC 174:B>BC 174:P>PA "
           "119:B>BA" in out, out)
    report("partner names map back to the tabulated labels (ZA of P23 is not one)",
           "BASE LD1 DT6 P2 None None" in out, out)
    report("a partner star (LE of P2_1) is built from the ISO-IR matrices",
           "ROUTE LE2:iso LE1:iso" in out, out)

    # coordinates: any arm, any q + G spelling, negative components
    code, r_out = run_group(["--parent", "Pm-3m", "--kpoint", "R"])
    report("--kpoint R: R4+ tilt subgroups inside the 34-row R table",
           code == 0 and len(table_rows(r_out)) == 34
           and ("R4+(0,0,a)", "140", "I4/mcm", "2", "6") in table_rows(r_out), r_out)
    code, out = run_group(["--supergroup", "Pm-3m", "--kpoint", "0.5", "0.5", "0.5"])
    report("--kpoint 0.5 0.5 0.5 (and the --supergroup alias) is the R table",
           code == 0 and out == r_out, out)
    code, out = run_group(["--parent", "Pm-3m", "--kpoint", "-0.5", "0.5", "0.5"])
    report("--kpoint -0.5 0.5 0.5 (negative component) is the R table",
           code == 0 and out == r_out, out)
    code, out = run_group(["--parent", "Pm-3m", "--kpoint", "-1/2", "1/2", "3/2"])
    report("--kpoint -1/2 1/2 3/2 (negative fraction, q + G) is the R table",
           code == 0 and out == r_out, out)
    probe = (
        "import contextlib, io\n"
        "from crystod.cli.group import main\n"
        "def text(*kpoint):\n"
        "    buffer = io.StringIO()\n"
        "    with contextlib.redirect_stdout(buffer):\n"
        "        main(['--parent', 'Pm-3m', '--kpoint', *kpoint])\n"
        "    return buffer.getvalue()\n"
        "m = text('M')\n"
        "print('KLINE', m.splitlines()[5:7])\n"
        "print('ARM', text('0', '0.5', '0.5') == m, text('1/2', '0', '-1/2') == m,\n"
        "      text('0.5 1.5 0') == m)\n"
        "gm = text('GM')\n"
        "print('NAME', text('gm') == gm, text('Gamma') == gm, text('G') == gm,\n"
        "      text('0', '0', '0') == gm)\n"
    )
    code, out = run_python(probe)
    report("a non-representative arm of the M star resolves to the M table",
           code == 0 and "ARM True True True" in out, out)
    report("k-point names: case ignored, G/Gamma accepted for GM",
           "NAME True True True True" in out, out)
    report("Kpoint block: the name alone, then the tabulated arm and the star",
           "KLINE ['M', 'k = (1/2, 1/2, 0) in the primitive basis, star of 3 arm(s)']"
           in out, out)

    # notes of the single-irrep mode, collected over the whole table
    code, out = run_group(["--parent", "214", "--kpoint", "N"])
    report("enantiomorph note over the whole N table of I4_132",
           code == 0 and "91 <-> 95, 92 <-> 96 are enantiomorphic partner types"
           in out, out)
    # Y of Cmce: every irrep carries the subgroup ISOSUBGROUP lists under
    # the same name (SUBGROUP/SUBGROUP_SG64_Y.txt); no label note is needed
    code, out = run_group(["--parent", "Cmce", "--kpoint", "Y"])
    cmce_y = {row[0]: row[1] for row in table_rows(out)}
    report("Cmce Y: Y1+..Y4- carry the ISOSUBGROUP subgroups by name",
           code == 0 and cmce_y == {
               "Y1+(a)": "53", "Y2+(a)": "61", "Y3+(a)": "55", "Y4+(a)": "56",
               "Y1-(a)": "54", "Y2-(a)": "62", "Y3-(a)": "60", "Y4-(a)": "57"}
           and "ISOTROPY Y" not in out, out)

    # W of Ibca (73) and of I2_12_12_1 (24, W and its partner WA): spgrep
    # 0.6.0 builds the valid two-dimensional small irrep there, then rejects
    # it with a Frobenius-Schur check that holds only for 2k = 0 ("not
    # irreducible"); runtime_compat rebuilds it from spgrep's own steps.
    # Reference: SUBGROUP/SUBGROUP_SG73_W.txt (ISOSUBGROUP); 24 has no table
    # and was checked against a build from the ISO-IR matrices
    code, out = run_group(["--parent", "73", "--kpoint", "W"])
    w_rows = sorted(line.split()[1:] for line in out.splitlines()
                    if line.startswith("W1("))
    report("--kpoint W of Ibca: six W1 rows (2x P-1 and 3x C2 at size 4 "
           "index 16, P1 at index 32)",
           code == 0 and "not enumerated" not in out
           and w_rows == sorted([["2", "P-1", "4", "16"]] * 2
                                + [["5", "C2", "4", "16"]] * 3
                                + [["1", "P1", "4", "32"]]), out)
    code, out = run_group(["--parent", "24", "--kpoint", "W"])
    w_rows = sorted(line.split()[1:] for line in out.splitlines()
                    if line.startswith("W1WA1("))
    report("--kpoint W of I2_12_12_1: four W1WA1 rows (3x C2 at size 4 "
           "index 8, P1 at index 16)",
           code == 0 and "not enumerated" not in out
           and w_rows == sorted([["5", "C2", "4", "8"]] * 3
                                + [["1", "P1", "4", "16"]]), out)

    # the wrapper itself, on the primitive ISO-IR operations of both groups:
    # where spgrep raises, irreducible, inequivalent and complete small
    # irreps (sum of the squared dimensions = order of the little co-group,
    # <chi_a, chi_b> = order delta_ab); everywhere else spgrep's own
    # arrays, unchanged
    from spgrep.core import (
        get_spacegroup_irreps_from_primitive_symmetry as _spgrep_irreps,
    )

    from crystod.runtime_compat import (
        get_spacegroup_irreps_from_primitive_symmetry as _wrapped_irreps,
    )
    from crystod.spacegroup_product import DEN as _DEN
    from crystod.spacegroup_product import SpaceGroupIrrepAlgebra as _Algebra

    rebuilt, unchanged, compared = [], True, 0
    for number in ("24", "73"):
        algebra = _Algebra(number)
        for kname in sorted(algebra.k_by_kname):
            arguments = dict(
                rotations=algebra.rotations,
                translations=np.array(algebra.translations, dtype=float) / _DEN,
                kpoint=np.array(algebra.k_by_kname[kname], dtype=float) / _DEN,
            )
            irreps, mapping = _wrapped_irreps(**arguments)
            try:
                reference = _spgrep_irreps(**arguments)
            except ValueError:
                order = len(mapping)
                chars = np.array([np.trace(rep, axis1=1, axis2=2)
                                  for rep in irreps])
                rebuilt.append((
                    f"{number} {kname}",
                    [rep.shape[1] for rep in irreps],
                    sum(rep.shape[1] ** 2 for rep in irreps) == order
                    # <chi_a, chi_b> = order delta_ab: irreducible and
                    # mutually inequivalent
                    and np.allclose(np.conj(chars) @ chars.T,
                                    order * np.eye(len(irreps)), atol=1e-6),
                ))
                continue
            compared += 1
            unchanged = unchanged and np.array_equal(
                np.asarray(reference[1]), np.asarray(mapping)
            ) and len(reference[0]) == len(irreps) and all(
                np.array_equal(a, b) for a, b in zip(reference[0], irreps)
            )
    report("spgrep wrapper rebuilds the 2-dim small irreps at 24 W, 24 WA and "
           "73 W, irreducible and complete",
           sorted(name for name, _, _ in rebuilt) == ["24 W", "24 WA", "73 W"]
           and all(dims == [2] and ok for _, dims, ok in rebuilt), str(rebuilt))
    report("spgrep wrapper returns spgrep's own output at every other k point "
           "of 24 and 73",
           unchanged and compared >= 10, f"compared {compared}")
    # the rebuild refuses a set that passes the dimension count and the
    # per-irrep norm but repeats one irrep (four copies of a 1-dim
    # representation at 73 W: sum d^2 = 4, |chi|^2 = 4 each, not
    # inequivalent) -- spgrep's own error is raised again.  spgrep itself
    # is made to raise first, so that only the rebuild sees the fake set
    import spgrep.core as _spgrep_core

    try:
        import spgrep.symmetry.enumerate as _spgrep_enumerate
    except ImportError:  # older spgrep layout: the wrapper imports spgrep.irreps
        import spgrep.irreps as _spgrep_enumerate

    def _spgrep_raises(*args, **kwargs):
        raise ValueError("Given representation is not irreducible: "
                         "indicator=2 (test)")

    _core = _spgrep_core.get_spacegroup_irreps_from_primitive_symmetry
    _chain = _spgrep_enumerate.enumerate_unitary_irreps_from_solvable_group_chain
    _spgrep_core.get_spacegroup_irreps_from_primitive_symmetry = _spgrep_raises
    _spgrep_enumerate.enumerate_unitary_irreps_from_solvable_group_chain = (
        lambda table, *args, **kwargs: [np.ones((len(table), 1, 1),
                                                dtype=complex)] * 4
    )
    _algebra = _Algebra("73")
    try:
        _wrapped_irreps(
            rotations=_algebra.rotations,
            translations=np.array(_algebra.translations, dtype=float) / _DEN,
            kpoint=np.array(_algebra.k_by_kname["W"], dtype=float) / _DEN,
        )
        _refused = "accepted"
    except ValueError as _exc:
        _refused = str(_exc)
    finally:
        _spgrep_core.get_spacegroup_irreps_from_primitive_symmetry = _core
        _spgrep_enumerate.enumerate_unitary_irreps_from_solvable_group_chain = _chain
    report("spgrep wrapper refuses a rebuilt set with equivalent irreps",
           "not irreducible" in _refused, _refused)

    # one failing irrep must not take the table down (independent of any
    # real failure: the enumeration of one irrep is made to raise)
    probe = (
        "import contextlib, io\n"
        "import numpy as np\n"
        "import crystod.isotropy_subgroup as iso\n"
        "from crystod.cli.group import main\n"
        "from crystod.group import isotropy_subgroups_at_kpoint\n"
        "original = iso.IsotropyAnalyzer.enumerate_directions\n"
        "broken = {'GM4-'}\n"
        "def enumerate_directions(self):\n"
        "    if broken is None or self.representation.label in broken:\n"
        "        raise np.linalg.LinAlgError('Singular matrix')\n"
        "    return original(self)\n"
        "iso.IsotropyAnalyzer.enumerate_directions = enumerate_directions\n"
        "def cli():\n"
        "    out, err = io.StringIO(), io.StringIO()\n"
        "    status = 0\n"
        "    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):\n"
        "        try:\n"
        "            main(['--parent', 'Pm-3m', '--kpoint', 'GM'])\n"
        "        except SystemExit as exc:\n"
        "            status = exc.code\n"
        "    return status, out.getvalue()\n"
        "status, text = cli()\n"
        "rows = [l for l in text.splitlines() if l.startswith('GM') and '(' in l]\n"
        "print('ONE', status, len(rows), any(l.startswith('GM4-') for l in rows),\n"
        "      any(l.startswith('GM5-') for l in rows))\n"
        "print('NOTE', 'note: GM4-: not enumerated (LinAlgError: Singular matrix)' in text)\n"
        "table = isotropy_subgroups_at_kpoint('Pm-3m', 'GM', with_settings=False)\n"
        "print('API', len(table), 'GM4-' in table, table.errors)\n"
        "broken = None\n"
        "status, text = cli()\n"
        "print('ALL', str(status).splitlines()[0], '* Order parameter' in text)\n"
        "table = isotropy_subgroups_at_kpoint('Pm-3m', 'GM', with_settings=False)\n"
        "print('APIALL', len(table), len(table.errors))\n"
    )
    code, out = run_python(probe)
    report("a failing irrep leaves the other irreps in the table (exit 0)",
           code == 0 and "ONE 0 23 False True" in out, out)
    report("the failed irrep gets one note line under the table",
           "NOTE True" in out and "Traceback" not in out, out)
    report("API: the failed irrep is reported in .errors, the others returned",
           "API 9 False {'GM4-': 'LinAlgError: Singular matrix'}" in out, out)
    report("no irrep enumerated: clean ERROR and nonzero exit, no table",
           "ALL ERROR: no irrep at GM could be enumerated: False" in out
           and "APIALL 0 10" in out, out)

    # errors
    code, out = run_group(["--parent", "Pm-3m"])
    report("--parent without --irrep rejected cleanly",
           code != 0 and "requires --irrep" in out and "Traceback" not in out, out)
    report("... and names --kpoint and the k points of the space group",
           "requires --irrep or --kpoint" in out
           and "GM (0, 0, 0), R (1/2, 1/2, 1/2), X (0, 1/2, 0), M (1/2, 1/2, 0)"
           in out, out)

    code, out = run_group(["--parent", "Pm-3m", "--kpoint", "Q"])
    report("unknown k-point name rejected with the available k points",
           code != 0 and 'k point "Q" is not tabulated' in out
           and "GM (0, 0, 0), R (1/2, 1/2, 1/2), X (0, 1/2, 0), M (1/2, 1/2, 0)"
           in out and "Traceback" not in out, out)

    code, out = run_group(["--parent", "Pm-3m", "--kpoint", "R4+"])
    report("an irrep label given to --kpoint points to --irrep and its k point",
           code != 0 and '"R4+" is an irrep label' in out
           and "its k point is R" in out and "--irrep R4+" in out
           and "Available k points" in out and "Traceback" not in out, out)

    code, out = run_group(["--parent", "Pm-3m", "--kpoint", "-0.5"])
    report("a single number given to --kpoint asks for a name or three coordinates",
           code != 0 and "either one name (e.g. GM) or three coordinates" in out
           and "Available k points" in out and "Traceback" not in out, out)

    code, out = run_group(["--parent", "Nope"])
    report("--parent with an unknown space group and nothing else says so",
           code != 0 and "requires --irrep or --kpoint" in out
           and '"Nope" is not recognized' in out and "Traceback" not in out, out)

    code, out = run_group(["--parent", "Pm-3m", "--kpoint", "0.25", "0", "0"])
    report("a k point on a symmetry line rejected with the available k points",
           code != 0 and "is not a special k point" in out
           and "Available k points" in out and "Traceback" not in out, out)

    code, out = run_group(["--parent", "Pm-3m", "--kpoint", "GM", "--irrep", "GM4-"])
    report("--kpoint together with --irrep rejected cleanly",
           code != 0 and "either --irrep" in out and "not both" in out
           and "Traceback" not in out, out)

    code, out = run_group(["--parent", "Pm-3m", "--kpoint", "GM",
                           "--order-parameter", "0", "0", "a"])
    report("--kpoint together with --order-parameter rejected cleanly",
           code != 0 and "not used with --kpoint" in out
           and "Traceback" not in out, out)

    # the other space-group modes share --kpoint and still take coordinates
    code, out = run_group(["--table", "--sg", "Pm-3m", "--kpoint", "GM"])
    report("a k-point name given to --table --sg asks for three coordinates",
           code != 0 and "expected 3 arguments" in out
           and "three coordinates" in out and "only accepted with --parent" in out
           and "Traceback" not in out, out)
    code, out = run_group(["--table", "--sg", "Pm-3m", "--kpoint", "0", "0"])
    report("--table --sg with two coordinates: argparse message unchanged",
           code != 0 and out.rstrip().endswith(
               "error: argument --kpoint: expected 3 arguments"), out)
    code, out = run_group(["--table", "--sg", "Pm-3m", "--kpoint", "-0.5", "0.5", "0"])
    code2, out2 = run_group(["--coset", "--sg", "Pm-3m", "--kpoint", "-1/2", "1/2", "0"])
    report("negative --kpoint coordinates still parse in --table and --coset",
           code == 0 and "M [-0.5, 0.5, 0.0]" in out
           and code2 == 0 and "[-0.5, 0.5, 0.0]" in out2, out + out2)

    code, out = run_group(["--parent", "Pm-3m", "--irrep", "QQ9"])
    report("unknown irrep rejected with available list",
           code != 0 and "not tabulated" in out and "Available irreps" in out
           and "Traceback" not in out, out)

    code, out = run_group(["--parent", "Pm-3m", "--irrep", "GM4-",
                           "--order-parameter", "0", "0"])
    report("wrong order-parameter length rejected cleanly",
           code != 0 and "needs 3 components" in out and "Traceback" not in out, out)

    code, out = run_group(["--product", "T2g", "T2g", "--pg", "m-3m",
                           "--irrep", "GM4-"])
    report("--irrep outside --parent rejected cleanly",
           code != 0 and "only used with --parent" in out
           and "Traceback" not in out, out)


def test_18_invariants() -> None:
    print("\n[18] crystod-group --parent --irrep --invariants / --secondary "
          "(invariant polynomials, coupling terms, secondary order parameters)")

    # (a) the octahedral-tilt irrep R4+ of Pm-3m: one quadratic invariant,
    # no cubic one, the square of I2 plus one new quartic invariant
    code, out = run_group(["--parent", "Pm-3m", "--irrep", "R4+",
                           "--invariants", "--degree", "4"])
    report("Pm-3m R4+ --invariants exit 0", code == 0, out)
    report("R4+: header block and the component order",
           "* Supergroup *" in out and "* Irrep *" in out
           and "* Invariant polynomials (degree <= 4) *" in out
           and "Order-parameter components: (Q1, Q2, Q3), in the order of the "
               "direction patterns (a, b, c)" in out, out)
    report("R4+: degree 2 -> 1 invariant Q1^2 + Q2^2 + Q3^2",
           "degree 2: 1 invariant" in out and "I2_1 = Q1^2 + Q2^2 + Q3^2" in out, out)
    report("R4+: degree 3 -> none (Landau condition)", "degree 3: none" in out, out)
    report("R4+: degree 4 -> I2_1^2 and one new invariant",
           "degree 4: 2 invariants (1 from lower degrees, 1 new)" in out
           and "\n  I2_1^2\n" in out
           and "I4_1 = Q1^2*Q2^2 + Q1^2*Q3^2 + Q2^2*Q3^2" in out, out)
    report("R4+: Molien check line, INVARIANTS citation, no subgroup table",
           "Numbers of invariants checked against the Molien series." in out
           and "ISOTROPY INVARIANTS (https://iso.byu.edu)" in out
           and "J. Appl. Cryst. 36, 951-952 (2003)" in out
           and "isotropy subgroups *" not in out and "ISOSUBGROUP" not in out, out)

    # (b) a cubic invariant: the three-arm M1+ of Pm-3m (k1+k2+k3 = 0);
    # the in-phase tilt M3+ has none (the axial components x y z change sign
    # under the fourfold rotations)
    code, out = run_group(["--parent", "Pm-3m", "--irrep", "M1+", "--invariants"])
    report("Pm-3m M1+: one cubic invariant Q1*Q2*Q3 (default degree 4)",
           code == 0 and "degree 3: 1 invariant" in out
           and "I3_1 = Q1*Q2*Q3" in out
           and "(degree <= 4)" in out and "(a; b; c)" in out, out)
    code, out = run_group(["--parent", "Pm-3m", "--irrep", "M3+", "--invariants"])
    report("Pm-3m M3+: no cubic invariant, 2 quartic",
           code == 0 and "degree 3: none" in out
           and "degree 4: 2 invariants" in out, out)

    # (c) the polar vector GM4-
    code, out = run_group(["--parent", "Pm-3m", "--irrep", "GM4-", "--invariants"])
    report("Pm-3m GM4-: degree 2 -> 1, degree 4 -> 2",
           code == 0 and "degree 2: 1 invariant" in out
           and "degree 3: none" in out and "degree 4: 2 invariants" in out, out)
    code, out = run_group(["--parent", "Pm-3m", "--irrep", "GM3+", "--invariants",
                           "--degree", "3"])
    report("Pm-3m GM3+ --degree 3: the cubic invariant of the E irrep",
           code == 0 and "(degree <= 3)" in out
           and "I3_1 = Q1^3 - 3*Q1*Q2^2" in out and "degree 4" not in out, out)

    # (d) error paths
    code, out = run_group(["--parent", "Pm-3m", "--invariants"])
    report("--invariants without --irrep rejected cleanly",
           code != 0 and "--invariants requires --irrep" in out
           and "Traceback" not in out, out)
    code, out = run_group(["--parent", "Pm-3m", "--kpoint", "R", "--invariants"])
    report("--invariants with --kpoint rejected cleanly",
           code != 0 and "not used with --kpoint" in out
           and "Traceback" not in out, out)
    code, out = run_group(["--parent", "Fm-3m", "--irrep", "W5", "--invariants",
                           "--degree", "8"])
    report("--degree 8 of the 12-component W5: refused before the work (75582 monomials)",
           code == 1 and "ERROR: degree 8 needs 75582 monomials" in out
           and "lower --degree" in out and "Traceback" not in out
           and "* Invariant polynomials" not in out, out)
    code, out = run_group(["--parent", "Pm-3m", "--irrep", "R4+", "--degree", "3"])
    report("--degree without --invariants rejected cleanly",
           code != 0 and "only used with --invariants" in out
           and "Traceback" not in out, out)
    code, out = run_group(["--table", "--pg", "m-3m", "--invariants"])
    report("--invariants outside --parent rejected cleanly",
           code != 0 and "only used with --parent" in out
           and "Traceback" not in out, out)

    # (e) self-consistency: nullspace dimension = Molien count (checked inside
    # the engine) for every special-point irrep of Pm-3m, and for a doubled
    # complex-type irrep and a 6- and a 12-component W irrep of Fm-3m; a
    # physically irreducible real representation has exactly one quadratic
    # invariant.  Every PRINTED polynomial is parsed back (^ -> **) and
    # evaluated at random points under every distinct matrix of the group
    # (tolerance 1e-5 where 6-decimal coefficients are printed).
    probe = (
        "import re, time\n"
        "import numpy as np\n"
        "from crystod import group\n"
        "from crystod.invariants import _polynomial_text\n"
        "rng = np.random.default_rng(5)\n"
        "def invariance(sg, label, basis):\n"
        "    rep = group.InducedRepresentation(group.SpaceGroupIrrepAlgebra(sg), label)\n"
        "    mats = {m.round(8).tobytes(): m for _, _, m in rep.elements}.values()\n"
        "    X = rng.normal(size=(6, len(basis.variables)))\n"
        "    worst, n_float = 0.0, 0\n"
        "    for d in basis.degrees:\n"
        "        for poly in basis.polynomials[d]:\n"
        "            text = poly.expression.replace('^', '**')\n"
        "            n_float += bool(re.search(r'\\d\\.\\d', text))\n"
        "            code = re.sub(r'Q(\\d+)', lambda m: f'x[:, {int(m.group(1)) - 1}]', text)\n"
        "            f = lambda x: eval(code, {'sqrt': np.sqrt, 'x': x})\n"
        "            ref = f(X)\n"
        "            for M in mats:\n"
        "                worst = max(worst, float(np.max(np.abs(f(X @ M.T) - ref))))\n"
        "    return worst, n_float\n"
        "algebra = group.SpaceGroupIrrepAlgebra('Pm-3m')\n"
        "bad, n, worst_pm3m = [], 0, 0.0\n"
        "for kname in ('GM', 'X', 'M', 'R'):\n"
        "    for irrep in algebra.irreps_by_kname[kname]:\n"
        "        n += 1\n"
        "        try:\n"
        "            basis = group.invariant_polynomials('Pm-3m', [irrep.name], 4)\n"
        "        except RuntimeError as exc:\n"
        "            bad.append(f'{irrep.name}: {exc}')\n"
        "            continue\n"
        "        counts = basis.counts\n"
        "        if (sorted(counts) != [1, 2, 3, 4]\n"
        "                or any(not isinstance(v, int) or v < 0 for v in counts.values())\n"
        "                or counts[2] != 1\n"
        "                or any(len(basis.polynomials[d]) + len(basis.products[d])\n"
        "                       != counts[d] for d in counts)):\n"
        "            bad.append(f'{irrep.name}: {counts}')\n"
        "        worst_pm3m = max(worst_pm3m, invariance('Pm-3m', irrep.name, basis)[0])\n"
        "print('PM3M', n, 'BAD', bad)\n"
        "print('PRINTED PM3M', worst_pm3m < 1e-9, f'{worst_pm3m:.1e}')\n"
        "for sg, label, tol in (('212', 'R3', 1e-9), ('218', 'R4', 1e-9),\n"
        "                       ('225', 'W1', 1e-5)):\n"
        "    basis = group.invariant_polynomials(sg, [label], 4)\n"
        "    worst, n_float = invariance(sg, label, basis)\n"
        "    print('PRINTED', sg, label, basis.counts[2] == 1, worst < tol, n_float > 0,\n"
        "          f'{worst:.1e}')\n"
        "    if sg == '225':\n"
        "        import sympy\n"
        "        rep = group.InducedRepresentation(group.SpaceGroupIrrepAlgebra(sg), label)\n"
        "        poly = next(p for d in basis.degrees for p in basis.polynomials[d]\n"
        "                    if re.search(r'\\d\\.\\d', p.expression))\n"
        "        symbols = [sympy.Symbol(v) for v in basis.variables]\n"
        "        f = sympy.lambdify(symbols, poly.sympy(basis.variables), 'numpy')\n"
        "        X = rng.normal(size=(6, len(symbols)))\n"
        "        err = max(float(np.max(np.abs(f(*(X @ m.T).T) - f(*X.T))))\n"
        "                  for _, _, m in rep.elements[:200])\n"
        "        print('SYMPY FULL PRECISION', err < 1e-11, f'{err:.1e}')\n"
        "row = np.array([[2, 0], [0, 2]])\n"
        "print('TEXT', repr(_polynomial_text(['1', '-1 + sqrt(3)'], row, ['Q1', 'Q2'])))\n"
        "print('TEXT', repr(_polynomial_text(['1', '-sqrt(2) - 1'], row, ['Q1', 'Q2'])))\n"
        "print('TEXT', repr(_polynomial_text(['1', '3 - 2*sqrt(3)'], row, ['Q1', 'Q2'])))\n"
        "for sg, label in (('P-43n', 'R4'), ('Fm-3m', 'W1'), ('Fm-3m', 'W5')):\n"
        "    start = time.time()\n"
        "    basis = group.invariant_polynomials(sg, [label], 4)\n"
        "    print('BIG', sg, label, len(basis.variables), basis.counts[2],\n"
        "          basis.counts[3], basis.counts[4], f'{time.time() - start:.1f}')\n"
    )
    code, out = run_python(probe)
    pm3m = re.search(r"PM3M (\d+) BAD \[\]", out)
    report("every GM/X/M/R irrep of Pm-3m: nullspace = Molien count at degrees 1-4, "
           "exactly one quadratic invariant",
           code == 0 and pm3m is not None and int(pm3m.group(1)) == 40, out)
    report("every printed polynomial of the Pm-3m sweep is invariant (parsed back)",
           "PRINTED PM3M True" in out, out)
    report("printed polynomials of 212 R3 (compound surds), 218 R4 (doubled) and "
           "225 W1 (6-decimal floats) are invariant",
           "PRINTED 212 R3 True True False" in out
           and "PRINTED 218 R4 True True False" in out
           and "PRINTED 225 W1 True True True" in out, out)
    report("InvariantPolynomial.sympy() uses the full-precision coefficients",
           "SYMPY FULL PRECISION True" in out, out)
    report("_polynomial_text treats a compound coefficient as one signed unit, "
           "positive terms first",
           "TEXT 'Q1^2 + (sqrt(3) - 1)*Q2^2'" in out
           and "TEXT 'Q1^2 - (1 + sqrt(2))*Q2^2'" in out
           and "TEXT 'Q1^2 - (2*sqrt(3) - 3)*Q2^2'" in out, out)
    big = {(m.group(1), m.group(2)): (int(m.group(3)), m.group(4), float(m.group(5)))
           for m in re.finditer(r"BIG (\S+) (\S+) (\d+) (\d+ \d+ \d+) ([\d.]+)", out)}
    report("P-43n R4 (complex pair, 6 components): 1, 0, 9 invariants in < 20 s",
           big.get(("P-43n", "R4"), (0, "", 99))[:2] == (6, "1 0 9")
           and big[("P-43n", "R4")][2] < 20, out)
    report("Fm-3m W1 (6 components): 1, 0, 3 invariants in < 20 s",
           big.get(("Fm-3m", "W1"), (0, "", 99))[:2] == (6, "1 0 3")
           and big[("Fm-3m", "W1")][2] < 20, out)
    report("Fm-3m W5 (12 components): 1, 0, 9 invariants in < 20 s",
           big.get(("Fm-3m", "W5"), (0, "", 99))[:2] == (12, "1 0 9")
           and big[("Fm-3m", "W5")][2] < 20, out)

    # (f) Landau and Lifshitz conditions
    probe = (
        "from crystod import group\n"
        "for label in ('R4+', 'M3+', 'M1+', 'GM3+'):\n"
        "    r = group.landau_lifshitz('Pm-3m', label)\n"
        "    print('LL', label, r.n_cubic, r.n_lifshitz, r.continuous_allowed)\n"
        "print('TYPE', type(r).__name__, group.LandauLifshitz.__name__,\n"
        "      group.InvariantBasis.__name__)\n"
    )
    code, out = run_python(probe)
    report("landau_lifshitz Pm-3m R4+: no cubic, no Lifshitz -> continuous allowed",
           code == 0 and "LL R4+ 0 0 True" in out, out)
    report("landau_lifshitz Pm-3m M3+: no cubic invariant (in-phase tilt)",
           "LL M3+ 0 0 True" in out, out)
    report("landau_lifshitz Pm-3m M1+ and GM3+: one cubic invariant -> first order",
           "LL M1+ 1 0 False" in out and "LL GM3+ 1 0 False" in out, out)
    report("crystod.group exports LandauLifshitz and InvariantBasis",
           "TYPE LandauLifshitz LandauLifshitz InvariantBasis" in out, out)

    # (g) the origin-shift fallback of InducedRepresentation follows the
    # conj+wrap rule of _refine_small_characters (x0 = 0 reproduces it)
    probe = (
        "import numpy as np\n"
        "from crystod.isotropy_subgroup import InducedRepresentation\n"
        "from crystod.spacegroup_product import SpaceGroupIrrepAlgebra\n"
        "for sg, label in (('4', 'Z1'), ('62', 'S1'), ('176', 'GM3+'), ('214', 'N1')):\n"
        "    algebra = SpaceGroupIrrepAlgebra(sg)\n"
        "    rep = InducedRepresentation(algebra, label)\n"
        "    table = {int(k) - 1: complex(v) for k, v in rep.irrep.characters.items()}\n"
        "    refined = algebra._refine_small_characters(np.asarray(rep.k), table)\n"
        "    shifted = rep._match_with_origin_shift(table)\n"
        "    ok = (refined is not None and shifted is not None\n"
        "          and set(refined) == set(shifted)\n"
        "          and all(abs(refined[op] - shifted[op]) < 1e-9 for op in refined))\n"
        "    print('MATCH', sg, label, ok)\n"
    )
    code, out = run_python(probe)
    report("_match_with_origin_shift agrees with _refine_small_characters "
           "(SG 4 Z1, 62 S1, 176 GM3+, 214 N1)",
           code == 0 and all(f"MATCH {case} True" in out for case in
                             ("4 Z1", "62 S1", "176 GM3+", "214 N1")), out)

    # (h) the subgroup tables around the realification guards are unchanged
    code, out = run_group(["--parent", "221", "--irrep", "R4+"])
    report("--parent 221 --irrep R4+ table unchanged",
           code == 0 and re.search(r"R4\+\(0,0,a\)\s+140 I4/mcm\s+2\s+6", out)
           and re.search(r"R4\+\(a,b,c\)\s+2 P-1\s+2\s+48", out) is not None, out)
    code, out = run_group(["--parent", "218", "--irrep", "R4"])
    report("--parent 218 --irrep R4 table unchanged (R4R5, 5 rows)",
           code == 0 and re.search(r"R4R5\(0,0,a,0,0,b\)\s+22 F222", out)
           and re.search(r"R4R5\(0,a,b,0,-b,a\)\s+82 I-4", out)
           and re.search(r"R4R5\(a,a,a,b,b,b\)\s+146 R3", out)
           and len(re.findall(r"^R4R5\(", out, re.M)) == 5, out)
    code, out = run_group(["--parent", "73", "--kpoint", "W"])
    report("--parent 73 --kpoint W table unchanged (W1, 6 rows)",
           code == 0 and re.search(r"W1\(a,b,c,d;-c,-d,a,b\)\s+2 P-1", out)
           and re.search(r"W1\(0,a,0,b;c,0,d,0\)\s+5 C2", out)
           and len(re.findall(r"^W1\(", out, re.M)) == 6, out)

    # (i) the 5000-monomial warning is a warnings.warn (no print from the
    # computation); the 20000-monomial cap is a ValueError raised up front
    probe = (
        "import warnings, numpy as np\n"
        "from crystod import invariants\n"
        "c4 = np.array([[0.0, -1.0], [1.0, 0.0]])\n"
        "group = [np.linalg.matrix_power(c4, k) for k in range(4)]\n"
        "invariants.MONOMIAL_WARNING = 2\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    invariants.invariants_of_matrices(group, [2], 4)\n"
        "print('WARNED', len(caught) == 1 and 'monomials' in str(caught[0].message))\n"
        "try:\n"
        "    invariants.invariants_of_matrices([np.eye(12)], [12], 8)\n"
        "except ValueError as exc:\n"
        "    print('CAP', 'degree 8 needs 75582 monomials' in str(exc))\n"
    )
    code, out = run_python(probe)
    report("monomial warning via warnings.warn (stdout silent), cap raises ValueError",
           code == 0 and out.startswith("WARNED True") and "CAP True" in out, out)

    # (j) direct sums: the hybrid improper trilinear term of I4/mmm
    # (Ca3Mn2O7 type: X2+ (+) X3- (+) polar GM5-) and of Cmcm Y2- (+) Y4+ with
    # the polar irrep of the HIF_survey/hif_rows.csv row (63, Y4+, Y2-, GM3-)
    code, out = run_group(["--parent", "I4/mmm", "--irrep", "X2+", "X3-", "GM5-",
                           "--invariants", "--degree", "3"])
    report("I4/mmm X2+ X3- GM5-: invariants by multidegree, variables <label>_n",
           code == 0 and "degree (2, 0, 0): 1 invariant" in out
           and "I(2,0,0)_1 = X2+_1^2 + X2+_2^2" in out
           and "degree (1, 1, 0): none" in out
           and "(a; b | c; d | e, f)" in out
           and out.index("degree (2, 0, 0)") < out.index("degree (1, 1, 0)")
           < out.index("degree (0, 0, 2)") < out.index("degree (3, 0, 0)"), out)
    report("I4/mmm X2+ X3- GM5-: (1,1,1) = 1, the lowest-order coupling term (trilinear)",
           "degree (1, 1, 1): 1 invariant" in out
           and "Lowest-order coupling term: degree (1, 1, 1), 1 invariant" in out, out)
    code, out = run_group(["--parent", "63", "--irrep", "Y2-", "Y4+", "GM3-",
                           "--invariants", "--degree", "3"])
    report("Cmcm Y2- Y4+ GM3- (LSNO): trilinear coupling term",
           code == 0 and "Lowest-order coupling term: degree (1, 1, 1), 1 invariant"
           in out, out)
    code, out = run_group(["--parent", "Pm-3m", "--irrep", "R4+", "M3+",
                           "--invariants", "--degree", "3"])
    report("Pm-3m R4+ M3+ to degree 3: no coupling term",
           code == 0 and "No coupling term up to total degree 3" in out, out)
    probe = (
        "from crystod import group\n"
        "t = group.coupling_terms('I4/mmm', ['X2+', 'X3-', 'GM5-'], 3)\n"
        "print('TRI', type(t).__name__, t.multidegree, len(t.polynomials), t.trilinear)\n"
        "t = group.coupling_terms('Pm-3m', ['R4+', 'M3+'], 4)\n"
        "print('BIQ', t.multidegree, len(t.polynomials), t.trilinear)\n"
        "print('NONE', group.coupling_terms('Pm-3m', ['R4+', 'M3+'], 3))\n"
        "try:\n"
        "    group.coupling_terms('Pm-3m', ['R4+'])\n"
        "except ValueError as exc:\n"
        "    print('ONE', 'at least two irreps' in str(exc))\n"
    )
    code, out = run_python(probe)
    report("coupling_terms API: trilinear (1,1,1), biquadratic (2,2) of R4+ M3+, None, "
           "one irrep -> ValueError",
           code == 0 and "TRI CouplingTerm (1, 1, 1) 1 True" in out
           and "BIQ (2, 2) 2 False" in out and "NONE None" in out
           and "ONE True" in out, out)

    # (k) Landau and Lifshitz lines: only in the free-energy modes
    #     (--invariants, --order-parameter, --secondary, --degree); the plain
    #     --irrep enumeration and the --kpoint table carry none
    landau_lines = ("  Landau condition: satisfied (no cubic invariant)\n"
                    "  Lifshitz condition: satisfied (no Lifshitz invariant)\n"
                    "  -> a continuous transition is allowed\n")

    def no_landau(text):
        return ("Landau condition" not in text and "Lifshitz condition" not in text
                and "continuous transition" not in text
                and re.search(r"^note: .*: Landau", text, re.M) is None)

    code, out = run_group(["--parent", "Pm-3m", "--irrep", "R4+"])
    report("--parent Pm-3m --irrep R4+ (plain): no Landau/Lifshitz lines",
           code == 0 and no_landau(out)
           and re.search(r"R4\+\(a,a,a\)\s+167 R-3c\s+2\s+8", out) is not None, out)
    code, out = run_group(["--parent", "Pm-3m", "--irrep", "R4+", "M3+"])
    report("--parent Pm-3m --irrep R4+ M3+ (plain): no Landau/Lifshitz lines",
           code == 0 and no_landau(out) and "* Coupled irreps *" in out, out)
    code, out = run_group(["--parent", "Pm-3m", "--kpoint", "R"])
    report("--parent Pm-3m --kpoint R (plain): table only, no Landau notes",
           code == 0 and no_landau(out)
           and re.search(r"R4\+\(0,0,a\)\s+140 I4/mcm\s+2\s+6", out) is not None, out)
    code, out = run_group(["--parent", "Pm-3m", "--kpoint", "M"])
    report("--parent Pm-3m --kpoint M (plain): table only, no Landau notes",
           code == 0 and no_landau(out) and "M1+(0;0;a)" in out, out)
    code, out = run_group(["--parent", "Pm-3m", "--irrep", "R4+", "--invariants"])
    report("--invariants R4+: Landau and Lifshitz satisfied -> allowed",
           code == 0 and "R4+: order parameter dimension 3\n" + landau_lines in out, out)
    code, out = run_group(["--parent", "Pm-3m", "--irrep", "R4+",
                           "--order-parameter", "a", "0", "0"])
    report("--order-parameter a 0 0 R4+ (no --invariants): the three lines",
           code == 0 and "R4+: order parameter dimension 3\n" + landau_lines in out
           and "R4+(a,0,0) -> I4/mcm (No. 140)" in out, out)
    code, out = run_group(["--parent", "Pm-3m", "--irrep", "R4+", "--order-parameter",
                           "a", "0", "0", "--secondary"])
    report("--secondary R4+ (a,0,0): the three lines",
           code == 0 and "R4+: order parameter dimension 3\n" + landau_lines in out
           and "* Secondary order parameters *" in out, out)
    code, out = run_group(["--parent", "Pm-3m", "--irrep", "M1+",
                           "--order-parameter", "a", "0", "0"])
    report("--order-parameter M1+: Landau violated (1 cubic invariant) -> forbidden",
           code == 0 and "Landau condition: violated (1 cubic invariant)" in out
           and "Lifshitz condition: satisfied (no Lifshitz invariant)" in out
           and "-> a continuous transition is forbidden (Landau)" in out, out)
    # (--invariants --degree 2 skips the slow W5 subgroup enumeration)
    code, out = run_group(["--parent", "Fm-3m", "--irrep", "W5", "--invariants",
                           "--degree", "2"])
    report("--parent Fm-3m --irrep W5: Lifshitz violated -> forbidden (Lifshitz)",
           code == 0 and "Lifshitz condition: violated (1 Lifshitz invariant)" in out
           and "-> a continuous transition is forbidden (Lifshitz)" in out, out)

    # (l) the free energy restricted to an --order-parameter direction
    code, out = run_group(["--parent", "Pm-3m", "--irrep", "R4+", "--invariants",
                           "--order-parameter", "a", "0", "0"])
    report("R4+ (a,0,0): restricted free energy a^2, a^4 [I4/mcm (140)]",
           code == 0 and "Restricted to (a,0,0) [I4/mcm (140)]:\ndegree 2: a^2\n"
           "degree 4: a^4\n" in out, out)
    code, out = run_group(["--parent", "Pm-3m", "--irrep", "R4+", "--invariants",
                           "--order-parameter", "a", "a", "a"])
    report("R4+ (a,a,a): restricted free energy a^2, a^4 [R-3c (167)]",
           code == 0 and "Restricted to (a,a,a) [R-3c (167)]:\ndegree 2: a^2\n"
           "degree 4: a^4\n" in out, out)
    code, out = run_group(["--parent", "Pm-3m", "--irrep", "R4+", "--invariants",
                           "--order-parameter", "a", "b", "0"])
    report("R4+ (a,b,0): degree 4 keeps two independent terms",
           code == 0 and "degree 2: a^2 + b^2\ndegree 4: a^4 + b^4; a^2*b^2" in out, out)

    # (m) secondary order parameters
    code, out = run_group(["--parent", "Pm-3m", "--irrep", "R4+",
                           "--order-parameter", "a", "0", "0", "--secondary"])
    rows = {m.group(1): m.groups() for m in re.finditer(
        r"^(\S+)\s+(\(\S+\))\s+(\d+)\s+(\d+)\s+(\S+)\s+(.+?)\s+(primary|polar|strain|other)$",
        out, re.M)}
    report("R4+ (a,0,0) --secondary: H = I4/mcm (140), index 6 after the subgroup block",
           code == 0 and "* Isotropy subgroup *" in out
           and "* Secondary order parameters *\nH = I4/mcm (140), index 6\n" in out, out)
    report("R4+ (a,0,0): GM1+ and GM3+ strain with n_free 1, R4+ primary, no polar",
           rows.get("GM1+", ())[3:] == ("1", "(a)", "Q^2 eta", "strain")
           and rows.get("GM3+", ())[3:4] == ("1",) and rows["GM3+"][6] == "strain"
           and rows.get("R4+", ())[3:] == ("1", "(a,0,0)", "-", "primary")
           and not any(r[6] == "polar" for r in rows.values()), out)
    code, out = run_group(["--parent", "I4/mmm", "--irrep", "X2+", "X3-",
                           "--order-parameter", "0", "a", "0", "c", "--secondary"])
    rows = {m.group(1): m.groups() for m in re.finditer(
        r"^(\S+)\s+(\(\S+\))\s+(\d+)\s+(\d+)\s+(\S+)\s+(.+?)\s+(primary|polar|strain|other)$",
        out, re.M)}
    report("I4/mmm X2+(0;a) X3-(0;c) -> Cmc2_1: polar GM5- (n_free 1, Q1^1 Q2^1 eta)",
           code == 0 and "H = Cmc2_1 (36), index 8" in out
           and rows.get("GM5-", ())[3:] == ("1", "(a,-a)", "Q1^1 Q2^1 eta", "polar"), out)
    code, out = run_group(["--parent", "Pm-3m", "--irrep", "R4+", "M3+",
                           "--order-parameter", "0", "a", "a", "d", "0", "0",
                           "--secondary"])
    rows = {m.group(1): m.groups() for m in re.finditer(
        r"^(\S+)\s+(\(\S+\))\s+(\d+)\s+(\d+)\s+(\S+)\s+(.+?)\s+(primary|polar|strain|other)$",
        out, re.M)}
    report("Pm-3m R4+(0,a,a) M3+(d;0;0) -> Pnma: X5+ and R5+ secondary, Gamma strains",
           code == 0 and "H = Pnma (62), index 24" in out
           and int(rows.get("X5+", ("",) * 4)[3] or 0) >= 1
           and int(rows.get("R5+", ("",) * 4)[3] or 0) >= 1
           and rows.get("X5+", ())[5:6] == ("Q1^1 Q2^1 eta",)
           and rows.get("GM1+", ())[6:] == ("strain",)
           and [r[6] for r in rows.values()].count("primary") == 2, out)
    probe = (
        "from crystod import group\n"
        "from crystod.secondary_order_parameter import analyze_secondary\n"
        "bad, n = [], 0\n"
        "for sg, labels in (('Pm-3m', 'R4+'), ('Pm-3m', 'M3+'),\n"
        "                   ('I4/mmm', ['X2+', 'X3-'])):\n"
        "    analyzer = group.IsotropyAnalyzer(sg, labels)\n"
        "    for projector, members in analyzer.enumerate_directions():\n"
        "        label, generic = analyzer.direction_label(projector)\n"
        "        if hasattr(analyzer.representation, 'dims') and min(\n"
        "                abs(generic[:2]).max(), abs(generic[2:]).max()) < 1e-8:\n"
        "            continue\n"
        "        n += 1\n"
        "        info, size, index, B, *_ = analyzer.subgroup_of(members)\n"
        "        rows = analyze_secondary(analyzer.algebra, analyzer.representation,\n"
        "                                 members, B, 4).rows\n"
        "        free = analyzer.fixed_space(members).shape[1]\n"
        "        gm1 = [r.n_free for r in rows if r.label == 'GM1+']\n"
        "        primary = sum(r.n_free for r in rows if r.kind == 'primary')\n"
        "        if gm1 != [1] or primary != free:\n"
        "            bad.append((sg, label, gm1, primary, free))\n"
        "print('SANITY', n, bad)\n"
        "rows = group.secondary_order_parameters('Pm-3m', 'R4+', 'a 0 0')\n"
        "print('API', type(rows[0]).__name__, rows[0].label, rows[0].kind,\n"
        "      rows[0].coupling_degree, str(rows[0].k))\n"
    )
    code, out = run_python(probe)
    sanity = re.search(r"SANITY (\d+) \[\]", out)
    report("every R4+, M3+ and coupled X2+ X3- stratum: n_free(GM1+) == 1 and the "
           "primary n_free == dimension of the fixed space",
           code == 0 and sanity is not None and int(sanity.group(1)) >= 14, out)
    report("secondary_order_parameters API returns SecondaryOrderParameter records",
           "API SecondaryOrderParameter GM1+ strain 3 (Fraction(0, 1), Fraction(0, 1), "
           "Fraction(0, 1))" in out, out)
    for args, expect in (
        (["--parent", "Pm-3m", "--irrep", "R4+", "--secondary"],
         "--secondary requires --order-parameter"),
        (["--parent", "Pm-3m", "--kpoint", "R", "--secondary"],
         "not used with --kpoint"),
        (["--table", "--pg", "m-3m", "--secondary"], "only used with --parent"),
    ):
        code, out = run_group(args)
        report(f"{' '.join(args[:4])} ... rejected cleanly",
               code != 0 and expect in out and "Traceback" not in out, out)

    # the identity irrep of a polar parent is a strain, never polar
    code, out = run_group(["--parent", "P6", "--irrep", "M1",
                           "--order-parameter", "a;0;0", "--secondary"])
    gm1 = re.search(r"^GM1 .*$", out, re.M)
    report("P6 M1 (a;0;0) --secondary: GM1 is 'strain', not 'polar'",
           code == 0 and gm1 is not None and gm1.group(0).split()[-1] == "strain"
           and " polar" not in out, out)
    # the monomial cap on the coupling search: a note, no traceback
    probe = (
        "import crystod.invariants as I\n"
        "I.MONOMIAL_LIMIT = 5\n"
        "from crystod.isotropy_subgroup import main\n"
        "main(['--parent', 'Pm-3m', '--irrep', 'R4+', '--order-parameter',\n"
        "      'a', '0', '0', '--secondary'])\n"
    )
    code, out = run_python(probe)
    report("--secondary above the monomial limit: '> N (monomial limit)', no traceback",
           code == 0 and "(monomial limit)" in out and "Traceback" not in out, out)


def test_19_child_lookup() -> None:
    print("\n[19] crystod-group --parent --child (reverse lookup over the cached "
          "isotropy table; fast stratum enumerator)")
    import time

    cache_root = tempfile.mkdtemp(prefix="crystod_isotable_")
    env = dict(os.environ, CRYSTOD_CACHE_DIR=cache_root)

    def group(args):
        return run_module("crystod.cli.group", args, env=env)

    def row(out, pattern):
        return re.search(pattern, out, flags=re.MULTILINE) is not None

    # (a) first run: the Pm-3m table is built (one progress line per k point)
    # and written below $CRYSTOD_CACHE_DIR
    t0 = time.time()
    code, out = group(["--parent", "Pm-3m", "--child", "I4/mcm"])
    first = time.time() - t0
    report("Pm-3m --child I4/mcm exit 0", code == 0 and "Traceback" not in out, out)
    report("--child: Supergroup / Subgroup / Isotropy table / result blocks",
           "* Supergroup *\nPm-3m (No. 221)\n" in out
           and "* Subgroup *\nI4/mcm (No. 140)\n" in out
           and "* Isotropy table *\n40 irreps at the special k points GM, R, X, M;" in out
           and "\n* Isotropy subgroups of type I4/mcm *\n"
               "irrep  k  direction      size  index  conventional basis" in out, out)
    report("--child: progress line per k point on the first run",
           all(f"building the isotropy table of Pm-3m: {k} ({i} of 4) ..." in out
               for i, k in enumerate(["GM", "R", "X", "M"], start=1)), out)
    report("Pm-3m -> I4/mcm: R4+ (0,0,a), size 2, index 6, basis and origin",
           row(out, r"^R4\+\s+R\s+\(0,0,a\)\s+2\s+6\s+\(-1,0,1\),\(1,0,1\),\(0,2,0\)"
                    r"\s+\(0,0,0\)$"), out)
    report("Pm-3m -> I4/mcm: two M5+ embeddings are two rows; 10 rows in all",
           len(re.findall(r"^M5\+\s+M\s", out, flags=re.MULTILINE)) == 2
           and len(re.findall(r"^[A-Z]+\d[+-]?\s+(R|M|X|GM)\s+\(", out,
                              flags=re.MULTILINE)) == 10, out)
    report("--child: ISOSUBGROUP citation", "J. Appl. Cryst. 49, 1849-1853 (2016)." in out,
           out)
    from crystod import __version__ as version
    cache_file = os.path.join(cache_root, "isotropy", f"221_{version}.json.gz")
    report("CRYSTOD_CACHE_DIR respected: <dir>/isotropy/221_<version>.json.gz written",
           os.path.isfile(cache_file) and f"221_{version}.json.gz (written)" in out,
           out + "\n" + str(os.listdir(cache_root)))
    report(f"building the Pm-3m table takes < 60 s ({first:.1f} s)", first < 60.0, out)
    # the R4+ row agrees with the --order-parameter setting of the same direction
    code, ref = group(["--parent", "Pm-3m", "--irrep", "R4+", "--order-parameter",
                       "0", "0", "a"])
    report("R4+ (0,0,a): the setting equals --order-parameter 0 0 a",
           "conventional basis (parent conventional units): (-1,0,1), (1,0,1), (0,2,0)"
           in ref and "origin: (0,0,0)" in ref, ref)

    # (b) second run reads the file
    t0 = time.time()
    code, out = group(["--parent", "Pm-3m", "--child", "R-3c"])
    second = time.time() - t0
    report("second run reads the cache: '(read)', no progress line",
           code == 0 and "(read)" in out and "building the isotropy table" not in out, out)
    report(f"second run well under the first ({second:.2f} s vs {first:.2f} s)",
           second < 0.5 * first, out)
    report("Pm-3m -> R-3c: R4+ (a,a,a), size 2, index 8",
           row(out, r"^R4\+\s+R\s+\(a,a,a\)\s+2\s+8\s"), out)

    # (c) single-irrep results and the coupled cases
    code, out = group(["--parent", "Pm-3m", "--child", "P4mm"])
    report("Pm-3m -> P4mm: GM4- (0,0,a), size 1, index 6, the only row",
           row(out, r"^GM4-\s+GM\s+\(0,0,a\)\s+1\s+6\s")
           and len(re.findall(r"^[A-Z]+\d[+-]?\s+(R|M|X|GM)\s+\(", out,
                              flags=re.MULTILINE)) == 1, out)
    code, out = group(["--parent", "Pm-3m", "--child", "Im-3"])
    report("Pm-3m -> Im-3: M3+ (a;a;a), size 4, index 8 (and M2+)",
           row(out, r"^M3\+\s+M\s+\(a;a;a\)\s+4\s+8\s")
           and row(out, r"^M2\+\s+M\s+\(a;a;a\)\s+4\s+8\s"), out)
    code, out = group(["--parent", "Pm-3m", "--child", "Pnma"])
    report("Pm-3m -> Pnma single irrep: only X5+/X5- (size 8, as ISOSUBGROUP)",
           code == 0 and row(out, r"^X5\+\s+X\s+\(0,a;0,b;c,0\)\s+8\s+48\s")
           and row(out, r"^X5-\s+X\s") and "R4+" not in out, out)
    code, out = group(["--parent", "Pm-3m", "--child", "62", "--size", "4"])
    report("Pm-3m -> Pnma size 4 (the R4+ + M3+ tilts): coupled-irreps hint, exit 0",
           code == 0 and "no single-irrep stratum of type Pnma (No. 62) with size 4;\n"
           "the subgroup may need two coupled irreps (--coupled)\n" in out
           and "* Coupled isotropy subgroups" not in out, out)
    code, out = group(["--parent", "I4/mmm", "--child", "Cmc2_1"])
    report("I4/mmm -> Cmc2_1 (36): no single-irrep stratum, exit 0",
           code == 0 and "no single-irrep stratum of type Cmc2_1 (No. 36);\n"
           "the subgroup may need two coupled irreps" in out
           and "37 irreps at the special k points GM, X, M, N, P;" in out, out)

    # (d) filters
    code, out = group(["--parent", "Pm-3m", "--child", "140", "--index", "12"])
    report("--index 12: the six M rows only, 'selected:' line",
           "selected: index 12" in out and "R4+" not in out
           and len(re.findall(r"^M\S+\s+M\s+\(", out, flags=re.MULTILINE)) == 6, out)
    code, out = group(["--parent", "Pm-3m", "--child", "140", "--size", "2",
                       "--kpoint", "R"])
    report("--size 2 --kpoint R: the four R rows",
           "selected: size 2, k point R" in out
           and len(re.findall(r"^R\S+\s+R\s+\(", out, flags=re.MULTILINE)) == 4, out)
    code, out = group(["--parent", "Pm-3m", "--child", "140", "--kpoint",
                       "1/2", "-1/2", "1/2"])
    report("--kpoint as coordinates of another arm (negative fraction)",
           code == 0 and "selected: k point R" in out, out)
    code, out = group(["--parent", "Pm-3m", "--child", "P-1", "--size", "2"])
    report("--child P-1 parses; R4+ (a,b,c) size 2 index 48",
           code == 0 and "P-1 (No. 2)" in out
           and row(out, r"^R4\+\s+R\s+\(a,b,c\)\s+2\s+48\s"), out)

    # (e) enantiomorphic partner (SUBGROUP/VALIDATION.md: Z3 of P4_2/mcm)
    code, out = group(["--parent", "P4_2/mcm", "--child", "P4_322"])
    report("P4_2/mcm -> P4_322: Z3 and Z4 (0,a), note on the partner P4_122",
           code == 0 and row(out, r"^Z3\s+Z\s+\(0,a\)\s+2\s+4\s")
           and row(out, r"^Z4\s+Z\s+\(0,a\)\s+2\s+4\s")
           and "note: 95 <-> 91 (P4_322 <-> P4_122) are enantiomorphic partner types" in out,
           out)
    # in a Sohncke parent the partner types are different strata: no merging
    code, out = group(["--parent", "93", "--child", "P4_122"])
    report("Sohncke P4_222 -> P4_122: Z1 and Z3 only, no partner note",
           code == 0 and row(out, r"^Z1\s+Z\s") and row(out, r"^Z3\s+Z\s")
           and not row(out, r"^Z2\s") and not row(out, r"^Z4\s")
           and "enantiomorphic" not in out, out)
    code, out = group(["--parent", "144", "--child", "P3_2"])
    report("Sohncke P3_1 -> P3_2: no GM1 row (the parent itself is P3_1)",
           code == 0 and not row(out, r"^GM1\s") and "enantiomorphic" not in out, out)
    # a pair spanning the k and the -k star is listed once, at the first star
    h_rows = re.findall(r"^H\d+HA\d+\s+(\S+)\s", out, flags=re.MULTILINE)
    report("P3_1 -> P3_2: H1HA1.. listed once, at H (not again at HA); 12 irreps",
           code == 0 and len(h_rows) == 3 and set(h_rows) == {"H"}
           and len(set(re.findall(r"^(H\d+HA\d+)\s", out, flags=re.MULTILINE))) == 3
           and "12 irreps at the special k points GM, A, H, K, L, M, HA, KA;" in out, out)
    code, out = group(["--parent", "144", "--child", "P3_2", "--kpoint", "HA"])
    report("--kpoint HA selects the rows listed at H",
           code == 0 and "selected: k point H" in out
           and row(out, r"^H\d+HA\d+\s+H\s"), out)

    # (f) --no-cache rebuilds; a damaged file is rebuilt
    code, out = group(["--parent", "Pm-3m", "--child", "140", "--no-cache"])
    report("--no-cache rebuilds and rewrites",
           code == 0 and "building the isotropy table of Pm-3m: GM" in out
           and "(written)" in out, out)
    with open(cache_file, "wb") as handle:
        handle.write(b"not a gzip file")
    code, out = group(["--parent", "Pm-3m", "--child", "140"])
    report("damaged cache file: rebuilt, no traceback",
           code == 0 and "(written)" in out and "Traceback" not in out, out)

    # (g) error paths: one line, no traceback
    code, out = group(["--child", "I4/mcm"])
    report("--child without --parent: one-line error",
           code != 0 and out.strip() == "ERROR: --child needs --parent SG (the parent "
           "space group), e.g. --parent Pm-3m --child I4/mcm.", out)
    code, out = group(["--parent", "Pm-3m", "--child", "Xyz"])
    report("unknown child symbol: one-line error",
           code != 0 and len(out.strip().splitlines()) == 1
           and out.startswith('ERROR: "Xyz" is not recognized') , out)
    code, out = group(["--parent", "Pm-3m", "--child", "I4/mcm", "--secondary"])
    report("--child --secondary without --coupled is rejected",
           code != 0 and "(--secondary needs --coupled)" in out and "Traceback" not in out,
           out)
    code, out = group(["--parent", "Pm-3m", "--child", "I4/mcm", "--kpoint", "Q"])
    report("unknown k point with --child: one-line error",
           code != 0 and len(out.strip().splitlines()) == 1
           and "special points: GM, R, X, M." in out, out)
    code, out = group(["--parent", "Pm-3m", "--irrep", "R4+", "--size", "2"])
    report("--size without --child is rejected",
           code != 0 and "--size is only used with --parent SG --child H." in out
           and "Traceback" not in out, out)
    code, out = group(["--parent", "Pm-3m", "--child", "140", "--irrep", "R4+"])
    report("--child with --irrep is rejected",
           code != 0 and "--irrep is not used with it" in out and "Traceback" not in out,
           out)

    # (h) Python API and agreement with the --kpoint tables
    probe = (
        "import numpy as np\n"
        "from crystod import group\n"
        "m = group.find_isotropy_irreps('Pm-3m', 'I4/mcm', size=2, kpoints=['R'])\n"
        "print('MATCH', [(x.label, x.kname, x.direction, x.number, x.size, x.index)\n"
        "                for x in m])\n"
        "print('FIELDS', type(m[0]).__name__, m[0].symbol, m[0].basis.tolist(),\n"
        "      m[0].origin.tolist())\n"
        "print('ENANT', [(x.label, x.number) for x in\n"
        "                group.find_isotropy_irreps('P4_2/mcm', 95)])\n"
        "table = group.isotropy_table('Pm-3m')\n"
        "print('TABLE', type(table).__name__, table.from_cache, len(table.irreps),\n"
        "      len(table.strata))\n"
        "ok = True\n"
        "for k in table.kpoints:\n"
        "    ref = group.isotropy_subgroups_at_kpoint('Pm-3m', k)\n"
        "    for label, subs in ref.items():\n"
        "        mine = [s for s in table.strata if s.label == label and s.kname == k]\n"
        "        ok = ok and [(s.direction, s.number, s.size, s.index) for s in mine] == [\n"
        "            (s.direction, s.number, s.size, s.index) for s in subs]\n"
        "        ok = ok and all(np.allclose(a.basis, b.basis) and\n"
        "                        np.allclose(a.origin, b.origin) for a, b in zip(mine, subs))\n"
        "print('AGREE', ok)\n"
        "for bad in (('Pm-3m', 'Xyz'), ('Pm-3m', 999), ('Q', 'P1')):\n"
        "    try:\n"
        "        group.find_isotropy_irreps(*bad)\n"
        "        print('NOERR', bad)\n"
        "    except ValueError as exc:\n"
        "        print('VALUEERROR', bad[1])\n"
    )
    code, out = run_python(probe, env=env)
    report("API find_isotropy_irreps: the four R strata of I4/mcm with size 2",
           "MATCH [('R3+', 'R', '(0,a)', 140, 2, 6), ('R4+', 'R', '(0,0,a)', 140, 2, 6), "
           "('R3-', 'R', '(a,0)', 140, 2, 6), ('R5-', 'R', '(0,0,a)', 140, 2, 6)]" in out,
           out)
    report("API IsotropyMatch fields (symbol, basis, origin)",
           "FIELDS IsotropyMatch I4/mcm [[0.0, -1.0, 1.0], [0.0, 1.0, 1.0], "
           "[-2.0, 0.0, 0.0]] [0.0, -0.5, -0.5]" in out, out)
    report("API: the enantiomorphic partner rows are matches",
           "ENANT [('Z3', 91), ('Z4', 91)]" in out, out)
    report("API isotropy_table: cached IsotropyTable, 40 irreps",
           "TABLE IsotropyTable True 40" in out, out)
    report("API table rows == isotropy_subgroups_at_kpoint (direction, type, settings)",
           "AGREE True" in out, out)
    report("API: bad child / parent -> ValueError",
           out.count("VALUEERROR") == 3 and "NOERR" not in out, out)

    # (i) the fast enumerator equals the original pairwise closure (order,
    # representatives and stabilizers)
    probe = (
        "from crystod import group\n"
        "from crystod.isotropy_subgroup import _projector_key as key\n"
        "same = []\n"
        "for sg, labels in (('Pm-3m', 'R4+'), ('Pm-3m', 'X5+'), ('Im-3m', 'P5'),\n"
        "                   ('P-43n', 'R4'), ('Pm-3m', ['R4+', 'M3+']),\n"
        "                   ('I4/mmm', ['X3-', 'X2+'])):\n"
        "    an = group.IsotropyAnalyzer(sg, labels)\n"
        "    fast = an.enumerate_directions()\n"
        "    legacy = an.enumerate_directions(method='legacy')\n"
        "    same.append(len(fast) == len(legacy) and all(\n"
        "        key(a) == key(b) and [(i, tuple(t)) for i, t in ma] ==\n"
        "        [(i, tuple(t)) for i, t in mb] for (a, ma), (b, mb) in zip(fast, legacy)))\n"
        "print('SAME', same)\n"
        "try:\n"
        "    an.enumerate_directions(method='other')\n"
        "except ValueError as exc:\n"
        "    print('METHOD', exc)\n"
    )
    code, out = run_python(probe, env=env)
    report("enumerate_directions: fast == legacy for 6 representations",
           "SAME [True, True, True, True, True, True]" in out, out)
    report("enumerate_directions(method='other') -> ValueError",
           'METHOD unknown method "other"' in out, out)

    # (j) --coupled: pairs of irreps (same k point first, then different k)
    t0 = time.time()
    code, out = group(["--parent", "Pm-3m", "--child", "Pnma", "--size", "4", "--coupled"])
    elapsed = time.time() - t0
    report("Pm-3m --child Pnma --size 4 --coupled: exit 0, coupled block",
           code == 0 and "Traceback" not in out
           and "the subgroup may need two coupled irreps (see below)" in out
           and "\n* Coupled isotropy subgroups of type Pnma *\n"
               "40 irreps pass the point-group filter; 780 pairs searched\n"
               "(180 at the same k point, 600 at different k points)\n"
               "irrep1  dir1           irrep2  dir2           size  index  "
               "conventional basis" in out, out)
    report("Pm-3m -> Pnma (size 4): the tilt system R4+ (0,a,a) + M3+ (d;0;0), "
           "size 4, index 24, setting",
           row(out, r"^R4\+\s+\(0,a,a\)\s+M3\+\s+\(d;0;0\)\s+4\s+24\s+"
                    r"\(-1,-1,0\),\(0,0,-2\),\(1,-1,0\)\s+\(0,0,0\)$"), out)
    report("Pm-3m -> Pnma (size 4): 32 coupled rows, all index 24; both irreps needed",
           len(re.findall(r"^[RXM]\d[+-]\s+\(\S+\)\s+[XM]\d[+-]\s+\(\S+\)\s+4\s+24\s", out,
                          flags=re.MULTILINE)) == 32
           and "every row needs both irreps: neither irrep alone has this isotropy "
               "subgroup" in out, out)
    report(f"Pm-3m --child Pnma --size 4 --coupled takes < 60 s ({elapsed:.1f} s)",
           elapsed < 60.0, out)
    code, out = group(["--parent", "I4/mmm", "--child", "Cmc2_1", "--coupled"])
    report("I4/mmm -> Cmc2_1 (A2_1am): X2+ (0;a) + X3- (0;c), size 2, index 8",
           code == 0 and row(out, r"^X2\+\s+\(0;a\)\s+X3-\s+\(0;c\)\s+2\s+8\s"), out)
    code, out = group(["--parent", "Cmcm", "--child", "Pna2_1", "--coupled"])
    report("Cmcm -> Pna2_1: the pair Y2- + Y4+ (listed Y4+ (a), Y2- (b)), size 2, index 4",
           code == 0 and row(out, r"^Y4\+\s+\(a\)\s+Y2-\s+\(b\)\s+2\s+4\s"), out)
    code, out = group(["--parent", "Pm-3m", "--child", "Pnma", "--size", "4", "--coupled",
                       "--kpoint", "R"])
    report("--coupled --kpoint R: same-k pairs only, none gives Pnma (size 4)",
           code == 0 and "(45 at the same k point, 0 at different k points)" in out
           and "no coupled stratum of type Pnma (No. 62) with size 4, k point R\n" in out,
           out)
    code, out = group(["--parent", "Pm-3m", "--child", "Pnma", "--size", "4", "--coupled",
                       "--kpoint", "R", "M", "--secondary"])
    report("--coupled --secondary: secondary order parameters of every answer "
           "(R4+ + M3+: X5+, R5+, M2+)",
           code == 0 and out.count("* Secondary order parameters *") == 16
           and re.search(r"H = Pnma \(62\), index 24: R4\+\(0,a,a\) M3\+\(d;0;0\)\n"
                         r"(.*\n){1,12}?X5\+\s+\(0,1/2,0\)\s+6\s+1\s+\(0,0;a,a;0,0\)", out)
           is not None, out)
    probe = (
        "from crystod import group\n"
        "m = group.find_coupled_isotropy_irreps('Pm-3m', 'Pnma', size=4, kpoints=['R', 'M'])\n"
        "print('COUPLED', len(m), type(m[0]).__name__)\n"
        "x = [r for r in m if r.direction == 'R4+(0,a,a) M3+(d;0;0)'][0]\n"
        "print('ROW', x.label1, x.kname1, x.direction1, x.label2, x.kname2, x.direction2,\n"
        "      x.number, x.size, x.index, x.same_k, x.basis.tolist(), x.vector.shape)\n"
        "print('STR', x)\n"
        "both = group.find_isotropy_irreps('Pm-3m', 'Pnma', coupled=True, kpoints=['X', 'M'])\n"
        "print('BOTH', sorted({type(r).__name__ for r in both}))\n"
    )
    code, out = run_python(probe, env=env)
    report("API find_coupled_isotropy_irreps: CoupledIsotropyMatch records",
           "COUPLED 16 CoupledIsotropyMatch" in out
           and "ROW R4+ R (0,a,a) M3+ M (d;0;0) 62 4 24 False "
               "[[-1.0, -1.0, 0.0], [0.0, 0.0, -2.0], [1.0, -1.0, 0.0]] (6,)" in out
           and "STR R4+(0,a,a) M3+(d;0;0) -> Pnma (No. 62), size 4, index 24" in out, out)
    report("API find_isotropy_irreps(coupled=True): single and coupled records",
           "BOTH ['CoupledIsotropyMatch', 'IsotropyMatch']" in out, out)

    # (k) 50 seeded rows of the HIF survey (HIF_survey/hif_rows.csv, or
    # $CRYSTOD_HIF_ROWS): the coupled lookup of (parent, G12) at the row's k
    # point contains the row's irrep pair with its stratum (same subgroup,
    # index and either the row's direction string verbatim or the isotropy
    # subspace of the row's direction up to the parent); STALE (a direction
    # that does not give G12) must not occur for seed 2026
    hif_csv = os.environ.get("CRYSTOD_HIF_ROWS") or os.path.join(
        ROOT, "HIF_survey", "hif_rows.csv")
    if not os.path.isfile(hif_csv):
        print(f"  [SKIP] HIF survey sample ({hif_csv} not found; set CRYSTOD_HIF_ROWS)")
    else:
        import csv
        import random

        with open(hif_csv, newline="") as handle:
            hif_rows = list(csv.DictReader(handle))
        sample = random.Random(2026).sample(hif_rows, 50)
        by_parent: dict = {}
        for hif_row in sample:
            by_parent.setdefault(hif_row["parent_number"], []).append(hif_row)
        hif_probe = (
            "import contextlib, io, json, sys\n"
            "import numpy as np\n"
            "from crystod.isotropy_table import _isotropy_table, coupled_search, "
            "resolve_kpoints\n"
            "from crystod.isotropy_subgroup import (CoupledRepresentation, IsotropyAnalyzer,\n"
            "    _SCALED_PARAMETER, _order_parameter_tokens, _projector, _rep_cache,\n"
            "    kpoint_irrep_tables)\n"
            "from crystod.spacegroup_product import SpaceGroupIrrepAlgebra\n"
            "rows = json.loads(sys.argv[1])\n"
            "quiet = lambda: contextlib.redirect_stdout(io.StringIO())\n"
            "with quiet():\n"
            "    table = _isotropy_table(rows[0]['parent_number'], progress=None)\n"
            "    algebra = SpaceGroupIrrepAlgebra(rows[0]['parent_number'])\n"
            "reps, searches = {}, {}\n"
            "def vector(analyzer, tokens):\n"
            "    try:\n"
            "        with quiet():\n"
            "            return analyzer.resolve_direction(tokens)\n"
            "    except SystemExit:  # rounded decimal factors: taken at face value\n"
            "        rng, names, values = np.random.default_rng(3), {}, []\n"
            "        for token in tokens:\n"
            "            sign = -1.0 if token.startswith('-') else 1.0\n"
            "            body = token.lstrip('-')\n"
            "            if body in ('0', ''):\n"
            "                values.append(0.0)\n"
            "                continue\n"
            "            factor, name = 1.0, body\n"
            "            scaled = _SCALED_PARAMETER.fullmatch(body)\n"
            "            if scaled is not None:\n"
            "                factor, name = float(scaled.group(1)), scaled.group(2)\n"
            "            names.setdefault(name, float(rng.uniform(0.5, 1.5)))\n"
            "            values.append(sign * factor * names[name])\n"
            "        return np.array(values)\n"
            "for r in rows:\n"
            "    tag = (f\"{r['parent_number']} {r['kpoint']} {r['irrep1']}{r['dir1']} \"\n"
            "           f\"{r['irrep2']}{r['dir2']} -> {r['G12_number']}\")\n"
            "    k = r['kpoint']\n"
            "    if k not in reps:\n"
            "        with quiet():\n"
            "            reps[k] = kpoint_irrep_tables(algebra, k, lambda a: a.representation)[0]\n"
            "    rep1, rep2 = reps[k].get(r['irrep1']), reps[k].get(r['irrep2'])\n"
            "    if rep1 is None or rep2 is None:\n"
            "        print('NOREP', tag)\n"
            "        continue\n"
            "    g12 = int(r['G12_number'])\n"
            "    if (g12, k) not in searches:\n"
            "        searches[(g12, k)] = coupled_search(\n"
            "            table, g12, knames=resolve_kpoints(table, [k])).matches\n"
            "    row_parts = {(r['irrep1'], r['dir1']), (r['irrep2'], r['dir2'])}\n"
            "    if any({(m.label1, m.direction1), (m.label2, m.direction2)} == row_parts\n"
            "           and m.number == g12 and m.index == int(r['index'])\n"
            "           for m in searches[(g12, k)]):\n"
            "        print('FOUND', tag)  # the row's direction string verbatim\n"
            "        continue\n"
            "    coupled = CoupledRepresentation.from_parts(algebra, [rep1, rep2])\n"
            "    analyzer = IsotropyAnalyzer.from_representation(algebra, coupled)\n"
            "    eta = vector(analyzer, _order_parameter_tokens([r['dir1']])\n"
            "                 + _order_parameter_tokens([r['dir2']]))\n"
            "    members = [(i, t) for i, t, m in analyzer.elements\n"
            "               if np.allclose(m @ eta, eta, atol=1e-6)]\n"
            "    with quiet():\n"
            "        info = analyzer.subgroup_of(members)[0]\n"
            "    if int(info.number) != g12:\n"
            "        print('STALE', tag)\n"
            "        continue\n"
            "    fixed = _projector(analyzer.fixed_space(members))\n"
            "    found = [m for m in searches[(g12, k)]\n"
            "             if {m.label1, m.label2} == {r['irrep1'], r['irrep2']}\n"
            "             and m.number == g12 and m.index == int(r['index'])]\n"
            "    same = False\n"
            "    for m in found:\n"
            "        target, order = fixed, [rep1, rep2]\n"
            "        if m.label1 != r['irrep1']:\n"
            "            n1 = rep1.dimension\n"
            "            perm = list(range(n1, coupled.dimension)) + list(range(n1))\n"
            "            target, order = fixed[np.ix_(perm, perm)], [rep2, rep1]\n"
            "        cache = _rep_cache(CoupledRepresentation.from_parts(algebra, order))\n"
            "        images = np.matmul(np.matmul(cache.Ed, m.projector),\n"
            "                           cache.Ed.transpose(0, 2, 1))\n"
            "        same = same or bool(np.any(\n"
            "            np.max(np.abs(images - target[None]), axis=(1, 2)) < 1e-6))\n"
            "    print('FOUND' if found and same else 'MISSING', tag)\n"
        )
        t0 = time.time()
        processes = []
        # the large face-centred cubic parents first; at most 8 at a time
        for parent_number in sorted(by_parent, key=lambda p: -int(p)):
            while sum(p.poll() is None for p in processes) >= 8:
                time.sleep(0.2)
            processes.append(subprocess.Popen(
                [sys.executable, "-c", hif_probe, json.dumps(by_parent[parent_number])],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT,
                env=env,
            ))
        hif_out = ""
        for process in processes:
            try:
                hif_out += process.communicate(timeout=TIMEOUT_SECONDS)[0]
            except subprocess.TimeoutExpired:
                process.kill()
                hif_out += "TIMEOUT\n"
        hif_elapsed = time.time() - t0
        n_found = hif_out.count("FOUND ")
        n_stale = hif_out.count("STALE ")
        report(f"HIF survey, 50 seeded rows: the coupled lookup contains the row's pair "
               f"and stratum ({n_found} found, {n_stale} STALE)",
               n_found == 50 and n_stale == 0 and "MISSING" not in hif_out
               and "Traceback" not in hif_out, hif_out)
        report(f"HIF survey sample takes < 120 s ({hif_elapsed:.1f} s)", hif_elapsed < 120.0,
               hif_out)
    shutil.rmtree(cache_root, ignore_errors=True)


def test_20_subgroup_graph() -> None:
    print("\n[20] crystod-group --parent --irrep --graph (group-subgroup graph of the "
          "isotropy subgroups)")
    workdir = tempfile.mkdtemp(prefix="crystod_graph_")

    def group(args):
        return run_module("crystod.cli.group", args, cwd=workdir)

    # (a) the perovskite tilt systems, Howard and Stokes, Acta Cryst. B54, 782 (1998)
    code, out = group(["--parent", "Pm-3m", "--irrep", "R4+", "M3+", "--graph", "--graph-dot"])
    html_path = os.path.join(workdir, "SUBGROUP_Pm-3m_R4+_M3+.html")
    dot_path = os.path.join(workdir, "SUBGROUP_Pm-3m_R4+_M3+.dot")
    report("--parent Pm-3m --irrep R4+ M3+ --graph --graph-dot: exit 0, blocks",
           code == 0 and "Traceback" not in out
           and "\n* Coupled irreps *\nR4+: order parameter dimension 3\n" in out
           and "\n* Group-subgroup graph *\n25 nodes (the parent and 24 strata), "
               "56 edges (0 dashed)\n  index 1: Pm-3m (221)\n"
               "  index 6: P4/mbm (127), I4/mcm (140)\n" in out
           and "\n* Nodes *\nnode  subgroup        size  index  direction" in out
           and "\n* Output files *\n  SUBGROUP_Pm-3m_R4+_M3+.html\n"
               "  SUBGROUP_Pm-3m_R4+_M3+.dot\n" in out
           and "P4_2/nmc (137),\n    Immm (71)\n" in out  # labels are never split
           and "Conventions and validation: ISOSUBGROUP" in out
           and "Octahedral tilt systems of the perovskites" in out, out)
    report("--graph: node rows (Pnma below P4/mbm and Imma; the kernel P-1 at index 192)",
           re.search(r"^11\s+Pnma \(62\)\s+4\s+24\s+R4\+\(0,a,a\) M3\+\(d;0;0\)\s+2,6\s+-$",
                     out, flags=re.MULTILINE) is not None
           and re.search(r"^25\s+P-1 \(2\)\s+8\s+192\s+R4\+\(a,b,c\) M3\+\(d;e;f\)", out,
                         flags=re.MULTILINE) is not None, out)
    html = open(html_path, encoding="utf-8").read() if os.path.isfile(html_path) else ""
    report("--graph: the HTML has the inline SVG, the node labels and the hover handler",
           "<svg id=\"graph\"" in html and ">I4/mcm (140)</text>" in html
           and ">Pnma (62)</text>" in html and "R4+(0,a,a) M3+(d;0;0)" in html
           and 'data-basis="(-1,-1,0),(0,0,-2),(1,-1,0)"' in html
           and 'addEventListener("mouseenter"' in html and "stroke-dasharray" in html
           and "<script src" not in html and len(html) < 100000, html[-2000:])
    dot = open(dot_path, encoding="utf-8").read() if os.path.isfile(dot_path) else ""
    report("--graph-dot: a Graphviz digraph with 56 edges and the layers",
           dot.startswith('digraph "SUBGROUP_Pm-3m_R4+_M3+" {')
           and dot.count(" -> ") == 56 and "rank=same" in dot
           and 'n11 [label="Pnma (62)\\nR4+(0,a,a) M3+(d;0;0)"];' in dot, dot)

    # the 15 tilt systems and their group-subgroup lines.  Edge list derived
    # from the Glazer tilt patterns of Table 1 of Howard and Stokes (1998)
    # (a pattern lies below another when it specializes to it by setting
    # tilts equal or zero, up to a cubic permutation), which is what their
    # Fig. 1 draws; the graph additionally has the strata in which a tilt
    # axis carries both an in-phase and an out-of-phase tilt (R-3, P2_1/c,
    # Pmmn, and second embeddings of P4/mbm, Cmcm, C2/m, C2/c, P2_1/m, P-1),
    # which Howard and Stokes leave out
    probe = (
        "from crystod import group\n"
        "from crystod.subgroup_graph import _transitive_reduction\n"
        "g = group.subgroup_graph('Pm-3m', ['R4+', 'M3+'])\n"
        "print('TYPES', sorted({n.symbol for n in g.nodes}))\n"
        "hs = {'Pm-3m': '', 'Im-3': 'M3+(d;d;d)', 'Immm': 'M3+(d;e;f)',\n"
        "      'P4/mbm': 'M3+(0;0;d)', 'I4/mmm': 'M3+(0;d;d)', 'I4/mcm': 'R4+(0,0,a)',\n"
        "      'Imma': 'R4+(0,a,a)', 'R-3c': 'R4+(a,a,a)', 'C2/m': 'R4+(0,a,b)',\n"
        "      'C2/c': 'R4+(a,a,b)', 'P-1': 'R4+(a,b,c)',\n"
        "      'P4_2/nmc': 'R4+(0,0,a) M3+(d;d;0)', 'Cmcm': 'R4+(0,0,a) M3+(0;d;0)',\n"
        "      'Pnma': 'R4+(0,a,a) M3+(d;0;0)', 'P2_1/m': 'R4+(0,a,b) M3+(d;0;0)'}\n"
        "ids = {}\n"
        "for symbol, direction in hs.items():\n"
        "    found = [n.id for n in g.nodes if n.symbol == symbol and n.direction == direction]\n"
        "    if len(found) == 1:\n"
        "        ids[symbol] = found[0]\n"
        "print('HSNODES', len(ids))\n"
        "names = {v: k for k, v in ids.items()}\n"
        "order = list(ids.values())\n"
        "relation = {(order.index(a), order.index(b)) for a, b in g.inclusions\n"
        "            if a in names and b in names}\n"
        "edges = sorted(f'{names[order[a]]}>{names[order[b]]}'\n"
        "               for a, b in _transitive_reduction(len(order), relation))\n"
        "print('HSEDGES', ' '.join(edges))\n"
        "parent_edges = [e for e in g.edges if e.parent == 0]\n"
        "print('PARENT', len(parent_edges), any(e.dashed for e in g.edges))\n"
        "print('KINDS', g.nodes[0].kind, g.nodes[-1].kind, g.nodes[-1].label)\n"
        "g1 = group.subgroup_graph('Pm-3m', 'R4+')\n"
        "print('R4NODES', [n.label for n in g1.nodes])\n"
        "print('R4EDGES', ' '.join(sorted(f'{g1.nodes[e.parent].symbol}>'\n"
        "                                 f'{g1.nodes[e.child].symbol}' for e in g1.edges)))\n"
        "g2 = group.subgroup_graph('Pm-3m', 'M1+')\n"
        "print('M1', g2.landau['M1+'].n_cubic, sorted((g2.nodes[e.parent].kind, e.dashed)\n"
        "                                            for e in g2.edges))\n"
        "try:\n"
        "    group.subgroup_graph('Pm-3m', 'Q9')\n"
        "except ValueError:\n"
        "    print('VALUEERROR')\n"
    )
    code, out = run_python(probe)
    report("R4+ M3+: the node types are the 15 tilt systems of Howard and Stokes plus R-3, "
           "P2_1/c and Pmmn",
           "TYPES ['C2/c', 'C2/m', 'Cmcm', 'I4/mcm', 'I4/mmm', 'Im-3', 'Imma', 'Immm', "
           "'P-1', 'P2_1/c', 'P2_1/m', 'P4/mbm', 'P4_2/nmc', 'Pm-3m', 'Pmmn', 'Pnma', "
           "'R-3', 'R-3c']" in out and "HSNODES 15" in out, out)
    hs_edges = sorted(
        "Pm-3m>Im-3 Pm-3m>I4/mmm Pm-3m>P4/mbm Pm-3m>I4/mcm Pm-3m>Imma Pm-3m>R-3c "
        "Im-3>Immm I4/mmm>Immm I4/mmm>P4_2/nmc P4/mbm>Immm P4/mbm>Cmcm P4/mbm>Pnma "
        "I4/mcm>P4_2/nmc I4/mcm>Cmcm I4/mcm>C2/m I4/mcm>C2/c Imma>Pnma Imma>C2/m "
        "Imma>C2/c R-3c>C2/c Cmcm>P2_1/m Pnma>P2_1/m C2/m>P2_1/m C2/m>P-1 "
        "C2/c>P-1".split())
    report("R4+ M3+: the group-subgroup lines among the 15 tilt systems = the 25 lines "
           "derived from the Howard-Stokes Glazer patterns",
           "HSEDGES " + " ".join(hs_edges) in out, out)
    report("R4+ M3+: six solid lines from the parent (no cubic or Lifshitz invariant); "
           "parent first, kernel last",
           "PARENT 6 False" in out and "KINDS parent kernel P-1 (2)" in out, out)
    report("R4+ alone: Pm-3m, I4/mcm, R-3c, Imma, C2/m, C2/c, P-1 with the 10 lines of "
           "the figure",
           "R4NODES ['Pm-3m (221)', 'I4/mcm (140)', 'R-3c (167)', 'Imma (74)', "
           "'C2/m (12)', 'C2/c (15)', 'P-1 (2)']" in out
           and "R4EDGES " + " ".join(sorted(
               "Pm-3m>I4/mcm Pm-3m>R-3c Pm-3m>Imma I4/mcm>C2/m I4/mcm>C2/c R-3c>C2/c "
               "Imma>C2/m Imma>C2/c C2/m>P-1 C2/c>P-1".split())) in out, out)
    report("M1+ (cubic invariant Q1 Q2 Q3): dashed lines from the parent, solid between "
           "strata",
           "M1 1 [('parent', True), ('parent', True), ('stratum', False), "
           "('stratum', False), ('stratum', False)]" in out, out)
    report("API subgroup_graph: bad irrep -> ValueError", "VALUEERROR" in out, out)

    # (b) single irrep, --output, the terminal block of a dashed graph
    code, out = group(["--parent", "Pm-3m", "--irrep", "M1+", "--graph", "--output",
                       "graphs/m1.html"])
    report("--graph --output DIR/FILE: written there; dashed lines reported",
           code == 0 and os.path.isfile(os.path.join(workdir, "graphs", "m1.html"))
           and "5 nodes (the parent and 4 strata), 5 edges (2 dashed)" in out
           and "Octahedral tilt systems" not in out
           and re.search(r"^3\s+P4/mmm \(123\)\s+2\s+6\s+M1\+\(0;0;a\)\s+1\s+dashed$", out,
                         flags=re.MULTILINE) is not None
           and "* Output files *\n  graphs/m1.html\n" in out, out)
    html = open(os.path.join(workdir, "graphs", "m1.html"), encoding="utf-8").read() \
        if os.path.isfile(os.path.join(workdir, "graphs", "m1.html")) else ""
    report("--graph: dashed lines drawn dashed in the SVG",
           html.count('class="edge dashed"') == 2, html[:500])

    # (c) error paths
    code, out = group(["--parent", "Pm-3m", "--irrep", "R4+", "--graph-dot"])
    report("--graph-dot without --graph is rejected",
           code != 0 and "--graph-dot is only used with --parent SG --irrep IR [IR2 ...] "
           "--graph." in out and "Traceback" not in out, out)
    code, out = group(["--parent", "Pm-3m", "--kpoint", "R", "--graph"])
    report("--graph with --kpoint is rejected",
           code != 0 and "--kpoint is not used with it" in out and "Traceback" not in out, out)
    code, out = group(["--parent", "Pm-3m", "--child", "Pnma", "--graph"])
    report("--graph with --child is rejected",
           code != 0 and "--graph is not used with it" in out and "Traceback" not in out, out)
    code, out = group(["--table", "--pg", "m-3m", "--graph"])
    report("--graph without --parent is rejected",
           code != 0 and "--graph/--graph-dot are only used with --parent SG --irrep IR" in out
           and "Traceback" not in out, out)
    shutil.rmtree(workdir, ignore_errors=True)


def test_21_multiplet() -> None:
    print("\n[21] crystod-group --multiplet (multi-electron terms)")

    # single shells in Oh (textbook Tanabe-Sugano/Griffith terms; sorted by
    # descending spin multiplicity, so the Hund ground term comes first)
    code, out = run_group(["--multiplet", "T2g^2", "--pg", "m-3m"])
    report("(t2g)^2 = ^3T1g + ^1A1g + ^1Eg + ^1T2g",
           code == 0 and "* Term Symbols *" in out
           and "(T2g)^2 = ^3T1g + ^1A1g + ^1Eg + ^1T2g" in out, out)
    report("(t2g)^2 state count C(6,2) = 15",
           "check: 15 states = C(6,2) = 15" in out, out)

    code, out = run_group(["--multiplet", "Eg2", "--pg", "m-3m"])
    report("(eg)^2 = ^3A2g + ^1A1g + ^1Eg (quoting-free Eg2 token)",
           code == 0 and "(Eg)^2 = ^3A2g + ^1A1g + ^1Eg" in out, out)

    code, out = run_group(["--multiplet", "T2g^3", "--pg", "m-3m"])
    report("(t2g)^3 = ^4A2g + ^2Eg + ^2T1g + ^2T2g",
           code == 0 and "(T2g)^3 = ^4A2g + ^2Eg + ^2T1g + ^2T2g" in out
           and "20 states" in out, out)

    code, out = run_group(["--multiplet", "T2g^4", "--pg", "m-3m"])
    report("(t2g)^4 = (t2g)^2 terms (hole equivalence)",
           code == 0 and "(T2g)^4 = ^3T1g + ^1A1g + ^1Eg + ^1T2g" in out, out)

    code, out = run_group(["--multiplet", "T2g^6", "--pg", "m-3m"])
    report("(t2g)^6 closed shell = ^1A1g",
           code == 0 and "(T2g)^6 = ^1A1g" in out, out)

    # two inequivalent shells + ligand-field check of the parent orbital
    code, out = run_group(["--multiplet", "T2g1", "Eg1", "--pg", "m-3m"])
    report("(t2g)^1(eg)^1 = ^3T1g + ^3T2g + ^1T1g + ^1T2g",
           code == 0
           and "(T2g)^1 (Eg)^1 = ^3T1g + ^3T2g + ^1T1g + ^1T2g" in out, out)

    code, out = run_group(["--multiplet", "T2g2", "Eg1", "--pg", "m-3m",
                           "--orbital", "d"])
    report("(t2g)^2(eg)^1 quartets first: ^4T1g + ^4T2g + doublets",
           code == 0 and "= ^4T1g + ^4T2g + ^2A1g" in out and "2(^2Eg)" in out
           and "60 states" in out, out)
    report("--orbital d prints the ligand-field splitting",
           "1(Eg) + 1(T2g)" in out, out)

    # other point groups (dash-value merge for -43m; digit-suffix irrep A1)
    code, out = run_group(["--multiplet", "E^2", "--pg", "3m"])
    report("C3v (e)^2 = ^3A2 + ^1A1 + ^1E",
           code == 0 and "(E)^2 = ^3A2 + ^1A1 + ^1E" in out, out)

    code, out = run_group(["--multiplet", "E2", "--pg", "-43m"])
    report("Td (e)^2 = ^3A2 + ^1A1 + ^1E",
           code == 0 and "(E)^2 = ^3A2 + ^1A1 + ^1E" in out, out)

    code, out = run_group(["--multiplet", "A12", "--pg", "422"])
    report("digit-suffix irrep token A12 parsed as (A1)^2",
           code == 0 and "(A1)^2 = ^1A1" in out, out)

    # real pairs of complex-conjugate irreps (E of 3, Eg of m-3) and the
    # scalar characters of point group 1
    code, out = run_group(["--multiplet", "E2", "--pg", "3"])
    report("C3 (e)^2 = ^3A + ^1A + ^1E, 6 states, no warning",
           code == 0 and "(E)^2 = ^3A + ^1A + ^1E\ncheck: 6 states = C(4,2) = 6" in out
           and "WARNING" not in out, out)

    code, out = run_group(["--multiplet", "A2", "--pg", "1"])
    report("point group 1 accepted: (A)^2 = ^1A",
           code == 0 and "(A)^2 = ^1A" in out and "Traceback" not in out, out)

    code, out = run_group(["--multiplet", "Eg2", "--pg", "m-3", "--orbital", "d"])
    report("Th (eg)^2 energies A-8B / A+8B+4C / A+2C (the Oh values)",
           code == 0 and "1(Eg) + 1(Tg)" in out and "^3Ag: A - 8B" in out
           and "^1Ag: A + 8B + 4C" in out and "^1Eg: A + 2C" in out, out)

    import contextlib
    import io
    from itertools import combinations

    from phonopy.phonon.character_table import character_table as all_tables

    from crystod import multiplet
    from crystod.decompose_irrep import decompose, get_character_table
    from crystod.ligand_field import get_orbital_characters

    failures = []
    for point_group in all_tables:
        table = get_character_table(point_group)
        for orbital, expected in (("d", 45), ("f", 91)):
            counts = decompose(list(get_orbital_characters(orbital, table).values()), table)
            shells = [name for name, count in counts.items() for _ in range(count)]
            configs = [[f"{name}^2"] for name in shells] + [
                [f"{first}^1", f"{second}^1"] for first, second in combinations(shells, 2)
            ]
            states = 0
            for config in configs:
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    multiplet.main([f"--point-group={point_group}", "--config", *config])
                text = buffer.getvalue()
                match = re.search(r"check: (\d+) states", text)
                if "WARNING" in text or not match:
                    failures.append(f"{point_group} {' '.join(config)}: state-count warning")
                    continue
                states += int(match.group(1))
            if states != expected:
                failures.append(f"{orbital}^2 in {point_group}: {states} states (expected {expected})")
    report("d^2 / f^2 over every shell distribution: 45 / 91 states in all 32 point groups",
           not failures, "\n".join(failures))

    # Racah multiplet energies (--orbital; validated vs Tanabe-Sugano/Griffith)
    code, out = run_group(["--multiplet", "T2g3", "--pg", "m-3m",
                           "--orbital", "d"])
    report("(t2g)^3 Racah energies (Tanabe-Sugano table)",
           code == 0 and "^4A2g: 3A - 15B" in out
           and "^2Eg : 3A - 6B + 3C" in out
           and "^2T1g: 3A - 6B + 3C" in out
           and "^2T2g: 3A + 5C" in out, out)
    report("(t2g)^3 ground state ^4A2g for any B, C > 0",
           "^4A2g   (lowest for any B > 0, C > 0)" in out, out)

    code, out = run_group(["--multiplet", "T2g2", "--pg", "m-3m",
                           "--orbital", "d"])
    report("(t2g)^2 energies A-5B / A+B+2C / A+10B+5C",
           code == 0 and "^3T1g: A - 5B" in out
           and "^1A1g: A + 10B + 5C" in out
           and "^1Eg : A + B + 2C" in out, out)

    code, out = run_group(["--multiplet", "Eg2", "--pg", "m-3m",
                           "--orbital", "d"])
    report("(eg)^2 energies A-8B / A+2C / A+8B+4C",
           code == 0 and "^3A2g: A - 8B" in out and "^1Eg : A + 2C" in out
           and "^1A1g: A + 8B + 4C" in out, out)

    code, out = run_group(["--multiplet", "T2g1", "Eg1", "--pg", "m-3m",
                           "--orbital", "d"])
    report("(t2g)^1(eg)^1 ground ^3T2g = A-8B (resolves the Hund tie)",
           code == 0 and "^3T2g: A - 8B" in out and "^3T1g: A + 4B" in out
           and "^3T2g   (lowest for any B > 0, C > 0)" in out, out)

    code, out = run_group(["--multiplet", "T2g2", "Eg1", "--pg", "m-3m",
                           "--orbital", "d"])
    report("(t2g)^2(eg)^1 quartets 3A-15B / 3A-3B + CI blocks",
           code == 0 and "^4T2g: 3A - 15B" in out and "^4T1g: 3A - 3B" in out
           and "+- 3sqrt(2)B" in out and "configuration mixing" in out, out)

    # irrational coupled-parent off-diagonals (260828 bug report): these five
    # (t2g)^a(eg)^b configurations used to abort with a snap error before the
    # ground state was printed; the off-diagonals are genuinely r*sqrt(m)*B
    code, out = run_group(["--multiplet", "T2g3", "Eg1", "--pg", "m-3m",
                           "--orbital", "d"])
    report("(t2g)^3(eg)^1 completes with sqrt off-diagonals (d^4 ^5Eg ground)",
           code == 0 and "^5Eg" in out and "+-(5sqrt(3)B)" in out
           and "Ground-state Term Symbol" in out
           and "please report" not in out, out)

    code, out = run_group(["--multiplet", "T2g3", "Eg2", "--pg", "m-3m",
                           "--orbital", "d"])
    report("(t2g)^3(eg)^2 completes (d^5 ^6A1g ground, 2sqrt(3)B/4sqrt(3)B)",
           code == 0 and "^6A1g" in out and "+-(4sqrt(3)B)" in out
           and "please report" not in out, out)

    code, out = run_group(["--multiplet", "T2g4", "Eg2", "--pg", "m-3m",
                           "--orbital", "d"])
    report("(t2g)^4(eg)^2 completes (d^6 ^5T2g ground, 2sqrt(2)B)",
           code == 0 and "^5T2g" in out and "+-(2sqrt(2)B)" in out
           and "please report" not in out, out)

    code, out = run_group(["--multiplet", "T1u2", "--pg", "m-3m",
                           "--orbital", "p"])
    report("(p)^2 free-ion limit F0-5F2 / F0+F2 / F0+10F2",
           code == 0 and "^3T1g: F0 - 5F2" in out and "^1A1g: F0 + 10F2" in out
           and "^1Eg : F0 + F2" in out and "^1T2g: F0 + F2" in out, out)

    # f shells: reduced Slater parameters F0/F2/F4/F6, hydrogenic-ratio CI
    code, out = run_group(["--multiplet", "T1u2", "--pg", "m-3m",
                           "--orbital", "f"])
    report("f-shell (T1u)^2 energies in reduced F0/F2/F4/F6",
           code == 0 and "^3T1g: F0 - (35/4)F2 - (63/2)F4 - (1235/4)F6" in out
           and "^1A1g: F0 + (35/2)F2 + 126F4 + (1535/2)F6" in out
           and "(lowest for any positive Slater parameters)" in out, out)

    code, out = run_group(["--multiplet", "T1u3", "--pg", "m-3m",
                           "--orbital", "f"])
    report("f-shell (T1u)^3 = ^4A1u ground (t2g^3 analogue)",
           code == 0 and "(T1u)^3 = ^4A1u + ^2Eu + ^2T1u + ^2T2u" in out
           and "^4A1u: 3F0 - (105/4)F2 - (189/2)F4 - (3705/4)F6" in out, out)

    code, out = run_group(["--multiplet", "T1u2", "T2u2", "--pg", "m-3m",
                           "--orbital", "f"])
    report("f-shell (T1u)^2(T2u)^2 quintets + hydrogenic-ratio CI blocks",
           code == 0 and "^5A1g: 6F0 - 60F2 - 198F4 - 1716F6" in out
           and "^5Eg : 6F0 - 75F2 - 216F4 - 1443F6" in out
           and "hydrogenic 4f ratios" in out
           and "225 states" in out, out)

    # ground-state line without --orbital (Hund's rules)
    code, out = run_group(["--multiplet", "T2g3", "--pg", "m-3m"])
    report("Hund ground state ^4A2g printed without --orbital",
           code == 0 and "Ground-state Term Symbol (Hund's rules)" in out
           and "^4A2g" in out, out)

    code, out = run_group(["--multiplet", "T2g1", "Eg1", "--pg", "m-3m"])
    report("Hund tie lists candidates ^3T1g, ^3T2g",
           code == 0 and "candidates: ^3T1g, ^3T2g" in out, out)

    code, out = run_group(["--multiplet", "E2", "--pg", "3m", "--orbital", "d"])
    report("shell with multiplicity in the orbital splitting rejected cleanly",
           code != 0 and "not defined by symmetry alone" in out
           and "Traceback" not in out, out)

    # errors
    code, out = run_group(["--multiplet", "T2g^7", "--pg", "m-3m"])
    report("overfilled shell rejected cleanly",
           code != 0 and "holds 1 to 6 electrons" in out
           and "Traceback" not in out, out)

    code, out = run_group(["--multiplet", "Xx^2", "--pg", "m-3m"])
    report("unknown irrep rejected with available list",
           code != 0 and "not an irrep" in out and "Choose from" in out
           and "Traceback" not in out, out)

    code, out = run_group(["--multiplet", "T2g^2", "--pg", "m-3m",
                           "--orbital", "p"])
    report("shell missing from the parent-orbital splitting rejected",
           code != 0 and "does not occur in the p-orbital splitting" in out
           and "Traceback" not in out, out)

    code, out = run_group(["--multiplet", "T2g^2", "--sg", "Pm-3m"])
    report("--multiplet without --pg rejected cleanly",
           code != 0 and "requires --pg" in out and "Traceback" not in out, out)

    code, out = run_group(["--product", "T2g", "T2g", "--pg", "m-3m",
                           "--orbital", "d"])
    report("--orbital outside --multiplet rejected cleanly",
           code != 0 and "only used with --multiplet" in out
           and "Traceback" not in out, out)

    # --visualize: exact term eigenstates as an HTML page
    with tempfile.TemporaryDirectory() as tmp:
        code, out = run_group(["--multiplet", "T2g3", "--pg", "m-3m",
                               "--orbital", "d", "--visualize"], cwd=tmp)
        html_path = os.path.join(tmp, "Multiplet_m-3m_T2g3.html")
        report("--visualize exit 0 and default file name (* Output files * block)",
               code == 0 and "* Output files *\n  Term-state viewer written to" in out
               and os.path.isfile(html_path), out)
        if os.path.isfile(html_path):
            with open(html_path) as handle:
                html = handle.read()
            data = json.loads(
                re.search(r"const DATA = (\{.*?\});\n", html, re.S).group(1)
            )
            report("t2g orbitals identified as dxy, dyz, dxz",
                   data["shells"][0]["labels"] == ["dxy", "dyz", "dxz"],
                   html_path)
            ground = data["terms"][0]
            partner = ground["branches"][0]["partners"][0]
            report("^4A2g is the single determinant |dxy up, dyz up, dxz up>",
                   ground["symbol"] == {"mult": "4", "irrep": "A2g"}
                   and len(partner["dets"]) == 1 and partner["dets"][0]["c"] == "1"
                   and partner["dets"][0]["boxes"] == [[1, 1, 1]], html_path)
            norms_ok = all(
                abs(sum(e["cf"] ** 2
                        for e in branch["partners"][0]["dets"]) - 1.0) < 1e-4
                for term in data["terms"] for branch in term["branches"]
            )
            report("every term state is normalized", norms_ok, html_path)
            charge = partner["gc"]
            diagonal = [round(charge[i][i], 6) for i in range(5)]
            off_ok = all(
                abs(charge[i][j]) < 1e-6
                for i in range(5) for j in range(5) if i != j
            )
            report("^4A2g charge density matrix = t2g projector "
                   "(diag 1,1,0,1,0 over dxy,dyz,dz2,dxz,dx2-y2)",
                   diagonal == [1.0, 1.0, 0.0, 1.0, 0.0] and off_ok, html_path)
            report("density surface data present for every term",
                   all("gc" in branch["partners"][0] and "gs" in branch["partners"][0]
                       for term in data["terms"] for branch in term["branches"]),
                   html_path)

        code, out = run_group(["--multiplet", "T2g2", "Eg1", "--pg", "m-3m",
                               "--orbital", "d", "--visualize",
                               "--output", "states.html"], cwd=tmp)
        report("--visualize --output custom name (two-shell configuration)",
               code == 0 and os.path.isfile(os.path.join(tmp, "states.html")),
               out)

        # point group 1 (scalar characters) and real-pair irreps (norm 2|G|)
        for args, name in ((["A2", "--pg", "1", "--orbital", "s"], "Multiplet_1_A2.html"),
                           (["Eg2", "--pg", "m-3", "--orbital", "d"], "Multiplet_m-3_Eg2.html")):
            vcode, vout = run_group(["--multiplet", *args, "--visualize"], cwd=tmp)
            report(f"--multiplet {' '.join(args)} --visualize: exit 0, no Traceback",
                   vcode == 0 and "Traceback" not in vout and "ERROR" not in vout
                   and os.path.isfile(os.path.join(tmp, name)), vout)

        report("coupled-parent CI matrices printed (Tanabe-Sugano basis)",
               "CI matrices in the coupled-parent basis" in out
               and "T2g^2(^1A1g) Eg^1(^2Eg)" in out, out)
        report("^2Eg parent matrix: 3A + 8B + 6C / 3A - B + 3C / +-10B",
               "<1|H|1> = 3A + 8B + 6C" in out
               and "<2|H|2> = 3A - B + 3C" in out
               and "<1|H|2> = +-(10B)" in out, out)
        report("^2T1g parent off-diagonal 3B (eigenvalues +-3sqrt(2)B)",
               "<1|H|2> = +-(3B)" in out, out)

        # f shell: mixed basis combinations get short symbols + a legend
        code, out = run_group(["--multiplet", "T1u2", "--pg", "m-3m",
                               "--orbital", "f", "--visualize"], cwd=tmp)
        f_path = os.path.join(tmp, "Multiplet_m-3m_T1u2.html")
        report("f-shell --visualize exit 0", code == 0 and os.path.isfile(f_path), out)
        if os.path.isfile(f_path):
            with open(f_path) as handle:
                f_html = handle.read()
            report("mixed f combinations shortened with a legend",
                   "Orbital basis functions" in f_html
                   and "t1u(1) = " in f_html and "fz3" in f_html, f_path)

    code, out = run_group(["--multiplet", "T2g2", "--pg", "m-3m",
                           "--visualize"])
    report("--visualize without --orbital rejected cleanly",
           code != 0 and "needs --orbital" in out and "Traceback" not in out,
           out)

    code, out = run_group(["--product", "T2g", "T2g", "--pg", "m-3m",
                           "--visualize"])
    report("--visualize outside --multiplet rejected cleanly",
           code != 0 and "only used with --multiplet" in out
           and "Traceback" not in out, out)


def test_22_poscar2cif() -> None:
    print("\n[22] crystod-group --poscar2cif / --cif2poscar (Bilbao-style CIF)")

    def cif_ops(text: str) -> set:
        return set(
            line.split()[1]
            for line in text.splitlines()
            if line[:4].strip().isdigit()
        )

    reference_path = os.path.join(ROOT, "example", "test_POSCARs", "221_PPOSCAR_ScF3.cif")
    with tempfile.TemporaryDirectory() as tmp:
        poscar = os.path.join(tmp, "221_PPOSCAR_ScF3")
        shutil.copy(POSCAR_ScF3, poscar)
        code, out = run_group(["--poscar2cif", "-c", poscar])
        cif_path = poscar + ".cif"
        report("ScF3 conversion exits 0 and writes <POSCAR>.cif",
               code == 0 and os.path.isfile(cif_path)
               and "Pm-3m (No. 221)" in out, out)
        content = open(cif_path).read()
        report("Bilbao layout: aligned keys, quoted H-M, 4-decimal cell",
               "_symmetry_Int_Tables_number        221" in content
               and '_symmetry_space_group_name_H-M     "Pm-3m"' in content
               and "_cell_length_a                     4.0696" in content, content)
        reference = open(reference_path).read()
        report("48 operations, set identical to the Bilbao reference CIF",
               len(cif_ops(content)) == 48
               and cif_ops(content) == cif_ops(reference), content)
        report("compact unquoted operator strings ('   1   x,y,z')",
               "   1   x,y,z" in content and "'x" not in content, content)
        report("a structure whose own symmetry matches the tolerance is "
               "converted silently",
               "WARNING" not in out, out)

        # a distortion below the tolerance is averaged away by the
        # standardization, so the CIF silently becomes the PARENT structure
        # (GaN F-43m -> I-42d moves N by 0.0002 A, 50x under the default
        # 0.01 A) -- a symmetry-mode analysis then finds no distortion at
        # all with nothing to explain why, so the conversion says so
        gan_child = os.path.join(
            ROOT, "example", "23_symmetry_mode", "debug5_GaN",
            "POSCAR-finish_GaN_I-42d")
        gan_copy = os.path.join(tmp, "POSCAR-finish_GaN_I-42d")
        shutil.copy(gan_child, gan_copy)
        code, out = run_group(["--poscar2cif", "-c", gan_copy])
        report("a sub-tolerance distortion is reported, not silently "
               "symmetrized away (GaN I-42d -> F-43m at 0.01 A)",
               code == 0 and "F-43m (No. 216)" in out
               and "WARNING" in out and "I-42d (No. 122)" in out
               and "smaller --tolerance" in out, out)
        code, out = run_group(
            ["--poscar2cif", "-c", gan_copy, "--tolerance", "0.0001"])
        report("the same structure converts to I-42d, warning-free, at "
               "--tolerance 0.0001",
               code == 0 and "I-42d (No. 122)" in out
               and "WARNING" not in out, out)
        report("one representative site per orbit (F1 + Sc1, occupancy 1.0000)",
               re.search(r"F1 F 0\.\d{5} 0\.\d{5} 0\.\d{5} 1\.0000", content)
               is not None
               and "Sc1 Sc 0.00000 0.00000 0.00000 1.0000" in content, content)
        report("_chemical_formula_sum written (IUCr style, cation first)",
               '_chemical_formula_sum              "Sc F3"' in content, content)

        # chemically ordered _chemical_formula_sum: cations before anions,
        # most special Wyckoff site first, then increasing valence -- the
        # La3Ni2O7 example (La on the 2-fold site before the lower-valence
        # Ni on the 4-fold site) and PbZrO3 (site tie, Pb+2 before Zr+4,
        # where electronegativity alone would order Zr first)
        poscar = os.path.join(tmp, "POSCAR_test")
        shutil.copy(os.path.join(ROOT, "example", "23_symmetry_mode",
                                 "POSCAR_test"), poscar)
        code, out = run_group(["--poscar2cif", "-c", poscar])
        content = open(poscar + ".cif").read()
        report("POSCAR_test: _chemical_formula_sum \"La3 Ni2 O7\" "
               "(symmetry rule beats valence)",
               code == 0
               and '_chemical_formula_sum              "La3 Ni2 O7"'
               in content, content)
        from crystod.poscar2cif import (
            chemical_formula_parts,
            format_chemical_formula,
        )

        import spglib as _spglib

        def _formula_of(path):
            import numpy as _np
            from pymatgen.core import Structure as _Structure

            try:
                s = _Structure.from_file(path)
            except Exception:
                s = _Structure.from_str(open(path).read(), fmt="poscar")
            cell = (_np.asarray(s.lattice.matrix), _np.asarray(s.frac_coords),
                    [site.specie.Z for site in s])
            std = _spglib.standardize_cell(
                cell, to_primitive=False, no_idealize=False, symprec=0.01)
            ds = _spglib.get_symmetry_dataset(std, symprec=1e-5)
            eq = (ds["equivalent_atoms"] if isinstance(ds, dict)
                  else ds.equivalent_atoms)
            return format_chemical_formula(chemical_formula_parts(std[2], eq))

        sym_example = os.path.join(ROOT, "example", "23_symmetry_mode")
        expectations = {
            os.path.join(sym_example, "POSCAR_K2SeO4_Pnma.cif"): "K2SeO4",
            os.path.join(sym_example, "221_PPOSCAR_SrTiO3.cif"): "SrTiO3",
            os.path.join(sym_example, "debug2_PbZrO3", "POSCAR_PbZrO3_Pm-3m.cif"):
                "PbZrO3",
            os.path.join(sym_example, "139_PPOSCAR_La3Ni2O7.cif"): "La3Ni2O7",
            POSCAR_ScF3: "ScF3",
            POSCAR_NaCl: "NaCl",
        }
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            got = {path: _formula_of(path) for path in expectations}
        report("chemical formula ordering (K2SeO4, SrTiO3, PbZrO3, "
               "La3Ni2O7, ScF3, NaCl)",
               got == expectations,
               str({os.path.basename(k): v for k, v in got.items()}))

        # nonsymmorphic: ITA-standard Pnma operators (spglib standardization)
        poscar = os.path.join(tmp, "62_PPOSCAR_CaTiO3")
        shutil.copy(os.path.join(ROOT, "example", "test_POSCARs", "62_PPOSCAR_CaTiO3"),
                    poscar)
        code, out = run_group(["--poscar2cif", "-c", poscar])
        content = open(poscar + ".cif").read()
        ita_pnma = {"x,y,z", "-x+1/2,-y,z+1/2", "x+1/2,-y+1/2,-z+1/2",
                    "-x,y+1/2,-z", "-x,-y,-z", "x+1/2,y,-z+1/2",
                    "-x+1/2,y+1/2,z+1/2", "x,-y+1/2,z"}
        report("CaTiO3 Pnma: ITA general-position operators",
               code == 0 and '"Pnma"' in content
               and cif_ops(content) == ita_pnma, content)

        # centred lattice: conventional cell with the centring translations
        poscar = os.path.join(tmp, "225_PPOSCAR_NaCl")
        shutil.copy(os.path.join(ROOT, "example", "test_POSCARs", "225_PPOSCAR_NaCl"),
                    poscar)
        code, out = run_group(["--poscar2cif", "-c", poscar])
        content = open(poscar + ".cif").read()
        report("NaCl Fm-3m: 192 conventional-cell operations with centring",
               code == 0 and '"Fm-3m"' in content
               and len(cif_ops(content)) == 192
               and "x,y+1/2,z+1/2" in cif_ops(content), content)

        # --output override
        target = os.path.join(tmp, "custom_name.cif")
        code, out = run_group(["--poscar2cif", "-c", poscar, "--output", target])
        report("--output overrides the default <POSCAR>.cif path",
               code == 0 and os.path.isfile(target), out)
        report("layout: * Structure * then a final * Output files * block",
               out.startswith("\n* Structure *\n")
               and out.rstrip().endswith("\n* Output files *\n"
                                         f"  Bilbao-style CIF written to: {target}"), out)
        code, out = run_group(["--poscar2cif", "-c", poscar, "--output", target])
        report("rewriting an existing CIF is noted in the Output files line",
               code == 0 and out.rstrip().endswith(
                   f"  Bilbao-style CIF written to: {target} (existing file overwritten)"),
               out)

        # --cif2poscar: inverse conversion (round trip)
        poscar = os.path.join(tmp, "221_PPOSCAR_SrTiO3")
        shutil.copy(POSCAR_SrTiO3, poscar)
        run_group(["--poscar2cif", "-c", poscar])
        os.remove(poscar)
        code, out = run_group(["--cif2poscar", "-c", poscar + ".cif"])
        report("--cif2poscar writes the input path without .cif (primitive)",
               code == 0 and os.path.isfile(poscar)
               and "primitive cell, 5 atoms" in out, out)
        report("--cif2poscar layout: final * Output files * block",
               out.rstrip().endswith(f"\n* Output files *\n  POSCAR written to: {poscar}"),
               out)
        content = open(poscar).read()
        report("POSCAR format: species lines, 'direct', element-tagged coords",
               "Sr Ti O" in content and "direct" in content
               and "0.500000 0.500000 0.500000 Ti" in content, content)

        bilbao = os.path.join(tmp, "ref_ScF3.cif")
        shutil.copy(reference_path, bilbao)
        code, out = run_group(["--cif2poscar", "-c", bilbao])
        report("genuine Bilbao CIF converts (ScF3, 4-atom primitive cell)",
               code == 0 and "Pm-3m (No. 221)" in out
               and "primitive cell, 4 atoms" in out, out)

        code, out = run_group(["--cif2poscar", "-c",
                               os.path.join(tmp, "225_PPOSCAR_NaCl.cif")])
        report("NaCl CIF -> 2-atom primitive cell by default",
               code == 0 and "primitive cell, 2 atoms" in out, out)

        code, out = run_group(["--cif2poscar", "-c",
                               os.path.join(tmp, "225_PPOSCAR_NaCl.cif"),
                               "--conventional",
                               "--output", os.path.join(tmp, "NaCl_conv")])
        report("--conventional -> 8-atom conventional cell",
               code == 0 and "conventional cell, 8 atoms" in out
               and os.path.isfile(os.path.join(tmp, "NaCl_conv")), out)

    # errors
    code, out = run_group(["--cif2poscar"])
    report("--cif2poscar without -c rejected cleanly",
           code != 0 and "requires -c/--cell" in out
           and "Traceback" not in out, out)

    code, out = run_group(["--product", "T2g", "T2g", "--pg", "m-3m",
                           "--conventional"])
    report("--conventional outside --cif2poscar rejected cleanly",
           code != 0 and "only used" in out and "Traceback" not in out, out)

    code, out = run_group(["--poscar2cif"])
    report("--poscar2cif without -c rejected cleanly",
           code != 0 and "requires -c/--cell" in out
           and "Traceback" not in out, out)

    code, out = run_group(["--poscar2cif", "-c", "no_such_POSCAR_file"])
    report("missing POSCAR rejected cleanly",
           code != 0 and "not found" in out and "Traceback" not in out, out)

    code, out = run_group(["--product", "T2g", "T2g", "--pg", "m-3m",
                           "-c", "POSCAR"])
    report("-c outside --poscar2cif rejected cleanly",
           code != 0 and "only used" in out and "Traceback" not in out, out)


def test_23_symmetry_mode() -> None:
    print("\n[23] crystod-group --supergroup-cif (symmetry-mode analysis)")

    import spglib

    if tuple(int(x) for x in spglib.__version__.split(".")[:2]) < (2, 4):
        print(f"  [SKIP] spglib {spglib.__version__} < 2.4: subgroup "
              "identification is unreliable in old spglib; run this section "
              "in the crystod env.")
        return

    example = os.path.join(ROOT, "example", "23_symmetry_mode")
    parent = os.path.join(example, "221_PPOSCAR_SrTiO3.cif")
    child = os.path.join(example, "140_PPOSCAR_SrTiO3.cif")

    with tempfile.TemporaryDirectory() as tmp:
        # the AMPLIMODES reference case (Bilbao PDF in ~/CrystOD-main/AMPLIMODES)
        code, out = run_group(
            ["--supergroup-cif", parent, "--subgroup-cif", child], cwd=tmp
        )
        report("SrTiO3 Pm-3m -> I4/mcm exits 0 and identifies both groups",
               code == 0 and "Pm-3m (No. 221)" in out
               and "I4/mcm (No. 140)" in out, out)
        report("R5- mode at R with amplitude 0.3303 A (AMPLIMODES value)",
               "R5-" in out and "(1/2,1/2,1/2)" in out
               and re.search(r"R5-\s+\S+\s+140 I4/mcm\s+1\s+0\.3303", out)
               is not None, out)
        report("max displacement 0.1651 A and total distortion 0.3303 A",
               "maximum atomic displacement: 0.1651 A" in out
               and "total distortion amplitude : 0.3303 A" in out, out)
        report("cell multiplication 2 and AMPLIMODES citation printed",
               "primitive cell multiplication: 2" in out
               and "J. Appl. Cryst. 42, 820-833 (2009)" in out, out)
        report("mode displacement VESTA file written (R5-)",
               os.path.isfile(
                   os.path.join(tmp, "221_PPOSCAR_SrTiO3_R5-.vesta")
               ) and "221_PPOSCAR_SrTiO3_R5-.vesta" in out, out)
        report("decomposition table saved as sym_mode_SrTiO3",
               os.path.isfile(os.path.join(tmp, "sym_mode_SrTiO3"))
               and "R5-" in open(os.path.join(tmp, "sym_mode_SrTiO3")).read(),
               out)
        default_out = out

        # where the files go: --output-dir (created, nested) or --no-files;
        # the current directory stays empty either way
        with tempfile.TemporaryDirectory() as cwd:
            code, out = run_group(["--supergroup-cif", parent, "--subgroup-cif", child,
                                   "--output-dir", os.path.join("modes", "sto")],
                                  cwd=cwd)
            target = os.path.join(cwd, "modes", "sto")
            report("--output-dir: table and VESTA file in the new nested directory",
                   code == 0 and os.path.isfile(os.path.join(target, "sym_mode_SrTiO3"))
                   and os.path.isfile(os.path.join(target, "221_PPOSCAR_SrTiO3_R5-.vesta"))
                   and sorted(os.listdir(cwd)) == ["modes"], out)
            report("--output-dir: the printed paths are the paths written",
                   "Decomposition table saved to "
                   + os.path.join("modes", "sto", "sym_mode_SrTiO3") in out
                   and os.path.join("modes", "sto", "221_PPOSCAR_SrTiO3_R5-.vesta")
                   in out, out)
            report("--output-dir: the files equal those of the default run",
                   open(os.path.join(target, "sym_mode_SrTiO3")).read()
                   == open(os.path.join(tmp, "sym_mode_SrTiO3")).read()
                   and open(os.path.join(target, "221_PPOSCAR_SrTiO3_R5-.vesta")).read()
                   == open(os.path.join(tmp, "221_PPOSCAR_SrTiO3_R5-.vesta")).read(), out)
            report("--output-dir: stdout differs from the default run only in paths",
                   out.replace(os.path.join("modes", "sto") + os.sep, "") == default_out,
                   out)
            code, out = run_group(["--supergroup-cif", parent, "--subgroup-cif", child,
                                   "--output-dir", target, "--conventional"], cwd=cwd)
            report("--output-dir with an absolute path and --conventional",
                   code == 0 and os.path.isfile(
                       os.path.join(target, "221_PPOSCAR_SrTiO3_R5-_conv.vesta"))
                   and os.path.join(target, "221_PPOSCAR_SrTiO3_R5-_conv.vesta") in out,
                   out)
            last_block = out.rstrip().rsplit("\n\n", 1)[-1]
            report("layout: * Output files * is the final block (table, display cell "
                   "nested under the VESTA list)",
                   last_block.startswith("* Output files *\n  Decomposition table saved to ")
                   and "\n  Mode displacement VESTA files (parent conventional basis):\n"
                       "    display cell in parent primitive units (rows):\n      (" in last_block
                   and "\n    " + os.path.join(target, "221_PPOSCAR_SrTiO3_R5-_conv.vesta")
                   in last_block, out)
        with tempfile.TemporaryDirectory() as cwd:
            code, out = run_group(["--supergroup-cif", parent, "--subgroup-cif", child,
                                   "--no-files"], cwd=cwd)
            report("--no-files: table printed, nothing written, no 'saved' lines",
                   code == 0 and os.listdir(cwd) == []
                   and re.search(r"R5-\s+\S+\s+140 I4/mcm\s+1\s+0\.3303", out)
                   is not None and "saved to" not in out and ".vesta" not in out
                   and "* Output files *" not in out,
                   out)
            code, out = run_group(["--supergroup-cif", parent, "--subgroup-cif", child,
                                   "--no-files", "--output-dir", "x"], cwd=cwd)
            report("--no-files with --output-dir rejected cleanly",
                   code != 0 and "exclude each other" in out
                   and os.listdir(cwd) == [] and "Traceback" not in out, out)
            code, out = run_group(["--parent", "Pm-3m", "--irrep", "GM4-",
                                   "--output-dir", "x"], cwd=cwd)
            report("--output-dir outside --supergroup-cif rejected cleanly",
                   code != 0 and "only used with --supergroup-cif" in out
                   and "Traceback" not in out, out)
            open(os.path.join(cwd, "occupied"), "w").close()
            code, out = run_group(["--supergroup-cif", parent, "--subgroup-cif", child,
                                   "--output-dir", "occupied"], cwd=cwd)
            report("--output-dir naming an existing file rejected before the analysis",
                   code != 0 and "not a directory" in out
                   and "Supergroup" not in out and "Traceback" not in out, out)

        # F-centred parent (ZrO2 fluorite -> tetragonal; second AMPLIMODES PDF)
        code, out = run_group(["--supergroup-cif",
                               os.path.join(example, "225_PPOSCAR_ZrO2.cif"),
                               "--subgroup-cif",
                               os.path.join(example, "137_PPOSCAR_ZrO2.cif")],
                              cwd=tmp)
        report("ZrO2 Fm-3m -> P4_2/nmc: X2- with amplitude 0.5773 A",
               code == 0 and "(1/2,0,1/2)" in out
               and re.search(r"X2-\s+\S+\s+137 P4_2/nmc\s+1\s+0\.5773", out)
               is not None, out)
        report("ZrO2 X2- VESTA file written",
               os.path.isfile(
                   os.path.join(tmp, "225_PPOSCAR_ZrO2_X2-.vesta")
               ), out)

        # C-centred parent (Cmcm -> Pnma, permuted child axes): regression for
        # the non-symmetric primitive-matrix conversion bug -- the row-vector
        # positions must use x_c A^-1, not x_c A^-T (third AMPLIMODES PDF, in
        # example/23_symmetry_mode/debug)
        debug = os.path.join(example, "debug1_LSNO")
        code, out = run_group(
            ["--supergroup-cif",
             os.path.join(debug, "CONTCAR-POSCAR_SrLi2Nb2O7_Cmcm.cif"),
             "--subgroup-cif",
             os.path.join(debug, "CONTCAR-POSCAR_SrLi2Nb2O7_Pnma.cif")],
            cwd=tmp,
        )
        report("SrLi2Nb2O7 Cmcm -> Pnma maps (C-centred parent, permuted axes)",
               code == 0 and "Cmcm (No. 63)" in out
               and "Pnma (No. 62)" in out, out)
        report("SrLi2Nb2O7 Y2- at (1/2,1/2,0) with amplitude 0.1361 A "
               "(AMPLIMODES value)",
               "(1/2,1/2,0)" in out
               and re.search(r"Y2-\s+\S+\s+62 Pnma\s+10\s+0\.1361", out)
               is not None, out)
        report("SrLi2Nb2O7 secondary GM1+ 0.0011 A, max displacement 0.0293 A, "
               "total 0.1362 A",
               re.search(r"GM1\+\s+\S+\s+63 Cmcm\s+9\s+0\.0011", out)
               is not None
               and "maximum atomic displacement: 0.0293 A" in out
               and "total distortion amplitude : 0.1362 A" in out, out)

        # pseudo-symmetric child (symmetry-breaking component ~0.001 A while
        # the fully symmetric GM1+ relaxation is ~0.3 A): regression for the
        # adaptive subgroup-member selection -- a fixed 0.05 A cutoff sees
        # every broken operation as intact, H becomes the full parent group,
        # and the Y2- mode was dropped ("distortion is not fully captured")
        code, out = run_group(
            ["--supergroup-cif",
             os.path.join(debug, "63_POSCAR_Li2SrNb2O7.cif"),
             "--subgroup-cif",
             os.path.join(debug, "62_POSCAR_Li2SrNb2O7.cif"),
             "--tolerance", "0.001"],
            cwd=tmp,
        )
        report("pseudo-symmetric Li2SrNb2O7 (--tolerance 0.001) decomposes",
               code == 0 and "Pnma (No. 62)" in out
               and re.search(r"Y2-\s+\S+\s+62 Pnma\s+10\s+0\.0030", out)
               is not None
               and re.search(r"GM1\+\s+\S+\s+63 Cmcm\s+9\s+0\.3257", out)
               is not None, out)

        # rhombohedral parent (R32 -> R3, Ag3SI): regression for the
        # centring-translation generation in the ISO-IR origin-shift fit --
        # the phonopy primitive matrix is column-convention, and using its
        # ROWS yields (1/3,1/3,1/3) instead of the obverse (2/3,1/3,1/3),
        # so no origin could match spglib's centring copies ("could not fit
        # the ISO-IR origin shift"); only R lattices distinguish the two.
        # ISODISTORT reference (example/23_symmetry_mode/Ag3SI/isotropy_*):
        # GM1 0.0741 (exact match) and GM2 0.7070 at ISODISTORT's polar
        # origin (Ag z pinned); CrystOD's minimum-distortion origin
        # (AMPLIMODES convention, cf. the BaTiO3 case) removes the acoustic
        # z translation from GM2: sqrt(0.7070^2 - 0.0757) = 0.6512 A.
        ag3si = os.path.join(example, "debug3_Ag3SI")
        code, out = run_group(
            ["--supergroup-cif",
             os.path.join(ag3si, "POSCAR_Ag3SI_R32.cif"),
             "--subgroup-cif",
             os.path.join(ag3si, "POSCAR_Ag3SI_R3.cif")],
            cwd=tmp,
        )
        report("Ag3SI R32 -> R3 (rhombohedral parent) decomposes",
               code == 0 and "R32 (No. 155)" in out
               and "R3 (No. 146)" in out, out)
        report("Ag3SI GM1 0.0741 A (ISODISTORT value) and polar GM2 "
               "0.6512 A (minimum-distortion origin)",
               re.search(r"GM1\s+\S+\s+155 R32\s+1\s+0\.0741", out)
               is not None
               and re.search(r"GM2\s+\S+\s+146 R3\s+4\s+0\.6512", out)
               is not None
               and "maximum atomic displacement: 0.3281 A" in out
               and "total distortion amplitude : 0.6554 A" in out
               and "note: the subgroup is polar" in out, out)

        # measured, not assumed: shifting CrystOD's field to ISODISTORT's
        # polar origin (Ag z-displacement pinned to zero) must land on
        # every ISODISTORT amplitude exactly -- the two programs differ
        # ONLY by the acoustic z translation inside GM2
        import numpy as _np

        from crystod.symmetry_mode import SymmetryModeAnalysis as _SMA
        _ag = _SMA(os.path.join(ag3si, "POSCAR_Ag3SI_R32.cif"),
                   os.path.join(ag3si, "POSCAR_Ag3SI_R3.cif"), 0.01)
        _modes = {m.irrep_name: m for m in _ag.modes}
        _gm2 = _modes["GM2"].projected_u
        _c_hat = _np.sum(_ag.L_parent, axis=0)
        _c_hat = _c_hat / _np.linalg.norm(_c_hat)
        _species = _np.asarray(_ag.ref_z)
        _shift = -float(
            _np.mean(_gm2[_species == 47] @ _c_hat)) * _c_hat
        _gm2 = _gm2 + _shift
        _component = {
            z: float(_np.linalg.norm(_gm2[_species == z] @ _c_hat))
            for z in (47, 53, 16)
        }
        report("Ag3SI at the ISODISTORT origin: GM1 0.07411, GM2 0.70697, "
               "I 0.16902, S 0.44628 (all exact)",
               abs(_np.linalg.norm(_modes["GM1"].projected_u) - 0.07411)
               < 5e-5
               and abs(_np.linalg.norm(_gm2) - 0.70697) < 5e-5
               and _component[47] < 5e-5
               and abs(_component[53] - 0.16902) < 5e-5
               and abs(_component[16] - 0.44628) < 5e-5,
               str(_component))

        # purely ferroelastic pair (La3Ni2O7 I4/mmm -> Fmmm): the child's
        # ATOMS keep every one of the 16 parent operations exactly, and only
        # the orthorhombic metric breaks 8 of them.  Since amplitudes are
        # measured in the strain-free parent-derived reference lattice, those
        # 8 belong in H -- but spglib reads the child as Fmmm (8 operations),
        # so deriving |H| from the child's space group alone used to cut the
        # spectrum in the middle of 16 exact zeros and abort.  AMPLIMODES
        # reference (debug4_La3Ni2O7/Symmetry_mode_analysis_...pdf): a single
        # GM1+ mode, isotropy subgroup I4/mmm itself, dim 4, 0.4666 A.
        la327 = os.path.join(example, "debug4_La3Ni2O7")
        code, out = run_group(
            ["--supergroup-cif",
             os.path.join(la327, "POSCAR_La3Ni2O7_I4mmm.cif"),
             "--subgroup-cif",
             os.path.join(la327, "POSCAR_La3Ni2O7_Fmmm_Nature.cif")],
            cwd=tmp,
        )
        report("La3Ni2O7 I4/mmm -> Fmmm (pure spontaneous strain) decomposes",
               code == 0 and "Fmmm (No. 69)" in out
               and re.search(r"GM1\+\s+\S+\s+139 I4/mmm\s+4\s+0\.4666", out)
               is not None
               and "maximum atomic displacement: 0.2669 A" in out, out)
        report("the strain-only symmetry lowering is called out in the report",
               "8 parent operations that the child's own metric breaks" in out
               and "spontaneous" in out, out)

        # distortion far below the symmetry tolerance (GaN F-43m -> I-42d,
        # whose whole distortion is 0.0002 A against the 0.01 A default):
        # the analysis must reproduce AMPLIMODES once the tolerance is tight
        # enough to see it, reading the POSCARs directly
        gan = os.path.join(example, "debug5_GaN")
        code, out = run_group(
            ["--supergroup-cif", os.path.join(gan, "POSCAR-finish_GaN_F-43m"),
             "--subgroup-cif", os.path.join(gan, "POSCAR-finish_GaN_I-42d"),
             "--tolerance", "0.0001"],
            cwd=tmp,
        )
        report("GaN F-43m -> I-42d: 0.0004 A W1 mode at --tolerance 0.0001",
               code == 0 and "I-42d (No. 122)" in out
               and re.search(r"W1\s+\S+\s+122 I-42d\s+1\s+0\.0004", out)
               is not None
               and "maximum atomic displacement: 0.0002 A" in out, out)

        # non-centrosymmetric cubic parent with an M-point order parameter
        # (CdC2N2 P-43m -> C222, index 4).  Structures rebuilt from the
        # AMPLIMODES output's own listing in
        # debug7_CdC2N2/Symmetry_mode_analysis_CdC2N2_P-43m.pdf.
        cdc2n2 = os.path.join(example, "debug7_CdC2N2")
        code, out = run_group(
            ["--supergroup-cif",
             os.path.join(cdc2n2, "POSCAR_CdC2N2_P-43m_bilbao.cif"),
             "--subgroup-cif",
             os.path.join(cdc2n2, "POSCAR_CdC2N2_C222_bilbao.cif")],
            cwd=tmp,
        )
        report("CdC2N2 P-43m -> C222: M5 0.1391 A with two ~zero Gamma "
               "secondaries (AMPLIMODES values)",
               code == 0
               and re.search(r"M5\s+\S+\s+21 C222\s+8\s+0\.1391", out)
               is not None
               and re.search(r"GM1\s+\S+\s+215 P-43m\s+2\s+0\.0001", out)
               is not None
               and re.search(r"GM3\s+\S+\s+16 P222\s+4\s+0\.0001", out)
               is not None
               and "maximum atomic displacement: 0.0319 A" in out, out)

        # the AlF3 tilt as AMPLIMODES itself was given it (rebuilt from
        # debug6_AlF3/Symmetry_mode_analysis_AlF3_Pm-3m_R-3c.pdf); the
        # 221/167_PPOSCAR_AlF3 pair above is a different, more strongly
        # tilted refinement of the same transition
        alf3 = os.path.join(example, "debug6_AlF3")
        code, out = run_group(
            ["--supergroup-cif",
             os.path.join(alf3, "POSCAR_AlF3_Pm-3m_bilbao.cif"),
             "--subgroup-cif",
             os.path.join(alf3, "POSCAR_AlF3_R-3c_bilbao.cif")],
            cwd=tmp,
        )
        report("AlF3 Pm-3m -> R-3c: R4+ 0.2881 A, max 0.1176 A "
               "(AMPLIMODES values)",
               code == 0
               and re.search(r"R4\+\s+\S+\s+167 R-3c\s+1\s+0\.2881", out)
               is not None
               and "maximum atomic displacement: 0.1176 A" in out
               and "total distortion amplitude : 0.2881 A" in out, out)

        # non-special (symmetry line) folding k points: the Pbam
        # antiferroelectric of PbZrO3 condenses SM = (1/4,1/4,0) and
        # S = (1/4,1/2,1/4) modes, which no special-point table carries;
        # the small irreps come from spgrep at the exact k, the names and
        # matrices from the bundled ISO-IR tables (AMPLIMODES reference:
        # debug2/Symmetry_mode_analysis_PbZrO3_Pm-3m_Pbam.pdf)
        debug2 = os.path.join(example, "debug2_PbZrO3")
        code, out = run_group(
            ["--supergroup-cif",
             os.path.join(debug2, "POSCAR_PbZrO3_Pm-3m.cif"),
             "--subgroup-cif",
             os.path.join(debug2, "POSCAR_PbZrO3_Pbam.cif")],
            cwd=tmp,
        )
        report("PbZrO3 Pm-3m -> Pbam (index 8, line k points) decomposes",
               code == 0 and "Pbam (No. 55)" in out
               and "primitive cell multiplication: 8" in out, out)
        report("SM2 at (1/4,1/4,0): Pbam, dim 5, amplitude 1.2871 A "
               "(AMPLIMODES value)",
               "(1/4,1/4,0)" in out
               and re.search(r"SM2\s+\S+\s+55 Pbam\s+5\s+1\.2871", out)
               is not None, out)
        report("S4 at (1/4,1/2,1/4): Cmce, dim 3, amplitude 0.4470 A",
               "(1/4,1/2,1/4)" in out
               and re.search(r"S4\s+\S+\s+64 Cmce\s+3\s+0\.4470", out)
               is not None, out)
        report("special-point companions: R4+ 1.5232, R5+ 0.0810, X3- 0.0374, "
               "M5- 0.0473 A",
               re.search(r"R4\+\s+\S+\s+74 Imma\s+1\s+1\.5232", out) is not None
               and re.search(r"R5\+\s+\S+\s+74 Imma\s+2\s+0\.0810", out)
               is not None
               and re.search(r"X3-\s+\S+\s+123 P4/mmm\s+2\s+0\.0374", out)
               is not None
               and re.search(r"M5-\s+\S+\s+51 Pmma\s+3\s+0\.0473", out)
               is not None, out)
        report("max displacement 0.4823 A and total distortion 2.0462 A "
               "(AMPLIMODES values)",
               "maximum atomic displacement: 0.4823 A" in out
               and "total distortion amplitude : 2.0462 A" in out, out)
        report("line-mode VESTA files written (SM2 and S4)",
               os.path.isfile(
                   os.path.join(tmp, "POSCAR_PbZrO3_Pm-3m_SM2.vesta")
               ) and os.path.isfile(
                   os.path.join(tmp, "POSCAR_PbZrO3_Pm-3m_S4.vesta")
               ), out)
        # measured, not assumed: freezing ONE mode's displacement field into
        # the reference structure must land exactly on the isotropy subgroup
        # the table claims for it
        import numpy as _np

        import spglib as _spglib

        from crystod.symmetry_mode import SymmetryModeAnalysis as _SMA
        _analysis = _SMA(
            os.path.join(debug2, "POSCAR_PbZrO3_Pm-3m.cif"),
            os.path.join(debug2, "POSCAR_PbZrO3_Pbam.cif"), 0.01)
        _lattice = _analysis.S_core @ _analysis.L_parent
        _S_inv = _np.linalg.inv(_analysis.S_core)
        _base = _np.mod(_analysis.ref_frac @ _S_inv, 1.0)
        _frozen_ok = True
        _claims = {}
        for _mode in _analysis.modes:
            _, _info, _ = _mode.label_info()
            _claims[_mode.irrep_name] = _info.number
            _disp = _mode.projected_u
            _peak = _np.max(_np.linalg.norm(_disp, axis=1))
            _frac = (_disp * (0.2 / _peak)) @ _np.linalg.inv(
                _analysis.L_parent) @ _S_inv
            _sg = _spglib.get_spacegroup(
                (_lattice, _base + _frac, _analysis.ref_z), symprec=1e-4)
            if int(_sg.split("(")[1].rstrip(")")) != _info.number:
                _frozen_ok = False
        report("every frozen single-mode field measures as its claimed "
               "subgroup (spglib)",
               _frozen_ok
               and _claims.get("SM2") == 55 and _claims.get("S4") == 64,
               str(_claims))

        # a line star whose ISO-IR matrices cannot be realified (from_isoir
        # stops with SystemExit) falls back on the spgrep basis instead of
        # stopping the analysis: same dimensions, subgroups and amplitudes
        import crystod.symmetry_mode as _sm_module

        def _refuse(*args, **kwargs):
            raise SystemExit("ERROR: could not realify (test)")

        _cir = _sm_module.ComputedInducedRepresentation
        _from_isoir = _cir.__dict__["from_isoir"]
        _cir.from_isoir = _refuse
        try:
            _fallback = _SMA(
                os.path.join(debug2, "POSCAR_PbZrO3_Pm-3m.cif"),
                os.path.join(debug2, "POSCAR_PbZrO3_Pbam.cif"), 0.01)
            _fallback_rows = [
                (_sm_module._kvector_string(m.kvec), m.dim,
                 m.label_info()[1].number, m.amplitude)
                for m in _fallback.modes
            ]
        except SystemExit as _exc:
            _fallback_rows = [str(_exc)]
        finally:
            _cir.from_isoir = _from_isoir
        _rows = [(_sm_module._kvector_string(m.kvec), m.dim,
                  m.label_info()[1].number, m.amplitude)
                 for m in _analysis.modes]
        report("PbZrO3: computed stars fall back on the spgrep basis when "
               "the ISO-IR one cannot be realified (same modes)",
               len(_fallback_rows) == len(_rows) == 6
               and all(len(a) == 4 and a[:3] == b[:3]
                       and abs(a[3] - b[3]) < 1e-9
                       for a, b in zip(_fallback_rows, _rows)),
               f"{_fallback_rows} vs {_rows}")
        report("decomposition table saved as sym_mode_PbZrO3",
               os.path.isfile(os.path.join(tmp, "sym_mode_PbZrO3"))
               and "Decomposition table saved to sym_mode_PbZrO3" in out
               and "SM2" in open(os.path.join(tmp, "sym_mode_PbZrO3")).read(),
               out)

        # a hexagonal line star (DT of P6/mmm at (0,0,1/4), a 4x cell along c;
        # synthetic GaN2 pair): the DT6 mode of P2_1 with the amplitude of the
        # whole distortion
        code, out = run_group(
            ["--supergroup-cif", os.path.join(example, "POSCAR_GaN2_P6mmm"),
             "--subgroup-cif", os.path.join(example, "POSCAR_GaN2_P21"), "--no-files"],
            cwd=tmp,
        )
        report("GaN2 P6/mmm -> P2_1 (hexagonal DT line, 4x along c): DT6 0.0949 A",
               code == 0
               and re.search(r"\(0,0,1/4\)\s+DT6\s+\(a,b;c,d\)\s+4 P2_1\s+8\s+0\.0949", out)
               is not None
               and "total distortion amplitude : 0.0949 A" in out, out)

        # K2SeO4 Pnma -> Pna2_1 (the incommensurate lock-in ferroelectric):
        # a 3x cell along a with the SM point at (1/3,0,0), plus a polar
        # GM4- (AMPLIMODES reference PDF in example/23_symmetry_mode)
        code, out = run_group(
            ["--supergroup-cif",
             os.path.join(example, "POSCAR_K2SeO4_Pnma.cif"),
             "--subgroup-cif",
             os.path.join(example, "POSCAR_K2SeO4_Pna21.cif")],
            cwd=tmp,
        )
        report("K2SeO4 Pnma -> Pna2_1 (SM at (1/3,0,0)) decomposes",
               code == 0 and "Pna21 (No. 33)" in out
               and "primitive cell multiplication: 3" in out, out)
        report("K2SeO4 amplitudes match AMPLIMODES (GM1+ 0.9467, GM4- "
               "0.4230, SM2 1.2885, SM3 0.1727 A)",
               re.search(r"GM1\+\s+\S+\s+62 Pnma\s+13\s+0\.9467", out)
               is not None
               and re.search(r"GM4-\s+\S+\s+33 Pna2_1\s+8\s+0\.4230", out)
               is not None
               and re.search(r"SM2\s+\S+\s+33 Pna2_1\s+16\s+1\.2885", out)
               is not None
               and re.search(r"SM3\s+\S+\s+62 Pnma\s+26\s+0\.1727", out)
               is not None, out)
        report("K2SeO4 max displacement 0.3813 A and total 1.6629 A "
               "(AMPLIMODES values)",
               "maximum atomic displacement: 0.3813 A" in out
               and "total distortion amplitude : 1.6629 A" in out, out)
        sym_path = os.path.join(tmp, "sym_mode_K2SeO4")
        sym_text = open(sym_path).read() if os.path.isfile(sym_path) else ""
        report("table saved as sym_mode_K2SeO4 (parent-composition name), "
               "content = the printed table",
               os.path.isfile(sym_path)
               and "* Symmetry-mode decomposition *" in sym_text
               and "(1/3,0,0)" in sym_text and "SM2" in sym_text
               and sym_text.strip() in out.replace("\r", ""), out)

        # per-irrep VESTA export: the multi-mode CaTiO3 case
        code, out = run_group(
            ["--supergroup-cif",
             os.path.join(example, "221_PPOSCAR_CaTiO3.cif"),
             "--subgroup-cif",
             os.path.join(example, "62_PPOSCAR_CaTiO3.cif")],
            cwd=tmp,
        )
        report("CaTiO3 writes one VESTA file per activated irrep",
               code == 0 and all(
                   os.path.isfile(
                       os.path.join(tmp, f"221_PPOSCAR_CaTiO3_{irrep}.vesta")
                   )
                   for irrep in ("X5+", "M2+", "M3+", "R4+", "R5+")
               ), out)
        vesta_text = open(
            os.path.join(tmp, "221_PPOSCAR_CaTiO3_R4+.vesta")
        ).read()
        report("R4+ VESTA carries arrow entries (VECTR section)",
               "VECTR" in vesta_text and "VECTT" in vesta_text, vesta_text[:500])

        # --conventional: mode VESTA files in the parent conventional basis
        code, out = run_group(
            ["--supergroup-cif",
             os.path.join(example, "139_PPOSCAR_La3Ni2O7.cif"),
             "--subgroup-cif",
             os.path.join(example, "63_PPOSCAR_La3Ni2O7.cif"),
             "--conventional"],
            cwd=tmp,
        )
        report("--conventional writes _conv VESTA files (I4/mmm parent)",
               code == 0
               and os.path.isfile(
                   os.path.join(tmp, "139_PPOSCAR_La3Ni2O7_X3-_conv.vesta")
               )
               and os.path.isfile(
                   os.path.join(tmp, "139_PPOSCAR_La3Ni2O7_GM1+_conv.vesta")
               )
               and "parent conventional basis" in out, out)
        conv_text = open(
            os.path.join(tmp, "139_PPOSCAR_La3Ni2O7_X3-_conv.vesta")
        ).read()
        report("_conv cell has the conventional 90-degree metric",
               " 90.000000  90.000000  90.000000" in conv_text
               and "20.070400" in conv_text, conv_text[:400])

        # polar subgroup: acoustic (free-origin) component removed
        code, out = run_group(["--supergroup-cif",
                               os.path.join(example, "221_PPOSCAR_BaTiO3.cif"),
                               "--subgroup-cif",
                               os.path.join(example, "99_PPOSCAR_BaTiO3.cif")],
                              cwd=tmp)
        report("BaTiO3 Pm-3m -> P4mm: polar GM4- with minimum-distortion origin",
               code == 0 and "(0,0,0)" in out
               and re.search(r"GM4-\s+\S+\s+99 P4mm\s+4\s+0\.2032", out)
               is not None, out)

        # the sublattice setting follows the orientation of the input files:
        # a cubic parent leaves 24 symmetry-equivalent settings tied on the
        # distortion, and the enumeration order used to pick one at random
        # (reported on PbTiO3 Pm-3m -> P4mm, example/23_symmetry_mode/
        # debug8_PbTiO3: the polarization along c of the P4mm CIF came out
        # along -b of the parent in the displacement table and in the GM4-
        # VESTA file).  The rows must now be the identity and every arrow
        # must lie along c.
        report("BaTiO3 child basis is the identity (c stays c)",
               "  (1, 0, 0)\n  (0, 1, 0)\n  (0, 0, 1)" in out, out)
        pbtio3_parent = os.path.join(tmp, "POSCAR_PbTiO3_Pm-3m")
        pbtio3_child = os.path.join(tmp, "POSCAR_PbTiO3_P4mm")
        pbtio3_child_y = os.path.join(tmp, "POSCAR_PbTiO3_P4mm_polar_y")
        with open(pbtio3_parent, "w") as handle:
            handle.write(
                "PbTiO3 Pm-3m\n1.0\n 3.9220 0 0\n 0 3.9220 0\n 0 0 3.9220\n"
                "Pb Ti O\n1 1 3\nDirect\n0 0 0\n0.5 0.5 0.5\n"
                "0.5 0 0.5\n0 0.5 0.5\n0.5 0.5 0\n"
            )
        with open(pbtio3_child, "w") as handle:
            handle.write(
                "PbTiO3 P4mm, polar axis c\n1.0\n 3.8924 0 0\n 0 3.8924 0\n"
                " 0 0 4.1048\nPb Ti O\n1 1 3\nDirect\n0 0 0.11044\n"
                "0.5 0.5 0.57093\n0.5 0 0.49782\n0 0.5 0.49782\n"
                "0.5 0.5 0.01319\n"
            )
        with open(pbtio3_child_y, "w") as handle:
            # the same structure with its polar axis along the SECOND
            # lattice vector (Cartesian y): the orientation must follow
            # the Cartesian frame of the file, not the axis labels
            handle.write(
                "PbTiO3 P4mm, polar axis along y\n1.0\n 3.8924 0 0\n"
                " 0 4.1048 0\n 0 0 3.8924\nPb Ti O\n1 1 3\nDirect\n"
                "0 0.11044 0\n0.5 0.57093 0.5\n0.5 0.01319 0.5\n"
                "0 0.49782 0.5\n0.5 0.49782 0\n"
            )

        def _vesta_arrows(path):
            text = open(path).read()
            block = text.split("VECTR", 1)[1].split("VECTT", 1)[0]
            arrows = []
            for line in block.splitlines():
                parts = line.split()
                if len(parts) == 4 and "." in parts[1]:
                    arrows.append([float(v) for v in parts[1:]])
            return np.array(arrows)

        code, out = run_group(["--supergroup-cif", pbtio3_parent,
                               "--subgroup-cif", pbtio3_child], cwd=tmp)
        arrows = _vesta_arrows(os.path.join(tmp, "POSCAR_PbTiO3_Pm-3m_GM4-.vesta"))
        report("PbTiO3 Pm-3m -> P4mm: identity setting, GM4- 0.3956 A, "
               "Pb/Ti displaced along +c",
               code == 0
               and "  (1, 0, 0)\n  (0, 1, 0)\n  (0, 0, 1)" in out
               and re.search(r"GM4-\s+\S+\s+99 P4mm\s+4\s+0\.3956", out)
               is not None
               and re.search(r"Pb\s+\(0\.00000,0\.00000,0\.00000\)"
                             r"\(\+0\.00000,\+0\.00000,\+0\.07240\)", out)
               is not None, out)
        report("PbTiO3 GM4- VESTA arrows all lie along c (five atoms)",
               len(arrows) == 5
               and np.all(np.abs(arrows[:, :2]) < 1e-6)
               and np.all(np.abs(arrows[:, 2]) > 0.3), str(arrows))
        code, out = run_group(["--supergroup-cif", pbtio3_parent,
                               "--subgroup-cif", pbtio3_child_y], cwd=tmp)
        arrows = _vesta_arrows(os.path.join(tmp, "POSCAR_PbTiO3_Pm-3m_GM4-.vesta"))
        report("polar axis along Cartesian y in the POSCAR: displacements and "
               "arrows follow the file's frame (along b)",
               code == 0
               and re.search(r"GM4-\s+\S+\s+99 P4mm\s+4\s+0\.3956", out)
               is not None
               and re.search(r"Pb\s+\(0\.00000,0\.00000,0\.00000\)"
                             r"\(\+0\.00000,\+0\.07240,\+0\.00000\)", out)
               is not None
               and len(arrows) == 5
               and np.all(np.abs(arrows[:, [0, 2]]) < 1e-6)
               and np.all(np.abs(arrows[:, 1]) > 0.3), out + str(arrows))

        # strongly tilted child (14% lattice strain; displaced-species anchor)
        code, out = run_group(["--supergroup-cif",
                               os.path.join(example, "221_PPOSCAR_AlF3.cif"),
                               "--subgroup-cif",
                               os.path.join(example, "167_PPOSCAR_AlF3.cif")],
                              cwd=tmp)
        report("AlF3 Pm-3m -> R-3c: large-tilt R4+ (index 2, strained lattice)",
               code == 0
               and re.search(r"R4\+\s+\S+\s+167 R-3c\s+1\s+0\.80", out)
               is not None, out)

        # cross-checks against crystod-phonon --modulation structures (section 33)
        modulation = os.path.join(ROOT, "example", "33_modulation", "ScF3_Pm-3m")
        parent_scf3 = os.path.join(modulation, "221_PPOSCAR_ScF3")

        code, out = run_group(["--supergroup-cif", parent_scf3,
                               "--subgroup-cif",
                               os.path.join(modulation, "POSCAR_R-3c")],
                              cwd=tmp)
        report("ScF3 R-3c: R4+ (a,a,a) -> 167 R-3c (modulation cross-check)",
               code == 0 and re.search(r"R4\+\s+\(a,a,a\)\s+167 R-3c", out)
               is not None, out)

        code, out = run_group(["--supergroup-cif", parent_scf3,
                               "--subgroup-cif",
                               os.path.join(modulation, "POSCAR_Pbnm")],
                              cwd=tmp)
        report("ScF3 Pbnm: two active modes R4+ -> Imma + M3+ -> P4/mbm",
               code == 0
               and re.search(r"R4\+\s+\S+\s+74 Imma\s+1\s+0\.8485", out) is not None
               and re.search(r"M3\+\s+\S+\s+127 P4/mbm\s+1\s+0\.6000", out)
               is not None, out)
        report("ScF3 Pbnm: inactive secondary X5+ listed with amplitude 0",
               re.search(r"X5\+\s+\S+\s+63 Cmcm\s+1\s+0\.0000", out) is not None, out)

        # complex-type irreps enter as their physically irreducible pair
        # D + D* on one line (label R1+R3+): before v0.4.3 the doubled
        # dimension of such an irrep made the projector of every folding
        # star that carries one crash, active or not.  Pyrite-type FeS2
        # (Pa-3) with seeded random displacements (0.03 A rms) in the
        # index-2 fcc-type supercell, stars GM and R: R1+R3+ / R1-R3- and
        # GM2+GM3+ / GM2-GM3- are complex pairs, R2+ / R2- pseudoreal (one
        # name, already real)
        import numpy as _np

        import spglib as _spglib

        from crystod.symmetry_mode import SymmetryModeAnalysis as _SMA
        fes2 = os.path.join(example, "POSCAR_FeS2_Pa-3")
        fes2_child = os.path.join(example, "POSCAR_FeS2_P1_fcc2")
        code, out = run_group(["--supergroup-cif", fes2, "--subgroup-cif",
                               fes2_child, "--no-files"], cwd=tmp)
        report("FeS2 Pa-3 -> P1 (fcc-type 2x cell): complex pairs R1+R3+ "
               "(dim 8, 0.0881 A), R1-R3- (16, 0.1059), GM2+GM3+ -> Pbca "
               "(0.0245), GM2-GM3- (0.0308), one line each",
               code == 0
               and re.search(r"R1\+R3\+\s+\S+\s+2 P-1\s+8\s+0\.0881", out)
               is not None
               and re.search(r"R1-R3-\s+\S+\s+2 P-1\s+16\s+0\.1059", out)
               is not None
               and re.search(r"GM2\+GM3\+\s+\(a,b\)\s+61 Pbca\s+2\s+0\.0245",
                             out) is not None
               and re.search(r"GM2-GM3-\s+\S+\s+19 P2_12_12_1\s+4\s+0\.0308",
                             out) is not None
               and re.search(r"^\S+\s+(R1\+|R3\+|GM2\+|GM3\+)\s", out,
                             re.MULTILINE) is None, out)
        report("FeS2: pseudoreal R2+ (dim 4, 0.0992 A) and R2- (dim 8, "
               "0.0554 A) keep one name; total 0.2540 A",
               re.search(r"R2\+\s+\S+\s+2 P-1\s+4\s+0\.0992", out) is not None
               and re.search(r"R2-\s+\S+\s+2 P-1\s+8\s+0\.0554", out)
               is not None
               and "total distortion amplitude : 0.2540 A" in out, out)
        _fes2 = _SMA(fes2, fes2_child, 0.01)
        _amplitudes = [mode.amplitude for mode in _fes2.modes]
        report("FeS2: the squared amplitudes add up to the squared total "
               "distortion",
               abs(_np.sqrt(_np.sum(_np.square(_amplitudes)))
                   - _fes2.total_distortion) < 1e-9
               and abs(_fes2.total_distortion - 0.2540) < 5e-5,
               f"{_amplitudes} {_fes2.total_distortion}")
        # freezing one pair's displacement field into the reference lands on
        # the isotropy subgroup the table claims for it
        _lattice = _fes2.S_core @ _fes2.L_parent
        _S_inv = _np.linalg.inv(_fes2.S_core)
        _base = _np.mod(_fes2.ref_frac @ _S_inv, 1.0)
        _frozen = {}
        for _mode in _fes2.modes:
            if len(re.findall(r"[A-Z]+\d+[+-]?", _mode.irrep_name)) != 2:
                continue
            _, _info, _ = _mode.label_info()
            _disp = _mode.projected_u
            _frac = (_disp * (0.2 / _np.max(_np.linalg.norm(_disp, axis=1)))) \
                @ _np.linalg.inv(_fes2.L_parent) @ _S_inv
            _sg = _spglib.get_spacegroup(
                (_lattice, _base + _frac, _fes2.ref_z), symprec=1e-4)
            _frozen[_mode.irrep_name] = (
                _info.number, int(_sg.split("(")[1].rstrip(")")))
        report("FeS2: every frozen pair field measures as its claimed "
               "subgroup (spglib)",
               len(_frozen) == 4
               and all(claim == found for claim, found in _frozen.values()),
               str(_frozen))

        # a polar line (LD of P4) and the general point of P1: the stars of k
        # and -k differ, and the pair carries ISOTROPY's physically
        # irreducible label LD1LE1 / GP1GQ1 (it was LD1LDA1 / GP1GPA1);
        # seeded random displacements in a 1x1x3 (P4) and 3x1x1 (P1) cell
        from phonopy.interface.vasp import write_vasp as _write_vasp
        from phonopy.structure.atoms import PhonopyAtoms as _Atoms

        _rng = _np.random.default_rng(7)

        def _pair_case(name, lattice, positions, numbers, dim):
            lattice = _np.asarray(lattice, dtype=float)
            positions = _np.asarray(positions, dtype=float)
            parent_file = os.path.join(tmp, f"POSCAR_{name}_parent")
            child_file = os.path.join(tmp, f"POSCAR_{name}_child")
            _write_vasp(parent_file, _Atoms(cell=lattice, numbers=numbers,
                                            scaled_positions=positions))
            n = _np.array(dim)
            cells = [(i, j, k) for i in range(n[0]) for j in range(n[1])
                     for k in range(n[2])]
            child = _np.concatenate([(positions + t) / n for t in cells])
            child_lattice = _np.diag(n) @ lattice
            child = child + _rng.normal(scale=0.01, size=child.shape) \
                / _np.linalg.norm(child_lattice, axis=1)
            _write_vasp(child_file, _Atoms(
                cell=child_lattice, scaled_positions=child % 1.0,
                numbers=list(numbers) * len(cells)))
            return run_group(["--supergroup-cif", parent_file, "--subgroup-cif",
                              child_file, "--no-files"], cwd=tmp)

        _x, _y, _z = 0.21, 0.13, 0.17
        code, out = _pair_case(
            "P4", _np.diag([4.0, 4.0, 3.0]),
            [[0, 0, 0], [0.5, 0.5, 0.4], [_x, _y, _z], [-_x, -_y, _z],
             [-_y, _x, _z], [_y, -_x, _z]], [8, 11, 1, 1, 1, 1], (1, 1, 3))
        report("P4 -> 1x1x3 P1: the LD/LE pairs of the polar line are "
               "LD1LE1 ... LD4LE4 at (0,0,1/3)",
               code == 0
               and all(re.search(rf"\(0,0,1/3\)\s+LD{n}LE{n}\s", out)
                       for n in (1, 2, 3, 4))
               and "LDA" not in out, out)
        code, out = _pair_case(
            "P1", [[4.0, 0.1, 0.2], [0.3, 4.5, 0.1], [0.2, 0.4, 5.0]],
            [[0, 0, 0], [0.31, 0.42, 0.13], [0.7, 0.2, 0.55]], [8, 11, 1],
            (3, 1, 1))
        report("P1 -> 3x1x1: the general point pairs as GP1GQ1 at (1/3,0,0)",
               code == 0 and re.search(r"\(1/3,0,0\)\s+GP1GQ1\s", out)
               is not None and "GPA" not in out, out)

        # the minimum-distortion atom mapping need not be a group-subgroup
        # setting: for KGeCl3 (P4/mmm pseudo-parent -> Cm, Materials Project
        # mp-998231, from a structure survey) the least-distortion sublattice is
        # a twin setting in which the child mirror is no parent operation,
        # and the run stopped with "not fully captured".  The mapping is
        # redone in the setting in which every child operation is a parent
        # operation, and a NOTE says so
        kgecl3 = os.path.join(example, "POSCAR_KGeCl3_P4mmm_pseudo")
        kgecl3_child = os.path.join(example, "POSCAR_KGeCl3_Cm")
        code, out = run_group(["--supergroup-cif", kgecl3, "--subgroup-cif",
                               kgecl3_child, "--no-files"], cwd=tmp)
        report("KGeCl3 P4/mmm -> Cm (twin setting): mapping redone in a "
               "group-subgroup setting, NOTE printed",
               code == 0 and "Cm (No. 8)" in out
               and "NOTE: the minimum-distortion atom mapping is not a "
               "group-subgroup" in out and "please report" not in out
               and "1 of the 2 operations of the child is a parent "
               "operation" in " ".join(out.split()), out)
        report("KGeCl3: GM3- (a) -> P4mm 0.1702 A, GM5- (a,a) -> Amm2 "
               "0.2482 A, total 0.3010 A",
               re.search(r"GM3-\s+\(a\)\s+99 P4mm\s+4\s+0\.1702", out)
               is not None
               and re.search(r"GM5-\s+\(a,a\)\s+38 Amm2\s+5\s+0\.2482", out)
               is not None
               and "maximum atomic displacement: 0.1876 A" in out
               and "total distortion amplitude : 0.3010 A" in out, out)
        _kg = _SMA(kgecl3, kgecl3_child, 0.01)
        _lattice = _kg.S_core @ _kg.L_parent
        _frac = (_kg.ref_frac + _kg.u_cart @ _np.linalg.inv(_kg.L_parent)) \
            @ _np.linalg.inv(_kg.S_core)
        _sg = _spglib.get_spacegroup(
            (_lattice, _np.mod(_frac, 1.0), _kg.ref_z), symprec=1e-3)
        report("KGeCl3: reference + displacement field is Cm itself "
               "(spglib), the members are the two child operations",
               _sg == "Cm (8)" and len(_kg.subgroup_members) == 2
               and _kg.mapping_note is not None,
               f"{_sg} {len(_kg.subgroup_members)}")

        # the printed origin shift p must place the child atoms (child
        # primitive fractional @ S + p) exactly on the stored reference +
        # displacement positions; two bookkeeping slips used to break it by
        # a uniform shift (a rejected origin-refinement step, and the
        # removal of the uniform polar displacement without moving p)
        from itertools import product as _product

        def _origin_gap(an):
            S = _np.asarray(an.mapping.S, dtype=float)
            S_inv = _np.linalg.inv(S)
            _, positions, numbers = an.child_prim
            placed = (_np.asarray(positions, dtype=float) @ S
                      + _np.asarray(an.mapping.p, dtype=float))
            numbers = _np.array([int(z) for z in numbers])
            stored = _np.asarray(an.mapping.child_frac, dtype=float)
            stored_z = _np.asarray(an.mapping.child_z)
            images = _np.array(list(_product((-1, 0, 1), repeat=3)),
                               dtype=float) @ S
            worst = 0.0
            for j in range(len(stored)):
                d = placed[numbers == stored_z[j]] - stored[j]
                d = d - _np.rint(d @ S_inv) @ S
                lengths = _np.linalg.norm(
                    (d[:, None, :] + images[None]) @ an.L_parent, axis=-1)
                worst = max(worst, float(lengths.min()))
            return worst

        _bt = _SMA(os.path.join(example, "221_PPOSCAR_BaTiO3.cif"),
                   os.path.join(example, "99_PPOSCAR_BaTiO3.cif"), 0.01)
        _gaps = [_origin_gap(_kg), _origin_gap(_bt)]
        # a uniform 0.05 A displacement along the polar axis of P4mm, with
        # the origin moved along (a consistent state, as a pairing leaves
        # it); taking it out of the displacements must take it out of p
        _rows = [_bt.L_parent.T @ _bt.algebra.rotations[i]
                 @ _np.linalg.inv(_bt.L_parent.T) - _np.eye(3)
                 for i, _ in _bt.subgroup_members]
        _axis = _np.linalg.svd(_np.vstack(_rows))[2][-1]
        _shift = (0.05 * _axis) @ _np.linalg.inv(_bt.L_parent)
        _p0 = _np.asarray(_bt.mapping.p, dtype=float).copy()
        _bt.mapping.p = _p0 + _shift
        _bt.mapping.u_frac = _bt.mapping.u_frac + _shift
        _bt.mapping.child_frac = _bt.mapping.child_frac + _shift
        _bt.core_u_frac = _bt.core_u_frac + _shift
        _bt.u_cart = _bt.core_u_frac @ _bt.L_parent
        _bt.core_child_frac = _bt.ref_frac + _bt.core_u_frac
        _gaps.append(_origin_gap(_bt))
        _bt._remove_acoustic_offset()
        _gaps.append(_origin_gap(_bt))
        report("origin shift places the child atoms on the stored positions "
               "(KGeCl3 redone mapping, BaTiO3 P4mm, and BaTiO3 after the "
               "uniform polar displacement is taken out)",
               max(_gaps) < 1e-9
               and _np.allclose(_bt.mapping.p, _p0, atol=1e-9),
               str(_gaps))

        # no group-subgroup consistent mapping at all: a specific message,
        # no traceback, no "please report" (SeO2 Pmc2_1, mp-560882, against
        # a Cmme pseudo-parent from the same survey: the parent mirror that
        # the child keeps fixes 8 O sites of the reference but 4 O atoms)
        code, out = run_group(
            ["--supergroup-cif", os.path.join(example, "POSCAR_SeO2_Cmme_pseudo"),
             "--subgroup-cif", os.path.join(example, "POSCAR_SeO2_Pmc21"),
             "--no-files"], cwd=tmp)
        report("SeO2 Cmme pseudo-parent -> Pmc2_1 stops with the Wyckoff-"
               "conflict message",
               code != 0 and "the Wyckoff splitting disagrees" in out
               and "fixes 8 O site(s) of the reference structure but 4 O "
               "atom(s) of the child" in out
               and "Traceback" not in out and "please report" not in out
               and "different --tolerance" not in out, out)

    # errors
    code, out = run_group(["--supergroup-cif", parent])
    report("--supergroup-cif without --subgroup-cif rejected cleanly",
           code != 0 and "requires --subgroup-cif" in out
           and "Traceback" not in out, out)

    code, out = run_group(["--supergroup-cif", parent,
                           "--subgroup-cif", "no_such_file.cif"])
    report("missing structure file rejected cleanly",
           code != 0 and "not found" in out and "Traceback" not in out, out)

    code, out = run_group(["--product", "T2g", "T2g", "--pg", "m-3m",
                           "--subgroup-cif", child])
    report("--subgroup-cif outside --supergroup-cif rejected cleanly",
           code != 0 and "only used with --supergroup-cif" in out
           and "Traceback" not in out, out)


# ---------------------------------------------------------------- 43. Python API
def test_43_python_api() -> None:
    print("\n[43] Python API (crystod.salc / group / phonon / bz / mag / md / mol / xrd / search)")

    # import hygiene: plain `import crystod` must not drag in the heavy stack
    probe = (
        "import sys, crystod\n"
        "import crystod.salc, crystod.group, crystod.phonon\n"
        "import crystod.bz, crystod.mag, crystod.md, crystod.mol\n"
        "import crystod.xrd, crystod.search\n"
        "heavy = [m for m in ('phonopy', 'spgrep', 'pyscf', 'pymatgen', 'matplotlib',\n"
        "                     'requests')\n"
        "         if m in sys.modules]\n"
        "print('HEAVY:' + ','.join(heavy))\n"
        "print('VERSION:' + crystod.__version__)\n"
    )
    code, out = run_python(probe)
    report("importing crystod and all API domains exit 0", code == 0, out)
    report("no heavy dependency imported by the API namespaces",
           "HEAVY:\n" in out or "HEAVY:" in out and out.split("HEAVY:")[1].startswith("\n"),
           out)
    import crystod as _crystod
    report("crystod.__version__ exposed", f"VERSION:{_crystod.__version__}" in out, out)

    # backward compatibility: the pre-0.3.6 import paths still work
    probe = (
        "from crystod import wigner_D_real\n"
        "from crystod.operations import find_star_arm\n"
        "from crystod.phonon_irreps import get_irrep_labels\n"
        "from crystod.isotropy_subgroup import IsotropyAnalyzer\n"
        "from crystod.cli.md import _parse_dim, build_parser\n"
        "print('OK')\n"
    )
    code, out = run_python(probe)
    report("pre-0.3.6 module import paths still work", code == 0 and "OK" in out, out)

    # the API and the CLI must enumerate the same isotropy subgroups
    probe = (
        "from crystod.group import isotropy_subgroups\n"
        "for s in isotropy_subgroups('Pm-3m', 'R4+'):\n"
        "    print(f'{s.label} {s.number} {s.symbol} {s.size} {s.index}')\n"
    )
    code, api_out = run_python(probe)
    report("crystod.group.isotropy_subgroups exit 0", code == 0, api_out)
    for direction, number, symbol in (("R4+(0,0,a)", "140", "I4/mcm"),
                                      ("R4+(a,a,a)", "167", "R-3c"),
                                      ("R4+(0,a,a)", "74", "Imma"),
                                      ("R4+(a,b,c)", "2", "P-1")):
        report(f"API subgroup {direction} -> {symbol}",
               f"{direction} {number} {symbol}" in api_out, api_out)
    code, cli_out = run_group(["--parent", "Pm-3m", "--irrep", "R4+"])
    api_pairs = {line.split()[0]: line.split()[2]
                 for line in api_out.splitlines() if line.startswith("R4+(")}
    cli_pairs = {line.split()[0]: line.split()[2]
                 for line in cli_out.splitlines() if line.startswith("R4+(")}
    report("API and crystod-group --parent agree on every direction",
           code == 0 and api_pairs and api_pairs == cli_pairs,
           f"api={api_pairs}\ncli={cli_pairs}")

    # every irrep of one k point: the API form of --parent SG --kpoint K
    probe = (
        "import crystod\n"
        "from crystod.group import (isotropy_subgroups, isotropy_subgroups_at_kpoint,\n"
        "                           KpointIsotropySubgroups)\n"
        "table = isotropy_subgroups_at_kpoint('Pm-3m', 'GM')\n"
        "print('KEYS', ' '.join(table))\n"
        "print('META', type(table).__name__, isinstance(table, dict), table.kpoint,\n"
        "      table.coordinates, table.n_arms, table.space_group,\n"
        "      table.space_group_number, table.errors)\n"
        "print('SAME', all(subs == isotropy_subgroups('Pm-3m', label)\n"
        "                  for label, subs in table.items()),\n"
        "      sum(len(subs) for subs in table.values()))\n"
        "print('BASIS', all(s.basis is not None for subs in table.values() for s in subs))\n"
        "for subs in table.values():\n"
        "    for s in subs:\n"
        "        print(f'ROW {s.label} {s.number} {s.symbol} {s.size} {s.index}')\n"
        "m = isotropy_subgroups_at_kpoint(221, [0, 0.5, 0.5], with_settings=False)\n"
        "print('ARM', m.kpoint, m.coordinates, m.n_arms, len(m),\n"
        "      m['M3+'][0].label, m['M3+'][0].basis)\n"
        "print('SPELL', isotropy_subgroups_at_kpoint('Pm-3m', ['-1/2', '1/2', '3/2'],\n"
        "                                            with_settings=False).kpoint)\n"
        "print('PAIR', list(isotropy_subgroups_at_kpoint('Pm-3', 'gm', with_settings=False)))\n"
        "print('EXPORT', crystod.isotropy_subgroups_at_kpoint.__name__,\n"
        "      crystod.phonon.isotropy_subgroups_at_kpoint.__name__)\n"
        "for args in (('Pm-3m', 'Q'), ('Pm-3m', [0.25, 0, 0]), ('Pm-3m', [0, 0]),\n"
        "             ('Pm3m', 'GM'), (999, 'GM'), ('Pm-3m', None)):\n"
        "    try:\n"
        "        isotropy_subgroups_at_kpoint(*args)\n"
        "        print('NOT RAISED', args)\n"
        "    except ValueError as exc:\n"
        "        if args == ('Pm-3m', 'Q'):\n"
        "            print('MESSAGE', exc)\n"
        "print('SURVIVED')\n"
    )
    code, api_out = run_python(probe)
    report("crystod.group.isotropy_subgroups_at_kpoint exit 0", code == 0, api_out)
    report("k-point API lists the irreps in table order",
           "KEYS GM1+ GM2+ GM3+ GM4+ GM5+ GM1- GM2- GM3- GM4- GM5-" in api_out, api_out)
    report("k-point API returns a dict with the k point as attributes",
           "META KpointIsotropySubgroups True GM [0.0, 0.0, 0.0] 1 Pm-3m 221 {}"
           in api_out, api_out)
    report("each entry equals isotropy_subgroups of that irrep (29 records, settings on)",
           "SAME True 29" in api_out and "BASIS True" in api_out, api_out)
    code, cli_out = run_group(["--parent", "Pm-3m", "--kpoint", "GM"])
    api_rows = [tuple(line.split()[1:]) for line in api_out.splitlines()
                if line.startswith("ROW ")]
    cli_rows = [tuple(line.split()) for line in cli_out.splitlines()
                if line.startswith("GM") and "(" in line]
    report("k-point API and crystod-group --parent --kpoint agree row by row",
           code == 0 and len(api_rows) == 29 and api_rows == cli_rows,
           f"api={api_rows}\ncli={cli_rows}")
    report("coordinates of a non-representative arm resolve to the star (M, 3 arms)",
           "ARM M [0.5, 0.5, 0.0] 3 10 M3+(0;0;a) None" in api_out
           and "SPELL R" in api_out, api_out)
    report("complex-conjugate pair appears once, under its pair label",
           "PAIR ['GM1+', 'GM2+GM3+', 'GM4+', 'GM1-', 'GM2-GM3-', 'GM4-']" in api_out,
           api_out)
    report("exported from crystod, crystod.group and crystod.phonon",
           "EXPORT isotropy_subgroups_at_kpoint isotropy_subgroups_at_kpoint"
           in api_out, api_out)
    report("bad k-point input raises ValueError and lists the k points",
           "SURVIVED" in api_out and "NOT RAISED" not in api_out
           and 'MESSAGE k point "Q" is not tabulated' in api_out
           and "R (1/2, 1/2, 1/2)" in api_out, api_out)

    # order-parameter form returns the conventional setting as data
    probe = (
        "from crystod.group import isotropy_subgroups\n"
        "s = isotropy_subgroups(221, 'R4+', order_parameter=['0', '0', 'a'])[0]\n"
        "print(s.symbol, s.number, s.size, s.index, s.n_free)\n"
        "print('BASIS', s.basis is not None, 'ORIGIN', s.origin is not None)\n"
    )
    code, out = run_python(probe)
    report("order-parameter direction resolves to I4/mcm",
           code == 0 and "I4/mcm 140 2 6 1" in out, out)
    report("conventional basis and origin returned as arrays",
           "BASIS True ORIGIN True" in out, out)

    # multi-arm star: both branches must print the arm grouping of the tables
    probe = (
        "from crystod.group import isotropy_subgroups\n"
        "for op in (['0', '0', 'a'], ['a', 'a', 'a'], ['a', '-a', '0']):\n"
        "    s = isotropy_subgroups('Pm-3m', 'M3+', order_parameter=op)[0]\n"
        "    print('OP', s.label, s.direction, s.n_free, s.symbol)\n"
        "for s in isotropy_subgroups('Pm-3m', 'M3+'):\n"
        "    print('EN', s.label, s.direction, s.n_free, s.symbol)\n"
    )
    code, api_out = run_python(probe)
    report("multi-arm order parameter groups components arm by arm",
           code == 0 and "OP M3+(0;0;a) (0;0;a) 1 P4/mbm" in api_out, api_out)
    report("a sign-flipped component is one free parameter, not two",
           "OP M3+(a;-a;0) (a;-a;0) 1 I4/mmm" in api_out, api_out)

    # every direction the API prints must feed back into it unchanged
    probe = (
        "import re\n"
        "from crystod.group import isotropy_subgroups\n"
        "bad = 0\n"
        "for s in isotropy_subgroups('Pm-3m', 'X5+', with_settings=False):\n"
        "    tokens = re.split(r'[,;]', s.direction.strip('()'))\n"
        "    r = isotropy_subgroups('Pm-3m', 'X5+', order_parameter=tokens,\n"
        "                           with_settings=False)[0]\n"
        "    if (r.number, r.index, r.n_free, r.direction) != (\n"
        "            s.number, s.index, s.n_free, s.direction):\n"
        "        bad += 1\n"
        "        print('MISMATCH', s.direction, s.n_free, r.n_free)\n"
        "print('MISMATCHES', bad)\n"
    )
    code, out = run_python(probe)
    report("every enumerated direction round-trips through order_parameter",
           code == 0 and "MISMATCHES 0" in out, out)

    # hexagonal/trigonal strata carry composite directions like 0.282a, which
    # the resolver cannot honour: they must be refused, never silently wrong
    probe = (
        "import re\n"
        "from crystod.group import isotropy_subgroups\n"
        "wrong = rejected = 0\n"
        "for sg, ir in (('P6_3/mmc', 'H3'), ('P-3m1', 'K3'), ('P6_3mc', 'K3'),\n"
        "               ('P6/mmm', 'K3'), ('P6_3/mcm', 'K3')):\n"
        "    for s in isotropy_subgroups(sg, ir, with_settings=False):\n"
        "        tokens = re.split(r'[,;]', s.direction.strip('()'))\n"
        "        try:\n"
        "            r = isotropy_subgroups(sg, ir, order_parameter=tokens,\n"
        "                                   with_settings=False)[0]\n"
        "            if (r.number, r.index, r.n_free) != (s.number, s.index, s.n_free):\n"
        "                wrong += 1\n"
        "        except ValueError:\n"
        "            rejected += 1\n"
        "print('WRONG', wrong, 'REJECTED', rejected)\n"
    )
    code, out = run_python(probe)
    report("composite directions are refused, never silently mis-resolved",
           code == 0 and "WRONG 0 " in out and "REJECTED 0" not in out, out)
    report("both branches format the same direction identically",
           "EN M3+(0;0;a) (0;0;a) 1 P4/mbm" in api_out, api_out)
    code, cli_out = run_group(["--parent", "Pm-3m", "--irrep", "M3+",
                               "--order-parameter", "0", "0", "a"])
    report("API direction string matches crystod-group --parent",
           code == 0 and "M3+(0;0;a) -> P4/mbm" in cli_out, cli_out)

    # bad input must raise a catchable exception, never kill the caller
    probe = (
        "from crystod.group import isotropy_subgroups\n"
        "from crystod.phonon import label_phonon_modes\n"
        "for args, kwargs in ((('Pm3m', 'R5-'), {}), ((221, 'DT5'), {}),\n"
        "                     ((999, 'R5-'), {}),\n"
        "                     ((221, 'R5-'), {'order_parameter': ['0', '0', '0']})):\n"
        "    try:\n"
        "        isotropy_subgroups(*args, **kwargs)\n"
        "        print('NOT RAISED', args)\n"
        "    except ValueError:\n"
        "        pass\n"
        "print('SURVIVED')\n"
    )
    code, out = run_python(probe)
    report("bad API input raises ValueError instead of exiting the process",
           code == 0 and "SURVIVED" in out and "NOT RAISED" not in out, out)

    # the same contract holds for every function of every domain, not just
    # the phonon-subgroup entry points
    probe = (
        "import crystod\n"
        "from crystod.group import isotropy_subgroups, IsotropySubgroup\n"
        "for call in (\"crystod.bz.get_special_kpoints('NOSUCH')\",\n"
        "             \"crystod.mol.load_molecule('/nonexistent.xyz')\",\n"
        "             \"crystod.group.isotropy_subgroups(221, [])\"):\n"
        "    try:\n"
        "        eval(call)\n"
        "        print('NOT RAISED', call)\n"
        "    except ValueError:\n"
        "        pass\n"
        "a = isotropy_subgroups('Pm-3m', 'R4+', with_settings=False)[0]\n"
        "b = isotropy_subgroups(221, 'R4+', with_settings=False)[0]\n"
        "print('EQ', a == b, type(a).__name__, isinstance(a, IsotropySubgroup))\n"
        "print('WRAPS', crystod.group.compute_star.__name__)\n"
        "print('SURVIVED')\n"
    )
    code, out = run_python(probe)
    report("SystemExit is translated in every API domain, not just phonon",
           code == 0 and "SURVIVED" in out and "NOT RAISED" not in out, out)
    report("wrapping preserves dataclass equality, type and function names",
           "EQ True IsotropySubgroup True" in out and "WRAPS compute_star" in out, out)

    # phonon API: labeling and imaginary-mode subgroups on the SrTiO3 example
    if os.path.isdir(PHONON_IRREP_DIR):
        probe = (
            "import phonopy\n"
            "from crystod.phonon import (label_phonon_modes, imaginary_mode_subgroups,\n"
            "                            commensurate_qpoints, scan_imaginary_modes)\n"
            "ph = phonopy.load(supercell_matrix=[4, 4, 4], primitive_matrix='auto',\n"
            "                  unitcell_filename='221_PPOSCAR_SrTiO3',\n"
            "                  force_sets_filename='FORCE_SETS')\n"
            "modes = label_phonon_modes(ph, [0.5, 0.5, 0.5])\n"
            "m = modes[0]\n"
            "print('LOWEST', m.band_indices, round(m.frequency, 4), m.labels,\n"
            "      m.qpoint_label, m.degeneracy, m.is_imaginary)\n"
            "arm = label_phonon_modes(ph, [-0.5, 0.5, 0.5])[0]\n"
            "print('ARM', arm.qpoint_label, arm.labels)\n"
            "qs = commensurate_qpoints(ph)\n"
            "print('NQ', len(qs))\n"
            "print('QEXACT', all(abs(round(x * 4) / 4 - x) < 1e-12\n"
            "                    for q in qs for x in q))\n"
            "res = imaginary_mode_subgroups(ph, [0.5, 0.5, 0.5])\n"
            "print('NRES', len(res), 'NSUB', len(res[0].subgroups),\n"
            "      res[0].space_group, res[0].space_group_number)\n"
            "print('SUBS', sorted(s.symbol for s in res[0].subgroups))\n"
            "print('SCAN', [(r.mode.qpoint_label, r.mode.labels) for r in scan_imaginary_modes(ph)])\n"
        )
        code, out = run_python(probe, cwd=PHONON_IRREP_DIR)
        report("phonon API on the SrTiO3 example exit 0", code == 0, out)
        report("label_phonon_modes labels the R-point soft mode",
               "LOWEST (1, 2, 3) -1.0867 ('R5-',) R 3 True" in out, out)
        report("a non-representative star arm gets the same label",
               "ARM R ('R5-',)" in out, out)
        report("commensurate q points of a 4x4x4 supercell counted",
               "NQ 64" in out, out)
        report("commensurate q points are exact fractions, not snapped decimals",
               "QEXACT True" in out, out)
        report("imaginary_mode_subgroups returns one level with six subgroups",
               "NRES 1 NSUB 6 Pm-3m 221" in out, out)
        report("subgroups of the R-point rotation mode",
               "SUBS ['C2/c', 'C2/m', 'I4/mcm', 'Imma', 'P-1', 'R-3c']" in out, out)
        report("scan_imaginary_modes finds exactly the R-point level",
               "SCAN [('R', ('R5-',))]" in out, out)
    else:
        report("phonon_irrep example data found", False, PHONON_IRREP_DIR)

    # the frame of an input in the ISO-IR setting is the input's own (frame
    # rule 1): SrTiO3 with Sr at the origin calls the R tilt R5-, with Ti at
    # the origin R4+; an input at any other origin gets the canonical frame of
    # the crystal (frame rule 2), which here puts Ti at the origin
    if os.path.isdir(PHONON_IRREP_DIR):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(PHONON_IRREP_DIR, "221_PPOSCAR_SrTiO3")
            for name, shift in (("POSCAR_Sr_origin", (0, 0, 0)),
                                ("POSCAR_Ti_origin", (0.5, 0.5, 0.5)),
                                ("POSCAR_shifted", (0.1234, 0.2345, 0.3456))):
                _write_cell_variant(source, os.path.join(tmp, name), shift=shift)
            probe = (
                "import phonopy\n"
                "from crystod.phonon import label_phonon_modes\n"
                f"forces = {os.path.join(PHONON_IRREP_DIR, 'FORCE_SETS')!r}\n"
                "for name in ('POSCAR_Sr_origin', 'POSCAR_Ti_origin', 'POSCAR_shifted'):\n"
                "    ph = phonopy.load(supercell_matrix=[4, 4, 4], primitive_matrix='auto',\n"
                "                      unitcell_filename=name, force_sets_filename=forces)\n"
                "    mode = label_phonon_modes(ph, [0.5, 0.5, 0.5])[0]\n"
                "    print('TILT', name, mode.band_indices, mode.labels)\n"
            )
            code, out = run_python(probe, cwd=tmp)
        report("SrTiO3 R tilt: R5- with Sr at the origin, R4+ with Ti at the origin "
               "and at a generic origin",
               code == 0
               and "TILT POSCAR_Sr_origin (1, 2, 3) ('R5-',)" in out
               and "TILT POSCAR_Ti_origin (1, 2, 3) ('R4+',)" in out
               and "TILT POSCAR_shifted (1, 2, 3) ('R4+',)" in out, out)

    # IsoIrrep.match_k returns the canonical line parameter (the smallest
    # |parameter|, positive first) on the same arm for every copy k + G: DT of
    # F-43m at (0,0,-1/2), (0,0,3/2), (0,0,-5/2) is the arm (0,0,-a) at a = 1/2
    # (it used to return the arm (0,0,a) at a = -1/2, 3/2, -5/2)
    probe = (
        "from crystod.isoir import load_isoir_irreps\n"
        "dt3 = [ir for ir in load_isoir_irreps(216) if ir.label == 'DT3'][0]\n"
        "for k in ([0, 0, -0.5], [0, 0, 1.5], [0, 0, -2.5], [0, 0, 0.5]):\n"
        "    arm, p = dt3.match_k(k)\n"
        "    print('MATCH', k, arm, [round(float(v), 6) + 0.0 for v in p])\n"
    )
    code, out = run_python(probe)
    report("IsoIrrep.match_k: one canonical parameter for k and its copies k + G",
           code == 0
           and all(f"MATCH {k} 5 [0.0, 0.5, 0.0]" in out
                   for k in ("[0, 0, -0.5]", "[0, 0, 1.5]", "[0, 0, -2.5]"))
           and "MATCH [0, 0, 0.5] 4 [0.0, 0.5, 0.0]" in out, out)

    # frame rule 2: a description that is not in the ISO-IR setting (generic
    # origin shift, plus a unimodular basis change, a 2x1x1 supercell or a
    # rigid rotation) gets one canonical frame, so the atoms have the same
    # ISO-IR coordinates in every description; Si3Cl8 has coordinates such as
    # 0.07465 on a rounding boundary of the old 4-decimal comparison
    probe = (
        "import numpy as np, spglib\n"
        "from phonopy.interface.calculator import read_crystal_structure\n"
        "from crystod.isoir import canonical_iso_frame, load_isoir_irreps\n"
        "COPIES = {'P': [], 'I': [(.5, .5, .5)], 'C': [(.5, .5, 0)], 'A': [(0, .5, .5)],\n"
        "          'B': [(.5, 0, .5)], 'F': [(0, .5, .5), (.5, 0, .5), (.5, .5, 0)],\n"
        "          'R': [(2 / 3, 1 / 3, 1 / 3), (1 / 3, 2 / 3, 2 / 3)]}\n"
        "def describe(cell, C, shift, Q=None):\n"
        "    lattice, positions, numbers = cell\n"
        "    C = np.array(C, dtype=float)\n"
        "    box = np.array(np.meshgrid(*[range(-2, 3)] * 3)).reshape(3, -1).T\n"
        "    out, species = [], []\n"
        "    for x, z in zip(positions, numbers):\n"
        "        for y in ((x + shift + box) @ np.linalg.inv(C)) % 1.0:\n"
        "            if out and any(s == z and np.max(np.abs(r - np.rint(r))) < 1e-6\n"
        "                           for r, s in zip(np.array(out) - y, species)):\n"
        "                continue\n"
        "            out.append(y)\n"
        "            species.append(z)\n"
        "    new = C @ lattice if Q is None else C @ lattice @ Q.T\n"
        "    return new, np.array(out), np.array(species)\n"
        "def iso_atoms(frame, cell, centering):\n"
        "    P, origin = frame\n"
        "    x = cell[1] @ np.asarray(P).T + origin\n"
        "    copies = [np.zeros(3)] + [np.array(v) for v in COPIES[centering]]\n"
        "    x = np.concatenate([x + c for c in copies])\n"
        "    return np.tile(cell[2], len(copies)), x - np.floor(x + 1e-5)\n"
        "def same(a, b, tol=2e-5):\n"
        "    def covered(p, q):\n"
        "        for z, x in zip(*p):\n"
        "            d = q[1][q[0] == z] - x\n"
        "            d = d - np.rint(d)\n"
        "            if len(d) == 0 or np.min(np.max(np.abs(d), axis=1)) > tol:\n"
        "                return False\n"
        "        return True\n"
        "    return covered(a, b) and covered(b, a)\n"
        "c, s = np.cos(0.7), np.sin(0.7)\n"
        "Q = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ np.array([[1, 0, 0], [0, c, s], [0, -s, c]])\n"
        "shift = np.array([0.0731, 0.1593, 0.2417])\n"
        "descriptions = [('shift', np.eye(3), None),\n"
        "                ('basis', [[1, 1, 0], [0, 1, 0], [-1, 0, 1]], None),\n"
        "                ('2x1x1', np.diag([2, 1, 1]), None), ('rotated', np.eye(3), Q)]\n"
        f"root = {ROOT!r}\n"
        "for path in ('test_POSCARs/221_PPOSCAR_SrTiO3', 'test_POSCARs/227_PPOSCAR_Si',\n"
        "             'test_POSCARs/219_PPOSCAR_Si3Cl8',\n"
        "             '32_phonon_vector/MoS2_P-6m2/187_PPOSCAR_MoS2',\n"
        "             'test_POSCARs/99_PPOSCAR_BaTiO3', 'test_POSCARs/82_PPOSCAR_AlPO4'):\n"
        "    structure, _ = read_crystal_structure(f'{root}/example/{path}', interface_mode='vasp')\n"
        "    cell = (np.array(structure.cell), np.array(structure.scaled_positions),\n"
        "            np.array(structure.numbers))\n"
        "    sgnum = spglib.get_symmetry_dataset(cell, symprec=1e-5).number\n"
        "    centering = load_isoir_irreps(sgnum)[0].centering\n"
        "    atoms = []\n"
        "    for tag, C, rotation in descriptions:\n"
        "        other = describe(cell, C, shift, rotation)\n"
        "        frame = canonical_iso_frame(sgnum, other, 1e-5)\n"
        "        atoms.append(None if frame is None else iso_atoms(frame, other, centering))\n"
        "    differ = [tag for (tag, _, _), a in zip(descriptions[1:], atoms[1:])\n"
        "              if a is None or atoms[0] is None or not same(atoms[0], a)]\n"
        "    print('FRAME', path.split('/')[-1], sgnum, 'differ', differ)\n"
    )
    code, out = run_python(probe)
    report("frame rule 2: shifted, re-based, 2x1x1 and rotated descriptions of six "
           "crystals (221, 227, 219 Si3Cl8, 187, 99, 82) give the same ISO-IR "
           "coordinates",
           code == 0 and out.count("FRAME ") == 6 and out.count("differ []") == 6, out)

    # rule 2 compares coordinates at 0.71, 0.071 and 0.0071 times the
    # tolerance, not at round multiples of it: LaFeAsO with its coordinates
    # rounded to 5 decimals (CIF precision) has gaps of exactly 1e-5, and at
    # levels 1, 0.1 and 0.01 times the tolerance two origin shifts of it got
    # different frames (S1 and S2 exchanged at S)
    probe = (
        "import contextlib, io\n"
        "import numpy as np\n"
        "from phonopy.interface.calculator import read_crystal_structure\n"
        "from phonopy.structure.atoms import PhonopyAtoms\n"
        "from crystod.phonon import SymmetryOnlyVibrations\n"
        f"path = {os.path.join(ROOT, 'example', 'test_POSCARs', '73_PPOSCAR_LaFeAsO')!r}\n"
        "cell, _ = read_crystal_structure(path, interface_mode='vasp')\n"
        "rounded = np.round(cell.scaled_positions, 5) % 1.0\n"
        "order = np.argsort(cell.numbers, kind='stable')\n"
        "for shift in ((0.6251, 0.8972, 0.7757), (0.4679, 0.303, 0.2784)):\n"
        "    shifted = PhonopyAtoms(cell=cell.cell, numbers=cell.numbers[order],\n"
        "                           scaled_positions=((rounded + shift) % 1.0)[order])\n"
        "    with contextlib.redirect_stdout(io.StringIO()):\n"
        "        labels = SymmetryOnlyVibrations(shifted).describe_mode_spaces([0.5, 0, 0])[2]\n"
        "    print('S', ' '.join(labels))\n"
    )
    code, out = run_python(probe)
    lines = re.findall(r"^S (.*)$", out, re.M)
    report("5-decimal LaFeAsO (Iba2) at two origins: the same labels at S",
           code == 0 and len(lines) == 2 and lines[0] == lines[1]
           and lines[0].startswith("S"), out)

    # the reduction into the unit cell uses the same level: two O atoms of a
    # 5-decimal Pmn2_1 cell differ by exactly 1e-5 along the polar axis, and
    # with the wrap at the full tolerance origin shifts of the cell split into
    # two frames (Z1 and Z3 exchanged at Z)
    probe = (
        "import contextlib, io\n"
        "import numpy as np\n"
        "from phonopy.structure.atoms import PhonopyAtoms\n"
        "from crystod.phonon import SymmetryOnlyVibrations\n"
        "lattice = np.diag([3.1, 4.3, 5.7])\n"
        "base = np.array([[0.18680, 0.38840, 0.58300], [0.45940, 0.93020, 0.08300],\n"
        "                 [0.95940, 0.38840, 0.58300], [0.68680, 0.93020, 0.08300],\n"
        "                 [0.48580, 0.23240, 0.58301], [0.16040, 0.08620, 0.08301],\n"
        "                 [0.66040, 0.23240, 0.58301], [0.98580, 0.08620, 0.08301],\n"
        "                 [0.07310, 0.51260, 0.33360], [0.57310, 0.80600, 0.83360]])\n"
        "numbers = [8] * 8 + [82] * 2\n"
        "shift = np.array([0.54820, 0.22780, 0.88900])\n"
        "for positions in (base, np.round((base + shift) % 1.0, 5)):\n"
        "    cell = PhonopyAtoms(cell=lattice, numbers=numbers, scaled_positions=positions)\n"
        "    with contextlib.redirect_stdout(io.StringIO()):\n"
        "        labels = SymmetryOnlyVibrations(cell).describe_mode_spaces([0, 0, 0.5])[2]\n"
        "    print('Z', ' '.join(labels))\n"
    )
    code, out = run_python(probe)
    lines = re.findall(r"^Z (.*)$", out, re.M)
    report("5-decimal Pmn2_1 at two origins: the same labels at Z",
           code == 0 and len(lines) == 2 and lines[0] == lines[1]
           and lines[0].startswith("Z"), out)

    # rule 1 has no size limit: a 9x9x9 supercell of AlPO4 (4374 atoms, 729
    # cells) keeps the frame of the cell (rule 1 stopped at 512 cells, and
    # the supercell got a rule-2 frame in which P was PA)
    probe = (
        "import numpy as np\n"
        "from phonopy.interface.calculator import read_crystal_structure\n"
        "from phonopy.structure.cells import get_supercell\n"
        "from crystod.isoir import canonical_iso_frame\n"
        f"path = {os.path.join(ROOT, 'example', 'test_POSCARs', '82_PPOSCAR_AlPO4')!r}\n"
        "cell, _ = read_crystal_structure(path, interface_mode='vasp')\n"
        "def frame(c):\n"
        "    return canonical_iso_frame(82, (c.cell, c.scaled_positions, c.numbers), 1e-5)\n"
        "P, origin = frame(cell)\n"
        "Q, shift = frame(get_supercell(cell, np.diag([9, 9, 9])))\n"
        "gap = shift - origin\n"
        "print('SAME', np.allclose(Q, 9 * P, atol=1e-8),\n"
        "      np.allclose(gap - np.rint(gap), 0, atol=1e-8))\n"
    )
    code, out = run_python(probe)
    report("9x9x9 supercell of AlPO4 keeps the frame of the cell (rule 1, no size limit)",
           code == 0 and "SAME True True" in out, out)

    # domain namespaces expose their advertised names
    probe = (
        "import crystod\n"
        "for name in ('salc', 'group', 'phonon', 'bz', 'mag', 'md', 'mol', 'xrd', 'search'):\n"
        "    module = getattr(crystod, name)\n"
        "    missing = [n for n in module.__all__ if not hasattr(module, n)]\n"
        "    print(name, len(module.__all__), 'MISSING' if missing else 'OK', missing or '')\n"
    )
    code, out = run_python(probe)
    report("every advertised API symbol resolves", code == 0 and "MISSING" not in out, out)
    report("all nine API domains are populated",
           code == 0 and all(f"\n{name} " in "\n" + out for name in
                             ("salc", "group", "phonon", "bz", "mag", "md", "mol", "xrd",
                              "search")), out)


# ---------------------------------------------------------------- 44. crystod-xrd
def _xrd_table(path: str) -> list[tuple]:
    """(h, k, l, mult, d, two_theta, intensity, line) rows of a crystod-xrd table."""
    rows = []
    for line in open(path, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        h, k, l, mult, d, tt, inten, radiation, _families = line.rstrip("\n").split(",", 8)
        rows.append((int(h), int(k), int(l), int(mult), float(d), float(tt), float(inten), radiation))
    return rows


def test_44_xrd() -> None:
    print("\n[44] crystod-xrd (powder X-ray diffraction patterns)")

    with tempfile.TemporaryDirectory() as tmp:
        code, out = run_xrd(["-c", POSCAR_ScF3], cwd=tmp)
        report("crystod-xrd -c ScF3 exit 0", code == 0, out)
        table = os.path.join(tmp, "XRD_221_PPOSCAR_ScF3_CuKa.txt")
        report("default names: XRD_{cell}_CuKa.txt and .pdf written",
               os.path.isfile(table)
               and os.path.isfile(os.path.join(tmp, "XRD_221_PPOSCAR_ScF3_CuKa.pdf")), out)
        report("default radiation is the Cu K-alpha doublet (Ka1 + Ka2, 2:1)",
               "CuKa1 = 1.5405929 A, CuKa2 = 1.5444274 A  (Ka1 : Ka2 = 2 : 1)" in out, out)
        report("default profile is Lorentzian", "lorentzian profile, width 0.1 deg" in out, out)
        rows = _xrd_table(table) if os.path.isfile(table) else []
        # reference values: the original script/xrd_pattern_poscar.py on the
        # same POSCAR (pymatgen XRDCalculator, RIETAN-FP wavelengths)
        report("ScF3 CuKa: 36 reflections in 10-120 deg (18 hkl x 2 lines)", len(rows) == 36,
               f"{len(rows)} rows")
        first = rows[0] if rows else None
        report("strongest peak (1 0 0) at 21.8217 deg, Ka2 partner at half intensity",
               first is not None and first[:4] == (1, 0, 0, 6)
               and abs(first[5] - 21.8217) < 1e-3 and abs(first[6] - 100.0) < 1e-3
               and rows[1][7] == "CuKa2" and abs(rows[1][6] - 50.0) < 1e-3, str(rows[:2]))
        peak_210 = [r for r in rows if r[:3] == (2, 1, 0) and r[7] == "CuKa1"]
        report("(2 1 0) CuKa1 reproduces the original script (50.0793 deg, 34.4288)",
               len(peak_210) == 1 and abs(peak_210[0][5] - 50.0793) < 1e-3
               and abs(peak_210[0][6] - 34.4288) < 1e-3, str(peak_210))
        wavelengths = {"CuKa1": 1.5405929, "CuKa2": 1.5444274}
        bragg = all(abs(2 * np.degrees(np.arcsin(wavelengths[r[7]] / (2 * r[4]))) - r[5]) < 2e-3
                    for r in rows)
        report("every peak obeys Bragg's law 2theta = 2 asin(lambda / 2d)", bool(rows) and bragg)
        report("coincident families are listed ((3 0 0) + (2 2 1) in cubic)",
               "+ (2 2 1) x24" in out, out)

        code, out = run_xrd(["-c", POSCAR_ScF3, "--xraytype", "cuka1", "--peak-profile",
                             "gaussian", "--output", "mono"], cwd=tmp)
        rows = _xrd_table(os.path.join(tmp, "mono.txt")) if code == 0 else []
        report("--xraytype is case-insensitive; a single line gives 18 peaks, all CuKa1",
               code == 0 and len(rows) == 18 and {r[7] for r in rows} == {"CuKa1"}, out)
        report("--peak-profile gaussian and --output PREFIX",
               "gaussian profile" in out and os.path.isfile(os.path.join(tmp, "mono.pdf")), out)

        code, out = run_xrd(["-c", POSCAR_ScF3, "--xraytype", "MoKa", "--two-theta", "5", "30",
                             "--output", "mo"], cwd=tmp)
        rows = _xrd_table(os.path.join(tmp, "mo.txt")) if code == 0 else []
        report("--two-theta restricts the window (MoKa, 5-30 deg)",
               code == 0 and rows and all(5 <= r[5] <= 30 for r in rows)
               and {r[7] for r in rows} == {"MoKa1", "MoKa2"}, out)

        code, out = run_xrd(["-c", POSCAR_ScF3, "--xraytype", "XeKa"], cwd=tmp)
        report("unknown --xraytype refused with the list of names",
               code != 0 and "unknown X-ray type" in out and "CuKa1" in out
               and "Traceback" not in out, out)
        code, out = run_xrd(["-c", POSCAR_ScF3, "--peak-profile", "voigt"], cwd=tmp)
        report("unknown --peak-profile refused by argparse", code != 0 and "invalid choice" in out, out)
        code, out = run_xrd(["-c", os.path.join(tmp, "missing")], cwd=tmp)
        report("missing POSCAR refused cleanly",
               code != 0 and "No POSCAR" in out and "Traceback" not in out, out)

    with tempfile.TemporaryDirectory() as tmp:
        code, out = run_xrd(["--example"], cwd=tmp)
        report("crystod-xrd --example lists ScF3 and SrTiO3",
               code == 0 and "ScF3" in out and "SrTiO3" in out, out)
        code, out = run_xrd(["--example", "SrTiO3"], cwd=tmp)
        report("crystod-xrd --example SrTiO3 (CuKa1, gaussian) runs",
               code == 0 and "Running: crystod-xrd -c 221_PPOSCAR_SrTiO3 --xraytype CuKa1" in out
               and os.path.isfile(os.path.join(tmp, "XRD_221_PPOSCAR_SrTiO3_CuKa1.txt")), out)

    # hexagonal lattices: pymatgen indexes their reflections with Miller-Bravais
    # (h k i l); the printed table and the text file keep (h k l), i = -(h + k)
    # dropped. ZnO (P6_3mc) and AlF3 (R-3c in the hexagonal setting of its CIF)
    family_pattern = re.compile(r"\(-?\d+ -?\d+ -?\d+\) x\d+( \+ \(-?\d+ -?\d+ -?\d+\) x\d+)*")

    def families_three_index(path: str) -> bool:
        fields = [line.rstrip("\n").split(",", 8)[8] for line in open(path, encoding="utf-8")
                  if line.strip() and not line.startswith("#")]
        return bool(fields) and all(family_pattern.fullmatch(f) for f in fields)

    poscar_zno = os.path.join(ROOT, "example", "test_POSCARs", "186_PPOSCAR_ZnO")
    cif_alf3 = os.path.join(ROOT, "example", "23_symmetry_mode", "167_PPOSCAR_AlF3.cif")
    with tempfile.TemporaryDirectory() as tmp:
        code, out = run_xrd(["-c", poscar_zno, "--output", "zno"], cwd=tmp)
        table = os.path.join(tmp, "zno.txt")
        rows = _xrd_table(table) if code == 0 and os.path.isfile(table) else []
        report("hexagonal ZnO (P6_3mc): exit 0, no traceback, 58 reflections (29 hkl x 2 lines)",
               code == 0 and "Traceback" not in out and "ZnO, P6_3mc" in out and len(rows) == 58,
               out)
        strongest = max(rows, key=lambda r: r[6]) if rows else None
        report("ZnO: strongest peak (1 0 1) x12 at 35.7474 deg (CuKa1), (0 0 2) at d = c/2",
               strongest is not None and strongest[:4] == (1, 0, 1, 12)
               and strongest[7] == "CuKa1" and abs(strongest[5] - 35.7474) < 1e-3
               and any(r[:4] == (0, 0, 2, 2) and abs(r[4] - 2.653410) < 1e-5 for r in rows),
               str(strongest))
        report("ZnO: the printed table and the families column use (h k l)",
               re.search(r"^\s+1\s+0\s+1\s+12\s+2\.509764\s+35\.7474\s+100\.000\s+CuKa1$",
                         out, re.M) is not None
               and os.path.isfile(table) and families_three_index(table), out)

        code, out = run_python(
            "from pymatgen.core import Structure\n"
            f"s = Structure.from_file({cif_alf3!r})\n"
            "s.to(filename='POSCAR_AlF3_hex', fmt='poscar')\n"
            "print('HEXAGONAL', s.lattice.is_hexagonal(), len(s))\n", cwd=tmp)
        report("AlF3 CIF (R-3c) written as a POSCAR of the hexagonal cell (24 atoms)",
               code == 0 and "HEXAGONAL True 24" in out, out)
        code, out = run_xrd(["-c", "POSCAR_AlF3_hex", "--output", "alf3"], cwd=tmp)
        table = os.path.join(tmp, "alf3.txt")
        rows = _xrd_table(table) if code == 0 and os.path.isfile(table) else []
        report("rhombohedral AlF3 in the hexagonal setting: exit 0, no traceback",
               code == 0 and "Traceback" not in out and "AlF3, R-3c" in out and bool(rows), out)
        strongest = max(rows, key=lambda r: r[6]) if rows else None
        report("AlF3: strongest peak at 24.8038 deg (d = 3.586650 A, CuKa1), (0 0 6) at d = c/6",
               strongest is not None and strongest[7] == "CuKa1"
               and abs(strongest[5] - 24.8038) < 1e-3 and abs(strongest[4] - 3.586650) < 1e-5
               and any(r[:4] == (0, 0, 6, 2) and abs(r[4] - 2.102767) < 1e-5 for r in rows),
               str(strongest))
        report("AlF3: the families column uses (h k l)",
               os.path.isfile(table) and families_three_index(table), out)

    probe = (
        "import numpy as np\n"
        "from crystod import xrd\n"
        f"s = xrd.load_structure({POSCAR_ScF3!r})\n"
        "p = xrd.compute_xrd_pattern(s, 'CuKa', (10, 120))\n"
        "print('NPEAK', len(p.peaks), p.space_group, p.formula)\n"
        "for profile in ('gaussian', 'lorentzian'):\n"
        "    x, y = xrd.smear_pattern(p, profile, width=0.1, npoints=200000, two_theta_range=(0, 180))\n"
        "    area = float(np.sum(y) * (x[1] - x[0]))\n"
        "    print('AREA', profile, round(area / p.intensity.sum(), 3))\n"
        "try:\n"
        "    xrd.compute_xrd_pattern(s, 'XeKa')\n"
        "except ValueError as exc:\n"
        "    print('VALUEERROR', 'unknown X-ray type' in str(exc))\n"
    )
    code, out = run_python(probe)
    report("API: crystod.xrd.compute_xrd_pattern (36 peaks, Pm-3m, ScF3)",
           code == 0 and "NPEAK 36 Pm-3m ScF3" in out, out)
    report("API: the Gaussian profile has unit area per peak", "AREA gaussian 1.0" in out, out)
    report("API: the Lorentzian profile has unit area per peak (tails cut at 0/180 deg)",
           "AREA lorentzian 0.99" in out or "AREA lorentzian 1.0" in out, out)
    report("API: bad input raises ValueError through the namespace", "VALUEERROR True" in out, out)

    probe = (
        "from crystod import xrd\n"
        "peak = xrd.Peak(hkl=(1, 0, -1, 1), multiplicity=12,\n"
        "                families=(((1, 0, -1, 1), 12), ((2, -1, -1, 0), 6)),\n"
        "                d=2.5, two_theta=35.7, intensity=100.0, line='CuKa1')\n"
        "print('HKL', peak.hkl, peak.families_label)\n"
        f"s = xrd.load_structure({poscar_zno!r})\n"
        "p = xrd.compute_xrd_pattern(s, 'CuKa1', (10, 120))\n"
        "best = max(p.peaks, key=lambda q: q.intensity)\n"
        "print('ZNO', p.space_group, all(len(q.hkl) == 3 for q in p.peaks), best.hkl)\n"
        "try:\n"
        "    xrd.Peak(hkl=(1, 0, 0, 1), multiplicity=1, families=(((1, 0, 0, 1), 1),),\n"
        "             d=1.0, two_theta=90.0, intensity=1.0, line='CuKa1')\n"
        "except ValueError as exc:\n"
        "    print('REFUSED', 'Miller-Bravais' in str(exc))\n"
    )
    code, out = run_python(probe)
    report("API: Peak stores Miller-Bravais (h k i l) as (h k l), families too",
           code == 0 and "HKL (1, 0, 1) (1 0 1) x12 + (2 -1 0) x6" in out, out)
    report("API: compute_xrd_pattern on ZnO gives (h k l) peaks, strongest (1, 0, 1)",
           "ZNO P6_3mc True (1, 0, 1)" in out, out)
    report("API: four indices with i != -(h + k) are refused", "REFUSED True" in out, out)


# ---------------------------------------------------------------- 45. crystod-search
# The Materials Project is never contacted here.  A stub of its
# materials/summary endpoint serves example/45_mp_search/mp_summary_Sr-Ti-O.json.gz
# (recorded from the live API on 2026-09-18) on 127.0.0.1, and every
# crystod-search run gets MP_API_ENDPOINT pointing at it, a dummy key only the
# stub accepts, a HOME without a pymatgen settings file (so no real key can leak
# in) and a dead proxy for every host but 127.0.0.1 (so a request that escaped
# the stub would fail instead of reaching the network).
#
# The stub answers the way the live API was seen to answer on the recording
# date (the same IDs, order and total_doc for the queries crystod-search sends,
# checked side by side): filters on formula (pymatgen, as the server parses
# it), chemsys (with * wildcards), elements, exclude_elements, material_ids
# (legacy numbers or letters in, letters out), theoretical, is_stable,
# deprecated (false unless asked), energy_above_hull/band_gap/nsites ranges and
# spacegroup_number; _fields projection; one _sort_fields field with a "-" for
# descending, nested symmetry.* and boolean fields being silently ignored; and
# _limit (default 100, at most 1000) / _skip paging.  Where MongoDB leaves the
# order open, the stub picks one: ties of a sort cut by _limit come in an order
# that depends on the window (the live API does return overlapping pages for
# band_gap and nsites), and the unsorted order of formula/elements queries,
# which the live server reads from other indexes, is the recorded chemsys order.
MP_STUB_KEY = "crystod-testsuite-stub-api-key-0"  # 32 characters, accepted by the stub only
MP_SUMMARY_PATH = "/materials/summary/"
MP_MAX_LIMIT = 1000
MP_DEFAULT_LIMIT = 100
# fields the live API sorts on (checked 2026-09-18); any other sort field is ignored
MP_SORTABLE = frozenset({"material_id", "formula_pretty", "formula_anonymous", "chemsys",
                         "nelements", "nsites", "band_gap", "energy_above_hull"})
MP_LEGACY_ID_CUTOFF = 3347529  # the website writes IDs up to this one as numbers
_MP_ELEMENTS = frozenset("""
H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu
Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba
La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb
Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs
Mt Ds Rg Cn Nh Fl Mc Lv Ts Og
""".split())


class _MPStubError(Exception):
    def __init__(self, status: int, body) -> None:
        super().__init__(status)
        self.status, self.body = status, body


def _mp_alpha_id(text: str) -> str:
    """mp-5229 / mp-aaaaahtd -> mp-aaaaahtd, the spelling the API answers with."""
    prefix, _, body = text.strip().lower().partition("-")
    if body.isdigit():
        value, letters = int(body), ""
        while value:
            value, digit = divmod(value, 26)
            letters = chr(ord("a") + digit) + letters
        body = letters.rjust(8, "a")
    return f"{prefix}-{body}"


def _mp_website_id(alpha: str) -> str:
    """mp-aaaaahtd -> mp-5229; IDs past the cut point stay alphabetical."""
    prefix, _, body = alpha.partition("-")
    value = 0
    for letter in body:
        value = value * 26 + ord(letter) - ord("a")
    return f"{prefix}-{value}" if value <= MP_LEGACY_ID_CUTOFF else alpha


def _mp_invalid(name: str, text: str, kind: str) -> _MPStubError:
    error_type, message = {
        "int": ("int_parsing",
                "Input should be a valid integer, unable to parse string as an integer"),
        "float": ("float_parsing",
                  "Input should be a valid number, unable to parse string as a number"),
        "bool": ("bool_parsing", "Input should be a valid boolean, unable to interpret input"),
    }[kind]
    return _MPStubError(422, {"detail": [{"type": error_type, "loc": ["query", name],
                                          "msg": message, "input": text}]})


def _mp_parse(params: dict, name: str, kind: str):
    text = params[name]
    try:
        if kind == "int":
            return int(text)
        if kind == "float":
            return float(text)
    except ValueError:
        raise _mp_invalid(name, text, kind) from None
    lowered = text.strip().lower()
    if lowered in ("true", "1", "yes", "on", "t", "y"):
        return True
    if lowered in ("false", "0", "no", "off", "f", "n"):
        return False
    raise _mp_invalid(name, text, kind)


def _mp_element_list(params: dict, name: str) -> list:
    text = params[name]
    if len(text) > 60:
        raise _MPStubError(422, {"detail": [{
            "type": "string_too_long", "loc": ["query", name],
            "msg": "String should have at most 60 characters", "input": text,
            "ctx": {"max_length": 60}}]})
    symbols = [s.strip() for s in text.split(",")]
    if not all(s in _MP_ELEMENTS for s in symbols):
        raise _MPStubError(400, {"detail": "Please provide a comma-seperated list of elements"})
    return symbols


def _mp_formula_test(text: str):
    """The formula criteria of the server (emmet's formula_to_criteria) as a predicate."""
    from pymatgen.core import Composition, DummySpecies

    formulas = [f.strip() for f in text.split(",")]

    def amounts(composition) -> dict:
        return {str(el): n for el, n in composition.reduced_composition.items()}

    def same(doc: dict, wanted: dict) -> bool:
        return all(abs((doc["composition_reduced"].get(el) or 0.0) - n) < 1e-6
                   for el, n in wanted.items())

    try:
        if "*" in text:
            if len(formulas) > 1:
                raise _MPStubError(400, {"detail": "Wild cards only supported for single "
                                                   "formula queries."})
            dummies = "ADEGJLMQRXZ"
            filled = text.replace("*", "{}").format(*dummies[:text.count("*")])
            composition = Composition(Composition(filled).get_integer_formula_and_factor()[0])
            anonymous = composition.reduced_composition.anonymized_formula
            real = {el: n for el, n in amounts(composition).items() if el in _MP_ELEMENTS}
            return lambda d: d["formula_anonymous"] == anonymous and same(d, real)
        compositions = [Composition(f) for f in formulas]
        if any(isinstance(el, DummySpecies) for c in compositions for el in c):
            wanted = {c.anonymized_formula for c in compositions}  # ABO3 -> ABC3
            return lambda d: d["formula_anonymous"] in wanted
        if len(compositions) == 1:
            wanted = amounts(compositions[0])  # O -> {O: 2}, as pymatgen reduces it
            return lambda d: d["nelements"] == len(wanted) and same(d, wanted)
        pretty = {c.reduced_formula for c in compositions}
        return lambda d: d["formula_pretty"] in pretty
    except _MPStubError:
        raise
    except Exception:
        raise _MPStubError(400, {"detail": "Problem processing one or more provided "
                                           "formulas."}) from None


def _mp_chemsys_test(text: str):
    systems = [s.strip() for s in text.split(",")]
    if "*" in text:
        if len(systems) > 1:
            raise _MPStubError(400, {"detail": "Wild cards only supported for single "
                                               "chemsys queries."})
        parts = systems[0].split("-")
        real = [p for p in parts if p != "*"]
        return lambda d: d["nelements"] == len(parts) and all(el in d["elements"] for el in real)
    wanted = {"-".join(sorted(s.split("-"))) for s in systems}
    return lambda d: d["chemsys"] in wanted


def mp_stub_answer(documents: list, params: dict, api_version: str) -> dict:
    """The body the live materials/summary endpoint returns for ``params``.

    Raises:
        _MPStubError: The status and body of the live API's error answer.
    """
    tests = []
    if "formula" in params:
        tests.append(_mp_formula_test(params["formula"]))
    if "chemsys" in params:
        tests.append(_mp_chemsys_test(params["chemsys"]))
    if "elements" in params:
        wanted = _mp_element_list(params, "elements")
        tests.append(lambda d: all(el in d["elements"] for el in wanted))
    if "exclude_elements" in params:
        unwanted = _mp_element_list(params, "exclude_elements")
        tests.append(lambda d: not any(el in d["elements"] for el in unwanted))
    asked = None
    if "material_ids" in params:
        asked = {}
        for material_id in params["material_ids"].split(","):
            asked.setdefault(_mp_alpha_id(material_id), len(asked))
        tests.append(lambda d: d["material_id"] in asked)
    for name in ("theoretical", "is_stable"):
        if name in params:
            flag = _mp_parse(params, name, "bool")
            tests.append(lambda d, n=name, f=flag: d.get(n) is f)
    for field, kind in (("energy_above_hull", "float"), ("band_gap", "float"), ("nsites", "int")):
        for bound in ("min", "max"):
            if f"{field}_{bound}" not in params:
                continue
            limit = _mp_parse(params, f"{field}_{bound}", kind)
            if bound == "min":
                tests.append(lambda d, f=field, x=limit: d.get(f) is not None and d[f] >= x)
            else:
                tests.append(lambda d, f=field, x=limit: d.get(f) is not None and d[f] <= x)
    if "spacegroup_number" in params:
        try:
            number = int(params["spacegroup_number"])
        except ValueError:
            raise _MPStubError(500, "Internal Server Error") from None
        tests.append(lambda d: d["symmetry"]["number"] == number)
    deprecated = _mp_parse(params, "deprecated", "bool") if "deprecated" in params else False
    tests.append(lambda d: bool(d.get("deprecated")) is deprecated)

    matches = [d for d in documents if all(test(d) for test in tests)]
    if asked is not None:  # unsorted, the IDs come back in the order asked for
        matches.sort(key=lambda d: asked[d["material_id"]])

    try:
        limit = int(params.get("_limit", MP_DEFAULT_LIMIT))
    except ValueError:
        limit = MP_DEFAULT_LIMIT  # the live API ignores an unreadable _limit
    skip = _mp_parse(params, "_skip", "int") if "_skip" in params else 0
    if limit > MP_MAX_LIMIT:
        raise _MPStubError(400, {"detail": "Requested more data per query than allowed by this "
                                           f"endpoint. The max limit is {MP_MAX_LIMIT} entries"})
    if skip < 0 or limit < 0:
        raise _MPStubError(400, {"detail": "Cannot request negative _skip or _limit values"})

    sort_fields = [f for f in params.get("_sort_fields", "").split(",") if f]
    if len(sort_fields) > 1:
        raise _MPStubError(400, {"detail": "Please provide at most 1 field(s) to sort with"})
    if sort_fields and sort_fields[0].lstrip("-") in MP_SORTABLE:
        field = sort_fields[0].lstrip("-")
        window = skip + limit
        if limit and window < len(matches):
            # a sort cut by a limit is a top-k sort in MongoDB: equal keys come in an
            # order that depends on the window, so consecutive pages can repeat or
            # drop documents of equal key
            rank = {d["material_id"]:
                    hashlib.sha256(f"{window}:{d['material_id']}".encode()).digest()
                    for d in matches}
            matches = sorted(matches, key=lambda d: rank[d["material_id"]])

        def key(doc):
            value = doc.get(field)
            return (value is not None, value if value is not None else 0)  # null first

        ascending = sorted(matches, key=key)
        if not sort_fields[0].startswith("-"):
            matches = ascending
        elif any(isinstance(d.get(field), str) for d in matches):
            matches = ascending[::-1]  # the live API walks its string index backwards
        else:
            matches = sorted(matches, key=key, reverse=True)

    page = matches[skip:skip + limit] if limit else matches[skip:]
    fields = [f.strip() for f in params.get("_fields", "material_id").split(",") if f.strip()]
    data = []
    for doc in page:
        if "structure" in fields and "structure" not in doc:
            raise _MPStubError(500, {"detail": "testsuite stub: the fixture holds no structure "
                                               f"for {doc['material_id']}"})
        data.append({f: doc[f] for f in fields if f in doc})
    return {"data": data, "meta": {
        "api_version": api_version,
        "time_stamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_doc": len(matches), "facet": None, "max_limit": MP_MAX_LIMIT,
        "default_fields": ["material_id"]}}


class _MPStubHandler(http.server.BaseHTTPRequestHandler):
    """GET /materials/summary/ answered from the fixture, as the live API answers it."""

    def do_GET(self) -> None:  # noqa: N802 (the http.server naming)
        url = urlsplit(self.path)
        params = {name: values[-1] for name, values in
                  parse_qs(url.query, keep_blank_values=True).items()}
        key = self.headers.get("x-api-key")
        self.server.requests.append({"path": url.path, "params": params,
                                     "key_ok": key == MP_STUB_KEY,
                                     "user_agent": self.headers.get("user-agent", "")})
        try:
            if not key:
                raise _MPStubError(401, {"message": "No API key found in request"})
            if key != MP_STUB_KEY:
                raise _MPStubError(401, {"message": "Invalid authentication credentials"})
            if url.path != MP_SUMMARY_PATH:
                raise _MPStubError(404, {"detail": "Not Found"})
            status, body = 200, mp_stub_answer(self.server.documents, params,
                                               self.server.api_version)
        except _MPStubError as exc:
            status, body = exc.status, exc.body
        text = body if isinstance(body, str) else json.dumps(body)
        payload = text.encode()
        self.send_response(status)
        self.send_header("content-type",
                         "text/plain" if isinstance(body, str) else "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:  # keep the test output clean
        pass


def start_mp_stub(fixture_path: str) -> http.server.ThreadingHTTPServer:
    """Serve the fixture on 127.0.0.1 (a free port) from a daemon thread."""
    import pymatgen.core  # noqa: F401  (formula queries; loaded here, not in a handler thread)

    with gzip.open(fixture_path, "rt", encoding="utf-8") as handle:
        fixture = json.load(handle)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _MPStubHandler)
    server.daemon_threads = True
    server.documents = fixture["documents"]
    server.api_version = fixture["api_version"]
    server.requests = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


_MP_DEAD_PROXY = "http://127.0.0.1:9"  # the discard port: nothing listens there
_MP_TABLE_HEADER = ("Formula", "Space group", "Material ID", "Band Gap (eV)",
                    "Energy Above Hull (eV/atom)", "Sites")


def _mp_env(home: str, endpoint: str, key: str | None = MP_STUB_KEY) -> dict:
    """Environment of a crystod-search run against ``endpoint``, offline."""
    env = {name: value for name, value in os.environ.items()
           if not name.startswith("PMG_") and name.lower() not in (
               "mp_api_key", "mp_api_endpoint", "http_proxy", "https_proxy", "all_proxy",
               "no_proxy")}
    env.update(HOME=home, MP_API_ENDPOINT=endpoint)
    for name in ("http_proxy", "https_proxy", "all_proxy"):
        env[name] = env[name.upper()] = _MP_DEAD_PROXY
    env["no_proxy"] = env["NO_PROXY"] = "127.0.0.1,localhost"
    if key is not None:
        env["MP_API_KEY"] = key
    return env


def _mp_rows(out: str) -> list[dict]:
    """The rows of the crystod-search table in ``out``."""
    lines = out.splitlines()
    start = next((i for i, line in enumerate(lines)
                  if line.split()[:1] == ["Formula"] and "Material ID" in line), None)
    rows = []
    if start is None:
        return rows
    for line in lines[start + 1:]:
        tokens = line.split()
        if not tokens:
            break
        star = len(tokens) == 7 and tokens[3] == "*"
        if star:
            del tokens[3]
        try:
            formula, symbol, material_id, gap, ehull, sites = tokens
            rows.append({"formula": formula, "spg": symbol, "id": material_id, "star": star,
                         "gap": None if gap == "-" else float(gap),
                         "ehull": None if ehull == "-" else float(ehull), "sites": int(sites)})
        except ValueError:  # not a table row: skipped, so the row counts fail instead
            continue
    return rows


def _mp_elements(formula: str) -> set:
    return set(re.findall(r"[A-Z][a-z]?", formula))


def _mp_id_value(material_id: str) -> int:
    body = material_id.split("-", 1)[1]
    if body.isdigit():
        return int(body)
    value = 0
    for letter in body:
        value = value * 26 + ord(letter) - ord("a")
    return value


def _mp_http(url: str, key: str | None = MP_STUB_KEY) -> tuple[int, object]:
    """GET ``url`` directly (no proxy); (status, decoded JSON body)."""
    request = urllib.request.Request(url, headers={"x-api-key": key} if key else {})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _mp_read(path: str) -> str:
    """The text of ``path``, or "" when there is no such file."""
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        return ""


def _poscar_structure(path: str):
    """(pymatgen Structure, ASE Atoms, spglib number at 1e-3, comment line) of a POSCAR;
    (None, [], 0, "") when the file is missing."""
    if not os.path.isfile(path):
        return None, [], 0, ""
    import ase.io
    import spglib
    from pymatgen.core import Structure

    structure = Structure.from_file(path)
    atoms = ase.io.read(path, format="vasp")
    cell = (atoms.cell[:], atoms.get_scaled_positions(), atoms.numbers)
    dataset = spglib.get_symmetry_dataset(cell, symprec=1e-3)
    number = getattr(dataset, "number", None) or dataset["number"]
    with open(path, encoding="utf-8") as handle:
        comment = handle.readline().rstrip("\n")
    return structure, atoms, number, comment


def test_45_search() -> None:
    print("\n[45] crystod-search (Materials Project search and POSCAR download, offline stub)")
    if not os.path.isfile(MP_FIXTURE):
        report("Materials Project fixture found", False, MP_FIXTURE)
        return
    with gzip.open(MP_FIXTURE, "rt", encoding="utf-8") as handle:
        fixture = json.load(handle)
    current = [d for d in fixture["documents"] if not d["deprecated"]]
    spg_number = {d["symmetry"]["symbol"]: d["symmetry"]["number"] for d in current}
    ternary_starred = {"mp-5532", "mp-3349", "mp-28740", "mp-540640", "mp-5229", "mp-31213"}
    srtio3_ids = ["mp-4651", "mp-551830", "mp-5229", "mp-776018", "mp-aaaieiuj"]
    reference_poscar = os.path.join(MP_SEARCH_DIR, "POSCAR_Sr2TiO4_I4mmm_mp-5532")
    reference_pposcar = os.path.join(MP_SEARCH_DIR, "PPOSCAR_Sr2TiO4_I4mmm_mp-5532")

    server = start_mp_stub(MP_FIXTURE)
    endpoint = f"http://127.0.0.1:{server.server_address[1]}"
    requests_log = server.requests
    home = tempfile.mkdtemp(prefix="crystod_home_")
    env = _mp_env(home, endpoint)

    def search(args: list[str], cwd: str | None = None, environment: dict | None = None):
        return run_search(args, cwd=cwd or home, env=environment or env)

    def refused(code: int, out: str, text: str) -> bool:
        return code != 0 and text in out and "Traceback" not in out

    try:
        # ---- the stub itself answers like the live API
        status, body = _mp_http(f"{endpoint}/materials/summary/?chemsys=Sr-Ti-O&_limit=1001")
        ok = status == 400 and body == {"detail": "Requested more data per query than allowed "
                                                  "by this endpoint. The max limit is 1000 entries"}
        status, body = _mp_http(f"{endpoint}/materials/summary/?chemsys=Sr-Ti-O", key=None)
        ok = ok and status == 401 and body == {"message": "No API key found in request"}
        status, body = _mp_http(f"{endpoint}/materials/summary/?material_ids=mp-5229,mp-aaaaaieu")
        ok = ok and status == 200 and [d["material_id"] for d in body["data"]] == [
            "mp-aaaaahtd", "mp-aaaaaieu"] and body["meta"]["total_doc"] == 2
        report("stub: 400 above _limit 1000, 401 without a key, alphabetical IDs out", ok,
               json.dumps(body)[:500])
        requests_log.clear()

        # ---- a formula: the SrTiO3 table of the website
        code, out = search(["SrTiO3"])
        rows = _mp_rows(out)
        report("crystod-search SrTiO3 exit 0, answered by the local stub",
               code == 0 and len(requests_log) == 1 and requests_log[0]["key_ok"], out)
        header = next((line for line in out.splitlines() if "Material ID" in line), "")
        positions = [header.find(label) for label in _MP_TABLE_HEADER]
        report("table header: Formula, Space group, Material ID, Band Gap (eV), "
               "Energy Above Hull (eV/atom), Sites",
               -1 not in positions and positions == sorted(positions), header)
        report("SrTiO3: five polymorphs in energy-above-hull order, IDs as on the website",
               [r["id"] for r in rows] == srtio3_ids
               and [r["spg"] for r in rows] == ["I4/mcm", "I4/mcm", "Pm-3m", "P6_3/mmc", "R-3"],
               out)
        report("the experimentally observed mp-5229 is starred ('mp-5229 *'), no other",
               "SrTiO3   Pm-3m        mp-5229 *            1.766" in out
               and [r["id"] for r in rows if r["star"]] == ["mp-5229"], out)
        report("band gap, energy above hull and sites of each row (mp-776018: 1.736, 0.039, 30)",
               any(r["id"] == "mp-776018" and r["gap"] == 1.736 and r["ehull"] == 0.039
                   and r["sites"] == 30 for r in rows), out)
        report("summary line, download hint, star legend and Materials Project citation",
               "5 materials, sorted by energy above hull" in out
               and "crystod-search --get mp-4651" in out and "* experimentally observed" in out
               and "A. Jain et al., APL Mater. 1, 011002 (2013)" in out, out)
        params = requests_log[0]["params"] if requests_log else {}
        report("request: x-api-key, CrystOD user agent, formula, deprecated=false, table "
               "fields, _sort_fields=energy_above_hull",
               params.get("formula") == "SrTiO3" and params.get("deprecated") == "false"
               and params.get("_sort_fields") == "energy_above_hull"
               and set(params.get("_fields", "").split(",")) >= {
                   "material_id", "formula_pretty", "symmetry", "band_gap",
                   "energy_above_hull", "nsites", "theoretical"}
               and requests_log[0]["user_agent"].startswith("CrystOD/"), str(requests_log[:1]))

        # ---- a chemical system: the ternary compounds of exactly Sr, Ti and O
        code, out = search(["Sr-Ti-O"])
        rows = _mp_rows(out)
        report("Sr-Ti-O: 23 ternary compounds, each made of exactly Sr, Ti and O",
               code == 0 and len(rows) == 23 and "23 materials" in out
               and "chemical system Sr-Ti-O (compounds of exactly these elements)" in out
               and all(_mp_elements(r["formula"]) == {"Sr", "Ti", "O"} for r in rows), out)
        report("Sr-Ti-O: exactly six starred (mp-5532, mp-3349, mp-28740, mp-540640, "
               "mp-5229, mp-31213)",
               {r["id"] for r in rows if r["star"]} == ternary_starred
               and sum(r["star"] for r in rows) == 6, out)
        report("Sr-Ti-O: sorted by the energy above the hull",
               [r["ehull"] for r in rows] == sorted(r["ehull"] for r in rows), out)
        all_ternary = {r["id"] for r in rows}

        code, out = search(["Sr-Ti-O", "--experimental"])
        rows = _mp_rows(out)
        code2, out2 = search(["Sr-Ti-O", "--exp"])
        report("--experimental (alias --exp): the six starred ternaries only",
               code == 0 and {r["id"] for r in rows} == ternary_starred
               and all(r["star"] for r in rows) and "experimentally observed only" in out
               and requests_log[-2]["params"].get("theoretical") == "false"
               and code2 == 0 and _mp_rows(out2) == rows, out)

        code, out = search(["Sr-Ti-O", "--stable"])
        rows = _mp_rows(out)
        report("--stable: the four ternaries on the hull (is_stable=true)",
               code == 0 and {r["id"] for r in rows} == {"mp-5532", "mp-3349", "mp-28740",
                                                         "mp-4651"}
               and all(r["ehull"] == 0.0 for r in rows)
               and requests_log[-1]["params"].get("is_stable") == "true", out)

        code, out = search(["Sr-Ti-O", "--ehull", "0.05"])
        rows = _mp_rows(out)
        report("--ehull 0.05: 14 rows, none above 0.05 eV/atom",
               code == 0 and len(rows) == 14 and all(r["ehull"] <= 0.05 for r in rows)
               and "energy above hull <= 0.05 eV/atom" in out, out)

        code, out = search(["Sr-Ti-O", "--band-gap", "1", "2"])
        rows = _mp_rows(out)
        code2, out2 = search(["Sr-Ti-O", "--sites", "5", "20"])
        rows2 = _mp_rows(out2)
        report("--band-gap 1 2 (15 rows) and --sites 5 20 (10 rows) stay in range",
               code == 0 and len(rows) == 15 and all(1 <= r["gap"] <= 2 for r in rows)
               and code2 == 0 and len(rows2) == 10 and all(5 <= r["sites"] <= 20 for r in rows2),
               out + out2)

        code, out = search(["Sr-Ti-O", "--spg", "P63/mmc"])
        rows = _mp_rows(out)
        code2, out2 = search(["Sr-Ti-O", "--spg", "194"])
        report("--spg P63/mmc and --spg 194: the hexagonal SrTiO3 mp-776018 only",
               code == 0 and [r["id"] for r in rows] == ["mp-776018"]
               and "space group P6_3/mmc (No. 194)" in out
               and requests_log[-2]["params"].get("spacegroup_number") == "194"
               and code2 == 0 and _mp_rows(out2) == rows, out + out2)

        code, out = search(["Sr-Ti-O", "--spg", "Fm-3m"])
        report("a filter nothing matches prints 'no material matches.' and exits 0",
               code == 0 and "no material matches." in out and not _mp_rows(out)
               and "Download a POSCAR" not in out, out)

        code, out = search(["Sr-Ti-O", "--subsystems"])
        rows = _mp_rows(out)
        report("--subsystems: all 216 materials of Sr-Ti-O, its binaries and elements",
               code == 0 and len(rows) == 216 and len(current) == 216
               and "chemical system Sr-Ti-O and all its subsystems" in out
               and {r["formula"] for r in rows} >= {"Sr", "Ti", "O2", "SrO", "TiO2", "SrTiO3"}
               and requests_log[-1]["params"].get("chemsys")
               == "O,Sr,Ti,O-Sr,O-Ti,Sr-Ti,O-Sr-Ti", out[-2000:])
        code, out = search(["Sr-Ti-O", "--subsystems", "--exclude", "Ti"])
        rows = _mp_rows(out)
        report("--exclude Ti: 58 materials of O, Sr and O-Sr, none with Ti",
               code == 0 and len(rows) == 58 and not any("Ti" in _mp_elements(r["formula"])
                                                         for r in rows)
               and "without Ti" in out, out[-2000:])

        # ---- the other query forms of the website's search box
        code, out = search(["Sr,Ti,O"])
        rows = _mp_rows(out)
        report("Sr,Ti,O: materials containing Sr, Ti and O (elements=Sr,Ti,O)",
               code == 0 and "materials containing Sr, Ti and O" in out and len(rows) == 23
               and requests_log[-1]["params"].get("elements") == "Sr,Ti,O"
               and all(_mp_elements(r["formula"]) >= {"Sr", "Ti", "O"} for r in rows), out)
        code, out = search(["ABO3"])
        code2, out2 = search(["*TiO3"])
        report("anonymous formula ABO3 and wildcard formula *TiO3 (fixture: the SrTiO3 polymorphs)",
               code == 0 and "anonymous formula ABO3 (each letter stands for any element)" in out
               and [r["id"] for r in _mp_rows(out)] == srtio3_ids
               and code2 == 0 and "formula *TiO3" in out2
               and [r["id"] for r in _mp_rows(out2)] == srtio3_ids, out + out2)
        code, out = search(["Sr-*"])
        rows = _mp_rows(out)
        report("wildcard chemical system Sr-*: the Sr binaries (O-Sr and Sr-Ti)",
               code == 0 and "each * stands for one more element" in out and len(rows) == 20
               and all(len(_mp_elements(r["formula"])) == 2 and "Sr" in r["formula"]
                       for r in rows), out)
        code, out = search(["mp-5229,mp-5532"])
        code2, out2 = search(["mp-aaaaahtd"])
        report("ID queries: mp-5229,mp-5532 lists both; mp-aaaaahtd is shown as 'mp-5229 *'",
               code == 0 and "material IDs mp-5229, mp-5532" in out
               and {r["id"] for r in _mp_rows(out)} == {"mp-5229", "mp-5532"}
               and code2 == 0 and "material ID mp-5229" in out2
               and [r["id"] for r in _mp_rows(out2)] == ["mp-5229"] and "mp-5229 *" in out2,
               out + out2)

        # ---- sorting and truncation
        for key, field, value in (("gap", "band_gap", lambda r: r["gap"]),
                                  ("sites", "nsites", lambda r: r["sites"]),
                                  ("id", "material_id", lambda r: _mp_id_value(r["id"])),
                                  ("formula", "formula_pretty", lambda r: r["formula"]),
                                  ("spg", "symmetry.number", lambda r: spg_number[r["spg"]])):
            # the server cannot sort on the nested symmetry.number: spg fetches
            # the list in ID order and sorts it locally
            sent = "material_id" if key == "spg" else field
            code, out = search(["Sr-Ti-O", "--sort", key])
            rows = _mp_rows(out)
            values = [value(r) for r in rows]
            report(f"--sort {key}: the 23 ternaries in ascending {field} "
                   f"(_sort_fields={sent})",
                   code == 0 and {r["id"] for r in rows} == all_ternary
                   and values == sorted(values)
                   and requests_log[-1]["params"].get("_sort_fields") == sent,
                   f"{field} in the listed order: {values}\n"
                   f"IDs: {[r['id'] for r in rows]}")

        code, out = search(["Sr-Ti-O", "--max", "3"])
        rows = _mp_rows(out)
        report("--max 3: three rows on the hull and a truncation note (3 of 23)",
               code == 0 and len(rows) == 3 and all(r["ehull"] == 0.0 for r in rows)
               and {r["id"] for r in rows} <= {"mp-5532", "mp-3349", "mp-28740", "mp-4651"}
               and "3 of 23 materials listed, sorted by energy above hull; narrow the query "
                   "or raise --max to see more." in out
               and requests_log[-1]["params"].get("_limit") == "1000", out)

        # ---- downloading POSCAR files
        with tempfile.TemporaryDirectory() as tmp:
            code, out = search(["--get", "mp-5532"], cwd=tmp)
            path = os.path.join(tmp, "POSCAR_Sr2TiO4_I4mmm_mp-5532")
            report("--get mp-5532 writes POSCAR_Sr2TiO4_I4mmm_mp-5532 into the current directory",
                   code == 0 and os.path.isfile(path)
                   and os.listdir(tmp) == [os.path.basename(path)]
                   and "Wrote POSCAR_Sr2TiO4_I4mmm_mp-5532: Sr2TiO4, I4/mmm, 14 atoms" in out, out)
            downloaded = _mp_read(path)
            report("the file equals the POSCAR a live run wrote (example/45_mp_search)",
                   downloaded != "" and downloaded == _mp_read(reference_poscar), downloaded)
            structure, atoms, number, comment = _poscar_structure(path)
            report("the POSCAR holds the conventional cell: Sr2TiO4, 14 atoms, a = b, "
                   "orthogonal axes, spglib No. 139 at 1e-3 (pymatgen and ASE)",
                   structure is not None and len(structure) == 14 and len(atoms) == 14
                   and structure.composition.reduced_formula == "Sr2TiO4" and number == 139
                   and abs(structure.lattice.a - structure.lattice.b) < 1e-9
                   and np.allclose(structure.lattice.angles, 90.0),
                   f"{structure and structure.composition} {len(atoms)} {number}")
            report("comment line names formula, space group, ID and cell",
                   comment == "Sr2TiO4 I4/mmm mp-5532 (Materials Project, conventional cell)",
                   comment)
            last = requests_log[-1]["params"]
            report("the download asks for exactly that ID with the structure field",
                   last.get("material_ids") in ("mp-5532", "mp-aaaaaieu")
                   and "structure" in last.get("_fields", "").split(","), str(last))

            code, out = search(["--get", "mp-5532"], cwd=tmp)
            report("running it again keeps the identical file ('Kept', exit 0)",
                   code == 0 and "Kept POSCAR_Sr2TiO4_I4mmm_mp-5532 (identical file already there)"
                   in out, out)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write("edited by the user\n")
            code, out = search(["--get", "mp-5532"], cwd=tmp)
            edited = _mp_read(path)
            report("a different file of that name is refused (exit != 0, no traceback, untouched)",
                   refused(code, out, "already exists with different content")
                   and edited.endswith("edited by the user\n"), out)
            code, out = search(["--get", "mp-5532", "--force"], cwd=tmp)
            replaced = _mp_read(path)
            report("--force replaces it ('Replaced')",
                   code == 0 and "Replaced POSCAR_Sr2TiO4_I4mmm_mp-5532" in out
                   and replaced == downloaded, out)

            code, out = search(["--get", "mp-5532", "-o", "STO214.vasp"], cwd=tmp)
            report("-o NAME writes that file",
                   code == 0 and "Wrote STO214.vasp" in out
                   and _mp_read(os.path.join(tmp, "STO214.vasp")) == downloaded, out)
            code, out = search(["--get", "mp-5532", "--directory"], cwd=tmp)
            report("--directory writes Sr2TiO4_I4mmm_mp-5532/POSCAR",
                   code == 0
                   and os.path.isfile(os.path.join(tmp, "Sr2TiO4_I4mmm_mp-5532", "POSCAR"))
                   and "Wrote Sr2TiO4_I4mmm_mp-5532/POSCAR" in out.replace(os.sep, "/"), out)

        with tempfile.TemporaryDirectory() as tmp:
            code, out = search(["--get", "mp-5532", "--cell", "primitive"], cwd=tmp)
            path = os.path.join(tmp, "PPOSCAR_Sr2TiO4_I4mmm_mp-5532")
            structure, atoms, number, comment = _poscar_structure(path)
            report("--cell primitive writes PPOSCAR_Sr2TiO4_I4mmm_mp-5532: 7 atoms, still No. 139",
                   code == 0 and os.listdir(tmp) == [os.path.basename(path)]
                   and "Wrote PPOSCAR_Sr2TiO4_I4mmm_mp-5532: Sr2TiO4, I4/mmm, 7 atoms" in out
                   and len(atoms) == 7 and number == 139
                   and comment.endswith("(Materials Project, primitive cell)"), out)
            report("the PPOSCAR equals the one a live run wrote (example/45_mp_search)",
                   _mp_read(path) != "" and _mp_read(path) == _mp_read(reference_pposcar),
                   _mp_read(path))
            code, out = search(["--get", "mp-5532"], cwd=tmp)
            report("POSCAR_ (conventional) and PPOSCAR_ (primitive) of one material live "
                   "side by side",
                   code == 0 and sorted(os.listdir(tmp)) == ["POSCAR_Sr2TiO4_I4mmm_mp-5532",
                                                             "PPOSCAR_Sr2TiO4_I4mmm_mp-5532"], out)
            code, out = search(["--get", "mp-5532", "--cell", "primitive", "--directory"], cwd=tmp)
            report("--cell primitive --directory writes Sr2TiO4_I4mmm_mp-5532/PPOSCAR",
                   code == 0
                   and os.path.isfile(os.path.join(tmp, "Sr2TiO4_I4mmm_mp-5532", "PPOSCAR"))
                   and "Wrote Sr2TiO4_I4mmm_mp-5532/PPOSCAR" in out.replace(os.sep, "/"), out)

        with tempfile.TemporaryDirectory() as tmp:
            code, out = search(["--get", "mp-aaaaahtd"], cwd=tmp)
            code2, out2 = search(["--get", "mp-3732049", "mp-776018"], cwd=tmp)
            report("--get mp-aaaaahtd writes POSCAR_SrTiO3_Pm-3m_mp-5229 (website spelling)",
                   code == 0 and os.path.isfile(os.path.join(tmp, "POSCAR_SrTiO3_Pm-3m_mp-5229"))
                   and "5 atoms" in out, out)
            report("--get mp-3732049 mp-776018: POSCAR_SrTiO3_R-3_mp-aaaieiuj and "
                   "POSCAR_SrTiO3_P63mmc_mp-776018",
                   code2 == 0
                   and os.path.isfile(os.path.join(tmp, "POSCAR_SrTiO3_R-3_mp-aaaieiuj"))
                   and os.path.isfile(os.path.join(tmp, "POSCAR_SrTiO3_P63mmc_mp-776018"))
                   and "WARNING" not in out2, out2)

        with tempfile.TemporaryDirectory() as tmp:
            code, out = search(["SrTiO3", "--get"], cwd=tmp)
            report("a query with a bare --get downloads every listed material (5 POSCARs)",
                   code == 0 and sorted(os.listdir(tmp)) == sorted([
                       "POSCAR_SrTiO3_I4mcm_mp-4651", "POSCAR_SrTiO3_I4mcm_mp-551830",
                       "POSCAR_SrTiO3_Pm-3m_mp-5229", "POSCAR_SrTiO3_P63mmc_mp-776018",
                       "POSCAR_SrTiO3_R-3_mp-aaaieiuj"]), out)
            code, out = search(["Sr-Ti-O", "--experimental", "--get", "--directory"], cwd=tmp)
            made = sorted(name for name in os.listdir(tmp)
                          if os.path.isfile(os.path.join(tmp, name, "POSCAR")))
            report("Sr-Ti-O --experimental --get --directory: six {formula}_{spg}_{ID}/POSCAR",
                   code == 0 and len(made) == 6 and "Sr2TiO4_I4mmm_mp-5532" in made
                   and "SrTiO3_Pm-3m_mp-5229" in made, out + str(made))

        with tempfile.TemporaryDirectory() as tmp:
            count = len(requests_log)
            code, out = search(["--get", "mp-5229", "MP-999999999"], cwd=tmp)
            report("an unknown ID is named as typed; nothing is written",
                   refused(code, out, "the Materials Project has no material MP-999999999")
                   and os.listdir(tmp) == [] and len(requests_log) > count, out)
            code, out = search(["--get", "mp-aaaditbh"], cwd=tmp)
            report("a deprecated entry is not served (the API hides it) and is reported cleanly",
                   refused(code, out, "has no material mp-aaaditbh")
                   and "deprecated entries are not served" in out, out)
            code, out = search(["--get", "mp-5229", "mp-aaaaahtd", "-o", "STO.vasp"], cwd=tmp)
            report("-o accepts two spellings of one material (mp-5229 = mp-aaaaahtd)",
                   code == 0 and os.path.isfile(os.path.join(tmp, "STO.vasp")), out)

        # ---- bad input is refused before any request
        count = len(requests_log)
        for args, text in (
                (["--get", "5229"], "'5229' is not a Materials Project ID"),
                (["--get", "mp-abc1"], "'mp-abc1' is not a Materials Project ID"),
                (["srtio3"], "element symbols start with a capital letter"),
                (["Sr-Xx-O"], "'Xx' in 'Sr-Xx-O' is not an element symbol"),
                (["Sr,Ti,O", "--subsystems"], "--subsystems needs a chemical system"),
                (["Sr-Ti-O", "--spg", "231"], "space-group number 231 is not in 1-230"),
                (["Sr-Ti-O", "--spg", "P7"], "unknown space group 'P7'"),
                (["Sr-Ti-O", "--ehull", "-1"], "(--ehull) must be a number >= 0"),
                (["Sr-Ti-O", "--ehull", "nan"], "(--ehull) must be a number >= 0"),
                (["Sr-Ti-O", "--band-gap", "3", "1"], "(--band-gap MIN MAX) needs 0 <= MIN <= MAX"),
                (["Sr-Ti-O", "--band-gap", "-1", "1"], "(--band-gap MIN MAX) needs 0 <= MIN <= MAX"),
                (["Sr - Ti - O", "--spg", "P7"], "unknown space group 'P7'"),
                (["Ti-O-O"], "'Ti-O-O' names O more than once"),
                (["CuSO4.5H2O"], "matches integer formulas only"),
                (["mvc-123"], "knows mp- IDs only"),
                (["Sr-Ti-O", "--exclude", ","], "(--exclude) name no element"),
                (["Sr-Ti-O", "--exclude", "Xx"], "'Xx' in '--exclude' is not an element symbol"),
                (["Sr-Ti-O", "--max", "0"], "must be at least 1")):
            code, out = search(args)
            report(f"refused cleanly: {' '.join(args)}", refused(code, out, text), out)
        report("none of the refused inputs sent a request", len(requests_log) == count,
               str(requests_log[count:]))

        # ---- command-line misuse (argparse errors, exit 2)
        for args, text in (
                (["SrTiO3", "--get", "mp-5229"], "not both"),
                (["--get"], "--get needs material IDs"),
                ([], "give a query"),
                (["--get", "mp-5229", "--experimental"], "--experimental filters a search"),
                (["--get", "mp-5229", "mp-5532", "-o", "x"], "-o/--output names one file"),
                (["--get", "mp-5229", "-o", "x", "--directory"], "exclude each other"),
                (["Sr-Ti-O", "--sort", "density"], "invalid choice")):
            code, out = search(args)
            report(f"argparse error: crystod-search {' '.join(args)}",
                   code == 2 and text in out and "Traceback" not in out, out)
        code, out = search(["SrTiO3", "--get", "-o", "x"])
        report("a query listing five materials with --get -o is refused after the table",
               code == 2 and "names one file but 5 materials are listed" in out
               and not os.path.exists(os.path.join(home, "x")), out)

        # ---- API key and network problems
        count = len(requests_log)
        code, out = search(["SrTiO3"], environment=_mp_env(home, endpoint, key=None))
        report("no key anywhere: ERROR with the URL that issues one, no request sent",
               refused(code, out, "no Materials Project API key found")
               and "https://next-gen.materialsproject.org/api" in out
               and len(requests_log) == count, out)
        code, out = search(["SrTiO3"], environment=_mp_env(home, endpoint, key="0123456789abcdef"))
        report("a 16-character key is recognized as a key of the retired legacy API",
               refused(code, out, "has 16 characters")
               and "16-character keys belong to the retired legacy API" in out, out)
        code, out = search(["SrTiO3"], environment=_mp_env(home, endpoint, key="x" * 32))
        report("a wrong key: the 401 answer of the server is reported",
               refused(code, out, "rejected the API key (Invalid authentication credentials)"), out)
        with tempfile.TemporaryDirectory() as other_home:
            with open(os.path.join(other_home, ".pmgrc.yaml"), "w", encoding="utf-8") as handle:
                handle.write(f"PMG_MAPI_KEY: {MP_STUB_KEY}\n")
            code, out = search(["SrTiO3"], environment=_mp_env(other_home, endpoint, key=None))
            report("PMG_MAPI_KEY of ~/.pmgrc.yaml is used when MP_API_KEY is not set",
                   code == 0 and "mp-5229 *" in out, out)
        with socket.socket() as probe_socket:
            probe_socket.bind(("127.0.0.1", 0))
            closed = probe_socket.getsockname()[1]
        code, out = search(["SrTiO3"], environment=_mp_env(home, f"http://127.0.0.1:{closed}"))
        report("an unreachable server: ERROR naming it, no traceback",
               refused(code, out, f"could not reach the Materials Project API at "
                                  f"http://127.0.0.1:{closed}"), out)
        many = ["Ac", "Ag", "Al", "Am", "Ar", "As", "At", "Au", "Ba", "Be", "Bh", "Bi", "Bk",
                "Br", "Ca", "Cd", "Ce", "Cf", "Cl", "Cm", "Co"]
        code, out = search(["Sr-Ti-O", "--exclude", *many])
        report("a server-side rejection (--exclude longer than 60 characters) is an ERROR line",
               refused(code, out, "the Materials Project API answered HTTP 422")
               and "at most 60 characters" in out, out)

        # ---- --example (no input files; the searches need the stub)
        with tempfile.TemporaryDirectory() as tmp:
            count = len(requests_log)
            code, out = search(["--example"], cwd=tmp)
            report("--example lists SrTiO3, Sr-Ti-O and mp-5229 without any request",
                   code == 0 and all(name in out for name in ("SrTiO3", "Sr-Ti-O", "mp-5229"))
                   and "= crystod-search --get mp-5229" in out and len(requests_log) == count,
                   out)
            code, out = search(["--example", "SrTiO3"], cwd=tmp)
            report("--example SrTiO3 runs 'crystod-search SrTiO3' through the stub",
                   code == 0 and "Running: crystod-search SrTiO3" in out and "mp-5229 *" in out,
                   out)
            code, out = search(["--example", "mp-5229"], cwd=tmp)
            report("--example mp-5229 writes POSCAR_SrTiO3_Pm-3m_mp-5229",
                   code == 0 and os.path.isfile(os.path.join(tmp, "POSCAR_SrTiO3_Pm-3m_mp-5229")),
                   out)
        code, out = run_cli(["--search", "SrTiO3"])
        report("crystod --search points to crystod-search",
               code != 0 and "crystod-search QUERY" in out and "Traceback" not in out, out)

        # ---- Python API (crystod.search)
        probe = (
            "import sys\n"
            "import crystod.search\n"
            "from crystod import search\n"
            "print('AFTER_IMPORT', [m for m in ('requests', 'pymatgen') if m in sys.modules])\n"
            "n = search.normalize_material_id\n"
            "print('IDS', n('mp-3347529'), n('mp-3347530'), n('mp-aaaaahtd'), n('mp-3732049'),\n"
            "      n('MP-5229'), n('mp-aaahilzd'))\n"
            "print('ROUNDTRIP', all(n(n(f'mp-{v}')) == n(f'mp-{v}') for v in\n"
            "      (1, 25, 26, 5229, 3347529, 3347530, 3732049, 10**9)))\n"
            "print('AFTER_IDS', [m for m in ('requests', 'pymatgen') if m in sys.modules])\n"
        )
        code, out = run_python(probe, env=env)
        report("import crystod.search (and normalize_material_id) loads neither requests "
               "nor pymatgen",
               code == 0 and "AFTER_IMPORT []" in out and "AFTER_IDS []" in out, out)
        report("normalize_material_id: numbers up to mp-3347529, letters above, both ways",
               "IDS mp-3347529 mp-aaahilze mp-5229 mp-aaaieiuj mp-5229 mp-3347529" in out
               and "ROUNDTRIP True" in out, out)
        probe = (
            "from crystod import search\n"
            "for call in (lambda: search.normalize_material_id('5229'),\n"
            "             lambda: search.interpret_query('srtio3'),\n"
            "             lambda: search.search_materials('Sr-Xx'),\n"
            "             lambda: search.resolve_space_group('P7'),\n"
            "             lambda: search.fetch_materials('mp-abc1')):\n"
            "    try:\n"
            "        call()\n"
            "        print('NOT RAISED')\n"
            "    except ValueError as exc:\n"
            "        print('VALUEERROR', str(exc)[:40])\n"
            "try:\n"
            "    search.search_materials('SrTiO3')\n"
            "except ValueError:\n"
            "    print('WRONG: ValueError')\n"
            "except search.MaterialsProjectError as exc:\n"
            "    print('MPERROR', isinstance(exc, RuntimeError),\n"
            "          'next-gen.materialsproject.org/api' in str(exc))\n"
            "print('SURVIVED')\n"
        )
        code, out = run_python(probe, env=_mp_env(home, endpoint, key=None))
        report("API: bad input raises ValueError through crystod.search",
               code == 0 and out.count("VALUEERROR") == 5 and "NOT RAISED" not in out, out)
        report("API: a missing key raises MaterialsProjectError (a RuntimeError), not ValueError",
               "MPERROR True True" in out and "SURVIVED" in out, out)
        probe = (
            "from crystod import search\n"
            "r = search.search_materials('Sr-Ti-O', experimental=True)\n"
            "m = {x.material_id: x for x in r.materials}['mp-5229']\n"
            "print('RESULT', r.total, len(r.materials), r.truncated, r.filters)\n"
            "print('LABEL', repr(m.label), m.formula, m.space_group, m.space_group_number,\n"
            "      m.experimental, m.url)\n"
            "print('TABLE', r.table_lines()[0].split()[:2])\n"
            "(f,) = search.fetch_materials('mp-aaaaahtd')\n"
            "print('FETCH', f.material_id, len(f.structure), f.cell, search.poscar_filename(f))\n"
        )
        code, out = run_python(probe, env=env)
        report("API: search_materials through the stub returns Material records "
               "(label 'mp-5229 *')",
               code == 0 and "RESULT 6 6 False ('experimentally observed only',)" in out
               and "LABEL 'mp-5229 *' SrTiO3 Pm-3m 221 True "
                   "https://next-gen.materialsproject.org/materials/mp-5229" in out
               and "TABLE ['Formula', 'Space']" in out, out)
        report("API: fetch_materials('mp-aaaaahtd') gives mp-5229, 5 atoms, conventional cell",
               "FETCH mp-5229 5 conventional POSCAR_SrTiO3_Pm-3m_mp-5229" in out, out)
        probe = (
            "from crystod import mp_search\n"
            "full = mp_search.search_materials('Sr-Ti-O', subsystems=True, sort='gap')\n"
            "mp_search._PAGE_SIZE = 20  # page through the same 216 materials 20 at a time\n"
            "paged = mp_search.search_materials('Sr-Ti-O', subsystems=True, sort='gap')\n"
            "ids = [m.material_id for m in paged.materials]\n"
            "print('FULL', len(full.materials), full.total)\n"
            "print('PAGED', len(ids), len(set(ids)), paged.total,\n"
            "      set(ids) == {m.material_id for m in full.materials})\n"
            "gaps = [m.band_gap for m in paged.materials]\n"
            "print('ORDERED', gaps == sorted(gaps))\n"
        )
        code, out = run_python(probe, env=env)
        report("paging with _skip lists every material exactly once (216 of 216, pages of 20, "
               "--sort gap)",
               code == 0 and "FULL 216 216" in out and "PAGED 216 216 216 True" in out
               and "ORDERED True" in out, out)

        report("every request reached the stub's endpoint with the dummy key (the one "
               "deliberately wrong key aside)",
               bool(requests_log) and all(r["path"] == MP_SUMMARY_PATH for r in requests_log)
               and sum(not r["key_ok"] for r in requests_log) == 1,
               f"{len(requests_log)} requests, "
               f"{sum(not r['key_ok'] for r in requests_log)} with another key")
    finally:
        server.shutdown()
        server.server_close()
        shutil.rmtree(home, ignore_errors=True)

    if os.environ.get("CRYSTOD_LIVE_MP") == "1":
        with tempfile.TemporaryDirectory() as tmp:
            code, out = run_search(["SrTiO3"], cwd=tmp)  # the real endpoint and the user's key
        report("live Materials Project: crystod-search SrTiO3 lists 'mp-5229 *'",
               code == 0 and "mp-5229 *" in out, out)
    else:
        print("  [SKIP] live Materials Project search (set CRYSTOD_LIVE_MP=1, with an API key, "
              "to run one)")


SECTIONS = {
    1: test_01_wigner_d,
    2: test_02_salc,
    3: test_03_hybridization,
    4: test_04_selection_rules,
    5: test_05_star_of_k,
    6: test_06_visualize_basis,
    7: test_07_main_command,
    8: test_08_direct_product,
    9: test_09_symmetric_square,
    10: test_10_decompose_irrep,
    11: test_11_ligand_field_split,
    12: test_12_basis_function,
    13: test_13_tensor_form,
    14: test_14_generate_basis_function,
    15: test_15_show_coset,
    16: test_16_correlate,
    17: test_17_isotropy,
    18: test_18_invariants,
    19: test_19_child_lookup,
    20: test_20_subgroup_graph,
    21: test_21_multiplet,
    22: test_22_poscar2cif,
    23: test_23_symmetry_mode,
    24: test_24_group_command,
    25: test_25_bz,
    26: test_26_bz_supercell,
    27: test_27_bz_command,
    28: test_28_phonon_irrep,
    29: test_29_phonon_activity,
    30: test_30_phonon_fatband,
    31: test_31_phonon_lt,
    32: test_32_phonon_vector,
    33: test_33_modulation,
    34: test_34_vibration,
    35: test_35_phonon_command,
    36: test_36_spin_basis,
    37: test_37_mag_command,
    38: test_38_xdatcar2adp,
    39: test_39_md_command,
    40: test_40_mol,
    41: test_41_molod,
    42: test_42_mol_command,
    43: test_43_python_api,
    44: test_44_xrd,
    45: test_45_search,
}


def main() -> None:
    for path, label in ((POSCAR_ScF3, "ScF3 POSCAR"), (POSCAR_SrTiO3, "SrTiO3 POSCAR")):
        if not os.path.isfile(path):
            print(f"ERROR: {label} not found: {path}")
            sys.exit(2)

    selected = sorted({int(arg) for arg in sys.argv[1:]}) if len(sys.argv) > 1 else sorted(SECTIONS)
    unknown = [number for number in selected if number not in SECTIONS]
    if unknown:
        print(f"ERROR: unknown section number(s): {unknown} (valid: 1-{max(SECTIONS)})")
        sys.exit(2)

    for number in selected:
        SECTIONS[number]()

    print(f"\n{'=' * 50}\n  Total: {PASS} passed, {FAIL} failed\n{'=' * 50}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
