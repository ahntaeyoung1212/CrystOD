#!/usr/bin/env python
"""
CrystOD full test suite (run inside the `crystod` conda env).

Usage:
    conda activate crystod
    cd ~/CrystOD-main
    python testsuite.py            # run everything
    python testsuite.py 8 12       # run only sections 8 and 12

Sections (grouped by command; example/<NN>_* directories share the numbers):
  -- library core --
   1. wigner_D_real regression (pure numpy)
  -- crystod (main command) --
   2. crystod (SALC)            crystal-orbital irreps
   3. crystod --atomic-orbital  hybridization analysis
   4. crystod --star-of-k       star of k
   5. crystod --visualize       SALC coefficients + 3D HTML viewer
   6. crystod main command      extras (aliases/errors/removed flags)
  -- crystod-group --
   7. crystod-group --product   point-group direct products
   8. crystod-group --decompose reducible-representation decomposition
   9. crystod-group --ligand-field orbital splitting in a point-group field
  10. crystod-group --basis     polynomial basis classification
  11. crystod-group --generate-basis automatic polynomial bases
  12. crystod-group --coset     coset decompositions
  13. crystod-group             seven-mode extras
  -- crystod-bz --
  14. crystod-bz                Brillouin-zone plot (seekpath auto k-path)
  15. crystod-bz --trans-mat    unit-cell + supercell Brillouin-zone plot
  16. crystod-bz                sectioned-command extras (--show-kpoint/identity/errors/removed flags)
  -- crystod-phonon --
  17. crystod-phonon --irreps   phonon irrep labeling (phonopy data)
  18. crystod-phonon --fatband  element-projected phonon fatbands (phonopy data)
  19. crystod-phonon --lt       longitudinal/transverse-resolved phonon band
  20. crystod-phonon --vector   phonon eigenvector VESTA export (phonopy data)
  21. crystod-phonon --modulation modulated structures (known space groups)
  22. crystod-phonon --vibration symmetry-only vibration bases
  23. crystod-phonon            six-mode extras
  -- crystod-mag --
  24. crystod-mag               symmetry-adapted spin bases (cluster multipoles / SAMM)
  25. crystod-mag               --format qe / --conventional extras
  -- crystod-md --
  26. crystod-md --adp          ADPs from an MD XDATCAR trajectory
  27. crystod-md                --adp / --summary extras
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
POSCAR_ScF3 = os.path.join(ROOT, "example", "test_POSCARs", "221_PPOSCAR_ScF3")
POSCAR_SrTiO3 = os.path.join(ROOT, "example", "test_POSCARs", "221_PPOSCAR_SrTiO3")
MODULATION_DIR = os.path.join(ROOT, "example", "21_modulation", "ScF3_Pm-3m")
PHONON_IRREP_DIR = os.path.join(ROOT, "example", "17_phonon_irrep", "SrTiO3_Pm-3m")
PHONON_VECTOR_DIR = os.path.join(ROOT, "example", "20_phonon_vector", "Si_Fd-3m")
XDATCAR_ADP_DIR = os.path.join(ROOT, "example", "26_xdatcar2adp", "ScF3_Pm-3m_NpT_300K")
PHONON_FATBAND_DIR = os.path.join(ROOT, "example", "18_phonon_fatband", "ScF3_Pm-3m")
PHONON_LT_DIR = os.path.join(ROOT, "example", "19_phonon_lt", "ScF3_Pm-3m")

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


def run_module(module: str, args: list[str], cwd: str | None = None) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", module] + args,
            capture_output=True,
            text=True,
            cwd=cwd or ROOT,
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


# ---------------------------------------------------------------- 4. crystod --star-of-k
def test_04_star_of_k() -> None:
    print("\n[4] crystod --star-of-k")
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
           code != 0 and "POSCAR file not found" in out and "Traceback" not in out, out)


# ---------------------------------------------------------------- 5. crystod --visualize
def test_05_visualize_basis() -> None:
    print("\n[5] crystod --visualize (SALC viewer)")
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


# ---------------------------------------------------------------- 6. crystod main command extras
def test_06_main_command() -> None:
    print("\n[6] crystod main command (SALC without mode flag)")

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
    report("--star-of-k accepts a high-symmetry label (was broken pre-v0.3.0)",
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

        code, out = run_cli(["-c", si_poscar, "--element", "Si", "--orbital", "p",
                             "--kpoint", "0", "0", "0", "--conventional"], cwd=tmp)
        report("--conventional without --visualize rejected cleanly",
               code != 0 and "--visualize" in out and "Traceback" not in out, out)

    code, out = run_cli(["--version"])
    report("--version prints the package version", code == 0 and "CrystOD 0.3" in out, out)

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
    code, out = run_cli(["-c", POSCAR_SrTiO3, "--element", "Ti", "--orbital", "d",
                         "--kpoint", "GM"])
    report("k-point label rejected in SALC mode (numeric only)",
           code != 0 and "three coordinates" in out and "Traceback" not in out, out)
    code, out = run_cli(["-c", POSCAR_SrTiO3, "--element", "Ti", "--orbital", "d",
                         "--atomic-orbital", "O_p"])
    report("--element combined with --atomic-orbital rejected cleanly",
           code != 0 and "--atomic-orbital alone" in out and "Traceback" not in out, out)

    # removed flat forms give replacement guidance
    for flag, replacement in (("--salc", "crystod -c"), ("--visualize-basis", "crystod --visualize")):
        code, out = run_cli([flag])
        report(f"removed {flag} flag points to the new main command",
               code != 0 and "removed in v0.3.0" in out and replacement in out, out)


# ---------------------------------------------------------------- 7. crystod-group --product
def test_07_direct_product() -> None:
    print("\n[7] crystod-group --product")
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

    code, out = run_group(["--table", "--point-group", "3m"])
    report("character table of 3m exit 0", code == 0, out)
    report("table lists A1 and E", "A1" in out and "E" in out, out)


# ---------------------------------------------------------------- 8. crystod-group --decompose
def test_08_decompose_irrep() -> None:
    print("\n[8] crystod-group --decompose")
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


# ---------------------------------------------------------------- 9. crystod-group --ligand-field
def test_09_ligand_field_split() -> None:
    print("\n[9] crystod-group --ligand-field")
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


# ---------------------------------------------------------------- 10. crystod-group --basis
def test_10_basis_function() -> None:
    print("\n[10] crystod-group --basis")
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

    code, out = run_group(["--basis", "x*Ry - y*Rx", "--point-group", "m-3m"])
    report("toroidal x*Ry - y*Rx -> T1u", code == 0 and "T1u" in out, out)

    code, out = run_group(["--basis", "Rx", "Ry", "Rz", "--space-group", "Pm-3m"])
    report("no --kpoint: all special k points analyzed",
           code == 0 and "analyzing all special k points" in out
           and out.count("k-point (primitive)") >= 4, out)
    report("axial decompositions at GM and R are gerade",
           "GM4+" in out and "R4+" in out, out)


# ---------------------------------------------------------------- 11. crystod-group --generate-basis
def test_11_generate_basis_function() -> None:
    print("\n[11] crystod-group --generate-basis")
    code, out = run_group(["--generate-basis", "--point-group", "m-3m"])
    report("point-group mode exit 0", code == 0, out)
    report("all three orders printed",
           all(key in out for key in ("1st order", "2nd order", "3rd order")), out)
    report("1st order -> T1u; 3rd order -> A2u",
           "T1u" in out and "A2u" in out, out)

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


# ---------------------------------------------------------------- 12. crystod-group --coset
def test_12_show_coset() -> None:
    print("\n[12] crystod-group --coset")
    code, out = run_group(["--coset", "--point-group", "m-3m", "--subgroup", "4/mmm"])
    report("point-group mode exit 0", code == 0, out)
    report("index [G:H] = 3", "index [G:H] = 3" in out, out)
    report("three cosets listed", out.count("coset ") >= 3, out)

    code, out = run_group(["--coset", "--space-group", "Pm-3m", "--kpoint", "0.5", "0.5", "0"])
    report("space-group mode exit 0", code == 0, out)
    report("index [G:G_k] = |star of k| = 3", "= 3" in out and "G_k" in out, out)


# ---------------------------------------------------------------- 13. crystod-group extras
def test_13_group_command() -> None:
    print("\n[13] crystod-group (sectioned command: 7 group-theory modes)")

    def run_group(args: list[str], cwd: str | None = None) -> tuple[int, str]:
        return run_module("crystod.cli.group", args, cwd)

    code, out = run_group(["--product", "T2g", "T2g", "--pg", "m-3m"])
    report("--product T2g T2g exit 0", code == 0, out)
    report("T2g x T2g = A1g + Eg + T1g + T2g",
           all(f"({name})" in out for name in ("A1g", "Eg", "T1g", "T2g")), out)

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

    # error handling
    code, out = run_group(["--product", "T2g", "T2g"])
    report("--product without --pg rejected cleanly",
           code != 0 and "requires --pg" in out and "Traceback" not in out, out)
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
               code != 0 and "removed in v0.3.0" in out and "crystod-group" in out, out)


# ---------------------------------------------------------------- 14. crystod-bz
def test_14_bz() -> None:
    print("\n[14] crystod-bz (Brillouin-zone plot)")
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


# ---------------------------------------------------------------- 15. crystod-bz --trans-mat
def test_15_bz_supercell() -> None:
    print("\n[15] crystod-bz --trans-mat (ScF3, Pm-3m -> transformed lattice)")
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


# ---------------------------------------------------------------- 16. crystod-bz extras
def test_16_bz_command() -> None:
    print("\n[16] crystod-bz (sectioned command: unit-cell / supercell BZ)")

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
                   code != 0 and "removed in v0.3.0" in out and "crystod-bz" in out, out)

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


# ---------------------------------------------------------------- 17. crystod-phonon --irreps
def test_17_phonon_irrep() -> None:
    print("\n[17] crystod-phonon --irreps (SrTiO3, 4x4x4 FORCE_SETS)")
    if not os.path.isdir(PHONON_IRREP_DIR):
        report("example data found", False, PHONON_IRREP_DIR)
        return
    with tempfile.TemporaryDirectory() as tmp:
        # copy inputs so phonon_irreps.yaml in the example folder is not overwritten
        # (--readfc / FORCE_CONSTANTS input is covered by the Si runs in
        # sections 20 and 23; the SrTiO3 example ships FORCE_SETS only)
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


# ---------------------------------------------------------------- 18. crystod-phonon --fatband
def test_18_phonon_fatband() -> None:
    print("\n[18] crystod-phonon --fatband (ScF3, 4x4x4 FORCE_SETS)")
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


# ---------------------------------------------------------------- 19. crystod-phonon --lt
def test_19_phonon_lt() -> None:
    print("\n[19] crystod-phonon --lt (ScF3, 4x4x4 FORCE_SETS)")
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

    phonon = phonopy_load(
        supercell_matrix=[4.0, 4.0, 4.0],
        primitive_matrix="auto",
        unitcell_filename=os.path.join(PHONON_LT_DIR, "221_PPOSCAR_ScF3"),
        force_sets_filename=os.path.join(PHONON_LT_DIR, "FORCE_SETS"),
        is_nac=False,
    )
    q = [0.1, 0.0, 0.0]
    phonon.run_qpoints([q], with_eigenvectors=True)
    eigvecs = phonon.get_qpoints_dict()["eigenvectors"][0][np.newaxis]
    rec = np.linalg.inv(np.array(phonon.primitive.cell)).T
    ratio = get_longitudinal_ratio(np.array([q]), eigvecs, rec)[0]
    freqs = phonon.get_qpoints_dict()["frequencies"][0]
    acoustic = np.argsort(freqs)[:3]
    report("acoustic set near GM splits into 2 T + 1 L along [100]",
           sorted(np.round(ratio[acoustic], 2))[:2] == [0.0, 0.0]
           and round(max(ratio[acoustic]), 2) > 0.9,
           str(ratio[acoustic]))


# ---------------------------------------------------------------- 20. crystod-phonon --vector
def test_20_phonon_vector() -> None:
    print("\n[20] crystod-phonon --vector (Si, 4x4x4 FC)")
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
            axis_pure = bool(arrows) and all(
                abs(float(a)) < 1e-5 and abs(float(b)) < 1e-5 and abs(abs(float(c)) - 1.5) < 1e-4
                for a, b, c in arrows
            )
            report("GM mode 4 arrows purely along c in conventional cell", axis_pure)

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


# ---------------------------------------------------------------- 21. crystod-phonon --modulation
def test_21_modulation() -> None:
    print("\n[21] crystod-phonon --modulation (known space groups from example/21_modulation README)")
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


# ---------------------------------------------------------------- 22. crystod-phonon --vibration
def test_22_vibration() -> None:
    print("\n[22] crystod-phonon --vibration")
    code, out = run_phonon(["--vibration", "-c", POSCAR_ScF3, "--qpoint", "R"])
    report("ScF3 q = R exit 0", code == 0, out)
    report("irrep-grouped mode spaces listed", "Mode Space" in out, out)
    report("mode spaces numbered from 1", "Mode Space  1:" in out and "Mode Space  0:" not in out, out)
    report("high-symmetry q-point list shown", "Available high-symmetry q-points" in out, out)

    # (0, 0.5, 0.5) is an M arm; irreptables tabulates only (0.5, 0.5, 0)
    code, out = run_phonon(["--vibration", "-c", POSCAR_ScF3, "--qpoint", "0", "0.5", "0.5"])
    report("non-representative M arm labeled via star mapping",
           code == 0 and "M5+(2)" in out and "irrep_" not in out, out)

    with tempfile.TemporaryDirectory() as tmp:
        out_poscar = os.path.join(tmp, "POSCAR_vibration")
        code, out = run_phonon(
            ["--vibration", "-c", POSCAR_ScF3, "--qpoint", "R",
             "--mode-index", "1", "--component-index", "1", "--output", out_poscar]
        )
        report("mode export exit 0", code == 0, out)
        report("commensurate supercell reported", "supercell size" in out, out)
        report("displaced POSCAR written", os.path.isfile(out_poscar))


# ---------------------------------------------------------------- 23. crystod-phonon extras
def test_23_phonon_command() -> None:
    print("\n[23] crystod-phonon (sectioned command: 6 phonon modes)")

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

    # removed flat flags give replacement guidance
    for flag in ("--vibration", "--phonon-irrep", "--phonon-fatband", "--phonon-lt",
                 "--phonon-vector", "--modulation"):
        code, out = run_cli([flag])
        report(f"removed {flag} flag points to crystod-phonon",
               code != 0 and "removed in v0.3.0" in out and "crystod-phonon" in out, out)


# ---------------------------------------------------------------- 24. crystod-mag
def test_24_spin_basis() -> None:
    print("\n[24] crystod-mag (AlNi3, Ni 3c cluster, Mn3Ir-type)")
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
    poscar_327 = os.path.join(ROOT, "example", "24_spin_basis", "La3Ni2O7_I4mmm", "139_PPOSCAR_La3Ni2O7")
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
    poscar_hex = os.path.join(ROOT, "example", "24_spin_basis", "LuFeO3_P63cm", "185_PPOSCAR_LuFeO3")
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


# ---------------------------------------------------------------- 25. crystod-mag extras
def test_25_mag_command() -> None:
    print("\n[25] crystod-mag (sectioned command: symmetry-adapted spin bases)")
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

        ceo2 = os.path.join(ROOT, "example", "05_visualized_basis", "CeO2_Fm-3m", "225_PPOSCAR_CeO2")
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
               code != 0 and "removed in v0.3.0" in out and "crystod-mag" in out, out)


# ---------------------------------------------------------------- 26. crystod-md --adp
def test_26_xdatcar2adp() -> None:
    print("\n[26] crystod-md --adp (ScF3 NpT 300K, truncated trajectory)")
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


# ---------------------------------------------------------------- 27. crystod-md extras
def test_27_md_command() -> None:
    print("\n[27] crystod-md (sectioned command: MD trajectory -> ADPs / summary)")

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
        # truncated trajectory (293 frames), as in section 26
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
               code != 0 and "removed in v0.3.0" in out and "crystod-md" in out, out)


SECTIONS = {
    1: test_01_wigner_d,
    2: test_02_salc,
    3: test_03_hybridization,
    4: test_04_star_of_k,
    5: test_05_visualize_basis,
    6: test_06_main_command,
    7: test_07_direct_product,
    8: test_08_decompose_irrep,
    9: test_09_ligand_field_split,
    10: test_10_basis_function,
    11: test_11_generate_basis_function,
    12: test_12_show_coset,
    13: test_13_group_command,
    14: test_14_bz,
    15: test_15_bz_supercell,
    16: test_16_bz_command,
    17: test_17_phonon_irrep,
    18: test_18_phonon_fatband,
    19: test_19_phonon_lt,
    20: test_20_phonon_vector,
    21: test_21_modulation,
    22: test_22_vibration,
    23: test_23_phonon_command,
    24: test_24_spin_basis,
    25: test_25_mag_command,
    26: test_26_xdatcar2adp,
    27: test_27_md_command,
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
