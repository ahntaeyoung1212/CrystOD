#!/usr/bin/env python
"""
CrystOD full test suite (run inside the `crystod` conda env).

Usage:
    conda activate crystod
    cd ~/CrystOD-main
    python testsuite.py            # run everything
    python testsuite.py 8 12       # run only sections 8 and 12

Sections:
   1. wigner_D_real regression (pure numpy)
   2. --salc                    crystal-orbital irreps
   3. --salc --atomic-orbital   hybridization analysis
   4. --basis-function          polynomial basis classification
   5. --direct-product          point-group direct products
   6. --phonon-irrep            phonon irrep labeling (phonopy data)
   7. --vibration               symmetry-only vibration bases
   8. --modulation              modulated structures (known space groups)
   9. --star-of-k               star of k
  10. --show-coset              coset decompositions
  11. --generate-basis-function automatic polynomial bases
  12. --visualize-basis         SALC coefficients + 3D HTML
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
MODULATION_DIR = os.path.join(ROOT, "example", "modulation", "ScF3_Pm-3m")
PHONON_IRREP_DIR = os.path.join(ROOT, "example", "phonon_irrep", "SrTiO3_Pm-3m")

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


def run_cli(args: list[str], cwd: str | None = None) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "crystod"] + args,
            capture_output=True,
            text=True,
            cwd=cwd or ROOT,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return -1, f"TIMEOUT after {TIMEOUT_SECONDS} s"
    return proc.returncode, proc.stdout + proc.stderr


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


# ---------------------------------------------------------------- 2. --salc
def test_02_salc() -> None:
    print("\n[2] --salc (crystal-orbital irreps)")
    code, out = run_cli(
        ["--salc", "--poscar", POSCAR_SrTiO3, "--element", "Ti", "--orbital", "d",
         "--kpoint", "0", "0", "0"]
    )
    report("SrTiO3 Ti_d at GM exit 0", code == 0, out)
    report("Ti_d at GM: GM3+ (eg) and GM5+ (t2g)",
           "GM3+" in out and "GM5+" in out, out)

    code, out = run_cli(
        ["--salc", "--poscar", POSCAR_ScF3, "--element", "F", "--orbital", "p",
         "--kpoint", "0", "0", "0"]
    )
    report("ScF3 F_p at GM exit 0", code == 0, out)
    report("F_p at GM: 2.0 [GM4-(3)] + 1.0 [GM5-(3)]",
           "2.0 [GM4-(3)]" in out and "1.0 [GM5-(3)]" in out, out)

    code, out = run_cli(
        ["--salc", "--poscar", POSCAR_SrTiO3, "--element", "Ti", "--orbital", "d"]
    )
    report("all special k points mode exit 0", code == 0, out)
    report("all special k points mode lists several k points",
           out.count("k point (primitive)") >= 3, out)


# ---------------------------------------------------------------- 3. hybridization
def test_03_hybridization() -> None:
    print("\n[3] --salc --atomic-orbital (hybridization)")
    code, out = run_cli(
        ["--salc", "--poscar", POSCAR_SrTiO3, "--atomic-orbital", "Ti_d", "O_p",
         "--kpoint", "0", "0", "0"]
    )
    report("Ti_d O_p at GM exit 0", code == 0, out)
    report("result section present", "* Result *" in out, out)
    report("GM irreps listed", "GM" in out.split("* Result *")[-1], out)


# ---------------------------------------------------------------- 4. basis-function
def test_04_basis_function() -> None:
    print("\n[4] --basis-function")
    code, out = run_cli(["--basis-function", "x", "y", "z", "--point-group", "m-3m"])
    report("x y z in m-3m exit 0", code == 0, out)
    report("x y z in m-3m -> T1u", "T1u" in out, out)

    code, out = run_cli(
        ["--basis-function", "x", "y", "z", "--space-group", "Pm-3m",
         "--kpoint", "0", "0", "0"]
    )
    report("x y z at GM in Pm-3m exit 0", code == 0, out)
    report("x y z at GM -> GM4-", "GM4-" in out, out)

    code, out = run_cli(
        ["--basis-function", "x^2-y^2", "2z^2-x^2-y^2", "xy", "yz", "zx",
         "--space-group", "Pm-3m", "--kpoint", "0", "0", "0"]
    )
    report("d-type set at GM exit 0", code == 0, out)
    report("d-type set -> GM3+ and GM5+", "GM3+" in out and "GM5+" in out, out)
    report("no numerical-noise blowup", re.search(r"\d{15,}", out) is None, out)


# ---------------------------------------------------------------- 5. direct-product
def test_05_direct_product() -> None:
    print("\n[5] --direct-product")
    code, out = run_cli(
        ["--direct-product", "--point-group", "m-3m", "--irreps", "T2g", "T2g"]
    )
    report("T2g x T2g in m-3m exit 0", code == 0, out)
    report("T2g x T2g = A1g + Eg + T1g + T2g",
           all(f"({name})" in out for name in ("A1g", "Eg", "T1g", "T2g")), out)

    code, out = run_cli(
        ["--direct-product", "--point-group", "m-3m", "--irreps", "T2g", "T2g", "T1u"]
    )
    report("T2g x T2g x T1u exit 0", code == 0, out)
    report("triple product contains A2u (Raman/IR selection logic)", "(A2u)" in out, out)

    code, out = run_cli(["--direct-product", "--point-group", "3m", "--show-irrep-table"])
    report("character table of 3m exit 0", code == 0, out)
    report("table lists A1 and E", "A1" in out and "E" in out, out)


# ---------------------------------------------------------------- 6. phonon-irrep
def test_06_phonon_irrep() -> None:
    print("\n[6] --phonon-irrep (SrTiO3, 4x4x4 FC)")
    if not os.path.isdir(PHONON_IRREP_DIR):
        report("example data found", False, PHONON_IRREP_DIR)
        return
    with tempfile.TemporaryDirectory() as tmp:
        # copy inputs so phonon_irreps.yaml in the example folder is not overwritten
        for name in ("221_PPOSCAR_SrTiO3", "FORCE_CONSTANTS"):
            shutil.copy(os.path.join(PHONON_IRREP_DIR, name), tmp)
        code, out = run_cli(
            ["--phonon-irrep", "--dim", "4 4 4", "--poscar", "221_PPOSCAR_SrTiO3",
             "--readfc"],
            cwd=tmp,
        )
        report("exit code 0", code == 0, out)
        yaml_path = os.path.join(tmp, "phonon_irreps.yaml")
        report("phonon_irreps.yaml written", os.path.isfile(yaml_path))
        if os.path.isfile(yaml_path):
            text = open(yaml_path).read()
            report("yaml contains GM point irreps", "GM" in text, text[:500])
            report("yaml contains R point irreps", "R" in text, text[:500])


# ---------------------------------------------------------------- 7. vibration
def test_07_vibration() -> None:
    print("\n[7] --vibration")
    code, out = run_cli(["--vibration", "--poscar", POSCAR_ScF3, "--qpoint", "R"])
    report("ScF3 q = R exit 0", code == 0, out)
    report("irrep-grouped mode spaces listed", "Mode Space" in out, out)
    report("high-symmetry q-point list shown", "Available high-symmetry q-points" in out, out)

    with tempfile.TemporaryDirectory() as tmp:
        out_poscar = os.path.join(tmp, "POSCAR_vibration")
        code, out = run_cli(
            ["--vibration", "--poscar", POSCAR_ScF3, "--qpoint", "R",
             "--mode-index", "0", "--component-index", "0", "--output", out_poscar]
        )
        report("mode export exit 0", code == 0, out)
        report("commensurate supercell reported", "supercell size" in out, out)
        report("displaced POSCAR written", os.path.isfile(out_poscar))


# ---------------------------------------------------------------- 8. modulation
def test_08_modulation() -> None:
    print("\n[8] --modulation (known space groups from example/modulation README)")
    yaml_path = os.path.join(MODULATION_DIR, "phonopy_params.yaml")
    if not os.path.isfile(yaml_path):
        report("phonopy_params.yaml found", False, yaml_path)
        return

    with tempfile.TemporaryDirectory() as tmp:
        out_poscar = os.path.join(tmp, "POSCAR_R-3c")
        code, out = run_cli(
            ["--modulation", "--yaml", yaml_path, "--qpoint", "0.5", "0.5", "0.5",
             "--mode", "0", "1", "2", "--amplitude", "0.3", "--output", out_poscar]
        )
        report("R4+(a,a,a) exit 0", code == 0, out)
        report("R4+(a,a,a) -> R-3c", "R-3c" in out, out)
        report("star of q displayed", "Star of q" in out, out)
        report("POSCAR written", os.path.isfile(out_poscar))

        out_poscar = os.path.join(tmp, "POSCAR_I4mcm")
        code, out = run_cli(
            ["--modulation", "--yaml", yaml_path, "--qpoint", "0.5", "0.5", "0.5",
             "--mode", "0", "--amplitude", "0.3", "--output", out_poscar]
        )
        report("R4+(0,0,a) -> I4/mcm", code == 0 and "I4/mcm" in out.replace("I4mcm", "I4/mcm"), out)

        out_poscar = os.path.join(tmp, "POSCAR_multi_q")
        code, out = run_cli(
            ["--modulation", "--yaml", yaml_path,
             "--qpoint1", "0", "0.5", "0.5", "--mode1", "0", "--amplitude1", "0.3",
             "--qpoint2", "0.5", "0", "0.5", "--mode2", "0", "--amplitude2", "0.3",
             "--output", out_poscar]
        )
        report("multi-q M3+(a;a;0) exit 0", code == 0, out)
        report("multi-q M3+(a;a;0) -> I4/mmm",
               "I4/mmm" in out.replace("I4mmm", "I4/mmm"), out)
        report("star of q displayed for each q", out.count("Star of q") >= 2, out)


# ---------------------------------------------------------------- 9. star-of-k
def test_09_star_of_k() -> None:
    print("\n[9] --star-of-k")
    code, out = run_cli(["--star-of-k", "--poscar", POSCAR_ScF3, "--kpoint", "0.5", "0.5", "0"])
    report("M point exit 0", code == 0, out)
    report("M point: |star of k| = 3", "|star of k| = 3" in out, out)

    code, out = run_cli(["--star-of-k", "--poscar", POSCAR_ScF3, "--kpoint", "0.5", "0.5", "0.5"])
    report("R point: |star of k| = 1", code == 0 and "|star of k| = 1" in out, out)

    code, out = run_cli(["--star-of-k", "--poscar", "NO_SUCH_POSCAR", "--kpoint", "0", "0", "0"])
    report("missing POSCAR gives clear error (no traceback)",
           code != 0 and "POSCAR file not found" in out and "Traceback" not in out, out)


# ---------------------------------------------------------------- 10. show-coset
def test_10_show_coset() -> None:
    print("\n[10] --show-coset")
    code, out = run_cli(["--show-coset", "--point-group", "m-3m", "--subgroup", "4/mmm"])
    report("point-group mode exit 0", code == 0, out)
    report("index [G:H] = 3", "index [G:H] = 3" in out, out)
    report("three cosets listed", out.count("coset ") >= 3, out)

    code, out = run_cli(["--show-coset", "--space-group", "Pm-3m", "--kpoint", "0.5", "0.5", "0"])
    report("space-group mode exit 0", code == 0, out)
    report("index [G:G_k] = |star of k| = 3", "= 3" in out and "G_k" in out, out)


# ---------------------------------------------------------------- 11. generate-basis-function
def test_11_generate_basis_function() -> None:
    print("\n[11] --generate-basis-function")
    code, out = run_cli(["--generate-basis-function", "--point-group", "m-3m"])
    report("point-group mode exit 0", code == 0, out)
    report("all three orders printed",
           all(key in out for key in ("1st order", "2nd order", "3rd order")), out)
    report("1st order -> T1u; 3rd order -> A2u",
           "T1u" in out and "A2u" in out, out)

    code, out = run_cli(
        ["--generate-basis-function", "--space-group", "Pm-3m", "--kpoint", "0", "0", "0",
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


# ---------------------------------------------------------------- 12. visualize-basis
def test_12_visualize_basis() -> None:
    print("\n[12] --visualize-basis")
    code, out = run_cli(
        ["--visualize-basis", "--poscar", POSCAR_ScF3, "--element", "F", "--orbital", "p",
         "--kpoint", "0", "0", "0"]
    )
    report("F_p at GM exit 0", code == 0, out)
    report("decomposition matches --salc (2 GM4- + 1 GM5-)",
           "2.0 [GM4-(3)]" in out and "1.0 [GM5-(3)]" in out, out)
    report("SALC coefficients printed", "SALC basis functions" in out, out)

    code, out = run_cli(
        ["--visualize-basis", "--poscar", POSCAR_ScF3, "--element", "Sc", "--orbital", "d",
         "--kpoint", "0", "0", "0", "--real-coefficient"]
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
            ["--visualize-basis", "--poscar", POSCAR_ScF3, "--element", "F", "--orbital", "p",
             "--kpoint", "0", "0", "0", "--output", html]
        )
        report("HTML export exit 0", code == 0, out)
        exists = os.path.isfile(html)
        report("HTML file created", exists)
        if exists:
            text = open(html).read()
            report("HTML contains plotly + dropdown menu",
                   "plotly" in text and "updatemenus" in text)

        html_m = os.path.join(tmp, "salc_M.html")
        code, out = run_cli(
            ["--visualize-basis", "--poscar", POSCAR_ScF3, "--element", "F", "--orbital", "p",
             "--kpoint", "0.5", "0.5", "0", "--output", html_m]
        )
        report("k = M (supercell + Bloch phase) exit 0", code == 0, out)
        report("k = M HTML created", os.path.isfile(html_m))


SECTIONS = {
    1: test_01_wigner_d,
    2: test_02_salc,
    3: test_03_hybridization,
    4: test_04_basis_function,
    5: test_05_direct_product,
    6: test_06_phonon_irrep,
    7: test_07_vibration,
    8: test_08_modulation,
    9: test_09_star_of_k,
    10: test_10_show_coset,
    11: test_11_generate_basis_function,
    12: test_12_visualize_basis,
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
