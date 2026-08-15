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
   3. crystod --atomic-orbital  hybridization analysis, and the crystal-orbital
                                diagrams of --diagram (extended Hueckel) and
                                --diagram --pyscf (skipped without pyscf)
   4. crystod --star-of-k       star of k
   5. crystod --visualize       SALC coefficients + 3D HTML viewer
   6. crystod main command      extras (aliases/errors/removed flags)
  -- crystod-group --
   7. crystod-group --product   point-group and space-group irrep direct products
   8. crystod-group --decompose reducible-representation decomposition
   9. crystod-group --ligand-field orbital splitting in a point-group field
  10. crystod-group --basis     polynomial basis classification
  11. crystod-group --generate-basis automatic polynomial bases
  12. crystod-group --coset     coset decompositions
  13. crystod-group --supergroup  isotropy subgroups of space-group irreps
  14. crystod-group --multiplet   spin multiplicities of irrep-shell configurations
  15. crystod-group --poscar2cif / --cif2poscar  POSCAR <-> Bilbao-style CIF
  16. crystod-group --supergroup-cif  symmetry-mode (AMPLIMODES-style) analysis
  17. crystod-group             eleven-mode extras
  -- crystod-bz --
  18. crystod-bz                Brillouin-zone plot (seekpath auto k-path)
  19. crystod-bz --trans-mat    unit-cell + supercell Brillouin-zone plot
  20. crystod-bz                sectioned-command extras (--show-kpoint/identity/errors/removed flags)
  -- crystod-phonon --
  21. crystod-phonon --irreps   phonon irrep labeling (phonopy data)
  22. crystod-phonon --fatband  element-projected phonon fatbands (phonopy data)
  23. crystod-phonon --lt       longitudinal/transverse-resolved phonon band
  24. crystod-phonon --vector   phonon eigenvector VESTA export (phonopy data)
  25. crystod-phonon --modulation modulated structures (known space groups)
  26. crystod-phonon --vibration symmetry-only vibration bases
  27. crystod-phonon            seven-mode extras, incl. --subgroup (isotropy
                                subgroups of the imaginary modes)
  -- crystod-mag --
  28. crystod-mag               symmetry-adapted spin bases (cluster multipoles / SAMM)
  29. crystod-mag               --format qe / --conventional extras
  -- crystod-md --
  30. crystod-md --adp          ADPs from an MD XDATCAR trajectory
  31. crystod-md                --adp / --summary extras
  -- crystod-mol --
  32. crystod-mol               molecular point groups and molecular SALCs
  33. crystod-mol --diagram     MO diagram from symmetry + overlap (extended Hueckel)
  34. crystod-mol               --align / --show-matrix / --visualize / error extras
  -- Python API --
  35. crystod.salc/group/phonon/bz/mag/md/mol  library API (import hygiene,
                                backward compatibility, API-vs-CLI agreement)
"""

from __future__ import annotations

import json
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
POSCAR_NaCl = os.path.join(ROOT, "example", "test_POSCARs", "225_PPOSCAR_NaCl")
MODULATION_DIR = os.path.join(ROOT, "example", "25_modulation", "ScF3_Pm-3m")
PHONON_IRREP_DIR = os.path.join(ROOT, "example", "21_phonon_irrep", "SrTiO3_Pm-3m")
PHONON_VECTOR_DIR = os.path.join(ROOT, "example", "24_phonon_vector", "Si_Fd-3m")
XDATCAR_ADP_DIR = os.path.join(ROOT, "example", "30_xdatcar2adp", "ScF3_Pm-3m_NpT_300K")
PHONON_FATBAND_DIR = os.path.join(ROOT, "example", "22_phonon_fatband", "ScF3_Pm-3m")
PHONON_LT_DIR = os.path.join(ROOT, "example", "23_phonon_lt", "ScF3_Pm-3m")
XYZ_DIR = os.path.join(ROOT, "example", "test_XYZs")

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


def run_python(script: str, cwd: str | None = None) -> tuple[int, str]:
    """Run a Python snippet in a fresh interpreter (for the library API)."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
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
        report("R: near-dependent diffuse Bloch combination removed",
               "near-dependent diffuse Bloch combination(s) removed" in out
               and "overlap floor" in out, out)
        report("diagram HTML written with k-point variants",
               os.path.isfile(html_path), out)
        if os.path.isfile(html_path):
            with open(html_path) as handle:
                html = handle.read()
            report("HTML has the four k-point buttons and variants",
                   html.count('class="kbtn') == 4
                   and "const VARIANTS = [" in html
                   and "R (1/2,1/2,1/2)" in html, html_path)
            report("levels carry supercell wave-function sketches",
                   html.count('"orb": [[') > 50
                   and html.count('"geom":') == 4
                   and "jacobi3" in html, html_path)
            report("frontier-centered view window (HOMO/LUMO midpoint ±8 eV)",
                   re.search(r'"eMin": -2[01]\.\d+', html) is not None
                   and "Show all energy levels" in html, html_path)

        # multi-shell same-l sketch filter (Sc-p = 2p+3p+4p): the shells
        # must accumulate with their STO radial amplitudes at r0 = 2 bohr,
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
        code, out = run_cli(pyscf_argv + ["--output", pyscf_path])
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
                                          "--output", conv_path])
        conv_html = ""
        if code == 0 and os.path.isfile(conv_path):
            with open(conv_path) as handle:
                conv_html = handle.read()
        report("--diagram --pyscf --conventional exit 0 on the reused --chk",
               code == 0 and "[read from" in out
               and "conventional cell (F centring), 1 x 1 x 1" in conv_html
               and "drawn on the conventional cell" in conv_html
               and '"orb": [[' in conv_html, out)


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
           code != 0 and "No POSCAR named NO_SUCH_POSCAR!" in out
           and "Traceback" not in out, out)


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
           code == 0 and "M1 x P1 = PA2" in out and "1 x 2 = 2 -> 2 = 2" in out,
           out)

    # ---- line terms: ISO-IR (ISOTROPY) labels
    code, out = run_group(["--product", "N1", "P1", "--sg", "I4_132"])
    report("N1 x P1 (I4_132) lands on the DT line with ISO-IR labels",
           code == 0 and all(f"DT{i}" in out for i in (1, 2, 3, 4))
           and "[non-tabulated]" in out
           and "6 x 4 = 24 -> 6 + 6 + 6 + 6 = 24" in out, out)

    code, out = run_group(["--product", "P1", "X1", "--sg", "I4"])
    report("P1 x X1 (I4) +/-k line stars disambiguated (LD vs LDA)",
           code == 0 and "P1 x X1 = LD1 + LD2 + LDA1 + LDA2" in out
           and "2 x 2 = 4 -> 1 + 1 + 1 + 1 = 4" in out, out)

    code, out = run_group(["--product", "R4-", "Q9", "--sg", "Pm-3m"])
    report("unknown space-group irrep label rejected with available list",
           code != 0 and "not tabulated" in out and "Available irreps" in out
           and "Traceback" not in out, out)

    code, out = run_group(["--product", "T2g", "T2g", "--pg", "m-3m", "--sg", "Pm-3m"])
    report("--product with both --pg and --sg rejected cleanly",
           code != 0 and "exactly one" in out and "Traceback" not in out, out)


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

    # non-special k: shares the ISO-IR fallback of --basis
    code, out = run_group(
        ["--generate-basis", "--space-group", "Pm-3m",
         "--kpoint", "0", "0", "0.1", "--order", "1"]
    )
    report("--generate-basis on the DT line labeled via ISO-IR",
           code == 0 and "DT1(1)" in out and "DT5(2)" in out, out)


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


# ---------------------------------------------------------------- 17. crystod-group extras
def test_17_group_command() -> None:
    print("\n[17] crystod-group (sectioned command: 7 group-theory modes)")

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

    # --parent is --supergroup under the name of the value it takes
    code, out = run_group(["--parent", "Pm-3m", "--irrep", "GM4-",
                           "--order-parameter", "0", "0", "a"])
    code_alias, out_alias = run_group(["--supergroup", "Pm-3m", "--irrep", "GM4-",
                                       "--order-parameter", "0", "0", "a"])
    report("--parent alias accepted",
           code == 0 and "P4mm" in out, out)
    report("--parent and --supergroup give the same output",
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
               code != 0 and "removed in v0.3.0" in out and "crystod-group" in out, out)


# ---------------------------------------------------------------- 18. crystod-bz
def test_18_bz() -> None:
    print("\n[18] crystod-bz (Brillouin-zone plot)")
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


# ---------------------------------------------------------------- 19. crystod-bz --trans-mat
def test_19_bz_supercell() -> None:
    print("\n[19] crystod-bz --trans-mat (ScF3, Pm-3m -> transformed lattice)")
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


# ---------------------------------------------------------------- 20. crystod-bz extras
def test_20_bz_command() -> None:
    print("\n[20] crystod-bz (sectioned command: unit-cell / supercell BZ)")

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


# ---------------------------------------------------------------- 21. crystod-phonon --irreps
def test_21_phonon_irrep() -> None:
    print("\n[21] crystod-phonon --irreps (SrTiO3, 4x4x4 FORCE_SETS)")
    if not os.path.isdir(PHONON_IRREP_DIR):
        report("example data found", False, PHONON_IRREP_DIR)
        return
    with tempfile.TemporaryDirectory() as tmp:
        # copy inputs so phonon_irreps.yaml in the example folder is not overwritten
        # (--readfc / FORCE_CONSTANTS input is covered by the Si runs in
        # sections 24 and 27; the SrTiO3 example ships FORCE_SETS only)
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


# ---------------------------------------------------------------- 22. crystod-phonon --fatband
def test_22_phonon_fatband() -> None:
    print("\n[22] crystod-phonon --fatband (ScF3, 4x4x4 FORCE_SETS)")
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


# ---------------------------------------------------------------- 23. crystod-phonon --lt
def test_23_phonon_lt() -> None:
    print("\n[23] crystod-phonon --lt (ScF3, 4x4x4 FORCE_SETS)")
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


# ---------------------------------------------------------------- 24. crystod-phonon --vector
def test_24_phonon_vector() -> None:
    print("\n[24] crystod-phonon --vector (Si, 4x4x4 FC)")
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


# ---------------------------------------------------------------- 25. crystod-phonon --modulation
def test_25_modulation() -> None:
    print("\n[25] crystod-phonon --modulation (known space groups from example/25_modulation README)")
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


# ---------------------------------------------------------------- 26. crystod-phonon --vibration
def test_26_vibration() -> None:
    print("\n[26] crystod-phonon --vibration")
    code, out = run_phonon(["--vibration", "-c", POSCAR_ScF3, "--qpoint", "R"])
    report("ScF3 q = R exit 0", code == 0, out)
    report("irrep-grouped mode spaces listed", "Mode Space" in out, out)
    report("mode spaces numbered from 1", "Mode Space  1:" in out and "Mode Space  0:" not in out, out)
    report("high-symmetry q-point list shown", "Available high-symmetry q-points" in out, out)

    # (0, 0.5, 0.5) is an M arm; irreptables tabulates only (0.5, 0.5, 0)
    code, out = run_phonon(["--vibration", "-c", POSCAR_ScF3, "--qpoint", "0", "0.5", "0.5"])
    report("non-representative M arm labeled via star mapping",
           code == 0 and "M5+(2)" in out and "irrep_" not in out, out)

    # non-special q (T line): labels fall back to the ISO-IR (ISOTROPY) tables
    code, out = run_phonon(
        ["--vibration", "-c", POSCAR_SrTiO3, "--qpoint", "0.5", "0.5", "0.4"]
    )
    report("T-line q named via ISO-IR",
           code == 0 and "Selected q-point: T =" in out, out)
    report("T-line mode spaces labeled via ISO-IR",
           "T5(2)" in out and "irrep_" not in out, out)

    with tempfile.TemporaryDirectory() as tmp:
        out_poscar = os.path.join(tmp, "POSCAR_vibration")
        code, out = run_phonon(
            ["--vibration", "-c", POSCAR_ScF3, "--qpoint", "R",
             "--mode-index", "1", "--component-index", "1", "--output", out_poscar]
        )
        report("mode export exit 0", code == 0, out)
        report("commensurate supercell reported", "supercell size" in out, out)
        report("displaced POSCAR written", os.path.isfile(out_poscar))


# ---------------------------------------------------------------- 27. crystod-phonon extras
def test_27_phonon_command() -> None:
    print("\n[27] crystod-phonon (sectioned command: 6 phonon modes)")

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
                   code != 0 and "does not match" in out and "Traceback" not in out, out)

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
                   "crystod-phonon --modulation --qpoint 0.5 0.5 0.5 --mode 1 --amplitude 0.3" in out,
                   out)
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
            try:
                import spglib as _spglib
                from ase.io import read as _ase_read

                from crystod.modulation import classify_distorted_structure

                parent = _ase_read(os.path.join(tmp, "221_PPOSCAR_SrTiO3"), format="vasp")
                parent_cell = (parent.cell[:], parent.get_scaled_positions(),
                               parent.get_atomic_numbers())
                mismatches = []
                for name, key in expected.items():
                    path = os.path.join(tmp, name)
                    if not os.path.isfile(path):
                        mismatches.append(f"{name}: not written")
                        continue
                    atoms = _ase_read(path, format="vasp")
                    number, _symbol, size, index = classify_distorted_structure(
                        atoms, parent_cell, symprec=1e-5
                    )
                    if (number, size, index) != key:
                        mismatches.append(
                            f"{name}: measured {(number, size, index)}, expected {key}"
                        )
                report("every written structure has the space group it claims",
                       not mismatches, "\n".join(mismatches))
            except ImportError as exc:  # pragma: no cover - spglib/ase always present
                report("structure verification imports available", False, str(exc))

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
               code != 0 and "removed in v0.3.0" in out and "crystod-phonon" in out, out)


# ---------------------------------------------------------------- 28. crystod-mag
def test_28_spin_basis() -> None:
    print("\n[28] crystod-mag (AlNi3, Ni 3c cluster, Mn3Ir-type)")
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
    poscar_327 = os.path.join(ROOT, "example", "28_spin_basis", "La3Ni2O7_I4mmm", "139_PPOSCAR_La3Ni2O7")
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
    poscar_hex = os.path.join(ROOT, "example", "28_spin_basis", "LuFeO3_P63cm", "185_PPOSCAR_LuFeO3")
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


# ---------------------------------------------------------------- 29. crystod-mag extras
def test_29_mag_command() -> None:
    print("\n[29] crystod-mag (sectioned command: symmetry-adapted spin bases)")
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


# ---------------------------------------------------------------- 30. crystod-md --adp
def test_30_xdatcar2adp() -> None:
    print("\n[30] crystod-md --adp (ScF3 NpT 300K, truncated trajectory)")
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


# ---------------------------------------------------------------- 31. crystod-md extras
def test_31_md_command() -> None:
    print("\n[31] crystod-md (sectioned command: MD trajectory -> ADPs / summary)")

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
        # truncated trajectory (293 frames), as in section 30
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


# ---------------------------------------------------------------- 32. crystod-mol
def test_32_mol() -> None:
    print("\n[32] crystod-mol (molecular point groups and molecular SALCs)")

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


# ------------------------------------------------- 33. crystod-mol --diagram
def test_33_molod() -> None:
    print("\n[33] crystod-mol --diagram (MO diagram from symmetry + overlap)")

    def run_mol(args: list[str], cwd: str | None = None) -> tuple[int, str]:
        return run_module("crystod.cli.mol", args, cwd)

    molod_dir = os.path.join(ROOT, "example", "33_molod")
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


# ---------------------------------------------------------------- 34. crystod-mol extras
def test_34_mol_command() -> None:
    print("\n[34] crystod-mol (extras: --align / --show-matrix / --visualize / errors)")

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


# ------------------------------------------- 13. crystod-group --supergroup
def test_13_isotropy() -> None:
    print("\n[13] crystod-group --supergroup (isotropy subgroups)")

    import spglib

    if tuple(int(x) for x in spglib.__version__.split(".")[:2]) < (2, 4):
        print(f"  [SKIP] spglib {spglib.__version__} < 2.4: subgroup identification "
              "is unreliable in old spglib; run this section in the crystod env.")
        return

    # single order-parameter directions (the ISOSUBGROUP reference cases)
    code, out = run_group(["--supergroup", "Pm-3m", "--irrep", "GM4-",
                           "--order-parameter", "0", "0", "a"])
    report("GM4- (0,0,a) -> P4mm", code == 0 and "P4mm (No. 99)" in out, out)
    report("index 6 and conventional basis printed",
           "index 6" in out and "conventional basis" in out, out)

    code, out = run_group(["--supergroup", "Pm-3m", "--irrep", "GM4-",
                           "--order-parameter", "a", "a", "0"])
    report("GM4- (a,a,0) -> Amm2", code == 0 and "Amm2 (No. 38)" in out, out)

    code, out = run_group(["--supergroup", "Pm-3m", "--irrep", "GM4-",
                           "--order-parameter", "a", "a", "a"])
    report("GM4- (a,a,a) -> R3m", code == 0 and "R3m (No. 160)" in out, out)

    # full enumeration (validated against the ISOSUBGROUP table for Pm-3m GM)
    code, out = run_group(["--supergroup", "Pm-3m", "--irrep", "GM4-"])
    report("GM4- enumeration exit 0", code == 0, out)
    report("GM4- enumerates P4mm/R3m/Amm2/Pm/Cm/P1",
           all(name in out for name in
               ("99 P4mm", "160 R3m", "38 Amm2", "6 Pm", "8 Cm", "1 P1")), out)

    code, out = run_group(["--supergroup", "Pm-3m", "--irrep", "GM3+"])
    report("GM3+ -> P4/mmm + Pmmm (indices 3, 6)",
           code == 0 and "123 P4/mmm" in out and "47 Pmmm" in out, out)

    code, out = run_group(["--supergroup", "Pm-3m", "--irrep", "GM1+"])
    report("GM1+ (identity irrep) keeps the supergroup",
           code == 0 and "221 Pm-3m" in out, out)

    # zone-boundary irreps: cell enlargement (perovskite octahedral tilts)
    code, out = run_group(["--supergroup", "Pm-3m", "--irrep", "R4+"])
    report("R4+ tilt subgroups I4/mcm + R-3c + Imma (Howard-Stokes)",
           code == 0 and "140 I4/mcm" in out and "167 R-3c" in out
           and "74 Imma" in out, out)
    report("R4+ doubles the cell (size 2)",
           re.search(r"140 I4/mcm\s+2\s+6", out) is not None, out)

    code, out = run_group(["--supergroup", "Pm-3m", "--irrep", "M3+"])
    report("M3+ tilt subgroups P4/mbm + Im-3 + I4/mmm",
           code == 0 and "127 P4/mbm" in out and "204 Im-3" in out
           and "139 I4/mmm" in out, out)
    report("direction column carries the irrep label (arms ; separated)",
           re.search(r"M3\+\(0;0;a\)\s+127 P4/mbm", out) is not None, out)

    code, out = run_group(["--supergroup", "Pm-3m", "--irrep", "M3+",
                           "--order-parameter", "a", "a", "a"])
    report("M3+ (a,a,a) -> Im-3 with 2x2x2 cell (size 4)",
           code == 0 and "Im-3 (No. 204)" in out and "cell size 4" in out, out)

    # complex-type irrep at a non-symmorphic zone-boundary point: the
    # physically irreducible doubled form, paired with the conjugate irrep
    # (validated entry by entry against ISOSUBGROUP SG230 P tables)
    code, out = run_group(["--supergroup", "Ia-3d", "--irrep", "P2"])
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
    code, out = run_group(["--supergroup", "Ia-3d", "--irrep", "P3"])
    report("P3 realified (R32/R-3/R3 subgroups as in ISOSUBGROUP)",
           code == 0 and re.search(r"155 R32\s+4\s+32", out) is not None
           and re.search(r"148 R-3\s+4\s+32", out) is not None
           and re.search(r"146 R3\s+4\s+64", out) is not None, out)

    # +k/-k pairing (the -k star is not in the star of k): P/PA of I-42d,
    # tabulated in the conjugate gauge
    code, out = run_group(["--supergroup", "122", "--irrep", "P1"])
    report("I-42d P1 -> P1PA1 pair (I-4 subgroup as in ISOSUBGROUP)",
           code == 0 and "P1PA1: order parameter dimension 4" in out
           and re.search(r"82 I-4\s+4\s+8", out) is not None
           and re.search(r"1 P1\s+4\s+32", out) is not None, out)

    # wrapped-translation gauge at N of I4_132: conj+wrap selection of the
    # spgrep candidate (chiral subgroups P4_122 vs P4_322 distinguish N1/N3)
    code, out = run_group(["--supergroup", "214", "--irrep", "N1"])
    report("I4_132 N1 (wrapped-gauge N star) -> C222 + P4_122 + R32",
           code == 0 and re.search(r"21 C222\s+2\s+12", out) is not None
           and re.search(r"91 P4_122\s+4\s+12", out) is not None
           and re.search(r"155 R32\s+4\s+16", out) is not None, out)
    report("enantiomorphic-partner note printed (91 <-> 95)",
           "91 <-> 95 are enantiomorphic partner types" in out, out)

    # crystod (ISO-IR data files) vs ISOTROPY-software label-convention note
    code, out = run_group(["--supergroup", "Ia-3d", "--irrep", "N1"])
    report("ISO-IR-vs-ISOTROPY label note at N of Ia-3d",
           code == 0 and "crystod N1 = ISOTROPY N2" in out
           and "SUBGROUP/VALIDATION.md" in out, out)

    # mixed translation denominators (H of P3: k = (1/3,1/3,1/2)): the
    # translation grid must be the lcm (6), not the max (3)
    code, out = run_group(["--supergroup", "P3", "--irrep", "H1"])
    report("P3 H1 -> H1HA1 with 6x cell (lcm translation grid)",
           code == 0 and "H1HA1" in out
           and re.search(r"143 P3\s+6\s+6", out) is not None, out)

    # coupled irreps (I4/mmm X3- + X2+: n=2 Ruddlesden-Popper rotation+tilt;
    # hybrid-improper-ferroelectric ground state Cmc2_1 = A2_1am)
    code, out = run_group(["--supergroup", "I4/mmm", "--irrep", "X3-", "X2+"])
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

    code, out = run_group(["--supergroup", "I4/mmm", "--irrep", "X3-", "X2+",
                           "--order-parameter", "0", "a", "0", "c"])
    report("coupled explicit direction resolves to Cmc2_1",
           code == 0 and "X3-(0;a) X2+(0;c) -> Cmc2_1 (No. 36)" in out
           and "cell size 2, index 8" in out, out)

    code, out = run_group(["--supergroup", "I4/mmm", "--irrep", "X3-", "X2+",
                           "--order-parameter", "0", "a"])
    report("coupled order-parameter length checked against total dim",
           code != 0 and "needs 4 components" in out
           and "X3- + X2+" in out and "Traceback" not in out, out)

    # errors
    code, out = run_group(["--supergroup", "Pm-3m"])
    report("--supergroup without --irrep rejected cleanly",
           code != 0 and "requires --irrep" in out and "Traceback" not in out, out)

    code, out = run_group(["--supergroup", "Pm-3m", "--irrep", "QQ9"])
    report("unknown irrep rejected with available list",
           code != 0 and "not tabulated" in out and "Available irreps" in out
           and "Traceback" not in out, out)

    code, out = run_group(["--supergroup", "Pm-3m", "--irrep", "GM4-",
                           "--order-parameter", "0", "0"])
    report("wrong order-parameter length rejected cleanly",
           code != 0 and "needs 3 components" in out and "Traceback" not in out, out)

    code, out = run_group(["--product", "T2g", "T2g", "--pg", "m-3m",
                           "--irrep", "GM4-"])
    report("--irrep outside --supergroup rejected cleanly",
           code != 0 and "only used with --supergroup" in out
           and "Traceback" not in out, out)


def test_14_multiplet() -> None:
    print("\n[14] crystod-group --multiplet (multi-electron terms)")

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
        report("--visualize exit 0 and default file name",
               code == 0 and "Term-state viewer written to" in out
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


def test_15_poscar2cif() -> None:
    print("\n[15] crystod-group --poscar2cif / --cif2poscar (Bilbao-style CIF)")

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
        report("one representative site per orbit (F1 + Sc1, occupancy 1.0000)",
               re.search(r"F1 F 0\.\d{5} 0\.\d{5} 0\.\d{5} 1\.0000", content)
               is not None
               and "Sc1 Sc 0.00000 0.00000 0.00000 1.0000" in content, content)

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

        # --cif2poscar: inverse conversion (round trip)
        poscar = os.path.join(tmp, "221_PPOSCAR_SrTiO3")
        shutil.copy(POSCAR_SrTiO3, poscar)
        run_group(["--poscar2cif", "-c", poscar])
        os.remove(poscar)
        code, out = run_group(["--cif2poscar", "-c", poscar + ".cif"])
        report("--cif2poscar writes the input path without .cif (primitive)",
               code == 0 and os.path.isfile(poscar)
               and "primitive cell, 5 atoms" in out, out)
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


def test_16_symmetry_mode() -> None:
    print("\n[16] crystod-group --supergroup-cif (symmetry-mode analysis)")

    import spglib

    if tuple(int(x) for x in spglib.__version__.split(".")[:2]) < (2, 4):
        print(f"  [SKIP] spglib {spglib.__version__} < 2.4: subgroup "
              "identification is unreliable in old spglib; run this section "
              "in the crystod env.")
        return

    example = os.path.join(ROOT, "example", "16_symmetry_mode")
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
        # example/16_symmetry_mode/debug)
        debug = os.path.join(example, "debug")
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

        # cross-checks against crystod-phonon --modulation structures (section 25)
        modulation = os.path.join(ROOT, "example", "25_modulation", "ScF3_Pm-3m")
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


# ---------------------------------------------------------------- 35. Python API
def test_35_python_api() -> None:
    print("\n[35] Python API (crystod.salc / group / phonon / bz / mag / md / mol)")

    # import hygiene: plain `import crystod` must not drag in the heavy stack
    probe = (
        "import sys, crystod\n"
        "import crystod.salc, crystod.group, crystod.phonon\n"
        "import crystod.bz, crystod.mag, crystod.md, crystod.mol\n"
        "heavy = [m for m in ('phonopy', 'spgrep', 'pyscf', 'pymatgen', 'matplotlib')\n"
        "         if m in sys.modules]\n"
        "print('HEAVY:' + ','.join(heavy))\n"
        "print('VERSION:' + crystod.__version__)\n"
    )
    code, out = run_python(probe)
    report("importing crystod and all API domains exit 0", code == 0, out)
    report("no heavy dependency imported by the API namespaces",
           "HEAVY:\n" in out or "HEAVY:" in out and out.split("HEAVY:")[1].startswith("\n"),
           out)
    report("crystod.__version__ exposed", "VERSION:0.3" in out, out)

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
    code, cli_out = run_group(["--supergroup", "Pm-3m", "--irrep", "R4+"])
    api_pairs = {line.split()[0]: line.split()[2]
                 for line in api_out.splitlines() if line.startswith("R4+(")}
    cli_pairs = {line.split()[0]: line.split()[2]
                 for line in cli_out.splitlines() if line.startswith("R4+(")}
    report("API and crystod-group --supergroup agree on every direction",
           code == 0 and api_pairs and api_pairs == cli_pairs,
           f"api={api_pairs}\ncli={cli_pairs}")

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
    code, cli_out = run_group(["--supergroup", "Pm-3m", "--irrep", "M3+",
                               "--order-parameter", "0", "0", "a"])
    report("API direction string matches crystod-group --supergroup",
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

    # domain namespaces expose their advertised names
    probe = (
        "import crystod\n"
        "for name in ('salc', 'group', 'phonon', 'bz', 'mag', 'md', 'mol'):\n"
        "    module = getattr(crystod, name)\n"
        "    missing = [n for n in module.__all__ if not hasattr(module, n)]\n"
        "    print(name, len(module.__all__), 'MISSING' if missing else 'OK', missing or '')\n"
    )
    code, out = run_python(probe)
    report("every advertised API symbol resolves", code == 0 and "MISSING" not in out, out)
    report("all seven API domains are populated",
           code == 0 and all(f"\n{name} " in "\n" + out for name in
                             ("salc", "group", "phonon", "bz", "mag", "md", "mol")), out)


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
    13: test_13_isotropy,
    14: test_14_multiplet,
    15: test_15_poscar2cif,
    16: test_16_symmetry_mode,
    17: test_17_group_command,
    18: test_18_bz,
    19: test_19_bz_supercell,
    20: test_20_bz_command,
    21: test_21_phonon_irrep,
    22: test_22_phonon_fatband,
    23: test_23_phonon_lt,
    24: test_24_phonon_vector,
    25: test_25_modulation,
    26: test_26_vibration,
    27: test_27_phonon_command,
    28: test_28_spin_basis,
    29: test_29_mag_command,
    30: test_30_xdatcar2adp,
    31: test_31_md_command,
    32: test_32_mol,
    33: test_33_molod,
    34: test_34_mol_command,
    35: test_35_python_api,
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
