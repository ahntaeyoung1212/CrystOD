#!/usr/bin/env python3
"""Generate the reference atomic-level data for the crystal-orbital
diagrams: one neutral-atom PySCF calculation per element with the user's
standard driver (~/script/calc_pyscf.py), archived verbatim as
reference/atomic_level_{El}.

    python3 ~/script/calc_pyscf.py --xyz XYZ_{El}.xyz --spin {2S} \
        --basis def2-svp [--ecp def2-svp] > atomic_level_{El}

- spin = 2S of the neutral-atom ground state (Hund);
- theory: the driver default (Hartree-Fock; RHF for 2S = 0, else UHF), so
  the core eigenvalues are Koopmans binding energies;
- elements beyond Kr need the def2 effective core potential (the def2
  bases are ECP-based there); the removed core shells simply do not exist
  in the data -- exactly the pseudopotential philosophy.

Run inside reference/:  python3 ../script/generate_atomic_levels.py
Afterwards run script/collect_atomic_levels.py to regenerate
crystod/atomic_levels.py from the archived files.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CALC = Path.home() / "script" / "calc_pyscf.py"
BASIS = "def2-svp"

# 2S (number of unpaired electrons) of the neutral-atom ground state
SPINS = {
    "H": 1, "Li": 1, "Be": 0, "B": 1, "C": 2, "N": 3, "O": 2, "F": 1,
    "Na": 1, "Mg": 0, "Al": 1, "Si": 2, "P": 3, "S": 2, "Cl": 1,
    "K": 1, "Ca": 0, "Sc": 1, "Ti": 2, "V": 3, "Cr": 6, "Mn": 5,
    "Fe": 4, "Co": 3, "Ni": 2, "Cu": 1, "Zn": 0, "Ga": 1, "Ge": 2,
    "As": 3, "Se": 2, "Br": 1, "Rb": 1, "Sr": 0, "Y": 1, "Zr": 2, "Nb": 5,
    "Mo": 6, "Cd": 0, "Sn": 2, "Sb": 3, "Te": 2, "I": 1, "Cs": 1,
    "Ba": 0,
    # rare earths (Hund ground configurations: La 5d1 6s2, Ce 4f1 5d1 6s2,
    # Pr..Eu 4f(n) 6s2, Gd 4f7 5d1 6s2, Tb..Yb 4f(n) 6s2, Lu 4f14 5d1 6s2)
    "La": 1, "Ce": 2, "Pr": 3, "Nd": 4, "Pm": 5, "Sm": 6, "Eu": 7,
    "Gd": 8, "Tb": 5, "Dy": 4, "Ho": 3, "Er": 2, "Tm": 1, "Yb": 0, "Lu": 1,
    "Pb": 2, "Bi": 3,
}

# def2 sets are ECP-based beyond Kr
ECP_ELEMENTS = {"Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Cd", "Sn", "Sb", "Te",
                "I", "Cs", "Ba", "La", "Pb", "Bi"}

# def2 does not cover Ce-Lu: use ECP-based lanthanide sets that keep the
# 4f shell in the valence space (Stuttgart RSC 1997 / ECP28MWB for Ce-Yb;
# CRENBL / Xe-core for Lu).  These run through the built-in PySCF path
# below (the external driver restricts --basis to common sets).  La uses
# def2-svp but ALSO runs built-in: its near-degenerate 5d1 ground state
# must be produced by the same SCF recipe that the collect script re-runs.
BASIS_OVERRIDES = {
    "La": "def2-svp",
    "Ce": "stuttgartrsc", "Pr": "stuttgartrsc", "Nd": "stuttgartrsc",
    "Pm": "stuttgartrsc", "Sm": "stuttgartrsc", "Eu": "stuttgartrsc",
    "Gd": "stuttgartrsc", "Tb": "stuttgartrsc", "Dy": "stuttgartrsc",
    "Ho": "stuttgartrsc", "Er": "stuttgartrsc", "Tm": "stuttgartrsc",
    "Yb": "stuttgartrsc", "Lu": "crenbl",
}


def basis_of(element: str) -> str:
    """Basis-set name used for one element's archived calculation."""
    return BASIS_OVERRIDES.get(element, BASIS)


HARTREE_TO_EV = 27.211386


def run_builtin(element: str, spin: int, out) -> None:
    """Neutral-atom HF with PySCF directly, archived in the driver format.

    Used for the elements whose basis is not offered by the external
    driver (BASIS_OVERRIDES).  Plain RHF/UHF first, second-order (newton)
    solver as the convergence fallback -- script/collect_atomic_levels.py
    re-runs the same recipe (and refreshes the archive when the open-shell
    degeneracy makes the re-run land on an equivalent broken-symmetry
    solution with a permuted orbital spectrum).
    """
    from pyscf import gto

    basis = basis_of(element)
    mol = gto.M(atom=f"{element} 0 0 0", basis=basis, ecp=basis,
                spin=spin, verbose=0)
    mf = converge_atom(mol, spin)
    if not mf.converged:
        raise RuntimeError(f"SCF for {element} did not converge")
    write_archive(element, spin, basis, mf, out)


def converge_atom(mol, spin: int):
    """Converge the neutral-atom HF: plain SCF, then the second-order
    (newton) solver, then newton from the spherically averaged atomic
    guess.  Shared with script/collect_atomic_levels.py so the archive
    and the collected table always come from the same recipe."""
    from pyscf import scf

    mf = scf.RHF(mol) if spin == 0 else scf.UHF(mol)
    mf.verbose = 0
    mf.kernel()
    if not mf.converged:
        mf = mf.newton()
        mf.kernel()
    if not mf.converged:
        base = scf.RHF(mol) if spin == 0 else scf.UHF(mol)
        base.verbose = 0
        dm0 = base.get_init_guess(mol, key="atom")
        mf = base.newton()
        mf.kernel(dm0=dm0)
    return mf


def write_archive(element: str, spin: int, basis: str, mf, out) -> None:
    """Write one converged atomic calculation in the driver text format."""
    lines = [
        "==========================================",
        " PySCF Calculation (built-in atomic run)",
        "==========================================",
        f"Element       : {element}",
        f"Basis         : {basis}",
        "Charge        : 0",
        f"Spin (2S)     : {spin}",
        f"ECP           : {basis}",
        "Theory        : scf",
        "------------------------------------------",
        f"Method        : {'RHF' if spin == 0 else 'UHF'}",
        f"Total Energy  : {mf.e_tot * HARTREE_TO_EV:.6f} eV",
        "",
        "MO energies (Hartree / eV):",
    ]
    if spin == 0:
        channels = [(None, mf.mo_energy)]
    else:
        channels = [("Alpha", mf.mo_energy[0]), ("Beta", mf.mo_energy[1])]
    for channel_name, energies in channels:
        if channel_name:
            lines.append(f"  {channel_name} MO energies:")
        for index, energy in enumerate(energies, start=1):
            lines.append(
                f"  MO {index:2d}: {float(energy):11.6f}  "
                f"{float(energy) * HARTREE_TO_EV:11.4f}"
            )
    lines.append("==========================================")
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    reference = Path(__file__).resolve().parent.parent / "reference"
    template = (reference / "XYZ_H.xyz").read_text()
    selected = sys.argv[1:]
    unknown = [element for element in selected if element not in SPINS]
    if unknown:
        raise SystemExit(f"ERROR: unknown element(s) {', '.join(unknown)}.")
    failed = []
    count = 0
    for element, spin in SPINS.items():
        if selected and element not in selected:
            continue
        count += 1
        out = reference / f"atomic_level_{element}"
        print(f" {element:<2} (2S={spin}, {basis_of(element)}"
              + (", ecp" if element in ECP_ELEMENTS or element in BASIS_OVERRIDES
                 else "") + ") ... ",
              end="", flush=True)
        if element in BASIS_OVERRIDES:
            try:
                run_builtin(element, spin, out)
            except Exception as error:
                print("FAILED")
                failed.append(element)
                print(f"    {error}")
                continue
            print("ok")
            continue
        xyz = reference / f"XYZ_{element}.xyz"
        if element != "H":
            xyz.write_text(template.replace("H", element))
        command = [
            sys.executable, str(CALC), "--xyz", xyz.name,
            "--spin", str(spin), "--basis", BASIS,
        ]
        if element in ECP_ELEMENTS:
            command += ["--ecp", BASIS]
        result = subprocess.run(
            command, cwd=reference, capture_output=True, text=True
        )
        if result.returncode != 0 or "MO energies" not in result.stdout:
            print("FAILED")
            failed.append(element)
            print(result.stdout[-500:], result.stderr[-500:])
            continue
        out.write_text(result.stdout)
        print("ok")
    if failed:
        raise SystemExit(f"FAILED: {', '.join(failed)}")
    print(f"\nAll {count} element(s) written to {reference}/atomic_level_*")


if __name__ == "__main__":
    main()
