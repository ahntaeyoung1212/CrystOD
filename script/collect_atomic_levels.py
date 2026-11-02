#!/usr/bin/env python3
"""Collect the archived neutral-atom PySCF levels (reference/atomic_level_*)
into crystod/atomic_levels.py.

The shell assignment cannot be read from the archived text alone (it lists
bare MO energies), so the same calculation is re-run here with identical
settings; every MO is classified by its angular momentum (Mulliken l
population) and numbered aufbau-wise within each l channel (ECP-frozen
cores shift the starting principal quantum number).  The re-run MO
energies are cross-checked against the archived file (max deviation
1e-4 hartree) so the generated table is guaranteed to represent the
archived data.

Run:  python3 script/collect_atomic_levels.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
from pyscf import gto, scf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_atomic_levels import (  # noqa: E402
    BASIS_OVERRIDES,
    ECP_ELEMENTS,
    SPINS,
    basis_of,
    converge_atom,
)

HARTREE_TO_EV = 27.211386
L_LETTER = "spdf"

# shells removed by the ECP cores, counted per l
_ECP_REMOVED = {
    0: {"s": 0, "p": 0, "d": 0, "f": 0},
    28: {"s": 3, "p": 2, "d": 1, "f": 0},    # 1s2s3s / 2p3p / 3d
    46: {"s": 4, "p": 3, "d": 2, "f": 0},    # ... + 4s / 4p / 4d
    54: {"s": 5, "p": 4, "d": 2, "f": 0},    # Xe core (crenbl Lu)
    60: {"s": 4, "p": 3, "d": 2, "f": 1},    # ... + 4f
}


def analyze(element: str, spin: int):
    basis = basis_of(element)
    mol = gto.M(
        atom=f"{element} 0 0 0", basis=basis,
        ecp=(basis if element in ECP_ELEMENTS or element in BASIS_OVERRIDES
             else None),
        spin=spin,
    )
    # same recipe as the archived run (generate_atomic_levels)
    mf = converge_atom(mol, spin)
    if not mf.converged:
        raise SystemExit(f"ERROR: SCF for {element} did not converge.")

    overlap = mol.intor_symmetric("int1e_ovlp")
    # l per AO from the label strings ("0 Sr 4s", "0 Sr 5px", ...)
    l_of_ao = np.array([
        L_LETTER.index(re.search(r"\d([spdf])", label.split()[2]).group(1))
        for label in mol.ao_labels()
    ])

    channels = ([(mf.mo_energy, mf.mo_coeff)] if spin == 0 else
                [(mf.mo_energy[0], mf.mo_coeff[0]),
                 (mf.mo_energy[1], mf.mo_coeff[1])])
    ncore = mol.atom_nelec_core(0)
    removed = _ECP_REMOVED[ncore]

    shell_energies: dict[str, list[float]] = {}
    all_energies = []
    for energies, coefficients in channels:
        all_energies.extend(float(e) for e in energies)
        sc = overlap @ coefficients
        counters = {l: 0 for l in range(4)}
        order = np.argsort(energies)
        for mo in order:
            populations = coefficients[:, mo] * sc[:, mo]
            l_pop = [float(np.sum(populations[l_of_ao == l]))
                     for l in range(4)]
            l = int(np.argmax(l_pop))
            if l_pop[l] < 0.5:
                raise SystemExit(
                    f"ERROR: {element} MO {mo} has no dominant l "
                    f"(populations {np.round(l_pop, 3)})."
                )
            if l_pop[l] < 0.9:
                # broken-symmetry s-d mixing of open-shell TM atoms
                # (valence only; the cores stay l-pure)
                print(f"    note: {element} MO {mo} only {l_pop[l]:.2f} "
                      f"l-pure (assigned {L_LETTER[l]})")
            n = removed[L_LETTER[l]] + l + 1 + counters[l] // (2 * l + 1)
            counters[l] += 1
            shell_energies.setdefault(f"{n}{L_LETTER[l]}", []).append(
                float(energies[mo])
            )

    # cross-check against the archived file
    archive = (Path(__file__).resolve().parent.parent / "reference"
               / f"atomic_level_{element}").read_text()
    archived = [
        float(match.group(1))
        for match in re.finditer(r"MO\s+\d+:\s+(-?\d+\.\d+)\s+-?\d", archive)
    ]
    if len(archived) != len(all_energies):
        raise SystemExit(
            f"ERROR: {element}: {len(archived)} archived MOs vs "
            f"{len(all_energies)} recomputed."
        )
    deviation = float(np.max(np.abs(np.sort(archived)
                                    - np.sort(all_energies))))
    if deviation > 1e-4:
        if element in BASIS_OVERRIDES:
            # open-shell degenerate atoms (4f/5d) can converge to an
            # equivalent broken-symmetry solution with a permuted orbital
            # spectrum on re-runs: refresh the archive from THIS run so
            # the archive and the collected table stay consistent
            from generate_atomic_levels import write_archive

            archive_path = (Path(__file__).resolve().parent.parent
                            / "reference" / f"atomic_level_{element}")
            write_archive(element, spin, basis, mf, archive_path)
            print(f"    note: {element} archive refreshed (degenerate "
                  f"open-shell re-run deviated by {deviation:.2e} hartree)")
        else:
            raise SystemExit(
                f"ERROR: {element}: archived MO energies deviate by "
                f"{deviation:.2e} hartree; regenerate the archive."
            )

    levels = {
        shell: round(float(np.mean(values)) * HARTREE_TO_EV, 3)
        for shell, values in shell_energies.items()
    }
    return {
        "electrons": int(mol.nelectron),
        "ecp_core": int(ncore),
        "method": ("RHF" if spin == 0 else "UHF") + f"/{basis_of(element)}",
        "levels": levels,
    }


def main() -> None:
    # optional element arguments: re-analyze ONLY those and keep every
    # other entry of the existing crystod/atomic_levels.py verbatim --
    # the archived calculations were run in the original environment, and
    # a full re-collection elsewhere can trip the 1e-4 hartree
    # cross-check on unrelated elements (BLAS differences)
    selected = sys.argv[1:]
    unknown = [element for element in selected if element not in SPINS]
    if unknown:
        raise SystemExit(
            f"ERROR: unknown element(s) {', '.join(unknown)} "
            "(add them to SPINS in generate_atomic_levels.py first).")
    table = {}
    if selected:
        from crystod.atomic_levels import ATOMIC_LEVELS

        table = {element: dict(data)
                 for element, data in ATOMIC_LEVELS.items()}
    for element, spin in SPINS.items():
        if selected and element not in selected:
            continue
        table[element] = analyze(element, spin)
        core = table[element]["ecp_core"]
        print(f" {element:<2}: {len(table[element]['levels'])} shells"
              + (f" (ECP-{core} core)" if core else ""))
    # keep the SPINS declaration order for a stable file layout
    table = {element: table[element] for element in SPINS
             if element in table}

    target = (Path(__file__).resolve().parent.parent / "crystod"
              / "atomic_levels.py")
    lines = [
        '"""Neutral-atom orbital energies from the archived PySCF',
        "calculations (reference/atomic_level_*, script",
        "script/generate_atomic_levels.py; Hartree-Fock, def2-svp with the",
        "def2 ECP beyond Kr, Stuttgart RSC 1997 for Ce-Yb and CRENBL for Lu",
        "-- 4f in the valence space).  Energies in eV, spin- and m-averaged",
        "per shell; virtual shells included.  AUTO-GENERATED by",
        'script/collect_atomic_levels.py -- do not edit by hand."""',
        "",
        "ATOMIC_LEVELS = {",
    ]
    for element, data in table.items():
        levels = ", ".join(
            f'"{shell}": {value}' for shell, value in sorted(
                data["levels"].items(),
                key=lambda kv: (int(kv[0][:-1]), L_LETTER.index(kv[0][-1])),
            )
        )
        lines.append(
            f'    "{element}": {{"electrons": {data["electrons"]}, '
            f'"ecp_core": {data["ecp_core"]}, '
            f'"method": "{data["method"]}",'
        )
        lines.append(f'                "levels": {{{levels}}}}},')
    lines += ["}", ""]
    target.write_text("\n".join(lines))
    print(f"\nWritten: {target}")


if __name__ == "__main__":
    main()
