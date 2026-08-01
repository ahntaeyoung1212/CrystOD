"""Density of states and partial charges from the PySCF density matrix
(crystod --dos --pyscf).

The --chk file of ``crystod --diagram --pyscf`` records the converged
density matrices -- everything the band step needs.  This module reads it
back (or runs the SCF once) and diagonalizes NON-self-consistently on a
dense k mesh, VASP-style:

* total DOS and element x angular-momentum PDOS, Gaussian-broadened over
  the mesh (Loewdin |S^1/2 C|^2 weights by default, --projection mulliken
  for Re[C* (S C)]);
* Loewdin/Mulliken partial charges per atom with the per-shell breakdown,
  from the same occupied states (their sum reproduces the electron count
  exactly, a built-in sanity check);
* outputs in the calc_pyscf_dos.py style: a matplotlib PDF with VESTA
  element colors, a CSV of every curve, and a text summary.

Unlike the diagram's composition lists, nothing here is projected onto
fragment eigenstates -- the weights live directly in the (orthogonalized)
AO basis, so the double-counting of overlapping fragment states cannot
occur.  The Loewdin/Mulliken choice itself remains a convention, as every
atomic partition of a molecular density is.
"""

from __future__ import annotations

import numpy as np

HARTREE_TO_EV = 27.211386245988
ANGULAR_LETTERS = "spdfgh"


def _uniform_mesh(mesh):
    n1, n2, n3 = (int(n) for n in mesh)
    return np.array([
        [i / n1, j / n2, k / n3]
        for i in range(n1) for j in range(n2) for k in range(n3)
    ], dtype=float)


def _sqrt_matrix(matrix):
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, 0.0, None)
    return (vectors * np.sqrt(values)) @ vectors.conj().T


def report_and_write(cell, *, sublattice_left, sublattice_right,
                     dos_kmesh, sigma, projection, align, output_path,
                     structure_label, xrange=None, symprec=1e-5,
                     electrons=None, oxidation=None, basis=None,
                     pseudo=None, xc="pbe", kmesh=None, ke_cutoff=200.0,
                     scf_sigma=0.0, no_ghost=False, max_l=None, chk=None,
                     onsite=False, verbose=0):
    """Dense-mesh DOS/PDOS + partial charges from the (restarted) SCF."""
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .crystal_orbital_pyscf import PySCFCrystalOrbitalDiagram
    from .mo_diagram import element_color
    from .visualize_pyscf import _canonical_split

    if sublattice_left and sublattice_right:
        left, right = sublattice_left, sublattice_right
    else:
        left, right, _ = _canonical_split(cell, None)
    # the DOS only ever uses the crystal density: --onsite skips the two
    # fragment SCFs entirely (and accepts the crystal-only chk files an
    # --onsite diagram run writes)
    diagram = PySCFCrystalOrbitalDiagram(
        cell, left, right, symprec=symprec, electrons=electrons,
        oxidation=oxidation, basis=basis or "gth-dzvp-molopt-sr",
        pseudo=pseudo or "gth-pbe", xc=xc, kmesh=kmesh, ke_cutoff=ke_cutoff,
        sigma=scf_sigma, no_ghost=no_ghost, max_l=max_l,
        projection=projection, chk=chk, onsite=onsite, verbose=verbose,
    )
    print(f" * DOS from the PySCF density matrix: "
          f"{diagram.formula['left'] + diagram.formula['right']} *")
    print(f"   basis {diagram.basis_name} / pseudo {diagram.pseudo_name} / "
          f"functional {diagram.xc.upper()}; SCF mesh "
          f"{'x'.join(map(str, diagram.kmesh))}, DOS mesh "
          f"{'x'.join(map(str, dos_kmesh))}, {projection} projection")
    diagram.run()

    pcell = diagram.cells["mo"]
    kpoints = _uniform_mesh(dos_kmesh)
    kpts_abs = pcell.get_abs_kpts(kpoints)
    print(f"   non-self-consistent bands on {len(kpoints)} k points ...")
    energies, coefficients = diagram.mean_field["mo"].get_bands(
        kpts_abs, cell=pcell, dm_kpts=diagram.density_matrix["mo"])
    energies = [np.asarray(e).real * HARTREE_TO_EV for e in energies]
    overlaps = pcell.pbc_intor("int1e_ovlp", hermi=1, kpts=kpts_abs)

    # zero-temperature filling of the whole mesh at once (handles metals:
    # the Fermi level is where the sorted-state count reaches N)
    n_k = len(kpoints)
    n_electrons = float(diagram.electrons)
    all_states = np.sort(np.concatenate(energies))
    fill_index = int(round(n_electrons / 2.0 * n_k))
    fermi = 0.5 * (all_states[fill_index - 1] + all_states[fill_index]) \
        if fill_index < all_states.size else all_states[-1]
    vbm = float(all_states[fill_index - 1])
    cbm = float(all_states[fill_index]) if fill_index < all_states.size else None
    reference = vbm if align == "vbm" else 0.0

    # per-k projection weights and the accumulated charges
    groups: dict[tuple[str, str], np.ndarray] = {}   # (element, shell letter)
    shell_rows: dict[tuple[str, str], list] = {}     # (element, nl shell)
    atom_rows: dict[int, list] = {}
    for block in diagram.ao_blocks:
        letter = ANGULAR_LETTERS[block.l]
        rows = list(range(block.offset, block.offset + block.n_ao))
        key = (block.element, letter)
        groups[key] = np.concatenate([groups[key], rows]) \
            if key in groups else np.array(rows, dtype=int)
        shell_rows.setdefault((block.element, block.shell), []).extend(rows)
        atom_rows.setdefault(block.sites[0], []).extend(rows)

    padding = max(2.0, 4.0 * sigma)
    emin = min(float(e.min()) for e in energies) - reference - padding
    emax = max(float(e.max()) for e in energies) - reference + padding
    grid = np.linspace(emin, emax, 3000)
    total_dos = np.zeros_like(grid)
    pdos = {key: np.zeros_like(grid) for key in groups}
    # atomic charges are reported in BOTH conventions: with the diffuse
    # molopt basis they disagree wildly (Loewdin says ScF3 is made of
    # neutral atoms, Sc +0.004, because the diffuse Sc 5s/4d claim the
    # density sitting on the F sites; Mulliken gives Sc +1.57 / F -0.53)
    # -- an atomic partition of a continuous density is a convention, and
    # showing one number would hide that
    conventions = ("lowdin", "mulliken")
    charges = {name: {key: 0.0 for key in shell_rows} for name in conventions}
    atom_charge = {name: {atom: 0.0 for atom in atom_rows}
                   for name in conventions}
    weight_k = 2.0 / n_k    # closed shell, uniform mesh
    norm = 1.0 / (sigma * np.sqrt(2.0 * np.pi))

    for k_index in range(n_k):
        band_e = energies[k_index] - reference
        c = np.asarray(coefficients[k_index])
        overlap = np.asarray(overlaps[k_index])
        both = {
            "mulliken": (c.conj() * (overlap @ c)).real,
            "lowdin": np.abs(_sqrt_matrix(overlap) @ c) ** 2,
        }
        weights = both[projection]
        occupied = energies[k_index] <= fermi + 1e-9
        # DOS / PDOS: every band, Gaussian-broadened
        for band, energy in enumerate(band_e):
            shape = weight_k * norm * np.exp(
                -0.5 * ((grid - energy) / sigma) ** 2)
            total_dos += shape
            for key, rows in groups.items():
                pdos[key] += float(weights[rows, band].sum()) * shape
        # partial charges: occupied bands only, both conventions
        for name in conventions:
            for key, rows in shell_rows.items():
                charges[name][key] += weight_k * float(
                    both[name][np.asarray(rows)][:, occupied].sum())
            for atom, rows in atom_rows.items():
                atom_charge[name][atom] += weight_k * float(
                    both[name][np.asarray(rows)][:, occupied].sum())

    stem = output_path or f"DOS_{structure_label}"
    if stem.endswith(".pdf"):
        stem = stem[:-4]
    pdf_path, csv_path, txt_path = (stem + ".pdf", stem + ".csv", stem + ".txt")

    # ---- plot (calc_pyscf_dos.py style) ------------------------------------
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.plot(grid, total_dos, color="black", lw=1.8, label="total")
    linestyles = {"s": "-", "p": "--", "d": "-.", "f": ":"}
    for (element, letter), curve in pdos.items():
        if float(np.max(np.abs(curve))) < 1e-10:
            continue
        axis.plot(grid, curve, color=element_color(element),
                  ls=linestyles.get(letter, "-"), lw=1.4, alpha=0.85,
                  label=f"{element} {letter}-PDOS")
    if align == "vbm":
        axis.axvline(0.0, color="0.55", lw=0.9, ls=":")
        axis.set_xlabel("Energy - VBM (eV)")
    else:
        axis.set_xlabel("Energy (eV)")
    if xrange is not None:
        axis.set_xlim(xrange)
    axis.set_ylabel("DOS (states/eV/cell)")
    axis.set_title(f"{structure_label}  {diagram.xc.upper()}"
                   f"/{diagram.basis_name}")
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(pdf_path, dpi=200)
    plt.close(figure)

    # ---- CSV ---------------------------------------------------------------
    headers = ["energy_eV", "total_DOS"] + [
        f"{element}_{letter}_PDOS" for element, letter in pdos]
    table = np.column_stack([grid, total_dos]
                            + [pdos[key] for key in pdos])
    np.savetxt(csv_path, table, delimiter=",",
               header=",".join(headers), comments="")

    # ---- partial charges ---------------------------------------------------
    valence = {diagram.symbols[atom]: float(charge)
               for atom, charge in zip(range(len(diagram.symbols)),
                                       pcell.atom_charges())}
    lines = []
    for name in conventions:
        total_n = sum(atom_charge[name].values())
        lines.append(f" * {name.capitalize()} partial charges "
                     f"(sum of populations = {total_n:.4f} electrons, "
                     f"expected {n_electrons:g}) *")
        for atom in sorted(atom_rows):
            element = diagram.symbols[atom]
            population = atom_charge[name][atom]
            net = valence[element] - population
            count = max(1, sum(1 for a in atom_rows
                               if diagram.symbols[a] == element))
            shells = "  ".join(
                f"{shell} {charges[name][(el, shell)] / count:.3f}"
                for (el, shell) in charges[name] if el == element)
            lines.append(f"   {element}{atom}: population {population:7.4f}  "
                         f"net charge {net:+.3f}   ({shells})")
    lines.append("   (an atomic partition of a continuous density is a "
                 "convention: with this diffuse basis Loewdin tends toward "
                 "neutral atoms, Mulliken keeps the ionic picture)")
    gap = (cbm - vbm) if cbm is not None else None
    lines.append(f"   VBM {vbm - reference:+.3f} eV, "
                 + (f"CBM {cbm - reference:+.3f} eV, gap {gap:.3f} eV"
                    if gap is not None else "no empty states in the basis")
                 + f" (Fermi filling on the {'x'.join(map(str, dos_kmesh))} mesh)")
    report = "\n".join(lines)
    print(report)
    with open(txt_path, "w") as handle:
        handle.write(
            f"# DOS/PDOS + {projection} partial charges for {structure_label}\n"
            f"# {diagram.xc.upper()}/{diagram.basis_name}, SCF mesh "
            f"{'x'.join(map(str, diagram.kmesh))}, DOS mesh "
            f"{'x'.join(map(str, dos_kmesh))}, sigma {sigma:g} eV, "
            f"reference {'VBM' if align == 'vbm' else 'absolute'}\n"
            + report + "\n")
    print(f"DOS plot written to {pdf_path}")
    print(f"DOS curves written to {csv_path}")
    print(f"Summary written to {txt_path}")


def main(argv: list[str] | None = None) -> None:
    import argparse
    from pathlib import Path

    from .crystal_orbital_diagram import parse_oxidation_tokens

    parser = argparse.ArgumentParser(
        description="DOS/PDOS and partial charges from the PySCF density "
                    "matrix (crystod --dos --pyscf).")
    parser.add_argument("--poscar", default="POSCAR")
    parser.add_argument("--co-left", nargs="+", default=None, metavar="FORMULA",
                        help="fragment split, only to match an existing --chk "
                        "(default: first element vs the rest)")
    parser.add_argument("--co-right", nargs="+", default=None, metavar="FORMULA")
    parser.add_argument("--dos-kmesh", type=int, nargs=3, default=(8, 8, 8),
                        metavar=("N1", "N2", "N3"),
                        help="dense non-self-consistent mesh (default 8 8 8)")
    parser.add_argument("--sigma", type=float, default=0.15,
                        help="Gaussian broadening in eV (default 0.15)")
    parser.add_argument("--projection", choices=("lowdin", "mulliken"),
                        default="lowdin")
    parser.add_argument("--align", choices=("vbm", "absolute"), default="vbm")
    parser.add_argument("--xrange", type=float, nargs=2, default=None,
                        metavar=("XMIN", "XMAX"))
    parser.add_argument("--electrons", type=float, default=None)
    parser.add_argument("--oxidation", nargs="+", default=None, metavar="EL=Q")
    parser.add_argument("--basis", default="gth-dzvp-molopt-sr")
    parser.add_argument("--pseudo", default="gth-pbe")
    parser.add_argument("--xc", default="pbe")
    parser.add_argument("--kmesh", type=int, nargs=3, default=None,
                        metavar=("N1", "N2", "N3"),
                        help="SCF mesh (must match an existing --chk)")
    parser.add_argument("--ke-cutoff", type=float, default=200.0)
    parser.add_argument("--scf-sigma", type=float, default=0.0)
    parser.add_argument("--no-ghost", action="store_true")
    parser.add_argument("--max-l", type=int, default=None)
    parser.add_argument("--chk", default=None, metavar="FILE")
    parser.add_argument("--onsite", action="store_true",
                        help="run (or reuse) only the crystal SCF -- the DOS "
                        "never needs the fragment densities; also accepts "
                        "the crystal-only chk files an --onsite diagram run "
                        "writes")
    parser.add_argument("--output", default=None)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args(argv)

    from .star_of_k import read_poscar_or_exit

    cell = read_poscar_or_exit(args.poscar)
    stem = Path(args.poscar).name
    for extension in (".vasp", ".poscar"):
        if stem.lower().endswith(extension):
            stem = stem[: -len(extension)]
    report_and_write(
        cell,
        sublattice_left=args.co_left,
        sublattice_right=args.co_right,
        dos_kmesh=args.dos_kmesh,
        sigma=args.sigma,
        projection=args.projection,
        align=args.align,
        output_path=args.output,
        structure_label=stem,
        xrange=args.xrange,
        symprec=args.tolerance,
        electrons=args.electrons,
        oxidation=(parse_oxidation_tokens(args.oxidation)
                   if args.oxidation else None),
        basis=args.basis,
        pseudo=args.pseudo,
        xc=args.xc,
        kmesh=args.kmesh,
        ke_cutoff=args.ke_cutoff,
        scf_sigma=args.scf_sigma,
        no_ghost=args.no_ghost,
        max_l=args.max_l,
        chk=args.chk,
        onsite=args.onsite,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
