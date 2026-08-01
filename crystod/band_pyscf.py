"""Electronic band structure and fatbands from the PySCF density matrix.

``crystod --band --pyscf``: VASP-style two-step -- the SCF (or a ``--chk``
restart, including the crystal-only checkpoints of ``--onsite`` runs)
provides the converged density matrix, and ``get_bands`` diagonalizes it
non-self-consistently along the automatic seekpath high-symmetry path.
The Fermi level / VBM / CBM come from a zero-temperature filling of the
uniform SCF mesh (metal-safe), and the plot is referenced to the VBM by
default.  ``--fatband`` adds the same per-AO projections the DOS and the
crystal-orbital compositions use (``--projection`` Loewdin |S^(1/2)c|^2 by
default): one overview page colored per element (VESTA colors) plus one
page per element with the per-l (s/p/d/f) breakdown.  The idea: read the
whole band structure first, then zoom into the special k points with the
crystal-orbital diagram.
"""
from __future__ import annotations

import numpy as np

from .crystal_orbital_pyscf import HARTREE_TO_EV
from .dos_pyscf import ANGULAR_LETTERS, _sqrt_matrix, _uniform_mesh

# distinct colors for the l channels on the per-element pages
L_COLORS = {"s": "#1f77b4", "p": "#e07b39", "d": "#2ca02c",
            "f": "#9467bd", "g": "#8c564b"}


def _prettify(label: str) -> str:
    if label in ("GAMMA", "G"):
        return r"$\Gamma$"
    if "_" in label:
        base, sub = label.split("_", 1)
        return f"$\\mathrm{{{base}}}_{{{sub}}}$"
    return label


def _seekpath_path(diagram, tolerance: float, npoints: int):
    """(kpts_frac, labels, breaks, path_text) from seekpath on the
    diagram's primitive cell."""
    try:
        import seekpath
    except ImportError:
        raise SystemExit("ERROR: seekpath is required for the automatic "
                         "k-path (`pip install seekpath`).")
    from pyscf.data.elements import charge

    numbers = [charge(symbol) for symbol in diagram.symbols]
    result = seekpath.get_path(
        (diagram.lattice, diagram.positions, numbers), symprec=tolerance)
    if not np.allclose(np.array(result["primitive_lattice"], dtype=float),
                       diagram.lattice, atol=1e-4):
        print("   NOTE: the cell is not the seekpath standardized primitive "
              "cell;\n         the path coordinates refer to the "
              "standardized cell.")
    coords = result["point_coords"]
    # continuous label chains from seekpath's (start, end) pairs
    chains: list[list[str]] = []
    for start, end in result["path"]:
        if chains and chains[-1][-1] == start:
            chains[-1].append(end)
        else:
            chains.append([start, end])
    kpts: list[np.ndarray] = []
    labels: list[tuple[int, str]] = []
    breaks: list[int] = []
    for chain_index, chain in enumerate(chains):
        if chain_index > 0:
            breaks.append(len(kpts))
        for leg_index in range(len(chain) - 1):
            leg = np.linspace(np.asarray(coords[chain[leg_index]]),
                              np.asarray(coords[chain[leg_index + 1]]),
                              npoints)
            if leg_index > 0:
                leg = leg[1:]   # the joint is already there
            else:
                labels.append((len(kpts), chain[leg_index]))
            kpts.extend(leg)
            labels.append((len(kpts) - 1, chain[leg_index + 1]))
    path_text = "  |  ".join("-".join(chain) for chain in chains)
    return np.array(kpts), labels, breaks, path_text


def _distances(kpts_frac, reciprocal, breaks):
    cartesian = kpts_frac @ reciprocal
    steps = np.linalg.norm(np.diff(cartesian, axis=0), axis=1)
    for index in breaks:
        if 0 < index <= len(steps):
            steps[index - 1] = 0.0
    return np.concatenate([[0.0], np.cumsum(steps)])


def _ticks(labels, distances):
    ticks, texts = [], []
    for index, name in labels:
        position = float(distances[index])
        text = _prettify(name)
        if ticks and abs(position - ticks[-1]) < 1e-8:
            if texts[-1] != text:
                texts[-1] = f"{texts[-1]}|{text}"
            continue
        ticks.append(position)
        texts.append(text)
    return ticks, texts


def _band_axis(axis, distances, ticks, texts, window, reference_label):
    axis.axhline(0.0, color="crimson", linewidth=0.8, linestyle="--",
                 zorder=0)
    for tick in ticks[1:-1]:
        axis.axvline(tick, color="0.3", linewidth=0.6, zorder=0)
    axis.set_xticks(ticks)
    axis.set_xticklabels(texts)
    axis.set_xlim(distances[0], distances[-1])
    axis.set_ylim(*window)
    axis.set_ylabel("E (eV)" if reference_label is None
                    else f"E $-$ {reference_label} (eV)")
    axis.tick_params(direction="in", top=True, right=True)


def report_and_write(cell, *, sublattice_left, sublattice_right, fatband,
                     band_points, projection, align, output_path,
                     structure_label, window=None, symprec=1e-5,
                     electrons=None, oxidation=None, basis=None,
                     pseudo=None, xc="pbe", kmesh=None, ke_cutoff=200.0,
                     scf_sigma=0.0, no_ghost=False, max_l=None, chk=None,
                     onsite=False, verbose=0):
    """Terminal report + band-structure (and fatband) plots."""
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
    diagram = PySCFCrystalOrbitalDiagram(
        cell, left, right, symprec=symprec, electrons=electrons,
        oxidation=oxidation, basis=basis or "gth-dzvp-molopt-sr",
        pseudo=pseudo or "gth-pbe", xc=xc, kmesh=kmesh, ke_cutoff=ke_cutoff,
        sigma=scf_sigma, no_ghost=no_ghost, max_l=max_l,
        projection=projection, chk=chk, onsite=onsite, verbose=verbose,
    )
    print(f" * Band structure from the PySCF density matrix: "
          f"{diagram.formula['left'] + diagram.formula['right']} *")
    print(f"   basis {diagram.basis_name} / pseudo {diagram.pseudo_name} / "
          f"functional {diagram.xc.upper()}; SCF mesh "
          f"{'x'.join(map(str, diagram.kmesh))}"
          + (f", {projection} fatband projection" if fatband else ""))
    diagram.run()

    pcell = diagram.cells["mo"]
    mean_field = diagram.mean_field["mo"]
    density = diagram.density_matrix["mo"]

    # ---- Fermi level / VBM / CBM from the uniform SCF mesh -----------------
    mesh_abs = pcell.get_abs_kpts(_uniform_mesh(diagram.kmesh))
    mesh_energies, _ = mean_field.get_bands(mesh_abs, cell=pcell,
                                            dm_kpts=density)
    mesh_energies = [np.asarray(e).real * HARTREE_TO_EV
                     for e in mesh_energies]
    n_k = len(mesh_abs)
    all_states = np.sort(np.concatenate(mesh_energies))
    half_filling = float(diagram.electrons) / 2.0
    # an odd/fractional electron count (rigid-band doping) has no integer
    # band cut: treat it like a metal and reference E_F, never a
    # fictitious VBM from a rounded occupation
    partial = abs(half_filling - round(half_filling)) > 1e-6
    fill_index = int(round(half_filling * n_k))
    vbm = float(all_states[fill_index - 1])
    cbm = (float(all_states[fill_index])
           if fill_index < all_states.size else None)
    fermi = 0.5 * (vbm + cbm) if cbm is not None else vbm
    metallic = partial or (cbm is not None and (cbm - vbm) < 0.05)
    if align != "vbm":
        reference, reference_label, reference_name = 0.0, None, "absolute"
    elif metallic:
        reference, reference_label, reference_name = (fermi,
                                                      "E$_\\mathrm{F}$",
                                                      "E_F")
    else:
        reference, reference_label, reference_name = (vbm,
                                                      "E$_\\mathrm{VBM}$",
                                                      "VBM")

    # ---- the k path ---------------------------------------------------------
    kpts_frac, labels, breaks, path_text = _seekpath_path(
        diagram, symprec, band_points)
    print(f"   k path : {path_text}  ({len(kpts_frac)} points, "
          f"{band_points} per leg)")
    reciprocal = 2.0 * np.pi * np.linalg.inv(diagram.lattice).T
    distances = _distances(kpts_frac, reciprocal, breaks)
    ticks, texts = _ticks(labels, distances)

    kpts_abs = pcell.get_abs_kpts(kpts_frac)
    print(f"   non-self-consistent bands on {len(kpts_abs)} k points ...")
    energies, coefficients = mean_field.get_bands(kpts_abs, cell=pcell,
                                                  dm_kpts=density)
    bands = np.array([np.asarray(e).real for e in energies]) * HARTREE_TO_EV \
        - reference

    if metallic:
        kind = "fractional" if partial else "metallic"
        print(f"   {kind} filling on the SCF mesh; E_F = {fermi:.3f} eV "
              "(absolute scale)")
    elif cbm is None:
        print("   no empty states in the basis (every band filled); "
              "band-edge report skipped")
    else:
        nocc = int(round(float(diagram.electrons) / 2.0))
        top = bands[:, nocc - 1]
        bottom = bands[:, nocc]
        gap = float(bottom.min() - top.max())

        def _where(index):
            position = distances[index]
            for tick, text in zip(ticks, texts):
                if abs(position - tick) < 1e-6:
                    return text.replace("$", "").replace("\\mathrm", "") \
                        .replace("{", "").replace("}", "").replace("\\Gamma",
                                                                   "GM")
            return "(" + ",".join(f"{x:.3g}" for x in kpts_frac[index]) + ")"

        print(f"   VBM {top.max():+.3f} eV at {_where(int(top.argmax()))}, "
              f"CBM {bottom.min():+.3f} eV at {_where(int(bottom.argmin()))}"
              f"  ->  gap {gap:.3f} eV on this path "
              f"(mesh gap {cbm - vbm:.3f} eV)")

    if window is None:
        window = (max(float(bands.min()) - 1.0, -25.0),
                  min(float(bands.max()) + 1.0, 15.0))

    stem = output_path or f"BAND_{structure_label}"
    if stem.endswith(".pdf"):
        stem = stem[:-4]

    # ---- plain band structure ----------------------------------------------
    figure, axis = plt.subplots(figsize=(5.0, 6.0))
    for band in range(bands.shape[1]):
        axis.plot(distances, bands[:, band], color="#1f4e9c", linewidth=1.1,
                  zorder=3)
    _band_axis(axis, distances, ticks, texts, window, reference_label)
    axis.set_title(f"{structure_label}  ({diagram.xc.upper()})", fontsize=10)
    figure.tight_layout()
    figure.savefig(f"{stem}.pdf")
    plt.close(figure)
    outputs = [f"{stem}.pdf"]

    # ---- fatbands -----------------------------------------------------------
    if fatband:
        groups: dict[tuple[str, str], np.ndarray] = {}
        for block in diagram.ao_blocks:
            key = (block.element, ANGULAR_LETTERS[block.l])
            rows = np.arange(block.offset, block.offset + block.n_ao)
            groups[key] = (np.concatenate([groups[key], rows])
                           if key in groups else rows)
        elements = list(dict.fromkeys(el for el, _ in groups))
        weights = {key: np.zeros_like(bands) for key in groups}
        overlaps = pcell.pbc_intor("int1e_ovlp", hermi=1, kpts=kpts_abs)
        for k_index in range(len(kpts_abs)):
            c = np.asarray(coefficients[k_index])
            S = np.asarray(overlaps[k_index])
            if projection == "mulliken":
                gross = (c.conj() * (S @ c)).real
            else:
                gross = np.abs(_sqrt_matrix(S) @ c) ** 2
            for key, rows in groups.items():
                weights[key][k_index] = gross[rows].sum(axis=0)

        x_scatter = np.repeat(distances[:, None], bands.shape[1], axis=1)

        def _fat(axis, weight, color, label):
            for band in range(bands.shape[1]):
                axis.plot(distances, bands[:, band], color="0.25",
                          linewidth=0.4, zorder=2)
            axis.scatter(x_scatter.ravel(), bands.ravel(),
                         s=np.clip(weight, 0.0, None).ravel() * 45.0,
                         color=color, alpha=0.75, edgecolors="none",
                         zorder=3, label=label)
            _band_axis(axis, distances, ticks, texts, window,
                       reference_label)
            axis.legend(loc="upper right", fontsize=8, framealpha=0.9,
                        markerscale=2.0)

        # overview: every element on one panel (VESTA colors)
        figure, axis = plt.subplots(figsize=(5.4, 6.0))
        for element in elements:
            total = sum(weights[key] for key in groups if key[0] == element)
            _fat(axis, total, element_color(element), element)
        axis.set_title(f"{structure_label} fatbands ({projection})",
                       fontsize=10)
        figure.tight_layout()
        figure.savefig(f"{stem}_fatband.pdf")
        plt.close(figure)
        outputs.append(f"{stem}_fatband.pdf")

        # one page per element: total + per-l panels
        for element in elements:
            letters = [letter for el, letter in groups if el == element]
            figure, axes = plt.subplots(
                1, 1 + len(letters), figsize=(3.1 * (1 + len(letters)), 6.0),
                sharey=True)
            axes = np.atleast_1d(axes)
            total = sum(weights[key] for key in groups if key[0] == element)
            _fat(axes[0], total, element_color(element), f"{element} total")
            for axis, letter in zip(axes[1:], letters):
                _fat(axis, weights[(element, letter)],
                     L_COLORS.get(letter, "0.4"), f"{element} {letter}")
                axis.set_ylabel("")
            axes[0].set_title(f"{structure_label}: {element} "
                              f"({projection})", fontsize=10)
            figure.tight_layout()
            figure.savefig(f"{stem}_fatband_{element}.pdf")
            plt.close(figure)
            outputs.append(f"{stem}_fatband_{element}.pdf")

    # ---- data files ---------------------------------------------------------
    with open(f"{stem}.csv", "w") as handle:
        keys = sorted(groups) if fatband else []
        handle.write("distance,band,energy_eV"
                     + "".join(f",{el}_{letter}" for el, letter in keys)
                     + "\n")
        for band in range(bands.shape[1]):
            for k_index in range(len(distances)):
                row = (f"{distances[k_index]:.6f},{band},"
                       f"{bands[k_index, band]:.6f}")
                row += "".join(f",{weights[key][k_index, band]:.4f}"
                               for key in keys)
                handle.write(row + "\n")
    outputs.append(f"{stem}.csv")

    with open(f"{stem}.txt", "w") as handle:
        handle.write(f"{structure_label}  {diagram.xc.upper()}/"
                     f"{diagram.basis_name}\n"
                     f"k path: {path_text}\n"
                     f"energies referenced to {reference_name}\n")
        if metallic:
            handle.write(f"{'fractional' if partial else 'metallic'} "
                         f"filling (SCF mesh); E_F = {fermi:.4f} eV "
                         "(absolute)\n")
        elif cbm is None:
            handle.write("no empty states in the basis\n")
        else:
            handle.write(f"VBM {vbm:.4f} eV, CBM {cbm:.4f} eV (absolute); "
                         f"mesh gap {cbm - vbm:.4f} eV\n")
    outputs.append(f"{stem}.txt")

    print("Band structure written to " + ", ".join(outputs))


def main(argv: list[str] | None = None) -> None:
    import argparse
    from pathlib import Path

    from .crystal_orbital_diagram import parse_oxidation_tokens

    parser = argparse.ArgumentParser(
        description="Electronic band structure / fatbands from the PySCF "
                    "density matrix (crystod --band --pyscf).")
    parser.add_argument("--poscar", default="POSCAR")
    parser.add_argument("--co-left", nargs="+", default=None,
                        metavar="FORMULA",
                        help="fragment split, only to match an existing "
                        "--chk (default: first element vs the rest)")
    parser.add_argument("--co-right", nargs="+", default=None,
                        metavar="FORMULA")
    parser.add_argument("--fatband", action="store_true",
                        help="element- and (element, l)-projected fatbands")
    parser.add_argument("--band-points", type=int, default=41,
                        help="k points per path leg (default 41)")
    parser.add_argument("--projection", choices=("lowdin", "mulliken"),
                        default="lowdin")
    parser.add_argument("--align", choices=("vbm", "absolute"),
                        default="vbm")
    parser.add_argument("--window", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"),
                        help="energy window of the plots in eV")
    parser.add_argument("--electrons", type=float, default=None)
    parser.add_argument("--oxidation", nargs="+", default=None,
                        metavar="EL=Q")
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
                        help="run (or reuse) only the crystal SCF; also "
                        "accepts the crystal-only chk files an --onsite "
                        "diagram run writes")
    parser.add_argument("--output", default=None,
                        help="output stem (default BAND_<structure>)")
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
        fatband=args.fatband,
        band_points=args.band_points,
        projection=args.projection,
        align=args.align,
        output_path=args.output,
        structure_label=stem,
        window=tuple(args.window) if args.window else None,
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
