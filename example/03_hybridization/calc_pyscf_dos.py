#!/usr/bin/env python3

__author__ = "Yasuhide Mochizuki"
__copyright__ = "Copyright 2026, Tokyo Univ of Sci, Mochizuki group"
__version__ = "2.0"
__maintainer__ = "Yasuhide Mochizuki"
__email__ = "mochizuki@rs.tus.ac.jp"
__status__ = "Development"
__date__ = "May 14th, 2026"

import argparse
import io
import os
import tempfile

import numpy as np
from pyscf import gto, scf, dft, mp, cc

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "matplotlib"),
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HARTREE_TO_EV = 27.211386
ANGULAR_CHANNELS = ("s", "p", "d", "f")
VESTA_FALLBACK_COLORS = {
    "H": "#ffcccc",
    "C": "#4c4c4c",
    "N": "#b0b9e6",
    "O": "#fe0300",
    "F": "#b0b9e6",
    "P": "#c09cc2",
    "S": "#fffa00",
    "Cl": "#31fc02",
    "Br": "#7e3102",
    "I": "#940094",
}

COMMON_BASIS_CHOICES = [
    "sto-3g",
    "3-21g",
    "6-31g",
    "6-31g*",
    "6-31g**",
    "6-31+g",
    "6-31+g*",
    "6-31+g**",
    "6-311g",
    "6-311g*",
    "6-311g**",
    "6-311+g",
    "6-311+g*",
    "6-311+g**",
    "cc-pvdz",
    "cc-pvtz",
    "cc-pvqz",
    "aug-cc-pvdz",
    "aug-cc-pvtz",
    "aug-cc-pvqz",
    "def2-svp",
    "def2-tzvp",
    "def2-tzvpp",
    "def2-qzvp",
]
THEORY_ALIASES = {"scf", "dft", "mp2", "ccsd", "ccsd_t"}


def read_xyz(xyz_file):
    with open(xyz_file, "r") as f:
        lines = f.readlines()

    natoms = int(lines[0])
    atom_lines = lines[2 : 2 + natoms]

    atoms = []
    for line in atom_lines:
        elem, x, y, z = line.split()
        atoms.append((elem, float(x), float(y), float(z)))

    return atoms


def atoms_to_pyscf_string(atoms):
    return "; ".join([f"{e} {x} {y} {z}" for e, x, y, z in atoms])


def build_mf(mol, method):
    method = method.lower()
    if method == "rhf":
        return scf.RHF(mol)
    if method == "uhf":
        return scf.UHF(mol)
    if method == "rohf":
        return scf.ROHF(mol)
    raise ValueError("Unknown method: choose rhf, uhf, or rohf")


def run_main(atom_str, args, method):
    mol = gto.M(
        atom=atom_str,
        basis=args.basis,
        charge=args.charge,
        spin=args.spin,
        unit=args.unit,
    )

    if args.theory == "dft":
        mf = dft.RKS(mol) if args.spin == 0 else dft.UKS(mol)
        mf.xc = args.xc
    else:
        mf = build_mf(mol, method)
    mf.verbose = 0
    mf.kernel()

    if not mf.converged:
        print("WARNING: SCF did not converge. DOS will be generated from the last cycle.")

    e_tot = mf.e_tot
    if args.theory == "mp2":
        post = mp.MP2(mf)
        post.verbose = 0
        post.kernel()
        e_tot = post.e_tot
    elif args.theory == "ccsd":
        post = cc.CCSD(mf)
        post.verbose = 0
        post.kernel()
        e_tot = mf.e_tot + post.e_corr
    elif args.theory == "ccsd_t":
        post = cc.CCSD(mf)
        post.verbose = 0
        post.kernel()
        e_tot = mf.e_tot + post.e_corr + post.ccsd_t()

    return mf, e_tot


def split_mo_channels(mf):
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
    mo_occ = getattr(mf, "mo_occ", None)

    if isinstance(mo_coeff, (tuple, list)):
        labels = ("alpha", "beta")
        return [
            {
                "label": labels[i],
                "energy": np.asarray(mo_energy[i]),
                "coeff": np.asarray(mo_coeff[i]),
                "occ": None if mo_occ is None else np.asarray(mo_occ[i]),
                "state_factor": 1.0,
            }
            for i in range(len(mo_coeff))
        ]

    mo_energy = np.asarray(mo_energy)
    mo_coeff = np.asarray(mo_coeff)

    if mo_energy.ndim == 2 and mo_coeff.ndim == 3 and mo_energy.shape[0] == 2:
        labels = ("alpha", "beta")
        return [
            {
                "label": labels[i],
                "energy": mo_energy[i],
                "coeff": mo_coeff[i],
                "occ": None if mo_occ is None else np.asarray(mo_occ[i]),
                "state_factor": 1.0,
            }
            for i in range(2)
        ]

    return [
        {
            "label": "restricted",
            "energy": mo_energy,
            "coeff": mo_coeff,
            "occ": None if mo_occ is None else np.asarray(mo_occ),
            "state_factor": 2.0,
        }
    ]


def ao_angular_groups(mol):
    groups = {channel: [] for channel in ANGULAR_CHANNELS}

    for ao_index, label in enumerate(mol.ao_labels(fmt=False)):
        # label is usually (atom_id, element, "2p", "x"), etc.
        shell_name = label[2]
        angular = next((char for char in shell_name if char in ANGULAR_CHANNELS), None)
        if angular is not None:
            groups[angular].append(ao_index)

    return {channel: np.asarray(indices, dtype=int) for channel, indices in groups.items()}


def ao_element_groups(mol):
    groups = {}

    for ao_index, label in enumerate(mol.ao_labels(fmt=False)):
        atom_id = label[0]
        element = mol.atom_symbol(atom_id)
        groups.setdefault(element, []).append(ao_index)

    return {
        element: np.asarray(indices, dtype=int)
        for element, indices in sorted(groups.items())
    }


def ao_element_angular_groups(mol):
    groups = {}

    for ao_index, label in enumerate(mol.ao_labels(fmt=False)):
        atom_id = label[0]
        element = mol.atom_symbol(atom_id)
        shell_name = label[2]
        angular = next((char for char in shell_name if char in ANGULAR_CHANNELS), None)
        if angular is None:
            continue
        groups.setdefault((element, angular), []).append(ao_index)

    angular_order = {angular: i for i, angular in enumerate(ANGULAR_CHANNELS)}
    ordered_keys = sorted(groups, key=lambda key: (key[0], angular_order.get(key[1], len(ANGULAR_CHANNELS))))
    return {key: np.asarray(groups[key], dtype=int) for key in ordered_keys}


def lowdin_projector(mol):
    overlap = mol.intor_symmetric("int1e_ovlp")
    eigval, eigvec = np.linalg.eigh(overlap)
    eigval = np.clip(eigval, 0.0, None)
    return (eigvec * np.sqrt(eigval)) @ eigvec.T


def mo_angular_weights(mol, coeff, groups, projection):
    if projection == "lowdin":
        projected_coeff = lowdin_projector(mol) @ coeff
        return {
            channel: np.sum(np.abs(projected_coeff[indices, :]) ** 2, axis=0)
            if len(indices) > 0
            else np.zeros(coeff.shape[1])
            for channel, indices in groups.items()
        }

    overlap_coeff = mol.intor_symmetric("int1e_ovlp") @ coeff
    return {
        channel: np.einsum(
            "mi,mi->i",
            np.conjugate(coeff[indices, :]),
            overlap_coeff[indices, :],
        ).real
        if len(indices) > 0
        else np.zeros(coeff.shape[1])
        for channel, indices in groups.items()
    }


def mo_group_weights(mol, coeff, groups, projection):
    if projection == "lowdin":
        projected_coeff = lowdin_projector(mol) @ coeff
        return {
            group: np.sum(np.abs(projected_coeff[indices, :]) ** 2, axis=0)
            if len(indices) > 0
            else np.zeros(coeff.shape[1])
            for group, indices in groups.items()
        }

    overlap_coeff = mol.intor_symmetric("int1e_ovlp") @ coeff
    return {
        group: np.einsum(
            "mi,mi->i",
            np.conjugate(coeff[indices, :]),
            overlap_coeff[indices, :],
        ).real
        if len(indices) > 0
        else np.zeros(coeff.shape[1])
        for group, indices in groups.items()
    }


def vesta_element_colors(elements):
    try:
        from pymatgen.vis.structure_vtk import EL_COLORS

        color_table = EL_COLORS.get("VESTA", {})
        colors = {}
        for element in elements:
            rgb = color_table.get(element)
            if rgb is None:
                colors[element] = None
            else:
                colors[element] = tuple(channel / 255.0 for channel in rgb)
        if any(color is not None for color in colors.values()):
            fallback_cmap = plt.get_cmap("tab20")
            for i, element in enumerate(elements):
                if colors[element] is None:
                    colors[element] = fallback_cmap(i % fallback_cmap.N)
            return colors
    except Exception:
        pass

    fallback_cmap = plt.get_cmap("tab20")
    return {
        element: VESTA_FALLBACK_COLORS.get(element, fallback_cmap(i % fallback_cmap.N))
        for i, element in enumerate(elements)
    }


def gaussian(x, center, sigma):
    return np.exp(-0.5 * ((x - center) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def histogram_edges(emin, emax, binwidth):
    n_bins = max(1, int(np.ceil((emax - emin) / binwidth - 1.0e-12)))
    edges = emin + np.arange(n_bins + 1) * binwidth
    edges[-1] = emax
    return edges


def homo_energy_ev(channels):
    occupied_energies = []

    for channel in channels:
        if channel["occ"] is None:
            continue
        occupied = channel["energy"][channel["occ"] > 1.0e-8]
        if occupied.size:
            occupied_energies.append(np.max(occupied))

    if not occupied_energies:
        return 0.0

    return max(occupied_energies) * HARTREE_TO_EV


def build_dos(mf, args):
    channels = split_mo_channels(mf)
    groups = ao_angular_groups(mf.mol)
    element_groups = ao_element_angular_groups(mf.mol) if args.element_pdos else ao_element_groups(mf.mol)
    element_labels = list(element_groups)

    reference_ev = homo_energy_ev(channels) if args.align == "homo" else 0.0
    all_energies = np.concatenate([channel["energy"] for channel in channels])
    all_energies_ev = all_energies * HARTREE_TO_EV - reference_ev

    use_histogram = args.binwidth is not None
    padding = args.padding if use_histogram else max(args.padding, 6.0 * args.sigma)
    emin = np.min(all_energies_ev) - padding if args.emin is None else args.emin
    emax = np.max(all_energies_ev) + padding if args.emax is None else args.emax

    if use_histogram:
        bin_edges = histogram_edges(emin, emax, args.binwidth)
        bin_widths = np.diff(bin_edges)
        grid = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    else:
        bin_edges = None
        bin_widths = None
        grid = np.linspace(emin, emax, args.npts)

    dos = np.zeros_like(grid)
    pdos = {channel: np.zeros_like(grid) for channel in ANGULAR_CHANNELS}
    element_pdos = {element: np.zeros_like(grid) for element in element_labels}
    spin_dos = {channel["label"]: np.zeros_like(grid) for channel in channels}
    spin_pdos = {
        channel["label"]: {angular: np.zeros_like(grid) for angular in ANGULAR_CHANNELS}
        for channel in channels
    }
    spin_element_pdos = {
        channel["label"]: {element: np.zeros_like(grid) for element in element_labels}
        for channel in channels
    }
    mo_rows = []

    for channel in channels:
        spin_label = channel["label"]
        weights = mo_angular_weights(mf.mol, channel["coeff"], groups, args.projection)
        element_weights = mo_group_weights(mf.mol, channel["coeff"], element_groups, args.projection)
        state_factor = channel["state_factor"] if args.spin_degeneracy else 1.0
        energies_ev = channel["energy"] * HARTREE_TO_EV - reference_ev

        if use_histogram:
            total_weights = np.full_like(energies_ev, state_factor, dtype=float)
            hist, _ = np.histogram(energies_ev, bins=bin_edges, weights=total_weights)
            channel_dos = hist / bin_widths
            dos += channel_dos
            spin_dos[spin_label] += channel_dos

            for angular in ANGULAR_CHANNELS:
                hist, _ = np.histogram(
                    energies_ev,
                    bins=bin_edges,
                    weights=state_factor * weights[angular],
                )
                channel_pdos = hist / bin_widths
                pdos[angular] += channel_pdos
                spin_pdos[spin_label][angular] += channel_pdos

            for element in element_labels:
                hist, _ = np.histogram(
                    energies_ev,
                    bins=bin_edges,
                    weights=state_factor * element_weights[element],
                )
                channel_pdos = hist / bin_widths
                element_pdos[element] += channel_pdos
                spin_element_pdos[spin_label][element] += channel_pdos

        for mo_index, energy_ev in enumerate(energies_ev):
            if not use_histogram:
                broadening = gaussian(grid, energy_ev, args.sigma)
                channel_dos = state_factor * broadening
                dos += channel_dos
                spin_dos[spin_label] += channel_dos
                for angular in ANGULAR_CHANNELS:
                    channel_pdos = state_factor * weights[angular][mo_index] * broadening
                    pdos[angular] += channel_pdos
                    spin_pdos[spin_label][angular] += channel_pdos
                for element in element_labels:
                    channel_pdos = state_factor * element_weights[element][mo_index] * broadening
                    element_pdos[element] += channel_pdos
                    spin_element_pdos[spin_label][element] += channel_pdos

            occ = None if channel["occ"] is None else channel["occ"][mo_index]
            mo_rows.append(
                {
                    "spin": channel["label"],
                    "mo": mo_index + 1,
                    "energy_ev": energy_ev,
                    "occ": occ,
                    "weights": {angular: weights[angular][mo_index] for angular in ANGULAR_CHANNELS},
                }
            )

    return (
        grid,
        dos,
        pdos,
        element_pdos,
        spin_dos,
        spin_pdos,
        spin_element_pdos,
        mo_rows,
        reference_ev,
        bin_widths,
        element_labels,
    )


def plot_dos(
    grid,
    dos,
    pdos,
    element_pdos,
    spin_dos,
    spin_pdos,
    spin_element_pdos,
    element_labels,
    args,
    title,
    bin_widths,
    method,
):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    colors = {
        "s": "#1f77b4",
        "p": "#d62728",
        "d": "#2ca02c",
        "f": "#9467bd",
    }
    linestyles = {
        "s": "-",
        "p": "--",
        "d": "-.",
        "f": ":",
    }
    if args.element_pdos:
        unique_elements = []
        for element, _angular in element_labels:
            if element not in unique_elements:
                unique_elements.append(element)
    else:
        unique_elements = list(element_labels)
    element_colors = vesta_element_colors(unique_elements)
    spin_resolved = method == "uhf" and {"alpha", "beta"}.issubset(spin_dos)

    if spin_resolved:
        spin_signs = {"alpha": 1.0, "beta": -1.0}
        spin_names = {"alpha": "alpha", "beta": "beta"}

        for spin_label in ("alpha", "beta"):
            signed_dos = spin_signs[spin_label] * spin_dos[spin_label]
            if args.binwidth is None:
                ax.plot(
                    grid,
                    signed_dos,
                    color="black",
                    lw=1.8,
                    ls="-" if spin_label == "alpha" else "--",
                    label=f"{spin_names[spin_label]} total",
                )
            else:
                ax.bar(
                    grid,
                    signed_dos,
                    width=0.92 * bin_widths,
                    align="center",
                    color="0.82" if spin_label == "alpha" else "0.72",
                    edgecolor="black",
                    linewidth=0.6,
                    label=f"{spin_names[spin_label]} total",
                )

            if args.element_pdos:
                for element, angular in element_labels:
                    signed_pdos = spin_signs[spin_label] * spin_element_pdos[spin_label][(element, angular)]
                    if np.max(np.abs(signed_pdos)) <= 1.0e-10:
                        continue
                    if args.binwidth is None:
                        ax.plot(
                            grid,
                            signed_pdos,
                            color=element_colors.get(element),
                            ls=linestyles[angular],
                            lw=1.2,
                            alpha=0.8 if spin_label == "alpha" else 0.55,
                            label=f"{spin_names[spin_label]} {element} {angular}-PDOS",
                        )
                    else:
                        ax.step(
                            grid,
                            signed_pdos,
                            where="mid",
                            color=element_colors.get(element),
                            ls=linestyles[angular],
                            lw=1.2,
                            alpha=0.8 if spin_label == "alpha" else 0.55,
                            label=f"{spin_names[spin_label]} {element} {angular}-PDOS",
                        )
            else:
                for angular in ANGULAR_CHANNELS:
                    signed_pdos = spin_signs[spin_label] * spin_pdos[spin_label][angular]
                    if np.max(np.abs(signed_pdos)) <= 1.0e-10:
                        continue
                    if args.binwidth is None:
                        ax.plot(
                            grid,
                            signed_pdos,
                            color=colors[angular],
                            ls=linestyles[angular],
                            lw=1.2,
                            alpha=1.0 if spin_label == "alpha" else 0.72,
                            label=f"{spin_names[spin_label]} {angular}-PDOS",
                        )
                    else:
                        ax.step(
                            grid,
                            signed_pdos,
                            where="mid",
                            color=colors[angular],
                            ls=linestyles[angular],
                            lw=1.2,
                            alpha=1.0 if spin_label == "alpha" else 0.72,
                            label=f"{spin_names[spin_label]} {angular}-PDOS",
                        )
        ax.axhline(0.0, color="0.55", lw=0.8)
    elif args.binwidth is None:
        ax.plot(grid, dos, color="black", lw=1.8, label="total")
    else:
        ax.bar(
            grid,
            dos,
            width=0.92 * bin_widths,
            align="center",
            color="0.82",
            edgecolor="black",
            linewidth=0.6,
            label="total",
        )

    if not spin_resolved and not args.element_pdos:
        for angular in ANGULAR_CHANNELS:
            if np.max(np.abs(pdos[angular])) > 1.0e-10:
                if args.binwidth is None:
                    ax.plot(
                        grid,
                        pdos[angular],
                        color=colors[angular],
                        ls=linestyles[angular],
                        lw=1.4,
                        label=f"{angular}-PDOS",
                    )
                else:
                    ax.step(
                        grid,
                        pdos[angular],
                        where="mid",
                        color=colors[angular],
                        ls=linestyles[angular],
                        lw=1.4,
                        label=f"{angular}-PDOS",
                    )

    if not spin_resolved and args.element_pdos:
        for element, angular in element_labels:
            if np.max(np.abs(element_pdos[(element, angular)])) > 1.0e-10:
                if args.binwidth is None:
                    ax.plot(
                        grid,
                        element_pdos[(element, angular)],
                        color=element_colors.get(element),
                        ls=linestyles[angular],
                        lw=1.4,
                        alpha=0.85,
                        label=f"{element} {angular}-PDOS",
                    )
                else:
                    ax.step(
                        grid,
                        element_pdos[(element, angular)],
                        where="mid",
                        color=element_colors.get(element),
                        ls=linestyles[angular],
                        lw=1.4,
                        alpha=0.85,
                        label=f"{element} {angular}-PDOS",
                    )

    if args.align == "homo":
        ax.axvline(0.0, color="0.55", lw=0.9, ls=":")
        ax.set_xlabel("Energy - HOMO (eV)")
    else:
        ax.set_xlabel("Orbital energy (eV)")

    if args.xrange is not None:
        ax.set_xlim(args.xrange)
    if args.yrange is not None:
        ax.set_ylim(args.yrange)

    if spin_resolved:
        ax.set_ylabel("DOS (states/eV; beta plotted negative)")
    else:
        ax.set_ylabel("DOS (states/eV)")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi)
    plt.close(fig)


def write_csv(
    grid,
    dos,
    pdos,
    element_pdos,
    spin_dos,
    spin_pdos,
    spin_element_pdos,
    element_labels,
    csv_file,
    method,
    include_element_pdos,
):
    columns = [grid, dos]
    headers = ["energy_eV", "total_DOS"]

    if not include_element_pdos:
        columns.extend(pdos[channel] for channel in ANGULAR_CHANNELS)
        headers.extend(["s_PDOS", "p_PDOS", "d_PDOS", "f_PDOS"])

    if include_element_pdos:
        for element_label in element_labels:
            element, angular = element_label
            columns.append(element_pdos[(element, angular)])
            headers.append(f"{element}_{angular}_PDOS")

    if method == "uhf" and {"alpha", "beta"}.issubset(spin_dos):
        for spin_label in ("alpha", "beta"):
            columns.append(spin_dos[spin_label])
            headers.append(f"{spin_label}_DOS")
            if include_element_pdos:
                for element_label in element_labels:
                    element, angular = element_label
                    columns.append(spin_element_pdos[spin_label][(element, angular)])
                    headers.append(f"{spin_label}_{element}_{angular}_PDOS")
            else:
                for angular in ANGULAR_CHANNELS:
                    columns.append(spin_pdos[spin_label][angular])
                    headers.append(f"{spin_label}_{angular}_PDOS")

    data = np.column_stack(columns)
    header = ",".join(headers)
    np.savetxt(csv_file, data, delimiter=",", header=header, comments="")


def print_mo_table(mo_rows, align):
    energy_header = "E-HOMO(eV)" if align == "homo" else "energy(eV)"
    print("\nMO projected weights:")
    print(f"spin         MO    {energy_header:>10s}    occ        s        p        d        f")
    print("--------------------------------------------------------------------------")

    for row in mo_rows:
        occ_text = "NA" if row["occ"] is None else f"{row['occ']:.3g}"
        print(
            f"{row['spin']:<10s} {row['mo']:3d} "
            f"{row['energy_ev']:12.5f} {occ_text:>6s} "
            f"{row['weights']['s']:8.4f} "
            f"{row['weights']['p']:8.4f} "
            f"{row['weights']['d']:8.4f} "
            f"{row['weights']['f']:8.4f}"
        )


def default_csv_name(output_name):
    root, _ = os.path.splitext(output_name)
    return root + ".csv"


def default_output_name(xyz_file):
    root, _ = os.path.splitext(os.path.basename(xyz_file))
    return f"DOS_{root}.pdf"


def default_txt_name(output_name):
    root, _ = os.path.splitext(output_name)
    return root + ".txt"


def build_summary_text(args, method, mo_rows, reference_ev, csv_file, txt_file, element_labels, total_energy_ev):
    out = io.StringIO()

    print("==========================================", file=out)
    print(" PySCF DOS-like Calculation", file=out)
    print("==========================================", file=out)
    print(f"XYZ file      : {args.xyz}", file=out)
    print(f"Method        : {method.upper()}", file=out)
    print(f"Theory        : {args.theory}", file=out)
    if args.theory == "dft":
        print(f"XC functional : {args.xc}", file=out)
    print(f"Basis         : {args.basis}", file=out)
    print(f"Total Energy  : {total_energy_ev:.6f} eV", file=out)
    print(f"Charge        : {args.charge}", file=out)
    print(f"Spin (2S)     : {args.spin}", file=out)
    print(f"Projection    : {args.projection}", file=out)
    if args.binwidth is None:
        print("DOS mode      : gaussian broadening", file=out)
        print(f"Gaussian sigma: {args.sigma:.3f} eV", file=out)
    else:
        print("DOS mode      : histogram", file=out)
        print(f"Bin width     : {args.binwidth:.3f} eV", file=out)
    if args.align == "homo":
        print(f"Energy zero   : HOMO = {reference_ev:.6f} eV", file=out)
    else:
        print("Energy zero   : absolute MO energy", file=out)
    print(f"Plot output   : {args.output}", file=out)
    print(f"CSV output    : {csv_file}", file=out)
    print(f"Text output   : {txt_file}", file=out)
    if args.element_pdos:
        labels = [f"{element}({angular})" for element, angular in element_labels]
        print(f"Element PDOS  : {', '.join(labels)}", file=out)

    if not args.quiet:
        energy_header = "E-HOMO(eV)" if args.align == "homo" else "energy(eV)"
        print("\nMO projected weights:", file=out)
        print(f"spin         MO    {energy_header:>10s}    occ        s        p        d        f", file=out)
        print("--------------------------------------------------------------------------", file=out)

        for row in mo_rows:
            occ_text = "NA" if row["occ"] is None else f"{row['occ']:.3g}"
            print(
                f"{row['spin']:<10s} {row['mo']:3d} "
                f"{row['energy_ev']:12.5f} {occ_text:>6s} "
                f"{row['weights']['s']:8.4f} "
                f"{row['weights']['p']:8.4f} "
                f"{row['weights']['d']:8.4f} "
                f"{row['weights']['f']:8.4f}",
                file=out,
            )

    print("==========================================", file=out)
    return out.getvalue()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a molecule DOS-like plot by broadening PySCF MO energies "
            "and projecting each MO onto AO angular momentum channels."
        )
    )

    parser.add_argument("--xyz", required=True, help="Input XYZ file")
    parser.add_argument(
        "--basis",
        default="sto-3g",
        type=str.lower,
        help="Basis set",
    )
    parser.add_argument(
        "--theory",
        default="scf",
        choices=["scf", "dft", "mp2", "ccsd", "ccsd_t"],
        help="Electronic-structure level",
    )
    parser.add_argument(
        "--xc",
        default="b3lyp",
        type=str.lower,
        help="XC functional for DFT",
    )
    parser.add_argument(
        "--method",
        default="auto",
        choices=["auto", "rhf", "uhf", "rohf"],
        help="SCF method",
    )
    parser.add_argument("--spin", type=int, default=0, help="Spin = 2S")
    parser.add_argument("--charge", type=int, default=0, help="Total charge")
    parser.add_argument(
        "--unit",
        default="Angstrom",
        choices=["Angstrom", "Bohr"],
        help="XYZ coordinate unit",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.3,
        help="Gaussian broadening width in eV",
    )
    parser.add_argument(
        "--binwidth",
        "--bin-width",
        dest="binwidth",
        type=float,
        default=None,
        help="Use histogram DOS with this bin width in eV instead of Gaussian broadening",
    )
    parser.add_argument("--npts", type=int, default=2000, help="Number of energy grid points")
    parser.add_argument("--padding", type=float, default=3.0, help="Energy padding in eV")
    parser.add_argument("--emin", type=float, default=None, help="Minimum plotted energy in eV")
    parser.add_argument("--emax", type=float, default=None, help="Maximum plotted energy in eV")
    parser.add_argument(
        "--xrange",
        type=float,
        nargs=2,
        metavar=("XMIN", "XMAX"),
        default=None,
        help="X-axis range in eV, e.g. --xrange -10 10",
    )
    parser.add_argument(
        "--yrange",
        type=float,
        nargs=2,
        metavar=("YMIN", "YMAX"),
        default=None,
        help="Y-axis range in states/eV, e.g. --yrange 0 2",
    )
    parser.add_argument(
        "--align",
        default="absolute",
        choices=["homo", "absolute"],
        help="Energy reference",
    )
    parser.add_argument(
        "--projection",
        default="lowdin",
        choices=["lowdin", "mulliken"],
        help="AO population analysis used for PDOS weights",
    )
    parser.add_argument(
        "--no-spin-degeneracy",
        dest="spin_degeneracy",
        action="store_false",
        help="Do not count restricted MOs as doubly spin-degenerate states",
    )
    parser.set_defaults(spin_degeneracy=True)
    parser.add_argument("--output", default=None, help="Output plot file")
    parser.add_argument("--csv", default=None, help="Output CSV file")
    parser.add_argument("--txt", default=None, help="Output text summary file")
    parser.add_argument("--dpi", type=int, default=200, help="Raster output resolution")
    parser.add_argument(
        "--element-pdos",
        action="store_true",
        help="Plot and export element-and-angular-momentum resolved PDOS using VESTA element colors",
    )
    parser.add_argument("--quiet", action="store_true", help="Do not print the MO weight table")

    args = parser.parse_args()
    if args.basis in THEORY_ALIASES:
        if args.theory == "scf":
            args.theory = args.basis
        args.basis = "cc-pvdz"
        print(f"Note          : interpreted '--basis {args.theory}' as theory; basis set set to cc-pvdz")
    elif args.basis not in COMMON_BASIS_CHOICES:
        parser.error("Unknown basis set")

    if args.output is None:
        args.output = default_output_name(args.xyz)

    if args.binwidth is not None and args.binwidth <= 0.0:
        parser.error("--binwidth must be positive")

    if args.xrange is not None:
        if args.emin is not None or args.emax is not None:
            parser.error("--xrange cannot be used together with --emin or --emax")
        args.emin, args.emax = args.xrange

    if args.emin is not None and args.emax is not None and args.emin >= args.emax:
        parser.error("energy range must satisfy minimum < maximum")

    if args.yrange is not None and args.yrange[0] >= args.yrange[1]:
        parser.error("--yrange must satisfy YMIN < YMAX")

    atoms = read_xyz(args.xyz)

    if args.method == "auto":
        method = "rhf" if args.spin == 0 else "uhf"
    else:
        method = args.method

    atom_str = atoms_to_pyscf_string(atoms)
    mf, e_tot = run_main(atom_str, args, method)

    (
        grid,
        dos,
        pdos,
        element_pdos,
        spin_dos,
        spin_pdos,
        spin_element_pdos,
        mo_rows,
        reference_ev,
        bin_widths,
        element_labels,
    ) = build_dos(mf, args)
    csv_file = default_csv_name(args.output) if args.csv is None else args.csv
    txt_file = default_txt_name(args.output) if args.txt is None else args.txt

    title = f"{os.path.basename(args.xyz)}  {args.theory.upper()}/{args.basis}"
    plot_dos(
        grid,
        dos,
        pdos,
        element_pdos,
        spin_dos,
        spin_pdos,
        spin_element_pdos,
        element_labels,
        args,
        title,
        bin_widths,
        method,
    )
    write_csv(
        grid,
        dos,
        pdos,
        element_pdos,
        spin_dos,
        spin_pdos,
        spin_element_pdos,
        element_labels,
        csv_file,
        method,
        args.element_pdos,
    )
    summary_text = build_summary_text(
        args, method, mo_rows, reference_ev, csv_file, txt_file, element_labels, e_tot * HARTREE_TO_EV
    )
    with open(txt_file, "w") as f:
        f.write(summary_text)
    print(summary_text, end="")


if __name__ == "__main__":
    main()
