#!/usr/bin/env python3

__author__ = "Yasuhide Mochizuki"
__copyright__ = "Copyright 2026, Tokyo Univ of Sci, Mochizuki group"
__version__ = "2.2"
__maintainer__ = "Yasuhide Mochizuki"
__email__ = "mochizuki@rs.tus.ac.jp"
__status__ = "Development"
__date__ = "May 18th, 2026"

import argparse
import numpy as np
from pyscf import gto, scf, dft, mp, cc
from pyscf.hessian import thermo

HARTREE_TO_EV = 27.211386

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

COMMON_XC_CHOICES = [
    "lda,vwn",
    "pbe",
    "pbe0",
    "b3lyp",
    "blyp",
    "bp86",
    "m06",
    "m06-2x",
    "wb97x",
    "wb97x-d",
]


def read_xyz(xyz_file):
    with open(xyz_file, 'r') as f:
        lines = f.readlines()

    natoms = int(lines[0])
    atom_lines = lines[2:2 + natoms]

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
    elif method == "uhf":
        return scf.UHF(mol)
    elif method == "rohf":
        return scf.ROHF(mol)
    else:
        raise ValueError("Unknown method: choose rhf, uhf, or rohf")


def build_ks(mol, spin, xc):
    if spin == 0:
        mf = dft.RKS(mol)
    else:
        mf = dft.UKS(mol)
    mf.xc = xc
    return mf


def run_scf(atom_str, args, method):
    mol = gto.M(
        atom=atom_str,
        basis=args.basis,
        charge=args.charge,
        spin=args.spin,
        unit=args.unit
    )

    mf = build_mf(mol, method)
    mf.verbose = 0
    mf.kernel()

    return mf


def run_main_calculation(atom_str, args, method):
    mol = gto.M(
        atom=atom_str,
        basis=args.basis,
        charge=args.charge,
        spin=args.spin,
        unit=args.unit
    )

    if args.theory == "dft":
        mf = build_ks(mol, args.spin, args.xc)
        mf.verbose = 0
        mf.kernel()
        return mf, mf.e_tot, {"label": f"DFT ({args.xc})"}

    if args.theory == "scf":
        mf = build_mf(mol, method)
        mf.verbose = 0
        mf.kernel()
        return mf, mf.e_tot, {"label": method.upper()}

    # post-HF methods use SCF reference
    mf = build_mf(mol, method)
    mf.verbose = 0
    mf.kernel()

    if args.theory == "mp2":
        mp2_obj = mp.MP2(mf)
        mp2_obj.verbose = 0
        mp2_obj.kernel()
        info = {
            "label": f"{method.upper()}-MP2",
            "e_ref": mf.e_tot,
            "e_corr": mp2_obj.e_corr,
        }
        return mf, mp2_obj.e_tot, info

    if args.theory == "ccsd":
        cc_obj = cc.CCSD(mf)
        cc_obj.verbose = 0
        cc_obj.kernel()
        e_tot = mf.e_tot + cc_obj.e_corr
        info = {
            "label": f"{method.upper()}-CCSD",
            "e_ref": mf.e_tot,
            "e_corr_ccsd": cc_obj.e_corr,
        }
        return mf, e_tot, info

    if args.theory == "ccsd_t":
        cc_obj = cc.CCSD(mf)
        cc_obj.verbose = 0
        cc_obj.kernel()
        et = cc_obj.ccsd_t()
        e_tot = mf.e_tot + cc_obj.e_corr + et
        info = {
            "label": f"{method.upper()}-CCSD(T)",
            "e_ref": mf.e_tot,
            "e_corr_ccsd": cc_obj.e_corr,
            "e_corr_t": et,
        }
        return mf, e_tot, info

    raise RuntimeError(f"Unknown theory: {args.theory}")


def run_mom_excited_scf(atom_str, args, method):
    if method != "uhf":
        raise RuntimeError("MOM excited-state SCF is currently supported with method=uhf.")

    # First, converge the ground/reference solution
    mol = gto.M(
        atom=atom_str,
        basis=args.basis,
        charge=args.charge,
        spin=args.spin,
        unit=args.unit
    )
    ref_mf = build_mf(mol, method)
    ref_mf.verbose = 0
    ref_mf.kernel()

    mo0 = ref_mf.mo_coeff
    occ0 = np.array(ref_mf.mo_occ, copy=True)
    target_occ = np.array(occ0, copy=True)

    spin_idx = 0 if args.promote_spin == "alpha" else 1
    spin_label = "Alpha" if spin_idx == 0 else "Beta"

    occ_spin = target_occ[spin_idx]
    occ_idx = np.where(occ_spin > 0.5)[0]
    vir_idx = np.where(occ_spin < 0.5)[0]
    if len(occ_idx) == 0 or len(vir_idx) == 0:
        raise RuntimeError(f"Cannot determine occupied/virtual orbitals for {spin_label}.")

    promote_from = args.promote_from if args.promote_from is not None else int(occ_idx[-1])
    promote_to = args.promote_to if args.promote_to is not None else int(vir_idx[0])

    nmo = target_occ.shape[1]
    if not (0 <= promote_from < nmo and 0 <= promote_to < nmo):
        raise RuntimeError(
            f"Promotion indices out of range: nmo={nmo}, "
            f"promote_from={promote_from}, promote_to={promote_to}"
        )
    if target_occ[spin_idx, promote_from] < 1.0:
        raise RuntimeError(
            f"{spin_label} MO {promote_from + 1} is not occupied in reference state."
        )
    if target_occ[spin_idx, promote_to] > 0.0:
        raise RuntimeError(
            f"{spin_label} MO {promote_to + 1} is already occupied in reference state."
        )

    # Build a single-electron promotion target occupation
    target_occ[spin_idx, promote_from] -= 1.0
    target_occ[spin_idx, promote_to] += 1.0

    mf = build_mf(mol, method)
    mf.verbose = 0
    mf = scf.addons.mom_occ(mf, mo0, target_occ)
    mf.kernel(dm0=ref_mf.make_rdm1())

    return mf, promote_from, promote_to, spin_label


def rhf_energy_decomposition(mf, dm):
    hcore = mf.get_hcore()
    j, k = mf.get_jk(dm=dm)
    e_one = np.einsum("ij,ji->", hcore, dm).real
    e_u = 0.5 * np.einsum("ij,ji->", j, dm).real
    e_j = -0.25 * np.einsum("ij,ji->", k, dm).real
    e_nuc = mf.energy_nuc()
    e_tot = e_one + e_u + e_j + e_nuc
    return {
        "e_one": e_one,
        "e_u": e_u,
        "e_j": e_j,
        "e_nuc": e_nuc,
        "e_tot": e_tot,
    }


def uhf_energy_decomposition(mf, dma, dmb):
    hcore = mf.get_hcore()
    dm_tot = dma + dmb
    j = mf.get_j(dm=dm_tot)
    ka, kb = mf.get_k(dm=(dma, dmb))

    e_one = (np.einsum("ij,ji->", hcore, dma).real +
             np.einsum("ij,ji->", hcore, dmb).real)
    e_u = 0.5 * np.einsum("ij,ji->", j, dm_tot).real
    e_j = -0.5 * (
        np.einsum("ij,ji->", ka, dma).real +
        np.einsum("ij,ji->", kb, dmb).real
    )
    e_nuc = mf.energy_nuc()
    e_tot = e_one + e_u + e_j + e_nuc
    return {
        "e_one": e_one,
        "e_u": e_u,
        "e_j": e_j,
        "e_nuc": e_nuc,
        "e_tot": e_tot,
    }


def decompose_total_energy(mf):
    hcore = mf.get_hcore()
    t_ao = mf.mol.intor_symmetric("int1e_kin")
    vne_ao = hcore - t_ao
    enuc = mf.energy_nuc()

    if isinstance(mf, scf.uhf.UHF):
        dma, dmb = mf.make_rdm1()
        dm_tot = dma + dmb
        e_kin = np.einsum("ij,ji->", t_ao, dm_tot).real
        e_vne = np.einsum("ij,ji->", vne_ao, dm_tot).real
        j = mf.get_j(dm=dm_tot)
        ka, kb = mf.get_k(dm=(dma, dmb))
        e_u = 0.5 * np.einsum("ij,ji->", j, dm_tot).real
        e_j = -0.5 * (
            np.einsum("ij,ji->", ka, dma).real +
            np.einsum("ij,ji->", kb, dmb).real
        )
    else:
        dm = mf.make_rdm1()
        e_kin = np.einsum("ij,ji->", t_ao, dm).real
        e_vne = np.einsum("ij,ji->", vne_ao, dm).real
        j, k = mf.get_jk(dm=dm)
        e_u = 0.5 * np.einsum("ij,ji->", j, dm).real
        e_j = -0.25 * np.einsum("ij,ji->", k, dm).real

    etot = e_kin + e_vne + e_u + e_j + enuc
    return {
        "e_kin": e_kin,
        "e_vne": e_vne,
        "e_u": e_u,
        "e_j": e_j,
        "e_nuc": enuc,
        "e_tot": etot,
    }


def print_total_energy_decomposition(mf):
    d = decompose_total_energy(mf)
    print("\nTotal-energy decomposition:")
    print(f"  Kinetic (T)       : {d['e_kin'] * HARTREE_TO_EV:.6f} eV")
    print(f"  Nuc-elec (V_ne)   : {d['e_vne'] * HARTREE_TO_EV:.6f} eV")
    print(f"  Hartree (U)       : {d['e_u'] * HARTREE_TO_EV:.6f} eV")
    print(f"  Exchange (J)      : {d['e_j'] * HARTREE_TO_EV:.6f} eV")
    print(f"  Nuc-nuc (V_nn)    : {d['e_nuc'] * HARTREE_TO_EV:.6f} eV")
    print(f"  Sum               : {d['e_tot'] * HARTREE_TO_EV:.6f} eV")


def print_fixed_occ_decomposition(atom_str, args):
    if args.method not in ("auto", "rhf", "uhf"):
        raise RuntimeError("--fixed-occ-decomp supports RHF/UHF only.")

    # auto choice follows current script behavior
    if args.method == "auto":
        method = "rhf" if args.spin == 0 else "uhf"
    else:
        method = args.method

    if method == "rhf" and args.spin != 0:
        raise RuntimeError("RHF fixed-occ decomposition requires --spin 0. Use --method uhf for open-shell.")
    if method == "uhf" and args.spin == 0:
        raise RuntimeError("UHF fixed-occ decomposition is for open-shell cases. Use --method rhf for closed-shell.")

    mol = gto.M(
        atom=atom_str,
        basis=args.basis,
        charge=args.charge,
        spin=args.spin,
        unit=args.unit
    )
    mf = build_mf(mol, method)
    mf.verbose = 0
    mf.kernel()

    if method == "rhf":
        occ_ref = np.array(mf.mo_occ, copy=True)
        occ_idx = np.where(occ_ref > 1.0)[0]
        vir_idx = np.where(occ_ref < 1.0)[0]
        if len(occ_idx) == 0 or len(vir_idx) == 0:
            raise RuntimeError("Cannot determine occupied/virtual orbitals for RHF reference.")

        promote_from = args.promote_from if args.promote_from is not None else int(occ_idx[-1])
        promote_to = args.promote_to if args.promote_to is not None else int(vir_idx[0])

        nmo = occ_ref.shape[0]
        if not (0 <= promote_from < nmo and 0 <= promote_to < nmo):
            raise RuntimeError(
                f"Promotion indices out of range: nmo={nmo}, "
                f"promote_from={promote_from}, promote_to={promote_to}"
            )
        if occ_ref[promote_from] < 1.0:
            raise RuntimeError(f"MO {promote_from + 1} is not occupied in reference RHF.")
        if occ_ref[promote_to] > 1.0:
            raise RuntimeError(f"MO {promote_to + 1} is already occupied in reference RHF.")

        occ_exc = np.array(occ_ref, copy=True)
        occ_exc[promote_from] -= 1.0
        occ_exc[promote_to] += 1.0

        dm_ref = mf.make_rdm1(mf.mo_coeff, occ_ref)
        dm_exc = mf.make_rdm1(mf.mo_coeff, occ_exc)

        d_ref = rhf_energy_decomposition(mf, dm_ref)
        d_exc = rhf_energy_decomposition(mf, dm_exc)
        promo_text = f"MO {promote_from + 1} -> MO {promote_to + 1} (fixed orbitals)"
        model_text = "RHF"
    else:
        spin_idx = 0 if args.promote_spin == "alpha" else 1
        spin_label = "Alpha" if spin_idx == 0 else "Beta"
        occ_ref = np.array(mf.mo_occ, copy=True)
        occ_spin = occ_ref[spin_idx]
        occ_idx = np.where(occ_spin > 0.5)[0]
        vir_idx = np.where(occ_spin < 0.5)[0]
        if len(occ_idx) == 0 or len(vir_idx) == 0:
            raise RuntimeError(f"Cannot determine occupied/virtual orbitals for {spin_label}.")

        promote_from = args.promote_from if args.promote_from is not None else int(occ_idx[-1])
        promote_to = args.promote_to if args.promote_to is not None else int(vir_idx[0])
        nmo = occ_spin.shape[0]
        if not (0 <= promote_from < nmo and 0 <= promote_to < nmo):
            raise RuntimeError(
                f"Promotion indices out of range: nmo={nmo}, "
                f"promote_from={promote_from}, promote_to={promote_to}"
            )
        if occ_ref[spin_idx, promote_from] < 0.5:
            raise RuntimeError(f"{spin_label} MO {promote_from + 1} is not occupied in reference UHF.")
        if occ_ref[spin_idx, promote_to] > 0.5:
            raise RuntimeError(f"{spin_label} MO {promote_to + 1} is already occupied in reference UHF.")

        occ_exc = np.array(occ_ref, copy=True)
        occ_exc[spin_idx, promote_from] -= 1.0
        occ_exc[spin_idx, promote_to] += 1.0

        dma_ref, dmb_ref = mf.make_rdm1(mf.mo_coeff, occ_ref)
        dma_exc, dmb_exc = mf.make_rdm1(mf.mo_coeff, occ_exc)
        d_ref = uhf_energy_decomposition(mf, dma_ref, dmb_ref)
        d_exc = uhf_energy_decomposition(mf, dma_exc, dmb_exc)
        promo_text = f"{spin_label} MO {promote_from + 1} -> MO {promote_to + 1} (fixed orbitals)"
        model_text = "UHF"

    print("------------------------------------------")
    print(f"Fixed-orbital Occupation Decomposition ({model_text})")
    if method == "rhf":
        print("Reference     : closed-shell SCF occupancy")
    else:
        print("Reference     : open-shell SCF occupancy")
    print(f"Promotion     : {promo_text}")
    for key, label in (
        ("e_one", "One-electron"),
        ("e_u", "Hartree (U)"),
        ("e_j", "Exchange (J)"),
        ("e_nuc", "Nuclear rep."),
        ("e_tot", "Total"),
    ):
        v0 = d_ref[key] * HARTREE_TO_EV
        v1 = d_exc[key] * HARTREE_TO_EV
        dv = (d_exc[key] - d_ref[key]) * HARTREE_TO_EV
        print(f"{label:13s}: {v0:12.6f} -> {v1:12.6f} eV   (Delta = {dv:10.6f} eV)")


def print_results(mf, e_tot, method_label, info=None):
    print("------------------------------------------")
    print(f"Method        : {method_label}")
    if info:
        if "e_ref" in info:
            print(f"Reference E   : {info['e_ref'] * HARTREE_TO_EV:.6f} eV")
        if "e_corr" in info:
            print(f"MP2 corr E    : {info['e_corr'] * HARTREE_TO_EV:.6f} eV")
        if "e_corr_ccsd" in info:
            print(f"CCSD corr E   : {info['e_corr_ccsd'] * HARTREE_TO_EV:.6f} eV")
        if "e_corr_t" in info:
            print(f"(T) corr E    : {info['e_corr_t'] * HARTREE_TO_EV:.6f} eV")
    print(f"Total Energy  : {e_tot * HARTREE_TO_EV:.6f} eV")

    # <S^2>
    if hasattr(mf, "spin_square"):
        s2, mult = mf.spin_square()
        print(f"<S^2>          : {s2:.6f}")
        print(f"Multiplicity  : {mult:.1f}")

    # MO energies
    print("\nMO energies (Hartree / eV):")
    if isinstance(mf.mo_energy, np.ndarray):
        if mf.mo_energy.ndim == 1: # RHF/ROHF case
            for i, e in enumerate(mf.mo_energy):
                print(f"  MO {i+1:2d}: {e:10.6f}  {e*HARTREE_TO_EV:10.4f}")
        elif mf.mo_energy.ndim == 2: # UHF case where mo_energy is a 2D array
            print("  Alpha MO energies:")
            for i, e in enumerate(mf.mo_energy[0]): # Iterate over the first row (alpha)
                print(f"  MO {i+1:2d}: {e:10.6f}  {e*HARTREE_TO_EV:10.4f}")
            print("  Beta MO energies:")
            for i, e in enumerate(mf.mo_energy[1]): # Iterate over the second row (beta)
                print(f"  MO {i+1:2d}: {e:10.6f}  {e*HARTREE_TO_EV:10.4f}")
        else:
            print("  Unexpected mf.mo_energy dimensions.")
    else: # Fallback for other types, though unlikely with current PySCF versions for SCF
        print("  Unexpected mf.mo_energy type. Cannot print MO energies.")


def print_zpe(mf):
    print("\nZPE analysis:")
    if mf.mol.natm < 2:
        print("  Skipped: ZPE requires at least 2 atoms.")
        return

    hess = mf.Hessian().kernel()
    vib = thermo.harmonic_analysis(mf.mol, hess)
    freqs = np.array(vib["freq_wavenumber"], dtype=float)
    real_freqs = freqs[np.isfinite(freqs) & (freqs > 1e-6)]
    if real_freqs.size == 0:
        print("  No positive real vibrational frequencies found.")
        return

    freq_au = np.array(vib["freq_au"]).real
    freq_au = freq_au[freq_au > 1e-8]
    zpe_h = 0.5 * np.sum(freq_au)
    print(f"  Number of modes : {real_freqs.size}")
    print(f"  ZPE             : {zpe_h * HARTREE_TO_EV:.6f} eV")


def pes_scan(atoms, args, method):
    if len(atoms) != 2:
        raise RuntimeError("PES scan is implemented only for diatomic molecules.")

    elem1, _, _, _ = atoms[0]
    elem2, _, _, _ = atoms[1]

    R_list = np.linspace(args.rmin, args.rmax, args.npts)

    print("\nPES scan:")
    print("R (Bohr)    E_SCF (Hartree)    E_SCF (eV)")
    print("------------------------------------------------")

    for R in R_list:
        atom_str = f"{elem1} 0 0 0; {elem2} 0 0 {R}"
        mf = run_scf(atom_str, args, method)
        E = mf.e_tot
        print(f"{R:7.3f}     {E:14.8f}     {E*HARTREE_TO_EV:10.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="PySCF total energy / MO / <S^2> / PES calculator"
    )

    parser.add_argument("--xyz", required=True, help="Input XYZ file")
    parser.add_argument("--basis", default="sto-3g", type=str.lower,
                        choices=COMMON_BASIS_CHOICES,
                        help="Basis set")
    parser.add_argument("--method", default="auto",
                        choices=["auto", "rhf", "uhf", "rohf"],
                        help="SCF method")
    parser.add_argument("--theory", default="scf",
                        choices=["scf", "dft", "mp2", "ccsd", "ccsd_t"],
                        help="Electronic-structure level")
    parser.add_argument("--xc", default="b3lyp", type=str.lower,
                        choices=COMMON_XC_CHOICES,
                        help="XC functional for DFT")
    parser.add_argument("--spin", type=int, default=0, help="Spin = 2S")
    parser.add_argument("--charge", type=int, default=0, help="Total charge")
    parser.add_argument("--unit", default="Angstrom",
                        choices=["Angstrom", "Bohr"],
                        help="XYZ coordinate unit")

    # PES options
    parser.add_argument("--pes", action="store_true",
                        help="Perform PES scan (diatomic only)")
    parser.add_argument("--rmin", type=float, default=0.5, help="Minimum R (Bohr)")
    parser.add_argument("--rmax", type=float, default=6.0, help="Maximum R (Bohr)")
    parser.add_argument("--npts", type=int, default=25, help="Number of points")
    parser.add_argument("--mom", action="store_true",
                        help="Use MOM-based excited-state Delta-SCF (UHF only)")
    parser.add_argument("--promote-from", type=int, default=None,
                        help="0-based MO index to depopulate for MOM")
    parser.add_argument("--promote-to", type=int, default=None,
                        help="0-based MO index to populate for MOM")
    parser.add_argument("--promote-spin", default="alpha",
                        choices=["alpha", "beta"],
                        help="Spin channel for MOM promotion")
    parser.add_argument("--zpe", action="store_true",
                        help="Compute harmonic ZPE from SCF/DFT Hessian")
    parser.add_argument("--fixed-occ-decomp", action="store_true",
                        help="Decompose RHF energy terms for fixed-orbital occupation change")
    parser.add_argument("--decompose-total-energy", action="store_true",
                        help="Print decomposition of SCF total energy into T, V_ne, J, K, V_nn")

    args = parser.parse_args()

    atoms = read_xyz(args.xyz)

    # automatic method choice
    if args.method == "auto":
        method = "rhf" if args.spin == 0 else "uhf"
    else:
        method = args.method

    atom_str = atoms_to_pyscf_string(atoms)

    print("==========================================")
    print(" PySCF Calculation")
    print("==========================================")
    print(f"XYZ file      : {args.xyz}")
    print(f"Basis         : {args.basis}")
    print(f"Charge        : {args.charge}")
    print(f"Spin (2S)     : {args.spin}")
    print(f"Theory        : {args.theory}")
    if args.theory == "dft":
        print(f"XC functional : {args.xc}")
    if args.mom:
        print("Excited-state : MOM Delta-SCF enabled")
        if args.promote_from is not None and args.promote_to is not None:
            print(
                f"Promotion     : {args.promote_spin} MO {args.promote_from + 1} "
                f"-> MO {args.promote_to + 1}"
            )
        else:
            print(f"Promotion     : {args.promote_spin} HOMO -> LUMO (auto)")

    if args.pes:
        if args.theory != "scf":
            raise RuntimeError("--pes is currently supported only with --theory scf.")
        pes_scan(atoms, args, method)
    elif args.fixed_occ_decomp:
        if args.theory != "scf":
            raise RuntimeError("--fixed-occ-decomp supports --theory scf only.")
        print_fixed_occ_decomposition(atom_str, args)
    else:
        if args.mom:
            if args.theory != "scf":
                raise RuntimeError("--mom is currently supported only with --theory scf.")
            mf, p_from, p_to, p_spin = run_mom_excited_scf(atom_str, args, method)
            print(f"MOM used      : {p_spin} MO {p_from + 1} -> MO {p_to + 1}")
            e_tot = mf.e_tot
            method_label = method.upper()
            info = None
        else:
            mf, e_tot, info = run_main_calculation(atom_str, args, method)
            method_label = info["label"]
        print_results(mf, e_tot, method_label, info)
        if args.decompose_total_energy:
            if args.theory != "scf":
                print("\nNote          : decomposition uses SCF reference density.")
            print_total_energy_decomposition(mf)
        if args.zpe:
            if args.theory in ("mp2", "ccsd_t"):
                print("\nNote          : ZPE is evaluated from SCF reference Hessian.")
            print_zpe(mf)

    print("==========================================")


if __name__ == "__main__":
    main()
