#!/usr/bin/env python3
r"""Solve a periodic solid with molecular-chemistry machinery, and time it.

Read this first
---------------
``pyscf_band_structure.py`` is **already** a localized-basis, frozen-core
calculation.  ``gth-dzvp-molopt-sr`` is a contracted *Gaussian* basis in the
same sense as 6-31G -- two functions per valence shell plus polarization,
centred on the atoms -- and ``gth-pbe`` freezes the core exactly like an
effective core potential.  Nothing is ever expanded in plane waves, and the
variational parameters are the AO coefficients c_{mu n}(k), not Fourier
components.  That combination (norm-conserving pseudopotential + localized
atomic orbitals) *is* the SIESTA model.

What the FFT grid in that script is
-----------------------------------
``--ke-cutoff`` does not set a basis.  It sets the uniform real-space grid on
which the Hartree potential and the exchange-correlation potential are
evaluated -- the same object SIESTA calls ``MeshCutoff``.  A molecular code
avoids a uniform grid in two ways, and this script lets you switch both:

``--coulomb``   how the Hartree term is built
                ``fft``   uniform grid + FFT Poisson solve      (SIESTA / default)
                ``gdf``   Gaussian density fitting, analytic    (molecular)
                ``rsdf``  range-separated Gaussian fitting      (molecular)
``--xc-grid``   where the exchange-correlation energy is integrated
                ``uniform`` the same uniform grid
                ``becke``   atom-centred Becke grid             (molecular)

and ``--pseudo none`` with a molecular basis (``6-31g``, ``def2-svp``, ...)
makes it a genuine all-electron calculation.

Measured on ScF3, PBE, 2x2x2 k-mesh = 4 irreducible k-points
------------------------------------------------------------
============================ ========= ========== ========= ========
route                          t(SCF)   t/band-k   gap (eV)  quad.pts
============================ ========= ========== ========= ========
gth-szv   fft ke=100 uniform      36 s     0.67 s     4.601     50653
gth-dzvp  fft ke=100 uniform      80 s     1.30 s     5.673     50653
gth-dzvp  fft ke=100 becke        93 s     1.45 s     5.685     66028
gth-dzvp  fft ke=200 becke       182 s     2.89 s     5.839     66028
gth-dzvp  fft ke=200 uniform     225 s     3.46 s     5.840    132651
gth-dzvp  gdf        becke     did not converge in 400 s at 1x1x1
sto-3g AE gdf        becke     did not converge in 780 s at 2x2x2
============================ ========= ========== ========= ========

The last two rows are the point of this script, and the result is the opposite
of the intuition.  Gaussian density fitting did not finish even a *single*
k-point in the time the FFT route needs for a converged 2x2x2 calculation, with
libcint reporting near-singular Rys roots throughout.  The analytic two-electron
integrals have to be summed over lattice images, and for a solid that costs far
more than one FFT on a 37^3 grid; all-electron core Gaussians make both the
fitting integrals and the grid worse still.  The uniform FFT grid is not the
price of a plane-wave basis -- it is what makes the localized-basis solid cheap.

What actually buys speed, in order of effect: fewer k-points, a lower
``--ke-cutoff``, a smaller basis, and ``--xc-grid becke`` once the cutoff is
high (it reproduced the ke=200 gap to 0.001 eV with half the grid points, for
about 80% of the time).  Note also how much of the gap is grid-converged:
dropping ke=200 to ke=100 costs 0.17 eV, and dropping DZVP to SZV costs 1.24 eV.

Examples
--------
Reproduce the table above on your own structure::

    python pyscf_band_localized.py 221_PPOSCAR_ScF3 --benchmark

Try the molecular Coulomb route yourself -- expect to wait, or to give up::

    python pyscf_band_localized.py 221_PPOSCAR_ScF3 --recipe molecular \
        --kmesh 1 1 1 --path "GAMMA R" --npoints 2

All-electron, no pseudopotential at all (slower again)::

    python pyscf_band_localized.py 221_PPOSCAR_ScF3 --recipe all-electron \
        --kmesh 1 1 1 --path "GAMMA R" --npoints 2

The fast localized-basis route, which is what the benchmark recommends::

    python pyscf_band_localized.py 221_PPOSCAR_ScF3 --recipe balanced --kmesh 4 4 4

Outputs are the same as ``pyscf_band_structure.py``: ``<prefix>.pdf/.dat/.json``,
plus ``--fatband`` / ``--dos`` / ``--save-mo``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# The POSCAR reader, k-path construction, plotting and analysis are shared with
# the VASP-style driver in this directory; only the integration strategy differs.
from pyscf_band_structure import (
    HARTREE2EV,
    ao_indices_by_element,
    band_edges,
    compute_bands,
    element_color,
    group_path,
    interpolate_path,
    merge_ticks,
    mulliken_weights,
    parse_vasp_kpoints,
    path_distances,
    plot_bands,
    plot_dos,
    read_poscar,
    seekpath_reference,
    write_dat,
)

BOHR = 0.52917721092

# name -> (basis, pseudo, ke_cutoff, coulomb, xc_grid)
RECIPES = {
    # the SIESTA model: norm-conserving pseudopotential, localized orbitals,
    # uniform mesh only for the potentials
    "siesta": ("gth-dzvp-molopt-sr", "gth-pbe", 200.0, "fft", "uniform"),
    # same, but the exchange-correlation part done the molecular way; this
    # reproduced the siesta gap to 0.001 eV at ~80% of the cost
    "balanced": ("gth-dzvp-molopt-sr", "gth-pbe", 200.0, "fft", "becke"),
    # single-zeta and a coarse mesh: seconds per k-point, dispersions still
    # qualitatively right, gaps are not
    "fast": ("gth-szv-molopt-sr", "gth-pbe", 100.0, "fft", "uniform"),
    # no uniform grid anywhere: analytic Gaussian Coulomb + atom-centred grid
    "molecular": ("gth-dzvp-molopt-sr", "gth-pbe", 100.0, "gdf", "becke"),
    # and with the core electrons put back in; the cutoff only caps the auxiliary
    # grid the density fitting still needs, it is not a basis cutoff
    "all-electron": ("6-31g", "none", 400.0, "gdf", "becke"),
}


def build_cell(lattice, symbols, frac, args):
    from pyscf.pbc import gto

    cell = gto.Cell()
    cell.a = lattice
    cell.atom = [(symbol, tuple(position)) for symbol, position in zip(symbols, frac @ lattice)]
    cell.unit = "Angstrom"
    cell.basis = args.basis
    all_electron = args.pseudo.lower() in {"none", "off", "false"}
    if not all_electron:
        cell.pseudo = args.pseudo
    # A uniform grid has to resolve the 1s cusp, which needs a cutoff a hundred
    # times higher than a frozen-core valence density does. Refuse rather than
    # silently return numbers from an under-converged grid.
    if all_electron and args.coulomb == "fft" and (args.ke_cutoff or 0) < 2000.0:
        raise SystemExit(
            "ERROR: an all-electron cell (--pseudo none) with --coulomb fft needs a uniform\n"
            "       grid fine enough for the core density -- several thousand Hartree, which\n"
            "       is not practical. Use --coulomb gdf (or rsdf), or keep a pseudopotential.\n"
            "       Pass --ke-cutoff 2000 or more if you really want to try."
        )
    if args.ke_cutoff:
        cell.ke_cutoff = args.ke_cutoff
    cell.precision = args.precision
    cell.max_memory = args.max_memory
    cell.verbose = args.verbose
    if args.symmetry:
        cell.space_group_symmetry = True
        cell.symmorphic = False
    cell.build()
    return cell


def build_mean_field(cell, kpts, args):
    """KRKS/KRHF with the requested Coulomb scheme and exchange-correlation grid."""
    from pyscf.pbc import dft, df, scf
    from pyscf.pbc.dft import gen_grid

    if args.xc.lower() in {"hf", "hartree-fock"}:
        mf = scf.KRHF(cell, kpts)
        needs_exchange = True
    else:
        mf = dft.KRKS(cell, kpts)
        mf.xc = args.xc
        needs_exchange = abs(mf._numint.hybrid_coeff(mf.xc)) > 1e-10

    coulomb = args.coulomb
    if needs_exchange and coulomb == "fft":
        print(
            f"NOTE: '{args.xc}' needs exact exchange, which FFT density fitting handles\n"
            "      poorly; switching to --coulomb gdf."
        )
        coulomb = "gdf"

    if coulomb == "gdf":
        mf = mf.density_fit()
    elif coulomb == "rsdf":
        mf.with_df = df.RSDF(cell, np.asarray(getattr(kpts, "kpts", kpts)))

    if args.xc_grid == "becke" and hasattr(mf, "grids"):
        mf.grids = gen_grid.BeckeGrids(cell)
        mf.grids.level = args.grid_level

    mf.conv_tol = args.conv_tol
    mf.max_cycle = args.max_cycle
    mf.max_memory = args.max_memory
    mf.init_guess = args.init_guess
    if args.sigma > 0:
        from pyscf.pbc.scf import addons

        mf = addons.smearing_(mf, sigma=args.sigma / HARTREE2EV, method=args.smearing)
    return mf, coulomb


def run_scf(cell, args):
    """Build the k-mesh, converge, and return (mf_full_bz, info)."""
    kmesh = np.asarray(args.kmesh, dtype=int)
    if np.any(kmesh < 1):
        raise SystemExit("ERROR: every --kmesh entry must be at least 1")
    if args.symmetry:
        kpts = cell.make_kpts(kmesh, space_group_symmetry=True, time_reversal_symmetry=True)
        n_irreducible, n_full = kpts.nkpts_ibz, kpts.nkpts
    else:
        kpts = cell.make_kpts(kmesh)
        n_irreducible = n_full = len(kpts)

    mf, coulomb = build_mean_field(cell, kpts, args)
    started = time.time()
    energy = mf.kernel()
    elapsed = time.time() - started
    info = dict(
        kmesh=kmesh,
        n_irreducible=n_irreducible,
        n_full=n_full,
        coulomb=coulomb,
        energy=float(energy),
        converged=bool(mf.converged),
        t_scf=elapsed,
        n_grid=len(mf.grids.coords) if hasattr(mf, "grids") else 0,
    )
    if not mf.converged:
        return None, info
    if args.symmetry:
        mf = mf.to_khf()
    return mf, info


# ---------------------------------------------------------------------------
# benchmark mode
# ---------------------------------------------------------------------------
BENCH_KPTS = np.array(
    [[0.0, 0, 0], [0, 0.25, 0], [0, 0.5, 0], [0.25, 0.5, 0], [0.5, 0.5, 0], [0.5, 0.5, 0.5]]
)


def benchmark(lattice, symbols, frac, args) -> None:
    """Time every requested recipe on this structure and print one table."""
    # 'molecular' and 'all-electron' are not in the default set: on ScF3 neither
    # finished a 2x2x2 SCF in a quarter of an hour. Ask for them explicitly.
    names = args.benchmark_recipes or ["fast", "siesta", "balanced"]
    print(
        f"\nBenchmark on {args.poscar}: {args.benchmark_kmesh[0]}x{args.benchmark_kmesh[1]}"
        f"x{args.benchmark_kmesh[2]} k-mesh, {args.xc.upper()}, "
        f"{len(BENCH_KPTS)} band k-points\n"
    )
    header = (
        f"{'recipe':13s} {'basis':22s} {'coul':5s} {'xcgrid':7s} "
        f"{'nAO':>4s} {'ngrid':>7s} {'t_SCF/s':>9s} {'t_band/k':>9s} {'gap/eV':>8s}"
    )
    print(header)
    print("-" * len(header))
    rows = []
    for name in names:
        if name not in RECIPES:
            raise SystemExit(f"ERROR: unknown recipe '{name}'; choose from {', '.join(RECIPES)}")
        basis, pseudo, ke_cutoff, coulomb, xc_grid = RECIPES[name]
        case = argparse.Namespace(**vars(args))
        case.basis, case.pseudo, case.ke_cutoff = basis, pseudo, ke_cutoff
        case.coulomb, case.xc_grid = coulomb, xc_grid
        case.kmesh = args.benchmark_kmesh
        try:
            cell = build_cell(lattice, symbols, frac, case)
            mf, info = run_scf(cell, case)
            if mf is None:
                raise RuntimeError(f"SCF did not converge in {info['t_scf']:.0f} s")
            started = time.time()
            energies, _ = compute_bands(
                mf, cell, cell.get_abs_kpts(BENCH_KPTS), mf.make_rdm1(), 0, want_coeff=False
            )
            t_band = (time.time() - started) / len(BENCH_KPTS)
            nocc = cell.nelectron // 2
            gap = float(energies[:, nocc].min() - energies[:, nocc - 1].max()) * HARTREE2EV
            row = dict(
                recipe=name, basis=basis, coulomb=coulomb, xc_grid=xc_grid,
                nao=int(cell.nao_nr()), ngrid=int(info["n_grid"]),
                t_scf=info["t_scf"], t_band_per_k=t_band, gap=gap,
                energy=info["energy"], n_irreducible=int(info["n_irreducible"]),
            )
            print(
                f"{name:13s} {basis:22s} {coulomb:5s} {xc_grid:7s} "
                f"{row['nao']:4d} {row['ngrid']:7d} {row['t_scf']:9.1f} "
                f"{row['t_band_per_k']:9.2f} {row['gap']:8.3f}",
                flush=True,
            )
        except KeyboardInterrupt:
            print(f"{name:13s} interrupted by user", flush=True)
            row = dict(recipe=name, basis=basis, coulomb=coulomb, xc_grid=xc_grid,
                       error="interrupted")
        except Exception as exc:
            print(f"{name:13s} {basis:22s} {coulomb:5s} {xc_grid:7s}  -> {exc}", flush=True)
            row = dict(recipe=name, basis=basis, coulomb=coulomb, xc_grid=xc_grid,
                       error=str(exc))
        rows.append(row)

    done = [row for row in rows if "gap" in row]
    if done:
        best = min(done, key=lambda row: row["t_scf"])
        print(
            f"\nFastest converged recipe: '{best['recipe']}' "
            f"({best['t_scf']:.0f} s SCF, gap {best['gap']:.3f} eV).\n"
            "Compare its gap with the slowest one before trusting it: the cheap recipes\n"
            "save time on the grid and the basis, and that is exactly where gaps move."
        )
    output = Path(f"{args.output or 'benchmark_' + args.poscar.name}.json")
    output.write_text(json.dumps(rows, indent=1) + "\n")
    print(f"\n[output]\n    {output}")


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("poscar", type=Path, help="VASP POSCAR/PPOSCAR/CONTCAR of the unit cell")

    group = parser.add_argument_group("integration strategy")
    group.add_argument(
        "--recipe", choices=tuple(RECIPES), default=None,
        help="preset for --basis/--pseudo/--ke-cutoff/--coulomb/--xc-grid; "
        "any of those given explicitly still wins",
    )
    group.add_argument(
        "--coulomb", choices=("fft", "gdf", "rsdf"), default=None,
        help="Hartree term: uniform-grid FFT (SIESTA-like) or Gaussian density fitting (molecular)",
    )
    group.add_argument(
        "--xc-grid", choices=("uniform", "becke"), default=None,
        help="exchange-correlation quadrature: uniform mesh or atom-centred Becke grid",
    )
    group.add_argument("--grid-level", type=int, default=3, help="Becke grid level (1-9)")

    group = parser.add_argument_group("electronic structure")
    group.add_argument("--xc", default="pbe", help="exchange-correlation functional, or 'hf'")
    group.add_argument("--basis", default=None, help="Gaussian basis set")
    group.add_argument("--pseudo", default=None, help="GTH pseudopotential, or 'none' for all-electron")
    group.add_argument("--ke-cutoff", type=float, default=None, help="uniform-grid cutoff in Hartree")
    group.add_argument("--precision", type=float, default=1e-8, help="integral screening precision")
    group.add_argument("--conv-tol", type=float, default=1e-8, help="SCF convergence (Hartree)")
    group.add_argument("--max-cycle", type=int, default=100, help="maximum SCF iterations")
    group.add_argument(
        "--init-guess", default="hcore", choices=("hcore", "minao", "atom", "1e"),
        help="SCF initial guess; hcore is the only one safe for every GTH basis",
    )
    group.add_argument("--sigma", type=float, default=0.0, help="smearing width in eV (0 = none)")
    group.add_argument("--smearing", choices=("fermi", "gauss"), default="gauss")

    group = parser.add_argument_group("k-points")
    group.add_argument("--kmesh", type=int, nargs=3, default=(2, 2, 2), metavar=("N1", "N2", "N3"))
    group.add_argument("--no-symmetry", dest="symmetry", action="store_false")
    group.add_argument("--kpoints", type=Path, default=None, help="VASP KPOINTS supplying the band path")
    group.add_argument("--path", default=None, help='band path labels, e.g. "GAMMA X M GAMMA R"')
    group.add_argument("--npoints", type=int, default=11, help="k-points per leg")
    group.add_argument("--tolerance", type=float, default=1e-5, help="seekpath symmetry tolerance")
    group.add_argument("--band-chunk", type=int, default=0, help="band k-points per get_bands call")

    group = parser.add_argument_group("benchmark")
    group.add_argument(
        "--benchmark", action="store_true",
        help="time the recipes on this structure, write a JSON table, and stop",
    )
    group.add_argument(
        "--benchmark-recipes", nargs="+", default=None, metavar="NAME",
        help="which recipes to time (default: fast siesta balanced; "
        f"'molecular' and 'all-electron' are excluded because they take hours; all: {', '.join(RECIPES)})",
    )
    group.add_argument(
        "--benchmark-kmesh", type=int, nargs=3, default=(2, 2, 2), metavar=("N1", "N2", "N3"),
        help="k-mesh used for the benchmark",
    )

    group = parser.add_argument_group("output")
    group.add_argument("--output", type=Path, default=None, help="output prefix")
    group.add_argument("--emin", type=float, default=-8.0, help="lower edge of the plot window (eV)")
    group.add_argument("--emax", type=float, default=10.0, help="upper edge of the plot window (eV)")
    group.add_argument("--fatband", action="store_true", help="element-projected fatbands")
    group.add_argument("--dos", action="store_true", help="density of states from the SCF mesh")
    group.add_argument("--dos-sigma", type=float, default=0.15, help="DOS broadening in eV")
    group.add_argument("--save-mo", action="store_true", help="write <prefix>_mo.npz")
    group.add_argument("--max-memory", type=float, default=4000.0, help="PySCF memory limit in MB")
    group.add_argument("--verbose", type=int, default=3, help="PySCF verbosity")
    return parser


def resolve_recipe(args) -> None:
    """Fill in whatever the recipe covers and the user did not override."""
    recipe = RECIPES[args.recipe or "balanced"]
    defaults = dict(zip(("basis", "pseudo", "ke_cutoff", "coulomb", "xc_grid"), recipe))
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    resolve_recipe(args)
    if args.emin >= args.emax:
        raise SystemExit("ERROR: --emin must be smaller than --emax")

    lattice, symbols, frac = read_poscar(args.poscar)
    formula = "".join(f"{s}{symbols.count(s)}" for s in dict.fromkeys(symbols))

    if args.benchmark:
        benchmark(lattice, symbols, frac, args)
        return

    print("=" * 74)
    print(f"  Localized-basis periodic calculation : {formula}   ({args.poscar})")
    print("=" * 74)
    all_electron = args.pseudo.lower() in {"none", "off", "false"}
    print(f"  recipe     : {args.recipe or 'balanced (default)'}")
    print(f"  basis      : {args.basis}  ({'all-electron' if all_electron else args.pseudo})")
    print(f"  Coulomb    : {args.coulomb}   exchange-correlation grid : {args.xc_grid}")

    cell = build_cell(lattice, symbols, frac, args)
    print(f"  AOs        : {cell.nao_nr()}   electrons : {cell.nelectron}")
    if args.coulomb == "fft":
        print(f"  FFT mesh   : {list(map(int, cell.mesh))}  (ke_cutoff = {cell.ke_cutoff} Hartree)")
    if cell.nelectron % 2:
        raise SystemExit("ERROR: odd electron count; this restricted driver cannot handle it.")
    nocc = cell.nelectron // 2

    # ---------------- the k-path, resolved before anything expensive --------
    if args.kpoints is not None:
        kpts_band_frac, vertex_labels, breaks, kind = parse_vasp_kpoints(args.kpoints, args.npoints)
        print(f"\n[k-path] from {args.kpoints} ({kind})")
    else:
        point_coords, seek_path, primitive, sg_symbol, sg_number = seekpath_reference(
            lattice, symbols, frac, args.tolerance
        )
        print(f"\n[k-path] space group {sg_symbol} (#{sg_number})")
        if args.path:
            chains: list[list[str]] = [[]]
            for token in args.path.replace(",", " ").split():
                if token == "|":
                    chains.append([])
                    continue
                key = token if token in point_coords else token.upper()
                if key not in point_coords:
                    raise SystemExit(
                        f"ERROR: '{token}' is not a high-symmetry point of this cell.\n"
                        f"       available: {' '.join(sorted(point_coords))}"
                    )
                chains[-1].append(key)
            chains = [chain for chain in chains if len(chain) > 1]
            if not chains:
                raise SystemExit("ERROR: --path needs at least two labels per segment")
        else:
            chains = group_path(seek_path)
        kpts_band_frac, vertex_labels, breaks = interpolate_path(chains, point_coords, args.npoints)
        print(f"    {'  '.join('-'.join(chain) for chain in chains)}")

    recip = cell.reciprocal_vectors() / BOHR
    distances = path_distances(kpts_band_frac, recip, breaks)
    ticks, tick_labels = merge_ticks(vertex_labels, distances, breaks)
    breaks_distance = [float(distances[i]) for i in breaks if 0 < i < len(distances)]
    kpts_band_abs = cell.get_abs_kpts(kpts_band_frac)
    print(f"    {len(kpts_band_frac)} zero-weight k-points")
    if args.coulomb in {"gdf", "rsdf"}:
        print(
            "    WARNING: Gaussian density fitting has to build three-centre integrals for\n"
            "             every new k-point, so this step scales far worse than with\n"
            "             --coulomb fft. Keep --npoints small."
        )

    print(f"\n[step 1] SCF on a {'x'.join(map(str, args.kmesh))} Gamma-centred mesh")
    mf, info = run_scf(cell, args)
    print(f"    k-points   : {info['n_irreducible']} irreducible of {info['n_full']}")
    print(f"    quadrature : {info['n_grid']} points")
    print(f"    E_total    = {info['energy']:.10f} Hartree")
    print(f"    converged  = {info['converged']}   ({info['t_scf']:.1f} s)")
    if mf is None:
        raise SystemExit(
            "ERROR: the SCF did not converge. Try --sigma 0.2, a smaller --kmesh,\n"
            "       or a different --init-guess."
        )

    mesh_kpts_frac = cell.get_scaled_kpts(np.asarray(mf.kpts))
    mesh_energies = np.array([np.asarray(e, dtype=float) for e in mf.mo_energy])
    dm = mf.make_rdm1()

    # ---------------- step 2: the zero-weight k-points ---------------------
    print("\n[step 2] non-SCF diagonalisation of the converged Kohn-Sham Hamiltonian")
    started = time.time()
    path_energies, coefficients = compute_bands(
        mf, cell, kpts_band_abs, dm, args.band_chunk, want_coeff=args.fatband or args.save_mo
    )
    t_band = time.time() - started

    # ---------------- analysis and output ----------------------------------
    vbm, cbm, gap, direct_gap, k_vbm, cbm_info = band_edges(
        mesh_energies, path_energies, mesh_kpts_frac, kpts_band_frac, nocc
    )
    try:
        fermi = float(np.asarray(mf.get_fermi()).ravel()[0])
    except Exception:
        fermi = vbm
    print("\n[step 3] band edges")
    print(f"    VBM   = {vbm * HARTREE2EV: .6f} eV  at k = {np.round(k_vbm, 4).tolist()}")
    if cbm is None:
        gap = direct_gap = None
        print("    no unoccupied band available; enlarge the basis")
    else:
        print(f"    CBM   = {cbm * HARTREE2EV: .6f} eV  at k = {np.round(cbm_info[0], 4).tolist()}")
        print(f"    gap (indirect) = {gap * HARTREE2EV:.4f} eV")
        print(f"    gap (direct)   = {direct_gap * HARTREE2EV:.4f} eV")

    metallic = gap is None or gap * HARTREE2EV < 1e-3
    reference = fermi if metallic else vbm
    ylabel = r"$E - E_{\mathrm{F}}$  (eV)" if metallic else r"$E - E_{\mathrm{VBM}}$  (eV)"
    energies_ev = (path_energies - reference) * HARTREE2EV
    fermi_ev = (fermi - reference) * HARTREE2EV
    prefix = args.output or Path(f"localized_{args.poscar.name}")

    title = (
        f"{formula}   {args.xc.upper()}/{args.basis}   "
        f"{args.coulomb}+{args.xc_grid}   {'x'.join(map(str, args.kmesh))} mesh"
    )
    dat_path = Path(f"{prefix}.dat")
    write_dat(dat_path, distances, energies_ev, title, "E_Fermi" if metallic else "VBM")
    plot_path = Path(f"{prefix}.pdf")
    plot_bands(
        plot_path, distances, energies_ev, ticks, tick_labels, breaks_distance,
        title=title, energy_window=(args.emin, args.emax), ylabel=ylabel, fermi_ev=fermi_ev,
    )
    written = [dat_path, plot_path]

    weights = elements = None
    if args.fatband:
        elements, groups = ao_indices_by_element(cell, symbols)
        weights = mulliken_weights(cell, kpts_band_abs, coefficients, groups)
        for index, element in enumerate(elements):
            fat_path = Path(f"{prefix}_{element}.pdf")
            plot_bands(
                fat_path, distances, energies_ev, ticks, tick_labels, breaks_distance,
                title=f"{formula}   {element}-projected (Mulliken)",
                energy_window=(args.emin, args.emax), ylabel=ylabel, fermi_ev=fermi_ev,
                weights=weights[:, :, index], color=element_color(element),
                weight_label=f"{element} character",
            )
            written.append(fat_path)

    if args.save_mo:
        mo_path = Path(f"{prefix}_mo.npz")
        np.savez_compressed(
            mo_path,
            kpts_frac=kpts_band_frac, kpts_abs=kpts_band_abs, distance=distances,
            mo_energy_eV=path_energies * HARTREE2EV, mo_coeff=np.array(coefficients),
            overlap=np.array(cell.pbc_intor("int1e_ovlp", hermi=1, kpts=kpts_band_abs)),
            ao_labels=np.array(cell.ao_labels()), lattice_Angstrom=lattice,
            fractional_positions=frac, symbols=np.array(symbols), n_occupied=nocc,
        )
        written.append(mo_path)

    dos_data = None
    if args.dos:
        dos_path = Path(f"{prefix}_dos.pdf")
        grid, dos = plot_dos(
            dos_path,
            (mesh_energies - reference) * HARTREE2EV,
            np.full(len(mesh_energies), 1.0 / len(mesh_energies)),
            args.dos_sigma, (args.emin, args.emax),
            f"{formula}  DOS ({'x'.join(map(str, args.kmesh))} mesh)", ylabel,
        )
        dos_data = {"energy_eV": grid.tolist(), "dos": dos.tolist()}
        written.append(dos_path)

    summary = {
        "structure": str(args.poscar),
        "formula": formula,
        "recipe": args.recipe or "balanced",
        "method": {
            "xc": args.xc, "basis": args.basis, "pseudo": args.pseudo,
            "all_electron": all_electron, "coulomb": info["coulomb"],
            "xc_grid": args.xc_grid, "n_quadrature_points": info["n_grid"],
            "ke_cutoff_Hartree": cell.ke_cutoff, "nao": int(cell.nao_nr()),
        },
        "timing_seconds": {"scf": info["t_scf"], "bands": t_band,
                           "bands_per_kpoint": t_band / max(1, len(kpts_band_frac))},
        "scf": {"kmesh": list(map(int, args.kmesh)),
                "n_kpts_irreducible": info["n_irreducible"], "n_kpts_full": info["n_full"],
                "total_energy_Hartree": info["energy"], "converged": info["converged"],
                "nelectron": int(cell.nelectron), "n_occupied_bands": int(nocc),
                "mesh_kpts_frac": mesh_kpts_frac.tolist(),
                "mesh_eigenvalues_eV": (mesh_energies * HARTREE2EV).tolist()},
        "band": {"kpts_frac": kpts_band_frac.tolist(),
                 "distance_invAngstrom": distances.tolist(),
                 "energy_zero": "E_Fermi" if metallic else "VBM",
                 "eigenvalues_eV_shifted": energies_ev.tolist(),
                 "ticks": ticks, "tick_labels": tick_labels,
                 "discontinuities": breaks_distance},
        "edges": {"VBM_eV": vbm * HARTREE2EV,
                  "CBM_eV": None if cbm is None else cbm * HARTREE2EV,
                  "E_Fermi_eV": fermi * HARTREE2EV,
                  "gap_indirect_eV": None if gap is None else gap * HARTREE2EV,
                  "gap_direct_eV": None if direct_gap is None else direct_gap * HARTREE2EV,
                  "k_VBM": np.round(k_vbm, 6).tolist()},
    }
    if weights is not None:
        summary["band"]["mulliken_elements"] = elements
        summary["band"]["mulliken_weights"] = weights.tolist()
    if dos_data is not None:
        summary["dos"] = dos_data
    json_path = Path(f"{prefix}.json")
    json_path.write_text(json.dumps(summary, indent=1) + "\n")
    written.append(json_path)

    print(f"\n    SCF {info['t_scf']:.1f} s, bands {t_band:.1f} s "
          f"({t_band / max(1, len(kpts_band_frac)):.2f} s per k-point)")
    print("\n[output]")
    for item in written:
        print(f"    {item}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
