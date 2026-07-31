"""PySCF eigen-levels in the SALC viewer (crystod --visualize --pyscf).

The SALC viewer shows the symmetry-adapted *basis* -- the states before any
Hamiltonian.  This module puts the actual PySCF eigenstates into the same
page: one table row per degenerate partner with the level energy as a
fourth column, and the clicked row renders the eigenvector's wave function
(all elements, all shells s..f) on the k-commensurate display cell.

Three views of one compound share one calculation (and one --chk file):

* ``--sublattice Sc``  -- the Sc sublattice fragment: formal-charge ions +
  ghost basis of the removed sublattice + its point-charge Madelung field,
  exactly the left/right columns of ``crystod --diagram --pyscf``;
* ``--sublattice F3``  -- the other fragment;
* no ``--sublattice``  -- the full crystal (the states after bonding).

The k points are the automatic special points of the space group (one HTML
per k point; ``--kpoint GM`` restricts the output).  Energies are put on
the shared deep-level-aligned scale of the diagram, so the three pages are
directly comparable (``--no-align`` keeps each calculation's own G=0
reference).

Gauge note: PySCF Bloch AOs carry the phase on the lattice translation
alone (the atomic gauge), which is exactly the exp(2 pi i k . T) image
phase the viewer applies -- the eigenvectors are passed as-is, and the
drawn field is the physical Re[psi], as in the diagram sketches.
"""

from __future__ import annotations

import numpy as np

# lobes below this norm are dropped from the drawing: channels are scaled
# to sqrt(population), so 0.045 corresponds to a 0.2% Loewdin share
_CHANNEL_FLOOR = 0.045

# default energy window of the displayed column: HOMO - 15 eV to
# LUMO + 10 eV.  Anchoring both edges to the frontier levels keeps the
# window meaningful even for a huge-gap fragment spectrum (Sc^3+ has a
# 27 eV HOMO-LUMO gap, where a midpoint-centered window contains nothing),
# while still cutting the deep semicore and high diffuse states whose
# hundreds of plotly surfaces would bloat the page (--window overrides).
_WINDOW_BELOW_HOMO = 15.0
_WINDOW_ABOVE_LUMO = 10.0


def _level_partner_channels(diagram, level, diagonalize=False, semicore=None):
    """Per-partner {atom: [(l, coefficients), ...]} for one level.

    Same construction as the diagram sketches: signs and orientation from
    the same-l shells accumulated with their contracted-GTO radial
    weights, sizes from the selected population measure (--projection),
    with one calibration factor per (atom, l) channel from the multiplet
    sums so symmetry-equivalent atoms draw exactly equal.  The probe
    radius is chosen PER CHANNEL as the one of AOBlock.radial_profile
    where the accumulated amplitude is largest -- a single fixed radius
    can sit right on the orthogonalization node of a semicore-carrying
    channel (Sc s of the valence R1+ of ScF3 is -0.0013 at 2 bohr, so its
    drawn sign would be numerical noise).  Fragment levels draw only
    their own sublattice's components (the ghost weight is a variational
    tail, reported numerically elsewhere).  f shells are reordered from
    PySCF (m = -3..3) to the viewer convention.

    ``diagonalize`` canonicalizes the degenerate partners (RREF +
    orthogonalization, as in the diagram sketches): the arbitrary unitary
    mixture the SCF returns is rotated to the sparsest axis-aligned
    combinations (a tilted d_z2 becomes z-aligned); energies and the
    spanned space are unchanged.

    ``semicore`` (--valence-only) is a set of (element, shell) names whose
    contribution is dropped from the drawing: the <r> of a semicore shell
    is far inside the bond (gth Sc 3s: 0.77 A vs the 2.03 A Sc-F bond), so
    its admixture in a valence level is the on-site orthogonality tail
    against the deep semicore band, not bonding -- drawing it hides the
    chemistry (the valence R1+ sigma level of ScF3 looks "antibonding"
    although its Sc 4s component bonds).  A level DOMINATED by such a
    shell (the semicore band itself) keeps it.
    """
    from .crystal_orbital_pyscf import _reorder_to_pyscf
    from .visualize_basis import realify_basis_space

    rows, is_real = realify_basis_space(level.vectors.T)
    rows = np.asarray(rows)
    overlap = level.overlap
    sqrt_overlap = level.sqrt_overlap
    f_reorder = _reorder_to_pyscf(3)
    column = getattr(level, "column", "mo")
    blocks = [b for b in diagram.ao_blocks
              if (column == "mo" or b.column == column) and b.l <= 3]
    n_radii = max((len(b.radial_profile) for b in blocks), default=1) or 1

    raw = []   # per partner: {key: [channel per radius]}
    pop: dict[tuple[int, int], float] = {}
    amp2: dict[tuple[int, int], np.ndarray] = {}
    for vector in rows:
        v = np.asarray(vector, dtype=complex)
        if diagram.projection == "mulliken":
            gross = (v.conj() * (overlap @ v)).real
        else:
            gross = np.abs(sqrt_overlap @ v) ** 2
        channels: dict[tuple[int, int], list] = {}
        total_gross = float(gross.sum()) or 1.0
        for block in blocks:
            if semicore and (block.element, block.shell) in semicore:
                share = float(
                    gross[block.offset:block.offset + block.n_ao].sum())
                if share < 0.4 * total_gross:
                    continue  # orthogonality tail, not the semicore band
            coefficients = np.asarray(
                v[block.offset:block.offset + block.n_ao])
            if block.l == 3:
                coefficients = f_reorder.T @ coefficients
            key = (block.sites[0], block.l)
            profile = (block.radial_profile
                       or (block.radial,) * n_radii)
            entry = channels.setdefault(
                key, [np.zeros(block.n_ao, dtype=complex)
                      for _ in range(n_radii)])
            for radius_index in range(n_radii):
                entry[radius_index] = (entry[radius_index]
                                       + coefficients * profile[radius_index])
            pop[key] = pop.get(key, 0.0) + float(
                gross[block.offset:block.offset + block.n_ao].sum())
        for key, per_radius in channels.items():
            norms = np.array([float(np.linalg.norm(c)) ** 2
                              for c in per_radius])
            amp2[key] = amp2.get(key, 0.0) + norms
        raw.append(channels)

    # per (atom, l): the probe radius with the largest multiplet amplitude
    best = {key: int(np.argmax(values)) for key, values in amp2.items()}
    partners = []
    for channels in raw:
        atoms: dict[int, list] = {}
        for (site, l_channel), per_radius in sorted(channels.items()):
            key = (site, l_channel)
            reference = float(amp2[key][best[key]])
            if reference < 1e-24:
                continue
            factor = np.sqrt(max(pop[key], 0.0) / reference)
            scaled = per_radius[best[key]] * factor
            if float(np.linalg.norm(scaled)) < _CHANNEL_FLOOR:
                continue
            atoms.setdefault(site, []).append((l_channel, scaled))
        partners.append(atoms)

    if diagonalize and is_real and len(partners) > 1:
        from .molecular_salc import _rref_orthogonal

        keys = sorted({(site, l_channel)
                       for atoms in partners
                       for site, entries in atoms.items()
                       for l_channel, _ in entries})
        widths = {key: 2 * key[1] + 1 for key in keys}
        flat = []
        for atoms in partners:
            lookup = {(site, l_channel): channel
                      for site, entries in atoms.items()
                      for l_channel, channel in entries}
            flat.append(np.concatenate([
                np.real(lookup.get(key, np.zeros(widths[key])))
                for key in keys]))
        work = np.array(flat)
        peak = float(np.max(np.abs(work))) or 1.0
        work[:, np.max(np.abs(work), axis=0) < 0.05 * peak] = 0.0
        canonical = _rref_orthogonal(list(work))
        if len(canonical) == len(partners):
            scale = float(np.max(np.abs(work))) or 1.0
            partners = []
            for row in canonical:
                row = np.asarray(row) * scale  # rref rows are orthonormal
                atoms = {}
                start = 0
                for key in keys:
                    channel = row[start:start + widths[key]]
                    start += widths[key]
                    if float(np.linalg.norm(channel)) >= _CHANNEL_FLOOR:
                        atoms.setdefault(key[0], []).append((key[1], channel))
                partners.append(atoms)
    return partners


def _canonical_split(cell, sublattice_tokens):
    """(left_tokens, right_tokens, displayed_column) with a POSCAR-stable
    side assignment, so the Sc page, the F3 page and the crystal page of
    one compound share one --chk file."""
    from .crystal_orbital_diagram import parse_fragment_formula
    from .runtime_compat import get_chemical_symbols

    symbols = get_chemical_symbols(cell)
    elements = list(dict.fromkeys(symbols))
    if sublattice_tokens:
        pairs = parse_fragment_formula(sublattice_tokens, "--sublattice")
        chosen = [element for element, _ in pairs]
        remainder = [element for element in elements if element not in chosen]
        if not remainder:
            raise SystemExit(
                "ERROR: --sublattice must leave at least one element for "
                "the removed sublattice.")
        # the side containing the first atom's element is always "left"
        if symbols[0] in chosen:
            return sublattice_tokens, remainder, "left"
        return remainder, sublattice_tokens, "right"
    return [symbols[0]], [e for e in elements if e != symbols[0]], "mo"


def report_and_write(cell, *, sublattice, bonds, real_coefficient,
                     kpoint_filter, output_path, structure_label,
                     window=None, diagonalize=False, valence_only=False,
                     symprec=1e-5, electrons=None,
                     oxidation=None, basis=None, pseudo=None, xc="pbe",
                     kmesh=None, ke_cutoff=200.0, sigma=0.0,
                     degeneracy_tol=None, align=True, no_ghost=False,
                     symmetrize=True, max_l=None, projection="lowdin",
                     chk=None, verbose=0):
    """Solve the three periodic SCFs and write one viewer page per k."""
    from .crystal_orbital_pyscf import PySCFCrystalOrbitalDiagram
    from .crystal_orbital_spgrep import format_kpoint
    from .visualize_basis import (
        SymmetryAdaptedOrbitalBasis,
        write_html_visualization,
    )

    left, right, column = _canonical_split(cell, sublattice)
    diagram = PySCFCrystalOrbitalDiagram(
        cell, left, right, symprec=symprec, electrons=electrons,
        oxidation=oxidation, basis=basis or "gth-dzvp-molopt-sr",
        pseudo=pseudo or "gth-pbe", xc=xc, kmesh=kmesh, ke_cutoff=ke_cutoff,
        sigma=sigma, degeneracy_tol=degeneracy_tol, no_ghost=no_ghost,
        symmetrize=symmetrize, max_l=max_l, projection=projection, chk=chk,
        verbose=verbose,
    )
    described = ("crystal" if column == "mo"
                 else f"{diagram.formula[column]} sublattice")
    print(f" * PySCF levels for the SALC viewer: {described} "
          f"(fragments {diagram.formula['left']} | {diagram.formula['right']}) *")
    print(f"   basis {diagram.basis_name} / pseudo {diagram.pseudo_name} / "
          f"functional {diagram.xc.upper()}, "
          f"{'x'.join(map(str, diagram.kmesh))} k-mesh, "
          f"ke_cutoff {diagram.ke_cutoff:g} Hartree")
    diagram.run()

    kpoints = diagram.special_kpoints()
    diagram.prepare_bands([kpoint for _, kpoint in kpoints])
    records = []
    for name, kpoint in kpoints:
        levels, _ = diagram.solve_at(kpoint)
        records.append({"name": name, "kpoint": kpoint, "levels": levels})
    note = "raw column reference (--no-align)"
    if align:
        shifts, _ = diagram.align_fragment_columns(records)
        note = ("deep-level aligned: shifts left "
                f"{shifts['left']:+.2f} / mo {shifts['mo']:+.2f} / right "
                f"{shifts['right']:+.2f} eV vs the {shifts['reference']} "
                "fragment reference")
    print(f"   {note}")

    # --valence-only: shells whose occupied fragment bands sit far below the
    # crystal VBM on the aligned scale are energetically inert semicore
    # states (Sc 3s/3p and F 2s of ScF3); their admixture in a valence level
    # is the on-site orthogonality tail, dropped from the drawings
    semicore = set()
    if valence_only:
        vbm = max((lv.energy for record in records
                   for lv in record["levels"]["mo"] if lv.electrons > 0),
                  default=0.0)
        deepest: dict[tuple[str, str], float] = {}
        for record in records:
            for fragment_column in ("left", "right"):
                for lv in record["levels"][fragment_column]:
                    if lv.electrons <= 0:
                        continue
                    parts = lv.label.split()
                    if len(parts) >= 3:
                        key = (parts[0], parts[1])
                        deepest[key] = max(deepest.get(key, -1e30), lv.energy)
        semicore = {key for key, top in deepest.items() if top < vbm - 12.0}
        if semicore:
            print("   --valence-only: semicore shells "
                  + ", ".join(f"{el} {sh}" for el, sh in sorted(semicore))
                  + " dropped from the drawings (kept in levels they dominate)")
        else:
            print("   --valence-only: no semicore shells found")

    if kpoint_filter:
        wanted = [r for r in records if r["name"] == kpoint_filter]
        if not wanted:
            available = ", ".join(r["name"] for r in records)
            raise SystemExit(
                f"ERROR: k point '{kpoint_filter}' is not a special point "
                f"here (available: {available}).")
        selected = wanted
    else:
        selected = records

    # energy window over the displayed column (all plotly surfaces of a
    # full fragment spectrum would make the page enormous)
    every = [level for record in records
             for level in record["levels"][column]]
    occupied = [lv.energy for lv in every if lv.electrons > 0]
    empty = [lv.energy for lv in every if lv.electrons == 0]
    if window is not None:
        view_lo, view_hi = float(window[0]), float(window[1])
    elif occupied and empty:
        view_lo = max(occupied) - _WINDOW_BELOW_HOMO
        view_hi = min(empty) + _WINDOW_ABOVE_LUMO
    else:
        energies = [lv.energy for lv in every]
        view_lo, view_hi = min(energies) - 1.0, max(energies) + 1.0

    orbitals = SymmetryAdaptedOrbitalBasis(cell=cell, symprec=symprec)
    n_atoms = len(diagram.symbols)
    outputs = []
    for record in selected:
        name, kpoint = record["name"], record["kpoint"]
        levels = [lv for lv in record["levels"][column]
                  if view_lo <= lv.energy <= view_hi]
        dropped = len(record["levels"][column]) - len(levels)
        level_modes = []
        for mode_number, level in enumerate(levels, start=1):
            for component, atoms in enumerate(
                    _level_partner_channels(diagram, level,
                                            diagonalize=diagonalize,
                                            semicore=semicore),
                    start=1):
                level_modes.append({
                    "space": mode_number,
                    "irrep": level.label,
                    "component": component,
                    "energy": round(float(level.energy), 2),
                    "el": level.electrons,
                    "atoms": atoms,
                })
        if not level_modes:
            print(f"   {name}: no levels inside the window, skipped")
            continue
        info = {
            "formula": diagram.formula["left"] + diagram.formula["right"],
            "space_group": (
                f"{diagram.builder.spglib_dataset['international']} "
                f"(#{diagram.builder.spglib_dataset['number']})"),
            "element_orbital": f"PySCF levels: {described}",
            "kpoint": f"{name} {format_kpoint(kpoint)}",
            "real_coefficient": real_coefficient,
            "decomposition": (
                f"{len(levels)} levels in [{view_lo:.1f}, {view_hi:.1f}] eV"
                + (f" ({dropped} outside)" if dropped else "")
                + f"; {note}"),
        }
        tag = ("crystal" if column == "mo" else diagram.formula[column])
        if output_path and len(selected) == 1:
            page_path = output_path
        else:
            stem = (output_path[:-5] if output_path
                    and output_path.endswith(".html")
                    else output_path or f"SALC_pyscf_{structure_label}_{tag}")
            page_path = f"{stem}_{name}.html"
        write_html_visualization(
            page_path, orbitals, [], [],
            list(range(n_atoms)), 0, list(kpoint),
            title=(f"PySCF levels: {described} ({diagram.xc.upper()}) "
                   f"at {name} {format_kpoint(kpoint)}"),
            info=info, bonds=bonds,
            level_modes=level_modes,
            basis_heading="PySCF levels (click to show)",
        )
        outputs.append(page_path)
        print(f"   {name}: {len(levels)} levels "
              f"({len(level_modes)} partners) -> {page_path}")
    if outputs:
        print("SALC viewer pages written: " + ", ".join(outputs))


def main(argv: list[str] | None = None) -> None:
    import argparse
    from pathlib import Path

    from .crystal_orbital_diagram import parse_oxidation_tokens

    parser = argparse.ArgumentParser(
        description="PySCF eigen-levels in the SALC viewer "
                    "(crystod --visualize --pyscf).")
    parser.add_argument("--poscar", default="POSCAR")
    parser.add_argument("--sublattice", nargs="+", default=None,
                        metavar="FORMULA",
                        help="show this fragment sublattice's levels "
                        "(e.g. Sc or F3); omit for the full crystal")
    parser.add_argument("--bond", nargs=3, action="append", default=None,
                        metavar=("EL1", "EL2", "MAX"),
                        help="draw EL1-EL2 bonds up to MAX Angstrom "
                        "(VESTA-style, with coordination polyhedra)")
    parser.add_argument("--real-coefficient", action="store_true")
    parser.add_argument("--kpoint", default=None,
                        help="restrict to one special k point label (e.g. GM)")
    parser.add_argument("--window", nargs=2, type=float, default=None,
                        metavar=("LO", "HI"),
                        help="energy window in eV on the aligned scale "
                        "(default: HOMO-15 .. LUMO+10 of the shown column)")
    parser.add_argument("--diagonalize", action="store_true",
                        help="canonicalize degenerate partners (RREF): the "
                        "arbitrary SCF mixture is rotated to axis-aligned "
                        "s/p/d/f combinations; energies are unchanged")
    parser.add_argument("--valence-only", action="store_true",
                        help="drop semicore shells (occupied fragment bands "
                        "> 12 eV below the crystal VBM, e.g. Sc 3s/3p, F 2s) "
                        "from the drawings: their admixture in valence "
                        "levels is the on-site orthogonality tail, whose "
                        "node makes a bonding sigma level look antibonding")
    parser.add_argument("--electrons", type=float, default=None)
    parser.add_argument("--oxidation", nargs="+", default=None, metavar="EL=Q")
    parser.add_argument("--basis", default="gth-dzvp-molopt-sr")
    parser.add_argument("--pseudo", default="gth-pbe")
    parser.add_argument("--xc", default="pbe")
    parser.add_argument("--kmesh", type=int, nargs=3, default=None,
                        metavar=("N1", "N2", "N3"))
    parser.add_argument("--ke-cutoff", type=float, default=200.0)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--degeneracy-tol", type=float, default=None)
    parser.add_argument("--no-align", action="store_true")
    parser.add_argument("--no-ghost", action="store_true")
    parser.add_argument("--no-symmetrize", action="store_true")
    parser.add_argument("--max-l", type=int, default=None)
    parser.add_argument("--projection", choices=("lowdin", "mulliken"),
                        default="lowdin")
    parser.add_argument("--chk", default=None, metavar="FILE")
    parser.add_argument("--output", default=None)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args(argv)

    from phonopy.interface.calculator import read_crystal_structure

    cell, _ = read_crystal_structure(args.poscar, interface_mode="vasp")
    stem = Path(args.poscar).name
    for extension in (".vasp", ".poscar"):
        if stem.lower().endswith(extension):
            stem = stem[: -len(extension)]
    bonds = None
    if args.bond:
        bonds = [(el1, el2, float(max_length))
                 for el1, el2, max_length in args.bond]
    report_and_write(
        cell,
        sublattice=args.sublattice,
        bonds=bonds,
        real_coefficient=args.real_coefficient,
        kpoint_filter=args.kpoint,
        output_path=args.output,
        structure_label=stem,
        window=args.window,
        diagonalize=args.diagonalize,
        valence_only=args.valence_only,
        symprec=args.tolerance,
        electrons=args.electrons,
        oxidation=(parse_oxidation_tokens(args.oxidation)
                   if args.oxidation else None),
        basis=args.basis,
        pseudo=args.pseudo,
        xc=args.xc,
        kmesh=args.kmesh,
        ke_cutoff=args.ke_cutoff,
        sigma=args.sigma,
        degeneracy_tol=args.degeneracy_tol,
        align=not args.no_align,
        no_ghost=args.no_ghost,
        symmetrize=not args.no_symmetrize,
        max_l=args.max_l,
        projection=args.projection,
        chk=args.chk,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
