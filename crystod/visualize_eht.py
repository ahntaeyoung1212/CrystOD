"""Extended-Hueckel eigen-levels in the SALC viewer (crystod --visualize).

The extended-Hueckel counterpart of ``--visualize --pyscf``: without
``--element``/``--orbital`` the SALC viewer shows the eigenstates of the
symmetry + extended-Hueckel crystal-orbital engine (the same engine as
``crystod --diagram``) -- one table row per degenerate partner with the
level energy, and the clicked row renders the eigenvector's wave function
on the k-commensurate display cell (``--conventional`` for the
conventional cell).  No SCF and no parameters beyond the structure: the
full-electron STO basis, the VSIP/archived-level diagonal and the
point-charge ligand field are all tabulated.

Three views of one compound, as in the PySCF version:

* ``--sublattice Sc``  -- the Sc sublattice fragment (its own block of
  the shared Hamiltonian, in the point-charge field of the removed
  sublattice);
* ``--sublattice F3``  -- the other fragment;
* no ``--sublattice``  -- the full crystal (the states after bonding).

All three columns come from ONE Hamiltonian, so they share one energy
reference by construction -- no deep-level alignment step exists here.

The k points are the automatic special points of the space group (one
HTML per k point; ``--kpoint GM`` restricts the output).
"""

from __future__ import annotations

import numpy as np

from .visualize_pyscf import (
    _CHANNEL_FLOOR,
    _WINDOW_ABOVE_LUMO,
    _WINDOW_BELOW_HOMO,
    _canonical_split,
)

# STO probe radii (bohr) for the drawn lobe signs, as in the PySCF viewer:
# the per-channel best radius avoids sitting on the orthogonalization node
# of a semicore-carrying channel (Sc s of a valence level)
_PROBE_RADII = (1.5, 2.0, 2.5, 3.0)

_ANGULAR = {0: 0.28209479, 1: 0.48860251, 2: 0.63078313, 3: 0.74635267}


def _level_partner_channels(diagram, level, sqrt_overlap, site_phase,
                            diagonalize=False, semicore=None):
    """Per-partner {atom: [(l, coefficients), ...]} for one EHT level.

    Same construction as the PySCF viewer: signs and orientation from the
    same-l shells accumulated with their STO radial amplitudes, sizes
    calibrated to the Loewdin populations with one factor per (atom, l)
    channel from the multiplet sums, probe radius chosen per channel as
    the one where the accumulated amplitude is largest.  Fragment levels
    iterate their own sublattice's shells only (their vectors are exactly
    zero on the other block anyway -- no ghost basis in the EHT engine).

    Gauge: the EHT engine's Bloch orbitals carry the SITE phase
    (bloch_overlap: exp(2 pi i k.(T + x_j - x_i))), while the viewer
    applies exp(2 pi i k.T) per displayed image only (the atomic gauge
    of the PySCF pages).  ``site_phase`` = diag(exp(2 pi i k.x_i)) is
    the per-AO gauge transform into the viewer convention, applied to
    the eigenvectors BEFORE realification -- at k = 0 it is identity,
    but at zone-boundary points the inter-sublattice factors are complex
    (e.g. i for the F sites of ScF3 at R) and drawing the raw
    coefficients would misplace the relative phases between sublattices.
    ``sqrt_overlap`` must already be in the same atomic gauge
    (D S^(1/2) D+, built by the caller).
    """
    from .point_charge_field import _primitives
    from .visualize_basis import realify_basis_space

    rows, is_real = realify_basis_space(
        (level.vectors * site_phase[:, None]).T)
    rows = np.asarray(rows)
    column = getattr(level, "column", "mo")
    n_radii = len(_PROBE_RADII)

    specs = [spec for spec in diagram.specs
             if (column == "mo" or spec.column == column) and spec.l <= 3]
    radial_weights = {}
    for spec in specs:
        radial_weights[id(spec)] = tuple(
            _ANGULAR[spec.l] * sum(
                c * radius ** (n - 1) * np.exp(-z * radius)
                for c, n, z in _primitives(spec.n, spec.zeta)
            )
            for radius in _PROBE_RADII
        )

    raw = []   # per partner: {(site, l): [channel per radius]}
    pop: dict[tuple[int, int], float] = {}
    amp2: dict[tuple[int, int], np.ndarray] = {}
    for vector in rows:
        v = np.asarray(vector, dtype=complex)
        gross = np.abs(sqrt_overlap @ v) ** 2
        total_gross = float(gross.sum()) or 1.0
        channels: dict[tuple[int, int], list] = {}
        for spec in specs:
            width = 2 * spec.l + 1
            if semicore and (spec.element, spec.shell) in semicore:
                share = float(
                    gross[spec.offset:spec.offset + spec.n_ao].sum())
                if share < 0.4 * total_gross:
                    continue  # orthogonality tail, not the semicore band
            profile = radial_weights[id(spec)]
            for site_pos, site in enumerate(spec.sites):
                start = spec.offset + site_pos * width
                coefficients = v[start:start + width]
                key = (site, spec.l)
                entry = channels.setdefault(
                    key, [np.zeros(width, dtype=complex)
                          for _ in range(n_radii)])
                for radius_index in range(n_radii):
                    entry[radius_index] = (
                        entry[radius_index]
                        + coefficients * profile[radius_index])
                pop[key] = pop.get(key, 0.0) + float(
                    gross[start:start + width].sum())
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


def report_and_write(cell, *, sublattice, bonds, real_coefficient,
                     kpoint_filter, output_path, structure_label,
                     window=None, diagonalize=False, valence_only=False,
                     symprec=1e-5, electrons=None, oxidation=None,
                     conventional=False):
    """Solve the extended-Hueckel levels and write one viewer page per k."""
    from .crystal_orbital_diagram import CrystalOrbitalDiagram
    from .crystal_orbital_spgrep import format_kpoint
    from .visualize_basis import (
        SymmetryAdaptedOrbitalBasis,
        write_html_visualization,
    )

    left, right, column = _canonical_split(cell, sublattice)
    diagram = CrystalOrbitalDiagram(
        cell, left, right, symprec=symprec, electrons=electrons,
        oxidation=oxidation, conventional=conventional,
    )
    described = ("crystal" if column == "mo"
                 else f"{diagram.formula[column]} sublattice")
    print(f" * Extended-Hueckel levels for the SALC viewer: {described} "
          f"(fragments {diagram.formula['left']} | {diagram.formula['right']}) *")
    print("   full-electron STO basis, point charges "
          + " ".join(f"{element}{diagram.oxidation[element]:+g}"
                     for element in dict.fromkeys(diagram.symbols))
          + f", {int(diagram.electrons)} electrons per cell")

    kpoints = diagram.special_kpoints()
    records = []
    for name, kpoint in kpoints:
        levels, _ = diagram.solve_at(kpoint)
        records.append({"name": name, "kpoint": kpoint, "levels": levels})
    note = ("one shared extended-Hueckel Hamiltonian: fragment and "
            "crystal columns on one energy reference")
    print(f"   {note}")

    # --valence-only: same criterion as the PySCF viewer (shells whose
    # occupied fragment bands sit >12 eV below the crystal VBM)
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

    # energy window over the displayed column (a full-electron EHT
    # spectrum reaches the deep 1s cores; hundreds of plotly surfaces
    # would bloat the page).  The lower edge is anchored to the global
    # HOMO - 15 eV, but never above any single k point's own top occupied
    # level: a metal's occupied bandwidth can exceed 15 eV, and a purely
    # global anchor would silently drop whole k-point pages (fcc Al: every
    # GM level sits below X's frontier)
    every = [level for record in records
             for level in record["levels"][column]]
    occupied = [lv.energy for lv in every if lv.electrons > 0]
    empty = [lv.energy for lv in every if lv.electrons == 0]
    if window is not None:
        view_lo, view_hi = float(window[0]), float(window[1])
    elif occupied and empty:
        view_lo = max(occupied) - _WINDOW_BELOW_HOMO
        per_k_top = [
            max((lv.energy for lv in record["levels"][column]
                 if lv.electrons > 0), default=None)
            for record in records
        ]
        per_k_top = [value for value in per_k_top if value is not None]
        if per_k_top:
            view_lo = min(view_lo, min(per_k_top) - 1.0)
        view_hi = min(empty) + _WINDOW_ABOVE_LUMO
    else:
        energies = [lv.energy for lv in every]
        view_lo, view_hi = min(energies) - 1.0, max(energies) + 1.0

    orbitals = SymmetryAdaptedOrbitalBasis(cell=cell, symprec=symprec)
    n_atoms = len(diagram.symbols)
    outputs = []
    for record in selected:
        name, kpoint = record["name"], record["kpoint"]
        # per-AO gauge transform into the viewer's atomic gauge (see
        # _level_partner_channels): exp(2 pi i k.x_i) per AO row
        site_phase = np.ones(diagram.n_ao, dtype=complex)
        for spec in diagram.specs:
            width = 2 * spec.l + 1
            for site_pos, site in enumerate(spec.sites):
                start = spec.offset + site_pos * width
                site_phase[start:start + width] = np.exp(
                    2j * np.pi * float(np.dot(kpoint,
                                              diagram.positions[site])))
        # Loewdin populations calibrate the drawn lobe sizes, as in the
        # diagram's display compositions; the overlap is transformed into
        # the same atomic gauge (D S D+) so |S^(1/2) c| is gauge-consistent
        overlap = diagram.bloch_overlap(kpoint)
        overlap = (site_phase[:, None] * overlap) * site_phase.conj()[None, :]
        eigenvalues, eigenvectors = np.linalg.eigh(overlap)
        sqrt_overlap = (eigenvectors * np.sqrt(np.clip(eigenvalues.real,
                                                       0.0, None))
                        ) @ eigenvectors.conj().T
        levels = [lv for lv in record["levels"][column]
                  if view_lo <= lv.energy <= view_hi]
        dropped = len(record["levels"][column]) - len(levels)
        level_modes = []
        for mode_number, level in enumerate(levels, start=1):
            for component, atoms in enumerate(
                    _level_partner_channels(diagram, level, sqrt_overlap,
                                            site_phase,
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
            "element_orbital": f"extended-Hueckel levels: {described}",
            "kpoint": f"{name} {format_kpoint(kpoint)}",
            # the level pages ALWAYS realify the degenerate partners
            # (whenever a real basis exists) and the drawn field is Re[psi]
            # regardless, so the sidebar honestly says "real coefficients";
            # --real-coefficient is accepted for CLI symmetry with the
            # basis viewer but changes nothing here
            "real_coefficient": True,
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
                    else output_path or f"SALC_eht_{structure_label}_{tag}")
            page_path = f"{stem}_{name}.html"
        write_html_visualization(
            page_path, orbitals, [], [],
            list(range(n_atoms)), 0, list(kpoint),
            title=(f"Extended-Hueckel levels: {described} "
                   f"at {name} {format_kpoint(kpoint)}"),
            info=info, bonds=bonds,
            conventional=conventional,
            level_modes=level_modes,
            basis_heading="extended-Hueckel levels (click to show)",
        )
        outputs.append(page_path)
        print(f"   {name}: {len(levels)} levels "
              f"({len(level_modes)} partners) -> {page_path}")
    if outputs:
        print("SALC viewer pages written: " + ", ".join(outputs))
    else:
        raise SystemExit(
            f"ERROR: no levels inside the energy window "
            f"[{view_lo:.1f}, {view_hi:.1f}] eV at the selected k "
            "point(s); widen it with --window LO HI.")


def main(argv: list[str] | None = None) -> None:
    import argparse
    from pathlib import Path

    from .crystal_orbital_diagram import parse_oxidation_tokens

    parser = argparse.ArgumentParser(
        description="Extended-Hueckel eigen-levels in the SALC viewer."
    )
    parser.add_argument("--poscar", default="POSCAR")
    parser.add_argument("--sublattice", nargs="+", default=None,
                        metavar="FORMULA",
                        help="display one fragment sublattice instead of "
                        "the crystal, e.g. Sc or F3")
    parser.add_argument("--bond", nargs=3, action="append", default=None,
                        metavar=("EL1", "EL2", "MAX"))
    parser.add_argument("--real-coefficient", action="store_true")
    parser.add_argument("--kpoint", default=None,
                        help="restrict to one special k point label (e.g. GM)")
    parser.add_argument("--window", nargs=2, type=float, default=None,
                        metavar=("LO", "HI"),
                        help="energy window in eV (default: HOMO-15 to "
                        "LUMO+10 of the displayed column)")
    parser.add_argument("--diagonalize", action="store_true",
                        help="canonicalize degenerate partners (RREF)")
    parser.add_argument("--valence-only", action="store_true",
                        help="drop semicore orthogonality tails from the "
                        "drawings")
    parser.add_argument("--electrons", type=float, default=None)
    parser.add_argument("--oxidation", nargs="+", default=None, metavar="EL=Q")
    parser.add_argument("--conventional", action="store_true",
                        help="display in the conventional cell instead of "
                        "the primitive k-commensurate supercell")
    parser.add_argument("--output", default=None)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args(argv)

    from .star_of_k import read_poscar_or_exit

    cell = read_poscar_or_exit(args.poscar)
    stem = Path(args.poscar).name
    for extension in (".vasp", ".poscar"):
        if stem.lower().endswith(extension):
            stem = stem[: -len(extension)]
    bonds = None
    if args.bond:
        try:
            bonds = [(el1, el2, float(max_length))
                     for el1, el2, max_length in args.bond]
        except ValueError:
            raise SystemExit(
                "ERROR: --bond expects a numeric maximum length in "
                "Angstroms, e.g. --bond Sc F 2.3.")
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
        conventional=args.conventional,
    )


if __name__ == "__main__":
    main()
