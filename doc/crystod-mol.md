# crystod-mol

Molecular point-group detection, molecular SALCs (symmetry-adapted linear
combinations of atomic orbitals), and molecular-orbital diagrams — the
molecular counterpart of the crystalline analyses, working on molecules (XYZ
coordinate files) instead of periodic structures.

## 32. Molecular point groups and molecular SALCs

*Example directory: `example/32_molecular_salc` (testsuite section 32;
molecule files in `example/test_XYZs`)*

### Point-group detection (`--symmetry`)

Detect the point group of a molecule — the molecular analogue of
`phonopy --symmetry` for crystals — using pymatgen's `PointGroupAnalyzer`:

```bash
crystod-mol --symmetry --xyz XYZ_O2.xyz
```

```
* Molecule *
XYZ_O2.xyz (O2, 2 atoms)

* Point group *
D*h (linear molecule; continuous non-crystallographic group)
```

For the 32 crystallographic point groups, the report also prints the
Hermann-Mauguin symbol — the international crystallographic name of the group,
as opposed to the Schoenflies name used above — and the symmetry operations
sorted into classes, i.e. sets of operations that the group maps onto one
another and that therefore share the same character (e.g. NH3:
`C3v (Hermann-Mauguin: 3m)`, `E, 2C3, 3sgv`). `--tolerance` (in Angstrom,
default 0.3 as in pymatgen) sets how much distortion the symmetry detection
will tolerate, like `phonopy --tolerance`.

### Molecular SALCs (`--element`/`--orbital`)

Build the molecular SALCs — symmetry-adapted linear combinations of atomic
orbitals, i.e. the combinations of the atomic orbitals on equivalent sites
that transform cleanly under the symmetry of the molecule — for the sites of
one element. Three steps are carried out. First the site-permutation
representation of the selected sites is constructed: a record of how each
symmetry operation shuffles those sites among themselves. Its characters (the
traces of the corresponding matrices) are then multiplied, class by class, by
the characters of the atomic orbital in question (s/p/d/f). Finally the
product is decomposed into the irreducible representations (irreps) of the
molecular point group — the elementary symmetry types A1, E, T2, ... — and the
explicit SALCs are projected out with the projection-operator technique:

```bash
crystod-mol --xyz XYZ_NH3.xyz --element H --orbital s
```

```
* Molecule *
XYZ_NH3.xyz (H3N, 4 atoms)

* Point group *
C3v (Hermann-Mauguin: 3m)

* Target sites (H, 3 sites; center-of-mass frame) *
H1: ( 0.807734,  0.466331, -0.298390)
H2: (-0.807709,  0.466374, -0.298391)
H3: (-0.000025, -0.932662, -0.298459)

* Reducible representation (H sites x s orbital) *
class:               E   2C3  3sgv
chi(perm):           3     0     1
chi(s):              1     1     1
chi(total):          3     0     1

* Decomposition *
Gamma = 1(A1) + 1(E)

* SALCs (orbital axes = input Cartesian axes) *
A1: [s(H1) + s(H2) + s(H3)]
E: [s(H1) - s(H3), s(H1) - 2 s(H2) + s(H3)]
```

The irrep labels come from the same point-group character tables as
`crystod-group` (`--decompose`/`--ligand-field`), so molecular and crystalline
analyses share one labeling convention. The detected Schoenflies group (the `C3v`-style name) is mapped to its
Hermann-Mauguin symbol, and the molecular symmetry operations are matched onto
the exact matrices of the character table; this also removes the numerical
noise of the input geometry from the SALC coefficients. For p/d/f orbitals a
symmetry operation does not merely move an orbital to another site, it also
rotates the orbital itself, so the site-permutation representation is
multiplied by the real-orbital Wigner-D representation — the matrices
describing how the real p/d/f orbitals rotate (`wigner_D_real`, section 1 of
{doc}`crystod`). The SALCs therefore mix the orbital components correctly —
e.g. NH3 `--element N --orbital p` gives `A1: [pz(N1)]`,
`E: [px(N1), py(N1)]`.

Options:

- `--align` rotates the molecule into the standard point-group orientation
  (the highest-order rotation axis, the principal axis, placed along z) before
  the analysis, so the SALC coefficients follow the textbook axis convention — e.g. the (arbitrarily rotated) CH4
  example with `--element C --orbital d --align` gives the clean crystal-field
  splitting `E: [dz2, dx2-y2]`, `T2: [dxy, dyz, dxz]`;
- `--show-matrix` prints the site-permutation matrix of every symmetry
  operation;
- `--tolerance` as in `--symmetry`.

### 3D visualization (`--visualize`)

`--visualize` additionally writes the SALCs as a standalone interactive 3D
HTML page — the same viewer as the crystalline `crystod --visualize`
(section 5 of {doc}`crystod`), with the orbital lobes drawn at each site
(+ yellow / − cyan, VESTA style), a sidebar listing every SALC basis vector
by irrep for one-click switching, and a camera-synced x/y/z compass:

```bash
crystod-mol --xyz XYZ_NH3.xyz --element H --orbital s --visualize --bond N H 1.2
# -> SALC_XYZ_NH3_H_s.html
```

`--output` selects the HTML file name (default:
`SALC_{molecule}_{element}_{orbital}.html`); `--bond EL1 EL2 MAX` draws bonds
up to `MAX` Angstroms (repeatable). Since the viewer works on a molecule
rather than a crystal, no cell edges are drawn and the compass shows the
Cartesian x/y/z axes; combining with `--align` shows the SALCs in the
standard point-group orientation.

The viewer below is the live output of exactly that command — the H 1s SALCs
of NH3, i.e. the `A1: [1s(H1) + 1s(H2) + 1s(H3)]` and E combinations printed
above, drawn as VESTA-style + yellow / − cyan lobes on the three hydrogens.
Click a row of the SALC table in the sidebar to switch basis vector and drag
to rotate:

```{raw} html
<iframe src="_static/embed/SALC_XYZ_NH3_H_s.html" width="100%" height="620" loading="lazy" style="border:1px solid #8884; border-radius:8px; background:#fff;"></iframe>
<p style="margin-top:0.3em"><a href="_static/embed/SALC_XYZ_NH3_H_s.html" target="_blank">Open the NH3 H-1s SALC viewer full-screen</a></p>
```

The SALC analysis supports the 32 crystallographic point groups; for linear
molecules (D\*h/C\*v) analyze a finite subgroup with `crystod-group
--decompose` instead.

## 33. Molecular-orbital diagrams from symmetry + overlap (`--diagram`)

*Example directory: `example/33_molod` (testsuite section 33)*

`crystod-mol --diagram` draws the molecular-orbital diagram of a single-center
molecule (one central atom plus the surrounding ligands, i.e. the atoms bonded
to it, e.g. NH3, CH4, SF6) from **symmetry and overlap alone**. The level
ordering follows from the symmetry of the orbitals and from how strongly they
overlap in space; no self-consistent quantum-chemistry calculation — no
iterative solution of the electronic structure — is run:

```bash
crystod-mol --diagram --xyz XYZ_NH3.xyz
# -> MolOD_XYZ_NH3.html
```

```
* Molecule *
XYZ_NH3.xyz (NH3, 4 atoms)

* Point group *
C3v (Hermann-Mauguin: 3m)

* Fragments *
central atom: N; ligands: 3 H

* Valence Atomic Orbital (AO) parameters (single-zeta STO, extended Hueckel) *
N 2s:  zeta = 1.950 / bohr,  H_ii = -26.0 eV
N 2p:  zeta = 1.950 / bohr,  H_ii = -13.4 eV
H 1s:  zeta = 1.300 / bohr,  H_ii = -13.6 eV

* Ligand SALCs (standard point-group axes) *
-- 3H 1s --
A1: [1s(H1) + 1s(H2) + 1s(H3)]
E: [1s(H1) - 1s(H3), 1s(H1) - 2 1s(H2) + 1s(H3)]

* Ligand SALC | central AO overlap integrals *
  A1:  < a1 (E =  -16.44 eV) | N 2s >  S = 0.7205
  A1:  < a1 (E =  -16.44 eV) | N 2p >  S = 0.2387
   E:  < e (E =  -11.16 eV) | N 2p >  S = 0.5687

* Molecular orbitals (Wolfsberg-Helmholz, K = 1.75) *
    MO    E (eV)  occ  composition
   4a1     21.14    0  70% SALC a1, 25% N 2s, 5% N 2p
 2e x2      2.78    0  59% SALC e, 41% N 2p
   3a1    -13.75    2  95% N 2p, 4% SALC a1, 1% N 2s
 1e x2    -16.49    4  59% N 2p, 41% SALC e
   2a1    -28.00    2  73% N 2s, 27% SALC a1

* Electron filling (8 valence electrons) *
(2a1)^2 (1e)^4 (3a1)^2
(MO numbering counts the core shells, not shown: N 1s -> a1)
HOMO = 3a1 (-13.75 eV), LUMO = 2e (2.78 eV), gap = 16.53 eV
```

The written `MolOD_XYZ_NH3.html` is the diagram below — this is the live
output, not a screenshot: hover or click any level to see its composition and
a drag-rotatable orbital sketch, zoom the energy window with Ctrl/Cmd +
scroll:

```{raw} html
<iframe src="_static/embed/MolOD_XYZ_NH3.html" width="100%" height="660" loading="lazy" style="border:1px solid #8884; border-radius:8px; background:#fff;"></iframe>
<p style="margin-top:0.3em"><a href="_static/embed/MolOD_XYZ_NH3.html" target="_blank">Open the NH3 MO diagram full-screen</a></p>
```

The construction follows the textbook route, made quantitative step by step:

1. the molecular point group is detected and the ligand atomic orbitals are
   symmetry-adapted per irrep (the SALCs of section 32) — for NH3,
   `A1: [1s(H1) + 1s(H2) + 1s(H3)]` and
   `E: [1s(H1) - 1s(H3), 1s(H1) - 2 1s(H2) + 1s(H3)]`;
2. every valence orbital — the outer, chemically active orbital of each atom —
   is represented by a single-zeta Slater-type orbital (STO), i.e. by one
   exponential function per orbital, with the standard extended-Hückel
   exponents. Each orbital also carries a diagonal energy H_ii, taken from its
   valence-state ionization energy, which measures how tightly that orbital
   binds an electron in the free atom;
3. all two-center overlap integrals between the STOs — the numbers S that
   measure how much two orbitals on different atoms occupy the same region of
   space — are evaluated **exactly**, by Gauss-Laguerre x Gauss-Legendre
   quadrature in prolate-spheroidal coordinates, to machine precision and
   validated against the analytic H 1s-1s formula. Ligand-ligand overlap is
   therefore *not* neglected. The overlaps between each ligand SALC and each
   central-atom atomic orbital (AO) are printed per irrep
   (NH3: `< a1 | N 2s > S = 0.72`, `< e | N 2p > S = 0.57`);
4. orbitals belonging to different irreps cannot mix, so the problem falls
   apart into one small block per irrep. Within each block the generalized
   eigenvalue problem — the secular equation whose solutions are the
   molecular-orbital energies and their coefficients — is solved with the
   Wolfsberg-Helmholz off-diagonal elements
   H_ij = K S_ij (H_ii + H_jj)/2 (K = 1.75). A large overlap integral
   therefore directly produces a large splitting between the bonding and the
   antibonding level (this is symmetry-adapted extended Hückel);
5. the result is written as an interactive HTML/SVG diagram **by default**:
   four columns (isolated ligand AOs | ligand-group SALCs | molecular
   orbitals | central-atom AOs), joined by dashed correlation lines — each line
   ties a molecular orbital to a fragment orbital it is built from, and is
   drawn the more heavily the larger that contribution is. Electron arrows mark
   the occupied levels, the HOMO (highest occupied molecular orbital) and the
   LUMO (lowest unoccupied molecular orbital) are flagged, and a click/hover
   panel gives the composition of every level in percent. The energy window is adjustable in the page itself (min/max input boxes, Ctrl/Cmd + scroll or pinch to zoom, drag to pan) — useful when deep ligand levels stretch the scale. Hovering or clicking any level additionally shows an **orbital sketch** in the details panel: the molecule drawn with VESTA-style +/− lobes (yellow/cyan), built from that level's actual eigenvector, i.e. from the computed mixing coefficients of that orbital rather than from a schematic picture. The sketch is drag-rotatable, and buttons flip through the degenerate partners — the orbitals that symmetry forces to have exactly the same energy.

The molecular-orbital (MO) numbering counts the core shells, as in
photoelectron spectroscopy. CH4 therefore gives the textbook
`(2a1)^2 (1t2)^6` (1a1 = C 1s core; the antibonding 2t2 and 3a1 stay empty),
and NH3 gives `(2a1)^2 (1e)^4 (3a1)^2` with the 3a1 lone pair as HOMO. SF6
fills 48 valence electrons up to the nonbonding F 2p block (1t1g/3eg) —
levels that are neither bonding nor antibonding — which sits just below the
6a1g LUMO. The terminal report lists the AO parameters, the ligand SALCs, the
overlap integrals between the two fragments, and the full MO table (energy,
occupation, composition).

`--center EL` selects the central atom explicitly (default: the atom closest
to the molecular center); `--output`/`--tolerance` as usual. Elements H-Cl
are parameterized. The energies are semi-quantitative, as extended Hückel theory always is: read
them for level orderings and trends, not as accurate absolute values. The
intended use is as the *crystal-orbital diagram* counterpart for materials
design, where the same symmetry-plus-overlap logic connects crystal SALCs
({doc}`crystod`) to band structures.

References: M. Wolfsberg and L. Helmholz, J. Chem. Phys. 20, 837 (1952);
R. Hoffmann, J. Chem. Phys. 39, 1397 (1963).

### Two-fragment diagrams (`--ao-left`/`--ao-right`)

`--ao-left`/`--ao-right` (without `--pyscf`) replace the central-atom
architecture by **any submolecule split by chemical formula** — the
extended-Hückel counterpart of the three-column `--pyscf` diagram, for
molecules without a single center:

```bash
crystod-mol --diagram --xyz XYZ_C6H6.xyz --ao-left H6 --ao-right C6
# -> MolOD_XYZ_C6H6.html   (H6 MOs | C6H6 MOs | C6 MOs)
```

All three columns are solved in the **one molecular AO space**: since the
Wolfsberg-Helmholz off-diagonal H_ij = K S_ij (H_ii + H_jj)/2 depends only on
the two orbitals, the fragment's (H, S) sub-block *is* the isolated fragment
— its generalized eigenstates are the pre-bonding fragment MOs (the
eigenstates of the SALC splitting of section 32: benzene's H6 column shows
the 1a1g < 1e1u < 1e2g < 1b2u ladder of the six H 1s SALCs), and the
molecular MOs are projected onto them through the shared overlap matrix
(correlation lines + composition panel, COOP bonding colors, orbital
sketches, VESTA element colors — as in the `--pyscf` diagram). Irrep labels
are assigned from the MO characters after solving, with the omitted core
shells counted in the numbering (benzene fills
`(2a1g)^2 (2e1u)^4 (2e2g)^4 (2b2u)^2 (3a1g)^2 (3e1u)^4 (1a2u)^2 (1b1u)^2
(3e2g)^4 (1e1g)^4` with the π ladder 1a2u < 1e1g (HOMO) < 1e2u (LUMO) <
1b1g — the Hückel result, and label-compatible with the `--pyscf` diagram);
a fragment that keeps a *higher* symmetry than the molecule (the CO of CH3OH
under Cs: its π pair stays exactly degenerate and χ(E) = 2 matches no Cs
irrep) is decomposed with the irrep projectors (a' + a''). Fragments need
not be invariant under the full molecular point group — labels are simply
omitted when no consistent assignment exists (and for linear molecules,
which have no crystallographic character table).

```bash
crystod-mol --diagram --xyz XYZ_CH3OH.xyz --ao-left H4 --ao-right CO
crystod-mol --diagram --xyz XYZ_O2.xyz --ao-left O --ao-right O
```

### Quantitative diagrams with PySCF (`--pyscf`)

`--diagram --pyscf` replaces the extended-Hückel estimate by **three
self-consistent-field (SCF) calculations sharing one AO space** (PySCF) —
three genuine electronic-structure calculations, each iterated until the
orbitals and the field they feel agree with one another, and all expressed in
the *same* set of atomic-orbital basis functions. The three are the full
molecule and its two fragments, the latter carrying **ghost basis functions**
on the atoms that have been removed.

```{admonition} Why ghost atoms? The basis-set superposition error (BSSE)
:class: note

A quantum-chemistry calculation builds the wave function — the mathematical
object from which every property of the electrons follows — out of a
**finite** set of atom-centred basis functions, so its quality depends on how
many functions sit near the electrons. Compute the O atom of H2O in the basis
of O alone, then the whole molecule in the basis of O **and** the two H atoms,
and the molecule has extra freedom to describe its electrons — more functions
to build them from — that the isolated fragment never had. Its energy drops
for a reason that has nothing to do with chemical bonding.
The spurious part of the stabilization is the **basis-set superposition error
(BSSE)** — an artefact of the finite basis rather than a physical effect. It
makes computed interaction energies look too attractive, and it does not
vanish simply by enlarging the basis on one side only.

The standard cure is the **counterpoise correction** of Boys and Bernardi:
compute each fragment in the *full* molecular basis, keeping the basis
functions of the removed atoms but not their nuclei or electrons. Such
nucleus-free, electron-free basis centres are called **ghost atoms**. Both
fragments then carry exactly the same basis-set quality as the molecule, so
the difference E(mol) − E(left) − E(right) contains bonding only.

CrystOD needs this twice over. First, the interaction energy printed under the
diagram is BSSE-free. Second, because all three calculations live in *one and
the same* AO space, every molecular MO can be **projected exactly** onto the
fragment MOs — written as an exact combination of them, with no fitting and no
approximation. That is what makes the correlation lines and the percentage
compositions quantitative rather than merely indicative. The price is that a few
fragment solutions end up living mostly on the ghost centres; those are
filtered out, as described just below.
```

The fragment levels are therefore true pre-bonding states — the orbitals each
half of the molecule has *before* the two halves are allowed to interact —
described in the full molecular basis. The molecular MOs are projected exactly
onto them, which is what the correlation lines and the composition panel
display, and the printed interaction energy E(mol) − E(left) − E(right) is
BSSE-free. The diagram becomes three columns:
left fragment | molecule | right fragment. The hover panel shows the orbital sketch of every level, fragment MOs
included, drawn from the PySCF MO coefficients. Fragment sketches show the
real atoms only — the small counterpoise tails on the ghost basis are left
out — and degenerate partners are canonicalized, i.e. the arbitrary mixing
within a set of equal-energy orbitals is fixed to one standard choice, so the
sketches match the SALCs drawn by `--visualize` exactly. Each fragment level is judged by its real-atom Mulliken population — the share of the electron density that Mulliken's standard partitioning assigns to the real atoms rather than to the ghost centres. A level sitting mostly on the ghost basis (population < 35%) is a BSSE artifact rather than a genuine fragment state, so it is dropped from the diagram and from the compositions. The remaining projections are weighted by the real-atom population; in benzene this makes the pure-π 1e1g HOMO come out as 100% C6 1e1g, with no spurious "H6" character borrowed from ghost-carbon levels.

```bash
crystod-mol --diagram --xyz XYZ_H2O.xyz --pyscf                # H2 | H2O | O
crystod-mol --diagram --xyz XYZ_O2.xyz --pyscf --spin 2 --ao-left O --ao-right O
crystod-mol --diagram --xyz XYZ_CH3OH.xyz --pyscf --ao-left H4 --ao-right CO
crystod-mol --diagram --xyz XYZ_C6H6.xyz --pyscf --ao-left H6 --ao-right C6
```

The benzene run produces the diagram below (live output of the last command):
the H6 cage on the left, the C6 ring on the right, and the pure-π `1e1g` HOMO /
`1e2u` LUMO of benzene in the middle — click the π levels to see the familiar
nodal patterns in the orbital sketch:

```{raw} html
<iframe src="_static/embed/MolOD_XYZ_C6H6_pyscf.html" width="100%" height="660" loading="lazy" style="border:1px solid #8884; border-radius:8px; background:#fff;"></iframe>
<p style="margin-top:0.3em"><a href="_static/embed/MolOD_XYZ_C6H6_pyscf.html" target="_blank">Open the benzene PySCF MO diagram full-screen</a></p>
```

- The default fragments are the ligand cage | central atom, as in the
  symmetry-only mode. `--ao-left`/`--ao-right` instead select **any partition
  by chemical formula**, which is required for molecules that have no single
  central atom (benzene, split as `H6`/`C6`) and for homonuclear diatomics —
  two-atom molecules of one element (`O` / `O`, displayed as O(L)/O(R)).
- Fragments are **spin/spherically averaged**: the electrons of a partly
  filled frontier shell are spread evenly over it, so that each degenerate
  orbital of the shell takes the same fractional occupation. Without this, a
  partially filled shell (C 2p², the t2² of an H4 cage) would break the
  point-group symmetry, whereas the pre-bonding reference should preserve it.
- Irrep labels come from crystod's own point-group machinery: for every MO the
  characters under the exact character-table operations are evaluated with the
  symmetry representation on the PySCF AO basis. The labels therefore agree
  with the symmetry-only diagram and with `crystod-group` — CH4 `1t2` HOMO,
  H2O `1b2`, benzene `1e1g` HOMO / `1e2u` LUMO, all quoted at the Koopmans
  level, i.e. reading the orbital energies themselves as the ionization
  energies. Linear molecules use PySCF's Dooh/Coov labels rendered as σ/π/δ
  (triplet O2: `(1πu)^4 (1πg)^2`, with two single arrows on the half-filled
  1πg).
- `--basis` (default def2-svp), `--theory scf|dft`, `--xc` (default b3lyp),
  `--charge` and `--spin` follow the conventions of `script/calc_pyscf.py`.
  `--theory` chooses between a plain self-consistent-field (Hartree-Fock)
  calculation and density-functional theory, and `--xc` names the
  exchange-correlation functional used in the latter. `--spin` is given as 2S,
  the number of unpaired electrons, and defaults to the smallest value
  compatible with the electron count (e.g. `--spin 2` for triplet O2).
  Open-shell systems — those with unpaired electrons — are treated with the
  restricted open-shell methods ROHF and ROKS (restricted open-shell
  Hartree-Fock and Kohn-Sham), so that the diagram keeps a single set of
  orbital energies rather than one per spin. Core levels — the deep,
  chemically inert inner-shell orbitals
  (O 1s at −560 eV, I 1s at −33 keV, ...) — are computed and included.
  Whenever levels lie below −40 eV, the default energy window is clamped to
  the chemically relevant range [−40, 15] eV so that the valence region stays
  legible, and the LUMO is never hidden. The **Show all energy levels** button
  in the page expands the window to the full range.

The SCF engine is PySCF — a software dependency, not the source of the
fragment/irrep construction above. If you use PySCF in your research,
please cite: Q. Sun et al., J. Chem. Phys. 153, 024109 (2020);
Q. Sun et al., WIREs Comput. Mol. Sci. 8, e1340 (2018);
Q. Sun, J. Comput. Chem. 36, 1664 (2015).

