# crystod-mol

Molecular point-group detection, molecular SALCs, and molecular-orbital
diagrams — the molecular counterpart of the crystalline analyses, working on
molecules (XYZ files) instead of periodic structures.

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

For the 32 crystallographic point groups, the Hermann-Mauguin symbol and the
symmetry operations grouped by class are printed as well (e.g. NH3:
`C3v (Hermann-Mauguin: 3m)`, `E, 2C3, 3sgv`). `--tolerance` (Angstrom,
default 0.3 as in pymatgen) controls the symmetry detection, like
`phonopy --tolerance`.

### Molecular SALCs (`--element`/`--orbital`)

Build the molecular SALCs (symmetry-adapted linear combinations of atomic
orbitals) for the sites of an element: the site-permutation representation of
the selected sites is constructed, its characters are multiplied class by
class by the characters of the atomic orbital (s/p/d/f), the product is
decomposed into the irreps of the molecular point group, and the explicit
SALCs are projected out with the projection-operator technique:

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
analyses share one labeling convention. The detected Schoenflies group is
mapped to its Hermann-Mauguin symbol, the molecular symmetry operations are
matched onto the exact character-table matrices (which also removes the
numerical noise of the input geometry from the SALC coefficients), and for
p/d/f orbitals the site-permutation representation is multiplied by the
real-orbital Wigner-D representation (`wigner_D_real`, section 1 of
{doc}`crystod`), so the SALCs mix orbital components correctly — e.g. NH3
`--element N --orbital p` gives `A1: [pz(N1)]`, `E: [px(N1), py(N1)]`.

Options:

- `--align` rotates the molecule into the standard point-group orientation
  (principal axis along z) before the analysis, so the SALC coefficients
  follow the textbook axis convention — e.g. the (arbitrarily rotated) CH4
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
crystod-mol --xyz XYZ_NH3.xyz --element H --orbital p --visualize --bond N H 1.2
# -> SALC_XYZ_NH3_H_p.html
```

`--output` selects the HTML file name (default:
`SALC_{molecule}_{element}_{orbital}.html`); `--bond EL1 EL2 MAX` draws bonds
up to `MAX` Angstroms (repeatable). Since the viewer works on a molecule
rather than a crystal, no cell edges are drawn and the compass shows the
Cartesian x/y/z axes; combining with `--align` shows the SALCs in the
standard point-group orientation.

The SALC analysis supports the 32 crystallographic point groups; for linear
molecules (D\*h/C\*v) analyze a finite subgroup with `crystod-group
--decompose` instead.

## 33. Molecular-orbital diagrams from symmetry + overlap (`--diagram`)

*Example directory: `example/33_molod` (testsuite section 33)*

`crystod-mol --diagram` draws the molecular-orbital diagram of a single-center
molecule (one central atom + surrounding ligands, e.g. NH3, CH4, SF6) from
**symmetry and overlap alone** — no self-consistent quantum-chemistry
calculation:

```bash
crystod-mol --diagram --xyz XYZ_NH3.xyz
# -> MolOD_XYZ_NH3.html
```

The construction follows the textbook route, made quantitative step by step:

1. the molecular point group is detected and the ligand atomic orbitals are
   symmetry-adapted per irrep (the SALCs of section 32) — for NH3,
   `A1: [1s(H1) + 1s(H2) + 1s(H3)]` and
   `E: [1s(H1) - 1s(H3), 1s(H1) - 2 1s(H2) + 1s(H3)]`;
2. every valence orbital is a single-zeta Slater-type orbital (STO) with the
   standard extended-Hückel exponents and diagonal energies H_ii
   (valence-state ionization energies);
3. all two-center overlap integrals between the STOs are evaluated **exactly**
   (Gauss-Laguerre x Gauss-Legendre quadrature in prolate-spheroidal
   coordinates, machine precision — validated against the analytic H 1s-1s
   formula), so ligand-ligand overlap is *not* neglected; the
   ligand-SALC | central-AO overlap integrals are printed per irrep
   (NH3: `< a1 | N 2s > S = 0.72`, `< e | N 2p > S = 0.57`);
4. within each irrep block the generalized eigenvalue problem with the
   Wolfsberg-Helmholz off-diagonals H_ij = K S_ij (H_ii + H_jj)/2 (K = 1.75)
   is solved — a large overlap integral therefore directly produces a large
   bonding/antibonding splitting (symmetry-adapted extended Hückel);
5. the result is written as an interactive HTML/SVG diagram **by default**:
   four columns (isolated ligand AOs | ligand-group SALCs | molecular
   orbitals | central-atom AOs) with dashed correlation lines weighted by the
   orbital composition, electron arrows, HOMO/LUMO markers, and a click/hover
   panel showing the composition of every level in percent. The energy window is adjustable in the page itself (min/max input boxes, Ctrl/Cmd + scroll or pinch to zoom, drag to pan) -- useful when deep ligand levels stretch the scale. Hovering or clicking any level additionally shows an **orbital sketch** in the details panel — the molecule with VESTA-style +/− lobes (yellow/cyan) built from that level's actual eigenvector, drag-rotatable, with buttons to flip through degenerate partners.

The MO numbering counts the core shells as in photoelectron spectroscopy, so
CH4 gives the textbook `(2a1)^2 (1t2)^6` (1a1 = C 1s core; antibonding 2t2
and 3a1 empty), NH3 gives `(2a1)^2 (1e)^4 (3a1)^2` with the 3a1 lone pair as
HOMO, and SF6 fills 48 valence electrons up to the nonbonding F 2p block
(1t1g/3eg) below the 6a1g LUMO. The terminal report lists the AO parameters,
the ligand SALCs, the inter-fragment overlap integrals, and the full MO table
(energy, occupation, composition).

`--center EL` selects the central atom explicitly (default: the atom closest
to the molecular center); `--output`/`--tolerance` as usual. Elements H-Cl
are parameterized. The energies are semi-quantitative (extended Hückel); the
intended use is the *crystal-orbital diagram* counterpart for materials
design, where the same symmetry + overlap logic connects crystal SALCs
({doc}`crystod`) to band structures.

References: M. Wolfsberg and L. Helmholz, J. Chem. Phys. 20, 837 (1952);
R. Hoffmann, J. Chem. Phys. 39, 1397 (1963).

### Quantitative diagrams with PySCF (`--pyscf`)

`--diagram --pyscf` replaces the extended-Hückel estimate by **three
self-consistent-field calculations sharing one AO space** (PySCF): the full
molecule, and the two fragments with **ghost basis functions** on the removed
atoms — i.e. counterpoise-consistent, so the fragment levels are true
pre-bonding states in the molecular basis, the molecular MOs can be
**projected exactly** onto the fragment MOs (correlation lines and
composition panel), and the printed interaction energy
E(mol) − E(left) − E(right) is BSSE-free. The diagram becomes three columns:
left fragment | molecule | right fragment. The hover panel shows the orbital
sketch of every level (fragment MOs included), drawn from the PySCF MO
coefficients; fragment sketches show the real atoms only (the small
counterpoise tails on the ghost basis are omitted) and degenerate
partners are canonicalized, so they match the visualized SALCs of
`--visualize` exactly.

```bash
crystod-mol --diagram --xyz XYZ_H2O.xyz --pyscf                # H2 | H2O | O
crystod-mol --diagram --xyz XYZ_O2.xyz --pyscf --spin 2 --ao-left O --ao-right O
crystod-mol --diagram --xyz XYZ_CH3OH.xyz --pyscf --ao-left H4 --ao-right CO
crystod-mol --diagram --xyz XYZ_C6H6.xyz --pyscf --ao-left H6 --ao-right C6
```

- The default fragments are the ligand cage | central atom, as in the
  symmetry-only mode; `--ao-left`/`--ao-right` select **any partition by
  chemical formula** — required for molecules without a single center
  (benzene `H6`/`C6`) and for homonuclear diatomics (`O` / `O`, displayed as
  O(L)/O(R)).
- Fragments are **spin/spherically averaged** (fractional occupation of
  degenerate frontier shells): a partially filled shell (C 2p², the t2² of an
  H4 cage) would otherwise break the point-group symmetry, whereas the
  pre-bonding reference should keep it.
- Irrep labels come from crystod's own point-group machinery — the characters
  of every MO under the exact character-table operations, evaluated with the
  symmetry representation on the PySCF AO basis — so the labels agree with
  the symmetry-only diagram and with `crystod-group` (CH4 `1t2` HOMO, H2O
  `1b2`, benzene `1e1g` HOMO / `1e2u` LUMO at the Koopmans level); linear
  molecules use PySCF's Dooh/Coov labels rendered as σ/π/δ (triplet O2:
  `(1πu)^4 (1πg)^2`, two single arrows on the half-filled 1πg).
- `--basis` (default def2-svp), `--theory scf|dft`, `--xc` (default b3lyp),
  `--charge`, and `--spin` (2S; default by electron parity, e.g. `--spin 2`
  for triplet O2) follow the conventions of `script/calc_pyscf.py`;
  open-shell calculations use ROHF/ROKS so the diagram keeps a single
  orbital-energy set. Core levels (O 1s at −560 eV, ...) are computed and
  included — the default energy window starts at the valence, pan down to see
  them. PySCF is an optional dependency (only needed for `--pyscf`).

References: Q. Sun et al., WIREs Comput. Mol. Sci. 8, e1340 (2018);
Q. Sun et al., J. Chem. Phys. 153, 024109 (2020).

