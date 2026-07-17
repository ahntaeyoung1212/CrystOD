# crystod-mol

Molecular point-group detection and molecular SALCs — the molecular
counterpart of the crystalline analyses, working on molecules (XYZ files)
instead of periodic structures.

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
