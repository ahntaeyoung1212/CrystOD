# Your first analysis

This page walks through one crystal, cubic ScF3, from the structure file to the
crystal-orbital diagram in four commands: the irreps of a shell, the orbitals
that are allowed to hybridize, the diagram itself, and the SALCs in 3D. The
input is bundled with the package, so `pip install CrystOD` is all that has to
have happened ([Installation](install.md)); nothing needs to be cloned or
downloaded.

Every command below is shown twice: once through `--example`, which copies the
bundled input into the current directory and runs the ordinary command line,
and once as that ordinary command line, which is what you type for a structure
of your own (`-c POSCAR`; the file is read in VASP POSCAR format, as VASP
itself reads it: direct or Cartesian coordinates, and a scale factor or a
negative cell volume on line 2).

## 0. The bundled inputs

Run `crystod --example` without a name to see what ships with the main command:

```bash
crystod --example
```

```
Examples bundled with crystod (run one with --example NAME):

  ScF3_d        Sc 3d crystal-orbital (SALC) irreps of cubic ScF3 at every special k point
                = crystod -c 221_PPOSCAR_ScF3 --element Sc --orbital d
  SrTiO3_d      Ti 3d crystal-orbital irreps of cubic SrTiO3 (eg/t2g splitting at Gamma)
                = crystod -c 221_PPOSCAR_SrTiO3 --element Ti --orbital d
  ScF3_diagram  extended-Hueckel crystal-orbital diagram of ScF3, one HTML page per k point
                = crystod --diagram -c 221_PPOSCAR_ScF3 --co-left Sc --co-right F3

More examples, a directory per feature with its README and inputs: crystod-example
```

`--example NAME` writes the input file(s) of that example into the current
directory, prints the equivalent command line after `Running:`, and runs it.
A file that is already there and identical is kept; one that differs is never
overwritten (the run stops with an error naming it), so run the examples in an
empty directory. `crystod-phonon`, `crystod-mol`, `crystod-bz`, `crystod-xrd`,
`crystod-search` and `crystod-slab` have `--example` too. The last line points at
{doc}`crystod-example`, which copies the worked example directories of every
feature (a README with commands and output, and the inputs) out of the
package.

## 1. Which irreps does the Sc 3d shell span?

```bash
mkdir first-analysis && cd first-analysis
crystod --example ScF3_d
```

```
Wrote 221_PPOSCAR_ScF3 (bundled example input)
Running: crystod -c 221_PPOSCAR_ScF3 --element Sc --orbital d

 ### Inputed cell was converted into primitive cell. ###

* Space group *
 Pm-3m (221)

* Element (number of atoms) *
 Sc (1)

* Wyckoff letters and site symmetry letters *
 ['a']
 ['m-3m']

* Atomic Orbital *
 d

* Crystal Orbitals *
 k point (primitive):  GM [0.0, 0.0, 0.0]
 little group of k  :  Pm-3m (221)
 irreps             :  1.0 [GM3+(2)] + 1.0 [GM5+(3)]

 k point (primitive):  R [0.5, 0.5, 0.5]
 little group of k  :  Pm-3m (221)
 irreps             :  1.0 [R3+(2)] + 1.0 [R5+(3)]

 k point (primitive):  X [0.0, 0.5, 0.0]
 little group of k  :  P4/mmm (123)
 irreps             :  1.0 [X1+(1)] + 1.0 [X2+(1)] + 1.0 [X4+(1)] + 1.0 [X5+(2)]

 k point (primitive):  M [0.5, 0.5, 0.0]
 little group of k  :  P4/mmm (123)
 irreps             :  1.0 [M1+(1)] + 1.0 [M2+(1)] + 1.0 [M4+(1)] + 1.0 [M5+(2)]
```

The five Sc *d* orbitals sit in an octahedron of F, and the symmetry alone
splits them into the e<sub>g</sub> pair (`GM3+`, `R3+`) and the t<sub>2g</sub>
triple (`GM5+`, `R5+`) wherever the little group is the full cubic group. At X
and M the little group is only tetragonal (`P4/mmm`), and the triple splits
further into a singlet and a doublet. The `(2)`/`(3)` after each label is the
dimension of the irrep, i.e. the degeneracy of that crystal-orbital level.

With your own structure:

```bash
crystod -c POSCAR --element Sc --orbital d              # every special k point
crystod -c POSCAR --element Sc --orbital d --kpoint 0 0 0
```

`--kpoint` takes three primitive reciprocal coordinates (fractions such as
`1/2` are allowed); without it, every special k point of the space group is
analyzed. Documentation: [2. Irreps of SALC](crystod.md#2-irreps-of-salc).

## 2. Which orbitals are allowed to hybridize?

Two shells can only mix where they share an irrep. `--atomic-orbital` lists,
for every little-group irrep at one k point, the shells that transform as it:

```bash
crystod -c 221_PPOSCAR_ScF3 --atomic-orbital Sc-d F-p --kpoint 0.5 0.5 0.5
```

```
* Atomic Orbital *
 ['Sc_d', 'F_p']

* k point (primitive) *
 R [0.5, 0.5, 0.5]

* Result *
 R1+(1): F(p)
 R1-(1):
 R2+(1):
 R2-(1):
 R3+(2): Sc(d) F(p)
 R3-(2):
 R4+(3): F(p)
 R4-(3):
 R5+(3): Sc(d) F(p)
 R5-(3):
```

Sc *d* and F *p* meet on two lines, `R3+` and `R5+`: these are the
sigma-type (e<sub>g</sub>) and pi-type (t<sub>2g</sub>) Sc-F interactions,
and each of them will produce a bonding and an antibonding crystal orbital in
the diagram of the next step. `R4+` carries F *p* alone, so that set stays
nonbonding whatever the energies are. `R1+` also lists F *p* alone here, but
only because the question was about Sc *d*: Sc 4s transforms as `R1+` too, and
the diagram of the next step shows the two mixing.
Documentation: [3. Crystal-orbital diagrams](crystod.md#3-crystal-orbital-diagrams).

## 3. The crystal-orbital diagram

```bash
crystod --example ScF3_diagram
# -> CrystOD_221_PPOSCAR_ScF3.html
```

```
Kept 221_PPOSCAR_ScF3 (identical to the bundled example input)
Running: crystod --diagram -c 221_PPOSCAR_ScF3 --co-left Sc --co-right F3

 ### Inputed cell was converted into primitive cell. ###

* Space group *
 Pm-3m (221)

* Fragments (full-electron basis: core + valence shells; atomic levels from reference/atomic_level_*) *
 left  Sc    : Sc 1s 2s 2p 3s 3p 4s 4p 3d x1 site(s), 21 electrons
 right F3    : F 1s 2s 2p x3 site(s), 27 electrons
 electrons per cell in the diagram: 48 (all electrons of the neutral atoms; override with --electrons)

* Ligand-field point charges (removed sublattice) *
 left  Sc     feels the F^-1 lattice (multipole ligand field + penetration; ...)
 right F3     feels the Sc^+3 lattice (multipole ligand field + penetration; ...)
 ...

* k point R (1/2,1/2,1/2) *
   (1 near-dependent diffuse Bloch combination(s) below overlap floor 0.2: energies marked ~ are first-order Loewdin estimates; the variational extended-Hueckel values diverge)
   Sc        : ... Sc 3d R5+ (-9.08), Sc 3d R3+ (-7.62), Sc 4p R4- (-5.76), Sc 4s R1+ (12.73)
   F3        : F 1s R4- (-717.49), F 2s R4- (-40.00), F 2p R1+ (-19.67), F 2p R3+ (-19.33), F 2p R5+ (-17.56), F 2p R4+ (-17.30)
   crystal   :
     ...
     R3+ #1        -19.46 eV  x2  4e   F 2p R3+ 93.5%  Sc 3d R3+ 6.5%
     R1+ #4        -19.38 eV  x1  2e   F 2p R1+ 88.3%  Sc 4s R1+ 11.4%  Sc 3s R1+ 0.3%
     R5+ #1        -17.70 eV  x3  6e   F 2p R5+ 95.4%  Sc 3d R5+ 4.6%
     R4+ #1        -17.30 eV  x3  6e   F 2p R4+ 100.0%
     R5+ #2         -8.18 eV  x3       Sc 3d R5+ 95.4%  F 2p R5+ 4.6%
     R1+ #5        ~-5.24 eV  x1       Sc 4s R1+ 87.1%  F 2p R1+ 10.7%  Sc 3s R1+ 2.2%
     R3+ #2         -5.21 eV  x2       Sc 3d R3+ 93.5%  F 2p R3+ 6.5%
     ...

* Dipole selection rules at R (1/2,1/2,1/2) *
   VBM R4+ #1 (-17.30 eV) -> CBM R5+ #2 (-8.18 eV): forbidden
   first allowed: R4- #4 (-38.94 eV) -> R5+ #2 (-8.18 eV), dE = 30.76 eV: allowed (x, y, z)
   (vertical transitions in the little group of k; polarizations in the Cartesian axes x, y, z of the input cell)
 ...

Crystal-orbital diagram written to CrystOD_221_PPOSCAR_ScF3.html
```

The two lines of step 2 have become level pairs: `R3+ #1`/`R3+ #2` and
`R5+ #1`/`R5+ #2` are the bonding and antibonding Sc *d*-F *p* combinations,
F-dominated below and Sc-dominated above, while `R4+ #1` is 100 % F 2p, the
nonbonding set. The 48 electrons per cell fill everything up to the F 2p
manifold (`4e`, `6e` in the occupation column), so the Sc 3d levels are the
empty conduction states of the d<sup>0</sup> compound. Each fragment here
feels the other sublattice as a lattice of point charges (F<sup>-1</sup>,
Sc<sup>+3</sup>), which is what splits the Sc 3d shell into `R5+` below `R3+`
already on the fragment side.

Each k point ends with the dipole selection rule of its band edge. At R the
top of the F 2p band (`R4+ #1`) and the bottom of the Sc 3d band (`R5+ #2`)
are both even under inversion, so light cannot drive the transition between
them (`forbidden`); the `first allowed:` line names the allowed occupied ->
empty pair of smallest energy difference, with its polarizations. On the
page, clicking two levels one after the other shows the same verdict
([4. Dipole selection rules](crystod.md#4-dipole-selection-rules---diagram)).

Open `CrystOD_221_PPOSCAR_ScF3.html` in a browser: the Sc fragment levels
stand on the left, the F3 fragment levels on the right, the crystal orbitals
with their irrep labels in the middle, joined to the fragment levels they are
made of; hovering a level shows a sketch of its orbital composition, and the
page carries one such diagram per special k point (GM, R, X, M). The page
written by this exact command is embedded in the
[`--diagram` section](crystod.md#3-crystal-orbital-diagrams) of the manual.

The engine here is extended Hückel with atomic levels from archived
neutral-atom calculations, which is why the command needs nothing but the
structure; the irrep labels and the allowed mixings are exact, the level order
is not always. The same command with `--pyscf` runs a periodic DFT calculation
instead and draws the diagram from it (this needs the optional dependency,
`pip install 'CrystOD[quantum]'`):

```bash
crystod --diagram --pyscf -c POSCAR --co-left Sc --co-right F3
```

## 4. The SALCs in 3D

The levels of the diagram are built from symmetry-adapted linear combinations
(SALCs) of the atomic orbitals. `--visualize` writes the ones of a chosen shell
at a chosen k point as an interactive 3D page:

```bash
crystod -c 221_PPOSCAR_ScF3 --element Sc --orbital d --kpoint R --visualize
# -> SALC_Sc_d_R.html
```

```
* k point (primitive) *
 R [0.5, 0.5, 0.5]

* Irreducible Decomposition *
 1.0 [R3+(2)] + 1.0 [R5+(3)]

* SALC basis functions (irrep-grouped) *
 Mode Space 1: irrep = R3+(2), dimension = 2
   component 1:
     Sc1 (atom 0): d_z2: +0.7071, d_x2-y2: (+0.0000+0.7071j)
   component 2:
     Sc1 (atom 0): d_z2: +0.7071, d_x2-y2: (+0.0000-0.7071j)

 Mode Space 2: irrep = R5+(3), dimension = 3
   component 1:
     Sc1 (atom 0): d_xy: +1.0000
   component 2:
     Sc1 (atom 0): d_yz: +1.0000
   component 3:
     Sc1 (atom 0): d_xz: +1.0000

* Output files *
 Saved 3D visualization to: SALC_Sc_d_R.html
```

The decomposition is the one of step 1, now with the coefficients: the
t<sub>2g</sub> SALCs are the plain `d_xy`, `d_yz`, `d_xz` orbitals, and the
e<sub>g</sub> pair comes out as the complex combinations
`d_z2 ± i d_x2-y2`; add `--real-coefficient` to get `d_z2` and `d_x2-y2`
themselves. In the page, the 2x2x2 supercell commensurate with R is shown and
the Bloch phase of R = (1/2, 1/2, 1/2) flips the sign of the lobes from one
cell to the next; click a row of the SALC table to switch the displayed basis
vector and drag to rotate (the page loads plotly.js from `cdn.plot.ly`, so
the browser needs a network connection to draw it, unlike the self-contained
diagram page of step 3). `--bond Sc F 2.5` adds the ScF6 octahedra, and
without `--kpoint` one page per special k point is written. In `--visualize`
mode `--kpoint` accepts a label (`GM`, `X`, `M`, `R`) as well as coordinates.
Documentation: [6. SALC basis visualization](crystod.md#6-salc-basis-visualization---visualize).

## Where next

- [CrystOD for phonopy users](for-phonopy-users.md): the same shape of
  analysis on phonopy data, from `phonon_irreps.yaml` to the space groups an
  imaginary mode can condense into.
- [Python API](python-api.md): every command above as a function
  (`crystod.salc`, `crystod.group`, `crystod.phonon`, ...), for use inside
  your own scripts.
- [Tutorials](tutorials.md): three Jupyter notebooks that run on the bundled
  inputs, with the API and the command line side by side.
- [crystod-search](crystod-search.md): the POSCAR of any compound of the
  Materials Project (`crystod-search SrTiO3`, then `crystod-search --get
  mp-5229`), to run the same analysis on a crystal of your choice.
- The full reference of the main command, section by section, is
  [crystod](crystod.md); the other commands are listed on the
  [front page](index.md#commands).
