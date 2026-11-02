# CrystOD for phonopy users

`crystod-phonon` runs on the files a phonopy calculation leaves behind and
adds the symmetry layer: which irrep every mode belongs to, in the labels of
the space-group tables, and which structures an imaginary mode can lower the
crystal to. Nothing has to be recomputed on the phonopy side, and the phonopy
API objects can be handed to CrystOD directly from Python.

## What you already have

| phonopy file | how `crystod-phonon` uses it |
|---|---|
| unit cell (`POSCAR`) | `-c POSCAR` |
| `FORCE_SETS` | read by default from the current directory, as phonopy does (`--modulation` also looks next to the `-c` file, and `--unfold` reads the child's and the parent's next to their POSCARs; the other modes stop with a one-line error when it is missing) |
| `FORCE_CONSTANTS` | `--readfc` |
| the supercell (`DIM`) | `--dim "4 4 4"` (or `--dim 4 4 4`; a nine-value diagonal matrix works too) |
| `phonopy_disp.yaml` | supplies the supercell to `--modulation` when `--dim` is omitted |
| `phonopy_params.yaml` | `--yaml phonopy_params.yaml`, in place of `-c` + `--dim` in `--modulation` and `--subgroup` |
| `BORN` | read only with `--nac`, in every mode that reads force data except `--unfold` (`--irreps`, `--fatband`, `--lt`, `--vector`, `--modulation`, `--subgroup`; with `--yaml` the NAC parameters stored in the yaml come first); phonopy itself reads `./BORN` by default. With `--irreps --nac` it also gives the mode effective charges and the static dielectric tensor |

The unit cell is reduced to its primitive cell (`primitive_matrix="auto"`),
so q points are given in primitive reciprocal coordinates, either as
three numbers or as a special-point label (`GM`, `R`, `X`, `M`, ...).

A unit cell plus 4x4x4 `FORCE_SETS` of cubic SrTiO3 is bundled with the
package, and `--example` copies them into the current directory and runs the
ordinary command line, so everything on this page can be tried before your own
data is at hand:

```bash
crystod-phonon --example
```

```
Examples bundled with crystod-phonon (run one with --example NAME):

  SrTiO3           ISO-IR irrep labels of the phonon modes of cubic SrTiO3 (writes phonon_irreps.yaml)
                   = crystod-phonon --irreps --dim '4 4 4' -c 221_PPOSCAR_SrTiO3
  SrTiO3_subgroup  space groups the imaginary R-point phonon of cubic SrTiO3 can condense into
                   = crystod-phonon --subgroup --dim '4 4 4' -c 221_PPOSCAR_SrTiO3 --qpoint R

More examples, a directory per feature with its README and inputs: crystod-example
```

## Irreps at every special point (`--irreps`)

phonopy's `--irreps` (the `IRREPS` tag) works one q point per run, and
assigns Mulliken labels of the point group (`T1u`, `Eg`, ...) at the Gamma
point only; elsewhere it prints the characters. `crystod-phonon --irreps`
labels every degenerate level at every special q point of the space group in
one run, with the ISO-IR labels (`GM4-`, `R5-`, `X5+`, ... - the Miller-Love
convention of the ISOTROPY Software Suite, `GM4-` being `T1u` in m-3m). These
are the labels the isotropy-subgroup tables and the direct-product tables use,
so a level can be carried straight into `crystod-group --parent` or
`--product`.

```bash
mkdir phonopy-guide && cd phonopy-guide
crystod-phonon --example SrTiO3
```

```
Wrote 221_PPOSCAR_SrTiO3 (bundled example input)
Wrote FORCE_SETS (bundled example input)
Running: crystod-phonon --irreps --dim '4 4 4' -c 221_PPOSCAR_SrTiO3

* Gamma-point activity *
GM4- [T1u] x4 (3 IR, 1 acoustic)
GM5- [T2u] x1 (silent)

* Band path *
  k path (seekpath): GM-X-M-GM-R-X | R-M  (306 q points)
  Labeled points: X, GM, R, M

* Output files *
  Phonon irreps written to: phonon_irreps.yaml
  Phonon dispersion with irrep labels written to: phonon_irreps.pdf
```

The result is the file, `phonon_irreps.yaml` (`-o FILE` writes it to another
path; the dispersion with the labels, `phonon_irreps.pdf`, is described
below), a YAML file: the special points of the space group, then for each of
them the modes grouped into degenerate
sets (`band_sets`), with their irrep and frequency in THz, and at Gamma the
Mulliken symbol and the infrared/Raman activity:

```yaml
space_group: Pm-3m
nac: false
special_points:
- q_label: GM
  q_position: [0.0, 0.0, 0.0]
- q_label: R
  q_position: [0.5, 0.5, 0.5]
- q_label: X
  q_position: [0.0, 0.5, 0.0]
- q_label: M
  q_position: [0.5, 0.5, 0.0]

irreps:
- q_label: GM
  q_position: [0.0, 0.0, 0.0]
  activity_summary: GM4- x4 (3 IR, 1 acoustic), GM5- x1 (silent)
  band_sets:
  - band_indices: [1, 2, 3]
    irrep_label: ['GM4-(3)']
    mulliken: T1u
    frequency:   0.0000003685
    activity: [acoustic]
  - band_indices: [4, 5, 6]
    irrep_label: ['GM4-(3)']
    mulliken: T1u
    frequency:   2.6629186664
    activity: [IR]
  - band_indices: [7, 8, 9]
    irrep_label: ['GM4-(3)']
    mulliken: T1u
    frequency:   4.7144147359
    activity: [IR]
  - band_indices: [10, 11, 12]
    irrep_label: ['GM5-(3)']
    mulliken: T2u
    frequency:   7.3464877515
    activity: [silent]
  - band_indices: [13, 14, 15]
    irrep_label: ['GM4-(3)']
    mulliken: T1u
    frequency:  15.3054117938
    activity: [IR]

- q_label: R
  q_position: [0.5, 0.5, 0.5]
  band_sets:
  - band_indices: [1, 2, 3]
    irrep_label: ['R5-(3)']
    frequency:  -1.0867090338
  - band_indices: [4, 5, 6]
    irrep_label: ['R4-(3)']
    frequency:   3.9890931052
  ...
```

At Gamma the 15 modes of the five-atom cell are 4 x `GM4-` (T1u: the acoustic
triplet plus the three infrared-active modes) + 1 x `GM5-` (T2u, the silent
mode), the textbook count for a cubic perovskite. At R the lowest level is a
negative-frequency `R5-` triplet: the antiferrodistortive octahedral-rotation
instability of cubic SrTiO3, which the rest of this page follows.
`band_indices` are 1-based, the numbering of phonopy's `band.yaml` and the
one `--vector --mode` and `--modulation --mode` take. Any YAML library reads
the file; `crystod.phonon.read_phonon_irreps("phonon_irreps.yaml")` returns it
as a dictionary (the keys are listed in
[28. Phonon irreps](crystod-phonon.md#28-phonon-irreps---irreps)).

`--all-irreps` also labels the midpoint of every seekpath path segment, i.e.
the symmetry lines (`DT`, `Z`, `SM`, `LD`, `S`, `T` for Pm-3m), and writes
`phonon_irreps_all.yaml` instead, so the two surveys coexist:

```bash
crystod-phonon --irreps --dim "4 4 4" -c 221_PPOSCAR_SrTiO3 --all-irreps
# -> phonon_irreps_all.yaml
```

```yaml
k_path: GM-X-M-GM-R-X | R-M  # seekpath
path_midpoints:  # midpoints of the k-path segments, ISO-IR k-vector types
- q_label: DT
  segment: GM-X
  q_position: [0.0, 0.25, 0.0]
- q_label: Z
  segment: X-M
  q_position: [0.25, 0.5, 0.0]
  ...

- q_label: DT
  segment: GM-X
  q_position: [0.0, 0.25, 0.0]
  band_sets:
  - band_indices: [1, 2]
    irrep_label: ['DT5(2)']
  ...
```

This is what tells two branches of the same line apart (which `DT5` is which)
and takes a few times longer than the special-points-only default (about 10 s
against 2 s for this example). `--readfc` reads `FORCE_CONSTANTS` in place of
`FORCE_SETS` in every mode.

Every `--irreps` run also draws the picture: the dispersion along the
seekpath path (or `--band`), as `phonopy -p` draws it, with the label of
every level at the special points and the imaginary levels in red, in
`phonon_irreps.pdf` next to the YAML file (the YAML name with `.pdf`):

```bash
crystod-phonon --irreps --dim "4 4 4" -c 221_PPOSCAR_SrTiO3
# -> phonon_irreps.yaml, phonon_irreps.pdf
```

Documentation: [28. Phonon irreps](crystod-phonon.md#28-phonon-irreps---irreps),
[Symmetry lines as well](crystod-phonon.md#symmetry-lines-as-well---all-irreps),
[The dispersion with the labels](crystod-phonon.md#the-dispersion-with-the-labels-phonon_irrepspdf).

## Imaginary modes: which structures? (`--subgroup`)

An imaginary mode says the structure is unstable; its irrep says what it can
turn into. `--subgroup` labels the imaginary levels and lists the isotropy
subgroups of their irreps, i.e. every order-parameter direction of the
degenerate level and the space group it condenses into:

```bash
crystod-phonon --example SrTiO3_subgroup
```

```
Kept 221_PPOSCAR_SrTiO3 (identical to the bundled example input)
Kept FORCE_SETS (identical to the bundled example input)
Running: crystod-phonon --subgroup --dim '4 4 4' -c 221_PPOSCAR_SrTiO3 --qpoint R

* Parent structure *
Pm-3m (No. 221)

* Imaginary mode at q = (0.5, 0.5, 0.5) (R) *
mode 1, 2, 3: -1.086709 THz, irrep R5- (degeneracy 3)

* Order parameter directions and isotropy subgroups *
irrep                subgroup           size  index
R5-(0,0,a)           140 I4/mcm         2     6
R5-(a,a,a)           167 R-3c           2     8
R5-(0,a,a)           74 Imma            2     12
R5-(0,a,b)           12 C2/m            2     24
R5-(a,a,b)           15 C2/c            2     24
R5-(a,b,c)           2 P-1              2     48

The distortion of each order-parameter direction can be generated with
crystod-phonon --modulation, or by adding --modulate here.
```

This is the R-point story of SrTiO3 in one table. The `R5-` triplet is the
rotation of the TiO6 octahedra about one, two or three cubic axes: rotating
about one axis, `(0,0,a)`, gives the tetragonal `I4/mcm` phase that SrTiO3
actually adopts below 105 K; equal rotations about all three, `(a,a,a)`, give
`R-3c` (the LaAlO3 structure); two axes, `(0,a,a)`, give `Imma`; and the
unequal combinations `(0,a,b)`, `(a,a,b)`, `(a,b,c)` lower the symmetry to
`C2/m`, `C2/c` and `P-1`. `size` is the primitive cell of the daughter phase
in units of the parent one, `index` the index of the subgroup. Freezing in a
single eigenvector, the way a MODULATION run does, explores one of these six
directions, and which one depends on the eigenvector the diagonalizer happened
to return for the degenerate level.

Without `--qpoint` every q point commensurate with the supercell is scanned
(64 of them for 4x4x4), most unstable level first, so the unstable q points do
not have to be known beforehand; for this force set only the R triplet turns
up. `--threshold` sets what counts as imaginary (default `-0.1` THz), and
`--yaml phonopy_params.yaml` replaces `-c` + `--dim`.

`--modulate` writes the distorted structure of every direction as well
(`--example` passes further options on):

```bash
crystod-phonon --example SrTiO3_subgroup --modulate
```

```
* Distorted structures (amplitude 0.3 A) *
R5-(0,0,a)             I4/mcm     -> MPOSCAR_R_R5-_0-0-a_I4mcm
    crystod-phonon --modulation -c 221_PPOSCAR_SrTiO3 --dim "4 4 4" --qpoint 0.5 0.5 0.5 --mode 1 --amplitude 0.3
R5-(0,a,a)             Imma       -> MPOSCAR_R_R5-_0-a-a_Imma
    crystod-phonon --modulation -c 221_PPOSCAR_SrTiO3 --dim "4 4 4" --qpoint 0.5 0.5 0.5 --mode 1 2 --amplitude 0.3 0.3
R5-(0,a,b)             C2/m       -> MPOSCAR_R_R5-_0-a-b_C2m
    crystod-phonon --modulation -c 221_PPOSCAR_SrTiO3 --dim "4 4 4" --qpoint 0.5 0.5 0.5 --mode 1 2 --amplitude 0.3 0.16434
R5-(a,a,a)             R-3c       -> MPOSCAR_R_R5-_a-a-a_R-3c
    crystod-phonon --modulation -c 221_PPOSCAR_SrTiO3 --dim "4 4 4" --qpoint 0.5 0.5 0.5 --mode 1 2 3 --amplitude 0.3 0.3 0.3
R5-(a,a,b)             C2/c       -> MPOSCAR_R_R5-_a-a-b_C2c
    crystod-phonon --modulation -c 221_PPOSCAR_SrTiO3 --dim "4 4 4" --qpoint 0.5 0.5 0.5 --mode 1 2 3 --amplitude 0.3 0.3 0.16434
R5-(a,b,c)             P-1        -> MPOSCAR_R_R5-_a-b-c_P-1
    crystod-phonon --modulation -c 221_PPOSCAR_SrTiO3 --dim "4 4 4" --qpoint 0.5 0.5 0.5 --mode 1 2 3 --amplitude 0.3 0.16434 0.08913
```

Six POSCAR files, one per direction, named after the subgroup, each with the
`--modulation` command that reproduces it; the space group of every file is
measured with spglib, not assumed. These are the candidate structures to relax
in a structure search. `--amplitude` scales all of them (default 0.3 A).
Documentation: [35. Subgroups from imaginary modes](crystod-phonon.md#35-subgroups-from-imaginary-modes---subgroup),
[Generating the daughter structures](crystod-phonon.md#generating-the-daughter-structures---modulate),
and the theory in [Isotropy subgroups](theory-isotropy-subgroups.md).

## `--modulation` versus phonopy's `MODULATION`

phonopy's `MODULATION` tag (`MODULATION = 2 2 2, 1/2 1/2 1/2 1 0.3`:
supercell, q point, band index, amplitude) freezes one eigenvector of the
dynamical matrix into `MPOSCAR`. For a non-degenerate mode that is all there
is to it. For a degenerate level, the eigenvector behind a band index is
whatever the diagonalizer returned, an arbitrary combination within the level,
and the structure it produces has no defined symmetry. `crystod-phonon
--modulation` block-diagonalizes the dynamical matrix in the irrep-projected
basis first, so that each mode number is one component of the order parameter:
`--mode 1` of the R triplet is the `(0,0,a)` rotation, `--mode 1 2 3` with
equal amplitudes is `(a,a,a)`. It also prints the irrep and degeneracy of every
mode, the star of q, and the space group of the structure it wrote, which it
names after that space group.

What is frozen in is the same physical displacement: phonopy's eigenvector
divided by the square root of the atomic mass, with its Bloch phase, so for a
non-degenerate mode at a time-reversal-invariant q the pattern is the one
`MODULATION` writes, up to its sign. Only the amplitude is counted differently.
phonopy multiplies that field by its amplitude divided by the square root of
the number of supercell atoms (a quantity in Å·√amu); `--amplitude` is in Å,
the norm of the displacement of one primitive cell. See
[What is frozen in](crystod-phonon.md#what-is-frozen-in).

Without `--mode`, the mode table is printed so the modes can be chosen:

```bash
crystod-phonon --modulation -c 221_PPOSCAR_SrTiO3 --qpoint 0.5 0.5 0.5
```

```

* Phonon source *
  Supercell 4x4x4 inferred from the 320 atoms of FORCE_SETS; pass --dim if that is not the supercell of your force calculation. Primitive cell: 5 atoms of the 5-atom input cell (primitive_matrix auto).

* Selected Q point *
  Loading '221_PPOSCAR_SrTiO3 + FORCE_SETS' at q = [0.5, 0.5, 0.5]...

* Phonon modes at q = [0.5 0.5 0.5] *
 Mode    Freq (THz)         Irrep   Degeneracy
--------------------------------------------------
    1       -1.0867        R5-(3)            3
    2       -1.0867        R5-(3)            3
    3       -1.0867        R5-(3)            3
    4        3.9891        R4-(3)            3
    5        3.9891        R4-(3)            3
    6        3.9891        R4-(3)            3
    7       11.6887        R5+(3)            3
    ...
   15       23.2302        R2-(1)            1

* Star of q (arms related by the space-group rotations) *
  |G| = 48, |G_k| = 48, |star of k| = 1
  arm 1: k = [+0.5, +0.5, +0.5]

No --mode given. Choose mode number(s) from the table above and rerun with
--mode (and optionally --amplitude) to generate a modulated structure.
```

Then with the modes:

```bash
crystod-phonon --modulation -c 221_PPOSCAR_SrTiO3 --qpoint 0.5 0.5 0.5 --mode 1 2 3 --amplitude 0.3
```

```
...

* Modulation *
  q-point: [0.5, 0.5, 0.5]
  Modes: [1, 2, 3]
  Amplitudes (A): [0.3, 0.3, 0.3]

* Symmetry of the generated structure *
  Space group: R-3c (#167)
  Hall symbol: -R 3 2"c

* Output files *
  Modulated structure written to: MPOSCAR_R_mode1+2+3_R5-_R-3c
```

Note that `--dim` was not given: `--modulation` takes the supercell from
`--dim` if present, else from the `supercell_matrix` of a `phonopy_disp.yaml`
(or `phonopy_params.yaml`) in the current directory, else from the atom count
of `FORCE_SETS`, and always prints which. `--yaml phonopy_params.yaml` in
place of `-c` writes the same file byte for byte. Several q points combine
through `--qpoint1/--mode1/--amplitude1`, `--qpoint2/...`, which is how the
arms of a multi-arm star (the three M points of a perovskite) are frozen in
together. Any arm, written as any `q + G`, is accepted: `-0.5 0.5 0` and
`0.5 0.5 0` differ by a reciprocal lattice vector and give one structure, and
`--qpoint` and `--qpointN` also take special-point labels (`--qpoint R`).
Where `MODULATION` takes the supercell you give it, `--modulation` builds the
supercell of the modulated structure from q, with at most 12 cells per axis:
give q as fractions with denominators up to 12; a q such as `0.15 0 0`
(3/20) is refused with an ERROR that names the smallest commensurate
supercell (`20x1x1`).
Documentation: [33. Phonon modulation](crystod-phonon.md#33-phonon-modulation---modulation),
[Where the supercell comes from](crystod-phonon.md#where-the-supercell-comes-from).

## Fatbands, L/T character and eigenvectors

`--fatband` draws the dispersion along the seekpath path with the
element-projected weight of every mode as the dot size, one PDF per element in
the VESTA element colors, straight from `-c` + `FORCE_SETS`; no `band.conf`
or `band.yaml` is needed. `--element O` restricts it to one element,
`--band`/`--band-labels` give a manual path, and `--nac` reads `BORN` for the
LO/TO splitting (the files are written as `fatband_nac_<El>.pdf` and
`phonon_band_LT_nac.pdf`, so both versions coexist).
Documentation: [30. Phonon fatbands](crystod-phonon.md#30-phonon-fatbands---fatband).

```bash
crystod-phonon --fatband --dim 4 4 4 -c 221_PPOSCAR_SrTiO3
```

```

* Structure *
  Space group: Pm-3m (#221)

* Band path *
  k-path (seekpath): $\Gamma$-X-M-$\Gamma$-R-X  R-M
  Computing phonon band structure with eigenvectors (306 q-points)...

* Output files *
  Fatband for Sr written to: fatband_Sr.pdf
  Fatband for Ti written to: fatband_Ti.pdf
  Fatband for O written to: fatband_O.pdf
```

`--lt` draws the same dispersion colored by the longitudinal/transverse
character of each mode, the norm of the eigenvector projected onto the
propagation direction (red = longitudinal, blue = transverse, white = mixed),
valid along diagonal path segments as well; with `--nac` the split-off LO
branches come out purely red.
Documentation: [31. Longitudinal/transverse bands](crystod-phonon.md#31-longitudinaltransverse-bands---lt).

```bash
crystod-phonon --lt --dim 4 4 4 -c 221_PPOSCAR_SrTiO3
# -> phonon_band_LT.pdf
```

`--vector` diagonalizes the dynamical matrix at one q point, prints the mode
table with irrep labels, and exports the chosen modes as `.vesta` files with
displacement arrows; `--mode` takes several numbers and sums them, degenerate
levels are exported in the same symmetry-adapted form as `--modulation`, and
`--conventional` writes the conventional cell. For a zone-boundary q point the
commensurate supercell and the Bloch phases are applied automatically:
Documentation: [32. Phonon eigenvectors](crystod-phonon.md#32-phonon-eigenvectors---vector).

```bash
crystod-phonon --vector --dim "4 4 4" -c 221_PPOSCAR_SrTiO3 --qpoint R --mode 1 2 3
```

```
...

* Selected Q point *
  Selected q-point: R = [0.5, 0.5, 0.5]

* Phonon modes at q = R *
 Mode    Freq (THz)  Irrep
----------------------------------------
    1       -1.0867  R5-(3)
    2       -1.0867  R5-(3)
    3       -1.0867  R5-(3)
    4        3.9891  R4-(3)
    ...

* Displacement patterns *
  Commensurate supercell for visualization: 2x2x2 primitive cells
  + mode 1: R5-(3), -1.0867 THz
  + mode 2: R5-(3), -1.0867 THz
  + mode 3: R5-(3), -1.0867 THz
  Arrows are scaled so the largest displacement is 1.5 A; adjust arrow size in VESTA via Edit > Vectors or Properties > Vectors if needed.

* Output files *
  Mode table written to: phonon_modes_SrTiO3_R.txt
  Sum of modes 1+2+3 written to: POSCAR_SrTiO3_R_mode01+02+03_R5-.vesta
```

`--vibration` needs no forces at all: it lists the
irrep-grouped vibration spaces at a q point from the crystal symmetry alone
([34. Vibration bases](crystod-phonon.md#34-vibration-bases---vibration)).

`--visualize` is the interactive counterpart of `--vector`, and of the
`band.yaml` that phonopy writes with `--eigvecs`: one run writes that file
along the `--irreps` path, as `band_vector.yaml` (phonopy's format, an
`eigenvector` of `[re, im]` pairs per atom for every band, labels `GM`, `X`,
...), and a standalone HTML page that shows every mode at the special points
of the path: the dispersion with a dot on every level, the mode table with
the ISO-IR set labels of `phonon_irreps.yaml` (Mulliken symbol and activity
at GM; imaginary levels in red) and a 3D scene with the symmetry-adapted
displacement pattern of the selected mode as arrows, on the commensurate
supercell, with an animation of the vibration and, with `--bond Ti O 2.1`,
the coordination polyhedra.
Documentation: [52. Phonon mode viewer](crystod-phonon.md#52-phonon-mode-viewer---visualize).

```bash
crystod-phonon --visualize --dim 4 4 4 -c 221_PPOSCAR_SrTiO3 --bond Ti O 2.1
# -> phonon_visualize.html, band_vector.yaml
```

## Band unfolding (`--unfold`) and phonopy's `Unfolding`

phonopy's Python API unfolds bands with `phonopy.unfolding.core.Unfolding`;
`crystod-phonon --unfold` computes the same projection (P. B. Allen et al.,
Phys. Rev. B 87, 085322 (2013)) from the files of two phonopy runs, one for
the distorted child and one for its high-symmetry parent, each `FORCE_SETS`
(or `FORCE_CONSTANTS`, with `--child-readfc` / `--parent-readfc`) read next to
its POSCAR:

```bash
crystod-phonon --unfold --child-poscar child/62_PPOSCAR_CaTiO3 --child-dim 2 2 2 \
               --parent-poscar parent/221_PPOSCAR_CaTiO3 --parent-dim 2 2 2
# -> phonon_unfold.pdf, phonon_unfold.dat
```

It differs from `Unfolding` in four ways:

- The child eigenvectors are converted to the ideal-site phase before the
  projection. `Unfolding` keeps the Bloch phase of the actual atomic
  positions, so for a distorted child its weights depend on the supercell the
  force constants are written in: the force constants of the example (Pnma
  CaTiO3 and R-3c AlF3, 2x2x2 `FORCE_SETS`), written in a 3x3x3 instead of
  the 2x2x2 supercell (`Phonopy.ph2ph`), change its weights by up to 0.10 and
  0.22, and those of `--unfold` by 6e-12 at most (weights of the degenerate
  groups at the 306 k of the example path). On the 2x2x2 supercell its
  weights differ from those of `--unfold` by up to 0.07 and 0.14; with the
  ideal-site phase added they agree to 1e-11.
- The transformation matrix and the origin shift are not inputs: they are the
  setting `crystod-group --supergroup-cif PARENT --subgroup-cif CHILD` finds
  for the two structures, and `--trans-mat` (rows, child = T @ parent) picks
  another of the equivalent settings.
- The child is diagonalized in its own cell (60 x 60 for CaTiO3) instead of
  the force-constant supercell, so the 306 k of the example take 0.07 s
  rather than about 22 s (`Unfolding` on the 2x2x2 supercell; about 700 s on
  the 3x3x3 one).
- Every unfolded level gets its parent irrep, and the run adds the arms of
  the stars of the path vertices, the average over the orientation domains,
  and the parent k and irreps the child Γ modes come from.

Documentation: [51. Phonon band unfolding](crystod-phonon.md#51-phonon-band-unfolding---unfold).

## The Python API on a live Phonopy object

Everything above is also a function of `crystod.phonon`, taking the
`Phonopy` object you already have. `label_phonon_modes` is `--irreps` without
the YAML file, `imaginary_mode_subgroups` is `--subgroup` at one q point, and
`scan_imaginary_modes` is the full scan:

```python
import phonopy
from crystod.phonon import label_phonon_modes, imaginary_mode_subgroups

ph = phonopy.load(supercell_matrix=[4, 4, 4], primitive_matrix="auto",
                  unitcell_filename="221_PPOSCAR_SrTiO3",
                  force_sets_filename="FORCE_SETS")

for mode in label_phonon_modes(ph, [0.5, 0.5, 0.5]):
    print(mode)

for result in imaginary_mode_subgroups(ph, [0.5, 0.5, 0.5]):
    mode = result.mode
    print(f"{mode.frequency:.4f} THz  {'+'.join(mode.labels)}  (degeneracy {mode.degeneracy})")
    for sub in result.subgroups:
        print("   ", sub.label, "->", sub.symbol)
```

```
modes 1,2,3: -1.0867 THz  R5-
modes 4,5,6: 3.9891 THz  R4-
modes 7,8,9: 11.6887 THz  R5+
modes 10,11,12: 12.6621 THz  R4-
modes 13,14: 14.8133 THz  R3-
modes 15: 23.2302 THz  R2-
-1.0867 THz  R5-  (degeneracy 3)
    R5-(0,0,a) -> I4/mcm
    R5-(a,a,a) -> R-3c
    R5-(0,a,a) -> Imma
    R5-(0,a,b) -> C2/m
    R5-(a,a,b) -> C2/c
    R5-(a,b,c) -> P-1
```

Each `PhononMode` carries `band_indices` (1-based), `frequency`, `labels`,
`qpoint`, `qpoint_label`, `degeneracy` and `is_imaginary`; a q point given as
another arm of the star (`[-0.5, 0.5, 0.5]`) is mapped onto the tabulated one.
`scan_imaginary_modes(ph)` runs the same over every commensurate q point
(`commensurate_qpoints(ph)`, 64 here), most unstable level first, and a level
whose irrep has no tabulated subgroups comes back with the reason in
`result.errors` instead of stopping the scan:

```python
from crystod.phonon import scan_imaginary_modes, commensurate_qpoints

len(commensurate_qpoints(ph))          # 64
for result in scan_imaginary_modes(ph):
    print(result.mode, [sub.symbol for sub in result.subgroups], result.errors)
```

```
modes 1,2,3: -1.0867 THz  R5- ['I4/mcm', 'R-3c', 'Imma', 'C2/m', 'C2/c', 'P-1'] {}
```

```{note}
Build the `Phonopy` object with `primitive_matrix="auto"`, as above. With an
identity primitive matrix on a supercell-sized unit cell, a zone-boundary
instability folds onto the supercell Gamma point, where no irrep of the parent
space group can be attached to it. The `-c` route of the command line always
uses `"auto"`.
```

Documentation: [Phonon modes and their irreps](python-api.md#phonon-modes-and-their-irreps),
[From imaginary phonons to subgroups](python-api.md#from-imaginary-phonons-to-subgroups).

## Where next

- [Python API](python-api.md): the full API page, including
  `crystod.group.isotropy_subgroups` for the subgroups of any irrep without a
  phonon calculation.
- [Tutorials](tutorials.md): notebook 01 (phonon irrep labeling, with the
  labels drawn on a matplotlib dispersion) and notebook 02 (the isotropy
  subgroup search, including the experiment that shows what freezing one
  eigenvector at a time misses).
- [crystod-phonon](crystod-phonon.md): the reference of every mode flag, and
  [17. Isotropy subgroups](crystod-group.md#17-isotropy-subgroups---parent)
  for the same tables from `crystod-group --parent Pm-3m --irrep R5-`.
