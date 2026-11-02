# crystod-phonon

Phonon analyses on top of [phonopy](https://phonopy.github.io/phonopy/) data
(POSCAR + `FORCE_SETS`, or `FORCE_CONSTANTS` with `--readfc`, or
`phonopy_params.yaml`). Nine mode flags: `--irreps`, `--fatband`, `--lt`,
`--vector`, `--modulation`, `--vibration`, `--subgroup`, `--unfold`,
`--visualize`.

| I want to ... | command |
|---|---|
| label every mode with its irrep | `crystod-phonon --irreps --dim 4 4 4 -c POSCAR` |
| draw the dispersion with the irrep of every level at the special points | `crystod-phonon --irreps --dim 4 4 4 -c POSCAR` (the figure, `phonon_irreps.pdf`, comes with every `--irreps` run) |
| know which Gamma modes are IR / Raman active or silent | `crystod-phonon --irreps --dim 4 4 4 -c POSCAR` (no forces: `--vibration -c POSCAR --qpoint GM`) |
| get the form of the Raman tensor of each Raman-active mode | `crystod-phonon --irreps --dim 4 4 4 -c POSCAR --raman-tensor` (no forces: `--vibration -c POSCAR --qpoint GM --raman-tensor`) |
| get mode effective charges and the static dielectric constant | `crystod-phonon --irreps --dim 4 4 4 -c POSCAR --nac` (with `BORN`) |
| see which Wyckoff orbit gives which Gamma modes | `crystod-phonon --vibration -c POSCAR --qpoint GM` |
| see which element carries which branch | `crystod-phonon --fatband --dim 4 4 4 -c POSCAR` |
| see which branch is longitudinal | `crystod-phonon --lt --dim 4 4 4 -c POSCAR` |
| draw one mode as arrows in VESTA | `crystod-phonon --vector --dim 4 4 4 -c POSCAR --qpoint GM --mode 4` |
| freeze a mode into a structure | `crystod-phonon --modulation -c POSCAR --qpoint 0.5 0.5 0.5 --mode 1 2 3` |
| list the symmetry-allowed modes, no forces | `crystod-phonon --vibration -c POSCAR --qpoint R` |
| know which phases an instability can reach | `crystod-phonon --subgroup --dim 4 4 4 -c POSCAR` |
| ... and get those structures | `crystod-phonon --subgroup --dim 4 4 4 -c POSCAR --modulate` |
| see the phonons of a distorted phase in the zone of its parent, with the parent irreps | `crystod-phonon --unfold --child-poscar POSCAR_child --child-dim 2 2 2 --parent-poscar POSCAR_parent --parent-dim 2 2 2` |
| see every mode as arrows and animation in the browser, by irrep, with the dispersion | `crystod-phonon --visualize --dim 4 4 4 -c POSCAR` (`--bond Ti O 2.1` adds the bonds and polyhedra, `--all-irreps` the midpoints of the path segments) |

## 28. Phonon irreps (`--irreps`)

*Example directory: `example/28_phonon_irrep` (testsuite section 28); after pip install: `crystod-example 28`*

```{note}
The `example/...` paths on this page are those of a clone of the repository,
used from its root. After `pip install CrystOD`, run `crystod-example NN` (the
number of the example directory) in your working directory first: it copies
that directory and the shared `test_POSCARs/` into `./example/`, and the
commands then run from there unchanged (see [crystod-example](crystod-example.md)).
```

Label the phonon modes at the special q points with their irreducible
representations and write `phonon_irreps.yaml`, and next to it the phonon
dispersion with these labels, `phonon_irreps.pdf`
([below](#the-dispersion-with-the-labels-phonon_irrepspdf)):

```bash
cd example/28_phonon_irrep/SrTiO3_Pm-3m
crystod-phonon --irreps --dim 4 4 4 -c 221_PPOSCAR_SrTiO3
# -> phonon_irreps.yaml, phonon_irreps.pdf   (--readfc reads FORCE_CONSTANTS instead of FORCE_SETS)
```

The example directories carry `FORCE_SETS`. For `--readfc`, write
`FORCE_CONSTANTS` from it with phonopy 4 first, as the README of each
`28_phonon_irrep` directory does:

```bash
phonopy-init -d -c 221_PPOSCAR_SrTiO3 --dim 4 4 4
phonopy phonopy_disp.yaml --writefc --fc-calc traditional
crystod-phonon --irreps --dim 4 4 4 -c 221_PPOSCAR_SrTiO3 --readfc
```

With `--fc-calc traditional` these are the force constants `crystod-phonon`
builds from `FORCE_SETS`, and `phonon_irreps.yaml` is the same up to the
~1e-7 THz noise of the acoustic modes at Γ (phonopy's default fit, symfc,
shifts the SrTiO3 frequencies by up to 0.004 THz). Do not leave a
`FORCE_CONSTANTS` next to `FORCE_SETS` unless you mean to use it: phonopy
reads a `FORCE_CONSTANTS` it finds in the working directory even when
`crystod-phonon` hands it `FORCE_SETS` (no `--readfc`), and that file then
wins, so the "FORCE_SETS" run silently uses the force constants.

`-o FILE` writes the YAML file to another path (`-o SrTiO3_irreps.yaml`, also
with `--all-irreps` below), and the figure next to it with the same name and
`.pdf` (`SrTiO3_irreps.pdf`); the `* Output files *` block at the end of the
run names the files written.

The file lists, for every special q point, the modes grouped into degenerate
sets with their ISO-IR (ISOTROPY, Miller-Love) irrep labels and frequencies in
THz — note the antiferrodistortive soft mode at R, an imaginary
(negative-frequency) R5- triplet:

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
  ...

- q_label: R
  q_position: [0.5, 0.5, 0.5]
  band_sets:
  - band_indices: [1, 2, 3]
    irrep_label: ['R5-(3)']
    frequency:  -1.0867090338
  ...
```

`band_indices` are 1-based, so a level can be fed straight back into
`--vector --mode` or `--modulation --mode`. At Γ each set also carries the
Mulliken symbol of its irrep in a `mulliken:` line after `irrep_label:`, and
its spectroscopic activity in an `activity:` line (section 29). The terminal
prints the Γ activity summary, the k path of the figure and the names of the
two files.

The file is YAML, and any YAML library reads it (`yaml.safe_load`);
`crystod.phonon.read_phonon_irreps` returns it as a dictionary and checks it:

| key | content |
|---|---|
| `space_group` | Hermann-Mauguin symbol |
| `nac` | `true` with `--nac` |
| `dielectric_electronic`, `dielectric_static` | eps_inf and eps_0, 3x3 (only with `--nac`) |
| `special_points` | list of `q_label`, `q_position` (fractions of the primitive reciprocal basis) |
| `k_path`, `path_midpoints` | the seekpath path and the midpoints of its segments, each with `q_label`, `segment`, `q_position` (only in `phonon_irreps_all.yaml`) |
| `irreps` | one entry per q point: `q_label`, `q_position`, `segment` (at a path midpoint), `activity_summary` (at Γ), `band_sets` |
| `band_sets` | one entry per degenerate set: `band_indices` (1-based), `irrep_label` (list; a set holding several irreps lists them all), `frequency` (THz, negative for an imaginary mode), and at Γ `mulliken`, `activity` and, with `--nac`, `mode_effective_charge` and `dielectric_contribution` |

```python
from crystod import phonon

data = phonon.read_phonon_irreps("phonon_irreps.yaml")
for point in data["irreps"]:
    for band_set in point["band_sets"]:
        print(point["q_label"], band_set["band_indices"],
              band_set["irrep_label"], band_set["frequency"])
```
The name `R5-` refers to the setting of the input file, which has Sr at the
origin; with Ti at the origin, or with an origin that is not a standard one,
the same mode is `R4+` — see
[Irrep labels and the frame of a structure](theory-representations.md#irrep-labels-and-the-frame-of-a-structure)
for the rules every command follows (origin, axes, supercells, the names of
-k stars).

### NAC and the BORN file (`--nac`)

The non-analytical term correction is applied only with `--nac`, in every mode
that reads force data (`--irreps`, `--fatband`, `--lt`, `--vector`,
`--modulation`, `--subgroup`; not `--unfold`, which refuses `--nac`): it
then reads `BORN` from the working
directory (`--modulation` also looks next to the cell file; with `--yaml`, the
NAC parameters stored in the yaml come first, as in phonopy), and stops with
an error when there is none. Without `--nac` a `BORN` file lying in the
directory is not read. The header of `phonon_irreps.yaml` records the choice
(`nac: true` / `nac: false`), and the terminal says where the charges came
from:

```
NAC: Born effective charges and dielectric tensor read from BORN.
```

At Γ itself phonopy applies no q direction, so the `--irreps` labels and
frequencies at GM are those of the TO modes either way; NAC changes the
frequencies near Γ, e.g. the LO branch of a `--vector` or `--modulation` run
at a small q.

### Symmetry lines as well (`--all-irreps`)

The default survey covers the special *points*. `--all-irreps` additionally
labels the midpoint of every seekpath path segment — the symmetry **lines**
(DT, Z, SM, LD, S, T, ...) — and writes `phonon_irreps_all.yaml` instead, so
both surveys can coexist in one directory:

```bash
crystod-phonon --irreps --dim 4 4 4 -c 221_PPOSCAR_SrTiO3 --all-irreps
# -> phonon_irreps_all.yaml, phonon_irreps_all.pdf
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
    frequency:   2.1278014037
  - band_indices: [3]
    irrep_label: ['DT1(1)']
    frequency:   3.7542651577
  ...
```

The entries of the lines follow those of the special points in the same
`irreps` list and carry the `segment` key; a midpoint whose labeling fails
gets `band_sets: []`.

This is the labeling that tells a branch apart along a line — which of the two
`DT5` branches is which — and it is slower than the special-points-only
default. On a line the ISO-IR label is evaluated at the canonical value of the
line parameter (the smallest one, positive when the two signs are equally
small), so the same physical k point gets the same label whichever arm of the
star or copy k + G is given (DT of F-43m at (0,0,-1/2) is evaluated on the arm
(0,0,-a) at a = 1/2). The `k_path` and `segment` names are seekpath's; for an
input outside the ISO-IR setting, an endpoint to which the frame of the labels
gives another ISO-IR type than spglib's frame, in which seekpath names its
points, is printed with its ISO-IR name (the seekpath `P` of a shifted I-4
cell, which carries `PA` labels, as `GM-X-PA-N-...`).

`crystod-phonon --visualize --all-irreps` shows the modes at these midpoints
in the browser, with the same labels ([section 52](#the-segment-midpoints---all-irreps)).

### The dispersion with the labels (`phonon_irreps.pdf`)

`phonon_irreps.yaml` lists the irreps point by point; every `--irreps` run
also puts them on the phonon dispersion. It computes the bands along the k
path with phonopy, labels every level at the special points of the path, and
draws both into one figure, `phonon_irreps.pdf`, next to the YAML file. No
`band.yaml` and no plotting script of your own are needed; the figure at the
top of the README and of the [documentation start page](index.md) is the one
this command writes:

```bash
cd example/28_phonon_irrep/SrTiO3_Pm-3m
crystod-phonon --irreps -c 221_PPOSCAR_SrTiO3 --dim "4 4 4"
# -> phonon_irreps.yaml, phonon_irreps.pdf
```

It is the `--irreps` run of the start of this section (same `FORCE_SETS`,
same `phonon_irreps.yaml`); the `* Band path *` block of the terminal names
the path drawn and the points that carry labels:

```
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

The run takes about 4.5 s for this example, one second of it for the bands
and the figure. The YAML file is the result of the run: when the figure
cannot be drawn (seekpath or the labeling failing on an unusual cell), the
`* Band path *` block gives the reason in one
`NOTE: the dispersion figure was not drawn (...)` line, `* Output files *`
lists the YAML file alone, and the run still ends normally.

```{figure} images/hero_phonon_irrep.png
:name: fig-phonon-irreps-plot
:width: 95%

The `phonon_irreps.pdf` of `crystod-phonon --irreps -c 221_PPOSCAR_SrTiO3 --dim "4 4 4"`,
converted to PNG for this page: cubic SrTiO3 (Pm-3m, 4x4x4 FORCE_SETS) along the seekpath path
Γ-X-M-Γ-R-X|R-M. Every dot is a degenerate level at a special point, with its
ISO-IR label; the imaginary R5- triplet (drawn at negative frequency) is red.
```

#### Reading the figure

- **One dot, one degenerate level.** The label is the ISO-IR small irrep of
  the level with its dimension in parentheses, exactly the `irrep_label` of
  `phonon_irreps.yaml` (`GM4-(3)`, `R3-(2)`); a one-dimensional irrep is
  written without its `(1)` (`X3-`, `R2-`). The dimensions at one point add up
  to the number of bands: 3 + 3 + 3 + 3 + 2 + 1 = 15 at R for the five atoms
  of SrTiO3.
- **Γ.** The `GM4-(3)` at 0 THz is the acoustic triplet; the three other
  `GM4-(3)` (2.66, 4.71 and 15.31 THz) are the IR-active T1u modes and
  `GM5-(3)` at 7.35 THz is the silent T2u mode, as the
  `* Gamma-point activity *` block says (see
  [29. Phonon activity](#29-phonon-activity-ir-raman-mode-charges) for the
  Raman/IR selection rules and the Mulliken names).
- **Red: the instability.** A level below `--threshold` (default -0.1 THz)
  is imaginary; phonopy gives it as a negative frequency and it is drawn in
  red. Here it is `R5-(3)` at -1.09 THz, the rotation of the TiO6 octahedra
  that drives the cubic-to-tetragonal transition of SrTiO3 (`R5-` in the
  setting of this file, with Sr at the origin; `R4+` with Ti at the origin,
  see the note on frames above). Which structures it can condense into is
  the subject of [35. Subgroups from imaginary modes](#35-subgroups-from-imaginary-modes---subgroup):
  `crystod-phonon --subgroup -c 221_PPOSCAR_SrTiO3 --dim 4 4 4 --qpoint R`
  gives I4/mcm for the direction (0,0,a).
- **Close levels.** Labels that would overlap are moved apart, each kept as
  close to its level as the room allows, and joined to its dot by a leader
  line: at the right-hand M, `M2-` (12.74 THz), `M5-(2)` (12.86 THz) and
  `M1+` (13.13 THz). Two labels at one height are two levels: `X2+` at
  7.842 THz and `X1+` at 7.850 THz are separate one-dimensional levels (two
  dots); one set holding two irreps would read `X1+ + X2+` (the acoustic set
  `GM1 + GM5(2)` of P4mm BaTiO3 at 0 THz is such a set).
- **Every point once.** A point is labeled at its last place on the path that
  is not a break: Γ at the middle vertex rather than at the left edge, X at
  its first vertex rather than at `X|R`, and M at the right edge, with its
  labels to the left. Where the path jumps (`X|R`), the two points get their
  labels on either side; two equivalent points with the same levels share
  one set (`U|K` of fcc Si).
- **The lines between the points are not labeled.** The branches along the
  symmetry lines carry irreps of their own (`DT1`, `DT5(2)`, ... on Γ-X);
  `--all-irreps` lists them in `phonon_irreps_all.yaml` (the figure is the
  same with or without it), and a `--band` path with a vertex on the line
  labels them in the figure (below).

The labels are read at the vertex itself (through the tabulated arm of its
star, whose spectrum is the same band by band), so every dot lies on the
bands it labels: within 1e-7 THz in the four examples of this section
(SrTiO3, ScF3, BaTiO3, Si).

#### Options

The figure is a PDF named after the YAML file: `phonon_irreps.pdf`
(`SrTiO3_irreps.pdf` with `-o SrTiO3_irreps.yaml`, `phonon_irreps_all.pdf`
with `--all-irreps`). These options of `--irreps` shape it:

| option | effect |
|---|---|
| `--band "..."` | a path of your own, in fractions of the primitive reciprocal basis; commas separate continuous pieces (default: the seekpath path of the primitive cell, as in `--fatband` and `--lt`) |
| `--band-labels "..."` | names of the `--band` vertices (default: their ISO-IR k-vector type, so a vertex on a line is named and labeled too) |
| `--npoints N` | q points per leg of the path (default 51) |
| `--threshold F` | frequency in THz below which a level is red (default -0.1, the threshold of `--subgroup`) |
| `--nac` | bands with the LO-TO splitting (needs `BORN`); see the note below |
| `--readfc`, `--tolerance` | as for the YAML file: `FORCE_CONSTANTS` in place of `FORCE_SETS`; the degeneracy tolerance of the labels |

A malformed `--band`, a `--band-labels` count that does not fit it, or
`--npoints` below 2 stops the run with one line before the survey.

A path of your own, here with a vertex in the middle of Γ-X, so that the
branches of the line DT are labeled as well:

```bash
crystod-phonon --irreps -c 221_PPOSCAR_SrTiO3 --dim "4 4 4" \
    --band "0 0 0  1/4 0 0  1/2 0 0, 1/2 1/2 0  1/2 1/2 1/2"
# -> phonon_irreps.yaml, phonon_irreps.pdf   (add -o band.yaml to write band.yaml, band.pdf and keep the seekpath figure)
```

```
* Band path *
  k path (--band): GM-DT-X | M-R  (153 q points)
  Labeled points: GM, DT, X, M, R
```

The vertex (1/4, 0, 0) is named DT by its ISO-IR k-vector type and gets the
`DT1`, `DT2`, `DT5(2)` labels; at the jump `X|M` the X labels go to the left
of the break and the M labels to the right.

```{note}
With `--nac` the bands show the LO-TO splitting near Γ, but the Γ labels are
those of the TO modes (phonopy applies no q direction at Γ itself; see
[NAC and the BORN file](#nac-and-the-born-file---nac)): the LO branches end at
Γ without labels, and the `* Band path *` block says so in a `NOTE:` line.
```

Text is set in Arial, or in DejaVu Sans where Arial is not installed; the
figure is 16 cm x 12 cm.

## 29. Phonon activity (IR, Raman, mode charges)

*Example directory: `example/29_phonon_activity` (testsuite section 29); after pip install: `crystod-example 29`*

The Γ phonons are the ones first-order infrared absorption and Raman
scattering see. This section reads the spectroscopic side of a phonon
calculation off them, for assigning measured spectra or checking a
calculation against them: which modes are IR active, Raman active or silent,
the Raman tensor of each line and, with Born effective charges, what each IR
mode adds to the static dielectric constant. The group theory:
a mode is IR active when its irrep occurs in the vector representation, and
Raman active when it occurs in the symmetric square of the vector
representation, the representation of the polarizability tensor. There is no
mode flag of its own; the results are printed by these commands:

| output | command |
|---|---|
| IR / Raman / silent / acoustic of every Γ set, with Mulliken symbols | `crystod-phonon --irreps --dim 4 4 4 -c POSCAR` |
| the same from the structure alone, no forces | `crystod-phonon --vibration -c POSCAR --qpoint GM` ([section 34](#spectroscopic-activity-at-γ)) |
| mode effective charges, Δε of every Γ set, ε<sub>0</sub>, LST check | `crystod-phonon --irreps --dim 4 4 4 -c POSCAR --nac` (with `BORN`) |
| Raman tensor of every Raman-active irrep | add `--raman-tensor` to either of the first two |

### Spectroscopic activity

At Γ every degenerate set gets its spectroscopic activity, `IR`, `Raman`,
both, `silent` or `acoustic`, and the terminal prints the summary in a
`* Gamma-point activity *` block, one irrep per line:

```bash
crystod-phonon --irreps --dim 4 4 4 -c 221_PPOSCAR_SrTiO3
```

```

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

How to read it:

- `GM4- [T1u]` is the ISO-IR label with its Mulliken symbol in brackets;
  `x4` is the number of degenerate sets that carry it, followed by what they
  are: here one acoustic set and three IR-active optical sets. The
  `activity_summary:` line of `phonon_irreps.yaml` is the same summary
  without the brackets, and every Γ set of the file has its `mulliken:` and
  `activity:` lines (section 28).
- **IR active**: the irrep occurs in the vector representation (character
  `tr R`), so the mode carries an electric dipole. **Raman active**: it
  occurs in the symmetric square of the vector representation (character
  `(tr(R)^2 + tr(R^2)) / 2`). **silent**: neither.
- **acoustic**: the three rigid translations, reported as `acoustic` rather
  than IR although their irrep is that of the vector representation; a set
  holds as many of them as the summed rigid-translation weight of its
  mass-weighted eigenvectors.
- "Active" means allowed by symmetry: the intensity of an allowed line can
  still be too weak to observe.
- `unknown` marks a set whose characters are not those of a representation
  (too small a `--tolerance` split a degenerate set; raise it), `?` a set
  that could not be labelled.

For cubic SrTiO3 this is the textbook result: GM4- (T1u) is IR active, GM5-
(T2u) silent, and nothing is Raman active; the optical GM5+ (T2g) triplet of
Si is Raman active only; in tetragonal BaTiO3 (P4mm) GM1 (A1) and GM5 (E) are
IR and Raman active and GM2 (B1) is Raman only. The multiplicities come from
the characters alone and agree with the selection-rule table of the Bilbao
SAM program.

A set can hold more than one irrep: the acoustic set of a polar crystal
(GM1 + GM5 of BaTiO3, all at 0 THz), or optical irreps within `--tolerance`
of each other. Such a set is split per irrep: its acoustic bands go to the
irreps of the vector representation, every other copy keeps its own
activity, and the `activity:` line lists every kind that occurs
(`[IR, silent, acoustic]`). The summary therefore does not depend on the
tolerance: with `--tolerance 2.7` the first twelve bands of SrTiO3 form one
set GM4- + GM5-, and the summary is still
`GM4- x4 (3 IR, 1 acoustic), GM5- x1 (silent)`.

**Mulliken symbols.** The symbols in brackets (and the `mulliken:` lines,
`A1 + E` for a set holding two irreps) come from phonopy's point-group
character tables, matched to the characters of the ISO-IR irreps after the
rotations are carried to the conventional standard setting as phonopy does
it, so they are the symbols phonopy itself prints. B1/B2 (and B1/B2/B3)
therefore follow phonopy's axis convention, which can differ from other
tables: the Bilbao POINT/SAM and ISOTROPY tables swap B1 and B2 of 622 and
6/mmm (the silent mode of MgB2 is B1g here and B2g there), and in the P-4m2
settings (space groups 115-120) B1 is the irrep that transforms as xy. The
symbols refer to the axes of the standard setting even for a cell given in
another setting (Pbnm instead of Pnma, for instance). A complex irrep and its
conjugate both get the symbol of the physically irreducible pair (`E` of the
point group 3), so a one-dimensional member of a pair is shown as `GM2 [E]`;
other tables name the members 1E and 2E.

### Mode effective charges and the dielectric tensor (`--irreps --nac`)

With `--nac`, the Born effective charges and the electronic dielectric tensor
of `BORN` also give the polar response of the Γ phonons (the input is that of
`example/30_phonon_fatband/ScF3_Pm-3m`):

```bash
crystod-phonon --irreps --dim 4 4 4 -c 221_PPOSCAR_ScF3 --nac
```

```
NAC: Born effective charges and dielectric tensor read from BORN.

* Gamma-point activity *
GM4- [T1u] x3 (2 IR, 1 acoustic)
GM5- [T2u] x1 (silent)

* Gamma-point dielectric response (Born effective charges from BORN) *
  set  irrep          freq (THz)  activity             |Z~| (e/sqrt(amu))  Delta eps (xx, yy, zz)
    1  GM4-               0.0000  acoustic                         0.0000  -
    2  GM5-               4.2122  silent                           0.0000    0.0000    0.0000    0.0000
    3  GM4-               6.2446  IR                               0.7313    2.9993    2.9993    2.9993
    4  GM4-              13.9231  IR                               1.4304    2.3086    2.3086    2.3086
  eps_inf (diag):          2.2839    2.2839    2.2839
  sum Delta eps (diag):    5.3079    5.3079    5.3079
  eps_0 (diag):            7.5919    7.5919    7.5919
  sum_kappa Z*_kappa: max |deviation| = 1.00e-08 (acoustic sum rule)
  LST check: prod (nu_LO/nu_TO)^2 = 3.3240, eps_0/eps_inf = 3.3240 (q -> 0 along [1 0 0])

* Band path *
  k path (seekpath): GM-X-M-GM-R-X | R-M  (306 q points)
  Labeled points: X, GM, R, M
  NOTE: at GM the labels are those of the TO modes (phonopy applies no q direction at GM itself); the LO branches end there without labels

* Output files *
  Phonon irreps written to: phonon_irreps.yaml
  Phonon dispersion with irrep labels written to: phonon_irreps.pdf
```

How to read it (one row per degenerate set, then the totals):

- the **mode effective charge** of a band is the dipole its displacement
  carries: the Born charge tensor of every atom applied to that atom's
  displacement in the mode (the eigenvector divided by the square root of the
  mass, in amu), summed over the atoms; unit e/sqrt(amu). `|Z~|` is the root
  of the sum of its squares over the bands of the set, which does not depend
  on how the degenerate eigenvectors were chosen;
- the **oscillator strength** of a set is the sum over its bands of the outer
  product of the mode effective charge with itself;
- its **contribution to the static dielectric tensor** is
  `Delta eps = C S / (Omega nu^2)`, with `S` the oscillator strength,
  `Omega` the primitive-cell volume in Å^3, `nu` the frequency in THz and
  `C = e^2 / (eps_vac amu)` in these units, 4.4225e4; the acoustic bands are
  left out;
- `eps_0 = eps_inf + sum Delta eps`, with `eps_inf` the electronic tensor of
  `BORN`;
- the **acoustic sum rule** says that the Born charges of all atoms of the cell
  add up to zero; the line gives the largest element of that sum, a check of
  the `BORN` file;
- the **Lyddane-Sachs-Teller check** compares the product over the optical
  bands of `(nu_LO / nu_TO)^2` with `eps_0 / eps_inf` along the same direction:
  `nu_LO` are phonopy's frequencies with the non-analytical term for q -> 0
  along [1 0 0] (reduced reciprocal coordinates), `nu_TO` those without it.
  Only the bands that carry a dipole along q change, so the product is that of
  the LO/TO pairs. Both numbers come out of independent calculations, and
  their agreement checks the charges, the frequencies and the units together.

An imaginary optical mode with a dipole (a polar soft mode) enters with
`nu^2 < 0`, so its `Delta eps` is negative; the table then adds the line
`note: N imaginary optical band(s) (bands ...) contribute with negative sign;
eps_0 is not a static dielectric constant of an unstable structure`. The
eigenvectors of each degenerate subspace are made real before the charges are
formed (the dynamical matrix at Γ is real), so `|Z~|` does not depend on the
eigenvectors phonopy returns inside it.

`phonon_irreps.yaml` gets the two tensors in its header and two lines per Γ
set:

```yaml
nac: true
dielectric_electronic: [[2.283939, 0.000000, 0.000000], [0.000000, 2.283939, 0.000000], [0.000000, 0.000000, 2.283939]]
dielectric_static: [[7.591860, 0.000000, 0.000000], [0.000000, 7.591860, 0.000000], [0.000000, 0.000000, 7.591860]]
...
  - band_indices: [10, 11, 12]
    irrep_label: ['GM4-(3)']
    mulliken: T1u
    frequency:  13.9231282991
    activity: [IR]
    mode_effective_charge: [[0.000000, -0.011905, -0.825778], [0.000000, -0.825778, 0.011905], [-0.825864, 0.000000, 0.000000]]
    dielectric_contribution: [2.308629, 2.308629, 2.308629]
```

`mode_effective_charge` holds one Cartesian vector per band (its direction
inside a degenerate set follows phonopy's choice of eigenvectors);
`dielectric_contribution` is the diagonal of `Delta eps`.

### Raman tensors (`--raman-tensor`)

`--raman-tensor` adds the symmetry-allowed form of the Raman tensor of every
Raman-active irrep among the Γ phonons. It works with `--irreps` and, from the
structure alone, with `--vibration --qpoint GM`:

```bash
crystod-phonon --irreps --dim 4 4 4 -c 227_PPOSCAR_Si --raman-tensor
```

```

* Gamma-point activity *
GM4- [T1u] x1 (acoustic)
GM5+ [T2g] x1 (Raman)

* Raman tensors (Cartesian axes of the input cell) *
  GM5+ [T2g] (3 tensors):  [[0, 0, 0], [0, 0, a], [0, a, 0]]   [[0, 0, a], [0, 0, 0], [a, 0, 0]]   [[0, a, 0], [a, 0, 0], [0, 0, 0]]

* Band path *
  k path (seekpath): GM-X-U | K-GM-L-W-X  (306 q points)
  Labeled points: U|K, GM, L, W, X

* Output files *
  Phonon irreps written to: phonon_irreps.yaml
  Phonon dispersion with irrep labels written to: phonon_irreps.pdf
```

How to read it:

- The tensors are written in the Cartesian axes of the input cell (the axes
  of the POSCAR lattice vectors), so that they can be compared with a
  polarization geometry in the laboratory frame of that cell. The Mulliken
  symbol in brackets refers to the axes of the standard setting, so for a
  cell in a non-standard setting (Pbnm instead of Pnma) the tensor of B1g
  need not be xy.
- The tensors of one irrep are a basis of the symmetric second-rank tensors
  that transform as it (the character projector of the irrep on the six
  components xx, yy, zz, yz, xz, xy), chosen sparsest first, each with its
  own free constant `a` and exact entries where possible (`-2*a`,
  `sqrt(3)/2*a`). A complex irrep is listed together with its conjugate
  (`GM2GM3`).
- For an irrep that occurs once in the symmetric square the tensors are its
  partners (the cubic Eg pair `diag(a, a, -2a)` and `diag(a, -a, 0)`). A
  one-dimensional irrep that occurs twice (A1 of 4mm and 6mm) prints its two
  basis tensors, and a mode's tensor is a combination of them
  (`diag(a, a, b)`).

An irrep of dimension two that occurs twice (E of 3m, -3m and 32, and the
complex E pair of 3 and -3) is printed per partner instead, with constants
shared across the partners, as in the tables:

```bash
crystod-phonon --vibration -c 166_PPOSCAR_Bi --qpoint GM --raman-tensor
```

```
...
* Raman tensors (Cartesian axes of the input cell) *
  GM1+ [A1g] (2 tensors):  [[a, 0, 0], [0, a, 0], [0, 0, 0]]   [[0, 0, 0], [0, 0, 0], [0, 0, a]]
  GM3+ [Eg] (2 partners, constants a, b):  [[a, 0, 0], [0, -a, b], [0, b, 0]]   [[0, a, b], [a, 0, 0], [b, 0, 0]]
```

The two partners of a mode of the irrep have these forms with the same `a`
and `b`; partner 2 follows from partner 1 by symmetry, and its overall sign
is a convention (Loudon's second Eg tensor of -3m is the negative of the one
printed here). For the complex E pair of 3 and -3 the partners carry four
constants, `[[a, d, c], [d, -a, b], [c, b, 0]]` and
`[[d, -a, -b], [-a, -d, c], [-b, c, 0]]`. In a rotated setting the entries
become combinations such as `a - sqrt(3)*b`. The forms agree with the tables
of R. Loudon, Adv. Phys. 13, 423 (1964):

| crystal | irrep | tensors |
|---|---|---|
| Si (Fd-3m) | GM5+ (T2g) | yz, xz, xy pairs |
| SrTiO3 (Pm-3m) | none | `no Raman-active irrep at Gamma` |
| BaTiO3 (P4mm) | GM1 (A1) | diag(a, a, 0), diag(0, 0, a) |
| | GM2 (B1) | diag(a, -a, 0) |
| | GM5 (E) | yz, xz pairs |
| ZnO (P6_3mc) | GM1 (A1) | diag(a, a, 0), diag(0, 0, a) |
| | GM5 (E2) | diag(a, -a, 0), xy pair |
| | GM6 (E1) | yz, xz pairs |
| Bi (R-3m) | GM1+ (A1g) | diag(a, a, 0), diag(0, 0, a) |
| | GM3+ (Eg) | [[a, 0, 0], [0, -a, b], [0, b, 0]], [[0, a, b], [a, 0, 0], [b, 0, 0]] |
| ZnRe2O8 (R-3) | GM2+GM3+ (Eg) | 2 partners, 4 constants (Loudon's C3i form) |

A cell rotated in space gives the rotated tensors (diamond Si turned by 45°
about z has `diag(a, -a, 0)` in place of the xy pair).

**Limits.** The activities and tensors are symmetry selection rules for
first-order (one-phonon) IR absorption and Raman scattering at Γ: they say
which lines can appear and in which polarizations, not their intensities
(the dielectric table gives the IR oscillator strengths from the Born
charges; Raman intensities need the polarizability derivatives). The point
group is that of the atomic structure; magnetic order is not taken into
account. `--raman-tensor` takes Γ only (`--vibration` at another q point
stops with an error).

**Python API.** The MCP tool is `crystod_phonon_activity` (the Γ activity
table, the Wyckoff-orbit breakdown and the Raman tensors of a POSCAR, without
force constants). `gamma_mode_activities` prints nothing on stdout: the
primitive-cell note of the commands is left out, so it cannot reach the
protocol stream of the MCP server (see `quiet` in
[What is written](#what-is-written)):

```python
from crystod import phonon
from crystod.vasp_io import read_poscar_cell

cell = read_poscar_cell("221_PPOSCAR_SrTiO3")    # or a phonopy object with forces
phonon.gamma_mode_activities(cell)               # list of Activity records, one per mode space or set
phonon.gamma_raman_tensors(read_poscar_cell("227_PPOSCAR_Si"))   # list of RamanTensors
phonon.mulliken_symbols(cell)                    # {'GM1+': 'A1g', ..., 'GM4-': 'T1u', ...}

import phonopy                                   # example/30_phonon_fatband/ScF3_Pm-3m
ph = phonopy.load(unitcell_filename="221_PPOSCAR_ScF3", supercell_matrix=[4, 4, 4],
                  force_sets_filename="FORCE_SETS", born_filename="BORN", is_nac=True)
phonon.dielectric_response(ph).eps_static        # DielectricResponse; diagonal 7.5919
phonon.mode_effective_charges(ph)                # (n_bands, 3) array at Gamma, e/sqrt(amu); here (12, 3)
```

## 30. Phonon fatbands (`--fatband`)

*Example directory: `example/30_phonon_fatband` (testsuite section 30); after pip install: `crystod-example 30`*

Plot phonon fatbands colored by the element-projected phonon density (sum of
squared eigenvector components over each element's atoms), directly from
POSCAR + `FORCE_SETS` (or `FORCE_CONSTANTS` with `--readfc`):

```bash
cd example/30_phonon_fatband/ScF3_Pm-3m
crystod-phonon --fatband -c 221_PPOSCAR_ScF3 --dim 4 4 4
# -> fatband_Sc.pdf, fatband_F.pdf
```

```{figure} images/fatband_F.png
:name: fig-fatband-f
:width: 75%

`fatband_F.pdf` for ScF3 (4x4x4 FORCE_SETS): the F-projected weight (shading)
concentrates on the soft rotational branches around R and M and on the
high-frequency stretching bands, while the Sc-dominated mid-frequency bands
stay unshaded (they appear in `fatband_Sc.pdf` instead).
```

The space group is detected, the high-symmetry k-path is generated
automatically with seekpath, and the band structure is computed with
eigenvectors and band connection through the phonopy API — no `band.yaml`
needed. One PDF per element is written, in the VESTA default element colors,
with dot sizes proportional to the projected weight.

Options: `--element F` restricts the output to one element; `--band`/`--band-labels`
supply a manual k-path instead of seekpath; `--npoints` sets the q-point density
per path leg (default 51); `--projection-direction "0 0 1"` projects the
displacements onto a direction in reduced coordinates before squaring.
Plotting style based on `script/phonon_fatband.py` by Hiroki Koiso.

### LO/TO splitting (`--nac`)

```bash
crystod-phonon --fatband -c 221_PPOSCAR_ScF3 --dim 4 4 4 --nac
# -> fatband_nac_Sc.pdf, fatband_nac_F.pdf
```

```{figure} images/fatband_nac_F.png
:name: fig-fatband-nac-f
:width: 75%

`fatband_nac_F.pdf` — the same F-projected fatband with the non-analytical
term correction (a `BORN` file in the current directory). The LO/TO splitting
lifts the highest F-dominated branch at Gamma from 13.9 to 20.5 THz, while the
zone-boundary soft modes at R and M are untouched: the correction acts only in
the long-wavelength limit. The plot title carries an "(NAC)" tag, and the file
names get a `nac_` prefix so corrected and uncorrected fatbands coexist.
```

## 31. Longitudinal/transverse bands (`--lt`)

*Example directory: `example/31_phonon_lt` (testsuite section 31); after pip install: `crystod-example 31`*

Plot the phonon band structure colored by the longitudinal/transverse character
of each mode (red = longitudinal, blue = transverse, white = mixed or Gamma):

```bash
cd example/31_phonon_lt/ScF3_Pm-3m
crystod-phonon --lt -c 221_PPOSCAR_ScF3 --dim 4 4 4
crystod-phonon --lt -c 221_PPOSCAR_ScF3 --dim 4 4 4 --nac
# -> phonon_band_LT.pdf   /   phonon_band_LT_nac.pdf
```

```{figure} images/phonon_band_LT.png
:name: fig-phonon-lt
:width: 75%

`phonon_band_LT.pdf` for ScF3: red = longitudinal, blue = transverse,
white = mixed. Along GM-X the acoustic set splits visibly into one red L
branch and two blue T branches; the flat band near 14 THz is purely
transverse throughout the zone.
```

The longitudinal character of a mode is `sqrt(sum_atoms |q_hat . e_atom|^2)`,
the norm of the eigenvector projection onto the propagation direction, which is
valid along any path direction including diagonal segments such as GM-R. With
`--nac` the split-off LO branches show up as purely red. Based on
`script/LT_phonon_band.py` maintained by Hiroki Koiso, after Qijing Zheng.

## 32. Phonon eigenvectors (`--vector`)

*Example directory: `example/32_phonon_vector` (testsuite section 32); after pip install: `crystod-example 32`*

Diagonalize the dynamical matrix directly at the selected q point via the
phonopy API, list the modes with their frequencies and irrep labels, and export
the selected eigenvectors as `.vesta` files with per-atom displacement arrows:

```bash
cd example/32_phonon_vector/Si_Fd-3m
crystod-phonon --vector --dim "4 4 4" -c 227_PPOSCAR_Si --qpoint GM
```

```

* Structure *
  Space group: Fd-3m (#227)

* Available high-symmetry q-points *
  GM       [0.0, 0.0, 0.0]
  L        [0.5, 0.5, 0.5]
  X        [0.5, 0.0, 0.5]
  W        [0.5, 0.25, 0.75]

* Selected Q point *
  Selected q-point: GM = [0.0, 0.0, 0.0]

* Phonon modes at q = GM *
 Mode    Freq (THz)  Irrep
----------------------------------------
    1        0.0000  GM4-(3)
    2        0.0000  GM4-(3)
    3        0.0000  GM4-(3)
    4       14.9571  GM5+(3)
    5       14.9571  GM5+(3)
    6       14.9571  GM5+(3)

* Displacement patterns *
  No --mode given; exporting all 6 modes as individual VESTA files.
  Commensurate supercell for visualization: 1x1x1
  + mode 1: GM4-(3), 0.0000 THz
  ...
  + mode 6: GM5+(3), 14.9571 THz
  Arrows are scaled so the largest displacement is 1.5 A; adjust arrow size in VESTA via Edit > Vectors or Properties > Vectors if needed.

* Output files *
  Mode table written to: phonon_modes_Si_GM.txt
  Mode 1 written to: POSCAR_Si_GM_mode1_GM4-.vesta
  ...
  Mode 6 written to: POSCAR_Si_GM_mode6_GM5+.vesta
```

The mode table of the `* Phonon modes at q = ... *` block is also saved as
`phonon_modes_<formula>_<qlabel>.txt`.

Select the modes with `--mode` (1-based, as in the table above); several mode
numbers are summed into one displacement pattern:

```bash
# one optical GM5+ mode
crystod-phonon --vector --dim "4 4 4" -c 227_PPOSCAR_Si --qpoint GM --mode 4

# the whole triplet, summed
crystod-phonon --vector --dim "4 4 4" -c 227_PPOSCAR_Si --qpoint GM --mode 4 5 6

# X point: the commensurate supercell (2x1x2) and Bloch phases are applied automatically
# (--readfc: the development tree keeps a FORCE_CONSTANTS here, which is
#  not in the pip package; a crystod-example copy has FORCE_SETS only, so write
#  one as in section 28 or drop --readfc)
crystod-phonon --vector --dim "4 4 4" -c 227_PPOSCAR_Si --qpoint X --mode 1 --readfc

# conventional cell (POSCAR_Si_GM_mode4+5+6_GM5+_conv.vesta)
crystod-phonon --vector --dim "4 4 4" -c 227_PPOSCAR_Si --qpoint GM --mode 4 5 6 --conventional
```

Files are auto-named `POSCAR_<formula>_<qlabel>_mode<N>_<irrep>.vesta` (mode
numbers zero-padded so `ls` lists them in order) and open directly in
[VESTA](https://jp-minerals.org/vesta/), showing the equilibrium structure with
red arrows for the real part of the mass-weighted eigenvector displacement,
rescaled so that the largest displacement equals `--amplitude` (default
1.5 Angstrom).

Degenerate modes are exported as **symmetry-adapted** eigenvectors: the
dynamical matrix is block-diagonalized in the spgrep irrep-projected basis (the
same construction as `--modulation`), so the three degenerate GM5+ optical
modes of Si point exactly along the cubic axes instead of the arbitrary tilted
combinations a plain eigensolver returns. The exported vectors are exact
eigenvectors of the phonopy dynamical matrix, verified internally against the
phonopy spectrum.

`--qpoint` accepts a high-symmetry label (`GM`, `X`, `L`, ...) or three
primitive reciprocal coordinates (fractions allowed). For a q point that is not
special, the file name carries its ISO-IR k-vector type (`SM`, `DT`, ...);
`--keep-q-coords` names it by coordinates instead (see section 33), so a scan
along one symmetry line does not overwrite itself.

## 33. Phonon modulation (`--modulation`)

*Example directory: `example/33_modulation` (testsuite section 33); after pip install: `crystod-example 33`*

Generate modulated (displaced) structures from symmetry-adapted phonon modes.
The inputs are the ones you already have — the unit cell and `FORCE_SETS`
(or `FORCE_CONSTANTS` with `--readfc`); a `phonopy_params.yaml` works too, but
none has to be produced first:

```bash
cd example/33_modulation/ScF3_Pm-3m
crystod-phonon --modulation -c 221_PPOSCAR_ScF3 --qpoint 0.5 0.5 0.5 --mode 1 2 3 --amplitude 0.3
crystod-phonon --modulation --yaml phonopy_params.yaml --qpoint 0.5 0.5 0.5 --mode 1 2 3 --amplitude 0.3
```

`--qpoint` takes three primitive reciprocal coordinates or a special-point
label, resolved as in `--vector` and `--subgroup` (`--qpoint R` is
`--qpoint 0.5 0.5 0.5` here; the numbered `--qpointN` take the same forms).
The two write the same structure, byte for byte. `FORCE_SETS` is looked for in
the current directory first and then next to the `-c` file, so
`-c ../221_PPOSCAR_ScF3` works from a scratch directory. `--yaml` and `-c` are
alternatives, not a pair:

```
crystod-phonon: error: --modulation takes either --yaml or -c (with FORCE_SETS/FORCE_CONSTANTS), not both.
```

### Where the supercell comes from

`--modulation` is the one mode that does not need `--dim` —
`--irreps`/`--fatband`/`--lt`/`--vector`/`--subgroup` still do. It resolves the
supercell of the force calculation in this order, and always prints what it
used, so an inferred supercell is never silent:

| | source | when |
|---|---|---|
| 1 | `--dim "4 4 4"` | given explicitly; wins over the rest |
| 2 | the `supercell_matrix` of `phonopy_disp.yaml` (or `phonopy_params.yaml`) | that file sits in the current directory |
| 3 | the atom count of `FORCE_SETS`, or of `FORCE_CONSTANTS` with `--readfc` | nothing else says |

```bash
crystod-phonon --modulation -c 221_PPOSCAR_ScF3 --dim "4 4 4" --qpoint 0.5 0.5 0.5 --mode 1 2 3 --amplitude 0.3
crystod-phonon --modulation -c 221_PPOSCAR_ScF3 --readfc --qpoint 0.5 0.5 0.5 --mode 1 2 3 --amplitude 0.3
# --readfc reads FORCE_CONSTANTS (the shipped example directory carries FORCE_SETS)
```

```

* Phonon source *
  Supercell 4x4x4 read from phonopy_disp.yaml. Primitive cell: 4 atoms of the 4-atom input cell (primitive_matrix auto).

* Selected Q point *
  Loading '221_PPOSCAR_ScF3 + FORCE_SETS' at q = [0.5, 0.5, 0.5]...
```

```

* Phonon source *
  Supercell 4x4x4 inferred from the 256 atoms of FORCE_SETS; pass --dim if that is not the supercell of your force calculation. Primitive cell: 4 atoms of the 4-atom input cell (primitive_matrix auto).
```

Case 3 counts atoms, so it fixes only the product `n1*n2*n3`; the shape is
picked to respect the metric of the cell (`|a| = |b|` forces `n1 = n2`). It is
a guess, the message says so, and `--dim` overrides it. For the ScF3 example
every route in the table — and `--yaml` — writes the same file, byte for byte.

A supercell that is *not* diagonal is refused rather than guessed at — any
diagonal guess with the right atom count loads without complaint, so the error
would otherwise be silent:

```
ERROR: 'phonopy_disp.yaml' states a supercell_matrix that is not a positive diagonal matrix; this workflow supports diagonal supercells only. Inferring one instead would silently give the wrong answer, since any guess with the right atom count loads without complaint. Use --yaml phonopy_params.yaml, or give the diagonal supercell with --dim "n n n".
```

```{note}
The `-c` route takes the primitive cell the way every other `crystod-phonon`
mode does — `primitive_matrix="auto"` — while a `phonopy_params.yaml` carries
whatever primitive matrix it was built with. For a primitive input cell (any P
lattice) the two coincide, which is why the commands above agree byte for byte;
hand `-c` a *centred conventional* cell and the `-c` route works in the smaller
primitive cell instead. The printed `Primitive cell:` line always says which
one you got.
```

### Choosing the modes

When `--mode` is omitted, only the mode table and the star of q are printed, so
the modes at a q point can be inspected before choosing:

```bash
crystod-phonon --modulation -c 221_PPOSCAR_ScF3 --qpoint 0.5 0.5 0.5
```

```

* Phonon source *
  Supercell 4x4x4 read from phonopy_disp.yaml. Primitive cell: 4 atoms of the 4-atom input cell (primitive_matrix auto).

* Selected Q point *
  Loading '221_PPOSCAR_ScF3 + FORCE_SETS' at q = [0.5, 0.5, 0.5]...

* Phonon modes at q = [0.5 0.5 0.5] *
 Mode    Freq (THz)         Irrep   Degeneracy
--------------------------------------------------
    1        1.4169        R4+(3)            3
    2        1.4169        R4+(3)            3
    3        1.4169        R4+(3)            3
    4        7.1689        R5+(3)            3
    5        7.1689        R5+(3)            3
    6        7.1689        R5+(3)            3
    7       11.3334        R4-(3)            3
    8       11.3334        R4-(3)            3
    9       11.3334        R4-(3)            3
   10       13.6922        R3+(2)            2
   11       13.6922        R3+(2)            2
   12       18.7661        R1+(1)            1

* Star of q (arms related by the space-group rotations) *
  |G| = 48, |G_k| = 48, |star of k| = 1
  arm 1: k = [+0.5, +0.5, +0.5]

No --mode given. Choose mode number(s) from the table above and rerun with
--mode (and optionally --amplitude) to generate a modulated structure.
```

Then apply the chosen modes:

```bash
crystod-phonon --modulation -c 221_PPOSCAR_ScF3 \
  --qpoint 0.5 0.5 0.5 --mode 1 2 3 --amplitude 0.3
# -> MPOSCAR_R_mode1+2+3_R4+_R-3c
```

The soft octahedral-rotation triplet of ScF3 is the lowest R4+ set, modes 1-3;
applying all three with equal amplitudes condenses the `(a,a,a)` direction, and
the space group of the modulated structure is detected and printed as **R-3c** —
matching the `--parent Pm-3m --irrep R4+` isotropy table of `crystod-group`
(section 17). The output name follows
`MPOSCAR_{q}_{mode}_{irrep}_{subgroup}` unless `--output` is given, and a single
`--amplitude` applies to all selected modes.

```{note}
The space group is reported at a 0.1 Å tolerance by default, which is what makes
a nearly-symmetric distortion read as the symmetric one. Pass `--tolerance 1e-5`
to classify the structure at the tolerance the order-parameter direction implies:
`--mode 1 2 3 --amplitude 0.3 0.15 0.075` is the direction `(a,b,c)` and prints
`P-1` at 1e-5, but `C2/c` at the default, because the smallest of the three
displacements is below 0.1 Å. For a parent structure that is itself
symmetric at 1e-5, the modes are exact to rounding error, so the symmetry of a
direction holds at 1e-5 too; a space group that appears only at the default
tolerance belongs to a nearby, more symmetric direction, not to the one frozen
in, and the 1e-5 result is the one to report. `--tolerance` also sets the
tolerance of the mode construction itself (default 1e-5).
```

Different q points can be combined with numbered argument sets
(`--qpoint1/--mode1/--amplitude1`, `--qpoint2/--mode2/--amplitude2`, ...):

```bash
crystod-phonon --modulation --yaml phonopy_params.yaml \
  --qpoint1 0 0.5 0.5 --mode1 1 --amplitude1 0.3 \
  --qpoint2 0.5 0 0.5 --mode2 1 --amplitude2 0.3 \
  --qpoint3 0.5 0.5 0 --mode3 1 --amplitude3 0.3 \
  --output POSCAR_multi_q_arms
```

The star of q is displayed for each selected q point, which is what makes
combining arms of the same star (the three X or M arms of a perovskite)
straightforward. The X point of body-centred tetragonal Sr3Ti2O7 has two arms,
`(0,0,1/2)` and `(1/2,1/2,0)`; its lowest mode, X3-, frozen into both with
equal amplitudes is the direction `X3-(a;a)`:

```bash
cd example/33_modulation/Sr3Ti2O7_I4mmm
# a crystod-example copy has no phonopy_params.yaml (the file is
# not in the pip package): make it from the unit cell and FORCE_SETS with
# phonopy 4 first
phonopy-init -d -c 139_PPOSCAR_Sr3Ti2O7 --dim 4 4 4
phonopy phonopy_disp.yaml --sp
crystod-phonon --modulation --yaml phonopy_params.yaml \
  --qpoint1 0 0 0.5 --mode1 1 --amplitude1 0.3 \
  --qpoint2 0.5 0.5 0 --mode2 1 --amplitude2 0.3
```

```
...

* Modulation *
  Term 1: q = [0.0, 0.0, 0.5], modes = [1], amplitudes (A) = [0.3]
  Term 2: q = [0.5, 0.5, 0.0], modes = [1], amplitudes (A) = [0.3]

* Symmetry of the generated structure *
  Space group: P4_2/mnm (#136)
  Hall symbol: -P 4n 2n

* Output files *
  Modulated structure written to: MPOSCAR_X_mode1_X3-_X_mode1_X3-_P4_2mnm
```

The blocks before `* Modulation *` are those of a single q point, repeated for
each term: `* Selected Q point *`, `* Phonon modes at q = ... *` and `* Star of q ... *`.

One arm alone gives `X3-(0;a)`, Cmcm, and unequal amplitudes `X3-(a;b)`, Pnnm
(check it with `--tolerance 1e-5`: at the default tolerance, amplitudes as
different as 0.3 and 0.16 still read as P4_2/mnm) — the rows of
`crystod-group --parent I4/mmm --irrep X3-`. `--qpoint2 -0.5 0.5 0`, the same
point written with another reciprocal lattice vector, writes the same
structure.

### What is frozen in

The displacement of atom *j* in the cell at lattice translation **R** is

```
u_j(R) = A Re[ v_j exp(2 pi i q.R) ],   v_j = e_j exp(2 pi i q.x_j) / sqrt(m_j)
```

with `A` the `--amplitude`, `e` phonopy's eigenvector of the mode (the
mass-weighted eigenvector of phonopy's dynamical matrix, in its atom-position
phase convention), `x_j` and `m_j` the fractional position and the mass of atom
*j*, and `v` normalized over the primitive cell. This is the harmonic
eigen-displacement of the mode — heavy atoms move less than light ones — and for
a non-degenerate mode at a time-reversal-invariant q it is, up to its sign, the
pattern phonopy's own `MODULATION` freezes in (the test suite compares the two,
section 33).

- **`--amplitude` (Å) is the norm of the displacement of one primitive cell.**
  At a time-reversal-invariant q (`2q` a reciprocal lattice vector: Γ and the
  zone-boundary points such as X, M, R, N) `v` is real, and a mode frozen in
  with amplitude `A` displaces every primitive cell of the supercell by a
  pattern of norm exactly `A` (the default 0.3 Å). At any other q the complex
  vector `A v` has the norm `A`, and the real displacement varies from cell to
  cell along the modulation.
- **q may be written any way.** It does not have to be the arm a table lists,
  and `q + G` is the same modulation as `q`: `(0.5, 0.5, 0)` and
  `(-0.5, 0.5, 0)` of Sr3Ti2O7 differ by a reciprocal lattice vector and write
  the same structure. Two different arms of the star are two different
  modulations (domains related by a rotation of the parent); the printed star
  lists them.
- **Degenerate levels at a time-reversal-invariant q.** The mode vectors are
  real, and the partners of a degenerate level are orthonormal real patterns
  along directions that symmetry operations fix up to sign; single partners and
  equal-amplitude combinations give order-parameter directions of the irrep
  (`--mode 1 2 3` of the R4+ triplet of ScF3 is `(a,a,a)`, R-3c). Where several
  such bases exist (the `(a,0)` and `(a,a)` directions of a two-dimensional
  irrep), the one whose single partners freeze into the most symmetric
  subgroups is taken — the largest point group, then the lowest space-group
  number — and the partners are listed in that order, so which subgroup
  `--mode N` gives does not depend on the origin, the orientation or the
  lattice basis of the input cell. (This is not always ISOTROPY's `(a,0)`: for
  the M5- pair of ScF3 a single partner gives Pmma, which ISOTROPY lists as
  `(a,a)`; `--subgroup --modulate` generates every direction either way.)
  Equivalent partners (domains of one direction) are listed along the
  conventional crystal axes where those tell them apart: `--mode 1`, `2` and
  `3` of the R4+ triplet of ScF3 rotate the octahedra about a, b and c, and the
  x partner of a tetragonal pair comes before the y partner. Where they do not
  — the two Pnma partners of the X point of diamond Si, whose mirrors have the
  same orientations but lie in different places — the displacement patterns
  decide (read along the crystal axes, atom by atom, so this order follows the
  order of the atoms in the cell). Which domain comes out is a convention, but
  neither an origin shift nor a rigid rotation of the input changes it. The
  signs are conventions too: at Γ a partner has its largest component along
  the crystal axes positive; at any other such q reversing a partner only
  translates it by a lattice vector (which of the two copies is written
  depends on the origin), and the partners of one level are signed relative
  to each other, so `--mode 1 2 3` freezes into the same domain for every
  origin. The arms of a star are signed independently of each other: two
  arms differ at most by a translation, but with three or more (the M arms of
  a cubic crystal) an origin shift can turn a sum over arms into another
  domain of the same subgroup. At Γ, where no lattice translation can undo a
  sign, a partner that one of those operations reverses freezes into a
  lower-symmetry direction than its companion (a Γ E pair of a trigonal or
  hexagonal crystal, for instance). Where time reversal pairs a complex irrep
  with its conjugate, the pair forms one real level of twice the dimension
  (the Degeneracy column says so), whose partners are real but not all along
  such directions (the displacement patterns fix them instead, again
  independently of the origin).
- **At any other q the global phase is a convention.** The mode is a complex
  Bloch wave, and its overall phase — a shift of the modulation along the
  lattice — is fixed by a convention, not by the physics: the atom with the
  largest displacement is placed at the end of the major axis of the ellipse
  it moves on, in the cell at R = 0 (its largest component along the crystal
  axes positive). The rule does not depend on the orientation of the input,
  and an origin shift at most translates the result by a lattice vector. A
  structure frozen there is one member of a family related by that phase, and
  its space group is the one of that member; a single partner of a degenerate
  level is a complex Bloch partner, and its structure depends on the phase in
  the same way.
- **The cell is used as it is.** The analysis runs on the primitive cell of the
  phonopy object — its origin, orientation and lattice basis are kept in the
  output (the atoms are only grouped by species, as a POSCAR needs), nothing
  is re-standardized. That cell must be primitive
  (`primitive_matrix="auto"`, which `-c` uses); a `phonopy_params.yaml` that
  stores the conventional cell of a centred lattice as its primitive cell is
  refused with a message saying so. A structure that is symmetric only within a
  loose `--tolerance` is refused as well: symmetrize it before the force
  calculation.
- **q must be commensurate.** A q that no supercell of at most 12 cells per axis
  holds (`0.15`, for instance) is refused before any mode is computed, and the
  message names the smallest supercell that holds it (`20x1x1` for
  `0.15 = 3/20`); give q as fractions with a denominator of at most 12 (`1/7`).
  The mode table at such a q can still be previewed without `--mode`.
  `--vibration` makes the same check.

### Scanning a symmetry line (`--keep-q-coords`)

A q point that is not a special point is named after its ISO-IR k-vector type,
so a scan along one line would keep overwriting the same file.
`--keep-q-coords` names the output by coordinates instead:

```bash
crystod-phonon --modulation --qpoint 0.25 0.25 0 --mode 1 --amplitude 0.2
crystod-phonon --modulation --qpoint 0.25 0.25 0 --mode 1 --amplitude 0.2 --keep-q-coords
```

```
  Modulated structure written to: MPOSCAR_SM_mode1_SM3_Pmma               # default
  Modulated structure written to: MPOSCAR_q_0.25_0.25_0_mode1_SM3_Pmma    # --keep-q-coords
```

The flag works the same way in `--vector` mode.

## 34. Vibration bases (`--vibration`)

*Example directory: `example/34_vibration` (testsuite section 34); after pip install: `crystod-example 34`*

`--vibration` lists the symmetry-adapted displacement patterns of a structure
at one q point, grouped by irrep, from the crystal symmetry alone: no force
data are needed. It answers which irreps the vibrations at q span before any
phonon calculation (the counts of the Bilbao SAM program at Γ), and it writes
any one pattern as a displaced structure. The group theory: the
displacements of the atoms form the mechanical representation, which the
projection operators of the small irreps at q split into the mode spaces.

```bash
crystod-phonon --vibration -c 221_PPOSCAR_ScF3 --qpoint R
```

```
 ### Inputed cell was converted into primitive cell. ###

* Q points (primitive) *
  GM       (0, 0, 0)
  R        (1/2, 1/2, 1/2)
  M        (1/2, 1/2, 0)
  X        (0, 1/2, 0)

* Selected Q point *
  R        (1/2, 1/2, 1/2)

* Irrep-grouped vibration spaces *
  Mode Space  1: irrep = R1+(1), dimension = 1, component numbers = 1..1
  Mode Space  2: irrep = R3+(2), dimension = 2, component numbers = 1..2
  Mode Space  3: irrep = R4+(3), dimension = 3, component numbers = 1..3
  Mode Space  4: irrep = R4-(3), dimension = 3, component numbers = 1..3
  Mode Space  5: irrep = R5+(3), dimension = 3, component numbers = 1..3
```

How to read it:

- `* Q points (primitive) *` lists the ISO-IR special points of the space
  group, named as the irrep labels at them are named, as fractions of the
  reciprocal basis vectors of the primitive cell the mode spaces use (`1/3`,
  `-1/2`; a coordinate that is no simple fraction is written with four
  decimals). They are the points `--irreps` surveys (the `special_points:` of
  `phonon_irreps.yaml`, which may list them in another order), in the order
  of seekpath's special-point table (GM, R, M, X for Pm-3m; GM, A, K, H, M, L
  for P6_3mc). `--list-qpoints` prints only this block and stops.
- `* Selected Q point *` is the point of `--qpoint`.
- Each `Mode Space` line is one copy of an irrep: its ISO-IR label with the
  dimension in parentheses, and the numbers of its components (the values
  `--component-index` takes). The 12 displacement degrees of freedom at R
  group into 1 + 2 + 3 + 3 + 3, the same irreps as the `--modulation` mode
  table of section 33.
- At Γ the lines also carry the Mulliken symbol and the activity, and two
  more blocks follow, three with `--raman-tensor`
  ([Spectroscopic activity at Γ](#spectroscopic-activity-at-γ)).

Select one symmetry-allowed mode component, build the commensurate supercell,
and export a displaced structure:

```bash
crystod-phonon --vibration -c 221_PPOSCAR_ScF3 --qpoint R \
  --mode-index 3 --component-index 1 --output POSCAR_vibration
```

`--qpoint` accepts either one of the listed names (any case; `GAMMA`, `G` and
`Γ` also mean GM, and seekpath names such as `H_2` are accepted for their
points) or three primitive reciprocal
coordinates (fractions such as `1/3` allowed). Coordinates are named after the
listed point they are, or the arm of its star they lie on, else after their
ISO-IR k-vector type (`T` for a point on the R-M line of a cubic crystal, `PA`
for the -k partner of P). `--mode-index` adds a `* Selected basis vector *`
block with the selected space and component, the commensurate supercell and
the first displacement vectors. Add `--export-npz mode_data.npz` to save
positions, displacement vectors, symbols, and lattice for notebook-side
visualization. `--amplitude` takes one value (default 0.3 Å), since one
component is written. The supercell is the smallest one commensurate with q,
with at most 12 cells per axis as in `--modulation`: with `--mode-index`, a q
that no such supercell holds (`0.15 0 0`) is refused, and the message names the
smallest supercell that holds it (`20x1x1`); the mode spaces at such a q are
still listed without `--mode-index`. The POSCAR lists the atoms grouped by
species (`Sc F` / `8 24`), in the order the species first appear.

### What is written

The component is frozen in as

```
u_j(R) = A Re[ w_j exp(2 pi i q.R) ]
```

with `A` the `--amplitude` and `w` a unit-norm symmetry-adapted partner of the
selected mode space, which includes the Bloch factor `exp(2 pi i q.x_j)` of the
position `x_j` of atom *j* in the primitive cell that is written out. There are
no force constants and no masses here: `w` is a displacement pattern that
transforms as its irrep, not a normal mode. Its partners follow the conventions
of `--modulation` ([What is frozen in](#what-is-frozen-in)), with every mode
space treated as one level:

- **At a time-reversal-invariant q** (Γ and zone-boundary points such as X, M,
  R, N, Y) every component is real, and the displacement of each primitive cell
  has the norm `A`. A single component freezes into an isotropy subgroup of its
  irrep: for a one-dimensional irrep the single-arm subgroup that
  `crystod-group --parent SG --irrep IR` lists (all six X3- spaces of Sr3Ti2O7
  give Cmcm, the Y2- spaces of a Cmcm crystal give Pnma), and for the partners
  of a larger space the directions `--mode` takes in `--modulation` (the three
  R4+ components of ScF3 are the I4/mcm rotations about a, b and c, in that
  order). Where an irrep occurs once at q, the written pattern is the
  `--modulation` mode of that irrep. An origin shift of the input gives every
  component of an irrep the same space group. The labels refer to the input's
  own origin when the input is already in the ISO-IR setting (the empty cube
  centre of ScF3 is such an origin, and the R irreps are renamed with it:
  `R5-` for the `R4+` rotations); any other description of the same crystal
  is labelled in a canonical frame, which keeps the input's own origin when
  that is the origin of a standard description. Shifted copies of the crystal
  therefore get the same labels whatever their orientation or basis, while an
  unshifted re-based cell or a non-diagonal supercell can get names that
  differ from theirs, or from the cell's, by a normalizer element (the rules
  and their limits are in
  [Irrep labels and the frame of a structure](theory-representations.md#irrep-labels-and-the-frame-of-a-structure)).
- **How a time-reversal-invariant q is written** (`q`, `q + G` or `-q`) does not
  change the structure of a given irrep label, occurrence and component: the
  k-th mode space carrying a label writes the same file for every spelling. The
  mode-space numbers themselves follow the order in which the irreps are
  listed at the q given, and that order can differ between spellings (SrTiO3
  I4/mcm lists M3+ first at `0.5 0.5 -0.5` and M1+ first at `1.5 0.5 -0.5`), so
  pick the mode space by its label in the listing of the q you use.
- **An irrep that occurs several times** has one mode space per occurrence.
  Without force constants nothing selects a combination of them, and any
  combination freezes into the same subgroup. The spaces are an orthonormal
  basis of all the patterns of that irrep fixed by the structure itself, not
  the copies the projection happens to return (those depend on how q is
  written): they are split by the atom orbit they live on (in the order of
  the first atom of each orbit), then by how the atoms move along the
  conventional crystal axes, then by the products of the displacements of
  neighbouring atoms, and listed in that order.
- **A complex irrep and its conjugate**, which time reversal joins into one real
  space at a time-reversal-invariant q, share the real partners of that space:
  the first half belongs to the irrep whose label comes first (T1 before T2,
  GM2+ before GM3+), whichever of the two is listed first.
- **At any other q** the component is a complex Bloch wave whose global phase is
  a convention, as in `--modulation`, and the space group of the structure
  depends on it; for an irrep of dimension two or more the phases of the
  components relative to the first follow the irrep matrices built at the q
  given, so `q + G` can write another structure for them.
- **The cell is the spglib primitive cell of the input** (the
  `Inputed cell was converted into primitive cell` note), which is also the cell
  the supercell is built from; the q coordinates and the Bloch factors refer to
  it. The commands print that note on stdout; in Python,
  `phonon.SymmetryOnlyVibrations(cell, quiet=True)` runs the same analysis
  without it (for library use, and for the MCP server, whose stdout is its
  protocol stream).

### Spectroscopic activity at Γ

*Example directory: `example/29_phonon_activity` (testsuite section 29); after pip install: `crystod-example 29`*

At `--qpoint GM` every mode space gets the Mulliken symbol and the activity of
its irrep (the rules of [Spectroscopic activity](#spectroscopic-activity)),
followed by the summary, one irrep per line, from the structure alone:

```bash
crystod-phonon --vibration -c 221_PPOSCAR_SrTiO3 --qpoint GM
```

```
...
* Irrep-grouped vibration spaces *
  Mode Space  1: irrep = GM4-(3) [T1u], dimension = 3, component numbers = 1..3, activity = IR
  Mode Space  2: irrep = GM4-(3) [T1u], dimension = 3, component numbers = 1..3, activity = IR
  Mode Space  3: irrep = GM4-(3) [T1u], dimension = 3, component numbers = 1..3, activity = IR
  Mode Space  4: irrep = GM4-(3) [T1u], dimension = 3, component numbers = 1..3, activity = IR
  Mode Space  5: irrep = GM5-(3) [T2u], dimension = 3, component numbers = 1..3, activity = silent

* Gamma-point activity *
GM4- [T1u] x4 (3 IR, 1 acoustic)
GM5- [T2u] x1 (silent)
  (acoustic: one set per occurrence of the irrep in the vector representation;
   without force constants the mode spaces are symmetry-adapted patterns,
   not normal modes)
...
```

The symbol in square brackets is the Mulliken symbol of the irrep, printed at
Γ only, after the label in every Γ block; its conventions (phonopy's tables
and axis convention, B1/B2 of the standard setting) are those of
[section 29](#29-phonon-activity-ir-raman-mode-charges).

Without force constants the mode spaces are symmetry-adapted patterns, not
normal modes, so the acoustic set is not one of them: the summary counts it
from the vector representation (one set per occurrence of an irrep there).
This is the Γ table of the Bilbao SAM program (SrTiO3: 4 T1u + T2u). With
`--export-npz` the file also holds an `activities` array, one entry per mode
space. Away from Γ nothing is added.

The Γ run also breaks the mode spaces down by orbit of equivalent atoms (the
Wyckoff-position table of SAM):

```
* Wyckoff-orbit breakdown (Gamma) *
  Wyckoff orbit Sr (1a): GM4- [T1u]
  Wyckoff orbit Ti (1b): GM4- [T1u]
  Wyckoff orbit O (3c): 2 GM4- [T1u] + GM5- [T2u]
  sum over the orbits: 4 GM4- + GM5- (equals the mode-space list)
```

For each orbit (spglib's equivalent atoms, with the Wyckoff letter and its
multiplicity in the conventional cell) the displacements of its atoms form a
representation whose character for an operation is the number of orbit atoms
it maps onto themselves (modulo lattice translations) times `tr R`; that
representation is decomposed into the Γ irreps of the mode spaces. The last
line checks that the orbits add up to the mode-space list. The library form
is `crystod.phonon.wyckoff_orbit_decomposition`. `--raman-tensor` prints the
Raman tensors after the breakdown (see
[Raman tensors](#raman-tensors---raman-tensor)); at any other q point it is an
error. The whole Γ output of wurtzite ZnO:

```bash
crystod-phonon --vibration -c 186_PPOSCAR_ZnO --qpoint GM --raman-tensor
```

```
 ### Inputed cell was converted into primitive cell. ###

* Q points (primitive) *
  GM       (0, 0, 0)
  A        (0, 0, 1/2)
  K        (1/3, 1/3, 0)
  H        (1/3, 1/3, 1/2)
  M        (1/2, 0, 0)
  L        (1/2, 0, 1/2)

* Selected Q point *
  GM       (0, 0, 0)

* Irrep-grouped vibration spaces *
  Mode Space  1: irrep = GM1(1) [A1], dimension = 1, component numbers = 1..1, activity = IR+Raman
  Mode Space  2: irrep = GM1(1) [A1], dimension = 1, component numbers = 1..1, activity = IR+Raman
  Mode Space  3: irrep = GM4(1) [B1], dimension = 1, component numbers = 1..1, activity = silent
  Mode Space  4: irrep = GM4(1) [B1], dimension = 1, component numbers = 1..1, activity = silent
  Mode Space  5: irrep = GM5(2) [E2], dimension = 2, component numbers = 1..2, activity = Raman
  Mode Space  6: irrep = GM5(2) [E2], dimension = 2, component numbers = 1..2, activity = Raman
  Mode Space  7: irrep = GM6(2) [E1], dimension = 2, component numbers = 1..2, activity = IR+Raman
  Mode Space  8: irrep = GM6(2) [E1], dimension = 2, component numbers = 1..2, activity = IR+Raman

* Gamma-point activity *
GM1 [A1] x2 (1 IR+Raman, 1 acoustic)
GM4 [B1] x2 (silent)
GM5 [E2] x2 (Raman)
GM6 [E1] x2 (1 IR+Raman, 1 acoustic)
  (acoustic: one set per occurrence of the irrep in the vector representation;
   without force constants the mode spaces are symmetry-adapted patterns,
   not normal modes)

* Wyckoff-orbit breakdown (Gamma) *
  Wyckoff orbit Zn (2b): GM1 [A1] + GM4 [B1] + GM5 [E2] + GM6 [E1]
  Wyckoff orbit O (2b): GM1 [A1] + GM4 [B1] + GM5 [E2] + GM6 [E1]
  sum over the orbits: 2 GM1 + 2 GM4 + 2 GM5 + 2 GM6 (equals the mode-space list)

* Raman tensors (Cartesian axes of the input cell) *
  GM1 [A1] (2 tensors):  [[a, 0, 0], [0, a, 0], [0, 0, 0]]   [[0, 0, 0], [0, 0, 0], [0, 0, a]]
  GM5 [E2] (2 tensors):  [[a, 0, 0], [0, -a, 0], [0, 0, 0]]   [[0, a, 0], [a, 0, 0], [0, 0, 0]]
  GM6 [E1] (2 tensors):  [[0, 0, 0], [0, 0, a], [0, a, 0]]   [[0, 0, a], [0, 0, 0], [a, 0, 0]]
```

Both 2b orbits give A1 + B1 + E2 + E1 as in SAM: the B1 pair is silent, E2 is
Raman active only, and A1 and E1 are IR and Raman active, as observed for
wurtzite ZnO. At any other q point the same layout is printed without the
Γ-only blocks and without the brackets.

## 35. Subgroups from imaginary modes (`--subgroup`)

*Testsuite section 35*

When a phonon band goes imaginary, the structure is unstable against the
distortion of that mode, and the symmetry it can lower to is fixed by the
irrep of the mode. `--subgroup` labels every imaginary mode and lists the
isotropy subgroups of its irrep — the possible daughter phases — in one step:

```bash
crystod-phonon --subgroup -c 221_PPOSCAR_SrTiO3 --dim "4 4 4"
```

```
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
```

Without `--qpoint`, every q point commensurate with the supercell is scanned
(64 of them for a 4x4x4 supercell) and the imaginary levels are reported most
unstable first, so the unstable q points do not have to be known in advance.
With `--qpoint R` (a label or three coordinates) only that q point is analyzed.
The name in parentheses is that of the tabulated star q belongs to, in the
frame of the labels; a q point outside the tabulated stars (a point on a
line, or the -k partner `PA` of `P` in I-4) is printed without a name, its
levels carrying their ISO-IR labels (`PA3`, ...).
`--threshold` sets the frequency below which a mode counts as imaginary
(default `-0.1` THz; the acoustic modes at Γ, rigid translations, are never
listed, whatever the threshold), and `--yaml phonopy_params.yaml` may be used
instead of `--dim`/`-c`.

Each listed direction is one order-parameter direction of the degenerate
level. Freezing in a single eigenvector explores only one of them: the R-point
triplet of cubic SrTiO3 gives `I4/mcm` along `(0,0,a)`, but the same triplet
also reaches `R-3c` along `(a,a,a)` and `Imma` along `(0,a,a)`. The same
enumeration is available programmatically as
[`crystod.phonon.imaginary_mode_subgroups`](python-api.md).

### Generating the daughter structures (`--modulate`)

`--modulate` runs the [`--modulation`](#33-phonon-modulation---modulation) step
for every listed direction, so the candidate structures of a structure search
come out of the same command:

```bash
crystod-phonon --subgroup -c 221_PPOSCAR_SrTiO3 --dim "4 4 4" --qpoint R --modulate
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

One POSCAR per direction, plus the `--modulation` command that reproduces it —
copy-paste it as printed and it rewrites exactly that file, so an amplitude or
a mode can be changed afterwards without working out the combination again.
The file is named after the space group `--modulate` measured at 1e-5; the
command reports the one at `--modulation`'s default 0.1 Å, where a direction
with a small component can read as a more symmetric neighbour (`R5-(a,b,c)`
above prints `C2/c`): add `--tolerance 1e-5` to the command to see the same
group.
`--amplitude` scales all of them (default 0.3 Å). The names read
`MPOSCAR_{q}_{irrep}_{direction}_{spacegroup}`; when one q point carries two
imaginary levels of the *same* irrep, the second gains a `_mode{n}` suffix
(`n` = its first mode number) so the two sets do not overwrite each other.

Which combination of the degenerate modes realizes which direction is not fixed
by any convention CrystOD could assume, so it is **measured, not assumed**:
candidate combinations are generated (all with positive coefficients first,
then the same ones with sign changes, which directions such as
`N1+(a;-a;a;a)` need), the space group of each generated
structure is determined with spglib, and the (space group, cell size, index)
triple is matched against the enumerated table. When two directions of one
irrep share that triple — `R5+` of Pm-3m puts both `(0,a,b)` and `(a,a,b)` at
C2/m, size 2, index 24 — the conventional cell metric separates them, so a
domain of one is never written out under the other's name, and the pattern of
zero and equal components decides which label a structure takes.

A direction that no candidate reproduces is **reported**, never guessed at:

```
  note: no candidate reproduced M5-(0,0;a,0;0,a) (I4/mmm); generate it with --modulation by hand.
```

Every enumerated row therefore ends up either as a file or as a note. The
directions that need a note belong to large order parameters (a multi-arm star
times a degenerate level), where the candidate search is cut off before it
reaches them — not only low-symmetry ones: for X5- of cubic ScF3 (three arms
times a pair) the six-component directions such as `X5-(a,a;a,a;a,a)` (R-3c)
are among them.

A star with several arms is handled the same way, through the multi-q form of
`--modulation` — the M point of a perovskite needs one arm for `M3+(0;0;a)`,
two for `(0;a;a)` and three for `(a;a;a)`:

```
M3+(0;0;a)             P4/mbm     -> MPOSCAR_M_M3+_0_0_a_P4mbm
    crystod-phonon --modulation -c 221_PPOSCAR_ScF3 --dim "4 4 4" --qpoint 0.5 0.5 0 --mode 1 --amplitude 0.3
M3+(0;a;a)             I4/mmm     -> MPOSCAR_M_M3+_0_a_a_I4mmm
    crystod-phonon --modulation -c 221_PPOSCAR_ScF3 --dim "4 4 4" --qpoint1 0.5 0.5 0 --mode1 1 --amplitude1 0.3 --qpoint2 0.5 0 0.5 --mode2 1 --amplitude2 0.3
M3+(a;a;a)             Im-3       -> MPOSCAR_M_M3+_a_a_a_Im-3
    crystod-phonon --modulation -c 221_PPOSCAR_ScF3 --dim "4 4 4" --qpoint1 0.5 0.5 0 --mode1 1 --amplitude1 0.3 --qpoint2 0.5 0 0.5 --mode2 1 --amplitude2 0.3 --qpoint3 0 0.5 0.5 --mode3 1 --amplitude3 0.3
```

## 51. Phonon band unfolding (`--unfold`)

*Example directory: `example/51_phonon_unfold` (testsuite section 51); after pip install: `crystod-example 51`*

The phonons of a distorted phase are computed in its own, larger cell and
come out folded into its smaller Brillouin zone: Pnma CaTiO3 (20 atoms, four
cubic cells) has 60 branches at every k, and which of them carries a cubic
mode cannot be read off its band plot. `--unfold` draws the phonons of the
distorted structure (the *child*) in the Brillouin zone of its high-symmetry
*parent* instead. Child mode J, computed at the child wave vector K = M k,
appears at the parent wave vector k with the weight P_J(k) of its projection
onto the Bloch states of the parent translations (P. B. Allen, T. Berlijn,
D. A. Casavant and J. M. Soler, Phys. Rev. B 87, 085322 (2013)), and every
unfolded level is labeled with its parent irrep. The modes that are
imaginary in the parent, the octahedral tilts R4+ and M3+ of cubic CaTiO3 and
its polar Γ mode GM4- [T1u], reappear in the child at real frequencies, and the
run lists the child Γ levels they go into.

```bash
cd example/51_phonon_unfold/CaTiO3_Pnma
crystod-phonon --unfold --child-poscar child/62_PPOSCAR_CaTiO3 --child-dim 2 2 2 \
               --parent-poscar parent/221_PPOSCAR_CaTiO3 --parent-dim 2 2 2
# -> phonon_unfold.pdf, phonon_unfold.dat
```

What is read:

- the child: its POSCAR (`--child-poscar`, or `-c`) and the `FORCE_SETS` of
  its supercell (`--child-dim`, or `--dim`; `FORCE_CONSTANTS` with
  `--child-readfc`), next to the POSCAR, else in the working directory;
- the parent: its POSCAR (`--parent-poscar`, always needed), which fixes the
  transformation, the band path and the irrep labels. Its force data are
  optional: with `--parent-dim` its `FORCE_SETS` (`FORCE_CONSTANTS` with
  `--parent-readfc`) is read next to the parent POSCAR, and the run adds the
  parent bands, the parent levels at the special points and the child levels
  the soft parent modes go into.

In each place a force file named after the POSCAR comes first:
`FORCE_SETS_POSCAR_child` next to `POSCAR_child` (the name macer gives the
force data of each structure; `FORCE_CONSTANTS_POSCAR_child` with
`--child-readfc`), then `FORCE_SETS`. macer names that file after the POSCAR
it was given, the unrelaxed input, while the forces belong to the cell it
relaxed (the `POSCAR` and `CONTCAR-<name>` it writes): give the relaxed cell
as the POSCAR, with the force file named after it or as `FORCE_SETS` next to
it. With these per-structure names, child and parent can share one
directory: with `POSCAR_child`, `POSCAR_parent`, `FORCE_SETS_POSCAR_child`
and `FORCE_SETS_POSCAR_parent` in the working directory the command is
literally

```bash
crystod-phonon --unfold --child-poscar POSCAR_child --child-dim 2 2 2 --parent-poscar POSCAR_parent --parent-dim 2 2 2
```

The example keeps the two structures in two directories instead (`child/`
and `parent/`, each with its `FORCE_SETS`). When the child and the parent
would read the same file the run stops with an ERROR. Every phonopy load runs
in an empty temporary directory with absolute paths, so a `FORCE_CONSTANTS`
lying in the working directory is never read in place of `FORCE_SETS` (the
trap described in section 28). The two structures need no common setting,
origin or atom order: the transformation and the pairing of the atoms are
worked out from the structures themselves. The run prints seven blocks,
explained below, and takes about 13 s for this example on a Mac Studio M3
Ultra (10 s for the AlF3 example, an R-3c child of 8 atoms).

```{figure} images/phonon_unfold_CaTiO3.png
:name: fig-phonon-unfold-catio3
:width: 100%

Page 1 of `phonon_unfold.pdf` for Pnma CaTiO3 unfolded onto Pm-3m (2x2x2
`FORCE_SETS` for both). Top left: the parent bands, with the imaginary tilts
at M (M3+, -5.44 THz) and R (R4+, -5.74 THz) and the polar mode GM4- [T1u] at
Γ (-1.44 THz) drawn below zero. Top middle: the child along the same path (one
domain), marker area and color proportional to the weight, with the child
point of every vertex on top; the child has no imaginary mode, and the weight
of the tilts sits on child modes between about 4 and 10 THz. Top right: the
average over the six orientation domains. Bottom: every arm of the stars of
the path vertices; the shaded columns are the arms that fold onto the child Γ
point, among them the M arm (1/2, 0, 1/2) of the in-phase tilt, which the path
does not pass. The notes beside it give the pseudo-parent cell and explain the
panels.
```

### The transformation

The transformation between the two cells is not an input. It is computed
from the two structures by the symmetry-mode analysis of
[`crystod-group --supergroup-cif PARENT --subgroup-cif CHILD`](crystod-group.md#23-symmetry-mode-analysis---supergroup-cif),
every child atom is then paired with its ideal site in that setting, and the
setup block prints the setting in two conventions:

```
* Unfolding setup *
child          child/62_PPOSCAR_CaTiO3: Pnma (No. 62), 20 atoms, cell used as given
child forces   child/FORCE_SETS, supercell 2 2 2
parent         parent/221_PPOSCAR_CaTiO3: Pm-3m (No. 221), 5 atoms, cell used as given
parent forces  parent/FORCE_SETS, supercell 2 2 2
setting        crystod-group --supergroup-cif parent/221_PPOSCAR_CaTiO3
               --subgroup-cif child/62_PPOSCAR_CaTiO3 (1 of 24 equivalent settings)
transformation parent -> child as crystod-group prints it (columns of P: the child basis in
               parent units, (a', b', c') = (a, b, c) P; last column: the origin shift p,
               x_parent = P x_child + p):
  [     1     0    -1 ] [   1/2 ]
  [     0     2     0 ] [     0 ]
  [     1     0     1 ] [   1/2 ]
M (rows)       1 0 1  0 2 0  -1 0 1
               (A_child = M A_parent, k_child = M k_parent; the form --trans-mat takes)
origin shift   p = (1/2, 0, 1/2) (parent fractional)
cells          N = 4 parent cells in the child cell; n_child = 20, n_parent = 5 atoms (Ca Ti O O O)
pseudo-parent  A_pp = inv(M) A_child: a = 3.8182 3.8078 3.8182 A, angles 90.000 88.983 90.000 deg
strain         lengths -0.69 -0.96 -0.69 %, volume -2.35 % against the parent cell (a = 3.8449 A);
               rotation 45.0 deg
displacement   max 0.4278 A (O), rms 0.3442 A in the child lattice;
               0.4314 A in the strain-free parent lattice
               (crystod-group: 0.4314 A)
condensed      X5+ 0.4889, M2+ 0.0095, M3+ 0.8660, R4+ 1.1901, R5+ 0.0954 A
               (crystod-group mode amplitudes)
folding        parent k folding onto the child Gamma point: GM (0, 0, 0), X (0, 1/2, 0),
               M (1/2, 0, 1/2), R (1/2, 1/2, 1/2)
```

- The three rows `[ 1 0 -1 ] [ 1/2 ]`, ... are the setting exactly as
  crystod-group prints it for the same two files (its
  `* Transformation matrix (parent -> child) *` block, the AMPLIMODES
  convention): the columns of P are the child basis vectors in parent units,
  the last column is the origin shift p, and `x_parent = P x_child + p`.
- `M (rows)` is the same setting in the convention of the unfolding and of
  `--trans-mat`: the rows of M are the child lattice vectors in parent units,
  `A_child = M A_parent` with the lattice vectors as rows, and fractional wave
  vectors transform as `k_child = M k_parent`. M is the transpose of P when,
  as here, the two files are the cells of the unfolding; a centered parent
  (a conventional I4/mmm file, say) is unfolded on its primitive cell, and M
  then refers to that cell. `origin shift` is the child origin in parent
  fractional coordinates.
- `pseudo-parent` is the parent cell as it sits inside the child,
  `A_pp = inv(M) A_child`, and `strain` compares it with the parent cell:
  -2.35 % in volume for CaTiO3, -9.66 % for AlF3, whose tilt pulls the
  lattice in. The weights do not depend on this strain (they depend only on
  the translations and the atom sites), but the parent and the child panels
  are compared at the same fractional k, which for AlF3 means two cells of
  quite different volume. `rotation` is the rigid rotation between the child
  and the parent frames (polar decomposition), applied to the unfolded
  vectors before the irrep projection.
- `displacement` is the largest displacement of a child atom from its ideal
  site; measured in the strain-free parent lattice it is crystod-group's
  maximum atomic displacement. `condensed` lists crystod-group's mode
  amplitudes, and `folding` the N = det M parent k that fold onto the child
  Γ point.

The child sits in the parent in 24 equivalent settings here (`M0 W^T` for
the 24 proper rotations W of m-3m), which describe one crystal in
differently oriented frames. `--trans-mat` chooses another of them: nine
integers, row-wise, `child = T @ parent` (the crystod-bz convention). The
origin shift is not an option; it follows from the setting (`p = W p0 + w`
for the parent operation (W, w) printed in the setting row), and a matrix
that is not one of the equivalent settings is refused with the list of the
valid ones. With the long Pnma axis along the parent c:

```bash
crystod-phonon --unfold --child-poscar child/62_PPOSCAR_CaTiO3 --child-dim 2 2 2 \
               --parent-poscar parent/221_PPOSCAR_CaTiO3 --parent-dim 2 2 2 \
               --trans-mat 1 1 0  0 0 2  1 -1 0 -o phonon_unfold_c
```

```
* Unfolding setup *
...
setting        --trans-mat 1 1 0  0 0 2  1 -1 0: the crystod-group setting 1 0 1  0 2 0  -1 0 1
               transformed by the parent operation W = 0 0 1  1 0 0  0 1 0, w = (0, 0, 0)
               (24 equivalent settings)
...
M (rows)       1 1 0  0 0 2  1 -1 0
               (A_child = M A_parent, k_child = M k_parent; the form --trans-mat takes)
origin shift   p = (1/2, 1/2, 0) (parent fractional)
...
folding        parent k folding onto the child Gamma point: GM (0, 0, 0), X (0, 0, 1/2),
               M (1/2, 1/2, 0), R (1/2, 1/2, 1/2)
```

Settings related by an operation of the child describe the same domain and
give the same spectrum: crystod-group's setting for AlF3,
`-1 -1 0  -1 0 -1  0 -1 -1`, and `--trans-mat 1 1 0  0 1 1  1 0 1` differ by
a two-fold rotation of the R-3c child, and their frequencies, weights and
labels are identical (`example/51_phonon_unfold/AlF3_R-3c/README`). Other
settings put other arms of a star on the path.

### Which path

The path is the parent's: the seekpath path of the parent primitive cell,
the one `--fatband` uses (GM-X-M-GM-R-X|R-M for Pm-3m, 51 points per leg),
in fractional coordinates of the parent reciprocal basis; `--band`,
`--band-labels` and `--npoints` give another one in the same coordinates.
At every parent k the child is computed at K = M k, and the path block prints
where each vertex lands in the child:

```
* Unfolding path *
path           GM-X-M-GM-R-X|R-M (seekpath, parent primitive basis), 51 points per leg, 306 k
distances      1/A without 2 pi, reciprocal metric of the pseudo-parent cell A_pp
vertex  k (parent)           K = M k (child)       child point
GM      (0, 0, 0)            (0, 0, 0)             GM
X       (0, 1/2, 0)          (0, 1, 0)             GM
M       (1/2, 1/2, 0)        (1/2, 1, -1/2)        U
GM      (0, 0, 0)            (0, 0, 0)             GM
R       (1/2, 1/2, 1/2)      (1, 1, 0)             GM
X       (0, 1/2, 0)          (0, 1, 0)             GM
R       (1/2, 1/2, 1/2)      (1, 1, 0)             GM
M       (1/2, 1/2, 0)        (1/2, 1, -1/2)        U
```

The arms of one parent star are no longer equivalent in the child. X
(0, 1/2, 0), the arm along the long Pnma axis, folds onto the child Γ point,
where the X5+ distortion is condensed, while X (1/2, 0, 0) and X (0, 0, 1/2)
go to the child U point. Of the three M arms, (1/2, 1/2, 0) on the path and
(0, 1/2, 1/2) go to U; the condensed in-phase tilt M3+ lives on the third
arm, (1/2, 0, 1/2), which the path does not pass. The star-arm block lists
every arm of the stars of the path vertices; arms with the same class letter
are related by an operation of the child or by time reversal and give the
same unfolded spectrum, and the lower panel of page 1 draws them all:

```
* Star arms of the path vertices *
point groups   child 8 operations (Pnma), parent 48 (Pm-3m): 6 orientation domains
star  arm k (parent)         K = M k mod 1        child point  to child GM  path  class
GM    (0, 0, 0)              (0, 0, 0)            GM           yes          path  a
X     (0, 1/2, 0)            (0, 0, 0)            GM           yes          path  a
X     (1/2, 0, 0)            (1/2, 0, 1/2)        U            -            -     b
X     (0, 0, 1/2)            (1/2, 0, 1/2)        U            -            -     b
M     (1/2, 1/2, 0)          (1/2, 0, 1/2)        U            -            path  a
M     (1/2, 0, 1/2)          (0, 0, 0)            GM           yes          -     b
M     (0, 1/2, 1/2)          (1/2, 0, 1/2)        U            -            -     a
R     (1/2, 1/2, 1/2)        (0, 0, 0)            GM           yes          path  a
  (arms with the same class letter are related by a child operation or by time reversal and give the
   same unfolded spectrum; 'to child GM' = the arm folds onto the child Gamma point, where a
   condensed distortion lives)
  NOTE: the condensed arm M (1/2, 0, 1/2) is not on the band path; it is shown in the lower panel of
        page 1 and in the child GM mode origin.
```

In the setting of `--trans-mat 1 1 0  0 0 2  1 -1 0` the condensed arms are
X (0, 0, 1/2) and M (1/2, 1/2, 0): the M point of the path is then the
in-phase tilt, and the X point of the path goes to the child U point. No
setting puts both condensed arms on the path, because the condensed X arm
lies along the long axis and the condensed M arm has no component along it,
while the path X (0, 1/2, 0) and M (1/2, 1/2, 0) share the b direction:

```
* Star arms of the path vertices *
...
star  arm k (parent)         K = M k mod 1        child point  to child GM  path  class
...
X     (0, 1/2, 0)            (1/2, 0, 1/2)        U            -            path  a
X     (1/2, 0, 0)            (1/2, 0, 1/2)        U            -            -     a
X     (0, 0, 1/2)            (0, 0, 0)            GM           yes          -     b
M     (1/2, 1/2, 0)          (0, 0, 0)            GM           yes          path  a
...
  NOTE: the condensed arm X (0, 0, 1/2) is not on the band path; it is shown in the lower panel of
        page 1 and in the child GM mode origin.
```

The right panel of page 1 is the domain average. A crystal of the child
phase usually holds all its orientation domains, `|G_parent| / |G_child|` of
them (48/8 = 6 for Pnma, 48/12 = 4 for R-3c), and a parent k meets each
domain in another arm of its star. The average over all arms of the star of
k (every arm is unfolded; m equivalent arms out of n carry m/n of the
weight) is the spectrum of a crystal with equal domain populations: it has
the cubic symmetry again, its weights are the same in every equivalent
setting, and it is written as the second dataset of the `.dat` file.

### Weights

The weight of child mode J at the parent k is the squared norm of its
projection onto the Bloch states of the parent translations:

```
u_J(k; kappa, alpha) = N^(-1/2) sum_{s: kappa(s) = kappa} exp(-2 pi i G.x0_s) exp(2 pi i K.d_s) e_J(K; s, alpha) exp(-2 pi i k.p)
P_J(k) = sum_{kappa, alpha} |u_J(k; kappa, alpha)|^2          (M k = K + G)
```

with `e_J` the phonopy eigenvector of the child (mass-weighted, unit norm),
s the child atoms on the sites of parent atom kappa, `x0_s` the ideal site
of atom s, `d_s = x_s - x0_s` its displacement and p the origin shift.
P_J(k) lies between 0 and 1, and two sum rules hold to round-off: at every k
the weights of all child modes add up to 3 n_parent (15 for CaTiO3; the
summary block prints the deviation, and above 1e-6 the run stops with an
internal ERROR), and the weights of one child level (its degenerate modes
together) over the N parent k that fold onto its K add up to its number of
modes, 1 for a nondegenerate mode. An undistorted supercell of the parent
gives weights of 0 or 1 at exactly the parent frequencies; the further the
weights spread over the child modes, the more the distortion mixes the parent
branches.

Degenerate child modes (`|df| <= 1e-4` THz) are rotated within their level so
that their unfolded vectors are orthogonal before the weights are taken: the
weight of one mode of a degenerate pair would otherwise depend on the
eigenvector basis phonopy returns. The rotation is made at each k separately,
so the sum rule over the N parent k holds per level, not per mode. Inside a
level only the sum has a physical meaning: the rotated modes take the places
of the level's members in index order (the largest weight first), and where two
levels closer than 1e-4 THz form one group, the pairing of a per-mode weight
with a frequency is arbitrary; read such groups by their P_g. Page 1 draws one
marker per child mode and leaves out the modes with a weight below
`--weight-min` (default 0.02); page 2, the labels and the tables work per
degenerate group with its weight P_g, and a group with P_g below
`--weight-min` is not drawn and gets the label `-`. The `.dat` file keeps
every row.

The factor `exp(2 pi i K.d_s)` converts phonopy's eigenvectors, whose Bloch
phase refers to the actual atomic positions, to the ideal sites the parent
translations act on. phonopy's own `phonopy.unfolding.core.Unfolding` keeps the
actual-position phase: on these distorted children its weights differ from
the ones above by up to 0.07 (CaTiO3) and 0.14 (AlF3), and they change with
the supercell the force constants are written in. `--unfold` does not use it
(see [CrystOD for phonopy users](for-phonopy-users.md#band-unfolding---unfold-and-phonopys-unfolding)).

### Parent irreps of the unfolded levels

The unfolded vector of a child mode lives in the space of the parent
displacements at k, so the projection operators of the irreps of the little
group of k in the parent split it, `P_J^(i)(k) = ||Pi_i(k) u_J(k)||^2` after
the rigid rotation of the setup block has brought it into the parent frame.
These are the projectors behind `--vibration` (section 34), and the labels
are the ISO-IR labels `--irreps` gives the parent POSCAR, with the Mulliken
symbol at Γ. Like those, they refer to the origin of the parent file: with
Ti at the origin the two tilts are R4+ and M3+, with Ca at the origin the same
tilts are R5- and M2+ (see
[Irrep labels and the frame of a structure](theory-representations.md#irrep-labels-and-the-frame-of-a-structure)).

A degenerate group of child modes gets one label for all its members:

- the irrep holding at least 50 % of the group weight P_g (`M3+`);
- else the smallest set of irreps that the child symmetry cannot separate
  and that holds at least 50 % together, written `A/B`: at k = (0.49, 0.49, 0)
  on the Σ line the level at 3.559 THz has 39 % SM1, 39 % SM2 and 22 % SM3
  and is labeled `SM1/SM2`;
- else `mixed`, and `-` below `--weight-min`.

The shares are compared after rounding to 1e-9, so round-off never decides:
of two irreps with exactly 50 % each, the one first in the natural order of
the irreps (GM1+, GM1-, GM2+, ...) is the label, and the fractions column
lists both; of two sets with the same number of irreps and the same share,
the set whose sorted names come first.

The child symmetry cannot separate two irreps when one child level can hold
weight in both: when they share an irrep of the child operations that fix
k, or when an irrep of those operations in the one and an irrep in the other
occur together in one irrep of the little group of K = M k in the child (or
in its time-reversal partner). The second case arises where a child
operation, or time reversal, takes k to another arm that folds onto the same
K and joins the partners of a degenerate child level: k = (0, 1/4, 0) of
CaTiO3 folds onto the child Y point, the two partners of a two-dimensional
Y2 level carry different irreps of the operations fixing k, and the level at
3.837 THz holds 87 % DT5 and 13 % DT1, so DT1, DT2 and DT5 form one set
there. Within such a set the mixing is real, and the label is a
decomposition of the unfolded vector, not a symmetry of the child. The block
prints these sets point by point (`not separated by the child symmetry:`);
where every irrep is separated, as at X of CaTiO3, every level is pure to
round-off (unless two child levels happen to be degenerate). The weight of
every level lies only in parent irreps that share an irrep of the operations
fixing k with it; the summary checks this (`weight forbidden by the child
symmetry`). A pair of irreps that time reversal joins into
one physically irreducible representation is one unit, written as
`label_phonon_modes` writes it (`S1+S2`, `U2++U3+` or `R1+R2` when a Pnma cell
is unfolded onto itself); the cubic parents of the examples have no such
pairs. `--no-irreps` skips the labels (no irreps block, a one-page PDF, about
a third faster).

```
* Unfolded irreps at special points *
...
M (1/2, 1/2, 0) -> child U
  parent levels: M3+ -5.4389, M5- 1.9796 x2, M2- 2.3734, M3- 4.3106, M5- 7.0016 x2, M5+ 8.4297 x2,
                 M4+ 12.9346, M5- 13.7378 x2, M3- 15.3635, M2+ 17.3250, M1+ 24.5424
  not separated by the child symmetry: {M1+, M2+, M3+, M4+, M5+}, {M2-, M3-, M5-}
  freq(THz)   n    P_g  label              fractions
     2.8034   2  0.691  M3-                M3- 0.932, M5- 0.068
     2.8140   2  0.453  M2-                M2- 0.677, M5- 0.323
     3.1479   2  0.352  M5-                M5- 0.926, M2- 0.074
     3.5553   2  0.758  M5-                M5- 0.833, M3- 0.167
     3.9796   2  0.447  M3+                M3+ 0.953, M5+ 0.032, M4+ 0.015
     4.1702   2  0.289  M3-                M3- 0.798, M5- 0.202
     5.1424   2  0.768  M2-                M2- 0.586, M5- 0.414
     5.6350   2  0.435  M3+                M3+ 0.831, M5+ 0.162
...
R (1/2, 1/2, 1/2) -> child GM
  parent levels: R4+ -5.7403 x3, R5+ 3.4194 x3, R4- 12.4092 x3, R5+ 12.7656 x3, R3+ 17.1171 x2,
                 R1+ 25.2933
  not separated by the child symmetry: {R1+, R3+, R4+, R5+}
  freq(THz)   n    P_g  label              fractions
     4.3684   1  0.348  R4+                R4+ 0.679, R5+ 0.321
     4.6861   1  0.233  R5+                R5+ 0.518, R4+ 0.482
     5.1343   1  0.796  R4+                R4+ 0.959, R5+ 0.041
     5.2623   1  0.337  R4+                R4+ 0.667, R5+ 0.333
     5.5078   1  0.926  R4+                R4+ 1.000
     5.7633   1  0.932  R5+                R5+ 1.000
...
  (rows: the degenerate groups of child modes (|df| <= 1e-4 THz) with P_g >= 0.10; the .dat file has
   all groups. The label is the irrep holding at least 50 % of P_g, else the smallest set of irreps
   that the child symmetry cannot separate and that together hold at least 50 % (written A/B), else
   mixed; two irreps that time reversal joins count as one (written S1+S2).)
```

The rows are the degenerate groups of child modes with P_g >= 0.10 at the
distinct path vertices (the `.dat` file has all of them), with the number of
modes n, the label and the largest irrep fractions. `parent levels` (with
`--parent-dim`) are the levels of the parent itself, labeled with the same
projectors. The M3+ level of the M arm on the path (-5.44 THz in the
parent; this arm is not the condensed one) reappears in the child mainly at
3.98 and 5.64 THz, and the R4+ weight sits at 4.4 to 9.6 THz, mostly on the
levels at 5.13 and 5.51 THz.

```{figure} images/phonon_unfold_irreps_CaTiO3.png
:name: fig-phonon-unfold-irreps-catio3
:width: 100%

Page 2 of `phonon_unfold.pdf`: the parent bands colored by their parent
irrep, the unfolded child colored by the label of each degenerate group (one
marker per group, area = P_g, gray for `mixed`), and the groups at GM, X, M
and R as bars of length P_g split into their irreps, named where
P_g >= 0.5. The legend has one row per k type, with the Mulliken symbols at
Γ; colors repeat between k types.
```

### Where the child Γ modes come from

The child Γ point collects the N parent k that fold onto it, here GM,
X (0, 1/2, 0), M (1/2, 0, 1/2) and R. The child Γ modes are unfolded onto
all of them, so this block also covers the condensed M arm that the path
misses; the weights of one level add up to its number of modes. Every child
level is listed twice: first with its child irrep and its origin (the
folding k of its largest weight P_g(k), the label there and that weight),
then with its weight and label at each folding k (a child with more folding
k gets several such tables, so that no line is wider than 100 columns). The
child irreps are those `label_phonon_modes` gives the child in its own
setting (Pnma labels, Mulliken symbols and activity), and the parent labels
at GM carry the Mulliken symbols of the parent, as in the other blocks:

```
* Child GM mode origin *
child GM (K = 0), 60 modes, from the parent k folding onto it: GM (0, 0, 0), X (0, 1/2, 0),
  M (1/2, 0, 1/2) (not on the path), R (1/2, 1/2, 1/2)
modes     freq(THz)  n  child irrep                            origin
1,2,3        0.0000  3  GM2-+GM3-+GM4- [B1u+B3u+B2u] acoustic  GM GM4- [T1u] 3.00
4            2.9239  1  GM4- [B2u] IR                          X X5- 0.82
...
11           4.3684  1  GM1+ [Ag] Raman                        M M3+ 0.58
...
15           5.1343  1  GM4+ [B2g] Raman                       R R4+ 0.80
...
18           5.5078  1  GM2+ [B1g] Raman                       R R4+ 0.93
...
  (origin: the folding k with the largest P_g(k) of the level, its label there and that P_g(k))
modes     freq(THz)  n  GM                X             M             R
1,2,3        0.0000  3  3.000 GM4- [T1u]  0.000 -       0.000 -       0.000 -
4            2.9239  1  0.128 GM4- [T1u]  0.821 X5-     0.000 -       0.050 R4-
...
11           4.3684  1  0.000 -           0.071 X5+     0.581 M3+     0.348 R4+
...
15           5.1343  1  0.000 -           0.200 X5+     0.004 -       0.796 R4+
...
18           5.5078  1  0.000 -           0.027 X1+     0.047 M5+     0.926 R4+
...
  (columns: weight P_g(k) of the child level at each folding k and its label there ('-' below 0.02);
   the weights of a level add up to its number of modes)
soft parent levels and the child GM levels they go into (content c_g >= 0.02):
  GM (0, 0, 0) GM4- [T1u] -1.4414 x3: 6.250 B3u 0.60, 3.973 B2u 0.59, 6.247 B1u 0.51,
    7.288 B2u 0.35, 3.292 B3u 0.32, 3.438 B1u 0.26, 9.085 B1u 0.15, 8.585 B3u 0.05, ...;
    mean 5.83 THz
  M (1/2, 0, 1/2) M3+ -5.4389: 4.368 Ag 0.58, 8.103 Ag 0.22, 9.621 Ag 0.12, 7.172 Ag 0.07;
    mean 6.06 THz
  R (1/2, 1/2, 1/2) R4+ -5.7403 x3: 5.508 B1g 0.93, 5.134 B2g 0.76, 8.103 Ag 0.33, 4.368 Ag 0.24,
    5.262 B2g 0.22, 9.621 Ag 0.22, 4.686 Ag 0.11, 7.172 Ag 0.09, ...; mean 6.09 THz
```

Every Raman-active mode of Pnma CaTiO3 comes from X, M or R: the parent Γ
point has only odd modes (T1u and T2u), so its weight on the gerade child
modes is zero. With `--parent-dim` the soft parent levels (below -0.1 THz at
the folding k, a fixed cutoff: `--unfold` refuses `--threshold`) are followed
into the child: the content `c_g` of child level g is the summed squared
overlap of its unfolded vectors with the eigenvectors of the parent level.
The in-phase tilt M3+ goes mainly into the Ag mode at
4.37 THz (content 0.58), the antiphase tilt R4+ into B1g 5.51 THz (0.93),
B2g 5.13 THz (0.76) and a series of Ag and B2g modes, both with a
content-weighted mean near 6 THz; the polar GM4- level of the parent goes
into IR-active modes only.

### Output

The summary block closes the run with the checks and the timings:

```
* Unfolding summary *
k points       306 on the path; 2768 arms of their stars unfolded for the domain average
               (6 orientation domains)
modes          60 child modes at every k; 18360 (k, mode) pairs, 14832 with weight >= 0.02
sum rule       max |sum_J P_J(k) - 15| = 3.4e-14 on the path, 2.5e-14 for the domain average
arm classes    arms of one class agree to 7.1e-13 THz and 8.1e-12 in weight
               (child operations and time reversal)
irreps         projectors 2.5e-15 (completeness, idempotency, hermiticity);
               per-irrep sum rule 1.4e-14; parent levels pure to 4.7e-15;
               weight forbidden by the child symmetry 1.9e-22
labels         ISO-IR labels computed at 16 k and copied along 6 straight runs (0 mismatches);
               groups with P_g >= 0.02: 13900 one irrep, 789 sets (A/B), 0 mixed
timings (CPU)  mapping 2.9 s, phonopy 0.9 s, unfolding 0.07 s, star arms and domain average 0.6 s,
               irreps 2.9 s, child GM origin 0.2 s, figures 5.4 s, total 13.3 s

* Output files *
  Unfolded bands (2 pages) written to: phonon_unfold.pdf
  Weights and labels written to: phonon_unfold.dat
```

- `-o PREFIX` (default `phonon_unfold`) names the two files, `PREFIX.pdf`
  and `PREFIX.dat` (a directory, `-o out/`, gets `out/phonon_unfold.pdf` and
  `.dat`); `--export-npz FILE` adds a third with every array (`FILE.npz`;
  `phonon_unfold.npz` inside a directory).
- The PDF has two pages (one with `--no-irreps`): page 1 with the parent
  bands (without `--parent-dim`, the child modes at K = M k as thin gray
  lines, folded, without weights), the child on the path, the domain average
  and the star arms; page 2 with the parent irreps. With one marker per child
  mode and k, the CaTiO3 file is about 10 MB. Every text is measured in the
  font in use (Arial, or DejaVu Sans where Arial is not installed, as on most
  Linux systems, whose glyphs are roughly 10-15 % wider) and fitted to its room:
  titles and notes are set smaller or wrapped (the pseudo-parent cell and the
  meaning of the panels are in the notes of page 1), labels that would run
  together are set smaller or go into staggered rows, the k labels of many
  star arms (26 for an fcc parent) are set upright, with a wider panel when
  even that does not fit, and the labels of a level panel on page 2 are
  spaced by the height of their glyphs, with 1 pt to spare because
  matplotlib's PDF output can draw such a label up to 1 pt higher than its
  PNG output. A crowded level panel (a parent of 20 atoms has 57 labeled
  levels at Γ when unfolded onto itself) makes the panels taller and its
  labels smaller, down to 3 pt, and beyond that keeps the labels of the
  largest P_g (a note under the legend says so; the terminal block lists
  every level). Test section 51 checks the result on the ink of every text
  of both pages (matplotlib's Agg rendering at 600 dpi, in the module's font
  and in DejaVu Sans) for that crowded case and for Si onto itself (26 star
  arms): no text leaves its page, no two texts overlap by more than 0.25 pt,
  and the labels of a level panel stay inside it and 1 pt apart.
- The `.dat` file (about 6 MB for CaTiO3) holds two datasets separated by two
  blank lines, with the conventions in its header. Dataset 0 is the path, one
  row per k and child mode: `k_index distance kx ky kz mode frequency weight
  group_weight label fraction` (frequencies in THz, imaginary ones negative;
  `fraction` is the share of P_g held by the label). Dataset 1 is the domain
  average, one row per class of equivalent arms and mode: `k_index distance
  class multiplicity n_arms akx aky akz mode frequency weight`, with the
  classes numbered from 1 at each k and `akx aky akz` the arm of the class
  that was unfolded (the k index runs from 0, the modes from 1). In gnuplot:
  `plot 'phonon_unfold.dat' index 0 using 2:7:(sqrt($8)) with points pt 7 ps variable`
  (and `index 1 using 2:10:(sqrt($11))` for the average).
- The `.npz` file (`np.load(FILE, allow_pickle=False)`) holds the setting
  (`M`, `origin_shift`, `setting_P`, `setting_p`, `equivalent_settings`), the
  cells and the atom map (`A_child`, `A_parent`, `A_pp`, `kappa`,
  `ideal_frac_child`, ...), the path (`k_parent`, `K_child`, `distances`,
  ticks), the bands (`frequencies`, `weights`, `weights_canonical`,
  `group_weight`, `degenerate_group`, `vectors_canonical`), the star arms
  (`arm_*`), the domain average (`davg_*`), the irreps (`irrep_units`,
  `irrep_weights`, `group_label`, `group_fraction`, and with `--parent-dim`
  `parent_frequencies`, `parent_labels`), the child Γ origin (`gamma_*`), the
  checks and timings, and the conventions as a string (`conventions`). Per
  mode, `weights` is the phonopy basis, and `weights_canonical`,
  `vectors_canonical`, `arm_weights_canonical`, `davg_weights`,
  `irrep_weights` and `gamma_weights` (canonical at each of the N folding k)
  the rotated degenerate modes, of which only the sums over a level have a
  physical meaning; `group_weight`, `group_label`, `group_fraction`,
  `gamma_group_label` and `gamma_child_irrep` are per level, every mode
  carrying the value of its level.

### Options

| option | meaning |
|---|---|
| `--child-poscar FILE` (`-c`) | the child (distorted) structure; its force file is read next to it, else in the working directory (`FORCE_SETS_<file name>` before `FORCE_SETS`) |
| `--child-dim N N N` (`--dim`) | the supercell of the child force calculation (required) |
| `--child-readfc` (`--readfc`) | read the child's `FORCE_CONSTANTS` instead of `FORCE_SETS` |
| `--parent-poscar FILE` (`--parent`) | the parent structure (required); the path, the labels and the origin refer to its primitive cell |
| `--parent-dim N N N` | the supercell of the parent force calculation: the parent's `FORCE_SETS` (`FORCE_SETS_<file name>` first) is read next to its POSCAR, and the parent bands, levels and soft-mode destinations are added |
| `--parent-readfc` | read the parent's `FORCE_CONSTANTS` instead (with `--parent-dim`) |
| `--trans-mat T` (`--trans-matrix`) | another of the equivalent settings: nine integers, row-wise, child = T @ parent (default: the setting of crystod-group) |
| `--band`, `--band-labels`, `--npoints` | a manual path in parent primitive coordinates, its labels, and the points per leg (default 51) |
| `--no-irreps` | skip the parent-irrep labels |
| `--weight-min W` | the weight cut (default 0.02): page 1 leaves out the markers of the modes below it, page 2, the labels and the tables the degenerate groups with P_g below it (label `-`) |
| `-o PREFIX` | the names of the PDF and `.dat` files (default `phonon_unfold`) |
| `--export-npz FILE` | write every array to an `.npz` file |
| `--tolerance` | the tolerance of the symmetry-mode analysis that finds the setting, in (0, 0.5] Å (default 0.01, as crystod-group) |

### Limitations

- No NAC: `--nac` is refused, because unfolding the LO/TO-split bands near Γ
  needs the direction of approach to Γ.
- The parent must be related to the child by a group-subgroup setting that
  `crystod-group --supergroup-cif` finds, with the same composition per
  parent cell and no displacement above 1.8 Å.
- A child file that is not a primitive cell is unfolded on its primitive
  cell, as a centered parent is (the setup block says so); an exact supercell
  of the parent is thereby reduced to the parent cell itself and has N = 1.
- phonopy symmetrizes the force constants it builds from a `FORCE_SETS` but
  takes a `FORCE_CONSTANTS` as it is, so the two files of one calculation
  give slightly different weights. For the CaTiO3 example the two sets of
  force constants differ by up to 2.1e-3 eV/Å² and the weights on the path
  by up to 1.6e-3; for AlF3 by up to 5.0e-3 eV/Å² and 4.1e-5. In the domain
  average a pair of modes at the edge of the degeneracy tolerance (1e-4 THz)
  may be one level with one file and two with the other, and then swap its
  two weights (one pair of CaTiO3 does); their sum agrees, and only the sum
  has a meaning (see [Weights](#weights)).
- The usual (type-1) `FORCE_SETS` stores the number of supercell atoms but
  not the shape of the supercell. A wrong number of atoms stops the run with
  an ERROR; a wrong `--child-dim` (or `--parent-dim`) with the right number,
  `4 2 1` for a 2x2x2 calculation, is not caught by that check. phonopy then
  either stops with its own ERROR ("Input forces are not enough to calculate
  force constants", as for both example parents and the AlF3 child) or, as
  for the CaTiO3 child, builds wrong force constants that show up as spurious
  imaginary frequencies.
- The labels of a distorted child are a decomposition of the unfolded vectors
  into parent irreps, not symmetries of the child: within a set the child
  symmetry cannot separate the mixing is real, and near 50 % the rule can move
  a level between one irrep and a set.
- The domain average assumes equal domain populations and averages at equal
  fractional k of the pseudo-parent cells; for a strongly strained child
  (AlF3, -9.66 % in volume) these are not exactly the same Cartesian wave
  vectors in every domain.
- The origin of the child modes in parent k and irreps is listed at the child
  Γ point only (not along a path of the child zone); phonons only, no
  electronic bands.
- Checked so far on centrosymmetric children (Pnma and R-3c) of the
  symmorphic parent Pm-3m, and on a Pnma cell unfolded onto itself; a polar or
  chiral child and a non-symmorphic parent with a distorted child have not
  been tested.

In Python the steps are functions of `crystod.phonon`: `load_unfold_inputs`,
`find_parent_mapping`, `make_parent_path`, `unfold`, `domain_average`,
`irrep_resolve` and `gamma_origin`
([Band unfolding onto the parent zone](python-api.md#band-unfolding-onto-the-parent-zone)).

## 52. Phonon mode viewer (`--visualize`)

*Example directory: `example/52_phonon_visualize` (testsuite section 52); after pip install: `crystod-example 52`*

`--visualize` writes one standalone HTML page that shows every phonon mode at
the special points of the k path, grouped by irrep: the dispersion along the
path (the bands, path and labeled points of the `--irreps` figure, with a dot
on every degenerate level of the special points), the mode table of each
point with the ISO-IR label of every degenerate set (the labels of
`phonon_irreps.yaml`, with the Mulliken symbol and the IR/Raman/silent/acoustic
activity at GM), and a 3D scene with the displacement pattern of the selected
mode as arrows, which an animation turns into the vibration. It is the phonon
counterpart of the SALC viewer of `crystod --visualize`, and the browser
counterpart of `--vector`: the partners of a degenerate level are the same
symmetry-adapted eigenvectors. Next to the page the run writes
`band_vector.yaml`, phonopy's `band.yaml` with the eigenvector of every band
along the same path. With `--all-irreps`, the pair of `--irreps
--all-irreps`, the page also shows the midpoint of every segment of the path
(the symmetry lines DT, Z, SM, ... with their line irreps;
[below](#the-segment-midpoints---all-irreps)).

```bash
cd example/52_phonon_visualize/SrTiO3_Pm-3m
crystod-phonon --visualize --dim 4 4 4 -c 221_PPOSCAR_SrTiO3 --bond Ti O 2.1
# -> phonon_visualize.html, band_vector.yaml   (-o FILE renames the page)
```

```

* Structure *
  Space group: Pm-3m (#221)
  Formula: SrTiO3
  Supercell: 4 x 4 x 4 (FORCE_SETS)

* Band path *
  k path (seekpath): GM-X-M-GM-R-X | R-M  (306 q points)
  Labeled points: X, GM, R, M

* Phonon modes at q = GM (0, 0, 0) *
Band  Freq (THz)  Irrep    Comp.  Mulliken  Activity
   1      0.0000  GM4-(3)      1  T1u       acoustic
   2      0.0000  GM4-(3)      2  T1u       acoustic
   3      0.0000  GM4-(3)      3  T1u       acoustic
   4      2.6629  GM4-(3)      1  T1u       IR
  ...
  10      7.3465  GM5-(3)      1  T2u       silent
  ...
  15     15.3054  GM4-(3)      3  T1u       IR
  Display cell: primitive, 1 x 1 x 1 cells

  ... (X and M)

* Phonon modes at q = R (1/2, 1/2, 1/2) *
Band  Freq (THz)  Irrep   Comp.
   1     -1.0867  R5-(3)      1
   2     -1.0867  R5-(3)      2
   3     -1.0867  R5-(3)      3
   4      3.9891  R4-(3)      1
  ...
  15     23.2302  R2-         1
  Display cell: primitive, 2 x 2 x 2 cells

* Output files *
  Phonon band structure with eigenvectors written to: band_vector.yaml
  Phonon mode viewer written to: phonon_visualize.html
```

The terminal lists, for every q point of the page, the table the page shows
(the band, its frequency, the label of its degenerate set and its component
inside the set; at GM also the Mulliken symbol and the activity) and the
display cell of its pattern. A run takes about 3 s for these 4x4x4 data; the
page is about 100 kB and `band_vector.yaml` 4.0 MB (306 q points, 15 bands).

Open the page in a web browser (on macOS `open phonon_visualize.html`). It
opens on the first q point and its first band; a hash added to the address
opens another one: `phonon_visualize.html#q=R&mode=1&view=a` selects the
first point named R (`G`, `GM` or `GAMMA` for Γ), band 1 and the view down
the a axis. That is the figure below: the imaginary R5- triplet of SrTiO3 (the
antiphase rotation of the TiO6 octahedra; R4+ with Ti at the origin), whose
first partner rotates the octahedra about a, in antiphase from one layer to
the next, so that the arrows of two layers meet at every apex in this view.

```{figure} images/phonon_visualize_SrTiO3_R.png
:width: 100%
:alt: The phonon mode viewer of SrTiO3: dispersion, mode table and the R5- octahedral rotation

`phonon_visualize.html#q=R&mode=1&view=a` of the SrTiO3 example (`--bond Ti O
2.1`): the dispersion with the imaginary R branch in red and the selected
level ringed, the R table with the imaginary rows in red, and the first R5-
partner down the a axis, 2 x 2 x 2 primitive cells.
```

### Reading the page

- **q point**: one button per labeled special point of the path, with its
  coordinates as fractions of the primitive reciprocal basis, in the order
  in which the path first reaches it (GM, X, M, R). A point the path visits
  twice (GM) sits at the vertex the `--irreps` figure labels. With
  `--all-irreps` the segment midpoints follow on a row of their own
  ([below](#the-segment-midpoints---all-irreps)).
- **Mode table**: `Band` (1-based, frequency order), `Freq (THz)`
  (negative: imaginary), `Irrep` (the label of the degenerate set as
  `phonon_irreps.yaml` writes it: `GM4-(3)`, `X3-` for a one-dimensional
  irrep, `GM1 + GM5(2)` for a set holding several irreps), `Comp.` (the
  component inside the set, 1 to its dimension) and, at GM, `Mulliken` (`T1u`)
  and `Activity` (`IR`, `Raman`, `IR+Raman`, `silent`, `acoustic`). The rows
  of a level below `--threshold` (default -0.1 THz) are red. Click a row to
  show the mode; the up and down arrow keys step through the table.
- **Dispersion**: click a dot to select its q point and the first component
  of its level; the selected level is ringed, imaginary branches and dots
  are red. Hovering a dot shows its label and frequency. With `--all-irreps`
  the segment midpoints have their dots too, in the middle of their legs.
- **Arrows**: the real part of the displacement pattern at phase 0, from the
  equilibrium position of every atom; a displacement below 2 % of the
  largest gets no arrow. The amplitude slider sets the largest displacement
  in Å (default 0.25 times the nearest-neighbor distance, rounded to 0.05 Å:
  0.5 Å for ScF3 and SrTiO3).
- **animate**: the atoms move along the pattern as the phase runs through a
  cycle (the speed slider sets the cycles per second); arrows, bonds and
  polyhedra move with them. Unchecking returns to the static pattern.
- **Display cell**: the smallest supercell over which the Bloch phase of q
  is periodic, as `--vector` builds it: 1 x 1 x 1 primitive cells at GM,
  1 x 2 x 1 at X, 2 x 2 x 1 at M, 2 x 2 x 2 at R (`--conventional`: the
  conventional cell times the multiples q needs; a segment midpoint:
  1 x 4 x 1 at DT (0, 1/4, 0)). Atoms on the cell faces are drawn on both
  faces, as VESTA does, each with the phase of its own lattice translation.
  The line under the display controls names the cell, and at a segment
  midpoint its segment (`Segment: GM-X (midpoint)`).
- **Bonds and polyhedra** (`--bond EL1 EL2 MAX`, repeatable): the bonds up to
  MAX Å and the coordination polyhedra around EL1, as in `crystod --visualize`;
  the periodic images that complete the polyhedra are drawn too, and move
  with the pattern.
- **View along a / b / c**: an orthographic view down that axis of the
  display cell (`reset` returns to the perspective view); drag to rotate,
  scroll to zoom; the compass in the corner follows the view. The
  `show ...` checkboxes hide the arrows, the cell edges and axes, the atoms,
  the bonds and the polyhedra.

The second example, `example/52_phonon_visualize/ScF3_Pm-3m`, opens on the
acoustic GM4- [T1u] triplet (`crystod-phonon --visualize --dim 4 4 4 -c
221_PPOSCAR_ScF3`); its lowest R level is the octahedral rotation R4+ at
+1.42 THz, real in this harmonic calculation.

```{figure} images/phonon_visualize_ScF3.png
:width: 100%
:alt: The phonon mode viewer of ScF3 in its default view

The ScF3 page as it opens: GM, band 1 (an acoustic GM4- [T1u] mode, every
atom displaced alike), 1 x 1 x 1 primitive cell, perspective view.
```

### The segment midpoints (`--all-irreps`)

`--all-irreps` is the pair of
[`--irreps --all-irreps`](#symmetry-lines-as-well---all-irreps): besides the
labeled special points, the page shows the midpoint of every segment of the
path, a point on each symmetry line, as a q point of its own: its mode table
with the ISO-IR line irreps, its display cell, the arrows, the animation, the
bonds and the polyhedra. The page is then `phonon_visualize_all.html`, so that
it and the default page can coexist (`-o` still renames it;
`band_vector.yaml` is the same file).

```bash
cd example/52_phonon_visualize/SrTiO3_Pm-3m
crystod-phonon --visualize --dim 4 4 4 -c 221_PPOSCAR_SrTiO3 --bond Ti O 2.1 --all-irreps
# -> phonon_visualize_all.html, band_vector.yaml
```

```
* Band path *
  k path (seekpath): GM-X-M-GM-R-X | R-M  (306 q points)
  Labeled points: X, GM, R, M
  Segment midpoints: DT (GM-X), Z (X-M), SM (M-GM), LD (GM-R), S (R-X), T (R-M)

  ... (GM, X, M and R as above)

* Phonon modes at q = DT (0, 1/4, 0) *
Band  Freq (THz)  Irrep   Comp.
   1      2.1278  DT5(2)      1
   2      2.1278  DT5(2)      2
   3      3.7543  DT1         1
  ...
  15     19.8724  DT1         1
  Segment: GM-X
  Display cell: primitive, 1 x 4 x 1 cells

  ... (Z, SM, LD, S and T)
```

The midpoints follow the special points in path order, as in
`phonon_irreps_all.yaml`: for the seekpath path the same points, names and
segments; for a `--band` path the midpoints of its own legs, named by their
ISO-IR k-vector type. Each is labeled at the midpoint itself, so its labels
are those of `phonon_irreps_all.yaml` band by band (testsuite section 52
compares them). Its display cell is commensurate with the exact fractions of
the midpoint: DT (0, 1/4, 0) gets 1 x 4 x 1 primitive cells (20 atoms),
LD (1/4, 1/4, 1/4) 4 x 4 x 4 (320 atoms). A midpoint whose display cell would
hold more than 1000 atoms is not drawn; a NOTE in `* Band path *` and on the
page names it and the `--qpoint` that draws it. On the fcc path of Si these
are the two SM midpoints, of X-U and of K-GM:

```
  NOTE: the midpoint SM (3/16, 3/16, 3/8) of K-GM is not drawn: its display cell would hold 4096 atoms (give --qpoint 3/16 3/16 3/8 to draw it)
```

while DT (1/4, 0, 1/4) of GM-X, LD of GM-L, Q of L-W and V of W-X are drawn
(the largest on 2 x 8 x 8 primitive cells). A midpoint that coincides with a
special point or with an earlier midpoint (modulo a reciprocal lattice
vector) is left out. For the SrTiO3 example the run takes about 4.5 s and the
page is about 195 kB.

On the page the midpoints have a row of buttons of their own under
"Segment midpoints" (`DT (0, 1/4, 0)`; when two midpoints share a name, two
arms of one line type, the button adds the segment: `SM (X-U)` and
`SM (K-GM)` for Si with `--conventional`) and a dot on every level in the
middle of their leg of the dispersion, which selects them as the dots of the
special points do. The hash opens a midpoint by its name (`#q=DT`, the first
point so named), by its name and segment (`#q=DT(GM-X)`) or by the position
of its button (`#q=5`, the fifth q button).

### `band_vector.yaml`

`band_vector.yaml` is the `band.yaml` that phonopy writes for its band
structure with eigenvectors (`phonopy --eigvecs`), along the path of the page
(the `--irreps` path: seekpath, or `--band`, 51 q points per leg or
`--npoints`), its vertices named by their ISO-IR labels:

```
nqpoint: 306
npath: 6
segment_nqpoint:          # q points per leg
- 51
...
labels:                   # the two ends of every leg
- [ 'GM', 'X' ]
- [ 'X', 'M' ]
...
reciprocal_lattice:       # rows a*, b*, c* (1/Angstrom, without 2 pi)
natom: 5
lattice: ...              # the primitive cell (Angstrom)
points: ...               # symbol, fractional coordinates and mass of every atom
phonon:
- q-position: [ 0.0000000, 0.0000000, 0.0000000 ]
  distance: 0.0000000
  band:
  - # 1
    frequency: 0.0000003685          # THz, negative when imaginary
    eigenvector:
    - # atom 1
      - [ 0.68807195292903, 0.00000000000000 ]   # [re, im] of x, y, z
      - [ -0.00414863873523, 0.00000000000000 ]
      - [ -0.06380814382605, 0.00000000000000 ]
    - # atom 2
      ...
```

Each band carries its eigenvector: one list per atom of the primitive cell
with the `[re, im]` pairs of its x, y and z components, the mass-weighted
eigenvector of phonopy's dynamical matrix in phonopy's atom-position phase
convention (unit norm over the whole vector). These are phonopy's plain
eigenvectors at every q of the path, not the symmetry-adapted partners the
page draws (within a degenerate level they can be any orthonormal
combination). phonopy's own tools (`phonopy-bandplot`) and any YAML reader
(`yaml.safe_load`) read the file. Its name is fixed; `-o` renames the page
only.

### How the pattern is built

phonopy's displacement of atom j in the cell at lattice vector R_l, for the
mode with eigenvector e at q, is

u_j(l) = e_j exp(2 pi i q.(R_l + tau_j)) / sqrt(m_j)

(tau_j the position of atom j in the primitive cell, m_j its mass), with one
overall phase chosen so that the real part is as large as possible (phonopy's
MODULATION choice). The vectors e are those of `--vector`: the dynamical
matrix is block-diagonalized in the irrep-projected basis
(`build_symmetry_adapted_modes`), so that the partners of a degenerate level
transform with the irrep matrices and point along directions the symmetry
fixes, instead of the arbitrary combinations an eigensolver returns. The
page receives, per mode, only the atomic-gauge vector of every primitive
atom, u_a = e_a exp(2 pi i q.tau_a) / sqrt(m_a) times that phase, normalized
so that the largest displacement of the display cell is 1, and draws the atom
a at lattice translation T (primitive basis) displaced by

Re[u_a exp(i (2 pi q.T - phi))] x amplitude

with phi the phase of the animation (0 for the arrows). Testsuite section 52
checks that this reproduces phonopy's modulation on the display cell, atom by
atom, to 1e-6. At a time-reversal-invariant q (2q a reciprocal lattice
vector: GM, X, M, R of a cubic cell) the patterns of the symmetry-adapted
partners are real: every atom moves back and forth along its arrow. At any
other q (K of a hexagonal crystal, a point on a line) a pattern is complex in
general: the atoms run on ellipses, and the arrows show the phase-0
snapshot. With `--nac`, the patterns and labels at GM are those of the TO
modes (phonopy applies no direction of approach at GM itself; the NOTE of
`* Band path *` says so), while the bands near GM and `band_vector.yaml`
carry the LO-TO splitting.

### Options

| option | meaning |
|---|---|
| `--qpoint R` or `--qpoint 0.5 0.5 0` | show that q point only (a label, or three primitive reciprocal coordinates, fractions allowed); default every labeled special point of the path. A point that is not a vertex of the path gets no dot on the dispersion |
| `--all-irreps` | also the midpoint of every segment of the path, as `--irreps --all-irreps` labels them (the symmetry lines DT, Z, SM, ...); the page is named `phonon_visualize_all.html`; not with `--qpoint` |
| `--band`, `--band-labels`, `--npoints` | the path, as in `--irreps` (default seekpath, 51 q points per leg) |
| `--threshold F` | frequency in THz below which a level is drawn in red (default -0.1) |
| `--conventional` | display the conventional cell instead of the primitive cell |
| `--bond EL1 EL2 MAX` | bonds and coordination polyhedra (repeatable), as in `crystod --visualize` |
| `--nac` | the non-analytical term correction with `BORN` (TO patterns at GM) |
| `--readfc` | read `FORCE_CONSTANTS` instead of `FORCE_SETS` |
| `--tolerance` | the degeneracy tolerance of the irrep labels in THz (default 1e-3) |
| `-o FILE` | the name of the page (default `phonon_visualize.html`, with `--all-irreps` `phonon_visualize_all.html`); `band_vector.yaml` keeps its name |

### Limitations

- The display cell of a general q is large: each axis is multiplied by the
  denominator of q (fractions with denominators up to 48), so `--qpoint 0.3
  0.1 0.2` draws 10 x 10 x 5 primitive cells, 2651 atoms for ScF3. Above
  1000 atoms the run prints a NOTE in the block of that q point; the page
  still works, more slowly. The cell is a diagonal multiple of the primitive
  (or conventional) cell, not the smallest supercell commensurate with q.
- With `--all-irreps`, a segment midpoint whose display cell would hold more
  than 1000 atoms is skipped with a NOTE (the 16ths of the fcc path: 4096
  atoms for Si); `--qpoint` with its coordinates draws it.
- The bond search of `--bond` compares every EL1 atom with every EL2 atom in
  the 27 periodic images of the display cell (vectorized with NumPy, but its
  cost grows with the square of the number of atoms): about 1 s for a few
  hundred atoms.
- The page loads plotly 2.32.0 from its CDN, as the SALC viewer does, so the
  first opening needs network access (the browser keeps it afterwards).
- Arrows are in Å, but the atoms are drawn as markers of a fixed size in
  pixels: a short displacement can hide inside its atom. Zoom in, raise the
  amplitude, uncheck `show atoms`, or look down an axis.
- During the animation the polyhedra keep the faces of their equilibrium
  shape, which is right for the small amplitudes a vibration has.
- The modes are shown at the labeled special points of the path, with
  `--all-irreps` also at the midpoint of every segment, or at one `--qpoint`
  (any q, by its coordinates); not at every q of the dispersion.

In Python the page is built by `crystod.phonon.phonon_viewer_data` and
written by `write_phonon_viewer_html`; `write_band_vector_yaml` writes the
YAML file ([Phonon mode viewer](python-api.md#phonon-mode-viewer)).
