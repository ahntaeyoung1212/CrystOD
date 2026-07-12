# crystod-phonon

Phonon analyses on top of [phonopy](https://phonopy.github.io/phonopy/) data
(POSCAR + `FORCE_SETS`, or `FORCE_CONSTANTS` with `--readfc`, or
`phonopy_params.yaml`). Six mode flags: `--irreps`, `--fatband`, `--lt`,
`--vector`, `--modulation`, `--vibration`.

## 17. Phonon irreducible representations (`--irreps`)

*Example directory: `example/17_phonon_irrep` (testsuite section 17)*

Label the phonon modes at the special q points with their irreducible
representations and write `phonon_irreps.yaml`:

```bash
crystod-phonon --irreps -c example/test_POSCARs/221_PPOSCAR_SrTiO3 --dim="4 4 4" --readfc
```

## 18. Element-projected phonon fatbands (`--fatband`)

*Example directory: `example/18_phonon_fatband` (testsuite section 18)*

Plot phonon fatbands colored by the element-projected phonon density (sum of
squared eigenvector components over each element's atoms), directly from
POSCAR + `FORCE_SETS` (or `FORCE_CONSTANTS` with `--readfc`):

```bash
cd example/18_phonon_fatband/ScF3_Pm-3m
crystod-phonon --fatband -c 221_PPOSCAR_ScF3 --dim 4 4 4
```

The space group is detected, the high-symmetry k-path is generated automatically
with seekpath, the phonon band structure is computed with eigenvectors and band
connection via the phonopy API (no `band.yaml` needed), and one
`fatband_<element>.pdf` is written per element, colored with the VESTA default
element colors and with dot sizes proportional to the projected weight. For ScF3
this cleanly separates the F-dominated soft rotational branches (R/M points)
from the Sc-dominated mid-frequency bands.

Options: `--element F` restricts the output to one element; `--nac` applies the
non-analytical term correction (LO/TO splitting) using a `BORN` file in the
current directory; `--band`/`--band-labels` supply a manual k-path instead of
seekpath; `--npoints` sets the q-point density per path leg (default 51);
`--projection-direction "0 0 1"` projects the displacements onto a direction in
reduced coordinates before squaring.
Plotting style based on `script/phonon_fatband.py` by Hiroki Koiso.

```bash
crystod-phonon --fatband -c 221_PPOSCAR_ScF3 --dim 4 4 4 --nac
```

NAC is applied only when `--nac` is given. With `--nac` the output files are
named `fatband_nac_<element>.pdf`, so corrected and uncorrected fatbands can
coexist side by side.

## 19. Longitudinal/transverse-resolved phonon bands (`--lt`)

*Example directory: `example/19_phonon_lt` (testsuite section 19)*

Plot the phonon band structure colored by the longitudinal/transverse character
of each mode (red = longitudinal, blue = transverse, white = mixed or Gamma):

```bash
cd example/19_phonon_lt/ScF3_Pm-3m
crystod-phonon --lt -c 221_PPOSCAR_ScF3 --dim 4 4 4
crystod-phonon --lt -c 221_PPOSCAR_ScF3 --dim 4 4 4 --nac
```

The longitudinal character of a mode is
`sqrt(sum_atoms |q_hat . e_atom|^2)`, i.e. the norm of the eigenvector
projection onto the propagation direction (valid along any path direction,
including diagonal segments such as GM-R). Output: `phonon_band_LT.pdf`, or
`phonon_band_LT_nac.pdf` with `--nac` — with NAC the split-off LO branches show
up as purely red. Based on `script/LT_phonon_band.py` maintained by Hiroki
Koiso, after Qijing Zheng.

## 20. Phonon eigenvector visualization (`--vector`)

*Example directory: `example/20_phonon_vector` (testsuite section 20)*

Diagonalize the dynamical matrix directly at the selected q point via the
phonopy API, list the modes with their frequencies and irrep labels, and export
the selected eigenvectors as `.vesta` files with per-atom displacement arrows:

```bash
cd example/20_phonon_vector/Si_Fd-3m

# List the modes at GM and export ALL of them as individual VESTA files;
# the mode table is also saved as phonon_modes_Si_GM.txt
crystod-phonon --vector --dim "4 4 4" -c 227_PPOSCAR_Si --qpoint GM

# Export one optical GM5+ mode only (mode numbers are 1-based)
crystod-phonon --vector --dim "4 4 4" -c 227_PPOSCAR_Si --qpoint GM --mode 4

# Multiple selected modes are summed into one displacement pattern
crystod-phonon --vector --dim "4 4 4" -c 227_PPOSCAR_Si --qpoint GM --mode 4 5 6

# X point: the commensurate supercell (2x1x2) and Bloch phases are applied automatically
crystod-phonon --vector --dim "4 4 4" -c 227_PPOSCAR_Si --qpoint X --mode 1 --readfc

# Conventional-cell output (POSCAR_Si_GM_mode4+5+6_GM5+_conv.vesta etc.)
crystod-phonon --vector --dim "4 4 4" -c 227_PPOSCAR_Si --qpoint GM --mode 4 5 6 --conventional
```

Output files are auto-named `POSCAR_<formula>_<qlabel>_mode<N>_<irrep>.vesta`
(mode numbers zero-padded so `ls` lists them in order) and open directly in
[VESTA](https://jp-minerals.org/vesta/), showing the equilibrium structure with
red arrows for the real part of the mass-weighted eigenvector displacement.

Degenerate modes are exported as **symmetry-adapted** eigenvectors: the
dynamical matrix is block-diagonalized in the spgrep irrep-projected basis (the
same construction as `--modulation`), so that, e.g., the three degenerate GM5+
optical modes of Si point exactly along the cubic axes instead of the arbitrary
tilted combinations a plain eigensolver returns. The exported vectors are exact
eigenvectors of the phonopy dynamical matrix (verified internally against the
phonopy spectrum).

`--conventional` outputs the structure and arrows in the conventional cell
(`_conv` file-name suffix); for fractional q points the conventional cell is
multiplied until the Bloch phase is commensurate. `--qpoint` accepts a
high-symmetry label (`GM`, `X`, `L`, ...) or three primitive reciprocal
coordinates (fractions allowed). When `--mode` is omitted, ALL modes are
exported as individual VESTA files. Arrows are rescaled so that the largest
total displacement equals `--amplitude` (default 1.5 Angstrom).

## 21. Symmetry-adapted phonon modulation (`--modulation`)

*Example directory: `example/21_modulation` (testsuite section 21)*

Generate modulated (displaced) structures from symmetry-adapted phonon modes:

```bash
crystod-phonon --modulation --yaml example/21_modulation/ScF3_Pm-3m/phonopy_params.yaml \
  --qpoint 0.5 0.5 0.5 --mode 1 2 3 --amplitude 0.3
```

When `--mode` is omitted, only the mode table (mode number, frequency, irrep,
degeneracy) and the star of q are printed, so you can inspect the modes at a q
point first and then choose which mode(s) to apply:

```bash
crystod-phonon --modulation --yaml example/21_modulation/ScF3_Pm-3m/phonopy_params.yaml --qpoint 0.5 0.5 0.5
```

If `phonopy_params.yaml` exists in the current directory, `--yaml` can be
omitted. If a single amplitude is given, it is applied to all selected modes.
When `--output` is omitted, the file is auto-named
`MPOSCAR_{q}_{mode}_{irrep}_{subgroup}` — e.g. `MPOSCAR_R_mode1+2+3_R4+_R-3c`.
The space group of the modulated structure is detected and printed
(e.g. R4+(a,a,a) of Pm-3m ScF3 gives R-3c).

Different q points can be combined with numbered argument sets
(`--qpoint1/--mode1/--amplitude1`, `--qpoint2/--mode2/--amplitude2`, ...):

```bash
crystod-phonon --modulation \
  --yaml example/21_modulation/ScF3_Pm-3m/phonopy_params.yaml \
  --qpoint1 0 0.5 0.5 --mode1 1 --amplitude1 0.3 \
  --qpoint2 0.5 0 0.5 --mode2 1 --amplitude2 0.3 \
  --qpoint3 0.5 0.5 0 --mode3 1 --amplitude3 0.3 \
  --output POSCAR_multi_q_arms
```

The star of q is displayed for each selected q point, which is useful when
combining arms of the same star in multi-q modulations.

## 22. Symmetry-only vibration bases (`--vibration`)

*Example directory: `example/22_vibration` (testsuite section 22)*

List the available high-symmetry q points and irrep-grouped vibration spaces
from crystal symmetry alone (no force data needed):

```bash
crystod-phonon --vibration -c example/test_POSCARs/221_PPOSCAR_ScF3 --qpoint R
crystod-phonon --vibration -c example/test_POSCARs/221_PPOSCAR_ScF3 --qpoint 0.5 0.5 0
```

Select one symmetry-allowed mode component, build the commensurate supercell,
and export a displaced structure:

```bash
crystod-phonon --vibration \
  -c example/test_POSCARs/221_PPOSCAR_ScF3 \
  --qpoint R \
  --mode-index 3 \
  --component-index 1 \
  --output POSCAR_vibration
```

`--qpoint` accepts either three primitive reciprocal coordinates or a
high-symmetry label. Add `--export-npz mode_data.npz` to save positions,
displacement vectors, symbols, and lattice for notebook-side visualization
(`python3 vibration_viewer.py --npz mode_data.npz` converts or inspects it).
