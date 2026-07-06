# CrystOD

`crystod` is a Python package and command-line toolset for symmetry analysis of crystal orbitals, SALCs, basis functions, phonon irreducible representations, symmetry-adapted phonon vibration bases, symmetry-adapted phonon modulations, and point-group direct products.

## Features

- `crystod --salc` for crystal-orbital / atomic-band irreducible representations from atomic orbitals.
- `crystod --salc --atomic-orbital ...` for orbital hybridization analysis at a selected atomic orbitals (e.g. Sc_d F_p).
- `crystod --basis-function` for classifying polynomial basis functions into point-group irreps.
- `crystod --phonon-irrep` for phonon irrep labeling from `phonopy` data.
- `crystod --vibration` for symmetry-allowed vibration bases from crystal symmetry alone.
- `crystod --modulation` for generating modulated structures from symmetry-adapted phonon modes.
- `crystod --direct-product` for point-group irrep direct products and character-table display.

## Requirements

- Python 3.9 or later
- Main Python dependencies:
  `phonopy`, `spglib`, `spgrep`, `irrep`, `irreptables`, `ase`, `numpy`, `pandas`, `openpyxl`, `sympy`

## Installation

Recommended:

```bash
conda create -n crystod python=3.11
conda activate crystod

unzip ~/Downloads/CrystOD-main.zip
mv ~/Downloads/CrystOD-main ~
cd ~/CrystOD-main
pip install phonopy spglib spgrep irrep irreptables numpy openpyxl pandas

python3 -m pip install -e ~/CrystOD-main --no-deps --no-build-isolation
```

If your default Python environment uses an old `setuptools`, avoid `--no-build-isolation` for editable install.

Example:

```bash
python3 -m pip install -e . --no-deps
```

If needed, upgrade the build backend first:

```bash
python3 -m pip install -U "setuptools>=68" wheel
```

## Quick Start

Show global help:

```bash
crystod --help
```

## 1. IrReps (Irreducible Representations) of SALC (Symmetry-Adapted Linear Combination) for a Selected Element and Orbital

Example:

```bash
crystod --salc --poscar example/test_POSCARs/221_PPOSCAR_SrTiO3 --element Ti --orbital d
```

Spinor example with irrep table:

```bash
crystod --salc --poscar example/test_POSCARs/221_PPOSCAR_SrTiO3 --element Ti --orbital d \
  --spinor --kpoint 0 0 0 --show-irrep-table
```

## 2. Hybridization Analysis

Example:

```bash
crystod --salc --poscar example/test_POSCARs/221_PPOSCAR_SrTiO3 --atomic-orbital Ti_d O_p --kpoint 0 0 0
```

## 3. Phonon Irreducible Representations

Example:

```bash
crystod --phonon-irrep --poscar example/test_POSCARs/221_PPOSCAR_SrTiO3 --dim="4 4 4" --readfc
```

## 4. Basis Functions

Examples:

```bash
crystod --basis-function x y z --point-group m-3m
crystod --basis-function x y z --space-group Pm-3m --kpoint 0 0 0
crystod --basis-function xyz --space-group Pm-3m --kpoint 0.5 0.3 0 --show-irrep-table
crystod --basis-function z^2 --point-group m-3m
crystod --basis-function "x(y^2-z^2)" --point-group="m-3m"
crystod --basis-function "x^2-y^2" "2z^2-x^2-y^2" xy yz zx --space-group="Pm-3m" --kpoint 0 0 0
crystod --basis-function xyz "z^3" "y^3" "x^3" "y(z^2-x^2)" "z(x^2-y^2)" "x(z^2-y^2)" --space-group="Pm-3m" --kpoint 0.5 0.5 0
crystod --basis-function xyz "z^3" "yz^2" "xz^2" "z(x^2-y^2)" "y(3x^2-y^2)" "x(x^2-3y^2)" --space-group="P6/mmm" --kpoint 0 0 0
```

The input functions are automatically closed under the selected point group or the little group of the selected space-group k-point, then decomposed into irreps. When the k-point is listed in `irreptables`, physical labels such as `GM4-(3)` or `R4-(3)` are shown; otherwise, `spgrep` generic labels such as `irrep_2(1)` are used.

## 5. Direct Products of Point-Group Irreps

Example:

```bash
crystod --direct-product --point-group m-3m --irreps T2g T2g T1u
```

## 6. Symmetry-Only Vibration Bases

List the available high-symmetry q-points and irrep-grouped vibration spaces:

```bash
crystod --vibration --poscar example/test_POSCARs/221_PPOSCAR_ScF3 --qpoint R
crystod --vibration --poscar example/test_POSCARs/221_PPOSCAR_ScF3 --qpoint 0.5 0.5 0
```

Select one symmetry-allowed mode component, build the commensurate supercell, and export a displaced structure:

```bash
crystod --vibration \
  --poscar example/test_POSCARs/221_PPOSCAR_ScF3 \
  --qpoint R \
  --mode-index 2 \
  --component-index 0 \
  --output POSCAR_vibration
```

`--qpoint` accepts either three primitive reciprocal coordinates or a high-symmetry label such as `GM`, `X`, `M`, or `R`.
Add `--export-npz mode_data.npz` to save positions, displacement vectors, symbols, and lattice for notebook-side visualization.
The exported `.npz` file is a NumPy archive, not a directly viewable graphics file. You can inspect or convert it with:

```bash
python3 vibration_viewer.py --npz mode_data.npz
python3 vibration_viewer.py --npz mode_data.npz --write-poscar POSCAR_view
python3 vibration_viewer.py --npz mode_data.npz --write-trajectory vibration.xyz
```

## 7. Symmetry-Adapted Phonon Modulation

Example:

```bash
crystod --modulation --yaml example/modulation/ScF3_Pm-3m/phonopy_params.yaml --qpoint 0.5 0.5 0.5 --mode 0 1 2 --amplitude 0.3
crystod --modulation --yaml example/modulation/ScF3_Pm-3m/phonopy_params.yaml --qpoint1 0.5 0.5 0.5 --mode1 0 1 2 --amplitude1 0.3 \
  --qpoint2 0.5 0.5 0 --mode2 0 --amplitude2 0.3 
```

If `phonopy_params.yaml` exists in the current directory, `--yaml` can be omitted.
If a single amplitude is given, it is applied to all selected modes.
Different q-points can also be combined by using numbered argument sets such as
`--qpoint1/--mode1/--amplitude1`, `--qpoint2/--mode2/--amplitude2`, and so on.
Example:

```bash
crystod --modulation \
  --yaml example/modulation/ScF3_Pm-3m/phonopy_params.yaml \
  --qpoint1 0 0.5 0.5 --mode1 0 --amplitude1 0.3 \
  --qpoint2 0.5 0 0.5 --mode2 0 --amplitude2 0.3 \
  --qpoint3 0.5 0.5 0 --mode3 0 --amplitude3 0.3 \
  --output POSCAR_multi_q_arms
```

Show the point-group character table:

```bash
crystod --direct-product --point-group 3m --show-irrep-table
```

Show both the table and the direct-product decomposition:

```bash
crystod --direct-product --point-group m-3m --irreps T2g T2g T1u --show-irrep-table
```

## Command Summary

- `crystod --salc --poscar POSCAR --element ELEMENT --orbital ORBITAL`
- `crystod --salc --poscar POSCAR --atomic-orbital Ni_d O_p --kpoint kx ky kz`
- `crystod --basis-function BASIS1 BASIS2 ... --point-group PG`
- `crystod --basis-function BASIS1 BASIS2 ... --space-group SG --kpoint kx ky kz`
- `crystod --phonon-irrep --dim "nx ny nz" --poscar POSCAR [--readfc]`
- `crystod --vibration --poscar POSCAR --qpoint QLABEL_OR_QX QY QZ [--mode-index N]`
- `crystod --modulation --qpoint qx qy qz --mode MODE1 MODE2 ... [--yaml phonopy_params.yaml]`
- `crystod --modulation --qpoint1 qx1 qy1 qz1 --mode1 ... --qpoint2 qx2 qy2 qz2 --mode2 ... [--amplitude1 ... --amplitude2 ...]`
- `crystod --direct-product --point-group PG --irreps IRREP1 IRREP2 ...`
- `crystod --direct-product --point-group PG --show-irrep-table`

## Notes

- `--show-irrep-table` in SALC mode prints the little-group irrep table at the selected `k` point.
- `--show-irrep-table` in direct-product mode prints the point-group character table.
- Some workflows depend on the versions of `phonopy`, `spglib`, `spgrep`, and `irreptables`. `crystod` includes compatibility helpers for newer environments, but keeping these packages reasonably up to date is recommended.

 ## Contributors
- **Yasuhide Mochizuki** — [ahntaeyoung1212@gmail.com](mailto:ahntaeyoung1212@gmail.com), Tokyo University of Science
- **Hiroki Koiso** — [akabane0308koiso@gmail.com](mailto:akabane0308koiso@gmail.com), Institute of Science Tokyo

## License

GPLv3
