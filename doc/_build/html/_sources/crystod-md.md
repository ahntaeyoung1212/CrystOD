# crystod-md

MD-trajectory analyses. Two mode flags: `--adp` (atomic displacement parameters
as CIF) and `--summary` (time-averaged lattice statistics).

## 30. ADPs from an MD trajectory (`--adp`)

*Example directory: `example/30_xdatcar2adp` (testsuite section 30)*

Compute the time-averaged structure and anisotropic displacement parameters
(ADPs, U_ij) from a molecular-dynamics `XDATCAR` trajectory and write them as a
CIF file:

```bash
cd example/30_xdatcar2adp/ScF3_Pm-3m_NpT_300K
crystod-md --adp --dim 4 4 4 --start-step 1000 --xdatcar XDATCAR --output ADP.cif
```

```
Supercell info:
  atoms          : 256
  composition    : {'Sc': 64, 'F': 192}
  analyzed steps : 3001

Space group: Pm-3m (No. 221)
Atoms in unit cell   : 4
Symmetry operations  : 48
Asymmetric-unit sites: 2
  Sc0: mult=1, coords=(1.00000, 0.99999, 1.00000)
  F1: mult=3, coords=(0.00004, 0.50001, 0.00006)

ADP constraints per Wyckoff position:
  Wyckoff 0 (Sc): site-symmetry order=48, U11=U22, U11=U33, U22=U33, U12=0, U13=0, U23=0
  Wyckoff 1 (F): site-symmetry order=16, U11=U33, U12=0, U13=0, U23=0

Coordinate unwrapping done: 256 atoms x 3001 steps

Site       Ueq (A^2)    Constraint
------------------------------------------------------------
Sc0        0.005484     U11=U22, U11=U33, U22=U33, U12=0, U13=0, U23=0
F1         0.023520     U11=U33, U12=0, U13=0, U23=0

Saved: ADP.cif
```

The written CIF ends with the symmetry-constrained `_atom_site_aniso_U_*`
loop, ready for thermal-ellipsoid display in VESTA — for NpT ScF3 at 300 K the
F ellipsoids are strongly anisotropic (U11 = U33 >> U22, the pancake shape
perpendicular to the Sc-F-Sc bond that drives the negative thermal expansion),
while Sc stays isotropic by symmetry:

```
loop_
 _atom_site_aniso_label
 _atom_site_aniso_U_11
 _atom_site_aniso_U_22
 _atom_site_aniso_U_33
 _atom_site_aniso_U_23
 _atom_site_aniso_U_13
 _atom_site_aniso_U_12
 Sc0     0.00548   0.00548   0.00548   0.00000   0.00000   0.00000
 F1      0.03265   0.00527   0.03265   0.00000   0.00000   0.00000
```

`--dim` is the MD supercell size relative to the unit cell, given as three
diagonal values or a nine-value diagonal matrix, quoted or unquoted
(`--dim 4 4 4`, `--dim="4 4 4"`, `--dim 4 0 0 0 4 0 0 0 4`; non-diagonal
matrices are rejected). `--start-step` discards the first N MD steps as
equilibration (default 0). `--xdatcar` defaults to `XDATCAR` and `--output` to
`ADP.cif`. `--format` selects the trajectory format (currently `vasp` only;
LAMMPS support is planned).

The workflow: the supercell trajectory is folded and grouped into unit-cell
sites, the space group of the time-averaged unit cell is detected with spglib
(`--tolerance`, default 0.1), the per-atom displacement covariances are rotated
into the asymmetric-unit representatives and averaged over all
symmetry-equivalent atoms and MD steps, and site-symmetry constraints
(e.g. `U11=U22=U33, U12=U13=U23=0` on a cubic site) are enforced by projection.
The CIF contains the symmetry operations, the asymmetric-unit sites, and the
`_atom_site_aniso_U_*` loop, ready for thermal-ellipsoid display in VESTA. Both
fixed-cell and variable-cell (NpT, repeated-header) XDATCAR files are supported.

Based on `script/xdatcar_to_adp.py` by Ko Sato; the CrystOD port reproduces its
output exactly while replacing the pymatgen reader with a fast built-in parser.

## Trajectory summary (`--summary`)

*Example directory: `example/30_xdatcar2adp` (testsuite sections 30-31: `--summary` runs in the CLI-regression section)*

`crystod-md --summary` reports summary statistics of the same trajectory: the
time-averaged lattice parameters (a, b, c, alpha, beta, gamma) and cell volume
with standard deviations over the selected step range — useful for extracting
the equilibrium cell from an NpT run:

```bash
crystod-md --summary --start-step 1000 --xdatcar XDATCAR
crystod-md --summary --start-step 1000 --end-step 15000   # inclusive step range
```

```
Trajectory info:
  atoms          : 256
  composition    : {'Sc': 64, 'F': 192}
  total steps    : 4001
  analyzed steps : 3001 (step 1000 .. 4000)

Time-averaged cell (mean +/- std):
  a (A)       :      16.149940 +/- 0.077310
  b (A)       :      16.155625 +/- 0.067497
  c (A)       :      16.154106 +/- 0.068162
  alpha (deg) :      89.954896 +/- 0.214945
  beta (deg)  :      89.990825 +/- 0.262055
  gamma (deg) :      89.997261 +/- 0.217921
  V (A^3)     :    4214.828940 +/- 45.946154
  V/atom (A^3):      16.464176

NOTE: XDATCAR stores no energies or temperatures; Etot/T averages require OSZICAR/vasprun.xml.
```

`--start-step`, `--xdatcar`, and `--format` behave exactly as in `--adp` mode;
`--end-step` optionally truncates the range (default: last step). Note that
XDATCAR stores no energies or temperatures, so Etot/T averages are outside the
scope of `--summary`. Based on `script/md_summary.py`.
