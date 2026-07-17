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

`--start-step`, `--xdatcar`, and `--format` behave exactly as in `--adp` mode;
`--end-step` optionally truncates the range (default: last step). Note that
XDATCAR stores no energies or temperatures, so Etot/T averages are outside the
scope of `--summary`. Based on `script/md_summary.py`.
