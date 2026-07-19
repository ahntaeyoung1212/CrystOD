# Quick start

Show global help (the epilog lists all sectioned commands):

```bash
crystod --help
```

A first analysis — the crystal-orbital irreps of the Ti *d* manifold of SrTiO3
at every special k point:

```bash
crystod -c example/test_POSCARs/221_PPOSCAR_SrTiO3 --element Ti --orbital d
```

## Shared section numbering

Every feature of CrystOD carries one section number, used consistently in three places:

- the numbered sections of this documentation,
- `python testsuite.py <N>` (the regression tests of that feature),
- `example/<N>_*` (a worked example directory with real captured output).

The numbers are grouped by command:

| Sections | Command | Features |
|---|---|---|
| 1 | (library core) | Wigner D matrices — theoretical background, section 1 of {doc}`crystod` |
| 2–6 | `crystod` | 2–3 SALC & hybridization, 4 star of k, 5 SALC viewer, 6 CLI regression |
| 7–17 | `crystod-group` | 7 product, 8 decompose, 9 ligand field, 10 basis, 11 generate-basis, 12 coset, 13 isotropy subgroups (`--supergroup`), 14 multi-electron terms (`--multiplet`), 15 POSCAR <-> Bilbao-style CIF (`--poscar2cif` / `--cif2poscar`), 16 symmetry-mode analysis (`--supergroup-cif`), 17 CLI regression |
| 18–20 | `crystod-bz` | 18 Brillouin zone, 19 supercell BZ, 20 CLI regression |
| 21–27 | `crystod-phonon` | 21 irreps, 22 fatband, 23 LT bands, 24 eigenvectors, 25 modulation, 26 vibration, 27 CLI regression |
| 28–29 | `crystod-mag` | 28 spin bases, 29 CLI regression |
| 30–31 | `crystod-md` | 30 ADPs (`--adp`) and `--summary`, 31 CLI regression |
| 32–34 | `crystod-mol` | 32 molecular point groups & SALCs, 33 MO diagrams (`--diagram`, incl. `--pyscf`), 34 CLI regression |

Sections 6, 17, 20, 27, 29, 31, and 34 are command-line-interface regression tests
(every argument form plus removed-flag errors); they have no documentation
section of their own.

## Command summary

- `crystod -c POSCAR --element ELEMENT --orbital ORBITAL [--kpoint kx ky kz]` (k omitted: all special k points)
- `crystod -c POSCAR --atomic-orbital Ni_d O_p --kpoint kx ky kz`
- `crystod --visualize -c POSCAR --element EL --orbital ORB --kpoint kx ky kz [--real-coefficient] [--bond EL1 EL2 MAX] [--conventional] [--output FILE.html]`
- `crystod --star-of-k -c POSCAR --kpoint QLABEL_OR_KX KY KZ`
- `crystod-group --product IRREP1 IRREP2 ... --point-group PG`
- `crystod-group --product IRREP1 IRREP2 ... --space-group SG`   (full space-group irreps, e.g. R4- R5+ for Pm-3m)
- `crystod-group --table --point-group PG`
- `crystod-group --decompose --point-group PG [--characters X1 X2 ...]`
- `crystod-group --ligand-field ORBITAL --point-group PG`
- `crystod-group --basis BASIS1 BASIS2 ... --point-group PG`
- `crystod-group --basis BASIS1 BASIS2 ... --space-group SG [--kpoint kx ky kz]`
- `crystod-group --generate-basis --point-group PG [--order 1 2 3]`
- `crystod-group --coset --point-group PG --subgroup H`
- `crystod-group --coset --space-group SG --kpoint kx ky kz`
- `crystod-group --supergroup SG --irrep IR [IR2 ...] [--order-parameter 0 0 a]`
- `crystod-group --multiplet IRREP^N|IRREPN [IRREP^N|IRREPN ...] --point-group PG [--orbital s|p|d|f] [--visualize [--output FILE.html]]`
- `crystod-group --poscar2cif -c POSCAR [--tolerance 0.01] [--output FILE.cif]`
- `crystod-group --cif2poscar -c FILE.cif [--conventional] [--tolerance 0.01] [--output POSCAR]`
- `crystod-group --supergroup-cif HIGH.cif --subgroup-cif LOW.cif [--tolerance 0.01]`
- `crystod-bz -c POSCAR [--band ... --band-labels ...] [--output FILE.html]`
- `crystod-bz -c POSCAR --trans-mat "t11 t12 t13  t21 t22 t23  t31 t32 t33"`
- `crystod-bz --show-kpoint --space-group SG`
- `crystod-phonon --irreps --dim "nx ny nz" -c POSCAR [--readfc]`
- `crystod-phonon --fatband --dim nx ny nz -c POSCAR [--element EL] [--nac]`
- `crystod-phonon --lt --dim nx ny nz -c POSCAR [--nac]`
- `crystod-phonon --vector --dim "nx ny nz" -c POSCAR --qpoint Q [--mode N1 N2 ...] [--conventional]`
- `crystod-phonon --modulation --qpoint qx qy qz [--mode ...] [--amplitude ...] [--yaml phonopy_params.yaml]`
- `crystod-phonon --vibration -c POSCAR --qpoint Q [--mode-index N]`
- `crystod-mag -c POSCAR --element EL [--qpoint Q] [--format vasp|qe] [--conventional]`
- `crystod-md --adp --dim nx ny nz [--start-step N] [--xdatcar XDATCAR] [--output ADP.cif]`
- `crystod-mol --symmetry --xyz FILE.xyz [--tolerance TOL]`
- `crystod-mol --xyz FILE.xyz --element EL --orbital s|p|d|f [--align] [--show-matrix] [--visualize]`
- `crystod-mol --diagram --xyz FILE.xyz [--center EL] [--tolerance TOL] [-o FILE.html]`
- `crystod-mol --diagram --xyz FILE.xyz --pyscf [--basis BAS] [--theory scf|dft] [--xc XC] [--charge N] [--spin 2S] [--ao-left FORMULA --ao-right FORMULA]`
- `crystod-md --summary [--start-step N] [--end-step M] [--xdatcar XDATCAR]`

## Notes

- The pre-v0.3.0 flat modes (`crystod --<mode>`) were removed in v0.3.0; invoking
  one prints the equivalent sectioned command.
- `--show-irrep-table` in SALC mode prints the little-group irrep table at the
  selected k point; in `--product` mode it prints the point-group character table.
- Some workflows depend on the versions of `phonopy`, `spglib`, `spgrep`, and
  `irreptables`. CrystOD includes compatibility helpers for newer environments,
  but keeping these packages reasonably up to date is recommended.
