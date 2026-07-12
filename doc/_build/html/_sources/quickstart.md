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
| 7–13 | `crystod-group` | 7 product, 8 decompose, 9 ligand field, 10 basis, 11 generate-basis, 12 coset, 13 CLI regression |
| 14–16 | `crystod-bz` | 14 Brillouin zone, 15 supercell BZ, 16 CLI regression |
| 17–23 | `crystod-phonon` | 17 irreps, 18 fatband, 19 LT bands, 20 eigenvectors, 21 modulation, 22 vibration, 23 CLI regression |
| 24–25 | `crystod-mag` | 24 spin bases, 25 CLI regression |
| 26–27 | `crystod-md` | 26 ADPs (`--adp`) and `--summary`, 27 CLI regression |

Sections 6, 13, 16, 23, 25, and 27 are command-line-interface regression tests
(every argument form plus removed-flag errors); they have no documentation
section of their own.

## Command summary

- `crystod -c POSCAR --element ELEMENT --orbital ORBITAL [--kpoint kx ky kz]` (k omitted: all special k points)
- `crystod -c POSCAR --atomic-orbital Ni_d O_p --kpoint kx ky kz`
- `crystod --visualize -c POSCAR --element EL --orbital ORB --kpoint kx ky kz [--real-coefficient] [--bond EL1 EL2 MAX] [--conventional] [--output FILE.html]`
- `crystod --star-of-k -c POSCAR --kpoint QLABEL_OR_KX KY KZ`
- `crystod-group --product IRREP1 IRREP2 ... --point-group PG`
- `crystod-group --table --point-group PG`
- `crystod-group --decompose --point-group PG [--characters X1 X2 ...]`
- `crystod-group --ligand-field ORBITAL --point-group PG`
- `crystod-group --basis BASIS1 BASIS2 ... --point-group PG`
- `crystod-group --basis BASIS1 BASIS2 ... --space-group SG [--kpoint kx ky kz]`
- `crystod-group --generate-basis --point-group PG [--order 1 2 3]`
- `crystod-group --coset --point-group PG --subgroup H`
- `crystod-group --coset --space-group SG --kpoint kx ky kz`
- `crystod-bz -c POSCAR [--band ... --band-labels ...] [--output FILE.html]`
- `crystod-bz -c POSCAR --trans-mat "t11 t12 t13  t21 t22 t23  t31 t32 t33"`
- `crystod-phonon --irreps --dim "nx ny nz" -c POSCAR [--readfc]`
- `crystod-phonon --fatband --dim nx ny nz -c POSCAR [--element EL] [--nac]`
- `crystod-phonon --lt --dim nx ny nz -c POSCAR [--nac]`
- `crystod-phonon --vector --dim "nx ny nz" -c POSCAR --qpoint Q [--mode N1 N2 ...] [--conventional]`
- `crystod-phonon --modulation --qpoint qx qy qz [--mode ...] [--amplitude ...] [--yaml phonopy_params.yaml]`
- `crystod-phonon --vibration -c POSCAR --qpoint Q [--mode-index N]`
- `crystod-mag -c POSCAR --element EL [--qpoint Q] [--format vasp|qe] [--conventional]`
- `crystod-md --adp --dim nx ny nz [--start-step N] [--xdatcar XDATCAR] [--output ADP.cif]`
- `crystod-md --summary [--start-step N] [--end-step M] [--xdatcar XDATCAR]`

## Notes

- The pre-v0.3.0 flat modes (`crystod --<mode>`) were removed in v0.3.0; invoking
  one prints the equivalent sectioned command.
- `--show-irrep-table` in SALC mode prints the little-group irrep table at the
  selected k point; in `--product` mode it prints the point-group character table.
- Some workflows depend on the versions of `phonopy`, `spglib`, `spgrep`, and
  `irreptables`. CrystOD includes compatibility helpers for newer environments,
  but keeping these packages reasonably up to date is recommended.
