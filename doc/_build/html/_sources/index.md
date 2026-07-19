# CrystOD

**CrystOD** is a Python package and command-line toolset for symmetry analysis of
crystal orbitals, SALCs (symmetry-adapted linear combinations), basis functions,
phonon irreducible representations, symmetry-adapted phonon vibration bases and
modulations, magnetic (spin) multipole bases, and point-group representation theory.

The package is organized phonopy-style: one main command plus one sectioned command
per research domain.

| Command | Domain |
|---|---|
| `crystod` | crystal-orbital SALC analysis (main command), `--visualize`, `--star-of-k` |
| `crystod-group` | point/space-group representation-theory calculator |
| `crystod-bz` | Brillouin-zone plots (unit cell and supercell folding), special-k-point tables |
| `crystod-phonon` | phonon analyses (irreps, fatband, LT bands, eigenvectors, modulation, vibration) |
| `crystod-mag` | symmetry-adapted spin bases (cluster multipoles / SAMM) |
| `crystod-md` | MD-trajectory analyses (ADPs, lattice summary) |
| `crystod-mol` | molecular point groups, molecular SALCs, and MO diagrams (XYZ files) |

Every feature carries a shared section number used consistently across this
documentation, `testsuite.py`, and the `example/` directories of the repository:
feature *N* is tested by `python testsuite.py N` and demonstrated in `example/<N>_*`.

```{toctree}
:maxdepth: 1
:caption: Getting started

install
quickstart
```

```{toctree}
:maxdepth: 2
:caption: Commands

crystod
crystod-group
crystod-bz
crystod-phonon
crystod-mag
crystod-md
crystod-mol
```

```{toctree}
:maxdepth: 1
:caption: Reference

citation
changelog
```

## Citation

If you use CrystOD in your research, please cite:

> H. Koiso, S. Yoshida, T. Nagai, T. Isobe, A. Nakajima, and Y. Mochizuki,
> "Thermal expansion and phase stability of BF3 (B = Sc, Y, La, Al, Ga, In) from first principles",
> [Physical Review B **110**, 064104 (2024)](https://doi.org/10.1103/PhysRevB.110.064104).

## Contributors

- **Yasuhide Mochizuki** — Tokyo University of Science
- **Hiroki Koiso** — Institute of Science Tokyo

## License

GPLv3
