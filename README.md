# CrystOD

**Cryst**al **O**rbital **D**iagram — a Python package and command-line toolset for the
symmetry analysis of crystals and molecules: crystal-orbital and SALC irreducible
representations, space-group irrep direct products, isotropy subgroups, phonon irreps,
symmetry-adapted phonon modulations, molecular point groups and MO diagrams.

**Documentation: <https://mochizuki-tus.github.io/CrystOD/>**

Every irrep label follows one convention throughout — the ISO-IR (ISOTROPY, Miller–Love)
tables, which ship inside the package — at the special k points and equally on symmetry
lines, planes and general points.

## The seven commands

| command | what it gives you |
|---|---|
| `crystod` | crystal-orbital / SALC irreps from atomic orbitals, orbital hybridization, crystal-orbital diagrams (extended Hückel or PySCF), band structure, DOS, 3D SALC viewers |
| `crystod-group` | direct products of point- and space-group irreps, reducible-representation decomposition, ligand-field splitting, polynomial basis functions, coset decompositions, isotropy subgroups, multi-electron terms, POSCAR ↔ CIF, symmetry-mode (AMPLIMODES-style) analysis |
| `crystod-bz` | interactive 3D Brillouin zones, automatic or manual k-paths, supercell (folded) BZs, special k points of any space group |
| `crystod-phonon` | phonon irrep labeling, element-projected fatbands, longitudinal/transverse bands, eigenvector VESTA export, symmetry-adapted modulations, symmetry-only vibration bases, isotropy subgroups of imaginary modes |
| `crystod-mag` | symmetry-adapted spin bases (cluster multipoles / SAMM) with ready-to-paste VASP `MAGMOM` or Quantum ESPRESSO input |
| `crystod-md` | atomic displacement parameters (ADPs) and time-averaged cells from an MD trajectory |
| `crystod-mol` | molecular point groups, molecular SALCs, and MO diagrams from symmetry + overlap (or PySCF) |

Several of these are offline counterparts of the Bilbao Crystallographic Server and
ISOTROPY tools (DIRPRO, ISOSUBGROUP, AMPLIMODES) and were cross-validated against them;
see the documentation for the validation details.

## Installation

```bash
pip install CrystOD
```

Requires Python 3.9 or later. The main dependencies (`phonopy`, `spglib`, `spgrep`,
`ase`, `seekpath`, `pymatgen`, `pyscf`, `numpy`, `scipy`, `sympy`, `pandas`,
`matplotlib`) are installed automatically.

To also get the worked examples and the test suite, clone the repository instead:

```bash
conda create -n crystod python=3.11 && conda activate crystod
```

```bash
git clone https://github.com/ahntaeyoung1212/CrystOD.git && cd CrystOD && pip install -e .
```

## Quick start

Irreps of the Sc 3d crystal orbitals of ScF₃ at every special k point:

```bash
crystod -c 221_PPOSCAR_ScF3 --element Sc --orbital d
```

Which space groups can the imaginary phonons of cubic SrTiO₃ condense into — and the
distorted structures themselves:

```bash
crystod-phonon --subgroup -c 221_PPOSCAR_SrTiO3 --dim "4 4 4" --qpoint R --modulate
```

Freeze a chosen mode combination into a structure (a unit cell plus `FORCE_SETS` is all
you need):

```bash
crystod-phonon --modulation -c 221_PPOSCAR_ScF3 --qpoint 0.5 0.5 0.5 --mode 1 2 3 --amplitude 0.3
```

Which subgroup a distortion of a given irrep and order-parameter direction leaves behind:

```bash
crystod-group --parent Pm-3m --irrep R4+
```

An MO diagram of a molecule from symmetry and overlap alone:

```bash
crystod-mol --diagram --xyz XYZ_CH4.xyz
```

Every command prints its own examples with `--help`, and the documentation shows the
output of each one.

## Python API

Every analysis is also a Python function, grouped into one module per command, so a part
of CrystOD can be used inside another program:

```python
import crystod

subgroups = crystod.group.isotropy_subgroups("Pm-3m", "R4+")
results = crystod.phonon.scan_imaginary_modes(phonon)   # a live phonopy object
```

`crystod.salc`, `crystod.group`, `crystod.phonon`, `crystod.bz`, `crystod.mag`,
`crystod.md`, `crystod.mol`. Attribute access is lazy, so `import crystod` plus all seven
domains costs ~0.09 s and pulls in nothing heavier than NumPy — phonopy, spgrep, PySCF
and matplotlib load only when a function that needs them is called.

## Documentation

The full manual — one page per command, every feature shown as *command in → output out*,
with the theory collected in separate background pages — is at
<https://mochizuki-tus.github.io/CrystOD/>.

The sources are MyST Markdown in `doc/`. To build them locally:

```bash
pip install -e ".[doc]" && python -m sphinx -b html doc doc/_build/html
```

## Testing

```bash
python testsuite.py
```

Runs the full regression suite (35 sections) against the data in `example/`; a section
can be run alone with `python testsuite.py 27`.

## Data sources and acknowledgements

- Irrep tables: **ISO-IR** dataset of the ISOTROPY Software Suite, shipped as
  `crystod/CIR_data.txt.gz` — H. T. Stokes, B. J. Campbell and R. Cordes,
  *Acta Cryst.* **A69**, 388–395 (2013), <https://iso.byu.edu>.
- Isotropy subgroups validated against **ISOSUBGROUP** — H. T. Stokes, S. van Orden and
  B. J. Campbell, *J. Appl. Cryst.* **49**, 1849–1853 (2016).
- Symmetry-mode analysis validated against **AMPLIMODES** — D. Orobengoa, C. Capillas,
  M. I. Aroyo and J. M. Perez-Mato, *J. Appl. Cryst.* **42**, 820–833 (2009).
- Built on phonopy, spglib, spgrep, ASE, seekpath, pymatgen and PySCF.

## Contributors

- **Yasuhide Mochizuki** — Tokyo University of Science ([mochizuki@rs.tus.ac.jp](mailto:mochizuki@rs.tus.ac.jp))
- **Hiroki Koiso** — Institute of Science Tokyo

## Citation

If you use CrystOD in your research, please cite:

> H. Koiso, S. Yoshida, T. Nagai, T. Isobe, A. Nakajima, and Y. Mochizuki,
> "Thermal expansion and phase stability of BF3 (B = Sc, Y, La, Al, Ga, In) from first
> principles", [Physical Review B **110**, 064104 (2024)](https://doi.org/10.1103/PhysRevB.110.064104).

```bibtex
@article{CrystOD,
  title   = {Thermal expansion and phase stability of $B$F$_3$ ($B$ = Sc, Y, La, Al, Ga, In) from first principles},
  author  = {Koiso, Hiroki and Yoshida, Suguru and Nagai, Takayuki and Isobe, Toshihiro and Nakajima, Akira and Mochizuki, Yasuhide},
  journal = {Phys. Rev. B},
  volume  = {110},
  pages   = {064104},
  year    = {2024},
  doi     = {10.1103/PhysRevB.110.064104},
}
```

## Changelog

See the [changelog](https://mochizuki-tus.github.io/CrystOD/changelog.html) for the
release history (`doc/changelog.md` in this repository).

## License

MIT License — see [LICENSE](LICENSE).
