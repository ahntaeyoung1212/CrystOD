[![PyPI version](https://img.shields.io/pypi/v/CrystOD)](https://pypi.org/project/CrystOD/)
[![Python](https://img.shields.io/pypi/pyversions/CrystOD)](https://pypi.org/project/CrystOD/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ahntaeyoung1212/CrystOD/blob/main/LICENSE)
[![Tests](https://img.shields.io/badge/tests-testsuite.py-blue)](https://mochizuki-tus.github.io/CrystOD/contributing.html#running-the-tests)
[![Docs](https://img.shields.io/badge/docs-mochizuki--tus.github.io%2FCrystOD-blue)](https://mochizuki-tus.github.io/CrystOD/)

# CrystOD

**The offline symmetry lab for crystals and molecules.**

CrystOD answers questions like:
- "Which irreps do the Ti *d* orbitals belong to in SrTiO₃?"
- "What space groups can this imaginary phonon condense into?"
- "What symmetry-adapted spin bases does this magnetic structure have?"
- "Show me the MO diagram of this molecule from symmetry alone."

It replaces clicking through Bilbao / ISOTROPY / AMPLIMODES with a single
pip-installable Python package and eleven CLI commands — fully scriptable,
cross-validated against the reference servers, and offline: every calculation
runs locally, and the only command that goes online is `crystod-search`, which
fetches structures from the Materials Project. Viewing some of the HTML output
does need a network connection: the 3D viewers written by `crystod --visualize`,
`crystod-phonon --visualize`, `crystod-bz` and `crystod-mol --visualize` load
plotly.js from `cdn.plot.ly` when the page is opened. The crystal-orbital and MO diagram pages are
self-contained.

**Documentation: <https://mochizuki-tus.github.io/CrystOD/>**

<p align="center">
  <img src="https://mochizuki-tus.github.io/CrystOD/_images/hero_crystal_orbital.png" width="45%" alt="Crystal-orbital diagram of ScF3 at the R point: Sc and F3 sublattice levels on the sides, the crystal orbitals with irrep labels in the middle, and the orbital sketch of the selected level" />
  <img src="https://mochizuki-tus.github.io/CrystOD/_images/hero_phonon_irrep.png" width="45%" alt="Phonon dispersion of cubic SrTiO3 with the ISO-IR irrep label of every level at the special k points; the imaginary R5- mode in red" />
</p>
<p align="center">
  <img src="https://mochizuki-tus.github.io/CrystOD/_images/hero_brillouin_zone.png" width="45%" alt="Interactive 3D Brillouin zone of ScF3 with the special k points and the seekpath k path" />
  <img src="https://mochizuki-tus.github.io/CrystOD/_images/hero_mo_diagram.png" width="45%" alt="MO diagram of CH4 from symmetry and overlap, with the 1t2 HOMO selected and its orbital sketch" />
</p>
<p align="center"><em>
Top row: <code>crystod --diagram -c 221_PPOSCAR_ScF3 --co-left Sc --co-right F3</code> and
<code>crystod-phonon --irreps -c 221_PPOSCAR_SrTiO3 --dim "4 4 4"</code> (the labels drawn on the phonopy dispersion, phonon_irreps.pdf, which every <code>--irreps</code> run writes);
bottom row: <code>crystod-bz -c 221_PPOSCAR_ScF3</code> and
<code>crystod-mol --diagram --xyz XYZ_CH4.xyz</code>. Every input is bundled with the package (see <a href="#try-it-without-any-input-file">below</a>).
</em></p>

Every irrep label follows one convention throughout — the ISO-IR (ISOTROPY, Miller–Love)
tables, which ship inside the package — at the special k points and equally on symmetry
lines, planes and general points. The double-valued irreps of spin-orbit coupling, which
the tables do not hold, are named from the same labels with the numbering of Bradley &
Cracknell (1972), Table 6.5 (`-GM8+`; two exceptions, stated in the output; not the Bilbao
numbering), the same on every arm of a star and in `crystod --spinor`, `--extract-irrep` and
the WAVECAR-overlap engine ([convention](https://mochizuki-tus.github.io/CrystOD/crystod.html#double-valued-irreps-spinor)).

## The eleven commands

| command | what it gives you |
|---|---|
| `crystod` | crystal-orbital / SALC irreps from atomic orbitals, orbital hybridization, crystal-orbital diagrams (extended Hückel, PySCF, or finished VASP runs — PROCAR projections, or WAVECAR overlaps with no energy alignment and spin-orbit runs included) with dipole selection rules between levels, band structure, DOS, 3D SALC viewers, the irreps of the bands of a finished VASP run read straight from its WAVECAR (`--extract-irrep`) |
| `crystod-group` | direct products of point- and space-group irreps and their symmetric and antisymmetric squares, Jahn-Teller active modes, reducible-representation decomposition, ligand-field splitting, polynomial basis functions, property-tensor forms (Neumann's principle), coset decompositions, point-group correlation tables, compatibility relations along symmetry lines and the subduction of parent irreps to the Gamma point of an isotropy subgroup, isotropy subgroups, their group-subgroup graph and the reverse lookup of the irreps (or pairs of irreps) giving a subgroup, invariant polynomials and secondary order parameters, multi-electron terms, POSCAR ↔ CIF, axis settings of a CIF / POSCAR, symmetry-mode (AMPLIMODES-style) analysis with the parent-to-child transformation matrix |
| `crystod-bz` | interactive 3D Brillouin zones, automatic or manual k-paths, supercell (folded) BZs, special k points of any space group |
| `crystod-phonon` | phonon irrep labeling, IR/Raman activity and Raman tensors, dielectric response from Born charges, element-projected fatbands, longitudinal/transverse bands, eigenvector VESTA export, symmetry-adapted modulations, symmetry-only vibration bases, isotropy subgroups of imaginary modes, phonon band unfolding of a distorted phase onto the parent Brillouin zone with parent irreps, interactive 3D viewer of the phonon modes by irrep (`--visualize`) |
| `crystod-mag` | symmetry-adapted spin bases (cluster multipoles / SAMM) with ready-to-paste VASP `MAGMOM` or Quantum ESPRESSO input; magnetic space group (BNS), magnetic point group, responses allowed by symmetry and FM / AFM / altermagnet (d/g/i-wave) class of every basis vector |
| `crystod-md` | atomic displacement parameters (ADPs) and time-averaged cells from an MD trajectory; projection onto the symmetry-adapted modes of the parent (order-parameter time series Q(t) per irrep) or onto the phonon eigenvectors, with effective frequencies |
| `crystod-mol` | molecular point groups, molecular SALCs, and MO diagrams from symmetry + overlap (or PySCF) |
| `crystod-xrd` | powder X-ray diffraction patterns: Bragg peak list (hkl, d, 2θ, intensity) and the broadened pattern, for Cu/Mo/Ag/Co/Fe/Cr radiation or neutrons; superlattice reflections of a distorted phase indexed on the parent lattice and assigned to parent k points and irreps (`--parent`) |
| `crystod-search` | Materials Project search by formula (`SrTiO3`), chemical system (`Sr-Ti-O`), elements or ID, listing space group, band gap, energy above hull and sites with the experimentally observed entries starred; `--get mp-5532` downloads the POSCAR |
| `crystod-slab` | surface slabs cut from a bulk structure: every termination of given Miller indices (also four-index, or a list file) or of every index up to a maximum that is distinct under the crystal's Laue group (pymatgen `SlabGenerator`), with atoms, thickness, vacuum, area, polar (dipole) and symmetric (equivalent faces) tests, `--max-atoms`, `--no-polar`, `--symmetrize`, and POSCAR files; interactive when no index is given |
| `crystod-example` | the worked examples that ship with the package: lists the example directories (one per feature, with a README of commands and their output, and the inputs) and copies them into a working directory |

Several of these are offline counterparts of the Bilbao Crystallographic Server and
ISOTROPY tools (DIRPRO, ISOSUBGROUP, AMPLIMODES) and were cross-validated against them;
see the documentation for the validation details.

## Installation

```bash
pip install CrystOD
```

This installs the eleven commands with everything needed for the symmetry analysis, the
extended-Hückel crystal-orbital diagrams, the phonon irreps and the MO diagrams.
Requires Python 3.10 or later. The dependencies (`phonopy`, `spglib`, `spgrep`, `ase`,
`seekpath`, `pymatgen`, `numpy`, `scipy`, `sympy`, `pandas`, `matplotlib`, `requests`) are
installed automatically; `openpyxl` is not among them, so install it yourself if you use
`crystod --show-irrep-table`, which also writes the table to `irrep_table.xlsx`.
`crystod-search` also needs a free Materials Project API key, from
<https://next-gen.materialsproject.org/api>: `export MP_API_KEY=<your key>`, or store it once
with `pmg config --add PMG_MAPI_KEY <your key>`.

```bash
pip install "CrystOD[quantum]"
```

This adds PySCF for the quantitative engines — `crystod --diagram/--band/--dos/--visualize
--pyscf` and `crystod-mol --diagram --pyscf`. PySCF is about 500 MB with its dependencies,
which is why it is an extra; a `--pyscf` run without it stops with a one-line error that
names this command.

The worked examples come with the package as well (`crystod-example`, see
[below](#worked-examples)), without their large or unpublished data. The test suite
and a development setup need a clone of the repository instead:

```bash
conda create -n crystod python=3.11 && conda activate crystod
```

```bash
git clone https://github.com/ahntaeyoung1212/CrystOD.git && cd CrystOD && pip install -e ".[quantum]"
```

## Quick start

### With your own structure

```bash
# Your POSCAR -> crystal-orbital irreps of the Ti d orbitals
crystod -c POSCAR --element Ti --orbital d

# Your POSCAR + FORCE_SETS (phonopy, 2x2x2 supercell) -> phonon irrep labels
# (--dim is the supercell the forces were computed in)
crystod-phonon --irreps -c POSCAR --dim "2 2 2"

# Space-group algebra (no structure file needed)
crystod-group --parent Pm-3m --irrep R4+      # isotropy subgroups of one irrep
crystod-group --parent Pm-3m --kpoint GM      # ... of every irrep at a k point

# No POSCAR yet? Search the Materials Project and download one
crystod-search Sr-Ti-O            # the Sr-Ti-O compounds; * = experimentally observed
crystod-search --get mp-5532      # -> POSCAR_Sr2TiO4_I4mmm_mp-5532
```

### Try it without any input file

A few small inputs ship inside the package, so the first run needs nothing but the
`pip install`:

```bash
crystod --example ScF3_d          # Sc 3d crystal-orbital irreps of ScF3
crystod-phonon --example SrTiO3   # phonon irrep labels of SrTiO3 (4x4x4 FORCE_SETS; writes phonon_irreps.yaml)
crystod-mol --example CH4         # MO diagram of methane (writes MolOD_XYZ_CH4.html)
crystod-bz --example ScF3         # 3D Brillouin zone of ScF3 (writes BZ_221_PPOSCAR_ScF3.html)
crystod-xrd --example ScF3        # powder XRD pattern of ScF3 for Cu K-alpha (table + PDF)
crystod-slab --example SrTiO3     # surface slabs of SrTiO3 up to (1 1 1), polar or not (POSCAR files)
```

`--example NAME` copies the input files of that example into the working directory,
prints the equivalent ordinary command line (`Running: crystod -c 221_PPOSCAR_ScF3
--element Sc --orbital d`) and runs it, so the files are there to edit and re-run.
`--example` alone lists the examples bundled with that command.

### Worked examples

Every feature also has a worked example: a directory with a README of commands and
their captured output, and the input files those commands read. All 44 example
directories ship with the package, without their large or unpublished data (about
7.8 MB in all), and `crystod-example` copies them into a working directory:

```bash
crystod-example                   # list the example directories (01_wigner_d ... 53_slab)
crystod-example 17                # copy one into ./example/17_isotropy_subgroup
crystod-example 29 33 -o work     # several, into work/ instead of ./example
crystod-example --all -o examples # all of them
```

The shared input structures (`test_POSCARs/`, `test_XYZs/`) and any other example
directory a README uses are copied along, so the relative paths in the READMEs work
in the copy; then `cd example/17_isotropy_subgroup` and follow its README. A file
already in place is kept if it is identical, and if it differs nothing is copied.
Large data (full MD trajectories, force-constant files), test-suite reference data
and generated output are not in the pip package, and the data of unpublished work is
not distributed; the READMEs mark the few commands that need such data
("not in the pip package", "unpublished, not distributed").

### More

Which space groups the imaginary phonons of cubic SrTiO₃ can condense into — and the
distorted structures themselves:

```bash
crystod-phonon --subgroup -c 221_PPOSCAR_SrTiO3 --dim "4 4 4" --qpoint R --modulate
```

Freeze a chosen mode combination into a structure (a unit cell plus `FORCE_SETS` is all
you need; `crystod-example 33` copies those of ScF3):

```bash
crystod-example 33 && cd example/33_modulation/ScF3_Pm-3m
crystod-phonon --modulation -c 221_PPOSCAR_ScF3 --qpoint 0.5 0.5 0.5 --mode 1 2 3 --amplitude 0.3
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
by_irrep = crystod.group.isotropy_subgroups_at_kpoint("Pm-3m", "GM")   # {irrep label: subgroups}
results = crystod.phonon.scan_imaginary_modes(phonon)   # a live phonopy object
```

`crystod.salc`, `crystod.group`, `crystod.phonon`, `crystod.bz`, `crystod.mag`,
`crystod.md`, `crystod.mol`, `crystod.xrd`, `crystod.search`, `crystod.slab`. Attribute access is lazy, so
`import crystod` plus all ten domains costs ~0.2 s and pulls in nothing heavier than NumPy —
phonopy, spgrep, PySCF and matplotlib load only when a function that needs them is called.

The API reference is at <https://mochizuki-tus.github.io/CrystOD/api/>. Three Jupyter
notebooks in [`tutorials/`](https://github.com/ahntaeyoung1212/CrystOD/tree/main/tutorials) walk through one workflow each, on the bundled
example inputs and with the Python API and the command line side by side:
[`01_phonon_irrep_labeling.ipynb`](https://github.com/ahntaeyoung1212/CrystOD/blob/main/tutorials/01_phonon_irrep_labeling.ipynb),
[`02_isotropy_subgroup_search.ipynb`](https://github.com/ahntaeyoung1212/CrystOD/blob/main/tutorials/02_isotropy_subgroup_search.ipynb) and
[`03_mo_diagram.ipynb`](https://github.com/ahntaeyoung1212/CrystOD/blob/main/tutorials/03_mo_diagram.ipynb).

### MCP server

[`crystod-mcp/`](https://github.com/ahntaeyoung1212/CrystOD/tree/main/crystod-mcp) in this repository packages the same analyses as the tools
of a Model Context Protocol server, so that an LLM client can run CrystOD on local
structure files; see the
[crystod-mcp](https://mochizuki-tus.github.io/CrystOD/crystod-mcp.html) page of the
documentation for the setup.

## Testing

```bash
python testsuite.py
```

Runs the full regression suite (53 sections) against the data in `example/`; a section
can be run alone with `python testsuite.py 35`. The `--pyscf` checks are skipped when
PySCF is not installed. GitHub Actions runs the suite on every push
([test.yml](https://github.com/ahntaeyoung1212/CrystOD/blob/main/.github/workflows/test.yml)) and `ruff` ([lint.yml](https://github.com/ahntaeyoung1212/CrystOD/blob/main/.github/workflows/lint.yml)).

## Contributing

Bug reports, questions and pull requests are welcome; [CONTRIBUTING.md](https://github.com/ahntaeyoung1212/CrystOD/blob/main/CONTRIBUTING.md)
describes the development setup, the code style and the pull-request process.

## Data sources and acknowledgements

- Irrep tables: **ISO-IR** dataset of the ISOTROPY Software Suite, shipped as
  `crystod/CIR_data.txt.gz` — H. T. Stokes, B. J. Campbell and R. Cordes,
  *Acta Cryst.* **A69**, 388–395 (2013), <https://iso.byu.edu>.
- Isotropy subgroups validated against **ISOSUBGROUP** — H. T. Stokes, S. van Orden and
  B. J. Campbell, *J. Appl. Cryst.* **49**, 1849–1853 (2016).
- Symmetry-mode analysis validated against **AMPLIMODES** — D. Orobengoa, C. Capillas,
  M. I. Aroyo and J. M. Perez-Mato, *J. Appl. Cryst.* **42**, 820–833 (2009).
- Structures and properties of `crystod-search`: the **Materials Project** — A. Jain *et al.*,
  *APL Mater.* **1**, 011002 (2013), <https://doi.org/10.1063/1.4812323>; the data are
  licensed under CC BY 4.0.
- Surface slabs of `crystod-slab`: pymatgen's `SlabGenerator` — W. Sun and G. Ceder,
  *Surf. Sci.* **617**, 53–59 (2013), <https://doi.org/10.1016/j.susc.2013.05.016>; R. Tran
  *et al.*, *Sci. Data* **3**, 160080 (2016), <https://doi.org/10.1038/sdata.2016.80>.
- Built on phonopy, spglib, spgrep, ASE, seekpath and pymatgen, and optionally on PySCF.

## Contributors

- **Yasuhide Mochizuki** — Tokyo University of Science ([mochizuki@rs.tus.ac.jp](mailto:mochizuki@rs.tus.ac.jp))
- **Hiroki Koiso** — Institute of Science Tokyo
- **Ryuju Kiribayashi** — Institute of Science Tokyo

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

`crystod-slab`: please also cite R. Kiribayashi *et al.*, *ACS Appl. Mater. Interfaces* **17**,
61872–61887 (2025), <https://doi.org/10.1021/acsami.5c17978>.

## License

MIT License — see [LICENSE](https://github.com/ahntaeyoung1212/CrystOD/blob/main/LICENSE).
