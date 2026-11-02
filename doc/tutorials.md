# Tutorials

Three Jupyter notebooks in the `tutorials/` directory of the repository walk
through one CrystOD workflow each, with the Python API and the command line
side by side. They run on the example inputs bundled with the package
(`crystod.examples`), need nothing but `pip install CrystOD` and Jupyter, and
do not use PySCF. The notebooks are saved with their outputs, so they can be
read on GitHub or nbviewer without running anything.

| Notebook | For | Time |
|---|---|---|
| [01_phonon_irrep_labeling](https://github.com/ahntaeyoung1212/CrystOD/blob/main/tutorials/01_phonon_irrep_labeling.ipynb) | phonopy users | ~10 min |
| [02_isotropy_subgroup_search](https://github.com/ahntaeyoung1212/CrystOD/blob/main/tutorials/02_isotropy_subgroup_search.ipynb) | structural phase transitions | ~15 min |
| [03_mo_diagram](https://github.com/ahntaeyoung1212/CrystOD/blob/main/tutorials/03_mo_diagram.ipynb) | inorganic and computational chemists | ~10 min |

## 1. Phonon irrep labeling

Starting from a phonopy object of cubic SrTiO3 (the bundled 4x4x4 force sets),
`crystod.phonon.label_phonon_modes` labels every degenerate level at a q point
with its ISO-IR irrep, the star arms are mapped onto the tabulated q point
automatically, and the labels are drawn onto a matplotlib dispersion along the
seekpath path. The notebook ends with the command-line form,
`crystod-phonon --irreps`, and the `phonon_irreps.yaml` it writes.
Documentation: [28. Phonon irreps](crystod-phonon.md#28-phonon-irreps---irreps),
[Python API](python-api.md).
[View on GitHub](https://github.com/ahntaeyoung1212/CrystOD/blob/main/tutorials/01_phonon_irrep_labeling.ipynb)
| [View on nbviewer](https://nbviewer.org/github/ahntaeyoung1212/CrystOD/blob/main/tutorials/01_phonon_irrep_labeling.ipynb)

## 2. Isotropy subgroup search

The symmetry-lowering step of a structure search: `crystod.group.isotropy_subgroups`
enumerates the order-parameter directions of the octahedral-rotation irrep `R4+`
of Pm-3m and the space group each condenses into,
`crystod.phonon.imaginary_mode_subgroups` and `scan_imaginary_modes` do the same
for the imaginary R-point phonon of SrTiO3 straight from the phonopy object, and
a short experiment shows why freezing one eigenvector at a time finds I4/mcm and
C2/m but never R-3c or Imma. The candidate structures are then generated with
`crystod-phonon --subgroup --modulate` and their space groups re-measured with
spglib. Documentation: [35. Subgroups from imaginary modes](crystod-phonon.md#35-subgroups-from-imaginary-modes---subgroup),
[17. Isotropy subgroups](crystod-group.md#17-isotropy-subgroups---parent),
[Isotropy subgroups (theory)](theory-isotropy-subgroups.md).
[View on GitHub](https://github.com/ahntaeyoung1212/CrystOD/blob/main/tutorials/02_isotropy_subgroup_search.ipynb)
| [View on nbviewer](https://nbviewer.org/github/ahntaeyoung1212/CrystOD/blob/main/tutorials/02_isotropy_subgroup_search.ipynb)

## 3. MO diagram

For CH4 and NH3, `crystod.mol.load_molecule` and `get_symmetry` detect the point
group, `project_salcs` builds the H 1s SALCs (a1 + t2 for methane, a1 + e for
ammonia), and `crystod.mol.MODiagram` turns the SALCs and the central-atom
orbitals into a symmetry-adapted extended-Hückel MO diagram: the levels with
their irreps and occupations as a table, a matplotlib level diagram, and the
interactive HTML page that `crystod-mol --diagram` writes, embedded in the
notebook. Documentation: [43. Molecular-orbital diagrams](crystod-mol.md#43-molecular-orbital-diagrams---diagram),
[How the orbital diagrams are computed](theory-orbital-diagrams.md).
[View on GitHub](https://github.com/ahntaeyoung1212/CrystOD/blob/main/tutorials/03_mo_diagram.ipynb)
| [View on nbviewer](https://nbviewer.org/github/ahntaeyoung1212/CrystOD/blob/main/tutorials/03_mo_diagram.ipynb)

## Running the notebooks

```bash
pip install CrystOD jupyterlab    # the notebooks need no PySCF; any Jupyter front end works
git clone https://github.com/ahntaeyoung1212/CrystOD.git    # only for the notebooks
cd CrystOD/tutorials
jupyter lab
```

The notebooks themselves come from the repository (a clone, or the three
`.ipynb` files downloaded); their inputs are installed by pip with
`crystod.examples`. CrystOD 0.4.4 or later is required (Tutorial 1 reads
`phonon_irreps.yaml` with `crystod.phonon.read_phonon_irreps`, new in 0.4.4);
the first cell of every notebook prints the installed version. Since 0.4.5,
`crystod-example 28 17 45 43` also copies the example directories behind these
tutorials into `./example/` (see {doc}`crystod-example`). Each notebook begins by moving into a fresh temporary
directory (`tempfile.mkdtemp()`) and reads its inputs through
`crystod.examples.example_path(...)`, so everything it writes
(`phonon_irreps.yaml`, `MPOSCAR_*`, `MolOD_*.html`) lands there and never in the
repository. The command-line steps run in that directory through `subprocess`,
after `crystod.examples.copy_example_files` has placed the inputs next to them
-- the two steps that `crystod-phonon --example SrTiO3` performs in one line.
