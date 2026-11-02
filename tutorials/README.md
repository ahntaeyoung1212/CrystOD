# CrystOD tutorials

Three Jupyter notebooks that walk through one CrystOD workflow each, on the
Python API and the command line side by side. They run on the example inputs
bundled with the package (`crystod.examples`), so nothing has to be cloned or
computed beforehand, and none of them needs PySCF.

| Notebook | For | Time |
|---|---|---|
| [`01_phonon_irrep_labeling.ipynb`](01_phonon_irrep_labeling.ipynb) | phonopy users: ISO-IR irrep labels on the phonon dispersion of SrTiO3 | ~10 min |
| [`02_isotropy_subgroup_search.ipynb`](02_isotropy_subgroup_search.ipynb) | structural phase transitions: which space groups the imaginary R-point phonon of SrTiO3 condenses into, and the distorted candidate structures | ~15 min |
| [`03_mo_diagram.ipynb`](03_mo_diagram.ipynb) | inorganic and computational chemists: the MO diagrams of CH4 and NH3 from symmetry and overlap | ~10 min |

The notebooks are saved with their outputs, so they can be read on GitHub or
on [nbviewer](https://nbviewer.org/github/ahntaeyoung1212/CrystOD/tree/main/tutorials/)
without running anything.

## Running them

```bash
pip install CrystOD jupyterlab    # the notebooks need no PySCF; any Jupyter front end works
git clone https://github.com/ahntaeyoung1212/CrystOD.git    # only for the notebooks
cd CrystOD/tutorials
jupyter lab
```

The notebooks themselves come from the repository (a clone, or the three
`.ipynb` files downloaded), but their inputs are installed by pip with
`crystod.examples`. CrystOD 0.4.4 or later is required (Tutorial 1 reads
`phonon_irreps.yaml` with `crystod.phonon.read_phonon_irreps`, new in 0.4.4);
the first cell of every notebook prints the installed version. phonopy,
matplotlib, pandas and pymatgen are installed with CrystOD.

Since 0.4.5 the example directories of the manual, each with its README and
inputs, ship with pip too: `crystod-example` lists them, and
`crystod-example 28 17 45 43` copies the ones behind these tutorials (phonon
irreps, isotropy subgroups, the structure-search script, MO diagrams) into
`./example/`.

## The temporary-directory convention

Every notebook starts with

```python
import os, tempfile
workdir = tempfile.mkdtemp(prefix="crystod_tutorial_")
os.chdir(workdir)
```

and reads its inputs through `crystod.examples.example_path(...)`, which
returns the absolute path of a file inside the installed package. Everything a
notebook writes (`phonon_irreps.yaml`, `MPOSCAR_*`, `MolOD_*.html`) therefore
lands in that fresh directory, never in `tutorials/`, and the notebooks do not
depend on where they are opened from. The command-line steps are run with
`subprocess` in the same directory, after copying the example inputs there with
`crystod.examples.copy_example_files` -- the same two steps that
`crystod-phonon --example SrTiO3` performs in one line.

To re-execute all three from a shell:

```bash
jupyter nbconvert --to notebook --execute --inplace 0*.ipynb
```
