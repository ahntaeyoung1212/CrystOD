# Installation

## Requirements

- Python 3.9 or later
- Main Python dependencies:
  `phonopy`, `spglib`, `spgrep`, `irrep`, `irreptables`, `ase`, `seekpath`,
  `pymatgen`, `numpy`, `pandas`, `openpyxl`, `sympy`

## Recommended setup (conda + git clone)

```bash
conda create -n crystod python=3.11
conda activate crystod

git clone https://github.com/ahntaeyoung1212/CrystOD.git
cd CrystOD
pip install -e .
```

The editable install (`-e`) keeps the commands pointing at the cloned source
tree, so `git pull` is enough to update, and the `example/` directories and
`testsuite.py` used throughout this documentation are right there.

## From PyPI (once released)

After the PyPI release, no clone is needed:

```bash
conda create -n crystod python=3.11
conda activate crystod
pip install crystod
```

This installs the commands and their dependencies only; clone the repository
as above if you also want the worked examples and the test suite.

## Operation check

Run the full test suite in the repository root (inside the `crystod` environment):

```bash
python3 testsuite.py           # run everything (34 sections)
python3 testsuite.py 13 16     # run selected sections only
```

Every section number corresponds to one documented feature and one
`example/<N>_*` directory; see {doc}`quickstart`.
