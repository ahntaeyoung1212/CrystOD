# Installation

## Requirements

- Python 3.9 or later
- Main Python dependencies:
  `phonopy`, `spglib`, `spgrep`, `irrep`, `irreptables`, `ase`, `seekpath`,
  `pymatgen`, `numpy`, `pandas`, `openpyxl`, `sympy`

## Recommended setup (conda)

```bash
conda create -n crystod python=3.11
conda activate crystod

unzip ~/Downloads/CrystOD-main.zip
mv ~/Downloads/CrystOD-main ~
cd ~/CrystOD-main
pip install -e .
```

## Operation check

Run the full test suite in the repository root (inside the `crystod` environment):

```bash
python3 testsuite.py           # run everything (33 sections)
python3 testsuite.py 13 16     # run selected sections only
```

Every section number corresponds to one documented feature and one
`example/<N>_*` directory; see {doc}`quickstart`.
