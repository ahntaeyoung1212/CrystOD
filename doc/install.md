# Installation

## Requirements

- Python 3.9 or later
- Main Python dependencies:
  `phonopy`, `spglib`, `spgrep`, `irrep`, `irreptables`, `ase`, `seekpath`,
  `numpy`, `pandas`, `openpyxl`, `sympy`

## Recommended setup (conda)

```bash
conda create -n crystod python=3.11
conda activate crystod

unzip ~/Downloads/CrystOD-main.zip
mv ~/Downloads/CrystOD-main ~
cd ~/CrystOD-main
pip install phonopy spglib spgrep irrep irreptables ase seekpath numpy openpyxl pandas

python3 -m pip install -e ~/CrystOD-main --no-deps --no-build-isolation
```

If your default Python environment uses an old `setuptools`, avoid
`--no-build-isolation` for the editable install:

```bash
python3 -m pip install -e . --no-deps
```

If needed, upgrade the build backend first:

```bash
python3 -m pip install -U "setuptools>=68" wheel
```

## Operation check

Run the full test suite in the repository root (inside the `crystod` environment):

```bash
python3 testsuite.py           # run everything (27 sections)
python3 testsuite.py 17 20     # run selected sections only
```

Every section number corresponds to one documented feature and one
`example/<N>_*` directory; see {doc}`quickstart`.
