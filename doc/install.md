# Installation

## Requirements

- Python 3.10 or later
- Main Python dependencies (installed automatically):
  `phonopy`, `spglib`, `spgrep`, `ase`, `seekpath`, `pymatgen`,
  `numpy`, `scipy`, `sympy`, `pandas`, `matplotlib`, `requests`
- Optional: `pyscf`, for the quantitative `--pyscf` engines only (see below)
- Optional: `openpyxl` (or `XlsxWriter`), only for `crystod --element EL
  --orbital ORB --kpoint ... --show-irrep-table` and `crystod --atomic-orbital
  ... --show-irrep-table`, which also write the printed irrep table to
  `irrep_table.xlsx` in the working directory; neither is installed with
  CrystOD, and without one these runs stop with `ModuleNotFoundError: No
  module named 'openpyxl'` (`pip install openpyxl`)
- For `crystod-search` only: a network connection and a free Materials
  Project API key (see below)
- For viewing the 3D HTML viewers written by `crystod --visualize`,
  `crystod-phonon --visualize`, `crystod-bz` and `crystod-mol --visualize`: a
  network connection in the browser, because these pages load plotly.js from
  `cdn.plot.ly` when they are opened. The commands that write them run offline, and the
  crystal-orbital and MO diagram pages are self-contained.

## From PyPI (Recommended)

```bash
pip install CrystOD
```

This installs the eleven commands and their dependencies: everything needed for
the symmetry analysis, the extended-Hückel crystal-orbital diagrams, the phonon
irreps and the MO diagrams. A few example inputs are bundled, so the first run
needs no file of your own (`crystod --example ScF3_d`; see {doc}`quickstart`),
and so are the worked examples: `crystod-example` lists the example
directories of the features and `crystod-example 17` copies one, with its
README and inputs, into `./example` (see {doc}`crystod-example`); they come
without their large or unpublished data. The test suite and development need a
clone of the repository, as below.

### Optional: PySCF for the quantitative engines

```bash
pip install "CrystOD[quantum]"
```

The `[quantum]` extra adds PySCF, which only the quantitative engines use:
`crystod --diagram --pyscf`, `--band --pyscf`, `--dos --pyscf`,
`--visualize --pyscf`, and `crystod-mol --diagram --pyscf`. Everything else
(SALC irreps, extended-Hückel diagrams, group theory, phonon irreps and
modulations, spin bases, Brillouin zones, molecular SALCs) runs without it.
PySCF is about 500 MB with its dependencies, which is why it is not installed
by default; a `--pyscf` run without it stops with a one-line `ERROR:` that
names the command and this `pip install` line. The extra pins `pyscf<2.14`
because PySCF 2.14.0 cannot run a cell with zero electrons, which is what a
fully ionized fragment (Al³⁺ of AlN, for example) becomes under a GTH
pseudopotential in `--diagram --pyscf`.

### Materials Project API key (`crystod-search`)

`crystod-search` searches the Materials Project and downloads structures from
it, and the Materials Project API needs a personal key. The key is free: log
in at <https://next-gen.materialsproject.org/api> and copy the key shown
there. Then either export it (add the line to `~/.bashrc` or `~/.zshrc` to
keep it),

```bash
export MP_API_KEY=<your key>
```

or store it once in the pymatgen settings file, where pymatgen's own
`MPRester` finds it as well:

```bash
pmg config --add PMG_MAPI_KEY <your key>
```

Nothing else has to be installed: the HTTP client, `requests`, is a dependency
of CrystOD. Current keys have 32 characters; a 16-character key belongs to
the retired legacy API and is refused with a message saying so. The other
commands never go online and need no key (only some of the HTML pages they
write load plotly.js from the web when opened; see the requirements above). See
[The API key](crystod-search.md#the-api-key) for the details.

## Setup for following this manual (conda + git clone)

```bash
conda create -n crystod python=3.11
conda activate crystod

git clone https://github.com/ahntaeyoung1212/CrystOD.git
cd CrystOD
pip install -e ".[quantum]"
```

The editable install (`-e`) keeps the commands pointing at the cloned source
tree, so `git pull` is enough to update, and the `example/` directories and
`testsuite.py` used throughout this documentation are right there (the example
data that is not in the pip package stays in the CrystOD development tree; see
{doc}`crystod-example`).
`[quantum]` brings in PySCF for the `--pyscf` sections of this manual; `pip install -e
".[dev]"` additionally installs what is needed to build this documentation and
to run the tutorial notebooks (Sphinx, ruff, nbconvert, ipykernel).

## Operation check

Run the full test suite in the repository root (inside the `crystod` environment):

```bash
python3 testsuite.py           # run everything (53 sections)
python3 testsuite.py 17 23     # run selected sections only
```

Every section number corresponds to one documented feature and one
`example/<N>_*` directory; see {doc}`quickstart`. The test suite is not
installed by pip: it reads the `example/` directories of the repository.
Without PySCF the PySCF-dependent checks of sections 3, 7 and 43 are skipped
(the suite prints `[SKIP] pyscf not installed`) and every other check runs as
usual.

## Troubleshooting

### `ModuleNotFoundError: No module named 'crystod'` even though `pip show crystod` finds it

The command that ran is not the one that belongs to the environment the package
was installed into. Compare the two:

```bash
pip show crystod | grep Location   # where the package is
which crystod                      # which launcher script actually runs
```

A mismatch — `which` reporting e.g. `~/miniconda3/bin/crystod` while the package
sits in `~/miniconda3/envs/<env>/lib/python3.11/site-packages` — means the shell
is running a launcher script left behind by an earlier installation (a removed
editable install, or an environment whose Python was replaced). The first line of
that script points at the other interpreter, and that interpreter has no
`crystod` to import.

Remove the leftovers and let the environment's own scripts take over:

```bash
conda deactivate
pip uninstall -y crystod
rm -f ~/miniconda3/bin/crystod ~/miniconda3/bin/crystod-*
conda activate <env>
pip install --force-reinstall --no-deps crystod
hash -r
which crystod                      # now inside the environment
```

Every command is also a module, which bypasses the launcher scripts entirely and
is a reliable fallback:

```bash
python -m crystod --help
python -m crystod.cli.bz -c POSCAR --band "0 0 0  0 1/2 0" --band-labels "GM X"
```

`crystod.cli.bz`, `.group`, `.phonon`, `.mag`, `.md`, `.mol`, `.xrd`, `.search`,
`.slab`, `.example` correspond to `crystod-bz`, `crystod-group`, `crystod-phonon`,
`crystod-mag`, `crystod-md`, `crystod-mol`, `crystod-xrd`, `crystod-search`,
`crystod-slab`, `crystod-example`.

### PySCF uses only one CPU core on macOS (`--pyscf` features)

Everything computed by the PySCF engines — `crystod --diagram --pyscf`,
`--band --pyscf`, `--dos --pyscf`, and `crystod-mol --diagram --pyscf`,
i.e. what the `[quantum]` extra installs — inherits PySCF's own OpenMP
threading, and the PyPI *wheel* of PySCF for macOS (Apple silicon) is built
**without OpenMP**: every SCF runs on one core no matter how many the machine
has. Check what your environment got:

```bash
python -c "from pyscf import lib; print(lib.num_threads())"
```

`1` on a multi-core machine (PySCF may also warn "OpenMP is not
available") means the OpenMP-less wheel. This is specific to the macOS
wheels; the symmetry + extended-Hückel engines never call PySCF and are
unaffected either way.

The fix is to rebuild PySCF from source against Homebrew's `libomp`.
Measured on a 32-core Apple-silicon Mac Studio: the SCF wall-clock time
improves about 8×, and the energies are identical to the wheel's digit
for digit.

Prerequisites (once):

```bash
brew install libomp cmake
```

Then, inside the environment (replace `2.13.1` with whatever
`pip show pyscf` reports, and the job count with your core count):

```bash
conda activate crystod
export CMAKE_CONFIGURE_ARGS="-DCMAKE_PREFIX_PATH=/opt/homebrew/opt/libomp"
export CMAKE_POLICY_VERSION_MINIMUM=3.5
export CMAKE_BUILD_PARALLEL_LEVEL=8
pip install --force-reinstall --no-deps --no-binary pyscf pyscf==2.13.1
```

What the three variables do:

- `CMAKE_CONFIGURE_ARGS` points CMake's OpenMP detection at Homebrew's
  `libomp` (Apple's clang does not bundle one).
- `CMAKE_POLICY_VERSION_MINIMUM=3.5` is required with CMake ≥ 4, which
  otherwise refuses the older CMake project files of PySCF's bundled
  libraries.
- `CMAKE_BUILD_PARALLEL_LEVEL` sets the compile parallelism; the source
  build (it compiles `libcint`, `libxc`, `xcfun`) takes several minutes.

`--no-deps` guarantees nothing but PySCF itself is touched, and
`--no-binary pyscf` forces the source build. Verify afterwards:

```bash
python -c "from pyscf import lib; print(lib.num_threads())"   # = core count
otool -L $(python -c "import pyscf, os; print(os.path.dirname(pyscf.__file__))")/lib/libnp_helper.dylib | grep omp
#   /opt/homebrew/opt/libomp/lib/libomp.dylib   <- linked
```

The thread count of a run can then be capped with the usual
`OMP_NUM_THREADS` environment variable. One caveat: any later
`pip install --upgrade pyscf` — including a `crystod` reinstall that
happens to upgrade PySCF — silently brings the OpenMP-less wheel back;
re-run the rebuild after such an upgrade.
