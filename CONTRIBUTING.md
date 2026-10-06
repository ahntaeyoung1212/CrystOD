# Contributing to CrystOD

This page is the entry point for anyone who wants to fix a bug, add a feature or
improve the documentation of CrystOD. It describes how the repository is organized,
how it is tested and how a release is made.

## Getting started

- **Bug reports and feature requests** go to
  <https://github.com/ahntaeyoung1212/CrystOD/issues>. For a wrong or missing
  symmetry result, attach the input structure (POSCAR or XYZ) and the exact command
  line; if you compared against another program (Bilbao Crystallographic Server,
  ISOTROPY/ISODISTORT, AMPLIMODES, phonopy), say which one and paste its answer.
- **Small fixes** (a typo, a docstring, an error message) can go straight into a pull
  request.
- **Larger changes** are best discussed in an issue first, so that the design is
  settled before the work is done.

## Development setup

CrystOD is developed on Linux and macOS with Python 3.11 in a conda environment; the
package itself supports Python 3.10 and later (`requires-python` in `pyproject.toml`).

```bash
conda create -n crystod python=3.11 && conda activate crystod
git clone https://github.com/ahntaeyoung1212/CrystOD.git && cd CrystOD
pip install -e ".[dev]"
```

The editable install (`-e`) makes the checkout importable and puts the nine console
scripts (`crystod`, `crystod-group`, `crystod-bz`, `crystod-phonon`, `crystod-mag`,
`crystod-md`, `crystod-mol`, `crystod-xrd`, `crystod-search`) on the path, so an edit
takes effect on the next run. The extras defined in `pyproject.toml` are:

| extra | contents | needed for |
|---|---|---|
| `quantum` | `pyscf` | the `--pyscf` engines (`crystod --diagram/--band/--dos/--visualize --pyscf`, `crystod-mol --diagram --pyscf`) and the test-suite checks that exercise them |
| `doc` | `sphinx`, `myst-parser`, `sphinx-book-theme`, `sphinx-copybutton` | building the documentation |
| `dev` | `quantum` + `doc` + `ruff`, `nbconvert`, `ipykernel`, `build`, `twine` | everything a contributor needs: linting, executing the tutorial notebooks, building a release |
| `all` | `quantum` + `doc` | a user install with every optional feature |

PySCF (about 500 MB with its dependencies) is optional on purpose: everything except
the `--pyscf` engines runs without it, and a `--pyscf` command stops with a one-line
`ERROR` that names the extra to install (`crystod/_optional.py`). Do not add an
unconditional `import pyscf` anywhere.

## Running the tests

The regression suite is `testsuite.py` at the repository root: plain Python, no
test-runner dependency. Run it from the repository root inside the environment:

```bash
python testsuite.py          # all 45 sections
python testsuite.py 17       # section 17 only
python testsuite.py 3 41     # sections 3 and 41
```

Each check prints `[PASS]` or `[FAIL]` with its name (and, on failure, the output of
the command it ran); the run ends with `Total: N passed, M failed` and exits with
status 1 if anything failed. The suite reads its inputs from `example/` and writes to
temporary directories, and every individual command has a 900 s timeout. Without
PySCF the PySCF-dependent checks in sections 3, 7 and 41 are skipped with a `[SKIP]`
line and the rest of the suite still runs; the CI matrix installs `[quantum]`, so
nothing is skipped there.

**The MCP server.** `crystod-mcp/` is a separate package with its own pytest suite
(22 tests: every tool on the bundled inputs plus a protocol round trip); run it after
installing the package next to CrystOD:

```bash
pip install -e "./crystod-mcp[test]"
python -m pytest -q crystod-mcp/tests
```

The `mcp` job of the Tests workflow runs it on every push.

**Section numbers.** One number identifies a feature everywhere in the repository:
section *N* of `testsuite.py` tests it, `example/NN_*/` holds its worked example (the
inputs, plus a line in `example/README` with the command), and section *N* of the
command's page in `doc/` documents it (`## 17. Isotropy subgroups (--parent)` in
`doc/crystod-group.md`, for example). The docstring at the top of `testsuite.py`
lists all 45 sections grouped by command: 1 library core, 2-7 `crystod`, 8-24
`crystod-group`, 25-27 `crystod-bz`, 28-35 `crystod-phonon`, 36-37 `crystod-mag`,
38-39 `crystod-md`, 40-42 `crystod-mol`, 43 Python API, 44 `crystod-xrd`, 45
`crystod-search`. A few sections hold the "extras" of a command (aliases, error
messages, removed flags) and have no example directory of their own.

`.github/workflows/test.yml` runs the full suite on Linux (Python 3.10, 3.12, 3.13)
and macOS (3.12) for every push and pull request to `main`.

## Building the docs

The documentation is MyST Markdown under `doc/`, built with Sphinx and
`sphinx-book-theme` (phonopy-style). With the `doc` (or `dev`) extra installed:

```bash
sphinx-build -b html -W --keep-going doc doc/_build/html
```

then open `doc/_build/html/index.html`. `-W` turns warnings into errors, and that is
what `.github/workflows/docs.yml` checks on every push. The published site,
<https://mochizuki-tus.github.io/CrystOD/>, lives in the laboratory website
repository and is copied there by the maintainer, so the workflow deploys nothing:
it verifies that the build is clean and keeps the HTML as a run artifact. `README.md`
and the corresponding pages in `doc/` (`install.md`, `quickstart.md`, the command
tables) are maintained in parallel; change both.

## Code style

- **ruff.** `ruff check .` from the repository root must pass;
  `.github/workflows/lint.yml` runs the same command. The configuration is
  `ruff.toml`: Python 3.10 target, line length 100 for new code, and a deliberately
  small rule set (syntax errors, undefined and redefined names, invalid comparisons,
  bare `except`), the rules that point at a bug rather than a preference, each
  enabled because it is clean on the whole tree. The comments in `ruff.toml` list
  the rules and the candidates that are not enabled yet.
- **Docstrings** are Google style (`Args:`, `Returns:`, `Raises:`) with a one-line
  summary first. Every public function and class needs one; they are what `help()`
  and the API documentation show.
- **English** in code, comments, docstrings, documentation and commit messages.
- **Errors.** The implementation modules double as command-line code and report bad
  input with `raise SystemExit("ERROR: ...")`: one sentence that names the remedy.
  The API layer (`crystod/_api.py`) turns that into a `ValueError` for library
  callers, so a function that follows this rule serves both. A dependency's
  traceback is never the message a user sees.
- **Imports.** An implementation module may import phonopy, spgrep, pyscf or
  matplotlib at module level, because the API domain modules load implementation
  modules lazily (PEP 562). `crystod/__init__.py`, the domain modules and any
  `--help` path must stay free of them (`import crystod` plus all nine domains is
  measured at about 0.09 s and a test in section 43 keeps it that way).

## How to add a feature

A feature touches the same places every time, in this order:

1. **Implementation module**, `crystod/<feature>.py`: a function or class that takes
   plain Python/NumPy inputs and returns data; the terminal report and any HTML or
   VESTA output are separate functions. Existing modules are the template
   (`crystod/isotropy_subgroup.py`, `crystod/phonon_subgroups.py`).
2. **API export**: add the public name to `_EXPORTS` in the domain module of the
   command it belongs to (`crystod/salc.py`, `group.py`, `phonon.py`, `bz.py`,
   `mag.py`, `md.py`, `mol.py`, `xrd.py`, `search.py`) as
   `"name": ("implementation_module", "attribute")`.
   That is the whole registration; `crystod.<domain>.name` then resolves lazily.
3. **Command line**: the flag in `build_parser()` of `crystod/cli/<command>.py` and
   its branch in `main()`, with an example in the `--help` epilog. A feature that
   needs PySCF calls `require_pyscf_or_exit()` before importing it.
4. **Tests**: checks in the section of that command in `testsuite.py` (or a new
   `test_NN_...` function registered in the `SECTIONS` dict at the bottom of the
   file), written with `run_cli()` and `report()` like the neighbouring checks.
   Assert on the physics (a space group, an irrep label, an amplitude), not only on
   the exit status, and say where the reference value comes from.
5. **Documentation**: the numbered section in `doc/<command>.md` with the command
   line and its output; the command table in `README.md` if a new mode was added;
   an entry in `doc/changelog.md`.
6. **Example**: `example/NN_<feature>/` with the inputs and a line in
   `example/README`. Generated outputs are git-ignored (`example/**/*.html`, `*.csv`,
   `*.chk`, ...); commit the inputs only.

## Commit messages and pull requests

Commit subjects follow the Conventional-Commits-like pattern already in the log: a
lower-case area prefix, a colon and an imperative summary, as in
`doc: add a desktop toggle for the right-hand table of contents` or
`diagram: URL-selectable k point and embed-friendly level panel`. Use the command or
module as the area (`phonon:`, `group:`, `mol:`, `diagram:`, `api:`, `cli:`), or
`doc:`, `test:`, `ci:`, `examples:` for the surrounding files. The body says *why*,
and for a symmetry result, what it was validated against. A release commit is simply
`CrystOD vX.Y.Z`.

A pull request should

- target `main` and pass the three workflows (Tests, Lint, docs);
- add or extend test-suite checks for the change, and update the documentation and
  the changelog together with the code;
- describe the change, the reason and the validation in its description, in English.

## Release process

1. Bump the version in **three files**, `pyproject.toml` (`version`),
   `crystod/__init__.py` (`__version__`) and `CITATION.cff` (`version` and
   `date-released`), and add the release section to `doc/changelog.md` (newest
   first). The documentation reads its version from `pyproject.toml`.
2. Run the full suite and the docs build, then build and check the distribution
   locally: `python -m build && twine check dist/*`.
3. Commit, tag and push:

   ```bash
   git commit -am "CrystOD v0.4.0"
   git tag v0.4.0
   git push origin main --tags
   ```

4. The tag triggers `.github/workflows/publish.yml`: the `build` job refuses a tag
   that does not match the three version fields, builds the sdist and the wheel,
   runs `twine check` and stores them as an artifact; the `publish` job then waits
   for approval in the `release` environment and uploads with PyPI Trusted
   Publishing (OpenID Connect; no API token is stored in the repository). The
   one-time setup of the publisher on pypi.org and of the `release` environment on
   GitHub is spelled out at the top of `publish.yml`.
5. Copy the freshly built documentation to the laboratory website repository.

## About DEVELOPMENT.md

`DEVELOPMENT.md` is the internal development log: a dated, Japanese-language record
of decisions, validations and dead ends. It is kept outside the public repository
(it is listed in `.gitignore`) and is not translated. This file is the entry point
for new contributors; anything a contributor needs to know belongs here or in `doc/`.
