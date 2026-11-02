# API reference

Every CrystOD analysis is a Python function or class as well as a command.
The public API is split into ten domain modules, one per command; each
module carries the vocabulary of the command it mirrors (ISO-IR irrep labels,
k-point names, order-parameter directions), and the pages below are generated
from the docstrings of those modules.

| Module | Command | Scope |
|---|---|---|
| [`crystod.salc`](salc.md) | `crystod` | SALC irreps at a k point, crystal-orbital diagrams (extended Hueckel and PySCF), SALC coefficient bases, the star of k |
| [`crystod.group`](group.md) | `crystod-group` | character tables, direct products and reduction, isotropy subgroups, symmetry-mode analysis, multiplets, ISO-IR tables |
| [`crystod.phonon`](phonon.md) | `crystod-phonon` | phonon irrep labels, subgroups of imaginary modes, symmetry-adapted eigenvectors and VESTA export, LT character, modulations, symmetry-only vibrations |
| [`crystod.bz`](bz.md) | `crystod-bz` | Brillouin-zone polyhedra, seekpath and manual k paths, supercell folding, special k-point tables |
| [`crystod.mag`](mag.md) | `crystod-mag` | spin representations at q and their symmetry-adapted bases, cluster-multipole (FM/AFM) classification |
| [`crystod.md`](md.md) | `crystod-md` | XDATCAR trajectories, site-symmetry constraints on anisotropic displacement parameters |
| [`crystod.mol`](mol.md) | `crystod-mol` | molecular point groups, molecular SALCs, MO diagrams (extended Hueckel, fragments, PySCF) |
| [`crystod.xrd`](xrd.md) | `crystod-xrd` | powder X-ray diffraction patterns: Bragg peaks for a radiation, broadening, table and figure writers |
| [`crystod.search`](search.md) | `crystod-search` | Materials Project search (formula, chemical system, elements, IDs, filters), structure download in the standardized cells, POSCAR writers |
| [`crystod.slab`](slab.md) | `crystod-slab` | surface slabs: every termination of given or of all distinct Miller indices (pymatgen `SlabGenerator`), polar and symmetric tests, the slab table, POSCAR writers |

```{note}
Importing these modules is cheap. `import crystod` and the ten domain
modules pull in nothing heavier than NumPy: each name is resolved on first
access (PEP 562), and phonopy, spgrep, spglib, seekpath, pymatgen, requests
and PySCF are imported only by the function or class that needs them. Bad input raises
`ValueError` from the functions of these namespaces, where the command line
prints `ERROR: ...` and exits; the classes are the implementation classes
themselves and raise `SystemExit` as the command does.
```

The [Python API](../python-api.md) page is the narrative guide: it walks
through the common workflows (isotropy subgroups of an irrep, phonon-mode
labeling, the crystal-orbital and MO diagrams from Python) with worked
examples. The [tutorials](../tutorials.md) are the same workflows as
executable notebooks. This section is the complete listing, one page per
module, with the signature and the Google-style docstring of every public
name; the `[source]` link of each entry opens the implementation module the
name is resolved from.

```{toctree}
:maxdepth: 1

salc
group
phonon
bz
mag
md
mol
xrd
search
slab
```
