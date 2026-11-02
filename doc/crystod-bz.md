# crystod-bz

Interactive 3D Brillouin-zone plots, and the special-k-point tables behind
them.

| I want to ... | command |
|---|---|
| plot the BZ with its recommended k path | `crystod-bz -c POSCAR` |
| plot it with my own k path | `crystod-bz -c POSCAR --band "..." --band-labels "..."` |
| see how a supercell folds the BZ | `crystod-bz -c POSCAR --trans-mat "0 1 2  -1 0 2  1 -1 2"` |
| list the special k points of a space group | `crystod-bz --show-kpoint --sg Pnma` |

## 25. Brillouin zone plot

*Example directory: `example/25_brillouin_zone` (testsuite section 25); after pip install: `crystod-example 25`*

Plot the first Brillouin zone as an interactive 3D HTML file, together with the
recommended high-symmetry k-path:

```bash
crystod-bz -c 221_PPOSCAR_ScF3
crystod-bz -c 221_PPOSCAR_ScF3 --output BZ_ScF3_Pm-3m.html
```

```
* Structure *
Space group: Pm-3m (#221)

* Recommended k-path (seekpath) *
  GAMMA    ( 0.0000,  0.0000,  0.0000)
  X        ( 0.0000,  0.5000,  0.0000)
  M        ( 0.5000,  0.5000,  0.0000)
  R        ( 0.5000,  0.5000,  0.5000)
  Path: GAMMA-X-M-GAMMA-R-X   R-M

* Output files *
  Wrote Brillouin-zone visualization: BZ_221_PPOSCAR_ScF3.html
```

The written HTML is a live 3D plot — the one below is the actual output of the
command above (drag to rotate, scroll to zoom, hover the k points for their
coordinates):

```{raw} html
<iframe src="_static/embed/BZ_221_PPOSCAR_ScF3.html" width="100%" height="560" loading="lazy" style="border:1px solid #8884; border-radius:8px; background:#fff;"></iframe>
<p style="margin-top:0.3em"><a href="_static/embed/BZ_221_PPOSCAR_ScF3.html" target="_blank">Open this Brillouin zone full-screen</a></p>
```

`-c`/`--cell` selects the structure file (default: `POSCAR`); the former
`--poscar` spelling is kept as an alias.

The space group of the structure is detected, and the corresponding
high-symmetry k-path (e.g. GM-X-M-GM-R-X, M-R for Pm-3m) is generated
automatically with [seekpath](https://github.com/giovannipizzi/seekpath).
Disconnected path segments are drawn separately so that every plotted line is a
continuous band path. The detected space group, the high-symmetry k points with
their fractional coordinates, and the k-path are also printed to the terminal.

If `--output` is omitted, the plot is saved in the current directory as
`BZ_{structure name}.html` (e.g. `BZ_221_PPOSCAR_ScF3.html`).

A custom path can be supplied instead of the automatic one with
`--band` / `--band-labels`; coordinates may be given as decimals or fractions,
comma-separated into continuous segments:

```bash
crystod-bz -c 221_PPOSCAR_ScF3 \
    --band "0 0 0  0 1/2 0  1/2 1/2 0  0 0 0  1/2 1/2 1/2  0 1/2 0, 1/2 1/2 0  1/2 1/2 1/2" \
    --band-labels "GM X M GM R X M R"
```

Note: if the input cell differs from the seekpath standardized primitive cell,
the Brillouin zone and k-path are drawn for the standardized primitive cell
(a NOTE is printed in that case). Manual `--band` coordinates always refer to
the reciprocal basis of the input structure, and the zone is then drawn for
the input cell as it is.

The polyhedron does not depend on the basis in which the cell is given. It is
built on a reduced (Selling-reduced) copy of the reciprocal basis, for which
the Voronoi construction around the origin contains every lattice plane that
bounds the zone, and its volume is checked against `|det B|`, the volume of
the reciprocal unit cell (`B`: the reciprocal basis vectors as rows). An
oblique cell with a manual path, such as a raw monoclinic cell with
a = 4, b = 5, c = 6 Å and β = 150°, therefore gives the true Wigner-Seitz cell
(8 faces and 12 corners, of volume `|det B|`). Only
the construction uses the reduced basis: the k path, the drawn
b<sub>1</sub>, b<sub>2</sub>, b<sub>3</sub> arrows and the hover coordinates
stay in the basis of the input structure.

Zone corners that coincide within a relative tolerance (1e-5 times the length
of the shortest reciprocal lattice vector) are merged, and faces that collapse
are dropped, so a cell that is cubic up to noise of about 1e-6 Å is drawn as a
cube with 6 faces and 8 corners, not as a polyhedron with tiny extra faces.
The command always uses this default; its `--tolerance` option is the symmetry
tolerance of seekpath/spglib and does not change the merge.

**Python API.** `crystod.bz.get_brillouin_zone_3d(rec_lat, tolerance=1e-5)`
returns the `(vertices, ridges, facets)` of the zone, with the reciprocal basis
vectors as the rows of `rec_lat`. `tolerance` is the corner-merge threshold in
units of the shortest reciprocal lattice vector; `tolerance=0` keeps every
corner of the Voronoi construction. A sheared basis of the simple cubic
lattice gives the cube:

```python
import numpy as np
from crystod import bz

a = 4.07 * np.eye(3)
T = np.array([[1, 2, 0], [0, 1, 0], [0, 0, 1]])   # another basis of the same lattice
rec_lat = np.linalg.inv(T @ a).T
vertices, ridges, facets = bz.get_brillouin_zone_3d(rec_lat)
print(len(vertices), len(facets))                   # 8 6
```

### Special k points of a space group (`--show-kpoint`)

With `--show-kpoint`, no plot is produced; instead the special (high-symmetry)
k points of the space group given by `--space-group` are printed:

```bash
crystod-bz --show-kpoint --space-group Pnma
```

```
* Space group *
Pnma (No. 62)

* K points (primitive) *
GM: (0, 0, 0)
X: (1/2, 0, 0)
Y: (0, 1/2, 0)
Z: (0, 0, 1/2)
S: (1/2, 1/2, 0)
T: (0, 1/2, 1/2)
U: (1/2, 0, 1/2)
R: (1/2, 1/2, 1/2)
```

The k points and their ISO-IR (ISOTROPY, Miller-Love) labels are taken from
the bundled ISO-IR tables — the same
definition used by the SALC/irrep analyses (`crystod`, `crystod-mag`,
`crystod-group`, `crystod-phonon --irreps`) — and are given in the primitive
reciprocal basis. For centred lattices, whose conventional and primitive cells
differ, the coordinates in the conventional reciprocal basis are printed as
well. The centring is that of the ISO-IR standard setting of the group (origin
choice 2, monoclinic unique axis b with cell choice 1, hexagonal axes for R),
so the centring letters that occur are F, I, C, R and A, the last only for
Amm2, Abm2, Ama2 and Aba2 (Nos. 38-41). No standard setting is B-centred, so a
B-centred table never appears:

```bash
crystod-bz --show-kpoint --space-group Fm-3m
```

```
* Space group *
Fm-3m (No. 225)

* K points (primitive) *
GM: (0, 0, 0)
X: (1/2, 0, 1/2)
L: (1/2, 1/2, 1/2)
W: (1/2, 1/4, 3/4)

* K points (conventional) *
GM: (0, 0, 0)
X: (0, 1, 0)
L: (1/2, 1/2, 1/2)
W: (1/2, 1, 0)
```

`--space-group` also accepts the symbol of a non-standard setting (`Pbnm`,
`Amam`, `Bbmm`, `Bmm2`, ...). The symbol only selects the space group: the
table is still that of the ISO-IR standard setting, the heading names that
setting, and a NOTE names the setting that was requested. The coordinates are
not transformed to the requested axes:

```bash
crystod-bz --show-kpoint --space-group Pbnm
```

```
* Space group *
Pnma (No. 62)
NOTE: "Pbnm" names the non-standard setting P 2_1/b 2_1/n 2_1/m of space group No. 62.
      The labels and coordinates below refer to the ISO-IR standard setting
      P 2_1/n 2_1/m 2_1/a (axes and centring of Pnma), not to the requested axes.

* K points (primitive) *
GM: (0, 0, 0)
X: (1/2, 0, 0)
Y: (0, 1/2, 0)
Z: (0, 0, 1/2)
S: (1/2, 1/2, 0)
T: (0, 1/2, 1/2)
U: (1/2, 0, 1/2)
R: (1/2, 1/2, 1/2)
```

In the same way `Amam` and `Bbmm` print the C-centred table of Cmcm
(No. 63), primitive and conventional, with the NOTE. A standard symbol or a
space-group number (`--space-group 62`) prints no NOTE.

**Python API.** `crystod.bz.get_special_kpoints(symbol)` returns
`(sg_type, names, primitive, conventional)`. `sg_type` is the spglib record of
the ISO-IR standard setting that the coordinates refer to, so its
`international_short`, `international_full` and `hall_number` can differ from
the symbol given (the space-group `number` is that of the group):
`bz.get_special_kpoints("Pbnm")[0].international_short` is `'Pnma'`, and
`"Amam"` and `"Bbmm"` give `'Cmcm'`. `conventional` is `None` for a primitive
lattice.

## 26. Supercell Brillouin zone (`--trans-mat`)

*Example directory: `example/26_bz_supercell` (testsuite section 26); after pip install: `crystod-example 26`*

```{note}
The `example/...` paths on this page are those of a clone of the repository,
used from its root. After `pip install CrystOD`, run `crystod-example NN` (the
number of the example directory) in your working directory first: it copies
that directory and the shared `test_POSCARs/` into `./example/`, and the
commands then run from there unchanged (see [crystod-example](crystod-example.md)).
```

Plot the first Brillouin zone of a unit cell (black, dotted) together with the
Brillouin zone of a transformed (super)lattice (red) as an interactive 3D HTML file:

```bash
crystod-bz -c example/test_POSCARs/221_PPOSCAR_ScF3 \
    --trans-mat "0 1 2   -1 0 2   1 -1 2" --output BZ_supercell.html
```

```
* Transformation matrix (unit cell -> supercell) *
  [  0.0000   1.0000   2.0000]
  [ -1.0000   0.0000   2.0000]
  [  1.0000  -1.0000   2.0000]
  Volume ratio |det T| = 6

* Supercell lattice (rows) *
  [  0.000000   4.069590   8.139180]
  [ -4.069590   0.000000   8.139180]
  [  4.069590  -4.069590   8.139180]

* Unit-cell q-points folding onto the supercell Gamma point (6) *
  (0, 0, 0)
  (1/3, -1/3, 1/6)
  (-1/3, 1/3, 1/3)
  (0, 0, 1/2)
  (1/3, -1/3, -1/3)
  (-1/3, 1/3, -1/6)

* Output files *
  Wrote supercell Brillouin-zone visualization: BZ_supercell.html
```

```{raw} html
<iframe src="_static/embed/BZ_supercell_221_PPOSCAR_ScF3.html" width="100%" height="560" loading="lazy" style="border:1px solid #8884; border-radius:8px; background:#fff;"></iframe>
<p style="margin-top:0.3em"><a href="_static/embed/BZ_supercell_221_PPOSCAR_ScF3.html" target="_blank">Open the folded-BZ plot full-screen</a> — the small red polyhedra are the supercell BZ tiled at the six folded q points; the black dotted cell is the unit-cell BZ.</p>
```

`--trans-mat` is the row-wise unit-cell-to-supercell transformation matrix
(`L_super = T L_unit`). Its nine entries must be integers, since the supercell
vectors are integer combinations of the unit-cell vectors; a fraction equal to
an integer (`4/2`) is read as that integer. Any other entry stops the command
with an ERROR and no plot:

```bash
crystod-bz -c example/test_POSCARs/221_PPOSCAR_ScF3 --trans-mat "3/2 0 0  0 1 0  0 0 1"
```

```
ERROR: --trans-mat must have integer entries (the supercell vectors are integer combinations of the unit-cell vectors); got 3/2.
```

`--trans-mat` defaults to the identity matrix, which plots the unit-cell BZ
only (section 25); any non-identity matrix switches to this combined
unit-cell + supercell plot.

The supercell BZ is automatically tiled at every supercell reciprocal-lattice
point folded into the unit-cell BZ — exactly the `|det T|` unit-cell q points
that fold onto the Gamma point of the supercell, which are also printed to the
terminal and shown on hover in the plot. This visualizes Brillouin-zone folding
when lowering the symmetry from a supergroup to a subgroup cell (e.g. a Pm-3m
perovskite into a 6x larger cell). Both reciprocal bases are drawn (unit cell:
black dotted, supercell: red/green/blue).

Both zones are built on reduced reciprocal bases (section 25), so a sheared
matrix gives the correct supercell zone, of volume `|det B_super|`
(`B_super`: the supercell reciprocal basis). For example `--trans-mat "1 2 0  0 1 0  0 0 1"`
(`|det T| = 1`) describes the unit-cell lattice itself: one folded point,
(0, 0, 0), and a red zone that coincides with the black one. The folded
q points are brought into the unit-cell zone by the minimum-image rule, which
finds the nearest image also for an oblique unit-cell basis.

**Python API.** `crystod.bz.parse_transformation_matrix(text)` reads a
`--trans-mat` string and `crystod.bz.get_folded_gamma_points(trans_mat, rec_lat)`
returns the `|det T|` folded points in Cartesian coordinates. Both raise
`ValueError` for a matrix with a non-integer entry:

```python
import numpy as np
from crystod import bz

rec_lat = np.linalg.inv(4.07 * np.eye(3)).T
bz.get_folded_gamma_points(np.diag([1.5, 1.0, 1.0]), rec_lat)
# ValueError: the transformation matrix must have integer entries (a supercell of the unit cell).
```

If `--output` is omitted, the plot is saved as `BZ_supercell_{POSCAR name}.html`.
Based on `script/supercell_BZ.py` by Hiroki Koiso.
