# crystod-bz

Interactive 3D Brillouin-zone plots — automatic (seekpath) or manual
high-symmetry k-path, and, with a non-identity `--trans-mat`, the unit-cell BZ
together with the folded BZ of a transformed (super)lattice. With
`--show-kpoint --space-group SG` (symbol or number; `--spacegroup`/`--sg`
also accepted), the special k points of any space group are
printed instead (primitive and, for centred lattices, conventional coordinates).

## 18. Brillouin zone plot

*Example directory: `example/18_brillouin_zone` (testsuite section 18)*

Plot the first Brillouin zone as an interactive 3D HTML file, together with the
recommended high-symmetry k-path:

```bash
crystod-bz -c 221_PPOSCAR_ScF3
crystod-bz -c 221_PPOSCAR_ScF3 --output BZ_ScF3_Pm-3m.html
```

```
Space group: Pm-3m (#221)

Recommended k-path (seekpath):
  GAMMA    ( 0.0000,  0.0000,  0.0000)
  X        ( 0.0000,  0.5000,  0.0000)
  M        ( 0.5000,  0.5000,  0.0000)
  R        ( 0.5000,  0.5000,  0.5000)

Path: GAMMA-X-M-GAMMA-R-X   R-M

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
the reciprocal basis of the input structure.

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

The k points and their CDML labels are taken from `irreptables` — the same
definition used by the SALC/irrep analyses (`crystod`, `crystod-mag`,
`crystod-group`, `crystod-phonon --irreps`) — and are given in the primitive
reciprocal basis. For centred lattices (F, I, C, A, B, R), whose conventional
and primitive cells differ, the coordinates in the conventional reciprocal
basis are printed as well:

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

## 19. Supercell Brillouin zone (`--trans-mat`)

*Example directory: `example/19_bz_supercell` (testsuite section 19)*

Plot the first Brillouin zone of a unit cell (black, dotted) together with the
Brillouin zone of a transformed (super)lattice (red) as an interactive 3D HTML file:

```bash
crystod-bz -c example/test_POSCARs/221_PPOSCAR_ScF3 \
    --trans-mat "0 1 2   -1 0 2   1 -1 2" --output BZ_supercell.html
```

```
Transformation matrix (unit cell -> supercell):
  [  0.0000   1.0000   2.0000]
  [ -1.0000   0.0000   2.0000]
  [  1.0000  -1.0000   2.0000]
Volume ratio |det T| = 6

Unit-cell q-points folding onto the supercell Gamma point (6):
  (0, 0, 0)
  (1/3, -1/3, 1/6)
  (-1/3, 1/3, 1/3)
  (0, 0, 1/2)
  (1/3, -1/3, -1/3)
  (-1/3, 1/3, -1/6)

Wrote supercell Brillouin-zone visualization: BZ_supercell_221_PPOSCAR_ScF3.html
```

```{raw} html
<iframe src="_static/embed/BZ_supercell_221_PPOSCAR_ScF3.html" width="100%" height="560" loading="lazy" style="border:1px solid #8884; border-radius:8px; background:#fff;"></iframe>
<p style="margin-top:0.3em"><a href="_static/embed/BZ_supercell_221_PPOSCAR_ScF3.html" target="_blank">Open the folded-BZ plot full-screen</a> — the small red polyhedra are the supercell BZ tiled at the six folded q points; the black dotted cell is the unit-cell BZ.</p>
```

`--trans-mat` is the row-wise unit-cell-to-supercell transformation matrix
(`L_super = T L_unit`; fractions such as `1/2` are allowed). It defaults to the
identity matrix, which plots the unit-cell BZ only (section 18); any
non-identity matrix switches to this combined unit-cell + supercell plot.

The supercell BZ is automatically tiled at every supercell reciprocal-lattice
point folded into the unit-cell BZ — exactly the `|det T|` unit-cell q points
that fold onto the Gamma point of the supercell, which are also printed to the
terminal and shown on hover in the plot. This visualizes Brillouin-zone folding
when lowering the symmetry from a supergroup to a subgroup cell (e.g. a Pm-3m
perovskite into a 6x larger cell). Both reciprocal bases are drawn (unit cell:
black dotted, supercell: red/green/blue).

If `--output` is omitted, the plot is saved as `BZ_supercell_{POSCAR name}.html`.
Based on `script/supercell_BZ.py` by Hiroki Koiso.
