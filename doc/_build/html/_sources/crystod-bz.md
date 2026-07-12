# crystod-bz

Interactive 3D Brillouin-zone plots — automatic (seekpath) or manual
high-symmetry k-path, and, with a non-identity `--trans-mat`, the unit-cell BZ
together with the folded BZ of a transformed (super)lattice.

## 14. Brillouin zone plot

*Example directory: `example/14_brillouin_zone` (testsuite section 14)*

Plot the first Brillouin zone as an interactive 3D HTML file, together with the
recommended high-symmetry k-path:

```bash
crystod-bz -c 221_PPOSCAR_ScF3
crystod-bz -c 221_PPOSCAR_ScF3 --output BZ_ScF3_Pm-3m.html
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

## 15. Supercell Brillouin zone (`--trans-mat`)

*Example directory: `example/15_bz_supercell` (testsuite section 15)*

Plot the first Brillouin zone of a unit cell (black, dotted) together with the
Brillouin zone of a transformed (super)lattice (red) as an interactive 3D HTML file:

```bash
crystod-bz -c example/test_POSCARs/221_PPOSCAR_ScF3 \
    --trans-mat "0 1 2   -1 0 2   1 -1 2" --output BZ_supercell.html
```

`--trans-mat` is the row-wise unit-cell-to-supercell transformation matrix
(`L_super = T L_unit`; fractions such as `1/2` are allowed). It defaults to the
identity matrix, which plots the unit-cell BZ only (section 14); any
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
