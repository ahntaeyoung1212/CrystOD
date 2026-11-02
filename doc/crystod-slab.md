# crystod-slab

Surface slabs cut from a bulk crystal structure: for each Miller index, every
distinct termination built with pymatgen's `SlabGenerator`, listed with its
number of atoms, thickness, vacuum, area and formula, tested for a dipole
(polar) and for two equivalent faces (symmetric), and written as a VASP POSCAR
file. The Miller indices are given one by one, as a list file, or as every
index up to a maximum that is distinct under the Laue group of the crystal;
with none of them, the command asks for the indices and the main settings
interactively.

| I want to ... | command |
|---|---|
| cut every distinct surface up to (1 1 1), every termination | `crystod-slab -c POSCAR --max-index 1` |
| cut one surface, or a few | `crystod-slab -c POSCAR --index 1 0 0 --index 1 1 0` |
| give a hexagonal surface in four-index notation | `crystod-slab -c POSCAR --index 0 0 0 1` |
| cut the surfaces listed in a file | `crystod-slab -c POSCAR --index-list indices.txt` |
| be asked for the indices and the settings | `crystod-slab -c POSCAR` |
| write the non-polar slabs only | `... --no-polar` |
| get symmetric terminations instead (atoms removed from one face) | `... --symmetrize` |
| leave out the slabs with more than 60 atoms | `... --max-atoms 60` |
| make the slabs thicker and the vacuum wider | `... --slab-size 15 --vacuum-size 20` |
| use the textbook indices of a primitive fcc, bcc, ... POSCAR | `... --conventional` |
| list the slabs and write nothing | `... --no-files` |
| try it without a structure file | `crystod-slab --example SrTiO3` |

## 53. Surface slabs

*Example directory: `example/53_slab` (testsuite section 53); after pip install: `crystod-example 53`*

```{note}
The `example/...` paths on this page are those of a clone of the repository,
used from its root. After `pip install CrystOD`, run `crystod-example NN` (the
number of the example directory) in your working directory first: it copies
that directory and the shared `test_POSCARs/` into `./example/`, and the
commands then run from there unchanged (see [crystod-example](crystod-example.md)).
```

`crystod-slab` began as `script/slabkit.py`, written by Ryuju Kiribayashi
(2024); when you use it, please cite R. Kiribayashi *et al.*, *ACS Appl.
Mater. Interfaces* **17**, 61872 (2025) (see the citation at the end of this
section). The slabs are built by pymatgen's
[`SlabGenerator`](https://pymatgen.org/pymatgen.core.html#pymatgen.core.surface.SlabGenerator),
as in the script, in four steps for each Miller index `(h k l)`:

1. **Oriented unit cell.** A cell of the bulk lattice whose `a` and `b`
   vectors span the `(h k l)` plane and whose `c` vector is a lattice vector
   out of it (with `--max-normal-search N`, the combination of lattice vectors
   up to N that is closest to the surface normal).
2. **Terminations.** The atoms of the oriented cell are grouped into planes
   along `c` (atoms within 0.1 Å of each other along the normal share a
   plane), and a cut half-way between two neighbouring planes is a
   termination. Its position is the **shift**, in fractional `c` of the
   oriented cell. Terminations that give the same slab are merged, so each
   slab listed is distinct.
3. **Slab and vacuum.** Whole oriented cells are stacked along `c`: as many
   as it takes for their height to reach `--slab-size` (the minimum slab
   height, default 10 Å) hold the atoms, and as many as it takes to reach
   `--vacuum-size` (default 15 Å) are left empty. Both minima are thus
   rounded up to whole oriented cells, and the atoms span the stacked height
   less the gap at the cut, so the **thickness** of the table can come out
   below `--slab-size`.
4. **Reduction and centering.** The in-plane cell is reduced to a primitive
   one when pymatgen can (`--no-primitive` keeps that of the oriented cell),
   and the slab is moved so that its centre of mass sits at the middle of the
   cell (`c = 1/2`; `--no-center` keeps it where it was cut). pymatgen keeps
   the in-plane cell of the oriented unit cell when that cell cannot be
   reduced to the same `a` and `b`: the conventional NaCl (1 1 1) slab, for
   example, stays 2 x 2 (56.11 Å², 32 atoms), while the same surface cut from
   the primitive axes is 1 x 1 (14.03 Å², 8 atoms). `primitive in-plane cell:
   yes` in the settings block reports the setting, not the outcome.

Each slab is then measured and tested, and written unless a filter skips it:

```bash
crystod-slab -c example/test_POSCARs/221_PPOSCAR_SrTiO3 --max-index 1
```

```
* Structure *
 221_PPOSCAR_SrTiO3: SrTiO3, Pm-3m (No. 221), 5 atoms
 a, b, c = 3.9451, 3.9451, 3.9451 A; alpha, beta, gamma = 90, 90, 90 deg
 Miller indices refer to the axes of this cell.

* Slab settings *
 minimum slab height: 10 A   minimum vacuum: 15 A   centered: yes   primitive in-plane cell: yes
 polar slabs: written
 oxidation states for the dipole: Sr+2 Ti+4 O-2 (guessed)
 maximum atoms per slab: no limit

* Miller indices *
 indices with |h|,|k|,|l| <= 1, distinct under the Laue group m-3m: (1 1 1) (1 1 0) (1 0 0)

* Slabs *
  #  h k l  shift  atoms  thickness  vacuum   area  formula    polar  symmetric  status
  1  1 1 1  0.250     25      10.25   17.08  26.96  Sr5Ti5O15  yes    no         written
  2  1 1 0  0.250     20       9.76   18.13  22.01  Sr4Ti4O12  yes    no         written
  3  1 0 0  0.250     15       9.86   17.75  15.56  Sr3Ti3O9   no     no         written
 thickness: extent of the atoms along the surface normal (whole oriented cells are stacked to the
 minimum slab height, and the atoms span them less the gap at the cut); vacuum: cell height along
 the normal minus the thickness (both in A); area: in-plane cell area (A^2); shift: pymatgen's
 termination shift (fractional c of the oriented cell); polar: dipole per area from the oxidation
 states (Slab.is_polar; ? = no states); symmetric: an operation of the slab turns the normal over
 (e.g. inversion, a mirror or glide plane parallel to the surface, a 2-fold axis in it), so the two
 faces are equivalent (Slab.is_symmetric).

* Output files *
 #1 -> POSCAR_slab_SrTiO3_111_t1
 #2 -> POSCAR_slab_SrTiO3_110_t1
 #3 -> POSCAR_slab_SrTiO3_100_t1

If you use crystod-slab in your research, please cite:
  R. Kiribayashi, K. Sunada, T. Isobe, K. Kobayashi, T. Nagai, H. Ishiguro, Y. Mochizuki,
  and A. Nakajima, "Preparation and Antiviral Activity of La2CuO4 and Y2Cu2O5 with
  Mechanistic Insights from First Principles", ACS Appl. Mater. Interfaces 17, 61872-61887
  (2025). https://doi.org/10.1021/acsami.5c17978
The slabs are built with pymatgen's SlabGenerator:
  W. Sun and G. Ceder, Surf. Sci. 617, 53 (2013); R. Tran et al., Sci. Data 3, 160080 (2016).

If you use CrystOD in your research, please cite:
  H. Koiso and Y. Mochizuki et al., Phys. Rev. B 110, 064104 (2024).
  https://doi.org/10.1103/PhysRevB.110.064104
```

The `--max-index 1` list holds the indices with `|h|, |k|, |l| <= 1` that are
distinct under the Laue group of the crystal: the rotations of its space
group (found by spglib at `--tolerance`, in the axes of the cell used) and
the inversion that turns a slab over, under which a Miller index `h` goes to
`h R` and `-h R`. Of the 26 indices, cubic symmetry leaves three families,
the first of each in pymatgen's order (low indices first). Each row of the
table is one slab, numbered `t1, t2, ...` within its index in pymatgen's
order (here every index has a single termination: the two cuts of the
(1 0 0) oriented cell, both between an SrO and a TiO<sub>2</sub> plane, give
the same slab, with an SrO face on one side and a TiO<sub>2</sub> face on the
other). The columns:

- **thickness**: the distance between the outermost atoms along the surface
  normal. pymatgen counts the slab in whole oriented cells, here three cells
  of 3.945 Å (11.84 Å) for the minimum slab height of 10 Å, and the atoms
  span five of the six plane spacings of 1.97 Å in them, 9.86 Å; the
  thickness can therefore come out below `--slab-size`.
- **vacuum**: the rest of the cell height along the normal. Four cells
  (15.78 Å) were added for `--vacuum-size 15`, and the cell is 27.62 Å high,
  so 17.75 Å separate the slab from its periodic image.
- **area**: the in-plane area of the slab cell (Å²); **formula**: the
  composition of the slab cell.
- **polar** and **symmetric**: see the next section; **status**: written, or
  why not (`skipped: polar`, `skipped: 25 > 16 atoms`, `not written (--no-files)`).

The files hold the slab in direct coordinates, one block per element in the
order of the input POSCAR, with the index, the termination, the slab formula
and the shift on the comment line:

```
SrTiO3 (1 0 0) slab, termination t1, Sr3Ti3O9, shift 0.250, crystod-slab
1.0
   3.945130000000    0.000000000000    0.000000000000
   0.000000000000    3.945130000000    0.000000000000
   0.000000000000    0.000000000000   27.615910000000
Sr Ti O
3 3 9
direct
   0.000000000000    0.000000000000    0.388233648117 Sr
   0.000000000000    0.000000000000    0.531090790974 Sr
   0.000000000000    0.000000000000    0.673947933831 Sr
   0.500000000000    0.500000000000    0.316805076688 Ti
...
```

The two citation paragraphs end every run that prints the table, and the
`--help` text as well.

### Polar and symmetric slabs

In Tasker's classification of ionic surfaces (P. W. Tasker, *J. Phys. C* **12**,
4977 (1979)), a type I surface is a stack of neutral planes of the bulk
stoichiometry (NaCl (1 0 0) and (1 1 0)), a type II surface a stack of charged
planes whose repeat unit is symmetric and so carries no dipole (the F-Ca-F
unit of fluorite (1 1 1)), and a type III surface a stack of alternately
charged planes whose repeat unit has a dipole along the normal (NaCl (1 1 1),
ZnO (0 0 0 1)). A type III slab carries a dipole that grows with its
thickness, and its electrostatic energy diverges with it, so the surface as
cut is unstable: a real one reconstructs, is compensated or takes a
non-stoichiometric termination.

`crystod-slab` reports two properties of every slab, both as pymatgen defines
them:

- **polar**: the slab, decorated with the oxidation states, has a dipole per
  area along the normal above 10<sup>-3</sup> e/Å (`Slab.is_polar`). The
  states are pymatgen's guess from the bulk composition (`Sr+2 Ti+4 O-2
  (guessed)` in the settings block) or the ones `--oxidation Sr=2 Ti=4 O=-2`
  gives (`(given)`; a fraction such as `Fe=8/3` is read too). Given states
  must balance the bulk (`--oxidation Sr=2 Ti=3 O=-2` stops with `ERROR:
  oxidation states Sr+2 Ti+3 O-2 (--oxidation) leave SrTiO3 charged (-1 e per
  formula unit).`), and states that are all 0 mean no ions. With no guess (a
  metal, an unusual stoichiometry) or all-0 states the polarity is `?`, the
  slabs are kept and a NOTE says so.
- **symmetric**: an operation of the slab turns the surface normal over
  (inversion, a mirror or glide plane parallel to the surface, a two-fold
  axis lying in it), so the two faces are equivalent (`Slab.is_symmetric`,
  at `--tolerance`, default 0.1 Å).

The rock-salt cases come out as in the textbook (the primitive NaCl POSCAR
needs `--conventional`, see the next section):

```bash
crystod-slab -c example/test_POSCARs/225_PPOSCAR_NaCl --conventional --max-index 1 --no-files
```

```
...
* Slabs *
  #  h k l shift atoms thickness vacuum  area formula  polar symmetric status
  1  1 1 1 0.250    32     11.50  18.07 56.11 Na16Cl16 yes   no        not written (--no-files)
  2  1 1 0 0.250    12     10.06  18.11 22.91 Na6Cl6   no    yes       not written (--no-files)
  3  1 0 0 0.250     8      8.54  19.92 16.20 Na4Cl4   no    yes       not written (--no-files)
...
```

The SrTiO<sub>3</sub> (1 1 0) and (1 1 1) slabs above are polar (stacks of
SrTiO/O<sub>2</sub> and SrO<sub>3</sub>/Ti planes); the (1 0 0) slab, a stack
of neutral SrO and TiO<sub>2</sub> planes, is not, but its two faces differ.
`--no-polar` lists the polar slabs and does not write them; `--max-atoms N`
does the same for slabs of more than N atoms, and every slab still appears in
the table with the reason:

```bash
crystod-slab -c example/test_POSCARs/221_PPOSCAR_SrTiO3 --index 1 1 0 --no-polar
```

```
...
* Slabs *
  #  h k l  shift  atoms  thickness  vacuum   area  formula    polar  symmetric  status
  1  1 1 0  0.250     20       9.76   18.13  22.01  Sr4Ti4O12  yes    no         skipped: polar
...

* Output files *
 none (every slab was skipped)
...
```

`--symmetrize` asks for symmetric slabs instead (pymatgen's
`SlabGenerator.get_slabs(symmetrize=True)`, with the two faces compared at
`--tolerance` as in the symmetric column): atoms are removed from one face of
every asymmetric slab until the two faces are equivalent, so the
stoichiometry changes, and a slab with two equivalent faces has no dipole.
On (1 0 0) of SrTiO<sub>3</sub> the two results are the TiO<sub>2</sub>- and
the SrO-terminated slabs:

```bash
crystod-slab -c example/test_POSCARs/221_PPOSCAR_SrTiO3 --index 1 0 0 --symmetrize
```

```
...
* Slab settings *
 minimum slab height: 10 A   minimum vacuum: 15 A   centered: yes   primitive in-plane cell: yes
 polar slabs: symmetrized terminations (atoms removed from one face of every asymmetric slab)
 oxidation states for the dipole: Sr+2 Ti+4 O-2 (guessed)
 maximum atoms per slab: no limit

* Miller indices *
 indices given: (1 0 0)

* Slabs *
  #  h k l  shift  atoms  thickness (A)  vacuum (A)  area (A^2)  formula   polar  symmetric  status
  1  1 0 0  0.250     13           7.89       19.73       15.56  Sr2Ti3O8  no     yes        written
  2  1 0 0  0.250     12           7.89       19.73       15.56  Sr3Ti2O7  no     yes        written
...

* Output files *
 #1 -> POSCAR_slab_SrTiO3_100_s1
 #2 -> POSCAR_slab_SrTiO3_100_s2
...
```

The symmetrized slabs are numbered `s1, s2, ...` (files `..._s1`, `..._s2`),
so they never take the names of the slabs of a plain run; those of one
termination share its shift, and the formula tells them apart. "Not polar"
means no dipole about the slab centre, not charge neutrality: the two
(1 0 0) slabs above are neutral, but the symmetrized (1 1 0) slabs of
SrTiO<sub>3</sub>, Sr<sub>3</sub>Ti<sub>3</sub>O<sub>11</sub> and
Sr<sub>4</sub>Ti<sub>4</sub>O<sub>10</sub>, and the (1 1 1) slabs of NaCl,
Na<sub>12</sub>Cl<sub>16</sub> and Na<sub>16</sub>Cl<sub>12</sub>, carry
formal charges of -4 and +4 per cell: the Tasker type III dipole is traded
for a charged, non-stoichiometric slab. A run with a charged slab adds a
**charge** column (the net formal charge per cell from the oxidation
states) and a NOTE, as in the dialogue run below; the bundled `crystod-slab
--example ScF3` gives Sc<sub>2</sub>F<sub>7</sub> (-1) and
Sc<sub>3</sub>F<sub>8</sub> (+1). `--symmetrize` implies `--no-polar`. An
index for which no symmetric termination is found adds no row and prints a
NOTE instead; when no operation of the bulk turns the normal over, as for
the polar (0 0 0 1) of wurtzite ZnO (point group 6mm), it says that none may
exist:

```
 NOTE: no symmetric termination found for (0 0 1) (removing atoms from one face did not make the
       faces equivalent before the slab had no more atoms than the bulk cell; none may exist: no
       operation of the bulk turns this normal over, so a thicker slab will not help)
```

### The cell and the Miller indices

The Miller indices refer to the axes of the cell in the POSCAR, as given. For
a primitive cell of a centred lattice these are not the textbook indices,
and the `* Structure *` block says so:

```bash
crystod-slab -c example/test_POSCARs/225_PPOSCAR_NaCl --max-index 1 --no-files
```

```
* Structure *
 225_PPOSCAR_NaCl: NaCl, Fm-3m (No. 225), 2 atoms
 a, b, c = 4.0246, 4.0246, 4.0246 A; alpha, beta, gamma = 60, 60, 60 deg
 Miller indices refer to the axes of this cell.
 NOTE: the cell is not the conventional standard cell (8 atoms); textbook Miller indices need
       --conventional.
...
* Miller indices *
 indices with |h|,|k|,|l| <= 1, distinct under the Laue group m-3m: (1 1 1) (1 1 0) (1 1 -1)
   (1 0 -1)
...
```

`--conventional` first standardizes the structure to the conventional
standard cell (pymatgen `get_conventional_standard_structure`, at
`--tolerance`), and the indices then refer to that cell: the run of the
previous section prints `standardized conventional cell: 8 atoms (input: 2)`
and the familiar (1 1 1), (1 1 0), (1 0 0). The slab files keep the element
order of the input POSCAR either way.

The families of `--max-index` are found in the axes of the cell used, so
they hold for any setting. On the primitive rhombohedral cell of R-3c
AlF<sub>3</sub> the 3-fold axis permutes `h`, `k` and `l`, and the five
families are

```
 indices with |h|,|k|,|l| <= 1, distinct under the Laue group -3m: (1 1 1) (1 1 0) (1 1 -1) (1 0 0)
   (1 0 -1)
```

pymatgen's own `get_symmetrically_distinct_miller_indices` merges indices
under the symmetry of the lattice and, for a trigonal crystal, of its
hexagonal setting; on this cell it lists (1 1 0) and (1 0 1), which are the
same surface, and leaves out (1 1 -1), and on a cell whose lattice has more
symmetry than the crystal (a P1 cell, pyrite in Pa-3, a pseudo-cubic metric)
it leaves out surfaces. `crystod-slab` uses the crystal's own rotations
instead; for a crystal that is not trigonal and whose lattice has no more
symmetry than the crystal, such as cubic SrTiO<sub>3</sub>, the two lists
agree.

On hexagonal axes an index can be given in the four-index Miller-Bravais form
`h k i l`; `i = -(h + k)` is checked and dropped, as `crystod-xrd` drops it
from its indices:

```bash
crystod-slab -c example/test_POSCARs/186_PPOSCAR_ZnO --index 1 0 -1 0 --index 0 0 0 1 --no-files
```

```
...
* Miller indices *
 indices given: (1 0 -1 0) = (1 0 0) (0 0 0 1) = (0 0 1)
 (h k i l) is read as (h k l): the redundant i = -(h + k) is dropped

* Slabs *
  #  h k l shift atoms thickness vacuum  area formula polar symmetric status
  1  1 0 0 0.000    16      9.49  18.99 17.45 Zn8O8   no    yes       not written (--no-files)
  2  1 0 0 0.500    16     10.44  18.04 17.45 Zn8O8   no    yes       not written (--no-files)
  3  0 0 1 0.060     8      9.97  16.56  9.37 Zn4O4   yes   no        not written (--no-files)
  4  0 0 1 0.310     8      8.60  17.93  9.37 Zn4O4   yes   no        not written (--no-files)
...
```

The non-polar m-plane (1 0 -1 0) of wurtzite ZnO has two terminations, and
the polar c-plane (0 0 0 1) two as well. An index whose `i` is not `-(h + k)`
stops the run:

```
ERROR: (1 0 0 1) is not a Miller-Bravais index (h k i l): i = 0, but -(h + k) = -1.
```

An index with a common factor is the plane family of the reduced one
(`--index 2 0 0` cuts (1 0 0) and prints `NOTE: (2 0 0) is the plane family
(1 0 0)`), and an index given twice is listed once. `--index-list FILE` reads
one index per line, `1 0 0`, `1,0,0` or `1 0 -1 0`, with `#` comments and
blank lines skipped (a four-index line is shown as `(1 0 -1 0) = (1 0 0)`,
as with `--index`):

```
# low-index faces of SrTiO3
1 0 0
1,1,0

1 1 1   # the polar one
```

A line that is not an index stops the run with its line number
(`ERROR: indices.txt, line 2: Miller indices are integers; got 'abc' in 'abc'.`).
Exactly one of `--index`, `--index-list` and `--max-index` may be given.

### The interactive dialogue

With no index source on the command line, `crystod-slab -c POSCAR` asks the
questions of the original slabkit.py (the structure file only when it is
missing; the slab height before the vacuum); Enter keeps the default in
brackets:

```bash
crystod-slab -c example/test_POSCARs/221_PPOSCAR_SrTiO3
```

```
* Miller indices *
 How should the Miller indices be given?
  1) one index (h k l)
  2) a list file (one index per line)
  3) every symmetrically distinct index up to a maximum
  4) quit
 Choice [1-4]: 3
 Maximum Miller index [1]: 2
 Minimum slab height in Angstrom [10]: 
 Minimum vacuum height in Angstrom [15]: 
 Maximum number of atoms per slab [no limit]: 60
 Allow polar slabs? yes/no [yes]: no
 Symmetrize the slabs instead (atoms removed from one face)? yes/no [no]: yes

* Equivalent command line *
 crystod-slab -c example/test_POSCARs/221_PPOSCAR_SrTiO3 --max-index 2 --max-atoms 60 --symmetrize

* Structure *
 221_PPOSCAR_SrTiO3: SrTiO3, Pm-3m (No. 221), 5 atoms
...
* Slabs *
  #  h k l  shift  atoms  thickness  vacuum   area  formula    charge  polar  symmetric  status
  1  1 1 1  0.250     21       9.11   18.22  26.96  Sr4Ti5O12      +4  no     yes        written
  2  1 1 1  0.250     24       9.11   18.22  26.96  Sr5Ti4O15      -4  no     yes        written
  3  1 1 0  0.250     17       8.37   19.53  22.01  Sr3Ti3O11      -4  no     yes        written
  4  1 1 0  0.250     18       8.37   19.53  22.01  Sr4Ti4O10      +4  no     yes        written
  5  1 0 0  0.250     13       7.89   19.73  15.56  Sr2Ti3O8        0  no     yes        written
  6  1 0 0  0.250     12       7.89   19.73  15.56  Sr3Ti2O7        0  no     yes        written
  7  2 2 1  0.125     38       9.21   17.10  46.69  Sr7Ti8O23       0  no     yes        written
  8  2 2 1  0.125     37       9.21   17.10  46.69  Sr8Ti7O22       0  no     yes        written
  9  2 1 1  0.125     37      11.27   17.72  38.12  Sr7Ti7O23      -4  no     yes        written
 10  2 1 1  0.125     38      11.27   17.72  38.12  Sr8Ti8O22      +4  no     yes        written
 11  2 1 0  0.125     28       8.82   19.41  34.80  Sr5Ti6O17       0  no     yes        written
 12  2 1 0  0.125     27       8.82   19.41  34.80  Sr6Ti5O16       0  no     yes        written
...
 NOTE: #1-#4, #9, #10: non-stoichiometric, with a net formal charge (charge column); 'polar: no'
       means no dipole about the slab centre, not a neutral slab.
...
```

The `* Equivalent command line *` block repeats the run as an ordinary command
for the next time (or for a script); from there on the output is that of the
command. Choice 1 asks for one index (`h k l`, or `h k i l` on hexagonal
axes; `(1 1 0)` and `1,1,0` are read too, and the command line gets
`--index 1 1 0`), choice 2 for the name of a list file, and choice 4 prints
` quit: nothing was done.` and exits with status 0. The symmetrize question
comes only after `no` to the polar one. Options given on the command line are
kept: their values become the defaults of the dialogue (`--max-atoms 16`
shows `[16]`) and the ones it does not ask about (`--no-files`,
`--conventional`, `--output-dir`, ...) are carried into the equivalent
command line, the sizes exactly as given (`--slab-size 10.0000004`). Without
`-c` the structure is `./POSCAR`, and when there is no such file the dialogue
first asks `Structure file [POSCAR]:` and asks again for a file it cannot
read. The structure given with `-c` and the `--oxidation` states are checked
before the first question. An invalid answer is asked again, up to three
times; an empty answer to a question without a default, the end of the
input, or Ctrl-C stops with an ERROR line.

### Outputs and other options

| option | effect |
|---|---|
| `--slab-size T` | minimum slab height (default 10 Å, pymatgen `min_slab_size`): whole oriented cells are stacked until they reach it, and the atoms span that height less the gap at the cut |
| `--vacuum-size V` | minimum vacuum height (default 15 Å), likewise in whole oriented cells |
| `--in-unit-planes` | `--slab-size` and `--vacuum-size` count `d_hkl` lattice planes of the cell instead of Å (one plane can hold several atomic layers, e.g. SrO + TiO<sub>2</sub> for SrTiO<sub>3</sub> (1 0 0); still rounded up to whole oriented cells) |
| `--max-atoms N` | slabs of more than N atoms are listed as `skipped: 25 > 16 atoms` and not written (default: no limit) |
| `--no-polar` | polar slabs are listed as `skipped: polar` and not written |
| `--symmetrize` | symmetric terminations (atoms removed from one face, the faces compared at `--tolerance`; files `..._s1`, ...; implies `--no-polar`) |
| `--oxidation EL=Q ...` | the oxidation states of the polarity test, one per element and balancing the bulk (`--oxidation Sr=2 Ti=4 O=-2`); default: pymatgen's guess |
| `--conventional` | standardize to the conventional standard cell first; the indices refer to that cell |
| `--no-center` | do not move the slab to the middle of the cell |
| `--no-primitive` | keep the in-plane cell of the oriented unit cell instead of reducing it to a primitive one |
| `--lll-reduce` | reduce the in-plane cell to its shortest `a`, `b` and make `c` as perpendicular to the surface as whole in-plane vectors allow; `a` and `b` stay in the surface (pymatgen's own `lll_reduce` LLL-reduces the whole slab cell and can tilt it out of the surface: on SrTiO<sub>3</sub> (3 3 2) with `--slab-size 5 --vacuum-size 5` its area is 62.26 Å² against the 73.00 Å² of the surface cell) |
| `--orthogonal-c` | make `c` perpendicular to the surface (`Slab.get_orthogonal_c_slab`) before the measurements and the files |
| `--max-normal-search N` | search the combinations of lattice vectors up to N for a `c` as perpendicular to the surface as possible (pymatgen) |
| `--tolerance T` | symmetry tolerance in Å of the printed space group, the Laue group of `--max-index`, `--conventional`, the symmetric-slab test and `--symmetrize` (default 0.1) |
| `--output-dir DIR` | directory of the POSCAR files, created if missing (default: the current directory); checked before the run |
| `--force` | replace a `POSCAR_slab_*` file of different content (without it the run stops before writing anything) |
| `--no-files` | list the slabs only; write nothing (`none (--no-files)` under `* Output files *`) |

The `* Slab settings *` block of every run shows the two minimum heights (in
lattice planes with `--in-unit-planes`: `minimum slab height: 4 d_hkl lattice
planes`), the centering and the primitive reduction, and adds a line for
`--lll-reduce` (`reduced in-plane cell (--lll-reduce): yes`), `--orthogonal-c`
(`c perpendicular to the surface: yes`) and `--max-normal-search` (`max
normal search: 1`). The files are named `POSCAR_slab_{formula}_{h}{k}{l}_t{n}`,
with the reduced formula of the bulk in the element order of the input
(never grouped: `Ba2YCu3O6`, not pymatgen's `Ba2Y(CuO2)3`), a negative index
written as `-1` (`POSCAR_slab_NaCl_10-1_t1`), the components separated by `_`
when one of them is 10 or more (`POSCAR_slab_SrTiO3_1_1_10_t1`), and `s{n}`
instead of `t{n}` for the slabs of `--symmetrize`. As `crystod-search` does,
an identical file already there is kept (`#1 -> POSCAR_slab_SrTiO3_111_t1
(kept: identical file already there)`), and a different one stops the run
before any file is written (`ERROR: POSCAR_slab_SrTiO3_100_t1 already exists
with different content (--force replaces it).`); `--force` replaces it
(`(replaced)`). The slabs of pymatgen `generate_all_slabs` with the same
sizes, options and indices are the same slabs.

Bad input stops with one line, `ERROR: ...`, and exit status 1: two index
sources (`ERROR: give one index source, --index, --index-list or --max-index
(got --index and --max-index).`), `--max-index 0`, a slab or vacuum size of
0 or less, the zero index (`ERROR: (0 0 0) is not a plane: give at least one
nonzero index.`), two indices after one `--index` (`ERROR: --index takes one
Miller index; got '1 0 0 1 1 0' (for several: --index 1 0 0 --index 1 1 0).`),
a missing structure or list file or a directory in its place, an incomplete,
unbalanced or misspelled `--oxidation` (`ERROR: no oxidation state for Ti, O
(--oxidation needs one per element: Sr=Q Ti=Q O=Q).`), an `--output-dir` that
cannot be created (`ERROR: --output-dir afile/sub: afile is not a
directory.`), and a structure with partially occupied or mixed sites. A run
in which every slab is skipped still prints the table and exits with 0.

### From Python

```python
from crystod import slab

structure = slab.load_structure("example/test_POSCARs/221_PPOSCAR_SrTiO3")
print(slab.distinct_miller_indices(structure, 1))
records = slab.generate_slabs(structure, max_index=1)
print(records.oxidation_states, records.oxidation_source)
for record in records:
    print(record.label, record.n_atoms, record.formula, record.polar, record.symmetric,
          f"{record.thickness:.2f} {record.vacuum:.2f} {record.area:.2f}")

symmetric = slab.generate_slabs(structure, (1, 1, 0), symmetrize=True)
for record in symmetric:
    print(record.label, record.formula, record.charge, record.polar, record.symmetric,
          slab.slab_filename(record))
print(slab.write_slab_poscar(symmetric[0], directory="slabs"))
print(slab.write_slab_poscar(symmetric[0], directory="slabs"))

print(slab.format_table(records, max_atoms=16).splitlines()[1])
print(slab.parse_miller_index("0 0 0 1"), slab.parse_miller_index("1,1,0"))
try:
    slab.parse_miller_index("1 0 0 1")
except ValueError as exc:
    print(exc)
```

```
[(1, 1, 1), (1, 1, 0), (1, 0, 0)]
{'Sr': 2.0, 'Ti': 4.0, 'O': -2.0} guessed
(1 1 1) t1 25 Sr5Ti5O15 True False 10.25 17.08 26.96
(1 1 0) t1 20 Sr4Ti4O12 True False 9.76 18.13 22.01
(1 0 0) t1 15 Sr3Ti3O9 False False 9.86 17.75 15.56
(1 1 0) s1 Sr3Ti3O11 -4.0 False True POSCAR_slab_SrTiO3_110_s1
(1 1 0) s2 Sr4Ti4O10 4.0 False True POSCAR_slab_SrTiO3_110_s2
('slabs/POSCAR_slab_SrTiO3_110_s1', 'wrote')
('slabs/POSCAR_slab_SrTiO3_110_s1', 'kept')
  1  1 1 1 0.250    25     10.25  17.08 26.96 Sr5Ti5O15 yes   no        skipped: 25 > 16 atoms
(0, 0, 1) (1, 1, 0)
(1 0 0 1) is not a Miller-Bravais index (h k i l): i = 0, but -(h + k) = -1.
```

`generate_slabs` takes one index, a list of indices or `max_index`, and the
options of the command as keywords (`slab_size`, `vacuum_size`,
`symmetrize`, `oxidation_states`, `tolerance`, `center`, `primitive`,
`lll_reduce`, `in_unit_planes`, `max_normal_search`, `orthogonal_c`). It
returns a `SlabSet`, a list of `SlabRecord`s that also carries the indices,
the oxidation states and their source, and the NOTE lines (`notes`); each
record holds the pymatgen `Slab` (`record.slab`), the termination number and
shift, the measurements of the table (`polar` is `None` without oxidation
states) and the net formal charge (`charge`, `charged`). `load_structure(path,
conventional=True)` is `--conventional`. `format_table` returns the table as
one string (`crystod.search.format_table` returns a list of lines), and
`write_slab_poscar` returns `(path, status)` with the status `wrote`, `kept`
or `replaced` as `crystod.search.write_poscar` does (`overwrite=True` is
`--force`). Importing `crystod.slab` does not import pymatgen, and bad input
raises `ValueError` with the text of the command's ERROR. The full list of
functions is on the [crystod.slab](api/slab.md) page of the API reference.

```{admonition} Citation
If you use `crystod-slab` in your research, please cite R. Kiribayashi,
K. Sunada, T. Isobe, K. Kobayashi, T. Nagai, H. Ishiguro, Y. Mochizuki, and
A. Nakajima, "Preparation and Antiviral Activity of La<sub>2</sub>CuO<sub>4</sub>
and Y<sub>2</sub>Cu<sub>2</sub>O<sub>5</sub> with Mechanistic Insights from First
Principles", [*ACS Appl. Mater. Interfaces* **17**, 61872–61887 (2025)](https://doi.org/10.1021/acsami.5c17978).

The slabs are built with pymatgen's `SlabGenerator`: W. Sun and G. Ceder,
"Efficient creation and convergence of surface slabs",
[*Surf. Sci.* **617**, 53–59 (2013)](https://doi.org/10.1016/j.susc.2013.05.016);
R. Tran *et al.*, "Surface energies of elemental crystals",
[*Sci. Data* **3**, 160080 (2016)](https://doi.org/10.1038/sdata.2016.80).
```
