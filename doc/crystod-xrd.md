# crystod-xrd

Powder X-ray and neutron diffraction patterns: the Bragg peaks of the
structure in a POSCAR, with their Miller indices, multiplicities, `d`
spacings, `2θ` positions and relative intensities, written as a text table and
drawn as a broadened pattern; and, for a distorted structure, its reflections
indexed on the parent lattice with the superlattice reflections assigned to
parent k points and irreps.

| I want to ... | command |
|---|---|
| compute the pattern for Cu Kα (Kα1 + Kα2) | `crystod-xrd -c POSCAR` |
| use a single, monochromatic line | `crystod-xrd -c POSCAR --xraytype CuKa1` |
| use another anode | `crystod-xrd -c POSCAR --xraytype MoKa` |
| draw Gaussian instead of Lorentzian peaks | `... --peak-profile gaussian` |
| restrict the angular window | `... --two-theta 5 60` |
| compute the neutron pattern (nuclear scattering) | `crystod-xrd -c POSCAR --radiation neutron --wavelength 1.5` |
| index the superlattice reflections of a distorted structure on its parent, with k points and irreps | `crystod-xrd --parent PARENT.cif -c CHILD.cif` |
| try it without a structure file | `crystod-xrd --example ScF3` |

## 46. Powder X-ray diffraction patterns

*Example directory: `example/46_xrd_pattern` (testsuite section 46); after pip install: `crystod-example 46`*

```{note}
The `example/...` paths on this page are those of a clone of the repository,
used from its root. After `pip install CrystOD`, run `crystod-example NN` (the
number of the example directory) in your working directory first: it copies
that directory and the shared `test_POSCARs/` into `./example/`, and the
commands then run from there unchanged (see [crystod-example](crystod-example.md)).
```

The intensities are computed with the
[`XRDCalculator`](https://pymatgen.org/pymatgen.analysis.diffraction.html) of
pymatgen: atomic scattering factors, the Lorentz-polarization factor and the
multiplicity of each family of planes, for the structure exactly as given in
the POSCAR (no symmetrization, no thermal factors).

```bash
crystod-xrd -c example/test_POSCARs/221_PPOSCAR_ScF3
```

```
* Structure *
 221_PPOSCAR_ScF3: ScF3, Pm-3m

* Radiation *
 CuKa: CuKa1 = 1.5405929 A, CuKa2 = 1.5444274 A  (Ka1 : Ka2 = 2 : 1)

* Bragg peaks (36) in 10 - 120 deg *
    h   k   l  mult       d (A)  2theta (deg)  intensity  line
    1   0   0     6    4.069590       21.8217    100.000  CuKa1
    1   0   0     6    4.069590       21.8766     50.000  CuKa2
    1   1   0    12    2.877635       31.0530     11.215  CuKa1
    1   1   0    12    2.877635       31.1322      5.606  CuKa2
    ...
    3   0   0     6    1.356530       69.1999     13.937  CuKa1   + (2 2 1) x24
    3   0   0     6    1.356530       69.3968      6.964  CuKa2   + (2 2 1) x24
    ...

* Output files *
 Peak table written to: XRD_221_PPOSCAR_ScF3_CuKa.txt
 Pattern (lorentzian profile, width 0.1 deg) written to: XRD_221_PPOSCAR_ScF3_CuKa.pdf
```

Families of planes with the same `d` spacing, such as the cubic `(3 0 0)` and
`(2 2 1)`, contribute to one peak; the first family is listed in the columns
and the others after the line name.

```{figure} images/xrd_ScF3_CuKa.png
:width: 90%

The pattern of ScF<sub>3</sub> for the Cu Kα doublet with Lorentzian peaks
(the default). The tick marks under the curve are the Bragg positions; the
Kα1/Kα2 splitting grows with the angle and is resolved above about 60°.
```

### The radiation (`--xraytype`)

A **doublet** (`CuKa`, the default, and `AgKa`, `MoKa`, `CoKa`, `FeKa`,
`CrKa`) superposes the Kα1 and Kα2 patterns with the 2:1 intensity ratio of
the two lines. This is what a laboratory diffractometer without a Kα2
monochromator records: every reflection appears twice, the Kα2 partner at half
the intensity and slightly higher angle. A **single line** gives the
monochromatic pattern. The names are case-insensitive.

| name | wavelength (Å) | name | wavelength (Å) |
|---|---|---|---|
| `CuKa1` | 1.5405929 | `CoKa1` | 1.788996 |
| `CuKa2` | 1.5444274 | `CoKa2` | 1.792835 |
| `CuKb` | 1.392234 | `FeKa1` | 1.936041 |
| `MoKa1` | 0.70931715 | `FeKa2` | 1.939973 |
| `MoKa2` | 0.713607 | `CrKa1` | 2.289726 |
| `AgKa1` | 0.55942178 | `CrKa2` | 2.293651 |
| `AgKa2` | 0.5638131 | | |

The wavelengths are those of the RIETAN-FP manual (F. Izumi and K. Momma).

### The drawn pattern (`--peak-profile`, `--width`)

Every peak is broadened into a profile of unit area scaled by its intensity,
so the integrated intensity of a peak does not depend on the profile:

- `lorentzian` (default): `(w/π) / ((2θ − 2θ₀)² + w²)`, `w` the half width at
  half maximum;
- `gaussian`: `exp(−(2θ − 2θ₀)² / 2w²) / (w √(2π))`, `w` the standard deviation.

`--width` sets `w` in degrees (default 0.1). The profile only shapes the
figure; the peak table is the same for both.

### Outputs and other options

| option | effect |
|---|---|
| `--two-theta MIN MAX` | the window of the peak list and the plot (default 10 120) |
| `--min-intensity PERCENT` | drop reflections weaker than this (default 0: every reflection is kept, including the very weak ones pymatgen would hide below 0.001 % of the strongest) |
| `-o PREFIX` | write `PREFIX.txt` and `PREFIX.pdf` (default `XRD_{cell file}_{xraytype}`) |
| `--show` | also open the pattern in a matplotlib window |
| `--tolerance` | symmetry tolerance of the printed space-group symbol (default 0.01 Å) |

The text table is comma-separated, one reflection per line:
`h, k, l, multiplicity, d, two_theta, intensity, line, families`, with the
structure, the radiation and the wavelengths in the header.

The Miller indices always refer to the cell of the POSCAR and are always
three. On a hexagonal cell (hexagonal and trigonal crystals, rhombohedral ones
in the hexagonal setting) the redundant Miller-Bravais index `i = -(h + k)` is
left out: the `(1 0 -1 1)` reflection of wurtzite ZnO is listed as `(1 0 1)`.

### From Python

```python
from crystod import xrd

structure = xrd.load_structure("POSCAR")
pattern = xrd.compute_xrd_pattern(structure, "CuKa", (10, 120))
for peak in pattern.peaks[:4]:
    print(peak.hkl, round(peak.two_theta, 3), round(peak.intensity, 2), peak.line)
two_theta, intensity = xrd.smear_pattern(pattern, "lorentzian", width=0.1)
```

The full list of functions is on the [crystod.xrd](api/xrd.md) page of the API
reference.

```{admonition} Citation
The intensities come from pymatgen: S. P. Ong *et al.*, "Python Materials
Genomics (pymatgen): A robust, open-source python library for materials
analysis", *Comput. Mater. Sci.* **68**, 314-319 (2013).
```

## 47. Superlattice reflections and neutron patterns (`--parent`, `--radiation neutron`)

*Example directory: `example/47_xrd_superlattice` (testsuite section 47); after pip install: `crystod-example 47`*

`--parent PARENT -c CHILD` answers the question a powder pattern of a
distorted phase raises: which of its reflections are new, at which k point of
the parent Brillouin zone they sit, and which distortion (parent irrep) makes
them appear. The two structures (CIF or POSCAR, ordered: a partially
occupied or mixed site is refused with an ERROR) are paired by the
symmetry-mode analysis of
[`crystod-group --supergroup-cif`](crystod-group.md#23-symmetry-mode-analysis---supergroup-cif),
which also supplies the k stars that fold onto the child cell and the
displacement field of every irrep. Every reflection family of the child is
indexed on the parent lattice: integer parent indices make a **fundamental**
reflection (printed with the child intensity and that of the parent reference
structure on the same scale), fractional ones a **superlattice** reflection,
whose fractional part is the parent k point. The intensity of a superlattice
reflection is split among the irreps of its k star by evaluating the
structure factor of the parent structure plus each irrep's displacements
alone; a reflection carried by one irrep (share of at least 95 %) is
**diagnostic** for it, the one to look for in a measured pattern.

The structure pairs below are CIF files of `example/23_symmetry_mode`
(`crystod-example 47` copies that directory along with its own):

```bash
crystod-xrd --parent example/23_symmetry_mode/221_PPOSCAR_SrTiO3.cif \
            -c example/23_symmetry_mode/140_PPOSCAR_SrTiO3.cif
```

```
* Structures *
parent: 221_PPOSCAR_SrTiO3.cif  SrTiO3, Pm-3m (No. 221)
child : 140_PPOSCAR_SrTiO3.cif  SrTiO3, I4/mcm (No. 140)
child primitive basis in parent primitive units (rows): (0, 1, 1), (0, -1, 1), (1, 0, -1)
primitive cell multiplication: 2
parent indices: the cell of Pm-3m (primitive = conventional)
child indices: conventional cell of I4/mcm

* Radiation *
CuKa: CuKa1 = 1.5405929 A (no Debye-Waller factor)
the table lists the CuKa1 positions; the drawn pattern adds CuKa2 at its weight

* Reflections (60 families, 17 superlattice) in 10 - 120 deg *
   2theta     d (A)  (h k l)_child    (h k l)_parent k      I_child  I_parent  contributions
  22.4704   3.95355  (0 0 2) x2       (0 0 1)        GM       1.953     1.953  -
  22.5663   3.93696  (1 1 0) x4       (1 0 0)        GM       4.107     3.874  -
  32.0578   2.78970  (1 1 2) x8       (1 0 1)        GM     100.000   100.995  -
  32.1269   2.78385  (0 2 0) x4       (1 1 0)        GM      51.205    50.208  -
  37.8510   2.37498  (1 2 1) x16      (3 1 1)/2      R        0.640         -  R5- 100 %
  39.5608   2.27618  (0 2 2) x8       (1 1 1)        GM      31.302    32.089  -
  ...
(I_child: strongest child family = 100; I_parent: the parent reference structure on the same scale;
 contributions: share of |F|^2 among the irreps of the k star, each with its own displacements alone)

* Superlattice reflections by k point *
R (1/2,1/2,1/2): 17 families (+ 11 absent in the child); irreps of the star: R5- (0.3303 A)
  all families from R5- (the only irrep of the star)
  R5- extinct at first order for |h| = |k|: 11 families absent in the child
(diagnostic: share >= 95%; extinct at first order: the term of the irrep's structure factor
 linear in its displacements vanishes; positions in degrees 2theta)

* Output files *
  Reflection table written to: XRD_140_PPOSCAR_SrTiO3_vs_221_PPOSCAR_SrTiO3_CuKa.txt
  Patterns (lorentzian profile, width 0.1 deg) written to: XRD_140_PPOSCAR_SrTiO3_vs_221_PPOSCAR_SrTiO3_CuKa.pdf
```

The antiphase octahedral tilt of I4/mcm SrTiO<sub>3</sub> doubles the cell
along all three cubic axes, and every superlattice reflection has half-odd
parent indices, `(o o o)/2`, at the R point. The irrep is R5- because the
parent CIF puts Sr at the origin; in the Ti-origin setting of the same
structure (as in the CaTiO<sub>3</sub> files below) the same tilt is R4+.
The label is the one `--supergroup-cif` prints for the pair.

A structure with several order parameters, orthorhombic Pnma CaTiO<sub>3</sub>
(tilt system a<sup>-</sup>a<sup>-</sup>c<sup>+</sup>):

```bash
crystod-xrd --parent example/23_symmetry_mode/221_PPOSCAR_CaTiO3.cif \
            -c example/23_symmetry_mode/62_PPOSCAR_CaTiO3.cif --xraytype CuKa1
```

```
...
* Reflections (186 families, 122 superlattice) in 10 - 120 deg *
   2theta     d (A)  (h k l)_child    (h k l)_parent k      I_child  I_parent  contributions
  20.0533   4.42429  (0 1 1) x4       (-1 1 1)/2     R        0.286         -  R5+ 99 %, R4+ 1 %
  23.0306   3.85862  (1 0 1) x4       (0 0 1)        GM      10.121    13.185  -
  23.1012   3.84700  (0 2 0) x2       (0 1 0)        GM       4.171     6.539  -
  25.8092   3.44917  (1 1 1) x8       (0 1 2)/2      X        4.573         -  X5+ 100 %
  32.4884   2.75370  (2 0 0) x2       (1 0 1)        GM      25.073    26.746  -
  32.8483   2.72434  (1 2 1) x8       (0 1 1)        GM     100.000   103.757  -
  33.1038   2.70390  (0 0 2) x2       (-1 0 1)       GM      23.024    25.385  -
  34.5679   2.59265  (2 1 0) x4       (2 1 2)/2      X        1.525         -  X5+ 100 %
  36.5900   2.45388  (2 0 1) x4       (1 0 3)/2      M        1.082         -  M3+ 100 %
  37.0074   2.42716  (1 0 2) x4       (-1 0 3)/2     M        1.785         -  M3+ 100 %
  38.4755   2.33786  (2 1 1) x8       (1 1 3)/2      R        1.922         -  R4+ 98 %, R5+ 2 %
  38.8308   2.31728  (0 3 1) x4       (-1 3 1)/2     R        4.492         -  R4+ 100 %
  38.8755   2.31471  (1 1 2) x8       (-1 1 3)/2     R        4.365         -  R4+ 91 %, R5+ 9 %
  ...

* Superlattice reflections by k point *
X (0,1/2,0): 41 families (+ 12 absent in the child); irreps of the star: X5+ (0.5196 A)
  all families from X5+ (the only irrep of the star)
  X5+ extinct at first order for h = -l: 12 families absent in the child
M (1/2,1/2,0): 40 families (+ 24 absent in the child); irreps of the star: M2+ (0.0064 A), M3+ (0.8874 A)
  diagnostic for M2+: none
  diagnostic for M3+: (1 0 3)/2 at 36.59, (-1 0 3)/2 at 37.01, (1 2 3)/2 at 43.72, (-1 2 3)/2 at 44.08, (3 0 5)/2 at 70.52, (-3 0 5)/2 at 71.84, (-3 2 5)/2 at 76.62, (1 2 7)/2 at 94.09, and 32 more (see the table)
  M2+ extinct at first order for |h| = |l|: 24 families absent in the child
  M3+ extinct at first order for |h| = |l|: 24 families absent in the child
R (1/2,1/2,1/2): 41 families (+ 11 absent in the child); irreps of the star: R4+ (1.2458 A), R5+ (0.1047 A)
  diagnostic for R4+: (1 1 3)/2 at 38.48, (-1 3 1)/2 at 38.83, (-1 5 1)/2 at 62.73, (1 5 3)/2 at 72.44, (3 3 5)/2 at 81.20, (-3 5 3)/2 at 82.32, (-1 7 1)/2 at 91.31, (1 7 3)/2 at 100.33, and 3 more (see the table)
  diagnostic for R5+: (-1 1 1)/2 at 20.05, (3 1 5)/2 at 71.74, (-5 1 5)/2 at 91.98
  R4+ extinct at first order at: (-1 1 1)/2 at 20.05, (-3 3 3)/2 at 62.98, (3 1 5)/2 at 71.74, (1 3 7)/2 at 99.88; 11 families absent in the child
  R5+ extinct at first order for h = l: 11 families absent in the child
...
```

```{figure} images/xrd_CaTiO3_superlattice.png
:width: 90%

The Cu Kα1 pattern of Pnma CaTiO<sub>3</sub> (black) over that of its cubic
parent (grey). The grey ticks are the fundamental reflections, the coloured
ones the superlattice reflections of the X, M and R stars; the strongest
diagnostic reflections of each irrep of a star with two irreps are labelled.
```

How to read it:

- **Indices.** `(h k l)_child` refers to the conventional cell of the child
  space group (spglib's standard setting) and comes with the multiplicity of
  the family (the orbit under the child point group plus Friedel partners).
  `(h k l)_parent` is the same reflection on the conventional cell of the
  parent, written with a common denominator: `(1 1 3)/2` is (1/2, 1/2, 3/2).
  For a centred parent the conversion `h_conv = h_prim A^-T` is printed under
  `* Structures *` (Fm-3m ZrO<sub>2</sub> in the example directory: every
  fundamental has h, k, l all even or all odd, the superlattice reflections of
  the X point have mixed parity). Families that index to different parent k
  stars are never merged.
- **k.** `GM` for a fundamental reflection; otherwise the parent k star
  (ISO-IR names of the symmetry-mode analysis, also non-special ones such as
  the SM point (1/3,0,0) of the K<sub>2</sub>SeO<sub>4</sub> superstructure).
  The k vector printed in the summary is the star's representative; the
  reflections themselves sit on whichever arms the child's orientation selects.
  Two different stars can carry the same name (two points of one symmetry
  line, as DT (0,1/6,0) and DT (0,1/3,0) of a six-fold modulation); the k
  column then adds the representative, `DT(0,1/3,0)`, and each star has its
  own summary. In a parent with no operation taking k to -k (possible only
  in parents without inversion), -k can be a star of its own; a
  complex-conjugate pair then spans the two stars (`GP1GQ1` of a P1 parent
  tripled along a, the families at GQ being the Friedel partners of those
  at GP). Both stars list the pair with its one amplitude, their families
  are first order, and a note under the summary says so.
- **I_child, I_parent.** Intensities with the strongest child family = 100.
  For a fundamental reflection the parent reference structure is evaluated at
  the same reflection with the same multiplicity and Lorentz factor and on
  the same scale (structure factors per parent primitive cell), so `I_parent`
  next to `I_child` measures how much the distortion changed it.
- **contributions.** For a superlattice reflection the share of `|F_X|^2`
  of each irrep X of the star, `F_X` the structure factor of the parent
  structure plus the displacements of X alone (the parent itself contributes
  nothing). Shares are exact for each irrep on its own; the interference
  between irreps of one star is left out. Irreps of other stars contribute
  only through higher-order terms and are not listed. A star along which the
  child is not distorted can still carry intensity from such terms
  (harmonics of a modulation, couplings of two stars): its reflections read
  `higher order`, and its summary says so.
- **Diagnostic** reflections (share >= 95 %) are listed per irrep in the
  summary; their observation in a measured pattern points to that irrep. A
  star with a single irrep reads `all families from X (the only irrep of the
  star)`. A diagnostic family at which its irrep is extinct at first order is
  marked `(higher order)`: it is weak, carried by higher powers of the
  displacements of that irrep.
- **Extinct at first order**: the term of `F_X` linear in the displacements
  of X vanishes at that reflection. When a condition on the parent indices
  holds for exactly these families of the star (`|h| = |k| = |l|`, `h = l`,
  `l = 0`, ...), it is printed; families that are absent in the child altogether
  (systematic absences of the child space group, not in the table) are
  counted. A reflection that is extinct at first order can still have a
  share from higher orders: the `(3 3 3)/2`-type family of CaTiO<sub>3</sub>
  is extinct at first order for R4+, yet its exact share is 28 % R4+, from
  the cubic term of the large R4+ tilt (1.25 Å).

**Physical interpretation: the Glazer rules.** Glazer classified the
reflections of tilted perovskites by the parity of their pseudocubic indices
(A. M. Glazer, *Acta Cryst.* **A31**, 756 (1975)), and Woodward and Reaney
added the cation displacements (P. M. Woodward and I. M. Reaney, *Acta Cryst.*
**B61**, 387 (2005)): antiphase tilts give `(o o o)/2` reflections at R,
in-phase tilts `(o o e)/2` at M, antiparallel A-cation shifts `(e e o)/2` at
X, and a tilt about one axis is extinct for the two indices across that axis
being equal. The output reproduces these rules from the structure factors,
in the axes of the mapping: SrTiO<sub>3</sub> (a<sup>0</sup>a<sup>0</sup>c<sup>-</sup>)
has its R reflections extinct for `|h| = |k|`; in CaTiO<sub>3</sub> the
in-phase axis is the parent y axis (the doubled b axis of Pnma), so the M3+
reflections are `(o e o)/2` and M3+ is extinct for `|h| = |l|` (all 24 such
families are absent in the child), the X5+ reflections are `(e o e)/2`, and
the a<sup>-</sup>a<sup>-</sup> tilt R4+ is extinct at first order for
`|h| = |k| = |l|`, which makes the `(1 1 1)/2`-type family at 20.05° a
diagnostic reflection of R5+ (99 %). R4+ dominates 34 of the 41 R
families.

**Neutrons.** `--radiation neutron --wavelength W` (Å) uses the coherent
nuclear scattering lengths of pymatgen's `NDCalculator` and the Lorentz factor
without polarization term, both for the plain pattern and with `--parent`;
the output prefix is `ND_{cell file}_{W}A` (`ND_{child}_vs_{parent}_{W}A`
with `--parent`), and `--xraytype` is refused with it, as is `--wavelength`
with X-rays:

```bash
crystod-xrd -c example/test_POSCARs/221_PPOSCAR_ScF3 --radiation neutron --wavelength 1.5
```

```
* Structure *
 221_PPOSCAR_ScF3: ScF3, Pm-3m

* Radiation *
 neutron, 1.5 A (nuclear scattering only; no magnetic scattering, no Debye-Waller factor)

* Bragg peaks (20) in 10 - 120 deg *
    h   k   l  mult       d (A)  2theta (deg)  intensity  line
    1   0   0     6    4.069590       21.2399    100.000  neutron
    1   1   0    12    2.877635       30.2150     13.923  neutron
    ...

* Output files *
 Peak table written to: ND_221_PPOSCAR_ScF3_1.5A.txt
 Pattern (lorentzian profile, width 0.1 deg) written to: ND_221_PPOSCAR_ScF3_1.5A.pdf
```

A child that keeps the parent lattice (P4mm BaTiO<sub>3</sub> from Pm-3m)
prints `no superlattice reflections (the child keeps the parent lattice)` and
still lists the fundamentals with `I_child` and `I_parent`.

**Files.** `XRD_{child}_vs_{parent}_{xraytype}.txt` (comma separated:
`two_theta, d, h_child, k_child, l_child, multiplicity, h_parent, k_parent,
l_parent, k point, I_child, I_parent, contributions`, parent indices as
fractions, the k point with spaces for the commas of a `DT(0,1/3,0)` label,
contributions as `irrep share` pairs separated by `;`) and `.pdf`
(the two patterns broadened as in section 46, with `--peak-profile` and
`--width`). For a Kα doublet the table lists the Kα1 positions and the drawn
pattern adds Kα2 at half weight. `--min-intensity` drops weak families from
the table and the `.txt` file only (a line says how many); the k-point
summary, its diagnostic reflections and its extinction rules are built from
every family. `--two-theta`, `--tolerance` (of the symmetry-mode analysis,
default 0.01 Å) and `-o` work as in section 46.

**Limits.** No Debye-Waller factor, no magnetic scattering, no preferred
orientation, absorption or anomalous dispersion; the intensity scale is
per pattern (strongest child family = 100). The shares treat each irrep on
its own (no interference between irreps of one star) and the extinction
statements are first order in the displacements. The pairing needs a child
whose primitive cell is an integer multiple of the parent's, as for
`--supergroup-cif`, and two ordered structures: a CIF with a partially
occupied or mixed site stops with one ERROR line that names the first such
site (`... has 1 partially occupied or mixed site(s) (first: K:0.5, Na:0.5
at (0.0000, 0.0000, 0.0000)) ...`); give an ordered model instead.

From Python:

```python
from crystod import xrd

pattern = xrd.superlattice_reflections("221_PPOSCAR_CaTiO3.cif", "62_PPOSCAR_CaTiO3.cif",
                                       xray_type="CuKa1")
for reflection in pattern.superlattice[:4]:
    print(reflection.hkl_parent_label, reflection.kname, reflection.contributions_label)
for summary in pattern.kpoints:
    print(summary.kname, summary.irreps, len(summary.reflections))
print(xrd.format_superlattice_pattern(pattern))

neutron = xrd.compute_xrd_pattern(xrd.load_structure("POSCAR"), radiation="neutron",
                                  wavelength=1.5)
```

For a complex-conjugate pair listed under a star and its separate -k star,
`summary.shared` names the other star (`(('GP1GQ1', 'GQ (2/3,0,0)'),)` for
the GP star of the P1 parent tripled along a; empty otherwise). A partially
occupied or mixed-site structure raises `ValueError` with the text of the
command's ERROR.
