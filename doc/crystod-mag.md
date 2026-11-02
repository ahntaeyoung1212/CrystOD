# crystod-mag

Symmetry-adapted spin bases (cluster multipoles / SAMM): every
symmetry-distinct magnetic order allowed on the sites of one element, printed
with its irrep, its per-atom spin directions, a ready-to-paste VASP or Quantum
ESPRESSO magnetization input, and a VESTA file with spin arrows.

| I want to ... | command |
|---|---|
| enumerate the magnetic orders at q = 0 | `crystod-mag -c POSCAR --element Ni --qpoint 0 0 0` |
| survey every special k point | `crystod-mag -c POSCAR --element Ni` |
| get a Quantum ESPRESSO input instead | `... --format qe` |
| build a magnetic supercell at q ≠ 0 | `crystod-mag -c POSCAR --element Ni --qpoint R` |
| get the magnetic space group, the responses allowed by symmetry and the altermagnet / AFM class of every basis vector | `crystod-mag -c POSCAR --element Ti --qpoint GM --classify-only` ([37](#37-magnetic-space-groups-and-collinear-classes---classify-only)) |

## 36. Symmetry-adapted spin bases

*Example directory: `example/36_spin_basis` (testsuite section 36); after pip install: `crystod-example 36`*

```{note}
The `example/...` paths on this page are those of a clone of the repository,
used from its root. After `pip install CrystOD`, run `crystod-example NN` (the
number of the example directory) in your working directory first: it copies
that directory and the shared `test_POSCARs/` into `./example/`, and the
commands then run from there unchanged (see [crystod-example](crystod-example.md)).
```

Treat the spins on the sites of a magnetic element as axial-vector degrees of
freedom and decompose them into space-group irreps at a q point by projection —
a complete, symmetry-exhaustive enumeration of the ferromagnetic and
antiferromagnetic arrangements, following the cluster-multipole /
symmetry-adapted multipole moment (SAMM) framework of M.-T. Suzuki *et al.*
[PRB **95**, 094406 (2017); PRB **99**, 174407 (2019)]:

```bash
crystod-mag -c example/test_POSCARs/221_PPOSCAR_AlNi3 --element Ni --qpoint 0 0 0
crystod-mag -c example/test_POSCARs/221_PPOSCAR_AlNi3 --element Ni --qpoint 0 0 0 --format qe

# survey mode: spin-multipole irreps at ALL special k points
crystod-mag -c example/test_POSCARs/221_PPOSCAR_AlNi3 --element Ni
```

The output opens with the magnetic sites and the complete irrep enumeration —
for the Ni 3c cluster of AlNi3 (the Mn3Ir geometry), 9 spin degrees of freedom
decompose into exactly three symmetry-adapted families:

```

* Structure *
  Space group: Pm-3m (#221)
  Magnetic sites: Ni x 3
    Ni1: [0.5, 0.5, 0.0]
    Ni2: [0.5, 0.0, 0.5]
    Ni3: [0.0, 0.5, 0.5]

* Selected Q point *
  Selected q-point: GAMMA = [0.0, 0.0, 0.0]

* Spin (axial-vector) representation *
  Spin (axial-vector) representation on Ni sites: 9 dimensions
  Decomposition: 2 x GM4+(3) + GM5+(3)

* Symmetry-adapted spin bases *
  GM4+(3): dim 3 [FM, dipole]
  GM4+(3): dim 3 [AFM, octupole]
  GM5+(3): dim 3 [AFM, octupole]
```

Every basis vector is then printed with its per-atom spin directions, net
moment, its magnetic space group and class (section
[37](#37-magnetic-space-groups-and-collinear-classes---classify-only)), and
the magnetization input. The [111] component of the GM4+ cluster
octupole is the experimentally realized 120-degree structure of Mn3Ir — note
the exactly vanishing net moment:

```
* GM4+(3) [AFM, octupole] *
  ...
  component 111:
    Ni1: S = [ 0.4082,  0.4082,  0.8165]
    Ni2: S = [ 0.4082, -0.8165, -0.4082]
    Ni3: S = [-0.8165,  0.4082, -0.4082]
    net moment: [0.0, 0.0, -0.0]
    magnetic space group: R-3m' (BNS 166.101, type III), 12 operations, 6 with time reversal
    magnetic point group: -3m'
    allowed by symmetry: net magnetization along [11-1]; linear magnetoelectric effect none; piezomagnetism 4 components
    collinear order: noncollinear (not classified)
    spin directions (all atoms, POSCAR order):
      Al1: [0, 0, 0]
      Ni2: [0.4082, 0.4082, 0.8165]
      Ni3: [0.4082, -0.8165, -0.4082]
      Ni4: [-0.8165, 0.4082, -0.4082]
    MAGMOM = 0 0 0   0.4082 0.4082 0.8165   0.4082 -0.8165 -0.4082   -0.8165 0.4082 -0.4082
...

* Output files *
  ...
  Spin basis written to: POSCAR_AlNi3_spin_GM4+_octupole_111.vesta
  ...
```

Per-atom spin directions and a ready-to-paste noncollinear magnetization input
are printed by default for every basis vector. `--format` selects the input
format:

- `vasp` (default) prints a noncollinear `MAGMOM` line;
- `qe` prints the Quantum ESPRESSO counterpart — `noncolin = .true.` with
  per-type `starting_magnetization(i)` / `angle1(i)` (polar angle from z) /
  `angle2(i)` (azimuth from x) for the `&SYSTEM` namelist, where the magnetic
  element is split into one atom type per distinct spin direction (the atom
  membership of each type is printed as comments for
  `ATOMIC_SPECIES`/`ATOMIC_POSITIONS`).

With `--format qe`, the same 120-degree octupole becomes (Ni split into three
types because the three spins point in three different directions):

```
    Quantum ESPRESSO noncollinear magnetization (&SYSTEM):
      noncolin = .true.
      ! type 1 (Al): Al1 -- non-magnetic
      ! type 2 (Ni1): Ni2  S = [0.4082, 0.4082, 0.8165]
      starting_magnetization(2) = 1.0000
      angle1(2) = 35.2611
      angle2(2) = 45.0000
      ! type 3 (Ni2): Ni3  S = [0.4082, -0.8165, -0.4082]
      starting_magnetization(3) = 1.0000
      angle1(3) = 114.0928
      angle2(3) = -63.4378
      ! type 4 (Ni3): Ni4  S = [-0.8165, 0.4082, -0.4082]
      starting_magnetization(4) = 1.0000
      angle1(4) = 114.0928
      angle2(4) = 153.4378
      ! angle1 = polar angle from the z axis (deg); angle2 = azimuth from the x axis in the xy plane (deg).
      ! Split Ni into the types above in ATOMIC_SPECIES / ATOMIC_POSITIONS to realize this spin arrangement.
```

When `--qpoint` is omitted, the spin (axial-vector) irrep decomposition is
listed for every special k point of the space group — the magnetic counterpart
of the `crystod` SALC survey (e.g. for AlNi3: GM: `2.0 [GM4+(3)] + 1.0 [GM5+(3)]`,
R: `R2+ + R3+ + R4+ + R5+`, ...). At a time-reversal-invariant point a complex
irrep and its conjugate are counted once under their pair label, as with
`--qpoint` (`2.0 [GM3+GM4+(2)]` for the Ti sites of P4/m Ti2Ga3).

For the Mn3Ir-type Ni 3c cluster of AlNi3 this yields
`9 dims = 2 x GM4+(3) + GM5+(3)`: the GM4+ (T1g) cluster dipole (FM), the GM4+
(T1g) cluster octupole (AFM: the experimentally realized 120-degree structure of
Mn3Ir, which shares the irrep with the dipole and hence allows the anomalous
Hall effect), and the GM5+ (T2g) cluster octupole (AFM). Every AFM basis
satisfies `sum_i S_i = 0` exactly. FM and AFM refer to the net moment; the
copies of an irrep are named after the ranks at which it occurs among the
magnetic multipoles, lowest first: dipole (rank 1, the FM copy), quadrupole
(rank 2), octupole (rank 3), hexadecapole (rank 4), ... In a centrosymmetric
crystal the even ranks belong to the irreps odd under inversion, so an AFM
basis is not always an octupole or higher: the in-plane orders of Cr2O3 are
`GM3-(2) [AFM, quadrupole]` (section
[37](#37-magnetic-space-groups-and-collinear-classes---classify-only)).

Each basis vector is exported as
`POSCAR_<formula>_spin_<irrep>_<dipole|quadrupole|octupole|...>_<direction>.vesta`
(e.g. `POSCAR_AlNi3_spin_GM4+_octupole_111.vesta` for the 120-degree Mn3Ir-type
state), with spin arrows on the magnetic atoms scaled so that the largest spin
gets `--amplitude` Angstroms (default 1.5). For q != 0 (e.g. `--qpoint R`) the
commensurate magnetic supercell is built automatically, with the Bloch phase
applied to the spins and the MAGMOM/VESTA output referring to the supercell.
The supercell takes the denominator of each coordinate of q along the
primitive axes, up to 12; a q that does not fit such a cell (`0 0 0.15`,
that is 3/20) is refused with an ERROR that names the smallest commensurate
supercell (1x1x20), instead of a cell that would cut the modulation.
`--conventional` exports the spin structures in the conventional cell instead of
the primitive cell (VESTA files get a `_conv` suffix, and for q != 0 the
conventional cell is multiplied until the Bloch phase is commensurate, again
with at most 12 cells per axis: NaCl at `1/5 1/7 0` is refused with
`--conventional`, and the ERROR names the 5x7x1 primitive supercell that works
without it). `--tolerance` sets the symmetry tolerance (default 1e-5, a
distance in Angstrom). It is used by spglib and for matching the sites onto
each other, both in the spin representation and in the collinear
classification, and the Bloch phases are built from the site positions
symmetrized by the space group, so a structure slightly off its ideal
positions decomposes and classifies like the ideal one (an AlNi3 cell with
the Ni coordinates off by about 1e-4 gives `2 x GM4+(3) + GM5+(3)` and the
groups and classes of the ideal cell with `--tolerance 1e-2`; the listings
keep the input positions).

The input must be a primitive cell. It is used as given (the spin listings
and MAGMOM follow its atom order), and a conventional cell or a supercell is
refused with the same one-line ERROR in the survey and the `--qpoint` modes
(the PPOSCAR of `phonopy --symmetry` is a primitive cell). For the two-atom
cubic cell of bcc Fe:

```
ERROR: the input cell is not primitive (2 atoms, the primitive cell has 1); crystod-mag uses the cell as given (the spin listings and MAGMOM follow its atom order) and needs a primitive cell, such as the PPOSCAR of phonopy --symmetry (--conventional then exports the spin structures in the conventional cell).
```

```{seealso}
**Theory:** the construction is the SALC projection used elsewhere in CrystOD
([1. Theoretical background](theory-representations.md)) with the Cartesian
part replaced by `det(R) R`, since spins are axial vectors; irrep labels come
from spgrep + the bundled ISO-IR (ISOTROPY, Miller-Love) tables as usual.
Within a multiply-occurring irrep the unique net-moment (dipole) combination is
split off from the net-zero (higher-multipole) ones; this needs every copy of
the irrep in the same real partner basis, so a two-dimensional irrep that
spgrep gives in a complex basis (the E irreps of trigonal and hexagonal
crystals, GM5+ of rutile) is first brought to a real form, and the partners of
all copies are turned together onto directions that a two-fold axis or a
mirror fixes (for an in-plane E irrep the a axis of the hexagonal cell, x of
a standard cell, and the direction perpendicular to it, in the 321 and the
312 settings alike), as `crystod-phonon --vibration` chooses the
displacement partners. A one-dimensional complex irrep and its conjugate
are merged into one real two-dimensional space under their pair label
(`GM3+GM4+(2)`, the label `crystod-group --parent` prints), at q = 0 and at
the other time-reversal-invariant q points (`Z3+Z4+(2)`); its partners are
the real and imaginary parts of the complex basis vector, the real partners
that `crystod-phonon --vibration` gives the two members of the pair. Multipole ranks
(dipole, quadrupole, octupole, ...) are assigned representation-theoretically from the
parity-resolved angular-momentum characters (Suzuki's Table III logic): the
rank-p magnetic multipole transforms as angular momentum p with parity
(-1)^(p+1) under inversion.
```

## 37. Magnetic space groups and collinear classes (`--classify-only`)

*Example directory: `example/37_magnetic_group` (testsuite section 37); after pip install: `crystod-example 37`*

Every spin basis vector that `crystod-mag --qpoint` exports is also an
arrangement of moments with its own symmetry. For each exported component
the command gives the magnetic space group (MSG) of that arrangement, its
magnetic point group (MPG), the macroscopic responses the MPG allows by
symmetry (a net magnetization, the linear magnetoelectric effect,
piezomagnetism) and, for a collinear arrangement, its class: ferromagnet,
ferrimagnet, conventional antiferromagnet, or altermagnet with its d-, g- or
i-wave type. It answers "which of these orders is an altermagnet, and which
one allows a magnetoelectric effect?" for every order the symmetry of the
crystal admits. `--classify-only` prints the table alone:

```bash
crystod-mag -c example/test_POSCARs/136_PPOSCAR_TiO2 --element Ti --qpoint GM --classify-only
```

```
...
* Symmetry-adapted spin bases *
  GM3+(1): dim 1 [FM, dipole]
  GM5+(2): dim 2 [FM, dipole]
  GM5+(2): dim 2 [AFM, octupole]
  GM2+(1): dim 1 [AFM, octupole]

* Magnetic groups and collinear classes *
basis                    component  MSG (BNS)             type  MPG      net magnetization  class
GM3+(1) [FM, dipole]     z          P4_2/mn'm' (136.501)  III   4/mm'm'  along z            ferromagnet
GM5+(2) [FM, dipole]     1m10       Cmm'm' (65.486)       III   mm'm'    along [1-10]       ferromagnet
GM5+(2) [FM, dipole]     110        Cmm'm' (65.486)       III   mm'm'    along [110]        ferromagnet
GM5+(2) [AFM, octupole]  1m10       Cmm'm' (65.486)       III   mm'm'    along [1-10]       altermagnet (d-wave)
GM5+(2) [AFM, octupole]  110        Cmm'm' (65.486)       III   mm'm'    along [110]        altermagnet (d-wave)
GM2+(1) [AFM, octupole]  z          P4_2'/mnm' (136.499)  III   4'/mmm'  none               altermagnet (d-wave)

MSG: magnetic space group from the moment directions (BNS number, type I-IV);
MPG: magnetic point group; net magnetization: the direction allowed by symmetry;
class: from the signs of the moments along their common axis only (nonrelativistic).
MSG and MPG symbols are in the standard BNS setting; directions are in the Cartesian axes of the input cell.
...
Reference (collinear classes): L. Smejkal, J. Sinova and T. Jungwirth, Phys. Rev. X 12, 031042 (2022).
```

Without `--classify-only` every component of the listing gets four more
lines after its net moment, here for the antiparallel pair along c (the
RuO2 / MnF2 order):

```
* GM2+(1) [AFM, octupole] *
  component z:
    Ti1: S = [ 0.0000,  0.0000,  1.0000]
    Ti2: S = [ 0.0000,  0.0000, -1.0000]
    net moment: [0.0, 0.0, 0.0]
    magnetic space group: P4_2'/mnm' (BNS 136.499, type III), 16 operations, 8 with time reversal
    magnetic point group: 4'/mmm'
    allowed by symmetry: net magnetization none; linear magnetoelectric effect none; piezomagnetism 2 components
    collinear order: altermagnet (d-wave; spin splitting allowed by symmetry ~ k_x k_y; spin-degenerate planes perpendicular to [100], [010])
    spin directions (all atoms, POSCAR order):
...
```

How to read it:

- `magnetic space group` is the group of the arrangement as spglib finds it
  (axial moments, `get_magnetic_symmetry_dataset`), written with its
  Belov-Neronova-Smirnova (BNS) symbol and number (`P4_2'/mnm'`, 136.499;
  the symbol from the pymatgen table, the number from spglib) and its type:
  I, no operation with time reversal; II, the grey group of a non-magnetic
  crystal; III, half of the operations combined with time reversal (primed);
  IV, a lattice translation combined with time reversal (the subscript of
  the lattice letter, `P_I`, `C_c`, `R_I`). The operations are counted in
  the magnetic cell. `MagneticGroupInfo.og_number` holds the
  Opechowski-Guccione number.
- `magnetic point group` is derived from the BNS symbol: the lattice letter
  and its subscript are dropped, glides (a, b, c, n, d, e) become m, screw
  axes become plain rotations (`4_2'` becomes `4'`), every prime is kept,
  and a type II or IV group is the point group followed by `1'`
  (`P_I4/mnc` gives `4/mmm1'`). The symbol is then checked against the
  operations: the group it generates must have the same rotations with the
  same time-reversal flags (a symbol that disagrees is printed with `?`;
  the derivation is tested on all 1651 magnetic space groups).
- `allowed by symmetry` lists what the MPG allows: the net magnetization (a
  time-odd axial vector, Jahn symbol `aeV`) with its direction in the
  Cartesian axes of the input cell (`along z`, `in the xy plane`,
  `any direction`, `none`), and the numbers of independent components of the
  linear magnetoelectric tensor (`aeV2`, `P_i = alpha_ij H_j`) and of the
  piezomagnetic tensor (`aeV[V2]`, `M_i = Lambda_ijk sigma_jk`), computed as
  in section 13 of [crystod-group](crystod-group.md) with the time-reversal
  flag of every operation. Symmetry decides what is allowed, not how large
  it is: an allowed response can be small. An AFM basis vector that shares
  its irrep with the FM dipole (the GM4+ octupoles of AlNi3) allows a net
  magnetization by symmetry although its own moments add up to zero.
- `collinear order` classifies a collinear arrangement from the signs of its
  moments along their common axis: the operations of the space group of the
  non-magnetic crystal, combined with every lattice translation inside the
  magnetic cell, either keep the sign pattern, reverse it (exchange the two
  sublattices), or do neither. A nonzero net moment gives a ferromagnet
  (all signs equal) or a ferrimagnet; a compensated arrangement with no
  exchanging operation is listed as `compensated, sublattices not related by
  symmetry`; if a pure translation exchanges the sublattices, the order is a
  conventional antiferromagnet (`T tau`), as it is if the inversion does
  (`P T`): the bands stay spin degenerate in the nonrelativistic limit.
  When only rotations or mirrors (proper or improper, with or without a
  fractional translation) exchange the sublattices, the order is an
  altermagnet.
- The altermagnet wave type follows L. Smejkal, J. Sinova and T. Jungwirth,
  Phys. Rev. X 12, 031042 (2022): with `chi_A(R) = +1` on the rotation parts
  that keep the pattern and `-1` on those that exchange the sublattices, the
  wave is the lowest even `l` (2, 4, 6: d, g, i) at which `chi_A` occurs in
  the angular-momentum representation `D_l`. The lowest-order spin splitting
  allowed by symmetry has the form of the polynomials of that degree with `p(R k) = chi_A(R) p(k)` (`k_x k_y` for
  rutile, `k_x k_z (k_x^2 - 3 k_y^2)` for the g-wave hematite order), in
  the Cartesian axes of the input cell. The spin-degenerate planes are those
  fixed by a mirror or a two-fold rotation that exchanges the sublattices
  (both give the same plane, because the nonrelativistic bands satisfy
  `E(k) = E(-k)`); lines along the axes of the other exchanging rotations
  are listed when they do not lie in one of the planes.
- Two frames meet in one row. The MSG and MPG symbols are those of the
  standard BNS setting (pymatgen's table), whatever the orientation of the
  input cell; the net magnetization, the spin-splitting polynomials and the
  degenerate planes are given in the Cartesian axes of the input cell. A row
  `Pnn'm' ... mm'm' ... along y` is therefore consistent: the standard
  setting of Pnn'm' allows a moment along its own x axis, which is y of the
  input cell.
- The `component` column holds the tag of the VESTA file name of the
  component: `x`, `y`, `z`, `111`, `1m10`, ..., with `_2`, `_3` when the
  same tag occurs again in the same basis (`x_2` is the second `x`
  component and the file `..._x_2.vesta`), and `comp<N>` for a component
  without a single direction: the N-th basis vector of its space, counted
  from 1 like the mode numbers of the other commands
  (`POSCAR_AlNi3_spin_R4+_R_comp1.vesta` is the first of the three R4+
  components).
- When spglib finds the operations of an arrangement but cannot identify
  the group in the cell given, the cell is re-expressed by its Niggli- and
  Delaunay-reduced bases and a few sheared bases of the same lattice; if no
  basis is identified, the line reads `magnetic space group: not identified
  by spglib (N operations, M with time reversal)`, the MSG, type and MPG
  cells of the table hold `-`, and the responses and the class are still
  computed from the operations.
- At q = 0 every AFM basis vector has a zero net moment, and the closing
  sentence `All AFM bases satisfy sum_i S_i = 0` says so. Two cases need
  more than one FM row per irrep or a merged irrep:
  - A one-dimensional complex irrep and its conjugate (the little co-group
    is 3, -3, 4, -4, 4/m, 6, -6, 6/m, 23 or m-3, for example GM3+ and GM4+
    of P4/m) carry the same real spin arrangements: time reversal joins
    them into one physically irreducible real space of dimension two. They
    are listed once under their pair label, `GM3+GM4+(2)`, the label that
    `crystod-group --parent` prints for the same pair, and each copy has
    two real partners, the real and imaginary parts of the complex basis
    vector (the real partners `crystod-phonon --vibration` gives the two
    members). Every operation acts on the two partners as a rotation, so no
    symmetry element fixes a direction in the space, and the partners
    follow a convention instead: the net moment of the first partner of the
    FM dipole copy lies along the a axis (x of a standard cell) and that of
    the second perpendicular to it, in tetragonal, trigonal and hexagonal
    crystals alike; the first partner of an AFM copy carries as much
    of its moments along the a axis as it can (then along the direction
    perpendicular to a and c, then the third axis). The net moments are
    separated on the complex copies, so the pair has at most one FM dipole
    copy and net-zero AFM copies. A note under the table names every merged
    pair. The same pairs are merged at the other time-reversal-invariant q
    points (`Z3+Z4+(2)` at Z of P4/m), where the irreps of a pair can also
    be two-dimensional: each copy then gives one real four-dimensional space
    under the pair label, the real parts of its partners followed by the
    imaginary parts (`A5A6(4)` for Zn at A of ZnO, P6_3mc).
  - An irrep that occurs more than once in a single moment (the moments
    perpendicular to the two-fold axis of a monoclinic crystal, any moment
    of a triclinic one) has one FM row per allowed axis: the uniform moments
    along the a axis, the direction perpendicular to a and c, and the third
    axis of that frame (the Cartesian axes of a triclinic cell), projected
    onto the irrep. The AFM rows are the rest of the space, each with its
    moments along one of those axes where symmetry keeps the axes apart.

**Moment directions versus sign pattern.** The MSG and MPG use the actual
moment directions, as a relativistic (spin-orbit coupled) description
requires; the class uses only the sign pattern, as in the nonrelativistic
spin-group description of altermagnets. The two answer different questions:
in MnO2 (Pnnm) the antiparallel pair along z has the type I group Pnnm and
along x the type III group Pnn'm' (standard setting; here it allows a net
magnetization along y of the input cell), while both are d-wave
altermagnets:

```bash
crystod-mag -c example/test_POSCARs/58_PPOSCAR_MnO2 --element Mn --qpoint GM --classify-only
```

```
...
* Magnetic groups and collinear classes *
basis                    component  MSG (BNS)        type  MPG    net magnetization  class
GM1+(1) [AFM, octupole]  z          Pnnm (58.393)    I     mmm    none               altermagnet (d-wave)
GM2+(1) [FM, dipole]     z          Pn'n'm (58.397)  III   m'm'm  along z            ferromagnet
GM4+(1) [FM, dipole]     y          Pnn'm' (58.398)  III   mm'm'  along y            ferromagnet
GM4+(1) [AFM, octupole]  x          Pnn'm' (58.398)  III   mm'm'  along y            altermagnet (d-wave)
GM3+(1) [FM, dipole]     x          Pnn'm' (58.398)  III   mm'm'  along x            ferromagnet
GM3+(1) [AFM, octupole]  y          Pnn'm' (58.398)  III   mm'm'  along x            altermagnet (d-wave)
...
```

**Magnetoelectric Cr2O3 and hematite.** On the four Cr sites of R-3c
(`example/37_magnetic_group/167_PPOSCAR_Cr2O3`, rhombohedral primitive cell
with the hexagonal c axis along z), the order + - + - along c (GM1- z) is the
Cr2O3 order: R-3'c' (167.106), MPG -3'm', a conventional antiferromagnet
(`P T`) with the linear magnetoelectric effect allowed by symmetry,
`alpha_11 = alpha_22` and `alpha_33`. The order + - - + (GM1+ z, the
hematite order below the Morin transition) is the type I group R-3c
(167.103), a g-wave altermagnet with piezomagnetism allowed. With the
Python API, `format_magnetic_responses` prints the tensor forms:

```python
from phonopy.interface.vasp import read_vasp
from crystod import mag

result = mag.analyze_spin_basis(read_vasp("167_PPOSCAR_Cr2O3"), "Cr", "GM")
cr2o3 = next(c for c in result.components if c.label.startswith("GM1-"))
print(cr2o3.group.bns_symbol, cr2o3.group.magnetic_point_group)
print("\n".join(mag.format_magnetic_responses(cr2o3.responses)))
```

```
R-3'c' -3'm'
  net magnetization: forbidden by symmetry
  linear magnetoelectric effect: allowed by symmetry, 2 independent components
    alpha  1        2        3
    1      alpha11  0        0
    2      0        alpha11  0
    3      0        0        alpha33
  piezomagnetism: forbidden by symmetry
```

The orders with the moments in the plane belong to the two-dimensional irreps
GM3+ and GM3-. Their partners are taken along the two-fold axis of R-3c that
lies along a (x of this cell) and perpendicular to it (y), the same pair for
every copy of an irrep, so the FM dipole separates from the AFM orders and
every arrangement keeps a monoclinic group with that two-fold axis:

```bash
crystod-mag -c example/37_magnetic_group/167_PPOSCAR_Cr2O3 --element Cr --qpoint GM --classify-only
```

```
...
GM3+(2) [FM, dipole]         x          C2/c (15.85)      I     2/m    along x            ferromagnet
GM3+(2) [FM, dipole]         y          C2'/c' (15.89)    III   2'/m'  in the yz plane    ferromagnet
GM3+(2) [AFM, octupole]      y          C2/c (15.85)      I     2/m    along x            altermagnet (g-wave)
GM3+(2) [AFM, octupole]      x          C2'/c' (15.89)    III   2'/m'  in the yz plane    altermagnet (g-wave)
GM3-(2) [AFM, quadrupole]    x          C2'/c (15.87)     III   2'/m   none               antiferromagnet (P T)
GM3-(2) [AFM, quadrupole]    y          C2/c' (15.88)     III   2/m'   none               antiferromagnet (P T)
GM3-(2) [AFM, quadrupole]    y_2        C2'/c (15.87)     III   2'/m   none               antiferromagnet (P T)
GM3-(2) [AFM, quadrupole]    x_2        C2/c' (15.88)     III   2/m'   none               antiferromagnet (P T)
...
```

The GM3+ octupole is the hematite order + - - + with the moments in the
plane: C2/c with the moments perpendicular to the two-fold axis, C2'/c' with
the moments along it, where a net magnetization perpendicular to the moments
(in the yz plane) is allowed by symmetry. The first GM3- copy is the order
+ - + - of Cr2O3 with the moments in the plane, the second the order + + - -.
Each AFM copy lists its partners in the order of the FM partners they
transform like, not by their own moment directions: the first octupole row
(moments along y) has the group of the FM dipole along x.

**Complex conjugate pairs.** In P4/m the irreps GM3+ and GM4+ (and GM3- and
GM4-) are one-dimensional and complex; each pair forms one real
two-dimensional space, listed under its pair label. For the Ti 4k sites of
Ti2Ga3 the in-plane FM dipole and the in-plane octupole both belong to
GM3+GM4+:

```bash
crystod-mag -c example/test_POSCARs/83_PPOSCAR_Ti2Ga3 --element Ti --qpoint GM --classify-only
```

```
...
  Decomposition: GM1+(1) + 2 x GM1-(1) + 2 x GM3+GM4+(2) + GM3-GM4-(2) + GM2+(1) + 2 x GM2-(1)

* Symmetry-adapted spin bases *
  GM1+(1): dim 1 [FM, dipole]
  GM1-(1): dim 1 [AFM, quadrupole]
  GM1-(1): dim 1 [AFM, hexadecapole]
  GM3+GM4+(2): dim 2 [FM, dipole]
  GM3+GM4+(2): dim 2 [AFM, octupole]
  GM3-GM4-(2): dim 2 [AFM, quadrupole]
  GM2+(1): dim 1 [AFM, octupole]
  GM2-(1): dim 1 [AFM, quadrupole]
  GM2-(1): dim 1 [AFM, quadrupole]

* Magnetic groups and collinear classes *
basis                          component  MSG (BNS)       type  MPG    net magnetization  class
GM1+(1) [FM, dipole]           z          P4/m (83.43)    I     4/m    along z            ferromagnet
GM1-(1) [AFM, quadrupole]      comp1      P4/m' (83.46)   III   4/m'   none               noncollinear
GM1-(1) [AFM, hexadecapole]    comp1      P4/m' (83.46)   III   4/m'   none               noncollinear
GM3+GM4+(2) [FM, dipole]       x          P2'/m' (10.46)  III   2'/m'  in the xy plane    ferromagnet
GM3+GM4+(2) [FM, dipole]       y          P2'/m' (10.46)  III   2'/m'  in the xy plane    ferromagnet
GM3+GM4+(2) [AFM, octupole]    x          P2'/m' (10.46)  III   2'/m'  in the xy plane    altermagnet (d-wave)
GM3+GM4+(2) [AFM, octupole]    y          P2'/m' (10.46)  III   2'/m'  in the xy plane    altermagnet (d-wave)
GM3-GM4-(2) [AFM, quadrupole]  z          P2'/m (10.44)   III   2'/m   none               antiferromagnet (P T)
GM3-GM4-(2) [AFM, quadrupole]  z_2        P2'/m (10.44)   III   2'/m   none               antiferromagnet (P T)
GM2+(1) [AFM, octupole]        z          P4'/m (83.45)   III   4'/m   none               altermagnet (d-wave)
GM2-(1) [AFM, quadrupole]      comp1      P4'/m' (83.47)  III   4'/m'  none               noncollinear
GM2-(1) [AFM, quadrupole]      comp1_2    P4'/m' (83.47)  III   4'/m'  none               noncollinear
...
component: the tag of the listing and of the VESTA file name (comp<N>: the N-th basis vector of its space).
GM3+GM4+(2): a complex irrep and its conjugate, merged into one real space (the physically irreducible representation).
GM3-GM4-(2): a complex irrep and its conjugate, merged into one real space (the physically irreducible representation).
```

The in-plane octupole shares the pair with the dipole, so it allows a net
magnetization in the plane by symmetry although its own moments add up to
zero, as the GM4+ octupoles of AlNi3 do. In Python,
`SpinBasisAnalysis.pair_labels` lists the merged pairs.

**q != 0.** The classification refers to the real spin field of the magnetic
cell, with the global phase fixed as in the VESTA export; the
`* Magnetic cell *` block says so. The spin of site j in the cell at lattice
translation T is `Re(w_j exp(2 pi i q.T))`, where w_j includes the Bloch factor
`exp(2 pi i q.x_j)` of the site position (the lattice gauge, as in
`crystod-phonon --modulation`), so every arrangement transforms like its
irrep whatever the positions of the sites. For the NiO-type order on the Na
sites of rock salt at L (moments along [111]):

```bash
crystod-mag -c example/test_POSCARs/225_PPOSCAR_NaCl --element Na --qpoint L --classify-only
```

```
...
* Magnetic groups and collinear classes *
basis                  component  MSG (BNS)         type  MPG    net magnetization  class
L2+(1) [AFM (q != 0)]  111        R_I-3c (167.108)  IV    -3m1'  none               antiferromagnet (T tau)
...
```

At a time-reversal-invariant q != 0 a complex irrep and its conjugate are
listed under their pair label, as at q = 0, also when they are
two-dimensional. For Zn at A of ZnO (P6_3mc) the decomposition is
`A2A3(2) + A5A6(4)`, the list of bases reads `A5A6(4): dim 4 [AFM (q != 0)]`,
and the table has its four real partners as `comp1` ... `comp4`: the real
parts of the two complex partners followed by their imaginary parts.

**Limits.** Only collinear arrangements are classified; a noncollinear
component gets `noncollinear (not classified)` (its MSG and responses are
still given). The table describes the components as `crystod-mag` exports
them: for an irrep of dimension two or three each exported component is one
arrangement, and another partner direction can have another MSG. For
q != 0 the MSG depends on the global phase convention of the real field.
The MPG symbol is derived from the BNS symbol and checked; the operations
(`MagneticGroupInfo.rotations`, `.translations`, `.time_reversals`) are the
authority. `--classify-only` needs `--qpoint`; the survey mode (no
`--qpoint`) does not classify. At q = 0 the partners of a two- or
three-dimensional irrep lie along directions that symmetry fixes; which of
several equivalent choices is printed is a convention (in a trigonal or
hexagonal crystal the a axis of the input cell and the direction
perpendicular to it, whichever of the equivalent in-plane axes spglib calls
a), and a rotated input cell gives the same groups and classes with the
directions rotated. The partners of a merged complex pair are not fixed by
symmetry and follow the convention described above.

**Python API.**

```python
from phonopy.interface.vasp import read_vasp
from crystod import mag

result = mag.analyze_spin_basis(read_vasp("136_PPOSCAR_TiO2"), "Ti", "GM")
for row in mag.magnetic_summary_rows(result):
    print(row)
component = result.components[-1]                 # GM2+(1) [AFM, octupole], z
component.group.bns_number, component.collinear.wave, component.collinear.spin_splitting
# ('136.499', 'd-wave', ('k_x k_y',))
```

`mag.magnetic_space_group(lattice, positions, numbers, magmoms)`,
`mag.macroscopic_responses(group)` and `mag.classify_collinear(...)` work on
any cell and spin field; their docstrings give the arguments.
`analyze_spin_basis` takes `--tolerance` as `symprec` and raises `ValueError`,
with the text of the ERROR lines of the command, for a cell that is not
primitive and for a q that does not fit a magnetic cell of at most 12 cells
along each axis.
