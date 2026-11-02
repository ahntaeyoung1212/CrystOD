# Theoretical background

This page collects the group theory that CrystOD evaluates internally: the
representation matrices of the symmetry operations, the characters that reduce
them to irreducible representations, and the projection operator that turns an
irrep into an actual symmetry-adapted function. None of it has a command-line
flag — it is the shared basis of `crystod`, `crystod-group` and `crystod-mag`,
and it is what the output labels of those commands mean.

*Example directory: `example/01_wigner_d` (testsuite section 1); after pip install: `crystod-example 01`*

```{note}
This section is theoretical background, **not a command-line mode**: there is
no CLI flag for it. Everything described here is what CrystOD evaluates
internally every time one of the commands is executed; the machinery is exposed
through the Python API only (`crystod.operations.wigner_D_real`). No prior
familiarity with group theory is assumed.
```

## What is a representation matrix?

Take a symmetry operation of a crystal — say a 90-degree rotation about the z
axis, `C4z` (x -> y, y -> -x, z -> z). Apply it to the three p orbitals, which
have the same shapes as the functions x, y, z:

- p_x -> p_y, p_y -> -p_x, p_z -> p_z.

Each orbital turns into a *linear combination* of the orbitals of the same
shell. Collecting the coefficients into a matrix gives the **representation
matrix** `D(R)` of the operation on that shell. CrystOD writes the image of the
j-th orbital into the j-th **column**: `f_j(R^-1 r) = sum_i D[i, j] f_i(r)`,
the convention in which `D(AB) = D(A) D(B)` (the transpose, with the images in
the rows, is `D(R^-1)`). For p orbitals the matrix is then simply the 3x3
rotation matrix itself; its first column says p_x -> p_y:

```
D^(1)(C4z) =  [ 0 -1  0 ]      (basis order: x, y, z;
              [ 1  0  0 ]       column j = image of orbital j)
              [ 0  0  1 ]
```

For d orbitals the same idea gives a 5x5 matrix. Under `C4z`:
xy -> -xy, yz -> -xz, z^2 -> z^2, xz -> yz, x^2-y^2 -> -(x^2-y^2), so, with
the same column convention (the yz column holds -1 in the xz row, the xz
column +1 in the yz row),

```
D^(2)(C4z) =  [-1  0  0  0  0 ]      (basis order: xy, yz, z^2, xz, x^2-y^2;
              [ 0  0  0  1  0 ]       column j = image of orbital j)
              [ 0  0  1  0  0 ]
              [ 0 -1  0  0  0 ]
              [ 0  0  0  0 -1 ]
```

These matrices for the orbital shells (any `l`) are the **Wigner D matrices**
on real spherical harmonics. `crystod.operations.wigner_D_real(l, R)` returns
them for an arbitrary O(3) operation `R` (rotation, rotoinversion, or mirror):

```python
import numpy as np
from crystod.operations import wigner_D_real

c4z = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

wigner_D_real(1, c4z)           # l = 1 (p): equals the 3x3 rotation matrix itself
wigner_D_real(2, c4z)           # l = 2 (d): the 5x5 matrix above
np.trace(wigner_D_real(2, c4z)) # character of C4z on the d shell: -1
wigner_D_real(3, -np.eye(3))    # inversion: (-1)^l x identity (parity), here -1 x 1(7)
```

Key properties (verified in testsuite section 1):

- for proper rotations at `l` = 1 the matrix equals `R` itself;
- inversion is represented by `(-1)^l` times the identity — even shells
  (s, d, ...) are unchanged, odd shells (p, f, ...) flip sign (the *parity* of
  the orbital);
- the map is a group homomorphism, `D(AB) = D(A) D(B)` — performing two
  operations in sequence is the same as multiplying their matrices;
- all matrices are orthogonal, `D D^T = 1`.

## How CrystOD computes D for any l

Rotation matrices act directly on (x, y, z), so `l` = 1 is trivial — but how do
we get the 5x5, 7x7, ... matrices for d, f, ... shells without working out every
monomial by hand? CrystOD follows the classic quantum-mechanics route
(prototyped in `matsym/wigner_d.ipynb` by Hiroki Koiso):

1. **Split off the inversion.** Any O(3) operation is a proper rotation times
   (possibly) the inversion: `R = det(R) x R_proper`. Work with the proper
   rotation `R_proper = det(R) R` first.
2. **Convert the rotation to Euler angles** (alpha, beta, gamma) in the ZYZ
   convention.
3. **Evaluate Wigner's formula** for the complex D matrix `D^(l)(alpha, beta,
   gamma)` — the standard result for how the complex spherical harmonics
   `Y_l^m` (m = -l..l) transform under rotations. This works for *any* l.
4. **Change basis from complex to real orbitals.** The real orbitals are fixed
   linear combinations of `Y_l^m` (e.g. `p_x = (Y_1^-1 - Y_1^1)/sqrt(2)`), so a
   unitary matrix `C` converts the complex D matrix to the real-orbital one:
   `D_real = C D_complex C^-1`.
5. **Restore the parity.** If the original operation was improper
   (`det(R) = -1`), multiply by the inversion eigenvalue `(-1)^l`.

The production implementation in `crystod/operations.py` performs these steps
in pure NumPy (no symbolic algebra), with the orbital ordering
p: (x, y, z) / d: (xy, yz, z^2, xz, x^2-y^2) / f: (7 components).

## From matrices to irreps: characters and the reduction formula

The trace of a representation matrix, `chi(R) = tr D(R)`, is called the
**character** of the operation. Characters are the workhorse of applied group
theory because they do not depend on the basis choice, and because tabulated
**character tables** list the characters `chi_Gamma(R)` of each *irreducible
representation* (irrep) Gamma — the elementary building blocks into which any
representation decomposes. The number of times an irrep appears is given by the
reduction formula

```
n_Gamma = (1/|G|) * sum_g  chi_Gamma(g)* chi(g),
```

an average over all `|G|` operations of the group. A complex-conjugate pair
stored as one real irrep in the point-group tables (such as `Eg` of `-3` or
`m-3`) has norm `sum_g |chi_Gamma(g)|^2 = 2|G|`, so the reduction divides by
the norm of each irrep rather than by `|G|`. This is exactly how the
ligand-field splitting of `crystod-group --ligand-field` (section 11) works:
the characters of the d shell in the
m-3m field, `chi(g) = tr D^(2)(g)`, reduce to `Eg + T2g` — the familiar
two-below-three splitting of d orbitals in an octahedral crystal field.

In a crystal, a space-group operation `g = {R | t}` does two things to an
atomic orbital: it moves the atom to a symmetry-equivalent site, and it mixes
the orbital components on that site with `D^(l)(R)`. The character of the
crystal-orbital representation at a k point is therefore the trace of
`D^(l)(R)` summed over the atoms that `g` maps onto themselves (modulo a
lattice translation), weighted by the Bloch phase `exp(-ik.t)` of that
translation. Reducing these characters against the little-group irrep table
gives the irrep content printed by the SALC analysis and the crystal-orbital
diagrams of `crystod` (sections 2-3).

## From irreps to basis functions: the projection operator

Knowing *how many times* an irrep appears is half the story; the other half is
*what the symmetry-adapted functions look like*. They are extracted with the
**projection operator**

```
P^(Gamma) = (d_Gamma/|G|) * sum_g  chi_Gamma(g)* O(g),
```

where `O(g)` is the representation matrix of `g` on some convenient trial basis
and `d_Gamma` is the irrep dimension. Applied to a trial function, `P^(Gamma)`
kills every component except the part transforming as Gamma.

The workflow — prototyped in `matsym/get_basis_functions.ipynb` by Hiroki
Koiso — is the engine of `crystod-group --basis` / `--generate-basis`
(sections 12 and 14 of `crystod-group`):

1. get the symmetry operations from **spglib** and the irreps from **spgrep**;
2. build the representation matrices `O(g)` of the trial basis — for the linear
   monomials (x, y, z) these are the rotation matrices themselves; for the
   quadratic monomials (x^2, y^2, z^2, xy, yz, zx) a 6x6 matrix follows from
   substituting the rotated coordinates into each monomial, and so on;
3. project with `P^(Gamma)` (spgrep's `project_to_irrep`) and read off the
   symmetry-adapted polynomials — e.g. in m-3m the quadratic monomials separate
   into `x^2+y^2+z^2` (A1g), `(2z^2-x^2-y^2, x^2-y^2)` (Eg), and
   `(xy, yz, zx)` (T2g).

It is worth *looking* at the matrices of step 2 before projecting. The two
galleries below (regenerated from the notebook by
`doc/make_rep_matrix_figures.py`) show `O(g)` for **all 48 operations** of the
Pm-3m point group of ScF3 at the Gamma point — one small panel per operation,
with red = +1, blue = -1, white = 0.

```{figure} images/rep_matrices_linear.png
:name: fig-rep-linear
:width: 85%

The 48 representation matrices on the linear basis (x, y, z) — i.e. the 3x3
rotation matrices themselves. Red = +1, blue = -1, white = 0.
```

Two things are immediately visible. First, every panel has exactly one colored
cell per row and per column: for a high-symmetry (cubic) group the operations
do nothing more exotic than **permute the basis functions and flip signs** —
these are *signed permutation matrices* (the first panel, the identity, is the
plain red diagonal). Second, no single monomial stays put in every panel, which
is the visual way of saying that x, y, z individually are *not*
symmetry-adapted: they mix, and only the projection operator can disentangle
the combinations that transform cleanly.

```{figure} images/rep_matrices_quadratic.png
:name: fig-rep-quadratic
:width: 85%

The same 48 operations on the quadratic basis (x^2, y^2, z^2, xy, yz, zx) —
now 6x6 signed permutation matrices.
```

The quadratic gallery shows one more feature: a **block structure**. The
upper-left 3x3 block (x^2, y^2, z^2) and the lower-right 3x3 block
(xy, yz, zx) never mix — a rotation can turn x^2 into y^2, or xy into -yz, but
never a square into a product. Note also that the squares block is *always
red*: a square can never acquire a minus sign, which is why the totally
symmetric average `x^2+y^2+z^2` (A1g) survives. The projection operator is
nothing but a weighted average of these 48 panels — multiply each panel by the
irrep character `chi_Gamma(g)*` and sum — and the block structure you see here
is exactly what reappears in the result: the squares block yields
`A1g + Eg`, the products block yields `T2g`, reproducing step 3 above (and the
`crystod-group --generate-basis --order 2` output of section 14 of
`crystod-group`).

The SALCs of the main `crystod` command and of its SALC viewer (section 6 of
`crystod`) come from the *same* projection with
the trial basis replaced by the atomic orbitals on all symmetry-equivalent
sites — `O(g)` then combines the site permutation, the Bloch phases, and
`D^(l)(R)` from section 1.2 above. And replacing `D^(1)(R) = R` by the
axial-vector
representation `det(R) R` (magnetic moments do not flip under inversion) turns
the same machinery into the spin-multipole bases of `crystod-mag` (section 36).

*Further reading:* B. Souvignier, "Representations of crystallographic groups"
([MaThCryst summer school notes, Nancy 2010](https://www.crystallography.fr/mathcryst/pdf/nancy2010/Souvignier_irrep_slides.pdf));
the spgrep documentation, e.g. the
[symmetry-adapted tensor example](https://spglib.github.io/spgrep/examples/symmetry_adapted_tensor.html).
Run `python demo_wigner_d.py` in `example/01_wigner_d` for a printed walk-through.

## Irrep labels and the frame of a structure

Every space-group irrep label that CrystOD prints — the SALC analysis,
`--atomic-orbital` and `--diagram` of `crystod`, `crystod-phonon --irreps`,
`--vibration`, `--vector` and `--modulation`, `crystod-mag` and the
space-group modes of `crystod-group` — comes from the bundled ISO-IR tables of
the ISOTROPY Software Suite (Miller-Love convention: `GM4-`, `R5-`, `DT5`,
`LD3`). The small irreps themselves are computed by spgrep, and each one is
named by comparing its characters with the tabulated characters at its k
point.

A label is defined only relative to a description of the crystal in the
ISO-IR standard setting, and the setting leaves a choice. Moving the origin to
another point of the same site symmetry (Sr or Ti at the origin of Pm-3m, Si
on Wyckoff 8a or 8b of Fd-3m) or turning the axes by a lattice rotation that
maps the space group onto itself (the sixfold axis of P-6m2, the twofold axis
that reverses the polar axis of P4) gives another standard description — an
element of the Euclidean normalizer of the group — and it permutes labels at
zone-boundary points: the octahedral tilt of SrTiO3 is `R5-` with Sr at the
origin and `R4+` with Ti at the origin. CrystOD fixes the frame by three
rules, so that one structure gets one set of names in every command:

1. **A cell that is already in the ISO-IR setting keeps its own axes and
   origin**: the conventional cell, its primitive cell in the standard
   centring, and a diagonal supercell of either (n1 x n2 x n3 multiples of its
   axes with the same origin, of any size). The setting chosen in the input
   file is respected.
2. **Any other cell gets a canonical frame of the crystal** (a shifted origin,
   rotated or re-based axes, a non-diagonal supercell such as sqrt2 x sqrt2 x
   1, a supercell of such a cell). The candidates are spglib's standard frame
   composed with the proper lattice rotations that map the group onto itself
   and with the origin shifts of the normalizer; frames that keep the cell's
   own origin are preferred when there are any (when that origin is the
   origin of a standard description, such as Sr or Ti in SrTiO3); among the
   candidates the one with the smallest sorted list of atomic positions is
   taken, coordinates being compared, and reduced into the unit cell, at
   0.71, 0.071 and 0.0071 times the symmetry tolerance (not round multiples
   of it, so that coordinates given to a few decimals cannot fall exactly on
   a level). Along a polar axis, where the origin is free, it is put on an
   atom of the species with the smallest atomic number (centring copies
   included).
3. **A working cell derived from the input cell** (phonopy's primitive cell,
   the spglib standardization) **inherits the frame of the input cell**.

| input | rule | labels |
|---|---|---|
| `221_PPOSCAR_SrTiO3` (Sr at the origin) | 1 | R tilt `R5-` |
| the same with Ti at the origin | 1 | R tilt `R4+` |
| the same shifted by (0.1234, 0.2345, 0.3456) | 2 | R tilt `R4+` (Ti at the origin) |
| `221_PPOSCAR_ScF3` with the empty cube centre at the origin | 1 | R spaces `R2-`, `R3-`, `R4-`, `R5+`, `R5-` (Sc at the origin: `R1+`, `R3+`, `R4+`, `R4-`, `R5+`) |
| 2x2x2 and 2x1x1 supercells of `221_PPOSCAR_SrTiO3` | 1 | Ti d at R `R3- + R4-`, as the cell |
| 2x2x1 supercell of MoS2 (P-6m2) | 1 | Mo d at M `M1..M4` and at L `L1..L4`, as the cell |
| sqrt2 x sqrt2 x 1 supercell of `221_PPOSCAR_SrTiO3` (rows (1,1,0), (-1,1,0), (0,0,1)), Sr at the origin | 2, own origin kept | R tilt `R5-` |
| the same supercell shifted by (0.1234, 0.2345, 0.3456) | 2 | R tilt `R4+` |
| `131_PPOSCAR_CuO` (P4_2/mmc) and its 1x1x2 supercell | 1 | Cu p at A `A1 + A2 + A4`, at R `R1- + 2 R2- + R3- + 2 R4-` |
| sqrt2 x sqrt2 x 1 supercell of `131_PPOSCAR_CuO`, shifted or not; the shifted cell | 2 | Cu p at A `A1 + A2 + A3`, at R `2 R1+ + R2+ + 2 R3+ + R4+` |

Inputs whose origin is a general point (shifted copies of the crystal) are
therefore labelled in one frame, whatever their shift, orientation, basis or
cell size. An input that is not in the ISO-IR setting but has its origin at
the origin of a standard description keeps that origin under rule 2 and is
named accordingly, so it and a shifted copy of it can name a mode
differently: the unshifted sqrt2 x sqrt2 x 1 supercell of SrTiO3 has the
Sr-origin tilt `R5-`, its shifted copy `R4+`. A diagonal supercell of a cell
in the ISO-IR setting keeps that cell's axes and origin and is labelled like
the cell; a non-diagonal supercell is labelled by rule 2 and can get names
that differ from the cell's by a normalizer element (`A3` for `A4` and `R+`
for `R-` in CuO). Two inputs that are both in the ISO-IR setting but differ by
a normalizer element (Sr or Ti at the origin) name the same mode differently,
so compare labels between inputs given in the same setting.

### k points: star arms, k + G, lines and -k

Any arm of a star and any copy k + G of a k point carry the same name and
labels. On a symmetry line or plane the ISO-IR label depends on the value of
the line parameter, and one k point has one value per arm and per copy k + G.
The label is evaluated at the canonical value, the smallest parameter (the
positive one when the two signs are equally small), so a k point gets the
same label however it is written: `crystod-group --table --sg 216 --kpoint
-1/4 -1/4 0` and `--kpoint 3/4 3/4 0` print the same rows (`DT4` with the
characters 1, -1, -1, 1), because DT of F-43m at the conventional (0,0,-1/2)
is evaluated on the arm (0,0,-a) at a = 1/2. On a line that no symmetry
element reverses (LD of P2_1, DT of P6_3) the two signs belong to different
stars, k and its -k partner (next paragraph).

In a group without inversion the star of -k can differ from the star of k.
At a special point ISO-IR tabulates one of the two (P of I-4, not PA). On a
line, a plane or the general point the ISO-IR entry covers both signs of the
parameter when -k lies on the same line or plane at the opposite parameter
(LD of P2_1 and P4, DT of P6_3, GP), and CrystOD gives the ISO-IR name to the
star at the canonical parameter (the smallest, the positive one first): that
is the name ISOTROPY gives when that parameter value is entered, its
parametrization of every line and plane being the same as that of ISO-IR.
Where -k lies on another line (the line P of P3 through K and H, whose -k
partner runs through K' and H'), both signs of the parameter of P keep the
name P. The other star, the -k partner, is named as the ISOTROPY software
names it in its physically irreducible labels (`P1PA1`, `LD1LE1`, `DT6DU6`,
`GP1GQ1`): LE for LD, DU for DT, SN for SM, GQ for GP, the suffix A for the
other letters (PA, KA, HA, WA, BA, ...), and the suffix C, as ISOTROPY uses
it, for P, B, C, D and E in P3, P3_1 and P3_2, for P and C in P31m and P31c,
for D in P3m1 and P3c1 and for B and E in P-6. The partner irreps carry the
numbers of the tabulated ones and are their complex conjugates: `PA1` is the
complex conjugate of `P1`, `LE1` that of `LD1`. Over all 230 groups these
partners are exactly the k types whose -k lies outside the star.
`crystod -c 82_PPOSCAR_AlPO4 --element Al --orbital p --kpoint -0.25 -0.25
-0.25` prints the k point `PA` and `1.0 [PA2(1)] + 1.0 [PA3(1)] + 1.0
[PA4(1)]`; `crystod-group --table --sg 173 --kpoint 0 0 7/12` prints the k
point `DU` of P6_3 with the rows `DU2`, `DU1`, `DU6`, ..., the conjugates of
`DT2`, `DT1`, `DT6`, ... at (0,0,5/12). `crystod-group --product`, `--parent`
and `--supergroup-cif` use the same names (`M1 x P1 = PA1` in I-42d,
`P1 x X1 = LD1 + LD2 + LE1 + LE2` in I4, the pairs `P1PA1`, `LD1LE1`,
`GP1GQ1`).

Names and labels refer to one frame. The special points of the ISO-IR
tables that a command lists or surveys (`crystod` and `--atomic-orbital`,
`--diagram`, `--visualize` without `--kpoint`, `crystod-phonon --irreps`,
`--vector` and `--subgroup`, the `crystod-mag` survey, and the MCP tools
`crystod_phonon_irreps` and `crystod_crystal_orbital_irreps`) are named in
the frame of the labels, a point that is there the -k partner of a
tabulated one being replaced by its -k; a name given to them (`--kpoint P`,
`--qpoint P`) is resolved in that frame; and coordinates given on the
command line are named by the same labeller. An input outside the ISO-IR
setting can differ from spglib's basis by a normalizer element that turns P
into PA: `82_PPOSCAR_AlPO4` shifted by (0.0731, 0.1593, 0.2417) lists
`P [-0.25, -0.25, -0.25]` above `1.0 [P1(1)] + 1.0 [P2(1)] + 1.0 [P3(1)]`
(Al p), analyzes that point for `--kpoint P`, and names (1/4, 1/4, 1/4)
`PA`, with the labels `PA1`, `PA2`, `PA3`; a species-sorted 2x2x2 supercell
of `143_PPOSCAR_HgBr`, which spglib re-bases to (-b, -a, -c), lists `H` with
`H1 + H2 + H3` as the cell does. Negative coordinates can be given back as
printed, as decimals or as fractions (`--kpoint -1/4 -1/4 -1/4`).

The seekpath lists (`crystod-phonon --vibration`, `crystod-mag --qpoint`,
`crystod --star-of-k` and `--visualize --kpoint`) keep seekpath's names
(`GAMMA`, `H_2`, ...). A seekpath point to which the frame of the labels
gives another ISO-IR type than spglib's frame, in which seekpath names it,
is moved to the point that has that type (the seekpath `P` of the shifted
AlPO4 above is listed at (-1/4, -1/4, -1/4), with P labels), and the
coordinates of a listed point are named as the list names them. The k-path
of `crystod-phonon --irreps --all-irreps` cannot move its endpoints; an
endpoint of that kind is printed with its ISO-IR name (`GM-X-PA-N-...` for
the shifted AlPO4). seekpath's letters are not the ISO-IR names in every
lattice: for a triclinic cell its X, Y, Z, R, T, U and V denote other points
than the ISO-IR letters do. The seekpath `Z` of `1_PPOSCAR_RbBe2F5` is
(-1/2, 0, 0), whose vibrations carry the label `X1`, while `crystod --kpoint
Z` analyzes the ISO-IR Z, (0, 0, 1/2), with `Z1`; give such a point by its
coordinates when the two commands are compared.

### Limits of the frame rules

- Rule 2 keeps the cell's own origin when that origin is the origin of a
  standard description. An unshifted description with rotated or re-based
  axes, or a non-diagonal supercell, can therefore name a mode differently
  from a shifted copy of itself: the sqrt2 x sqrt2 x 1 supercell of SrTiO3
  has the tilt `R5-` with Sr at the origin and `R4+` when shifted. The labels
  agree among descriptions whose origin is a general point.
- Rule 1 follows the axes of the input, including the sense of a polar axis:
  a P4 cell and the same cell with the polar axis reversed, both in the ISO-IR
  setting, exchange the complex pair `GM3`/`GM4`. It also accepts the
  (b, -a, c) setting of the C-centred orthorhombic groups whose symbol that
  exchange leaves unchanged (C222_1, Cmm2, Ccc2, Cmmm and Cccm, SG 21, 35,
  37, 65 and 66): the supercell (a+b, -a+b, c) of the conventional cell is a
  2x2x1 supercell of the primitive cell of that setting, and its names follow
  those axes (`GM3` and `GM4`, for example, exchanged against the standard
  setting).
- A P1 cell is in the ISO-IR setting whatever its axes and origin, and a P-1
  cell whatever its axes when its origin is an inversion centre (rule 1), so
  their k-point names follow the axes of the input. A non-diagonal supercell
  of a P1 cell and a P-1 cell whose origin is a general point take rule 2.
- Near the symmetry tolerance an accidental lattice symmetry (a metric within
  the tolerance of a higher one) can enter or leave the candidates of rule 2
  and switch the canonical frame.
- An orthorhombic group whose symbol is invariant under a cyclic permutation
  of the axes (Pbca) with a = b exactly can still be labelled differently for
  different descriptions: that permutation is not a rotation of the lattice
  and is not among the candidates.

## Multi-electron terms: symmetrized products and the Pauli principle

### Coulomb multiplet energies: Gaunt coefficients, Racah parameters and configuration interaction

The two-electron integrals behind the multiplet energies of
`crystod-group --multiplet --orbital` (section 21 of the `crystod-group`
page) are built from exact Gaunt coefficients (F^0 = A + 7C/5,
F^2 = 49B + 7C, F^4 = 63C/5), the Coulomb Hamiltonian over the Slater
determinants of the configuration, and each term is isolated by S^2 and
point-group projectors; doubly-occurring terms mix (configuration
interaction) and their two energies are printed in closed form
(e.g. 3A - 3B + 3C +- 3sqrt(2)B in (t2g)^2(eg)^1). Every run is closed by
the trace identity sum (2S+1) dim E = tr(H_ee); the free-ion limit is
reproduced exactly ((T1u)^2 with --orbital p gives ^3P = F0 - 5F2,
^1D = F0 + F2, ^1S = F0 + 10F2).

### The Pauli principle in multi-electron terms: antisymmetrized squares and CI matrices

Of the plain direct product T2g x T2g = A1g + Eg + T1g + T2g (as printed by
`crystod-group --product`, section 8 of the `crystod-group` page), the Pauli
principle pairs only the antisymmetric square (T1g) with the spin
triplet — `crystod-group --multiplet` performs this antisymmetrization
exactly, for any filling of any shell (hole equivalence and closed shells come out
automatically: (t2g)^4 gives the (t2g)^2 terms, (t2g)^6 gives ^1A1g).
Validated against the standard crystal-field term tables (t2g^n, eg^n,
t2g^2 eg^1 in Oh; e^2 in Td and C3v).

For two-shell configurations, the doubly-occurring terms (the CI pairs) are additionally printed as their **CI matrix in the coupled-parent basis** |shell1(S1 Gamma1) shell2(S2 Gamma2)> — the representation used by the Tanabe-Sugano/Griffith strong-field tables — e.g. for (t2g)^2(eg)^1: the ^2Eg block is <t2g^2(^1A1g)eg|H|...> = 3A + 8B + 6C, <t2g^2(^1Eg)eg|H|...> = 3A - B + 3C, off-diagonal ±10B, whose eigenvalues are exactly the printed 3A + (7/2)B + (9/2)C ± (1/2)√(481B² + 54BC + 9C²) (the off-diagonal sign is a basis convention; books may differ).
