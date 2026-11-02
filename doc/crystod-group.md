# crystod-group

The point/space-group representation-theory calculator. One mode flag per task:
`--product`, `--jahn-teller`, `--table`, `--decompose`, `--ligand-field`, `--basis`,
`--generate-basis`, `--tensor`, `--coset`, `--correlate`, `--parent`, `--multiplet`, `--poscar2cif`,
`--cif2poscar`, `--cif-axis`, `--supergroup-cif`; `--parent` also takes `--invariants`,
`--secondary`, `--child` and `--graph`.
Point groups are selected with `--pg`/`--pointgroup`/`--point-group` and
space groups with `--sg`/`--spacegroup`/`--space-group`, by symbol or by
number (labels starting with `-`, such as `-43m`, are accepted). No structure
file is needed.

| I want to ... | command |
|---|---|
| multiply irreps | `crystod-group --product T2g T2g T1u --pg m-3m` |
| split a square into its symmetric and antisymmetric parts | `crystod-group --product T2g T2g --pg m-3m --symmetric --antisymmetric` |
| find the Jahn-Teller active modes of a degenerate level | `crystod-group --jahn-teller Eg --pg m-3m` |
| see a character table | `crystod-group --table --pg 3m` |
| split an orbital in a crystal field | `crystod-group --ligand-field d --pg m-3m` |
| classify basis functions | `crystod-group --basis x y z --pg m-3m` |
| get the allowed form of a property tensor (piezoelectric, elastic, Raman, ...) | `crystod-group --tensor piezoelectric --pg 4mm` |
| get the Raman tensor of every Raman-active irrep | `crystod-group --tensor raman --pg m-3m` |
| see which Gamma modes of the low-symmetry phase a parent irrep becomes | `crystod-group --correlate --parent Pm-3m --irrep R4+ --order-parameter 0 0 a` |
| get the compatibility relations along a symmetry line | `crystod-group --correlate --sg Pm-3m --kpoint GM X` |
| get the correlation table of a point group and a subgroup | `crystod-group --correlate --pg m-3m --subgroup 4/mmm` |
| know which subgroup a distortion gives | `crystod-group --parent Pm-3m --irrep R4+` (`--supergroup` = same) |
| list the subgroups of every irrep at a k point | `crystod-group --parent Pm-3m --kpoint GM` |
| list the Landau free-energy terms of an irrep | `crystod-group --parent Pm-3m --irrep R4+ --invariants` |
| check whether a transition can be continuous (Landau and Lifshitz conditions) | `crystod-group --parent Pm-3m --irrep M1+ --invariants --degree 3` |
| get the free energy along one order-parameter direction | `crystod-group --parent Pm-3m --irrep R4+ --invariants --order-parameter a b 0` |
| find the coupling term of several order parameters | `crystod-group --parent I4/mmm --irrep X2+ X3- GM5- --invariants --degree 3` |
| list the secondary order parameters of a distortion | `crystod-group --parent Pm-3m --irrep R4+ --order-parameter a 0 0 --secondary` |
| find the irreps that give a known subgroup | `crystod-group --parent Pm-3m --child I4/mcm` |
| find the pairs of irreps that give a known subgroup | `crystod-group --parent Pm-3m --child Pnma --size 4 --coupled` |
| draw the group-subgroup graph of the isotropy subgroups | `crystod-group --parent Pm-3m --irrep R4+ M3+ --graph` |
| get the term symbols of a configuration | `crystod-group --multiplet T2g2 --pg m-3m --orbital d` |
| convert POSCAR <-> CIF | `crystod-group --poscar2cif -c POSCAR` |
| see which axis setting my CIF is in, and rewrite it in another | `crystod-group --cif-axis -c FILE.cif`, then `crystod-group --cif2poscar -c FILE.cif --spacegroup-set Amam` |
| decompose an observed distortion | `crystod-group --supergroup-cif HIGH.cif --subgroup-cif LOW.cif` |

## 8. Direct products (`--product`)

*Example directory: `example/08_direct_product` (testsuite section 8); after pip install: `crystod-example 08`*

### 8.1 Point-group irreps (`--pg`/`--point-group`)

```bash
crystod-group --product T2g T2g T1u --point-group m-3m
```

```
* Point group *
m-3m

* Direct product *
T2g*T2g*T1u

* Result *
 1(A1u) + 1(A2u) + 2(Eu) + 4(T1u) + 3(T2u)
```

(`--product T2g T2g` alone gives `1(A1g) + 1(Eg) + 1(T1g) + 1(T2g)`; its
symmetric and antisymmetric parts are printed with `--symmetric` and
`--antisymmetric`, section 9, and are the singlet and triplet terms of
`--multiplet T2g2`, section 21.)

### 8.2 Space-group irreps (`--sg`/`--space-group`)

Decompose the direct product of **full space-group irreps** (the irreps at
high-symmetry k points, induced over their whole star) into full space-group
irreps with ISO-IR (ISOTROPY, Miller-Love) labels:

```bash
crystod-group --product R4- R5+ --sg Pm-3m
```

```
...
* Direct product (full space-group irreps) *
R4- x R5+ = GM2- + GM3- + GM4- + GM5-

* Dimension check (star size x small dim) *
3 x 3 = 9 -> 1 + 2 + 3 + 3 = 9
```

The k points of the factors may differ and three or more factors are accepted.
The wavevector selection rule k1 + k2 = k3 (mod reciprocal lattice) over the
star arms decides which stars appear, and the dimension check under the result
verifies the reduction. Products landing on symmetry lines outside the
tabulated special points (DT, SM, LD, ...) are decomposed with on-the-fly
`spgrep` small irreps and marked `[non-tabulated]`. When the stars of k and -k
are distinct (a line that no symmetry element reverses), the star at the
positive line parameter keeps the ISO-IR name and the other one carries the
name ISOTROPY gives the -k partner, with the conjugate irreps
(`P1 x X1 = LD1 + LD2 + LE1 + LE2` in I4: `LE1` is the conjugate of `LD1`;
DU for DT, SN for SM, GQ for GP, an `A` suffix for the other letters, and a
`C` suffix for some lines and planes of the trigonal groups and P-6, such as
PC of P3).
At a special point whose -k star is not tabulated (PA of I-4, KA and HA of
P3) and on such lines `crystod`, `--atomic-orbital`, `crystod-phonon` and
`crystod-mag` use the same names (`PA1` is the complex conjugate of `P1`);
see
[Irrep labels and the frame of a structure](theory-representations.md#irrep-labels-and-the-frame-of-a-structure).

This is the offline counterpart of the **DIRPRO** program of the Bilbao
Crystallographic Server, cross-validated against it line by line (9007
products, 20 space groups covering all Bravais classes; see
`example/08_direct_product/README`). If you use this feature in a
publication, please cite M. I. Aroyo, A. Kirov, C. Capillas, J. M. Perez-Mato
and H. Wondratschek, *Acta Cryst.* **A62**, 115-128 (2006), and the ISO-IR
tables: H. T. Stokes, B. J. Campbell and R. Cordes, *Acta Cryst.* **A69**,
388-395 (2013).

### 8.3 Character tables for point groups (`--table --point-group`)

```bash
crystod-group --table --point-group 3m
```

```
* IrRep Table *
table:
irrep  E(1)  C3(2)  sgv(3)
   A1     1      1       1
   A2     1      1      -1
    E     2     -1       0
```

The columns are the conjugacy classes with their sizes in parentheses. The
same table is printed alongside a decomposition with `--show-irrep-table`:

```bash
crystod-group --product T2g T2g T1u --point-group m-3m --show-irrep-table
```

### 8.4 Character tables for space groups (`--table --space-group`)

With `--space-group` and a `--kpoint`, the characters of the **small irreps of
the little group** at that k point are tabulated — the space-group counterpart
of the point-group table above:

```bash
crystod-group --table --space-group Pm-3m --kpoint 0 0.5 0.5
```

```
* Space group *
Pm-3m (221)

* k-point (primitive) *
 M [0.0, 0.5, 0.5]

* IrRep Table *
little group: P4/mmm (123)
table:
               irrep  1  2_100  2_010  2_001  4^+_100  4^-_100  2_011  2_01-1  -1  m_100  m_010  m_001  -4^+_100  -4^-_100  m_011  m_01-1
 irrep_1(1) = M1+(1)  1      1      1      1        1        1      1       1   1      1      1      1         1         1      1       1
 irrep_2(1) = M1-(1)  1      1      1      1        1        1      1       1  -1     -1     -1     -1        -1        -1     -1      -1
 irrep_3(1) = M3+(1)  1      1     -1     -1        1        1     -1      -1   1      1     -1     -1         1         1     -1      -1
 irrep_4(1) = M3-(1)  1      1     -1     -1        1        1     -1      -1  -1     -1      1      1        -1        -1      1       1
 irrep_5(2) = M5+(2)  2     -2      0      0        0        0      0       0   2     -2      0      0         0         0      0       0
 irrep_6(2) = M5-(2)  2     -2      0      0        0        0      0       0  -2      2      0      0         0         0      0       0
 irrep_7(1) = M2+(1)  1      1      1      1       -1       -1     -1      -1   1      1      1      1        -1        -1     -1      -1
 irrep_8(1) = M2-(1)  1      1      1      1       -1       -1     -1      -1  -1     -1     -1     -1         1         1      1       1
 irrep_9(1) = M4+(1)  1      1     -1     -1       -1       -1      1       1   1      1     -1     -1        -1        -1      1       1
irrep_10(1) = M4-(1)  1      1     -1     -1       -1       -1      1       1  -1     -1      1      1         1         1     -1      -1
```

Note that `(0, 1/2, 1/2)` is a non-representative arm of the M star: the
header names it `M` and the rows carry the ISO-IR labels transported from the
tabulated arm (`irrep_N` is the internal `spgrep` name, `MN+/-` the physical
label). Unlike a point-group table the columns are individual symmetry
operations in Seitz notation, not classes, because at a general k point the
Bloch phases of a class need not coincide. These are exactly the characters
against which the SALC, phonon, and spin analyses are reduced.

Any coordinates of a k point are accepted, and every spelling of one k point
gets the same labels. On a symmetry line the label is evaluated at the
canonical value of the line parameter (the smallest one, positive when the two
signs are equally small): `--kpoint -1/4 -1/4 0` and `--kpoint 3/4 3/4 0` of
F-43m print the same table, whose row with the characters 1, -1, -1, 1 is
`DT4`, while at `--kpoint 1/4 1/4 0` that row is `DT3`. A k point whose star
ISO-IR tabulates only through -k is named with an `A` suffix and its rows are
the complex conjugates of the tabulated ones (`--sg 82 --kpoint -1/4 -1/4 -1/4`
prints `PA` with the rows `PA1`, `PA4`, `PA2`, `PA3`). On a line that no
symmetry element reverses the star at the negative parameter is the -k partner
of the star at the positive one and carries ISOTROPY's partner name with the
conjugate rows (`--sg 173 --kpoint 0 0 7/12` of P6_3 prints `DU` with the rows
`DU2`, `DU1`, `DU6`, ..., the conjugates of `DT2`, `DT1`, `DT6`, ... at
(0,0,5/12)); see
[Irrep labels and the frame of a structure](theory-representations.md#irrep-labels-and-the-frame-of-a-structure).

## 9. Symmetrized squares and Jahn-Teller modes (`--symmetric`, `--antisymmetric`, `--jahn-teller`)

*Example directory: `example/09_symmetric_square` (testsuite section 9); after pip install: `crystod-example 09`*

`--symmetric` and `--antisymmetric` split the square of one irrep into its two
parts, and `--jahn-teller` reads the vibronic selection rules off them. They
answer which quadratic forms, strains or two-electron terms an irrep carries,
and which vibrations can distort a degenerate electronic level: questions of
ligand-field, Jahn-Teller and Landau analyses. The symmetric part `[IR x IR]`
has the characters `(chi(g)^2 + chi(g^2))/2` and holds the quadratic forms of
the components, the vibronic couplings and the orbital parts of two-electron
singlets; the antisymmetric part `{IR x IR}`, with `(chi(g)^2 - chi(g^2))/2`,
holds the two-electron triplets.

### 9.1 Point groups

```bash
crystod-group --product T2g T2g --pg m-3m --symmetric --antisymmetric
```

```
* Point group *
m-3m

* Direct product *
T2g*T2g

* Result *
 1(A1g) + 1(Eg) + 1(T1g) + 1(T2g)

* Symmetric square *
[T2g x T2g] = A1g + Eg + T2g
dimension: 6 = 1 + 2 + 3

* Antisymmetric square *
{T2g x T2g} = T1g
dimension: 3 = 3
```

How to read it:

- `* Result *` is the plain product of section 8.1; each flag adds one block.
  The two irreps given to `--product` must be the same.
- `[T2g x T2g]` and `{T2g x T2g}` are the symmetric and antisymmetric parts;
  the `dimension:` line checks that they add up to n(n+1)/2 and n(n-1)/2.
- The two parts are the spin singlets and triplets of two electrons in the
  level: `--multiplet T2g2` (section 21) gives `^3T1g` from the antisymmetric
  square and `^1A1g + ^1Eg + ^1T2g` from the symmetric one.
- The point-group tables list a pair of complex-conjugate irreps as one real
  irrep (`E` of 3, 4, 6, `Eg` of m-3, ...), counted once per real copy:
  `--product E E --pg 3` gives `2(A) + 1(E)`, with `[E x E] = A + E` and
  `{E x E} = A`.

### 9.2 Jahn-Teller and pseudo-Jahn-Teller modes

A vibration of symmetry Gamma_Q couples linearly to a degenerate level Gamma
when Gamma_Q occurs in `[Gamma x Gamma]`; the totally symmetric part does not
lower the symmetry, so the **Jahn-Teller active** modes are `[Gamma x Gamma]`
without the identity irrep:

```bash
crystod-group --jahn-teller Eg --pg m-3m
```

```
* Point group *
m-3m

* Jahn-Teller active modes *
[Eg x Eg] = A1g + Eg
JT-active: Eg
  (linear vibronic coupling allowed by symmetry; the Jahn-Teller theorem)
```

This is the E x e problem of an octahedral Eg level (the Q2, Q3 modes);
`--jahn-teller T1u --pg m-3m` gives `JT-active: Eg + T2g`, and a
nondegenerate level prints `JT-active: none`. For one degenerate level the
answer is the Jahn-Teller theorem: the linear vibronic coupling to the listed
modes is allowed by symmetry.

With two irreps the output lists the **symmetry-allowed pseudo-Jahn-Teller
coupling modes**: a mode Q can mix the two levels when Gamma_Q occurs in the
plain product, and the non-totally-symmetric irreps of the product are listed:

```bash
crystod-group --jahn-teller Eg T2g --pg m-3m
```

```
* Point group *
m-3m

* Pseudo-Jahn-Teller coupling *
Eg x T2g = T1g + T2g
symmetry-allowed coupling modes: T1g + T2g
  (necessary condition only: a mode Q can mix the two levels when Gamma_Q
   occurs in the product; whether the mixing destabilizes the high-symmetry
   structure depends on the energy gap and the coupling strength, which
   symmetry does not give)
```

How to read it:

- `JT-active` (one irrep) is `[Gamma x Gamma]` without the identity irrep:
  the modes that couple linearly to the degenerate level.
- `symmetry-allowed coupling modes` (two irreps) are the non-totally-symmetric
  irreps of the plain product: the modes that can mix the two levels. This is
  a necessary condition only; whether the mixing makes the high-symmetry
  structure unstable depends on the energy gap and the vibronic coupling
  constant, which symmetry does not give (I. B. Bersuker, *The Jahn-Teller
  Effect*, Cambridge University Press, 2006).
- With two identical labels (`--jahn-teller T1u T1u --pg m-3m`) an extra line
  gives the first-order Jahn-Teller modes of that level,
  `same degenerate level (first-order Jahn-Teller): [T1u x T1u] - A1g = Eg + T2g`,
  and the allowed-modes line is labelled `two different levels of symmetry
  T1u, ...`: the antisymmetric part (T1g here) can couple two distinct levels
  of that symmetry but never a level with itself.

`--jahn-teller` works with point groups (the local symmetry of a site or a
molecule) and with spinless levels. Whether a mode of the listed symmetry
exists depends on the structure: `crystod-phonon --vibration` gives the
vibrational irreps of a crystal (section 34 of
[crystod-phonon](crystod-phonon.md)).

### 9.3 Space groups

With `--sg`, the squares of a full space-group irrep are reduced into the full
irreps at the stars of `k_a + k_b` (sums of two arms of the star), with the
ISO-IR labels of section 8.2:

```bash
crystod-group --product R4+ R4+ --sg Pm-3m --symmetric --antisymmetric
```

```
* Space group *
Pm-3m (No. 221)

* K points (primitive basis) *
R: (1/2, 1/2, 1/2)   star of 1 arm(s)
GM: (0, 0, 0)   star of 1 arm(s)

* Direct product (full space-group irreps) *
R4+ x R4+ = GM1+ + GM3+ + GM4+ + GM5+
...
* Symmetric square (full space-group irreps) *
[R4+ x R4+] = GM1+ + GM3+ + GM5+
dimension: 6 = 1 + 2 + 3

* Antisymmetric square (full space-group irreps) *
{R4+ x R4+} = GM4+
dimension: 3 = 3
...
```

How to read it:

- For R, `2k` is a reciprocal-lattice vector, so every term is at Gamma: the
  strains the octahedral tilt drives (GM3+ and GM5+, see `--secondary` in
  section 18) and the single quadratic invariant (GM1+, `I2_1` of section 18).
- For a star of several arms the square also has terms at the sums of two
  arms: `[X5+ x X5+]` of Pm-3m has `GM1+ + GM2+ + 2GM3+ + GM5+` at Gamma and
  `M1+ + M4+ + M5+` at M (X + X' = M).
- A complex-type irrep is squared in its physically irreducible real form
  `D + D*`, which a `note:` line names (`--product P1 P1 --sg I4/mcm
  --symmetric` squares `P1P3`, dimension 4). The identity irrep then also
  occurs in the antisymmetric square, since the real form has an invariant
  antisymmetric form.

**Limits.** `--jahn-teller` takes point groups only, and both modes treat
spinless (single-valued) irreps. With `--sg` the squared irrep must be a
tabulated irrep of a special k point.

**Python API.** The point-group squares are `{irrep: multiplicity}`
dictionaries (zeros included), the space-group square a
`SquareDecomposition`; there is no MCP tool for this section:

```python
from crystod import group

ct = group.get_character_table("m-3m")
group.symmetric_square(ct, "m-3m", "T2g")       # {'A1g': 1, ..., 'Eg': 1, 'T2g': 1, ...}
group.antisymmetric_square(ct, "m-3m", "T2g")   # {..., 'T1g': 1, ...}
group.jahn_teller_modes("m-3m", "Eg")            # JahnTellerModes(active={'Eg': 1}, ...)
group.SpaceGroupIrrepAlgebra("Pm-3m").decompose_square("R4+")   # SquareDecomposition
```

## 10. Representation decomposition (`--decompose`)

*Example directory: `example/10_decompose_irrep` (testsuite section 10); after pip install: `crystod-example 10`*

Decompose a reducible representation into the irreps of a point group by entering
its characters class by class (educational / hand-analysis companion to `--product`):

```bash
crystod-group --decompose --point-group 3m
```

```
* Point group *
3m

* Reducible representation *
1E: 3
2C3: 0
3sgv: 1

* Result *
1(A1) + 1(E)
```

The characters can also be given at once for non-interactive use:

```bash
crystod-group --decompose --point-group 3m --characters 3 0 1
```

The class order and multiplicities follow the prompt (e.g. `1E`, `2C3`, `3sgv` for
3m); the character table can be checked with `crystod-group --table`.

Characters that are not those of a representation are refused, not rounded
into an answer. The reduction formula has to give non-negative integer
multiplicities that reproduce the characters entered; otherwise the command
prints an `ERROR` (on stderr) in place of the `* Result *` block and exits
with status 1, in the interactive mode as well:

```bash
crystod-group --decompose --point-group 3m --characters 2 0 1
```

```
* Point group *
3m

* Reducible representation *
1E: 2
2C3: 0
3sgv: 1
ERROR: the characters 2 0 1 are not those of a representation of 3m (classes 1E, 2C3, 3sgv):
the reduction formula gives non-integer multiplicities A1 0.833, A2 -0.167, E 0.667; the nearest integer combination 1(A1) + 1(E) has the characters 3 0 1, not 2 0 1.
Check the values and their class order (crystod-group --table --pg 3m).
```

Non-integer multiplicities usually mean a typing error or characters entered
in another class order (`--pg=-4 --characters 5 1 -1` gives A 1.500, B 0.500,
E 1.500). Negative ones mean a difference of representations:
`--pg 3m --characters 0 0 2` is A1 - A2 and is refused with
`negative multiplicities A2 -1 (a difference of representations)`.
In Python, `crystod.group.decompose` still rounds without this check; the
check itself is `crystod.decompose_irrep.representation_problem`
([Python API](python-api.md)).
Based on `script/decomose_to_irreps.py` by Hiroki Koiso.

## 11. Ligand-field splitting (`--ligand-field`)

*Example directory: `example/11_ligand_field_split` (testsuite section 11); after pip install: `crystod-example 11`*

Decompose an atomic orbital (s, p, d, f, g, h, i) into the irreps of a point
group — the crystal-field / ligand-field splitting of the orbital in the given
point-symmetric environment:

```bash
crystod-group --ligand-field d --point-group m-3m
```

```
* Point group *
m-3m

* Orbital *
d

* Reducible representation of the d orbital in the m-3m field *
1E: 5
8C3: -1
6C2: 1
6C4: -1
3C4^2: 1
1i: 5
6S4: -1
8S6: -1
3sgh: 1
6sgd: 1

* Result *
1(Eg) + 1(T2g)
```

— the t2g/eg splitting of an octahedral field, with the characters it was
reduced from. Any shell up to i works, in any of the 32 point groups
(`--ligand-field f --point-group 4/mmm` gives `1(A2u) + 1(B1u) + 1(B2u) + 2(Eu)`).
In the ten point groups whose tables list a pair of complex-conjugate
one-dimensional irreps as one real two-dimensional irrep (4, -4, 4/m, 3, -3,
6, -6, 6/m, 23, m-3), such an irrep counts once per doubly degenerate real
level: `--ligand-field d --point-group -4` gives `1(A) + 2(B) + 1(E)`.

The characters of the (2l+1)-dimensional orbital representation are generated
from the standard angular-momentum formulas
(chi(C(a)) = sin((l+1/2)a)/sin(a/2), chi(S(a)) = cos((l+1/2)a)/cos(a/2)) and
decomposed with the same reduction engine as `--decompose`.
Based on `script/ligand_field_spliting.py` by Hiroki Koiso.

## 12. Basis functions (`--basis`)

*Example directory: `example/12_basis_function` (testsuite section 12); after pip install: `crystod-example 12`*

```bash
crystod-group --basis x y z --point-group m-3m
crystod-group --basis x y z --space-group Pm-3m --kpoint 0 0 0
crystod-group --basis xyz --space-group Pm-3m --kpoint 0.5 0.3 0 --show-irrep-table
crystod-group --basis "x(y^2-z^2)" --point-group="m-3m"
crystod-group --basis "x^2-y^2" "2z^2-x^2-y^2" xy yz zx --space-group="Pm-3m" --kpoint 0 0 0
```

The input functions are automatically closed under the selected point group or
the little group of the selected space-group k point, then decomposed into
irreps. The irreps carry ISO-IR labels at every k point — special points and
generic lines alike (`GM4-(3)` at GM; `Z2(1)` in the third example above).

Besides the polar coordinates x, y, z, the **axial-vector components
`Rx`, `Ry`, `Rz`** (rotations, angular momenta, magnetic moments) are supported;
they transform with `det(R) R` and therefore land in the parity partners of the
polar bases:

```bash
crystod-group --basis Rx Ry Rz --point-group m-3m                  # -> T1g (x y z gives T1u)
crystod-group --basis Rx Ry Rz --space-group Pm-3m --kpoint 0 0 0  # -> GM4+
crystod-group --basis "x*Ry - y*Rx" --point-group m-3m             # toroidal component -> T1u
```

The axial run shows the sign pattern responsible for the parity flip — the
rotation classes keep the polar characters, while every improper class
(i, S4, S6, sgh, sgd) changes sign relative to (x, y, z):

```
* Input basis functions *
 Rx, Ry, Rz

* Reducible characters *
  E: 3
  C3: 0
  C2: -1
  C4: 1
  C4^2: -1
  i: 3
  S4: 1
  S6: 0
  sgh: -1
  sgd: -1

* Decomposition *
 1.0 [T1g]

* Irreducible representations for basis functions *
  T1g: [Rx, Ry, Rz]
```

This makes the magnetic (spin) irreps directly comparable with the
`crystod-mag` labels (e.g. the GM4+ cluster dipole of AlNi3).

When `--kpoint` is omitted in space-group mode, all special k points of the
space group are analyzed automatically (as in the `crystod` SALC survey),
except the points of non-symmorphic groups described in the next subsection,
which are listed as skipped:

```bash
crystod-group --basis Rx Ry Rz --space-group Pm-3m
# -> GM4+(3), R4+(3), M3+(1) + M5+(2), X3+(1) + X5+(2)
```

### Bloch phases and non-symmorphic space groups

In space-group mode a polynomial f(x, y, z) about the origin stands for the
Bloch-modulated function exp(i k.r) f(r). A little-group operation {R|t}
acts on it as the rotation R times the Bloch phase exp(-2 pi i k.t), and the
`* Little-group characters *` block includes these phases, as the small-irrep
characters of `--table` do. At DT of the diamond structure the screw
rotations and glide reflections (fractional translation t) carry the factor
exp(-2 pi i k.t) = 0.9511-0.309j, and x, y, z decompose completely:

```bash
crystod-group --basis x y z --sg Fd-3m --kpoint 0.1 0 0.1
```

```
...
* Little-group characters *
  1: 3
  2_010: -1
  4^+_010: 0.9511-0.309j
  4^-_010: 0.9511-0.309j
  m_100: 0.9511-0.309j
  m_001: 0.9511-0.309j
  m_101: 1
  m_-101: 1

* Decomposition *
 1.0 [DT1(1)] + 1.0 [DT5(2)]

* Irreducible representations for basis functions *
  DT1(1): [y]
  DT5(2): [x, z]
```

The phases follow the multiplication of the little group only if
(kR1 - k).t2 is an integer for every rotation R1 and every translation t2 of
the little group, where R1 and t2 may come from two different operations
{R1|t1} and {R2|t2}. The product of these operations is
{R1R2|R1t2 + t1}, whose phase exp(-2 pi i k.(R1t2 + t1)) equals the product
of the two phases exp(-2 pi i k.t1) exp(-2 pi i k.t2) only when
k.(R1t2) - k.t2 = (kR1 - k).t2 is an integer. The condition
holds at GM, at every k point of a symmorphic group and at k points inside
the Brillouin zone. It fails at many zone-boundary points of non-symmorphic
groups, where the small irreps correspond to projective representations of
the little co-group. There the phase-carrying action on the polynomials does
not follow the group multiplication, so the polynomials span no
representation of the little group and cannot be decomposed into its small
irreps. Such a k point is refused instead of being given an incomplete
decomposition.

With `--kpoint` at such a point, the command prints only an `ERROR` (on
stderr) and exits with status 1. The message names a pair for which the
condition fails: the operation whose rotation R1 maps k to k + G, and the
operation whose translation t2 makes G.t2 a non-integer. These can be two
different operations. At X of Pnma, 2_010 supplies G and 2_100 supplies t:

```bash
crystod-group --basis x y z --sg Pnma --kpoint 0.5 0 0
```

```
ERROR: polynomial basis functions are not defined at k = [0.5, 0.0, 0.0] (X) of the non-symmorphic space group Pnma (62).
A polynomial in x, y, z about the origin is acted on by the rotation part R of each little-group operation {R|t}, times the Bloch phase exp(-2 pi i k.t). These phases follow the group multiplication only if (kR - k).t is an integer for every R and t of the little group, which holds at GM, at every k of a symmorphic group and inside the Brillouin zone, but not at this zone-boundary point. Here 2_010 maps k to k + G with G = (-1, 0, 0) and 2_100 carries t = (1/2, 1/2, 1/2) (primitive basis), so G.t = 1/2 (mod 1). The polynomials therefore span no representation of the little group, and a decomposition into its small irreps would be incomplete.
Use GM (--kpoint 0 0 0), a k point inside the zone, or a SALC of orbitals on the actual Wyckoff sites (crystod -c POSCAR --element EL --orbital ORB --kpoint ...).
```

At GM the same group decomposes as usual:

```bash
crystod-group --basis x y z --sg Pnma --kpoint 0 0 0
```

```
...
* Decomposition *
 1.0 [GM2-(1)] + 1.0 [GM4-(1)] + 1.0 [GM3-(1)]

* Irreducible representations for basis functions *
  GM2-(1): [z]
  GM4-(1): [y]
  GM3-(1): [x]
```

M of rutile is another such point:

```bash
crystod-group --basis x y z --sg P4_2/mnm --kpoint 0.5 0.5 0
```

stops with the same `ERROR` for M of P4_2/mnm (136). Here one operation,
2_100, supplies both G and t. The sentence that names it reads:

```
Here 2_100 maps k to k + G with G = (0, -1, 0) and 2_100 carries t = (1/2, 1/2, 1/2) (primitive basis), so G.t = 1/2 (mod 1).
```

Without `--kpoint`, the points of this kind are named on a `Skipped` line of
the `* Special k points *` block, and the other special points are analyzed.
In Pnma all seven points other than GM are skipped:

```bash
crystod-group --basis x y z --sg Pnma
```

```
* Special k points *
No --kpoint given; analyzing all special k points: GM, X, Y, Z, S, T, U, R
Skipped (non-symmorphic zone-boundary points where the polynomials span no representation of the little group): X, Y, Z, S, T, U, R

* Space group *
Pnma (62)
...
```

The output continues with the GM block shown above, and the command exits
with status 0. At such k points the symmetry of the functions is set by the
atoms on their Wyckoff sites, not by a polynomial about the origin. Use the
site-based Bloch sums of the SALC analysis instead,
`crystod -c POSCAR --element EL --orbital ORB --kpoint ...`
([section 2 of crystod](crystod.md#2-irreps-of-salc)). For the Ti p orbitals
of CaTiO3 (Pnma) at X it gives `3.0 [X1(2)] + 3.0 [X2(2)]`.

### Complex-type irreps

For a one-dimensional irrep with complex characters, such as GM3 and GM4 of
P4, the function listed under a label transforms with the characters of that
label in `--table`. Both use the active convention O_R f(r) = f(R^-1 r), as
the SALC analysis of `crystod` does:

```bash
crystod-group --basis x y z --sg P4 --kpoint 0 0 0
```

```
...
* Irreducible representations for basis functions *
  GM1(1): [z]
  GM3(1): [-Ix + y]
  GM4(1): [Ix + y]
```

`I` is the imaginary unit, so `-Ix + y` = -i(x + i y). The four-fold rotation
4^+_001 multiplies it by -i, which is the character of GM3 in
`crystod-group --table --sg P4 --kpoint 0 0 0`. A p shell at the origin of P4
gives the same SALC, GM3 = (p_x + i p_y)/sqrt2. `--generate-basis` lists
the same functions. CrystOD 0.4.3 and earlier printed the complex-conjugate
function under each label; the multiplicities and the point-group results
did not change.

## 13. Property tensors (`--tensor`)

*Example directory: `example/13_tensor_form` (testsuite section 13); after pip install: `crystod-example 13`*

`--tensor KIND` gives the symmetry-allowed form of a property tensor of a
crystal: which components of the dielectric, piezoelectric, elastic, Raman or
any other tensor can be nonzero, and how the others follow from them. It is
the offline counterpart of the TENSOR program of the Bilbao Crystallographic
Server and of the tables in J. F. Nye, *Physical Properties of Crystals*, for
anyone who sets up a measurement or checks a computed tensor. The group
theory is Neumann's principle: a property tensor is invariant under every
operation of the point group, so the allowed tensors are the invariants of
the tensor representation.

```bash
crystod-group --tensor piezoelectric --pg 4mm
```

```
* Point group *
  4mm (C4v), order 8, tetragonal, Laue class 4/mmm, non-centrosymmetric, polar
  source: point group 4mm
  axes: x || a, y || b, z || c
  4 || [001] (z)
  m perpendicular to [100] (x), [010] (y), [110], [1-10]

* Tensor *
  kind: piezoelectric strain coefficients d_ijk, P_i = d_ijk sigma_jk
  Jahn symbol: V[V2] (polar, time-reversal even)
  rank: 3
  intrinsic symmetry: symmetric in (jk)
  matrix notation (Nye): d_iL, L = 1..6 for jk = 11, 22, 33, 23, 13, 12; d_iL = d_ijk for L = 1-3 and 2 d_ijk for L = 4-6

* Independent components *
  3: d15, d31, d33

* Matrix form *
d  1    2    3    4    5    6
1  0    0    0    0    d15  0
2  0    0    0    d15  0    0
3  d31  d31  d33  0    0    0

* Relations *
  d24 = d15
  d32 = d31
  every other component is zero
```

The point group can also come from a space group (`--sg`, its default
setting) or from a structure (`-c`, spglib's standard conventional cell,
`--tolerance` default 1e-5):

```bash
crystod-group --tensor elastic --sg P6_3mc
```

```
* Point group *
  6mm (C6v), point group of P6_3mc (No. 186), order 12, hexagonal, Laue class 6/mmm, non-centrosymmetric, polar
  source: space group P6_3mc (No. 186)
  axes: x || a, z || c, y = z x x of the conventional cell
...
* Independent components *
  5: c11, c12, c13, c33, c44
...
* Relations *
  c22 = c11
  c23 = c13
  c55 = c44
  c66 = (c11-c12)/2
  every other component is zero
```

```bash
crystod-group --tensor piezoelectric -c 186_PPOSCAR_ZnO
```

```
* Point group *
  6mm (C6v), point group of P6_3mc (No. 186), order 12, hexagonal, Laue class 6/mmm, non-centrosymmetric, polar
  source: structure 186_PPOSCAR_ZnO (symprec 1e-05)
  axes: x || a, z || c, y = z x x of the standard conventional cell
  tensor axes in the Cartesian frame of the input file: x = (1.0000, 0.0000, 0.0000), y = (0.0000, 1.0000, 0.0000), z = (0.0000, 0.0000, 1.0000)
...
* Independent components *
  3: d15, d31, d33
...
```

KIND is one of the names below or any Jahn symbol (quote the brackets on the
command line, `--tensor "V[V2]"`):

| KIND | Jahn symbol | tensor | matrix notation |
|---|---|---|---|
| `dielectric` | `[V2]` | permittivity eps_ij (any symmetric polar second-rank property) | 3x3 |
| `pyroelectric` | `V` | pyroelectric coefficients p_i | vector |
| `piezoelectric` | `V[V2]` | strain coefficients d_ijk, P_i = d_ijk sigma_jk | 3x6, d_iL = 2 d_ijk for L = 4-6 |
| `elastic` | `[[V2]2]` | stiffness c_ijkl | 6x6, c_LM = c_ijkl |
| `compliance` | `[[V2]2]` | compliance s_ijkl | 6x6, factors 2 and 4 on shear indices |
| `gyration` | `e[V2]` | optical-activity gyration tensor (axial) | 3x3 |
| `raman` | `[V2]` | Raman tensor of each Raman-active irrep | per irrep |

Other Jahn symbols: `V` (polar vector), products such as `V[V2]` or
`[V2][V2]`, symmetric powers `[X n]`, antisymmetric powers `{X n}`, plain
powers `V2`, `V3`; the prefix `e` makes the tensor axial (an extra factor
det R) and `a` odd under time reversal. Up to rank 6.

How to read it:

- `* Point group *` names the group and its axes: x, y, z along a, b, c for
  the orthogonal lattices; x || a and z || c (hexagonal axes) for the trigonal
  and hexagonal groups; y || b (the unique axis) and z || c for the
  monoclinic groups. The rotation axes and mirror normals are listed as
  lattice directions, so the orientation of the two-setting groups is
  explicit (with `--pg`, phonopy's setting: `-42m` with the twofold axes along
  x and y, `3m` and `-6m2` with a mirror normal to x; with `--sg`, `P31m` and
  `P3m1`, or `P-42m` and `P-4m2`, give differently oriented forms). With `-c`
  the line `tensor axes in the Cartesian frame of the input file` gives these
  axes in the frame of the POSCAR.
- `* Independent components *` are the free constants, chosen in Nye's order
  (11, 22, 33, 23, 13, 12; normal before shear, longitudinal first), so that
  the familiar ones come out: c11, c12, c13, c33, c44 with
  `c66 = (c11-c12)/2` for hexagonal stiffness, `s66 = 2(s11-s12)` for
  compliance, d11, d14 with `d26 = -2d11` for 32.
- `* Matrix form *` is the tensor in Nye's matrix notation (`d_iL` with the
  factor 2, `s_LM` with the factors 2 and 4), with every entry written in
  the independent components; `* Relations *` lists the same as equations.
- A time-odd tensor (prefix `a`, e.g. `a[V2]` or the magnetization `aeV`)
  vanishes identically, because the point group of a non-magnetic crystal
  contains time reversal (the grey group G1'); the `* Relations *` block
  says so.

The counts reproduce Nye's tables for all 32 point groups (testsuite section
13): dielectric 6, 4, 3, 2, 1 from triclinic to cubic; piezoelectric 18 (1),
10 (m), 8 (2), 6 (3), 5 (mm2), 4 (4, -4, 3m, 6), 3 (222, 4mm, 6mm), 2 (-42m,
32, -6), 1 (422, 622, -6m2, 23, -43m) and 0 for 432 and the eleven
centrosymmetric groups; elastic 21, 13, 9, 7 or 6 (tetragonal, trigonal), 5
(hexagonal) and 3 (cubic); pyroelectric only in the ten polar classes.

**Limits.** This mode takes non-magnetic point groups only (time-odd tensors
are zero there); the Python function
`crystod.tensor_form.tensor_form_of_operations` accepts per-operation
time-reversal flags for a magnetic point group. The forms are symmetry
constraints; the values of the constants need a calculation or a
measurement.

### Raman tensors

`--tensor raman` prints the symmetric second-rank tensors that transform as
each Raman-active irrep, the forms that decide which polarization geometries
can show a Raman line:

```bash
crystod-group --tensor raman --pg m-3m
```

```
* Point group *
  m-3m (Oh), order 48, cubic, Laue class m-3m, centrosymmetric
...
* Raman tensors *
  Cartesian axes of the point group above; every Raman-active irrep, Mulliken labels of the phonopy character table
  A1g (1 tensor):  [[a, 0, 0], [0, a, 0], [0, 0, a]]
  Eg (2 tensors):  [[a, 0, 0], [0, a, 0], [0, 0, -2*a]]   [[a, 0, 0], [0, -a, 0], [0, 0, 0]]
  T2g (3 tensors):  [[0, 0, 0], [0, 0, a], [0, a, 0]]   [[0, 0, a], [0, 0, 0], [a, 0, 0]]   [[0, a, 0], [a, 0, 0], [0, 0, 0]]
```

With `-c` the Raman-active Gamma phonon irreps of the structure are listed
instead, with ISO-IR and Mulliken labels, in the Cartesian axes of the input
file:

```bash
crystod-group --tensor raman -c 186_PPOSCAR_ZnO
```

```
...
* Raman tensors *
  Cartesian axes of the input cell (not the standard axes above); the Gamma phonon irreps of the structure, ISO-IR [Mulliken]
  GM1 [A1] (2 tensors):  [[a, 0, 0], [0, a, 0], [0, 0, 0]]   [[0, 0, 0], [0, 0, 0], [0, 0, a]]
  GM5 [E2] (2 tensors):  [[a, 0, 0], [0, -a, 0], [0, 0, 0]]   [[0, a, 0], [a, 0, 0], [0, 0, 0]]
  GM6 [E1] (2 tensors):  [[0, 0, 0], [0, 0, a], [0, a, 0]]   [[0, 0, a], [0, 0, 0], [a, 0, 0]]
```

How to read it:

- One line per Raman-active irrep: its label (Mulliken with `--pg`/`--sg`;
  ISO-IR and Mulliken with `-c`), the number of tensors and the tensors as
  3x3 matrices, each with its own free constant `a`.
- For an irrep that occurs once in the symmetric square the tensors are its
  partners (the cubic Eg pair `diag(a, a, -2a)` and `diag(a, -a, 0)`); a
  one-dimensional irrep that occurs twice (A1 of 6mm) lists two basis
  tensors, and a mode's tensor is a combination of them (`diag(a, a, b)`).
  A repeated two-dimensional irrep (E of 3m) is printed per partner with
  shared constants, and a complex-pair E that occurs once (4, 6, 23) as a
  basis of its two tensors, as `crystod-phonon --raman-tensor` prints them
  (section 29 of [crystod-phonon](crystod-phonon.md)).
- The Mulliken labels follow phonopy's character tables and axis
  convention (B1/B2 of the standard setting).

The two routes give the same tensors (checked for Si, BaTiO3 and ZnO).

### Python API

`tensor_form` returns a `TensorForm`, `raman_forms` a list of `RamanTensors`
records (`label`, `tensors`); the MCP tool is `crystod_tensor_form` (`kind`
and one of `point_group`, `space_group`, `poscar`):

```python
from crystod import group

form = group.tensor_form("4mm", "piezoelectric")
form.independent        # ('d15', 'd31', 'd33')
form.relations          # ('d24 = d15', 'd32 = d31')
form.tensors.shape      # (3, 3, 3, 3): the full tensor of each independent component
print(group.format_tensor_form(group.tensor_form(None, "elastic", space_group="P6_3mc")))
[r.label for r in group.raman_forms("m-3m")]   # ['A1g', 'Eg', 'T2g']
```

## 14. Basis-function generation (`--generate-basis`)

*Example directory: `example/14_generate_basis_function` (testsuite section 14); after pip install: `crystod-example 14`*

Automatically generate 1st-3rd order polynomial basis functions
(`x, y, z` / `x^2, ..., zx` / `x^3, ..., xyz`) classified by irreducible representation:

```bash
crystod-group --generate-basis --point-group m-3m
crystod-group --generate-basis --point-group m-3m --order 2
crystod-group --generate-basis --space-group Pm-3m --kpoint 0 0 0
crystod-group --generate-basis --space-group Pm-3m --kpoint 0 0 0 --order 2 3 --show-irrep-table
```

This is the automated counterpart of `--basis`: for each requested order, all
monomials of that degree are decomposed into the irreps of the point group, or
of the little group of the selected space-group k point. The group blocks
(`* Point group *`, or `* Space group *`, `* Little group of k *` and
`* k-point (primitive) *`) come once at the top, and every block of an order
carries the order in its title. The second-order run at GM of Pm-3m, for
example, ends with

```
* Decomposition: 2nd order (quadratic) *
 1.0 [GM1+(1)] + 1.0 [GM3+(2)] + 1.0 [GM5+(3)]

* Irreducible representations for basis functions: 2nd order (quadratic) *
  GM1+(1): [x^2 + y^2 + z^2]
  GM3+(2): [-2 x^2 + y^2 + z^2, x^2 - 2 y^2 + z^2]
  GM5+(3): [xy, yz, xz]
```

— the quadratic polynomials sorted into the breathing mode, the Eg pair, and
the T2g triple (see also the projection-operator walk-through in the
[theoretical background](theory-representations.md)).

In space-group mode `--kpoint` is required, and the rules of
[section 12](#bloch-phases-and-non-symmorphic-space-groups) apply. The Bloch
phases are attached to the little-group operations, so every monomial is
accounted for. At DT of Fd-3m (`--kpoint 0.1 0 0.1 --order 2`) the six
quadratic monomials give `DT1(1): [y^2, x^2 + z^2]`, `DT5(2): [xy, yz]`,
`DT3(1): [xz]` and `DT2(1): [-x^2 + z^2]`. At a zone-boundary point where the
phases do not form a representation (`--sg Pnma --kpoint 0.5 0 0`), the
command stops with the `ERROR` shown there.

## 15. Coset decomposition (`--coset`)

*Example directory: `example/15_show_coset` (testsuite section 15); after pip install: `crystod-example 15`*

Point-group mode decomposes G into left cosets g H of a subgroup H:

```bash
crystod-group --coset --point-group m-3m --subgroup 4/mmm
```

```
* Groups *
 G = m-3m (order 48)
 H = 4/mmm (order 16)

* Coset decomposition G = sum_i g_i H *
 index [G:H] = 3
 coset 1 (representative: E):
   { E, C4#1, C4#2, C4^2#1, C4^2#2, C4^2#3, C2#1, C2#4, i, S4#1, S4#2, sgh#1, sgh#2, sgh#3, sgd#4, sgd#1 }
 coset 2 (representative: C3#1):
   { C3#1, C2#3, C4#5, C3#7, C3#3, C3#5, C4#6, C2#6, S6#1, sgd#3, S4#5, S6#7, S6#3, S6#5, sgd#6, S4#6 }
 coset 3 (representative: C3#2):
   { C3#2, C4#4, C2#2, C3#8, C3#4, C3#6, C4#3, C2#5, S6#2, S4#4, sgd#2, S6#8, S6#4, S6#6, sgd#5, S4#3 }
```

Space-group mode decomposes the rotation group of G into right cosets G_k g of
the little co-group of a k point — one coset per arm of the star of k:

```bash
crystod-group --coset --space-group Pm-3m --kpoint 0.5 0.5 0
```

```
* Little co-group G_k *
 order |G_k| = 16 (|G| = 48)
 { 1, 2_100, 2_010, 2_001, 4^+_001, 4^-_001, 2_110, 2_-110, -1, m_100, m_010, m_001, -4^+_001, -4^-_001, m_110, m_-110 }

* Coset decomposition G = sum_i G_k g_i (one coset per arm of the star) *
 index [G:G_k] = |star of k| = 3
 coset 1 (representative: 1, k arm = [+0.5, +0.5, +0]):
   { 1, 2_100, 2_010, 2_001, 4^+_001, 4^-_001, 2_110, 2_-110, -1, m_100, m_010, m_001, -4^+_001, -4^-_001, m_110, m_-110 }
 coset 2 (representative: 3^+_111, k arm = [+0.5, +0, +0.5]):
   { 3^+_111, 3^+_1-1-1, 3^+_-11-1, 3^+_-1-11, 4^+_100, 4^-_100, 2_011, 2_01-1, ... }
 coset 3 (representative: 3^-_111, k arm = [+0, +0.5, +0.5]):
   { 3^-_111, 3^-_1-1-1, 3^-_-11-1, 3^-_-1-11, 4^+_010, 4^-_010, 2_101, 2_-101, ... }
```

Each coset carries the k arm it generates, so this is the operation-level view
of the star printed by `crystod --star-of-k`. For the point-group mode, H must
be expressed in the same axes convention as G; a clear error message is printed
otherwise.

## 16. Correlation and compatibility (`--correlate`)

*Example directory: `example/16_correlate` (testsuite section 16); after pip install: `crystod-example 16`*

`--correlate` answers what becomes of a level or a mode when the symmetry is
lowered: how an orbital level splits under a site distortion, which Gamma
modes of a low-symmetry phase a parent mode turns into (the Raman lines that
appear below a transition), and how band or phonon labels connect along a
symmetry line. The group theory is the subduction of a representation:
restricted to a subgroup H, an irrep of G is reduced into the irreps of H. It
has three forms, one example each below:

- with `--pg G --subgroup H`, the **correlation table** of a point group:
  every irrep of G restricted to each inequivalent orientation of the
  subgroup type H, the counterpart of CORREL of the Bilbao Crystallographic
  Server;
- with `--parent SG --irrep IR --order-parameter C...`, the **subduction** of
  parent irreps to the Gamma point of the isotropy subgroup H of that
  direction: which Gamma irreps of the low-symmetry phase a parent irrep
  becomes (for example, which Raman-active modes a zone-boundary soft mode
  turns into);
- with `--sg SG --kpoint K0 K1`, the **compatibility relations** of the small
  irreps at two special points along the symmetry line joining them, the
  counterpart of COMPATIBILITY RELATIONS of the Bilbao Crystallographic
  Server.

### Point-group correlation tables

```bash
crystod-group --correlate --pg m-3m --subgroup 4/mmm
```

```
* Point group *
m-3m (order 48)

* Subgroup *
4/mmm (order 16, index 3): 1 inequivalent orientation
  4/mmm (4 || z): 3 conjugate subgroups; C2' = 2_x, 2_y; C2'' = 2_[110], 2_[1-10]

* Correlation table *
irrep  4/mmm (4 || z)
A1g    A1g
A2g    B1g
Eg     A1g + B1g
T1g    A2g + Eg
T2g    B2g + Eg
A1u    A1u
A2u    B1u
Eu     A1u + B1u
T1u    A2u + Eu
T2u    B2u + Eu

note: x, y, z are the axes [100], [010], [001] of the table of m-3m; 2_d is
      the twofold rotation about d, m_d the mirror normal to d; the symbols of
      H are those of phonopy's table of H with its classes as listed under
      * Subgroup * (B1/B2/B3 follow them).
```

A tetragonal elongation of an octahedral site splits T2g into B2g + Eg and Eg
into A1g + B1g. How to read it:

- All subgroups of G are generated (98 for m-3m), and those of type H are
  grouped into conjugacy classes under G. Each class is one column: an
  orientation of H that no operation of G turns into another. Its header
  names the principal axis of H in the axes of G; `* Subgroup *` gives the
  number of conjugate subgroups in the class and the classes of H that fix
  its orientation about that axis (`2_d` is the twofold rotation about the
  direction d, `m_d` the mirror normal to d).
- The symbols of H are those of phonopy's character table of H, with its
  classes placed as listed. B1/B2 (and B1/B2/B3) therefore follow those
  classes: B1 of 4/mmm is the irrep that is symmetric under C2' = 2_x, 2_y.
  For -42m, 3m, -3m and -6m2, phonopy has two settings (-42m and -4m2, 3m1
  and 31m, -3m1 and -31m, -6m2 and -62m); when the axes of G match one of
  them, the header names it (`3m (3 || [001], 31m setting)` in 6/mmm). Off
  the axes of G (3m along [111] of m-3m) the two settings give the same
  labels and no setting is named.
- Directions are x, y, z for the cubic, tetragonal, orthorhombic and
  monoclinic groups, and [uvw] in the hexagonal axes a, b, c for the trigonal
  and hexagonal groups (`--pg 6/mmm --subgroup mmm` gives the column
  `mmm (2 || [001])` with C2x = 2_[100] and C2y = 2_[120]).
- A complex-conjugate pair that phonopy merges into one real E (point groups
  3, 4, 6, -3, -4, -6, 4/m, 6/m, 23, m-3) keeps that symbol: E of 4 becomes
  2 B of 2.
- Every row is checked: the multiplicities are integers and the dimensions
  add up to that of the irrep of G.

When H occurs in several orientations, the table has one column per
orientation. The mirror-plane subgroups mm2 of m-3m come in three:

```bash
crystod-group --correlate --pg m-3m --subgroup mm2
```

```
...
* Subgroup *
mm2 (order 4, index 12): 3 inequivalent orientations
  mm2 (2 || z, m_y): 3 conjugate subgroups; sgvxz = m_y; sgvyz = m_x
  mm2 (2 || z, m_[110]): 3 conjugate subgroups; sgvxz = m_[110]; sgvyz = m_[1-10]
  mm2 (2 || [110]): 6 conjugate subgroups; sgvxz = m_[1-10]; sgvyz = m_z

* Correlation table *
irrep  mm2 (2 || z, m_y)  mm2 (2 || z, m_[110])  mm2 (2 || [110])
A1g    A1                 A1                     A1
A2g    A1                 A2                     B2
Eg     2 A1               A1 + A2                A1 + B2
T1g    A2 + B1 + B2       A2 + B1 + B2           A2 + B1 + B2
T2g    A2 + B1 + B2       A1 + B1 + B2           A1 + A2 + B1
A1u    A2                 A2                     A2
A2u    A2                 A1                     B1
Eu     2 A2               A1 + A2                A2 + B1
T1u    A1 + B1 + B2       A1 + B1 + B2           A1 + B1 + B2
T2u    A1 + B1 + B2       A2 + B1 + B2           A1 + A2 + B2
...
```

The twofold axis along z goes with either the axial mirrors (m_x, m_y) or the
diagonal ones (m_[110], m_[1-10]); a third class has it along a face diagonal.
Columns with the same principal axis add the first element of the first class
that fixes the orientation to the header. Likewise, 4/mmm has three classes of
2/m (twofold axis along z, along x or y, along a diagonal). A type that is not
a subgroup of G gives an error that lists the subgroup types of G.

### Subduction to the isotropy subgroup

```bash
crystod-group --correlate --parent Pm-3m --irrep R4+ --order-parameter 0 0 a \
    --irrep-list R4+ R5+ M3+ GM4-
```

```
* Supergroup *
Pm-3m (No. 221)

* Isotropy subgroup *
R4+(0,0,a) -> I4/mcm (No. 140)
cell size 2, index 6
sublattice basis (parent primitive units): (2,0,0), (1,1,0), (1,0,1)
conventional basis (parent conventional units): (-1,0,1), (1,0,1), (0,2,0)
origin: (0,0,0)

* Subduction to the Gamma point of H *
H = I4/mcm (No. 140); Gamma labels in the ISO-IR frame of H:
basis (parent conventional units): (-1,0,1), (1,0,1), (0,2,0)
origin: (0,0,0)
  R4+ (R, dim 3) -> GM1+ [A1g] + GM5+ [Eg]
  R5+ (R, dim 3) -> GM4+ [B2g] + GM5+ [Eg]
  M3+ (M, dim 3) -> 3 components at non-Gamma k of H
  GM4- (GM, dim 3) -> GM3- [A2u] + GM5- [Eu]
note: GM1 of H occurs in a primary irrep once per free parameter of the direction
```

The octahedral tilt R4+ of SrTiO3 becomes A1g + Eg of I4/mcm: the soft mode
of the 105 K transition is Raman active below it. How to read it:

- The first two blocks are those of `--parent --order-parameter` (section 17).
  `--irrep-list` names the parent irreps to subduce, any tabulated irrep of
  the parent; without it, the `--irrep` labels are subduced.
- The translations of H keep the part of an irrep whose wave vectors belong to
  the reciprocal lattice of H (its Gamma sector). That part is decomposed into
  Gamma irreps of H; the rest is counted as `components at non-Gamma k of H`
  (the three arms of M3+ are not reciprocal-lattice vectors of I4/mcm).
- The Gamma labels of H are ISO-IR labels in the frame printed above them
  (basis and origin in parent conventional units). They depend on that frame:
  normalizer-equivalent settings of H exchange labels such as GM2+/GM3+/GM4+
  of an orthorhombic group, so compare labels only together with the frame.
  The Mulliken symbols in brackets come from phonopy's point-group tables in
  the axes of spglib's conventional cell of H (here the a axis of I4/mcm is a
  face diagonal of the cubic cell, so the Eg strain GM3+ of Pm-3m becomes
  A1g + B2g, not A1g + B1g).
- A complex Gamma irrep of H and its complex-conjugate partner form one
  physically irreducible term with the ISOTROPY pair label and one Mulliken
  symbol, e.g. `--irrep GM4+ --order-parameter a a a` (R-3) gives
  `GM1+ [Ag] + GM2+GM3+ [Eg]`.
- Two checks are applied to every line: the identity irrep GM1 of H occurs in
  a primary irrep once per free parameter of the direction (here one), and
  the dimensions add up to the size of the Gamma sector.

Coupled order parameters work as in `--parent`. For the Pnma phase of CaTiO3
(a-a-c+ tilts):

```bash
crystod-group --correlate --parent Pm-3m --irrep R4+ M3+ --order-parameter 0 a a d 0 0
```

```
...
* Subduction to the Gamma point of H *
H = Pnma (No. 62); Gamma labels in the ISO-IR frame of H:
basis (parent conventional units): (1,1,0), (0,0,-2), (-1,1,0)
origin: (0,0,0)
  R4+ (R, dim 3) -> GM1+ [Ag] + GM2+ [B1g] + GM4+ [B2g]
  M3+ (M, dim 3) -> GM1+ [Ag] + 2 components at non-Gamma k of H
...
```

Both irreps contain Ag once. R4+ becomes three Raman-active modes, while only
the condensed arm of M3+ is at the Gamma point of Pnma.

### Compatibility relations

```bash
crystod-group --correlate --sg Pm-3m --kpoint GM X
```

```
* Space group *
Pm-3m (No. 221)

* Compatibility relations *
line DT between GM (0, 0, 0) and X (0, 1/2, 0), labeled at kdelta = (0, 1/24, 0) (primitive basis)
  GM1+ (dim 1) -> DT1
  GM2+ (dim 1) -> DT2
  GM3+ (dim 2) -> DT1 + DT2
  GM4+ (dim 3) -> DT4 + DT5
  GM5+ (dim 3) -> DT3 + DT5
  GM1- (dim 1) -> DT4
  GM2- (dim 1) -> DT3
  GM3- (dim 2) -> DT3 + DT4
  GM4- (dim 3) -> DT1 + DT5
  GM5- (dim 3) -> DT2 + DT5
  X1+  (dim 1) -> DT1
  X2+  (dim 1) -> DT2
  X3+  (dim 1) -> DT4
  X4+  (dim 1) -> DT3
  X5+  (dim 2) -> DT5
  X1-  (dim 1) -> DT4
  X2-  (dim 1) -> DT3
  X3-  (dim 1) -> DT1
  X4-  (dim 1) -> DT2
  X5-  (dim 2) -> DT5
```

- The line is the segment from the tabulated K0 to the nearest arm of the
  star of K1 with the largest little group; `--line L` selects another one by
  its ISO-IR name when several join the two points. With one name
  (`--kpoint GM`), every line from that point to another special point is
  listed (symmetry planes are skipped). A segment selected with two names
  that lies on a symmetry plane, e.g. `--kpoint GM X --line A`, is headed
  `plane` instead of `line`.
- The small irreps of both end points are restricted to the little group of
  the point `kdelta` of the line, with the translation phases corrected for
  the difference of the wave vectors, and decomposed into the ISO-IR irreps
  of the line at `kdelta`. Their dimensions always add up to the dimension of
  the end-point irrep.
- ISO-IR numbering of the Delta line of Pm-3m (little co-group 4mm, with the
  characters of `crystod-group --table --sg Pm-3m --kpoint 0 1/4 0`; m_100,
  m_001 are the axial mirrors, m_101, m_-101 the diagonal ones): DT1 = A1
  (Delta1 of Bouckaert, Smoluchowski and Wigner), DT2 = B1 (Delta2, +1 on the
  axial mirrors), DT3 = B2 (Delta2', +1 on the diagonal mirrors), DT4 = A2
  (Delta1'), DT5 = E (Delta5). So GM4- (T1u) -> Delta1 + Delta5 and GM5+ (T2g)
  -> Delta2' + Delta5. The CDML numbering of the Bilbao tables can differ from
  ISO-IR for the one-dimensional line irreps (DT3 and DT4 are exchanged in
  Fm-3m); compare characters, not numbers, across the two conventions.

The same works for non-symmorphic groups and zone-boundary lines, e.g. the
Gamma-Delta-X line of silicon (`--sg Fd-3m --kpoint GM X`: GM5+ -> DT3 + DT5,
X1 -> DT1 + DT3) and the Z line of Pm-3m (`--kpoint X M`).

**Limits.** All three forms treat single-valued (spinless) irreps. The
subduction takes the tabulated irreps of the special k points of the parent
and resolves only the Gamma part of H by label (the rest is counted); the
compatibility relations join two special points along a line or a plane of
the ISO-IR tables (with one name, symmetry planes are skipped).

**Python API.** `correlation_table` returns a `CorrelationTable`,
`subduce_to_child` a list of `Subduction` records (one per parent irrep) and
`compatibility_relations` a list of `Compatibility` records (one per
end-point irrep); the `format_*` functions print them as the command does.
The MCP tool is `crystod_correlate`, which takes the arguments of exactly one
form (`point_group` and `subgroup`; `space_group` and `kpoints`; `parent`,
`irreps` and `direction`):

```python
from crystod import group

print(group.format_correlation_table(group.correlation_table("m-3m", "4/mmm")))
rows = group.subduce_to_child("Pm-3m", ["R4+"], "0 0 a")
rows[0].gamma_irreps, rows[0].mulliken    # [('GM1+', 1), ('GM5+', 1)] {'GM1+': 'A1g', 'GM5+': 'Eg'}
print(group.format_compatibility(group.compatibility_relations("Pm-3m", "GM", "X")))
```

## 17. Isotropy subgroups (`--parent`)

*Example directory: `example/17_isotropy_subgroup` (testsuite section 17); after pip install: `crystod-example 17`*

When a distortion transforming as an irrep condenses, the symmetry drops from
the parent group to the **isotropy subgroup** H(eta) = {g : D(g) eta = eta},
which depends on the order-parameter direction eta. The value the flag takes is
the **parent** group; `--supergroup SG` is kept as an alias of `--parent SG`
(backward compatibility) and should not be confused with `--supergroup-cif`,
the symmetry-mode analysis of section 23.

Omit `--order-parameter` to enumerate every direction type:

```bash
crystod-group --parent Pm-3m --irrep R4+
```

```
...
* Irrep *
R4+: order parameter dimension 3

* Order parameter directions and isotropy subgroups *
irrep                subgroup           size  index
R4+(0,0,a)           140 I4/mcm         2     6    
R4+(a,a,a)           167 R-3c           2     8    
R4+(0,a,a)           74 Imma            2     12   
R4+(0,a,b)           12 C2/m            2     24   
R4+(a,a,b)           15 C2/c            2     24   
R4+(a,b,c)           2 P-1              2     48   
...
```

— the complete Howard-Stokes octahedral-tilt classification of perovskites, in
one command. Give a direction to get that subgroup in full, with the cell
relation you need to build it. With a direction (and with `--invariants`,
`--degree` and `--secondary`), three lines under the irrep give the two
symmetry conditions for a continuous (second-order) transition: the
**Landau condition** (no cubic invariant of the order parameter) and the
**Lifshitz condition** (no invariant of the form Q_i dQ_j/dx - Q_j dQ_i/dx,
which would make the order parameter modulate away from its k point); both
are explained in [section 18](#18-invariant-polynomials---invariants):

```bash
crystod-group --parent Pm-3m --irrep GM4- --order-parameter 0 0 a
```

```
...
* Irrep *
GM4-: order parameter dimension 3
  Landau condition: satisfied (no cubic invariant)
  Lifshitz condition: satisfied (no Lifshitz invariant)
  -> a continuous transition is allowed

* Isotropy subgroup *
GM4-(0,0,a) -> P4mm (No. 99)
cell size 1, index 6
sublattice basis (parent primitive units): (1,0,0), (0,1,0), (0,0,1)
conventional basis (parent conventional units): (0,0,1), (1,0,0), (0,1,0)
origin: (0,0,0)
...
```

Zone-boundary irreps carry their full star (order-parameter dimension =
arms x small dimension) and the cell enlargement is detected automatically.
Letters in `--order-parameter` are free parameters.
Following ISOTROPY, the order-parameter components are grouped arm by arm:
`;` separates the star arms and `,` the components within one arm (R4+ of
Pm-3m: one arm x small dim 3 -> `(a,a,b)`; M3+: three arms x small dim 1 ->
`(a;b;c)`; X5+: three arms x small dim 2 -> `(a,b;0,0;0,0)`).
A direction read off a table can be passed back as it is printed:
`--order-parameter a -a 0`, `--order-parameter "a;-a;0"` and
`--order-parameter "(a;-a;0)"` are the same direction (`M3+(a;-a;0) ->
I4/mmm (No. 139)` for M3+ of Pm-3m). Every token after `--order-parameter`
up to the next `--option` is a component, so a leading minus sign is never
taken for an option. A component may carry an exact factor (`2a`, `-0.5a`,
`1/2a`) or be a sum of such terms (`a+b-c-d`, `2a-1/2b`), as the tables print
them: `--parent Ia-3d --irrep P1 --order-parameter 2a 2b 2c 2d a+b-c-d
-a+b-c+d a+b+c+d a-b-c+d` gives C2 (No. 5), index 96. The decimals the
tables print for irrational factors (`0.282a` at K of P6_3/mmc,
`-0.3999b+0.917c` at W of Fm-3m) are rounded, so they are refused, alone or
inside a sum: read those entries from the listing instead. A decimal is
printed with as many digits as it takes not to read as an exact fraction it
is not (`0.7504a`, never `0.75a` = 3/4 a), and a ratio is printed as small
integers only when it is exactly that ratio, so every direction a table
prints either reads back to its own subgroup or is refused.

When the induced irrep is of complex or pseudoreal type (Frobenius-Schur
indicator 0 or -1), the real order parameter transforms as the physically
irreducible **doubled real form**: the order-parameter dimension doubles and
the irrep is reported under an ISOTROPY-style pair label
(`crystod-group --parent Ia-3d --irrep P2` prints `P1P2`, order-parameter
dimension 8). Separately tabulated +k/-k stars pair across the stars in the
same way (`P1` of I-42d -> `P1PA1`, `H1` of P3 -> `H1HA1`). The partner is
the irrep with the same real form on every group element, lattice
translations included, which gives the pair labels of the ISOSUBGROUP tables
(`K2KA2` and `K3KA3` at K of P3, `H3H6` and `H4H5` at H of P6_3/m).
`--irrep` takes a pair label as `--kpoint` prints it (`--parent P6_3/m
--irrep H1H2 H3H6` is `--irrep H1 H3`): the order parameter is built from the
member listed first in the tables, the member the `--kpoint` table uses, so
its components read the same; two irreps that are not a complex-conjugate
pair are refused with the pair of the first one (`H1H3`: the order parameter
of H1 is H1H2).

```{seealso}
**Theory:** [Complex- and pseudoreal-type irreps](theory-isotropy-subgroups.md) — the realification of D + D*, and the conjugate-gauge and origin matching behind these pair labels.
```

### Every irrep of a k point (`--kpoint`)

With `--kpoint` in place of `--irrep`, the tables of **all irreps of one
special k point** are printed as a single table, in the order of the ISO-IR
tables — the survey to start from when the irrep of the distortion is not
known yet:

```bash
crystod-group --parent Pm-3m --kpoint GM
```

```
* Supergroup *
Pm-3m (No. 221)

* Kpoint *
GM
k = (0, 0, 0) in the primitive basis, star of 1 arm(s)

* Order parameter directions and isotropy subgroups *
irrep                subgroup           size  index
GM1+(a)              221 Pm-3m          1     1    
GM2+(a)              200 Pm-3           1     2    
GM3+(a,0)            123 P4/mmm         1     3    
GM3+(a,b)            47 Pmmm            1     6    
GM4+(0,0,a)          83 P4/m            1     6    
GM4+(a,a,a)          148 R-3            1     8    
GM4+(0,a,a)          12 C2/m            1     12   
GM4+(a,b,c)          2 P-1              1     24   
GM5+(a,a,a)          166 R-3m           1     4    
GM5+(0,0,a)          65 Cmmm            1     6    
GM5+(a,a,b)          12 C2/m            1     12   
GM5+(a,b,c)          2 P-1              1     24   
GM1-(a)              207 P432           1     2    
GM2-(a)              215 P-43m          1     2    
GM3-(a,0)            89 P422            1     6    
GM3-(0,a)            111 P-42m          1     6    
GM3-(a,b)            16 P222            1     12   
GM4-(0,0,a)          99 P4mm            1     6    
GM4-(a,a,a)          160 R3m            1     8    
GM4-(0,a,a)          38 Amm2            1     12   
GM4-(0,a,b)          6 Pm               1     24   
GM4-(a,a,b)          8 Cm               1     24   
GM4-(a,b,c)          1 P1               1     48   
GM5-(0,0,a)          115 P-4m2          1     6    
GM5-(a,a,a)          155 R32            1     8    
GM5-(0,a,a)          38 Amm2            1     12   
GM5-(a,a,b)          5 C2               1     24   
GM5-(0,a,b)          6 Pm               1     24   
GM5-(a,b,c)          1 P1               1     48   
...
```

Each irrep contributes exactly the rows of its own `--irrep` table, so
`GM4-` above is the ferroelectric table (P4mm, R3m, Amm2, ...) and `GM3+`
the Jahn-Teller one. The k point is given by its ISO-IR name (`GM`, `R`,
`X`, `M`, ...; the case is ignored, and `G`/`Gamma` are read as `GM`) or by
three coordinates in the primitive reciprocal basis, fractions allowed
(write thirds as `1/3`, or with three decimals, `0.333`).
Coordinates may be those of any arm of the star and may differ from it by a
reciprocal lattice vector: `--kpoint 0.5 0.5 0.5`, `--kpoint -1/2 1/2 1/2`
and `--kpoint R` print the same table, and `--kpoint 0 0.5 0.5` is the M
table. The second line of the `* Kpoint *` block gives the tabulated arm and
the number of arms of the star. Only the tabulated special points can be
listed: a name the space group does not have, or coordinates on a symmetry
line or plane or at a general point, stop with an error that lists the
available k points with their coordinates (so does `--parent SG` with neither
`--irrep` nor `--kpoint`); an irrep label given to `--kpoint` is answered
with its k point and the `--irrep` form. `--kpoint` cannot be combined with
`--irrep` or `--order-parameter`.

The two members of a complex-conjugate pair share one physically irreducible
order parameter and are listed once, under their pair label
(`--parent Pm-3 --kpoint GM` lists `GM2+GM3+`, not `GM2+` and `GM3+`). The
block of a pair is the `--irrep` table of its first member (`GM2+`), and the
order-parameter components refer to the basis built from that member: to
resolve a direction read off the table, pass that label
(`--irrep GM2+ --order-parameter ...`). The partner (`--irrep GM3+`) reaches
the same subgroups, but its components may refer to a different basis of the
same order parameter. A pair whose partner sits at the -k point (`K2KA2` at K
of P3) has a single member at the k point and is listed under the pair label
as well.

The label column is as wide as the longest label of the k point. The
notes of the single-irrep mode are collected under the table. If the
enumeration of one irrep fails, the others are still printed and the failed
irrep is named in a `note: LABEL: not enumerated (...)` line; the command
exits with an error only when no irrep of the k point could be enumerated.
Most k points take a few seconds. The W point of the F-centred cubic groups
(Fm-3m, Fd-3m, Fm-3c, Fd-3c), with order parameters of up to 12 dimensions,
takes 20 to 40 seconds. On an Apple M3 Ultra, `--kpoint W` took 21 s for
Fm-3c, 22 s for Fd-3m, 28 s for Fm-3m and 37 s for Fd-3c, and the
12-dimensional `--irrep W5` of Fm-3m alone took 19 s. Nothing is printed
until the table is complete.
From Python, `crystod.group.isotropy_subgroups_at_kpoint("Pm-3m", "GM")`
returns the same table as data ([Python API](python-api.md)).

### Coupled order parameters (several irreps)

Giving `--irrep` **several labels** enumerates the isotropy subgroups of the
**coupled** order parameters — the stabilizers on the direct sum of the
irreps, i.e. the space groups reached when several distortions condense
simultaneously:

```bash
crystod-group --parent I4/mmm --irrep X3- X2+
```

```
...
* Order parameter directions and isotropy subgroups (X3- alone) *
irrep                subgroup           size  index
X3-(0;a)             63 Cmcm            2     4
X3-(a;a)             136 P4_2/mnm       4     4
X3-(a;b)             58 Pnnm            4     8

* Order parameter directions and isotropy subgroups (X2+ alone) *
irrep                subgroup           size  index
X2+(0;c)             64 Cmce            2     4
X2+(c;c)             127 P4/mbm         4     4
X2+(c;d)             55 Pbam            4     8

* Order parameter directions and isotropy subgroups (coupled) *
irrep                subgroup           size  index
X3-(0;a) X2+(0;c)    36 Cmc2_1          2     8
X3-(0;a) X2+(c;0)    62 Pnma            4     8
X3-(a;a) X2+(c;c)    38 Amm2            4     16
X3-(0;a) X2+(c;d)    26 Pmc2_1          4     16
X3-(a;b) X2+(0;c)    31 Pmn2_1          4     16
X3-(a;b) X2+(c;d)    6 Pm               4     32
...
```

The single-irrep tables of every given irrep are printed first, then the
coupled table. Its first column groups the components irrep by irrep, and
every irrep keeps its own free-parameter letters (X3-: a, b; X2+: c, d --
the amplitudes of different irreps are always independent); every coupled
direction condenses *all* the irreps with nonzero amplitude (a zero irrep
would just reproduce the single-irrep tables above). The arm combinations
matter: for
the n = 2 Ruddlesden-Popper structure above, condensing the octahedral
rotation (X2+) and tilt (X3-) at the *same* X arm gives the polar
hybrid-improper ferroelectric ground state `Cmc2_1` (= A2_1am, as in
Ca3Ti2O7), while *crossed* arms give nonpolar `Pnma`. `--order-parameter`
then takes the concatenated components (`--order-parameter 0 a 0 c` above
resolves to Cmc2_1), and `--parent Pm-3m --irrep R4+ M3+` reproduces the
full Howard-Stokes table of *mixed* perovskite tilt systems (a-a-c+ =
`R4+(0,a,a) M3+(a;0;0)` -> Pnma, a+a+c- -> P4_2/nmc, a0b-c+ -> Cmcm, ...).

`--parent` is the offline counterpart of **ISOSUBGROUP** of the ISOTROPY
Software Suite (https://iso.byu.edu) and reproduces its published strata
tables. Where a stratum's enantiomorphic partner differs from the ISOTROPY
software, a note is printed under the output.

```{seealso}
**Theory:** [Validation against ISOSUBGROUP](theory-isotropy-subgroups.md) — the exhaustive validation sweep, the known exceptions, and the citation to give if you use this feature.
```

## 18. Invariant polynomials (`--invariants`)

*Example directory: `example/18_invariants` (testsuite section 18); after pip install: `crystod-example 18`*

`--invariants` writes down the Landau free energy of a structural (or any
other) order parameter: the terms allowed in the expansion, whether a
continuous transition is possible, how several order parameters couple and
which secondary distortions a given direction drives. It is meant for anyone
who models a phase transition or interprets a distortion found by a
calculation or a refinement. The group theory: the free energy is a
polynomial in the order-parameter components that every operation of the
parent group leaves unchanged, and the command lists a basis of these
**invariant polynomials** degree by degree, up to `--degree N` (default 4),
in place of the isotropy-subgroup table:

```bash
crystod-group --parent Pm-3m --irrep R4+ --invariants --degree 4
```

```
* Supergroup *
Pm-3m (No. 221)

* Irrep *
R4+: order parameter dimension 3
  Landau condition: satisfied (no cubic invariant)
  Lifshitz condition: satisfied (no Lifshitz invariant)
  -> a continuous transition is allowed

* Invariant polynomials (degree <= 4) *
Order-parameter components: (Q1, Q2, Q3), in the order of the direction patterns (a, b, c)
degree 2: 1 invariant
  I2_1 = Q1^2 + Q2^2 + Q3^2
degree 3: none
degree 4: 2 invariants (1 from lower degrees, 1 new)
  I2_1^2
  I4_1 = Q1^2*Q2^2 + Q1^2*Q3^2 + Q2^2*Q3^2
Numbers of invariants checked against the Molien series.

Conventions and validation: ISOTROPY INVARIANTS (https://iso.byu.edu):
D. M. Hatch and H. T. Stokes, J. Appl. Cryst. 36, 951-952 (2003).
```

How to read it:

- `Q1, Q2, ...` are the order-parameter components in the real basis of the
  direction patterns of `--parent` (component `Q1` is the letter `a` of the
  first slot, and so on, arm by arm; a complex-type pair such as `R4R5` of
  P-43n has its real and imaginary components arm by arm).
- For every degree the number of linearly independent invariants is given.
  Products of lower-degree invariants (`I2_1^2`) are listed by name; the
  remaining **new** invariants are named `I<degree>_<n>` and written out in
  reduced row-echelon form (graded lexicographic order, `Q1^n` first), with
  exact coefficients (fractions, multiples of `sqrt(2)`, `sqrt(3)`,
  `sqrt(6)`) where they exist and six decimals otherwise.
- The invariants are found numerically, as the polynomials unchanged by the
  generators of the group, and their number at every degree is checked
  against the **Molien series**, computed independently from the traces of
  the representation matrices. A mismatch stops the command with an error.
- `degree 3: none` is the **Landau condition**, one of the two symmetry
  requirements for a continuous transition; it is necessary but not
  sufficient. The other is the **Lifshitz condition** (below). The three-arm
  M1+ of Pm-3m, for instance, has the cubic invariant `I3_1 = Q1*Q2*Q3`
  (`example/18_invariants`), while the in-phase tilt M3+ has none.
- Coefficients printed with six decimals are not exact: they appear when the
  real basis chosen for the irrep (at the W points of Fm-3m, for example)
  makes the coefficients irrational, so they depend on that basis. The
  Python objects keep the full-precision values.

ISOTROPY's INVARIANTS uses its own real basis of the irrep, related to the
CrystOD basis by an orthogonal transformation: the polynomials can look
different (permuted or rotated components), while the numbers of invariants
agree.

**Limits.** `--invariants` takes the tabulated irreps of the special k points
(not `--kpoint` and not the symmetry lines). A degree whose linear problem
has more than 20000 monomials (a 12-component irrep at degree 8, for
example) is refused before any work with `ERROR: ... lower --degree`; above
5000 a warning is printed.

### Landau and Lifshitz conditions

The runs with `--invariants`, `--order-parameter`, `--degree` or
`--secondary` print both conditions under the irrep; the plain enumeration
`--parent SG --irrep IR [IR2 ...]` and the `--parent SG --kpoint K` table
leave them out. The breathing irrep M1+ of Pm-3m fails the Landau condition:

```bash
crystod-group --parent Pm-3m --irrep M1+ --invariants --degree 3
```

```
* Supergroup *
Pm-3m (No. 221)

* Irrep *
M1+: order parameter dimension 3 (star of 3 arm(s) x small dim 1)
  Landau condition: violated (1 cubic invariant)
  Lifshitz condition: satisfied (no Lifshitz invariant)
  -> a continuous transition is forbidden (Landau)

* Invariant polynomials (degree <= 3) *
Order-parameter components: (Q1, Q2, Q3), in the order of the direction patterns (a; b; c)
degree 2: 1 invariant
  I2_1 = Q1^2 + Q2^2 + Q3^2
degree 3: 1 invariant
  I3_1 = Q1*Q2*Q3
...
```

How to read it:

- **Landau condition**: the number of cubic invariants (the `degree 3`
  count above). A cubic term makes the transition first order.
- **Lifshitz condition**: the number of independent Lifshitz invariants
  Q_i dQ_j/dx - Q_j dQ_i/dx, i.e. the multiplicity of the identity in the
  antisymmetric square of the irrep times the polar vector. A Lifshitz term
  makes a modulated (incommensurate) phase near the k point more favourable
  than the commensurate one; at W of Fm-3m, for example, W5 has one.
- The verdict line says whether a continuous transition is allowed, or
  which condition forbids it (`(Landau)`, `(Lifshitz)`, `(Landau and
  Lifshitz)`).
- Both conditions are symmetry requirements: necessary, not sufficient. A
  transition that satisfies both can still be first order.

Both counts are computed from the characters, for the physically irreducible
representation (the doubled real form of a complex pair). The cubic counts
are those of the Molien-checked invariants above; the Lifshitz counts have
not been compared with ISOTROPY's `SHOW LIFSHITZ` over all irreps.

### Direct sums and coupling terms

Several `--irrep` labels give the invariants of the **direct sum**: the
variables are named `<label>_<n>`, and the invariants are listed by
**multidegree** (the degree in the components of every irrep), in the order
of total degree and then from the first irrep to the last. A multidegree
whose every entry is at least 1 is a **coupling term**; the lowest one is
named after the list. With two octahedral modes and a polar mode of the
n = 2 Ruddlesden-Popper parent, this is the trilinear term of hybrid improper
ferroelectricity (Benedek and Fennie, Phys. Rev. Lett. 106, 107204 (2011)):

```bash
crystod-group --parent I4/mmm --irrep X2+ X3- GM5- --invariants --degree 3
```

```
...
* Invariant polynomials (degree <= 3) *
Order-parameter components: (X2+_1, X2+_2, X3-_1, X3-_2, GM5-_1, GM5-_2), in the order of the direction patterns (a; b | c; d | e, f)
degree (2, 0, 0): 1 invariant
  I(2,0,0)_1 = X2+_1^2 + X2+_2^2
degree (1, 1, 0): none
degree (1, 0, 1): none
degree (0, 2, 0): 1 invariant
  I(0,2,0)_1 = X3-_1^2 + X3-_2^2
...
degree (1, 1, 1): 1 invariant
  I(1,1,1)_1 = X2+_1*X3-_1*GM5-_1 + X2+_1*X3-_1*GM5-_2 + X2+_2*X3-_2*GM5-_1 - X2+_2*X3-_2*GM5-_2
...
Lowest-order coupling term: degree (1, 1, 1), 1 invariant
Numbers of invariants checked against the Molien series.
...
```

The components line separates the irreps with ` | `. The coupling term
counts every invariant of its multidegree, products of lower-degree
invariants included: for the two perovskite tilts `--irrep R4+ M3+` the
lowest coupling is the biquadratic `degree (2, 2), 2 invariants`, and up to
degree 3 the line reads `No coupling term up to total degree 3`. A coupling
invariant may still vanish for a particular order-parameter direction (the
amplitudes may sit on star arms that the term does not connect);
`--secondary` below tests the coupling in a given direction.

### Free energy along a direction

With `--order-parameter`, `--invariants` also prints the free energy
restricted to that direction: every invariant is evaluated on the direction
pattern, polynomials that vanish or depend linearly on the others are
dropped, and the rest is given per degree (separated by `; `):

```bash
crystod-group --parent Pm-3m --irrep R4+ --invariants --order-parameter a b 0
```

```
...
Restricted to (a,b,0) [C2/m (12)]:
degree 2: a^2 + b^2
degree 4: a^4 + b^4; a^2*b^2
...
```

`(a,0,0)` [I4/mcm] and `(a,a,a)` [R-3c] both reduce to `a^2` and `a^4`: a
one-parameter direction has a single term per even degree. The bracket names
the isotropy subgroup of the direction.

### Secondary order parameters (`--secondary`)

When an order parameter condenses along a direction, every other irrep with
a component fixed by the isotropy subgroup H may appear as well, driven by a
coupling term `Q^m eta`. `--secondary` (with `--order-parameter`) lists
these **secondary order parameters** after the isotropy subgroup:

```bash
crystod-group --parent Pm-3m --irrep R4+ --order-parameter a 0 0 --secondary
```

```
...
* Isotropy subgroup *
R4+(a,0,0) -> I4/mcm (No. 140)
cell size 2, index 6
...

* Secondary order parameters *
H = I4/mcm (140), index 6
irrep  k              dim  n_free  direction   coupling  type
GM1+   (0,0,0)        1    1       (a)         Q^2 eta   strain
GM3+   (0,0,0)        2    1       (0.577a,a)  Q^2 eta   strain
R4+    (1/2,1/2,1/2)  3    1       (a,0,0)     -         primary
coupling: lowest invariant Q^m eta (Q primary, eta this irrep) that does not vanish on the fixed space of H, total degree <= 4
...
```

- `irrep`, `k`, `dim`: the ISO-IR label (pair label for a complex pair), the
  tabulated k vector of its star (primitive basis) and the order-parameter
  dimension.
- `n_free`: the number of free parameters of the irrep fixed by H, i.e. the
  dimension of its fixed space, computed as the average of the character
  over H.
- `direction`: that fixed space as a direction pattern of the irrep (its own
  letters, starting from `a`).
- `coupling`: the lowest coupling term to the primary order parameter Q that
  does not vanish when Q lies in the fixed space of H (the stratum of the
  direction, which is the direction itself unless it is non-generic, such as
  Fm-3m W1 `(a;0;0;0;0;0)` whose fixed space is `(a;b;0;0;0;0)`),
  `Q^m eta` (`Q1^i Q2^j eta` for two primary irreps; equally low terms are
  separated by `, `); `> N` when there is none up to total degree N
  (`--degree`, default 4), and `> N (monomial limit)` when the search stopped
  because the next degree has more than 20000 monomials. It is `-` for the
  primary irreps.
- `type`: `primary` (a given irrep), `polar` (a Gamma irrep other than the
  identity in the polar vector representation: the subgroup is polar),
  `strain` (the identity irrep, or a Gamma irrep in the symmetric square of
  the vector representation: a spontaneous strain), `other`.
- A listed irrep is *allowed* to appear: its component is not forbidden by
  the symmetry of H, and the coupling term says at which order in Q it is
  driven. The table does not say how large it is; that needs the coupling
  coefficients of the free energy.

The I4/mcm tilt phase thus has a volume strain (GM1+) and a tetragonal
strain (GM3+) and nothing polar. For the hybrid improper ferroelectric,
`--parent I4/mmm --irrep X2+ X3- --order-parameter 0 a 0 c --secondary`
(H = Cmc2_1) lists `GM5-` as `polar` with the coupling `Q1^1 Q2^1 eta`, and
for the a-a-c+ perovskite tilts, `--parent Pm-3m --irrep R4+ M3+
--order-parameter 0 a a d 0 0 --secondary` (H = Pnma) lists X5+ (`Q1^1 Q2^1
eta`), R5+ and M2+ besides the Gamma strains: the standard secondary modes of
Pnma perovskites.

**Limits.** The candidates are the irreps of the Gamma point and of the tabulated special
k points with a star arm in the reciprocal lattice of the subgroup's
translations (the others average to zero over those translations). Points of
that reciprocal lattice on symmetry lines or planes have no tabulated irreps
and are not covered; when there are any, a `not covered` line lists them.

**Python API.** The invariants come as an `InvariantBasis`, the Landau and
Lifshitz conditions as a `LandauLifshitz` record, the lowest coupling term as
a `CouplingTerm` (`None` when there is none up to `max_degree`), and the
secondary order parameters as a list of `SecondaryOrderParameter` records.
The MCP tool is `crystod_invariants` (`space_group`, `irreps`, `degree`, and
an optional `direction` that adds the restricted free energy and the
secondary order parameters):

```python
from crystod import group

basis = group.invariant_polynomials("Pm-3m", ["R4+"], degree=4)
print(basis.counts)                         # {1: 0, 2: 1, 3: 0, 4: 2}
print(basis.polynomials[4][0].expression)   # Q1^2*Q2^2 + Q1^2*Q3^2 + Q2^2*Q3^2
print(group.landau_lifshitz("Pm-3m", "R4+"))
# LandauLifshitz(n_cubic=0, n_lifshitz=0, continuous_allowed=True)
term = group.coupling_terms("I4/mmm", ["X2+", "X3-", "GM5-"], max_degree=3)
print(term.multidegree, term.trilinear)     # (1, 1, 1) True
for row in group.secondary_order_parameters("Pm-3m", "R4+", "a 0 0"):
    print(row.label, row.n_free, row.direction_pattern, row.coupling, row.kind)
```

`crystod.invariants.invariants_of_matrices(matrices, parts, degree)` takes any
finite group of real orthogonal matrices (one per element), e.g. a point-group
representation.

## 19. Reverse lookup (`--child`)

*Example directory: `example/19_child_lookup` (testsuite section 19); after pip install: `crystod-example 19`*

Section 17 goes from an irrep to its subgroups. `--child` goes the other
way: given the parent G and the space-group type H of an observed
low-symmetry phase (symbol or number), it lists every order-parameter
direction of every irrep at the special k points of G whose isotropy
subgroup has type H, and with `--coupled` the pairs of irreps that give H
together. It answers "which distortion of the parent can produce this
phase?", the offline counterpart of the reverse search of ISOSUBGROUP. Each
row is a stratum: the set of order-parameter directions with one isotropy
subgroup, up to conjugacy in G.

The single-irrep form:

```bash
crystod-group --parent Pm-3m --child I4/mcm
```

```
building the isotropy table of Pm-3m: GM (1 of 4) ...
building the isotropy table of Pm-3m: R (2 of 4) ...
building the isotropy table of Pm-3m: X (3 of 4) ...
building the isotropy table of Pm-3m: M (4 of 4) ...

* Supergroup *
Pm-3m (No. 221)

* Subgroup *
I4/mcm (No. 140)

* Isotropy table *
40 irreps at the special k points GM, R, X, M; 243 strata
cache: ~/.cache/crystod/isotropy/221_0.4.6.json.gz (written)

* Isotropy subgroups of type I4/mcm *
irrep  k  direction      size  index  conventional basis         origin
R3+    R  (0,a)          2     6      (0,-1,1),(0,1,1),(-2,0,0)  (0,1/2,1/2)
R4+    R  (0,0,a)        2     6      (-1,0,1),(1,0,1),(0,2,0)   (0,0,0)
R3-    R  (a,0)          2     6      (0,-1,1),(0,1,1),(-2,0,0)  (1/2,0,0)
R5-    R  (0,0,a)        2     6      (-1,0,1),(1,0,1),(0,2,0)   (1/2,1/2,1/2)
M5+    M  (0,0;0,a;a,0)  4     12     (2,0,0),(0,2,0),(0,0,2)    (0,0,0)
M5+    M  (0,0;a,0;0,a)  4     12     (2,0,0),(0,2,0),(0,0,2)    (1/2,1/2,1/2)
M1-    M  (0;a;a)        4     12     (2,0,0),(0,2,0),(0,0,2)    (0,0,1/2)
M2-    M  (0;a;a)        4     12     (2,0,0),(0,2,0),(0,0,2)    (0,0,1/2)
M3-    M  (0;a;a)        4     12     (2,0,0),(0,2,0),(0,0,2)    (1/2,1/2,0)
M4-    M  (0;a;a)        4     12     (2,0,0),(0,2,0),(0,0,2)    (1/2,1/2,0)
...
```

How to read it:

- `irrep`, `k`, `direction`: the stratum, exactly as
  `--parent Pm-3m --kpoint K` lists it; `--parent SG --irrep IR
  --order-parameter ...` with that direction reproduces the row. The
  octahedral tilt `R4+(0,0,a)` of the perovskites is the second row.
- `size`, `index`: the cell size (primitive volume ratio) and the index
  [G:H].
- `conventional basis`, `origin`: the setting of the subgroup (rows in
  parent conventional units), as `--order-parameter` prints it; the origin is
  reduced into [0,1) by a parent lattice translation, so `--order-parameter`
  may print it shifted by whole cell vectors.
- The progress lines and the `* Isotropy table *` block report the cached
  table the lookup reads (below).

**The isotropy table.** The first `--child` run for a parent enumerates all
strata of all irreps at all special k points of the ISO-IR tables (one
progress line per k point on stderr) and writes them, with their settings,
a generic order-parameter vector and the projector onto every stratum, to

```
$CRYSTOD_CACHE_DIR/isotropy/<number>_<version>.json.gz
```

where `CRYSTOD_CACHE_DIR` defaults to `~/.cache/crystod` and `<version>` is
the installed CrystOD version (the table follows the label and setting
conventions of that version, so no table is shipped with the package). Later
runs read the file (the `(read)` state), so a lookup takes about a second;
`--no-cache` rebuilds the table and rewrites the file. The file also records
a fingerprint of the CrystOD sources it was built with (the enumeration,
label and setting code and the ISO-IR data); a file with another fingerprint
is rebuilt, so an updated installation never reads a stale table. Building takes a few
seconds for most parents (Pm-3m 5 s, I4/mmm 2 s) and about a minute for the
large W stars of the face-centred cubic groups.

**Several rows of one type.** A subgroup type can be embedded in G in several
ways (orientations of the subgroup axes, origins of its cell); each embedding
is its own stratum and its own row, told apart by the conventional basis and
origin: the two `M5+` rows above are I4/mcm cells with their origins at
different points of the parent cell. Different irreps of one type (`R4+` and `R5-`) also give
different embeddings. A complex-type pair whose partner sits at the -k star
(`H1HA1` of P3_1 spans H and HA) is one order parameter and is listed once,
at the first of the two stars; `--kpoint HA` selects the same rows.

**Filters.** `--size N` and `--index N` keep the strata with that cell size or
index [G:H]; `--kpoint K [K ...]` keeps those k points (names, or three
coordinates of any arm of one point). The filters are listed on a `selected:`
line:

```bash
crystod-group --parent Pm-3m --child 140 --index 12 --kpoint M
```

**No single irrep.** A subgroup that needs two coupled order parameters is not
in the table. The perovskite Pnma tilt system (a-b+a-, cell size 4) is the
coupled R4+ + M3+ result of section 17; a single irrep gives Pnma only with
cell size 8 (`X5+`, `X5-`, as in ISOSUBGROUP):

```bash
crystod-group --parent Pm-3m --child Pnma --size 4
```

```
...
* Isotropy subgroups of type Pnma *
no single-irrep stratum of type Pnma (No. 62) with size 4;
the subgroup may need two coupled irreps (--coupled)
...
```

**Coupled pairs (`--coupled`).** `--coupled` searches the pairs of irreps,
first the pairs of two irreps at the same k point, then the pairs at
different k points, and lists every coupled stratum (both amplitudes
nonzero) whose isotropy subgroup has type H, in a block after the
single-irrep one:

```bash
crystod-group --parent Pm-3m --child Pnma --size 4 --coupled --kpoint R M
```

```
...
* Coupled isotropy subgroups of type Pnma *
20 irreps pass the point-group filter; 190 pairs searched
(90 at the same k point, 100 at different k points)
irrep1  dir1     irrep2  dir2     size  index  conventional basis           origin
R4+     (0,a,a)  M2+     (d;0;0)  4     24     (-1,-1,0),(0,0,-2),(1,-1,0)  (0,0,0)
R4+     (0,a,a)  M3+     (d;0;0)  4     24     (-1,-1,0),(0,0,-2),(1,-1,0)  (0,0,0)
R4+     (0,a,a)  M2-     (d;0;0)  4     24     (0,0,2),(-1,-1,0),(1,-1,0)   (0,1/2,1/2)
...
R5-     (0,a,a)  M3-     (d;0;0)  4     24     (0,0,2),(-1,-1,0),(1,-1,0)   (0,1/2,0)
every row needs both irreps: neither irrep alone has this isotropy subgroup
...
```

The second row is the tilt system a-b+a- of section 17 (`R4+(0,a,a)
M3+(d;0;0)`), with the setting `--order-parameter 0 a a d 0 0` prints; the
other rows are the same space-group type from other pairs (other atoms
moving, or the cell shifted). Without `--kpoint` all 780 pairs of the 40
irreps of Pm-3m are searched (32 rows of size 4, about 10 s). The directions
use the letters of `--parent SG --irrep IR1 IR2` (the second irrep's letters
continue after the first one's), and a row is listed only when both irreps
are needed: a pair in which one irrep alone already has the subgroup (the
other one being one of its secondary order parameters) is left out. Other
known cases: `--parent I4/mmm --child Cmc2_1 --coupled` gives the hybrid
improper ferroelectric `X2+ (0;a)` + `X3- (0;c)` (size 2, index 8) among
others, and `--parent Cmcm --child Pna2_1 --coupled` the pair `Y4+ (a)` +
`Y2- (b)`.

How the pairs are searched: every isotropy subgroup of a coupled order
parameter is `S1 & S2'`, with `S1` the isotropy subgroup of a single stratum
of the first irrep (the representative of the table) and `S2'` any conjugate
of one of the second, and its fixed space is the sum of the fixed spaces in
the two irreps. The single strata of the cached table are the candidates;
only irreps with a stratum whose point group has an order divisible by that
of H take part (and, with `--size` and `--index`, whose size and index
divide the requested ones), and a candidate is identified with spglib only
after its point group and cell size match. One progress line per pair of k
points goes to stderr once the search has run for a few seconds.
`--secondary` adds, for every row, the secondary order parameters of section
18 (`* Secondary order parameters *` blocks, as `--parent SG --irrep IR1 IR2
--order-parameter ... --secondary` prints them; `--degree N` sets the
highest coupling degree).

**Enantiomorphic partners.** In a parent with improper operations, the two
members of an enantiomorphic pair (P4_122/P4_322, P4_132/P4_332, ...) are
stabilizers of order parameters of one stratum related by an improper
operation of the parent, so either may be listed (section 17). A stratum
whose listed type is the partner of H then counts as a match, and a `note:`
names those rows:

```bash
crystod-group --parent P4_2/mcm --child P4_322
```

```
...
* Isotropy subgroups of type P4_322 *
irrep  k  direction  size  index  conventional basis       origin
Z3     Z  (0,a)      2     4      (1,0,0),(0,1,0),(0,0,2)  (0,0,3/4)
Z4     Z  (0,a)      2     4      (1,0,0),(0,1,0),(0,0,2)  (0,0,1/4)

note: 95 <-> 91 (P4_322 <-> P4_122) are enantiomorphic partner types: an
improper operation of the parent maps the order parameter of such a stratum to
another one of the same stratum whose stabilizer is the partner. The rows
Z3(0,a) at Z, Z4(0,a) at Z are listed with a stabilizer of type P4_122
(No. 91).
...
```

A Sohncke parent (no improper operation, e.g. P4_222 or P3_1) is different:
there the two types are different, non-conjugate subgroups from
different strata, only rows of type H itself are listed and no note is
printed (`--parent P4_222 --child P4_122` gives `Z1` and `Z3`, while `Z2` and
`Z4` give P4_322, as in ISOSUBGROUP).

**Limits.** Only the special k points of the ISO-IR tables are covered (no
symmetry lines or planes), and `--coupled` searches pairs of two distinct
irreps (not three or more order parameters, and not two copies of one
irrep).

**Python API.** The matches come as a list of `IsotropyMatch` records, the
coupled ones as `CoupledIsotropyMatch` records and the whole table as an
`IsotropyTable`:

```python
from crystod import group

for match in group.find_isotropy_irreps("Pm-3m", "R-3c", size=2):
    print(match)        # R4+(a,a,a) (R) -> R-3c (No. 167), size 2, index 8 ...
for match in group.find_coupled_isotropy_irreps("Pm-3m", "Pnma", size=4):
    print(match)        # R4+(0,a,a) X5+(0,0;d,d;0,0) -> Pnma (No. 62), size 4, index 24 ...
table = group.isotropy_table("Pm-3m")       # cached as above
for stratum in table.select([140], kpoints=["R"]):
    print(stratum.label, stratum.direction, stratum.basis, stratum.vector)
```

`find_isotropy_irreps(..., coupled=True)` returns both kinds in one list.
The MCP tool is `crystod_find_isotropy_irreps` (`parent`, `child`, the
optional filters `size`, `index`, `kpoints`, and `coupled`), which reads the
same cached table.

## 20. Group-subgroup graph (`--graph`)

*Example directory: `example/20_subgroup_graph` (testsuite section 20); after pip install: `crystod-example 20`*

`--graph` draws the isotropy subgroups of one irrep, or of the direct sum of
several, as a group-subgroup lattice: the parent at the top, every stratum,
and the kernel (the subgroup of the generic order parameter) at the bottom,
in layers by index. It shows at a glance which phases a set of order
parameters can produce and which of them are group-subgroup related (a
requirement for a continuous transition between two of them), the picture
behind tilt classifications such as that of Howard and Stokes. The group
theory: a line joins two isotropy subgroups when one is a maximal subgroup
of a conjugate of the other among the nodes. For the two perovskite tilts:

```bash
crystod-group --parent Pm-3m --irrep R4+ M3+ --graph --graph-dot
```

```
...
* Group-subgroup graph *
25 nodes (the parent and 24 strata), 56 edges (0 dashed)
  index 1: Pm-3m (221)
  index 6: P4/mbm (127), I4/mcm (140)
  index 8: R-3c (167), Im-3 (204)
  index 12: Imma (74), I4/mmm (139), P4/mbm (127)
  index 24: C2/m (12), C2/c (15), Pnma (62), Cmcm (63), P4_2/nmc (137),
    Immm (71)
  index 48: Cmcm (63), P-1 (2), P2_1/m (11), C2/m (12), P2_1/c (14), Pmmn (59)
  index 64: R-3 (148)
  index 96: C2/c (15), P-1 (2), P2_1/m (11)
  index 192: P-1 (2)

* Nodes *
node  subgroup        size  index  direction              maximal in   from parent
1     Pm-3m (221)     1     1      -                      -            -
2     P4/mbm (127)    2     6      M3+(0;0;d)             1            solid
3     I4/mcm (140)    2     6      R4+(0,0,a)             1            solid
...
11    Pnma (62)       4     24     R4+(0,a,a) M3+(d;0;0)  2,6          -
...
25    P-1 (2)         8     192    R4+(a,b,c) M3+(d;e;f)  21,22,23,24  -
maximal in: the nodes in which this subgroup is maximal (among the nodes)
from parent: the line from the parent; dashed when the Landau or the
Lifshitz condition of its irrep fails (no continuous transition)

* Output files *
  SUBGROUP_Pm-3m_R4+_M3+.html
  SUBGROUP_Pm-3m_R4+_M3+.dot
...
```

How to read it:

- `* Group-subgroup graph *` lists the subgroups layer by layer (index
  [G:H]); `edges` counts the lines, `dashed` those from the parent that
  cannot be continuous (below).
- `* Nodes *` has one row per stratum: every stratum of the representation,
  including those in which only some of the irreps condense (the
  single-irrep tables of section 17, `R4+(0,0,a)` = I4/mcm, `M3+(0;0;d)` =
  P4/mbm, ...). Two nodes of one type (P4/mbm with cell size 2 and 4) are
  different embeddings, told apart by their directions.
- `maximal in` lists the nodes a node hangs from: B lies below A when some
  parent image of the fixed space of A lies inside the fixed space of B (for
  isotropy subgroups, H_B is a subgroup of a conjugate of H_A); the inclusion
  is transitively reduced.
- `from parent` is `solid` or `dashed` for a line from the parent and `-`
  for a node that does not hang from it.

The 15 tilt systems of Howard and Stokes (a0a0a0 Pm-3m, a+a+a+ Im-3, a+b+c+
Immm, a0a0c+ P4/mbm, a0b+b+ I4/mmm, a0a0c- I4/mcm, a0b-b- Imma, a-a-a- R-3c,
a0b-c- C2/m, a-b-b- C2/c, a-b-c- P-1, a+a+c- P4_2/nmc, a0b+c- Cmcm, a+b-b-
Pnma, a+b-c- P2_1/m) are 15 of the 25 nodes, and the lines of the graph among
them are the 25 group-subgroup relations that their Fig. 1 draws (the
testsuite derives this list independently from the Glazer patterns of their
Table 1, by setting tilts equal or zero). The other ten nodes are
the strata in which one axis carries an in-phase and an out-of-phase tilt at
once (R-3, P2_1/c, Pmmn, and second embeddings of P4/mbm, Cmcm, C2/m, C2/c,
P2_1/m and P-1), which the tilt classification leaves out; the generic
P-1 with cell size 8 is the kernel.

**Dashed lines.** A line from the parent is dashed when the irrep that alone
gives the stratum violates the Landau condition (a cubic invariant) or the
Lifshitz condition (a Lifshitz invariant) of section 18, so that the
transition from the parent cannot be continuous; it is also dashed when the
stratum needs several irreps at once. R4+ and M3+ have neither invariant, so
every line above is solid; the breathing irrep M1+ has the cubic invariant
`Q1*Q2*Q3`:

```bash
crystod-group --parent Pm-3m --irrep M1+ --graph
```

```
...
* Nodes *
node  subgroup      size  index  direction   maximal in  from parent
1     Pm-3m (221)   1     1      -           -           -
2     Im-3m (229)   4     4      M1+(a;a;a)  1           dashed
3     P4/mmm (123)  2     6      M1+(0;0;a)  1           dashed
4     I4/mmm (139)  4     12     M1+(a;a;b)  2,3         -
5     Immm (71)     4     24     M1+(a;b;c)  4           -
maximal in: the nodes in which this subgroup is maximal (among the nodes)
from parent: the line from the parent; dashed when the Landau or the
Lifshitz condition of its irrep fails (no continuous transition)
...
```

Lines between two strata are drawn solid: whether the transition between two
low-symmetry phases can be continuous (the cubic terms of the free energy
restricted to the smaller fixed space) is not tested.

**Files.** The HTML page `SUBGROUP_<SG>_<irreps>.html` (the symbol without
`/`, e.g. `SUBGROUP_I4mmm_X2+_X3-.html`; `--output FILE` renames it) holds
the graph as an inline SVG, about 20 kB, with no external libraries;
hovering a node shows its direction, the condensing and the primary irreps,
the cell size, the index, the number of free parameters, the conventional
basis and the origin, and highlights its lines. `--graph-dot` also writes
the graph in the Graphviz DOT language next to it (`.dot`, the layers as
`rank=same`), for `dot -Tpdf`.

**Limits.** The nodes are the strata of the irreps given (special k points
only), not every subgroup of G; lines between two strata are not tested for
continuity.

**Python API.** `group.subgroup_graph("Pm-3m", ["R4+", "M3+"])` returns a
`SubgroupGraph` with the `GraphNode` and `GraphEdge` records, the full
inclusion relation (`inclusions`, `contains(a, b)`) and the Landau and
Lifshitz records of the irreps; `subgroup_graph.format_html` and
`format_dot` render it. The MCP tool is `crystod_subgroup_graph` (`parent`,
`irreps`; the nodes by index layer and the solid or dashed edge list, as
text).

```{seealso}
C. J. Howard and H. T. Stokes, "Group-Theoretical Analysis of Octahedral
Tilting in Perovskites", Acta Cryst. B54, 782-789 (1998).
```

## 21. Multi-electron terms (`--multiplet`)

*Example directory: `example/21_multiplet` (testsuite section 21); after pip install: `crystod-example 21`*

The Pauli-allowed many-electron states (spin multiplicity 2S+1 + spatial
irrep) of an electron configuration over point-group irrep shells, sorted by
descending spin multiplicity (Hund-rule ground term first):

```bash
crystod-group --multiplet T2g2 --pg m-3m --orbital d
# -> (T2g)^2 = ^3T1g + ^1A1g + ^1Eg + ^1T2g   (15 states = C(6,2))

crystod-group --multiplet T2g2 Eg1 --pg m-3m
# -> ^4T1g + ^4T2g + ^2A1g + ^2A2g + 2(^2Eg) + 2(^2T1g) + 2(^2T2g)
```

Shell tokens are written `T2g2` or `T2g^2` (equivalent; the ^-free form
needs no quoting in shells where `^` is a glob character, e.g. zsh).

The ground-state term symbol is always printed (Hund's rules); with
`--orbital`, the exact Coulomb multiplet energies of every term are
computed in Racah parameters (A, B, C for d shells; reduced Slater-Condon
F_k otherwise) and the ground state is determined by energy:

```bash
crystod-group --multiplet T2g3 --pg m-3m --orbital d
```

```
* Multiplet Energies (Racah parameters A, B, C; Coulomb part only) *
^4A2g: 3A - 15B
^2Eg : 3A - 6B + 3C
^2T1g: 3A - 6B + 3C
^2T2g: 3A + 5C

* Ground-state Term Symbol (within this configuration) *
^4A2g   (lowest for any B > 0, C > 0)
```

— the Tanabe-Sugano strong-field table of (t2g)^3. The energies come from
the Coulomb Hamiltonian over the Slater determinants of the configuration,
each term isolated by S^2 and point-group projectors; doubly-occurring terms
mix (configuration interaction) and their two energies are printed in closed
form (e.g. 3A - 3B + 3C +- 3sqrt(2)B in (t2g)^2(eg)^1).

```{seealso}
**Theory:** [How the multiplet energies are computed](theory-representations.md) — the Gaunt coefficients, the Racah / Slater-Condon reduction, the CI matrices, and the internal consistency checks.
```

f shells are fully supported (`--orbital f`; A2u + T1u + T2u shells in
m-3m), with energies in the reduced Slater-Condon parameters F0, F2, F4,
F6 — e.g. `--multiplet T1u3 --pg m-3m --orbital f` gives
^4A1u = 3F0 - (105/4)F2 - (189/2)F4 - (3705/4)F6 as the ground state (the
f analogue of (t2g)^3 -> ^4A2g); numeric CI blocks and the ground-state
selection use the hydrogenic 4f ratios F4/F2 = 0.138, F6/F2 = 0.0151.

Several tokens denote inequivalent shells, coupled by spatial direct
products and spin angular-momentum addition; the optional
`--orbital s|p|d|f` prints the ligand-field splitting of the parent
atomic orbital (section 11) and verifies the occupied shells occur in it.
Every result closes with a state-count check (product of C(2 dim, n)).
`--orbital` takes only s, p, d and f, because the Coulomb multiplet energies
are implemented for l <= 3. The g, h and i shells are refused with an `ERROR`
and exit status 1, before any term is printed:

```bash
crystod-group --multiplet Eg2 --pg m-3m --orbital g
```

```
* Point group *
m-3m

ERROR: orbital 'g' is not supported: the Coulomb multiplet energies are implemented for l <= 3 only (the splitting alone: crystod-group --ligand-field g --pg m-3m). Choose from: s, p, d, f
```

The ligand-field splitting of these shells alone is `--ligand-field`
(section 11), which handles every shell up to i.

`--multiplet` applies the Pauli principle exactly: of the plain product
T2g x T2g (section 8) only the antisymmetric square pairs with the spin
triplet, so hole equivalence and closed shells come out automatically
((t2g)^4 gives the (t2g)^2 terms, (t2g)^6 gives ^1A1g).

```{seealso}
**Theory:** [The Pauli principle and the CI matrices](theory-representations.md) — the antisymmetrization, the CI matrices in the coupled-parent basis of the Tanabe-Sugano / Griffith strong-field tables, and the validation against the standard crystal-field term tables.
```

### Visualizing the term eigenstates (`--visualize`)

With `--orbital`, `--visualize` writes the **exact eigenstates of every term** as an interactive HTML page (`Multiplet_{pg}_{config}.html`): the term list in a sidebar (Hund/energy ground state marked), and for the selected term the full **Slater-determinant expansion** — every determinant drawn as an orbital box diagram (t2g: dxy, dyz, dxz | eg: dz2, dx2-y2, identified from the parent orbital) with up/down arrows and the exact expansion coefficient (1, ±1/2, ±1/√2, ±√3/2, ...):

```bash
crystod-group --multiplet "T2g^2" --pg m-3m --orbital d --visualize
```

The page below is the live output of that command — the four terms of
(t2g)^2 (`^3T1g + ^1A1g + ^1Eg + ^1T2g`, the Hund ground term `^3T1g` marked)
in the sidebar; pick one to see its Slater-determinant expansion as orbital
box diagrams and the drag-rotatable charge/spin-density surface of that
eigenstate:

```{raw} html
<iframe src="_static/embed/Multiplet_m-3m_T2g2.html" width="100%" height="660" loading="lazy" style="border:1px solid #8884; border-radius:8px; background:#fff;"></iframe>
<p style="margin-top:0.3em"><a href="_static/embed/Multiplet_m-3m_T2g2.html" target="_blank">Open the (t2g)<sup>2</sup> multiplet viewer full-screen</a></p>
```

A term eigenstate is in general a superposition of determinants, not a single
box configuration: the ^4A2g of (t2g)^3 *is* the single determinant
|dxy↑ dyz↑ dxz↑⟩ (coefficient 1), while a ^4T1g partner of (t2g)^2(eg)^1 is
√3/2 |dx2-y2↑; dxy↑ dyz↑⟩ + 1/2 |dz2↑; dxy↑ dyz↑⟩. What the page shows:

- states at the highest spin projection Ms = S, with degenerate spatial
  partners switchable (canonicalized, so any orthogonal mixture is
  equivalent);
- one tab per state for configuration-mixed terms (the CI pairs, e.g. the two
  ^2T1g), each with its Coulomb energy at the reference parameters;
- a drag-rotatable 3D surface of the **charge and spin density** (angular
  part) of every state, computed from the one-particle reduced density matrix
  of the term eigenstate and colored by the local spin polarization — the
  real-space picture behind orbital ordering and Jahn-Teller physics. The
  (t2g)^3 ^4A2g shows the cubic-symmetric t2g flower, fully spin-polarized;
  the ^2Eg partners keep the cubic charge density but carry an anisotropic
  spin density.

`--output` selects the file name. For f shells, symmetry-mixed basis functions
(e.g. the t1u combination of fx(x2-3y2) and fxz2) get short symbols `t1u(1)`,
... in the boxes, expanded in an *Orbital basis functions* legend on the page.

## 22. POSCAR <-> CIF (`--poscar2cif` / `--cif2poscar`)

*Example directory: `example/22_poscar2cif` (testsuite section 22); after pip install: `crystod-example 22`*

Convert a POSCAR into a CIF laid out like the files of the Bilbao
Crystallographic Server, and back:

```bash
crystod-group --poscar2cif -c 221_PPOSCAR_SrTiO3 [--tolerance 0.01]
```

```
* Structure *
input      : 221_PPOSCAR_SrTiO3
space group: Pm-3m (No. 221), tolerance 0.01
48 symmetry operations, 3 independent sites

* Output files *
  Bilbao-style CIF written to: 221_PPOSCAR_SrTiO3.cif
```

```
data_221_PPOSCAR_SrTiO3
_chemical_formula_sum              "Sr Ti O3"
_symmetry_Int_Tables_number        221
_symmetry_space_group_name_H-M     "Pm-3m"
_cell_length_a                     3.9451
...
loop_
_symmetry_equiv_pos_site_id
_symmetry_equiv_pos_as_xyz
   1   x,y,z
...
O1 O 0.50000 0.00000 0.50000 1.0000
Sr1 Sr 0.00000 0.00000 0.00000 1.0000
Ti1 Ti 0.50000 0.50000 0.50000 1.0000
```

and back:

```bash
crystod-group --cif2poscar -c 221_PPOSCAR_SrTiO3.cif [--conventional]
# -> 221_PPOSCAR_SrTiO3 (primitive cell; --conventional for the conventional cell)
```

The structure is brought to the spglib-standardized conventional cell (the
ITA setting and origin, as used by Bilbao); the CIF lists the space-group
number, the quoted Hermann-Mauguin symbol, 4-decimal cell parameters, the
full conventional-cell symmetry operations as compact `x+1/2,-y,z` strings
(proper operations first, centring translations included — 192 for Fm-3m),
and one representative site per Wyckoff orbit (5-decimal coordinates,
occupancy 1.0000). This differs from the pymatgen `CifWriter` layout;
`--output` overrides the default `<POSCAR>.cif` path. Validated against a
Bilbao reference file (identical operator set), the ITA Pnma general
positions, and pymatgen round-trip re-reading.

The [`_chemical_formula_sum`](https://www.iucr.org/__data/iucr/cifdic_html/1/cif_core.dic/Cchemical_formula.html)
field carries the reduced formula in the conventional chemical order —
cations before anions; among the cations, the element on the most special
Wyckoff site (the letter closest to *a*) first, then increasing valence.
That is what makes SrTiO3, KNbO3 and PbZrO3 come out in the familiar
order (site tie broken by valence — electronegativity alone would write
ZrPbO3), and La3Ni2O7 keep La first (the 2-fold site beats Ni's 4-fold
one even though Ni carries the lower valence). Valences are
oxidation-state guesses; electronegativity is the fallback.

`--cif2poscar` accepts any CIF flavour (Bilbao or pymatgen), expands the
symmetry operations, and writes the spglib-standardized primitive cell —
the working format of the other crystod commands — as a POSCAR in the
crystod test-file style (6-decimal `direct` coordinates with element tags)
to the input path without `.cif`. Round trips reproduce the original
primitive structure exactly (SrTiO3, ScF3, F-centred NaCl: 2-atom
primitive by default, 8-atom conventional with `--conventional`).

### Axis settings (`--cif-axis`, `--spacegroup-set`)

A CIF or POSCAR is usually written in the default setting of its space
group, but a space group with a choice of axes, unique axis, cell choice,
origin or rhombohedral/hexagonal axes has several settings, the Hall-symbol
entries of the International Tables (90 of the 230 groups, up to 18 for the
monoclinic ones). `--cif-axis` lists every setting of the space group of a
CIF or POSCAR, in the order of VESTA's "Setting" list and the International
Tables, with the lattice parameters the file's own cell has in each of them:

```bash
crystod-group --cif-axis -c POSCAR_La3Ni2O7_Cmcm.cif [--tolerance 0.01]
```

```
* Structure *
input      : POSCAR_La3Ni2O7_Cmcm.cif
space group: Cmcm (No. 63), tolerance 0.01
input cell : a = 20.31300, b = 5.60630, c = 5.45710 A, alpha = 90.0000, beta = 90.0000, gamma = 90.0000 deg

* CIF axis settings *
No.  setting              a (A)       b (A)       c (A)     alpha      beta     gamma
1    Cmcm (a, b, c)    20.31300     5.60630     5.45710   90.0000   90.0000   90.0000   <- input
2    Ccmm (b, a, -c)    5.60630    20.31300     5.45710   90.0000   90.0000   90.0000
3    Amma (c, a, b)     5.45710    20.31300     5.60630   90.0000   90.0000   90.0000
4    Amam (-c, b, a)    5.45710     5.60630    20.31300   90.0000   90.0000   90.0000
5    Bbmm (b, c, a)     5.60630     5.45710    20.31300   90.0000   90.0000   90.0000
6    Bmmb (a, -c, b)   20.31300     5.45710     5.60630   90.0000   90.0000   90.0000
(the Hall-symbol settings of spglib, in the order of VESTA and the International Tables;
 lengths and angles of the input cell in the axes of each setting, x_setting = P x_input + p;
 write one with --cif2poscar -c FILE --spacegroup-set <symbol | No. | choice>)
```

The listed lengths and angles are those of the input cell itself in the axes
of each setting (not idealized), so the table answers "which axis is the
long one in Amam?" directly. Fractional coordinates transform as
`x_setting = P x_input + p` (column form), and the lattice vectors of the
setting are the rows of `inv(P).T L_input`. The first setting is spglib's
standardization of the input (the input itself when it is the conventional
cell of one of the settings); the other settings of a monoclinic or
orthorhombic group follow from it with the matrices of the International
Tables, so the lengths always follow the axes text: the axis permutation
(`(-c, b, a)`: a' = -c, b' = b, c' = a) and, for the monoclinic cell
choices, a2 = -a1 - c1, c2 = a1 and a3 = c1, c3 = -a1 - c1 (unique axis b;
the other unique axes permute these). The settings of the other groups
(origin choices, hexagonal or rhombohedral axes) are spglib's standardization
into each Hall setting.

`<- input` marks the setting the file is in: the input cell has exactly the
symmetry operations of that Hall symbol, origin included, and its row then
has `P` = 1 and `p` = 0. In No. 68 (Ccce, Aeaa, Bbeb) the origin-choice-1
settings share their Hall symbols in pairs (`1` and `1ba-c`, `1cab` and
`1-cba`, `1bca` and `1a-cb`), so a file in one of them marks both rows of the
pair; the second row lists the permuted lengths of the same cell. A primitive
cell of a centred lattice, a supercell or a cell with a shifted origin is the
conventional cell of no listed setting; the table then gives the parameters
of the conventional cell of every setting and ends with the line
`input cell : not the conventional cell of a listed setting`.

The setting symbol is spglib's, except that a monoclinic symbol drops the `1`
axes of the full symbol (`P 1 2_1/n 1` -> `P2_1/n`, where spglib writes
`P2_1/c` for all nine settings of No. 14) and an orthorhombic symbol is read
from the full symbol (the `B 2/b 2/e 2/b` setting of No. 68 is `Bbeb`). The
axes column spells out spglib's choice: an axis permutation `(-c, b, a)`, a
unique axis and cell choice `(unique axis b, cell choice 2)`, an origin
choice `(origin choice 2)`, both `(b, a, -c; origin choice 1)`, or
`(hexagonal axes)` / `(rhombohedral axes)`.

`--cif2poscar --spacegroup-set SETTING` writes the structure in one of the
listed settings; `SETTING` is the symbol (spaces, `/` and `_` may be left
out: `Amam`, `'A m a m'`, `P21n`), the No. of the table (`4`) or spglib's
choice string (`-cba`, `b2`, `2`, `R`, `2ba-c`). The input may be a CIF or a
POSCAR:

```bash
crystod-group --cif2poscar -c POSCAR_La3Ni2O7_Cmcm.cif --spacegroup-set Amam
```

```
* Structure *
input      : POSCAR_La3Ni2O7_Cmcm.cif
space group: Cmcm (No. 63), tolerance 0.01
setting    : Amam (-c, b, a), No. 4 of 6
transform  : x_Amam = P x_input + p, P rows (0, 0, -1), (0, 1, 0), (1, 0, 0); p = (0, 0, 0)
conventional cell (Amam setting), 48 atoms: a = 5.45710, b = 5.60630, c = 20.31300 A, alpha = 90.0000, beta = 90.0000, gamma = 90.0000 deg

* Output files *
  POSCAR written to: POSCAR_La3Ni2O7_Cmcm_Amam
```

The POSCAR is the conventional cell of that setting (so `--conventional` is
implied): spglib's idealized conventional cell of the first setting, the
same standardization `--cif2poscar --conventional` writes, carried into the
chosen setting by the transformation of the table and rotated so that a lies
along x and b in the xy plane; here c = 20.313 A is the third axis and the
atoms sit at `P x_input + p`. Re-reading it, spglib still reports Cmcm
(No. 63) in its default setting, and the file has exactly the symmetry
operations of the Amam Hall symbol (Hall number 301), so `--cif-axis` marks
row 4 as its setting.

The output is the input path without `.cif` plus `_<setting>`, the symbol
without `/` and `_` (`_Amam`, `_Pbnm`, `_Fd-3m`); `--output` overrides it.
Several settings can share one symbol: the cell choices of P2_1/n (`b2`,
`c2`, `a2`), the two origin choices of Fd-3m and of the other
centrosymmetric groups with two origins, and the hexagonal and rhombohedral
axes of the R groups. Such a symbol is refused with the list of the settings
that carry it ("give the No. or the choice"), and the file name gets the
choice as well: `14_PPOSCAR_ZrO2_P21n_b2`, `227_PPOSCAR_Si_Fd-3m_2`,
`167_PPOSCAR_AlF3_R-3c_R`. An unknown setting is refused with the list of
all settings. In Python, `crystod.group.list_space_group_settings(cell)`
returns the table (a `SettingTable` of `SpaceGroupSetting` records),
`resolve_setting(table, spec)` one setting and
`transform_to_setting(cell, spec)` its conventional cell.

## 23. Symmetry-mode analysis (`--supergroup-cif`)

*Example directory: `example/23_symmetry_mode` (testsuite section 23); after pip install: `crystod-example 23`*

Decompose the distortion between a high-symmetry and a low-symmetry
structure of the same compound into symmetry-adapted modes of the parent
space group — the offline counterpart of **AMPLIMODES** of the Bilbao
Crystallographic Server:

```bash
crystod-group --supergroup-cif 221_PPOSCAR_SrTiO3.cif --subgroup-cif 140_PPOSCAR_SrTiO3.cif
```

```
* Symmetry-mode decomposition *
k-vector         irrep   direction    isotropy subgroup   dim  amplitude (A)
(1/2,1/2,1/2)    R5-     (0,0,a)      140 I4/mcm          1    0.3303
(direction: R5- sits on its ISOTROPY representative as listed by
 crystod-group --parent 221 --irrep R5-; the input child as given sits on the
 domain-equivalent direction R5- (a,0,0))
```

The direction column names the order parameter as ISOTROPY does: `R5-(0,0,a)`
is the I4/mcm row of `crystod-group --parent Pm-3m --irrep R5-` (section 17),
and the note under the table gives the domain-equivalent direction that the
child file itself describes (see *Which order-parameter direction is
printed?* below).

A second example, the n = 2 Ruddlesden-Popper nickelate La3Ni2O7
(I4/mmm -> Cmcm), with `--conventional`:

```bash
crystod-group --supergroup-cif 139_PPOSCAR_La3Ni2O7.cif --subgroup-cif 63_PPOSCAR_La3Ni2O7.cif --conventional
```

```
* Supergroup (parent) structure *
I4/mmm (No. 139)

* Subgroup (distorted) structure *
Cmcm (No. 63)

* Cell relation *
child primitive basis in parent primitive units (rows):
  (0, 0, -1)
  (1, 1, 1)
  (1, -1, 0)
origin shift (parent primitive fractional): (1/2, 0, 1/2)
primitive cell multiplication: 2
(setting: of 8 equivalent sublattice bases the one closest to the orientation of the
 input files; the child axes are rotated 98.4 deg against the parent axes)

* Atom pairings and displacements (parent primitive setting) *
...

* Distortion amplitude *
maximum atomic displacement: 0.4097 A
total distortion amplitude : 1.1489 A
(normalized within the primitive cell of the distorted structure)

* Transformation matrix (parent -> child) *
I4/mmm (No. 139) -> Cmcm (No. 63), conventional settings of the input files
  [     0     1    -1 ] [   1/2 ]
  [     0     1     1 ] [     0 ]
  [     1     0     0 ] [   1/2 ]
(columns of P: the child basis vectors a', b', c' in parent units, (a', b', c') = (a, b, c) P;
 last column: the child origin p in parent fractional coordinates, x_parent = P x_child + p;
 the AMPLIMODES / Bilbao convention)

* Symmetry-mode decomposition *
k-vector         irrep   direction    isotropy subgroup   dim  amplitude (A)
(0,0,0)          GM1+    (a)          139 I4/mmm          4    0.1313
(0,0,1/2)        X3-     (0;a)        63 Cmcm             6    1.1413

* Normalized mode components (parent primitive fractional, per 1 A) *
...

* Output files *
  Decomposition table saved to sym_mode_La3Ni2O7
  Mode displacement VESTA files (parent conventional basis):
    display cell in parent primitive units (rows):
      (0, 2, 2)
      (2, 0, 2)
      (1, 1, 0)
    139_PPOSCAR_La3Ni2O7_GM1+_conv.vesta  (amplitude 0.1313 A)
    139_PPOSCAR_La3Ni2O7_X3-_conv.vesta  (amplitude 1.1413 A)
  Arrows are scaled so the largest displacement is 1.5 A per file; adjust in VESTA via Edit > Vectors if needed.
```

The distortion is dominated by the zone-boundary octahedral-tilt mode
`X3-(0;a)`, whose isotropy subgroup is exactly the observed Cmcm — the same
entry as in the single-irrep table of section 17, on which the child file
already sits, so that no note follows the table — while the totally
symmetric `GM1+` is only a small secondary relaxation of the free
coordinates within I4/mmm. `--conventional` writes the per-mode displacement
VESTA files in the **parent conventional basis** (the `_conv` suffix; the
body-centred I lattice makes the conventional cell twice the primitive one,
hence the printed display-cell rows), so the arrows can be inspected in the
familiar tetragonal setting instead of the primitive one.

The output also contains the automatically determined cell relation
(sublattice basis + origin shift in the parent primitive setting), the
atom-by-atom displacement table (maximum displacement, total distortion),
the transformation matrix from the parent to the child conventional setting,
the number of independent modes per irrep, and the normalized polarization
vectors; inputs may be CIFs or POSCARs, whose space groups spglib identifies
at the tolerance `--tolerance` (in Å, default 0.01). The `* Transformation matrix
(parent -> child) *` block uses the convention of AMPLIMODES and of the
International Tables: the columns of `P` are the child conventional basis
vectors in units of the parent conventional basis, `(a', b', c') = (a, b, c) P`,
and the last column is the child origin `p` in parent fractional
coordinates, so that a point with child coordinates `x'` sits at
`x = P x' + p` in the parent cell. Both cells are the conventional settings
of the input files (when spglib cannot relate an input cell to its standard
setting, the block says that it uses the standard settings instead). The
child basis is the sublattice setting the analysis uses, the one closest to
the orientation of the input files among the equivalent ones, so the matrix
can differ from the AMPLIMODES one by a point operation of the parent (for
SrTiO3 Pm-3m -> I4/mcm it is exactly the AMPLIMODES matrix,
`[[1, 1, 0], [-1, 1, 0], [0, 0, 2]]` with origin `(1/2, 1/2, 1/2)`; for a
polar subgroup the origin along the polar direction is the one that
minimizes the distortion, as in AMPLIMODES). Amplitudes follow the AMPLIMODES normalization (within the
primitive cell of the distorted structure). Multi-irrep distortions
decompose completely — the Pbnm perovskite gives R4+ -> Imma and
M3+ -> P4/mbm plus the inactive secondaries X5+/M2+/R5+. The direction and
isotropy-subgroup columns are computed with the same induced-irrep machinery
as `--parent` (section 17).

**Output files.** Every run writes the transformation block and the
decomposition table to `sym_mode_<formula>` (named by the parent
composition) and one VESTA file
per activated irrep, `<parent file>_<irrep>.vesta` (`_conv.vesta` with
`--conventional`), into the current directory, and prints the path of each
file. `--output-dir DIR` writes them into `DIR` instead (created if missing;
the printed paths then start with `DIR`), and `--no-files` writes nothing
and prints the analysis only, which suits batch runs over many structure
pairs:

```bash
crystod-group --supergroup-cif parent.cif --subgroup-cif child.cif --output-dir modes/child
crystod-group --supergroup-cif parent.cif --subgroup-cif child.cif --no-files
```

The two options exclude each other, and both are refused outside
`--supergroup-cif`. In Python, `crystod.group.SymmetryModeAnalysis` returns
the same decomposition and writes no files.

**Which of the equivalent settings is reported?** A symmetric parent leaves
the sublattice basis degenerate: every basis S·W (W a point operation of the
parent) pairs the atoms equally well and merely presents the *same*
distortion in a rotated setting — 24 of them for a cubic parent. CrystOD
picks the one whose child axes are rotated least against the parent axes,
both lattices taken as the input files orient them (a CIF is placed with *c*
along *z* and *a* in the *xz* plane; a POSCAR keeps its lattice vectors as
written), so the displacement table and the per-irrep VESTA files follow the
axes of the subgroup file: PbTiO3 P4mm polarized along *c* is shown
polarized along *c* of the cubic cell, not along *b*, and a POSCAR whose
polar axis lies along Cartesian *y* comes out along *b*. When no setting
aligns the axes — standard settings that permute them, a √2×√2 cell, a
rhombohedral child in its hexagonal setting — the residual rotation is
printed with the cell relation, as in the La3Ni2O7 example above (Cmcm puts
the long axis first: 90° plus the 45° in-plane rotation of the √2 cell).
Irrep, isotropy subgroup, number of modes and amplitude do not depend on the
setting, and neither does the direction column (next paragraph).

**Which order-parameter direction is printed?** The irreps are labelled with
the ISO-IR tables of ISOTROPY, and the direction column follows ISOTROPY as
well: it names each order parameter by the ISOTROPY representative of its
stratum, the row that `crystod-group --parent SG --irrep IR` (section 17)
and ISOSUBGROUP list for that isotropy subgroup (for an irrep at a
non-special k point, which these tables do not list, the simplest direction
of the stratum), whichever of the domain-equivalent orientations the child
file describes. When the child as
given sits on another direction of the same stratum (a parent operation
carries one onto the other), a `(direction: ...)` note under the table names
it; the note is saved in `sym_mode_<formula>` with the table. The SrTiO3
child above sits on `R5-(a,0,0)`, and the table prints `R5-(0,0,a)`, the
I4/mcm row of `--parent Pm-3m --irrep R5-`. V2O3 R-3c -> C2/c is the same
case for a two-dimensional irrep (its structures are in `debug9_V2O3/` of
the example directory, which is not in the pip package):

```bash
crystod-group --supergroup-cif debug9_V2O3/POSCAR_V2O3_R-3c.cif --subgroup-cif debug9_V2O3/POSCAR_V2O3_C2c.cif --conventional
```

```
* Transformation matrix (parent -> child) *
R-3c (No. 167) -> C2/c (No. 15), conventional settings of the input files
  [     1     1  -1/3 ] [   1/3 ]
  [    -1     1   1/3 ] [   1/6 ]
  [     0     0   1/3 ] [   1/6 ]
(columns of P: the child basis vectors a', b', c' in parent units, (a', b', c') = (a, b, c) P;
 last column: the child origin p in parent fractional coordinates, x_parent = P x_child + p;
 the AMPLIMODES / Bilbao convention)

* Symmetry-mode decomposition *
k-vector         irrep   direction    isotropy subgroup   dim  amplitude (A)
(0,0,0)          GM1+    (a)          167 R-3c            2    0.0425
(0,0,0)          GM3+    (a,0)        15 C2/c             5    0.1222
(direction: GM3+ sits on its ISOTROPY representative as listed by
 crystod-group --parent 167 --irrep GM3+; the input child as given sits on the
 domain-equivalent direction GM3+ (0.577a,-a))
```

`GM3+(a,0)` is the C2/c row of `crystod-group --parent R-3c --irrep GM3+`;
the child file is the monoclinic domain `(0.577a,-a)` of the same stratum.
Only the label is moved to the representative: the cell relation, the
transformation matrix (P, p) above, the displacement table, the normalized
mode components and the VESTA files keep describing the child as given, in
the setting closest to the input files, and the number of modes and the
amplitude are the same for every domain. When the child already sits on the
representative, as the La3Ni2O7 child above does, no note is printed.

With several irreps the rows have to describe one domain together, so one
parent operation moves all of them: the largest mode that has
domain-equivalent directions is put on its `--parent` row, and each smaller
mode, in order of decreasing amplitude, on the simplest direction left to it
in that domain. A secondary mode can therefore read differently from its own
`--parent` row; the note then names the deciding mode and the
representatives of the other strata on their own. The Pbnm perovskite
(`POSCAR_ScF3_Pbnm` and the CaTiO3 pair of `example/23_symmetry_mode`)
prints `R4+(0,a,a)`, the Imma row of `--parent Pm-3m --irrep R4+`, with
`M3+(a;0;0)`, as in the coupled row `R4+(0,a,a) M3+(d;0;0)` (Pnma) of
`--parent Pm-3m --irrep R4+ M3+`, although M3+ on its own is tabulated as
`M3+(0;0;a)`. The isotropy-subgroup column is that of the direction the
child as given sits on in that irrep (74 Imma for R4+ of a Pnma child),
conjugate to the subgroup of the direction shown: the same type and index,
except for a child on the mirror-image domain of a stratum of an
enantiomorphic type (P4_32_12 for a row tabulated as P4_12_12), whose
partner type the note names. In Python, each entry of
`SymmetryModeAnalysis(...).modes` returns the printed direction, subgroup
and index from `label_info()`; `domain_label` is the direction of the child
as given and `representative_label` the representative of its stratum on
its own.

### Order parameters at non-special k points

A few materials condense modes at k points that are **not special points**
of the parent — points on symmetry lines with a fractional free parameter.
The classic case is the PbZrO3 antiferroelectric, whose Pbam ground state
(8x the cubic primitive cell) involves the Sigma point (1/4,1/4,0) and the
S point (1/4,1/2,1/4). Its two structures are in `debug2_PbZrO3/` of the
example directory, which is not in the pip package (it stays in the CrystOD
development tree):

```bash
crystod-group --supergroup-cif debug2_PbZrO3/POSCAR_PbZrO3_Pm-3m.cif --subgroup-cif debug2_PbZrO3/POSCAR_PbZrO3_Pbam.cif
```

```
* Transformation matrix (parent -> child) *
Pm-3m (No. 221) -> Pbam (No. 55), conventional settings of the input files
  [     1    -2     0 ] [     0 ]
  [     1     2     0 ] [   1/2 ]
  [     0     0     2 ] [   1/2 ]
(columns of P: the child basis vectors a', b', c' in parent units, (a', b', c') = (a, b, c) P;
 last column: the child origin p in parent fractional coordinates, x_parent = P x_child + p;
 the AMPLIMODES / Bilbao convention)

* Symmetry-mode decomposition *
k-vector         irrep   direction                      isotropy subgroup   dim  amplitude (A)
(0,1/2,0)        X3-     (0;a;0)                        123 P4/mmm          2    0.0374
(1/4,1/4,0)      SM2     (0;0;a;0.332a;0;0;0;0;0;0;0;0) 55 Pbam             5    1.2871
(1/2,1/2,0)      M5-     (a,-a;0,0;0,0)                 51 Pmma             3    0.0473
(1/4,1/2,1/4)    S4      (0;0;0;0;0;0;0;0;0;0;a;0.332a) 64 Cmce             3    0.4470
(1/2,1/2,1/2)    R4+     (0,a,a)                        74 Imma             1    1.5232
(1/2,1/2,1/2)    R5+     (0,a,-a)                       74 Imma             2    0.0810
(non-special k points: the order-parameter components are expressed in the bundled
 ISO-IR matrix basis, whose phase gauge may differ from the ISOTROPY web tables;
 the isotropy subgroup, dimension and amplitude are gauge-independent)
(direction: the rows describe one domain of the child: the largest mode, R4+,
 sits on its ISOTROPY representative as listed by
 crystod-group --parent 221 --irrep R4+, and the smaller modes on the simplest
 directions left in that domain; listed on their own, the other strata read
 X3- (0;0;a), SM2 (0;0;0;0;0;0;0;0;0;0;a;0.332a), M5- (0,0;0,0;a,a) and
 R5+ (0,a,a))
```

Every number agrees with the Bilbao AMPLIMODES reference — the dominant
antipolar Pb mode `SM2` (1.2871 Å) and the octahedral tilt `R4+`
(1.5232 Å), with the secondary `S4`, `R5+`, `X3-` and `M5-` — including
each mode's isotropy subgroup. The command does not re-measure the
subgroups: testsuite section 23 freezes each single-mode displacement field
of this example into the parent structure and checks with spglib that it
measures as the isotropy subgroup in the table. The `(direction: ...)` note
is the one of *Which order-parameter direction is printed?* above: R4+, the
largest mode, sits on its `--parent Pm-3m --irrep R4+` row and fixes the
domain in which the other rows are written. Small irreps at such points are computed with spgrep at the exact
k and named/represented through the bundled ISO-IR tables (SM2, S4, ...);
since the ISOTROPY *web* tables fix their order-parameter axes with a phase
gauge that is not part of the bundled data, the direction *pattern* of a
line mode may differ from ISOSUBGROUP's while the subgroup, dimension and
amplitude — gauge-independent quantities — always agree.

A second line-point case, the K2SeO4 lock-in ferroelectric
(Pnma -> Pna2₁ at 3x the cell along *a*, SM = (1/3,0,0)):

```bash
crystod-group --supergroup-cif POSCAR_K2SeO4_Pnma.cif --subgroup-cif POSCAR_K2SeO4_Pna21.cif
```

```
* Transformation matrix (parent -> child) *
Pnma (No. 62) -> Pna21 (No. 33), conventional settings of the input files
  [     3     0     0 ] [   1/2 ]
  [     0     0    -1 ] [ 0.9986 ]
  [     0     1     0 ] [   1/2 ]
(columns of P: the child basis vectors a', b', c' in parent units, (a', b', c') = (a, b, c) P;
 last column: the child origin p in parent fractional coordinates, x_parent = P x_child + p;
 the AMPLIMODES / Bilbao convention)

* Symmetry-mode decomposition *
k-vector         irrep   direction    isotropy subgroup   dim  amplitude (A)
(0,0,0)          GM1+    (a)          62 Pnma             13   0.9467
(0,0,0)          GM4-    (a)          33 Pna2_1           8    0.4230
(1/3,0,0)        SM2     (a;0.974a)   33 Pna2_1           16   1.2885
(1/3,0,0)        SM3     (a;0.974a)   62 Pnma             26   0.1727
(non-special k points: the order-parameter components are expressed in the bundled
 ISO-IR matrix basis, whose phase gauge may differ from the ISOTROPY web tables;
 the isotropy subgroup, dimension and amplitude are gauge-independent)
...

* Output files *
  Decomposition table saved to sym_mode_K2SeO4
  ...
```

again matching AMPLIMODES entry by entry (including the polar `GM4-` that
carries the spontaneous polarization of the lock-in phase). The
transformation block and the decomposition table of every run are also
**saved as a text file**, `sym_mode_{formula}`, named by the parent composition in the conventional
chemical order of the `_chemical_formula_sum` rule of section 22
(`sym_mode_K2SeO4`, `sym_mode_SrTiO3`, `sym_mode_La3Ni2O7`, ...), so a
batch of analyses leaves one table per compound. Two parents of the same
composition therefore write the same file, and the second run replaces the
table of the first; `--output-dir DIR` keeps them apart. The README of
`example/23_symmetry_mode` does this for its second La3Ni2O7 pair, the
I4/mmm -> Cmcm distortion above with the structures in another description
(`--supergroup-cif POSCAR_La3Ni2O7_I4mmm.cif --subgroup-cif
POSCAR_La3Ni2O7_Cmcm.cif --output-dir La3Ni2O7_I4mmm_Cmcm`: GM1+ 0.1313,
X3- 1.1414 Å, total 1.1489 Å), so that the `sym_mode_La3Ni2O7` of the run
above stays as it is.

### Comparing with the Bilbao and ISOTROPY web tools

The physically meaningful columns — which irreps are active, their k-vectors,
isotropy subgroups, dimensions and amplitudes — reproduce AMPLIMODES and
ISODISTORT case for case. Four things can legitimately read differently, and
none of them is a disagreement about the distortion:

- **The origin of a polar subgroup is free**, so the amplitude of a polar
  irrep depends on where it is pinned. CrystOD places it at the minimum of
  the total distortion (the AMPLIMODES convention) and prints a `note: the
  subgroup is polar` line when it does. ISODISTORT pins the first orbit
  instead, which typically leaves its polar amplitude larger by exactly the
  removed rigid translation: the two differ by that translation and nothing
  else, so every other irrep is untouched.
- **Parity superscripts of a k ≠ 0 irrep depend on the parent's origin.**
  Two descriptions of the same crystal related by a translation that is in
  the Euclidean normalizer but not in the space group (rutile VO₂ with the
  metal at 2a rather than 2b) give the same subgroup and the same amplitude,
  but swap `R1+` and `R1-`.
- **The k-vector is printed in the parent's primitive basis**, while Bilbao
  quotes the conventional one. For a C-centred monoclinic parent the M point
  reads `(1/2,1/2,1/2)` here and `(0,1,1/2)` there — the same point.
- **Order-parameter directions are those of ISOTROPY.** The direction
  column is written in the basis of the bundled ISO-IR tables and names each
  stratum by its ISOTROPY representative, the row of `crystod-group --parent`
  and of ISOSUBGROUP, whatever domain the child file describes (the note
  under the table names that domain). AMPLIMODES gives the direction of the
  domain its transformation matrix describes, in the basis of the Bilbao
  representation matrices, so the letter pattern can differ (`(a,-a,-a)`
  against `(a,a,a)`) while naming the same stratum: for V2O3 R-3c -> C2/c
  AMPLIMODES prints `GM3+ (-0.500a,0.866a)`, where CrystOD prints
  `GM3+ (a,0)` and names the input domain `(0.577a,-a)` in the note. The
  isotropy subgroup column, which is basis-independent, is the one to
  compare.

Two differences are *not* conventions and mean the input needs attention. If
the reported distortion is far larger than expected, check that the reference
cell the other program used really is the strain-free parent supercell — a
transformation matrix that misplaces it by a fraction of a cell dumps the
whole rigid offset into the fully symmetric mode. And if the analysis reports
no distortion at all, the child was probably symmetrized away during
`--poscar2cif`: displacements below the tolerance are averaged out (the
conversion warns about this), so pass a smaller `--tolerance`.

A symmetry lowering can also be **purely a spontaneous strain**: in
La₃Ni₂O₇ I4/mmm → Fmmm the atoms keep every parent operation exactly and only
the orthorhombic metric breaks the four-fold axis. The displacive
decomposition then holds a single fully symmetric mode, and the report says
which part of the symmetry lowering it is not carrying.

### Complex-type irreps: one line per conjugate pair

At zone-boundary points of non-symmorphic groups, and at Γ of groups whose
point group has complex irreps (3, -3, 4, -4, 4/m, 6, -6, 6/m, 23, m-3), an
irrep can be of **complex type**: its characters are not real, and its
complex conjugate is another tabulated irrep. A real displacement field
always carries the two together, with conjugate coefficients, so they form
one physically irreducible order parameter. The decomposition lists such a
pair on **one line**, labelled by both names as ISODISTORT does (`GM2+GM3+`,
`R1+R3+`, `B1BA1` when the conjugate lives on the -k star); its amplitude
is that of the projection onto the pair, and `dim` counts the independent
modes of the pair. A **pseudoreal** irrep (Frobenius-Schur indicator -1) is
already the real form of itself and keeps its single name. Pyrite-type FeS2
(Pa-3) with small random displacements in the fcc-type double cell, whose
folding stars Γ and R carry both kinds (`example/23_symmetry_mode`):

```bash
crystod-group --supergroup-cif POSCAR_FeS2_Pa-3 --subgroup-cif POSCAR_FeS2_P1_fcc2
```

```
* Transformation matrix (parent -> child) *
Pa-3 (No. 205) -> P1 (No. 1), conventional settings of the input files
  [     0     1     1 ] [     0 ]
  [     1     0     1 ] [     0 ]
  [     1     1     0 ] [     0 ]
(columns of P: the child basis vectors a', b', c' in parent units, (a', b', c') = (a, b, c) P;
 last column: the child origin p in parent fractional coordinates, x_parent = P x_child + p;
 the AMPLIMODES / Bilbao convention)

* Symmetry-mode decomposition *
k-vector         irrep    direction    isotropy subgroup   dim  amplitude (A)
(0,0,0)          GM1+     (a)          205 Pa-3            1    0.0163
(0,0,0)          GM2+GM3+ (a,b)        61 Pbca             2    0.0245
(0,0,0)          GM4+     (a,b,c)      2 P-1               9    0.1407
(0,0,0)          GM1-     (a)          198 P2_13           2    0.0277
(0,0,0)          GM2-GM3- (a,b)        19 P2_12_12_1       4    0.0308
(0,0,0)          GM4-     (a,b,c)      1 P1                18   0.1011
(1/2,1/2,1/2)    R1+R3+   (a,b,c,d)    2 P-1               8    0.0881
(1/2,1/2,1/2)    R2+      (a,b,c,d)    2 P-1               4    0.0992
(1/2,1/2,1/2)    R1-R3-   (a,b,c,d)    2 P-1               16   0.1059
(1/2,1/2,1/2)    R2-      (a,b,c,d)    2 P-1               8    0.0554
```

The squared amplitudes add up to the squared total distortion (0.2540 Å),
and freezing a pair's projected field into the parent gives exactly the
isotropy subgroup of its line. A folding star that carries a complex-type
irrep is analysed as one pair line, whether the irrep is active or not.
When the conjugate lives on the -k star (a parent with no operation taking
k to -k), the line carries the k vector of the first of the two stars and
its amplitude covers both; in Python the mode entry names the -k star in
`partner_kname` and `partner_kvec` (None for every other entry), and
`crystod-xrd --parent` assigns the reflections of both stars to the pair.

### When the mapping is redone in a group-subgroup setting (the NOTE line)

The cell relation is first chosen by the least total distortion. For a
pseudo-symmetric parent that minimum can be a **twin setting**: a slightly
strained sublattice or a shifted origin pairs the atoms marginally better,
although some operation of the child is not a parent operation in it.
Every operation of the child must then leave the mapped structure invariant
exactly (the child is idealized by spglib), and the displacement field must
be invariant under them. If the least-distortion mapping fails either test,
it is redone: only sublattice settings that carry every child operation onto
a parent operation are considered, the origin is solved from the
translation parts of the child operations (free only along polar
directions, where the minimum-distortion origin is kept), the atoms are
paired so that the field respects the child symmetry, and the least total
distortion among these settings wins. The cell relation then carries a NOTE:

```bash
crystod-group --supergroup-cif POSCAR_KGeCl3_P4mmm_pseudo --subgroup-cif POSCAR_KGeCl3_Cm
```

```
* Cell relation *
child primitive basis in parent primitive units (rows):
  (1, 0, 0)
  (0, 1, 0)
  (0, 0, 1)
origin shift (parent primitive fractional): (1/2, 1/2, 1)
primitive cell multiplication: 1
NOTE: the minimum-distortion atom mapping is not a group-subgroup setting (only
1 of the 2 operations of the child is a parent operation in it; the next one is
broken by 1.07e-01 A); the atoms were re-paired in a setting in which every
operation of the child space group Cm is a parent operation (maximum
displacement 0.1876 A -> 0.1876 A).
...
* Transformation matrix (parent -> child) *
P4/mmm (No. 123) -> Cm (No. 8), conventional settings of the input files
  [     1    -1     0 ] [ 0.5015 ]
  [     1     1     0 ] [ 0.5015 ]
  [     0     0     1 ] [ 0.9873 ]
(columns of P: the child basis vectors a', b', c' in parent units, (a', b', c') = (a, b, c) P;
 last column: the child origin p in parent fractional coordinates, x_parent = P x_child + p;
 the AMPLIMODES / Bilbao convention)

* Symmetry-mode decomposition *
k-vector         irrep   direction    isotropy subgroup   dim  amplitude (A)
(0,0,0)          GM3-    (a)          99 P4mm             4    0.1702
(0,0,0)          GM5-    (a,a)        38 Amm2             5    0.2482
```

The NOTE needs no action: it says that the table belongs to the redone
mapping, and gives the largest atomic displacement before and after. In a
twin setting the two are often equal (here the twin and the group-subgroup
setting pair the atoms equally well); a larger value after the redo means
the least-distortion pairing had moved some atom to a site the child
symmetry does not allow. Cm is polar, so symmetry leaves the origin free
within the mirror plane; it is placed at the minimum of the total
distortion, (0.5015, 0.5015, 0.9873) here, and each component is printed as
the nearest fraction with a denominator of at most 24, hence `1` for
0.9873. Analyses whose least-distortion mapping already is a group-subgroup
setting are not affected: on 2938 structure pairs of a materials survey and
on the earlier inputs of testsuite section 23 the cell relation, the
displacements, the mode table and the total are the same as without this
step.

### Error messages of the mapping stage

When no group-subgroup setting can be found, the run stops with one of four
messages, each followed by the reason the least-distortion mapping was
rejected:

- **`the child space group ... is not a subgroup of the parent ... in any of
  the N sublattice setting(s)`**: no sublattice of the right multiplicity
  carries the point group of the child onto parent operations. The two
  files are different structure types, the cell multiplication is not the
  one expected, or spglib reads the parent in a higher group than intended
  at this `--tolerance` (the message prints the groups spglib found).
- **`... is not a subgroup of the parent ...: its point group embeds in ...,
  but in none of them does an origin shift turn the translation parts ...`**:
  the rotations fit, but the screw axes, glide planes or centring of the
  child are not those of any parent operation, so the parent is not a
  supergroup of the child (a Pbam pseudo-parent of a Pna2₁ child is a
  typical case).
- **`the child is not a displacive distortion of this parent ... the Wyckoff
  splitting disagrees`**: a parent operation kept by the child fixes a
  different number of sites of one element in the parent-derived reference
  than in the child (for example 8 O sites against 4 O atoms), so no atom
  pairing can respect the child symmetry at any distance.
- **`no atom pairing consistent with it keeps every atom within 1.8 A`**: a
  group-subgroup setting exists, but in it some atom would have to move more
  than 1.8 Å — a different polymorph or stacking sequence (CdI2 polytypes),
  a large rigid shift, or a wrong parent.

Before the mapping, a file with a partially occupied or mixed site (a
disordered CIF) is refused with `ERROR: FILE has N partially occupied or
mixed site(s) (first: ...)`: the analysis pairs one atom per site and does
not choose a species for you; give an ordered model of the structure.

`--tolerance` (the spglib symmetry tolerance in Å, default 0.01 for
`--supergroup-cif`) enters these checks only through the space groups spglib
identifies in the two files; it is not a threshold of the mapping. A
mapping is accepted when the child operations leave the mapped structure
and the displacement field invariant to machine precision (10⁻⁶ Å): spglib
idealizes the child, so a symmetric pairing always passes, and a pairing
that exchanges atoms breaks the symmetry by an interatomic distance, however
weak the distortion. Two further messages, `not fully captured by the listed
modes` and `the group-subgroup consistent atom mapping leaves a symmetry
violation`, are internal consistency checks that the mapping stage rules
out; they ask for the case to be reported.

```{seealso}
**Theory:** [Symmetry-mode analysis internals](theory-isotropy-subgroups.md) — the lattice and origin handling, the completeness checks, the validation against Bilbao AMPLIMODES, and the citation to give if you use this feature.
```
