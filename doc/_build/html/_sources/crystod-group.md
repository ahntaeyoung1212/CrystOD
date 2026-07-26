# crystod-group

The point/space-group representation-theory calculator. One mode flag per task:
`--product`, `--table`, `--decompose`, `--ligand-field`, `--basis`,
`--generate-basis`, `--coset`, `--supergroup`, `--multiplet`, `--poscar2cif`,
`--cif2poscar`, `--supergroup-cif`.
Point groups are selected with `--pg`/`--pointgroup`/`--point-group` and
space groups with `--sg`/`--spacegroup`/`--space-group`, by symbol or by
number (labels starting with `-`, such as `-43m`, are accepted).

## 7. Direct products of point-group and space-group irreps (`--product`)

*Example directory: `example/07_direct_product` (testsuite section 7)*

### 7.1 Point-group irreps (`--pg`/`--point-group`)

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

(`--product T2g T2g` alone gives `1(A1g) + 1(Eg) + 1(T1g) + 1(T2g)` — the
symmetric/antisymmetric split of this square is what `--multiplet` performs in
section 14.)

### 7.2 Space-group irreps (`--sg`/`--space-group`)

Decompose the direct product of **full space-group irreps** (the irreps at
high-symmetry k points, induced over their whole star) into full space-group
irreps with CDML labels:

```bash
crystod-group --product R4- R5+ --sg Pm-3m
```

```
* Direct product (full space-group irreps) *
R4- x R5+ = GM2- + GM3- + GM4- + GM5-

* Dimension check (star size x small dim) *
3 x 3 = 9 -> 1 + 2 + 3 + 3 = 9
```

The k points of the factors may differ, three or more factors are accepted,
and the space group may be given by symbol or number. The wavevector
selection rule k1 + k2 = k3 (mod reciprocal lattice) over the star arms
determines which stars appear; reduction coefficients are computed by exact
character algebra over the finite factor group. Products landing on symmetry
lines outside the tabulated special points (DT, SM, V, T, S, ...) are
decomposed with on-the-fly `spgrep` small irreps; missing -k stars of polar
groups (CDML "A" points, e.g. PA of I-43m) are synthesized as conjugate
irreps. Line terms are named from a DIRPRO-fitted CDML table where
available, and otherwise from the ISO-IR (ISOTROPY, Miller-Love) tables —
marked `[non-tabulated; ISO-IR labels]` in the report, since the two
conventions can differ at lines (e.g. CDML V of I4/mmm = ISOTROPY LD).
Distinct +k/-k line stars of acentric groups take the CDML "A" suffix
(`P1 x X1 = LD1 + LD2 + LDA1 + LDA2` in I4).

This is the offline counterpart of the **DIRPRO** program of the Bilbao
Crystallographic Server, cross-validated against it line by line (9007
products, 20 space groups covering all Bravais classes; see
`example/07_direct_product/README`). If you use this feature in a
publication, please cite M. I. Aroyo, A. Kirov, C. Capillas, J. M. Perez-Mato
and H. Wondratschek, *Acta Cryst.* **A62**, 115-128 (2006).

### 7.3 Character tables for point groups (`--table --point-group`)

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

### 7.4 Character tables for space groups (`--table --space-group`)

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
               irrep  1  2_001  2_010  2_100  4^-_100  2_011  2_01-1  4^+_100  -1  m_001  m_010  m_100  -4^-_100  m_011  m_01-1  -4^+_100
 irrep_1(1) = M1+(1)  1      1      1      1        1      1       1        1   1      1      1      1         1      1       1         1
 irrep_2(1) = M1-(1)  1      1      1      1        1      1       1        1  -1     -1     -1     -1        -1     -1      -1        -1
 irrep_3(1) = M3+(1)  1     -1     -1      1        1     -1      -1        1   1     -1     -1      1         1     -1      -1         1
 irrep_4(1) = M3-(1)  1     -1     -1      1        1     -1      -1        1  -1      1      1     -1        -1      1       1        -1
 irrep_5(2) = M5+(2)  2      0      0     -2        0      0       0        0   2      0      0     -2         0      0       0         0
 irrep_6(2) = M5-(2)  2      0      0     -2        0      0       0        0  -2      0      0      2         0      0       0         0
 irrep_7(1) = M2+(1)  1      1      1      1       -1     -1      -1       -1   1      1      1      1        -1     -1      -1        -1
 irrep_8(1) = M2-(1)  1      1      1      1       -1     -1      -1       -1  -1     -1     -1     -1         1      1       1         1
 irrep_9(1) = M4+(1)  1     -1     -1      1       -1      1       1       -1   1     -1     -1      1        -1      1       1        -1
irrep_10(1) = M4-(1)  1     -1     -1      1       -1      1       1       -1  -1      1      1     -1         1     -1      -1         1
```

Note that `(0, 1/2, 1/2)` is a non-representative arm of the M star: the
header names it `M` and the rows carry the CDML labels transported from the
tabulated arm (`irrep_N` is the internal `spgrep` name, `MN+/-` the physical
label). Unlike a point-group table the columns are individual symmetry
operations in Seitz notation, not classes, because at a general k point the
Bloch phases of a class need not coincide. These are exactly the characters
against which the SALC, phonon, and spin analyses are reduced.

## 8. Reducible-representation decomposition (`--decompose`)

*Example directory: `example/08_decompose_irrep` (testsuite section 8)*

Decompose a reducible representation into the irreps of a point group by entering
its characters class by class (educational / hand-analysis companion to `--product`):

```bash
crystod-group --decompose --point-group 3m
```

```
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
Based on `script/decomose_to_irreps.py` by Hiroki Koiso.

## 9. Ligand-field splitting (`--ligand-field`)

*Example directory: `example/09_ligand_field_split` (testsuite section 9)*

Decompose an atomic orbital (s, p, d, f, g, h, i) into the irreps of a point
group — the crystal-field / ligand-field splitting of the orbital in the given
point-symmetric environment:

```bash
crystod-group --ligand-field d --point-group m-3m
# * Result *  1(Eg) + 1(T2g)

crystod-group --ligand-field f --point-group 4/mmm
# * Result *  1(A2u) + 1(B1u) + 1(B2u) + 2(Eu)
```

The characters of the (2l+1)-dimensional orbital representation are generated
from the standard angular-momentum formulas
(chi(C(a)) = sin((l+1/2)a)/sin(a/2), chi(S(a)) = cos((l+1/2)a)/cos(a/2)) and
decomposed with the same reduction engine as `--decompose`.
Based on `script/ligand_field_spliting.py` by Hiroki Koiso.

## 10. Basis functions (`--basis`)

*Example directory: `example/10_basis_function` (testsuite section 10)*

```bash
crystod-group --basis x y z --point-group m-3m
crystod-group --basis x y z --space-group Pm-3m --kpoint 0 0 0
crystod-group --basis xyz --space-group Pm-3m --kpoint 0.5 0.3 0 --show-irrep-table
crystod-group --basis "x(y^2-z^2)" --point-group="m-3m"
crystod-group --basis "x^2-y^2" "2z^2-x^2-y^2" xy yz zx --space-group="Pm-3m" --kpoint 0 0 0
```

The input functions are automatically closed under the selected point group or
the little group of the selected space-group k point, then decomposed into
irreps. When the k point is listed in `irreptables`, physical labels such as
`GM4-(3)` are shown; otherwise `spgrep` generic labels such as `irrep_2(1)` are used.

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
space group are analyzed automatically (as in the `crystod` SALC survey):

```bash
crystod-group --basis Rx Ry Rz --space-group Pm-3m
# -> GM4+(3), R4+(3), M3+(1) + M5+(2), X3+(1) + X5+(2)
```

## 11. Automatic generation of polynomial basis functions (`--generate-basis`)

*Example directory: `example/11_generate_basis_function` (testsuite section 11)*

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
of the little group of the selected space-group k point. The second-order run
at GM of Pm-3m, for example, ends with

```
* Decomposition *
 1.0 [GM1+(1)] + 1.0 [GM3+(2)] + 1.0 [GM5+(3)]

* Irreducible representations for basis functions *
  GM1+(1): [x^2 + y^2 + z^2]
  GM3+(2): [-2 x^2 + y^2 + z^2, x^2 - 2 y^2 + z^2]
  GM5+(3): [xy, yz, xz]
```

— the quadratic polynomials sorted into the breathing mode, the Eg pair, and
the T2g triple (see also the projection-operator walk-through in section 1.4 of
the `crystod` page).

## 12. Coset decomposition (`--coset`)

*Example directory: `example/12_show_coset` (testsuite section 12)*

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
the little co-group of a k point (one coset per arm of the star of k):

```bash
crystod-group --coset --space-group Pm-3m --kpoint 0.5 0.5 0
```

For the point-group mode, H must be expressed in the same axes convention as G;
a clear error message is printed otherwise. The index `[G:H]`
(or `[G:G_k] = |star of k|`) and the members of each coset are listed.

## 13. Isotropy subgroups: supergroup-subgroup relations (`--supergroup`)

*Example directory: `example/13_isotropy_subgroup` (testsuite section 13)*

When a distortion transforming as an irrep condenses, the symmetry drops from
the supergroup to the **isotropy subgroup** H(eta) = {g : D(g) eta = eta},
which depends on the order-parameter direction eta:

```bash
crystod-group --supergroup Pm-3m --irrep GM4- --order-parameter 0 0 a
# -> P4mm (No. 99), index 6, conventional basis and origin

crystod-group --supergroup Pm-3m --irrep R4+
# -> all direction types: I4/mcm, R-3c, Imma, C2/m, C2/c, P-1 (cell size 2)
```

Zone-boundary irreps carry their full star (order-parameter dimension =
arms x small dimension) and the cell enlargement is detected automatically —
`--supergroup Pm-3m --irrep R4+` reproduces the complete Howard-Stokes
octahedral-tilt classification of perovskites. Letters in `--order-parameter`
are free parameters; omit the option to enumerate every direction type.
Following ISOTROPY, the order-parameter components are grouped arm by arm:
`;` separates the star arms and `,` the components within one arm (R4+ of
Pm-3m: one arm x small dim 3 -> `(a,a,b)`; M3+: three arms x small dim 1 ->
`(a;b;c)`; X5+: three arms x small dim 2 -> `(a,b;0,0;0,0)`).

### Complex- and pseudoreal-type irreps (doubled real form)

When the Frobenius-Schur indicator of the induced irrep vanishes (complex
type) or is -1 (pseudoreal type) — as at zone-boundary points of
non-symmorphic space groups, where the translation phases are genuinely
complex — the real order parameter transforms as the **physically
irreducible doubled real form** (the realification of D + D*), the
dimension doubles, and the output carries the ISOTROPY-style pair label:

```bash
crystod-group --supergroup Ia-3d --irrep P2
```

```
* Irrep *
P1P2: order parameter dimension 8 (star of 2 arm(s) x small dim 2 x 2; complex-type irrep -> physically irreducible real form)

* Order parameter directions and isotropy subgroups *
irrep                   subgroup           size  index
P1P2(a,0,b,0;a,0,b,0)   23 I222            4     48
P1P2(0,a,0,b;0,a,0,b)   24 I2_12_12_1      4     48
P1P2(0,0,0,0;0,a,0,b)   82 I-4             4     48
P1P2(a,b,c,d;-d,a,b,c)  2 P-1              4     96
P1P2(0,a,0,b;0,c,0,d)   5 C2               4     96
P1P2(a,b,c,d;a,b,c,d)   5 C2               4     96
P1P2(a,b,c,d;e,f,g,h)   1 P1               4     192
```

+k/-k pairs whose -k star is tabulated separately pair across the stars
(I-42d `P1` -> `P1PA1`, P3 `H1` -> `H1HA1`); conjugate-gauge and
origin-choice tabulations are matched automatically, and real-type irreps
whose induced matrices are complex (P3 of Ia-3d) are realified exactly
through the antilinear real structure of the group-averaged intertwiner.

### Coupled order parameters (several irreps)

Giving `--irrep` **several labels** enumerates the isotropy subgroups of the
**coupled** order parameters — the stabilizers on the direct sum of the
irreps, i.e. the space groups reached when several distortions condense
simultaneously:

```bash
crystod-group --supergroup I4/mmm --irrep X3- X2+
```

```
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
resolves to Cmc2_1), and `--supergroup Pm-3m --irrep R4+ M3+` reproduces the
full Howard-Stokes table of *mixed* perovskite tilt systems (a-a-c+ =
`R4+(0,a,a) M3+(a;0;0)` -> Pnma, a+a+c- -> P4_2/nmc, a0b-c+ -> Cmcm, ...).

This is the offline counterpart of **ISOSUBGROUP** of the ISOTROPY Software
Suite (https://iso.byu.edu), and is validated against it exhaustively: a
sweep over the 910 downloaded ISOSUBGROUP tables in `SUBGROUP/` (space
groups 16-230, every parameter-free high-symmetry k point — 3535 irreps;
`script/validate_isosubgroup.py`) reproduces the complete (subgroup, size,
index) multiset of every strata table, 3533 of 3535 irreps agreeing (25 up
to the enantiomorphic partner — a representative choice within one stratum
orbit — and 44 up to a verified CDML-vs-ISOTROPY irrep-label swap, recorded
in `SUBGROUP/VALIDATION.md` and printed as a note under the output whenever
an affected irrep or an enantiomorphic subgroup appears); the only
exceptions are the W point of Ibca (spgrep cannot construct its pseudoreal
small irreps) and the rhombohedral L star of R-3m (not tabulated in
`irreptables`). If you use this feature, please cite: H. T. Stokes,
S. van Orden and B. J. Campbell, "Tool for Generating Isotropy Subgroups of
Crystallographic Space Groups", J. Appl. Cryst. 49, 1849-1853 (2016).

## 14. Multi-electron terms (`--multiplet`)

*Example directory: `example/14_multiplet` (testsuite section 14)*

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

— the Tanabe-Sugano strong-field table of (t2g)^3. The two-electron
integrals are built from exact Gaunt coefficients (F^0 = A + 7C/5,
F^2 = 49B + 7C, F^4 = 63C/5), the Coulomb Hamiltonian over the Slater
determinants of the configuration, and each term is isolated by S^2 and
point-group projectors; doubly-occurring terms mix (configuration
interaction) and their two energies are printed in closed form
(e.g. 3A - 3B + 3C +- 3sqrt(2)B in (t2g)^2(eg)^1). Every run is closed by
the trace identity sum (2S+1) dim E = tr(H_ee); the free-ion limit is
reproduced exactly ((T1u)^2 with --orbital p gives ^3P = F0 - 5F2,
^1D = F0 + F2, ^1S = F0 + 10F2).

f shells are fully supported (`--orbital f`; A2u + T1u + T2u shells in
m-3m), with energies in the reduced Slater-Condon parameters F0, F2, F4,
F6 — e.g. `--multiplet T1u3 --pg m-3m --orbital f` gives
^4A1u = 3F0 - (105/4)F2 - (189/2)F4 - (3705/4)F6 as the ground state (the
f analogue of (t2g)^3 -> ^4A2g); numeric CI blocks and the ground-state
selection use the hydrogenic 4f ratios F4/F2 = 0.138, F6/F2 = 0.0151.

Of the plain direct product T2g x T2g = A1g + Eg + T1g + T2g (section 7),
the Pauli principle pairs only the antisymmetric square (T1g) with the spin
triplet — `--multiplet` performs this antisymmetrization exactly, for any
filling of any shell (hole equivalence and closed shells come out
automatically: (t2g)^4 gives the (t2g)^2 terms, (t2g)^6 gives ^1A1g).
Several tokens denote inequivalent shells, coupled by spatial direct
products and spin angular-momentum addition; the optional
`--orbital s|p|d|f|...` prints the ligand-field splitting of the parent
atomic orbital (section 9) and verifies the occupied shells occur in it.
Every result closes with a state-count check (product of C(2 dim, n)).
Validated against the standard crystal-field term tables (t2g^n, eg^n,
t2g^2 eg^1 in Oh; e^2 in Td and C3v).

For two-shell configurations, the doubly-occurring terms (the CI pairs) are additionally printed as their **CI matrix in the coupled-parent basis** |shell1(S1 Gamma1) shell2(S2 Gamma2)> — the representation used by the Tanabe-Sugano/Griffith strong-field tables — e.g. for (t2g)^2(eg)^1: the ^2Eg block is <t2g^2(^1A1g)eg|H|...> = 3A + 8B + 6C, <t2g^2(^1Eg)eg|H|...> = 3A - B + 3C, off-diagonal ±10B, whose eigenvalues are exactly the printed 3A + (7/2)B + (9/2)C ± (1/2)√(481B² + 54BC + 9C²) (the off-diagonal sign is a basis convention; books may differ).

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

A term eigenstate is in general a superposition of determinants, not a single box configuration — e.g. the ^4A2g of (t2g)^3 *is* the single determinant |dxy↑ dyz↑ dxz↑⟩ (coefficient 1), while a ^4T1g partner of (t2g)^2(eg)^1 is √3/2 |dx2-y2↑; dxy↑ dyz↑⟩ + 1/2 |dz2↑; dxy↑ dyz↑⟩. States are shown at the highest spin projection Ms = S; degenerate spatial partners are switchable (canonicalized, so any orthogonal mixture is equivalent); configuration-mixed terms (the CI pairs, e.g. the two ^2T1g) get one tab per state with its Coulomb energy at the reference parameters. `--output` selects the file name. For f shells, symmetry-mixed basis functions (e.g. the t1u combination of fx(x2-3y2) and fxz2) get short symbols `t1u(1)`, ... in the boxes, expanded in an *Orbital basis functions* legend on the page. Each state also gets a drag-rotatable 3D surface of its **charge and spin density** (angular part), computed exactly from the one-particle reduced density matrix of the term eigenstate: r(θ,φ) ∝ n(θ,φ), colored by the local spin polarization — the real-space picture behind orbital ordering and Jahn-Teller physics (e.g. the (t2g)^3 ^4A2g shows the cubic-symmetric t2g flower, fully spin-polarized, while the ^2Eg partners keep the cubic charge density but carry an anisotropic spin density).

## 15. POSCAR <-> Bilbao-style CIF (`--poscar2cif` / `--cif2poscar`)

*Example directory: `example/15_poscar2cif` (testsuite section 15)*

Convert a POSCAR into a CIF laid out like the files of the Bilbao
Crystallographic Server, and back:

```bash
crystod-group --poscar2cif -c 221_PPOSCAR_ScF3 [--tolerance 0.01]
# -> 221_PPOSCAR_ScF3.cif

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

`--cif2poscar` accepts any CIF flavour (Bilbao or pymatgen), expands the
symmetry operations, and writes the spglib-standardized primitive cell —
the working format of the other crystod commands — as a POSCAR in the
crystod test-file style (6-decimal `direct` coordinates with element tags)
to the input path without `.cif`. Round trips reproduce the original
primitive structure exactly (SrTiO3, ScF3, F-centred NaCl: 2-atom
primitive by default, 8-atom conventional with `--conventional`).

## 16. Symmetry-mode analysis (`--supergroup-cif`)

*Example directory: `example/16_symmetry_mode` (testsuite section 16)*

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
```

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
  (-1, 0, 0)
  (0, -1, 0)
  (1, 1, 2)
origin shift (parent primitive fractional): (1/2, 1, 1/2)
primitive cell multiplication: 2

maximum atomic displacement: 0.4097 A
total distortion amplitude : 1.1489 A
(normalized within the primitive cell of the distorted structure)

* Symmetry-mode decomposition *
k-vector         irrep   direction    isotropy subgroup   dim  amplitude (A)
(0,0,0)          GM1+    (a)          139 I4/mmm          4    0.1313
(0,0,1/2)        X3-     (a;0)        63 Cmcm             6    1.1413

* Mode displacement VESTA files (parent conventional basis) *
display cell in parent primitive units (rows):
  (0, 2, 2)
  (2, 0, 2)
  (1, 1, 0)
  139_PPOSCAR_La3Ni2O7_GM1+_conv.vesta  (amplitude 0.1313 A)
  139_PPOSCAR_La3Ni2O7_X3-_conv.vesta  (amplitude 1.1413 A)
```

The distortion is dominated by the zone-boundary octahedral-tilt mode
`X3-(a;0)`, whose isotropy subgroup is exactly the observed Cmcm — the same
entry as in the single-irrep table of section 13 — while the totally
symmetric `GM1+` is only a small secondary relaxation of the free
coordinates within I4/mmm. `--conventional` writes the per-mode displacement
VESTA files in the **parent conventional basis** (the `_conv` suffix; the
body-centred I lattice makes the conventional cell twice the primitive one,
hence the printed display-cell rows), so the arrows can be inspected in the
familiar tetragonal setting instead of the primitive one.

The output also contains the automatically determined cell relation
(sublattice basis + origin shift), the atom-by-atom displacement table
(maximum displacement, total distortion), the number of independent modes
per irrep, and the normalized polarization vectors; inputs may be CIFs or
POSCARs. Amplitudes follow the AMPLIMODES normalization (within the
primitive cell of the distorted structure). Multi-irrep distortions
decompose completely — the Pbnm perovskite gives R4+ -> Imma and
M3+ -> P4/mbm plus the inactive secondaries X5+/M2+/R5+. The direction and
isotropy-subgroup columns are computed with the induced-irrep machinery of
section 13, non-invariant subgroup lattices are enlarged to the largest
parent-invariant sublattice (complete k stars, exact amplitude rescaling),
polar subgroups get the minimum-distortion origin (acoustic component
removed), the lattice matching tolerates strong relaxation (principal
strains up to 20%), and a projector-completeness check closes every run.
Validated against the Bilbao AMPLIMODES output (SrTiO3 Pm-3m -> I4/mcm:
R5- 0.3303 A; F-centred ZrO2 Fm-3m -> P4_2/nmc: X2- 0.5773 A), the
ferroelectric BaTiO3 -> P4mm and large-tilt AlF3 -> R-3c cases, and the
section-25 modulation structures. If you use this feature, please cite:
D. Orobengoa, C. Capillas, M. I. Aroyo and J. M. Perez-Mato, "AMPLIMODES:
symmetry-mode analysis on the Bilbao Crystallographic Server",
J. Appl. Cryst. 42, 820-833 (2009).
