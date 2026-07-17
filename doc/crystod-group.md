# crystod-group

The point/space-group representation-theory calculator. One mode flag per task:
`--product`, `--table`, `--decompose`, `--ligand-field`, `--basis`,
`--generate-basis`, `--coset`, `--supergroup`, `--multiplet`, `--poscar2cif`,
`--cif2poscar`, `--supergroup-cif`.
Point groups are selected with `--pg`/`--point-group` and space groups with
`--sg`/`--space-group` (labels starting with `-`, such as `-43m`, are accepted).

## 7. Direct products of point-group and space-group irreps (`--product`)

*Example directory: `example/07_direct_product` (testsuite section 7)*

```bash
crystod-group --product T2g T2g T1u --point-group m-3m
```

### Space-group irreps (`--sg`/`--space-group`)

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
irreps.

This is the offline counterpart of the **DIRPRO** program of the Bilbao
Crystallographic Server, cross-validated against it line by line (9007
products, 20 space groups covering all Bravais classes; see
`example/07_direct_product/README`). If you use this feature in a
publication, please cite M. I. Aroyo, A. Kirov, C. Capillas, J. M. Perez-Mato
and H. Wondratschek, *Acta Cryst.* **A62**, 115-128 (2006).

Show the point-group character table:

```bash
crystod-group --table --point-group 3m
```

Show both the table and the direct-product decomposition:

```bash
crystod-group --product T2g T2g T1u --point-group m-3m --show-irrep-table
```

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
of the little group of the selected space-group k point.

## 12. Coset decomposition (`--coset`)

*Example directory: `example/12_show_coset` (testsuite section 12)*

Point-group mode decomposes G into left cosets g H of a subgroup H:

```bash
crystod-group --coset --point-group m-3m --subgroup 4/mmm
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

This is the offline counterpart of **ISOSUBGROUP** of the ISOTROPY Software
Suite (https://iso.byu.edu), and is validated against it (all Pm-3m GM
irreps, entry by entry). If you use this feature, please cite: H. T. Stokes,
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
