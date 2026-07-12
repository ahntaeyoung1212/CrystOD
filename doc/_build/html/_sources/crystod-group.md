# crystod-group

The point/space-group representation-theory calculator. One mode flag per task:
`--product`, `--table`, `--decompose`, `--ligand-field`, `--basis`,
`--generate-basis`, `--coset`. Point groups are selected with `--pg`/`--point-group`
and space groups with `--sg`/`--space-group` (labels starting with `-`, such as
`-43m`, are accepted).

## 7. Direct products of point-group irreps (`--product`)

*Example directory: `example/07_direct_product` (testsuite section 7)*

```bash
crystod-group --product T2g T2g T1u --point-group m-3m
```

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
