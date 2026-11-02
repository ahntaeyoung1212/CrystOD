# Order parameters and isotropy subgroups

Background for the symmetry-lowering tools of `crystod-group`: how a complex or
pseudoreal irrep is turned into the physically irreducible real form that an
order parameter actually lives in (`--parent`, section 17), how the
symmetry-mode decomposition of a distorted structure is constructed
(`--supergroup-cif`, section 23), and how both are validated against the
ISOTROPY and Bilbao reference implementations.

## Complex- and pseudoreal-type irreps (doubled real form)

When the Frobenius-Schur indicator of the induced irrep vanishes (complex
type) or is -1 (pseudoreal type) — as at zone-boundary points of
non-symmorphic space groups, where the translation phases are genuinely
complex — the real order parameter transforms as the **physically
irreducible doubled real form** (the realification of D + D*), the
dimension doubles, and the output of `crystod-group --parent` carries
the ISOTROPY-style pair label:

```bash
crystod-group --parent Ia-3d --irrep P2
```

```
* Irrep *
P1P2: order parameter dimension 8 (star of 2 arm(s) x small dim 2 x 2; complex-type irrep -> physically irreducible real form)

* Order parameter directions and isotropy subgroups *
irrep                                               subgroup           size  index
P1P2(a,a,b,b;-a,a,-b,b)                             23 I222            4     48
P1P2(a,-a,b,-b;a,a,b,b)                             24 I2_12_12_1      4     48
P1P2(0,0,0,0;0,a,0,b)                               82 I-4             4     48
P1P2(2a,2b,2c,2d;a+b-c+d,a+b+c-d,a-b+c+d,-a+b+c+d)  2 P-1              4     96
P1P2(0,a,0,b;c,d,d,-c)                              5 C2               4     96
P1P2(a,b,c,d;-a,b,-c,d)                             5 C2               4     96
P1P2(a,b,c,d;e,f,g,h)                               1 P1               4     192
```

The partner named in the pair label is the tabulated irrep whose real form
is the same representation: the one with the same real part of the character
on every group element, lattice translations included. The characters of the
coset representatives alone do not decide it — at H of P6_3/m, P of I4/mcm
or K of P3 several irreps share them and differ only on the translations.
+k/-k pairs whose -k star is tabulated separately pair across the stars
(I-42d `P1` -> `P1PA1`, P3 `H1` -> `H1HA1`); conjugate-gauge and
origin-choice tabulations are matched automatically, and real-type irreps
whose induced matrices are complex (P3 of Ia-3d) are realified exactly
through the antilinear real structure of the group-averaged intertwiner.

## Symmetry-mode analysis: algorithm internals and AMPLIMODES validation

In the symmetry-mode analysis of `crystod-group --supergroup-cif`
(section 23 of the `crystod-group` page), the direction and
isotropy-subgroup columns are computed with the same induced-irrep
machinery as `crystod-group --parent` (the isotropy-subgroup
construction described on this page),
non-invariant subgroup lattices are enlarged to the largest
parent-invariant sublattice (complete k stars, exact amplitude rescaling),
polar subgroups get the minimum-distortion origin (acoustic component
removed), the lattice matching tolerates strong relaxation (principal
strains up to 20%), and a projector-completeness check closes every run.
The direction column names each mode by the ISOTROPY representative of its
stratum (the row that `--parent` lists), with one parent operation applied
to all modes so that the rows describe one domain of the child; the
direction of the child as given is named in a note under the table and is
`domain_label` in Python.
Validated against the Bilbao AMPLIMODES output (SrTiO3 Pm-3m -> I4/mcm:
R5- 0.3303 A; F-centred ZrO2 Fm-3m -> P4_2/nmc: X2- 0.5773 A), the
ferroelectric BaTiO3 -> P4mm and large-tilt AlF3 -> R-3c cases, and the
modulation structures of `crystod-phonon --modulation` (section 33). If you
use this feature, please cite:
D. Orobengoa, C. Capillas, M. I. Aroyo and J. M. Perez-Mato, "AMPLIMODES:
symmetry-mode analysis on the Bilbao Crystallographic Server",
J. Appl. Cryst. 42, 820-833 (2009).

### The atom mapping and its group-subgroup setting

Both structures are standardized by spglib, and the parent is expressed in
the primitive setting of the ISO-IR tables. The child primitive basis is
written as an integer matrix S over the parent primitive basis (rows; det S
is the cell multiplication), and the child atoms are placed at
x = Sᵀ x_child + p for an origin shift p. The first mapping is the one of
least total distortion Σ|u|² over the candidate matrices S (principal
strains up to 20%), each with a greedy nearest-site pairing and a
continuous origin refinement; a tie between settings that a symmetric
parent makes equivalent goes to the one closest to the orientation of the
input files.

That minimum is not necessarily a **group-subgroup setting**. The child is
idealized by spglib, so in a correct (S, p) every child operation (R, τ)
becomes a parent operation (W, v) with W = Sᵀ R S⁻ᵀ and an exact
translation part, and the subgroup H is the set of parent operations that
leave the mapped structure invariant to machine precision (10⁻⁸ Å). A
strained twin setting or a shifted origin can pair the atoms marginally
better while some child rotation is not a parent rotation in it, or an
origin can match the rotations but not the screw and glide translations;
then fewer operations are exact than the child's own group has, and
thresholding the spectrum of near-misses accepts pseudo-operations. The
mapping is therefore validated before the decomposition: (a) at least as
many exact operations as the child group has, (b) every child operation,
carried over by S and p, among them, and (c) a displacement field invariant
under them to machine precision, |u(gx) − R_g u(x)| ≤ 10⁻⁶ Å (a pairing can
break the symmetry of an invariant structure by exchanging atoms, and then
does so by an interatomic distance). None of the three depends on
`--tolerance`, which only sets how spglib identifies the two space groups.

A mapping that fails either test is redone in a group-subgroup setting:

1. **Sublattice.** Only the bases S in which Sᵀ R S⁻ᵀ is the rotation of a
   parent operation for every child rotation R are kept.
2. **Origin.** The translation parts give a congruence system
   (I − W_k) p ≡ v_k − Sᵀ τ_k (mod ℤ³) over the child operations. Stacking
   the matrices I − W_k and reducing them to diagonal form (U A V = D, U
   and V unimodular) yields every solution class modulo ℤ³ and the free
   (polar) directions; a nonzero residual of the compatibility conditions
   means that the translation parts fit no origin, so the child group is
   not a subgroup of the parent in that setting. Along polar directions the
   origin is not fixed by symmetry: the first child atom of every species
   is put level with each reference atom of its species, through the
   lattice image whose non-polar remainder is shortest, and each start is
   refined twice by the polar part of the mean displacement (the
   AMPLIMODES minimum-distortion origin).
3. **Pairing.** At each origin the atoms are paired by the optimal
   assignment per species (minimum image, at most 1.8 Å per atom). It is
   accepted only if the field it gives is invariant under the conjugated
   child operations; otherwise the reference is paired orbit by orbit with
   child atoms of the same stabilizer, the rest of each orbit following by
   symmetry, and a child atom half a lattice vector from its site (whose
   displacement the site symmetry fixes only modulo the lattice) is
   rejected. A parent operation kept by the child that fixes a different
   number of sites of one species in the reference and in the child (a
   Wyckoff conflict) excludes every symmetric pairing at that origin.
4. **Choice and members.** Among the valid (S, p) the least Σ|u|² wins
   (ties: the orientation rule above). The subgroup members are the child
   operations themselves, plus any parent operation that leaves both the
   structure and the field exactly invariant (the strain-only part of a
   ferroelastic lowering), and the structure and the field must be
   invariant under all of them to machine precision before the
   decomposition starts.

When no valid (S, p) exists the run stops with the reason: no sublattice
carries the child point group, the translation parts are incompatible,
every setting has a Wyckoff conflict, or no symmetric pairing stays within
1.8 Å. On a survey of materials-database structure pairs this step resolves
25 of the 32 pairs that a projection without it reports as "the distortion
is not fully captured" (7 stop with a specific error), and 66 of the 106
that it reports as "cannot separate the surviving from the broken parent
operations" (40 stop with a specific error). For every solved pair the
reference plus the displacement field has exactly the child's space group
(spglib, 10⁻³ Å). Since the displacement field is invariant under H, the final completeness check (the listed modes sum to
the whole field) can only fail on an internal inconsistency.

### Complex-type irreps: the D + D* projector

The displacement representation is decomposed star by star. For an induced
irrep τ of dimension d_τ, the isotypic projector over the star block is
P_τ = (d_τ/|G|) Σ_g χ_τ(g)* D(g), with the characters split per star arm
so that lattice translations enter as phases. A real displacement field u
has conjugate Fourier components on the arms k and −k, and for a
**complex-type** τ (Frobenius-Schur indicator 0) the conjugate irrep τ* is a
different irrep: the physically irreducible real representation is
ρ = τ ⊕ τ*, of real dimension 2 d_τ, with the real projector

  P_ρ = P_τ + P_τ*,

reported as one mode whose amplitude is |P_ρ u| and whose number of
independent modes is tr(P_ρ P_H). On a star that contains its −k arms, P_τ*
is the complex conjugate of P_τ with the arms k and −k exchanged; on a star
without them, P_τ* lives on the −k star, the projected field is
2 Re(P_τ û) and the mode count 2 Re tr(P_τ P_H), and the −k partner is not
listed again. The partner is recognized from the conjugate per-arm
characters, independently of the labels; the line carries both labels
(`GM2+GM3+`). A **pseudoreal** τ (indicator −1) is equivalent to its
conjugate, P_τ is already real, and the single name stays; a real-type τ
keeps the real part (P_τ + P_τ*)/2 = P_τ as before. In general
P_ρ = dim ρ/(e |G|) Σ_g χ_ρ(g) D(g) with e = 1, 2, 4 for real, complex
and pseudoreal type. The squared amplitudes of all lines add up to the
squared total distortion.

## Validation of `crystod-group --parent` against ISOSUBGROUP

`crystod-group --parent` is the offline counterpart of **ISOSUBGROUP**
of the ISOTROPY Software Suite (https://iso.byu.edu), and is validated
against every reference table at hand: a
sweep over the 910 downloaded ISOSUBGROUP tables in `SUBGROUP/` (space
groups 62-230, every parameter-free high-symmetry k point — 3535 irreps;
`script/validate_isosubgroup.py 62 230`) reproduces the complete (subgroup,
size, index) multiset of every strata table under the same irrep label:
3506 irreps agree exactly and 28 up to the enantiomorphic partner (a
representative choice within one stratum orbit, printed as a note under the
output whenever an enantiomorphic subgroup appears); no table agrees only
under another label, so no label note is printed (results in
`SUBGROUP/VALIDATION.md`; the last full sweep ran on 2026-10-05 with v0.4.3
and took about 32 minutes). The one irrep of the 3535 not compared is the
six-dimensional L1 in the reference table of R-3m (No. 166) at L: the file
`SUBGROUP_SG166_L.txt` holds the L table of R-3c (No. 167), so it lists an
irrep that R-3m does not have (the L irreps of R-3m are L1+, L2+, L1- and
L2-, which `--parent 166 --kpoint L` enumerates). Four reference files hold
no result rows (SG 167 at F and GM, SG 207 at M and X), so those irreps are
not compared and are not counted in the 3535, and there are no reference tables for space groups 1-61.
If you use this feature,
please cite: H. T. Stokes,
S. van Orden and B. J. Campbell, "Tool for Generating Isotropy Subgroups of
Crystallographic Space Groups", J. Appl. Cryst. 49, 1849-1853 (2016).
