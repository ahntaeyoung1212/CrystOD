# crystod (main command)

The main command performs the SALC (symmetry-adapted linear combination)
analysis of crystal orbitals — no mode flag is needed. It also hosts the
interactive SALC viewer (`--visualize`) and the star-of-k display (`--star-of-k`).
Section 1 below summarizes the theoretical background on which every command
of this page (and of `crystod-group` / `crystod-mag`) is built.

## 1. Wigner D matrices and the projection operator (theoretical background)

*Example directory: `example/01_wigner_d` (testsuite section 1)*

```{note}
This section is theoretical background, **not a command-line mode**: there is
no CLI flag for it. Everything described here is what CrystOD evaluates
internally every time one of the commands is executed; the machinery is exposed
through the Python API only (`crystod.operations.wigner_D_real`). No prior
familiarity with group theory is assumed.
```

### 1.1 What is a representation matrix?

Take a symmetry operation of a crystal — say a 90-degree rotation about the z
axis, `C4z` (x -> y, y -> -x, z -> z). Apply it to the three p orbitals, which
have the same shapes as the functions x, y, z:

- p_x -> p_y, p_y -> -p_x, p_z -> p_z.

Each orbital turns into a *linear combination* of the orbitals of the same
shell. Collecting the coefficients into a matrix gives the **representation
matrix** `D(R)` of the operation on that shell. For p orbitals it is simply the
3x3 rotation matrix itself:

```
D^(1)(C4z) =  [ 0 -1  0 ]      (basis order: x, y, z)
              [ 1  0  0 ]
              [ 0  0  1 ]
```

For d orbitals the same idea gives a 5x5 matrix. Under `C4z`:
xy -> -xy, yz -> -xz, z^2 -> z^2, xz -> yz, x^2-y^2 -> -(x^2-y^2), so

```
D^(2)(C4z) =  [-1  0  0  0  0 ]      (basis order: xy, yz, z^2, xz, x^2-y^2)
              [ 0  0  0 -1  0 ]
              [ 0  0  1  0  0 ]
              [ 0  1  0  0  0 ]
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

### 1.2 How CrystOD computes D for any l

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

### 1.3 From matrices to irreps: characters and the reduction formula

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

an average over all `|G|` operations of the group. This is exactly how the
ligand-field splitting works (section 9): the characters of the d shell in the
m-3m field, `chi(g) = tr D^(2)(g)`, reduce to `Eg + T2g` — the familiar
two-below-three splitting of d orbitals in an octahedral crystal field.

In a crystal, a space-group operation `g = {R | t}` does two things to an
atomic orbital: it moves the atom to a symmetry-equivalent site, and it mixes
the orbital components on that site with `D^(l)(R)`. The character of the
crystal-orbital representation at a k point is therefore the trace of
`D^(l)(R)` summed over the atoms that `g` maps onto themselves (modulo a
lattice translation), weighted by the Bloch phase `exp(-ik.t)` of that
translation. Reducing these characters against the little-group irrep table
gives the irrep content printed by sections 2-3.

### 1.4 From irreps to basis functions: the projection operator

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
(sections 10-11):

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
`crystod-group --generate-basis --order 2` output of section 11).

The SALCs of the main command (section 5) come from the *same* projection with
the trial basis replaced by the atomic orbitals on all symmetry-equivalent
sites — `O(g)` then combines the site permutation, the Bloch phases, and
`D^(l)(R)` from section 1.2. And replacing `D^(1)(R) = R` by the axial-vector
representation `det(R) R` (magnetic moments do not flip under inversion) turns
the same machinery into the spin-multipole bases of `crystod-mag` (section 28).

*Further reading:* B. Souvignier, "Representations of crystallographic groups"
([MaThCryst summer school notes, Nancy 2010](https://www.crystallography.fr/mathcryst/pdf/nancy2010/Souvignier_irrep_slides.pdf));
the spgrep documentation, e.g. the
[symmetry-adapted tensor example](https://spglib.github.io/spgrep/examples/symmetry_adapted_tensor.html).
Run `python demo_wigner_d.py` in `example/01_wigner_d` for a printed walk-through.

## 2. Irreps of SALC for a selected element and orbital

*Example directory: `example/02_salc` (testsuite section 2)*

Decompose the crystal orbitals built from a selected element/orbital into the
irreducible representations of the little group at each special k point:

```bash
crystod -c example/test_POSCARs/221_PPOSCAR_SrTiO3 --element Ti --orbital d
```

Spinor example with irrep table:

```bash
crystod -c example/test_POSCARs/221_PPOSCAR_SrTiO3 --element Ti --orbital d \
  --spinor --kpoint 0 0 0 --show-irrep-table
```

When `--kpoint` is omitted, all special k points of the space group are analyzed.
Any arm of a special-point star is labeled correctly (since v0.3.1).

## 3. Hybridization analysis and crystal-orbital diagrams

*Example directory: `example/03_hybridization` (testsuite section 3)*

Analyze the hybridization between selected atomic orbitals (`ELEMENT_ORBITAL` pairs):

```bash
crystod -c example/test_POSCARs/221_PPOSCAR_SrTiO3 --atomic-orbital Ti_d O_p --kpoint 0 0 0
```

For each little-group irrep at the k point, the orbitals that transform as it
are listed — orbitals sharing a line are symmetry-allowed to hybridize.

### Crystal-orbital diagrams (`--diagram --co-left ... --co-right ...`)

`--diagram` draws the quantitative **crystal orbital diagram (COD)** — the
crystalline analogue of the molecular-orbital diagram of
`crystod-mol --diagram --ao-left ... --ao-right ...`.
`--co-left`/`--co-right` split the crystal into two fragment sublattices by
chemical formula (every atom must belong to one side; a count such as `O3`
validates against the primitive cell). Each fragment is treated with its
**full-electron basis** (WIEN2k-style): every core *and* valence shell of
every atom — valence shells with the extended-Hueckel parameters, core
shells with Slater-rule STO exponents and the **archived neutral-atom
PySCF levels** (`reference/atomic_level_{El}`: one Hartree-Fock/def2-svp
calculation per element, generated by `script/generate_atomic_levels.py`
and collected into `crystod/atomic_levels.py` with angular-momentum shell
assignment and an archive cross-check; beyond Kr the def2 ECP freezes the
deep cores, which are then omitted exactly as in pseudopotential DFT) —
and each fragment
**feels the removed sublattice as a point-charge lattice** with the formal
oxidation states (guessed with pymatgen; override with
`--oxidation Sr=+2 Ti=+4 O=-2`): the Sc fragment of ScF3 sits in the field
of the F lattice with Q = -1, the F3 fragment in the field of the Sc
lattice with Q = +3 — the Madelung ligand field of the pre-bonding states:

```bash
crystod --diagram -c 221_PPOSCAR_ScF3 --co-left Sc --co-right F3 --atomic-orbital Sc-3d F-2p
# -> CrystOD_221_PPOSCAR_ScF3.html

crystod --diagram -c 221_PPOSCAR_SrTiO3 --co-left SrTi --co-right O3 --atomic-orbital Ti-3d Ti-4s O-2p
```

At every special k point of the space group,

1. the Bloch orbitals of each fragment (e.g. Sr 1s...5p + Ti 1s...3d | O
   1s 2s 2p) are symmetry-adapted per irrep of the little group of k (the
   site-symmetry induced representations of the hybridization analysis
   above);
2. all intra- and inter-fragment overlaps are evaluated **exactly** as
   Bloch lattice sums of STO overlap integrals (single-zeta s/p, standard
   double-zeta d, sigma/pi/delta Slater-Koster assembly rotated with the
   exact real Wigner-D matrices), with a per-shell-pair lattice-sum cutoff
   probed from the actual STO tails (diffuse cation shells reach 30+ bohr);
3. the point-charge ligand field enters as **exact same-site matrix
   elements** <phi_i|q/|r-R||phi_j> — Laplace expansion into real
   spherical harmonics (the wigner_D_real orbital convention) with
   closed-form radial integrals, Loewdin-consistent with the identity
   on-site metric; near charge shells explicitly, the long-range tail by
   Ewald summation — validated against the NaCl Madelung constant and the
   exact 6Dq/-4Dq octahedral splitting ratio, so the fragment d states
   show the true electrostatic t2g/eg splitting. The background-dependent
   monopole of the (charged) sublattice array is omitted: it largely
   cancels against the intra-atomic charging energy absent from the
   extended-Hueckel VSIPs, and the omitted jellium-referenced values are
   printed for comparison with charged-cell DFT references;
4. the generalized eigenvalue problem with the Wolfsberg-Helmholz
   Hamiltonian H_ij = K S_ij (H_ii + H_jj)/2 (the diagonal carries the
   same-orbital neighbour-cell Bloch sums; the ligand-field blocks are
   added identically to the fragment and crystal Hamiltonians, so the
   three columns share one energy reference, and the symmetry invariance
   of H is self-checked per k point) is solved: fragment orbitals sharing
   an irrep split into bonding and antibonding crystal orbitals, orbitals
   without a partner remain rigorously nonbonding — the COD mixing rule;
5. an interactive HTML page is written with **one energy diagram per k
   point** (buttons switch the k point): fragment | crystal orbitals |
   fragment columns, correlation lines weighted by the composition,
   electron arrows, HOMO/LUMO markers, and the adjustable energy window,
   which opens on **±8 eV around the HOMO/LUMO midpoint** (the VBM/CBM
   region; "Show all energy levels" reveals the core shells). Every level
   carries a hover **wave-function sketch**: Re[psi] of all its
   atomic-orbital components on the k-commensurate supercell (2x2x2 at R,
   ...) with VESTA-style +/- lobes and VESTA-style periodic boundary
   conditions — atoms on the supercell boundary are drawn at every
   translationally equivalent position (a corner atom at all eight
   corners) with *identical* lobes, since the Bloch function is exactly
   periodic over the k-commensurate supercell (e^{ik·T_super} = 1), and
   the supercell outline appears as a dashed frame (drag-rotatable,
   degenerate partners switchable).

For ScF3 (`--co-left Sc --co-right F3`) the diagram shows the textbook
result: the fragment Sc-3d states split into t2g below eg (the octahedral
field of the F^-1 charges); at GM the F-2s GM1+/GM3+ states form sigma
bonds with Sc-4s/Sc-3d(eg) while the t2g-derived GM5+ stays 100% pure
(nonbonding); at R the eg-derived R3+ states split strongly (sigma/sigma*),
the t2g-derived R5+ states split weakly (pi/pi*), and R4+ remains pure
F-2p — the paradigmatic perovskite crystal orbital diagram.

- electrons default to the full neutral-atom counts of the archived
  atomic data (ScF3: 48; SrTiO3: 56 per cell with the ECP-28-frozen Sr
  core), filling the flat core levels and exactly the anion valence
  manifolds — the d0 insulator; `--electrons N` overrides;
- `--kpoint GM` restricts the analysis to one special point (labels only);
- `--output`/`--tolerance` as usual. Note that the irrep labels at
  zone-boundary points depend on the origin of the input structure (as in
  every SALC analysis); use the setting of your reference when comparing.

Elements H-Bi of the standard extended-Hueckel tables are parameterized
(3d/4d transition metals with double-zeta d shells). The terminal report
prints the fragment levels (labeled by their dominant element+shell) and
the full crystal-orbital table (energy, irrep, degeneracy, occupation,
composition) at every k point. Physics note: in dense cation sublattices
the diffuse valence shells (Sr 5s/5p, Ti 4s/4p, ...) make a few Bloch
combinations nearly linearly dependent on the rest of the basis; for those
the extended-Hueckel energies diverge (the well-known overlap catastrophe),
so combinations with an overlap eigenvalue below 0.2 are removed by
canonical orthogonalization — the terminal report counts them per k point.

### Quantitative crystal-orbital diagrams (`--diagram --pyscf`)

`--pyscf` replaces the extended-Hueckel model by three **periodic PySCF
calculations** that share one atomic-orbital space — the crystalline
counterpart of `crystod-mol --diagram --pyscf`:

```bash
crystod --diagram -c 221_PPOSCAR_ScF3 --pyscf --co-left Sc --co-right F3
crystod --diagram -c 221_PPOSCAR_SrTiO3 --pyscf --co-left SrTi --co-right O3
# -> CrystOD_{cell}_pyscf.html
```

| calculation    | real atoms                   | point charges        |
|----------------|------------------------------|----------------------|
| left fragment  | the `--co-left` sublattice   | the right sublattice |
| right fragment | the `--co-right` sublattice  | the left sublattice  |
| crystal        | everything                   | none                 |

The removed sublattice stays in the basis as **ghost atoms** (basis functions
without a nucleus), so all three calculations span the same AO space and the
crystal orbitals can be projected exactly onto the fragment Bloch orbitals
(counterpoise-consistent, as in the molecular version); in addition it acts on
the fragment as its **formal-charge point lattice**, evaluated as a
jellium-referenced FFT potential — validated against
`crystod.point_charge_field.ewald_site_potential` to 2e-6 eV for a neutral
charge array. Each fragment is therefore one sublattice in the Madelung field
of the other: the electronic state before chemical bond formation.

Both the remaining ions and the replacing point charges carry the formal
oxidation states (`--oxidation Sc=+3 F=-1` to override), which makes **every
cell neutral**: the monopole divergence disappears, every electron count is
even, and the fragment counts add up to the crystal's (ScF3: 8 + 24 = 32).
Neutrality alone does **not** put the three calculations on one absolute
scale, though — each pins its own G = 0 (cell-averaged) potential to zero,
and the value of that average depends on the second moment of the cell's
charge density, which changes when an ion is replaced by a bare point charge.
The raw columns are therefore offset by one rigid, k-independent constant
each (ScF3: Sc column ~2 eV, F3 column ~5 eV below the crystal), the same
reference problem as in band-offset calculations. The diagram removes it by
**deep-level (XPS-style) alignment**: the deepest chemically inert fragment
level — one that reappears in the crystal at ≥ 80% purity, e.g. the Sc 3s
semicore — must keep its energy across bond formation, and the energy zero
is put at its *pre-bonding* (fragment) value. Only the near-purest anchor
pairs across the k points set the constant (printed with anchor, purity and
k-spread); `--no-align` keeps the raw references instead.

A crystal-orbital diagram needs the crystal orbitals only where the irreps are
tabulated, so the diagram is built at the **special points of the space group**
(`--kpoint GM` for one of them) and the SCF runs on a small regular mesh whose
default follows the lattice constants, `n_i = round(8 A / |a_i|)` — 2x2x2 for a
~4 A cell; `--kmesh 4 4 4` to override. Every level, fragment and crystal, is
labelled with its little-group irrep using crystod's own machinery (the
site-permutation representation at k combined with the real-orbital Wigner
matrices, projected with the spgrep characters); the representation is
**verified against PySCF's own overlap matrix** (`D+ S D = S`, residual printed
per k point) before it is used. The terminal report also lists the
site-symmetry induced representation of every (element, shell) block — the
`--element/--orbital` decomposition, recomputed from the same representation —
and the same-irrep coupling `<phi_left|F(k)|phi_right>` of the converged
crystal Fock operator with the resulting two-level mixing fraction, since two
levels of one irrep only mix appreciably when that matrix element is large
compared with their separation.

Crystal-orbital lines are colored by **bonding character** (extended-
Hückel and `--pyscf` alike, and the same applies to the molecular
`crystod-mol --diagram` pages): blue = bonding, black = nonbonding,
red = antibonding, from the COOP-style left–right overlap population
P = 2 Re[c_L† S_LR c_R] of each eigenstate (charge accumulated between
the sublattices; exactly 0 for symmetry-nonbonding states; the value
appears in the level tooltip and the legend in the page footer). The
fragment/sublattice columns are drawn in the **VESTA color of each
level's dominant element** — a composite sublattice like SrTi shows Sr
levels in the Sr color and Ti levels in the Ti color — with occupation
still visible through the electron arrows. Semicore orthogonality tails (shells
whose own bands lie ≥ 10 eV below their fragment column's HOMO) are
excluded unless the level *is* that semicore band — with the Sc 3s tail
included, the bonding σ R1+ of ScF3 would read P = −0.015 instead of
+0.112. Antibonding |P| is systematically larger than bonding |P| (the
usual non-orthogonal COOP asymmetry).

The raw SCF has **no point-group constraint**: the FFT grid breaks
degeneracies numerically (<1 meV for semilocal functionals, but several
0.1 eV for grid-evaluated exact exchange of hybrids at low cutoffs). All
displayed levels are therefore re-diagonalized from the **group-averaged
Fock** F̄ = (1/|G|) Σ_g D(g)† F D(g), which is exactly invariant under the
verified AO representation — degeneracies are exact by construction, and the
raw breaking magnitude is printed per k point (`--no-symmetrize` disables
this for debugging). `--max-l L` drops basis shells above l = L, e.g.
`--max-l 2` removes the f polarization functions whose tails otherwise
appear as small cation-4f weights in the anion-p valence compositions.

A fragment level may carry weight on the **ghost basis** of the removed
sublattice — the symmetry-allowed variational tail of the fragment density
toward the point charges (for ScF3 it appears exactly in the irreps with a
Sc partner: GM4⁻ = Sc p, R1⁺ = Sc s, and not in GM5⁻). Fragment sketches
therefore draw only the fragment's own components, and the ghost weight of
each level is reported in the terminal and in the Level-details panel.
`--no-ghost` applies the hard constraint instead: the removed sublattice's
functions are excluded from the fragment variational space entirely (no
fragment wave function can sit on the removed atoms; costs the counterpoise
consistency — ~2 mHa BSSE per ScF3 fragment — and shifts some crystal-orbital
composition onto the other fragment's polarization shells, which then have to
describe the tail region).

The **composition list** — the Level-details bars, the hover tooltip and
the terminal all quote one identical list — is the per-(element, shell)
**AO population of the PySCF eigenvector** (the same partial-charge measure
as `crystod --dos --pyscf`): Löwdin |coefficient|² in the symmetrically
orthogonalized basis by default — the orthonormal set closest to the atomic
orbitals, i.e. the intuitive "squared LCAO weight" made rigorous for a
non-orthogonal basis (non-negative, summing to exactly 100%; Mulliken's
overlap cross terms instead go negative or overshoot on diffuse empty
levels). Symmetry does the orbital selection: a crystal state of irrep Γ
only picks up the Γ-adapted combination of each shell (symmetry-forbidden
shells project to ~0 and are culled at 0.1%), so every entry carries the
crystal irrep — `Sc 3d R5+ 76.0%`. Projections onto the *fragment
eigenstates* are deliberately **not** displayed as percentages (they still
position the correlation lines and the alignment anchors): a fragment level
such as "Sc 3d R5+" itself mixes 3d and 4d AO character, so its weights
disagree with the AO populations, and the two lists shown side by side
read as a contradiction.

Options: `--basis` (default `gth-dzvp-molopt-sr`), `--pseudo` (default
`gth-pbe`), `--xc` (default `pbe`, or `hf`), `--ke-cutoff` (default 200
Hartree), `--kmesh`, `--sigma` (Fermi smearing in eV; a non-converging SCF
escalates through a smearing/level-shift ladder automatically and reports it).
`--chk FILE` is the WAVECAR analogue: after the SCFs the three converged
density matrices (all the band step needs) and the defining parameters are
saved to FILE, and a rerun with the file present skips all three SCFs — the
parameters are verified first, and a mismatch aborts naming the offending
options. `--projection lowdin|mulliken` selects the population measure for
the sketch lobe sizes and the per-(element, shell) rows (default Löwdin;
the two measures agree in kind on occupied levels but disagree strongly on
diffuse empty ones, where atomic attribution is convention-dependent). The
full same-irrep resonance-integral tables `<φ_left|F(k)|φ_right>` (all
pairs, |H|, ΔE, mixing fraction, per k point, aligned scale) are written to
`<output-stem>_coupling.txt` next to the HTML. Crystal-column levels are
labeled `GM4- #2` — the second GM4⁻ multiplet from the bottom, the same
`#N` convention as the fragment columns (`GM4-(2)` read like a degeneracy
count); the extended-Hückel `--diagram` uses the same format.
The hover wave-function sketches are embedded here too, drawn from the PySCF
AO coefficients: lobe signs and orientation come from the r0 = 2 bohr radial
amplitudes (so semicore orthogonalization tails keep their inverted phases),
while lobe **sizes** follow the per-(atom, l) Löwdin population — a diffuse
gth-dzvp Sc 4p has 5.4x the amplitude of an F 2p at r0, which used to draw an
~8%-population Sc admixture at 83% of the largest F lobe; it now draws at its
electron weight (~0.4 relative). The calibration factor of each (atom, l)
channel comes from the multiplet-summed populations, so symmetry-equivalent
atoms draw exactly equal lobes and the sigma/pi contrast between degenerate
partners survives. For speed, the SCF solves only the irreducible wedge of the
k-mesh and the fragments start from the crystal density restricted to their
own sublattice block; `--ke-cutoff 100` roughly halves the cost and
`--basis gth-szv-molopt-sr` is a fast draft mode. crystod sets the PySCF
OpenMP thread count to all cores automatically -- but note that the macOS
pip wheels of pyscf are built *without* OpenMP (a warning is printed), in
which case each run is single-threaded and the way to use a many-core
machine is to run several systems in parallel.

## 4. Star of k

*Example directory: `example/04_star_of_k` (testsuite section 4)*

Display the star of k: the set of inequivalent k points generated from a given
k point by the space-group rotations (`k' = k R`, modulo reciprocal lattice):

```bash
crystod --star-of-k -c example/test_POSCARs/221_PPOSCAR_ScF3 --kpoint 0.5 0.5 0
crystod --star-of-k -c example/test_POSCARs/221_PPOSCAR_ScF3 --kpoint M
```

The output shows `|G|`, the little co-group order `|G_k|`, `|star of k|`, and each
arm with its coset-representative operation in Seitz notation.

The star of q is also displayed automatically in `crystod-phonon --modulation`
for each selected q point, which is useful when combining arms of the same star
in multi-q modulations.

## 5. SALC basis visualization (`--visualize`)

*Example directory: `example/05_visualized_basis` (testsuite section 5)*

Build the SALCs of a selected element/orbital at a k point, print the
irreducible decomposition and per-atom SALC coefficients, and export an
interactive 3D HTML visualization:

```bash
crystod -c example/test_POSCARs/221_PPOSCAR_ScF3 --element F --orbital p --kpoint 0 0 0 --visualize
crystod -c example/test_POSCARs/221_PPOSCAR_ScF3 --element Sc --orbital d --kpoint 0 0 0 --real-coefficient --visualize
crystod -c example/test_POSCARs/221_PPOSCAR_ScF3 --element F --orbital p --kpoint GM --output SALC_F-p_GM.html --visualize
```

### PySCF eigen-levels in the viewer (`--visualize --pyscf`)

The SALCs above are the symmetry-adapted *basis* — the states before any
Hamiltonian. With `--pyscf` the same page shows the actual **PySCF
eigenstates**, either of one fragment sublattice (the pre-bonding states
in the removed sublattice's point-charge field, exactly the
`--diagram --pyscf` columns) or of the full crystal (the states after
bonding):

```bash
crystod --visualize --pyscf -c 221_PPOSCAR_ScF3 --sublattice Sc --bond Sc F 3 --real-coefficient --chk scf3.chk
crystod --visualize --pyscf -c 221_PPOSCAR_ScF3 --sublattice F3 --bond Sc F 3 --real-coefficient --chk scf3.chk
crystod --visualize --pyscf -c 221_PPOSCAR_ScF3 --bond Sc F 3 --real-coefficient --chk scf3.chk
```

No `--element/--orbital/--kpoint` are needed: the special k points come
from the space group automatically (one page per k point; `--kpoint GM`
restricts the output). The SALC-basis table becomes **Mode | Irrep |
Comp. | Energy (eV)** — one row per degenerate partner, level labels in
the diagram convention (`GM4- #2`, `Sc 3d R3+`), energies on the shared
deep-level-aligned scale, so the Sc, F3 and crystal pages are directly
comparable (`--no-align` keeps the raw references). Clicking a row draws
the eigenvector's wave function: every element, all shells s..f, lobe
sizes calibrated to the `--projection` populations with one shared
normalization per level, on the k-commensurate display cell with the
usual VESTA-style bonds and polyhedra. All `--diagram --pyscf` options
apply; with a shared `--chk` the three commands above pay the SCF once
(~5 s per page set afterwards). The default energy window is
HOMO−15 .. LUMO+10 eV of the displayed column (`--window LO HI`
overrides; a full fragment spectrum would put hundreds of plotly
surfaces on one page).

`--diagonalize` canonicalizes the degenerate partners (RREF, as in the
diagram sketches): the SCF returns an arbitrary unitary mixture within
a multiplet, which draws e.g. a tilted d_z², and the canonical rotation
makes the components axis-aligned (the R5+ t2g triplet of ScF3 becomes
pure d_xy / d_yz / d_xz) — energies are unchanged. Each channel's sign
is probed at the radius (1.5–3 bohr) where its accumulated amplitude is
largest, since a fixed radius can sit on an orthogonalization node.
Reading note: a valence level can *faithfully* look "antibonding" around
a semicore-carrying atom — the F 2p σ R1+ level of ScF3 carries
Sc 3s(semicore) c = +0.26 vs Sc 4s c = +0.05, so the true wave function
(verified against real-space `pbc_eval_gto`) has a radial node 0.7 Å
from Sc and is negative near the atom: the orthogonality tail against
the −50 eV Sc 3s band, not a drawing error. The scale separation is the
⟨r⟩ one: in this basis ⟨r⟩(Sc 3s) = 0.77 Å vs ⟨r⟩(Sc 4s) = 1.86 Å
against the 2.03 Å bond — the compact semicore shell cannot bond, its
admixture is on-site orthogonality. `--valence-only` applies exactly
this reasoning to the drawing: shells whose occupied fragment bands lie
more than 12 eV below the crystal VBM on the aligned scale (Sc 3s, Sc
3p, F 2s here; detected automatically and printed) are dropped from the
drawn wave functions, so the σ level shows its bonding Sc 4s component
(Sc s flips from −0.48 to +0.47 against the same F p lobes), while
levels a semicore shell dominates — the semicore bands themselves —
keep it.

The irreducible decomposition is consistent with the plain SALC analysis
(e.g. `2.0 [GM4-(3)] + 1.0 [GM5-(3)]` for F_p at GM in Pm-3m ScF3).
`--kpoint` accepts either three primitive reciprocal coordinates or a
high-symmetry label such as `GM`, `X`, `M`, or `R`. For k != 0, the commensurate
supercell is built automatically and the Bloch phase factor is applied to the
displayed coefficients.

The HTML file is always written, auto-named `SALC_{element}_{orbital}_{kpoint}.html`
(`--output` overrides the path). It is a standalone interactive viewer (plotly via
CDN): a sidebar shows the structure summary, the irreducible decomposition, and a
clickable table of all SALC basis vectors (mode space / irrep / component); the
main viewport shows the orbital lobes at each atom, colored by sign in the VESTA
convention (positive = yellow, negative = blue) and rendered opaque with
VESTA-like lighting by default, so the front/back occlusion of overlapping lobes
is always correct. The lobe-opacity slider switches to a translucent view, where
a distance-based fade keeps near/far lobes distinguishable. Cell/atom display
controls and a camera-synchronized a/b/c compass (a red, b green, c blue;
with `--conventional` it shows both lattices — the primitive vectors as short
pastel arrows labeled `a_prim`/`b_prim`/`c_prim` and the conventional vectors
of the displayed cell as full-color arrows labeled `a_conv`/`b_conv`/`c_conv`)
sit in
the sidebar and the lower-left corner.

**The viewer layout is modeled after the
[phonon website](https://henriquemiranda.github.io/phononwebsite/) by Henrique
Miranda ([github.com/henriquemiranda/phononwebsite](https://github.com/henriquemiranda/phononwebsite),
BSD-3-Clause) — CrystOD imitates its sidebar-plus-viewport design (no code is
copied; the 3D rendering uses plotly).**

Further options:

- `--mode-index N` restricts the output to one irrep-grouped mode space (1-based).
- `--conventional` displays the SALC in the conventional cell instead of the
  primitive cell (the primitive-to-conventional matrix is derived from the
  detected centring, exactly as in `crystod-phonon --vector`; file names get a
  `_conv` suffix, and for k != 0 the conventional cell is multiplied until the
  Bloch phase is commensurate).
- `--bond EL1 EL2 MAX` (repeatable, e.g.
  `crystod --visualize -c 221_PPOSCAR_ScF3 --element F --orbital p --kpoint 1/2 1/2 1/2 --real-coefficient --bond Sc F 2.3`)
  draws bonds between the two elements up to MAX Angstroms and renders the
  coordination polyhedra around the EL1 atoms as translucent convex hulls, as
  in VESTA's Polyhedral style; atoms at the cell boundary are completed
  VESTA-style so the polyhedra are not cut off.

### Example 1: Sc d orbitals of ScF3 at the R point

```bash
crystod -c 221_PPOSCAR_ScF3 --element Sc --orbital d --kpoint R --visualize --real-coefficient --bond Sc F 2.5
```

The viewer below is the live output of that command — an R-point example, so
the commensurate 2x2x2 supercell is built and the Bloch phase alternates the
lobe signs from cell to cell (the Sc d SALCs at R decompose as
`R3+(2) + R5+(3)`). Click any row of the SALC table in the sidebar to switch
the displayed basis vector, toggle the ScF6 polyhedra, and drag to rotate:

```{raw} html
<iframe src="_static/embed/SALC_Sc_d_R.html" width="100%" height="660" loading="lazy" style="border:1px solid #8884; border-radius:8px; background:#fff;"></iframe>
<p style="margin-top:0.3em"><a href="_static/embed/SALC_Sc_d_R.html" target="_blank">Open the ScF3 SALC viewer full-screen</a></p>
```

### Example 2: Ce f orbitals of CeO2 in the conventional cell (`--conventional`)

```bash
crystod --visualize -c 225_PPOSCAR_CeO2 --element Ce --orbital f --kpoint 0 0 0 --bond Ce O 3 --real-coefficient --conventional
```

CeO2 is face-centred (Fm-3m), so its primitive cell is the small rhombohedral
one — hardly the picture one has in mind for fluorite. `--conventional`
switches the display to the cubic conventional cell (four formula units, the
familiar CeO8 cube arrangement) while the SALC coefficients themselves are
unchanged. The corner compass then shows **both** lattices: the primitive
vectors as short pastel arrows (a<sub>prim</sub>, b<sub>prim</sub>,
c<sub>prim</sub> — the face diagonals) and the conventional vectors of the
displayed cell as full-color arrows (a<sub>conv</sub>, b<sub>conv</sub>,
c<sub>conv</sub> — the cubic axes):

```{raw} html
<iframe src="_static/embed/SALC_Ce_f_GM_conv.html" width="100%" height="660" loading="lazy" style="border:1px solid #8884; border-radius:8px; background:#fff;"></iframe>
<p style="margin-top:0.3em"><a href="_static/embed/SALC_Ce_f_GM_conv.html" target="_blank">Open the CeO2 conventional-cell SALC viewer full-screen</a></p>
```

- `--real-coefficient` re-combines degenerate SALC components into
  real-coefficient form whenever the projected space is closed under complex
  conjugation (real-type irreps at k = -k points). For example, the GM3+ (Eg)
  SALCs of Sc_d, which are otherwise produced as `(d_z2 +/- i d_x2-y2)/sqrt(2)`,
  become `d_z2` and `d_x2-y2`. The spanned space is unchanged; only the unitary
  basis choice within the irrep space is rotated.
