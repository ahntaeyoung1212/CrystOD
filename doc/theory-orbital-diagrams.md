# How the orbital diagrams are computed

The orbital diagrams of CrystOD — the molecular ones of `crystod-mol --diagram`
(section 43) and the crystal ones of `crystod --diagram` (section 3) — are built
by the same recipe: symmetry-adapt the fragment orbitals, evaluate every overlap
between them, and solve the resulting eigenvalue problem. This page describes
that engine, in its qualitative (extended-Hückel) and quantitative (PySCF) forms.
Reading the diagrams themselves needs none of this; the command pages are
self-contained.

## How the molecular-orbital diagram is built (extended-Hückel engine)

The construction follows the textbook route, made quantitative step by step:

1. the molecular point group is detected and the ligand atomic orbitals are
   symmetry-adapted per irrep (the molecular SALCs of `crystod-mol --element
   --orbital`, section 42) — for NH3,
   `A1: [1s(H1) + 1s(H2) + 1s(H3)]` and
   `E: [1s(H1) - 1s(H3), 1s(H1) - 2 1s(H2) + 1s(H3)]`;
2. every valence orbital — the outer, chemically active orbital of each atom —
   is represented by a Slater-type orbital (STO) with the standard
   extended-Hückel exponents: one exponential function per orbital
   (single-zeta) for the s and p shells, and for most valence d and f shells
   a double-zeta contraction of two exponentials with fixed coefficients,
   which the AO-parameter block of the report prints. Each orbital also
   carries a diagonal energy H_ii, taken from its
   valence-state ionization energy, which measures how tightly that orbital
   binds an electron in the free atom;
3. all two-center overlap integrals between the STOs — the numbers S that
   measure how much two orbitals on different atoms occupy the same region of
   space — are evaluated **exactly**, by Gauss-Laguerre x Gauss-Legendre
   quadrature in prolate-spheroidal coordinates, to machine precision and
   validated against the analytic H 1s-1s formula. Ligand-ligand overlap is
   therefore *not* neglected. The overlaps between each ligand SALC and each
   central-atom atomic orbital (AO) are printed per irrep
   (NH3: `< a1 | N 2s > S = 0.72`, `< e | N 2p > S = 0.57`);
4. orbitals belonging to different irreps cannot mix, so the problem falls
   apart into one small block per irrep. Within each block the generalized
   eigenvalue problem — the secular equation whose solutions are the
   molecular-orbital energies and their coefficients — is solved with the
   Wolfsberg-Helmholz off-diagonal elements
   H_ij = K S_ij (H_ii + H_jj)/2 (K = 1.75). A large overlap integral
   therefore directly produces a large splitting between the bonding and the
   antibonding level (this is symmetry-adapted extended Hückel).

## Why ghost atoms? The basis-set superposition error (BSSE)

A quantum-chemistry calculation builds the wave function — the mathematical
object from which every property of the electrons follows — out of a
**finite** set of atom-centred basis functions, so its quality depends on how
many functions sit near the electrons. Compute the O atom of H2O in the basis
of O alone, then the whole molecule in the basis of O **and** the two H atoms,
and the molecule has extra freedom to describe its electrons — more functions
to build them from — that the isolated fragment never had. Its energy drops
for a reason that has nothing to do with chemical bonding.
The spurious part of the stabilization is the **basis-set superposition error
(BSSE)** — an artefact of the finite basis rather than a physical effect. It
makes computed interaction energies look too attractive, and it does not
vanish simply by enlarging the basis on one side only.

The standard cure is the **counterpoise correction** of Boys and Bernardi:
compute each fragment in the *full* molecular basis, keeping the basis
functions of the removed atoms but not their nuclei or electrons. Such
nucleus-free, electron-free basis centres are called **ghost atoms**. Both
fragments then carry exactly the same basis-set quality as the molecule, so
the difference E(mol) − E(left) − E(right) contains bonding only.

CrystOD needs this twice over. First, the interaction energy printed under the
diagram is BSSE-free. Second, because all three calculations live in *one and
the same* AO space, every molecular MO can be **projected exactly** onto the
fragment MOs — written as an exact combination of them, with no fitting and no
approximation. That is what makes the correlation lines and the percentage
compositions quantitative rather than merely indicative. The price is that a few
fragment solutions end up living mostly on the ghost centres; those are
filtered out by the real-atom Mulliken-population test described on the
`crystod-mol --diagram --pyscf` command page (section 43).

## Two conventions that keep the molecular fragment columns honest

**Spin and spherical averaging.** The electrons of a partly filled frontier
shell are spread evenly over it, so every degenerate orbital of the shell takes
the same fractional occupation. Without this, a partially filled shell (C 2p²,
the t2² of an H4 cage) would break the point-group symmetry of the fragment,
whereas the pre-bonding reference is supposed to preserve it.

**The ghost-population filter.** A fragment level may end up living mostly on
the ghost centres — a variational tail of the fragment density toward the atoms
that were removed, not a genuine state of that fragment. Each fragment level is
therefore judged by its real-atom Mulliken population, the share of the density
that Mulliken's partitioning assigns to the real atoms rather than to the ghost
centres. Below 35% the level is a BSSE artefact and is dropped from the diagram
and from the compositions; the surviving projections are weighted by the
real-atom population. In benzene this is what makes the pure-π 1e1g HOMO come
out as 100% C6 1e1g, with no spurious "H6" character borrowed from
ghost-carbon levels. Fragment orbital sketches likewise draw the real atoms
only, and degenerate partners are canonicalized so that they match the SALCs
drawn by `--visualize` exactly.

In the crystalline `--pyscf` diagram the same tail appears in the irreps where
the fragment has a partner (for ScF3: GM4⁻ = Sc p, R1⁺ = Sc s, and not in
GM5⁻). There the test is put on the ghost share itself: a fragment level
(a degenerate set) with more than 35% of its Mulliken population on the ghost
basis of the removed sublattice is dropped from the diagram and from the
compositions (`GHOST_FRACTION_THRESHOLD = 0.35` in
`crystod/crystal_orbital_pyscf.py`). The levels that survive keep their
projections unweighted and report their ghost weight instead — in the terminal
report from 5% and in the Level details of the HTML page from 0.5% — and their
sketches draw the fragment's own sublattice only. `--no-ghost` removes the
ghost basis from the fragment variational space entirely — at the cost of the
counterpoise consistency (~2 mHa BSSE per ScF3 fragment).

## How the crystal-orbital diagram is built (extended-Hückel engine)

This is what `crystod --diagram --co-left ... --co-right ...` (section 3 of
`crystod`) evaluates at every special k point of the space group:

1. the Bloch orbitals of each fragment (e.g. Sr 1s...5p + Ti 1s...3d | O
   1s 2s 2p) are symmetry-adapted per irrep of the little group of k (the
   site-symmetry induced representations of the `--atomic-orbital`
   hybridization analysis);
2. all intra- and inter-fragment overlaps are evaluated **exactly** as
   Bloch lattice sums of STO overlap integrals (single-zeta s/p, the same
   single- or double-zeta d and f shells as the molecular engine,
   sigma/pi/delta/phi Slater-Koster assembly rotated with the exact real
   Wigner-D matrices), with a per-shell-pair lattice-sum cutoff
   probed from the actual STO tails (diffuse cation shells reach 30+ bohr);
3. the point-charge ligand field enters as **exact same-site matrix
   elements** <phi_i|q/|r-R||phi_j> — Laplace expansion into real
   spherical harmonics (the `wigner_D_real` real-orbital convention of the
   representation-matrix background above) with
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
   without a partner remain rigorously nonbonding — the COD mixing rule.

The interactive HTML page that these steps produce is described with the
command in section 3 of `crystod`.

One numerical caveat belongs to step 4. In dense cation sublattices the diffuse
valence shells (Sr 5s/5p, Ti 4s/4p, ...) make a few Bloch combinations nearly
linearly dependent on the rest of the basis, and for those the extended-Hückel
energies diverge — the well-known **overlap catastrophe**. Combinations with an
overlap eigenvalue below 0.2 are therefore removed by canonical
orthogonalization, and the terminal report counts them per k point. A second,
purely conventional caveat: the irrep labels at zone-boundary points refer to
the origin of the input structure when the input is already in the ISO-IR
setting (Sr or Ti at the origin of SrTiO3), as in every SALC analysis, so
compare against a reference in the same setting. A diagonal supercell of
such an input (of any size) is labelled like the cell; any other
description of the same crystal is labelled in a canonical frame, which keeps
the input's own origin when that is the origin of a standard description, so
shifted copies get the same labels whatever their orientation or basis, while
an unshifted re-based cell or a non-diagonal supercell can get names that
differ from theirs, or from the cell's, by a normalizer element (see
[Irrep labels and the frame of a structure](theory-representations.md#irrep-labels-and-the-frame-of-a-structure)).

## Ghost atoms, cell neutrality and the deep-level alignment (crystal `--pyscf`)

In the quantitative crystal-orbital diagram (`crystod --diagram --pyscf`,
section 3 of `crystod`) the three periodic SCF calculations — left fragment,
right fragment, crystal — are tied together like this.

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

## Why `--onsite` takes the fragment columns from the crystal Fock

The default `crystod --diagram --pyscf` mode (section 3 of `crystod`) solves
each fragment as its own SCF (formal-charge ions + point charges + ghost
basis), so the three columns live on three different electrostatic
references; deep-level alignment removes one rigid constant per column,
but the *site-dependent* part of the point-charge-model error remains — a
weakly bonding level can appear ~1.5 eV *above* its parent (SrTiO3: the
chemically inert Sr 4s column proves the shift is pure environment).
`--onsite` removes that error structurally, by reading the fragment columns
off the converged crystal Fock operator instead.

Whole-sublattice blocks are *not* used for those columns: with a diffuse
basis the raw AO block lets cation functions fall
variationally into the removed side's potential wells (disguised anion
states), and orthogonalized blocks load strongly-overlapped shells with
large orthogonalization penalties — the per-shell Rayleigh quotient has
no variational freedom to abuse.

## What the composition percentages mean (Löwdin vs Mulliken)

The composition list quoted in three places — the Level-details bars, the hover
tooltip and the terminal — is one and the same list: the per-(element, shell)
**AO population of the eigenvector**, the same partial-charge measure as
`crystod --dos --pyscf`. The default is Löwdin, |c|² in the symmetrically
orthogonalized basis: the orthonormal set closest to the atomic orbitals, i.e.
the intuitive "squared LCAO weight" made rigorous for a non-orthogonal basis
(non-negative, summing to exactly 100%). Mulliken's overlap cross terms instead
go negative or overshoot on diffuse empty levels, which is why
`--projection mulliken` is offered but not the default; the two measures agree
in kind on occupied levels and disagree strongly on empty ones, where atomic
attribution is convention-dependent in the first place.

Symmetry does the orbital selection: a crystal state of irrep Γ only picks up
the Γ-adapted combination of each shell (symmetry-forbidden shells project to
~0 and are culled at 0.1%), so every entry carries the crystal irrep —
`Sc 3d R5+ 76.0%`. Projections onto the *fragment eigenstates* are deliberately
**not** shown as percentages, although they still position the correlation lines
and the alignment anchors: a fragment level such as "Sc 3d R5+" itself mixes 3d
and 4d AO character, so its weights disagree with the AO populations, and the
two lists side by side would read as a contradiction.

The wave-function sketches follow the same measure. Lobe *signs and
orientations* come from the r0 = 2 bohr radial amplitudes, so orthogonalization
tails keep their inverted phases, while lobe *sizes* follow the per-(atom, l)
Löwdin population — a diffuse gth-dzvp Sc 4p has 5.4x the amplitude of an F 2p
at r0, which would otherwise draw an ~8%-population Sc admixture at 83% of the
largest F lobe. The calibration factor of each (atom, l) channel comes from the
multiplet-summed populations, so symmetry-equivalent atoms draw exactly equal
lobes and the sigma/pi contrast between degenerate partners survives.

## Why a valence level can look antibonding (`--valence-only`)

A valence level can *faithfully* look "antibonding" around an atom that carries
a semicore shell. In ScF3 the F 2p σ R1+ level carries Sc 3s (semicore)
c = +0.26 against Sc 4s c = +0.05, so the true wave function — verified against
a real-space `pbc_eval_gto` evaluation — has a radial node 0.7 Å from Sc and is
negative near the atom. That is the orthogonality tail against the −50 eV Sc 3s
band, not a drawing error.

The scale separation is the ⟨r⟩ one: in this basis ⟨r⟩(Sc 3s) = 0.77 Å against
⟨r⟩(Sc 4s) = 1.86 Å and a 2.03 Å bond — the compact semicore shell cannot bond,
and its admixture is on-site orthogonality. `--valence-only` applies exactly
this reasoning to the drawing: shells whose occupied fragment bands lie more
than 12 eV below the crystal VBM on the aligned scale (Sc 3s, Sc 3p, F 2s here;
detected automatically and printed) are dropped from the drawn wave functions,
so the σ level shows its bonding Sc 4s component — Sc s flips from −0.48 to
+0.47 against the same F p lobes — while levels a semicore shell dominates (the
semicore bands themselves) keep it.

The same asymmetry appears in the COOP colors: semicore orthogonality tails
(shells whose own bands lie ≥ 10 eV below their fragment column's HOMO) are
excluded from the overlap population unless the level *is* that semicore band.
With the Sc 3s tail included, the bonding σ R1+ of ScF3 would read P = −0.015
instead of +0.112. Antibonding |P| is systematically larger than bonding |P|,
the usual non-orthogonal COOP asymmetry.

## What the crystal-orbital regression test asserts

Testsuite section 3 exercises the crystal `--diagram --pyscf` engine with a
deliberately small run, skipped when pyscf is not installed:
`--diagram --pyscf --onsite` on NaCl at X with `--kmesh 1 1 1
--ke-cutoff 80`, which needs a single crystal SCF and about a minute of
wall clock. It asserts the report (one SCF and no fragment SCF, the
`D+ S D = S` verification, the induced representations of every shell,
the Cl-3p valence band bonding with Na 3p, the written coupling table)
and two invariants of the embedded wave-function sketches, which would
otherwise need manual inspection: the drawn σ lobe phases along the Na–Cl
bonds must reproduce the bonding/antibonding letter that the engine
derives independently from the COOP overlap population, and the Na/Cl
ratio of the p-channel lobe sizes must equal the square root of the
ratio of the level's Löwdin populations — the per-(atom, l) lobe-size
calibration documented with the command in section 3 of `crystod`, whose
collapse the earlier extended-Hückel tests could not see. A `--conventional` rerun against the written `--chk` then
covers the conventional-cell display path with no second SCF.

## References for the extended-Hückel construction

M. Wolfsberg and L. Helmholz, J. Chem. Phys. 20, 837 (1952);
R. Hoffmann, J. Chem. Phys. 39, 1397 (1963).
