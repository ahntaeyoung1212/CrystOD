# crystod-mag

## 28. Symmetry-adapted spin bases: cluster multipoles / SAMM

*Example directory: `example/28_spin_basis` (testsuite section 28)*

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

When `--qpoint` is omitted, the spin (axial-vector) irrep decomposition is
listed for every special k point of the space group — the magnetic counterpart
of the `crystod` SALC survey (e.g. for AlNi3: GM: `2.0 [GM4+(3)] + 1.0 [GM5+(3)]`,
R: `R2+ + R3+ + R4+ + R5+`, ...).

For the Mn3Ir-type Ni 3c cluster of AlNi3 this yields
`9 dims = 2 x GM4+(3) + GM5+(3)`: the GM4+ (T1g) cluster dipole (FM), the GM4+
(T1g) cluster octupole (AFM: the experimentally realized 120-degree structure of
Mn3Ir, which shares the irrep with the dipole and hence allows the anomalous
Hall effect), and the GM5+ (T2g) cluster octupole (AFM). Every AFM basis
satisfies `sum_i S_i = 0` exactly.

The construction is the SALC projection used elsewhere in CrystOD with the
Cartesian part replaced by `det(R) R` (spins are axial vectors); irrep labels
come from spgrep + irreptables as usual. Within a multiply-occurring irrep the
unique net-moment (dipole) combination is split off from the net-zero
(higher-multipole) ones, and multipole ranks (dipole, octupole, ...) are
assigned representation-theoretically from the parity-resolved
angular-momentum characters (Suzuki's Table III logic).

Each basis vector is exported as
`POSCAR_<formula>_spin_<irrep>_<dipole|octupole|...>_<direction>.vesta`
(e.g. `POSCAR_AlNi3_spin_GM4+_octupole_111.vesta` for the 120-degree Mn3Ir-type
state), with spin arrows on the magnetic atoms. For q != 0 (e.g. `--qpoint R`)
the commensurate magnetic supercell is built automatically, with the Bloch phase
applied to the spins and the MAGMOM/VESTA output referring to the supercell.
`--conventional` exports the spin structures in the conventional cell instead of
the primitive cell (VESTA files get a `_conv` suffix, and for q != 0 the
conventional cell is multiplied until the Bloch phase is commensurate).
