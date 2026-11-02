"""Demo of crystod.operations.wigner_D_real (testsuite section 1).

wigner_D_real(l, R) returns the (2l+1)x(2l+1) representation matrix of the
O(3) operation R (3x3 Cartesian matrix) on the real spherical harmonics of
angular momentum l. This is the core engine behind every orbital-symmetry
analysis in CrystOD (SALC, ligand field, basis functions, ...).
"""
import numpy as np

from crystod.operations import wigner_D_real

np.set_printoptions(precision=3, suppress=True)

# C4 rotation about z
c4z = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

print("* l = 1 (p orbitals): D(C4z) equals the 3x3 rotation matrix itself *")
print(wigner_D_real(1, c4z), end="\n\n")

print("* l = 2 (d orbitals): D(C4z) *")
print(wigner_D_real(2, c4z), end="\n\n")

print("* inversion on l = 0..3: D(-E) = (-1)^l x identity (parity) *")
inversion = -np.eye(3)
for l in range(4):
    d = wigner_D_real(l, inversion)
    print(f"  l = {l}: D = {d[0, 0]:+.0f} x identity({2 * l + 1})")
print()

print("* character of C4z on the d shell (trace of D): expected -1 *")
print(f"  chi_d(C4z) = {np.trace(wigner_D_real(2, c4z)):+.1f}")
