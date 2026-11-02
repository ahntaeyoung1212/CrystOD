from phonopy.interface.calculator import read_crystal_structure
from crystod import salc

for name in ("224_PPOSCAR_Ag2O", "227_PPOSCAR_Si"):
    cell, _ = read_crystal_structure("../test_POSCARs/" + name, interface_mode="vasp")
    builder = salc.SymmetryAdaptedOrbitalBasis(cell=cell)
    table = salc.little_group_dipole_table(builder, [0, 0, 0])
    print(name, "GM5+ -> GM1+:", table.components("GM5+", "GM1+"),
          " GM5+ -> GM4-:", table.components("GM5+", "GM4-"))
