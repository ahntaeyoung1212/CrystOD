"""
__author__ = "Hiroki Koiso"
__copyright__ = "Copyright 2023, Nakajima group"
__version__ = "1.1"
__maintainer__ = "Hiroki Koiso, Yasuhide Mochizuki"
__email__ = "koiso.h.aa@m.titech.ac.jp, mochizuki.y.af@m.titech.ac.jp"
__status__ = "Development"
__date__ = "October 26, 2023"
"""

from phonopy.phonon.character_table import character_table as all_character_tables
from argparse import ArgumentParser, RawTextHelpFormatter, RawDescriptionHelpFormatter, ArgumentDefaultsHelpFormatter
import numpy as np

### PARSER STRUCTURE ###
class MyHelpFormatter(RawTextHelpFormatter, RawDescriptionHelpFormatter, ArgumentDefaultsHelpFormatter):
    pass
desc = """
This program decomposes a reducible representation to ireducible representations.
"""
parser = ArgumentParser(description=desc, formatter_class=MyHelpFormatter)
parser.add_argument("--point-group", "-pg", dest="pg", required=True, type=str,
                    help="Point group")
args = parser.parse_args()
### --------------- ###


""" get character table """
try:
    ct = all_character_tables[args.pg][0]
except:
    print(f"ERROR: \"{args.pg}\" is not in point groups.")
    print("You need to chose from the follownig: ")
    print(list(all_character_tables.keys()))
    exit(0)


all_irreps = list(ct['character_table'].keys())
list_mul = [np.array(i).shape[0] for i in ct['mapping_table'].values()]
mul = np.array([list_mul])
sym_op_with_mul = [str(multiplicity) + symmetry_operation
                   for symmetry_operation, multiplicity in zip(list(ct['rotation_list']), list_mul)]


def decompo(rrep):
    """ Decompose reducible representation.

    Parameter
    ----------
    rrep : list
        Characters that you want to decomposite.
    
    Return
    ----------
    results : dict, key=irrep: value=(number of irrep)
        
    """

    irreps_num_list = []
    for character in ct['character_table'].values():
        character = np.array([character])
        irreps_num = round(float(np.dot(mul * character, np.array(list(rrep.values())).T) / mul.sum()))
        irreps_num_list.append(irreps_num)

    results = {irrep: num for irrep, num in zip(all_irreps, irreps_num_list)}
    
    return results


""" output """
output = f"""
* Point group *
{args.pg}

* Reducible representation *
"""
print(output)

rrep = {rotation: float(input(f"{rotation}: ")) for rotation in tuple(sym_op_with_mul)}
result = "+".join([f" {value}({key}) " for key, value in decompo(rrep).items() if value > 0])

print("\n")
print("* Result *")
print(result)
print("\n")
