"""
__author__ = "Hiroki Koiso"
__copyright__ = "Copyright 2023, Nakajima group"
__version__ = "1.1"
__maintainer__ = "Hiroki Koiso, Yasuhide Mochizuki"
__email__ = "koiso.h.aa@m.titech.ac.jp, mochizuki.y.af@m.titech.ac.jp"
__status__ = "Development"
__date__ = "October 23, 2023"
"""

from phonopy.phonon.character_table import character_table as all_character_tables
from argparse import ArgumentParser, RawTextHelpFormatter, RawDescriptionHelpFormatter, ArgumentDefaultsHelpFormatter
import numpy as np
import re

### PARSER STRUCTURE ###
class MyHelpFormatter(RawTextHelpFormatter, RawDescriptionHelpFormatter, ArgumentDefaultsHelpFormatter):
    pass
desc = """
This program calculates the ligand field split in given point symmetric field.

# Input:
1. Point group
2. Orbital

# Example:
python3 irreps_dirpro.py --point-group="m-3m" --orbital="d"
"""
parser = ArgumentParser(description =desc, formatter_class=MyHelpFormatter)
parser.add_argument("--point-group", "-pg", dest="pg", required=True, type=str,
                    help="Point group")
parser.add_argument("--orbital", "-orbital", dest="orbital", required=True, type=str, default=None,
                    help="Input the orbital which you want to know how to split at given point group field. You can choose from following; s, p, d, f, g, h, i")
args = parser.parse_args()
### --------------- ###


""" get character table """
try:
    ct = all_character_tables[args.pg][0]
except:
    print(f"ERROR: \"{args.pg}\" is not in point groups.")
    print("You need to choose from the following point groups: ")
    print(list(all_character_tables.keys()))
    exit(0)


def get_rrep_orbital(orbital):
    """ Calculate the reducible representation of the orbital in given point symmetric field.

    Parameter
    ----------
    irreps : list
        list of irreps that you want to calculate the direct product
    
    Return
    ----------
    result : dict, key = rotaion: value = character
        
    """
    # azimuthal number (l)
    orbital_azimuthal_num = {"s" : 0, "p" : 1, "d" : 2, "f" : 3, "g" : 4, "h" : 5, "i" : 6}
    
    # calculate characters
    characters = []
    l = orbital_azimuthal_num[orbital]
    for rotation in ct["rotation_list"]:
        if rotation == "E": characters.append(2*l+1)
        elif rotation == "i": characters.append((-1)**l * (2*l+1))
        elif "sg" in rotation: 
            characters.append(((-1)**l) * np.sin((l + 1/2) * np.pi) / np.sin(np.pi / 2))
        else: 
            rotation_num = [float(i) for i in re.findall(r'\d', rotation)]
            
            # get number of rotation. 
            n = rotation_num[0]
            index = 1
            while index < len(rotation_num):
                n /= rotation_num[index]
                index += 1
            alpha = 2 * np.pi / n

            if "C" in rotation:
                characters.append(np.sin((l + 1/2) * alpha) / np.sin(alpha / 2))
            elif "S" in rotation:
                characters.append(np.cos((l + 1/2) * alpha) / np.cos(alpha / 2))
    characters = [round(character) for character in characters]
    
    character_result = {rotation: character for rotation, character in zip(ct["rotation_list"], characters)}
    return character_result


def decompo(rrep):
    """ Decomposite reducible representation.

    Parameter
    ----------
    rrep : list
        Characters that you want to decomposite.
    
    Return
    ----------
    results : dict, key=irrep: value=(number of irrep)
        
    """
    
    all_irreps = list(ct['character_table'].keys())

    mul = np.array([[np.array(i).shape[0] for i in ct['mapping_table'].values()]])

    irreps_num_list = []
    for character in ct['character_table'].values():
        character = np.array([character])
        irreps_num = int(float(np.dot(mul * character, np.array(list(rrep.values())).T) / mul.sum()))
        irreps_num_list.append(irreps_num)

    results = {irrep: num for irrep, num in zip(all_irreps, irreps_num_list)}
    
    return results


rrep = get_rrep_orbital(args.orbital)

""" output """
result = "+".join([f" {value}({key}) " for key, value in decompo(rrep).items() if value > 0])
output = f"""
* Point group *
{args.pg}

* Orbital *
{args.orbital}

* Reducible representation of {args.orbital} orbital in {args.pg} point symmetric field *
"""
print(output)
for rotation, character in rrep.items():
    print(f"{rotation}: {character}")
print("\n")

print("* Result *")
print(result)
print("\n")