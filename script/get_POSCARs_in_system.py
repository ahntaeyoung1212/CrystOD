# -*- coding: utf-8 -*-
from pymatgen.ext.matproj import MPRester
import argparse
import os

parser = argparse.ArgumentParser(description="get all relevant POSCARs of system A B ... \
                                              POSCAR_[full chemical formula]_[id in materials project]")
parser.add_argument("element", metavar='A', type=str, nargs='+', help='name of element, e.g., Cu')
parser.add_argument("--directory", "-d", action="store_true", help='output POSCAR at directories named as chemical '
                                                                   'formula')

args_element = parser.parse_args().element
args_directory = parser.parse_args().directory

#MP_API_KEY = os.environ.get("MP_API_KEY")

with MPRester() as mpr:
    compounds = mpr.get_entries_in_chemsys(args_element, property_data=["structure", "full_formula", "spacegroup"])
    #print(compounds)

    if args_directory:
        for material in compounds:
            mp_id = material.entry_id
            directory = "_".join([material.data["full_formula"], mp_id])
            os.mkdir(directory)

            poscar = material.data["structure"].to(fmt="poscar").replace('\n', '  # ' + mp_id + '\n', 1)

            with open(directory + '//POSCAR', 'w') as f:
                f.write(poscar)
    else:
        for material in compounds:
            mp_id = material.entry_id
            poscar_name = "_".join(["POSCAR", material.data["full_formula"], mp_id])

            poscar = material.data["structure"].to(fmt="poscar").replace('\n', '  # ' + mp_id + '\n', 1)

            with open(poscar_name, 'w') as f:
                f.write(poscar)
