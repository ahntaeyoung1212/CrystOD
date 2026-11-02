# -*- coding: utf-8 -*-
from pymatgen.ext.matproj import MPRester
import argparse
import os

parser = argparse.ArgumentParser(description="get all relevant POSCARs of system A B ... \
                                              POSCAR_[full chemical formula]_[id in materials project]")
parser.add_argument("--mpid", "-id", type=str, dest="mpid", help='id of materials project, e.g., mp-19007')

args_mpid = parser.parse_args().mpid


with MPRester() as mpr:
    compounds = mpr.get_entry_by_material_id(args_mpid, property_data=["structure", "full_formula", "spacegroup"])

#print(compounds)

#print(compounds, poscar)
poscar_name = "_".join(["POSCAR", compounds.data["full_formula"], args_mpid])
poscar = compounds.data["structure"].to(fmt="poscar").replace('\n', '  # ' + args_mpid + '\n', 1)

with open(poscar_name, 'w') as f:
    f.write(poscar)

#    if args_directory:
#        for material in compounds:
#            mp_id = material.entry_id
#            directory = "_".join([material.data["full_formula"], mp_id])
#            os.mkdir(directory)
#
#            poscar = material.data["structure"].to(fmt="poscar").replace('\n', '  # ' + mp_id + '\n', 1)
#
#            with open(directory + '//POSCAR', 'w') as f:
#                f.write(poscar)
#    else:
#        for material in compounds:
#            mp_id = material.entry_id
#            poscar_name = "_".join(["POSCAR", material.data["full_formula"], mp_id])
#
#            poscar = material.data["structure"].to(fmt="poscar").replace('\n', '  # ' + mp_id + '\n', 1)
#
#            with open(poscar_name, 'w') as f:
#                f.write(poscar)