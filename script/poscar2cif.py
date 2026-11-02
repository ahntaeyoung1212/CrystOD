__author__ = "Yasuhide Mochizuki"
__copyright__ = "Copyright 2023, Nakajima group"
__version__ = "1.2"
__maintainer__ = "Yasuhide Mochizuki"
__email__ = "mochizuki.y.af@m.titech.ac.jp"
__status__ = "Finished"
__date__ = "Nov 9, 2023"


import argparse
from pymatgen.io.cif import CifWriter
from pymatgen.core.structure import Structure

### PARSER STRUCTURE ###
parser = argparse.ArgumentParser(description="This program converts POSCAR files into CIF files.")
parser.add_argument("--poscar", "-p", dest="name_of_poscar", default="POSCAR", type=str,
                    help="Input the path of the poscar file."
                         "e.g. python poscar2cif.py --poscar POSCAR")
parser.add_argument("--tolerance", "-t", dest="deg_of_tolerance", default=0.001, type=float,
                    help="Input float number for the consideration of symmetrical tolerance.")
args = parser.parse_args()


poscar_name = args.name_of_poscar
tol = args.deg_of_tolerance


### CHECK THE NAME ###
if poscar_name == "POSCAR":
    print(f"I will try to convert {poscar_name} to a CIF file as {poscar_name}.cif; "
          f"I strongly suggest you to detect which POSCAR you want to convert.")
else:
    print(f"I will convert {poscar_name} to a CIF file as {poscar_name}.cif")


### POSCAR TO CIF ###
s = Structure.from_file(poscar_name)
cif_data = CifWriter(s, symprec=tol)


### WRITE CIF ###
cif_data.write_file(f"{poscar_name}_pmg.cif")

### MODIFY CIF FILE ###
#start_number = 1
#input_filename = f"{poscar_name}.cif"
#output_filename = f"{poscar_name}_test.cif"
#
#def update_label(line, number):
#    parts = line.split()
#    if len(parts) == 2:
#        label = parts[1]
#        new_label = label[:-1] + str(number)
#        return parts[0] + "  " + new_label + "\n"
#    return line
#
#with open(input_filename, "r") as infile, open(output_filename, "w") as outfile:
#    for line in infile:
#        updated_line = update_label(line, start_number)
#        outfile.write(updated_line)
#        start_number += 1

