import argparse
import os
from pymatgen.io.cif import CifParser
from pymatgen.io.vasp.inputs import Poscar

__author__ = "Ko Sato"
__copyright__ = "Copyright 2023, Nakajima group"
__version__ = "1.0"
__maintainer__ = "Ko Sato"
__email__ = "sato.k.co@m.titech.ac.jp"
__status__ = "Finished"
__date__ = "November 22, 2023"

### PARSER STRUCTURE ###
parser = argparse.ArgumentParser(description="This program converts CIF files into POSCAR files.")
parser.add_argument("--cif", "-c", dest="name_of_cif", default="None", type=str,
                    help="Input the path of the cif file."
                         "e.g. python cif2poscar.py --cif POSCAR.cif")
args = parser.parse_args()
cif_name = args.name_of_cif
poscar_name = os.path.splitext(cif_name)[0]
path_poscar = os.path.abspath(poscar_name)

def cif_to_poscar(cif_file, poscar_file):
    cif_parser = CifParser(cif_file)
    structure = cif_parser.get_structures()[0]

    poscar = Poscar(structure)
    poscar.write_file(poscar_file)

#cif_file_path = "/Users/ko/Downloads/test.cif"
#poscar_file_path = "/Users/ko/Downloads/POSCAR_test"

cif_to_poscar(cif_name, path_poscar)

