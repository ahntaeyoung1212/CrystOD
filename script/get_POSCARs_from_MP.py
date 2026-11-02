"""
__author__ = "Hiroki Koiso, Yasuhide Mochizuki"
__email__ = "akabane0308koiso@gmail.com"
__status__ = "Development"
__date__ = "January 22, 2024"
"""

from argparse import ArgumentParser, RawTextHelpFormatter, RawDescriptionHelpFormatter, ArgumentDefaultsHelpFormatter
import os, sys
import re

### EXTRACT API KEY FROM .pmgrc.yaml ###
home_directory = os.path.expanduser("~")
pmgrc = os.path.join(home_directory, ".pmgrc_new.yaml")

with open(pmgrc, "r") as f:
    content_pmgrc = f.read()

match = re.search(r"PMG_MAPI_KEY:\s*(\S+)", content_pmgrc)

if match:
    API_KEY = match.group(1)
else:
    print("ERROR: ENTER HERE YOUR API KEY!")
    API_KEY = "YOUR_API_KEY"


### Parser structure ###
class MyHelpFormatter(RawTextHelpFormatter, RawDescriptionHelpFormatter, ArgumentDefaultsHelpFormatter):
    pass
desc = """
This script extracts POSCARs from Materials Project.
"""
parser = ArgumentParser(description=desc, formatter_class=MyHelpFormatter)
parser.add_argument("--formula", "-f", dest="formula", type=str, default=None,
                    help="A formula including anonymized formula (e.g., Fe2O3, ABO3, Si*). ")
parser.add_argument("--chemical-system", "-chemsys", dest="chemsys", type=str, default=None,
                    help="A chemical system (e.g., Li-Fe-O, Si-*).")
parser.add_argument("--exp-obsterved", "-exp-obs", dest="exp_obs", action="store_false", default=None,
                    help="Whether the material is experimentally obsearved.")
parser.add_argument("--is-metal", "-metal", dest="is_metal", action="store_true", default=None,
                    help="Whether the material is considered a metal.")
parser.add_argument("--is-stable", "-stable", dest="is_stable", action="store_true", default=None,
                    help="Whether the material lies on the convex energy hull.")
parser.add_argument("--spg-symbol", "-spg-sym", dest="spg_symbol", type=str, default=None,
                    help="Space group symbol of the material in international short symbol notation.")
parser.add_argument("--spg-number", "-spg-num", dest="spg_num", type=int, default=None,
                    help="Space group number of material.")
parser.add_argument("--num-sites", "-num-sites", dest="num_sites", nargs="*", type=int, default=None,
                    help="Minimum and maximum number of sites to consider. Input as --num-sites (Min num) (Max num).")
parser.add_argument("--exclude-elements", "-exc-elem", dest="exclude_elements", nargs="*", type=str, default=None,
                    help="Elements to exclude. Input as --exclude-elements (Element 1) (Element 2) ... .")
args = parser.parse_args()

# Whether the query is correct
if (args.formula is None) and (args.chemsys is None):
    print("ERROR!!\nDid not receive the query of the chemical composition. "
          "Please specify eather --formula or --chemical-system.")
    sys.exit(1)
elif (args.formula is not None) and (args.chemsys is not None):
    print("ERROR!!\nReceived two different queries of the chemical composition. "
          "Please specify eather --formula or --chemical-system.")
    sys.exit(1)


from mp_api.client import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp import Poscar

# Extract the information of the materials in Materials Project using API
with MPRester(API_KEY) as m:
    data = m.materials.summary.search(
        formula=args.formula, 
        chemsys=args.chemsys, 
        theoretical=args.exp_obs, 
        is_metal=args.is_metal, 
        is_stable=args.is_stable, 
        spacegroup_number=args.spg_num, 
        spacegroup_symbol=args.spg_symbol, 
        num_sites=args.num_sites, 
        exclude_elements=args.exclude_elements
        )

# Make directory
if args.formula:
    dir_name = args.formula
else:
    dir_name = args.chemsys

main_path = os.path.abspath(f'{dir_name}'.replace('*', 'x'))
os.makedirs(main_path)
os.chdir(main_path)

for d in data:
    # Find symmetry
    symmetry = SpacegroupAnalyzer(d.structure, symprec=0.1)
    spg_symbol = str(symmetry.get_space_group_symbol()).replace('/', '').replace('_', '')
    
    # Get primitive cell
    p_cell = symmetry.find_primitive()
    poscar = Poscar(p_cell)
    
    # Write POSCAR
    poscar_name = f"POSCAR_{d.formula_pretty}_{spg_symbol}_{str(d.material_id)}"
    poscar.write_file(poscar_name)
