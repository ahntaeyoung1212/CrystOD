import math
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os.path
# from matplotlib.backends.backend_pdf import PdfPages
# from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.analysis.diffraction.core import AbstractDiffractionPatternCalculator
from pymatgen.core.composition import Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

__author__ = "Yasuhide Mochizuki"
__copyright__ = "Copyright 2026, Mochizuki group"
__version__ = "1.0"
__maintainer__ = "Yasuhide Mochizuki"
__email__ = "ahntaeyoung1212@gmail.com"
__status__ = "Development"
__date__ = "Mar 16, 2026"

AbstractDiffractionPatternCalculator.SCALED_INTENSITY_TOL = 0  # Monkey Patch

parser = argparse.ArgumentParser(description="This program derives the X-ray diffraction patterns of the input structure.")
parser.add_argument("--poscar", "-p", dest="poscar_path", default="POSCAR", type=str,
                    help="Input POSCAR file name.")
parser.add_argument("--xraytype", "-xrt", dest="xray_type", default="CuKa", type=str,
                    help="Specify the X-ray type in one of those: "
                         "CuKa (mixture of CuKa1 and CuKa2 in 2:1 ratio), "
                         "CuKa1 (1.5405929), CuKa2 (1.5444274), CuKb (1.392234), "
                         "AgKa1 (0.55942178), AgKa2 (0.5638131), "
                         "MoKa1 (0.70931715), MoKa2 (0.713607), "
                         "CoKa1 (1.788996), CoKa2 (1.792835), "
                         "FeKa1 (1.936041), FeKa2 (1.939973), "
                         "CrKa1 (2.289726), and CrKa2 (2.293651)."
                         "These wavelength values are extracted from RIETAN-FP_manual.")
parser.add_argument("--show", "-s", dest="show_figure", default="T", type=str,
                    help="If you want me to show the XRD figure through matplotlib, "
                         "command me like python3 xrd_pattern_poscar.py --show T or -s T")
parser.add_argument("--peak_profile", "-pp", dest="peak_profile", default="lorentzian", type=str,
                    help="Specify the peak profile function: 'gaussian' or 'lorentzian'. Default is 'lorentzian'.")
#parser.add_argument("--reference_xrd", "-rxrd", dest="ref_xrd", default="XRD.txt", type=str, help="Input text file name.")
args = parser.parse_args()


wavelength_list = {"CuKa1": 1.5405929,
                   "CuKa2": 1.5444274,
                   "CuKb": 1.392234,
                   "AgKa1": 0.55942178,
                   "AgKa2": 0.5638131,
                   "MoKa1": 0.70931715,
                   "MoKa2": 0.713607,
                   "CoKa1": 1.788996,
                   "CoKa2": 1.792835,
                   "FeKa1": 1.936041,
                   "FeKa2": 1.939973,
                   "CrKa1": 2.289726,
                   "CrKa2": 2.293651}
# These values are extracted from RIETAN-FP_manual.

poscar = Poscar.from_file(args.poscar_path, read_velocities=False)
s = poscar.structure
name_poscar = os.path.basename(args.poscar_path) # It will be used in naming the figure.

if args.xray_type == "CuKa":
    calculator1 = XRDCalculator(wavelength=wavelength_list["CuKa1"])
    pattern1 = calculator1.get_pattern(s, two_theta_range=(0, 120))
    calculator2 = XRDCalculator(wavelength=wavelength_list["CuKa2"])
    pattern2 = calculator2.get_pattern(s, two_theta_range=(0, 120))

    pattern2.y *= 0.5  # Intensity of CuKa2 is half of CuKa1

    x = np.concatenate((pattern1.x, pattern2.x))
    y = np.concatenate((pattern1.y, pattern2.y))
    hkls = pattern1.hkls + pattern2.hkls
    d_hkls = np.concatenate((pattern1.d_hkls, pattern2.d_hkls))

    sorted_indices = np.argsort(x)
    x = x[sorted_indices]
    y = y[sorted_indices]
    hkls = [hkls[i] for i in sorted_indices]
    d_hkls = d_hkls[sorted_indices]

    from collections import namedtuple
    CombinedPattern = namedtuple('CombinedPattern', ['x', 'y', 'hkls', 'd_hkls'])
    pattern = CombinedPattern(x, y, hkls, d_hkls)
else:
    input_wavelength = wavelength_list[args.xray_type]
    calculator = XRDCalculator(wavelength=input_wavelength)
    pattern = calculator.get_pattern(s, two_theta_range=(0, 120))

xrd_data_set = []

for number_of_peaks in range(len(pattern.x)):
    index_h = pattern.hkls[number_of_peaks][0]['hkl'][0]
    index_k = pattern.hkls[number_of_peaks][0]['hkl'][1]
    index_l = pattern.hkls[number_of_peaks][0]['hkl'][2]
    multi = pattern.hkls[number_of_peaks][0]['multiplicity']
    distance = pattern.d_hkls[number_of_peaks]
    two_theta = pattern.x[number_of_peaks]
    intensity = pattern.y[number_of_peaks]
    xrd_data_set.append([index_h, index_k, index_l, multi, distance, two_theta, intensity])


comp = Composition(s.formula)
reduced_formula = comp.reduced_formula
space_group = SpacegroupAnalyzer(s).get_space_group_symbol()

filename = "{}_{}_{}.txt"
filename_png = "{}_{}.png"
filename_pdf = "{}_{}.pdf"
if args.xray_type == "CuKa":
    header = "h, k, l, multiplicity, d, two theta, intensity, calculated with CuKa1 and CuKa2 (2:1 ratio)"
    np.savetxt(filename.format(args.poscar_path, reduced_formula, args.xray_type), xrd_data_set,
               fmt=["%d", "%d", "%d", "%d", "%s", "%s", "%s"], delimiter=',',
               header=header)
else:
    header_contents = "h, k, l, multiplicity, d, two theta, intensity, calculated with the wavelength of {} {} Angstrom"
    np.savetxt(filename.format(args.poscar_path, reduced_formula, args.xray_type), xrd_data_set,
               fmt=["%d", "%d", "%d", "%d", "%s", "%s", "%s"], delimiter=',',
               header=header_contents.format(args.xray_type, input_wavelength))

#np.savetxt(sys.stdout.buffer,xrd_data_set)



def gaussian_smearing_function(x, peak_position):
    SIGMA = 0.1
    INTENSITY = 1
    return INTENSITY * math.exp(-(x-peak_position)**2/(math.sqrt(2)*SIGMA**2)) / (math.sqrt(2*math.pi)*SIGMA)

def lorentzian_smearing_function(x, peak_position):
    SIGMA = 0.1
    INTENSITY = 1
    return INTENSITY * SIGMA / ((x - peak_position)**2 + SIGMA**2)

def smeared_xrd(x):
    val = 0
    smearing_func = None
    if args.peak_profile == 'gaussian':
        smearing_func = gaussian_smearing_function
    elif args.peak_profile == 'lorentzian':
        smearing_func = lorentzian_smearing_function

    if smearing_func:
        for peak_position, peak_intensity in zip(pattern.x, pattern.y):
            val += peak_intensity * smearing_func(x, peak_position)
    return val

x_plot = list(np.linspace(0, 120, 5000))
y_plot = np.array([smeared_xrd(theta) for theta in x_plot])

# ref_data = np.loadtxt(args.ref_xrd)
# theta_2 = []
# intensity_2 = []
# file = open(args.ref_xrd)
# for loop in range(len(file.readlines())):
#     theta_2.append(ref_data[loop][0])
#     intensity_2.append(ref_data[loop][1] / 5)
#
# x2_plot = theta_2
# y2_plot = intensity_2

xrd_figure = plt.figure()
# plt.plot(x2_plot, y2_plot, linestyle = "solid", linewidth = "0.5", color = "red",label = "Experimental Data")
plt.plot(x_plot, y_plot, linestyle="solid", linewidth="0.75", color="red", label=name_poscar)
plt.vlines(pattern.x, -max(y_plot) / 5, -max(y_plot) / 10, linestyle='solid', color="red", linewidth=0.5)

plt.xlim(10, 120)
# plt.ylim(0, 30)
plt.xlabel("2$\\theta$ (deg.)", fontsize=12)
plt.ylabel("Intensity", labelpad=20, fontsize=12)
plt.tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)



plt.legend(fontsize=12)         #fontsize for the figure
plt.tick_params(labelsize=12)   #fontsize for the axis numbers

if args.show_figure == "T":
    plt.show()
elif args.show_figure == "F":
    print("Note: you did not command to show the figure.")

xrd_figure.savefig(f"XRD_{name_poscar}.pdf")
# plt.savefig(filename_png.format(args.poscar_path,reduced_formula))

# pdf = PdfPages(filename_pdf.format(args.poscar_path,reduced_formula))
# pdf.savefig(xrd_figure)
# pdf.close()
