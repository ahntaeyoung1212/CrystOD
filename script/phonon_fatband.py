"""
__author__ = "Hiroki Koiso"
__version__ = "2.0"
__maintainer__ = "Hiroki Koiso"
__email__ = "koiso.h.aa@m.titech.ac.jp" 
__status__ = "Development"
__date__ = "January 4, 2024"
"""

import numpy as np
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
from argparse import ArgumentParser, RawTextHelpFormatter, RawDescriptionHelpFormatter, ArgumentDefaultsHelpFormatter


### Parser structure ###
class MyHelpFormatter(RawTextHelpFormatter, RawDescriptionHelpFormatter, ArgumentDefaultsHelpFormatter):
    pass
desc = """
This program plots the phonon fatband. 
Before using this program, you need to make band.yaml including eigenvectors.
"""
parser = ArgumentParser(description=desc, formatter_class=MyHelpFormatter)
parser.add_argument("--yaml", "-yaml", dest="name_of_yaml", default="band.yaml", type=str,
                    help="Input a band.yaml file.")
parser.add_argument("--element", "-element", dest="element", required=True, type=str, default=None,
                    help="Element")
parser.add_argument("--projection-direction", "-pd", dest="direction", type=str, default=None,
                    help="Projection direction is specified in reduced coordinates, i.e., with respect to a, b, c axes.")
parser.add_argument("--color", "-c", dest="color", type=str, default='blue', 
                    help="Input the color of dots you want. Default is blue.")
parser.add_argument("--save", "-s", dest="save", action="store_true", 
                    help="Save plot data as png")
args = parser.parse_args()

### ------------------------------ ###

import yaml
try:
    from yaml import CLoader as Loader
except:
    from yaml import Loader

############################################################

# see http://staff.ustc.edu.cn/~zqj/posts/Phonopy-Rutile-TiO2/
def read_ph_yaml(filename):
    with open(filename, 'r') as f:
        data = yaml.load(f, Loader=Loader)

    freqs   = []
    dists   = []
    qpoints = []
    labels  = []
    eigvec  = []
    Acell   = np.array(data['lattice'])
    Bcell   = np.array(data['reciprocal_lattice'])
    chemical_symbol = [symbol['symbol'] for symbol in data['points']]

    for v in data['phonon']:
        if 'label' in v:
            labels.append(v['label'])
        else:
            labels.append(None)
        freqs.append([f['frequency'] for f in v['band']])
        if 'eigenvector' in v['band'][0]:
            eigvec.append([np.array(f['eigenvector']) for f in v['band']])
        qpoints.append(v['q-position'])
        dists.append(v['distance'])

    if all(x is None for x in labels):
        if 'labels' in data:
            ss = np.array(data['labels'])
            ss = np.where((ss == 'GM') | (ss == 'Gamma'), '$\Gamma$', ss)
            labels = list(ss[0])
            for ii, f in enumerate(ss[:-1,1] == ss[1:,0]):
                if not f:
                    labels[-1] += r'|' + ss[ii+1, 0]
                labels.append(ss[ii+1, 1])
        else:
            labels = []

    return (Acell,
            Bcell,
            np.array(dists),
            np.array(freqs),
            np.array(qpoints),
            data['segment_nqpoint'],
            labels, eigvec, chemical_symbol)

############################################################
# phonon band
Acell, Bcell, D1, F1, Q1, B1, L1, E1, chemical_symbols = read_ph_yaml(args.name_of_yaml)

# calculate pdos
def get_pdos_each_kpoint(eigenvector, projection=False) -> np.ndarray:
    """ calculate the pdos for each phonon modes """
    num_kpt = len(eigenvector)
    num_index = len(eigenvector[0])
    num_atom = eigenvector[0][0].shape[0]

    pdos_each_kpt = np.zeros([num_kpt, num_index, num_atom])
    for ikpt in range(num_kpt):
        for index in range(num_index):
            for iatom in range(num_atom):
                atom_eigenvector = eigenvector[ikpt][index][iatom]
                atom_eigenvector = atom_eigenvector[...,0] + atom_eigenvector[...,1]*1j
                if projection:
                    atom_eigenvector = atom_eigenvector * direction
                pdos = np.dot(atom_eigenvector.conjugate(), atom_eigenvector)
                pdos_each_kpt[ikpt, index, iatom] = np.real(pdos)
    return pdos_each_kpt

if not args.direction == None:
    projection = True
    fractional_direction = np.array(args.direction.split(), dtype=float)
    cartesian_direction = np.dot(fractional_direction, Acell)
    direction = cartesian_direction / np.linalg.norm(cartesian_direction)
else:
    projection = False
pdos_each_kpoint = get_pdos_each_kpoint(E1, projection)

# get the pdos with target element
if args.element in chemical_symbols:
    target_element_positions = [i for i, symbol in enumerate(chemical_symbols) if symbol == args.element]
else:
    print(f"ERROR : {args.element} is not in this compound !!")
    exit(0)
element_strat, element_end = target_element_positions[0], target_element_positions[-1]+1
target_pdos = np.sum(pdos_each_kpoint[:, :, element_strat:element_end], axis=2)

############################################################

# centimeters in inches
cm = 1/2.54

# matplotlib setting
plt.rcParams['font.family'] = 'Arial'
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.direction"] = "in"

# figure size (cm)
fig_width = 5.00
fig_hight = 4.00

# plot_line_width
plot_line_width = 0.5

# fontsize
label_fontsize = 10
tick_fontsize = 9

# axis width
axis=['top','bottom','left','right']
line_width=[0.5, 0.5, 0.5, 0.5]

# set figure
fig = plt.figure(figsize=(fig_width*cm, fig_hight*cm),
                dpi=480,
                facecolor="w"
            )

ax = plt.subplot()

############################################################

# title
ax.set_title(args.element)

# x-axis range
ax.set_xlim(D1.min(), D1.max())

# x-axis ticks
ax.set_xticks(D1[np.r_[[0], np.cumsum(B1)-1]])
ax.yaxis.set_minor_locator(AutoMinorLocator())
if L1:
    ax.set_xticklabels(L1)

ax.tick_params(labelsize=tick_fontsize, width=0.5)

# label
ax.set_ylabel('Frequency (THz)', labelpad=5, fontsize=label_fontsize)
# ax.set_ylim(0, F1.max()+1)
ax.axhline(y=0,linestyle='--', color='black', lw=0.5)

# axis width
for a,w in zip(axis, line_width):  # change axis width
    ax.spines[a].set_linewidth(w)

for ii in range(0, F1.shape[1]):
    ik = 0
    for nseg in B1:
        x = D1[ik:ik+nseg]
        y = F1[ik:ik+nseg, ii]
        #s = pdos_each_kpoint[ik:ik+nseg, ii, 0]
        s = target_pdos[ik:ik+nseg, ii]
        
        ax.plot(
            x, y,
            lw=plot_line_width, color='black'
        )
        ax.scatter(
            x, y, 
            s=s*10/len(target_element_positions), color=args.color, alpha=0.1
        )
        ik += nseg

for ii in np.cumsum(B1)[:-1]:
    ax.axvline(
        x=D1[ii], ls='dotted',
        color='black', lw=0.5
    )

############################################################
plt.tight_layout(pad=0.5)
if args.save:
    plt.savefig(f'phonon_fatband_{args.element}.png')
plt.show()
