"""
__author__ = "Qijing Zheng" (see http://staff.ustc.edu.cn/~zqj/posts/Phonopy-Rutile-TiO2/)
__version__ = "1.0"
__maintainer__ = "Hiroki Koiso"
__email__ = "koiso.h.aa@m.titech.ac.jp" 
__status__ = "Development"
__date__ = "July 28, 2023"
"""

import numpy as np
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection
import argparse


### Parser structure ###
desc = """
This program plots the phonon band. 
Before using this program, you need to make band.yaml which include eigenvectors.
"""
parser = argparse.ArgumentParser(description=desc)
parser.add_argument("--yaml", "-yaml", dest="name_of_yaml", default="band.yaml", type=str,
                    help="Input a band.yaml file.")
args = parser.parse_args()


### centimeters in inches ###
cm = 1/2.54

### matplotlib setting ###
plt.rcParams['font.family'] = 'Arial'
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.direction"] = "in"

import yaml
try:
    from yaml import CLoader as Loader
except:
    from yaml import Loader

############################################################

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

    return (Bcell,
            np.array(dists),
            np.array(freqs),
            np.array(qpoints),
            data['segment_nqpoint'],
            labels, eigvec)


############################################################
# the phonon band
Bcell, D1, F1, Q1, B1, L1, E1 = read_ph_yaml(args.name_of_yaml)

assert E1, "PHONON EIGENVECTORs MUST NOT BE EMPTY!"
E1 = np.asarray(E1)
nqpts, nbnds, natoms, _, _ = E1.shape
# real and imaginary  part of the phonon polarization vector
preal, pimag = E1[..., 0], E1[..., 1]

RatioLT = np.zeros_like(F1)
for ii in range(nqpts):
    q = np.dot(Q1[ii], Bcell)
    # exclude Gamma point
    if np.linalg.norm(q) > 1E-10:
        RatioLT[ii,:] = np.linalg.norm(
            (preal[ii] + 1j * pimag[ii]) * q / np.linalg.norm(q),
            axis=(1, 2)
        )
    # exclude Gamma point
    else:
        RatioLT[ii,:] = 0.5


############################################################
fig = plt.figure(figsize=(11.07*cm, 8.31*cm),
                dpi=480,
                facecolor="w"
            )

ax = plt.subplot()

############################################################
norm = mpl.colors.Normalize(vmin=0, vmax=1)
s_m = mpl.cm.ScalarMappable(cmap='bwr', norm=norm)
s_m.set_array([RatioLT])

for ii in range(0, F1.shape[1]):
    ik = 0
    for nseg in B1:
        x = D1[ik:ik+nseg]
        y = F1[ik:ik+nseg, ii]
        z = RatioLT[ik:ik+nseg, ii]
        
        ax.plot(
            x, y,
            lw=1.0, color='k', alpha=0.6
        )
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        lc = LineCollection(segments,
                            colors=[s_m.to_rgba(ww)
                                    for ww in (z[1:] + z[:-1])/2.]
                            )
        lc.set_linewidth(1.0)
        ax.add_collection(lc)

        ik += nseg

for ii in np.cumsum(B1)[:-1]:
    ax.axvline(
        x=D1[ii], ls='dotted',
        color='black', alpha=0.8, lw=0.5
    )

divider = make_axes_locatable(ax)
ax_cbar = divider.append_axes('right', size='3%', pad=0.02)
cbar = plt.colorbar(s_m, cax=ax_cbar, ticks=[0, 1])
cbar.set_ticklabels(['T', 'L'])

ax.set_xlim(D1.min(), D1.max())
ax.set_xticks(D1[np.r_[[0], np.cumsum(B1)-1]])
if L1:
    ax.set_xticklabels(L1)
ax.set_ylabel('Frequency (THz)', labelpad=5)
ax.yaxis.set_minor_locator(AutoMinorLocator())
#ax.set_ylim(0, F1.max()+1)
ax.axhline(y=0,linestyle='--', color='black', lw=0.5)

############################################################
plt.tight_layout(pad=0.5)
#plt.show()
plt.savefig('phonon_band_LT_c.pdf')
