"""
__author__ = "Hiroki Koiso"
__copyright__ = "Copyright 2023, Nakajima group"
__version__ = "1.0"
__maintainer__ = "Hiroki Koiso"
__email__ = "koiso.h.aa@m.titech.ac.jp"
__status__ = "Development"
__date__ = "August 8, 2023"
"""

#import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from fractions import Fraction
import plotly.graph_objs as go
import numpy as np
import argparse


### Parser structure ###
desc = """
This program plots the first Brillouin zone of the supercell. 
"""
parser = argparse.ArgumentParser(description=desc)
parser.add_argument("--poscar", "-pos", dest="poscar", default="POSCAR", type=str,
                    help="Input the name of POSCAR file.")
parser.add_argument("--trans-matrix", "-trans-mat", dest="trans_mat", type=str, default=None, 
                    help="Input the transform trans_mat like -trans_mat=\"1 1 0   1 0 1   0 1 1\"")
parser.add_argument("--save", "-s", dest="save", default=None, action="store_true", 
                    help="Save plot data as html")
args = parser.parse_args()


def get_lattice_vector(poscar):
    """ Get lattice vector from POSCAR

    Parameter
    ----------
    poscar : str
        file path of POSCAR
    
    Return
    ----------
    lattice vector : ndarray, shape=(3,3), raw vectors [a1, a2, a3]^T
        lattice vector
        
    """
    with open(poscar, 'r') as f:
        for _ in range(1):
            f.readline()
        # Read lattice vector
        multiple_num = float(f.readline())
        lattice_vectors = []
        for _ in range(3):
            vector = f.readline().split()
            lattice_vectors.append([float(component) for component in vector])
        return np.array(lattice_vectors) * multiple_num


#### This function was made by Qijing Zheng. ###
### http://staff.ustc.edu.cn/~zqj/posts/howto-plot-brillouin-zone/ ###
def get_brillouin_zone_3d(cell):
    """ Generate the Brillouin Zone of a given cell.
    The BZ is the Wigner-Seitz cell of the reciprocal lattice, which can be constructed by Voronoi decompositionto the reciprocal lattice. 
    A Voronoi diagram is a subdivision of the space into the nearest neighborhoods of a given set of points. 

    https://en.wikipedia.org/wiki/Wigner%E2%80%93Seitz_cell
    https://docs.scipy.org/doc/scipy/reference/tutorial/spatial.html#voronoi-diagrams

    Parameters
    ----------
    cell : ndarray, shape=(3, 3), raw vectors [b1, b2, b3]^T
        reciprocal lattice vector  

    Returns
    ----------
    vertices : tuple
        vertices of BZ
    ridges : list of list
        ridges of BZ
    faces : list of list
        faces of BZ
    """
    cell = np.asarray(cell, dtype=float)
    assert cell.shape == (3, 3)

    px, py, pz = np.tensordot(cell, np.mgrid[-1:2, -1:2, -1:2], axes=[0, 0])
    points = np.c_[px.ravel(), py.ravel(), pz.ravel()]

    vor = Voronoi(points)

    bz_facets = []
    bz_ridges = []
    bz_vertices = []

    for pid, rid in zip(vor.ridge_points, vor.ridge_vertices):
        # WHY 13 ????
        # The Voronoi ridges/facets are perpendicular to the lines drawn between the
        # input points. The 14th input point is [0, 0, 0].
        if(pid[0] == 13 or pid[1] == 13):
            bz_ridges.append(vor.vertices[np.r_[rid, [rid[0]]]])
            bz_facets.append(vor.vertices[rid])
            bz_vertices += rid

    bz_vertices = list(set(bz_vertices))

    return vor.vertices[bz_vertices], bz_ridges, bz_facets

def split_list(l, n):
    """ split list to sublist 

    parameter
    ---------
    l: list
    n: element count of sublist

    Returns
    ----------
    sublists

    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


lat = get_lattice_vector(args.poscar)
# reciprocal lattice vector
rec_lat = np.linalg.inv(lat).T

# lattice and reciprocal lattice vector of supercell
trans_mat = np.array(list(split_list([float(Fraction(n)) for n in  args.trans_mat.split()], 3)))
super_lat = np.dot(trans_mat, lat)
rec_super_lat = np.linalg.inv(super_lat).T

# get the first Brillouin zone
v, e, f = get_brillouin_zone_3d(rec_lat)
s_v, s_e, s_f = get_brillouin_zone_3d(rec_super_lat)


""" plot """
fig = go.Figure()

# Base
basis_labs = ['<i>b<sub>1</sub></i>', '<i>b<sub>2</sub></i>', '<i>b<sub>3</sub></i>']
for ii, basis in enumerate(rec_lat):
    bx, by, bz = basis
    fig.add_trace(
            go.Scatter3d(
                x=[0, bx],
                y=[0, by],
                z=[0, bz],
                opacity=0.8,
                hoverinfo='skip',
                mode='lines+text',
                line=dict(
                    color='black',
                    width=3,
                    dash='dot'
                ),
                text=['', basis_labs[ii]],
                textfont=dict(size=20),
            )
        )
# Base (supercell)
basis_clrs = ['red', 'green', 'blue']
basis_labs = ['<i>b<sub>1</sub></i>', '<i>b<sub>2</sub></i>', '<i>b<sub>3</sub></i>']
for ii, basis in enumerate(rec_super_lat):
    bx, by, bz = basis
    fig.add_trace(
            go.Scatter3d(
                x=[0, bx],
                y=[0, by],
                z=[0, bz],
                opacity=0.8,
                hoverinfo='skip',
                mode='lines+text',
                line=dict(
                    color=basis_clrs[ii],
                    width=6,
                ),
                text=['', basis_labs[ii]],
                textfont=dict(color=basis_clrs[ii], size=25),
            )
        )

# Edige
for edge in e:
    fig.add_trace(
            go.Scatter3d(
                x=edge[:, 0],
                y=edge[:, 1],
                z=edge[:, 2],
                opacity=0.8,
                hoverinfo='skip',
                mode='lines',
                line=dict(
                    color='black',
                    width=3,
                    dash='dot'
                ),
            ),
    )
# Edige (supercell)
#orign_shift = np.dot(np.array([[0, 0, 0], [0, 1, 0]]), rec_super_lat)
orign_shift = np.dot(np.array([[0, 0, 0], [0, 1, 0], [0, 1, 1]]), rec_super_lat)
for shift in orign_shift:
    sx, sy, sz = shift
    for edge_t in s_e:
        fig.add_trace(
                go.Scatter3d(
                    x=edge_t[:, 0]+sx,
                    y=edge_t[:, 1]+sy,
                    z=edge_t[:, 2]+sz,
                    opacity=0.8,
                    hoverinfo='skip',
                    mode='lines',
                    line=dict(
                        color='red',
                        width=3,
                    ),
                ),
        )

# Faces (supercell)
#for face in s_f:
#    fig.add_trace(
#            go.Mesh3d(
#                x=face[:, 0],
#                y=face[:, 1],
#                z=face[:, 2],
#                hoverinfo='skip',
#                opacity=0.3,
#                color='gray',
#                alphahull=-1,
#                ),
#        )


fig.update_layout(
    showlegend=False,
    scene=dict(
        xaxis_visible=False, 
        yaxis_visible=False, 
        zaxis_visible=False), 
    )

fig.show()
if args.save:
    fig.write_html("BZ.html")


# for matplotlib
#
#""" plot """
#fig = plt.figure()
#ax = plt.subplot(111, projection='3d')
#ax.axis("off")
#
## plot the basis
#color_list = ['red', 'green', 'blue']
#for i, (vec, color) in enumerate(zip(rec_lat, color_list)):
#    ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color=color, arrow_length_ratio=0.1)
#    ax.text(vec[0], vec[1], vec[2], f'b{i+1}', fontsize=12)
#
# plot lattice vector
#norms = np.linalg.norm(lat, axis=1)
#unit_vec_lat = lat / norms[:, np.newaxis] / 5
#for i, (vec, color) in enumerate(zip(unit_vec_lat, color_list)):
#    ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color=color, arrow_length_ratio=0.1)
#    ax.text(vec[0], vec[1], vec[2], f'a{i+1}', fontsize=12)
#
#
## plot the first Brillouin zone
#for xx in e:
#    ax.plot(xx[:, 0], xx[:, 1], xx[:, 2], color='k', lw=1.0)
#
##v_rc = np.dot(v, np.linalg.inv(rec_lat))
##for i, qpoint in enumerate(v):
##    ax.text(qpoint[0]+0.005, qpoint[1]+0.005, qpoint[2]+0.005, f'({v_rc[i, 0]: .3f}, {v_rc[i, 1]: .3f}, {v_rc[i, 2]: .3f})', fontsize=7)
#
#
#plt.show()
