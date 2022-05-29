import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import copy
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

def superellipse(xc, yc, a, b, rot, stiffness, num_points):
    ang = rot*(np.pi/180)
    t = np.linspace(0, 2*np.pi, 1000)
    u = b*(np.abs(np.cos(t))**(2/stiffness))*np.sign(np.cos(t))
    v = a*(np.abs(np.sin(t))**(2/stiffness))*np.sign(np.sin(t))
    x = xc+u*np.cos(ang)-v*np.sin(ang)
    y = yc+u*np.sin(ang)+v*np.cos(ang)
    return x, y

def bytscl(array, max = None , min = None , nan = 0, top = 255):
    if max is None: max = np.nanmax(array)
    if min is None: min = np.nanmin(array)
    return np.maximum(np.minimum(((top+0.9999)*(array-min)/(max-min)).astype(np.int16), top),0)

# LOAD DATA
zsun = 20.0
df = pd.read_pickle('Inputs/ALS_III_Symposium_v1.0.pkl')
df = df[df['Cat'] == 'M']
df = df.reset_index(drop = True)
df['x'] = ((df['ALS3_dist_P50']*np.cos(np.deg2rad(df['GLAT']))*np.cos(np.deg2rad(df['GLON'])))).values
df['y'] = ((df['ALS3_dist_P50']*np.cos(np.deg2rad(df['GLAT']))*np.sin(np.deg2rad(df['GLON'])))).values
df['z'] = ((df['ALS3_dist_P50']*np.sin(np.deg2rad(df['GLAT'])))+zsun).values
df = df[abs(df['z']) < 200]
x_vals = df['x'].values
y_vals = df['y'].values
z_vals = df['z'].values

ns = len(x_vals)
xl = [-4000, 4000]
yl = [-4000, 4000]
dd = 100
nx = int(round((xl[1]-xl[0])/dd, 0))
ny = int(round((yl[1]-yl[0])/dd, 0))
xx = xl[0]+(np.arange(nx)+0.5)*dd
yy = yl[0]+(np.arange(ny)+0.5)*dd
w, z  = np.zeros((nx, ny)), np.zeros((nx, ny))
for i in tqdm(range(nx), leave = False):
    for j in range(ny):
        d2 = ((x_vals-xx[i])**2+(y_vals-yy[j])**2)
        wij = np.exp(-0.5*d2/(dd**2))
        w[i, j] = sum(wij)
        z[i, j] = sum(z_vals*wij)/w[i, j]

w = np.transpose(w)
z = np.transpose(z)
zb = bytscl(z, min = -125, max = 125, top = 250)
zb = np.ma.masked_where(w < 3, zb)
z = np.ma.masked_where(w < 3, z)
ColorMap = copy(plt.cm.seismic)
ColorMap.set_bad('black')
plt.rcParams['text.usetex'] = True
plt.style.use('dark_background')
plt.figure(figsize = (12, 12))
im = plt.imshow(z, cmap = ColorMap, norm = TwoSlopeNorm(0), extent = [-4000, 4000, 4000, -4000], interpolation = 'gaussian')
plt.plot(0, 0, marker = '*', color = 'yellow', markeredgecolor = 'black', markersize = 14)
plt.gca().invert_yaxis()
plt.text(3850, 0, r'$l =\; 0^{\circ}$', ha = 'right', va = 'center', fontsize = 18)
plt.text(0, 3850, r'$l =\; 90^{\circ}$', ha = 'center', va = 'top', fontsize = 18)
plt.text(-3850, 0, r'$l =\; 180^{\circ}$', ha = 'left', va = 'center', fontsize = 18)
plt.text(0, -3850, r'$l =\; 270^{\circ}$', ha = 'center', va = 'bottom', fontsize = 18)
plt.xlabel(r'$\it{x}$ [pc]', fontsize = 18)
plt.ylabel(r'$\it{y}$ [pc]', fontsize = 18)
plt.minorticks_on()
plt.grid(which = 'major', linestyle = ':')
plt.grid(which = 'minor', linestyle = ':', alpha = 0.33)
plt.tick_params(axis = 'both', direction = 'in', which = 'major', length = 5, width = 0.7, labelsize = 16)
plt.tick_params(axis = 'both', direction = 'in', which = 'minor', length = 3, width = 0.7)
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes('right', size = '5%', pad = 0.05)
colbar = plt.gcf().colorbar(im, cax = cax, orientation = 'vertical')
colbar.ax.minorticks_on()
colbar.ax.tick_params(axis = 'both', direction = 'in', which = 'major', length = 5, width = 0.7, labelsize = 16)
colbar.ax.tick_params(axis = 'both', direction = 'in', which = 'minor', length = 3, width = 0.7)
colbar.set_label(r'$\bar{z}$ [pc]', fontsize = 18)
# plt.show()
# plt.savefig('Outputs/KDEmap.pdf', bbox_inches = 'tight', pad_inches = 0.2)
plt.savefig('Outputs/KDEmap.png', format = 'png', dpi = 300)
