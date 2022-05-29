import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from scipy.interpolate import interpn
from matplotlib.colors import Normalize

def density_scatter(x, y, sort = True, bins = 20, **kwargs):
    data, x_e, y_e = np.histogram2d(x, y, bins = bins, density = True)
    z = interpn((0.5*(x_e[1:]+x_e[:-1]), 0.5*(y_e[1:]+y_e[:-1])), data, np.vstack([x,y]).T, method = "splinef2d", bounds_error = False)
    z[np.where(np.isnan(z))] = 0.0
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
    plt.scatter(x, y, c = z, **kwargs)

# LOAD ALS II DATA
df = pd.read_pickle('Inputs/ALS_III_Symposium_v1.0.pkl')
df = df[df['Cat'].isin(['M'])]

hd = 31.8
zsun = 20
GalCenterDist = 8178
df['x'] = (df['ALS3_dist_P50']*np.cos(np.deg2rad(df['GLAT']))*np.cos(np.deg2rad(df['GLON'])))
df['y'] = (df['ALS3_dist_P50']*np.cos(np.deg2rad(df['GLAT']))*np.sin(np.deg2rad(df['GLON'])))
df['z'] = (df['ALS3_dist_P50']*np.sin(np.deg2rad(df['GLAT'])))
x = np.sqrt((df['x']-GalCenterDist)**2+df['y']**2).values
y = (df['z']+zsun).values
cols = np.array(['blue' if (l < 180) and (l >= 0) else 'red' for l in df['GLON'].values])

bin_size = 350
bin_limits = np.arange(5000, 11000+bin_size, bin_size)
bin_centers = bin_limits[:-1]+(bin_size/2)
y_P50_Blue, y_P16_Blue, y_P84_Blue, y_P02_Blue, y_P98_Blue = [], [], [], [], []
for i in range(len(bin_limits)-1):
    indx = np.where((x >= bin_limits[i]) & (x < bin_limits[i+1]) & (cols == 'blue'))[0]
    y_P50_Blue.append(np.percentile(y[indx], 50))
    y_P02_Blue.append(np.percentile(y[indx], 2.2750131948178987))
    y_P16_Blue.append(np.percentile(y[indx], 15.865525393145702))
    y_P84_Blue.append(np.percentile(y[indx], 84.1344746068543))
    y_P98_Blue.append(np.percentile(y[indx], 97.7249868051821))

y_P50_Red, y_P16_Red, y_P84_Red, y_P02_Red, y_P98_Red = [], [], [], [], []
for i in range(len(bin_limits)-1):
    indx = np.where((x >= bin_limits[i]) & (x < bin_limits[i+1]) & (cols == 'red'))[0]
    y_P50_Red.append(np.percentile(y[indx], 50))
    y_P02_Red.append(np.percentile(y[indx], 2.2750131948178987))
    y_P16_Red.append(np.percentile(y[indx], 15.865525393145702))
    y_P84_Red.append(np.percentile(y[indx], 84.1344746068543))
    y_P98_Red.append(np.percentile(y[indx], 97.7249868051821))

plt.rcParams['text.usetex'] = True
plt.style.use('dark_background')
plt.figure(figsize = (20, 12))
x_plot = x[np.where(cols == 'blue')[0]]
y_plot = y[np.where(cols == 'blue')[0]]
density_scatter(x_plot, y_plot, bins = [20, 20], s = 35, cmap = 'gist_heat', edgecolor = 'none')
plt.axhline(y = 0, linestyle = '-', color = 'white', linewidth = 2, path_effects = [pe.Stroke(linewidth = 1.5, foreground = 'w', alpha = 0.7), pe.Normal()])

color_I_II = 'blue'
color_III_IV = 'springgreen'

# plt.plot(bin_centers[2:-2], y_P02_Red[2:-2], linestyle = 'dotted', color = color_III_IV, linewidth = 2)
# plt.plot(bin_centers, y_P16_Red, linestyle = '--', color = color_III_IV, linewidth = 2, path_effects = [pe.Stroke(linewidth = 3, foreground = 'k'), pe.Normal()])
# plt.plot(bin_centers, y_P50_Red, linestyle = '-', color = color_III_IV, linewidth = 2, path_effects = [pe.Stroke(linewidth = 3, foreground = 'k'), pe.Normal()])
# plt.plot(bin_centers, y_P84_Red, linestyle = '--', color = color_III_IV, linewidth = 2, path_effects = [pe.Stroke(linewidth = 3, foreground = 'k'), pe.Normal()])
# plt.plot(bin_centers[2:-2], y_P98_Red[2:-2], linestyle = 'dotted', color = color_III_IV, linewidth = 2)
#
# plt.plot(bin_centers[2:-2], y_P02_Blue[2:-2], linestyle = 'dotted', color = color_I_II, linewidth = 2)
plt.plot(bin_centers, y_P16_Blue, linestyle = '--', color = color_I_II, linewidth = 2, path_effects = [pe.Stroke(linewidth = 3, foreground = 'k'), pe.Normal()])
plt.plot(bin_centers, y_P50_Blue, linestyle = '-', color = color_I_II, linewidth = 2, path_effects = [pe.Stroke(linewidth = 3, foreground = 'k'), pe.Normal()])
plt.plot(bin_centers, y_P84_Blue, linestyle = '--', color = color_I_II, linewidth = 2, path_effects = [pe.Stroke(linewidth = 3, foreground = 'k'), pe.Normal()])
# plt.plot(bin_centers[2:-2], y_P98_Blue[2:-2], linestyle = 'dotted', color = color_I_II, linewidth = 2)

plt.plot(GalCenterDist, zsun, marker = '*', color = 'yellow', markeredgecolor = 'black', markersize = 20)
# plt.legend(handles = [Line2D([0], [0], linestyle = '-', color = color_I_II, linewidth = 1.5, path_effects = [pe.Stroke(linewidth = 3, foreground = 'w', alpha = 0.5), pe.Normal()], label = r'Median $\it{z}$ for the I $\&$ II Quadrants'),
#                       Line2D([0], [0], linestyle = '-', color = color_III_IV, linewidth = 1.5, path_effects = [pe.Stroke(linewidth = 3, foreground = 'w', alpha = 0.5), pe.Normal()], label = r'Median $\it{z}$ for the III $\&$ IV Quadrants'),
#                       Line2D([0], [0], linestyle = '--', color = color_I_II, linewidth = 1.5, path_effects = [pe.Stroke(linewidth = 3, foreground = 'w', alpha = 0.5), pe.Normal()], label = r'Percentiles $15.9\;\% - 84.1\;\%$ for the I $\&$ II Quadrants'),
#                       Line2D([0], [0], linestyle = '--', color = color_III_IV, linewidth = 1.5, path_effects = [pe.Stroke(linewidth = 3, foreground = 'w', alpha = 0.5), pe.Normal()], label = r'Percentiles $15.9\;\% - 84.1\;\%$ for the III $\&$ IV Quadrants'),
#                       Line2D([0], [0], linestyle = 'dotted', color = color_I_II, linewidth = 2, label = r'Percentiles $2.3\;\% - 97.7\;\%$ for the I $\&$ II Quadrants'),
#                       Line2D([0], [0], linestyle = 'dotted', color = color_III_IV, linewidth = 2, label = r'Percentiles $2.3\;\% - 97.7\;\%$ for the III $\&$ IV Quadrants')],
#                       loc = 'lower right', bbox_to_anchor = (0.99, 0.01), fontsize = 14, labelspacing = 0.1)
plt.xlabel(r'$\sqrt{(d_{SgrA^*}-\it{x})^2+\it{y}^2}$ [pc]', fontsize = 18)
plt.ylabel(r'$\it{z}$ [pc]', fontsize = 18)
plt.minorticks_on()
plt.grid(which = 'major', linestyle = ':')
plt.grid(which = 'minor', linestyle = ':', alpha = 0.33)
plt.xlim(4500, 11500)
plt.ylim(-450, 450)
plt.tick_params(axis = 'both', direction = 'in', which = 'major', length = 5, width = 0.7, labelsize = 16)
plt.tick_params(axis = 'both', direction = 'in', which = 'minor', length = 3, width = 0.7)
# plt.show()
plt.savefig('Outputs/GalWarp_I_II_Quad_03.png', format = 'png', dpi = 300)
