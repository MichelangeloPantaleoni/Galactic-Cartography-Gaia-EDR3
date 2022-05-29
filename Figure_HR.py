import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# LOAD DATA
df = pd.read_pickle('Inputs/ALS_III_Symposium_v1.0.pkl')
df = df[df['Cat'].isin(['M', 'I', 'E', 'L', 'H'])]
data_zams = pd.read_csv('Inputs/ZAMS_track.txt', delim_whitespace = True)
data_extc = pd.read_csv('Inputs/Extinction_tracks.txt', delim_whitespace = True)
df_zams = data_zams[data_zams.columns[[1, 0]]].copy()
df_10kK = data_extc[data_extc.columns[[2, 1]]].copy()
df_20kK = data_extc[data_extc.columns[[4, 3]]].copy()
df_10kK.rename(columns = {'G_10':'G', '(B-R)_10':'B-R'}, inplace = True)
df_20kK.rename(columns = {'G_20':'G', '(B-R)_20':'B-R'}, inplace = True)
x_zams = df_zams['B-R'].values[::-1]
y_zams = df_zams['G'].values[::-1]
x_10kK = df_10kK['B-R'].values
y_10kK = df_10kK['G'].values
x_20kK = df_20kK['B-R'].values
y_20kK = df_20kK['G'].values
df_marks = data_extc[data_extc[data_extc.columns[0]]%1 == 0]

# CREATE PLOT
catalogs = ['M', 'I', 'L', 'H', 'E']
catalog_labels = ['M - Likely massive stars', 'I - High/intermediate-mass stars', 'L - Intermediate/low-mass stars', 'H - High-gravity stars', 'E - Extragalactic stars']
colors = ['blueviolet', 'royalblue', 'darkorange', 'crimson', 'white']
# colors = ['lightsteelblue','lightsteelblue','lightsteelblue','lightsteelblue','lightsteelblue']
sizes = [7.0, 7.0, 7.0, 8.0, 55]
markers = ['o', 'o', 'o', 'o', '*']
plt.style.use('dark_background')
plt.figure(figsize = (10, 10))
# for i in range(len(catalogs)):
    # plt.scatter(df[df['Cat'] == catalogs[i]]['B-R'], df[df['Cat'] == catalogs[i]]['G_abs_P50'], color = colors[i], s = sizes[i], marker = markers[i], linewidths = 0)
# plt.plot(x_20kK, y_20kK, '--b')
# plt.plot(x_10kK, y_10kK, '--c')
plt.plot(x_zams, y_zams, '-w')
# plt.scatter(df_marks['(B-R)_20'], df_marks['G_20'], marker = 'o', s = 35, color = 'k', edgecolors = 'w', linewidths = 1.5, zorder = 10)
# plt.scatter(df_marks['(B-R)_10'], df_marks['G_10'], marker = 'o', s = 35, color = 'k', edgecolors = 'w', linewidths = 1.5, zorder = 10)
# plt.text(3.8, 8.22, r'$20\;k$K Extinction track', color = 'w', fontsize = 12, rotation = -24.9)
# plt.text(3.6, 9.92, r'$10\;k$K Extinction track', color = 'w', fontsize = 12, rotation = -24.9)
plt.text(2.06, 11.35, 'ZAMS', color = 'w', fontsize = 12, rotation = -47)
plt.xlabel(r'$G_\mathrm{BP}-G_\mathrm{RP}$', fontsize = 16)
plt.ylabel(r'$G_\mathrm{abs}$', fontsize = 16)
# plt.legend(handles =  [Line2D([0], [0], marker = 'o', color = 'none', markerfacecolor = colors[0], markeredgecolor = 'none', markersize = 7, label = catalog_labels[0]),
#                        Line2D([0], [0], marker = 'o', color = 'none', markerfacecolor = colors[1], markeredgecolor = 'none', markersize = 7, label = catalog_labels[1]),
#                        Line2D([0], [0], marker = 'o', color = 'none', markerfacecolor = colors[2], markeredgecolor = 'none', markersize = 7, label = catalog_labels[2]),
#                        Line2D([0], [0], marker = 'o', color = 'none', markerfacecolor = colors[3], markeredgecolor = 'none', markersize = 7, label = catalog_labels[3]),
#                        Line2D([0], [0], marker = '*', color = 'none', markerfacecolor = colors[4], markeredgecolor = 'none', markersize = 11, label = catalog_labels[4])],
#                        loc = 'upper right', bbox_to_anchor = (0.99, 0.99), fontsize = 10.5, labelspacing = 0.15)
plt.minorticks_on()
plt.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 13)
plt.grid(which = 'major', linestyle = ':')
plt.grid(which = 'minor', linestyle = ':', alpha = 0.33)
plt.gca().set_axisbelow(True)
plt.gca().invert_yaxis()
plt.xlim(-0.9, 5.4)
plt.ylim(14.0, -9.1)
plt.gca().set_yticks(np.arange(-10, 14, 2.5)[1:])
plt.gca().set_yticklabels(np.arange(-10, 14, 2.5)[1:])
plt.savefig('Outputs/HR_Fig_01.png', format = 'png', dpi = 300)
plt.close('all')
