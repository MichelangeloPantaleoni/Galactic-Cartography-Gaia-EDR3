import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
from matplotlib.lines import Line2D

def make_circle(xc, yc, r, num_points):
    theta = np.linspace(0, 2*np.pi, num_points)
    x = xc+r*np.cos(theta)
    y = yc+r*np.sin(theta)
    return x, y

def superellipse(xc, yc, a, b, rot, stiffness, num_points):
    ang = rot*(np.pi/180)
    t = np.linspace(0, 2*np.pi, 1000)
    u = b*(np.abs(np.cos(t))**(2/stiffness))*np.sign(np.cos(t))
    v = a*(np.abs(np.sin(t))**(2/stiffness))*np.sign(np.sin(t))
    x = xc+u*np.cos(ang)-v*np.sin(ang)
    y = yc+u*np.sin(ang)+v*np.cos(ang)
    return x, y

# LOAD DATA
df = pd.read_pickle('Inputs/ALS_III_Symposium_v1.0.pkl')
df = df[df['Cat'].isin(['M', 'I', 'L', 'H'])]

# GALAXY MAPS
cats = {'catalogs': ['M', 'I', 'L', 'H'],
        'catalog_labels': ['M - Likely massive stars', 'I - High/intermediate-mass stars', 'L - Intermediate/low-mass stars', 'H - High-gravity stars'],
        'colors': ['blueviolet', 'royalblue', 'darkorange', 'crimson']}
plt.rcParams['text.usetex'] = True
plt.figure(figsize = (12, 12))
for j in range(len(cats['catalogs'])):
    df_set = df[df['Cat'] == cats['catalogs'][j]]
    x = (df_set['ALS3_dist_P50']*np.cos(np.deg2rad(df_set['GLAT']))*np.cos(np.deg2rad(df_set['GLON'])))
    y = (df_set['ALS3_dist_P50']*np.cos(np.deg2rad(df_set['GLAT']))*np.sin(np.deg2rad(df_set['GLON'])))
    plt.plot(x, y, '.', color = cats['colors'][j], markersize = 3, markeredgewidth = 0.0)
rangedist = [1000, 2000, 3000, 4000, 5000]
l_chosen = [311.5, 142.0, 282, 45, 225]
uncertainty = []
for i in range(len(rangedist)):
    df_set = df[abs(df['ALS2_dist_P50']-rangedist[i]) <= 50]
    uncertainty.append((np.mean(df_set['ALS2_dist_P84'])-np.mean(df_set['ALS2_dist_P16']))/2)
    x_center = np.mean(df_set['ALS3_dist_P50']*np.cos(np.deg2rad(df_set['GLAT'])))*np.cos(np.deg2rad(l_chosen[i]))
    y_center = np.mean(df_set['ALS3_dist_P50']*np.cos(np.deg2rad(df_set['GLAT'])))*np.sin(np.deg2rad(l_chosen[i]))
    x_low = np.mean(df_set['ALS3_dist_P16']*np.cos(np.deg2rad(df_set['GLAT'])))*np.cos(np.deg2rad(l_chosen[i]))
    y_low = np.mean(df_set['ALS3_dist_P16']*np.cos(np.deg2rad(df_set['GLAT'])))*np.sin(np.deg2rad(l_chosen[i]))
    x_high = np.mean(df_set['ALS3_dist_P84']*np.cos(np.deg2rad(df_set['GLAT'])))*np.cos(np.deg2rad(l_chosen[i]))
    y_high = np.mean(df_set['ALS3_dist_P84']*np.cos(np.deg2rad(df_set['GLAT'])))*np.sin(np.deg2rad(l_chosen[i]))
    plt.plot([x_low, x_high], [y_low, y_high], linestyle = '-', color = 'black')
    plt.plot(x_center, y_center, marker = 'o', color = 'white', markeredgecolor = 'black', markersize = 5)
plt.plot(0, 0, marker = '*', color = 'yellow', markeredgecolor = 'black', markersize = 12)
# x_cepspur, y_cepspur = superellipse(-1190, 260, 390, 1450, 35, 2.5, 1000)
# plt.plot(x_cepspur, y_cepspur, linestyle = '-', linewidth = 2.5, color = 'black', alpha = 0.9)
# plt.text(-1530, 680, 'Cepheus spur', ha = 'center', va = 'center', fontsize = 18, alpha = 1.0, rotation = 33.5)
# x_velOB, y_velOB = superellipse(-150, -2100, 225, 710, 79, 2.5, 1000)
# plt.plot(x_velOB, y_velOB, linestyle = '-', linewidth = 2.5, color = 'black', alpha = 0.9)
# plt.text(-230, -3000, 'Vela OB1', ha = 'center', va = 'center', fontsize = 18, alpha = 1.0, rotation = 0)
plt.text(3850, 0, r'$l =\; 0^{\circ}$', horizontalalignment = 'right', verticalalignment = 'center', fontsize = 18)
plt.text(0, 3850, r'$l =\; 90^{\circ}$', horizontalalignment = 'center', verticalalignment = 'top', fontsize = 18)
plt.text(-3850, 0, r'$l =\; 180^{\circ}$', horizontalalignment = 'left', verticalalignment = 'center', fontsize = 18)
plt.text(0, -3850, r'$l =\; 270^{\circ}$', horizontalalignment = 'center', verticalalignment = 'bottom', fontsize = 18)
plt.xlabel('$\it{x}$ [pc]', fontsize = 18)
plt.ylabel('$\it{y}$ [pc]', fontsize = 18)
plt.minorticks_on()
plt.grid(which = 'major', linestyle = ':')
plt.grid(which = 'minor', linestyle = ':', alpha = 0.33)
plt.tick_params(axis = 'both', direction = 'in', which = 'major', length = 5, width = 0.7, labelsize = 16)
plt.tick_params(axis = 'both', direction = 'in', which = 'minor', length = 3, width = 0.7)
plt.xlim(-4000, 4000)
plt.ylim(-4000, 4000)
plt.legend(handles =  [Line2D([0], [0], marker = 'o', color = 'none', markerfacecolor = cats['colors'][0], markeredgecolor = 'none', markersize = 7, label = cats['catalog_labels'][0]),
                       Line2D([0], [0], marker = 'o', color = 'none', markerfacecolor = cats['colors'][1], markeredgecolor = 'none', markersize = 7, label = cats['catalog_labels'][1]),
                       Line2D([0], [0], marker = 'o', color = 'none', markerfacecolor = cats['colors'][2], markeredgecolor = 'none', markersize = 7, label = cats['catalog_labels'][2]),
                       Line2D([0], [0], marker = 'o', color = 'none', markerfacecolor = cats['colors'][3], markeredgecolor = 'none', markersize = 7, label = cats['catalog_labels'][3]),
                       Line2D([0], [0], marker = '*', color = 'none', markerfacecolor = 'yellow', markeredgecolor = 'black', markersize = 11, label = 'The Sun')],
                       loc = 'upper right', bbox_to_anchor = (0.99, 0.99), fontsize = 12, labelspacing = 0.15)
# plt.savefig('Outputs/Figure_Map_A.pdf', bbox_inches = 'tight', pad_inches = 0.2)
plt.savefig('Outputs/Figure_Map_A.png', format = 'png', dpi = 200)
# plt.close()
