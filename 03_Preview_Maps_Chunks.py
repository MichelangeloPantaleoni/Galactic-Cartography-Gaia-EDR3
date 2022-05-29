import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# LOAD DATA
df = pd.read_pickle('Outputs/ALS_III_Symposium_v0.8.pkl')
df = df[(~df['Plx'].isna()) & (df['Plx_corr']/df['Plx_err_corr'] > 3) & (df['RUWE'] < 3)]
df = df.reset_index(drop = True)
x = (df['ALS3_dist_mode']*np.cos(np.deg2rad(df['GLAT']))*np.cos(np.deg2rad(df['GLON'])))
y = (df['ALS3_dist_mode']*np.cos(np.deg2rad(df['GLAT']))*np.sin(np.deg2rad(df['GLON'])))

# CREATE FRAMES
plt.rcParams['text.usetex'] = True
plt.style.use('dark_background')
map_size = [24000, 12000, 6000, 3000, 1000]
mark_size_stars = [0.5, 1, 1.5, 2.5, 6]
mark_size_special = [6, 9, 11, 12, 13]
for i in range(len(map_size)):
    plt.figure(figsize = (16, 9))
    plt.scatter(x, y, color = 'white', s = mark_size_stars[i], edgecolor = 'none')
    plt.plot(8178, 0, '*', color = 'crimson', markersize = mark_size_special[i])
    plt.plot(0, 0, marker = '*', color = 'springgreen', markersize = mark_size_special[i])
    plt.xlabel('$\it{x}$ [pc]', fontsize = 18)
    plt.ylabel('$\it{y}$ [pc]', fontsize = 18)
    plt.minorticks_on()
    plt.tick_params(axis = 'both', direction = 'in', which = 'major', length = 5, width = 0.7, labelsize = 16)
    plt.tick_params(axis = 'both', direction = 'in', which = 'minor', length = 3, width = 0.7)
    plt.xlim(-map_size[i], map_size[i])
    plt.ylim(-map_size[i]*(9/16), map_size[i]*(9/16))
    plt.savefig('Outputs/Maps/Map_'+str(int(map_size[i]/1000))+'kpc.png', format = 'png', dpi = 200)
    plt.close()
