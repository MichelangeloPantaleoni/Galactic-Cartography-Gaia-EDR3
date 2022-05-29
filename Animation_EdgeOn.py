import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# LOAD ALS II DATA
df = pd.read_pickle('Inputs/ALS_III_Symposium_v1.0.pkl')
df = df[df['Cat'].isin(['M', 'I'])]
df = df[df['ALS3_dist_P50'] < 3000]
zsun = 20.0

# GALAXY MAPS
num_frames = 1000
phi = np.linspace(0, 360, num_frames+1)[:-1]
phi = np.array(list(phi[np.where(phi >= 180)[0]])+list(phi[np.where(phi < 180)]))
map_name = 'GalEdge_Maps'
params = {'map_size': [16000, 7000, 3500, 1100, 700],
          'mark_size_stars': [1, 2, 3.5, 5.5, 8],
          'mark_size_special': [5, 10, 11, 12, 13]}
cats = {'catalogs': ['SAG', 'ORI', 'CEP', 'PER'],
        'colors': ['blueviolet', 'cyan', 'yellow', 'crimson']}
plt.style.use('dark_background')
plt.rcParams['text.usetex'] = True
max_size_dot = 8.5
min_size_dot = 0.5
far_limit = 2000
close_limit_size = 8.5
far_limit_size = 0.5
b = np.log(max_size_dot/min_size_dot)/(2*far_limit)
a = min_size_dot*np.exp(far_limit*b)
for j in range(len(phi)):
    print('  '+str(j+1)+' of '+str(len(phi)))
    plt.figure(figsize = (16.5, 8))
    if phi[j] >= 90:
        l_f = f'{phi[j]-90:.1f}'
    else:
        l_f = f'{phi[j]+270:.1f}'
    if phi[j] >= 270:
        l_i = f'{phi[j]-270:.1f}'
    else:
        l_i = f'{phi[j]+90:.1f}'
    if phi[j] >= 180:
        l_foreground = f'{phi[j]-180:.1f}'
    else:
        l_foreground = f'{phi[j]+180:.1f}'
    df['u'] = (df['ALS3_dist_P50']*np.cos(np.deg2rad(df['GLAT']))*np.sin(np.deg2rad(phi[j])-np.deg2rad(df['GLON'])))
    df['v'] = (df['ALS3_dist_P50']*np.cos(np.deg2rad(df['GLAT']))*np.cos(np.deg2rad(phi[j])-np.deg2rad(df['GLON'])))
    df['z'] = (df['ALS3_dist_P50']*np.sin(np.deg2rad(df['GLAT'])))
    df = df.sort_values(by = 'v', ascending = False)
    for k in range(len(cats['catalogs'])):
        df.loc[df['Reg'] == cats['catalogs'][k], 'colors'] = cats['colors'][k]
    df['sizes'] = df['v'].apply(lambda x: close_limit_size if x < -far_limit else far_limit_size if x > far_limit else a*np.exp(-x*b))
    plt.scatter(df['u'].values, df['z'].values+zsun, color = df['colors'].values, s = df['sizes'], edgecolor = 'none')
    plt.axhline(y = 0, linestyle = '--', color = 'crimson', linewidth = 0.7)
    if (phi[j] < 270) and (phi[j] > 90):
        plt.plot(0, zsun, marker = '*', color = 'springgreen', markersize = 12)
        plt.plot(8178*np.sin(np.deg2rad(phi[j])), 0, '*', color = 'crimson', markersize = 18)
    else:
        plt.plot(8178*np.sin(np.deg2rad(phi[j])), 0, '*', color = 'crimson', markersize = 5)
        plt.plot(0, zsun, marker = '*', color = 'springgreen', markersize = 12)
    plt.text(0, 463, 'Background', horizontalalignment = 'center', verticalalignment = 'center', fontsize = 14)
    plt.text(0, 428, r'$l =\; $'+f'{phi[j]:.1f}'+r'$^{\circ}$', horizontalalignment = 'center', verticalalignment = 'center', fontsize = 16)
    plt.text(1500, 440, r'$l =\; $'+l_f+r'$^{\circ}$', horizontalalignment = 'left', verticalalignment = 'center', fontsize = 16)
    plt.text(-1900, 440, r'$l =\; $'+l_i+r'$^{\circ}$', horizontalalignment = 'left', verticalalignment = 'center', fontsize = 16)
    plt.text(-2100, 440, r'$\longleftarrow$', horizontalalignment = 'center', verticalalignment = 'center', fontsize = 22)
    plt.text(2100, 440, r'$\longrightarrow$', horizontalalignment = 'center', verticalalignment = 'center', fontsize = 22)
    plt.text(0, -447, 'Foreground', horizontalalignment = 'center', verticalalignment = 'center', fontsize = 14)
    plt.text(0, -480, r'$l =\; $'+l_foreground+r'$^{\circ}$', horizontalalignment = 'center', verticalalignment = 'center', fontsize = 16)
    plt.text(1500, -460, r'$l =\; $'+l_f+r'$^{\circ}$', horizontalalignment = 'left', verticalalignment = 'center', fontsize = 16)
    plt.text(-1900, -460, r'$l =\; $'+l_i+r'$^{\circ}$', horizontalalignment = 'left', verticalalignment = 'center', fontsize = 16)
    plt.text(-2100, -460, r'$\longleftarrow$', horizontalalignment = 'center', verticalalignment = 'center', fontsize = 22)
    plt.text(2100, -460, r'$\longrightarrow$', horizontalalignment = 'center', verticalalignment = 'center', fontsize = 22)
    plt.ylabel(r'$z$ [pc]', fontsize = 17)
    plt.minorticks_on()
    plt.tick_params(axis = 'both', direction = 'in', which = 'major', length = 5, width = 0.7, labelsize = 13)
    plt.tick_params(axis = 'both', direction = 'in', which = 'minor', length = 3, width = 0.7)
    plt.xlim(-3000, 3000)
    plt.ylim(-400, 400)
    file_name = 'Outputs/Animation Python Frames/Frame_'+str(j+1).rjust(4, '0')+'.png'
    plt.savefig(file_name, format = 'png', dpi = 200)
    plt.close('all')
