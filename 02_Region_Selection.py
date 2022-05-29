import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
import matplotlib.path as mpltPath
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

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
df_final_sample = pd.read_pickle('Outputs/ALS_III_Symposium_v0.91.pkl')
df_final_sample['x'] = (df_final_sample['ALS3_dist_P50']*np.cos(np.deg2rad(df_final_sample['GLAT']))*np.cos(np.deg2rad(df_final_sample['GLON'])))
df_final_sample['y'] = (df_final_sample['ALS3_dist_P50']*np.cos(np.deg2rad(df_final_sample['GLAT']))*np.sin(np.deg2rad(df_final_sample['GLON'])))
df_final_sample['z'] = (df_final_sample['ALS3_dist_P50']*np.sin(np.deg2rad(df_final_sample['GLAT'])))
df = df_final_sample[df_final_sample['Cat'].isin(['M', 'I', 'L', 'H'])]
df = df.reset_index(drop = True)
points = np.array([(df['x'].values[i], df['y'].values[i]) for i in range(len(df['x'].values))])

# CREATE REGIONS
x_cepspur, y_cepspur = superellipse(-970, 400, 340, 1450, 38, 2.3, 1000)
x_circ_1, y_circ_1 = make_circle(14400, -5100, 14480, 5000)
x_circ_2, y_circ_2 = make_circle(14300, -5700, 16950, 5000)
cepspur = mpltPath.Path([(x_cepspur[i], y_cepspur[i]) for i in range(len(x_cepspur))])
circ_1 = mpltPath.Path([(x_circ_1[i], y_circ_1[i]) for i in range(len(x_circ_1))])
circ_2 = mpltPath.Path([(x_circ_2[i], y_circ_2[i]) for i in range(len(x_circ_2))])

# MAKE NEW DATA FRAME
cond_sag = circ_1.contains_points(points)
cond_cep = cepspur.contains_points(points)
cond_per = (~circ_2.contains_points(points)) & (~cond_cep)
cond_ori =  (~cond_cep) & (~cond_per) & (~cond_sag)
df.loc[cond_sag, 'Reg'] = 'SAG'
df.loc[cond_cep, 'Reg'] = 'CEP'
df.loc[cond_ori, 'Reg'] = 'ORI'
df.loc[cond_per, 'Reg'] = 'PER'

# VISUALIZE GALAXY MAP
plt.rcParams['text.usetex'] = True
plt.figure(figsize = (12, 12))
x = df_final_sample[df_final_sample['Cat'].isin(['M', 'I'])]['x'].values
y = df_final_sample[df_final_sample['Cat'].isin(['M', 'I'])]['y'].values
plt.scatter(x, y, c = 'k', s = 1, alpha = 0.2)
plt.plot(x_cepspur, y_cepspur, linestyle = '-', linewidth = 2, color = 'black', alpha = 0.2)
plt.plot(x_circ_1, y_circ_1, linestyle = '--', color = 'black', linewidth = 2, alpha = 0.2)
plt.plot(x_circ_2, y_circ_2, linestyle = '--', color = 'black', linewidth = 2, alpha = 0.2)
plt.text(3850, 0, r'$l =\; 0^{\circ}$', ha = 'right', va = 'center', fontsize = 18)
plt.text(0, 3850, r'$l =\; 90^{\circ}$', ha = 'center', va = 'top', fontsize = 18)
plt.text(-3850, 0, r'$l =\; 180^{\circ}$', ha = 'left', va = 'center', fontsize = 18)
plt.text(0, -3850, r'$l =\; 270^{\circ}$', ha = 'center', va = 'bottom', fontsize = 18)
plt.xlabel('$\it{x}$ [pc]', fontsize = 18)
plt.ylabel('$\it{y}$ [pc]', fontsize = 18)
plt.minorticks_on()
plt.grid(which = 'major', linestyle = ':', alpha = 0.40)
plt.grid(which = 'minor', linestyle = ':', alpha = 0.20)
plt.tick_params(axis = 'both', direction = 'in', which = 'major', length = 5, width = 0.7, labelsize = 16)
plt.tick_params(axis = 'both', direction = 'in', which = 'minor', length = 3, width = 0.7)
plt.xlim(-4000, 4000)
plt.ylim(-4000, 4000)
plt.savefig('Outputs/Map_Selection.png', format = 'png', dpi = 200)

# MERGE CATALOG WITH ORIGINAL AND SAVE
# Merge REG data from M+I sample into the final sample ()
df = pd.merge(left = df_final_sample, right = df[['ID_ALS', 'Reg']], on = 'ID_ALS', how = 'left')
df_original = pd.read_pickle('Inputs/ALS_III_Symposium_Original.pkl')
df_original = df_original[['ID_ALS', 'ID_GOSC', 'JM_dist_mode', 'JM_dist_P50',
                           'JM_dist_mean', 'JM_dist_HDI_low', 'JM_dist_HDI_high',
                           'JM_dist_P16_low', 'JM_dist_P84_high','SpT_ALS',
                           'SpT_GOSC', 'SpT_Simbad', 'Dflag', 'ID_DUP',
                           'Other_Crossmatch_Candidates', 'Comments']]
# Merge matched data with the rest of the catalog
df = pd.merge(left = df_original, right = df, on = 'ID_ALS', how = 'left')
df['Reg'] = df['Reg'].replace(np.nan, '')
df.loc[df['Dflag'] != 'N', 'Cat'] = 'D'
df.loc[df['Cat'].isna(), 'Cat'] = 'U'
df.rename(columns = {'JM_dist_mode':'ALS2_dist_mode', 'JM_dist_P50':'ALS2_dist_P50',
                     'JM_dist_mean':'ALS2_dist_mean', 'JM_dist_HDI_low':'ALS2_dist_HDIl',
                     'JM_dist_HDI_high':'ALS2_dist_HDIh', 'JM_dist_P16_low':'ALS2_dist_P16',
                     'JM_dist_P84_high':'ALS2_dist_P84'}, inplace = True)
df = df[['ID_ALS', 'ID_EDR3', 'ID_GOSC', 'RA', 'DEC', 'GLON', 'GLAT', 'Plx',
         'Plx_err', 'Plx_zero', 'Plx_corr', 'Plx_err_corr', 'RUWE', 'BJ1_dist',
         'BJ1_dist_low', 'BJ1_dist_high', 'BJ2_dist', 'BJ2_dist_low', 'BJ2_dist_high',
         'ALS2_dist_mode', 'ALS2_dist_mean', 'ALS2_dist_P50', 'ALS2_dist_P16',
         'ALS2_dist_P84', 'ALS2_dist_HDIl', 'ALS2_dist_HDIh','ALS3_dist_mode',
         'ALS3_dist_mean', 'ALS3_dist_P50', 'ALS3_dist_P16', 'ALS3_dist_P84',
         'ALS3_dist_HDIl', 'ALS3_dist_HDIh', 'PM_RA', 'PM_RA_err', 'PM_DEC',
         'PM_DEC_err', 'G_mag', 'G_corr', 'G_err', 'BP_mag', 'BP_err', 'RP_mag',
         'RP_err', 'B-R', 'CCDIST', 'G_abs_mode', 'G_abs_HDI', 'G_abs_P50',
         'G_abs_P1684', 'SpT_ALS', 'SpT_GOSC', 'SpT_Simbad', 'Cat', 'Reg', 'Dflag',
         'ID_DUP', 'Other_Crossmatch_Candidates', 'Comments']]
df.to_pickle('Outputs/ALS_III_Symposium_v1.0.pkl')
