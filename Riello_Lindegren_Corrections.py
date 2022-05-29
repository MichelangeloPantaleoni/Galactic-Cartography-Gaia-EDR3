import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from zero_point import zpt
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

def correct_gband(bp_rp, astrometric_params_solved, phot_g_mean_mag, phot_g_mean_flux):
    if np.isscalar(bp_rp) or np.isscalar(astrometric_params_solved) or np.isscalar(phot_g_mean_mag) \
                    or np.isscalar(phot_g_mean_flux):
        bp_rp = np.float64(bp_rp)
        astrometric_params_solved = np.int64(astrometric_params_solved)
        phot_g_mean_mag = np.float64(phot_g_mean_mag)
        phot_g_mean_flux = np.float64(phot_g_mean_flux)
    if not (bp_rp.shape == astrometric_params_solved.shape == phot_g_mean_mag.shape == phot_g_mean_flux.shape):
        raise ValueError('Function parameters must be of the same shape!')
    do_not_correct = np.isnan(bp_rp) | (phot_g_mean_mag<=13) | (astrometric_params_solved != 95)
    bright_correct = np.logical_not(do_not_correct) & (phot_g_mean_mag>13) & (phot_g_mean_mag<=16)
    faint_correct = np.logical_not(do_not_correct) & (phot_g_mean_mag>16)
    bp_rp_c = np.clip(bp_rp, 0.25, 3.0)
    correction_factor = np.ones_like(phot_g_mean_mag)
    correction_factor[faint_correct] = 1.00525 - 0.02323*bp_rp_c[faint_correct] + \
        0.01740*np.power(bp_rp_c[faint_correct],2) - 0.00253*np.power(bp_rp_c[faint_correct],3)
    correction_factor[bright_correct] = 1.00876 - 0.02540*bp_rp_c[bright_correct] + \
        0.01747*np.power(bp_rp_c[bright_correct],2) - 0.00277*np.power(bp_rp_c[bright_correct],3)
    gmag_corrected = phot_g_mean_mag - 2.5*np.log10(correction_factor)
    gflux_corrected = phot_g_mean_flux * correction_factor
    return gmag_corrected

def interpol(x, x_data, y_data):
    if sum(x_data > x) == len(x_data):
        return y_data[0]
    elif sum(x_data > x) == 0:
        return y_data[-1]
    elif sum(x_data == x) == 0:
        x1 = np.max(x_data[x_data < x])
        x2 = np.min(x_data[x_data > x])
        y1 = y_data[x_data == x1]
        y2 = y_data[x_data == x2]
        return (((x-x1)*(y2-y1)/(x2-x1))+y1)[0]
    else:
        return y_data[x == x_data][0]

# LOAD ALS III Preliminary
df = pd.read_pickle('Inputs/ALS_III_Symposium_GaiaData.pkl')

# CORRECT PARALLAX (Lindegren)
zpvals = []
zpt.load_tables()
for i in tqdm(range(len(df)), desc = 'Zero Point Parallaxes'):
    g, nu, psc, ecl, ap = df.loc[i, ['G_mag', 'nu_eff_used_in_astrometry', 'pseudocolour', 'ecl_lat', 'AstroParam']].values
    if ap != 3:
        # If warnings are enabled then some warnings might jump but the values
        # where this occur will be extrapolated outside the established intervals
        zpvals.append(zpt.get_zpt(g, nu, psc, ecl, ap, _warnings = True))
    else:
        zpvals.append(0.0)
df['Plx_zero'] = zpvals
df['Plx_zero'] = df['Plx_zero'].replace(np.nan, 0.0)
# Correct zero point bias
df['Plx_corr'] = df['Plx']-df['Plx_zero']

# CORRECT PARALLAX UNCERTAINTIES (Maíz & Pantaleoni)
plx_err_corr = []
sigma_s = 0.0103
for i in tqdm(range(len(df)), desc = 'Parallax external uncertainties'):
    g, plx_err, ruwe = df.loc[i, ['G_mag', 'Plx_err', 'RUWE']].values
    if ruwe < 1.4:
        k_ext = 1.0
    elif (ruwe >= 1.4) and (ruwe < 2.0):
        k_ext = 1.66
    elif (ruwe >= 2.0) and (ruwe <= 3.0):
        k_ext = 1.92
    else:
        plx_err_corr.append(plx_err)
        continue
    if (g <= 11.0) or (g > 18.0):
        k = 1.1
    elif (g > 11.0) and (g <= 12.0):
        k = 1.1+0.6*(g-11)
    elif (g > 12.0) and (g <= 18.0):
        k = 1.7-0.1*(g-12)
    else:
        plx_err_corr.append(plx_err)
        continue
    plx_err_corr.append(np.hypot(k*k_ext*plx_err, sigma_s))
df['Plx_err_corr'] = plx_err_corr

# CORRECT G (Riello)
g_cor_riello = []
for i in tqdm(range(len(df)), desc = 'G_mag Riello Correction'):
    br, ap, g, gflux = df.loc[i, ['B-R', 'AstroParam', 'G_mag', 'phot_g_mean_flux']].values
    g_cor_riello.append(correct_gband(br, ap, g, gflux))

# CORRECT CORRECTED G (Weiler & Maíz)
# Load Weiler & Maíz table
df_corr = pd.read_csv('Inputs/Weiler_Corrections.csv')
df_corr = df_corr.rename(columns = {df_corr.columns[0]:'G_mag', df_corr.columns[2]:'G_bias'})[['G_mag', 'G_bias']]
# # View how Weiler-Maíz correction works
# x = np.linspace(min(g_cor_riello), max(g_cor_riello), 1000)
# y = [interpol(i, df_corr['G_mag'].values, df_corr['G_bias'].values) for i in x]
# plt.scatter(x, y, s = 1, c = 'k')
# plt.axvline(x = min(df_corr['G_mag']), linestyle = '--', color = 'r')
# plt.axvline(x = max(df_corr['G_mag']), linestyle = '--', color = 'r')
# plt.xlim(min(g_cor_riello), max(g_cor_riello))
# plt.xlabel(r'$G_{EDR3}$ corrected from Riello et al.')
# plt.ylabel(r'$G_{EDR3}$ bias from Weiler & Máiz')
# plt.show(block = False)
# Apply Weiler & Maíz correction
g_mag = df_corr['G_mag'].values
g_bias = df_corr['G_bias'].values
g_cor = []
for i in tqdm(range(len(df)), desc = 'G_mag Weiler Correction'):
    if np.isnan(g_cor_riello[i]):
        g_cor.append(np.nan)
    else:
        g_cor.append(interpol(df.loc[i, 'G_mag'], g_mag, g_bias))
df['G_corr'] = np.array(g_cor_riello) + np.array(g_cor)

# CALCULATE PHOTOMETRIC UNCERTAINTIES
# Calculate G uncertainties
df['G_err'] = np.hypot((-2.5/np.log(10))*(df['phot_g_mean_flux_error']/df['phot_g_mean_flux']), 0.0027553202)
# Calculate BP uncertainties
df['BP_err'] = np.hypot((-2.5/np.log(10))*(df['phot_bp_mean_flux_error']/df['phot_bp_mean_flux']), 0.0027901700)
# Calculate RP uncertainties
df['RP_err'] = np.hypot((-2.5/np.log(10))*(df['phot_rp_mean_flux_error']/df['phot_rp_mean_flux']), 0.0037793818)

# SAVE CATALOG
df = df[['ID_ALS', 'ID_EDR3', 'RA', 'DEC', 'GLON', 'GLAT', 'Plx', 'Plx_err',
         'Plx_zero', 'Plx_corr', 'Plx_err_corr', 'RUWE', 'BJ1_dist', 'BJ1_dist_low',
         'BJ1_dist_high', 'BJ2_dist', 'BJ2_dist_low', 'BJ2_dist_high', 'PM_RA',
         'PM_RA_err', 'PM_DEC', 'PM_DEC_err', 'G_mag', 'G_corr', 'G_err', 'BP_mag',
         'BP_err', 'RP_mag', 'RP_err', 'B-R']]
df.to_pickle('Outputs/ALS_III_Symposium_v0.7.pkl')
