import numpy as np
import pandas as pd
from astropy.table import Table
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.optimize import curve_fit

# LOAD DATA
df = pd.read_pickle('Inputs/ALS_III_Symposium_v0.7.pkl')
df = df[~(df['Plx'].isna()) & (df['RUWE'] <= 3.0) & (df['Plx_corr']/df['Plx_err_corr'] >= 3.0)]
df = df[~df['B-R'].isna()]
df = df.reset_index(drop = True)

# DATA POINTS
bg = (df['BP_mag']-df['G_corr']).values
gr = (df['G_corr']-df['RP_mag']).values

# POLINOMIAL FIT
step = 0.025
low_limit = -0.1
high_limit = 1.6
bins = np.arange(low_limit, high_limit+step, step)
x = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
y = [np.mean(gr[np.where((bg >= bins[i]) & (bg < bins[i+1]))[0]]) for i in range(len(bins)-1)]
order = 6
coeffs = np.polyfit(x, y, order)
fit_func = np.poly1d(coeffs)
x_fit = np.linspace(min(bg), max(bg), 10001)
y_fit = fit_func(x_fit)

# LOGARITHMIC FIT
step = 0.025
low_limit_log = 1.5
high_limit_log = 4.0
bins = np.arange(low_limit_log, high_limit_log+step, step)
x = np.array([(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)])
y = []
for i in range(len(bins)-1):
    v = gr[np.where((bg >= bins[i]) & (bg < bins[i+1]))[0]]
    if len(v) == 0:
        y.append(np.nan)
    else:
        y.append(np.nanmean(v))
y = np.array(y)
x = x[~np.isnan(y)]
y = y[~np.isnan(y)]
coeffs = np.polyfit(np.log(x), y, 1)
x_log_fit = np.linspace(0.001, max(bg), 10001)
y_log_fit = coeffs[0]*np.log(x_log_fit)+coeffs[1]

# ATTACH POLYNOMIAL AND LOGARITHMIC FITS
s1 = np.array([(x_fit[i], y_fit[i]) for i in range(len(x_fit))])
s2 = np.array([(x_log_fit[i], y_log_fit[i]) for i in range(len(x_log_fit))])
change_x = x_fit[np.argmin(distance.cdist(s1, s2).min(axis = 1))]
start_x = min(bg)
end_x = max(bg)
x_fit_segment = x_fit[np.where(x_fit < change_x)]
y_fit_segment = y_fit[np.where(x_fit < change_x)]
x_log_fit_segment = x_log_fit[np.where(x_log_fit > change_x)]
y_log_fit_segment = y_log_fit[np.where(x_log_fit > change_x)]
x_final_fit = np.array(list(x_fit_segment)+list(x_log_fit_segment))
y_final_fit = np.array(list(y_fit_segment)+list(y_log_fit_segment))
indx = np.where((x_final_fit >= start_x) & (x_final_fit <= end_x))
x_final_fit = x_final_fit[indx]
y_final_fit = y_final_fit[indx]

# PLOT DATA AND POLINOMIAL FIT
plt.style.use('dark_background')
plt.figure(figsize = (16, 9))
plt.scatter(bg, gr, color = 'cyan', s = 0.8)
plt.plot(x_fit, y_fit, linestyle = '-', color = 'crimson', label = 'Polynomial fit')
plt.plot(x_log_fit, y_log_fit, linestyle = '-', color = 'purple', label = 'Logarithmic fit')
# plt.plot(x_final_fit, y_final_fit, linestyle = '-', color = 'yellow')
plt.xlabel(r'$BP-G_{corr}$')
plt.ylabel(r'$G_{corr}-RP$')
expand = 0.1
plt.xlim(min(bg)-(max(bg)-min(bg))*expand, max(bg)+(max(bg)-min(bg))*expand)
plt.ylim(min(gr)-(max(gr)-min(gr))*expand, max(gr)+(max(gr)-min(gr))*expand)
plt.legend(loc = 'lower right', bbox_to_anchor = (0.98, 0.02), fontsize = 10.5, labelspacing = 0.15)
plt.show(block = False)
plt.savefig('Outputs/CC_Locus_Fit.png', format = 'png', dpi = 300)

# SAVE COLOR-COLOR LOCUS FINAL FIT
df_cclocus = pd.DataFrame({'BG':x_final_fit, 'GR':y_final_fit})
df_cclocus.to_pickle('Outputs/BG_GR_Locus.pkl')
tab_cclocus = Table.from_pandas(df_cclocus)
tab_cclocus.write('Outputs/BG_GR_Locus.fits', format = 'fits')
