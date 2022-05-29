import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

def CCDIST(bg, gr, bg_fit, gr_fit):
    dis = []
    for i in range(len(bg)):
        dis.append(np.min(np.hypot(bg[i]-bg_fit, gr[i]-gr_fit)))
    return np.array(dis)

# LOAD DATA
print('\n  Loading data')
df = pd.read_pickle('Outputs/ALS_III_Symposium_v0.8.pkl')
df = df[(~df['Plx'].isna()) & (df['RUWE'] <= 3.0) & (df['Plx_corr']/df['Plx_err_corr'] >= 3.0) & (~df['B-R'].isna())]
df['B-G'] = df['BP_mag']-df['G_corr']
df['G-R'] = df['G_corr']-df['RP_mag']
bg = df['B-G'].values
gr = df['G-R'].values
df_locus = pd.read_pickle('Outputs/BG_GR_Locus.pkl')
BG_fit = df_locus['BG'].values
GR_fit = df_locus['GR'].values
df['CCDIST'] = CCDIST(df['B-G'].values, df['G-R'].values, BG_fit, GR_fit)

# SELECTION OF CCDIST CUTOFFS
cutoffs_examples = [0.155, 0.130, 0.110, 0.095, 0.080, 0.070, 0.055, 0.040]
cutoff_final = 0.150

# VISUALIZE CCDIST CUTOFF
print('  Saving CCDIST cutoff diagram')
plt.style.use('dark_background')
ccdist_thresholds = np.logspace(-4, 0, 10000)
size_sample = [sum(abs(df['CCDIST']) < cut) for cut in ccdist_thresholds]
plt.figure(figsize = (16, 9))
plt.axhline(y = len(df), linestyle = '--', color = 'crimson')
colors = pylab.cm.cool(np.linspace(0, 1, len(cutoffs_examples)))
for i, cutoff in enumerate(cutoffs_examples):
    plt.axvline(x = cutoff, linestyle = ':', color = colors[i])
    size = sum(abs(df['CCDIST']) < cutoff)
    text = 'New sample size = '+str(size)+' stars | '+f'{100*size/len(df):.1f}'+\
           ' % | Stars removed = '+str(len(df)-size)
    plt.gca().text(cutoff*(1-10**(-1)), 10**2, text, rotation = 90, color = colors[i], fontsize = 9, verticalalignment = 'bottom')
plt.gca().text(10**(-3.5), len(df)*(1-0.03), 'Sample before the removal of bad colours | '+str(len(df))+' stars', color = 'crimson', fontsize = 11)
plt.plot(ccdist_thresholds, size_sample, '-w')
plt.xscale('log')
plt.xlim(10**(-4), 10**0)
plt.xlabel('|CCDIST| cutoff values')
plt.ylabel('Sample size after |CCDIST| cutoff')
plt.gca().set_xticks([0.0001, 0.001, 0.01, 1]+cutoffs_examples)
plt.gca().set_xticklabels([0.0001, 0.001, 0.01, 1]+cutoffs_examples)
for i, ticklabel in enumerate(plt.gca().get_xticklabels()):
    if i not in range(4):
        ticklabel.set_rotation(90)
for i in range(len(cutoffs_examples)):
    plt.gca().get_xticklabels()[4+i].set_color(colors[i])
plt.savefig('Outputs/SampleSize_VS_CCdistCutoff.png', dpi = 300, bbox_inches = 'tight')
plt.close('all')

# VISUALIZE GAIA COLOR-COLOR DIAGRAM
color_selected = 'cyan'
color_expelled = 'crimson'
color_curves = 'springgreen'
df_selected = df[abs(df['CCDIST']) < cutoff_final]
df_expelled = df[abs(df['CCDIST']) >= cutoff_final]
size = len(df_selected)
text = 'New sample size = '+str(size)+' stars | '+f'{100*size/len(df):.1f}'+\
       ' % | Stars removed = '+str(len(df)-size)
print('  Saving Gaia color-color diagram for the '+str(cutoff_final)+' cutoff')
text_GaiaSimbad_CC = 'Gaia-Simbad color-color diagrams selected by |CCDIST| < '+str(cutoff_final)+' cutoff\n'
fig = plt.figure(figsize = (16, 9))
plt.scatter(df_selected['B-G'], df_selected['G-R'], color = color_selected, s = 0.8, label = 'Selected data')
plt.scatter(df_expelled['B-G'], df_expelled['G-R'], color = color_expelled, s = 1.2, label = 'Excluded data')
plt.plot(BG_fit, GR_fit, linestyle = '-', color = 'crimson', label = 'Curve fit')
plt.xlabel(r'$BP-G_{corr}$')
plt.ylabel(r'$G_{corr}-RP$')
plt.legend(loc = 'lower right')
plt.gcf().text(0.513, 0.9, text_GaiaSimbad_CC, horizontalalignment = 'center', fontsize = 13)
plt.gcf().text(0.513, 0.9, text, horizontalalignment = 'center', fontsize = 11, color = colors[i])
expand = 0.1
plt.xlim(min(bg)-(max(bg)-min(bg))*expand, max(bg)+(max(bg)-min(bg))*expand)
plt.ylim(min(gr)-(max(gr)-min(gr))*expand, max(gr)+(max(gr)-min(gr))*expand)
plt.savefig('Outputs/CC_Locus_Selection.png', dpi = 400)
plt.close('all')
