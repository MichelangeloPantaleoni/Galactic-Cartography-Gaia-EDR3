import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def CCDIST(bg, gr, bg_fit, gr_fit):
    ccdist = []
    for i in range(len(bg)):
        if sum(np.isnan(np.array([bg[i], gr[i]]))) == 0:
            ccdist.append(np.min(np.hypot(bg[i]-bg_fit, gr[i]-gr_fit)))
        else:
            ccdist.append(np.nan)
    return np.array(ccdist)

# LOAD DATA
print('\n  Loading data')
df = pd.read_pickle('Outputs/ALS_III_Symposium_v0.8.pkl')
df_locus = pd.read_pickle('Outputs/BG_GR_Locus.pkl')
BG_fit = df_locus['BG'].values
GR_fit = df_locus['GR'].values

# CHANGE EXTRAGALACTIC STARS DISTANCES
extragalactic_ids = ['19597', '15895', '15896', '19598', '18185', '18840', '18845']
lmc = extragalactic_ids[1:]
smc = extragalactic_ids[0]
# From Pietrzynski et al. 2019 and Cioni et al. 2000
df.loc[df['ID_ALS'].isin(lmc), 'ALS3_dist_P50'] = 49590
df.loc[df['ID_ALS'] == smc, 'ALS3_dist_P50'] = 62800
df.loc[df['ID_ALS'].isin(lmc), 'ALS3_dist_mode'] = 49590
df.loc[df['ID_ALS'] == smc, 'ALS3_dist_mode'] = 62800
df.loc[df['ID_ALS'].isin(lmc), 'ALS3_dist_mean'] = 49590
df.loc[df['ID_ALS'] == smc, 'ALS3_dist_mean'] = 62800
df.loc[df['ID_ALS'].isin(lmc), 'ALS3_dist_P16'] = 49040
df.loc[df['ID_ALS'] == smc, 'ALS3_dist_P16'] = 6000
df.loc[df['ID_ALS'].isin(lmc), 'ALS3_dist_HDIl'] = 49040
df.loc[df['ID_ALS'] == smc, 'ALS3_dist_HDIl'] = 6000
df.loc[df['ID_ALS'].isin(lmc), 'ALS3_dist_P84'] = 50140
df.loc[df['ID_ALS'] == smc, 'ALS3_dist_P84'] = 65600
df.loc[df['ID_ALS'].isin(lmc), 'ALS3_dist_HDIh'] = 50140
df.loc[df['ID_ALS'] == smc, 'ALS3_dist_HDIh'] = 65600

# CCDIST AND ABSOLUTE PHOTOMETRY
print('  Calculating CCDIST')
df['CCDIST'] = CCDIST((df['BP_mag']-df['G_corr']).values, (df['G_corr']-df['RP_mag'].values), BG_fit, GR_fit)
print('  Calculating absolute photometry')
df['G_abs_mode'] = (df['G_corr']-5*np.log10(df['ALS3_dist_mode'])+5).round(4)
dist_err = (df['ALS3_dist_HDIh']-df['ALS3_dist_HDIl'])/2
df['G_abs_HDI'] = (np.sqrt(df['G_err']**2+(5*dist_err/(np.log(10)*df['ALS3_dist_mode']))**2)).round(4)
df['G_abs_P50'] = (df['G_corr']-5*np.log10(df['ALS3_dist_P50'])+5).round(4)
dist_err = (df['ALS3_dist_P84']-df['ALS3_dist_P16'])/2
df['G_abs_P1684'] = (np.sqrt(df['G_err']**2+(5*dist_err/(np.log(10)*df['ALS3_dist_P50']))**2)).round(4)

# FILTERING AND CAT COLUMN
df['Cat'] = ''
# Astrometric filtering
df.loc[df[(df['Plx'].isna()) | (df['RUWE'] > 3) | (df['Plx_corr']/df['Plx_err_corr'] < 3)].index, 'Cat'] = 'A'
# Photometric filtering
df.loc[df[(df['Cat'] != 'A') & ((df['B-R'].isna()) | (df['CCDIST'] >= 0.15))].index, 'Cat'] = 'C'
# Extragalactic
extragalactic_ids = ['19597', '15895', '15896', '19598', '18185', '18840', '18845']
df.loc[df['ID_ALS'].isin(extragalactic_ids), 'Cat'] = 'E'

# SAVE DATA FRAME
df.to_pickle('Outputs/ALS_III_Symposium_v0.9.pkl')
