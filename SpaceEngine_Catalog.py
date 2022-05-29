import numpy as np
import pandas as pd
import astropy.units as u
from astropy.io import ascii
from astropy.table import Table
from astropy.coordinates import SkyCoord

method = 2
split_regions = False

df = pd.read_pickle('Inputs/ALS_III_Symposium_v1.0.pkl')
df = df[df['Cat'].isin(['M', 'I'])]
coords = SkyCoord(l = df['GLON'], b = df['GLAT'], unit = (u.deg, u.deg), frame = 'galactic')
coords = coords.transform_to('icrs')
df['RA'] = coords.ra.hourangle
df['DEC'] = coords.dec.deg
if method == 1:
    # Corrección para que se vea bien en SpaceEngine
    # Faltaría otra corrección pues las estrellas cercanas al sol son menos
    # brillantes y desaparecen si te alejas
    df['G_corr'] -= 2.0
    df['G_app'] = df['G_corr']
elif method == 2:
    M = -5
    df['G_app'] = 5*np.log10(df['ALS3_dist_P50'])-5+M
df = df[['ID_ALS', 'RA', 'DEC', 'ALS3_dist_P50', 'G_app', 'Reg']]
df['ID_ALS'] = df['ID_ALS'].apply(lambda x: 'ALS '+x)
df['SpType'] = 'O2V'
df['MassSol'] = ''
df['RadSol'] = ''
df['Temperature'] = ''
SE_cols = ['Name', 'RA', 'Dec', 'Dist', 'AppMagn', 'SpecClass', 'MassSol', 'RadSol', 'Temperature']
df_all = df[['ID_ALS', 'RA', 'DEC', 'ALS3_dist_P50', 'G_app', 'SpType', 'MassSol', 'RadSol', 'Temperature']]
t = Table.from_pandas(df_all)
ascii.write(t, 'Outputs/SpaceEngine Animation/ALS_III_Catalog.csv', format = 'csv',
            names = SE_cols, overwrite = True, fast_writer = False)
if split_regions:
    for reg in ['CEP', 'ORI', 'PER', 'SAG']:
        df_reg = df[df['Reg'] == reg].reset_index(drop = True)
        df_reg = df_reg[['ID_ALS', 'RA', 'DEC', 'ALS3_dist_P50', 'G_app', 'SpType', 'MassSol', 'RadSol', 'Temperature']]
        t = Table.from_pandas(df_reg)
        ascii.write(t, 'Outputs/SpaceEngine Animation/ALS_III_Catalog_'+reg+'.csv', format = 'csv',
                    names = SE_cols, overwrite = True, fast_writer = False)
