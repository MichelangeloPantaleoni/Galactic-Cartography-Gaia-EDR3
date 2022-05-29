import os
import numpy as np
import pandas as pd
from astropy.table import Table
from astroquery.gaia import Gaia

# LOAD ALS III Preliminary
df = pd.read_pickle('Outputs/ALS_III_Symposium.pkl')
df['ID_ALS'] = df['ID_ALS'].astype(int)
df = df.sort_values(by = 'ID_ALS')
df['ID_ALS'] = df['ID_ALS'].astype(str)
df['ID_EDR3'] = df['ID_EDR3'].astype(str)
df = df[df['ID_EDR3'] != ''][['ID_ALS', 'ID_EDR3']]
df = df.reset_index(drop = True)

# ADD DATA FROM GAIA EDR3
file_gaia = 'Outputs/Gaia_Job_Data.pkl'
if 'Gaia_Job_Data.pkl' not in os.listdir('Outputs'):
    print('\n  Querying Gaia EDR3 data')
    adql_command_gaia = "SELECT source_id, ra, dec, parallax, parallax_error, pmra, pmra_error, pmdec, \
                         pmdec_error, phot_bp_mean_mag, phot_rp_mean_mag, phot_g_mean_mag, \
                         phot_g_mean_flux, phot_bp_mean_flux, phot_rp_mean_flux, phot_g_mean_flux_error, \
                         phot_bp_mean_flux_error, phot_rp_mean_flux_error, pmra_pmdec_corr, bp_rp, \
                         l, b, ruwe, ecl_lat, pseudocolour, astrometric_params_solved, nu_eff_used_in_astrometry \
                         FROM gaiaedr3.gaia_source \
                         WHERE source_id IN " + \
                         str(tuple(df['ID_EDR3'].values))
    job_gaia = Gaia.launch_job_async(adql_command_gaia, dump_to_file = False)
    df_job_gaia = job_gaia.get_results().to_pandas()
    df_job_gaia['source_id'] = df_job_gaia['source_id'].astype(str)
    df_job_gaia.to_pickle(file_gaia)
else:
    df_job_gaia = pd.read_pickle(file_gaia)

# MERGE GAIA JOB DATA WITH ALS III
df = pd.merge(left = df, right = df_job_gaia, left_on = 'ID_EDR3', right_on = 'source_id', how = 'left')
df.drop(['source_id'], axis = 1, inplace = True)
df.rename(columns = {'ra':'RA', 'dec':'DEC', 'parallax':'Plx', 'parallax_error':'Plx_err', 'pmra':'PM_RA',
                     'pmra_error':'PM_RA_err', 'pmdec':'PM_DEC', 'pmdec_error':'PM_DEC_err',
                     'phot_g_mean_mag':'G_mag', 'phot_bp_mean_mag':'BP_mag', 'phot_rp_mean_mag':'RP_mag',
                     'l':'GLON', 'b':'GLAT', 'ruwe':'RUWE', 'bp_rp':'B-R',
                     'astrometric_params_solved':'AstroParam'}, inplace = True)

# ADD DATA FROM BAILER-JONES EDR3
file_bj = 'Outputs/BailerJones_Job_Data.pkl'
if 'BailerJones_Job_Data.pkl' not in os.listdir('Outputs'):
    print('\n  Querying Bailer-Jones data')
    adql_command_bj = "SELECT source_id, r_med_geo, r_lo_geo, r_hi_geo, \
                         r_med_photogeo, r_lo_photogeo, r_hi_photogeo \
                         FROM external.gaiaedr3_distance \
                         WHERE source_id IN " + \
                         str(tuple(df['ID_EDR3'].values))
    job_bj = Gaia.launch_job_async(adql_command_bj, dump_to_file = False)
    df_job_bj = job_bj.get_results().to_pandas()
    df_job_bj['source_id'] = df_job_bj['source_id'].astype(str)
    df_job_bj.to_pickle(file_bj)
else:
    df_job_bj = pd.read_pickle(file_bj)

# MERGE BAILER-JONES JOB DATA WITH ALS III
df = pd.merge(left = df, right = df_job_bj, left_on = 'ID_EDR3', right_on = 'source_id', how = 'left')
df.drop(['source_id'], axis = 1, inplace = True)
df.rename(columns = {'r_med_geo':'BJ1_dist', 'r_lo_geo':'BJ1_dist_low',
                     'r_hi_geo':'BJ1_dist_high', 'r_med_photogeo':'BJ2_dist',
                     'r_lo_photogeo':'BJ2_dist_low', 'r_hi_photogeo':'BJ2_dist_high'}, inplace = True)

# SAVE DATA
df = df[['ID_ALS', 'ID_EDR3', 'RA', 'DEC', 'GLON', 'GLAT', 'Plx', 'Plx_err', 'RUWE',
         'BJ1_dist', 'BJ1_dist_low', 'BJ1_dist_high', 'BJ2_dist', 'BJ2_dist_low',
         'BJ2_dist_high', 'PM_RA', 'PM_RA_err', 'PM_DEC', 'PM_DEC_err', 'G_mag',
         'BP_mag', 'RP_mag', 'B-R', 'phot_g_mean_flux', 'phot_bp_mean_flux',
         'phot_rp_mean_flux', 'phot_g_mean_flux_error', 'phot_bp_mean_flux_error',
         'phot_rp_mean_flux_error', 'pmra_pmdec_corr', 'ecl_lat', 'pseudocolour',
         'AstroParam', 'nu_eff_used_in_astrometry']]
df.to_pickle('Outputs/ALS_III_Symposium_GaiaData.pkl')
