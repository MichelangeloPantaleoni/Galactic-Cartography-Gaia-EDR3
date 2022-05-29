import os
import numpy as np
import pandas as pd

# LOAD DATA CHUNKS
distance_columns = ['ID_ALS', 'ALS3_dist_mode', 'ALS3_dist_mean', 'ALS3_dist_P50',
                    'ALS3_dist_P16', 'ALS3_dist_P84', 'ALS3_dist_HDIl', 'ALS3_dist_HDIh']
df_chunks = []
chunks_path = 'Outputs/Distance Data'

# MERGE DATA CHUNKS
for chunks in os.listdir(chunks_path):
    df_chunks.append(pd.read_csv(chunks_path+'/'+chunks, names = distance_columns))
df = pd.concat(df_chunks)
df = df.sort_values(by = 'ID_ALS')
df['ID_ALS'] = df['ID_ALS'].astype(str)
df = df.reset_index(drop = True)

# LOAD ALS III
df_ALS3 = pd.read_pickle('Inputs/ALS_III_Symposium_v0.7.pkl')

# MERGE WITH ALS III
df = pd.merge(left = df_ALS3, right = df, on = 'ID_ALS', how = 'left')

# SAVE DATA
df.to_pickle('Outputs/ALS_III_Symposium_v0.8.pkl')
