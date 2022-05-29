import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def is_under(xp, yp, x_curve, y_curve, point_on_line = True, return_values = [True, False, None]):
    if xp > x_curve[-1] or xp < x_curve[0]:
        # Point is outside the defined limits for the curve array
        return return_values[2]
    if yp > max(y_curve):
        # Point is above all the points in the curve array
        return return_values[1]
    elif yp < min(y_curve):
        # Point is below all the points in the curve array
        return return_values[0]
    if xp in x_curve:
        # Point shares x coordinate with some point of the curve array
        ys = y_curve[np.where(x_curve == xp)[0][0]]
        if yp > ys:
            return return_values[1]
        elif yp < ys:
            return return_values[0]
        else:
            # The point is part of the curve array
            return point_on_line
    # Obtain the coordinates of the closest points in the curve array
    x_dif = x_curve-xp
    for i in range(len(x_dif)-1):
        if x_dif[i] < 0 and x_dif[i+1] > 0:
            xi = x_curve[i]
            yi = y_curve[i]
            xf = x_curve[i+1]
            yf = y_curve[i+1]
    if yp > yi and yp > yf:
        # Point is above the limits of the interval in the curve array
        return return_values[1]
    elif yp < yi and yp < yf:
        # Point is below the limits of the interval in the curve array
        return return_values[0]
    m = (yf-yi)/(xf-xi)
    c = yi-m*xi
    ys = m*xp+c
    if yp > ys:
        # Point is above the line connecting the curve array dots
        return return_values[1]
    elif yp < ys:
        # Point is below the line connecting the curve array dots
        return return_values[0]
    else:
        # Point is in the line connecting the curve array dots
        return point_on_line

# LOAD DATA
df = pd.read_pickle('Inputs/ALS_III_Symposium_v0.9.pkl')
df_HR = df[~df['Cat'].isin(['A', 'C'])]
hr_als_ids = df_HR['ID_ALS'].values
data_zams = pd.read_csv('Inputs/ZAMS_track.txt', delim_whitespace = True)
data_extc = pd.read_csv('Inputs/Extinction_tracks.txt', delim_whitespace = True)
df_zams = data_zams[data_zams.columns[[1, 0]]].copy()
df_10kK = data_extc[data_extc.columns[[2, 1]]].copy()
df_20kK = data_extc[data_extc.columns[[4, 3]]].copy()
df_10kK.rename(columns = {'G_10':'G', '(B-R)_10':'B-R'}, inplace = True)
df_20kK.rename(columns = {'G_20':'G', '(B-R)_20':'B-R'}, inplace = True)

# DIVIDE DATA SET BY HR DIAGRAM REGIONS
xpnts = (df_HR['B-R']).values
ypnts = df_HR['G_abs_P50'].values
x_zams = df_zams['B-R'].values[::-1]
y_zams = df_zams['G'].values[::-1]
x_10kK = df_10kK['B-R'].values
y_10kK = df_10kK['G'].values
x_20kK = df_20kK['B-R'].values
y_20kK = df_20kK['G'].values

# Identify “High gravity stars”
# Below ZAMS set
select_HGS1 = np.array([is_under(xpnts[i], -ypnts[i], x_zams, -y_zams) for i in range(len(df_HR))])
# Left of ZAMS set
select_HGS2 = np.array([is_under(ypnts[i], xpnts[i], y_zams, x_zams) for i in range(len(df_HR))])
# Below or left of ZAMS set
select_HGS = [True if (select_HGS1[i] == True or select_HGS2[i] == True) else False for i in range(len(df_HR))]

# Identify “Normal intermediate+low-mass stars”
# Right of ZAMS set
select_NILMS1 = np.array([is_under(ypnts[i], -xpnts[i], y_zams, -x_zams) for i in range(len(df_HR))])
# Below 10kK track
select_NILMS2 = np.array([is_under(xpnts[i], -ypnts[i], x_10kK, -y_10kK) for i in range(len(df_HR))])
# Right of ZAMS and below 10kK track set
select_NILMS = [True if (select_NILMS1[i] == True and select_NILMS2[i] == True) else False for i in range(len(df_HR))]

# Identify “Possible intermediate-mass stars”
# Right of ZAMS set
select_PIMS1 = select_NILMS1
# Left of 20kK track set
select_PIMS2 = np.array([is_under(ypnts[i], xpnts[i], y_20kK, x_20kK) for i in range(len(df_HR))])
# Above ZAMS and Left 20kK track and not in NILMS set
select_PIMS = [True if (select_PIMS1[i] == True and select_PIMS2[i] == True and select_NILMS[i] == False) else False for i in range(len(df_HR))]

# Identify “Likely massive stars”
# Not in HGS, nor NILMS nor PIMS set
select_LMS = [True if (select_HGS[i] == False and select_PIMS[i] == False and select_NILMS[i] == False) else False for i in range(len(df_HR))]

# Create categories
categories = ['']*len(df_HR)
for i in range(len(df_HR)):
    # Category of High-gravity stars
    if select_HGS[i]:
        categories[i] = 'H'
    # Category of Low/intermediate-mass stars
    if select_NILMS[i]:
        categories[i] = 'L'
    # Category of High/intermediate-mass stars
    if select_PIMS[i]:
        categories[i] = 'I'
    # Category of Likely massive stars
    if select_LMS[i]:
        categories[i] = 'M'

df.loc[df['ID_ALS'].isin(hr_als_ids), 'Cat'] = categories
# Moving stars of category (special cases where HR diagram is not so usefull)
high_gravity_ids = ['2481', '4340', '8032', '8157', '11775', '17097', '17217',
                    '17383', '19255', '19605', '19623', '19783', '591', '630',
                    '982', '1150', '1274', '1275', '1349', '2018', '9311',
                    '9313', '9317', '9343', '10448', '10702', '10795', '11394',
                    '11497', '11570', '11634', '15821', '16132', '16632', '16639',
                    '16654', '17215', '18862', '18889', '18901', '18962', '18969',
                    '19021', '19042', '19270', '19784']
massive_ids = ['14882', '14868']
extragalactic_ids = ['19597', '15895', '15896', '19598', '18185', '18840', '18845']
df.loc[(df['ID_ALS'].isin(high_gravity_ids)) &  (df['Cat'].isin(['M', 'I', 'L'])), 'Cat'] = 'H'
df.loc[(df['ID_ALS'].isin(massive_ids)) & (df['Cat'].isin(['I', 'L', 'H'])), 'Cat'] = 'M'
df.loc[(df['ID_ALS'].astype(int) >= 22000) & (df['ID_ALS'].astype(int) < 24000) & (df['Cat'].isin(['I', 'L', 'H'])), 'Cat'] = 'M'
df.loc[df['ID_ALS'].isin(extragalactic_ids), 'Cat'] = 'E'

# SAVE PLOTS
cats = ['M', 'I', 'L', 'H', 'E']
set_names = ['Likely massive stars', 'High/intermediate-mass stars', 'Low/intermediate-mass stars', 'High-gravity stars', 'Extragalactic stars']
df_plot = df[df['Cat'].isin(cats)]
for i in range(len(cats)):
    cat_set = df_plot['Cat'].values == cats[i]
    set_size = str(sum(cat_set))
    colors = ['cyan' if i else 'dimgrey' for i in cat_set]
    plt.style.use('dark_background')
    plt.figure(figsize = (12, 10))
    plt.scatter(xpnts, ypnts, color = colors, s = 0.5)
    plt.plot(x_zams, y_zams, '-w', label = 'ZAMS')
    plt.plot(x_20kK, y_20kK, '-b', label = '20kK Extinction track')
    plt.plot(x_10kK, y_10kK, '-c', label = '10kK Extinction track')
    plt.title(set_names[i]+' in the Hertzsprung-Russell diagram | '+set_size+' stars')
    plt.xlabel(r'$BP-RP$')
    plt.ylabel(r'$G_{abs}$')
    plt.legend()
    plt.gca().invert_yaxis()
    plt.savefig('Outputs/HR_'+cats[i]+'.png', bbox_inches = 'tight')
    plt.close('all')

# SAVE DATA
df.to_pickle('Outputs/ALS_III_Symposium_v0.91.pkl')
