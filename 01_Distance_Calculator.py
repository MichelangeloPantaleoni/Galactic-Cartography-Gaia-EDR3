import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Interesting cases to analyse:
# ALS 33, 147, 268, 303, 391, 394, 403, 413, 1035, 1104 and 2481

def spline(X, Y, T, sigma = 1.0):
    n = min(len(X), len(Y))
    if n <= 2:
        print('X and Y must be arrays of 3 or more elements.')
    if sigma != 1.0:
        sigma = min(sigma, 0.001)
    yp = np.zeros(2*n)
    delx1 = X[1]-X[0]
    dx1 = (Y[1]-Y[0])/delx1
    nm1 = n-1
    nmp = n+1
    delx2 = X[2]-X[1]
    delx12 = X[2]-X[0]
    c1 = -(delx12+delx1)/(delx12*delx1)
    c2 = delx12/(delx1*delx2)
    c3 = -delx1/(delx12*delx2)
    slpp1 = c1*Y[0]+c2*Y[1]+c3*Y[2]
    deln = X[nm1]-X[nm1-1]
    delnm1 = X[nm1-1]-X[nm1-2]
    delnn = X[nm1]-X[nm1-2]
    c1 = (delnn+deln)/(delnn*deln)
    c2 = -delnn/(deln*delnm1)
    c3 = deln/(delnn*delnm1)
    slppn = c3*Y[nm1-2]+c2*Y[nm1-1]+c1*Y[nm1]
    sigmap = sigma*nm1/(X[nm1]-X[0])
    dels = sigmap*delx1
    exps = np.exp(dels)
    sinhs = 0.5*(exps-1/exps)
    sinhin = 1/(delx1*sinhs)
    diag1 = sinhin*(dels*0.5*(exps+1/exps)-sinhs)
    diagin = 1/diag1
    yp[0] = diagin*(dx1-slpp1)
    spdiag = sinhin*(sinhs-dels)
    yp[n] = diagin*spdiag
    delx2 = X[1:]-X[:-1]
    dx2 = (Y[1:]-Y[:-1])/delx2
    dels = sigmap*delx2
    exps = np.exp(dels)
    sinhs = 0.5*(exps-1/exps)
    sinhin = 1/(delx2*sinhs)
    diag2 = sinhin*(dels*(0.5*(exps+1/exps))-sinhs)
    diag2 = np.concatenate([np.array([0]), diag2[:-1]+diag2[1:]])
    dx2nm1 = dx2[nm1-1]
    dx2 = np.concatenate([np.array([0]), dx2[1:]-dx2[:-1]])
    spdiag = sinhin*(sinhs-dels)
    for i in range(1, nm1):
        diagin = 1/(diag2[i]-spdiag[i-1]*yp[i+n-1])
        yp[i] = diagin*(dx2[i]-spdiag[i-1]*yp[i-1])
        yp[i+n] = diagin*spdiag[i]
    diagin = 1/(diag1-spdiag[nm1-1]*yp[n+nm1-1])
    yp[nm1] = diagin*(slppn-dx2nm1-spdiag[nm1-1]*yp[nm1-1])
    for i in range(n-2, -1, -1):
        yp[i] = yp[i]-yp[i+n]*yp[i+1]
    m = len(T)
    subs = np.repeat(nm1, m)
    s = X[nm1]-X[0]
    sigmap = sigma*nm1/s
    j = 0
    for i in range(1, nm1+1):
        while T[j] < X[i]:
            subs[j] = i
            j += 1
            if j == m:
                break
        if j == m:
            break
    subs1 = subs-1
    del1 = T-X[subs1]
    del2 = X[subs]-T
    dels = X[subs]-X[subs1]
    exps1 = np.exp(sigmap*del1)
    sinhd1 = 0.5*(exps1-1/exps1)
    exps = np.exp(sigmap*del2)
    sinhd2 = 0.5*(exps-1/exps)
    exps = exps1*exps
    sinhs = 0.5*(exps-1/exps)
    spl = (yp[subs]*sinhd1+yp[subs1]*sinhd2)/sinhs+((Y[subs]-yp[subs])*del1+(Y[subs1]-yp[subs1])*del2)/dels
    if m == 1:
        return spl[0]
    else:
        return spl

def INDIVDIST_single(pi0, spi, lat, zsun, hd, hh, frach, rvl = None, nrvl = 20001, GAUSS = False, DISK = False, RUNAWAY = False):
    # pi0 and spi in arcseconds
    if len(rvl) >= 10:
        nrvl = len(rvl)
    else:
        rvl = np.arange(nrvl)
    rv = 0.5*(rvl[:nrvl-1]+rvl[1:nrvl])
    drv = rvl[1:nrvl]-rvl[:nrvl-1]
    nrv = nrvl-1
    frachlocal = frach
    if DISK:
        frachlocal = 0.0
    if RUNAWAY:
        frachlocal = 1.0
    npi0 = 1
    if lat >= 0.0 and lat <= 1.0:
        latlocal = 1.0
    elif lat < 0.0 and lat >= -1.0:
        latlocal = -1.0
    else:
        latlocal = lat
    maux1 = 1/spi
    maux2 = pi0/spi
    m = np.outer(maux1, 1/rv)-maux2
    rv2drv = rv*rv*drv
    zaux = rv*np.sin(latlocal*np.pi/180)
    if GAUSS:
        rhod = np.exp(-0.5*((zaux+zsun)/hd)**2)
    else:
        rhod = 1/(np.cosh((zaux+zsun)/(2*hd))**2)
    rhoh = np.exp(-0.5*((zaux+zsun)/hh)**2)
    fint = rv2drv*((1-frachlocal)*rhod+frachlocal*rhoh)*np.exp(-0.5*(m**2))
    f = np.zeros(fint.shape)
    if sum(fint[0,:] != 0) == 0:
        # En este caso es oportuno aumentar los límites de rvl o aumentar
        # la resolución de rvl.
        return (rv, fint[0], -1, -1, -1, -1, -1)
    fint[0,:] = fint[0,:]/sum(fint[0,:])
    f[0,:] = fint[0,:]/drv
    mean = sum(fint[0,:]*rv)
    sigma = np.sqrt(sum(fint[0,:]*(rv-mean)**2))
    ff = np.zeros(nrvl)
    for j in range(1, nrv):
        ff[j] = ff[j-1]+fint[0 ,j-1]
    pos = np.where((ff >= 0.0001) & (ff <= 0.9999))[0]
    if len(pos) < 3:
        # Hay que aumentar la resolución de rvl
        return (rv, fint[0], -2, -2, -2, -2, -2)
    else:
        aux = spline(ff[pos], rvl[pos], [0.15865525393150702, 0.50, 0.841344746068543])
        median = aux[1]
        d_16 = aux[0]
        d_84 = aux[2]
        aux = max(f[0,:])
        pos = list(f[0,:]).index(aux)
        mode = rv[pos]
        return (rv, fint[0], mode, median, mean, d_16, d_84)

def rvl(dist_est, dist_low, dist_high, width = 10, num_slices = 10000):
    siglow = dist_est-dist_low
    sighigh = dist_high-dist_est
    if siglow == 0.0:
        siglow = 1.0
    if sighigh == 0.0:
        sighigh = 1.0
    start = dist_est-width*siglow
    end = dist_est+width*sighigh
    if start < 0.0:
        start = 0.0
    return np.linspace(start, end, int(num_slices))

def HDI(rv, pdf, p = 0.682689492):
    m = max(pdf)
    step = m/2
    k = 0
    k_prev = 1
    while k_prev != k:
        k_prev = k
        pdfbool = (pdf > m-k)
        p_current = sum(pdf*pdfbool)
        if p_current > p:
            k -= step
        elif p_current < p:
            k += step
        step /= 2
    active_range = [i for i in rv*pdfbool if i > 0.0]
    HDI_low = min(active_range)
    HDI_high = max(active_range)
    return (HDI_low, HDI_high)

def pdf_plot(rv, pdf, star_name, title_params, mean, mode, median, d_16, d_84, HDI_low, HDI_high, bj_dist, bj_dist_low, bj_dist_high, ip_dist, ip_dist_low, ip_dist_high, x_low, x_high, activate_log_scale = False, on_screen_time = 0):
    plt.style.use('dark_background')
    if ip_dist != None:
        plt.axvline(x = ip_dist, ls = '-', c = 'firebrick', label = 'Inverse of the parallax distance')
        plt.axvline(x = ip_dist_low, ls = ':', c = 'firebrick', label = 'Inverse of the parallax uncertainty')
        plt.axvline(x = ip_dist_high, ls = ':', c = 'firebrick')
    if bj_dist != None:
        plt.axvline(x = bj_dist, ls = '-', c = 'darkslateblue', label = 'Bailer-Jones et al. distance estimate')
        plt.axvline(x = bj_dist_low, ls = ':', c = 'darkslateblue', label = 'Bailer-Jones et al. distance interval')
        plt.axvline(x = bj_dist_high, ls = ':', c = 'darkslateblue')
    plt.axvline(x = mean, ls = '-', c = 'yellow', label = 'Mean (arithmetic)')
    plt.axvline(x = median, ls = '-', c = 'magenta', label = 'Median (50th percentile)')
    plt.axvline(x = mode, ls = '-', c = 'springgreen', label = 'Mode (highest probability density)')
    plt.axvline(x = d_16, ls = '--', c = 'mediumorchid', label = 'Range of the 16th to 84th percentile')
    plt.axvline(x = d_84, ls = '--', c = 'mediumorchid')
    plt.axvline(x = HDI_low, ls = '--', c = 'cyan', label = 'Range of the highest density interval\nwith probability p = 68.27%')
    plt.axvline(x = HDI_high, ls = '--', c = 'cyan')
    plt.plot(rv, pdf, '-w')
    plt.xlim(x_low, x_high)
    plt.ylim(-0.01*max(pdf), 1.05*max(pdf))
    plt.title('Posterior distribution for the distance to '+str(star_name)+title_params)
    plt.xlabel('Distance [pc]')
    plt.ylabel('Probability density')
    if activate_log_scale:
        plt.xscale('log')
    plt.legend(loc = 'upper right')
    plt.gcf().set_size_inches(16, 9)
    plt.gca().get_yaxis().get_offset_text().set_position((-0.035, 0))
    plt.savefig('Outputs/Posterior Distributions/'+star_name+'.png', bbox_inches='tight', dpi = 200)
    if on_screen_time != 0:
        plt.show(block = False)
        plt.pause(on_screen_time)
    plt.close('all')

# LOAD DATA
df = pd.read_pickle('Inputs/ALS_III_Symposium_v0.7.pkl')
df_short = df[~df['Plx'].isna()]
df_short.reset_index(drop = True, inplace = True)
df_short_size = str(len(df_short))

star_names_ALS = np.array(['ALS '+id for id in df_short['ID_ALS'].values])
star_names_EDR3 = np.array(['Gaia EDR3 '+id for id in df_short['ID_EDR3'].values])
plx = df_short['Plx_corr'].values/1000
plx_err = df_short['Plx_err_corr'].values/1000
lat = df_short['GLAT'].values
bj_dist = df_short['BJ1_dist'].values
bj_dist_low = df_short['BJ1_dist_low'].values
bj_dist_high = df_short['BJ1_dist_high'].values
ip_dist = 1000/df_short['Plx_corr'].values
ip_dist_low = 1000/(df_short['Plx_corr'].values+df_short['Plx_err_corr'].values)
ip_dist_high = 1000/(df_short['Plx_corr'].values-df_short['Plx_err_corr'].values)
for i in range(len(df_short)):
    if np.isnan(bj_dist[i]):
        if (ip_dist[i] < 0) or (ip_dist_low[i] < 0) or (ip_dist_high[i] < 0):
            ip_dist[i] = 1000
            ip_dist_low[i] = 10
            ip_dist_high[i] = 5000
        bj_dist[i] = ip_dist[i]
        bj_dist_low[i] = ip_dist_low[i]
        bj_dist_high[i] = ip_dist_high[i]

# MODEL AND ALGORITHM PARAMETERS
num_chunks = 4
plotting = False
disk_activate = (abs(lat) < 1.0) # Modelo a usar según el correo de Jesús del 05/10/2020

# SUBDIVIDE IN CHUNKS
if num_chunks != 1:
    chunk_num = int(input('Chunk number of '+str(num_chunks)+' = '))
    data_path = 'Outputs/Distance Data/Distances_part_'+str(chunk_num).rjust(2, '0')+'.txt'
    chunk_limits = np.linspace(0, len(df_short), num_chunks+1).round().astype(int)
    initial_i, final_i = chunk_limits[chunk_num-1:chunk_num+1]
else:
    data_path = 'Outputs/Distance Data/Distances.txt'
    initial_i, final_i = [0, len(df_short)]

# ADJUST, SAVE AND VISUALIZE POSTERIOR DISTRIBUTIONS
pdf_min_threshold = 1/10000
for i in range(initial_i, final_i):
    indx = df_short[df_short['ID_ALS'] == df_short.loc[i]['ID_ALS']].index[0]+1
    print('\n  '+star_names_ALS[i]+' | '+str(indx)+' of '+df_short_size+' [ '+\
          f'{indx*100/int(df_short_size):.2f}'+' % ]')
    step_size_init = 1/30
    step_size = step_size_init
    final_dist_num_steps = 100000
    action = 'Initial RV estimate'.ljust(50)
    bj_sigma_high = bj_dist_high[i]-bj_dist[i]
    if bj_sigma_high == 0:
        bj_sigma_high = bj_dist[i]
    bj_dist_current = bj_dist[i]

    s = 0
    cnt = 0
    step = 1
    # Initialize problems
    problem_1 = True
    problem_2 = True
    problem_3 = True
    while problem_1 or problem_2 or problem_3:
        num_slices_use = (bj_dist_current+10*step*(bj_sigma_high))/step_size
        RVL = rvl(bj_dist[i], 0.0, bj_dist_high[i], 10*step, num_slices_use)
        rv, pdf, mode, median, mean, d_16, d_84 = INDIVDIST_single(plx[i], plx_err[i], lat[i], 20.0, 31.8, 490.0, 0.039, rvl = RVL, DISK = disk_activate[i])
        print('  '+str(s)+') '+action+' | RV upper limit '+f'{rv[-1]:0.2f}'+\
              ' pc'+' | '+'Mode = '+f'{mode:9.2f}'+' pc'+' | '+'Median = '+\
              f'{median:9.2f}'+' pc'+' | '+'Mean = '+f'{mean:9.2f}'+' pc'+\
              ' | '+'Distance differential = '+f'{rv[1]-rv[0]:0.5f}'+' pc')
        m = max(pdf)
        pdf_upper_dist = pdf[-1]
        problem_1 = (pdf_upper_dist > m*pdf_min_threshold)
        problem_2 = (mode == -1)
        problem_3 = (mode == -2)
        if problem_1:
            if cnt == 9:
                step *= 2
                cnt = 0
            else:
                step += 1
                cnt += 1
            if (rv[-1] > 100000*step_size):
                step_size *= 10
                action = 'Expand RV upper limit & lower resolution 10 times'.ljust(50)
            else:
                action = 'Expand RV upper limit'.ljust(50)
        elif problem_2:
            if step_size > 1/10000:
                action = 'Increase resolution 10 times'
                step_size /= 10
            else:
                action = 'Expand RV upper limit'.ljust(50)
                if cnt == 9:
                    step *= 2
                    cnt = 0
                else:
                    step += 1
                    cnt += 1
                step_size = step_size_init
        elif problem_3:
            action = 'Increase resolution 10 times'.ljust(50)
            step_size /= 10
        s += 1
        del RVL
    pdf_above_threshold_indexes = [i for i in np.arange(len(pdf))*(pdf > m*pdf_min_threshold) if i !=0]
    x_low = rv[pdf_above_threshold_indexes[0]]
    x_high = rv[pdf_above_threshold_indexes[-1]]
    if (x_high-x_low)/final_dist_num_steps > step_size_init:
        final_dist_num_steps = (x_high-x_low)/step_size_init
    RVL = np.linspace(x_low, x_high, int(final_dist_num_steps))
    rv, pdf, mode, median, mean, d_16, d_84 = INDIVDIST_single(plx[i], plx_err[i], lat[i], 20.0, 31.8, 490.0, 0.039, rvl = RVL, DISK = disk_activate[i])
    HDI_low, HDI_high = HDI(rv, pdf)

    # PLOT AND SAVE DATA
    action = 'Saving final results'.ljust(50)
    print('  '+str(s)+') '+action+' | RV upper limit '+f'{rv[-1]:0.2f}'+' pc'+\
          ' | '+'Mode = '+f'{mode:9.2f}'+' pc'+' | '+'Median = '+\
          f'{median:9.2f}'+' pc'+' | '+'Mean = '+f'{mean:9.2f}'+' pc'+' | '+\
          'Distance differential = '+f'{rv[1]-rv[0]:0.5f}'+' pc')
    # Plotting posterior distributions
    if plotting:
        title_params = ' | '+star_names_EDR3[i]+'\n | '+r'$\sigma_{\pi}/\pi$ $=$ '+f'{plx_err[i]/abs(plx[i]):.2f}'+' | '+\
                       r'$b$ $=$ '+f'{abs(lat[i]):.2f}'+r'$^{\circ}$'+' | '+\
                       r'Mode $=$ '+f'{mode:{".0f" if mode >= 500 else ".1f" if (mode >= 100) and (mode < 500) else ".2f" if (mode >= 30) and (mode < 100) else ".3f"}}'+r' $pc$'+' | '+\
                       r'Median $=$ '+f'{median:{".0f" if median >= 500 else ".1f" if (median >= 100) and (median < 500) else ".2f" if (median >= 30) and (median < 100) else ".3f"}}'+r' $pc$'+' | '+\
                       r'Mean $=$ '+f'{mean:{".0f" if mean >= 500 else ".1f" if (mean >= 100) and (mean < 500) else ".2f" if (mean >= 30) and (mean < 100) else ".3f"}}'+r' $pc$'+' | '+\
                       r'HDI $=$ ('+f'{HDI_low:{".0f" if HDI_low >= 500 else ".1f" if (HDI_low >= 100) and (HDI_low < 500) else ".2f" if (HDI_low >= 30) and (HDI_low < 100) else ".3f"}}'+' - '+f'{HDI_high:{".0f" if HDI_high >= 500 else ".1f" if (HDI_high >= 100) and (HDI_high < 500) else ".2f" if (HDI_high >= 30) and (HDI_high < 100) else ".3f"}}'+r') $pc$'+' | '+\
                       r'PR1684 $=$ ('+f'{d_16:{".0f" if d_16 >= 500 else ".1f" if (d_16 >= 100) and (d_16 < 500) else ".2f" if (d_16 >= 30) and (d_16 < 100) else ".3f"}}'+' - '+f'{d_84:{".0f" if d_84 >= 500 else ".1f" if (d_84 >= 100) and (d_84 < 500) else ".2f" if (d_84 >= 30) and (d_84 < 100) else ".3f"}}'+r') $pc$'+' | '
        if (mode-x_low) < 0.1*(x_high-x_low):
            activate_log_scale = True
        else:
            activate_log_scale = False
        pdf_plot(rv, pdf, star_names_ALS[i], title_params, mean, mode, median, d_16, d_84, HDI_low, HDI_high, bj_dist[i], bj_dist_low[i], bj_dist_high[i], ip_dist[i], ip_dist_low[i], ip_dist_high[i], x_low, x_high, activate_log_scale)
    # Saving dataInputs/
    if i == initial_i:
        f = open(data_path, 'w')
    else:
        f = open(data_path, 'a')
    f.write(','.join([star_names_ALS[i][4:], f'{mode:0.4f}', f'{mean:0.4f}', f'{median:0.4f}', f'{d_16:0.4f}', f'{d_84:0.4f}', f'{HDI_low:0.4f}', f'{HDI_high:0.4f}'])+'\n')
    f.close()
