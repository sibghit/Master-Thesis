'''
Created on 15.07.2019

@author: ullah
'''
import os
import numpy as np
import pandas as pd
import pickle as pkl
from dateutil import parser
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

uni_path = ('C:\\Users\\ullah\\OneDrive\\')
home_path = ('O:\\')

path = uni_path

dir = os.path.abspath(os.path.join(path,
r'UOS Germany/Thesis/sibghat_neckar_data/'))
dir1 = os.path.abspath(os.path.join(path,
r'scripts_cloud/Assignment 9/old_model'))
dir2 = os.path.abspath(os.path.join(path,
r'scripts_cloud/Assignment 15/old_model'))
out_dir = os.path.abspath(os.path.join(path,
r'UOS Germany/Thesis/Assignment 16/'))

dates = pd.date_range('1961-01-01', '2015-12-31')
names = ['420', '3421', '3465', '3470']
# station = '420'
dates_start = '1961-01-01'
dates_end = '2015-12-31'
for j in names:
    station = j
    daily_discharge = (pd.read_csv(dir + '\\neckar_daily_discharge_1961_2015.csv',
    sep=';', parse_dates=[1], index_col=0))
    daily_discharge = daily_discharge.loc[dates_start:dates_end, [station]]

    pet = (pd.read_csv(dir + '\\rockenau_pet_1961_2015_lumped.csv',
    sep=';', parse_dates=[1], index_col=0, nrows=dates.size))
    pet = pet.loc[dates_start:dates_end, [station]]

    ppt = (pd.read_csv(dir + '\\rockenau_ppt_1961_2015_lumped.csv',
    sep=';', parse_dates=[1], index_col=0, nrows=dates.size))
    ppt = ppt.loc[dates_start:dates_end, [station]]

    temp = (pd.read_csv(dir + '\\rockenau_tem_1961_2015_lumped.csv',
    sep=';', parse_dates=[1], index_col=0, nrows=dates.size))
    temp = temp.loc[dates_start:dates_end, [station]]

    watersheds = (pd.read_csv(dir + '\\watersheds_cumm_cat_areas.csv',
    sep=';',
    index_col=0))

    # Files Naming
    (station_p, station_d, station_c, station_j) = ('_' + station + '.pkl',
    '_' + station + '.pdf', '_' + station + '.csv', '_' + station + '.jpg')
    state = '_full'

    Area = (watersheds.loc[float(station), ['cumm_area']]).values

    cols = (temp_col, prec_col, pet_col, snow_col, not_snow_col, am_col, sm_col,
    et_col, rn_col, us_col, ul_col, rn_uu_col, rn_ul_col, rn_d_col, rn_l_col,
    Q_obs_col, Q_sim_col) = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16)

    rs = np.zeros((daily_discharge.shape[0], len(cols)))
    ini_v = np.zeros(len(cols))

    rs = np.vstack((ini_v, rs))
    strt_idx = 1
    off_idx = 365

    rs[strt_idx:, temp_col] = np.array(temp[station])
    rs[strt_idx:, prec_col] = np.array(ppt[station])
    rs[strt_idx:, pet_col] = np.array(pet[station])
    rs[strt_idx:, Q_obs_col] = np.array(daily_discharge[station])

    flow_diff = np.diff(np.hstack((0, (np.array(daily_discharge[station])))))

    Ris_val_per = 30
    rec_val_per = 30

    pos = np.where(flow_diff > 0, 1, 0)
    neg = np.where(flow_diff < 0, 1, 0)
    pos_elm = np.multiply(pos, flow_diff)
    neg_elm = np.multiply(neg, flow_diff)
    pos_arr = np.delete(pos_elm, np.where(pos_elm == 0), axis=0)
    neg_arr = np.delete(neg_elm, np.where(neg_elm == 0), axis=0)
    flow_diff_contains = {'zeros':np.count_nonzero(flow_diff == 0),
    '+ve_#' :pos_arr.size, '-ve_#':neg_arr.size}
    # print('flow_diff_contains = ', flow_diff_contains)
    # print('\n')
    Q_sort_asc = np.sort(pos_arr)
    Q_sort_des = -np.sort(-neg_arr)
    R_indc = int((Q_sort_asc.size - (Ris_val_per / 100) * Q_sort_asc.size))
    r_indc = int((Q_sort_des.size - (rec_val_per / 100) * Q_sort_des.size))
    rising_limit = Q_sort_asc[R_indc]
    recession_limit = Q_sort_des[r_indc]

    # print('rising_limit= ', rising_limit)
    # print('recession_limit= ', recession_limit)
    # print('\n')

    if (station == '420'):
        p = pd.read_csv(dir2 + '\\opt_full_par' + station_c, index_col=0)
    elif (station == '3421'):
        p = pd.read_csv(dir2 + '\\opt_full_par' + station_c, index_col=0)
    elif (station == '3465'):
        p = pd.read_csv(dir1 + '\\opt_full_par' + station_c, index_col=0)
    elif (station == '3470'):
        p = pd.read_csv(dir1 + '\\opt_full_par' + station_c, index_col=0)

    tt = p.at['tt', 'val']
    cmelt = p.at['cmelt', 'val']
    FC = p.at['FC', 'val']
    Beta_1 = p.at['Beta_1', 'val']
    Beta_2 = p.at['Beta_2', 'val']
    PWP = p.at['PWP', 'val']
    L1 = p.at['L1', 'val']
    K_uu = p.at['K_uu', 'val']
    K_ul = p.at['K_ul', 'val']
    K_d = p.at['K_d', 'val']
    k_ll = p.at['k_ll', 'val']

    def objective (x):
        Beta = x

        for i in range(strt_idx, rs.shape[0]):
            pr = rs[i - 1]
            cr = rs[i]

    #         if cr[Q_obs_col] - pr[Q_obs_col] >= rising_limit:
    #
    #             Beta = Beta_1
    #
    #         elif cr[Q_obs_col] - pr[Q_obs_col] <= recession_limit:
    #
    #             Beta = Beta_2

        #     snow,not_snow
            if cr[temp_col] < tt:
                snow = pr[snow_col] + cr[prec_col]
                not_snow = 0
            else:
                snow = max(0, pr[snow_col] - (cmelt * (cr[temp_col] - tt)))
                not_snow = (min(pr[snow_col], cmelt * (cr[temp_col] - tt)) +
                 cr[prec_col])
            cr[snow_col] = snow
            cr[not_snow_col] = not_snow
        #     am,et,sm
            am = pr[sm_col] + (not_snow * (1 - (pr[sm_col] / FC) ** Beta))
            cr[am_col] = am
            if(pr[sm_col] > PWP):
                et = min(am, cr[pet_col])
            else:
                et = min(am, (pr[sm_col] / FC) * (cr[pet_col]))
            cr[et_col] = et
            sm = max(0, am - et)
            cr[sm_col] = sm
        #     rn
            rn = not_snow * (pr[sm_col] / FC) ** Beta
            cr[rn_col] = rn
        #     rn_uu,rn_ul,rn_d,us
            rn_uu = max(0, K_uu * (pr[us_col] - L1))
            cr[rn_uu_col] = rn_uu
            rn_ul = K_ul * (pr[us_col] - rn_uu)
            cr[rn_ul_col] = rn_ul
            rn_d = K_d * (pr[us_col] - rn_uu - rn_ul)
            cr[rn_d_col] = rn_d
            us = pr[us_col] + rn - rn_uu - rn_ul - rn_d
            cr[us_col] = us
        #   ul,rn_l
            rn_l = k_ll * pr[ul_col]
            cr[rn_l_col] = rn_l
            ul = pr[ul_col] + rn_d - rn_l
            cr[ul_col] = ul
        Qmm = (rs[strt_idx:, rn_uu_col] + rs[strt_idx:, rn_ul_col] +
        rs[strt_idx:, rn_l_col])
        Q_sim = (Qmm / (1000 * 24 * 3600)) * Area
        rs[strt_idx:, Q_sim_col] = Q_sim
        qmean = np.average(rs[off_idx:, Q_obs_col])
        qobs_qmean_sq = (rs[off_idx:, Q_obs_col] - qmean) ** 2
        qsim_qobs_sq = (rs[off_idx:, Q_sim_col] - rs[off_idx:, Q_obs_col]) ** 2
        NS = 1 - (np.sum(qsim_qobs_sq) / np.sum(qobs_qmean_sq))

        return NS

    print('NS with using Beta_1 ' + station + ' = ', objective(Beta_1))
    print('NS with using Beta_2 ' + station + ' = ', objective(Beta_2))

'''
op_420_ris = np.genfromtxt(dir2 + '\opt_ris_par' + '_420.csv', dtype=float, delimiter=',')
op_420_rec = np.genfromtxt(dir2 + '\opt_rec_par' + '_420.csv', dtype=float, delimiter=',')
op_3421_ris = np.genfromtxt(dir2 + '\opt_ris_par' + '_3421.csv', dtype=float, delimiter=',')
op_3421_rec = np.genfromtxt(dir2 + '\opt_rec_par' + '_3421.csv', dtype=float, delimiter=',')
op_3465_ris = np.genfromtxt(dir1 + '\opt_ris_par' + '_3465.csv', dtype=float, delimiter=',')
op_3465_rec = np.genfromtxt(dir1 + '\opt_rec_par' + '_3465.csv', dtype=float, delimiter=',')
op_3470_ris = np.genfromtxt(dir1 + '\opt_ris_par' + '_3470.csv', dtype=float, delimiter=',')
op_3470_rec = np.genfromtxt(dir1 + '\opt_rec_par' + '_3470.csv', dtype=float, delimiter=',')
parameters = np.genfromtxt(dir + '\\parameters\\Parameters.csv',
dtype=float, delimiter=',')
lb = np.array([parameters[1, 2], parameters[2, 2], parameters[4, 2],
parameters[5, 2], parameters[6, 2], parameters[7, 2], parameters[8, 2],
parameters[9, 2], parameters[10, 2], parameters[11, 2]])
ub = np.array([parameters[1, 3], parameters[2, 3], parameters[4, 3],
parameters[5, 3], parameters[6, 3], parameters[7, 3], parameters[8, 3],
parameters[9, 3], parameters[10, 3], parameters[11, 3]])

cols = (opt_420_ris_col, opt_420_rec_col, opt_3421_ris_col, opt_3421_rec_col,
opt_3465_ris_col, opt_3465_rec_col, opt_3470_ris_col, opt_3470_rec_col, lb_col,
ub_col, scale_420_ris_col, scale_420_rec_col , scale_3421_ris_col,
scale_3421_rec_col, scale_3465_ris_col, scale_3465_rec_col, scale_3470_ris_col,
scale_3470_rec_col) = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)

x = (np.empty((op_420_ris[2:, 1].shape[0], len(cols)))) * np.nan

x[:, opt_420_ris_col] = op_420_ris[2:, 1]
x[:, opt_420_rec_col] = op_420_rec[2:, 1]
x[:, opt_3421_ris_col] = op_3421_ris[2:, 1]
x[:, opt_3421_rec_col] = op_3421_rec[2:, 1]
x[:, opt_3465_ris_col] = op_3465_ris[2:, 1]
x[:, opt_3465_rec_col] = op_3465_rec[2:, 1]
x[:, opt_3470_ris_col] = op_3470_ris[2:, 1]
x[:, opt_3470_rec_col] = op_3470_rec[2:, 1]
x[:, lb_col] = lb
x[:, ub_col] = ub

for i in range(x.shape[0]):
    scale_420_ris = (x[i, opt_420_ris_col] - x[i, lb_col]) / (x[i, ub_col] -
    x[i, lb_col])
    x[i, scale_420_ris_col] = scale_420_ris
    scale_420_rec = (x[i, opt_420_rec_col] - x[i, lb_col]) / (x[i, ub_col] -
    x[i, lb_col])
    x[i, scale_420_rec_col] = scale_420_rec
    scale_3421_ris = (x[i, opt_3421_ris_col] - x[i, lb_col]) / (x[i, ub_col] -
    x[i, lb_col])
    x[i, scale_3421_ris_col] = scale_3421_ris
    scale_3421_rec = (x[i, opt_3421_rec_col] - x[i, lb_col]) / (x[i, ub_col] -
    x[i, lb_col])
    x[i, scale_3421_rec_col] = scale_3421_rec
    scale_3465_ris = (x[i, opt_3465_ris_col] - x[i, lb_col]) / (x[i, ub_col] -
    x[i, lb_col])
    x[i, scale_3465_ris_col] = scale_3465_ris
    scale_3465_rec = (x[i, opt_3465_rec_col] - x[i, lb_col]) / (x[i, ub_col] -
    x[i, lb_col])
    x[i, scale_3465_rec_col] = scale_3465_rec
    scale_3470_ris = (x[i, opt_3470_ris_col] - x[i, lb_col]) / (x[i, ub_col] -
    x[i, lb_col])
    x[i, scale_3470_ris_col] = scale_3470_ris
    scale_3470_rec = (x[i, opt_3470_rec_col] - x[i, lb_col]) / (x[i, ub_col] -
    x[i, lb_col])
    x[i, scale_3470_rec_col] = scale_3470_rec

# my_xticks = ['tt', 'cmelt', 'FC', 'Beta', 'PWP', 'L1', 'K_uu', 'K_ul', 'K_d', 'k_ll']
my_xticks = ['TT', 'Cmelt', 'FC', 'Beta', 'PWP', 'L1', 'K_uu', 'K_ul', 'K_d', 'K_ll']

x_axis = np.arange(len(my_xticks))
fig = plt.figure(figsize=(15, 5))
plt.xticks(x_axis, my_xticks)
plt.plot(x_axis, x[:, scale_420_ris_col], lw=1.5, alpha=1,
label='optimized parameter at rising')
plt.plot(x_axis, x[:, scale_420_rec_col], lw=1.5, alpha=1,
label='optimized parameter at recession')
plt.grid()
plt.ylabel('Normalized Values [-]', fontsize=12)
plt.xlabel('Optimized Parameters [-]', fontsize=12)
plt.legend()
plt.show()

# my_xticks = ['tt', 'cmelt', 'FC', 'Beta', 'PWP', 'L1', 'K_uu', 'K_ul', 'K_d', 'k_ll']
x_axis = np.arange(len(my_xticks))
fig = plt.figure(figsize=(15, 5))
plt.xticks(x_axis, my_xticks)
plt.plot(x_axis, x[:, scale_3421_ris_col], lw=1.5, alpha=1,
label='optimized parameter at rising')
plt.plot(x_axis, x[:, scale_3421_rec_col], lw=1.5, alpha=1,
label='optimized parameter at recession')
plt.grid()
plt.ylabel('Normalized Values [-]', fontsize=12)
plt.xlabel('Optimized Parameters [-]', fontsize=12)
plt.legend()
plt.show()

# my_xticks = ['tt', 'cmelt', 'FC', 'Beta', 'PWP', 'L1', 'K_uu', 'K_ul', 'K_d', 'k_ll']
x_axis = np.arange(len(my_xticks))
fig = plt.figure(figsize=(15, 5))
plt.xticks(x_axis, my_xticks)
plt.plot(x_axis, x[:, scale_3465_ris_col], lw=1.5, alpha=1,
label='optimized parameter at rising')
plt.plot(x_axis, x[:, scale_3465_rec_col], lw=1.5, alpha=1,
label='optimized parameter at recession')
plt.grid()
plt.ylabel('Normalized Values [-]', fontsize=12)
plt.xlabel('Optimized Parameters [-]', fontsize=12)
plt.legend()
plt.show()

# my_xticks = ['tt', 'cmelt', 'FC', 'Beta', 'PWP', 'L1', 'K_uu', 'K_ul', 'K_d', 'k_ll']
x_axis = np.arange(len(my_xticks))
fig = plt.figure(figsize=(15, 5))
plt.xticks(x_axis, my_xticks)
plt.plot(x_axis, x[:, scale_3470_ris_col], lw=1.5, alpha=1,
label='optimized parameter at rising')
plt.plot(x_axis, x[:, scale_3470_rec_col], lw=1.5, alpha=1,
label='optimized parameter at recession')
plt.grid()
plt.ylabel('Normalized Values [-]', fontsize=12)
plt.xlabel('Optimized Parameters [-]', fontsize=12)
plt.legend()
plt.show()

# for i in range (scale_420_ris_col, scale_3470_rec_col, 2):
#     my_xticks = ['tt', 'cmelt', 'FC', 'Beta', 'PWP', 'L1', 'K_uu', 'K_ul', 'K_d', 'k_ll']
#     x_axis = np.arange(len(my_xticks))
#     fig = plt.figure(figsize=(15, 5))
#     plt.xticks(x_axis, my_xticks)
#     plt.plot(x_axis, x[:, i], lw=1.5, alpha=1,
#     label='optimized parameter at rising')
#     plt.plot(x_axis, x[:, i + 1], lw=1.5, alpha=1,
#     label='optimized parameter at recession')
#     plt.grid()
#     plt.ylabel('Values [scaled]', fontsize=12)
#     plt.xlabel('Optimized Parameters [Names]', fontsize=12)
#     plt.legend()
#     plt.show()

'''
