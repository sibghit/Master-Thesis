'''
Created on 30.05.2019

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
out_dir = os.path.abspath(os.path.join(path,
r'UOS Germany/Thesis/Assignment 10/'))

dates = pd.date_range('1961-01-01', '2015-12-31')
names = ['420', '3421', '3465', '3470']
station = '3465'
dates_start = '1966-01-01'
dates_end = '1970-12-31'

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

rs = np.empty((daily_discharge.shape[0], len(cols)))
with open('rs' + station_p, 'rb') as f:
    ini_v = np.asarray(pkl.load(f))[-1]

rs = np.vstack((ini_v, rs))
strt_idx = 1
off_idx = 365

rs[strt_idx:, temp_col] = np.array(temp[station])
rs[strt_idx:, prec_col] = np.array(ppt[station])
rs[strt_idx:, pet_col] = np.array(pet[station])
rs[strt_idx:, Q_obs_col] = np.array(daily_discharge[station])

flow_diff = np.diff(rs[:, Q_obs_col])

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
print('flow_diff_contains = ', flow_diff_contains)
print('\n')
Q_sort_asc = np.sort(pos_arr)
Q_sort_des = -np.sort(-neg_arr)
R_indc = int((Q_sort_asc.size - (Ris_val_per / 100) * Q_sort_asc.size))
r_indc = int((Q_sort_des.size - (rec_val_per / 100) * Q_sort_des.size))
rising_limit = Q_sort_asc[R_indc]
recession_limit = Q_sort_des[r_indc]

print('rising_limit= ', rising_limit)
print('recession_limit= ', recession_limit)
print('\n')
rising_flag = True
recession_flag = True
threshhold_flag = True
# flag
if rising_flag:
    rissing_Beta = np.where(flow_diff >= rising_limit, 1, 0)
else:
    rissing_Beta = np.where(flow_diff >= rising_limit, 0, 0)
if recession_flag:
    recession_Beta = np.where(flow_diff <= recession_limit, 1, 0)
else:
    recession_Beta = np.where(flow_diff <= recession_limit, 0, 0)
if threshhold_flag:
    threshhold_Beta = (np.where((flow_diff > recession_limit) &
        (flow_diff < rising_limit), 1, 0))
else:
    threshhold_Beta = (np.where((flow_diff > recession_limit) &
        (flow_diff < rising_limit), 0, 0))

print('rising Beta_count =', np.count_nonzero(rissing_Beta))
print('recession Beta_count =', np.count_nonzero(recession_Beta))
print('threshhold Beta_count =', np.count_nonzero(threshhold_Beta))

P = np.genfromtxt(dir1 + '\\opt' + state + '_par' + station_c,
dtype=float, delimiter=',')

tt = P [2, 1]
cmelt = P [3, 1]
FC = P [4, 1]
Beta_1 = P [5, 1]
Beta_2 = P[6, 1]
Beta_3 = np.mean(np.array([Beta_1, Beta_2]))
PWP = P [7, 1]
L1 = P [8, 1]
K_uu = P [9, 1]
K_ul = P [10, 1]
K_d = P [11, 1]
k_ll = P [12, 1]

for i in range(strt_idx, rs.shape[0]):
    pr = rs[i - 1]
    cr = rs[i]

    if cr[Q_obs_col] - pr[Q_obs_col] >= rising_limit:
        Beta = Beta_1
    elif cr[Q_obs_col] - pr[Q_obs_col] <= recession_limit:
        Beta = Beta_2
    else:
        Beta = Beta_3

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
print('NS = ', NS)

q_simp = pd.read_pickle ('rs_fc' + station_p)

col = (qob_col, q_simq_col, q_simp_col) = (0, 1, 2)
ps = np.zeros((rs.shape[0], len(col)))
ps[:, qob_col] = rs[:, Q_obs_col]
ps[:, q_simq_col] = rs[:, Q_sim_col]
ps[:, q_simp_col] = np.array(q_simp['Q_sim'])

col_df = ('qob_col', 'q_simq_col', 'q_simp_col')
idx = pd.date_range(parser.parse(dates_start) + timedelta(days=-1), dates_end)
ps_df = pd.DataFrame(ps, columns=col_df, index=idx)

for i in range(int(ps_df.index[1].strftime('%Y')), int(ps_df.index[-1].strftime('%Y')) + 1, 2):
    df = ps_df.loc[str(i):str(i + 1), ['qob_col', 'q_simq_col', 'q_simp_col']]
    fig = df.plot(lw=1.5, alpha=0.5, figsize=(15, 3)).get_figure()
    plt.title('Qobs Vs Qsim ')
    plt.ylabel('Discharge', fontsize=12)
    plt.xlabel('Time', fontsize=12)
    plt.grid()
    plt.legend(loc='best')
    plt.show()
#     fig.savefig(out_dir+'\\comparison\\'+station+'\\Q_obsVsQ_sim_'+str(i))

ps_asc = ps[ps[:, qob_col].argsort()]
ps_df1 = pd.DataFrame(ps_asc, columns=col_df)

for i in range(ps_df1.index[0], ps_df1.index[-365], 365 * 2):
    df1 = ps_df1.iloc[i:i + (365 * 2), :]
    fig = df1.plot(lw=1.5, alpha=0.5, figsize=(15, 3)).get_figure()
    plt.title('Qobs Vs Qsim ')
    plt.ylabel('Discharge', fontsize=12)
    plt.xlabel('Time', fontsize=12)
    plt.grid()
    plt.legend(loc='best')
    plt.show()
#     fig.savefig(out_dir+'\\comparison\\'+station+'\\Q_obsVsQ_sim_sort_'+str(i))

# for i in range(0,ps_asc.shape[0]-365,365*2):
#     fig = plt.figure(figsize=(15, 3))
#     plt.plot(ps_asc[i:i+365*2,qob_col],lw=1.5,alpha=0.5,label='qob')
#     plt.plot(ps_asc[i:i+365*2,q_simq_col],lw=1.5,alpha=0.5,label='qsimq')
#     plt.plot(ps_asc[i:i+365*2,q_simp_col],lw=1.5,alpha=0.5,label='qsimp')
#     plt.legend()
#     plt.grid()
#     plt.show()
#     fig.savefig(out_dir+'\\comparison\\'+station+'\\Q_obsVsQ_sim_sortplt_'+str(i))
