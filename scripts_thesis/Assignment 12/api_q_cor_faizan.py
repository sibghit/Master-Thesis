'''
Created on 14.06.2019

@author: ullah
'''
import os
import numpy as np
import pandas as pd
import pickle as pkl
from scipy import optimize
from dateutil import parser
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import differential_evolution

uni_path = ('C:\\Users\\ullah\\OneDrive\\')
home_path = ('O:\\')

path = uni_path

dir = os.path.abspath(os.path.join(path,
r'UOS Germany/Thesis/sibghat_neckar_data/'))
dir1 = os.path.abspath(os.path.join(path,
r'scripts_cloud/Assignment 9/old_model'))
out_dir = os.path.abspath(os.path.join(path,
r'UOS Germany/Thesis/Assignment 12/'))

dates = pd.date_range('1961-01-01', '2015-12-31')
names = ['420', '3421', '3465', '3470']
station = '3465'
dates_start = '1961-01-01'
dates_end = '2015-12-31'

# Files Naming
(station_p, station_d, station_c, station_j, station_e) = ('_' + station
+'.pkl', '_' + station + '.pdf', '_' + station + '.csv',
'_' + station + '.jpg', '_' + station + '.xlsx')
state = '_full'

ppt = (pd.read_csv(dir + '\\rockenau_ppt_1961_2015_lumped.csv',
sep=';', parse_dates=[1], index_col=0, nrows=dates.size))
ppt = ppt.loc[dates_start:dates_end, [station]]

daily_discharge = (pd.read_csv(dir + '\\neckar_daily_discharge_1961_2015.csv',
sep=';', parse_dates=[1], index_col=0))
daily_discharge = daily_discharge.loc[dates_start:dates_end, [station]]

cols = (prec_col, Q_obs_col, api_3_col, api_4_col, api_5_col, api_6_col,
api_7_col, api_8_col, api_9_col, api_10_col, api_11_col, api_12_col,
api_13_col, api_14_col) = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)

aps = np.zeros((ppt.shape[0], len(cols)))

aps[:, prec_col] = np.array(ppt[station])
aps[:, Q_obs_col] = np.array(daily_discharge[station])

precipitations = (p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13,
 p14, p15) = (aps[:15, prec_col])


def objective(x):

    k = x[0]

#     api_ini = [(p4 + (k) * p3 + (k ** 2) * p2 + (k ** 3) * p1),
#     (p5 + (k) * p4 + (k ** 2) * p3 + (k ** 3) * p2 + (k ** 4) * p1),
#     (p6 + (k) * p5 + (k ** 2) * p4 + (k ** 3) * p3 + (k ** 4) * p2 + (k ** 5) * p1),
#     (p7 + (k) * p6 + (k ** 2) * p5 + (k ** 3) * p4 + (k ** 4) * p3 + (k ** 5) * p2 + (k ** 6) * p1),
#     (p8 + (k) * p7 + (k ** 2) * p6 + (k ** 3) * p5 + (k ** 4) * p4 + (k ** 5) * p3 + (k ** 6) * p2 + (k ** 7) * p1),
#     (p9 + (k) * p8 + (k ** 2) * p7 + (k ** 3) * p6 + (k ** 4) * p5 + (k ** 5) * p4 + (k ** 6) * p3 + (k ** 7) * p2 + (k ** 8) * p1),
#     (p10 + (k) * p9 + (k ** 2) * p8 + (k ** 3) * p7 + (k ** 4) * p6 + (k ** 5) * p5 + (k ** 6) * p4 + (k ** 7) * p3 + (k ** 8) * p2 + (k ** 9) * p1),
#     (p11 + (k) * p10 + (k ** 2) * p9 + (k ** 3) * p8 + (k ** 4) * p7 + (k ** 5) * p6 + (k ** 6) * p5 + (k ** 7) * p4 + (k ** 8) * p3 + (k ** 9) * p2 + (k ** 10) * p1),
#     (p12 + (k) * p11 + (k ** 2) * p10 + (k ** 3) * p9 + (k ** 4) * p8 + (k ** 5) * p7 + (k ** 6) * p6 + (k ** 7) * p5 + (k ** 8) * p4 + (k ** 9) * p3 + (k ** 10) * p2 + (k ** 11) * p1),
#     (p13 + (k) * p12 + (k ** 2) * p11 + (k ** 3) * p10 + (k ** 4) * p9 + (k ** 5) * p8 + (k ** 6) * p7 + (k ** 7) * p6 + (k ** 8) * p5 + (k ** 9) * p4 + (k ** 10) * p3 + (k ** 11) * p2 + (k ** 12) * p1),
#     (p14 + (k) * p13 + (k ** 2) * p12 + (k ** 3) * p11 + (k ** 4) * p10 + (k ** 5) * p9 + (k ** 6) * p8 + (k ** 7) * p7 + (k ** 8) * p6 + (k ** 9) * p5 + (k ** 10) * p4 + (k ** 11) * p3 + (k ** 12) * p2 + (k ** 13) * p1),
#     (p15 + (k) * p14 + (k ** 2) * p13 + (k ** 3) * p12 + (k ** 4) * p11 + (k ** 5) * p10 + (k ** 6) * p9 + (k ** 7) * p8 + (k ** 8) * p7 + (k ** 9) * p6 + (k ** 10) * p5 + (k ** 11) * p4 + (k ** 12) * p3 + (k ** 13) * p2 + (k ** 14) * p1)]

    for s in range(3, len(precipitations)):
#         api = api_ini[s - 3]
#         aps[s, s - 1] = api

        for i in range(s + 1, aps.shape[0]):
            pr = aps[i - 1]
            cr = aps[i]
            api = (k * pr[s - 1]) + cr[prec_col]
            cr[s - 1] = api
    cols = ('prec', 'Q_obs', 'api_3', 'api_4', 'api_5', 'api_6',
    'api_7', 'api_8', 'api_9', 'api_10', 'api_11', 'api_12',
    'api_13', 'api_14')
    idx = pd.date_range(dates_start, dates_end)
    aps_df = pd.DataFrame(aps, columns=cols, index=idx)

    corrs = []
    for col in cols[2:]:
        corrs.append(np.corrcoef(
            aps_df['Q_obs'].values[10:],
            aps_df[col].values[10:])[0, 1])

#     corr3 = aps_df['Q_obs'].corr(aps_df['api_3'])
#     corr4 = aps_df['Q_obs'].corr(aps_df['api_4'])
#     corr5 = aps_df['Q_obs'].corr(aps_df['api_5'])
#     corr6 = aps_df['Q_obs'].corr(aps_df['api_6'])
#     corr7 = aps_df['Q_obs'].corr(aps_df['api_7'])
#     corr8 = aps_df['Q_obs'].corr(aps_df['api_8'])
#     corr9 = aps_df['Q_obs'].corr(aps_df['api_9'])
#     corr10 = aps_df['Q_obs'].corr(aps_df['api_10'])
#     corr11 = aps_df['Q_obs'].corr(aps_df['api_11'])
#     corr12 = aps_df['Q_obs'].corr(aps_df['api_12'])
#     corr13 = aps_df['Q_obs'].corr(aps_df['api_13'])
#     corr14 = aps_df['Q_obs'].corr(aps_df['api_14'])
#     return (1 - corr3, 1 - corr4, 1 - corr5, 1 - corr6, 1 - corr7, 1 - corr8,
#     1 - corr9, 1 - corr10, 1 - corr11, 1 - corr12, 1 - corr13, 1 - corr14)
#     return (corr3, corr4, corr5, corr6, corr7, corr8, corr9, corr10, corr11,
#     corr12, corr13, corr14)
#     return (1 - corr14)#
    return corrs


# initial guess
n = 1
x0 = np.empty(n)
x0[0] = 0.5

# show initial objective
print('Initial corr: ' + str(objective(x0)))
# print('ini_j: ', (objective(x0).index(max(objective(x0))) + 3))

# # optimize
# x0_bounds = (0, 1)
#
# bounds = [x0_bounds]
#
# # result = differential_evolution(objective, bounds, maxiter=30, popsize=100)
# result = differential_evolution(objective, bounds)
# x = result[0].x
# # show final objective
# print('Optimised corr: ' + str(1 - objective(x)))
# # print('opt_j: ', (objective(x).index(max(objective(x))) + 3))
# # print (solution)
# print('Solution')
# print('k = ' + str(x[0]))
#
# col = ('prec', 'Q_obs', 'api_3', 'api_4', 'api_5', 'api_6',
# 'api_7', 'api_8', 'api_9', 'api_10', 'api_11', 'api_12',
# 'api_13', 'api_14')
# idx = pd.date_range(dates_start, dates_end)
# aps_df = pd.DataFrame(aps, columns=col, index=idx)
# pd.to_pickle(aps_df, 'aps' + station_p)
# aps_df.to_excel('aps.xlsx', sheet_name='Sheet1')
# print(aps_df)

