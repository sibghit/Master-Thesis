'''
Created on 2.07.2019

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
r'UOS Germany/Thesis/Assignment 15/'))

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

aps = np.zeros((ppt.shape[0] , len(cols)))
crs = np.zeros(len(cols) + 1)

aps[:, prec_col] = np.array(ppt[station])
aps[:, Q_obs_col] = np.array(daily_discharge[station])

off_idx = 30

for s in range(3, len(cols) + 1):

    def objective(x):
        k = x[0]

        for i in range(s, aps.shape[0]):
            (p, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14) = (aps[i, prec_col],
            aps[i - 1, prec_col], aps[i - 2, prec_col], aps[i - 3, prec_col], aps[i - 4, prec_col], aps[i - 5, prec_col], aps[i - 6, prec_col],
            aps[i - 7, prec_col], aps[i - 8, prec_col], aps[i - 9, prec_col], aps[i - 10, prec_col], aps[i - 11, prec_col], aps[i - 12, prec_col], aps[i - 13, prec_col],
            aps[i - 14, prec_col])

            api_i = [(p + (k) * p1 + (k ** 2) * p2 + (k ** 3) * p3),
            (p + (k) * p1 + (k ** 2) * p2 + (k ** 3) * p3 + (k ** 4) * p4),
            (p + (k) * p1 + (k ** 2) * p2 + (k ** 3) * p3 + (k ** 4) * p4 + (k ** 5) * p5),
            (p + (k) * p1 + (k ** 2) * p2 + (k ** 3) * p3 + (k ** 4) * p4 + (k ** 5) * p5 + (k ** 6) * p6),
            (p + (k) * p1 + (k ** 2) * p2 + (k ** 3) * p3 + (k ** 4) * p4 + (k ** 5) * p5 + (k ** 6) * p6 + (k ** 7) * p7),
            (p + (k) * p1 + (k ** 2) * p2 + (k ** 3) * p3 + (k ** 4) * p4 + (k ** 5) * p5 + (k ** 6) * p6 + (k ** 7) * p7 + (k ** 8) * p8),
            (p + (k) * p1 + (k ** 2) * p2 + (k ** 3) * p3 + (k ** 4) * p4 + (k ** 5) * p5 + (k ** 6) * p6 + (k ** 7) * p7 + (k ** 8) * p8 + (k ** 9) * p9),
            (p + (k) * p1 + (k ** 2) * p2 + (k ** 3) * p3 + (k ** 4) * p4 + (k ** 5) * p5 + (k ** 6) * p6 + (k ** 7) * p7 + (k ** 8) * p8 + (k ** 9) * p9 + (k ** 10) * p10),
            (p + (k) * p1 + (k ** 2) * p2 + (k ** 3) * p3 + (k ** 4) * p4 + (k ** 5) * p5 + (k ** 6) * p6 + (k ** 7) * p7 + (k ** 8) * p8 + (k ** 9) * p9 + (k ** 10) * p10 + (k ** 11) * p11),
            (p + (k) * p1 + (k ** 2) * p2 + (k ** 3) * p3 + (k ** 4) * p4 + (k ** 5) * p5 + (k ** 6) * p6 + (k ** 7) * p7 + (k ** 8) * p8 + (k ** 9) * p9 + (k ** 10) * p10 + (k ** 11) * p11 + (k ** 12) * p12),
            (p + (k) * p1 + (k ** 2) * p2 + (k ** 3) * p3 + (k ** 4) * p4 + (k ** 5) * p5 + (k ** 6) * p6 + (k ** 7) * p7 + (k ** 8) * p8 + (k ** 9) * p9 + (k ** 10) * p10 + (k ** 11) * p11 + (k ** 12) * p12 + (k ** 13) * p13),
            (p + (k) * p1 + (k ** 2) * p2 + (k ** 3) * p3 + (k ** 4) * p4 + (k ** 5) * p5 + (k ** 6) * p6 + (k ** 7) * p7 + (k ** 8) * p8 + (k ** 9) * p9 + (k ** 10) * p10 + (k ** 11) * p11 + (k ** 12) * p12 + (k ** 13) * p13 + (k ** 14) * p14)]

            api = api_i[s - 3]
            aps[i, s - 1] = api
        corr = np.corrcoef(aps[off_idx:, Q_obs_col], aps[off_idx:, s - 1])[0, 1]
        crs[s] = corr
        return(1 - corr)

    # optimize
    x0_bounds = (0, 1)

    bounds = [x0_bounds]

    result = differential_evolution(objective, bounds)
    x = result[0].x
    # show final objective
    print('Optimised_corr_' + str(s) + ': ' + str(1 - objective(x)))

    # print (solution)
    print('k = ' + str(x[0]))
    print('\n')

crs = list(crs)
columns = ('prec', 'Q_obs', 'api_3', 'api_4', 'api_5', 'api_6',
'api_7', 'api_8', 'api_9', 'api_10', 'api_11', 'api_12',
'api_13', 'api_14')
idx = pd.date_range(dates_start, dates_end)
aps_df = pd.DataFrame(aps, columns=columns, index=idx)
pd.to_pickle(aps_df, 'aps' + station_p)
aps_df.to_excel('aps.xlsx', sheet_name='Sheet1')
print('j: ', crs.index(max(crs[3:])))
print(aps_df)
fig = plt.figure()
x_axis = np.arange(3, len(cols) + 1)
plt.plot(x_axis, crs[3:])
plt.title('days Vs Correl (Q, API)')
plt.xlabel('days')
plt.ylabel('correl')
plt.grid()
plt.show()
fig.savefig(out_dir + '\\graphs' + 'api_q' + station_d)
