'''
Created on 03.06.2019

@author: ullah
'''
import os
import numpy as np
import pandas as pd
import pickle as pkl
from dateutil import parser
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
r'UOS Germany/Thesis/Assignment 10/'))

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

cols = (prec_col, api_col) = (0, 1)
pps = np.zeros((ppt.shape[0], len(cols)))
pps[:, prec_col] = np.array(ppt[station])
strt_idx = 3
pmean = np.average(pps[strt_idx:, prec_col])


def objective(x):
    k = x[0]
    cp = pps[strt_idx, prec_col]
    pp1 = pps[strt_idx - 1, prec_col]
    pp2 = pps[strt_idx - 2, prec_col]
    pp3 = pps[strt_idx - 3, prec_col]
    api = (cp + (k * pp1) + ((k ** 2) * pp2) + ((k ** 3) * pp3))
    pps[strt_idx, api_col] = api

    for i in range(strt_idx + 1, ppt.shape[0]):
        pr = pps[i - 1]
        cr = pps[i]
        api = k * pr[api_col] + cr[prec_col]
        cr[api_col] = api
    p_pmean_sq = (pps[strt_idx:, prec_col] - pmean) ** 2
    api_p_sq = (pps[strt_idx:, api_col] - pps[strt_idx:, prec_col]) ** 2
    NS = 1 - (np.sum(api_p_sq) / np.sum(p_pmean_sq))
    return 1 - NS


# initial guess
n = 1
x0 = np.empty(n)
x0[0] = 0.90

# show initial objective
print('Initial NS: ' + str(1 - objective(x0)))

# optimize

x0_bounds = (0.8, 0.9)

bounds = [x0_bounds]

result = differential_evolution(objective, bounds)
x = result[0].x
# show final objective
print('Optimised NS: ' + str(1 - objective(x)))

# print (solution)
print('Solution')
print('k = ' + str(x[0]))

col = ('prec', 'api')
idx = pd.date_range(dates_start, dates_end)
pps_df = pd.DataFrame(pps, columns=col, index=idx)
pd.to_pickle(pps_df, 'pps' + station_p)
