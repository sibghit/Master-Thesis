'''
Created on 2.07.2019

@author: ullah
'''
import os
import numpy as np
import pandas as pd
import pickle as pkl
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

beta_arr = pd.read_pickle('beta_arr' + station_p)
beta_arr = np.hstack((0, (np.array(beta_arr['betas_org']))))

parameters = pd.read_csv(dir1 + '\\opt_full_par' + station_c, index_col=0)
Beta_1 = parameters.at['Beta_1', 'val']
Beta_2 = parameters.at['Beta_2', 'val']
Beta_3 = np.mean([Beta_1, Beta_2])

rs = pd.read_pickle('rs_full' + station_p)
rs['beta_arr'] = np.asarray(beta_arr)

b1 = rs.loc[rs.beta_arr == Beta_1]
b2 = rs.loc[rs.beta_arr == Beta_2]
b3 = rs.loc[rs.beta_arr == Beta_3]

(pm1, pM1) = (b1['prec'].min(), b1['prec'].max())
(pm2, pM2) = (b2['prec'].min(), b2['prec'].max())
(pm3, pM3) = (b3['prec'].min(), b3['prec'].max())

print('prec range @ Beta_1: ' , (pm1, pM1))
print('prec range @ Beta_2: ' , (pm2, pM2))
print('prec range @ Beta_3: ' , (pm3, pM3))

(pm1, pM1) = (b1['snow'].min(), b1['snow'].max())
(pm2, pM2) = (b2['snow'].min(), b2['snow'].max())
(pm3, pM3) = (b3['snow'].min(), b3['snow'].max())

print('snow range @ Beta_1: ' , (pm1, pM1))
print('snow range @ Beta_2: ' , (pm2, pM2))
print('snow range @ Beta_3: ' , (pm3, pM3))

(pm1, pM1) = (b1['not_snow'].min(), b1['not_snow'].max())
(pm2, pM2) = (b2['not_snow'].min(), b2['not_snow'].max())
(pm3, pM3) = (b3['not_snow'].min(), b3['not_snow'].max())

print('not_snow range @ Beta_1: ' , (pm1, pM1))
print('not_snow range @ Beta_2: ' , (pm2, pM2))
print('not_snow range @ Beta_3: ' , (pm3, pM3))

