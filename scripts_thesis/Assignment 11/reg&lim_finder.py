'''
Created on 03.06.2019

@author: ullah
'''
import os
import numpy as np
import pandas as pd
import pickle as pkl
from dateutil import parser
import matplotlib.pyplot as plt
from sklearn import linear_model
import statsmodels.formula.api as sm
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
dates_start = '1961-01-01'
dates_end = '2015-12-31'

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

# Files Naming
(station_p, station_d, station_c, station_j, station_e) = ('_' + station
+'.pkl', '_' + station + '.pdf', '_' + station + '.csv',
'_' + station + '.jpg', '_' + station + '.xlsx')

parameters = pd.read_csv(dir1 + '\\opt_full_par' + station_c, index_col=0)
Beta_1 = parameters.at['Beta_1', 'val']
Beta_2 = parameters.at['Beta_2', 'val']
Beta_3 = np.mean([Beta_1, Beta_2])

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
# Flags
if rising_flag:
    rissing_Beta = np.where(flow_diff >= rising_limit, Beta_1, 0)
else:
    rissing_Beta = np.where(flow_diff >= rising_limit, 0, 0)
if recession_flag:
    recession_Beta = np.where(flow_diff <= recession_limit, Beta_2, 0)
else:
    recession_Beta = np.where(flow_diff <= recession_limit, 0, 0)
if threshhold_flag:
    threshhold_Beta = np.where((flow_diff > recession_limit) &
    (flow_diff < rising_limit), Beta_3, 0)
else:
    threshhold_Beta = np.where((flow_diff > recession_limit) &
    (flow_diff < rising_limit), 0, 0)

print('rising values =', np.count_nonzero(rissing_Beta))
print('recession values =', np.count_nonzero(recession_Beta))
print('threshhold values =', np.count_nonzero(threshhold_Beta))
print('\n')
Betas = rissing_Beta + recession_Beta + threshhold_Beta
with open('beta_arr' + station_p, 'wb') as f:
    pkl.dump(Betas, f)
with open('ris_lim' + station_p, 'wb') as f:
    pkl.dump(rising_limit, f)
with open('rec_lim' + station_p, 'wb') as f:
    pkl.dump(recession_limit, f)

cols = (temp_col, prec_col, pet_col, Betas_col) = (0, 1, 2, 3)
rs = np.zeros((daily_discharge.shape[0], len(cols)))
strt_idx = 0

rs[strt_idx:, temp_col] = np.array(temp[station])
rs[strt_idx:, prec_col] = np.array(ppt[station])
rs[strt_idx:, pet_col] = np.array(pet[station])
rs[strt_idx:, Betas_col] = Betas

col = ('temp', 'prec', 'pet', 'Betas')
idx = pd.date_range(dates_start, dates_end)
rs_df = pd.DataFrame(rs, columns=col, index=idx)
result = sm.ols(formula="Betas ~ prec + temp + pet", data=rs_df).fit()
print(result.params)
print(result.summary())
pd.to_pickle(result.params, 'result_params' + station_p)

reg = linear_model.LinearRegression()
reg.fit(rs_df[['prec', 'temp', 'pet']], rs_df['Betas'])
print(reg.coef_)

rs_df['comp_Betas'] = (rs_df['temp'] * result.params['temp'] +
rs_df['prec'] * result.params['prec'] + rs_df['pet'] * result.params['pet'] +
result.params['Intercept'])
print(rs_df)
