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

# beta_arr = beta_arr.loc[dates_start:dates_end, [station]]
parameters = pd.read_csv(dir1 + '\\opt_full_par' + station_c, index_col=0)
Beta_1 = parameters.at['Beta_1', 'val']
Beta_2 = parameters.at['Beta_2', 'val']
Beta_3 = np.mean([Beta_1, Beta_2])

ppt = (pd.read_csv(dir + '\\rockenau_ppt_1961_2015_lumped.csv',
sep=';', parse_dates=[1], index_col=0, nrows=dates.size))
ppt = ppt.loc[dates_start:dates_end, [station]]

daily_discharge = (pd.read_csv(dir + '\\neckar_daily_discharge_1961_2015.csv',
sep=';', parse_dates=[1], index_col=0))
daily_discharge = daily_discharge.loc[dates_start:dates_end, [station]]

pet = (pd.read_csv(dir + '\\rockenau_pet_1961_2015_lumped.csv',
sep=';', parse_dates=[1], index_col=0, nrows=dates.size))
pet = pet.loc[dates_start:dates_end, [station]]

temp = (pd.read_csv(dir + '\\rockenau_tem_1961_2015_lumped.csv',
sep=';', parse_dates=[1], index_col=0, nrows=dates.size))
temp = temp.loc[dates_start:dates_end, [station]]

cols = (prec_col, api_col, api_0_col, api_1_col, betas_col,
betas_sim_col) = (0, 1, 2, 3, 4, 5)
pps = np.zeros((ppt.shape[0], len(cols)))

pps[:, prec_col] = np.array(ppt[station])
pps[:, betas_col] = np.array(beta_arr['betas_org'])

n_steps = 14

bmean = np.average(pps[n_steps:, betas_col])

k = 0

for i in range(n_steps, pps.shape[0]):
    api_step = 0.0
    for j in range(i - n_steps, i + 1):
        api_step += (pps[j, prec_col] * (k ** (i - j)))
    pps[i, api_0_col] = api_step

api_diff_0 = np.diff(pps[:, api_0_col])
(ll_0, ul_0) = (np.min(api_diff_0), np.max(api_diff_0))

k = 1

for i in range(n_steps, pps.shape[0]):
    api_step = 0.0
    for j in range(i - n_steps, i + 1):
        api_step += (pps[j, prec_col] * (k ** (i - j)))
    pps[i, api_1_col] = api_step
api_diff_1 = np.diff(pps[:, api_1_col])
(ll_1, ul_1) = (np.min(api_diff_1), np.max(api_diff_1))


def objective(x):
    k = x[0]
    rising_limit = x[1]
    recession_limit = x[2]
    scale_power_b1 = x[3]
    scale_power_b2 = x[4]
#     shift_steps = int(x[5])
#     dividing_api = x[5]

#     print(
#         '%+7.4f, %+7.4f, %+7.4f, %+7.4f, %+7.4f, %0.2d' % (
#         x[0], x[1], x[2], x[3], x[4], int(x[5])))

    print(
        '%+7.4f, %+7.4f, %+7.4f, %+7.4f, %+7.4f' % (
        x[0], x[1], x[2], x[3], x[4]))

    for i in range(n_steps, pps.shape[0]):
        api_step = 0.0
        for j in range(i - n_steps, i + 1):
            api_step += (pps[j, prec_col] * (k ** (i - j)))

#         if api_step < dividing_api:
#             api_step = np.log(api_step)
#
#         else:
#             api_step = np.exp(api_step)

        pps[i, api_col] = api_step

    for i in range(n_steps, pps.shape[0]):
        if ((pps[i , api_col] - pps[i - 1, api_col]) ** scale_power_b1) >= rising_limit:
            Beta = Beta_1
        elif ((pps[i , api_col] - pps[i - 1, api_col]) ** scale_power_b2) <= recession_limit:
            Beta = Beta_2
        else:
            Beta = Beta_3

        pps[i, betas_sim_col] = Beta

    b_bmean_sq = (pps[n_steps:, betas_col] - bmean) ** 2
    bsim_b_sq = (pps[n_steps:, betas_sim_col] - pps[n_steps:, betas_col]) ** 2
    NS = 1 - (np.sum(bsim_b_sq) / np.sum(b_bmean_sq))
    return 1 - NS
#     corr = np.corrcoef(pps[n_steps:, betas_col], pps[n_steps:, betas_sim_col])[0, 1]
#     return 1 - corr


# optimize
x0_bounds = (0, 1)  # k#
x1_bounds = (0, max(ul_0, ul_1))  # rising_limit#
x2_bounds = (min(ll_0, ll_1), 0)  # recession_limit#
# x1_bounds = (0, 1000) #rising_limit#
# x2_bounds = (-1000, 0) #recession_limit#

# x3_bounds = (-4, +4) #scale_power_b1#
x3_bounds = (1, 1)  # scale_power_b1#
x4_bounds = (1, 1)  # x4_bounds = (-4, +4) #scale_power_b2#

# x5_bounds = (2, min(n_steps, 5)) #shift_steps#
#
# assert x5_bounds[0] < x5_bounds[1]

# x5_bounds = (0, 105)  # shift_steps#
bounds = [x0_bounds, x1_bounds, x2_bounds, x3_bounds, x4_bounds]
# bounds = [x0_bounds, x1_bounds, x2_bounds, x3_bounds, x4_bounds]

# result = differential_evolution(objective, bounds, maxiter=30, popsize=100)
result = differential_evolution(objective, bounds)
x = result[0].x
# show final objective
print('Optimised : ' + str(1 - objective(x)))

# print (solution)
print('Solution')
print('k = ' + str(x[0]))
print('rising_limit = ' + str(x[1]))
print('recession_limit = ' + str(x[2]))
# print('scale_power_b1 = ' + str(x[3]))
# print('scale_power_b2 = ' + str(x[4]))
# print('shift_steps = ' + str(x[5]))
# print('dividing_api = ' + str(x[6]))

print('\n')
betas_org_contains = {'Beta_1':np.count_nonzero(pps[:, betas_col] == Beta_1),
'Beta_2':np.count_nonzero(pps[:, betas_col] == Beta_2),
'Beta_3':np.count_nonzero(pps[:, betas_col] == Beta_3)}
print('betas_org_contains = ', betas_org_contains)

betas_sim_contains = {'Beta_1':np.count_nonzero(pps[:, betas_sim_col] == Beta_1),
'Beta_2':np.count_nonzero(pps[:, betas_sim_col] == Beta_2),
'Beta_3':np.count_nonzero(pps[:, betas_sim_col] == Beta_3)}
print('betas_sim_contains = ', betas_sim_contains)

# col = ('prec', 'api', 'api_0', 'api_1', 'betas', 'betas_sim')
# idx = pd.date_range(dates_start, dates_end)
# pps_df = pd.DataFrame(pps, columns=col, index=idx)
# pd.to_pickle(pps_df, 'pps' + station_p)
# pps_df.to_excel('pps.xlsx', sheet_name='Sheet1')
#
# df = pd.DataFrame({'params':['k', 'rising_limit', 'recession_limit', 'scale_power'], 'val': x})
# df.set_index('params', inplace=True)
# df.to_pickle('opt_k_ris_rec' + station_p)
#
# with open('itarr_k_ris_rec' + station_p, 'wb') as f:
#     pkl.dump(result[1], f)
#
# print(pps_df)
# print(result)
# print(x1_bounds, x2_bounds)
#
# pps_df['Q_obs'] = daily_discharge
# pps_df['pet'] = pet
# pps_df['temp'] = temp
#
# corr = pps_df.corr()
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
# fig.colorbar(cax)
# ticks = np.arange(0, len(pps_df.columns), 1)
# ax.set_xticks(ticks)
# plt.xticks(rotation=90)
# ax.set_yticks(ticks)
# ax.set_xticklabels(pps_df.columns)
# ax.set_yticklabels(pps_df.columns)
# plt.show()

