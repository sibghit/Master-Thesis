'''
Created on 11.06.2019

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
r'UOS Germany/Thesis/Assignment 12/'))

dates = pd.date_range('1961-01-01', '2015-12-31')
names = ['420', '3421', '3465', '3470']
station = '3465'
dates_start = '1966-01-01'
dates_end = '1970-12-31'

pet = (pd.read_csv(dir + '\\rockenau_pet_1961_2015_lumped.csv',
sep=';', parse_dates=[1], index_col=0, nrows=dates.size))
pet = pet.loc[dates_start:dates_end, [station]]

ppt = (pd.read_csv(dir + '\\rockenau_ppt_1961_2015_lumped.csv',
sep=';', parse_dates=[1], index_col=0, nrows=dates.size))
ppt = ppt.loc[dates_start:dates_end, [station]]

temp = (pd.read_csv(dir + '\\rockenau_tem_1961_2015_lumped.csv',
sep=';', parse_dates=[1], index_col=0, nrows=dates.size))
temp = temp.loc[dates_start:dates_end, [station]]

daily_discharge = (pd.read_csv(dir + '\\neckar_daily_discharge_1961_2015.csv',
sep=';', parse_dates=[1], index_col=0))
daily_discharge = daily_discharge.loc[dates_start:dates_end, [station]]

watersheds = (pd.read_csv(dir + '\\watersheds_cumm_cat_areas.csv', sep=';',
index_col=0))

# Files Naming
(station_p, station_d, station_c, station_j) = ('_' + station + '.pkl',
'_' + station + '.pdf', '_' + station + '.csv', '_' + station + '.jpg')
state = '_full'

Area = (watersheds.loc[float(station), ['cumm_area']]).values

pps = (pd.read_pickle('pps' + station_p))
api = pps.loc[dates_start:dates_end, ['api']]

cols = (temp_col, prec_col, pet_col, snow_col, not_snow_col, am_col, sm_col,
et_col, rn_col, us_col, ul_col, rn_uu_col, rn_ul_col, rn_d_col, rn_l_col,
Q_obs_col, Q_sim_col, api_col) = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
11, 12, 13, 14, 15, 16, 17)

rs = np.zeros((ppt.shape[0], len(cols)))

with open('rs' + station_p, 'rb') as f:
    ini_v = np.asarray(pkl.load(f))[-1]

rs = np.vstack((ini_v, rs))
strt_idx = 1
off_idx = 0

rs[strt_idx:, temp_col] = np.array(temp[station])
rs[strt_idx:, prec_col] = np.array(ppt[station])
rs[strt_idx:, pet_col] = np.array(pet[station])
rs[strt_idx:, Q_obs_col] = np.array(daily_discharge[station])
rs[strt_idx:, api_col] = np.array(api['api'])

rising_limit = pd.read_pickle('ris_lim' + station_p)
recession_limit = pd.read_pickle('rec_lim' + station_p)
result_params = pd.read_pickle('result_params' + station_p)
Intercept = result_params['Intercept']
prec_coef = result_params['prec']
temp_coef = result_params['temp']
pet_coef = result_params['pet']

print('rising_limit= ', rising_limit)
print('recession_limit= ', recession_limit)

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

#     interest = prec_col
    interest = api_col
#     interest = Q_obs_col

    if cr[interest] - pr[interest] >= rising_limit:
        Beta = Beta_1
    elif cr[interest] - pr[interest] <= recession_limit:
        Beta = Beta_2
    else:
        Beta = Beta_3

#     Beta = ((prec_coef * cr[prec_col]) + (temp_coef * cr[temp_col]) +
#     (pet_coef * cr[pet_col]) + Intercept)

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

y_snow = rs[strt_idx - 1:, snow_col]
y_not_snow = rs[strt_idx:, not_snow_col]
y_am = rs[strt_idx:, am_col]
y_sm = rs[strt_idx - 1:, sm_col]
y_et = rs[strt_idx:, et_col]
y_rn_uu = rs[strt_idx:, rn_uu_col]
y_rn = rs[strt_idx:, rn_col]
y_us = rs[strt_idx - 1:, us_col]
y_ul = rs[strt_idx - 1:, ul_col]
y_rn_ul = rs[strt_idx:, rn_ul_col]
y_Q_obs = rs[strt_idx:, Q_obs_col]
y_rn_l = rs[strt_idx:, rn_l_col]
y_rn_d = rs[strt_idx:, rn_d_col]
y_Q_tot = Qmm
y_Q_sim = rs[strt_idx:, Q_sim_col]

# fig = plt.figure(figsize=(15, 30))
fig = plt.figure()
plt.subplot(14, 1, 1)
plt.plot(y_snow, lw=0.5, label="snow")
# plt.title('Snow')
plt.legend()
plt.subplot(14, 1, 2)
plt.plot(y_not_snow, lw=0.5, label="not_snow")
# plt.title ('available liquid water')
plt.legend()
plt.subplot(14, 1, 3)
plt.plot(y_am, lw=0.5, label="am")
# plt.title ('available soil moisture')
plt.legend()
plt.subplot(14, 1, 4)
plt.plot(y_sm, lw=0.5, label="sm")
# plt.title ('Soil Mositure')
plt.legend()
plt.subplot(14, 1, 5)
plt.plot(y_et, lw=0.5, label="et")
# plt.title ('ET')
plt.legend()
plt.subplot(14, 1, 6)
plt.plot(y_rn_uu, lw=0.5, label="rn_uu")
# plt.title('Q_uu')
plt.legend()
plt.subplot(14, 1, 7)
plt.plot(y_rn, lw=0.5, label="rn")
# plt.title ('Potential Runoff')
plt.legend()
plt.subplot(14, 1, 8)
plt.plot(y_us, lw=0.5, label="us")
# plt.title ('S1')
plt.legend()
plt.subplot(14, 1, 9)
plt.plot(y_ul, lw=0.5, label="ul")
# plt.title('S2')
plt.legend()
plt.subplot(14, 1, 10)
plt.plot(y_rn_ul, lw=0.5, label="rn_ul")
# plt.title ('Q_ul')
plt.legend()
plt.subplot(14, 1, 11)
plt.plot(y_Q_tot, lw=0.5, label="Q_tot")
# plt.title ('Q_tot')
plt.legend()
plt.subplot(14, 1, 12)
plt.plot(y_rn_l, lw=0.5, label="rn_l")
# plt.title('Q_ll')
plt.legend()
plt.subplot(14, 1, 13)
plt.plot(y_rn_d, lw=0.5, label="rn_d")
# plt.title ('Q_d')
plt.legend()
plt.subplot(14, 1, 14)
plt.plot(y_Q_sim, alpha=0.5, lw=0.5, label="Q_sim")
plt.plot(y_Q_obs, alpha=0.5, lw=0.5, label="Q_obs")
# plt.title ('Comparison')
plt.legend()
# plt.show()
# plots=plt.show()
# plt.show()
fig.savefig(out_dir + '\\graphs\\plots_fc' + station_d)
print("\n")

# Mass balance check
inflow = np.sum(rs[strt_idx:, prec_col])
ET = np.sum (rs[strt_idx:, et_col])
Q = np.sum (Qmm)
outflow = ET + Q
print("inflow (precipitaion) = ", inflow)
print("outflow (ET+Q) =", outflow)
print ("delta =", inflow - outflow)
print("\n")
fig = plt.figure()
y_inflow = rs[strt_idx:, prec_col]
y_outflow = rs[strt_idx:, et_col] + Qmm
comm_inflows = np.cumsum(y_inflow)
comm_outflows = np.cumsum(y_outflow)
plt.plot(comm_inflows, lw=1.5, alpha=0.5, label="I/F (prec_cumulative)")
plt.plot(comm_outflows, lw=1.5, alpha=0.5, label="O/F (cumulative(Qmm+et))")
plt.title ('Mass balance check')
plt.grid()
plt.legend()
# plt.show()
fig.savefig(out_dir + '\\graphs\\comm_inflow_vs_comm_outflow_fc' + station_d)
print('NS = ', NS)
col = ('temp', 'prec', 'pet', 'snow', 'not_snow', 'am', 'sm', 'et',
'rn', 'us', 'ul', 'rn_uu', 'rn_ul', 'rn_d', 'rn_l', 'Q_obs', 'Q_sim', 'api')

idx = pd.date_range(parser.parse(dates_start) + timedelta(days=-1) ,
dates_end)
hbv = pd.DataFrame(rs, columns=col, index=idx)

pd.to_pickle(hbv, 'rs_fc' + station_p)
