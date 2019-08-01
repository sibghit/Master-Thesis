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
out_dir = os.path.abspath(os.path.join(path,
r'UOS Germany/Thesis/Assignment 16/'))

dates = pd.date_range('1961-01-01', '2015-12-31')
names = ['420', '3421', '3465', '3470']
station = '3465'
dates_start = '1966-01-01'
dates_end = '1970-12-31'

# Files Naming
(station_p, station_d, station_c, station_j) = ('_' + station + '.pkl',
'_' + station + '.pdf', '_' + station + '.csv', '_' + station + '.jpg')
state = '_full'

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

beta_org = pd.read_pickle('beta_arr' + station_p).loc[dates_start:dates_end]

Area = (watersheds.loc[float(station), ['cumm_area']]).values

cols = (temp_col, prec_col, pet_col, snow_col, not_snow_col, am_col, sm_col,
et_col, rn_col, us_col, ul_col, rn_uu_col, rn_ul_col, rn_d_col, rn_l_col,
Q_obs_col, Q_sim_col) = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
11, 12, 13, 14, 15, 16)

rs = np.zeros((daily_discharge.shape[0], len(cols)))

with open('rs' + station_p, 'rb') as f:
    ini_v = np.asarray(pkl.load(f))[-1]

rs = np.vstack((ini_v, rs))
strt_idx = 1
off_idx = 1

rs[strt_idx:, temp_col] = np.array(temp[station])
rs[strt_idx:, prec_col] = np.array(ppt[station])
rs[strt_idx:, pet_col] = np.array(pet[station])
rs[strt_idx:, Q_obs_col] = np.array(daily_discharge[station])

rising_limit = pd.read_pickle('ris_lim' + station_p)
recession_limit = pd.read_pickle('rec_lim' + station_p)

print('rising_limit= ', rising_limit)
print('recession_limit= ', recession_limit)

parameters = pd.read_csv(dir1 + '\\opt_full_par' + station_c, index_col=0)

tt = parameters.at['tt', 'val']
cmelt = parameters.at['cmelt', 'val']
FC = parameters.at['FC', 'val']
Beta_1 = parameters.at['Beta_1', 'val']
Beta_2 = parameters.at['Beta_2', 'val']
Beta_3 = np.mean(np.array([Beta_1, Beta_2]))
PWP = parameters.at['PWP', 'val']
L1 = parameters.at['L1', 'val']
K_uu = parameters.at['K_uu', 'val']
K_ul = parameters.at['K_ul', 'val']
K_d = parameters.at['K_d', 'val']
k_ll = parameters.at['k_ll', 'val']


def objective(x):

    Beta = x

    for i in range(strt_idx, rs.shape[0]):
        pr = rs[i - 1]
        cr = rs[i]

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
    return rs[strt_idx:, Q_sim_col], NS


print('\n')
betas = [Beta_1, Beta_2, Beta_3]
for i in betas:
    print(objective(i)[1])
print('\n')

cols_b = (q1_col, q2_col, q3_col, betas_org_col, betas_sim_col) = (0, 1, 2,
3, 4)
bs = np.zeros((beta_org.shape[0], len(cols_b)))
ini_v_b = np.array([rs[0, Q_sim_col], rs[0, Q_sim_col], rs[0, Q_sim_col],
np.nan, np.nan])

bs = np.vstack((ini_v_b, bs))

bs[strt_idx:, q1_col] = objective(Beta_1)[0]
bs[strt_idx:, q2_col] = objective(Beta_2)[0]
bs[strt_idx:, q3_col] = objective(Beta_3)[0]
bs[strt_idx:, betas_org_col] = np.array(beta_org['betas_org'])

for i in range(strt_idx, bs.shape[0]):

    pr = bs[i - 1]
    cr = bs[i]
    condR1 = (cr[q1_col] - pr[q1_col] >= rising_limit)
    condR2 = (cr[q2_col] - pr[q2_col] >= rising_limit)
    condR3 = (cr[q3_col] - pr[q3_col] >= rising_limit)
    cond_sum_R = sum((condR1, condR2, condR3))

    condr1 = (cr[q1_col] - pr[q1_col] <= recession_limit)
    condr2 = (cr[q2_col] - pr[q2_col] <= recession_limit)
    condr3 = (cr[q3_col] - pr[q3_col] <= recession_limit)
    cond_sum_r = sum((condr1, condr2, condr3))

    cond = 3

    if cond_sum_R >= cond:
        beta = Beta_1

    elif cond_sum_r >= cond :
        beta = Beta_2

    else:
        beta = Beta_3

    cr[betas_sim_col] = beta

col = ('q1', 'q2', 'q3', 'betas_org', 'betas_sim')

idx = pd.date_range(parser.parse(dates_start) + timedelta(days=-1) ,
dates_end)
bs_df = pd.DataFrame(bs, columns=col, index=idx)
# print(bs_df)

bmean = np.average(bs[off_idx:, betas_org_col])

bobs_bmean_sq = (bs[off_idx:, betas_org_col] - bmean) ** 2
bsim_bobs_sq = (bs[off_idx:, betas_sim_col] - bs[off_idx:, betas_org_col]) ** 2
NS = 1 - (np.sum(bsim_bobs_sq) / np.sum(bobs_bmean_sq))
print('NS_Betas = ', NS)

col = ('temp', 'prec', 'pet', 'snow', 'not_snow', 'am', 'sm', 'et',
'rn', 'us', 'ul', 'rn_uu', 'rn_ul', 'rn_d', 'rn_l', 'Q_obs', 'Q_sim')

idx = pd.date_range(parser.parse(dates_start) + timedelta(days=-1) ,
dates_end)
rs_df = pd.DataFrame(rs, columns=col, index=idx)
rs_df['betas_org'] = bs_df['betas_org']
rs_df['betas_sim'] = bs_df['betas_sim']
rs = rs_df.to_numpy()


def objective(x):

    intrested = x

    for i in range(strt_idx, rs.shape[0]):
        pr = rs[i - 1]
        cr = rs[i]

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
        am = pr[sm_col] + (not_snow * (1 - (pr[sm_col] / FC) ** cr[intrested]))
        cr[am_col] = am
        if(pr[sm_col] > PWP):
            et = min(am, cr[pet_col])
        else:
            et = min(am, (pr[sm_col] / FC) * (cr[pet_col]))
        cr[et_col] = et
        sm = max(0, am - et)
        cr[sm_col] = sm
    #     rn
        rn = not_snow * (pr[sm_col] / FC) ** cr[intrested]
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


org_beta = -2
sim_beta = -1
print('NS of model @ sim_betas = ', objective(sim_beta))
print('NS of model @ org_betas = ', objective(org_beta))
print('\n')

betas_org_contains = {'Beta_1':np.count_nonzero((np.array(beta_org['betas_org'])) == Beta_1),
'Beta_2':np.count_nonzero((np.array(beta_org['betas_org'])) == Beta_2),
'Beta_3':np.count_nonzero((np.array(beta_org['betas_org'])) == Beta_3)}
print('betas_org_contains = ', betas_org_contains)

betas_sim_contains = {'Beta_1':np.count_nonzero(bs[:, betas_sim_col] == Beta_1),
'Beta_2':np.count_nonzero(bs[:, betas_sim_col] == Beta_2),
'Beta_3':np.count_nonzero(bs[:, betas_sim_col] == Beta_3)}
print('betas_sim_contains = ', betas_sim_contains)

col = ('temp', 'prec', 'pet', 'snow', 'not_snow', 'am', 'sm', 'et',
'rn', 'us', 'ul', 'rn_uu', 'rn_ul', 'rn_d', 'rn_l', 'Q_obs', 'Q_sim',
'betas_org', 'betas_sim')

idx = pd.date_range(parser.parse(dates_start) + timedelta(days=-1) ,
dates_end)
hbv = pd.DataFrame(rs, columns=col, index=idx)
pd.to_pickle(hbv, 'rs_fc' + station_p)

b1 = rs_df.loc[rs_df.betas_org == Beta_1]
b2 = rs_df.loc[rs_df.betas_org == Beta_2]
b3 = rs_df.loc[rs_df.betas_org == Beta_3]

betas_at_org_Beta_1 = {'Beta_1':np.count_nonzero(np.array(b1['betas_sim']) == Beta_1),
'Beta_2':np.count_nonzero(np.array(b1['betas_sim']) == Beta_2),
'Beta_3':np.count_nonzero(np.array(b1['betas_sim']) == Beta_3)}
print('betas_at_org_Beta_1 = ', betas_at_org_Beta_1)

betas_at_org_Beta_2 = {'Beta_1':np.count_nonzero(np.array(b2['betas_sim']) == Beta_1),
'Beta_2':np.count_nonzero(np.array(b2['betas_sim']) == Beta_2),
'Beta_3':np.count_nonzero(np.array(b2['betas_sim']) == Beta_3)}
print('betas_at_org_Beta_2 = ', betas_at_org_Beta_2)

betas_at_org_Beta_3 = {'Beta_1':np.count_nonzero(np.array(b3['betas_sim']) == Beta_1),
'Beta_2':np.count_nonzero(np.array(b3['betas_sim']) == Beta_2),
'Beta_3':np.count_nonzero(np.array(b3['betas_sim']) == Beta_3)}
print('betas_at_org_Beta_3 = ', betas_at_org_Beta_3)

B1 = [np.count_nonzero(np.array(b1['betas_sim']) == Beta_1),
np.count_nonzero(np.array(b1['betas_sim']) == Beta_2),
np.count_nonzero(np.array(b1['betas_sim']) == Beta_3)]

B2 = [np.count_nonzero(np.array(b2['betas_sim']) == Beta_1),
np.count_nonzero(np.array(b2['betas_sim']) == Beta_2),
np.count_nonzero(np.array(b2['betas_sim']) == Beta_3)]

B3 = [np.count_nonzero(np.array(b3['betas_sim']) == Beta_1),
np.count_nonzero(np.array(b3['betas_sim']) == Beta_2),
np.count_nonzero(np.array(b3['betas_sim']) == Beta_3)]


def graph(x):
    original_betas = x[0]
    simulated_betas = x[1]

    labels = ['Beta_1', 'Beta_2', 'Beta_3']

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, original_betas, width, label='Original Betas')
    rects2 = ax.bar(x + width / 2, simulated_betas, width, label='Simulated Betas')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Betas count')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.show()


graph(np.array([[b1.shape[0], np.nan, np.nan], B1]))
graph(np.array([[np.nan, b2.shape[0], np.nan], B2]))
graph(np.array([[np.nan, np.nan, b3.shape[0]], B3]))
