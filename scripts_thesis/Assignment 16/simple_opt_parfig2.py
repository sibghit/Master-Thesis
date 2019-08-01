'''
Created on 15.07.2019

@author: ullah
'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

uni_path = ('C:\\Users\\ullah\\OneDrive\\')
home_path = ('O:\\')

path = uni_path

dir = os.path.abspath(os.path.join(path, r'UOS Germany/Thesis/sibghat_neckar_data/'))
dir1 = os.path.abspath(os.path.join(path,
r'scripts_cloud/Assignment 9/old_model/Optimization_without_state'))
out_dir = os.path.abspath(os.path.join(path,
r'UOS Germany/Thesis/Assignment 9/old_model/Optimization_without_state/'))

parameters = np.genfromtxt(dir + '\\parameters\\Parameters.csv',
dtype=float, delimiter=',')
lb = np.array([parameters[1, 2], parameters[2, 2], parameters[4, 2],
parameters[5, 2], parameters[6, 2], parameters[7, 2], parameters[8, 2],
parameters[9, 2], parameters[10, 2], parameters[11, 2]])
ub = np.array([parameters[1, 3], parameters[2, 3], parameters[4, 3],
parameters[5, 3], parameters[6, 3], parameters[7, 3], parameters[8, 3],
parameters[9, 3], parameters[10, 3], parameters[11, 3]])

op_420 = np.genfromtxt(dir1 + '\opt_par' + '_420.csv', dtype=float, delimiter=',', defaultfmt="2f%2")
op_3421 = np.genfromtxt(dir1 + '\opt_par' + '_3421.csv', dtype=float, delimiter=',')
op_3465 = np.genfromtxt(dir1 + '\opt_par' + '_3465.csv', dtype=float, delimiter=',')
op_3470 = np.genfromtxt(dir1 + '\opt_par' + '_3470.csv', dtype=float, delimiter=',')
cols = (opt_420_col, opt_3421_col, opt_3465_col, opt_3470_col, lb_col, ub_col,
scale_420_col, scale_3421_col, scale_3465_col, scale_3470_col) = (0, 1, 2, 3, 4,
5, 6, 7, 8, 9)

x = (np.empty((op_420[2:, 1].shape[0], len(cols)))) * np.nan

x[:, opt_420_col] = op_420[2:, 1]
x[:, opt_3421_col] = op_3421[2:, 1]
x[:, opt_3465_col] = op_3465[2:, 1]
x[:, opt_3470_col] = op_3470[2:, 1]
x[:, lb_col] = lb
x[:, ub_col] = ub

for i in range(x.shape[0]):
    scale_420 = (x[i, opt_420_col] - x[i, lb_col]) / (x[i, ub_col] - x[i, lb_col])
    x[i, scale_420_col] = scale_420
    scale_3421 = (x[i, opt_3421_col] - x[i, lb_col]) / (x[i, ub_col] - x[i, lb_col])
    x[i, scale_3421_col] = scale_3421
    scale_3465 = (x[i, opt_3465_col] - x[i, lb_col]) / (x[i, ub_col] - x[i, lb_col])
    x[i, scale_3465_col] = scale_3465
    scale_3470 = (x[i, opt_3470_col] - x[i, lb_col]) / (x[i, ub_col] - x[i, lb_col])
    x[i, scale_3470_col] = scale_3470

# my_xticks = ['tt', 'cmelt', 'FC', 'Beta', 'PWP', 'L1', 'K_uu', 'K_ul', 'K_d', 'k_ll']
# x_axis = np.arange(len(my_xticks))
# fig = plt.figure(figsize=(15, 5))
# plt.xticks(x_axis, my_xticks)
# plt.plot(x_axis, x[:, scale_420_col], lw=1.5, alpha=1, color='green', label='Kirchentellinsfurt/Neckar')
# plt.plot(x_axis, x[:, scale_3421_col], lw=1.5, alpha=1, color='red', label='Vaihingen/Enz')
# plt.plot(x_axis, x[:, scale_3465_col], lw=1.5, alpha=1, color='blue', label='Stein/Kocher')
# plt.plot(x_axis, x[:, scale_3470_col], lw=1.5, alpha=1, color='brown', label='Untergriesheim/Jagst')
#
# # plt.title ('Optimized Parameters')
# plt.grid()
# plt.ylabel('Values [scaled]', fontsize=12)
# plt.xlabel('Optimized Parameters [Names]', fontsize=12)
# plt.legend()
# plt.show()

my_xticks = ['tt', 'cmelt', 'FC', 'Beta', 'PWP', 'L1', 'K_uu', 'K_ul', 'K_d', 'k_ll']
x_axis = np.arange(len(my_xticks))
n = op_420[2:, 1]
# np.set_printoptions(precision=3)
# np.round(n, 1)

# n = ['%+7.4f, %+7.4f, %+7.4f, %+7.4f, %+7.4f, %+7.4f,%+7.4f, %+7.4f, %+7.4f, %+7.4f' % (
#         op_420[2, 1], op_420[3, 1], op_420[4, 1],
#         op_420[5, 1], op_420[6, 1], op_420[7, 1],
#         op_420[8, 1], op_420[9, 1], op_420[10, 1],
#         op_420[11, 1])]
# n = ['%+7.4f' % (op_420[2, 1]), ' %+7.4f' % (op_420[3, 1])]
# , %+7.4f, %+7.4f, %+7.4f, %+7.4f,%+7.4f, %+7.4f, %+7.4f, %+7.4f' % (
#         , , op_420[4, 1],
#         op_420[5, 1], op_420[6, 1], op_420[7, 1],
#         op_420[8, 1], op_420[9, 1], op_420[10, 1],
#         op_420[11, 1])]

# n = np.asarray(n)
# print(n)
# fig = plt.figure(figsize=(15, 5))

# plt.plot(x_axis, x[:, scale_420_col], lw=1.5, alpha=1, color='green', label='Kirchentellinsfurt/Neckar')
# plt.plot(x_axis, x[:, scale_3421_col], lw=1.5, alpha=1, color='red', label='Vaihingen/Enz')
# plt.plot(x_axis, x[:, scale_3465_col], lw=1.5, alpha=1, color='blue', label='Stein/Kocher')
# plt.plot(x_axis, x[:, scale_3470_col], lw=1.5, alpha=1, color='brown', label='Untergriesheim/Jagst')

# plt.title ('Optimized Parameters')
# plt.grid()
# plt.ylabel('Values [scaled]', fontsize=12)
# plt.xlabel('Optimized Parameters [Names]', fontsize=12)
# plt.legend()
# plt.show()

# plt.plot(x_axis, x[:, scale_420_col])

fig, ax = plt.subplots(figsize=(5, 3), dpi=96)
plt.xticks(x_axis, my_xticks)
ax.plot(x_axis, x[:, scale_420_col])
# plt.title('NS_vs_index_ACI')
plt.xlabel('param')
plt.ylabel('scaled_value')
for i, txt in enumerate(n):
    ax.annotate(txt, (x_axis[i], x[:, scale_420_col][i]))
plt.show()
# fig.savefig('NS_vs_index_ACI_420.jpg')

