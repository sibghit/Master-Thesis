'''
Created on 12.06.2019

@author: ullah
'''
'''
import numpy as np
from scipy.optimize import minimize


def objective(x):
    (a, b, c, d) = (x[0], x[1], x[2], x[3])
    return a * d * (a + b + c) + c


def constraint1(x):
    (a, b, c, d) = (x[0], x[1], x[2], x[3])
    return a * b * c * d - 25.0


def constraint2(x):
    (a, b, c, d) = (x[0], x[1], x[2], x[3])
    return a ** 2 + b ** 2 + c ** 2 + d ** 2 - 40


# initial guesses
n = 4
y0 = np.empty(n)
y0[0] = 5
y0[1] = 5
y0[2] = 5
y0[3] = 5

# show initial objective
print('Initial Objective: ' + str(objective(y0)))

# optimize
bnd = (1, 5)
bounds = (bnd, bnd, bnd, bnd)
con1 = {'type': 'ineq', 'fun': constraint1}
con2 = {'type': 'eq', 'fun': constraint2}
cons = ([con1, con2])
solution = minimize(objective, y0, method='SLSQP', bounds=bounds, constraints=cons)
x = solution.x

# show final objective
print('Final Objective: ' + str(objective(x)))

# print solution
print('Solution')
print('a = ' + str(x[0]))
print('b = ' + str(x[1]))
print('c = ' + str(x[2]))
print('d = ' + str(x[3]))
'''
'''
from scipy.optimize import differential_evolution
import numpy as np


def objective(x):
    (a, b) = (x[0], x[1])
    arg1 = -0.2 * np.sqrt(0.5 * (a ** 2 + b ** 2))
    arg2 = 0.5 * (np.cos(2. * np.pi * a) + np.cos(2. * np.pi * b))
    return -20. * np.exp(arg1) - np.exp(arg2) + 20. + np.e


# initial guesses
n = 2
y0 = np.empty(n)
y0[0] = 3
y0[1] = -3

# show initial objective
print('Initial Objective: ' + str(objective(y0)))

# optimize
bounds = [(-5, 5), (-5, 5)]
solution = differential_evolution(objective, bounds)

x = solution[0].x

# show final objective
print('Final Objective: ' + str(objective(x)))

# print solution
print('Solution')
print('a = ' + str(x[0]))
print('b = ' + str(x[1]))

'''
import numpy as np
from scipy.optimize import minimize


def objective(x):
    (a, b) = (x[0], x[1])
    arg1 = -0.2 * np.sqrt(0.5 * (a ** 2 + b ** 2))
    arg2 = 0.5 * (np.cos(2. * np.pi * a) + np.cos(2. * np.pi * b))
    return -20. * np.exp(arg1) - np.exp(arg2) + 20. + np.e


# initial guesses
n = 2
x0 = np.empty(n)
x0[0] = 3
x0[1] = -3

# show initial objective
print('Initial Objective: ' + str(objective(x0)))

# optimize
bounds = [(-5, 5), (-5, 5)]
solution = minimize(objective, x0, method='SLSQP', bounds=bounds)
x = solution.x

# show final objective
print('Final Objective: ' + str(objective(x)))

# print solution
print('Solution')
print('a = ' + str(x[0]))
print('b = ' + str(x[1]))
