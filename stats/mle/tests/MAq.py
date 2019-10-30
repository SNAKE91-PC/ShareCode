'''
Created on Dec 8, 2018

@author: snake91
'''

from mle import simulate as sim
from mle import likelihood as logL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

import mle.constraint as cons

# def constraint(y):
#     
#     I = np.diag(np.ones(shape = (1 * len(y))))[0: len(y)-1]
#     F = np.vstack((y, I))
#     
#     eigenvalues, eigenvec = np.linalg.eig(F) # @UnusedVariable
#     
#     f = lambda x: (np.real(x)**2 + np.imag(x)**2 ) > 0
#     
#     if all(f(eigenvalues)):
#         return 1
#     else:
#         return -1
#     


x = sim.maqGaussian(t = 500, psi = [0.8])#, 0.6])
params = opt.minimize(logL.maxMAqN, x0 = (0.,), args = x, bounds = ((-0.9, 0.9),), \
                      constraints= ({'type': 'ineq', 'fun': lambda y: cons.consMAq(y) }))
# 
print(params)

# params = opt.minimize(logL.maxMA1N, x0 = (0.,), args = x, bounds = ((-0.9, 0.9),), \
#                       constraints= ({'type': 'ineq', 'fun': lambda y: 1-np.sum(np.abs(y)) }))

print(params)
plot_acf(x, lags = 10)
plot_pacf(x, lags = 10)