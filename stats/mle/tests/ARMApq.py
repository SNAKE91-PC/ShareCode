'''
Created on Dec 25, 2018

@author: snake91
'''

from mle import simulate as sim
from mle import likelihood as logL
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import scipy.stats as st
import scipy.optimize as opt
import scipy.linalg as slin
from copy import deepcopy
import pandas as pd


phiList = [0.5]
psiList = [-0.3]
np.random.seed(1)
y1 = sim.armapqGaussian(t = 500, phi = phiList, psi = psiList)#, y0 = [0.])

# np.random.seed(1)
# y2 = sim.arma11Gaussian(t = 500, phi = 0.5, psi = -0.5)



bounds = tuple([(-0.99, 0.99) for i in range(0, len(phiList) + len(psiList))] + [(0.001, None)]) 
x0 = tuple( [0. for i in range(len(phiList) + len(psiList))] + [ 1 ] )

# x = opt.minimize(fun = logL.maxARMApqN, x0 = x0, bounds = bounds, args = (y1,1,1), tol=10e-16)

y1 = np.asmatrix(y1)

x = opt.minimize(fun = logL.maxVARMApqN, x0 = x0, bounds = bounds, args = (y1, 1, 1), tol = 10e-16)

print(x)

plot_acf(y1, lags = 10)
plot_pacf(y1, lags = 10)

# plot_acf(y2, lags = 10)
# plot_pacf(y2, lags = 10)