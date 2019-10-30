'''
Created on Jan 6, 2019

@author: snake91
'''


import numpy as np
import scipy.linalg as slin
import scipy.optimize as opt
import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt

from mle import simulate as sim
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf


x = sim.arpGaussian(t = 500, phi = [0.5, 0.2])
acfList = acf(x, nlags = 20)


def empirical_acf(params, n):
    
    rhoList = [1.]
    
    p = len(params)
    
    for i in range(0, n):
        rho = 0.
        for p in range(len(params)):
            
            rho += params[p] * rhoList[i-p]
    
        rhoList.append(rho)
        
    return np.array(rhoList)
    
params = opt.minimize(lambda x, n: np.sum((empirical_acf(x, n)-acfList)**2), x0=(0.5, 0.2), args = 20)

print(params)

a = params

plt.plot(empirical_acf(range(0, len(acfList)), a), label = 'fitted')
plt.plot(acfList, label = 'true')
plt.legend()

y = np.random.normal(size = 500)


z = slin.toeplitz(acf(x, nlags = 500))
zinv = slin.cholesky(z)

v = np.asmatrix(y) * np.asmatrix(zinv)

v = np.asarray(v[0])
# plot_pacf(v, lags = 10)
# plot_acf(v, lags = 10)
# 
# plot_pacf(x, lags = 10)



