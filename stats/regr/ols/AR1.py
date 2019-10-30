'''
Created on Dec 26, 2018

@author: snake91
'''

import scipy.optimize as opt
from mle import simulate as sim
import pandas as pd
import numpy as np

x = sim.ar1Gaussian(t = 500, phi = 0.5)


def minAR1ols(params, x):
    
    a = params
    
    return np.mean((x[1:] - a * x[:-1])**2)


a = opt.minimize(fun = minAR1ols, x0 = (0.,), bounds = ((-0.99, 0.99),), args = x)

print(a)