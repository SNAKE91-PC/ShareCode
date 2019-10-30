'''
Created on Sep 2, 2019

@author: snake91
'''


import scipy.optimize as opt
from mle import simulate as sim
import pandas as pd
import numpy as np

phi = [0.5, 0.2]
y0 = [0.,0.]

x = sim.arpGaussian(t = 50000, phi = phi)


def minARpols(params, x):

    x = pd.Series(x)
    
    for i in range(0, len(params)):
        x -= params[i]*x.shift(i+1)

    return np.mean(x.dropna()**2)


def minARpquantile(params, x, q):

    x = pd.Series(x)
    
    for i in range(0, len(params)):
        x -= params[i]*x.shift(i+1) 

    x = x.dropna()
    y = np.where(x > 0,  q * np.abs(x), (1-q) * np.abs(x))
    
    return np.sum(y)


def minARplad(params, x):
    
    ### only median
    x = pd.Series(x)
    
    for i in range(0, len(params)):
        x -= params[i]*x.shift(i+1) 

    x = x.dropna()
    y = np.abs(x)
    
    return np.sum(y)


a = opt.minimize(fun = minARpols, x0 = (0.,0.), bounds = ((-0.99, 0.99),(-0.99, 0.99)), args = x)

q = 0.5
b = opt.minimize(fun = minARpquantile, x0 = (0.,0.), bounds = ((-0.99, 0.99),(-0.99, 0.99)), args = (x, q))
c = opt.minimize(fun = minARplad, x0 = (0.,0.), bounds = ((-0.99, 0.99),(-0.99, 0.99)), args = (x))

print(a)
print(b)
print(c)

print("")

