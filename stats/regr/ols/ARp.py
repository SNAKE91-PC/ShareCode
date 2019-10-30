'''
Created on Dec 26, 2018

@author: snake91
'''

import scipy.optimize as opt
from mle import simulate as sim
import pandas as pd
import numpy as np
from copy import deepcopy

phi = [0.5, 0.2]
y0 = [0.,0.]

x = sim.arpGaussian(t = 500, phi = phi)


def minARpols(params, x):

    x = pd.Series(x)
    
    for i in range(0, len(params)):
        x -= params[i]*x.shift(i+1)

    return np.mean(x.dropna()**2)
#     return np.mean(((x[1:] - a * x[:-1])[1:] - b * x[ :-2])**2)


a = opt.minimize(fun = minARpols, x0 = (0.,0.), bounds = ((-0.99, 0.99),(-0.99, 0.99)), args = x)

print(a)