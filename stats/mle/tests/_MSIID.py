'''
Created on Feb 27, 2019
 
@author: snake91
'''
 


import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo

from mle.likelihood import maxMSIID
from mle.simulate import msiidN

np.random.seed(10)

transmat = {0: [0.9, 0.1], 1: [0.2, 0.8]} #np.matrix([[0.9, 0.1], [0.2, 0.8]])
x = msiidN(t = 1000, transmat = transmat, startstate = 1, paramsmean = [-6, 5], paramsvar = [0.2, 20])


bounds = [(None, None), (1e-8, None), (None, None), (1e-8, None)] + [(0,1), (0,1)] #,(0,1),(0,1)]
bounds_de = [(-10, 10), (1e-8, 50), (10, 10), (1e-8, 50)] + [(0,1), (0,1)]

def probcons(x):
    
    dim = np.int(np.sqrt(len(x)))
    transmat = np.asmatrix(x)
#     transmat = np.reshape(transmat, newshape = (dim, dim))
    
    flatten = lambda l: [item for sublist in l for item in sublist]
    
    
    pList = flatten(transmat.sum(axis = 0).tolist()) #+ flatten(transmat.sum(axis = 1).tolist()) 

#     print(np.max(pList) - np.min(pList)) # 1-1)
    
    return np.max(pList) - np.min(pList) # 1-1 



cons = {'type': 'eq', 'fun': lambda y: probcons(y[4:])}


prob0 = 0.5
params = spo.minimize(fun = maxMSIID, x0 = (0, 1, 0, 1, 0.5, 0.5), args = (x, prob0), bounds = bounds, constraints = cons, method = 'trust-constr')
# params = spo.differential_evolution(func = maxMSIID, args = (x, prob0), bounds = bounds_de)
# params = spo.brute(func = maxMSIID, ranges = bounds_de, args = (x, prob0)) 

print("")
