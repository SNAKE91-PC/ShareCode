'''
Created on Feb 26, 2019

@author: snake91
'''


import numpy as np
import scipy.optimize as spo
import scipy.stats as st
import matplotlib.pyplot as plt
from functools import partial

from joblib import Parallel, delayed
import multiprocessing

dof = 4
n_sample = 200
simulation = 100000

# flatten = lambda l: [item for sublist in l for item in sublist]
def Fstat(dof, n_sample, func):
    
    sample = [func(size = n_sample) for i in range(dof)] #@unusedvariable
    m_sample = [np.mean(sample[i]) for i in range(len(sample))] 
    s_sample = [np.var(sample[i]) for i in range(len(sample))] # var within groups
    M = np.mean(sample)
#     S = np.var(sample)
    
    mse = np.mean(s_sample) # mean var within groups
    msb = np.sum([(m_sample[i] - M)**2 for i in range(len(m_sample))])/(len(m_sample)-1) * len(sample[0]) # mean var between groups
    
    F = msb/mse
    
#     print(F)

    return F


normal = partial(np.random.normal, loc = 0, scale = 1)
FdistrNormal = [Fstat(dof, n_sample, normal) for i in range(simulation)] #@unusedvariable

tstud = partial(np.random.standard_t, df = 4)
FdistrT = [Fstat(dof, n_sample, tstud) for i in range(simulation)]

plt.hist(FdistrNormal, bins = 2000, histtype='step')
plt.hist(FdistrT, bins = 2000, histtype='step')





