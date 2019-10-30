'''
Created on Feb 18, 2019

@author: snake91
'''


import rpy2
from mle import simulate as sim
from mle import likelihood as logL
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.optimize as opt
import scipy.linalg as slin
from copy import deepcopy
import pandas as pd


from rpy2.robjects import r, pandas2ri, numpy2ri
from rpy2.robjects.packages import importr

from arch import arch_model

pandas2ri.activate()
numpy2ri.activate()

import mle.constraint as cons

T = 10000

a0    = np.asmatrix([0.1, 0.1]).T

a1 = np.asmatrix([
                     [0.2, 0.2],
                     [0.1, 0.3]
                    ])

b1  = np.asmatrix([
                      [0.1, 0.1],
                      [0.1,  0.1]
                    ])

corrMatrix = np.asmatrix([
                            [1, 0.5],
                            [0.5, 1]
                        ])

alpha = [a1]
beta  = [b1]


try:
    ccgarch = importr('ccgarch')
except:
    utils = importr('utils')
    utils.install_packages('ccgarch')
    ccgarch = importr('ccgarch')
    
X = sim.mgarcheccc(t = T, a0 = a0, alphaMatrix = alpha, betaMatrix = beta, corrMatrix = corrMatrix)


a = np.array(a0.T.tolist()[0])
A = np.asarray(alpha[0])
B =  np.asarray(beta[0])
R = np.asarray(corrMatrix)
 
res = ccgarch.eccc_estimation(a = a, A= A, B = B, R = R, dvar = np.asarray(X.T), model = 'extended')

print(res[-1]) #perfect

print(np.corrcoef(X))  # unconditional corr of the res different than Y unc corr

# eccc = ccgarch.eccc_sim(a, A, B, R, method ='diagonal')

    
#                 )
params = logL.maxMGARCHECCCpqN(X, alpha = 1, beta = 1)
    
    
print(params)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    