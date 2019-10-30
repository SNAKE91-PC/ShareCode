

'''
Created on Feb 9, 2018

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

T = 20000

# TODO: alpha coeff seem to be squared when estimated
a0    = np.asmatrix([0.1, 0.1]).T
alpha = np.asmatrix([0.5, 0.5]).T
beta  = np.asmatrix([0.3, 0.3]).T

corrMatrix = np.asmatrix([
                            [1, 0.5],
                            [0.5, 1]
                        ])

try:
    ccgarch = importr('ccgarch')
except:
    utils = importr('utils')
    utils.install_packages('ccgarch')
    ccgarch = importr('ccgarch')
    
X = sim.mgarchccc(t = T, a0 = a0, alpha = alpha, beta = beta, corrMatrix = corrMatrix)


a = np.array(a0.T.tolist()[0])
A = np.diag(alpha.T.tolist()[0])
B =  np.diag(beta.T.tolist()[0])
R = np.asarray(corrMatrix)
 
res = ccgarch.eccc_estimation(a = a, A= A, B = B, R = R, dvar = np.asarray(X.T), model = 'diagonal')

print(res[-1]) #looks ok

print(np.corrcoef(X)) 

# eccc = ccgarch.eccc_sim(a, A, B, R, method ='diagonal')

    
    
    