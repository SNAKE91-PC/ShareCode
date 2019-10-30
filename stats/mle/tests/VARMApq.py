'''
Created on Dec 24, 2018

@author: snake91
'''


from mle import simulate as sim
from mle import likelihood as logL
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.varmax import VARMAX
import scipy.stats as st
import scipy.optimize as opt
import scipy.linalg as slin
from copy import deepcopy
import pandas as pd

t = 1000

import mle.constraint as cons

np.random.seed(10)

p1 = np.asmatrix([
                    [0.2, 0.3], 
                    [-0.6, -0.2]
                ])
  


q1 = np.asmatrix([
                    [-0., 0.],
                    [0., -0.],
                 ])
p = [p1]#, p2] #, p3]
q = [q1]#q1]

# y0 = np.asmatrix([[0., 0., 0.]]).T #, [0., 0., 0.]

X = sim.varmapqGaussian(t = t, pMatrix = p, qMatrix = q)#, y0 = y0)

y = VARMAX(X.T, order = (1,1)).fit()

print(y.summary())

x1 = np.asarray(X[0,:]).reshape(t)
x2 = np.asarray(X[1,:]).reshape(t)
# x3 = np.asarray(X[2,:]).reshape(t)



# nprocess = X.shape[0]
pLag = len(p)
qLag = len(q)
# 

params = logL.maxVARMApqN(X, pLag, qLag)

print(params)




