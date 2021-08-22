'''
Created on Jan 12, 2019

@author: snake91
'''



from mle.mleclass import mleobj
from mle import simulate as sim

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

import mle.likelihood as logL


n = 500

gbm = np.zeros(shape = (1,n))
retgbm = np.zeros(shape = (1,n-1))
gbm[0] = 1

for i in range(1, n):
    retgbm[0,i-1] = np.exp((0 - 0.5 * (1/n)**2 ) + (1/n) * np.random.normal())
    gbm[0,i] = gbm[0,i-1] * retgbm[0,i-1]
    
    
## calculating returns

retDict = {}
retList = [1, 2, 5, 10]
for t in retList:
    
    retDict[t] = np.array([gbm[0,j] / gbm[0,j-t] for j in range(t, n)])
    

phiList = []
psiList = [0.] * 8
bounds = tuple([(-0.99, 0.99) for i in range(0, len(phiList) + len(psiList))] + [(0.001, None)]) 
x0 = tuple( [0. for i in range(len(phiList) + len(psiList))] + [ 1 ] )

x = opt.minimize(fun = logL.maxARMApqN, x0 = x0, bounds = bounds, args = (retDict[10],1,1), tol=10e-16)

print("") 


