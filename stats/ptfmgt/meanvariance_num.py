'''
Created on 27 Dec 2019

@author: snake91
'''

import scipy.optimize as spo
from scipy.stats import invwishart
import scipy.linalg as splin
import scipy.optimize as spo
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
import os

import pathos.pools as pp

np.random.seed(10)

df = 100
ivs = invwishart(df = df, scale = np.identity(df))
rivs = ivs.rvs(size = 2)

def standardize(x):

    diag = deepcopy(np.diag(x))
    y = deepcopy(x)
    
    for i in range(len(diag)):
        y[i,:] = y[i,:] / np.sqrt(diag[i])
        y[:,i] = y[:,i] / np.sqrt(diag[i]) 
    
    return y


corrmatrix = list(map(lambda x: standardize(x),rivs))


def VarPort(params, sigma):
    
    return np.dot(np.dot(params.T, sigma), params)

def RetPort(params, mu):
    
    return np.dot(params.T, mu) 


def minVarPort(params, mu, sigma):
    # sqrt(WT * covmat * W)
    return VarPort(params, sigma) / RetPort(params, mu) 




idxVar = 0
nSim = 500
    
    
muStocks = np.random.multivariate_normal(mean = np.zeros(df), cov = rivs[idxVar])
    

def optim(args):#covmatrix, muStocks, muPort, option): 
    """
        options:
            support:
                unbounded
                no-short
            criterion:
                mean-variance
                mean-variance-kurt
    """
    
    covmatrix = args[0]
    muStocks = args[1]
    muPort = args[2]
    
    try:
        optionDict = args[3] #@unusedvariable
    except IndexError:
        pass # default option
    
    df = len(muStocks)
    x0 = np.ones(df)/df

    if optionDict['support'] == 'unbounded':
        bounds = [(None,None)] * df
    elif optionDict['support'] == 'no-short':
        bounds = [(0, 1)] * df
    else:
        raise NotImplementedError
        
    if optionDict['criterion'] == 'mean-variance':
        objfunction = VarPort
    elif optionDict['criterion'] == 'mean-variance-kurt':
        not NotImplementedError
    else:
        not NotImplementedError
        
        
    def consfunc(x, muStocks, muPort):
    
        return np.max( [np.abs(np.sum(x) - 1), np.abs(RetPort(x, muStocks) / muPort - 1)])
#         return np.min([np.log(np.abs(np.sum(x))),  np.log(np.abs(RetPort(x, muStocks)/muPort))])

    cons = {'type': 'eq', 'fun' : lambda x: consfunc(x, muStocks, muPort)}
    
    ub = spo.minimize(fun = objfunction, x0 = x0, args = covmatrix, bounds = bounds, constraints = cons)
    
    print(1-np.sum(ub.x))
    return (np.sqrt(ub.fun), RetPort(ub.x, muStocks), np.sum(ub.x))
            
 
idxVar = 0

V = rivs[idxVar]
invV = np.linalg.inv(V)
A = np.dot(np.dot(np.ones(shape = (V.shape[0])), invV), muStocks.T)
B = np.dot(np.dot(muStocks, invV), muStocks.T)
C = np.dot(np.dot(np.ones(shape = (V.shape[0])), invV), np.ones(shape = V.shape[0]).T)
D = B * C - A**2


covmatrix = rivs[idxVar]
ncore = 4
pool = pp.ProcessPool(ncore)

muPort = np.random.uniform(np.random.uniform(low =-1, high = 1, size = int(nSim)))

dataUb = zip([ covmatrix ]*len(muPort), [muStocks] * len(muPort), muPort, [{'support' : 'unbounded', 'criterion': 'mean-variance'}] * len(muPort) )
curveUb = pool.map(optim, dataUb)

dataNb = zip([ covmatrix ]*len(muPort), [muStocks] * len(muPort), muPort, [{'support' : 'no-short', 'criterion' : 'mean-variance'}] * len(muPort))
curveNb = pool.map(optim, dataNb)


curveUb_x =list(map(lambda x: x[0], curveUb))
curveNb_x =list(map(lambda x: x[0], curveNb))

curveUb_y = list(map(lambda x: x[1], curveUb))
curveNb_y = list(map(lambda x: x[1], curveNb))

curveUb_z = list(map(lambda x: x[2], curveUb))
curveNb_z = list(map(lambda x: x[2], curveNb))


eff = [np.sqrt((-1+C*dev_std**2)*(D/C**2))+A/C for dev_std in np.sort(curveUb_x)]
ineff = [A/C-np.sqrt((-1+C*dev_std**2)*(D/C**2)) for dev_std in np.sort(curveUb_x)]


plt.scatter(curveUb_x, curveUb_y, s = 30, edgecolors = 'blue', linewidths = 0.5, label = 'num optim unbounded')
plt.scatter(curveNb_x, curveNb_y, s = 30, color = 'green', edgecolors = 'black', linewidths = 0.5, label = 'num optim no-short')

plt.plot( np.sort(curveUb_x), eff, linestyle = '--', color = 'red', linewidth = 2, label = 'analytical')
plt.plot( np.sort(curveUb_x), ineff, linestyle = '--', color = 'red', linewidth = 2) #, label = 'analytical (ineff)'

plt.xlim(xmax = 0.15)
plt.legend()


plt.savefig(os.getcwd() + "/pics/" + "Numerical_Efficient_Frontier.svg")

print("")
 
 
 
 
 
 
 
 
