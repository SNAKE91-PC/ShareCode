'''
Created on 9 Jan 2020

@author: snake91
'''

from scipy.stats import invwishart
import numpy as np
import matplotlib.pyplot as plt

import scipy.optimize as spo

from copy import deepcopy
import os

import pathos.pools as pp

np.random.seed(10)

df = 200
ivs = invwishart(df = df, scale = np.identity(df))
rivs = ivs.rvs(size = 2)

class dummy(object):
    pass

idxVar = 0
nSim = 100000
nPoint = 100
    
retStocks = np.random.multivariate_normal(mean = np.zeros(df), cov = rivs[idxVar], size = nSim)
muPort = np.linspace(-1, 1, nPoint)
perc = 97.5


def CVaRPort(params, retStocks, perc):
    
    retScenarios = np.sort((params.T * retStocks).sum(axis = 1))
    percScenario = np.percentile(retScenarios, perc)
    
    return percScenario + np.mean(list(filter(lambda x: x - percScenario >=0 , retScenarios)))

def VaRPort(params, retStocks, perc):
    
    retScenarios = np.sort((params.T * retStocks).sum(axis = 1))
    percScenario = np.percentile(retScenarios, perc)
    
    return percScenario# + np.mean(list(filter(lambda x: x - percScenario >=0 , retScenarios)))
#     return np.dot(np.dot(params.T, sigma), params)

def RetPort(params, mu):
    
    return np.dot(params.T, mu) 



def EmpiricalOptim(args):#covmatrix, muStocks, muPort, option): 
    """
        options:
            support:
                unbounded
                no-short
            criterion:
                mean-variance
                mean-variance-kurt
    """
    
    print("starting..", args[-1])
    retStocks = args[0]
    muPort = args[1] # target
    muStocks = np.mean(retStocks, axis = 0) # empirical mean
    perc = 97.5
#     covmatrix = args[0]
#     muStocks = args[1]
#     muPort = args[2]
    
    try:
        optionDict = args[2] #@unusedvariable
    except IndexError:
        pass # default option
    
    df = retStocks.shape[1]
    x0 = np.ones(df)/df

    if optionDict['support'] == 'unbounded':
        bounds = [(None,None)] * df
    elif optionDict['support'] == 'no-short':
        bounds = [(0, 1)] * df
    else:
        raise NotImplementedError
        
    if optionDict['criterion'] == 'mean-cvar':
        objfunction = CVaRPort
    elif optionDict["criterion"] == 'mean-var':
        objfunction = VaRPort
    elif optionDict['criterion'] == 'mean-variance-kurt':
        not NotImplementedError
    else:
        not NotImplementedError
        
        
    def consfunc(x, muStocks, muPort):
    
        res = np.max([ np.abs(np.sum(x) - 1), np.abs(RetPort(x, muStocks)/muPort - 1) ])
        
        return res

    cons = {'type': 'eq', 'fun' : lambda x: consfunc(x, muStocks, muPort)}
    
    ub = dummy()
    ub.x = 0
    retP = 0
    
    while np.max([1-np.sum(ub.x), retP - muPort]) >= 10e-8:
        
        retStocksSample = retStocks[np.random.choice(retStocks.shape[0], 500, replace=False)] #250d of 200 stocks
        
        ub = spo.minimize(fun = objfunction, x0 = x0, args = (retStocksSample, perc), bounds = bounds, constraints = cons, method = 'trust-constr')

        retP = RetPort(ub.x, muStocks)
    
    print(1-np.sum(ub.x), retP - muPort)
    
    return (ub.fun, retP, np.sum(ub.x))


ncore = 6
pool = pp.ProcessPool(ncore)

dataUb = zip([retStocks]*nPoint, muPort, [{'support' : 'unbounded', 'criterion' : 'mean-var'}] * nPoint, list(np.arange(nPoint/ncore)) * ncore) 

x = pool.map(EmpiricalOptim, dataUb)

pool.close()

xvar = list(map(lambda x: x[0], x))
yret = list(map(lambda x: x[1], x))

plt.scatter(xvar, yret)

print("")
    
    
    
    
