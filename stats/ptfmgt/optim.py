'''
Created on 2 Jan 2020

@author: snake91
'''

import numpy as np
import scipy.optimize as spo

def VarPort(params, sigma):
    
    return np.dot(np.dot(params.T, sigma), params)

def RetPort(params, mu):
    
    return np.dot(params.T, mu) 


def minVarPort(params, mu, sigma):
    # sqrt(WT * covmat * W)
    return VarPort(params, sigma) / RetPort(params, mu) 


def ParametricOptim(args):#covmatrix, muStocks, muPort, option): 
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



def MeanVaREmpiricalOptim(args):
    
    
    
    
    
    return
            