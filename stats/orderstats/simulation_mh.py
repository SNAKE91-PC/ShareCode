'''
Created on Feb 19, 2019

@author: snake91
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.optimize as spo
import scipy.special as spe
from functools import partial

import pylab

def MH(func, cntsample): #numsample

    accepted_trials = []
    scale = 5
    
    floor = lambda x: 10e-14 if x < 10e-14 else x
    pdfproposal = lambda x, loc, scale: st.norm.cdf(x, loc = loc, scale = scale)
    qproposal = lambda loc, scale: np.random.normal(loc = loc, scale = scale)
    
    burnin = 100
    counter = 0
#     h = 0.001
    
    #for i in range(numsample):
    i = -1
    while True:
        
        i += 1   
        
        if i == 0:
            
            param_old = qproposal(loc = 0, scale = 1) 
    
            
        param_new = qproposal(loc = param_old, scale = scale)
        
        alpha = min(1, func(param_new) / floor(func(param_old)) * \
                    pdfproposal(param_old, param_new, scale) / floor(pdfproposal(param_new, param_old, scale)))
    
        
        u = np.random.uniform()        
        
        if alpha > u:
    
            if cntsample > burnin:

                accepted_trials.append(param_new)

                            
            param_old = param_new
        
            counter += 1
            
            print(counter, i, sep = ' ')
            if counter == cntsample:
                break
            
    return accepted_trials
    
    
def ecdf(series):
    
    ecdf = []
    series = np.sort(series)

    for i in range(len(series)):
        
        p = i / len(series)
        
        ecdf.append((series[i], p))
    
    return ecdf


def mse(param, x, y):
    
    a,b = param
    
#     x = list(filter(lambda x: x[1] <= q, x))
#     y = list(filter(lambda x: x[1] <= q, y))
    
    x = np.array(list(map(lambda x: x[0], x)))
    y = np.array(list(map(lambda x: x[0], y)))
    
    err = np.mean((a*x - b - y)**2)
    
    print(err)
    return err



f = lambda x: st.norm.cdf(x)**n

g = lambda x: (f(x + 0.00001) - f(x))/ 0.00001
# g = lambda x: n * st.norm.cdf(x)**(n-1) * st.norm.pdf(x)

gumbelpdf = lambda x: np.exp(-x - np.exp(-x))

gumbelquantile = lambda u: -np.log(-np.log(u))
