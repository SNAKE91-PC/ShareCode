'''
Created on 7 Dec 2019

@author: snake91
'''

import pandas as pd
import numpy as np
import scipy.optimize as spo
from mle.simulate import msiidN, sim_markovchain
from mle.likelihood import est_markovchain

import matplotlib.pyplot as plt

from copy import deepcopy

N = lambda x, mu, sigma : (1./np.sqrt(2*np.pi*sigma**2)) * np.exp(-(x-mu)**2 / (2*sigma**2))


flatten = lambda l: [item for sublist in l for item in sublist]



def mrs_est(theta, x, y, df, option = 'MLE'):
     
    alpha = theta[0:df]
    beta = theta[df:2 * df]
    sigma = theta[2 * df : 3 * df]
    
    probmat = theta[3*df:]
    probmat = np.asmatrix(probmat).reshape((df, df-1))
    probmat = np.asmatrix(np.hstack([probmat, 1 - probmat.sum(axis = 1)]))

    if np.min(1-probmat.sum(axis = 1)) > 0: #>0
        return np.inf
#     p11 = 1/(1+np.exp(-p11))
#     p22 = 1/(1+np.exp(-p22))

    #in order to make inference about what state we are in in period t we need the conditional
    # densities given the information set through t-1

    f = []
    for i in range(len(alpha)):
            
        f.append( N(y, alpha[i] + beta[i] * x, sigma[i]) )
#         f1 = N(y, alpha1 + beta1 * x, sigma1)
#         f2 = N(y, alpha2 + beta2 * x, sigma2)
#         f3 = N(y, alpha3 + beta3 * x, sigma3)

#     f = np.asarray([f1,f2,f3]).T
    f = np.asarray(f).T
    
    #S.forecast is the state value looking forward conditional on info up to time t
    #S.inf is the updated state value
      
    S_forecast = np.zeros(shape = (len(y), df))
    S_inf = deepcopy(S_forecast)
    ov = np.ones(df)
    
#     P = np.asarray([ [p11, 1-p11-p13, p13], [1-p22-p23, p22, p23], [1-p33-p32, p32, p33] ])
    P = probmat

    model_lik = np.zeros(len(y))
    
    S_inf[0, ] = (np.diag(P) * f[0,]) / np.dot(ov, np.diag(P) * f[0,].T)
    
    
    for i in range(1, len(y)):

        #in time t we first make our forecast of the state in t+1 based on the
        # data up to time t, then we update that forecast based on the data
        # available in t+1
        
        S_forecast[i,:] = np.dot(P, S_inf[i-1: i, :].T).T
        S_inf[i, : ] = (S_forecast[i,: ] * f[i,:]) / np.dot(S_forecast[i, :], f[i, :])
            
        model_lik[i] = max(1e-20, np.dot(ov, S_forecast[i] * f[i]))
     
    if option == 'MLE':
        logl = np.sum(np.log(model_lik[1:]))
         
        print(theta, -logl, sep = '')
        
        return -logl
    elif option == 'SE':
        
        return (S_inf, S_forecast)



    

#### SIMULATE PROCESSES
np.random.seed(10)

transmat = {0: [0.5, 0.3, 0.2], 1: [0.5, 0.2, 0.3], 2: [0.5, 0.1, 0.4]} #np.matrix([[0.9, 0.1], [0.2, 0.8]])

x = np.linspace(0, 100, 10000)


transitions = sim_markovchain(t = len(x), pmatrix = transmat, startvalue = 0)
est_prob = est_markovchain(transitions)

beta = [1, 4, 7]
intercept = [-5, 10, 5 ]

y = [intercept[transitions[i]] + beta[transitions[i]] * x[i] + np.random.normal() for i in range(len(transitions))]#states[0][i] + np.random.normal() for i in range(len(transitions))]


def consfunc(x, df):
    
    alpha = x[0:df] # @unusedvariable
    beta = x[df:2 * df] # @unusedvariable
    sigma = x[2 * df : 3 * df] # @unusedvariable
    
    probmat = x[3*df:]
    probmat = np.asmatrix(probmat).reshape((df, df-1))
    
    return np.min(1-probmat.sum(axis = 1)) #>0


df = len(beta) #n of states
probmat = np.matrix(np.ones(df**2)/df).reshape((df,df))


theta = [[0.] * df + [0.] * df + [1.] * df + flatten(probmat[:-1].tolist())  ] #0, 0, 0, 0, 0, 0, 1, 1, 1, 1e-8, 1-2e-8, 1e-8, 1-2e-8, 1e-8, 1-2e-8 ]
theta = flatten(theta)

# alpha, beta, sigma, probs
bounds_de = [(-10,15)] * df + [(0, 10)] * df + [(1e-8, 2)] * df + [(0,1)] * df * (df - 1)  
bounds_min =  [(None,None)] * df + [(None, None)] * df + [(1e-8, None)] * df + [(0,1)] * df * (df - 1)

cons = {'type': 'eq',  'fun' :lambda x: consfunc(x, df)}

# res1 = spo.minimize(mrs_est, x0 = theta, args = (x,y, df), bounds = bounds_min, constraints = cons, tol = 10e-16, method = 'trust-constr')
# print(res1)

res2 = spo.differential_evolution(func = mrs_est, bounds = bounds_de, args = (x,y,df), atol = 10e-16)

alpha = res2.x[0:df] # @unusedvariable
beta = res2.x[df:2 * df] # @unusedvariable
sigma = res2.x[2 * df : 3 * df] # @unusedvariable

probmat = res2.x[3*df:]
probmat = np.asmatrix(probmat).reshape((df, df-1))
probmat = np.asmatrix(np.hstack([probmat, 1 - probmat.sum(axis = 1)]))

print(res2)

theta = [ -5,10,5 ] + [ 1,4,7 ] + [ 1,1,1 ] + [ 0.5,0.3, 0.5,0.2, 0.5,0.1 ]

mrs_est(theta, x, y, df)

print("")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    