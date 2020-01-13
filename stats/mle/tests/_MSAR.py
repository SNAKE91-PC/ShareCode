'''
Created on 8 Nov 2019

@author: snake91

python translation of the EM algorithm (R) here
https://thesamuelsoncondition.com/2016/02/20/time-series-v-the-hamilton-filter/
'''


import pandas as pd
import numpy as np
import scipy.optimize as spo
from mle.simulate import msiidN, sim_markovchain
from mle.likelihood import est_markovchain

import matplotlib.pyplot as plt

from copy import deepcopy

N = lambda x, mu, sigma : (1./np.sqrt(2*np.pi*sigma**2)) * np.exp(-(x-mu)**2 / (2*sigma**2))


def mrs_est(theta, x, y, option = 'MLE'):
     
    alpha1, alpha2, beta1, beta2, sigma1, sigma2, p11, p22 = theta
#     p11 = 1/(1+np.exp(-p11))
#     p22 = 1/(1+np.exp(-p22))

    #in order to make inference about what state we are in in period t we need the conditional
    # densities given the information set through t-1
    
    f1 = N(y, alpha1 + beta1 * x, sigma1)
    f2 = N(y, alpha2 + beta2 * x, sigma2)

    f = np.asarray([f1,f2]).T
    
        #S.forecast is the state value looking forward conditional on info up to time t
    #S.inf is the updated state value
      
    S_forecast = np.zeros(shape = (len(y), 2))
    S_inf = deepcopy(S_forecast)
    ov = np.ones(2)
    
    P = np.asarray([[p11, 1-p11], [1-p22, p22]])
    model_lik = np.zeros(len(y))
    
    S_inf[0, ] = (np.diag(P) * f[0,]) / np.dot(ov, np.diag(P) * f[0,].T)
    
    
    for i in range(1, len(y)):

        #in time t we first make our forecast of the state in t based on the
        # data up to time t-1, then we update that forecast based on the data
        # available in t
        
        # expectation
        #
        S_forecast[i,:] = np.dot(P, S_inf[i-1: i, :].T).T
        
        # maximization
        # bayes theorem
        # posterior = likelihood x prior / evidence
        S_inf[i, : ] = (S_forecast[i,: ] * f[i,:]) / np.dot(S_forecast[i, :], f[i, :].T)
            
        model_lik[i] = max(1e-24, np.dot(ov, S_forecast[i] * f[i]))
     
    if option == 'MLE':
        logl = np.sum(np.log(model_lik[1:]))
         
        print(theta, -logl, sep = '')
        
        return -logl
    elif option == 'SE':
        
        return (S_inf, S_forecast)




def ham_smooth(theta, x, y):
    
    alpha1, alpha2, beta1, beta2, sigma1, sigma2, p11, p22 = theta
    S_inf, S_forecast = mrs_est(theta, x, y, option = 'SE')
    
    T = len(y)
    P_smooth = np.zeros(2 * len(y)).reshape((len(y), 2)) 
    P_smooth[T-1: ] = deepcopy(S_inf[T-1: ])
     
    for i in reversed(range(1, T-1)):
         
        #1. probability that we observe S(t)=1, S(t+1)=1
        #2. probability that we observe S(t)=1, S(t+1)=2
        #3. probability that we observe S(t)=2, S(t+1)=1
        #4. probability that we observe S(t)=2, S(t+1)=2
         
        #for #1 P[S(t)=1,S(t+1)=1|I(t+1)] = {P[S(t+1)=1|I(t+1)]*P[S(t)=1|I(t)]*P[S(t+1)=1|S(t)=1]}/P[S(t+1)=1|I(t)]
        p1 = (S_inf[i, 0] * S_inf[i-1, 0] * p11)/ S_forecast[i,0]
  
        #for #2 we have
        p2 = (S_inf[i, 1] * S_inf[i-1, 0] * (1-p11)) / S_forecast[i, 1]
         
        #for #3 we have
        p3 = (S_inf[i, 0] * S_inf[i-1, 1]) * (1-p22) / S_forecast[i, 0]
         
        #for #4 we have
        p4 = (S_inf[i, 1] * S_inf[i-1, 1] * p22) / S_forecast[i, 1]
         
        P_smooth[i, 0] = p1 + p2
        P_smooth[i, 1] = p3 + p4
         
     
    return P_smooth
    
    
    

#### SIMULATE PROCESSES
np.random.seed(10)

transmat = {0: [0.5, 0.5], 1: [0.5, 0.5]} #np.matrix([[0.9, 0.1], [0.2, 0.8]])

x = np.linspace(0, 10, 100)


transitions = sim_markovchain(t = len(x), pmatrix = transmat, startvalue = 0)
est_prob = est_markovchain(transitions)

# states = {0 : x, 1: x}
beta = [1, 4]
intercept = [-5, 10 ]

y1 = [intercept[transitions[i]] + beta[transitions[i]] * x[i] + np.random.normal() for i in range(len(transitions))]#states[0][i] + np.random.normal() for i in range(len(transitions))]

y = y1 #np.hstack([y1, y2])



theta_new = [-5, 10, 1, 4, 1, 1, 0.5, 0.5]
mrs_est(theta_new, x, y) #18941.68118

theta = []
bounds = []

# cons = {'type': 'eq',  'fun' :lambda x: cons(x)}

theta = [0, 0, 0, 0, 1, 1, 0., 1.]
bounds_de = [(-10,10), (-10,10), (-5, 5), (-5, 5), (1e-8, 2), (1e-8, 2), (0,1), (0,1)]  
bounds_min = [(None,None), (None,None), (None, None), (None, None), (1e-8, None), (1e-8, None), (0,1), (0,1)]

res1 = spo.differential_evolution(func = mrs_est, bounds = bounds_de, args = (x,y), atol = 10e-16)
print(res1)

res2 = spo.minimize(mrs_est, x0 = theta, args = (x,y), bounds = bounds_min, tol = 10e-16)
print(res2)


psmooth = ham_smooth(theta, x, y)
# print(res)

print("")

# res1 = ham_smooth(res.x, x, y)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    